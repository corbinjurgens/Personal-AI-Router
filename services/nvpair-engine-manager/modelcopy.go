// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// modelcopy.go is the destination half of peer-to-peer model copy:
// engine:remote-copy-model {node, engine, model} fetches a model a pinned peer
// already holds straight into this node's engine store, over the peer's ec
// routes (modelfiles.go), instead of downloading it from the internet again.
//
// Each file is downloaded beside its final name as "<final>.part", resumed with
// an HTTP Range request after an interruption (partials survive retries and
// restarts), verified against the source's sha256, and only then renamed into
// place. Files already present with a matching hash are skipped. The file that
// makes the model visible to the engine is written last (modellayout.go), so an
// interrupted copy never leaves a half model the engine would try to load.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// copyProgress is the engine:remote-progress payload for a model copy: the
// shared remote progress fields plus the file being worked on and byte counts
// across the whole model.
type copyProgress struct {
	remoteProgress
	File       string `json:"file,omitempty"`
	BytesDone  int64  `json:"bytesDone,omitempty"`
	BytesTotal int64  `json:"bytesTotal,omitempty"`
}

// copyModelResult is the engine:remote-copy-model result (under "result").
type copyModelResult struct {
	Engine       string `json:"engine"`
	Model        string `json:"model"`
	Files        int    `json:"files"`
	FilesSkipped int    `json:"filesSkipped"`
	BytesTotal   int64  `json:"bytesTotal"`
	BytesCopied  int64  `json:"bytesCopied"`
}

// modelFileSource is where a copy reads from: a peer's ec routes in production.
type modelFileSource interface {
	// modelFiles lists the model's files with sizes and sha256, relaying the
	// source's hashing progress frames.
	modelFiles(ctx context.Context, engine, model string, onProgress func(streamFrame)) (modelFileList, error)
	// modelFile opens one listed file from offset. start is the offset the body
	// actually begins at: offset for a ranged reply, 0 for a full one.
	modelFile(ctx context.Context, engine, model, path string, offset int64) (body io.ReadCloser, start int64, err error)
}

var (
	// diskFreeBytes reports the space available to this user on the volume
	// holding path. A variable so tests can stand in for the filesystem.
	diskFreeBytes = diskFree
	// copyFreeSpaceHeadroom is kept free beyond the bytes a copy still needs.
	copyFreeSpaceHeadroom int64 = 256 << 20
	// copyFileAttempts bounds how many times one file's download is tried;
	// each retry resumes where the last stopped.
	copyFileAttempts = 5
	// copyRetryBackoff is multiplied by the attempt number between tries.
	copyRetryBackoff = 2 * time.Second
	// copyStallTimeout abandons (and then resumes) a download that receives
	// no bytes for this long.
	copyStallTimeout = 60 * time.Second
	// copyProgressInterval spaces downloading progress frames.
	copyProgressInterval = 500 * time.Millisecond
)

// errRangeNotSatisfiable is a 416 reply: the partial is longer than the source
// file, so the download starts over.
var errRangeNotSatisfiable = errors.New("range not satisfiable")

// permanentCopyError is a failure a retry cannot fix.
type permanentCopyError struct{ error }

func (e permanentCopyError) Unwrap() error { return e.error }

// activeModelCopies refuses a second concurrent copy of the same model, and
// copyDestLocks serializes writers of one destination file (two models can
// share an Ollama blob).
var (
	activeModelCopies sync.Map
	copyDestLocks     = &keyedLocks{m: map[string]*keyedLock{}}
)

type keyedLocks struct {
	mu sync.Mutex
	m  map[string]*keyedLock
}

type keyedLock struct {
	mu   sync.Mutex
	refs int
}

func (k *keyedLocks) lock(key string) func() {
	k.mu.Lock()
	l := k.m[key]
	if l == nil {
		l = &keyedLock{}
		k.m[key] = l
	}
	l.refs++
	k.mu.Unlock()
	l.mu.Lock()
	return func() {
		l.mu.Unlock()
		k.mu.Lock()
		l.refs--
		if l.refs == 0 {
			delete(k.m, key)
		}
		k.mu.Unlock()
	}
}

// runRemoteCopy answers engine:remote-copy-model. Like the other engine:remote-*
// methods it runs on its own goroutine and settles when the copy finishes, with
// live engine:remote-progress frames meanwhile (op "copy", node = the source).
func (m *Manager) runRemoteCopy(ctx context.Context, msg *Message) {
	var p remoteParam
	if !m.parse(msg, &p) {
		return
	}
	switch {
	case p.Node == "":
		m.codec.RespondError(msg.ID, -32602, "node is required")
		return
	case p.Engine == "":
		m.codec.RespondError(msg.ID, -32602, "engine is required")
		return
	case p.Model == "":
		m.codec.RespondError(msg.ID, -32602, "model is required")
		return
	}
	peer, ok := m.peers.lookup(p.Node)
	if !ok {
		m.codec.RespondError(msg.ID, -32000, "node "+p.Node+" is not a discovered ec peer")
		return
	}
	client, err := m.remoteClient(ctx, peer)
	if err != nil {
		m.codec.RespondError(msg.ID, -32000, err.Error())
		return
	}
	opID := newOpID()
	emit := func(cp copyProgress) {
		cp.OpID, cp.Node, cp.Engine, cp.Op = opID, peer.nodeID, p.Engine, "copy"
		_ = m.codec.Notify("engine:remote-progress", cp)
	}
	res, err := m.exec.CopyModelFrom(ctx, client, p.Engine, p.Model, emit)
	if err != nil {
		// Terminal frame for a UI whose synchronous call already gave up.
		emit(copyProgress{remoteProgress: remoteProgress{Stage: "error", Percent: -1, Message: err.Error()}})
		m.codec.RespondError(msg.ID, -32000, err.Error())
		return
	}
	m.codec.Respond(msg.ID, map[string]any{"opId": opID, "result": res})
}

// CopyModelFrom copies model for engine from src into this node's engine store.
func (e *Executor) CopyModelFrom(ctx context.Context, src modelFileSource, engine, model string, emit func(copyProgress)) (copyModelResult, error) {
	if emit == nil {
		emit = func(copyProgress) {}
	}
	stage := func(stage, msg string) {
		emit(copyProgress{remoteProgress: remoteProgress{Stage: stage, Message: msg}})
	}
	st, store, err := e.modelStoreFor(engine)
	if err != nil {
		return copyModelResult{}, err
	}
	status, err := e.Status(engine)
	if err != nil {
		return copyModelResult{}, err
	}
	if !status.Installed {
		return copyModelResult{}, fmt.Errorf("%s is not installed on this node; install it before copying a model to it", status.DisplayName)
	}
	key := engine + "\x00" + model
	if _, busy := activeModelCopies.LoadOrStore(key, struct{}{}); busy {
		return copyModelResult{}, fmt.Errorf("a copy of %q is already running on this node", model)
	}
	defer activeModelCopies.Delete(key)

	stage("listing", "")
	list, err := src.modelFiles(ctx, engine, model, func(f streamFrame) {
		if f.Type != "progress" {
			return
		}
		p := copyProgress{remoteProgress: remoteProgress{Stage: f.Stage, Message: f.Message}, File: f.Message}
		if wirePercentIncluded(f.Percent) {
			p.Percent = f.Percent
		}
		emit(p)
	})
	if err != nil {
		return copyModelResult{}, fmt.Errorf("list %q on the source: %w", model, err)
	}
	plan, err := e.planModelCopy(st, store, model, list)
	if err != nil {
		return copyModelResult{}, err
	}
	res, err := runCopyPlan(ctx, src, engine, model, plan, emit)
	if err != nil {
		return res, err
	}
	stage("finalizing", "")
	if err := plan.finalize(ctx); err != nil {
		return res, err
	}
	e.pokeLoaded()
	emit(copyProgress{
		remoteProgress: remoteProgress{Stage: "done", Percent: 100},
		BytesDone:      res.BytesTotal, BytesTotal: res.BytesTotal,
	})
	slog.Info("model copied from peer", "engine", engine, "model", model,
		"files", res.Files, "skipped", res.FilesSkipped, "bytes", res.BytesCopied)
	return res, nil
}

// runCopyPlan checks what is already present, refuses a copy the disk cannot
// hold, and downloads the rest in plan order.
func runCopyPlan(ctx context.Context, src modelFileSource, engine, model string, plan copyPlan, emit func(copyProgress)) (copyModelResult, error) {
	res := copyModelResult{Engine: engine, Model: model, Files: len(plan.files)}
	for _, f := range plan.files {
		res.BytesTotal += f.Size
	}
	var done int64
	var lastEmit time.Time
	report := func(stage, file string, force bool) {
		if !force && time.Since(lastEmit) < copyProgressInterval {
			return
		}
		lastEmit = time.Now()
		p := copyProgress{remoteProgress: remoteProgress{Stage: stage}, File: file, BytesDone: done, BytesTotal: res.BytesTotal}
		if res.BytesTotal > 0 {
			p.Percent = int(done * 100 / res.BytesTotal)
		}
		emit(p)
	}

	// Already present with the right content: skip. Otherwise count what is
	// still missing, net of any partial a previous attempt left.
	skip := make([]bool, len(plan.files))
	var needed int64
	for i, f := range plan.files {
		report("checking", f.Path, true)
		ok, err := destMatches(ctx, f)
		if err != nil {
			return res, err
		}
		if ok {
			skip[i] = true
			res.FilesSkipped++
			done += f.Size
			continue
		}
		have := int64(0)
		if info, err := os.Stat(f.dest + ".part"); err == nil && info.Mode().IsRegular() && info.Size() <= f.Size {
			have = info.Size()
		}
		needed += f.Size - have
	}
	if needed > 0 {
		if err := checkFreeSpace(plan.root, needed); err != nil {
			return res, err
		}
	}

	for i, f := range plan.files {
		if skip[i] {
			continue
		}
		base := done
		report("downloading", f.Path, true)
		copied, skipped, err := downloadPlannedFile(ctx, src, engine, model, f, func(fileDone int64) {
			done = base + fileDone
			report("downloading", f.Path, false)
		})
		if err != nil {
			return res, err
		}
		done = base + f.Size
		if skipped {
			res.FilesSkipped++
		}
		res.BytesCopied += copied
		report("downloading", f.Path, true)
	}
	return res, nil
}

// destMatches reports whether f's destination already holds the listed content.
func destMatches(ctx context.Context, f plannedFile) (bool, error) {
	info, err := os.Stat(f.dest)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if info.IsDir() {
		return false, fmt.Errorf("%s is a directory on this node", f.dest)
	}
	if !info.Mode().IsRegular() || info.Size() != f.Size {
		return false, nil
	}
	sum, err := fileHashes.sum(ctx, f.dest, nil)
	if err != nil {
		return false, err
	}
	return sum == f.SHA256, nil
}

// checkFreeSpace refuses a copy that would leave less than the headroom free.
func checkFreeSpace(root string, needed int64) error {
	dir := nearestExistingDir(root)
	free, err := diskFreeBytes(dir)
	if err != nil {
		return fmt.Errorf("could not check free disk space at %s: %w", dir, err)
	}
	if want := needed + copyFreeSpaceHeadroom; free < uint64(want) {
		return fmt.Errorf("not enough free disk space on this node: the copy needs %s more and %s is free at %s (with %s kept in reserve)",
			formatBytes(needed), formatBytes(int64(free)), dir, formatBytes(copyFreeSpaceHeadroom))
	}
	return nil
}

func nearestExistingDir(path string) string {
	for p := filepath.Clean(path); ; {
		if info, err := os.Stat(p); err == nil && info.IsDir() {
			return p
		}
		parent := filepath.Dir(p)
		if parent == p {
			return p
		}
		p = parent
	}
}

func formatBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return strconv.FormatInt(n, 10) + " B"
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit && exp < 4; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTP"[exp])
}

// downloadPlannedFile brings one file into place: resumed download into
// "<dest>.part", sha256 check, the plan's pre-rename check, then an atomic
// rename. It returns the bytes fetched over the network, and skipped when
// another writer placed a matching file while this one waited for its lock.
func downloadPlannedFile(ctx context.Context, src modelFileSource, engine, model string, f plannedFile, onBytes func(int64)) (copied int64, skipped bool, err error) {
	unlock := copyDestLocks.lock(f.dest)
	defer unlock()
	if ok, err := destMatches(ctx, f); err != nil || ok {
		return 0, ok, err
	}
	if err := os.MkdirAll(filepath.Dir(f.dest), 0o755); err != nil {
		return 0, false, err
	}
	part := f.dest + ".part"
	retriedFresh := false
	for attempt := 1; ; attempt++ {
		sum, resumedFrom, n, err := fetchPart(ctx, src, engine, model, f, part, onBytes)
		copied += n
		if err == nil {
			if sum != f.SHA256 {
				_ = os.Remove(part)
				// A partial left by an earlier attempt may predate a change on
				// the source; start over once before calling it a mismatch.
				if resumedFrom > 0 && !retriedFresh {
					retriedFresh = true
					continue
				}
				return copied, false, fmt.Errorf("sha256 mismatch for %s: expected %s, got %s; the download was discarded", f.Path, f.SHA256, sum)
			}
			if f.check != nil {
				if err := f.check(part); err != nil {
					return copied, false, fmt.Errorf("%s: %w", f.Path, err)
				}
			}
			if err := os.Rename(part, f.dest); err != nil {
				return copied, false, err
			}
			fileHashes.remember(f.dest, sum)
			return copied, false, nil
		}
		var perm permanentCopyError
		if errors.As(err, &perm) || ctx.Err() != nil || attempt >= copyFileAttempts {
			return copied, false, fmt.Errorf("download %s: %w", f.Path, err)
		}
		slog.Warn("model copy download interrupted; resuming", "engine", engine, "model", model,
			"file", f.Path, "attempt", attempt, "err", err)
		select {
		case <-ctx.Done():
			return copied, false, ctx.Err()
		case <-time.After(time.Duration(attempt) * copyRetryBackoff):
		}
	}
}

// fetchPart makes one attempt at completing part from wherever it stands, and
// returns the sha256 of the whole file, the offset it resumed from, and the
// bytes it fetched.
func fetchPart(ctx context.Context, src modelFileSource, engine, model string, f plannedFile, part string, onBytes func(int64)) (sum string, resumedFrom, fetched int64, err error) {
	fh, err := os.OpenFile(part, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return "", 0, 0, permanentCopyError{err}
	}
	defer fh.Close()
	info, err := fh.Stat()
	if err != nil {
		return "", 0, 0, err
	}
	offset := info.Size()
	if offset > f.Size {
		if err := fh.Truncate(0); err != nil {
			return "", 0, 0, err
		}
		offset = 0
	}
	// Hash what is already there, then the rest as it arrives, so a fresh
	// download is read from disk only once.
	h := sha256.New()
	if offset > 0 {
		if _, err := copyWithContext(ctx, h, io.NewSectionReader(fh, 0, offset), nil); err != nil {
			return "", offset, 0, err
		}
		onBytes(offset)
	}
	if offset < f.Size {
		n, err := fetchRange(ctx, src, engine, model, f, fh, h, offset, onBytes)
		fetched = n
		if err != nil {
			return "", offset, fetched, err
		}
	}
	if err := fh.Sync(); err != nil {
		return "", offset, fetched, err
	}
	return hex.EncodeToString(h.Sum(nil)), offset, fetched, nil
}

// fetchRange downloads f from offset into fh (positioned by WriteAt), feeding h.
func fetchRange(ctx context.Context, src modelFileSource, engine, model string, f plannedFile, fh *os.File, h io.Writer, offset int64, onBytes func(int64)) (int64, error) {
	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	body, start, err := src.modelFile(attemptCtx, engine, model, f.Path, offset)
	if errors.Is(err, errRangeNotSatisfiable) {
		_ = fh.Truncate(0)
		return 0, err
	}
	if err != nil {
		return 0, err
	}
	defer body.Close()
	if start != offset {
		if start != 0 {
			return 0, fmt.Errorf("source answered from offset %d, asked for %d", start, offset)
		}
		// A full reply: discard the partial and the prefix hashed from it.
		if err := fh.Truncate(0); err != nil {
			return 0, err
		}
		if r, ok := h.(interface{ Reset() }); ok {
			r.Reset()
		}
		offset = 0
	}
	stall := time.AfterFunc(copyStallTimeout, cancel)
	defer stall.Stop()
	w := &offsetWriter{f: fh, off: offset}
	remaining := f.Size - offset
	n, err := copyWithContext(attemptCtx, io.MultiWriter(w, h), io.LimitReader(body, remaining), func(n int64) {
		stall.Reset(copyStallTimeout)
		onBytes(offset + n)
	})
	if err != nil {
		if ctx.Err() == nil && attemptCtx.Err() != nil {
			return n, fmt.Errorf("no data for %s", copyStallTimeout)
		}
		return n, err
	}
	if n < remaining {
		return n, io.ErrUnexpectedEOF
	}
	return n, nil
}

type offsetWriter struct {
	f   *os.File
	off int64
}

func (w *offsetWriter) Write(p []byte) (int, error) {
	n, err := w.f.WriteAt(p, w.off)
	w.off += int64(n)
	return n, err
}

// ---- ec client side ----

func modelQueryValues(engine, model string) url.Values {
	return url.Values{"engine": {engine}, "model": {model}}
}

// modelFiles implements modelFileSource over a peer's GET /v1/models/files.
func (c *remoteClient) modelFiles(ctx context.Context, engine, model string, onProgress func(streamFrame)) (modelFileList, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+controlModelFilesPath+"?"+modelQueryValues(engine, model).Encode(), nil)
	if err != nil {
		return modelFileList{}, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		c.forgetAddress()
		return modelFileList{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		return modelFileList{}, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	dec := json.NewDecoder(resp.Body)
	for {
		var f streamFrame
		if err := dec.Decode(&f); err != nil {
			if err == io.EOF {
				return modelFileList{}, fmt.Errorf("the listing ended without a result")
			}
			return modelFileList{}, fmt.Errorf("decode listing frame: %w", err)
		}
		switch f.Type {
		case "progress":
			if onProgress != nil {
				onProgress(f)
			}
		case "error":
			return modelFileList{}, fmt.Errorf("%s", f.Message)
		case "result":
			var list modelFileList
			if err := json.Unmarshal(f.Result, &list); err != nil {
				return modelFileList{}, fmt.Errorf("decode listing: %w", err)
			}
			return list, nil
		}
	}
}

// modelFile implements modelFileSource over a peer's GET /v1/models/file.
func (c *remoteClient) modelFile(ctx context.Context, engine, model, path string, offset int64) (io.ReadCloser, int64, error) {
	q := modelQueryValues(engine, model)
	q.Set("path", path)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+controlModelFilePath+"?"+q.Encode(), nil)
	if err != nil {
		return nil, 0, err
	}
	if offset > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(offset, 10)+"-")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		c.forgetAddress()
		return nil, 0, err
	}
	switch resp.StatusCode {
	case http.StatusOK:
		return resp.Body, 0, nil
	case http.StatusPartialContent:
		start, err := contentRangeStart(resp.Header.Get("Content-Range"))
		if err != nil {
			resp.Body.Close()
			return nil, 0, err
		}
		return resp.Body, start, nil
	case http.StatusRequestedRangeNotSatisfiable:
		resp.Body.Close()
		return nil, 0, errRangeNotSatisfiable
	}
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	resp.Body.Close()
	err = fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	if resp.StatusCode >= 400 && resp.StatusCode < 500 {
		return nil, 0, permanentCopyError{err}
	}
	return nil, 0, err
}

// contentRangeStart parses the first byte offset of "bytes <start>-<end>/<size>".
func contentRangeStart(v string) (int64, error) {
	rest, ok := strings.CutPrefix(v, "bytes ")
	if !ok {
		return 0, fmt.Errorf("bad Content-Range %q", v)
	}
	first, _, ok := strings.Cut(rest, "-")
	if !ok {
		return 0, fmt.Errorf("bad Content-Range %q", v)
	}
	start, err := strconv.ParseInt(first, 10, 64)
	if err != nil || start < 0 {
		return 0, fmt.Errorf("bad Content-Range %q", v)
	}
	return start, nil
}
