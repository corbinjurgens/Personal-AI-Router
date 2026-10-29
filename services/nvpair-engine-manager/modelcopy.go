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
	"bufio"
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

	"nvpair-shared/nodepolicy"
)

// copyProgress is the engine:remote-progress payload for a model copy: the
// shared remote progress fields plus the file being worked on and byte counts
// across the whole model.
type copyProgress struct {
	remoteProgress
	SourceNodeID      string `json:"sourceNodeId"`
	DestinationNodeID string `json:"destinationNodeId"`
	Model             string `json:"model"`
	ExportID          string `json:"exportId,omitempty"`
	File              string `json:"file,omitempty"`
	BytesDone         int64  `json:"bytesDone,omitempty"`
	BytesTotal        int64  `json:"bytesTotal,omitempty"`
}

// copyModelResult is the engine:remote-copy-model result (under "result").
type copyModelResult struct {
	Engine                string         `json:"engine"`
	Model                 string         `json:"model"`
	Files                 int            `json:"files"`
	FilesSkipped          int            `json:"filesSkipped"`
	BytesTotal            int64          `json:"bytesTotal"`
	BytesCopied           int64          `json:"bytesCopied"`
	Layout                string         `json:"layout,omitempty"`
	ExportID              string         `json:"exportId,omitempty"`
	Durability            copyDurability `json:"durability,omitempty"`
	RequiresConfiguration bool           `json:"requiresConfiguration,omitempty"`
}

// modelFileSource is where a copy reads from: a peer's ec routes in production.
type modelFileSource interface {
	// modelFiles lists the model's files with sizes and sha256, relaying the
	// source's hashing progress frames.
	modelFiles(ctx context.Context, engine, model string, onProgress func(streamFrame)) (modelFileList, error)
	// modelFile opens one listed file from offset. start is the offset the body
	// actually begins at: offset for a ranged reply, 0 for a full one.
	modelFile(ctx context.Context, engine, model, path, exportID string, offset int64) (body io.ReadCloser, start int64, err error)
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
	gate chan struct{}
	refs int
}

func (k *keyedLocks) lock(ctx context.Context, key string) (func(), error) {
	k.mu.Lock()
	l := k.m[key]
	if l == nil {
		l = &keyedLock{gate: make(chan struct{}, 1)}
		k.m[key] = l
	}
	l.refs++
	k.mu.Unlock()
	drop := func() {
		k.mu.Lock()
		l.refs--
		if l.refs == 0 {
			delete(k.m, key)
		}
		k.mu.Unlock()
	}
	select {
	case l.gate <- struct{}{}:
		if err := ctx.Err(); err != nil {
			<-l.gate
			drop()
			return nil, err
		}
		return func() { <-l.gate; drop() }, nil
	case <-ctx.Done():
		drop()
		return nil, ctx.Err()
	}
}

// runRemoteCopy answers engine:remote-copy-model. Like the other engine:remote-*
// methods it runs on its own goroutine and settles when the copy finishes, with
// live engine:remote-progress frames meanwhile (op "copy", node = the source).
func (m *Manager) runRemoteCopy(ctx context.Context, msg *Message) {
	var legacy remoteParam
	if !m.parse(msg, &legacy) {
		return
	}
	switch {
	case legacy.Node == "":
		m.codec.RespondError(msg.ID, -32602, "node is required")
		return
	case legacy.Engine == "":
		m.codec.RespondError(msg.ID, -32602, "engine is required")
		return
	case legacy.Model == "":
		m.codec.RespondError(msg.ID, -32602, "model is required")
		return
	}
	if err := validateModelRequest(msg.Params, true); err != nil {
		m.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	self, err := m.copySelfID()
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	p := copyRequest{OpID: newOpID(), SourceNodeID: legacy.Node, DestinationNodeID: self, Engine: legacy.Engine, Model: legacy.Model}
	data, err := json.Marshal(p)
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	canonical := *msg
	canonical.Params = data
	m.runCopy(ctx, &canonical)
}

// CopyModelFrom copies model for engine from src into this node's engine store.
func (e *Executor) CopyModelFrom(ctx context.Context, src modelFileSource, engine, model string, emit func(copyProgress)) (copyModelResult, error) {
	return e.copyModelFrom(ctx, src, engine, model, "", nil, nil, emit)
}

func (e *Executor) copyModelFrom(ctx context.Context, src modelFileSource, engine, model, expectedExportID string, exportLayouts, importLayouts []string, emit func(copyProgress)) (copyModelResult, error) {
	if err := nodepolicy.ValidateModelIdentifier(model); err != nil {
		return copyModelResult{}, err
	}
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
	key := store.root + "\x00" + engine + "\x00" + model
	if _, busy := activeModelCopies.LoadOrStore(key, struct{}{}); busy {
		return copyModelResult{}, copyConflict(fmt.Sprintf("a copy of %q is already running on this node", model))
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
	if expectedExportID != "" && list.ExportID != expectedExportID {
		return copyModelResult{}, copyConflict("source export changed; retry identity does not match")
	}
	layout := list.Layout
	if layout == "" {
		layout = store.adapter
	}
	for _, supported := range [][]string{exportLayouts, importLayouts} {
		if supported == nil {
			continue
		}
		found := false
		for _, name := range supported {
			if name == layout {
				found = true
				break
			}
		}
		if !found {
			return copyModelResult{}, fmt.Errorf("source export layout is not supported by both endpoints")
		}
	}
	plan, err := e.planModelCopy(st, store, model, list)
	if err != nil {
		return copyModelResult{}, err
	}
	if list.Layout == llamaArtifactsLayout && list.ExportID != "" {
		emit(copyProgress{remoteProgress: remoteProgress{Stage: "checking"}, ExportID: list.ExportID})
	}
	// Listing the source and planning the copy are read-only. Installing the
	// downloaded files is not: the writes run under the engine's
	// content-mutation guard (model_mutation.go), so the broker withdraws that
	// engine's model identity before the first file is written.
	var res copyModelResult
	if err := e.withModelMutation(ctx, engine, func() error {
		unlock, err := copyDestLocks.lock(ctx, plan.root+"\x00"+model)
		if err != nil {
			return err
		}
		defer unlock()
		if plan.preflight != nil {
			if err := plan.preflight(ctx); err != nil {
				return err
			}
		}
		var copyErr error
		res, copyErr = runCopyPlan(ctx, src, engine, model, plan, emit)
		if copyErr != nil {
			return copyErr
		}
		stage("finalizing", "")
		return plan.finalize(ctx)
	}); err != nil {
		return res, err
	}
	if plan.layout != "" {
		res.Layout, res.ExportID, res.RequiresConfiguration = plan.layout, plan.exportID, true
		res.Durability = *plan.durability
	} else {
		e.pokeLoaded()
	}
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
			p.Percent = copyPercent(done, res.BytesTotal)
		}
		emit(p)
	}

	// Already present with the right content: skip. Otherwise count what is
	// still missing, net of any partial a previous attempt left.
	skip := make([]bool, len(plan.files))
	var needed int64
	for i, f := range plan.files {
		report("checking", f.Path, true)
		if f.root != "" {
			if err := safeCopyDestination(f.root, f.dest+".part"); err != nil {
				return res, err
			}
		}
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
	if f.root != "" {
		if err := safeCopyDestination(f.root, f.dest); err != nil {
			return false, err
		}
	}
	info, err := os.Stat(f.dest)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if info.IsDir() {
		return false, copyConflict(fmt.Sprintf("copy conflict: %s is a directory on this node", f.Path))
	}
	if !info.Mode().IsRegular() || info.Size() != f.Size {
		return false, copyConflict(fmt.Sprintf("copy conflict: existing destination content differs for %s", f.Path))
	}
	sum, err := fileHashes.sum(ctx, f.dest, nil)
	if err != nil {
		return false, err
	}
	if sum != f.SHA256 {
		return false, copyConflict(fmt.Sprintf("copy conflict: existing destination content differs for %s", f.Path))
	}
	return true, nil
}

// checkFreeSpace refuses a copy that would leave less than the headroom free.
func checkFreeSpace(root string, needed int64) error {
	dir := nearestExistingDir(root)
	free, err := diskFreeBytes(dir)
	if err != nil {
		return fmt.Errorf("could not check free disk space on the destination: %w", err)
	}
	if want := uint64(needed) + uint64(copyFreeSpaceHeadroom); free < want {
		return fmt.Errorf("not enough free disk space on this node: the copy needs %s more and %s is free (with %s kept in reserve)",
			formatBytes(needed), formatBytes(int64(free)), formatBytes(copyFreeSpaceHeadroom))
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
	unlock, err := copyDestLocks.lock(ctx, f.dest)
	if err != nil {
		return 0, false, err
	}
	defer unlock()
	if ok, err := destMatches(ctx, f); err != nil || ok {
		return 0, ok, err
	}
	if err := safeCopyDestination(f.root, f.dest+".part"); err != nil {
		return 0, false, err
	}
	if err := os.MkdirAll(f.root, 0o755); err != nil {
		return 0, false, err
	}
	root, err := os.OpenRoot(f.root)
	if err != nil {
		return 0, false, err
	}
	defer root.Close()
	destRel, err := filepath.Rel(f.root, f.dest)
	if err != nil {
		return 0, false, err
	}
	if err := root.MkdirAll(filepath.Dir(destRel), 0o755); err != nil {
		return 0, false, err
	}
	part := f.dest + ".part"
	partRel := destRel + ".part"
	retriedFresh := false
	for attempt := 1; ; attempt++ {
		sum, resumedFrom, n, verified, err := fetchPart(ctx, src, engine, model, f, root, partRel, onBytes)
		copied += n
		if err == nil {
			if sum != f.SHA256 {
				if err := checkCopyPartial(root, partRel, verified); err != nil {
					return copied, false, err
				}
				_ = root.Remove(partRel)
				// A partial left by an earlier attempt may predate a change on
				// the source; start over once before calling it a mismatch.
				if resumedFrom > 0 && !retriedFresh {
					retriedFresh = true
					continue
				}
				return copied, false, fmt.Errorf("sha256 mismatch for %s: expected %s, got %s; the download was discarded", f.Path, f.SHA256, sum)
			}
			if err := checkCopyPartial(root, partRel, verified); err != nil {
				return copied, false, err
			}
			if f.check != nil {
				if err := f.check(part); err != nil {
					return copied, false, fmt.Errorf("%s: %w", f.Path, err)
				}
			}
			if ok, err := destMatches(ctx, f); err != nil || ok {
				return copied, ok, err
			}
			if err := checkCopyPartial(root, partRel, verified); err != nil {
				return copied, false, err
			}
			if err := root.Rename(partRel, destRel); err != nil {
				return copied, false, err
			}
			// Rename operates on names. Recheck its actual result as well as
			// the source name so a concurrent path swap cannot become a verified
			// completed file or a custom publication receipt.
			if err := checkCopyPartial(root, destRel, verified); err != nil {
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

func checkCopyPartial(root *os.Root, name string, verified os.FileInfo) error {
	info, err := root.Lstat(name)
	if err != nil || !sameCopyFile(info, verified) {
		return permanentCopyError{fmt.Errorf("copy partial pathname or regular-file identity changed")}
	}
	return nil
}

// fetchPart makes one attempt at completing part from wherever it stands, and
// returns the sha256 of the whole file, the offset it resumed from, and the
// bytes it fetched.
func fetchPart(ctx context.Context, src modelFileSource, engine, model string, f plannedFile, root *os.Root, part string, onBytes func(int64)) (sum string, resumedFrom, fetched int64, verified os.FileInfo, err error) {
	// Repeat containment and type checks on every attempt, including retries.
	// Root.OpenFile confines symlink traversal during the open itself; none of
	// the subsequent truncation or writes can follow an escaping path swap.
	if err := safeCopyDestination(f.root, filepath.Join(f.root, part)); err != nil {
		return "", 0, 0, nil, permanentCopyError{err}
	}
	before, err := root.Lstat(part)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", 0, 0, nil, permanentCopyError{err}
	}
	if err == nil && !before.Mode().IsRegular() {
		return "", 0, 0, nil, permanentCopyError{fmt.Errorf("copy partial must be a regular file")}
	}
	fh, err := root.OpenFile(part, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return "", 0, 0, nil, permanentCopyError{err}
	}
	defer fh.Close()
	info, err := fh.Stat()
	if err != nil {
		return "", 0, 0, nil, err
	}
	if err := checkCopyPartial(root, part, info); err != nil {
		return "", 0, 0, nil, err
	}
	if before != nil && !os.SameFile(before, info) {
		return "", 0, 0, nil, permanentCopyError{fmt.Errorf("copy partial changed while opening")}
	}
	offset := info.Size()
	if offset > f.Size {
		if err := fh.Truncate(0); err != nil {
			return "", 0, 0, nil, err
		}
		offset = 0
	}
	// Hash what is already there, then the rest as it arrives, so a fresh
	// download is read from disk only once.
	h := sha256.New()
	if offset > 0 {
		if _, err := copyWithContext(ctx, h, io.NewSectionReader(fh, 0, offset), nil); err != nil {
			return "", offset, 0, nil, err
		}
		onBytes(offset)
	}
	if offset < f.Size {
		n, err := fetchRange(ctx, src, engine, model, f, fh, h, offset, onBytes)
		fetched = n
		if err != nil {
			return "", offset, fetched, nil, err
		}
	}
	if err := fh.Sync(); err != nil {
		return "", offset, fetched, nil, err
	}
	verified, err = fh.Stat()
	if err != nil {
		return "", offset, fetched, nil, err
	}
	if err := checkCopyPartial(root, part, verified); err != nil {
		return "", offset, fetched, nil, err
	}
	return hex.EncodeToString(h.Sum(nil)), offset, fetched, verified, nil
}

// fetchRange downloads f from offset into fh (positioned by WriteAt), feeding h.
func fetchRange(ctx context.Context, src modelFileSource, engine, model string, f plannedFile, fh *os.File, h io.Writer, offset int64, onBytes func(int64)) (int64, error) {
	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	body, start, err := src.modelFile(attemptCtx, engine, model, f.Path, f.exportID, offset)
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
	if err := nodepolicy.ValidateModelIdentifier(model); err != nil {
		return modelFileList{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+controlModelFilesPath+"?"+modelQueryValues(engine, model).Encode(), nil)
	if err != nil {
		return modelFileList{}, err
	}
	resp, err := c.controlClient(false).Do(req)
	if err != nil {
		c.forgetAddress()
		return modelFileList{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		return modelFileList{}, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	reader := bufio.NewReaderSize(resp.Body, 64<<10)
	for {
		line, err := readCopyFrame(reader)
		if err != nil {
			return modelFileList{}, fmt.Errorf("decode listing frame: %w", err)
		}
		var f streamFrame
		if err := json.Unmarshal(line, &f); err != nil {
			if err == io.EOF {
				return modelFileList{}, fmt.Errorf("the listing ended without a result")
			}
			return modelFileList{}, fmt.Errorf("decode listing frame: %w", err)
		}
		switch f.Type {
		case "progress":
			if len(line) > maxCopyProgressFrame {
				return modelFileList{}, fmt.Errorf("copy progress frame exceeds bounds")
			}
			if onProgress != nil {
				onProgress(f)
			}
		case "error":
			return modelFileList{}, fmt.Errorf("%s", f.Message)
		case "result":
			if len(f.Result) > maxCopyManifest {
				return modelFileList{}, fmt.Errorf("copy manifest exceeds bounds")
			}
			list, err := decodeCopyManifest(f.Result)
			if err != nil {
				return modelFileList{}, fmt.Errorf("decode listing: %w", err)
			}
			return list, nil
		}
	}
}

// modelFile implements modelFileSource over a peer's GET /v1/models/file.
func (c *remoteClient) modelFile(ctx context.Context, engine, model, path, exportID string, offset int64) (io.ReadCloser, int64, error) {
	if err := nodepolicy.ValidateModelIdentifier(model); err != nil {
		return nil, 0, err
	}
	q := modelQueryValues(engine, model)
	q.Set("path", path)
	if exportID != "" {
		q.Set("exportId", exportID)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+controlModelFilePath+"?"+q.Encode(), nil)
	if err != nil {
		return nil, 0, err
	}
	if offset > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(offset, 10)+"-")
	}
	resp, err := c.controlClient(false).Do(req)
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
