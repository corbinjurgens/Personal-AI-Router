// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// modelfiles.go is the source half of peer-to-peer model copy: two ec routes a
// pinned peer uses to fetch a model's files straight from this node's engine
// store instead of downloading them again from the internet.
//
//   - GET /v1/models/files?engine=&model= lists the model's files, as paths
//     relative to the engine's store, with size and sha256.
//   - GET /v1/models/file?engine=&model=&path= serves one listed file, with HTTP
//     Range support so an interrupted download resumes.
//
// The file set is resolved per request from the engine's layout or an exact
// custom llama.cpp local binding (see modellayout.go and modelcopy_export.go).
// Only listed relative paths are accepted; external custom artifacts additionally
// require the source-local configured allowlist and issued export identity.

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
	"os"
	"strings"
	"sync"
	"time"

	"nvpair-shared/nodepolicy"
)

const (
	controlModelFilesPath = "/v1/models/files"
	controlModelFilePath  = "/v1/models/file"
)

// modelFile is one file of a model, relative to the engine's store.
type modelFile struct {
	// Path is slash-separated and relative to the layout root the engine uses
	// for this model (see modelFileList).
	Path      string              `json:"path"`
	Role      string              `json:"role,omitempty"`
	Size      int64               `json:"size"`
	SHA256    string              `json:"sha256"`
	Publisher *modelPublisherFile `json:"publisher,omitempty"`
	// abs is the file's resolved absolute path on the source. Never serialized.
	abs        string
	info       os.FileInfo
	asserted   string
	configured string
	sample     string
}

// modelFileList is the GET /v1/models/files result.
type modelFileList struct {
	Engine        string `json:"engine"`
	Model         string `json:"model"`
	Layout        string `json:"layout,omitempty"`
	ExportID      string `json:"exportId,omitempty"`
	binding       string
	configuration string
	// source retains request-local selection evidence, never peer-visible paths.
	source *sourceSnapshot
	// Revision is the snapshot a llama.cpp model's files were listed from (the
	// Hugging Face commit in its cache). Empty for other engines.
	Revision string      `json:"revision,omitempty"`
	Files    []modelFile `json:"files"`
}

// errModelNotFound marks a model this node does not hold, answered with a 404.
var errModelNotFound = errors.New("model not found")

// validModelRelPath reports whether p is a safe relative file path: forward
// slashes only, no empty, "." or ".." segment, nothing absolute, and none of the
// characters Windows treats specially in a path (drive letters, streams).
func validModelRelPath(p string) bool {
	if p == "" || len(p) > 1024 || strings.HasPrefix(p, "/") {
		return false
	}
	if strings.ContainsAny(p, "\\:\x00*?\"<>|") {
		return false
	}
	for _, seg := range strings.Split(p, "/") {
		for _, r := range seg {
			if r < 32 {
				return false
			}
		}
		if seg == "" || seg == "." || seg == ".." {
			return false
		}
		// Windows strips trailing dots and spaces, so "a." and "a" would be
		// the same file there.
		if strings.HasSuffix(seg, ".") || strings.HasSuffix(seg, " ") {
			return false
		}
		base, _, _ := strings.Cut(strings.ToUpper(seg), ".")
		switch base {
		case "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "COM¹", "COM²", "COM³", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9", "LPT¹", "LPT²", "LPT³":
			return false
		}
	}
	return true
}

// validSHA256 reports whether s is a lowercase hex sha256 digest.
func validSHA256(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// confinedRegularFile resolves abs through any symlinks and checks the result is
// a regular file inside root (itself resolved). It returns the resolved path and
// its FileInfo.
func confinedRegularFile(root, abs string) (string, os.FileInfo, error) {
	realRoot, err := resolveExistingPath(root)
	if err != nil {
		return "", nil, err
	}
	real, err := resolveExistingPath(abs)
	if err != nil {
		return "", nil, err
	}
	if !pathWithinRoot(realRoot, real) {
		return "", nil, fmt.Errorf("%s resolves outside the model store", abs)
	}
	info, err := os.Stat(real)
	if err != nil {
		return "", nil, err
	}
	if !info.Mode().IsRegular() {
		return "", nil, fmt.Errorf("%s is not a regular file", abs)
	}
	// Windows Stat loads identity lazily. Pin it while this native binding
	// still names the captured object, before any later cache validation.
	if !os.SameFile(info, info) {
		return "", nil, fmt.Errorf("%s file identity is unavailable", abs)
	}
	return real, info, nil
}

// hashCache remembers the sha256 of files by absolute path, size and mtime, so a
// model listed twice (a retried copy, a second destination) is hashed once.
// Concurrent requests for the same file share one hashing pass.
type hashCache struct {
	mu       sync.Mutex
	done     map[string]hashEntry
	inflight map[hashEntry]*hashCall
	access   servingFileAccess
}

type hashEntry struct {
	path  string
	size  int64
	mtime int64
	sum   string
	info  os.FileInfo
}

type hashCall struct {
	done   chan struct{}
	info   os.FileInfo
	sum    string
	err    error
	onWait func() // Optional synchronization for tests that inject an in-flight call.
}

// fileHashes is the process-wide cache shared by the ec listing (source side)
// and the copy's already-present check (destination side).
var fileHashes = &hashCache{done: map[string]hashEntry{}, inflight: map[hashEntry]*hashCall{}}

// sum returns abs's sha256, computing it by streaming the file when the cache
// has no entry for its current size and mtime. onBytes, when set, receives the
// running count of bytes hashed.
func (c *hashCache) sum(ctx context.Context, abs string, onBytes func(int64)) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	e, err := captureSource(abs, false)
	if err != nil {
		return "", err
	}
	return c.sumBoundKey(ctx, e, abs, onBytes)
}

// sumBound retains the requesting object's authority across cache hits, waits
// and retries. Only the path-based wrappers may capture a current object.
func (c *hashCache) sumBound(ctx context.Context, e sourceEvidence, onBytes func(int64)) (string, error) {
	return c.sumBoundKey(ctx, e, e.target, onBytes)
}

func (c *hashCache) sumBoundKey(ctx context.Context, e sourceEvidence, abs string, onBytes func(int64)) (string, error) {
	return c.sumBoundKeyAccess(ctx, e, abs, onBytes, c.access)
}

func (c *hashCache) sumBoundKeyAccess(ctx context.Context, e sourceEvidence, abs string, onBytes func(int64), access servingFileAccess) (string, error) {
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		if e.directory || validateSource(e) != nil {
			return "", fmt.Errorf("invalid hash source evidence")
		}
		info := e.info
		key := hashEntry{path: abs, size: info.Size(), mtime: info.ModTime().UnixNano()}
		c.mu.Lock()
		if cached, ok := c.done[abs]; ok && cached.size == key.size && cached.mtime == key.mtime && cached.info != nil && os.SameFile(cached.info, info) {
			c.mu.Unlock()
			if err := validateSource(e); err != nil {
				return "", err
			}
			if err := ctx.Err(); err != nil {
				return "", err
			}
			return cached.sum, nil
		}
		if call, ok := c.inflight[key]; ok {
			c.mu.Unlock()
			if call.info != nil && !os.SameFile(info, call.info) {
				return "", fmt.Errorf("hash source changed")
			}
			if call.onWait != nil {
				call.onWait()
			}
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-call.done:
				if err := ctx.Err(); err != nil {
					return "", err
				}
				if err := validateSource(e); err != nil {
					return "", err
				}
				if errors.Is(call.err, context.Canceled) || errors.Is(call.err, context.DeadlineExceeded) {
					continue
				}
				return call.sum, call.err
			}
		}
		call := &hashCall{done: make(chan struct{}), info: info}
		if c.inflight == nil {
			c.inflight = make(map[hashEntry]*hashCall)
		}
		c.inflight[key] = call
		c.mu.Unlock()

		call.sum, call.err = c.hashBoundAccess(ctx, e, onBytes, access)
		c.mu.Lock()
		delete(c.inflight, key)
		if ctx.Err() != nil {
			call.sum, call.err = "", ctx.Err()
		}
		if call.err == nil {
			// Store the original pinned identity, never a later path stat.
			if err := validateSource(e); err == nil {
				key.sum = call.sum
				key.info = info
				if c.done == nil {
					c.done = make(map[string]hashEntry)
				}
				c.done[abs] = key
			} else {
				call.sum, call.err = "", fmt.Errorf("hash source changed")
			}
		}
		c.mu.Unlock()
		close(call.done)
		return call.sum, call.err
	}
}

// remember records a digest the caller just verified, such as a downloaded file
// before its rename, so the next already-present check need not re-read it.
func (c *hashCache) remember(abs, sum string) {
	e, err := captureSource(abs, false)
	if err != nil {
		return
	}
	c.mu.Lock()
	if c.done == nil {
		c.done = make(map[string]hashEntry)
	}
	c.done[abs] = hashEntry{path: abs, size: e.size, mtime: e.mtime, sum: sum, info: e.info}
	c.mu.Unlock()
}

// hashFile streams abs through sha256 without holding it in memory.
func hashFile(ctx context.Context, abs string, onBytes func(int64)) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	e, err := captureSource(abs, false)
	if err != nil {
		return "", err
	}
	return fileHashes.sumBoundKey(ctx, e, abs, onBytes)
}

func (c *hashCache) hashBound(ctx context.Context, e sourceEvidence, onBytes func(int64)) (string, error) {
	return c.hashBoundAccess(ctx, e, onBytes, c.access)
}

func (c *hashCache) hashBoundAccess(ctx context.Context, e sourceEvidence, onBytes func(int64), access servingFileAccess) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if e.directory || validateSource(e) != nil {
		return "", fmt.Errorf("invalid hash source evidence")
	}
	f, err := access.checkedOpen(ctx, e.configured, e.target, e.info)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	n, err := copyWithContext(ctx, h, io.NewSectionReader(servingContextReaderAt{ctx: ctx, reader: f, expected: e}, 0, e.size), onBytes)
	if err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	after, err := f.Stat()
	if n != e.size || err != nil || !sameSource(e.info, after, false) || validateSource(e) != nil {
		return "", fmt.Errorf("hash source changed")
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// copyWithContext copies src to dst in 1 MiB chunks, checking ctx between
// chunks and reporting the running total to onBytes.
func copyWithContext(ctx context.Context, dst io.Writer, src io.Reader, onBytes func(int64)) (int64, error) {
	buf := make([]byte, 1<<20)
	var total int64
	for {
		if err := ctx.Err(); err != nil {
			return total, err
		}
		n, rerr := src.Read(buf)
		if n > 0 {
			if _, werr := dst.Write(buf[:n]); werr != nil {
				return total, werr
			}
			total += int64(n)
			if onBytes != nil {
				onBytes(total)
			}
		}
		if rerr == io.EOF {
			return total, nil
		}
		if rerr != nil {
			return total, rerr
		}
	}
}

// modelQuery reads and validates the engine and model query parameters shared
// by both routes.
func modelQuery(w http.ResponseWriter, r *http.Request) (engine, model string, ok bool) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return "", "", false
	}
	q := r.URL.Query()
	engine, model = q.Get("engine"), q.Get("model")
	if engine == "" || model == "" {
		http.Error(w, `"engine" and "model" are required`, http.StatusBadRequest)
		return "", "", false
	}
	if err := nodepolicy.ValidateModelIdentifier(model); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return "", "", false
	}
	return engine, model, true
}

func modelLookupStatus(err error) int {
	if errors.Is(err, errModelNotFound) {
		return http.StatusNotFound
	}
	if errors.Is(err, errModelCopyUnsupported) {
		return http.StatusNotImplemented
	}
	return http.StatusInternalServerError
}

// handleModelFiles streams the model's file list. The body is NDJSON in the same
// frame shape as the other streaming ec routes: progress frames while files are
// hashed (stage "hashing", message = the file's path), then one result frame
// whose result is the modelFileList, or one error frame. Hashing a large model
// for the first time takes longer than an ordinary response-header budget, and
// the frames keep the connection visibly alive meanwhile.
func (s *controlServer) handleModelFiles(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	engine, model, ok := modelQuery(w, r)
	if !ok {
		return
	}
	release, err := s.exec.acquireModelRead(engine, model, false)
	if err != nil {
		http.Error(w, "model content unavailable", http.StatusConflict)
		return
	}
	defer release()
	st, err := s.exec.state(engine)
	if err != nil {
		http.Error(w, "model content unavailable", http.StatusConflict)
		return
	}
	generation := issuedCopyGeneration(st, model)
	issued := false
	defer func() {
		if !issued {
			invalidateCopyGeneration(st, model, generation)
		}
	}()
	list, err := s.exec.modelSourceFilesWithAccess(ctx, engine, model, s.access)
	if err != nil {
		http.Error(w, "model source unavailable", modelLookupStatus(err))
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()
	enc := json.NewEncoder(w)
	writeFrame := func(f streamFrame) bool {
		if err := enc.Encode(f); err != nil {
			cancel()
			return false
		}
		flusher.Flush()
		return true
	}
	if err := fileHashes.hashModelFilesAccess(ctx, &list, func(path string, pct int) {
		writeFrame(streamFrame{Type: "progress", Engine: engine, Op: "copy", Stage: "hashing", Percent: pct, Message: path})
	}, s.access); err != nil {
		writeFrame(streamFrame{Type: "error", Engine: engine, Op: "copy", Message: "model source unavailable"})
		return
	}
	if err := s.exec.recheckCopyExportAccess(ctx, list, s.access); err != nil {
		writeFrame(streamFrame{Type: "error", Message: "source binding changed"})
		return
	}
	if list.Layout != llamaArtifactsLayout {
		list.ExportID, err = nativeExportID(list)
		if err != nil {
			writeFrame(streamFrame{Type: "error", Message: err.Error()})
			return
		}
	}
	_, store, storeErr := s.exec.modelStoreFor(engine)
	if storeErr == nil && store.adapter == "llamacpp" && list.source != nil {
		for i := range list.Files {
			list.Files[i].Publisher = publisherSelector(list, list.Files[i])
		}
	}
	res, err := marshalCopyManifest(list)
	if err != nil || len(res) > maxCopyManifest {
		if err == nil {
			err = fmt.Errorf("copy manifest exceeds bounds")
		}
		writeFrame(streamFrame{Type: "error", Engine: engine, Op: "copy", Message: err.Error()})
		return
	}
	newGeneration := rememberCopyExport(st, list)
	issued = writeFrame(streamFrame{Type: "result", Engine: engine, Op: "copy", Result: res})
	if !issued {
		invalidateCopyGeneration(st, model, newGeneration)
	}
}

// Optional observations must never prevent an otherwise bounded export. Keep
// the original manifest limit and byte identity; omit selectors as one complete
// set rather than presenting partial publisher coverage as complete.
func marshalCopyManifest(list modelFileList) ([]byte, error) {
	data, err := json.Marshal(list)
	if err != nil || len(data) <= maxCopyManifest {
		return data, err
	}
	list.Files = append([]modelFile(nil), list.Files...)
	for i := range list.Files {
		list.Files[i].Publisher = nil
	}
	return json.Marshal(list)
}

// hashProgressInterval spaces hashing progress frames.
var hashProgressInterval = 500 * time.Millisecond

// hashModelFiles fills in every missing sha256 in list, reporting progress per
// file at most every hashProgressInterval.
func hashModelFiles(ctx context.Context, list *modelFileList, progress func(path string, pct int)) error {
	return fileHashes.hashModelFiles(ctx, list, progress)
}

func (c *hashCache) hashModelFiles(ctx context.Context, list *modelFileList, progress func(path string, pct int)) error {
	return c.hashModelFilesAccess(ctx, list, progress, c.access)
}

func (c *hashCache) hashModelFilesAccess(ctx context.Context, list *modelFileList, progress func(path string, pct int), access servingFileAccess) error {
	for i := range list.Files {
		f := &list.Files[i]
		// HF cache blob names are not full-byte observations. Use the existing
		// checked full hash/cache even when the native resolver supplied a name.
		if list.Layout == "" && list.Revision != "" && list.source != nil {
			f.SHA256 = ""
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if f.info == nil {
			return fmt.Errorf("missing source evidence for %s", f.Path)
		}
		evidence := sourceEvidence{configured: f.configured, target: f.abs, info: f.info, size: f.Size, mtime: f.info.ModTime().UnixNano()}
		if err := validateSource(evidence); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if f.SHA256 != "" {
			if f.asserted != "" && f.asserted != f.SHA256 {
				return fmt.Errorf("full SHA256 disagrees with the configured artifact for %s", f.Path)
			}
			continue
		}
		var last time.Time
		sum, err := c.sumBoundKeyAccess(ctx, evidence, evidence.target, func(done int64) {
			if time.Since(last) < hashProgressInterval || f.Size <= 0 {
				return
			}
			last = time.Now()
			if progress != nil {
				progress(f.Path, copyPercent(done, f.Size))
			}
		}, access)
		if err != nil {
			if list.Layout == llamaArtifactsLayout {
				return fmt.Errorf("hash source artifact %s failed or changed", f.Path)
			}
			return fmt.Errorf("hash %s: %w", f.Path, err)
		}
		if f.asserted != "" && f.asserted != sum {
			return fmt.Errorf("full SHA256 disagrees with the configured artifact for %s", f.Path)
		}
		if err := validateSource(evidence); err != nil {
			return fmt.Errorf("source artifact changed for %s", f.Path)
		}
		f.SHA256 = sum
	}
	if list.Layout == llamaArtifactsLayout {
		var err error
		list.ExportID, err = artifactExportID(*list)
		return err
	}
	return nil
}

// handleModelFile serves one of the model's listed files. http.ServeContent
// answers Range (and If-Range) requests, which is how the destination resumes a
// partial download.
func (s *controlServer) handleModelFile(w http.ResponseWriter, r *http.Request) {
	engine, model, ok := modelQuery(w, r)
	if !ok {
		return
	}
	release, err := s.exec.acquireModelRead(engine, model, false)
	if err != nil {
		http.Error(w, "model content unavailable", http.StatusConflict)
		return
	}
	defer release()
	rel := r.URL.Query().Get("path")
	if !validModelRelPath(rel) {
		http.Error(w, "invalid path", http.StatusBadRequest)
		return
	}
	st, err := s.exec.state(engine)
	if err != nil {
		http.Error(w, "export expired", http.StatusConflict)
		return
	}
	list, ok := issuedCopyList(st, model, r.URL.Query().Get("exportId"))
	if !ok {
		http.Error(w, "export expired; list the model again", http.StatusConflict)
		return
	}
	var selected modelFile
	for _, f := range list.Files {
		if f.Path == rel {
			selected = f
			break
		}
	}
	if selected.abs == "" {
		http.Error(w, "path is not a file of this model", http.StatusNotFound)
		return
	}
	if err := s.exec.recheckCopyExportAccess(r.Context(), list, s.access); err != nil {
		http.Error(w, "source binding changed", http.StatusConflict)
		return
	}
	if !checkIssuedCopyExport(st, list) {
		http.Error(w, "source binding changed or export expired; list the model again", http.StatusConflict)
		return
	}
	evidence := sourceEvidence{configured: selected.configured, target: selected.abs, info: selected.info, size: selected.Size}
	if selected.info != nil {
		evidence.mtime = selected.info.ModTime().UnixNano()
	}
	file, err := s.access.checkedOpen(r.Context(), evidence.configured, evidence.target, evidence.info)
	if err != nil {
		http.Error(w, "source artifact changed", http.StatusConflict)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		http.Error(w, "file unavailable", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	http.ServeContent(w, r, "", info.ModTime(), io.NewSectionReader(servingContextReaderAt{ctx: r.Context(), reader: file, expected: evidence}, 0, evidence.size))
	final, statErr := file.Stat()
	if statErr != nil || !sameSource(evidence.info, final, false) || validateSource(evidence) != nil {
		slog.Warn("model source changed during file response")
	}
}
