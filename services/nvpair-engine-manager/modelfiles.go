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
// The file set is resolved per request from the engine's own layout (see
// modellayout.go) and nothing outside it is ever served: a path is accepted only
// when it is exactly one of the model's listed files, and every listed file must
// resolve, after symlinks, inside the engine's store.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	controlModelFilesPath = "/v1/models/files"
	controlModelFilePath  = "/v1/models/file"
)

// modelFile is one file of a model, relative to the engine's store.
type modelFile struct {
	// Path is slash-separated and relative to the layout root the engine uses
	// for this model (see modelFileList).
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
	// abs is the file's resolved absolute path on the source. Never serialized.
	abs string
}

// modelFileList is the GET /v1/models/files result.
type modelFileList struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
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
		if seg == "" || seg == "." || seg == ".." {
			return false
		}
		// Windows strips trailing dots and spaces, so "a." and "a" would be
		// the same file there.
		if strings.HasSuffix(seg, ".") || strings.HasSuffix(seg, " ") {
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
	realRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return "", nil, err
	}
	real, err := filepath.EvalSymlinks(abs)
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
	return real, info, nil
}

// hashCache remembers the sha256 of files by absolute path, size and mtime, so a
// model listed twice (a retried copy, a second destination) is hashed once.
// Concurrent requests for the same file share one hashing pass.
type hashCache struct {
	mu       sync.Mutex
	done     map[string]hashEntry
	inflight map[hashEntry]*hashCall
}

type hashEntry struct {
	path  string
	size  int64
	mtime int64
	sum   string
}

type hashCall struct {
	wg  sync.WaitGroup
	sum string
	err error
}

// fileHashes is the process-wide cache shared by the ec listing (source side)
// and the copy's already-present check (destination side).
var fileHashes = &hashCache{done: map[string]hashEntry{}, inflight: map[hashEntry]*hashCall{}}

// sum returns abs's sha256, computing it by streaming the file when the cache
// has no entry for its current size and mtime. onBytes, when set, receives the
// running count of bytes hashed.
func (c *hashCache) sum(ctx context.Context, abs string, onBytes func(int64)) (string, error) {
	info, err := os.Stat(abs)
	if err != nil {
		return "", err
	}
	key := hashEntry{path: abs, size: info.Size(), mtime: info.ModTime().UnixNano()}
	c.mu.Lock()
	if e, ok := c.done[abs]; ok && e.size == key.size && e.mtime == key.mtime {
		c.mu.Unlock()
		return e.sum, nil
	}
	if call, ok := c.inflight[key]; ok {
		c.mu.Unlock()
		call.wg.Wait()
		return call.sum, call.err
	}
	call := &hashCall{}
	call.wg.Add(1)
	c.inflight[key] = call
	c.mu.Unlock()

	call.sum, call.err = hashFile(ctx, abs, onBytes)
	c.mu.Lock()
	delete(c.inflight, key)
	if call.err == nil {
		// Store only if the file did not change while it was being read.
		if after, err := os.Stat(abs); err == nil && after.Size() == key.size && after.ModTime().UnixNano() == key.mtime {
			key.sum = call.sum
			c.done[abs] = key
		}
	}
	c.mu.Unlock()
	call.wg.Done()
	return call.sum, call.err
}

// remember records a digest the caller just verified, such as a downloaded file
// before its rename, so the next already-present check need not re-read it.
func (c *hashCache) remember(abs, sum string) {
	info, err := os.Stat(abs)
	if err != nil {
		return
	}
	c.mu.Lock()
	c.done[abs] = hashEntry{path: abs, size: info.Size(), mtime: info.ModTime().UnixNano(), sum: sum}
	c.mu.Unlock()
}

// hashFile streams abs through sha256 without holding it in memory.
func hashFile(ctx context.Context, abs string, onBytes func(int64)) (string, error) {
	f, err := os.Open(abs)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := copyWithContext(ctx, h, f, onBytes); err != nil {
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
	engine, model, ok := modelQuery(w, r)
	if !ok {
		return
	}
	list, err := s.exec.modelSourceFiles(r.Context(), engine, model)
	if err != nil {
		http.Error(w, err.Error(), modelLookupStatus(err))
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
	writeFrame := func(f streamFrame) {
		_ = enc.Encode(f)
		flusher.Flush()
	}
	if err := hashModelFiles(r.Context(), &list, func(path string, pct int) {
		writeFrame(streamFrame{Type: "progress", Engine: engine, Op: "copy", Stage: "hashing", Percent: pct, Message: path})
	}); err != nil {
		writeFrame(streamFrame{Type: "error", Engine: engine, Op: "copy", Message: err.Error()})
		return
	}
	res, err := json.Marshal(list)
	if err != nil {
		writeFrame(streamFrame{Type: "error", Engine: engine, Op: "copy", Message: err.Error()})
		return
	}
	writeFrame(streamFrame{Type: "result", Engine: engine, Op: "copy", Result: res})
}

// hashProgressInterval spaces hashing progress frames.
var hashProgressInterval = 500 * time.Millisecond

// hashModelFiles fills in every missing sha256 in list, reporting progress per
// file at most every hashProgressInterval.
func hashModelFiles(ctx context.Context, list *modelFileList, progress func(path string, pct int)) error {
	for i := range list.Files {
		f := &list.Files[i]
		if f.SHA256 != "" {
			continue
		}
		var last time.Time
		sum, err := fileHashes.sum(ctx, f.abs, func(done int64) {
			if time.Since(last) < hashProgressInterval || f.Size <= 0 {
				return
			}
			last = time.Now()
			progress(f.Path, int(done*100/f.Size))
		})
		if err != nil {
			return fmt.Errorf("hash %s: %w", f.Path, err)
		}
		f.SHA256 = sum
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
	rel := r.URL.Query().Get("path")
	if !validModelRelPath(rel) {
		http.Error(w, "invalid path", http.StatusBadRequest)
		return
	}
	list, err := s.exec.modelSourceFiles(r.Context(), engine, model)
	if err != nil {
		http.Error(w, err.Error(), modelLookupStatus(err))
		return
	}
	var abs string
	for _, f := range list.Files {
		if f.Path == rel {
			abs = f.abs
			break
		}
	}
	if abs == "" {
		http.Error(w, "path is not a file of this model", http.StatusNotFound)
		return
	}
	file, err := os.Open(abs)
	if err != nil {
		http.Error(w, "file unavailable", http.StatusNotFound)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		http.Error(w, "file unavailable", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	http.ServeContent(w, r, "", info.ModTime(), file)
}
