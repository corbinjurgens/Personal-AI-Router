// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// These observations have no admission, export-identity or serving authority.
// A peer's record selects a provider tuple; only this destination's catalogue
// owner may acquire the statement used to compare its fully checked bytes.
type modelPublisherFile struct {
	Path             string `json:"path"`
	Status           string `json:"status"`
	Provider         string `json:"provider,omitempty"`
	Repository       string `json:"repository,omitempty"`
	Revision         string `json:"revision,omitempty"`
	File             string `json:"file,omitempty"`
	Algorithm        string `json:"algorithm"`
	ExpectedDigest   string `json:"expectedDigest,omitempty"`
	ActualDigest     string `json:"actualDigest"`
	AcquisitionBasis string `json:"acquisitionBasis"`
	AcquiredAt       string `json:"acquiredAt,omitempty"`
}

type modelPublisherComparison struct {
	Status   string               `json:"status"`
	Files    []modelPublisherFile `json:"files"`
	Replayed bool                 `json:"replayed,omitempty"`
}

func (f *modelFile) UnmarshalJSON(data []byte) error {
	type file modelFile
	var value file
	if err := decodePublisherObject(data, &value, "path", "role", "size", "sha256", "publisher"); err != nil {
		return err
	}
	*f = modelFile(value)
	if f.Publisher != nil && (f.Publisher.Path != f.Path || f.Publisher.ActualDigest != f.SHA256) {
		return fmt.Errorf("publisher comparison does not match its file")
	}
	return nil
}

func publisherTupleValid(p modelPublisherFile) bool {
	return p.Provider == "huggingface" && len(p.Repository) <= 193 &&
		llamaCPPRepoPattern.MatchString(p.Repository) && !strings.Contains(p.Repository, "..") &&
		lowerHex(p.Revision, 40) && validPublisherPath(p.File)
}

func validPublisherPath(path string) bool {
	return utf8.ValidString(path) && validModelRelPath(path) && !strings.ContainsRune(path, '\x7f')
}

func (p modelPublisherFile) valid() bool {
	if !validPublisherPath(p.Path) || p.Algorithm != "sha256" || !validSHA256(p.ActualDigest) {
		return false
	}
	hasTuple := p.Provider != "" || p.Repository != "" || p.Revision != "" || p.File != ""
	if hasTuple && !publisherTupleValid(p) {
		return false
	}
	if p.Status == "unavailable" {
		return p.ExpectedDigest == "" && p.AcquiredAt == "" && p.AcquisitionBasis == "unavailable"
	}
	if !hasTuple || !validSHA256(p.ExpectedDigest) || (p.AcquisitionBasis != "provider" && p.AcquisitionBasis != "cache") || len(p.AcquiredAt) > 30 {
		return false
	}
	if len(p.AcquiredAt) != 20 && (len(p.AcquiredAt) < 22 || p.AcquiredAt[19] != '.') {
		return false
	}
	at, err := time.Parse(time.RFC3339Nano, p.AcquiredAt)
	if err != nil || !strings.HasSuffix(p.AcquiredAt, "Z") || at.Year() < 2000 || at.After(time.Now().Add(5*time.Minute)) {
		return false
	}
	return p.Status == "verified" && p.ExpectedDigest == p.ActualDigest || p.Status == "mismatch" && p.ExpectedDigest != p.ActualDigest
}

// Validate exact keys, uniqueness and Unicode before Go's permissive decoder
// can fold names, replace lone surrogates or hide duplicate observations.
func decodePublisherObject(data []byte, target any, allowed ...string) error {
	if len(data) > maxCatalogBody || !utf8.Valid(data) {
		return fmt.Errorf("invalid publisher comparison")
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	if token, err := dec.Token(); err != nil || token != json.Delim('{') {
		return fmt.Errorf("invalid publisher comparison object")
	}
	seen := map[string]bool{}
	for dec.More() {
		token, err := dec.Token()
		key, ok := token.(string)
		known := false
		for _, name := range allowed {
			known = known || key == name
		}
		if err != nil || !ok || !known || seen[key] {
			return fmt.Errorf("unknown or duplicate publisher comparison field")
		}
		seen[key] = true
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return fmt.Errorf("invalid publisher comparison field")
		}
		if len(raw) > 0 && raw[0] == '"' && !copyStringUnicode(raw) {
			return fmt.Errorf("invalid publisher comparison Unicode")
		}
	}
	if _, err := dec.Token(); err != nil || dec.Decode(new(json.RawMessage)) != io.EOF {
		return fmt.Errorf("invalid publisher comparison trailer")
	}
	return json.Unmarshal(data, target)
}

func (p *modelPublisherFile) UnmarshalJSON(data []byte) error {
	type record modelPublisherFile
	var value record
	if err := decodePublisherObject(data, &value, "path", "status", "provider", "repository", "revision", "file", "algorithm", "expectedDigest", "actualDigest", "acquisitionBasis", "acquiredAt"); err != nil {
		return err
	}
	*p = modelPublisherFile(value)
	for _, name := range []string{"provider", "repository", "revision", "file", "expectedDigest", "acquiredAt"} {
		if raw, present := downloadIdentityField(data, name); present {
			var text string
			if json.Unmarshal(raw, &text) != nil || text == "" {
				return fmt.Errorf("empty publisher comparison field")
			}
		}
	}
	if !p.valid() {
		return fmt.Errorf("invalid publisher file comparison")
	}
	return nil
}

func publisherAggregate(files []modelPublisherFile) string {
	status := "verified"
	if len(files) == 0 {
		return "unavailable"
	}
	for _, f := range files {
		if f.Status == "mismatch" {
			return "mismatch"
		}
		if f.Status != "verified" {
			status = "unavailable"
		}
	}
	return status
}

func (p *modelPublisherComparison) UnmarshalJSON(data []byte) error {
	type comparison modelPublisherComparison
	var value comparison
	if err := decodePublisherObject(data, &value, "status", "files", "replayed"); err != nil {
		return err
	}
	*p = modelPublisherComparison(value)
	return validatePublisherComparison(p, len(p.Files))
}

func validatePublisherComparison(p *modelPublisherComparison, files int) error {
	if p == nil {
		return nil
	}
	if len(p.Files) == 0 || len(p.Files) != files || len(p.Files) > maxCopyArtifacts || p.Status != publisherAggregate(p.Files) {
		return fmt.Errorf("incomplete publisher comparison")
	}
	seen := map[string]bool{}
	for _, f := range p.Files {
		key := copyPathKey(f.Path)
		if !f.valid() || seen[key] {
			return fmt.Errorf("invalid publisher comparison file")
		}
		seen[key] = true
	}
	return nil
}

// Only the native HF snapshot resolver provides this mapping. Custom aliases,
// mutable refs, names, URLs and local LFS blob filenames provide no mapping.
func publisherSelector(list modelFileList, f modelFile) *modelPublisherFile {
	repo, _, err := splitLlamaModel(list.Model)
	p := modelPublisherFile{Path: f.Path, Status: "unavailable", Algorithm: "sha256", ActualDigest: f.SHA256, AcquisitionBasis: "unavailable"}
	if err == nil && list.Layout == "" && lowerHex(list.Revision, 40) {
		p.Provider, p.Repository, p.Revision, p.File = "huggingface", repo, list.Revision, f.Path
		if !publisherTupleValid(p) {
			return nil
		}
		return &p
	}
	return nil
}

func publisherResult(list modelFileList, replayed bool) *modelPublisherComparison {
	p := &modelPublisherComparison{Replayed: replayed}
	for _, f := range list.Files {
		value := modelPublisherFile{Path: f.Path, Status: "unavailable", Algorithm: "sha256", ActualDigest: f.SHA256, AcquisitionBasis: "unavailable"}
		if f.Publisher != nil {
			value = *f.Publisher
		}
		p.Files = append(p.Files, value)
	}
	p.Status = publisherAggregate(p.Files)
	return p
}

// Overwrite every peer observation before it can enter a destination receipt.
// The supplied catalogue is the destination Manager's owner. Nil is an
// explicitly unavailable boundary for executor-only, network-free callers.
func compareModelPublisher(ctx context.Context, list *modelFileList, adapter string, catalog *catalogService) {
	for i := range list.Files {
		f := &list.Files[i]
		peer := f.Publisher
		local := modelPublisherFile{Path: f.Path, Status: "unavailable", Algorithm: "sha256", ActualDigest: f.SHA256, AcquisitionBasis: "unavailable"}
		selector := publisherSelector(*list, *f)
		if adapter == "llamacpp" && peer != nil && selector != nil && peer.Provider == selector.Provider && peer.Repository == selector.Repository && peer.Revision == selector.Revision && peer.File == selector.File && peer.Path == f.Path {
			local = *selector
			if catalog != nil {
				expected, at, basis, ok := catalog.publisherDigest(ctx, local)
				if ok {
					local.ExpectedDigest, local.AcquiredAt, local.AcquisitionBasis = expected, at, basis
					local.Status = "mismatch"
					if expected == local.ActualDigest {
						local.Status = "verified"
					}
				}
			}
		}
		f.Publisher = nil
		if local.Provider != "" {
			f.Publisher = &local
		}
	}
}

type publisherRevisionCache struct {
	mu       sync.Mutex
	digests  map[string]string
	fetched  time.Time
	failed   time.Time
	inflight chan struct{}
}

func (c *catalogService) publisherRevision(key string) *publisherRevisionCache {
	c.searchMu.Lock()
	defer c.searchMu.Unlock()
	if c.publisherRevisions == nil {
		c.publisherRevisions = map[string]*publisherRevisionCache{}
	}
	entry := c.publisherRevisions[key]
	if entry == nil {
		entry = &publisherRevisionCache{}
		c.publisherRevisions[key] = entry
	}
	for i, old := range c.publisherOrder {
		if old == key {
			c.publisherOrder = append(c.publisherOrder[:i], c.publisherOrder[i+1:]...)
			break
		}
	}
	c.publisherOrder = append(c.publisherOrder, key)
	for len(c.publisherOrder) > llamaCPPSearchCacheLimit {
		delete(c.publisherRevisions, c.publisherOrder[0])
		c.publisherOrder = c.publisherOrder[1:]
	}
	return entry
}

func (c *catalogService) publisherDigest(ctx context.Context, p modelPublisherFile) (string, string, string, bool) {
	if !publisherTupleValid(p) {
		return "", "", "", false
	}
	entry := c.publisherRevision(p.Repository + "@" + p.Revision)
	for {
		if ctx.Err() != nil {
			return "", "", "", false
		}
		entry.mu.Lock()
		if !entry.fetched.IsZero() && time.Since(entry.fetched) < catalogCacheTTL {
			digest, at := entry.digests[p.File], entry.fetched.UTC().Format(time.RFC3339Nano)
			entry.mu.Unlock()
			return digest, at, "cache", validSHA256(digest)
		}
		if !entry.failed.IsZero() && time.Since(entry.failed) < catalogRetryAfterFailure {
			entry.mu.Unlock()
			return "", "", "", false
		}
		if entry.inflight != nil {
			done := entry.inflight
			entry.mu.Unlock()
			select {
			case <-ctx.Done():
				return "", "", "", false
			case <-done:
				continue
			}
		}
		entry.inflight = make(chan struct{})
		entry.mu.Unlock()
		digests, err := c.fetchPublisherRevision(ctx, p)
		entry.mu.Lock()
		if err == nil {
			entry.digests, entry.fetched, entry.failed = digests, time.Now(), time.Time{}
		} else {
			entry.failed = time.Now()
		}
		digest, at := entry.digests[p.File], entry.fetched.UTC().Format(time.RFC3339Nano)
		close(entry.inflight)
		entry.inflight = nil
		entry.mu.Unlock()
		if err != nil || !validSHA256(digest) {
			return "", "", "", false
		}
		return digest, at, "provider", true
	}
}

func (c *catalogService) fetchPublisherRevision(ctx context.Context, p modelPublisherFile) (map[string]string, error) {
	// All path components have already been validated; the endpoint is never
	// taken from peer metadata. No tokens, credentials or arbitrary resolve URL.
	body, err := c.fetchHFBody(ctx, c.endpoint()+"/"+p.Repository+"/revision/"+p.Revision+"?blobs=true")
	if err != nil {
		return nil, err
	}
	if !utf8.Valid(body) {
		return nil, fmt.Errorf("invalid publisher metadata encoding")
	}
	for _, name := range []string{"id", "sha", "private", "gated", "siblings"} {
		raw, unique := downloadIdentityField(body, name)
		if !unique || len(raw) == 0 || raw[0] == '"' && !copyStringUnicode(raw) {
			return nil, fmt.Errorf("ambiguous publisher revision")
		}
	}
	var row struct {
		ID       string            `json:"id"`
		SHA      string            `json:"sha"`
		Private  *bool             `json:"private"`
		Gated    json.RawMessage   `json:"gated"`
		Siblings []json.RawMessage `json:"siblings"`
	}
	if err := json.Unmarshal(body, &row); err != nil || row.ID != p.Repository || row.SHA != p.Revision || row.Private == nil || *row.Private || string(bytes.TrimSpace(row.Gated)) != "false" || len(row.Siblings) == 0 || len(row.Siblings) > maxCopyArtifacts {
		return nil, fmt.Errorf("publisher revision is unavailable")
	}
	digests := map[string]string{}
	for _, raw := range row.Siblings {
		name, unique := downloadIdentityField(raw, "rfilename")
		var file string
		if !unique || !copyStringUnicode(name) || json.Unmarshal(name, &file) != nil || !validPublisherPath(file) {
			return nil, fmt.Errorf("invalid publisher file")
		}
		if _, duplicate := digests[file]; duplicate {
			return nil, fmt.Errorf("duplicate publisher file")
		}
		digest := ""
		var sibling struct {
			LFS json.RawMessage `json:"lfs"`
		}
		if json.Unmarshal(raw, &sibling) != nil {
			return nil, fmt.Errorf("invalid publisher file metadata")
		}
		if len(sibling.LFS) > 0 && !bytes.Equal(bytes.TrimSpace(sibling.LFS), []byte("null")) {
			lfs, unique := downloadIdentityField(raw, "lfs")
			sha, exact := downloadIdentityField(lfs, "sha256")
			if !unique || !exact || !copyStringUnicode(sha) || json.Unmarshal(sha, &digest) != nil || !validSHA256(digest) {
				return nil, fmt.Errorf("invalid publisher digest")
			}
		}
		digests[file] = digest
	}
	return digests, nil
}
