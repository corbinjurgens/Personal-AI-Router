// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLinkedModelPeerBoundary(t *testing.T) {
	t.Run("native file request without a listing is refused", func(t *testing.T) {
		root := t.TempDir()
		ex := copyTestExecutor(t, "ollama", root, true)
		st, err := ex.state("ollama")
		if err != nil {
			t.Fatal(err)
		}
		st.running, st.port = true, 12345 // Synthetic state; the transport below owns every request.
		spec := &ActionResult{Array: "models", Field: "name"}
		st.manifest.Actions = map[string]Action{"list_models": {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/api/tags"}, Result: spec}}
		ex.client = &http.Client{Transport: memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
			if r.Method != http.MethodGet || r.URL.Path != "/api/tags" {
				return nil, fmt.Errorf("unexpected native request")
			}
			return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"models":[{"name":"tiny"}]}`))}, nil
		})}
		raw, err := ex.Action(context.Background(), "ollama", "list_models", nil)
		if err != nil {
			t.Fatalf("fake native recognition: %v", err)
		}
		models, ok := extractReadyStringsResult(raw, spec, st.manifest)
		if !ok || len(models) != 1 || models[0] != "tiny" {
			t.Fatalf("fake native membership = %v, valid=%v", models, ok)
		}
		fixture := writeOllamaModel(t, filepath.Join(root, "models"), "tiny", 128, 3)
		query := url.Values{"engine": {"ollama"}, "model": {"tiny"}, "path": {fixture.manifestRel}}
		response := httptest.NewRecorder()
		(&controlServer{exec: ex}).handleModelFile(response, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+query.Encode(), nil))
		if response.Code != http.StatusConflict {
			t.Fatalf("unissued native file status = %d, want %d", response.Code, http.StatusConflict)
		}
		if bytes.Equal(response.Body.Bytes(), fixture.manifest) {
			t.Fatal("unissued native file response contains manifest bytes")
		}
	})
}

func TestLinkedModelReadBinding(t *testing.T) {
	t.Run("small source rejects a substituted reopened object before reading", func(t *testing.T) {
		root := t.TempDir()
		path := filepath.Join(root, "selected.ini")
		other := filepath.Join(root, "unselected.ini")
		writeTestFile(t, path, []byte("selected source"))
		writeTestFile(t, other, []byte("unselected file"))
		var opens int
		var forbiddenReads int
		access := servingFileAccess{open: func(name string) (servingArtifactFile, error) {
			opens++
			if opens == 2 {
				f, err := os.Open(other)
				if err != nil {
					return nil, err
				}
				return &linkedReadCounter{File: f, reads: &forbiddenReads}, nil
			}
			return os.Open(name)
		}}
		if _, _, err := access.source(context.Background(), path); err == nil {
			t.Fatal("substituted source was accepted")
		}
		if forbiddenReads != 0 {
			t.Fatalf("forbidden source reads = %d, want 0", forbiddenReads)
		}
		if opens != 2 {
			t.Fatalf("opened %d times, want the substituted second open", opens)
		}
	})
}

func TestLinkedModelResolution(t *testing.T) {
	t.Run("exact Ollama blob may link outside the store", func(t *testing.T) {
		root := t.TempDir()
		fixture := writeOllamaModel(t, root, "tiny", 128, 3)
		for rel, data := range fixture.blobs {
			target := filepath.Join(t.TempDir(), "blob")
			writeTestFile(t, target, data)
			path := filepath.Join(root, filepath.FromSlash(rel))
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(target, path); err != nil {
				t.Skipf("symlink unavailable: %v", err)
			}
		}
		list, err := ollamaSourceFiles(root, "tiny")
		if err != nil {
			t.Fatal(err)
		}
		if len(list.Files) != len(fixture.blobs)+1 {
			t.Fatalf("unexpected selected files: %+v", list.Files)
		}
	})
}

type linkedReadCounter struct {
	*os.File
	reads *int
}

func (f *linkedReadCounter) ReadAt(p []byte, offset int64) (int, error) {
	*f.reads++
	return f.File.ReadAt(p, offset)
}
