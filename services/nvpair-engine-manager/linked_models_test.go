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
		ex := copyTestExecutor(t, "llamacpp", root, true)
		fixture := writeLlamaNativeModel(t, root, 128, 3)
		st, err := ex.state("llamacpp")
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
			return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"models":[{"name":"` + fixture.model + `"}]}`))}, nil
		})}
		raw, err := ex.Action(context.Background(), "llamacpp", "list_models", nil)
		if err != nil {
			t.Fatalf("fake native recognition: %v", err)
		}
		models, ok := extractReadyStringsResult(raw, spec, st.manifest)
		if !ok || len(models) != 1 || models[0] != fixture.model {
			t.Fatalf("fake native membership = %v, valid=%v", models, ok)
		}
		query := url.Values{"engine": {"llamacpp"}, "model": {fixture.model}, "path": {fixture.rel}}
		response := httptest.NewRecorder()
		(&controlServer{exec: ex}).handleModelFile(response, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+query.Encode(), nil))
		if response.Code != http.StatusConflict {
			t.Fatalf("unissued native file status = %d, want %d", response.Code, http.StatusConflict)
		}
		if bytes.Equal(response.Body.Bytes(), fixture.data) {
			t.Fatal("unissued native file response contains model bytes")
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

type linkedReadCounter struct {
	*os.File
	reads *int
}

func (f *linkedReadCounter) ReadAt(p []byte, offset int64) (int, error) {
	*f.reads++
	return f.File.ReadAt(p, offset)
}
