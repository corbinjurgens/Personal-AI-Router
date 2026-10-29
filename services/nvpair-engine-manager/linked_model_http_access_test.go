// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"nvpair-shared/clustertrust"
)

func TestLinkedModelHTTPAccess(t *testing.T) {
	t.Run("custom external export", testLinkedModelHTTPAccessCustom)
	root := t.TempDir()
	ex := copyTestExecutor(t, "ollama", root, true)
	fixture := writeOllamaModel(t, filepath.Join(root, "models"), "tiny", 128, 3)
	copyTestNativeModels(t, ex, "ollama", "tiny")
	counts := &linkedHashCounts{}
	access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		counts.opens.Add(1)
		return &linkedHashFile{File: f, counts: counts}, nil
	}}
	f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
		mux := (&controlServer{exec: ex, mesh: mesh, access: access}).mux()
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if (r.Method != http.MethodGet && r.Method != http.MethodHead) || (r.URL.Path != controlModelFilesPath && r.URL.Path != controlModelFilePath) {
				http.NotFound(w, r)
				return
			}
			mux.ServeHTTP(w, r)
		})
	})
	ctx := remoteFixtureContext(t)
	client := f.client(t, ctx)
	list, err := client.modelFiles(ctx, "ollama", "tiny", nil)
	if err != nil {
		t.Fatal(err)
	}
	body, _, err := client.modelFile(ctx, "ollama", "tiny", fixture.manifestRel, list.ExportID, 0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := io.ReadAll(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, fixture.manifest) || counts.readAt.Load() == 0 || counts.opens.Load() != counts.closes.Load() {
		t.Fatalf("pinned response mismatch or reader not closed: opens=%d closes=%d reads=%d", counts.opens.Load(), counts.closes.Load(), counts.readAt.Load())
	}
	q := url.Values{"engine": {"ollama"}, "model": {"tiny"}, "path": {fixture.manifestRel}, "exportId": {list.ExportID}}
	before := counts.readAt.Load()
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, client.base+controlModelFilePath+"?"+q.Encode(), nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.controlClient(false).Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK || counts.readAt.Load() != before {
		t.Fatalf("HEAD status=%d body reads=%d", resp.StatusCode, counts.readAt.Load()-before)
	}
	body, offset, err := client.modelFile(ctx, "ollama", "tiny", fixture.manifestRel, list.ExportID, 2)
	if err != nil {
		t.Fatal(err)
	}
	got, err = io.ReadAll(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	if offset != 2 || !bytes.Equal(got, fixture.manifest[2:]) {
		t.Fatalf("range offset=%d bytes=%d", offset, len(got))
	}
	other := filepath.Join(root, "other")
	writeTestFile(t, other, []byte("forbidden"))
	reads := 0
	closed := &linkedHashCounts{}
	server := &controlServer{exec: ex, access: servingFileAccess{open: func(string) (servingArtifactFile, error) {
		f, err := os.Open(other)
		if err != nil {
			return nil, err
		}
		closed.opens.Add(1)
		return &linkedHashFile{File: f, counts: closed, forbidden: true}, nil
	}}}
	query := url.Values{"engine": {"ollama"}, "model": {"tiny"}, "path": {fixture.manifestRel}, "exportId": {list.ExportID}}
	response := httptest.NewRecorder()
	server.handleModelFile(response, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+query.Encode(), nil))
	if response.Code != http.StatusConflict || reads != 0 || closed.readAt.Load() != 0 || closed.opens.Load() != 1 || closed.closes.Load() != 1 {
		t.Fatalf("status=%d reads=%d opens=%d closes=%d", response.Code, closed.readAt.Load(), closed.opens.Load(), closed.closes.Load())
	}
	t.Run("cancellation retains lease until body completes", func(t *testing.T) {
		writer := &modelDownloadBodyWriter{ResponseRecorder: httptest.NewRecorder(), entered: make(chan struct{}), release: make(chan struct{})}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		request := httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+q.Encode(), nil).WithContext(ctx)
		done := make(chan struct{})
		go func() { defer close(done); (&controlServer{exec: ex, access: access}).handleModelFile(writer, request) }()
		<-writer.entered
		cancel()
		if err := ex.withModelMutation(context.Background(), "ollama", func() error { return nil }); err == nil {
			t.Error("mutation acquired lease while response was active")
		}
		close(writer.release)
		<-done
		release, err := ex.downloads.acquireWrite("ollama")
		if err != nil {
			t.Fatalf("lease retained after response: %v", err)
		}
		release()
		if counts.opens.Load() != counts.closes.Load() {
			t.Fatalf("opens=%d closes=%d", counts.opens.Load(), counts.closes.Load())
		}
	})
}

func testLinkedModelHTTPAccessCustom(t *testing.T) {
	f := newCustomCopyFixture(t)
	var mu sync.Mutex
	counts := map[string]*linkedHashCounts{}
	access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		mu.Lock()
		counter := counts[path]
		if counter == nil {
			counter = &linkedHashCounts{}
			counts[path] = counter
		}
		mu.Unlock()
		counter.opens.Add(1)
		return &linkedHashFile{File: file, counts: counter}, nil
	}}
	peer := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
		mux := (&controlServer{exec: f.source, mesh: mesh, access: access}).mux()
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if (r.Method != http.MethodGet && r.Method != http.MethodHead) || (r.URL.Path != controlModelFilesPath && r.URL.Path != controlModelFilePath) {
				http.NotFound(w, r)
				return
			}
			mux.ServeHTTP(w, r)
		})
	})
	ctx := remoteFixtureContext(t)
	client := peer.client(t, ctx)
	list, err := client.modelFiles(ctx, "llamacpp", reportedCopyAlias, nil)
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	preset, artifact := counts[f.preset], counts[f.artifact]
	mu.Unlock()
	if preset == nil || artifact == nil || preset.readAt.Load() == 0 || artifact.readAt.Load() == 0 {
		t.Fatal("listing did not read selector and artifact through injected reader")
	}
	body, offset, err := client.modelFile(ctx, "llamacpp", reportedCopyAlias, list.Files[0].Path, list.ExportID, 0)
	if err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(f.artifact)
	if err != nil {
		t.Fatal(err)
	}
	if offset != 0 || !bytes.Equal(data, want) {
		t.Fatal("custom GET returned unexpected bytes")
	}
	before := artifact.readAt.Load()
	body, offset, err = client.modelFile(ctx, "llamacpp", reportedCopyAlias, list.Files[0].Path, list.ExportID, 2)
	if err != nil {
		t.Fatal(err)
	}
	data, err = io.ReadAll(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	if offset != 2 || !bytes.Equal(data, want[2:]) || artifact.readAt.Load() <= before {
		t.Fatal("custom range did not use injected artifact reader")
	}
	q := url.Values{"engine": {"llamacpp"}, "model": {reportedCopyAlias}, "path": {list.Files[0].Path}, "exportId": {list.ExportID}}
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, client.base+controlModelFilePath+"?"+q.Encode(), nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.controlClient(false).Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("custom HEAD status=%d", resp.StatusCode)
	}
}
