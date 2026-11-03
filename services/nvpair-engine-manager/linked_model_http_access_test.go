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
	"sync"
	"testing"

	"nvpair-shared/clustertrust"
)

func TestLinkedModelHTTPAccess(t *testing.T) {
	t.Run("custom external export", testLinkedModelHTTPAccessCustom)
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
	t.Run("cancellation retains lease until body completes", func(t *testing.T) {
		ex := f.source
		writer := &modelDownloadBodyWriter{ResponseRecorder: httptest.NewRecorder(), entered: make(chan struct{}), release: make(chan struct{})}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		request := httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+q.Encode(), nil).WithContext(ctx)
		done := make(chan struct{})
		go func() { defer close(done); (&controlServer{exec: ex, access: access}).handleModelFile(writer, request) }()
		<-writer.entered
		cancel()
		if err := ex.withModelMutation(context.Background(), "llamacpp", func() error { return nil }); err == nil {
			t.Error("mutation acquired lease while response was active")
		}
		close(writer.release)
		<-done
		release, err := ex.downloads.acquireWrite("llamacpp")
		if err != nil {
			t.Fatalf("lease retained after response: %v", err)
		}
		release()
	})
}
