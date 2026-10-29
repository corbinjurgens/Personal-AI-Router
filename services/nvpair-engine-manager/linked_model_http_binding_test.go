// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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

func TestLinkedModelHTTPBinding(t *testing.T) {
	t.Run("issued snapshot is detached", testLinkedModelHTTPBindingClone)
	t.Run("stale selector and artifact refuse before content reads", testLinkedModelHTTPBindingSubstitution)
	t.Run("issued membership and expiration", testLinkedModelHTTPBindingMembership)
	t.Run("concurrent relisting", testLinkedModelHTTPBindingRelisting)
	t.Run("hash drift prevents publication", testLinkedModelHTTPBindingHashDrift)
	t.Run("selected export through pinned routes", testLinkedModelHTTPBindingSelected)
}

func testLinkedModelHTTPBindingRelisting(t *testing.T) {
	root := t.TempDir()
	fx := writeOllamaModel(t, filepath.Join(root, "models"), "tiny", 128, 3)
	ex := copyTestExecutor(t, "ollama", root, true)
	copyTestNativeModels(t, ex, "ollama", "tiny")
	_, client := modelFileServer(t, ex, nil)
	ctx := context.Background()
	list, err := client.modelFiles(ctx, "ollama", "tiny", nil)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errors := make(chan error, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 4; i++ {
			if _, err := client.modelFiles(ctx, "ollama", "tiny", nil); err != nil {
				errors <- err
				return
			}
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 4; i++ {
			body, _, err := client.modelFile(ctx, "ollama", "tiny", fx.manifestRel, list.ExportID, 0)
			if err != nil {
				errors <- err
				return
			}
			got, err := io.ReadAll(body)
			closeErr := body.Close()
			if err != nil {
				errors <- err
				return
			}
			if closeErr != nil {
				errors <- closeErr
				return
			}
			if !bytes.Equal(got, fx.manifest) {
				errors <- fmt.Errorf("incorrect response during relisting")
				return
			}
		}
	}()
	wg.Wait()
	close(errors)
	for err := range errors {
		t.Fatal(err)
	}
}

func testLinkedModelHTTPBindingSelected(t *testing.T) {
	for _, kind := range []string{"preset", "typed"} {
		t.Run(kind, func(t *testing.T) {
			var ex *Executor
			var root, model string
			if kind == "preset" {
				f := newLlamaServingFixture(t)
				for i := range f.request.Settings.Serving.Models {
					item := &f.request.Settings.Serving.Models[i]
					data, err := os.ReadFile(filepath.Join(f.root, item.ModelID+".gguf"))
					if err != nil {
						t.Fatal(err)
					}
					item.Artifacts[0].SHA256 = sha256Hex(data)
				}
				applyLlamaServing(t, f, approveLlamaServing(t, f))
				settingsState(t, f.ex).modelsDir = f.root
				ex, root, model = f.ex, f.root, "medium"
			} else {
				f := newModelLoadFixture(t)
				f.st.plat.Runtime = convertModelLoadFixture(t, f)
				f.st.plat.Runtime.servingSnapshot = nil
				ex, root, model = f.ex, f.root, "medium-model"
			}
			forbidden := &linkedHashCounts{}
			access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
				file, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				return &linkedHashFile{File: file, counts: forbidden, forbidden: path == filepath.Join(root, "weak.gguf")}, nil
			}}
			peer := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
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
			client := peer.client(t, ctx)
			list, err := client.modelFiles(ctx, "fake", model, nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(list.Files) != 1 {
				t.Fatalf("files=%d", len(list.Files))
			}
			want, err := os.ReadFile(filepath.Join(root, "medium.gguf"))
			if err != nil {
				t.Fatal(err)
			}
			for _, offset := range []int64{0, 2} {
				body, gotOffset, err := client.modelFile(ctx, "fake", model, list.Files[0].Path, list.ExportID, offset)
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
				if gotOffset != offset || !bytes.Equal(got, want[offset:]) {
					t.Fatalf("offset=%d bytes=%d", gotOffset, len(got))
				}
			}
			query := url.Values{"engine": {"fake"}, "model": {model}, "path": {list.Files[0].Path}, "exportId": {list.ExportID}}
			req, err := http.NewRequestWithContext(ctx, http.MethodHead, client.base+controlModelFilePath+"?"+query.Encode(), nil)
			if err != nil {
				t.Fatal(err)
			}
			resp, err := client.controlClient(false).Do(req)
			if err != nil {
				t.Fatal(err)
			}
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			if err := resp.Body.Close(); err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != http.StatusOK || len(body) != 0 || forbidden.forbiddenBytes.Load() != 0 {
				t.Fatalf("HEAD=%d body=%d unrelated bytes=%d", resp.StatusCode, len(body), forbidden.forbiddenBytes.Load())
			}
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			st.plat.Runtime.Env["PAIR_TEST_CONFIG_DRIFT"] = "changed"
			req, err = http.NewRequestWithContext(ctx, http.MethodHead, client.base+controlModelFilePath+"?"+query.Encode(), nil)
			if err != nil {
				t.Fatal(err)
			}
			resp, err = client.controlClient(false).Do(req)
			if err != nil {
				t.Fatal(err)
			}
			if err := resp.Body.Close(); err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != http.StatusConflict || forbidden.forbiddenBytes.Load() != 0 {
				t.Fatalf("changed configuration status=%d unrelated bytes=%d", resp.StatusCode, forbidden.forbiddenBytes.Load())
			}
		})
	}
}

type driftResponseWriter struct {
	*httptest.ResponseRecorder
	mutate func()
}

func (w *driftResponseWriter) Write(data []byte) (int, error) {
	var frame streamFrame
	if json.Unmarshal(data, &frame) == nil && frame.Type == "progress" {
		w.mutate()
	}
	return w.ResponseRecorder.Write(data)
}

func testLinkedModelHTTPBindingHashDrift(t *testing.T) {
	root := t.TempDir()
	fx := writeOllamaModel(t, filepath.Join(root, "models"), "tiny", 128, 3)
	ex := copyTestExecutor(t, "ollama", root, true)
	copyTestNativeModels(t, ex, "ollama", "tiny")
	manifest := filepath.Join(root, "models", filepath.FromSlash(fx.manifestRel))
	var once sync.Once
	var mutationErr error
	server := &controlServer{exec: ex}
	response := &driftResponseWriter{ResponseRecorder: httptest.NewRecorder(), mutate: func() {
		once.Do(func() {
			mutationErr = os.Rename(manifest, manifest+".old")
			if mutationErr == nil {
				mutationErr = os.WriteFile(manifest, fx.manifest, 0600)
			}
		})
	}}
	server.handleModelFiles(response, httptest.NewRequest(http.MethodGet, controlModelFilesPath+"?engine=ollama&model=tiny", nil))
	if mutationErr != nil {
		t.Fatal(mutationErr)
	}
	if _, err := os.Stat(manifest + ".old"); err != nil {
		t.Fatalf("hash progress did not trigger drift: %v", err)
	}
	dec := json.NewDecoder(response.Body)
	var result, failure bool
	for {
		var frame streamFrame
		err := dec.Decode(&frame)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		result = result || frame.Type == "result"
		failure = failure || frame.Type == "error"
	}
	if result || !failure {
		t.Fatalf("drift published result=%v error=%v", result, failure)
	}
}

func testLinkedModelHTTPBindingMembership(t *testing.T) {
	root := t.TempDir()
	fx := writeOllamaModel(t, filepath.Join(root, "models"), "tiny", 128, 3)
	ex := copyTestExecutor(t, "ollama", root, true)
	copyTestNativeModels(t, ex, "ollama", "tiny")
	list := copyTestSourceListing(t, ex, "ollama", "tiny")
	server := &controlServer{exec: ex}
	request := func(engine, model, path, id string) int {
		query := url.Values{"engine": {engine}, "model": {model}, "path": {path}, "exportId": {id}}
		response := httptest.NewRecorder()
		server.handleModelFile(response, httptest.NewRequest(http.MethodHead, controlModelFilePath+"?"+query.Encode(), nil))
		return response.Code
	}
	for _, tc := range []struct {
		name, engine, model, path, id string
		want                          int
	}{
		{"valid", "ollama", "tiny", fx.manifestRel, list.ExportID, http.StatusOK},
		{"no listing", "ollama", "tiny", fx.manifestRel, "", http.StatusConflict},
		{"wrong id", "ollama", "tiny", fx.manifestRel, "wrong", http.StatusConflict},
		{"wrong engine", "missing", "tiny", fx.manifestRel, list.ExportID, http.StatusConflict},
		{"wrong model", "ollama", "other", fx.manifestRel, list.ExportID, http.StatusConflict},
		{"unlisted path", "ollama", "tiny", "unlisted", list.ExportID, http.StatusNotFound},
		{"traversal", "ollama", "tiny", "../secret", list.ExportID, http.StatusBadRequest},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := request(tc.engine, tc.model, tc.path, tc.id); got != tc.want {
				t.Fatalf("status=%d want=%d", got, tc.want)
			}
		})
	}
	st, err := ex.state("ollama")
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 256; i++ {
		rememberCopyExport(&engineState{}, modelFileList{Model: fmt.Sprintf("eviction-%d", i), ExportID: "id"})
	}
	if got := request("ollama", "tiny", fx.manifestRel, list.ExportID); got != http.StatusConflict {
		t.Fatalf("evicted status=%d", got)
	}
	list = copyTestSourceListing(t, ex, "ollama", "tiny")
	invalidateCopyGeneration(st, "tiny", issuedCopyGeneration(st, "tiny"))
	if got := request("ollama", "tiny", fx.manifestRel, list.ExportID); got != http.StatusConflict {
		t.Fatalf("expired status=%d", got)
	}
	refreshed := copyTestSourceListing(t, ex, "ollama", "tiny")
	if got := request("ollama", "tiny", fx.manifestRel, refreshed.ExportID); got != http.StatusOK {
		t.Fatalf("relisted status=%d", got)
	}
	restarted := copyTestExecutor(t, "ollama", root, true)
	copyTestNativeModels(t, restarted, "ollama", "tiny")
	server.exec = restarted
	if got := request("ollama", "tiny", fx.manifestRel, refreshed.ExportID); got != http.StatusConflict {
		t.Fatalf("restart status=%d", got)
	}
	refreshed = copyTestSourceListing(t, restarted, "ollama", "tiny")
	if got := request("ollama", "tiny", fx.manifestRel, refreshed.ExportID); got != http.StatusOK {
		t.Fatalf("restart relist status=%d", got)
	}
}

func testLinkedModelHTTPBindingClone(t *testing.T) {
	st := &engineState{}
	list := modelFileList{Model: "immutable", ExportID: "id", source: &sourceSnapshot{
		recognition: []byte("original"), selectors: []selectorEvidence{{bytes: []byte("selector")}},
		inventory: &engineInventory{candidates: []string{"model"}},
	}}
	rememberCopyExport(st, list)
	list.source.recognition[0] = 'X'
	list.source.selectors[0].bytes[0] = 'X'
	list.source.inventory.candidates[0] = "changed"
	got, ok := issuedCopyList(st, list.Model, list.ExportID)
	if !ok || string(got.source.recognition) != "original" || string(got.source.selectors[0].bytes) != "selector" || got.source.inventory.candidates[0] != "model" {
		t.Fatal("issued snapshot changed with caller-owned data")
	}
	got.source.recognition[0] = 'X'
	again, _ := issuedCopyList(st, list.Model, list.ExportID)
	if string(again.source.recognition) != "original" {
		t.Fatal("lookup exposed issued snapshot to mutation")
	}
}

func testLinkedModelHTTPBindingSubstitution(t *testing.T) {
	for _, target := range []string{"selector", "artifact", "directory", "root"} {
		t.Run(target, func(t *testing.T) {
			root := t.TempDir()
			fx := writeOllamaModel(t, filepath.Join(root, "models"), "tiny", 128, 3)
			ex := copyTestExecutor(t, "ollama", root, true)
			copyTestNativeModels(t, ex, "ollama", "tiny")
			_, client := modelFileServer(t, ex, nil)
			list, err := client.modelFiles(context.Background(), "ollama", "tiny", nil)
			if err != nil {
				t.Fatal(err)
			}
			rel := fx.manifestRel
			if target == "artifact" {
				for path := range fx.blobs {
					rel = path
					break
				}
			}
			path := filepath.Join(root, "models", filepath.FromSlash(rel))
			if target == "directory" {
				path = filepath.Join(root, "models", "blobs")
			} else if target == "root" {
				path = filepath.Join(root, "models")
			}
			if err := os.Rename(path, path+".old"); err != nil {
				t.Fatal(err)
			}
			if target == "directory" || target == "root" {
				err = os.Mkdir(path, 0700)
			} else {
				err = os.WriteFile(path, []byte("replacement"), 0600)
			}
			if err != nil {
				t.Fatal(err)
			}
			counts := &linkedHashCounts{}
			server := &controlServer{exec: ex, access: servingFileAccess{open: func(path string) (servingArtifactFile, error) {
				file, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				counts.opens.Add(1)
				return &linkedHashFile{File: file, counts: counts, forbidden: true}, nil
			}}}
			query := url.Values{"engine": {"ollama"}, "model": {"tiny"}, "path": {fx.manifestRel}, "exportId": {list.ExportID}}
			response := httptest.NewRecorder()
			server.handleModelFile(response, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+query.Encode(), nil))
			if response.Code != http.StatusConflict || counts.bytes.Load() != 0 {
				t.Fatalf("status=%d forbidden bytes=%d", response.Code, counts.bytes.Load())
			}
		})
	}
}
