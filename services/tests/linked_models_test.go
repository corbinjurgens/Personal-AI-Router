// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/modelconfig"
	"nvpair-shared/modeldownload"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func TestLinkedModelsProcess(t *testing.T) {
	t.Run("linked custom receipt requires fresh target review", func(t *testing.T) {
		modelSettingsCopyThenFreshReceiptInspection(t, func(t *testing.T, model controllerCopyModel) {
			for rel, path := range model.paths {
				target := filepath.Join(t.TempDir(), filepath.Base(path))
				modelSettingsWrite(t, target, model.data[rel])
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(target, path); err != nil {
					t.Skipf("symlink unavailable: %v", err)
				}
			}
		})
	})
	for _, tc := range []struct {
		name, alias string
		reverse     bool
	}{
		{"custom direct", "linked-custom-direct", false},
		{"custom reverse", "linked-custom-reverse", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newControllerCopyFixture(t)
			linkedTargets := t.TempDir()
			f.directoryToManagers(t, tc.reverse)
			linkedWatchSentinel(t, f, []byte("unlisted sentinel"))
			model := f.seed(t, tc.alias, 7)
			var sentinel []byte
			for i := range model.manifest.Files {
				artifact := &model.manifest.Files[i]
				data := modelSettingsGGUF("blk." + string(rune('0'+i)) + ".weight")
				path := model.paths[artifact.Path]
				target := filepath.Join(linkedTargets, tc.alias, strconv.Itoa(i), filepath.Base(path))
				modelSettingsWrite(t, target, data)
				if i == 0 {
					sentinel = []byte("unlisted sentinel")
					modelSettingsWrite(t, filepath.Join(filepath.Dir(target), "unlisted.gguf"), sentinel)
				}
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
				if tc.reverse && i == 1 {
					if err := os.Link(target, path); err != nil {
						t.Skipf("hardlink unavailable: %v", err)
					}
				} else if err := os.Symlink(target, path); err != nil {
					t.Skipf("symlink unavailable: %v", err)
				}
				model.data[artifact.Path] = data
				artifact.Size, artifact.SHA256 = int64(len(data)), phase04SHA(data)
			}
			model.manifest.ExportID = phase04SHA(mustMarshal(t, struct {
				Version   int                      `json:"version"`
				Model     string                   `json:"model"`
				Artifacts []controllerCopyArtifact `json:"artifacts"`
			}{1, model.manifest.Model, model.manifest.Files}))
			response := f.sourceEM.call(t, "engine:inspect-model", map[string]any{"request": map[string]string{"engine": "llamacpp", "modelId": tc.alias}})
			if response.Error != nil {
				t.Fatalf("inspection failed: %+v", response.Error)
			}
			var inspect struct {
				Status    string            `json:"status"`
				Artifacts []json.RawMessage `json:"artifacts"`
			}
			task2Decode(t, response.Result, &inspect)
			if inspect.Status != "resolved" || len(inspect.Artifacts) != len(model.data) || bytes.Contains(response.Result, []byte(f.source.base)) {
				t.Fatalf("inspection=%s", response.Result)
			}
			reader := downloadTransferReader(t, f.target, f.source)
			port := f.files[f.source.uuid].port
			if tc.reverse {
				port = f.blocked
				address := "127.0.0.1:" + strconv.Itoa(port)
				adapter := clustertrust.NewPeerServiceClient(f.target.mesh, clustertrust.PeerServiceClientOptions{
					BaseConfigDir: f.target.base,
					ResolveRoute: func(peer string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
						return clustertrust.PeerServiceRoute{Incarnation: "fixture", Address: address, Protocol: clustertrust.ReverseTLS, Generation: 1}, peer == f.source.uuid && service == noderec.ServiceEngineControl
					},
					DirectTimeout: time.Second,
				})
				reader = adapter.Client(f.source.uuid, noderec.ServiceEngineControl)
				t.Cleanup(reader.CloseIdleConnections)
			}
			base := "https://127.0.0.1:" + strconv.Itoa(port)
			query := url.Values{"engine": {"llamacpp"}, "model": {tc.alias}, "path": {model.manifest.Files[0].Path}}
			status, body := linkedProcessResponse(t, reader, base, http.MethodGet, "/v1/models/file", query, "")
			if status != http.StatusConflict || bytes.Contains(body, sentinel) {
				t.Fatalf("unissued GET status=%d", status)
			}
			for _, method := range []string{http.MethodGet, http.MethodHead} {
				status, body = linkedProcessResponse(t, reader, base, method, "/v1/models/file", query, "bytes=0-7")
				if status != http.StatusConflict || bytes.Contains(body, sentinel) {
					t.Fatalf("unissued %s range status=%d", method, status)
				}
			}
			linkedProcessListing(t, reader, base, model)
			query.Set("exportId", model.manifest.ExportID)
			status, body = linkedProcessResponse(t, reader, base, http.MethodHead, "/v1/models/file", query, "")
			if status != http.StatusOK || len(body) != 0 {
				t.Fatalf("HEAD status=%d bytes=%d", status, len(body))
			}
			status, body = linkedProcessResponse(t, reader, base, http.MethodGet, "/v1/models/file", query, "bytes=2-")
			if status != http.StatusPartialContent || !bytes.Equal(body, model.data[model.manifest.Files[0].Path][2:]) {
				t.Fatalf("range status=%d bytes=%d", status, len(body))
			}
			outsideSibling := filepath.Join(filepath.Dir(mustLinkedTarget(t, model.paths[model.manifest.Files[0].Path])), "unlisted.gguf")
			for _, refusal := range []struct {
				path   string
				status int
			}{
				{"model/unlisted.gguf", http.StatusNotFound},
				{"../unlisted.gguf", http.StatusBadRequest},
				{"/unlisted.gguf", http.StatusBadRequest},
				{outsideSibling, http.StatusBadRequest},
			} {
				query.Set("path", refusal.path)
				for _, method := range []string{http.MethodGet, http.MethodHead} {
					for _, rangeHeader := range []string{"", "bytes=0-7"} {
						status, body = linkedProcessResponse(t, reader, base, method, "/v1/models/file", query, rangeHeader)
						if status != refusal.status || bytes.Contains(body, sentinel) || bytes.Contains(body, []byte(f.source.base)) {
							t.Fatalf("unlisted %q %s status=%d, want %d", refusal.path, method, status, refusal.status)
						}
					}
				}
			}
			for _, changed := range []url.Values{
				{"engine": {"llamacpp"}, "model": {tc.alias}, "path": {model.manifest.Files[0].Path}, "exportId": {"wrong"}},
				{"engine": {"ollama"}, "model": {tc.alias}, "path": {model.manifest.Files[0].Path}, "exportId": {model.manifest.ExportID}},
				{"engine": {"llamacpp"}, "model": {"not-listed"}, "path": {model.manifest.Files[0].Path}, "exportId": {model.manifest.ExportID}},
			} {
				for _, method := range []string{http.MethodGet, http.MethodHead} {
					for _, rangeHeader := range []string{"", "bytes=0-7"} {
						status, body = linkedProcessResponse(t, reader, base, method, "/v1/models/file", changed, rangeHeader)
						if status != http.StatusConflict || bytes.Contains(body, sentinel) || bytes.Contains(body, []byte(f.source.base)) {
							t.Fatalf("invalid identity %s status=%d", method, status)
						}
					}
				}
			}
			for _, selector := range []string{"not-listed", outsideSibling, filepath.Dir(outsideSibling), "../unrecognized-loose.gguf"} {
				status, body = linkedProcessResponse(t, reader, base, http.MethodGet, "/v1/models/files", url.Values{"engine": {"llamacpp"}, "model": {selector}}, "")
				// These are unconfigured custom selectors, not issued file paths.
				// LIST reports the existing bounded source-unavailable response.
				if status != http.StatusInternalServerError || !bytes.Equal(body, []byte("model source unavailable\n")) || bytes.Contains(body, sentinel) || bytes.Contains(body, []byte(f.source.base)) {
					t.Fatalf("unrecognized LIST status=%d", status)
				}
			}
			wrongEngine := f.request(tc.alias, "")
			wrongEngine.Engine = "ollama"
			for _, bad := range []controllerCopyRequest{f.request(tc.alias, "wrong"), f.request("not-listed", ""), wrongEngine, f.request(filepath.Dir(outsideSibling), "")} {
				id := f.broker.send(t, "engine:copy-model", bad, true)
				refused := controllerCopyResponse(t, f.broker, id)
				if refused.Error == nil || bytes.Contains(mustMarshal(t, refused.Error), []byte(f.source.base)) {
					t.Fatalf("invalid COPY selection result=%+v", refused.Error)
				}
				f.assertNoReceipt(t, model)
				f.idle(t)
			}
			request := f.request(tc.alias, "")
			id := f.broker.send(t, "engine:copy-model", request, true)
			f.assertCopied(t, model, task2Response(t, f.broker, id))
			controllerCopyTerminal(t, f.broker, request, "done")
			f.idle(t)
			linkedExactDestination(t, filepath.Join(controllerCopyAliasDir(f.stores[f.target.uuid], model.manifest.Model), model.manifest.ExportID), model.data)
			if got, err := os.ReadFile(filepath.Join(filepath.Dir(mustLinkedTarget(t, model.paths[model.manifest.Files[0].Path])), "unlisted.gguf")); err != nil || !bytes.Equal(got, sentinel) {
				t.Fatalf("sentinel changed: %v", err)
			}
			path := model.paths[model.manifest.Files[0].Path]
			replacement := filepath.Join(linkedTargets, tc.alias, "replacement.gguf")
			modelSettingsWrite(t, replacement, model.data[model.manifest.Files[0].Path])
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(replacement, path); err != nil {
				t.Fatal(err)
			}
			query.Set("path", model.manifest.Files[0].Path)
			status, body = linkedProcessResponse(t, reader, base, http.MethodGet, "/v1/models/file", query, "")
			if status != http.StatusConflict || bytes.Contains(body, model.data[model.manifest.Files[0].Path]) {
				t.Fatalf("retarget old ID status=%d", status)
			}
			linkedProcessListing(t, reader, base, model)
			if !tc.reverse {
				linkedRestartManagers(t, f, nil)
				query.Set("path", model.manifest.Files[0].Path)
				status, body = linkedProcessResponse(t, reader, base, http.MethodGet, "/v1/models/file", query, "")
				if status != http.StatusConflict || bytes.Contains(body, sentinel) {
					t.Fatalf("restart old ID status=%d", status)
				}
				linkedProcessListing(t, reader, base, model)
			}
			f.assertController(t)
		})
	}
	t.Run("selected typed export without Runtime Serving ignores unrelated drift", func(t *testing.T) {
		f := newControllerCopyFixture(t)
		model := linkedSyntheticModel(t, f, "typed-selected", true)
		other := linkedSyntheticModel(t, f, "typed-unrelated", false)
		linkedRestartManagers(t, f, nil)
		linkedConfigureSelected(t, f, model.manifest.Model)
		for _, path := range other.paths {
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
		}
		id := f.broker.send(t, "engine:copy-model", f.request(model.manifest.Model, ""), true)
		f.assertCopied(t, model, controllerCopyResponse(t, f.broker, id))
		f.idle(t)
		f.assertController(t)
	})
	t.Run("completed receipt retry with source offline", func(t *testing.T) {
		f := newControllerCopyFixture(t)
		completed := f.seedSized(t, "linked-offline", 11, 128)
		id := f.broker.send(t, "engine:copy-model", f.request(completed.manifest.Model, ""), true)
		f.assertCopied(t, completed, controllerCopyResponse(t, f.broker, id))
		f.idle(t)
		f.sourceEM.stop()
		before := len(f.files[f.source.uuid].requests())
		request := f.request(completed.manifest.Model, completed.manifest.ExportID)
		id = f.broker.send(t, "engine:copy-model", request, true)
		f.assertCopiedArtifacts(t, completed, controllerCopyResponse(t, f.broker, id))
		if len(f.files[f.source.uuid].requests()) != before {
			t.Fatal("offline completed receipt contacted source")
		}
		f.sourceEM = linkedStartManager(t, f, f.source, filepath.Join(f.source.base, "cluster"), nil)
		f.directoryToManagers(t, false)
		f.idle(t)
	})
	t.Run("destination escape refused", func(t *testing.T) {
		f := newControllerCopyFixture(t)
		model := f.seed(t, "linked-escape", 29)
		parent := filepath.Join(f.stores[f.target.uuid], ".pair-copies")
		if err := os.MkdirAll(parent, 0700); err != nil {
			t.Fatal(err)
		}
		escape := t.TempDir()
		sentinel := filepath.Join(escape, "sentinel")
		modelSettingsWrite(t, sentinel, []byte("unchanged"))
		if err := os.Symlink(escape, controllerCopyAliasDir(f.stores[f.target.uuid], model.manifest.Model)); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		id := f.broker.send(t, "engine:copy-model", f.request(model.manifest.Model, ""), true)
		if response := controllerCopyResponse(t, f.broker, id); response.Error == nil {
			t.Fatal("destination escape copied")
		}
		if data := readLiveDurableFile(t, sentinel); string(data) != "unchanged" {
			t.Fatal("destination escape changed sentinel")
		}
		f.idle(t)
	})
	t.Run("retarget after listing refuses copy and fresh retry succeeds", func(t *testing.T) {
		f := newControllerCopyFixture(t)
		f.directoryToManagers(t, true)
		model := f.seed(t, "linked-swap", 23)
		path := model.paths[model.manifest.Files[0].Path]
		data := model.data[model.manifest.Files[0].Path]
		first, replacement := filepath.Join(t.TempDir(), "first"), filepath.Join(t.TempDir(), "replacement")
		modelSettingsWrite(t, first, data)
		modelSettingsWrite(t, replacement, data)
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(first, path); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		observer := f.files[f.source.uuid]
		observer.mu.Lock()
		observer.fault = &controllerCopyFault{model: model.manifest.Model, beforeGET: func() error {
			if err := os.Remove(path); err != nil {
				return err
			}
			return os.Symlink(replacement, path)
		}}
		observer.mu.Unlock()
		defer func() { observer.mu.Lock(); observer.fault = nil; observer.mu.Unlock() }()
		request := f.request(model.manifest.Model, "")
		id := f.broker.send(t, "engine:copy-model", request, true)
		response := controllerCopyResponse(t, f.broker, id)
		if response.Error == nil {
			t.Fatal("retargeted issued export copied")
		}
		f.assertNoReceipt(t, model)
		f.idle(t)
		observer.mu.Lock()
		called := observer.fault.beforeGET == nil
		observer.fault = nil
		observer.mu.Unlock()
		if !called {
			t.Fatal("custom COPY refusal did not reach the after-LIST file request")
		}
		if got := mustLinkedTarget(t, path); got != replacement {
			t.Fatalf("custom COPY retarget = %q, want %q", got, replacement)
		}
		reader := downloadTransferReader(t, f.target, f.source)
		linkedProcessListing(t, reader, "https://127.0.0.1:"+strconv.Itoa(observer.port), model)
		request = f.request(model.manifest.Model, "")
		id = f.broker.send(t, "engine:copy-model", request, true)
		f.assertCopied(t, model, controllerCopyResponse(t, f.broker, id))
		f.idle(t)
	})
	t.Run("recognized HF cache direct", func(t *testing.T) {
		f := newControllerCopyFixture(t)
		const model, revision = "fixture/linked-GGUF:Q4_K_M", "0123456789abcdef0123456789abcdef01234567"
		linkedWatchSentinel(t, f, []byte("outside sentinel"))
		fake := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet {
				http.NotFound(w, r)
				return
			}
			switch r.URL.Path {
			case "/ready":
				io.WriteString(w, `{"role":"router"}`)
			case "/models":
				io.WriteString(w, `{"data":[{"id":"fixture/linked-GGUF:Q4_K_M","status":{"value":"unloaded"}}]}`)
			default:
				http.NotFound(w, r)
			}
		}))
		t.Cleanup(fake.Close)
		f.sourceEM.stop()
		f.writeManifestWithRouter(t, f.source, portOfURL(t, fake.URL))
		f.sourceEM = linkedStartManager(t, f, f.source, filepath.Join(f.source.base, "cluster"), nil)
		f.directoryToManagers(t, false)
		cache := filepath.Join(f.stores[f.source.uuid], "models--fixture--linked-GGUF")
		modelSettingsWrite(t, filepath.Join(cache, "refs", "main"), []byte(revision))
		outside := t.TempDir()
		data := modelSettingsGGUF("blk.0.weight")
		target := filepath.Join(outside, "weights.gguf")
		modelSettingsWrite(t, target, data)
		modelSettingsWrite(t, filepath.Join(outside, "unlisted.gguf"), []byte("outside sentinel"))
		modelSettingsWrite(t, filepath.Join(f.stores[f.source.uuid], "unrecognized-loose.gguf"), data)
		link := filepath.Join(cache, "snapshots", revision, "model-Q4_K_M.gguf")
		if err := os.MkdirAll(filepath.Dir(link), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(target, link); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		reader := downloadTransferReader(t, f.target, f.source)
		base := "https://127.0.0.1:" + strconv.Itoa(f.files[f.source.uuid].port)
		files := map[string][]byte{"model-Q4_K_M.gguf": data}
		listed := linkedNativeRead(t, f, false, "llamacpp", model, files)
		status, body := linkedProcessResponse(t, reader, base, http.MethodGet, "/v1/models/files", url.Values{"engine": {"llamacpp"}, "model": {"fixture/unrecognized-loose-GGUF:Q4_K_M"}}, "")
		if status != http.StatusNotFound || !bytes.Equal(body, []byte("model source unavailable\n")) || bytes.Contains(body, data) || bytes.Contains(body, []byte(outside)) {
			t.Fatalf("unrecognized loose source status=%d", status)
		}
		ref := filepath.Join(cache, "refs", "main")
		if err := os.Remove(ref); err != nil {
			t.Fatal(err)
		}
		modelSettingsWrite(t, ref, []byte(revision))
		linkedOldNativeRefused(t, f, false, listed, data)
		linkedNativeCopyRetarget(t, f, "llamacpp", model, func() error {
			if err := os.Remove(ref); err != nil {
				return err
			}
			return os.WriteFile(ref, []byte(revision), 0o600)
		})
		linkedNativeRead(t, f, false, "llamacpp", model, files)
		request := f.request(model, "")
		id := f.broker.send(t, "engine:copy-model", request, true)
		response := task2Response(t, f.broker, id)
		if response.Error != nil {
			t.Fatalf("HF copy: %+v", response.Error)
		}
		var result struct {
			Result struct {
				Layout   string `json:"layout"`
				ExportID string `json:"exportId"`
			} `json:"result"`
		}
		task2Decode(t, response.Result, &result)
		if result.Result.Layout != "" || result.Result.ExportID != "" {
			t.Fatalf("native result=%+v", result)
		}
		got := readLiveDurableFile(t, filepath.Join(f.stores[f.target.uuid], "models--fixture--linked-GGUF", "snapshots", revision, "model-Q4_K_M.gguf"))
		if !bytes.Equal(got, data) || phase04SHA(got) != phase04SHA(data) {
			t.Fatal("HF destination bytes differ")
		}
		linkedExactDestination(t, f.stores[f.target.uuid], map[string][]byte{"models--fixture--linked-GGUF/refs/main": []byte(revision), "models--fixture--linked-GGUF/snapshots/" + revision + "/model-Q4_K_M.gguf": data})
		if got := readLiveDurableFile(t, filepath.Join(outside, "unlisted.gguf")); string(got) != "outside sentinel" {
			t.Fatal("outside sentinel changed")
		}
		f.idle(t)
	})
	t.Run("recognized Ollama linked blobs direct", func(t *testing.T) {
		f := newControllerCopyFixture(t)
		linkedWatchSentinel(t, f, []byte("private sentinel"))
		fake := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet {
				http.NotFound(w, r)
				return
			}
			switch r.URL.Path {
			case "/ready":
				io.WriteString(w, `{"ready":true}`)
			case "/api/tags":
				io.WriteString(w, `{"models":[{"name":"linked-ollama"}]}`)
			case "/api/ps":
				io.WriteString(w, `{"models":[]}`)
			default:
				http.NotFound(w, r)
			}
		}))
		t.Cleanup(fake.Close)
		for _, n := range []*task1Node{f.source, f.target} {
			marker := filepath.Join(n.base, "synthetic-engine-installed")
			rt := map[string]any{"bin": marker, "port": 0, "ready": nil, "health": nil, "auth_statuses": []int{}, "env": map[string]string{"OLLAMA_MODELS": f.stores[n.uuid]}}
			if n == f.source {
				rt["port"] = portOfURL(t, fake.URL)
				rt["ready"] = map[string]any{"http": "http://127.0.0.1:{port}/ready", "status": http.StatusOK, "timeout_s": 1}
			}
			task1WriteJSON(t, filepath.Join(jointAppDir(n.base), "engines", "ollama.json"), map[string]any{"engine": "ollama", "models_dir": f.stores[n.uuid], "runtime": rt, "platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{marker}, "models_dir": f.stores[n.uuid], "runtime": rt}}})
		}
		linkedRestartManagers(t, f, nil)
		var state struct {
			Running bool `json:"running"`
		}
		task2Decode(t, f.sourceEM.call(t, "engine:status", map[string]string{"engine": "ollama"}).Result, &state)
		if !state.Running {
			t.Fatal("private fake Ollama was not recognized")
		}
		data := []byte("synthetic linked Ollama blob")
		digest := phase04SHA(data)
		outside := t.TempDir()
		target := filepath.Join(outside, "blob")
		modelSettingsWrite(t, target, data)
		modelSettingsWrite(t, filepath.Join(outside, "sentinel"), []byte("private sentinel"))
		path := filepath.Join(f.stores[f.source.uuid], "blobs", "sha256-"+digest)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(target, path); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		manifest := map[string]any{"schemaVersion": 2, "config": map[string]any{"digest": "sha256:" + digest, "size": len(data)}, "layers": []any{}}
		manifestPath := filepath.Join("manifests", "registry.ollama.ai", "library", "linked-ollama", "latest")
		modelSettingsWrite(t, filepath.Join(f.stores[f.source.uuid], manifestPath), mustMarshal(t, manifest))
		manifestBytes := readLiveDurableFile(t, filepath.Join(f.stores[f.source.uuid], manifestPath))
		files := map[string][]byte{"blobs/sha256-" + digest: data, filepath.ToSlash(manifestPath): manifestBytes}
		listed := linkedNativeRead(t, f, false, "ollama", "linked-ollama", files)
		selector := filepath.Join(f.stores[f.source.uuid], manifestPath)
		if err := os.Remove(selector); err != nil {
			t.Fatal(err)
		}
		modelSettingsWrite(t, selector, manifestBytes)
		linkedOldNativeRefused(t, f, false, listed, data)
		linkedNativeCopyRetarget(t, f, "ollama", "linked-ollama", func() error {
			if err := os.Remove(selector); err != nil {
				return err
			}
			return os.WriteFile(selector, manifestBytes, 0o600)
		})
		linkedNativeRead(t, f, false, "ollama", "linked-ollama", files)
		request := f.request("linked-ollama", "")
		request.Engine = "ollama"
		id := f.broker.send(t, "engine:copy-model", request, true)
		response := task2Response(t, f.broker, id)
		if response.Error != nil {
			t.Fatalf("Ollama copy: %+v", response.Error)
		}
		linkedNativeResult(t, response)
		got := readLiveDurableFile(t, filepath.Join(f.stores[f.target.uuid], "blobs", "sha256-"+digest))
		if !bytes.Equal(got, data) || phase04SHA(got) != digest {
			t.Fatal("Ollama destination differs")
		}
		if _, err := os.Stat(filepath.Join(f.stores[f.target.uuid], manifestPath)); err != nil {
			t.Fatal(err)
		}
		linkedExactDestination(t, f.stores[f.target.uuid], files)
		if got := readLiveDurableFile(t, filepath.Join(outside, "sentinel")); string(got) != "private sentinel" {
			t.Fatal("Ollama sentinel changed")
		}
		f.idle(t)
	})
	t.Run("recognized LM Studio linked tree reverse", func(t *testing.T) {
		f := newControllerCopyFixture(t)
		linkedWatchSentinel(t, f, []byte("private sentinel"))
		fake := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet {
				http.NotFound(w, r)
				return
			}
			switch r.URL.Path {
			case "/ready":
				io.WriteString(w, `{"ready":true}`)
			case "/api/v1/models":
				io.WriteString(w, `{"models":[{"key":"linked-mlx","loaded_instances":[]}]}`)
			default:
				http.NotFound(w, r)
			}
		}))
		t.Cleanup(fake.Close)
		image, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		rows := filepath.Join(f.source.base, "lms-rows.json")
		task1WriteJSON(t, rows, []map[string]string{{"modelKey": "linked-mlx", "path": "owner/linked-mlx"}})
		for _, n := range []*task1Node{f.source, f.target} {
			marker := filepath.Join(n.base, "synthetic-engine-installed")
			cli := filepath.Join(n.base, "never-start-cli")
			if n == f.source {
				cli = image
			}
			rt := map[string]any{"bin": marker, "cli": cli, "port": 0, "ready": nil, "health": nil, "auth_statuses": []int{}}
			if n == f.source {
				rt["port"] = portOfURL(t, fake.URL)
				rt["ready"] = map[string]any{"http": "http://127.0.0.1:{port}/ready", "status": http.StatusOK, "timeout_s": 1}
			}
			task1WriteJSON(t, filepath.Join(jointAppDir(n.base), "engines", "lmstudio.json"), map[string]any{"engine": "lmstudio", "actions": map[string]any{"delete_model": map[string]bool{"restart_after": false}}, "models_dir": f.stores[n.uuid], "runtime": rt, "platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{marker}, "models_dir": f.stores[n.uuid], "runtime": rt}}})
		}
		linkedRestartManagers(t, f, []string{"NVPAIR_TEST_MODEL_LMS_IMAGE=" + image, "NVPAIR_TEST_MODEL_LMS_ROWS=" + rows})
		f.directoryToManagers(t, true)
		outside := t.TempDir()
		modelSettingsWrite(t, filepath.Join(outside, "weights.safetensors"), []byte("linked mlx weights"))
		modelSettingsWrite(t, filepath.Join(outside, "config.json"), []byte("{}"))
		modelSettingsWrite(t, filepath.Join(outside, "nested", "empty.json"), nil)
		modelSettingsWrite(t, filepath.Join(filepath.Dir(outside), "unlisted-mlx"), []byte("private sentinel"))
		link := filepath.Join(f.stores[f.source.uuid], "owner", "linked-mlx")
		if err := os.MkdirAll(filepath.Dir(link), 0700); err != nil {
			t.Fatal(err)
		}
		linkedDirectory(t, outside, link)
		files := map[string][]byte{"owner/linked-mlx/weights.safetensors": []byte("linked mlx weights"), "owner/linked-mlx/config.json": []byte("{}"), "owner/linked-mlx/nested/empty.json": {}}
		listed := linkedNativeRead(t, f, true, "lmstudio", "linked-mlx", files)
		replacement := t.TempDir()
		for rel, data := range map[string][]byte{"weights.safetensors": []byte("linked mlx weights"), "config.json": []byte("{}"), "nested/empty.json": {}} {
			modelSettingsWrite(t, filepath.Join(replacement, rel), data)
		}
		if err := os.Remove(link); err != nil {
			t.Fatal(err)
		}
		linkedDirectory(t, replacement, link)
		linkedOldNativeRefused(t, f, true, listed, []byte("linked mlx weights"))
		linkedNativeCopyRetarget(t, f, "lmstudio", "linked-mlx", func() error {
			if err := os.Remove(link); err != nil {
				return err
			}
			if runtime.GOOS == "windows" {
				return exec.Command("cmd.exe", "/d", "/c", "mklink", "/J", link, outside).Run()
			}
			return os.Symlink(outside, link)
		})
		linkedNativeRead(t, f, true, "lmstudio", "linked-mlx", files)
		request := f.request("linked-mlx", "")
		request.Engine = "lmstudio"
		id := f.broker.send(t, "engine:copy-model", request, true)
		response := task2Response(t, f.broker, id)
		if response.Error != nil {
			t.Fatalf("LM Studio copy: %+v", response.Error)
		}
		linkedNativeResult(t, response)
		for rel, want := range map[string][]byte{"weights.safetensors": []byte("linked mlx weights"), "config.json": []byte("{}"), "nested/empty.json": {}} {
			got := readLiveDurableFile(t, filepath.Join(f.stores[f.target.uuid], "owner", "linked-mlx", rel))
			if !bytes.Equal(got, want) || phase04SHA(got) != phase04SHA(want) {
				t.Fatalf("LM Studio destination mismatch: %s", rel)
			}
		}
		linkedExactDestination(t, f.stores[f.target.uuid], files)
		if got := readLiveDurableFile(t, filepath.Join(filepath.Dir(outside), "unlisted-mlx")); string(got) != "private sentinel" {
			t.Fatal("LM Studio sentinel changed")
		}
		f.idle(t)
	})
	t.Run("completed linked receipt preserves failed catalog provenance", func(t *testing.T) {
		f := newControllerCopyFixture(t)
		model := f.seedSized(t, "fixture-linked-failed-provenance", 43, 128)
		for _, artifact := range model.manifest.Files {
			path := model.paths[artifact.Path]
			data := readLiveDurableFile(t, path)
			target := filepath.Join(t.TempDir(), filepath.Base(path))
			modelSettingsWrite(t, target, data)
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(target, path); err != nil {
				t.Skipf("symlink unavailable: %v", err)
			}
		}
		request := f.request(model.manifest.Model, "")
		id := f.broker.send(t, "engine:copy-model", request, true)
		f.assertCopied(t, model, controllerCopyResponse(t, f.broker, id))
		f.idle(t)
		manager, router := downloadTransferAttachRouter(t, f, f.target)
		reader := downloadTransferReader(t, f.source, f.target)
		router.setStatus(model.manifest.Model, "unloaded")
		downloadTransferExport(t, reader, f.files[f.target.uuid], model)
		pull := downloadTransferPull(t, manager, model.manifest.Model)
		router.waitStarted(t, model.manifest.Model)
		router.event(t, model.manifest.Model, "download_failed")
		if response := task2Response(t, manager, pull); response.Error == nil {
			t.Fatal("catalog failure after linked COPY succeeded")
		}
		router.setStatus(model.manifest.Model, "unloaded")
		before, failed := downloadTransferPhase(t, manager, model.manifest.Model, modeldownload.PhaseFailed)
		if !failed.BlocksReadiness || failed.Reason != modeldownload.ReasonDownloadFailed || slices.Contains(before.ByEngine["llamacpp"], model.manifest.Model) {
			t.Fatalf("failed catalog readiness marker absent: %+v", failed)
		}
		downloadTransferBlockedExport(t, reader, f.files[f.target.uuid], model)
		journalPath := filepath.Join(jointAppDir(f.target.base), "engine-bin", "model-download-state.json")
		journal := readLiveDurableFile(t, journalPath)
		var persisted struct {
			Version int `json:"version"`
			Entries []struct {
				ID         string `json:"id"`
				Engine     string `json:"engine"`
				Model      string `json:"model"`
				ContentKey string `json:"contentKey"`
				Naming     string `json:"naming"`
			} `json:"entries"`
		}
		task2Decode(t, journal, &persisted)
		if persisted.Version != 1 || len(persisted.Entries) != 1 {
			t.Fatalf("failed journal=%+v", persisted)
		}
		marker := persisted.Entries[0]
		if marker.ID != failed.ID || marker.Engine != "llamacpp" || marker.Model != model.manifest.Model || marker.ContentKey != model.manifest.Model || marker.Naming != "exact" {
			t.Fatalf("failed journal provenance=%+v", marker)
		}
		receiptPath := filepath.Join(controllerCopyAliasDir(f.stores[f.target.uuid], model.manifest.Model), "copy.json")
		receipt := readLiveDurableFile(t, receiptPath)
		requests := len(f.files[f.source.uuid].requests())
		mutations := downloadTransferMutations(manager)
		request = f.request(model.manifest.Model, model.manifest.ExportID)
		id = f.broker.send(t, "engine:copy-model", request, true)
		response := controllerCopyResponse(t, f.broker, id)
		f.assertCopiedArtifacts(t, model, response)
		var result struct {
			Result struct {
				FilesSkipped int   `json:"filesSkipped"`
				BytesCopied  int64 `json:"bytesCopied"`
			} `json:"result"`
		}
		task2Decode(t, response.Result, &result)
		if result.Result.FilesSkipped != len(model.manifest.Files) || result.Result.BytesCopied != 0 || len(f.files[f.source.uuid].requests()) != requests || downloadTransferMutations(manager) != mutations {
			t.Fatalf("receipt retry did not remain local: %+v requests %d -> %d mutations %d -> %d", result, requests, len(f.files[f.source.uuid].requests()), mutations, downloadTransferMutations(manager))
		}
		after := downloadTransferModels(t, manager)
		if !after.DownloadState.Equal(*before.DownloadState) || slices.Contains(after.ByEngine["llamacpp"], model.manifest.Model) || !bytes.Equal(journal, readLiveDurableFile(t, journalPath)) || !bytes.Equal(receipt, readLiveDurableFile(t, receiptPath)) {
			t.Fatal("receipt retry changed failed provenance or readiness")
		}
		downloadTransferBlockedExport(t, reader, f.files[f.target.uuid], model)
		f.idle(t)
		downloadTransferController(t, f)
	})
}

func linkedNativeCopyRetarget(t *testing.T, f *controllerCopyFixture, engine, model string, change func() error) {
	t.Helper()
	observer := f.files[f.source.uuid]
	observer.mu.Lock()
	observer.fault = &controllerCopyFault{model: model, beforeGET: change}
	observer.mu.Unlock()
	defer func() { observer.mu.Lock(); observer.fault = nil; observer.mu.Unlock() }()
	request := f.request(model, "")
	request.Engine = engine
	id := f.broker.send(t, "engine:copy-model", request, true)
	if response := controllerCopyResponse(t, f.broker, id); response.Error == nil {
		t.Fatal("native COPY accepted the retargeted issued source")
	}
	f.idle(t)
	observer.mu.Lock()
	called := observer.fault.beforeGET == nil
	observer.mu.Unlock()
	if !called {
		t.Fatal("native COPY refusal did not reach the after-LIST file request")
	}
}

func linkedNativeResult(t *testing.T, response jsonrpc.Message) {
	t.Helper()
	var result struct {
		Result struct {
			Layout                string `json:"layout"`
			ExportID              string `json:"exportId"`
			Durability            string `json:"durability"`
			RequiresConfiguration bool   `json:"requiresConfiguration"`
		} `json:"result"`
	}
	task2Decode(t, response.Result, &result)
	if result.Result.Layout != "" || result.Result.ExportID != "" || result.Result.Durability != "" || result.Result.RequiresConfiguration {
		t.Fatalf("native terminal semantics=%+v", result)
	}
}

func linkedSyntheticModel(t *testing.T, f *controllerCopyFixture, alias string, linked bool) controllerCopyModel {
	t.Helper()
	model := f.seedSized(t, alias, 7, 128)
	for i := range model.manifest.Files {
		artifact := &model.manifest.Files[i]
		data := modelSettingsGGUF("blk." + strconv.Itoa(i) + ".weight")
		path := model.paths[artifact.Path]
		if linked {
			target := filepath.Join(t.TempDir(), "outside.gguf")
			modelSettingsWrite(t, target, data)
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(target, path); err != nil {
				t.Skipf("symlink unavailable: %v", err)
			}
		} else {
			modelSettingsWrite(t, path, data)
		}
		model.data[artifact.Path] = data
		artifact.Size, artifact.SHA256 = int64(len(data)), phase04SHA(data)
	}
	model.manifest.ExportID = phase04SHA(mustMarshal(t, struct {
		Version   int                      `json:"version"`
		Model     string                   `json:"model"`
		Artifacts []controllerCopyArtifact `json:"artifacts"`
	}{1, alias, model.manifest.Files}))
	return model
}

func linkedExactDestination(t *testing.T, root string, expected map[string][]byte) {
	t.Helper()
	actual := strataContractArtifacts(t, root)
	for path, hash := range actual {
		if hash == "directory" {
			continue
		}
		want, ok := expected[filepath.ToSlash(path)]
		if !ok || hash != phase04SHA(want) {
			t.Fatalf("unexpected destination artifact %s hash=%s", path, hash)
		}
	}
	for path, want := range expected {
		if actual[filepath.FromSlash(path)] != phase04SHA(want) {
			t.Fatalf("missing destination artifact %s", path)
		}
	}
}

func linkedDirectory(t *testing.T, target, link string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		if output, err := exec.Command("cmd.exe", "/d", "/c", "mklink", "/J", link, target).CombinedOutput(); err != nil {
			t.Skipf("junction unavailable: %v %s", err, output)
		}
		return
	}
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("directory symlink unavailable: %v", err)
	}
}

func linkedWatchSentinel(t *testing.T, f *controllerCopyFixture, sentinel []byte) {
	t.Helper()
	observer := f.files[f.source.uuid]
	observer.mu.Lock()
	observer.forbidden = bytes.Clone(sentinel)
	observer.mu.Unlock()
	t.Cleanup(func() {
		observer.mu.Lock()
		defer observer.mu.Unlock()
		if observer.forbiddenResponseBytes != 0 {
			t.Errorf("forbidden successful HTTP response bytes=%d", observer.forbiddenResponseBytes)
		}
		t.Logf("forbidden successful HTTP response bytes=%d (HTTP evidence only)", observer.forbiddenResponseBytes)
	})
}

func linkedConfigureSelected(t *testing.T, f *controllerCopyFixture, model string) {
	t.Helper()
	var launch settings.LaunchState
	task2Decode(t, f.sourceEM.call(t, "engine:get-launch", map[string]string{"engine": "llamacpp"}).Result, &launch)
	if launch.Running || launch.Serving != nil {
		t.Fatal("selected configuration requires the private Off manager without Runtime.Serving")
	}
	// The ordinary fixture deliberately has no runtime listener. Give its
	// saved Off configuration an allocated private port before selected edits,
	// which must echo the complete target baseline.
	initial := settings.ConfigFromLaunch(launch, allocatePrivateCopyPort(t))
	initial.ServerPort = allocatePrivateCopyPort(t)
	f.sourceEM.call(t, "engine:configure-launch", settings.Configure{Engine: "llamacpp", OperationID: "linked-private-port", Settings: initial})
	task2Decode(t, f.sourceEM.call(t, "engine:get-launch", map[string]string{"engine": "llamacpp"}).Result, &launch)
	request := settings.Request{Engine: "llamacpp", Settings: settings.ConfigFromLaunch(launch, allocatePrivateCopyPort(t)), ModelChange: &modelconfig.ModelChange{ModelID: model, ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: 4096}, Slots: modelSettingsInt(1), GPU: &modelconfig.GPU{Mode: "cpu"}}}}}
	var preview settings.Preview
	task2Decode(t, f.sourceEM.call(t, "engine:preview-launch", settings.WorkerRequest[settings.Request]{Request: request}).Result, &preview)
	if len(preview.Errors) != 0 || preview.Conflict != nil || preview.Settings.Serving != nil || len(preview.Settings.ModelLoads) == 0 {
		t.Fatalf("selected typed preview=%+v", preview)
	}
	var configured settings.LaunchState
	task2Decode(t, f.sourceEM.call(t, "engine:configure-launch", settings.Configure{Engine: "llamacpp", OperationID: "linked-selected", Settings: preview.Settings}).Result, &configured)
	if configured.Running || configured.Serving != nil || !modelconfig.EqualLoads(configured.ModelLoads, preview.Settings.ModelLoads) {
		t.Fatal("selected typed configuration readback differs")
	}
}

func linkedNativeRead(t *testing.T, f *controllerCopyFixture, reverse bool, engine, model string, files map[string][]byte) controllerCopyManifest {
	t.Helper()
	client := downloadTransferReader(t, f.target, f.source)
	port := f.files[f.source.uuid].port
	if reverse {
		port = f.blocked
		address := "127.0.0.1:" + strconv.Itoa(port)
		adapter := clustertrust.NewPeerServiceClient(f.target.mesh, clustertrust.PeerServiceClientOptions{BaseConfigDir: f.target.base, DirectTimeout: time.Second,
			ResolveRoute: func(peer string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
				return clustertrust.PeerServiceRoute{Incarnation: "fixture", Address: address, Protocol: clustertrust.ReverseTLS, Generation: 1}, peer == f.source.uuid && service == noderec.ServiceEngineControl
			}})
		client = adapter.Client(f.source.uuid, noderec.ServiceEngineControl)
		t.Cleanup(client.CloseIdleConnections)
	}
	base := "https://127.0.0.1:" + strconv.Itoa(port)
	for path := range files {
		q := url.Values{"engine": {engine}, "model": {model}, "path": {path}}
		status, body := linkedProcessResponse(t, client, base, http.MethodGet, "/v1/models/file", q, "")
		if status != http.StatusConflict || (len(files[path]) > 0 && bytes.Equal(body, files[path])) {
			t.Fatalf("native unissued GET status=%d", status)
		}
		break
	}
	query := url.Values{"engine": {engine}, "model": {model}}
	status, data := linkedProcessResponse(t, client, base, http.MethodGet, "/v1/models/files", query, "")
	if status != http.StatusOK || bytes.Contains(data, []byte(f.source.base)) {
		t.Fatalf("native list status=%d", status)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	var list controllerCopyManifest
	results := 0
	for {
		var frame struct {
			Type   string          `json:"type"`
			Result json.RawMessage `json:"result"`
		}
		err := decoder.Decode(&frame)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if frame.Type == "progress" {
			continue
		}
		if frame.Type != "result" {
			t.Fatalf("native listing error: %s", data)
		}
		task2Decode(t, frame.Result, &list)
		results++
	}
	if results != 1 || list.Layout != "" || list.ExportID == "" || len(list.Files) != len(files) {
		t.Fatalf("native list=%+v", list)
	}
	for _, refusal := range []struct {
		path   string
		status int
	}{
		{"../unlisted", http.StatusBadRequest},
		{"/unlisted", http.StatusBadRequest},
		{"unlisted", http.StatusNotFound},
	} {
		q := url.Values{"engine": {engine}, "model": {model}, "path": {refusal.path}, "exportId": {list.ExportID}}
		for _, method := range []string{http.MethodGet, http.MethodHead} {
			for _, rangeHeader := range []string{"", "bytes=0-7"} {
				status, body := linkedProcessResponse(t, client, base, method, "/v1/models/file", q, rangeHeader)
				if status != refusal.status || bytes.Contains(body, []byte(f.source.base)) {
					t.Fatalf("native unlisted %q %s status=%d, want %d", refusal.path, method, status, refusal.status)
				}
			}
		}
	}
	first := list.Files[0].Path
	for _, q := range []url.Values{
		{"engine": {engine}, "model": {model}, "path": {first}, "exportId": {"wrong"}},
		{"engine": {"unknown-engine"}, "model": {model}, "path": {first}, "exportId": {list.ExportID}},
		{"engine": {engine}, "model": {"not-listed"}, "path": {first}, "exportId": {list.ExportID}},
	} {
		for _, method := range []string{http.MethodGet, http.MethodHead} {
			status, body := linkedProcessResponse(t, client, base, method, "/v1/models/file", q, "bytes=0-7")
			if status != http.StatusConflict || bytes.Contains(body, []byte(f.source.base)) {
				t.Fatalf("native invalid selection %s status=%d", method, status)
			}
		}
	}
	for _, file := range list.Files {
		want, ok := files[file.Path]
		if !ok || file.SHA256 != phase04SHA(want) {
			t.Fatalf("unexpected native file: %+v", file)
		}
		q := url.Values{"engine": {engine}, "model": {model}, "path": {file.Path}, "exportId": {list.ExportID}}
		status, got := linkedProcessResponse(t, client, base, http.MethodGet, "/v1/models/file", q, "")
		if status != http.StatusOK || !bytes.Equal(got, want) {
			t.Fatalf("native GET %s status=%d", file.Path, status)
		}
		status, got = linkedProcessResponse(t, client, base, http.MethodHead, "/v1/models/file", q, "")
		if status != http.StatusOK || len(got) != 0 {
			t.Fatalf("native HEAD %s status=%d", file.Path, status)
		}
		if len(want) > 1 {
			status, got = linkedProcessResponse(t, client, base, http.MethodGet, "/v1/models/file", q, "bytes=1-")
			if status != http.StatusPartialContent || !bytes.Equal(got, want[1:]) {
				t.Fatalf("native range %s status=%d", file.Path, status)
			}
		}
	}
	return list
}

func linkedOldNativeRefused(t *testing.T, f *controllerCopyFixture, reverse bool, list controllerCopyManifest, forbidden []byte) {
	t.Helper()
	client, base := linkedReader(t, f, reverse)
	for _, file := range list.Files {
		q := url.Values{"engine": {list.Engine}, "model": {list.Model}, "path": {file.Path}, "exportId": {list.ExportID}}
		for _, method := range []string{http.MethodGet, http.MethodHead} {
			status, body := linkedProcessResponse(t, client, base, method, "/v1/models/file", q, "")
			if status != http.StatusConflict || bytes.Contains(body, forbidden) {
				t.Fatalf("old native binding %s status=%d", method, status)
			}
		}
	}
}

func linkedReader(t *testing.T, f *controllerCopyFixture, reverse bool) (*http.Client, string) {
	t.Helper()
	port := f.files[f.source.uuid].port
	client := downloadTransferReader(t, f.target, f.source)
	if reverse {
		port = f.blocked
		address := "127.0.0.1:" + strconv.Itoa(port)
		adapter := clustertrust.NewPeerServiceClient(f.target.mesh, clustertrust.PeerServiceClientOptions{BaseConfigDir: f.target.base, DirectTimeout: time.Second, ResolveRoute: func(peer string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
			return clustertrust.PeerServiceRoute{Incarnation: "fixture", Address: address, Protocol: clustertrust.ReverseTLS, Generation: 1}, peer == f.source.uuid && service == noderec.ServiceEngineControl
		}})
		client = adapter.Client(f.source.uuid, noderec.ServiceEngineControl)
		t.Cleanup(client.CloseIdleConnections)
	}
	return client, "https://127.0.0.1:" + strconv.Itoa(port)
}

func linkedProcessListing(t *testing.T, client *http.Client, base string, model controllerCopyModel) {
	t.Helper()
	query := url.Values{"engine": {model.manifest.Engine}, "model": {model.manifest.Model}}
	status, body := linkedProcessResponse(t, client, base, http.MethodGet, "/v1/models/files", query, "")
	if status != http.StatusOK {
		t.Fatalf("listing status=%d body=%s", status, body)
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	seen := false
	for {
		var frame struct {
			Type   string          `json:"type"`
			Result json.RawMessage `json:"result"`
		}
		err := decoder.Decode(&frame)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if frame.Type == "progress" {
			continue
		}
		if frame.Type != "result" || seen {
			t.Fatalf("unexpected listing frame: %s", body)
		}
		var got controllerCopyManifest
		task2Decode(t, frame.Result, &got)
		if !bytes.Equal(mustMarshal(t, got), mustMarshal(t, model.manifest)) {
			t.Fatalf("listing mismatch: %+v", got)
		}
		seen = true
	}
	if !seen {
		t.Fatal("listing omitted result")
	}
}

func linkedRestartManagers(t *testing.T, f *controllerCopyFixture, env []string) {
	t.Helper()
	f.sourceEM.stop()
	f.targetEM.stop()
	f.sourceEM = linkedStartManager(t, f, f.source, filepath.Join(f.source.base, "cluster"), env)
	f.targetEM = linkedStartManager(t, f, f.target, f.copyCluster, nil)
	f.directoryToManagers(t, false)
}

func linkedStartManager(t *testing.T, f *controllerCopyFixture, n *task1Node, cluster string, extra []string) *task1Process {
	t.Helper()
	// Only the Off manager's private configuration rebind is acknowledged. This is
	// the parent handshake, not approval to start or review a native engine.
	reply := func(relay settings.Relay) task1SettingsReply {
		result := task1SettingsReply{ID: relay.ID, Error: "unexpected private settings request", ErrorCode: -32602}
		if relay.Method == "rebind" && relay.Request.Engine == "llamacpp" && relay.Caller == "" && relay.Request.Settings.Serving == nil {
			result.Error, result.ErrorCode, result.Result = "", 0, json.RawMessage(`{}`)
		}
		return result
	}
	p := startTask1ProcessWithSettings(t, n.base, engineMgrBin, extra, reply, "--cluster-dir", cluster, "--control-host", "127.0.0.1", "--control-port", strconv.Itoa(n.ports[noderec.ServiceEngineControl]), "--http-port", "0", "--loaded-poll-interval", "0")
	completeEngineManagerRecovery(t, func(method string, params any) jsonrpc.Message { return p.call(t, method, params) })
	p.send(t, "engine:node-projection", map[string]any{"policy": nodepolicy.Default().Advertisement(), "pausedLocally": false, "managementBlocked": false}, false)
	p.call(t, "engine:models", nil)
	return p
}

func mustLinkedTarget(t *testing.T, path string) string {
	t.Helper()
	target, err := os.Readlink(path)
	if err != nil {
		t.Fatal(err)
	}
	return target
}

func linkedProcessResponse(t *testing.T, client *http.Client, base, method, path string, query url.Values, rangeHeader string) (int, []byte) {
	t.Helper()
	req, err := http.NewRequest(method, base+path+"?"+query.Encode(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if rangeHeader != "" {
		req.Header.Set("Range", rangeHeader)
	}
	response, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	data, readErr := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	closeErr := response.Body.Close()
	if readErr != nil || closeErr != nil {
		t.Fatalf("read response: %v; close: %v", readErr, closeErr)
	}
	for _, selector := range []string{query.Get("path"), query.Get("model")} {
		if filepath.IsAbs(selector) && bytes.Contains(data, []byte(selector)) {
			t.Fatal("response disclosed a supplied host path")
		}
	}
	return response.StatusCode, data
}
