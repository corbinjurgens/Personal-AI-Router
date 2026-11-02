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
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/modeldownload"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

// The registered test image supplies only the private LM Studio row fixture.
// It never invokes an installed vendor CLI or accepts other commands.
func init() {
	image := os.Getenv("NVPAIR_TEST_LINKED_LMS_IMAGE")
	if image == "" || image != os.Args[0] {
		return
	}
	if len(os.Args) != 3 || os.Args[1] != "ls" || os.Args[2] != "--json" {
		os.Exit(2)
	}
	data, err := os.ReadFile(os.Getenv("NVPAIR_TEST_LINKED_LMS_ROWS"))
	if err != nil {
		os.Exit(2)
	}
	if _, err := os.Stdout.Write(data); err != nil {
		os.Exit(2)
	}
	os.Exit(0)
}

func TestLinkedModelsProcess(t *testing.T) {
	f := newControllerCopyFixture(t)
	linkedTargets := t.TempDir()
	var completed controllerCopyModel
	for _, tc := range []struct {
		name, alias string
		reverse     bool
	}{
		{"custom direct", "linked-custom-direct", false},
		{"custom reverse", "linked-custom-reverse", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f.directoryToManagers(t, tc.reverse)
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
				if err := os.Symlink(target, path); err != nil {
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
			for _, path := range []string{"model/unlisted.gguf", "../unlisted.gguf", "/unlisted.gguf", outsideSibling} {
				query.Set("path", path)
				status, body = linkedProcessResponse(t, reader, base, http.MethodGet, "/v1/models/file", query, "")
				if status == http.StatusOK || bytes.Contains(body, sentinel) || bytes.Contains(body, []byte(f.source.base)) {
					t.Fatalf("unlisted %q status=%d", path, status)
				}
			}
			for _, changed := range []url.Values{
				{"engine": {"llamacpp"}, "model": {tc.alias}, "path": {model.manifest.Files[0].Path}, "exportId": {"wrong"}},
				{"engine": {"ollama"}, "model": {tc.alias}, "path": {model.manifest.Files[0].Path}, "exportId": {model.manifest.ExportID}},
				{"engine": {"llamacpp"}, "model": {"not-listed"}, "path": {model.manifest.Files[0].Path}, "exportId": {model.manifest.ExportID}},
			} {
				status, body = linkedProcessResponse(t, reader, base, http.MethodGet, "/v1/models/file", changed, "")
				if status == http.StatusOK || bytes.Contains(body, sentinel) || bytes.Contains(body, []byte(f.source.base)) {
					t.Fatalf("invalid identity status=%d", status)
				}
			}
			request := f.request(tc.alias, "")
			id := f.broker.send(t, "engine:copy-model", request, true)
			f.assertCopied(t, model, task2Response(t, f.broker, id))
			if !tc.reverse {
				completed = model
			}
			controllerCopyTerminal(t, f.broker, request, "done")
			f.idle(t)
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
		})
	}
	t.Run("completed receipt retry with source offline", func(t *testing.T) {
		if completed.manifest.Model == "" {
			t.Skip("direct linked custom case unavailable")
		}
		f.sourceEM.stop()
		before := len(f.files[f.source.uuid].requests())
		request := f.request(completed.manifest.Model, completed.manifest.ExportID)
		id := f.broker.send(t, "engine:copy-model", request, true)
		f.assertCopiedArtifacts(t, completed, controllerCopyResponse(t, f.broker, id))
		if len(f.files[f.source.uuid].requests()) != before {
			t.Fatal("offline completed receipt contacted source")
		}
		f.sourceEM = linkedStartManager(t, f, f.source, filepath.Join(f.source.base, "cluster"), nil)
		f.directoryToManagers(t, false)
		f.idle(t)
	})
	t.Run("destination escape refused", func(t *testing.T) {
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
		observer.fault = nil
		observer.mu.Unlock()
		reader := downloadTransferReader(t, f.target, f.source)
		linkedProcessListing(t, reader, "https://127.0.0.1:"+strconv.Itoa(observer.port), model)
		request = f.request(model.manifest.Model, "")
		id = f.broker.send(t, "engine:copy-model", request, true)
		f.assertCopied(t, model, controllerCopyResponse(t, f.broker, id))
		f.idle(t)
	})
	t.Run("recognized HF cache direct", func(t *testing.T) {
		const model, revision = "fixture/linked-GGUF:Q4_K_M", "0123456789abcdef0123456789abcdef01234567"
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
		status, body := linkedProcessResponse(t, reader, base, http.MethodGet, "/v1/models/files", url.Values{"engine": {"llamacpp"}, "model": {"unrecognized-loose"}}, "")
		if status == http.StatusOK || bytes.Contains(body, data) || bytes.Contains(body, []byte(outside)) {
			t.Fatalf("unrecognized loose source status=%d", status)
		}
		linkedNativeRead(t, f, false, "llamacpp", model, map[string][]byte{"model-Q4_K_M.gguf": data})
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
		if got := readLiveDurableFile(t, filepath.Join(outside, "unlisted.gguf")); string(got) != "outside sentinel" {
			t.Fatal("outside sentinel changed")
		}
		f.idle(t)
	})
	t.Run("recognized Ollama linked blobs direct", func(t *testing.T) {
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
		task1WriteJSON(t, filepath.Join(f.stores[f.source.uuid], manifestPath), manifest)
		manifestBytes := readLiveDurableFile(t, filepath.Join(f.stores[f.source.uuid], manifestPath))
		linkedNativeRead(t, f, false, "ollama", "linked-ollama", map[string][]byte{"blobs/sha256-" + digest: data, filepath.ToSlash(manifestPath): manifestBytes})
		request := f.request("linked-ollama", "")
		request.Engine = "ollama"
		id := f.broker.send(t, "engine:copy-model", request, true)
		response := task2Response(t, f.broker, id)
		if response.Error != nil {
			t.Fatalf("Ollama copy: %+v", response.Error)
		}
		got := readLiveDurableFile(t, filepath.Join(f.stores[f.target.uuid], "blobs", "sha256-"+digest))
		if !bytes.Equal(got, data) || phase04SHA(got) != digest {
			t.Fatal("Ollama destination differs")
		}
		if _, err := os.Stat(filepath.Join(f.stores[f.target.uuid], manifestPath)); err != nil {
			t.Fatal(err)
		}
		if got := readLiveDurableFile(t, filepath.Join(outside, "sentinel")); string(got) != "private sentinel" {
			t.Fatal("Ollama sentinel changed")
		}
		f.idle(t)
	})
	t.Run("recognized LM Studio linked tree reverse", func(t *testing.T) {
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
			task1WriteJSON(t, filepath.Join(jointAppDir(n.base), "engines", "lmstudio.json"), map[string]any{"engine": "lmstudio", "models_dir": f.stores[n.uuid], "runtime": rt, "platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{marker}, "models_dir": f.stores[n.uuid], "runtime": rt}}})
		}
		linkedRestartManagers(t, f, []string{"NVPAIR_TEST_LINKED_LMS_IMAGE=" + image, "NVPAIR_TEST_LINKED_LMS_ROWS=" + rows})
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
		if err := os.Symlink(outside, link); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		linkedNativeRead(t, f, true, "lmstudio", "linked-mlx", map[string][]byte{"owner/linked-mlx/weights.safetensors": []byte("linked mlx weights"), "owner/linked-mlx/config.json": []byte("{}"), "owner/linked-mlx/nested/empty.json": {}})
		request := f.request("linked-mlx", "")
		request.Engine = "lmstudio"
		id := f.broker.send(t, "engine:copy-model", request, true)
		response := task2Response(t, f.broker, id)
		if response.Error != nil {
			t.Fatalf("LM Studio copy: %+v", response.Error)
		}
		for rel, want := range map[string][]byte{"weights.safetensors": []byte("linked mlx weights"), "config.json": []byte("{}"), "nested/empty.json": {}} {
			got := readLiveDurableFile(t, filepath.Join(f.stores[f.target.uuid], "owner", "linked-mlx", rel))
			if !bytes.Equal(got, want) || phase04SHA(got) != phase04SHA(want) {
				t.Fatalf("LM Studio destination mismatch: %s", rel)
			}
		}
		f.idle(t)
	})
	t.Run("completed linked receipt preserves failed catalog provenance", func(t *testing.T) {
		for _, n := range []*task1Node{f.source, f.target} {
			f.writeManifest(t, n)
			for _, engine := range []string{"ollama", "lmstudio"} {
				if err := os.Remove(filepath.Join(jointAppDir(n.base), "engines", engine+".json")); err != nil {
					t.Fatal(err)
				}
			}
		}
		linkedRestartManagers(t, f, nil)
		f.directoryToManagers(t, false)
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
			t.Fatalf("receipt retry did not remain local: %+v", result)
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

func linkedNativeRead(t *testing.T, f *controllerCopyFixture, reverse bool, engine, model string, files map[string][]byte) {
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
		if status == http.StatusOK || (len(files[path]) > 0 && bytes.Equal(body, files[path])) {
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
	for _, path := range []string{"../unlisted", "/unlisted", "unlisted"} {
		q := url.Values{"engine": {engine}, "model": {model}, "path": {path}, "exportId": {list.ExportID}}
		status, body := linkedProcessResponse(t, client, base, http.MethodGet, "/v1/models/file", q, "")
		if status == http.StatusOK || bytes.Contains(body, []byte(f.source.base)) {
			t.Fatalf("native unlisted GET status=%d", status)
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
	p := startTask1ProcessWithEnv(t, n.base, engineMgrBin, extra, "--cluster-dir", cluster, "--control-host", "127.0.0.1", "--control-port", strconv.Itoa(n.ports[noderec.ServiceEngineControl]), "--http-port", "0", "--loaded-poll-interval", "0")
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
	return response.StatusCode, data
}
