// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
)

type strataServingFixture struct {
	llamaServingFixture
	config strataServingConfig
}

func (f *strataServingFixture) prepareRuntimeHashes(t *testing.T, ctx context.Context, config *engines.ServingConfig) error {
	t.Helper()
	st := settingsState(t, f.ex)
	command, err := launchForState(st, f.request.Settings.ServerPort)
	if err != nil {
		t.Fatal(err)
	}
	rt := st.plat.Runtime
	rt.Serving = config
	return prepareStrataRuntimeHashes(ctx, command, rt)
}

func TestStrataRuntimeRejectsMisclassifiedSelectedFilesBeforeHashing(t *testing.T) {
	for _, name := range []string{"pack/dense.bin", "draft/dense.bin"} {
		t.Run(name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			path := filepath.Join(f.root, filepath.FromSlash(name))
			for i := range f.request.Settings.Serving.Models[0].Artifacts {
				artifact := &f.request.Settings.Serving.Models[0].Artifacts[i]
				if artifact.Binding.SourcePath == path {
					artifact.Role = "runtime:lib:misclassified-pack"
				}
			}
			p := previewSettings(t, f.ex, f.request)
			if !strings.Contains(p.Errors["serving"], "incorrect artifact role") {
				t.Fatalf("role rejection = %v", p.Errors)
			}
			// No declared file may have been streamed before the role rejection,
			// even if its own role was valid and sorted before the bad pack.
			for _, artifact := range f.request.Settings.Serving.Models[0].Artifacts {
				path := artifact.Binding.SourcePath
				if path == fakeEngineBin {
					continue // Shared by fixtures; its cache may already be warm.
				}
				fileHashes.mu.Lock()
				_, hashed := fileHashes.done[path]
				fileHashes.mu.Unlock()
				if hashed {
					t.Errorf("file fully hashed before role rejection: %s", path)
				}
			}
			assertNoLlamaSpawn(t, f.llamaServingFixture)
			assertNoSettingsOverride(t, f.ex)
		})
	}
}

func TestStrataPreparationPreservesPreviewPortConflict(t *testing.T) {
	f := newStrataServingFixture(t)
	launchPort := f.request.Settings.ServerPort
	f.request.Settings.ServerPort++
	p := previewSettings(t, f.ex, f.request)
	if len(p.Errors) != 0 || p.Conflict == nil || p.Conflict.ServerPort != f.request.Settings.ServerPort || p.Conflict.LaunchPort != launchPort {
		t.Fatalf("port conflict preview = %+v", p)
	}
	path := filepath.Join(f.root, "server.py")
	fileHashes.mu.Lock()
	_, hashed := fileHashes.done[path]
	fileHashes.mu.Unlock()
	if hashed {
		t.Fatal("unresolved launch conflict caused runtime hashing")
	}
}

func TestStrataRuntimeRejectsModelAliases(t *testing.T) {
	for _, name := range []string{"model-00001-of-00002.gguf", "pack/dense.bin", "draft/dense.bin"} {
		t.Run(name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			alias := filepath.Join(f.root, "lib", "model-alias.so")
			if err := os.Symlink(filepath.Join(f.root, filepath.FromSlash(name)), alias); err != nil {
				t.Fatal(err)
			}
			m := &f.request.Settings.Serving.Models[0]
			m.Artifacts = append(m.Artifacts, strataArtifact(t, alias, "runtime:lib:model-alias"))
			p := previewSettings(t, f.ex, f.request)
			if !strings.Contains(p.Errors["serving"], "model targets") {
				t.Fatalf("alias error = %v", p.Errors)
			}
			fileHashes.mu.Lock()
			_, hashed := fileHashes.done[alias]
			fileHashes.mu.Unlock()
			if hashed {
				t.Fatal("model alias was fully hashed")
			}
		})
	}
}

func TestStrataRuntimeRejectsHardlinkedModelAliasesBeforeHashing(t *testing.T) {
	for _, name := range []string{"model-00001-of-00002.gguf", "pack/dense.bin", "draft/dense.bin"} {
		t.Run(name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			path := filepath.Join(f.root, filepath.FromSlash(name))
			writeServingFixture(t, path, bytes.Repeat([]byte("x"), 256<<10))
			model := &f.request.Settings.Serving.Models[0]
			for i := range model.Artifacts {
				artifact := &model.Artifacts[i]
				if artifact.Binding.SourcePath == path {
					*artifact = strataArtifact(t, path, artifact.Role)
				}
			}
			alias := filepath.Join(f.root, "lib", "model-hardlink.so")
			if err := os.Link(path, alias); err != nil {
				t.Fatal(err)
			}
			modelInfo, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			aliasInfo, err := os.Stat(alias)
			if err != nil {
				t.Fatal(err)
			}
			if !os.SameFile(modelInfo, aliasInfo) {
				t.Fatal("fixture alias does not name the same physical model file")
			}
			model.Artifacts = append(model.Artifacts, strataArtifact(t, alias, "runtime:lib:model-alias"))
			p := previewSettings(t, f.ex, f.request)
			if !strings.Contains(p.Errors["serving"], "model targets") {
				t.Errorf("hard-link rejection = %v", p.Errors)
			}
			fileHashes.mu.Lock()
			_, hashed := fileHashes.done[alias]
			fileHashes.mu.Unlock()
			if hashed {
				t.Error("hard-linked model alias was fully hashed")
			}
			assertNoLlamaSpawn(t, f.llamaServingFixture)
			assertNoSettingsOverride(t, f.ex)
		})
	}
}

func TestStrataRuntimeRejectsGGUFExtensions(t *testing.T) {
	for _, name := range []string{"extra.gguf", "extra-00001-of-00002.GGUF"} {
		t.Run(name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			path := filepath.Join(f.root, name)
			writeServingFixture(t, path, []byte("synthetic undeclared model"))
			m := &f.request.Settings.Serving.Models[0]
			m.Artifacts = append(m.Artifacts, strataArtifact(t, path, "runtime:lib:extra"))
			p := previewSettings(t, f.ex, f.request)
			if !strings.Contains(p.Errors["serving"], "GGUF") {
				t.Fatalf("extension error = %v", p.Errors)
			}
		})
	}
}

func TestStrataSerialContextNativeMargin(t *testing.T) {
	f := newStrataServingFixture(t)
	f.config.Parallel = 1
	i := slices.Index(f.config.Args, "--batch")
	f.config.Args = append(f.config.Args[:i:i], f.config.Args[i+2:]...)
	f.config.Args[slices.Index(f.config.Args, "--max-context")+1] = "2147483647"
	f.saveConfig(t)
	p := previewSettings(t, f.ex, f.request)
	if len(p.Errors) == 0 {
		t.Fatal("native overflow margin accepted")
	}
}

func TestStrataTotalContextNativeMargin(t *testing.T) {
	f := newStrataServingFixture(t)
	f.config.Args[slices.Index(f.config.Args, "--max-context")+1] = "1073741820"
	f.saveConfig(t)
	p := previewSettings(t, f.ex, f.request)
	if len(p.Errors) == 0 {
		t.Fatal("native total context overflow margin accepted")
	}
}

// An unfinished shared hash models a slow runtime read without allocating a
// large fixture. Every public operation must leave Stop's lock available, and
// cancellation must return while that hash is still unfinished.
func TestStrataRuntimeHashOutsideOperationLock(t *testing.T) {
	for _, operation := range []string{"preview", "readback", "apply", "start", "restart", "wake", "restore", "set port", "action"} {
		t.Run(operation, func(t *testing.T) {
			f := newStrataServingFixture(t)
			p := approveLlamaServing(t, f.llamaServingFixture)
			applyLlamaServing(t, f.llamaServingFixture, p)
			st := settingsState(t, f.ex)
			path := filepath.Join(f.root, "lib", "attested.so")
			info, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			key := hashEntry{path: path, size: info.Size(), mtime: info.ModTime().UnixNano()}
			call := &hashCall{}
			call.wg.Add(1)
			fileHashes.mu.Lock()
			delete(fileHashes.done, path)
			fileHashes.inflight[key] = call
			fileHashes.mu.Unlock()
			defer func() {
				call.wg.Done()
				fileHashes.mu.Lock()
				delete(fileHashes.inflight, key)
				fileHashes.mu.Unlock()
			}()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan error, 1)
			go func() {
				var err error
				switch operation {
				case "preview":
					var result settings.Preview
					result, err = f.ex.PreviewLaunch(f.request, ctx)
					if len(result.Errors) > 0 {
						err = errors.New(result.Errors["serving"])
					}
				case "readback":
					var result settings.LaunchState
					result, err = f.ex.LaunchSettings("fake", ctx)
					if result.ServingReadback != nil && result.ServingReadback.Status == "mismatch" {
						err = errors.New(result.ServingReadback.Reason)
					}
				case "apply":
					_, err = f.ex.ConfigureLaunch(ctx, settings.Configure{Engine: "fake", Settings: p.Settings}, func() error { return nil })
				case "start":
					err = f.ex.Start(ctx, "fake")
				case "restart":
					err = f.ex.Restart(ctx, "fake")
				case "wake":
					err = f.ex.Wake(ctx, "fake")
				case "restore":
					err = f.ex.restoreEnabled(ctx, "fake")
				case "set port":
					_, err = f.ex.SetPort(ctx, "fake", f.request.Settings.ServerPort)
				case "action":
					err = f.ex.restartAfterAction(ctx, st, "fake", "test")
				}
				done <- err
			}()
			select {
			case err := <-done:
				t.Fatalf("operation finished before hash release: %v", err)
			case <-time.After(100 * time.Millisecond):
			}
			if !st.opMu.TryLock() {
				t.Fatal("runtime hash holds lifecycle operation lock")
			}
			st.opMu.Unlock()
			cancel()
			select {
			case err := <-done:
				if err == nil || !strings.Contains(err.Error(), "canceled") {
					t.Fatalf("cancellation = %v", err)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("runtime hash did not cancel")
			}
		})
	}
}

func TestStrataRuntimeChangedAfterPreparation(t *testing.T) {
	f := newStrataServingFixture(t)
	if err := f.prepareRuntimeHashes(t, context.Background(), f.request.Settings.Serving); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(f.root, "lib", "attested.so")
	writeServingFixture(t, path, []byte("changed after hashing"))
	st := settingsState(t, f.ex)
	command, err := launchForState(st, f.request.Settings.ServerPort)
	if err != nil {
		t.Fatal(err)
	}
	rt := st.plat.Runtime
	rt.Serving = f.request.Settings.Serving.Clone()
	// Refresh the bounded binding so this specifically exercises the hash
	// metadata guard, rather than the earlier attestation-binding guard.
	for i := range rt.Serving.Models[0].Artifacts {
		a := &rt.Serving.Models[0].Artifacts[i]
		if a.Binding.SourcePath == path {
			a.Binding, err = servingFileBinding(path)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	if _, err := compileStrataServing(command, rt, false); err == nil {
		t.Fatal("runtime changed during preparation accepted")
	}
}

func TestStrataLockedCompilerRejectsColdRuntimeHash(t *testing.T) {
	f := newStrataServingFixture(t)
	if err := f.prepareRuntimeHashes(t, context.Background(), f.request.Settings.Serving); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(f.root, "lib", "attested.so")
	strataRuntimeHashes.Lock()
	delete(strataRuntimeHashes.files, path)
	strataRuntimeHashes.Unlock()
	st := settingsState(t, f.ex)
	command, err := launchForState(st, f.request.Settings.ServerPort)
	if err != nil {
		t.Fatal(err)
	}
	rt := st.plat.Runtime
	rt.Serving = f.request.Settings.Serving.Clone()
	st.opMu.Lock()
	_, err = compileStrataServing(command, rt, false)
	st.opMu.Unlock()
	if err == nil || !strings.Contains(err.Error(), "changed during preparation") {
		t.Fatalf("cold hash error = %v", err)
	}
	strataRuntimeHashes.Lock()
	_, cached := strataRuntimeHashes.files[path]
	strataRuntimeHashes.Unlock()
	if cached {
		t.Fatal("locked compiler populated a cold runtime hash")
	}
}

func TestStrataRuntimeReplacedWithSameMetadata(t *testing.T) {
	f := newStrataServingFixture(t)
	if err := f.prepareRuntimeHashes(t, context.Background(), f.request.Settings.Serving); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(f.root, "lib", "attested.so")
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	// A rename preserves the old inode so the replacement cannot reuse it.
	if err := os.Rename(path, path+".old"); err != nil {
		t.Fatal(err)
	}
	writeServingFixture(t, path, bytes.Repeat([]byte("x"), int(before.Size())))
	if err := os.Chtimes(path, before.ModTime(), before.ModTime()); err != nil {
		t.Fatal(err)
	}
	if _, err := strataRuntimeDigest(path); err == nil {
		t.Fatal("replaced runtime passed file identity recheck")
	}
	// Re-attest the replacement before preparing it; preparation now checks
	// the entire declaration before it can refresh any full-file hash.
	for i := range f.request.Settings.Serving.Models[0].Artifacts {
		artifact := &f.request.Settings.Serving.Models[0].Artifacts[i]
		if artifact.Binding.SourcePath == path {
			*artifact = strataArtifact(t, path, artifact.Role)
		}
	}
	if err := f.prepareRuntimeHashes(t, context.Background(), f.request.Settings.Serving); err != nil {
		t.Fatal(err)
	}
	sum, err := strataRuntimeDigest(path)
	if err != nil {
		t.Fatal(err)
	}
	want := sha256.Sum256(bytes.Repeat([]byte("x"), int(before.Size())))
	if sum != hex.EncodeToString(want[:]) {
		t.Fatal("preparation reused hash of replaced runtime")
	}
}

func TestStrataRuntimeChangesWhileHashing(t *testing.T) {
	f := newStrataServingFixture(t)
	if err := f.prepareRuntimeHashes(t, context.Background(), f.request.Settings.Serving); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(f.root, "lib", "attested.so")
	artifact := strataArtifact(t, path, "runtime:lib:attested")
	config := f.request.Settings.Serving.Clone()
	// Keep the complete declared closure so role validation precedes the hash.
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	key := hashEntry{path: path, size: info.Size(), mtime: info.ModTime().UnixNano()}
	waiting := make(chan struct{})
	call := &hashCall{sum: artifact.SHA256, onWait: func() { close(waiting) }}
	call.wg.Add(1)
	fileHashes.mu.Lock()
	delete(fileHashes.done, path)
	fileHashes.inflight[key] = call
	fileHashes.mu.Unlock()
	released := false
	defer func() {
		if !released {
			call.wg.Done()
		}
		fileHashes.mu.Lock()
		delete(fileHashes.inflight, key)
		fileHashes.mu.Unlock()
	}()
	done := make(chan error, 1)
	go func() { done <- f.prepareRuntimeHashes(t, context.Background(), config) }()
	select {
	case <-waiting:
	case err := <-done:
		t.Fatalf("hash did not reach injected wait: %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("runtime hash did not reach injected wait")
	}
	writeServingFixture(t, path, []byte("replacement during runtime hash"))
	call.wg.Done()
	released = true
	select {
	case err := <-done:
		if err == nil || !strings.Contains(err.Error(), "changed during preparation") {
			t.Fatalf("unstable hash error = %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("hash did not finish after release")
	}
}

func TestStrataSettingsCanReplaceInvalidPriorDeclaration(t *testing.T) {
	for _, operation := range []string{"replace", "disable"} {
		t.Run(operation, func(t *testing.T) {
			f := newStrataServingFixture(t)
			p := approveLlamaServing(t, f.llamaServingFixture)
			st := settingsState(t, f.ex)
			st.plat.Runtime.Serving = p.Settings.Serving.Clone()
			st.plat.Runtime.Serving.Models[0].Artifacts[0].Binding.SourcePath = filepath.Join(f.root, "missing-prior-artifact")
			request := f.request
			request.Settings = p.Settings.Clone()
			if operation == "disable" {
				request.Settings.Serving = nil
			}
			preview := previewSettings(t, f.ex, request)
			if len(preview.Errors) != 0 {
				t.Fatalf("valid replacement rejected: %v", preview.Errors)
			}
			if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil }); err != nil {
				t.Fatalf("valid replacement apply: %v", err)
			}
		})
	}
}

func newStrataServingFixture(t *testing.T) strataServingFixture {
	t.Helper()
	ex := settingsExecutor(t, false)
	st := settingsState(t, ex)
	st.manifest.Family = "strata"
	root := t.TempDir()
	source := filepath.Join(root, "config.json")
	data, err := os.ReadFile("testdata/serving/strata-config.json")
	if err != nil {
		t.Fatal(err)
	}
	var config strataServingConfig
	if err := json.Unmarshal(data, &config); err != nil {
		t.Fatal(err)
	}
	// Resolve template paths before JSON encoding so Windows gets clean native
	// separators, including suffixes, with backslashes escaped by the encoder.
	resolvePath := func(value string) string {
		if suffix, ok := strings.CutPrefix(value, "/REPLACE_WITH_ABSOLUTE_PATH"); ok {
			return filepath.Join(root, filepath.FromSlash(suffix))
		}
		return value
	}
	config.Exe = resolvePath(config.Exe)
	config.CWD = resolvePath(config.CWD)
	config.Tokenizer = resolvePath(config.Tokenizer)
	for i, value := range config.Args {
		config.Args[i] = resolvePath(value)
	}
	for i, value := range config.LibDirs {
		config.LibDirs[i] = resolvePath(value)
	}
	data, err = json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	for _, dir := range []string{"pack", "draft", "tokenizer", "lib"} {
		if err := os.Mkdir(filepath.Join(root, dir), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	files := map[string]string{
		"server.py": "runtime:server", "strata": "runtime:native-exe", "profile.bin": "runtime:profile",
		"model-00001-of-00002.gguf": "model:1", "model-00002-of-00002.gguf": "model:2",
		"pack/index.txt": "model:pack-index", "pack/dense.bin": "model:pack-dense", "pack/experts.bin": "model:pack-experts",
		"draft/dense.txt": "draft", "draft/dense.bin": "draft", "draft/experts.bin": "draft",
		"tokenizer/vocab.json": "runtime:tokenizer:vocab.json", "tokenizer/merges.txt": "runtime:tokenizer:merges.txt",
		"tokenizer/token_type.json": "runtime:tokenizer:token_type.json", "tokenizer/chat_template.jinja": "runtime:tokenizer:chat_template.jinja",
		"lib/attested.so": "runtime:lib:attested",
	}
	st.plat.Runtime.Args = []string{filepath.Join(root, "server.py"), "--engine", "strata", "--config", source, "--host", "127.0.0.1", "--port", "{port}", "--lazy"}
	st.plat.Runtime.EditableLaunch = &EditableLaunch{FixedArgs: []string{filepath.Join(root, "server.py")}, Controls: []LaunchControl{{Flags: []string{"--host"}, Value: "{server.host}"}, {Flags: []string{"--port"}, Value: "{server.port}"}}}
	startLog := filepath.Join(root, "starts.jsonl")
	st.plat.Runtime.Env = map[string]string{"FAKE_ADDR": "127.0.0.1:{port}", "FAKE_START_LOG": startLog}
	st.plat.Runtime.Health = nil
	request := settingsRequest(t, ex)
	ram, vram := int64(4096), int64(0)
	model := engines.ServingModel{ModelID: "prepared-model", MaxOutputTokens: 128, Memory: engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram}}
	for _, name := range sortedKeys(files) {
		path := filepath.Join(root, filepath.FromSlash(name))
		writeServingFixture(t, path, []byte("synthetic "+name))
		model.Artifacts = append(model.Artifacts, strataArtifact(t, path, files[name]))
	}
	writeServingFixture(t, source, data)
	model.Artifacts = append(model.Artifacts, strataArtifact(t, source, "runtime:config"), strataArtifact(t, fakeEngineBin, "runtime:python"))
	request.Settings.Serving = &engines.ServingConfig{Version: 1, Models: []engines.ServingModel{model}}
	return strataServingFixture{llamaServingFixture: llamaServingFixture{ex: ex, request: request, source: source, startLog: startLog, root: root}, config: config}
}

func strataArtifact(t *testing.T, path, role string) engines.ServingArtifact {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	binding, err := servingFileBinding(path)
	if err != nil {
		t.Fatal(err)
	}
	return engines.ServingArtifact{Role: role, SHA256: hex.EncodeToString(sum[:]), Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}, Binding: binding}
}

func (f *strataServingFixture) saveConfig(t *testing.T) {
	t.Helper()
	data, err := json.Marshal(f.config)
	if err != nil {
		t.Fatal(err)
	}
	writeServingFixture(t, f.source, data)
	f.bindConfig(t)
}

func (f *strataServingFixture) bindConfig(t *testing.T) {
	t.Helper()
	for i, artifact := range f.request.Settings.Serving.Models[0].Artifacts {
		if artifact.Role == "runtime:config" {
			f.request.Settings.Serving.Models[0].Artifacts[i] = strataArtifact(t, f.source, "runtime:config")
		}
	}
}

func TestStrataServingColdPreviewAndApplyHaveNoEngineCalls(t *testing.T) {
	f := newStrataServingFixture(t)
	var calls atomic.Int32
	f.ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("unexpected engine call")
	})
	p := approveLlamaServing(t, f.llamaServingFixture)
	want := engines.ServingEnvelope{ModelID: "prepared-model", PerSlotContextTokens: 4096, TotalContextTokens: 8192, Slots: 2, MaxOutputTokens: 128, Memory: f.request.Settings.Serving.Models[0].Memory, Placement: []string{"--prefill=512", "--expert-cache=64", "--spec=4"}, Retention: "60", RouterCapacity: 1, NativeSourceDigest: p.Settings.Serving.Models[0].ExpectedNativeSourceDigest}
	if len(p.Serving.Configured) != 1 || !reflect.DeepEqual(p.Serving.Configured[0], want) || len(p.Serving.LaunchValidated) != 0 || len(p.Serving.Observed) != 0 {
		t.Fatalf("cold envelope = %+v", p.Serving)
	}
	if !strings.Contains(p.Serving.Reason, "lib_dirs") || !strings.Contains(p.Serving.Reason, "Python imports") {
		t.Fatalf("unverified reason = %s", p.Serving.Reason)
	}
	assertNoSettingsOverride(t, f.ex)
	assertNoLlamaSpawn(t, f.llamaServingFixture)
	if _, err := os.Stat(filepath.Join(f.ex.baseDir, "serving")); !os.IsNotExist(err) {
		t.Fatalf("preview wrote snapshot: %v", err)
	}
	applyLlamaServing(t, f.llamaServingFixture, p)
	readback, err := f.ex.LaunchSettings("fake")
	if err != nil || readback.ServingReadback.Status != "unverified" || len(readback.ServingReadback.LaunchValidated) != 0 {
		t.Fatalf("cold readback = %+v, %v", readback, err)
	}
	if calls.Load() != 0 {
		t.Fatalf("cold preview/apply/readback made %d engine calls", calls.Load())
	}
}

func TestStrataServingFakeProcessUsesApprovedSnapshot(t *testing.T) {
	f := newStrataServingFixture(t)
	p := approveLlamaServing(t, f.llamaServingFixture)
	applyLlamaServing(t, f.llamaServingFixture, p)
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(f.startLog)
	if err != nil {
		t.Fatal(err)
	}
	var args []string
	if err := json.Unmarshal(bytes.TrimSpace(data), &args); err != nil {
		t.Fatal(err)
	}
	i := slices.Index(args, "--config")
	if i < 0 || i+1 >= len(args) || args[i+1] == f.source || args[0] != filepath.Join(f.root, "server.py") || !slices.Contains(args, "--lazy") {
		t.Fatalf("spawn argv = %v", args)
	}
	snapshot, err := os.ReadFile(args[i+1])
	if err != nil {
		t.Fatal(err)
	}
	var config strataServingConfig
	if err := json.Unmarshal(snapshot, &config); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(config, f.config) {
		t.Fatalf("snapshot changed original targets: %+v", config)
	}
	s, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	if len(s.ServingReadback.LaunchValidated) != 1 || len(s.ServingReadback.Observed) != 0 {
		t.Fatalf("owned readback = %+v", s.ServingReadback)
	}
}

func TestStrataServingParallelSemantics(t *testing.T) {
	for _, tc := range []struct {
		name     string
		parallel int64
		args     []string
		slots    int64
	}{
		{"config supplies batch", 3, nil, 3}, {"native slots alias", 2, []string{"--slots", "2"}, 2}, {"serial config", 1, nil, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			i := slices.Index(f.config.Args, "--batch")
			f.config.Args = append(f.config.Args[:i:i], f.config.Args[i+2:]...)
			f.config.Args = append(f.config.Args, tc.args...)
			f.config.Parallel = tc.parallel
			f.saveConfig(t)
			p := approveLlamaServing(t, f.llamaServingFixture)
			if p.Serving.Configured[0].Slots != tc.slots {
				t.Fatalf("slots = %d", p.Serving.Configured[0].Slots)
			}
		})
	}
}

func TestStrataServingRejectsInvalidPreparedInputs(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*testing.T, *strataServingFixture)
	}{
		{"parallel clamp", func(t *testing.T, f *strataServingFixture) {
			f.config.Parallel = 9
			f.config.Args[slices.Index(f.config.Args, "--batch")+1] = "9"
		}},
		{"parallel missing", func(t *testing.T, f *strataServingFixture) { f.config.Parallel = 0 }},
		{"batch conflict", func(t *testing.T, f *strataServingFixture) { f.config.Args = append(f.config.Args, "--slots", "2") }},
		{"batch mismatch", func(t *testing.T, f *strataServingFixture) { f.config.Parallel = 3 }},
		{"batch one", func(t *testing.T, f *strataServingFixture) {
			f.config.Parallel = 1
			f.config.Args[slices.Index(f.config.Args, "--batch")+1] = "1"
		}},
		{"auto context", func(t *testing.T, f *strataServingFixture) {
			f.config.Args[slices.Index(f.config.Args, "--max-context")+1] = "auto"
		}},
		{"context overflow", func(t *testing.T, f *strataServingFixture) {
			f.config.Args[slices.Index(f.config.Args, "--max-context")+1] = "2147483647"
		}},
		{"output does not fit", func(t *testing.T, f *strataServingFixture) {
			f.request.Settings.Serving.Models[0].MaxOutputTokens = 4096
		}},
		{"no serve", func(t *testing.T, f *strataServingFixture) { f.config.Args = f.config.Args[1:] }},
		{"automatic placement", func(t *testing.T, f *strataServingFixture) {
			f.config.Args[slices.Index(f.config.Args, "--expert-cache")+1] = "auto"
		}},
		{"missing placement", func(t *testing.T, f *strataServingFixture) {
			i := slices.Index(f.config.Args, "--prefill")
			f.config.Args = append(f.config.Args[:i:i], f.config.Args[i+2:]...)
		}},
		{"spec clamp", func(t *testing.T, f *strataServingFixture) {
			f.config.Args[slices.Index(f.config.Args, "--spec")+1] = "9"
		}},
		{"unknown argument", func(t *testing.T, f *strataServingFixture) {
			f.config.Args = append(f.config.Args, "--unknown-file", "/undeclared")
		}},
		{"missing argument value", func(t *testing.T, f *strataServingFixture) {
			f.config.Args = append(f.config.Args, "--vram-reserve-mib")
		}},
		{"relative model", func(t *testing.T, f *strataServingFixture) {
			f.config.Args[slices.Index(f.config.Args, "--native")+1] = "model.gguf"
		}},
		{"nonfirst shard", func(t *testing.T, f *strataServingFixture) {
			f.config.Args[slices.Index(f.config.Args, "--native")+1] = filepath.Join(f.root, "model-00002-of-00002.gguf")
		}},
		{"missing shard", func(t *testing.T, f *strataServingFixture) {
			if err := os.Remove(filepath.Join(f.root, "model-00002-of-00002.gguf")); err != nil {
				t.Fatal(err)
			}
		}},
		{"missing draft", func(t *testing.T, f *strataServingFixture) {
			if err := os.Remove(filepath.Join(f.root, "draft", "dense.bin")); err != nil {
				t.Fatal(err)
			}
		}},
		{"relative library directory", func(t *testing.T, f *strataServingFixture) { f.config.LibDirs = []string{"lib"} }},
		{"missing library directory", func(t *testing.T, f *strataServingFixture) {
			f.config.LibDirs = []string{filepath.Join(f.root, "absent")}
		}},
		{"relative cwd", func(t *testing.T, f *strataServingFixture) { f.config.CWD = "." }},
		{"wrong model", func(t *testing.T, f *strataServingFixture) { f.config.ModelName = "different" }},
		{"native expert indirection", func(t *testing.T, f *strataServingFixture) {
			writeServingFixture(t, filepath.Join(f.root, "pack", "native_experts.txt"), []byte("indirect"))
		}},
		{"undeclared tokenizer", func(t *testing.T, f *strataServingFixture) {
			artifacts := f.request.Settings.Serving.Models[0].Artifacts
			f.request.Settings.Serving.Models[0].Artifacts = slices.DeleteFunc(artifacts, func(a engines.ServingArtifact) bool { return a.Role == "runtime:tokenizer:merges.txt" })
		}},
		{"missing binding", func(t *testing.T, f *strataServingFixture) {
			f.request.Settings.Serving.Models[0].Artifacts[0].Binding = nil
		}},
		{"duplicate binding", func(t *testing.T, f *strataServingFixture) {
			m := &f.request.Settings.Serving.Models[0]
			a := m.Artifacts[0]
			a.Role = "runtime:duplicate"
			m.Artifacts = append(m.Artifacts, a)
		}},
		{"wrong runtime role", func(t *testing.T, f *strataServingFixture) {
			for i, a := range f.request.Settings.Serving.Models[0].Artifacts {
				if a.Role == "runtime:server" {
					f.request.Settings.Serving.Models[0].Artifacts[i].Role = "model:wrong"
				}
			}
		}},
		{"wrong full runtime hash", func(t *testing.T, f *strataServingFixture) {
			for i, a := range f.request.Settings.Serving.Models[0].Artifacts {
				if a.Role == "runtime:server" {
					f.request.Settings.Serving.Models[0].Artifacts[i].SHA256 = strings.Repeat("a", 64)
				}
			}
		}},
		{"extra model", func(t *testing.T, f *strataServingFixture) {
			m := &f.request.Settings.Serving.Models[0]
			a := strataArtifact(t, filepath.Join(f.root, "server.py"), "model:extra")
			a.Binding.SourcePath = filepath.Join(f.root, "extra.gguf")
			writeServingFixture(t, a.Binding.SourcePath, []byte("extra"))
			a = strataArtifact(t, a.Binding.SourcePath, a.Role)
			m.Artifacts = append(m.Artifacts, a)
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			old, err := f.ex.LaunchSettings("fake")
			if err != nil {
				t.Fatal(err)
			}
			tc.mutate(t, &f)
			f.saveConfig(t)
			p := previewSettings(t, f.ex, f.request)
			if len(p.Errors) == 0 {
				t.Fatal("invalid prepared input accepted")
			}
			if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: f.request.Settings}, func() error { return nil }); err == nil {
				t.Fatal("invalid apply accepted")
			}
			after, err := f.ex.LaunchSettings("fake")
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(old, after) {
				t.Fatal("invalid request changed prior settings")
			}
			assertNoLlamaSpawn(t, f.llamaServingFixture)
			assertNoSettingsOverride(t, f.ex)
		})
	}
}

func TestStrataServingRejectsChangedApprovedFiles(t *testing.T) {
	for _, name := range []string{"profile.bin", "server.py", "strata", "lib/attested.so", "model-00002-of-00002.gguf", "pack/dense.bin", "draft/dense.bin", "tokenizer/chat_template.jinja", "config.json"} {
		t.Run(name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			p := approveLlamaServing(t, f.llamaServingFixture)
			applyLlamaServing(t, f.llamaServingFixture, p)
			before, err := os.ReadFile(filepath.Join(f.ex.overrideDir, "fake.json"))
			if err != nil {
				t.Fatal(err)
			}
			writeServingFixture(t, filepath.Join(f.root, filepath.FromSlash(name)), []byte("changed approved dependency"))
			if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: p.Settings}, func() error { return nil }); err == nil {
				t.Fatal("changed artifact applied")
			}
			if err := f.ex.Start(context.Background(), "fake"); err == nil {
				t.Fatal("changed artifact started")
			}
			after, err := os.ReadFile(filepath.Join(f.ex.overrideDir, "fake.json"))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(before, after) {
				t.Fatal("invalid apply changed saved settings")
			}
			assertNoLlamaSpawn(t, f.llamaServingFixture)
		})
	}
}

func TestStrataServingRejectsInputChangeBeforeSpawn(t *testing.T) {
	f := newStrataServingFixture(t)
	p := approveLlamaServing(t, f.llamaServingFixture)
	applyLlamaServing(t, f.llamaServingFixture, p)
	st := settingsState(t, f.ex)
	st.plat.Runtime.servingSnapshot.Release()
	st.plat.Runtime.servingSnapshot = nil
	vars := st.pathVars()
	vars["host"] = "127.0.0.1"
	// Use the same canonical port variables as lifecycle, with an injected
	// snapshot writer to change a selected source after compilation.
	command, err := launchForState(st, st.port)
	if err != nil {
		t.Fatal(err)
	}
	i := slices.Index(command.Args, "--port")
	vars["port"] = command.Args[i+1]
	err = f.ex.bringUpProcessWithSnapshot(context.Background(), st, "fake", st.plat.Runtime, st.port, vars, func(root string, data []byte) (string, func(), error) {
		path, release, err := prepareServingSnapshot(root, data)
		if err == nil {
			writeServingFixture(t, filepath.Join(f.root, "profile.bin"), []byte("changed after compile"))
		}
		return path, release, err
	})
	if err == nil || !strings.Contains(err.Error(), "recheck") {
		t.Fatalf("pre-spawn error = %v", err)
	}
	assertNoLlamaSpawn(t, f.llamaServingFixture)
}

func TestStrataServingRejectsServerLaunchModes(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*launchCommand)
	}{
		{"relative Python", func(c *launchCommand) { c.Bin = "python" }},
		{"Python switch", func(c *launchCommand) { c.Args = append([]string{"-m"}, c.Args...) }},
		{"duplicate server option", func(c *launchCommand) { c.Args = append(c.Args, "--lazy") }},
		{"missing server value", func(c *launchCommand) { c.Args = []string{c.Args[0], "--config"} }},
		{"mock engine", func(c *launchCommand) { c.Args[slices.Index(c.Args, "--engine")+1] = "mock" }},
		{"no lazy", func(c *launchCommand) { c.Args = c.Args[:len(c.Args)-1] }},
		{"unknown option", func(c *launchCommand) { c.Args = append(c.Args, "--gpu", "0") }},
		{"invalid port", func(c *launchCommand) { c.Args[slices.Index(c.Args, "--port")+1] = "0" }},
		{"native environment", func(c *launchCommand) { c.Env["STRATA_BATCH_MTP"] = "1" }},
		{"Python environment", func(c *launchCommand) { c.Env["PYTHONPATH"] = "/unapproved" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			c, err := launchForState(settingsState(t, f.ex), f.request.Settings.ServerPort)
			if err != nil {
				t.Fatal(err)
			}
			tc.mutate(&c)
			if _, err := strataLaunchSource(c); err == nil {
				t.Fatal("invalid server launch accepted")
			}
			assertNoLlamaSpawn(t, f.llamaServingFixture)
		})
	}
}

func TestStrataServingRejectsConfigGrammar(t *testing.T) {
	for _, data := range []string{`[]`, `{"parallel":2,"parallel":3}`, `{"parallel":null}`, `{"parallel":2.5}`, `{"memory_policy":{}}`, `{} {}`} {
		t.Run(data, func(t *testing.T) {
			if _, err := parseStrataServingConfig([]byte(data)); err == nil {
				t.Fatal("ambiguous config accepted")
			}
		})
	}
}

func TestStrataServingConfigDigestRequiresApproval(t *testing.T) {
	f := newStrataServingFixture(t)
	p := approveLlamaServing(t, f.llamaServingFixture)
	st := settingsState(t, f.ex)
	c, err := launchForState(st, f.request.Settings.ServerPort)
	if err != nil {
		t.Fatal(err)
	}
	rt := st.plat.Runtime
	rt.Serving = f.request.Settings.Serving
	if _, err := compileStrataServing(c, rt, true); err == nil {
		t.Fatal("unreviewed digest applied")
	}
	rt.Serving = p.Settings.Serving.Clone()
	rt.Serving.Models[0].ExpectedNativeSourceDigest = strings.Repeat("a", 64)
	if _, err := compileStrataServing(c, rt, false); err == nil {
		t.Fatal("conflicting digest accepted")
	}
}

func TestStrataServingCommandLaunchRejectedBeforeEffects(t *testing.T) {
	f := newStrataServingFixture(t)
	st := settingsState(t, f.ex)
	rt := st.plat.Runtime
	rt.Serving = f.request.Settings.Serving
	rt.Mode = "command"
	if err := f.ex.bringUp(context.Background(), st, "fake", rt, st.port, st.pathVars()); err == nil || !strings.Contains(err.Error(), "requires an owned process") {
		t.Fatalf("command mode error = %v", err)
	}
	assertNoLlamaSpawn(t, f.llamaServingFixture)
}

func TestStrataDirectoryRequiresCleanNativeSpelling(t *testing.T) {
	root := t.TempDir()
	if err := strataDirectory(root); err != nil {
		t.Fatalf("native directory rejected: %v", err)
	}
	reject := func(name, path string) {
		t.Run(name, func(t *testing.T) {
			if err := strataDirectory(path); err == nil || !strings.Contains(err.Error(), "absolute, clean, existing directories in native platform form") {
				t.Fatalf("directory spelling error = %v", err)
			}
		})
	}
	reject("trailing separator", root+string(filepath.Separator))
	reject("dot component", root+string(filepath.Separator)+".")
	if runtime.GOOS == "windows" {
		reject("forward slashes", filepath.ToSlash(root))
	}
}

func TestStrataServingRejectsAmbiguousDependencyPaths(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*strataServingFixture)
		want   string
	}{
		{"unclean native path", func(f *strataServingFixture) {
			f.config.Exe = f.root + string(filepath.Separator) + filepath.FromSlash("lib/../strata")
		}, "absolute and unambiguous"},
		{"conflicting file roles", func(f *strataServingFixture) { f.config.Exe = filepath.Join(f.root, "server.py") }, "ambiguous artifact roles"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			tc.change(&f)
			f.saveConfig(t)
			p := previewSettings(t, f.ex, f.request)
			if len(p.Errors) != 1 || !strings.Contains(p.Errors["serving"], tc.want) {
				t.Fatalf("dependency error = %+v; want %s", p.Errors, tc.want)
			}
			assertNoLlamaSpawn(t, f.llamaServingFixture)
			assertNoSettingsOverride(t, f.ex)
		})
	}
}

func TestStrataServingRejectsMultipleDeclaredModels(t *testing.T) {
	f := newStrataServingFixture(t)
	other := f.request.Settings.Serving.Models[0]
	other.ModelID = "second-model"
	f.request.Settings.Serving.Models = append(f.request.Settings.Serving.Models, other)
	p := previewSettings(t, f.ex, f.request)
	if len(p.Errors) == 0 {
		t.Fatal("multiple Strata model declarations accepted")
	}
	assertNoLlamaSpawn(t, f.llamaServingFixture)
	assertNoSettingsOverride(t, f.ex)
}

func TestStrataServingRecheckRejectsChangedPreparedEnvelope(t *testing.T) {
	f := newStrataServingFixture(t)
	p := approveLlamaServing(t, f.llamaServingFixture)
	applyLlamaServing(t, f.llamaServingFixture, p)
	st := settingsState(t, f.ex)
	c, err := launchForState(st, f.request.Settings.ServerPort)
	if err != nil {
		t.Fatal(err)
	}
	rt := st.plat.Runtime
	rt.Serving = p.Settings.Serving.Clone()
	prepared, err := compileStrataServing(c, rt, true)
	if err != nil {
		t.Fatal(err)
	}
	prepared.Envelopes[0].Slots++
	if err := recheckPreparedServing(st, c, rt, prepared); err == nil {
		t.Fatal("changed prepared envelope accepted")
	}
	assertNoLlamaSpawn(t, f.llamaServingFixture)
}

func TestStrataServingRejectsNonObjectAtConfigBoundary(t *testing.T) {
	if _, err := parseStrataServingConfig([]byte(`[]`)); err == nil || !strings.Contains(err.Error(), "must be a JSON object") {
		t.Fatalf("non-object config error = %v", err)
	}
}

func TestStrataServingRejectsOversizedSourceBeforeParsing(t *testing.T) {
	f := newStrataServingFixture(t)
	writeServingFixture(t, f.source, bytes.Repeat([]byte(" "), servingSmallFileLimit+1))
	p := previewSettings(t, f.ex, f.request)
	if !strings.Contains(p.Errors["serving"], "missing, unstable or oversized") {
		t.Fatalf("source size error = %+v", p.Errors)
	}
	assertNoLlamaSpawn(t, f.llamaServingFixture)
	assertNoSettingsOverride(t, f.ex)
}

func TestStrataServingRejectsOversizedCanonicalSnapshot(t *testing.T) {
	f := newStrataServingFixture(t)
	dir := filepath.Join(f.root, strings.Repeat("&", 32))
	if err := os.Mkdir(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	// JSON HTML escaping can expand a valid bounded source during canonicalization.
	f.config.LibDirs = slices.Repeat([]string{dir}, servingSmallFileLimit/(len(dir)+3)*3/4)
	var source bytes.Buffer
	encoder := json.NewEncoder(&source)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(f.config); err != nil {
		t.Fatal(err)
	}
	canonical, err := json.Marshal(f.config)
	if err != nil {
		t.Fatal(err)
	}
	if source.Len() > servingSmallFileLimit || len(canonical) <= servingSmallFileLimit {
		t.Fatalf("fixture sizes source=%d canonical=%d", source.Len(), len(canonical))
	}
	writeServingFixture(t, f.source, source.Bytes())
	f.bindConfig(t)
	p := previewSettings(t, f.ex, f.request)
	if !strings.Contains(p.Errors["serving"], "bounded Strata config snapshot") {
		t.Fatalf("snapshot size error = %+v", p.Errors)
	}
	assertNoLlamaSpawn(t, f.llamaServingFixture)
	assertNoSettingsOverride(t, f.ex)
}
