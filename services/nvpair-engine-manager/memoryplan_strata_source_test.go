// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
)

func preparedStrataSourceFixture(t *testing.T) (strataServingFixture, *engineState) {
	t.Helper()
	f := newStrataServingFixture(t)
	st := settingsState(t, f.ex)
	interpreter, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	gpu := int64(0)
	f.config.GPU, f.config.Parallel, f.config.LibDirs = &gpu, 1, []string{}
	f.config.Env = map[string]string{"STRATA_FILE_CACHE_MIB": "0", "STRATA_RESIDENT_HEADROOM_GIB": "0"}
	f.config.Args = []string{"--max-context", "8192", "--kv", "int8", "--prefill", "1024", "--native", filepath.Join(f.root, "model-00001-of-00002.gguf"), "--pack", filepath.Join(f.root, "pack"), "--expert-profile", filepath.Join(f.root, "profile.bin"), "--resident-budget-gib", "1", "--vram-reserve-mib", "1", "--expert-cache", "2", "--kv-resident", "1", "--spec", "2", "--no-prefill-borrow"}
	missingKey := filepath.Join(f.root, "missing.key")
	f.config.APIKeyFile = &missingKey
	if err := os.Remove(filepath.Join(f.root, "pack", "experts.bin")); err != nil {
		t.Fatal(err)
	}
	table := filepath.Join(f.root, "pack", "native_experts.txt")
	writeServingFixture(t, table, []byte("# strata native experts v3: fixture\n0 0 0 0 0 0 0 0 model-00002-of-00002.gguf\n"))
	writeServingFixture(t, f.config.Exe, []byte("inert\x00STRATA_BUILD/1:{\"schema_version\":1,\"version\":\"fixture\",\"backend\":\"cuda\",\"commit\":null,\"dirty\":null,\"cuda\":null}\n"))
	header := make([]byte, 32)
	copy(header, "GGUF")
	binary.LittleEndian.PutUint32(header[4:], 3)
	for _, name := range []string{"model-00001-of-00002.gguf", "model-00002-of-00002.gguf"} {
		writeServingFixture(t, filepath.Join(f.root, name), header)
	}
	artifacts := f.request.Settings.Serving.Models[0].Artifacts
	kept := make([]engines.ServingArtifact, 0, len(artifacts))
	for _, artifact := range artifacts {
		if artifact.Role == "draft" || strings.HasPrefix(artifact.Role, "runtime:lib:") || artifact.Role == "runtime:server" || artifact.Role == "runtime:python" || artifact.Binding.SourcePath == filepath.Join(f.root, "pack", "experts.bin") {
			continue
		}
		kept = append(kept, strataArtifact(t, artifact.Binding.SourcePath, artifact.Role))
	}
	for _, name := range strataMemoryPlanModules {
		path := filepath.Join(f.root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		writeServingFixture(t, path, []byte("inert module "+name))
		role := "runtime:module:" + name
		if name == "serve/server.py" {
			role = "runtime:server"
		}
		kept = append(kept, strataArtifact(t, path, role))
	}
	kept = append(kept, strataArtifact(t, table, "model"), strataArtifact(t, interpreter, "runtime:python"))
	f.request.Settings.Serving.Models[0].Artifacts = kept
	f.saveConfig(t)
	server := filepath.Join(f.root, "serve", "server.py")
	st.binPath = interpreter
	st.plat.Runtime.Args[0] = server
	st.plat.Runtime.EditableLaunch.FixedArgs[0] = server
	st.plat.Runtime.Serving = f.request.Settings.Serving
	st.manifest.MemoryPlan = &memoryPlanAdapter{Adapter: "strata"}
	st.plat.MemoryPlan = &memoryPlanCommand{Command: []string{interpreter, "-B", "-s", "-m", "serve.server", "--estimate-memory", "--memory-plan-request", "{memory_plan_request}"}, WorkingDirectory: f.root}
	command, err := launchForState(st, st.plat.Runtime.Port)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := deriveStrataServingInputExpected(context.Background(), command, st.plat.Runtime, strataServingAccess{}, nil); err != nil {
		t.Fatalf("prepared fixture admission: %v", err)
	}
	return f, st
}

func TestMemoryPlanStrataSourcePreparedQueryUsesFiniteCredentialFreeInput(t *testing.T) {
	f, st := preparedStrataSourceFixture(t)
	access := strataServingAccess{readKey: func(string, []engines.ServingArtifact) (string, *strataKeyIdentity, error) {
		t.Fatal("pricing attempted a key read")
		return "", nil, errors.New("refused")
	}}
	source, err := prepareStrataMemoryPlanSource(context.Background(), st, access)
	if err != nil {
		t.Fatal(err)
	}
	request, err := json.Marshal(source.binding.Request)
	if err != nil {
		t.Fatal(err)
	}
	var got strataMemoryPlanRequest
	if err := strataPlanDecode(request, &got); err != nil {
		t.Fatal(err)
	}
	if got.Model.ID != "prepared-model" || got.Config.Context != 8192 || got.Config.KVType != "int8" || got.Config.MicrobatchSize.Value != 1024 || got.Budget != nil || len(got.Extensions.Strata.Allowances) != 0 || bytes.Contains(request, []byte("api_key")) || bytes.Contains(request, []byte("missing.key")) {
		t.Fatalf("unexpected finite request: %s", request)
	}
	// The existing Go subprocess helper parses this exact request and emits its
	// inert default {ok:true}; it does not execute Python or a native engine.
	if _, err := source.query(context.Background(), f.ex, st, access); !errors.Is(err, errStrataMemoryPlanResponse) {
		t.Fatalf("fake CLI did not reach report validation: %v", err)
	}
}

func TestMemoryPlanStrataSourcePreparedQueryRechecksModuleBytes(t *testing.T) {
	f, st := preparedStrataSourceFixture(t)
	source, err := prepareStrataMemoryPlanSource(context.Background(), st, strataServingAccess{})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(f.root, "serve", "memory_plan.py")
	file := source.input.launch.evidence.artifacts[path]
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data[0] ^= 1
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, file.info.ModTime(), file.info.ModTime()); err != nil {
		t.Fatal(err)
	}
	if _, err := source.query(context.Background(), f.ex, st, strataServingAccess{}); !errors.Is(err, errStrataMemoryPlanDrift) {
		t.Fatalf("same-stat module rewrite escaped source recheck: %v", err)
	}
}

func TestMemoryPlanStrataSourcePreparedQueryRechecksSelection(t *testing.T) {
	f, st := preparedStrataSourceFixture(t)
	source, err := prepareStrataMemoryPlanSource(context.Background(), st, strataServingAccess{})
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.plat.MemoryPlan.WorkingDirectory = t.TempDir()
	st.mu.Unlock()
	if _, err := source.query(context.Background(), f.ex, st, strataServingAccess{}); !errors.Is(err, errStrataMemoryPlanDrift) {
		t.Fatalf("selected invocation drift escaped recheck: %v", err)
	}
}

func TestMemoryPlanStrataSourcePreparedSelectionRefusals(t *testing.T) {
	test := func(name string, change func(*testing.T, *engineState)) {
		t.Run(name, func(t *testing.T) {
			_, st := preparedStrataSourceFixture(t)
			change(t, st)
			if _, err := prepareStrataMemoryPlanSource(context.Background(), st, strataServingAccess{}); err == nil {
				t.Fatal("accepted an unbound pricing selection")
			}
		})
	}
	test("undeclared adapter", func(t *testing.T, st *engineState) { st.manifest.MemoryPlan = nil })
	test("different interpreter", func(t *testing.T, st *engineState) {
		st.plat.MemoryPlan.Command[0] = filepath.Join(t.TempDir(), "other")
	})
	test("different package directory", func(t *testing.T, st *engineState) { st.plat.MemoryPlan.WorkingDirectory = t.TempDir() })
	test("missing module declaration", func(t *testing.T, st *engineState) {
		model := &st.plat.Runtime.Serving.Models[0]
		for i, artifact := range model.Artifacts {
			if artifact.Binding.SourcePath == filepath.Join(st.plat.MemoryPlan.WorkingDirectory, "serve", "memory_plan.py") {
				model.Artifacts = append(model.Artifacts[:i:i], model.Artifacts[i+1:]...)
				return
			}
		}
		t.Fatal("fixture is missing the selected module")
	})
}

func TestMemoryPlanStrataSourceQueryOwnsContentAdmission(t *testing.T) {
	f, st := preparedStrataSourceFixture(t)
	release, err := f.ex.downloads.acquireWrite("fake")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.ex.queryStrataMemoryPlan(context.Background(), st); !errors.Is(err, errStrataMemoryPlan) {
		release()
		t.Fatalf("pricing bypassed the model content writer: %v", err)
	}
	release()
	if _, err := f.ex.queryStrataMemoryPlan(context.Background(), st); !errors.Is(err, errStrataMemoryPlanResponse) {
		t.Fatalf("admitted source did not reach the inert fake report: %v", err)
	}
}

func TestMemoryPlanStrataSourcePreparedQueryChecksDriftBeforeDecoding(t *testing.T) {
	f, st := preparedStrataSourceFixture(t)
	source, err := prepareStrataMemoryPlanSource(context.Background(), st, strataServingAccess{})
	if err != nil {
		t.Fatal(err)
	}
	request, err := json.Marshal(source.binding.Request)
	if err != nil {
		t.Fatal(err)
	}
	response, err := f.ex.runMemoryPlanCommand(context.Background(), source.invocation, request)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(f.root, "serve", "memory_plan.py")
	file := source.input.launch.evidence.artifacts[path]
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data[0] ^= 1
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, file.info.ModTime(), file.info.ModTime()); err != nil {
		t.Fatal(err)
	}
	if _, err := source.complete(context.Background(), st, response, strataServingAccess{}); !errors.Is(err, errStrataMemoryPlanDrift) {
		t.Fatalf("source drift was not rejected before report decoding: %v", err)
	}
}

func TestMemoryPlanStrataSourceModuleDigestDetectsSameStatBytes(t *testing.T) {
	f, st := preparedStrataSourceFixture(t)
	source, err := prepareStrataMemoryPlanSource(context.Background(), st, strataServingAccess{})
	if err != nil {
		t.Fatal(err)
	}
	before, err := strataMemoryPlanModuleDigest(context.Background(), f.root, source.input.launch.evidence.artifacts)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(f.root, "serve", "memory_plan.py")
	file := source.input.launch.evidence.artifacts[path]
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data[0] ^= 1
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, file.info.ModTime(), file.info.ModTime()); err != nil {
		t.Fatal(err)
	}
	after, err := strataMemoryPlanModuleDigest(context.Background(), f.root, source.input.launch.evidence.artifacts)
	if err != nil || after == before {
		t.Fatalf("same-stat module hash: before %s, after %s, error %v", before, after, err)
	}
}

func strataSourceIdentityFixture(t *testing.T) *strataServingInput {
	t.Helper()
	root := t.TempDir()
	pack := filepath.Join(root, "pack")
	if err := os.Mkdir(pack, 0700); err != nil {
		t.Fatal(err)
	}
	gpu := int64(0)
	input := &strataServingInput{
		launch:  &llamaServingLaunch{Config: &engines.ServingConfig{Models: []engines.ServingModel{{ModelID: "example"}}}, evidence: &llamaServingEvidence{artifacts: map[string]sourceEvidence{}}},
		config:  strataServingConfig{Exe: filepath.Join(root, "native"), CWD: root, GPU: &gpu, Parallel: 1, Env: map[string]string{"STRATA_FILE_CACHE_MIB": "0", "STRATA_RESIDENT_HEADROOM_GIB": "0"}},
		options: map[string]string{"--max-context": "4096", "--kv": "fp16", "--prefill": "128", "--native": filepath.Join(root, "model-00001-of-00002.gguf"), "--pack": pack, "--expert-profile": filepath.Join(root, "profile.bin"), "--resident-budget-gib": "1", "--vram-reserve-mib": "1", "--expert-cache": "4", "--kv-resident": "0", "--spec": "2", "--no-prefill-borrow": "true"},
	}
	build := []byte(`{"schema_version":1,"version":"fixture","backend":"cuda","commit":null,"dirty":null,"cuda":null}`)
	header := make([]byte, 32)
	copy(header, "GGUF")
	binary.LittleEndian.PutUint32(header[4:], 3)
	files := []struct {
		path, role string
		data       []byte
	}{
		{input.config.Exe, "runtime:native-exe", append(append([]byte("inert\x00STRATA_BUILD/1:"), build...), '\n')},
		{input.options["--expert-profile"], "runtime:profile", []byte("profile metadata")},
		{filepath.Join(pack, "index.txt"), "model", []byte("index metadata")},
		{filepath.Join(pack, "native_experts.txt"), "model", []byte("# strata native experts v3: fixture\n0 0 0 0 0 0 0 0 model-00002-of-00002.gguf\n")},
		{filepath.Join(pack, "dense.bin"), "model", []byte("model payload")},
		{input.options["--native"], "model", header},
		{filepath.Join(root, "model-00002-of-00002.gguf"), "model", header},
	}
	for _, file := range files {
		if err := os.WriteFile(file.path, file.data, 0600); err != nil {
			t.Fatal(err)
		}
		input.launch.Config.Models[0].Artifacts = append(input.launch.Config.Models[0].Artifacts, strataArtifact(t, file.path, file.role))
		evidence, err := captureSource(file.path, false)
		if err != nil {
			t.Fatal(err)
		}
		input.launch.evidence.artifacts[file.path] = evidence
	}
	return input
}

func TestMemoryPlanStrataSourceIndependentIdentity(t *testing.T) {
	input := strataSourceIdentityFixture(t)
	binding, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), servingFileAccess{})
	if err != nil {
		t.Fatal(err)
	}
	wantPaths := []string{input.options["--expert-profile"], filepath.Join(input.options["--pack"], "index.txt"), filepath.Join(input.options["--pack"], "native_experts.txt"), filepath.Join(input.options["--pack"], "dense.bin"), input.options["--native"], filepath.Join(input.config.CWD, "model-00002-of-00002.gguf")}
	if len(binding.Inputs) != len(wantPaths) || binding.Build.Version != "fixture" || binding.Build.Backend != "cuda" || binding.RuntimeSourceSHA256 != strings.Repeat("a", 64) {
		t.Fatalf("independent identity binding = %+v", binding)
	}
	for i, identity := range binding.Inputs {
		if identity.Path != wantPaths[i] || len(identity.FileID) != 2 {
			t.Fatalf("input %d identity = %+v", i, identity)
		}
		data, err := os.ReadFile(identity.Path)
		if err != nil {
			t.Fatal(err)
		}
		if i < 3 {
			sum := sha256.Sum256(data)
			if identity.ContentSHA256 == nil || *identity.ContentSHA256 != hex.EncodeToString(sum[:]) || identity.MetadataBytes != nil {
				t.Fatalf("small metadata identity = %+v", identity)
			}
		} else if i >= 4 {
			sum := sha256.Sum256(data[:24])
			if identity.ContentSHA256 != nil || identity.MetadataBytes == nil || *identity.MetadataBytes != 24 || identity.MetadataSHA256 == nil || *identity.MetadataSHA256 != hex.EncodeToString(sum[:]) {
				t.Fatalf("GGUF header identity = %+v", identity)
			}
		} else if identity.ContentSHA256 != nil || identity.MetadataSHA256 != nil {
			t.Fatalf("model payload was promoted to content identity: %+v", identity)
		}
	}
	native, err := os.ReadFile(input.config.Exe)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(native)
	if binding.ExecutableSHA256 != hex.EncodeToString(sum[:]) {
		t.Fatal("native executable identity differs from selected inert bytes")
	}
}

func TestMemoryPlanStrataSourceRefusesIdentityDrift(t *testing.T) {
	for _, name := range []string{"native", "profile.bin", "pack/index.txt", "pack/native_experts.txt", "pack/dense.bin", "model-00001-of-00002.gguf"} {
		t.Run(name, func(t *testing.T) {
			input := strataSourceIdentityFixture(t)
			path := filepath.Join(input.config.CWD, name)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), servingFileAccess{}); err == nil {
				t.Fatal("replacement passed original identity evidence")
			}
		})
	}
}

func TestMemoryPlanStrataSourceRejectsAmbiguousBuild(t *testing.T) {
	input := strataSourceIdentityFixture(t)
	data, err := os.ReadFile(input.config.Exe)
	if err != nil {
		t.Fatal(err)
	}
	data = append(data, data...)
	if err := os.WriteFile(input.config.Exe, data, 0600); err != nil {
		t.Fatal(err)
	}
	input.launch.Config.Models[0].Artifacts[0] = strataArtifact(t, input.config.Exe, "runtime:native-exe")
	file, err := captureSource(input.config.Exe, false)
	if err != nil {
		t.Fatal(err)
	}
	input.launch.evidence.artifacts[input.config.Exe] = file
	if _, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), servingFileAccess{}); err == nil {
		t.Fatal("accepted multiple embedded BUILD records")
	}
}

func TestMemoryPlanStrataSourceFindsBuildAcrossScanBoundary(t *testing.T) {
	input := strataSourceIdentityFixture(t)
	data, err := os.ReadFile(input.config.Exe)
	if err != nil {
		t.Fatal(err)
	}
	marker := bytes.Index(data, []byte("STRATA_BUILD/1:"))
	data = append(bytes.Repeat([]byte{0}, (64<<10)-marker-5), data...)
	rebindStrataSourceFixtureFile(t, input, input.config.Exe, data)
	binding, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), servingFileAccess{})
	if err != nil || binding.Build.Version != "fixture" {
		t.Fatalf("embedded BUILD crossing scan boundary: %+v, %v", binding.Build, err)
	}
}

func TestMemoryPlanStrataSourceHashesTensorDirectoryWithoutPayload(t *testing.T) {
	input := strataSourceIdentityFixture(t)
	data := make([]byte, 68)
	copy(data, "GGUF")
	binary.LittleEndian.PutUint32(data[4:], 3)
	binary.LittleEndian.PutUint64(data[8:], 1)
	binary.LittleEndian.PutUint64(data[24:], 1)
	data[32] = 'w'
	binary.LittleEndian.PutUint32(data[33:], 1)
	binary.LittleEndian.PutUint64(data[37:], 1)
	// F32, relative offset zero, followed by alignment padding and tensor data.
	copy(data[64:], []byte{1, 2, 3, 4})
	path := input.options["--native"]
	rebindStrataSourceFixtureFile(t, input, path, data)
	binding, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), servingFileAccess{})
	if err != nil {
		t.Fatal(err)
	}
	for _, identity := range binding.Inputs {
		if identity.Path == path {
			sum := sha256.Sum256(data[:57])
			if identity.MetadataBytes == nil || *identity.MetadataBytes != 57 || identity.MetadataSHA256 == nil || *identity.MetadataSHA256 != hex.EncodeToString(sum[:]) || identity.ContentSHA256 != nil {
				t.Fatalf("header/directory boundary = %+v", identity)
			}
			return
		}
	}
	t.Fatal("selected native shard is absent")
}

func rebindStrataSourceFixtureFile(t *testing.T, input *strataServingInput, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	for i, artifact := range input.launch.Config.Models[0].Artifacts {
		if artifact.Binding.SourcePath == path {
			input.launch.Config.Models[0].Artifacts[i] = strataArtifact(t, path, artifact.Role)
		}
	}
	file, err := captureSource(path, false)
	if err != nil {
		t.Fatal(err)
	}
	input.launch.evidence.artifacts[path] = file
}

func TestMemoryPlanStrataSourceRefusesUnadmittedPathBeforeReading(t *testing.T) {
	input := strataSourceIdentityFixture(t)
	path := filepath.Join(input.options["--pack"], "embd.bin")
	if err := os.WriteFile(path, []byte("unadmitted content"), 0600); err != nil {
		t.Fatal(err)
	}
	readUnadmitted := false
	access := servingFileAccess{open: func(openPath string) (servingArtifactFile, error) {
		if openPath == path {
			readUnadmitted = true
		}
		return os.Open(openPath)
	}}
	if _, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), access); err == nil || readUnadmitted {
		t.Fatalf("unadmitted path: read=%v, error=%v", readUnadmitted, err)
	}
}

func TestMemoryPlanStrataSourceRefusesWrongSplitFamily(t *testing.T) {
	input := strataSourceIdentityFixture(t)
	input.options["--native"] = filepath.Join(input.config.CWD, "model-00002-of-00002.gguf")
	if _, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), servingFileAccess{}); err == nil {
		t.Fatal("accepted a non-first split shard")
	}
}

func TestMemoryPlanStrataSourceCheckedReaderRefusesReplacement(t *testing.T) {
	input := strataSourceIdentityFixture(t)
	path := input.options["--expert-profile"]
	file := input.launch.evidence.artifacts[path]
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	replacement := filepath.Join(input.config.CWD, "replacement")
	if err := os.WriteFile(replacement, data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(replacement, file.info.ModTime(), file.info.ModTime()); err != nil {
		t.Fatal(err)
	}
	swapped := false
	access := servingFileAccess{open: func(openPath string) (servingArtifactFile, error) {
		if openPath == path && !swapped {
			swapped = true
			if err := os.Rename(replacement, path); err != nil {
				return nil, err
			}
		}
		return os.Open(openPath)
	}}
	if _, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), access); err == nil || !swapped {
		t.Fatalf("replacement during checked open: swapped=%v, error=%v", swapped, err)
	}
}

func TestMemoryPlanStrataSourceRejectsNativeBytesBehindWarmHash(t *testing.T) {
	input := strataSourceIdentityFixture(t)
	if _, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), servingFileAccess{}); err != nil {
		t.Fatal(err)
	}
	file := input.launch.evidence.artifacts[input.config.Exe]
	data, err := os.ReadFile(input.config.Exe)
	if err != nil {
		t.Fatal(err)
	}
	data[0] ^= 1 // Preserve the embedded BUILD while changing other executable bytes.
	if err := os.WriteFile(input.config.Exe, data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(input.config.Exe, file.info.ModTime(), file.info.ModTime()); err != nil {
		t.Fatal(err)
	}
	if _, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), servingFileAccess{}); err == nil {
		t.Fatal("accepted changed native bytes behind a warm size/mtime identity cache")
	}
}

func TestMemoryPlanStrataSourceSymlinkRetainsSelectedTarget(t *testing.T) {
	input := strataSourceIdentityFixture(t)
	path := input.options["--expert-profile"]
	target := filepath.Join(input.config.CWD, "profile-target")
	if err := os.Rename(path, target); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	rebindStrataSourceFixtureFile(t, input, path, data)
	binding, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), servingFileAccess{})
	if err != nil || binding.Inputs[0].Path != path {
		t.Fatalf("selected symlink identity: %+v, %v", binding.Inputs, err)
	}
	other := filepath.Join(input.config.CWD, "other-profile-target")
	if err := os.WriteFile(other, data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(other, path); err != nil {
		t.Fatal(err)
	}
	if _, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), servingFileAccess{}); err == nil {
		t.Fatal("retargeted symlink passed the selected source identity")
	}
}

func TestMemoryPlanStrataSourceRejectsUnsafeExpertTable(t *testing.T) {
	input := strataSourceIdentityFixture(t)
	path := filepath.Join(input.options["--pack"], "native_experts.txt")
	rebindStrataSourceFixtureFile(t, input, path, []byte("# strata native experts v3: fixture\n0 0 0 0 0 0 0 0 ../outside.gguf\n"))
	if _, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), servingFileAccess{}); err == nil {
		t.Fatal("accepted unsafe expert-table content")
	}
}

func TestMemoryPlanStrataSourceIndependentBindingRejectsExtraResponsePath(t *testing.T) {
	input := strataSourceIdentityFixture(t)
	binding, err := strataMemoryPlanIdentityBinding(context.Background(), input, strings.Repeat("a", 64), servingFileAccess{})
	if err != nil {
		t.Fatal(err)
	}
	data := strataPlanFixture(t, "explicit-not-evaluated")
	var response strataMemoryPlanResponse
	if err := json.Unmarshal(data, &response); err != nil {
		t.Fatal(err)
	}
	response.Engine.Build, response.Engine.ExecutableSHA256 = binding.Build, binding.ExecutableSHA256
	response.Model.ID, response.RequestedConfig = binding.Request.Model.ID, binding.Request.Config
	response.Extensions.Strata.Requested = binding.Request.Extensions.Strata
	response.Extensions.Strata.Inputs = append(binding.Inputs, strataMemoryPlanInputIdentity{Path: "/undeclared", FileID: []uint64{1, 2}})
	data, err = json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := decodeStrataMemoryPlan(data, 0, binding); err == nil {
		t.Fatal("accepted undeclared response input")
	}
	if bytes.Contains(data, []byte("STRATA_API_KEY")) {
		t.Fatal("fixture unexpectedly carries credentials")
	}
}

func TestMemoryPlanStrataSourceModuleClosure(t *testing.T) {
	want := []string{
		"serve/__init__.py", "serve/server.py", "serve/memory_plan.py", "serve/launch_plan.py",
		"serve/mcp.py", "serve/runconfig.py", "serve/credentials.py", "serve/winjob.py",
		"serve/structured.py", "tools/build_provenance.py", "tools/gguf_reader.py",
		"tools/prepare_unsloth_q2.py", "setup.py",
	}
	if !reflect.DeepEqual(strataMemoryPlanModules, want) {
		t.Fatalf("reviewed module closure = %v", strataMemoryPlanModules)
	}
	root := t.TempDir()
	evidence := make(map[string]sourceEvidence)
	for _, name := range strataMemoryPlanModules {
		path := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(name), 0600); err != nil {
			t.Fatal(err)
		}
		file, err := captureSource(path, false)
		if err != nil {
			t.Fatal(err)
		}
		evidence[path] = file
	}
	if _, err := strataMemoryPlanModuleDigest(context.Background(), root, evidence); err != nil {
		t.Fatal(err)
	}
	initPath := filepath.Join(root, "tools", "__init__.py")
	if err := os.WriteFile(initPath, []byte("new package"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := strataMemoryPlanModuleDigest(context.Background(), root, evidence); err == nil {
		t.Fatal("accepted changed tools package contract")
	}
	if err := os.Remove(initPath); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, strataMemoryPlanModules[0])
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("replacement"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := strataMemoryPlanModuleDigest(context.Background(), root, evidence); err == nil {
		t.Fatal("accepted replaced module")
	}
}

func TestMemoryPlanStrataSourceFiniteRequest(t *testing.T) {
	gpu := int64(0)
	input := &strataServingInput{
		launch:  &llamaServingLaunch{Config: &engines.ServingConfig{Models: []engines.ServingModel{{ModelID: "example"}}}},
		config:  strataServingConfig{Exe: "/native", CWD: "/work", GPU: &gpu, Parallel: 1, Env: map[string]string{"STRATA_FILE_CACHE_MIB": "0", "STRATA_RESIDENT_HEADROOM_GIB": "0"}},
		options: map[string]string{"--max-context": "4096", "--kv": "fp16", "--prefill": "128", "--native": "/model", "--pack": "/pack", "--expert-profile": "/profile", "--resident-budget-gib": "1", "--vram-reserve-mib": "1", "--expert-cache": "4", "--kv-resident": "0", "--spec": "2", "--no-prefill-borrow": "true"},
	}
	request, err := strataMemoryPlanFiniteRequest(input)
	if err != nil {
		t.Fatal(err)
	}
	if request.Model.ID != "example" || request.Budget != nil || request.Extensions.Strata.FileCacheBytes != 0 {
		t.Fatalf("request = %+v", request)
	}
	input.options["--spec"] = "3"
	if _, err := strataMemoryPlanFiniteRequest(input); err == nil {
		t.Fatal("accepted unsupported speculative window")
	}
}
