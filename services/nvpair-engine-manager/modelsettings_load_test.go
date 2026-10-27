// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
)

// Registered before TestMain's normal builds. The existing fixed package
// executable is the child; no new executable path needs firewall registration.
func runModelLoadChild() bool {
	if os.Getenv("PAIR_TYPED_LOAD_CHILD") != "1" {
		return false
	}
	index := slices.Index(os.Args, "--models-preset")
	if index < 0 || index+1 >= len(os.Args) {
		os.Exit(2)
	}
	data, err := os.ReadFile(os.Args[index+1])
	if err != nil {
		os.Exit(2)
	}
	sections, err := parseLlamaServingPreset(data)
	if err != nil {
		os.Exit(2)
	}
	if sections["medium-model"]["ctx-size"] != "16384" || sections["medium-model"]["parallel"] != "2" {
		os.Exit(2)
	}
	for _, options := range sections {
		if _, err := os.Stat(options["model"]); err != nil {
			os.Exit(2)
		}
	}
	capture, err := json.Marshal(struct {
		Args   []string
		Preset []byte
	}{os.Args[1:], data})
	if err != nil {
		os.Exit(2)
	}
	if err := os.WriteFile(os.Getenv("PAIR_TYPED_CAPTURE"), capture, 0o600); err != nil {
		os.Exit(2)
	}
	address := os.Getenv("PAIR_TYPED_ADDR")
	if address == "" {
		port := slices.Index(os.Args, "--port")
		if port < 0 || port+1 >= len(os.Args) {
			os.Exit(2)
		}
		address = net.JoinHostPort("127.0.0.1", os.Args[port+1])
	}
	listener, err := net.Listen("tcp", address)
	if err != nil {
		os.Exit(2)
	}
	server := http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })}
	if err := server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
		os.Exit(2)
	}
	return true
}

type modelLoadFixture struct {
	ex                    *Executor
	st                    *engineState
	rt                    Runtime
	command               launchCommand
	root, source, capture string
}

func newModelLoadFixture(t *testing.T) modelLoadFixture {
	t.Helper()
	f := newLlamaServingFixture(t)
	st := settingsState(t, f.ex)
	st.modelsDir = f.root
	bin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	st.binPath = bin
	st.plat.Detect = []string{bin}
	st.plat.DetectRequired = nil
	rt := st.plat.Runtime
	rt.Bin = bin
	rt.Env = map[string]string{"LLAMA_CACHE": f.root, "PAIR_TYPED_LOAD_CHILD": "1", "PAIR_TYPED_CAPTURE": filepath.Join(f.root, "capture.json"), "PAIR_TYPED_ADDR": "127.0.0.1:{port}"}
	for _, id := range []string{"medium", "weak"} {
		writeInspectionGGUF(t, filepath.Join(f.root, id+".gguf"), "llama", 0.25, 4096)
	}
	writeTestFile(t, f.source, []byte("[*]\nfit = off\nload-on-startup = false\n[weak-model]\nmodel = weak.gguf\nctx-size = 4096\nparallel = 1\ngpu-layers = 0\n[medium-model]\nmodel = medium.gguf\nctx-size = 16384\nparallel = 2\ngpu-layers = 0\n"))
	args := []string{"--models-preset", f.source, "--models-max", "1", "--log-disable"}
	rt.LaunchArgs, rt.Serving, rt.modelLoadOwner = &args, nil, f.ex
	st.plat.Runtime = rt
	command, err := launchForState(st, rt.Port)
	if err != nil {
		t.Fatal(err)
	}
	return modelLoadFixture{f.ex, st, rt, command, f.root, f.source, filepath.Join(f.root, "capture.json")}
}

func convertModelLoadFixture(t *testing.T, f modelLoadFixture) Runtime {
	t.Helper()
	conversion, err := f.ex.previewTypedLlamaConversion(context.Background(), f.st, f.command, f.rt)
	if err != nil {
		t.Fatal(err)
	}
	if len(conversion.Runtime.ModelLoads) != 2 || len(conversion.SourceDigest) != 64 || len(conversion.Diff) == 0 {
		t.Fatalf("incomplete conversion: %+v", conversion)
	}
	return conversion.Runtime
}

func typedFixtureCommand(t *testing.T, f modelLoadFixture, rt Runtime) launchCommand {
	t.Helper()
	vars := f.st.pathVars()
	vars["host"] = "127.0.0.1"
	vars["port"] = strconv.Itoa(rt.Port)
	command, err := resolveProcessLaunch(llamaServingBase(rt), rt.Bin, vars)
	if err != nil {
		t.Fatal(err)
	}
	return command
}

func captureModelLoadChild(t *testing.T, path string) ([]string, []byte) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var captured struct {
		Args   []string
		Preset []byte
	}
	if err := json.Unmarshal(data, &captured); err != nil {
		t.Fatal(err)
	}
	return captured.Args, captured.Preset
}

func TestModelSettingsLoadDeterministicCompleteCompile(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	command := typedFixtureCommand(t, f, rt)
	first, err := compilePreparedServing(f.st, command, rt, true)
	if err != nil {
		t.Fatal(err)
	}
	slices.Reverse(rt.ModelLoads)
	second, err := compilePreparedServing(f.st, command, rt, true)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Snapshot, second.Snapshot) || first.Digest != second.Digest {
		t.Fatal("input order changed deterministic output or identity")
	}
	sections, err := parseLlamaServingPreset(first.Snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if len(sections) != 2 || sections["medium-model"]["ctx-size"] != "16384" || sections["medium-model"]["parallel"] != "2" || sections["weak-model"]["ctx-size"] != "4096" {
		t.Fatalf("router allocation differs: %+v", sections)
	}
	if first.Config != nil || len(first.Envelopes) != 0 {
		t.Fatal("typed settings fabricated declarations")
	}
	if first.ModelLoads[0].ModelID != "medium-model" || first.ModelLoads[0].Resolved.PerSlotContext != 8192 || first.ModelLoads[0].Resolved.TotalContext != 16384 {
		t.Fatalf("incorrect requested/resolved arithmetic: %+v", first.ModelLoads)
	}
}

func TestModelSettingsLoadTypedOnlyLifecycleConsumesPreset(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	f.st.plat.Runtime = rt
	if rt.Serving != nil || !preparedServingRuntime(f.st, rt) || !llamaServingRuntime(f.st, rt) {
		t.Fatal("typed-only runtime missed lifecycle predicate")
	}
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	args, preset := captureModelLoadChild(t, f.capture)
	index := slices.Index(args, "--models-preset")
	if index < 0 || index+1 >= len(args) || args[index+1] == f.source {
		t.Fatal("child did not receive private generation")
	}
	for _, flag := range []string{"--ctx-size", "--parallel", "--sleep-idle-seconds", "--cache-ram"} {
		if slices.Contains(args, flag) {
			t.Fatalf("router argv contains per-model allocation %s", flag)
		}
	}
	sections, err := parseLlamaServingPreset(preset)
	if err != nil {
		t.Fatal(err)
	}
	if sections["medium-model"]["ctx-size"] != "16384" || sections["weak-model"]["ctx-size"] != "4096" {
		t.Fatal("child did not consume complete generated bytes")
	}
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	if state.Serving != nil || len(state.ModelLoads) != 2 || len(state.ServingReadback.LaunchValidated) != 0 {
		t.Fatal("readback lost loads or manufactured strict evidence")
	}
	if _, err := os.Stat(args[index+1]); err != nil {
		t.Fatal("readback removed active generation", err)
	}
}

func TestModelSettingsLoadFreshExecutorAndCrashRegeneration(t *testing.T) {
	test := func(name string, automatic bool) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			if automatic {
				f.rt.Port = 0
				delete(f.rt.Env, "PAIR_TYPED_ADDR")
				f.st.plat.Runtime = f.rt
				var err error
				f.command, err = launchForState(f.st, 0)
				if err != nil {
					t.Fatal(err)
				}
			}
			rt := convertModelLoadFixture(t, f)
			if err := f.ex.persistRuntimeServing("fake", rt.Port, rt.LaunchArgs, rt.LaunchEnv, &rt.Serving, &rt.ModelLoads); err != nil {
				t.Fatal(err)
			}
			manifest := *f.st.manifest
			platform := *f.st.plat
			platform.ModelsDir = f.root
			manifest.Platforms = map[string]Platform{runtime.GOOS + "/" + runtime.GOARCH: platform}
			base, err := json.Marshal(manifest)
			if err != nil {
				t.Fatal(err)
			}
			reg := NewRegistry()
			reg.bundledRaw["fake"] = base
			if err := reg.LoadOverrideDir(f.ex.overrideDir); err != nil {
				t.Fatal(err)
			}
			ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())
			acceptModelMutations(ex)
			running := make(chan EngineStatus, 8)
			ex.emit = func(method string, value any) {
				if status, ok := value.(EngineStatus); ok && method == "engine:state-changed" && status.Running {
					running <- status
				}
			}
			t.Cleanup(func() {
				if err := ex.Stop("fake"); err != nil {
					t.Error(err)
				}
			})
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			st.modelsDir = f.root
			if len(st.plat.Runtime.ModelLoads) != 2 || st.plat.Runtime.Serving != nil {
				t.Fatal("fresh executor lost persisted typed input")
			}
			if err := ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			<-running
			if automatic && (st.port <= 0 || st.plat.Runtime.Port != 0) {
				t.Fatal("automatic port was not allocated from unchanged zero-port intent")
			}
			_, before := captureModelLoadChild(t, f.capture)
			st.mu.Lock()
			proc := st.proc
			st.mu.Unlock()
			if proc == nil {
				t.Fatal("fresh start has no owned child")
			}
			if err := proc.cmd.Process.Kill(); err != nil {
				t.Fatal(err)
			}
			select {
			case <-proc.done:
			case <-time.After(5 * time.Second):
				t.Fatal("synthetic crash did not exit")
			}
			select {
			case <-running:
			case <-time.After(15 * time.Second):
				t.Fatal("saved-On crash recovery did not return to readiness")
			}
			st.mu.Lock()
			replacement := st.proc
			st.mu.Unlock()
			if replacement == nil || replacement == proc {
				t.Fatal("crash recovery did not replace the owned child")
			}
			_, after := captureModelLoadChild(t, f.capture)
			if !bytes.Equal(before, after) {
				t.Fatal("crash relaunch changed generated bytes")
			}
		})
	}
	test("configured port", false)
	test("automatic port", true)
}

func TestModelSettingsLoadPreSpawnDriftRefuses(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	f.st.plat.Runtime = rt
	vars := f.st.pathVars()
	vars["host"] = "127.0.0.1"
	vars["port"] = strconv.Itoa(rt.Port)
	err := f.ex.bringUpProcessWithSnapshot(context.Background(), f.st, "fake", rt, rt.Port, vars, func(dir string, data []byte) (string, func(), error) {
		path, release, err := prepareServingSnapshot(dir, data)
		if err != nil {
			return "", nil, err
		}
		writeInspectionGGUF(t, filepath.Join(f.root, "medium.gguf"), "llama", 0.75, 4096)
		return path, release, nil
	})
	if err == nil {
		t.Fatal("pre-spawn drift accepted")
	}
	if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
		t.Fatalf("drift spawned child: %v", err)
	}
	if f.st.proc != nil {
		t.Fatal("drift installed a child")
	}
}

func TestModelSettingsLoadConflictingSourcesRefused(t *testing.T) {
	test := func(name string, edit func(*testing.T, modelLoadFixture, *launchCommand)) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			rt := convertModelLoadFixture(t, f)
			command := typedFixtureCommand(t, f, rt)
			edit(t, f, &command)
			if _, err := compilePreparedServing(f.st, command, rt, false); err == nil {
				t.Fatal("conflicting allocation source accepted")
			}
		})
	}
	test("CLI context", func(t *testing.T, f modelLoadFixture, c *launchCommand) {
		c.Args = append(c.Args, "--ctx_size", "8192")
	})
	test("explicit environment", func(t *testing.T, f modelLoadFixture, c *launchCommand) { c.Env["LLAMA_ARG_CTX_SIZE"] = "16384" })
	test("inherited environment", func(t *testing.T, f modelLoadFixture, c *launchCommand) { t.Setenv("LLAMA_ARG_N_PARALLEL", "2") })
	test("external preset", func(t *testing.T, f modelLoadFixture, c *launchCommand) {
		c.Args = append(c.Args, "--models-preset", f.source)
	})
	test("CLI threads", func(t *testing.T, f modelLoadFixture, c *launchCommand) { c.Args = append(c.Args, "--threads", "8") })
}

func TestModelSettingsLoadPreSpawnContentSourceReplacementRefuses(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	f.st.plat.Runtime = rt
	vars := f.st.pathVars()
	vars["host"] = "127.0.0.1"
	vars["port"] = strconv.Itoa(rt.Port)
	err := f.ex.bringUpProcessWithSnapshot(context.Background(), f.st, "fake", rt, rt.Port, vars, func(dir string, data []byte) (string, func(), error) {
		path, release, err := prepareServingSnapshot(dir, data)
		if err != nil {
			return "", nil, err
		}
		f.ex.downloads.mu.Lock()
		f.ex.downloads.source = "replacement-source"
		f.ex.downloads.mu.Unlock()
		return path, release, nil
	})
	if err == nil {
		t.Fatal("content source replacement accepted before spawn")
	}
	if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
		t.Fatal("source replacement spawned child", err)
	}
}

func TestModelSettingsLoadResetAndDefaultSemantics(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	slots, threads := int32(3), int32(6)
	patched, err := patchModelLoad(rt.ModelLoads, "medium-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &slots, Threads: &threads}})
	if err != nil {
		t.Fatal(err)
	}
	reset, err := patchModelLoad(patched, "medium-model", modelconfig.LoadPatch{Clear: []string{"slots", "threads", "context", "gpu"}})
	if err != nil {
		t.Fatal(err)
	}
	rt.ModelLoads = reset
	prepared, err := compilePreparedServing(f.st, typedFixtureCommand(t, f, rt), rt, false)
	if err != nil {
		t.Fatal(err)
	}
	load := prepared.ModelLoads[0]
	if load.Requested.Context != nil || load.Requested.Slots != nil || load.Requested.Threads != nil || load.Requested.GPU != nil || load.Resolved.PerSlotContext != 8192 || load.Resolved.Slots != 1 {
		t.Fatalf("reset retained old intent: %+v", load)
	}
	sections, err := parseLlamaServingPreset(prepared.Snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := sections["medium-model"]["threads"]; exists {
		t.Fatal("reset retained hidden thread override")
	}
	if !reflect.DeepEqual(prepared.ModelLoads[1].Requested, rt.ModelLoads[1].Requested) {
		t.Fatal("reset changed another model")
	}
}

func TestModelSettingsLoadApplyRefusalBeforeAdmission(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	test := func(name string, stored, requested []modelconfig.ModelLoad) {
		t.Run(name, func(t *testing.T) {
			f.st.plat.Runtime.ModelLoads = stored
			f.ex.switchAdmission.Lock()
			if f.ex.switchMutations == nil {
				f.ex.switchMutations = make(map[string]int)
			}
			f.ex.switchMutations[""] = 1
			f.ex.switchAdmission.Unlock()
			defer func() { f.ex.switchAdmission.Lock(); delete(f.ex.switchMutations, ""); f.ex.switchAdmission.Unlock() }()
			_, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: settings.Config{ServerPort: rt.Port, ModelLoads: requested}}, func() error { t.Fatal("gated apply rebound"); return nil })
			if err == nil || !strings.Contains(err.Error(), "O1") {
				t.Fatalf("missing O1 refusal before admission: %v", err)
			}
			if f.st.running || f.st.proc != nil {
				t.Fatal("Off apply activated engine")
			}
			if _, err := os.Stat(filepath.Join(f.ex.overrideDir, "fake.json")); !os.IsNotExist(err) {
				t.Fatalf("gated apply persisted: %v", err)
			}
		})
	}
	test("new input", nil, rt.ModelLoads)
	test("unchanged echo", rt.ModelLoads, rt.ModelLoads)
	test("removal", rt.ModelLoads, nil)
}

func TestModelSettingsLoadPersistencePreservesOpaqueFields(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	host := runtime.GOOS + "/" + runtime.GOARCH
	path := filepath.Join(f.ex.overrideDir, "fake.json")
	writeTestFile(t, path, []byte(fmt.Sprintf(`{"engine":"fake","opaque":900719925474099312345,"runtime":{"opaque":1.0000000000000000001},"platforms":{"%s":{"runtime":{"opaque":9007199254740993}},"other/platform":{"runtime":{"port":42}}}}`, host)))
	serving := servingSettingsFixture()
	if err := f.ex.persistRuntimeServing("fake", rt.Port, rt.LaunchArgs, rt.LaunchEnv, &serving, &rt.ModelLoads); err != nil {
		t.Fatal(err)
	}
	if err := f.ex.persistPort("fake", rt.Port+1); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var record struct {
		Opaque    json.Number
		Runtime   struct{ Opaque json.Number }
		Platforms map[string]struct {
			Runtime struct {
				Opaque     json.Number
				Port       int
				ModelLoads []modelconfig.ModelLoad
				Serving    *engines.ServingConfig
				LaunchArgs []string `json:"launch_args"`
			}
		}
	}
	if err := json.Unmarshal(data, &record); err != nil {
		t.Fatal(err)
	}
	if record.Opaque.String() != "900719925474099312345" || record.Runtime.Opaque.String() != "1.0000000000000000001" || record.Platforms[host].Runtime.Opaque.String() != "9007199254740993" {
		t.Fatal("opaque numeric precision changed")
	}
	persisted := record.Platforms[host].Runtime
	if !modelconfig.EqualLoads(persisted.ModelLoads, rt.ModelLoads) || !persisted.Serving.Equal(serving) || record.Platforms["other/platform"].Runtime.Port != 42 || !slices.Equal(persisted.LaunchArgs, *rt.LaunchArgs) {
		t.Fatal("persistence lost unrelated model/platform/declaration/launch state")
	}
	if f.st.running || f.st.proc != nil {
		t.Fatal("private persistence activated stopped engine")
	}
	cleared := []modelconfig.ModelLoad{}
	if err := f.ex.persistRuntimeServing("fake", rt.Port, nil, nil, nil, &cleared); err != nil {
		t.Fatal(err)
	}
	if got := readSettingsOverride(t, f.ex); len(got.ModelLoads) != 0 || !got.Serving.Equal(serving) {
		t.Fatal("reset failed to clear loads or changed declaration")
	}
}

func TestModelSettingsLoadPersistenceFailurePreservesState(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	path := filepath.Join(f.ex.overrideDir, "fake.json")
	invalid := []byte(`{"engine":"fake","runtime":{"opaque":9007199254740993}}`)
	writeTestFile(t, path, invalid)
	failure := errors.New("synthetic override replacement failure")
	useEngineFileReplacer(t, durablefile.Replacer{Replace: func(string, string) error { return failure }})
	before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
	if err := f.ex.persistRuntimeServing("fake", rt.Port, rt.LaunchArgs, rt.LaunchEnv, &rt.Serving, &rt.ModelLoads); !errors.Is(err, failure) {
		t.Fatalf("expected replacement failure, got %v", err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(invalid, after) || !modelconfig.EqualLoads(before, f.st.plat.Runtime.ModelLoads) || f.st.running {
		t.Fatal("failed persistence changed disk/runtime")
	}
}

func TestModelSettingsLoadConversionRejectsCompleteRouter(t *testing.T) {
	test := func(name, extra string) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			data, err := os.ReadFile(f.source)
			if err != nil {
				t.Fatal(err)
			}
			data = append(data, []byte(extra)...)
			writeTestFile(t, f.source, data)
			before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
			_, refusal := f.ex.previewTypedLlamaConversion(context.Background(), f.st, f.command, f.rt)
			if refusal == nil {
				t.Fatal("unsupported router member partially converted")
			}
			after, err := os.ReadFile(f.source)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(data, after) || !modelconfig.EqualLoads(before, f.st.plat.Runtime.ModelLoads) {
				t.Fatal("refused conversion changed authority")
			}
		})
	}
	test("sampling field", "temp = 0.4\n")
	test("third model", "[third]\nmodel = third.gguf\ncache-prompt = true\n")
	test("unified KV", "kv-unified = true\n")
}

func TestModelSettingsLoadFiniteValidation(t *testing.T) {
	layers := int32(4)
	test := func(name string, r modelconfig.RequestedLoad, field string) {
		t.Run(name, func(t *testing.T) {
			_, _, err := typedLlamaModelOptions("exact", r, &layers)
			var typed *ServingFieldError
			if !errors.As(err, &typed) || !strings.Contains(typed.Field, field) {
				t.Fatalf("expected %s refusal, got %v", field, err)
			}
		})
	}
	zero, large := int32(0), int32(6)
	batch, ubatch := int32(128), int32(256)
	precision := "q8_0"
	test("zero slots", modelconfig.RequestedLoad{Slots: &zero}, "requested")
	test("overflow", modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: math.MaxInt32}}, "requested")
	test("too many layers", modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "layers", Layers: &large}}, "gpu.layers")
	test("batch inversion", modelconfig.RequestedLoad{BatchSize: &batch, UBatchSize: &ubatch}, "requested")
	test("unmapped precision", modelconfig.RequestedLoad{CacheTypeV: &precision}, "cacheTypeV")
	test("Auto unavailable", modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "auto"}}, "gpu")
}

func TestModelSettingsLoadPreviewPreservesCompleteRouterAndSources(t *testing.T) {
	f := newModelLoadFixture(t)
	f.command.Env["LLAMA_ARG_LOG_COLORS"] = "off"
	f.command.Args = append(f.command.Args, "--log-file", "model")
	conversion, err := f.ex.previewTypedLlamaConversion(context.Background(), f.st, f.command, f.rt)
	if err != nil {
		t.Fatal(err)
	}
	rt := conversion.Runtime
	if !slices.Contains(*rt.LaunchArgs, "--log-disable") || slices.Contains(*rt.LaunchArgs, "--models-preset") {
		t.Fatalf("conversion lost unrelated options or retained source: %v", *rt.LaunchArgs)
	}
	logFile := slices.Index(*rt.LaunchArgs, "--log-file")
	if logFile < 0 || logFile+1 >= len(*rt.LaunchArgs) || (*rt.LaunchArgs)[logFile+1] != "model" {
		t.Fatalf("conversion consumed an unrelated flag value: %v", *rt.LaunchArgs)
	}
	env, err := literalEnvironment(*rt.LaunchEnv)
	if err != nil {
		t.Fatal(err)
	}
	if env["LLAMA_ARG_LOG_COLORS"] != "off" {
		t.Fatal("conversion dropped unrelated native environment")
	}
	if f.st.plat.Runtime.ModelLoads != nil || f.st.running {
		t.Fatal("conversion preview changed settings or activation")
	}
	f.st.plat.Runtime = rt
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	preview, err := f.ex.PreviewLaunch(settings.Request{Engine: "fake", Settings: settings.ConfigFromLaunch(state, 54301)})
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("typed preview refused valid input: %+v %v", preview.Errors, err)
	}
	if preview.Settings.ServerPort != rt.Port || preview.Settings.ProxyPort != 54301 || !modelconfig.EqualLoads(preview.Settings.ModelLoads, rt.ModelLoads) || preview.Settings.Serving != nil {
		t.Fatal("typed preview changed unrelated settings or accepted resolution")
	}
	if preview.Impact == nil || preview.Impact.RestartScope != "unknown" || preview.Restart {
		t.Fatal("preview selected O1 restart behavior")
	}
	if _, err := os.Stat(filepath.Join(f.ex.overrideDir, "fake.json")); !os.IsNotExist(err) {
		t.Fatal("preview persisted settings", err)
	}
}

func TestModelSettingsLoadDeclaredServingValidatesGeneratedBytes(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	host, vram := int64(4096), int64(0)
	rt.Serving = &engines.ServingConfig{Version: 1}
	for _, load := range rt.ModelLoads {
		data, err := os.ReadFile(load.Resolved.Artifacts[0].Path)
		if err != nil {
			t.Fatal(err)
		}
		rt.Serving.Models = append(rt.Serving.Models, engines.ServingModel{ModelID: load.ModelID, MaxOutputTokens: 128, Memory: engines.ServingMemory{HostRAMBytes: &host, VRAMBytes: &vram}, Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: sha256Hex(data), Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}}}})
	}
	command := typedFixtureCommand(t, f, rt)
	prepared, err := compilePreparedServing(f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared.Envelopes) != len(rt.Serving.Models) || !reflect.DeepEqual(prepared.Config.Models[0].Memory, rt.Serving.Models[0].Memory) || prepared.Config.Models[0].Artifacts[0].SHA256 != rt.Serving.Models[0].Artifacts[0].SHA256 {
		t.Fatal("typed compilation altered declaration memory or asserted content")
	}
	rt.Serving = prepared.Config
	rt.ModelLoads = prepared.ModelLoads
	if _, err := compilePreparedServing(f.st, command, rt, true); err != nil {
		t.Fatal("reviewed declaration cannot validate generated bytes", err)
	}
	rt.Serving.Models[0].MaxOutputTokens = 100000
	if _, err := compilePreparedServing(f.st, command, rt, false); err == nil {
		t.Fatal("generated preset bypassed declaration output bound")
	}
}

func TestModelSettingsLoadInheritedCapacityChangesApproval(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	command := typedFixtureCommand(t, f, rt)
	index := slices.Index(command.Args, "--models-max")
	if index < 0 || index+1 >= len(command.Args) {
		t.Fatal("fixture has no explicit capacity to remove")
	}
	command.Args = append(slices.Clone(command.Args[:index]), command.Args[index+2:]...)
	t.Setenv("LLAMA_ARG_MODELS_MAX", "1")
	first, err := compilePreparedServing(f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	rt.ModelLoads = first.ModelLoads
	t.Setenv("LLAMA_ARG_MODELS_MAX", "2")
	if _, err := compilePreparedServing(f.st, command, rt, true); err == nil {
		t.Fatal("changed inherited router capacity retained approval")
	}
	second, err := compilePreparedServing(f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest == second.Digest {
		t.Fatal("effective router capacity is absent from typed identity")
	}
}

func TestModelSettingsLoadWorkerConversionPreviewAndApplyRefusal(t *testing.T) {
	f := newModelLoadFixture(t)
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	before := snapshotLlamaState(f.st)
	three := int32(3)
	request := settings.Request{Engine: "fake", Settings: settings.ConfigFromLaunch(state, 54301), ModelChange: &modelconfig.ModelChange{ModelID: "medium-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &three}}}}
	server, client := net.Pipe()
	t.Cleanup(func() { _ = server.Close(); _ = client.Close() })
	if err := client.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	manager := &Manager{exec: f.ex, codec: NewCodec(server)}
	call := func(method string, params any) Message {
		t.Helper()
		raw, err := json.Marshal(params)
		if err != nil {
			t.Fatal(err)
		}
		id := json.RawMessage(`"typed-preview"`)
		if !manager.handleSettingsMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: method, Params: raw}) {
			t.Fatalf("worker did not handle %s", method)
		}
		var reply Message
		if err := json.NewDecoder(client).Decode(&reply); err != nil {
			t.Fatal(err)
		}
		return reply
	}
	reply := call("engine:preview-launch", request)
	if reply.Error != nil {
		t.Fatalf("conversion preview refused: %+v", reply.Error)
	}
	var preview settings.Preview
	if err := modelconfig.DecodeStrict(reply.Result, &preview); err != nil {
		t.Fatal(err)
	}
	if len(preview.Errors) != 0 || len(preview.Settings.ModelLoads) != 2 || len(preview.PreviewDigest) != 64 || preview.Restart || preview.Impact == nil || preview.Impact.RestartScope != "unknown" {
		t.Fatalf("incomplete read-only preview: errors=%v digest=%s", preview.Errors, preview.PreviewDigest)
	}
	if preview.Settings.ModelLoads[0].ModelID != "medium-model" || preview.Settings.ModelLoads[0].Resolved.TotalContext != 24576 || preview.Settings.ModelLoads[1].Resolved.TotalContext != 4096 || preview.Settings.Serving != nil {
		t.Fatal("conversion did not preserve complete router and selected allocation")
	}
	var found bool
	for _, diff := range preview.Diff {
		if diff.Field != "models.medium-model.ctx-size" {
			continue
		}
		var old, next string
		if err := json.Unmarshal(diff.Before, &old); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(diff.After, &next); err != nil {
			t.Fatal(err)
		}
		if old != "16384" || next != "24576" {
			t.Fatalf("context diff %q -> %q", old, next)
		}
		found = true
	}
	if !found {
		t.Fatal("conversion preview omitted changed native context")
	}
	refused := call("engine:configure-launch", settings.Configure{Engine: "fake", OperationID: "O1-still-refused", Settings: preview.Settings})
	if refused.Error == nil {
		t.Fatal("read-only preview opened typed application")
	}
	if !reflect.DeepEqual(before.plat.Runtime, f.st.plat.Runtime) || f.st.running || f.st.proc != nil {
		t.Fatal("preview or refused apply changed runtime")
	}
	if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
		t.Fatal("preview spawned native work", err)
	}
	if _, err := os.Stat(filepath.Join(f.ex.overrideDir, "fake.json")); !os.IsNotExist(err) {
		t.Fatal("preview or refused apply persisted settings", err)
	}
	request.Settings.ServerPort++
	stale := call("engine:preview-launch", request)
	if stale.Error == nil {
		var invalid settings.Preview
		if err := modelconfig.DecodeStrict(stale.Result, &invalid); err != nil {
			t.Fatal(err)
		}
		if len(invalid.Errors) == 0 || invalid.PreviewDigest != "" {
			t.Fatal("stale baseline produced reviewed conversion")
		}
	}
}

func TestModelSettingsLoadContentFenceAndNativeBuildIdentity(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	command := typedFixtureCommand(t, f, rt)
	release, err := f.ex.downloads.acquireWrite("fake")
	if err != nil {
		t.Fatal(err)
	}
	_, compileErr := compilePreparedServing(f.st, command, rt, false)
	release()
	if compileErr == nil {
		t.Fatal("compiler bypassed content writer fence")
	}
	// A synthetic executable binding changes identity without running it.
	executable := filepath.Join(f.root, "synthetic-native")
	writeTestFile(t, executable, []byte("first synthetic native build"))
	command.Bin = executable
	first, err := compilePreparedServing(f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, executable, []byte("second synthetic native build"))
	second, err := compilePreparedServing(f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest == second.Digest || first.ModelLoads[0].Resolved.NativeBuildID == second.ModelLoads[0].Resolved.NativeBuildID {
		t.Fatal("native build binding omitted from typed identity")
	}
}
