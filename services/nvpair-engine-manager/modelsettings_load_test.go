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
		model := options["model"]
		if !filepath.IsAbs(model) {
			model = filepath.Join(filepath.Dir(os.Args[index+1]), model)
		}
		if _, err := os.Stat(model); err != nil {
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
	port := slices.Index(os.Args, "--port")
	if port < 0 || port+1 >= len(os.Args) {
		os.Exit(2)
	}
	address := net.JoinHostPort("127.0.0.1", os.Args[port+1])
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
	rt.Env = map[string]string{"LLAMA_CACHE": f.root, "PAIR_TYPED_LOAD_CHILD": "1", "PAIR_TYPED_CAPTURE": filepath.Join(f.root, "capture.json")}
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

func TestModelSettingsLoadPreSpawnNativeBuildDriftRefuses(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	native := filepath.Join(f.root, "synthetic-native-build")
	writeTestFile(t, native, []byte("first native binding"))
	rt.Bin = native
	f.st.binPath = native
	prepared, err := compilePreparedServing(f.st, typedFixtureCommand(t, f, rt), rt, false)
	if err != nil {
		t.Fatal(err)
	}
	rt.ModelLoads = prepared.ModelLoads
	f.st.plat.Runtime = rt
	vars := f.st.pathVars()
	vars["host"], vars["port"] = "127.0.0.1", strconv.Itoa(rt.Port)
	err = f.ex.bringUpProcessWithSnapshot(context.Background(), f.st, "fake", rt, rt.Port, vars, func(dir string, data []byte) (string, func(), error) {
		path, release, err := prepareServingSnapshot(dir, data)
		if err != nil {
			return "", nil, err
		}
		writeTestFile(t, native, []byte("replacement native binding"))
		return path, release, nil
	})
	if err == nil || !strings.Contains(err.Error(), "nativeBuildId") && !strings.Contains(err.Error(), "resolved") {
		t.Fatalf("native binding drift reached spawn: %v", err)
	}
	if f.st.proc != nil {
		t.Fatal("changed native build installed a child")
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

func previewModelLoadFixture(t *testing.T, f modelLoadFixture, model string, patch modelconfig.LoadPatch) settings.Preview {
	t.Helper()
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	preview, err := f.ex.PreviewLaunch(settings.Request{Engine: "fake", Settings: settings.ConfigFromLaunch(state, 54301), ModelChange: &modelconfig.ModelChange{ModelID: model, ExpectedPolicyRevision: "0", Load: &patch}})
	if err != nil || len(preview.Errors) != 0 || preview.Conflict != nil {
		t.Fatalf("model load preview: errors=%v conflict=%v err=%v", preview.Errors, preview.Conflict, err)
	}
	return preview
}

func readModelLoadRuntimeOverride(t *testing.T, e *Executor) Runtime {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(e.overrideDir, "fake.json"))
	if err != nil {
		t.Fatal(err)
	}
	data, err = e.reg.mergeOntoBundled(data)
	if err != nil {
		t.Fatal(err)
	}
	data, err = applyPlatformDefaults(data)
	if err != nil {
		t.Fatal(err)
	}
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	platform, ok := manifest.HostPlatform()
	if !ok {
		t.Fatal("persisted host platform missing")
	}
	return platform.Runtime
}

func TestModelSettingsLoadConversionPersistsWhileOff(t *testing.T) {
	f := newModelLoadFixture(t)
	if err := f.ex.desired.set("fake", false, intentPeer); err != nil {
		t.Fatal(err)
	}
	before, err := f.ex.desired.snapshot([]string{"fake"})
	if err != nil {
		t.Fatal(err)
	}
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	rebinds := 0
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings, Resume: true}, func() error { rebinds++; return nil })
	if err != nil {
		t.Fatal(err)
	}
	if result.Running || f.st.proc != nil || rebinds != 1 || !settings.ConfigFromLaunch(result, preview.Settings.ProxyPort).Equal(preview.Settings) {
		t.Fatalf("Off conversion result=%+v rebinds=%d", result, rebinds)
	}
	stored := readModelLoadRuntimeOverride(t, f.ex)
	if !modelconfig.EqualLoads(stored.ModelLoads, preview.Settings.ModelLoads) || stored.Serving != nil || stored.Port != f.rt.Port {
		t.Fatal("conversion lost typed settings, endpoint or serving independence")
	}
	after, err := f.ex.desired.snapshot([]string{"fake"})
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("Off conversion changed lifecycle intent: before=%+v after=%+v err=%v", before, after, err)
	}
	if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
		t.Fatal("Off conversion spawned native work", err)
	}
	if err := os.Remove(f.source); err != nil {
		t.Fatal(err)
	}
	readback, err := f.ex.LaunchSettings("fake")
	if err != nil || !modelconfig.EqualLoads(readback.ModelLoads, preview.Settings.ModelLoads) || readback.ServingReadback == nil || readback.ServingReadback.Status == "mismatch" {
		t.Fatalf("converted readback depends on removed external source: readback=%+v err=%v", readback.ServingReadback, err)
	}
}

func TestModelSettingsLoadConversionRejectsSourceChangedSincePreviewBeforeStop(t *testing.T) {
	f := newModelLoadFixture(t)
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	f.st.mu.Lock()
	previous := f.st.proc
	f.st.mu.Unlock()
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	changed := bytes.Replace(data, []byte("ctx-size = 16384"), []byte("ctx-size = 32768"), 1)
	if bytes.Equal(data, changed) {
		t.Fatal("fixture did not change the unrelated model context")
	}
	writeTestFile(t, f.source, changed)
	_, err = f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { t.Fatal("stale conversion reached rebind"); return nil })
	if err == nil || !strings.Contains(err.Error(), "changed since preview; preview again") {
		t.Fatalf("stale conversion refusal=%v", err)
	}
	f.st.mu.Lock()
	current, running := f.st.proc, f.st.running
	f.st.mu.Unlock()
	if !running || current != previous || len(f.st.plat.Runtime.ModelLoads) != 0 {
		t.Fatal("stale conversion stopped or changed the current router")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadConfigureRestartsOwnedRouter(t *testing.T) {
	f := newModelLoadFixture(t)
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	f.st.mu.Lock()
	previous := f.st.proc
	f.st.mu.Unlock()
	if previous == nil {
		t.Fatal("start did not install an owned child")
	}
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	if !preview.Restart || preview.Impact == nil || preview.Impact.RestartScope != "router" || !slices.Equal(preview.Impact.AffectedModels, []string{"medium-model", "weak-model"}) || preview.Impact.DrainTimeoutSeconds == nil || *preview.Impact.DrainTimeoutSeconds != 60 || preview.Impact.MayInterruptActive == nil || !*preview.Impact.MayInterruptActive || preview.Impact.StaysOff == nil || *preview.Impact.StaysOff || preview.Impact.ActiveWorkPolicy != "proceed-after-timeout" {
		t.Fatalf("incomplete running impact: %+v", preview.Impact)
	}
	rebinds := 0
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error {
		rebinds++
		select {
		case <-previous.done:
		default:
			t.Fatal("rebind preceded owned termination")
		}
		if !modelconfig.EqualLoads(readSettingsOverride(t, f.ex).ModelLoads, preview.Settings.ModelLoads) {
			t.Fatal("rebind preceded typed persistence")
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	f.st.mu.Lock()
	replacement := f.st.proc
	f.st.mu.Unlock()
	if !result.Running || replacement == nil || replacement == previous || rebinds != 1 || result.Serving != nil || !modelconfig.EqualLoads(result.ModelLoads, preview.Settings.ModelLoads) {
		t.Fatalf("typed restart result=%+v replacement=%p previous=%p rebinds=%d", result, replacement, previous, rebinds)
	}
	_, data := captureModelLoadChild(t, f.capture)
	sections, err := parseLlamaServingPreset(data)
	if err != nil {
		t.Fatal(err)
	}
	if sections["weak-model"]["ctx-size"] != "8192" || sections["weak-model"]["parallel"] != "2" || sections["medium-model"]["ctx-size"] != "16384" {
		t.Fatal("replacement did not consume the complete reviewed allocation")
	}
	if enabled, known, err := f.ex.desired.get("fake"); err != nil || !known || !enabled {
		t.Fatalf("restart changed saved On: enabled=%v known=%v err=%v", enabled, known, err)
	}
}

func TestModelSettingsLoadConfigureNoOpPreservesOwnedProcess(t *testing.T) {
	f := newModelLoadFixture(t)
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	request := settingsRequest(t, f.ex)
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: request.Settings}, func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	f.st.mu.Lock()
	previous := f.st.proc
	f.st.mu.Unlock()
	path := filepath.Join(f.ex.overrideDir, "fake.json")
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	one := int32(1)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &one}})
	if preview.Restart || preview.Impact == nil || preview.Impact.RestartScope != "none" || preview.Impact.MayInterruptActive == nil || *preview.Impact.MayInterruptActive {
		t.Fatalf("no-op proposed interruption: %+v", preview.Impact)
	}
	useEngineFileReplacer(t, durablefile.Replacer{Replace: func(string, string) error { t.Fatal("no-op attempted persistence"); return nil }})
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil })
	if err != nil || !result.Running {
		t.Fatalf("no-op apply: result=%+v err=%v", result, err)
	}
	f.st.mu.Lock()
	current := f.st.proc
	f.st.mu.Unlock()
	after, err := os.ReadFile(path)
	if err != nil || current != previous || !bytes.Equal(before, after) {
		t.Fatalf("no-op changed owned process or disk: err=%v", err)
	}
}

func TestModelSettingsLoadConversionRestartsOwnedExternalRouter(t *testing.T) {
	f := newModelLoadFixture(t)
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	f.st.mu.Lock()
	previous := f.st.proc
	f.st.mu.Unlock()
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	f.st.mu.Lock()
	replacement := f.st.proc
	f.st.mu.Unlock()
	if !result.Running || previous == nil || replacement == nil || replacement == previous || !modelconfig.EqualLoads(result.ModelLoads, preview.Settings.ModelLoads) {
		t.Fatalf("external conversion did not replace the owned router: %+v", result)
	}
	args, data := captureModelLoadChild(t, f.capture)
	index := slices.Index(args, "--models-preset")
	if index < 0 || index+1 >= len(args) || args[index+1] == f.source {
		t.Fatal("converted router still consumes the mutable external source")
	}
	sections, err := parseLlamaServingPreset(data)
	if err != nil {
		t.Fatal(err)
	}
	if sections["weak-model"]["parallel"] != "2" || sections["medium-model"]["ctx-size"] != "16384" {
		t.Fatal("converted router lost reviewed or unrelated model settings")
	}
}

func TestModelSettingsLoadConfigureVetoKeepsSavedConfigurationCold(t *testing.T) {
	test := func(name string, veto func(*testing.T, modelLoadFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			veto(t, f)
			before, err := f.ex.desired.snapshot([]string{"fake"})
			if err != nil {
				t.Fatal(err)
			}
			two := int32(2)
			preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
			if preview.Impact == nil || preview.Impact.StaysOff == nil || !*preview.Impact.StaysOff {
				t.Fatalf("cold veto impact: %+v", preview.Impact)
			}
			result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings, Resume: true}, func() error { return nil })
			if err != nil || result.Running || f.st.proc != nil || !modelconfig.EqualLoads(result.ModelLoads, preview.Settings.ModelLoads) {
				t.Fatalf("vetoed configuration result=%+v err=%v", result, err)
			}
			after, err := f.ex.desired.snapshot([]string{"fake"})
			if err != nil || !reflect.DeepEqual(before, after) {
				t.Fatalf("configuration changed intent or lock: before=%+v after=%+v err=%v", before, after, err)
			}
			if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
				t.Fatal("configuration bypassed activation veto", err)
			}
		})
	}
	test("paused by either origin", func(t *testing.T, f modelLoadFixture) {
		f.ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	})
	test("local engine lock", func(t *testing.T, f modelLoadFixture) {
		if _, err := f.ex.desired.setLocked("fake", true); err != nil {
			t.Fatal(err)
		}
	})
}

func TestModelSettingsLoadConfigureRejectsDriftBeforeStop(t *testing.T) {
	test := func(name string, drift func(*testing.T, modelLoadFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			f.st.plat.Runtime = convertModelLoadFixture(t, f)
			if err := f.ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			f.st.mu.Lock()
			previous := f.st.proc
			f.st.mu.Unlock()
			threads := int32(6)
			preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Threads: &threads}})
			drift(t, f)
			_, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { t.Fatal("drift reached rebind"); return nil })
			if err == nil {
				t.Fatal("stale typed resolution was accepted")
			}
			f.st.mu.Lock()
			current, running := f.st.proc, f.st.running
			f.st.mu.Unlock()
			if !running || current != previous {
				t.Fatal("source refusal stopped or replaced the owned child")
			}
			assertNoSettingsOverride(t, f.ex)
		})
	}
	test("artifact binding", func(t *testing.T, f modelLoadFixture) {
		writeInspectionGGUF(t, filepath.Join(f.root, "weak.gguf"), "llama", 0.75, 4096)
	})
	test("native build binding", func(t *testing.T, f modelLoadFixture) {
		path := filepath.Join(f.root, "different-native-build")
		writeTestFile(t, path, []byte("different synthetic native build"))
		f.st.mu.Lock()
		f.st.binPath = path
		f.st.mu.Unlock()
	})
}

func TestModelSettingsLoadConfigureAdoptedRouterHasNoEffects(t *testing.T) {
	f := newModelLoadFixture(t)
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	threads := int32(6)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Threads: &threads}})
	before := snapshotLlamaState(f.st)
	f.st.mu.Lock()
	f.st.running, f.st.adopted = true, true
	f.st.mu.Unlock()
	t.Cleanup(func() {
		f.st.mu.Lock()
		f.st.running, f.st.adopted = false, false
		f.st.mu.Unlock()
	})
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { t.Fatal("adopted router reached rebind"); return nil }); err == nil {
		t.Fatal("adopted router accepted a load change")
	}
	if !modelconfig.EqualLoads(before.plat.Runtime.ModelLoads, f.st.plat.Runtime.ModelLoads) || !f.st.running || !f.st.adopted || f.st.proc != nil {
		t.Fatal("refused apply changed adopted ownership or settings")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadConversionRechecksCompleteExternalSource(t *testing.T) {
	f := newModelLoadFixture(t)
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	recheck, err := f.ex.validateTypedLlamaConversion(context.Background(), f.st, f.rt, preview.Settings)
	if err != nil || recheck == nil {
		t.Fatalf("conversion source not captured: %v", err)
	}
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, f.source, append(data, []byte("\n# changed after preparation\n")...))
	if err := recheck(); err == nil {
		t.Fatal("external bytes changed without invalidating conversion")
	}
	assertNoSettingsOverride(t, f.ex)
	if f.st.running || f.st.proc != nil {
		t.Fatal("conversion revalidation activated the router")
	}
}

func TestModelSettingsLoadConversionRefusesPartialRouterBeforeEffects(t *testing.T) {
	f := newModelLoadFixture(t)
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	preview.Settings.ModelLoads = slices.Delete(preview.Settings.ModelLoads, 0, 1)
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { t.Fatal("partial conversion reached rebind"); return nil }); err == nil {
		t.Fatal("conversion dropped an existing router model")
	}
	if f.st.running || f.st.proc != nil || len(f.st.plat.Runtime.ModelLoads) != 0 {
		t.Fatal("partial conversion changed native state")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadConfigurePersistenceFailureRetainsOldRuntime(t *testing.T) {
	f := newModelLoadFixture(t)
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	request := settingsRequest(t, f.ex)
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: request.Settings}, func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
	path := filepath.Join(f.ex.overrideDir, "fake.json")
	disk, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	threads := int32(6)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Threads: &threads}})
	failure := errors.New("synthetic override preparation failure")
	useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(dir, base string, _ os.FileMode) (durablefile.File, error) {
		if dir == f.ex.overrideDir {
			return nil, failure
		}
		return os.CreateTemp(dir, base)
	}})
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { t.Fatal("failed persistence reached rebind"); return nil })
	if !errors.Is(err, failure) || result.Running || !modelconfig.EqualLoads(result.ModelLoads, before) {
		t.Fatalf("failed persistence adopted the new configuration: result=%+v err=%v", result, err)
	}
	after, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(disk, after) || !modelconfig.EqualLoads(f.st.plat.Runtime.ModelLoads, before) {
		t.Fatalf("failed persistence changed disk or runtime: %v", err)
	}
}

func TestModelSettingsLoadConversionSourceDriftBeforeRestartRefuses(t *testing.T) {
	f := newModelLoadFixture(t)
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error {
		writeTestFile(t, f.source, []byte("external source replaced after native persistence"))
		if err := os.Remove(f.capture); err != nil {
			t.Fatal(err)
		}
		return nil
	})
	if err == nil || result.Running || f.st.proc != nil || !modelconfig.EqualLoads(result.ModelLoads, preview.Settings.ModelLoads) {
		t.Fatalf("source drift was not retained as a cold accepted failure: result=%+v err=%v", result, err)
	}
	if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
		t.Fatal("changed conversion source spawned a replacement", err)
	}
}

func TestModelSettingsLoadConfigureRechecksSourcesAfterRebind(t *testing.T) {
	test := func(name string, drift func(*testing.T, modelLoadFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			f.st.plat.Runtime = convertModelLoadFixture(t, f)
			if err := f.ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			threads := int32(6)
			preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Threads: &threads}})
			result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error {
				drift(t, f)
				if err := os.Remove(f.capture); err != nil {
					t.Fatal(err)
				}
				return nil
			})
			if err == nil || result.Running || f.st.proc != nil || !modelconfig.EqualLoads(result.ModelLoads, preview.Settings.ModelLoads) {
				t.Fatalf("post-rebind drift result=%+v err=%v", result, err)
			}
			if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
				t.Fatal("changed source spawned a replacement", err)
			}
		})
	}
	test("artifact binding", func(t *testing.T, f modelLoadFixture) {
		writeInspectionGGUF(t, filepath.Join(f.root, "weak.gguf"), "llama", 0.75, 4096)
	})
	test("content source generation", func(t *testing.T, f modelLoadFixture) {
		f.ex.downloads.mu.Lock()
		f.ex.downloads.source = "replaced-during-rebind"
		f.ex.downloads.mu.Unlock()
	})
	test("native build binding", func(t *testing.T, f modelLoadFixture) {
		path := filepath.Join(f.root, "different-native-build")
		writeTestFile(t, path, []byte("different synthetic native build"))
		f.st.mu.Lock()
		f.st.binPath = path
		f.st.plat.Detect = []string{path}
		f.st.mu.Unlock()
	})
}

func TestModelSettingsLoadSetPortUsesTypedConfigurationOwner(t *testing.T) {
	f := newModelLoadFixture(t)
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	result, err := f.ex.SetPort(context.Background(), "fake", port)
	if err != nil {
		t.Fatal(err)
	}
	stored := readModelLoadRuntimeOverride(t, f.ex)
	if result.Running || f.st.proc != nil || stored.Port != port || len(stored.ModelLoads) != len(before) || stored.Serving != nil {
		t.Fatalf("typed port result=%+v stored=%+v", result, stored)
	}
	for i, load := range stored.ModelLoads {
		if load.ModelID != before[i].ModelID || !reflect.DeepEqual(load.Requested, before[i].Requested) || !reflect.DeepEqual(load.Resolved.Artifacts, before[i].Resolved.Artifacts) {
			t.Fatal("port change altered models, requested allocation or artifacts")
		}
	}
	if _, err := compilePreparedServing(f.st, typedFixtureCommand(t, f, f.st.plat.Runtime), f.st.plat.Runtime, true); err != nil {
		t.Fatal("port path persisted a stale native resolution", err)
	}
}

func TestModelSettingsLoadPinsAutomaticPortWithRestartableIdentity(t *testing.T) {
	f := newModelLoadFixture(t)
	f.rt.Port = 0
	f.st.plat.Runtime = f.rt
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	request := settingsRequest(t, f.ex)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	request.Settings.ServerPort = port
	command, err := launchForState(f.st, port)
	if err != nil {
		t.Fatal(err)
	}
	request.Settings.LaunchText, err = command.argumentText(llamaServingBase(f.st.plat.Runtime))
	if err != nil {
		t.Fatal(err)
	}
	preview, err := f.ex.PreviewLaunch(request)
	if err != nil || len(preview.Errors) != 0 || preview.Conflict != nil {
		t.Fatalf("pin automatic port preview: errors=%v conflict=%v err=%v", preview.Errors, preview.Conflict, err)
	}
	result, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil })
	if err != nil || !result.Running || result.EffectivePort != port {
		t.Fatalf("pin automatic port: result=%+v err=%v", result, err)
	}
	stored := readModelLoadRuntimeOverride(t, f.ex)
	if stored.Port != port || !modelconfig.EqualLoads(stored.ModelLoads, result.ModelLoads) {
		t.Fatal("pinned endpoint and native identity were not persisted together")
	}
	if err := f.ex.Stop("fake"); err != nil {
		t.Fatal(err)
	}
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal("pinned native identity failed next start", err)
	}
}

func TestModelSettingsLoadSetPortPinsAutomaticEndpoint(t *testing.T) {
	f := newModelLoadFixture(t)
	f.rt.Port = 0
	f.st.plat.Runtime = f.rt
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	result, err := f.ex.SetPort(context.Background(), "fake", port)
	if err != nil || result.Running || result.Port != port {
		t.Fatalf("pin automatic endpoint: result=%+v err=%v", result, err)
	}
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal("pinned port configuration failed start", err)
	}
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
	if preview.Restart {
		t.Fatal("unchanged Off preview requested a restart")
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

func TestModelSettingsLoadWorkerConversionPreviewAndApply(t *testing.T) {
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
	rebinds := 0
	manager.settingsRelay.send = func(_ context.Context, method string, value any) error {
		relay, ok := value.(settings.Relay)
		if !ok || method != "engine:settings-request" || relay.Method != "rebind" || !modelconfig.EqualLoads(relay.Request.Settings.ModelLoads, f.st.plat.Runtime.ModelLoads) {
			return fmt.Errorf("unexpected rebind: %s", method)
		}
		rebinds++
		raw, err := json.Marshal(settingsReply{ID: relay.ID, Result: json.RawMessage(`{}`)})
		if err != nil {
			return err
		}
		manager.settingsRelay.reply(raw)
		return nil
	}
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
	if len(preview.Errors) != 0 || len(preview.Settings.ModelLoads) != 2 || len(preview.PreviewDigest) != 64 || preview.Restart || preview.Impact == nil || preview.Impact.RestartScope != "router" || preview.Impact.DrainTimeoutSeconds == nil || *preview.Impact.DrainTimeoutSeconds != 60 || len(preview.Impact.AffectedModels) != 2 {
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
	if !reflect.DeepEqual(before.plat.Runtime, f.st.plat.Runtime) || f.st.running || f.st.proc != nil {
		t.Fatal("preview changed runtime")
	}
	if _, err := os.Stat(f.capture); !os.IsNotExist(err) {
		t.Fatal("preview spawned native work", err)
	}
	if _, err := os.Stat(filepath.Join(f.ex.overrideDir, "fake.json")); !os.IsNotExist(err) {
		t.Fatal("preview persisted settings", err)
	}
	applied := call("engine:configure-launch", settings.Configure{Engine: "fake", OperationID: "reviewed-conversion", Settings: preview.Settings})
	if applied.Error != nil {
		t.Fatalf("accepted conversion RPC refused: %+v", applied.Error)
	}
	var result settings.LaunchState
	if err := modelconfig.DecodeStrict(applied.Result, &result); err != nil {
		t.Fatal(err)
	}
	if !settings.ConfigFromLaunch(result, preview.Settings.ProxyPort).Equal(preview.Settings) || result.Running || f.st.proc != nil || rebinds != 1 || len(manager.settingsRelay.pending) != 0 {
		t.Fatalf("conversion RPC lost reviewed settings or activated: %+v rebinds=%d", result, rebinds)
	}
	if !modelconfig.EqualLoads(readSettingsOverride(t, f.ex).ModelLoads, preview.Settings.ModelLoads) {
		t.Fatal("conversion RPC did not persist accepted loads")
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

func TestModelSettingsLoadAutoFrozenRestartCompilation(t *testing.T) {
	f, request, owner := autoTestFrozenLaunch(t)
	rt := f.st.plat.Runtime
	rt.ModelLoads = modelconfig.CloneLoads(request.Settings.ModelLoads)
	command := typedFixtureCommand(t, f, rt)
	before := modelconfig.CloneLoads(rt.ModelLoads)
	owner.calls = 0
	owner.memory = nil
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: owner})
	prepared, err := compileTypedLlamaInputs(ctx, f.st, command, rt, true)
	if err != nil {
		t.Fatal("frozen restart/recovery failed without telemetry", err)
	}
	if owner.calls != 0 || !modelconfig.EqualLoads(before, prepared.ModelLoads) || prepared.Config != nil {
		t.Fatal("finite compilation retuned, consulted telemetry or created serving authority")
	}
	sections, err := parseLlamaServingPreset(prepared.Snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if sections["weak-model"]["fit"] != "off" || sections["weak-model"]["load-on-startup"] != "false" || sections["weak-model"]["ctx-size"] != "4096" {
		t.Fatalf("unexpected native result: %v", sections["weak-model"])
	}
	// Repeated cold compilation models recovery: only persisted settings survive,
	// and a fresh estimator must not be called for a saved allocation.
	f.ex.modelEstimates = modelEstimateState{run: func(context.Context, modelEstimateInput) modelconfig.Estimate {
		t.Fatal("recovery called estimator")
		return modelconfig.Estimate{}
	}}
	again, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, true)
	if err != nil || again.Digest != prepared.Digest || !bytes.Equal(again.Snapshot, prepared.Snapshot) {
		t.Fatalf("cold compilation changed frozen result: %v", err)
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadAutoFrozenIdentityDriftRefuses(t *testing.T) {
	test := func(name string, mutate func(*testing.T, modelLoadFixture, *Runtime, *launchCommand)) {
		t.Run(name, func(t *testing.T) {
			f, request, _ := autoTestFrozenLaunch(t)
			rt := f.st.plat.Runtime
			rt.ModelLoads = modelconfig.CloneLoads(request.Settings.ModelLoads)
			command := typedFixtureCommand(t, f, rt)
			mutate(t, f, &rt, &command)
			if _, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, true); err == nil {
				t.Fatal("frozen identity drift accepted")
			}
			assertNoSettingsOverride(t, f.ex)
			if f.st.running || f.st.proc != nil {
				t.Fatal("identity refusal activated engine")
			}
		})
	}
	test("artifact replacement", func(t *testing.T, _ modelLoadFixture, rt *Runtime, _ *launchCommand) {
		writeInspectionGGUF(t, rt.ModelLoads[1].Resolved.Artifacts[0].Path, "llama", 0.75, 4096)
	})
	test("native build replacement", func(t *testing.T, f modelLoadFixture, _ *Runtime, command *launchCommand) {
		command.Bin = filepath.Join(f.root, "changed-native")
		writeTestFile(t, command.Bin, []byte("changed native build"))
	})
	test("stale source preview digest", func(_ *testing.T, _ modelLoadFixture, rt *Runtime, _ *launchCommand) {
		rt.ModelLoads[1].Resolved.SourceInputDigest = strings.Repeat("f", 64)
	})
	test("stale preset preview digest", func(_ *testing.T, _ modelLoadFixture, rt *Runtime, _ *launchCommand) {
		rt.ModelLoads[1].Resolved.PresetDigest = strings.Repeat("f", 64)
	})
}

func TestModelSettingsLoadAutoConversionSourceRemainsBound(t *testing.T) {
	f := newModelLoadFixture(t)
	now := time.Now()
	owner := &autoTestMemoryOwner{memory: autoTestMemory(now)}
	f.ex.modelEstimates = modelEstimateState{now: func() time.Time { return now }, run: autoTestEstimate}
	r := autoTestRequest()
	r.Context = &modelconfig.Context{Mode: "auto", Floor: 4096, Cap: 8192}
	request := settingsRequest(t, f.ex)
	request.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: r}}
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: owner})
	preview, err := f.ex.previewModelLoadChange(ctx, f.st, request)
	if err != nil || len(preview.Errors) != 0 || preview.Auto == nil {
		t.Fatalf("conversion Auto preview: %+v %v", preview, err)
	}
	recheck, err := f.ex.validateTypedLlamaConversion(ctx, f.st, f.rt, preview.Settings)
	if err != nil || recheck == nil {
		t.Fatalf("Auto conversion was not accepted against exact source: %v", err)
	}
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, f.source, append(data, []byte("\n# source drift\n")...))
	if err := recheck(); err == nil {
		t.Fatal("source drift did not invalidate Auto conversion")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadAutoRecalculateAfterNativeBuildDrift(t *testing.T) {
	f, frozen, owner := autoTestFrozenLaunch(t)
	f.st.plat.Runtime.ModelLoads = modelconfig.CloneLoads(frozen.Settings.ModelLoads)
	before := modelconfig.CloneLoads(frozen.Settings.ModelLoads)
	writeTestFile(t, f.st.plat.Runtime.Bin, []byte("replacement synthetic Auto native build"))
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: owner})
	owner.calls = 0
	if _, err := f.ex.ConfigureLaunch(ctx, settings.Configure{Engine: "fake", Settings: frozen.Settings}, func() error { t.Fatal("stale Auto reached rebind"); return nil }); err == nil {
		t.Fatal("old Apply accepted native build drift")
	}
	command := typedFixtureCommand(t, f, f.st.plat.Runtime)
	if _, err := compileTypedLlamaInputs(ctx, f.st, command, f.st.plat.Runtime, true); err == nil {
		t.Fatal("old restart accepted native build drift")
	}
	if owner.calls != 0 {
		t.Fatal("Apply/restart consulted capacity after drift")
	}
	request := settingsRequest(t, f.ex)
	request.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: autoTestRequest()}}
	if err := f.ex.validateModelSettingsLaunch(ctx, request, false); err != nil {
		t.Fatal(err)
	}
	preview, err := f.ex.PreviewLaunch(request, ctx)
	if err != nil || len(preview.Errors) != 0 || preview.Auto == nil || preview.Auto.Resolved == nil {
		t.Fatalf("explicit Recalculate could not recover build drift: %+v %v", preview, err)
	}
	previous := rtLoadResolution(before, "weak-model")
	if owner.calls != 1 || preview.Auto.Resolved.NativeBuildID == previous.NativeBuildID || preview.Auto.ReviewedInputID == previous.ReviewedInputID {
		t.Fatal("Recalculate did not review the new build exactly once")
	}
	if modelAutoConversionSource(preview.Auto.Resolved) != modelAutoConversionSource(previous) {
		t.Fatal("Recalculate lost conversion source")
	}
	if !modelconfig.EqualLoads(before, f.st.plat.Runtime.ModelLoads) || f.st.running || f.st.proc != nil {
		t.Fatal("new preview changed accepted runtime")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsLoadAutoRecalculateStillChecksFrozenSibling(t *testing.T) {
	f, frozen, owner := autoTestFrozenLaunch(t)
	f.st.plat.Runtime.ModelLoads = modelconfig.CloneLoads(frozen.Settings.ModelLoads)
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: owner})
	request := settingsRequest(t, f.ex)
	r := autoTestRequest()
	r.Context = &modelconfig.Context{Mode: "auto", Floor: 4096, Cap: 8192}
	request.ModelChange = &modelconfig.ModelChange{ModelID: "medium-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: r}}
	if err := f.ex.validateModelSettingsLaunch(ctx, request, false); err != nil {
		t.Fatal(err)
	}
	preview, err := f.ex.PreviewLaunch(request, ctx)
	if err != nil || len(preview.Errors) != 0 || preview.Auto == nil {
		t.Fatalf("frozen sibling setup: %+v %v", preview, err)
	}
	for _, model := range []string{"weak-model", "medium-model"} {
		found := false
		for _, load := range preview.Settings.ModelLoads {
			if load.ModelID == model {
				found = modelLoadHasAuto(load.Requested) && load.Resolved != nil
			}
		}
		if !found {
			t.Fatalf("frozen sibling setup did not retain reviewed Auto for %s", model)
		}
	}
	f.st.plat.Runtime.ModelLoads = modelconfig.CloneLoads(preview.Settings.ModelLoads)
	before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
	writeTestFile(t, f.st.plat.Runtime.Bin, []byte("replacement build shared with frozen sibling"))
	owner.calls = 0
	request = settingsRequest(t, f.ex)
	request.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: autoTestRequest()}}
	if err := f.ex.validateModelSettingsLaunch(ctx, request, false); err != nil {
		t.Fatal(err)
	}
	refused, err := f.ex.PreviewLaunch(request, ctx)
	if err != nil {
		t.Fatal(err)
	}
	buildRefusal := false
	for _, reason := range refused.Errors {
		if strings.Contains(reason, "native build changed") {
			buildRefusal = true
		}
	}
	if !buildRefusal || refused.PreviewDigest != "" || owner.calls != 0 {
		t.Fatalf("Recalculate bypassed frozen sibling: errors=%v calls=%d", refused.Errors, owner.calls)
	}
	if !modelconfig.EqualLoads(before, f.st.plat.Runtime.ModelLoads) {
		t.Fatal("refused sibling preview changed accepted settings")
	}
	assertNoSettingsOverride(t, f.ex)
}
