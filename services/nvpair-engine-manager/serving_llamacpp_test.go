// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/fstest"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
)

type llamaServingFixture struct {
	ex       *Executor
	request  settings.Request
	source   string
	startLog string
	root     string
}

func TestModelCopyCustomAliasServingBinding(t *testing.T) {
	test := func(name string, edit func(*testing.T, llamaServingFixture), wantError bool) {
		t.Run(name, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			settingsState(t, f.ex).modelsDir = t.TempDir()
			for i := range f.request.Settings.Serving.Models {
				model := &f.request.Settings.Serving.Models[i]
				data, err := os.ReadFile(filepath.Join(f.root, model.ModelID+".gguf"))
				if err != nil {
					t.Fatal(err)
				}
				model.Artifacts[0].SHA256 = sha256Hex(data)
			}
			applyLlamaServing(t, f, approveLlamaServing(t, f))
			edit(t, f)
			list, err := f.ex.modelSourceFiles(context.Background(), "fake", "medium")
			if err == nil {
				err = hashModelFiles(context.Background(), &list, func(string, int) {})
			}
			if (err != nil) != wantError {
				t.Fatalf("list=%+v err=%v", list, err)
			}
			if err != nil && strings.Contains(err.Error(), f.root) {
				t.Fatal("binding refusal disclosed host path")
			}
			assertNoLlamaSpawn(t, f)
		})
	}
	test("applied exact declaration exports full asserted hashes", func(*testing.T, llamaServingFixture) {}, false)
	test("declaration preset disagreement refuses instead of catalog fallback", func(t *testing.T, f llamaServingFixture) {
		data, err := os.ReadFile(f.source)
		if err != nil {
			t.Fatal(err)
		}
		writeServingFixture(t, f.source, []byte(strings.Replace(string(data), "model = medium.gguf", "model = weak.gguf", 1)))
	}, true)
	test("approved binding change refuses export", func(t *testing.T, f llamaServingFixture) {
		writeServingFixture(t, filepath.Join(f.root, "medium.gguf"), []byte("replaced model"))
	}, true)
	test("changed asserted digest invalidates approved declaration", func(t *testing.T, f llamaServingFixture) {
		st := settingsState(t, f.ex)
		st.mu.Lock()
		st.plat.Runtime.Serving.Models[0].Artifacts[0].SHA256 = strings.Repeat("a", 64)
		st.mu.Unlock()
	}, true)
	t.Run("sample binding is never used as a full hash", func(t *testing.T) {
		f := newLlamaServingFixture(t)
		settingsState(t, f.ex).modelsDir = t.TempDir()
		// The original fixture deliberately asserts a synthetic full digest.
		// Apply binds the actual samples, but must not turn that into a full hash.
		applyLlamaServing(t, f, approveLlamaServing(t, f))
		list, err := f.ex.modelSourceFiles(context.Background(), "fake", "medium")
		if err != nil {
			t.Fatal(err)
		}
		err = hashModelFiles(context.Background(), &list, func(string, int) {})
		if err == nil || !strings.Contains(err.Error(), "full SHA256 disagrees") {
			t.Fatalf("full assertion mismatch=%v", err)
		}
		assertNoLlamaSpawn(t, f)
	})
}

func TestLlamaServingRetentionContract(t *testing.T) {
	for _, location := range []string{"global preset", "model preset", "CLI", "environment"} {
		t.Run(location, func(t *testing.T) {
			for _, value := range []string{"-1", "17", "0", "-2", "-01"} {
				t.Run(value, func(t *testing.T) {
					f := newLlamaServingFixture(t)
					data, err := os.ReadFile(f.source)
					if err != nil {
						t.Fatal(err)
					}
					text := strings.ReplaceAll(strings.ReplaceAll(string(data), "sleep-idle-seconds = 180\n", ""), "sleep-idle-seconds = 60\n", "")
					switch location {
					case "global preset":
						text = strings.Replace(text, "[*]\n", "[*]\nsleep-idle-seconds = "+value+"\n", 1)
					case "model preset":
						text = strings.Replace(text, "[medium]\n", "[medium]\nsleep-idle-seconds = "+value+"\n", 1)
					case "CLI":
						f.request.Settings.LaunchText += " --sleep-idle-seconds " + value
					case "environment":
						f.request.Settings.LaunchText = "LLAMA_ARG_SLEEP_IDLE_SECONDS=" + value + " " + f.request.Settings.LaunchText
					}
					writeServingFixture(t, f.source, []byte(text))
					preview := previewSettings(t, f.ex, f.request)
					valid := value == "-1" || value == "17"
					if (len(preview.Errors) == 0) != valid {
						t.Fatalf("retention %s: errors=%v", value, preview.Errors)
					}
					if valid {
						if preview.Serving.Configured[0].Retention != value {
							t.Fatalf("retention lost string value: %+v", preview.Serving.Configured[0])
						}
						applyLlamaServing(t, f, preview)
						state, err := f.ex.LaunchSettings("fake")
						if err != nil || state.ServingReadback.Configured[0].Retention != value {
							t.Fatalf("readback lost retention: %+v %v", state.ServingReadback, err)
						}
					}
				})
			}
		})
	}
}

func TestServingRetentionDisabledIsExact(t *testing.T) {
	for _, value := range []string{"-1", "0", "1", "-2", "", "-01", " -1"} {
		t.Run(value, func(t *testing.T) {
			if got := servingRetentionDisabled(engines.ServingEnvelope{Retention: value}); got != (value == "-1") {
				t.Fatalf("disabled(%q)=%v", value, got)
			}
		})
	}
}

func TestLlamaServingPreparedDigestChangesWithApprovedInputs(t *testing.T) {
	test := func(name string, edit func(*llamaServingFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			before := approveLlamaServing(t, f).Serving.Configured[0].NativeSourceDigest
			edit(&f)
			after := approveLlamaServing(t, f).Serving.Configured[0].NativeSourceDigest
			if before == after {
				t.Fatal("approved input change omitted from prepared digest")
			}
		})
	}
	editSource := func(f *llamaServingFixture, old, new string) {
		t.Helper()
		data, err := os.ReadFile(f.source)
		if err != nil {
			t.Fatal(err)
		}
		text := strings.Replace(string(data), old, new, 1)
		if text == string(data) {
			t.Fatal("fixture replacement missed")
		}
		writeServingFixture(t, f.source, []byte(text))
	}
	test("source bytes", func(f *llamaServingFixture) { editSource(f, "fit = off", "# changed source\nfit = off") })
	test("context envelope", func(f *llamaServingFixture) { editSource(f, "ctx-size = 16384", "ctx-size = 32768") })
	test("retention", func(f *llamaServingFixture) { editSource(f, "sleep-idle-seconds = 180", "sleep-idle-seconds = -1") })
	test("placement", func(f *llamaServingFixture) { editSource(f, "gpu-layers = 12", "gpu-layers = 13") })
	test("router capacity", func(f *llamaServingFixture) {
		f.request.Settings.LaunchText = strings.Replace(f.request.Settings.LaunchText, "--models-max 1", "--models-max 2", 1)
	})
	test("output declaration", func(f *llamaServingFixture) { f.request.Settings.Serving.Models[0].MaxOutputTokens++ })
	test("content assertion", func(f *llamaServingFixture) {
		f.request.Settings.Serving.Models[0].Artifacts[0].SHA256 = strings.Repeat("d", 64)
	})
}

func newLlamaServingFixture(t *testing.T) llamaServingFixture {
	t.Helper()
	ex := settingsExecutor(t, false)
	st := settingsState(t, ex)
	st.manifest.Family = "llamacpp"
	reg := loadWithOverrides(t, t.TempDir())
	manifest, ok := reg.Get("llamacpp")
	if !ok {
		t.Fatal("missing bundled llama manifest")
	}
	platform, ok := manifest.HostPlatform()
	if !ok {
		t.Fatal("missing host llama platform")
	}
	port := st.plat.Runtime.Port
	platform.Runtime.Bin = fakeEngineBin
	platform.Runtime.Port = port
	platform.Runtime.Ready = &Probe{HTTP: "http://127.0.0.1:{port}/", Status: 200, TimeoutS: 10}
	platform.Runtime.Health = nil
	root := t.TempDir()
	startLog := filepath.Join(root, "starts.jsonl")
	platform.Runtime.Env = map[string]string{"FAKE_ADDR": "127.0.0.1:{port}", "FAKE_START_LOG": startLog}
	graftPlatform(t, ex, *platform)
	data, err := os.ReadFile(filepath.Join("testdata", "serving", "llamacpp.ini"))
	if err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(root, "models.ini")
	writeServingFixture(t, source, data)
	writeServingFixture(t, filepath.Join(root, "medium.gguf"), []byte("synthetic medium model"))
	writeServingFixture(t, filepath.Join(root, "weak.gguf"), []byte("synthetic weak model"))
	args, err := formatLaunchText([]string{"--models-preset", source, "--models-max", "1"})
	if err != nil {
		t.Fatal(err)
	}
	request := settingsRequest(t, ex)
	request.Settings.LaunchText = args
	ram, vram := int64(4096), int64(0)
	request.Settings.Serving = &engines.ServingConfig{Version: 1}
	for _, id := range []string{"medium", "weak"} {
		request.Settings.Serving.Models = append(request.Settings.Serving.Models, engines.ServingModel{
			ModelID: id, MaxOutputTokens: 128, Memory: engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram},
			Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: strings.Repeat("a", 64), Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}}},
		})
	}
	return llamaServingFixture{ex: ex, request: request, source: source, startLog: startLog, root: root}
}

func approveLlamaServing(t *testing.T, f llamaServingFixture) settings.Preview {
	t.Helper()
	preview := previewSettings(t, f.ex, f.request)
	if len(preview.Errors) != 0 || preview.Conflict != nil {
		t.Fatalf("valid finite preview rejected: %v", preview.Errors)
	}
	return preview
}

func applyLlamaServing(t *testing.T, f llamaServingFixture, preview settings.Preview) {
	t.Helper()
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	if pending := settingsState(t, f.ex).plat.Runtime.servingSnapshot; pending != nil {
		if _, err := os.Stat(pending.Path); err != nil {
			t.Fatal("apply did not prepare a cold launch snapshot")
		}
	}
}

func assertNoLlamaSpawn(t *testing.T, f llamaServingFixture) {
	t.Helper()
	if _, err := os.Stat(f.startLog); !os.IsNotExist(err) {
		t.Fatalf("invalid serving launch spawned a process: %v", err)
	}
	st := settingsState(t, f.ex)
	st.mu.Lock()
	proc := st.proc
	st.mu.Unlock()
	if proc != nil {
		t.Fatal("invalid serving launch installed a process")
	}
}

func TestLlamaServingPreviewDerivesDifferentModelsWithoutEffects(t *testing.T) {
	f := newLlamaServingFixture(t)
	var calls atomic.Int32
	f.ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("unexpected preview engine call")
	})
	preview := approveLlamaServing(t, f)
	if preview.Serving == nil || len(preview.Serving.Configured) != 2 || len(preview.Serving.LaunchValidated) != 0 || len(preview.Serving.Observed) != 0 {
		t.Fatal("preview did not distinguish configured from runtime evidence")
	}
	medium, weak := preview.Serving.Configured[0], preview.Serving.Configured[1]
	if medium.PerSlotContextTokens != 8192 || medium.TotalContextTokens != 16384 || medium.Slots != 2 || medium.RouterCapacity != 1 || medium.Retention != "180" || !slices.Contains(medium.Placement, "gpu-layers=12") {
		t.Fatal("medium envelope differs from its native allocation")
	}
	if weak.PerSlotContextTokens != 2048 || weak.TotalContextTokens != 4096 || weak.Slots != 1 || weak.Retention != "60" || !slices.Contains(weak.Placement, "kv-unified-per-slot=2048") {
		t.Fatal("weak unified-KV envelope differs from its native allocation")
	}
	for _, model := range preview.Settings.Serving.Models {
		if len(model.ExpectedNativeSourceDigest) != 64 || model.Artifacts[0].Binding == nil {
			t.Fatal("preview did not return approval digest and candidate binding")
		}
	}
	if f.request.Settings.Serving.Models[0].Artifacts[0].Binding != nil {
		t.Fatal("preview aliased its caller's declaration")
	}
	assertNoSettingsOverride(t, f.ex)
	assertNoLlamaSpawn(t, f)
	if calls.Load() != 0 {
		t.Fatal("preview probed or loaded an engine")
	}
	if _, err := os.Stat(filepath.Join(f.ex.baseDir, "serving")); !os.IsNotExist(err) {
		t.Fatalf("preview wrote a snapshot: %v", err)
	}
}

func TestLlamaServingChildArgvHasNoImplicitEnvelopeOverrides(t *testing.T) {
	f := newLlamaServingFixture(t)
	preview := approveLlamaServing(t, f)
	applyLlamaServing(t, f, preview)
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
	for _, forbidden := range []string{"--ctx-size", "--parallel", "--sleep-idle-seconds", "--cache-ram"} {
		if slices.Contains(args, forbidden) {
			t.Fatalf("actual child argv contains overriding router flag %s", forbidden)
		}
	}
	index := slices.Index(args, "--models-preset")
	if index < 0 || index+1 >= len(args) || args[index+1] == f.source {
		t.Fatal("actual child argv did not select an immutable snapshot")
	}
	snapshot := args[index+1]
	content, err := os.ReadFile(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	sections, err := parseLlamaServingPreset(content)
	if err != nil {
		t.Fatal(err)
	}
	if sections["medium"]["ctx-size"] != "16384" || sections["medium"]["parallel"] != "2" || sections["weak"]["ctx-size"] != "4096" || sections["weak"]["parallel"] != "1" {
		t.Fatal("snapshot lost per-model native allocation")
	}
	if !filepath.IsAbs(sections["medium"]["model"]) || sections["medium"]["model"] != filepath.Join(f.root, "medium.gguf") {
		t.Fatal("snapshot does not select the original absolute model path")
	}
	// A subsequent read/preview must leave the active generation intact.
	request := settingsRequest(t, f.ex)
	if got := previewSettings(t, f.ex, request); len(got.Errors) != 0 {
		t.Fatal(got.Errors)
	}
	if _, err := os.Stat(snapshot); err != nil {
		t.Fatal("readback removed an active generation")
	}
	state, err := f.ex.LaunchSettings("fake")
	if err != nil || len(state.ServingReadback.LaunchValidated) != 2 || len(state.ServingReadback.Observed) != 0 {
		t.Fatalf("owned launch readback missing or falsely observed: %v", err)
	}
	if err := f.ex.Stop("fake"); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for {
		_, err := os.Stat(snapshot)
		if os.IsNotExist(err) {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("exited process snapshot not released: %v", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// writeOwnerRouterPreset replaces the fixture source with a preset shaped
// like the owner's approved MSI router preset: a top-level version key,
// global threads and Flash Attention, and per-model context, slots, batch
// sizes, K/V cache types, GPU offload and idle sleep.
func writeOwnerRouterPreset(t *testing.T, f llamaServingFixture) {
	t.Helper()
	writeServingFixture(t, f.source, []byte(`version = 1

[*]
fit = off
load-on-startup = false
threads = 8
threads-batch = 8
flash-attn = on
kv-offload = false
op-offload = false

[medium]
model = medium.gguf
ctx-size = 32768
parallel = 1
batch-size = 2048
ubatch-size = 512
cache-type-k = q8_0
cache-type-v = q8_0
gpu-layers = 0
sleep-idle-seconds = 600

[weak]
model = weak.gguf
ctx-size = 131072
parallel = 1
batch-size = 2048
ubatch-size = 512
cache-type-k = q8_0
cache-type-v = q8_0
gpu-layers = 8
sleep-idle-seconds = 600
`))
}

// Problem 1 (MSI install, 2026-10-23): the typed serving preview rejected the
// owner's native preset because its knobs were outside the adapter subset.
func TestLlamaServingAcceptsOwnerRouterPreset(t *testing.T) {
	f := newLlamaServingFixture(t)
	writeOwnerRouterPreset(t, f)
	preview := previewSettings(t, f.ex, f.request)
	if len(preview.Errors) != 0 || preview.Conflict != nil {
		t.Fatalf("owner router preset rejected by typed preview: %v", preview.Errors)
	}
	if preview.Serving == nil || len(preview.Serving.Configured) != 2 {
		t.Fatal("preview did not derive one envelope per declared model")
	}
	medium, weak := preview.Serving.Configured[0], preview.Serving.Configured[1]
	if medium.TotalContextTokens != 32768 || medium.Slots != 1 || medium.Retention != "600" {
		t.Fatalf("medium envelope differs from its preset: %+v", medium)
	}
	if weak.TotalContextTokens != 131072 || weak.Slots != 1 || weak.Retention != "600" {
		t.Fatalf("weak envelope differs from its preset: %+v", weak)
	}
	for _, want := range []string{"batch-size=2048", "ubatch-size=512", "cache-type-k=q8_0", "cache-type-v=q8_0", "flash-attn=on", "gpu-layers=8", "kv-offload=false", "op-offload=false"} {
		if !slices.Contains(weak.Placement, want) {
			t.Fatalf("weak placement lacks %s: %v", want, weak.Placement)
		}
	}
	assertNoSettingsOverride(t, f.ex)
	assertNoLlamaSpawn(t, f)
}

// Problem 2 (MSI install, 2026-10-23): both models ran at context 8192 because
// a router CLI --ctx-size overrode every per-model preset value. On the
// canonical typed path the child argv must carry no bundled router default and
// the snapshot must carry the per-model values.
func TestLlamaServingOwnerRouterPresetReachesChild(t *testing.T) {
	f := newLlamaServingFixture(t)
	writeOwnerRouterPreset(t, f)
	preview := approveLlamaServing(t, f)
	applyLlamaServing(t, f, preview)
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
	for _, forbidden := range []string{"--ctx-size", "--parallel", "--sleep-idle-seconds", "--cache-ram"} {
		if slices.Contains(args, forbidden) {
			t.Fatalf("router argv carries bundled default %s, which overrides every per-model preset value", forbidden)
		}
	}
	index := slices.Index(args, "--models-preset")
	if index < 0 || index+1 >= len(args) || args[index+1] == f.source {
		t.Fatal("router argv did not select an immutable snapshot")
	}
	content, err := os.ReadFile(args[index+1])
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(string(content), "\n") {
		if strings.HasPrefix(line, "version") {
			t.Fatal("snapshot materialized the reserved version key")
		}
	}
	sections, err := parseLlamaServingPreset(content)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]map[string]string{
		"medium": {"ctx-size": "32768", "parallel": "1", "sleep-idle-seconds": "600", "threads": "8", "threads-batch": "8", "flash-attn": "on", "batch-size": "2048", "cache-type-k": "q8_0", "kv-offload": "false", "op-offload": "false"},
		"weak":   {"ctx-size": "131072", "parallel": "1", "sleep-idle-seconds": "600", "threads": "8", "gpu-layers": "8", "ubatch-size": "512", "cache-type-v": "q8_0", "kv-offload": "false", "op-offload": "false"},
	}
	for model, options := range want {
		for key, value := range options {
			if sections[model][key] != value {
				t.Fatalf("snapshot %s lost %s = %s (got %q)", model, key, value, sections[model][key])
			}
		}
	}
}

// The pinned preset parser reverses negative aliases using is_truthy, which
// recognizes enabled as well as on/true/1. Positive values stay native-owned.
func TestLlamaServingOffloadPresetPolarity(t *testing.T) {
	test := func(name, key, value, want string) {
		t.Run(name, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			writeOwnerRouterPreset(t, f)
			data, err := os.ReadFile(f.source)
			if err != nil {
				t.Fatal(err)
			}
			text := strings.Replace(string(data), key+" = false", "no-"+key+" = "+value, 1)
			writeServingFixture(t, f.source, []byte(text))
			preview := approveLlamaServing(t, f)
			for _, envelope := range preview.Serving.Configured {
				if !slices.Contains(envelope.Placement, key+"="+want) {
					t.Fatalf("negative preset alias changed native polarity: placement=%v, want %s=%s", envelope.Placement, key, want)
				}
			}
			assertLlamaOffloadChild(t, f, preview, key, want, "")
		})
	}
	test("KV true disables", "kv-offload", "true", "false")
	test("KV false enables", "kv-offload", "false", "true")
	test("KV enabled disables", "kv-offload", "enabled", "false")
	test("KV other native value enables", "kv-offload", "native-value", "true")
	test("operation true disables", "op-offload", "true", "false")
	test("operation false enables", "op-offload", "false", "true")
	test("operation enabled disables", "op-offload", "enabled", "false")
	test("operation disabled enables", "op-offload", "disabled", "true")
}

func TestLlamaServingOffloadCLIPolarity(t *testing.T) {
	test := func(name, flag, key, want string) {
		t.Run(name, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			writeOwnerRouterPreset(t, f)
			f.request.Settings.LaunchText += " " + flag
			preview := approveLlamaServing(t, f)
			for _, envelope := range preview.Serving.Configured {
				if !slices.Contains(envelope.Placement, key+"="+want) {
					t.Fatalf("CLI alias changed native polarity: placement=%v, want %s=%s", envelope.Placement, key, want)
				}
			}
			assertLlamaOffloadChild(t, f, preview, key, want, flag)
		})
	}
	test("short KV disable", "-nkvo", "kv-offload", "false")
	test("long KV disable", "--no-kv-offload", "kv-offload", "false")
	test("long operation disable", "--no-op-offload", "op-offload", "false")
	test("short KV enable", "-kvo", "kv-offload", "true")
	test("long operation enable", "--op-offload", "op-offload", "true")
}

func assertLlamaOffloadChild(t *testing.T, f llamaServingFixture, preview settings.Preview, key, want, flag string) {
	t.Helper()
	applyLlamaServing(t, f, preview)
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
	if flag != "" && !slices.Contains(args, flag) {
		t.Fatalf("child lost explicit CLI token %q: %v", flag, args)
	}
	index := slices.Index(args, "--models-preset")
	if index < 0 || index+1 >= len(args) || args[index+1] == f.source {
		t.Fatal("child did not select an immutable snapshot")
	}
	data, err = os.ReadFile(args[index+1])
	if err != nil {
		t.Fatal(err)
	}
	// Inspect literal generated INI fields independently of the adapter parser.
	values := map[string]string{}
	section := ""
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "[") {
			section = strings.TrimSuffix(strings.TrimPrefix(line, "["), "]")
		} else if name, value, ok := strings.Cut(line, " = "); ok && name == key {
			values[section] = value
		}
	}
	for _, model := range []string{"medium", "weak"} {
		if values[model] != want {
			t.Fatalf("snapshot %s %s=%q, want %q", model, key, values[model], want)
		}
	}
}

func TestLlamaServingExplicitNativePrecedence(t *testing.T) {
	test := func(name, text string, wantMedium, wantWeak int64) {
		t.Run(name, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			if strings.HasPrefix(text, "LLAMA_") {
				f.request.Settings.LaunchText = text + " " + f.request.Settings.LaunchText
			} else {
				f.request.Settings.LaunchText += " " + text
			}
			preview := approveLlamaServing(t, f)
			if preview.Serving.Configured[0].TotalContextTokens != wantMedium || preview.Serving.Configured[1].TotalContextTokens != wantWeak {
				t.Fatal("effective preview does not follow native CLI over model preset over global preset")
			}
			applyLlamaServing(t, f, preview)
			state, err := f.ex.LaunchSettings("fake")
			if err != nil {
				t.Fatal(err)
			}
			tokens, err := parseLaunchText(state.LaunchText)
			if err != nil || !slices.Contains(tokens, strings.Fields(text)[0]) {
				t.Fatalf("explicit envelope option was lost: %v", err)
			}
			assertNoLlamaSpawn(t, f)
		})
	}
	test("CLI overrides model preset", "--ctx-size 32768", 32768, 32768)
	test("model preset overrides inherited child environment", "LLAMA_ARG_CTX_SIZE=32768", 16384, 4096)
}

func TestLlamaServingUnderscoreArgvNativePrecedence(t *testing.T) {
	for _, text := range []string{"--ctx_size 32768", "--ctx_size=32768", "--no_kv_unified", "--no_kv-unified"} {
		t.Run(text, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			// Also exercise snapshot replacement of the normalized source flag.
			f.request.Settings.LaunchText = strings.Replace(f.request.Settings.LaunchText, "--models-preset", "--models_preset", 1) + " " + text
			preview := approveLlamaServing(t, f)
			applyLlamaServing(t, f, preview)
			if err := f.ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile(f.startLog)
			if err != nil {
				t.Fatal(err)
			}
			var argv []string
			if err := json.Unmarshal(bytes.TrimSpace(data), &argv); err != nil {
				t.Fatal(err)
			}
			// Independent native oracle: pinned arg.cpp normalizes long option
			// names; server-models overlays explicit router CLI on each preset.
			var source string
			cli := map[string]string{}
			for i := 0; i < len(argv); i++ {
				name := argv[i]
				if strings.HasPrefix(name, "--") {
					name = strings.ReplaceAll(name, "_", "-")
				}
				switch name {
				case "--models-preset":
					i++
					source = argv[i]
				case "--ctx-size":
					i++
					cli["ctx-size"] = argv[i]
				case "--no-kv-unified":
					cli["kv-unified"] = "false"
				}
			}
			if source == "" || source == f.source || slices.Contains(argv, "--models_preset") {
				t.Fatal("native argv can reopen mutable source instead of snapshot")
			}
			data, err = os.ReadFile(source)
			if err != nil {
				t.Fatal(err)
			}
			// Read only literal generated INI fields, without PAIR's option parser.
			models := map[string]map[string]string{}
			section := ""
			for _, line := range strings.Split(string(data), "\n") {
				if strings.HasPrefix(line, "[") {
					section = strings.TrimSuffix(strings.TrimPrefix(line, "["), "]")
					models[section] = map[string]string{}
				} else if key, value, ok := strings.Cut(line, " = "); ok {
					models[section][key] = value
				}
			}
			for _, envelope := range preview.Serving.Configured {
				values := models[envelope.ModelID]
				for key, value := range cli {
					values[key] = value
				}
				total, err := strconv.ParseInt(values["ctx-size"], 10, 64)
				if err != nil {
					t.Fatal(err)
				}
				if total != envelope.TotalContextTokens || (strings.Contains(text, "ctx_size") && total != 32768) ||
					(strings.Contains(text, "kv") && values["kv-unified"] != "false") ||
					!slices.Contains(envelope.Placement, "kv-unified="+values["kv-unified"]) {
					t.Fatal("actual native CLI-over-preset allocation differs from approval")
				}
			}
		})
	}
}

func TestLlamaServingRejectsInvalidInputsWithoutEffects(t *testing.T) {
	tests := []struct {
		name string
		text string
		from string
		to   string
	}{
		{name: "duplicate short long aliases", text: "--ctx-size 8192 -c 8192"},
		{name: "conflicting repeated setting", text: "--ctx-size 8192 --ctx-size 16384"},
		{name: "CLI and environment conflict", text: "LLAMA_ARG_CTX_SIZE=8192 --ctx-size 8192"},
		{name: "invalid shadowed native environment", text: "LLAMA_ARG_CTX_SIZE=auto"},
		{name: "automatic context", text: "--ctx-size auto"},
		{name: "zero context", text: "--ctx-size 0"},
		{name: "negative context", text: "--ctx-size -1"},
		{name: "overflow context", text: "--ctx-size 2147483648"},
		{name: "automatic slots", text: "--parallel -1"},
		{name: "overflow slots", text: "--parallel 9223372036854775808"},
		{name: "insufficient allocation after CLI override", text: "--ctx-size 1024"},
		{name: "short bundle", text: "-c8192"},
		{name: "router capacity unlimited", text: "--models-max 0"},
		{name: "unsupported model selection", text: "--model another.gguf"},
		{name: "unsupported model alias", text: "--alias different"},
		{name: "unsupported inherited model selection", text: "LLAMA_ARG_MODEL=another.gguf"},
		{name: "unsupported draft selection", text: "--spec-draft-model another.gguf"},
		{name: "unsupported projector selection", text: "--mmproj another.gguf"},
		{name: "unsupported adapter selection", text: "--lora another.gguf"},
		{name: "unsupported context checkpoints", text: "--ctx-checkpoints 2"},
		{name: "unsupported prompt cache file", text: "--prompt-cache cache.bin"},
		{name: "unknown inheritance", from: "gpu-layers = 12", to: "extends = weak"},
		{name: "unknown include", from: "gpu-layers = 12", to: "include = extra.ini"},
		{name: "substitution", from: "medium.gguf", to: "${MODEL}"},
		{name: "startup load", from: "load-on-startup = false", to: "load-on-startup = true"},
		{name: "missing model", from: "medium.gguf", to: "missing.gguf"},
		{name: "missing split shard", from: "medium.gguf", to: "medium-00001-of-00002.gguf"},
		{name: "ambiguous KV allocation", from: "kv-unified = false", to: "kv-unified = true"},
		{name: "automatic fitting", from: "fit = off", to: "fit = on"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			var calls atomic.Int32
			f.ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
				calls.Add(1)
				return nil, errors.New("unexpected invalid-configuration engine call")
			})
			if tc.from != "" {
				data, err := os.ReadFile(f.source)
				if err != nil {
					t.Fatal(err)
				}
				writeServingFixture(t, f.source, []byte(strings.ReplaceAll(string(data), tc.from, tc.to)))
			}
			if strings.HasPrefix(tc.text, "LLAMA_") {
				f.request.Settings.LaunchText = tc.text + " " + f.request.Settings.LaunchText
			} else {
				f.request.Settings.LaunchText += " " + tc.text
			}
			preview := previewSettings(t, f.ex, f.request)
			if len(preview.Errors) == 0 {
				t.Fatal("invalid native envelope accepted")
			}
			if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: f.request.Settings}, func() error { t.Fatal("invalid apply rebound ports"); return nil }); err == nil {
				t.Fatal("invalid apply succeeded")
			}
			assertNoSettingsOverride(t, f.ex)
			assertNoLlamaSpawn(t, f)
			if calls.Load() != 0 {
				t.Fatal("invalid configuration reached an engine")
			}
		})
	}
}

func TestLlamaServingChangedSourceAfterPreviewPreservesSettings(t *testing.T) {
	f := newLlamaServingFixture(t)
	preview := approveLlamaServing(t, f)
	writeServingFixture(t, f.source, []byte("[*]\nctx-size = 4096\n"))
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { t.Fatal("conflicted apply rebound ports"); return nil }); err == nil {
		t.Fatal("changed source accepted after preview")
	}
	assertNoSettingsOverride(t, f.ex)
	assertNoLlamaSpawn(t, f)
}

// Keep this driven by the production table: adding an option must automatically
// exercise every native spelling and environment alias at preview and apply.
func TestLlamaServingOptionTableRejectsConflictsWithoutEffects(t *testing.T) {
	f := newLlamaServingFixture(t)
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	saved := readSettingsOverride(t, f.ex)
	var calls atomic.Int32
	f.ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("unexpected rejected-option engine call")
	})
	values := map[string][2]string{
		"models-preset": {f.source, f.source + ".different"}, "models-max": {"1", "2"},
		"ctx-size": {"8192", "16384"}, "parallel": {"1", "2"},
		"kv-unified-per-slot": {"2048", "4096"}, "n-predict": {"128", "256"},
		"gpu-layers": {"1", "2"}, "split-mode": {"layer", "none"},
		"tensor-split": {"1,1", "2,1"}, "main-gpu": {"0", "1"}, "device": {"none", "CPU"},
		"sleep-idle-seconds": {"60", "120"}, "cache-ram": {"512", "1024"}, "fit": {"off", "false"},
		"batch-size": {"512", "1024"}, "ubatch-size": {"128", "256"},
		"cache-type-k": {"f16", "q8_0"}, "cache-type-v": {"f16", "q8_0"}, "flash-attn": {"on", "off"},
	}
	for _, option := range llamaServingOptions {
		t.Run(option.key, func(t *testing.T) {
			pair, validValues := values[option.key]
			if !option.unsupported && !option.bool && !validValues {
				t.Fatal("interpreted option needs two valid conflict-test values")
			}
			if option.unsupported {
				pair = [2]string{"1", "2"}
			}
			base := []string{}
			if option.key != "models-preset" {
				base = append(base, "--models-preset", f.source)
			}
			if option.key != "models-max" {
				base = append(base, "--models-max", "1")
			}
			flagArgs := func(flag, value string) []string {
				if option.bool {
					return []string{flag}
				}
				return []string{flag, value}
			}
			reject := func(t *testing.T, assignments, args []string) {
				t.Helper()
				tokens := append(slices.Clone(assignments), base...)
				tokens = append(tokens, args...)
				text, err := formatLaunchText(tokens)
				if err != nil {
					t.Fatal(err)
				}
				request := f.request
				request.Settings.LaunchText = text
				if preview := previewSettings(t, f.ex, request); len(preview.Errors) == 0 {
					t.Fatal("conflicting or unsupported table entry accepted at preview")
				}
				if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: request.Settings}, func() error {
					t.Fatal("rejected table entry rebound ports")
					return nil
				}); err == nil {
					t.Fatal("conflicting or unsupported table entry accepted at apply")
				}
				if !reflect.DeepEqual(saved, readSettingsOverride(t, f.ex)) {
					t.Fatal("rejected table entry changed saved configuration")
				}
				assertNoLlamaSpawn(t, f)
				if calls.Load() != 0 {
					t.Fatal("rejected table entry reached HTTP")
				}
			}
			for _, flag := range option.flags {
				t.Run(flag, func(t *testing.T) {
					args := flagArgs(flag, pair[0])
					if !option.unsupported {
						other := flag
						if option.bool {
							other = option.flags[(slices.Index(option.flags, flag)+1)%len(option.flags)]
						}
						args = append(args, flagArgs(other, pair[1])...)
					}
					reject(t, nil, args)
				})
				if strings.HasPrefix(flag, "--") && strings.Contains(flag, "-") {
					// Native normalizes every underscore in a long option before
					// lookup. Derive full and mixed aliases from every table flag.
					for _, alias := range []string{"--" + strings.ReplaceAll(flag[2:], "-", "_"), "--" + strings.Replace(flag[2:], "-", "_", 1)} {
						if alias == flag {
							continue
						}
						t.Run(alias, func(t *testing.T) {
							args := flagArgs(alias, pair[0])
							if !option.unsupported {
								args = append(args, flagArgs(flag, pair[0])...)
							}
							reject(t, nil, args)
						})
					}
				}
			}
			for _, name := range []string{option.env, option.negEnv} {
				if name == "" {
					continue
				}
				t.Run(name, func(t *testing.T) {
					value := pair[0]
					if option.bool {
						value = "true"
					}
					var args []string
					if !option.unsupported {
						args = flagArgs(option.flags[0], pair[0])
					}
					reject(t, []string{name + "=" + value}, args)
				})
			}
		})
	}
}

// Independent of the production table, so deleting a row or native alias cannot
// delete its own coverage. Values are native spellings from pinned arg.cpp.
func TestLlamaServingRejectsPinnedAllocationControls(t *testing.T) {
	tests := []struct{ name, text string }{
		{"main sampling CLI", "--backend-sampling"}, {"main sampling short", "-bs"},
		{"main sampling environment", "LLAMA_ARG_BACKEND_SAMPLING=1"},
		{"main sampling empty environment", "LLAMA_ARG_BACKEND_SAMPLING="},
		{"main sampling false environment", "LLAMA_ARG_BACKEND_SAMPLING=0"},
		{"continuous batch", "--cont-batching"}, {"continuous batch short", "-nocb"},
		{"continuous batch environment", "LLAMA_ARG_CONT_BATCHING=1"},
		{"continuous batch negative environment", "LLAMA_ARG_NO_CONT_BATCHING="},
		{"control vector artifact", "--control-vector synthetic.gguf"},
		{"scaled control vector artifact", "--control-vector-scaled synthetic.gguf:1"},
		{"control vector placement", "--control-vector-layer-range 1 2"},
		{"default speculative allocation", "--spec-default"},
		{"canonical underscore duplicate", "--ctx-size 32768 --ctx_size 32768"},
		{"negative KV alias duplicate", "--no-kv-unified --no_kv_unified"},
		{"full SWA", "--swa-full"},
		{"context shift", "--no-context-shift"}, {"rope scaling", "--rope-scaling yarn"},
		{"rope scale", "--rope-scale 2"}, {"rope base", "--rope-freq-base 10000"},
		{"rope frequency", "--rope-freq-scale 0.5"}, {"yarn context", "--yarn-orig-ctx 4096"},
		{"yarn extrapolation", "--yarn-ext-factor 1"}, {"yarn attention", "--yarn-attn-factor 1"},
		{"yarn slow beta", "--yarn-beta-slow 1"}, {"yarn fast beta", "--yarn-beta-fast 32"},
		{"metadata context override", "--override-kv llama.context_length=int:32768"},
		{"tensor placement", "-ot blk.*=CPU"}, {"CPU MoE", "-cmoe"},
		{"partial CPU MoE", "-ncmoe 2"}, {"CPU FFN", "-ncffn 2"},
		{"weight repacking", "-nr"}, {"host buffer", "--no-host"},
		{"loading placement", "-lm mmap"}, {"lazy placement", "-lzm off"},
		{"NUMA placement", "--numa isolate"}, {"RPC placement", "--rpc 127.0.0.1:1"},
		{"router autoload", "--models-autoload"}, {"router no autoload", "--no-models-autoload"},
		{"router autoload environment", "LLAMA_ARG_MODELS_AUTOLOAD=1"},
		{"router negative autoload environment", "LLAMA_ARG_NO_MODELS_AUTOLOAD="},
		{"warmup", "--warmup"}, {"no warmup", "--no-warmup"},
		{"automatic projector", "--mmproj-auto"}, {"projector placement", "--no-mmproj-offload"},
		{"projector device", "-mmdev CPU"}, {"projector device environment", "MTMD_BACKEND_DEVICE=CPU"},
		{"image minimum", "--image-min-tokens 1"}, {"image maximum", "--image-max-tokens 256"},
		{"image batch", "--mtmd-batch-max-tokens 256"},
		{"draft key cache", "-ctkd q8_0"}, {"draft value cache", "-ctvd q8_0"},
		{"draft tensor placement", "-otd blk.*=CPU"}, {"draft CPU MoE", "-cmoed"},
		{"draft partial CPU MoE", "-ncmoed 2"}, {"draft device", "-devd CPU"},
		{"draft GPU layers", "-ngld 2"}, {"draft sampling placement", "--no-spec-draft-backend-sampling"},
		{"draft batch", "--spec-draft-n-max 16"}, {"speculative selection", "--spec-type draft-simple"},
		{"MTP selection", "--mtp"}, {"DFlash selection", "--dflash"}, {"Eagle3 selection", "--eagle3"},
		{"embedding preset", "--embd-gemma-default"}, {"FIM 1.5b preset", "--fim-qwen-1.5b-default"},
		{"FIM 3b preset", "--fim-qwen-3b-default"}, {"FIM 7b preset", "--fim-qwen-7b-default"},
		{"FIM 7b speculative preset", "--fim-qwen-7b-spec"}, {"FIM 14b speculative preset", "--fim-qwen-14b-spec"},
		{"FIM 30b preset", "--fim-qwen-30b-default"}, {"GPT OSS 20b preset", "--gpt-oss-20b-default"},
		{"GPT OSS 120b preset", "--gpt-oss-120b-default"}, {"vision 4b preset", "--vision-gemma-4b-default"},
		{"vision 12b preset", "--vision-gemma-12b-default"},
		{"negative KV environment aliases", "LLAMA_ARG_KV_UNIFIED=1 LLAMA_ARG_NO_KV_UNIFIED="},
		{"negative prompt cache CLI conflict", "LLAMA_ARG_NO_CACHE_PROMPT= --cache-prompt"},
		{"negative idle cache CLI conflict", "LLAMA_ARG_NO_CACHE_IDLE_SLOTS= --cache-idle-slots"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			if strings.HasPrefix(tc.text, "LLAMA_") || strings.HasPrefix(tc.text, "MTMD_") {
				f.request.Settings.LaunchText = tc.text + " " + f.request.Settings.LaunchText
			} else {
				f.request.Settings.LaunchText += " " + tc.text
			}
			if preview := previewSettings(t, f.ex, f.request); len(preview.Errors) == 0 {
				t.Fatal("unsupported pinned allocation control accepted")
			}
			if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: f.request.Settings}, func() error { return errors.New("unexpected rebind") }); err == nil {
				t.Fatal("unsupported pinned allocation control applied")
			}
			assertNoSettingsOverride(t, f.ex)
			assertNoLlamaSpawn(t, f)
		})
	}
}

func TestLlamaServingStartRechecksApprovedSource(t *testing.T) {
	f := newLlamaServingFixture(t)
	preview := approveLlamaServing(t, f)
	applyLlamaServing(t, f, preview)
	saved := readSettingsOverride(t, f.ex)
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	writeServingFixture(t, f.source, append(data, []byte("\n# changed after preview\n")...))
	if err := f.ex.Start(context.Background(), "fake"); err == nil {
		t.Fatal("changed approved source spawned")
	}
	if !reflect.DeepEqual(saved, readSettingsOverride(t, f.ex)) {
		t.Fatal("rejected start changed saved configuration")
	}
	assertNoLlamaSpawn(t, f)
}

func TestLlamaServingFinalRecheckPreventsSpawn(t *testing.T) {
	f := newLlamaServingFixture(t)
	preview := approveLlamaServing(t, f)
	applyLlamaServing(t, f, preview)
	st := settingsState(t, f.ex)
	st.opMu.Lock()
	defer st.opMu.Unlock()
	st.plat.Runtime.servingSnapshot.Release()
	st.plat.Runtime.servingSnapshot = nil // simulate recovery, which must prepare again
	vars := st.pathVars()
	vars["host"] = "127.0.0.1"
	vars["port"] = strconv.Itoa(st.port)
	prepare := func(root string, data []byte) (string, func(), error) {
		path, release, err := prepareServingSnapshot(root, data)
		if err != nil {
			return path, release, err
		}
		source, err := os.ReadFile(f.source)
		if err != nil {
			release()
			return "", nil, err
		}
		writeServingFixture(t, f.source, append(source, []byte("\n# changed after snapshot\n")...))
		return path, release, nil
	}
	err := f.ex.bringUpProcessWithSnapshot(context.Background(), st, "fake", st.plat.Runtime, st.port, vars, prepare)
	if err == nil {
		t.Error("source changed after preview and preparation still spawned")
	}
	if st.proc != nil {
		st.proc.stop(time.Second)
		st.proc = nil
	}
	assertNoLlamaSpawn(t, f)
}

func TestLlamaServingRecoveryIsDeterministic(t *testing.T) {
	f := newLlamaServingFixture(t)
	preview := approveLlamaServing(t, f)
	applyLlamaServing(t, f, preview)
	st := settingsState(t, f.ex)
	command, err := launchForState(st, st.port)
	if err != nil {
		t.Fatal(err)
	}
	rt := llamaServingBase(st.plat.Runtime)
	first, err := compileLlamaServing(command, rt, true)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(rt.Serving)
	if err != nil {
		t.Fatal(err)
	}
	var recovered engines.ServingConfig
	if err := json.Unmarshal(encoded, &recovered); err != nil {
		t.Fatal(err)
	}
	rt.Serving = &recovered
	second, err := compileLlamaServing(command, rt, true)
	if err != nil || !reflect.DeepEqual(first, second) {
		t.Fatalf("recovery changed prepared snapshot/digest: %v", err)
	}
}

func TestLlamaServingUndeclaredLoadRoutes(t *testing.T) {
	test := func(name string, call func(*testing.T, *Executor) error) {
		t.Run(name, func(t *testing.T) {
			ex, seen := modelOperationEngine(t)
			st, err := ex.state("variant")
			if err != nil {
				t.Fatal(err)
			}
			st.plat.Runtime.Serving = servingSettingsFixture()
			// A rejected load must reach the serving error before admission.
			act := st.manifest.Actions["warm"]
			act.RestartAfter = true
			st.manifest.Actions["warm"] = act
			// Admission reads this journal. A typed serving error proves it was
			// rejected before that read, rather than merely releasing a guard.
			writeServingFixture(t, ex.desired.path, []byte("{"))
			err = call(t, ex)
			if err == nil || len(seen()) != 0 {
				t.Fatal("undeclared load reached HTTP dispatch")
			}
			if !st.plat.Runtime.Serving.Equal(servingSettingsFixture()) {
				t.Fatal("rejected load changed serving settings")
			}
		})
	}
	test("semantic modelOperation load", func(t *testing.T, ex *Executor) error {
		_, err := ex.ModelLoad(context.Background(), "variant", "undeclared")
		var field *ServingFieldError
		if !errors.As(err, &field) || field.Field != "target" || field.ModelID != "undeclared" {
			t.Fatalf("load did not return typed model field error: %v", err)
		}
		return err
	})
	test("Manager runAction binding", func(t *testing.T, ex *Executor) error {
		var out bytes.Buffer
		manager := NewManager(NewCodec(&out), ex, nil, "")
		id := json.RawMessage(`1`)
		manager.runAction(context.Background(), &Message{ID: &id, Method: "engine:action", Params: json.RawMessage(`{"engine":"variant","action":"warm","params":{"target":"undeclared"}}`)})
		var reply Message
		if err := json.Unmarshal(out.Bytes(), &reply); err != nil {
			t.Fatal(err)
		}
		if reply.Error == nil || !strings.Contains(reply.Error.Message, "undeclared") {
			t.Fatal("generic action bypassed the declared load boundary")
		}
		return errors.New(reply.Error.Message)
	})
}

func TestLlamaServingDeclaredLoadStillDispatches(t *testing.T) {
	ex, seen := modelOperationEngine(t)
	st, err := ex.state("variant")
	if err != nil {
		t.Fatal(err)
	}
	st.plat.Runtime.Serving = servingSettingsFixture()
	if _, err := ex.ModelLoad(context.Background(), "variant", "exact/model"); err != nil {
		t.Fatal(err)
	}
	if len(seen()) != 1 {
		t.Fatal("declared model load did not dispatch once")
	}
}

func TestLlamaServingHFSymlinkRetargetFailsClosed(t *testing.T) {
	f := newLlamaServingFixture(t)
	model := filepath.Join(f.root, "medium.gguf")
	first := filepath.Join(f.root, "blob-one")
	second := filepath.Join(f.root, "blob-two")
	writeServingFixture(t, first, []byte("synthetic blob one"))
	writeServingFixture(t, second, []byte("synthetic blob two"))
	if err := os.Remove(model); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(first, model); err != nil {
		t.Skipf("symlink creation unavailable: %v", err)
	}
	preview := approveLlamaServing(t, f)
	before, err := os.ReadDir(f.root)
	if err != nil {
		t.Fatal(err)
	}
	applyLlamaServing(t, f, preview)
	pending := settingsState(t, f.ex).plat.Runtime.servingSnapshot
	data, err := os.ReadFile(pending.Path)
	if err != nil {
		t.Fatal(err)
	}
	sections, err := parseLlamaServingPreset(data)
	if err != nil {
		t.Fatal(err)
	}
	if sections["medium"]["model"] != model {
		t.Fatal("snapshot replaced the HF symlink with a copy or blob path")
	}
	after, err := os.ReadDir(f.root)
	if err != nil || len(before) != len(after) {
		t.Fatal("HF preparation created model copies")
	}
	if err := os.Remove(model); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(second, model); err != nil {
		t.Fatal(err)
	}
	if err := f.ex.Start(context.Background(), "fake"); err == nil {
		t.Fatal("retargeted source symlink accepted")
	}
	assertNoLlamaSpawn(t, f)
}

func TestLlamaServingLegacyDefaultsUnchanged(t *testing.T) {
	f := newLlamaServingFixture(t)
	st := settingsState(t, f.ex)
	command, err := launchForState(st, st.port)
	if err != nil {
		t.Fatal(err)
	}
	for flag, want := range map[string]string{"--ctx-size": "8192", "--parallel": "1"} {
		index := slices.Index(command.Args, flag)
		if index < 0 || index+1 >= len(command.Args) || command.Args[index+1] != want {
			t.Fatalf("legacy launch lost %s %s", flag, want)
		}
	}
}

func TestLlamaServingMissingSecondShardRejectsApply(t *testing.T) {
	f := newLlamaServingFixture(t)
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	writeServingFixture(t, f.source, bytes.ReplaceAll(data, []byte("medium.gguf"), []byte("medium-00001-of-00002.gguf")))
	writeServingFixture(t, filepath.Join(f.root, "medium-00001-of-00002.gguf"), []byte("synthetic first shard"))
	second := f.request.Settings.Serving.Models[0].Artifacts[0]
	second.Role = "model:2"
	f.request.Settings.Serving.Models[0].Artifacts = append(f.request.Settings.Serving.Models[0].Artifacts, second)
	if got := previewSettings(t, f.ex, f.request); len(got.Errors) == 0 {
		t.Fatal("missing later shard accepted")
	}
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: f.request.Settings}, func() error { return nil }); err == nil {
		t.Fatal("missing later shard applied")
	}
	assertNoSettingsOverride(t, f.ex)
	assertNoLlamaSpawn(t, f)
}

func TestLlamaServingPresetSymlinkRetargetRejectsApply(t *testing.T) {
	f := newLlamaServingFixture(t)
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	first, second := filepath.Join(f.root, "first.ini"), filepath.Join(f.root, "second.ini")
	writeServingFixture(t, first, data)
	writeServingFixture(t, second, data)
	if err := os.Remove(f.source); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(first, f.source); err != nil {
		t.Skipf("symlink creation unavailable: %v", err)
	}
	preview := approveLlamaServing(t, f)
	if err := os.Remove(f.source); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(second, f.source); err != nil {
		t.Fatal(err)
	}
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil }); err == nil {
		t.Fatal("retargeted preset accepted")
	}
	assertNoSettingsOverride(t, f.ex)
	assertNoLlamaSpawn(t, f)
}

func TestLlamaServingApplyRequiresReviewedDigestAndBindings(t *testing.T) {
	f := newLlamaServingFixture(t)
	if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: f.request.Settings}, func() error { return nil }); err == nil {
		t.Fatal("apply accepted unreviewed candidate bindings")
	}
	assertNoSettingsOverride(t, f.ex)
	assertNoLlamaSpawn(t, f)
}

func TestLlamaServingCapDerivesTotalWithoutSourceContext(t *testing.T) {
	f := newLlamaServingFixture(t)
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Replace(string(data), "ctx-size = 16384\n", "", 1)
	text = strings.Replace(text, "parallel = 2", "parallel = 2\nkv-unified-per-slot = 8192", 1)
	writeServingFixture(t, f.source, []byte(text))
	preview := approveLlamaServing(t, f)
	medium := preview.Serving.Configured[0]
	if medium.TotalContextTokens != 16384 || medium.PerSlotContextTokens != 8192 {
		t.Fatal("cap-only allocation does not follow native sizing")
	}
}

func TestLlamaServingContextArithmetic(t *testing.T) {
	for _, tc := range []struct {
		total, slots, cap string
		unified           bool
		want              int64
	}{
		{"16384", "2", "", false, 8192},
		{"16384", "2", "4096", false, 4096},
		{"16384", "2", "4096", true, 4096},
		{"32768", "4", "", false, 8192},
		{"2147483392", "1", "", false, 2147483392},
	} {
		t.Run(fmt.Sprintf("total %s slots %s cap %s unified %t", tc.total, tc.slots, tc.cap, tc.unified), func(t *testing.T) {
			f := newLlamaServingFixture(t)
			data, err := os.ReadFile(f.source)
			if err != nil {
				t.Fatal(err)
			}
			text := strings.Replace(string(data), "ctx-size = 16384", "ctx-size = "+tc.total, 1)
			text = strings.Replace(text, "parallel = 2", "parallel = "+tc.slots, 1)
			text = strings.Replace(text, "kv-unified = false", "kv-unified = "+strconv.FormatBool(tc.unified), 1)
			if tc.cap != "" {
				text = strings.Replace(text, "parallel = "+tc.slots, "parallel = "+tc.slots+"\nkv-unified-per-slot = "+tc.cap, 1)
			}
			writeServingFixture(t, f.source, []byte(text))
			preview := approveLlamaServing(t, f)
			got := preview.Serving.Configured[0]
			if got.PerSlotContextTokens != tc.want || strconv.FormatInt(got.TotalContextTokens, 10) != tc.total || strconv.FormatInt(got.Slots, 10) != tc.slots {
				t.Fatal("incorrect per-slot allocation")
			}
		})
	}
}

func TestLlamaServingRejectsNativePadding(t *testing.T) {
	for _, tc := range []struct{ name, total, slots, cap string }{
		{"unaligned total", "16386", "2", ""},
		{"unaligned sequence", "768", "2", ""},
		{"unaligned cap", "16384", "2", "4097"},
		{"padding exceeds int32", "2147483647", "1", ""},
		{"near limit sequence padding", "2147483392", "2", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			data, err := os.ReadFile(f.source)
			if err != nil {
				t.Fatal(err)
			}
			text := strings.Replace(string(data), "ctx-size = 16384", "ctx-size = "+tc.total, 1)
			text = strings.Replace(text, "parallel = 2", "parallel = "+tc.slots, 1)
			if tc.cap != "" {
				text = strings.Replace(text, "gpu-layers = 12", "gpu-layers = 12\nkv-unified-per-slot = "+tc.cap, 1)
			}
			writeServingFixture(t, f.source, []byte(text))
			preview := previewSettings(t, f.ex, f.request)
			if len(preview.Errors) == 0 {
				t.Fatal("native padding changed the approved allocation but preview accepted it")
			}
			_, err = f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: f.request.Settings}, func() error { t.Fatal("invalid apply rebound"); return nil })
			var field *ServingFieldError
			if !errors.As(err, &field) || !strings.Contains(err.Error(), "medium") || !strings.Contains(err.Error(), "256-token") {
				t.Fatalf("padding rejection lost its model field error: %v", err)
			}
			assertNoLlamaSpawn(t, f)
			assertNoSettingsOverride(t, f.ex)
		})
	}
}

func TestLlamaServingRejectsNoncanonicalModelIDs(t *testing.T) {
	for _, tc := range []struct{ name, medium, weak string }{
		{"normalized collision", "model:q4_k_m", "model:Q4_K_M"},
		{"single rewritten tag", "model:q4_k_m", "weak"},
		{"filename prefix stripped", "model:file-Q4_K_M", "weak"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			data, err := os.ReadFile(f.source)
			if err != nil {
				t.Fatal(err)
			}
			text := strings.Replace(string(data), "[medium]", "["+tc.medium+"]", 1)
			text = strings.Replace(text, "[weak]", "["+tc.weak+"]", 1)
			writeServingFixture(t, f.source, []byte(text))
			f.request.Settings.Serving.Models[0].ModelID = tc.medium
			f.request.Settings.Serving.Models[1].ModelID = tc.weak
			if preview := previewSettings(t, f.ex, f.request); len(preview.Errors) == 0 {
				t.Fatal("native model-ID rewrite accepted")
			}
			if _, err := f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: f.request.Settings}, func() error { return nil }); err == nil {
				t.Fatal("noncanonical IDs applied")
			}
			assertNoLlamaSpawn(t, f)
			assertNoSettingsOverride(t, f.ex)
		})
	}
}

func TestLlamaServingRejectsUnrepresentableResolvedPaths(t *testing.T) {
	for _, dirname := range []string{"models#private", "models;private"} {
		t.Run(dirname, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			data, err := os.ReadFile(f.source)
			if err != nil {
				t.Fatal(err)
			}
			root := filepath.Join(f.root, dirname)
			if err := os.Mkdir(root, 0o700); err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"medium.gguf", "weak.gguf"} {
				writeServingFixture(t, filepath.Join(root, name), []byte("synthetic model"))
			}
			f.source = filepath.Join(root, "models.ini")
			writeServingFixture(t, f.source, data)
			text, err := formatLaunchText([]string{"--models-preset", f.source, "--models-max", "1"})
			if err != nil {
				t.Fatal(err)
			}
			f.request.Settings.LaunchText = text
			preview := previewSettings(t, f.ex, f.request)
			if len(preview.Errors) == 0 {
				t.Fatal("native INI comment in resolved path accepted")
			}
			for _, reason := range preview.Errors {
				if !strings.Contains(reason, "medium") || strings.Contains(reason, root) {
					t.Fatal("preview path error must identify the model without its path")
				}
			}
			_, err = f.ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: f.request.Settings}, func() error { t.Fatal("invalid apply rebound"); return nil })
			var field *ServingFieldError
			if !errors.As(err, &field) || !strings.Contains(err.Error(), "medium") || strings.Contains(err.Error(), root) {
				t.Fatalf("apply lost safe model field error: %v", err)
			}
			// Exercise the spawn boundary directly with a fabricated persisted
			// declaration; it must validate before considering approval digests.
			st := settingsState(t, f.ex)
			rt := st.plat.Runtime
			rt.Serving = f.request.Settings.Serving.Clone()
			args := []string{"--models-preset", f.source, "--models-max", "1"}
			rt.LaunchArgs = &args
			vars := st.pathVars()
			vars["host"], vars["port"] = "127.0.0.1", strconv.Itoa(st.port)
			err = f.ex.bringUpProcess(context.Background(), st, "fake", rt, st.port, vars)
			if !errors.As(err, &field) || !strings.Contains(err.Error(), "medium") || strings.Contains(err.Error(), root) {
				t.Fatalf("spawn boundary lost safe model field error: %v", err)
			}
			assertNoLlamaSpawn(t, f)
			assertNoSettingsOverride(t, f.ex)
		})
	}
}

func TestLlamaServingPreservesLoggingControls(t *testing.T) {
	for _, tc := range []struct {
		name, text string
		inherited  bool
	}{
		{name: "CLI logging switch", text: "--log-disable"},
		{name: "explicit logging environment", text: "LLAMA_ARG_LOG_COLORS=off", inherited: false},
		{name: "inherited logging environment", inherited: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			if tc.inherited {
				t.Setenv("LLAMA_ARG_LOG_COLORS", "off")
			} else if strings.HasPrefix(tc.text, "LLAMA_") {
				f.request.Settings.LaunchText = tc.text + " " + f.request.Settings.LaunchText
			} else {
				f.request.Settings.LaunchText += " " + tc.text
			}
			preview := approveLlamaServing(t, f)
			applyLlamaServing(t, f, preview)
			if err := f.ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			st := settingsState(t, f.ex)
			st.mu.Lock()
			proc := st.proc
			st.mu.Unlock()
			if tc.text == "--log-disable" {
				if !slices.Contains(proc.cmd.Args, "--log-disable") {
					t.Fatal("actual child argv lost logging control")
				}
			} else if !slices.Contains(proc.cmd.Env, "LLAMA_ARG_LOG_COLORS=off") {
				t.Fatal("actual child environment lost logging control")
			}
			index := slices.Index(proc.cmd.Args, "--models-preset")
			if index < 0 || index+1 >= len(proc.cmd.Args) {
				t.Fatal("missing snapshot argument")
			}
			data, err := os.ReadFile(proc.cmd.Args[index+1])
			if err != nil {
				t.Fatal(err)
			}
			sections, err := parseLlamaServingPreset(data)
			if err != nil {
				t.Fatal(err)
			}
			for _, options := range sections {
				for _, key := range []string{"log-disable", "log-colors", "models-preset", "models-max"} {
					if _, exists := options[key]; exists {
						t.Fatalf("router-only option %s was materialized into model preset", key)
					}
				}
			}
		})
	}
}

func TestLlamaServingUnconfiguredLoadUnchanged(t *testing.T) {
	for _, family := range []string{"llamacpp", "other-family"} {
		t.Run(family, func(t *testing.T) {
			ex, seen := modelOperationEngine(t)
			st, err := ex.state("variant")
			if err != nil {
				t.Fatal(err)
			}
			st.manifest.Family = family
			if _, err := ex.ModelLoad(context.Background(), "variant", "undeclared"); err != nil {
				t.Fatal(err)
			}
			want := []recordedRequest{{method: http.MethodPost, path: "/v1/load", body: `{"target":"undeclared","wait":true}`}}
			if !reflect.DeepEqual(seen(), want) {
				t.Fatalf("unconfigured load changed: %#v", seen())
			}
		})
	}
}

func TestLlamaServingUnconfiguredLoadErrorsUnchanged(t *testing.T) {
	// Exact a03f026 dispatch error messages and RPC responses, for both routes.
	for _, route := range []string{"action", "semantic"} {
		for _, tc := range []struct {
			name, want         string
			running, cancelled bool
		}{
			{"cancelled running", `action "warm": Post "http://127.0.0.1:43210/v1/load": context canceled`, true, true},
			{"cancelled Off", `engine "variant" is not running`, false, true},
			{"transport failure", `action "warm": Post "http://127.0.0.1:43210/v1/load": synthetic transport failure`, true, false},
		} {
			t.Run(route+"/"+tc.name, func(t *testing.T) {
				ex, _ := modelOperationEngine(t)
				st, err := ex.state("variant")
				if err != nil {
					t.Fatal(err)
				}
				st.running, st.port = tc.running, 43210
				binding := st.manifest.Operations[engines.OperationLoad]
				binding.ResponseTimeoutS = 0
				st.manifest.Operations[engines.OperationLoad] = binding
				ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
					if tc.cancelled {
						return nil, r.Context().Err()
					}
					return nil, errors.New("synthetic transport failure")
				})
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				if tc.cancelled {
					cancel()
				}
				var out, want bytes.Buffer
				manager := NewManager(NewCodec(&out), ex, nil, "")
				id := json.RawMessage(`1`)
				if route == "action" {
					manager.runAction(ctx, &Message{ID: &id, Method: "engine:action", Params: json.RawMessage(`{"engine":"variant","action":"warm","params":{"target":"undeclared"}}`)})
				} else {
					manager.runModelOperation(ctx, &Message{ID: &id, Method: loadModelMethod, Params: json.RawMessage(`{"engine":"variant","model":"undeclared"}`)})
				}
				if err := NewCodec(&want).RespondError(&id, -32000, tc.want); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(out.Bytes(), want.Bytes()) {
					t.Fatalf("legacy error response changed:\ngot  %s\nwant %s", out.Bytes(), want.Bytes())
				}
			})
		}
	}
}

func servingPipeCommand(t *testing.T) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "pipecommand")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	if out, err := exec.Command("go", "build", "-o", bin, "./testdata/serving/pipecommand").CombinedOutput(); err != nil {
		t.Fatalf("build hermetic pipe command: %v\n%s", err, out)
	}
	return bin
}

func servingPipePID(t *testing.T, file string) int {
	t.Helper()
	deadline := time.NewTimer(3 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		data, err := os.ReadFile(file)
		if err == nil {
			pid, err := strconv.Atoi(string(data))
			if err == nil && pid > 0 {
				t.Cleanup(func() {
					if pidAlive(pid) {
						if err := signalPID(pid, true); err != nil && pidAlive(pid) {
							t.Errorf("clean up fake command: %v", err)
						}
					}
				})
				return pid
			}
		} else if !os.IsNotExist(err) {
			t.Fatal(err)
		}
		select {
		case <-deadline.C:
			t.Fatal("fake command never witnessed its PID")
		case <-tick.C:
		}
	}
}

func TestLlamaServingColdCommandLoadCancelledByStop(t *testing.T) {
	bin := servingPipeCommand(t)
	for _, operation := range []string{"user Stop", "StopAll", "caller cancel"} {
		t.Run(operation, func(t *testing.T) {
			ex := newTestExecutor(t, testEngineManifest(fakeEngineBin))
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			root := t.TempDir()
			parentFile, childFile := filepath.Join(root, "parent"), filepath.Join(root, "child")
			st.plat.Runtime.Mode = "command"
			st.manifest.Actions["warm"] = Action{Cmd: []string{bin, "hold", parentFile, childFile}}
			st.manifest.Operations = map[string]OperationBinding{engines.OperationLoad: {Action: "warm"}}
			ex.actionTimeout = time.Hour
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			loaded := make(chan error, 1)
			go func() { _, err := ex.ModelLoad(ctx, "fake", "synthetic"); loaded <- err }()
			parent, child := servingPipePID(t, parentFile), servingPipePID(t, childFile)
			if !pidAlive(parent) || !pidAlive(child) {
				t.Fatal("fake command tree did not reach live dispatch")
			}
			if st.running {
				t.Fatal("cold command fixture unexpectedly running")
			}
			if st.opMu.TryLock() {
				st.opMu.Unlock()
				t.Fatal("command load did not hold operation lock")
			}
			stopped := make(chan error, 1)
			go func() {
				switch operation {
				case "user Stop":
					stopped <- ex.Stop("fake")
				case "StopAll":
					ex.StopAll()
					stopped <- nil
				case "caller cancel":
					cancel()
					stopped <- nil
				}
			}()
			// Default Cancel kills the command; WaitDelay bounds descendant-held
			// pipes. servingPipePID owns descendant cleanup for this fixture.
			bound := time.NewTimer(stopGrace(Runtime{}) + 3*time.Second)
			defer bound.Stop()
			select {
			case err := <-stopped:
				if err != nil {
					t.Fatal(err)
				}
			case <-bound.C:
				cancel()
				if err := signalPID(parent, true); err != nil && pidAlive(parent) {
					t.Error(err)
				}
				if err := signalPID(child, true); err != nil && pidAlive(child) {
					t.Error(err)
				}
				<-stopped
				t.Fatal("Stop exceeded command cancellation bound")
			}
			select {
			case err := <-loaded:
				// Command cancellation retains the existing exit diagnostic rather
				// than introducing a new RPC error mapping for killed commands.
				if err == nil || !strings.HasPrefix(err.Error(), "action command failed: ") {
					t.Fatalf("cancelled command load reported success or lost diagnostic: %v", err)
				}
			case <-bound.C:
				cancel()
				if err := signalPID(child, true); err != nil && pidAlive(child) {
					t.Error(err)
				}
				<-loaded
				t.Fatal("cancelled command retained the load lock beyond its bound")
			}
			if pidAlive(parent) {
				t.Fatal("cancelled command retained its CLI process")
			}
			if !st.opMu.TryLock() {
				t.Fatal("cancelled command retained operation lock")
			}
			st.opMu.Unlock()
			st.mu.Lock()
			active := len(st.loads.active)
			st.mu.Unlock()
			if active != 0 {
				t.Fatal("cancelled command retained load registration")
			}
		})
	}
}

// exec decides whether a command that finished as its context expired keeps
// its own exit status from Cancel's error (os.ErrProcessDone keeps it). The
// default Cancel must retain Process.Kill's error parity after exit so a
// command racing its deadline can keep its successful exit status.
func TestLlamaServingCommandCancelAfterExitMatchesKill(t *testing.T) {
	bin := servingPipeCommand(t)
	finished := func() *exec.Cmd {
		cmd := exec.CommandContext(context.Background(), bin) // no arguments: exits at once
		var exitErr *exec.ExitError
		if err := cmd.Run(); !errors.As(err, &exitErr) || exitErr.ExitCode() != 2 {
			t.Fatalf("fixture without arguments = %v, want exit code 2", err)
		}
		if cmd.ProcessState == nil {
			t.Fatal("fixture command did not run")
		}
		return cmd
	}
	want := finished().Process.Kill()
	if want == nil {
		t.Fatal("Kill of a waited process succeeded")
	}
	if got := finished().Cancel(); !errors.Is(got, want) {
		t.Fatalf("cancel after exit = %v, want %v", got, want)
	}
}

func TestLlamaServingCommandExitedParentPipeBound(t *testing.T) {
	bin := servingPipeCommand(t)
	for _, operation := range []string{"user Stop", "StopAll", "caller cancel"} {
		t.Run(operation, func(t *testing.T) {
			ex := newTestExecutor(t, testEngineManifest(fakeEngineBin))
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			root := t.TempDir()
			parentFile, childFile := filepath.Join(root, "parent"), filepath.Join(root, "child")
			st.plat.Runtime.Mode = "command"
			st.manifest.Actions["warm"] = Action{Cmd: []string{bin, "exit", parentFile, childFile}}
			st.manifest.Operations = map[string]OperationBinding{engines.OperationLoad: {Action: "warm"}}
			ex.actionTimeout = time.Hour
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			loaded := make(chan error, 1)
			go func() { _, err := ex.ModelLoad(ctx, "fake", "synthetic"); loaded <- err }()
			parent, child := servingPipePID(t, parentFile), servingPipePID(t, childFile)
			witness, stopWitness := context.WithTimeout(context.Background(), 3*time.Second)
			defer stopWitness()
			for pidAlive(parent) && witness.Err() == nil {
				runtime.Gosched()
			}
			if pidAlive(parent) || !pidAlive(child) {
				t.Fatal("did not witness exited parent and live pipe-holding child")
			}
			if st.opMu.TryLock() {
				st.opMu.Unlock()
				t.Fatal("pipe-draining load did not retain operation lock")
			}
			done := make(chan error, 1)
			go func() {
				if operation == "user Stop" {
					done <- ex.Stop("fake")
				} else if operation == "StopAll" {
					ex.StopAll()
					done <- nil
				} else {
					cancel()
					done <- nil
				}
			}()
			// WaitDelay is independently necessary after a parent exits: its PID
			// cannot identify the orphan. No descendant-cleanup claim here;
			// this test owns cleanup even when production correctly closes pipes.
			bound := time.NewTimer(stopGrace(Runtime{}) + 3*time.Second)
			defer bound.Stop()
			select {
			case err := <-loaded:
				if !errors.Is(err, exec.ErrWaitDelay) {
					t.Fatalf("pipe drain lost bounded-wait diagnostic: %v", err)
				}
			case <-bound.C:
				if err := signalPID(child, true); err != nil && pidAlive(child) {
					t.Error(err)
				}
				cancel()
				<-loaded
				<-done
				t.Fatal("output-pipe drain exceeded test-owned bound without WaitDelay")
			}
			select {
			case err := <-done:
				if err != nil {
					t.Fatal(err)
				}
			case <-bound.C:
				t.Fatal("Stop retained lifecycle wait after pipe drain completed")
			}
			if !st.opMu.TryLock() {
				t.Fatal("bounded pipe drain retained operation lock")
			}
			st.opMu.Unlock()
		})
	}
}

func TestLlamaServingShutdownReleasesColdGeneration(t *testing.T) {
	f := newLlamaServingFixture(t)
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	st := settingsState(t, f.ex)
	first := st.plat.Runtime.servingSnapshot.Path
	// A replacement must release the superseded pending generation as well.
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	second := st.plat.Runtime.servingSnapshot.Path
	if first == second {
		t.Fatal("replacement reused a pending generation")
	}
	if _, err := os.Stat(first); !os.IsNotExist(err) {
		t.Fatalf("superseded pending generation survived: %v", err)
	}
	f.ex.StopAll()
	if st.plat.Runtime.servingSnapshot != nil {
		t.Fatal("orderly shutdown retained cold snapshot ownership")
	}
	if _, err := os.Stat(second); !os.IsNotExist(err) {
		t.Fatalf("orderly shutdown leaked cold generation: %v", err)
	}
	assertNoLlamaSpawn(t, f)
}

func TestLlamaServingLegacyNativeOutputUnchanged(t *testing.T) {
	for _, family := range []string{"llamacpp", "other-family"} {
		t.Run(family, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			st := settingsState(t, f.ex)
			st.manifest.Family = family
			marker := "native output preserved verbatim"
			rt := Runtime{Bin: fakeEngineBin, Args: []string{"echo", marker}, Ready: &Probe{HTTP: "http://127.0.0.1:{port}/", Status: http.StatusOK, TimeoutS: 1}}
			err := f.ex.bringUpProcess(context.Background(), st, "fake", rt, st.port, map[string]string{"host": "127.0.0.1", "port": strconv.Itoa(st.port)})
			if err == nil || !strings.Contains(err.Error(), marker) {
				t.Fatalf("unconfigured startup diagnostic lost native output: %v", err)
			}
			logs := st.logs.snapshot()
			if len(logs) != 1 || logs[0].Stream != "stdout" || logs[0].Text != marker {
				t.Fatal("unconfigured launch no longer copies native output exactly")
			}
		})
	}
}

func TestLlamaServingSafeStartFailures(t *testing.T) {
	t.Run("native exit reports stage and exit code", func(t *testing.T) {
		f := newLlamaServingFixture(t)
		f.request.Settings.LaunchText += " --fail-launch"
		applyLlamaServing(t, f, approveLlamaServing(t, f))
		err := f.ex.Start(context.Background(), "fake")
		if err == nil || !strings.Contains(err.Error(), "NE readiness") || !strings.Contains(err.Error(), "exit code 2") {
			t.Fatalf("native exit lost safe failure context: %v", err)
		}
		if strings.Contains(err.Error(), f.root) || strings.Contains(err.Error(), f.ex.baseDir) {
			t.Fatal("NE failure copied a private path")
		}
		logs := settingsState(t, f.ex).logs.snapshot()
		if len(logs) == 0 {
			t.Fatal("NE redacted native output missing from exported log ring")
		}
		for _, line := range logs {
			if strings.Contains(line.Text, f.root) || strings.Contains(line.Text, f.ex.baseDir) {
				t.Fatalf("NE log copied a private path: %q", line.Text)
			}
		}
	})
	t.Run("readiness timeout reports stage and duration", func(t *testing.T) {
		f := newLlamaServingFixture(t)
		st := settingsState(t, f.ex)
		st.plat.Runtime.Env["FAKE_START_DELAY"] = "3s"
		st.plat.Runtime.Ready.TimeoutS = 1
		applyLlamaServing(t, f, approveLlamaServing(t, f))
		err := f.ex.Start(context.Background(), "fake")
		if err == nil || !strings.Contains(err.Error(), "NE readiness") || !strings.Contains(err.Error(), "not ready after 1s") {
			t.Fatalf("readiness timeout lost stage or duration: %v", err)
		}
		if strings.Contains(err.Error(), f.root) || strings.Contains(err.Error(), f.ex.baseDir) {
			t.Fatal("readiness timeout exposed a private path")
		}
	})
	t.Run("preset validation reports stage", func(t *testing.T) {
		f := newLlamaServingFixture(t)
		applyLlamaServing(t, f, approveLlamaServing(t, f))
		writeServingFixture(t, f.source, []byte("[*]\nunsupported = true\n"))
		err := f.ex.Start(context.Background(), "fake")
		if err == nil || !strings.Contains(err.Error(), "NE preset validation") || strings.Contains(err.Error(), f.root) {
			t.Fatalf("validation failure lost safe stage: %v", err)
		}
		assertNoLlamaSpawn(t, f)
	})
}

// Capture the actual OS launch, normalizing only the private generation path.
func capturedLlamaServingLaunch(t *testing.T, ex *Executor) (launchCommand, []byte) {
	t.Helper()
	st := settingsState(t, ex)
	st.mu.Lock()
	proc := st.proc
	st.mu.Unlock()
	if proc == nil {
		t.Fatal("missing owned fake process")
	}
	args := slices.Clone(proc.cmd.Args[1:])
	i := slices.Index(args, "--models-preset")
	if i < 0 || i+1 >= len(args) {
		t.Fatal("missing actual snapshot argument")
	}
	data, err := os.ReadFile(args[i+1])
	if err != nil {
		t.Fatal(err)
	}
	args[i+1] = "<generation>"
	env := map[string]string{}
	for _, assignment := range proc.cmd.Env {
		key, value, ok := strings.Cut(assignment, "=")
		if !ok {
			t.Fatal("invalid actual child environment")
		}
		if key == "LLAMA_ARG_MODELS_PRESET" {
			value = "<generation>"
		}
		env[environmentKey(key)] = value
	}
	return launchCommand{Bin: proc.cmd.Path, Args: args, Env: env}, data
}

func TestLlamaServingFreshExecutorRecoversPersistedOverride(t *testing.T) {
	f := newLlamaServingFixture(t)
	st := settingsState(t, f.ex)
	base := *st.manifest
	base.Platforms = map[string]Platform{runtime.GOOS + "/" + runtime.GOARCH: *st.plat}
	data, err := json.Marshal(base)
	if err != nil {
		t.Fatal(err)
	}
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	first, firstSnapshot := capturedLlamaServingLaunch(t, f.ex)
	if err := f.ex.Stop("fake"); err != nil {
		t.Fatal(err)
	}
	reg := NewRegistry()
	if err := reg.LoadFS(fstest.MapFS{"manifests/fake.json": &fstest.MapFile{Data: data}}, "manifests"); err != nil {
		t.Fatal(err)
	}
	if err := reg.LoadOverrideDir(f.ex.overrideDir); err != nil {
		t.Fatal(err)
	}
	recovered := NewExecutor(reg, NewReporter(nil), func(string, any) {}, f.ex.baseDir)
	recovered.overrideDir = f.ex.overrideDir
	t.Cleanup(recovered.StopAll)
	if !settingsState(t, recovered).plat.Runtime.Serving.Equal(readSettingsOverride(t, f.ex).Serving) {
		t.Fatal("fresh registry did not recover the persisted serving override")
	}
	if err := recovered.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	second, secondSnapshot := capturedLlamaServingLaunch(t, recovered)
	if !reflect.DeepEqual(first, second) || !bytes.Equal(firstSnapshot, secondSnapshot) {
		t.Fatal("fresh executor changed effective launch or snapshot bytes")
	}
}

func waitServingSnapshotReleased(t *testing.T, path string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return
		} else if err != nil {
			t.Fatal("cannot inspect generation lifetime")
		}
		if time.Now().After(deadline) {
			t.Fatal("exited process generation was not released")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestLlamaServingReplacementKeepsRunningGeneration(t *testing.T) {
	f := newLlamaServingFixture(t)
	preview := approveLlamaServing(t, f)
	applyLlamaServing(t, f, preview)
	st := settingsState(t, f.ex)
	active := st.plat.Runtime.servingSnapshot.Path
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	owner := st.proc
	st.mu.Unlock()
	applyLlamaServing(t, f, preview)
	pending := st.plat.Runtime.servingSnapshot.Path
	applyLlamaServing(t, f, preview)
	st.mu.Lock()
	sameProcess := st.proc == owner
	latest := st.plat.Runtime.servingSnapshot.Path
	st.mu.Unlock()
	if !sameProcess || latest == active || latest == pending {
		t.Fatal("unchanged apply did not preserve process and independent pending lease")
	}
	if _, err := os.Stat(pending); !os.IsNotExist(err) {
		t.Fatal("superseded pending lease survived replacement")
	}
	if _, err := os.Stat(active); err != nil {
		t.Fatal("replacement removed the running process generation")
	}
	select {
	case <-owner.done:
		t.Fatal("generation owner exited during pending replacement")
	default:
	}
	if err := f.ex.Stop("fake"); err != nil {
		t.Fatal(err)
	}
	waitServingSnapshotReleased(t, active)
	if _, err := os.Stat(latest); err != nil {
		t.Fatal("process exit released an unrelated pending generation")
	}
	f.ex.StopAll()
	if _, err := os.Stat(latest); !os.IsNotExist(err) {
		t.Fatal("shutdown failed to release remaining pending generation")
	}
}

func TestLlamaServingLoadSerializedWithReplacement(t *testing.T) {
	f := newLlamaServingFixture(t)
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	st := settingsState(t, f.ex)
	st.manifest.Actions["warm"] = Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/ne-test-load"}}
	st.manifest.Operations = map[string]OperationBinding{engines.OperationLoad: {Action: "warm", ModelParam: "target"}}
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	// Prepare B (only weak/Y) while A still governs the running process.
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	a := strings.Index(string(data), "[medium]")
	b := strings.Index(string(data), "[weak]")
	if a < 0 || b < a {
		t.Fatal("invalid model fixture")
	}
	writeServingFixture(t, f.source, append(slices.Clone(data[:a]), data[b:]...))
	request := settingsRequest(t, f.ex)
	request.Settings.Serving.Models = request.Settings.Serving.Models[1:]
	request.Settings.Serving.Models[0].ExpectedNativeSourceDigest = ""
	preview := previewSettings(t, f.ex, request)
	if len(preview.Errors) != 0 {
		t.Fatal(preview.Errors)
	}
	entered, release := make(chan struct{}), make(chan struct{})
	releaseOnce := sync.OnceFunc(func() { close(release) })
	t.Cleanup(releaseOnce)
	original := f.ex.client.Transport
	var reachedB atomic.Bool
	f.ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/ne-test-load" {
			return original.RoundTrip(r)
		}
		close(entered) // authorized X under A, before the HTTP effect
		select {
		case <-release:
		case <-r.Context().Done():
			return nil, r.Context().Err()
		}
		st.mu.Lock()
		config := st.plat.Runtime.Serving.Clone()
		st.mu.Unlock()
		if len(config.Models) != 2 {
			reachedB.Store(true)
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{}`))}, nil
	})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	loadDone := make(chan error, 1)
	go func() {
		_, err := f.ex.Action(ctx, "fake", "warm", json.RawMessage(`{"target":"medium"}`))
		loadDone <- err
	}()
	select {
	case <-entered:
	case <-ctx.Done():
		t.Fatal("load did not reach authorization/dispatch barrier")
	}
	// The barrier must still own the lifecycle lock. Without the fix, let B
	// finish first, then resume X and observe the invalid dispatch to B.
	unguarded := st.opMu.TryLock()
	if unguarded {
		st.opMu.Unlock()
	}
	applyDone := make(chan error, 1)
	go func() {
		_, err := f.ex.ConfigureLaunch(ctx, settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil })
		applyDone <- err
	}()
	if unguarded {
		if err := <-applyDone; err != nil {
			t.Fatal(err)
		}
	} else if err := servingWaitForPhase(ctx, "(*Executor).ConfigureLaunch("); err != nil {
		t.Fatal(err)
	}
	releaseOnce()
	if err := <-loadDone; err != nil {
		t.Fatal(err)
	}
	if !unguarded {
		if err := <-applyDone; err != nil {
			t.Fatal(err)
		}
	}
	if reachedB.Load() || unguarded {
		t.Fatal("load X authorized under A was dispatched to replacement B")
	}
	if _, err := f.ex.ModelLoad(ctx, "fake", "medium"); err == nil {
		t.Fatal("replacement B still authorized X")
	}
}

func TestLlamaServingLoadGuardReleasesBeforeRestart(t *testing.T) {
	ex, seen := modelOperationEngine(t)
	st, err := ex.state("variant")
	if err != nil {
		t.Fatal(err)
	}
	st.plat.Runtime.Serving = servingSettingsFixture()
	binding := st.manifest.Operations[engines.OperationLoad]
	binding.ResponseTimeoutS = 0 // use the client carrying the dispatch barrier
	st.manifest.Operations[engines.OperationLoad] = binding
	act := st.manifest.Actions["warm"]
	act.RestartAfter = true
	st.manifest.Actions["warm"] = act
	original := ex.client.Transport
	ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
		if st.opMu.TryLock() {
			st.opMu.Unlock()
			return nil, errors.New("restart-after dispatch did not hold operation lock")
		}
		res, err := original.RoundTrip(r)
		st.mu.Lock()
		st.running = false // the restart owner must acquire opMu and observe Off
		st.mu.Unlock()
		return res, err
	})
	done := make(chan error, 1)
	go func() {
		_, err := ex.Action(context.Background(), "variant", "warm", json.RawMessage(`{"target":"exact/model"}`))
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil || len(seen()) != 1 {
			t.Fatalf("restart-after load failed: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("NE load guard deadlocked restart-after handling")
	}
}

// Observe the actual blocked call, not merely a goroutine being launched. This
// test-only stack witness needs no production hook or scheduling delay. The
// timeout bounds a failed witness; it never establishes an interleaving.
func servingWaitForPhase(ctx context.Context, frame string) error {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	buf := make([]byte, 1<<20)
	for {
		n := runtime.Stack(buf, true)
		for _, stack := range strings.Split(string(buf[:n]), "\n\n") {
			if strings.Contains(stack, frame) && (strings.Contains(stack, "sync.Mutex.Lock") || strings.Contains(stack, "residentLock(")) {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("missing blocked phase %s: %w", frame, ctx.Err())
		default:
			runtime.Gosched()
		}
	}
}

func TestLlamaServingLoadAcrossNEActivation(t *testing.T) {
	test := func(name string, removeA, semantic bool) {
		t.Run(name, func(t *testing.T) {
			ex := settingsExecutor(t, false)
			st := settingsState(t, ex)
			st.manifest.Actions = map[string]Action{"warm": {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/ne-test-load"}, RestartAfter: true}}
			st.manifest.Operations = map[string]OperationBinding{engines.OperationLoad: {Action: "warm", ModelParam: "target"}}
			request := settingsRequest(t, ex)
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			var removalDone chan error
			removed := make(chan struct{})
			var releaseAdmission func()
			if removeA {
				request.Settings.Serving = servingSettingsFixture()
				if _, err := ex.ConfigureLaunch(ctx, settings.Configure{Engine: "fake", Settings: request.Settings}, func() error { return nil }); err != nil {
					t.Fatal(err)
				}
				// Queue the real removal first. Then hold admission so the load
				// passes A's pre-check but cannot proceed until removal installs nil.
				st.opMu.Lock()
				request.Settings.Serving = nil
				removalDone = make(chan error, 1)
				go func() {
					_, err := ex.ConfigureLaunch(ctx, settings.Configure{Engine: "fake", Settings: request.Settings}, func() error { close(removed); return nil })
					removalDone <- err
				}()
				if err := servingWaitForPhase(ctx, "(*Executor).ConfigureLaunch("); err != nil {
					st.opMu.Unlock()
					t.Fatal(err)
				}
				ex.switchAdmission.Lock()
				releaseAdmission = sync.OnceFunc(ex.switchAdmission.Unlock)
				t.Cleanup(releaseAdmission)
			}
			entered, release := make(chan struct{}), make(chan struct{})
			releaseDispatch := sync.OnceFunc(func() { close(release) })
			t.Cleanup(releaseDispatch)
			var wrongGeneration atomic.Bool
			ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
				close(entered)
				select {
				case <-release:
				case <-r.Context().Done():
					return nil, r.Context().Err()
				}
				st.mu.Lock()
				wrongGeneration.Store(st.plat.Runtime.Serving != nil)
				st.running = false // restart-after must leave the synthetic endpoint Off
				st.mu.Unlock()
				return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"unchanged":true}`))}, nil
			})
			st.mu.Lock()
			st.running = !removeA
			st.mu.Unlock()
			loadDone := make(chan error, 1)
			go func() {
				var err error
				if semantic {
					_, err = ex.ModelLoad(ctx, "fake", "exact/model")
				} else {
					var out bytes.Buffer
					manager := NewManager(NewCodec(&out), ex, nil, "")
					id := json.RawMessage(`1`)
					manager.runAction(ctx, &Message{ID: &id, Method: "engine:action", Params: json.RawMessage(`{"engine":"fake","action":"warm","params":{"target":"exact/model"}}`)})
					var reply Message
					err = json.Unmarshal(out.Bytes(), &reply)
					if err == nil && reply.Error != nil {
						err = errors.New(reply.Error.Message)
					}
				}
				loadDone <- err
			}()
			if removeA {
				if err := servingWaitForPhase(ctx, "(*Executor).admitMutation("); err != nil {
					st.opMu.Unlock()
					t.Fatal(err)
				}
				// Removal runs while the load remains at admission. Observe nil
				// installed via its operation-lock release before letting load proceed.
				st.opMu.Unlock()
				select {
				case <-removed:
				case <-ctx.Done():
					t.Fatal("removal did not reach its apply callback")
				}
				st.opMu.Lock()
				st.opMu.Unlock()
				st.mu.Lock()
				if st.plat.Runtime.Serving != nil {
					st.mu.Unlock()
					t.Fatal("real removal did not install nil")
				}
				st.running = true
				st.mu.Unlock()
				releaseAdmission()
				if err := <-removalDone; err != nil {
					t.Fatal(err)
				}
			}
			select {
			case <-entered:
			case <-ctx.Done():
				t.Fatal("load did not reach nil-record dispatch")
			}
			unguarded := st.opMu.TryLock()
			if unguarded {
				st.opMu.Unlock()
			}
			st.mu.Lock()
			st.running = false
			st.mu.Unlock()
			// B is applied by the real settings owner, not a direct state rewrite.
			request.Settings.Serving = servingSettingsFixture()
			request.Settings.Serving.Models[0].ModelID = "replacement/model"
			applyDone := make(chan error, 1)
			go func() {
				_, err := ex.ConfigureLaunch(ctx, settings.Configure{Engine: "fake", Settings: request.Settings}, func() error { return nil })
				applyDone <- err
			}()
			if unguarded {
				if err := <-applyDone; err != nil {
					t.Fatal(err)
				}
			} else if err := servingWaitForPhase(ctx, "(*Executor).ConfigureLaunch("); err != nil {
				t.Fatal(err)
			}
			releaseDispatch()
			if err := <-loadDone; err != nil {
				t.Fatal(err)
			}
			if !unguarded {
				if err := <-applyDone; err != nil {
					t.Fatal(err)
				}
			}
			if unguarded || wrongGeneration.Load() {
				t.Fatal("nil-record load dispatched across activation of NE B")
			}
		})
	}
	test("nil to B action", false, false)
	test("nil to B semantic", false, true)
	test("A to nil to B action", true, false)
	test("A to nil to B semantic", true, true)
}

func TestLlamaServingBlockedLoadCancelledByStop(t *testing.T) {
	for _, stopAll := range []bool{false, true} {
		name := "user Stop"
		if stopAll {
			name = "StopAll"
		}
		t.Run(name, func(t *testing.T) {
			f := newLlamaServingFixture(t)
			applyLlamaServing(t, f, approveLlamaServing(t, f))
			st := settingsState(t, f.ex)
			st.manifest.Actions["warm"] = Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/ne-test-load"}}
			st.manifest.Operations = map[string]OperationBinding{engines.OperationLoad: {Action: "warm", ModelParam: "target"}}
			if err := f.ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			f.ex.actionTimeout = time.Hour // stop must cancel, not wait out an action budget
			entered := make(chan struct{})
			original := f.ex.client.Transport
			f.ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
				if r.URL.Path != "/ne-test-load" {
					return original.RoundTrip(r)
				}
				close(entered)
				<-r.Context().Done()
				return nil, r.Context().Err()
			})
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			loaded := make(chan error, 1)
			go func() { _, err := f.ex.ModelLoad(ctx, "fake", "medium"); loaded <- err }()
			select {
			case <-entered:
			case <-time.After(3 * time.Second):
				t.Fatal("load did not enter blocked dispatch")
			}
			if st.opMu.TryLock() {
				st.opMu.Unlock()
				cancel()
				t.Fatal("blocked load does not hold operation lock")
			}
			stopped := make(chan error, 1)
			go func() {
				if stopAll {
					f.ex.StopAll()
					stopped <- nil
				} else {
					stopped <- f.ex.Stop("fake")
				}
			}()
			select {
			case err := <-stopped:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(5 * time.Second):
				cancel() // failure cleanup only; production stop must do the cancellation
				<-stopped
				t.Fatal("stop exceeded existing grace bound behind held load")
			}
			if err := <-loaded; !errors.Is(err, context.Canceled) {
				t.Fatalf("blocked load lost cancellation: %v", err)
			}
			st.mu.Lock()
			running, proc, pending := st.running, st.proc, len(st.loads.active)
			st.mu.Unlock()
			if running || proc != nil || pending != 0 {
				t.Fatal("stop retained engine or load cancellation ownership")
			}
		})
	}
}

func TestLlamaServingLoadCallerCancellation(t *testing.T) {
	ex, _ := modelOperationEngine(t)
	st, err := ex.state("variant")
	if err != nil {
		t.Fatal(err)
	}
	entered := make(chan struct{})
	// Force the ordinary client so the fixture's transport sees this load.
	binding := st.manifest.Operations[engines.OperationLoad]
	binding.ResponseTimeoutS = 0
	st.manifest.Operations[engines.OperationLoad] = binding
	ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
		close(entered)
		<-r.Context().Done()
		return nil, r.Context().Err()
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := ex.ModelLoad(ctx, "variant", "exact/model"); done <- err }()
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("load never entered dispatch")
	}
	if st.opMu.TryLock() {
		st.opMu.Unlock()
		t.Fatal("load dispatch did not retain operation lock")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("load lost caller cancellation: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("cancelled dispatch retained operation lock")
	}
	if !st.opMu.TryLock() {
		t.Fatal("cancelled dispatch retained operation lock")
	}
	st.opMu.Unlock()
}

func TestLlamaServingQueuedLoadCallerCancellation(t *testing.T) {
	ex, seen := modelOperationEngine(t)
	st, err := ex.state("variant")
	if err != nil {
		t.Fatal(err)
	}
	st.opMu.Lock()
	release := sync.OnceFunc(st.opMu.Unlock)
	t.Cleanup(release)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := ex.ModelLoad(ctx, "variant", "exact/model"); done <- err }()
	if err := servingWaitForPhase(ctx, "(*Executor).guardServingLoad("); err != nil {
		t.Fatal(err)
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) || len(seen()) != 0 {
			t.Fatalf("queued load did not cancel without dispatch: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("cancelled load waited for lifecycle operation to release opMu")
	}
	// The holder has deliberately not released yet: returning cannot be
	// explained by the lifecycle operation completing first.
	st.mu.Lock()
	active := len(st.loads.active)
	st.mu.Unlock()
	if active != 0 {
		t.Fatal("cancelled waiter retained registration")
	}
	release()
}

func TestLlamaServingStopFencesArrivingLoads(t *testing.T) {
	ex, seen := modelOperationEngine(t)
	st, err := ex.state("variant")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.running = false // no process is needed to witness stop's operation-lock wait
	st.mu.Unlock()
	st.opMu.Lock()
	release := sync.OnceFunc(st.opMu.Unlock)
	t.Cleanup(release)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	queued := make(chan error, 1)
	go func() { _, err := ex.ModelLoad(ctx, "variant", "exact/model"); queued <- err }()
	if err := servingWaitForPhase(ctx, "(*Executor).guardServingLoad("); err != nil {
		t.Fatal(err)
	}
	stopped := make(chan error, 1)
	go func() { stopped <- ex.Stop("variant") }()
	if err := servingWaitForPhase(ctx, "(*Executor).stopFrom("); err != nil {
		t.Fatal(err)
	}
	arriving := make(chan error, 1)
	go func() { _, err := ex.ModelLoad(ctx, "variant", "exact/model"); arriving <- err }()
	for _, result := range []<-chan error{queued, arriving} {
		select {
		case err := <-result:
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("stop did not cancel queued/arriving load: %v", err)
			}
		case <-ctx.Done():
			t.Fatal("stop allowed queued/arriving load to wait for lifecycle holder")
		}
	}
	if len(seen()) != 0 {
		t.Fatal("stop-fenced load reached dispatch")
	}
	release()
	if err := <-stopped; err != nil {
		t.Fatal(err)
	}
}

func TestLlamaServingConcurrentReplaceAndLoad(t *testing.T) {
	ex := settingsExecutor(t, false)
	st := settingsState(t, ex)
	st.manifest.Actions = map[string]Action{"warm": {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/ne-test-load"}}}
	st.manifest.Operations = map[string]OperationBinding{engines.OperationLoad: {Action: "warm", ModelParam: "target"}}
	request := settingsRequest(t, ex)
	request.Settings.Serving = servingSettingsFixture()
	if _, err := ex.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: request.Settings}, func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.running = true
	st.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	firstDispatch := make(chan struct{})
	var calls, wrongGeneration, overlapped, contended atomic.Int32
	var replacing atomic.Bool
	ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
		var params struct {
			Target string `json:"target"`
		}
		if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
			return nil, err
		}
		st.mu.Lock()
		before := st.plat.Runtime.Serving.Clone()
		st.mu.Unlock()
		if calls.Load() == 0 {
			close(firstDispatch)
			if err := servingWaitForPhase(ctx, "(*Executor).ConfigureLaunch("); err != nil {
				return nil, err
			}
			contended.Add(1)
		}
		if replacing.Load() {
			overlapped.Add(1)
		}
		st.mu.Lock()
		after := st.plat.Runtime.Serving.Clone()
		st.mu.Unlock()
		if !before.Equal(after) || len(after.Models) != 1 || after.Models[0].ModelID != params.Target {
			wrongGeneration.Add(1)
		}
		calls.Add(1)
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{}`))}, nil
	})
	start := make(chan struct{})
	results := make(chan error, 2)
	const iterations = 256
	go func() {
		<-start
		select {
		case <-firstDispatch:
		case <-ctx.Done():
			results <- ctx.Err()
			return
		}
		for i := range iterations {
			config := request.Settings
			config.Serving = servingSettingsFixture()
			if i%2 != 0 {
				config.Serving.Models[0].ModelID = "replacement/model"
			}
			// Model a cold control endpoint without starting a process. Configure
			// uses its real apply/persistence owner; the callback exposes the
			// synthetic endpoint again only after the new record is installed.
			st.mu.Lock()
			st.running = false
			st.mu.Unlock()
			replacing.Store(true)
			_, err := ex.ConfigureLaunch(ctx, settings.Configure{Engine: "fake", Settings: config}, func() error {
				st.mu.Lock()
				st.running = true
				st.mu.Unlock()
				return nil
			})
			replacing.Store(false)
			if err != nil {
				results <- err
				return
			}
			runtime.Gosched()
		}
		results <- nil
	}()
	go func() {
		<-start
		for i := range iterations * 4 {
			model := "exact/model"
			if i%2 != 0 {
				model = "replacement/model"
			}
			_, err := ex.ModelLoad(ctx, "fake", model)
			var field *ServingFieldError
			if err != nil && !errors.As(err, &field) && err.Error() != `engine "fake" is not running` {
				results <- err
				return
			}
			runtime.Gosched()
		}
		results <- nil
	}()
	close(start)
	for range 2 {
		select {
		case err := <-results:
			if err != nil {
				t.Fatal(err)
			}
		case <-ctx.Done():
			t.Fatal("concurrent replacement/load did not finish")
		}
	}
	st.mu.Lock()
	st.running = false
	st.mu.Unlock()
	if calls.Load() == 0 || wrongGeneration.Load() != 0 || overlapped.Load() == 0 || contended.Load() == 0 {
		t.Fatalf("dispatches=%d, wrong-generation=%d, overlapping=%d, contended=%d", calls.Load(), wrongGeneration.Load(), overlapped.Load(), contended.Load())
	}
}

func TestLlamaServingRestartLoadConcurrentSwitchRelease(t *testing.T) {
	ex, p := switchFixture(t)
	p = prepareSwitch(t, ex, p)
	st, err := ex.state("other")
	if err != nil {
		t.Fatal(err)
	}
	st.plat.Runtime.Serving = servingSettingsFixture()
	st.manifest.Actions = map[string]Action{"warm": {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/ne-test-load"}, RestartAfter: true}}
	st.manifest.Operations = map[string]OperationBinding{engines.OperationLoad: {Action: "warm", ModelParam: "target"}}
	st.running = true
	dispatchEntered, releaseStepDone := make(chan struct{}), make(chan struct{})
	finishRelease := sync.OnceFunc(func() { close(releaseStepDone) })
	t.Cleanup(finishRelease)
	ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
		close(dispatchEntered)
		if st.opMu.TryLock() {
			st.opMu.Unlock()
			return nil, errors.New("Release-concurrent dispatch did not hold operation lock")
		}
		select {
		case <-releaseStepDone:
		case <-r.Context().Done():
			return nil, r.Context().Err()
		}
		st.mu.Lock()
		st.running = false // restart-after must acquire opMu and observe Off
		st.mu.Unlock()
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{}`))}, nil
	})
	ex.switchAdmission.Lock()
	releaseAdmission := sync.OnceFunc(ex.switchAdmission.Unlock)
	t.Cleanup(releaseAdmission)
	loadDone := make(chan error, 1)
	go func() {
		_, err := ex.ModelLoad(context.Background(), "other", "exact/model")
		loadDone <- err
	}()
	// While admission is held, witness the load actually waiting there, then
	// let the gate owner read operation
	// state before the actual Release step. The old order creates a lock cycle
	// here: load owns opMu waiting for admission; gate owner waits for opMu.
	phaseCtx, phaseCancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer phaseCancel()
	if err := servingWaitForPhase(phaseCtx, "(*Executor).admitMutation("); err != nil {
		t.Fatal(err)
	}
	releaseDone := make(chan error, 1)
	go func() {
		_, err := ex.LaunchSettings("other")
		releaseAdmission()
		if err == nil {
			<-dispatchEntered // real Release runs while the load still holds opMu
			result := ex.SwitchStep(context.Background(), settings.SwitchStepRequest{SwitchPrepareRequest: p, Step: settings.SwitchStepRelease})
			if result.Error != nil {
				err = errors.New(result.Error.Message)
			}
		}
		finishRelease()
		releaseDone <- err
	}()
	timer := time.NewTimer(3 * time.Second)
	defer timer.Stop()
	timedOut := false
	for _, done := range []<-chan error{loadDone, releaseDone} {
		select {
		case err := <-done:
			if err != nil {
				t.Error(err)
			}
		case <-timer.C:
			timedOut = true
			t.Error("RestartAfter load and switch Release deadlocked on admission/operation lock order")
			// Break only the test-owned gate so a red proof exits cleanly rather
			// than stranding goroutines or hanging executor cleanup.
			releaseAdmission()
			finishRelease()
			if err := <-done; err != nil {
				t.Error(err)
			}
		}
	}
	if timedOut {
		return
	}
	ex.switchAdmission.Lock()
	mutations := ex.switchMutations["other"]
	ex.switchAdmission.Unlock()
	if mutations != 0 {
		t.Fatalf("completed load retained %d mutation guards", mutations)
	}
}

func TestLlamaServingReplacementWinsBeforeLoadAdmission(t *testing.T) {
	ex, seen := modelOperationEngine(t)
	st, err := ex.state("variant")
	if err != nil {
		t.Fatal(err)
	}
	st.plat.Runtime.Serving = servingSettingsFixture()
	act := st.manifest.Actions["warm"]
	act.RestartAfter = true
	st.manifest.Actions["warm"] = act
	// The load passes its unlocked check, then waits at admission while a
	// replacement removes its model. Rejection under opMu must undo admission.
	ex.switchAdmission.Lock()
	releaseAdmission := sync.OnceFunc(ex.switchAdmission.Unlock)
	t.Cleanup(releaseAdmission)
	done := make(chan error, 1)
	go func() {
		_, err := ex.ModelLoad(context.Background(), "variant", "exact/model")
		done <- err
	}()
	phaseCtx, phaseCancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer phaseCancel()
	if err := servingWaitForPhase(phaseCtx, "(*Executor).admitMutation("); err != nil {
		t.Fatal(err)
	}
	st.opMu.Lock()
	releaseOperation := sync.OnceFunc(st.opMu.Unlock)
	t.Cleanup(releaseOperation)
	releaseAdmission()
	if err := servingWaitForPhase(phaseCtx, "(*Executor).guardServingLoad("); err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.plat.Runtime.Serving = servingSettingsFixture()
	st.plat.Runtime.Serving.Models[0].ModelID = "replacement/model"
	st.mu.Unlock()
	releaseOperation()
	select {
	case err := <-done:
		var field *ServingFieldError
		if !errors.As(err, &field) || field.ModelID != "exact/model" || len(seen()) != 0 {
			t.Fatalf("replacement-won load reached dispatch or lost typed error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("replacement-won load did not release its guards")
	}
	ex.switchAdmission.Lock()
	mutations := ex.switchMutations["variant"]
	ex.switchAdmission.Unlock()
	if mutations != 0 {
		t.Fatalf("rejected load retained %d mutation guards", mutations)
	}
	if !st.opMu.TryLock() {
		t.Fatal("rejected load retained operation lock")
	}
	st.opMu.Unlock()
}
