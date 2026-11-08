// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"nvpair-shared/modelconfig"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestModelConfigurationKeyUnapplied(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	result := inspectFixture(t, f)
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct{ ConfigurationKey, ConfigurationKeyReason string }
	if err := json.Unmarshal(data, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.ConfigurationKey != "" || wire.ConfigurationKeyReason != "not-applied" {
		t.Fatalf("unapplied key=%q reason=%q", wire.ConfigurationKey, wire.ConfigurationKeyReason)
	}
}

func measurementSemanticFixture(t *testing.T, f inspectionFixture) string {
	t.Helper()
	var semantic string
	result, err := f.ex.inspectModelArtifactsWithMeasurement(keyTestMemoryContext(), f.request, &semantic)
	if err != nil || result.Status != "resolved" || semantic == "" {
		t.Fatalf("semantic inspection: status=%s reason=%s semantic=%q error=%v", result.Status, result.Reason, semantic, err)
	}
	return semantic
}

func measurementTestEnvironment(t *testing.T) {
	t.Helper()
	// Positive controls describe a known native launch. Ambient developer
	// settings are separate inputs, exercised explicitly by the unknown case.
	for _, assignment := range os.Environ() {
		name, _, _ := strings.Cut(assignment, "=")
		if !strings.HasPrefix(name, "LLAMA_") && !strings.HasPrefix(name, "GGML_") && !strings.HasPrefix(name, "CUDA_") {
			continue
		}
		t.Setenv(name, "")
		if err := os.Unsetenv(name); err != nil {
			t.Fatal(err)
		}
	}
}

func TestMeasurementSemanticSourceCompleteness(t *testing.T) {
	measurementTestEnvironment(t)
	for _, tc := range []struct {
		name   string
		change func(*testing.T, inspectionFixture)
	}{
		{"requested cache", func(t *testing.T, f inspectionFixture) {
			f.st.plat.Runtime.ModelLoads[0].Requested.CacheTypeK = keyTestPointer("q8_0")
			compileDiagnosticFixtureLoads(t, f)
		}},
		{"extra launch option", func(t *testing.T, f inspectionFixture) {
			args := append([]string(nil), (*f.st.plat.Runtime.LaunchArgs)...)
			args = append(args, "--threads", "7")
			f.st.plat.Runtime.LaunchArgs = &args
		}},
		{"native environment", func(t *testing.T, f inspectionFixture) {
			env := []string{"GGML_CUDA_ENABLE_UNIFIED_MEMORY=1"}
			f.st.plat.Runtime.LaunchEnv = &env
		}},
		{"native runtime library", func(t *testing.T, f inspectionFixture) {
			writeTestFile(t, filepath.Join(f.root, "libggml-base.dylib"), []byte("replacement native library bytes"))
		}},
		{"native executable", func(t *testing.T, f inspectionFixture) {
			file, err := os.OpenFile(f.st.binPath, os.O_APPEND|os.O_WRONLY, 0)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := file.WriteString("\n# replacement native bytes\n"); err != nil {
				t.Fatal(err)
			}
			if err := file.Close(); err != nil {
				t.Fatal(err)
			}
		}},
		{"model content", func(t *testing.T, f inspectionFixture) {
			file, err := os.OpenFile(f.model, os.O_RDWR, 0)
			if err != nil {
				t.Fatal(err)
			}
			info, err := file.Stat()
			if err != nil {
				t.Fatal(err)
			}
			if _, err := file.WriteAt([]byte{1}, info.Size()-1); err != nil {
				t.Fatal(err)
			}
			if err := file.Close(); err != nil {
				t.Fatal(err)
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newAppliedKeyFixture(t)
			before := measurementSemanticFixture(t, f)
			tc.change(t, f)
			var after string
			_, err := f.ex.inspectModelArtifactsWithMeasurement(keyTestMemoryContext(), f.request, &after)
			if err != nil {
				t.Fatal(err)
			}
			if after == before {
				t.Fatal("speed-affecting source change reused semantic identity")
			}
		})
	}
}

func TestMeasurementSemanticExcludesIncidentalSources(t *testing.T) {
	measurementTestEnvironment(t)
	f := newAppliedKeyFixture(t)
	before := measurementSemanticFixture(t, f)
	f.st.runID = "new-engine-run"
	f.st.gen++
	f.ex.managerIncarnation = "new-manager"
	f.st.plat.Runtime.Port = 23456
	info, err := os.Stat(f.model)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(f.model, info.ModTime().Add(time.Second), info.ModTime().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	// An ordinary re-apply refreshes stat evidence without changing the sampled
	// content. Preserve the source owner's exact binding checks in this fixture.
	for i := range f.st.plat.Runtime.ModelLoads[0].Resolved.Artifacts {
		binding := &f.st.plat.Runtime.ModelLoads[0].Resolved.Artifacts[i]
		if binding.Path == f.model {
			binding.MtimeNS = fmt.Sprint(info.ModTime().Add(time.Second).UnixNano())
		}
	}
	// Recompile the same semantic configuration after the stat-only change.
	compileDiagnosticFixtureLoads(t, f)
	if after := measurementSemanticFixture(t, f); after != before {
		t.Fatalf("run/stat churn changed semantic identity: %s != %s", after, before)
	}
}

func TestMeasurementDraftArtifactParticipates(t *testing.T) {
	measurementTestEnvironment(t)
	f := newAppliedKeyFixture(t, func(f inspectionFixture) {
		writeInspectionGGUF(t, filepath.Join(f.root, "draft.gguf"), "qwen2", 0.99, 4096)
		preset, err := os.ReadFile(f.preset)
		if err != nil {
			t.Fatal(err)
		}
		writeTestFile(t, f.preset, append(preset, []byte("model-draft = draft.gguf\n")...))
	})
	before := measurementSemanticFixture(t, f)
	if artifacts := f.st.plat.Runtime.ModelLoads[0].Resolved.Artifacts; len(artifacts) != 2 || artifacts[1].Role != "draft" {
		t.Fatalf("draft positive control: %+v", artifacts)
	}
	file, err := os.OpenFile(filepath.Join(f.root, "draft.gguf"), os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteAt([]byte{7}, 4095); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	var after string
	_, err = f.ex.inspectModelArtifactsWithMeasurement(keyTestMemoryContext(), f.request, &after)
	if err != nil {
		t.Fatal(err)
	}
	if after == before {
		t.Fatal("changed consumed draft retained applicability")
	}
}

func TestMeasurementUnrepresentedEnvironmentFileIsUnknown(t *testing.T) {
	measurementTestEnvironment(t)
	f := newAppliedKeyFixture(t)
	measurementSemanticFixture(t, f)
	env := []string{"GGML_PRIVATE_CONFIG=/unobserved/native-config"}
	f.st.plat.Runtime.LaunchEnv = &env
	var semantic string
	_, err := f.ex.inspectModelArtifactsWithMeasurement(keyTestMemoryContext(), f.request, &semantic)
	if err != nil {
		t.Fatal(err)
	}
	if semantic != "" {
		t.Fatal("unobserved file-valued native environment claimed completeness")
	}
}

func TestMeasurementInheritedEnvironmentFileIsUnknown(t *testing.T) {
	measurementTestEnvironment(t)
	f := newAppliedKeyFixture(t)
	measurementSemanticFixture(t, f)
	t.Setenv("LLAMA_CPP_REPO", f.root)
	var semantic string
	if _, err := f.ex.inspectModelArtifactsWithMeasurement(keyTestMemoryContext(), f.request, &semantic); err != nil {
		t.Fatal(err)
	}
	if semantic != "" {
		t.Fatal("unobserved inherited file input claimed completeness")
	}
}

func TestMeasurementWorkingSetLimitParticipates(t *testing.T) {
	measurementTestEnvironment(t)
	f := newAppliedKeyFixture(t)
	f.st.plat.Runtime.WorkingSetLimitMiB = keyTestPointer(8192)
	before := measurementSemanticFixture(t, f)
	f.st.plat.Runtime.WorkingSetLimitMiB = keyTestPointer(4096)
	after := measurementSemanticFixture(t, f)
	if before == after {
		t.Fatal("changed process working-set cap reused measurement applicability")
	}
}

func TestModelConfigurationKeyExtraOptionParticipation(t *testing.T) {
	f := newAppliedKeyFixture(t)
	before := inspectAppliedKeyFixture(t, f)
	if before.ConfigurationKey == "" {
		t.Fatal("finite control key unavailable")
	}
	rt := f.st.plat.Runtime
	rt.ModelLoads[0].Requested.ExtraOptions = map[string]string{"expert-cache": "auto"}
	command, err := launchForState(f.st, rt.Port)
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	rt.ModelLoads = prepared.ModelLoads
	f.st.plat.Runtime = rt
	after := inspectAppliedKeyFixture(t, f)
	if after.ConfigurationKey == "" || after.ConfigurationKey == before.ConfigurationKey {
		t.Fatal("extra option omitted from configuration identity")
	}
	if after.Estimate.Status != "unknown" {
		t.Fatalf("fork key priced: %+v", after.Estimate)
	}
}

func TestModelConfigurationKeyNativeFitExecutionScope(t *testing.T) {
	on := "on"
	in := modelEstimateInput{
		Allocation: &modelconfig.ResolvedLoad{NativeBuildID: "sha256:" + strings.Repeat("a", 64), Options: modelconfig.RequestedLoad{Fit: &on}},
		Bindings:   []modelconfig.ArtifactSummary{{Role: "model", Identity: "artifact-a"}},
	}
	key := modelExecutionKey(in, "source-a", "options-a")
	if len(key) != 64 || modelConfigurationKey(in, "source", "options-a") != "" {
		t.Fatalf("native fit identity: execution=%s finite=%s", key, modelConfigurationKey(in, "source", "options-a"))
	}
	if modelExecutionKey(in, "source-a", "options-b") == key {
		t.Fatal("execution key ignored resolved options")
	}
	if modelExecutionKey(in, "source-b", "options-a") == key {
		t.Fatal("execution key ignored native launch source")
	}
	if modelExecutionKey(in, "", "options-a") != "" {
		t.Fatal("unverified native source received execution identity")
	}
	in.Allocation.NativeBuildID = "sha256:" + strings.Repeat("b", 64)
	if modelExecutionKey(in, "source-a", "options-a") == key {
		t.Fatal("execution key ignored native build")
	}
	in.Allocation.NativeBuildID = "sha256:" + strings.Repeat("a", 64)
	in.Bindings[0].Identity = "artifact-b"
	if modelExecutionKey(in, "source-a", "options-a") == key {
		t.Fatal("execution key ignored artifact identity")
	}
	if modelExecutionKey(modelEstimateInput{}, "source-a", "options-a") != "" {
		t.Fatal("unapplied load received execution identity")
	}
	in.Allocation.Options.Fit = nil
	if modelExecutionKey(in, "source-a", "options-a") != "" {
		t.Fatal("ordinary finite load received execution identity")
	}
}

func TestModelConfigurationKeyNativeFitUnapplied(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	on := "on"
	f.st.plat.Runtime.ModelLoads = []modelconfig.ModelLoad{{ModelID: f.request.ModelID, Requested: modelconfig.RequestedLoad{Fit: &on}}}
	result := inspectFixture(t, f)
	if result.ExecutionKey != "" || result.ConfigurationKey != "" || result.ConfigurationKeyReason != "not-applied" {
		t.Fatalf("unapplied fit identity=%+v", result)
	}
}

func TestModelConfigurationKeyStableInputs(t *testing.T) {
	base := estimateFixture("llama")
	base.Memory = &modelEstimateMemory{Target: modelBufferTestTargetMemory()}
	base.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: keyTestPointer(int32(1))}
	base.Allocation.DeviceBinding = autoTestDeviceBinding()
	key := func(in modelEstimateInput) string {
		return modelConfigurationKey(in, "native-source", modelConfigurationOptionsID(modelConfigurationOptions(*in.Allocation), []byte("preset")))
	}
	want := key(base)
	if want == "" {
		t.Fatal("finite bound configuration has no key")
	}
	t.Run("native source", func(t *testing.T) {
		optionsID := modelConfigurationOptionsID(modelConfigurationOptions(*base.Allocation), []byte("preset"))
		if modelConfigurationKey(base, "different-native-source", optionsID) == want {
			t.Fatal("native source change reused key")
		}
	})
	t.Run("artifact order", func(t *testing.T) {
		in := base
		in.Bindings = append([]modelconfig.ArtifactSummary(nil), base.Bindings...)
		second := in.Bindings[0]
		second.Identity = "second-artifact"
		in.Bindings = append(in.Bindings, second)
		ordered := key(in)
		in.Bindings[0], in.Bindings[1] = in.Bindings[1], in.Bindings[0]
		if key(in) == ordered {
			t.Fatal("artifact binding order was discarded")
		}
	})
	t.Run("telemetry producer and estimator churn", func(t *testing.T) {
		in := base
		in.Memory = &modelEstimateMemory{Identity: "new-telemetry", Source: "new-producer", Target: modelBufferTestTargetMemory()}
		in.Memory.Target.Observation.RunID = "new-producer"
		in.Memory.Target.ObservationID, in.Memory.Target.MapID = "new-observation", "new-map"
		in.BuildID, in.Support = "estimator-only-build-label", "new-estimator-revision"
		in.Bindings = append([]modelconfig.ArtifactSummary(nil), base.Bindings...)
		in.Bindings[0].Assurance = "different diagnostic"
		if got := key(in); got != want {
			t.Fatalf("key changed: %s != %s", got, want)
		}
	})
	for _, tc := range []struct {
		name   string
		change func(*modelEstimateInput)
	}{
		{"context", func(in *modelEstimateInput) { in.Allocation.PerSlotContext++ }},
		{"cache", func(in *modelEstimateInput) { in.Allocation.Options.CacheTypeK = keyTestPointer("q8_0") }},
		{"layers", func(in *modelEstimateInput) { in.Allocation.GPU.Layers = keyTestPointer(int32(2)) }},
		{"device", func(in *modelEstimateInput) {
			b := *in.Allocation.DeviceBinding
			b.DeviceID = "other-device"
			in.Allocation.DeviceBinding = &b
		}},
		{"artifact", func(in *modelEstimateInput) { in.Bindings[0].Identity = "other-artifact" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := base
			a := base.Allocation.Clone()
			in.Allocation = &a
			in.Bindings = append([]modelconfig.ArtifactSummary(nil), base.Bindings...)
			tc.change(&in)
			if key(in) == want {
				t.Fatal("changed configuration reused key")
			}
		})
	}
	t.Run("manual unbound GPU", func(t *testing.T) {
		in := base
		a := base.Allocation.Clone()
		in.Allocation = &a
		in.Allocation.DeviceBinding = nil
		if key(in) == "" || key(in) == want {
			t.Fatal("unbound advisory scope missing or conflated with bound device")
		}
	})
}

func keyTestPointer[T any](value T) *T { return &value }

func newFetchBuildKeyFixture(t *testing.T) inspectionFixture {
	t.Helper()
	return newAppliedKeyFixture(t, func(f inspectionFixture) {
		nativeBuildTestRuntime(t, f.st, t.TempDir())
		f.st.plat.Install = &Install{Fetch: &Fetch{URL: "https://fixture.invalid/releases/download/custom/native.tar.gz", SHA256: strings.Repeat("a", 64)}}
		if f.st.plat.Runtime.Env == nil {
			f.st.plat.Runtime.Env = map[string]string{}
		}
		f.st.plat.Runtime.Env["LD_LIBRARY_PATH"] = f.st.installDir
	})
}

func TestModelConfigurationKeyNativeBuildContent(t *testing.T) {
	test := func(name string, change func(*testing.T, inspectionFixture), changes bool) {
		t.Run(name, func(t *testing.T) {
			f := newFetchBuildKeyFixture(t)
			before := inspectAppliedKeyFixture(t, f)
			if before.ConfigurationKey == "" {
				t.Fatalf("fetch-installed finite load has no key: %+v", before)
			}
			change(t, f)
			after := inspectAppliedKeyFixture(t, f)
			if after.ConfigurationKey == "" {
				t.Fatalf("finite load lost key: %+v", after)
			}
			if (after.ConfigurationKey != before.ConfigurationKey) != changes {
				t.Fatalf("key changed=%v, want %v", after.ConfigurationKey != before.ConfigurationKey, changes)
			}
		})
	}
	test("executable touch", func(t *testing.T, f inspectionFixture) {
		info, err := os.Stat(f.st.binPath)
		if err != nil {
			t.Fatal(err)
		}
		touch := info.ModTime().Add(2 * time.Hour)
		if err := os.Chtimes(f.st.binPath, touch, touch); err != nil {
			t.Fatal(err)
		}
	}, false)
	test("moved build and loader path", func(t *testing.T, f inspectionFixture) {
		root := filepath.Join(t.TempDir(), "moved-build")
		if err := os.Rename(f.st.installDir, root); err != nil {
			t.Fatal(err)
		}
		f.st.installDir = root
		f.st.binPath = filepath.Join(root, filepath.Base(f.st.binPath))
		f.st.plat.Runtime.Bin = f.st.binPath
		f.st.plat.Runtime.Env["LD_LIBRARY_PATH"] = root
		if f.st.plat.Runtime.LaunchEnv != nil {
			environment, err := literalEnvironment(*f.st.plat.Runtime.LaunchEnv)
			if err != nil {
				t.Fatal(err)
			}
			environment["LD_LIBRARY_PATH"] = root
			assignments := literalAssignments(environment)
			f.st.plat.Runtime.LaunchEnv = &assignments
		}
	}, false)
	test("library bytes", func(t *testing.T, f inspectionFixture) {
		writeTestFile(t, filepath.Join(f.st.installDir, "libggml-base.dylib"), []byte("different native library bytes"))
	}, true)
	test("unrelated ambient environment", func(t *testing.T, _ inspectionFixture) {
		t.Setenv("PAIR_NATIVE_BUILD_UNRELATED", "changed")
	}, false)
	test("native ambient environment", func(t *testing.T, _ inspectionFixture) {
		t.Setenv("GGML_SOURCE_FIXTURE", "changed")
	}, true)
	test("explicit launch environment", func(t *testing.T, f inspectionFixture) {
		f.st.plat.Runtime.Env["PAIR_EXPLICIT_BUILD_FIXTURE"] = "changed"
	}, true)
}

func TestModelConfigurationKeyCPUExperts(t *testing.T) {
	in := estimateFixture("qwen3moe")
	in.Memory = &modelEstimateMemory{Target: modelBufferTestTargetMemory()}
	seen := map[string]string{}
	for _, tc := range []struct {
		name    string
		experts *modelconfig.CPUExperts
	}{
		{"inherited", nil},
		{"none", &modelconfig.CPUExperts{Mode: "none"}},
		{"all", &modelconfig.CPUExperts{Mode: "all"}},
		{"zero layers", &modelconfig.CPUExperts{Mode: "layers", Layers: keyTestPointer(int32(0))}},
		{"one layer", &modelconfig.CPUExperts{Mode: "layers", Layers: keyTestPointer(int32(1))}},
		{"two layers", &modelconfig.CPUExperts{Mode: "layers", Layers: keyTestPointer(int32(2))}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in.Allocation.Options.CPUExperts = tc.experts
			options := modelConfigurationOptions(*in.Allocation)
			if !reflect.DeepEqual(options.CPUExperts, tc.experts) {
				t.Fatal("effective options dropped CPU expert intent")
			}
			id := modelConfigurationOptionsID(options, []byte("same-native-snapshot"))
			key := modelConfigurationKey(in, "same-native-source", id)
			if key == "" {
				t.Fatal("finite configuration has no key")
			}
			if previous, duplicate := seen[key]; duplicate {
				t.Fatalf("%s reused %s configuration key", tc.name, previous)
			}
			seen[key] = tc.name
			if options.CPUExperts != nil && options.CPUExperts.Layers != nil {
				*options.CPUExperts.Layers = 99
				if *tc.experts.Layers == 99 {
					t.Fatal("configuration identity clone retained caller's layers pointer")
				}
			}
		})
	}
}

func writeKeyGGUF(t *testing.T, f inspectionFixture, arch string) {
	t.Helper()
	data := []byte("GGUF")
	u32 := func(v uint32) { data = binary.LittleEndian.AppendUint32(data, v) }
	u64 := func(v uint64) { data = binary.LittleEndian.AppendUint64(data, v) }
	str := func(v string) { u64(uint64(len(v))); data = append(data, v...) }
	numeric := map[string]uint32{"context_length": 4096, "block_count": 4, "embedding_length": 128, "feed_forward_length": 256, "attention.head_count": 4, "attention.head_count_kv": 2}
	if arch == "gemma2" {
		numeric["attention.sliding_window"] = 256
		numeric["attention.sliding_window_pattern"] = 2
	}
	u32(3)
	u64(4)
	u64(uint64(len(numeric) + 1))
	str("general.architecture")
	u32(8)
	str(arch)
	for key, value := range numeric {
		str(arch + "." + key)
		u32(4)
		u32(value)
	}
	for i := 0; i < 4; i++ {
		str(fmt.Sprintf("blk.%d.attn_q.weight", i))
		u32(1)
		u64(32)
		u32(0)
		u64(uint64(i * 128))
	}
	for len(data)%32 != 0 {
		data = append(data, 0)
	}
	data = append(data, make([]byte, 512)...)
	writeTestFile(t, f.model, data)
}

func TestModelConfigurationKeyPublicPricedInputs(t *testing.T) {
	for _, arch := range []string{"llama", "gemma2"} {
		t.Run(arch, func(t *testing.T) {
			f := newAppliedKeyFixture(t, func(f inspectionFixture) {
				writeKeyGGUF(t, f, arch)
				data, err := os.ReadFile(f.preset)
				if err != nil {
					t.Fatal(err)
				}
				writeTestFile(t, f.preset, []byte(strings.Replace(string(data), "gpu-layers = 0", "gpu-layers = 2", 1)))
			})
			result := inspectAppliedKeyFixture(t, f)
			if result.ConfigurationKey == "" || !reflect.DeepEqual(result.ConfigurationPools, []string{"host", "device"}) {
				t.Fatalf("configuration=%+v", result)
			}
			host, device, cache := false, false, false
			for _, row := range result.Estimate.Components {
				if row.Bytes != "" {
					host = host || row.DomainID == "host"
					device = device || row.DomainID == "offload-unmapped"
					cache = cache || row.Name == "attention-cache"
				}
			}
			if !host || !device || !cache {
				t.Fatalf("public compiled pricing missing: %+v", result.Estimate)
			}
		})
	}
}

func TestModelConfigurationKeyAppliedPublic(t *testing.T) {
	f := newAppliedKeyFixture(t)
	result := inspectAppliedKeyFixture(t, f)
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		ConfigurationKey, ConfigurationKeyReason string
		ConfigurationPools                       []string
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.ConfigurationKey) != 64 || wire.ConfigurationKeyReason != "" || len(wire.ConfigurationPools) != 1 || wire.ConfigurationPools[0] != "host" {
		t.Fatalf("applied configuration=%+v status=%s reason=%s", wire, result.Status, result.Reason)
	}
}

func newAppliedKeyFixture(t *testing.T, configure ...func(inspectionFixture)) inspectionFixture {
	t.Helper()
	f := newInspectionFixture(t, 4096)
	loadFixture := modelLoadFixture{ex: f.ex, st: f.st, root: f.root}
	autoTestNativeSource(t, &loadFixture)
	f.st.plat.Runtime.EditableLaunch = &EditableLaunch{}
	writeTestFile(t, f.preset, []byte("[exact-alias]\nmodel = main.gguf\nctx-size = 4096\nparallel = 1\ngpu-layers = 0\ncache-type-k = f16\ncache-type-v = f16\nflash-attn = off\nbatch-size = 64\nubatch-size = 32\nkv-offload = true\nop-offload = false\n"))
	for _, edit := range configure {
		edit(f)
	}
	rt := f.st.plat.Runtime
	command, err := launchForState(f.st, rt.Port)
	if err != nil {
		t.Fatal(err)
	}
	conversion, err := f.ex.previewTypedLlamaConversion(context.Background(), f.st, command, rt)
	if err != nil {
		t.Fatal(err)
	}
	rt = conversion.Runtime
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, loadFixture, rt), rt, false)
	if err != nil {
		t.Fatal(err)
	}
	if prepared == nil || len(prepared.ModelLoads) == 0 {
		t.Fatalf("applied key fixture has no compiled model loads: %+v", prepared)
	}
	rt.ModelLoads = prepared.ModelLoads
	f.st.plat.Runtime = rt
	f.request.ProfileKey = "exact-alias"
	return f
}

func keyTestMemoryContext() context.Context {
	return context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: modelTargetMemoryOwner{target: modelBufferTestTargetMemory()}})
}

func inspectAppliedKeyFixture(t *testing.T, f inspectionFixture) modelconfig.InspectResult {
	t.Helper()
	result, err := f.ex.InspectModel(keyTestMemoryContext(), f.request)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestModelConfigurationKeyDomainlessAndCapture(t *testing.T) {
	f := newAppliedKeyFixture(t)
	f.ex.modelEstimates.run = func(context.Context, modelEstimateInput) modelconfig.Estimate {
		return modelconfig.Estimate{Status: "unknown"}
	}
	ctx := keyTestMemoryContext()
	result, err := f.ex.InspectModel(ctx, f.request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Estimate.Status != "unknown" || len(result.Estimate.Domains) != 0 || result.ConfigurationKey == "" || !reflect.DeepEqual(result.ConfigurationPools, []string{"host"}) {
		t.Fatalf("domainless inspection=%+v", result)
	}
	command, err := launchForState(f.st, f.st.plat.Runtime.Port)
	if err != nil {
		t.Fatal(err)
	}
	var inputs map[string]modelEstimateInput
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, command, f.st.plat.Runtime, false, &inputs)
	if err != nil {
		t.Fatal(err)
	}
	capture := f.ex.newModelBufferCapture(f.st, "llamacpp", prepared, inputs, command, ctx)
	if capture == nil {
		t.Fatal("capture unavailable")
	}
	defer capture.retire()
	frozen := capture.frozen[f.request.ModelID]
	if frozen == nil || frozen.configurationKey != result.ConfigurationKey {
		t.Fatalf("capture key mismatch: %+v", frozen)
	}
	wantOptions := modelBufferHash(struct {
		Options modelconfig.RequestedLoad
		Preset  string
	}{frozen.options, modelLoadDigest(prepared.Snapshot)})
	expectedInput := inputs[f.request.ModelID]
	// Compare against the exact operation snapshot retained at capture, not a
	// second clock read that would change the diagnostic advisory identity.
	expectedInput.Memory = frozen.input.Memory
	wantInput := modelBufferHash(struct{ Advisory, Source, Options, Revision, Topology string }{modelBufferHash(expectedInput), capture.sourceID, wantOptions, modelEstimatorRevision, frozen.topologyID})
	expectedTopology := &modelBufferFrozen{input: inputs[f.request.ModelID]}
	modelBufferFreezeTopology(expectedTopology, frozen.input.Memory, "")
	if frozen.optionsID != wantOptions || frozen.inputID != wantInput || frozen.topologyID != expectedTopology.topologyID {
		t.Fatal("preexisting capture identities changed")
	}
}

func TestModelConfigurationKeyPools(t *testing.T) {
	in := estimateFixture("llama")
	if modelConfigurationPools(in) != nil {
		t.Fatal("missing pool evidence became host topology")
	}
	in.Memory = &modelEstimateMemory{Target: modelBufferTestTargetMemory()}
	if !reflect.DeepEqual(modelConfigurationPools(in), []string{"host"}) {
		t.Fatal("CPU pools")
	}
	in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: keyTestPointer(int32(0))}
	if !reflect.DeepEqual(modelConfigurationPools(in), []string{"host"}) {
		t.Fatal("zero-layer CPU pools")
	}
	in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: keyTestPointer(int32(1))}
	if !reflect.DeepEqual(modelConfigurationPools(in), []string{"host", "device"}) {
		t.Fatal("discrete pools")
	}
	in.Memory.Target.Observation.Domains = in.Memory.Target.Observation.Domains[:1]
	in.Memory.Target.Observation.Domains[0].Kind = "shared"
	if !reflect.DeepEqual(modelConfigurationPools(in), []string{"shared"}) {
		t.Fatal("UMA pools")
	}
}

func TestModelConfigurationKeyUnavailableCompiler(t *testing.T) {
	f := newAppliedKeyFixture(t)
	if err := os.Remove(f.st.plat.Runtime.Bin); err != nil {
		t.Fatal(err)
	}
	result := inspectFixture(t, f)
	if result.ConfigurationKey != "" || result.ConfigurationKeyReason != "not-finite" {
		t.Fatalf("unavailable compiler configuration=%+v", result)
	}
	for _, row := range result.Estimate.Components {
		if row.Name == "attention-cache" && row.Bytes != "" {
			t.Fatal("uncompiled allocation was priced")
		}
	}
}

func TestModelConfigurationKeySWARequiresPinnedSource(t *testing.T) {
	f := newAppliedKeyFixture(t)
	f.st.plat.Install = nil
	command, err := launchForState(f.st, f.st.plat.Runtime.Port)
	if err != nil {
		t.Fatal(err)
	}
	var inputs map[string]modelEstimateInput
	_, err = compileTypedLlamaInputs(context.Background(), f.st, command, f.st.plat.Runtime, false, &inputs)
	if err != nil {
		t.Fatal(err)
	}
	input, ok := inputs[f.request.ModelID]
	if !ok || input.Allocation == nil {
		t.Fatal("compiled SWA advisory input is missing")
	}
	if input.SWAFull != nil {
		t.Fatal("unproved source received pinned SWA default")
	}
}

func TestModelConfigurationKeySWADefaultHasIndependentSourceAssurance(t *testing.T) {
	test := func(name string, install *Install, known bool) {
		t.Run(name, func(t *testing.T) {
			f := newAppliedKeyFixture(t)
			f.st.plat.Install = install
			command, err := launchForState(f.st, f.st.plat.Runtime.Port)
			if err != nil {
				t.Fatal(err)
			}
			var inputs map[string]modelEstimateInput
			prepared, err := compileTypedLlamaInputs(context.Background(), f.st, command, f.st.plat.Runtime, false, &inputs)
			if err != nil || prepared == nil || len(prepared.ModelLoads) == 0 {
				t.Fatalf("compile=%+v error=%v", prepared, err)
			}
			input, ok := inputs[f.request.ModelID]
			if !ok || input.Allocation == nil {
				t.Fatal("compiled advisory input missing")
			}
			if known {
				if input.SWAFull == nil || *input.SWAFull {
					t.Fatal("known source did not supply the inherited false SWA default")
				}
			} else if input.SWAFull != nil {
				t.Fatal("content identity broadened inherited SWA default assurance")
			}
		})
	}
	test("known managed artifact declaration", &Install{Artifacts: []InstallArtifact{{Name: "server", URL: "https://fixture.invalid/releases/download/b12020/native", SHA256: strings.Repeat("a", 64)}}}, true)
	test("different build artifact declaration", &Install{Artifacts: []InstallArtifact{{Name: "server", URL: "https://fixture.invalid/releases/download/custom/native", SHA256: strings.Repeat("a", 64)}}}, false)
	test("fetch declaration does not prove the inherited default", &Install{Fetch: &Fetch{URL: "https://fixture.invalid/releases/download/b12020/native", SHA256: strings.Repeat("a", 64)}}, false)
}

func TestModelConfigurationKeyMissingTopologyDoesNotChangePool(t *testing.T) {
	f := newAppliedKeyFixture(t)
	target := modelBufferTestTargetMemory()
	target.Observation.Domains = target.Observation.Domains[:1]
	target.Observation.Domains[0].Kind = "shared"
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: modelTargetMemoryOwner{target: target}})
	known, err := f.ex.InspectModel(ctx, f.request)
	if err != nil {
		t.Fatal(err)
	}
	if known.ConfigurationKey == "" || !reflect.DeepEqual(known.ConfigurationPools, []string{"shared"}) {
		t.Fatalf("UMA control key=%q pools=%v", known.ConfigurationKey, known.ConfigurationPools)
	}
	command, err := launchForState(f.st, f.st.plat.Runtime.Port)
	if err != nil {
		t.Fatal(err)
	}
	var inputs map[string]modelEstimateInput
	prepared, err := compileTypedLlamaInputs(ctx, f.st, command, f.st.plat.Runtime, false, &inputs)
	if err != nil {
		t.Fatal(err)
	}
	control := f.ex.newModelBufferCapture(f.st, "llamacpp", prepared, inputs, command, ctx)
	if control == nil {
		t.Fatal("UMA control capture unavailable")
	}
	defer control.retire()
	if control.frozen[f.request.ModelID].configurationKey != known.ConfigurationKey {
		t.Fatal("UMA capture and inspection scopes differ")
	}
	for _, tc := range []struct {
		name string
		ctx  context.Context
	}{
		{"missing", context.Background()},
		{"expired", func() context.Context {
			stale := target.Clone()
			stale.Observation.Domains[0].Total.AgeMS = 6000
			stale.Observation.Domains[0].Free.AgeMS = 6000
			return context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: modelTargetMemoryOwner{target: stale}})
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := f.ex.InspectModel(tc.ctx, f.request)
			if err != nil {
				t.Fatal(err)
			}
			if got.ConfigurationKey != "" || len(got.ConfigurationPools) != 0 || got.ConfigurationKeyReason != "not-finite" {
				t.Fatalf("unproved pool key=%q pools=%v reason=%q", got.ConfigurationKey, got.ConfigurationPools, got.ConfigurationKeyReason)
			}
			command, err := launchForState(f.st, f.st.plat.Runtime.Port)
			if err != nil {
				t.Fatal(err)
			}
			var inputs map[string]modelEstimateInput
			prepared, err := compileTypedLlamaInputs(tc.ctx, f.st, command, f.st.plat.Runtime, false, &inputs)
			if err != nil {
				t.Fatal(err)
			}
			capture := f.ex.newModelBufferCapture(f.st, "llamacpp", prepared, inputs, command, tc.ctx)
			if capture == nil {
				t.Fatal("capture unavailable")
			}
			defer capture.retire()
			if capture.frozen[f.request.ModelID].configurationKey != "" {
				t.Fatal("capture invented a pool scope without current topology")
			}
		})
	}
}
