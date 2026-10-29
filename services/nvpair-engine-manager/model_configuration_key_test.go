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
	"reflect"
	"strings"
	"testing"
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
	rt.ModelLoads = prepared.ModelLoads
	f.st.plat.Runtime = rt
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
	if inputs[f.request.ModelID].SWAFull != nil {
		t.Fatal("unproved source received pinned SWA default")
	}
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
