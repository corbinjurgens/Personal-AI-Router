// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

func TestModelSettingsContractTargetMemoryFrozenCounterTimeline(t *testing.T) {
	f, _, _ := autoTestFrozenLaunch(t)
	candidatePrices := 0
	f.ex.modelEstimates = modelEstimateState{run: func(ctx context.Context, in modelEstimateInput) modelconfig.Estimate {
		if in.Allocation != nil && in.Allocation.ResolverRevision == modelAutoRevision && in.Allocation.ReviewedInputID == "" {
			candidatePrices++
		}
		return autoTestEstimate(ctx, in)
	}}
	r := autoTestRequest()
	r.Context = &modelconfig.Context{Mode: "auto", Floor: 4096, Cap: 4096}
	request := settingsRequest(t, f.ex)
	request.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: r}}
	uuid := autoTestDeviceBinding().DeviceID
	helpers := 0
	runner := func(ctx context.Context, _ modelDeviceRuntime, env []string) (modelDeviceReply, error) {
		helpers++
		if !slices.Contains(env, "CUDA_VISIBLE_DEVICES="+uuid) {
			t.Fatal("review helper did not use reviewed full UUID")
		}
		return modelDeviceReply{Devices: []modelNativeDevice{{Selector: "CPU", Kind: 0}, {Selector: "CUDA0", Kind: 1, UUID: uuid, PCI: "0000:ab:00.0", LUID: modelDevicesFixtureLUID}}}, ctx.Err()
	}
	memory := func(free string) *modelconfig.TargetMemory {
		fact := noderec.MemoryFact{Valid: true, Bytes: "8589934592", Source: "synthetic-target", SampledAt: time.Now().UTC()}
		available := fact
		available.Bytes = free
		return &modelconfig.TargetMemory{HostID: "synthetic-host", ObservationID: "sample-" + free, Observation: &noderec.MemoryObservation{RunID: "producer", Domains: []noderec.MemoryDomain{{ID: "host", Kind: "system", Total: fact, Free: available}, {ID: "gpu", Kind: "dedicated", DeviceIDs: []string{uuid, "pci:0000:ab:00.0", modelDevicesFixtureLUID}, Total: fact, Free: available}}}, Hop: noderec.NewMemoryHop("broker", noderec.MemoryBootClock)}
	}
	server, client := net.Pipe()
	t.Cleanup(func() { _ = server.Close(); _ = client.Close() })
	manager := &Manager{exec: f.ex, codec: NewCodec(server)}
	call := func(envelope settings.WorkerRequest[settings.Request]) Message {
		t.Helper()
		raw, err := json.Marshal(envelope)
		if err != nil {
			t.Fatal(err)
		}
		if err := client.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
			t.Fatal(err)
		}
		ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{DeviceRunner: runner})
		id := json.RawMessage(`1`)
		manager.handleSettingsMessage(ctx, &Message{JSONRPC: "2.0", ID: &id, Method: "engine:preview-launch", Params: raw})
		var reply Message
		if err := json.NewDecoder(client).Decode(&reply); err != nil {
			t.Fatal(err)
		}
		return reply
	}
	reply := call(settings.WorkerRequest[settings.Request]{Request: request, Memory: memory("8589934592"), NativeInitAllowed: true})
	if reply.Error != nil {
		t.Fatalf("target preview refused: %+v", reply.Error)
	}
	var preview settings.Preview
	if err := modelconfig.DecodeStrict(reply.Result, &preview); err != nil {
		t.Fatal(err)
	}
	if len(preview.Errors) != 0 || preview.Auto == nil || preview.Auto.Resolved == nil {
		t.Fatalf("target preview=%+v", preview)
	}
	before := helpers
	beforePrices := candidatePrices
	if before == 0 {
		t.Fatal("explicit review omitted the private metadata helper")
	}
	if beforePrices == 0 {
		t.Fatal("explicit review did not price Auto candidates")
	}
	unsupported := memory("8589934592")
	unsupported.Observation.Domains[1].DeviceIDs = append(unsupported.Observation.Domains[1].DeviceIDs, "MIG-unsupported-alias")
	fixedGPU := request.Clone()
	layers := int32(1)
	fixedGPU.ModelChange.Load.Set.GPU = &modelconfig.GPU{Mode: "layers", Layers: &layers}
	refused := call(settings.WorkerRequest[settings.Request]{Request: fixedGPU, Memory: unsupported, NativeInitAllowed: true})
	if refused.Error == nil {
		var advice settings.Preview
		if err := modelconfig.DecodeStrict(refused.Result, &advice); err != nil {
			t.Fatal(err)
		}
		if len(advice.Errors) == 0 || advice.Auto != nil && advice.Auto.Resolved != nil {
			t.Fatal("unsupported physical alias authorized a GPU allocation")
		}
	}
	if helpers != before {
		t.Fatal("unsupported physical topology initialized native metadata")
	}
	finite := request.Clone()
	finite.ModelChange = nil
	finite.Settings = preview.Settings.Clone()
	for _, free := range []string{"7516192768", "1"} {
		reply = call(settings.WorkerRequest[settings.Request]{Request: finite, Memory: memory(free), ValidateResources: true, ResourceModel: "weak-model"})
		if (reply.Error != nil) != (free == "1") {
			t.Fatalf("fresh B free=%s reply=%+v", free, reply)
		}
		if free != "1" {
			var validated settings.Preview
			if err := modelconfig.DecodeStrict(reply.Result, &validated); err != nil {
				t.Fatal(err)
			}
			if !validated.Settings.Equal(preview.Settings) {
				t.Fatal("fresh sufficient B replaced reviewed finite A")
			}
		}
		if helpers != before {
			t.Fatal("frozen validation invoked native helper")
		}
		if candidatePrices != beforePrices {
			t.Fatal("frozen validation searched Auto candidates")
		}
	}
	stale := memory("8589934592")
	for i := range stale.Observation.Domains {
		stale.Observation.Domains[i].Total.AgeMS = noderec.MemoryFreshMS + 1
		stale.Observation.Domains[i].Free.AgeMS = noderec.MemoryFreshMS + 1
	}
	if reply := call(settings.WorkerRequest[settings.Request]{Request: finite, Memory: stale, ValidateResources: true, ResourceModel: "weak-model"}); reply.Error == nil {
		t.Fatal("stale target evidence accepted")
	}
	if helpers != before {
		t.Fatal("stale frozen validation invoked helper")
	}
	if candidatePrices != beforePrices {
		t.Fatal("stale validation searched Auto candidates")
	}
	restarted := memory("7516192768")
	restarted.Observation.RunID = "replacement-producer"
	if reply := call(settings.WorkerRequest[settings.Request]{Request: finite, Memory: restarted, ValidateResources: true, ResourceModel: "weak-model"}); reply.Error != nil {
		t.Fatalf("same physical identity producer restart invalidated allocation: %+v", reply.Error)
	}
	if candidatePrices != beforePrices {
		t.Fatal("producer restart triggered Auto search")
	}
	reply = call(settings.WorkerRequest[settings.Request]{Request: request, Memory: memory("7516192768"), NativeInitAllowed: true})
	if reply.Error != nil {
		t.Fatalf("explicit Recalculate refused: %+v", reply.Error)
	}
	var recalculated settings.Preview
	if err := modelconfig.DecodeStrict(reply.Result, &recalculated); err != nil {
		t.Fatal(err)
	}
	if recalculated.Auto == nil || recalculated.Auto.ReviewedInputID == preview.Auto.ReviewedInputID || helpers <= before {
		t.Fatal("explicit Recalculate did not replace the reviewed input through the helper")
	}
	before = helpers
	beforePrices = candidatePrices
	reply = call(settings.WorkerRequest[settings.Request]{Request: finite})
	if reply.Error != nil {
		t.Fatalf("accepted finite replay required telemetry: %+v", reply.Error)
	}
	if helpers != before {
		t.Fatal("accepted replay invoked native helper")
	}
	if candidatePrices != beforePrices {
		t.Fatal("accepted replay searched Auto candidates")
	}
	// Advisory partial costs do not enable production Auto resolution.
	f.ex.modelEstimates = modelEstimateState{}
	reply = call(settings.WorkerRequest[settings.Request]{Request: request, Memory: memory("8589934592"), NativeInitAllowed: true})
	if reply.Error == nil {
		var partial settings.Preview
		if err := modelconfig.DecodeStrict(reply.Result, &partial); err != nil {
			t.Fatal(err)
		}
		if len(partial.Errors) == 0 || partial.Auto != nil && (partial.Auto.Resolved != nil || partial.Auto.Status == "fits") {
			t.Fatal("production partial provider authorized Auto through the worker route")
		}
	}
}

func TestModelSettingsContractHostProvenanceSurvivesUnavailableGPUMap(t *testing.T) {
	test := func(name string, deviceMap *modelDeviceMap) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			in := autoTestInput()
			layers := int32(1)
			in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &layers}
			in.Allocation.Options.GPU = &in.Allocation.GPU
			in.Allocation.DeviceBinding = nil
			fact := noderec.MemoryFact{Valid: true, Bytes: "8589934592", Source: "synthetic-independent-RAM", SampledAt: time.Now().UTC()}
			available := fact
			available.Bytes = "0"
			target := &modelconfig.TargetMemory{HostID: "synthetic-host", ObservationID: "sample", MapID: "unavailable-map", Observation: &noderec.MemoryObservation{RunID: "producer", Domains: []noderec.MemoryDomain{{ID: "host", Kind: "system", Total: fact, Free: available}}}, Hop: noderec.NewMemoryHop("broker", noderec.MemoryBootClock)}
			owner := modelTargetMemoryOwner{target: target, deviceMap: deviceMap}
			var err error
			in.Memory, err = owner.ModelAutoMemory(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			var state modelEstimateState
			out := state.estimate(context.Background(), in)
			if out.CapacityFit != "unknown" || out.FreeFit != "unknown" {
				t.Fatal("partial costs or unavailable GPU mapping advertised a comparison")
			}
			hostSeen, deviceSeen := false, false
			for _, domain := range out.Domains {
				if domain.Kind == "host" {
					hostSeen = true
					if domain.CapacityBytes != fact.Bytes || domain.FreeBytes != "0" || domain.CapacitySource != fact.Source || domain.FreeSource != fact.Source || domain.CapacityTimestamp != fact.SampledAt.Format(time.RFC3339Nano) || domain.FreeTimestamp != fact.SampledAt.Format(time.RFC3339Nano) || domain.RemainingValidityMS <= 0 || domain.RemainingValidityMS > noderec.MemoryFreshMS || domain.CapacityFit != "fits" || domain.FreeFit != "exceeds" || !slices.Contains(domain.Unknowns, "not priced (any pool): compute-workspace") {
						t.Fatalf("independent host facts disappeared with unavailable GPU map: %+v", domain)
					}
				}
				if domain.Kind == "device" || domain.ID == "offload-unmapped" {
					deviceSeen = true
					if domain.CapacityFit != "unknown" || domain.FreeFit != "unknown" || len(domain.Unknowns) == 0 || domain.DeviceID != "" {
						t.Fatalf("unproved device acquired comparison authority: %+v", domain)
					}
				}
			}
			if !hostSeen || !deviceSeen {
				t.Fatal("partial estimate omitted host or unproved device domain")
			}
		})
	}
	test("missing map", nil)
	test("unavailable map", &modelDeviceMap{ObservationID: "unavailable-map", HostPool: "untrusted-map-host", Unknown: "unsupported native mapping"})
	test("mismatched observation", &modelDeviceMap{ObservationID: "other-observation", HostPool: "untrusted-map-host"})
}

func TestModelSettingsContractInspectRelay(t *testing.T) {
	inspect := modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "example/model"}
	relay := settings.Relay{ID: "operation", Method: "inspect", Inspect: &inspect}
	if err := relay.Validate(); err != nil {
		t.Fatal(err)
	}
	relay.Request = settings.Request{Engine: "llamacpp"}
	if err := relay.Validate(); err == nil {
		t.Fatal("inspect accepted a settings mutation alongside the read")
	}
}

func TestModelSettingsContractOversizedOrdinaryPreviewRefused(t *testing.T) {
	e := settingsExecutor(t, false)
	p := settingsRequest(t, e)
	before, err := e.LaunchSettings(p.Engine)
	if err != nil {
		t.Fatal(err)
	}
	// This valid argument is below the existing 16 KiB text bound. Go's JSON
	// HTML escaping expands it, and preview repeats it in parsed arguments.
	p.Settings.LaunchText = "--fixture \"" + strings.Repeat("<", 8*1024) + "\""
	raw, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := settings.DecodeRequest(raw); err != nil {
		t.Fatalf("request must fit before expanded preview: %v", err)
	}
	preview, err := e.PreviewLaunch(p)
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("fixture must be an otherwise valid preview: error=%v fields=%v", err, preview.Errors)
	}
	if err := modelconfig.CheckPayloadSize(preview); err == nil {
		t.Fatal("fixture did not exceed the complete preview budget")
	}
	server, client := net.Pipe()
	t.Cleanup(func() { _ = server.Close(); _ = client.Close() })
	if err := client.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	m := &Manager{exec: e, codec: NewCodec(server)}
	id := json.RawMessage(`1`)
	raw, err = json.Marshal(settings.WorkerRequest[settings.Request]{Request: p})
	if err != nil {
		t.Fatal(err)
	}
	if !m.handleSettingsMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:preview-launch", Params: raw}) {
		t.Fatal("preview not handled")
	}
	var reply Message
	if err := json.NewDecoder(client).Decode(&reply); err != nil {
		t.Fatal(err)
	}
	if reply.Error == nil || reply.Error.Code != -32602 || !strings.Contains(reply.Error.Message, "64 KiB") {
		t.Fatalf("oversized preview not refused before apply: rpcError=%v resultBytes=%d", reply.Error, len(reply.Result))
	}
	after, err := e.LaunchSettings(p.Engine)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatal("refused preview changed launch state")
	}
}

func TestModelSettingsContractNativeOptionCoverage(t *testing.T) {
	options, err := llamaModelSettingsOptions()
	if err != nil {
		t.Fatal(err)
	}
	if err := validateModelSettingsOptionCoverage(llamaServingOptions, options); err != nil {
		t.Fatal(err)
	}
	for i, native := range llamaServingOptions {
		t.Run("rejects missing "+native.key, func(t *testing.T) {
			incomplete := slices.Delete(slices.Clone(options), i, i+1)
			if err := validateModelSettingsOptionCoverage(llamaServingOptions, incomplete); err == nil {
				t.Fatalf("coverage accepted a projection missing actual option %q", native.key)
			}
		})
	}
	t.Run("rejects new unprojected native entry", func(t *testing.T) {
		native := append(slices.Clone(llamaServingOptions), llamaServingOption{key: "new-allocation-option"})
		if err := validateModelSettingsOptionCoverage(native, options); err == nil {
			t.Fatal("coverage accepted a newly added native option without a projection")
		}
	})
}

func TestModelSettingsContractMatrixDocumentsActualNativeOptions(t *testing.T) {
	data, err := os.ReadFile("MANIFEST.md")
	if err != nil {
		t.Fatal(err)
	}
	for _, native := range llamaServingOptions {
		if !strings.Contains(string(data), "| `"+native.key+"` |") {
			t.Errorf("option matrix omits actual native option %q", native.key)
		}
	}
}

func TestModelSettingsContractDescriptorsAreCompleteAndDetached(t *testing.T) {
	options, err := llamaModelSettingsOptions()
	if err != nil {
		t.Fatal(err)
	}
	for _, option := range options {
		d := option.Descriptor
		if err := d.Validate(); err != nil {
			t.Fatalf("%s: %v", option.Native.key, err)
		}
		if d.Label == "" || d.Owner == "" || d.Phase == "" || d.DefaultSource == "" || d.IdentityEffect == "" || d.EstimateEffect == "" || option.Preset == "" {
			t.Fatalf("incomplete metadata for %s: %+v", option.Native.key, option)
		}
		if option.Native.unsupported && d.Supported {
			t.Fatalf("unsupported native option %s exposed as supported", option.Native.key)
		}
	}
	e := modelSettingsContractExecutor(t)
	p := modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "missing-exact-model"}
	first, err := e.InspectModel(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	for i := range first.Options {
		if first.Options[i].Field == "gpu" {
			first.Options[i].Values[0] = "changed by consumer"
		}
	}
	second, err := e.InspectModel(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	if err := second.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, d := range second.Options {
		if slices.Contains(d.Values, "changed by consumer") {
			t.Fatal("inspection descriptors alias consumer-owned values")
		}
	}
}

// No fixture invokes Detect, Start, Prepare or a native helper. Missing artifacts
// and absent review data remain invalid after the implementation bodies land.
func modelSettingsContractExecutor(t *testing.T) *Executor {
	t.Helper()
	root := t.TempDir()
	m := &Manifest{Engine: "llamacpp"}
	st := &engineState{manifest: m, plat: &Platform{Runtime: Runtime{Port: 45000}}, modelsDir: root, contentMu: make(chan struct{}, 1)}
	return &Executor{reg: &Registry{engines: map[string]*Manifest{"llamacpp": m}}, reporter: NewReporter(nil), engines: map[string]*engineState{"llamacpp": st}, baseDir: root, overrideDir: root, emit: func(method string, _ any) { t.Errorf("unexpected effect notification %s", method) }}
}

func TestModelSettingsContractInspectMissingArtifactsHasNoRuntimeEffects(t *testing.T) {
	e := modelSettingsContractExecutor(t)
	st := e.engines["llamacpp"]
	before := st.plat.Runtime
	result, err := e.InspectModel(context.Background(), modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "missing-exact-model"})
	if err != nil {
		t.Fatal(err)
	}
	if result.Status == "resolved" || len(result.Artifacts) != 0 || result.Estimate.Status == "estimated" || result.Estimate.HostBytes != "" || result.Estimate.DeviceBytes != "" {
		t.Fatalf("missing artifacts produced native/estimate authority: %+v", result)
	}
	if !reflect.DeepEqual(st.plat.Runtime, before) || st.running || st.proc != nil {
		t.Fatal("inspection changed runtime")
	}
	entries, err := os.ReadDir(e.overrideDir)
	if err != nil || len(entries) != 0 {
		t.Fatalf("inspection wrote files: entries=%v err=%v", entries, err)
	}
}

func TestModelSettingsContractStrictLaunchIngress(t *testing.T) {
	test := func(name, raw string) {
		t.Run(name, func(t *testing.T) {
			if _, err := decodeModelSettingsLaunch([]byte(raw)); err == nil || rpcCode(err) != -32602 {
				t.Fatalf("invalid request did not return invalid params: %v", err)
			}
		})
	}
	test("forged telemetry", `{"engine":"llamacpp","telemetry":{"freeBytes":"1"}}`)
	test("duplicate selected leaf", `{"engine":"llamacpp","modelChange":{"modelId":"m","expectedPolicyRevision":"0","load":{"set":{"slots":1,"slots":2}}}}`)
	test("caller supplied resolution", `{"engine":"llamacpp","modelChange":{"modelId":"m","expectedPolicyRevision":"0","load":{"set":{},"resolved":{}}}}`)
	test("case variant", `{"engine":"llamacpp","ModelChange":{}}`)
	test("oversize", `{"engine":"llamacpp"}`+strings.Repeat(" ", modelconfig.MaxPayloadBytes))
	test("unbounded Auto", `{"engine":"llamacpp","modelChange":{"modelId":"m","expectedPolicyRevision":"0","load":{"set":{"context":{"mode":"auto"}}}}}`)
	test("invalid context product", `{"engine":"llamacpp","modelChange":{"modelId":"m","expectedPolicyRevision":"0","load":{"set":{"context":{"mode":"fixed","tokens":2147483647},"slots":2}}}}`)
	t.Run("internal CORS guard remains usable", func(t *testing.T) {
		p, err := decodeModelSettingsLaunch([]byte(`{"engine":"llamacpp","preserveCORS":true}`))
		if err != nil || !p.PreserveCORS {
			t.Fatalf("internal guard: request=%+v err=%v", p, err)
		}
		if _, err := settings.DecodeRequest([]byte(`{"engine":"llamacpp","preserveCORS":true}`)); err == nil {
			t.Fatal("public boundary accepted internal caller authority")
		}
	})
}

func TestModelSettingsContractIncompleteLaunchRPCsHaveNoEffects(t *testing.T) {
	test := func(name, method, raw string) {
		t.Run(name, func(t *testing.T) {
			e := modelSettingsContractExecutor(t)
			st := e.engines["llamacpp"]
			before := st.plat.Runtime
			server, client := net.Pipe()
			t.Cleanup(func() { _ = server.Close(); _ = client.Close() })
			if err := client.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
				t.Fatal(err)
			}
			m := &Manager{exec: e, codec: NewCodec(server)}
			id := json.RawMessage(`1`)
			if method == "engine:preview-launch" {
				raw = `{"request":` + raw + `}`
			}
			if !m.handleSettingsMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: method, Params: json.RawMessage(raw)}) {
				t.Fatal("launch method was not handled")
			}
			var reply Message
			if err := json.NewDecoder(client).Decode(&reply); err != nil {
				t.Fatal(err)
			}
			if reply.Error == nil {
				// Later preview bodies may express field refusal or unavailable
				// Auto as read-only preview data rather than a transport error.
				var preview settings.Preview
				if method != "engine:preview-launch" || modelconfig.DecodeStrict(reply.Result, &preview) != nil || len(preview.Errors) == 0 && (preview.Auto == nil || preview.Auto.Resolved != nil) {
					t.Fatalf("incomplete request produced an applicable result: %+v", reply)
				}
			}
			if !reflect.DeepEqual(st.plat.Runtime, before) || st.running || st.proc != nil || len(e.switchMutations) != 0 || len(m.settingsRelay.pending) != 0 {
				t.Fatal("incomplete request changed runtime or entered mutation/rebind")
			}
			entries, err := os.ReadDir(e.overrideDir)
			if err != nil || len(entries) != 0 {
				t.Fatalf("incomplete request wrote settings: entries=%v err=%v", entries, err)
			}
		})
	}
	test("manual preview lacks artifacts", "engine:preview-launch", `{"engine":"llamacpp","modelChange":{"modelId":"missing-exact-model","expectedPolicyRevision":"0","load":{"set":{"slots":1}}}}`)
	test("Auto preview lacks target facts", "engine:preview-launch", `{"engine":"llamacpp","modelChange":{"modelId":"missing-exact-model","expectedPolicyRevision":"0","load":{"set":{"context":{"mode":"auto","floor":4096,"cap":8192}}}}}`)
	test("configure lacks accepted resolution", "engine:configure-launch", `{"engine":"llamacpp","operationId":"review-missing","settings":{"modelLoads":[{"modelId":"missing-exact-model","requested":{"slots":1}}]}}`)
}

func TestModelSettingsContractConfigureUsesValidatedTypedApplication(t *testing.T) {
	test := func(name string, edit func(*testing.T, modelLoadFixture, *settings.Config), wantError bool) {
		t.Run(name, func(t *testing.T) {
			f := newModelLoadFixture(t)
			f.st.plat.Runtime = convertModelLoadFixture(t, f)
			config := settingsRequest(t, f.ex).Settings
			before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
			edit(t, f, &config)
			server, client := net.Pipe()
			t.Cleanup(func() { _ = server.Close(); _ = client.Close() })
			if err := client.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
				t.Fatal(err)
			}
			m := &Manager{exec: f.ex, codec: NewCodec(server)}
			rebinds := 0
			m.settingsRelay.send = func(_ context.Context, method string, value any) error {
				relay, ok := value.(settings.Relay)
				if !ok || method != "engine:settings-request" || relay.Method != "rebind" {
					return fmt.Errorf("unexpected settings relay: %s", method)
				}
				rebinds++
				raw, err := json.Marshal(settingsReply{ID: relay.ID, Result: json.RawMessage(`{}`)})
				if err != nil {
					return err
				}
				m.settingsRelay.reply(raw)
				return nil
			}
			raw, err := json.Marshal(settings.Configure{Engine: "fake", OperationID: "reviewed-typed-apply", Settings: config})
			if err != nil {
				t.Fatal(err)
			}
			id := json.RawMessage(`1`)
			if !m.handleSettingsMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:configure-launch", Params: raw}) {
				t.Fatal("configure-launch was not handled")
			}
			var reply Message
			if err := json.NewDecoder(client).Decode(&reply); err != nil {
				t.Fatal(err)
			}
			if (reply.Error != nil) != wantError {
				t.Fatalf("configure-launch error=%+v wantError=%v", reply.Error, wantError)
			}
			if wantError {
				if rebinds != 0 || !modelconfig.EqualLoads(before, f.st.plat.Runtime.ModelLoads) {
					t.Fatal("refused typed RPC changed settings or rebound")
				}
				assertNoSettingsOverride(t, f.ex)
			} else {
				var state settings.LaunchState
				if err := modelconfig.DecodeStrict(reply.Result, &state); err != nil {
					t.Fatal(err)
				}
				if rebinds != 1 || !settings.ConfigFromLaunch(state, config.ProxyPort).Equal(config) || !modelconfig.EqualLoads(readSettingsOverride(t, f.ex).ModelLoads, config.ModelLoads) {
					t.Fatalf("typed RPC did not apply exact reviewed settings: rebinds=%d state=%+v", rebinds, state)
				}
			}
			if f.st.running || f.st.proc != nil || len(m.settingsRelay.pending) != 0 {
				t.Fatal("typed configuration activated or leaked its relay")
			}
		})
	}
	test("reviewed typed echo", func(*testing.T, modelLoadFixture, *settings.Config) {}, false)
	test("missing resolution", func(t *testing.T, f modelLoadFixture, config *settings.Config) {
		config.ModelLoads[0].Resolved = nil
	}, true)
	test("forged finite resolution", func(t *testing.T, f modelLoadFixture, config *settings.Config) {
		config.ModelLoads[0].Resolved.SourceInputDigest = strings.Repeat("a", 64)
	}, true)
	test("artifact drift", func(t *testing.T, f modelLoadFixture, config *settings.Config) {
		writeInspectionGGUF(t, config.ModelLoads[0].Resolved.Artifacts[0].Path, "llama", 0.75, 4096)
	}, true)
}

func TestModelSettingsContractCoupledAndAutoGuardsRemainClosed(t *testing.T) {
	f := newModelLoadFixture(t)
	p := settingsRequest(t, f.ex)
	p.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", PreviewDigest: strings.Repeat("a", 64), Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "auto"}}}}
	if err := f.ex.validateModelSettingsLaunch(context.Background(), p, false); err == nil {
		t.Fatal("Auto guard opened without a resolver")
	}
	one := int32(1)
	temperature := json.Number("0.5")
	p.ModelChange.Load = &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &one}}
	p.ModelChange.Policy = &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": &temperature}}
	if err := p.Validate(); err != nil {
		t.Fatalf("coupled guard fixture is invalid: %v", err)
	}
	if err := f.ex.validateModelSettingsLaunch(context.Background(), p, false); err == nil {
		t.Fatal("worker accepted a coupled preview outside its owner")
	}
	if err := f.ex.validateModelSettingsLaunch(context.Background(), p, true); err == nil {
		t.Fatal("worker accepted policy application outside its owner")
	}
	assertNoSettingsOverride(t, f.ex)
	if f.st.running || f.st.proc != nil {
		t.Fatal("refused model changes activated the engine")
	}
}

func TestModelSettingsContractEstimateAndAutoIncompleteInputs(t *testing.T) {
	p := modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "missing-exact-model"}
	request := modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "auto", Floor: 4096, Cap: 8192}}
	estimate, err := estimateModelSettings(context.Background(), p, request)
	if err != nil {
		t.Fatal(err)
	}
	if estimate.Status == "estimated" || estimate.HostBytes != "" || estimate.DeviceBytes != "" {
		t.Fatalf("incomplete inputs produced a fit claim: %+v", estimate)
	}
	proposal, err := resolveModelSettingsAuto(context.Background(), p, request)
	if proposal.Resolved != nil || proposal.ResolverRevision != "" || proposal.ReviewedInputID != "" || err == nil && proposal.Status != "unavailable" && proposal.Status != "unknown" {
		t.Fatalf("incomplete inputs produced Auto authority: proposal=%+v err=%v", proposal, err)
	}
	if request.Context.Floor != 4096 || request.Context.Cap != 8192 {
		t.Fatal("Auto changed explicit caller constraints")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := estimateModelSettings(ctx, p, request); err != context.Canceled {
		t.Fatalf("cancelled estimate = %v", err)
	}
}

func TestModelSettingsContractPinnedInspectIsReadOnlyAndCorrelated(t *testing.T) {
	relay := &settingsRelay{}
	relay.send = func(_ context.Context, method string, value any) error {
		request, ok := value.(settings.Relay)
		if !ok || method != "engine:settings-request" || request.Method != "inspect" || request.Inspect == nil || !reflect.DeepEqual(request.Request, settings.Request{}) || request.Inspect.NodeID != "" || request.Caller != "verified-peer" {
			return fmt.Errorf("invalid correlated inspect: %s %+v", method, value)
		}
		// Exercise the maximum model identifier beyond Request.Resolution's
		// public 128-byte budget: that field is never serialized for inspect.
		if request.Inspect.ModelID != strings.Repeat("m", 512) {
			return fmt.Errorf("inspect model identity was changed")
		}
		raw, err := json.Marshal(settingsReply{ID: request.ID, Result: json.RawMessage(`{"status":"unavailable"}`)})
		if err != nil {
			return err
		}
		relay.reply(raw)
		return nil
	}
	e := &Executor{settingsParent: relay.call}
	s := &controlServer{exec: e}
	for _, route := range s.settingsRoutes() {
		if route.path == settingsPath+"inspect" && (route.class != controlRead || route.method != http.MethodPost) {
			t.Fatalf("inspect route is not a POST read: %+v", route)
		}
	}
	body := `{"engine":"llamacpp","nodeId":"must-not-forward","modelId":"` + strings.Repeat("m", 512) + `"}`
	r := httptest.NewRequest(http.MethodPost, settingsPath+"inspect", strings.NewReader(body))
	w := httptest.NewRecorder()
	s.settingsHandler("inspect")(w, r, "verified-peer")
	if w.Code != http.StatusOK || len(relay.pending) != 0 {
		t.Fatalf("inspect route status=%d body=%s pending=%d", w.Code, w.Body.String(), len(relay.pending))
	}
}

func TestModelSettingsContractMalformedPinnedInspectNeverRelays(t *testing.T) {
	e := &Executor{settingsParent: func(context.Context, string, settings.Request, string) (json.RawMessage, error) {
		t.Fatal("invalid inspect reached broker")
		return nil, nil
	}}
	s := &controlServer{exec: e}
	for name, body := range map[string]string{
		"settings payload": `{"engine":"llamacpp","modelId":"m","settings":{}}`,
		"forged caller":    `{"engine":"llamacpp","modelId":"m","caller":"local"}`,
		"telemetry":        `{"engine":"llamacpp","modelId":"m","freeBytes":"1"}`,
		"duplicate ID":     `{"engine":"llamacpp","modelId":"m","modelId":"other"}`,
		"invalid model":    `{"engine":"llamacpp","modelId":"bad\nmodel"}`,
	} {
		t.Run(name, func(t *testing.T) {
			w := httptest.NewRecorder()
			s.settingsHandler("inspect")(w, httptest.NewRequest(http.MethodPost, settingsPath+"inspect", strings.NewReader(body)), "verified-peer")
			if w.Code != http.StatusBadRequest {
				t.Fatalf("invalid inspect status=%d body=%s", w.Code, w.Body.String())
			}
		})
	}
}

func TestModelSettingsContractInspectKeepsTypedRefusals(t *testing.T) {
	for _, code := range []int{-32010, -32011, -32012, -32014} {
		t.Run(fmt.Sprint(code), func(t *testing.T) {
			relay := &settingsRelay{}
			relay.send = func(_ context.Context, method string, value any) error {
				request, ok := value.(settings.Relay)
				if !ok || method != "engine:settings-request" || request.Inspect == nil {
					return fmt.Errorf("unexpected inspect relay: %s %+v", method, value)
				}
				raw, err := json.Marshal(settingsReply{ID: request.ID, Error: "target refusal", ErrorCode: code})
				if err != nil {
					return err
				}
				relay.reply(raw)
				return nil
			}
			e := &Executor{settingsParent: relay.call}
			s := &controlServer{exec: e}
			w := httptest.NewRecorder()
			r := httptest.NewRequest(http.MethodPost, settingsPath+"inspect", strings.NewReader(`{"engine":"llamacpp","modelId":"m"}`))
			s.settingsHandler("inspect")(w, r, "verified-peer")
			err := remoteStatusError("inspect", w.Code, w.Body.Bytes())
			if w.Code != http.StatusConflict || rpcCode(err) != code || len(relay.pending) != 0 {
				t.Fatalf("inspect refusal: HTTP=%d code=%d pending=%d err=%v", w.Code, rpcCode(err), len(relay.pending), err)
			}
		})
	}
}

func TestModelSettingsContractPinnedValidationKeepsInvalidParams(t *testing.T) {
	s := &controlServer{exec: &Executor{settingsParent: func(context.Context, string, settings.Request, string) (json.RawMessage, error) {
		t.Fatal("invalid inspection reached broker")
		return nil, nil
	}}}
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, settingsPath+"inspect", strings.NewReader(`{"engine":"llamacpp","modelId":"bad\nmodel"}`))
	s.settingsHandler("inspect")(w, r, "verified-peer")
	if w.Code != http.StatusBadRequest {
		t.Fatalf("validation status=%d", w.Code)
	}
	err := remoteStatusError("remote "+settingsPath+"inspect", w.Code, w.Body.Bytes())
	if rpcCode(err) != -32602 {
		t.Fatalf("remote validation code=%d: %v", rpcCode(err), err)
	}
	for _, prefix := range []string{"remote /v1/switch", "remote /v1/engine-settings-other/inspect"} {
		if got := rpcCode(remoteStatusError(prefix, w.Code, w.Body.Bytes())); got != codeOperationError {
			t.Fatalf("unrelated family %q changed code: %d", prefix, got)
		}
	}
}

func TestModelSettingsContractRegistryPreservesLoadsAndNumbers(t *testing.T) {
	r := NewRegistry()
	r.bundledRaw["llamacpp"] = []byte(`{"engine":"llamacpp","runtime":{"modelLoads":[{"modelId":"base","requested":{}}],"vendorInteger":9007199254740993},"platforms":{"linux/amd64":{"runtime":{"port":45000}}}}`)
	merged, err := r.mergeOntoBundled([]byte(`{"engine":"llamacpp","runtime":{"port":45001}}`))
	if err != nil {
		t.Fatal(err)
	}
	platforms, err := applyPlatformDefaults(merged)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(platforms), `9007199254740993`) || !strings.Contains(string(platforms), `"modelId":"base"`) {
		t.Fatalf("merge lost model loads or numeric precision: %s", platforms)
	}
	changed, err := r.mergeOntoBundled([]byte(`{"engine":"llamacpp","runtime":{"modelLoads":[]}}`))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(changed), `"modelId":"base"`) {
		t.Fatalf("explicit empty load collection did not replace inherited loads: %s", changed)
	}
}

func TestModelSettingsContractRegistryCanonicalizesHostLoads(t *testing.T) {
	m := validManifest()
	p := m.Platforms["linux/amd64"]
	p.Runtime.ModelLoads = []modelconfig.ModelLoad{{ModelID: "z", Requested: modelconfig.RequestedLoad{}}, {ModelID: "a", Requested: modelconfig.RequestedLoad{}}}
	m.Platforms["linux/amd64"] = p
	if err := m.Validate(); err != nil {
		t.Fatal(err)
	}
	got := m.Platforms["linux/amd64"].Runtime.ModelLoads
	if got[0].ModelID != "a" || got[1].ModelID != "z" {
		t.Fatalf("host model loads were not canonicalized: %+v", got)
	}
}

func TestModelSettingsContractRegistryRejectsInvalidLoadsBeforeMerge(t *testing.T) {
	for name, loads := range map[string]string{
		"duplicate exact model":  `[{"modelId":"m","requested":{}},{"modelId":"m","requested":{}}]`,
		"unknown load option":    `[{"modelId":"m","requested":{"telemetry":{}}}]`,
		"duplicate native field": `[{"modelId":"m","requested":{"slots":1,"slots":2}}]`,
		"invalid exact model":    `[{"modelId":"bad\nmodel","requested":{}}]`,
		"null collection":        `null`,
	} {
		t.Run(name, func(t *testing.T) {
			raw := []byte(`{"runtime":{"modelLoads":` + loads + `}}`)
			if _, err := applyPlatformDefaults(raw); err == nil {
				t.Fatal("invalid loads survived shared-runtime merge")
			}
			if _, err := applyPlatformDefaults([]byte(`{"platforms":{"linux/amd64":` + string(raw) + `}}`)); err == nil {
				t.Fatal("invalid loads survived host-platform merge")
			}
		})
	}
	t.Run("duplicate collection key", func(t *testing.T) {
		if _, err := applyPlatformDefaults([]byte(`{"runtime":{"modelLoads":[],"modelLoads":[]}}`)); err == nil {
			t.Fatal("duplicate typed collection key survived the preserving merge")
		}
	})
}
