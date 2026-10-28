// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

type modelBufferTestFixture struct {
	load      modelLoadFixture
	capture   *modelBufferCapture
	sequences [2]uint64
}

func newModelBufferTestFixture(t *testing.T, configure ...func(*modelLoadFixture)) *modelBufferTestFixture {
	t.Helper()
	f := newModelLoadFixture(t)
	for _, configure := range configure {
		configure(&f)
	}
	if err := os.Chmod(f.ex.baseDir, 0o700); err != nil {
		t.Fatal(err)
	}
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	autoTestNativeSource(t, &f) // synthetic regular files, never executed or loaded
	rt := f.st.plat.Runtime
	command := typedFixtureCommand(t, f, rt)
	preview, err := compilePreparedServing(f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	rt.ModelLoads = modelconfig.CloneLoads(preview.ModelLoads)
	f.st.plat.Runtime = rt
	var inputs map[string]modelEstimateInput
	prepared, err := compilePreparedServing(f.st, command, rt, true, &inputs)
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs) != len(prepared.ModelLoads) {
		t.Fatal("actual-start compiler did not forward advisory inputs")
	}
	c := f.ex.newModelBufferCapture(f.st, "fake", prepared, inputs, command)
	if c == nil || c.sourceID == "" {
		t.Fatal("synthetic native source binding unavailable")
	}
	c.bind(&managedProc{done: make(chan struct{})})
	f.st.mu.Lock()
	f.st.bufferCapture = c
	f.st.proc = c.proc
	f.st.running = true
	f.st.stopping = false
	f.st.adopted = false
	f.st.runID = "synthetic-observed-run"
	f.st.plat.Runtime.servingLaunch = prepared
	f.st.plat.Runtime.servingCommand = command
	c.publishRun(f.st.runID)
	f.st.mu.Unlock()
	t.Cleanup(func() {
		c.retire()
		f.ex.modelBuffers.close()
		f.st.mu.Lock()
		f.st.proc = nil
		f.st.running = false
		f.st.bufferCapture = nil
		f.st.mu.Unlock()
	})
	return &modelBufferTestFixture{load: f, capture: c}
}

func (f *modelBufferTestFixture) line(stream, line string) {
	i := modelBufferStream(stream)
	f.sequences[i]++
	f.capture.observe(managedOutputFrame{Stream: stream, Sequence: f.sequences[i], Line: line})
}
func (f *modelBufferTestFixture) initial(model string, port int) {
	f.line("stderr", fmt.Sprintf("srv         load: spawning server instance with name=%s on port %d", model, port))
	f.allocations(port, true)
}
func (f *modelBufferTestFixture) allocations(port int, ready bool) {
	for _, message := range []string{"srv   load_model: loading model '/private/not-retained.gguf'", "load_tensors: CPU model buffer size = 1.00 MiB", "llama_kv_cache: CPU KV buffer size = 0.25 MiB", "sched_reserve: reserving ...", "sched_reserve: CPU compute buffer size = 0.50 MiB", "sched_reserve: reserve took 1.00 ms, sched copies = 1"} {
		f.line("stdout", fmt.Sprintf("[%5d] %s", port, message))
	}
	if ready {
		f.line("stdout", fmt.Sprintf("[%5d] srv         main: model loaded", port))
	}
}
func (f *modelBufferTestFixture) confirm(t *testing.T, model string, sweep uint64) modelconfig.BufferObservation {
	t.Helper()
	raw, err := json.Marshal(map[string]any{"data": []any{map[string]any{"id": model, "status": map[string]string{"value": "loaded"}}}})
	if err != nil {
		t.Fatal(err)
	}
	frozen := f.capture.frozen[model]
	props := map[string]modelBufferProps{model: {Build: "b11146-7fe450e", RouterBuild: "b11146-7fe450e", Context: int64(frozen.context), Slots: int64(frozen.slots)}}
	f.load.ex.observeModelBuffers(context.Background(), f.load.st, f.capture, sweep, []string{model}, true, raw, props)
	f.load.ex.modelBuffers.mu.Lock()
	done := f.load.ex.modelBuffers.done
	f.load.ex.modelBuffers.mu.Unlock()
	if done != nil {
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("publication worker did not finish")
		}
	}
	return f.load.ex.projectModelBufferObservation(f.load.st, "fake", model, frozen.artifacts)
}

func TestModelBufferLoadCorrelation(t *testing.T) {
	t.Run("draft and main constructors remain mixed role occurrences", func(t *testing.T) {
		f := newModelBufferTestFixture(t, func(f *modelLoadFixture) {
			data, err := os.ReadFile(f.source)
			if err != nil {
				t.Fatal(err)
			}
			text := strings.Replace(string(data), "[medium-model]\nmodel = medium.gguf\n", "[medium-model]\nmodel = medium.gguf\nmodel-draft = weak.gguf\n", 1)
			if text == string(data) {
				t.Fatal("draft fixture did not update preset")
			}
			writeTestFile(t, f.source, []byte(text))
		})
		f.line("stderr", "srv         load: spawning server instance with name=medium-model on port 31001")
		f.allocations(31001, false)
		f.line("stdout", "[31001] load_tensors: CPU model buffer size = 1.00 MiB")
		f.line("stdout", "[31001] llama_kv_cache: CPU KV buffer size = 0.25 MiB")
		f.line("stdout", "[31001] srv         main: model loaded")
		o := f.confirm(t, "medium-model", 1)
		if len(o.Artifacts) != 2 || o.Artifacts[1].Role != "draft" || len(o.Rows) != 5 {
			t.Fatalf("mixed role report lost occurrences: %+v", o)
		}
		for _, row := range o.Rows {
			if row.Role != "mixed" {
				t.Fatal("constructor order was treated as an artifact role")
			}
		}
		for _, comparison := range o.Comparisons {
			if comparison.Status != "unknown" {
				t.Fatal("mixed role bytes became a component total")
			}
		}
	})
	t.Run("typed load without serving declaration", func(t *testing.T) {
		f := newModelBufferTestFixture(t)
		if f.load.st.plat.Runtime.Serving != nil {
			t.Fatal("fixture invented a serving declaration")
		}
		f.initial("medium-model", 31001)
		o := f.confirm(t, "medium-model", 1)
		if err := o.Validate(); err != nil {
			t.Fatal(err)
		}
		if !o.Current || len(o.Rows) != 3 || o.Persistence != "saved" || len(o.Baseline) == 0 {
			t.Fatalf("uncorrelated report: %+v", o)
		}
		if err := o.Validate(); err != nil {
			t.Fatal(err)
		}
		if o.SourceIdentity == o.UpstreamPin || o.BuildAssurance != "recognized-self-reported-pin" {
			t.Fatal("source digest was treated as source attestation")
		}
		data, err := os.ReadFile(f.load.ex.modelBuffers.store.path)
		if err != nil {
			t.Fatal(err)
		}
		for _, private := range []string{"not-retained", "medium-model", f.load.root, "PAIR_TYPED", "LLAMA_CACHE"} {
			if strings.Contains(string(data), private) {
				t.Fatalf("history retained private text %q", private)
			}
		}
	})
	t.Run("multiple children demultiplex by parent name and port", func(t *testing.T) {
		f := newModelBufferTestFixture(t)
		f.line("stderr", "srv         load: spawning server instance with name=medium-model on port 31001")
		f.initial("weak-model", 31002)
		f.allocations(31001, true)
		a := f.confirm(t, "medium-model", 1)
		b := f.confirm(t, "weak-model", 2)
		if a.SampleID == "" || b.SampleID == "" || a.SampleID == b.SampleID || a.PerSlotContext == b.PerSlotContext {
			t.Fatalf("children mixed: %+v %+v", a, b)
		}
	})
	t.Run("delayed spawn and reused port cannot correlate", func(t *testing.T) {
		f := newModelBufferTestFixture(t)
		f.allocations(31001, true)
		f.initial("medium-model", 31001)
		if got := f.confirm(t, "medium-model", 1); got.Current || len(got.Rows) != 0 {
			t.Fatalf("delayed boundaries mixed: %+v", got)
		}
	})
}

func TestModelBufferDeviceBinding(t *testing.T) {
	f := newModelBufferTestFixture(t)
	f.line("stderr", "srv         load: spawning server instance with name=medium-model on port 31001")
	for _, message := range []string{"srv   load_model: loading model 'not-an-identity'", "load_tensors: CUDA0 model buffer size = 1.00 MiB", "load_tensors: MTL0_Mapped model buffer size = 2.00 MiB", "sched_reserve: reserving ...", "sched_reserve: CUDA0 compute buffer size = 0.25 MiB", "sched_reserve: reserve took 1.00 ms, sched copies = 1", "srv         main: model loaded"} {
		f.line("stdout", "[31001] "+message)
	}
	o := f.confirm(t, "medium-model", 1)
	if len(o.Rows) != 3 {
		t.Fatalf("unmapped reports lost: %+v", o)
	}
	for _, row := range o.Rows {
		if row.DeviceBinding != "" || row.PoolIdentity != "" || row.ReportedMiB == "" {
			t.Fatal("ordinal became physical identity or number was lost")
		}
	}
	for _, comparison := range o.Comparisons {
		if comparison.Status != "unknown" {
			t.Fatal("unmapped device compared")
		}
	}
	t.Run("reviewed UUID and existing matching topology", func(t *testing.T) {
		memory := autoTestMemory(time.Now())
		memory.Target = modelBufferTestTargetMemory()
		allocation := estimateFixture("llama").Allocation
		allocation.DeviceBinding = autoTestDeviceBinding()
		frozen := &modelBufferFrozen{input: modelEstimateInput{Allocation: allocation}}
		modelBufferFreezeTopology(frozen, memory, allocation.DeviceBinding.DeviceID)
		if frozen.deviceBindingID == "" || frozen.devicePoolID == "" || frozen.hostPoolID == frozen.devicePoolID {
			t.Fatal("reviewed physical binding was not retained separately")
		}
		before := frozen.devicePoolID
		memory.Target.Observation.Domains[0], memory.Target.Observation.Domains[1] = memory.Target.Observation.Domains[1], memory.Target.Observation.Domains[0]
		reordered := &modelBufferFrozen{input: modelEstimateInput{Allocation: allocation}}
		modelBufferFreezeTopology(reordered, memory, allocation.DeviceBinding.DeviceID)
		if reordered.devicePoolID != before {
			t.Fatal("array order changed physical binding")
		}
		mismatch := &modelBufferFrozen{input: modelEstimateInput{Allocation: allocation}}
		modelBufferFreezeTopology(mismatch, memory, "GPU-22222222-2222-2222-2222-222222222222")
		if mismatch.devicePoolID != "" || mismatch.deviceBindingID != "" {
			t.Fatal("mismatched actual visibility became a physical binding")
		}
	})
}

func modelBufferTestTargetMemory() *modelconfig.TargetMemory {
	now := time.Now()
	fact := noderec.MemoryFact{Valid: true, Bytes: "8589934592", Source: "synthetic", SampledAt: now.UTC()}
	binding := autoTestDeviceBinding()
	return &modelconfig.TargetMemory{HostID: binding.HostID, ObservationID: "synthetic-observation", MapID: "synthetic-map", Observation: &noderec.MemoryObservation{RunID: "synthetic-producer", Domains: []noderec.MemoryDomain{{ID: "host", Kind: "system", Total: fact, Free: fact}, {ID: "gpu", Kind: "dedicated", DeviceIDs: []string{binding.DeviceID, "pci:0000:ab:00.0", modelDevicesFixtureLUID}, Total: fact, Free: fact}}}, Hop: noderec.NewMemoryHop("synthetic-broker", noderec.MemoryBootClock)}
}

func TestModelBufferCaptureLifetime(t *testing.T) {
	t.Run("contradictory child or router props override historical numbers", func(t *testing.T) {
		for _, change := range []func(*modelBufferProps){
			func(p *modelBufferProps) { p.Build = "b11147-7fe450e" },
			func(p *modelBufferProps) { p.Build = "b11146-7fe450e-dirty" },
			func(p *modelBufferProps) { p.RouterBuild = "other" },
			func(p *modelBufferProps) { p.Context++ },
			func(p *modelBufferProps) { p.Slots++ },
		} {
			f := newModelBufferTestFixture(t)
			f.initial("medium-model", 31001)
			_ = f.confirm(t, "medium-model", 1)
			f.line("stdout", "[31001] srv handle_sleep: server is entering sleeping state")
			f.line("stdout", "[31001] srv handle_sleep: server is exiting sleeping state")
			f.allocations(31001, false)
			frozen := f.capture.frozen["medium-model"]
			props := modelBufferProps{Build: "b11146-7fe450e", RouterBuild: "b11146-7fe450e", Context: int64(frozen.context), Slots: int64(frozen.slots)}
			change(&props)
			f.load.ex.observeModelBuffers(context.Background(), f.load.st, f.capture, 2, []string{"medium-model"}, true, json.RawMessage(`{"data":[{"id":"medium-model","status":{"value":"loaded"}}]}`), map[string]modelBufferProps{"medium-model": props})
			got := f.load.ex.projectModelBufferObservation(f.load.st, "fake", "medium-model", frozen.artifacts)
			if got.Status != "unknown" || got.Current || len(got.Rows) != 0 || got.Reason != "unsupported-build" {
				t.Fatal("contradictory build/allocation reused historical numbers")
			}
		}
	})
	t.Run("raw output before process publication remains detached", func(t *testing.T) {
		f := newModelBufferTestFixture(t)
		f.capture.bind(nil)
		f.load.st.mu.Lock()
		f.capture.publishRun("")
		f.load.st.proc = nil
		f.load.st.running = false
		f.load.st.mu.Unlock()
		f.initial("medium-model", 31001)
		if o := f.confirm(t, "medium-model", 1); o.Current || o.SampleID != "" {
			t.Fatal("pre-publication output acquired authority")
		}
		proc := &managedProc{done: make(chan struct{})}
		f.capture.bind(proc)
		f.load.st.mu.Lock()
		f.load.st.proc = proc
		f.load.st.running = true
		f.load.st.runID = "published-run"
		f.capture.publishRun(f.load.st.runID)
		f.load.st.mu.Unlock()
		if o := f.confirm(t, "medium-model", 2); !o.Current {
			t.Fatal("pre-publication capture did not bind the actual run")
		}
	})
	t.Run("changed generation during a loaded sweep cannot confirm wakeup", func(t *testing.T) {
		f := newModelBufferTestFixture(t)
		f.initial("medium-model", 31001)
		boundary := f.capture.captureBoundary()
		f.line("stdout", "[31001] srv handle_sleep: server is entering sleeping state")
		f.line("stdout", "[31001] srv handle_sleep: server is exiting sleeping state")
		f.allocations(31001, false)
		frozen := f.capture.frozen["medium-model"]
		props := map[string]modelBufferProps{"medium-model": {Build: "b11146-7fe450e", RouterBuild: "b11146-7fe450e", Context: int64(frozen.context), Slots: int64(frozen.slots)}}
		f.load.ex.observeModelBuffers(context.Background(), f.load.st, f.capture, 1, []string{"medium-model"}, true, json.RawMessage(`{"data":[{"id":"medium-model","status":{"value":"loaded"}}]}`), props, boundary)
		if o := f.load.ex.projectModelBufferObservation(f.load.st, "fake", "medium-model", frozen.artifacts); o.Current {
			t.Fatal("old loaded response confirmed a new generation")
		}
		if o := f.confirm(t, "medium-model", 2); !o.Current {
			t.Fatal("fresh wakeup sweep did not recover")
		}
	})
	t.Run("reload on a fresh port has a separate lifetime", func(t *testing.T) {
		f := newModelBufferTestFixture(t)
		f.initial("medium-model", 31001)
		first := f.confirm(t, "medium-model", 1)
		// server_monitor::run emits this through SRV_INF's __func__ prefix.
		f.line("stderr", "srv   run: instance name=medium-model exited with status 0")
		if exited := f.load.ex.projectModelBufferObservation(f.load.st, "fake", "medium-model", f.capture.frozen["medium-model"].artifacts); exited.Current {
			t.Fatal("exited child retained a current report")
		}
		f.initial("medium-model", 31002)
		second := f.confirm(t, "medium-model", 2)
		if !second.Current || second.SampleID == first.SampleID || len(second.Rows) != 3 {
			t.Fatal("reload reused the prior child aggregate")
		}
	})
	t.Run("clock reversal withdraws currentness", func(t *testing.T) {
		f := newModelBufferTestFixture(t)
		f.initial("medium-model", 31001)
		_ = f.confirm(t, "medium-model", 1)
		f.capture.mu.Lock()
		f.capture.wallHigh = time.Now().Add(time.Minute).UnixNano()
		f.capture.mu.Unlock()
		o := f.load.ex.projectModelBufferObservation(f.load.st, "fake", "medium-model", f.capture.frozen["medium-model"].artifacts)
		if o.Current || o.Reason != "clock-reversal" {
			t.Fatalf("clock reversal: %+v", o)
		}
	})
	t.Run("ordinary wakeup has no second initial ready marker", func(t *testing.T) {
		f := newModelBufferTestFixture(t)
		f.initial("medium-model", 31001)
		first := f.confirm(t, "medium-model", 1)
		f.line("stdout", "[31001] srv handle_sleep: server is entering sleeping state")
		if got := f.load.ex.projectModelBufferObservation(f.load.st, "fake", "medium-model", f.capture.frozen["medium-model"].artifacts); got.Current {
			t.Fatal("sleep retained currentness")
		}
		f.line("stdout", "[31001] srv handle_sleep: server is exiting sleeping state")
		f.allocations(31001, false)
		second := f.confirm(t, "medium-model", 2)
		if !second.Current || second.LoadPhase != "wakeup" || second.SampleID == first.SampleID || len(second.Rows) != len(first.Rows) {
			t.Fatalf("wakeup mixed generations: %+v", second)
		}
		artifacts := f.capture.frozen["medium-model"].artifacts
		f.capture.retire()
		if got := f.load.ex.projectModelBufferObservation(f.load.st, "fake", "medium-model", artifacts); got.Current {
			t.Fatal("retired process remained current")
		}
	})
	t.Run("oversized output invalidates complete lifetime", func(t *testing.T) {
		f := newModelBufferTestFixture(t)
		f.initial("medium-model", 31001)
		f.sequences[0]++
		f.capture.observe(managedOutputFrame{Stream: "stdout", Sequence: f.sequences[0], Failure: "oversized-frame"})
		if got := f.confirm(t, "medium-model", 1); got.Current {
			t.Fatal("dropped frame became current")
		}
	})
	t.Run("source replacement invalidates currentness", func(t *testing.T) {
		f := newModelBufferTestFixture(t)
		f.initial("medium-model", 31001)
		if !f.confirm(t, "medium-model", 1).Current {
			t.Fatal("missing starting report")
		}
		writeTestFile(t, filepath.Join(f.load.root, "libggml-base.dylib"), []byte("modified self-reporting same tuple"))
		if got := f.confirm(t, "medium-model", 2); got.Current {
			t.Fatal("source replacement inherited current sample")
		}
	})
	t.Run("actual-start hook binds owned process and published run", func(t *testing.T) {
		f := newModelLoadFixture(t)
		f.st.plat.Runtime = convertModelLoadFixture(t, f)
		if err := f.ex.Start(context.Background(), "fake"); err != nil {
			t.Fatal(err)
		}
		f.st.mu.Lock()
		c := f.st.bufferCapture
		bound := c != nil && c.currentLocked(f.st)
		frozen := false
		if c != nil {
			c.mu.Lock()
			frozen = len(c.frozen) > 0
			c.mu.Unlock()
		}
		if !bound || !frozen {
			f.st.mu.Unlock()
			t.Fatal("actual start did not bind compiler advisory to owned run")
		}
		f.st.mu.Unlock()
		if err := f.ex.Stop("fake"); err != nil {
			t.Fatal(err)
		}
	})
}

func TestModelBufferInspectionHasNoNativeEffects(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	now := time.Now()
	f.ex.modelEstimates.now = func() time.Time { return now }
	before := inspectFixture(t, f)
	store := f.ex.modelBuffers.store
	f.ex.modelBuffers.store = nil
	without := inspectFixture(t, f)
	f.ex.modelBuffers.store = store
	after := inspectFixture(t, f)
	before.LoadObservation = nil
	without.LoadObservation = nil
	after.LoadObservation = nil
	if !reflect.DeepEqual(before, without) || !reflect.DeepEqual(before, after) {
		t.Fatal("projection changed artifact/estimate results")
	}
	t.Run("existing explicit permission gated helper adds no observation calls", func(t *testing.T) {
		f := newInspectionFixture(t, 4096)
		f.st.installDir = f.root
		f.st.binPath = filepath.Join(f.root, "synthetic-native")
		f.st.plat.Runtime.Bin = f.st.binPath
		writeTestFile(t, f.st.binPath, []byte("synthetic executable binding, never executed"))
		writeTestFile(t, filepath.Join(f.root, "libggml-base.dylib"), []byte("synthetic library, never loaded"))
		f.st.plat.Install = &Install{Artifacts: []InstallArtifact{{Name: "server", URL: "https://fixture.invalid/releases/download/b11146/native", SHA256: strings.Repeat("a", 64)}}}
		helpers := 0
		runner := func(ctx context.Context, _ modelDeviceRuntime, _ []string) (modelDeviceReply, error) {
			helpers++
			return modelDeviceReply{Devices: []modelNativeDevice{{Selector: "CPU", Kind: 0}, {Selector: "CUDA0", Kind: 1, UUID: autoTestDeviceBinding().DeviceID, PCI: "0000:ab:00.0", LUID: modelDevicesFixtureLUID}}}, ctx.Err()
		}
		store := f.ex.modelBuffers.store
		baselineCalls := 0
		for _, enabled := range []bool{false, true} {
			if enabled {
				f.ex.modelBuffers.store = store
			} else {
				f.ex.modelBuffers.store = nil
			}
			before := helpers
			ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{DeviceRunner: runner})
			ctx = f.ex.modelMemoryContext(ctx, "llamacpp", modelBufferTestTargetMemory(), true)
			afterExplicit := helpers
			if _, err := f.ex.InspectModel(ctx, f.request); err != nil {
				t.Fatal(err)
			}
			if helpers != afterExplicit {
				t.Fatal("observation projection invoked an additional native helper")
			}
			if !enabled {
				baselineCalls = helpers - before
				if baselineCalls == 0 {
					t.Fatal("permitted explicit helper baseline was not exercised")
				}
			}
			if enabled && helpers-before != baselineCalls {
				t.Fatalf("explicit helper baseline changed with projection %v: %d", enabled, helpers-before)
			}
		}
		before := helpers
		ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{DeviceRunner: runner})
		ctx = f.ex.modelMemoryContext(ctx, "llamacpp", modelBufferTestTargetMemory(), false)
		if _, err := f.ex.InspectModel(ctx, f.request); err != nil {
			t.Fatal(err)
		}
		if helpers != before {
			t.Fatal("prohibited helper initialized because of observation")
		}
	})
}

func TestModelBufferResponseBudget(t *testing.T) {
	base := modelconfig.InspectResult{Engine: "llamacpp", ModelID: "exact", Status: "resolved", Artifacts: []modelconfig.ArtifactSummary{{Role: "model", Identity: strings.Repeat("a", 64), Size: "1", Assurance: "sampled"}}, Facts: &modelconfig.Facts{Status: "known"}, Estimate: modelEstimateUnknown("unavailable")}
	parser := &modelBufferParser{}
	for i := 0; i < 256; i++ {
		parser.consume(managedOutputFrame{Stream: "stdout", Sequence: uint64(i + 1), Line: "load_tensors: CPU model buffer size = 1.00 MiB"})
	}
	observation := modelconfig.BufferObservation{Status: "partial", SampleID: strings.Repeat("b", 64), InputIdentity: strings.Repeat("c", 64), SourceIdentity: strings.Repeat("d", 64), OptionsIdentity: strings.Repeat("e", 64), UpstreamPin: modelBufferPin, BuildAssurance: "recognized-self-reported-pin", EstimatorRevision: modelEstimatorRevision, Rows: parser.rows}
	t.Run("history checkpoint obeys its independent encoded ceiling", func(t *testing.T) {
		sample := modelBufferStoredSample{ModelIdentity: strings.Repeat("a", 64), Observation: observation.Clone()}
		sample.Observation.RecordedAt = time.Now().UTC().Format(time.RFC3339Nano)
		modelBufferBoundSample(&sample)
		if err := validateModelBufferSample(sample); err != nil {
			t.Fatal(err)
		}
		if len(sample.Observation.Rows) >= len(observation.Rows) || sample.Observation.Reason != "capture-budget" {
			t.Fatal("oversized checkpoint did not retain a partial bounded snapshot")
		}
		for range modelconfig.MaxArtifacts {
			sample.Observation.Artifacts = append(sample.Observation.Artifacts, modelconfig.ArtifactSummary{Role: "model", Identity: strings.Repeat("f", 64), Size: "1", Assurance: "sampled-local-binding"})
		}
		modelBufferBoundSample(&sample)
		if err := validateModelBufferSample(sample); err != nil {
			t.Fatal(err)
		}
		if sample.Observation.Status != "unknown" || sample.Observation.Current {
			t.Fatal("over-budget identity metadata remained a current report")
		}
	})
	t.Run("reduces only optional rows", func(t *testing.T) {
		got := attachModelBufferObservation(base, observation.Clone())
		if got.LoadObservation == nil || len(got.LoadObservation.Rows) == 0 || len(got.LoadObservation.Rows) >= len(observation.Rows) || got.LoadObservation.Reason != "response-budget" {
			t.Fatalf("summary was not deterministically reduced: %+v", got.LoadObservation)
		}
		if !reflect.DeepEqual(got.Artifacts, base.Artifacts) || !reflect.DeepEqual(got.Facts, base.Facts) || !reflect.DeepEqual(got.Estimate, base.Estimate) {
			t.Fatal("observation pressure changed original inspection")
		}
		data, err := json.Marshal(got.LoadObservation)
		if err != nil || len(data) > 4096 {
			t.Fatal("summary exceeds 4 KiB")
		}
	})
	test := func(name string, remaining int, wantOptional bool) {
		t.Run(name, func(t *testing.T) {
			original := base.Clone()
			original.Options = nil
			original.Facts.Unknowns = []string{""}
			expanded := original.Clone()
			options, err := modelInspectionOptions(nil)
			if err != nil {
				t.Fatal(err)
			}
			expanded.Options = options
			expanded.Estimate = modelEstimateUnknown("estimate omitted: complete inspection reply size limit")
			encoded, err := json.Marshal(expanded)
			if err != nil {
				t.Fatal(err)
			}
			original.Facts.Unknowns[0] = strings.Repeat("x", modelconfig.MaxPayloadBytes-1024-remaining-len(encoded))
			got := attachModelBufferObservation(original, observation.Clone())
			if (got.LoadObservation != nil) != wantOptional {
				t.Fatalf("optional field: %+v", got.LoadObservation)
			}
			if got.LoadObservation != nil {
				data, err := json.Marshal(got.LoadObservation)
				if err != nil || len(data) > 256 || got.LoadObservation.Status != "unknown" {
					t.Fatal("compact omission exceeds budget")
				}
			}
			if !reflect.DeepEqual(got.Artifacts, original.Artifacts) || !reflect.DeepEqual(got.Facts, original.Facts) {
				t.Fatal("facts were sacrificed to observation pressure")
			}
			got.Options = options
			got.Estimate = expanded.Estimate
			if err := got.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	}
	test("compact structured omission", 128, true)
	test("absent optional field preserves estimate fallback", 1, false)
}

func TestModelBufferSamplesDoNotChangeAuthority(t *testing.T) {
	f := newModelBufferTestFixture(t)
	files := func() map[string]string {
		t.Helper()
		values := map[string]string{}
		err := filepath.WalkDir(f.load.ex.baseDir, func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() || path == f.load.ex.modelBuffers.store.path {
				return nil
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			values[path] = modelLoadDigest(data)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return values
	}
	filesBefore := files()
	policyBefore := f.load.ex.nodeProjection.advertisement()
	sourceBefore := f.load.ex.memoryEvidenceSource(nil)
	desiredBefore, err := f.load.ex.desired.snapshot([]string{"fake"})
	if err != nil {
		t.Fatal(err)
	}
	before := modelconfig.CloneLoads(f.load.st.plat.Runtime.ModelLoads)
	estimateBefore := estimateFixtureResult(t, estimateFixture("llama"))
	f.initial("medium-model", 31001)
	o := f.confirm(t, "medium-model", 1)
	if o.SampleID == "" {
		t.Fatal("no observation exercised")
	}
	if !modelconfig.EqualLoads(before, f.load.st.plat.Runtime.ModelLoads) {
		t.Fatal("observation changed finite load authority")
	}
	estimateAfter := estimateFixtureResult(t, estimateFixture("llama"))
	estimateBefore.Timestamp = ""
	estimateAfter.Timestamp = ""
	if !reflect.DeepEqual(estimateBefore, estimateAfter) {
		t.Fatal("observation changed estimator output or margins")
	}
	if f.load.st.plat.Runtime.Serving != nil {
		t.Fatal("observation invented serving qualification")
	}
	desiredAfter, err := f.load.ex.desired.snapshot([]string{"fake"})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(filesBefore, files()) || !reflect.DeepEqual(policyBefore, f.load.ex.nodeProjection.advertisement()) || !sameMemoryEvidenceSource(sourceBefore, f.load.ex.memoryEvidenceSource(nil)) || !reflect.DeepEqual(desiredBefore, desiredAfter) {
		t.Fatal("observation changed policy, settings/journal/qualification files, admission identity, or desired state")
	}
}

func TestModelBufferSweepSharesNonLoadingProps(t *testing.T) {
	f := newModelBufferTestFixture(t)
	f.initial("medium-model", 31001)
	var calls []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/props" || r.URL.Query().Get("autoload") != "false" || r.Header.Get(engineIdentityProbeHeader) != "1" {
			t.Error("observation changed non-loading boundary")
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		calls = append(calls, r.URL.Query().Get("model"))
		fmt.Fprint(w, `{"build_info":"b11146-7fe450e","total_slots":2,"default_generation_settings":{"n_ctx":8192}}`)
	}))
	defer server.Close()
	_, port, err := net.SplitHostPort(strings.TrimPrefix(server.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	f.load.st.port, err = strconv.Atoi(port)
	if err != nil {
		t.Fatal(err)
	}
	raw := json.RawMessage(`{"data":[{"id":"medium-model","status":{"value":"loaded"}}]}`)
	frozen := f.capture.frozen["medium-model"]
	shared := map[string]modelBufferProps{"medium-model": {Build: "b11146-7fe450e", Context: int64(frozen.context), Slots: int64(frozen.slots)}}
	f.load.ex.observeModelBufferSweep(context.Background(), f.load.st, f.capture, 1, []string{"medium-model"}, true, raw, shared)
	if len(calls) != 1 || calls[0] != "" {
		t.Fatalf("shared child props were queried twice: %v", calls)
	}
	f.load.ex.observeModelBufferSweep(context.Background(), f.load.st, f.capture, 2, nil, true, json.RawMessage(`{"data":[]}`), nil)
	if len(calls) != 1 {
		t.Fatal("cold set queried props")
	}
}

func TestModelBufferPreparedDeclaredLoad(t *testing.T) {
	f := newLlamaServingFixture(t)
	st := settingsState(t, f.ex)
	native := modelLoadFixture{ex: f.ex, st: st, root: f.root}
	autoTestNativeSource(t, &native)
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	rt := st.plat.Runtime
	command, err := launchForState(st, rt.Port)
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := compilePreparedServing(st, command, rt, true)
	if err != nil {
		t.Fatal(err)
	}
	c := f.ex.newModelBufferCapture(st, "fake", prepared, nil, command)
	if c == nil || c.sourceID == "" {
		t.Fatal("validated declared launch was not capturable")
	}
	c.bind(&managedProc{done: make(chan struct{})})
	st.mu.Lock()
	st.bufferCapture = c
	st.proc = c.proc
	st.running = true
	st.stopping = false
	st.adopted = false
	st.runID = "declared-observed-run"
	st.plat.Runtime.servingLaunch = prepared
	st.plat.Runtime.servingCommand = command
	c.publishRun(st.runID)
	st.mu.Unlock()
	t.Cleanup(func() {
		c.retire()
		f.ex.modelBuffers.close()
		st.mu.Lock()
		st.proc = nil
		st.running = false
		st.bufferCapture = nil
		st.mu.Unlock()
	})
	fixture := &modelBufferTestFixture{load: native, capture: c}
	fixture.initial("medium", 31001)
	o := fixture.confirm(t, "medium", 1)
	if !o.Current || len(o.Rows) != 3 || o.RequestedOptions != nil {
		t.Fatalf("declared launch capture: %+v", o)
	}
	for _, component := range o.Baseline {
		if component.Status != "unknown" {
			t.Fatal("declaration manufactured estimate facts")
		}
	}
}

func TestModelBufferMalformedBuildDoesNotWithdrawServing(t *testing.T) {
	f, server := loadedReadbackFixture(t, "medium")
	for _, build := range []string{`{"unexpected":"object"}`, `"` + strings.Repeat("x", 257) + `"`, `null`} {
		server.mu.Lock()
		server.props = `{"default_generation_settings":{"n_ctx":8192},"total_slots":2,"build_info":` + build + `}`
		server.mu.Unlock()
		f.ex.sweepLoaded(context.Background(), nil, false)
		readbackProof(t, f, 1)
		if len(f.ex.modelIdentity().Engines["fake"].Serving) != 2 {
			t.Fatal("observation-only build evidence withdrew serving identity")
		}
	}
}

func TestModelBufferPropsAllowlist(t *testing.T) {
	for _, data := range []string{
		`{"build_info":"b11146-7fe450e"}`,
		`{"build_info":"b11146-7fe450e","default_generation_settings":{"n_ctx":8192,"temperature":0.7},"total_slots":2,"chat_template":"private template"}`,
	} {
		if got := modelBufferReadProps([]byte(data)); got.Build != "b11146-7fe450e" {
			t.Fatalf("supported props lost build tuple: %+v", got)
		}
	}
	for _, data := range []string{
		`{"build_info":"bad","build_info":"b11146-7fe450e"}`,
		`{"build_info":"b11146-7fe450e","BUILD_INFO":"bad"}`,
		`{"build_info":"b11146-7fe450e","default_generation_settings":{"n_ctx":1,"n_ctx":8192},"total_slots":2}`,
		`{"build_info":"b11146-7fe450e","default_generation_settings":{"N_CTX":8192},"total_slots":2}`,
		`{"build_info":"b11146-7fe450e","total_slots":1.5}`,
	} {
		if got := modelBufferReadProps([]byte(data)); got.Build != "" {
			t.Fatalf("ambiguous observation props accepted: %+v", got)
		}
	}
}

func TestModelBufferPublicationPressure(t *testing.T) {
	f := newModelBufferTestFixture(t)
	entered, release := make(chan struct{}), make(chan struct{})
	var enterOnce, releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	defer unblock()
	f.load.ex.modelBuffers.store.replacer = durablefile.Replacer{SyncDir: func(string) error {
		enterOnce.Do(func() { close(entered) })
		<-release
		return nil
	}}
	confirm := func(sweep uint64) {
		props := map[string]modelBufferProps{}
		for _, name := range []string{"medium-model", "weak-model"} {
			frozen := f.capture.frozen[name]
			props[name] = modelBufferProps{Build: "b11146-7fe450e", RouterBuild: "b11146-7fe450e", Context: int64(frozen.context), Slots: int64(frozen.slots)}
		}
		f.load.ex.observeModelBuffers(context.Background(), f.load.st, f.capture, sweep, []string{"medium-model", "weak-model"}, true, json.RawMessage(`{"data":[{"id":"medium-model","status":{"value":"loaded"}},{"id":"weak-model","status":{"value":"loaded"}}]}`), props)
	}
	f.initial("medium-model", 31001)
	confirm(1)
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not reach publication")
	}
	f.initial("weak-model", 31002)
	confirm(2)
	f.line("stdout", "[31001] srv handle_sleep: server is entering sleeping state")
	f.line("stdout", "[31001] srv handle_sleep: server is exiting sleeping state")
	f.allocations(31001, false)
	confirm(3)
	f.capture.mu.Lock()
	omitted := f.capture.children[31002].checkpoint.omitted.Load()
	f.capture.mu.Unlock()
	if !omitted {
		t.Fatal("coalesced checkpoint did not expose pending omission")
	}
	finished := make(chan struct{})
	go func() {
		f.line("stdout", "[31001] ordinary diagnostic")
		_ = f.load.ex.projectModelBufferObservation(f.load.st, "fake", "weak-model", f.capture.frozen["weak-model"].artifacts)
		f.capture.retire()
		close(finished)
	}()
	select {
	case <-finished:
	case <-time.After(5 * time.Second):
		t.Fatal("disk publication blocked output, Inspect, or retirement")
	}
	f.load.ex.modelBuffers.mu.Lock()
	done := f.load.ex.modelBuffers.done
	retained := f.load.ex.modelBuffers.bytes
	f.load.ex.modelBuffers.mu.Unlock()
	if retained <= 0 {
		t.Fatal("retired capture released memory still retained by jobs")
	}
	unblock()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("publication worker did not drain")
	}
	f.load.ex.modelBuffers.mu.Lock()
	defer f.load.ex.modelBuffers.mu.Unlock()
	if f.load.ex.modelBuffers.bytes != 0 || f.load.ex.modelBuffers.children != 0 {
		t.Fatal("retired capture retained its charged capacity after publication")
	}
}
