// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/engines"
)

func ownedServingFixture(t *testing.T) llamaServingFixture {
	t.Helper()
	f := newLlamaServingFixture(t)
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.ex.Stop("fake"); err != nil {
			t.Error(err)
		}
	})
	return f
}

func TestServingIdentityStableAcrossRestartGenerations(t *testing.T) {
	f := ownedServingFixture(t)
	first := f.ex.modelIdentity().Engines["fake"]
	if len(first.ModelDigests) != 2 || len(first.Serving) != 2 {
		t.Fatalf("missing declaration-derived identity: %+v", first)
	}
	if err := f.ex.Stop("fake"); err != nil {
		t.Fatal(err)
	}
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	second := f.ex.modelIdentity().Engines["fake"]
	if first.EngineRunID == second.EngineRunID || first.LaunchDigest != second.LaunchDigest || !reflect.DeepEqual(first.ModelDigests, second.ModelDigests) {
		t.Fatalf("restart identity changed content/launch or retained run: first=%+v second=%+v", first, second)
	}
	data, err := os.ReadFile(f.startLog)
	if err != nil {
		t.Fatal(err)
	}
	lines := bytes.Split(bytes.TrimSpace(data), []byte("\n"))
	if len(lines) != 2 {
		t.Fatalf("spawn records = %d", len(lines))
	}
	paths := make([]string, 0, 2)
	for _, line := range lines {
		var args []string
		if err := json.Unmarshal(line, &args); err != nil {
			t.Fatal(err)
		}
		for i, arg := range args {
			if arg == "--models-preset" && i+1 < len(args) {
				paths = append(paths, args[i+1])
			}
		}
	}
	if len(paths) != 2 || paths[0] == paths[1] {
		t.Fatalf("restart reused generation path: %v", paths)
	}
}

func TestServingLaunchDigestBindsOriginalLaunchAndPreparedDigest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "runtime")
	writeServingFixture(t, path, []byte("synthetic executable"))
	command := launchCommand{Bin: path, Args: []string{"--models-preset", "/approved/source"}, Env: map[string]string{"MODE": "finite"}}
	prepared := strings.Repeat("a", 64)
	first := servingLaunchDigest(command, prepared)
	if !contentDigest.MatchString(first) {
		t.Fatal("missing launch digest")
	}
	t.Run("prepared digest", func(t *testing.T) {
		if servingLaunchDigest(command, strings.Repeat("b", 64)) == first {
			t.Fatal("prepared inputs omitted")
		}
	})
	t.Run("arguments", func(t *testing.T) {
		changed := command
		changed.Args = append([]string{}, command.Args...)
		changed.Args = append(changed.Args, "--models-max", "2")
		if servingLaunchDigest(changed, prepared) == first {
			t.Fatal("argv omitted")
		}
	})
	t.Run("environment", func(t *testing.T) {
		changed := command
		changed.Env = map[string]string{"MODE": "changed"}
		if servingLaunchDigest(changed, prepared) == first {
			t.Fatal("environment omitted")
		}
	})
	t.Run("executable", func(t *testing.T) {
		writeServingFixture(t, path, []byte("changed executable bytes and size"))
		if servingLaunchDigest(command, prepared) == first {
			t.Fatal("executable omitted")
		}
	})
}

func TestServingContentManifestSemantics(t *testing.T) {
	model := engines.ServingModel{ModelID: "m", Artifacts: []engines.ServingArtifact{
		{Role: "model", SHA256: strings.Repeat("a", 64), Binding: &engines.ServingBinding{SizeBytes: 12}},
		{Role: "draft", SHA256: strings.Repeat("b", 64), Binding: &engines.ServingBinding{SizeBytes: 8}},
	}}
	first := servingContentDigest(model)
	if !contentDigest.MatchString(first) {
		t.Fatal("missing aggregate digest")
	}
	model.ModelID = "relocated"
	model.Artifacts[0].Binding.SourcePath = "/renewed/source"
	model.Artifacts[0].Binding.ResolvedTarget = "/renewed/target"
	model.Artifacts[0].Binding.MtimeNS = "123"
	model.Artifacts[0].Binding.SampleSHA256 = strings.Repeat("c", 64)
	model.Artifacts[0].Provenance.Reference = "operator label"
	if servingContentDigest(model) != first {
		t.Fatal("local metadata entered content digest")
	}
	test := func(name string, edit func(*engines.ServingModel)) {
		t.Run(name, func(t *testing.T) {
			copy := (&engines.ServingConfig{Models: []engines.ServingModel{model}}).Clone().Models[0]
			edit(&copy)
			if servingContentDigest(copy) == first {
				t.Fatal("consumed artifact change omitted")
			}
		})
	}
	test("role", func(m *engines.ServingModel) { m.Artifacts[0].Role = "tokenizer" })
	test("size", func(m *engines.ServingModel) { m.Artifacts[0].Binding.SizeBytes++ })
	test("asserted bytes", func(m *engines.ServingModel) { m.Artifacts[0].SHA256 = strings.Repeat("d", 64) })
	test("native shard order", func(m *engines.ServingModel) { m.Artifacts[0], m.Artifacts[1] = m.Artifacts[1], m.Artifacts[0] })
}

func TestCurrentServingEnvelopeColdValidationAndDetachment(t *testing.T) {
	f := ownedServingFixture(t)
	f.ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
		t.Error("cold validation probed or loaded the engine")
		return nil, errors.New("unexpected engine call")
	})
	envelope, id, err := f.ex.currentServingEnvelope(context.Background(), "fake", "medium")
	if err != nil {
		t.Fatal(err)
	}
	if envelope.ContentDigest != id.ModelDigests["medium"] || envelope.PerSlotContextTokens != 8192 || envelope.Slots != 2 {
		t.Fatalf("invalid current envelope: %+v", envelope)
	}
	*envelope.Memory.HostRAMBytes = 1
	envelope.Placement[0] = "changed"
	*id.Serving["weak"].Memory.HostRAMBytes = 2
	id.ModelDigests["medium"] = "changed"
	next, _, err := f.ex.currentServingEnvelope(context.Background(), "fake", "medium")
	if err != nil || *next.Memory.HostRAMBytes != 4096 || next.Placement[0] == "changed" {
		t.Fatalf("snapshot aliased launch: %+v %v", next, err)
	}
}

// The context fence changes state after compiler I/O and before the final
// owned-run check, without adding a production reader or compiler hook.
type servingFenceContext struct {
	context.Context
	calls           int
	afterValidation func()
}

func (c *servingFenceContext) Err() error {
	c.calls++
	if c.calls == 2 {
		c.afterValidation()
	}
	return c.Context.Err()
}

func TestCurrentServingEnvelopeFencesReplacementDuringValidation(t *testing.T) {
	test := func(name string, replace func(*engineState)) {
		t.Run(name, func(t *testing.T) {
			f := ownedServingFixture(t)
			st := settingsState(t, f.ex)
			ctx := &servingFenceContext{Context: context.Background(), afterValidation: func() { st.mu.Lock(); defer st.mu.Unlock(); replace(st) }}
			_, _, err := f.ex.currentServingEnvelope(ctx, "fake", "medium")
			var refusal *ServingIdentityRefusal
			if !errors.As(err, &refusal) {
				t.Fatalf("stale validator returned identity: %v", err)
			}
		})
	}
	test("replacement run", func(st *engineState) { st.runID = newEngineRunID() })
	test("replacement config with identical values", func(st *engineState) { st.plat.Runtime.Serving = st.plat.Runtime.Serving.Clone() })
}

func TestServingSymlinkRetargetWithdrawsIdentity(t *testing.T) {
	f := newLlamaServingFixture(t)
	path := filepath.Join(f.root, "medium.gguf")
	original := filepath.Join(f.root, "original.gguf")
	replacement := filepath.Join(f.root, "replacement.gguf")
	if err := os.Rename(path, original); err != nil {
		t.Fatal(err)
	}
	writeServingFixture(t, replacement, []byte("synthetic medium model"))
	if err := os.Symlink(original, path); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.ex.Stop("fake"); err != nil {
			t.Error(err)
		}
	})
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(replacement, path); err != nil {
		t.Fatal(err)
	}
	f.ex.refreshServingIdentity(context.Background())
	id := f.ex.modelIdentity().Engines["fake"]
	if len(id.ModelDigests) != 0 || id.Serving != nil {
		t.Fatal("retargeted artifact retained qualification identity")
	}
}

func TestServingUnstableSamplingRejectsBinding(t *testing.T) {
	path := filepath.Join(t.TempDir(), "changing-model")
	writeServingFixture(t, path, bytes.Repeat([]byte("x"), 4*servingSampleSize))
	file, err := os.OpenFile(path, os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		value := byte(0)
		for {
			select {
			case <-stop:
				return
			default:
			}
			if _, err := file.WriteAt([]byte{value}, 0); err != nil {
				return
			}
			value++
		}
	}()
	defer func() { close(stop); wg.Wait() }()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := servingFileBinding(path); err != nil {
			return
		}
	}
	t.Fatal("sampling never refused a concurrently changing artifact")
}

func TestServingRelocationWithRenewedBindingPreservesContent(t *testing.T) {
	f := newLlamaServingFixture(t)
	before := approveLlamaServing(t, f)
	newPath := filepath.Join(t.TempDir(), "relocated.gguf")
	data, err := os.ReadFile(filepath.Join(f.root, "medium.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	writeServingFixture(t, newPath, data)
	source, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	writeServingFixture(t, f.source, []byte(strings.Replace(string(source), "model = medium.gguf", "model = "+newPath, 1)))
	// The renewed request deliberately has no old binding or source approval.
	after := approveLlamaServing(t, f)
	if before.Serving.Configured[0].ContentDigest != after.Serving.Configured[0].ContentDigest {
		t.Fatal("relocation changed content identity")
	}
	if before.Serving.Configured[0].NativeSourceDigest == after.Serving.Configured[0].NativeSourceDigest {
		t.Fatal("relocation omitted from local launch/config identity")
	}
}

func TestServingIdentityIgnoresNativeInventoryDigests(t *testing.T) {
	f := ownedServingFixture(t)
	before := f.ex.modelIdentity().Engines["fake"]
	observed := modelInventoryObservation(t, f.ex, "fake")
	observed.models = map[string]string{"medium": strings.Repeat("f", 64), "undeclared": strings.Repeat("e", 64)}
	f.ex.storeInventory(nil, map[string]observedDigests{"fake": observed}, f.ex.sweeps.Add(1))
	after := f.ex.modelIdentity().Engines["fake"]
	if !reflect.DeepEqual(after.ModelDigests, before.ModelDigests) {
		t.Fatalf("inventory replaced declaration identity: %v", after.ModelDigests)
	}
	*after.Serving["medium"].Memory.HostRAMBytes = 1
	after.Serving["medium"].Placement[0] = "changed"
	if *f.ex.modelIdentity().Engines["fake"].Serving["medium"].Memory.HostRAMBytes != 4096 {
		t.Fatal("published identity aliases launch")
	}
}

func TestCurrentServingEnvelopeRefusals(t *testing.T) {
	test := func(name string, edit func(*engineState), model string) {
		t.Run(name, func(t *testing.T) {
			f := ownedServingFixture(t)
			st := settingsState(t, f.ex)
			st.mu.Lock()
			edit(st)
			st.mu.Unlock()
			_, _, err := f.ex.currentServingEnvelope(context.Background(), "fake", model)
			st.mu.Lock()
			st.adopted = false
			st.mu.Unlock()
			var refusal *ServingIdentityRefusal
			if !errors.As(err, &refusal) {
				t.Fatalf("want typed refusal, got %v", err)
			}
		})
	}
	test("undeclared model", func(*engineState) {}, "other")
	test("adopted run", func(st *engineState) { st.adopted = true }, "medium")
	test("missing run", func(st *engineState) { st.runID = "" }, "medium")
	test("missing preparation", func(st *engineState) { st.plat.Runtime.servingLaunch = nil }, "medium")
	test("compiler mismatch", func(st *engineState) {
		st.plat.Runtime.Serving = st.plat.Runtime.Serving.Clone()
		st.plat.Runtime.Serving.Models[0].MaxOutputTokens++
	}, "medium")
	test("latched drift", func(st *engineState) { st.plat.Runtime.servingDrift = "withdrawn" }, "medium")
}

func TestServingRunSnapshotRejectsReplacement(t *testing.T) {
	f := ownedServingFixture(t)
	st := settingsState(t, f.ex)
	s, err := snapshotServingRun(st)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateServingSnapshot(context.Background(), st, s); err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	oldRun := st.runID
	st.runID = newEngineRunID()
	if s.current(st) {
		t.Fatal("old validation accepted replacement run")
	}
	st.runID = oldRun
	st.plat.Runtime.Serving = st.plat.Runtime.Serving.Clone()
	st.plat.Runtime.Serving.Models[0].MaxOutputTokens++
	if s.current(st) {
		t.Fatal("old validation accepted replacement config")
	}
}

func TestServingDriftWithdrawsOnFullSweepAndLatches(t *testing.T) {
	f := ownedServingFixture(t)
	st := settingsState(t, f.ex)
	ownedPort := st.port
	server := attachLoadedLlamaFake(t, f, "medium")
	server.loaded = `{"data":[]}`
	path := filepath.Join(f.root, "medium.gguf")
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	changed := bytes.Clone(data)
	changed[0] ^= 1
	writeServingFixture(t, path, changed)
	if err := os.Chtimes(path, before.ModTime(), before.ModTime()); err != nil {
		t.Fatal(err)
	}
	f.ex.sweepLoaded(context.Background(), nil, false)
	if len(f.ex.modelIdentity().Engines["fake"].ModelDigests) != 2 {
		t.Fatal("partial sweep validated artifacts")
	}
	f.ex.sweepLoaded(context.Background(), nil, true)
	assertWithdrawn := func() {
		t.Helper()
		id := f.ex.modelIdentity().Engines["fake"]
		if len(id.ModelDigests) != 0 || id.Serving != nil {
			t.Fatalf("drift retained identity: %+v", id)
		}
	}
	assertWithdrawn()
	writeServingFixture(t, path, data)
	if err := os.Chtimes(path, before.ModTime(), before.ModTime()); err != nil {
		t.Fatal(err)
	}
	f.ex.sweepLoaded(context.Background(), nil, true)
	assertWithdrawn()
	readback, err := f.ex.LaunchSettings("fake")
	if err != nil || readback.ServingReadback.Status != "mismatch" || len(readback.ServingReadback.LaunchValidated) != 0 {
		t.Fatalf("missing latched mismatch readback: %+v %v", readback.ServingReadback, err)
	}
	// The HTTP observation fixture owns a separate loopback listener.
	st.mu.Lock()
	st.port = ownedPort
	st.mu.Unlock()
	if err := f.ex.Stop("fake"); err != nil {
		t.Fatal(err)
	}
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	if len(f.ex.modelIdentity().Engines["fake"].Serving) != 2 {
		t.Fatal("successful restart did not restore validated identity")
	}
}

func TestServingSparseModelNeverFullyHashed(t *testing.T) {
	f := newLlamaServingFixture(t)
	path := filepath.Join(f.root, "medium.gguf")
	file, err := os.OpenFile(path, os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := file.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			t.Error(err)
		}
	})
	prepareSparseFixture(t, file)
	err = file.Truncate(80 << 30)
	if err != nil {
		file.Close()
		t.Fatalf("extend sparse fixture: %v", err)
	}
	checkSparseFixtureAllocation(t, file, 80<<30)
	closeErr := file.Close()
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	finished := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			if ctx.Err() == context.DeadlineExceeded {
				panic("sparse preview/start/poll exceeded 10 seconds")
			}
		case <-finished:
		}
	}()
	defer close(finished)
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	if err := f.ex.Start(ctx, "fake"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.ex.Stop("fake"); err != nil {
			t.Error(err)
		}
	})
	f.ex.sweepLoaded(ctx, nil, true)
	if ctx.Err() != nil {
		t.Fatal(ctx.Err())
	}
	fileHashes.mu.Lock()
	defer fileHashes.mu.Unlock()
	for _, model := range []string{"medium.gguf", "weak.gguf"} {
		path := filepath.Join(f.root, model)
		if _, ok := fileHashes.done[path]; ok {
			t.Fatal("model entered full-file hash cache")
		}
		for key := range fileHashes.inflight {
			if key.path == path {
				t.Fatal("model full hash in progress")
			}
		}
	}
}
