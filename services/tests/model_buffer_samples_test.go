// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"errors"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/modelconfig"
)

func modelBufferHistory(t *testing.T, f *modelBufferProcessFixture) []modelconfig.BufferObservation {
	t.Helper()
	data := readLiveDurableFile(t, f.history)
	var samples []struct {
		ModelIdentity string                        `json:"modelIdentity"`
		Observation   modelconfig.BufferObservation `json:"observation"`
	}
	if err := modelconfig.DecodeStrict(data, &samples); err != nil {
		t.Fatal(err)
	}
	if len(samples) > 128 || len(data) > 4<<20 {
		t.Fatal("history exceeded count/encoded budget")
	}
	var observations []modelconfig.BufferObservation
	for _, sample := range samples {
		if len(sample.ModelIdentity) != 64 {
			t.Fatal("nonopaque correlation")
		}
		if err := sample.Observation.Validate(); err != nil {
			t.Fatal(err)
		}
		if sample.Observation.Current {
			t.Fatal("disk history claimed currentness")
		}
		observations = append(observations, sample.Observation)
	}
	for _, private := range []string{f.root, f.image, "private-fixture-path", "medium-model", "weak-model", f.token, "NVPAIR_TEST_", "--models-preset"} {
		if strings.Contains(string(data), private) {
			t.Fatalf("private source material retained: %q", private)
		}
	}
	return observations
}

func TestModelBufferSamplesProcess(t *testing.T) {
	t.Run("real manager captures repeated buffers, interleaved children and ordinary wakeup", func(t *testing.T) {
		f := newModelBufferProcessFixture(t)
		before, policy := f.snapshot(t), f.policy(t)
		f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
		f.control(t, "medium-model", "initial")
		first := f.observed(t, "medium-model", "initial")
		if first.BuildAssurance != "recognized-self-reported-pin" || first.UpstreamPin != "7fe450e19305b828c199d602c23a8337aaa1f03b" || first.SourceIdentity == first.UpstreamPin {
			t.Fatal("native source digest became clean-source attestation")
		}
		var stored *modelconfig.BufferObservation
		for _, sample := range modelBufferHistory(t, f) {
			if sample.SampleID == first.SampleID {
				copy := sample
				stored = &copy
			}
		}
		if stored == nil || len(stored.Rows) != 7 {
			t.Fatal("complete typed checkpoint lost repeated model/SWA/recurrent/output occurrences")
		}
		models, kv, compute := 0, 0, 0
		for _, row := range stored.Rows {
			switch row.Component {
			case "model":
				models++
			case "kv":
				kv++
				if row.ReportedMiB == "0.00" && row.UpperBytes != "5243" {
					t.Fatal("rounded zero became exact zero")
				}
			case "compute":
				compute++
				if row.ReportedMiB != "0.50" {
					t.Fatal("successive reserve blocks were summed or stale")
				}
			}
			if row.DeviceBinding != "" || row.PoolIdentity != "" {
				t.Fatal("unavailable topology acquired physical identity")
			}
		}
		if models != 2 || kv != 2 || compute != 1 {
			t.Fatal("occurrence categories changed")
		}
		f.control(t, "weak-model", "initial")
		weak := f.observed(t, "weak-model", "initial")
		if weak.SampleID == first.SampleID || weak.PerSlotContext == first.PerSlotContext {
			t.Fatal("child name/port demultiplexing mixed baselines")
		}
		f.control(t, "medium-model", "sleep")
		task1Eventually(t, "sleep withdraws currentness", 10*time.Second, func() bool {
			value := f.inspect(t, "medium-model").LoadObservation
			return value != nil && !value.Current
		})
		f.control(t, "medium-model", "wakeup") // No synthetic second initial-ready log.
		wake := f.observed(t, "medium-model", "wakeup")
		if wake.SampleID == first.SampleID {
			t.Fatal("wakeup reused a prior load generation")
		}
		f.control(t, "medium-model", "exit") // Exact server_monitor::run emitter.
		task1Eventually(t, "run exit withdraws currentness", 10*time.Second, func() bool {
			value := f.inspect(t, "medium-model").LoadObservation
			return value != nil && !value.Current
		})
		f.control(t, "medium-model", "reload")
		reload := f.observed(t, "medium-model", "initial")
		if reload.SampleID == first.SampleID || reload.SampleID == wake.SampleID {
			t.Fatal("reload retained old rows/sample identity")
		}
		after := f.snapshot(t)
		if !after.Settings.Equal(before.Settings) || after.Revision != before.Revision || !reflect.DeepEqual(policy, f.policy(t)) {
			t.Fatal("observations changed finite settings or policy")
		}
		f.control(t, "medium-model", "oversized")
		task1Eventually(t, "dropped frame invalidates load", 10*time.Second, func() bool {
			value := f.inspect(t, "medium-model").LoadObservation
			return value != nil && !value.Current
		})
		identity := f.engineIdentity(t, 1)
		f.rpcOK(t, "engine:stop", map[string]string{"engine": "llamacpp"})
		f.assertProcessStopped(t, identity)
		data, err := os.ReadFile(f.effects)
		if err != nil {
			t.Fatal(err)
		}
		if string(data) != "start\n" {
			t.Fatalf("unexpected native/helper/inference operation: %s", data)
		}
	})
	t.Run("mixed draft roles do not become per-role totals", func(t *testing.T) {
		f := newModelBufferProcessFixture(t, func(f *modelSettingsProcessFixture) {
			data := readLiveDurableFile(t, f.preset)
			updated := bytes.Replace(data, []byte("[medium-model]\nmodel = medium.gguf\n"), []byte("[medium-model]\nmodel = medium.gguf\nmodel-draft = weak.gguf\n"), 1)
			if bytes.Equal(data, updated) {
				t.Fatal("draft fixture missing section")
			}
			modelSettingsWrite(t, f.preset, updated)
		})
		f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
		f.control(t, "medium-model", "initial")
		value := f.observed(t, "medium-model", "initial")
		if len(value.Artifacts) != 2 || value.Artifacts[1].Role != "draft" {
			t.Fatal("exact draft bindings missing")
		}
		for _, row := range value.Rows {
			if row.Role != "mixed" {
				t.Fatal("constructor order fabricated main/draft roles")
			}
		}
		for _, comparison := range value.Comparisons {
			if comparison.Status != "unknown" {
				t.Fatal("mixed child acquired role-specific comparison")
			}
		}
		f.control(t, "medium-model", "unsupported")
		task1Eventually(t, "unsupported child/router tuple stays Unknown", 20*time.Second, func() bool {
			value := f.inspect(t, "medium-model").LoadObservation
			return value != nil && value.Status == "unknown" && !value.Current && len(value.Rows) == 0 && value.Reason == "unsupported-build"
		})
		f.rpcOK(t, "engine:stop", map[string]string{"engine": "llamacpp"})
	})
}

func TestModelBufferSamplesProcessRestart(t *testing.T) {
	f := newModelBufferProcessFixture(t)
	f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	f.control(t, "medium-model", "initial")
	first := f.observed(t, "medium-model", "initial")
	f.rpcOK(t, "engine:stop", map[string]string{"engine": "llamacpp"})
	f.p.cleanup()
	f.start(t)
	recovered := f.inspect(t, "medium-model").LoadObservation
	if recovered == nil || recovered.Current || recovered.SampleID != first.SampleID {
		t.Fatal("real manager restart failed historical-only recovery")
	}
	f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	f.control(t, "medium-model", "initial")
	replacement := f.observed(t, "medium-model", "initial")
	if replacement.SampleID == first.SampleID {
		t.Fatal("manager/router replacement reused capture incarnation")
	}
	f.rpcOK(t, "engine:stop", map[string]string{"engine": "llamacpp"})
	f.p.cleanup()
	corrupt := []byte(`[{"invalid":"preserve corrupt history"}]`)
	if err := os.WriteFile(f.history, corrupt, 0o600); err != nil {
		t.Fatal(err)
	}
	f.start(t)
	value := f.inspect(t, "medium-model").LoadObservation
	if value == nil || value.Reason != "history-unavailable" || value.Current {
		t.Fatal("corrupt recovery fabricated an observation")
	}
	f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	f.control(t, "medium-model", "initial")
	task1Eventually(t, "corrupt store write-disabled alongside live load", 90*time.Second, func() bool {
		value := f.inspect(t, "medium-model").LoadObservation
		return value != nil && value.Current && value.Persistence == "history-unavailable"
	})
	if data := readLiveDurableFile(t, f.history); !bytes.Equal(data, corrupt) {
		t.Fatal("live load overwrote corrupt history")
	}
	f.rpcOK(t, "engine:stop", map[string]string{"engine": "llamacpp"})
}

func TestModelBufferSamplesProcessNoNativeFromInspection(t *testing.T) {
	f := newModelBufferProcessFixture(t)
	before, policy := f.snapshot(t), f.policy(t)
	journal := readLiveDurableFile(t, filepath.Join(f.base, "engine-settings-operations.json"))
	for range 3 {
		value := f.inspect(t, "medium-model")
		if value.LoadObservation == nil || value.LoadObservation.Status != "unknown" || value.LoadObservation.Current {
			t.Fatal("cold inspection fabricated a current observation")
		}
	}
	if _, err := os.Stat(f.effects); !os.IsNotExist(err) {
		t.Fatalf("offline Inspect initialized a native/helper process: %v", err)
	}
	if _, err := os.Stat(f.history); !os.IsNotExist(err) {
		t.Fatalf("Inspect created a sample: %v", err)
	}
	if after := f.snapshot(t); !after.Settings.Equal(before.Settings) || after.Revision != before.Revision || !reflect.DeepEqual(policy, f.policy(t)) || !bytes.Equal(journal, readLiveDurableFile(t, filepath.Join(f.base, "engine-settings-operations.json"))) {
		t.Fatal("Inspect changed authority")
	}
	// Produce a real checkpoint first, then inspect the recovered history with
	// the real manager's existing poller disabled. A trap owns the old engine
	// port so any projection-added native network query is directly observable.
	f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	f.control(t, "medium-model", "initial")
	first := f.observed(t, "medium-model", "initial")
	f.rpcOK(t, "engine:stop", map[string]string{"engine": "llamacpp"})
	f.p.cleanup()
	listener, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(f.targetPort)))
	if err != nil {
		t.Fatal(err)
	}
	var requests atomic.Int64
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}), ReadHeaderTimeout: time.Second}
	serveDone := make(chan error, 1)
	go func() { serveDone <- server.Serve(listener) }()
	t.Cleanup(func() {
		if err := server.Close(); err != nil {
			t.Error(err)
		}
		if err := <-serveDone; !errors.Is(err, http.ErrServerClosed) {
			t.Error(err)
		}
	})
	manager := startTask1Process(t, f.base, engineMgrBin, "--http-port", "0", "--control-port", "0", "--control-host", "127.0.0.1", "--loaded-poll-interval", "0")
	completeEngineManagerRecovery(t, func(method string, params any) jsonrpc.Message { return manager.call(t, method, params) })
	networkBefore := requests.Load()
	effectsBefore := readLiveDurableFile(t, f.effects)
	for range 3 {
		response := manager.call(t, "engine:inspect-model", modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "medium-model"})
		value, err := modelconfig.DecodeInspectResult(response.Result)
		if err != nil {
			t.Fatal(err)
		}
		if value.LoadObservation == nil || value.LoadObservation.Current || value.LoadObservation.SampleID != first.SampleID {
			t.Fatal("historical Inspect lost its detached checkpoint")
		}
	}
	if requests.Load() != networkBefore || !bytes.Equal(effectsBefore, readLiveDurableFile(t, f.effects)) {
		t.Fatal("Inspect added network/native/helper/inference effects")
	}
	// Permission-gated positive-helper incremental baselines are injected in
	// TestModelBufferInspectionHasNoNativeEffects, never real native libraries.
}
