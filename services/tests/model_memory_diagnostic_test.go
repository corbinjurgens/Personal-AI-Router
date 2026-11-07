// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/shirou/gopsutil/v4/process"
	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

const modelDiagnosticFixtureEnv = "NVPAIR_TEST_MODEL_DIAGNOSTIC_FAULTS"

// Opt-in extension of the existing fixed-image llama router. Files in the
// fixture's private directory hold native HTTP operations, never product RPCs.
// Public acceptance below always talks to real broker/manager/proxy owners.
type modelDiagnosticFixtureRuntime struct {
	faults    string
	mu        sync.Mutex
	highCount int
	highReady chan struct{}
}

func (f *modelDiagnosticFixtureRuntime) wait(ctx context.Context, fault string) error {
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		_, err := os.Stat(filepath.Join(f.faults, fault))
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		if err != nil {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func (f *modelDiagnosticFixtureRuntime) serve(w http.ResponseWriter, r *http.Request, models map[string]*modelBufferFixtureModel, mu *sync.Mutex, load, unload func(string)) bool {
	if f.faults == "" || r.Method != http.MethodPost || !slices.Contains([]string{"/models/load", "/models/unload", "/v1/chat/completions/input_tokens", "/v1/chat/completions"}, r.URL.Path) {
		return false
	}
	w.Header().Set("Content-Type", "application/json")
	var body struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		MaxTokens   int  `json:"max_tokens"`
		Temperature int  `json:"temperature"`
		Stream      bool `json:"stream"`
		CachePrompt bool `json:"cache_prompt"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, (8<<20)+4096)).Decode(&body); err != nil {
		http.Error(w, "invalid synthetic request", http.StatusBadRequest)
		return true
	}
	mu.Lock()
	model := models[body.Model]
	loaded := model != nil && model.status == "loaded"
	slots := 0
	if model != nil {
		slots = model.slots
	}
	mu.Unlock()
	if model == nil {
		http.Error(w, "unknown synthetic model", http.StatusNotFound)
		return true
	}
	if r.URL.Path == "/models/load" || r.URL.Path == "/models/unload" {
		operation := strings.TrimPrefix(r.URL.Path, "/models/")
		if err := modelBufferFixtureEffect(operation + ":" + body.Model); err != nil {
			http.Error(w, "effect recording failed", http.StatusInternalServerError)
			return true
		}
		if operation == "load" {
			mu.Lock()
			model.status = "loading"
			mu.Unlock()
		}
		if err := f.wait(r.Context(), operation); err != nil {
			return true
		}
		mu.Lock()
		if operation == "load" {
			load(body.Model)
		} else {
			unload(body.Model)
		}
		mu.Unlock()
		if err := json.NewEncoder(w).Encode(map[string]bool{"success": true}); err != nil {
			return true
		}
		return true
	}
	// Match model_diagnostic_exercise.go's pinned native chat contract, including
	// template-inclusive counting, identity probes, disabled autoload/cache and
	// real SSE token/finish/timings events. No prompt text is recorded.
	if !loaded || r.URL.Query().Get("autoload") != "false" || r.URL.Query().Get("model") != body.Model || r.Header.Get("X-NVPAIR-Engine-Identity-Probe") != "1" || len(body.Messages) != 1 || body.Messages[0].Role != "user" || body.MaxTokens != 128 || body.Temperature != 0 || body.CachePrompt {
		http.Error(w, "synthetic native contract mismatch", http.StatusConflict)
		return true
	}
	tokens := len(body.Messages[0].Content)/4 + 8
	if r.URL.Path == "/v1/chat/completions/input_tokens" {
		if err := json.NewEncoder(w).Encode(map[string]any{"object": "response.input_tokens", "input_tokens": tokens}); err != nil {
			return true
		}
		return true
	}
	phase := "low"
	if tokens > 512 {
		phase = "high"
	}
	if err := modelBufferFixtureEffect(phase + ":start"); err != nil {
		http.Error(w, "effect recording failed", http.StatusInternalServerError)
		return true
	}
	if body.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "synthetic stream unavailable", http.StatusInternalServerError)
			return true
		}
		if _, err := io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"synthetic token\"},\"finish_reason\":null}]}\n\n"); err != nil {
			return true
		}
		flusher.Flush()
	}
	if phase == "high" {
		f.mu.Lock()
		if f.highReady == nil {
			f.highReady = make(chan struct{})
		}
		f.highCount++
		if f.highCount == slots {
			close(f.highReady)
		}
		ready := f.highReady
		f.mu.Unlock()
		select {
		case <-r.Context().Done():
			return true
		case <-ready:
		}
	}
	if err := f.wait(r.Context(), phase); err != nil {
		return true
	}
	// Preserve production sampling cadence. This synthetic native latency gives
	// low/high multiple fresh scanner samples and high a common decode window.
	timer := time.NewTimer(6 * time.Second)
	defer timer.Stop()
	select {
	case <-r.Context().Done():
		return true
	case <-timer.C:
	}
	choice := `"message":{"role":"assistant","content":"synthetic output"}`
	if body.Stream {
		choice = `"delta":{}`
	}
	frame := fmt.Sprintf("{\"choices\":[{%s,\"finish_reason\":\"length\"}],\"timings\":{\"prompt_n\":%d,\"cache_n\":0,\"prompt_ms\":200,\"prompt_per_second\":%d,\"predicted_n\":128,\"predicted_ms\":6000,\"predicted_per_second\":21}}", choice, tokens, tokens*5)
	if body.Stream {
		frame = "data: " + frame + "\n\ndata: [DONE]\n\n"
	}
	if _, err := io.WriteString(w, frame); err != nil {
		return true
	}
	if err := modelBufferFixtureEffect(phase + ":end"); err != nil {
		return true
	}
	return true
}

type modelDiagnosticProcessFixture struct {
	*modelBufferProcessFixture
	faults string
}

func newModelDiagnosticProcessFixture(t *testing.T, blocked string) *modelDiagnosticProcessFixture {
	t.Helper()
	faults := t.TempDir()
	if blocked != "" {
		modelSettingsWrite(t, filepath.Join(faults, blocked), nil)
	}
	f := newModelBufferProcessFixture(t, func(f *modelSettingsProcessFixture) {
		f.pinned(t)
		f.enableMemory(t, modelMemoryPool("ram", "system", "17179869184", "8589934592"))
		requireOwnerFirewallRules(t, readOwnerFirewallSnapshot(t), []string{f.image, brokerBin, engineMgrBin, scannerBin, proxyBin, f.proxyControl.image})
		path := filepath.Join(jointAppDir(f.base), "engines", "llamacpp.json")
		var manifest map[string]any
		modelSettingsDecode(t, readLiveDurableFile(t, path), &manifest)
		platforms, ok := manifest["platforms"].(map[string]any)
		if !ok {
			t.Fatal("missing diagnostic fixture platforms")
		}
		platform, ok := platforms[runtime.GOOS+"/"+runtime.GOARCH].(map[string]any)
		if !ok {
			t.Fatal("missing diagnostic fixture platform")
		}
		for _, owner := range []map[string]any{manifest, platform} {
			rt, ok := owner["runtime"].(map[string]any)
			if !ok {
				t.Fatal("missing diagnostic runtime")
			}
			env, ok := rt["env"].(map[string]any)
			if !ok {
				t.Fatal("missing diagnostic environment")
			}
			env[modelDiagnosticFixtureEnv] = faults
		}
		jointWriteJSON(t, path, manifest)
	})
	d := &modelDiagnosticProcessFixture{modelBufferProcessFixture: f, faults: faults}
	f.waitMemory(t, "17179869184")
	// Release barriers before the already-registered process cleanup runs.
	t.Cleanup(func() {
		for _, name := range []string{"load", "low", "high", "unload"} {
			d.fault(t, name, false)
		}
	})
	return d
}

func (f *modelDiagnosticProcessFixture) fault(t *testing.T, name string, hold bool) {
	t.Helper()
	path := filepath.Join(f.faults, name)
	if hold {
		modelSettingsWrite(t, path, nil)
	} else if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		t.Fatal(err)
	}
}

// The ordinary fixture's 90-second reply bound is shorter than intentional
// cleanup. Keep its correlation map, but allow the public 200-second cancel
// envelope to return before declaring a transport failure.
func (f *modelDiagnosticProcessFixture) rpc(t *testing.T, method string, params any) jsonrpc.Message {
	t.Helper()
	id := f.begin(t, method, params)
	if f.replies == nil {
		f.replies = map[string]jsonrpc.Message{}
	}
	timer := time.NewTimer(240 * time.Second)
	defer timer.Stop()
	for {
		select {
		case frame, ok := <-f.p.msgs:
			if !ok {
				t.Fatalf("broker closed during %s", method)
			}
			if frame.ID != nil && frame.Method == "" {
				if string(*frame.ID) == id {
					f.assertEngineImages(t)
					return frame
				}
				f.replies[string(*frame.ID)] = frame
			}
		case <-timer.C:
			t.Fatalf("public %s exceeded cancellation/cleanup envelope", method)
		}
	}
}

func (f *modelDiagnosticProcessFixture) call(t *testing.T, method string, params, out any) {
	t.Helper()
	reply := f.rpc(t, method, params)
	if reply.Error != nil {
		t.Fatalf("%s: code=%d message=%s", method, reply.Error.Code, reply.Error.Message)
	}
	if out != nil {
		if strings.HasPrefix(method, "engine:diagnostic-") {
			if err := modelconfig.DecodeStrict(reply.Result, out); err != nil {
				t.Fatalf("public diagnostic payload contains invalid/private fields: %v", err)
			}
		} else {
			modelSettingsDecode(t, reply.Result, out)
		}
	}
}

func (f *modelDiagnosticProcessFixture) preview(t *testing.T) modelconfig.DiagnosticPreview {
	t.Helper()
	var preview modelconfig.DiagnosticPreview
	f.call(t, "engine:diagnostic-preview", modelconfig.DiagnosticPreviewRequest{Engine: "llamacpp", ModelID: "medium-model", ProfileKey: "medium-model"}, &preview)
	if err := preview.Validate(); err != nil {
		t.Fatalf("public preview: %v", err)
	}
	if preview.ConfigurationKey == "" || preview.CPUModel == "" || preview.TotalRAMBytes != "17179869184" || !slices.Contains(preview.DrainEngines, "llamacpp") {
		t.Fatalf("preview lost target identity or node drain: %+v", preview)
	}
	return preview
}

func (f *modelDiagnosticProcessFixture) startRun(t *testing.T) modelconfig.DiagnosticRun {
	t.Helper()
	preview := f.preview(t)
	var run modelconfig.DiagnosticRun
	f.call(t, "engine:diagnostic-start", modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: preview.DiagnosticPreviewRequest, PreviewID: preview.PreviewID, Consent: true}, &run)
	if err := run.Validate(); err != nil || run.RunID == "" || run.Outcome != "running" {
		t.Fatalf("public diagnostic acceptance=%+v error=%v", run, err)
	}
	return run
}

func (f *modelDiagnosticProcessFixture) status(t *testing.T, id string) modelconfig.DiagnosticRun {
	t.Helper()
	var run modelconfig.DiagnosticRun
	f.call(t, "engine:diagnostic-status", modelconfig.DiagnosticRunRef{RunID: id}, &run)
	if err := run.Validate(); err != nil || run.RunID != id {
		t.Fatalf("public diagnostic status=%+v error=%v", run, err)
	}
	return run
}

func (f *modelDiagnosticProcessFixture) phase(t *testing.T, id, phase string) {
	t.Helper()
	task1Eventually(t, "public diagnostic phase "+phase, 90*time.Second, func() bool {
		run := f.status(t, id)
		if run.Outcome != "running" {
			t.Fatalf("diagnostic ended before %s: %+v", phase, run)
		}
		return run.Phase == phase
	})
}

func (f *modelDiagnosticProcessFixture) terminal(t *testing.T, id, outcome string) modelconfig.DiagnosticRun {
	t.Helper()
	var run modelconfig.DiagnosticRun
	task1Eventually(t, "public diagnostic terminal "+outcome, 240*time.Second, func() bool {
		run = f.status(t, id)
		return run.Outcome != "running" && (run.Phase == "done" || run.Outcome == "cleanup-failed")
	})
	if run.Outcome != outcome {
		t.Fatalf("terminal=%+v, want %s", run, outcome)
	}
	return run
}

func (f *modelDiagnosticProcessFixture) effects(t *testing.T) []string {
	t.Helper()
	data, err := os.ReadFile(f.modelBufferProcessFixture.effects)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	return strings.Fields(string(data))
}

func (f *modelDiagnosticProcessFixture) effect(t *testing.T, event string, count int) {
	t.Helper()
	task1Eventually(t, "synthetic native barrier "+event, 90*time.Second, func() bool {
		return modelDiagnosticCount(f.effects(t), event) == count
	})
}

func modelDiagnosticCount(values []string, want string) int {
	count := 0
	for _, value := range values {
		if value == want {
			count++
		}
	}
	return count
}

func (f *modelDiagnosticProcessFixture) savedPause(t *testing.T, want nodepolicy.Policy) {
	t.Helper()
	got := f.policy(t)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("diagnostic changed saved policy: got=%+v want=%+v", got, want)
	}
	var disk nodepolicy.Policy
	modelSettingsDecode(t, readLiveDurableFile(t, filepath.Join(jointAppDir(f.base), nodepolicy.FileName)), &disk)
	if !reflect.DeepEqual(disk, want) {
		t.Fatal("persisted pause/policy differs from the exact pre-run policy")
	}
}

func (f *modelDiagnosticProcessFixture) resident(t *testing.T, reason nodepolicy.ModelConfigurationReason) {
	t.Helper()
	task1Eventually(t, "resident owner settles "+string(reason), 90*time.Second, func() bool {
		var view struct {
			Policy             nodepolicy.Policy              `json:"policy"`
			ModelConfiguration *nodepolicy.ModelConfiguration `json:"modelConfiguration"`
		}
		f.call(t, "policy:get", nil, &view)
		if view.ModelConfiguration == nil || view.ModelConfiguration.PolicyRevision != view.Policy.Revision {
			return false
		}
		if err := view.ModelConfiguration.Validate(); err != nil {
			t.Fatal(err)
		}
		for _, entry := range view.ModelConfiguration.Models {
			if entry.Engine == "llamacpp" && entry.Model == "weak-model" {
				return entry.DesiredResident && entry.Reason == reason && (reason != "" || entry.State == nodepolicy.ModelStateSatisfied && entry.ObservedLoaded != nil && *entry.ObservedLoaded)
			}
		}
		return false
	})
}

func (f *modelDiagnosticProcessFixture) closure(t *testing.T, before nodepolicy.Policy, running bool, loaded []string) {
	t.Helper()
	// Wait for the public resident result before treating manager/proxy closure
	// as settled. Paused residents remain blocked and must not silently reload.
	if len(before.ResidentModels) != 0 {
		if before.Availability == nodepolicy.Paused {
			f.resident(t, nodepolicy.ModelReasonPaused)
		} else {
			f.resident(t, "")
		}
	}
	task1Eventually(t, "router state restored", 90*time.Second, func() bool { return f.snapshot(t).Running == running })
	var inventory neModels
	task1Eventually(t, "public loaded-model closure", 90*time.Second, func() bool {
		f.call(t, "engine:models", nil, &inventory)
		return slices.Equal(inventory.Loaded["llamacpp"], loaded)
	})
	f.engineProcessFixture.mu.Lock()
	identities := slices.Clone(f.engineIdentities["llamacpp"])
	f.engineProcessFixture.mu.Unlock()
	for i, identity := range identities {
		if running && i == len(identities)-1 {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			stopped, err := f.processStopped(ctx, identity)
			cancel()
			if err != nil || stopped {
				t.Fatalf("pre-existing router was not retained: stopped=%t error=%v", stopped, err)
			}
		} else {
			f.assertProcessStopped(t, identity)
		}
	}
	if running && len(identities) == 0 {
		t.Fatal("running manager snapshot lacks an owned router lifetime")
	}
	if !running {
		assertInteropListenerClosed(t, f.targetPort)
	}
	var checkpoint struct {
		PendingDiagnostic json.RawMessage `json:"pendingDiagnostic"`
	}
	modelSettingsDecode(t, readLiveDurableFile(t, filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json")), &checkpoint)
	if len(checkpoint.PendingDiagnostic) != 0 && !bytes.Equal(checkpoint.PendingDiagnostic, []byte("null")) {
		t.Fatal("terminal run released before durable checkpoint clearing")
	}
	f.savedPause(t, before)
	f.proxySnapshot(t)
	// A fresh real preview must acquire settings/diagnostic admission again.
	// This observes the public product, not an injected release acknowledgement.
	f.preview(t)
}

func (f *modelDiagnosticProcessFixture) stored(t *testing.T, run modelconfig.DiagnosticRun) modelconfig.MemoryProfile {
	t.Helper()
	var rows []struct {
		Engine     string                    `json:"engine"`
		ModelID    string                    `json:"modelId"`
		ProfileKey string                    `json:"profileKey"`
		Profile    modelconfig.MemoryProfile `json:"profile"`
	}
	modelSettingsDecode(t, readLiveDurableFile(t, modelMemoryStorePath(f.modelSettingsProcessFixture)), &rows)
	for _, row := range rows {
		if row.Profile.Measured == nil || row.Profile.Measured.RunID != run.RunID {
			continue
		}
		if err := row.Profile.Validate(); err != nil {
			t.Fatal(err)
		}
		if row.Engine != run.Engine || row.ModelID != run.ModelID || row.ProfileKey != run.ProfileKey || row.Profile.Current || row.Profile.Measured.Outcome != run.Outcome {
			t.Fatalf("stored scope/outcome differs from producing public run: %+v", row)
		}
		return row.Profile
	}
	t.Fatal("public producing run was not stored on target")
	return modelconfig.MemoryProfile{}
}

func (f *modelDiagnosticProcessFixture) availableResident(t *testing.T) nodepolicy.Policy {
	t.Helper()
	request := f.request(t, "diagnostic-resident-configuration", nil)
	request.ModelChange.ProfileKey, request.ModelChange.ModelID = "weak-model", "weak-model"
	request.ModelChange.Load.Set.Context.Tokens = 4096
	request.ModelChange.Load.Set.Slots = modelSettingsInt(1)
	request, _ = f.review(t, request)
	f.call(t, "engine:apply-settings", request, nil)
	f.settled(t, request.RequestID, "succeeded")
	policy := f.policy(t)
	for i := range policy.Profiles {
		if policy.Profiles[i].Key == "weak-model" {
			policy.Profiles[i].ContextTokens = 4096
		}
	}
	policy.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "weak-model"}}
	f.setPolicy(t, policy)
	f.call(t, "node:set-availability", map[string]string{"state": "available"}, nil)
	f.resident(t, "")
	return f.policy(t)
}

func (f *modelDiagnosticProcessFixture) peerLoadRefused(t *testing.T) {
	t.Helper()
	response, err := f.client.Post(fmt.Sprintf("https://127.0.0.1:%d/v1/models/load", f.ports.engineControl), "application/json", bytes.NewReader(mustMarshal(t, map[string]string{"engine": "llamacpp", "model": "weak-model"})))
	if err != nil {
		t.Fatal(err)
	}
	data, readErr := io.ReadAll(io.LimitReader(response.Body, 4096))
	if err := errors.Join(readErr, response.Body.Close()); err != nil {
		t.Fatal(err)
	}
	var failure struct {
		Code  string `json:"code"`
		Error string `json:"error"`
	}
	modelSettingsDecode(t, data, &failure)
	if response.StatusCode != http.StatusConflict || failure.Code != "operation_error" || !strings.Contains(failure.Error, "draining") {
		t.Fatalf("pinned public load: HTTP %d code=%s error=%s", response.StatusCode, failure.Code, failure.Error)
	}
}

func (f *modelDiagnosticProcessFixture) facadeAdmission(t *testing.T, want nodepolicy.RejectReason) {
	t.Helper()
	client := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{Proxy: nil}}
	defer client.CloseIdleConnections()
	request, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", f.targetFacadePort), bytes.NewReader(mustMarshal(t, map[string]any{"model": "weak-model", "messages": []map[string]string{{"role": "user", "content": "synthetic admission probe"}}, "max_tokens": 1})))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(nodepolicy.AdmissionWaitHeader, "0")
	response, err := client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	_, readErr := io.Copy(io.Discard, io.LimitReader(response.Body, 4096))
	if err := errors.Join(readErr, response.Body.Close()); err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusServiceUnavailable || response.Header.Get(nodepolicy.AdmissionHeader) != string(want) {
		t.Fatalf("real facade admission: HTTP %d reason=%s, want %s", response.StatusCode, response.Header.Get(nodepolicy.AdmissionHeader), want)
	}
}

func (f *modelDiagnosticProcessFixture) proxySnapshot(t *testing.T) nodepolicy.AdmissionSnapshotResult {
	t.Helper()
	var reply jsonrpc.Message
	observationWorkerHTTP(t, f.proxyControl.client, fmt.Sprintf("http://127.0.0.1:%d", f.proxyControl.port), "/control", jsonrpc.Message{JSONRPC: "2.0", Method: nodepolicy.MethodAdmissionSnapshot, Params: mustMarshal(t, nodepolicy.AdmissionSnapshotParams{})}, &reply)
	if reply.Error != nil {
		t.Fatalf("real proxy admission snapshot: %+v", reply.Error)
	}
	var snapshot nodepolicy.AdmissionSnapshotResult
	modelSettingsDecode(t, reply.Result, &snapshot)
	if snapshot.RunID == "" || snapshot.Active != 0 || snapshot.Queued != 0 {
		t.Fatalf("diagnostic left real serving admissions: %+v", snapshot)
	}
	return snapshot
}

func modelDiagnosticOwnedWorker(t *testing.T, image string) (*serviceProcessInventory, serviceProcessIdentity) {
	t.Helper()
	path, err := canonicalServicePath(image)
	if err != nil {
		t.Fatal(err)
	}
	inventory := &serviceProcessInventory{paths: []string{path}, known: map[int32]serviceProcessIdentity{}}
	owned, err := inventory.scan()
	if err != nil || len(owned) != 1 {
		t.Fatalf("serialized fixture worker inventory for %s: count=%d error=%v", filepath.Base(image), len(owned), err)
	}
	return inventory, owned[0]
}

func modelDiagnosticKillWorker(t *testing.T, image string) serviceProcessIdentity {
	t.Helper()
	inventory, identity := modelDiagnosticOwnedWorker(t, image)
	stopped, err := modelSettingsReplaceProcess(identity, func() error { return inventory.kill(identity) }, 15*time.Second)
	if err != nil || !stopped {
		t.Fatalf("owned worker termination: stopped=%t error=%v", stopped, err)
	}
	return identity
}

func modelDiagnosticSuspendManager(t *testing.T) func() {
	t.Helper()
	_, identity := modelDiagnosticOwnedWorker(t, engineMgrBin)
	proof, err := retainFixtureProcessLifetime(identity)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := proof.close(); err != nil {
			t.Error(err)
		}
	})
	manager, err := process.NewProcess(identity.PID)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	current, identityErr := serviceProcessIdentityAt(ctx, identity.PID)
	cancel()
	if identityErr != nil || current != identity {
		t.Fatalf("manager identity changed before suspend: %v", identityErr)
	}
	if err := manager.Suspend(); err != nil {
		t.Fatalf("suspend exact private manager lifetime: %v", err)
	}
	var once sync.Once
	resume := func() {
		once.Do(func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			stopped, err := proof.stopped(ctx)
			if err != nil {
				t.Errorf("inspect suspended manager lifetime: %v", err)
				return
			}
			if !stopped {
				current, err := serviceProcessIdentityAt(ctx, identity.PID)
				if err != nil || current != identity {
					t.Errorf("refuse to resume a replaced manager: %v", err)
					return
				}
				if err := manager.Resume(); err != nil {
					t.Errorf("resume exact private manager: %v", err)
				}
			}
		})
	}
	t.Cleanup(resume)
	return resume
}

func (f *modelDiagnosticProcessFixture) recovered(t *testing.T, id string) modelconfig.DiagnosticRun {
	t.Helper()
	var run modelconfig.DiagnosticRun
	task1Eventually(t, "public interrupted-run recovery", 240*time.Second, func() bool {
		reply := f.rpc(t, "engine:diagnostic-status", modelconfig.DiagnosticRunRef{RunID: id})
		if reply.Error != nil {
			return false // replacement is not yet published by the real owner
		}
		if err := modelconfig.DecodeStrict(reply.Result, &run); err != nil {
			t.Fatal(err)
		}
		return run.RunID == id && run.Outcome == "failed" && run.Phase == "done"
	})
	if err := run.Validate(); err != nil || run.Reason != "interrupted" {
		t.Fatalf("recovered public run=%+v error=%v", run, err)
	}
	return run
}

func TestModelMemoryDiagnosticProcess(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("White Windows serialized fixed-image diagnostic process acceptance only")
	}
	t.Run("configured pause completes stores and overlays exact current configuration", func(t *testing.T) {
		f := newModelDiagnosticProcessFixture(t, "")
		policy := f.policy(t)
		policy.Pause.OnActive, policy.Pause.DrainTimeoutSeconds = nodepolicy.FinishActive, 17
		policy.Pause.StopEngines, policy.Pause.UnloadModels = false, false
		policy.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "weak-model"}}
		before := f.setPolicy(t, policy)
		if before.Availability != nodepolicy.Paused || before.PauseMode != nodepolicy.PauseConfigured || f.snapshot(t).Running {
			t.Fatal("configured-pause precondition missing")
		}
		run := f.startRun(t)
		run = f.terminal(t, run.RunID, "completed")
		stored := f.stored(t, run)
		if run.Result == nil || !reflect.DeepEqual(*stored.Measured, *run.Result) || stored.Measured.Low == nil || stored.Measured.High == nil || !stored.Measured.OverlayEligible(stored.Measured.DiagnosticIdentity, []string{"host"}) {
			t.Fatalf("complete native exercise/pool coverage missing: %+v", stored.Measured)
		}
		value := f.inspect(t, "medium-model")
		if value.Estimate.EstimateSource == nil || value.Estimate.EstimateSource.Kind != "measured" || !slices.ContainsFunc(value.MemoryProfiles, func(p modelconfig.MemoryProfile) bool {
			return p.Current && p.Measured != nil && p.Measured.RunID == run.RunID
		}) {
			t.Fatal("ordinary inspection did not use the target-authenticated current measurement")
		}
		f.closure(t, before, false, nil)
		f.proxySnapshot(t)
		// A requested context change withdraws both currentness and overlay.
		request := f.request(t, "diagnostic-context-change", nil)
		request.ModelChange.Load.Set.Context.Tokens = 16384
		request, _ = f.review(t, request)
		f.call(t, "engine:apply-settings", request, nil)
		f.settled(t, request.RequestID, "succeeded")
		value = f.inspect(t, "medium-model")
		if slices.ContainsFunc(value.MemoryProfiles, func(p modelconfig.MemoryProfile) bool {
			return p.Current && p.Measured != nil && p.Measured.RunID == run.RunID
		}) || value.Estimate.EstimateSource != nil && value.Estimate.EstimateSource.Kind == "measured" {
			t.Fatal("configuration change retained the old measurement overlay")
		}
	})
	t.Run("Halt refuses both preview and stale consent without changing saved pause", func(t *testing.T) {
		f := newModelDiagnosticProcessFixture(t, "")
		preview := f.preview(t)
		f.call(t, "node:set-availability", map[string]string{"state": "paused", "pauseMode": "immediate"}, nil)
		before := f.policy(t)
		for _, request := range []struct {
			method string
			params any
		}{
			{"engine:diagnostic-preview", preview.DiagnosticPreviewRequest},
			{"engine:diagnostic-start", modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: preview.DiagnosticPreviewRequest, PreviewID: preview.PreviewID, Consent: true}},
		} {
			if reply := f.rpc(t, request.method, request.params); reply.Error == nil {
				t.Fatalf("Halt allowed %s", request.method)
			}
		}
		if f.snapshot(t).Running || len(f.effects(t)) != 0 {
			t.Fatal("Halt refusal started synthetic native work")
		}
		f.savedPause(t, before)
	})
	testCancel := func(name, blocked, phase, effect string, count int) {
		t.Run(name, func(t *testing.T) {
			f := newModelDiagnosticProcessFixture(t, blocked)
			before := f.policy(t)
			run := f.startRun(t)
			f.phase(t, run.RunID, phase)
			f.effect(t, effect, count)
			var cancelled modelconfig.DiagnosticRun
			f.call(t, "engine:diagnostic-cancel", modelconfig.DiagnosticRunRef{RunID: run.RunID}, &cancelled)
			run = f.terminal(t, run.RunID, "cancelled")
			if cancelled.RunID != run.RunID || cancelled.Outcome != "cancelled" {
				t.Fatal("cancel reply did not join the public producing run")
			}
			f.stored(t, run)
			f.fault(t, blocked, false)
			f.closure(t, before, false, nil)
		})
	}
	testCancel("cancel during native load joins and stops the owned router", "load", "loading", "load:medium-model", 1)
	testCancel("cancel during concurrent high exercise joins every request", "high", "high", "high:start", 2)
	t.Run("node reservation excludes public loads and resident reconciliation then restores resident", func(t *testing.T) {
		f := newModelDiagnosticProcessFixture(t, "high")
		before := f.availableResident(t)
		f.engineProcessFixture.mu.Lock()
		beforeOther := slices.Clone(f.engineIdentities["llamacpp-custom"])
		f.engineProcessFixture.mu.Unlock()
		run := f.startRun(t)
		f.phase(t, run.RunID, "high")
		f.effect(t, "high:start", 2)
		f.resident(t, nodepolicy.ModelReasonOperationBusy)
		effects := f.effects(t)
		if reply := f.rpc(t, "engine:load-model", map[string]string{"engine": "llamacpp", "model": "weak-model"}); reply.Error == nil {
			t.Fatal("public manual load bypassed the diagnostic drain")
		}
		f.peerLoadRefused(t)
		// Exercise the existing broker-internal on-demand wake input. Its real
		// lifecycle owner must not start another lifetime while drained.
		busyBefore := strings.Count(f.p.output.String(), "reserved by a memory measurement (busy)")
		observationWorkerHTTP(t, f.proxyControl.client, fmt.Sprintf("http://127.0.0.1:%d", f.proxyControl.port), "/emit", jsonrpc.Message{JSONRPC: "2.0", Method: engines.AddressMethod("llamacpp", nodepolicy.NotifyAdmissionWake), Params: mustMarshal(t, nodepolicy.WakeRequest{Engine: "llamacpp-custom"})}, nil)
		task1Eventually(t, "real internal wake receives manager reservation refusal", 30*time.Second, func() bool {
			return strings.Count(f.p.output.String(), "reserved by a memory measurement (busy)") > busyBefore
		})
		f.resident(t, nodepolicy.ModelReasonOperationBusy)
		f.facadeAdmission(t, nodepolicy.RejectDraining)
		f.engineProcessFixture.mu.Lock()
		afterOther := slices.Clone(f.engineIdentities["llamacpp-custom"])
		f.engineProcessFixture.mu.Unlock()
		if !slices.Equal(beforeOther, afterOther) {
			t.Fatal("internal wake started another engine lifetime during reservation")
		}
		if !slices.Equal(effects, f.effects(t)) {
			t.Fatal("competing public load or resident action reached synthetic native HTTP")
		}
		f.fault(t, "high", false)
		run = f.terminal(t, run.RunID, "completed")
		f.stored(t, run)
		f.closure(t, before, true, []string{"weak-model"})
		f.proxySnapshot(t)
	})
	t.Run("proxy replacement replays the node drain before opening admission", func(t *testing.T) {
		f := newModelDiagnosticProcessFixture(t, "high")
		before := f.availableResident(t)
		run := f.startRun(t)
		f.phase(t, run.RunID, "high")
		f.effect(t, "high:start", 2)
		oldRun := f.proxySnapshot(t).RunID
		modelDiagnosticKillWorker(t, f.proxyControl.image)
		waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(f.proxyControl.port)), 30*time.Second)
		f.proxyControl.ready(t, "llamacpp", f.targetFacadePort)
		var next nodepolicy.AdmissionSnapshotResult
		task1Eventually(t, "replacement proxy run", 60*time.Second, func() bool {
			next = f.proxySnapshot(t)
			return next.RunID != oldRun
		})
		f.facadeAdmission(t, nodepolicy.RejectDraining)
		f.resident(t, nodepolicy.ModelReasonOperationBusy)
		f.fault(t, "high", false)
		run = f.terminal(t, run.RunID, "completed")
		f.stored(t, run)
		f.closure(t, before, true, []string{"weak-model"})
		f.proxySnapshot(t)
	})
	t.Run("manager replacement recovers the durable interrupted run without resuming exercise", func(t *testing.T) {
		f := newModelDiagnosticProcessFixture(t, "high")
		before := f.policy(t)
		run := f.startRun(t)
		f.phase(t, run.RunID, "high")
		f.effect(t, "high:start", 2)
		f.replaceManager(t)
		run = f.recovered(t, run.RunID)
		f.stored(t, run)
		f.closure(t, before, false, nil)
		if modelDiagnosticCount(f.effects(t), "high:start") != 2 {
			t.Fatal("replacement manager resumed measurement")
		}
	})
	t.Run("broker restart recovers the durable interrupted run before activation", func(t *testing.T) {
		f := newModelDiagnosticProcessFixture(t, "high")
		before := f.policy(t)
		run := f.startRun(t)
		f.phase(t, run.RunID, "high")
		f.effect(t, "high:start", 2)
		f.crashBroker(t, newModelSettingsGate())
		f.replies = nil
		f.modelBufferProcessFixture.start(t)
		f.waitMemory(t, "17179869184")
		run = f.recovered(t, run.RunID)
		f.stored(t, run)
		f.closure(t, before, false, nil)
		if modelDiagnosticCount(f.effects(t), "high:start") != 2 {
			t.Fatal("broker restart resumed measurement")
		}
	})
	t.Run("cleanup transport timeout retains checkpoint and drains until real retry joins", func(t *testing.T) {
		f := newModelDiagnosticProcessFixture(t, "high")
		before := f.availableResident(t)
		run := f.startRun(t)
		f.phase(t, run.RunID, "high")
		f.effect(t, "high:start", 2)
		resume := modelDiagnosticSuspendManager(t)
		started := time.Now()
		var cancelled modelconfig.DiagnosticRun
		reply := f.rpc(t, "engine:diagnostic-cancel", modelconfig.DiagnosticRunRef{RunID: run.RunID})
		cancelled = f.status(t, run.RunID)
		if reply.Error == nil {
			t.Fatal("cleanup transport timeout returned a successful cancel acknowledgement")
		}
		if time.Since(started) < 5*time.Second || cancelled.Outcome != "cleanup-failed" {
			t.Fatalf("missing real cleanup timeout/retained failure: elapsed=%s run=%+v", time.Since(started), cancelled)
		}
		var checkpoint struct {
			PendingDiagnostic *struct {
				Run modelconfig.DiagnosticRun `json:"run"`
			} `json:"pendingDiagnostic"`
		}
		modelSettingsDecode(t, readLiveDurableFile(t, filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json")), &checkpoint)
		if checkpoint.PendingDiagnostic == nil || checkpoint.PendingDiagnostic.Run.RunID != run.RunID {
			t.Fatal("timeout dropped the durable cleanup obligation")
		}
		identity := f.engineIdentity(t, 1)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		stopped, stopErr := f.processStopped(ctx, identity)
		cancel()
		if stopErr != nil || stopped {
			t.Fatalf("timeout fault did not retain the live measured child: stopped=%t error=%v", stopped, stopErr)
		}
		f.facadeAdmission(t, nodepolicy.RejectDraining)
		f.resident(t, nodepolicy.ModelReasonOperationBusy)
		if reply := f.rpc(t, "engine:diagnostic-preview", modelconfig.DiagnosticPreviewRequest{Engine: "llamacpp", ModelID: "medium-model", ProfileKey: "medium-model"}); reply.Error == nil {
			t.Fatal("cleanup-failed released diagnostic admission without stopped proof")
		}
		f.savedPause(t, before)
		resume()
		f.fault(t, "high", false)
		f.call(t, "engine:diagnostic-cancel", modelconfig.DiagnosticRunRef{RunID: run.RunID}, &cancelled)
		run = f.terminal(t, run.RunID, "cancelled")
		f.stored(t, run)
		f.closure(t, before, true, []string{"weak-model"})
		f.proxySnapshot(t)
	})
	testPeer := func(name, outcome string) {
		t.Run(name, func(t *testing.T) {
			f := newModelDiagnosticProcessFixture(t, "high")
			before := f.policy(t)
			originBase := filepath.Dir(f.callerCluster)
			origin := modelMemoryOrigin(t, originBase, "model-settings-target", f.ports.engineControl)
			request := modelconfig.DiagnosticPreviewRequest{Engine: "llamacpp", NodeID: "model-settings-target", ModelID: "medium-model", ProfileKey: "medium-model"}
			peerCall := func(method string, params, out any) {
				t.Helper()
				// Reuse the origin's writer/reader and correlation, extending only
				// its ordinary ten-second wait for the diagnostic cleanup contract.
				id := origin.send(t, method, params, true)
				var reply jsonrpc.Message
				task1Eventually(t, "origin "+method, 240*time.Second, func() bool {
					origin.mu.Lock()
					defer origin.mu.Unlock()
					for _, frame := range origin.frames {
						if idEquals(frame.ID, id) {
							reply = frame
							return true
						}
					}
					if origin.readErr != nil {
						t.Fatalf("origin diagnostic transport: %v", origin.readErr)
					}
					return false
				})
				if reply.Error != nil {
					t.Fatalf("origin %s code=%d message=%s", method, reply.Error.Code, reply.Error.Message)
				}
				if err := modelconfig.DecodeStrict(reply.Result, out); err != nil {
					t.Fatalf("origin public diagnostic payload: %v", err)
				}
			}
			var preview modelconfig.DiagnosticPreview
			peerCall("engine:diagnostic-preview", request, &preview)
			if err := preview.Validate(); err != nil || preview.TotalRAMBytes != "17179869184" {
				t.Fatalf("peer preview lost target hardware: %v", err)
			}
			var run modelconfig.DiagnosticRun
			peerCall("engine:diagnostic-start", modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: request, PreviewID: preview.PreviewID, Consent: true}, &run)
			f.phase(t, run.RunID, "high")
			f.effect(t, "high:start", 2)
			var remote modelconfig.DiagnosticRun
			ref := modelconfig.DiagnosticRunRef{NodeID: request.NodeID, RunID: run.RunID}
			peerCall("engine:diagnostic-status", ref, &remote)
			if remote.RunID != run.RunID || remote.Phase != "high" {
				t.Fatal("origin status did not describe the target run")
			}
			if outcome == "cancelled" {
				peerCall("engine:diagnostic-cancel", ref, &remote)
			} else {
				f.fault(t, "high", false)
			}
			run = f.terminal(t, run.RunID, outcome)
			peerCall("engine:diagnostic-status", ref, &remote)
			if remote.RunID != run.RunID || remote.Outcome != outcome || remote.Phase != "done" {
				t.Fatalf("origin lost the target's terminal result: %+v", remote)
			}
			f.stored(t, run)
			originStore := filepath.Join(jointAppDir(originBase), "engine-bin", "model-memory-profiles.json")
			if _, err := os.Stat(originStore); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("peer run created an origin memory store or could not prove its absence: %v", err)
			}
			f.closure(t, before, false, nil)
		})
	}
	testPeer("completed peer diagnostic persists only on the measured target", "completed")
	testPeer("cancelled peer diagnostic persists only on the measured target", "cancelled")
}
