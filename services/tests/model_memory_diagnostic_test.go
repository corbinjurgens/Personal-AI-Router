// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/shirou/gopsutil/v4/process"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
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
		// Resume also starts the inert companion engine. Its existing /models
		// response reports unloaded status, so declare that observation for the
		// manager's node-wide effects fence and unload verification.
		otherPath := filepath.Join(jointAppDir(f.base), "engines", "llamacpp-custom.json")
		var other map[string]any
		modelSettingsDecode(t, readLiveDurableFile(t, otherPath), &other)
		actions, ok := other["actions"].(map[string]any)
		if !ok {
			t.Fatal("missing companion engine actions")
		}
		actions["loaded_models"] = map[string]any{
			"http":   map[string]any{"method": "GET", "path": "/models"},
			"result": map[string]any{"array": "data", "field": "id", "match": map[string]any{"field": "status.value", "in": []string{"loaded"}}},
		}
		jointWriteJSON(t, otherPath, other)
	})
	d := &modelDiagnosticProcessFixture{modelBufferProcessFixture: f, faults: faults}
	f.waitMemory(t, "17179869184")
	policy := f.policy(t)
	for i := range policy.Profiles {
		profile := &policy.Profiles[i]
		if profile.Key == "weak-model" {
			profile.ContextTokens, profile.MaxOutputTokens, profile.MaxConcurrent = 4096, 512, 1
			profile.MemoryBytes = 1048576
			// Saved operator declaration for the synthetic resident, not a
			// native qualification or a measured memory result.
			profile.Qualification = &nodepolicy.QualificationRef{ID: "diagnostic-fixture-resident", Revision: "1"}
		}
	}
	f.setPolicy(t, policy)
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

func (f *modelDiagnosticProcessFixture) review(t *testing.T, request settings.Request) (settings.Request, settings.Preview) {
	t.Helper()
	var preview settings.Preview
	f.call(t, "engine:preview-settings", request, &preview)
	if len(preview.Errors) != 0 || preview.Conflict != nil || preview.PreviewDigest == "" {
		t.Fatalf("preview refused: errors=%v conflict=%v", preview.Errors, preview.Conflict)
	}
	request.ModelChange.PreviewDigest = preview.PreviewDigest
	return request, preview
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
	reply := f.rpc(t, "engine:diagnostic-preview", modelconfig.DiagnosticPreviewRequest{Engine: "llamacpp", ModelID: "medium-model", ProfileKey: "medium-model"})
	if reply.Error != nil {
		inspection := f.inspect(t, "medium-model")
		t.Fatalf("diagnostic preview: code=%d message=%s configuration reason=%s", reply.Error.Code, reply.Error.Message, inspection.ConfigurationKeyReason)
	}
	if err := modelconfig.DecodeStrict(reply.Result, &preview); err != nil {
		t.Fatal(err)
	}
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
	return modelDiagnosticStartFromPreview(t, preview, func(request modelconfig.DiagnosticStartRequest) jsonrpc.Message {
		return f.rpc(t, "engine:diagnostic-start", request)
	})
}

func modelDiagnosticStartFromPreview(t *testing.T, preview modelconfig.DiagnosticPreview, start func(modelconfig.DiagnosticStartRequest) jsonrpc.Message) modelconfig.DiagnosticRun {
	t.Helper()
	request := modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: preview.DiagnosticPreviewRequest, PreviewID: preview.PreviewID, Consent: true}
	reply := start(request)
	if reply.Error != nil {
		t.Fatalf("engine:diagnostic-start: code=%d message=%s", reply.Error.Code, reply.Error.Message)
	}
	var run modelconfig.DiagnosticRun
	if err := modelconfig.DecodeStrict(reply.Result, &run); err != nil {
		t.Fatalf("public diagnostic acceptance contains invalid/private fields: %v", err)
	}
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
	disk, err := nodepolicy.Parse(readLiveDurableFile(t, filepath.Join(jointAppDir(f.base), nodepolicy.FileName)))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(disk, want) {
		t.Fatal("persisted pause/policy differs from the exact pre-run policy")
	}
}

func (f *modelDiagnosticProcessFixture) resident(t *testing.T, reason nodepolicy.ModelConfigurationReason) {
	t.Helper()
	last := ""
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
				state := string(entry.State) + "/" + string(entry.Reason)
				if state != last {
					t.Logf("resident observation: %s", state)
					last = state
				}
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
	before := f.snapshot(t)
	request := settings.Request{Engine: "llamacpp", RequestID: "diagnostic-resident-configuration", ExpectedRevision: before.Revision, Settings: before.Settings.Clone()}
	request.Settings.Serving = &engines.ServingConfig{Version: 1}
	for _, name := range []string{"medium", "weak"} {
		model, err := os.ReadFile(filepath.Join(f.root, name+".gguf"))
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(model)
		ram, vram := int64(1048576), int64(0)
		request.Settings.Serving.Models = append(request.Settings.Serving.Models, engines.ServingModel{
			ModelID: name + "-model", MaxOutputTokens: 512,
			Memory:    engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram},
			Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: hex.EncodeToString(digest[:]), Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}}},
		})
	}
	var preview settings.Preview
	f.call(t, "engine:preview-settings", request, &preview)
	if len(preview.Errors) != 0 || preview.Conflict != nil {
		t.Fatalf("resident declaration preview: errors=%v conflict=%v", preview.Errors, preview.Conflict)
	}
	// Apply sends back the declaration the preview returned, whose native source
	// digest reviewed it. Model loads stay the target's baseline.
	request.Settings.Serving = preview.Settings.Serving.Clone()
	request.ExpectedRevision = preview.Revision
	f.call(t, "engine:apply-settings", request, nil)
	f.settled(t, request.RequestID, "succeeded")
	// Bootstrap the same synthetic strict-qualification recipe as the NE/SR
	// fixture, bound to the real manager's launch and content identities.
	f.call(t, "node:set-availability", map[string]string{"state": "available"}, nil)
	var launch string
	identity := regexp.MustCompile(`engine identity engine=llamacpp run_id=(run-[0-9a-f]+) launch_digest=([0-9a-f]{64}) models_with_digest=2`)
	task1Eventually(t, "resident launch identity", 30*time.Second, func() bool {
		matches := identity.FindAllStringSubmatch(f.p.output.String(), -1)
		if len(matches) == 0 {
			return false
		}
		launch = matches[len(matches)-1][2]
		return true
	})
	snapshot := f.snapshot(t)
	if snapshot.Serving == nil {
		t.Fatal("resident serving evidence missing")
	}
	profile, ok := f.policy(t).ProfileFor("llamacpp", "weak-model")
	if !ok {
		t.Fatal("resident policy profile missing")
	}
	var qualifications []any
	for _, envelope := range snapshot.Serving.LaunchValidated {
		if envelope.ModelID == "weak-model" {
			qualifications = append(qualifications, map[string]any{
				"id": "diagnostic-fixture-resident", "revision": "1", "engine": "llamacpp", "model": envelope.ModelID,
				"modelDigests": []string{envelope.ContentDigest}, "launchDigests": []string{launch},
				"tokenizer": "synthetic-diagnostic-tokenizer", "chatTemplate": "synthetic-diagnostic-template",
				"contextTokens": envelope.PerSlotContextTokens, "maxOutputTokens": envelope.MaxOutputTokens,
				"requestOptions": profile.RequestOptions, "requestOptionsMode": profile.RequestOptionsMode,
				"routes":     []string{"/v1/chat/completions"},
				"shapes":     map[string]any{"tools": false, "structuredOutput": []string{}, "reasoningReplay": false},
				"accounting": map[string]any{"method": "utf8-bytes", "requestOverheadTokens": 8, "messageOverheadTokens": 4, "toolOverheadTokens": 0, "schemaOverheadTokens": 0},
				"evidence":   "Synthetic diagnostic resident wiring only; no native model, hardware or performance qualification.",
			})
		}
	}
	if len(qualifications) != 1 {
		t.Fatal("resident launch-validated envelope missing")
	}
	oldProxy := f.proxySnapshot(t).RunID
	jointWriteJSON(t, filepath.Join(jointAppDir(f.base), "local-qualification.json"), map[string]any{"version": 2, "qualifications": qualifications})
	modelDiagnosticKillWorker(t, f.proxyControl.image)
	waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(f.proxyControl.port)), 30*time.Second)
	f.proxyControl.ready(t, "llamacpp", f.targetFacadePort)
	task1Eventually(t, "qualification registry proxy replacement", 30*time.Second, func() bool {
		return f.proxySnapshot(t).RunID != oldProxy
	})
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
	// The pinned terminal ingress tests this node's actual admission decision.
	// Plaintext is an origin router, which can wait for other routing candidates.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("https://127.0.0.1:%d/v1/chat/completions", f.targetFacadePort), bytes.NewReader(mustMarshal(t, map[string]any{"model": "weak-model", "messages": []map[string]string{{"role": "user", "content": "synthetic admission probe"}}, "max_tokens": 1})))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(nodepolicy.AdmissionWaitHeader, "0")
	response, err := f.client.Do(request)
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
	last := ""
	task1Eventually(t, "public interrupted-run recovery", 240*time.Second, func() bool {
		reply := f.rpc(t, "engine:diagnostic-status", modelconfig.DiagnosticRunRef{RunID: id})
		if reply.Error != nil {
			return false // replacement is not yet published by the real owner
		}
		if err := modelconfig.DecodeStrict(reply.Result, &run); err != nil {
			t.Fatal(err)
		}
		state := run.Phase + "/" + run.Outcome + ": " + run.Reason
		if state != last {
			t.Logf("interrupted run recovery: %s", state)
			last = state
		}
		return run.RunID == id && run.Outcome == "failed" && run.Phase == "done"
	})
	if err := run.Validate(); err != nil || run.Reason != "interrupted" {
		t.Fatalf("recovered public run=%+v error=%v", run, err)
	}
	return run
}

// The broker owns the production public envelope. This fixture consumes its
// flattened run plus target node without admitting private manager authority.
func decodeModelDiagnosticPeerRun(raw json.RawMessage, target string) (modelconfig.DiagnosticRun, error) {
	var public struct {
		modelconfig.DiagnosticRun
		NodeID string `json:"nodeId,omitempty"`
	}
	if err := modelconfig.DecodeStrict(raw, &public); err != nil {
		return modelconfig.DiagnosticRun{}, err
	}
	if public.NodeID != target || target == "" {
		return modelconfig.DiagnosticRun{}, fmt.Errorf("peer diagnostic target: got %q, want %q", public.NodeID, target)
	}
	if err := public.DiagnosticRun.Validate(); err != nil {
		return modelconfig.DiagnosticRun{}, err
	}
	return public.DiagnosticRun, nil
}

func TestModelMemoryDiagnosticPeerRunDecode(t *testing.T) {
	const payload = `{"nodeId":"target","runId":"peer-run","engine":"llamacpp","modelId":"medium-model","profileKey":"medium-model","configurationKey":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","phase":"high","outcome":"running","startedAt":"2026-11-08T00:00:00Z","elapsedMs":123}`
	t.Run("accepts flattened public run and exact target", func(t *testing.T) {
		run, err := decodeModelDiagnosticPeerRun(json.RawMessage(payload), "target")
		if err != nil {
			t.Fatalf("decode public peer run: %v", err)
		}
		want := modelconfig.DiagnosticRun{
			RunID: "peer-run", Engine: "llamacpp", ModelID: "medium-model", ProfileKey: "medium-model",
			ConfigurationKey: strings.Repeat("a", 64), Phase: "high", Outcome: "running",
			StartedAt: "2026-11-08T00:00:00Z", ElapsedMS: 123,
		}
		if !reflect.DeepEqual(run, want) {
			t.Fatalf("public peer run=%+v, want %+v", run, want)
		}
	})
	reject := func(name, raw, target string) {
		t.Run(name, func(t *testing.T) {
			if _, err := decodeModelDiagnosticPeerRun(json.RawMessage(raw), target); err == nil {
				t.Fatal("accepted invalid public peer run")
			}
		})
	}
	withField := func(field string) string { return strings.TrimSuffix(payload, "}") + "," + field + "}" }
	reject("rejects unknown field", withField(`"unexpected":true`), "target")
	reject("rejects private token", withField(`"token":"private"`), "target")
	reject("rejects private marker", withField(`"marker":1`), "target")
	reject("rejects private observation", withField(`"observation":{}`), "target")
	reject("rejects duplicate target", withField(`"nodeId":"target"`), "target")
	reject("rejects case-folded duplicate target", withField(`"NodeID":"target"`), "target")
	reject("rejects wrong target", payload, "other-target")
	reject("rejects missing target", strings.Replace(payload, `"nodeId":"target",`, "", 1), "target")
	reject("rejects empty expected target", payload, "")
	reject("rejects invalid run", strings.Replace(payload, `"phase":"high"`, `"phase":"unknown"`, 1), "target")
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
		preview := f.preview(t)
		wantStart := modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: preview.DiagnosticPreviewRequest, PreviewID: preview.PreviewID, Consent: true}
		run := modelDiagnosticStartFromPreview(t, preview, func(request modelconfig.DiagnosticStartRequest) jsonrpc.Message {
			if request != wantStart {
				t.Fatalf("Start changed preview, target or consent: got=%+v want=%+v", request, wantStart)
			}
			return f.rpc(t, "engine:diagnostic-start", request)
		})
		run = f.terminal(t, run.RunID, "completed")
		stored := f.stored(t, run)
		if run.Result == nil || !reflect.DeepEqual(*stored.Measured, *run.Result) || stored.Measured.Low == nil || stored.Measured.High == nil || !stored.Measured.OverlayEligible(stored.Measured.DiagnosticIdentity, []string{"host"}) {
			t.Fatalf("complete native exercise/pool coverage missing: %+v", stored.Measured)
		}
		value := f.inspect(t, "medium-model")
		if value.Estimate.EstimateSource == nil || value.Estimate.EstimateSource.Kind != "measured" || !slices.ContainsFunc(value.MemoryProfiles, func(p modelconfig.MemoryProfile) bool {
			return p.Current && p.Measured != nil && p.Measured.RunID == run.RunID
		}) {
			again := f.inspect(t, "medium-model")
			t.Fatalf("ordinary inspection did not use the target-authenticated current measurement: status=%s inspectionReason=%s key=%s reason=%s source=%+v profiles=%+v; repeated status=%s inspectionReason=%s key=%s reason=%s source=%+v profiles=%+v", value.Status, value.Reason, value.ConfigurationKey, value.ConfigurationKeyReason, value.Estimate.EstimateSource, value.MemoryProfiles, again.Status, again.Reason, again.ConfigurationKey, again.ConfigurationKeyReason, again.Estimate.EstimateSource, again.MemoryProfiles)
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
		// Withdrawal is only established by a resolved read of the new
		// configuration; a busy, keyless inspection proves nothing here.
		if value.Status != "resolved" || value.ConfigurationKey == "" || value.ConfigurationKey == preview.ConfigurationKey {
			t.Fatalf("changed configuration was not inspected as resolved under a new key: status=%s reason=%s key=%q previewKey=%q", value.Status, value.Reason, value.ConfigurationKey, preview.ConfigurationKey)
		}
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
			peerCall := func(method string, params any) json.RawMessage {
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
				return reply.Result
			}
			var preview modelconfig.DiagnosticPreview
			if err := modelconfig.DecodeStrict(peerCall("engine:diagnostic-preview", request), &preview); err != nil {
				t.Fatalf("origin public diagnostic preview: %v", err)
			}
			if err := preview.Validate(); err != nil || preview.TotalRAMBytes != "17179869184" || preview.DiagnosticPreviewRequest != request {
				t.Fatalf("peer preview lost target hardware: %v", err)
			}
			peerRun := func(method string, params any) modelconfig.DiagnosticRun {
				t.Helper()
				run, err := decodeModelDiagnosticPeerRun(peerCall(method, params), request.NodeID)
				if err != nil {
					t.Fatalf("origin %s public diagnostic run: %v", method, err)
				}
				return run
			}
			run := peerRun("engine:diagnostic-start", modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: request, PreviewID: preview.PreviewID, Consent: true})
			if run.Engine != request.Engine || run.ModelID != request.ModelID || run.ProfileKey != request.ProfileKey {
				t.Fatalf("origin start accepted a different target profile: %+v", run)
			}
			f.phase(t, run.RunID, "high")
			f.effect(t, "high:start", 2)
			ref := modelconfig.DiagnosticRunRef{NodeID: request.NodeID, RunID: run.RunID}
			remote := peerRun("engine:diagnostic-status", ref)
			if remote.RunID != run.RunID || remote.Phase != "high" {
				t.Fatal("origin status did not describe the target run")
			}
			if outcome == "cancelled" {
				remote = peerRun("engine:diagnostic-cancel", ref)
				if remote.RunID != run.RunID || remote.Outcome != "cancelled" || remote.Phase != "done" {
					t.Fatalf("origin cancel did not join the exact target run: %+v", remote)
				}
			} else {
				f.fault(t, "high", false)
			}
			run = f.terminal(t, run.RunID, outcome)
			remote = peerRun("engine:diagnostic-status", ref)
			if remote.RunID != run.RunID || remote.Outcome != outcome || remote.Phase != "done" {
				t.Fatalf("origin lost the target's terminal result: %+v", remote)
			}
			remote = peerRun("engine:diagnostic-cancel", ref)
			if remote.RunID != run.RunID || remote.Outcome != outcome || remote.Phase != "done" {
				t.Fatalf("origin terminal cancel changed the exact target result: %+v", remote)
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
