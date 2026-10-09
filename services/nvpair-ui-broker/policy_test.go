// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/nodepolicy"
)

// policyCall is one request or notification a fake worker received.
type policyCall struct {
	method string
	params json.RawMessage
	notify bool
}

// fakeWorker records what the broker sends one worker and answers requests.
type fakeWorker struct {
	t      *testing.T
	codec  *Codec
	mu     sync.Mutex
	calls  []policyCall
	answer func(method string, params json.RawMessage) (any, string)
	signal chan struct{}
}

func (f *fakeWorker) serve() {
	for {
		msg, err := f.codec.Read()
		if err != nil {
			return
		}
		f.mu.Lock()
		f.calls = append(f.calls, policyCall{method: msg.Method, params: append(json.RawMessage(nil), msg.Params...), notify: !msg.IsRequest()})
		answer := f.answer
		f.mu.Unlock()
		select {
		case f.signal <- struct{}{}:
		default:
		}
		if !msg.IsRequest() {
			continue
		}
		var result any = map[string]bool{"ok": true}
		errText := ""
		if answer != nil {
			result, errText = answer(msg.Method, msg.Params)
		}
		if errText != "" {
			_ = f.codec.RespondError(msg.ID, -32000, errText)
		} else {
			_ = f.codec.Respond(msg.ID, result)
		}
	}
}

func (f *fakeWorker) snapshot() []policyCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]policyCall(nil), f.calls...)
}

func (f *fakeWorker) setAnswer(answer func(method string, params json.RawMessage) (any, string)) {
	f.mu.Lock()
	f.answer = answer
	f.mu.Unlock()
}

// methods lists the received method names, optionally only those with prefix.
func (f *fakeWorker) methods(prefix string) []string {
	var out []string
	for _, c := range f.snapshot() {
		if strings.HasPrefix(c.method, prefix) {
			out = append(out, c.method)
		}
	}
	return out
}

// waitFor polls until pred holds for the received calls.
func (f *fakeWorker) waitFor(what string, pred func([]policyCall) bool) []policyCall {
	f.t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		calls := f.snapshot()
		if pred(calls) {
			return calls
		}
		if time.Now().After(deadline) {
			f.t.Fatalf("timed out waiting for %s; received %v", what, f.methods(""))
		}
		select {
		case <-f.signal:
		case <-time.After(20 * time.Millisecond):
		}
	}
}

func newFakeWorkerPipe(t *testing.T) (*Peer, *fakeWorker) {
	t.Helper()
	client, server := net.Pipe()
	t.Cleanup(func() { _ = client.Close(); _ = server.Close() })
	peer := NewPeer(NewCodec(client))
	go peer.Serve(nil, nil)
	f := &fakeWorker{t: t, codec: NewCodec(server), signal: make(chan struct{}, 1)}
	go f.serve()
	return peer, f
}

// attachFakeProxy publishes a fake proxy process for every engine.
func attachFakeProxy(t *testing.T, b *Broker) (*proxyProcess, *fakeWorker) {
	t.Helper()
	peer, f := newFakeWorkerPipe(t)
	pp := &proxyProcess{peer: peer}
	for _, profile := range engineProxyProfiles {
		b.setEngineProxyHandle(profile, pp)
	}
	return pp, f
}

// attachFakeEngine publishes a fake engine-manager.
func attachFakeEngine(t *testing.T, b *Broker) *fakeWorker {
	t.Helper()
	peer, f := newFakeWorkerPipe(t)
	b.setEngineMgr(&rpcWorker{peer: peer})
	return f
}

// syncBuffer is the client side of a test broker's codec.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.Write(p)
}

func (s *syncBuffer) Read([]byte) (int, error) { return 0, nil }

func (s *syncBuffer) frames() []Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []Message
	scanner := bufio.NewScanner(bytes.NewReader(s.buf.Bytes()))
	scanner.Buffer(make([]byte, 64<<10), 1<<20)
	for scanner.Scan() {
		var m Message
		if json.Unmarshal(scanner.Bytes(), &m) == nil {
			out = append(out, m)
		}
	}
	return out
}

func (s *syncBuffer) waitFrame(t *testing.T, what string, pred func(Message) bool) Message {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		for _, m := range s.frames() {
			if pred(m) {
				return m
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
	return Message{}
}

func newPolicyBroker(t *testing.T) (*Broker, *syncBuffer) {
	t.Helper()
	out := &syncBuffer{}
	b := &Broker{codec: NewCodec(out), nodeID: "self-node", clusterDir: filepath.Join(t.TempDir(), "cluster")}
	return b, out
}

func writePolicyFile(t *testing.T, path string, p nodepolicy.Policy) {
	t.Helper()
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func clientRequest(id int, method string, params any) *Message {
	raw := json.RawMessage(`{}`)
	if params != nil {
		raw, _ = json.Marshal(params)
	}
	idRaw := json.RawMessage(strings.TrimSpace(string(mustJSON(id))))
	return &Message{JSONRPC: "2.0", ID: &idRaw, Method: method, Params: raw}
}

func mustJSON(v any) []byte {
	data, _ := json.Marshal(v)
	return data
}

func isResponse(id int) func(Message) bool {
	want := string(mustJSON(id))
	return func(m Message) bool { return m.ID != nil && string(*m.ID) == want }
}

func availabilityPushes(calls []policyCall) []nodepolicy.SetAvailabilityParams {
	var out []nodepolicy.SetAvailabilityParams
	for _, c := range calls {
		if c.method == nodepolicy.MethodSetAvailability {
			var p nodepolicy.SetAvailabilityParams
			_ = json.Unmarshal(c.params, &p)
			out = append(out, p)
		}
	}
	return out
}

func TestNodePolicyLoadDefaultsCorruptAndPersist(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, nodepolicy.FileName)

	b, _ := newPolicyBroker(t)
	b.loadNodePolicy(path)
	if got := b.localPolicy(); got.Availability != nodepolicy.Available || got.Policy.Admission.MaxResidentModels != 1 {
		t.Fatalf("missing file did not yield the default policy: %+v", got)
	}

	if err := os.WriteFile(path, []byte(`{"version":1,"availability":"sideways"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	b, _ = newPolicyBroker(t)
	b.loadNodePolicy(path)
	if got := b.localPolicy(); got.Availability != nodepolicy.Available {
		t.Fatalf("invalid file was not replaced by the default: %+v", got)
	}
	if bad, err := os.ReadFile(path + ".bad"); err != nil || !strings.Contains(string(bad), "sideways") {
		t.Fatalf("invalid file was not kept as .bad: %q %v", bad, err)
	}

	paused := nodepolicy.Default()
	paused.Availability = nodepolicy.Paused
	paused.Admission.MaxResidentModels = 3
	writePolicyFile(t, path, paused)
	b, _ = newPolicyBroker(t)
	b.loadNodePolicy(path)
	if got := b.localPolicy(); got.Availability != nodepolicy.Paused || got.Policy.Admission.MaxResidentModels != 3 {
		t.Fatalf("valid file not loaded: %+v", got)
	}

	next := nodepolicy.Default()
	next.Idle.UnloadAfterMinutes = 7
	if _, err := b.setLocalPolicy(mustJSON(next)); err != nil {
		t.Fatal(err)
	}
	reloaded, _ := newPolicyBroker(t)
	reloaded.loadNodePolicy(path)
	got := reloaded.localPolicy()
	if got.Policy.Idle.UnloadAfterMinutes != 7 {
		t.Fatalf("policy:set was not persisted: %+v", got.Policy.Idle)
	}
	if got.Availability != nodepolicy.Paused {
		t.Fatalf("policy:set changed availability to %q; only node:set-availability may", got.Availability)
	}
	if entries, _ := filepath.Glob(filepath.Join(dir, ".node-policy-*")); len(entries) != 0 {
		t.Fatalf("atomic write left temp files: %v", entries)
	}
}

func TestPolicySetValidatesPushesAndNotifies(t *testing.T) {
	b, out := newPolicyBroker(t)
	_, proxy := attachFakeProxy(t, b)

	bad := map[string]any{"version": 1, "profiles": []any{map[string]any{"name": "x", "engine": "nope", "model": "m"}}}
	b.handleMessage(clientRequest(1, methodPolicySet, map[string]any{"policy": bad}))
	resp := out.waitFrame(t, "invalid policy:set response", isResponse(1))
	if resp.Error == nil || resp.Error.Code != -32602 {
		t.Fatalf("invalid policy accepted: %+v", resp)
	}

	good := map[string]any{"version": 1, "availability": "paused", "idle": map[string]any{"unloadAfterMinutes": 5}}
	b.handleMessage(clientRequest(2, methodPolicySet, map[string]any{"policy": good}))
	resp = out.waitFrame(t, "policy:set response", isResponse(2))
	if resp.Error != nil {
		t.Fatalf("valid policy refused: %+v", resp.Error)
	}
	var result struct {
		Policy nodepolicy.Policy `json:"policy"`
	}
	_ = json.Unmarshal(resp.Result, &result)
	if result.Policy.Availability != nodepolicy.Available || result.Policy.Idle.UnloadAfterMinutes != 5 || !result.Policy.Idle.StartOnDemand {
		t.Fatalf("policy:set result = %+v; want availability kept and omitted fields defaulted", result.Policy)
	}
	out.waitFrame(t, "policy:changed", func(m Message) bool { return m.Method == notifyPolicyChanged })
	proxy.waitFor("node/set-policy push", func(calls []policyCall) bool {
		for _, c := range calls {
			if c.method == nodepolicy.MethodSetPolicy && strings.Contains(string(c.params), `"unloadAfterMinutes":5`) {
				return true
			}
		}
		return false
	})

	b.handleMessage(clientRequest(3, methodPolicyGet, nil))
	resp = out.waitFrame(t, "policy:get response", isResponse(3))
	var got policyGetResult
	_ = json.Unmarshal(resp.Result, &got)
	if got.Policy.Idle.UnloadAfterMinutes != 5 || got.Availability != nodepolicy.Available {
		t.Fatalf("policy:get = %+v", got)
	}
}

func TestPolicyStateReplayedToRestartedProxy(t *testing.T) {
	b, _ := newPolicyBroker(t)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Availability = nodepolicy.Paused
	r.live = nodepolicy.Paused
	r.drains["ollama"] = true
	r.haveAdmission = true
	r.admission.Active = 4
	r.mu.Unlock()
	b.observeIntent(map[string]bool{"ollama": true, "lmstudio": false})
	b.observeEngineRunning("ollama", true)
	b.observeLoaded(map[string][]string{"ollama": {"qwen3:8b"}})

	pp, proxy := attachFakeProxy(t, b)
	b.replayPolicyToProxy(context.Background(), pp)
	calls := proxy.waitFor("all five pushes", func(calls []policyCall) bool {
		seen := map[string]bool{}
		for _, c := range calls {
			seen[c.method] = true
		}
		return seen[nodepolicy.MethodSetAvailability] && seen[nodepolicy.MethodSetPolicy] && seen[nodepolicy.MethodSetEngineIntent] &&
			seen[nodepolicy.MethodSetResidency] && seen[nodepolicy.MethodSetEngineDrain]
	})
	if calls[0].method != nodepolicy.MethodSetAvailability {
		t.Fatalf("first push after a restart = %s; availability must reach the proxy first", calls[0].method)
	}
	if p := availabilityPushes(calls); p[0].State != nodepolicy.Paused {
		t.Fatalf("replayed availability = %+v, want paused", p[0])
	}
	for _, c := range calls {
		switch c.method {
		case nodepolicy.MethodSetResidency:
			if !strings.Contains(string(c.params), `"ollama":["qwen3:8b"]`) {
				t.Errorf("residency = %s", c.params)
			}
		case nodepolicy.MethodSetEngineIntent:
			if !strings.Contains(string(c.params), `"ollama":true`) || !strings.Contains(string(c.params), `"lmstudio":false`) {
				t.Errorf("intent = %s", c.params)
			}
		case nodepolicy.MethodSetEngineDrain:
			if !strings.Contains(string(c.params), `"drain":true`) {
				t.Errorf("drain = %s", c.params)
			}
		}
	}
	r.mu.Lock()
	stale := r.haveAdmission
	r.mu.Unlock()
	if stale {
		t.Fatal("admission state from the dead proxy survived its restart")
	}
}

func TestProxyAdmissionNotificationsReachThePolicy(t *testing.T) {
	b, _ := newPolicyBroker(t)
	b.forwardProxyProcessNotification(0, 0, nodepolicy.NotifyAdmissionState, json.RawMessage(`{"active":3,"activeByEngine":{"ollama":3},"queued":1,"lastActivityMs":{"ollama":5}}`))
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	st, have := r.admission, r.haveAdmission
	r.mu.Unlock()
	if !have || st.Active != 3 || st.ActiveByEngine["ollama"] != 3 || st.LastActivityMs["ollama"] != 5 {
		t.Fatalf("admission state = %+v (have %v)", st, have)
	}
}

func TestResidencyFollowsEngineManager(t *testing.T) {
	b, _ := newPolicyBroker(t)
	_, proxy := attachFakeProxy(t, b)
	residency := func(calls []policyCall) string {
		last := ""
		for _, c := range calls {
			if c.method == nodepolicy.MethodSetResidency {
				last = string(c.params)
			}
		}
		return last
	}
	models := `{"engine":"ollama","models":{"models":["a"],"loadedByEngine":{"ollama":["a"]}}}`
	b.forwardEngineNotification(engineModelsChangedMethod, json.RawMessage(models))
	proxy.waitFor("residency with a", func(calls []policyCall) bool { return residency(calls) == `{"loadedByEngine":{"ollama":["a"]}}` })
	b.forwardEngineNotification(engineStateChangedMethod, json.RawMessage(`{"engine":"ollama","running":false}`))
	proxy.waitFor("residency without ollama", func(calls []policyCall) bool { return residency(calls) == `{"loadedByEngine":{}}` })
	b.forwardEngineNotification(engineIntentChanged, json.RawMessage(`{"enabledByEngine":{"ollama":true}}`))
	proxy.waitFor("intent push", func(calls []policyCall) bool {
		for _, c := range calls {
			if c.method == nodepolicy.MethodSetEngineIntent && strings.Contains(string(c.params), `"ollama":true`) {
				return true
			}
		}
		return false
	})
}

// pausingBroker is a broker with a fake proxy and engine-manager, ollama
// running with one model loaded, and the given pause settings.
func pausingBroker(t *testing.T, pause nodepolicy.PauseConfig) (*Broker, *syncBuffer, *fakeWorker, *fakeWorker) {
	t.Helper()
	b, out := newPolicyBroker(t)
	_, proxy := attachFakeProxy(t, b)
	engine := attachFakeEngine(t, b)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Pause = pause
	r.mu.Unlock()
	b.observeIntent(map[string]bool{"ollama": true})
	b.observeLoaded(map[string][]string{"ollama": {"m1"}})
	return b, out, proxy, engine
}

func TestPauseFinishWaitsForDrainThenUnloadsAndStops(t *testing.T) {
	b, out, proxy, engine := pausingBroker(t, nodepolicy.PauseConfig{OnActive: nodepolicy.FinishActive, DrainTimeoutSeconds: 30, UnloadModels: true, StopEngines: true})
	b.observeAdmission(nodepolicy.AdmissionState{Active: 1, ActiveByEngine: map[string]int{"ollama": 1}})

	done := make(chan nodepolicy.Availability, 1)
	go func() {
		a, err := b.setAvailability(context.Background(), nodepolicy.Paused)
		if err != nil {
			t.Errorf("pause: %v", err)
		}
		done <- a
	}()
	proxy.waitFor("draining push", func(calls []policyCall) bool {
		p := availabilityPushes(calls)
		return len(p) > 0 && p[0].State == nodepolicy.Draining && !p[0].CancelActive
	})
	out.waitFrame(t, "draining notification", func(m Message) bool {
		return m.Method == notifyAvailabilityChanged && strings.Contains(string(m.Params), `"draining"`)
	})
	select {
	case <-done:
		t.Fatal("pause finished while work was still running")
	case <-time.After(100 * time.Millisecond):
	}
	if got := engine.methods("engine:"); len(got) != 0 {
		t.Fatalf("engines touched before the drain finished: %v", got)
	}
	b.observeAdmission(nodepolicy.AdmissionState{Active: 0, ActiveByEngine: map[string]int{}})
	select {
	case a := <-done:
		if a != nodepolicy.Paused {
			t.Fatalf("pause returned %q", a)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("pause did not finish after the drain")
	}
	calls := engine.snapshot()
	if len(calls) != 2 || calls[0].method != engineUnloadModelMethod || calls[1].method != engineSleepMethod {
		t.Fatalf("engine calls = %v, want unload then sleep", engine.methods(""))
	}
	if !strings.Contains(string(calls[0].params), `"model":"m1"`) {
		t.Fatalf("unload params = %s", calls[0].params)
	}
	pushes := availabilityPushes(proxy.snapshot())
	if last := pushes[len(pushes)-1]; last.State != nodepolicy.Paused {
		t.Fatalf("final availability push = %+v", last)
	}
	out.waitFrame(t, "paused notification", func(m Message) bool {
		return m.Method == notifyAvailabilityChanged && strings.Contains(string(m.Params), `"paused"`)
	})
}

func TestPauseCancelPushesCancelActiveImmediately(t *testing.T) {
	b, _, proxy, engine := pausingBroker(t, nodepolicy.PauseConfig{OnActive: nodepolicy.CancelActive, DrainTimeoutSeconds: 30})
	if a, err := b.setAvailability(context.Background(), nodepolicy.Paused); err != nil || a != nodepolicy.Paused {
		t.Fatalf("pause = %q, %v", a, err)
	}
	p := availabilityPushes(proxy.snapshot())
	if len(p) < 2 || p[0].State != nodepolicy.Draining || !p[0].CancelActive {
		t.Fatalf("availability pushes = %+v, want draining with cancelActive first", p)
	}
	if got := engine.methods("engine:"); len(got) != 0 {
		t.Fatalf("pause without unload/stop touched engines: %v", got)
	}
}

func TestPauseDrainTimeoutCancelsActiveWork(t *testing.T) {
	prev := pauseCancelSettle
	pauseCancelSettle = 50 * time.Millisecond
	t.Cleanup(func() { pauseCancelSettle = prev })
	b, _, proxy, _ := pausingBroker(t, nodepolicy.PauseConfig{OnActive: nodepolicy.FinishActive, DrainTimeoutSeconds: 0})
	b.observeAdmission(nodepolicy.AdmissionState{Active: 2})
	if a, err := b.setAvailability(context.Background(), nodepolicy.Paused); err != nil || a != nodepolicy.Paused {
		t.Fatalf("pause = %q, %v", a, err)
	}
	p := availabilityPushes(proxy.snapshot())
	if len(p) != 3 || p[0].CancelActive || p[1].State != nodepolicy.Draining || !p[1].CancelActive || p[2].State != nodepolicy.Paused {
		t.Fatalf("availability pushes = %+v, want draining, draining+cancelActive, paused", p)
	}
}

func TestResumeWakesEnginesThePauseStopped(t *testing.T) {
	b, _, proxy, engine := pausingBroker(t, nodepolicy.PauseConfig{OnActive: nodepolicy.FinishActive, DrainTimeoutSeconds: 5, StopEngines: true})
	if _, err := b.setAvailability(context.Background(), nodepolicy.Paused); err != nil {
		t.Fatal(err)
	}
	if a, err := b.setAvailability(context.Background(), nodepolicy.Available); err != nil || a != nodepolicy.Available {
		t.Fatalf("resume = %q, %v", a, err)
	}
	engine.waitFor("engine:wake after resume", func(calls []policyCall) bool {
		for _, c := range calls {
			if c.method == engineWakeMethod && strings.Contains(string(c.params), `"ollama"`) {
				return true
			}
		}
		return false
	})
	p := availabilityPushes(proxy.snapshot())
	if last := p[len(p)-1]; last.State != nodepolicy.Available {
		t.Fatalf("final push = %+v", last)
	}
	if got := b.localPolicy(); got.Availability != nodepolicy.Available || got.Policy.Availability != nodepolicy.Available {
		t.Fatalf("resume state = %+v", got)
	}
}

func TestLatestAvailabilityRequestWins(t *testing.T) {
	b, _, proxy, _ := pausingBroker(t, nodepolicy.PauseConfig{OnActive: nodepolicy.FinishActive, DrainTimeoutSeconds: 600})
	b.observeAdmission(nodepolicy.AdmissionState{Active: 1})
	pauseErr := make(chan error, 1)
	go func() {
		_, err := b.setAvailability(context.Background(), nodepolicy.Paused)
		pauseErr <- err
	}()
	proxy.waitFor("draining push", func(calls []policyCall) bool { return len(availabilityPushes(calls)) > 0 })
	resumed := make(chan nodepolicy.Availability, 1)
	go func() {
		a, err := b.setAvailability(context.Background(), nodepolicy.Available)
		if err != nil {
			t.Errorf("resume: %v", err)
		}
		resumed <- a
	}()
	select {
	case err := <-pauseErr:
		if err != errAvailabilitySuperseded {
			t.Fatalf("superseded pause returned %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the pending pause was not superseded")
	}
	select {
	case a := <-resumed:
		if a != nodepolicy.Available {
			t.Fatalf("resume returned %q", a)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("resume deadlocked behind the drain")
	}
	p := availabilityPushes(proxy.snapshot())
	if last := p[len(p)-1]; last.State != nodepolicy.Available {
		t.Fatalf("final availability push = %+v, want available", last)
	}
}

func TestStartupPausedWithStopEnginesSkipsRestoreUntilResume(t *testing.T) {
	path := filepath.Join(t.TempDir(), nodepolicy.FileName)
	paused := nodepolicy.Default()
	paused.Availability = nodepolicy.Paused
	paused.Pause.StopEngines = true
	writePolicyFile(t, path, paused)
	b, _ := newPolicyBroker(t)
	b.loadNodePolicy(path)
	engine := attachFakeEngine(t, b)

	b.restoreEnabledEngines(b.getEngineMgr())
	time.Sleep(50 * time.Millisecond)
	for _, c := range engine.snapshot() {
		if c.method == restoreEnabledEnginesMethod {
			t.Fatal("a node paused with stopEngines restored its engines at startup")
		}
	}
	if _, err := b.setAvailability(context.Background(), nodepolicy.Available); err != nil {
		t.Fatal(err)
	}
	engine.waitFor("engine:restore-enabled after resume", func(calls []policyCall) bool {
		for _, c := range calls {
			if c.method == restoreEnabledEnginesMethod && c.notify {
				return true
			}
		}
		return false
	})
}

func TestStartupPausedWithoutStopEnginesStillRestores(t *testing.T) {
	path := filepath.Join(t.TempDir(), nodepolicy.FileName)
	paused := nodepolicy.Default()
	paused.Availability = nodepolicy.Paused
	writePolicyFile(t, path, paused)
	b, _ := newPolicyBroker(t)
	b.loadNodePolicy(path)
	engine := attachFakeEngine(t, b)
	b.restoreEnabledEngines(b.getEngineMgr())
	engine.waitFor("engine:restore-enabled", func(calls []policyCall) bool {
		return len(calls) > 0 && calls[0].method == restoreEnabledEnginesMethod
	})
}

func TestWakeRefusedWhenPausedOrSavedOff(t *testing.T) {
	b, out := newPolicyBroker(t)
	engine := attachFakeEngine(t, b)
	r := b.nodePolicyRuntime()

	r.mu.Lock()
	r.live = nodepolicy.Paused
	r.mu.Unlock()
	b.handleMessage(clientRequest(1, engineWakeMethod, map[string]string{"engine": "ollama"}))
	resp := out.waitFrame(t, "engine:wake refusal", isResponse(1))
	if resp.Error == nil || !strings.Contains(resp.Error.Message, "paused") {
		t.Fatalf("engine:wake while paused = %+v", resp)
	}
	b.wakeOnDemand("ollama")

	r.mu.Lock()
	r.live = nodepolicy.Available
	r.mu.Unlock()
	b.observeIntent(map[string]bool{"ollama": false, "lmstudio": true})
	b.wakeOnDemand("ollama")
	if got := engine.methods(engineWakeMethod); len(got) != 0 {
		t.Fatalf("wake reached engine-manager while paused or for an engine saved Off: %v", got)
	}

	b.handleAdmissionNotification(nodepolicy.NotifyAdmissionWake, json.RawMessage(`{"engine":"lmstudio"}`))
	engine.waitFor("on-demand wake", func(calls []policyCall) bool {
		return len(calls) == 1 && calls[0].method == engineWakeMethod && strings.Contains(string(calls[0].params), "lmstudio")
	})
	b.handleMessage(clientRequest(2, engineWakeMethod, map[string]string{"engine": "lmstudio"}))
	engine.waitFor("relayed client wake", func(calls []policyCall) bool { return len(calls) == 2 })
}

func TestAdmissionUnloadUnloadsThoseModels(t *testing.T) {
	b, _ := newPolicyBroker(t)
	engine := attachFakeEngine(t, b)
	b.handleAdmissionNotification(nodepolicy.NotifyAdmissionUnload, json.RawMessage(`{"engine":"ollama","models":["a","b"]}`))
	calls := engine.waitFor("two unloads", func(calls []policyCall) bool { return len(calls) == 2 })
	for i, model := range []string{"a", "b"} {
		if calls[i].method != engineUnloadModelMethod || !strings.Contains(string(calls[i].params), `"model":"`+model+`"`) || !strings.Contains(string(calls[i].params), `"engine":"ollama"`) {
			t.Fatalf("call %d = %s %s", i, calls[i].method, calls[i].params)
		}
	}
}

func TestIdlePolicyUnloadsThenStopsOnSchedule(t *testing.T) {
	b, _ := newPolicyBroker(t)
	engine := attachFakeEngine(t, b)
	clock := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.now = func() time.Time { return clock }
	r.current.Idle.UnloadAfterMinutes = 5
	r.current.Idle.StopEngineAfterMinutes = 20
	r.mu.Unlock()
	b.observeIntent(map[string]bool{"ollama": true, "lmstudio": false})
	b.observeLoaded(map[string][]string{"ollama": {"m1"}, "lmstudio": {"x"}})
	b.observeAdmission(nodepolicy.AdmissionState{LastActivityMs: map[string]int64{"ollama": clock.Add(2 * time.Minute).UnixMilli()}})

	tick := func(at time.Duration) {
		r.mu.Lock()
		r.now = func() time.Time { return clock.Add(at) }
		r.mu.Unlock()
		b.idleTick(context.Background())
	}
	tick(4 * time.Minute)
	if got := engine.methods(""); len(got) != 0 {
		t.Fatalf("idle policy acted before any threshold: %v", got)
	}
	tick(6 * time.Minute) // lmstudio idle 6m, ollama idle 4m since its last request
	calls := engine.snapshot()
	if len(calls) != 1 || !strings.Contains(string(calls[0].params), `"lmstudio"`) {
		t.Fatalf("calls at 6m = %v %v", engine.methods(""), calls)
	}
	tick(8 * time.Minute) // ollama now idle 6m
	calls = engine.snapshot()
	if len(calls) != 2 || calls[1].method != engineUnloadModelMethod || !strings.Contains(string(calls[1].params), `"ollama"`) {
		t.Fatalf("calls at 8m = %v", engine.methods(""))
	}
	tick(9 * time.Minute)
	if n := len(engine.snapshot()); n != 2 {
		t.Fatalf("idle unload repeated for the same idle period: %v", engine.methods(""))
	}
	tick(23 * time.Minute) // ollama idle 21m; lmstudio idle too but saved Off
	calls = engine.snapshot()
	if len(calls) != 3 || calls[2].method != engineSleepMethod || !strings.Contains(string(calls[2].params), `"ollama"`) {
		t.Fatalf("calls at 23m = %v", engine.methods(""))
	}
	for _, c := range calls {
		if c.method == engineSleepMethod && strings.Contains(string(c.params), "lmstudio") {
			t.Fatal("idle policy stopped an engine that is not saved On")
		}
	}

	// A new request resets the clock.
	b.observeAdmission(nodepolicy.AdmissionState{
		ActiveByEngine: map[string]int{"ollama": 1},
		LastActivityMs: map[string]int64{"ollama": clock.Add(30 * time.Minute).UnixMilli()},
	})
	tick(60 * time.Minute)
	if n := len(engine.snapshot()); n != 3 {
		t.Fatalf("idle policy acted on an engine with running work: %v", engine.methods(""))
	}
}

func TestRemotePolicyRelaysThroughEngineManager(t *testing.T) {
	b, out := newPolicyBroker(t)
	engine := attachFakeEngine(t, b)
	engine.setAnswer(func(method string, params json.RawMessage) (any, string) {
		switch method {
		case "engine:remote-policy-set":
			return map[string]any{"policy": map[string]any{"version": 1}}, ""
		case "engine:remote-availability-set":
			return map[string]any{"availability": "paused"}, ""
		}
		return map[string]any{"policy": map[string]any{}, "availability": "available"}, ""
	})
	b.handleMessage(clientRequest(1, methodPolicyGet, map[string]string{"nodeId": "peer-node"}))
	b.handleMessage(clientRequest(2, methodPolicySet, map[string]any{"nodeId": "peer-node", "policy": map[string]any{"version": 1}}))
	b.handleMessage(clientRequest(3, methodSetAvailability, map[string]any{"nodeId": "peer-node", "state": "paused"}))
	for id := 1; id <= 3; id++ {
		if resp := out.waitFrame(t, "remote policy response", isResponse(id)); resp.Error != nil {
			t.Fatalf("response %d error: %+v", id, resp.Error)
		}
	}
	calls := engine.snapshot()
	want := []string{"engine:remote-policy-get", "engine:remote-policy-set", "engine:remote-availability-set"}
	got := map[string]bool{}
	for _, c := range calls {
		got[c.method] = true
		if !strings.Contains(string(c.params), `"nodeId":"peer-node"`) {
			t.Errorf("%s lost its target: %s", c.method, c.params)
		}
	}
	for _, m := range want {
		if !got[m] {
			t.Errorf("engine-manager never received %s (got %v)", m, engine.methods(""))
		}
	}
	out.waitFrame(t, "remote policy:changed", func(m Message) bool {
		return m.Method == notifyPolicyChanged && strings.Contains(string(m.Params), `"peer-node"`)
	})
	out.waitFrame(t, "remote availability notification", func(m Message) bool {
		return m.Method == notifyAvailabilityChanged && strings.Contains(string(m.Params), `"peer-node"`)
	})
	if got := b.localPolicy().Availability; got != nodepolicy.Available {
		t.Fatalf("a remote pause paused this node: %q", got)
	}
}

func TestPolicyRelayFromPeerActsLocallyOnly(t *testing.T) {
	b, _ := newPolicyBroker(t)
	engine := attachFakeEngine(t, b)
	clustertrusttest.Join(t, b.clusterDir, "cluster-1", "self-node", "pinned-peer")

	reply := func(id string) map[string]any {
		var out map[string]any
		engine.waitFor("policy:reply "+id, func(calls []policyCall) bool {
			for _, c := range calls {
				if c.method == policyRelayReplyMethod && strings.Contains(string(c.params), `"id":"`+id+`"`) {
					_ = json.Unmarshal(c.params, &out)
					return true
				}
			}
			return false
		})
		return out
	}
	// A relayed request naming another node acts here and never hops again.
	b.forwardEngineNotification(policyRelayRequestMethod, mustJSON(map[string]any{
		"id": "r1", "method": "set-availability", "caller": "pinned-peer",
		"params": map[string]any{"nodeId": "third-node", "state": "paused"},
	}))
	if got := reply("r1"); got["error"] != nil {
		t.Fatalf("pinned relay refused: %v", got)
	}
	if got := b.localPolicy().Availability; got != nodepolicy.Paused {
		t.Fatalf("relayed pause did not pause this node: %q", got)
	}
	for _, c := range engine.snapshot() {
		if strings.HasPrefix(c.method, "engine:remote-") {
			t.Fatalf("a relayed request was forwarded again: %s", c.method)
		}
	}

	b.forwardEngineNotification(policyRelayRequestMethod, mustJSON(map[string]any{
		"id": "r2", "method": "set", "caller": "unpinned-peer",
		"params": map[string]any{"policy": map[string]any{"version": 1}},
	}))
	if got := reply("r2"); got["error"] == nil {
		t.Fatalf("relay from an unpinned caller accepted: %v", got)
	}

	b.forwardEngineNotification(policyRelayRequestMethod, mustJSON(map[string]any{
		"id": "r3", "method": "get", "caller": "pinned-peer", "params": map[string]any{},
	}))
	got := reply("r3")
	result, _ := got["result"].(map[string]any)
	if result["availability"] != string(nodepolicy.Paused) {
		t.Fatalf("relayed get = %v", got)
	}
}
