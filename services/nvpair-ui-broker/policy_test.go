// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/durablefile"
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
	conn   net.Conn
	codec  *Codec
	mu    sync.Mutex
	calls []policyCall
	// projections holds the engine:node-projection notifications apart from
	// calls: every save, pause and resume sends one, and they are state, not
	// the engine actions the tests that read calls assert on.
	projections []policyCall
	answer      func(method string, params json.RawMessage) (any, string)
	signal chan struct{}
	// held is closed by releaseHeld; a request whose answer is holdReply is
	// answered only then, while the worker keeps reading, as a slow proxy does.
	held     chan struct{}
	holdOnce sync.Once
}

// holdReply is what an answer function returns to leave the request unanswered
// until releaseHeld.
const holdReply = "\x00hold"

func (f *fakeWorker) releaseHeld() { f.holdOnce.Do(func() { close(f.held) }) }

func (f *fakeWorker) serve() {
	for {
		msg, err := f.codec.Read()
		if err != nil {
			return
		}
		f.mu.Lock()
		call := policyCall{method: msg.Method, params: append(json.RawMessage(nil), msg.Params...), notify: !msg.IsRequest()}
		if call.method == engineNodeProjectionMethod {
			f.projections = append(f.projections, call)
		} else {
			f.calls = append(f.calls, call)
		}
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
		if errText == holdReply {
			go func() {
				<-f.held
				_ = f.codec.RespondError(msg.ID, -32000, "answered after the bound")
			}()
			continue
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
	f := &fakeWorker{t: t, conn: server, codec: NewCodec(server), signal: make(chan struct{}, 1), held: make(chan struct{})}
	t.Cleanup(f.releaseHeld)
	go f.serve()
	return peer, f
}

// proxyAcks answers node/set-policy the way the real proxy does: it holds one
// revision, refuses a policy older than it, and acknowledges with its run id.
// Everything else, node/set-availability included, is answered {"ok":true}.
type proxyAcks struct {
	runID string
	mu    sync.Mutex
	held  string
}

func (a *proxyAcks) heldRevision() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.held
}

func (a *proxyAcks) answer(method string, params json.RawMessage) (any, string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	switch method {
	case nodepolicy.MethodSetPolicy:
		var in struct {
			Revision string `json:"revision"`
		}
		if err := json.Unmarshal(params, &in); err != nil {
			return nil, "unreadable policy"
		}
		if cmp, err := nodepolicy.CompareRevisions(in.Revision, a.held); err != nil || cmp < 0 {
			return nil, "revision " + in.Revision + " is older than the active revision"
		}
		a.held = in.Revision
	default:
		return map[string]bool{"ok": true}, ""
	}
	return nodepolicy.PolicyAck{Revision: a.held, RunID: a.runID}, ""
}

var fakeProxyRuns atomic.Int64

// attachAckingProxy publishes a fake proxy process for every engine, with a run
// id of its own, that acknowledges like the real one. It does not replay the
// policy to it: the broker's start does that, and a test calls
// replayPolicyToProxy when it wants the start.
func attachAckingProxy(t *testing.T, b *Broker) (*proxyProcess, *fakeWorker, *proxyAcks) {
	t.Helper()
	peer, f := newFakeWorkerPipe(t)
	pp := &proxyProcess{peer: peer}
	acks := &proxyAcks{runID: "run-" + strconv.FormatInt(fakeProxyRuns.Add(1), 10)}
	f.setAnswer(acks.answer)
	publishTestProxy(b, pp)
	return pp, f, acks
}

// attachFakeProxy is attachAckingProxy for a test that does not use the
// acknowledgement state.
func attachFakeProxy(t *testing.T, b *Broker) (*proxyProcess, *fakeWorker) {
	t.Helper()
	pp, f, _ := attachAckingProxy(t, b)
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
	if got := b.localPolicy(); got.Availability != nodepolicy.Paused || !got.RecoveryRequired {
		t.Fatalf("invalid file was not replaced by the paused recovery policy: availability=%q recoveryRequired=%t", got.Availability, got.RecoveryRequired)
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
	if _, err := setCurrent(b, mustJSON(next)); err != nil {
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
	if entries, _ := filepath.Glob(filepath.Join(dir, nodepolicy.FileName+".*.tmp")); len(entries) != 0 {
		t.Fatalf("atomic write left temp files: %v", entries)
	}
}

// activatedBroker is a broker on a fresh policy file whose acknowledging proxy
// started and was replayed the persisted policy: revision 0 is active. Its
// clock is fixed, so the revisions it assigns are 5000, 5001 and so on.
type activatedBroker struct {
	b     *Broker
	out   *syncBuffer
	pp    *proxyProcess
	proxy *fakeWorker
	acks  *proxyAcks
	path  string
}

func newActivatedBroker(t *testing.T) *activatedBroker {
	t.Helper()
	b, out := newPolicyBroker(t)
	path := filepath.Join(t.TempDir(), nodepolicy.FileName)
	b.loadNodePolicy(path)
	setClock(b, 5000)
	pp, proxy, acks := attachAckingProxy(t, b)
	b.replayPolicyToProxy(context.Background(), pp)
	ab := &activatedBroker{b: b, out: out, pp: pp, proxy: proxy, acks: acks, path: path}
	ab.wantActivation(t, "0", activationActive, "")
	return ab
}

// wantActivation checks the persisted revision against the activation fields
// policy:get reports: the active revision, the state and, when wantErr is not
// empty, that the error text names it.
func (ab *activatedBroker) wantActivation(t *testing.T, wantActive, wantState, wantErr string) policyGetResult {
	t.Helper()
	got := ab.b.localPolicy()
	if got.ActiveRevision != wantActive || got.Activation != wantState {
		t.Fatalf("policy:get at revision %q reports %+v, want %s at %q", got.Policy.Revision, got.policyActivation, wantState, wantActive)
	}
	if wantErr == "" && got.ActivationError != "" {
		t.Fatalf("policy:get reports an error %q for a %s revision", got.ActivationError, got.Activation)
	}
	if wantErr != "" && !strings.Contains(got.ActivationError, wantErr) {
		t.Fatalf("activation error = %q, want it to contain %q", got.ActivationError, wantErr)
	}
	return got
}

// setIdle is a local policy:set that changes only the idle unload minutes.
func (ab *activatedBroker) setIdle(t *testing.T, minutes int) policySetResult {
	t.Helper()
	result, err := ab.b.dispatchPolicy(context.Background(), methodPolicySet, policyParams{
		Policy:                    mustJSON(map[string]any{"version": nodepolicy.SchemaVersion, "idle": map[string]any{"unloadAfterMinutes": minutes}}),
		ExpectedPersistedRevision: mustJSON(ab.b.localPolicy().Policy.Revision),
	})
	if err != nil {
		t.Fatalf("policy:set: %v", err)
	}
	set, ok := result.(policySetResult)
	if !ok {
		t.Fatalf("policy:set result is %T", result)
	}
	return set
}

// lastPolicyChanged is the newest policy:changed this node announced.
func lastPolicyChanged(t *testing.T, out *syncBuffer) (nodepolicy.Policy, policyActivation) {
	t.Helper()
	var last *Message
	frames := out.frames()
	for i := range frames {
		if frames[i].Method == notifyPolicyChanged {
			last = &frames[i]
		}
	}
	if last == nil {
		t.Fatal("no policy:changed was announced")
	}
	var event struct {
		Policy nodepolicy.Policy `json:"policy"`
		policyActivation
	}
	if err := json.Unmarshal(last.Params, &event); err != nil {
		t.Fatalf("decode policy:changed: %v", err)
	}
	return event.Policy, event.policyActivation
}

func TestPolicySetValidatesPushesAndNotifies(t *testing.T) {
	b, out := newPolicyBroker(t)
	setClock(b, 5000)
	pp, proxy, _ := attachAckingProxy(t, b)
	b.replayPolicyToProxy(context.Background(), pp)

	bad := map[string]any{"version": 1, "profiles": []any{map[string]any{"name": "x", "engine": "nope", "model": "m"}}}
	b.handleMessage(clientRequest(1, methodPolicySet, map[string]any{"policy": bad, "expectedPersistedRevision": "0"}))
	resp := out.waitFrame(t, "invalid policy:set response", isResponse(1))
	if resp.Error == nil || resp.Error.Code != -32602 {
		t.Fatalf("invalid policy accepted: %+v", resp)
	}

	good := map[string]any{"version": 1, "availability": "paused", "idle": map[string]any{"unloadAfterMinutes": 5}}
	b.handleMessage(clientRequest(2, methodPolicySet, map[string]any{"policy": good, "expectedPersistedRevision": "0"}))
	resp = out.waitFrame(t, "policy:set response", isResponse(2))
	if resp.Error != nil {
		t.Fatalf("valid policy refused: %+v", resp.Error)
	}
	var result policySetResult
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		t.Fatalf("decode policy:set result %s: %v", resp.Result, err)
	}
	if result.Policy.Availability != nodepolicy.Available || result.Policy.Idle.UnloadAfterMinutes != 5 || !result.Policy.Idle.StartOnDemand {
		t.Fatalf("policy:set result = %+v; want availability kept and omitted fields defaulted", result.Policy)
	}
	// The proxy acknowledged the exact revision this set persisted, so the
	// result says it is enforced, not merely saved.
	if result.Policy.Revision != "5000" || result.ActiveRevision != "5000" || result.Activation != activationActive || result.ActivationError != "" {
		t.Fatalf("policy:set result = revision %q, %+v; want revision 5000 active at 5000", result.Policy.Revision, result.policyActivation)
	}
	changed := out.waitFrame(t, "policy:changed for revision 5000", func(m Message) bool {
		return m.Method == notifyPolicyChanged && strings.Contains(string(m.Params), `"revision":"5000"`)
	})
	var event struct {
		Policy nodepolicy.Policy `json:"policy"`
		policyActivation
	}
	if err := json.Unmarshal(changed.Params, &event); err != nil {
		t.Fatalf("decode policy:changed %s: %v", changed.Params, err)
	}
	if event.Policy.Revision != "5000" || event.ActiveRevision != "5000" || event.Activation != activationActive {
		t.Fatalf("policy:changed = revision %q, %+v; want revision 5000 active at 5000", event.Policy.Revision, event.policyActivation)
	}
	var pushed nodepolicy.Policy
	for _, c := range proxy.snapshot() {
		if c.method == nodepolicy.MethodSetPolicy {
			if err := json.Unmarshal(c.params, &pushed); err != nil {
				t.Fatalf("decode node/set-policy push %s: %v", c.params, err)
			}
		}
	}
	if pushed.Idle.UnloadAfterMinutes != 5 || pushed.Revision != "5000" {
		t.Fatalf("last node/set-policy push = revision %q, unloadAfterMinutes %d; want revision 5000 and 5", pushed.Revision, pushed.Idle.UnloadAfterMinutes)
	}

	b.handleMessage(clientRequest(3, methodPolicyGet, nil))
	resp = out.waitFrame(t, "policy:get response", isResponse(3))
	var got policyGetResult
	if err := json.Unmarshal(resp.Result, &got); err != nil {
		t.Fatalf("decode policy:get result %s: %v", resp.Result, err)
	}
	if got.Policy.Idle.UnloadAfterMinutes != 5 || got.Availability != nodepolicy.Available {
		t.Fatalf("policy:get = %+v", got)
	}
	if got.Policy.Revision != "5000" || got.ActiveRevision != "5000" || got.Activation != activationActive {
		t.Fatalf("policy:get = revision %q, %+v; want revision 5000 active at 5000", got.Policy.Revision, got.policyActivation)
	}
}

// waitPushesSettled returns once every push the observe calls scheduled has
// run, so none can reach a proxy attached afterwards ahead of its replay.
func waitPushesSettled(t *testing.T, b *Broker) {
	t.Helper()
	r := b.nodePolicyRuntime()
	deadline := time.Now().Add(3 * time.Second)
	for {
		r.mu.Lock()
		scheduled := len(r.pendingPush)
		r.mu.Unlock()
		if scheduled == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("%d scheduled pushes never ran", scheduled)
		}
		time.Sleep(time.Millisecond)
	}
	// A push that has left the pending set may still be sending.
	r.acquirePush(context.Background())
	r.releasePush()
}

func TestPolicyStateReplayedToRestartedProxy(t *testing.T) {
	b, out := newPolicyBroker(t)
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

	waitPushesSettled(t, b)
	pp, proxy, acks := attachAckingProxy(t, b)
	if got := b.activationOf("0"); got.Activation != activationPending || got.ActiveRevision != "" {
		t.Fatalf("before the replay the new proxy has acknowledged %+v; want pending with no active revision", got)
	}
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

	// The replayed policy is the persisted revision, acknowledged by this
	// process, and the announcement says so.
	got := b.localPolicy()
	if got.Activation != activationActive || got.ActiveRevision != got.Policy.Revision || got.Policy.Revision != "0" {
		t.Fatalf("after the replay policy:get = revision %q, %+v; want revision 0 active at 0", got.Policy.Revision, got.policyActivation)
	}
	if held := acks.heldRevision(); held != "0" {
		t.Fatalf("the restarted proxy holds revision %q, want 0", held)
	}
	announced, activation := lastPolicyChanged(t, out)
	if announced.Revision != "0" || activation.Activation != activationActive || activation.ActiveRevision != "0" {
		t.Fatalf("policy:changed after the replay = revision %q, %+v; want revision 0 active at 0", announced.Revision, activation)
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
		a, err := b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally)
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
	if a, err := b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil || a != nodepolicy.Paused {
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
	if a, err := b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil || a != nodepolicy.Paused {
		t.Fatalf("pause = %q, %v", a, err)
	}
	p := availabilityPushes(proxy.snapshot())
	if len(p) != 3 || p[0].CancelActive || p[1].State != nodepolicy.Draining || !p[1].CancelActive || p[2].State != nodepolicy.Paused {
		t.Fatalf("availability pushes = %+v, want draining, draining+cancelActive, paused", p)
	}
}

func TestResumeWakesEnginesThePauseStopped(t *testing.T) {
	b, _, proxy, engine := pausingBroker(t, nodepolicy.PauseConfig{OnActive: nodepolicy.FinishActive, DrainTimeoutSeconds: 5, StopEngines: true})
	if _, err := b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil {
		t.Fatal(err)
	}
	if a, err := b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedLocally); err != nil || a != nodepolicy.Available {
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
		_, err := b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally)
		pauseErr <- err
	}()
	proxy.waitFor("draining push", func(calls []policyCall) bool { return len(availabilityPushes(calls)) > 0 })
	resumed := make(chan nodepolicy.Availability, 1)
	go func() {
		a, err := b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedLocally)
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
	if _, err := b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedLocally); err != nil {
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
	b.handleMessage(clientRequest(2, methodPolicySet, map[string]any{"nodeId": "peer-node", "policy": map[string]any{"version": 1}, "expectedPersistedRevision": "77"}))
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
		if c.method == "engine:remote-policy-set" && !strings.Contains(string(c.params), `"expectedPersistedRevision":"77"`) {
			t.Errorf("%s lost its precondition: %s", c.method, c.params)
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

// routedPolicyDocument is a stored version 2 policy with origin-only routing
// content: a local and an API entry in one tier, an API-only tier, two
// accounts, two deployments and a listener.
const routedPolicyDocument = `{
  "version": 2,
  "revision": "12",
  "tiers": {
    "weak": [
      {"kind": "local", "engine": "ollama", "model": "qwen3:8b"},
      {"kind": "api", "deployment": "free-small"}
    ],
    "strong": [{"kind": "api", "deployment": "free-small"}]
  },
  "routing": {
    "accounts": [
      {"id": "free-acct", "credential": "NVPAIR_PROVIDER_SECRET_FREE", "billing": "free",
       "perRequestCeiling": "0", "dailyCeiling": "0", "monthlyCeiling": "0"},
      {"id": "paid-acct", "credential": "NVPAIR_PROVIDER_SECRET_PAID", "billing": "paid",
       "perRequestCeiling": "1000000", "dailyCeiling": "9007199254740993", "monthlyCeiling": "9223372036854775807"}
    ],
    "deployments": [
      {"id": "free-small", "account": "free-acct", "model": "small-1", "family": "alpha",
       "contextTokens": 32768, "maxOutputTokens": 4096, "enabled": false},
      {"id": "paid-big", "account": "paid-acct", "model": "big-1", "family": "beta",
       "contextTokens": 200000, "maxOutputTokens": 8192, "enabled": false}
    ],
    "listener": {"enabled": true, "port": 4321, "allowedOrigins": ["http://localhost:3000"]}
  }
}`

// routedBroker loads routedPolicyDocument, attaches a fake proxy and engine
// manager, and pins "pinned-peer" so it may relay.
type routedBroker struct {
	b      *Broker
	out    *syncBuffer
	proxy  *fakeWorker
	engine *fakeWorker
	path   string
	relays int
}

func newRoutedBroker(t *testing.T) *routedBroker {
	t.Helper()
	rb := newBrokerOnFile(t, routedPolicyDocument)
	if got := rb.b.localPolicy().Policy.Routing.Accounts; len(got) != 2 {
		t.Fatalf("loaded accounts = %d, want the 2 in the fixture (was the fixture rejected?)", len(got))
	}
	return rb
}

// newBrokerOnFile is a routedBroker whose policy file holds contents, valid or
// not, when the broker loads it.
func newBrokerOnFile(t *testing.T, contents string) *routedBroker {
	t.Helper()
	b, out := newPolicyBroker(t)
	_, proxy := attachFakeProxy(t, b)
	engine := attachFakeEngine(t, b)
	clustertrusttest.Join(t, b.clusterDir, "cluster-1", "self-node", "pinned-peer")
	path := filepath.Join(t.TempDir(), nodepolicy.FileName)
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("write policy file: %v", err)
	}
	b.loadNodePolicy(path)
	return &routedBroker{b: b, out: out, proxy: proxy, engine: engine, path: path}
}

func (rb *routedBroker) relay(t *testing.T, method string, params any) (result json.RawMessage, errText string) {
	t.Helper()
	reply := rb.relayReply(t, method, params)
	return reply.Result, reply.Error
}

type peerPolicyReply struct {
	Result    json.RawMessage `json:"result"`
	Error     string          `json:"error"`
	ErrorCode int             `json:"errorCode"`
}

// relayReply exercises the notification path and decodes the broker's wire reply.
func (rb *routedBroker) relayReply(t *testing.T, method string, params any) peerPolicyReply {
	t.Helper()
	rb.relays++
	id := "relay-" + strings.Repeat("x", rb.relays)
	rb.b.forwardEngineNotification(policyRelayRequestMethod, mustJSON(map[string]any{
		"id": id, "method": method, "caller": "pinned-peer", "params": params,
	}))
	calls := rb.engine.waitFor("policy:reply "+id, func(calls []policyCall) bool {
		for _, c := range calls {
			if c.method == policyRelayReplyMethod && strings.Contains(string(c.params), `"id":"`+id+`"`) {
				return true
			}
		}
		return false
	})
	for _, c := range calls {
		if c.method == policyRelayReplyMethod && strings.Contains(string(c.params), `"id":"`+id+`"`) {
			var reply peerPolicyReply
			if err := json.Unmarshal(c.params, &reply); err != nil {
				t.Fatalf("decode reply: %v", err)
			}
			return reply
		}
	}
	t.Fatalf("reply %s vanished", id)
	return peerPolicyReply{}
}

// peerSnapshot is a relayed policy:get: the document a peer would edit.
func (rb *routedBroker) peerSnapshot(t *testing.T) map[string]any {
	t.Helper()
	result, errText := rb.relay(t, "get", map[string]any{})
	if errText != "" {
		t.Fatalf("relayed get: %s", errText)
	}
	var got struct {
		Policy map[string]any `json:"policy"`
	}
	if err := json.Unmarshal(result, &got); err != nil || got.Policy == nil {
		t.Fatalf("decode relayed get %s: %v", result, err)
	}
	return got.Policy
}

// fileSections reads the policy file and returns its top-level sections.
func (rb *routedBroker) fileSections(t *testing.T) map[string]json.RawMessage {
	t.Helper()
	data, err := os.ReadFile(rb.path)
	if err != nil {
		t.Fatalf("read policy file: %v", err)
	}
	var sections map[string]json.RawMessage
	if err := json.Unmarshal(data, &sections); err != nil {
		t.Fatalf("decode policy file: %v", err)
	}
	return sections
}

func (rb *routedBroker) fileBytes(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile(rb.path)
	if err != nil {
		t.Fatalf("read policy file: %v", err)
	}
	return data
}

// setCurrent is a local policy:set that carries the revision a policy:get
// returns right now, so it is never stale.
func setCurrent(b *Broker, raw json.RawMessage) (nodepolicy.Policy, error) {
	return b.setLocalPolicy(raw, mustJSON(b.localPolicy().Policy.Revision))
}

// relaySet sends a relayed policy:set whose precondition is the revision the
// peer's own snapshot carries.
func (rb *routedBroker) relaySet(t *testing.T, doc map[string]any) (result json.RawMessage, errText string) {
	t.Helper()
	return rb.relay(t, "set", map[string]any{"policy": doc, "expectedPersistedRevision": doc["revision"]})
}

// localSet applies a local policy:set that edits the current full policy.
func (rb *routedBroker) localSet(t *testing.T, edit func(doc map[string]any)) {
	t.Helper()
	data, err := json.Marshal(rb.b.localPolicy().Policy)
	if err != nil {
		t.Fatalf("marshal current policy: %v", err)
	}
	var doc map[string]any
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("decode current policy: %v", err)
	}
	edit(doc)
	if _, err := setCurrent(rb.b, mustJSON(doc)); err != nil {
		t.Fatalf("local set: %v", err)
	}
}

// policyPushes counts node/set-policy calls the fake proxy received.
func policyPushes(f *fakeWorker) int { return len(f.methods(nodepolicy.MethodSetPolicy)) }

// leaks are strings that identify origin-only routing content.
var routingLeaks = []string{
	"routing", "NVPAIR_PROVIDER_SECRET", "free-acct", "paid-acct", "free-small", "paid-big",
	"deployment", `"api"`, "9223372036854775807", "allowedOrigins",
}

func TestPeerPolicyGetReturnsTheProjection(t *testing.T) {
	rb := newRoutedBroker(t)
	result, errText := rb.relay(t, "get", map[string]any{})
	if errText != "" {
		t.Fatalf("relayed get: %s", errText)
	}
	for _, leak := range routingLeaks {
		if strings.Contains(string(result), leak) {
			t.Errorf("relayed get contains %q: %s", leak, result)
		}
	}
	var got struct {
		Policy       nodepolicy.PeerPolicy `json:"policy"`
		Availability string                `json:"availability"`
	}
	if err := json.Unmarshal(result, &got); err != nil {
		t.Fatalf("decode relayed get: %v", err)
	}
	wantTiers := map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "qwen3:8b"}}}
	if !reflect.DeepEqual(got.Policy.Tiers, wantTiers) {
		t.Errorf("relayed tiers = %+v, want %+v", got.Policy.Tiers, wantTiers)
	}
	if got.Policy.Revision != "12" || got.Availability != "available" {
		t.Errorf("relayed revision/availability = %q/%q, want 12/available", got.Policy.Revision, got.Availability)
	}
}

// A paired node sees no activation fields in the policy it reads, so the answer
// to its node:set-availability carries none either; the node's own client gets
// the revision and the activation of it.
func TestRelayedSetAvailabilityShowsNoActivationFields(t *testing.T) {
	rb := newRoutedBroker(t)
	result, errText := rb.relay(t, "set-availability", map[string]any{"state": "paused"})
	if errText != "" {
		t.Fatalf("relayed set-availability: %s", errText)
	}
	var relayed map[string]json.RawMessage
	if err := json.Unmarshal(result, &relayed); err != nil {
		t.Fatalf("decode relayed set-availability %s: %v", result, err)
	}
	if len(relayed) != 1 || string(relayed["availability"]) != `"paused"` {
		t.Fatalf("relayed set-availability = %s, want only {\"availability\":\"paused\"}", result)
	}

	rb.b.handleMessage(clientRequest(7, methodSetAvailability, map[string]any{"state": "available"}))
	m := rb.out.waitFrame(t, "the local set-availability answer", isResponse(7))
	var local map[string]json.RawMessage
	if err := json.Unmarshal(m.Result, &local); err != nil {
		t.Fatalf("decode local set-availability %s: %v", m.Result, err)
	}
	for _, key := range []string{"availability", "revision", "activeRevision", "activation"} {
		if _, ok := local[key]; !ok {
			t.Errorf("local set-availability lost %q: %s", key, m.Result)
		}
	}
}

func TestLocalPolicyGetKeepsTheFullDocument(t *testing.T) {
	rb := newRoutedBroker(t)
	rb.b.handleMessage(clientRequest(1, methodPolicyGet, nil))
	resp := rb.out.waitFrame(t, "policy:get response", isResponse(1))
	if resp.Error != nil {
		t.Fatalf("policy:get: %+v", resp.Error)
	}
	var got policyGetResult
	if err := json.Unmarshal(resp.Result, &got); err != nil {
		t.Fatalf("decode policy:get: %v", err)
	}
	if len(got.Policy.Routing.Accounts) != 2 || got.Policy.Routing.Accounts[0].ID != "free-acct" {
		t.Errorf("local policy:get routing accounts = %+v, want the owner's full document", got.Policy.Routing.Accounts)
	}
	if len(got.Policy.Tiers["strong"]) != 1 || got.Policy.Tiers["strong"][0].Kind != nodepolicy.EntryAPI {
		t.Errorf("local policy:get strong tier = %+v, want the API entry", got.Policy.Tiers["strong"])
	}
}

func TestPeerPolicySetRoundTripKeepsProtectedSections(t *testing.T) {
	rb := newRoutedBroker(t)
	beforeRouting := mustJSON(rb.b.localPolicy().Policy.Routing)
	beforeTiers := mustJSON(rb.b.localPolicy().Policy.Tiers)

	doc := rb.peerSnapshot(t)
	doc["idle"].(map[string]any)["unloadAfterMinutes"] = 7
	result, errText := rb.relaySet(t, doc)
	if errText != "" {
		t.Fatalf("relayed set of an unchanged projection: %s", errText)
	}
	for _, leak := range routingLeaks {
		if strings.Contains(string(result), leak) {
			t.Errorf("relayed set result contains %q: %s", leak, result)
		}
	}

	sections := rb.fileSections(t)
	var gotRouting, gotTiers bytes.Buffer
	if err := json.Compact(&gotRouting, sections["routing"]); err != nil {
		t.Fatalf("compact routing: %v", err)
	}
	if err := json.Compact(&gotTiers, sections["tiers"]); err != nil {
		t.Fatalf("compact tiers: %v", err)
	}
	if gotRouting.String() != string(beforeRouting) {
		t.Errorf("stored routing changed:\n got %s\nwant %s", gotRouting.String(), beforeRouting)
	}
	if gotTiers.String() != string(beforeTiers) {
		t.Errorf("stored tiers changed (API entries must survive):\n got %s\nwant %s", gotTiers.String(), beforeTiers)
	}
	var idle nodepolicy.IdleConfig
	if err := json.Unmarshal(sections["idle"], &idle); err != nil || idle.UnloadAfterMinutes != 7 {
		t.Errorf("stored idle = %+v (%v), want the peer's device edit", idle, err)
	}

	calls := rb.proxy.waitFor("node/set-policy push", func(calls []policyCall) bool { return len(rb.proxy.methods(nodepolicy.MethodSetPolicy)) > 0 })
	var pushed nodepolicy.Policy
	for _, c := range calls {
		if c.method == nodepolicy.MethodSetPolicy {
			if err := json.Unmarshal(c.params, &pushed); err != nil {
				t.Fatalf("decode pushed policy: %v", err)
			}
		}
	}
	if string(mustJSON(pushed.Routing)) != string(beforeRouting) || pushed.Idle.UnloadAfterMinutes != 7 {
		t.Errorf("the proxy was pushed routing %s and idle %+v, want the preserved routing and the edit", mustJSON(pushed.Routing), pushed.Idle)
	}
}

func TestPeerPolicySetRejectsProtectedChangesWithoutEffect(t *testing.T) {
	test := func(name string, edit func(doc map[string]any)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			beforeFile := rb.fileBytes(t)
			before := rb.b.localPolicy().Policy
			doc := rb.peerSnapshot(t)
			edit(doc)

			// Holding the push slot keeps a scheduled push marked pending, so a
			// push that a rejected set wrongly scheduled is visible without waiting.
			r := rb.b.nodePolicyRuntime()
			r.acquirePush(context.Background())
			result, errText := rb.relaySet(t, doc)
			r.mu.Lock()
			pending := len(r.pendingPush)
			r.mu.Unlock()
			r.releasePush()

			if errText == "" {
				t.Fatalf("relayed set accepted a protected change: %s", result)
			}
			if strings.Contains(errText, "NVPAIR_PROVIDER_SECRET") {
				t.Errorf("rejection echoes a credential reference: %q", errText)
			}
			if got := rb.fileBytes(t); !bytes.Equal(got, beforeFile) {
				t.Errorf("policy file changed after a rejected set")
			}
			if got := rb.b.localPolicy().Policy; !reflect.DeepEqual(got, before) {
				t.Errorf("in-memory policy changed after a rejected set")
			}
			if pending != 0 || policyPushes(rb.proxy) != 0 {
				t.Errorf("a rejected set scheduled %d pushes and sent %d to the proxy", pending, policyPushes(rb.proxy))
			}
			for _, m := range rb.out.frames() {
				if m.Method == notifyPolicyChanged {
					t.Errorf("a rejected set notified clients: %s", m.Params)
				}
			}
		})
	}

	test("changed tier entry", func(doc map[string]any) {
		doc["tiers"].(map[string]any)["weak"] = []any{map[string]any{"kind": "local", "engine": "ollama", "model": "llama3:8b"}}
	})
	test("added tier entry", func(doc map[string]any) {
		doc["tiers"].(map[string]any)["medium"] = []any{map[string]any{"kind": "local", "engine": "ollama", "model": "llama3:8b"}}
	})
	test("changed tier policy", func(doc map[string]any) {
		doc["tierPolicy"].(map[string]any)["allowWeaker"] = true
	})
	test("injected routing section", func(doc map[string]any) {
		doc["routing"] = map[string]any{"limits": map[string]any{"maxAttempts": 10}}
	})
	test("injected capitalized routing section", func(doc map[string]any) {
		doc["Routing"] = map[string]any{"limits": map[string]any{"maxAttempts": 10}}
	})
	test("injected account", func(doc map[string]any) {
		doc["routing"] = map[string]any{"accounts": []any{map[string]any{
			"id": "extra", "credential": "NVPAIR_PROVIDER_SECRET_EXTRA", "billing": "free",
			"perRequestCeiling": "0", "dailyCeiling": "0", "monthlyCeiling": "0",
		}}}
	})
	test("injected api tier entry", func(doc map[string]any) {
		doc["tiers"].(map[string]any)["weak"] = []any{
			map[string]any{"kind": "local", "engine": "ollama", "model": "qwen3:8b"},
			map[string]any{"kind": "api", "deployment": "paid-big"},
		}
	})
}

func TestPeerPolicySetKeepsLocalChangesMadeBeforeItsSnapshot(t *testing.T) {
	rb := newRoutedBroker(t)
	rb.localSet(t, func(doc map[string]any) {
		accounts := doc["routing"].(map[string]any)["accounts"].([]any)
		accounts[1].(map[string]any)["monthlyCeiling"] = "5"
	})
	fresh := rb.peerSnapshot(t)
	fresh["idle"].(map[string]any)["unloadAfterMinutes"] = 9
	if _, errText := rb.relaySet(t, fresh); errText != "" {
		t.Fatalf("relayed set of a fresh projection: %s", errText)
	}
	got := rb.b.localPolicy().Policy
	if got.Routing.Accounts[1].MonthlyCeiling != "5" {
		t.Errorf("monthly ceiling = %q, want the local owner's lowered %q kept", got.Routing.Accounts[1].MonthlyCeiling, "5")
	}
	if got.Idle.UnloadAfterMinutes != 9 {
		t.Errorf("idle.unloadAfterMinutes = %d, want the peer's device edit", got.Idle.UnloadAfterMinutes)
	}
	var onDisk nodepolicy.Policy
	if err := json.Unmarshal(rb.fileBytes(t), &onDisk); err != nil || onDisk.Routing.Accounts[1].MonthlyCeiling != "5" {
		t.Errorf("stored monthly ceiling = %q (%v), want %q", onDisk.Routing.Accounts[1].MonthlyCeiling, err, "5")
	}
}

// noEffectBaseline records what a rejected set must leave as it was: the stored
// bytes, the revision, the in-memory policy, the proxy push count and the
// clients' announcements.
type noEffectBaseline struct {
	file     []byte
	policy   nodepolicy.Policy
	pushes   int
	changed  int
	rb       *routedBroker
	revision string
}

func (rb *routedBroker) baseline(t *testing.T) noEffectBaseline {
	t.Helper()
	policy := rb.b.localPolicy().Policy
	return noEffectBaseline{file: rb.fileBytes(t), policy: policy, revision: policy.Revision, pushes: policyPushes(rb.proxy), changed: len(policyChangedRevisions(t, rb.out)), rb: rb}
}

func (n noEffectBaseline) requireUnchanged(t *testing.T) {
	t.Helper()
	rb := n.rb
	if got := rb.fileBytes(t); !bytes.Equal(got, n.file) {
		t.Errorf("policy file changed after a rejected set")
	}
	got := rb.b.localPolicy().Policy
	if !reflect.DeepEqual(got, n.policy) || got.Revision != n.revision {
		t.Errorf("in-memory policy changed after a rejected set: revision %q, want %q", got.Revision, n.revision)
	}
	r := rb.b.nodePolicyRuntime()
	r.mu.Lock()
	pending := len(r.pendingPush)
	r.mu.Unlock()
	if pending != 0 || policyPushes(rb.proxy) != n.pushes {
		t.Errorf("a rejected set scheduled %d pushes and sent %d (was %d) to the proxy", pending, policyPushes(rb.proxy), n.pushes)
	}
	if got := len(policyChangedRevisions(t, rb.out)); got != n.changed {
		t.Errorf("a rejected set announced policy:changed %d times (was %d)", got, n.changed)
	}
}

func TestPeerPolicySetFromAStaleSnapshotIsAConflict(t *testing.T) {
	t.Run("local routing change is kept", func(t *testing.T) {
		rb := newRoutedBroker(t)
		stale := rb.peerSnapshot(t)
		rb.localSet(t, func(doc map[string]any) {
			accounts := doc["routing"].(map[string]any)["accounts"].([]any)
			accounts[1].(map[string]any)["monthlyCeiling"] = "5"
		})
		persisted := rb.b.localPolicy().Policy.Revision
		base := rb.baseline(t)
		stale["idle"].(map[string]any)["unloadAfterMinutes"] = 9
		result, errText := rb.relaySet(t, stale)
		if !strings.HasPrefix(errText, "policy conflict:") || !strings.Contains(errText, persisted) {
			t.Fatalf("stale relayed set = %q (result %s), want a policy conflict naming revision %s", errText, result, persisted)
		}
		base.requireUnchanged(t)
		if got := rb.b.localPolicy().Policy.Routing.Accounts[1].MonthlyCeiling; got != "5" {
			t.Errorf("monthly ceiling = %q, want the local owner's lowered %q kept", got, "5")
		}
	})

	t.Run("local tier change rejects the stale tiers", func(t *testing.T) {
		rb := newRoutedBroker(t)
		stale := rb.peerSnapshot(t)
		rb.localSet(t, func(doc map[string]any) {
			doc["tiers"].(map[string]any)["weak"] = []any{map[string]any{"kind": "local", "engine": "ollama", "model": "llama3:8b"}}
		})
		base := rb.baseline(t)
		stale["idle"].(map[string]any)["unloadAfterMinutes"] = 9
		if result, errText := rb.relaySet(t, stale); !strings.HasPrefix(errText, "policy conflict:") {
			t.Fatalf("a stale snapshot restored the old tiers: %q %s", errText, result)
		}
		base.requireUnchanged(t)
		want := []nodepolicy.TierEntry{{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "llama3:8b"}}
		if got := rb.b.localPolicy().Policy.Tiers["weak"]; !reflect.DeepEqual(got, want) {
			t.Errorf("weak tier = %+v, want the local owner's %+v", got, want)
		}
	})

	t.Run("a stale projection cannot mutate protected fields", func(t *testing.T) {
		rb := newRoutedBroker(t)
		stale := rb.peerSnapshot(t)
		rb.localSet(t, func(doc map[string]any) {
			doc["idle"].(map[string]any)["unloadAfterMinutes"] = 11
		})
		base := rb.baseline(t)
		stale["tierPolicy"].(map[string]any)["allowWeaker"] = true
		if result, errText := rb.relaySet(t, stale); !strings.HasPrefix(errText, "policy conflict:") {
			t.Fatalf("stale protected edit = %q %s, want a policy conflict", errText, result)
		}
		base.requireUnchanged(t)
	})
}

func TestPolicySetWithoutAPreconditionIsRefused(t *testing.T) {
	cases := map[string]json.RawMessage{
		"missing":      nil,
		"null":         json.RawMessage(`null`),
		"empty":        json.RawMessage(`""`),
		"number":       json.RawMessage(`12`),
		"leading zero": json.RawMessage(`"012"`),
		"signed":       json.RawMessage(`"-1"`),
		"overflow":     json.RawMessage(`"9223372036854775808"`),
		"padded":       json.RawMessage(`" 12"`),
	}
	for name, expected := range cases {
		t.Run(name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			base := rb.baseline(t)
			params := map[string]any{"policy": mustJSON(rb.b.localPolicy().Policy)}
			if expected != nil {
				params["expectedPersistedRevision"] = expected
			}
			rb.b.handleMessage(clientRequest(1, methodPolicySet, params))
			resp := rb.out.waitFrame(t, "policy:set response", isResponse(1))
			if resp.Error == nil || resp.Error.Code != -32602 || !strings.Contains(resp.Error.Message, "expectedPersistedRevision") {
				t.Fatalf("local set with a %s precondition = %+v, want invalid params naming expectedPersistedRevision", name, resp)
			}
			base.requireUnchanged(t)

			peerParams := map[string]any{"policy": rb.peerSnapshot(t)}
			if expected != nil {
				peerParams["expectedPersistedRevision"] = expected
			}
			if result, errText := rb.relay(t, "set", peerParams); !strings.Contains(errText, "expectedPersistedRevision") {
				t.Fatalf("relayed set with a %s precondition = %q %s, want it refused naming expectedPersistedRevision", name, errText, result)
			}
			base.requireUnchanged(t)
		})
	}
}

func TestPolicySetConflictIsNotInvalidParams(t *testing.T) {
	rb := newRoutedBroker(t)
	base := rb.baseline(t)
	// The fixture's persisted revision is 12.
	rb.b.handleMessage(clientRequest(1, methodPolicySet, map[string]any{"policy": mustJSON(rb.b.localPolicy().Policy), "expectedPersistedRevision": "11"}))
	resp := rb.out.waitFrame(t, "policy:set response", isResponse(1))
	if resp.Error == nil || resp.Error.Code == -32602 || !strings.HasPrefix(resp.Error.Message, "policy conflict:") || !strings.Contains(resp.Error.Message, "12") {
		t.Fatalf("stale local set = %+v, want a policy conflict naming revision 12 that is not invalid params", resp)
	}
	base.requireUnchanged(t)
}

func TestRelayedPolicySetWithoutExpectedPersistedRevisionIsInvalidParams(t *testing.T) {
	cases := []struct {
		name     string
		expected json.RawMessage
	}{
		{"missing", nil},
		{"null", json.RawMessage(`null`)},
		{"empty", json.RawMessage(`""`)},
		{"number", json.RawMessage(`12`)},
		{"noncanonical", json.RawMessage(`"012"`)},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			params := map[string]any{"policy": rb.peerSnapshot(t)}
			if c.expected != nil {
				params["expectedPersistedRevision"] = c.expected
			}
			base := rb.baseline(t)
			reply := rb.relayReply(t, "set", params)
			if reply.ErrorCode != -32602 || !strings.Contains(reply.Error, "expectedPersistedRevision") {
				t.Fatalf("relayed %s precondition: code %d, error %q; want -32602 naming expectedPersistedRevision", c.name, reply.ErrorCode, reply.Error)
			}
			base.requireUnchanged(t)
		})
	}
}

func TestStaleRemoteAllowedDraftIsConflictBeforeValidation(t *testing.T) {
	for _, source := range []string{"local", "peer"} {
		t.Run(source, func(t *testing.T) {
			rb := newRoutedBroker(t)
			var draft nodepolicy.Policy
			if err := json.Unmarshal(mustJSON(rb.b.localPolicy().Policy), &draft); err != nil {
				t.Fatalf("copy client A's policy: %v", err)
			}
			// Client A has a dirty draft enabling execution. Phase 1 refuses
			// that draft even at the current revision. Client B saves both
			// permissions off before A submits the old draft.
			for tier := range draft.Routing.Tiers {
				settings := draft.Routing.Tiers[tier]
				settings.RemoteAllowed = true
				settings.PaidAllowed = true
				draft.Routing.Tiers[tier] = settings
			}
			if err := draft.Validate(); err == nil {
				t.Fatal("dirty remote/paid draft must be invalid in Phase 1")
			}
			rb.localSet(t, func(doc map[string]any) {
				for _, settings := range doc["routing"].(map[string]any)["tiers"].(map[string]any) {
					settings.(map[string]any)["remoteAllowed"] = false
					settings.(map[string]any)["paidAllowed"] = false
				}
				doc["idle"].(map[string]any)["unloadAfterMinutes"] = 3
			})
			base := rb.baseline(t)
			params := map[string]any{"policy": draft, "expectedPersistedRevision": draft.Revision}
			var code int
			var message string
			if source == "local" {
				rb.b.handleMessage(clientRequest(1, methodPolicySet, params))
				resp := rb.out.waitFrame(t, "stale policy:set response", isResponse(1))
				if resp.Error == nil {
					t.Fatal("stale local draft was accepted")
				}
				code, message = resp.Error.Code, resp.Error.Message
			} else {
				reply := rb.relayReply(t, "set", params)
				code, message = reply.ErrorCode, reply.Error
			}
			if code != codePolicyConflict || !strings.HasPrefix(message, "policy conflict:") || !strings.Contains(message, base.revision) {
				t.Fatalf("stale dirty draft: code %d, error %q; want %d conflict naming revision %s", code, message, codePolicyConflict, base.revision)
			}
			base.requireUnchanged(t)
			for tier, settings := range rb.b.localPolicy().Policy.Routing.Tiers {
				if settings.RemoteAllowed || settings.PaidAllowed {
					t.Errorf("tier %s restored remote or paid permission", tier)
				}
			}
		})
	}
}

func TestRelayedPolicyValidationErrorsAreInvalidParams(t *testing.T) {
	test := func(name, method string, params func(*testing.T, *routedBroker) any) {
		t.Run(name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			body := params(t, rb)
			base := rb.baseline(t)
			reply := rb.relayReply(t, method, body)
			if reply.ErrorCode != -32602 || reply.Error == "" {
				t.Fatalf("code %d, error %q; want -32602 validation error", reply.ErrorCode, reply.Error)
			}
			base.requireUnchanged(t)
		})
	}
	test("malformed params", "set", func(*testing.T, *routedBroker) any { return []string{"invalid"} })
	test("invalid availability", "set-availability", func(*testing.T, *routedBroker) any { return map[string]any{"state": "invalid"} })
	test("missing policy", "set", func(*testing.T, *routedBroker) any { return map[string]any{"expectedPersistedRevision": "12"} })
	test("protected routing", "set", func(t *testing.T, rb *routedBroker) any {
		doc := rb.peerSnapshot(t)
		doc["routing"] = map[string]any{}
		return map[string]any{"policy": doc, "expectedPersistedRevision": "12"}
	})
}

func TestRelayedPolicySaveFailuresAreServerErrors(t *testing.T) {
	for _, boundary := range publicationBoundaries() {
		t.Run(boundary.name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			doc := rb.peerSnapshot(t)
			doc["idle"].(map[string]any)["unloadAfterMinutes"] = 7
			setClock(rb.b, 5)
			setReplacer(rb.b, boundary.rep)
			reply := rb.relayReply(t, "set", map[string]any{"policy": doc, "expectedPersistedRevision": "12"})
			want := "the policy could not be saved: "
			if boundary.landed {
				want = "the policy document with revision 13 was saved and is being applied but may not survive a power loss: "
			}
			if reply.ErrorCode != -32000 || !strings.HasPrefix(reply.Error, want) || !strings.Contains(reply.Error, errPublish.Error()) {
				t.Fatalf("code %d, error %q; want -32000 with %q and publication cause", reply.ErrorCode, reply.Error, want)
			}
		})
	}
}

// Two clients read revision 12. A lowers the paid monthly ceiling and the node
// policy; B, still on its draft of revision 12, tries to raise the ceiling back.
func TestStaleLocalDraftCannotOverwriteANewerPolicy(t *testing.T) {
	rb := newRoutedBroker(t)
	draft := mustJSON(rb.b.localPolicy().Policy)
	baseRevision := mustJSON(rb.b.localPolicy().Policy.Revision)

	rb.localSet(t, func(doc map[string]any) {
		accounts := doc["routing"].(map[string]any)["accounts"].([]any)
		accounts[1].(map[string]any)["monthlyCeiling"] = "5"
		doc["idle"].(map[string]any)["unloadAfterMinutes"] = 3
	})
	base := rb.baseline(t)

	var stale map[string]any
	if err := json.Unmarshal(draft, &stale); err != nil {
		t.Fatal(err)
	}
	stale["routing"].(map[string]any)["accounts"].([]any)[1].(map[string]any)["monthlyCeiling"] = "9000"
	_, err := rb.b.setLocalPolicy(mustJSON(stale), baseRevision)
	var conflict *policyConflictError
	if !errors.As(err, &conflict) || !strings.HasPrefix(err.Error(), "policy conflict:") || !strings.Contains(err.Error(), base.revision) {
		t.Fatalf("stale local set error = %v, want a policy conflict naming revision %s", err, base.revision)
	}
	base.requireUnchanged(t)
	got := rb.b.localPolicy().Policy
	if got.Routing.Accounts[1].MonthlyCeiling != "5" || got.Idle.UnloadAfterMinutes != 3 {
		t.Errorf("after the stale set the ceiling is %q and idle %d, want client A's %q and 3", got.Routing.Accounts[1].MonthlyCeiling, got.Idle.UnloadAfterMinutes, "5")
	}
}

// Writers holding the same revision race: one passes the check and the rest
// see its revision, so the stored policy is exactly one writer's.
func TestConcurrentSetsWithTheSameRevisionAdmitExactlyOne(t *testing.T) {
	rb := newRoutedBroker(t)
	base := rb.b.localPolicy().Policy.Revision
	const writers = 8
	snapshot := mustJSON(rb.peerSnapshot(t))
	var wg sync.WaitGroup
	results := make([]error, writers)
	start := make(chan struct{})
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			if i%2 == 0 {
				doc := map[string]any{"version": nodepolicy.SchemaVersion, "idle": map[string]any{"unloadAfterMinutes": i + 1}}
				_, results[i] = rb.b.setLocalPolicy(mustJSON(doc), mustJSON(base))
				return
			}
			var peer map[string]any
			if err := json.Unmarshal(snapshot, &peer); err != nil {
				results[i] = err
				return
			}
			peer["idle"].(map[string]any)["unloadAfterMinutes"] = i + 1
			_, results[i] = rb.b.setPeerPolicy(mustJSON(peer), mustJSON(base))
		}()
	}
	close(start)
	wg.Wait()
	passed := 0
	for i, err := range results {
		switch {
		case err == nil:
			passed++
		case !strings.HasPrefix(err.Error(), "policy conflict:"):
			t.Errorf("writer %d failed with %v, want success or a policy conflict", i, err)
		}
	}
	if passed != 1 {
		t.Fatalf("%d writers passed the check against revision %s, want exactly one", passed, base)
	}
}

// errPublish is the failure the publication tests inject at a durablefile step.
var errPublish = errors.New("injected publication failure")

// faultTemp is the temporary file durablefile writes, failing the chosen step.
// Close always releases the descriptor.
type faultTemp struct {
	durablefile.File
	writeErr, syncErr, closeErr error
	afterSyncErr                error
}

func (f faultTemp) Write(p []byte) (int, error) {
	if f.writeErr != nil {
		return 0, f.writeErr
	}
	return f.File.Write(p)
}

func (f faultTemp) Sync() error {
	if f.syncErr != nil {
		return f.syncErr
	}
	if err := f.File.Sync(); err != nil {
		return err
	}
	return f.afterSyncErr
}

func (f faultTemp) Close() error {
	err := f.File.Close()
	if f.closeErr != nil {
		return f.closeErr
	}
	return err
}

// faultingTemp is a Replacer that fails the temporary file's step f selects.
func faultingTemp(f faultTemp) durablefile.Replacer {
	return durablefile.Replacer{CreateTemp: func(dir, base string, perm os.FileMode) (durablefile.File, error) {
		file, err := os.OpenFile(filepath.Join(dir, base+".fault.tmp"), os.O_RDWR|os.O_CREATE|os.O_EXCL, perm)
		if err != nil {
			return nil, err
		}
		f.File = file
		return f, nil
	}}
}

// setReplacer swaps the step functions the policy store publishes with.
func setReplacer(b *Broker, rep durablefile.Replacer) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.replacer = rep
	r.mu.Unlock()
}

// setClock fixes the time revisions are assigned from.
func setClock(b *Broker, unixNano int64) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.now = func() time.Time { return time.Unix(0, unixNano) }
	r.mu.Unlock()
}

// freshBroker is a broker started on the files the previous one left behind.
func freshBroker(t *testing.T, path string) *Broker {
	t.Helper()
	b, _ := newPolicyBroker(t)
	b.loadNodePolicy(path)
	return b
}

func revisionNumber(t *testing.T, p nodepolicy.Policy) int64 {
	t.Helper()
	n, err := strconv.ParseInt(p.Revision, 10, 64)
	if err != nil {
		t.Fatalf("revision %q is not a decimal number: %v", p.Revision, err)
	}
	return n
}

// paidCeilingDoc is the broker's current policy with the paid account's
// monthly ceiling replaced, as a policy:set body.
func paidCeilingDoc(t *testing.T, b *Broker, ceiling string) json.RawMessage {
	t.Helper()
	var doc map[string]any
	if err := json.Unmarshal(mustJSON(b.localPolicy().Policy), &doc); err != nil {
		t.Fatalf("decode current policy: %v", err)
	}
	accounts, ok := doc["routing"].(map[string]any)["accounts"].([]any)
	if !ok || len(accounts) != 2 {
		t.Fatalf("fixture accounts = %v, want the 2 in routedPolicyDocument", accounts)
	}
	accounts[1].(map[string]any)["monthlyCeiling"] = ceiling
	return mustJSON(doc)
}

// policyChangedRevisions lists the revision of every policy:changed this node
// announced for itself, in order.
func policyChangedRevisions(t *testing.T, out *syncBuffer) []string {
	t.Helper()
	var revisions []string
	for _, m := range out.frames() {
		if m.Method != notifyPolicyChanged {
			continue
		}
		var event struct {
			NodeID string            `json:"nodeId"`
			Policy nodepolicy.Policy `json:"policy"`
		}
		if err := json.Unmarshal(m.Params, &event); err != nil {
			t.Fatalf("decode policy:changed: %v", err)
		}
		if event.NodeID == "self-node" {
			revisions = append(revisions, event.Policy.Revision)
		}
	}
	return revisions
}

// publicationBoundary is one durablefile step that can fail. landed is whether
// the new bytes are in place when it does: only a directory-sync failure comes
// after the replacement.
type publicationBoundary struct {
	name   string
	rep    durablefile.Replacer
	landed bool
}

func publicationBoundaries() []publicationBoundary {
	return []publicationBoundary{
		{"temp create", durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) { return nil, errPublish }}, false},
		{"temp write", faultingTemp(faultTemp{writeErr: errPublish}), false},
		{"file sync", faultingTemp(faultTemp{syncErr: errPublish}), false},
		{"file close", faultingTemp(faultTemp{closeErr: errPublish}), false},
		{"replace", durablefile.Replacer{Replace: func(string, string) error { return errPublish }}, false},
		{"directory sync", durablefile.Replacer{SyncDir: func(string) error { return errPublish }}, true},
	}
}

func publicationBoundariesWithFlushes() []publicationBoundary {
	return append(publicationBoundaries(),
		publicationBoundary{"after file sync", faultingTemp(faultTemp{afterSyncErr: errPublish}), false},
		publicationBoundary{"after directory or platform flush", durablefile.Replacer{SyncDir: failAfterPolicyDirectoryFlush}, true},
	)
}

// requireLandedPolicy checks the complete document at every publication surface.
func requireLandedPolicy(t *testing.T, rb *routedBroker, want nodepolicy.Policy) {
	t.Helper()
	if got := rb.b.localPolicy(); got.RecoveryRequired || !reflect.DeepEqual(got.Policy, want) {
		t.Errorf("landed replacement was not adopted in memory at revision %s (recovery=%t)", want.Revision, got.RecoveryRequired)
	}
	stored, err := nodepolicy.Parse(rb.fileBytes(t))
	if err != nil {
		t.Fatalf("parse landed file: %v", err)
	}
	if !reflect.DeepEqual(stored, want) {
		t.Errorf("landed file differs from the complete expected document at revision %s", want.Revision)
	}
	if got := freshBroker(t, rb.path).localPolicy(); got.RecoveryRequired || !reflect.DeepEqual(got.Policy, want) {
		t.Errorf("restart differs from the complete expected document at revision %s (recovery=%t)", want.Revision, got.RecoveryRequired)
	}
	var pushed nodepolicy.Policy
	for _, call := range rb.proxy.snapshot() {
		if call.method == nodepolicy.MethodSetPolicy {
			// Decode each push whole: an omitted pausedBy must not keep the
			// previous push's value.
			pushed = nodepolicy.Policy{}
			if err := json.Unmarshal(call.params, &pushed); err != nil {
				t.Fatalf("decode pushed policy: %v", err)
			}
		}
	}
	if !reflect.DeepEqual(pushed, want) {
		t.Errorf("proxy was not pushed the complete landed document at revision %s", want.Revision)
	}
	announced, _ := lastPolicyChanged(t, rb.out)
	if !reflect.DeepEqual(announced, want) {
		t.Errorf("policy:changed did not announce the complete landed document at revision %s", want.Revision)
	}
}

func TestPolicyPublicationFailureLeavesOneWholeDocument(t *testing.T) {
	for _, boundary := range publicationBoundaries() {
		name, rep, landed := boundary.name, boundary.rep, boundary.landed
		t.Run(name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			setClock(rb.b, 5)
			if _, err := setCurrent(rb.b, paidCeilingDoc(t, rb.b, "5")); err != nil {
				t.Fatalf("acknowledged ceiling reduction: %v", err)
			}
			acked := rb.b.localPolicy().Policy
			if got := acked.Routing.Accounts[1].MonthlyCeiling; got != "5" {
				t.Fatalf("acknowledged ceiling = %q, want %q", got, "5")
			}
			ackedBytes := rb.fileBytes(t)
			rb.proxy.waitFor("push of the acknowledged policy", func(calls []policyCall) bool { return policyPushes(rb.proxy) >= 1 })
			pushes := policyPushes(rb.proxy)
			changed := policyChangedRevisions(t, rb.out)

			raise := paidCeilingDoc(t, rb.b, "9")
			setReplacer(rb.b, rep)
			_, err := setCurrent(rb.b, raise)
			setReplacer(rb.b, durablefile.Replacer{})
			if !errors.Is(err, errPublish) {
				t.Fatalf("setLocalPolicy error = %v, want it to wrap the injected failure", err)
			}
			if landed {
				want, err := nodepolicy.Parse(raise)
				if err != nil {
					t.Fatalf("parse the attempted document: %v", err)
				}
				want.Revision = "14"
				requireLandedPolicy(t, rb, want)
				return
			}

			if got := rb.b.localPolicy().Policy; !reflect.DeepEqual(got, acked) {
				t.Errorf("failed publication changed the in-memory policy: revision %s -> %s", acked.Revision, got.Revision)
			}
			time.Sleep(100 * time.Millisecond)
			if got := policyPushes(rb.proxy); got != pushes {
				t.Errorf("failed publication pushed to the proxy: %d pushes, want %d", got, pushes)
			}
			if got := policyChangedRevisions(t, rb.out); !reflect.DeepEqual(got, changed) {
				t.Errorf("failed publication announced policy:changed %v, want %v", got, changed)
			}
			if leftovers, _ := filepath.Glob(filepath.Join(filepath.Dir(rb.path), "*.tmp")); len(leftovers) != 0 {
				t.Errorf("failed publication left temporary files: %v", leftovers)
			}

			fresh := freshBroker(t, rb.path).localPolicy()
			if fresh.RecoveryRequired {
				t.Fatalf("restart after a failed publication needs recovery")
			}
			if !reflect.DeepEqual(fresh.Policy, acked) {
				t.Errorf("restart loaded a document other than the acknowledged one (revision %s, monthly ceiling %q)",
					fresh.Policy.Revision, fresh.Policy.Routing.Accounts[1].MonthlyCeiling)
			}
			if got := rb.fileBytes(t); !bytes.Equal(got, ackedBytes) {
				t.Errorf("policy file bytes changed although the replacement never happened")
			}
		})
	}
}

func TestPolicyLandedReplacementReconcilesProxySupervision(t *testing.T) {
	rb := newRoutedBroker(t)
	setClock(rb.b, 5)
	// An unfinished bring-up defers reconciliation without starting a process.
	rb.b.proxySupMu.Lock()
	rb.b.proxySupPhase = proxySupStarting
	rb.b.proxySupMu.Unlock()
	want := rb.b.localPolicy().Policy
	want.Routing.Listener.Enabled = false
	want.Revision = "13"
	setReplacer(rb.b, durablefile.Replacer{SyncDir: func(string) error { return errPublish }})
	_, err := setCurrent(rb.b, mustJSON(want))
	setReplacer(rb.b, durablefile.Replacer{})
	if !errors.Is(err, errPublish) {
		t.Fatalf("listener set error = %v, want injected durability failure", err)
	}
	requireLandedPolicy(t, rb, want)
	rb.b.proxySupWorkMu.Lock()
	dirty := rb.b.proxySupDirty
	rb.b.proxySupWorkMu.Unlock()
	if !dirty {
		t.Fatal("landed listener change did not schedule proxy supervision reconciliation")
	}
}

func TestAvailabilityPublicationFailureLeavesOneWholeDocument(t *testing.T) {
	transitions := []struct {
		name string
		// from is the availability reached, acknowledged, before the failing
		// attempt; to is what the attempt asks for. Revisions come from a fixed
		// clock, so each save is the previous revision plus one.
		from, to            nodepolicy.Availability
		baseline, attempted string
	}{
		{"pause", nodepolicy.Available, nodepolicy.Paused, "12", "13"},
		{"resume", nodepolicy.Paused, nodepolicy.Available, "13", "14"},
	}
	for _, transition := range transitions {
		for _, boundary := range publicationBoundariesWithFlushes() {
			from, to, baseline, attempted := transition.from, transition.to, transition.baseline, transition.attempted
			rep, landed := boundary.rep, boundary.landed
			t.Run(transition.name+" fails at "+boundary.name, func(t *testing.T) {
				rb := newRoutedBroker(t)
				setClock(rb.b, 5)
				rb.b.replayPolicyToProxy(context.Background(), rb.b.policyProxy())
				if from == nodepolicy.Paused {
					if _, err := rb.b.setAvailability(context.Background(), from, nodepolicy.PausedLocally); err != nil {
						t.Fatalf("acknowledged %s: %v", from, err)
					}
				}
				before := rb.b.localPolicy()
				if before.Policy.Revision != baseline || before.Availability != from {
					t.Fatalf("baseline: revision %q live %q, want revision %q live %q", before.Policy.Revision, before.Availability, baseline, from)
				}
				beforeBytes := rb.fileBytes(t)
				callStart, frameStart := len(rb.proxy.snapshot()), len(rb.out.frames())
				pushes := len(availabilityPushes(rb.proxy.snapshot()))
				policyCount := policyPushes(rb.proxy)
				changed := policyChangedRevisions(t, rb.out)

				setReplacer(rb.b, rep)
				_, err := rb.b.setAvailability(context.Background(), to, nodepolicy.PausedLocally)
				setReplacer(rb.b, durablefile.Replacer{})
				if !errors.Is(err, errPublish) {
					t.Fatalf("setAvailability(%s) error = %v, want it to wrap the injected failure", to, err)
				}
				if leftovers, err := filepath.Glob(filepath.Join(filepath.Dir(rb.path), "*.tmp")); err != nil || len(leftovers) != 0 {
					t.Errorf("failed save left temporary files: %v (glob error: %v)", leftovers, err)
				}
				if landed {
					want := before.Policy
					want.Availability, want.Revision = to, attempted
					want.PausedBy = pausedByFor(to, nodepolicy.PausedLocally)
					requireLandedAvailability(t, rb, want, callStart, frameStart)
					var save *policySaveError
					if !errors.As(err, &save) || save.revision != attempted {
						t.Errorf("landed availability error = %v, want save warning for revision %s", err, attempted)
					}
					if _, err := rb.b.setAvailability(context.Background(), to, nodepolicy.PausedLocally); err != nil {
						t.Fatalf("repeat of completed %s: %v", to, err)
					}
					if got := rb.b.localPolicy().Policy.Revision; got != attempted {
						t.Errorf("repeat of completed transition changed revision to %s, want %s", got, attempted)
					}
					return
				}

				if after := rb.b.localPolicy(); !reflect.DeepEqual(after, before) {
					t.Errorf("failed save changed the state: revision %s -> %s, live %q -> %q",
						before.Policy.Revision, after.Policy.Revision, before.Availability, after.Availability)
				}
				time.Sleep(100 * time.Millisecond)
				if got := len(availabilityPushes(rb.proxy.snapshot())); got != pushes {
					t.Errorf("failed save pushed availability to the proxy: %d pushes, want %d", got, pushes)
				}
				if policyPushes(rb.proxy) != policyCount || len(rb.out.frames()) != frameStart {
					t.Error("failure before replacement pushed a policy or announced an availability transition")
				}
				if got := policyChangedRevisions(t, rb.out); !reflect.DeepEqual(got, changed) {
					t.Errorf("failed save announced policy:changed %v, want %v", got, changed)
				}
				fresh := freshBroker(t, rb.path).localPolicy()
				if fresh.RecoveryRequired {
					t.Fatal("restart after a failed save needs recovery")
				}
				want := before.Policy
				if !reflect.DeepEqual(fresh.Policy, want) || fresh.Availability != want.Availability {
					t.Errorf("restart loaded revision %q availability %q (live %q), want the wholly old document at revision %q availability %q",
						fresh.Policy.Revision, fresh.Policy.Availability, fresh.Availability, want.Revision, want.Availability)
				}
				if !bytes.Equal(rb.fileBytes(t), beforeBytes) {
					t.Error("policy file bytes changed although the replacement never happened")
				}

				// A failed attempt may have reached the disk, so its revision is spent
				// whether or not it landed and the retry takes the one after it.
				attemptedNumber, err := strconv.ParseInt(attempted, 10, 64)
				if err != nil {
					t.Fatal(err)
				}
				retryRevision := strconv.FormatInt(attemptedNumber+1, 10)
				if _, err := rb.b.setAvailability(context.Background(), to, nodepolicy.PausedLocally); err != nil {
					t.Fatalf("retry of %s: %v", to, err)
				}
				if got := rb.b.localPolicy(); got.Policy.Revision != retryRevision || got.Availability != to {
					t.Errorf("retry: revision %q live %q, want revision %q live %q", got.Policy.Revision, got.Availability, retryRevision, to)
				}
			})
		}
	}
}

// requireAvailabilityTransition checks the full admission push sequence and the
// live-state notifications emitted by one completed pause or resume.
func requireAvailabilityTransition(t *testing.T, rb *routedBroker, to nodepolicy.Availability, persisted bool, callStart, frameStart int) {
	t.Helper()
	states := []nodepolicy.Availability{nodepolicy.Available}
	methods := []string{nodepolicy.MethodSetAvailability}
	if to == nodepolicy.Paused {
		states = []nodepolicy.Availability{nodepolicy.Draining, nodepolicy.Paused}
	}
	if persisted {
		methods = append(methods, nodepolicy.MethodSetPolicy)
	}
	if to == nodepolicy.Paused {
		methods = append(methods, nodepolicy.MethodSetAvailability)
	}
	calls := rb.proxy.snapshot()[callStart:]
	var gotMethods []string
	var pushed []nodepolicy.SetAvailabilityParams
	for _, call := range calls {
		gotMethods = append(gotMethods, call.method)
		if call.method == nodepolicy.MethodSetAvailability {
			var params nodepolicy.SetAvailabilityParams
			if err := json.Unmarshal(call.params, &params); err != nil {
				t.Fatalf("decode availability push: %v", err)
			}
			pushed = append(pushed, params)
		}
	}
	if !reflect.DeepEqual(gotMethods, methods) {
		t.Errorf("transition push order = %v, want %v", gotMethods, methods)
	}
	var wantPushes []nodepolicy.SetAvailabilityParams
	for _, state := range states {
		wantPushes = append(wantPushes, nodepolicy.SetAvailabilityParams{State: state})
	}
	if !reflect.DeepEqual(pushed, wantPushes) {
		t.Errorf("availability pushes = %+v, want %+v", pushed, wantPushes)
	}
	var announced []nodepolicy.Availability
	for _, frame := range rb.out.frames()[frameStart:] {
		if frame.Method != notifyAvailabilityChanged {
			continue
		}
		var event struct {
			NodeID       string                  `json:"nodeId"`
			Availability nodepolicy.Availability `json:"availability"`
			Active       int                     `json:"active"`
		}
		if err := json.Unmarshal(frame.Params, &event); err != nil {
			t.Fatalf("decode availability notification: %v", err)
		}
		if event.NodeID != "self-node" || event.Active != 0 {
			t.Errorf("availability notification = %+v, want self-node with no active work", event)
		}
		announced = append(announced, event.Availability)
	}
	if !reflect.DeepEqual(announced, states) {
		t.Errorf("announced availability = %v, want %v", announced, states)
	}
	if got := rb.b.localPolicy().Availability; got != to {
		t.Errorf("live availability = %q, want completed %q", got, to)
	}
}

// pausedByFor is the pausedBy a transition to state requested by origin saves.
func pausedByFor(state nodepolicy.Availability, origin nodepolicy.PauseOrigin) nodepolicy.PauseOrigin {
	if state == nodepolicy.Paused {
		return origin
	}
	return ""
}

func requireLandedAvailability(t *testing.T, rb *routedBroker, want nodepolicy.Policy, callStart, frameStart int) {
	t.Helper()
	requireLandedPolicy(t, rb, want)
	requireAvailabilityTransition(t, rb, want.Availability, true, callStart, frameStart)
	got := rb.b.localPolicy()
	if got.ActiveRevision != want.Revision || got.Activation != activationActive || got.ActivationError != "" {
		t.Errorf("landed availability activation = %+v, want active revision %s", got.policyActivation, want.Revision)
	}
	_, announced := lastPolicyChanged(t, rb.out)
	if announced.ActiveRevision != want.Revision || announced.Activation != activationActive || announced.ActivationError != "" {
		t.Errorf("announced activation = %+v, want active revision %s", announced, want.Revision)
	}
	if got := freshBroker(t, rb.path).localPolicy().Availability; got != want.Availability {
		t.Errorf("restart live availability = %q, want %q", got, want.Availability)
	}
}

func TestAvailabilityVersion1FirstSavePublicationFaultBoundaries(t *testing.T) {
	test := func(name, fixture string, to nodepolicy.Availability) {
		for _, boundary := range publicationBoundariesWithFlushes() {
			t.Run(name+" fails at "+boundary.name, func(t *testing.T) {
				rb := newBrokerOnFile(t, fixture)
				setClock(rb.b, 5)
				rb.b.replayPolicyToProxy(context.Background(), rb.b.policyProxy())
				before := rb.b.localPolicy()
				if before.RecoveryRequired || before.Policy.Version != 2 || before.Policy.Revision != "0" || before.Policy.Idle.UnloadAfterMinutes != 4 {
					t.Fatalf("v1 fixture did not migrate in memory: %+v", before)
				}
				callStart, frameStart := len(rb.proxy.snapshot()), len(rb.out.frames())
				setReplacer(rb.b, boundary.rep)
				rb.b.handleMessage(clientRequest(1, methodSetAvailability, map[string]any{"state": to}))
				response := rb.out.waitFrame(t, "availability first save after migration", isResponse(1))
				setReplacer(rb.b, durablefile.Replacer{})
				wantMessage := "the " + string(to) + " state could not be saved: "
				if boundary.landed {
					wantMessage = "the policy document with revision 5 was saved and is being applied but may not survive a power loss: "
				}
				if response.Error == nil || response.Error.Code != -32000 || !strings.HasPrefix(response.Error.Message, wantMessage) || !strings.Contains(response.Error.Message, errPublish.Error()) {
					t.Fatalf("availability save error = %+v, want -32000 and %q wrapping the cause", response.Error, wantMessage)
				}
				if boundary.landed {
					want := nodepolicy.Default()
					want.Availability, want.Revision = to, "5"
					want.PausedBy = pausedByFor(to, nodepolicy.PausedLocally)
					want.Idle.UnloadAfterMinutes = 4
					requireLandedAvailability(t, rb, want, callStart, frameStart)
					var stored nodepolicy.Policy
					if err := json.Unmarshal(rb.fileBytes(t), &stored); err != nil {
						t.Fatalf("decode saved document: %v", err)
					}
					if stored.Version != 2 {
						t.Errorf("first availability save stored version %d, want 2", stored.Version)
					}
					return
				}
				if string(rb.fileBytes(t)) != fixture || !reflect.DeepEqual(rb.b.localPolicy(), before) {
					t.Error("failure before replacement changed the v1 bytes or migrated state")
				}
				if len(rb.proxy.snapshot()) != callStart || len(rb.out.frames()) != frameStart+1 {
					t.Error("failure before replacement pushed or announced a change")
				}
				if got := freshBroker(t, rb.path).localPolicy(); got.RecoveryRequired || !reflect.DeepEqual(got.Policy, before.Policy) || got.Availability != before.Availability {
					t.Error("restart did not load the unchanged migrated v1 document and availability")
				}
			})
		}
	}
	test("pause", `{"version":1,"availability":"available","idle":{"unloadAfterMinutes":4}}`, nodepolicy.Paused)
	test("resume", `{"version":1,"availability":"paused","idle":{"unloadAfterMinutes":4}}`, nodepolicy.Available)
}

func TestAvailabilityRelayedLandedReplacementAdoptsPolicy(t *testing.T) {
	test := func(name string, to nodepolicy.Availability, revision string) {
		for _, boundary := range publicationBoundariesWithFlushes() {
			if !boundary.landed {
				continue
			}
			t.Run(name+" fails at "+boundary.name, func(t *testing.T) {
				rb := newRoutedBroker(t)
				setClock(rb.b, 5)
				rb.b.replayPolicyToProxy(context.Background(), rb.b.policyProxy())
				if to == nodepolicy.Available {
					// A paired node may resume only a pause a paired node made.
					if _, err := rb.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedByPeer); err != nil {
						t.Fatalf("prepare paused node: %v", err)
					}
				}
				callStart, frameStart := len(rb.proxy.snapshot()), len(rb.out.frames())
				want := rb.b.localPolicy().Policy
				want.Availability, want.Revision = to, revision
				want.PausedBy = pausedByFor(to, nodepolicy.PausedByPeer)
				setReplacer(rb.b, boundary.rep)
				_, errText := rb.relay(t, "set-availability", map[string]any{"state": to})
				setReplacer(rb.b, durablefile.Replacer{})
				wantMessage := "the policy document with revision " + revision + " was saved and is being applied but may not survive a power loss: "
				if !strings.HasPrefix(errText, wantMessage) || !strings.Contains(errText, errPublish.Error()) {
					t.Errorf("relayed availability error = %q, want %q wrapping the cause", errText, wantMessage)
				}
				requireLandedAvailability(t, rb, want, callStart, frameStart)
			})
		}
	}
	test("pause", nodepolicy.Paused, "13")
	test("resume", nodepolicy.Available, "14")
}

func TestAvailabilityLandedReplacementCompletesEngineActions(t *testing.T) {
	const fixture = `{"version":2,"revision":"12","pause":{"onActive":"finish","drainTimeoutSeconds":30,"unloadModels":true,"stopEngines":true}}`
	test := func(name string, to nodepolicy.Availability, revision string) {
		t.Run(name, func(t *testing.T) {
			rb := newBrokerOnFile(t, fixture)
			setClock(rb.b, 5)
			rb.b.replayPolicyToProxy(context.Background(), rb.b.policyProxy())
			r := rb.b.nodePolicyRuntime()
			r.mu.Lock()
			r.running["ollama"] = true
			r.loaded["ollama"] = []string{"m1"}
			r.mu.Unlock()
			if to == nodepolicy.Available {
				if _, err := rb.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil {
					t.Fatalf("prepare paused engines: %v", err)
				}
				if !rb.b.skipRestoreWhilePaused() {
					t.Fatal("paused stop-engines policy did not defer restoration")
				}
			}
			callStart, frameStart := len(rb.proxy.snapshot()), len(rb.out.frames())
			engineStart := len(rb.engine.snapshot())
			want := rb.b.localPolicy().Policy
			want.Availability, want.Revision = to, revision
			want.PausedBy = pausedByFor(to, nodepolicy.PausedLocally)
			setReplacer(rb.b, durablefile.Replacer{SyncDir: func(string) error { return errPublish }})
			got, err := rb.b.setAvailability(context.Background(), to, nodepolicy.PausedLocally)
			setReplacer(rb.b, durablefile.Replacer{})
			if got != to || !errors.Is(err, errPublish) {
				t.Fatalf("landed transition = %q, %v, want completed %q and durability warning", got, err, to)
			}
			requireLandedAvailability(t, rb, want, callStart, frameStart)
			wantMethods := []string{engineUnloadModelMethod, engineSleepMethod}
			if to == nodepolicy.Available {
				wantMethods = []string{restoreEnabledEnginesMethod, engineWakeMethod}
				rb.engine.waitFor("restore and wake after landed resume", func(calls []policyCall) bool { return len(calls) >= engineStart+2 })
			}
			calls := rb.engine.snapshot()[engineStart:]
			var gotMethods []string
			for _, call := range calls {
				gotMethods = append(gotMethods, call.method)
				if call.method == restoreEnabledEnginesMethod {
					continue
				}
				var params struct {
					Engine string `json:"engine"`
					Model  string `json:"model"`
				}
				if err := json.Unmarshal(call.params, &params); err != nil {
					t.Fatalf("decode %s params: %v", call.method, err)
				}
				if params.Engine != "ollama" || (call.method == engineUnloadModelMethod && params.Model != "m1") {
					t.Errorf("engine action %s params = %+v, want ollama and loaded model m1 for unload", call.method, params)
				}
			}
			if !reflect.DeepEqual(gotMethods, wantMethods) {
				t.Errorf("landed transition engine actions = %v, want %v", gotMethods, wantMethods)
			}
		})
	}
	test("pause unloads and sleeps", nodepolicy.Paused, "13")
	test("resume restores and wakes", nodepolicy.Available, "14")
}

func TestAvailabilityDuringRecoveryPreservesFileUntilPolicyReplacementLands(t *testing.T) {
	const corrupt = `{"version":2,"availability":"available","idle":{"startOnDemand":"invalid"}}`
	test := func(name string, to nodepolicy.Availability) {
		t.Run(name, func(t *testing.T) {
			rb := newBrokerOnFile(t, corrupt)
			setClock(rb.b, 5)
			if to == nodepolicy.Paused {
				if _, err := rb.b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedLocally); err != nil {
					t.Fatalf("prepare available recovery state: %v", err)
				}
			}
			callStart, frameStart := len(rb.proxy.snapshot()), len(rb.out.frames())
			attempts := 0
			setReplacer(rb.b, durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) {
				attempts++
				return nil, errPublish
			}})
			got, err := rb.b.setAvailability(context.Background(), to, nodepolicy.PausedLocally)
			setReplacer(rb.b, durablefile.Replacer{})
			if err != nil || got != to {
				t.Fatalf("recovery availability = %q, %v, want %q without saving", got, err, to)
			}
			want := nodepolicy.Default()
			want.Availability = to
			want.PausedBy = pausedByFor(to, nodepolicy.PausedLocally)
			if current := rb.b.localPolicy(); !current.RecoveryRequired || !reflect.DeepEqual(current.Policy, want) {
				t.Error("availability changed the recovery flag or the revision-zero recovery document")
			}
			if attempts != 0 || string(rb.fileBytes(t)) != corrupt {
				t.Error("availability attempted publication or overwrote the corrupt file during recovery")
			}
			requireAvailabilityTransition(t, rb, to, false, callStart, frameStart)
			if policyPushes(rb.proxy) != 0 || len(policyChangedRevisions(t, rb.out)) != 0 {
				t.Error("unsaved recovery availability pushed or announced a policy revision")
			}
			if fresh := freshBroker(t, rb.path).localPolicy(); !fresh.RecoveryRequired || fresh.Availability != nodepolicy.Paused {
				t.Error("restart did not preserve paused recovery on the corrupt file")
			}
			want.Revision, want.Idle.UnloadAfterMinutes = "5", 7
			setReplacer(rb.b, durablefile.Replacer{SyncDir: func(string) error { return errPublish }})
			_, err = setCurrent(rb.b, mustJSON(want))
			setReplacer(rb.b, durablefile.Replacer{})
			var save *policySaveError
			if !errors.Is(err, errPublish) || !errors.As(err, &save) || save.revision != "5" {
				t.Fatalf("explicit recovery set error = %v, want landed warning for revision 5", err)
			}
			requireLandedPolicy(t, rb, want)
			if got := rb.b.localPolicy().Availability; got != to {
				t.Errorf("landed recovery changed live availability to %q, want %q", got, to)
			}
		})
	}
	test("pause", nodepolicy.Paused)
	test("resume", nodepolicy.Available)
}

func TestPolicyRevisionNeverRepeatsAfterAFailedDirectorySync(t *testing.T) {
	rb := newRoutedBroker(t)
	setClock(rb.b, 5)
	if _, err := setCurrent(rb.b, paidCeilingDoc(t, rb.b, "5")); err != nil {
		t.Fatalf("first set: %v", err)
	}
	if got := rb.b.localPolicy().Policy.Revision; got != "13" {
		t.Fatalf("first revision = %q, want %q", got, "13")
	}
	setReplacer(rb.b, durablefile.Replacer{SyncDir: func(string) error { return errPublish }})
	if _, err := setCurrent(rb.b, paidCeilingDoc(t, rb.b, "4")); !errors.Is(err, errPublish) {
		t.Fatalf("failed set error = %v, want the injected failure", err)
	}
	setReplacer(rb.b, durablefile.Replacer{})
	if got := freshBroker(t, rb.path).localPolicy().Policy.Revision; got != "14" {
		t.Fatalf("the failed set left revision %q on disk, want %q", got, "14")
	}
	if _, err := setCurrent(rb.b, paidCeilingDoc(t, rb.b, "3")); err != nil {
		t.Fatalf("set after the failure: %v", err)
	}
	if got := rb.b.localPolicy().Policy.Revision; got != "15" {
		t.Fatalf("revision after a failed attempt = %q, want %q: 14 was already on disk", got, "15")
	}
}

func TestPolicyRevisionAssignment(t *testing.T) {
	t.Run("a local set takes the next revision whatever the client sends", func(t *testing.T) {
		rb := newRoutedBroker(t)
		setClock(rb.b, 5)
		for i, supplied := range []string{"999999", "not-a-number", "0"} {
			var doc map[string]any
			if err := json.Unmarshal(paidCeilingDoc(t, rb.b, strconv.Itoa(i+1)), &doc); err != nil {
				t.Fatal(err)
			}
			doc["revision"] = supplied
			policy, err := setCurrent(rb.b, mustJSON(doc))
			if err != nil {
				t.Fatalf("set with supplied revision %q: %v", supplied, err)
			}
			if want := strconv.Itoa(13 + i); policy.Revision != want {
				t.Errorf("revision after supplying %q = %q, want %q", supplied, policy.Revision, want)
			}
		}
	})

	t.Run("a clock past the previous revision wins", func(t *testing.T) {
		rb := newRoutedBroker(t)
		setClock(rb.b, 1<<60)
		policy, err := setCurrent(rb.b, paidCeilingDoc(t, rb.b, "5"))
		if err != nil {
			t.Fatal(err)
		}
		if policy.Revision != "1152921504606846976" {
			t.Fatalf("revision = %q, want the clock's 2^60 nanoseconds", policy.Revision)
		}
	})

	t.Run("a peer set takes the next revision whatever the peer sends", func(t *testing.T) {
		rb := newRoutedBroker(t)
		setClock(rb.b, 5)
		// Each spelling would fail or be honoured if the parser saw it.
		supplied := []struct {
			name  string
			key   string
			value any
		}{
			{"arbitrary", "revision", "7777"},
			{"malformed", "revision", "not-a-number"},
			{"leading zero", "revision", "01"},
			{"JSON number", "revision", 5},
			{"null", "revision", nil},
			{"other case", "Revision", "oops"},
		}
		for i, tc := range supplied {
			snapshot := rb.peerSnapshot(t)
			base := snapshot["revision"]
			snapshot[tc.key] = tc.value
			snapshot["idle"].(map[string]any)["unloadAfterMinutes"] = i + 1
			if _, errText := rb.relay(t, "set", map[string]any{"policy": snapshot, "expectedPersistedRevision": base}); errText != "" {
				t.Fatalf("relayed set with %s revision: %s", tc.name, errText)
			}
			got := rb.b.localPolicy().Policy
			if want := strconv.Itoa(13 + i); got.Revision != want || got.Idle.UnloadAfterMinutes != i+1 {
				t.Errorf("after the %s revision: revision %q idle %d, want revision %q and the peer's idle edit %d",
					tc.name, got.Revision, got.Idle.UnloadAfterMinutes, want, i+1)
			}
		}
	})

	t.Run("a local set ignores a revision in any spelling", func(t *testing.T) {
		rb := newRoutedBroker(t)
		setClock(rb.b, 5)
		var doc map[string]any
		if err := json.Unmarshal(paidCeilingDoc(t, rb.b, "5"), &doc); err != nil {
			t.Fatal(err)
		}
		delete(doc, "revision")
		doc["Revision"] = 5
		policy, err := setCurrent(rb.b, mustJSON(doc))
		if err != nil {
			t.Fatalf("set with a numeric Revision: %v", err)
		}
		if policy.Revision != "13" {
			t.Errorf("revision = %q, want %q", policy.Revision, "13")
		}
	})

	t.Run("a restart reloads the revision that was acknowledged", func(t *testing.T) {
		rb := newRoutedBroker(t)
		policy, err := setCurrent(rb.b, paidCeilingDoc(t, rb.b, "5"))
		if err != nil {
			t.Fatal(err)
		}
		if revisionNumber(t, policy) <= 12 {
			t.Fatalf("revision %s did not advance past 12", policy.Revision)
		}
		if got := freshBroker(t, rb.path).localPolicy().Policy.Revision; got != policy.Revision {
			t.Fatalf("restart loaded revision %q, want %q", got, policy.Revision)
		}
	})

	t.Run("overflow is an error and changes nothing", func(t *testing.T) {
		rb := newBrokerOnFile(t, strings.Replace(routedPolicyDocument, `"revision": "12"`, `"revision": "9223372036854775807"`, 1))
		before := rb.fileBytes(t)
		if _, err := setCurrent(rb.b, paidCeilingDoc(t, rb.b, "5")); err == nil {
			t.Fatal("a set at the maximum revision succeeded")
		}
		if got := rb.b.localPolicy().Policy.Revision; got != "9223372036854775807" {
			t.Errorf("revision = %q after the refused set", got)
		}
		if !bytes.Equal(rb.fileBytes(t), before) {
			t.Error("the refused set changed the policy file")
		}
	})

	t.Run("a persisted availability change takes the next revision and announces it", func(t *testing.T) {
		rb := newRoutedBroker(t)
		setClock(rb.b, 5)
		for _, step := range []struct {
			state nodepolicy.Availability
			want  string
		}{{nodepolicy.Paused, "13"}, {nodepolicy.Available, "14"}} {
			if _, err := rb.b.setAvailability(context.Background(), step.state, nodepolicy.PausedLocally); err != nil {
				t.Fatalf("set availability %s: %v", step.state, err)
			}
			got := rb.b.localPolicy().Policy
			if got.Revision != step.want || got.Availability != step.state {
				t.Errorf("after %s: revision %q availability %q, want revision %q", step.state, got.Revision, got.Availability, step.want)
			}
			var stored nodepolicy.Policy
			if err := json.Unmarshal(rb.fileBytes(t), &stored); err != nil {
				t.Fatalf("decode stored policy: %v", err)
			}
			if stored.Revision != step.want || stored.Availability != step.state {
				t.Errorf("stored after %s: revision %q availability %q, want revision %q", step.state, stored.Revision, stored.Availability, step.want)
			}
		}
		if got, want := policyChangedRevisions(t, rb.out), []string{"13", "14"}; !reflect.DeepEqual(got, want) {
			t.Errorf("policy:changed revisions = %v, want %v", got, want)
		}
	})

	t.Run("a failed availability save changes nothing", func(t *testing.T) {
		rb := newRoutedBroker(t)
		before := rb.b.localPolicy()
		setReplacer(rb.b, durablefile.Replacer{Replace: func(string, string) error { return errPublish }})
		if _, err := rb.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); !errors.Is(err, errPublish) {
			t.Fatalf("pause error = %v, want the injected failure", err)
		}
		setReplacer(rb.b, durablefile.Replacer{})
		if after := rb.b.localPolicy(); !reflect.DeepEqual(after, before) {
			t.Errorf("failed pause changed the state: availability %q -> %q", before.Availability, after.Availability)
		}
		if got := policyChangedRevisions(t, rb.out); len(got) != 0 {
			t.Errorf("failed pause announced policy:changed %v", got)
		}
	})
}

func TestPolicyRecovery(t *testing.T) {
	// A corrupt document whose last state was paused, and one that was available.
	const corruptPaused = `{"version":2,"availability":"paused","idle":{"unloadAfterMinutes":"soon"}}`
	const corruptAvailable = `{"version":2,"availability":"available","idle":{"unloadAfterMinutes":"soon"}}`

	t.Run("a corrupt file runs paused with remote and paid off", func(t *testing.T) {
		for name, contents := range map[string]string{"last state paused": corruptPaused, "last state available": corruptAvailable, "truncated": `{"version":2,`} {
			t.Run(name, func(t *testing.T) {
				rb := newBrokerOnFile(t, contents)
				got := rb.b.localPolicy()
				if !got.RecoveryRequired || got.Availability != nodepolicy.Paused || got.Policy.Availability != nodepolicy.Paused {
					t.Fatalf("recovery=%t live=%q persisted=%q, want recovery required and paused",
						got.RecoveryRequired, got.Availability, got.Policy.Availability)
				}
				if len(got.Policy.Tiers) != 0 || len(got.Policy.Profiles) != 0 {
					t.Errorf("recovery policy defines tiers %v or profiles %v", got.Policy.Tiers, got.Policy.Profiles)
				}
				for tier, routing := range got.Policy.Routing.Tiers {
					if routing.RemoteAllowed || routing.PaidAllowed {
						t.Errorf("recovery policy allows remote or paid execution for tier %s: %+v", tier, routing)
					}
				}
				if bad, err := os.ReadFile(rb.path + ".bad"); err != nil || string(bad) != contents {
					t.Errorf("kept copy = %q (%v), want the corrupt bytes", bad, err)
				}
			})
		}
	})

	t.Run("policy:get reports recoveryRequired", func(t *testing.T) {
		get := func(t *testing.T, rb *routedBroker) map[string]json.RawMessage {
			t.Helper()
			rb.b.handleMessage(clientRequest(1, methodPolicyGet, nil))
			resp := rb.out.waitFrame(t, "policy:get response", isResponse(1))
			var result map[string]json.RawMessage
			if err := json.Unmarshal(resp.Result, &result); err != nil {
				t.Fatalf("decode policy:get: %v", err)
			}
			return result
		}
		if got := string(get(t, newBrokerOnFile(t, corruptPaused))["recoveryRequired"]); got != "true" {
			t.Errorf("recoveryRequired after a corrupt file = %s, want true", got)
		}
		if got := string(get(t, newRoutedBroker(t))["recoveryRequired"]); got != "false" {
			t.Errorf("recoveryRequired after a good file = %s, want false", got)
		}
	})

	t.Run("an unreadable file needs recovery", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), nodepolicy.FileName)
		if err := os.Mkdir(path, 0o700); err != nil {
			t.Fatal(err)
		}
		got := freshBroker(t, path).localPolicy()
		if !got.RecoveryRequired || got.Availability != nodepolicy.Paused {
			t.Fatalf("recovery=%t live=%q, want recovery required and paused", got.RecoveryRequired, got.Availability)
		}
	})

	t.Run("a missing file is the default and needs no recovery", func(t *testing.T) {
		got := freshBroker(t, filepath.Join(t.TempDir(), nodepolicy.FileName)).localPolicy()
		if got.RecoveryRequired || got.Availability != nodepolicy.Available || got.Policy.Revision != "0" {
			t.Fatalf("recovery=%t live=%q revision=%q, want the available default at revision 0", got.RecoveryRequired, got.Availability, got.Policy.Revision)
		}
	})

	t.Run("availability changes stay in memory", func(t *testing.T) {
		rb := newBrokerOnFile(t, corruptAvailable)
		before := rb.fileBytes(t)
		for _, state := range []nodepolicy.Availability{nodepolicy.Available, nodepolicy.Paused, nodepolicy.Available} {
			if _, err := rb.b.setAvailability(context.Background(), state, nodepolicy.PausedLocally); err != nil {
				t.Fatalf("set availability %s: %v", state, err)
			}
			got := rb.b.localPolicy()
			if got.Availability != state || !got.RecoveryRequired || got.Policy.Revision != "0" {
				t.Errorf("after %s: live=%q recovery=%t revision=%q, want the change in memory, recovery still required and revision 0",
					state, got.Availability, got.RecoveryRequired, got.Policy.Revision)
			}
			if !bytes.Equal(rb.fileBytes(t), before) {
				t.Errorf("setting %s overwrote the corrupt file", state)
			}
		}
		if got := policyChangedRevisions(t, rb.out); len(got) != 0 {
			t.Errorf("unpersisted availability changes announced policy:changed %v", got)
		}
		pushes := availabilityPushes(rb.proxy.snapshot())
		if len(pushes) == 0 || pushes[len(pushes)-1].State != nodepolicy.Available {
			t.Errorf("proxy availability pushes = %+v, want the last to be %s", pushes, nodepolicy.Available)
		}
	})

	t.Run("a paired node cannot complete recovery", func(t *testing.T) {
		rb := newBrokerOnFile(t, corruptPaused)
		before := rb.fileBytes(t)
		if result, errText := rb.relaySet(t, rb.peerSnapshot(t)); errText == "" {
			t.Fatalf("a relayed set was accepted during recovery: %s", result)
		}
		if !rb.b.localPolicy().RecoveryRequired || !bytes.Equal(rb.fileBytes(t), before) {
			t.Error("the refused relayed set changed the recovery state or the file")
		}
	})

	t.Run("an explicit set clears it and the file reloads normally", func(t *testing.T) {
		rb := newBrokerOnFile(t, corruptAvailable)
		var doc map[string]any
		if err := json.Unmarshal(mustJSON(rb.b.localPolicy().Policy), &doc); err != nil {
			t.Fatal(err)
		}
		doc["idle"].(map[string]any)["unloadAfterMinutes"] = 7
		policy, err := setCurrent(rb.b, mustJSON(doc))
		if err != nil {
			t.Fatalf("explicit set: %v", err)
		}
		if policy.Revision == "0" {
			t.Error("the recovering set kept revision 0")
		}
		got := rb.b.localPolicy()
		if got.RecoveryRequired || got.Availability != nodepolicy.Paused {
			t.Errorf("after the set: recovery=%t live=%q, want recovery cleared and the node still paused", got.RecoveryRequired, got.Availability)
		}
		fresh := freshBroker(t, rb.path).localPolicy()
		if fresh.RecoveryRequired || fresh.Policy.Idle.UnloadAfterMinutes != 7 || fresh.Policy.Revision != policy.Revision {
			t.Errorf("restart: recovery=%t idle=%d revision=%q, want the saved document",
				fresh.RecoveryRequired, fresh.Policy.Idle.UnloadAfterMinutes, fresh.Policy.Revision)
		}
	})

	t.Run("a failed set leaves recovery in place", func(t *testing.T) {
		rb := newBrokerOnFile(t, corruptAvailable)
		before := rb.fileBytes(t)
		setReplacer(rb.b, durablefile.Replacer{Replace: func(string, string) error { return errPublish }})
		_, err := setCurrent(rb.b, mustJSON(rb.b.localPolicy().Policy))
		setReplacer(rb.b, durablefile.Replacer{})
		if !errors.Is(err, errPublish) {
			t.Fatalf("set error = %v, want the injected failure", err)
		}
		if !rb.b.localPolicy().RecoveryRequired || !bytes.Equal(rb.fileBytes(t), before) {
			t.Error("a failed set cleared recovery or changed the file")
		}
	})

	t.Run("a revision issued before corruption is never issued again", func(t *testing.T) {
		before := newBrokerOnFile(t, `{"version":2,"revision":"0"}`)
		setClock(before.b, 0)
		issued := map[string]bool{}
		for i := 1; i <= 3; i++ {
			policy, err := setCurrent(before.b, mustJSON(map[string]any{"version": 2, "idle": map[string]any{"unloadAfterMinutes": i}}))
			if err != nil {
				t.Fatalf("set %d: %v", i, err)
			}
			issued[policy.Revision] = true
		}
		if want := map[string]bool{"1": true, "2": true, "3": true}; !reflect.DeepEqual(issued, want) {
			t.Fatalf("revisions issued before corruption = %v, want %v", issued, want)
		}
		if err := os.WriteFile(before.path, []byte(corruptAvailable), 0o600); err != nil {
			t.Fatal(err)
		}
		after := freshBroker(t, before.path)
		setClock(after, 1_000_000)
		policy, err := setCurrent(after, mustJSON(map[string]any{"version": 2}))
		if err != nil {
			t.Fatalf("recovering set: %v", err)
		}
		if issued[policy.Revision] || revisionNumber(t, policy) <= 3 {
			t.Errorf("recovery issued revision %s, which matches or precedes one issued before corruption %v", policy.Revision, issued)
		}
	})
}

func TestPolicyVersion1FileIsRewrittenByTheFirstSave(t *testing.T) {
	const v1 = `{"version":1,"idle":{"unloadAfterMinutes":4}}`
	rb := newBrokerOnFile(t, v1)
	got := rb.b.localPolicy()
	if got.RecoveryRequired || got.Policy.Version != nodepolicy.SchemaVersion || got.Policy.Revision != "0" || got.Policy.Idle.UnloadAfterMinutes != 4 {
		t.Fatalf("migrated v1: recovery=%t version=%d revision=%q idle=%d", got.RecoveryRequired, got.Policy.Version, got.Policy.Revision, got.Policy.Idle.UnloadAfterMinutes)
	}
	if string(rb.fileBytes(t)) != v1 {
		t.Fatal("loading a v1 file rewrote it")
	}
	setClock(rb.b, 5)
	if _, err := rb.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil {
		t.Fatalf("pause: %v", err)
	}
	var stored nodepolicy.Policy
	if err := json.Unmarshal(rb.fileBytes(t), &stored); err != nil {
		t.Fatalf("decode the rewritten file: %v", err)
	}
	if stored.Version != nodepolicy.SchemaVersion || stored.Revision != "5" || stored.Availability != nodepolicy.Paused || stored.Idle.UnloadAfterMinutes != 4 {
		t.Fatalf("rewritten file: version=%d revision=%q availability=%q idle=%d, want version 2, revision 5, paused and the migrated idle setting",
			stored.Version, stored.Revision, stored.Availability, stored.Idle.UnloadAfterMinutes)
	}
}

func TestPolicyVersion1FirstSavePublicationFaultBoundaries(t *testing.T) {
	const v1 = `{"version":1,"availability":"paused","idle":{"unloadAfterMinutes":5}}`
	for _, boundary := range publicationBoundariesWithFlushes() {
		t.Run(boundary.name, func(t *testing.T) {
			rb := newBrokerOnFile(t, v1)
			setClock(rb.b, 5)
			before := rb.b.localPolicy()
			if before.RecoveryRequired || before.Policy.Version != 2 || before.Policy.Revision != "0" || before.Policy.Idle.UnloadAfterMinutes != 5 {
				t.Fatalf("v1 fixture did not migrate in memory: %+v", before)
			}
			want := nodepolicy.Default()
			want.Availability = nodepolicy.Paused
			want.Idle.UnloadAfterMinutes = 7
			want.Revision = "5"
			setReplacer(rb.b, boundary.rep)
			rb.b.handleMessage(clientRequest(1, methodPolicySet, map[string]any{
				"policy":                    map[string]any{"version": 2, "idle": map[string]any{"unloadAfterMinutes": 7}},
				"expectedPersistedRevision": "0",
			}))
			response := rb.out.waitFrame(t, "first save after migration", isResponse(1))
			setReplacer(rb.b, durablefile.Replacer{})
			if response.Error == nil || response.Error.Code != -32000 || !strings.Contains(response.Error.Message, errPublish.Error()) {
				t.Fatalf("first migrated save = %+v, want a server error wrapping the publication failure", response.Error)
			}
			if boundary.landed {
				requireLandedPolicy(t, rb, want)
				var version int
				if err := json.Unmarshal(rb.fileSections(t)["version"], &version); err != nil {
					t.Fatalf("decode saved version: %v", err)
				}
				if version != 2 {
					t.Fatalf("saved version = %d, want complete v2 document", version)
				}
				return
			}
			if string(rb.fileBytes(t)) != v1 {
				t.Error("failure before replacement changed the original v1 bytes")
			}
			if got := rb.b.localPolicy(); !reflect.DeepEqual(got, before) {
				t.Error("failure before replacement changed the migrated in-memory policy")
			}
			if policyPushes(rb.proxy) != 0 || len(policyChangedRevisions(t, rb.out)) != 0 {
				t.Error("failure before replacement pushed or announced a policy")
			}
			if got := freshBroker(t, rb.path).localPolicy(); got.RecoveryRequired || !reflect.DeepEqual(got.Policy, before.Policy) {
				t.Error("restart did not migrate the unchanged v1 document")
			}
		})
	}
}

func TestPolicyRecoveryPublicationFaultBoundaries(t *testing.T) {
	const corruptAvailable = `{"version":2,"availability":"available","idle":{"startOnDemand":"invalid"}}`
	for _, boundary := range publicationBoundariesWithFlushes() {
		t.Run(boundary.name, func(t *testing.T) {
			rb := newBrokerOnFile(t, corruptAvailable)
			setClock(rb.b, 5)
			before := rb.b.localPolicy()
			if !before.RecoveryRequired {
				t.Fatal("corrupt fixture did not require recovery")
			}
			original := rb.fileBytes(t)
			want := nodepolicy.Default()
			want.Availability = nodepolicy.Paused
			want.Idle.UnloadAfterMinutes = 7
			want.Revision = "5"
			setReplacer(rb.b, boundary.rep)
			_, err := setCurrent(rb.b, mustJSON(want))
			setReplacer(rb.b, durablefile.Replacer{})
			if !errors.Is(err, errPublish) {
				t.Fatalf("recovering set error = %v, want injected publication failure", err)
			}
			if boundary.landed {
				requireLandedPolicy(t, rb, want)
				if _, errText := rb.relaySet(t, rb.peerSnapshot(t)); errText != "" {
					t.Fatalf("peer set after landed recovery = %q, want recovery cleared", errText)
				}
				return
			}
			if got := rb.b.localPolicy(); !reflect.DeepEqual(got, before) || !bytes.Equal(rb.fileBytes(t), original) {
				t.Error("failure before replacement changed recovery or the corrupt file")
			}
			if policyPushes(rb.proxy) != 0 || len(policyChangedRevisions(t, rb.out)) != 0 {
				t.Error("failure before replacement pushed or announced a recovering policy")
			}
		})
	}
}

// setActivationWait sets how long one activation of b may wait.
func setActivationWait(b *Broker, d time.Duration) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.activationTimeout = d
	r.mu.Unlock()
}

// refuseSetPolicy makes the fake proxy refuse node/set-policy and acknowledge
// everything else as before.
func (ab *activatedBroker) refuseSetPolicy() {
	ab.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetPolicy {
			return nil, "policy refused"
		}
		return ab.acks.answer(method, params)
	})
}

func TestPolicyActivationFailureLeavesTheSavedRevisionAheadOfTheActiveOne(t *testing.T) {
	// Each fault happens after the new revision is durably saved. The proxy
	// keeps enforcing revision 0, and policy:set, policy:get and policy:changed
	// all report both revisions without claiming the saved one is enforced.
	test := func(name, wantErr string, fault func(t *testing.T, ab *activatedBroker)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			ab := newActivatedBroker(t)
			setActivationWait(ab.b, 100*time.Millisecond)
			fault(t, ab)

			set := ab.setIdle(t, 5)
			if set.Policy.Revision != "5000" || set.ActiveRevision != "0" || set.Activation != activationError {
				t.Fatalf("policy:set = revision %q, %+v; want revision 5000 in error with revision 0 active", set.Policy.Revision, set.policyActivation)
			}
			if !strings.Contains(set.ActivationError, wantErr) || !strings.Contains(set.ActivationError, "5000") {
				t.Fatalf("activation error = %q, want it to name revision 5000 and contain %q", set.ActivationError, wantErr)
			}
			ab.wantActivation(t, "0", activationError, wantErr)

			announced, activation := lastPolicyChanged(t, ab.out)
			if announced.Revision != "5000" || activation.ActiveRevision != "0" || activation.Activation != activationError {
				t.Fatalf("policy:changed = revision %q, %+v; want revision 5000 in error with revision 0 active", announced.Revision, activation)
			}
			if data, err := os.ReadFile(ab.path); err != nil || !strings.Contains(string(data), `"revision": "5000"`) {
				t.Fatalf("the saved file does not hold revision 5000 (read error %v): %s", err, data)
			}
		})
	}
	test("proxy refuses the push", "refused", func(t *testing.T, ab *activatedBroker) { ab.refuseSetPolicy() })
	test("push cannot reach the proxy", "did not acknowledge", func(t *testing.T, ab *activatedBroker) {
		if err := ab.proxy.conn.Close(); err != nil {
			t.Fatalf("close the proxy pipe: %v", err)
		}
	})
	test("acknowledgement is lost", "deadline exceeded", func(t *testing.T, ab *activatedBroker) {
		release := make(chan struct{})
		t.Cleanup(func() { close(release) })
		ab.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
			if method == nodepolicy.MethodSetPolicy {
				<-release
			}
			return ab.acks.answer(method, params)
		})
	})
	test("acknowledgement is from another proxy run", "different proxy run", func(t *testing.T, ab *activatedBroker) {
		ab.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
			if method == nodepolicy.MethodSetPolicy {
				return nodepolicy.PolicyAck{Revision: "5000", RunID: "run-earlier-process"}, ""
			}
			return ab.acks.answer(method, params)
		})
	})
	test("acknowledgement names another revision", "not the revision 5000", func(t *testing.T, ab *activatedBroker) {
		ab.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
			if method == nodepolicy.MethodSetPolicy {
				return nodepolicy.PolicyAck{Revision: "4999", RunID: ab.acks.runID}, ""
			}
			return ab.acks.answer(method, params)
		})
	})
	test("acknowledgement is unreadable", "unreadable", func(t *testing.T, ab *activatedBroker) {
		ab.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
			if method == nodepolicy.MethodSetPolicy {
				return map[string]bool{"ok": true}, ""
			}
			return ab.acks.answer(method, params)
		})
	})
}

func TestPolicyActivationRefusedPushKeepsThePriorProxyConfiguration(t *testing.T) {
	ab := newActivatedBroker(t)
	ab.setIdle(t, 3)
	ab.refuseSetPolicy()
	set := ab.setIdle(t, 9)
	if set.Policy.Revision != "5001" || set.ActiveRevision != "5000" || set.Activation != activationError {
		t.Fatalf("policy:set = revision %q, %+v; want revision 5001 in error with revision 5000 active", set.Policy.Revision, set.policyActivation)
	}
	if held := ab.acks.heldRevision(); held != "5000" {
		t.Fatalf("the proxy holds revision %q after a refused push, want its prior 5000", held)
	}
}

func TestPolicyActivationRecoversOnTheNextSuccessfulPush(t *testing.T) {
	ab := newActivatedBroker(t)
	ab.refuseSetPolicy()
	ab.setIdle(t, 5)
	ab.wantActivation(t, "0", activationError, "refused")

	ab.proxy.setAnswer(ab.acks.answer)
	ab.b.pushPolicyNow(context.Background(), pushPolicy)
	got := ab.wantActivation(t, "5000", activationActive, "")
	if got.Policy.Revision != "5000" {
		t.Fatalf("persisted revision = %q, want 5000", got.Policy.Revision)
	}
}

func TestPolicyActivationWithoutARunningProxyIsPending(t *testing.T) {
	b, _ := newPolicyBroker(t)
	setClock(b, 5000)
	ab := &activatedBroker{b: b}
	set := ab.setIdle(t, 5)
	if set.Policy.Revision != "5000" || set.ActiveRevision != "" || set.Activation != activationPending || set.ActivationError != "" {
		t.Fatalf("policy:set with no proxy = revision %q, %+v; want revision 5000 pending with nothing active", set.Policy.Revision, set.policyActivation)
	}
}

func TestPolicyActivationNeedsTheReplayToStartTheIncarnation(t *testing.T) {
	b, _ := newPolicyBroker(t)
	setClock(b, 5000)
	attachAckingProxy(t, b)
	ab := &activatedBroker{b: b}
	set := ab.setIdle(t, 5)
	if set.Activation != activationPending || set.ActiveRevision != "" {
		t.Fatalf("policy:set on a proxy that was not replayed = %+v; want pending with nothing active", set.policyActivation)
	}
}

func TestPolicyActivationDiscardsAnAnswerFromAnEarlierProxyProcess(t *testing.T) {
	ab := newActivatedBroker(t)
	old := ab.pp

	// The proxy restarts; the new process holds revision 0 until it is told more.
	pp2, proxy2, acks2 := attachAckingProxy(t, ab.b)
	ab.b.replayPolicyToProxy(context.Background(), pp2)
	ab.proxy, ab.acks = proxy2, acks2
	ab.wantActivation(t, "0", activationActive, "")
	ab.refuseSetPolicy()
	ab.setIdle(t, 5)
	ab.wantActivation(t, "0", activationError, "refused")

	// A push to the earlier process answers late, naming exactly the saved
	// revision with a well-formed acknowledgement. It must not activate it.
	late, err := json.Marshal(nodepolicy.PolicyAck{Revision: "5000", RunID: "run-1-late"})
	if err != nil {
		t.Fatalf("encode the late acknowledgement: %v", err)
	}
	ab.b.recordActivation(old, "5000", late, nil, nil)
	ab.wantActivation(t, "0", activationError, "refused")
}

func TestPolicyReplayAfterProxyRestartActivatesThePersistedRevision(t *testing.T) {
	ab := newActivatedBroker(t)
	ab.setIdle(t, 5)
	ab.wantActivation(t, "5000", activationActive, "")

	pp2, _, acks2 := attachAckingProxy(t, ab.b)
	// The new process is published but has not been told anything yet.
	if got := ab.b.activationOf("5000"); got.Activation != activationPending || got.ActiveRevision != "" {
		t.Fatalf("a restarted proxy that was not replayed reports %+v; want pending with nothing active", got)
	}
	ab.b.replayPolicyToProxy(context.Background(), pp2)
	ab.wantActivation(t, "5000", activationActive, "")
	if held := acks2.heldRevision(); held != "5000" {
		t.Fatalf("the restarted proxy holds revision %q, want 5000", held)
	}
	announced, activation := lastPolicyChanged(t, ab.out)
	if announced.Revision != "5000" || activation.Activation != activationActive || activation.ActiveRevision != "5000" {
		t.Fatalf("policy:changed after the restart = revision %q, %+v; want revision 5000 active at 5000", announced.Revision, activation)
	}
}

func TestPolicyActivationFollowsPauseAndResume(t *testing.T) {
	ab := newActivatedBroker(t)
	ab.setIdle(t, 5)

	if a, err := ab.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil || a != nodepolicy.Paused {
		t.Fatalf("pause = %q, %v", a, err)
	}
	got := ab.wantActivation(t, "5001", activationActive, "")
	if got.Policy.Revision != "5001" || got.Policy.Availability != nodepolicy.Paused {
		t.Fatalf("after the pause the saved policy is revision %q, %q; want 5001, paused", got.Policy.Revision, got.Policy.Availability)
	}
	announced, activation := lastPolicyChanged(t, ab.out)
	if announced.Revision != "5001" || activation.Activation != activationActive || activation.ActiveRevision != "5001" {
		t.Fatalf("policy:changed after the pause = revision %q, %+v; want revision 5001 active at 5001", announced.Revision, activation)
	}

	if a, err := ab.b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedLocally); err != nil || a != nodepolicy.Available {
		t.Fatalf("resume = %q, %v", a, err)
	}
	got = ab.wantActivation(t, "5002", activationActive, "")
	if got.Policy.Revision != "5002" || got.Policy.Availability != nodepolicy.Available {
		t.Fatalf("after the resume the saved policy is revision %q, %q; want 5002, available", got.Policy.Revision, got.Policy.Availability)
	}
	if held := ab.acks.heldRevision(); held != "5002" {
		t.Fatalf("the proxy holds revision %q after the resume, want 5002", held)
	}
}

func TestPolicyActivationPauseAfterARefusedPushDeliversThePolicy(t *testing.T) {
	ab := newActivatedBroker(t)
	// Revision 5000 changes the policy content but the proxy refuses it.
	ab.refuseSetPolicy()
	ab.setIdle(t, 5)
	ab.wantActivation(t, "0", activationError, "refused")

	// The proxy recovers. Pausing saves revision 5001, whose content is revision
	// 5000's: availability goes to the proxy first, then the whole document,
	// and only that policy push activates the revision.
	ab.proxy.setAnswer(ab.acks.answer)
	before := len(ab.proxy.snapshot())
	if a, err := ab.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil || a != nodepolicy.Paused {
		t.Fatalf("pause = %q, %v", a, err)
	}
	got := ab.wantActivation(t, "5001", activationActive, "")
	if got.Policy.Revision != "5001" || got.Policy.Idle.UnloadAfterMinutes != 5 {
		t.Fatalf("after the pause the saved policy is revision %q, unloadAfterMinutes %d; want 5001 and 5", got.Policy.Revision, got.Policy.Idle.UnloadAfterMinutes)
	}
	if held := ab.acks.heldRevision(); held != "5001" {
		t.Fatalf("the proxy holds revision %q, want 5001 (the content it refused is delivered by the pause)", held)
	}
	var order []string
	for _, c := range ab.proxy.snapshot()[before:] {
		if c.method == nodepolicy.MethodSetAvailability || c.method == nodepolicy.MethodSetPolicy {
			order = append(order, c.method)
		}
	}
	if len(order) < 2 || order[0] != nodepolicy.MethodSetAvailability || order[1] != nodepolicy.MethodSetPolicy {
		t.Fatalf("pause pushes = %v, want node/set-availability first, then node/set-policy", order)
	}
}

func TestPolicyActivationPauseWhileThePolicyIsStillRefusedStaysInError(t *testing.T) {
	ab := newActivatedBroker(t)
	ab.refuseSetPolicy()
	ab.setIdle(t, 5)

	if a, err := ab.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil || a != nodepolicy.Paused {
		t.Fatalf("pause = %q, %v", a, err)
	}
	got := ab.wantActivation(t, "0", activationError, "refused")
	if got.Policy.Revision != "5001" {
		t.Fatalf("persisted revision = %q, want 5001", got.Policy.Revision)
	}
	if held := ab.acks.heldRevision(); held != "0" {
		t.Fatalf("the proxy holds revision %q, want 0 (it never took the 5000 content)", held)
	}
}

func TestPolicyActivationInRecoveryReportsTheRevisionPushed(t *testing.T) {
	b, _ := newPolicyBroker(t)
	path := filepath.Join(t.TempDir(), nodepolicy.FileName)
	if err := os.WriteFile(path, []byte("not json"), 0o600); err != nil {
		t.Fatalf("write the corrupt policy file: %v", err)
	}
	b.loadNodePolicy(path)
	pp, proxy, acks := attachAckingProxy(t, b)
	b.replayPolicyToProxy(context.Background(), pp)
	ab := &activatedBroker{b: b, acks: acks, path: path}

	got := ab.wantActivation(t, "0", activationActive, "")
	if !got.RecoveryRequired || got.Policy.Revision != "0" || got.Policy.Availability != nodepolicy.Paused {
		t.Fatalf("recovery policy:get = revision %q, %q, recoveryRequired %v; want revision 0, paused, true", got.Policy.Revision, got.Policy.Availability, got.RecoveryRequired)
	}

	// A resume during recovery is not saved, so no new revision is claimed and
	// no policy follows the availability push.
	policyPushesBefore := len(proxy.methods(nodepolicy.MethodSetPolicy))
	if a, err := b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedLocally); err != nil || a != nodepolicy.Available {
		t.Fatalf("resume = %q, %v", a, err)
	}
	got = ab.wantActivation(t, "0", activationActive, "")
	if got.Policy.Revision != "0" || got.Policy.Availability != nodepolicy.Available {
		t.Fatalf("after an unsaved resume policy:get = revision %q, %q; want revision 0, available", got.Policy.Revision, got.Policy.Availability)
	}
	if got := len(proxy.methods(nodepolicy.MethodSetPolicy)); got != policyPushesBefore {
		t.Fatalf("an unsaved resume sent %d node/set-policy pushes, want none", got-policyPushesBefore)
	}
	if data, err := os.ReadFile(path); err != nil || string(data) != "not json" {
		t.Fatalf("recovery overwrote the unusable file (read error %v): %s", err, data)
	}
}

// waitActivation polls policy:get until the activation reaches state.
func (ab *activatedBroker) waitActivation(t *testing.T, state string) policyGetResult {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		got := ab.b.localPolicy()
		if got.Activation == state {
			return got
		}
		if time.Now().After(deadline) {
			t.Fatalf("policy:get still reports %+v at revision %q; want %s", got.policyActivation, got.Policy.Revision, state)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestPolicyActivationSetDuringAHungReplayReturnsPendingWithinTheBound(t *testing.T) {
	b, out := newPolicyBroker(t)
	path := filepath.Join(t.TempDir(), nodepolicy.FileName)
	b.loadNodePolicy(path)
	setClock(b, 5000)
	setActivationWait(b, 100*time.Millisecond)
	pp, proxy, acks := attachAckingProxy(t, b)
	release := make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetAvailability {
			<-release
		}
		return acks.answer(method, params)
	})
	replayed := make(chan struct{})
	go func() {
		b.replayPolicyToProxy(context.Background(), pp)
		close(replayed)
	}()
	proxy.waitFor("the replay's availability push", func(calls []policyCall) bool {
		return len(availabilityPushes(calls)) > 0
	})

	// The replay holds the push slot while the proxy sits on its answer.
	ab := &activatedBroker{b: b, out: out, proxy: proxy, acks: acks}
	done := make(chan policySetResult, 1)
	go func() { done <- ab.setIdle(t, 5) }()
	var set policySetResult
	select {
	case set = <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("policy:set is wedged behind a hung replay; its wait must stay within the bound")
	}
	if set.Policy.Revision != "5000" || set.ActiveRevision != "" || set.Activation != activationPending || set.ActivationError != "" {
		t.Fatalf("policy:set behind a hung replay = revision %q, %+v; want revision 5000 pending with nothing active", set.Policy.Revision, set.policyActivation)
	}
	if data, err := os.ReadFile(path); err != nil || !strings.Contains(string(data), `"revision": "5000"`) {
		t.Fatalf("the saved file does not hold revision 5000 (read error %v)", err)
	}

	// The push the set left scheduled still goes out once the replay is done.
	close(release)
	select {
	case <-replayed:
	case <-time.After(3 * time.Second):
		t.Fatal("the replay did not finish after the proxy answered")
	}
	got := ab.waitActivation(t, activationActive)
	if got.ActiveRevision != "5000" || got.Policy.Revision != "5000" {
		t.Fatalf("after the replay policy:get = revision %q, %+v; want revision 5000 active at 5000", got.Policy.Revision, got.policyActivation)
	}
	if held := acks.heldRevision(); held != "5000" {
		t.Fatalf("the proxy holds revision %q, want 5000", held)
	}
}

func TestPolicyActivationAProxyThatStopsReadingDoesNotWedgeLaterSets(t *testing.T) {
	b, _ := newPolicyBroker(t)
	b.loadNodePolicy(filepath.Join(t.TempDir(), nodepolicy.FileName))
	setClock(b, 5000)
	setActivationWait(b, 100*time.Millisecond)

	// Nothing ever reads the far end of this pipe, so a write to it blocks.
	client, server := net.Pipe()
	t.Cleanup(func() { _ = client.Close(); _ = server.Close() })
	pp := &proxyProcess{stdin: client}
	pp.attachPeer(client)
	go pp.peer.Serve(nil, nil)
	publishTestProxy(b, pp)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.incarnation = pp
	r.mu.Unlock()
	ab := &activatedBroker{b: b}

	for _, minutes := range []int{5, 6} {
		done := make(chan policySetResult, 1)
		go func() { done <- ab.setIdle(t, minutes) }()
		select {
		case set := <-done:
			if set.ActiveRevision != "" || set.Activation != activationError {
				t.Fatalf("policy:set (idle %d) to a proxy that stopped reading = %+v; want error with nothing active", minutes, set.policyActivation)
			}
		case <-time.After(3 * time.Second):
			t.Fatalf("policy:set (idle %d) is wedged behind a proxy that stopped reading", minutes)
		}
	}
	if got := b.localPolicy().Policy.Idle.UnloadAfterMinutes; got != 6 {
		t.Fatalf("persisted idle unload = %d, want the second set's 6", got)
	}
}

func TestPolicyActivationConcurrentSetsEachDescribeTheirOwnRevision(t *testing.T) {
	ab := newActivatedBroker(t)
	setActivationWait(ab.b, 3*time.Second)
	release := make(chan struct{})
	ab.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetPolicy && strings.Contains(string(params), `"revision":"5000"`) {
			<-release
		}
		return ab.acks.answer(method, params)
	})

	results := make(chan policySetResult, 2)
	go func() { results <- ab.setIdle(t, 5) }()
	ab.proxy.waitFor("the push of revision 5000", func(calls []policyCall) bool {
		for _, c := range calls {
			if c.method == nodepolicy.MethodSetPolicy && strings.Contains(string(c.params), `"revision":"5000"`) {
				return true
			}
		}
		return false
	})
	// Set B persists revision 5001 while the proxy holds set A's answer.
	go func() { results <- ab.setIdle(t, 9) }()
	for deadline := time.Now().Add(3 * time.Second); ab.b.localPolicy().Policy.Revision != "5001"; {
		if time.Now().After(deadline) {
			t.Fatal("the second set never persisted revision 5001")
		}
		time.Sleep(5 * time.Millisecond)
	}
	close(release)

	byRevision := map[string]policySetResult{}
	for range 2 {
		select {
		case set := <-results:
			byRevision[set.Policy.Revision] = set
		case <-time.After(5 * time.Second):
			t.Fatal("a policy:set never returned")
		}
	}
	a, b := byRevision["5000"], byRevision["5001"]
	// A asks about revision 5000, which the proxy acknowledged: active, whether
	// or not revision 5001 has been acknowledged by the time A's reply is built.
	if a.Activation != activationActive || a.ActiveRevision < "5000" {
		t.Fatalf("set A (revision 5000) = %+v; want active at 5000 or later", a.policyActivation)
	}
	if b.Activation != activationActive || b.ActiveRevision != "5001" {
		t.Fatalf("set B (revision 5001) = %+v; want active at 5001", b.policyActivation)
	}
}

func TestPolicyActivationDescribesTheRevisionAsked(t *testing.T) {
	b, _ := newPolicyBroker(t)
	pp := &proxyProcess{}
	r := b.nodePolicyRuntime()
	test := func(name, active, failure, revision string, want policyActivation) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.incarnation, r.activeRevision, r.activationErr = pp, active, failure
			if got := r.activationLocked(pp, revision); got != want {
				t.Fatalf("activation of %q with active %q = %+v, want %+v", revision, active, got, want)
			}
		})
	}
	test("the acknowledged revision", "5000", "", "5000", policyActivation{"5000", activationActive, ""})
	test("an older revision a newer acknowledgement covers", "5000", "", "4000", policyActivation{"5000", activationActive, ""})
	test("a newer revision not acknowledged", "5000", "", "5001", policyActivation{"5000", activationPending, ""})
	test("a newer revision whose push failed", "5000", "refused", "5001", policyActivation{"5000", activationError, "refused"})
	test("an acknowledged revision despite a later failure", "5000", "refused", "5000", policyActivation{"5000", activationActive, ""})
	test("nothing acknowledged", "", "", "0", policyActivation{"", activationPending, ""})
	t.Run("another proxy process", func(t *testing.T) {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.incarnation, r.activeRevision, r.activationErr = pp, "5000", ""
		if got, want := r.activationLocked(&proxyProcess{}, "5000"), (policyActivation{"", activationPending, ""}); got != want {
			t.Fatalf("activation against a different process = %+v, want %+v", got, want)
		}
	})
}

// waitLastPolicyChanged polls until the newest policy:changed announces
// revision with the given activation state.
func waitLastPolicyChanged(t *testing.T, out *syncBuffer, revision, state string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		var frames []Message
		for _, m := range out.frames() {
			if m.Method == notifyPolicyChanged {
				frames = append(frames, m)
			}
		}
		if len(frames) > 0 {
			var event struct {
				Policy nodepolicy.Policy `json:"policy"`
				policyActivation
			}
			if err := json.Unmarshal(frames[len(frames)-1].Params, &event); err != nil {
				t.Fatalf("decode policy:changed: %v", err)
			}
			if event.Policy.Revision == revision && event.Activation == state {
				return
			}
			if time.Now().After(deadline) {
				t.Fatalf("the newest policy:changed is revision %q, %+v; want revision %s %s", event.Policy.Revision, event.policyActivation, revision, state)
			}
		} else if time.Now().After(deadline) {
			t.Fatal("no policy:changed was announced")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestPolicyActivationLaterActivationByTheScheduledPushIsAnnounced(t *testing.T) {
	ab := newActivatedBroker(t)
	setActivationWait(ab.b, 100*time.Millisecond)
	r := ab.b.nodePolicyRuntime()
	// While the slot is taken, the set cannot push within its bound.
	r.acquirePush(context.Background())
	set := ab.setIdle(t, 5)
	if set.Policy.Revision != "5000" || set.Activation != activationPending || set.ActiveRevision != "0" {
		r.releasePush()
		t.Fatalf("policy:set behind a busy slot = revision %q, %+v; want revision 5000 pending with 0 active", set.Policy.Revision, set.policyActivation)
	}
	waitLastPolicyChanged(t, ab.out, "5000", activationPending)
	r.releasePush()

	// The scheduled push activates the revision after the set already replied;
	// that, too, is announced.
	ab.waitActivation(t, activationActive)
	waitLastPolicyChanged(t, ab.out, "5000", activationActive)
}

func TestPolicyActivationSlotReleasedAtTheDeadlineStillDeliversThePush(t *testing.T) {
	// The slot comes free just before the bound runs out, so the set takes it
	// with next to nothing left of its budget. Wherever the request would have
	// failed unsent, the push is left scheduled instead: never an error, and
	// the revision is delivered and announced as active.
	for offset := 97 * time.Millisecond; offset <= 101*time.Millisecond; offset += 200 * time.Microsecond {
		// In memory, so a save does not move the deadline away from the timer.
		b, out := newPolicyBroker(t)
		setClock(b, 5000)
		setActivationWait(b, 100*time.Millisecond)
		pp, proxy, acks := attachAckingProxy(t, b)
		b.replayPolicyToProxy(context.Background(), pp)
		ab := &activatedBroker{b: b, out: out, proxy: proxy, acks: acks}
		r := ab.b.nodePolicyRuntime()
		r.acquirePush(context.Background())
		time.AfterFunc(offset, r.releasePush)
		set := ab.setIdle(t, 5)
		if set.Activation == activationError {
			t.Fatalf("slot released at %v: policy:set = %+v, want an unsent push to stay pending, not an error", offset, set.policyActivation)
		}
		got := ab.waitActivation(t, activationActive)
		if got.Policy.Revision != "5000" || got.ActiveRevision != "5000" {
			t.Fatalf("slot released at %v: policy:get = revision %q, %+v; want revision 5000 active at 5000", offset, got.Policy.Revision, got.policyActivation)
		}
		if held := ab.acks.heldRevision(); held != "5000" {
			t.Fatalf("slot released at %v: the proxy holds revision %q, want 5000", offset, held)
		}
		waitLastPolicyChanged(t, ab.out, "5000", activationActive)
	}
}

// awaitResponse waits up to d for the response to request id. A hung proxy
// answer is cut off at the proxy call bound (5 s), so waitFrame's 3 s is too
// short for the timed-out cases.
func awaitResponse(t *testing.T, out *syncBuffer, id int, d time.Duration) Message {
	t.Helper()
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		for _, m := range out.frames() {
			if isResponse(id)(m) {
				return m
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("no response to request %d within %v", id, d)
	return Message{}
}

func TestPolicySetPublicationFailureIsAServerError(t *testing.T) {
	// The request was fine; the broker could not write the file. -32602 tells a
	// caller to change the request, so it is kept for invalid input, and a
	// failed publication is the ordinary server error at every boundary.
	for _, boundary := range publicationBoundaries() {
		name, rep, landed := boundary.name, boundary.rep, boundary.landed
		t.Run(name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			setClock(rb.b, 5)
			setReplacer(rb.b, rep)
			rb.b.handleMessage(clientRequest(1, methodPolicySet, map[string]any{
				"policy":                    json.RawMessage(paidCeilingDoc(t, rb.b, "9")),
				"expectedPersistedRevision": rb.b.localPolicy().Policy.Revision,
			}))
			resp := rb.out.waitFrame(t, "policy:set response", isResponse(1))
			wantMessage := "the policy could not be saved: "
			if landed {
				wantMessage = "the policy document with revision 13 was saved and is being applied but may not survive a power loss: "
			}
			if resp.Error == nil || resp.Error.Code != -32000 || !strings.HasPrefix(resp.Error.Message, wantMessage) || !strings.Contains(resp.Error.Message, errPublish.Error()) {
				t.Fatalf("policy:set publication error = %+v, want -32000 and %q wrapping the cause", resp.Error, wantMessage)
			}

			setReplacer(rb.b, durablefile.Replacer{})
			rb.b.handleMessage(clientRequest(2, methodPolicySet, map[string]any{
				"policy":                    map[string]any{"version": 1, "profiles": []any{map[string]any{"name": "x", "engine": "nope", "model": "m"}}},
				"expectedPersistedRevision": rb.b.localPolicy().Policy.Revision,
			}))
			resp = rb.out.waitFrame(t, "invalid policy:set response", isResponse(2))
			if resp.Error == nil || resp.Error.Code != -32602 {
				t.Fatalf("invalid policy:set = %+v, want -32602", resp.Error)
			}
		})
	}
}

// failAvailability answers node/set-availability from fail, which gets the
// 1-based number of the push since this call and returns "" to acknowledge it,
// "hang" to leave it unanswered until the proxy double's releaseHeld, or any
// other text to refuse it. Everything else is answered by the proxy double.
func (ab *activatedBroker) failAvailability(fail func(n int) string) {
	var mu sync.Mutex
	count := 0
	ab.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetAvailability {
			mu.Lock()
			count++
			n := count
			mu.Unlock()
			switch reason := fail(n); reason {
			case "":
			case "hang":
				return nil, holdReply
			default:
				return nil, reason
			}
		}
		return ab.acks.answer(method, params)
	})
}

func TestPolicyActivationAvailabilityPushFaults(t *testing.T) {
	always := func(reason string) func(int) string { return func(int) string { return reason } }
	// Revisions come from the fixed clock: the pause or resume under test is
	// 5000 (resume: after a healthy pause at 5000, 5001); the retry that heals
	// it takes the next one.
	cases := []struct {
		name string
		// pausedFirst pauses with a healthy proxy before the fault is set.
		pausedFirst bool
		to          nodepolicy.Availability
		fail        func(n int) string
		// wantError is whether the reported activation of the new revision is
		// an error once the request returns; wantText is what it must say.
		wantError bool
		wantText  string
		// healByRepeating heals with the same request again; otherwise with a
		// policy:set, which sends the availability again before its policy.
		healByRepeating bool
		// policyDelivered is whether the proxy gets the revision's policy: not
		// when a hung availability push uses up the whole activation budget.
		policyDelivered bool
	}{
		{"pause refused", false, nodepolicy.Paused, always("availability refused"), true, "refused availability", true, true},
		{"pause final push refused", false, nodepolicy.Paused, func(n int) string {
			if n == 2 {
				return "availability refused"
			}
			return ""
		}, true, "refused availability paused", false, true},
		{"pause final push timed out", false, nodepolicy.Paused, func(n int) string {
			if n == 2 {
				return "hang"
			}
			return ""
		}, true, "did not acknowledge availability paused", false, true},
		{"pause draining push refused, paused push acknowledged", false, nodepolicy.Paused, func(n int) string {
			if n == 1 {
				return "availability refused"
			}
			return ""
		}, false, "", false, true},
		{"resume refused", true, nodepolicy.Available, always("availability refused"), true, "refused availability available", false, true},
		{"resume timed out", true, nodepolicy.Available, always("hang"), true, "did not acknowledge availability available", true, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ab := newActivatedBroker(t)
			// A hung availability push is cut off at the proxy call bound (5 s); the
			// shorter activation wait keeps the retry after it from taking as long.
			setActivationWait(ab.b, 500*time.Millisecond)
			next := 5000
			if c.pausedFirst {
				if a, err := ab.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil || a != nodepolicy.Paused {
					t.Fatalf("healthy pause = %q, %v", a, err)
				}
				ab.wantActivation(t, "5000", activationActive, "")
				next = 5001
			}
			attempted := strconv.Itoa(next)
			ab.failAvailability(c.fail)

			ab.b.handleMessage(clientRequest(1, methodSetAvailability, map[string]any{"state": c.to}))
			resp := awaitResponse(t, ab.out, 1, 20*time.Second)
			if resp.Error != nil {
				t.Fatalf("node:set-availability = error %+v; the state was reached, so the call succeeds and its result carries the activation", resp.Error)
			}
			var result availabilitySetResult
			if err := json.Unmarshal(resp.Result, &result); err != nil {
				t.Fatalf("decode node:set-availability result %s: %v", resp.Result, err)
			}
			if result.Availability != c.to || result.Revision != attempted {
				t.Fatalf("node:set-availability result = %q at revision %q, want %q at revision %s", result.Availability, result.Revision, c.to, attempted)
			}
			got := ab.b.localPolicy()
			if got.Policy.Revision != attempted || got.Policy.Availability != c.to || got.Availability != c.to {
				t.Fatalf("policy:get = revision %q, saved %q, live %q; want the change saved and reached: revision %s, %q", got.Policy.Revision, got.Policy.Availability, got.Availability, attempted, c.to)
			}
			if !c.wantError {
				if result.Activation != activationActive || result.ActiveRevision != attempted || result.ActivationError != "" {
					t.Fatalf("result = %+v, want revision %s active: the newest availability push was acknowledged", result.policyActivation, attempted)
				}
				ab.wantActivation(t, attempted, activationActive, "")
				return
			}
			// The proxy acknowledged the policy carrying the revision, which alone
			// must not activate it: it still admits as it did before.
			if held := ab.acks.heldRevision(); c.policyDelivered && held != attempted {
				t.Fatalf("the proxy holds revision %q, want %s: the policy push was acknowledged", held, attempted)
			}
			if result.Activation != activationError || !strings.Contains(result.ActivationError, c.wantText) {
				t.Fatalf("result = %+v, want activation error naming %q", result.policyActivation, c.wantText)
			}
			if result.ActiveRevision != "" {
				t.Fatalf("result reports revision %q active although its availability was not acknowledged", result.ActiveRevision)
			}
			ab.wantActivation(t, result.ActiveRevision, activationError, c.wantText)
			waitLastPolicyChanged(t, ab.out, attempted, activationError)
			if data, err := os.ReadFile(ab.path); err != nil || !strings.Contains(string(data), `"revision": "`+attempted+`"`) {
				t.Fatalf("the saved file does not hold revision %s (read error %v)", attempted, err)
			}

			// The proxy recovers and the next successful push activates.
			ab.proxy.releaseHeld()
			ab.proxy.setAnswer(ab.acks.answer)
			before := len(ab.proxy.snapshot())
			healed := strconv.Itoa(next + 1)
			if c.healByRepeating {
				ab.b.handleMessage(clientRequest(2, methodSetAvailability, map[string]any{"state": c.to}))
				resp = awaitResponse(t, ab.out, 2, 20*time.Second)
				var again availabilitySetResult
				if resp.Error != nil || json.Unmarshal(resp.Result, &again) != nil {
					t.Fatalf("repeated node:set-availability = %+v", resp)
				}
				if again.Availability != c.to || again.Revision != healed || again.Activation != activationActive || again.ActiveRevision != healed || again.ActivationError != "" {
					t.Fatalf("repeated node:set-availability = %+v at revision %q, want revision %s active", again.policyActivation, again.Revision, healed)
				}
			} else {
				if set := ab.setIdle(t, 5); set.Policy.Revision != healed || set.Activation != activationActive || set.ActiveRevision != healed {
					t.Fatalf("policy:set after the proxy recovered = revision %q, %+v, want revision %s active", set.Policy.Revision, set.policyActivation, healed)
				}
			}
			ab.wantActivation(t, healed, activationActive, "")
			waitLastPolicyChanged(t, ab.out, healed, activationActive)
			var order []string
			for _, call := range ab.proxy.snapshot()[before:] {
				if call.method == nodepolicy.MethodSetAvailability || call.method == nodepolicy.MethodSetPolicy {
					order = append(order, call.method)
				}
			}
			if len(order) < 2 || order[0] != nodepolicy.MethodSetAvailability || order[1] != nodepolicy.MethodSetPolicy {
				t.Fatalf("pushes that healed it = %v, want node/set-availability, then node/set-policy", order)
			}
		})
	}
}

func TestPolicyActivationRecoveryAvailabilityRefusedIsNotActive(t *testing.T) {
	// During recovery nothing is saved, so the revision at stake is the
	// recovery policy's "0": a pause the proxy refuses must not leave it active.
	b, out := newPolicyBroker(t)
	path := filepath.Join(t.TempDir(), nodepolicy.FileName)
	if err := os.WriteFile(path, []byte("not json"), 0o600); err != nil {
		t.Fatalf("write the corrupt policy file: %v", err)
	}
	b.loadNodePolicy(path)
	pp, proxy, acks := attachAckingProxy(t, b)
	b.replayPolicyToProxy(context.Background(), pp)
	ab := &activatedBroker{b: b, out: out, pp: pp, proxy: proxy, acks: acks, path: path}
	ab.wantActivation(t, "0", activationActive, "")

	ab.failAvailability(func(int) string { return "availability refused" })
	b.handleMessage(clientRequest(1, methodSetAvailability, map[string]any{"state": nodepolicy.Available}))
	resp := awaitResponse(t, out, 1, 5*time.Second)
	var result availabilitySetResult
	if resp.Error != nil || json.Unmarshal(resp.Result, &result) != nil {
		t.Fatalf("node:set-availability = %+v", resp)
	}
	if result.Availability != nodepolicy.Available || result.Revision != "0" || result.Activation != activationError || !strings.Contains(result.ActivationError, "refused availability available") {
		t.Fatalf("result = %q revision %q %+v, want available at revision 0 in error", result.Availability, result.Revision, result.policyActivation)
	}
	waitLastPolicyChanged(t, out, "0", activationError)
	if data, err := os.ReadFile(path); err != nil || string(data) != "not json" {
		t.Fatalf("recovery overwrote the unusable file (read error %v): %s", err, data)
	}
}

func TestPeerPolicySetPublicationFailureLeavesOneWholeDocument(t *testing.T) {
	// The relayed set merges the peer's device fields into the stored policy and
	// then persists the result; each publication boundary can fail the persist.
	for _, boundary := range publicationBoundaries() {
		name, rep, landed := boundary.name, boundary.rep, boundary.landed
		t.Run(name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			setClock(rb.b, 5)
			doc := rb.peerSnapshot(t)
			doc["idle"].(map[string]any)["unloadAfterMinutes"] = 7
			if _, errText := rb.relaySet(t, doc); errText != "" {
				t.Fatalf("acknowledged peer set: %s", errText)
			}
			rb.proxy.waitFor("push of the acknowledged policy", func([]policyCall) bool { return policyPushes(rb.proxy) >= 1 })
			before := rb.b.localPolicy()
			beforeBytes := rb.fileBytes(t)
			pushes := policyPushes(rb.proxy)
			changed := policyChangedRevisions(t, rb.out)

			attempt := rb.peerSnapshot(t)
			attempt["idle"].(map[string]any)["unloadAfterMinutes"] = 9
			setReplacer(rb.b, rep)
			result, errText := rb.relaySet(t, attempt)
			setReplacer(rb.b, durablefile.Replacer{})
			if !strings.Contains(errText, errPublish.Error()) || len(result) != 0 && string(result) != "null" {
				t.Fatalf("relayed set that cannot be published = result %s, error %q, want an error naming the injected failure and no result", result, errText)
			}

			if landed {
				want := before.Policy
				want.Idle.UnloadAfterMinutes = 9
				want.Revision = "14"
				requireLandedPolicy(t, rb, want)
				if !strings.HasPrefix(errText, "the policy document with revision 14 was saved and is being applied but may not survive a power loss: ") {
					t.Errorf("landed peer set diagnostic = %q, want saved revision and power-loss warning", errText)
				}
			} else {
				if after := rb.b.localPolicy(); !reflect.DeepEqual(after, before) {
					t.Errorf("failed publication changed the in-memory state: revision %s -> %s, idle %d -> %d",
						before.Policy.Revision, after.Policy.Revision, before.Policy.Idle.UnloadAfterMinutes, after.Policy.Idle.UnloadAfterMinutes)
				}
				time.Sleep(100 * time.Millisecond)
				if got := policyPushes(rb.proxy); got != pushes {
					t.Errorf("failed publication pushed to the proxy: %d pushes, want %d", got, pushes)
				}
				if got := policyChangedRevisions(t, rb.out); !reflect.DeepEqual(got, changed) {
					t.Errorf("failed publication announced policy:changed %v, want %v", got, changed)
				}
				if leftovers, _ := filepath.Glob(filepath.Join(filepath.Dir(rb.path), "*.tmp")); len(leftovers) != 0 {
					t.Errorf("failed publication left temporary files: %v", leftovers)
				}
			}

			fresh := freshBroker(t, rb.path).localPolicy()
			if fresh.RecoveryRequired {
				t.Fatal("restart after a failed publication needs recovery")
			}
			spent := revisionNumber(t, before.Policy) + 1
			if landed {
				want := before.Policy
				want.Idle.UnloadAfterMinutes = 9
				want.Revision = strconv.FormatInt(spent, 10)
				if !reflect.DeepEqual(fresh.Policy, want) {
					t.Errorf("restart loaded a document that is neither wholly old nor wholly new: %+v", fresh.Policy)
				}
			} else {
				if !reflect.DeepEqual(fresh.Policy, before.Policy) {
					t.Errorf("restart loaded revision %s (idle %d), want the acknowledged revision %s (idle 7)",
						fresh.Policy.Revision, fresh.Policy.Idle.UnloadAfterMinutes, before.Policy.Revision)
				}
				if got := rb.fileBytes(t); !bytes.Equal(got, beforeBytes) {
					t.Error("policy file bytes changed although the replacement never happened")
				}
			}

			// A landed edit spent its draft's base revision: the caller must reload.
			// Before replacement, the same draft can still be retried.
			if landed {
				if _, errText := rb.relaySet(t, attempt); !strings.HasPrefix(errText, "policy conflict:") {
					t.Fatalf("stale peer retry error = %q, want policy conflict", errText)
				}
				attempt = rb.peerSnapshot(t)
			}
			if _, errText := rb.relaySet(t, attempt); errText != "" {
				t.Fatalf("retry of the same peer set: %s", errText)
			}
			if got, want := rb.b.localPolicy().Policy.Revision, strconv.FormatInt(spent+1, 10); got != want {
				t.Errorf("retry took revision %q, want %q: %d was already spent", got, want, spent)
			}
		})
	}
}
