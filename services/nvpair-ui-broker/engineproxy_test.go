// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/errors"
	"nvpair-shared/nodepolicy"
)

// The broker's table must cover exactly the shared engine set, in the same
// order. Preparation order is load-bearing: Ollama goes first because its
// preparation reserves any inherited OLLAMA_HOST alias that later engines have
// to route around.
func TestEngineProxyTableMatchesSharedEngines(t *testing.T) {
	if len(engineProxyProfiles) == 0 {
		t.Fatal("no engine proxy profiles")
	}
	if got := engineProxyProfiles[0].Name; got != "ollama" {
		t.Fatalf("first profile = %q, want ollama prepared first", got)
	}
	for _, p := range engineProxyProfiles {
		if p.FacadePort == p.EnginePortBase {
			t.Errorf("%s: facade and backend base are both %d; the proxy and engine would collide",
				p.Name, p.FacadePort)
		}
		if p.ComponentName() == "" || p.DisplayName == "" {
			t.Errorf("%s: incomplete identity %+v", p.Name, p.Engine)
		}
		// An empty probe path would silently become a GET of the root, which
		// is Ollama's convention and wrong for anything OpenAI-compatible —
		// the engine would read as down whenever it is actually up.
		if p.HealthProbePath == "" {
			t.Errorf("%s: no health probe path", p.Name)
		}
	}
}

// The probe path is the one advertiser value that is per-engine, and getting it
// wrong makes a healthy engine look permanently down rather than failing loudly.
func TestEngineHealthProbePaths(t *testing.T) {
	for _, tc := range []struct {
		engine string
		want   string
	}{
		{"ollama", "/"},
		{"lmstudio", "/v1/models"},
		{"llamacpp", "/health"},
	} {
		p, ok := engineProxyProfileFor(tc.engine)
		if !ok {
			t.Fatalf("no profile for %s", tc.engine)
		}
		if p.HealthProbePath != tc.want {
			t.Errorf("%s health probe = %q, want %q", tc.engine, p.HealthProbePath, tc.want)
		}
	}
}

// proxyport.go and lmstudioport.go restate ports and error-ID prefixes the
// engine table already carries, because they feed int32 atomics throughout the
// package. They are deleted when those files collapse; until then this is what
// stops them drifting from the table.
func TestBrokerConstantsMatchTheEngineTable(t *testing.T) {
	for _, tc := range []struct {
		engine       string
		facade       int
		backendStart int
		blockedID    string
	}{
		{"ollama", managedOllamaFacadePort, managedOllamaBackendStart, portOwnershipBlockedID},
		{"lmstudio", managedLMStudioFacadePort, managedLMStudioBackendStart, lmstudioPortOwnershipBlockedID},
	} {
		t.Run(tc.engine, func(t *testing.T) {
			p, ok := engineProxyProfileFor(tc.engine)
			if !ok {
				t.Fatalf("no profile for %s", tc.engine)
			}
			if tc.facade != p.FacadePort {
				t.Errorf("facade constant = %d, table says %d", tc.facade, p.FacadePort)
			}
			if tc.backendStart != p.EnginePortBase {
				t.Errorf("backend-start constant = %d, table says %d", tc.backendStart, p.EnginePortBase)
			}
			if want := p.ComponentName() + ":port-ownership-blocked"; tc.blockedID != want {
				t.Errorf("blocked error id = %q, want %q", tc.blockedID, want)
			}
		})
	}
	if want := ollamaProxyProfile.ComponentName() + ":port-bumped"; proxyPortBumpedID != want {
		t.Errorf("bumped error id = %q, want %q", proxyPortBumpedID, want)
	}
}

// Every engine's relocation authority is pinned explicitly. Getting these
// backwards does not fail
// to compile — it silently changes which engine the broker believes it may stop.
func TestEngineOwnershipAssignments(t *testing.T) {
	for _, tc := range []struct {
		engine string
		want   engineOwnership
	}{
		{"ollama", adoptedEngine},
		{"lmstudio", managedEngine},
		{"llamacpp", managedEngine},
	} {
		p, ok := engineProxyProfileFor(tc.engine)
		if !ok {
			t.Fatalf("no profile for %s", tc.engine)
		}
		if p.Ownership != tc.want {
			t.Errorf("%s ownership = %v, want %v", tc.engine, p.Ownership, tc.want)
		}
	}
}

// The two engines' port choreography diverges on exactly one input, and this is
// it: the engine is running on its own facade port, so the port is unavailable.
//
// An adopted engine must block — the broker cannot tell its own engine from a
// stranger there, and has authority over neither. A managed engine must plan
// the move, because engine-manager can stop and reposition it; blocking instead
// would break the most common LM Studio install, where it is already running on
// 1234.
//
// This is the regression guard for collapsing the two planners into one. A
// change that makes both engines agree here has broken one of them.
func TestOwnershipDecidesTheOccupiedFacadeOutcome(t *testing.T) {
	for _, tc := range []struct {
		engine    string
		wantMove  bool
		wantBlock string
	}{
		{engine: "ollama", wantBlock: "Ollama is already running on the compatibility port"},
		{engine: "lmstudio", wantMove: true},
	} {
		t.Run(tc.engine, func(t *testing.T) {
			p, ok := engineProxyProfileFor(tc.engine)
			if !ok {
				t.Fatalf("no profile for %s", tc.engine)
			}
			// Running on the facade, which is therefore taken; the backend base
			// is free.
			status := ollamaPortStatus{Running: true, Port: p.FacadePort}
			available := func(port int) bool { return port == p.EnginePortBase }

			got := planManagedEnginePorts(p, true, status, available)

			if tc.wantMove {
				want := managedPortPlan{Enabled: true, BackendPort: p.EnginePortBase}
				if got != want {
					t.Fatalf("plan = %+v, want %+v (a managed engine must be moved, not refused)", got, want)
				}
				return
			}
			if got.Blocked != tc.wantBlock {
				t.Fatalf("plan = %+v, want blocked with %q (an adopted engine must not be displaced)", got, tc.wantBlock)
			}
		})
	}
}

// --proxy-engines selects which engines the one binary is started for.
func TestParseProxyEngines(t *testing.T) {
	for _, tc := range []struct {
		name    string
		csv     string
		want    []string
		wantErr bool
	}{
		{name: "default set", csv: "ollama,lmstudio,llamacpp", want: []string{"ollama", "lmstudio", "llamacpp"}},
		{name: "single engine", csv: "lmstudio", want: []string{"lmstudio"}},
		{name: "whitespace and blanks are tolerated", csv: " ollama , , lmstudio ", want: []string{"ollama", "lmstudio"}},
		{name: "duplicates collapse", csv: "ollama,ollama", want: []string{"ollama"}},
		{name: "empty selects nothing", csv: "", want: nil},
		// Silently fronting the engines it did recognize would look like the
		// flag worked.
		{name: "unknown engine fails", csv: "ollama,vllm", wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseProxyEngines(tc.csv)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parseProxyEngines(%q) = %v, want an error", tc.csv, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseProxyEngines(%q): %v", tc.csv, err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("parseProxyEngines(%q) = %v, want %v", tc.csv, got, tc.want)
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Fatalf("parseProxyEngines(%q) = %v, want %v", tc.csv, got, tc.want)
				}
			}
		})
	}
}

// An engine left out of --proxy-engines is not started, and neither is any
// engine when the binary could not be resolved.
func TestProxyEnabledHonorsSelectionAndBinary(t *testing.T) {
	b := &Broker{proxyPath: "/path/to/nvpair-proxy", proxyEngines: []string{"ollama"}}
	if !b.proxyEnabled(ollamaProxyProfile) {
		t.Error("ollama was selected but is not enabled")
	}
	if b.proxyEnabled(lmstudioProxyProfile) {
		t.Error("lmstudio was not selected but is enabled")
	}

	b = &Broker{proxyEngines: []string{"ollama", "lmstudio"}}
	if b.proxyEnabled(ollamaProxyProfile) {
		t.Error("no proxy binary resolved, but the engine is enabled")
	}
}

// Preparing a facade is not read-only. For a managed engine the backend move
// runs inside preparation, so preparing an engine whose proxy is then never
// started relocates the engine off its own stock port and leaves nothing
// serving it — every existing client on that port breaks.
//
// Ollama cannot show the symptom, because its move is deferred until its proxy
// proves it holds the facade. That asymmetry is exactly why the guard has to be
// at the preparation call rather than relied on further down.
func TestDeselectedEngineIsNotPrepared(t *testing.T) {
	settings, settingsCodec := newTestRPCWorkerPipe(t)
	engine, engineCodec := newTestRPCWorkerPipe(t)

	// Every preparation begins by reading the managed-port policy, so counting
	// those reads is what proves how many engines were prepared — one for
	// Ollama and no more. Asserting on engine:set-port instead would depend on
	// whether 1234/1235 happen to be free on this machine, and would pass for
	// the wrong reason whenever they are not.
	policyReads := make(chan struct{}, 4)
	go func() {
		for {
			msg, err := settingsCodec.Read()
			if err != nil {
				return
			}
			if msg.Method == "settings/get-force-ports" {
				policyReads <- struct{}{}
			}
			_ = settingsCodec.Respond(msg.ID, map[string]bool{"value": true})
		}
	}()
	go func() {
		for {
			msg, err := engineCodec.Read()
			if err != nil {
				return
			}
			_ = engineCodec.Respond(msg.ID, json.RawMessage(`{}`))
		}
	}()

	b := &Broker{
		nodeID:            "local-node",
		proxyPath:         "/path/to/nvpair-proxy",
		proxyEngines:      []string{"ollama"},
		ollamaPortReady:   make(chan struct{}),
		lmstudioPortReady: make(chan struct{}),
	}
	b.setSettings(settings)
	b.setEngineMgr(engine)

	// Keep the managed policy branch deterministic without binding any of the
	// real engines' compatibility or backend ports.
	useSyntheticBrokerPortChecks(t, b)
	b.prepareEnabledFacades()

	select {
	case <-policyReads:
	case <-time.After(2 * time.Second):
		t.Fatal("the selected engine was never prepared; the fixture is not exercising preparation")
	}
	select {
	case <-policyReads:
		t.Fatal("a deselected engine was prepared; its backend would be relocated with no proxy to claim the port")
	case <-time.After(300 * time.Millisecond):
	}
}

// A stranger on the facade blocks both engines. This is the companion to the
// test above: the divergence is about who the occupant is, not about whether an
// occupied facade matters.
func TestAStrangerOnTheFacadeBlocksEveryEngine(t *testing.T) {
	for _, p := range engineProxyProfiles {
		t.Run(p.Name, func(t *testing.T) {
			// The engine is stopped somewhere else entirely, so whoever holds
			// the facade is not it.
			status := ollamaPortStatus{Port: p.EnginePortBase + 100}
			available := func(port int) bool { return port != p.FacadePort }

			got := planManagedEnginePorts(p, true, status, available)

			if got.Blocked != "the compatibility port is already in use" {
				t.Fatalf("plan = %+v, want blocked on the occupied facade", got)
			}
		})
	}
}

// An unaddressed errors:report or errors:clear is the proxy process speaking
// for itself, as the routing listener does. It must reach the pipeline, stamped
// with this node, rather than be dropped as an unknown process notification.
func TestUnaddressedProxyErrorsReachThePipeline(t *testing.T) {
	const id = "nvpair-proxy:routing-listener-blocked"
	b := &Broker{nodeID: "local-node"}
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()
	b.setErrors(&errorsProcess{peer: NewPeer(NewCodec(client))})

	received := make(chan *Message, 2)
	go func() {
		codec := NewCodec(server)
		for range 2 {
			msg, err := codec.Read()
			if err != nil {
				return
			}
			received <- msg
		}
	}()

	report, err := json.Marshal(errors.ServiceError{ID: id, Message: "port in use", Severity: "error", Action: "none"})
	if err != nil {
		t.Fatal(err)
	}
	b.forwardProxyProcessNotification(0, 0, "errors:report", report)
	b.forwardProxyProcessNotification(0, 0, "errors:clear", json.RawMessage(`{"id":"`+id+`"}`))

	for _, want := range []string{"errors:report", "errors:clear"} {
		select {
		case msg := <-received:
			if msg.Method != want {
				t.Fatalf("forwarded %q, want %q", msg.Method, want)
			}
			var got struct {
				ID        string `json:"id"`
				NodeID    string `json:"nodeId"`
				ClearedBy string `json:"clearedBy"`
			}
			if err := json.Unmarshal(msg.Params, &got); err != nil {
				t.Fatalf("decode %s params: %v", want, err)
			}
			if got.ID != id {
				t.Errorf("%s id = %q, want %q", want, got.ID, id)
			}
			if want == "errors:report" && got.NodeID != "local-node" {
				t.Errorf("report nodeId = %q, want the broker's node", got.NodeID)
			}
			if want == "errors:clear" && got.ClearedBy != "local-node" {
				t.Errorf("clear clearedBy = %q, want the broker's node", got.ClearedBy)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("timed out waiting for %s", want)
		}
	}
}

// stickyError returns the cached report for id, or nil.
func (b *Broker) stickyError(id string) *errors.ServiceError {
	b.stickyErrorsMu.Lock()
	defer b.stickyErrorsMu.Unlock()
	if e, ok := b.stickyErrors[id]; ok {
		return &e
	}
	return nil
}

// errorsPipe attaches a fresh fake nvpair-errors and returns what it receives.
func errorsPipe(t *testing.T, b *Broker) <-chan *Message {
	t.Helper()
	client, server := net.Pipe()
	t.Cleanup(func() { _ = client.Close(); _ = server.Close() })
	b.setErrors(&errorsProcess{peer: NewPeer(NewCodec(client))})
	received := make(chan *Message, 8)
	go func() {
		codec := NewCodec(server)
		for {
			msg, err := codec.Read()
			if err != nil {
				return
			}
			received <- msg
		}
	}()
	return received
}

// The routing listener's collision report is sent once and nvpair-errors holds
// errors only in memory, so the broker replays it to a restarted nvpair-errors
// until the proxy clears it.
func TestRoutingListenerCollisionSurvivesAnErrorsRestart(t *testing.T) {
	const id = "nvpair-proxy:routing-listener-blocked"
	b := &Broker{nodeID: "local-node"}
	report, err := json.Marshal(errors.ServiceError{ID: id, Message: "port in use", Severity: "error", Action: "none"})
	if err != nil {
		t.Fatal(err)
	}
	wait := func(t *testing.T, ch <-chan *Message, method string) {
		t.Helper()
		select {
		case msg := <-ch:
			var got errors.ServiceError
			if err := json.Unmarshal(msg.Params, &got); err != nil {
				t.Fatalf("decode %s: %v", msg.Method, err)
			}
			if msg.Method != method || got.ID != id {
				t.Fatalf("received %s for %q, want %s for %q", msg.Method, got.ID, method, id)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("timed out waiting for %s", method)
		}
	}

	first := errorsPipe(t, b)
	b.forwardProxyProcessNotification(0, 0, "errors:report", report)
	wait(t, first, "errors:report")

	restarted := errorsPipe(t, b)
	b.replayStickyErrors()
	wait(t, restarted, "errors:report")

	b.forwardProxyProcessNotification(0, 0, "errors:clear", json.RawMessage(`{"id":"`+id+`"}`))
	wait(t, restarted, "errors:clear")
	if cached := b.stickyError(id); cached != nil {
		t.Fatalf("a cleared collision is still cached: %+v", cached)
	}

	again := errorsPipe(t, b)
	b.replayStickyErrors()
	select {
	case msg := <-again:
		t.Fatalf("replayed %s after the collision was cleared", msg.Method)
	case <-time.After(300 * time.Millisecond):
	}
}

// Both sticky ids share the one cache and replay.
func TestStickyErrorsReplayEveryCachedIDInOrder(t *testing.T) {
	b := &Broker{nodeID: "local-node"}
	ids := []string{routingListenerBlockedID, ollamaHostAliasBlockedID}
	for _, id := range ids {
		b.forwardErrorsReport(errors.ServiceError{ID: id, Message: "m"})
	}
	b.forwardErrorsReport(errors.ServiceError{ID: "unrelated:error", Message: "m"})

	received := errorsPipe(t, b)
	b.replayStickyErrors()
	var got []string
	for range ids {
		select {
		case msg := <-received:
			var e errors.ServiceError
			if err := json.Unmarshal(msg.Params, &e); err != nil {
				t.Fatal(err)
			}
			got = append(got, e.ID)
		case <-time.After(2 * time.Second):
			t.Fatalf("replayed %v, want both sticky ids", got)
		}
	}
	if want := []string{"nvpair-proxy:routing-listener-blocked", "ollama-proxy:ollama-host-alias-blocked"}; got[0] != want[0] || got[1] != want[1] {
		t.Errorf("replayed %v, want %v", got, want)
	}
	select {
	case msg := <-received:
		t.Errorf("replayed an unrelated report: %s", msg.Params)
	case <-time.After(200 * time.Millisecond):
	}
}

// fakeProxyEnv makes the test binary act as a proxy child, so supervision runs
// against a real process without a facade port or the proxy binary. It is read
// in init because the child must answer before the test framework parses the
// argv the broker gives it. Its value picks the behaviour.
const (
	fakeProxyEnv = "NVPAIR_BROKER_TEST_FAKE_PROXY"
	// fakeProxyAcceptsFacades makes the child enable a facade instead of
	// refusing it.
	fakeProxyAcceptsFacades = "accept-facades"
	// fakeProxyHangs makes the child read its requests and never answer.
	fakeProxyHangs = "hang"
	// fakeProxyHoldsReplay holds the last replay acknowledgement until the
	// test sends a log/set-level notification on the same input pipe.
	fakeProxyHoldsReplay = "hold-replay"
)

func init() {
	if os.Getenv(fakeProxyEnv) != "" {
		runFakeProxy()
		os.Exit(0)
	}
}

// runFakeProxy acknowledges every policy push, refuses every facade/enable
// unless told to accept it, and exits when its stdin closes, which is how the
// broker stops a proxy.
func runFakeProxy() {
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(nil, 1<<20)
	out := json.NewEncoder(os.Stdout)
	for in.Scan() {
		var req struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if json.Unmarshal(in.Bytes(), &req) != nil || len(req.ID) == 0 || os.Getenv(fakeProxyEnv) == fakeProxyHangs {
			continue
		}
		reply := map[string]any{"jsonrpc": "2.0", "id": req.ID}
		switch req.Method {
		case nodepolicy.MethodSetPolicy:
			var pol nodepolicy.SetPolicyParams
			if err := json.Unmarshal(req.Params, &pol); err != nil {
				os.Exit(2)
			}
			reply["result"] = nodepolicy.PolicyAck{OriginNodeID: pol.OriginNodeID, Revision: pol.Policy.Revision, RunID: "fake-run"}
		case nodepolicy.MethodSetModelIdentity:
			if os.Getenv(fakeProxyEnv) == fakeProxyHoldsReplay {
				// The actual proxy reader observes this only after policy was
				// acknowledged. The final replay RPC remains unanswered until
				// the test has changed the listener and installed its Stop barrier.
				if out.Encode(map[string]any{"jsonrpc": "2.0", "method": nodepolicy.NotifyAdmissionState, "params": nodepolicy.AdmissionState{}}) != nil {
					return
				}
				if !in.Scan() {
					return
				}
				var release struct {
					ID     json.RawMessage `json:"id"`
					Method string          `json:"method"`
				}
				if json.Unmarshal(in.Bytes(), &release) != nil || len(release.ID) != 0 || release.Method != "log/set-level" {
					os.Exit(2)
				}
			}
			reply["result"] = map[string]any{}
		case "facade/enable":
			if os.Getenv(fakeProxyEnv) == fakeProxyAcceptsFacades {
				var spec enableFacadeRequest
				_ = json.Unmarshal(req.Params, &spec)
				reply["result"] = enableFacadeReply{Engine: spec.Engine, Port: 40124}
			} else {
				reply["error"] = map[string]any{"code": -32000, "message": "refused"}
			}
		case fakeProxyEnvironmentMethod:
			reply["result"] = fakeProxyEnvironment()
		default:
			reply["result"] = map[string]any{}
		}
		if out.Encode(reply) != nil {
			return
		}
	}
}

// testListenerPort is a port the fake proxy never binds, so the tests below
// need no free port.
const testListenerPort = 40123

// supervisedBroker is a broker with a policy file, a client codec and the fake
// child as its proxy binary. It is driven the way a client drives it: through
// handleMessage.
type supervisedBroker struct {
	b    *Broker
	out  *syncBuffer
	path string
	next int
}

// newSupervisedBroker loads a policy whose listener is enabled or not. The
// broker's lifetime is cancelled before supervision is shut down, as Serve does.
func newSupervisedBroker(t *testing.T, listener bool, facades ...string) *supervisedBroker {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(fakeProxyEnv, "1")
	b, out := newPolicyBroker(t)
	useSyntheticBrokerPortChecks(t, b)
	b.proxyPath = exe
	b.proxyEngines = facades
	b.ollamaPortReady = make(chan struct{})
	b.lmstudioPortReady = make(chan struct{})
	path := filepath.Join(t.TempDir(), nodepolicy.FileName)
	if listener {
		doc := nodepolicy.Default()
		doc.Routing.Listener = nodepolicy.Listener{Enabled: true, Port: testListenerPort, AllowedOrigins: []string{}}
		writePolicyFile(t, path, doc)
	}
	b.loadNodePolicy(path)
	ctx, cancel := context.WithCancel(context.Background())
	b.setServeCtx(ctx)
	t.Cleanup(func() {
		cancel()
		b.shutdownProxySupervision()
	})
	return &supervisedBroker{b: b, out: out, path: path}
}

// policyResponse is a policy:set answer, decoded.
type policyResponse struct {
	Message
	activation string
	elapsed    time.Duration
}

// call sends one request and waits for its response.
func (sb *supervisedBroker) call(t *testing.T, method string, params any, within time.Duration) policyResponse {
	t.Helper()
	sb.next++
	id := sb.next
	started := time.Now()
	sb.b.handleMessage(clientRequest(id, method, params))
	deadline := started.Add(within)
	for time.Now().Before(deadline) {
		for _, m := range sb.out.frames() {
			if isResponse(id)(m) {
				var result struct {
					Activation string `json:"activation"`
				}
				_ = json.Unmarshal(m.Result, &result)
				return policyResponse{Message: m, activation: result.Activation, elapsed: time.Since(started)}
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("%s was not answered within %s", method, within)
	return policyResponse{}
}

// setListener is a policy:set that changes only the routing listener.
func (sb *supervisedBroker) setListener(t *testing.T, enabled bool, port int) policyResponse {
	t.Helper()
	doc := sb.b.localPolicy().Policy
	doc.Routing.Listener = nodepolicy.Listener{Enabled: enabled, Port: port, AllowedOrigins: []string{}}
	return sb.call(t, methodPolicySet, map[string]any{"policy": doc, "expectedPersistedRevision": doc.Revision}, 2*policyActivationTimeout)
}

func (sb *supervisedBroker) mustSetListener(t *testing.T, enabled bool) policyResponse {
	t.Helper()
	got := sb.setListener(t, enabled, testListenerPort)
	if got.Error != nil {
		t.Fatalf("policy:set listener enabled=%t: %+v", enabled, got.Error)
	}
	return got
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("never happened: %s", what)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// waitReconciled returns once the background supervision pass has nothing left
// to do, so a test can assert that a policy change did not touch the proxy.
func waitReconciled(t *testing.T, b *Broker) {
	t.Helper()
	eventually(t, "the supervision pass finishing", func() bool {
		b.proxySupWorkMu.Lock()
		defer b.proxySupWorkMu.Unlock()
		return !b.proxySupWorking
	})
}

func exited(pp *proxyProcess) bool {
	select {
	case <-pp.Done():
		return true
	default:
		return false
	}
}

func requireProxyRunning(t *testing.T, pp *proxyProcess, why string) {
	t.Helper()
	if pp == nil {
		t.Fatalf("no proxy process: %s", why)
	}
	if exited(pp) {
		t.Fatalf("the proxy process exited: %s", why)
	}
}

func requireProxyExited(t *testing.T, pp *proxyProcess, why string) {
	t.Helper()
	eventually(t, "the proxy process exiting: "+why, func() bool { return exited(pp) })
}

// A node with no facade runs the proxy for the routing listener alone: a
// policy:set that enables the listener starts it in the background, the
// revision is active once the new process has acknowledged the replay, and a
// policy:set that disables it stops it.
func TestProxySupervisionRunsTheListenerWithoutFacades(t *testing.T) {
	sb := newSupervisedBroker(t, false)
	b := sb.b

	b.startProxySupervision(false)
	if b.getProxySupervisor() != nil || b.proxyProcessHandle() != nil {
		t.Fatal("a node with no facade and no listener started a proxy")
	}

	sb.mustSetListener(t, true)
	eventually(t, "the listener's proxy starting", func() bool { return b.proxyProcessHandle() != nil })
	first := b.proxyProcessHandle()
	requireProxyRunning(t, first, "the policy enables the listener")
	if b.getProxySupervisor() == nil {
		t.Fatal("the listener's proxy is not supervised")
	}
	if b.engineProxyHandle(ollamaProxyProfile) != nil {
		t.Fatal("a facade is published on a node with none configured")
	}
	if got := b.policyProxy(); got != first {
		t.Fatalf("the policy talks to %p, want the listener's process %p", got, first)
	}
	eventually(t, "the revision activating in the new process", func() bool {
		return b.localPolicy().Activation == activationActive
	})

	spawns := b.currentOllamaProxyGeneration()
	if got := sb.mustSetListener(t, false); got.activation == activationError {
		t.Fatal("disabling the listener reported a failed activation for the proxy it stopped")
	}
	requireProxyExited(t, first, "the listener was disabled and no facade is configured")
	waitReconciled(t, b)
	if got := b.localPolicy(); got.Activation == activationError {
		t.Fatalf("the stopped proxy left a failed activation: %s", got.ActivationError)
	}
	if b.getProxySupervisor() != nil || b.proxyProcessHandle() != nil || b.policyProxy() != nil {
		t.Fatal("a stopped proxy is still supervised or published")
	}
	if got := b.currentOllamaProxyGeneration(); got != spawns {
		t.Fatalf("stopping a proxy with no facade configured spawned another (%d -> %d)", spawns, got)
	}

	sb.mustSetListener(t, true)
	eventually(t, "a fresh proxy for the re-enabled listener", func() bool {
		again := b.proxyProcessHandle()
		return again != nil && again != first
	})
}

// observedProxyStop holds Stop before it closes stdin, keeping the process
// published while the test delivers a failed activation from its closing peer.
type observedProxyStop struct {
	io.WriteCloser
	beforeClose func()
}

func (s observedProxyStop) Close() error {
	s.beforeClose()
	return s.WriteCloser.Close()
}

func TestProxySpawnRetiresActivationWhenListenerDisabledDuringReplay(t *testing.T) {
	sb := newSupervisedBroker(t, true)
	b := sb.b
	t.Setenv(fakeProxyEnv, fakeProxyHoldsReplay)
	stopped, stopReached, releaseStop := publicationBarrier(t)
	started := make(chan struct{})
	go func() {
		b.startProxySupervision(false)
		close(started)
	}()

	// The child reports admission state at the final replay RPC, after policy
	// acknowledgement, then holds that RPC's reply on its actual input reader.
	// Spawn cannot reach the no-facade/no-listener decision until released.
	r := b.nodePolicyRuntime()
	eventually(t, "the final replay RPC reaching the proxy reader", func() bool {
		r.mu.Lock()
		defer r.mu.Unlock()
		return r.haveAdmission
	})
	pp := b.policyProxy()
	requireProxyRunning(t, pp, "replay has not finished bring-up")
	var releaseOnce sync.Once
	releaseReplay := func() {
		releaseOnce.Do(func() {
			if err := pp.peer.Notify("log/set-level", map[string]string{"level": "info"}); err != nil {
				t.Errorf("release final replay acknowledgement: %v", err)
			}
		})
	}
	t.Cleanup(releaseReplay)
	if got := b.localPolicy(); got.Activation != activationActive {
		t.Fatalf("replayed policy activation = %+v, want active", got.policyActivation)
	}
	// Publish the listener change at precisely this boundary. The policy
	// persistence path sets current under this same lock before activation.
	r.mu.Lock()
	r.current.Routing.Listener.Enabled = false
	r.current.Revision = "5000"
	r.mu.Unlock()
	pp.stdin = observedProxyStop{WriteCloser: pp.stdin, beforeClose: stopped}
	releaseReplay()
	awaitPublication(t, stopReached)

	// The transport fails while Stop still owns the published process. Its
	// answer must be discarded even before the process exits or is withdrawn.
	pp.peer.Close()
	b.activatePolicyNow()
	got := b.localPolicy()
	if got.Activation != activationPending || got.ActivationError != "" {
		t.Errorf("activation during deliberate spawn teardown = %+v, want pending without an error", got.policyActivation)
	}
	r.mu.Lock()
	failure := r.activationErr
	r.mu.Unlock()
	if failure != "" {
		t.Errorf("recorded an activation failure from the deliberately stopped spawn: %s", failure)
	}
	releaseStop()
	awaitPublication(t, started)
	if b.getProxySupervisor() != nil || b.policyProxy() != nil {
		t.Fatal("the aborted listener-only spawn is still published")
	}
}

// A node restarted with the listener already enabled in its saved policy runs
// the proxy from startup, with no facade configured.
func TestProxySupervisionStartsForAnEnabledListenerWithoutFacades(t *testing.T) {
	sb := newSupervisedBroker(t, true)

	sb.b.startProxySupervision(false)

	requireProxyRunning(t, sb.b.proxyProcessHandle(), "the saved policy enables the listener")
	if sb.b.getProxySupervisor() == nil {
		t.Fatal("the listener's proxy is not supervised")
	}
}

// Every facade failing keeps the proxy while the policy enables the listener,
// and the failed engines take the terminal path a spawn with some facade up
// gives them, because this incarnation is not retried.
func TestProxySupervisionKeepsTheProxyWhenEveryFacadeFailsAndTheListenerIsEnabled(t *testing.T) {
	sb := newSupervisedBroker(t, true, ollamaProxyProfile.Name)
	b := sb.b
	b.ollamaState().managedFacade.Store(true)

	b.startProxySupervision(true)

	requireProxyRunning(t, b.proxyProcessHandle(), "the listener is enabled and every facade was refused")
	if b.getProxySupervisor() == nil {
		t.Fatal("the proxy is not supervised")
	}
	if b.engineProxyHandle(ollamaProxyProfile) != nil {
		t.Fatal("a refused facade is published as up")
	}
	requireGateOpens(t, b.ollamaPortReady, "the refused facade is terminal for the kept process")
	if b.ollamaState().managedFacade.Load() {
		t.Fatal("the refused facade kept its managed claim in a process that is not retried")
	}
	if got := b.localPolicy().Activation; got != activationActive {
		t.Fatalf("activation = %q, want %q", got, activationActive)
	}
}

// With the listener disabled and every facade failing, a start fails as it
// always has: no process stays and the supervisor is dropped.
func TestProxySupervisionFailsAStartWithNoFacadeAndNoListener(t *testing.T) {
	sb := newSupervisedBroker(t, false, ollamaProxyProfile.Name)
	sb.b.ollamaState().managedFacade.Store(true)

	sb.b.startProxySupervision(true)

	if sb.b.getProxySupervisor() != nil || sb.b.proxyProcessHandle() != nil {
		t.Fatal("a proxy with no facade and no listener was kept")
	}
	requireGateOpens(t, sb.b.ollamaPortReady, "a failed start releases the startup gate")
}

func TestProxySupervisionEnablingTheListenerRecoversFailedFacadeStartup(t *testing.T) {
	sb := newSupervisedBroker(t, false, ollamaProxyProfile.Name)
	b := sb.b
	b.startProxySupervision(true)
	if b.getProxySupervisor() != nil || b.proxyProcessHandle() != nil {
		t.Fatal("the refused facade unexpectedly kept the initial proxy")
	}

	got := sb.mustSetListener(t, true)
	if got.elapsed > policyActivationTimeout+2*time.Second {
		t.Fatalf("policy:set took %s, exceeding its activation bound", got.elapsed)
	}
	waitReconciled(t, b)
	requireProxyRunning(t, b.proxyProcessHandle(), "enabling the listener after failed facade startup")
	if b.getProxySupervisor() == nil {
		t.Fatal("the recovered listener's proxy is not supervised")
	}
	if b.engineProxyHandle(ollamaProxyProfile) != nil {
		t.Fatal("the refused facade is published as up")
	}
	eventually(t, "the recovered listener's revision activating", func() bool {
		got := b.localPolicy()
		return got.Activation == activationActive && got.ActiveRevision == got.Policy.Revision
	})
}

func TestProxySupervisionDisablingTheListenerStopsAfterEveryFacadeFails(t *testing.T) {
	sb := newSupervisedBroker(t, true, ollamaProxyProfile.Name)
	b := sb.b
	b.startProxySupervision(true)
	pp := b.proxyProcessHandle()
	requireProxyRunning(t, pp, "the enabled listener keeps the proxy after every facade was refused")
	if b.engineProxyHandle(ollamaProxyProfile) != nil {
		t.Fatal("the refused facade is published as up")
	}
	spawns := b.currentOllamaProxyGeneration()

	got := sb.mustSetListener(t, false)
	if got.elapsed > policyActivationTimeout+2*time.Second {
		t.Fatalf("policy:set took %s, exceeding its activation bound", got.elapsed)
	}
	waitReconciled(t, b)
	if !exited(pp) {
		t.Fatal("disabling the listener left a proxy running with every facade refused")
	}
	if b.getProxySupervisor() != nil || b.proxyProcessHandle() != nil || b.policyProxy() != nil {
		t.Fatal("the idle proxy is still supervised or published")
	}
	if got := b.currentOllamaProxyGeneration(); got != spawns {
		t.Fatalf("stopping the idle proxy spawned a replacement (%d -> %d)", spawns, got)
	}
}

// A serving facade keeps the supervisor Serve gave it across listener changes.
func TestProxySupervisionLeavesAFacadeNodesSupervisorAlone(t *testing.T) {
	sb := newSupervisedBroker(t, true, ollamaProxyProfile.Name)
	t.Setenv(fakeProxyEnv, fakeProxyAcceptsFacades)
	b := sb.b
	b.startProxySupervision(true)
	pp, sup := b.proxyProcessHandle(), b.getProxySupervisor()
	requireProxyRunning(t, pp, "the facade is accepted")

	for _, enabled := range []bool{false, true, false} {
		sb.mustSetListener(t, enabled)
		waitReconciled(t, b)
		if b.getProxySupervisor() != sup || b.proxyProcessHandle() != pp {
			t.Fatalf("setting the listener to %t replaced the supervisor or the process of a node with a facade", enabled)
		}
		requireProxyRunning(t, pp, "a listener change must not stop a facade node's proxy")
		if b.engineProxyHandle(ollamaProxyProfile) != pp {
			t.Fatal("the accepted facade is not published on its original process")
		}
	}
}

// During a crash backoff the facade node's supervisor still has its restart
// budget; a listener change must neither replace it nor end the backoff.
func TestProxySupervisionKeepsTheBackoffOfAFacadeNodeAcrossListenerChanges(t *testing.T) {
	sb := newSupervisedBroker(t, false, ollamaProxyProfile.Name)
	t.Setenv(fakeProxyEnv, fakeProxyAcceptsFacades)
	b := sb.b
	b.startProxySupervision(true)
	pp, sup := b.proxyProcessHandle(), b.getProxySupervisor()
	requireProxyRunning(t, pp, "the facade is accepted")

	if err := pp.cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	requireProxyExited(t, pp, "killed")
	for _, enabled := range []bool{true, false} {
		sb.mustSetListener(t, enabled)
		waitReconciled(t, b)
		if b.getProxySupervisor() != sup {
			t.Fatalf("setting the listener to %t during the backoff replaced the supervisor", enabled)
		}
	}
	eventually(t, "the supervisor's own restart", func() bool {
		again := b.proxyProcessHandle()
		return again != nil && again != pp
	})
	if b.getProxySupervisor() != sup {
		t.Fatal("the restart replaced the supervisor")
	}
}

// publicationBarrier pauses a lifecycle publisher without a timing window.
// Cleanup releases it before the fixture joins its owned supervisor.
func publicationBarrier(t *testing.T) (func(), <-chan struct{}, func()) {
	t.Helper()
	reached, released := make(chan struct{}), make(chan struct{})
	var signal, release sync.Once
	unblock := func() { release.Do(func() { close(released) }) }
	t.Cleanup(unblock)
	return func() {
		signal.Do(func() { close(reached) })
		<-released
	}, reached, unblock
}

func awaitPublication(t *testing.T, reached <-chan struct{}) {
	t.Helper()
	select {
	case <-reached:
	case <-time.After(10 * time.Second):
		t.Fatal("proxy did not reach the publication barrier")
	}
}

// Set before Start launches the monitor, keeping the real recovery mechanism
// but making its restart budget practical for repeated race tests.
func fastProxyRestartBudget(b *Broker) func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			b.getProxySupervisor().policy = restartPolicy{
				baseDelay: time.Millisecond, maxDelay: time.Millisecond, maxAttempts: 1,
			}
		})
	}
}

func TestProxySupervisionDefersDisableUntilRespawnPublication(t *testing.T) {
	sb := newSupervisedBroker(t, true, ollamaProxyProfile.Name)
	b := sb.b
	pause, reached, release := publicationBarrier(t)
	spawned := 0
	fastBudget := fastProxyRestartBudget(b)
	b.beforeProxySpawnPublished = func() {
		fastBudget()
		spawned++
		if spawned == 2 {
			pause()
		}
	}
	b.startProxySupervision(true)
	first, sup := b.proxyProcessHandle(), b.getProxySupervisor()
	requireProxyRunning(t, first, "the listener keeps the refused-facade proxy")
	if err := first.cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	awaitPublication(t, reached)
	respawned := b.proxyProcessHandle()
	requireProxyRunning(t, respawned, "respawn is paused before listener-only publication")
	if respawned == first || respawned.bringUpDone.Load() {
		t.Fatal("respawn barrier did not hold the new incarnation before publication")
	}

	sb.mustSetListener(t, false)
	waitReconciled(t, b)
	if b.getProxySupervisor() != sup || exited(respawned) {
		t.Fatal("disable interrupted unfinished bring-up")
	}
	release()
	eventually(t, "deferred disable retiring the idle supervisor", func() bool {
		return b.getProxySupervisor() == nil
	})
	waitReconciled(t, b)
	if !exited(respawned) || b.proxyProcessHandle() != nil {
		t.Fatal("deferred disable left an idle proxy after respawn publication")
	}
	if got := b.currentOllamaProxyGeneration(); got != 2 {
		t.Fatalf("spawn generation = %d, want exactly the initial spawn and its respawn", got)
	}
}

func TestProxySupervisionDefersEnableUntilExhaustionPublication(t *testing.T) {
	sb := newSupervisedBroker(t, false, ollamaProxyProfile.Name)
	t.Setenv(fakeProxyEnv, fakeProxyAcceptsFacades)
	b := sb.b
	b.beforeProxySpawnPublished = fastProxyRestartBudget(b)
	pause, reached, release := publicationBarrier(t)
	b.beforeProxyExhaustionPublished = pause
	b.startProxySupervision(true)
	first, sup := b.proxyProcessHandle(), b.getProxySupervisor()
	requireProxyRunning(t, first, "the initial facade is accepted")
	// The recovery spawn refuses its facade, exhausting the existing budget.
	t.Setenv(fakeProxyEnv, "1")
	if err := first.cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	awaitPublication(t, reached)
	if b.proxySupExhausted.Load() || b.proxyProcessHandle() != nil {
		t.Fatal("terminal barrier did not hold exhaustion before publication")
	}

	got := sb.mustSetListener(t, true)
	if got.activation == activationActive {
		t.Fatal("an absent proxy reported the enabled listener revision active")
	}
	revision := b.localPolicy().Policy.Revision
	waitReconciled(t, b)
	if b.getProxySupervisor() != sup {
		t.Fatal("enable replaced the supervisor before exhaustion was published")
	}
	release()
	eventually(t, "deferred enable replacing the exhausted supervisor", func() bool {
		return b.getProxySupervisor() != sup
	})
	waitReconciled(t, b)
	requireProxyRunning(t, b.proxyProcessHandle(), "the pending enable survives exhaustion publication")
	eventually(t, "the exact deferred revision activating", func() bool {
		got := b.localPolicy()
		return got.Activation == activationActive && got.ActiveRevision == revision && got.Policy.Revision == revision
	})
	if got := b.currentOllamaProxyGeneration(); got != 3 {
		t.Fatalf("spawn generation = %d, want initial, failed recovery and one replacement", got)
	}
}

func TestProxySupervisionExhaustionWithoutPendingChangeStaysExhausted(t *testing.T) {
	sb := newSupervisedBroker(t, true, ollamaProxyProfile.Name)
	b := sb.b
	b.beforeProxySpawnPublished = fastProxyRestartBudget(b)
	pause, reached, release := publicationBarrier(t)
	b.beforeProxyExhaustionPublished = pause
	b.startProxySupervision(true)
	first, sup := b.proxyProcessHandle(), b.getProxySupervisor()
	requireProxyRunning(t, first, "the saved listener enables the refused-facade proxy")
	if err := first.cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	eventually(t, "the budgeted respawn completing", func() bool {
		pp := b.proxyProcessHandle()
		return pp != nil && pp != first && pp.bringUpDone.Load()
	})
	if err := b.proxyProcessHandle().cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	awaitPublication(t, reached)
	spawns := b.currentOllamaProxyGeneration()
	if spawns != 2 {
		t.Fatalf("spawn generation = %d, want initial plus the budgeted respawn", spawns)
	}
	release()
	// Join the exhausted monitor so its publication and wakeup have completed;
	// waitReconciled then drains any work that wakeup actually scheduled.
	sup.Stop()
	waitReconciled(t, b)
	if !b.proxySupExhausted.Load() || b.getProxySupervisor() != sup || b.proxyProcessHandle() != nil {
		t.Fatal("exhaustion with no pending listener change restarted supervision")
	}
	if got := b.currentOllamaProxyGeneration(); got != spawns {
		t.Fatalf("exhaustion renewed the restart budget (%d -> %d)", spawns, got)
	}
}

func TestProxySupervisionUnrelatedPolicyEditPreservesExhaustion(t *testing.T) {
	sb := newSupervisedBroker(t, true)
	b := sb.b
	b.beforeProxySpawnPublished = fastProxyRestartBudget(b)
	pause, reached, release := publicationBarrier(t)
	b.beforeProxyExhaustionPublished = pause
	b.startProxySupervision(false)
	first, sup := b.proxyProcessHandle(), b.getProxySupervisor()
	requireProxyRunning(t, first, "the saved listener starts the proxy")
	if err := first.cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	eventually(t, "the budgeted respawn completing", func() bool {
		pp := b.proxyProcessHandle()
		return pp != nil && pp != first && pp.bringUpDone.Load()
	})
	if err := b.proxyProcessHandle().cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	awaitPublication(t, reached)
	release()
	sup.Stop()
	waitReconciled(t, b)
	if !b.proxySupExhausted.Load() || b.currentOllamaProxyGeneration() != 2 {
		t.Fatal("the fixture did not exhaust its initial spawn and one restart")
	}

	doc := b.localPolicy().Policy
	doc.Idle.UnloadAfterMinutes = 5
	got := sb.call(t, methodPolicySet, map[string]any{"policy": doc, "expectedPersistedRevision": doc.Revision}, 2*policyActivationTimeout)
	if got.Error != nil {
		t.Fatalf("policy:set idle timeout: %+v", got.Error)
	}
	waitReconciled(t, b)
	stored := b.localPolicy().Policy
	if stored.Idle.UnloadAfterMinutes != 5 || stored.Revision == doc.Revision {
		t.Fatal("the unrelated policy edit was not persisted with a new revision")
	}
	if b.getProxySupervisor() != sup || !b.proxySupExhausted.Load() || b.proxyProcessHandle() != nil {
		t.Fatal("an unrelated policy edit renewed exhausted supervision")
	}
	if got := b.currentOllamaProxyGeneration(); got != 2 {
		t.Fatalf("unrelated policy edit spawned generation %d, want 2", got)
	}

	got = sb.setListener(t, true, testListenerPort+1)
	if got.Error != nil {
		t.Fatalf("policy:set listener port: %+v", got.Error)
	}
	waitReconciled(t, b)
	if b.getProxySupervisor() == sup || b.proxySupExhausted.Load() {
		t.Fatal("a listener port change did not renew exhausted supervision")
	}
	requireProxyRunning(t, b.proxyProcessHandle(), "the changed listener renews the budget")
	if got := b.currentOllamaProxyGeneration(); got != 3 {
		t.Fatalf("listener port change spawned generation %d, want 3", got)
	}
}

func TestProxySupervisionQueuedStartHonorsServeCancellation(t *testing.T) {
	sb := newSupervisedBroker(t, false)
	b := sb.b
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	b.setServeCtx(ctx)
	b.startProxySupervision(false)
	b.proxySupMu.Lock()
	var unlocked sync.Once
	unlock := func() { unlocked.Do(b.proxySupMu.Unlock) }
	t.Cleanup(unlock)
	sb.mustSetListener(t, true)
	cancel()
	unlock()
	waitReconciled(t, b)
	if got := b.currentOllamaProxyGeneration(); got != 0 {
		t.Fatalf("queued reconciliation spawned generation %d after Serve cancellation, want 0", got)
	}
	if b.getProxySupervisor() != nil || b.proxyProcessHandle() != nil {
		t.Fatal("cancelled Serve retained proxy supervision")
	}
}

func TestProxySupervisionQueuedStartHonorsShutdownEntry(t *testing.T) {
	sb := newSupervisedBroker(t, false)
	b := sb.b
	pause, reached, release := publicationBarrier(t)
	b.beforeProxyShutdownLocked = pause
	b.startProxySupervision(false)
	b.proxySupMu.Lock()
	var unlocked sync.Once
	unlock := func() { unlocked.Do(b.proxySupMu.Unlock) }
	t.Cleanup(unlock)
	sb.mustSetListener(t, true)
	stopped := make(chan struct{})
	go func() {
		b.shutdownProxySupervision()
		close(stopped)
	}()
	t.Cleanup(func() {
		release()
		unlock()
		awaitPublication(t, stopped)
	})
	awaitPublication(t, reached)
	// Shutdown has entered but cannot take the mutex until this barrier is
	// released. Let the already-queued reconciliation finish ahead of it.
	unlock()
	waitReconciled(t, b)
	if got := b.currentOllamaProxyGeneration(); got != 0 {
		t.Fatalf("queued reconciliation spawned generation %d after shutdown entry, want 0", got)
	}
	if b.getProxySupervisor() != nil || b.proxyProcessHandle() != nil {
		t.Fatal("shutdown entry allowed queued proxy supervision")
	}
	release()
	awaitPublication(t, stopped)
}

// A listener change cannot start the proxy before Serve has decided whether it
// runs, nor after shutdown, when nothing would stop it.
func TestProxySupervisionIgnoresPolicyChangesOutsideItsLifetime(t *testing.T) {
	sb := newSupervisedBroker(t, false)
	b := sb.b
	sb.mustSetListener(t, true)
	waitReconciled(t, b)
	if b.getProxySupervisor() != nil {
		t.Fatal("a policy change started the proxy before startup decided")
	}

	b.startProxySupervision(false)
	pp := b.proxyProcessHandle()
	requireProxyRunning(t, pp, "the policy enabled the listener before startup")
	b.shutdownProxySupervision()
	requireProxyExited(t, pp, "shutdown")

	sb.mustSetListener(t, false)
	sb.mustSetListener(t, true)
	waitReconciled(t, b)
	if b.getProxySupervisor() != nil || b.proxyProcessHandle() != nil {
		t.Fatal("a policy change started a proxy after shutdown")
	}
}

// policy:set stays within the activation bound whatever the proxy does: the
// start and the stop run in the background, and an unresponsive child leaves
// the revision unactivated rather than holding the request.
func TestProxySupervisionKeepsPolicySetWithinItsBoundAgainstAnUnresponsiveProxy(t *testing.T) {
	sb := newSupervisedBroker(t, false)
	t.Setenv(fakeProxyEnv, fakeProxyHangs)
	b := sb.b
	b.startProxySupervision(false)
	limit := policyActivationTimeout + 2*time.Second

	enabled := sb.mustSetListener(t, true)
	if enabled.elapsed > limit {
		t.Fatalf("enabling the listener took %s, over the %s bound", enabled.elapsed, limit)
	}
	if enabled.activation == activationActive {
		t.Fatalf("a proxy that never answers activated the revision: %+v", enabled)
	}
	eventually(t, "the unresponsive proxy starting", func() bool { return b.getProxySupervisor() != nil })

	disabled := sb.mustSetListener(t, false)
	if disabled.elapsed > limit {
		t.Fatalf("disabling the listener took %s, over the %s bound", disabled.elapsed, limit)
	}
	if disabled.activation == activationActive {
		t.Fatalf("a proxy that never answers activated the revision: %+v", disabled)
	}
}

// The listener and a facade cannot be given one port: policy:set refuses a
// listener port that any configured facade uses, however the broker knows it.
func TestPolicySetRefusesAListenerPortAFacadeUses(t *testing.T) {
	const planned, bound = 40555, 40556
	for _, tc := range []struct {
		name  string
		port  int
		setup func(b *Broker)
	}{
		{"compatibility port", ollamaProxyProfile.FacadePort, func(*Broker) {}},
		{"planned port", planned, func(b *Broker) { b.ollamaState().startupPort.Store(planned) }},
		{"bound port", bound, func(b *Broker) {
			publishTestProxy(b, &proxyProcess{facadeState: readyFacade(ollamaProxyProfile.Name, bound)})
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sb := newSupervisedBroker(t, false, ollamaProxyProfile.Name)
			tc.setup(sb.b)
			before := sb.b.localPolicy().Policy

			got := sb.setListener(t, true, tc.port)

			if got.Error == nil || got.Error.Code != -32602 || !strings.Contains(got.Error.Message, "routing.listener.port") {
				t.Fatalf("policy:set with a facade's port %d: error %+v, want -32602 naming routing.listener.port", tc.port, got.Error)
			}
			if after := sb.b.localPolicy().Policy; after.Revision != before.Revision || after.Routing.Listener.Enabled {
				t.Fatalf("a refused listener port changed the policy: %+v", after.Routing.Listener)
			}
		})
	}

	t.Run("a node that does not front the engine", func(t *testing.T) {
		sb := newSupervisedBroker(t, false)
		if got := sb.setListener(t, true, ollamaProxyProfile.FacadePort); got.Error != nil {
			t.Fatalf("policy:set refused a port no facade uses: %+v", got.Error)
		}
	})
}

// The other direction: moving a facade onto the enabled listener's port is
// refused, so the overlap cannot be reached from either side.
func TestFacadeSetPortRefusesTheListenersPort(t *testing.T) {
	sb := newSupervisedBroker(t, true, ollamaProxyProfile.Name)
	call := func(port int) policyResponse {
		return sb.call(t, ollamaProxyProfile.ComponentName()+":set-port", map[string]int{"port": port}, 5*time.Second)
	}

	got := call(testListenerPort)
	if got.Error == nil || !strings.Contains(got.Error.Message, "routing listener") {
		t.Fatalf("set-port onto the listener's port: error %+v, want a refusal naming the routing listener", got.Error)
	}

	// Only that port is refused, and only while the listener is enabled.
	other := call(testListenerPort + 1)
	if other.Error != nil && strings.Contains(other.Error.Message, "routing listener") {
		t.Fatalf("set-port onto another port was refused for the listener: %+v", other.Error)
	}
	sb.mustSetListener(t, false)
	released := call(testListenerPort)
	if released.Error != nil && strings.Contains(released.Error.Message, "routing listener") {
		t.Fatalf("a disabled listener still holds its port: %+v", released.Error)
	}
}
