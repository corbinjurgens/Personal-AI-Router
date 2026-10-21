// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
)

const (
	egressProcessOrigin = "g-t4-origin"
	egressProcessModel  = "g-t4-concrete"
	egressProcessRef    = "NVPAIR_PROVIDER_SECRET_G_T4_UNUSED"
	egressBindingError  = "egress lease binding, sequence or deadline refused"
)

// PC3 executes these cases. Wire success below is process-control evidence;
// the disabled build never turns a lease acknowledgement into public dispatch.
func TestPublicEgressLeaseProcessDisabled(t *testing.T) {
	upstream := newEgressProcessUpstream(t)
	f := newProxyServiceFixture(t)
	// Reuse the existing, registered empty-directory scanner role. No mDNS,
	// installed engine discovery, or additional worker executable is needed.
	self, err := os.Executable()
	if err != nil {
		t.Fatalf("locate existing scanner fixture: %v", err)
	}
	jointCopyExecutable(t, self, f.scannerPath)
	f.env = append(f.env, "NVPAIR_TEST_SUBSCRIPTION_SCANNER="+f.scannerPath,
		"NVPAIR_TEST_SUBSCRIPTION_PORT="+strconv.Itoa(freePort(t)),
		"USERPROFILE="+f.configDir, "OLLAMA_HOST=")
	policy := egressProcessPolicy(t, "17", upstream.publicPort)
	jointWriteJSON(t, f.policyPath(), policy)
	f.start()
	c := f.readyClient()
	nextID := 2000
	call := func(t *testing.T, method string, params any) jsonrpc.Message {
		t.Helper()
		nextID++
		m, err := c.call(strconv.Itoa(nextID), method, egressProcessJSON(t, params), serviceCallTimeout)
		if err != nil {
			t.Fatalf("broker %s transport: %v", method, err)
		}
		return *m
	}
	view := egressProcessActivePolicy(t, call, policy.Revision)
	if !reflect.DeepEqual(view.Policy, policy) {
		t.Fatal("broker changed the disabled staged policy")
	}
	stored := assertStoredPolicy(t, f.policyPath(), policy)
	egressProcessAddConcrete(t, call, "llamacpp-proxy:", upstream.localPort)
	client := egressProcessHTTPClient(t)
	egressProcessAssertConcrete(t, client, f.facadePort)
	before := upstream.localCalls.Load()

	t.Run("policy stays active across renewal intervals", func(t *testing.T) {
		assertActive := func() {
			t.Helper()
			reply := call(t, "policy:get", map[string]any{})
			egressProcessOK(t, reply)
			snapshot := decodeServicePolicySnapshot(t, &reply)
			if snapshot.Activation != "active" || snapshot.ActiveRevision != policy.Revision || snapshot.Policy.Revision != policy.Revision {
				t.Fatalf("policy changed during renewal: state=%q active=%q persisted=%q, want active revision %q",
					snapshot.Activation, snapshot.ActiveRevision, snapshot.Policy.Revision, policy.Revision)
			}
		}
		// Observe the real broker/proxy across renewal ticks without waiting
		// for an error state to recover. Each direct read must remain active.
		window := time.NewTimer(2 * nodepolicy.EgressLeaseRenewalInterval)
		defer window.Stop()
		tick := time.NewTicker(nodepolicy.EgressLeaseRenewalInterval / 4)
		defer tick.Stop()
		assertActive()
		for {
			select {
			case <-window.C:
				assertActive()
				return
			case <-tick.C:
				assertActive()
			}
		}
	})

	t.Run("model configuration and listing grant no public authority", func(t *testing.T) {
		got := call(t, "policy:get", map[string]any{})
		egressProcessOK(t, got)
		var configuration struct {
			ModelConfiguration *nodepolicy.ModelConfiguration `json:"modelConfiguration"`
		}
		egressProcessDecode(t, got.Result, &configuration)
		if configuration.ModelConfiguration == nil {
			t.Fatal("policy readback omitted modelConfiguration")
		}
		if err := configuration.ModelConfiguration.Validate(); err != nil {
			t.Fatalf("invalid model configuration: %v", err)
		}
		if configuration.ModelConfiguration.PolicyRevision != "17" {
			t.Fatalf("model configuration revision = %q, want 17", configuration.ModelConfiguration.PolicyRevision)
		}
		body := egressProcessHTTP(t, client, http.MethodGet, f.facadePort, "/v1/models", nil, http.StatusOK)
		var listing struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		egressProcessDecode(t, body, &listing)
		ids := make([]string, 0, len(listing.Data))
		for _, model := range listing.Data {
			ids = append(ids, model.ID)
		}
		for _, want := range []string{egressProcessModel, "weak", "weak-api", "weak-local"} {
			if !slices.Contains(ids, want) {
				t.Fatalf("model listing omitted %q: %v", want, ids)
			}
		}
		egressProcessAssertDisabled(t, client, f.facadePort)
	})

	t.Run("enabled deployment and remote or paid permissions are refused", func(t *testing.T) {
		reject := func(name string, change func(*nodepolicy.Policy), message string) {
			t.Helper()
			t.Run(name, func(t *testing.T) {
				draft := egressProcessPolicy(t, "17", upstream.publicPort)
				change(&draft)
				reply := call(t, "policy:set", map[string]any{"policy": draft, "expectedPersistedRevision": "17"})
				egressProcessRefused(t, reply, -32602, message)
				after := egressProcessActivePolicy(t, call, "17")
				if !reflect.DeepEqual(after.Policy, policy) || !bytes.Equal(readPolicyBytes(t, f.policyPath()), stored) {
					t.Fatal("rejected public configuration changed the active or persisted policy")
				}
			})
		}
		reject("qualified deployment remains disabled", func(p *nodepolicy.Policy) {
			p.Routing.Deployments[0].Enabled = true
		}, `routing.deployments[0] "g-public": a deployment cannot be enabled: remote execution is not available yet`)
		reject("remote permission remains disabled", func(p *nodepolicy.Policy) {
			rules := p.Routing.Tiers["weak"]
			rules.RemoteAllowed = true
			p.Routing.Tiers["weak"] = rules
		}, "routing.tiers.weak: remote and paid execution cannot be enabled: remote execution is not available yet")
		reject("paid permission remains disabled", func(p *nodepolicy.Policy) {
			rules := p.Routing.Tiers["weak"]
			rules.RemoteAllowed, rules.PaidAllowed = true, true
			p.Routing.Tiers["weak"] = rules
		}, "routing.tiers.weak: remote and paid execution cannot be enabled: remote execution is not available yet")
	})

	t.Run("public lease requests and notifications cannot authorize dispatch", func(t *testing.T) {
		codec := jsonrpc.NewCodecMaxFrameWithWriteInterrupt(c.conn, jsonrpc.WorkerFrameBytes, c.closeConnection)
		for _, method := range []string{nodepolicy.MethodRenewEgressLease, nodepolicy.MethodRevokeEgressLease} {
			params := map[string]string{"runId": "forged-run", "sequence": "9223372036854775807"}
			if method == nodepolicy.MethodRenewEgressLease {
				params["revision"], params["notAfterNanos"] = "17", "0"
			}
			egressProcessRefused(t, call(t, method, params), -32601, "method not found: "+method)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			err := codec.NotifyContext(ctx, method, params)
			cancel()
			if err != nil {
				t.Fatalf("write forged public notification: %v", err)
			}
		}
		// The following response fences the notifications on the same pipe,
		// proving pipe survival while public dispatch remains disabled. It does
		// not prove that forged notifications cannot alter an enabled lease.
		egressProcessActivePolicy(t, call, "17")
		egressProcessAssertDisabled(t, client, f.facadePort)
	})
	if got := upstream.localCalls.Load(); got != before {
		t.Fatalf("listing or refused API request dispatched concrete inference: got %d, want %d", got, before)
	}
	egressProcessAssertConcrete(t, client, f.facadePort)
	if err := f.cleanupChecked(); err != nil {
		t.Fatalf("stop disabled broker/proxy fixture: %v", err)
	}
	f.assertStopped()
	upstream.assertNoPublic(t)
}

func TestPublicEgressLeaseProcessEOF(t *testing.T) {
	base := t.TempDir()
	upstream := newEgressProcessUpstream(t)
	p := startEgressProcessProxy(t, base)
	policy := egressProcessPolicy(t, "17", upstream.publicPort)
	ack := p.activate(t, policy)
	anchor := p.renew(t, ack.RunID, "17", "1", "0")
	grant := nodepolicy.RenewEgressLeaseParams{RunID: ack.RunID, Revision: "17", Sequence: "2", NotAfterNanos: egressProcessDeadline(t, anchor)}
	reply, oldFrame, oldID := p.callFrame(t, nodepolicy.MethodRenewEgressLease, grant)
	egressProcessRenewAck(t, reply, ack.RunID, "17", "2")
	port := p.enable(t)

	cancelled := make(chan struct{})
	committed := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	stop := func() { releaseOnce.Do(func() { close(release) }) }
	stream := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "unexpected concrete route", http.StatusNotFound)
			return
		}
		var request struct {
			Model  string `json:"model"`
			Stream bool   `json:"stream"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&request); err != nil {
			t.Errorf("decode streaming concrete request: %v", err)
			return
		}
		if request.Model != egressProcessModel || !request.Stream {
			t.Error("stream reached the fake engine without the concrete model and stream flag")
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ready\"}}]}\n\n"); err != nil {
			t.Errorf("write committed concrete event: %v", err)
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			t.Errorf("flush committed concrete event: %v", err)
			return
		}
		close(committed)
		select {
		case <-r.Context().Done():
			close(cancelled)
		case <-release:
		}
	}))
	t.Cleanup(func() { stop(); stream.Close() })
	egressProcessAddConcrete(t, p.call, "llamacpp:", portOfURL(t, stream.URL))
	client := egressProcessHTTPClient(t)
	client.Timeout = 0 // The request context owns its lifetime throughout the EOF proof.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port),
		strings.NewReader(`{"model":"g-t4-concrete","messages":[],"stream":true}`))
	if err != nil {
		t.Fatalf("create streaming concrete request: %v", err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		t.Fatalf("start streaming concrete request: %v", err)
	}
	t.Cleanup(func() {
		if err := response.Body.Close(); err != nil {
			t.Errorf("close concrete stream: %v", err)
		}
	})
	if response.StatusCode != http.StatusOK {
		t.Fatalf("concrete stream status = %d, want 200", response.StatusCode)
	}
	reader := bufio.NewReader(response.Body)
	line, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("read first committed concrete event: %v", err)
	}
	data, ok := strings.CutPrefix(strings.TrimSpace(line), "data: ")
	if !ok {
		t.Fatal("first committed concrete event was not SSE data")
	}
	var event struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	egressProcessDecode(t, []byte(data), &event)
	if len(event.Choices) != 1 || event.Choices[0].Delta.Content != "ready" {
		t.Fatal("client did not receive the committed concrete event")
	}
	egressProcessBarrier(t, committed, "upstream stream commitment")
	readDone := make(chan struct{})
	var streamReadErr error
	go func() { _, streamReadErr = io.Copy(io.Discard, reader); close(readDone) }()
	t.Cleanup(func() {
		cancel()
		stop()
		select {
		case <-readDone:
		case <-time.After(5 * time.Second):
			t.Error("stream client reader did not join")
		}
	})
	select {
	case <-cancelled:
		t.Fatal("upstream was cancelled before broker stdin EOF")
	case <-readDone:
		t.Fatal("client stream ended before broker stdin EOF")
	case <-p.done:
		t.Fatal("proxy exited before broker stdin EOF")
	case <-time.After(200 * time.Millisecond):
	}
	// Leave the HTTP request, response body and parent stdout reader alive.
	// This closes only this child's trusted-parent input, not its client.
	if err := p.stdio.closeInput(); err != nil {
		t.Fatalf("close broker stdin only: %v", err)
	}
	egressProcessBarrier(t, cancelled, "upstream request-context cancellation after broker EOF")
	if err := ctx.Err(); err != nil {
		t.Fatalf("client context ended during the broker EOF proof: %v", err)
	}
	p.awaitExit(t)
	select {
	case <-readDone:
		// EOF and an interrupted committed stream are both valid outcomes.
		t.Logf("concrete_stream_join read_error=%t", streamReadErr != nil)
	case <-time.After(5 * time.Second):
		t.Fatal("client stream outlived the exited proxy")
	}

	replacement := startEgressProcessProxy(t, base)
	newAck := replacement.activate(t, policy)
	newAnchor := replacement.renew(t, newAck.RunID, "17", "1", "0")
	if newAnchor.RunID == anchor.RunID {
		t.Fatal("replacement reused the departed proxy run")
	}
	newPort := replacement.enable(t)
	egressProcessAssertDisabled(t, client, newPort)
	// Replay the exact old bytes through the NEW pipe. Rejection can depend on
	// the old run, sequence or deadline; this is a compound replay check. The
	// incarnation case below isolates wrong-run rejection with an anchor-only
	// renewal and a fresh sequence. A cached reply cannot satisfy this read.
	egressProcessRefused(t, replacement.exchange(t, oldID, oldFrame), -32602, egressBindingError)
	replacement.renew(t, newAck.RunID, "17", "2", "0")
	egressProcessAssertDisabled(t, client, newPort)
	replacement.stop(t)
	upstream.assertNoPublic(t)
}

func TestPublicEgressLeaseProcessIncarnation(t *testing.T) {
	upstream := newEgressProcessUpstream(t)
	p := startEgressProcessProxy(t, t.TempDir())
	policy := egressProcessPolicy(t, "17", upstream.publicPort)
	egressProcessRefused(t, p.call(t, nodepolicy.MethodRenewEgressLease, nodepolicy.RenewEgressLeaseParams{
		RunID: "unbound-run", Revision: "17", Sequence: "1", NotAfterNanos: "0",
	}), -32602, egressBindingError)
	ack := p.activate(t, policy)
	port := p.enable(t)
	egressProcessAddConcrete(t, p.call, "llamacpp:", upstream.localPort)
	client := egressProcessHTTPClient(t)
	egressProcessAssertConcrete(t, client, port)
	anchor := p.renew(t, ack.RunID, "17", "1", "0")
	deadline := egressProcessDeadline(t, anchor)
	p.renew(t, ack.RunID, "17", "3", deadline)
	egressProcessAssertDisabled(t, client, port)
	rejectRenew := func(name, run, revision, sequence, notAfter, message string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			egressProcessRefused(t, p.call(t, nodepolicy.MethodRenewEgressLease, nodepolicy.RenewEgressLeaseParams{
				RunID: run, Revision: revision, Sequence: sequence, NotAfterNanos: notAfter,
			}), -32602, message)
		})
	}
	rejectRenew("duplicate grant", ack.RunID, "17", "3", deadline, egressBindingError)
	rejectRenew("reordered grant", ack.RunID, "17", "2", deadline, egressBindingError)
	rejectRenew("wrong run cannot poison highwater", "wrong-run", "17", "100", "0", egressBindingError)
	rejectRenew("wrong revision cannot poison highwater", ack.RunID, "18", "100", "0", egressBindingError)
	rejectRenew("noncanonical sequence is rejected by the wire validator", ack.RunID, "17", "04", "0",
		"invalid params: sequence must be canonical decimal digits within the signed 64-bit range")
	anchor = p.renew(t, ack.RunID, "17", "4", "0")
	p.revoke(t, ack.RunID, "6")
	rejectRenew("pre-revoke grant cannot restore authority", ack.RunID, "17", "5", egressProcessDeadline(t, anchor), egressBindingError)
	egressProcessRefused(t, p.call(t, nodepolicy.MethodRevokeEgressLease, nodepolicy.RevokeEgressLeaseParams{
		RunID: ack.RunID, Sequence: "6",
	}), -32602, egressBindingError)
	egressProcessAssertDisabled(t, client, port)
	anchor = p.renew(t, ack.RunID, "17", "7", "0")
	p.renew(t, ack.RunID, "17", "8", egressProcessDeadline(t, anchor))

	policy.Revision = "18"
	newAck := p.activate(t, policy)
	if newAck.RunID != ack.RunID {
		t.Fatal("policy activation changed the live process run")
	}
	rejectRenew("old revision after activation", ack.RunID, "17", "100", "0", egressBindingError)
	rejectRenew("activation preserves sequence highwater", ack.RunID, "18", "8", "0", egressBindingError)
	anchor = p.renew(t, ack.RunID, "18", "9", "0")
	p.renew(t, ack.RunID, "18", "10", egressProcessDeadline(t, anchor))
	egressProcessRefused(t, p.call(t, nodepolicy.MethodRevokeEgressLease, nodepolicy.RevokeEgressLeaseParams{
		RunID: "wrong-run", Sequence: "100",
	}), -32602, egressBindingError)
	egressProcessRefused(t, p.call(t, nodepolicy.MethodRevokeEgressLease, nodepolicy.RevokeEgressLeaseParams{
		RunID: ack.RunID, Sequence: "9",
	}), -32602, egressBindingError)
	p.renew(t, ack.RunID, "18", "11", "0")
	p.revoke(t, ack.RunID, "12")
	rejectRenew("old grant after final revoke", ack.RunID, "18", "10", "0", egressBindingError)
	p.renew(t, ack.RunID, "18", "13", "0")
	egressProcessAssertDisabled(t, client, port)
	if got := upstream.localCalls.Load(); got != 1 {
		t.Fatalf("lease traffic or refused API inference reached concrete engine: calls=%d, want 1", got)
	}
	egressProcessAssertConcrete(t, client, port)
	p.stop(t)
	upstream.assertNoPublic(t)
}

type egressProcessCall func(*testing.T, string, any) jsonrpc.Message

func egressProcessPolicy(t *testing.T, revision string, publicPort int) nodepolicy.Policy {
	t.Helper()
	if _, exists := os.LookupEnv(egressProcessRef); exists {
		t.Fatal("synthetic unused account reference unexpectedly exists in the environment")
	}
	p := nodepolicy.Default()
	p.Revision = revision
	p.Tiers["weak"] = []nodepolicy.TierEntry{
		{Kind: nodepolicy.EntryAPI, Deployment: "g-public"},
		{Kind: nodepolicy.EntryLocal, Engine: "llamacpp", Model: egressProcessModel},
	}
	p.Routing.Accounts = []nodepolicy.Account{{ID: "g-account", Credential: egressProcessRef, Billing: nodepolicy.BillingPaid,
		PerRequestCeiling: "1000", DailyCeiling: "10000", MonthlyCeiling: "100000"}}
	p.Routing.Deployments = []nodepolicy.Deployment{{ID: "g-public", Account: "g-account", Model: "g-t4-public", Family: "g-family",
		ContextTokens: 8192, MaxOutputTokens: 1024, Enabled: false,
		// The .invalid name is defensive only. This fake endpoint does not
		// intercept publicClient.Do, so its counter cannot prove that no public
		// transport was attempted or qualify a public transport.
		Endpoint: fmt.Sprintf("https://public-egress.invalid:%d/v1/chat/completions", publicPort),
		Protocol: nodepolicy.ProtocolOpenAIChatCompletions,
		Prices:   &nodepolicy.Prices{Currency: "USD", Unit: "nanoUSDPerMillionTokens", InputPerMillion: "0", OutputPerMillion: "0"},
		Transport: &nodepolicy.Transport{ConnectTimeoutSeconds: 1, HeaderTimeoutSeconds: 2, IdleTimeoutSeconds: 2,
			MaxResponseBytes: 65536, MaxErrorBodyBytes: 4096, MaxEventBytes: 4096},
		Qualification: &nodepolicy.QualificationRef{ID: "g-synthetic-not-qualified", Revision: "1"}}}
	if err := p.Validate(); err != nil {
		t.Fatalf("invalid disabled staged policy: %v", err)
	}
	// Parse the actual wire representation so expected normalized state is
	// independent of the child's subsequent policy readback.
	parsed, err := nodepolicy.Parse(egressProcessJSON(t, p))
	if err != nil {
		t.Fatalf("parse disabled staged policy: %v", err)
	}
	return parsed
}

func egressProcessActivePolicy(t *testing.T, call egressProcessCall, revision string) servicePolicySnapshot {
	t.Helper()
	deadline := time.NewTimer(10 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(25 * time.Millisecond)
	defer tick.Stop()
	for {
		reply := call(t, "policy:get", map[string]any{})
		view := decodeServicePolicySnapshot(t, &reply)
		if view.Policy.Revision != revision {
			t.Fatalf("persisted revision = %q, want %q", view.Policy.Revision, revision)
		}
		if view.Activation == "active" && view.ActiveRevision == revision {
			return view
		}
		select {
		case <-tick.C:
		case <-deadline.C:
			t.Fatalf("policy did not activate: desired=%q active=%q state=%q", revision, view.ActiveRevision, view.Activation)
		}
	}
}

type egressProcessUpstream struct {
	localPort, publicPort int
	localCalls            atomic.Int32
	publicCalls           atomic.Int32
	unexpectedCalls       atomic.Int32
}

func newEgressProcessUpstream(t *testing.T) *egressProcessUpstream {
	t.Helper()
	u := &egressProcessUpstream{}
	local := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet && r.URL.Path == "/models" {
			if _, err := io.WriteString(w, `{"object":"list","data":[{"id":"g-t4-concrete","object":"model"}]}`); err != nil {
				t.Errorf("write concrete model list: %v", err)
			}
			return
		}
		var request struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&request); err != nil {
			u.unexpectedCalls.Add(1)
			http.Error(w, "invalid fixture request", http.StatusBadRequest)
			return
		}
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" || request.Model != egressProcessModel {
			u.unexpectedCalls.Add(1)
			http.Error(w, "unexpected fixture request", http.StatusBadRequest)
			return
		}
		u.localCalls.Add(1)
		if _, err := io.WriteString(w, `{"id":"g-concrete","object":"chat.completion","model":"g-t4-concrete","choices":[{"message":{"role":"assistant","content":"local-control"},"finish_reason":"stop"}]}`); err != nil {
			t.Errorf("write concrete response: %v", err)
		}
	}))
	public := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		u.publicCalls.Add(1)
		w.WriteHeader(http.StatusBadGateway)
	}))
	t.Cleanup(local.Close)
	t.Cleanup(public.Close)
	u.localPort, u.publicPort = portOfURL(t, local.URL), portOfURL(t, public.URL)
	return u
}

func (u *egressProcessUpstream) assertNoPublic(t *testing.T) {
	t.Helper()
	if got := u.publicCalls.Load(); got != 0 {
		t.Fatalf("fake public endpoint calls = %d, want zero", got)
	}
	if got := u.unexpectedCalls.Load(); got != 0 {
		t.Fatalf("non-concrete calls escaped to the fake local engine: %d", got)
	}
}

func egressProcessAddConcrete(t *testing.T, call egressProcessCall, prefix string, port int) {
	t.Helper()
	reply := call(t, prefix+"node/add-manual", map[string]any{"id": "g-concrete-node", "host": "127.0.0.1", "port": port,
		"addresses": []string{"127.0.0.1"}, "models": []string{egressProcessModel}})
	egressProcessOK(t, reply)
	var added struct {
		Added bool `json:"added"`
	}
	egressProcessDecode(t, reply.Result, &added)
	if !added.Added {
		t.Fatal("fresh concrete fixture node was not added")
	}
	reply = call(t, prefix+"node/select", map[string]string{"id": "g-concrete-node"})
	egressProcessOK(t, reply)
	var selected struct {
		ID string `json:"id"`
	}
	egressProcessDecode(t, reply.Result, &selected)
	if selected.ID != "g-concrete-node" {
		t.Fatalf("selected concrete fixture = %q, want g-concrete-node", selected.ID)
	}
}

func egressProcessHTTPClient(t *testing.T) *http.Client {
	t.Helper()
	client := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{}}
	t.Cleanup(client.CloseIdleConnections)
	return client
}

func egressProcessHTTP(t *testing.T, client *http.Client, method string, port int, path string, body []byte, wantStatus int) []byte {
	t.Helper()
	request, err := http.NewRequest(method, fmt.Sprintf("http://127.0.0.1:%d%s", port, path), bytes.NewReader(body))
	if err != nil {
		t.Fatalf("create fixture HTTP request: %v", err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		t.Fatalf("fixture HTTP %s %s: %v", method, path, err)
	}
	data, readErr := io.ReadAll(io.LimitReader(response.Body, (128<<10)+1))
	closeErr := response.Body.Close()
	if readErr != nil || closeErr != nil || len(data) > 128<<10 {
		t.Fatalf("read fixture HTTP response: read=%v close=%v bytes=%d", readErr, closeErr, len(data))
	}
	if response.StatusCode != wantStatus {
		t.Fatalf("fixture HTTP %s %s status=%d, want %d", method, path, response.StatusCode, wantStatus)
	}
	return data
}

func egressProcessAssertDisabled(t *testing.T, client *http.Client, port int) {
	t.Helper()
	body := egressProcessHTTP(t, client, http.MethodPost, port, "/v1/chat/completions",
		[]byte(`{"model":"weak-api","messages":[],"stream":false}`), http.StatusBadRequest)
	var refused struct {
		Code   string `json:"code"`
		Reason string `json:"reason"`
		Model  string `json:"model"`
	}
	egressProcessDecode(t, body, &refused)
	if refused.Code != "feature_unavailable" || refused.Reason != "alias_not_inference_ready" || refused.Model != "weak-api" {
		t.Fatalf("API inference refusal = %+v", refused)
	}
}

func egressProcessAssertConcrete(t *testing.T, client *http.Client, port int) {
	t.Helper()
	body := egressProcessHTTP(t, client, http.MethodPost, port, "/v1/chat/completions",
		[]byte(`{"model":"g-t4-concrete","messages":[],"stream":false}`), http.StatusOK)
	var result struct {
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	egressProcessDecode(t, body, &result)
	if result.Model != egressProcessModel || len(result.Choices) != 1 || result.Choices[0].Message.Content != "local-control" {
		t.Fatal("concrete positive control did not return the fake engine's structured result")
	}
}

// closeInput deliberately does not close the read side. In particular, the
// EOF test must keep draining stdout until the child has cancelled and exited.
type egressProcessStdio struct {
	io.Reader
	io.Writer
	input     io.WriteCloser
	output    io.ReadCloser
	inputOnce sync.Once
	inputErr  error
}

func (s *egressProcessStdio) closeInput() error {
	s.inputOnce.Do(func() { s.inputErr = s.input.Close() })
	return s.inputErr
}

func (s *egressProcessStdio) Close() error {
	inputErr := s.closeInput()
	outputErr := s.output.Close()
	if errors.Is(outputErr, os.ErrClosed) {
		outputErr = nil // exec.Wait already closes StdoutPipe after its drain.
	}
	return errors.Join(inputErr, outputErr)
}

type egressProcessProxy struct {
	cmd       *exec.Cmd
	stdio     *egressProcessStdio
	client    *serviceRPCClient
	output    serviceOutput
	done      chan struct{}
	waitErr   error
	nextID    int
	ports     []int
	inventory *serviceProcessInventory
}

func startEgressProcessProxy(t *testing.T, base string) *egressProcessProxy {
	t.Helper()
	cluster := filepath.Join(base, "cluster")
	if err := os.MkdirAll(cluster, 0o700); err != nil {
		t.Fatalf("create private proxy cluster directory: %v", err)
	}
	path, err := canonicalServicePath(proxyBin)
	if err != nil {
		t.Fatalf("resolve registered proxy fixture path: %v", err)
	}
	p := &egressProcessProxy{cmd: exec.Command(proxyBin, "--cluster-dir", cluster, "--log-level", "error"), done: make(chan struct{}), nextID: 3000,
		inventory: &serviceProcessInventory{paths: []string{path}, known: make(map[int32]serviceProcessIdentity)}}
	p.cmd.Dir = base
	p.cmd.Env = append(os.Environ(), "HOME="+base, "XDG_CONFIG_HOME="+base, "APPDATA="+base, "LOCALAPPDATA="+base, "USERPROFILE="+base, "OLLAMA_HOST=")
	p.cmd.Stderr = &p.output
	p.cmd.WaitDelay = 2 * time.Second
	input, err := p.cmd.StdinPipe()
	if err != nil {
		t.Fatalf("proxy stdin pipe: %v", err)
	}
	output, err := p.cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("proxy stdout pipe: %v; close input: %v", err, input.Close())
	}
	if err := p.cmd.Start(); err != nil {
		t.Fatalf("start registered proxy: %v; close pipes: %v", err, errors.Join(input.Close(), output.Close()))
	}
	p.stdio = &egressProcessStdio{Reader: output, Writer: input, input: input, output: output}
	p.client = newServiceRPCClient(p.stdio)
	go func() {
		<-p.client.done
		p.waitErr = p.cmd.Wait()
		close(p.done)
	}()
	t.Cleanup(func() {
		if err := p.stdio.closeInput(); err != nil {
			t.Errorf("close proxy parent input: %v", err)
		}
		select {
		case <-p.done:
		case <-time.After(8 * time.Second):
			t.Error("proxy required forced cleanup")
			if err := p.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
				t.Errorf("kill owned proxy: %v", err)
			}
			if err := p.stdio.output.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
				t.Errorf("close stuck proxy stdout: %v", err)
			}
		}
		p.awaitExit(t)
		if err := p.client.close(); err != nil {
			t.Errorf("join proxy codec reader: %v", err)
		}
		for _, port := range p.ports {
			assertInteropListenerClosed(t, port)
		}
	})
	owned, err := p.inventory.scan()
	if err != nil || len(owned) != 1 || owned[0].PID != int32(p.cmd.Process.Pid) {
		t.Fatalf("registered proxy inventory: owned=%+v error=%v", owned, err)
	}
	t.Logf("egress_proxy_started pid=%d path=%q", p.cmd.Process.Pid, path)
	return p
}

func (p *egressProcessProxy) awaitExit(t *testing.T) {
	t.Helper()
	select {
	case <-p.done:
		if p.waitErr != nil {
			t.Fatalf("proxy exit: %v; stderr: %s", p.waitErr, p.output.String())
		}
	case <-time.After(8 * time.Second):
		t.Fatal("proxy did not drain stdout and exit within the EOF deadline")
	}
	p.client.mu.Lock()
	readErr := errors.Join(p.client.readErr, p.client.budgetErr)
	p.client.mu.Unlock()
	if readErr != nil {
		t.Fatalf("proxy codec drain: %v", readErr)
	}
	owned, err := p.inventory.scan()
	if err != nil || len(owned) != 0 {
		t.Fatalf("proxy survivor inventory: owned=%+v error=%v", owned, err)
	}
	t.Logf("egress_proxy_exited pid=%d owned_running=0", p.cmd.Process.Pid)
}

func (p *egressProcessProxy) stop(t *testing.T) {
	t.Helper()
	if err := p.stdio.closeInput(); err != nil {
		t.Fatalf("close proxy parent input: %v", err)
	}
	p.awaitExit(t)
}

func (p *egressProcessProxy) call(t *testing.T, method string, params any) jsonrpc.Message {
	t.Helper()
	reply, _, _ := p.callFrame(t, method, params)
	return reply
}

func (p *egressProcessProxy) callFrame(t *testing.T, method string, params any) (jsonrpc.Message, []byte, string) {
	t.Helper()
	p.nextID++
	id := strconv.Itoa(p.nextID)
	rawID := json.RawMessage(id)
	frame := append(egressProcessJSON(t, jsonrpc.Message{JSONRPC: "2.0", ID: &rawID, Method: method, Params: egressProcessJSON(t, params)}), '\n')
	return p.exchange(t, id, frame), frame, id
}

func (p *egressProcessProxy) exchange(t *testing.T, id string, frame []byte) jsonrpc.Message {
	t.Helper()
	writeDone := make(chan error, 1)
	go func() {
		n, err := p.stdio.Write(frame)
		if err == nil && n != len(frame) {
			err = io.ErrShortWrite
		}
		writeDone <- err
	}()
	select {
	case err := <-writeDone:
		if err != nil {
			t.Fatalf("write proxy frame: %v", err)
		}
	case <-time.After(5 * time.Second):
		closeErr := p.stdio.closeInput()
		select {
		case <-writeDone:
		case <-time.After(time.Second):
			t.Error("proxy frame writer did not join after closing stdin")
		}
		t.Fatalf("proxy frame write timed out; input close: %v", closeErr)
	}
	reply, err := p.client.response(id, 5*time.Second)
	if err != nil {
		t.Fatalf("read proxy reply %s: %v", id, err)
	}
	return *reply
}

func (p *egressProcessProxy) activate(t *testing.T, policy nodepolicy.Policy) nodepolicy.PolicyAck {
	t.Helper()
	reply := p.call(t, nodepolicy.MethodSetPolicy, nodepolicy.SetPolicyParams{OriginNodeID: egressProcessOrigin, Policy: policy})
	egressProcessOK(t, reply)
	var ack nodepolicy.PolicyAck
	egressProcessDecode(t, reply.Result, &ack)
	if ack.OriginNodeID != egressProcessOrigin || ack.Revision != policy.Revision || ack.RunID == "" {
		t.Fatalf("origin-bound policy acknowledgement = %+v", ack)
	}
	return ack
}

func (p *egressProcessProxy) enable(t *testing.T) int {
	t.Helper()
	for range 8 {
		port := freePort(t)
		reply := p.call(t, "facade/enable", map[string]any{"engine": "llamacpp", "port": port, "ignorePersistedPort": true})
		if reply.Error != nil && isBindRace(reply.Error) {
			continue
		}
		egressProcessOK(t, reply)
		var enabled struct {
			Engine string `json:"engine"`
			Port   int    `json:"port"`
		}
		egressProcessDecode(t, reply.Result, &enabled)
		if enabled.Engine != "llamacpp" || enabled.Port != port {
			t.Fatalf("facade acknowledgement = %+v, want llamacpp port %d", enabled, port)
		}
		p.ports = append(p.ports, port)
		return port
	}
	t.Fatal("all proxy facade port reservations lost their bind race")
	return 0
}

func (p *egressProcessProxy) renew(t *testing.T, run, revision, sequence, deadline string) nodepolicy.RenewEgressLeaseResult {
	t.Helper()
	reply := p.call(t, nodepolicy.MethodRenewEgressLease, nodepolicy.RenewEgressLeaseParams{
		RunID: run, Revision: revision, Sequence: sequence, NotAfterNanos: deadline,
	})
	return egressProcessRenewAck(t, reply, run, revision, sequence)
}

func egressProcessRenewAck(t *testing.T, reply jsonrpc.Message, run, revision, sequence string) nodepolicy.RenewEgressLeaseResult {
	t.Helper()
	egressProcessOK(t, reply)
	var ack nodepolicy.RenewEgressLeaseResult
	egressProcessDecode(t, reply.Result, &ack)
	if ack.RunID != run || ack.Revision != revision || ack.Sequence != sequence {
		t.Fatalf("renewal acknowledgement = %+v, want run=%q revision=%q sequence=%q", ack, run, revision, sequence)
	}
	return ack
}

func (p *egressProcessProxy) revoke(t *testing.T, run, sequence string) {
	t.Helper()
	reply := p.call(t, nodepolicy.MethodRevokeEgressLease, nodepolicy.RevokeEgressLeaseParams{RunID: run, Sequence: sequence})
	egressProcessOK(t, reply)
	var ack nodepolicy.RevokeEgressLeaseResult
	egressProcessDecode(t, reply.Result, &ack)
	if ack.RunID != run || ack.Sequence != sequence {
		t.Fatalf("revoke acknowledgement = %+v, want run=%q sequence=%q", ack, run, sequence)
	}
}

func egressProcessDeadline(t *testing.T, anchor nodepolicy.RenewEgressLeaseResult) string {
	t.Helper()
	now, err := strconv.ParseInt(anchor.NowNanos, 10, 64)
	if err != nil || now < 0 || now > (1<<63-1)-int64(4*time.Second) {
		t.Fatalf("invalid monotonic anchor %q: %v", anchor.NowNanos, err)
	}
	return strconv.FormatInt(now+int64(4*time.Second), 10)
}

func egressProcessJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode process fixture JSON: %v", err)
	}
	return raw
}

func egressProcessDecode(t *testing.T, raw []byte, target any) {
	t.Helper()
	if err := json.Unmarshal(raw, target); err != nil {
		t.Fatalf("decode process fixture JSON: %v", err)
	}
}

func egressProcessOK(t *testing.T, reply jsonrpc.Message) {
	t.Helper()
	if reply.Error != nil || len(reply.Result) == 0 || string(reply.Result) == "null" {
		t.Fatalf("process RPC failed: error=%+v result_bytes=%d", reply.Error, len(reply.Result))
	}
}

func egressProcessRefused(t *testing.T, reply jsonrpc.Message, code int, message string) {
	t.Helper()
	if reply.Error == nil || reply.Error.Code != code || reply.Error.Message != message || len(reply.Result) != 0 {
		t.Fatalf("process RPC refusal: error=%+v result_bytes=%d; want code=%d message=%q", reply.Error, len(reply.Result), code, message)
	}
}

func egressProcessBarrier(t *testing.T, done <-chan struct{}, description string) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(8 * time.Second):
		t.Fatalf("timed out waiting for %s", description)
	}
}
