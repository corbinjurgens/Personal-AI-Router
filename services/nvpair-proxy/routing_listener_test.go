// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Coverage for the engine-independent routing listener. Every listener here
// binds an ephemeral loopback port; none uses an engine's or the live
// application's port.

import (
	"bufio"
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

// listenerPolicy is the staged hybrid policy (weak API-only, medium mixed,
// strong empty) at revision, carrying the given listener settings.
func listenerPolicy(t *testing.T, revision string, listener nodepolicy.Listener) nodepolicy.Policy {
	t.Helper()
	pol, err := nodepolicy.Parse([]byte(apiTierPolicy))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	pol.Revision = revision
	pol.Routing.Listener = listener
	if err := pol.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	return pol
}

// zeroFacadeProxy is a process with no engine facade at all.
func zeroFacadeProxy(t *testing.T) (*Proxy, *recordingWriter) {
	t.Helper()
	w := &recordingWriter{}
	p := NewProxy(NewCodec(w))
	t.Cleanup(func() { p.shutdown(context.Background()) })
	return p, w
}

func activate(t *testing.T, p *Proxy, w *recordingWriter, pol nodepolicy.Policy) {
	t.Helper()
	rpcOK(t, p, w, nodepolicy.MethodSetPolicy, pol)
}

// freeRoutingPort returns a port that was free on the IPv4 wildcard a moment
// ago. freeTCPPort probes only 127.0.0.1, and on macOS that succeeds beside a
// wildcard listener an earlier test has not yet released, which the routing
// listener rightly refuses as a collision.
func freeRoutingPort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "0.0.0.0:0")
	if err != nil {
		t.Fatalf("reserve free port: %v", err)
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
}

// boundPort returns the port the proxy's routing listener is bound to, or 0.
func boundPort(p *Proxy) int {
	p.routing.mu.Lock()
	defer p.routing.mu.Unlock()
	return p.routing.port
}

// answer is one HTTP response read in full.
type answer struct {
	status int
	header http.Header
	body   []byte
}

func (a answer) code(t *testing.T) string {
	t.Helper()
	var got struct {
		Code string `json:"code"`
	}
	if err := json.Unmarshal(a.body, &got); err != nil {
		t.Fatalf("decode error body %q: %v", a.body, err)
	}
	return got.Code
}

var noKeepAlive = &http.Client{
	Timeout:   5 * time.Second,
	Transport: &http.Transport{DisableKeepAlives: true},
}

// call sends one request to the listener on port. mutate adjusts the request
// before it is sent, for headers and a Host override.
func call(t *testing.T, port int, method, target string, mutate func(*http.Request)) answer {
	t.Helper()
	req, err := http.NewRequest(method, "http://127.0.0.1:"+strconv.Itoa(port)+target, strings.NewReader(`{"model":"weak"}`))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	if mutate != nil {
		mutate(req)
	}
	resp, err := noKeepAlive.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, target, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read %s %s: %v", method, target, err)
	}
	return answer{status: resp.StatusCode, header: resp.Header, body: body}
}

func withHeader(name, value string) func(*http.Request) {
	return func(r *http.Request) { r.Header.Set(name, value) }
}

func withHost(host string) func(*http.Request) {
	return func(r *http.Request) { r.Host = host }
}

// refuses reports whether nothing accepts connections on port.
func refuses(port int) bool {
	conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), time.Second)
	if err == nil {
		_ = conn.Close()
		return false
	}
	return true
}

func listedIDs(t *testing.T, a answer) []string {
	t.Helper()
	var list struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int    `json:"created"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := json.Unmarshal(a.body, &list); err != nil {
		t.Fatalf("decode model list %q: %v", a.body, err)
	}
	if list.Object != "list" || list.Data == nil {
		t.Fatalf("model list = %q, want an object %q with a data array", a.body, "list")
	}
	ids := []string{}
	for _, d := range list.Data {
		if d.Object != "model" || d.OwnedBy != "pair" {
			t.Errorf("record %+v, want object %q owned by %q", d, "model", "pair")
		}
		ids = append(ids, d.ID)
	}
	return ids
}

func TestRoutingListener_ListsAliasesWithNoEngineRunning(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: freeRoutingPort(t)}))
	port := boundPort(p)
	if port == 0 {
		t.Fatal("listener is not bound after activating an enabled policy")
	}
	if n := len(p.enabledFacades()); n != 0 {
		t.Fatalf("%d facades enabled, want none", n)
	}

	got := call(t, port, http.MethodGet, "/v1/models", nil)
	if got.status != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", got.status, got.body)
	}
	// weak is API-only and strong has no entry, so only weak and medium list.
	want := []string{"weak", "weak-api", "weak-local", "medium", "medium-api", "medium-local"}
	if ids := listedIDs(t, got); !slices.Equal(ids, want) {
		t.Errorf("listed %v, want %v", ids, want)
	}
	if ct := got.header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
}

func TestRoutingListener_ListsAnEmptyPolicyAsAnEmptyArray(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	pol := nodepolicy.Default()
	pol.Revision = "1"
	pol.Routing.Listener = nodepolicy.Listener{Enabled: true, Port: freeRoutingPort(t)}
	activate(t, p, w, pol)

	got := call(t, boundPort(p), http.MethodGet, "/v1/models", nil)
	if ids := listedIDs(t, got); len(ids) != 0 {
		t.Errorf("listed %v, want nothing", ids)
	}
}

func TestRoutingListener_ListFollowsTheActivatedPolicy(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	listener := nodepolicy.Listener{Enabled: true, Port: freeRoutingPort(t)}
	activate(t, p, w, listenerPolicy(t, "1", listener))
	port := boundPort(p)

	pol := nodepolicy.Default()
	pol.Revision = "2"
	pol.Routing.Listener = listener
	pol.Tiers["strong"] = []nodepolicy.TierEntry{{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "big:70b"}}
	activate(t, p, w, pol)

	if boundPort(p) != port {
		t.Fatalf("an unchanged listener setting rebound %d to %d", port, boundPort(p))
	}
	got := call(t, port, http.MethodGet, "/v1/models", nil)
	if ids, want := listedIDs(t, got), []string{"strong", "strong-api", "strong-local"}; !slices.Equal(ids, want) {
		t.Errorf("listed %v, want %v", ids, want)
	}
}

func TestRoutingListener_IsOffUntilEnabled(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{}))
	if port := boundPort(p); port != 0 {
		t.Fatalf("a default policy bound a listener on %d", port)
	}
	activate(t, p, w, listenerPolicy(t, "2", nodepolicy.Listener{Port: freeRoutingPort(t)}))
	if port := boundPort(p); port != 0 {
		t.Fatalf("a policy with a port but not enabled bound a listener on %d", port)
	}
}

func TestRoutingListener_BindsLoopbackOnly(t *testing.T) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		t.Fatalf("InterfaceAddrs: %v", err)
	}
	var lan string
	for _, a := range addrs {
		if ipNet, ok := a.(*net.IPNet); ok && ipNet.IP.To4() != nil && !ipNet.IP.IsLoopback() {
			lan = ipNet.IP.String()
			break
		}
	}
	if lan == "" {
		t.Skip("no non-loopback IPv4 address on this host to probe")
	}
	p, w := zeroFacadeProxy(t)
	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: freeRoutingPort(t)}))
	port := boundPort(p)

	if refuses(port) {
		t.Fatal("loopback does not accept connections")
	}
	conn, err := net.DialTimeout("tcp", net.JoinHostPort(lan, strconv.Itoa(port)), time.Second)
	if err == nil {
		_ = conn.Close()
		t.Fatalf("the listener accepted a connection on %s:%d", lan, port)
	}
}

func TestRoutingListener_RejectsInferenceWithoutSideEffects(t *testing.T) {
	const selfID = "self-uuid"
	tc := ollamaCase(t)
	models := append([]string{"mid:7b"}, allAliases...)
	newFixture := func(t *testing.T) (*Proxy, *recordingWriter, *recordingEngine) {
		t.Helper()
		redirectConfigDir(t)
		w := &recordingWriter{}
		local := newRecordingEngine(t, http.StatusOK)
		disc := NewDiscovery()
		disc.AddManual(selfNode(selfID, tc.profile.FacadePort, models...))
		p := newTestProxy(tc.profile, NewCodec(w), disc, tc.profile.FacadePort)
		setBackend(t, p.soleFacade(), local.URL, true)
		p.noteSelf(selfID, tc.profile.Name, models)
		p.SetPriority([]string{selfID})
		p.admission.setIntent(map[string]bool{tc.profile.Name: true})
		t.Cleanup(func() { p.shutdown(context.Background()) })
		return p, w, local
	}
	// settled is what the proxy holds on behalf of a request.
	settled := func(t *testing.T, p *Proxy) {
		t.Helper()
		p.priorityMu.Lock()
		reservations := len(p.priorityReservations)
		p.priorityMu.Unlock()
		a := p.admission
		a.mu.Lock()
		defer a.mu.Unlock()
		if reservations != 0 || a.active != 0 || len(a.lastActivity) != 0 || len(a.inflight) != 0 || len(a.wakeSentAt) != 0 {
			t.Errorf("proxy holds reservations=%d active=%d activity=%d inflight=%d wakes=%d, want nothing",
				reservations, a.active, len(a.lastActivity), len(a.inflight), len(a.wakeSentAt))
		}
	}

	t.Run("control: the same request through the facade reaches the engine", func(t *testing.T) {
		p, _, local := newFixture(t)
		rec := httptest.NewRecorder()
		p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"mid:7b"}`)))
		if rec.Code != http.StatusOK || local.hits() != 1 {
			t.Fatalf("control: status %d with %d engine hits, want 200 and 1", rec.Code, local.hits())
		}
	})

	t.Run("every inference route", func(t *testing.T) {
		p, w, local := newFixture(t)
		activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: freeRoutingPort(t)}))
		port := boundPort(p)
		for _, path := range []string{
			"/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/messages",
			"/api/generate", "/api/chat", "/api/embeddings", "/api/embed",
		} {
			got := call(t, port, http.MethodPost, path, nil)
			if got.status != http.StatusNotImplemented || got.code(t) != "feature_unavailable" {
				t.Errorf("POST %s = %d %s, want 501 feature_unavailable", path, got.status, got.body)
			}
		}
		if n := local.hits(); n != 0 {
			t.Errorf("engine received %d request(s), want none", n)
		}
		settled(t, p)
		for _, f := range frames(t, w) {
			if f.ID == nil && f.Method != "errors:clear" {
				t.Errorf("proxy emitted %q for an inference attempt, want only the listener's error clear", f.Method)
			}
		}
	})

	t.Run("every model name form", func(t *testing.T) {
		p, w, local := newFixture(t)
		activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: freeRoutingPort(t)}))
		port := boundPort(p)
		bodies := []string{
			`{"model":"mid:7b"}`,
			`{"model":"mid:7b","stream":true}`,
			`{"model":""}`,
			`{}`,
			`not json`,
			`{"name":"weak-api"}`,
		}
		for _, alias := range allAliases {
			for _, form := range []string{alias, strings.ToUpper(alias), " " + alias + " ", alias + ":latest"} {
				encoded, err := json.Marshal(map[string]string{"model": form})
				if err != nil {
					t.Fatal(err)
				}
				bodies = append(bodies, string(encoded))
			}
		}
		for _, body := range bodies {
			got := call(t, port, http.MethodPost, "/v1/chat/completions", func(r *http.Request) {
				r.Body = io.NopCloser(strings.NewReader(body))
				r.ContentLength = int64(len(body))
			})
			if got.status != http.StatusNotImplemented || got.code(t) != "feature_unavailable" {
				t.Errorf("body %q = %d %s, want 501 feature_unavailable", body, got.status, got.body)
			}
		}
		if n := local.hits(); n != 0 {
			t.Errorf("engine received %d request(s), want none", n)
		}
		settled(t, p)
	})
}

func TestRoutingListener_RejectsMethodsAndPaths(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: freeRoutingPort(t)}))
	port := boundPort(p)

	test := func(name, method, target string, wantStatus int, wantCode string) {
		t.Run(name, func(t *testing.T) {
			got := call(t, port, method, target, nil)
			if got.status != wantStatus {
				t.Fatalf("%s %s = %d %s, want %d", method, target, got.status, got.body, wantStatus)
			}
			if method != http.MethodHead && got.code(t) != wantCode {
				t.Errorf("code = %q, want %q", got.code(t), wantCode)
			}
		})
	}
	test("POST to the model list", http.MethodPost, "/v1/models", http.StatusMethodNotAllowed, "method_not_allowed")
	test("PUT to the model list", http.MethodPut, "/v1/models", http.StatusMethodNotAllowed, "method_not_allowed")
	test("DELETE to the model list", http.MethodDelete, "/v1/models", http.StatusMethodNotAllowed, "method_not_allowed")
	test("HEAD of the model list", http.MethodHead, "/v1/models", http.StatusMethodNotAllowed, "")
	test("GET of an inference route", http.MethodGet, "/v1/chat/completions", http.StatusMethodNotAllowed, "method_not_allowed")
	test("PATCH of an inference route", http.MethodPatch, "/v1/chat/completions", http.StatusMethodNotAllowed, "method_not_allowed")
	test("OPTIONS without an origin", http.MethodOptions, "/v1/models", http.StatusMethodNotAllowed, "method_not_allowed")
	test("model detail path", http.MethodGet, "/v1/models/weak", http.StatusNotFound, "not_found")
	test("trailing slash on the list", http.MethodGet, "/v1/models/", http.StatusNotFound, "not_found")
	test("trailing slash on an inference route", http.MethodPost, "/v1/chat/completions/", http.StatusNotFound, "not_found")
	test("engine native model list", http.MethodGet, "/api/tags", http.StatusNotFound, "not_found")
	test("llama.cpp model list", http.MethodGet, "/models", http.StatusNotFound, "not_found")
	test("engine passthrough", http.MethodPost, "/api/show", http.StatusNotFound, "not_found")
	test("root", http.MethodGet, "/", http.StatusNotFound, "not_found")

	t.Run("query string on an inference route", func(t *testing.T) {
		for _, target := range []string{"/v1/chat/completions?x=1", "/v1/chat/completions?", "/api/chat?model=weak-api"} {
			got := call(t, port, http.MethodPost, target, nil)
			if got.status != http.StatusBadRequest || got.code(t) != "query_not_allowed" {
				t.Errorf("POST %s = %d %s, want 400 query_not_allowed", target, got.status, got.body)
			}
		}
	})
	t.Run("query string on the model list is ignored", func(t *testing.T) {
		got := call(t, port, http.MethodGet, "/v1/models?x=1", nil)
		if got.status != http.StatusOK {
			t.Errorf("status = %d, want 200", got.status)
		}
	})
}

func TestRoutingListener_RebindsAndDisablesOnlyItsOwnListener(t *testing.T) {
	redirectConfigDir(t)
	w := &recordingWriter{}
	p := NewProxy(NewCodec(w))
	p.serveCtx = t.Context()
	facadePort := enableOnFreePort(t, p, "ollama")
	t.Cleanup(func() { p.shutdown(context.Background()) })

	foreign, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("foreign listener: %v", err)
	}
	defer foreign.Close()

	first, second := freeRoutingPort(t), freeRoutingPort(t)
	for second == first {
		second = freeRoutingPort(t)
	}

	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: first}))
	if boundPort(p) != first || call(t, first, http.MethodGet, "/v1/models", nil).status != http.StatusOK {
		t.Fatalf("listener is not serving on %d", first)
	}

	activate(t, p, w, listenerPolicy(t, "2", nodepolicy.Listener{Enabled: true, Port: second}))
	if boundPort(p) != second || call(t, second, http.MethodGet, "/v1/models", nil).status != http.StatusOK {
		t.Fatalf("listener did not move to %d", second)
	}
	if !refuses(first) {
		t.Errorf("old port %d still accepts connections after a rebind", first)
	}

	activate(t, p, w, listenerPolicy(t, "3", nodepolicy.Listener{Port: second}))
	if boundPort(p) != 0 || !refuses(second) {
		t.Errorf("disabled listener still bound: port %d, refuses=%v", boundPort(p), refuses(second))
	}

	if refuses(facadePort) {
		t.Errorf("the facade on %d stopped accepting connections", facadePort)
	}
	conn, err := net.DialTimeout("tcp", foreign.Addr().String(), time.Second)
	if err != nil {
		t.Errorf("a listener this code does not own stopped accepting: %v", err)
	} else {
		_ = conn.Close()
	}
}

func TestRoutingListener_DisablingClosesKeptAliveConnections(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	port := freeRoutingPort(t)
	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: port}))

	client := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{}}
	defer client.CloseIdleConnections()
	target := "http://127.0.0.1:" + strconv.Itoa(port) + "/v1/models"
	resp, err := client.Get(target)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()

	activate(t, p, w, listenerPolicy(t, "2", nodepolicy.Listener{}))
	if resp, err := client.Get(target); err == nil {
		_ = resp.Body.Close()
		t.Fatalf("the kept-alive connection still served after disabling: %d", resp.StatusCode)
	}
}

func TestRoutingListener_ReportsACollisionAndNeverPicksAnotherPort(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	held, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("occupy a port: %v", err)
	}
	defer held.Close()
	taken := held.Addr().(*net.TCPAddr).Port

	// The policy still activates: a port clash must not block unrelated
	// settings such as a spending ceiling.
	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: taken}))
	if got := p.admission.currentPolicy().Revision; got != "1" {
		t.Fatalf("active revision = %q, want 1 despite the collision", got)
	}
	if port := boundPort(p); port != 0 {
		t.Fatalf("listener bound %d after a collision on %d", port, taken)
	}

	reports := notificationsOf(t, w, "errors:report")
	if len(reports) != 1 {
		t.Fatalf("%d errors:report notifications, want 1", len(reports))
	}
	var report struct {
		ID       string `json:"id"`
		Message  string `json:"message"`
		Severity string `json:"severity"`
	}
	if err := json.Unmarshal(reports[0], &report); err != nil {
		t.Fatalf("decode report: %v", err)
	}
	if report.ID != "nvpair-proxy:routing-listener-blocked" || report.Severity != "error" ||
		!strings.Contains(report.Message, fmt.Sprintf("127.0.0.1:%d", taken)) {
		t.Errorf("report = %+v, want the listener id, severity error and the address %d", report, taken)
	}

	// Freeing the port and replaying the same revision, as the broker does
	// after a proxy restart, binds it and clears the error.
	_ = held.Close()
	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: taken}))
	if boundPort(p) != taken {
		t.Fatalf("replay did not bind the freed port %d", taken)
	}
	clears := notificationsOf(t, w, "errors:clear")
	if len(clears) == 0 || !strings.Contains(string(clears[len(clears)-1]), "nvpair-proxy:routing-listener-blocked") {
		t.Errorf("errors:clear notifications = %s, want the listener id last", clears)
	}
}

// facadeProxy is a process with an Ollama and an LM Studio facade enabled on
// ephemeral ports, keyed by engine.
func facadeProxy(t *testing.T) (*Proxy, *recordingWriter, map[string]int) {
	t.Helper()
	redirectConfigDir(t)
	w := &recordingWriter{}
	p := NewProxy(NewCodec(w))
	p.serveCtx = t.Context()
	ports := map[string]int{
		"ollama":   enableOnFreePort(t, p, "ollama"),
		"lmstudio": enableOnFreePort(t, p, "lmstudio"),
	}
	t.Cleanup(func() { p.shutdown(context.Background()) })
	return p, w, ports
}

// facadeRoutes is one inference route each facade answers itself, whatever the
// routing listener does.
var facadeRoutes = map[string]string{
	"ollama":   "/api/chat",
	"lmstudio": "/v1/chat/completions",
}

// askFacade posts a request for a model no node serves to one facade.
func askFacade(t *testing.T, engine string, port int) answer {
	t.Helper()
	target := "http://127.0.0.1:" + strconv.Itoa(port) + facadeRoutes[engine]
	resp, err := noKeepAlive.Post(target, "application/json", strings.NewReader(`{"model":"no-such-model"}`))
	if err != nil {
		t.Fatalf("ask the %s facade: %v", engine, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read the %s facade's answer: %v", engine, err)
	}
	return answer{status: resp.StatusCode, header: resp.Header, body: body}
}

func askFacades(t *testing.T, ports map[string]int) map[string]answer {
	t.Helper()
	out := make(map[string]answer, len(ports))
	for engine, port := range ports {
		out[engine] = askFacade(t, engine, port)
	}
	return out
}

func assertFacadesUnchanged(t *testing.T, ports map[string]int, before map[string]answer) {
	t.Helper()
	for engine, want := range before {
		got := askFacade(t, engine, ports[engine])
		if got.status != want.status || string(got.body) != string(want.body) {
			t.Errorf("%s facade answered %d %q after the listener event, want the unchanged %d %q",
				engine, got.status, got.body, want.status, want.body)
		}
		if strings.Contains(string(got.body), routingReasonNotInferenceReady) {
			t.Errorf("%s facade was answered by the routing listener: %s", engine, got.body)
		}
	}
}

func TestRoutingListener_MovingOntoATakenPortLeavesNoListener(t *testing.T) {
	p, w, ports := facadeProxy(t)
	working := freeRoutingPort(t)
	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: working}))
	if boundPort(p) != working {
		t.Fatalf("listener did not bind %d", working)
	}
	before := askFacades(t, ports)

	held, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("occupy a port: %v", err)
	}
	defer held.Close()
	taken := held.Addr().(*net.TCPAddr).Port
	activate(t, p, w, listenerPolicy(t, "2", nodepolicy.Listener{Enabled: true, Port: taken}))

	if got := p.admission.currentPolicy().Revision; got != "2" {
		t.Errorf("active revision = %q, want 2 despite the collision", got)
	}
	if boundPort(p) != 0 {
		t.Errorf("listener bound %d, want none after the configured port collided", boundPort(p))
	}
	if !refuses(working) {
		t.Errorf("the previous port %d still serves although the policy no longer names it", working)
	}
	if n := len(notificationsOf(t, w, "errors:report")); n != 1 {
		t.Errorf("%d errors:report notifications, want 1", n)
	}
	assertFacadesUnchanged(t, ports, before)
}

// holdAddress listens on address (an ephemeral port) and answers every request
// with marker, returning the port.
func holdAddress(t *testing.T, address, marker string) int {
	t.Helper()
	ln, err := net.Listen("tcp", address)
	if err != nil {
		t.Skipf("cannot listen on %s here: %v", address, err)
	}
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(marker))
	}))
	_ = srv.Listener.Close()
	srv.Listener = ln
	srv.Start()
	t.Cleanup(srv.Close)
	return ln.Addr().(*net.TCPAddr).Port
}

// A port that another socket holds on a wildcard address or on the IPv6
// loopback address is a collision even though a loopback bind beside it would
// succeed on some platforms. The holder keeps answering its own route and the
// listener never binds.
func TestRoutingListener_ReportsAnotherHolderAsACollision(t *testing.T) {
	for _, tc := range []struct {
		name, address, dial string
		reachableOnIPv4     bool
	}{
		{"IPv6 wildcard", "[::]:0", "127.0.0.1", true},
		{"IPv4 wildcard", "0.0.0.0:0", "127.0.0.1", true},
		{"IPv6 loopback", "[::1]:0", "[::1]", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, w := zeroFacadeProxy(t)
			const marker = "the foreign holder"
			taken := holdAddress(t, tc.address, marker)

			activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: taken}))

			if got := p.admission.currentPolicy().Revision; got != "1" {
				t.Errorf("active revision = %q, want 1 despite the collision", got)
			}
			if port := boundPort(p); port != 0 {
				t.Errorf("listener bound %d beside a holder on %s", port, tc.address)
			}
			if !tc.reachableOnIPv4 && !refuses(taken) {
				t.Errorf("something accepts connections on 127.0.0.1:%d, but the foreign holder is bound to ::1 only", taken)
			}
			if n := len(notificationsOf(t, w, "errors:report")); n != 1 {
				t.Errorf("%d errors:report notifications, want 1", n)
			}
			resp, err := noKeepAlive.Post("http://"+tc.dial+":"+strconv.Itoa(taken)+"/api/chat", "application/json", strings.NewReader(`{}`))
			if err != nil {
				t.Fatalf("ask the holder: %v", err)
			}
			defer resp.Body.Close()
			body, _ := io.ReadAll(resp.Body)
			if string(body) != marker {
				t.Errorf("the holder's route answered %d %q, want its own %q", resp.StatusCode, body, marker)
			}
		})
	}
}

// One of this process's own facades is a wildcard holder too: the listener
// refuses its port, and the facade keeps answering its own route.
func TestRoutingListener_ReportsAFacadePortAsACollision(t *testing.T) {
	p, w, ports := facadeProxy(t)
	before := askFacades(t, ports)

	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: ports["ollama"]}))

	if got := p.admission.currentPolicy().Revision; got != "1" {
		t.Errorf("active revision = %q, want 1 despite the collision", got)
	}
	if port := boundPort(p); port != 0 {
		t.Errorf("listener bound %d on a facade's port", port)
	}
	if n := len(notificationsOf(t, w, "errors:report")); n != 1 {
		t.Errorf("%d errors:report notifications, want 1", n)
	}
	assertFacadesUnchanged(t, ports, before)
}

// The reverse order: a facade asked onto the port the listener holds fails to
// bind, as the broker's retry on another port expects, and neither side moves.
func TestRoutingListener_RefusesAFacadeOnItsPort(t *testing.T) {
	t.Run("enable", func(t *testing.T) {
		redirectConfigDir(t)
		w := &recordingWriter{}
		p := NewProxy(NewCodec(w))
		p.serveCtx = t.Context()
		t.Cleanup(func() { p.shutdown(context.Background()) })
		port := freeRoutingPort(t)
		activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: port}))
		if boundPort(p) != port {
			t.Fatalf("listener did not bind %d", port)
		}

		_, err := p.enableFacade(enableFacadeParams{Engine: "ollama", Port: port, IgnorePersistedPort: true})
		if !stderrors.Is(err, errFacadeBindFailed) {
			t.Fatalf("enabling a facade on the listener's port = %v, want errFacadeBindFailed", err)
		}
		if got := call(t, port, http.MethodGet, "/v1/models", nil); got.status != http.StatusOK {
			t.Errorf("the listener answered %d after the refused bind, want 200", got.status)
		}
	})
	t.Run("set-port", func(t *testing.T) {
		p, w, ports := facadeProxy(t)
		port := freeRoutingPort(t)
		activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: port}))
		if boundPort(p) != port {
			t.Fatalf("listener did not bind %d", port)
		}
		before := askFacades(t, ports)

		if err := p.facadeFor("lmstudio").setPort(port); err == nil {
			t.Fatal("moving a facade onto the listener's port succeeded")
		}
		if got := call(t, port, http.MethodGet, "/v1/models", nil); got.status != http.StatusOK {
			t.Errorf("the listener answered %d after the refused move, want 200", got.status)
		}
		assertFacadesUnchanged(t, ports, before)
	})
}

// The listener is local state: nothing the proxy sends upward names its port or
// announces it for discovery. The only frames it adds are the error pipeline's
// report and clear, and a clear never carries a port.
func TestRoutingListener_IsNeverAnnouncedBeyondTheErrorPipeline(t *testing.T) {
	p, w, _ := facadeProxy(t)
	seen := len(w.lines())

	port := freeRoutingPort(t)
	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: port}))
	if got := call(t, port, http.MethodGet, "/v1/models", nil); got.status != http.StatusOK {
		t.Fatalf("listener answered %d, want 200", got.status)
	}
	activate(t, p, w, listenerPolicy(t, "2", nodepolicy.Listener{}))

	held, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("occupy a port: %v", err)
	}
	defer held.Close()
	activate(t, p, w, listenerPolicy(t, "3", nodepolicy.Listener{Enabled: true, Port: held.Addr().(*net.TCPAddr).Port}))

	for _, line := range w.lines()[seen:] {
		var f frame
		if err := json.Unmarshal(line, &f); err != nil {
			t.Fatalf("bad frame %q: %v", line, err)
		}
		if f.ID != nil {
			continue // the response to node/set-policy
		}
		switch f.Method {
		case "errors:clear":
			if strings.Contains(string(line), strconv.Itoa(port)) {
				t.Errorf("a clear names the listener port: %s", line)
			}
		case "errors:report":
		default:
			t.Errorf("the listener caused a %q notification: %s", f.Method, line)
		}
	}
}

func TestRoutingListener_DeniesBrowserOriginsUnlessExactlyAllowed(t *testing.T) {
	const allowed = "http://localhost:3000"
	p, w := zeroFacadeProxy(t)
	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{
		Enabled: true, Port: freeRoutingPort(t), AllowedOrigins: []string{allowed, "https://app.example.test"},
	}))
	port := boundPort(p)

	t.Run("no origin is a plain local client", func(t *testing.T) {
		got := call(t, port, http.MethodGet, "/v1/models", nil)
		if got.status != http.StatusOK || got.header.Get("Access-Control-Allow-Origin") != "" {
			t.Errorf("status %d, allow-origin %q; want 200 and no CORS grant", got.status, got.header.Get("Access-Control-Allow-Origin"))
		}
	})
	t.Run("exact origin is allowed", func(t *testing.T) {
		for _, path := range []string{"/v1/models"} {
			got := call(t, port, http.MethodGet, path, withHeader("Origin", allowed))
			if got.status != http.StatusOK {
				t.Fatalf("status = %d, want 200", got.status)
			}
			if v := got.header.Get("Access-Control-Allow-Origin"); v != allowed {
				t.Errorf("Access-Control-Allow-Origin = %q, want %q", v, allowed)
			}
			if !slices.Contains(got.header.Values("Vary"), "Origin") {
				t.Errorf("Vary = %v, want Origin", got.header.Values("Vary"))
			}
		}
	})
	t.Run("allowed origin can read the inference refusal", func(t *testing.T) {
		got := call(t, port, http.MethodPost, "/v1/chat/completions", withHeader("Origin", allowed))
		if got.status != http.StatusNotImplemented || got.header.Get("Access-Control-Allow-Origin") != allowed {
			t.Errorf("status %d, allow-origin %q; want 501 granting %q", got.status, got.header.Get("Access-Control-Allow-Origin"), allowed)
		}
	})

	denied := func(name string, mutate func(*http.Request)) {
		t.Run(name, func(t *testing.T) {
			for _, request := range []struct{ method, path string }{
				{http.MethodGet, "/v1/models"},
				{http.MethodPost, "/v1/chat/completions"},
			} {
				got := call(t, port, request.method, request.path, mutate)
				if got.status != http.StatusForbidden || got.code(t) != "origin_not_allowed" {
					t.Errorf("%s %s = %d %s, want 403 origin_not_allowed", request.method, request.path, got.status, got.body)
				}
				if v := got.header.Get("Access-Control-Allow-Origin"); v != "" {
					t.Errorf("a denied origin was granted %q", v)
				}
				if strings.Contains(string(got.body), `"weak"`) {
					t.Errorf("a denied origin received the alias list: %s", got.body)
				}
			}
		})
	}
	denied("unlisted origin", withHeader("Origin", "http://evil.test"))
	denied("other port", withHeader("Origin", "http://localhost:3001"))
	denied("other scheme", withHeader("Origin", "https://localhost:3000"))
	denied("different case", withHeader("Origin", "http://LOCALHOST:3000"))
	denied("trailing slash", withHeader("Origin", allowed+"/"))
	denied("prefix of an allowed origin", withHeader("Origin", "http://localhost:300"))
	denied("null origin", withHeader("Origin", "null"))
	denied("empty origin header", func(r *http.Request) { r.Header["Origin"] = []string{""} })
	denied("two origin headers", func(r *http.Request) { r.Header["Origin"] = []string{allowed, allowed} })

	t.Run("an empty allow list denies every origin", func(t *testing.T) {
		activate(t, p, w, listenerPolicy(t, "2", nodepolicy.Listener{Enabled: true, Port: port}))
		got := call(t, port, http.MethodGet, "/v1/models", withHeader("Origin", allowed))
		if got.status != http.StatusForbidden {
			t.Errorf("status = %d, want 403", got.status)
		}
	})
}

func TestRoutingListener_AnswersPreflightOnlyForAllowedOrigins(t *testing.T) {
	const allowed = "http://localhost:3000"
	p, w := zeroFacadeProxy(t)
	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{
		Enabled: true, Port: freeRoutingPort(t), AllowedOrigins: []string{allowed},
	}))
	port := boundPort(p)

	preflight := func(origin, method string, headers string) func(*http.Request) {
		return func(r *http.Request) {
			if origin != "" {
				r.Header.Set("Origin", origin)
			}
			r.Header.Set("Access-Control-Request-Method", method)
			if headers != "" {
				r.Header.Set("Access-Control-Request-Headers", headers)
			}
		}
	}

	t.Run("allowed origin may post to an inference route", func(t *testing.T) {
		got := call(t, port, http.MethodOptions, "/v1/chat/completions", preflight(allowed, http.MethodPost, "Content-Type, authorization"))
		if got.status != http.StatusNoContent {
			t.Fatalf("status = %d %s, want 204", got.status, got.body)
		}
		for name, want := range map[string]string{
			"Access-Control-Allow-Origin":  allowed,
			"Access-Control-Allow-Methods": "POST, OPTIONS",
			"Access-Control-Allow-Headers": "Content-Type, Authorization",
			"Access-Control-Max-Age":       routingPreflightMaxAge,
		} {
			if v := got.header.Get(name); v != want {
				t.Errorf("%s = %q, want %q", name, v, want)
			}
		}
	})
	t.Run("allowed origin may get the model list", func(t *testing.T) {
		got := call(t, port, http.MethodOptions, "/v1/models", preflight(allowed, http.MethodGet, ""))
		if got.status != http.StatusNoContent || got.header.Get("Access-Control-Allow-Methods") != "GET, OPTIONS" {
			t.Errorf("status %d allow-methods %q, want 204 and GET, OPTIONS", got.status, got.header.Get("Access-Control-Allow-Methods"))
		}
	})

	deny := func(name, path string, wantStatus int, wantCode string, mutate func(*http.Request)) {
		t.Run(name, func(t *testing.T) {
			got := call(t, port, http.MethodOptions, path, mutate)
			if got.status != wantStatus || got.code(t) != wantCode {
				t.Errorf("OPTIONS %s = %d %s, want %d %s", path, got.status, got.body, wantStatus, wantCode)
			}
			if v := got.header.Get("Access-Control-Allow-Methods"); v != "" {
				t.Errorf("a refused preflight advertised methods %q", v)
			}
		})
	}
	deny("unlisted origin", "/v1/chat/completions", http.StatusForbidden, "origin_not_allowed",
		preflight("http://evil.test", http.MethodPost, ""))
	deny("method the route does not take", "/v1/chat/completions", http.StatusForbidden, "preflight_not_allowed",
		preflight(allowed, http.MethodDelete, ""))
	deny("post asked of the model list", "/v1/models", http.StatusForbidden, "preflight_not_allowed",
		preflight(allowed, http.MethodPost, ""))
	deny("header outside the allowed set", "/v1/chat/completions", http.StatusForbidden, "preflight_not_allowed",
		preflight(allowed, http.MethodPost, "content-type, x-api-key"))
	deny("unknown path", "/v1/nope", http.StatusNotFound, "not_found",
		preflight(allowed, http.MethodPost, ""))
	deny("origin without a requested method", "/v1/models", http.StatusMethodNotAllowed, "method_not_allowed",
		withHeader("Origin", allowed))
}

func TestRoutingListener_OriginAllowListFollowsActivationWithoutARebind(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	port := freeRoutingPort(t)
	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{
		Enabled: true, Port: port, AllowedOrigins: []string{"http://localhost:3000"},
	}))
	activate(t, p, w, listenerPolicy(t, "2", nodepolicy.Listener{
		Enabled: true, Port: port, AllowedOrigins: []string{"http://localhost:4000"},
	}))

	if got := call(t, port, http.MethodGet, "/v1/models", withHeader("Origin", "http://localhost:3000")); got.status != http.StatusForbidden {
		t.Errorf("the removed origin got %d, want 403", got.status)
	}
	if got := call(t, port, http.MethodGet, "/v1/models", withHeader("Origin", "http://localhost:4000")); got.status != http.StatusOK {
		t.Errorf("the added origin got %d, want 200", got.status)
	}
}

func TestRoutingListener_RejectsHostsThatAreNotItsLoopbackName(t *testing.T) {
	const allowed = "http://localhost:3000"
	p, w := zeroFacadeProxy(t)
	activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{
		Enabled: true, Port: freeRoutingPort(t), AllowedOrigins: []string{allowed},
	}))
	port := boundPort(p)
	other := strconv.Itoa(port + 1)
	own := strconv.Itoa(port)

	for _, host := range []string{
		"evil.test:" + own,
		"evil.test",
		"127.0.0.1",
		"127.0.0.1:" + other,
		"localhost:" + other,
		"127.0.0.2:" + own,
		"0.0.0.0:" + own,
		"[::1]:" + own,
		"localhost.evil.test:" + own,
		"127.0.0.1.evil.test:" + own,
	} {
		t.Run("rejects "+host, func(t *testing.T) {
			for _, request := range []struct{ method, path string }{
				{http.MethodGet, "/v1/models"},
				{http.MethodPost, "/v1/chat/completions"},
				{http.MethodGet, "/nowhere"},
			} {
				got := call(t, port, request.method, request.path, func(r *http.Request) {
					r.Host = host
					r.Header.Set("Origin", allowed)
				})
				if got.status != http.StatusForbidden || got.code(t) != "host_not_allowed" {
					t.Errorf("%s %s with Host %q = %d %s, want 403 host_not_allowed", request.method, request.path, host, got.status, got.body)
				}
			}
		})
	}
	for _, host := range []string{"127.0.0.1:" + own, "localhost:" + own, "LocalHost:" + own} {
		t.Run("accepts "+host, func(t *testing.T) {
			if got := call(t, port, http.MethodGet, "/v1/models", withHost(host)); got.status != http.StatusOK {
				t.Errorf("Host %q = %d, want 200", host, got.status)
			}
		})
	}
}

func TestRoutingListener_ReturnsAfterARestartWhenThePolicyIsReplayed(t *testing.T) {
	port := freeRoutingPort(t)
	pol := listenerPolicy(t, "7", nodepolicy.Listener{Enabled: true, Port: port})

	first, firstOut := zeroFacadeProxy(t)
	activate(t, first, firstOut, pol)
	if boundPort(first) != port {
		t.Fatalf("first process did not bind %d", port)
	}
	first.shutdown(context.Background())
	if !refuses(port) {
		t.Fatalf("port %d still accepts connections after the process shut down", port)
	}

	restarted, out := zeroFacadeProxy(t)
	if boundPort(restarted) != 0 || !refuses(port) {
		t.Fatal("a fresh process listened before the broker replayed any policy")
	}
	activate(t, restarted, out, pol)
	if got := call(t, port, http.MethodGet, "/v1/models", nil); got.status != http.StatusOK {
		t.Errorf("after replay status = %d, want 200", got.status)
	}
	// A fresh process also clears an error its predecessor left in the
	// pipeline, so a listener that now binds does not stay flagged.
	if n := len(notificationsOf(t, out, "errors:clear")); n != 1 {
		t.Errorf("%d errors:clear notifications after the replay, want 1", n)
	}
}

func TestRoutingListener_StaleRevisionDoesNotChangeTheListener(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	port := freeRoutingPort(t)
	activate(t, p, w, listenerPolicy(t, "5", nodepolicy.Listener{Enabled: true, Port: port}))

	other := freeRoutingPort(t)
	rpcErr(t, p, w, nodepolicy.MethodSetPolicy, listenerPolicy(t, "4", nodepolicy.Listener{Enabled: true, Port: other}))
	if boundPort(p) != port || !refuses(other) {
		t.Errorf("a refused older revision changed the listener: bound %d, %d refuses=%v", boundPort(p), other, refuses(other))
	}
	rpcErr(t, p, w, nodepolicy.MethodSetPolicy, listenerPolicy(t, "4", nodepolicy.Listener{}))
	if boundPort(p) != port {
		t.Errorf("a refused older revision disabled the listener")
	}
}

func TestRoutingListener_ShutdownClosesItForGood(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	port := freeRoutingPort(t)
	pol := listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: port})
	activate(t, p, w, pol)

	p.shutdown(context.Background())
	if !refuses(port) {
		t.Fatalf("port %d still accepts connections after shutdown", port)
	}
	p.routing.reconcile()
	if boundPort(p) != 0 {
		t.Error("a reconcile after shutdown rebound the listener")
	}
}

func TestRoutingListener_HostMatchesCaseInsensitivelyButNotByPrefix(t *testing.T) {
	h := &routingHandler{port: 8123}
	for host, want := range map[string]bool{
		"127.0.0.1:8123":  true,
		"localhost:8123":  true,
		"LOCALHOST:8123":  true,
		"localhost:08123": false,
		"localhost:812":   false,
		"localhost:":      false,
		"":                false,
		"[::1]:8123":      false,
	} {
		if got := h.hostAllowed(host); got != want {
			t.Errorf("hostAllowed(%q) = %v, want %v", host, got, want)
		}
	}
}

// A client that stalls mid-body must still get its refusal, and neither it nor
// a client that stalls mid-header may hold the control loop when the listener is
// disabled or moved.
func TestRoutingListener_StalledClientsNeitherDelayAnAnswerNorHoldTheControlLoop(t *testing.T) {
	dial := func(t *testing.T, port int) net.Conn {
		t.Helper()
		conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), time.Second)
		if err != nil {
			t.Fatalf("dial: %v", err)
		}
		t.Cleanup(func() { _ = conn.Close() })
		return conn
	}
	send := func(t *testing.T, conn net.Conn, text string) {
		t.Helper()
		if _, err := io.WriteString(conn, text); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	head := func(port int) string {
		return "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:" + strconv.Itoa(port) +
			"\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n"
	}

	for name, newListener := range map[string]nodepolicy.Listener{
		"disabled": {},
		"moved":    {Enabled: true, Port: freeRoutingPort(t)},
	} {
		t.Run(name, func(t *testing.T) {
			p, w := zeroFacadeProxy(t)
			port := freeRoutingPort(t)
			activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: port}))

			// Headers and part of the declared body, then silence.
			stalledBody := dial(t, port)
			send(t, stalledBody, head(port)+`{"model":`)
			if err := stalledBody.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
				t.Fatal(err)
			}
			resp, err := http.ReadResponse(bufio.NewReader(stalledBody), nil)
			if err != nil {
				t.Fatalf("no refusal within 1s of a stalled body: %v", err)
			}
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusNotImplemented {
				t.Errorf("status = %d, want 501", resp.StatusCode)
			}

			// Headers never finished.
			stalledHeader := dial(t, port)
			send(t, stalledHeader, "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:")

			started := time.Now()
			next := listenerPolicy(t, "2", newListener)
			activate(t, p, w, next)
			if elapsed := time.Since(started); elapsed > time.Second {
				t.Errorf("activating the new listener setting took %v with stalled clients connected, want under 1s", elapsed)
			}
			if !refuses(port) {
				t.Errorf("old port %d still accepts connections", port)
			}
		})
	}
}
