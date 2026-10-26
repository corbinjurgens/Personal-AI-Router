// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Coverage for the engine-independent routing listener. Every listener here
// binds an ephemeral loopback port; none uses an engine's or the live
// application's port.

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
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
	p := newBoundProxy(NewCodec(w))
	t.Cleanup(func() { p.shutdown(context.Background()) })
	return p, w
}

func activate(t *testing.T, p *Proxy, w *recordingWriter, pol nodepolicy.Policy) {
	t.Helper()
	rpcOK(t, p, w, nodepolicy.MethodSetPolicy, pol)
}

// freeRoutingPort is only a candidate, never a promise that the subsequent
// production bind will succeed. Positive setup uses startRoutingPolicy below.
func freeRoutingPort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve free port: %v", err)
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
}

// startRoutingPolicy establishes a working listener for tests whose subject is
// not a bind collision. A competing process can take a candidate between close
// and activation (or hold the same port on another address), so retry setup on
// a fresh port. Collision and same-port replay assertions use activate directly.
func startRoutingPolicy(t *testing.T, p *Proxy, w *recordingWriter, pol nodepolicy.Policy) int {
	t.Helper()
	pol.Routing.Listener.Enabled = true
	previous := boundPort(p)
	for range 20 {
		pol.Routing.Listener.Port = freeRoutingPort(t)
		activate(t, p, w, pol)
		if got := boundPort(p); got == pol.Routing.Listener.Port {
			return got
		}
		if got := boundPort(p); got != 0 && got != previous {
			t.Fatalf("listener bound %d, want configured port %d", got, pol.Routing.Listener.Port)
		}
	}
	t.Fatal("could not establish routing listener after fresh-port setup retries")
	return 0
}

func startRouting(t *testing.T, p *Proxy, w *recordingWriter, revision string, origins ...string) int {
	t.Helper()
	return startRoutingPolicy(t, p, w, listenerPolicy(t, revision, nodepolicy.Listener{AllowedOrigins: origins}))
}

// assertRoutingClosed checks the captured socket, which cannot be replaced by
// an unrelated listener that acquires the released address. It also covers a
// raw socket leaked before http.Server.Serve registers it.
func assertRoutingClosed(t *testing.T, ln net.Listener) {
	t.Helper()
	tcp, ok := ln.(*net.TCPListener)
	if !ok {
		t.Fatalf("routing listener = %T, want *net.TCPListener", ln)
	}
	if err := tcp.SetDeadline(time.Now()); !stderrors.Is(err, net.ErrClosed) {
		t.Errorf("routing socket deadline error = %v, want net.ErrClosed", err)
	}
}

func routingSocket(p *Proxy) net.Listener {
	p.routing.mu.Lock()
	defer p.routing.mu.Unlock()
	return p.routing.ln
}

// replayRouting retries the same policy only if an independent bind observes a
// competing loopback socket after the handoff. A product failure on a free port
// is fatal; this must not turn a broken replay into successful fresh-port setup.
func replayRouting(t *testing.T, p *Proxy, w *recordingWriter, pol nodepolicy.Policy) {
	t.Helper()
	for range 20 {
		activate(t, p, w, pol)
		if boundPort(p) == pol.Routing.Listener.Port {
			return
		}
		occupied := false
		for _, host := range []string{"127.0.0.1", "::1"} {
			ln, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(pol.Routing.Listener.Port)))
			if stderrors.Is(err, syscall.EADDRINUSE) {
				occupied = true
			}
			if err == nil {
				if err := ln.Close(); err != nil {
					t.Fatal(err)
				}
			}
		}
		if !occupied {
			t.Fatalf("replay did not bind released port %d", pol.Routing.Listener.Port)
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("released routing port stayed occupied throughout replay retries")
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

func TestRoutingListener_StopClosesListenerBeforeServeStarts(t *testing.T) {
	test := func(name string, stop func(*testing.T, *Proxy, *recordingWriter)) {
		t.Run(name, func(t *testing.T) {
			p, w := zeroFacadeProxy(t)
			entered := make(chan struct{}, 2)
			release := make(chan struct{})
			t.Cleanup(func() { close(release) })
			p.routing.beforeServe = func() {
				entered <- struct{}{}
				<-release
			}
			port := startRouting(t, p, w, "1")
			socket := routingSocket(p)
			if got := boundPort(p); got != port {
				t.Fatalf("bound port = %d, want %d", got, port)
			}
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("Serve goroutine did not reach the gate")
			}

			stop(t, p, w)

			assertRoutingClosed(t, socket)
		})
	}
	test("final shutdown", func(t *testing.T, p *Proxy, _ *recordingWriter) {
		p.shutdown(context.Background())
		if port := boundPort(p); port != 0 {
			t.Errorf("bound port after shutdown = %d, want 0", port)
		}
	})
	test("policy disables listener", func(t *testing.T, p *Proxy, w *recordingWriter) {
		activate(t, p, w, listenerPolicy(t, "2", nodepolicy.Listener{}))
		if port := boundPort(p); port != 0 {
			t.Errorf("bound port after disable = %d, want 0", port)
		}
	})
	test("policy moves port", func(t *testing.T, p *Proxy, w *recordingWriter) {
		port := startRouting(t, p, w, "2")
		if got := boundPort(p); got != port {
			t.Errorf("bound port after move = %d, want %d", got, port)
		}
	})
}

func TestRoutingListener_ListsAliasesWithNoEngineRunning(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	startRouting(t, p, w, "1")
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
	startRoutingPolicy(t, p, w, pol)

	got := call(t, boundPort(p), http.MethodGet, "/v1/models", nil)
	if ids := listedIDs(t, got); len(ids) != 0 {
		t.Errorf("listed %v, want nothing", ids)
	}
}

func TestRoutingListener_ListFollowsTheActivatedPolicy(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	port := startRouting(t, p, w, "1")
	listener := nodepolicy.Listener{Enabled: true, Port: port}

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
	p, w := zeroFacadeProxy(t)
	port := startRouting(t, p, w, "1")
	addr := routingSocket(p).Addr().(*net.TCPAddr)
	if !addr.IP.Equal(net.IPv4(127, 0, 0, 1)) {
		t.Errorf("routing socket bound %s, want 127.0.0.1", addr)
	}
	if got := call(t, port, http.MethodGet, "/v1/models", nil); got.status != http.StatusOK {
		t.Errorf("loopback response = %d, want 200", got.status)
	}
}

func TestRoutingListener_InferenceUnqualifiedOrUnsupportedHasNoDispatchEffects(t *testing.T) {
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
		startRouting(t, p, w, "1")
		port := boundPort(p)
		for _, path := range []string{
			"/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/messages",
			"/api/generate", "/api/chat", "/api/embeddings", "/api/embed",
		} {
			got := call(t, port, http.MethodPost, path, nil)
			if got.status != http.StatusNotFound {
				t.Errorf("POST %s = %d %s, want 404 without qualified candidates", path, got.status, got.body)
			}
		}
		if n := local.hits(); n != 0 {
			t.Errorf("engine received %d request(s), want none", n)
		}
		settled(t, p)
		if n := len(notificationsOf(t, w, workloadSubmittedMethod)); n != 0 {
			t.Errorf("unqualified requests submitted %d workloads", n)
		}
		ownedAssertHandlerDrained(t, p, nil)
	})

	t.Run("every model name form", func(t *testing.T) {
		p, w, local := newFixture(t)
		startRouting(t, p, w, "1")
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
			var request struct {
				Model string `json:"model"`
			}
			parseErr := json.Unmarshal([]byte(body), &request)
			want := http.StatusBadRequest
			if parseErr == nil && nodepolicy.IsReservedAlias(request.Model) {
				want = http.StatusNotFound
			}
			if got.status != want {
				t.Errorf("body %q = %d %s, want %d", body, got.status, got.body, want)
			}
		}
		if n := local.hits(); n != 0 {
			t.Errorf("engine received %d request(s), want none", n)
		}
		settled(t, p)
		ownedAssertHandlerDrained(t, p, nil)
	})
}

func TestRoutingListener_RejectsMethodsAndPaths(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	startRouting(t, p, w, "1")
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
	p := newBoundProxy(NewCodec(w))
	p.serveCtx = t.Context()
	facadePort := enableOnFreePort(t, p, "ollama")
	t.Cleanup(func() { p.shutdown(context.Background()) })

	foreign, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("foreign listener: %v", err)
	}
	defer foreign.Close()

	first := startRouting(t, p, w, "1")
	firstSocket := routingSocket(p)
	if boundPort(p) != first || call(t, first, http.MethodGet, "/v1/models", nil).status != http.StatusOK {
		t.Fatalf("listener is not serving on %d", first)
	}

	second := startRouting(t, p, w, "2")
	secondSocket := routingSocket(p)
	if boundPort(p) != second || call(t, second, http.MethodGet, "/v1/models", nil).status != http.StatusOK {
		t.Fatalf("listener did not move to %d", second)
	}
	assertRoutingClosed(t, firstSocket)

	activate(t, p, w, listenerPolicy(t, "3", nodepolicy.Listener{Port: second}))
	if boundPort(p) != 0 {
		t.Errorf("disabled listener still bound: port %d", boundPort(p))
	}
	assertRoutingClosed(t, secondSocket)

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
	port := startRouting(t, p, w, "1")

	conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	reader := bufio.NewReader(conn)
	if got, err := routingExchange(conn, reader, "127.0.0.1", port, http.MethodGet, "/v1/models"); err != nil || got.status != http.StatusOK {
		t.Fatalf("initial keep-alive response = %+v, %v", got, err)
	}

	activate(t, p, w, listenerPolicy(t, "2", nodepolicy.Listener{}))
	if got, err := routingExchange(conn, reader, "127.0.0.1", port, http.MethodGet, "/v1/models"); err == nil {
		t.Fatalf("the kept-alive connection still served after disabling: %d", got.status)
	} else if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
		t.Fatalf("kept-alive connection remained open after disabling: %v", err)
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
	replayRouting(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: taken}))
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
	p := newBoundProxy(NewCodec(w))
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
	}
}

func TestRoutingListener_MovingOntoATakenPortKeepsTheWorkingListener(t *testing.T) {
	p, w, ports := facadeProxy(t)
	working := startRouting(t, p, w, "1")
	workingSocket := routingSocket(p)
	reportsBefore := len(notificationsOf(t, w, "errors:report"))
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
	if boundPort(p) != working || routingSocket(p) != workingSocket {
		t.Fatalf("failed rebind replaced working listener: port %d, want %d", boundPort(p), working)
	}
	if got := call(t, working, http.MethodGet, routingModelsPath, nil); got.status != http.StatusOK {
		t.Fatalf("working listener after refused rebind: status %d", got.status)
	}
	if n := len(notificationsOf(t, w, "errors:report")) - reportsBefore; n != 1 {
		t.Errorf("%d errors:report notifications, want 1", n)
	}
	assertFacadesUnchanged(t, ports, before)
}

func routingHTTPServer(p *Proxy) *http.Server {
	p.routing.mu.Lock()
	defer p.routing.mu.Unlock()
	return p.routing.srv
}

func TestRoutingListener_HasNoStreamWriteDeadline(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	startRouting(t, p, w, "1")
	srv := routingHTTPServer(p)
	if srv.WriteTimeout != 0 {
		t.Fatalf("stream write timeout = %v, want no server-wide deadline", srv.WriteTimeout)
	}
	if srv.ReadHeaderTimeout != proxyReadHeaderTimeout || srv.IdleTimeout != proxyServerIdleTimeout {
		t.Fatalf("header/idle bounds changed: %v/%v", srv.ReadHeaderTimeout, srv.IdleTimeout)
	}
}

func TestRoutingListener_RebindUsesOneServerAndConnectionPorts(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	first := startRouting(t, p, w, "1")
	srv := routingHTTPServer(p)
	conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(first)), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	reader := bufio.NewReader(conn)
	if got, err := routingExchange(conn, reader, "localhost", first, http.MethodGet, routingModelsPath); err != nil || got.status != http.StatusOK {
		t.Fatalf("initial connection: status %d, error %v", got.status, err)
	}
	for revision := 2; revision <= 5; revision++ {
		old := routingSocket(p)
		port := startRouting(t, p, w, strconv.Itoa(revision))
		if routingHTTPServer(p) != srv {
			t.Fatal("rebind replaced the HTTP server")
		}
		assertRoutingClosed(t, old)
		if got := call(t, port, http.MethodGet, routingModelsPath, nil); got.status != http.StatusOK {
			t.Fatalf("new listener: status %d", got.status)
		}
		if got, err := routingExchange(conn, reader, "localhost", first, http.MethodGet, routingModelsPath); err != nil || got.status != http.StatusOK {
			t.Fatalf("old connection lost accepted port: status %d, error %v", got.status, err)
		}
		if got, err := routingExchange(conn, reader, "localhost", port, http.MethodGet, routingModelsPath); err != nil || got.status != http.StatusForbidden || got.code(t) != routingCodeHostNotAllowed {
			t.Fatalf("old connection accepted new Host port: status %d, error %v", got.status, err)
		}
		if got := call(t, port, http.MethodGet, routingModelsPath, withHost(net.JoinHostPort("localhost", strconv.Itoa(first)))); got.status != http.StatusForbidden || got.code(t) != routingCodeHostNotAllowed {
			t.Fatalf("new connection accepted old Host port: status %d", got.status)
		}
	}
}

// routingExchange uses an existing connection and never redials a released
// address. Both the foreign-holder and keep-alive tests need that distinction.
func routingExchange(conn net.Conn, reader *bufio.Reader, host string, port int, method, path string) (answer, error) {
	if err := conn.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
		return answer{}, err
	}
	if _, err := fmt.Fprintf(conn, "%s %s HTTP/1.1\r\nHost: %s\r\nContent-Length: 0\r\n\r\n", method, path, net.JoinHostPort(host, strconv.Itoa(port))); err != nil {
		return answer{}, err
	}
	resp, err := http.ReadResponse(reader, nil)
	if err != nil {
		return answer{}, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return answer{}, err
	}
	return answer{status: resp.StatusCode, header: resp.Header, body: body}, nil
}

// holdAddress retains both the foreign listener and a verified connection to it.
func holdAddress(t *testing.T, address, dial, marker string) (int, func() answer) {
	t.Helper()
	for range 20 {
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
		port := ln.Addr().(*net.TCPAddr).Port
		conn, err := net.DialTimeout("tcp", net.JoinHostPort(dial, strconv.Itoa(port)), time.Second)
		if err != nil {
			srv.Close()
			continue
		}
		reader := bufio.NewReader(conn)
		ask := func() answer {
			t.Helper()
			if _, err := ln.(*net.TCPListener).SyscallConn(); err != nil {
				t.Fatalf("foreign holder's listener was closed: %v", err)
			}
			got, err := routingExchange(conn, reader, dial, port, http.MethodPost, "/api/chat")
			if err != nil {
				t.Fatalf("ask the holder: %v", err)
			}
			return got
		}
		// Wildcard listeners may be shadowed by an unrelated specific-address
		// listener. Establish identity before the action, then retain this exact
		// connection so later address reuse cannot redirect the assertion.
		if got, err := routingExchange(conn, reader, dial, port, http.MethodPost, "/api/chat"); err != nil || got.status != http.StatusOK || string(got.body) != marker {
			_ = conn.Close()
			srv.Close()
			continue
		}
		t.Cleanup(srv.Close)
		t.Cleanup(func() { _ = conn.Close() })
		return port, ask
	}
	t.Fatal("could not connect to the foreign holder after fresh-port setup retries")
	return 0, nil
}

// A port that another socket holds on a wildcard address or on the IPv6
// loopback address is a collision even though a loopback bind beside it would
// succeed on some platforms. The holder keeps answering its own route and the
// listener never binds.
func TestRoutingListener_ReportsAnotherHolderAsACollision(t *testing.T) {
	for _, tc := range []struct {
		name, address, dial string
	}{
		{"IPv6 wildcard", "[::]:0", "127.0.0.1"},
		{"IPv4 wildcard", "0.0.0.0:0", "127.0.0.1"},
		{"IPv6 loopback", "[::1]:0", "::1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, w := zeroFacadeProxy(t)
			marker := rand.Text()
			taken, ask := holdAddress(t, tc.address, tc.dial, marker)

			activate(t, p, w, listenerPolicy(t, "1", nodepolicy.Listener{Enabled: true, Port: taken}))

			if got := p.admission.currentPolicy().Revision; got != "1" {
				t.Errorf("active revision = %q, want 1 despite the collision", got)
			}
			if port := boundPort(p); port != 0 {
				t.Errorf("listener bound %d beside a holder on %s", port, tc.address)
			}
			if ln := routingSocket(p); ln != nil {
				t.Errorf("proxy owns a routing socket %v beside the foreign holder", ln.Addr())
			}
			if n := len(notificationsOf(t, w, "errors:report")); n != 1 {
				t.Errorf("%d errors:report notifications, want 1", n)
			}
			got := ask()
			if got.status != http.StatusOK || string(got.body) != marker {
				t.Errorf("the holder's route answered %d %q, want its own %q", got.status, got.body, marker)
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
		p := newBoundProxy(NewCodec(w))
		p.serveCtx = t.Context()
		t.Cleanup(func() { p.shutdown(context.Background()) })
		port := startRouting(t, p, w, "1")
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
		port := startRouting(t, p, w, "1")
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

	port := startRouting(t, p, w, "1")
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
		case nodepolicy.NotifyStrictReadiness:
			// The report that follows each node/set-policy acknowledgement.
		default:
			t.Errorf("the listener caused a %q notification: %s", f.Method, line)
		}
	}
}

func TestRoutingListener_DeniesBrowserOriginsUnlessExactlyAllowed(t *testing.T) {
	const allowed = "http://localhost:3000"
	p, w := zeroFacadeProxy(t)
	startRouting(t, p, w, "1", allowed, "https://app.example.test")
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
		if got.status != http.StatusNotFound || got.header.Get("Access-Control-Allow-Origin") != allowed {
			t.Errorf("status %d, allow-origin %q; want 404 granting %q", got.status, got.header.Get("Access-Control-Allow-Origin"), allowed)
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
	startRouting(t, p, w, "1", allowed)
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
	port := startRouting(t, p, w, "1", "http://localhost:3000")
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
	startRouting(t, p, w, "1", allowed)
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
	first, firstOut := zeroFacadeProxy(t)
	port := startRouting(t, first, firstOut, "7")
	pol := first.admission.currentPolicy()
	socket := routingSocket(first)
	if boundPort(first) != port {
		t.Fatalf("first process did not bind %d", port)
	}
	first.shutdown(context.Background())
	assertRoutingClosed(t, socket)

	restarted, out := zeroFacadeProxy(t)
	if boundPort(restarted) != 0 || routingSocket(restarted) != nil {
		t.Fatal("a fresh process listened before the broker replayed any policy")
	}
	replayRouting(t, restarted, out, pol)
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
	port := startRouting(t, p, w, "5")
	socket := routingSocket(p)

	other := freeRoutingPort(t)
	rpcErr(t, p, w, nodepolicy.MethodSetPolicy, listenerPolicy(t, "4", nodepolicy.Listener{Enabled: true, Port: other}))
	if boundPort(p) != port || routingSocket(p) != socket {
		t.Errorf("a refused older revision changed the listener: bound %d, want %d", boundPort(p), port)
	}
	rpcErr(t, p, w, nodepolicy.MethodSetPolicy, listenerPolicy(t, "4", nodepolicy.Listener{}))
	if boundPort(p) != port {
		t.Errorf("a refused older revision disabled the listener")
	}
}

func TestRoutingListener_ShutdownClosesItForGood(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	startRouting(t, p, w, "1")
	socket := routingSocket(p)

	p.shutdown(context.Background())
	assertRoutingClosed(t, socket)
	p.routing.reconcile()
	if boundPort(p) != 0 {
		t.Error("a reconcile after shutdown rebound the listener")
	}
}

func TestRoutingListener_HostMatchesCaseInsensitivelyButNotByPrefix(t *testing.T) {
	h := &routingHandler{}
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
		if got := h.hostAllowed(host, 8123); got != want {
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
		return "POST /v1/chat/completions?denied=1 HTTP/1.1\r\nHost: 127.0.0.1:" + strconv.Itoa(port) +
			"\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n"
	}

	for _, name := range []string{"disabled", "moved"} {
		t.Run(name, func(t *testing.T) {
			p, w := zeroFacadeProxy(t)
			port := startRouting(t, p, w, "1")
			socket := routingSocket(p)

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
			if resp.StatusCode != http.StatusBadRequest {
				t.Errorf("status = %d, want query refusal 400", resp.StatusCode)
			}

			// Headers never finished.
			stalledHeader := dial(t, port)
			send(t, stalledHeader, "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:")

			started := time.Now()
			if name == "moved" {
				startRouting(t, p, w, "2")
			} else {
				activate(t, p, w, listenerPolicy(t, "2", nodepolicy.Listener{}))
			}
			limit := time.Second
			if name == "disabled" {
				limit = routingShutdownGrace + 500*time.Millisecond
			}
			if elapsed := time.Since(started); elapsed > limit {
				t.Errorf("activating the new listener setting took %v with stalled clients connected, want under %v", elapsed, limit)
			}
			assertRoutingClosed(t, socket)
		})
	}
}

// Replace the handler before the existing Serve gate opens. This exercises the
// production server lifecycle independently of inference selection.
func routingLifetimeFixture(t *testing.T, p *Proxy, w *recordingWriter, handler http.Handler) int {
	t.Helper()
	gate := make(chan struct{})
	defer close(gate)
	p.routing.beforeServe = func() { <-gate }
	port := startRouting(t, p, w, "1")
	srv := routingHTTPServer(p)
	original := srv.Handler
	srv.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/fixture/") {
			handler.ServeHTTP(w, r)
			return
		}
		original.ServeHTTP(w, r)
	})
	return port
}

func TestRoutingListener_StreamSurvivesRebindAndFormerWriteTimeout(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	release := make(chan struct{})
	finish := sync.OnceFunc(func() { close(release) })
	defer finish()
	done := make(chan error, 1)
	first := routingLifetimeFixture(t, p, w, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.WriteString(w, "first\n"); err != nil {
			done <- err
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			done <- err
			return
		}
		select {
		case <-release:
		case <-r.Context().Done():
			done <- r.Context().Err()
			return
		}
		if _, err := io.WriteString(w, "last\n"); err != nil {
			done <- err
			return
		}
		done <- http.NewResponseController(w).Flush()
	}))
	client := &http.Client{Timeout: proxyReadHeaderTimeout + 5*time.Second, Transport: &http.Transport{DisableKeepAlives: true}}
	resp, err := client.Get("http://127.0.0.1:" + strconv.Itoa(first) + "/fixture/stream")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	reader := bufio.NewReader(resp.Body)
	if line, err := reader.ReadString('\n'); err != nil || line != "first\n" {
		t.Fatalf("initial stream frame %q, error %v", line, err)
	}
	srv := routingHTTPServer(p)
	old := routingSocket(p)
	port := startRouting(t, p, w, "2")
	if routingHTTPServer(p) != srv {
		t.Fatal("streaming rebind replaced the server")
	}
	assertRoutingClosed(t, old)
	if got := call(t, port, http.MethodGet, routingModelsPath, nil); got.status != http.StatusOK {
		t.Fatalf("new listener status %d", got.status)
	}
	timer := time.NewTimer(proxyReadHeaderTimeout + 100*time.Millisecond)
	defer timer.Stop()
	select {
	case err := <-done:
		t.Fatalf("stream ended during rebind: %v", err)
	case <-timer.C:
	}
	finish()
	if tail, err := io.ReadAll(reader); err != nil || string(tail) != "last\n" {
		t.Fatalf("terminal stream frame %q, error %v", tail, err)
	}
	if err := <-done; err != nil {
		t.Fatalf("terminal flush: %v", err)
	}
}

func TestRoutingListener_DisableAndReenableCreatesFreshServer(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	startRouting(t, p, w, "1")
	srv := routingHTTPServer(p)
	activate(t, p, w, listenerPolicy(t, "2", nodepolicy.Listener{}))
	if routingHTTPServer(p) != nil || routingSocket(p) != nil || boundPort(p) != 0 {
		t.Fatal("disabled listener retains acceptance or server state")
	}
	port := startRouting(t, p, w, "3")
	if routingHTTPServer(p) == srv {
		t.Fatal("re-enable reused a shut down HTTP server")
	}
	if got := call(t, port, http.MethodGet, routingModelsPath, nil); got.status != http.StatusOK {
		t.Fatalf("re-enabled listener status %d", got.status)
	}
}

func TestRoutingListener_ShutdownClosesBlockedBodyWithinGrace(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	entered := make(chan struct{})
	done := make(chan error, 1)
	port := routingLifetimeFixture(t, p, w, http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		close(entered)
		_, err := io.ReadAll(r.Body)
		done <- err
	}))
	conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if _, err := fmt.Fprintf(conn, "POST /fixture/read HTTP/1.1\r\nHost: localhost:%d\r\nContent-Length: 100\r\n\r\nx", port); err != nil {
		t.Fatal(err)
	}
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("body reader did not start")
	}
	socket := routingSocket(p)
	ctx, cancel := context.WithTimeout(context.Background(), routingShutdownGrace)
	defer cancel()
	started := time.Now()
	p.shutdown(ctx)
	if elapsed := time.Since(started); elapsed > routingShutdownGrace+500*time.Millisecond {
		t.Fatalf("blocked body shutdown took %v", elapsed)
	}
	assertRoutingClosed(t, socket)
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("incomplete request body unexpectedly completed")
		}
	case <-time.After(time.Second):
		t.Fatal("forced socket close did not release the body reader")
	}
}

func inferenceListener(t *testing.T, fx *tierFixture) int {
	t.Helper()
	t.Cleanup(func() { fx.p.shutdown(context.Background()) })
	return startRoutingPolicy(t, fx.p, fx.w, fx.p.admission.currentPolicy())
}

func inferencePost(t *testing.T, port int, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost,
		"http://127.0.0.1:"+strconv.Itoa(port)+routeChatCompletions, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := noKeepAlive.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

func waitInferenceDrained(t *testing.T, p *Proxy) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		requests := len(p.activeRoutingRequests())
		p.buffers.mu.Lock()
		used := p.buffers.used
		p.buffers.mu.Unlock()
		if requests == 0 && used == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("inference did not drain: requests=%d bytes=%d", requests, used)
		}
		time.Sleep(time.Millisecond)
	}
	ownedAssertHandlerDrained(t, p, nil)
}

func TestModelControlsSelectionListenerShortAlias512ByteConcrete(t *testing.T) {
	test := func(name, model string) {
		t.Run(name, func(t *testing.T) {
			pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: model}}})
			pol.TierPolicy.AllowStronger = false
			fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"peer": {model}}, nil)
			fx.advertise(t, "peer", map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole(model)}, strictCapabilityFor("ollama", model))
			pinRecordingPairPeer(t, fx.p.soleFacade(), "peer", fx.eng["peer"])
			port := inferenceListener(t, fx)
			resp := inferencePost(t, port, `{"model":"weak"}`)
			if _, err := io.ReadAll(resp.Body); err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != http.StatusOK || fx.eng["peer"].hits() != 1 {
				t.Fatalf("public HTTP listener status/upstream hits = %d/%d", resp.StatusCode, fx.eng["peer"].hits())
			}
			var forwarded struct {
				Model string `json:"model"`
			}
			if err := json.Unmarshal([]byte(fx.eng["peer"].lastBody()), &forwarded); err != nil {
				t.Fatal(err)
			}
			if len(model) != 512 || forwarded.Model != model {
				t.Fatal("listener changed the selected 512-byte concrete model")
			}
			strict := decodeStrictHeader(t, fx.eng["peer"].lastHeader())
			if strict.Request.OriginNodeID != fx.p.originNodeID || strict.Request.ProxyRunID != fx.p.runID || strict.Request.RequestID != "1" {
				t.Fatalf("peer dispatch lost canonical key: %+v", strict.Request)
			}
			if nodepolicy.PublicDispatchAvailable() {
				t.Fatal("provider API egress enabled")
			}
			waitInferenceDrained(t, fx.p)
		})
	}
	test("ASCII concrete model", strings.Repeat("a", 508)+":tag")
	test("multibyte concrete model", strings.Repeat("界", 169)+"a:tag")
}

func TestModelControlsIngressListenerKeyedRefusal(t *testing.T) {
	test := func(name, body string) {
		t.Run(name, func(t *testing.T) {
			var logs bytes.Buffer
			previous := slog.Default()
			slog.SetDefault(slog.New(slog.NewTextHandler(&logs, nil)))
			t.Cleanup(func() { slog.SetDefault(previous) })
			fx := hybridPeerFixture(t, map[string]int{"peer": http.StatusOK})
			port := inferenceListener(t, fx)
			before := fx.p.requestSequence
			resp := inferencePost(t, port, body)
			data, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			var refusal struct {
				Error string `json:"error"`
				Code  string `json:"code"`
			}
			if err := json.Unmarshal(data, &refusal); err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != http.StatusBadRequest || refusal.Error != "invalid model identifier" || (refusal.Code != "request-body" && refusal.Code != "request-identifier") {
				t.Fatalf("identifier status/refusal = %d/%+v", resp.StatusCode, refusal)
			}
			waitInferenceDrained(t, fx.p)
			if fx.p.requestSequence != before+1 || fx.eng["peer"].hits() != 0 || len(notificationsOf(t, fx.w, workloadSubmittedMethod)) != 0 {
				t.Fatal("invalid model lacked exactly one canonical key or caused dispatch effects")
			}
			if logs.Len() == 0 {
				t.Fatal("log capture did not observe the rejection path")
			}
			if bytes.Contains(data, []byte("sentinel")) || bytes.Contains(bytes.Join(fx.w.lines(), nil), []byte("sentinel")) || bytes.Contains(logs.Bytes(), []byte("sentinel")) {
				t.Fatal("invalid model echoed to response, notification or log")
			}
		})
	}
	test("newline", `{"model":"sentinel\nmodel"}`)
	test("DEL", `{"model":"sentinel\u007fmodel"}`)
	test("C1", `{"model":"sentinel\u0085model"}`)
	test("raw invalid UTF8", "{\"model\":\"sentinel"+string([]byte{0xff})+"model\"}")
	test("folded name", `{"model":"weak","N\u0041ME":"sentinel\nmodel"}`)
}

func TestRoutingListener_InferenceModelMembers(t *testing.T) {
	test := func(name, body string, wantStatus, wantHits int, wantCode string) {
		t.Run(name, func(t *testing.T) {
			fx := hybridPeerFixture(t, map[string]int{"peer": http.StatusOK})
			port := inferenceListener(t, fx)
			resp := inferencePost(t, port, body)
			data, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != wantStatus || fx.eng["peer"].hits() != wantHits {
				t.Fatalf("status/hits = %d/%d, want %d/%d; body %s", resp.StatusCode, fx.eng["peer"].hits(), wantStatus, wantHits, data)
			}
			if wantCode != "" && (answer{body: data}).code(t) != wantCode {
				t.Fatalf("refusal = %s, want code %s", data, wantCode)
			}
			if wantHits != 0 {
				var forwarded struct {
					Model   string `json:"model"`
					Payload struct {
						Model string `json:"model"`
					} `json:"payload"`
				}
				if err := json.Unmarshal([]byte(fx.eng["peer"].lastBody()), &forwarded); err != nil {
					t.Fatal(err)
				}
				if forwarded.Model != "q:1b" {
					t.Fatalf("forwarded model = %q, want the selected concrete model", forwarded.Model)
				}
				if strings.Contains(body, `"payload"`) && forwarded.Payload.Model != "weak-api" {
					t.Fatalf("nested payload changed: %+v", forwarded.Payload)
				}
				decodeStrictHeader(t, fx.eng["peer"].lastHeader())
			} else if len(notificationsOf(t, fx.w, workloadSubmittedMethod)) != 0 {
				t.Fatal("refused request submitted a workload")
			}
			waitInferenceDrained(t, fx.p)
		})
	}
	test("bare alias", hybridTestBody, http.StatusOK, 1, "")
	test("local alias", `{"model":"weak-local"}`, http.StatusOK, 1, "")
	test("API alias never falls through to local", `{"model":"weak-api"}`, http.StatusNotFound, 0, "")
	test("nested model is payload", `{"model":"weak","payload":{"model":"weak-api"}}`, http.StatusOK, 1, "")
	test("nested model cannot select", `{"payload":{"model":"weak"}}`, http.StatusBadRequest, 0, "unsupported-alias")
	test("escaped model key", `{"mo\u0064el":"weak"}`, http.StatusOK, 1, "")
	test("escaped alias value", `{"model":"we\u0061k"}`, http.StatusOK, 1, "")
	test("duplicate escaped key", `{"model":"weak","mo\u0064el":"weak-local"}`, http.StatusBadRequest, 0, "ambiguous-model")
	test("duplicate unsupported first", `{"model":"weak-api","model":"weak"}`, http.StatusBadRequest, 0, "ambiguous-model")
	test("duplicate unsupported last", `{"model":"weak","model":"ordinary"}`, http.StatusBadRequest, 0, "ambiguous-model")
	test("case folded model key", `{"Model":"weak"}`, http.StatusBadRequest, 0, "ambiguous-model")
	test("ordinary concrete model", `{"model":"q:1b"}`, http.StatusBadRequest, 0, "unsupported-alias")
	test("padded alias", `{"model":" weak "}`, http.StatusBadRequest, 0, "unsupported-alias")
	test("tagged alias", `{"model":"weak:latest"}`, http.StatusBadRequest, 0, "unsupported-alias")
	test("malformed JSON", `{"model":"weak",`, http.StatusBadRequest, 0, "unsupported-alias")
	test("all identifiers checked", `{"model":"`+strings.Repeat("x", maxRequestIdentifierBytes+1)+`","model":"weak"}`, http.StatusBadRequest, 0, "request-body")
}

func TestRoutingListener_InferenceGatesPrecedeBodyAndOwner(t *testing.T) {
	test := func(name, method, path string, mutate func(*http.Request), want int) {
		t.Run(name, func(t *testing.T) {
			p, _ := zeroFacadeProxy(t)
			source := &ownedBodySource{budget: p.buffers, remaining: 100}
			r := httptest.NewRequest(method, "http://localhost:8123"+path, nil)
			r.Body, r.ContentLength = source, 100
			r = r.WithContext(context.WithValue(r.Context(), routingPortContextKey{}, 8123))
			mutate(r)
			w := httptest.NewRecorder()
			(&routingHandler{host: p}).ServeHTTP(w, r)
			if w.Code != want || source.reads != 0 || len(p.activeRoutingRequests()) != 0 || p.requestSequence != 0 {
				t.Fatalf("gate status/reads/sequence = %d/%d/%d, want %d/0/0", w.Code, source.reads, p.requestSequence, want)
			}
			ownedAssertHandlerDrained(t, p, nil)
		})
	}
	unchanged := func(*http.Request) {}
	test("host", http.MethodPost, routeChatCompletions, func(r *http.Request) { r.Host = "evil.test:8123" }, http.StatusForbidden)
	test("origin", http.MethodPost, routeChatCompletions, withHeader("Origin", "https://evil.test"), http.StatusForbidden)
	test("path", http.MethodPost, "/no-such-path", unchanged, http.StatusNotFound)
	test("method", http.MethodGet, routeChatCompletions, unchanged, http.StatusMethodNotAllowed)
	test("query", http.MethodPost, routeChatCompletions+"?model=weak", unchanged, http.StatusBadRequest)
	test("bare query", http.MethodPost, routeChatCompletions+"?", unchanged, http.StatusBadRequest)
	test("preflight", http.MethodOptions, routeChatCompletions, unchanged, http.StatusMethodNotAllowed)
}

func TestRoutingListener_InferenceUsesIndividualBodyLimit(t *testing.T) {
	fx := hybridPeerFixture(t, map[string]int{"peer": http.StatusOK})
	fx.p.maxRequestBytes = 8
	port := inferenceListener(t, fx)
	resp := inferencePost(t, port, hybridTestBody)
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusRequestEntityTooLarge || (answer{body: data}).code(t) != "request-body" || fx.eng["peer"].hits() != 0 {
		t.Fatalf("oversized request = %d %s, hits %d", resp.StatusCode, data, fx.eng["peer"].hits())
	}
	waitInferenceDrained(t, fx.p)
}

func TestRoutingListener_InferencePolicySnapshot(t *testing.T) {
	fx := hybridPeerFixture(t, map[string]int{"a-busy": http.StatusServiceUnavailable, "b-ready": http.StatusOK})
	policyA := fx.p.admission.currentPolicy()
	policyA.Revision = "1"
	policyA.Routing.Limits = nodepolicy.Limits{
		MaxAttempts: 1, PreOutputBudgetSeconds: 30, QueueSeconds: 2, ColdStartSeconds: 3, MaxOutputTokens: 128,
	}
	policyA.Routing.Tiers = nodepolicy.Default().Routing.Tiers
	policyA.Routing.Tiers["weak"] = nodepolicy.TierRouting{BareMode: nodepolicy.BareLocal}
	t.Cleanup(func() { fx.p.shutdown(context.Background()) })
	port := startRoutingPolicy(t, fx.p, fx.w, policyA)

	// Send only the first byte. Canonical registration is the positive barrier
	// proving the ingress snapshot exists before the next policy is activated.
	begin := func(budget time.Duration) (net.Conn, *routingSpend, time.Time) {
		t.Helper()
		conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), time.Second)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = conn.Close() })
		if err := conn.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
			t.Fatal(err)
		}
		sent := time.Now()
		if _, err := fmt.Fprintf(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost:%d\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s", port, len(hybridTestBody), hybridTestBody[:1]); err != nil {
			t.Fatal(err)
		}
		waitForCond(t, 2*time.Second, "held inference canonical registration", func() bool {
			return len(fx.p.activeRoutingRequests()) == 1
		})
		owner := fx.p.routingRequestControl(fx.p.activeRoutingRequests()[0])
		if owner == nil {
			t.Fatal("held inference has no canonical owner")
		}
		owner.mu.Lock()
		deadline := owner.deadline
		owner.mu.Unlock()
		if deadline.Before(sent.Add(budget)) || deadline.After(time.Now().Add(budget)) {
			t.Fatalf("registered deadline = %v, want ingress time plus %v", deadline, budget)
		}
		return conn, owner, deadline
	}
	finish := func(conn net.Conn) answer {
		t.Helper()
		if _, err := io.WriteString(conn, hybridTestBody[1:]); err != nil {
			t.Fatal(err)
		}
		resp, err := http.ReadResponse(bufio.NewReader(conn), nil)
		if err != nil {
			t.Fatal(err)
		}
		data, readErr := io.ReadAll(resp.Body)
		closeErr := resp.Body.Close()
		if readErr != nil || closeErr != nil {
			t.Fatalf("read/close inference response: %v/%v", readErr, closeErr)
		}
		waitInferenceDrained(t, fx.p)
		return answer{status: resp.StatusCode, header: resp.Header, body: data}
	}
	connA, ownerA, deadlineA := begin(30 * time.Second)
	// Build B independently: currentPolicy returns shared read-only maps and
	// slices, so mutating A's map here would invalidate the activation barrier.
	policyB := fx.p.admission.currentPolicy()
	policyB.Revision = "2"
	policyB.Routing.Limits = nodepolicy.Limits{
		MaxAttempts: 2, PreOutputBudgetSeconds: 60, QueueSeconds: 6, ColdStartSeconds: 7, MaxOutputTokens: 256,
	}
	policyB.Routing.Tiers = nodepolicy.Default().Routing.Tiers
	policyB.Routing.Tiers["weak"] = nodepolicy.TierRouting{BareMode: nodepolicy.BareAPI}
	activate(t, fx.p, fx.w, policyB)
	if got := fx.p.admission.currentPolicy().Revision; got != "2" {
		t.Fatalf("active revision = %q, want B", got)
	}
	gotA := finish(connA)
	ownerA.mu.Lock()
	gotDeadlineA, attemptsA := ownerA.deadline, ownerA.ordinal
	ownerA.mu.Unlock()
	if gotA.status != http.StatusServiceUnavailable || fx.eng["a-busy"].hits() != 1 || fx.eng["b-ready"].hits() != 0 || attemptsA != 1 {
		t.Fatalf("A status/busy/ready/attempts = %d/%d/%d/%d, want 503/1/0/1; body %s",
			gotA.status, fx.eng["a-busy"].hits(), fx.eng["b-ready"].hits(), attemptsA, gotA.body)
	}
	if !gotDeadlineA.Equal(deadlineA) {
		t.Fatalf("A deadline changed after B activation: %v -> %v", deadlineA, gotDeadlineA)
	}
	strict := decodeStrictHeader(t, fx.eng["a-busy"].lastHeader())
	if strict.Request != ownerA.key || strict.PreOutputBudgetMs <= 0 || strict.PreOutputBudgetMs > 30000 ||
		strict.QueueBudgetMs <= 0 || strict.QueueBudgetMs > 2000 || strict.ColdStartBudgetMs <= 0 || strict.ColdStartBudgetMs > 3000 {
		t.Fatalf("A strict request = %+v, want A's canonical key and finite A budgets", strict)
	}
	var forwarded struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal([]byte(fx.eng["a-busy"].lastBody()), &forwarded); err != nil {
		t.Fatal(err)
	}
	if forwarded.Model != "q:1b" {
		t.Fatalf("A forwarded model = %q, want qualified local q:1b", forwarded.Model)
	}

	connB, ownerB, deadlineB := begin(60 * time.Second)
	gotB := finish(connB)
	ownerB.mu.Lock()
	gotDeadlineB, attemptsB := ownerB.deadline, ownerB.ordinal
	ownerB.mu.Unlock()
	if gotB.status != http.StatusNotFound || fx.eng["a-busy"].hits() != 1 || fx.eng["b-ready"].hits() != 0 || attemptsB != 0 {
		t.Fatalf("B status/busy/ready/attempts = %d/%d/%d/%d, want 404/1/0/0; body %s",
			gotB.status, fx.eng["a-busy"].hits(), fx.eng["b-ready"].hits(), attemptsB, gotB.body)
	}
	if !gotDeadlineB.Equal(deadlineB) {
		t.Fatalf("B deadline changed during dispatch: %v -> %v", deadlineB, gotDeadlineB)
	}
}

func TestRoutingListener_InferenceEmptyAdapterCannotAcquireLaterFacade(t *testing.T) {
	fx := hybridPeerFixture(t, map[string]int{"peer": http.StatusOK})
	f := fx.p.soleFacade()
	fx.p.facadeMu.Lock()
	delete(fx.p.facades, f.profile.Name)
	fx.p.facadeMu.Unlock()
	port := inferenceListener(t, fx)
	conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if err := conn.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := fmt.Fprintf(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost:%d\r\nContent-Length: %d\r\n\r\n%s", port, len(hybridTestBody), hybridTestBody[:1]); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for len(fx.p.activeRoutingRequests()) != 1 {
		if time.Now().After(deadline) {
			t.Fatal("adapter did not enter the shared owner")
		}
		time.Sleep(time.Millisecond)
	}
	fx.p.facadeMu.Lock()
	fx.p.facades[f.profile.Name] = f
	fx.p.facadeMu.Unlock()
	if _, err := io.WriteString(conn, hybridTestBody[1:]); err != nil {
		t.Fatal(err)
	}
	resp, err := http.ReadResponse(bufio.NewReader(conn), nil)
	if err != nil {
		t.Fatal(err)
	}
	data, readErr := io.ReadAll(resp.Body)
	closeErr := resp.Body.Close()
	if readErr != nil || closeErr != nil {
		t.Fatalf("read/close refusal: %v/%v", readErr, closeErr)
	}
	if resp.StatusCode != http.StatusNotFound || fx.eng["peer"].hits() != 0 || len(notificationsOf(t, fx.w, workloadSubmittedMethod)) != 0 {
		t.Fatalf("empty adapter acquired later facade: %d %s, hits %d", resp.StatusCode, data, fx.eng["peer"].hits())
	}
	waitInferenceDrained(t, fx.p)
	// A new request can use the enabled facade and its own ID sequence.
	ready := inferencePost(t, port, hybridTestBody)
	if _, err := io.Copy(io.Discard, ready.Body); err != nil {
		t.Fatal(err)
	}
	if ready.StatusCode != http.StatusOK || fx.eng["peer"].hits() != 1 {
		t.Fatalf("new request status/hits = %d/%d", ready.StatusCode, fx.eng["peer"].hits())
	}
	waitInferenceDrained(t, fx.p)
}

// Replace only a private peer endpoint; qualification and strict advertisements
// remain those of hybridPeerFixture. No installed port enters discovery.
func inferencePeerHandler(t *testing.T, fx *tierFixture, id string, handler http.Handler) {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	n := nodeFor(t, id, server.URL)
	n.Models = []string{"q:1b"}
	fx.p.soleFacade().discovery.AddManual(n)
	fx.advertise(t, id, map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("q:1b")}, strictCapabilityFor("ollama", "q:1b"))
}

func TestRoutingListener_InferencePartialResponseNeverFallsBack(t *testing.T) {
	fx := hybridPeerFixture(t, map[string]int{"a-partial": http.StatusOK, "b-ready": http.StatusOK})
	var hits atomic.Int32
	inferencePeerHandler(t, fx, "a-partial", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, continuitySSEEvent); err != nil {
			t.Error(err)
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			t.Error(err)
			return
		}
		panic(http.ErrAbortHandler)
	}))
	port := inferenceListener(t, fx)
	resp := inferencePost(t, port, `{"model":"weak","stream":true}`)
	data, err := io.ReadAll(resp.Body)
	if err != nil || !strings.HasPrefix(string(data), continuitySSEEvent) || resp.StatusCode != http.StatusOK {
		t.Fatalf("partial response = %d %q, %v", resp.StatusCode, data, err)
	}
	assertContinuityError(t, streamProtocolSSE, data, streamInterruptReasonReadFailed)
	if hits.Load() != 1 || fx.eng["b-ready"].hits() != 0 {
		t.Fatalf("partial/fallback hits = %d/%d", hits.Load(), fx.eng["b-ready"].hits())
	}
	waitInferenceDrained(t, fx.p)
}

func TestRoutingListener_InferenceHealthyStreamSurvivesActivationAndFormerWriteTimeout(t *testing.T) {
	fx := hybridPeerFixture(t, map[string]int{"peer": http.StatusOK})
	release := make(chan struct{})
	finish := sync.OnceFunc(func() { close(release) })
	defer finish()
	done := make(chan error, 1)
	inferencePeerHandler(t, fx, "peer", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, continuitySSEEvent); err != nil {
			done <- err
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			done <- err
			return
		}
		select {
		case <-release:
		case <-r.Context().Done():
			done <- r.Context().Err()
			return
		}
		_, err := io.WriteString(w, "data: [DONE]\n\n")
		done <- err
	}))
	port := inferenceListener(t, fx)
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost,
		"http://127.0.0.1:"+strconv.Itoa(port)+routeChatCompletions, strings.NewReader(`{"model":"weak","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Timeout: proxyReadHeaderTimeout + 5*time.Second, Transport: &http.Transport{DisableKeepAlives: true}}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	first := make([]byte, len(continuitySSEEvent))
	if _, err := io.ReadFull(resp.Body, first); err != nil || string(first) != continuitySSEEvent {
		t.Fatalf("first frame %q, %v", first, err)
	}
	if len(fx.p.activeRoutingRequests()) != 1 {
		t.Fatal("committed stream is missing its canonical owner")
	}
	pol := fx.p.admission.currentPolicy()
	pol.Revision = "2"
	pol.Routing.Limits.PreOutputBudgetSeconds = 1
	started := time.Now()
	activate(t, fx.p, fx.w, pol)
	pol.Revision = "3"
	activate(t, fx.p, fx.w, pol)
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("activations stalled behind committed inference for %v", elapsed)
	}
	timer := time.NewTimer(proxyReadHeaderTimeout + 100*time.Millisecond)
	defer timer.Stop()
	select {
	case err := <-done:
		t.Fatalf("committed stream ended before release: %v", err)
	case <-timer.C:
	}
	finish()
	tail, err := io.ReadAll(resp.Body)
	if err != nil || string(tail) != "data: [DONE]\n\n" || resp.StatusCode != http.StatusOK {
		t.Fatalf("stream tail %q, %v, status %d", tail, err, resp.StatusCode)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	waitInferenceDrained(t, fx.p)
}

func TestRoutingListener_InferenceBlockedBodyShutdownWithinGrace(t *testing.T) {
	p, w := zeroFacadeProxy(t)
	port := startRouting(t, p, w, "1")
	conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if _, err := fmt.Fprintf(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost:%d\r\nContent-Length: 100\r\n\r\n{", port); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for len(p.activeRoutingRequests()) != 1 {
		if time.Now().After(deadline) {
			t.Fatal("body reader did not register its owner")
		}
		time.Sleep(time.Millisecond)
	}
	started := time.Now()
	p.shutdown(context.Background())
	if elapsed := time.Since(started); elapsed > routingShutdownGrace+500*time.Millisecond {
		t.Fatalf("body shutdown took %v", elapsed)
	}
	waitInferenceDrained(t, p)
}

func TestRoutingListener_InferenceBlockedClientShutdownWithinGrace(t *testing.T) {
	fx := hybridPeerFixture(t, map[string]int{"peer": http.StatusOK})
	done := make(chan struct{})
	inferencePeerHandler(t, fx, "peer", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(done)
		w.Header().Set("Content-Type", "text/event-stream")
		chunk := `data: {"choices":[{"index":0,"delta":{"content":"` + strings.Repeat("x", 1024) + `"}}]}` + "\n\n"
		for {
			if _, err := io.WriteString(w, chunk); err != nil {
				return
			}
			if err := http.NewResponseController(w).Flush(); err != nil {
				return
			}
			if r.Context().Err() != nil {
				return
			}
		}
	}))
	port := inferenceListener(t, fx)
	conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	tcp, ok := conn.(*net.TCPConn)
	if !ok {
		t.Fatalf("client connection = %T", conn)
	}
	if err := tcp.SetReadBuffer(1024); err != nil {
		t.Fatal(err)
	}
	body := `{"model":"weak","stream":true}`
	if _, err := fmt.Fprintf(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost:%d\r\nContent-Length: %d\r\n\r\n%s", port, len(body), body); err != nil {
		t.Fatal(err)
	}
	// Never read from this client. An unbounded private producer must fill its
	// socket buffers: require positive, then stalled sink progress before stop.
	deadline := time.Now().Add(5 * time.Second)
	var previous int64
	for {
		var sink *responseSink
		fx.p.identityMu.Lock()
		for _, request := range fx.p.requests {
			request.owner.mu.Lock()
			if request.owner.committed {
				sink = request.owner.sink
			}
			request.owner.mu.Unlock()
		}
		fx.p.identityMu.Unlock()
		if sink != nil {
			sink.mu.Lock()
			written, produced, flushed := sink.written, sink.produced, sink.flushed
			sink.mu.Unlock()
			if written > 0 && written == previous && (produced > written || flushed < written) {
				break
			}
			previous = written
		}
		if time.Now().After(deadline) {
			t.Fatalf("client never demonstrated a stalled response sink; last completed bytes=%d", previous)
		}
		time.Sleep(100 * time.Millisecond)
	}
	started := time.Now()
	fx.p.shutdown(context.Background())
	if elapsed := time.Since(started); elapsed > routingShutdownGrace+500*time.Millisecond {
		t.Fatalf("blocked client shutdown took %v", elapsed)
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("shutdown did not close the upstream body")
	}
	waitInferenceDrained(t, fx.p)
}
