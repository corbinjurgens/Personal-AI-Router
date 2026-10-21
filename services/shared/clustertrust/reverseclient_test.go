// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"context"
	"crypto/tls"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/noderec"
)

type serviceClientConnContextKey struct{}

type serviceClientFixture struct {
	adapter      *PeerServiceClient
	mesh         *Mesh
	peer         *Mesh
	dir          string
	peerDir      string
	server       *httptest.Server
	url          string
	mu           sync.Mutex
	routes       map[noderec.ServiceKey]PeerServiceRoute
	directCalls  map[noderec.ServiceKey]int
	reverseCalls map[noderec.ServiceKey]int
	direct       func(context.Context, noderec.ServiceKey) (net.Conn, error)
	reverse      func(context.Context, noderec.ServiceKey) (net.Conn, error)
	dispatchMu   sync.Mutex
	dispatches   []net.Conn
	closed       chan struct{}
}

// assertDispatchConnections labels server connections by their first dispatch.
// Equal labels require reuse; different labels require distinct connections.
// Accepted sockets that never carry a request do not enter this sequence.
func (f *serviceClientFixture) assertDispatchConnections(t *testing.T, want ...int) {
	t.Helper()
	f.dispatchMu.Lock()
	defer f.dispatchMu.Unlock()
	if len(f.dispatches) != len(want) {
		t.Fatalf("dispatch connections: got %d dispatches, want %d (identities %v)", len(f.dispatches), len(want), want)
	}
	for i, conn := range f.dispatches {
		for j := 0; j < i; j++ {
			if (conn == f.dispatches[j]) != (want[i] == want[j]) {
				t.Fatalf("dispatch connections %d and %d: same=%v, want identities %v", j, i, conn == f.dispatches[j], want)
			}
		}
	}
}

func (f *serviceClientFixture) recordDispatch(t *testing.T, r *http.Request) bool {
	t.Helper()
	conn, ok := r.Context().Value(serviceClientConnContextKey{}).(net.Conn)
	if !ok {
		t.Error("request has no server connection identity")
		return false
	}
	f.dispatchMu.Lock()
	f.dispatches = append(f.dispatches, conn)
	f.dispatchMu.Unlock()
	return true
}

func newServiceClientFixture(t *testing.T, handler http.Handler) *serviceClientFixture {
	t.Helper()
	selfPEM, selfKey, _ := genLeaf(t, "self")
	peerPEM, peerKey, _ := genLeaf(t, "peer")
	self, dir := meshDir(t, selfPEM, selfKey, map[string]string{"peer": string(peerPEM)})
	peer, peerDir := meshDir(t, peerPEM, peerKey, map[string]string{"self": string(selfPEM)})
	f := &serviceClientFixture{mesh: self, peer: peer, dir: dir, peerDir: peerDir, routes: make(map[noderec.ServiceKey]PeerServiceRoute), directCalls: make(map[noderec.ServiceKey]int), reverseCalls: make(map[noderec.ServiceKey]int), closed: make(chan struct{}, 1)}
	f.server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if uuid, ok := peer.VerifyClientPin(r); !ok || uuid != "self" {
			t.Error("inner listener did not receive the currently pinned client certificate")
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		host, _, err := net.SplitHostPort(r.Host)
		if err != nil {
			t.Error(err)
			return
		}
		if r.TLS.ServerName != host {
			t.Errorf("TLS ServerName = %q, want original host %q", r.TLS.ServerName, host)
		}
		f.mu.Lock()
		known := false
		for _, route := range f.routes {
			if route.Address == r.Host {
				known = true
			}
		}
		f.mu.Unlock()
		if !known {
			t.Errorf("Host = %q, not an original route URL host", r.Host)
		}
		if !f.recordDispatch(t, r) {
			http.Error(w, "missing connection identity", http.StatusInternalServerError)
			return
		}
		handler.ServeHTTP(w, r)
	}))
	f.server.TLS = peer.ServerTLSConfig()
	f.server.Config.ConnContext = func(ctx context.Context, conn net.Conn) context.Context {
		return context.WithValue(ctx, serviceClientConnContextKey{}, conn)
	}
	f.server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateClosed {
			select {
			case f.closed <- struct{}{}:
			default:
			}
		}
	}
	f.server.StartTLS()
	t.Cleanup(f.server.Close)
	_, port, err := net.SplitHostPort(f.server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	f.url = "https://peer.example:" + port
	f.routes[noderec.ServiceCluster] = PeerServiceRoute{Incarnation: "v1", Address: "peer.example:" + port, Protocol: ReverseTLS, Generation: 1}
	f.routes[noderec.ServiceOllama] = PeerServiceRoute{Incarnation: "v1", Address: "proxy.example:12345", Protocol: ReverseTLS, Generation: 1}
	dial := func(ctx context.Context, _ noderec.ServiceKey) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, "tcp", f.server.Listener.Addr().String())
	}
	f.direct = dial
	remote := reverseTestManager(t, ReverseOptions{LocalUUID: "peer", ResolveService: func(ctx context.Context, peer ReversePeer, service noderec.ServiceKey) (ReverseTarget, error) {
		return ReverseTarget{Protocol: ReverseTLS, Dial: func(ctx context.Context) (net.Conn, error) { return dial(ctx, service) }}, nil
	}})
	local := reverseTestManager(t, ReverseOptions{LocalUUID: "self"})
	reverseTestConnect(t, local, remote, "self", reverseTestID)
	f.reverse = func(ctx context.Context, service noderec.ServiceKey) (net.Conn, error) {
		return local.Open(ctx, ReversePeer{"peer", "v1"}, service)
	}
	f.adapter = NewPeerServiceClient(self, PeerServiceClientOptions{
		PeerClientOptions: PeerClientOptions{Timeout: 2 * time.Second}, DirectTimeout: 80 * time.Millisecond, ProbeInterval: time.Hour,
		ResolveRoute: func(uuid string, service noderec.ServiceKey) (PeerServiceRoute, bool) {
			f.mu.Lock()
			defer f.mu.Unlock()
			route, ok := f.routes[service]
			return route, ok && uuid == "peer"
		},
		DialContext: func(ctx context.Context, _, address string) (net.Conn, error) {
			f.mu.Lock()
			service := noderec.ServiceCluster
			for key, route := range f.routes {
				if route.Address == address {
					service = key
					break
				}
			}
			f.directCalls[service]++
			dial := f.direct
			f.mu.Unlock()
			return dial(ctx, service)
		},
		DialReverse: func(ctx context.Context, _, uuid string, service noderec.ServiceKey) (net.Conn, error) {
			if uuid != "peer" {
				return nil, errors.New("unexpected peer UUID")
			}
			f.mu.Lock()
			f.reverseCalls[service]++
			dial := f.reverse
			f.mu.Unlock()
			return dial(ctx, service)
		},
	})
	t.Cleanup(f.adapter.CloseIdle)
	return f
}

func serviceClientGet(t *testing.T, client *http.Client, target string) string {
	t.Helper()
	response, err := client.Get(target)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", response.StatusCode)
	}
	return string(body)
}

func writeServiceResponse(t *testing.T, w http.ResponseWriter, body string) {
	t.Helper()
	if _, err := io.WriteString(w, body); err != nil {
		t.Errorf("write test response: %v", err)
	}
}

func TestPeerServiceClientDirectFirst(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/original/path" || r.URL.RawQuery != "test=1" {
			t.Errorf("request URL = %s", r.URL)
		}
		writeServiceResponse(t, w, "direct")
	}))
	if got := serviceClientGet(t, f.adapter.Client("peer", noderec.ServiceCluster), f.url+"/original/path?test=1"); got != "direct" {
		t.Fatalf("body = %q", got)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.reverseCalls) != 0 {
		t.Fatalf("reachable direct endpoint opened reverse: %v", f.reverseCalls)
	}
}

func TestPeerServiceClientTCPTimeoutFallsBackWithinCallerDeadline(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { writeServiceResponse(t, w, "reverse") }))
	var attempts atomic.Int32
	f.direct = func(ctx context.Context, _ noderec.ServiceKey) (net.Conn, error) {
		attempts.Add(1)
		<-ctx.Done()
		return nil, ctx.Err()
	}
	client := f.adapter.Client("peer", noderec.ServiceCluster)
	client.Timeout = 350 * time.Millisecond
	start := time.Now()
	if got := serviceClientGet(t, client, f.url); got != "reverse" {
		t.Fatalf("body = %q", got)
	}
	if elapsed := time.Since(start); elapsed >= client.Timeout {
		t.Fatalf("fallback consumed caller budget: %v", elapsed)
	}
	if got := serviceClientGet(t, client, f.url); got != "reverse" {
		t.Fatalf("second body = %q", got)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("direct attempts = %d, want exactly one (cached failure)", got)
	}
}

func TestPeerServiceClientTLSResetFallsBack(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { writeServiceResponse(t, w, "reverse") }))
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	firstTLSByte := make(chan byte, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		var first [1]byte
		if _, err := io.ReadFull(conn, first[:]); err == nil {
			firstTLSByte <- first[0]
		}
		if tcp, ok := conn.(*net.TCPConn); ok {
			if err := tcp.SetLinger(0); err != nil {
				t.Error(err)
			}
		}
	}()
	f.direct = func(ctx context.Context, _ noderec.ServiceKey) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, "tcp", listener.Addr().String())
	}
	if got := serviceClientGet(t, f.adapter.Client("peer", noderec.ServiceCluster), f.url); got != "reverse" {
		t.Fatalf("body = %q", got)
	}
	select {
	case got := <-firstTLSByte:
		if got != 22 {
			t.Fatalf("first TLS record type = %d, want handshake 22", got)
		}
	case <-time.After(time.Second):
		t.Fatal("direct TCP never received a TLS handshake")
	}
}

func TestPeerServiceClientTLSHandshakeBudgetLeavesReverseTime(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { writeServiceResponse(t, w, "reverse") }))
	var directClosed = make(chan struct{})
	f.direct = func(ctx context.Context, _ noderec.ServiceKey) (net.Conn, error) {
		client, server := net.Pipe()
		go func() {
			defer close(directClosed)
			defer server.Close()
			if _, err := io.Copy(io.Discard, server); err != nil {
				t.Error(err)
			}
		}()
		return client, nil
	}
	client := f.adapter.Client("peer", noderec.ServiceCluster)
	client.Timeout = 350 * time.Millisecond
	if got := serviceClientGet(t, client, f.url); got != "reverse" {
		t.Fatalf("body = %q", got)
	}
	select {
	case <-directClosed:
	case <-time.After(time.Second):
		t.Fatal("stalled TLS connection survived fallback")
	}
}

func TestPeerServiceClientIndependentServiceHealth(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { writeServiceResponse(t, w, "ok") }))
	f.routes[noderec.ServiceOllama] = PeerServiceRoute{Incarnation: "v1", Address: "proxy.example:12345", Protocol: ReverseTLS, Generation: 1}
	reachable := f.direct
	f.direct = func(ctx context.Context, service noderec.ServiceKey) (net.Conn, error) {
		if service == noderec.ServiceOllama {
			return nil, errors.New("proxy blocked")
		}
		return reachable(ctx, service)
	}
	proxyURL := "https://proxy.example:12345"
	serviceClientGet(t, f.adapter.Client("peer", noderec.ServiceOllama), proxyURL)
	serviceClientGet(t, f.adapter.Client("peer", noderec.ServiceCluster), f.url)
	serviceClientGet(t, f.adapter.Client("peer", noderec.ServiceOllama), proxyURL)
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.directCalls[noderec.ServiceCluster] != 1 || f.directCalls[noderec.ServiceOllama] != 1 || f.reverseCalls[noderec.ServiceOllama] != 2 || f.reverseCalls[noderec.ServiceCluster] != 0 {
		t.Fatalf("route attempts direct=%v reverse=%v", f.directCalls, f.reverseCalls)
	}
}

func TestPeerServiceClientReadmissionAndGenerationResetHealth(t *testing.T) {
	test := func(name string, change func(*PeerServiceRoute)) {
		t.Run(name, func(t *testing.T) {
			f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { writeServiceResponse(t, w, "ok") }))
			reachable := f.direct
			f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("blocked") }
			client := f.adapter.Client("peer", noderec.ServiceCluster)
			serviceClientGet(t, client, f.url)
			f.mu.Lock()
			route := f.routes[noderec.ServiceCluster]
			change(&route)
			f.routes[noderec.ServiceCluster] = route
			f.direct = reachable
			f.mu.Unlock()
			serviceClientGet(t, client, f.url)
			f.mu.Lock()
			defer f.mu.Unlock()
			if f.reverseCalls[noderec.ServiceCluster] != 1 {
				t.Fatalf("new incarnation/generation reused failed health: %v", f.reverseCalls)
			}
		})
	}
	test("same certificate new admission", func(r *PeerServiceRoute) { r.Incarnation = "v2" })
	test("new endpoint generation", func(r *PeerServiceRoute) { r.Generation = 2 })
}

type serviceSlotWaitContext struct {
	context.Context
	entered chan struct{}
	once    sync.Once
}

func (c *serviceSlotWaitContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.entered) })
	return c.Context.Done()
}

func TestPeerServiceClientAuthorityChangesPreserveConnectionCap(t *testing.T) {
	test := func(name string, reuse bool, change func(*PeerServiceRoute)) {
		t.Run(name, func(t *testing.T) {
			var dispatches atomic.Int32
			f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				dispatches.Add(1)
				writeServiceResponse(t, w, "held")
			}))
			f.adapter.opts.ReuseConnections = reuse
			method := http.MethodGet
			makeBody := func() io.Reader { return nil }
			if reuse {
				method = http.MethodPost
				makeBody = func() io.Reader { return strings.NewReader("mutation") }
			}
			client := f.adapter.Client("peer", noderec.ServiceCluster)
			client.Timeout = 0
			var responses []*http.Response
			t.Cleanup(func() {
				for _, response := range responses {
					if err := response.Body.Close(); err != nil {
						t.Errorf("close held response: %v", err)
					}
				}
			})
			for range 8 {
				req, err := http.NewRequest(method, f.url, makeBody())
				if err != nil {
					t.Fatal(err)
				}
				response, err := client.Do(req)
				if err != nil {
					t.Fatal(err)
				}
				responses = append(responses, response)
			}
			f.mu.Lock()
			route := f.routes[noderec.ServiceCluster]
			change(&route)
			f.routes[noderec.ServiceCluster] = route
			f.mu.Unlock()

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			waiting := &serviceSlotWaitContext{Context: ctx, entered: make(chan struct{})}
			req, err := http.NewRequestWithContext(waiting, method, f.url, makeBody())
			if err != nil {
				t.Fatal(err)
			}
			type result struct {
				response *http.Response
				err      error
			}
			completed := make(chan result, 1)
			go func() {
				// RoundTrip first evaluates Done when selecting its connection
				// slot. Calling it directly avoids http.Client's context watchers.
				response, err := client.Transport.RoundTrip(req)
				completed <- result{response, err}
			}()
			select {
			case <-waiting.entered:
			case <-ctx.Done():
				t.Fatal("ninth request did not reach the connection gate")
			}
			f.adapter.mu.Lock()
			health := f.adapter.entries[peerServiceKey{"peer", noderec.ServiceCluster}]
			occupied, capacity := len(health.slots), cap(health.slots)
			f.adapter.mu.Unlock()
			if occupied != 8 || capacity != 8 {
				t.Fatalf("new authority gate has %d/%d occupied slots, want 8/8", occupied, capacity)
			}
			if got := dispatches.Load(); got != 8 {
				t.Fatalf("dispatches before releasing an old response = %d, want 8", got)
			}
			select {
			case got := <-completed:
				if got.response != nil {
					responses = append(responses, got.response)
				}
				t.Fatalf("ninth request completed while all eight old responses were open: %v", got.err)
			default:
			}

			body, err := io.ReadAll(responses[0].Body)
			if err != nil || string(body) != "held" {
				t.Fatalf("old response after authority change = %q, %v", body, err)
			}
			if err := responses[0].Body.Close(); err != nil {
				t.Fatal(err)
			}
			select {
			case got := <-completed:
				if got.err != nil {
					t.Fatal(got.err)
				}
				responses = append(responses, got.response)
				if got.response.StatusCode != http.StatusOK {
					t.Fatalf("ninth response status = %d, want 200", got.response.StatusCode)
				}
			case <-ctx.Done():
				t.Fatal("ninth request did not complete after releasing an old response")
			}
			if got := dispatches.Load(); got != 9 {
				t.Fatalf("dispatches after releasing one old response = %d, want 9", got)
			}
			if got := len(health.slots); got != 8 {
				t.Fatalf("occupied slots after EOF and Close = %d, want 8 (one release)", got)
			}
		})
	}
	test("generation", false, func(route *PeerServiceRoute) { route.Generation++ })
	test("readmission", false, func(route *PeerServiceRoute) { route.Incarnation = "v2" })
	test("reusable generation", true, func(route *PeerServiceRoute) { route.Generation++ })
	test("reusable readmission", true, func(route *PeerServiceRoute) { route.Incarnation = "v2" })
}

// serviceSlotFixture holds peerMaxConnsPerHost streaming responses open to one
// peer service, over direct connections or over reverse sessions only.
type serviceSlotFixture struct {
	*serviceClientFixture
	client     *http.Client
	streams    []*http.Response
	cancels    []context.CancelFunc
	dispatches atomic.Int32
}

func newServiceSlotFixture(t *testing.T, reverse bool) *serviceSlotFixture {
	t.Helper()
	release := make(chan struct{})
	s := &serviceSlotFixture{}
	s.serviceClientFixture = newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.dispatches.Add(1)
		if r.URL.Path != "/stream" {
			writeServiceResponse(t, w, "ok")
			return
		}
		writeServiceResponse(t, w, "start-")
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(func() {
		close(release)
		for _, response := range s.streams {
			if err := response.Body.Close(); err != nil {
				t.Errorf("close held stream: %v", err)
			}
		}
		for _, cancel := range s.cancels {
			cancel()
		}
	})
	if reverse {
		s.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("blocked") }
	}
	// These cases measure slot admission, not TLS setup. Use the normal direct
	// budget so a healthy loopback handshake survives host load under -race.
	s.adapter.opts.DirectTimeout = time.Second
	s.client = s.adapter.Client("peer", noderec.ServiceCluster)
	s.client.Timeout = 0
	for range peerMaxConnsPerHost {
		ctx, cancel := context.WithCancel(context.Background())
		s.cancels = append(s.cancels, cancel)
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.url+"/stream", nil)
		if err != nil {
			t.Fatal(err)
		}
		response, err := s.client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		s.streams = append(s.streams, response)
	}
	wantDirect, wantReverse := peerMaxConnsPerHost, 0
	if reverse {
		// The first stream's failed direct attempt defers direct probing.
		wantDirect, wantReverse = 1, peerMaxConnsPerHost
	}
	if direct, reverse := s.dialCounts(); direct != wantDirect || reverse != wantReverse {
		t.Fatalf("saturating dials direct=%d reverse=%d, want %d/%d", direct, reverse, wantDirect, wantReverse)
	}
	if got := s.occupiedSlots(); got != peerMaxConnsPerHost {
		t.Fatalf("occupied slots after saturation = %d, want %d", got, peerMaxConnsPerHost)
	}
	return s
}

func (s *serviceSlotFixture) dialCounts() (int, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.directCalls[noderec.ServiceCluster], s.reverseCalls[noderec.ServiceCluster]
}

func (s *serviceSlotFixture) occupiedSlots() int {
	s.adapter.mu.Lock()
	defer s.adapter.mu.Unlock()
	return len(s.adapter.entries[peerServiceKey{"peer", noderec.ServiceCluster}].slots)
}

type serviceRoundTripResult struct {
	response *http.Response
	err      error
}

// roundTrip dispatches through the adapter's transport, so the result carries
// the transport's own error rather than http.Client's wrapping.
func (s *serviceSlotFixture) roundTrip(t *testing.T, ctx context.Context) <-chan serviceRoundTripResult {
	t.Helper()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.url, nil)
	if err != nil {
		t.Fatal(err)
	}
	completed := make(chan serviceRoundTripResult, 1)
	go func() {
		response, err := s.client.Transport.RoundTrip(req)
		completed <- serviceRoundTripResult{response, err}
	}()
	return completed
}

// requireNoDispatch checks that a refused request reserved no slot, dialed
// nothing and reached no handler.
func (s *serviceSlotFixture) requireNoDispatch(t *testing.T, direct, reverse int) {
	t.Helper()
	if gotDirect, gotReverse := s.dialCounts(); gotDirect != direct || gotReverse != reverse {
		t.Fatalf("refused request dialed: direct=%d reverse=%d, want %d/%d", gotDirect, gotReverse, direct, reverse)
	}
	if got := s.dispatches.Load(); got != int32(peerMaxConnsPerHost) {
		t.Fatalf("dispatches = %d, want %d", got, peerMaxConnsPerHost)
	}
	if got := s.occupiedSlots(); got != peerMaxConnsPerHost {
		t.Fatalf("occupied slots = %d, want %d", got, peerMaxConnsPerHost)
	}
}

// requireServiceOK reads a successful response to completion and closes it.
func requireServiceOK(t *testing.T, got serviceRoundTripResult) {
	t.Helper()
	if got.err != nil {
		t.Fatalf("request failed: %v", got.err)
	}
	defer got.response.Body.Close()
	body, err := io.ReadAll(got.response.Body)
	if err != nil || got.response.StatusCode != http.StatusOK || string(body) != "ok" {
		t.Fatalf("response = %d %q, %v; want 200 \"ok\"", got.response.StatusCode, body, err)
	}
}

func requireSlotOccupied(t *testing.T, got serviceRoundTripResult) {
	t.Helper()
	if got.response != nil {
		got.response.Body.Close()
		t.Fatal("occupied request returned a response")
	}
	var unavailable *ReverseUnavailableError
	if !errors.Is(got.err, ErrPeerSlotOccupied) || errors.As(got.err, &unavailable) {
		t.Fatalf("error = %v, want ErrPeerSlotOccupied distinct from ReverseUnavailableError", got.err)
	}
}

func TestPeerServiceClientSlotWait(t *testing.T) {
	const bound = 250 * time.Millisecond
	for _, path := range []struct {
		name    string
		reverse bool
	}{{"direct", false}, {"reverse", true}} {
		t.Run(path.name, func(t *testing.T) {
			t.Run("nonblocking request is occupied immediately", func(t *testing.T) {
				s := newServiceSlotFixture(t, path.reverse)
				direct, reverse := s.dialCounts()
				started := time.Now()
				select {
				case got := <-s.roundTrip(t, WithPeerSlotWait(context.Background(), 0)):
					requireSlotOccupied(t, got)
				case <-time.After(5 * time.Second):
					t.Fatal("nonblocking request waited for a slot")
				}
				if elapsed := time.Since(started); elapsed >= bound {
					t.Fatalf("nonblocking request took %v, want well under %v", elapsed, bound)
				}
				s.requireNoDispatch(t, direct, reverse)
			})

			t.Run("bounded request is occupied at its deadline", func(t *testing.T) {
				s := newServiceSlotFixture(t, path.reverse)
				direct, reverse := s.dialCounts()
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				started := time.Now()
				got := <-s.roundTrip(t, WithPeerSlotWait(ctx, bound))
				elapsed := time.Since(started)
				requireSlotOccupied(t, got)
				if elapsed < bound || ctx.Err() != nil {
					t.Fatalf("bounded request failed after %v (context %v), want occupied at %v", elapsed, ctx.Err(), bound)
				}
				s.requireNoDispatch(t, direct, reverse)
			})

			t.Run("bounded request takes a slot freed within its wait", func(t *testing.T) {
				s := newServiceSlotFixture(t, path.reverse)
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				waiting := &serviceSlotWaitContext{Context: ctx, entered: make(chan struct{})}
				completed := s.roundTrip(t, WithPeerSlotWait(waiting, 4*time.Second))
				select {
				case <-waiting.entered:
				case got := <-completed:
					t.Fatalf("bounded request completed while every slot was held: %v", got.err)
				}
				if err := s.streams[0].Body.Close(); err != nil {
					t.Fatal(err)
				}
				requireServiceOK(t, <-completed)
				if got := s.occupiedSlots(); got != peerMaxConnsPerHost-1 {
					t.Fatalf("occupied slots after completion = %d, want %d", got, peerMaxConnsPerHost-1)
				}
			})

			t.Run("default request still blocks until a slot frees", func(t *testing.T) {
				s := newServiceSlotFixture(t, path.reverse)
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				waiting := &serviceSlotWaitContext{Context: ctx, entered: make(chan struct{})}
				completed := s.roundTrip(t, waiting)
				select {
				case <-waiting.entered:
				case got := <-completed:
					t.Fatalf("default request completed while every slot was held: %v", got.err)
				}
				// Outlast the bounded case: the default has no occupied result.
				select {
				case got := <-completed:
					t.Fatalf("default request stopped waiting before a slot freed: %v", got.err)
				case <-time.After(2 * bound):
				}
				if got := s.dispatches.Load(); got != int32(peerMaxConnsPerHost) {
					t.Fatalf("dispatches while waiting = %d, want %d", got, peerMaxConnsPerHost)
				}
				if err := s.streams[0].Body.Close(); err != nil {
					t.Fatal(err)
				}
				requireServiceOK(t, <-completed)
			})

			t.Run("canceled wait returns the context error and takes no slot", func(t *testing.T) {
				s := newServiceSlotFixture(t, path.reverse)
				direct, reverse := s.dialCounts()
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				waiting := &serviceSlotWaitContext{Context: ctx, entered: make(chan struct{})}
				completed := s.roundTrip(t, WithPeerSlotWait(waiting, 4*time.Second))
				select {
				case <-waiting.entered:
				case got := <-completed:
					t.Fatalf("bounded request completed while every slot was held: %v", got.err)
				}
				cancel()
				got := <-completed
				if got.response != nil {
					got.response.Body.Close()
				}
				if !errors.Is(got.err, context.Canceled) || errors.Is(got.err, ErrPeerSlotOccupied) {
					t.Fatalf("canceled wait error = %v, want context.Canceled", got.err)
				}
				s.requireNoDispatch(t, direct, reverse)
			})

			t.Run("canceled and closed streams release their slots", func(t *testing.T) {
				s := newServiceSlotFixture(t, path.reverse)
				s.cancels[0]()
				if _, err := io.ReadAll(s.streams[0].Body); !errors.Is(err, context.Canceled) {
					t.Fatalf("canceled stream read error = %v, want context.Canceled", err)
				}
				if got := s.occupiedSlots(); got != peerMaxConnsPerHost-1 {
					t.Fatalf("occupied slots after cancellation = %d, want %d", got, peerMaxConnsPerHost-1)
				}
				if err := s.streams[1].Body.Close(); err != nil {
					t.Fatal(err)
				}
				if got := s.occupiedSlots(); got != peerMaxConnsPerHost-2 {
					t.Fatalf("occupied slots after Close = %d, want %d", got, peerMaxConnsPerHost-2)
				}
				for range 2 {
					requireServiceOK(t, <-s.roundTrip(t, WithPeerSlotWait(context.Background(), 0)))
				}
				if got := s.occupiedSlots(); got != peerMaxConnsPerHost-2 {
					t.Fatalf("occupied slots after nonblocking requests = %d, want %d", got, peerMaxConnsPerHost-2)
				}
			})
		})
	}
}

func TestPeerServiceClientPeriodicDirectRecovery(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { writeServiceResponse(t, w, "ok") }))
	reachable := f.direct
	f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("blocked") }
	client := f.adapter.Client("peer", noderec.ServiceCluster)
	serviceClientGet(t, client, f.url)
	f.mu.Lock()
	f.direct = reachable
	f.mu.Unlock()
	f.adapter.mu.Lock()
	f.adapter.entries[peerServiceKey{"peer", noderec.ServiceCluster}].nextProbe = time.Now().Add(-time.Second)
	f.adapter.mu.Unlock()
	serviceClientGet(t, client, f.url)
	serviceClientGet(t, client, f.url)
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.reverseCalls[noderec.ServiceCluster] != 1 {
		t.Fatalf("healthy direct probe did not recover: reverse=%v", f.reverseCalls)
	}
}

func TestPeerServiceClientDirectProbePreservesActiveReverseResponse(t *testing.T) {
	release := make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/stream" {
			writeServiceResponse(t, w, "probe")
			return
		}
		writeServiceResponse(t, w, "start-")
		w.(http.Flusher).Flush()
		select {
		case <-release:
			writeServiceResponse(t, w, "end")
		case <-r.Context().Done():
			t.Error("direct recovery canceled active reverse response")
		}
	}))
	reachable := f.direct
	f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("blocked") }
	client := f.adapter.Client("peer", noderec.ServiceCluster)
	response, err := client.Get(f.url + "/stream")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	f.mu.Lock()
	f.direct = reachable
	f.mu.Unlock()
	f.adapter.mu.Lock()
	f.adapter.entries[peerServiceKey{"peer", noderec.ServiceCluster}].nextProbe = time.Now().Add(-time.Second)
	f.adapter.mu.Unlock()
	if got := serviceClientGet(t, client, f.url+"/probe"); got != "probe" {
		t.Fatalf("probe response = %q", got)
	}
	once.Do(func() { close(release) })
	body, err := io.ReadAll(response.Body)
	if err != nil || string(body) != "start-end" {
		t.Fatalf("active reverse response = %q, %v", body, err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.reverseCalls[noderec.ServiceCluster] != 1 || f.directCalls[noderec.ServiceCluster] != 2 {
		t.Fatalf("recovery routes direct=%v reverse=%v", f.directCalls, f.reverseCalls)
	}
}

func TestPeerServiceClientPinRotationRejectsExistingReverseSession(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("rotated pin dispatched HTTP")
		w.WriteHeader(http.StatusOK)
	}))
	f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("blocked") }
	replacement, _, _ := genLeaf(t, "peer")
	writePin(t, f.dir, "peer", string(replacement))
	f.mesh.Refresh()
	_, err := f.adapter.Client("peer", noderec.ServiceCluster).Get(f.url)
	var unavailable *ReverseUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("stale reverse pin error = %v, want typed unavailable", err)
	}
}

func TestPeerServiceClientAdmissionChangesDuringHandshake(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("stale admission dispatched HTTP")
		w.WriteHeader(http.StatusOK)
	}))
	reachable := f.direct
	f.direct = func(ctx context.Context, service noderec.ServiceKey) (net.Conn, error) {
		conn, err := reachable(ctx, service)
		f.mu.Lock()
		route := f.routes[service]
		route.Incarnation = "v2"
		f.routes[service] = route
		f.mu.Unlock()
		return conn, err
	}
	_, err := f.adapter.Client("peer", noderec.ServiceCluster).Get(f.url)
	var unavailable *ReverseUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("authority replacement error = %v", err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.reverseCalls) != 0 {
		t.Fatalf("obsolete request adopted replacement incarnation: %v", f.reverseCalls)
	}
}

func TestPeerServiceClientAdmissionPublishedByFinalHandoff(t *testing.T) {
	selfPEM, selfKey, _ := genLeaf(t, "self")
	peerPEM, _, _ := genLeaf(t, "peer")
	mesh, _ := meshDir(t, selfPEM, selfKey, map[string]string{"peer": string(peerPEM)})
	var dispatches, reads atomic.Int32
	var replaced atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		dispatches.Add(1)
		writeServiceResponse(t, w, "stale request")
	}))
	defer server.Close()
	u, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	adapter := NewPeerServiceClient(mesh, PeerServiceClientOptions{
		PeerClientOptions: PeerClientOptions{Timeout: time.Second},
		ResolveRoute: func(string, noderec.ServiceKey) (PeerServiceRoute, bool) {
			route := PeerServiceRoute{"admission-1", u.Host, ReversePlain, 1}
			if replaced.Load() {
				route.Incarnation = "admission-2"
			}
			if reads.Add(1) == 4 {
				replaced.Store(true)
			}
			return route, true
		},
	})
	req, err := http.NewRequest(http.MethodPost, server.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	response, err := adapter.Client("peer", noderec.ServiceNodeInfo).Do(req)
	if response != nil {
		response.Body.Close()
	}
	var unavailable *ReverseUnavailableError
	if !errors.As(err, &unavailable) || unavailable.PeerUUID != "peer" {
		t.Fatalf("published replacement returned %v, want typed unavailable", err)
	}
	if !replaced.Load() || reads.Load() < 5 || dispatches.Load() != 0 {
		t.Fatalf("handoff boundary: replaced=%v reads=%d dispatches=%d, want replacement observed before any dispatch", replaced.Load(), reads.Load(), dispatches.Load())
	}
}

func TestPeerServiceClientAuthorityAtFirstApplicationWrite(t *testing.T) {
	test := func(name string, plain, reverse bool) {
		t.Run(name, func(t *testing.T) {
			var dispatches atomic.Int32
			handler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				dispatches.Add(1)
				writeServiceResponse(t, w, "stale request")
			})
			f := newServiceClientFixture(t, handler)
			service, target := noderec.ServiceCluster, f.url
			if plain {
				server := httptest.NewServer(handler)
				defer server.Close()
				u, err := url.Parse(server.URL)
				if err != nil {
					t.Fatal(err)
				}
				service, target = noderec.ServiceNodeInfo, server.URL
				f.routes[service] = PeerServiceRoute{"admission-1", u.Host, ReversePlain, 1}
				dial := func(ctx context.Context, _ noderec.ServiceKey) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "tcp", server.Listener.Addr().String())
				}
				f.direct, f.reverse = dial, dial
			}
			if reverse {
				f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("direct blocked") }
			}
			var replaced, handedOff atomic.Bool
			resolve := f.adapter.opts.ResolveRoute
			f.adapter.opts.ResolveRoute = func(uuid string, service noderec.ServiceKey) (PeerServiceRoute, bool) {
				route, ok := resolve(uuid, service)
				if replaced.Load() {
					route.Incarnation = "admission-2"
				}
				return route, ok
			}
			trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) {
				if !plain {
					secure, ok := info.Conn.(interface{ ConnectionState() tls.ConnectionState })
					if !ok || !secure.ConnectionState().HandshakeComplete {
						t.Error("application boundary was reached before TLS completed")
					}
				}
				handedOff.Store(true)
				replaced.Store(true)
			}}
			req, err := http.NewRequestWithContext(httptrace.WithClientTrace(context.Background(), trace), http.MethodPost, target, nil)
			if err != nil {
				t.Fatal(err)
			}
			response, err := f.adapter.Client("peer", service).Do(req)
			if response != nil {
				response.Body.Close()
			}
			var unavailable *ReverseUnavailableError
			if !errors.As(err, &unavailable) {
				t.Fatalf("first-write replacement returned %v, want typed unavailable", err)
			}
			if !handedOff.Load() || dispatches.Load() != 0 {
				t.Fatalf("first application write: handedOff=%v dispatches=%d, want ready connection and zero dispatches", handedOff.Load(), dispatches.Load())
			}
			f.mu.Lock()
			defer f.mu.Unlock()
			if reverse && f.reverseCalls[service] != 1 {
				t.Fatalf("reverse opens=%v, want one before first-write rejection", f.reverseCalls)
			}
			if !reverse && len(f.reverseCalls) != 0 {
				t.Fatalf("first-write rejection replayed through reverse: %v", f.reverseCalls)
			}
		})
	}
	test("plain direct", true, false)
	test("plain reverse", true, true)
	test("TLS direct", false, false)
	test("TLS reverse", false, true)
}

func TestPeerServiceClientCancellationPropagates(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("canceled request dispatched")
		w.WriteHeader(http.StatusOK)
	}))
	f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("blocked") }
	started := make(chan struct{})
	stopped := make(chan struct{})
	f.reverse = func(ctx context.Context, _ noderec.ServiceKey) (net.Conn, error) {
		close(started)
		<-ctx.Done()
		close(stopped)
		return nil, ctx.Err()
	}
	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, f.url, nil)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := f.adapter.Client("peer", noderec.ServiceCluster).Do(req); done <- err }()
	<-started
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel error = %v", err)
	}
	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Fatal("reverse dial did not observe request cancellation")
	}
}

func TestPeerServiceClientCanceledBeforeDial(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("canceled request dispatched")
		w.WriteHeader(http.StatusOK)
	}))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, f.url, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.adapter.Client("peer", noderec.ServiceCluster).Do(req)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("already canceled error = %v", err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.directCalls) != 0 || len(f.reverseCalls) != 0 {
		t.Fatalf("already canceled call dialed: direct=%v reverse=%v", f.directCalls, f.reverseCalls)
	}
}

func TestPeerServiceClientReusableCancellationDuringDirectSetupSkipsReverse(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("canceled request dispatched")
		w.WriteHeader(http.StatusOK)
	}))
	f.adapter.opts.ReuseConnections = true
	ctx, cancel := context.WithCancel(context.Background())
	f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) {
		cancel()
		return nil, errors.New("direct setup failed")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, f.url, strings.NewReader("snapshot"))
	if err != nil {
		t.Fatal(err)
	}
	response, err := f.adapter.Client("peer", noderec.ServiceCluster).Do(req)
	if response != nil {
		response.Body.Close()
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled setup error = %v, want context.Canceled", err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if direct, reverse := f.directCalls[noderec.ServiceCluster], f.reverseCalls[noderec.ServiceCluster]; direct != 1 || reverse != 0 {
		t.Fatalf("canceled setup routes: direct=%d reverse=%d, want 1/0", direct, reverse)
	}
}

func TestPeerServiceClientBothPathsUnavailable(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("unavailable route dispatched")
		w.WriteHeader(http.StatusOK)
	}))
	directErr, reverseErr := errors.New("direct failed"), errors.New("reverse failed")
	f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, directErr }
	f.reverse = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, reverseErr }
	_, err := f.adapter.Client("peer", noderec.ServiceCluster).Get(f.url)
	var unavailable *ReverseUnavailableError
	if !errors.As(err, &unavailable) || unavailable.PeerUUID != "peer" || !errors.Is(err, directErr) || !errors.Is(err, reverseErr) {
		t.Fatalf("both-path result = %v", err)
	}
}

type serviceRequestBody struct {
	io.Reader
	closed atomic.Int32
}

func (b *serviceRequestBody) Close() error { b.closed.Add(1); return nil }

func TestPeerServiceClientSetupFailureClosesRequestBody(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("setup failure dispatched request")
		w.WriteHeader(http.StatusOK)
	}))
	f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("direct failed") }
	f.reverse = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("reverse failed") }
	body := &serviceRequestBody{Reader: strings.NewReader("mutation")}
	req, err := http.NewRequest(http.MethodPost, f.url, body)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.adapter.Client("peer", noderec.ServiceCluster).Do(req); err == nil {
		t.Fatal("failed setup returned success")
	}
	if got := body.closed.Load(); got != 1 {
		t.Fatalf("failed setup closed request body %d times, want once", got)
	}
}

func TestPeerServiceClientMutationsNeverReplay(t *testing.T) {
	test := func(name, header string, body io.Reader, expectedBody string) {
		t.Run(name, func(t *testing.T) {
			var mutations atomic.Int32
			f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodGet {
					writeServiceResponse(t, w, "warm")
					return
				}
				mutations.Add(1)
				if header != "" && r.Header.Get(header) != "keep-this-value" {
					t.Errorf("lost idempotency header %s", header)
				}
				gotBody, err := io.ReadAll(r.Body)
				if err != nil || string(gotBody) != expectedBody {
					t.Errorf("mutation body = %q, %v; want %q", gotBody, err, expectedBody)
				}
				conn, _, err := w.(http.Hijacker).Hijack()
				if err != nil {
					t.Error(err)
					return
				}
				conn.Close()
			}))
			client := f.adapter.Client("peer", noderec.ServiceCluster)
			serviceClientGet(t, client, f.url)
			req, err := http.NewRequest(http.MethodPost, f.url+"/mutation", body)
			if err != nil {
				t.Fatal(err)
			}
			if header != "" {
				req.Header.Set(header, "keep-this-value")
			}
			_, err = client.Do(req)
			if err == nil {
				t.Fatal("mutation reset returned success")
			}
			if got := mutations.Load(); got != 1 {
				t.Fatalf("mutation dispatched %d times, want exactly one", got)
			}
			f.assertDispatchConnections(t, 0, 1)

			f.mu.Lock()
			replayed := len(f.reverseCalls) != 0
			f.mu.Unlock()
			if replayed {
				t.Fatal("dispatched mutation fell back")
			}
			serviceClientGet(t, client, f.url)
			f.mu.Lock()
			defer f.mu.Unlock()
			if f.reverseCalls[noderec.ServiceCluster] != 1 {
				t.Fatalf("established failure did not switch the next request: %v", f.reverseCalls)
			}
		})
	}
	test("body with GetBody", "", strings.NewReader("mutation"), "mutation")
	test("bodyless Idempotency-Key", "Idempotency-Key", nil, "")
	test("bodyless X-Idempotency-Key", "X-Idempotency-Key", nil, "")
	// GetBody is nil for this reader, but the keyed body still goes unchanged.
	test("keyed streaming body", "Idempotency-Key", io.LimitReader(strings.NewReader("mutation"), 8), "mutation")
}

func TestPeerServiceClientReverseMutationNeverReplays(t *testing.T) {
	var mutations atomic.Int32
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			writeServiceResponse(t, w, "warm")
			return
		}
		mutations.Add(1)
		body, err := io.ReadAll(r.Body)
		if err != nil || string(body) != "mutation" || r.Header.Get("Idempotency-Key") != "preserved" {
			t.Errorf("mutation body/header changed: %q %v", body, err)
		}
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		conn.Close()
	}))
	f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("direct blocked") }
	client := f.adapter.Client("peer", noderec.ServiceCluster)
	serviceClientGet(t, client, f.url)
	req, err := http.NewRequest(http.MethodPost, f.url, strings.NewReader("mutation"))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Idempotency-Key", "preserved")
	if _, err := client.Do(req); err == nil {
		t.Fatal("reverse mutation reset succeeded")
	}
	f.assertDispatchConnections(t, 0, 1)
	if mutations.Load() != 1 {
		t.Fatalf("reverse mutation dispatches=%d, want one dispatch", mutations.Load())
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.reverseCalls[noderec.ServiceCluster] != 2 {
		t.Fatalf("reverse attempted replay: calls=%v", f.reverseCalls)
	}
}

func TestPeerServiceClientPartialResponseNeverReplays(t *testing.T) {
	var dispatched atomic.Int32
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		dispatched.Add(1)
		w.Header().Set("Content-Length", "100")
		writeServiceResponse(t, w, "partial")
		w.(http.Flusher).Flush()
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		conn.Close()
	}))
	response, err := f.adapter.Client("peer", noderec.ServiceCluster).Get(f.url)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if string(body) != "partial" || !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("partial response = %q, %v", body, err)
	}
	if got := dispatched.Load(); got != 1 {
		t.Fatalf("partial response dispatched %d times", got)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.reverseCalls) != 0 {
		t.Fatalf("partial response replayed through reverse: %v", f.reverseCalls)
	}
}

func TestPeerServiceClientMutationRedirectIsExposed(t *testing.T) {
	var dispatched atomic.Int32
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		dispatched.Add(1)
		http.Redirect(w, r, "/again", http.StatusTemporaryRedirect)
	}))
	response, err := f.adapter.Client("peer", noderec.ServiceCluster).Post(f.url, "text/plain", strings.NewReader("mutation"))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusTemporaryRedirect || dispatched.Load() != 1 {
		t.Fatalf("redirect status=%d dispatches=%d", response.StatusCode, dispatched.Load())
	}
}

type serviceFileBody struct {
	*os.File
	closed atomic.Int32
}

func (b *serviceFileBody) Close() error { b.closed.Add(1); return b.File.Close() }

func TestPeerServiceClientRejectedRedirectClosesRecreatedBody(t *testing.T) {
	test := func(name string, status int) {
		t.Run(name, func(t *testing.T) {
			var dispatches atomic.Int32
			f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				dispatches.Add(1)
				body, err := io.ReadAll(r.Body)
				if err != nil || string(body) != "mutation" {
					t.Errorf("original mutation body=%q error=%v", body, err)
				}
				w.Header().Set("Location", "/again")
				w.WriteHeader(status)
				writeServiceResponse(t, w, "original redirect response")
			}))
			file, err := os.CreateTemp(t.TempDir(), "redirect-body-")
			if err != nil {
				t.Fatal(err)
			}
			original := &serviceFileBody{File: file}
			t.Cleanup(func() {
				if original.closed.Load() == 0 {
					original.Close()
				}
			})
			if _, err := io.WriteString(file, "mutation"); err != nil {
				t.Fatal(err)
			}
			if _, err := file.Seek(0, io.SeekStart); err != nil {
				t.Fatal(err)
			}
			req, err := http.NewRequest(http.MethodPost, f.url, original)
			if err != nil {
				t.Fatal(err)
			}
			req.ContentLength = 8
			var reopened atomic.Int32
			var recreated *serviceFileBody
			req.GetBody = func() (io.ReadCloser, error) {
				reopened.Add(1)
				file, err := os.Open(original.Name())
				if err != nil {
					return nil, err
				}
				recreated = &serviceFileBody{File: file}
				return recreated, nil
			}
			t.Cleanup(func() {
				if recreated != nil && recreated.closed.Load() == 0 {
					recreated.Close()
				}
			})
			response, err := f.adapter.Client("peer", noderec.ServiceCluster).Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			body, err := io.ReadAll(response.Body)
			if err != nil || string(body) != "original redirect response" || response.StatusCode != status {
				t.Fatalf("exposed redirect: status=%d body=%q err=%v", response.StatusCode, body, err)
			}
			if response.TLS == nil || !response.TLS.HandshakeComplete {
				t.Fatal("post-handshake wrapper lost response TLS state")
			}
			if reopened.Load() != 1 || recreated == nil || recreated.closed.Load() != 1 || original.closed.Load() < 1 || dispatches.Load() != 1 {
				t.Fatalf("redirect ownership: reopen=%d recreated=%v original closes=%d dispatches=%d", reopened.Load(), recreated, original.closed.Load(), dispatches.Load())
			}
			// Read checks os.File's closed state on every platform; Windows
			// Stat instead exposes the native invalid-handle error.
			var probe [1]byte
			if _, err := recreated.Read(probe[:]); !errors.Is(err, os.ErrClosed) {
				t.Fatalf("recreated redirect file remains open: %v", err)
			}
		})
	}
	test("307 temporary redirect", http.StatusTemporaryRedirect)
	test("308 permanent redirect", http.StatusPermanentRedirect)
}

func TestPeerServiceClientFailingGetBodyPreservesNativeError(t *testing.T) {
	var dispatches atomic.Int32
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		dispatches.Add(1)
		if r.Method == http.MethodGet {
			writeServiceResponse(t, w, "next request")
			return
		}
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Error(err)
		}
		w.Header().Set("Location", "/again")
		w.WriteHeader(http.StatusTemporaryRedirect)
		writeServiceResponse(t, w, "redirect response")
	}))
	client := f.adapter.Client("peer", noderec.ServiceCluster)
	reopenErr := errors.New("cannot recreate resource-owning request body")
	var reopened atomic.Int32
	for range peerMaxConnsPerHost {
		req, err := http.NewRequest(http.MethodPost, f.url, strings.NewReader("mutation"))
		if err != nil {
			t.Fatal(err)
		}
		req.GetBody = func() (io.ReadCloser, error) { reopened.Add(1); return nil, reopenErr }
		response, err := client.Do(req)
		if response != nil {
			response.Body.Close()
		}
		if !errors.Is(err, reopenErr) || response != nil {
			t.Fatalf("failing GetBody: response=%v err=%v, want native factory error without a response", response, err)
		}
	}
	if got := serviceClientGet(t, client, f.url); got != "next request" {
		t.Fatalf("redirect error lost slot ownership: body=%q", got)
	}
	if reopened.Load() != int32(peerMaxConnsPerHost) || dispatches.Load() != int32(peerMaxConnsPerHost+1) {
		t.Fatalf("redirect error attempted replay: factories=%d dispatches=%d", reopened.Load(), dispatches.Load())
	}
}

type serviceFailingReader struct{ err error }

func (r serviceFailingReader) Read([]byte) (int, error) { return 0, r.err }

func TestPeerServiceClientResponseBodyTerminalPathsReleaseOnce(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	test := func(name string, ctx context.Context, reader io.Reader, read bool, wantErr error) {
		t.Run(name, func(t *testing.T) {
			var released atomic.Int32
			body := &peerServiceBody{ReadCloser: io.NopCloser(reader), ctx: ctx, failed: func() {}, finish: func() { released.Add(1) }}
			if read {
				_, err := io.ReadAll(body)
				if !errors.Is(err, wantErr) {
					t.Fatalf("body read error=%v, want %v", err, wantErr)
				}
			}
			if err := body.Close(); err != nil {
				t.Fatal(err)
			}
			if err := body.Close(); err != nil {
				t.Fatal(err)
			}
			if got := released.Load(); got != 1 {
				t.Fatalf("terminal body released ownership %d times, want once", got)
			}
		})
	}
	test("EOF then Close", context.Background(), strings.NewReader("complete"), true, nil)
	test("read error then Close", context.Background(), serviceFailingReader{io.ErrUnexpectedEOF}, true, io.ErrUnexpectedEOF)
	test("canceled socket read then Close", canceled, serviceFailingReader{net.ErrClosed}, true, context.Canceled)
	test("unread body Close", context.Background(), strings.NewReader("unread"), false, nil)
}

func TestPeerServiceClientPlainNodeInfoFallback(t *testing.T) {
	test := func(name string, partial bool) {
		t.Run(name, func(t *testing.T) {
			selfPEM, selfKey, _ := genLeaf(t, "self")
			peerPEM, _, _ := genLeaf(t, "peer")
			mesh, _ := meshDir(t, selfPEM, selfKey, map[string]string{"peer": string(peerPEM)})
			var directDispatch, reverseDispatch atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				directDispatch.Add(1)
				if partial {
					w.Header().Set("Content-Length", "100")
					writeServiceResponse(t, w, "partial")
					w.(http.Flusher).Flush()
				}
				conn, _, err := w.(http.Hijacker).Hijack()
				if err != nil {
					t.Error(err)
					return
				}
				conn.Close()
			}))
			defer server.Close()
			reverse := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				reverseDispatch.Add(1)
				writeServiceResponse(t, w, "node-info")
			}))
			defer reverse.Close()
			u, err := url.Parse(server.URL)
			if err != nil {
				t.Fatal(err)
			}
			adapter := NewPeerServiceClient(mesh, PeerServiceClientOptions{PeerClientOptions: PeerClientOptions{Timeout: time.Second}, ResolveRoute: func(string, noderec.ServiceKey) (PeerServiceRoute, bool) {
				return PeerServiceRoute{"v1", u.Host, ReversePlain, 1}, true
			}, DialReverse: func(ctx context.Context, _, _ string, service noderec.ServiceKey) (net.Conn, error) {
				if service != noderec.ServiceNodeInfo {
					return nil, errors.New("wrong service")
				}
				return (&net.Dialer{}).DialContext(ctx, "tcp", reverse.Listener.Addr().String())
			}})
			response, err := adapter.Client("peer", noderec.ServiceNodeInfo).Get(server.URL + "/v1/node-info")
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			body, err := io.ReadAll(response.Body)
			if partial {
				if string(body) != "partial" || !errors.Is(err, io.ErrUnexpectedEOF) || reverseDispatch.Load() != 0 {
					t.Fatalf("partial GET replay: body=%q err=%v reverse=%d", body, err, reverseDispatch.Load())
				}
			} else if string(body) != "node-info" || err != nil || reverseDispatch.Load() != 1 {
				t.Fatalf("GET fallback: body=%q err=%v reverse=%d", body, err, reverseDispatch.Load())
			}
			if directDispatch.Load() != 1 {
				t.Fatalf("direct GET dispatched %d times", directDispatch.Load())
			}
		})
	}
	test("GET fails before response", false)
	test("partial GET response stays exposed", true)
}

func TestPeerServiceClientPlainReverseUsesRemainingHeaderBudget(t *testing.T) {
	selfPEM, selfKey, _ := genLeaf(t, "self")
	peerPEM, _, _ := genLeaf(t, "peer")
	mesh, _ := meshDir(t, selfPEM, selfKey, map[string]string{"peer": string(peerPEM)})
	reverse := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		timer := time.NewTimer(70 * time.Millisecond)
		defer timer.Stop()
		select {
		case <-timer.C:
			writeServiceResponse(t, w, "slow node-info")
		case <-r.Context().Done():
		}
	}))
	defer reverse.Close()
	adapter := NewPeerServiceClient(mesh, PeerServiceClientOptions{
		PeerClientOptions: PeerClientOptions{Timeout: time.Second}, DirectTimeout: 20 * time.Millisecond,
		ResolveRoute: func(string, noderec.ServiceKey) (PeerServiceRoute, bool) {
			return PeerServiceRoute{"v1", "original.example:12345", ReversePlain, 1}, true
		},
		DialContext: func(context.Context, string, string) (net.Conn, error) { return nil, errors.New("direct unavailable") },
		DialReverse: func(ctx context.Context, _, _ string, service noderec.ServiceKey) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp", reverse.Listener.Addr().String())
		},
	})
	if got := serviceClientGet(t, adapter.Client("peer", noderec.ServiceNodeInfo), "http://original.example:12345/v1/node-info"); got != "slow node-info" {
		t.Fatalf("reverse body = %q", got)
	}
}

func TestPeerServiceClientNegativeHeaderTimeoutStillFallsBack(t *testing.T) {
	selfPEM, selfKey, _ := genLeaf(t, "self")
	peerPEM, _, _ := genLeaf(t, "peer")
	mesh, _ := meshDir(t, selfPEM, selfKey, map[string]string{"peer": string(peerPEM)})
	var directDispatch, reverseDispatch atomic.Int32
	direct := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { directDispatch.Add(1); <-r.Context().Done() }))
	defer direct.Close()
	reverse := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reverseDispatch.Add(1)
		writeServiceResponse(t, w, "reverse node-info")
	}))
	defer reverse.Close()
	u, err := url.Parse(direct.URL)
	if err != nil {
		t.Fatal(err)
	}
	adapter := NewPeerServiceClient(mesh, PeerServiceClientOptions{
		PeerClientOptions: PeerClientOptions{Timeout: 250 * time.Millisecond, ResponseHeaderTimeout: -time.Second}, DirectTimeout: 30 * time.Millisecond,
		ResolveRoute: func(string, noderec.ServiceKey) (PeerServiceRoute, bool) {
			return PeerServiceRoute{"admission-1", u.Host, ReversePlain, 1}, true
		},
		DialReverse: func(ctx context.Context, _, _ string, service noderec.ServiceKey) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp", reverse.Listener.Addr().String())
		},
	})
	start := time.Now()
	if got := serviceClientGet(t, adapter.Client("peer", noderec.ServiceNodeInfo), direct.URL); got != "reverse node-info" {
		t.Fatalf("fallback body=%q", got)
	}
	if directDispatch.Load() != 1 || reverseDispatch.Load() != 1 || time.Since(start) >= 250*time.Millisecond {
		t.Fatalf("negative header timeout: direct=%d reverse=%d elapsed=%v", directDispatch.Load(), reverseDispatch.Load(), time.Since(start))
	}
}

func serviceClientPost(t *testing.T, client *http.Client, target string) {
	t.Helper()
	response, err := client.Post(target, "text/plain", strings.NewReader("mutation"))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil || response.StatusCode != http.StatusOK || string(body) != "ok" {
		t.Fatalf("POST response: status=%d body=%q err=%v, want 200 ok", response.StatusCode, body, err)
	}
	if response.TLS == nil || !response.TLS.HandshakeComplete {
		t.Fatal("pooled response lost TLS metadata")
	}
}

func TestPeerServiceClientReusesEligiblePosts(t *testing.T) {
	test := func(name string, reverse bool) {
		t.Run(name, func(t *testing.T) {
			var dispatches atomic.Int32
			f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				dispatches.Add(1)
				body, err := io.ReadAll(r.Body)
				if err != nil || string(body) != "mutation" || r.Method != http.MethodPost || r.Header.Get("Idempotency-Key") != "preserved" {
					t.Errorf("POST changed: method=%s body=%q header=%q err=%v", r.Method, body, r.Header.Get("Idempotency-Key"), err)
				}
				writeServiceResponse(t, w, "ok")
			}))
			f.adapter.opts.ReuseConnections = true
			if reverse {
				f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("blocked") }
			}
			client := f.adapter.Client("peer", noderec.ServiceCluster)
			var reused bool
			for i := range 2 {
				ctx, cancel := context.WithCancel(context.Background())
				trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) { reused = info.Reused }}
				req, err := http.NewRequestWithContext(httptrace.WithClientTrace(ctx, trace), http.MethodPost, f.url, strings.NewReader("mutation"))
				if err != nil {
					cancel()
					t.Fatal(err)
				}
				req.Header.Set("Idempotency-Key", "preserved")
				response, err := client.Do(req)
				if err != nil {
					cancel()
					t.Fatal(err)
				}
				body, readErr := io.ReadAll(response.Body)
				closeErr := response.Body.Close()
				cancel() // The completed request must not own the idle reverse stream.
				if readErr != nil || closeErr != nil || string(body) != "ok" || response.StatusCode != http.StatusOK {
					t.Fatalf("POST response: body=%q read=%v close=%v status=%d", body, readErr, closeErr, response.StatusCode)
				}
				if response.TLS == nil || !response.TLS.HandshakeComplete || reused != (i == 1) {
					t.Fatalf("request %d: reused=%v TLS=%v", i, reused, response.TLS)
				}
			}
			f.assertDispatchConnections(t, 0, 0)
			if dispatches.Load() != 2 {
				t.Fatalf("dispatches=%d, want 2 dispatches over one connection", dispatches.Load())
			}
			f.mu.Lock()
			defer f.mu.Unlock()
			wantReverse := 0
			if reverse {
				wantReverse = 1
			}
			if f.directCalls[noderec.ServiceCluster] != 1 || f.reverseCalls[noderec.ServiceCluster] != wantReverse {
				t.Fatalf("dial counts: direct=%v reverse=%v", f.directCalls, f.reverseCalls)
			}
		})
	}
	test("direct", false)
	test("reverse", true)
}

func TestPeerServiceClientIneligibleMutationsRemainFresh(t *testing.T) {
	test := func(name, method, header string, makeBody func() io.Reader) {
		t.Run(name, func(t *testing.T) {
			var dispatches atomic.Int32
			f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				dispatches.Add(1)
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Error(err)
				}
				writeServiceResponse(t, w, "ok")
			}))
			f.adapter.opts.ReuseConnections = true
			client := f.adapter.Client("peer", noderec.ServiceCluster)
			for range 2 {
				req, err := http.NewRequest(method, f.url, makeBody())
				if err != nil {
					t.Fatal(err)
				}
				if header != "" {
					req.Header.Set(header, "preserved")
				}
				response, err := client.Do(req)
				if err != nil {
					t.Fatal(err)
				}
				_, readErr := io.Copy(io.Discard, response.Body)
				closeErr := response.Body.Close()
				if readErr != nil || closeErr != nil || response.StatusCode != http.StatusOK {
					t.Fatalf("fresh response: status=%d read=%v close=%v", response.StatusCode, readErr, closeErr)
				}
			}
			f.assertDispatchConnections(t, 0, 1)
			if dispatches.Load() != 2 {
				t.Fatalf("dispatches=%d, want 2", dispatches.Load())
			}
		})
	}
	test("bodyless keyed POST", http.MethodPost, "Idempotency-Key", func() io.Reader { return nil })
	test("NoBody keyed POST", http.MethodPost, "X-Idempotency-Key", func() io.Reader { return http.NoBody })
	test("unknown length POST", http.MethodPost, "Idempotency-Key", func() io.Reader { return io.LimitReader(strings.NewReader("mutation"), 8) })
	test("recreatable PUT", http.MethodPut, "", func() io.Reader { return strings.NewReader("mutation") })
}

type serviceStaleConn struct {
	net.Conn
	fail atomic.Bool
}

func (c *serviceStaleConn) Write(data []byte) (int, error) {
	if c.fail.Load() {
		return 0, errors.New("injected stale connection write failure")
	}
	return c.Conn.Write(data)
}

func TestPeerServiceClientStaleReuseNeverReplays(t *testing.T) {
	var dispatches atomic.Int32
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		dispatches.Add(1)
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Error(err)
		}
		writeServiceResponse(t, w, "ok")
	}))
	f.adapter.opts.ReuseConnections = true
	reachable := f.direct
	var stale *serviceStaleConn
	f.direct = func(ctx context.Context, service noderec.ServiceKey) (net.Conn, error) {
		conn, err := reachable(ctx, service)
		if err != nil {
			return nil, err
		}
		stale = &serviceStaleConn{Conn: conn}
		return stale, nil
	}
	client := f.adapter.Client("peer", noderec.ServiceCluster)
	serviceClientPost(t, client, f.url)
	stale.fail.Store(true)
	var reused atomic.Bool
	trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) { reused.Store(info.Reused) }}
	req, err := http.NewRequestWithContext(httptrace.WithClientTrace(context.Background(), trace), http.MethodPost, f.url, strings.NewReader("mutation"))
	if err != nil {
		t.Fatal(err)
	}
	var recreated atomic.Int32
	req.GetBody = func() (io.ReadCloser, error) {
		recreated.Add(1)
		return io.NopCloser(strings.NewReader("mutation")), nil
	}
	req.Header.Set("Idempotency-Key", "preserved")
	if response, err := client.Do(req); err == nil {
		response.Body.Close()
		t.Fatal("stale connection succeeded")
	}
	f.assertDispatchConnections(t, 0)
	if !reused.Load() || recreated.Load() != 0 || dispatches.Load() != 1 {
		t.Fatalf("stale request: reused=%v factories=%d dispatches=%d", reused.Load(), recreated.Load(), dispatches.Load())
	}
	f.mu.Lock()
	direct, reverse := f.directCalls[noderec.ServiceCluster], f.reverseCalls[noderec.ServiceCluster]
	f.mu.Unlock()
	if direct != 1 || reverse != 0 {
		t.Fatalf("stale request redialed: direct=%d reverse=%d", direct, reverse)
	}
	serviceClientPost(t, client, f.url)
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.reverseCalls[noderec.ServiceCluster] != 1 || dispatches.Load() != 2 {
		t.Fatalf("next caller did not own fallback: reverse=%v dispatches=%d", f.reverseCalls, dispatches.Load())
	}
}

func TestPeerServiceClientReusableLostReplyNeverReplays(t *testing.T) {
	test := func(name string, reverse bool) {
		t.Run(name, func(t *testing.T) {
			var accepted atomic.Int32
			f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil || string(body) != "mutation" {
					t.Errorf("receiver body=%q err=%v", body, err)
				}
				if accepted.Add(1) != 2 {
					writeServiceResponse(t, w, "ok")
					return
				}
				conn, _, err := w.(http.Hijacker).Hijack()
				if err != nil {
					t.Error(err)
					return
				}
				if err := conn.Close(); err != nil {
					t.Error(err)
				}
			}))
			f.adapter.opts.ReuseConnections = true
			if reverse {
				f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("blocked") }
			}
			client := f.adapter.Client("peer", noderec.ServiceCluster)
			serviceClientPost(t, client, f.url)
			req, err := http.NewRequest(http.MethodPost, f.url, strings.NewReader("mutation"))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("X-Idempotency-Key", "preserved")
			var recreated atomic.Int32
			req.GetBody = func() (io.ReadCloser, error) {
				recreated.Add(1)
				return io.NopCloser(strings.NewReader("mutation")), nil
			}
			if response, err := client.Do(req); err == nil {
				response.Body.Close()
				t.Fatal("lost reply returned success")
			}
			f.assertDispatchConnections(t, 0, 0)
			if accepted.Load() != 2 || recreated.Load() != 0 {
				t.Fatalf("lost reply: accepted=%d factories=%d, want no replay", accepted.Load(), recreated.Load())
			}
			serviceClientPost(t, client, f.url)
			f.assertDispatchConnections(t, 0, 0, 1)
			if accepted.Load() != 3 {
				t.Fatalf("next caller: accepted=%d, want replacement setup", accepted.Load())
			}
		})
	}
	test("direct", false)
	test("reverse", true)
}

func TestPeerServiceClientReusableAuthorityRotation(t *testing.T) {
	test := func(name string, reverse bool, rotate func(*testing.T, *serviceClientFixture) string) {
		t.Run(name, func(t *testing.T) {
			var dispatches atomic.Int32
			f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				dispatches.Add(1)
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Error(err)
				}
				writeServiceResponse(t, w, "ok")
			}))
			f.adapter.opts.ReuseConnections = true
			if reverse {
				f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("blocked") }
			}
			client := f.adapter.Client("peer", noderec.ServiceCluster)
			serviceClientPost(t, client, f.url)
			f.adapter.mu.Lock()
			old := f.adapter.entries[peerServiceKey{"peer", noderec.ServiceCluster}]
			if len(old.idle) != 1 {
				f.adapter.mu.Unlock()
				t.Fatal("first request did not create an idle pooled connection")
			}
			f.adapter.mu.Unlock()
			target := rotate(t, f)
			serviceClientPost(t, client, target)
			f.adapter.mu.Lock()
			retired, idle := old.retired, len(old.idle)
			f.adapter.mu.Unlock()
			f.assertDispatchConnections(t, 0, 1)
			if !retired || idle != 0 || dispatches.Load() != 2 {
				t.Fatalf("rotation: retired=%v old idle=%d dispatches=%d, want new connection", retired, idle, dispatches.Load())
			}
		})
	}
	incarnation := func(t *testing.T, f *serviceClientFixture) string {
		t.Helper()
		f.mu.Lock()
		route := f.routes[noderec.ServiceCluster]
		route.Incarnation = "v2"
		f.routes[noderec.ServiceCluster] = route
		f.mu.Unlock()
		return f.url
	}
	generation := func(t *testing.T, f *serviceClientFixture) string {
		t.Helper()
		f.mu.Lock()
		route := f.routes[noderec.ServiceCluster]
		route.Generation++
		f.routes[noderec.ServiceCluster] = route
		f.mu.Unlock()
		return f.url
	}
	address := func(t *testing.T, f *serviceClientFixture) string {
		t.Helper()
		f.mu.Lock()
		route := f.routes[noderec.ServiceCluster]
		route.Address = "replacement.example:12345"
		f.routes[noderec.ServiceCluster] = route
		f.mu.Unlock()
		return "https://replacement.example:12345"
	}
	peer := func(t *testing.T, f *serviceClientFixture) string {
		t.Helper()
		cert, key, _ := genLeaf(t, "peer")
		writeIdentity(t, f.peerDir, cert, key)
		f.peer.Refresh()
		writePin(t, f.dir, "peer", string(cert))
		f.mesh.Refresh()
		return f.url
	}
	self := func(t *testing.T, f *serviceClientFixture) string {
		t.Helper()
		cert, key, _ := genLeaf(t, "self")
		writeIdentity(t, f.dir, cert, key)
		f.mesh.Refresh()
		writePin(t, f.peerDir, "self", string(cert))
		f.peer.Refresh()
		return f.url
	}
	test("direct readmission", false, incarnation)
	test("reverse readmission", true, incarnation)
	test("direct generation", false, generation)
	test("reverse generation", true, generation)
	test("direct address", false, address)
	test("reverse address", true, address)
	test("direct peer certificate", false, peer)
	test("reverse peer certificate", true, peer)
	test("direct local certificate", false, self)
	test("reverse local certificate", true, self)
}

func TestPeerServiceClientReusableRevocationRetiresIdle(t *testing.T) {
	test := func(name string, revoke func(*serviceClientFixture)) {
		t.Run(name, func(t *testing.T) {
			var dispatches atomic.Int32
			f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				dispatches.Add(1)
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Error(err)
				}
				writeServiceResponse(t, w, "ok")
			}))
			f.adapter.opts.ReuseConnections = true
			client := f.adapter.Client("peer", noderec.ServiceCluster)
			serviceClientPost(t, client, f.url)
			revoke(f)
			if response, err := client.Post(f.url, "text/plain", strings.NewReader("mutation")); err == nil {
				response.Body.Close()
				t.Fatal("revoked authority dispatched")
			}
			f.adapter.mu.Lock()
			h := f.adapter.entries[peerServiceKey{"peer", noderec.ServiceCluster}]
			retired, idle := h.retired, len(h.idle)
			f.adapter.mu.Unlock()
			f.assertDispatchConnections(t, 0)
			if !retired || idle != 0 || dispatches.Load() != 1 {
				t.Fatalf("revocation: retired=%v idle=%d dispatches=%d", retired, idle, dispatches.Load())
			}
		})
	}
	test("admission revoked", func(f *serviceClientFixture) {
		f.mu.Lock()
		delete(f.routes, noderec.ServiceCluster)
		f.mu.Unlock()
	})
	test("protocol no longer authorized", func(f *serviceClientFixture) {
		f.mu.Lock()
		route := f.routes[noderec.ServiceCluster]
		route.Protocol = ReversePlain
		f.routes[noderec.ServiceCluster] = route
		f.mu.Unlock()
	})
	test("pin revoked despite disk pin", func(f *serviceClientFixture) {
		f.adapter.opts.ResolvePin = func(string) ([]byte, bool) { return nil, false }
	})
}

func TestPeerServiceClientReusableAuthorityAtDispatch(t *testing.T) {
	test := func(name string, reverse bool) {
		t.Run(name, func(t *testing.T) {
			var dispatches atomic.Int32
			f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				dispatches.Add(1)
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Error(err)
				}
				writeServiceResponse(t, w, "ok")
			}))
			f.adapter.opts.ReuseConnections = true
			if reverse {
				f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("blocked") }
			}
			client := f.adapter.Client("peer", noderec.ServiceCluster)
			serviceClientPost(t, client, f.url)
			var reused atomic.Bool
			trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) {
				reused.Store(info.Reused)
				f.mu.Lock()
				route := f.routes[noderec.ServiceCluster]
				route.Generation++
				f.routes[noderec.ServiceCluster] = route
				f.mu.Unlock()
			}}
			req, err := http.NewRequestWithContext(httptrace.WithClientTrace(context.Background(), trace), http.MethodPost, f.url, strings.NewReader("mutation"))
			if err != nil {
				t.Fatal(err)
			}
			response, err := client.Do(req)
			if response != nil {
				response.Body.Close()
			}
			var unavailable *ReverseUnavailableError
			f.assertDispatchConnections(t, 0)
			if !errors.As(err, &unavailable) || !reused.Load() || dispatches.Load() != 1 {
				t.Fatalf("dispatch authority: err=%v reused=%v dispatches=%d", err, reused.Load(), dispatches.Load())
			}
		})
	}
	test("direct", false)
	test("reverse", true)
}

func TestPeerServiceClientReusableRedirectClosesRecreatedBody(t *testing.T) {
	test := func(name string, status int) {
		t.Run(name, func(t *testing.T) {
			var mutations atomic.Int32
			f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Error(err)
				}
				if r.URL.Path == "/warm" {
					writeServiceResponse(t, w, "ok")
					return
				}
				mutations.Add(1)
				w.Header().Set("Location", "/again")
				w.WriteHeader(status)
				writeServiceResponse(t, w, "redirect response")
			}))
			f.adapter.opts.ReuseConnections = true
			client := f.adapter.Client("peer", noderec.ServiceCluster)
			serviceClientPost(t, client, f.url+"/warm")
			original := &serviceRequestBody{Reader: strings.NewReader("mutation")}
			recreated := &serviceRequestBody{Reader: strings.NewReader("mutation")}
			req, err := http.NewRequest(http.MethodPost, f.url, original)
			if err != nil {
				t.Fatal(err)
			}
			req.ContentLength = 8
			var factories atomic.Int32
			req.GetBody = func() (io.ReadCloser, error) { factories.Add(1); return recreated, nil }
			response, err := client.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			body, readErr := io.ReadAll(response.Body)
			closeErr := response.Body.Close()
			if readErr != nil || closeErr != nil || response.StatusCode != status || string(body) != "redirect response" {
				t.Fatalf("redirect response: status=%d body=%q read=%v close=%v", response.StatusCode, body, readErr, closeErr)
			}
			f.assertDispatchConnections(t, 0, 0)
			if factories.Load() != 1 || recreated.closed.Load() != 1 || original.closed.Load() < 1 || mutations.Load() != 1 {
				t.Fatalf("redirect: factories=%d recreated closes=%d original closes=%d mutations=%d", factories.Load(), recreated.closed.Load(), original.closed.Load(), mutations.Load())
			}
		})
	}
	test("307 temporary redirect", http.StatusTemporaryRedirect)
	test("308 permanent redirect", http.StatusPermanentRedirect)
}

func TestPeerServiceClientReusableCancellationPreservesOtherResponse(t *testing.T) {
	release := make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Error(err)
		}
		if r.URL.Path == "/warm" {
			writeServiceResponse(t, w, "ok")
			return
		}
		writeServiceResponse(t, w, "start-")
		w.(http.Flusher).Flush()
		select {
		case <-release:
			writeServiceResponse(t, w, "end")
		case <-r.Context().Done():
			if r.URL.Path == "/keep" {
				t.Error("another caller's cancellation closed the held reverse stream")
			}
		}
	}))
	f.adapter.opts.ReuseConnections = true
	f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("blocked") }
	client := f.adapter.Client("peer", noderec.ServiceCluster)
	serviceClientPost(t, client, f.url+"/warm")
	keep, err := client.Post(f.url+"/keep", "text/plain", strings.NewReader("mutation"))
	if err != nil {
		t.Fatal(err)
	}
	defer keep.Body.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, f.url+"/cancel", strings.NewReader("mutation"))
	if err != nil {
		t.Fatal(err)
	}
	response, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	_, readErr := io.ReadAll(response.Body)
	closeErr := response.Body.Close()
	if !errors.Is(readErr, context.Canceled) || closeErr != nil {
		t.Fatalf("canceled body: read=%v close=%v", readErr, closeErr)
	}
	f.adapter.mu.Lock()
	occupied := len(f.adapter.entries[peerServiceKey{"peer", noderec.ServiceCluster}].slots)
	f.adapter.mu.Unlock()
	if occupied != 1 {
		t.Fatalf("occupied slots after cancellation=%d, want only the held response", occupied)
	}
	serviceClientPost(t, client, f.url+"/warm")
	once.Do(func() { close(release) })
	body, err := io.ReadAll(keep.Body)
	if err != nil || string(body) != "start-end" {
		t.Fatalf("unrelated response: body=%q err=%v", body, err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.directCalls[noderec.ServiceCluster] != 1 || f.reverseCalls[noderec.ServiceCluster] != 3 {
		t.Fatalf("cancellation routes: direct=%v reverse=%v, want only canceled stream replaced", f.directCalls, f.reverseCalls)
	}
}

func TestPeerServiceClientReusableReverseReconnect(t *testing.T) {
	var dispatches atomic.Int32
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		dispatches.Add(1)
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Error(err)
		}
		writeServiceResponse(t, w, "ok")
	}))
	f.adapter.opts.ReuseConnections = true
	f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("blocked") }
	remote := reverseTestManager(t, ReverseOptions{LocalUUID: "peer", ResolveService: func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
		return ReverseTarget{Protocol: ReverseTLS, Dial: func(ctx context.Context) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp", f.server.Listener.Addr().String())
		}}, nil
	}})
	local := reverseTestManager(t, ReverseOptions{LocalUUID: "self"})
	reverseTestConnect(t, local, remote, "self", reverseTestID)
	f.reverse = func(ctx context.Context, service noderec.ServiceKey) (net.Conn, error) {
		return local.Open(ctx, ReversePeer{"peer", "v1"}, service)
	}
	client := f.adapter.Client("peer", noderec.ServiceCluster)
	serviceClientPost(t, client, f.url)
	local.Revoke("peer")
	remote.Revoke("self")
	reverseTestConnect(t, local, remote, "self", reverseTestID)
	// The old stream may be noticed before Write or fail during it. Neither
	// case lets this caller silently open a replacement on the new session.
	response, err := client.Post(f.url, "text/plain", strings.NewReader("mutation"))
	if response != nil {
		response.Body.Close()
	}
	if err == nil || dispatches.Load() != 1 {
		t.Fatalf("old-session request: err=%v dispatches=%d, want no replay", err, dispatches.Load())
	}
	f.mu.Lock()
	opens := f.reverseCalls[noderec.ServiceCluster]
	f.mu.Unlock()
	if opens != 1 {
		t.Fatalf("old-session request opened %d reverse streams, want original only", opens)
	}
	serviceClientPost(t, client, f.url)
	serviceClientPost(t, client, f.url)
	f.assertDispatchConnections(t, 0, 1, 1)
	if dispatches.Load() != 3 {
		t.Fatalf("reconnected session: dispatches=%d, want next caller's pooled replacement", dispatches.Load())
	}
}

func TestPeerServiceClientReusableIdleReaping(t *testing.T) {
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Error(err)
		}
		writeServiceResponse(t, w, "ok")
	}))
	f.adapter.opts.ReuseConnections = true
	client := f.adapter.Client("peer", noderec.ServiceCluster)
	serviceClientPost(t, client, f.url)
	f.adapter.mu.Lock()
	h := f.adapter.entries[peerServiceKey{"peer", noderec.ServiceCluster}]
	if len(h.idle) != 1 || h.idle[0].timer == nil {
		f.adapter.mu.Unlock()
		t.Fatal("pooled connection has no idle reaper")
	}
	// Exercise the production timer callback without waiting the full idle TTL.
	h.idle[0].timer.Reset(time.Millisecond)
	f.adapter.mu.Unlock()
	select {
	case <-f.closed:
	case <-time.After(time.Second):
		t.Fatal("idle reaper did not close the actual socket")
	}
	f.adapter.mu.Lock()
	idle, global := len(h.idle), f.adapter.idle
	f.adapter.mu.Unlock()
	if idle != 0 || global != 0 {
		t.Fatalf("idle reaper retained local=%d global=%d entries", idle, global)
	}
	serviceClientPost(t, client, f.url)
	f.assertDispatchConnections(t, 0, 1)
}

func TestPeerServiceClientReusableProtocolChange(t *testing.T) {
	test := func(name string, reverse bool) {
		t.Run(name, func(t *testing.T) {
			var dispatches atomic.Int32
			handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				dispatches.Add(1)
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Error(err)
				}
				writeServiceResponse(t, w, "ok")
			})
			f := newServiceClientFixture(t, handler)
			f.adapter.opts.ReuseConnections = true
			f.routes[noderec.ServiceNodeInfo] = f.routes[noderec.ServiceCluster]
			plain := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if !f.recordDispatch(t, r) {
					http.Error(w, "missing connection identity", http.StatusInternalServerError)
					return
				}
				handler.ServeHTTP(w, r)
			}))
			plain.Config.ConnContext = f.server.Config.ConnContext
			plain.Start()
			defer plain.Close()
			var usePlain atomic.Bool
			reachable, reverseDial := f.direct, f.reverse
			dialPlain := func(ctx context.Context) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, "tcp", plain.Listener.Addr().String())
			}
			f.direct = func(ctx context.Context, service noderec.ServiceKey) (net.Conn, error) {
				if reverse {
					return nil, errors.New("blocked")
				}
				if usePlain.Load() {
					return dialPlain(ctx)
				}
				return reachable(ctx, service)
			}
			f.reverse = func(ctx context.Context, service noderec.ServiceKey) (net.Conn, error) {
				if usePlain.Load() {
					return dialPlain(ctx)
				}
				return reverseDial(ctx, service)
			}
			client := f.adapter.Client("peer", noderec.ServiceNodeInfo)
			serviceClientPost(t, client, f.url)
			f.adapter.mu.Lock()
			old := f.adapter.entries[peerServiceKey{"peer", noderec.ServiceNodeInfo}]
			f.adapter.mu.Unlock()
			f.mu.Lock()
			route := f.routes[noderec.ServiceNodeInfo]
			route.Protocol = ReversePlain // Address, generation and incarnation stay identical.
			f.routes[noderec.ServiceNodeInfo] = route
			f.mu.Unlock()
			usePlain.Store(true)
			for range 2 {
				response, err := client.Post("http://"+route.Address, "text/plain", strings.NewReader("mutation"))
				if err != nil {
					t.Fatal(err)
				}
				body, readErr := io.ReadAll(response.Body)
				closeErr := response.Body.Close()
				if readErr != nil || closeErr != nil || response.StatusCode != http.StatusOK || string(body) != "ok" || response.TLS != nil {
					t.Fatalf("plain response: status=%d body=%q read=%v close=%v TLS=%v", response.StatusCode, body, readErr, closeErr, response.TLS)
				}
			}
			f.adapter.mu.Lock()
			retired, idle := old.retired, len(old.idle)
			f.adapter.mu.Unlock()
			f.assertDispatchConnections(t, 0, 1, 1)
			if !retired || idle != 0 || dispatches.Load() != 3 {
				t.Fatalf("protocol change: retired=%v old idle=%d dispatches=%d", retired, idle, dispatches.Load())
			}
		})
	}
	test("direct", false)
	test("reverse", true)
}

func TestPeerServiceClientReusableCanceledIdleProbeRecovers(t *testing.T) {
	fail := make(chan struct{})
	entered := make(chan struct{})
	var failOnce sync.Once
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Error(err)
		}
		switch r.URL.Path {
		case "/fail":
			w.Header().Set("Content-Length", "16")
			writeServiceResponse(t, w, "partial")
			w.(http.Flusher).Flush()
			select {
			case <-fail:
				conn, _, err := w.(http.Hijacker).Hijack()
				if err != nil {
					t.Error(err)
					return
				}
				if err := conn.Close(); err != nil {
					t.Error(err)
				}
			case <-r.Context().Done():
			}
		case "/cancel":
			close(entered)
			<-r.Context().Done()
		default:
			writeServiceResponse(t, w, "ok")
		}
	}))
	t.Cleanup(func() { failOnce.Do(func() { close(fail) }) })
	f.adapter.opts.ReuseConnections = true
	client := f.adapter.Client("peer", noderec.ServiceCluster)
	failed, err := client.Post(f.url+"/fail", "text/plain", strings.NewReader("mutation"))
	if err != nil {
		t.Fatal(err)
	}
	defer failed.Body.Close()
	serviceClientPost(t, client, f.url+"/warm")
	failOnce.Do(func() { close(fail) })
	body, readErr := io.ReadAll(failed.Body)
	closeErr := failed.Body.Close()
	if string(body) != "partial" || !errors.Is(readErr, io.ErrUnexpectedEOF) || closeErr != nil {
		t.Fatalf("failed direct response: body=%q read=%v close=%v", body, readErr, closeErr)
	}
	key := peerServiceKey{"peer", noderec.ServiceCluster}
	f.adapter.mu.Lock()
	h := f.adapter.entries[key]
	idle, unhealthy := len(h.idle), !h.nextProbe.IsZero()
	idleDirect := idle == 1 && h.idle[0].direct
	due := time.Now().Add(-time.Second)
	h.nextProbe = due
	f.adapter.mu.Unlock()
	if !unhealthy || !idleDirect {
		t.Fatalf("probe setup: unhealthy=%v idle=%d direct=%v, want failed health with one idle direct connection", unhealthy, idle, idleDirect)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var reused atomic.Bool
	trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) { reused.Store(info.Reused) }}
	req, err := http.NewRequestWithContext(httptrace.WithClientTrace(ctx, trace), http.MethodPost, f.url+"/cancel", strings.NewReader("mutation"))
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		response, err := client.Do(req)
		if response != nil {
			response.Body.Close()
		}
		done <- err
	}()
	select {
	case <-entered:
	case err := <-done:
		t.Fatalf("probe ended before reaching the listener: %v", err)
	case <-time.After(time.Second):
		t.Fatal("probe did not reach the listener")
	}
	f.adapter.mu.Lock()
	probing := h.probing
	f.adapter.mu.Unlock()
	if !probing || !reused.Load() {
		t.Fatalf("due probe: probing=%v reused=%v, want an idle direct connection holding the probe reservation", probing, reused.Load())
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled probe error=%v, want context.Canceled", err)
	}
	f.adapter.mu.Lock()
	probing, nextProbe, slots := h.probing, h.nextProbe, len(h.slots)
	f.adapter.mu.Unlock()
	if probing || nextProbe != due || slots != 0 {
		t.Fatalf("canceled probe: probing=%v nextProbe=%v slots=%d, want released reservation, unchanged due time %v and no occupied slots", probing, nextProbe, slots, due)
	}
	serviceClientPost(t, client, f.url+"/recover")
	serviceClientPost(t, client, f.url+"/warm")
	f.adapter.mu.Lock()
	healthy := h.nextProbe.IsZero() && !h.probing
	f.adapter.mu.Unlock()
	f.mu.Lock()
	direct, reverse := f.directCalls[noderec.ServiceCluster], f.reverseCalls[noderec.ServiceCluster]
	f.mu.Unlock()
	f.assertDispatchConnections(t, 0, 1, 1, 2, 2)
	if !healthy || direct != 3 || reverse != 0 {
		t.Fatalf("direct recovery: healthy=%v direct=%d reverse=%d, want one replacement direct connection and reusable healthy state", healthy, direct, reverse)
	}
}

func TestPeerServiceClientReusableCancellationPreservesConcurrentProbe(t *testing.T) {
	test := func(name string, oldProbe bool) {
		t.Run(name, func(t *testing.T) {
			entered := make(chan struct{})
			fail := make(chan struct{})
			var failOnce sync.Once
			f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Error(err)
				}
				switch r.URL.Path {
				case "/cancel":
					close(entered)
					<-r.Context().Done()
				case "/fail":
					w.Header().Set("Content-Length", "16")
					writeServiceResponse(t, w, "partial")
					w.(http.Flusher).Flush()
					select {
					case <-fail:
					case <-r.Context().Done():
						return
					}
					conn, _, err := w.(http.Hijacker).Hijack()
					if err != nil {
						t.Error(err)
						return
					}
					if err := conn.Close(); err != nil {
						t.Error(err)
					}
				default:
					writeServiceResponse(t, w, "ok")
				}
			}))
			t.Cleanup(func() { failOnce.Do(func() { close(fail) }) })
			f.adapter.opts.ReuseConnections = true
			// Leave time to observe cancellation and another caller's route while
			// the later probe holds its setup reservation.
			f.adapter.opts.DirectTimeout = time.Second
			client := f.adapter.Client("peer", noderec.ServiceCluster)
			failed, err := client.Post(f.url+"/fail", "text/plain", strings.NewReader("mutation"))
			if err != nil {
				t.Fatal(err)
			}
			defer failed.Body.Close()
			serviceClientPost(t, client, f.url+"/warm")
			key := peerServiceKey{"peer", noderec.ServiceCluster}
			if oldProbe {
				f.adapter.mu.Lock()
				f.adapter.entries[key].nextProbe = time.Now().Add(-time.Second)
				f.adapter.mu.Unlock()
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, f.url+"/cancel", strings.NewReader("mutation"))
			if err != nil {
				t.Fatal(err)
			}
			canceled := make(chan error, 1)
			go func() {
				response, err := client.Do(req)
				if response != nil {
					response.Body.Close()
				}
				canceled <- err
			}()
			select {
			case <-entered:
			case err := <-canceled:
				t.Fatalf("cancel target ended before reaching the listener: %v", err)
			case <-time.After(time.Second):
				t.Fatal("cancel target did not reach the listener")
			}
			failOnce.Do(func() { close(fail) })
			body, readErr := io.ReadAll(failed.Body)
			closeErr := failed.Body.Close()
			if string(body) != "partial" || !errors.Is(readErr, io.ErrUnexpectedEOF) || closeErr != nil {
				t.Fatalf("failed direct response: body=%q read=%v close=%v", body, readErr, closeErr)
			}
			f.adapter.mu.Lock()
			h := f.adapter.entries[key]
			unhealthy := !h.nextProbe.IsZero()
			due := time.Now().Add(-time.Second)
			h.nextProbe = due
			f.adapter.mu.Unlock()
			if !unhealthy {
				t.Fatal("direct response failure did not mark health failed")
			}
			probeEntered := make(chan struct{})
			release := make(chan struct{})
			var releaseOnce, probeOnce sync.Once
			t.Cleanup(func() { releaseOnce.Do(func() { close(release) }) })
			f.mu.Lock()
			reachable := f.direct
			f.direct = func(ctx context.Context, service noderec.ServiceKey) (net.Conn, error) {
				probeOnce.Do(func() { close(probeEntered) })
				select {
				case <-release:
					return reachable(ctx, service)
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			}
			f.mu.Unlock()
			type result struct {
				response *http.Response
				err      error
			}
			probed := make(chan result, 1)
			go func() {
				response, err := client.Post(f.url+"/probe", "text/plain", strings.NewReader("mutation"))
				probed <- result{response, err}
			}()
			select {
			case <-probeEntered:
			case <-time.After(time.Second):
				t.Fatal("new direct probe did not start")
			}
			cancel()
			if err := <-canceled; !errors.Is(err, context.Canceled) {
				t.Fatalf("canceled request error=%v, want context.Canceled", err)
			}
			f.adapter.mu.Lock()
			probing, nextProbe := h.probing, h.nextProbe
			f.adapter.mu.Unlock()
			if !probing || nextProbe != due {
				t.Fatalf("concurrent probe after cancellation: probing=%v nextProbe=%v, want reserved probe and unchanged due time %v", probing, nextProbe, due)
			}
			serviceClientPost(t, client, f.url+"/other")
			releaseOnce.Do(func() { close(release) })
			probe := <-probed
			if probe.err != nil {
				t.Fatal(probe.err)
			}
			body, readErr = io.ReadAll(probe.response.Body)
			closeErr = probe.response.Body.Close()
			if string(body) != "ok" || readErr != nil || closeErr != nil || probe.response.StatusCode != http.StatusOK {
				t.Fatalf("probe recovery: body=%q read=%v close=%v status=%d", body, readErr, closeErr, probe.response.StatusCode)
			}
			f.mu.Lock()
			direct, reverse := f.directCalls[noderec.ServiceCluster], f.reverseCalls[noderec.ServiceCluster]
			f.mu.Unlock()
			if direct != 3 || reverse != 1 {
				t.Fatalf("concurrent routes: direct=%d reverse=%d, want one reserved direct probe and reverse for the other caller", direct, reverse)
			}
		})
	}
	test("healthy direct request owns no probe", false)
	test("old probe cannot release its successor", true)
}

func TestPeerServiceClientCloseIdlePreservesExposedResponse(t *testing.T) {
	release := make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	var dispatches atomic.Int32
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		dispatches.Add(1)
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Error(err)
		}
		if r.URL.Path != "/held" {
			writeServiceResponse(t, w, "ok")
			return
		}
		writeServiceResponse(t, w, "start-")
		w.(http.Flusher).Flush()
		select {
		case <-release:
			writeServiceResponse(t, w, "end")
		case <-r.Context().Done():
			t.Error("CloseIdle interrupted an exposed response")
		}
	}))
	f.adapter.opts.ReuseConnections = true
	client := f.adapter.Client("peer", noderec.ServiceCluster)
	held, err := client.Post(f.url+"/held", "text/plain", strings.NewReader("mutation"))
	if err != nil {
		t.Fatal(err)
	}
	defer held.Body.Close()
	serviceClientPost(t, client, f.url+"/warm")
	key := peerServiceKey{"peer", noderec.ServiceCluster}
	f.adapter.mu.Lock()
	old := f.adapter.entries[key]
	idleBefore := len(old.idle)
	f.adapter.mu.Unlock()
	f.assertDispatchConnections(t, 0, 1)
	if idleBefore != 1 {
		t.Fatalf("cleanup setup: idle=%d, want one idle beside one exposed response", idleBefore)
	}

	f.adapter.CloseIdle()
	select {
	case <-f.closed:
	case <-time.After(time.Second):
		t.Fatal("CloseIdle did not close the idle socket")
	}
	f.adapter.mu.Lock()
	retired, idle, global := old.retired, len(old.idle), f.adapter.idle
	f.adapter.mu.Unlock()
	if !retired || idle != 0 || global != 0 {
		t.Fatalf("CloseIdle state: retired=%v idle=%d global=%d", retired, idle, global)
	}

	once.Do(func() { close(release) })
	body, readErr := io.ReadAll(held.Body)
	closeErr := held.Body.Close()
	if string(body) != "start-end" || readErr != nil || closeErr != nil {
		t.Fatalf("exposed response after CloseIdle: body=%q read=%v close=%v", body, readErr, closeErr)
	}
	f.adapter.mu.Lock()
	oldIdle := len(old.idle)
	f.adapter.mu.Unlock()
	if oldIdle != 0 {
		t.Fatalf("retired response re-entered its old pool: idle=%d", oldIdle)
	}

	serviceClientPost(t, client, f.url+"/replacement")
	f.adapter.mu.Lock()
	replacement := f.adapter.entries[key]
	f.adapter.mu.Unlock()
	f.assertDispatchConnections(t, 0, 1, 2)
	if replacement == old || dispatches.Load() != 3 {
		t.Fatalf("replacement: new health=%v dispatches=%d", replacement != old, dispatches.Load())
	}
}

func TestPeerServiceClientDropUnauthorized(t *testing.T) {
	test := func(name string, change func(*serviceClientFixture), wantRetired bool) {
		t.Run(name, func(t *testing.T) {
			f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Error(err)
				}
				writeServiceResponse(t, w, "ok")
			}))
			f.adapter.opts.ReuseConnections = true
			serviceClientPost(t, f.adapter.Client("peer", noderec.ServiceCluster), f.url)
			key := peerServiceKey{"peer", noderec.ServiceCluster}
			f.adapter.mu.Lock()
			h := f.adapter.entries[key]
			var original *peerServiceReuse
			if len(h.idle) == 1 {
				original = h.idle[0]
			}
			f.adapter.mu.Unlock()
			if original == nil {
				t.Fatal("setup did not retain the original idle connection")
			}

			change(f)
			f.adapter.DropUnauthorized()

			f.adapter.mu.Lock()
			retired, idle, global := h.retired, len(h.idle), f.adapter.idle
			keptOriginal := idle == 1 && h.idle[0] == original
			f.adapter.mu.Unlock()
			if retired != wantRetired || (idle == 0) != wantRetired || (global == 0) != wantRetired {
				t.Fatalf("DropUnauthorized state: retired=%v idle=%d global=%d, want retired=%v", retired, idle, global, wantRetired)
			}
			if wantRetired {
				select {
				case <-f.closed:
				case <-time.After(time.Second):
					t.Fatal("DropUnauthorized did not close the idle socket")
				}
			} else if !keptOriginal {
				t.Fatal("DropUnauthorized replaced the original idle connection")
			}
		})
	}
	test("keeps authorized peer", func(*serviceClientFixture) {}, false)
	test("retires peer without route", func(f *serviceClientFixture) {
		f.mu.Lock()
		delete(f.routes, noderec.ServiceCluster)
		f.mu.Unlock()
	}, true)
	test("retires peer with new incarnation", func(f *serviceClientFixture) {
		f.mu.Lock()
		route := f.routes[noderec.ServiceCluster]
		route.Incarnation = "v2"
		f.routes[noderec.ServiceCluster] = route
		f.mu.Unlock()
	}, true)
}

func TestPeerServiceClientReusableProbePreservesReverseResponse(t *testing.T) {
	release := make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	f := newServiceClientFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Error(err)
		}
		if r.URL.Path != "/stream" {
			writeServiceResponse(t, w, "ok")
			return
		}
		writeServiceResponse(t, w, "start-")
		w.(http.Flusher).Flush()
		select {
		case <-release:
			writeServiceResponse(t, w, "end")
		case <-r.Context().Done():
			t.Error("direct recovery closed the active reusable reverse response")
		}
	}))
	f.adapter.opts.ReuseConnections = true
	reachable := f.direct
	f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, errors.New("blocked") }
	client := f.adapter.Client("peer", noderec.ServiceCluster)
	serviceClientPost(t, client, f.url)
	response, err := client.Post(f.url+"/stream", "text/plain", strings.NewReader("mutation"))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	f.mu.Lock()
	f.direct = reachable
	f.mu.Unlock()
	f.adapter.mu.Lock()
	f.adapter.entries[peerServiceKey{"peer", noderec.ServiceCluster}].nextProbe = time.Now().Add(-time.Second)
	f.adapter.mu.Unlock()
	serviceClientPost(t, client, f.url)
	serviceClientPost(t, client, f.url)
	once.Do(func() { close(release) })
	body, err := io.ReadAll(response.Body)
	if err != nil || string(body) != "start-end" {
		t.Fatalf("held reverse response: body=%q err=%v", body, err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.assertDispatchConnections(t, 0, 0, 1, 1)
	if f.directCalls[noderec.ServiceCluster] != 2 || f.reverseCalls[noderec.ServiceCluster] != 1 {
		t.Fatalf("direct recovery: direct=%v reverse=%v", f.directCalls, f.reverseCalls)
	}
}
