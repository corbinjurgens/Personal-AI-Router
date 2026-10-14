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

type serviceClientFixture struct {
	adapter      *PeerServiceClient
	mesh         *Mesh
	dir          string
	server       *httptest.Server
	url          string
	mu           sync.Mutex
	routes       map[noderec.ServiceKey]PeerServiceRoute
	directCalls  map[noderec.ServiceKey]int
	reverseCalls map[noderec.ServiceKey]int
	direct       func(context.Context, noderec.ServiceKey) (net.Conn, error)
	reverse      func(context.Context, noderec.ServiceKey) (net.Conn, error)
	connections  atomic.Int32
}

func newServiceClientFixture(t *testing.T, handler http.Handler) *serviceClientFixture {
	t.Helper()
	selfPEM, selfKey, _ := genLeaf(t, "self")
	peerPEM, peerKey, _ := genLeaf(t, "peer")
	self, dir := meshDir(t, selfPEM, selfKey, map[string]string{"peer": string(peerPEM)})
	peer, _ := meshDir(t, peerPEM, peerKey, map[string]string{"self": string(selfPEM)})
	f := &serviceClientFixture{mesh: self, dir: dir, routes: make(map[noderec.ServiceKey]PeerServiceRoute), directCalls: make(map[noderec.ServiceKey]int), reverseCalls: make(map[noderec.ServiceKey]int)}
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
		handler.ServeHTTP(w, r)
	}))
	f.server.TLS = peer.ServerTLSConfig()
	f.server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			f.connections.Add(1)
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
			if got := f.connections.Load(); got != 2 {
				t.Fatalf("connections after warm GET and mutation = %d, want 2 fresh connections", got)
			}
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
	if mutations.Load() != 1 || f.connections.Load() != 2 {
		t.Fatalf("reverse mutation dispatches=%d connections=%d, want one dispatch and two fresh connections", mutations.Load(), f.connections.Load())
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
			if _, err := recreated.Stat(); !errors.Is(err, os.ErrClosed) {
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
