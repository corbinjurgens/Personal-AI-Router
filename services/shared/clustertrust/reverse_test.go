// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/binary"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hashicorp/yamux"
	"nvpair-shared/noderec"
)

const reverseTestID = "11111111111111111111111111111111"

func reverseTestManager(t *testing.T, opts ReverseOptions) *ReverseManager {
	t.Helper()
	if opts.Validate == nil {
		opts.Validate = func(peer ReversePeer) bool { return peer.Incarnation == "v1" }
	}
	if opts.ResolveService == nil {
		opts.ResolveService = func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
			return ReverseTarget{}, errors.New("not registered")
		}
	}
	if opts.SetupTimeout == 0 {
		opts.SetupTimeout = time.Second
	}
	m, err := NewReverseManager(opts)
	if err != nil {
		t.Fatalf("create reverse manager: %v", err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Errorf("close reverse manager: %v", err)
		}
	})
	return m
}

func reverseTestConnect(t *testing.T, a, b *ReverseManager, initiator, id string) {
	t.Helper()
	left, right := net.Pipe()
	if err := a.Register(ReversePeer{b.opts.LocalUUID, "v1"}, left, initiator, id); err != nil {
		right.Close()
		t.Fatalf("register first end: %v", err)
	}
	if err := b.Register(ReversePeer{a.opts.LocalUUID, "v1"}, right, initiator, id); err != nil {
		t.Fatalf("register second end: %v", err)
	}
	for _, end := range []struct {
		manager *ReverseManager
		peer    string
	}{{a, b.opts.LocalUUID}, {b, a.opts.LocalUUID}} {
		m := end.manager
		m.mu.Lock()
		s := m.sessions[end.peer]
		m.mu.Unlock()
		select {
		case <-s.ready:
		case <-s.ctx.Done():
			t.Fatal("control setup closed session")
		case <-time.After(3 * time.Second):
			t.Fatal("control setup timed out")
		}
	}
}

func reverseTestRawControl(t *testing.T, raw *yamux.Session, initiator bool) *yamux.Stream {
	t.Helper()
	var stream *yamux.Stream
	var err error
	if initiator {
		stream, err = raw.OpenStream()
	} else {
		stream, err = raw.AcceptStream()
	}
	if err != nil {
		t.Fatalf("raw control stream: %v", err)
	}
	stream.SetDeadline(time.Now().Add(3 * time.Second))
	if initiator {
		if _, err := io.WriteString(stream, "PAIR-REVERSE-CONTROL/1\n"); err != nil {
			t.Fatalf("control header: %v", err)
		}
		var ack [1]byte
		if _, err := io.ReadFull(stream, ack[:]); err != nil {
			t.Fatalf("control acknowledgment: %v", err)
		}
		if ack[0] != 0 {
			t.Fatalf("control acknowledgment = %d", ack[0])
		}
	} else {
		var header [23]byte
		if _, err := io.ReadFull(stream, header[:]); err != nil {
			t.Fatalf("control header: %v", err)
		}
		if string(header[:]) != "PAIR-REVERSE-CONTROL/1\n" {
			t.Fatalf("control header = %q", header)
		}
		if _, err := stream.Write([]byte{0}); err != nil {
			t.Fatalf("control acknowledgment: %v", err)
		}
	}
	stream.SetDeadline(time.Time{})
	t.Cleanup(func() { abortReverseStream(stream) })
	return stream
}

func reverseTestRaw(t *testing.T, opts ReverseOptions) (*ReverseManager, *yamux.Session) {
	t.Helper()
	m, raw, _ := reverseTestRawWithControl(t, opts)
	return m, raw
}

func reverseTestRawWithControl(t *testing.T, opts ReverseOptions) (*ReverseManager, *yamux.Session, *yamux.Stream) {
	t.Helper()
	opts.LocalUUID = "b"
	m := reverseTestManager(t, opts)
	left, right := net.Pipe()
	cfg := yamux.DefaultConfig()
	cfg.LogOutput = io.Discard
	cfg.StreamCloseTimeout = time.Second
	raw, err := yamux.Client(left, cfg)
	if err != nil {
		t.Fatalf("create raw yamux client: %v", err)
	}
	t.Cleanup(func() { raw.Close() })
	if err := m.Register(ReversePeer{"a", "v1"}, right, "a", reverseTestID); err != nil {
		t.Fatalf("register raw session: %v", err)
	}
	control := reverseTestRawControl(t, raw, true)
	m.mu.Lock()
	s := m.sessions["a"]
	m.mu.Unlock()
	select {
	case <-s.ready:
	case <-time.After(time.Second):
		t.Fatal("raw control setup did not finish")
	}
	return m, raw, control
}

func reverseTestStream(t *testing.T, raw *yamux.Session) *yamux.Stream {
	t.Helper()
	stream, err := raw.OpenStream()
	if err != nil {
		t.Fatalf("open raw stream: %v", err)
	}
	if err := stream.SetDeadline(time.Now().Add(3 * time.Second)); err != nil {
		t.Fatalf("set stream deadline: %v", err)
	}
	t.Cleanup(func() { abortReverseStream(stream) })
	return stream
}

func reverseTestAdmit(t *testing.T, stream net.Conn, service string) {
	t.Helper()
	if _, err := io.WriteString(stream, "PAIR-REVERSE/1 "+service+"\n"); err != nil {
		t.Fatalf("write service header: %v", err)
	}
	var ack [1]byte
	if _, err := io.ReadFull(stream, ack[:]); err != nil {
		t.Fatalf("read service admission: %v", err)
	}
	if ack[0] != 0 {
		t.Fatalf("service admission = %d, want 0", ack[0])
	}
}

func reverseTestClosed(t *testing.T, conn net.Conn) {
	t.Helper()
	if err := conn.SetReadDeadline(time.Now().Add(2 * time.Second)); err != nil && !errors.Is(err, io.ErrClosedPipe) && !errors.Is(err, net.ErrClosed) {
		t.Fatalf("set closure deadline: %v", err)
	}
	var b [1]byte
	if _, err := conn.Read(b[:]); err == nil {
		t.Fatal("closed stream accepted data")
	} else if timeout, ok := err.(net.Error); ok && timeout.Timeout() {
		t.Fatalf("stream did not close: %v", err)
	}
}

func TestReverseUpgradeHeaders(t *testing.T) {
	for _, tc := range []struct {
		name, connection, upgrade string
		want                      bool
	}{
		{"upgrade token", "keep-alive, Upgrade", "nvpair-reverse/1", true},
		{"case insensitive", "UPGRADE", "NVPAIR-REVERSE/1", true},
		{"missing token", "keep-alive", "nvpair-reverse/1", false},
		{"substring token", "not-upgrade", "nvpair-reverse/1", false},
		{"wrong protocol", "upgrade", "websocket", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := http.Header{"Connection": []string{tc.connection}, "Upgrade": []string{tc.upgrade}}
			if got := IsReverseUpgrade(h); got != tc.want {
				t.Fatalf("upgrade = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestReverseRegisterRejectsStaleAuthority(t *testing.T) {
	m := reverseTestManager(t, ReverseOptions{LocalUUID: "b"})
	left, right := net.Pipe()
	defer left.Close()
	err := m.Register(ReversePeer{"a", "v0"}, right, "a", reverseTestID)
	var unavailable *ReverseUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("stale registration error = %v, want typed unavailable", err)
	}
	reverseTestClosed(t, left)
	if m.Has(ReversePeer{"a", "v0"}) {
		t.Fatal("stale peer acquired usable session")
	}
}

func TestReverseDuplicateConvergesInEitherDialDirection(t *testing.T) {
	a := reverseTestManager(t, ReverseOptions{LocalUUID: "a"})
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b"})
	reverseTestConnect(t, a, b, "b", reverseTestID)
	if !a.Has(ReversePeer{"b", "v1"}) || !b.Has(ReversePeer{"a", "v1"}) {
		t.Fatal("sole higher UUID dial direction must be available")
	}
	reverseTestConnect(t, a, b, "a", "22222222222222222222222222222222")
	for _, m := range []*ReverseManager{a, b} {
		m.mu.Lock()
		for _, s := range m.sessions {
			if s.initiator != "a" {
				t.Errorf("winner initiator = %q, want a", s.initiator)
			}
		}
		m.mu.Unlock()
	}
	left, right := net.Pipe()
	defer right.Close()
	if err := a.Register(ReversePeer{"b", "v1"}, left, "b", "00000000000000000000000000000000"); err == nil {
		t.Fatal("losing duplicate accepted")
	}
	reverseTestClosed(t, right)
	if !a.Has(ReversePeer{"b", "v1"}) {
		t.Fatal("rejected duplicate displaced winner")
	}
}

func TestReverseDuplicateSameDirectionUsesSharedSessionID(t *testing.T) {
	a := reverseTestManager(t, ReverseOptions{LocalUUID: "a"})
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b"})
	reverseTestConnect(t, a, b, "a", "22222222222222222222222222222222")
	reverseTestConnect(t, a, b, "a", reverseTestID)
	for _, m := range []*ReverseManager{a, b} {
		m.mu.Lock()
		for _, s := range m.sessions {
			if s.id != reverseTestID {
				t.Errorf("winner session ID = %q, want %q", s.id, reverseTestID)
			}
		}
		m.mu.Unlock()
	}
}

func TestReverseRevocationClosesActiveForwarding(t *testing.T) {
	var valid atomic.Bool
	valid.Store(true)
	dialed := make(chan net.Conn, 1)
	resolve := func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
		return ReverseTarget{Protocol: ReversePlain, Dial: func(context.Context) (net.Conn, error) { left, right := net.Pipe(); dialed <- right; return left, nil }}, nil
	}
	a := reverseTestManager(t, ReverseOptions{LocalUUID: "a"})
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b", Validate: func(ReversePeer) bool { return valid.Load() }, ResolveService: resolve})
	reverseTestConnect(t, a, b, "a", reverseTestID)
	conn, err := a.Open(context.Background(), ReversePeer{"b", "v1"}, noderec.ServiceNodeInfo)
	if err != nil {
		t.Fatalf("open stream: %v", err)
	}
	defer conn.Close()
	var local net.Conn
	select {
	case local = <-dialed:
	case <-time.After(2 * time.Second):
		t.Fatal("service not dialed")
	}
	defer local.Close()
	valid.Store(false)
	b.Revalidate()
	reverseTestClosed(t, conn)
	reverseTestClosed(t, local)
	if b.Has(ReversePeer{"a", "v1"}) {
		t.Fatal("revoked session still usable")
	}
	if _, err := a.Open(context.Background(), ReversePeer{"b", "v1"}, noderec.ServiceNodeInfo); err == nil {
		t.Fatal("closed peer still opens streams")
	}
}

func TestReverseExplicitRevokePreservesOtherPeer(t *testing.T) {
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b"})
	a := reverseTestManager(t, ReverseOptions{LocalUUID: "a"})
	c := reverseTestManager(t, ReverseOptions{LocalUUID: "c"})
	reverseTestConnect(t, a, b, "a", reverseTestID)
	reverseTestConnect(t, c, b, "c", reverseTestID)
	b.Revoke("a")
	if b.Has(ReversePeer{"a", "v1"}) {
		t.Fatal("explicitly revoked peer still present")
	}
	if !b.Has(ReversePeer{"c", "v1"}) {
		t.Fatal("unrelated peer lost its session")
	}
}

func TestReverseRejectsMalformedServiceHeaders(t *testing.T) {
	for _, tc := range []struct{ name, header string }{
		{"unknown service", "PAIR-REVERSE/1 localhost:14321\n"},
		{"third peer", "PAIR-REVERSE/1 ni/a\n"},
		{"missing prefix", "ni\n"},
		{"oversized", strings.Repeat("a", 64)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			_, raw := reverseTestRaw(t, ReverseOptions{ResolveService: func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
				calls.Add(1)
				return ReverseTarget{}, nil
			}})
			stream := reverseTestStream(t, raw)
			if _, err := io.WriteString(stream, tc.header); err != nil {
				t.Fatalf("write header: %v", err)
			}
			reverseTestClosed(t, stream)
			if got := calls.Load(); got != 0 {
				t.Fatalf("invalid header reached resolver %d times", got)
			}
		})
	}
}

func TestReverseAuthorityChangeDuringHeaderSkipsResolver(t *testing.T) {
	var valid atomic.Bool
	valid.Store(true)
	var resolutions, dials atomic.Int32
	m, raw := reverseTestRaw(t, ReverseOptions{
		SetupTimeout: 5 * time.Second,
		Validate:     func(ReversePeer) bool { return valid.Load() },
		ResolveService: func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
			resolutions.Add(1)
			return ReverseTarget{Protocol: ReversePlain, Dial: func(context.Context) (net.Conn, error) {
				dials.Add(1)
				return nil, errors.New("unexpected local dial")
			}}, nil
		},
	})
	waitActive := func(want int) {
		t.Helper()
		deadline := time.Now().Add(time.Second)
		for {
			m.mu.Lock()
			active := m.active
			m.mu.Unlock()
			if active == want {
				return
			}
			if time.Now().After(deadline) {
				t.Fatalf("active streams = %d, want %d", active, want)
			}
			time.Sleep(time.Millisecond)
		}
	}
	stream := reverseTestStream(t, raw)
	if _, err := io.WriteString(stream, "PAIR-REVERSE/1 ni"); err != nil {
		t.Fatalf("write incomplete service header: %v", err)
	}
	// Reservation must finish under the old authority while the header still
	// lacks its newline. Do not Revalidate: that would close the session before
	// the completed header can exercise the resolver boundary.
	waitActive(1)
	valid.Store(false)
	if _, err := io.WriteString(stream, "\n"); err != nil {
		t.Fatalf("complete service header: %v", err)
	}
	reverseTestClosed(t, stream)
	waitActive(0)
	if got := resolutions.Load(); got != 0 {
		t.Errorf("stale authority reached resolver %d times, want 0", got)
	}
	if got := dials.Load(); got != 0 {
		t.Errorf("stale authority reached local dial %d times, want 0", got)
	}
}

func TestReverseSlowHeaderDoesNotStallOtherStreams(t *testing.T) {
	resolve := func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
		return ReverseTarget{Protocol: ReversePlain, Dial: func(context.Context) (net.Conn, error) {
			left, right := net.Pipe()
			go func() { defer right.Close(); io.WriteString(right, "ready") }()
			return left, nil
		}}, nil
	}
	_, raw := reverseTestRaw(t, ReverseOptions{SetupTimeout: 150 * time.Millisecond, ResolveService: resolve})
	slow := reverseTestStream(t, raw)
	if _, err := io.WriteString(slow, "PAIR-"); err != nil {
		t.Fatalf("write partial header: %v", err)
	}
	fast := reverseTestStream(t, raw)
	reverseTestAdmit(t, fast, "ni")
	var response [5]byte
	if _, err := io.ReadFull(fast, response[:]); err != nil {
		t.Fatalf("unrelated stream stalled: %v", err)
	}
	if string(response[:]) != "ready" {
		t.Fatalf("response=%q, want ready", response)
	}
	reverseTestClosed(t, slow)
}

func TestReverseRejectsPlainModeForMixedListeners(t *testing.T) {
	var dials atomic.Int32
	_, raw := reverseTestRaw(t, ReverseOptions{ResolveService: func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
		return ReverseTarget{Protocol: ReversePlain, Dial: func(context.Context) (net.Conn, error) { dials.Add(1); return nil, errors.New("unexpected dial") }}, nil
	}})
	stream := reverseTestStream(t, raw)
	if _, err := io.WriteString(stream, "PAIR-REVERSE/1 em\n"); err != nil {
		t.Fatalf("write header: %v", err)
	}
	var ack [1]byte
	if _, err := io.ReadFull(stream, ack[:]); err != nil {
		t.Fatalf("read rejection: %v", err)
	}
	if ack[0] != 1 {
		t.Fatalf("admission=%d, want rejection 1", ack[0])
	}
	if dials.Load() != 0 {
		t.Fatal("plain mixed listener was dialed")
	}
}

func TestReverseRejectsInvalidTLSBeforeLoopbackDial(t *testing.T) {
	for _, tc := range []struct {
		name    string
		payload []byte
	}{
		{"plaintext", []byte("GET / HTTP/1.1\r\n\r\n")},
		{"TLS prefixed junk", []byte{22, 3, 3, 0, 4, 1, 0, 0, 0}},
		{"invalid record type", []byte{23, 3, 3, 0, 1, 0}},
		{"record too large", []byte{22, 3, 3, 255, 255}},
		{"handshake too large", []byte{22, 3, 3, 0, 4, 1, 255, 255, 255}},
		{"slow first bytes", nil},
		{"partial TLS record", []byte{22, 3, 3, 0, 100, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var dials atomic.Int32
			_, raw := reverseTestRaw(t, ReverseOptions{SetupTimeout: 120 * time.Millisecond, ResolveService: func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
				return ReverseTarget{Protocol: ReverseTLS, Dial: func(context.Context) (net.Conn, error) { dials.Add(1); return nil, errors.New("unexpected dial") }}, nil
			}})
			stream := reverseTestStream(t, raw)
			reverseTestAdmit(t, stream, "em")
			if len(tc.payload) > 0 {
				if _, err := stream.Write(tc.payload); err != nil {
					t.Fatalf("write first protocol bytes: %v", err)
				}
			}
			reverseTestClosed(t, stream)
			if dials.Load() != 0 {
				t.Fatal("invalid TLS bytes reached loopback listener")
			}
		})
	}
}

func TestReverseInnerPinnedTLSStreamsLargeResponseConcurrently(t *testing.T) {
	selfPEM, selfKey, _ := genLeaf(t, "a")
	peerPEM, peerKey, _ := genLeaf(t, "b")
	selfMesh, _ := meshDir(t, selfPEM, selfKey, map[string]string{"b": string(peerPEM)})
	peerMesh, _ := meshDir(t, peerPEM, peerKey, map[string]string{"a": string(selfPEM)})
	chunk := bytes.Repeat([]byte("streamed-data-"), 4096)
	var requests atomic.Int32
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if uuid, ok := peerMesh.VerifyClientPin(r); !ok || uuid != "a" {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		requests.Add(1)
		for i := 0; i < 64; i++ {
			if _, err := w.Write(chunk); err != nil {
				return
			}
			w.(http.Flusher).Flush()
		}
	}))
	srv.TLS = peerMesh.ServerTLSConfig()
	srv.StartTLS()
	defer srv.Close()
	resolve := func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
		return ReverseTarget{Protocol: ReverseTLS, Dial: func(ctx context.Context) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp", srv.Listener.Addr().String())
		}}, nil
	}
	a := reverseTestManager(t, ReverseOptions{LocalUUID: "a"})
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b", ResolveService: resolve})
	reverseTestConnect(t, a, b, "a", reverseTestID)
	tlsConfig, ok := selfMesh.ClientTLSConfig("b")
	if !ok {
		t.Fatal("missing pinned client TLS config")
	}
	transport := &http.Transport{TLSClientConfig: tlsConfig, DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
		return a.Open(ctx, ReversePeer{"b", "v1"}, noderec.ServiceEngineManager)
	}}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport, Timeout: 10 * time.Second}
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := client.Get(srv.URL)
			if err != nil {
				t.Errorf("inner mTLS request: %v", err)
				return
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Errorf("status=%d, want 200", resp.StatusCode)
				return
			}
			n, err := io.Copy(io.Discard, resp.Body)
			if err != nil {
				t.Errorf("stream response: %v", err)
			}
			if n != 3670016 {
				t.Errorf("stream bytes=%d, want 3670016", n)
			}
		}()
	}
	wg.Wait()
	if requests.Load() != 3 {
		t.Fatalf("pinned application requests=%d, want 3", requests.Load())
	}
}

func TestReverseHalfClosePreservesResponseAfterRequestEOF(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer listener.Close()
	done := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			done <- err
			return
		}
		defer conn.Close()
		body, err := io.ReadAll(conn)
		if err != nil {
			done <- err
			return
		}
		if string(body) != "request" {
			done <- errors.New("incorrect request bytes")
			return
		}
		// Response completion outlives setup, including after request EOF.
		time.Sleep(150 * time.Millisecond)
		_, err = io.WriteString(conn, "response after EOF")
		done <- err
	}()
	resolve := func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
		return ReverseTarget{Protocol: ReversePlain, Dial: func(ctx context.Context) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp", listener.Addr().String())
		}}, nil
	}
	a := reverseTestManager(t, ReverseOptions{LocalUUID: "a", SetupTimeout: 75 * time.Millisecond})
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b", SetupTimeout: 75 * time.Millisecond, ResolveService: resolve})
	reverseTestConnect(t, a, b, "a", reverseTestID)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	conn, err := a.Open(ctx, ReversePeer{"b", "v1"}, noderec.ServiceNodeInfo)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer conn.Close()
	if _, err := io.WriteString(conn, "request"); err != nil {
		t.Fatalf("write request: %v", err)
	}
	half, ok := conn.(interface{ CloseWrite() error })
	if !ok {
		t.Fatal("reverse stream has no half-close")
	}
	if err := half.CloseWrite(); err != nil {
		t.Fatalf("half-close: %v", err)
	}
	response, err := io.ReadAll(conn)
	if err != nil {
		t.Fatalf("read response after EOF: %v", err)
	}
	if string(response) != "response after EOF" {
		t.Fatalf("response=%q", response)
	}
	if err := <-done; err != nil {
		t.Fatalf("local service: %v", err)
	}
}

func TestReverseStreamBudgetsRejectExcessWithoutStallingOtherPeers(t *testing.T) {
	b, raw := reverseTestRaw(t, ReverseOptions{PerSessionStreams: 1, MaxStreams: 2, SetupTimeout: time.Second})
	first := reverseTestStream(t, raw)
	if _, err := io.WriteString(first, "PAIR-"); err != nil {
		t.Fatalf("write first header: %v", err)
	}
	second := reverseTestStream(t, raw)
	reverseTestClosed(t, second)
	if b.Has(ReversePeer{"a", "v1"}) {
		t.Fatal("over-budget session still usable")
	}
	c := reverseTestManager(t, ReverseOptions{LocalUUID: "c"})
	reverseTestConnect(t, c, b, "c", reverseTestID)
	if !b.Has(ReversePeer{"c", "v1"}) {
		t.Fatal("unrelated peer could not register")
	}
}

func TestReverseProcessBudgetSpansPeerSessions(t *testing.T) {
	b, raw := reverseTestRaw(t, ReverseOptions{PerSessionStreams: 2, MaxStreams: 1})
	first := reverseTestStream(t, raw)
	if _, err := io.WriteString(first, "PAIR-"); err != nil {
		t.Fatalf("write partial header: %v", err)
	}
	// Synchronize with reservation through the manager lock, not a timing sleep.
	deadline := time.Now().Add(time.Second)
	for {
		b.mu.Lock()
		active := b.active
		b.mu.Unlock()
		if active == 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("first stream not reserved")
		}
		time.Sleep(time.Millisecond)
	}
	c := reverseTestManager(t, ReverseOptions{LocalUUID: "c"})
	reverseTestConnect(t, c, b, "c", reverseTestID)
	if _, err := c.Open(context.Background(), ReversePeer{"b", "v1"}, noderec.ServiceNodeInfo); err == nil {
		t.Fatal("process-wide excess stream admitted")
	}
	if !b.Has(ReversePeer{"a", "v1"}) {
		t.Fatal("process excess closed unrelated existing peer")
	}
}

func TestReverseCancellationAndShutdownCloseUnclosedClientStream(t *testing.T) {
	for _, mode := range []string{"cancel", "deadline", "shutdown"} {
		t.Run(mode, func(t *testing.T) {
			resolve := func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
				return ReverseTarget{Protocol: ReversePlain, Dial: func(context.Context) (net.Conn, error) {
					left, right := net.Pipe()
					t.Cleanup(func() { right.Close() })
					return left, nil
				}}, nil
			}
			a := reverseTestManager(t, ReverseOptions{LocalUUID: "a"})
			b := reverseTestManager(t, ReverseOptions{LocalUUID: "b", ResolveService: resolve})
			reverseTestConnect(t, a, b, "a", reverseTestID)
			ctx, cancel := context.WithCancel(context.Background())
			if mode == "deadline" {
				cancel()
				ctx, cancel = context.WithTimeout(context.Background(), 100*time.Millisecond)
			}
			defer cancel()
			conn, err := a.Open(ctx, ReversePeer{"b", "v1"}, noderec.ServiceNodeInfo)
			if err != nil {
				t.Fatalf("open: %v", err)
			}
			if mode == "deadline" {
				<-ctx.Done()
			} else if mode == "cancel" {
				cancel()
			} else {
				done := make(chan struct{})
				go func() { a.Close(); close(done) }()
				select {
				case <-done:
				case <-time.After(2 * time.Second):
					t.Fatal("shutdown waits for caller to close stream")
				}
			}
			var buf [1]byte
			if _, err := conn.Read(buf[:]); err == nil {
				t.Fatal("cancelled stream stayed readable")
			}
			conn.Close()
		})
	}
}

func TestReverseClientHelloParserBoundsConsumedBytes(t *testing.T) {
	left, right := net.Pipe()
	defer left.Close()
	defer right.Close()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	go func() { client := tls.Client(right, &tls.Config{ServerName: "test"}); client.HandshakeContext(ctx) }()
	prefix, err := readReverseClientHello(ctx, left)
	if err != nil {
		t.Fatalf("valid ClientHello rejected: %v", err)
	}
	if len(prefix) < 5 || len(prefix) > 65536 || prefix[0] != 22 {
		t.Fatalf("captured TLS prefix length=%d first byte=%d", len(prefix), prefix[0])
	}
}

func TestReverseRotatedIncarnationReplacesAndClosesOldSession(t *testing.T) {
	var incarnation atomic.Value
	incarnation.Store("v1")
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b", Validate: func(peer ReversePeer) bool { return peer.Incarnation == incarnation.Load().(string) }})
	a := reverseTestManager(t, ReverseOptions{LocalUUID: "a"})
	reverseTestConnect(t, a, b, "a", reverseTestID)
	incarnation.Store("v2")
	left, right := net.Pipe()
	cfg := yamux.DefaultConfig()
	cfg.LogOutput = io.Discard
	raw, err := yamux.Client(left, cfg)
	if err != nil {
		t.Fatalf("new incarnation session: %v", err)
	}
	defer raw.Close()
	if err := b.Register(ReversePeer{"a", "v2"}, right, "a", "ffffffffffffffffffffffffffffffff"); err != nil {
		t.Fatalf("register rotated incarnation: %v", err)
	}
	reverseTestRawControl(t, raw, true)
	b.mu.Lock()
	next := b.sessions["a"]
	b.mu.Unlock()
	select {
	case <-next.ready:
	case <-time.After(time.Second):
		t.Fatal("rotated control setup did not finish")
	}
	if !b.Has(ReversePeer{"a", "v2"}) {
		t.Fatal("new incarnation did not replace earlier lower session ID")
	}
	if b.Has(ReversePeer{"a", "v1"}) {
		t.Fatal("old incarnation still has authority")
	}
	if _, err := b.Open(context.Background(), ReversePeer{"a", "v1"}, noderec.ServiceNodeInfo); err == nil {
		t.Fatal("old incarnation opened a new stream")
	}
	deadline := time.NewTimer(time.Second)
	defer deadline.Stop()
	a.mu.Lock()
	old := a.sessions["b"]
	a.mu.Unlock()
	if old != nil {
		select {
		case <-old.session.CloseChan():
		case <-deadline.C:
			t.Fatal("old incarnation session was not closed")
		}
	}
}

func TestReverseOpenSetupCancellationReleasesCapacity(t *testing.T) {
	a := reverseTestManager(t, ReverseOptions{LocalUUID: "a", PerSessionStreams: 1, SetupTimeout: time.Second})
	left, right := net.Pipe()
	cfg := yamux.DefaultConfig()
	cfg.LogOutput = io.Discard
	raw, err := yamux.Server(right, cfg)
	if err != nil {
		t.Fatalf("raw server: %v", err)
	}
	defer raw.Close()
	if err := a.Register(ReversePeer{"b", "v1"}, left, "a", reverseTestID); err != nil {
		t.Fatalf("register: %v", err)
	}
	reverseTestRawControl(t, raw, false)
	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() {
		conn, err := a.Open(ctx, ReversePeer{"b", "v1"}, noderec.ServiceNodeInfo)
		if conn != nil {
			conn.Close()
		}
		result <- err
	}()
	stream, err := raw.AcceptStream()
	if err != nil {
		t.Fatalf("accept setup stream: %v", err)
	}
	defer abortReverseStream(stream)
	cancel()
	select {
	case err := <-result:
		var unavailable *ReverseUnavailableError
		if !errors.As(err, &unavailable) {
			t.Fatalf("setup cancellation = %v, want typed unavailable", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("setup cancellation failed to return")
	}
	closed := make(chan struct{})
	go func() { a.Close(); close(closed) }()
	select {
	case <-closed:
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled setup retained its stream reservation")
	}
}

func TestReverseForwardCancellationClosesBothReaders(t *testing.T) {
	a, peerA := net.Pipe()
	defer peerA.Close()
	b, peerB := net.Pipe()
	defer peerB.Close()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- ForwardReverse(ctx, a, b, nil) }()
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("forward cancellation error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancelled forwarding blocked")
	}
	reverseTestClosed(t, peerA)
	reverseTestClosed(t, peerB)
}

func TestReverseOuterSessionDoesNotAuthorizeUnpinnedInnerClient(t *testing.T) {
	selfPEM, selfKey, _ := genLeaf(t, "a")
	peerPEM, peerKey, _ := genLeaf(t, "b")
	otherPEM, otherKey, _ := genLeaf(t, "other")
	selfMesh, _ := meshDir(t, selfPEM, selfKey, map[string]string{"b": string(peerPEM)})
	peerMesh, _ := meshDir(t, peerPEM, peerKey, map[string]string{"a": string(selfPEM)})
	otherCert, err := tls.X509KeyPair(otherPEM, otherKey)
	if err != nil {
		t.Fatalf("parse unpinned client fixture: %v", err)
	}
	var requests atomic.Int32
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, ok := peerMesh.VerifyClientPin(r); !ok {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		requests.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	srv.TLS = peerMesh.ServerTLSConfig()
	srv.StartTLS()
	defer srv.Close()
	resolve := func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
		return ReverseTarget{Protocol: ReverseTLS, Dial: func(ctx context.Context) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp", srv.Listener.Addr().String())
		}}, nil
	}
	a := reverseTestManager(t, ReverseOptions{LocalUUID: "a"})
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b", ResolveService: resolve})
	reverseTestConnect(t, a, b, "a", reverseTestID)
	tlsConfig, ok := selfMesh.ClientTLSConfig("b")
	if !ok {
		t.Fatal("missing server pin")
	}
	tlsConfig.Certificates = []tls.Certificate{otherCert}
	transport := &http.Transport{TLSClientConfig: tlsConfig, DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
		return a.Open(ctx, ReversePeer{"b", "v1"}, noderec.ServiceEngineManager)
	}}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport, Timeout: 3 * time.Second}
	resp, err := client.Get(srv.URL)
	if err != nil {
		t.Fatalf("unpinned inner client HTTP response: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("unpinned inner client status = %d, want 403", resp.StatusCode)
	}
	if got := requests.Load(); got != 0 {
		t.Fatalf("unpinned client reached %d application handlers", got)
	}
}

func TestReverseCloseCancelsRemoteForwarderAndPreservesOtherStream(t *testing.T) {
	dialed := make(chan net.Conn, 2)
	resolve := func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
		return ReverseTarget{Protocol: ReversePlain, Dial: func(context.Context) (net.Conn, error) { left, right := net.Pipe(); dialed <- right; return left, nil }}, nil
	}
	a := reverseTestManager(t, ReverseOptions{LocalUUID: "a", PerSessionStreams: 2})
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b", PerSessionStreams: 2, ResolveService: resolve})
	reverseTestConnect(t, a, b, "a", reverseTestID)
	first, err := a.Open(context.Background(), ReversePeer{"b", "v1"}, noderec.ServiceNodeInfo)
	if err != nil {
		t.Fatalf("first open: %v", err)
	}
	defer first.Close()
	var localFirst net.Conn
	select {
	case localFirst = <-dialed:
	case <-time.After(time.Second):
		t.Fatal("first local dial missing")
	}
	defer localFirst.Close()
	second, err := a.Open(context.Background(), ReversePeer{"b", "v1"}, noderec.ServiceNodeInfo)
	if err != nil {
		t.Fatalf("second open: %v", err)
	}
	defer second.Close()
	var localSecond net.Conn
	select {
	case localSecond = <-dialed:
	case <-time.After(time.Second):
		t.Fatal("second local dial missing")
	}
	defer localSecond.Close()
	// This local service keeps its response side open after request EOF. Only
	// an explicit stream cancellation can release the remote forwarding slot.
	first.Close()
	reverseTestClosed(t, localFirst)
	deadline := time.Now().Add(time.Second)
	for {
		b.mu.Lock()
		active := b.active
		b.mu.Unlock()
		if active == 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("remote active slots=%d, want 1 after close", active)
		}
		time.Sleep(time.Millisecond)
	}
	if !a.Has(ReversePeer{"b", "v1"}) || !b.Has(ReversePeer{"a", "v1"}) {
		t.Fatal("closing one stream closed its session")
	}
	writeDone := make(chan error, 1)
	go func() { _, err := io.WriteString(localSecond, "still open"); writeDone <- err }()
	second.SetReadDeadline(time.Now().Add(time.Second))
	var payload [10]byte
	if _, err := io.ReadFull(second, payload[:]); err != nil {
		t.Fatalf("unrelated stream: %v", err)
	}
	if string(payload[:]) != "still open" {
		t.Fatalf("unrelated response=%q", payload)
	}
	if err := <-writeDone; err != nil {
		t.Fatalf("unrelated local write: %v", err)
	}
}

func TestReverseControlRejectsWrongDirectionAndSlowFrames(t *testing.T) {
	for _, tc := range []struct {
		name  string
		frame []byte
	}{
		{"control stream cancellation", []byte{0, 0, 0, 1}},
		{"wrong direction", []byte{0, 0, 0, 2}},
		{"slow partial frame", []byte{0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, raw, control := reverseTestRawWithControl(t, ReverseOptions{SetupTimeout: 120 * time.Millisecond})
			if _, err := control.Write(tc.frame); err != nil && !(len(tc.frame) == 4 && errors.Is(err, yamux.ErrSessionShutdown)) {
				t.Fatalf("write control frame: %v", err)
			}
			select {
			case <-raw.CloseChan():
			case <-time.After(time.Second):
				t.Fatal("invalid or incomplete control frame did not close session")
			}
		})
	}
}

func TestReversePendingCancellationBudgetIsFinite(t *testing.T) {
	m, raw, control := reverseTestRawWithControl(t, ReverseOptions{PerSessionStreams: 2})
	var frame [4]byte
	for _, id := range []uint32{101, 103} {
		binary.BigEndian.PutUint32(frame[:], id)
		if _, err := control.Write(frame[:]); err != nil {
			t.Fatalf("write pending cancel: %v", err)
		}
	}
	deadline := time.Now().Add(time.Second)
	for {
		m.mu.Lock()
		s := m.sessions["a"]
		m.mu.Unlock()
		if s == nil {
			t.Fatal("session closed before reaching pending cancellation cap")
		}
		s.mu.Lock()
		pending := len(s.pendingCancel)
		s.mu.Unlock()
		if pending == 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("pending cancellations=%d, want 2", pending)
		}
		time.Sleep(time.Millisecond)
	}
	binary.BigEndian.PutUint32(frame[:], 105)
	if _, err := control.Write(frame[:]); err != nil && !errors.Is(err, yamux.ErrSessionShutdown) {
		t.Fatalf("write excess cancellation: %v", err)
	}
	select {
	case <-raw.CloseChan():
	case <-time.After(time.Second):
		t.Fatal("excess future cancellations did not close session")
	}
}

func TestReverseSessionLimitBoundsControlStreamOverhead(t *testing.T) {
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b", MaxSessions: 1})
	a := reverseTestManager(t, ReverseOptions{LocalUUID: "a"})
	reverseTestConnect(t, a, b, "a", reverseTestID)
	left, right := net.Pipe()
	defer left.Close()
	if err := b.Register(ReversePeer{"c", "v1"}, right, "c", reverseTestID); err == nil {
		t.Fatal("process-wide session cap exceeded")
	}
	reverseTestClosed(t, left)
	if !b.Has(ReversePeer{"a", "v1"}) {
		t.Fatal("session cap rejected existing peer")
	}
}

func TestReverseControlSetupRejectsMalformedAndSlowHeaders(t *testing.T) {
	for _, tc := range []struct {
		name   string
		header []byte
	}{
		{"malformed control header", bytes.Repeat([]byte("x"), 23)},
		{"slow control header", []byte("PAIR-")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			m := reverseTestManager(t, ReverseOptions{LocalUUID: "b", SetupTimeout: 120 * time.Millisecond, ResolveService: func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
				calls.Add(1)
				return ReverseTarget{}, nil
			}})
			left, right := net.Pipe()
			cfg := yamux.DefaultConfig()
			cfg.LogOutput = io.Discard
			raw, err := yamux.Client(left, cfg)
			if err != nil {
				t.Fatalf("raw client: %v", err)
			}
			defer raw.Close()
			if err := m.Register(ReversePeer{"a", "v1"}, right, "a", reverseTestID); err != nil {
				t.Fatalf("register: %v", err)
			}
			stream := reverseTestStream(t, raw)
			if _, err := stream.Write(tc.header); err != nil && !errors.Is(err, yamux.ErrSessionShutdown) {
				t.Fatalf("write invalid setup header: %v", err)
			}
			select {
			case <-raw.CloseChan():
			case <-time.After(time.Second):
				t.Fatal("invalid control setup did not close")
			}
			if calls.Load() != 0 {
				t.Fatal("control header reached service resolver")
			}
			if m.Has(ReversePeer{"a", "v1"}) {
				t.Fatal("failed setup registered usable authority")
			}
		})
	}
}

func TestReverseSlowClientHelloDoesNotStallPlainNodeInfo(t *testing.T) {
	resolve := func(_ context.Context, _ ReversePeer, service noderec.ServiceKey) (ReverseTarget, error) {
		if service == noderec.ServiceEngineManager {
			return ReverseTarget{Protocol: ReverseTLS, Dial: func(context.Context) (net.Conn, error) { return nil, errors.New("slow TLS must not dial") }}, nil
		}
		return ReverseTarget{Protocol: ReversePlain, Dial: func(context.Context) (net.Conn, error) {
			left, right := net.Pipe()
			go func() { defer right.Close(); io.WriteString(right, "ready") }()
			return left, nil
		}}, nil
	}
	_, raw := reverseTestRaw(t, ReverseOptions{SetupTimeout: 150 * time.Millisecond, ResolveService: resolve})
	slow := reverseTestStream(t, raw)
	reverseTestAdmit(t, slow, "em")
	fast := reverseTestStream(t, raw)
	reverseTestAdmit(t, fast, "ni")
	var response [5]byte
	if _, err := io.ReadFull(fast, response[:]); err != nil {
		t.Fatalf("plain node-info stalled behind ClientHello: %v", err)
	}
	if string(response[:]) != "ready" {
		t.Fatalf("response=%q, want ready", response)
	}
	reverseTestClosed(t, slow)
}

type reverseTestPauseConn struct {
	net.Conn
	paused  atomic.Bool
	blocked chan struct{}
	closed  chan struct{}
	once    sync.Once
}

func (c *reverseTestPauseConn) Read(b []byte) (int, error) {
	for {
		if c.paused.Load() {
			select {
			case c.blocked <- struct{}{}:
			default:
			}
			<-c.closed
			return 0, net.ErrClosed
		}
		n, err := c.Conn.Read(b)
		if timeout, ok := err.(net.Error); n == 0 && c.paused.Load() && ok && timeout.Timeout() {
			continue
		}
		return n, err
	}
}
func (c *reverseTestPauseConn) Close() error {
	c.once.Do(func() { close(c.closed) })
	return c.Conn.Close()
}
func (c *reverseTestPauseConn) pause(t *testing.T) {
	t.Helper()
	c.paused.Store(true)
	if err := c.Conn.SetReadDeadline(time.Now()); err != nil {
		t.Fatalf("interrupt peer read: %v", err)
	}
	select {
	case <-c.blocked:
	case <-time.After(time.Second):
		t.Fatal("peer read did not pause")
	}
	if err := c.Conn.SetReadDeadline(time.Time{}); err != nil {
		t.Fatalf("clear peer read deadline: %v", err)
	}
}

func reverseTestPingPeer(t *testing.T, manager *ReverseManager) (*yamux.Session, *reverseTestPauseConn) {
	t.Helper()
	left, right := net.Pipe()
	peer := &reverseTestPauseConn{Conn: left, blocked: make(chan struct{}, 1), closed: make(chan struct{})}
	cfg := yamux.DefaultConfig()
	cfg.LogOutput = io.Discard
	cfg.EnableKeepAlive = false
	raw, err := yamux.Client(peer, cfg)
	if err != nil {
		t.Fatalf("ping peer client: %v", err)
	}
	t.Cleanup(func() { raw.Close() })
	if err := manager.Register(ReversePeer{"a", "v1"}, right, "a", reverseTestID); err != nil {
		t.Fatalf("register ping peer: %v", err)
	}
	reverseTestRawControl(t, raw, true)
	if _, err := raw.Ping(); err != nil {
		t.Fatalf("ordinary initial ping: %v", err)
	}
	return raw, peer
}

func reverseTestPingFrame(id uint32) [12]byte {
	frame := [12]byte{0, 2, 0, 1}
	binary.BigEndian.PutUint32(frame[8:12], id)
	return frame
}

func TestReversePingFloodHasBoundedWorkersAndPreservesOtherPeer(t *testing.T) {
	resolve := func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error) {
		return ReverseTarget{Protocol: ReversePlain, Dial: func(context.Context) (net.Conn, error) {
			left, right := net.Pipe()
			go func() { defer right.Close(); io.WriteString(right, "ready") }()
			return left, nil
		}}, nil
	}
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b", SetupTimeout: 2 * time.Second, ResolveService: resolve})
	_, peer := reverseTestPingPeer(t, b)
	c := reverseTestManager(t, ReverseOptions{LocalUUID: "c"})
	reverseTestConnect(t, c, b, "c", reverseTestID)
	peer.pause(t)
	b.mu.Lock()
	active := b.active
	b.mu.Unlock()
	if active != 0 {
		t.Fatalf("application streams before ping flood=%d, want 0", active)
	}
	baseline := runtime.NumGoroutine()
	type observation struct {
		sent, peak int
		err        error
	}
	result := make(chan observation, 1)
	go func() {
		got := observation{peak: runtime.NumGoroutine()}
		for id := uint32(0); id < 512; id++ {
			frame := reverseTestPingFrame(id)
			n, err := peer.Conn.Write(frame[:])
			if n == 12 {
				got.sent++
			}
			if count := runtime.NumGoroutine(); count > got.peak {
				got.peak = count
			}
			if err != nil {
				got.err = err
				break
			}
		}
		result <- got
	}()
	var got observation
	select {
	case got = <-result:
	case <-time.After(time.Second):
		peer.Close()
		t.Fatal("ping flood was not bounded by peer shutdown")
	}
	if got.err == nil {
		t.Fatal("unresponsive peer consumed every ping without being closed")
	}
	// Eight queued replies, one active reply, and the rejecting query are the
	// fixed wire admission limit. No application-stream budget is involved.
	if got.sent < 1 || got.sent > 10 {
		t.Fatalf("ping frames consumed=%d, want between 1 and 10", got.sent)
	}
	if extra := got.peak - baseline; extra > 12 {
		t.Fatalf("ping flood allocated %d extra goroutines, want at most 12", extra)
	}
	t.Logf("consumed ping frames=%d; peak additional goroutines=%d", got.sent, got.peak-baseline)
	deadline := time.Now().Add(time.Second)
	for b.Has(ReversePeer{"a", "v1"}) {
		if time.Now().After(deadline) {
			t.Fatal("flooding peer remains usable")
		}
		time.Sleep(time.Millisecond)
	}
	if !b.Has(ReversePeer{"c", "v1"}) {
		t.Fatal("ping flood closed unrelated peer")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	conn, err := c.Open(ctx, ReversePeer{"b", "v1"}, noderec.ServiceNodeInfo)
	if err != nil {
		t.Fatalf("unrelated peer open: %v", err)
	}
	defer conn.Close()
	var response [5]byte
	if _, err := io.ReadFull(conn, response[:]); err != nil {
		t.Fatalf("unrelated peer response: %v", err)
	}
	if string(response[:]) != "ready" {
		t.Fatalf("unrelated response=%q", response)
	}
}

func TestReverseBlockedPingReplyClosesAtSocketWriteDeadline(t *testing.T) {
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b", SetupTimeout: 150 * time.Millisecond})
	_, peer := reverseTestPingPeer(t, b)
	peer.pause(t)
	started := time.Now()
	frame := reverseTestPingFrame(42)
	if _, err := peer.Conn.Write(frame[:]); err != nil {
		t.Fatalf("one ping query: %v", err)
	}
	deadline := time.Now().Add(time.Second)
	for b.Has(ReversePeer{"a", "v1"}) {
		if time.Now().After(deadline) {
			peer.Close()
			t.Fatal("blocked socket write exceeded its bound")
		}
		time.Sleep(time.Millisecond)
	}
	elapsed := time.Since(started)
	if elapsed < 75*time.Millisecond || elapsed > 750*time.Millisecond {
		t.Fatalf("single blocked reply shutdown=%v, want 75ms..750ms for 150ms deadline", elapsed)
	}
	if _, err := peer.Conn.Write(frame[:]); err == nil {
		t.Fatal("expired ping reply socket remained writable")
	}
	t.Logf("single blocked reply shutdown=%v", elapsed)
}

type reverseTestFragmentConn struct{ net.Conn }

func (c *reverseTestFragmentConn) Read(b []byte) (int, error) {
	if len(b) <= 12 && len(b) > 3 {
		b = b[:3]
	} else if len(b) > 113 {
		b = b[:113]
	}
	return c.Conn.Read(b)
}
func (c *reverseTestFragmentConn) Write(b []byte) (int, error) {
	if len(b) <= 12 && len(b) > 3 {
		b = b[:3]
	} else if len(b) > 257 {
		b = b[:257]
	}
	return c.Conn.Write(b)
}

func TestReverseMuxPingRepliesPreserveFragmentedLargeDuplexFrames(t *testing.T) {
	left, right := net.Pipe()
	cfg := yamux.DefaultConfig()
	cfg.LogOutput = io.Discard
	cfg.EnableKeepAlive = false
	cfg.ConnectionWriteTimeout = 5 * time.Second
	client, err := yamux.Client(newReverseMuxConn(&reverseTestFragmentConn{left}, 5*time.Second), cfg)
	if err != nil {
		t.Fatalf("guarded client: %v", err)
	}
	defer client.Close()
	server, err := yamux.Server(newReverseMuxConn(&reverseTestFragmentConn{right}, 5*time.Second), cfg)
	if err != nil {
		t.Fatalf("guarded server: %v", err)
	}
	defer server.Close()
	out, err := client.OpenStream()
	if err != nil {
		t.Fatalf("open duplex stream: %v", err)
	}
	in, err := server.AcceptStream()
	if err != nil {
		t.Fatalf("accept duplex stream: %v", err)
	}
	out.SetDeadline(time.Now().Add(10 * time.Second))
	in.SetDeadline(time.Now().Add(10 * time.Second))
	// Payloads intentionally resemble ping headers; only frame boundaries may
	// be interpreted as control traffic.
	clientChunk := bytes.Repeat([]byte{0, 2, 0, 1, 0, 0, 0, 0, 1, 2, 3, 4}, 2048)
	serverChunk := bytes.Repeat([]byte{0, 2, 0, 2, 0, 0, 0, 0, 5, 6, 7, 8}, 2048)
	clientWant := sha256.Sum256(bytes.Repeat(clientChunk, 128))
	serverWant := sha256.Sum256(bytes.Repeat(serverChunk, 128))
	var wg sync.WaitGroup
	write := func(stream *yamux.Stream, chunk []byte) {
		defer wg.Done()
		for i := 0; i < 128; i++ {
			if _, err := stream.Write(chunk); err != nil {
				t.Errorf("duplex write: %v", err)
				return
			}
		}
		if err := stream.Close(); err != nil {
			t.Errorf("duplex half-close: %v", err)
		}
	}
	read := func(stream *yamux.Stream, want [32]byte) {
		defer wg.Done()
		hash := sha256.New()
		n, err := io.Copy(hash, stream)
		if err != nil {
			t.Errorf("duplex read: %v", err)
			return
		}
		if n != 3145728 {
			t.Errorf("duplex bytes=%d, want 3145728", n)
		}
		if !bytes.Equal(hash.Sum(nil), want[:]) {
			t.Error("ping reply corrupted duplex payload")
		}
	}
	ping := func(session *yamux.Session) {
		defer wg.Done()
		for i := 0; i < 16; i++ {
			if _, err := session.Ping(); err != nil {
				t.Errorf("duplex ping: %v", err)
				return
			}
		}
	}
	wg.Add(6)
	go write(out, clientChunk)
	go write(in, serverChunk)
	go read(out, serverWant)
	go read(in, clientWant)
	go ping(client)
	go ping(server)
	wg.Wait()
}

func reverseTestPendingMuxFrame(t *testing.T, timeout time.Duration) (*reverseMuxConn, net.Conn) {
	t.Helper()
	left, right := net.Pipe()
	conn := newReverseMuxConn(left, timeout)
	t.Cleanup(func() {
		right.Close()
		// On a regression, let the original frame writer release its token
		// after the assertion failed, keeping cleanup from hiding the failure.
		conn.Write([]byte("x"))
		conn.Close()
	})
	var header [12]byte
	binary.BigEndian.PutUint32(header[4:8], 1)
	binary.BigEndian.PutUint32(header[8:12], 1)
	readDone := make(chan error, 1)
	go func() {
		var got [12]byte
		_, err := io.ReadFull(right, got[:])
		if err == nil && got != header {
			err = errors.New("outgoing header changed")
		}
		readDone <- err
	}()
	if n, err := conn.Write(header[:]); err != nil || n != 12 {
		t.Fatalf("complete data header: n=%d err=%v", n, err)
	}
	if err := <-readDone; err != nil {
		t.Fatalf("peer data header: %v", err)
	}
	var reply [12]byte
	reply[1] = 2
	binary.BigEndian.PutUint16(reply[2:4], 2)
	conn.replies <- reply
	deadline := time.Now().Add(time.Second)
	for len(conn.replies) != 0 {
		if time.Now().After(deadline) {
			t.Fatal("reply worker did not dequeue header")
		}
		runtime.Gosched()
	}
	return conn, right
}

func TestReverseMuxCloseDoesNotWaitForMissingFramePayload(t *testing.T) {
	conn, peer := reverseTestPendingMuxFrame(t, time.Second)
	closed := make(chan error, 1)
	go func() { closed <- conn.Close() }()
	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("close pending frame: %v", err)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("close waited for a frame payload that will never follow")
	}
	select {
	case <-conn.replyDone:
	default:
		t.Fatal("close returned with reply worker still running")
	}
	if n, err := conn.Write([]byte("x")); n != 0 || !errors.Is(err, net.ErrClosed) {
		t.Fatalf("payload after close: n=%d err=%v, want closed", n, err)
	}
	var b [1]byte
	if n, err := peer.Read(b[:]); n != 0 || !errors.Is(err, io.EOF) {
		t.Fatalf("closed partial frame emitted bytes: n=%d err=%v", n, err)
	}
}

func TestReverseMuxMissingPayloadBoundsReplyBoundaryWait(t *testing.T) {
	conn, peer := reverseTestPendingMuxFrame(t, 150*time.Millisecond)
	started := time.Now()
	select {
	case <-conn.replyDone:
	case <-time.After(time.Second):
		t.Fatal("missing payload retained reply worker beyond its wait budget")
	}
	elapsed := time.Since(started)
	if elapsed < 75*time.Millisecond || elapsed > 750*time.Millisecond {
		t.Fatalf("frame-boundary expiry=%v, want 75ms..750ms for a 150ms budget", elapsed)
	}
	if err := conn.Close(); err != nil {
		t.Fatalf("close expired frame: %v", err)
	}
	var b [1]byte
	if n, err := peer.Read(b[:]); n != 0 || !errors.Is(err, io.EOF) {
		t.Fatalf("reply entered incomplete frame: n=%d err=%v", n, err)
	}
	t.Logf("missing-payload reply boundary expired after %v", elapsed)
}
