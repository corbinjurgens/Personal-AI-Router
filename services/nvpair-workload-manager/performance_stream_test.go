// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/tls"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
	"nvpair-shared/performancewire"
)

// performancePipeListener hands in-memory connections to an http.Server, so a
// stream test serves real TLS and streaming responses without a socket.
type performancePipeListener struct {
	conns chan net.Conn
	done  chan struct{}
	once  sync.Once
}

func (l *performancePipeListener) Accept() (net.Conn, error) {
	select {
	case conn := <-l.conns:
		return conn, nil
	case <-l.done:
		return nil, net.ErrClosed
	}
}

func (l *performancePipeListener) Close() error {
	l.once.Do(func() { close(l.done) })
	return nil
}

func (l *performancePipeListener) Addr() net.Addr { return &net.TCPAddr{Port: 1} }

// performanceStreamServe points the fixture's stream client at handler, served
// over pinned cluster TLS as uuid-peer on in-memory connections. It returns a
// function that closes the server and every open stream.
func performanceStreamServe(t *testing.T, f *performanceFixture, handler http.Handler) func() {
	t.Helper()
	peerMesh := clustertrust.Open(f.peerDir)
	listener := &performancePipeListener{conns: make(chan net.Conn), done: make(chan struct{})}
	server := &http.Server{Handler: handler}
	served := make(chan struct{})
	go func() {
		defer close(served)
		_ = server.Serve(tls.NewListener(listener, peerMesh.ServerTLSConfig()))
	}()
	f.manager.performance.streams = clustertrust.NewPeerServiceClient(f.manager.mesh, clustertrust.PeerServiceClientOptions{
		ResolveRoute: f.manager.broadcaster.resolveRoute,
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			client, peer := net.Pipe()
			select {
			case listener.conns <- peer:
				return client, nil
			case <-listener.done:
			case <-ctx.Done():
			}
			_ = client.Close()
			_ = peer.Close()
			return nil, net.ErrClosed
		},
		DialReverse: func(context.Context, string, string, noderec.ServiceKey) (net.Conn, error) {
			return nil, errors.New("test reverse path disabled")
		},
	})
	var once sync.Once
	stop := func() {
		once.Do(func() {
			_ = server.Close()
			<-served
		})
	}
	t.Cleanup(stop)
	return stop
}

// performancePeer is uuid-peer's own manager, whose stream handler serves
// the summaries it publishes. Its clock is separate from the receiver's.
type performancePeer struct {
	manager *Manager
	mu      sync.Mutex
	now     time.Time
}

func newPerformancePeer(t *testing.T, f *performanceFixture) *performancePeer {
	t.Helper()
	p := &performancePeer{manager: NewManager(NewCodec(codecNop{}), 0, "uuid-peer", f.peerDir), now: time.Now()}
	p.manager.performance.now = func() time.Time {
		p.mu.Lock()
		defer p.mu.Unlock()
		return p.now
	}
	t.Cleanup(p.manager.broadcaster.clients.CloseIdle)
	return p
}

func (p *performancePeer) advance(d time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.now = p.now.Add(d)
}

func (p *performancePeer) publish(t *testing.T, summary performancewire.Summary) {
	t.Helper()
	if err := p.manager.performance.publish(performanceEncode(t, summary)); err != nil {
		t.Fatalf("peer publish: %v", err)
	}
}

func (p *performancePeer) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc(performanceStreamPath, p.manager.server.handlePerformanceStream)
	return mux
}

// shortKeepalive makes stream keepalives frequent for the test.
func shortKeepalive(t *testing.T) {
	t.Helper()
	previous := performanceStreamKeepalive
	performanceStreamKeepalive = 20 * time.Millisecond
	t.Cleanup(func() { performanceStreamKeepalive = previous })
}

// performanceSubscribe runs one consumeStream for uuid-peer. The returned
// channel closes once the stream ends; cancel ends it.
func performanceSubscribe(t *testing.T, f *performanceFixture) (<-chan struct{}, context.CancelFunc) {
	t.Helper()
	performanceAddPeer(f.manager)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		f.manager.performance.consumeStream(ctx, "uuid-peer")
	}()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("performance stream did not stop")
		}
	})
	return done, cancel
}

func performanceEventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func performanceStreamEnded(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("stream did not end")
	}
}

func performanceStreamOpen(t *testing.T, done <-chan struct{}, within time.Duration) {
	t.Helper()
	select {
	case <-done:
		t.Fatal("stream ended")
	case <-time.After(within):
	}
}

// performanceLines serves the given NDJSON lines once, then keepalives until
// the request ends.
func performanceLines(lines ...string) http.Handler {
	return performanceLinesThen(true, lines...)
}

// performanceLinesThen serves lines, then keepalives until the request ends
// when keep is set, or ends the response at once otherwise.
func performanceLinesThen(keep bool, lines ...string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		for _, line := range lines {
			_, _ = w.Write([]byte(line + "\n"))
		}
		http.NewResponseController(w).Flush()
		if !keep {
			return
		}
		ticker := time.NewTicker(20 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-r.Context().Done():
				return
			case <-ticker.C:
				if _, err := w.Write([]byte("null\n")); err != nil {
					return
				}
				http.NewResponseController(w).Flush()
			}
		}
	})
}

func performanceLine(t *testing.T, summary performancewire.Summary, elapsedMs int64) string {
	t.Helper()
	return `{"summary":` + string(performanceEncode(t, summary)) + `,"elapsedMs":` + strconv.FormatInt(elapsedMs, 10) + `}`
}

func TestPerformanceStreamNewerRevisionEmitsReceivedOnce(t *testing.T) {
	shortKeepalive(t)
	f := newPerformanceFixture(t)
	peer := newPerformancePeer(t, f)
	summary := performanceSummary("uuid-peer", 1)
	want := performancewire.ReportedRate{Samples: 1, DurationNs: 400_000_000, TokensPerSecond: 55}
	summary.Entries = []performancewire.Entry{{
		Key:     performancewire.ProfileKey{Host: "uuid-peer", Engine: "llamacpp-custom", Model: "model", Fingerprint: "profile"},
		Samples: 1, Completed: 1, NewestObservedAtMs: 1000, EngineDecode: &want,
	}}
	peer.publish(t, summary)
	performanceStreamServe(t, f, peer.handler())
	performanceSubscribe(t, f)
	performanceEventually(t, "first summary", func() bool { return len(f.summaries(t)) == 1 })
	got := f.summaries(t)[0]
	if got.Summary.Revision != 1 || got.Summary.Entries[0].EngineDecode == nil || *got.Summary.Entries[0].EngineDecode != want {
		t.Fatalf("received = %+v, want revision 1 with reported decode %+v", got, want)
	}
	peer.publish(t, performanceSummary("uuid-peer", 2))
	performanceEventually(t, "second summary", func() bool { return len(f.summaries(t)) == 2 })
	// An identical republication reaches the receiver as the same revision:
	// it refreshes freshness but emits no second replacement.
	statuses := len(f.statuses(t))
	peer.publish(t, performanceSummary("uuid-peer", 2))
	performanceEventually(t, "same-revision status", func() bool { return len(f.statuses(t)) > statuses })
	if got := f.summaries(t); len(got) != 2 || got[1].Summary.Revision != 2 {
		t.Fatalf("received = %+v, want revisions 1 and 2 once each", got)
	}
}

func TestPerformanceStreamCoalescesToNewestPublication(t *testing.T) {
	shortKeepalive(t)
	f := newPerformanceFixture(t)
	peer := newPerformancePeer(t, f)
	for revision := int64(1); revision <= 100; revision++ {
		peer.publish(t, performanceSummary("uuid-peer", revision))
	}
	performanceStreamServe(t, f, peer.handler())
	performanceSubscribe(t, f)
	performanceEventually(t, "summary", func() bool { return len(f.summaries(t)) == 1 })
	time.Sleep(100 * time.Millisecond)
	if got := f.summaries(t); len(got) != 1 || got[0].Summary.Revision != 100 {
		t.Fatalf("received = %+v, want only revision 100", got)
	}
}

func TestPerformanceStreamFirstFrameCarriesCurrentElapsed(t *testing.T) {
	shortKeepalive(t)
	f := newPerformanceFixture(t)
	peer := newPerformancePeer(t, f)
	peer.publish(t, performanceSummary("uuid-peer", 1))
	peer.advance(20 * time.Second)
	// An identical republication keeps the original publication anchor.
	peer.publish(t, performanceSummary("uuid-peer", 1))
	peer.advance(20 * time.Second)
	performanceStreamServe(t, f, peer.handler())
	performanceSubscribe(t, f)
	performanceEventually(t, "summary", func() bool { return len(f.summaries(t)) == 1 })
	if got := f.summaries(t)[0].SampleElapsedMs; got != 40_000 {
		t.Fatalf("sample elapsed = %d, want the 40000 ms age at stream open", got)
	}
}

func TestPerformanceStreamWithheldBeforeAnyFrameStaysOpen(t *testing.T) {
	shortKeepalive(t)
	f := newPerformanceFixture(t)
	peer := newPerformancePeer(t, f)
	peer.publish(t, performanceSummary("uuid-peer", 1))
	performanceWrite(t, filepath.Join(f.peerDir, "admission.json"), []byte(`{"clusterId":"test-cluster","epoch":2,"counter":2,"activated":2}`))
	performanceStreamServe(t, f, peer.handler())
	done, _ := performanceSubscribe(t, f)
	performanceStreamOpen(t, done, 200*time.Millisecond)
	if len(f.messages()) != 0 || len(f.manager.performance.received) != 0 {
		t.Fatal("withheld summary reached the receiver")
	}
}

// Once a stream has delivered a frame, withholding ends it, so keepalives can
// no longer keep the receiver's copy fresh and the 90 s stale rule applies.
func TestPerformanceStreamWithheldAfterFrameEndsStream(t *testing.T) {
	test := func(name string, withhold func(*testing.T, *performanceFixture, *performancePeer)) {
		t.Run(name, func(t *testing.T) {
			shortKeepalive(t)
			f := newPerformanceFixture(t)
			peer := newPerformancePeer(t, f)
			peer.publish(t, performanceSummary("uuid-peer", 1))
			performanceStreamServe(t, f, peer.handler())
			done, _ := performanceSubscribe(t, f)
			performanceEventually(t, "summary", func() bool { return len(f.summaries(t)) == 1 })
			withhold(t, f, peer)
			performanceStreamEnded(t, done)
			if len(f.summaries(t)) != 1 {
				t.Fatal("withheld publication was sent again")
			}
			f.advance(performanceStaleAfter)
			f.manager.performance.sweep()
			statuses := f.statuses(t)
			if last := statuses[len(statuses)-1]; last.Status != "stale" {
				t.Fatalf("statuses = %+v, want stale after withholding", statuses)
			}
		})
	}
	// Only the sender's admission changes; the receiver's view of it lags.
	test("sender admission changed", func(t *testing.T, f *performanceFixture, _ *performancePeer) {
		performanceWrite(t, filepath.Join(f.peerDir, "admission.json"), []byte(`{"clusterId":"test-cluster","epoch":2,"counter":2,"activated":2}`))
	})
	test("past retention", func(t *testing.T, f *performanceFixture, peer *performancePeer) {
		retention := time.Duration(performancewire.RetentionMs)*time.Millisecond + time.Millisecond
		peer.advance(retention)
		f.advance(retention)
	})
}

// A frame with an older revision than the receipt holds is ignored, and the
// keepalives after it do not refresh that newer receipt.
func TestPerformanceStreamOlderRevisionDoesNotRefresh(t *testing.T) {
	f := newPerformanceFixture(t)
	if code := f.post(t, performanceSummary("uuid-peer", 5)); code != http.StatusOK {
		t.Fatalf("initial status = %d", code)
	}
	performanceStreamServe(t, f, performanceLines(performanceLine(t, performanceSummary("uuid-peer", 1), 0)))
	done, _ := performanceSubscribe(t, f)
	performanceStreamOpen(t, done, 100*time.Millisecond)
	f.advance(performanceStatusRefresh)
	time.Sleep(100 * time.Millisecond)
	if n := len(f.statuses(t)); n != 1 {
		t.Fatalf("statuses = %d, want only the initial fresh status", n)
	}
	f.advance(performanceStaleAfter)
	time.Sleep(100 * time.Millisecond)
	f.manager.performance.sweep()
	statuses := f.statuses(t)
	if len(statuses) != 2 || statuses[1].Status != "stale" {
		t.Fatalf("statuses = %+v, want stale behind an older-revision stream", statuses)
	}
}

// consumeStream reports a healthy stream only when it delivered lines and did
// not end on a refused or oversized frame, so refusals keep backing off.
func TestPerformanceStreamHealthDrivesBackoff(t *testing.T) {
	test := func(name string, want bool, lines ...string) {
		t.Run(name, func(t *testing.T) {
			f := newPerformanceFixture(t)
			performanceStreamServe(t, f, performanceLinesThen(false, lines...))
			performanceAddPeer(f.manager)
			if got := f.manager.performance.consumeStream(context.Background(), "uuid-peer"); got != want {
				t.Fatalf("healthy = %v, want %v", got, want)
			}
		})
	}
	test("accepted then closed", true, performanceLine(t, performanceSummary("uuid-peer", 1), 0), "null")
	test("keepalive only", true, "null")
	test("refused frame", false, performanceLine(t, performanceSummary("uuid-third-party", 1), 0))
	test("malformed frame", false, `{"summary":{},"elapsedMs":-1}`)
	test("over line limit", false, `{"summary":"`+strings.Repeat("x", performanceStreamMaxLine)+`","elapsedMs":0}`)
	test("no line", false)
}

func TestPerformanceStreamKeepaliveRefreshesFreshness(t *testing.T) {
	shortKeepalive(t)
	f := newPerformanceFixture(t)
	peer := newPerformancePeer(t, f)
	peer.publish(t, performanceSummary("uuid-peer", 1))
	performanceStreamServe(t, f, peer.handler())
	performanceSubscribe(t, f)
	performanceEventually(t, "first status", func() bool { return len(f.statuses(t)) == 1 })
	time.Sleep(100 * time.Millisecond)
	if n := len(f.statuses(t)); n != 1 {
		t.Fatalf("statuses = %d within the refresh interval, want 1", n)
	}
	f.advance(performanceStatusRefresh)
	performanceEventually(t, "refreshed status", func() bool { return len(f.statuses(t)) == 2 })
	time.Sleep(100 * time.Millisecond)
	statuses := f.statuses(t)
	if len(statuses) != 2 || statuses[1].Status != "fresh" || statuses[1].SampleElapsedMs != performanceStatusRefresh.Milliseconds() {
		t.Fatalf("statuses = %+v, want one refresh per interval", statuses)
	}
	// Keepalives keep lastSeen current, so the stale rule does not fire.
	f.advance(performanceStaleAfter - time.Second)
	time.Sleep(100 * time.Millisecond)
	f.manager.performance.sweep()
	if last := f.statuses(t)[len(f.statuses(t))-1]; last.Status != "fresh" {
		t.Fatalf("live stream went %s", last.Status)
	}
	if len(f.summaries(t)) != 1 {
		t.Fatal("keepalives re-emitted the summary")
	}
}

func TestPerformanceStreamClosedGoesStale(t *testing.T) {
	shortKeepalive(t)
	f := newPerformanceFixture(t)
	peer := newPerformancePeer(t, f)
	peer.publish(t, performanceSummary("uuid-peer", 1))
	stop := performanceStreamServe(t, f, peer.handler())
	done, _ := performanceSubscribe(t, f)
	performanceEventually(t, "summary", func() bool { return len(f.statuses(t)) == 1 })
	stop()
	performanceStreamEnded(t, done)
	f.advance(performanceStaleAfter - time.Millisecond)
	f.manager.performance.sweep()
	if n := len(f.statuses(t)); n != 1 {
		t.Fatalf("statuses = %d before the stale interval, want 1", n)
	}
	f.advance(time.Millisecond)
	f.manager.performance.sweep()
	statuses := f.statuses(t)
	if len(statuses) != 2 || statuses[1].Status != "stale" {
		t.Fatalf("statuses = %+v, want stale 90 s after the stream closed", statuses)
	}
}

func TestPerformanceStreamAdmissionChangeRetiresReceipt(t *testing.T) {
	shortKeepalive(t)
	f := newPerformanceFixture(t)
	peer := newPerformancePeer(t, f)
	peer.publish(t, performanceSummary("uuid-peer", 1))
	performanceStreamServe(t, f, peer.handler())
	done, _ := performanceSubscribe(t, f)
	performanceEventually(t, "summary", func() bool { return len(f.statuses(t)) == 1 })
	// Let applySummary finish its last recheck (and a few keepalives pass)
	// before readmission; a change during that recheck retires it at once.
	time.Sleep(100 * time.Millisecond)
	f.peerEpoch(t, 2)
	performanceStreamEnded(t, done)
	f.manager.performance.sweep()
	statuses := f.statuses(t)
	if len(statuses) != 2 || statuses[1].Status != "unknown" {
		t.Fatalf("statuses = %+v, want unknown after readmission", statuses)
	}
	if state := f.manager.performance.received["uuid-peer"]; state == nil || !state.retired || state.payload != nil {
		t.Fatal("readmission did not retire the receipt")
	}
}

func TestPerformanceStreamEndsOnRefusedFrame(t *testing.T) {
	t.Run("same revision changed contents", func(t *testing.T) {
		f := newPerformanceFixture(t)
		changed := performanceSummary("uuid-peer", 1)
		changed.Window.MaxSamplesPerKey = 63
		performanceStreamServe(t, f, performanceLines(performanceLine(t, performanceSummary("uuid-peer", 1), 0), performanceLine(t, changed, 0)))
		done, _ := performanceSubscribe(t, f)
		performanceStreamEnded(t, done)
		if len(f.summaries(t)) != 1 || len(f.statuses(t)) != 1 {
			t.Fatal("changed same-revision content was accepted")
		}
	})
	t.Run("measuring host differs from peer", func(t *testing.T) {
		f := newPerformanceFixture(t)
		performanceStreamServe(t, f, performanceLines(performanceLine(t, performanceSummary("uuid-third-party", 1), 0)))
		done, _ := performanceSubscribe(t, f)
		performanceStreamEnded(t, done)
		if len(f.messages()) != 0 {
			t.Fatal("forwarded summary reached broker")
		}
	})
	t.Run("malformed frame", func(t *testing.T) {
		f := newPerformanceFixture(t)
		performanceStreamServe(t, f, performanceLines(`{"summary":{},"elapsedMs":-1}`))
		done, _ := performanceSubscribe(t, f)
		performanceStreamEnded(t, done)
		if len(f.messages()) != 0 {
			t.Fatal("malformed frame reached broker")
		}
	})
}

func TestPerformanceFrameDecoding(t *testing.T) {
	summary := `{"version":2}`
	valid := func(name, line string, elapsed int64) {
		t.Run(name, func(t *testing.T) {
			frame, err := decodePerformanceFrame([]byte(line))
			if err != nil || frame.ElapsedMs != elapsed || string(frame.Summary) != summary {
				t.Fatalf("frame = %+v, %v", frame, err)
			}
		})
	}
	invalid := func(name, line string) {
		t.Run(name, func(t *testing.T) {
			if _, err := decodePerformanceFrame([]byte(line)); err == nil {
				t.Fatal("frame accepted")
			}
		})
	}
	valid("zero", `{"summary":`+summary+`,"elapsedMs":0}`, 0)
	valid("retention", `{"summary":`+summary+`,"elapsedMs":`+strconv.FormatInt(performancewire.RetentionMs, 10)+`}`, performancewire.RetentionMs)
	invalid("missing elapsed", `{"summary":`+summary+`}`)
	invalid("missing summary", `{"elapsedMs":0}`)
	invalid("negative", `{"summary":`+summary+`,"elapsedMs":-1}`)
	invalid("leading zero", `{"summary":`+summary+`,"elapsedMs":01}`)
	invalid("fraction", `{"summary":`+summary+`,"elapsedMs":1.0}`)
	invalid("exponent", `{"summary":`+summary+`,"elapsedMs":1e3}`)
	invalid("string", `{"summary":`+summary+`,"elapsedMs":"0"}`)
	invalid("int64 overflow", `{"summary":`+summary+`,"elapsedMs":9223372036854775808}`)
	invalid("past retention", `{"summary":`+summary+`,"elapsedMs":`+strconv.FormatInt(performancewire.RetentionMs+1, 10)+`}`)
	invalid("unknown field", `{"summary":`+summary+`,"elapsedMs":0,"extra":1}`)
	invalid("trailing JSON", `{"summary":`+summary+`,"elapsedMs":0} {}`)
}

func TestPerformanceStreamRouteRequiresPinnedPeer(t *testing.T) {
	f := newPerformanceFixture(t)
	w := httptest.NewRecorder()
	f.manager.server.handlePerformanceStream(w, httptest.NewRequest(http.MethodGet, performanceStreamPath, nil))
	if w.Code != http.StatusForbidden {
		t.Fatalf("unauthenticated stream status = %d, want 403", w.Code)
	}
	// The former summary POST is gone from the events endpoint.
	if code := f.manager.server.pinnedDispatch(t, f, performancewire.MethodSummary); code != http.StatusBadRequest {
		t.Fatalf("%s status = %d, want 400", performancewire.MethodSummary, code)
	}
}

// pinnedDispatch posts a JSON-RPC notification for method to the events
// endpoint as uuid-peer, over the in-memory TLS stream seam.
func (s *Server) pinnedDispatch(t *testing.T, f *performanceFixture, method string) int {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc(eventsPath, s.handleEvents)
	self := clustertrust.Open(f.selfDir)
	peer := clustertrust.Open(f.peerDir)
	ts := httptest.NewUnstartedServer(mux)
	ts.Listener.Close()
	listener := &performancePipeListener{conns: make(chan net.Conn), done: make(chan struct{})}
	ts.Listener = listener
	ts.TLS = self.ServerTLSConfig()
	ts.StartTLS()
	t.Cleanup(ts.Close)
	cfg, ok := peer.ClientTLSConfig("uuid-self")
	if !ok {
		t.Fatal("peer cannot build a client for self")
	}
	client := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg,
		DialContext: func(context.Context, string, string) (net.Conn, error) {
			client, server := net.Pipe()
			listener.conns <- server
			return client, nil
		}}}
	t.Cleanup(client.CloseIdleConnections)
	resp, err := client.Post("https://pipe"+eventsPath, "application/json", strings.NewReader(`{"jsonrpc":"2.0","method":"`+method+`","params":{}}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	return resp.StatusCode
}

func TestPerformanceSubscriptionsFollowAdmittedTargets(t *testing.T) {
	f := newPerformanceFixture(t)
	f.manager.performance.streams = clustertrust.NewPeerServiceClient(f.manager.mesh, clustertrust.PeerServiceClientOptions{
		ResolveRoute: f.manager.broadcaster.resolveRoute,
		DialContext: func(context.Context, string, string) (net.Conn, error) {
			return nil, errors.New("test dial disabled")
		},
		DialReverse: func(context.Context, string, string, noderec.ServiceKey) (net.Conn, error) {
			return nil, errors.New("test reverse path disabled")
		},
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	active := make(map[string]context.CancelFunc)
	// A relay alias cannot authorize the peer's measuring identity.
	f.manager.peers.Replace([]PeerNode{{ID: "legacy-host-alias", Addresses: []string{"192.0.2.10"}, Port: 14320, TXT: []string{"cluster-uuid=uuid-peer"}}})
	f.manager.performance.syncStreams(ctx, active)
	if len(active) != 0 {
		t.Fatal("legacy alias acquired a performance subscription")
	}
	performanceAddPeer(f.manager)
	f.manager.performance.syncStreams(ctx, active)
	if len(active) != 1 || active["uuid-peer"] == nil {
		t.Fatalf("subscriptions = %v, want uuid-peer", active)
	}
	performanceWrite(t, filepath.Join(f.selfDir, "members.json"), []byte(`[]`))
	f.manager.performance.syncStreams(ctx, active)
	if len(active) != 0 {
		t.Fatal("revoked peer kept its subscription")
	}
	// The cancelled keepStream returns from its backoff wait at once; give it
	// time to stop reading the cluster directory before TempDir cleanup.
	cancel()
	time.Sleep(100 * time.Millisecond)
}

func TestPerformanceStreamDefaults(t *testing.T) {
	if performanceStreamKeepalive != 10*time.Second || performanceStreamIdle != 25*time.Second ||
		performanceStatusRefresh != 30*time.Second || performanceSweepEvery != 30*time.Second || performanceStaleAfter != 90*time.Second {
		t.Fatal("performance stream cadence differs from the plan (10 s keepalive, 25 s idle, 30 s refresh and sweep, 90 s stale)")
	}
}
