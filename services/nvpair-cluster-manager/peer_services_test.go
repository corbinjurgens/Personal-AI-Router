// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/durablefile"
	"nvpair-shared/noderec"
	"nvpair-shared/reach"
)

// ownedPeerListener keeps its address reserved across each physical dial. A
// closed listener cannot silently authorize a later owner of the same port.
type ownedPeerListener struct {
	net.Listener
	mu     sync.RWMutex
	closed bool
}

func (l *ownedPeerListener) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true
	return l.Listener.Close()
}

type peerDialGuard struct {
	mu                          sync.Mutex
	targets                     map[string]string
	owned                       map[string]*ownedPeerListener
	logical                     map[string]int
	probes                      map[string]int
	setup                       map[string]func(context.Context) error
	physical, rejected, reverse atomic.Int32
}

func newPeerDialGuard(t *testing.T) *peerDialGuard {
	t.Helper()
	g := &peerDialGuard{targets: make(map[string]string), owned: make(map[string]*ownedPeerListener), logical: make(map[string]int), probes: make(map[string]int), setup: make(map[string]func(context.Context) error)}
	t.Cleanup(func() {
		g.mu.Lock()
		defer g.mu.Unlock()
		if g.rejected.Load() != 0 {
			t.Errorf("guard rejected physical dial: %d", g.rejected.Load())
		}
		t.Logf("owned-dial counts logical=%v chooser=%v physical=%d reverse=%d rejected=%d", g.logical, g.probes, g.physical.Load(), g.reverse.Load(), g.rejected.Load())
	})
	return g
}

func (g *peerDialGuard) allow(t *testing.T, logical string, listener *ownedPeerListener) {
	t.Helper()
	physical := listener.Addr().String()
	host, port, err := net.SplitHostPort(physical)
	if err != nil || !net.ParseIP(host).IsLoopback() || port == "0" {
		t.Fatalf("non-owned listener address %q", physical)
	}
	g.mu.Lock()
	g.owned[physical] = listener
	g.targets[physical] = physical
	g.targets[logical] = physical
	g.mu.Unlock()
}

func (g *peerDialGuard) physicalDial(ctx context.Context, network, address string) (net.Conn, error) {
	host, _, err := net.SplitHostPort(address)
	g.mu.Lock()
	owner := g.owned[address]
	g.mu.Unlock()
	if err != nil || network != "tcp" || !net.ParseIP(host).IsLoopback() || owner == nil {
		g.rejected.Add(1)
		return nil, errors.New("unowned physical destination")
	}
	owner.mu.RLock()
	defer owner.mu.RUnlock()
	if owner.closed {
		return nil, errors.New("owned listener closed")
	}
	g.physical.Add(1)
	return (&net.Dialer{}).DialContext(ctx, network, address)
}

func (g *peerDialGuard) dial(ctx context.Context, network, logical string) (net.Conn, error) {
	g.mu.Lock()
	g.logical[logical]++
	physical, ok := g.targets[logical]
	setup := g.setup[logical]
	g.mu.Unlock()
	if setup != nil {
		if err := setup(ctx); err != nil {
			return nil, err
		}
	}
	if !ok {
		return nil, errors.New("synthetic unavailable candidate")
	}
	return g.physicalDial(ctx, network, physical)
}

func (g *peerDialGuard) probe(network, logical string, timeout time.Duration) (net.Conn, error) {
	g.mu.Lock()
	g.probes[logical]++
	physical, ok := g.targets[logical]
	g.mu.Unlock()
	if !ok {
		return nil, errors.New("synthetic unavailable chooser candidate")
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return g.physicalDial(ctx, network, physical)
}

func (g *peerDialGuard) attempted(logical string) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.logical[logical]
}

type peerWalkFixture struct {
	a, b       *Manager
	guard      *peerDialGuard
	listener   *ownedPeerListener
	descriptor noderec.PeerServicesDescriptor
	output     bytes.Buffer
	handlers   atomic.Int32
	lan, tail  string
}

func newPeerWalkFixture(t *testing.T) *peerWalkFixture {
	t.Helper()
	f := &peerWalkFixture{a: newReverseTestManager(t), b: newReverseTestManager(t), guard: newPeerDialGuard(t)}
	f.descriptor = noderec.PeerServicesDescriptor{HostUUID: f.b.identity.NodeUUID, Name: "walk-peer", Addresses: []string{"192.0.2.55", "100.90.0.8"}, Services: []noderec.PeerService{{Service: noderec.ServiceNodeInfo, Port: 23456}}}
	if err := noderec.ValidatePeerServicesDescriptor(f.descriptor, f.b.identity.NodeUUID); err != nil {
		t.Fatal(err)
	}
	f.listener = startWalkServer(t, f.b, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.handlers.Add(1)
		if err := json.NewEncoder(w).Encode(f.descriptor); err != nil {
			t.Error(err)
		}
	}))
	f.b.port = f.listener.Addr().(*net.TCPAddr).Port
	pairReverseManagers(t, f.a, f.b)
	f.lan = net.JoinHostPort("192.0.2.55", fmt.Sprint(f.b.port))
	f.tail = net.JoinHostPort("100.90.0.8", fmt.Sprint(f.b.port))
	f.a.setMemberAddr(f.b.identity.NodeUUID, "192.0.2.55", f.b.port)
	persistPeerServicesMembers(t, f.a, f.b)
	f.guard.allow(t, f.lan, f.listener)
	f.install(t, f.a, f.guard, nil)
	f.install(t, f.b, newPeerDialGuard(t), nil)
	f.a.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), &f.output})
	f.a.serviceRefresh.source = "first-process"
	return f
}

func startWalkServer(t *testing.T, peer *Manager, handler http.Handler) *ownedPeerListener {
	t.Helper()
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, ok := peer.verifyClientPin(r); !ok {
			http.Error(w, "unpaired fixture", http.StatusForbidden)
			return
		}
		handler.ServeHTTP(w, r)
	}))
	owned := &ownedPeerListener{Listener: server.Listener}
	server.Listener = owned
	server.TLS = peer.buildServerTLSConfig()
	server.StartTLS()
	t.Cleanup(server.Close)
	return owned
}

func (f *peerWalkFixture) install(t *testing.T, m *Manager, guard *peerDialGuard, reverse func(context.Context, string, string, noderec.ServiceKey) (net.Conn, error)) {
	t.Helper()
	opts := m.peerServiceOptions()
	opts.DirectTimeout = 2 * time.Second
	opts.DialReverse = reverse
	replaceClusterPeerServices(t, m, opts, guard)
}

func (f *peerWalkFixture) addTail(t *testing.T) {
	t.Helper()
	f.a.browser = newBrowser()
	f.a.browser.seed(f.b.identity.NodeUUID, "100.90.0.8", f.b.port)
	f.guard.allow(t, f.tail, f.listener)
}

func assertWalkUnchanged(t *testing.T, f *peerWalkFixture) {
	t.Helper()
	member, ok := f.a.memberByNodeID(f.b.identity.NodeUUID)
	disk := storedHintMember(t, f.a, f.b.identity.NodeUUID)
	pin, pinned := f.a.trust.Get(f.b.identity.NodeUUID)
	if !ok || !pinned || member.State != stateMember || member.AdmissionEpoch != 1 || len(member.AdvertisedAddresses) != 0 || len(disk.AdvertisedAddresses) != 0 || pin.AdmissionEpoch != 1 || pin.CertFingerprint != f.b.identity.CertFingerprint {
		t.Fatalf("failed walk changed authority/hints: memory=%+v disk=%+v pin=%+v", member, disk, pin)
	}
	assertNoPeerServicesClaim(t, f.output.Bytes())
}

func TestPeerServicesLearnPersistReopenTailnet(t *testing.T) {
	f := newPeerWalkFixture(t)
	f.a.mesh.Refresh()
	incarnation, ok := f.a.mesh.PeerServiceIncarnation(f.b.identity.NodeUUID)
	if !ok {
		t.Fatal("original admission missing")
	}
	identity, fingerprint := f.a.identity.NodeUUID, f.a.identity.CertFingerprint
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := f.a.refreshPeerServices(ctx, f.b.identity.NodeUUID); err != nil {
		t.Fatal(err)
	}
	if learned := storedHintMember(t, f.a, f.b.identity.NodeUUID); !slices.Equal(learned.AdvertisedAddresses, f.descriptor.Addresses) {
		t.Fatalf("authenticated learning not durable: %+v", learned)
	}
	if f.guard.attempted(f.lan) == 0 || f.handlers.Load() != 1 {
		t.Fatal("initial authenticated LAN read did not occur")
	}
	f.a.peerServices.CloseIdle()
	f.a.removalPeerServices.CloseIdle()
	f.a.clients.CloseIdle()
	if err := f.a.reverse.manager.Close(); err != nil {
		t.Fatal(err)
	}
	base := filepath.Dir(f.a.clusterDir)
	reopened, err := NewManager(NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), io.Discard}), base, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := reopened.reverse.manager.Close(); err != nil {
			t.Error(err)
		}
	})
	if reopened.identity.NodeUUID != identity || reopened.identity.CertFingerprint != fingerprint || filepath.Dir(reopened.clusterDir) != base || len(reopened.peerRoutes) != 0 {
		t.Fatal("reopen changed original identity/profile or retained route state")
	}
	if cid, epoch := reopened.currentAdmission(); cid != "cluster-1" || epoch != 1 {
		t.Fatalf("reopen admission %s/%d", cid, epoch)
	}
	pin, pinned := reopened.trust.Get(f.b.identity.NodeUUID)
	member, exists := reopened.memberByNodeID(f.b.identity.NodeUUID)
	if !pinned || pin.CertFingerprint != f.b.identity.CertFingerprint || pin.AdmissionEpoch != 1 || !exists || !slices.Equal(member.AdvertisedAddresses, f.descriptor.Addresses) {
		t.Fatal("reopen lost learned hints or original pin/member")
	}
	reopened.browser = newBrowser()
	if hosts, _, ok := reopened.browser.Resolve(f.b.identity.NodeUUID); ok || len(hosts) != 0 {
		t.Fatal("reopen browser is not empty")
	}
	guard := newPeerDialGuard(t)
	guard.allow(t, f.tail, f.listener) // LAN is deliberately unmapped and never reaches the OS.
	f.install(t, reopened, guard, nil)
	f.output.Reset()
	reopened.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), &f.output})
	reopened.serviceRefresh.source = "reopened-process"
	start := time.Now().UnixMilli()
	if err := reopened.refreshPeerServices(ctx, f.b.identity.NodeUUID); err != nil {
		t.Fatal(err)
	}
	message, err := NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(f.output.Bytes()), io.Discard}).Read()
	if err != nil {
		t.Fatal(err)
	}
	var claim noderec.PeerServicesClaim
	if err := json.Unmarshal(message.Params, &claim); err != nil {
		t.Fatal(err)
	}
	if message.Method != peerServicesEvent || claim.Descriptor.HostUUID != f.b.identity.NodeUUID || !slices.Equal(claim.Descriptor.Addresses, f.descriptor.Addresses) || !slices.Equal(claim.Descriptor.Services, f.descriptor.Services) || claim.Incarnation != incarnation || claim.Source != "reopened-process" || claim.Sequence != 1 || claim.ObservedAt < start || claim.ObservedAt > time.Now().UnixMilli() {
		t.Fatalf("reopen claim metadata=%+v", claim)
	}
	if guard.attempted(f.tail) == 0 || guard.physical.Load() == 0 || guard.reverse.Load() != 0 || f.handlers.Load() != 2 {
		t.Fatalf("tailnet bootstrap missing: tail=%d physical=%d reverse=%d handlers=%d", guard.attempted(f.tail), guard.physical.Load(), guard.reverse.Load(), f.handlers.Load())
	}
}

func TestPeerServicesWholeAttemptFairness(t *testing.T) {
	test := func(name string, stage string) {
		t.Run(name, func(t *testing.T) {
			f := newPeerWalkFixture(t)
			f.addTail(t)
			entered, exited := make(chan struct{}), make(chan struct{})
			bad := startWalkServer(t, f.b, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				close(entered)
				defer close(exited)
				if stage == "body" || stage == "EOF" {
					w.Header().Set("Content-Length", "100")
					w.WriteHeader(http.StatusOK)
					if _, err := w.Write([]byte("{")); err != nil {
						t.Error(err)
						return
					}
					w.(http.Flusher).Flush()
					if stage == "EOF" {
						return
					}
				}
				if stage == "invalid" {
					if _, err := w.Write([]byte(`{"unexpected":true}`)); err != nil {
						t.Error(err)
					}
					return
				}
				<-r.Context().Done()
			}))
			f.guard.allow(t, f.lan, bad)
			if stage == "setup" {
				f.guard.setup[f.lan] = func(ctx context.Context) error { close(entered); defer close(exited); <-ctx.Done(); return ctx.Err() }
			}
			var remaining time.Duration
			later := make(chan struct{})
			f.guard.setup[f.tail] = func(ctx context.Context) error {
				deadline, ok := ctx.Deadline()
				if ok {
					remaining = time.Until(deadline)
				}
				close(later)
				return ctx.Err()
			}
			ctx, cancel := context.WithTimeout(context.Background(), 1200*time.Millisecond)
			defer cancel()
			err := f.a.refreshPeerServices(ctx, f.b.identity.NodeUUID)
			select {
			case <-entered:
			default:
				t.Fatal("first candidate was not attempted")
			}
			select {
			case <-exited:
			case <-time.After(2 * time.Second):
				t.Fatal("stalled first attempt was not cancelled/joined")
			}
			if stage == "invalid" {
				if err == nil {
					t.Fatal("complete invalid descriptor accepted")
				}
				if f.guard.attempted(f.tail) != 0 {
					t.Fatal("complete invalid descriptor advanced")
				}
				assertWalkUnchanged(t, f)
				return
			}
			if err != nil {
				t.Fatalf("later authenticated candidate starved: %v", err)
			}
			select {
			case <-later:
			default:
				t.Fatal("later candidate did not receive an attempt")
			}
			if remaining <= 0 || ctx.Err() != nil || f.handlers.Load() != 1 {
				t.Fatalf("later request lacked parent time: remaining=%s parent=%v handlers=%d", remaining, ctx.Err(), f.handlers.Load())
			}
		})
	}
	test("blocking setup", "setup")
	test("stalled headers", "headers")
	test("incomplete body waits for cancellation", "body")
	test("incomplete body EOF", "EOF")
	test("complete invalid descriptor is terminal", "invalid")
}

func TestPeerServicesStaleReverseDoesNotStarveHint(t *testing.T) {
	f := newPeerWalkFixture(t)
	f.addTail(t)
	f.guard.setup[f.lan] = func(context.Context) error { return errors.New("synthetic direct unavailable") }
	entered, exited := make(chan struct{}), make(chan struct{})
	f.install(t, f.a, f.guard, func(ctx context.Context, _ string, _ string, _ noderec.ServiceKey) (net.Conn, error) {
		close(entered)
		defer close(exited)
		<-ctx.Done()
		return nil, ctx.Err()
	})
	ctx, cancel := context.WithTimeout(context.Background(), 1200*time.Millisecond)
	defer cancel()
	if err := f.a.refreshPeerServices(ctx, f.b.identity.NodeUUID); err != nil {
		t.Fatalf("stale reverse starved authenticated hint: %v", err)
	}
	select {
	case <-entered:
	default:
		t.Fatal("stale reverse was not attempted")
	}
	select {
	case <-exited:
	default:
		t.Fatal("stale reverse did not observe cancellation")
	}
	if f.guard.reverse.Load() != 1 || f.guard.attempted(f.tail) == 0 || ctx.Err() != nil || f.handlers.Load() != 1 {
		t.Fatalf("reverse=%d later=%d parent=%v handlers=%d", f.guard.reverse.Load(), f.guard.attempted(f.tail), ctx.Err(), f.handlers.Load())
	}
}

func TestPeerServicesCancelledWalkPreservesMembership(t *testing.T) {
	t.Run("parent cancelled during held chooser", func(t *testing.T) {
		f := newPeerWalkFixture(t)
		f.addTail(t)
		entered, release, returned := make(chan struct{}, 2), make(chan struct{}), make(chan struct{}, 2)
		var once sync.Once
		unblock := func() { once.Do(func() { close(release) }) }
		f.a.peerAddrs = reach.NewChooser(reach.WithDial(func(string, string, time.Duration) (net.Conn, error) {
			entered <- struct{}{}
			defer func() { returned <- struct{}{} }()
			<-release
			return nil, errors.New("synthetic held chooser refusal")
		}))
		ctx, cancel := context.WithCancel(context.Background())
		done, joined := make(chan error, 1), make(chan struct{})
		go func() { defer close(joined); done <- f.a.refreshPeerServices(ctx, f.b.identity.NodeUUID) }()
		defer func() {
			cancel()
			unblock()
			select {
			case <-joined:
			case <-time.After(2 * time.Second):
				t.Error("held-chooser refresh did not join")
			}
			for range 2 {
				select {
				case <-returned:
				case <-time.After(2 * time.Second):
					t.Error("held chooser probe did not join")
				}
			}
		}()
		for range 2 {
			select {
			case <-entered:
			case <-time.After(2 * time.Second):
				t.Fatal("chooser did not enter both synthetic probes")
			}
		}
		cancel()
		select {
		case err := <-done:
			if !errors.Is(err, context.Canceled) {
				t.Errorf("held chooser cancellation=%v", err)
			}
		case <-time.After(200 * time.Millisecond):
			t.Error("parent cancellation did not interrupt CM chooser wait")
		}
		f.guard.mu.Lock()
		logical := len(f.guard.logical)
		f.guard.mu.Unlock()
		if logical != 0 || f.guard.reverse.Load() != 0 || f.handlers.Load() != 0 {
			t.Fatal("cancelled confirmation dispatched descriptor work")
		}
		assertWalkUnchanged(t, f)
	})
	t.Run("parent cancelled during setup", func(t *testing.T) {
		f := newPeerWalkFixture(t)
		f.addTail(t)
		ctx, cancel := context.WithCancel(context.Background())
		entered := make(chan struct{})
		f.guard.setup[f.lan] = func(ctx context.Context) error { close(entered); <-ctx.Done(); return ctx.Err() }
		done := make(chan error, 1)
		joined := make(chan struct{})
		go func() { defer close(joined); done <- f.a.refreshPeerServices(ctx, f.b.identity.NodeUUID) }()
		defer func() {
			cancel()
			select {
			case <-joined:
			case <-time.After(3 * time.Second):
				t.Error("cancelled worker did not join")
			}
		}()
		select {
		case <-entered:
		case <-time.After(3 * time.Second):
			t.Fatal("walk did not enter setup")
		}
		cancel()
		select {
		case err := <-done:
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("cancel error=%v", err)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("cancelled walk did not stop")
		}
		if f.guard.attempted(f.tail) != 0 || f.handlers.Load() != 0 {
			t.Fatal("cancelled walk advanced")
		}
		assertWalkUnchanged(t, f)
	})
	t.Run("no endpoint succeeds", func(t *testing.T) {
		f := newPeerWalkFixture(t)
		f.addTail(t)
		f.guard.targets = make(map[string]string)
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if err := f.a.refreshPeerServices(ctx, f.b.identity.NodeUUID); err == nil {
			t.Fatal("unavailable walk succeeded")
		}
		if f.guard.attempted(f.lan) == 0 || f.guard.attempted(f.tail) == 0 || f.guard.physical.Load() != 0 || f.handlers.Load() != 0 {
			t.Fatal("synthetic unavailable candidates were not fully walked")
		}
		assertWalkUnchanged(t, f)
	})
}

// containHintDials fences both transport owners separately. These fixtures
// never need chooser probes or reverse IPC; HTTP uses only literal loopback.
func containHintDials(t *testing.T, m *Manager, wantHTTP bool) *peerDialGuard {
	t.Helper()
	g := newPeerDialGuard(t)
	opts := m.peerServiceOptions()
	replaceClusterPeerServices(t, m, opts, g)
	t.Cleanup(func() {
		g.mu.Lock()
		defer g.mu.Unlock()
		if g.reverse.Load() != 0 || (len(g.logical) > 0) != wantHTTP {
			t.Errorf("hint dial containment logical=%v reverse=%d wantHTTP=%v", g.logical, g.reverse.Load(), wantHTTP)
		}
	})
	return g
}

type peerHintFixture struct {
	m, remote   *Manager
	descriptor  noderec.PeerServicesDescriptor
	incarnation string
	peer        clustertrust.ReversePeer
	output      bytes.Buffer
}

func newPeerHintFixture(t *testing.T) *peerHintFixture {
	t.Helper()
	f := &peerHintFixture{m: newReverseTestManager(t), remote: newReverseTestManager(t)}
	containHintDials(t, f.m, false)
	containHintDials(t, f.remote, false)
	f.remote.port = 25421 // Data only: these managers never start listeners.
	pairReverseManagers(t, f.m, f.remote)
	f.m.setMemberAddr(f.remote.identity.NodeUUID, "192.0.2.55", f.remote.port)
	persistPeerServicesMembers(t, f.m, f.remote)
	f.m.mesh.Refresh()
	var ok bool
	f.incarnation, ok = f.m.mesh.PeerServiceIncarnation(f.remote.identity.NodeUUID)
	if !ok {
		t.Fatal("private hint fixture lacks durable admission")
	}
	f.peer, ok = f.m.reversePeer(f.remote.identity.NodeUUID)
	if !ok {
		t.Fatal("private hint fixture lacks current peer")
	}
	f.descriptor = noderec.PeerServicesDescriptor{
		HostUUID: f.remote.identity.NodeUUID, Name: "hint-peer",
		Addresses: []string{"192.0.2.55", "100.90.0.8"}, Services: []noderec.PeerService{},
	}
	if err := noderec.ValidatePeerServicesDescriptor(f.descriptor, f.peer.UUID); err != nil {
		t.Fatal(err)
	}
	f.m.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), &f.output})
	t.Cleanup(f.m.peerServices.CloseIdle)
	t.Cleanup(f.m.removalPeerServices.CloseIdle)
	return f
}

func (f *peerHintFixture) commit() error {
	return f.m.commitPeerServiceHints(f.descriptor, f.incarnation, f.peer)
}

func storedHintMember(t *testing.T, m *Manager, uuid string) ClusterNode {
	t.Helper()
	data, err := os.ReadFile(m.membersPath())
	if err != nil {
		t.Fatal(err)
	}
	var members []ClusterNode
	if err := json.Unmarshal(data, &members); err != nil {
		t.Fatal(err)
	}
	for _, member := range members {
		if member.NodeUUID == uuid {
			return member
		}
	}
	t.Fatalf("stored member %s missing", uuid)
	return ClusterNode{}
}

func assertPeerHints(t *testing.T, f *peerHintFixture, want []string) {
	t.Helper()
	member, ok := f.m.memberByNodeID(f.peer.UUID)
	if !ok || !slices.Equal(member.AdvertisedAddresses, want) {
		t.Fatalf("memory hints = %+v, want %v", member, want)
	}
	stored := storedHintMember(t, f.m, f.peer.UUID)
	if !slices.Equal(stored.AdvertisedAddresses, want) {
		t.Fatalf("disk hints = %v, want %v", stored.AdvertisedAddresses, want)
	}
}

func assertNoPeerServicesClaim(t *testing.T, output []byte) {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(output))
	for {
		var message Message
		if err := decoder.Decode(&message); err == io.EOF {
			return
		} else if err != nil {
			t.Fatal(err)
		}
		if message.Method == peerServicesEvent {
			t.Fatal("failed hint commit published a peer-services claim")
		}
	}
}

func reopenPeerHints(t *testing.T, f *peerHintFixture, want []string) {
	t.Helper()
	reopened, err := NewManager(NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), io.Discard}), filepath.Dir(f.m.clusterDir), 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := reopened.reverse.manager.Close(); err != nil {
			t.Error(err)
		}
	})
	t.Cleanup(reopened.peerServices.CloseIdle)
	t.Cleanup(reopened.removalPeerServices.CloseIdle)
	member, ok := reopened.memberByNodeID(f.peer.UUID)
	if !ok || !slices.Equal(member.AdvertisedAddresses, want) {
		t.Fatalf("original-base reopen hints = %+v, want %v", member, want)
	}
	if cid, epoch := reopened.currentAdmission(); cid != "cluster-1" || epoch != 1 {
		t.Fatalf("reopen admission = %s/%d", cid, epoch)
	}
	if pin, ok := reopened.trust.Get(f.peer.UUID); !ok || pin.AdmissionEpoch != 1 || pin.CertFingerprint != f.remote.identity.CertFingerprint {
		t.Fatalf("reopen changed original peer pin: %+v", pin)
	}
}

func TestMemberAdvertisedAddressesDeepCopy(t *testing.T) {
	f := newPeerHintFixture(t)
	member, _ := f.m.memberByNodeID(f.peer.UUID)
	member.AdvertisedAddresses = []string{"100.90.0.8"}
	f.m.upsertMember(member)
	member.AdvertisedAddresses[0] = "input mutation"
	lookup, _ := f.m.memberByNodeID(f.peer.UUID)
	lookup.AdvertisedAddresses[0] = "lookup mutation"
	for _, snapshot := range f.m.snapshotNodes() {
		if snapshot.NodeUUID == f.peer.UUID {
			snapshot.AdvertisedAddresses[0] = "snapshot mutation"
		}
	}
	if err := f.m.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	assertPeerHints(t, f, []string{"100.90.0.8"})
}

func TestMemberAdvertisedAddressesCompatibility(t *testing.T) {
	f := newPeerHintFixture(t)
	assertPeerHints(t, f, nil)
	reopenPeerHints(t, f, nil)
	for _, addresses := range [][]string{nil, {}} {
		member, _ := f.m.memberByNodeID(f.peer.UUID)
		member.AdvertisedAddresses = addresses
		data, err := json.Marshal(member)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil {
			t.Fatal(err)
		}
		if _, exists := fields["advertisedAddresses"]; exists {
			t.Fatal("nil or empty hints were not omitted")
		}
	}
}

func TestPeerServicesHintCommitBindsAdmission(t *testing.T) {
	test := func(name string, mutate func(*testing.T, *peerHintFixture), wantSuccess bool) {
		t.Run(name, func(t *testing.T) {
			f := newPeerHintFixture(t)
			f.m.rosterMu.Lock()
			locked := true
			done := make(chan error, 1)
			finished := make(chan struct{})
			go func() { defer close(finished); done <- f.commit() }()
			defer func() {
				if locked {
					f.m.rosterMu.Unlock()
				}
				select {
				case <-finished:
				case <-time.After(8 * time.Second):
					t.Error("hint worker did not join after boundary release")
				}
			}()
			deadline := time.Now().Add(5 * time.Second)
			for {
				stack := make([]byte, 256*1024)
				n := runtime.Stack(stack, true)
				blocked := false
				for _, worker := range strings.Split(string(stack[:n]), "\n\n") {
					if strings.Contains(worker, "commitPeerServiceHints") && strings.Contains(worker, "sync.runtime_SemacquireMutex") {
						blocked = true
					}
				}
				if blocked {
					break
				}
				if time.Now().After(deadline) {
					f.m.rosterMu.Unlock()
					locked = false
					t.Fatal("hint commit did not wait at composition boundary")
				}
				runtime.Gosched()
			}
			mutate(t, f)
			before := storedHintMember(t, f.m, f.peer.UUID)
			f.m.rosterMu.Unlock()
			locked = false
			select {
			case err := <-done:
				if (err == nil) != wantSuccess {
					t.Fatalf("commit error = %v, want success %v", err, wantSuccess)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("hint commit failed to finish")
			}
			if wantSuccess {
				assertPeerHints(t, f, f.descriptor.Addresses)
			} else {
				after := storedHintMember(t, f.m, f.peer.UUID)
				if !slices.Equal(after.AdvertisedAddresses, before.AdvertisedAddresses) || after.AdmissionEpoch != before.AdmissionEpoch {
					t.Fatalf("stale response changed replacement disk state: before=%+v after=%+v", before, after)
				}
				member, _ := f.m.memberByNodeID(f.peer.UUID)
				if !slices.Equal(member.AdvertisedAddresses, before.AdvertisedAddresses) {
					t.Fatal("stale response changed replacement memory hints")
				}
			}
			assertNoPeerServicesClaim(t, f.output.Bytes())
		})
	}
	test("current admission", func(*testing.T, *peerHintFixture) {}, true)
	test("same certificate newer peer admission", func(t *testing.T, f *peerHintFixture) {
		readmitClusterPeer(t, f.m, f.remote)
		member, _ := f.m.memberByNodeID(f.peer.UUID)
		member.AdvertisedAddresses = []string{"100.90.0.9"}
		f.m.upsertMember(member)
		if err := f.m.persistMembersErr(); err != nil {
			t.Fatal(err)
		}
	}, false)
	test("local teardown and rejoin", func(t *testing.T, f *peerHintFixture) {
		if err := f.m.teardownClusterLocalLocked(); err != nil {
			t.Fatal(err)
		}
		activateTestCluster(t, f.m, "cluster-1")
		f.m.addSelfMember()
		pinTrusted(t, f.m, f.peer.UUID, string(f.remote.identity.CertPEM), f.remote.identity.CertFingerprint)
		f.m.upsertMember(&ClusterNode{NodeUUID: f.peer.UUID, ID: "replacement", ClusterID: "cluster-1", AdmissionEpoch: 1, State: stateMember, AdvertisedAddresses: []string{"100.90.0.9"}})
		if err := f.m.persistMembersErr(); err != nil {
			t.Fatal(err)
		}
	}, false)
	test("revoked pin", func(t *testing.T, f *peerHintFixture) { f.m.trust.Forget(f.peer.UUID) }, false)
	test("pending member", func(t *testing.T, f *peerHintFixture) {
		member, _ := f.m.memberByNodeID(f.peer.UUID)
		member.State = statePendingOutbound
		f.m.upsertMember(member)
	}, false)
}

func TestPeerServicesHintPersistenceFailure(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	guard := containHintDials(t, a, true)
	containHintDials(t, b, false)
	t.Cleanup(a.peerServices.CloseIdle)
	t.Cleanup(a.removalPeerServices.CloseIdle)
	setDescriptorSelf(t, b)
	setReverseServices(t, b, 1)
	server := httptest.NewUnstartedServer(http.HandlerFunc(b.handlePeerServices))
	owned := &ownedPeerListener{Listener: server.Listener}
	server.Listener = owned
	server.TLS = b.buildServerTLSConfig()
	server.StartTLS()
	defer server.Close()
	b.port = server.Listener.Addr().(*net.TCPAddr).Port
	guard.allow(t, server.Listener.Addr().String(), owned)
	pairReverseManagers(t, a, b)
	persistPeerServicesMembers(t, a, b)
	var output bytes.Buffer
	a.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), &output})
	failure := errors.New("injected member persistence failure")
	injectClusterReplacer(t, durablefile.Replacer{CreateTemp: func(dir, base string, mode os.FileMode) (durablefile.File, error) {
		if dir == a.clusterDir && base == membersFile {
			return nil, failure
		}
		return createHintTemp(dir, base, mode)
	}})
	if err := a.refreshPeerServices(context.Background(), b.identity.NodeUUID); !errors.Is(err, failure) {
		t.Fatalf("fetch persistence error = %v, want injected failure", err)
	}
	member, _ := a.memberByNodeID(b.identity.NodeUUID)
	if len(member.AdvertisedAddresses) != 0 || len(storedHintMember(t, a, b.identity.NodeUUID).AdvertisedAddresses) != 0 {
		t.Fatal("failed fetch retained speculative hints")
	}
	assertNoPeerServicesClaim(t, output.Bytes())
}

func TestPeerServicesHintsClearedWithMembership(t *testing.T) {
	test := func(name string, clear func(*testing.T, *peerHintFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newPeerHintFixture(t)
			if err := f.commit(); err != nil {
				t.Fatal(err)
			}
			clear(t, f)
			activateTestCluster(t, f.m, "cluster-1")
			f.m.addSelfMember()
			if err := f.m.trust.Pin(&TrustedPin{NodeUUID: f.peer.UUID, NodeID: f.remote.identity.NodeID, ClusterID: "cluster-1", AdmissionEpoch: 2, CertPem: string(f.remote.identity.CertPEM), CertFingerprint: f.remote.identity.CertFingerprint}); err != nil {
				t.Fatal(err)
			}
			f.m.upsertMember(&ClusterNode{NodeUUID: f.peer.UUID, ID: "new-admission", ClusterID: "cluster-1", AdmissionEpoch: 2, State: stateMember})
			if err := f.m.persistMembersErr(); err != nil {
				t.Fatal(err)
			}
			assertPeerHints(t, f, nil)
			reopened, err := NewManager(NewCodec(struct {
				io.Reader
				io.Writer
			}{bytes.NewReader(nil), io.Discard}), filepath.Dir(f.m.clusterDir), 0)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := reopened.reverse.manager.Close(); err != nil {
					t.Error(err)
				}
			})
			t.Cleanup(reopened.peerServices.CloseIdle)
			t.Cleanup(reopened.removalPeerServices.CloseIdle)
			node, ok := reopened.memberByNodeID(f.peer.UUID)
			if !ok || len(node.AdvertisedAddresses) != 0 {
				t.Fatalf("reopen leaked deleted hints: %+v", node)
			}
		})
	}
	test("remove member", func(t *testing.T, f *peerHintFixture) { f.m.removeMemberByUUID(f.peer.UUID) })
	test("reset membership", func(t *testing.T, f *peerHintFixture) { f.m.resetMembership() })
	test("local teardown", func(t *testing.T, f *peerHintFixture) {
		if err := f.m.teardownClusterLocal(); err != nil {
			t.Fatal(err)
		}
	})
}

func TestDescriptorAddresses(t *testing.T) {
	test := func(name string, scanner, overlay, want []string) {
		t.Run(name, func(t *testing.T) {
			originalScanner, originalOverlay := slices.Clone(scanner), slices.Clone(overlay)
			got := descriptorAddresses(scanner, overlay)
			if !slices.Equal(got, want) {
				t.Fatalf("descriptor addresses = %v, want %v", got, want)
			}
			if again := descriptorAddresses(scanner, overlay); !slices.Equal(again, got) {
				t.Fatalf("repeated assembly = %v, want stable order %v", again, got)
			}
			if !slices.Equal(scanner, originalScanner) || !slices.Equal(overlay, originalOverlay) {
				t.Fatalf("assembly mutated inputs: scanner=%v overlay=%v", scanner, overlay)
			}
			descriptor := noderec.PeerServicesDescriptor{
				HostUUID: "descriptor-test", Name: "descriptor", Addresses: got,
				Services: []noderec.PeerService{},
			}
			err := noderec.ValidatePeerServicesDescriptor(descriptor, descriptor.HostUUID)
			if len(want) == 0 {
				if err == nil {
					t.Fatal("overlay fabricated a valid descriptor without scanner addresses")
				}
			} else if err != nil {
				t.Fatalf("assembled descriptor rejected: %v", err)
			}
			if len(got) > 0 {
				got[0] = "changed output"
				if !slices.Equal(scanner, originalScanner) || !slices.Equal(overlay, originalOverlay) {
					t.Fatal("assembled output aliases an input")
				}
			}
		})
	}
	test("LAN plus overlay", []string{"192.168.42.8"}, []string{"100.91.248.18"}, []string{"192.168.42.8", "100.91.248.18"})
	test("duplicate overlap", []string{"192.168.42.8", "100.91.248.18", "192.168.42.8"}, []string{"100.91.248.18", "100.91.248.18"}, []string{"192.168.42.8", "100.91.248.18"})
	test("full cap reserves overlay", []string{"192.168.42.8", "192.168.42.9", "192.168.42.10", "192.168.42.11"}, []string{"100.91.248.18"}, []string{"192.168.42.8", "192.168.42.9", "192.168.42.10", "100.91.248.18"})
	test("full cap keeps only first missing overlay", []string{"192.168.42.8", "192.168.42.9", "192.168.42.10", "192.168.42.11"}, []string{"100.91.248.19", "100.91.248.18"}, []string{"192.168.42.8", "192.168.42.9", "192.168.42.10", "100.91.248.18"})
	test("multiple overlays limited room", []string{"192.168.42.8", "192.168.42.9"}, []string{"100.91.248.20", "100.91.248.18", "100.91.248.19"}, []string{"192.168.42.8", "192.168.42.9", "100.91.248.18", "100.91.248.19"})
	test("no overlay", []string{"192.168.42.8", "192.168.42.9", "192.168.42.8"}, nil, []string{"192.168.42.8", "192.168.42.9"})
	test("empty scanner", nil, []string{"100.91.248.18"}, nil)
	test("empty inputs", nil, nil, nil)
}

func setDescriptorSelf(t *testing.T, m *Manager) {
	t.Helper()
	m.setSelfServicesIdentity([]noderec.DirectoryNode{{HostUUID: m.identity.NodeUUID, Name: "descriptor-host", IP: "192.168.42.8", IPs: []string{"192.168.42.8"}}})
}

func persistPeerServicesMembers(t *testing.T, managers ...*Manager) {
	t.Helper()
	for _, manager := range managers {
		if err := manager.persistMembersErr(); err != nil {
			t.Fatalf("persist descriptor fixture membership: %v", err)
		}
	}
}

func TestPeerServicesSelfUsesCurrentLocalRegistrations(t *testing.T) {
	m := newReverseTestManager(t)
	setDescriptorSelf(t, m)
	setReverseServices(t, m, 30, noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 22441}, noderec.RegisterParams{Service: noderec.ServiceEngineControl, Port: 22442})
	d, err := m.selfServicesDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	if d.HostUUID != m.identity.NodeUUID || d.Addresses[0] != "192.168.42.8" || len(d.Services) != 2 {
		t.Fatalf("self snapshot=%+v", d)
	}
	setReverseServices(t, m, 31, noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 22443})
	d, err = m.selfServicesDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Services) != 1 || d.Services[0].Service != noderec.ServiceNodeInfo || d.Services[0].Port != 22443 {
		t.Fatalf("withdrawal/rebind=%+v", d.Services)
	}
	setReverseServices(t, m, 32)
	d, err = m.selfServicesDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	if d.Services == nil || len(d.Services) != 0 {
		t.Fatalf("complete withdrawal=%+v", d.Services)
	}
}

func TestPeerServicesRequiresSelfIdentityAndInitializedRegistry(t *testing.T) {
	m := newReverseTestManager(t)
	setDescriptorSelf(t, m)
	if _, err := m.selfServicesDescriptor(); err == nil {
		t.Fatal("uninitialized registrations accepted")
	}
	setReverseServices(t, m, 0)
	m.setSelfServicesIdentity([]noderec.DirectoryNode{{HostUUID: "other", Name: "other", IP: "192.168.42.9"}})
	if _, err := m.selfServicesDescriptor(); err == nil {
		t.Fatal("third party identity accepted")
	}
	m.setSelfServicesIdentity([]noderec.DirectoryNode{{HostUUID: m.identity.NodeUUID, Name: "self", IP: "127.0.0.1"}})
	if _, err := m.selfServicesDescriptor(); err == nil {
		t.Fatal("loopback self descriptor accepted")
	}
	m.setSelfServicesIdentity([]noderec.DirectoryNode{{HostUUID: m.identity.NodeUUID, Name: "self"}})
	if _, err := m.selfServicesDescriptor(); err == nil {
		t.Fatal("missing scanner addresses accepted")
	}
	m.setSelfServicesIdentity([]noderec.DirectoryNode{{HostUUID: m.identity.NodeUUID, Name: "self", IPs: []string{"192.168.42.8", "192.168.42.9", "192.168.42.10", "127.0.0.1"}}})
	if _, err := m.selfServicesDescriptor(); err == nil {
		t.Fatal("invalid scanner address hidden by overlay reservation")
	}
}

func TestPeerServicesEndpointRequiresCurrentAdmission(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	pairReverseManagers(t, a, b)
	persistPeerServicesMembers(t, a, b)
	setDescriptorSelf(t, b)
	setReverseServices(t, b, 1, noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 23441})
	cert, err := x509.ParseCertificate(a.identity.Cert.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	request := func() *http.Request {
		r := httptest.NewRequest(http.MethodGet, peerServicesPath, nil)
		r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{cert}}
		return r
	}
	w := httptest.NewRecorder()
	b.handlePeerServices(w, request())
	if w.Code != http.StatusOK {
		t.Fatalf("current peer status=%d body=%s", w.Code, w.Body)
	}
	d, err := noderec.DecodePeerServices(w.Body, b.identity.NodeUUID)
	if err != nil {
		t.Fatal(err)
	}
	if d.Services[0].Port != 23441 {
		t.Fatalf("services=%+v", d.Services)
	}
	b.removeMemberByUUID(a.identity.NodeUUID)
	persistPeerServicesMembers(t, b)
	w = httptest.NewRecorder()
	b.handlePeerServices(w, request())
	if w.Code != http.StatusForbidden {
		t.Fatalf("removed admitted member status=%d", w.Code)
	}
}

func TestPeerServicesEndpointRejectsBodyAndQuery(t *testing.T) {
	m := newReverseTestManager(t)
	test := func(name, method, url string, body io.Reader, want int) {
		t.Run(name, func(t *testing.T) {
			r := httptest.NewRequest(method, url, body)
			w := httptest.NewRecorder()
			m.handlePeerServices(w, r)
			if w.Code != want {
				t.Fatalf("status=%d want=%d", w.Code, want)
			}
		})
	}
	test("post", http.MethodPost, peerServicesPath, nil, http.StatusMethodNotAllowed)
	test("body", http.MethodGet, peerServicesPath, bytes.NewBufferString("{}"), http.StatusBadRequest)
	test("query", http.MethodGet, peerServicesPath+"?peer=other", nil, http.StatusBadRequest)
	test("unpaired", http.MethodGet, peerServicesPath, nil, http.StatusForbidden)
}

func TestPeerServicesFetchBindsOriginalAdmission(t *testing.T) {
	test := func(name string, retire bool) {
		t.Run(name, func(t *testing.T) {
			a, b := newReverseTestManager(t), newReverseTestManager(t)
			guard := containHintDials(t, a, true)
			containHintDials(t, b, false)
			setDescriptorSelf(t, b)
			setReverseServices(t, b, 1, noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 24441})
			held := make(chan struct{})
			release, releaseHeld := context.WithCancel(context.Background())
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if retire {
					close(held)
					select {
					case <-release.Done():
					case <-r.Context().Done():
						return
					}
				}
				b.handlePeerServices(w, r)
			}))
			server.TLS = b.buildServerTLSConfig()
			owned := &ownedPeerListener{Listener: server.Listener}
			server.Listener = owned
			server.StartTLS()
			defer server.Close()
			defer releaseHeld()
			b.port = server.Listener.Addr().(*net.TCPAddr).Port
			guard.allow(t, server.Listener.Addr().String(), owned)
			pairReverseManagers(t, a, b)
			persistPeerServicesMembers(t, a, b)
			a.mesh.Refresh()
			originalIncarnation, admitted := a.mesh.PeerServiceIncarnation(b.identity.NodeUUID)
			if !admitted {
				t.Fatal("paired fixture has no durable admission")
			}
			var published bytes.Buffer
			a.codec = NewCodec(struct {
				io.Reader
				io.Writer
			}{bytes.NewReader(nil), &published})
			a.serviceRefresh.source = "test-process"
			result := make(chan error, 1)
			go func() { result <- a.refreshPeerServices(context.Background(), b.identity.NodeUUID) }()
			if retire {
				select {
				case <-held:
				case err := <-result:
					t.Fatalf("descriptor fetch completed before held response: %v", err)
				case <-time.After(5 * time.Second):
					t.Fatal("descriptor fetch did not reach held response")
				}
				// Retire and re-admit the same certificate. A pin-only comparison
				// would accept the held response as fresh work for the new epoch.
				a.removeMemberByUUID(b.identity.NodeUUID)
				persistPeerServicesMembers(t, a)
				epoch, err := b.reserveAdmissionEpoch()
				if err != nil {
					t.Fatalf("reserve same-certificate readmission: %v", err)
				}
				if epoch != 2 {
					t.Fatalf("readmission epoch = %d, want 2", epoch)
				}
				if err := b.activateAdmission("cluster-1", epoch); err != nil {
					t.Fatalf("activate same-certificate readmission: %v", err)
				}
				b.addSelfMember()
				persistPeerServicesMembers(t, b)
				node := &ClusterNode{NodeUUID: b.identity.NodeUUID, ID: b.identity.NodeID, IPAddress: "127.0.0.1", Port: b.port, ClusterID: "cluster-1", AdmissionEpoch: 2, State: stateMember}
				a.upsertMember(node)
				pin, ok := a.trust.Get(b.identity.NodeUUID)
				if !ok {
					t.Fatal("peer pin disappeared")
				}
				pin.AdmissionEpoch = 2
				if err := a.trust.Pin(pin); err != nil {
					t.Fatal(err)
				}
				persistPeerServicesMembers(t, a)
				a.mesh.Refresh()
				currentIncarnation, admitted := a.mesh.PeerServiceIncarnation(b.identity.NodeUUID)
				if !admitted || currentIncarnation == originalIncarnation {
					t.Fatal("readmission did not establish a distinct current durable incarnation")
				}
				releaseHeld()
			}
			var err error
			select {
			case err = <-result:
			case <-time.After(10 * time.Second):
				t.Fatal("descriptor fetch did not finish")
			}
			if retire {
				if err == nil {
					t.Fatal("held response accepted after same-certificate readmission")
				}
				codec := NewCodec(struct {
					io.Reader
					io.Writer
				}{bytes.NewReader(published.Bytes()), io.Discard})
				for {
					msg, readErr := codec.Read()
					if readErr == io.EOF {
						break
					}
					if readErr != nil {
						t.Fatal(readErr)
					}
					if msg.Method == peerServicesEvent {
						t.Fatal("held response published an obsolete services claim")
					}
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			msg, err := NewCodec(struct {
				io.Reader
				io.Writer
			}{bytes.NewReader(published.Bytes()), io.Discard}).Read()
			if err != nil {
				t.Fatal(err)
			}
			var claim noderec.PeerServicesClaim
			if err := json.Unmarshal(msg.Params, &claim); err != nil {
				t.Fatal(err)
			}
			if msg.Method != peerServicesEvent || claim.Descriptor.HostUUID != b.identity.NodeUUID || claim.Descriptor.Services[0].Port != 24441 || claim.Incarnation == "" || claim.Sequence != 1 || claim.Source != "test-process" {
				t.Fatalf("claim=%+v method=%s", claim, msg.Method)
			}
		})
	}
	test("current pinned direct peer", false)
	test("same certificate readmission rejects held response", true)
}

func TestPeerServicesBlockedDirectUsesRealReverse(t *testing.T) {
	a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, true)
	pairClusterFixtures(t, a, b)
	persistPeerServicesMembers(t, a.m, b.m)
	setDescriptorSelf(t, b.m)
	inner := httptest.NewUnstartedServer(http.HandlerFunc(b.m.handlePeerServices))
	inner.Listener = reverseForwardListener{Listener: inner.Listener, manager: b.m}
	inner.Config.ConnContext = b.m.reverseConnContext
	inner.TLS = b.m.buildServerTLSConfig()
	inner.StartTLS()
	defer inner.Close()
	setReverseServices(t, b.m, 2, noderec.RegisterParams{Service: noderec.ServiceCluster, Port: inner.Listener.Addr().(*net.TCPAddr).Port}, noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 26441})
	startPrivateReverse(t, a.m)
	connectReverseManagers(t, a.m, b.m)
	reverse := countClusterReverseDials(t, a)
	var published bytes.Buffer
	a.m.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), &published})
	a.m.serviceRefresh.source = "reverse-process"
	if err := a.m.refreshPeerServices(context.Background(), b.m.identity.NodeUUID); err != nil {
		t.Fatal(err)
	}
	msg, err := NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(published.Bytes()), io.Discard}).Read()
	if err != nil {
		t.Fatal(err)
	}
	var claim noderec.PeerServicesClaim
	if err := json.Unmarshal(msg.Params, &claim); err != nil {
		t.Fatal(err)
	}
	if reverse.Load() != 1 || claim.Descriptor.Addresses[0] != "192.168.42.8" || len(claim.Descriptor.Services) != 2 {
		t.Fatalf("reverse calls=%d descriptor=%+v", reverse.Load(), claim.Descriptor)
	}
	member, ok := b.m.memberByNodeID(a.m.identity.NodeUUID)
	if !ok || member.IPAddress != "localhost" || member.Port != a.m.port {
		t.Fatalf("reverse gateway poisoned member locator: %+v", member)
	}
}

func TestPeerServicesUnavailableRoutePreservesMembership(t *testing.T) {
	a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, true)
	pairClusterFixtures(t, a, b)
	persistPeerServicesMembers(t, a.m, b.m)
	startPrivateReverse(t, a.m)
	countClusterReverseDials(t, a)
	before, ok := a.m.mesh.PeerServiceIncarnation(b.m.identity.NodeUUID)
	if !ok {
		t.Fatal("missing initial authority")
	}
	if err := a.m.refreshPeerServices(context.Background(), b.m.identity.NodeUUID); err == nil {
		t.Fatal("unavailable direct/reverse route succeeded")
	}
	a.m.mesh.Refresh()
	after, ok := a.m.mesh.PeerServiceIncarnation(b.m.identity.NodeUUID)
	if !ok || after != before {
		t.Fatal("route failure altered durable admission/pin")
	}
	member, ok := a.m.memberByNodeID(b.m.identity.NodeUUID)
	if !ok || member.State != stateMember {
		t.Fatalf("route failure altered membership: %+v", member)
	}
}
