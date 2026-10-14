// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

type clusterPeerFixture struct {
	m           *Manager
	address     string
	rosters     atomic.Int32
	removes     atomic.Int32
	rosterReply http.HandlerFunc
	// afterRemove runs after the real handler finishes durable teardown.
	afterRemove func()
}

func newClusterPeerFixture(t *testing.T, blocked bool) *clusterPeerFixture {
	t.Helper()
	f := &clusterPeerFixture{m: newReverseTestManager(t)}
	t.Cleanup(f.m.peerServices.CloseIdle)
	t.Cleanup(f.m.removalPeerServices.CloseIdle)
	mux := http.NewServeMux()
	mux.HandleFunc(clustertrust.ReverseUpgradePath, f.m.handleReverseUpgrade)
	mux.HandleFunc(rosterPath, func(w http.ResponseWriter, r *http.Request) {
		f.rosters.Add(1)
		if f.rosterReply != nil {
			f.rosterReply(w, r)
			return
		}
		f.m.handleRoster(w, r)
	})
	mux.HandleFunc(membersRemovePath, func(w http.ResponseWriter, r *http.Request) {
		f.removes.Add(1)
		f.m.handleMembersRemove(w, r)
		if f.afterRemove != nil {
			f.afterRemove()
		}
	})
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	f.m.port = listener.Addr().(*net.TCPAddr).Port
	f.address = net.JoinHostPort("localhost", strconv.Itoa(f.m.port))
	f.m.addSelfMember()
	if blocked {
		listener = rejectingPairingListener{Listener: listener}
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- f.m.serveSplit(ctx, listener, http.NewServeMux(), mux) }()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-done:
			if err != nil {
				t.Error(err)
			}
		case <-time.After(5 * time.Second):
			t.Error("cluster listener shutdown stalled")
		}
	})
	// The gateway's service registration is loopback-only and differs from the
	// peer's original URL. The inner handshake still uses the peer's exact pin.
	inner := httptest.NewUnstartedServer(mux)
	inner.Listener = reverseForwardListener{Listener: inner.Listener, manager: f.m}
	inner.Config.ConnContext = f.m.reverseConnContext
	inner.TLS = f.m.buildServerTLSConfig()
	inner.StartTLS()
	t.Cleanup(inner.Close)
	setReverseServices(t, f.m, 1, noderec.RegisterParams{
		Service: noderec.ServiceCluster, Port: inner.Listener.Addr().(*net.TCPAddr).Port,
	})
	return f
}

func pairClusterFixtures(t *testing.T, a, b *clusterPeerFixture) {
	t.Helper()
	pairReverseManagers(t, a.m, b.m)
	if !a.m.setMemberAddr(b.m.identity.NodeUUID, "localhost", b.m.port) ||
		!b.m.setMemberAddr(a.m.identity.NodeUUID, "localhost", a.m.port) {
		t.Fatal("set public cluster addresses")
	}
}

func countClusterReverseDials(t *testing.T, m *Manager) *atomic.Int32 {
	t.Helper()
	count := new(atomic.Int32)
	opts := m.peerServiceOptions()
	opts.DirectTimeout = 80 * time.Millisecond
	opts.DialReverse = func(ctx context.Context, base, uuid string, service noderec.ServiceKey) (net.Conn, error) {
		if service != noderec.ServiceCluster {
			t.Errorf("reverse service = %q, want cluster", service)
		}
		count.Add(1)
		return clustertrust.DialReverse(ctx, base, uuid, service)
	}
	replaceClusterPeerServices(t, m, opts)
	return count
}

func replaceClusterPeerServices(t *testing.T, m *Manager, opts clustertrust.PeerServiceClientOptions) {
	t.Helper()
	m.peerServices.CloseIdle()
	m.removalPeerServices.CloseIdle()
	opts.ReuseConnections = true
	m.peerServices = clustertrust.NewPeerServiceClient(m.mesh, opts)
	opts.ReuseConnections = false
	m.removalPeerServices = clustertrust.NewPeerServiceClient(m.mesh, opts)
	t.Cleanup(m.peerServices.CloseIdle)
	t.Cleanup(m.removalPeerServices.CloseIdle)
}

func TestClusterRosterDirectWorksWithoutReverseSession(t *testing.T) {
	a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, false)
	pairClusterFixtures(t, a, b)
	reverse := countClusterReverseDials(t, a.m)
	outcome, proven := a.m.reconcileWith([]string{b.address}, b.m.identity.NodeUUID)
	if outcome != reconcileAccepted || proven || b.rosters.Load() != 1 || reverse.Load() != 0 {
		t.Fatalf("outcome=%v proof=%v rosters=%d reverse=%d, want accepted/false/1/0", outcome, proven, b.rosters.Load(), reverse.Load())
	}
}

func TestClusterRosterFailsOverBetweenCurrentEndpoints(t *testing.T) {
	a, b, blocked := newClusterPeerFixture(t, false), newClusterPeerFixture(t, false), newClusterPeerFixture(t, true)
	pairClusterFixtures(t, a, b)
	a.m.setMemberAddr(b.m.identity.NodeUUID, "localhost", blocked.m.port)
	a.m.browser = newBrowser()
	a.m.browser.seed(b.m.identity.NodeUUID, "localhost", b.m.port)
	reverse := countClusterReverseDials(t, a.m)
	outcome, proven := a.m.reconcileWith([]string{blocked.address, b.address}, b.m.identity.NodeUUID)
	if outcome != reconcileAccepted || proven || b.rosters.Load() != 1 || reverse.Load() != 1 {
		t.Fatalf("endpoint failover outcome=%v proof=%v roster=%d reverse=%d, want accepted/false/1/1", outcome, proven, b.rosters.Load(), reverse.Load())
	}
}

func TestClusterRosterDoesNotReplayExposedResponse(t *testing.T) {
	a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, false)
	pairClusterFixtures(t, a, b)
	b.rosterReply = func(w http.ResponseWriter, r *http.Request) {
		b.m.handleRoster(httptest.NewRecorder(), r)
		w.Header().Set("Content-Length", "100")
		w.WriteHeader(http.StatusOK)
		if _, err := w.Write([]byte("{")); err != nil {
			t.Error(err)
			return
		}
		w.(http.Flusher).Flush()
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		if err := conn.Close(); err != nil {
			t.Error(err)
		}
	}
	reverse := countClusterReverseDials(t, a.m)
	// A second candidate must not receive a retry after the 200 headers and
	// partial body have been exposed, even though the first body is unusable.
	other := net.JoinHostPort("127.0.0.1", strconv.Itoa(b.m.port))
	a.m.browser = newBrowser()
	a.m.browser.seed(b.m.identity.NodeUUID, "127.0.0.1", b.m.port)
	outcome, proven := a.m.reconcileWith([]string{b.address, other}, b.m.identity.NodeUUID)
	if outcome != reconcileAccepted || proven || b.rosters.Load() != 1 || reverse.Load() != 0 {
		t.Fatalf("exposed response outcome=%v proof=%v roster=%d reverse=%d, want accepted/false/1/0", outcome, proven, b.rosters.Load(), reverse.Load())
	}
}

func TestClusterRosterBlockedTLSUsesRealReverseSession(t *testing.T) {
	a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, true)
	pairClusterFixtures(t, a, b)
	startPrivateReverse(t, a.m)
	connectReverseManagers(t, a.m, b.m)
	reverse := countClusterReverseDials(t, a.m)
	for range 2 {
		outcome, proven := a.m.reconcileWith([]string{b.address}, b.m.identity.NodeUUID)
		if outcome != reconcileAccepted || proven {
			t.Fatalf("reverse roster outcome=%v proof=%v, want accepted/false", outcome, proven)
		}
	}
	if b.rosters.Load() != 2 || reverse.Load() != 1 {
		t.Fatalf("rosters=%d reverse=%d, want 2 rosters on 1 reverse connection", b.rosters.Load(), reverse.Load())
	}
	// The tunneled listener sees the gateway's loopback source. It must not
	// replace the reachable peer's original public address with that source.
	node, ok := b.m.memberByNodeID(a.m.identity.NodeUUID)
	if !ok || node.IPAddress != "localhost" || node.Port != a.m.port {
		t.Fatalf("reverse roster corrupted public peer address: %+v", node)
	}
	startPrivateReverse(t, b.m)
	if outcome, proven := b.m.reconcileWith([]string{a.address}, a.m.identity.NodeUUID); outcome != reconcileAccepted || proven {
		t.Fatalf("return roster outcome=%v proof=%v, want accepted/false", outcome, proven)
	}
}

func TestClusterRosterUnavailableRouteDoesNotRemoveMembership(t *testing.T) {
	a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, true)
	pairClusterFixtures(t, a, b)
	startPrivateReverse(t, a.m)
	reverse := countClusterReverseDials(t, a.m)
	a.m.reconcilePeersAndMaybeSelfRemove()
	if cid, epoch := a.m.currentAdmission(); cid != "cluster-1" || epoch != 1 {
		t.Fatalf("route failure changed admission: (%q,%d), want cluster-1/1", cid, epoch)
	}
	if _, ok := a.m.trust.Get(b.m.identity.NodeUUID); !ok {
		t.Fatal("route failure removed peer pin")
	}
	if node, ok := a.m.memberByNodeID(b.m.identity.NodeUUID); !ok || node.State != stateMember {
		t.Fatalf("route failure removed member: %+v", node)
	}
	if reverse.Load() != 1 || b.rosters.Load() != 0 {
		t.Fatalf("reverse=%d rosters=%d, want attempted unavailable route/0 dispatches", reverse.Load(), b.rosters.Load())
	}
}

func TestClusterRosterDisconnectDoesNotLearnGatewaySource(t *testing.T) {
	a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, true)
	pairClusterFixtures(t, a, b)
	startPrivateReverse(t, a.m)
	connectReverseManagers(t, a.m, b.m)
	finished := make(chan struct{})
	b.rosterReply = func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			close(finished)
			return
		}
		if err := r.Body.Close(); err != nil {
			t.Error(err)
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		// The body is already dispatched, but the reverse session vanishes
		// before the roster owner records its authenticated source address.
		b.m.reverse.manager.Close()
		b.m.handleRoster(w, r)
		close(finished)
	}
	a.m.reconcileWith([]string{b.address}, b.m.identity.NodeUUID)
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("disconnected roster did not finish")
	}
	node, ok := b.m.memberByNodeID(a.m.identity.NodeUUID)
	if !ok || node.IPAddress != "localhost" || node.Port != a.m.port || b.rosters.Load() != 1 {
		t.Fatalf("gateway source learned after disconnect: member=%+v dispatched=%d", node, b.rosters.Load())
	}
}

func TestClusterRosterPairwiseThreePeerFanout(t *testing.T) {
	a, b, c := newClusterPeerFixture(t, false), newClusterPeerFixture(t, true), newClusterPeerFixture(t, false)
	pairClusterFixtures(t, a, b)
	pairClusterFixtures(t, a, c)
	// A's authenticated endorsement lets B learn C, and C learn B. They must
	// then exchange directly with one another rather than routing through A.
	for _, peer := range []*Manager{b.m, c.m} {
		if err := a.m.trust.AddEndorsements(peer.identity.NodeUUID, []Endorsement{
			a.m.endorsePeer(peer.identity.NodeUUID, peer.identity.CertFingerprint, 1),
		}); err != nil {
			t.Fatal(err)
		}
	}
	startPrivateReverse(t, a.m)
	startPrivateReverse(t, c.m)
	connectReverseManagers(t, a.m, b.m)
	for _, target := range []*clusterPeerFixture{b, c} {
		if outcome, _ := a.m.reconcileWith([]string{target.address}, target.m.identity.NodeUUID); outcome != reconcileAccepted {
			t.Fatalf("A fanout to %s = %v, want accepted", target.m.identity.NodeUUID, outcome)
		}
	}
	for _, local := range []*clusterPeerFixture{a, b, c} {
		for _, remote := range []*clusterPeerFixture{a, b, c} {
			if local == remote {
				continue
			}
			if _, ok := local.m.trust.Get(remote.m.identity.NodeUUID); !ok {
				t.Fatalf("%s did not learn pin %s", local.m.identity.NodeUUID, remote.m.identity.NodeUUID)
			}
			member, ok := local.m.memberByNodeID(remote.m.identity.NodeUUID)
			if !ok || member.State != stateMember || member.AdmissionEpoch != 1 || member.ClusterID != "cluster-1" {
				t.Fatalf("fanout did not converge member %s: %+v", remote.m.identity.NodeUUID, member)
			}
		}
	}
	connectReverseManagers(t, c.m, b.m)
	reverse := countClusterReverseDials(t, c.m)
	aBefore := a.rosters.Load()
	bBefore := b.rosters.Load()
	if outcome, _ := c.m.reconcileWith([]string{b.address}, b.m.identity.NodeUUID); outcome != reconcileAccepted {
		t.Fatalf("C to B pairwise reconcile = %v, want accepted", outcome)
	}
	if reverse.Load() != 1 || b.rosters.Load() != bBefore+1 || a.rosters.Load() != aBefore {
		t.Fatalf("pairwise fanout reverse=%d B=%d A=%d, want 1/%d/%d", reverse.Load(), b.rosters.Load(), a.rosters.Load(), bBefore+1, aBefore)
	}
}

func TestClusterPeerRouteUsesCurrentAuthority(t *testing.T) {
	test := func(name string, mutate func(*Manager, *Manager)) {
		t.Run(name, func(t *testing.T) {
			a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, false)
			pairClusterFixtures(t, a, b)
			client, err := a.m.peerClient(b.m.identity.NodeUUID)
			if err != nil {
				t.Fatal(err)
			}
			mutate(a.m, b.m)
			response, err := client.Post("https://"+b.address+rosterPath, "application/json", nil)
			if response != nil {
				response.Body.Close()
			}
			if err == nil || b.rosters.Load() != 0 {
				t.Fatalf("stale client err=%v dispatched=%d, want rejection before dispatch", err, b.rosters.Load())
			}
		})
	}
	test("forgotten pin remains on disk", func(a, b *Manager) { a.trust.Forget(b.identity.NodeUUID) })
	test("pending member", func(a, b *Manager) {
		node, _ := a.memberByNodeID(b.identity.NodeUUID)
		node.State = statePendingOutbound
		a.upsertMember(node)
	})
	test("admission mismatch", func(a, b *Manager) {
		node, _ := a.memberByNodeID(b.identity.NodeUUID)
		node.AdmissionEpoch++
		a.upsertMember(node)
	})
	test("teardown pending", func(a, b *Manager) { a.teardownPending.Store(true) })
	test("endpoint moved", func(a, b *Manager) { a.setMemberAddr(b.identity.NodeUUID, "127.0.0.2", b.port+1) })
}

func TestClusterPeerAdaptersRejectRevokedAuthority(t *testing.T) {
	test := func(name string, mutate func(*testing.T, *Manager, *Manager)) {
		t.Run(name, func(t *testing.T) {
			a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, false)
			pairClusterFixtures(t, a, b)
			reverse := countClusterReverseDials(t, a.m)
			if outcome, _ := a.m.reconcileWith([]string{b.address}, b.m.identity.NodeUUID); outcome != reconcileAccepted {
				t.Fatalf("prime roster outcome = %v, want accepted", outcome)
			}
			roster, err := a.m.peerClient(b.m.identity.NodeUUID)
			if err != nil {
				t.Fatal(err)
			}
			removal, err := a.m.removalPeerClient(b.m.identity.NodeUUID)
			if err != nil {
				t.Fatal(err)
			}
			mutate(t, a.m, b.m)
			for _, request := range []struct {
				client *http.Client
				path   string
			}{{roster, rosterPath}, {removal, membersRemovePath}} {
				response, err := request.client.Post("https://"+b.address+request.path, "application/json", bytes.NewReader([]byte("{}")))
				if response != nil {
					response.Body.Close()
				}
				if err == nil {
					t.Fatalf("revoked authority allowed %s dispatch", request.path)
				}
			}
			if b.rosters.Load() != 1 || b.removes.Load() != 0 || reverse.Load() != 0 {
				t.Fatalf("revoked authority roster=%d removals=%d reverse=%d, want priming roster only/0/0", b.rosters.Load(), b.removes.Load(), reverse.Load())
			}
		})
	}
	test("pin revoked with roster idle", func(t *testing.T, a, b *Manager) { a.trust.Forget(b.identity.NodeUUID) })
	test("endpoint moved with roster idle", func(t *testing.T, a, b *Manager) {
		if !a.setMemberAddr(b.identity.NodeUUID, "localhost", b.port+1) {
			t.Fatal("move peer endpoint")
		}
	})
	test("admission mismatch with roster idle", func(t *testing.T, a, b *Manager) {
		node, ok := a.memberByNodeID(b.identity.NodeUUID)
		if !ok {
			t.Fatal("missing peer before admission mutation")
		}
		node.AdmissionEpoch++
		a.upsertMember(node)
	})
}

func TestClusterPeerRouteRevokedDuringDial(t *testing.T) {
	test := func(name string, mutate func(*testing.T, *Manager, *Manager)) {
		t.Run(name, func(t *testing.T) {
			a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, false)
			pairClusterFixtures(t, a, b)
			entered, release := make(chan struct{}), make(chan struct{})
			opts := a.m.peerServiceOptions()
			opts.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
				conn, err := (&net.Dialer{}).DialContext(ctx, network, addr)
				if err != nil {
					return nil, err
				}
				close(entered)
				select {
				case <-release:
					return conn, nil
				case <-ctx.Done():
					conn.Close()
					return nil, ctx.Err()
				}
			}
			var reverse atomic.Int32
			opts.DialReverse = func(context.Context, string, string, noderec.ServiceKey) (net.Conn, error) {
				reverse.Add(1)
				return nil, errors.New("unexpected reverse after authority change")
			}
			replaceClusterPeerServices(t, a.m, opts)
			client, err := a.m.peerClient(b.m.identity.NodeUUID)
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			go func() {
				resp, err := client.Post("https://"+b.address+rosterPath, "application/json", nil)
				if resp != nil {
					resp.Body.Close()
				}
				done <- err
			}()
			select {
			case <-entered:
			case <-time.After(time.Second):
				t.Fatal("request did not reach direct dial")
			}
			mutate(t, a.m, b.m)
			close(release)
			select {
			case err := <-done:
				if err == nil || b.rosters.Load() != 0 || reverse.Load() != 0 {
					t.Fatalf("changed authority err=%v roster=%d reverse=%d, want error/0/0", err, b.rosters.Load(), reverse.Load())
				}
			case <-time.After(2 * time.Second):
				t.Fatal("changed authority request stalled")
			}
		})
	}
	test("pin revoked during dial", func(t *testing.T, a, b *Manager) { a.trust.Forget(b.identity.NodeUUID) })
	test("endpoint changes during dial", func(t *testing.T, a, b *Manager) { a.setMemberAddr(b.identity.NodeUUID, "127.0.0.2", b.port+1) })
	test("same certificate readmitted during dial", readmitClusterPeer)
}

func readmitClusterPeer(t *testing.T, local, peer *Manager) {
	t.Helper()
	if err := local.trust.Remove(peer.identity.NodeUUID); err != nil {
		t.Fatal(err)
	}
	if err := local.trust.Pin(&TrustedPin{NodeUUID: peer.identity.NodeUUID, NodeID: peer.identity.NodeID,
		ClusterID: "cluster-1", AdmissionEpoch: 2, CertPem: string(peer.identity.CertPEM), CertFingerprint: peer.identity.CertFingerprint}); err != nil {
		t.Fatal(err)
	}
	node, ok := local.memberByNodeID(peer.identity.NodeUUID)
	if !ok {
		t.Fatal("readmission fixture lost member")
	}
	node.AdmissionEpoch = 2
	local.upsertMember(node)
}

func TestClusterPeerRouteChangesBeforeFirstApplicationWrite(t *testing.T) {
	test := func(name string, mutate func(*testing.T, *Manager, *Manager)) {
		t.Run(name, func(t *testing.T) {
			a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, false)
			pairClusterFixtures(t, a, b)
			reverse := countClusterReverseDials(t, a.m)
			client, err := a.m.peerClient(b.m.identity.NodeUUID)
			if err != nil {
				t.Fatal(err)
			}
			var handoff atomic.Int32
			ctx := httptrace.WithClientTrace(context.Background(), &httptrace.ClientTrace{
				GotConn: func(httptrace.GotConnInfo) {
					handoff.Add(1)
					mutate(t, a.m, b.m)
				},
			})
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://"+b.address+rosterPath, nil)
			if err != nil {
				t.Fatal(err)
			}
			response, err := client.Do(req)
			if response != nil {
				response.Body.Close()
			}
			if err == nil || handoff.Load() != 1 || b.rosters.Load() != 0 || reverse.Load() != 0 {
				t.Fatalf("before-write authority err=%v handoff=%d roster=%d reverse=%d, want error/1/0/0", err, handoff.Load(), b.rosters.Load(), reverse.Load())
			}
		})
	}
	test("endpoint changes after TLS", func(t *testing.T, a, b *Manager) {
		a.setMemberAddr(b.identity.NodeUUID, "localhost", b.port+1)
	})
	test("same certificate readmission after TLS", readmitClusterPeer)
}

func TestClusterPeerRouteGenerationTracksSelectedEndpoint(t *testing.T) {
	a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, false)
	pairClusterFixtures(t, a, b)
	first, ok := a.m.resolvePeerServiceRoute(b.m.identity.NodeUUID, noderec.ServiceCluster)
	if !ok || first.Address != b.address || first.Protocol != clustertrust.ReverseTLS || first.Generation == 0 {
		t.Fatalf("initial route = %+v admitted=%v", first, ok)
	}
	other := net.JoinHostPort("127.0.0.2", strconv.Itoa(b.m.port+1))
	a.m.browser = newBrowser()
	a.m.browser.seed(b.m.identity.NodeUUID, "127.0.0.2", b.m.port+1)
	if !a.m.selectPeerServiceAddress(b.m.identity.NodeUUID, other) {
		t.Fatal("current advertised endpoint could not be selected")
	}
	next, ok := a.m.resolvePeerServiceRoute(b.m.identity.NodeUUID, noderec.ServiceCluster)
	if !ok || next.Address != other || next.Generation <= first.Generation || next.Incarnation != first.Incarnation {
		t.Fatalf("selected endpoint route = %+v, previous %+v", next, first)
	}
	if !a.m.selectPeerServiceAddress(b.m.identity.NodeUUID, b.address) {
		t.Fatal("original current endpoint could not be selected")
	}
	back, ok := a.m.resolvePeerServiceRoute(b.m.identity.NodeUUID, noderec.ServiceCluster)
	if !ok || back.Address != b.address || back.Generation <= next.Generation {
		t.Fatalf("returned endpoint route = %+v, previous %+v", back, next)
	}
	if _, ok := a.m.resolvePeerServiceRoute(b.m.identity.NodeUUID, noderec.ServiceWorkload); ok {
		t.Fatal("cluster route authorized an unrelated service")
	}
	if a.m.selectPeerServiceAddress(b.m.identity.NodeUUID, "unregistered.example:14321") {
		t.Fatal("caller selected an endpoint absent from current membership and discovery")
	}
}

func TestClusterPeerRouteIncarnationChangesOnReadmission(t *testing.T) {
	a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, false)
	pairClusterFixtures(t, a, b)
	before, ok := a.m.resolvePeerServiceRoute(b.m.identity.NodeUUID, noderec.ServiceCluster)
	if !ok {
		t.Fatal("initial route missing")
	}
	readmitClusterPeer(t, a.m, b.m)
	after, ok := a.m.resolvePeerServiceRoute(b.m.identity.NodeUUID, noderec.ServiceCluster)
	if !ok || after.Incarnation == before.Incarnation || after.Address != before.Address {
		t.Fatalf("same-certificate readmission route = %+v, previous %+v", after, before)
	}
}
