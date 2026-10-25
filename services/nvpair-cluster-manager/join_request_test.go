// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"eapnoob"
	"nvpair-shared/discovery"
	"nvpair-shared/durablefile"
)

func joinTestManager(t *testing.T) *Manager {
	t.Helper()
	m, err := NewManager(NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), io.Discard}), t.TempDir(), defaultPort)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(m.peerServices.CloseIdle)
	t.Cleanup(m.removalPeerServices.CloseIdle)
	t.Cleanup(m.clients.CloseIdle)
	return m
}

func joinTestID(t *testing.T) string {
	t.Helper()
	id, err := newJoinRequestID()
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func joinTestRecord(t *testing.T, m *Manager, direction string) *joinRequestRecord {
	t.Helper()
	now := m.joinTime()
	id := joinTestID(t)
	cid, _ := m.clusterIdentity()
	_, epoch := m.currentAdmission()
	r := &joinRequestRecord{JoinRequest: JoinRequest{RequestID: id, Direction: direction, State: "awaiting-approval", PeerAddress: "127.0.0.1", PeerPort: 1, CreatedAt: now.UnixMilli(), ExpiresAt: now.Add(m.effectiveInviteTTL()).UnixMilli()}, source: "127.0.0.1", clusterID: cid, admission: epoch, generation: m.clusterGen.Load(), sessionGeneration: m.sessGen.Load()}
	m.joinMu.Lock()
	m.joinRequests[id] = r
	m.joinMu.Unlock()
	return r
}

func joinTestRPC(t *testing.T, m *Manager, method string, params any) Message {
	t.Helper()
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	m.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), &output})
	id := json.RawMessage(`1`)
	m.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: method, Params: data})
	decoder := json.NewDecoder(&output)
	for {
		var result Message
		if err := decoder.Decode(&result); err != nil {
			t.Fatalf("read RPC response: %v", err)
		}
		if result.ID != nil {
			return result
		}
	}
}

func TestJoinRequestEligibilityUsesAuthoritativeStores(t *testing.T) {
	test := func(name string, mutate func(*testing.T, *Manager), wantSolo bool, wantReason string) {
		t.Run(name, func(t *testing.T) {
			m := joinTestManager(t)
			mutate(t, m)
			got := m.joinSnapshot()
			if got.Supported != true || got.CanAskToJoin != (wantReason == "") || got.RequiresSoloReplacement != wantSolo || got.BlockedReason != wantReason || got.Requests == nil {
				t.Fatalf("snapshot = %+v", got)
			}
		})
	}
	test("fresh node", func(*testing.T, *Manager) {}, false, "")
	test("intentional solo requires consent", func(t *testing.T, m *Manager) {
		if _, _, err := m.foundCluster("named solo"); err != nil {
			t.Fatal(err)
		}
	}, true, "")
	test("offline durable peer blocks", func(t *testing.T, m *Manager) {
		data, err := json.Marshal([]ClusterNode{{NodeUUID: "offline-peer", State: stateMember}})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(m.membersPath(), data, 0600); err != nil {
			t.Fatal(err)
		}
	}, false, "already-clustered")
	test("malformed roster blocks", func(t *testing.T, m *Manager) {
		if err := os.WriteFile(m.membersPath(), []byte(`{`), 0600); err != nil {
			t.Fatal(err)
		}
	}, false, "state-unavailable")
	test("unreadable roster blocks", func(t *testing.T, m *Manager) {
		if err := os.Mkdir(m.membersPath(), 0700); err != nil {
			t.Fatal(err)
		}
	}, false, "state-unavailable")
	test("skipped pin still blocks", func(t *testing.T, m *Manager) {
		if err := os.WriteFile(filepath.Join(m.trust.dir, "invalid.json"), []byte(`{`), 0600); err != nil {
			t.Fatal(err)
		}
	}, false, "already-clustered")
	test("initialization publication blocks", func(t *testing.T, m *Manager) {
		if err := os.WriteFile(m.remoteManagementInitializingPath(), []byte(`{}`), 0600); err != nil {
			t.Fatal(err)
		}
	}, false, "state-unavailable")
	test("teardown marker blocks", func(t *testing.T, m *Manager) {
		if err := os.WriteFile(m.teardownMarkerPath(), []byte(`{}`), 0600); err != nil {
			t.Fatal(err)
		}
	}, false, "state-unavailable")
	test("receipt rollback blocks", func(t *testing.T, m *Manager) {
		receipt := map[string]pairingReceipt{"receipt": {InviteID: "receipt", ClusterID: "c", LocalEpoch: 1, Rollback: true}}
		data, err := json.Marshal(receipt)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(m.clusterDir, pairingReceiptsFile), data, 0600); err != nil {
			t.Fatal(err)
		}
	}, false, "pairing-busy")
	test("corrupt receipts block", func(t *testing.T, m *Manager) {
		if err := os.WriteFile(filepath.Join(m.clusterDir, pairingReceiptsFile), []byte(`[]`), 0600); err != nil {
			t.Fatal(err)
		}
	}, false, "state-unavailable")
	test("competing pairing blocks", func(t *testing.T, m *Manager) { m.putSession(&pairingSession{inviteID: "other", role: roleJoiner}) }, false, "pairing-busy")
}

func TestJoinRequestPendingApprovalDoesNotMutateAuthority(t *testing.T) {
	m := joinTestManager(t)
	if _, _, err := m.foundCluster("admitting cluster"); err != nil {
		t.Fatal(err)
	}
	before := joinTestAuthority(t, m)
	p := 12345
	id := joinTestID(t)
	reply, status, reason := m.applyJoinRequestWire("10.0.0.2", joinRequestWire{Version: 1, Operation: "request", RequestID: id, Requester: &joinRequester{"requester-uuid", "unverified\nname"}, Port: &p})
	if status != http.StatusOK || reason != "" || reply.State != "awaiting-approval" || reply.InviteID != nil {
		t.Fatalf("create = %+v, status %d, reason %q", reply, status, reason)
	}
	if !reflect.DeepEqual(before, joinTestAuthority(t, m)) {
		t.Fatal("pending request changed admission, members, pins, or management grants")
	}
	response := joinTestRPC(t, m, "cluster:respond-to-join-request", map[string]any{"requestId": id, "accept": true, "allowRemoteManagement": true})
	if response.Error != nil {
		t.Fatalf("approve: %v", response.Error)
	}
	var approved JoinRequest
	if err := json.Unmarshal(response.Result, &approved); err != nil {
		t.Fatal(err)
	}
	if approved.State != "awaiting-requester" || approved.InviteID == nil || approved.PeerName != "unverifiedname" {
		t.Fatalf("approval = %+v", approved)
	}
	if !reflect.DeepEqual(before, joinTestAuthority(t, m)) {
		t.Fatal("approval started EAP or changed authority before requester readiness")
	}
	if len(m.sessions) != 0 {
		t.Fatal("EAP started before ready")
	}
	second := joinTestRPC(t, m, "cluster:respond-to-join-request", map[string]any{"requestId": id, "accept": true, "allowRemoteManagement": true})
	if second.Error != nil {
		t.Fatal(second.Error)
	}
	var duplicate JoinRequest
	if err := json.Unmarshal(second.Result, &duplicate); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(approved, duplicate) {
		t.Fatal("duplicate approval changed request or child ID")
	}
}

func joinTestAuthority(t *testing.T, m *Manager) map[string]json.RawMessage {
	t.Helper()
	out := map[string]json.RawMessage{}
	doc, marker := m.remoteManagementPaths()
	for _, path := range []string{m.admissionPath(), m.membersPath(), doc, marker} {
		data, err := os.ReadFile(path)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		var value any
		if err := json.Unmarshal(data, &value); err != nil {
			t.Fatal(err)
		}
		canonical, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		out[filepath.Base(path)] = canonical
	}
	pins, err := json.Marshal(m.trust.List())
	if err != nil {
		t.Fatal(err)
	}
	out["pins"] = pins
	return out
}

func TestJoinRequestLocalRPCRejectsInvalidParams(t *testing.T) {
	cases := []struct {
		name, method string
		params       any
	}{
		{"solo consent mandatory", "cluster:ask-to-join", map[string]any{"address": "127.0.0.1"}},
		{"port range", "cluster:ask-to-join", map[string]any{"address": "127.0.0.1", "port": 0, "replaceSoloCluster": false}},
		{"unknown getter field", "cluster:get-join-requests", map[string]any{"pin": true}},
		{"accept mandatory", "cluster:respond-to-join-request", map[string]any{"requestId": "bad"}},
		{"grant mandatory", "cluster:respond-to-join-request", map[string]any{"requestId": "jr-00000000000000000000000000000000", "accept": true}},
		{"decline carries no grant", "cluster:respond-to-join-request", map[string]any{"requestId": "jr-00000000000000000000000000000000", "accept": false, "allowRemoteManagement": true}},
		{"invalid cancel ID", "cluster:cancel-join-request", map[string]any{"requestId": "bad"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := joinTestManager(t)
			reply := joinTestRPC(t, m, tc.method, tc.params)
			if reply.Error == nil || reply.Error.Code != codeInvalidParams {
				t.Fatalf("error = %+v", reply.Error)
			}
		})
	}
}

func TestJoinRequestSelfTargetAtServiceBoundary(t *testing.T) {
	for _, address := range []string{"127.0.0.1", "::1", "localhost"} {
		t.Run(address, func(t *testing.T) {
			m := joinTestManager(t)
			reply := joinTestRPC(t, m, "cluster:ask-to-join", map[string]any{"address": address, "port": m.port, "replaceSoloCluster": false})
			if reply.Error == nil || reply.Error.Code != codePrecondition || reply.Error.Message != "self-target" {
				t.Fatalf("error = %+v", reply.Error)
			}
			if len(m.joinRequests) != 0 {
				t.Fatal("self-target allocated request")
			}
		})
	}
}

func TestJoinRequestStaleApprovalAndExpiry(t *testing.T) {
	test := func(name string, change func(*Manager, *time.Time), want string) {
		t.Run(name, func(t *testing.T) {
			m := joinTestManager(t)
			if _, _, err := m.foundCluster("solo"); err != nil {
				t.Fatal(err)
			}
			now := time.Now()
			m.joinNow = func() time.Time { return now }
			r := joinTestRecord(t, m, "incoming")
			change(m, &now)
			snapshot := m.joinSnapshot()
			if len(snapshot.Requests) != 1 || snapshot.Requests[0].State != want {
				t.Fatalf("requests = %+v", snapshot.Requests)
			}
			reply := joinTestRPC(t, m, "cluster:respond-to-join-request", map[string]any{"requestId": r.RequestID, "accept": true, "allowRemoteManagement": false})
			if reply.Error == nil || reply.Error.Code != codeInvalidState {
				t.Fatalf("late approval error = %+v", reply.Error)
			}
			if len(m.sessions) != 0 {
				t.Fatal("late approval started EAP")
			}
		})
	}
	test("composition changed", func(m *Manager, _ *time.Time) { m.clusterGen.Add(1) }, "rejected")
	test("leave and rejoin generation", func(m *Manager, _ *time.Time) { m.sessGen.Add(1) }, "rejected")
	test("deadline from own clock", func(m *Manager, now *time.Time) { *now = now.Add(m.effectiveInviteTTL()) }, "expired")
}

func TestJoinRequestCancelCannotUnpairDurableCommit(t *testing.T) {
	m := joinTestManager(t)
	r := joinTestRecord(t, m, "outgoing")
	id := "exact-invite"
	r.InviteID = &id
	r.State = "pairing"
	m.putInvite(&Invite{InviteID: id, State: inviteStatePaired, CreatedAt: time.Now().UnixMilli()})
	reply := joinTestRPC(t, m, "cluster:cancel-join-request", map[string]string{"requestId": r.RequestID})
	if reply.Error != nil {
		t.Fatal(reply.Error)
	}
	var result JoinRequest
	if err := json.Unmarshal(reply.Result, &result); err != nil {
		t.Fatal(err)
	}
	if result.State != "paired" {
		t.Fatalf("request state = %q", result.State)
	}
	inv, ok := m.getInvite(id)
	if !ok || inv.State != inviteStatePaired {
		t.Fatal("cancel changed completed exact invite")
	}
}

func TestJoinRequestUnconfirmedCommitCannotReportPaired(t *testing.T) {
	m := joinTestManager(t)
	r := joinTestRecord(t, m, "incoming")
	id := "exact-invite"
	r.InviteID = &id
	r.State = "pairing"
	r.ready = true
	m.putInvite(&Invite{InviteID: id, State: inviteStatePaired, CreatedAt: time.Now().UnixMilli()})
	m.putSession(&pairingSession{inviteID: id, role: roleInviter, awaitingAck: true})
	snapshot := m.joinSnapshot()
	if snapshot.Requests[0].State != "pairing" || snapshot.Requests[0].Reason != reasonAwaitingConfirmation {
		t.Fatalf("unconfirmed result = %+v", snapshot.Requests[0])
	}
	m.deleteSession(id)
	if m.joinSnapshot().Requests[0].State != "pairing" {
		t.Fatal("session loss falsely confirmed durable outcome")
	}
	m.confirmJoinRequest(id)
	if m.joinSnapshot().Requests[0].State != "paired" {
		t.Fatal("authenticated exact confirmation did not settle request")
	}
}

func newJoinRequestPairingFixture(t *testing.T) outboundPairingFixture {
	t.Helper()
	member := joinTestManager(t)
	requester := joinTestManager(t)
	if _, _, err := member.foundCluster("join-test"); err != nil {
		t.Fatal(err)
	}
	incoming := joinTestRecord(t, member, "incoming")
	outgoing := joinTestRecord(t, requester, "outgoing")
	delete(requester.joinRequests, outgoing.RequestID)
	outgoing.RequestID = incoming.RequestID
	requester.joinRequests[outgoing.RequestID] = outgoing
	id := "correlated-invite"
	incoming.InviteID = &id
	incoming.State = "pairing"
	incoming.ready = true
	incoming.PeerNodeUUID = requester.identity.NodeUUID
	incoming.allowRemoteManagement = true
	outgoing.InviteID = &id
	outgoing.State = "pairing"
	outgoing.ready = true
	outgoing.PeerNodeUUID = member.identity.NodeUUID
	server := httptest.NewServer(http.HandlerFunc(requester.handlePairing))
	t.Cleanup(server.Close)
	_, requesterPortText, err := net.SplitHostPort(server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	requester.port, err = strconv.Atoi(requesterPortText)
	if err != nil {
		t.Fatal(err)
	}
	memberServer := httptest.NewServer(http.HandlerFunc(member.handlePairing))
	t.Cleanup(memberServer.Close)
	_, memberPortText, err := net.SplitHostPort(memberServer.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	member.port, err = strconv.Atoi(memberPortText)
	if err != nil {
		t.Fatal(err)
	}
	cid, _ := member.clusterIdentity()
	member.putInvite(&Invite{InviteID: id, FromNodeUUID: member.identity.NodeUUID, ClusterID: cid, State: inviteStatePending, CreatedAt: time.Now().UnixMilli()})
	sess, pin, err := member.runInitialExchange(id, strings.TrimPrefix(server.URL, "http://"), cid, member.sessGen.Load(), true)
	if err != nil {
		t.Fatal("correlated Initial Exchange failed")
	}
	peer, ok := requester.getSession(id)
	if !ok {
		t.Fatal("missing joiner session")
	}
	return outboundPairingFixture{inviter: member, joiner: requester, session: sess, joinSession: peer, pin: pin}
}

func TestJoinRequestExactTranscriptAndIndependentGrants(t *testing.T) {
	f := newJoinRequestPairingFixture(t)
	member, requester, sess, peer, pin := f.inviter, f.joiner, f.session, f.joinSession, f.pin
	id := sess.inviteID
	requestID := member.joinRequestForInvite(id)
	if sess.peerPairing.JoinRequestID != requestID || peer.peerPairing.JoinRequestID != requestID || !validJoinRequestID(requestID) {
		t.Fatal("request ID missing from authenticated transcript")
	}
	if sess.allowRemoteManagement == nil || !*sess.allowRemoteManagement || peer.allowRemoteManagement != nil {
		t.Fatal("member choice leaked into requester grant")
	}
	requesterChoice := false
	peer.allowRemoteManagement = &requesterChoice
	if err := peer.peer.OOBInputNoob(noobFromPIN(pin)); err != nil {
		t.Fatal("could not submit PIN")
	}
	peer.ready = true
	blob, err := sess.server.Start()
	if err != nil {
		t.Fatal(err)
	}
	for {
		rr := httptest.NewRecorder()
		requester.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: id}, blob)
		if rr.Code != http.StatusOK {
			t.Fatalf("completion status = %d", rr.Code)
		}
		var env pairingEnvelope
		if err := json.Unmarshal(rr.Body.Bytes(), &env); err != nil {
			t.Fatal(err)
		}
		response, err := base64.StdEncoding.DecodeString(env.Msg)
		if err != nil {
			t.Fatal(err)
		}
		out, err := sess.server.Receive(response)
		if err != nil || out.Err != nil {
			t.Fatal("correct PIN completion failed")
		}
		if out.Done {
			if !out.Success {
				t.Fatal("correct PIN was refused")
			}
			if err := member.onInviterPaired(id, sess); err != nil {
				t.Fatal(err)
			}
			// Deliver authenticated Success through the canonical joiner commit.
			rr = httptest.NewRecorder()
			requester.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: id}, out.Send)
			if rr.Code != http.StatusOK {
				t.Fatalf("success status = %d", rr.Code)
			}
			break
		}
		blob = out.Send
	}
	if peer.peer.State() != eapnoob.StateRegistered {
		t.Fatal("requester did not complete EAP")
	}
	member.rosterMu.Lock()
	memberGrant := member.remoteManagementSnapshotLocked()
	member.rosterMu.Unlock()
	requester.rosterMu.Lock()
	requesterGrant := requester.remoteManagementSnapshotLocked()
	requester.rosterMu.Unlock()
	if len(memberGrant.Grants) != 1 || !memberGrant.Grants[0].Allowed || len(requesterGrant.Grants) != 1 || requesterGrant.Grants[0].Allowed {
		t.Fatal("local grant choices were not independently committed")
	}
	if got := requester.joinSnapshot().Requests[0]; got.State != "paired" {
		t.Fatalf("requester outcome = %+v", got)
	}
	member.ackPairing(&pairingEnvelope{InviteID: id}, requester.identity.NodeUUID)
	if got := member.joinSnapshot().Requests[0]; got.State != "paired" {
		t.Fatalf("member outcome = %+v", got)
	}
}

func TestJoinRequestPeerMismatchAndUnrelatedInvite(t *testing.T) {
	m := joinTestManager(t)
	r := joinTestRecord(t, m, "outgoing")
	id := "exact-child"
	r.InviteID = &id
	r.ready = true
	r.PeerNodeUUID = "expected-peer"
	test := func(name, invite, request, peer string, want bool) {
		t.Run(name, func(t *testing.T) {
			if got := m.matchJoinRequestPeer(invite, &PairingInfo{JoinRequestID: request, NodeUUID: peer}); got != want {
				t.Fatalf("match = %t, want %t", got, want)
			}
		})
	}
	test("exact correlation", id, r.RequestID, "expected-peer", true)
	test("wrong authenticated UUID", id, r.RequestID, "other-peer", false)
	test("missing transcript request", id, "", "expected-peer", false)
	test("wrong request", id, joinTestID(t), "expected-peer", false)
	test("unrelated invite cannot satisfy request", "other-child", r.RequestID, "expected-peer", false)
	if m.joinRequestAllowsInvite("normal-unrelated") {
		t.Fatal("ordinary invite raced a service-owned outgoing request")
	}
}

func TestJoinRequestConcurrentSnapshotsAndExpiry(t *testing.T) {
	m := joinTestManager(t)
	r := joinTestRecord(t, m, "outgoing")
	r.ExpiresAt = m.joinTime().Add(-time.Second).UnixMilli()
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); m.joinSnapshot() }()
	}
	wg.Wait()
	if got := m.joinSnapshot().Requests; len(got) != 1 || got[0].State != "expired" {
		t.Fatalf("requests = %+v", got)
	}
}

func TestJoinRequestUnsupportedPeerPreservesSolo(t *testing.T) {
	m := joinTestManager(t)
	if _, _, err := m.foundCluster("keep this solo"); err != nil {
		t.Fatal(err)
	}
	before := joinTestAuthority(t, m)
	server := httptest.NewServer(http.NotFoundHandler())
	defer server.Close()
	r := joinTestRecord(t, m, "outgoing")
	r.replaceSolo = true
	m.driveJoinRequest(r.RequestID, []string{strings.TrimPrefix(server.URL, "http://")})
	if got := m.joinSnapshot().Requests[0]; got.State != "failed" || got.Reason != "unsupported-peer" {
		t.Fatalf("result = %+v", got)
	}
	if !reflect.DeepEqual(before, joinTestAuthority(t, m)) {
		t.Fatal("unsupported peer dissolved solo cluster")
	}
}

func TestJoinRequestSoloReplacementWaitsForApproval(t *testing.T) {
	m := joinTestManager(t)
	if _, _, err := m.foundCluster("replace after approval"); err != nil {
		t.Fatal(err)
	}
	beforeID, _ := m.clusterIdentity()
	identity := m.identity.NodeUUID
	r := joinTestRecord(t, m, "outgoing")
	r.replaceSolo = true
	child := "approved-child"
	ready := make(chan struct{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		data, err := io.ReadAll(req.Body)
		if err != nil {
			t.Error(err)
			return
		}
		wire, reason := decodeJoinRequestWire(data)
		if reason != "" {
			t.Error(reason)
			return
		}
		state := "awaiting-requester"
		if wire.Operation == "request" {
			if cid, _ := m.clusterIdentity(); cid != beforeID {
				t.Error("solo removed before approval")
			}
		}
		if wire.Operation == "ready" {
			if cid, _ := m.clusterIdentity(); cid != "" {
				t.Error("ready sent before teardown")
			}
			state = "declined"
			ready <- struct{}{}
		}
		_ = json.NewEncoder(w).Encode(joinRequestReply{Version: 1, RequestID: r.RequestID, State: state, CreatedAt: r.CreatedAt, ExpiresAt: r.ExpiresAt, InviteID: &child})
	}))
	defer server.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	m.joinContext = ctx
	m.driveJoinRequest(r.RequestID, []string{strings.TrimPrefix(server.URL, "http://")})
	select {
	case <-ready:
	default:
		t.Fatal("no ready after approved teardown")
	}
	if cid, _ := m.clusterIdentity(); cid != "" {
		t.Fatal("decline resurrected old admission")
	}
	if m.identity.NodeUUID != identity {
		t.Fatal("solo replacement changed device identity")
	}
	if !m.admissionWasRetired() {
		t.Fatal("solo admission was not durably retired")
	}
	if got := m.joinSnapshot().Requests[0]; got.State != "declined" {
		t.Fatalf("outcome = %+v", got)
	}
}

func TestJoinRequestAddressResolutionUsesExplicitPort(t *testing.T) {
	m := joinTestManager(t)
	p := 12345
	endpoints, err := m.inviteEndpoints("::1", &p, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(endpoints) != 1 || endpoints[0] != net.JoinHostPort("::1", strconv.Itoa(p)) {
		t.Fatalf("endpoints = %v", endpoints)
	}
}

func TestJoinRequestHostnameSelectorCorrelatesCanonicalPeerUUID(t *testing.T) {
	m := joinTestManager(t)
	peer := joinTestManager(t)
	requestStarted := make(chan struct{})
	releaseRequest := make(chan struct{})
	release := sync.OnceFunc(func() { close(releaseRequest) })
	cancelReceived := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var wire joinRequestWire
		if err := json.NewDecoder(req.Body).Decode(&wire); err != nil {
			t.Error(err)
			return
		}
		if wire.Operation == "request" {
			close(requestStarted)
			<-releaseRequest
		} else if wire.Operation == "cancel" {
			defer close(cancelReceived)
		}
		now := m.joinTime()
		if err := json.NewEncoder(w).Encode(joinRequestReply{Version: 1, RequestID: wire.RequestID, State: "awaiting-approval", CreatedAt: now.UnixMilli(), ExpiresAt: now.Add(time.Minute).UnixMilli()}); err != nil {
			t.Error(err)
		}
	}))
	defer server.Close()
	defer release()
	host, portText, err := net.SplitHostPort(server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}
	hostname := "discovered-member"
	m.browser = newBrowser()
	m.browser.relay[peer.identity.NodeUUID] = discovery.Node{ID: hostname, Addresses: []string{host}, Port: port, TXT: []string{"uuid=" + peer.identity.NodeUUID}}
	response := joinTestRPC(t, m, "cluster:ask-to-join", map[string]any{"nodeId": hostname, "replaceSoloCluster": false})
	if response.Error != nil {
		t.Fatalf("ask by hostname: %+v", response.Error)
	}
	var result JoinRequest
	if err := json.Unmarshal(response.Result, &result); err != nil {
		t.Fatal(err)
	}
	if result.PeerNodeUUID != peer.identity.NodeUUID || result.PeerAddress != host || result.PeerPort != port {
		t.Fatalf("hostname resolution = %+v", result)
	}
	select {
	case <-requestStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("request did not reach the owned discovery endpoint")
	}
	id := "hostname-child"
	m.joinMu.Lock()
	r := m.joinRequests[result.RequestID]
	r.InviteID, r.State, r.ready = &id, "pairing", true
	m.joinMu.Unlock()
	if _, _, err := peer.foundCluster("hostname member"); err != nil {
		t.Fatal(err)
	}
	incoming := joinTestRecord(t, peer, "incoming")
	delete(peer.joinRequests, incoming.RequestID)
	incoming.RequestID = result.RequestID
	incoming.InviteID, incoming.State, incoming.ready = &id, "pairing", true
	incoming.PeerNodeUUID = m.identity.NodeUUID
	peer.joinRequests[result.RequestID] = incoming
	cid, _ := peer.clusterIdentity()
	peer.putInvite(&Invite{InviteID: id, FromNodeUUID: peer.identity.NodeUUID, ClusterID: cid, State: inviteStatePending, CreatedAt: time.Now().UnixMilli()})
	pairingServer := httptest.NewServer(http.HandlerFunc(m.handlePairing))
	defer pairingServer.Close()
	if _, _, err := peer.runInitialExchange(id, strings.TrimPrefix(pairingServer.URL, "http://"), cid, peer.sessGen.Load(), false); err != nil {
		t.Fatal("hostname-selected member failed the exact Initial Exchange correlation")
	}
	sess, live := m.getSession(id)
	if !live || sess.peerPairing == nil || sess.peerPairing.NodeUUID != result.PeerNodeUUID || sess.peerPairing.JoinRequestID != result.RequestID {
		t.Fatal("Initial Exchange did not retain the canonical member UUID and exact request ID")
	}
	info := peer.localPairingInfo(server.Listener.Addr().String())
	info.JoinRequestID = result.RequestID
	if !m.matchJoinRequestPeer(id, info) {
		t.Fatal("hostname selection rejected the exact transcript peer UUID")
	}
	info.NodeUUID = "different-authenticated-peer"
	if m.matchJoinRequestPeer(id, info) {
		t.Fatal("hostname selection accepted a mismatched transcript peer UUID")
	}
	m.joinMu.Lock()
	m.finishJoinRequestLocked(r, "canceled", "")
	m.joinMu.Unlock()
	// Release the in-flight prelude; its late reply must retain cancellation.
	release()
	select {
	case <-cancelReceived:
	case <-time.After(5 * time.Second):
		t.Fatal("request worker did not stop after its canceled late reply")
	}
	if got := m.joinSnapshot().Requests[0]; got.State != "canceled" {
		t.Fatalf("late reply changed terminal state: %+v", got)
	}
}

func TestJoinRequestUnresolvedSelectorDoesNotBecomePeerUUID(t *testing.T) {
	m := joinTestManager(t)
	selector := "unresolved-hostname"
	endpoints, uuid, err := m.joinRequestEndpoints("127.0.0.1", nil, &selector)
	if err != nil {
		t.Fatal(err)
	}
	if len(endpoints) != 1 || uuid != "" {
		t.Fatalf("unresolved selector = %v, UUID %q", endpoints, uuid)
	}
}

func TestJoinRequestInitialLoopStopsOnCancellation(t *testing.T) {
	test := func(name string, cancel func(*testing.T, *Manager, string, string)) {
		t.Run(name, func(t *testing.T) {
			m, r, target, blocked, attempts := joinRequestLoopFixture(t)
			cid, _ := m.clusterIdentity()
			id := *r.InviteID
			done := make(chan map[string]any, 1)
			go func() { done <- m.completeInitialInvite(id, []string{target}, cid, m.sessGen.Load(), false) }()
			select {
			case <-blocked:
			case <-time.After(5 * time.Second):
				t.Fatal("Initial Exchange did not repeat discovery")
			}
			cancel(t, m, r.RequestID, id)
			select {
			case result := <-done:
				if result["state"] != inviteStateCanceled {
					t.Fatalf("exchange outcome = %v", result["state"])
				}
			case <-time.After(time.Second):
				t.Fatal("cancellation did not interrupt the active HTTP wait")
			}
			joinRequestAssertStoppedLoop(t, m, r.RequestID, id, target, cid, "canceled", attempts)
		})
	}
	test("remote request cancel", func(t *testing.T, m *Manager, requestID, _ string) {
		if _, status, reason := m.applyJoinRequestWire("127.0.0.1", joinRequestWire{Version: 1, Operation: "cancel", RequestID: requestID}); status != http.StatusOK {
			t.Fatalf("cancel = %d %q", status, reason)
		}
	})
	test("local exact invite cancel", func(t *testing.T, m *Manager, _, inviteID string) {
		response := joinTestRPC(t, m, "cluster:cancel-invite", map[string]any{"inviteId": inviteID})
		if response.Error != nil {
			t.Fatalf("cancel invite: %+v", response.Error)
		}
	})
}

func TestJoinRequestInitialLoopStopsAtOriginalDeadline(t *testing.T) {
	m, r, target, blocked, attempts := joinRequestLoopFixture(t)
	// The child is fresh, but only the remaining original request lifetime is
	// available. No request loop or expiry sweep runs in this fixture.
	r.CreatedAt = m.joinTime().Add(-m.effectiveInviteTTL()).UnixMilli()
	r.ExpiresAt = m.joinTime().Add(time.Second).UnixMilli()
	cid, _ := m.clusterIdentity()
	id := *r.InviteID
	done := make(chan map[string]any, 1)
	go func() { done <- m.completeInitialInvite(id, []string{target}, cid, m.sessGen.Load(), false) }()
	select {
	case <-blocked:
	case <-time.After(5 * time.Second):
		t.Fatal("Initial Exchange did not repeat discovery")
	}
	select {
	case result := <-done:
		if result["state"] != inviteStateExpired {
			t.Fatalf("exchange outcome = %v", result["state"])
		}
	case <-time.After(3 * time.Second):
		t.Fatal("original request deadline did not interrupt the active HTTP wait")
	}
	joinRequestAssertStoppedLoop(t, m, r.RequestID, id, target, cid, "expired", attempts)
}

func TestJoinRequestInitialLoopStopsWhenExactSessionIsRemoved(t *testing.T) {
	m, r, target, blocked, attempts := joinRequestLoopFixture(t)
	cid, _ := m.clusterIdentity()
	id := *r.InviteID
	done := make(chan error, 1)
	go func() {
		// The alternate is the same owned listener: an invalidated exact
		// session must not start a replacement at any endpoint.
		_, _, err := m.pairAtFirstWorkingEndpoint(id, []string{target, target}, cid, m.sessGen.Load(), false)
		done <- err
	}()
	select {
	case <-blocked:
	case <-time.After(5 * time.Second):
		t.Fatal("Initial Exchange did not repeat discovery")
	}
	m.deleteSession(id)
	select {
	case err := <-done:
		if !errors.Is(err, errPairingAbandoned) {
			t.Fatalf("session removal outcome = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("session removal did not interrupt the active HTTP wait")
	}
	if attempts.Load() != 3 {
		t.Fatalf("Initial HTTP attempts after session removal = %d", attempts.Load())
	}
	if _, live := m.getSession(id); live {
		t.Fatal("session removal started a replacement session")
	}
}

func joinRequestLoopFixture(t *testing.T) (*Manager, *joinRequestRecord, string, <-chan struct{}, *atomic.Int32) {
	t.Helper()
	m := joinTestManager(t)
	m.port = 1
	if _, _, err := m.foundCluster("bounded Initial Exchange"); err != nil {
		t.Fatal(err)
	}
	r := joinTestRecord(t, m, "incoming")
	id := "repeating-initial-child"
	r.InviteID, r.State, r.ready = &id, "pairing", true
	cid, _ := m.clusterIdentity()
	m.putInvite(&Invite{InviteID: id, FromNodeUUID: m.identity.NodeUUID, ClusterID: cid, State: inviteStatePending, CreatedAt: time.Now().UnixMilli()})
	blocked := make(chan struct{})
	attempts := &atomic.Int32{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var env pairingEnvelope
		if err := json.NewDecoder(req.Body).Decode(&env); err != nil {
			t.Error(err)
			return
		}
		if env.InviteID != id {
			t.Error("HTTP attempt lost the exact child invite ID")
			return
		}
		if env.Phase == "cancel" {
			respondPairing(w, nil)
			return
		}
		if env.Phase != "initial" {
			t.Errorf("unexpected phase %q", env.Phase)
			return
		}
		if attempts.Add(1) == 3 {
			close(blocked)
			<-req.Context().Done()
			return
		}
		respondPairing(w, []byte(`{"Type":1,"PeerState":0}`))
	}))
	t.Cleanup(server.Close)
	return m, r, strings.TrimPrefix(server.URL, "http://"), blocked, attempts
}

func joinRequestAssertStoppedLoop(t *testing.T, m *Manager, requestID, inviteID, target, cid, want string, attempts *atomic.Int32) {
	t.Helper()
	if got := m.joinSnapshot().Requests[0]; got.RequestID != requestID || got.State != want {
		t.Fatalf("request outcome = %+v", got)
	}
	inv, ok := m.getInvite(inviteID)
	if !ok || string(inv.State) != want || inv.Pin != nil {
		t.Fatalf("child outcome = %+v, exists %t", inv, ok)
	}
	if _, live := m.getSession(inviteID); live {
		t.Fatal("terminal Initial Exchange retained its session")
	}
	// A late worker/retry must keep the recorded outcome and send no more HTTP.
	result := m.completeInitialInvite(inviteID, []string{target}, cid, m.sessGen.Load(), false)
	if result["state"] != inv.State || attempts.Load() != 3 {
		t.Fatalf("late result = %v, Initial HTTP attempts = %d", result["state"], attempts.Load())
	}
}

func TestJoinRequestRedirectingHandoffNeverContactsRedirectTarget(t *testing.T) {
	m := joinTestManager(t)
	if _, _, err := m.foundCluster("member"); err != nil {
		t.Fatal(err)
	}
	var contacted atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { contacted.Add(1); w.WriteHeader(http.StatusOK) }))
	defer target.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()
	host, portText, err := net.SplitHostPort(strings.TrimPrefix(redirect.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}
	id := joinTestID(t)
	if _, status, reason := m.applyJoinRequestWire(host, joinRequestWire{Version: 1, Operation: "request", RequestID: id, Requester: &joinRequester{NodeUUID: "requester", Name: "requester"}, Port: &port}); status != http.StatusOK {
		t.Fatalf("create status %d, reason %q", status, reason)
	}
	response := joinTestRPC(t, m, "cluster:respond-to-join-request", map[string]any{"requestId": id, "accept": true, "allowRemoteManagement": false})
	if response.Error != nil {
		t.Fatal(response.Error)
	}
	m.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), io.Discard})
	if _, status, reason := m.applyJoinRequestWire(host, joinRequestWire{Version: 1, Operation: "ready", RequestID: id}); status != http.StatusOK {
		t.Fatalf("ready status %d, reason %q", status, reason)
	}
	deadline := time.Now().Add(3 * time.Second)
	for {
		got := m.joinSnapshot().Requests[0]
		if got.State == "failed" {
			if got.Reason != "redirect-refused" {
				t.Fatalf("reason = %q", got.Reason)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("handoff did not fail: %+v", got)
		}
		time.Sleep(10 * time.Millisecond)
	}
	if contacted.Load() != 0 {
		t.Fatal("handoff contacted redirect target")
	}
	if len(m.trust.List()) != 0 {
		t.Fatal("redirecting handoff established trust")
	}
}

func TestJoinRequestRedirectingCompletionNeverContactsRedirectTarget(t *testing.T) {
	m := joinTestManager(t)
	r := joinTestRecord(t, m, "incoming")
	id := "completion-redirect"
	r.InviteID = &id
	r.State = "pairing"
	r.ready = true
	var contacted atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { contacted.Add(1); w.WriteHeader(http.StatusOK) }))
	defer target.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()
	sess := &pairingSession{inviteID: id, role: roleInviter, joinerAddr: strings.TrimPrefix(redirect.URL, "http://"), createdAt: time.Now().UnixMilli()}
	m.putInvite(&Invite{InviteID: id, State: inviteStatePending, CreatedAt: sess.createdAt})
	m.putSession(sess)
	m.driveOutboundCompletion(sess)
	got := m.joinSnapshot().Requests[0]
	if got.State != "failed" || got.Reason != "redirect-refused" {
		t.Fatalf("completion outcome = %+v", got)
	}
	if contacted.Load() != 0 {
		t.Fatal("completion contacted redirect target")
	}
}

func TestJoinRequestWrongPINRetainsCanonicalFailure(t *testing.T) {
	f := newOutboundPairingFixture(t)
	r := joinTestRecord(t, f.inviter, "incoming")
	id := f.session.inviteID
	r.InviteID = &id
	r.State = "pairing"
	r.ready = true
	wrong := "000000"
	if wrong == f.pin {
		wrong = "000001"
	}
	if err := f.joinSession.peer.OOBInputNoob(noobFromPIN(wrong)); err != nil {
		t.Fatal("wrong PIN submission failed before authentication")
	}
	f.joinSession.ready = true
	server := httptest.NewServer(http.HandlerFunc(f.joiner.handlePairing))
	defer server.Close()
	f.session.joinerAddr = strings.TrimPrefix(server.URL, "http://")
	mBefore := len(f.inviter.trust.List())
	peerBefore := len(f.joiner.trust.List())
	f.inviter.driveOutboundCompletion(f.session)
	got := f.inviter.joinSnapshot().Requests[0]
	if got.State != "failed" || got.Reason != reasonIncorrectPIN {
		t.Fatalf("wrong PIN outcome = %+v", got)
	}
	if len(f.inviter.trust.List()) != mBefore || len(f.joiner.trust.List()) != peerBefore {
		t.Fatal("wrong PIN changed trust")
	}
}

func TestJoinRequestFailedSoloTeardownNeverSendsReady(t *testing.T) {
	m := joinTestManager(t)
	if _, _, err := m.foundCluster("keep unless teardown succeeds"); err != nil {
		t.Fatal(err)
	}
	before := joinTestAuthority(t, m)
	r := joinTestRecord(t, m, "outgoing")
	r.replaceSolo = true
	child := "child-not-started"
	var ready atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var wire joinRequestWire
		if err := json.NewDecoder(req.Body).Decode(&wire); err != nil {
			t.Error(err)
			return
		}
		if wire.Operation == "ready" {
			ready.Add(1)
		}
		_ = json.NewEncoder(w).Encode(joinRequestReply{Version: 1, RequestID: r.RequestID, State: "awaiting-requester", CreatedAt: r.CreatedAt, ExpiresAt: r.ExpiresAt, InviteID: &child})
	}))
	defer server.Close()
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	clusterFileReplacer = durablefile.Replacer{Replace: func(from, to string) error {
		if to == m.teardownMarkerPath() {
			return errors.New("isolated teardown write fault")
		}
		return os.Rename(from, to)
	}}
	m.driveJoinRequest(r.RequestID, []string{strings.TrimPrefix(server.URL, "http://")})
	if ready.Load() != 0 {
		t.Fatal("ready sent after failed durable teardown")
	}
	if !reflect.DeepEqual(before, joinTestAuthority(t, m)) {
		t.Fatal("failed teardown changed original solo authority")
	}
	if got := m.joinSnapshot().Requests[0]; got.State != "rejected" || got.Reason != "state-unavailable" {
		t.Fatalf("result = %+v", got)
	}
}

func TestJoinRequestConcurrentApprovalAndReadyStartsOneChild(t *testing.T) {
	m := joinTestManager(t)
	if _, _, err := m.foundCluster("member"); err != nil {
		t.Fatal(err)
	}
	var callbacks atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callbacks.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()
	host, p, err := net.SplitHostPort(server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(p)
	if err != nil {
		t.Fatal(err)
	}
	id := joinTestID(t)
	if _, status, reason := m.applyJoinRequestWire(host, joinRequestWire{Version: 1, Operation: "request", RequestID: id, Requester: &joinRequester{"requester", "Requester"}, Port: &port}); status != http.StatusOK {
		t.Fatalf("create = %d %q", status, reason)
	}
	params, err := json.Marshal(map[string]any{"requestId": id, "accept": true, "allowRemoteManagement": false})
	if err != nil {
		t.Fatal(err)
	}
	var clients sync.WaitGroup
	for i := 0; i < 8; i++ {
		clients.Add(1)
		go func() {
			defer clients.Done()
			rpcID := json.RawMessage(`1`)
			m.handleMessage(&Message{JSONRPC: "2.0", ID: &rpcID, Method: "cluster:respond-to-join-request", Params: params})
		}()
	}
	clients.Wait()
	if len(m.invites) != 0 || len(m.sessions) != 0 {
		t.Fatal("concurrent approval started pairing before ready")
	}
	for i := 0; i < 8; i++ {
		clients.Add(1)
		go func() {
			defer clients.Done()
			_, status, _ := m.applyJoinRequestWire(host, joinRequestWire{Version: 1, Operation: "ready", RequestID: id})
			if status != http.StatusOK {
				t.Errorf("ready status = %d", status)
			}
		}()
	}
	clients.Wait()
	deadline := time.Now().Add(3 * time.Second)
	for {
		if got := m.joinSnapshot().Requests[0]; got.State == "failed" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("isolated callback did not finish")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if callbacks.Load() != 1 {
		t.Fatalf("callback count = %d, want exactly one child", callbacks.Load())
	}
	m.memMu.Lock()
	defer m.memMu.Unlock()
	if len(m.invites) != 1 {
		t.Fatalf("invite count = %d", len(m.invites))
	}
}

func TestJoinRequestUncertainCommitSurvivesInviteCacheExpiry(t *testing.T) {
	m := joinTestManager(t)
	r := joinTestRecord(t, m, "incoming")
	id := "uncertain-child"
	r.InviteID = &id
	r.State = "pairing"
	r.ready = true
	created := time.Now().Add(-2 * m.effectiveInviteTTL()).UnixMilli()
	r.CreatedAt = created
	r.ExpiresAt = created + m.effectiveInviteTTL().Milliseconds()
	m.putInvite(&Invite{InviteID: id, State: inviteStatePaired, CreatedAt: created, RespondedAt: &created})
	m.putSession(&pairingSession{inviteID: id, role: roleInviter, createdAt: created, awaitingAck: true})
	// Sweep before the first getter/push: even this ordering must preserve the
	// exact unconfirmed commit rather than fabricating an expired request.
	m.expirePendingInvites(time.Now())
	got := m.joinSnapshot().Requests[0]
	if got.State != "pairing" || got.Reason != reasonAwaitingConfirmation {
		t.Fatalf("uncertain result = %+v", got)
	}
	if _, ok := m.getInvite(id); !ok {
		t.Fatal("evicted an uncertain exact commit")
	}
	if snapshot := m.joinSnapshot(); snapshot.CanAskToJoin || snapshot.BlockedReason != "pairing-busy" {
		t.Fatalf("eligibility = %+v", snapshot)
	}
}

func TestJoinRequestRequestDeadlineBoundsCompositionWait(t *testing.T) {
	m := joinTestManager(t)
	m.rosterMu.Lock()
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	done := make(chan int, 1)
	go func() {
		_, status, _ := m.applyJoinRequestWireContext(ctx, "127.0.0.1", joinRequestWire{Version: 1, Operation: "status", RequestID: joinTestID(t)})
		done <- status
	}()
	select {
	case status := <-done:
		if status != http.StatusTooManyRequests {
			t.Errorf("busy response = %d", status)
		}
	case <-time.After(time.Second):
		t.Error("composition wait ignored request deadline")
	}
	m.rosterMu.Unlock()
	if !m.inviteMu.TryLock() {
		t.Fatal("deadline left outer composition owner locked")
	}
	m.inviteMu.Unlock()
}

func TestJoinRequestTriesResolvedAlternateWithoutChangingID(t *testing.T) {
	m := joinTestManager(t)
	r := joinTestRecord(t, m, "outgoing")
	var requests atomic.Int32
	old := httptest.NewServer(http.NotFoundHandler())
	defer old.Close()
	selected := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var wire joinRequestWire
		if err := json.NewDecoder(req.Body).Decode(&wire); err != nil {
			t.Error(err)
			return
		}
		if wire.RequestID != r.RequestID {
			t.Error("endpoint retry changed request ID")
		}
		if wire.Operation == "request" {
			requests.Add(1)
		}
		_ = json.NewEncoder(w).Encode(joinRequestReply{Version: 1, RequestID: r.RequestID, State: "declined", CreatedAt: r.CreatedAt, ExpiresAt: r.ExpiresAt})
	}))
	defer selected.Close()
	m.driveJoinRequest(r.RequestID, []string{strings.TrimPrefix(old.URL, "http://"), strings.TrimPrefix(selected.URL, "http://")})
	if requests.Load() != 1 {
		t.Fatal("did not reach resolved alternate")
	}
	got := m.joinSnapshot().Requests[0]
	if got.State != "declined" || got.RequestID != r.RequestID {
		t.Fatalf("result = %+v", got)
	}
	_, portText, err := net.SplitHostPort(selected.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}
	if got.PeerPort != port {
		t.Fatalf("selected port = %d, want advertised alternate %d", got.PeerPort, port)
	}
}

func TestJoinRequestCorrelatedCancellationNeverFollowsRedirect(t *testing.T) {
	m := joinTestManager(t)
	r := joinTestRecord(t, m, "incoming")
	id := "canceled-correlated-child"
	r.InviteID = &id
	r.State = "canceled"
	r.terminalAt = m.joinTime().UnixMilli()
	var contacted atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { contacted.Add(1); respondPairing(w, nil) }))
	defer target.Close()
	var cancellations atomic.Int32
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var env pairingEnvelope
		if err := json.NewDecoder(r.Body).Decode(&env); err != nil {
			t.Error(err)
			return
		}
		if env.InviteID != id || env.Phase != "cancel" {
			t.Error("cancellation lost exact child identity")
		}
		cancellations.Add(1)
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()
	m.notifyJoinerCanceled(strings.TrimPrefix(redirect.URL, "http://"), id)
	if cancellations.Load() != 1 || contacted.Load() != 0 {
		t.Fatalf("cancellations = %d, redirect contacts = %d", cancellations.Load(), contacted.Load())
	}
}

func TestJoinRequestCorrelatedReceiptReplayNeverFollowsRedirect(t *testing.T) {
	f := newJoinRequestPairingFixture(t)
	_ = f.authenticatedSuccess(t)
	receipt, ok := f.inviter.getPairingReceipt(f.session.inviteID)
	if !ok {
		t.Fatal("missing durable correlated receipt")
	}
	var contacted atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { contacted.Add(1); respondPairing(w, nil) }))
	defer target.Close()
	var replays atomic.Int32
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var env pairingEnvelope
		if err := json.NewDecoder(r.Body).Decode(&env); err != nil {
			t.Error(err)
			return
		}
		if env.InviteID != receipt.InviteID || env.Phase != "outbound-completion" {
			t.Error("receipt replay lost exact child identity")
		}
		replays.Add(1)
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()
	receipt.Addr = strings.TrimPrefix(redirect.URL, "http://")
	if err := f.inviter.savePairingReceipt(receipt); err != nil {
		t.Fatal(err)
	}
	restarted, err := NewManager(NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), io.Discard}), filepath.Dir(f.inviter.clusterDir), f.inviter.port)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(restarted.clients.CloseIdle)
	t.Cleanup(restarted.peerServices.CloseIdle)
	t.Cleanup(restarted.removalPeerServices.CloseIdle)
	if len(restarted.joinRequests) != 0 {
		t.Fatal("restart retained memory-only requests")
	}
	restarted.resumeOutboundPairings()
	if replays.Load() != 1 || contacted.Load() != 0 {
		t.Fatalf("replays = %d, redirect contacts = %d", replays.Load(), contacted.Load())
	}
	inv, ok := restarted.getInvite(receipt.InviteID)
	if !ok || inv.State != inviteStatePending || inv.Reason != reasonAwaitingConfirmation {
		t.Fatal("replay redirect changed uncertain durable outcome")
	}
	if len(restarted.trust.List()) != 1 {
		t.Fatal("replay redirect removed a durable peer")
	}
}
