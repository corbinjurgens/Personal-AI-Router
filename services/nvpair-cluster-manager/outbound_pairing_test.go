// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"eapnoob"
	"nvpair-shared/clustertrust"
)

type outboundPairingFixture struct {
	inviter     *Manager
	joiner      *Manager
	session     *pairingSession
	joinSession *pairingSession
	pin         string
}

func newOutboundPairingFixture(t *testing.T) outboundPairingFixture {
	t.Helper()
	a := testManagerAt(t, t.TempDir(), 1)
	b := testManagerAt(t, t.TempDir(), 2)
	activateTestCluster(t, a, "outbound-test-cluster")
	server := httptest.NewServer(http.HandlerFunc(b.handlePairing))
	t.Cleanup(server.Close)
	id := "outbound-test-invite"
	a.putInvite(&Invite{InviteID: id, FromNodeUUID: a.identity.NodeUUID,
		ClusterID: "outbound-test-cluster", State: inviteStatePending, CreatedAt: time.Now().UnixMilli()})
	sess, pin, err := a.runInitialExchange(id, strings.TrimPrefix(server.URL, "http://"), "outbound-test-cluster", a.sessGen.Load(), false)
	if err != nil {
		t.Fatalf("initial exchange: %v", err)
	}
	joinSession, ok := b.getSession(id)
	if !ok || sess.completionMode != "inviter-v1" || joinSession.completionMode != "inviter-v1" {
		t.Fatal("initial exchange did not negotiate inviter-v1")
	}
	joinChoice := false
	joinSession.allowRemoteManagement = &joinChoice
	return outboundPairingFixture{a, b, sess, joinSession, pin}
}

func (f outboundPairingFixture) ready(t *testing.T) {
	t.Helper()
	if err := f.joinSession.peer.OOBInputNoob(noobFromPIN(f.pin)); err != nil {
		t.Fatalf("enter pin: %v", err)
	}
	f.joinSession.ready = true
}

// authenticatedSuccess stops at the durable inviter commit, before delivering
// Success to the joiner. Tests can fault each join transaction boundary here.
func (f outboundPairingFixture) authenticatedSuccess(t *testing.T) []byte {
	t.Helper()
	f.ready(t)
	blob, err := f.session.server.Start()
	if err != nil {
		t.Fatal(err)
	}
	for {
		rr := httptest.NewRecorder()
		f.joiner.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: f.session.inviteID}, blob)
		if rr.Code != http.StatusOK {
			t.Fatalf("completion HTTP status = %d: %s", rr.Code, rr.Body.String())
		}
		var env pairingEnvelope
		if err := json.Unmarshal(rr.Body.Bytes(), &env); err != nil {
			t.Fatal(err)
		}
		response, err := base64.StdEncoding.DecodeString(env.Msg)
		if err != nil {
			t.Fatal(err)
		}
		out, err := f.session.server.Receive(response)
		if err != nil || out.Err != nil {
			t.Fatalf("server completion: %v / %v", err, out.Err)
		}
		if out.Done {
			if !out.Success {
				t.Fatal("server failed correct PIN")
			}
			if err := f.inviter.onInviterPaired(f.session.inviteID, f.session); err != nil {
				t.Fatalf("inviter commit: %v", err)
			}
			return out.Send
		}
		blob = out.Send
	}
}

func (f outboundPairingFixture) confirm(t *testing.T, cert *x509.Certificate, joinerEpoch uint64) *httptest.ResponseRecorder {
	t.Helper()
	rr := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, pairingPath, nil)
	r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{cert}}
	_, epoch := f.inviter.currentAdmission()
	f.joiner.handlePairingConfirmation(rr, r, &pairingEnvelope{InviteID: f.session.inviteID,
		ClusterID: "outbound-test-cluster", InviterEpoch: epoch, JoinerEpoch: joinerEpoch})
	return rr
}

func identityCertificate(t *testing.T, m *Manager) *x509.Certificate {
	t.Helper()
	info, err := json.Marshal(m.localPairingInfo("", 1))
	if err != nil {
		t.Fatal(err)
	}
	_, cert, err := parsePairingInfo(info)
	if err != nil {
		t.Fatal(err)
	}
	return cert
}

func TestOutboundPairingRejectsForgedSuccessAndModeSwitch(t *testing.T) {
	f := newOutboundPairingFixture(t)
	f.ready(t)
	for _, msg := range []string{`{"eap":"success"}`, `{"Type":2}`, `{"Type":6}`} {
		rr := httptest.NewRecorder()
		f.joiner.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: f.session.inviteID}, []byte(msg))
		if rr.Code != http.StatusConflict {
			t.Fatalf("out-of-order completion %s status = %d, want 409", msg, rr.Code)
		}
	}
	if cid, epoch := f.joiner.currentAdmission(); cid != "" || epoch != 0 {
		t.Fatalf("forged completion activated admission %q/%d", cid, epoch)
	}
	if f.joinSession.peer.State() != eapnoob.StateOOBReceived {
		t.Fatalf("rejected message changed EAP state: %s", f.joinSession.peer.State())
	}
	rr := httptest.NewRecorder()
	f.joiner.handlePairingInitial(rr, &pairingEnvelope{InviteID: f.session.inviteID}, []byte(`{"Type":2}`), "127.0.0.1")
	if rr.Code != http.StatusConflict {
		t.Fatalf("new Initial changed negotiated transcript: status %d", rr.Code)
	}
	rr = httptest.NewRecorder()
	f.inviter.handlePairingCompletion(rr, &pairingEnvelope{InviteID: f.session.inviteID}, nil, "127.0.0.1")
	if rr.Code != http.StatusConflict {
		t.Fatalf("legacy completion switched negotiated mode: status %d", rr.Code)
	}
}

func TestOutboundPairingFinalReplyIsIdempotent(t *testing.T) {
	f := newOutboundPairingFixture(t)
	final := f.authenticatedSuccess(t)
	for attempt := 0; attempt < 2; attempt++ {
		rr := httptest.NewRecorder()
		f.joiner.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: f.session.inviteID}, final)
		if rr.Code != http.StatusOK {
			t.Fatalf("final delivery %d status = %d", attempt, rr.Code)
		}
	}
	inv, _ := f.joiner.getInvite(f.session.inviteID)
	if inv.State != inviteStatePaired {
		t.Fatalf("joiner state = %q, want paired", inv.State)
	}
	_, epoch := f.joiner.currentAdmission()
	if epoch != f.joinSession.localAdmissionEpoch {
		t.Fatalf("repeat final changed admission: %d", epoch)
	}
	if rr := f.confirm(t, identityCertificate(t, f.inviter), epoch); rr.Code != http.StatusOK {
		t.Fatalf("matching confirmation status = %d", rr.Code)
	}
	if rr := f.confirm(t, identityCertificate(t, f.inviter), epoch+1); rr.Code != http.StatusForbidden {
		t.Fatalf("wrong-incarnation confirmation status = %d", rr.Code)
	}
	stranger := testManagerAt(t, t.TempDir(), 3)
	if rr := f.confirm(t, identityCertificate(t, stranger), epoch); rr.Code != http.StatusForbidden {
		t.Fatalf("wrong-transcript certificate status = %d", rr.Code)
	}
}

func TestOutboundPairingRemoteManagementChoicesAreIndependent(t *testing.T) {
	f := newOutboundPairingFixture(t)
	inviterChoice, joinerChoice := true, false
	f.session.allowRemoteManagement = &inviterChoice
	f.joinSession.allowRemoteManagement = &joinerChoice
	final := f.authenticatedSuccess(t)
	rr := httptest.NewRecorder()
	f.joiner.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: f.session.inviteID}, final)
	if rr.Code != http.StatusOK {
		t.Fatalf("completion status=%d, want 200", rr.Code)
	}
	inviterGrant := clustertrust.Open(f.inviter.clusterDir).RemoteManagement(f.joiner.identity.NodeUUID)
	joinerGrant := clustertrust.Open(f.joiner.clusterDir).RemoteManagement(f.inviter.identity.NodeUUID)
	if !inviterGrant.Allowed || inviterGrant.State != "allowed" {
		t.Fatalf("inviter local allow = %+v", inviterGrant)
	}
	if joinerGrant.Allowed || joinerGrant.State != "denied" {
		t.Fatalf("joiner local deny was overwritten by inviter: %+v", joinerGrant)
	}
}

func TestOutboundPairingJoinCommitFailureCanConfirmWithoutPin(t *testing.T) {
	test := func(name, boundary string) {
		t.Run(name, func(t *testing.T) {
			f := newOutboundPairingFixture(t)
			final := f.authenticatedSuccess(t)
			path := f.joiner.membersPath()
			if boundary == "pin" {
				path = f.joiner.trust.pinPath(f.inviter.identity.NodeUUID)
			}
			if boundary == "admission" {
				path = f.joiner.admissionPath()
			}
			restore := conflictPairingPath(t, path)
			rr := httptest.NewRecorder()
			f.joiner.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: f.session.inviteID}, final)
			inv, _ := f.joiner.getInvite(f.session.inviteID)
			want := inviteStateFailed
			if boundary != "admission" {
				want = inviteStatePending
			}
			if inv.State != want {
				t.Fatalf("commit failure reported %q, want %q", inv.State, want)
			}
			cert := identityCertificate(t, f.inviter)
			r := httptest.NewRequest(http.MethodPost, rosterPath, nil)
			r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{cert}}
			if _, ok := f.joiner.verifyClientPin(r); ok {
				t.Fatal("failed join granted general peer authority")
			}
			rr = f.confirm(t, cert, f.joinSession.localAdmissionEpoch)
			if rr.Code != http.StatusOK {
				t.Fatalf("failed pin prevents exact-invite confirmation: %d", rr.Code)
			}
			var outcome pairingEnvelope
			if err := json.Unmarshal(rr.Body.Bytes(), &outcome); err != nil {
				t.Fatal(err)
			}
			if outcome.State != want {
				t.Fatalf("authenticated outcome = %q, want %q", outcome.State, want)
			}
			restore()
			f.joiner.expirePendingInvites(time.Now())
			inv, _ = f.joiner.getInvite(f.session.inviteID)
			if inv.State != inviteStateFailed {
				t.Fatalf("durable compensation outcome = %q, want failed", inv.State)
			}
			f.inviter.handlePairingTerminalSignal(&pairingEnvelope{InviteID: f.session.inviteID}, inviteStateFailed, "", "cluster:invite-failed", "")
			if _, ok := f.inviter.trust.Get(f.joiner.identity.NodeUUID); !ok {
				t.Fatal("unauthenticated failure rolled back inviter")
			}
			f.inviter.handlePairingTerminalSignal(&pairingEnvelope{InviteID: f.session.inviteID}, inviteStateFailed, "", "cluster:invite-failed", f.joiner.identity.NodeUUID)
			if _, ok := f.inviter.trust.Get(f.joiner.identity.NodeUUID); ok {
				t.Fatal("matching authenticated commit failure retained inviter pin")
			}
		})
	}
	test("membership persistence failure", "member")
	test("pin persistence failure", "pin")
	test("admission activation failure", "admission")
}

func TestOutboundPairingUnknownConfirmationKeepsCommittedInviter(t *testing.T) {
	f := newOutboundPairingFixture(t)
	f.authenticatedSuccess(t)
	// The fixture's listener is plain HTTP only: authenticated confirmation
	// fails. This cannot be treated as the joiner's failed commit.
	f.inviter.confirmOutboundPairing(f.session)
	if _, ok := f.inviter.trust.Get(f.joiner.identity.NodeUUID); !ok {
		t.Fatal("unknown confirmation rolled back committed inviter pin")
	}
	inv, _ := f.inviter.getInvite(f.session.inviteID)
	if inv.State != inviteStatePaired {
		t.Fatalf("unknown confirmation changed committed outcome to %q", inv.State)
	}
}

func TestOutboundPairingCompetingReadyCommitRechecksStandalone(t *testing.T) {
	f := newOutboundPairingFixture(t)
	c := testManagerAt(t, t.TempDir(), 3)
	activateTestCluster(t, c, "other-cluster")
	id := "competing-ready-invite"
	c.putInvite(&Invite{InviteID: id, FromNodeUUID: c.identity.NodeUUID,
		ClusterID: "other-cluster", State: inviteStatePending, CreatedAt: time.Now().UnixMilli()})
	sess, pin, err := c.runInitialExchange(id, f.session.joinerAddr, "other-cluster", c.sessGen.Load(), false)
	if err != nil {
		t.Fatal(err)
	}
	joinSess, ok := f.joiner.getSession(id)
	if !ok {
		t.Fatal("second inbound session missing")
	}
	other := outboundPairingFixture{c, f.joiner, sess, joinSess, pin}
	firstFinal := f.authenticatedSuccess(t)
	secondFinal := other.authenticatedSuccess(t)
	// Both inviters completed the MAC exchange while the same joiner was
	// standalone. Their final requests now compete for the composition commit.
	rr := httptest.NewRecorder()
	f.joiner.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: f.session.inviteID}, firstFinal)
	rr = httptest.NewRecorder()
	f.joiner.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: other.session.inviteID}, secondFinal)
	if cid, epoch := f.joiner.currentAdmission(); cid != "outbound-test-cluster" || epoch != f.joinSession.localAdmissionEpoch {
		t.Fatalf("losing invitation changed winner admission to %q/%d", cid, epoch)
	}
	if _, pinned := f.joiner.trust.Get(c.identity.NodeUUID); pinned {
		t.Fatal("losing invitation pinned its inviter")
	}
	if _, pinned := f.joiner.trust.Get(f.inviter.identity.NodeUUID); !pinned {
		t.Fatal("losing invitation erased winner's pin")
	}
	inv, _ := f.joiner.getInvite(id)
	if inv.State != inviteStateFailed {
		t.Fatalf("losing completion reported %q", inv.State)
	}
}

func TestOutboundPairingTerminalCacheExpiresWithoutRollback(t *testing.T) {
	f := newOutboundPairingFixture(t)
	final := f.authenticatedSuccess(t)
	rr := httptest.NewRecorder()
	f.joiner.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: f.session.inviteID}, final)
	f.joiner.expirePendingInvites(time.Now().Add(6 * time.Minute))
	if _, live := f.joiner.getSession(f.session.inviteID); live {
		t.Fatal("completed joiner session survived terminal cache TTL")
	}
	if _, live := f.joiner.getInvite(f.session.inviteID); live {
		t.Fatal("terminal invite survived cache TTL")
	}
	if _, pinned := f.joiner.trust.Get(f.inviter.identity.NodeUUID); !pinned {
		t.Fatal("terminal cache expiry undid durable membership pin")
	}
}

func TestOutboundPairingReceiptRecoversCommitAndFailureAcrossRestart(t *testing.T) {
	test := func(name string, deliverFinal bool) {
		t.Run(name, func(t *testing.T) {
			f := newOutboundPairingFixture(t)
			final := f.authenticatedSuccess(t)
			if deliverFinal {
				rr := httptest.NewRecorder()
				f.joiner.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: f.session.inviteID}, final)
				if rr.Code != http.StatusOK {
					t.Fatal(rr.Code)
				}
			} else {
				inv, _ := f.joiner.getInvite(f.session.inviteID)
				if _, err := f.joiner.stagePairingReceiptLocked(f.joinSession, inv); err != nil {
					t.Fatal(err)
				}
				// A process crash after receipt and provisional member persistence,
				// before admission activation. Startup must roll this state back.
				f.joiner.addSelfMemberForAdmission(inv.ClusterID, f.joinSession.localAdmissionEpoch)
				if err := f.joiner.persistMembersErr(); err != nil {
					t.Fatal(err)
				}
			}
			restarted := testManagerAt(t, filepath.Dir(f.joiner.clusterDir), 4)
			f.joiner = restarted
			rr := f.confirm(t, identityCertificate(t, f.inviter), f.joinSession.localAdmissionEpoch)
			if rr.Code != http.StatusOK {
				t.Fatalf("restart confirmation status = %d: %s", rr.Code, rr.Body.String())
			}
			var out pairingEnvelope
			if err := json.Unmarshal(rr.Body.Bytes(), &out); err != nil {
				t.Fatal(err)
			}
			want := inviteStateFailed
			if deliverFinal {
				want = inviteStatePaired
			}
			if out.State != want {
				t.Fatalf("restart outcome = %q, want %q", out.State, want)
			}
		})
	}
	test("after durable join commit", true)
	test("before admission activation", false)
}

func TestPairingCompletionCapabilityKeepsV2EpochRequired(t *testing.T) {
	m := testManagerAt(t, t.TempDir(), 5)
	pi := m.localPairingInfo("", 1)
	pi.AdmissionEpoch = 0
	for _, version := range []int{2, 3} {
		pi.V = version
		raw, err := json.Marshal(pi)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := parsePairingInfo(raw); err == nil {
			t.Fatalf("v%d without admission epoch accepted", version)
		}
	}
}

func TestOutboundPairingNetworkWaitDoesNotHoldGlobalLocks(t *testing.T) {
	f := newOutboundPairingFixture(t)
	started := make(chan struct{}, 1)
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var env pairingEnvelope
		if err := json.NewDecoder(r.Body).Decode(&env); err != nil {
			http.Error(w, "invalid test request", http.StatusBadRequest)
			return
		}
		if env.Phase == "outbound-ready" {
			started <- struct{}{}
			<-release
		}
		respondPairingEnvelope(w, pairingEnvelope{InviteID: env.InviteID, State: inviteStatePending})
	}))
	defer server.Close()
	f.session.joinerAddr = strings.TrimPrefix(server.URL, "http://")
	done := make(chan struct{})
	go func() { f.inviter.driveOutboundCompletion(f.session); close(done) }()
	select {
	case <-started:
	case <-time.After(time.Second):
		close(release)
		t.Fatal("readiness network wait did not start")
	}
	if !f.inviter.inviteMu.TryLock() {
		close(release)
		t.Fatal("readiness network wait held invitation lock")
	}
	f.inviter.inviteMu.Unlock()
	if !f.inviter.rosterMu.TryLock() {
		close(release)
		t.Fatal("readiness network wait held composition lock")
	}
	f.inviter.rosterMu.Unlock()
	params, err := json.Marshal(cancelInviteParams{InviteID: f.session.inviteID})
	if err != nil {
		close(release)
		t.Fatal(err)
	}
	f.inviter.handleCancelInvite(&Message{Params: params})
	close(release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("cancel did not stop readiness poll")
	}
	inv, _ := f.inviter.getInvite(f.session.inviteID)
	if inv.State != inviteStateCanceled {
		t.Fatalf("cancel during network wait state = %q", inv.State)
	}
}
