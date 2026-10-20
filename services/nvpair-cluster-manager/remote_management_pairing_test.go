// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
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
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/durablefile"
)

func TestInviteRequiresLocalRemoteManagementChoiceBeforePairingWork(t *testing.T) {
	for _, tc := range []struct {
		name   string
		choice string
	}{
		{name: "missing"},
		{name: "null", choice: `,"allowRemoteManagement":null`},
		{name: "nonboolean", choice: `,"allowRemoteManagement":"false"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newTestManager(t)
			var output bytes.Buffer
			m.codec = NewCodec(struct {
				io.Reader
				io.Writer
			}{strings.NewReader(""), &output})
			id := json.RawMessage("1")
			params := json.RawMessage(`{"address":"127.0.0.1","port":1` + tc.choice + `}`)

			m.handleInviteNode(&Message{ID: &id, Params: params})

			response := decodePairingRPCResponse(t, &output)
			if response.Error == nil || response.Error.Code != codeInvalidParams {
				t.Fatalf("error = %+v, want code %d", response.Error, codeInvalidParams)
			}
			if cid, _ := m.clusterIdentity(); cid != "cluster-1" {
				t.Fatalf("cluster identity = %q, want unchanged cluster-1", cid)
			}
			m.memMu.Lock()
			invites := len(m.invites)
			m.memMu.Unlock()
			if invites != 0 {
				t.Fatalf("invite count = %d, want 0", invites)
			}
			m.sessMu.Lock()
			sessions := len(m.sessions)
			m.sessMu.Unlock()
			if sessions != 0 {
				t.Fatalf("session count = %d, want 0", sessions)
			}
		})
	}
}

func TestRespondAcceptRequiresLocalRemoteManagementChoice(t *testing.T) {
	for _, tc := range []struct {
		name   string
		choice string
	}{
		{name: "missing"},
		{name: "null", choice: `,"allowRemoteManagement":null`},
		{name: "nonboolean", choice: `,"allowRemoteManagement":1`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newTestManager(t)
			var output bytes.Buffer
			m.codec = NewCodec(struct {
				io.Reader
				io.Writer
			}{strings.NewReader(""), &output})
			inviteID := "inv-permission-required"
			m.putInvite(&Invite{InviteID: inviteID, ClusterID: "cluster-1", State: inviteStatePending})
			sess := &pairingSession{inviteID: inviteID, role: roleJoiner}
			m.putSession(sess)
			id := json.RawMessage("2")
			params := json.RawMessage(`{"inviteId":"` + inviteID + `","accept":true,"pin":"123456"` + tc.choice + `}`)

			m.handleRespondToInvite(&Message{ID: &id, Params: params})

			response := decodePairingRPCResponse(t, &output)
			if response.Error == nil || response.Error.Code != codeInvalidParams {
				t.Fatalf("error = %+v, want code %d", response.Error, codeInvalidParams)
			}
			stored, ok := m.getInvite(inviteID)
			if !ok || stored.State != inviteStatePending {
				t.Fatalf("invite = %+v, found=%v, want pending", stored, ok)
			}
			current, ok := m.getSession(inviteID)
			if !ok || current != sess || current.allowRemoteManagement != nil {
				t.Fatalf("session = %+v, found=%v, want unchanged session without a choice", current, ok)
			}
		})
	}
}

func TestRespondDeclineDoesNotRequireLocalRemoteManagementChoice(t *testing.T) {
	m := newTestManager(t)
	var output bytes.Buffer
	m.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), &output})
	inviteID := "inv-decline-without-permission"
	m.putInvite(&Invite{InviteID: inviteID, ClusterID: "cluster-1", State: inviteStatePending})
	m.putSession(&pairingSession{inviteID: inviteID, role: roleJoiner})
	id := json.RawMessage("3")
	params := json.RawMessage(`{"inviteId":"` + inviteID + `","accept":false}`)

	m.handleRespondToInvite(&Message{ID: &id, Params: params})

	response := decodePairingRPCResponse(t, &output)
	if response.Error != nil {
		t.Fatalf("error = %+v, want successful decline", response.Error)
	}
	stored, ok := m.getInvite(inviteID)
	if !ok || stored.State != inviteStateDeclined {
		t.Fatalf("invite = %+v, found=%v, want declined", stored, ok)
	}
	if _, ok := m.getSession(inviteID); ok {
		t.Fatal("declined invite session remains installed")
	}
}

func decodePairingRPCResponse(t *testing.T, output *bytes.Buffer) Message {
	t.Helper()
	var response Message
	if err := json.NewDecoder(output).Decode(&response); err != nil {
		t.Fatalf("decode JSON-RPC response: %v", err)
	}
	return response
}

type localPermissionPairing struct {
	target  *Manager
	peer    *Manager
	session *pairingSession
	invite  *Invite
	info    *PairingInfo
	cert    *x509.Certificate
}

// This fixture uses both real identity stores and no network listener. The two
// completion transports converge on these same role-specific commit owners.
func newLocalPermissionPairing(t *testing.T, role pairingRole, allowed bool) localPermissionPairing {
	t.Helper()
	m := testManagerAt(t, t.TempDir(), 1)
	peer := testManagerAt(t, t.TempDir(), 2)
	clusterID := "permission-pairing"
	var epoch uint64
	if role == roleInviter {
		epoch = activateTestCluster(t, m, clusterID)
	} else {
		activateTestCluster(t, peer, clusterID)
		var err error
		epoch, err = m.reserveAdmissionEpoch()
		if err != nil {
			t.Fatal(err)
		}
	}
	pi := peer.localPairingInfo("", 1)
	raw, err := json.Marshal(pi)
	if err != nil {
		t.Fatal(err)
	}
	pi, cert, err := parsePairingInfo(raw)
	if err != nil {
		t.Fatal(err)
	}
	inv := &Invite{InviteID: "permission-invite", ClusterID: clusterID,
		State: inviteStatePending, CreatedAt: time.Now().UnixMilli()}
	if role == roleInviter {
		inv.FromNodeUUID = m.identity.NodeUUID
	} else {
		inv.FromNodeUUID = peer.identity.NodeUUID
	}
	sess := &pairingSession{inviteID: inv.InviteID, role: role, localAdmissionEpoch: epoch,
		allowRemoteManagement: &allowed, peerPairing: pi, peerCert: cert}
	if role == roleInviter {
		sess.clusterID = clusterID
	}
	m.putInvite(inv)
	m.putSession(sess)
	return localPermissionPairing{m, peer, sess, inv, pi, cert}
}

func (f localPermissionPairing) commit() error {
	if f.session.role == roleJoiner {
		return f.target.commitJoinerPairing(f.session, f.invite)
	}
	committed, err := f.target.commitPairing(f.session, f.info, f.cert, time.Now().UnixMilli())
	if err == nil && !committed {
		return fmt.Errorf("inviter transaction did not commit")
	}
	return err
}

func TestPairingRemoteManagementPersistsTargetLocalChoice(t *testing.T) {
	test := func(name string, role pairingRole, allowed bool) {
		t.Run(name, func(t *testing.T) {
			f := newLocalPermissionPairing(t, role, allowed)
			if err := f.commit(); err != nil {
				t.Fatal(err)
			}
			got := clustertrust.Open(f.target.clusterDir).RemoteManagement(f.peer.identity.NodeUUID)
			wantState := "denied"
			if allowed {
				wantState = "allowed"
			}
			if got.Allowed != allowed || got.State != wantState || got.Revision != "2" {
				t.Fatalf("permission = %+v, want allowed=%t state=%s revision=2", got, allowed, wantState)
			}
			receipt, ok := f.target.getPairingReceipt(f.invite.InviteID)
			if !ok || receipt.Rollback || receipt.State != inviteStatePaired || receipt.PermissionRevision != "1" || receipt.AllowRemoteManagement == nil || *receipt.AllowRemoteManagement != allowed {
				t.Fatalf("durable transaction receipt = %+v, found=%t", receipt, ok)
			}
			if _, err := os.Stat(f.target.remoteManagementInitializingPath()); !os.IsNotExist(err) {
				t.Fatalf("completed transaction guard remains: %v", err)
			}
		})
	}
	test("inviter allows joiner", roleInviter, true)
	test("inviter denies joiner", roleInviter, false)
	test("joiner allows inviter", roleJoiner, true)
	test("joiner denies inviter", roleJoiner, false)
}

func TestPairingRemoteManagementReceiptStagingFailureCreatesNoAuthority(t *testing.T) {
	test := func(name string, role pairingRole, landed bool) {
		t.Run(name, func(t *testing.T) {
			f := newLocalPermissionPairing(t, role, true)
			original := clusterFileReplacer
			t.Cleanup(func() { clusterFileReplacer = original })
			path := filepath.Join(f.target.clusterDir, pairingReceiptsFile)
			clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
				if target == path {
					if landed {
						if err := os.Rename(tmp, target); err != nil {
							return err
						}
						return durablefile.ErrReplaced
					}
					return errors.New("receipt staging fault")
				}
				return os.Rename(tmp, target)
			}}
			if err := f.commit(); err == nil {
				t.Fatal("reported pairing success without durable local intent")
			}
			if _, ok := f.target.trust.Get(f.peer.identity.NodeUUID); ok {
				t.Fatal("staging failure created a pin")
			}
			if _, ok := f.target.memberByNodeID(f.peer.identity.NodeUUID); ok {
				t.Fatal("staging failure created a member")
			}
			if got := clustertrust.Open(f.target.clusterDir).RemoteManagement(f.peer.identity.NodeUUID); got.Allowed {
				t.Fatalf("staging failure allowed peer: %+v", got)
			}
		})
	}
	test("inviter before replacement", roleInviter, false)
	test("inviter after replacement", roleInviter, true)
	test("joiner before replacement", roleJoiner, false)
	test("joiner after replacement", roleJoiner, true)
}

func TestPairingRemoteManagementGrantFailureCompensatesActivatedAdmission(t *testing.T) {
	test := func(name string, role pairingRole, landed bool) {
		t.Run(name, func(t *testing.T) {
			f := newLocalPermissionPairing(t, role, true)
			original := clusterFileReplacer
			t.Cleanup(func() { clusterFileReplacer = original })
			path, _ := f.target.remoteManagementPaths()
			calls := 0
			clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
				if target != path {
					return os.Rename(tmp, target)
				}
				calls++
				if !landed {
					return errors.New("grant replacement fault")
				}
				if err := os.Rename(tmp, target); err != nil {
					return err
				}
				return durablefile.ErrReplaced
			}}
			if err := f.commit(); err == nil {
				t.Fatal("reported success after grant persistence failure")
			}
			if calls == 0 {
				t.Fatal("fault did not reach the grant writer")
			}
			if got := clustertrust.Open(f.target.clusterDir).RemoteManagement(f.peer.identity.NodeUUID); got.Allowed {
				t.Fatalf("fresh reader allowed a failed transaction: %+v", got)
			}
			if role == roleJoiner {
				if cid, epoch := f.target.currentAdmission(); cid != "" || epoch != 0 {
					t.Fatalf("activated admission survived failed grant: %q/%d", cid, epoch)
				}
				f.target.admissionMu.Lock()
				activated, counter := f.target.admissionActivated, f.target.admissionCounter
				f.target.admissionMu.Unlock()
				if activated != f.session.localAdmissionEpoch || counter < activated {
					t.Fatalf("retirement changed high-water marks: activated=%d counter=%d", activated, counter)
				}
			}
		})
	}
	test("inviter before replacement", roleInviter, false)
	test("inviter after replacement", roleInviter, true)
	test("joiner before replacement", roleJoiner, false)
	test("joiner after replacement", roleJoiner, true)
}

func TestPairingRemoteManagementTerminalReceiptFaultKeepsIndependentReadersDenied(t *testing.T) {
	test := func(name string, role pairingRole, landed bool) {
		t.Run(name, func(t *testing.T) {
			f := newLocalPermissionPairing(t, role, true)
			original := clusterFileReplacer
			originalRemove := removeFile
			t.Cleanup(func() {
				clusterFileReplacer = original
				removeFile = originalRemove
			})
			receiptPath := filepath.Join(f.target.clusterDir, pairingReceiptsFile)
			terminalReached := false
			compensationReceiptFaulted := false
			removeFile = func(path string) error {
				if terminalReached && path == f.target.trust.pinPath(f.peer.identity.NodeUUID) {
					return errors.New("compensation pin deletion fault")
				}
				return originalRemove(path)
			}
			clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
				if target == receiptPath {
					data, err := os.ReadFile(tmp)
					if err != nil {
						return err
					}
					var receipts map[string]pairingReceipt
					if err := json.Unmarshal(data, &receipts); err != nil {
						return err
					}
					if receipts[f.invite.InviteID].State == inviteStatePaired {
						terminalReached = true
						if got := clustertrust.Open(f.target.clusterDir).RemoteManagement(f.peer.identity.NodeUUID); got.Allowed {
							t.Fatalf("grant published before terminal receipt: %+v", got)
						}
						if landed {
							if err := os.Rename(tmp, target); err != nil {
								return err
							}
							return durablefile.ErrReplaced
						}
						return errors.New("terminal receipt fault")
					}
					if terminalReached {
						compensationReceiptFaulted = true
					}
				}
				if terminalReached && target != f.target.remoteManagementInitializingPath() {
					return errors.New("compensation persistence fault")
				}
				return os.Rename(tmp, target)
			}}
			if err := f.commit(); err == nil || !terminalReached || !compensationReceiptFaulted {
				t.Fatalf("terminal fault error=%v terminal=%t compensation receipt=%t", err, terminalReached, compensationReceiptFaulted)
			}
			if got := clustertrust.Open(f.target.clusterDir).RemoteManagement(f.peer.identity.NodeUUID); got.Allowed {
				t.Fatalf("failed compensation exposed a landed allow: %+v", got)
			}
			if !f.session.rollbackPending {
				t.Fatal("failed compensation lost its live retry")
			}
			if err := f.target.finalizeRemoteManagementMigrationLocked(); err == nil {
				t.Fatal("migration finalizer released an active pairing guard")
			}
			clusterFileReplacer = original
			removeFile = originalRemove
			restarted := testManagerAt(t, filepath.Dir(f.target.clusterDir), 3)
			if got := clustertrust.Open(restarted.clusterDir).RemoteManagement(f.peer.identity.NodeUUID); got.Allowed {
				t.Fatalf("restart replayed aborted permission: %+v", got)
			}
			receipt, ok := restarted.getPairingReceipt(f.invite.InviteID)
			if !ok || receipt.Rollback || receipt.State != inviteStateFailed {
				t.Fatalf("restart outcome = %+v found=%t, want failed", receipt, ok)
			}
		})
	}
	test("inviter terminal save fails before replace", roleInviter, false)
	test("inviter terminal save remains uncertain", roleInviter, true)
	test("joiner terminal save fails before replace", roleJoiner, false)
	test("joiner terminal save remains uncertain", roleJoiner, true)
}

func TestPairingRemoteManagementReceiptReplayPreservesLaterLocalDeny(t *testing.T) {
	f := newLocalPermissionPairing(t, roleInviter, true)
	if err := f.commit(); err != nil {
		t.Fatal(err)
	}
	receipt, ok := f.target.getPairingReceipt(f.invite.InviteID)
	if !ok {
		t.Fatal("completed receipt missing")
	}
	if err := f.target.setRemoteManagementLocked(f.peer.identity.NodeUUID, false, "2"); err != nil {
		t.Fatal(err)
	}
	if outcome := f.target.receiptOutcome(receipt, f.info, f.cert); outcome != inviteStatePaired {
		t.Fatalf("completed membership outcome=%s, want paired", outcome)
	}
	restarted := testManagerAt(t, filepath.Dir(f.target.clusterDir), 3)
	if got := clustertrust.Open(restarted.clusterDir).RemoteManagement(f.peer.identity.NodeUUID); got.Allowed || got.State != "denied" || got.Revision != "3" {
		t.Fatalf("stale receipt overwrote local deny: %+v", got)
	}
}

func TestPairingRemoteManagementResumeOutboundReceiptPreservesLaterLocalDeny(t *testing.T) {
	f := newOutboundPairingFixture(t)
	allow := true
	f.session.allowRemoteManagement = &allow
	final := f.authenticatedSuccess(t)
	if len(final) == 0 {
		t.Fatal("completed EAP exchange has no success transcript")
	}
	host, portText, err := net.SplitHostPort(f.session.joinerAddr)
	if err != nil || host != "127.0.0.1" {
		t.Fatalf("pairing replay must use the fixture's IPv4 loopback endpoint: address=%q error=%v", f.session.joinerAddr, err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Timeout: pairingHTTPTimeout}
	t.Cleanup(client.CloseIdleConnections)
	if _, err := postPairingBlob(client, f.session.joinerAddr, f.session.inviteID, "outbound-completion", final); err != nil {
		t.Fatalf("deliver authenticated success to joiner: %v", err)
	}
	if inv, ok := f.joiner.getInvite(f.session.inviteID); !ok || inv.State != inviteStatePaired {
		t.Fatalf("joiner did not complete pairing: invite=%+v found=%t", inv, ok)
	}
	receipt, ok := f.inviter.getPairingReceipt(f.session.inviteID)
	if !ok || receipt.Rollback || receipt.State != inviteStatePaired || receipt.Addr != f.session.joinerAddr || receipt.AllowRemoteManagement == nil || !*receipt.AllowRemoteManagement {
		t.Fatalf("completed inviter receipt is not replayable: found=%t state=%s rollback=%t", ok, receipt.State, receipt.Rollback)
	}
	// The fixture drives the EAP rounds directly. Retain the real final blob
	// as the production outbound driver does before its inviter commit.
	receipt.SuccessBlob = append([]byte(nil), final...)
	if err := f.inviter.savePairingReceipt(receipt); err != nil {
		t.Fatal(err)
	}
	if permission := clustertrust.Open(f.inviter.clusterDir).RemoteManagement(f.joiner.identity.NodeUUID); !permission.Allowed || permission.Revision != "2" {
		t.Fatalf("fixture did not publish the original local allow: %+v", permission)
	}
	f.inviter.rosterMu.Lock()
	err = f.inviter.setRemoteManagementLocked(f.joiner.identity.NodeUUID, false, "2")
	f.inviter.rosterMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	grantPath, _ := f.inviter.remoteManagementPaths()
	denyBytes, err := os.ReadFile(grantPath)
	if err != nil {
		t.Fatal(err)
	}
	f.inviter = testManagerAt(t, filepath.Dir(f.inviter.clusterDir), 1)
	m := f.inviter
	if _, live := m.getSession(receipt.InviteID); live {
		t.Fatal("NewManager unexpectedly installed an outbound replay session")
	}
	// Ack starts best-effort roster reconciliation. Keep that follow-up on the
	// existing fixture endpoint rather than its synthetic transcript port.
	m.setMemberAddr(f.joiner.identity.NodeUUID, host, port)
	member, present := m.memberByNodeID(f.joiner.identity.NodeUUID)
	if !present || member.IPAddress != host || member.Port != port {
		t.Fatal("fixture did not retain a bounded loopback reconciliation route")
	}
	pi, cert, err := parsePairingInfo(receipt.PeerInfo)
	if err != nil {
		t.Fatal(err)
	}
	m.rosterMu.Lock()
	matched := m.receiptCommitMatchesLocked(receipt, pi, cert)
	m.rosterMu.Unlock()
	if !matched {
		t.Fatal("replay fixture no longer has the receipt's exact committed admission")
	}
	assertDenyUnchanged := func(step string) {
		t.Helper()
		current, err := os.ReadFile(grantPath)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(current, denyBytes) {
			t.Fatalf("%s changed the saved local deny document", step)
		}
		if permission := clustertrust.Open(m.clusterDir).RemoteManagement(f.joiner.identity.NodeUUID); permission.Allowed || permission.State != "denied" || permission.Revision != "3" {
			t.Fatalf("%s overwrote the later local deny: %+v", step, permission)
		}
	}
	assertDenyUnchanged("restart")
	m.resumeOutboundPairings()
	replayed, live := m.getSession(receipt.InviteID)
	if !live || replayed == f.session || replayed.role != roleInviter || !replayed.awaitingAck || replayed.rollbackPending ||
		replayed.clusterID != receipt.ClusterID || replayed.localAdmissionEpoch != receipt.LocalEpoch || replayed.joinerAddr != receipt.Addr ||
		replayed.completionMode != inviterCompletionV1 || !bytes.Equal(replayed.completionResponse, final) || replayed.peerPairing == nil ||
		replayed.peerPairing.NodeUUID != pi.NodeUUID || replayed.peerPairing.AdmissionEpoch != pi.AdmissionEpoch || replayed.peerCert == nil ||
		!bytes.Equal(replayed.peerCert.Raw, cert.Raw) || replayed.allowRemoteManagement == nil || !*replayed.allowRemoteManagement {
		t.Fatalf("resume did not install the exact awaiting-ack session: found=%t", live)
	}
	if inv, ok := m.getInvite(receipt.InviteID); !ok || inv.State != inviteStatePending || inv.Reason != reasonAwaitingConfirmation {
		t.Fatalf("resume did not publish awaiting-confirmation state: invite=%+v found=%t", inv, ok)
	}
	assertDenyUnchanged("actual outbound receipt replay")
	// The existing fixture listener handles plain pairing HTTP. Exercise the
	// authenticated confirmation and ack through their existing handler seams.
	confirmed := f.confirm(t, identityCertificate(t, m), f.joinSession.localAdmissionEpoch)
	var outcome pairingEnvelope
	if err := json.Unmarshal(confirmed.Body.Bytes(), &outcome); err != nil {
		t.Fatal(err)
	}
	if confirmed.Code != http.StatusOK || outcome.State != inviteStatePaired || outcome.InviteID != receipt.InviteID ||
		outcome.ClusterID != receipt.ClusterID || outcome.InviterEpoch != receipt.LocalEpoch || outcome.JoinerEpoch != pi.AdmissionEpoch {
		t.Fatalf("authenticated completed-pairing confirmation = HTTP %d %+v", confirmed.Code, outcome)
	}
	assertDenyUnchanged("authenticated confirmation")
	body, err := json.Marshal(pairingEnvelope{InviteID: receipt.InviteID, Phase: "ack"})
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest(http.MethodPost, pairingPath, bytes.NewReader(body))
	r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{cert}}
	rr := httptest.NewRecorder()
	m.handlePairing(rr, r)
	if rr.Code != http.StatusOK {
		t.Fatalf("authenticated replay ack status=%d, want 200", rr.Code)
	}
	if _, live := m.getSession(receipt.InviteID); live {
		t.Fatal("authenticated ack did not finalize the installed replay session")
	}
	if inv, ok := m.getInvite(receipt.InviteID); !ok || inv.State != inviteStatePaired {
		t.Fatalf("authenticated ack did not finalize membership: invite=%+v found=%t", inv, ok)
	}
	assertDenyUnchanged("authenticated ack")
}

func TestPairingRemoteManagementRejectsNetworkPermissionFields(t *testing.T) {
	m := testManagerAt(t, t.TempDir(), 1)
	for _, field := range []string{"allowRemoteManagement", "allowed", "remoteManagement"} {
		t.Run(field, func(t *testing.T) {
			raw, err := json.Marshal(m.localPairingInfo("", 1))
			if err != nil {
				t.Fatal(err)
			}
			raw = append(raw[:len(raw)-1], []byte(`,"`+field+`":true}`)...)
			if _, _, err := parsePairingInfo(raw); err == nil {
				t.Fatal("accepted network permission in authenticated PairingInfo")
			}
			rr := httptest.NewRecorder()
			m.handlePairing(rr, httptest.NewRequest(http.MethodPost, pairingPath,
				strings.NewReader(`{"inviteId":"forged","phase":"initial","`+field+`":true}`)))
			if rr.Code != http.StatusBadRequest {
				t.Fatalf("network permission envelope status=%d, want 400", rr.Code)
			}
		})
	}
}

func TestPairingRemoteManagementGuardIsolation(t *testing.T) {
	test := func(name string, role pairingRole) {
		t.Run(name, func(t *testing.T) {
			f := newLocalPermissionPairing(t, role, true)
			doc, _, err := f.target.readRemoteManagementLocked()
			if err != nil {
				t.Fatal(err)
			}
			if err := f.target.ensureRemoteManagementGuard(doc.MigrationID); err != nil {
				t.Fatal(err)
			}
			if err := f.commit(); err == nil {
				t.Fatal("guarded permission store allowed pairing")
			}
			if _, pinned := f.target.trust.Get(f.peer.identity.NodeUUID); pinned {
				t.Fatal("guarded pairing created a pin")
			}
			if _, member := f.target.memberByNodeID(f.peer.identity.NodeUUID); member {
				t.Fatal("guarded pairing created membership")
			}
		})
	}
	test("inviter commit", roleInviter)
	test("joiner commit", roleJoiner)
}

func TestPairingRemoteManagementCleanupPreservesLaterLocalDeny(t *testing.T) {
	f := newLocalPermissionPairing(t, roleInviter, true)
	if err := f.commit(); err != nil {
		t.Fatal(err)
	}
	if err := f.target.setRemoteManagementLocked(f.peer.identity.NodeUUID, false, "2"); err != nil {
		t.Fatal(err)
	}
	if err := f.target.rollbackPairingSessionCompositionLocked(f.session, f.invite); err != nil {
		t.Fatal(err)
	}
	doc, _, err := f.target.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	if doc.Revision != "3" || len(doc.Grants) != 1 || doc.Grants[0].PeerUUID != f.peer.identity.NodeUUID || doc.Grants[0].Allowed {
		t.Fatalf("compensation erased or rewrote a later deny: %+v", doc)
	}
}

func TestPairingRemoteManagementCleanupRemovesOnlyExactTransactionGrant(t *testing.T) {
	f := newLocalPermissionPairing(t, roleInviter, true)
	if err := f.commit(); err != nil {
		t.Fatal(err)
	}
	if err := f.target.rollbackPairingSessionCompositionLocked(f.session, f.invite); err != nil {
		t.Fatal(err)
	}
	doc, _, err := f.target.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	if doc.Revision != "3" || len(doc.Grants) != 0 {
		t.Fatalf("exact transaction grant survived compensation: %+v", doc)
	}
}

func TestPairingRemoteManagementCleanupPreservesNewerPeerAdmission(t *testing.T) {
	f := newLocalPermissionPairing(t, roleInviter, true)
	if err := f.commit(); err != nil {
		t.Fatal(err)
	}
	newer := *f.info
	newer.ClusterID = f.invite.ClusterID
	newer.AdmissionEpoch++
	now := time.Now().UnixMilli()
	f.target.recordMember(&newer, stateMember, &now)
	if err := f.target.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	if err := f.target.pinPeer(&newer, f.cert, nil); err != nil {
		t.Fatal(err)
	}
	if err := f.target.setRemoteManagementLocked(newer.NodeUUID, true, "2"); err != nil {
		t.Fatal(err)
	}
	if err := f.target.rollbackPairingSessionCompositionLocked(f.session, f.invite); err != nil {
		t.Fatal(err)
	}
	pin, pinned := f.target.trust.Get(newer.NodeUUID)
	member, present := f.target.memberByNodeID(newer.NodeUUID)
	if !pinned || !present || pin.AdmissionEpoch != newer.AdmissionEpoch || member.AdmissionEpoch != newer.AdmissionEpoch {
		t.Fatal("old transaction removed a newer peer admission")
	}
	if got := clustertrust.Open(f.target.clusterDir).RemoteManagement(newer.NodeUUID); !got.Allowed || got.Revision != "3" {
		t.Fatalf("old transaction changed newer admission's grant: %+v", got)
	}
}

func TestPairingRemoteManagementSameAdmissionRepinPreservesLocalChoice(t *testing.T) {
	f := newLocalPermissionPairing(t, roleInviter, false)
	if err := f.commit(); err != nil {
		t.Fatal(err)
	}
	admitted := *f.info
	admitted.ClusterID = f.invite.ClusterID
	if err := f.target.pinPeer(&admitted, f.cert, nil); err != nil {
		t.Fatal(err)
	}
	if got := clustertrust.Open(f.target.clusterDir).RemoteManagement(f.peer.identity.NodeUUID); got.Allowed || got.State != "denied" || got.Revision != "2" {
		t.Fatalf("same-admission repin changed local decision: %+v", got)
	}
}

func TestPairingRemoteManagementLegacyReceiptCannotInventPermission(t *testing.T) {
	f := newLocalPermissionPairing(t, roleInviter, false)
	if err := f.commit(); err != nil {
		t.Fatal(err)
	}
	receipt, ok := f.target.getPairingReceipt(f.invite.InviteID)
	if !ok {
		t.Fatal("completed receipt missing")
	}
	receipt.AllowRemoteManagement = nil
	receipt.PermissionRevision, receipt.PermissionMigrationID, receipt.PermissionIncarnation = "", "", ""
	if err := f.target.savePairingReceipt(receipt); err != nil {
		t.Fatal(err)
	}
	doc, _, err := f.target.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	doc.Grants = make([]clustertrust.RemoteManagementGrant, 0)
	data, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	path, _ := f.target.remoteManagementPaths()
	if err := remoteManagementWrite(path, data); err != nil {
		t.Fatal(err)
	}
	if state := f.target.receiptOutcome(receipt, f.info, f.cert); state != inviteStatePaired {
		t.Fatalf("legacy membership outcome=%s, want paired", state)
	}
	if got := clustertrust.Open(f.target.clusterDir).RemoteManagement(f.peer.identity.NodeUUID); got.Allowed || got.State != "unreviewed" || got.Revision != "2" {
		t.Fatalf("legacy receipt invented a grant: %+v", got)
	}
}

// Build the pre-PG profile before its first real migration. Its receipt has
// only the old membership-confirmation fields; no PG transaction or permission
// setter creates B's initial allow. The upgrade's captured pin is its authority.
func newLegacyPairingRosterReplayProbe(t *testing.T) pairingRosterReplayProbe {
	t.Helper()
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	relay := remoteManagementAddPeer(t, m)
	cid, localEpoch := m.currentAdmission()
	info := peer.localPairingInfo("", 1)
	info.ClusterID = cid
	raw, err := json.Marshal(info)
	if err != nil {
		t.Fatal(err)
	}
	info, cert, err := parsePairingInfo(raw)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.trust.AddEndorsements(peer.identity.NodeUUID,
		[]Endorsement{m.endorsePeer(peer.identity.NodeUUID, peer.identity.CertFingerprint, info.AdmissionEpoch)}); err != nil {
		t.Fatal(err)
	}
	receipt := pairingReceipt{InviteID: "legacy-awaiting-ack", Role: roleInviter,
		ClusterID: cid, LocalEpoch: localEpoch, PeerInfo: raw,
		State: inviteStatePaired, ExpiresAt: time.Now().Add(time.Hour).UnixMilli()}
	legacyBytes, err := json.Marshal(map[string]pairingReceipt{receipt.InviteID: receipt})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(m.clusterDir, pairingReceiptsFile), legacyBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	// NewManager captures and completes the one-time migration from these
	// actual old pins and loads the unchanged pre-PG receipt from disk.
	m = testManagerAt(t, filepath.Dir(m.clusterDir), 15129)
	loaded, ok := m.getPairingReceipt(receipt.InviteID)
	if !ok || loaded.AllowRemoteManagement != nil || loaded.PermissionRevision != "" ||
		loaded.PermissionMigrationID != "" || loaded.PermissionIncarnation != "" || loaded.LocalCertFingerprint != "" {
		t.Fatalf("upgrade did not retain an actual legacy receipt: receipt=%+v found=%t", loaded, ok)
	}
	if state := m.receiptOutcome(loaded, info, cert); state != inviteStatePaired {
		t.Fatalf("legacy committed receipt outcome=%s, want paired", state)
	}
	doc, marker, err := m.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	captured, grandfathered := false, false
	for _, authority := range marker.Snapshot.Peers {
		if authority.PeerUUID == peer.identity.NodeUUID {
			captured = true
		}
	}
	for _, grant := range doc.Grants {
		if grant.PeerUUID == peer.identity.NodeUUID && grant.Allowed && grant.DecisionRevision == "1" {
			grandfathered = true
		}
	}
	if marker.State != "complete" || doc.Revision != "1" || !captured || !grandfathered {
		t.Fatalf("B was not allowed by the original migration snapshot: doc=%+v marker=%+v", doc, marker)
	}
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(peer.identity.NodeUUID); !permission.Allowed || permission.Revision != "1" {
		t.Fatalf("fresh reader lacks B's real grandfathered allow: %+v", permission)
	}
	// Install the same exact awaiting-ack composition as receipt resumption,
	// without invoking its outbound confirmation transports in this probe.
	inv := &Invite{InviteID: receipt.InviteID, FromNodeUUID: m.identity.NodeUUID,
		ClusterID: cid, State: inviteStatePending, Reason: reasonAwaitingConfirmation}
	sess := &pairingSession{inviteID: receipt.InviteID, role: roleInviter,
		clusterID: cid, localAdmissionEpoch: localEpoch, peerPairing: info, peerCert: cert,
		awaitingAck: true, completionMode: inviterCompletionV1}
	m.putInvite(inv)
	m.putSession(sess)
	cached := m.buildLocalRoster()
	validB := false
	for i := range cached.Members {
		entry := &cached.Members[i]
		if entry.NodeUUID == peer.identity.NodeUUID && entry.AdmissionEpoch == info.AdmissionEpoch && m.entryEndorsed(entry, cid) {
			validB = true
		}
	}
	if !validB {
		t.Fatal("legacy cached roster lacks B's valid original pairing endorsement")
	}
	gossipPeer := testManagerAt(t, t.TempDir(), 4)
	gossipEpoch := activateTestCluster(t, gossipPeer, cid)
	cached.Members = append(cached.Members, permissionProbeRosterEntry(m, gossipPeer, gossipEpoch))
	wire, err := json.Marshal(cached)
	if err != nil {
		t.Fatal(err)
	}
	var frozen Roster
	if err := json.Unmarshal(wire, &frozen); err != nil {
		t.Fatal(err)
	}
	return pairingRosterReplayProbe{pairing: localPermissionPairing{m, peer, sess, inv, info, cert},
		relay: relay, gossipPeer: gossipPeer, cached: &frozen}
}

func TestPairingRemoteManagementLegacyReceiptUnwritableRollbackDeniesFreshMesh(t *testing.T) {
	p := newLegacyPairingRosterReplayProbe(t)
	p.editUnrelatedGrant(t)
	f := p.pairing
	journal := filepath.Join(f.target.clusterDir, pairingReceiptsFile)
	before, err := os.ReadFile(journal)
	if err != nil {
		t.Fatal(err)
	}
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	intentFault := errors.New("legacy rollback journal unwritable")
	attempts := 0
	clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
		if target == journal {
			attempts++
			return intentFault
		}
		return os.Rename(tmp, target)
	}}
	body, err := json.Marshal(pairingEnvelope{InviteID: f.invite.InviteID, Phase: "fail"})
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest(http.MethodPost, pairingPath, bytes.NewReader(body))
	r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{f.cert}}
	rr := httptest.NewRecorder()
	f.target.handlePairing(rr, r)
	if rr.Code != http.StatusOK || attempts != 1 || !f.session.rollbackPending {
		t.Fatalf("authenticated legacy failure did not reach its journal fault: status=%d attempts=%d pending=%t", rr.Code, attempts, f.session.rollbackPending)
	}
	inv, ok := f.target.getInvite(f.invite.InviteID)
	if !ok || inv.State != inviteStatePending || inv.Reason != reasonRollbackPending {
		t.Fatalf("failed rollback did not retain its pending invite: invite=%+v found=%t", inv, ok)
	}
	after, err := os.ReadFile(journal)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("pre-replacement journal failure changed the legacy receipt")
	}
	if _, pinned := f.target.trust.Get(f.peer.identity.NodeUUID); pinned {
		t.Fatal("failed rollback retained B's live process pin")
	}
	if permission := clustertrust.Open(f.target.clusterDir).RemoteManagement(f.peer.identity.NodeUUID); permission.Allowed {
		t.Fatalf("fresh Mesh allowed legacy B after authenticated failure with an unwritable rollback journal: %+v", permission)
	}
}

func TestPairingRemoteManagementLegacyReceiptCachedRosterCannotReviveGrandfatheredGrant(t *testing.T) {
	p := newLegacyPairingRosterReplayProbe(t)
	p.editUnrelatedGrant(t)
	p.failPairingAuthenticated(t)
	m := p.pairing.target
	if _, pinned := m.trust.Get(p.pairing.peer.identity.NodeUUID); pinned {
		t.Fatal("legacy authenticated rollback did not remove B's pin")
	}
	if _, member := m.memberByNodeID(p.pairing.peer.identity.NodeUUID); member {
		t.Fatal("legacy authenticated rollback did not remove B's member")
	}
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); permission.Allowed {
		t.Fatalf("legacy B was allowed before roster replay: %+v", permission)
	}
	p.replayAndRequireUnrelatedGossip(t)
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); permission.Allowed {
		t.Fatalf("cached original roster revived B's compensated legacy grandfathered grant: %+v", permission)
	}
}

func TestPairingRemoteManagementLegacyReceiptRollbackPreservesLaterLocalChoice(t *testing.T) {
	for _, tc := range []struct {
		name    string
		allowed bool
	}{{name: "explicit deny"}, {name: "explicit reallow", allowed: true}} {
		t.Run(tc.name, func(t *testing.T) {
			p := newLegacyPairingRosterReplayProbe(t)
			p.editUnrelatedGrant(t)
			m := p.pairing.target
			m.rosterMu.Lock()
			err := m.setRemoteManagementLocked(p.pairing.peer.identity.NodeUUID, tc.allowed, "2")
			m.rosterMu.Unlock()
			if err != nil {
				t.Fatal(err)
			}
			p.failPairingAuthenticated(t)
			p.replayAndRequireUnrelatedGossip(t)
			if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); permission.Allowed != tc.allowed || permission.Revision != "3" {
				t.Fatalf("legacy compensation changed a later explicit local choice: %+v, want allowed=%t revision=3", permission, tc.allowed)
			}
		})
	}
}

func TestPairingRemoteManagementLegacyReceiptJournalFailureRecoversOriginalReceipt(t *testing.T) {
	p := newLegacyPairingRosterReplayProbe(t)
	p.editUnrelatedGrant(t)
	m := p.pairing.target
	journal := filepath.Join(m.clusterDir, pairingReceiptsFile)
	before, err := os.ReadFile(journal)
	if err != nil {
		t.Fatal(err)
	}
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	journalFault := errors.New("legacy abort journal unavailable through restart")
	clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
		if target == journal {
			return journalFault
		}
		return os.Rename(tmp, target)
	}}
	p.signalPairingFailureAuthenticated(t)
	assertGuarded := func() {
		t.Helper()
		owner, err := m.pairingGuardOwner()
		if err != nil || owner != p.pairing.invite.InviteID {
			t.Fatalf("legacy journal failure lost abort correlation: owner=%q error=%v", owner, err)
		}
		if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); permission.Allowed {
			t.Fatalf("legacy journal failure published authority: %+v", permission)
		}
		after, err := os.ReadFile(journal)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(before, after) {
			t.Fatal("failed abort write changed the original legacy receipt")
		}
	}
	assertGuarded()
	if restarted, err := NewManager(m.codec, filepath.Dir(m.clusterDir), 5); restarted != nil || !errors.Is(err, journalFault) {
		t.Fatalf("startup served an unconfirmed legacy abort: manager=%v error=%v", restarted != nil, err)
	}
	assertGuarded()
	clusterFileReplacer = original
	m = testManagerAt(t, filepath.Dir(m.clusterDir), 5)
	p.pairing.target = m
	receipt, ok := m.getPairingReceipt(p.pairing.invite.InviteID)
	if !ok || receipt.Rollback || receipt.State != inviteStateFailed || receipt.AllowRemoteManagement != nil ||
		receipt.PermissionRevision != "" || receipt.PermissionMigrationID != "" || receipt.PermissionIncarnation != "" {
		t.Fatalf("original legacy receipt did not recover to an abort without consent: receipt=%+v found=%t", receipt, ok)
	}
	if _, pinned := m.trust.Get(p.pairing.peer.identity.NodeUUID); pinned {
		t.Fatal("restart retained the aborted legacy pin")
	}
	if _, member := m.memberByNodeID(p.pairing.peer.identity.NodeUUID); member {
		t.Fatal("restart retained the aborted legacy member")
	}
	if _, err := os.Stat(m.remoteManagementInitializingPath()); !os.IsNotExist(err) {
		t.Fatalf("completed legacy abort retained a guard: %v", err)
	}
	p.replayAndRequireUnrelatedGossip(t)
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); permission.Allowed {
		t.Fatalf("cached roster revived the legacy abort recovered from its original receipt: %+v", permission)
	}
}

func TestPairingRemoteManagementLegacyReceiptGuardFailurePreservesComposition(t *testing.T) {
	test := func(name string, landed bool) {
		t.Run(name, func(t *testing.T) {
			p := newLegacyPairingRosterReplayProbe(t)
			p.editUnrelatedGrant(t)
			m := p.pairing.target
			journal := filepath.Join(m.clusterDir, pairingReceiptsFile)
			before, err := os.ReadFile(journal)
			if err != nil {
				t.Fatal(err)
			}
			original := clusterFileReplacer
			t.Cleanup(func() { clusterFileReplacer = original })
			guardFault := errors.New("legacy abort guard unavailable")
			attempts := 0
			clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
				if target == m.remoteManagementInitializingPath() {
					attempts++
					if !landed {
						return guardFault
					}
					if err := os.Rename(tmp, target); err != nil {
						return err
					}
					return durablefile.ErrReplaced
				}
				return os.Rename(tmp, target)
			}}
			p.signalPairingFailureAuthenticated(t)
			wantAttempts := 1
			if landed {
				wantAttempts = 2
			}
			if attempts != wantAttempts || !p.pairing.session.rollbackPending {
				t.Fatalf("guard fault attempts=%d pending=%t, want attempts=%d and pending abort", attempts, p.pairing.session.rollbackPending, wantAttempts)
			}
			if _, pinned := m.trust.Get(p.pairing.peer.identity.NodeUUID); !pinned {
				t.Fatal("unconfirmed guard changed the live legacy pin")
			}
			if _, member := m.memberByNodeID(p.pairing.peer.identity.NodeUUID); !member {
				t.Fatal("unconfirmed guard changed legacy composition")
			}
			after, err := os.ReadFile(journal)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(before, after) {
				t.Fatal("unconfirmed guard published an abort receipt")
			}
			permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID)
			if permission.Allowed != !landed {
				t.Fatalf("guard replacement outcome did not control publication: landed=%t permission=%+v", landed, permission)
			}
			if landed {
				owner, err := m.pairingGuardOwner()
				if err != nil || owner != p.pairing.invite.InviteID {
					t.Fatalf("uncertain legacy guard lost correlation: owner=%q error=%v", owner, err)
				}
			}
			clusterFileReplacer = original
			if landed {
				m = testManagerAt(t, filepath.Dir(m.clusterDir), 5)
				p.pairing.target = m
			} else if err := p.retryPairingRollback(); err != nil {
				t.Fatalf("retry legacy guard after restoring storage: %v", err)
			}
			receipt, ok := m.getPairingReceipt(p.pairing.invite.InviteID)
			if !ok || receipt.Rollback || receipt.State != inviteStateFailed || receipt.AllowRemoteManagement != nil {
				t.Fatalf("retried legacy guard did not complete an abort: receipt=%+v found=%t", receipt, ok)
			}
			if _, err := os.Stat(m.remoteManagementInitializingPath()); !os.IsNotExist(err) {
				t.Fatalf("retried legacy guard was not released: %v", err)
			}
			p.replayAndRequireUnrelatedGossip(t)
			if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); permission.Allowed {
				t.Fatalf("retried legacy abort allowed stale membership: %+v", permission)
			}
		})
	}
	test("guard never replaced", false)
	test("guard replaced without durability confirmation", true)
}

func TestPairingRemoteManagementLegacyReceiptAbsentGrantRequiresDurableConfirmation(t *testing.T) {
	test := func(name string, restart bool) {
		t.Run(name, func(t *testing.T) {
			p := newLegacyPairingRosterReplayProbe(t)
			p.editUnrelatedGrant(t)
			m := p.pairing.target
			grantPath, _ := m.remoteManagementPaths()
			original := clusterFileReplacer
			t.Cleanup(func() { clusterFileReplacer = original })
			assertGuarded := func() {
				t.Helper()
				owner, err := m.pairingGuardOwner()
				if err != nil || owner != p.pairing.invite.InviteID {
					t.Fatalf("legacy cleanup lost its guard: owner=%q error=%v", owner, err)
				}
				if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); permission.Allowed {
					t.Fatalf("legacy cleanup exposed authority: %+v", permission)
				}
			}
			deletionReplacements := 0
			clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
				if target != grantPath {
					return os.Rename(tmp, target)
				}
				data, err := os.ReadFile(tmp)
				if err != nil {
					return err
				}
				doc, err := clustertrust.DecodeRemoteManagementDocument(data)
				if err != nil {
					return err
				}
				for _, grant := range doc.Grants {
					if grant.PeerUUID == p.pairing.peer.identity.NodeUUID {
						t.Fatal("legacy cleanup fault must land the owned grant deletion")
					}
				}
				if err := os.Rename(tmp, target); err != nil {
					return err
				}
				deletionReplacements++
				return durablefile.ErrReplaced
			}}
			p.signalPairingFailureAuthenticated(t)
			if deletionReplacements != 2 || !p.pairing.session.rollbackPending {
				t.Fatalf("uncertain legacy deletion replacements=%d pending=%t, want 2 and pending", deletionReplacements, p.pairing.session.rollbackPending)
			}
			assertGuarded()
			if _, pinned := m.trust.Get(p.pairing.peer.identity.NodeUUID); pinned {
				t.Fatal("fixture needs B's pin absent before cleanup confirmation")
			}
			if _, member := m.memberByNodeID(p.pairing.peer.identity.NodeUUID); member {
				t.Fatal("fixture needs B's member absent before cleanup confirmation")
			}
			deleted, err := os.ReadFile(grantPath)
			if err != nil {
				t.Fatal(err)
			}
			confirmationFault := errors.New("legacy already-absent grant confirmation fault")
			confirmationAttempts := 0
			clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
				if target == grantPath {
					confirmationAttempts++
					return confirmationFault
				}
				return os.Rename(tmp, target)
			}}
			if restart {
				if reopened, err := NewManager(m.codec, filepath.Dir(m.clusterDir), 5); reopened != nil || !errors.Is(err, confirmationFault) {
					t.Fatalf("startup accepted an unconfirmed absent legacy grant: manager=%v error=%v", reopened != nil, err)
				}
			} else if err := p.retryPairingRollback(); !errors.Is(err, confirmationFault) {
				t.Fatalf("legacy absent-entry retry error=%v, want confirmation fault", err)
			}
			if confirmationAttempts != 1 {
				t.Fatalf("legacy absent-entry confirmation attempts=%d, want 1", confirmationAttempts)
			}
			assertGuarded()
			after, err := os.ReadFile(grantPath)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(deleted, after) {
				t.Fatal("failed legacy confirmation changed the landed deletion")
			}
			clusterFileReplacer = original
			if restart {
				m = testManagerAt(t, filepath.Dir(m.clusterDir), 5)
				p.pairing.target = m
			} else if err := p.retryPairingRollback(); err != nil {
				t.Fatalf("confirm legacy cleanup after restoring storage: %v", err)
			}
			receipt, ok := m.getPairingReceipt(p.pairing.invite.InviteID)
			if !ok || receipt.Rollback || receipt.State != inviteStateFailed || receipt.AllowRemoteManagement != nil {
				t.Fatalf("legacy cleanup did not finish without inventing consent: receipt=%+v found=%t", receipt, ok)
			}
			if _, err := os.Stat(m.remoteManagementInitializingPath()); !os.IsNotExist(err) {
				t.Fatalf("confirmed legacy cleanup retained a guard: %v", err)
			}
			p.replayAndRequireUnrelatedGossip(t)
			if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); permission.Allowed {
				t.Fatalf("cached roster revived a confirmed legacy cleanup: %+v", permission)
			}
		})
	}
	test("live retry without pin or member", false)
	test("restart without pin or member", true)
}

func TestPairingRemoteManagementLegacyReceiptPendingMigrationCompletesAfterAbort(t *testing.T) {
	test := func(name string, draft bool) {
		t.Run(name, func(t *testing.T) {
			m := remoteManagementUpgradeFixture(t)
			peer := remoteManagementAddPeer(t, m)
			relay := remoteManagementAddPeer(t, m)
			cid, epoch := m.currentAdmission()
			info := peer.localPairingInfo("", 1)
			info.ClusterID = cid
			raw, err := json.Marshal(info)
			if err != nil {
				t.Fatal(err)
			}
			info, cert, err := parsePairingInfo(raw)
			if err != nil {
				t.Fatal(err)
			}
			if err := m.trust.AddEndorsements(peer.identity.NodeUUID,
				[]Endorsement{m.endorsePeer(peer.identity.NodeUUID, peer.identity.CertFingerprint, info.AdmissionEpoch)}); err != nil {
				t.Fatal(err)
			}
			receipt := pairingReceipt{InviteID: "legacy-pending-migration-abort", Role: roleInviter,
				ClusterID: cid, LocalEpoch: epoch, PeerInfo: raw, State: inviteStatePaired,
				Rollback: true, ExpiresAt: time.Now().Add(time.Hour).UnixMilli()}
			if err := m.savePairingReceipt(receipt); err != nil {
				t.Fatal(err)
			}
			if err := m.captureRemoteManagementMigrationLocked(); err != nil {
				t.Fatal(err)
			}
			grantPath, markerPath := m.remoteManagementPaths()
			pendingBytes, err := os.ReadFile(markerPath)
			if err != nil {
				t.Fatal(err)
			}
			pending, err := clustertrust.DecodeRemoteManagementMigration(pendingBytes)
			if err != nil || pending.State != "pending" {
				t.Fatalf("first migration is not genuinely pending: marker=%+v error=%v", pending, err)
			}
			if draft {
				original := clusterFileReplacer
				t.Cleanup(func() { clusterFileReplacer = original })
				completionFault := errors.New("migration stopped after draft, before marker completion")
				clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
					if target == markerPath {
						data, err := os.ReadFile(tmp)
						if err != nil {
							return err
						}
						marker, err := clustertrust.DecodeRemoteManagementMigration(data)
						if err != nil {
							return err
						}
						if marker.State == "complete" {
							return completionFault
						}
					}
					return os.Rename(tmp, target)
				}}
				m.rosterMu.Lock()
				err = m.finalizeRemoteManagementMigrationLocked()
				m.rosterMu.Unlock()
				if !errors.Is(err, completionFault) {
					t.Fatalf("migration did not reach the real draft boundary: %v", err)
				}
				clusterFileReplacer = original
				draftBytes, err := os.ReadFile(grantPath)
				if err != nil {
					t.Fatal(err)
				}
				doc, err := clustertrust.DecodeRemoteManagementDocument(draftBytes)
				if err != nil || doc.Revision != "1" {
					t.Fatalf("migration draft=%+v error=%v", doc, err)
				}
				found := false
				for _, grant := range doc.Grants {
					if grant.PeerUUID == peer.identity.NodeUUID && grant.Allowed && grant.DecisionRevision == "1" {
						found = true
					}
				}
				if !found {
					t.Fatal("real unpublished migration draft lacks B's original allow")
				}
			} else if _, err := os.Stat(grantPath); !os.IsNotExist(err) {
				t.Fatalf("pre-draft migration already has a permission document: %v", err)
			}
			cached := m.buildLocalRoster()
			gossipPeer := testManagerAt(t, t.TempDir(), 4)
			gossipEpoch := activateTestCluster(t, gossipPeer, cid)
			cached.Members = append(cached.Members, permissionProbeRosterEntry(m, gossipPeer, gossipEpoch))
			wire, err := json.Marshal(cached)
			if err != nil {
				t.Fatal(err)
			}
			var frozen Roster
			if err := json.Unmarshal(wire, &frozen); err != nil {
				t.Fatal(err)
			}
			if permission := clustertrust.Open(m.clusterDir).RemoteManagement(peer.identity.NodeUUID); permission.Allowed {
				t.Fatalf("pending migration published B before abort recovery: %+v", permission)
			}
			m = testManagerAt(t, filepath.Dir(m.clusterDir), 5)
			completed, ok := m.getPairingReceipt(receipt.InviteID)
			if !ok || completed.Rollback || completed.State != inviteStateFailed || completed.AllowRemoteManagement != nil {
				t.Fatalf("pending migration blocked exact legacy abort recovery: receipt=%+v found=%t", completed, ok)
			}
			if _, pinned := m.trust.Get(peer.identity.NodeUUID); pinned {
				t.Fatal("first-upgrade abort retained the old pin")
			}
			if _, member := m.memberByNodeID(peer.identity.NodeUUID); member {
				t.Fatal("first-upgrade abort retained the old member")
			}
			doc, marker, err := m.readRemoteManagementLocked()
			if err != nil || marker.State != "complete" || marker.ID != pending.ID || doc.Revision != "1" {
				t.Fatalf("first-upgrade abort did not finalize its original migration: doc=%+v marker=%+v error=%v", doc, marker, err)
			}
			beforeSnapshot, err := json.Marshal(pending.Snapshot)
			if err != nil {
				t.Fatal(err)
			}
			afterSnapshot, err := json.Marshal(marker.Snapshot)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(beforeSnapshot, afterSnapshot) {
				t.Fatal("abort recovery resnapshotted the original migration authority")
			}
			if permission := clustertrust.Open(m.clusterDir).RemoteManagement(relay.identity.NodeUUID); !permission.Allowed || permission.Revision != "1" {
				t.Fatalf("first-upgrade abort changed unrelated grandfathered C: %+v", permission)
			}
			m.rosterMu.Lock()
			err = m.setRemoteManagementLocked(relay.identity.NodeUUID, false, "1")
			m.rosterMu.Unlock()
			if err != nil {
				t.Fatal(err)
			}
			p := pairingRosterReplayProbe{pairing: localPermissionPairing{target: m, peer: peer, info: info, cert: cert},
				relay: relay, gossipPeer: gossipPeer, cached: &frozen}
			p.replayAndRequireUnrelatedGossip(t)
			if permission := clustertrust.Open(m.clusterDir).RemoteManagement(peer.identity.NodeUUID); permission.Allowed {
				t.Fatalf("valid roster revived an original migration draft after abort: %+v", permission)
			}
		})
	}
	test("before migration draft exists", false)
	test("after real migration draft lands", true)
}

func TestPairingRemoteManagementLegacyReceiptRollbackPreservesNewerPeerAdmission(t *testing.T) {
	p := newLegacyPairingRosterReplayProbe(t)
	p.editUnrelatedGrant(t)
	m := p.pairing.target
	newer := *p.pairing.info
	newer.AdmissionEpoch++
	now := time.Now().UnixMilli()
	m.recordMember(&newer, stateMember, &now)
	if err := m.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	if err := m.pinPeer(&newer, p.pairing.cert, nil); err != nil {
		t.Fatal(err)
	}
	m.rosterMu.Lock()
	err := m.setRemoteManagementLocked(newer.NodeUUID, true, "2")
	m.rosterMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	p.failPairingAuthenticated(t)
	p.replayAndRequireUnrelatedGossip(t)
	pin, pinned := m.trust.Get(newer.NodeUUID)
	member, present := m.memberByNodeID(newer.NodeUUID)
	if !pinned || !present || pin.AdmissionEpoch != newer.AdmissionEpoch || member.AdmissionEpoch != newer.AdmissionEpoch {
		t.Fatal("legacy abort changed the newer peer admission")
	}
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(newer.NodeUUID); !permission.Allowed || permission.Revision != "3" {
		t.Fatalf("legacy abort changed the newer peer's explicit allow: %+v", permission)
	}
}

func TestPairingRemoteManagementLegacyReceiptRollbackPreservesNewerLocalAdmission(t *testing.T) {
	p := newLegacyPairingRosterReplayProbe(t)
	p.editUnrelatedGrant(t)
	m := p.pairing.target
	if err := m.clearAdmission(); err != nil {
		t.Fatal(err)
	}
	newEpoch := activateTestCluster(t, m, p.pairing.invite.ClusterID)
	if newEpoch <= p.pairing.session.localAdmissionEpoch {
		t.Fatal("fixture did not advance the local admission")
	}
	m.addSelfMemberForAdmission(p.pairing.invite.ClusterID, newEpoch)
	if err := m.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	m.rosterMu.Lock()
	err := m.setRemoteManagementLocked(p.pairing.peer.identity.NodeUUID, true, "2")
	m.rosterMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	p.failPairingAuthenticated(t)
	if cid, epoch := m.currentAdmission(); cid != p.pairing.invite.ClusterID || epoch != newEpoch {
		t.Fatalf("legacy abort changed the newer local admission: cluster=%q epoch=%d", cid, epoch)
	}
	if _, pinned := m.trust.Get(p.pairing.peer.identity.NodeUUID); !pinned {
		t.Fatal("legacy abort removed the newer local admission's peer pin")
	}
	if _, member := m.memberByNodeID(p.pairing.peer.identity.NodeUUID); !member {
		t.Fatal("legacy abort removed the newer local admission's peer member")
	}
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); !permission.Allowed || permission.Revision != "3" {
		t.Fatalf("legacy abort changed the newer local admission's explicit allow: %+v", permission)
	}
}

type pairingRosterReplayProbe struct {
	pairing    localPermissionPairing
	relay      *Manager
	gossipPeer *Manager
	cached     *Roster
}

func permissionProbeRosterEntry(target, peer *Manager, epoch uint64) RosterEntry {
	return RosterEntry{
		NodeUUID: peer.identity.NodeUUID, NodeID: peer.identity.NodeID, Name: peer.identity.Name,
		AdmissionEpoch: epoch, CertPem: string(peer.identity.CertPEM), CertFingerprint: peer.identity.CertFingerprint,
		Endorsements: []Endorsement{target.endorsePeer(peer.identity.NodeUUID, peer.identity.CertFingerprint, epoch)},
	}
}

// The cached roster carries the target's actual signed pairing endorsement for
// B and is replayed as trusted relay C. Accepting independent valid gossip D
// after rollback proves rejecting every roster cannot satisfy these cases.
func newPairingRosterReplayProbe(t *testing.T) pairingRosterReplayProbe {
	t.Helper()
	f := newLocalPermissionPairing(t, roleInviter, true)
	if err := f.target.finalizePairing(f.invite.InviteID, f.session, f.info, f.cert); err != nil {
		t.Fatalf("publish B pairing: %v", err)
	}
	relay := testManagerAt(t, t.TempDir(), 3)
	relayEpoch := activateTestCluster(t, relay, f.invite.ClusterID)
	relayRoster := &Roster{ClusterID: f.invite.ClusterID,
		Members: []RosterEntry{permissionProbeRosterEntry(f.target, relay, relayEpoch)}}
	if !f.target.mergeRoster(relayRoster, f.peer.identity.NodeUUID) {
		t.Fatal("valid endorsed relay C was not admitted through roster merge")
	}
	if _, pinned := f.target.trust.Get(relay.identity.NodeUUID); !pinned {
		t.Fatal("roster merge did not establish relay C's pin")
	}
	cached := f.target.buildLocalRoster()
	validB := false
	for i := range cached.Members {
		entry := &cached.Members[i]
		if entry.NodeUUID == f.peer.identity.NodeUUID && entry.AdmissionEpoch == f.info.AdmissionEpoch && f.target.entryEndorsed(entry, cached.ClusterID) {
			validB = true
		}
	}
	if !validB {
		t.Fatal("cached roster lacks B's valid original pairing endorsement")
	}
	gossipPeer := testManagerAt(t, t.TempDir(), 4)
	gossipEpoch := activateTestCluster(t, gossipPeer, f.invite.ClusterID)
	cached.Members = append(cached.Members, permissionProbeRosterEntry(f.target, gossipPeer, gossipEpoch))
	// Freeze wire bytes so later pin/endorsement mutations cannot update the
	// stale snapshot this relay would have retained before B's rollback.
	wire, err := json.Marshal(cached)
	if err != nil {
		t.Fatal(err)
	}
	var frozen Roster
	if err := json.Unmarshal(wire, &frozen); err != nil {
		t.Fatal(err)
	}
	return pairingRosterReplayProbe{pairing: f, relay: relay, gossipPeer: gossipPeer, cached: &frozen}
}

func (p pairingRosterReplayProbe) editUnrelatedGrant(t *testing.T) {
	t.Helper()
	m := p.pairing.target
	before := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID)
	if !before.Allowed {
		t.Fatalf("B lacks the pairing grant before C's local edit: %+v", before)
	}
	m.rosterMu.Lock()
	err := m.setRemoteManagementLocked(p.relay.identity.NodeUUID, false, before.Revision)
	m.rosterMu.Unlock()
	if err != nil {
		t.Fatalf("edit C's unrelated grant: %v", err)
	}
	after := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID)
	if !after.Allowed || after.Incarnation != before.Incarnation || after.Revision == before.Revision {
		t.Fatalf("C's edit did not advance only the document revision for B: before=%+v after=%+v", before, after)
	}
}

func (p pairingRosterReplayProbe) signalPairingFailureAuthenticated(t *testing.T) {
	t.Helper()
	f := p.pairing
	body, err := json.Marshal(pairingEnvelope{InviteID: f.invite.InviteID, Phase: "fail"})
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest(http.MethodPost, pairingPath, bytes.NewReader(body))
	r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{f.cert}}
	rr := httptest.NewRecorder()
	f.target.handlePairing(rr, r)
	if rr.Code != http.StatusOK {
		t.Fatalf("authenticated B failure status=%d, want 200", rr.Code)
	}
}

func (p pairingRosterReplayProbe) failPairingAuthenticated(t *testing.T) {
	t.Helper()
	p.signalPairingFailureAuthenticated(t)
	f := p.pairing
	receipt, ok := f.target.getPairingReceipt(f.invite.InviteID)
	if !ok || receipt.Rollback || receipt.State != inviteStateFailed {
		t.Fatalf("authenticated failure did not finish compensation: receipt=%+v found=%t", receipt, ok)
	}
}

func (p pairingRosterReplayProbe) retryPairingRollback() error {
	m, sess := p.pairing.target, p.pairing.session
	m.inviteMu.Lock()
	defer m.inviteMu.Unlock()
	sess.mu.Lock()
	defer sess.mu.Unlock()
	inv, ok := m.getInvite(sess.inviteID)
	if !ok {
		return fmt.Errorf("compensation invite missing")
	}
	return m.rollbackPairingSessionLocked(sess, inv)
}

func (p pairingRosterReplayProbe) replayAndRequireUnrelatedGossip(t *testing.T) {
	t.Helper()
	m := p.pairing.target
	if _, pinned := m.trust.Get(p.gossipPeer.identity.NodeUUID); pinned {
		t.Fatal("positive gossip control D was already pinned before replay")
	}
	if !m.mergeRoster(p.cached, p.relay.identity.NodeUUID) {
		t.Fatal("cached roster rejected independent valid gossip entry D")
	}
	if _, pinned := m.trust.Get(p.gossipPeer.identity.NodeUUID); !pinned {
		t.Fatal("cached roster did not admit independent valid gossip entry D")
	}
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.gossipPeer.identity.NodeUUID); permission.Allowed || permission.State != "unreviewed" {
		t.Fatalf("gossip D must be admitted without inventing a management grant: %+v", permission)
	}
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.relay.identity.NodeUUID); permission.Allowed || permission.State != "denied" {
		t.Fatalf("roster replay changed C's unrelated explicit deny: %+v", permission)
	}
}

func TestPairingRemoteManagementCachedRosterCannotReviveRolledBackGrant(t *testing.T) {
	p := newPairingRosterReplayProbe(t)
	p.editUnrelatedGrant(t)
	p.failPairingAuthenticated(t)
	if _, pinned := p.pairing.target.trust.Get(p.pairing.peer.identity.NodeUUID); pinned {
		t.Fatal("authenticated rollback did not remove B's transaction pin")
	}
	if _, member := p.pairing.target.memberByNodeID(p.pairing.peer.identity.NodeUUID); member {
		t.Fatal("authenticated rollback did not remove B's transaction member")
	}
	if permission := clustertrust.Open(p.pairing.target.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); permission.Allowed {
		t.Fatalf("B was allowed immediately after rollback: %+v", permission)
	}
	p.replayAndRequireUnrelatedGossip(t)
	if permission := clustertrust.Open(p.pairing.target.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); permission.Allowed {
		t.Fatalf("cached pre-rollback roster revived B's old transaction grant: %+v", permission)
	}
}

func TestPairingRemoteManagementCachedRosterPreservesLaterSamePeerDeny(t *testing.T) {
	p := newPairingRosterReplayProbe(t)
	p.editUnrelatedGrant(t)
	m := p.pairing.target
	before := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID)
	m.rosterMu.Lock()
	err := m.setRemoteManagementLocked(p.pairing.peer.identity.NodeUUID, false, before.Revision)
	m.rosterMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	p.failPairingAuthenticated(t)
	p.replayAndRequireUnrelatedGossip(t)
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); permission.Allowed {
		t.Fatalf("cached roster overwrote B's later explicit deny: %+v", permission)
	}
	m.rosterMu.Lock()
	doc, _, err := m.readRemoteManagementLocked()
	m.rosterMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	denyPreserved := false
	for _, grant := range doc.Grants {
		if grant.PeerUUID == p.pairing.peer.identity.NodeUUID && grant.Incarnation == before.Incarnation && !grant.Allowed {
			denyPreserved = true
		}
	}
	if !denyPreserved {
		t.Fatalf("cleanup or gossip discarded the later same-peer deny: %+v", doc)
	}
}

func TestPairingRemoteManagementCachedRosterPreservesLaterSamePeerReallow(t *testing.T) {
	p := newPairingRosterReplayProbe(t)
	p.editUnrelatedGrant(t)
	m := p.pairing.target
	before := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID)
	m.rosterMu.Lock()
	err := m.setRemoteManagementLocked(p.pairing.peer.identity.NodeUUID, true, before.Revision)
	doc, _, readErr := m.readRemoteManagementLocked()
	m.rosterMu.Unlock()
	if err != nil || readErr != nil {
		t.Fatalf("explicit same-value allow failed: update=%v read=%v", err, readErr)
	}
	reallowRecorded := false
	for _, grant := range doc.Grants {
		if grant.PeerUUID == p.pairing.peer.identity.NodeUUID && grant.Incarnation == before.Incarnation && grant.Allowed && grant.DecisionRevision == "4" {
			reallowRecorded = true
		}
	}
	if doc.Revision != "4" || !reallowRecorded {
		t.Fatalf("same-value local allow did not establish a new decision: %+v", doc)
	}
	p.failPairingAuthenticated(t)
	p.replayAndRequireUnrelatedGossip(t)
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); !permission.Allowed || permission.Incarnation != before.Incarnation || permission.Revision != "4" {
		t.Fatalf("old transaction removed the later same-value local allow: %+v", permission)
	}
}

func TestPairingRemoteManagementCachedRosterPreservesNewerAdmissionGrant(t *testing.T) {
	p := newPairingRosterReplayProbe(t)
	p.editUnrelatedGrant(t)
	m := p.pairing.target
	oldPermission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID)
	newEpoch := p.pairing.info.AdmissionEpoch + 1
	newRoster := &Roster{ClusterID: p.pairing.invite.ClusterID,
		Members: []RosterEntry{permissionProbeRosterEntry(m, p.pairing.peer, newEpoch)}}
	if !m.mergeRoster(newRoster, p.relay.identity.NodeUUID) {
		t.Fatal("valid newer B admission was not accepted through roster merge")
	}
	m.rosterMu.Lock()
	err := m.setRemoteManagementLocked(p.pairing.peer.identity.NodeUUID, true, oldPermission.Revision)
	m.rosterMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	newPermission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID)
	if !newPermission.Allowed || newPermission.Incarnation == oldPermission.Incarnation {
		t.Fatalf("new admission did not acquire its distinct explicit grant: old=%+v new=%+v", oldPermission, newPermission)
	}
	p.failPairingAuthenticated(t)
	p.replayAndRequireUnrelatedGossip(t)
	pin, pinned := m.trust.Get(p.pairing.peer.identity.NodeUUID)
	member, present := m.memberByNodeID(p.pairing.peer.identity.NodeUUID)
	if !pinned || !present || pin.AdmissionEpoch != newEpoch || member.AdmissionEpoch != newEpoch {
		t.Fatal("old rollback or cached roster changed the newer B admission")
	}
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(p.pairing.peer.identity.NodeUUID); !permission.Allowed || permission.Incarnation != newPermission.Incarnation {
		t.Fatalf("old transaction changed the newer admission's explicit grant: %+v", permission)
	}
}

func TestPairingRemoteManagementCleanupAlreadyAbsentGrantRetainsGuardUntilConfirmed(t *testing.T) {
	test := func(name string, restart bool) {
		t.Run(name, func(t *testing.T) {
			p := newPairingRosterReplayProbe(t)
			p.editUnrelatedGrant(t)
			f := p.pairing
			m := f.target
			grantPath, _ := m.remoteManagementPaths()
			original := clusterFileReplacer
			t.Cleanup(func() { clusterFileReplacer = original })

			assertGuardedAbort := func() {
				t.Helper()
				receipt, ok := m.getPairingReceipt(f.invite.InviteID)
				if !ok || !receipt.Rollback || receipt.State == inviteStateFailed || !f.session.rollbackPending {
					t.Fatalf("unfinished compensation lost its abort intent: receipt=%+v found=%t pending=%t", receipt, ok, f.session.rollbackPending)
				}
				guarded, err := m.remoteManagementGuard(receipt.PermissionMigrationID)
				if err != nil || !guarded {
					t.Fatalf("unfinished compensation lost its publication guard: guarded=%t error=%v", guarded, err)
				}
				owner, err := m.pairingGuardOwner()
				if err != nil || owner != f.invite.InviteID {
					t.Fatalf("unfinished compensation guard owner=%q error=%v", owner, err)
				}
				if permission := clustertrust.Open(m.clusterDir).RemoteManagement(f.peer.identity.NodeUUID); permission.Allowed {
					t.Fatalf("fresh reader allowed unfinished compensation: %+v", permission)
				}
			}
			retry := func() error {
				m.inviteMu.Lock()
				defer m.inviteMu.Unlock()
				f.session.mu.Lock()
				defer f.session.mu.Unlock()
				inv, ok := m.getInvite(f.invite.InviteID)
				if !ok {
					return fmt.Errorf("compensation invite missing")
				}
				return m.rollbackPairingSessionLocked(f.session, inv)
			}

			deletionReplacements := 0
			clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
				if target != grantPath {
					return os.Rename(tmp, target)
				}
				data, err := os.ReadFile(tmp)
				if err != nil {
					return err
				}
				doc, err := clustertrust.DecodeRemoteManagementDocument(data)
				if err != nil {
					return err
				}
				for _, grant := range doc.Grants {
					if grant.PeerUUID == f.peer.identity.NodeUUID {
						t.Fatal("fault must land deletion of B's owned grant")
					}
				}
				if err := os.Rename(tmp, target); err != nil {
					return err
				}
				deletionReplacements++
				return durablefile.ErrReplaced
			}}
			body, err := json.Marshal(pairingEnvelope{InviteID: f.invite.InviteID, Phase: "fail"})
			if err != nil {
				t.Fatal(err)
			}
			r := httptest.NewRequest(http.MethodPost, pairingPath, bytes.NewReader(body))
			r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{f.cert}}
			rr := httptest.NewRecorder()
			m.handlePairing(rr, r)
			if rr.Code != http.StatusOK || deletionReplacements != 2 {
				t.Fatalf("authenticated failure status=%d deletion replacements=%d, want 200 and 2 uncertain replacements", rr.Code, deletionReplacements)
			}
			assertGuardedAbort()
			deletedBytes, err := os.ReadFile(grantPath)
			if err != nil {
				t.Fatal(err)
			}
			deletedDoc, err := clustertrust.DecodeRemoteManagementDocument(deletedBytes)
			if err != nil {
				t.Fatal(err)
			}
			for _, grant := range deletedDoc.Grants {
				if grant.PeerUUID == f.peer.identity.NodeUUID {
					t.Fatal("B's grant was not absent before the confirming retry")
				}
			}

			confirmationFault := errors.New("already-absent grant confirmation fault")
			confirmationAttempts := 0
			clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
				if target == grantPath {
					confirmationAttempts++
					return confirmationFault
				}
				return os.Rename(tmp, target)
			}}
			if err := retry(); !errors.Is(err, confirmationFault) {
				t.Fatalf("already-absent grant retry error=%v, want confirming-write failure", err)
			}
			if confirmationAttempts != 1 {
				t.Fatalf("confirming grant rewrite attempts=%d, want 1", confirmationAttempts)
			}
			assertGuardedAbort()
			unchangedBytes, err := os.ReadFile(grantPath)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(unchangedBytes, deletedBytes) {
				t.Fatal("pre-replacement confirmation failure changed the landed deletion")
			}

			clusterFileReplacer = original
			if restart {
				m = testManagerAt(t, filepath.Dir(m.clusterDir), 5)
				p.pairing.target = m
			} else if err := retry(); err != nil {
				t.Fatalf("confirming live compensation: %v", err)
			}
			receipt, ok := m.getPairingReceipt(f.invite.InviteID)
			if !ok || receipt.Rollback || receipt.State != inviteStateFailed {
				t.Fatalf("restored storage did not finish compensation: receipt=%+v found=%t", receipt, ok)
			}
			if _, err := os.Stat(m.remoteManagementInitializingPath()); !os.IsNotExist(err) {
				t.Fatalf("confirmed compensation did not release its guard: %v", err)
			}
			p.replayAndRequireUnrelatedGossip(t)
			if permission := clustertrust.Open(m.clusterDir).RemoteManagement(f.peer.identity.NodeUUID); permission.Allowed {
				t.Fatalf("cached roster revived a deletion after confirmed compensation: %+v", permission)
			}
		})
	}
	test("live retry", false)
	test("restart recovery", true)
}
