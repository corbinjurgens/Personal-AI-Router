// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
)

func assertJoinerCompensationPending(t *testing.T, f outboundPairingFixture) {
	t.Helper()
	inv, ok := f.joiner.getInvite(f.session.inviteID)
	if !ok || inv.State != inviteStatePending || inv.Reason != reasonRollbackPending || !f.joinSession.rollbackPending {
		t.Fatalf("unfinished join compensation = %+v, want pending/rollback-pending", inv)
	}
	r := httptest.NewRequest(http.MethodPost, rosterPath, nil)
	r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{identityCertificate(t, f.inviter)}}
	if _, allowed := f.joiner.verifyClientPin(r); allowed {
		t.Fatal("unfinished join compensation granted general peer authority")
	}
	rr := f.confirm(t, r.TLS.PeerCertificates[0], f.joinSession.localAdmissionEpoch)
	var outcome pairingEnvelope
	if err := json.Unmarshal(rr.Body.Bytes(), &outcome); err != nil {
		t.Fatal(err)
	}
	if rr.Code != http.StatusOK || outcome.State != inviteStatePending || outcome.Reason != reasonRollbackPending {
		t.Fatalf("unfinished compensation confirmation = HTTP %d %+v", rr.Code, outcome)
	}
	if self, exists := f.joiner.memberByNodeID(f.joiner.identity.NodeUUID); !exists || self.AdmissionEpoch != f.joinSession.localAdmissionEpoch {
		t.Fatal("unfinished compensation lost its exact provisional admission marker")
	}
}

func TestJoinerCompensationPartialWritesRetryAndRecover(t *testing.T) {
	test := func(name string, pinFault, memberFault, restart bool) {
		t.Run(name, func(t *testing.T) {
			f := newOutboundPairingFixture(t)
			final := f.authenticatedSuccess(t)
			var restores []func()
			previous := f.joiner.trust.onChange
			faulted := false
			f.joiner.trust.SetOnChange(func() {
				if previous != nil {
					previous()
				}
				if faulted {
					return
				}
				faulted = true
				if _, exists := f.joiner.trust.Get(f.inviter.identity.NodeUUID); !exists {
					t.Fatal("fault must follow the durable pin commit")
				}
				restores = append(restores, conflictPairingPath(t, f.joiner.admissionPath()))
				if pinFault {
					restores = append(restores, conflictPairingPath(t, f.joiner.trust.pinPath(f.inviter.identity.NodeUUID)))
				}
				if memberFault {
					restores = append(restores, conflictPairingPath(t, f.joiner.membersPath()))
				}
			})
			f.joiner.handleOutboundCompletion(httptest.NewRecorder(), &pairingEnvelope{InviteID: f.session.inviteID}, final)
			if !faulted {
				t.Fatal("fault did not reach admission activation")
			}
			assertJoinerCompensationPending(t, f)
			inv, _ := f.joiner.getInvite(f.session.inviteID)
			if err := f.joiner.commitJoinerPairing(f.joinSession, inv); err == nil {
				t.Fatal("unfinished provisional admission allowed another commit")
			}
			receipt, ok := f.joiner.getPairingReceipt(f.session.inviteID)
			if !ok || !receipt.Rollback || receipt.State == inviteStateFailed {
				t.Fatalf("unfinished durable join journal = %+v", receipt)
			}
			if !pinFault {
				if _, err := os.Stat(f.joiner.trust.pinPath(f.inviter.identity.NodeUUID)); !os.IsNotExist(err) {
					t.Fatalf("successful pin deletion before member fault = %v", err)
				}
			}
			// Expiry cannot discard a known incomplete durable compensation.
			f.joiner.expirePendingInvites(time.Now().Add(2 * f.joiner.effectiveInviteTTL()))
			assertJoinerCompensationPending(t, f)
			for _, restore := range restores {
				restore()
			}
			if restart {
				f.joiner = testManagerAt(t, filepath.Dir(f.joiner.clusterDir), 9)
				receipt, ok = f.joiner.getPairingReceipt(f.session.inviteID)
				if !ok || receipt.Rollback || receipt.State != inviteStateFailed {
					t.Fatalf("restart did not complete join compensation: %+v", receipt)
				}
			} else {
				f.joiner.expirePendingInvites(time.Now())
				if inv, _ := f.joiner.getInvite(f.session.inviteID); inv.State != inviteStateFailed {
					t.Fatalf("completed join compensation = %+v", inv)
				}
			}
			if _, pinned := f.joiner.trust.Get(f.inviter.identity.NodeUUID); pinned {
				t.Fatal("completed join compensation retained inviter authority")
			}
			r := httptest.NewRequest(http.MethodPost, rosterPath, nil)
			r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{identityCertificate(t, f.inviter)}}
			if _, allowed := f.joiner.verifyClientPin(r); allowed {
				t.Fatal("completed/recovered compensation accepted the old transcript as general authority")
			}
			if _, member := f.joiner.memberByNodeID(f.inviter.identity.NodeUUID); member {
				t.Fatal("completed join compensation retained inviter membership")
			}
			if _, member := f.joiner.memberByNodeID(f.joiner.identity.NodeUUID); member {
				t.Fatal("completed join compensation retained provisional self admission")
			}
			rr := f.confirm(t, identityCertificate(t, f.inviter), f.joinSession.localAdmissionEpoch)
			var outcome pairingEnvelope
			if err := json.Unmarshal(rr.Body.Bytes(), &outcome); err != nil {
				t.Fatal(err)
			}
			if rr.Code != http.StatusOK || outcome.State != inviteStateFailed {
				t.Fatalf("completed exact-invite confirmation = HTTP %d %+v", rr.Code, outcome)
			}
		})
	}
	test("pin fails after member cleanup succeeds then retries", true, false, false)
	test("member fails after pin cleanup succeeds then retries", false, true, false)
	test("pin failure recovers on restart", true, false, true)
	test("member failure recovers on restart", false, true, true)
	test("both failures recover on restart", true, true, true)
}

func TestJoinerCompensationPreservesExactAdmissionBoundaries(t *testing.T) {
	t.Run("a newer provisional local admission survives old compensation", func(t *testing.T) {
		f := newOutboundPairingFixture(t)
		f.authenticatedSuccess(t)
		inv, _ := f.joiner.getInvite(f.session.inviteID)
		receipt := f.joiner.newPairingReceipt(f.joinSession, inv, nil)
		epoch, err := f.joiner.reserveAdmissionEpoch()
		if err != nil {
			t.Fatal(err)
		}
		if epoch <= receipt.LocalEpoch {
			t.Fatal("fixture did not advance provisional admission")
		}
		pi := f.joinSession.peerPairing
		now := time.Now().UnixMilli()
		f.joiner.recordMember(pi, stateMember, &now)
		f.joiner.addSelfMemberForAdmission(receipt.ClusterID, epoch)
		if err := f.joiner.persistMembersErr(); err != nil {
			t.Fatal(err)
		}
		if err := f.joiner.pinPeer(pi, identityCertificate(t, f.inviter), nil); err != nil {
			t.Fatal(err)
		}
		if err := f.joiner.rollbackPairingPeerLocked(receipt); err != nil {
			t.Fatal(err)
		}
		self, present := f.joiner.memberByNodeID(f.joiner.identity.NodeUUID)
		if !present || self.AdmissionEpoch != epoch {
			t.Fatal("old compensation changed newer provisional local admission")
		}
		if _, pinned := f.joiner.trust.Get(pi.NodeUUID); !pinned {
			t.Fatal("old compensation removed newer provisional composition's pin")
		}
	})
	t.Run("an activated exact admission cannot become failed compensation", func(t *testing.T) {
		f := newOutboundPairingFixture(t)
		final := f.authenticatedSuccess(t)
		f.joiner.handleOutboundCompletion(httptest.NewRecorder(), &pairingEnvelope{InviteID: f.session.inviteID}, final)
		receipt, ok := f.joiner.getPairingReceipt(f.session.inviteID)
		if !ok {
			t.Fatal("completed join receipt missing")
		}
		if err := f.joiner.rollbackPairingPeerLocked(receipt); err == nil {
			t.Fatal("activated join admission was falsely compensated")
		}
		if _, pinned := f.joiner.trust.Get(f.inviter.identity.NodeUUID); !pinned {
			t.Fatal("refused compensation removed activated authority")
		}
	})
	t.Run("a newer peer incarnation survives old join compensation", func(t *testing.T) {
		f := newOutboundPairingFixture(t)
		f.authenticatedSuccess(t)
		inv, _ := f.joiner.getInvite(f.session.inviteID)
		receipt := f.joiner.newPairingReceipt(f.joinSession, inv, nil)
		pi := *f.joinSession.peerPairing
		pi.AdmissionEpoch++
		now := time.Now().UnixMilli()
		f.joiner.recordMember(&pi, stateMember, &now)
		f.joiner.addSelfMemberForAdmission(receipt.ClusterID, receipt.LocalEpoch)
		if err := f.joiner.persistMembersErr(); err != nil {
			t.Fatal(err)
		}
		if err := f.joiner.pinPeer(&pi, identityCertificate(t, f.inviter), nil); err != nil {
			t.Fatal(err)
		}
		if err := f.joiner.rollbackPairingPeerLocked(receipt); err != nil {
			t.Fatal(err)
		}
		pin, pinned := f.joiner.trust.Get(pi.NodeUUID)
		member, present := f.joiner.memberByNodeID(pi.NodeUUID)
		if !pinned || !present || pin.AdmissionEpoch != pi.AdmissionEpoch || member.AdmissionEpoch != pi.AdmissionEpoch {
			t.Fatal("old join compensation removed a newer peer incarnation")
		}
	})
	t.Run("a newer local admission survives old join compensation", func(t *testing.T) {
		f := newOutboundPairingFixture(t)
		f.authenticatedSuccess(t)
		inv, _ := f.joiner.getInvite(f.session.inviteID)
		receipt := f.joiner.newPairingReceipt(f.joinSession, inv, nil)
		epoch := activateTestCluster(t, f.joiner, receipt.ClusterID)
		if epoch <= receipt.LocalEpoch {
			t.Fatal("fixture did not advance local admission")
		}
		pi := f.joinSession.peerPairing
		now := time.Now().UnixMilli()
		f.joiner.recordMember(pi, stateMember, &now)
		f.joiner.addSelfMemberForAdmission(receipt.ClusterID, epoch)
		if err := f.joiner.persistMembersErr(); err != nil {
			t.Fatal(err)
		}
		if err := f.joiner.pinPeer(pi, identityCertificate(t, f.inviter), nil); err != nil {
			t.Fatal(err)
		}
		if err := f.joiner.rollbackPairingPeerLocked(receipt); err != nil {
			t.Fatal(err)
		}
		cid, current := f.joiner.currentAdmission()
		if cid != receipt.ClusterID || current != epoch {
			t.Fatal("old join compensation changed newer local admission")
		}
		if _, pinned := f.joiner.trust.Get(pi.NodeUUID); !pinned {
			t.Fatal("old join compensation removed newer local composition's pin")
		}
	})
}

func TestJoinerCompensationWithoutWritableJournalPreservesFallback(t *testing.T) {
	f := newOutboundPairingFixture(t)
	final := f.authenticatedSuccess(t)
	var restores []func()
	faulted := false
	f.joiner.trust.SetOnChange(func() {
		if faulted {
			return
		}
		faulted = true
		restores = append(restores, conflictPairingPath(t, f.joiner.admissionPath()),
			conflictPairingPath(t, filepath.Join(f.joiner.clusterDir, pairingReceiptsFile)))
	})
	f.joiner.handleOutboundCompletion(httptest.NewRecorder(), &pairingEnvelope{InviteID: f.session.inviteID}, final)
	assertJoinerCompensationPending(t, f)
	for _, restore := range restores {
		restore()
	}
	// A failed retry-intent write must preserve the provisional self admission
	// for exact cleanup. PG's staged abort and guard also correlate restart
	// recovery; a failed startup pin removal must still refuse authority.
	restorePin := conflictPairingPath(t, f.joiner.trust.pinPath(f.inviter.identity.NodeUUID))
	codec := NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), io.Discard})
	if manager, err := NewManager(codec, filepath.Dir(f.joiner.clusterDir), 9); err == nil || manager != nil {
		t.Fatalf("faulted fallback startup served authority: manager present=%t error=%v", manager != nil, err)
	}
	restorePin()
	restarted := testManagerAt(t, filepath.Dir(f.joiner.clusterDir), 9)
	if _, pinned := restarted.trust.Get(f.inviter.identity.NodeUUID); pinned {
		t.Fatal("fallback restart restored a failed join's durable pin")
	}
	if _, member := restarted.memberByNodeID(restarted.identity.NodeUUID); member {
		t.Fatal("fallback restart restored a failed join's provisional admission")
	}
}

func TestAwaitingPairingConfirmationExpiresWithoutCompensation(t *testing.T) {
	f := committedRollbackFixture(t)
	f.inviter.finishInviteReason(f.session.inviteID, inviteStatePending, reasonAwaitingConfirmation)
	f.inviter.expirePendingInvites(time.Now().Add(2 * f.inviter.effectiveInviteTTL()))
	inv, ok := f.inviter.getInvite(f.session.inviteID)
	if !ok || inv.State != inviteStateExpired {
		t.Fatalf("awaiting-confirmation exceeded its existing invite TTL: %+v", inv)
	}
	if _, pinned := f.inviter.trust.Get(f.joiner.identity.NodeUUID); !pinned {
		t.Fatal("confirmation expiry compensated a committed peer without authenticated failure")
	}
	if _, member := f.inviter.memberByNodeID(f.joiner.identity.NodeUUID); !member {
		t.Fatal("confirmation expiry removed committed membership")
	}
}

// Preserve the exact existing durable bytes while placing a portable directory
// conflict at a write boundary. Restoring the old path models a failed write's
// unchanged durable contents and works even with root/Windows test runners.
func conflictPairingPath(t *testing.T, path string) func() {
	t.Helper()
	backup := path + ".fault-backup"
	_, statErr := os.Stat(path)
	existed := statErr == nil
	if statErr != nil && !os.IsNotExist(statErr) {
		t.Fatal(statErr)
	}
	if existed {
		if err := os.Rename(path, backup); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "blocks-removal"), []byte("fault"), 0o600); err != nil {
		t.Fatal(err)
	}
	restored := false
	restore := func() {
		if restored {
			return
		}
		restored = true
		if err := os.RemoveAll(path); err != nil {
			t.Fatal(err)
		}
		if existed {
			if err := os.Rename(backup, path); err != nil {
				t.Fatal(err)
			}
		}
	}
	t.Cleanup(restore)
	return restore
}

func committedRollbackFixture(t *testing.T) outboundPairingFixture {
	t.Helper()
	f := newOutboundPairingFixture(t)
	final := f.authenticatedSuccess(t)
	_, ok := f.inviter.getInvite(f.session.inviteID)
	if !ok {
		t.Fatal("inviter record missing")
	}
	receipt, ok := f.inviter.getPairingReceipt(f.session.inviteID)
	if !ok {
		t.Fatal("inviter transaction receipt missing")
	}
	receipt.SuccessBlob = final
	if err := f.inviter.savePairingReceipt(receipt); err != nil {
		t.Fatal(err)
	}
	return f
}

func failCommittedPairing(t *testing.T, f outboundPairingFixture) {
	t.Helper()
	f.inviter.handlePairingTerminalSignal(&pairingEnvelope{InviteID: f.session.inviteID}, inviteStateFailed, "",
		"cluster:invite-failed", f.joiner.identity.NodeUUID)
}

func assertRollbackPending(t *testing.T, f outboundPairingFixture) {
	t.Helper()
	inv, ok := f.inviter.getInvite(f.session.inviteID)
	if !ok || inv.State != "pending" || inv.Reason != "rollback-pending" {
		t.Fatalf("incomplete compensation status = %+v, want pending/rollback-pending", inv)
	}
	if sess, ok := f.inviter.getSession(f.session.inviteID); !ok || !sess.awaitingAck || !sess.rollbackPending {
		t.Fatal("incomplete compensation lost its exact committed-session retry")
	}
}

func TestPairingRollbackRetriesEitherWriteOrderBeforeTerminalFailure(t *testing.T) {
	test := func(name string, pinFault, memberFault bool) {
		t.Run(name, func(t *testing.T) {
			f := committedRollbackFixture(t)
			var restores []func()
			if pinFault {
				restores = append(restores, conflictPairingPath(t, f.inviter.trust.pinPath(f.joiner.identity.NodeUUID)))
			}
			if memberFault {
				restores = append(restores, conflictPairingPath(t, f.inviter.membersPath()))
			}
			failCommittedPairing(t, f)
			assertRollbackPending(t, f)
			if !pinFault {
				if _, err := os.Stat(f.inviter.trust.pinPath(f.joiner.identity.NodeUUID)); !os.IsNotExist(err) {
					t.Fatalf("successful pin compensation did not land before member failure: %v", err)
				}
			}
			if !memberFault {
				data, err := os.ReadFile(f.inviter.membersPath())
				if err != nil {
					t.Fatal(err)
				}
				var members []ClusterNode
				if err := json.Unmarshal(data, &members); err != nil {
					t.Fatal(err)
				}
				for _, member := range members {
					if member.NodeUUID == f.joiner.identity.NodeUUID {
						t.Fatal("successful member compensation did not land before pin failure")
					}
				}
			}
			receipt, ok := f.inviter.getPairingReceipt(f.session.inviteID)
			if !ok || !receipt.Rollback || receipt.State == inviteStateFailed {
				t.Fatalf("incomplete compensation receipt = %+v", receipt)
			}
			// Neither an ack nor a pre-commit decline can erase known failure.
			f.inviter.ackPairing(&pairingEnvelope{InviteID: f.session.inviteID}, f.joiner.identity.NodeUUID)
			f.inviter.handlePairingTerminalSignal(&pairingEnvelope{InviteID: f.session.inviteID}, inviteStateDeclined, "", "cluster:invite-declined", "")
			assertRollbackPending(t, f)
			for _, restore := range restores {
				restore()
			}
			// Sweep retry must operate on a committed pending invite, rather
			// than treating it as an uncommitted invite or expiring its journal.
			f.inviter.expirePendingInvites(time.Now())
			inv, _ := f.inviter.getInvite(f.session.inviteID)
			if inv.State != "failed" {
				t.Fatalf("durable compensation status = %q, want failed", inv.State)
			}
			data, err := os.ReadFile(f.inviter.membersPath())
			if err != nil {
				t.Fatal(err)
			}
			var durableMembers []ClusterNode
			if err := json.Unmarshal(data, &durableMembers); err != nil {
				t.Fatal(err)
			}
			for _, member := range durableMembers {
				if member.NodeUUID == f.joiner.identity.NodeUUID {
					t.Fatal("terminal failure was published before durable member cleanup")
				}
			}
			if _, err := os.Stat(f.inviter.trust.pinPath(f.joiner.identity.NodeUUID)); !os.IsNotExist(err) {
				t.Fatalf("terminal failure was published before durable pin cleanup: %v", err)
			}
			restarted := testManagerAt(t, filepath.Dir(f.inviter.clusterDir), 6)
			if _, pinned := restarted.trust.Get(f.joiner.identity.NodeUUID); pinned {
				t.Fatal("completed rollback restored old durable pin")
			}
			if _, member := restarted.memberByNodeID(f.joiner.identity.NodeUUID); member {
				t.Fatal("completed rollback restored old durable member")
			}
		})
	}
	test("pin deletion fails after member write succeeds", true, false)
	test("member write fails after pin deletion succeeds", false, true)
	test("both compensation writes fail", true, true)
}

func TestPairingRollbackJournalReplaysBeforeServing(t *testing.T) {
	f := committedRollbackFixture(t)
	restorePin := conflictPairingPath(t, f.inviter.trust.pinPath(f.joiner.identity.NodeUUID))
	restoreMember := conflictPairingPath(t, f.inviter.membersPath())
	failCommittedPairing(t, f)
	assertRollbackPending(t, f)
	restorePin()
	restoreMember()
	// No in-process retry: startup itself must finish the compensation.
	restarted := testManagerAt(t, filepath.Dir(f.inviter.clusterDir), 7)
	if _, pinned := restarted.trust.Get(f.joiner.identity.NodeUUID); pinned {
		t.Fatal("restart served the old durable pin before replaying rollback")
	}
	receipt, ok := restarted.getPairingReceipt(f.session.inviteID)
	if !ok || receipt.Rollback || receipt.State != "failed" {
		t.Fatalf("restart did not durably finish rollback: %+v", receipt)
	}
}

func TestPairingRollbackTerminalReceiptFailureStaysPending(t *testing.T) {
	f := committedRollbackFixture(t)
	var restore func()
	onChange := f.inviter.trust.onChange
	f.inviter.trust.SetOnChange(func() {
		if onChange != nil {
			onChange()
		}
		if restore == nil {
			// The intent is already durable when pin removal announces its
			// change. Fault only the final receipt write after compensation.
			restore = conflictPairingPath(t, filepath.Join(f.inviter.clusterDir, pairingReceiptsFile))
		}
	})
	failCommittedPairing(t, f)
	assertRollbackPending(t, f)
	if restore == nil {
		t.Fatal("fault did not reach the post-pin terminal receipt boundary")
	}
	if _, err := os.Stat(f.inviter.trust.pinPath(f.joiner.identity.NodeUUID)); !os.IsNotExist(err) {
		t.Fatalf("terminal receipt fault preceded durable pin compensation: %v", err)
	}
	restore()
	f.inviter.expirePendingInvites(time.Now())
	if inv, _ := f.inviter.getInvite(f.session.inviteID); inv.State != "failed" {
		t.Fatalf("terminal receipt retry did not publish completed failure: %+v", inv)
	}
}

func TestPairingRollbackRecoveryFailsClosedUntilStorageIsWritable(t *testing.T) {
	f := committedRollbackFixture(t)
	restore := conflictPairingPath(t, f.inviter.trust.pinPath(f.joiner.identity.NodeUUID))
	failCommittedPairing(t, f)
	assertRollbackPending(t, f)
	codec := NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), io.Discard})
	restarted, err := NewManager(codec, filepath.Dir(f.inviter.clusterDir), 7)
	if err == nil || restarted != nil || !strings.Contains(err.Error(), "recover pairing rollback") {
		t.Fatalf("failed journal replay served authority: manager=%v error=%v", restarted, err)
	}
	restore()
	restarted = testManagerAt(t, filepath.Dir(f.inviter.clusterDir), 7)
	if _, pinned := restarted.trust.Get(f.joiner.identity.NodeUUID); pinned {
		t.Fatal("recovery restored authority after the storage fault was repaired")
	}
}

func TestPairingRollbackWithoutWritableIntentDoesNotRepublishSuccess(t *testing.T) {
	f := committedRollbackFixture(t)
	restore := conflictPairingPath(t, filepath.Join(f.inviter.clusterDir, pairingReceiptsFile))
	failCommittedPairing(t, f)
	assertRollbackPending(t, f)
	if _, pinned := f.inviter.trust.Get(f.joiner.identity.NodeUUID); pinned {
		t.Fatal("known failure retained exact pin in live authorization")
	}
	restore()
	// The retry-intent write failed, but PG's durable pairing-owned guard still
	// correlates the original receipt with a confirmed local abort. Recovery
	// completes that exact compensation instead of querying a paired outcome.
	restarted := testManagerAt(t, filepath.Dir(f.inviter.clusterDir), 8)
	restarted.resumeOutboundPairings()
	receipt, ok := restarted.getPairingReceipt(f.session.inviteID)
	if !ok || receipt.Rollback || receipt.State != inviteStateFailed {
		t.Fatalf("guarded recovery did not complete exact failed compensation: found=%t state=%s rollback=%t", ok, receipt.State, receipt.Rollback)
	}
	if inv, exists := restarted.getInvite(f.session.inviteID); exists {
		t.Fatalf("completed abort recreated an outbound invite: %+v", inv)
	}
	if _, live := restarted.getSession(f.session.inviteID); live {
		t.Fatal("completed abort recreated an outbound session")
	}
	if _, pinned := restarted.trust.Get(f.joiner.identity.NodeUUID); pinned {
		t.Fatal("completed abort retained its peer pin")
	}
	if _, member := restarted.memberByNodeID(f.joiner.identity.NodeUUID); member {
		t.Fatal("completed abort retained its peer member")
	}
	if _, err := os.Stat(restarted.remoteManagementInitializingPath()); !os.IsNotExist(err) {
		t.Fatalf("completed abort did not release its publication guard: %v", err)
	}
	if permission := clustertrust.Open(restarted.clusterDir).RemoteManagement(f.joiner.identity.NodeUUID); permission.Allowed {
		t.Fatalf("completed abort retained management authority: %+v", permission)
	}
}

func TestPairingRollbackPreservesNewerPeerIncarnation(t *testing.T) {
	f := committedRollbackFixture(t)
	restore := conflictPairingPath(t, f.inviter.membersPath())
	failCommittedPairing(t, f)
	assertRollbackPending(t, f)
	restore()
	pi := *f.session.peerPairing
	pi.AdmissionEpoch++
	now := time.Now().UnixMilli()
	f.inviter.recordMember(&pi, stateMember, &now)
	if err := f.inviter.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	cert := identityCertificate(t, f.joiner)
	if err := f.inviter.pinPeer(&pi, cert, nil); err != nil {
		t.Fatal(err)
	}
	f.inviter.expirePendingInvites(time.Now())
	pin, ok := f.inviter.trust.Get(pi.NodeUUID)
	member, memberOK := f.inviter.memberByNodeID(pi.NodeUUID)
	if !ok || !memberOK || pin.AdmissionEpoch != pi.AdmissionEpoch || member.AdmissionEpoch != pi.AdmissionEpoch {
		t.Fatal("old exact-tuple retry removed a newer peer admission")
	}
	if inv, _ := f.inviter.getInvite(f.session.inviteID); inv.State != "failed" {
		t.Fatalf("old exact invite status = %q, want failed", inv.State)
	}
}

func TestPairingRollbackPreservesNewerLocalIncarnation(t *testing.T) {
	f := committedRollbackFixture(t)
	restore := conflictPairingPath(t, filepath.Join(f.inviter.clusterDir, pairingReceiptsFile))
	failCommittedPairing(t, f)
	assertRollbackPending(t, f)
	restore()
	if err := f.inviter.clearAdmission(); err != nil {
		t.Fatal(err)
	}
	epoch := activateTestCluster(t, f.inviter, "outbound-test-cluster")
	if epoch <= f.session.localAdmissionEpoch {
		t.Fatal("fixture did not advance the local admission")
	}
	f.inviter.addSelfMemberForAdmission("outbound-test-cluster", epoch)
	if err := f.inviter.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	pi := *f.session.peerPairing
	if err := f.inviter.pinPeer(&pi, identityCertificate(t, f.joiner), nil); err != nil {
		t.Fatal(err)
	}
	f.inviter.expirePendingInvites(time.Now())
	if cid, gotEpoch := f.inviter.currentAdmission(); cid != "outbound-test-cluster" || gotEpoch != epoch {
		t.Fatalf("old compensation changed newer local admission %q/%d", cid, gotEpoch)
	}
	if _, pinned := f.inviter.trust.Get(pi.NodeUUID); !pinned {
		t.Fatal("old local-incarnation compensation removed the newer composition's peer pin")
	}
	if _, member := f.inviter.memberByNodeID(pi.NodeUUID); !member {
		t.Fatal("old local-incarnation compensation removed the newer composition's peer member")
	}
}

func capturePairingNotifications(m *Manager) *bytes.Buffer {
	var output bytes.Buffer
	m.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), &output})
	return &output
}

func assertEmptyPublicMembershipAndNotification(t *testing.T, m *Manager, output *bytes.Buffer) {
	t.Helper()
	id := json.RawMessage("1")
	m.handleGetInitial(&Message{ID: &id})
	decoder := json.NewDecoder(bytes.NewReader(output.Bytes()))
	changed, snapshot := false, false
	for decoder.More() {
		var msg Message
		if err := decoder.Decode(&msg); err != nil {
			t.Fatal(err)
		}
		var members struct {
			Nodes []ClusterNode `json:"nodes"`
		}
		if msg.Method == "nodes:changed" {
			if err := json.Unmarshal(msg.Params, &members); err != nil {
				t.Fatal(err)
			}
			changed = true
		} else if msg.ID != nil {
			if err := json.Unmarshal(msg.Result, &members); err != nil {
				t.Fatal(err)
			}
			snapshot = true
		} else {
			continue
		}
		if len(members.Nodes) != 0 {
			t.Fatalf("failed invite retained public membership: %+v", members.Nodes)
		}
	}
	if !changed || !snapshot {
		t.Fatalf("receiver cleanup proof: nodes:changed=%v nodes:get-initial=%v", changed, snapshot)
	}
}

func TestPairingReceiptWriteFailureClearsPendingMembership(t *testing.T) {
	f := newOutboundPairingFixture(t)
	final := f.authenticatedSuccess(t)
	output := capturePairingNotifications(f.joiner)
	conflictPairingPath(t, filepath.Join(f.joiner.clusterDir, pairingReceiptsFile))
	rr := httptest.NewRecorder()
	f.joiner.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: f.session.inviteID}, final)
	if inv, _ := f.joiner.getInvite(f.session.inviteID); inv.State != "failed" {
		t.Fatalf("receipt failure outcome = %+v", inv)
	}
	assertEmptyPublicMembershipAndNotification(t, f.joiner, output)
	if cid, epoch := f.joiner.currentAdmission(); cid != "" || epoch != 0 {
		t.Fatalf("receipt failure activated admission %q/%d", cid, epoch)
	}
	if rr := f.confirm(t, identityCertificate(t, f.inviter), f.joinSession.localAdmissionEpoch); rr.Code != http.StatusOK {
		t.Fatalf("completed transcript lost pairing-only failure authority: %d", rr.Code)
	}
}

func TestPendingPairingTranscriptCannotConfirm(t *testing.T) {
	f := newOutboundPairingFixture(t)
	if rr := f.confirm(t, identityCertificate(t, f.inviter), f.joinSession.localAdmissionEpoch); rr.Code != http.StatusForbidden {
		t.Fatalf("pending transcript confirmed a terminal outcome: %d", rr.Code)
	}
}

func TestWrongPINReceiverPublishesEmptyMembership(t *testing.T) {
	f := newOutboundPairingFixture(t)
	output := capturePairingNotifications(f.joiner)
	if f.pin == "000000" {
		f.pin = "000001"
	} else {
		f.pin = "000000"
	}
	f.ready(t)
	blob, err := f.session.server.Start()
	if err != nil {
		t.Fatal(err)
	}
	for {
		rr := httptest.NewRecorder()
		f.joiner.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: f.session.inviteID}, blob)
		if rr.Code != http.StatusOK {
			t.Fatalf("wrong-PIN round status = %d", rr.Code)
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
		if err != nil {
			t.Fatal(err)
		}
		if out.Done {
			if out.Success || len(out.Send) == 0 {
				t.Fatal("wrong PIN did not produce its terminal EAP error")
			}
			rr = httptest.NewRecorder()
			f.joiner.handleOutboundCompletion(rr, &pairingEnvelope{InviteID: f.session.inviteID}, out.Send)
			if rr.Code != http.StatusOK {
				t.Fatal(rr.Code)
			}
			break
		}
		blob = out.Send
	}
	if inv, _ := f.joiner.getInvite(f.session.inviteID); inv.State != "failed" || inv.Reason != "incorrect-pin" {
		t.Fatalf("receiver wrong-PIN terminal status = %+v", inv)
	}
	assertEmptyPublicMembershipAndNotification(t, f.joiner, output)
}
