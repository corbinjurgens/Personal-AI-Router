// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const pairingReceiptsFile = "pairing-receipts.json"

// Receipts contain public transcript certificates and exact admission tuples.
// They cannot authorize any endpoint other than exact-invite confirmation.
// Membership/pin/admission remain the authority for deciding whether commit
// landed; the receipt only supplies correlation unavailable in those stores.
type pairingReceipt struct {
	InviteID    string          `json:"inviteId"`
	Role        pairingRole     `json:"role"`
	ClusterID   string          `json:"clusterId"`
	LocalEpoch  uint64          `json:"localEpoch"`
	PeerInfo    json.RawMessage `json:"peerInfo"`
	Addr        string          `json:"addr,omitempty"`
	SuccessBlob []byte          `json:"successBlob,omitempty"`
	ExpiresAt   int64           `json:"expiresAt"`
	State       InviteState     `json:"state,omitempty"`
	Rollback    bool            `json:"rollback,omitempty"`
}

func (m *Manager) newPairingReceipt(sess *pairingSession, inv *Invite, success []byte) pairingReceipt {
	info, _ := json.Marshal(sess.peerPairing) // PairingInfo has only JSON scalar fields.
	epoch := sess.localAdmissionEpoch
	if sess.role == roleInviter && epoch == 0 {
		_, epoch = m.currentAdmission()
	}
	return pairingReceipt{InviteID: sess.inviteID, Role: sess.role, ClusterID: inv.ClusterID,
		LocalEpoch: epoch, PeerInfo: info, Addr: sess.joinerAddr,
		SuccessBlob: append([]byte(nil), success...), ExpiresAt: time.Now().Add(m.effectiveInviteTTL()).UnixMilli()}
}

func (m *Manager) loadPairingReceipts() error {
	data, err := os.ReadFile(filepath.Join(m.clusterDir, pairingReceiptsFile))
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	var receipts map[string]pairingReceipt
	if err := json.Unmarshal(data, &receipts); err != nil {
		return err
	}
	if len(receipts) > maxPreInviteSessions {
		return fmt.Errorf("too many pairing receipts")
	}
	for id, receipt := range receipts {
		if receipt.InviteID != id || receipt.LocalEpoch == 0 || receipt.ClusterID == "" ||
			(receipt.Role != roleInviter && receipt.Role != roleJoiner) {
			return fmt.Errorf("invalid pairing receipt")
		}
		if _, _, err := parsePairingInfo(receipt.PeerInfo); err != nil {
			return err
		}
	}
	m.pairingReceipts = receipts
	return nil
}

func (m *Manager) savePairingReceipt(receipt pairingReceipt) error {
	m.receiptMu.Lock()
	defer m.receiptMu.Unlock()
	next := make(map[string]pairingReceipt)
	for id, current := range m.pairingReceipts {
		if current.Rollback || current.ExpiresAt > time.Now().UnixMilli() {
			next[id] = current
		}
	}
	next[receipt.InviteID] = receipt
	if len(next) > maxPreInviteSessions {
		return fmt.Errorf("pairing receipt capacity exhausted")
	}
	data, err := json.Marshal(next)
	if err != nil {
		return err
	}
	if err := atomicWrite(filepath.Join(m.clusterDir, pairingReceiptsFile), data, 0o600); err != nil {
		return err
	}
	m.pairingReceipts = next
	return nil
}

func (m *Manager) getPairingReceipt(id string) (pairingReceipt, bool) {
	m.receiptMu.Lock()
	defer m.receiptMu.Unlock()
	receipt, ok := m.pairingReceipts[id]
	return receipt, ok && (receipt.Rollback || receipt.ExpiresAt > time.Now().UnixMilli())
}

func (m *Manager) receiptOutcome(receipt pairingReceipt, pi *PairingInfo, cert *x509.Certificate) InviteState {
	if receipt.Rollback {
		return inviteStatePending
	}
	if receipt.Role == roleInviter && receipt.State == inviteStateFailed {
		return inviteStateFailed
	}
	if receipt.Role == roleJoiner && (receipt.State == inviteStatePaired || receipt.State == inviteStateFailed) {
		return receipt.State
	}
	m.rosterMu.Lock()
	defer m.rosterMu.Unlock()
	if m.receiptCommitMatchesLocked(receipt, pi, cert) {
		return inviteStatePaired
	}
	// The durable activated high-water mark can prove the exact join never
	// activated. Missing pins after a later leave/removal are an unknown outcome,
	// not evidence that a previously activated pairing failed its commit.
	m.admissionMu.Lock()
	activated := m.admissionActivated
	m.admissionMu.Unlock()
	if receipt.Role == roleJoiner && activated < receipt.LocalEpoch {
		return inviteStateFailed
	}
	return ""
}

// Caller holds the composition boundary. Recovery must use the same exact
// tuple at inspection and at installation, because removal can race startup.
func (m *Manager) receiptCommitMatchesLocked(receipt pairingReceipt, pi *PairingInfo, cert *x509.Certificate) bool {
	cid, epoch := m.currentAdmission()
	member, memberOK := m.memberByNodeID(pi.NodeUUID)
	pin, pinOK := m.trust.Get(pi.NodeUUID)
	if !m.teardownPending.Load() && cid == receipt.ClusterID && epoch == receipt.LocalEpoch &&
		memberOK && pinOK && member.ClusterID == cid && pin.ClusterID == cid &&
		member.AdmissionEpoch == pi.AdmissionEpoch && pin.AdmissionEpoch == pi.AdmissionEpoch &&
		m.trust.MatchDER(pi.NodeUUID, cert.Raw) {
		return true
	}
	return false
}

func (m *Manager) resumeOutboundPairings() {
	m.receiptMu.Lock()
	var receipts []pairingReceipt
	for _, receipt := range m.pairingReceipts {
		if receipt.Role == roleInviter && !receipt.Rollback && receipt.State != inviteStateFailed && receipt.ExpiresAt > time.Now().UnixMilli() {
			receipts = append(receipts, receipt)
		}
	}
	m.receiptMu.Unlock()
	for _, receipt := range receipts {
		pi, cert, err := parsePairingInfo(receipt.PeerInfo)
		if err != nil || m.receiptOutcome(receipt, pi, cert) != inviteStatePaired {
			continue
		}
		// EAP need not be resumed: only its already-authenticated final Success
		// and pairing-only confirmation remain, both bound to the exact tuple.
		sess := &pairingSession{inviteID: receipt.InviteID, role: roleInviter,
			clusterID: receipt.ClusterID, joinerAddr: receipt.Addr,
			localAdmissionEpoch: receipt.LocalEpoch,
			peerPairing:         pi, peerCert: cert, awaitingAck: true,
			completionMode: inviterCompletionV1, completionResponse: receipt.SuccessBlob,
			createdAt: time.Now().UnixMilli()}
		m.inviteMu.Lock()
		installed := false
		m.withClusterComposition(func() {
			if !m.receiptCommitMatchesLocked(receipt, pi, cert) {
				return
			}
			m.putInvite(&Invite{InviteID: receipt.InviteID, FromNodeUUID: m.identity.NodeUUID,
				ClusterID: receipt.ClusterID, State: inviteStatePending, Reason: reasonAwaitingConfirmation, CreatedAt: sess.createdAt})
			m.putSession(sess)
			installed = true
		})
		m.inviteMu.Unlock()
		if !installed {
			continue
		}
		client := &http.Client{Timeout: pairingHTTPTimeout}
		_, _ = postPairingBlob(client, receipt.Addr, receipt.InviteID, "outbound-completion", receipt.SuccessBlob)
		client.CloseIdleConnections()
		m.confirmOutboundPairing(sess)
	}
}

// Caller holds rosterMu. Both durable deletion and live pin denial use this
// local-incarnation guard, including a newer provisional join admission.
func (m *Manager) pairingRollbackMatchesAdmissionLocked(receipt pairingReceipt) (bool, error) {
	cid, epoch := m.currentAdmission()
	if receipt.Role == roleInviter {
		if cid != receipt.ClusterID || epoch != receipt.LocalEpoch {
			return false, nil // this invite cannot compensate a different local admission
		}
	} else {
		if cid != "" || epoch != 0 {
			if cid == receipt.ClusterID && epoch == receipt.LocalEpoch {
				return false, fmt.Errorf("cannot compensate an activated join admission")
			}
			return false, nil // preserve a newer activated local admission
		}
		m.admissionMu.Lock()
		activated := m.admissionActivated
		m.admissionMu.Unlock()
		if activated >= receipt.LocalEpoch {
			return false, nil
		}
		if self, ok := m.memberByNodeID(m.identity.NodeUUID); ok && self.AdmissionEpoch != 0 &&
			(self.ClusterID != receipt.ClusterID || self.AdmissionEpoch != receipt.LocalEpoch) {
			return false, nil // preserve a newer provisional local admission too
		}
	}
	return true, nil
}

// Applies only the exact tuple. Pin/member writes are checked independently so
// either partial order is retryable; the durable pin is read after Forget too.
// Caller holds rosterMu and has persisted the receipt's rollback intent.
func (m *Manager) rollbackPairingPeerLocked(receipt pairingReceipt) error {
	pi, cert, err := parsePairingInfo(receipt.PeerInfo)
	if err != nil {
		return err
	}
	eligible, err := m.pairingRollbackMatchesAdmissionLocked(receipt)
	if err != nil || !eligible {
		return err
	}
	var rollbackErr error
	data, readErr := os.ReadFile(m.trust.pinPath(pi.NodeUUID))
	if readErr != nil && !os.IsNotExist(readErr) {
		rollbackErr = readErr
		if pin, ok := m.trust.Get(pi.NodeUUID); ok && pin.ClusterID == receipt.ClusterID &&
			pin.AdmissionEpoch == pi.AdmissionEpoch && pin.CertFingerprint == certFingerprintFromDER(cert.Raw) {
			rollbackErr = errors.Join(readErr, m.trust.Remove(pi.NodeUUID))
			m.trust.Forget(pi.NodeUUID)
		}
	} else if readErr == nil {
		var pin TrustedPin
		if err := json.Unmarshal(data, &pin); err != nil {
			rollbackErr = err
		} else if pin.NodeUUID == pi.NodeUUID && pin.ClusterID == receipt.ClusterID &&
			pin.AdmissionEpoch == pi.AdmissionEpoch && pin.CertFingerprint == certFingerprintFromDER(cert.Raw) {
			if err := m.trust.Remove(pi.NodeUUID); err != nil {
				m.trust.Forget(pi.NodeUUID)
				rollbackErr = err
			}
		}
	}
	if member, ok := m.memberByNodeID(pi.NodeUUID); ok &&
		member.ClusterID == receipt.ClusterID && member.AdmissionEpoch == pi.AdmissionEpoch {
		m.removeMemberByUUID(pi.NodeUUID)
	}
	var provisionalSelf *ClusterNode
	if receipt.Role == roleJoiner && rollbackErr == nil {
		if self, ok := m.memberByNodeID(m.identity.NodeUUID); ok &&
			self.ClusterID == receipt.ClusterID && self.AdmissionEpoch == receipt.LocalEpoch {
			provisionalSelf = self
			m.removeMemberByUUID(m.identity.NodeUUID)
		}
	}
	// Persist even when a previous attempt already removed the live member but
	// failed its durable write. A newer member is retained in this snapshot.
	memberErr := m.persistMembersErr()
	if memberErr != nil && provisionalSelf != nil {
		// Keep this in-memory transaction marker too: another invitation cannot
		// treat unfinished compensation as standalone eligibility.
		m.upsertMember(provisionalSelf)
	}
	return errors.Join(rollbackErr, memberErr)
}

// rollbackPairingSessionLocked never advertises a completed failure until the
// compensation and terminal receipt are durable. Caller holds inviteMu/sess.mu.
func (m *Manager) rollbackPairingSessionLocked(sess *pairingSession, inv *Invite) (rollbackErr error) {
	m.withClusterComposition(func() {
		rollbackErr = m.rollbackPairingSessionCompositionLocked(sess, inv)
	})
	return rollbackErr
}

// Shared compensation owner for an inviter's committed peer and a joiner's
// provisional admission. Caller holds sess.mu and the composition boundary.
func (m *Manager) rollbackPairingSessionCompositionLocked(sess *pairingSession, inv *Invite) (rollbackErr error) {
	sess.rollbackPending = true
	m.finishInviteReason(sess.inviteID, inviteStatePending, reasonRollbackPending)
	receipt, exists := m.getPairingReceipt(sess.inviteID)
	if !exists {
		receipt = m.newPairingReceipt(sess, inv, sess.completionResponse)
		if sess.role == roleInviter {
			receipt.ClusterID = sess.clusterID
		}
	}
	receipt.Rollback = true
	if current, live := m.getSession(sess.inviteID); !live || current != sess {
		return errPairingCommitStale
	}
	if err := m.savePairingReceipt(receipt); err != nil {
		// There is no durable failure claim when even the intent cannot be
		// saved. Deny the exact old pin in this process and retain the retry.
		eligible, _ := m.pairingRollbackMatchesAdmissionLocked(receipt)
		if pin, ok := m.trust.Get(sess.peerPairing.NodeUUID); ok && eligible &&
			pin.ClusterID == receipt.ClusterID && pin.AdmissionEpoch == sess.peerPairing.AdmissionEpoch &&
			pin.CertFingerprint == certFingerprintFromDER(sess.peerCert.Raw) {
			m.trust.Forget(pin.NodeUUID)
		}
		return err
	}
	if err := m.rollbackPairingPeerLocked(receipt); err != nil {
		return err
	}
	receipt.Rollback = false
	receipt.State = inviteStateFailed
	receipt.ExpiresAt = time.Now().Add(m.effectiveInviteTTL()).UnixMilli()
	if err := m.savePairingReceipt(receipt); err != nil {
		return err
	}
	sess.rollbackPending = false
	m.finishInvite(sess.inviteID, inviteStateFailed)
	if sess.role == roleInviter {
		m.deleteSession(sess.inviteID)
	}
	return nil
}

func (m *Manager) recoverPairingRollbacks() error {
	for _, receipt := range m.pairingReceipts {
		if !receipt.Rollback {
			continue
		}
		m.rosterMu.Lock()
		err := m.rollbackPairingPeerLocked(receipt)
		if err == nil {
			receipt.Rollback = false
			receipt.State = inviteStateFailed
			receipt.ExpiresAt = time.Now().Add(m.effectiveInviteTTL()).UnixMilli()
			err = m.savePairingReceipt(receipt)
		}
		m.rosterMu.Unlock()
		if err != nil {
			return err // no serving a partially compensated admission on restart
		}
	}
	return nil
}

// Caller holds inviteMu. Retain bounded exact-tuple rollback records until
// writes succeed; an expired confirmation window cannot erase known failure.
func (m *Manager) retryPairingRollbacksLocked() {
	m.sessMu.Lock()
	var sessions []*pairingSession
	for _, sess := range m.sessions {
		sessions = append(sessions, sess)
	}
	m.sessMu.Unlock()
	for _, sess := range sessions {
		if !sess.mu.TryLock() {
			continue
		}
		inv, exists := m.getInvite(sess.inviteID)
		if exists && sess.rollbackPending {
			if err := m.rollbackPairingSessionLocked(sess, inv); err != nil {
				log.Printf("pairing %s: rollback remains pending: %v", sess.inviteID, err)
			} else {
				if sess.role == roleInviter {
					m.emitPairingFailure(sess.inviteID)
					m.maybeLeaveInviteCreatedClusterLocked()
				} else if addr := sess.legacyNotifyAddress(); addr != "" && sess.peerCert != nil {
					go m.notifyInviterAuthenticated(addr, append([]byte(nil), sess.peerCert.Raw...), sess.inviteID, "fail")
				}
			}
			m.emitNodesChanged()
		}
		sess.mu.Unlock()
	}
}
