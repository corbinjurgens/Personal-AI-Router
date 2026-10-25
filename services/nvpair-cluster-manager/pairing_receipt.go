// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

const pairingReceiptsFile = "pairing-receipts.json"

// Receipts contain public transcript certificates and exact admission tuples.
// They cannot authorize any endpoint other than exact-invite confirmation.
// Membership/pin/admission remain the authority for deciding whether commit
// landed; the receipt only supplies correlation unavailable in those stores.
type pairingReceipt struct {
	InviteID              string          `json:"inviteId"`
	Role                  pairingRole     `json:"role"`
	ClusterID             string          `json:"clusterId"`
	LocalEpoch            uint64          `json:"localEpoch"`
	PeerInfo              json.RawMessage `json:"peerInfo"`
	Addr                  string          `json:"addr,omitempty"`
	SuccessBlob           []byte          `json:"successBlob,omitempty"`
	ExpiresAt             int64           `json:"expiresAt"`
	State                 InviteState     `json:"state,omitempty"`
	Rollback              bool            `json:"rollback,omitempty"`
	AllowRemoteManagement *bool           `json:"allowRemoteManagement,omitempty"`
	PermissionRevision    string          `json:"permissionRevision,omitempty"`
	PermissionMigrationID string          `json:"permissionMigrationId,omitempty"`
	PermissionIncarnation string          `json:"permissionIncarnation,omitempty"`
	LocalCertFingerprint  string          `json:"localCertFingerprint,omitempty"`
}

func (m *Manager) newPairingReceipt(sess *pairingSession, inv *Invite, success []byte) pairingReceipt {
	info, _ := json.Marshal(sess.peerPairing) // PairingInfo has only JSON scalar fields.
	epoch := sess.localAdmissionEpoch
	if sess.role == roleInviter && epoch == 0 {
		_, epoch = m.currentAdmission()
	}
	return pairingReceipt{InviteID: sess.inviteID, Role: sess.role, ClusterID: inv.ClusterID,
		LocalEpoch: epoch, PeerInfo: info, Addr: sess.joinerAddr,
		AllowRemoteManagement: sess.allowRemoteManagement,
		LocalCertFingerprint:  m.identity.CertFingerprint,
		SuccessBlob:           append([]byte(nil), success...), ExpiresAt: time.Now().Add(m.effectiveInviteTTL()).UnixMilli()}
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
		if receipt.AllowRemoteManagement != nil {
			revision, err := strconv.ParseUint(receipt.PermissionRevision, 10, 64)
			if err != nil || revision == 0 || strconv.FormatUint(revision, 10) != receipt.PermissionRevision || receipt.LocalCertFingerprint == "" {
				return fmt.Errorf("invalid pairing permission revision")
			}
			migrationID, err := hex.DecodeString(receipt.PermissionMigrationID)
			if err != nil || len(migrationID) != 32 || hex.EncodeToString(migrationID) != receipt.PermissionMigrationID {
				return fmt.Errorf("invalid pairing permission migration identity")
			}
			if receipt.PermissionIncarnation != "" {
				incarnation, err := hex.DecodeString(receipt.PermissionIncarnation)
				if err != nil || len(incarnation) != 32 || hex.EncodeToString(incarnation) != receipt.PermissionIncarnation {
					return fmt.Errorf("invalid pairing permission incarnation")
				}
			}
		} else if receipt.PermissionRevision != "" || receipt.PermissionMigrationID != "" {
			return fmt.Errorf("pairing permission revision without local choice")
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
	if err := remoteManagementWrite(filepath.Join(m.clusterDir, pairingReceiptsFile), data); err != nil {
		return err
	}
	m.pairingReceipts = next
	return nil
}

// An incomplete PG transaction is an abort intent. Recovery compensates it,
// rather than replaying a staged allow after a later local permission edit.
// Caller holds rosterMu and sess.mu, before the first authority mutation.
func (m *Manager) stagePairingReceiptLocked(sess *pairingSession, inv *Invite) (pairingReceipt, error) {
	if sess.allowRemoteManagement == nil {
		return pairingReceipt{}, fmt.Errorf("pairing requires a local remote management choice")
	}
	doc, marker, err := m.readRemoteManagementLocked()
	if err != nil || marker.State != "complete" {
		return pairingReceipt{}, fmt.Errorf("remote management unavailable for pairing")
	}
	receipt := m.newPairingReceipt(sess, inv, sess.completionResponse)
	receipt.PermissionRevision = doc.Revision
	receipt.PermissionMigrationID = doc.MigrationID
	receipt.Rollback = true
	if err := m.savePairingReceipt(receipt); err != nil {
		return receipt, err
	}
	if err := m.ensurePairingGuardLocked(receipt); err != nil {
		return receipt, errors.Join(err, m.rollbackPairingSessionCompositionLocked(sess, inv))
	}
	return receipt, nil
}

// The durable success record follows every authority and grant write. A failed
// acknowledgement leaves the original abort intent for compensation/restart.
func (m *Manager) completePairingReceiptLocked(receipt pairingReceipt) error {
	pi, _, err := parsePairingInfo(receipt.PeerInfo)
	if err != nil {
		return err
	}
	if owner, err := m.pairingGuardOwner(); err != nil || owner != receipt.InviteID {
		return fmt.Errorf("pairing publication guard mismatch: %w", err)
	}
	incarnation, ok := m.mesh.PeerServiceIncarnation(pi.NodeUUID)
	if !ok {
		return fmt.Errorf("paired peer has no current incarnation")
	}
	// Capture the existing Mesh token after admission exists, before any grant
	// write. Cleanup can then identify this grant even after pin deletion.
	receipt.PermissionIncarnation = incarnation
	if err := m.savePairingReceipt(receipt); err != nil {
		return err
	}
	if err := m.setRemoteManagementWithGuardLocked(pi.NodeUUID, *receipt.AllowRemoteManagement, receipt.PermissionRevision, receipt.PermissionMigrationID); err != nil {
		return err
	}
	receipt.Rollback = false
	receipt.State = inviteStatePaired
	if err := m.savePairingReceipt(receipt); err != nil {
		return err
	}
	if err := m.releasePairingGuardLocked(receipt); err != nil {
		return err
	}
	if err := m.codec.Notify("cluster:remote-management-changed", m.remoteManagementSnapshotLocked()); err != nil {
		log.Printf("emit paired remote management change failed")
	}
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
	if receipt.State == inviteStateFailed {
		return inviteStateFailed
	}
	m.rosterMu.Lock()
	defer m.rosterMu.Unlock()
	if m.receiptCommitMatchesLocked(receipt, pi, cert) && m.pairingPermissionAvailableLocked(receipt) {
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

func (m *Manager) pairingPermissionAvailableLocked(receipt pairingReceipt) bool {
	doc, marker, err := m.readRemoteManagementLocked()
	if err != nil || marker.State != "complete" {
		return false
	}
	// Old receipts can confirm membership but never create a PG permission.
	if receipt.AllowRemoteManagement == nil {
		return true
	}
	current, err := strconv.ParseUint(doc.Revision, 10, 64)
	expected, expectedErr := strconv.ParseUint(receipt.PermissionRevision, 10, 64)
	return err == nil && expectedErr == nil && current > expected && receipt.State == inviteStatePaired && receipt.PermissionIncarnation != ""
}

// Caller holds the composition boundary. Recovery must use the same exact
// tuple at inspection and at installation, because removal can race startup.
func (m *Manager) receiptCommitMatchesLocked(receipt pairingReceipt, pi *PairingInfo, cert *x509.Certificate) bool {
	if receipt.AllowRemoteManagement != nil && receipt.LocalCertFingerprint != m.identity.CertFingerprint {
		return false
	}
	cid, epoch := m.currentAdmission()
	member, memberOK := m.memberByNodeID(pi.NodeUUID)
	pin, pinOK := m.trust.Get(pi.NodeUUID)
	if !m.teardownPending.Load() && cid == receipt.ClusterID && epoch == receipt.LocalEpoch &&
		memberOK && pinOK && member.State == stateMember && member.ClusterID == cid && pin.ClusterID == cid &&
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
			localAdmissionEpoch:   receipt.LocalEpoch,
			allowRemoteManagement: receipt.AllowRemoteManagement,
			peerPairing:           pi, peerCert: cert, awaitingAck: true,
			completionMode: inviterCompletionV1, completionResponse: receipt.SuccessBlob,
			createdAt: time.Now().UnixMilli()}
		m.inviteMu.Lock()
		installed := false
		m.withClusterComposition(func() {
			if !m.receiptCommitMatchesLocked(receipt, pi, cert) || !m.pairingPermissionAvailableLocked(receipt) {
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
		client := pairingHTTPClient()
		_, replayErr := postPairingBlob(client, receipt.Addr, receipt.InviteID, "outbound-completion", receipt.SuccessBlob)
		if errors.Is(replayErr, errPairingRedirect) {
			log.Printf("pairing %s: receipt replay refused an HTTP redirect", receipt.InviteID)
		}
		client.CloseIdleConnections()
		m.confirmOutboundPairing(sess)
	}
}

// Caller holds rosterMu. Both durable deletion and live pin denial use this
// local-incarnation guard, including a newer provisional join admission.
func (m *Manager) pairingRollbackMatchesAdmissionLocked(receipt pairingReceipt) (bool, error) {
	if receipt.AllowRemoteManagement != nil && receipt.LocalCertFingerprint != m.identity.CertFingerprint {
		return false, nil
	}
	if receipt.AllowRemoteManagement == nil {
		marker, err := m.pairingGuardMigration(receipt)
		if err != nil {
			return false, err
		}
		local := marker.Snapshot.Local
		if remoteManagementAuthorityMatches(local, local.PeerUUID, local.CertFingerprint, receipt.ClusterID, receipt.LocalEpoch) &&
			(local.PeerUUID != m.identity.NodeUUID || local.CertFingerprint != m.identity.CertFingerprint) {
			return false, nil // the migration proves a different local certificate
		}
	}
	cid, epoch := m.currentAdmission()
	if receipt.Role == roleInviter {
		if cid != receipt.ClusterID || epoch != receipt.LocalEpoch {
			return false, nil // this invite cannot compensate a different local admission
		}
	} else {
		if cid != "" || epoch != 0 {
			if cid == receipt.ClusterID && epoch == receipt.LocalEpoch {
				if receipt.Rollback && receipt.AllowRemoteManagement != nil {
					return true, nil
				}
				return false, fmt.Errorf("cannot compensate an activated join admission")
			}
			return false, nil // preserve a newer activated local admission
		}
		m.admissionMu.Lock()
		activated := m.admissionActivated
		m.admissionMu.Unlock()
		if activated >= receipt.LocalEpoch && !(receipt.Rollback && receipt.AllowRemoteManagement != nil && activated == receipt.LocalEpoch && m.admissionWasRetired()) {
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
	if err := m.removePairingRemoteManagementLocked(receipt, pi, certFingerprintFromDER(cert.Raw)); err != nil {
		rollbackErr = err
	}
	// Grant failure is after join activation. Retire only this admission,
	// retaining its monotonic counter and activated high-water mark. A newer
	// admission was excluded by pairingRollbackMatchesAdmissionLocked above.
	if receipt.Role == roleJoiner && receipt.Rollback && receipt.AllowRemoteManagement != nil {
		cid, epoch := m.currentAdmission()
		if cid == receipt.ClusterID && epoch == receipt.LocalEpoch {
			if err := m.clearAdmission(); err != nil {
				rollbackErr = errors.Join(rollbackErr, err)
			} else {
				m.setClusterIdentity("", "")
			}
		}
	}
	data, readErr := os.ReadFile(m.trust.pinPath(pi.NodeUUID))
	if readErr != nil && !os.IsNotExist(readErr) {
		rollbackErr = errors.Join(rollbackErr, readErr)
		if pin, ok := m.trust.Get(pi.NodeUUID); ok && pin.ClusterID == receipt.ClusterID &&
			pin.AdmissionEpoch == pi.AdmissionEpoch && pin.CertFingerprint == certFingerprintFromDER(cert.Raw) {
			rollbackErr = errors.Join(rollbackErr, m.trust.Remove(pi.NodeUUID))
			m.trust.Forget(pi.NodeUUID)
		}
	} else if readErr == nil {
		var pin TrustedPin
		if err := json.Unmarshal(data, &pin); err != nil {
			rollbackErr = errors.Join(rollbackErr, err)
		} else if pin.NodeUUID == pi.NodeUUID && pin.ClusterID == receipt.ClusterID &&
			pin.AdmissionEpoch == pi.AdmissionEpoch && pin.CertFingerprint == certFingerprintFromDER(cert.Raw) {
			if err := m.trust.Remove(pi.NodeUUID); err != nil {
				m.trust.Forget(pi.NodeUUID)
				rollbackErr = errors.Join(rollbackErr, err)
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
	// Confirmation expiry cannot discard an existing abort correlation.
	m.receiptMu.Lock()
	receipt, exists := m.pairingReceipts[sess.inviteID]
	m.receiptMu.Unlock()
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
	guarded := receipt.PermissionMigrationID != "" || (receipt.AllowRemoteManagement == nil && exists)
	if guarded {
		if err := m.ensurePairingGuardLocked(receipt); err != nil {
			return err // never compensate behind another transaction's guard
		}
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
		// Keep provisional/composition evidence until the intent is confirmed.
		// The durable guard denies independent readers and correlates an existing
		// PG or legacy receipt for abort recovery, even if it was still paired.
		return err
	}
	if receipt.AllowRemoteManagement == nil && !guarded {
		// An old in-memory session without a receipt must first establish its
		// durable correlation; never leave an orphaned publication guard.
		if err := m.ensurePairingGuardLocked(receipt); err != nil {
			return err
		}
		guarded = true
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
	if guarded {
		if err := m.releasePairingGuardLocked(receipt); err != nil {
			return err
		}
	}
	sess.rollbackPending = false
	m.finishInvite(sess.inviteID, inviteStateFailed)
	if sess.role == roleInviter {
		m.deleteSession(sess.inviteID)
	}
	return nil
}

func (m *Manager) recoverPairingRollbacks() error {
	guardOwner, err := m.pairingGuardOwner()
	if err != nil {
		return err
	}
	if guardOwner != "" {
		receipt, ok := m.pairingReceipts[guardOwner]
		if !ok {
			return fmt.Errorf("pairing publication guard has no receipt")
		}
		// A crash before unlink never publishes a grant, even if a terminal
		// replacement landed. Abort the exact guarded transaction on restart.
		receipt.Rollback = true
		m.rosterMu.Lock()
		if receipt.AllowRemoteManagement != nil && receipt.PermissionMigrationID == "" {
			err = fmt.Errorf("pairing publication guard has no permission correlation")
		}
		if err == nil {
			err = m.ensurePairingGuardLocked(receipt)
		}
		if err == nil {
			err = m.savePairingReceipt(receipt)
		}
		m.rosterMu.Unlock()
		if err != nil {
			return err
		}
	}
	var receipts []pairingReceipt
	if guardOwner != "" {
		receipts = append(receipts, m.pairingReceipts[guardOwner])
	}
	for id, receipt := range m.pairingReceipts {
		if id != guardOwner {
			receipts = append(receipts, receipt)
		}
	}
	for _, receipt := range receipts {
		if !receipt.Rollback {
			continue
		}
		m.rosterMu.Lock()
		var err error
		guarded := receipt.AllowRemoteManagement == nil || receipt.PermissionMigrationID != ""
		if guarded {
			err = m.ensurePairingGuardLocked(receipt)
		}
		if err == nil {
			err = m.rollbackPairingPeerLocked(receipt)
		}
		if err == nil {
			receipt.Rollback = false
			receipt.State = inviteStateFailed
			receipt.ExpiresAt = time.Now().Add(m.effectiveInviteTTL()).UnixMilli()
			err = m.savePairingReceipt(receipt)
		}
		if err == nil && guarded {
			err = m.releasePairingGuardLocked(receipt)
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
