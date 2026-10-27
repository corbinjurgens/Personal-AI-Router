// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"nvpair-shared/clustertrust"
	"nvpair-shared/durablefile"
)

var errRemoteManagementConflict = errors.New("remote management revision conflict")

// remoteManagementWrite never treats a landed but unconfirmed replacement as
// success. In particular, it leaves landed denials in place on failure.
func remoteManagementWrite(path string, data []byte) error {
	err := clusterFileReplacer.WriteFile(path, data, 0o600)
	if !errors.Is(err, durablefile.ErrReplaced) {
		return err
	}
	landed, readErr := remoteManagementRead(path)
	if readErr != nil || !bytes.Equal(landed, data) {
		return fmt.Errorf("remote management replacement unconfirmed: %w", errors.Join(err, readErr))
	}
	if confirmErr := clusterFileReplacer.WriteFile(path, data, 0o600); confirmErr != nil {
		return fmt.Errorf("remote management durability unconfirmed: %w", errors.Join(err, confirmErr))
	}
	return nil
}

func remoteManagementRead(path string) ([]byte, error) {
	f, err := durablefile.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, clustertrust.RemoteManagementMaxBytes+1))
	if err != nil || len(data) > clustertrust.RemoteManagementMaxBytes {
		return nil, fmt.Errorf("remote management file unreadable or oversized: %w", err)
	}
	return data, nil
}

func (m *Manager) remoteManagementPaths() (string, string) {
	return filepath.Join(m.clusterDir, clustertrust.RemoteManagementFile), filepath.Join(m.clusterDir, clustertrust.RemoteManagementMigrationFile)
}

func (m *Manager) remoteManagementInitializingPath() string {
	return filepath.Join(m.clusterDir, clustertrust.RemoteManagementInitializingFile)
}

// The guard is a publication barrier, not authority. Its presence (including
// unreadable or malformed bytes) prevents every reader and writer from allowing.
func (m *Manager) remoteManagementGuard(id string) (bool, error) {
	if _, err := os.Lstat(m.remoteManagementInitializingPath()); os.IsNotExist(err) {
		return false, nil
	} else if err != nil {
		return true, err
	}
	data, err := remoteManagementRead(m.remoteManagementInitializingPath())
	if err != nil {
		return true, err
	}
	fields, err := decodeRemoteManagementParams(data, map[string]bool{"version": true, "id": true, "pairingInviteId": true})
	if err != nil || (len(fields) != 2 && len(fields) != 3) || string(fields["version"]) != "1" {
		return true, fmt.Errorf("invalid remote management initialization guard")
	}
	var storedID string
	if err := json.Unmarshal(fields["id"], &storedID); err != nil || storedID != id {
		return true, fmt.Errorf("conflicting remote management initialization guard")
	}
	if raw, ok := fields["pairingInviteId"]; ok {
		var inviteID string
		if err := json.Unmarshal(raw, &inviteID); err != nil || inviteID == "" {
			return true, fmt.Errorf("invalid pairing publication guard")
		}
	}
	return true, nil
}

func (m *Manager) pairingGuardOwner() (string, error) {
	data, err := remoteManagementRead(m.remoteManagementInitializingPath())
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	var guard struct {
		ID       string `json:"id"`
		InviteID string `json:"pairingInviteId"`
	}
	if err := json.Unmarshal(data, &guard); err != nil {
		return "", err
	}
	if _, err := m.remoteManagementGuard(guard.ID); err != nil {
		return "", err
	}
	return guard.InviteID, nil
}

// A legacy receipt has no permission choice or migration metadata. Its existing
// immutable marker supplies only abort correlation, never permission consent.
func (m *Manager) pairingGuardMigration(receipt pairingReceipt) (clustertrust.RemoteManagementMigration, error) {
	_, path := m.remoteManagementPaths()
	data, err := remoteManagementRead(path)
	if err != nil {
		return clustertrust.RemoteManagementMigration{}, err
	}
	marker, err := clustertrust.DecodeRemoteManagementMigration(data)
	if err != nil {
		return marker, err
	}
	if receipt.AllowRemoteManagement != nil && receipt.PermissionMigrationID != marker.ID {
		return marker, fmt.Errorf("pairing permission migration mismatch")
	}
	return marker, nil
}

func (m *Manager) ensurePairingGuardLocked(receipt pairingReceipt) error {
	marker, err := m.pairingGuardMigration(receipt)
	if err != nil {
		return err
	}
	guarded, err := m.remoteManagementGuard(marker.ID)
	if err != nil {
		return err
	}
	owner, err := m.pairingGuardOwner()
	// Startup may already hold the migration's negative fence. Transfer it to
	// the legacy abort without any interval in which authority is published.
	if err != nil || (guarded && owner != receipt.InviteID && !(receipt.AllowRemoteManagement == nil && owner == "")) {
		return fmt.Errorf("pairing publication guard belongs to another transaction: %w", err)
	}
	data, err := json.Marshal(struct {
		Version  int    `json:"version"`
		ID       string `json:"id"`
		InviteID string `json:"pairingInviteId"`
	}{Version: 1, ID: marker.ID, InviteID: receipt.InviteID})
	if err != nil {
		return err
	}
	return remoteManagementWrite(m.remoteManagementInitializingPath(), data)
}

func (m *Manager) releasePairingGuardLocked(receipt pairingReceipt) error {
	marker, err := m.pairingGuardMigration(receipt)
	if err != nil {
		return err
	}
	if guarded, err := m.remoteManagementGuard(marker.ID); err != nil || !guarded {
		return fmt.Errorf("pairing publication guard cannot be released: %w", err)
	}
	owner, err := m.pairingGuardOwner()
	if err != nil || owner != receipt.InviteID {
		return fmt.Errorf("pairing publication guard cannot be released: %w", err)
	}
	if receipt.AllowRemoteManagement == nil && marker.State == "pending" {
		// Recovery precedes migration finalization. Return its fence instead of
		// exposing a draft document or blocking the finalizer on a terminal abort.
		return m.ensureRemoteManagementGuard(marker.ID)
	}
	return removeFileWithRetry(m.remoteManagementInitializingPath())
}

func (m *Manager) ensureRemoteManagementGuard(id string) error {
	if _, err := m.remoteManagementGuard(id); err != nil {
		return err
	}
	data, err := json.Marshal(struct {
		Version int    `json:"version"`
		ID      string `json:"id"`
	}{Version: 1, ID: id})
	if err != nil {
		return err
	}
	return remoteManagementWrite(m.remoteManagementInitializingPath(), data)
}

func (m *Manager) remoteManagementStartupMustStop(err error) bool {
	if _, guardErr := os.Lstat(m.remoteManagementInitializingPath()); !os.IsNotExist(guardErr) {
		return true
	}
	if errors.Is(err, durablefile.ErrReplaced) {
		return true
	}
	_, path := m.remoteManagementPaths()
	data, readErr := remoteManagementRead(path)
	if readErr != nil {
		return true
	}
	marker, decodeErr := clustertrust.DecodeRemoteManagementMigration(data)
	return decodeErr != nil || marker.State == "pending"
}

func (m *Manager) readRemoteManagementLocked() (clustertrust.RemoteManagementDocument, clustertrust.RemoteManagementMigration, error) {
	return m.readRemoteManagementWithGuardLocked("")
}

// Only the composition-locked pairing transaction may operate behind its
// matching publication guard. Every public reader/setter remains unavailable.
func (m *Manager) readRemoteManagementWithGuardLocked(guardID string) (clustertrust.RemoteManagementDocument, clustertrust.RemoteManagementMigration, error) {
	docPath, markerPath := m.remoteManagementPaths()
	var doc clustertrust.RemoteManagementDocument
	var marker clustertrust.RemoteManagementMigration
	if guardID == "" {
		if _, err := os.Lstat(m.remoteManagementInitializingPath()); !os.IsNotExist(err) {
			return doc, marker, fmt.Errorf("remote management initialization unavailable")
		}
	} else if guarded, err := m.remoteManagementGuard(guardID); err != nil || !guarded {
		return doc, marker, fmt.Errorf("pairing publication guard unavailable: %w", err)
	}
	data, err := remoteManagementRead(markerPath)
	if err != nil {
		return doc, marker, err
	}
	marker, err = clustertrust.DecodeRemoteManagementMigration(data)
	if err != nil {
		return doc, marker, err
	}
	data, err = remoteManagementRead(docPath)
	if err != nil {
		return doc, marker, err
	}
	doc, err = clustertrust.DecodeRemoteManagementDocument(data)
	if err != nil || doc.MigrationID != marker.ID || (guardID != "" && marker.ID != guardID) {
		return doc, marker, fmt.Errorf("inconsistent remote management storage: %w", err)
	}
	return doc, marker, nil
}

// captureRemoteManagementMigrationLocked runs before startup recovery changes
// pins or admissions. A pending marker is immutable across every retry.
func (m *Manager) captureRemoteManagementMigrationLocked() error {
	docPath, markerPath := m.remoteManagementPaths()
	data, markerErr := remoteManagementRead(markerPath)
	if markerErr == nil {
		marker, err := clustertrust.DecodeRemoteManagementMigration(data)
		if err != nil {
			return err
		}
		if _, err := m.remoteManagementGuard(marker.ID); err != nil {
			return err
		}
		// A previous process may have stopped after an uncertain replacement.
		// Confirm the immutable bytes before any startup mutation producer runs.
		return remoteManagementWrite(markerPath, data)
	}
	if !os.IsNotExist(markerErr) {
		return markerErr
	}
	if _, err := os.Lstat(m.remoteManagementInitializingPath()); !os.IsNotExist(err) {
		return fmt.Errorf("remote management initialization guard exists without marker")
	}
	if _, err := remoteManagementRead(docPath); !os.IsNotExist(err) {
		return fmt.Errorf("remote management document exists without migration marker")
	}
	cid, epoch := m.currentAdmission()
	localFingerprint, err := certFingerprintFromPEM(m.identity.CertPEM)
	if err != nil {
		return fmt.Errorf("read local migration certificate: %w", err)
	}
	snapshot := clustertrust.RemoteManagementMigrationSnapshot{
		Local: clustertrust.RemoteManagementMigrationAuthority{PeerUUID: m.identity.NodeUUID, CertFingerprint: localFingerprint, ClusterID: cid, AdmissionEpoch: epoch},
		Peers: make([]clustertrust.RemoteManagementMigrationAuthority, 0),
	}
	for _, pin := range m.trust.List() {
		// Pin fingerprint metadata is optional. Capture the actual pinned leaf,
		// without modifying the existing trust document or its public schema.
		fingerprint, err := remoteManagementPinFingerprint(pin)
		if err != nil {
			return fmt.Errorf("read peer migration certificate: %w", err)
		}
		snapshot.Peers = append(snapshot.Peers, clustertrust.RemoteManagementMigrationAuthority{PeerUUID: pin.NodeUUID, CertFingerprint: fingerprint, ClusterID: pin.ClusterID, AdmissionEpoch: pin.AdmissionEpoch})
	}
	sort.Slice(snapshot.Peers, func(i, j int) bool { return snapshot.Peers[i].PeerUUID < snapshot.Peers[j].PeerUUID })
	id, err := clustertrust.RemoteManagementMigrationID(snapshot)
	if err != nil {
		return err
	}
	marker := clustertrust.RemoteManagementMigration{Version: 1, ID: id, State: "pending", Snapshot: snapshot}
	data, err = json.Marshal(marker)
	if err != nil {
		return err
	}
	if _, err := clustertrust.DecodeRemoteManagementMigration(data); err != nil {
		return err
	}
	return remoteManagementWrite(markerPath, data)
}

func remoteManagementAuthorityMatches(captured clustertrust.RemoteManagementMigrationAuthority, uuid, fingerprint, clusterID string, epoch uint64) bool {
	if captured.PeerUUID != uuid || captured.CertFingerprint != fingerprint {
		return false
	}
	if captured.ClusterID != "" && captured.ClusterID != clusterID {
		return false
	}
	if captured.AdmissionEpoch == 0 {
		return epoch == legacyAdmissionEpoch
	}
	return captured.AdmissionEpoch == epoch && captured.ClusterID == clusterID
}

func remoteManagementPinFingerprint(pin *TrustedPin) (string, error) {
	der, err := validatePin(pin, pin.NodeUUID)
	if err != nil {
		return "", err
	}
	return certFingerprintFromDER(der), nil
}

// finalizeRemoteManagementMigrationLocked requires rosterMu and completed
// legacy admission normalization. It never enumerates live pins for authority.
// Missing legacy local admission leaves the original marker pending until the
// identity-restore owner invokes this helper after normalization.
func (m *Manager) finalizeRemoteManagementMigrationLocked() error {
	m.mesh.Refresh()
	docPath, markerPath := m.remoteManagementPaths()
	data, err := remoteManagementRead(markerPath)
	if err != nil {
		return err
	}
	marker, err := clustertrust.DecodeRemoteManagementMigration(data)
	if err != nil {
		return err
	}
	if owner, err := m.pairingGuardOwner(); err != nil || owner != "" {
		return fmt.Errorf("pairing publication requires receipt compensation: %w", err)
	}
	if marker.State == "complete" {
		guarded, err := m.remoteManagementGuard(marker.ID)
		if err != nil {
			return err
		}
		if !guarded {
			_, _, err := m.readRemoteManagementLocked()
			return err
		}
		existing, err := remoteManagementRead(docPath)
		if err != nil {
			return err
		}
		doc, err := clustertrust.DecodeRemoteManagementDocument(existing)
		if err != nil || doc.MigrationID != marker.ID {
			return fmt.Errorf("inconsistent guarded remote management document")
		}
		// Preserve the current revision and all decisions. A guard never permits
		// reconstructing a missing document from the original snapshot.
		if err := remoteManagementWrite(docPath, existing); err != nil {
			return err
		}
		if err := remoteManagementWrite(markerPath, data); err != nil {
			return err
		}
		return removeFile(m.remoteManagementInitializingPath())
	}
	if err := m.ensureRemoteManagementGuard(marker.ID); err != nil {
		return err
	}
	cid, epoch := m.currentAdmission()
	if epoch == 0 && len(marker.Snapshot.Peers) != 0 && !m.admissionWasRetired() {
		return nil
	}
	if existing, readErr := remoteManagementRead(docPath); readErr == nil {
		doc, decodeErr := clustertrust.DecodeRemoteManagementDocument(existing)
		if decodeErr != nil || doc.MigrationID != marker.ID || doc.Revision != "1" {
			return fmt.Errorf("pending remote management migration has inconsistent document")
		}
		// Confirm the same document, never reconstruct grants after a crash.
		if err := remoteManagementWrite(docPath, existing); err != nil {
			return err
		}
	} else if !os.IsNotExist(readErr) {
		return readErr
	} else {
		doc := clustertrust.RemoteManagementDocument{Version: 1, Revision: "1", MigrationID: marker.ID, Grants: make([]clustertrust.RemoteManagementGrant, 0)}
		localFingerprint, err := certFingerprintFromPEM(m.identity.CertPEM)
		if err != nil {
			return fmt.Errorf("read local migration certificate: %w", err)
		}
		localMatches := !m.admissionWasRetired() && remoteManagementAuthorityMatches(marker.Snapshot.Local, m.identity.NodeUUID, localFingerprint, cid, epoch)
		if localMatches {
			for _, captured := range marker.Snapshot.Peers {
				pin, ok := m.trust.Get(captured.PeerUUID)
				if !ok {
					continue
				}
				fingerprint, err := remoteManagementPinFingerprint(pin)
				if err != nil || !remoteManagementAuthorityMatches(captured, pin.NodeUUID, fingerprint, pin.ClusterID, pin.AdmissionEpoch) {
					continue
				}
				incarnation, ok := m.peerServiceIncarnation(pin.NodeUUID)
				if ok {
					doc.Grants = append(doc.Grants, clustertrust.RemoteManagementGrant{PeerUUID: pin.NodeUUID, Incarnation: incarnation, Allowed: true, DecisionRevision: doc.Revision})
				}
			}
		}
		encoded, err := json.Marshal(doc)
		if err != nil {
			return err
		}
		if _, err := clustertrust.DecodeRemoteManagementDocument(encoded); err != nil {
			return err
		}
		if err := remoteManagementWrite(docPath, encoded); err != nil {
			return err
		}
	}
	marker.State = "complete"
	data, err = json.Marshal(marker)
	if err != nil {
		return err
	}
	if err := remoteManagementWrite(markerPath, data); err != nil {
		return err
	}
	// Both authority files are now confirmed. Visible unlink safely publishes
	// them; crash resurrection of this guard can only deny until reconfirmation.
	return removeFile(m.remoteManagementInitializingPath())
}

type remoteManagementViewGrant struct {
	PeerUUID string `json:"peerUuid"`
	Allowed  bool   `json:"allowed"`
	State    string `json:"state"`
}

type remoteManagementSnapshot struct {
	Version  int                         `json:"version"`
	Revision string                      `json:"revision"`
	Grants   []remoteManagementViewGrant `json:"grants"`
}

func (m *Manager) remoteManagementSnapshotLocked() remoteManagementSnapshot {
	view := remoteManagementSnapshot{Version: 1, Grants: make([]remoteManagementViewGrant, 0)}
	doc, marker, err := m.readRemoteManagementLocked()
	if err == nil && marker.State == "complete" {
		view.Revision = doc.Revision
	}
	for _, member := range m.snapshotNodes() {
		uuid := member.NodeUUID
		if member.State != stateMember || uuid == m.identity.NodeUUID {
			continue
		}
		if _, ok := m.trust.Get(uuid); !ok {
			continue
		}
		// RemoteManagement also reads the Mesh incarnation. snapshotNodes has
		// released memMu before storage ownership, matching the writer order.
		m.membersStoreMu.Lock()
		permission := m.mesh.RemoteManagement(uuid)
		m.membersStoreMu.Unlock()
		view.Grants = append(view.Grants, remoteManagementViewGrant{PeerUUID: uuid, Allowed: permission.Allowed, State: permission.State})
	}
	sort.Slice(view.Grants, func(i, j int) bool { return view.Grants[i].PeerUUID < view.Grants[j].PeerUUID })
	return view
}

// The local setter and guarded pairing commit share one CAS grant writer.
// Pairing retains its expected revision before pin/member/admission mutation;
// no receipt replay can overwrite a later local decision. Caller holds rosterMu.
func (m *Manager) setRemoteManagementLocked(peerUUID string, allowed bool, expectedRevision string) error {
	return m.setRemoteManagementWithGuardLocked(peerUUID, allowed, expectedRevision, "")
}

func (m *Manager) setRemoteManagementWithGuardLocked(peerUUID string, allowed bool, expectedRevision, guardID string) error {
	doc, marker, err := m.readRemoteManagementWithGuardLocked(guardID)
	if err != nil || marker.State != "complete" {
		return fmt.Errorf("remote management unavailable")
	}
	if expectedRevision != doc.Revision {
		return errRemoteManagementConflict
	}
	revision, err := strconv.ParseUint(doc.Revision, 10, 64)
	if err != nil || revision == math.MaxUint64 {
		return fmt.Errorf("remote management revision exhausted")
	}
	incarnation, ok := m.peerServiceIncarnation(peerUUID)
	if !ok || peerUUID == m.identity.NodeUUID || m.teardownPending.Load() {
		return fmt.Errorf("peer has no current admission")
	}
	grants := make([]clustertrust.RemoteManagementGrant, 0, len(doc.Grants)+1)
	oldState := "unreviewed"
	for _, grant := range doc.Grants {
		if grant.PeerUUID == peerUUID && grant.Incarnation == incarnation {
			oldState = strconv.FormatBool(grant.Allowed)
		}
		current, ok := m.peerServiceIncarnation(grant.PeerUUID)
		if ok && current == grant.Incarnation && grant.PeerUUID != peerUUID {
			grants = append(grants, grant)
		}
	}
	nextRevision := strconv.FormatUint(revision+1, 10)
	grants = append(grants, clustertrust.RemoteManagementGrant{PeerUUID: peerUUID, Incarnation: incarnation, Allowed: allowed, DecisionRevision: nextRevision})
	sort.Slice(grants, func(i, j int) bool { return grants[i].PeerUUID < grants[j].PeerUUID })
	doc.Grants, doc.Revision = grants, nextRevision
	data, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	if _, err := clustertrust.DecodeRemoteManagementDocument(data); err != nil {
		return err
	}
	path, _ := m.remoteManagementPaths()
	var writeErr error
	// A local allow must remain unpublished until its document is confirmed.
	// Pairing already owns a matching guard and releases it with its receipt;
	// never finalize or remove that transaction's guard here. Denials retain
	// their immediate landed-byte semantics without adding a publication gap.
	guardLocalAllow := allowed && guardID == ""
	if guardLocalAllow {
		writeErr = m.ensureRemoteManagementGuard(doc.MigrationID)
	}
	if writeErr == nil {
		writeErr = remoteManagementWrite(path, data)
		if writeErr != nil && guardLocalAllow && !errors.Is(writeErr, durablefile.ErrReplaced) {
			// The document never landed. Only our confirmed local guard was
			// added, so remove it to restore the prior authoritative decision.
			// An uncertain guard preparation never reaches this branch, and a
			// pairing-owned guard must remain with its receipt owner.
			if owner, ownerErr := m.pairingGuardOwner(); ownerErr != nil {
				writeErr = errors.Join(writeErr, ownerErr)
			} else if owner != "" {
				writeErr = errors.Join(writeErr, errors.New("pairing owns publication guard"))
			} else if guarded, guardErr := m.remoteManagementGuard(doc.MigrationID); guardErr != nil {
				writeErr = errors.Join(writeErr, guardErr)
			} else if !guarded {
				writeErr = errors.Join(writeErr, errors.New("local publication guard missing"))
			} else if cleanupErr := removeFile(m.remoteManagementInitializingPath()); cleanupErr != nil {
				writeErr = errors.Join(writeErr, cleanupErr)
			}
		}
	}
	if writeErr == nil && guardLocalAllow {
		// This same owner also recovers a crash or uncertain local allow on
		// startup: confirm existing bytes, never reconstruct or reset choices.
		writeErr = m.finalizeRemoteManagementMigrationLocked()
	}
	outcome := "committed"
	if writeErr != nil {
		outcome = "failed"
		if errors.Is(writeErr, durablefile.ErrReplaced) {
			outcome = "uncertain"
		}
	}
	log.Printf("remote management actor=local peer=%s attemptRevision=%s previousRevision=%s incarnation=%s old=%s new=%t outcome=%s", peerUUID, nextRevision, expectedRevision, incarnation, oldState, allowed, outcome)
	return writeErr
}

// Compensation removes only this transaction's grant or its original migrated
// admission's grandfathered decision, proven by the immutable snapshot.
// Each entry records its own decision revision: an unrelated peer edit cannot
// hide this transaction, while even a same-value later local choice supersedes it.
func (m *Manager) removePairingRemoteManagementLocked(receipt pairingReceipt, pi *PairingInfo, fingerprint string) error {
	legacy := receipt.AllowRemoteManagement == nil
	if !legacy && (receipt.PermissionRevision == "" || receipt.PermissionIncarnation == "") {
		return nil
	}
	marker, err := m.pairingGuardMigration(receipt)
	if err != nil {
		return err
	}
	doc, marker, err := m.readRemoteManagementWithGuardLocked(marker.ID)
	if legacy && marker.State == "pending" && os.IsNotExist(err) {
		// No draft has ever granted authority. Finalization will exclude the
		// aborted pin; the migration fence remains until that work completes.
		return nil
	}
	if err != nil || (!legacy && marker.State != "complete") {
		return fmt.Errorf("remote management unavailable for compensation: %w", err)
	}
	if marker.State == "pending" && doc.Revision != "1" {
		return fmt.Errorf("pending remote management migration has inconsistent document")
	}
	decisionRevision, incarnation, allowed := "1", receipt.PermissionIncarnation, true
	legacyOwned := false
	if legacy {
		// Revision 1 came only from the original immutable migration snapshot.
		// This proof survives pin/member deletion and a failed journal write.
		if remoteManagementAuthorityMatches(marker.Snapshot.Local, m.identity.NodeUUID, m.identity.CertFingerprint, receipt.ClusterID, receipt.LocalEpoch) {
			for _, captured := range marker.Snapshot.Peers {
				if remoteManagementAuthorityMatches(captured, pi.NodeUUID, fingerprint, receipt.ClusterID, pi.AdmissionEpoch) {
					legacyOwned = true
					break
				}
			}
		}
	} else {
		expected, err := strconv.ParseUint(receipt.PermissionRevision, 10, 64)
		if err != nil || expected == math.MaxUint64 {
			return fmt.Errorf("invalid pairing permission revision")
		}
		decisionRevision, allowed = strconv.FormatUint(expected+1, 10), *receipt.AllowRemoteManagement
	}
	removed := false
	grants := make([]clustertrust.RemoteManagementGrant, 0, len(doc.Grants))
	for _, grant := range doc.Grants {
		if grant.PeerUUID == pi.NodeUUID && grant.Allowed == allowed && grant.DecisionRevision == decisionRevision &&
			((legacy && legacyOwned) || (!legacy && grant.Incarnation == incarnation)) {
			removed = true
			continue
		}
		grants = append(grants, grant)
	}
	if removed {
		doc.Grants = grants
		if marker.State == "complete" {
			revision, err := strconv.ParseUint(doc.Revision, 10, 64)
			if err != nil || revision == math.MaxUint64 {
				return fmt.Errorf("remote management revision exhausted")
			}
			doc.Revision = strconv.FormatUint(revision+1, 10)
		}
		// A pending migration document is an unpublished revision-1 draft.
		// Preserve that revision so the existing finalizer can confirm it.
	}
	// Also confirm an already-absent or superseded decision before releasing
	// the publication guard: a prior removal replacement may have been uncertain.
	data, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	if _, err := clustertrust.DecodeRemoteManagementDocument(data); err != nil {
		return err
	}
	path, _ := m.remoteManagementPaths()
	return remoteManagementWrite(path, data)
}

// Strict local RPC decoding rejects duplicate fields as well as unknown keys.
func decodeRemoteManagementParams(data []byte, allowed map[string]bool) (map[string]json.RawMessage, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return nil, fmt.Errorf("expected params object")
	}
	fields := make(map[string]json.RawMessage)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		key, ok := token.(string)
		if !ok || !allowed[key] {
			return nil, fmt.Errorf("unknown params field")
		}
		if _, exists := fields[key]; exists {
			return nil, fmt.Errorf("duplicate params field")
		}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, err
		}
		fields[key] = value
	}
	if _, err := decoder.Token(); err != nil {
		return nil, err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing params")
	}
	return fields, nil
}

func (m *Manager) handleGetRemoteManagement(msg *Message) {
	if _, err := decodeRemoteManagementParams(msg.Params, map[string]bool{}); err != nil {
		m.codec.RespondError(msg.ID, codeInvalidParams, "invalid params")
		return
	}
	m.rosterMu.Lock()
	view := m.remoteManagementSnapshotLocked()
	m.rosterMu.Unlock()
	m.codec.Respond(msg.ID, view)
}

func (m *Manager) handleSetRemoteManagement(msg *Message) {
	fields, err := decodeRemoteManagementParams(msg.Params, map[string]bool{"peerUuid": true, "allowed": true, "expectedRevision": true})
	var peerUUID, revision string
	var allowed bool
	if err != nil || len(fields) != 3 || json.Unmarshal(fields["peerUuid"], &peerUUID) != nil || peerUUID == "" || json.Unmarshal(fields["expectedRevision"], &revision) != nil || revision == "" || (string(fields["allowed"]) != "true" && string(fields["allowed"]) != "false") || json.Unmarshal(fields["allowed"], &allowed) != nil {
		m.codec.RespondError(msg.ID, codeInvalidParams, "invalid params")
		return
	}
	// Reuse the shared schema for principal and canonical revision validation.
	validation, marshalErr := json.Marshal(clustertrust.RemoteManagementDocument{
		Version: 1, Revision: revision, MigrationID: strings.Repeat("0", 64),
		Grants: []clustertrust.RemoteManagementGrant{{PeerUUID: peerUUID, Incarnation: strings.Repeat("0", 64), Allowed: allowed, DecisionRevision: revision}},
	})
	if marshalErr != nil {
		m.codec.RespondError(msg.ID, codeInvalidParams, "invalid params")
		return
	}
	if _, err := clustertrust.DecodeRemoteManagementDocument(validation); err != nil {
		m.codec.RespondError(msg.ID, codeInvalidParams, "invalid params")
		return
	}
	m.rosterMu.Lock()
	err = m.setRemoteManagementLocked(peerUUID, allowed, revision)
	view := m.remoteManagementSnapshotLocked()
	m.rosterMu.Unlock()
	if err != nil {
		code := codeInternalError
		if errors.Is(err, errRemoteManagementConflict) {
			code = -32010
		}
		m.codec.RespondError(msg.ID, code, "remote management update failed")
		if errors.Is(err, durablefile.ErrReplaced) {
			if notifyErr := m.codec.Notify("cluster:remote-management-changed", view); notifyErr != nil {
				log.Printf("emit remote management change failed")
			}
		}
		return
	}
	m.codec.Respond(msg.ID, view)
	if err := m.codec.Notify("cluster:remote-management-changed", view); err != nil {
		log.Printf("emit remote management change failed")
	}
}
