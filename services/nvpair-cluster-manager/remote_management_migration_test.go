// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"nvpair-shared/clustertrust"
	"nvpair-shared/durablefile"
)

func remoteManagementUpgradeFixture(t *testing.T) *Manager {
	t.Helper()
	m := testManagerAt(t, t.TempDir(), 15129)
	activateTestCluster(t, m, "management-test")
	doc, marker := m.remoteManagementPaths()
	// Model a pre-PG profile. Product code never deletes either file.
	for _, path := range []string{doc, marker} {
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
	}
	return m
}

func remoteManagementAddPeer(t *testing.T, m *Manager) *Manager {
	t.Helper()
	peer := testManagerAt(t, t.TempDir(), 15130)
	if err := m.trust.Pin(&TrustedPin{
		NodeUUID: peer.identity.NodeUUID, NodeID: peer.identity.NodeID,
		Name: peer.identity.Name, CertPem: string(peer.identity.CertPEM),
		CertFingerprint: peer.identity.CertFingerprint, ClusterID: "management-test", AdmissionEpoch: 1,
	}); err != nil {
		t.Fatal(err)
	}
	m.upsertMember(&ClusterNode{NodeUUID: peer.identity.NodeUUID, ID: peer.identity.NodeID, ClusterID: "management-test", AdmissionEpoch: 1, State: stateMember})
	if err := m.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	m.mesh.Refresh()
	return peer
}

func TestRemoteManagementMigrationNeverSnapshotsLatePeer(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	first := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	_, markerPath := m.remoteManagementPaths()
	before, err := os.ReadFile(markerPath)
	if err != nil {
		t.Fatal(err)
	}
	late := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if got := m.mesh.RemoteManagement(first.identity.NodeUUID); !got.Allowed {
		t.Fatalf("original peer = %+v, want allowed", got)
	}
	if got := m.mesh.RemoteManagement(late.identity.NodeUUID); got.Allowed || got.State != "unreviewed" {
		t.Fatalf("late peer = %+v, want unreviewed denial", got)
	}
	doc, marker, err := m.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	if doc.Revision != "1" || len(doc.Grants) != 1 || len(marker.Snapshot.Peers) != 1 || marker.Snapshot.Peers[0].PeerUUID != first.identity.NodeUUID {
		t.Fatalf("migration changed original set: doc=%+v marker=%+v original=%s", doc, marker, before)
	}
}

func TestRemoteManagementCompletedMigrationCannotRepairLostDocument(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	docPath, markerPath := m.remoteManagementPaths()
	before, err := os.ReadFile(markerPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(docPath); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err == nil {
		t.Fatal("lost document silently repaired")
	}
	if got := m.mesh.RemoteManagement(peer.identity.NodeUUID); got.Allowed || got.State != "unavailable" {
		t.Fatalf("lost document permission = %+v", got)
	}
	after, err := os.ReadFile(markerPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("completed marker changed")
	}
}

func TestRemoteManagementMigrationPreservesLaterDenyAndCAS(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.setRemoteManagementLocked(peer.identity.NodeUUID, false, "1"); err != nil {
		t.Fatal(err)
	}
	if err := m.setRemoteManagementLocked(peer.identity.NodeUUID, true, "1"); !errors.Is(err, errRemoteManagementConflict) {
		t.Fatalf("stale write error = %v", err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if got := m.mesh.RemoteManagement(peer.identity.NodeUUID); got.Allowed || got.State != "denied" || got.Revision != "2" {
		t.Fatalf("later denial overwritten: %+v", got)
	}
}

func TestRemoteManagementPendingDocumentRetryUsesOriginalBytes(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	docPath, markerPath := m.remoteManagementPaths()
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
		if target == markerPath {
			return errors.New("marker completion fault")
		}
		return os.Rename(tmp, target)
	}}
	if err := m.finalizeRemoteManagementMigrationLocked(); err == nil {
		t.Fatal("marker completion failure ignored")
	}
	before := mustRemoteManagementRead(t, docPath)
	clusterFileReplacer = original
	late := remoteManagementAddPeer(t, m)
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	after := mustRemoteManagementRead(t, docPath)
	if !bytes.Equal(before, after) {
		t.Fatal("pending document reconstructed")
	}
	if !m.mesh.RemoteManagement(peer.identity.NodeUUID).Allowed || m.mesh.RemoteManagement(late.identity.NodeUUID).Allowed {
		t.Fatal("retry widened original authority")
	}
}

func mustRemoteManagementRead(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestRemoteManagementMigrationRejectsReadmittedOriginalPeer(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	pin, ok := m.trust.Get(peer.identity.NodeUUID)
	if !ok {
		t.Fatal("fixture peer missing")
	}
	pin.AdmissionEpoch = 2
	if err := m.trust.Pin(pin); err != nil {
		t.Fatal(err)
	}
	m.upsertMember(&ClusterNode{NodeUUID: peer.identity.NodeUUID, ID: peer.identity.NodeID, ClusterID: "management-test", AdmissionEpoch: 2, State: stateMember})
	if err := m.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if got := m.mesh.RemoteManagement(peer.identity.NodeUUID); got.Allowed || got.State != "unreviewed" {
		t.Fatalf("readmitted peer grandfathered: %+v", got)
	}
}

func TestRemoteManagementTeardownRetainsCompletedMarker(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	_, markerPath := m.remoteManagementPaths()
	before := mustRemoteManagementRead(t, markerPath)
	m.teardownClusterLocal()
	after := mustRemoteManagementRead(t, markerPath)
	if !bytes.Equal(before, after) {
		t.Fatal("teardown deleted or changed migration marker")
	}
	if got := m.mesh.RemoteManagement(peer.identity.NodeUUID); got.Allowed {
		t.Fatalf("teardown retained effective grant: %+v", got)
	}
}

func TestRemoteManagementEmptyProfileGetsCompletedMarker(t *testing.T) {
	m := testManagerAt(t, t.TempDir(), 15131)
	doc, marker, err := m.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	if marker.State != "complete" || len(marker.Snapshot.Peers) != 0 || len(doc.Grants) != 0 || doc.Revision != "1" {
		t.Fatalf("empty profile migration = %+v %+v", marker, doc)
	}
}

func TestRemoteManagementMissingLegacyAdmissionStaysPending(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	remoteManagementAddPeer(t, m)
	if err := os.Remove(m.admissionPath()); err != nil {
		t.Fatal(err)
	}
	m.admissionClusterID, m.admissionEpoch = "", 0
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	docPath, _ := m.remoteManagementPaths()
	if _, err := os.Stat(docPath); !os.IsNotExist(err) {
		t.Fatalf("legacy migration published before identity restore: %v", err)
	}
}

func TestRemoteManagementLegacyIdentityRestoreFinalizesOriginalSnapshot(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := os.Remove(m.admissionPath()); err != nil {
		t.Fatal(err)
	}
	m.admissionClusterID, m.admissionEpoch = "", 0
	m.admissionCounter, m.admissionActivated = 0, 0
	m.setClusterIdentity("", "")
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	late := remoteManagementAddPeer(t, m)
	id := json.RawMessage(`1`)
	m.handleSetIdentity(&Message{ID: &id, Params: json.RawMessage(`{"clusterId":"management-test"}`)})
	doc, marker, err := m.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	if marker.State != "complete" || doc.Revision != "1" || len(doc.Grants) != 1 || doc.Grants[0].PeerUUID != peer.identity.NodeUUID {
		t.Fatalf("legacy restore did not retain original authority: %+v %+v", doc, marker)
	}
	if got := m.mesh.RemoteManagement(peer.identity.NodeUUID); !got.Allowed {
		t.Fatalf("original peer denied: %+v", got)
	}
	if got := m.mesh.RemoteManagement(late.identity.NodeUUID); got.Allowed {
		t.Fatalf("late peer grandfathered: %+v", got)
	}
}

func TestRemoteManagementUncertainCompletionDeniesIndependentMesh(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	_, markerPath := m.remoteManagementPaths()
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	markerReplacements := 0
	clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
		if err := os.Rename(tmp, target); err != nil {
			return err
		}
		if target == markerPath {
			markerReplacements++
			return durablefile.ErrReplaced
		}
		return nil
	}}
	if err := m.finalizeRemoteManagementMigrationLocked(); !errors.Is(err, durablefile.ErrReplaced) {
		t.Fatalf("completion error = %v, want durability uncertainty", err)
	}
	if markerReplacements != 2 {
		t.Fatalf("marker replacement attempts = %d, want 2", markerReplacements)
	}
	independent := clustertrust.Open(m.clusterDir)
	if permission := independent.RemoteManagement(peer.identity.NodeUUID); permission.Allowed {
		t.Fatalf("independent reader allowed uncertain migration completion: %+v", permission)
	}
}

func TestRemoteManagementGuardWriteFailurePublishesNoDocument(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	guardPath := m.remoteManagementInitializingPath()
	clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
		if target == guardPath {
			return errors.New("guard write fault")
		}
		return os.Rename(tmp, target)
	}}
	if err := m.finalizeRemoteManagementMigrationLocked(); err == nil {
		t.Fatal("guard failure ignored")
	}
	docPath, _ := m.remoteManagementPaths()
	if _, err := os.Stat(docPath); !os.IsNotExist(err) {
		t.Fatalf("document published before guard: %v", err)
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer.identity.NodeUUID); got.Allowed {
		t.Fatalf("guard failure allowed peer: %+v", got)
	}
}

func TestRemoteManagementCompleteGuardRecoveryPreservesLaterDeny(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.setRemoteManagementLocked(peer.identity.NodeUUID, false, "1"); err != nil {
		t.Fatal(err)
	}
	docPath, markerPath := m.remoteManagementPaths()
	beforeDoc := mustRemoteManagementRead(t, docPath)
	beforeMarker := mustRemoteManagementRead(t, markerPath)
	doc, _, err := m.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	if err := m.ensureRemoteManagementGuard(doc.MigrationID); err != nil {
		t.Fatal(err)
	}
	if err := m.setRemoteManagementLocked(peer.identity.NodeUUID, true, "2"); err == nil {
		t.Fatal("setter crossed initialization guard")
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer.identity.NodeUUID); got.Allowed || got.State != "unavailable" {
		t.Fatalf("guarded permission = %+v", got)
	}
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(m.remoteManagementInitializingPath()); !os.IsNotExist(err) {
		t.Fatalf("confirmed guard remains: %v", err)
	}
	if !bytes.Equal(beforeDoc, mustRemoteManagementRead(t, docPath)) || !bytes.Equal(beforeMarker, mustRemoteManagementRead(t, markerPath)) {
		t.Fatal("recovery reconstructed authority")
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer.identity.NodeUUID); got.Allowed || got.State != "denied" || got.Revision != "2" {
		t.Fatalf("recovery lost later deny: %+v", got)
	}
}

func TestRemoteManagementCorruptGuardDeniesWithoutRepair(t *testing.T) {
	for _, guard := range []string{`{"version":1,"id":"wrong"}`, `{"version":1,"version":1,"id":"wrong"}`, `null`, `{"version":1}`} {
		t.Run(guard, func(t *testing.T) {
			m := remoteManagementUpgradeFixture(t)
			peer := remoteManagementAddPeer(t, m)
			if err := m.captureRemoteManagementMigrationLocked(); err != nil {
				t.Fatal(err)
			}
			if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(m.remoteManagementInitializingPath(), []byte(guard), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := m.finalizeRemoteManagementMigrationLocked(); err == nil {
				t.Fatal("corrupt guard repaired")
			}
			if err := m.setRemoteManagementLocked(peer.identity.NodeUUID, true, "1"); err == nil {
				t.Fatal("setter ignored corrupt guard")
			}
			if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer.identity.NodeUUID); got.Allowed {
				t.Fatalf("corrupt guard allowed peer: %+v", got)
			}
			if string(mustRemoteManagementRead(t, m.remoteManagementInitializingPath())) != guard {
				t.Fatal("corrupt guard changed")
			}
		})
	}
}

func TestRemoteManagementGuardRemovalFailureRetainsDenial(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	original := removeFile
	t.Cleanup(func() { removeFile = original })
	removeFile = func(path string) error {
		if path == m.remoteManagementInitializingPath() {
			return errors.New("guard unlink fault")
		}
		return original(path)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err == nil {
		t.Fatal("guard unlink failure ignored")
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer.identity.NodeUUID); got.Allowed {
		t.Fatalf("failed unlink allowed peer: %+v", got)
	}
	removeFile = original
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer.identity.NodeUUID); !got.Allowed {
		t.Fatalf("confirmed recovery denied original peer: %+v", got)
	}
}

func TestRemoteManagementUnconfirmedGuardDoesNotPublishDocument(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
		if err := os.Rename(tmp, target); err != nil {
			return err
		}
		if target == m.remoteManagementInitializingPath() {
			return durablefile.ErrReplaced
		}
		return nil
	}}
	if err := m.finalizeRemoteManagementMigrationLocked(); !errors.Is(err, durablefile.ErrReplaced) {
		t.Fatalf("guard uncertainty = %v", err)
	}
	docPath, _ := m.remoteManagementPaths()
	if _, err := os.Stat(docPath); !os.IsNotExist(err) {
		t.Fatalf("document published before durable guard: %v", err)
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer.identity.NodeUUID); got.Allowed {
		t.Fatalf("uncertain guard allowed peer: %+v", got)
	}
}

func TestRemoteManagementCompleteGuardMissingDocumentCannotResnapshot(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	doc, _, err := m.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	if err := m.ensureRemoteManagementGuard(doc.MigrationID); err != nil {
		t.Fatal(err)
	}
	docPath, markerPath := m.remoteManagementPaths()
	before := mustRemoteManagementRead(t, markerPath)
	if err := os.Remove(docPath); err != nil {
		t.Fatal(err)
	}
	late := remoteManagementAddPeer(t, m)
	if err := m.finalizeRemoteManagementMigrationLocked(); err == nil {
		t.Fatal("guard recovery reconstructed lost document")
	}
	if _, err := os.Stat(docPath); !os.IsNotExist(err) {
		t.Fatalf("missing document recreated: %v", err)
	}
	if !bytes.Equal(before, mustRemoteManagementRead(t, markerPath)) {
		t.Fatal("snapshot changed after storage loss")
	}
	if _, err := os.Lstat(m.remoteManagementInitializingPath()); err != nil {
		t.Fatalf("guard lost after storage loss: %v", err)
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(late.identity.NodeUUID); got.Allowed {
		t.Fatalf("late peer gained authority: %+v", got)
	}
}

func TestRemoteManagementInitialMarkerFailureStopsBeforeStartupNormalization(t *testing.T) {
	for _, landed := range []bool{false, true} {
		name := "before replacement"
		if landed {
			name = "repeated landed uncertainty"
		}
		t.Run(name, func(t *testing.T) {
			m := remoteManagementUpgradeFixture(t)
			peer := remoteManagementAddPeer(t, m)
			pin, ok := m.trust.Get(peer.identity.NodeUUID)
			if !ok {
				t.Fatal("fixture pin missing")
			}
			pin.ClusterID, pin.AdmissionEpoch = "", 0
			legacyPin, err := json.Marshal(pin)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(m.trust.pinPath(pin.NodeUUID), legacyPin, 0o600); err != nil {
				t.Fatal(err)
			}
			docPath, markerPath := m.remoteManagementPaths()
			original := clusterFileReplacer
			t.Cleanup(func() { clusterFileReplacer = original })
			attempts := 0
			clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
				if target != markerPath {
					return os.Rename(tmp, target)
				}
				attempts++
				if !landed {
					return errors.New("initial marker replace fault")
				}
				if err := os.Rename(tmp, target); err != nil {
					return err
				}
				return durablefile.ErrReplaced
			}}
			reopened, startupErr := NewManager(m.codec, filepath.Dir(m.clusterDir), m.port)
			if startupErr == nil || reopened != nil {
				t.Fatalf("startup escaped failed snapshot: manager=%v error=%v", reopened, startupErr)
			}
			wantAttempts := 1
			if landed {
				wantAttempts = 2
			}
			if attempts != wantAttempts {
				t.Fatalf("initial marker attempts = %d, want %d", attempts, wantAttempts)
			}
			if landed && !errors.Is(startupErr, durablefile.ErrReplaced) {
				t.Fatalf("startup lost uncertainty: %v", startupErr)
			}
			if !bytes.Equal(legacyPin, mustRemoteManagementRead(t, m.trust.pinPath(pin.NodeUUID))) {
				t.Fatal("startup normalized pin before snapshot confirmation")
			}
			for _, path := range []string{docPath, m.remoteManagementInitializingPath()} {
				if _, err := os.Stat(path); !os.IsNotExist(err) {
					t.Fatalf("startup published %s before snapshot confirmation: %v", path, err)
				}
			}
			if !landed {
				if _, err := os.Stat(markerPath); !os.IsNotExist(err) {
					t.Fatalf("failed initial replace left marker: %v", err)
				}
				return
			}
			pending, err := clustertrust.DecodeRemoteManagementMigration(mustRemoteManagementRead(t, markerPath))
			if err != nil {
				t.Fatal(err)
			}
			if pending.State != "pending" || len(pending.Snapshot.Peers) != 1 || pending.Snapshot.Peers[0].PeerUUID != peer.identity.NodeUUID {
				t.Fatalf("landed original snapshot = %+v", pending)
			}
			before, err := json.Marshal(pending.Snapshot)
			if err != nil {
				t.Fatal(err)
			}
			clusterFileReplacer = original
			late := remoteManagementAddPeer(t, m)
			reopened, err = NewManager(m.codec, filepath.Dir(m.clusterDir), m.port)
			if err != nil {
				t.Fatal(err)
			}
			_, completed, err := reopened.readRemoteManagementLocked()
			if err != nil {
				t.Fatal(err)
			}
			after, err := json.Marshal(completed.Snapshot)
			if err != nil {
				t.Fatal(err)
			}
			if completed.State != "complete" || completed.ID != pending.ID || !bytes.Equal(before, after) {
				t.Fatalf("retry replaced original snapshot: %+v", completed)
			}
			if got := reopened.mesh.RemoteManagement(peer.identity.NodeUUID); !got.Allowed {
				t.Fatalf("original legacy peer not grandfathered: %+v", got)
			}
			if got := reopened.mesh.RemoteManagement(late.identity.NodeUUID); got.Allowed {
				t.Fatalf("retry grandfathered late peer: %+v", got)
			}
		})
	}
}

func TestRemoteManagementLeaveCreateReadmissionRetainsMarkerAndDeniesOldGrant(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	m.addSelfMember()
	if err := m.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	beforePermission := m.mesh.RemoteManagement(peer.identity.NodeUUID)
	if !beforePermission.Allowed {
		t.Fatalf("initial permission = %+v", beforePermission)
	}
	_, oldEpoch := m.currentAdmission()
	docPath, markerPath := m.remoteManagementPaths()
	beforeMarker := mustRemoteManagementRead(t, markerPath)
	beforeDoc := mustRemoteManagementRead(t, docPath)
	pin, ok := m.trust.Get(peer.identity.NodeUUID)
	if !ok {
		t.Fatal("fixture pin missing")
	}
	for _, node := range m.snapshotNodes() {
		if node.NodeUUID != m.identity.NodeUUID && len(m.resolvePeerAddrs(node)) != 0 {
			t.Fatal("leave fixture must have no dialable peer addresses")
		}
	}
	left, err := m.leaveCluster()
	if err != nil || !left {
		t.Fatalf("leave = %t, %v", left, err)
	}
	if !bytes.Equal(beforeMarker, mustRemoteManagementRead(t, markerPath)) {
		t.Fatal("leave changed completed marker")
	}
	newID, _, err := m.foundCluster("After leave")
	if err != nil {
		t.Fatal(err)
	}
	_, newEpoch := m.currentAdmission()
	if newEpoch <= oldEpoch {
		t.Fatalf("create admission epoch = %d, want greater than %d", newEpoch, oldEpoch)
	}
	// Re-admit exactly the old peer certificate through the durable trust and
	// membership owners; no explicit permission choice is made in this fixture.
	pin.ClusterID = newID
	if err := m.trust.Pin(pin); err != nil {
		t.Fatal(err)
	}
	m.upsertMember(&ClusterNode{NodeUUID: pin.NodeUUID, ID: pin.NodeID, ClusterID: newID, AdmissionEpoch: pin.AdmissionEpoch, State: stateMember})
	if err := m.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	m.mesh.Refresh()
	current, ok := m.mesh.PeerServiceIncarnation(pin.NodeUUID)
	if !ok || current == beforePermission.Incarnation {
		t.Fatalf("readmission did not produce fresh authority: %q valid=%t", current, ok)
	}
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(beforeMarker, mustRemoteManagementRead(t, markerPath)) || !bytes.Equal(beforeDoc, mustRemoteManagementRead(t, docPath)) {
		t.Fatal("leave/create/readmission reran migration")
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(pin.NodeUUID); got.Allowed || got.State != "unreviewed" {
		t.Fatalf("old local-admission grant survived leave/create: %+v", got)
	}
}

func TestRemoteManagementMigrationUsesCertificateDespiteOptionalFingerprintMetadata(t *testing.T) {
	for _, metadata := range []string{"", "sha256:" + strings.Repeat("0", 64)} {
		name := "omitted fingerprint"
		if metadata != "" {
			name = "incorrect canonical fingerprint"
		}
		t.Run(name, func(t *testing.T) {
			m := remoteManagementUpgradeFixture(t)
			peer := remoteManagementAddPeer(t, m)
			pin, ok := m.trust.Get(peer.identity.NodeUUID)
			if !ok {
				t.Fatal("fixture pin missing")
			}
			pin.CertFingerprint = metadata
			if err := m.trust.Remove(pin.NodeUUID); err != nil {
				t.Fatal(err)
			}
			if err := m.trust.Pin(pin); err != nil {
				t.Fatal(err)
			}
			// Prepare existing admission endorsements through their own owner so
			// startup does not need an unrelated trust-document normalization.
			cid, epoch := m.currentAdmission()
			if err := m.migrateLegacyAdmissions(cid, epoch); err != nil {
				t.Fatal(err)
			}
			beforePin := mustRemoteManagementRead(t, m.trust.pinPath(pin.NodeUUID))
			if err := m.captureRemoteManagementMigrationLocked(); err != nil {
				t.Fatal(err)
			}
			_, markerPath := m.remoteManagementPaths()
			pending, err := clustertrust.DecodeRemoteManagementMigration(mustRemoteManagementRead(t, markerPath))
			if err != nil {
				t.Fatal(err)
			}
			if len(pending.Snapshot.Peers) != 1 || pending.Snapshot.Peers[0].CertFingerprint != peer.identity.CertFingerprint {
				t.Fatalf("snapshot used optional metadata: %+v", pending)
			}
			if !bytes.Equal(beforePin, mustRemoteManagementRead(t, m.trust.pinPath(pin.NodeUUID))) {
				t.Fatal("capture rewrote optional pin metadata")
			}
			reopened, err := NewManager(m.codec, filepath.Dir(m.clusterDir), m.port)
			if err != nil {
				t.Fatal(err)
			}
			doc, completed, err := reopened.readRemoteManagementLocked()
			if err != nil {
				t.Fatal(err)
			}
			if completed.ID != pending.ID || completed.State != "complete" || len(doc.Grants) != 1 || !doc.Grants[0].Allowed || doc.Grants[0].PeerUUID != pin.NodeUUID {
				t.Fatalf("valid leaf not grandfathered after reopen: %+v %+v", doc, completed)
			}
			if got := clustertrust.Open(m.clusterDir).RemoteManagement(pin.NodeUUID); !got.Allowed {
				t.Fatalf("valid optional-metadata pin denied: %+v", got)
			}
			if !bytes.Equal(beforePin, mustRemoteManagementRead(t, m.trust.pinPath(pin.NodeUUID))) {
				t.Fatal("migration/reopen rewrote optional pin metadata")
			}
		})
	}
}

func TestRemoteManagementPendingMigrationIgnoresFingerprintMetadataOnlyChange(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	_, markerPath := m.remoteManagementPaths()
	pending, err := clustertrust.DecodeRemoteManagementMigration(mustRemoteManagementRead(t, markerPath))
	if err != nil {
		t.Fatal(err)
	}
	pin, ok := m.trust.Get(peer.identity.NodeUUID)
	if !ok {
		t.Fatal("fixture pin missing")
	}
	pin.CertFingerprint = ""
	if err := m.trust.Remove(pin.NodeUUID); err != nil {
		t.Fatal(err)
	}
	if err := m.trust.Pin(pin); err != nil {
		t.Fatal(err)
	}
	beforePin := mustRemoteManagementRead(t, m.trust.pinPath(pin.NodeUUID))
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	_, completed, err := m.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	if completed.ID != pending.ID || completed.Snapshot.Peers[0].CertFingerprint != peer.identity.CertFingerprint {
		t.Fatal("metadata-only change changed captured authority")
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(pin.NodeUUID); !got.Allowed {
		t.Fatalf("unchanged certificate denied after metadata edit: %+v", got)
	}
	if !bytes.Equal(beforePin, mustRemoteManagementRead(t, m.trust.pinPath(pin.NodeUUID))) {
		t.Fatal("finalization repaired metadata")
	}
}

func TestRemoteManagementPendingMigrationRejectsChangedCertificateWithOldMetadata(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	pin, ok := m.trust.Get(peer.identity.NodeUUID)
	if !ok {
		t.Fatal("fixture pin missing")
	}
	newCert, _, err := generateLeaf(pin.NodeUUID, "replacement-peer")
	if err != nil {
		t.Fatal(err)
	}
	pin.CertPem = string(newCert)
	if err := m.trust.Remove(pin.NodeUUID); err != nil {
		t.Fatal(err)
	}
	if err := m.trust.Pin(pin); err != nil {
		t.Fatal(err)
	}
	beforePin := mustRemoteManagementRead(t, m.trust.pinPath(pin.NodeUUID))
	m.mesh.Refresh()
	if _, ok := m.mesh.PeerServiceIncarnation(pin.NodeUUID); !ok {
		t.Fatal("replacement pin must have current authority to discriminate certificate comparison")
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	doc, marker, err := m.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	if len(doc.Grants) != 0 || marker.Snapshot.Peers[0].CertFingerprint != peer.identity.CertFingerprint {
		t.Fatalf("replacement certificate inherited original grant: %+v %+v", doc, marker)
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(pin.NodeUUID); got.Allowed || got.State != "unreviewed" {
		t.Fatalf("replacement certificate grandfathered by stale metadata: %+v", got)
	}
	if !bytes.Equal(beforePin, mustRemoteManagementRead(t, m.trust.pinPath(pin.NodeUUID))) {
		t.Fatal("migration rewrote replacement pin")
	}
}

func TestRemoteManagementLegacyAdmissionNormalizationKeepsCertificateEvidence(t *testing.T) {
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	pin, ok := m.trust.Get(peer.identity.NodeUUID)
	if !ok {
		t.Fatal("fixture pin missing")
	}
	pin.ClusterID, pin.AdmissionEpoch, pin.CertFingerprint = "", 0, ""
	if err := m.trust.Remove(pin.NodeUUID); err != nil {
		t.Fatal(err)
	}
	if err := m.trust.Pin(pin); err != nil {
		t.Fatal(err)
	}
	beforePin := mustRemoteManagementRead(t, m.trust.pinPath(pin.NodeUUID))
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(beforePin, mustRemoteManagementRead(t, m.trust.pinPath(pin.NodeUUID))) {
		t.Fatal("capture normalized legacy pin")
	}
	cid, epoch := m.currentAdmission()
	if err := m.migrateLegacyAdmissions(cid, epoch); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	_, marker, err := m.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	captured := marker.Snapshot.Peers[0]
	if captured.ClusterID != "" || captured.AdmissionEpoch != 0 || captured.CertFingerprint != peer.identity.CertFingerprint {
		t.Fatalf("original legacy evidence changed: %+v", captured)
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(pin.NodeUUID); !got.Allowed {
		t.Fatalf("normalized legacy leaf not grandfathered: %+v", got)
	}
}
