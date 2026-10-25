// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

// assertMemberReaderSerialized holds a real snapshot writer at Sync, then
// hides its old target to model an unreadable Windows replacement interval.
// The reader must wait at storage ownership, without retaining memMu, until
// the old target is restored and the writer completes its replacement.
func assertMemberReaderSerialized(t *testing.T, m *Manager, method string, read func() error) {
	t.Helper()
	s := newHintWriteSchedule(t, m, nil)
	writer := s.start("member writer", m.persistMembersErr)
	if !assertHeldMemberStore(t, m, s) {
		t.Fatal("member writer did not own storage")
	}
	hidden := m.membersPath() + ".held"
	if err := os.Rename(m.membersPath(), hidden); err != nil {
		t.Fatalf("hide held member snapshot: %v", err)
	}
	restored := false
	defer func() {
		if !restored {
			if err := os.Rename(hidden, m.membersPath()); err != nil {
				t.Errorf("restore held member snapshot: %v", err)
			}
		}
		s.unblock()
	}()
	reader := s.start("member reader", read)
	deadline := time.Now().Add(5 * time.Second)
	for {
		select {
		case err := <-reader:
			t.Fatalf("member reader completed during replacement interval: %v", err)
		default:
		}
		stack := make([]byte, 256*1024)
		n := runtime.Stack(stack, true)
		blocked := false
		for _, worker := range strings.Split(string(stack[:n]), "\n\n") {
			if strings.Contains(worker, "(*Manager)."+method+"(") && strings.Contains(worker, "sync.runtime_SemacquireMutex") {
				blocked = true
				break
			}
		}
		if blocked {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("member reader did not wait at storage ownership")
		}
		runtime.Gosched()
	}
	if !m.memMu.TryLock() {
		t.Fatal("member reader retained memMu while waiting for storage")
	}
	m.memMu.Unlock()
	if err := os.Rename(hidden, m.membersPath()); err != nil {
		t.Fatalf("restore held member snapshot: %v", err)
	}
	restored = true
	s.unblock()
	if err := waitHintWriter(t, writer); err != nil {
		t.Fatalf("member writer: %v", err)
	}
	if err := waitHintWriter(t, reader); err != nil {
		t.Fatalf("member reader: %v", err)
	}
	s.join(t)
}

func TestPeerServiceIncarnationWaitsForMemberReplacement(t *testing.T) {
	f := newPeerHintFixture(t)
	assertMemberReaderSerialized(t, f.m, "peerServiceIncarnation", func() error {
		incarnation, ok := f.m.peerServiceIncarnation(f.peer.UUID)
		if !ok || incarnation != f.incarnation {
			return fmt.Errorf("incarnation = %q, valid = %t, want %q", incarnation, ok, f.incarnation)
		}
		return nil
	})
}

func TestPeerServicesRefreshWaitsForMemberReplacement(t *testing.T) {
	f := newPeerHintFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Stop after authority capture, before any chooser or HTTP dial.
	assertMemberReaderSerialized(t, f.m, "refreshPeerServices", func() error {
		if err := f.m.refreshPeerServices(ctx, f.peer.UUID); !errors.Is(err, context.Canceled) {
			return fmt.Errorf("refresh error = %v, want context cancellation after admission", err)
		}
		return nil
	})
}

func TestPeerServicesHintCommitWaitsForMemberReplacement(t *testing.T) {
	f := newPeerHintFixture(t)
	// Refresh and reversePeer run while the old members.json is hidden. Neither
	// reads it; the subsequent incarnation check must wait for the replacement.
	assertMemberReaderSerialized(t, f.m, "commitPeerServiceHints", f.commit)
	assertPeerHints(t, f, f.descriptor.Addresses)
}

func TestPairingReceiptCompletionWaitsForMemberReplacement(t *testing.T) {
	f := newPeerHintFixture(t)
	doc, marker, err := f.m.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	info, err := json.Marshal(f.remote.localPairingInfo(""))
	if err != nil {
		t.Fatal(err)
	}
	cid, epoch := f.m.currentAdmission()
	allow := true
	receipt := pairingReceipt{
		InviteID: "member-reader-pairing", Role: roleInviter, ClusterID: cid,
		LocalEpoch: epoch, PeerInfo: info, Rollback: true,
		ExpiresAt: time.Now().Add(time.Hour).UnixMilli(), AllowRemoteManagement: &allow,
		PermissionRevision: doc.Revision, PermissionMigrationID: marker.ID,
		LocalCertFingerprint: f.m.identity.CertFingerprint,
	}
	if err := f.m.savePairingReceipt(receipt); err != nil {
		t.Fatal(err)
	}
	f.m.rosterMu.Lock()
	err = f.m.ensurePairingGuardLocked(receipt)
	f.m.rosterMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	assertMemberReaderSerialized(t, f.m, "completePairingReceiptLocked", func() error {
		f.m.rosterMu.Lock()
		defer f.m.rosterMu.Unlock()
		return f.m.completePairingReceiptLocked(receipt)
	})
	completed, ok := f.m.getPairingReceipt(receipt.InviteID)
	if !ok || completed.Rollback || completed.State != inviteStatePaired || completed.PermissionIncarnation != f.incarnation {
		t.Fatalf("completed receipt = %+v, found = %t", completed, ok)
	}
	if permission := f.m.mesh.RemoteManagement(f.peer.UUID); !permission.Allowed || permission.Incarnation != f.incarnation {
		t.Fatalf("paired permission = %+v", permission)
	}
}

func TestRemoteManagementMigrationWaitsForMemberReplacement(t *testing.T) {
	f := newPeerHintFixture(t)
	docPath, markerPath := f.m.remoteManagementPaths()
	// Model a pre-permission profile whose already admitted peer must migrate.
	for _, path := range []string{docPath, markerPath} {
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
	}
	f.m.rosterMu.Lock()
	err := f.m.captureRemoteManagementMigrationLocked()
	f.m.rosterMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	assertMemberReaderSerialized(t, f.m, "finalizeRemoteManagementMigrationLocked", func() error {
		f.m.rosterMu.Lock()
		defer f.m.rosterMu.Unlock()
		return f.m.finalizeRemoteManagementMigrationLocked()
	})
	doc, marker, err := f.m.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	if marker.State != "complete" || len(doc.Grants) != 1 || doc.Grants[0].PeerUUID != f.peer.UUID ||
		doc.Grants[0].Incarnation != f.incarnation || !doc.Grants[0].Allowed {
		t.Fatalf("migrated permission = %+v, marker = %+v", doc, marker)
	}
}

func TestRemoteManagementSetterWaitsForMemberReplacement(t *testing.T) {
	f := newPeerHintFixture(t)
	other := newReverseTestManager(t)
	containHintDials(t, other, false)
	pinTrusted(t, f.m, other.identity.NodeUUID, string(other.identity.CertPEM), other.identity.CertFingerprint)
	f.m.upsertMember(&ClusterNode{NodeUUID: other.identity.NodeUUID, ID: "other-peer", ClusterID: "cluster-1", AdmissionEpoch: 1, State: stateMember})
	if err := f.m.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	f.m.mesh.Refresh()
	f.m.rosterMu.Lock()
	err := f.m.setRemoteManagementLocked(other.identity.NodeUUID, true, "1")
	f.m.rosterMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	assertMemberReaderSerialized(t, f.m, "setRemoteManagementWithGuardLocked", func() error {
		f.m.rosterMu.Lock()
		defer f.m.rosterMu.Unlock()
		return f.m.setRemoteManagementLocked(f.peer.UUID, false, "2")
	})
	doc, _, err := f.m.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	if doc.Revision != "3" || len(doc.Grants) != 2 {
		t.Fatalf("updated permissions = %+v", doc)
	}
	for _, grant := range doc.Grants {
		switch grant.PeerUUID {
		case f.peer.UUID:
			if grant.Allowed || grant.Incarnation != f.incarnation || grant.DecisionRevision != "3" {
				t.Fatalf("updated peer grant = %+v", grant)
			}
		case other.identity.NodeUUID:
			if !grant.Allowed || grant.DecisionRevision != "2" {
				t.Fatalf("unrelated grant was changed = %+v", grant)
			}
		default:
			t.Fatalf("unexpected grant = %+v", grant)
		}
	}
}

func TestRemoteManagementSnapshotWaitsForMemberReplacement(t *testing.T) {
	f := newPeerHintFixture(t)
	f.m.rosterMu.Lock()
	err := f.m.setRemoteManagementLocked(f.peer.UUID, true, "1")
	f.m.rosterMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	var view remoteManagementSnapshot
	assertMemberReaderSerialized(t, f.m, "remoteManagementSnapshotLocked", func() error {
		f.m.rosterMu.Lock()
		defer f.m.rosterMu.Unlock()
		view = f.m.remoteManagementSnapshotLocked()
		return nil
	})
	if view.Revision != "2" || len(view.Grants) != 1 || view.Grants[0].PeerUUID != f.peer.UUID || !view.Grants[0].Allowed || view.Grants[0].State != "allowed" {
		t.Fatalf("permission snapshot = %+v", view)
	}
}
