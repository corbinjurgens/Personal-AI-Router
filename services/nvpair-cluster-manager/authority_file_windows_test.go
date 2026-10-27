// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/windows"

	"nvpair-shared/clustertrust"
)

func writeClusterAuthorityJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode authority fixture: %v", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("seed authority fixture: %v", err)
	}
}

// An already-open DELETE handle deterministically rejects os.ReadFile while
// allowing the shared read opener. It neither mutates the file nor races a
// goroutine against a replacement.
func holdClusterAuthorityDeleteAccess(t *testing.T, path string) {
	t.Helper()
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		t.Fatalf("encode authority path: %v", err)
	}
	handle, err := windows.CreateFile(name, windows.DELETE,
		windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE,
		nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatalf("hold authority DELETE access: %v", err)
	}
	t.Cleanup(func() {
		if err := windows.CloseHandle(handle); err != nil {
			t.Errorf("close authority DELETE handle: %v", err)
		}
	})
	if _, err := os.ReadFile(path); !errors.Is(err, windows.ERROR_SHARING_VIOLATION) {
		t.Fatalf("ordinary reader error = %v, want ERROR_SHARING_VIOLATION", err)
	}
}

func TestClusterAuthorityReadsShareDeleteAccess(t *testing.T) {
	t.Run("members", func(t *testing.T) {
		m := &Manager{clusterDir: t.TempDir(), members: make(map[string]*ClusterNode)}
		writeClusterAuthorityJSON(t, m.membersPath(), []ClusterNode{{NodeUUID: "peer", State: stateMember, AdmissionEpoch: 7}})
		holdClusterAuthorityDeleteAccess(t, m.membersPath())
		if err := m.loadMembers(); err != nil {
			t.Fatalf("load members with DELETE access held: %v", err)
		}
		if got := m.members["peer"]; got == nil || got.AdmissionEpoch != 7 || got.State != stateMember {
			t.Fatalf("loaded member = %+v, want peer admission 7", got)
		}
	})
	t.Run("admission", func(t *testing.T) {
		m := &Manager{clusterDir: t.TempDir()}
		writeClusterAuthorityJSON(t, m.admissionPath(), admissionState{Counter: 3, Activated: 3, ClusterID: "cluster", Epoch: 3})
		holdClusterAuthorityDeleteAccess(t, m.admissionPath())
		if err := m.loadAdmission(); err != nil {
			t.Fatalf("load admission with DELETE access held: %v", err)
		}
		if cid, epoch := m.currentAdmission(); cid != "cluster" || epoch != 3 {
			t.Fatalf("loaded admission = %q, %d; want cluster, 3", cid, epoch)
		}
		m.clusterID = "cluster"
		m.rosterMu.Lock()
		available := m.joinAdmissionAvailableLocked()
		m.rosterMu.Unlock()
		if !available {
			t.Fatal("join admission reread refused a sharing-compatible writer")
		}
	})
	t.Run("trusted pin", func(t *testing.T) {
		dir := t.TempDir()
		store, err := newTrustStore(dir)
		if err != nil {
			t.Fatalf("create trusted directory: %v", err)
		}
		cert, _, err := generateLeaf("peer", "peer")
		if err != nil {
			t.Fatalf("generate peer certificate: %v", err)
		}
		writeClusterAuthorityJSON(t, store.pinPath("peer"), TrustedPin{NodeUUID: "peer", CertPem: string(cert), AdmissionEpoch: 7})
		holdClusterAuthorityDeleteAccess(t, store.pinPath("peer"))
		loaded, err := newTrustStore(dir)
		if err != nil {
			t.Fatalf("load trust store: %v", err)
		}
		if pin, ok := loaded.Get("peer"); !ok || pin.AdmissionEpoch != 7 {
			t.Fatalf("loaded pin = %+v, found = %v; want peer admission 7", pin, ok)
		}
	})
	t.Run("remote management bounded reader", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "permission.json")
		writeClusterAuthorityJSON(t, path, map[string]bool{"allowed": false})
		holdClusterAuthorityDeleteAccess(t, path)
		data, err := remoteManagementRead(path)
		if err != nil {
			t.Fatalf("read permission with DELETE access held: %v", err)
		}
		var result map[string]bool
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatalf("decode permission: %v", err)
		}
		if allowed, ok := result["allowed"]; !ok || allowed {
			t.Fatalf("permission = %v, want allowed false", result)
		}
	})
	t.Run("identity and keypair", func(t *testing.T) {
		dir := t.TempDir()
		original, err := loadOrMintIdentity(dir)
		if err != nil {
			t.Fatalf("mint fixture identity: %v", err)
		}
		for _, name := range []string{"identity.json", "node.crt", "node.key"} {
			holdClusterAuthorityDeleteAccess(t, filepath.Join(dir, name))
		}
		loaded, err := loadOrMintIdentity(dir)
		if err != nil {
			t.Fatalf("load identity with DELETE access held: %v", err)
		}
		if loaded.NodeUUID != original.NodeUUID || loaded.CertFingerprint != original.CertFingerprint {
			t.Fatal("loading identity changed its principal")
		}
	})
	t.Run("pairing receipts", func(t *testing.T) {
		m := &Manager{clusterDir: t.TempDir()}
		path := filepath.Join(m.clusterDir, pairingReceiptsFile)
		writeClusterAuthorityJSON(t, path, map[string]pairingReceipt{})
		holdClusterAuthorityDeleteAccess(t, path)
		if err := m.loadPairingReceipts(); err != nil {
			t.Fatalf("load receipts with DELETE access held: %v", err)
		}
	})
	t.Run("removal proofs", func(t *testing.T) {
		m := &Manager{clusterDir: t.TempDir(), removalProofs: make(map[string]RemovalProof)}
		writeClusterAuthorityJSON(t, m.removalProofsPath(), []RemovalProof{{Tombstone: Tombstone{NodeUUID: "peer", AdmissionEpoch: 7}}})
		holdClusterAuthorityDeleteAccess(t, m.removalProofsPath())
		if err := m.loadRemovalProofs(); err != nil {
			t.Fatalf("load removal proofs with DELETE access held: %v", err)
		}
		if got := m.removalProofs["peer"].Tombstone.AdmissionEpoch; got != 7 {
			t.Fatalf("loaded removal admission = %d, want 7", got)
		}
	})
	t.Run("tombstones", func(t *testing.T) {
		m := &Manager{clusterDir: t.TempDir(), tombstones: make(map[string]Tombstone)}
		writeClusterAuthorityJSON(t, m.tombstonesPath(), []Tombstone{{NodeUUID: "peer", AdmissionEpoch: 7}})
		holdClusterAuthorityDeleteAccess(t, m.tombstonesPath())
		m.loadTombstones()
		if got := m.tombstones["peer"].AdmissionEpoch; got != 7 {
			t.Fatalf("loaded tombstone admission = %d, want 7", got)
		}
	})
	t.Run("invite provenance", func(t *testing.T) {
		m := &Manager{clusterDir: t.TempDir(), clusterID: "cluster"}
		writeClusterAuthorityJSON(t, m.inviteCreatedClusterPath(), inviteCreatedClusterMarker{ClusterID: "cluster"})
		holdClusterAuthorityDeleteAccess(t, m.inviteCreatedClusterPath())
		if err := m.loadInviteCreatedCluster(); err != nil {
			t.Fatalf("load provenance with DELETE access held: %v", err)
		}
		if !m.isInviteCreatedCluster() {
			t.Fatal("lost invite-created cluster provenance")
		}
	})
}

func TestClusterJoinEligibilityReadsShareDeleteAccess(t *testing.T) {
	m := joinTestManager(t)
	writeClusterAuthorityJSON(t, m.membersPath(), []ClusterNode{})
	writeClusterAuthorityJSON(t, m.admissionPath(), admissionState{})
	receiptsPath := filepath.Join(m.clusterDir, pairingReceiptsFile)
	writeClusterAuthorityJSON(t, receiptsPath, map[string]pairingReceipt{})
	for _, path := range []string{
		m.membersPath(), m.admissionPath(), receiptsPath,
		filepath.Join(m.clusterDir, clustertrust.RemoteManagementFile),
		filepath.Join(m.clusterDir, clustertrust.RemoteManagementMigrationFile),
	} {
		holdClusterAuthorityDeleteAccess(t, path)
	}
	m.inviteMu.Lock()
	m.rosterMu.Lock()
	m.joinMu.Lock()
	solo, reason := m.joinEligibilityLocked("")
	m.joinMu.Unlock()
	m.rosterMu.Unlock()
	m.inviteMu.Unlock()
	if solo || reason != "" {
		t.Fatalf("fresh unclustered eligibility: solo=%v, reason=%q; want false, empty", solo, reason)
	}
}
