// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// Startup applies the same retention rule as savePairingReceipt: an expired
// receipt without an abort intent is dropped unread, so a format change to
// PairingInfo cannot strand a node on a receipt nothing will act on again.
func TestLoadPairingReceiptsRetention(t *testing.T) {
	expired := time.Now().Add(-time.Hour).UnixMilli()
	unexpired := time.Now().Add(time.Hour).UnixMilli()

	newFixture := func(t *testing.T) (*Manager, json.RawMessage, json.RawMessage) {
		t.Helper()
		m := testManagerAt(t, t.TempDir(), 15129)
		peer := testManagerAt(t, t.TempDir(), 15130)
		valid, err := json.Marshal(peer.localPairingInfo("", 1))
		if err != nil {
			t.Fatalf("marshal peer pairing info: %v", err)
		}
		var fields map[string]any
		if err := json.Unmarshal(valid, &fields); err != nil {
			t.Fatalf("decode peer pairing info: %v", err)
		}
		fields["nodeId"] = peer.identity.NodeUUID // the key the node-name change removed
		unknown, err := json.Marshal(fields)
		if err != nil {
			t.Fatalf("marshal pairing info with unknown key: %v", err)
		}
		return m, valid, unknown
	}
	receiptWith := func(peerInfo json.RawMessage, expiresAt int64, rollback bool) pairingReceipt {
		return pairingReceipt{InviteID: "stored", Role: roleInviter, ClusterID: "c", LocalEpoch: 1,
			PeerInfo: peerInfo, ExpiresAt: expiresAt, State: inviteStatePaired, Rollback: rollback}
	}
	writeReceipt := func(t *testing.T, m *Manager, receipt pairingReceipt) {
		t.Helper()
		data, err := json.Marshal(map[string]pairingReceipt{receipt.InviteID: receipt})
		if err != nil {
			t.Fatalf("marshal receipts: %v", err)
		}
		if err := os.WriteFile(filepath.Join(m.clusterDir, pairingReceiptsFile), data, 0o600); err != nil {
			t.Fatalf("write receipts: %v", err)
		}
	}
	assertLoaded := func(t *testing.T, m *Manager, wantPresent bool) {
		t.Helper()
		if err := m.loadPairingReceipts(); err != nil {
			t.Fatalf("load pairing receipts: %v", err)
		}
		if _, present := m.pairingReceipts["stored"]; present != wantPresent {
			t.Fatalf("receipt present=%t, want %t", present, wantPresent)
		}
	}
	assertRejected := func(t *testing.T, m *Manager) {
		t.Helper()
		if err := m.loadPairingReceipts(); err == nil {
			t.Fatal("load accepted a retained receipt with an unknown PairingInfo key")
		}
	}

	t.Run("drops expired receipt without validating it", func(t *testing.T) {
		m, _, unknown := newFixture(t)
		writeReceipt(t, m, receiptWith(unknown, expired, false))
		assertLoaded(t, m, false)
	})
	t.Run("rejects unexpired receipt with unknown key", func(t *testing.T) {
		m, _, unknown := newFixture(t)
		writeReceipt(t, m, receiptWith(unknown, unexpired, false))
		assertRejected(t, m)
	})
	t.Run("keeps valid expired rollback receipt", func(t *testing.T) {
		m, valid, _ := newFixture(t)
		writeReceipt(t, m, receiptWith(valid, expired, true))
		assertLoaded(t, m, true)
	})
	t.Run("rejects expired rollback receipt with unknown key", func(t *testing.T) {
		m, _, unknown := newFixture(t)
		writeReceipt(t, m, receiptWith(unknown, expired, true))
		assertRejected(t, m)
	})
	// completePairingReceiptLocked records success before releasing the guard.
	// A crash between them leaves the guard naming a non-rollback receipt that
	// recovery must still find after the receipt expired.
	t.Run("keeps expired receipt that owns the publication guard", func(t *testing.T) {
		m, valid, _ := newFixture(t)
		receipt := receiptWith(valid, expired, false)
		m.rosterMu.Lock()
		err := m.ensurePairingGuardLocked(receipt)
		m.rosterMu.Unlock()
		if err != nil {
			t.Fatalf("install publication guard: %v", err)
		}
		writeReceipt(t, m, receipt)
		assertLoaded(t, m, true)
	})
	t.Run("rejects guard-owned expired receipt with unknown key", func(t *testing.T) {
		m, _, unknown := newFixture(t)
		receipt := receiptWith(unknown, expired, false)
		m.rosterMu.Lock()
		err := m.ensurePairingGuardLocked(receipt)
		m.rosterMu.Unlock()
		if err != nil {
			t.Fatalf("install publication guard: %v", err)
		}
		writeReceipt(t, m, receipt)
		assertRejected(t, m)
	})
}
