// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"log/slog"
	"strings"
	"testing"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
)

func captureBoundRefusalJSON(t *testing.T) *syncBuffer {
	t.Helper()
	output := &syncBuffer{}
	previous := slog.Default()
	previousWriter, previousFlags := log.Writer(), log.Flags()
	slog.SetDefault(slog.New(slog.NewJSONHandler(output, nil)))
	t.Cleanup(func() {
		slog.SetDefault(previous)
		log.SetOutput(previousWriter)
		log.SetFlags(previousFlags)
	})
	return output
}

type boundRefusalRecord struct {
	Message             string `json:"msg"`
	Site                string `json:"site"`
	Reason              string `json:"reason"`
	NodeID              string `json:"node_id"`
	PeerID              string `json:"peer_id"`
	ClusterID           string `json:"cluster_id"`
	Epoch               uint64 `json:"epoch"`
	Counter             uint64 `json:"counter"`
	ExpectedIncarnation string `json:"expected_incarnation"`
	ObservedIncarnation string `json:"observed_incarnation"`
	Permission          string `json:"permission"`
}

func boundRefusalRecords(t *testing.T, output *syncBuffer) []boundRefusalRecord {
	t.Helper()
	output.mu.Lock()
	data := append([]byte(nil), output.buf.Bytes()...)
	output.mu.Unlock()
	decoder := json.NewDecoder(bytes.NewReader(data))
	var records []boundRefusalRecord
	for {
		var record boundRefusalRecord
		if err := decoder.Decode(&record); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			t.Fatalf("decode refusal log: %v", err)
		}
		if record.Message == "bound engine management refused" {
			records = append(records, record)
		}
	}
	return records
}

func TestBoundEnginePeerIncarnationRefusalDiagnostic(t *testing.T) {
	h := newBoundHarness(t, false, true)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
	mesh := clustertrust.Open(h.b.clusterDir)
	current, ok := mesh.PeerServiceIncarnation("peer")
	if !ok {
		t.Fatal("fixture peer lacks current authority")
	}
	h.request.PeerID = "peer"
	h.request.PeerIncarnation = strings.Repeat("0", 64)
	output := captureBoundRefusalJSON(t)
	result := h.b.prepareBoundEngine(context.Background(), h.pp, h.request)
	if result.Reason != "management-denied" || result.Code != -32012 {
		t.Fatalf("incarnation refusal = %+v", result)
	}
	records := boundRefusalRecords(t, output)
	if len(records) != 1 {
		t.Fatalf("refusal records = %+v", records)
	}
	record := records[0]
	if record.Site != "broker.peer_incarnation" || record.Reason != "incarnation_mismatch" ||
		record.NodeID != "target" || record.PeerID != "peer" || record.ClusterID != "cluster" ||
		record.Epoch == 0 || record.Counter < record.Epoch || record.Permission != "not_checked" ||
		record.ExpectedIncarnation != h.request.PeerIncarnation || record.ObservedIncarnation != current {
		t.Fatalf("refusal metadata = %+v", record)
	}
	h.request.PeerIncarnation = current
	if result := h.b.prepareBoundEngine(context.Background(), h.pp, h.request); !result.Success {
		t.Fatalf("ready target after valid authority = %+v", result)
	}
	if records := boundRefusalRecords(t, output); len(records) != 1 {
		t.Fatalf("successful lookup added a refusal log: %+v", records)
	}
	requireNoBoundEffects(t, h)
}

func TestBoundEngineRemoteManagementRefusalDiagnostic(t *testing.T) {
	b, _ := newPolicyBroker(t)
	clustertrusttest.Join(t, b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, b.clusterDir, "cluster", "target", "peer", false)
	output := captureBoundRefusalJSON(t)
	rmDeny(t, b.authorizeRemoteManagement("peer", "bound-engine", "diagnostic"))
	records := boundRefusalRecords(t, output)
	if len(records) != 1 {
		t.Fatalf("refusal records = %+v", records)
	}
	record := records[0]
	if record.Site != "broker.remote_management" || record.Reason != "permission_denied" ||
		record.NodeID != "target" || record.PeerID != "peer" || record.ClusterID != "cluster" ||
		record.Epoch == 0 || record.Counter < record.Epoch || record.Permission != "denied" {
		t.Fatalf("permission refusal metadata = %+v", record)
	}
	rmDeny(t, b.authorizeRemoteManagement("peer", "switch", "binding:diagnostic:1", true))
	if records := boundRefusalRecords(t, output); len(records) != 2 || records[1].Reason != "permission_denied" {
		t.Fatalf("automatic switch permission refusal = %+v", records)
	}
	grantRemoteManagement(t, b.clusterDir, "cluster", "target", "peer", true)
	if err := b.authorizeRemoteManagement("peer", "bound-engine", "diagnostic"); err != nil {
		t.Fatalf("granted authority refused: %v", err)
	}
	if records := boundRefusalRecords(t, output); len(records) != 2 {
		t.Fatalf("successful permission added a refusal log: %+v", records)
	}
}
