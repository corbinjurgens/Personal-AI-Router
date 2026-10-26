// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
)

type boundManagementLog struct {
	Time                string `json:"time"`
	Level               string `json:"level"`
	Message             string `json:"msg"`
	Site                string `json:"site"`
	Reason              string `json:"reason"`
	NodeID              string `json:"node_id"`
	PeerID              string `json:"peer_id"`
	ClusterID           string `json:"cluster_id"`
	Epoch               uint64 `json:"epoch"`
	Counter             uint64 `json:"counter"`
	ErrorCode           uint64 `json:"io_error_code"`
	ExpectedIncarnation string `json:"expected_incarnation"`
	ObservedIncarnation string `json:"observed_incarnation"`
	Permission          string `json:"permission"`
	ChangedCompared     bool   `json:"changed_component_authority_compared"`
	ChangedSame         bool   `json:"changed_component_authority_same"`
}

// These tests remain serial, like the other default-logger capture tests in
// this package. Parallel tests must not replace the process-wide logger.
func captureBoundManagementLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return &logs
}

func assertBoundManagementLog(t *testing.T, logs *bytes.Buffer, want boundManagementLog) {
	t.Helper()
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(logs.Bytes(), &fields); err != nil {
		t.Fatalf("decode management diagnostic fields: %v", err)
	}
	for _, name := range []string{"time", "level", "msg", "site", "reason", "node_id", "peer_id", "cluster_id", "epoch", "counter", "io_error_code", "expected_incarnation", "observed_incarnation", "permission", "changed_component_authority_compared", "changed_component_authority_same"} {
		if _, ok := fields[name]; !ok {
			t.Fatalf("management diagnostic missing field %q", name)
		}
	}
	decoder := json.NewDecoder(bytes.NewReader(logs.Bytes()))
	decoder.DisallowUnknownFields()
	var got boundManagementLog
	if err := decoder.Decode(&got); err != nil {
		t.Fatalf("decode management diagnostic: %v", err)
	}
	var extra boundManagementLog
	if err := decoder.Decode(&extra); err != io.EOF {
		t.Fatalf("expected exactly one management diagnostic, trailing decode: %v", err)
	}
	if got.Time == "" {
		t.Fatal("management diagnostic has no timestamp")
	}
	got.Time = ""
	want.Level, want.Message, want.Permission = "WARN", "bound engine management refused", "not_checked"
	if got != want {
		t.Fatalf("management diagnostic=%+v, want %+v", got, want)
	}
}

func newBoundPeerDiagnosticFixture(t *testing.T) (*bindingRequestFixture, string, string) {
	t.Helper()
	fx := newBindingRequestFixture(t)
	selfDir, peerDir := t.TempDir(), t.TempDir()
	clustertrusttest.Join(t, selfDir, "fixture-cluster", "destination")
	clustertrusttest.Join(t, peerDir, "fixture-cluster", "authenticated-origin")
	proxyReversePin(t, selfDir, peerDir, "authenticated-origin")
	fx.p.mesh = clustertrust.Open(selfDir)
	incarnation, ok := fx.p.mesh.PeerServiceIncarnation("authenticated-origin")
	if !ok {
		t.Fatal("missing fixture peer relationship")
	}
	return fx, selfDir, incarnation
}

func TestBoundIngressPeerCurrentSuccessDoesNotLog(t *testing.T) {
	fx, _, incarnation := newBoundPeerDiagnosticFixture(t)
	logs := captureBoundManagementLogs(t)
	ctx := context.WithValue(context.Background(), boundEnginePeerKey{}, boundEnginePeer{"authenticated-origin", incarnation})
	if !fx.f.boundIngressPeerCurrent(ctx) {
		t.Fatal("current pinned relationship refused")
	}
	if !fx.f.boundIngressPeerCurrent(context.Background()) {
		t.Fatal("local caller without peer context refused")
	}
	if logs.Len() != 0 {
		t.Fatal("accepted peer or local caller emitted a management diagnostic")
	}
}

func TestBoundIngressPeerCurrentRefusalDiagnostic(t *testing.T) {
	test := func(name string, wantCurrent bool, change func(*testing.T, *bindingRequestFixture, string)) {
		t.Run(name, func(t *testing.T) {
			fx, selfDir, expected := newBoundPeerDiagnosticFixture(t)
			change(t, fx, selfDir)
			observed, current, diagnostic := fx.p.mesh.PeerServiceIncarnationWithDiagnostic("authenticated-origin")
			if current != wantCurrent {
				t.Fatalf("fixture authority current=%v, want %v", current, wantCurrent)
			}
			reason := diagnostic.Predicate
			if current {
				if observed == expected {
					t.Fatal("replacement did not change fixture relationship")
				}
				reason = "incarnation_mismatch"
			} else if reason == "" || reason == "incarnation_mismatch" {
				t.Fatalf("authority refusal lacks a distinct owner predicate: %q", reason)
			}
			logs := captureBoundManagementLogs(t)
			ctx := context.WithValue(context.Background(), boundEnginePeerKey{}, boundEnginePeer{"authenticated-origin", expected})
			if fx.f.boundIngressPeerCurrent(ctx) {
				t.Fatal("stale or missing authority accepted")
			}
			assertBoundManagementLog(t, logs, boundManagementLog{
				Site: "proxy.peer_incarnation", Reason: reason, NodeID: diagnostic.NodeID,
				PeerID: "authenticated-origin", ClusterID: diagnostic.ClusterID,
				Epoch: diagnostic.Epoch, Counter: diagnostic.Counter, ErrorCode: diagnostic.ErrorCode,
				ExpectedIncarnation: expected, ObservedIncarnation: observed,
			})
		})
	}
	test("changed incarnation", true, func(t *testing.T, fx *bindingRequestFixture, selfDir string) {
		replacement := t.TempDir()
		clustertrusttest.Join(t, replacement, "fixture-cluster", "authenticated-origin")
		proxyReversePin(t, selfDir, replacement, "authenticated-origin")
		fx.p.mesh.Refresh()
	})
	test("missing admission authority", false, func(t *testing.T, _ *bindingRequestFixture, selfDir string) {
		if err := os.Remove(filepath.Join(selfDir, "admission.json")); err != nil {
			t.Fatal(err)
		}
	})
}

func TestPrepareLocalBoundPeerContextRefusalDiagnostic(t *testing.T) {
	test := func(name, reason string, ctx context.Context) {
		t.Run(name, func(t *testing.T) {
			fx := newBindingRequestFixture(t)
			logs := captureBoundManagementLogs(t)
			ctx = context.WithValue(ctx, boundEngineDeadlineKey{}, time.Now().Add(time.Second))
			evidence, result := fx.f.prepareLocalBound(ctx, "fixture", "authenticated-origin")
			if evidence != nil || result.Success || result.Reason != "management-denied" {
				t.Fatalf("invalid peer context preparation=%+v, evidence=%+v", result, evidence)
			}
			assertBoundManagementLog(t, logs, boundManagementLog{
				Site: "proxy.peer_context", Reason: reason, PeerID: "authenticated-origin",
			})
			select {
			case in := <-fx.requests:
				t.Fatalf("invalid peer context submitted preparation: %+v", in)
			default:
			}
		})
	}
	test("missing context", "peer_context_missing", context.Background())
	test("invalid context", "peer_context_missing", context.WithValue(context.Background(), boundEnginePeerKey{}, "invalid-context-sentinel"))
	test("different principal", "peer_context_mismatch", context.WithValue(context.Background(), boundEnginePeerKey{}, boundEnginePeer{"other-peer", "uncompared-incarnation"}))
}

func TestPrepareLocalBoundLocalCallerDoesNotRequirePeerContext(t *testing.T) {
	fx := newBindingRequestFixture(t)
	logs := captureBoundManagementLogs(t)
	ctx := context.WithValue(context.Background(), boundEnginePeerKey{}, "irrelevant-local-context")
	ctx = context.WithValue(ctx, boundEngineDeadlineKey{}, time.Now().Add(time.Second))
	ctx, cancel := context.WithCancel(ctx)
	cancel()
	evidence, result := fx.f.prepareLocalBound(ctx, "fixture", "")
	if evidence != nil || result.Success || result.Reason != "engine-switch-timeout" {
		t.Fatalf("local caller did not reach existing cancellation owner: result=%+v, evidence=%+v", result, evidence)
	}
	if logs.Len() != 0 {
		t.Fatal("local caller emitted a peer management diagnostic")
	}
}
