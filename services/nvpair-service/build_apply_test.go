// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"path/filepath"
	"testing"

	"nvpair-shared/buildapply"
	"nvpair-shared/jsonrpc"
)

func TestBuildApplyStatusStartupAuthority(t *testing.T) {
	f := newBuildApplyRunFixture(t)
	failed := buildapply.OutcomeFailed
	current := f.plan.receipt
	current.Outcome = &failed
	closed := f.plan.receipt
	succeeded := buildapply.OutcomeSucceeded
	closed.Outcome = &succeeded
	tests := []struct {
		name          string
		state         buildapply.JournalState
		err           error
		wantState     buildapply.StatusState
		wantAvailable bool
	}{
		{"idle", buildapply.JournalState{}, nil, buildapply.StatusIdle, true},
		{"pending", buildapply.JournalState{Current: &f.plan.receipt}, nil, buildapply.StatusPending, true},
		{"current with outcome", buildapply.JournalState{Current: &current}, nil, buildapply.StatusTerminal, true},
		{"uncorrelated terminal intent", buildapply.JournalState{TerminalPending: true}, nil, "", false},
		{"terminal pending", buildapply.JournalState{Current: &current, TerminalPending: true}, nil, buildapply.StatusUnknown, true},
		{"closed terminal", buildapply.JournalState{LastTerminal: &closed}, nil, buildapply.StatusTerminal, true},
		{"terminal intent with last terminal", buildapply.JournalState{LastTerminal: &closed, TerminalPending: true}, nil, buildapply.StatusUnknown, true},
		{"read error", buildapply.JournalState{}, errors.New("unreadable"), "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := readBuildApplyStatus(func() (buildapply.JournalState, error) { return tt.state, tt.err }, "")
			if result.Available != tt.wantAvailable {
				t.Fatalf("availability = %+v", result)
			}
			if !result.Available {
				if result.Value != nil {
					t.Fatalf("unavailable has value: %+v", result)
				}
				return
			}
			if result.Value == nil || result.Value.State != tt.wantState || result.Value.StartupAllowed != tt.state.StartupAllowed() {
				t.Fatalf("status = %+v, startup allowed = %v", result, tt.state.StartupAllowed())
			}
		})
	}
}

func TestBuildApplyLocalAdmission(t *testing.T) {
	profile := buildApplyTempDir(t)
	for _, key := range []string{"LOCALAPPDATA", "APPDATA", "HOME", "USERPROFILE", "XDG_CONFIG_HOME"} {
		t.Setenv(key, profile)
	}
	gui := buildapply.GUIFacts{PID: 42, Executable: filepath.Join(t.TempDir(), "PAIR.exe")}
	cases := []struct {
		method string
		params any
	}{
		{"service/build-apply-preview", buildapply.PreviewRequest{CandidatePath: t.TempDir(), GUI: gui}},
		{"service/build-apply-start", buildapply.StartRequest{PreviewID: "preview", GUI: gui}},
		{"service/build-apply-status", buildapply.StatusRequest{}},
		{"service/build-apply-gui-ready", buildapply.GUIReadyRequest{OperationID: "operation", GUI: gui}},
	}
	for _, test := range cases {
		t.Run(test.method, func(t *testing.T) {
			raw, err := json.Marshal(test.params)
			if err != nil {
				t.Fatal(err)
			}
			check := func(t *testing.T, params json.RawMessage, wantCode int) {
				t.Helper()
				s := newService(config{})
				c := &client{out: make(chan []byte, 2)}
				id := json.RawMessage(`"request"`)
				if !s.handleClientRequest(c, &jsonrpc.Message{JSONRPC: "2.0", ID: &id, Method: test.method, Params: params}) {
					t.Fatal("local request closed its client")
				}
				select {
				case frame := <-c.out:
					var response jsonrpc.Message
					if err := json.Unmarshal(frame, &response); err != nil {
						t.Fatal(err)
					}
					if wantCode != 0 {
						if response.Error == nil || response.Error.Code != wantCode {
							t.Fatalf("refusal = %+v", response)
						}
					} else {
						var result buildapply.Result[json.RawMessage]
						if response.Error != nil {
							t.Fatalf("unexpected RPC error: %+v", response.Error)
						}
						if err := json.Unmarshal(response.Result, &result); err != nil {
							t.Fatal(err)
						}
						switch test.method {
						case "service/build-apply-status":
							// Task 3's typed no-start status distinguishes idle from
							// unavailable instead of using Task 1's placeholder.
							status, err := buildapply.DecodeStatusResult(response.Result, "")
							if err != nil || !status.Available || status.Value == nil || status.Value.State != buildapply.StatusIdle || status.Value.Operation != nil {
								t.Fatalf("idle status = %+v, error=%v", status, err)
							}
						case "service/build-apply-start":
							// Task 3 refuses absent stop consent before admission,
							// without opening the journal or contacting the broker.
							if result.Available || result.Refusal == nil || result.Refusal.Code != "interruption_required" {
								t.Fatalf("no-stop refusal = %+v", result)
							}
						default:
							if result.Available || result.Refusal == nil || result.Refusal.Code != buildapply.RefusalUnavailable {
								t.Fatalf("result = %+v", result)
							}
						}
					}
				default:
					t.Fatal("local handler did not respond")
				}
				if s.broker != nil || len(s.pending) != 0 || s.restarts != 0 {
					t.Fatal("unavailable admission entered the broker lifecycle")
				}
				if len(c.out) != 0 {
					t.Fatal("unavailable admission emitted extra frames")
				}
			}
			t.Run("local result without broker", func(t *testing.T) { check(t, raw, 0) })
			for _, field := range []string{"nodeId", "remoteCaller", "command", "endpoint"} {
				t.Run("rejects "+field, func(t *testing.T) {
					var fields map[string]json.RawMessage
					if err := json.Unmarshal(raw, &fields); err != nil {
						t.Fatal(err)
					}
					fields[field] = json.RawMessage(`"untrusted"`)
					bad, err := json.Marshal(fields)
					if err != nil {
						t.Fatal(err)
					}
					check(t, bad, -32602)
				})
			}
		})
	}
}
