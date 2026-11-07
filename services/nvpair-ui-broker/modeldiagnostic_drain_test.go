// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

// Adapted from the audit probe: release must keep the effective proxy drain
// asserted while retiring only the diagnostic's registry contribution.
func TestNodeDrainReleasePreservesManualLoadHold(t *testing.T) {
	test := func(name string, active bool) {
		t.Run(name, func(t *testing.T) {
			b, _, proxy, op := diagnosticAcquisitionFixture(t)
			lease, err := b.drainNode(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			op.drain = lease
			r := b.nodePolicyRuntime()
			r.mu.Lock()
			manual := &modelConfigAttempt{granted: true,
				prepare:  nodepolicy.PrepareResidentParams{Purpose: nodepolicy.ResidentPurposeManual, Engine: "llamacpp"},
				evidence: modelConfigEvidence{Proxy: &proxyProcess{}}}
			if active {
				r.models.active = manual
			} else {
				r.models.unsettled = map[nodepolicy.ModelRef]residentUnsettled{{Engine: "llamacpp", Model: "held-model"}: {op: manual}}
			}
			r.mu.Unlock()
			before := len(proxy.snapshot())
			if err := lease.release(t.Context()); err != nil {
				t.Fatal(err)
			}
			acknowledged := false
			for _, call := range proxy.snapshot()[before:] {
				if call.method != nodepolicy.MethodSetEngineDrain {
					continue
				}
				var p nodepolicy.SetEngineDrainParams
				if err := json.Unmarshal(call.params, &p); err != nil {
					t.Fatal(err)
				}
				if p.Engine == "llamacpp" {
					if !p.Drain {
						t.Fatal("diagnostic release cleared the independent manual-load proxy drain")
					}
					acknowledged = true
				}
			}
			if !acknowledged {
				t.Fatal("effective manual-load drain was not pushed")
			}
			r.mu.Lock()
			registry := r.drains["llamacpp"]
			effective := r.engineDrainsLocked(b.policyProxy())["llamacpp"]
			if active {
				r.models.active = nil
			} else {
				r.models.unsettled = nil
			}
			retired := r.engineDrainsLocked(b.policyProxy())["llamacpp"]
			r.mu.Unlock()
			if registry || !effective || retired || len(lease.added) != 0 {
				t.Fatalf("release registry=%v effective=%v after manual retirement=%v ledger=%v", registry, effective, retired, lease.added)
			}
			if !b.settingsApplyMu.TryLock() {
				t.Fatal("successful release retained settings serialization")
			}
			b.settingsApplyMu.Unlock()
		})
	}
	test("unsettled old-proxy manual load", false)
	test("active old-proxy manual grant", true)
}

func TestNodeDrainReleaseRetainsOwnershipUntilCurrentAcknowledgement(t *testing.T) {
	for _, scenario := range []string{"missing acknowledgement", "refused acknowledgement", "proxy run changes"} {
		t.Run(scenario, func(t *testing.T) {
			b, _, proxy, op := diagnosticAcquisitionFixture(t)
			lease, err := b.drainNode(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			op.drain = lease
			r := b.nodePolicyRuntime()
			proxy.setAnswer(func(string, json.RawMessage) (any, string) {
				switch scenario {
				case "missing acknowledgement":
					return map[string]bool{}, ""
				case "refused acknowledgement":
					return nil, "drain refused"
				default:
					r.mu.Lock()
					r.runID = "replacement-run"
					r.mu.Unlock()
					return map[string]bool{"ok": true}, ""
				}
			})
			t.Cleanup(func() {
				proxy.setAnswer(nil)
				r.mu.Lock()
				r.runID = "proxy-run"
				r.mu.Unlock()
			})
			ctx, cancel := context.WithTimeout(t.Context(), time.Second)
			defer cancel()
			if err := lease.release(ctx); err == nil {
				t.Fatal("unacknowledged or retired release succeeded")
			}
			r.mu.Lock()
			held := r.drains[lease.added[0]]
			r.runID = "proxy-run"
			r.mu.Unlock()
			if !held || !lease.settingsHeld {
				t.Fatal("failed release lost diagnostic ownership")
			}
			if b.settingsApplyMu.TryLock() {
				b.settingsApplyMu.Unlock()
				t.Fatal("failed release allowed a settings writer")
			}
			proxy.setAnswer(nil)
			if err := lease.release(t.Context()); err != nil {
				t.Fatalf("retry release: %v", err)
			}
		})
	}
}
