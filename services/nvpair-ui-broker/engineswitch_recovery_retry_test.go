// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

func hideSwitchRecoveryManager(t *testing.T, b *Broker) *rpcWorker {
	t.Helper()
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	if b.engineMgr == nil {
		t.Fatal("missing test engine manager")
	}
	b.engineMgrRecovering = true
	return b.engineMgr
}

func TestSwitchRecoveryRetriesInspectionBeforePublishingManager(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	base := h.engine.answer
	var inspections atomic.Int32
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			if h.b.getEngineMgr() != nil {
				t.Error("manager published before recovery inspection succeeded")
			}
			if inspections.Add(1) == 1 {
				return nil, "temporary inspection failure"
			}
		}
		return base(method, raw)
	})
	if !h.b.recoverReplacementEngineMgr(w) {
		t.Fatal("transient inspection failure did not recover")
	}
	if inspections.Load() != 2 || h.b.getEngineMgr() != w {
		t.Fatalf("inspections = %d, manager = %p", inspections.Load(), h.b.getEngineMgr())
	}
	if h.b.engineSwitch.err != nil || h.b.engineSwitch.recoveryErr != nil {
		t.Fatalf("recovery retained errors: permanent %v, retryable %v", h.b.engineSwitch.err, h.b.engineSwitch.recoveryErr)
	}
	if steps := switchSteps(t, h.engine); len(steps) != 0 {
		t.Fatalf("inspection recovery executed forward steps: %v", steps)
	}
}

func TestSwitchRecoveryRetriesDrainReleaseWithoutRestart(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	r := switchStoreReceipt("retry-drain-release")
	switchStoreAccept(t, h.b.engineSwitch.store, r)
	runtime := h.b.nodePolicyRuntime()
	runtime.mu.Lock()
	runtime.drains[r.Request.SourceEngine] = true
	runtime.mu.Unlock()
	base := h.proxy.answer
	var releases atomic.Int32
	h.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetEngineDrain {
			var p nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if !p.Drain {
				if h.b.getEngineMgr() != nil {
					t.Error("manager published before drain release acknowledgement")
				}
				if releases.Add(1) == 1 {
					return nil, "proxy restarting"
				}
			}
		}
		return base(method, raw)
	})
	if !h.b.recoverReplacementEngineMgr(w) {
		t.Fatal("transient drain release failure did not recover")
	}
	if releases.Load() != 2 || h.b.getEngineMgr() != w {
		t.Fatalf("releases = %d, manager = %p", releases.Load(), h.b.getEngineMgr())
	}
	if got := h.receipt(t, r.RequestID); got.Phase != settings.SwitchFailedInterrupted || !got.RecoveryRequired {
		t.Fatalf("interrupted receipt = %+v", got)
	}
	if !h.b.engineSwitch.resolved[r.RequestID] || h.b.engineSwitch.recoveryErr != nil {
		t.Fatal("successful retry did not resolve conservative cleanup")
	}
	if steps := switchSteps(t, h.engine); len(steps) != 0 {
		t.Fatalf("drain recovery executed forward steps: %v", steps)
	}
}

func TestSwitchRecoveryExhaustionKeepsManagerHiddenAndReportsBlocked(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	var inspections atomic.Int32
	h.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			inspections.Add(1)
		}
		return nil, "inspection unavailable"
	})
	if h.b.recoverReplacementEngineMgr(w) || h.b.getEngineMgr() != nil {
		t.Fatal("exhausted recovery exposed the manager")
	}
	if inspections.Load() != 3 || h.b.engineSwitch.err != nil || h.b.engineSwitch.recoveryErr == nil {
		t.Fatalf("inspections = %d, permanent = %v, retryable = %v", inspections.Load(), h.b.engineSwitch.err, h.b.engineSwitch.recoveryErr)
	}
	h.b.stickyErrorsMu.Lock()
	report, ok := h.b.stickyErrors[engineSwitchRecoveryBlockedID]
	h.b.stickyErrorsMu.Unlock()
	if !ok || report.Severity != "error" || report.Message == "" {
		t.Fatalf("blocked recovery report = %+v, found = %v", report, ok)
	}
}

func TestSwitchRecoveryNeverClearsJournalCorruption(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	corruption := &engineSwitchRecoveryError{Err: errors.New("invalid journal")}
	h.b.engineSwitch.err = corruption
	if h.b.recoverReplacementEngineMgr(w) || h.b.getEngineMgr() != nil {
		t.Fatal("journal corruption exposed the manager")
	}
	if !errors.Is(h.b.engineSwitch.err, corruption) {
		t.Fatalf("journal corruption was replaced: %v", h.b.engineSwitch.err)
	}
	if calls := h.engine.snapshot(); len(calls) != 0 {
		t.Fatalf("corrupt journal allowed recovery RPCs: %v", calls)
	}
}

func TestSwitchRecoveryNeverRetriesManagerStorageUncertainty(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	var inspections atomic.Int32
	h.engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
		if method == settings.SwitchPrepareMethod {
			inspections.Add(1)
		}
		return settings.SwitchPrepareResult{Error: &settings.SwitchError{
			Code: settings.SwitchOperationErrorCode, Kind: settings.SwitchErrorStorage, Message: "checkpoint durability uncertain",
		}}, ""
	})
	if h.b.recoverReplacementEngineMgr(w) {
		t.Fatal("manager storage uncertainty was accepted")
	}
	if !switchRecoveryPermanent(h.b.engineSwitch.err) || inspections.Load() != 1 {
		t.Fatalf("permanent = %v, inspections = %d", h.b.engineSwitch.err, inspections.Load())
	}
	if h.b.recoverReplacementEngineMgr(w) || inspections.Load() != 1 || h.b.getEngineMgr() != nil {
		t.Fatal("subsequent recovery retried or cleared storage uncertainty")
	}
}

func TestSwitchRecoveryRetryDoesNotFollowReplacedManager(t *testing.T) {
	h := newSwitchHarness(t)
	w := hideSwitchRecoveryManager(t, h.b)
	peer, replacement := newFakeWorkerPipe(t)
	newWorker := &rpcWorker{peer: peer}
	var inspections atomic.Int32
	h.engine.setAnswer(func(_ string, _ json.RawMessage) (any, string) {
		inspections.Add(1)
		h.b.setEngineMgr(newWorker)
		return nil, "old manager disconnected"
	})
	if h.b.recoverReplacementEngineMgr(w) {
		t.Fatal("obsolete recovery published a replacement")
	}
	if inspections.Load() != 1 || len(replacement.snapshot()) != 0 || h.b.getEngineMgr() != nil {
		t.Fatalf("old inspections = %d, replacement calls = %v, published = %p", inspections.Load(), replacement.snapshot(), h.b.getEngineMgr())
	}
}
