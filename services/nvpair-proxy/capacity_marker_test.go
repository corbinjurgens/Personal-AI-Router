// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/nodepolicy"
)

func readCapacity(t *testing.T, path string) capacityDocument {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var doc capacityDocument
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	return doc
}

func TestStrictQuarantineSurvivesRestartUntilEvidence(t *testing.T) {
	for _, evidence := range []strictTerminalEvidence{strictAuthoritativeIdle, strictVerifiedReplacement} {
		t.Run(string(evidence), func(t *testing.T) {
			h := newStrictHarness(t)
			ticket := mustStrictTicket(t, h.a)
			if err := ticket.BeforeStrictDispatch(); err != nil {
				t.Fatal(err)
			}
			if doc := readCapacity(t, h.a.strict.path); doc.Version != 1 || len(doc.Slots) != 1 {
				t.Fatalf("document=%+v", doc)
			}
			ticket.Finish(strictUncertain)
			ticket.Finish(strictQualifiedComplete)
			restarted := newAdmHarness(t)
			configureStrict(t, restarted.a, h.a.strict.path)
			restarted.a.loadStrictCapacity()
			c := useStrictClock(restarted.a)
			c.Advance(24 * time.Hour)
			req := strictRequest()
			req.QueueBudgetMs = 0
			if _, err := restarted.a.AdmitStrict(context.Background(), "ollama", "m", req); !errors.Is(err, errStrictOccupied) {
				t.Fatalf("error=%v", err)
			}
			if err := restarted.a.reconcileStrictCapacity("ollama", "m", strictQualifiedComplete); err == nil {
				t.Fatal("nonauthoritative reconciliation accepted")
			}
			if err := restarted.a.reconcileStrictCapacity("ollama", "m", evidence); err != nil {
				t.Fatal(err)
			}
			if doc := readCapacity(t, h.a.strict.path); len(doc.Slots) != 0 {
				t.Fatalf("document=%+v", doc)
			}
			next := mustStrictTicket(t, restarted.a)
			next.Finish(strictNotDispatched)
		})
	}
}

func TestStrictDispatchPersistenceFailurePreventsDispatch(t *testing.T) {
	h := newStrictHarness(t)
	ticket := mustStrictTicket(t, h.a)
	failure := errors.New("injected storage failure")
	h.a.strict.write = func(string, []byte, os.FileMode) error { return failure }
	if err := ticket.BeforeStrictDispatch(); !errors.Is(err, failure) {
		t.Fatalf("error=%v", err)
	}
	ticket.Finish(strictUncertain)
	if h.a.currentActive() != 1 {
		t.Fatal("persistence failure released quarantine")
	}
}

func TestStrictLandedUncertainWritePreventsDispatchAndRecovers(t *testing.T) {
	h := newStrictHarness(t)
	ticket := mustStrictTicket(t, h.a)
	h.a.strict.write = durablefile.Replacer{SyncDir: func(string) error { return errors.New("injected directory sync failure") }}.WriteFile
	if err := ticket.BeforeStrictDispatch(); !errors.Is(err, durablefile.ErrReplaced) {
		t.Fatalf("error=%v", err)
	}
	ticket.Finish(strictUncertain)
	restarted := newAdmHarness(t)
	configureStrict(t, restarted.a, h.a.strict.path)
	restarted.a.loadStrictCapacity()
	if restarted.a.currentActive() != 1 {
		t.Fatal("landed marker not adopted")
	}
}

func TestStrictClearFailureRetainsCapacity(t *testing.T) {
	h := newStrictHarness(t)
	ticket := mustStrictTicket(t, h.a)
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatal(err)
	}
	h.a.strict.write = func(string, []byte, os.FileMode) error { return errors.New("injected clear failure") }
	ticket.Finish(strictQualifiedComplete)
	if h.a.currentActive() != 1 || len(readCapacity(t, h.a.strict.path).Slots) != 1 {
		t.Fatal("failed durable clear released capacity")
	}
	h.a.strict.write = durablefile.WriteFile
	if err := h.a.reconcileStrictCapacity("ollama", "m", strictAuthoritativeIdle); err != nil {
		t.Fatal(err)
	}
	if h.a.currentActive() != 0 {
		t.Fatal("reconciliation leaked slot")
	}
}

func TestStrictCorruptCapacityFailsClosed(t *testing.T) {
	for _, tc := range []struct{ name, document string }{
		{"unsupported version", `{"version":99,"slots":[]}`},
		{"missing version", `{"slots":[]}`},
		{"null version", `{"version":null,"slots":[]}`},
		{"missing slots", `{"version":1}`},
		{"null slots", `{"version":1,"slots":null}`},
		{"unknown engine", `{"version":1,"slots":[{"id":1,"engine":"unknown","model":"m:latest"}]}`},
		{"duplicate id", `{"version":1,"slots":[{"id":1,"engine":"ollama","model":"m:latest"},{"id":1,"engine":"ollama","model":"n:latest"}]}`},
		{"zero id", `{"version":1,"slots":[{"id":0,"engine":"ollama","model":"m:latest"}]}`},
		{"invalid model", `{"version":1,"slots":[{"id":1,"engine":"ollama","model":" "}]}`},
		{"unnormalized model", `{"version":1,"slots":[{"id":1,"engine":"ollama","model":"m"}]}`},
		{"reserved model", `{"version":1,"slots":[{"id":1,"engine":"lmstudio","model":"weak"}]}`},
		{"invalid later slot", `{"version":1,"slots":[{"id":1,"engine":"ollama","model":"m:latest"},{"id":2,"engine":"unknown","model":"n"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newStrictHarness(t)
			if err := os.WriteFile(h.a.strict.path, []byte(tc.document), 0600); err != nil {
				t.Fatal(err)
			}
			h.a.loadStrictCapacity()
			if _, err := h.a.AdmitStrict(context.Background(), "ollama", "m", strictRequest()); !errors.Is(err, errStrictUnavailable) {
				t.Fatalf("error=%v", err)
			}
			if h.a.currentActive() != 0 || len(h.a.strict.markers) != 0 || len(h.a.inflight) != 0 || h.a.nextTicket != 0 {
				t.Fatal("partially adopted corrupt document")
			}
		})
	}
}

func TestStrictValidCapacityRestoresAllSlots(t *testing.T) {
	h := newStrictHarness(t)
	if err := os.WriteFile(h.a.strict.path, []byte(`{"version":1,"slots":[{"id":7,"engine":"ollama","model":"m:latest"},{"id":9,"engine":"lmstudio","model":"other"}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	h.a.loadStrictCapacity()
	if h.a.strict.storageErr != nil || h.a.currentActive() != 2 || len(h.a.strict.markers) != 2 || h.a.nextTicket != 9 || h.a.inflight[admKey{engine: "ollama", model: "m:latest"}] != 1 || h.a.inflight[admKey{engine: "lmstudio", model: "other"}] != 1 {
		t.Fatalf("restoration error=%v active=%d next=%d markers=%v counts=%v", h.a.strict.storageErr, h.a.currentActive(), h.a.nextTicket, h.a.strict.markers, h.a.inflight)
	}
}

func TestStrictSpacedModelWriterRecoveryRoundTrip(t *testing.T) {
	h := newStrictHarness(t)
	const model = "folder/model name"
	capability := nodepolicy.StrictAdmissionCapability{Version: 1, Models: []nodepolicy.StrictAdmissionModel{{Engine: "lmstudio", Model: model, MaxConcurrent: 1, Identity: strictRequest().Identity}}}
	if err := h.a.installStrictCapability(capability); err != nil {
		t.Fatal(err)
	}
	h.a.setResidency(map[string][]string{"lmstudio": {model}})
	ticket, err := h.a.AdmitStrict(context.Background(), "lmstudio", model, strictRequest())
	if err != nil {
		t.Fatal(err)
	}
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatal(err)
	}
	restarted := newStrictHarness(t)
	restarted.a.strict.path = h.a.strict.path
	if err := restarted.a.installStrictCapability(capability); err != nil {
		t.Fatal(err)
	}
	restarted.a.setResidency(map[string][]string{"lmstudio": {model}})
	restarted.a.loadStrictCapacity()
	if restarted.a.strict.storageErr != nil || restarted.a.currentActive() != 1 {
		t.Fatalf("recovery error=%v active=%d", restarted.a.strict.storageErr, restarted.a.currentActive())
	}
	req := strictRequest()
	req.QueueBudgetMs = 0
	if next, err := restarted.a.AdmitStrict(context.Background(), "lmstudio", model, req); next != nil || !errors.Is(err, errStrictOccupied) {
		t.Fatalf("ticket=%v error=%v", next, err)
	}
	ticket.Finish(strictQualifiedComplete)
}

func TestStrictModelControlsRejectedConsistently(t *testing.T) {
	h := newStrictHarness(t)
	const model = "folder/model\x01name"
	if err := h.a.installStrictCapability(nodepolicy.StrictAdmissionCapability{Version: 1, Models: []nodepolicy.StrictAdmissionModel{{Engine: "lmstudio", Model: model, MaxConcurrent: 1, Identity: strictRequest().Identity}}}); err == nil {
		t.Fatal("control accepted by installer")
	}
	h.a.strict.markers[1] = capacityMarker{ID: 1, Engine: "lmstudio", Model: model}
	if err := h.a.persistStrictCapacityLocked(); err == nil {
		t.Fatal("control accepted by writer")
	}
	b, err := json.Marshal(capacityDocument{Version: 1, Slots: []capacityMarker{h.a.strict.markers[1]}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(h.a.strict.path, b, 0600); err != nil {
		t.Fatal(err)
	}
	h.a.strict.markers = make(map[uint64]capacityMarker)
	h.a.loadStrictCapacity()
	if h.a.strict.storageErr == nil || h.a.currentActive() != 0 {
		t.Fatal("control accepted by loader")
	}
}

func TestStrictAggregateMarkerLimitPreventsDispatch(t *testing.T) {
	h := newStrictHarness(t)
	ticket := mustStrictTicket(t, h.a)
	for id := uint64(2); id <= maxStrictCapacitySlots+1; id++ {
		h.a.strict.markers[id] = capacityMarker{ID: id, Engine: "lmstudio", Model: "other"}
	}
	writes := 0
	h.a.strict.write = func(string, []byte, os.FileMode) error { writes++; return nil }
	if err := ticket.BeforeStrictDispatch(); !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("dispatch error=%v", err)
	}
	if writes != 0 || len(h.a.strict.markers) != maxStrictCapacitySlots || h.a.currentActive() != 0 {
		t.Fatalf("writes=%d markers=%d active=%d", writes, len(h.a.strict.markers), h.a.currentActive())
	}
	// The writer independently rejects an oversized in-memory quarantine.
	h.a.strict.markers[maxStrictCapacitySlots+2] = capacityMarker{ID: maxStrictCapacitySlots + 2, Engine: "lmstudio", Model: "other"}
	if err := h.a.persistStrictCapacityLocked(); err == nil || writes != 0 {
		t.Fatalf("oversized persistence error=%v writes=%d", err, writes)
	}
}

func TestStrictFailedRecoveryReconciliationPreservesBytes(t *testing.T) {
	for _, scenario := range []string{"corrupt document", "transient read failure"} {
		t.Run(scenario, func(t *testing.T) {
			h := newStrictHarness(t)
			original := []byte(`{"version":1,"slots":[{"id":7,"engine":"ollama","model":"m:latest"}]}`)
			if scenario == "corrupt document" {
				original = []byte(`{"version":1,"slots":[`)
			}
			if err := os.WriteFile(h.a.strict.path, original, 0600); err != nil {
				t.Fatal(err)
			}
			failure := errors.New("transient read failure")
			if scenario == "transient read failure" {
				h.a.strict.read = func(string) ([]byte, error) { return nil, failure }
			}
			h.a.loadStrictCapacity()
			loadErr := h.a.strict.storageErr
			if loadErr == nil {
				t.Fatal("load did not fail")
			}
			h.a.strict.read = os.ReadFile
			writes := 0
			h.a.strict.write = func(string, []byte, os.FileMode) error { writes++; return nil }
			if err := h.a.reconcileStrictCapacity("lmstudio", "unrelated", strictAuthoritativeIdle); !errors.Is(err, loadErr) {
				t.Fatalf("reconcile error=%v want=%v", err, loadErr)
			}
			got, err := os.ReadFile(h.a.strict.path)
			if err != nil {
				t.Fatal(err)
			}
			if writes != 0 || !bytes.Equal(got, original) {
				t.Fatal("reconciliation rewrote untrusted storage")
			}
			if _, err := h.a.AdmitStrict(context.Background(), "ollama", "m", strictRequest()); !errors.Is(err, errStrictUnavailable) {
				t.Fatalf("admission error=%v", err)
			}
			restarted := newStrictHarness(t)
			restarted.a.strict.path = h.a.strict.path
			restarted.a.loadStrictCapacity()
			req := strictRequest()
			req.QueueBudgetMs = 0
			_, err = restarted.a.AdmitStrict(context.Background(), "ollama", "m", req)
			if scenario == "corrupt document" {
				if !errors.Is(err, errStrictUnavailable) {
					t.Fatalf("restart admission error=%v", err)
				}
			} else if !errors.Is(err, errStrictOccupied) || restarted.a.currentActive() != 1 {
				t.Fatalf("restart lost unresolved slot: error=%v active=%d", err, restarted.a.currentActive())
			}
		})
	}
}

func TestStrictUnrelatedReconciliationDoesNotWrite(t *testing.T) {
	h := newStrictHarness(t)
	ticket := mustStrictTicket(t, h.a)
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatal(err)
	}
	writes := 0
	h.a.strict.write = func(string, []byte, os.FileMode) error { writes++; return errors.New("unexpected write") }
	if err := h.a.reconcileStrictCapacity("lmstudio", "unrelated", strictAuthoritativeIdle); err != nil {
		t.Fatal(err)
	}
	if writes != 0 || h.a.currentActive() != 1 {
		t.Fatalf("writes=%d active=%d", writes, h.a.currentActive())
	}
	h.a.strict.write = durablefile.WriteFile
	ticket.Finish(strictQualifiedComplete)
}
