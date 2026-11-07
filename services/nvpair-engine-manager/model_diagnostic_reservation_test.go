// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"

	"nvpair-shared/nodepolicy"
)

func diagnosticReservationFixture(t *testing.T) (inspectionFixture, diagnosticReserveParams) {
	t.Helper()
	f := newAppliedKeyFixture(t)
	f.ex.nodeProjection.set(nodeProjection{})
	ctx := f.ex.modelMemoryContext(context.Background(), "llamacpp", modelBufferTestTargetMemory(), false)
	identity, err := f.ex.inspectModelArtifacts(ctx, f.request)
	if err != nil {
		t.Fatalf("inspect applied profile: %v", err)
	}
	fence, err := f.ex.diagnosticEffectsFence(ctx)
	if err != nil {
		t.Fatalf("observe effects: %v", err)
	}
	p := diagnosticReserveParams{RunID: "reservation-test", Engine: "llamacpp", ModelID: f.request.ModelID,
		ProfileKey: f.request.ProfileKey, ConfigurationKey: identity.ConfigurationKey,
		EffectsFence: fence, TargetMemory: modelBufferTestTargetMemory()}
	if p.ConfigurationKey == "" {
		p.ExecutionKey = identity.ExecutionKey
	}
	if p.ConfigurationKey == "" && p.ExecutionKey == "" {
		t.Fatal("fixture has no applied identity")
	}
	return f, p
}

func reserveDiagnosticFixture(t *testing.T, f inspectionFixture, p diagnosticReserveParams) diagnosticReservationResult {
	t.Helper()
	r, err := f.ex.reserveDiagnostic(context.Background(), p)
	if err != nil {
		t.Fatalf("reserve diagnostic: %v", err)
	}
	if r.Token == "" || r.RunID != p.RunID || r.State != "reserved" {
		t.Fatalf("reservation result = %+v", r)
	}
	return r
}

func TestDiagnosticReservationIdenticalRetryRetainsAuthority(t *testing.T) {
	f, p := diagnosticReservationFixture(t)
	first := reserveDiagnosticFixture(t, f, p)
	if retry := reserveDiagnosticFixture(t, f, p); retry != first {
		t.Fatalf("retry = %+v, first = %+v", retry, first)
	}
	changed := p
	changed.EffectsFence += "changed"
	if _, err := f.ex.reserveDiagnostic(context.Background(), changed); err == nil {
		t.Fatal("same run accepted different effects")
	}
	if err := f.ex.releaseDiagnostic(diagnosticReleaseParams{p.RunID, first.Token}); err != nil {
		t.Fatalf("release original authority: %v", err)
	}
	done, err := f.ex.admitMutation("llamacpp")
	if err != nil {
		t.Fatalf("retry leaked admission after release: %v", err)
	}
	done()
}

func TestDiagnosticReservationMismatchHasNoEffects(t *testing.T) {
	test := func(name string, change func(*diagnosticReserveParams)) {
		t.Run(name, func(t *testing.T) {
			f, p := diagnosticReservationFixture(t)
			before, err := os.ReadFile(f.preset)
			if err != nil {
				t.Fatal(err)
			}
			change(&p)
			if _, err := f.ex.reserveDiagnostic(context.Background(), p); err == nil {
				t.Fatal("stale preview reserved execution")
			}
			after, err := os.ReadFile(f.preset)
			if err != nil {
				t.Fatal(err)
			}
			if string(before) != string(after) {
				t.Fatal("reservation changed saved profile")
			}
			if f.st.running || f.st.proc != nil {
				t.Fatal("reservation started an engine")
			}
			// The inspection fixture also rejects HTTP, content mutations and runtime notifications.
			done, err := f.ex.admitMutation("llamacpp")
			if err != nil {
				t.Fatalf("failed comparison retained exclusion: %v", err)
			}
			done()
		})
	}
	test("configuration key", func(p *diagnosticReserveParams) { p.ExecutionKey = ""; p.ConfigurationKey = "stale" })
	test("execution key", func(p *diagnosticReserveParams) { p.ConfigurationKey = ""; p.ExecutionKey = "stale" })
	test("effects fence", func(p *diagnosticReserveParams) { p.EffectsFence = "stale" })
}

func TestDiagnosticReservationRefusesAdmittedWork(t *testing.T) {
	t.Run("mutation before lifecycle lock", func(t *testing.T) {
		f, p := diagnosticReservationFixture(t)
		done, err := f.ex.admitMutation("llamacpp")
		if err != nil {
			t.Fatal(err)
		}
		defer done()
		if _, err := f.ex.reserveDiagnostic(context.Background(), p); err == nil {
			t.Fatal("reserved over admitted mutation")
		}
	})
	t.Run("registered serving load", func(t *testing.T) {
		f, p := diagnosticReservationFixture(t)
		_, done := registerServingLoad(context.Background(), f.st)
		defer done()
		if _, err := f.ex.reserveDiagnostic(context.Background(), p); err == nil {
			t.Fatal("reserved over registered load")
		}
	})
}

func TestDiagnosticReservationBlocksOrdinaryExecution(t *testing.T) {
	f, p := diagnosticReservationFixture(t)
	r := reserveDiagnosticFixture(t, f, p)
	defer func() {
		if err := f.ex.releaseDiagnostic(diagnosticReleaseParams{p.RunID, r.Token}); err != nil {
			t.Error(err)
		}
	}()
	checkBusy := func(name string, err error) {
		t.Helper()
		var busy *diagnosticBusyError
		if !errors.As(err, &busy) {
			t.Fatalf("%s error = %v, want diagnostic busy", name, err)
		}
	}
	checkBusy("wake", f.ex.Wake(context.Background(), "llamacpp"))
	_, release, err := f.ex.guardServingLoad(context.Background(), f.st, "load_model", json.RawMessage(`{"model":"exact-alias"}`))
	defer release()
	checkBusy("serving load", err)
	done, err := f.ex.admitMutation("other-engine")
	if done != nil {
		defer done()
	}
	checkBusy("other engine mutation", err)
}

func TestDiagnosticReservationAuthorityAndConfirmedRelease(t *testing.T) {
	f, p := diagnosticReservationFixture(t)
	r := reserveDiagnosticFixture(t, f, p)
	if _, err := f.ex.diagnosticContext(context.Background(), p.Engine, p.RunID, "wrong"); err == nil {
		t.Fatal("wrong token accepted")
	}
	ctx, err := f.ex.diagnosticContext(context.Background(), p.Engine, p.RunID, r.Token)
	if err != nil {
		t.Fatal(err)
	}
	if done, err := f.ex.admitMutationContext(ctx, "other-engine"); err == nil {
		done()
		t.Fatal("engine-scoped authority escaped to another engine")
	}
	done, err := f.ex.admitMutationContext(ctx, p.Engine)
	if err != nil {
		t.Fatal(err)
	}
	release := diagnosticReleaseParams{p.RunID, r.Token}
	if err := f.ex.confirmDiagnosticWorkStopped(p.RunID, r.Token); err == nil {
		t.Fatal("confirmed while mutation active")
	}
	if err := f.ex.releaseDiagnostic(release); err == nil {
		t.Fatal("released active work")
	}
	done()
	if err := f.ex.releaseDiagnostic(release); err == nil {
		t.Fatal("retiring a mutation implicitly confirmed physical cleanup")
	}
	if err := f.ex.confirmDiagnosticWorkStopped(p.RunID, r.Token); err != nil {
		t.Fatal(err)
	}
	if err := f.ex.releaseDiagnostic(release); err != nil {
		t.Fatal(err)
	}
	if done, err := f.ex.admitMutationContext(ctx, p.Engine); err == nil {
		done()
		t.Fatal("released token retained authority")
	}
}

func TestDiagnosticReservationConfiguredPauseAuthorityCannotBypassHalt(t *testing.T) {
	f, p := diagnosticReservationFixture(t)
	projection := nodeProjection{PausedLocally: true, ManagementBlocked: true,
		Policy: nodepolicy.Advertisement{PauseMode: nodepolicy.PauseConfigured}}
	f.ex.nodeProjection.set(projection)
	r := reserveDiagnosticFixture(t, f, p)
	ctx, err := f.ex.diagnosticContext(context.Background(), p.Engine, p.RunID, r.Token)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.ex.diagnosticActivationVeto(ctx, p.Engine); err != nil {
		t.Fatalf("configured pause vetoed measured engine authority: %v", err)
	}
	other, err := f.ex.diagnosticContext(context.Background(), "other-engine", p.RunID, r.Token)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.ex.diagnosticActivationVeto(other, "other-engine"); err == nil {
		t.Fatal("cleanup authority activated an unmeasured engine")
	}
	projection.Policy.PauseMode = nodepolicy.PauseImmediate
	f.ex.nodeProjection.set(projection)
	if err := f.ex.diagnosticActivationVeto(ctx, p.Engine); err == nil {
		t.Fatal("reservation bypassed Halt")
	}
	if got := f.ex.nodeProjection.advertisement(); got == nil || got.PauseMode != nodepolicy.PauseImmediate {
		t.Fatalf("activation gate changed pause policy: %+v", got)
	}
	if err := f.ex.releaseDiagnostic(diagnosticReleaseParams{p.RunID, r.Token}); err != nil {
		t.Fatal(err)
	}
}
