// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
)

// intentRecorder captures engine:intent-changed payloads.
type intentRecorder struct {
	mu   sync.Mutex
	seen []intentParams
}

func (r *intentRecorder) emit(method string, params any) {
	if method != intentChangedMethod {
		return
	}
	data, _ := json.Marshal(params)
	var p intentParams
	_ = json.Unmarshal(data, &p)
	r.mu.Lock()
	r.seen = append(r.seen, p)
	r.mu.Unlock()
}

func (r *intentRecorder) last() (intentParams, int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.seen) == 0 {
		return intentParams{}, 0
	}
	return r.seen[len(r.seen)-1], len(r.seen)
}

func wakeTestExecutor(t *testing.T, baseDir string, rec *intentRecorder) *Executor {
	t.Helper()
	reg := NewRegistry()
	m := testEngineManifest(fakeEngineBin)
	reg.engines[m.Engine] = m
	emit := func(string, any) {}
	if rec != nil {
		emit = rec.emit
	}
	e := NewExecutor(reg, NewReporter(nil), emit, baseDir)
	t.Cleanup(e.StopAll)
	return e
}

func TestWakeRefusesUnknownAndOffIntent(t *testing.T) {
	e := wakeTestExecutor(t, t.TempDir(), nil)
	if err := e.Wake(context.Background(), "fake"); !errors.Is(err, errIntentOff) {
		t.Fatalf("wake with no saved intent = %v, want errIntentOff", err)
	}
	if err := e.setDesiredEnabled("fake", false); err != nil {
		t.Fatal(err)
	}
	if err := e.Wake(context.Background(), "fake"); !errors.Is(err, errIntentOff) {
		t.Fatalf("wake with intent Off = %v, want errIntentOff", err)
	}
	if st, _ := e.Status("fake"); st.Running {
		t.Fatal("wake started an engine saved Off")
	}
}

func TestSleepAndWakeKeepSavedIntent(t *testing.T) {
	baseDir := t.TempDir()
	rec := &intentRecorder{}
	e := wakeTestExecutor(t, baseDir, rec)
	ctx := context.Background()
	if err := e.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	if intent, n := rec.last(); n == 0 || !intent.EnabledByEngine["fake"] {
		t.Fatalf("start did not push intent On: %+v (%d pushes)", intent, n)
	}
	_, pushesBefore := rec.last()

	if err := e.Sleep("fake"); err != nil {
		t.Fatalf("sleep: %v", err)
	}
	if st, _ := e.Status("fake"); st.Running {
		t.Fatal("sleep left the engine running")
	}
	if enabled, known, err := e.desired.get("fake"); err != nil || !known || !enabled {
		t.Fatalf("intent after sleep = (%v, %v, %v), want known On", enabled, known, err)
	}

	if err := e.Wake(ctx, "fake"); err != nil {
		t.Fatalf("wake: %v", err)
	}
	if st, _ := e.Status("fake"); !st.Running {
		t.Fatal("wake did not start an engine saved On")
	}
	if _, n := rec.last(); n != pushesBefore {
		t.Fatalf("sleep/wake pushed intent changes (%d → %d); they must be intent-neutral", pushesBefore, n)
	}
	intent, err := e.Intent()
	if err != nil || !intent.EnabledByEngine["fake"] {
		t.Fatalf("Intent() = %+v, %v", intent, err)
	}

	if err := e.Stop("fake"); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if intent, _ := rec.last(); intent.EnabledByEngine["fake"] {
		t.Fatal("explicit stop did not push intent Off")
	}
}

func TestWakeRefusedAfterShutdown(t *testing.T) {
	e := wakeTestExecutor(t, t.TempDir(), nil)
	if err := e.setDesiredEnabled("fake", true); err != nil {
		t.Fatal(err)
	}
	e.StopAll()
	if err := e.Wake(context.Background(), "fake"); err == nil {
		t.Fatal("wake succeeded after StopAll")
	}
	if st, _ := e.Status("fake"); st.Running {
		t.Fatal("wake started an engine during shutdown")
	}
}
