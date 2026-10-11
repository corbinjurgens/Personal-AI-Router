// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/shirou/gopsutil/v4/process"
)

func TestServiceInspectionErrorWaitsForProvenDisappearance(t *testing.T) {
	inspectionErr := errors.New("executable inaccessible")
	calls := 0
	probe := func(ctx context.Context, pid int32) (bool, error) {
		if pid != 123 || ctx.Err() != nil {
			t.Fatalf("unexpected existence probe: pid=%d context=%v", pid, ctx.Err())
		}
		calls++
		return calls == 1, nil
	}
	if err := serviceInspectionErrorWithProbe(context.Background(), 123, "executable", inspectionErr, probe); err != nil {
		t.Fatalf("proven disappearance retains inspection error: %v", err)
	}
	if calls != 2 {
		t.Fatalf("existence probes=%d, want initial presence followed by proven absence", calls)
	}
}

func TestServiceInspectionErrorRetainsLiveProcessError(t *testing.T) {
	inspectionErr := errors.New("executable inaccessible")
	probe := func(context.Context, int32) (bool, error) { return true, nil }
	err := serviceInspectionErrorWithProbe(context.Background(), 123, "executable", inspectionErr, probe)
	if !errors.Is(err, inspectionErr) {
		t.Fatalf("live uninspectable process error=%v, want original inspection error", err)
	}
}

func TestServiceInspectionErrorResolvesProbeAmbiguityOnlyAfterAbsence(t *testing.T) {
	inspectionErr := errors.New("executable inaccessible")
	existenceErr := errors.New("existence unavailable")
	calls := 0
	probe := func(ctx context.Context, _ int32) (bool, error) {
		if ctx.Err() != nil {
			t.Fatalf("absence was not observed under an active context: %v", ctx.Err())
		}
		calls++
		if calls == 1 {
			return false, existenceErr
		}
		return false, nil
	}
	if err := serviceInspectionErrorWithProbe(context.Background(), 123, "executable", inspectionErr, probe); err != nil {
		t.Fatalf("successful independent absence did not resolve earlier ambiguity: %v", err)
	}
	if calls != 2 {
		t.Fatalf("existence probes=%d, want ambiguity followed by successful absence", calls)
	}
}

func TestServiceInspectionErrorRetainsExistenceErrors(t *testing.T) {
	inspectionErr := errors.New("executable inaccessible")
	existenceErr := errors.New("existence unavailable")
	t.Run("apparent absence with a probe error is ambiguous", func(t *testing.T) {
		probe := func(context.Context, int32) (bool, error) { return false, existenceErr }
		err := serviceInspectionErrorWithProbe(context.Background(), 123, "executable", inspectionErr, probe)
		if !errors.Is(err, inspectionErr) || !errors.Is(err, existenceErr) {
			t.Fatalf("ambiguous absence error=%v, want both original errors", err)
		}
	})
	t.Run("later presence does not erase an earlier probe error", func(t *testing.T) {
		calls := 0
		probe := func(context.Context, int32) (bool, error) {
			calls++
			if calls == 1 {
				return false, existenceErr
			}
			return true, nil
		}
		err := serviceInspectionErrorWithProbe(context.Background(), 123, "executable", inspectionErr, probe)
		if calls < 2 || !errors.Is(err, inspectionErr) || !errors.Is(err, existenceErr) {
			t.Fatalf("live process erased earlier probe error: calls=%d error=%v", calls, err)
		}
	})
}

func TestServiceInspectionErrorPreservesCallerCancellation(t *testing.T) {
	inspectionErr := errors.New("executable inaccessible")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	probe := func(context.Context, int32) (bool, error) {
		cancel()
		return false, nil
	}
	err := serviceInspectionErrorWithProbe(ctx, 123, "executable", inspectionErr, probe)
	if !errors.Is(err, inspectionErr) || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled existence probe error=%v, want original error and cancellation", err)
	}
}

func TestServiceInspectionErrorPreservesCallerDeadline(t *testing.T) {
	inspectionErr := errors.New("executable inaccessible")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	probe := func(ctx context.Context, _ int32) (bool, error) {
		<-ctx.Done()
		return false, ctx.Err()
	}
	err := serviceInspectionErrorWithProbe(ctx, 123, "executable", inspectionErr, probe)
	if !errors.Is(err, inspectionErr) || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expired existence probe error=%v, want original error and caller deadline", err)
	}
}

func TestServiceInspectionErrorPreservesEnumerationAndCallerErrors(t *testing.T) {
	enumerationErr := errors.New("enumeration unavailable")
	inspectionErr := errors.New("injected inspection failure")
	in := &serviceProcessInventory{enumerationError: enumerationErr, scanError: inspectionErr}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	owned, err := in.scanContext(ctx)
	if len(owned) != 0 || !errors.Is(err, enumerationErr) || !errors.Is(err, inspectionErr) || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled empty scan: identities=%+v error=%v, want no identities and all enumeration, inspection and caller errors", owned, err)
	}
}

type serviceKillProbe struct {
	path                string
	created             int64
	exeErr, createdErr  error
	signalErr           error
	exeCalls, timeCalls int
	signals             int
	afterCreation       func()
}

func (p *serviceKillProbe) ExeWithContext(context.Context) (string, error) {
	p.exeCalls++
	return p.path, p.exeErr
}

func (p *serviceKillProbe) CreateTimeWithContext(context.Context) (int64, error) {
	p.timeCalls++
	if p.afterCreation != nil {
		p.afterCreation()
	}
	return p.created, p.createdErr
}

func (p *serviceKillProbe) KillWithContext(context.Context) error {
	p.signals++
	return p.signalErr
}

func TestServiceInspectionErrorDuringKillRevalidation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nvpair-service")
	path, err := serviceExecutablePath(path)
	if err != nil {
		t.Fatalf("canonical fixture identity: %v", err)
	}
	want := serviceProcessIdentity{PID: 123, Path: path, Created: 456}
	inspectionErr := errors.New("proc_pidpath returned 0")
	probeErr := errors.New("stopped-state unavailable")
	newInventory := func() *serviceProcessInventory {
		return &serviceProcessInventory{paths: []string{path}, known: map[int32]serviceProcessIdentity{want.PID: want}}
	}
	openProbe := func(t *testing.T, p *serviceKillProbe) func(context.Context, int32) (serviceKillProcess, error) {
		t.Helper()
		return func(ctx context.Context, pid int32) (serviceKillProcess, error) {
			if pid != want.PID || ctx.Err() != nil {
				t.Fatalf("open process pid=%d context=%v", pid, ctx.Err())
			}
			return p, nil
		}
	}

	t.Run("metadata failure resolves only after independently proven stop", func(t *testing.T) {
		p := &serviceKillProbe{exeErr: inspectionErr}
		calls := 0
		stopped := func(ctx context.Context, pid int32) (bool, error) {
			if pid != want.PID || ctx.Err() != nil {
				t.Fatalf("independent stopped probe pid=%d context=%v", pid, ctx.Err())
			}
			if _, bounded := ctx.Deadline(); !bounded {
				t.Fatal("revalidation probe has no deadline")
			}
			calls++
			return calls == 2, nil
		}
		err := newInventory().killContextWithProbes(context.Background(), want, openProbe(t, p), stopped)
		if err != nil || calls != 2 || p.exeCalls != 1 || p.timeCalls != 0 || p.signals != 0 {
			t.Fatalf("stopped revalidation: error=%v probes=%d exe=%d time=%d signals=%d", err, calls, p.exeCalls, p.timeCalls, p.signals)
		}
	})

	t.Run("live process retains executable metadata error without signalling", func(t *testing.T) {
		p := &serviceKillProbe{exeErr: inspectionErr}
		calls := 0
		stopped := func(context.Context, int32) (bool, error) { calls++; return false, nil }
		err := newInventory().killContextWithProbes(context.Background(), want, openProbe(t, p), stopped)
		if !errors.Is(err, inspectionErr) || calls == 0 || p.exeCalls != 1 || p.timeCalls != 0 || p.signals != 0 {
			t.Fatalf("live revalidation: error=%v probes=%d exe=%d time=%d signals=%d", err, calls, p.exeCalls, p.timeCalls, p.signals)
		}
	})

	t.Run("apparent stop with probe failure retains both errors", func(t *testing.T) {
		p := &serviceKillProbe{exeErr: inspectionErr}
		stopped := func(context.Context, int32) (bool, error) { return true, probeErr }
		err := newInventory().killContextWithProbes(context.Background(), want, openProbe(t, p), stopped)
		if !errors.Is(err, inspectionErr) || !errors.Is(err, probeErr) || p.signals != 0 {
			t.Fatalf("ambiguous revalidation: error=%v signals=%d", err, p.signals)
		}
	})

	t.Run("cancelled stopped probe cannot resolve metadata error", func(t *testing.T) {
		p := &serviceKillProbe{exeErr: inspectionErr}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		stopped := func(context.Context, int32) (bool, error) { cancel(); return true, nil }
		err := newInventory().killContextWithProbes(ctx, want, openProbe(t, p), stopped)
		if !errors.Is(err, inspectionErr) || !errors.Is(err, context.Canceled) || p.signals != 0 {
			t.Fatalf("cancelled revalidation: error=%v signals=%d", err, p.signals)
		}
	})

	t.Run("deadline preserves metadata and caller errors", func(t *testing.T) {
		p := &serviceKillProbe{exeErr: inspectionErr}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		defer cancel()
		stopped := func(ctx context.Context, _ int32) (bool, error) { <-ctx.Done(); return true, ctx.Err() }
		err := newInventory().killContextWithProbes(ctx, want, openProbe(t, p), stopped)
		if !errors.Is(err, inspectionErr) || !errors.Is(err, context.DeadlineExceeded) || p.signals != 0 {
			t.Fatalf("expired revalidation: error=%v signals=%d", err, p.signals)
		}
	})

	t.Run("creation-time failure uses independent stopped evidence", func(t *testing.T) {
		p := &serviceKillProbe{path: path, createdErr: inspectionErr}
		calls := 0
		stopped := func(context.Context, int32) (bool, error) { calls++; return true, nil }
		err := newInventory().killContextWithProbes(context.Background(), want, openProbe(t, p), stopped)
		if err != nil || calls != 1 || p.timeCalls != 1 || p.signals != 0 {
			t.Fatalf("creation-time revalidation: error=%v probes=%d time=%d signals=%d", err, calls, p.timeCalls, p.signals)
		}
	})

	t.Run("not-running metadata still requires independent evidence", func(t *testing.T) {
		calls := 0
		open := func(context.Context, int32) (serviceKillProcess, error) { return nil, process.ErrorProcessNotRunning }
		stopped := func(context.Context, int32) (bool, error) { calls++; return true, probeErr }
		err := newInventory().killContextWithProbes(context.Background(), want, open, stopped)
		if calls == 0 || !errors.Is(err, process.ErrorProcessNotRunning) || !errors.Is(err, probeErr) {
			t.Fatalf("unproven not-running metadata: error=%v probes=%d", err, calls)
		}
	})

	testChanged := func(name string, p *serviceKillProbe, in *serviceProcessInventory) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			stoppedCalls := 0
			stopped := func(context.Context, int32) (bool, error) { stoppedCalls++; return true, nil }
			err := in.killContextWithProbes(context.Background(), want, openProbe(t, p), stopped)
			if err == nil || !strings.Contains(err.Error(), "refusing changed pid 123 identity") || p.signals != 0 || stoppedCalls != 0 {
				t.Fatalf("changed identity: error=%v signals=%d stopped_probes=%d", err, p.signals, stoppedCalls)
			}
		})
	}
	testChanged("changed executable is never signalled", &serviceKillProbe{path: filepath.Join(filepath.Dir(path), "other"), created: want.Created}, newInventory())
	testChanged("reused PID creation time is never signalled", &serviceKillProbe{path: path, created: want.Created + 1}, newInventory())
	testChanged("unowned executable is never signalled", &serviceKillProbe{path: path, created: want.Created}, &serviceProcessInventory{})

	t.Run("matching identity is checked before signalling", func(t *testing.T) {
		p := &serviceKillProbe{path: path, created: want.Created}
		stopped := func(context.Context, int32) (bool, error) {
			t.Fatal("matching identity unexpectedly probed stopped state")
			return false, nil
		}
		err := newInventory().killContextWithProbes(context.Background(), want, openProbe(t, p), stopped)
		if err != nil || p.exeCalls != 1 || p.timeCalls != 1 || p.signals != 1 {
			t.Fatalf("matching identity: error=%v exe=%d time=%d signals=%d", err, p.exeCalls, p.timeCalls, p.signals)
		}
	})

	t.Run("cancellation after identity reads prevents signal", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		p := &serviceKillProbe{path: path, created: want.Created, afterCreation: cancel}
		stopped := func(context.Context, int32) (bool, error) {
			t.Fatal("cancellation unexpectedly probed stopped state")
			return false, nil
		}
		err := newInventory().killContextWithProbes(ctx, want, openProbe(t, p), stopped)
		if !errors.Is(err, context.Canceled) || p.signals != 0 {
			t.Fatalf("cancelled identity read: error=%v signals=%d", err, p.signals)
		}
	})

	t.Run("signal failure stays visible even if process would be stopped", func(t *testing.T) {
		p := &serviceKillProbe{path: path, created: want.Created, signalErr: inspectionErr}
		stopped := func(context.Context, int32) (bool, error) {
			t.Fatal("signal failure unexpectedly reconciled")
			return true, nil
		}
		err := newInventory().killContextWithProbes(context.Background(), want, openProbe(t, p), stopped)
		if !errors.Is(err, inspectionErr) || p.signals != 1 {
			t.Fatalf("failed signal: error=%v signals=%d", err, p.signals)
		}
	})
}
