// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"errors"
	"testing"
	"time"
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
