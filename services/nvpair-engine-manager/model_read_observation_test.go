// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"os"
	"reflect"
	"slices"
	"sync"
	"testing"
	"time"

	"nvpair-shared/modelconfig"
)

// Continuous contention must refuse after one contention window, not before it
// and not after a renewed one. The slack covers the final attempt, scheduling
// and Windows timer granularity, and stays well below a second window.
func modelReadWithinBound(elapsed time.Duration) bool {
	return elapsed >= modelReadContentionBudget && elapsed < modelReadContentionBudget+modelReadContentionBudget/2
}

// An opMu owner needs the same content guard held by the first read attempt.
// The read must release its lease, retry from scratch and validate again.
func contendModelRead(t *testing.T, f inspectionFixture) (func(), func()) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	done := make(chan error, 1)
	var once sync.Once
	contend := func() {
		once.Do(func() {
			f.st.opMu.Lock()
			go func() {
				defer f.st.opMu.Unlock()
				ticker := time.NewTicker(time.Millisecond)
				defer ticker.Stop()
				for {
					f.ex.downloads.mu.Lock()
					readers := f.ex.downloads.readers[f.request.Engine]
					f.ex.downloads.mu.Unlock()
					if readers == 0 {
						release, err := f.ex.downloads.acquireWrite(f.request.Engine)
						if err == nil {
							release()
						}
						done <- err
						return
					}
					select {
					case <-ctx.Done():
						done <- ctx.Err()
						return
					case <-ticker.C:
					}
				}
			}()
		})
	}
	t.Cleanup(cancel)
	return contend, func() {
		t.Helper()
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("runtime owner could not acquire the released content guard: %v", err)
			}
		case <-ctx.Done():
			t.Fatal("read did not release its content lease for the runtime owner")
		}
	}
}

func TestModelReadInspectionContentionThenSuccess(t *testing.T) {
	f := newAppliedKeyFixture(t)
	contend, join := contendModelRead(t, f)
	f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
		if path == f.model {
			contend()
		}
		return os.Open(path)
	}
	result, err := f.ex.InspectModel(keyTestMemoryContext(), f.request)
	join()
	if err != nil {
		t.Fatal(err)
	}
	if result.Status != "resolved" || result.ConfigurationKey == "" || result.Facts == nil || len(result.Artifacts) != 1 {
		t.Fatalf("single inspection did not converge to actual current facts: %+v", result)
	}
	if f.ex.modelInspection.active.Load() || f.st.running || f.st.proc != nil {
		t.Fatal("read left a scan or engine operation active")
	}
}

func TestModelReadInspectionCompletedSnapshotChangeConverges(t *testing.T) {
	f := newAppliedKeyFixture(t)
	var once sync.Once
	changed := false
	f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
		if path == f.model {
			once.Do(func() {
				// Initial metadata I/O can outlast the contention allowance.
				// A late snapshot change must still get its first fresh read.
				<-time.After(modelReadContentionBudget)
				f.st.mu.Lock()
				f.st.gen++
				f.st.mu.Unlock()
				changed = true
			})
		}
		return os.Open(path)
	}
	result, err := f.ex.inspectModel(keyTestMemoryContext(), f.request, false)
	if err != nil {
		t.Fatal(err)
	}
	stable, err := f.ex.inspectModel(keyTestMemoryContext(), f.request, false)
	if err != nil {
		t.Fatal(err)
	}
	if !changed || result.Status != "resolved" || result.ConfigurationKey == "" || result.ConfigurationKey != stable.ConfigurationKey || !reflect.DeepEqual(result.Artifacts, stable.Artifacts) || !reflect.DeepEqual(result.Facts, stable.Facts) {
		t.Fatalf("retry did not return the same actual current facts as a fresh stable inspection: result=%+v stable=%+v", result, stable)
	}
}

func TestModelReadPreviewContentionThenSuccess(t *testing.T) {
	f := newAppliedKeyFixture(t)
	request := nativeOptionsPreviewRequest(t, f, nil)
	request.ModelChange.Load.Set.Context = &modelconfig.Context{Mode: "fixed", Tokens: 4096}
	contend, join := contendModelRead(t, f)
	memory := modelTargetMemoryOwner{target: modelBufferTestTargetMemory()}
	var deadlines []time.Time
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: diagnosticInspectionMemoryFunc(func(ctx context.Context) (*modelEstimateMemory, error) {
		deadline, ok := ctx.Deadline()
		if !ok {
			return nil, errors.New("preview metadata has no deadline")
		}
		deadlines = append(deadlines, deadline)
		contend()
		return memory.ModelAutoMemory(ctx)
	})})
	result, err := f.ex.previewModelLoadChange(ctx, f.st, request)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Errors) != 0 {
		t.Fatalf("preview refused before validating contention recovery: %v", result.Errors)
	}
	join()
	if len(deadlines) < 2 {
		t.Fatalf("preview did not repeat a bounded metadata read: %v", deadlines)
	}
	for _, deadline := range deadlines {
		if !deadline.Equal(deadlines[0]) {
			t.Fatalf("metadata deadline renewed across read retries: %v", deadlines)
		}
	}
	if len(result.Errors) != 0 || result.Conflict != nil || len(result.Settings.ModelLoads) == 0 {
		t.Fatalf("single preview did not converge after releasing its content lease: %+v", result)
	}
	if f.ex.modelInspection.active.Load() || f.st.running || f.st.proc != nil {
		t.Fatal("preview left a scan or engine operation active")
	}
}

// Authoritative validation succeeds; only the advisory estimate exhausts its
// internal deadline. The Preview stays valid with an Unknown estimate, and the
// estimate runs after the attempt released its content lease and scan slot.
func TestModelReadPreviewAdvisoryDeadlineKeepsPreview(t *testing.T) {
	f := newAppliedKeyFixture(t)
	request := nativeOptionsPreviewRequest(t, f, nil)
	request.ModelChange.Load.Set.Context = &modelconfig.Context{Mode: "fixed", Tokens: 4096}
	memory := modelTargetMemoryOwner{target: modelBufferTestTargetMemory()}
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: memory})
	ran, leaseHeld := false, false
	f.ex.modelEstimates.run = func(ctx context.Context, _ modelEstimateInput) modelconfig.Estimate {
		ran = true
		f.ex.downloads.mu.Lock()
		leaseHeld = f.ex.downloads.readers[f.request.Engine] != 0
		f.ex.downloads.mu.Unlock()
		leaseHeld = leaseHeld || f.ex.modelInspection.active.Load()
		<-ctx.Done()
		return modelconfig.Estimate{}
	}
	result, err := f.ex.previewModelLoadChange(ctx, f.st, request)
	if err != nil {
		t.Fatalf("advisory estimate expiration rejected a valid Preview: %v", err)
	}
	if !ran || leaseHeld {
		t.Fatalf("advisory estimate did not run after releasing the content lease: ran=%t leaseHeld=%t", ran, leaseHeld)
	}
	if len(result.Errors) != 0 || result.Conflict != nil || result.PreviewDigest == "" || result.Estimate == nil || !slices.Contains(result.Estimate.Unknowns, "estimator operation budget exhausted") {
		t.Fatalf("Preview lost its reviewed result or its Unknown advisory estimate: %+v", result)
	}
}

func TestModelReadInspectionContentionPastBound(t *testing.T) {
	f := newAppliedKeyFixture(t)
	f.st.opMu.Lock()
	defer f.st.opMu.Unlock()
	started := time.Now()
	result, err := f.ex.inspectModel(context.Background(), f.request, false)
	if err != nil {
		t.Fatal(err)
	}
	if !modelReadWithinBound(time.Since(started)) || result.Status != "busy" || result.ConfigurationKey != "" || result.ExecutionKey != "" || len(result.Artifacts) != 0 {
		t.Fatalf("still-busy inspection granted proof or refused before its bound: elapsed=%s result=%+v", time.Since(started), result)
	}
}

func TestModelReadInspectionHonorsCallerDeadline(t *testing.T) {
	f := newAppliedKeyFixture(t)
	f.st.opMu.Lock()
	defer f.st.opMu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err := f.ex.inspectModel(ctx, f.request, false)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("inspection did not retain the caller deadline: %v", err)
	}
}

func TestModelReadDiagnosticContentionThenSuccess(t *testing.T) {
	f := newAppliedKeyFixture(t)
	f.ex.nodeProjection.set(nodeProjection{})
	f.st.opMu.Lock()
	released := make(chan struct{})
	time.AfterFunc(25*time.Millisecond, func() { f.st.opMu.Unlock(); close(released) })
	p := modelconfig.DiagnosticPreviewRequest{Engine: f.request.Engine, ModelID: f.request.ModelID, ProfileKey: f.request.ProfileKey}
	result, err := f.ex.observeDiagnostic(context.Background(), p, modelBufferTestTargetMemory(), diagnosticTestHardware())
	<-released
	if err != nil {
		t.Fatal(err)
	}
	if err := result.Identity.ValidateRequested(); err != nil {
		t.Fatal(err)
	}
	if result.EffectsFence == "" || !result.Supported || f.st.running || f.st.proc != nil {
		t.Fatalf("diagnostic observation lost its verified fence or had effects: %+v", result)
	}
}

func TestModelReadDiagnosticContentionPastBound(t *testing.T) {
	f := newAppliedKeyFixture(t)
	f.ex.nodeProjection.set(nodeProjection{})
	f.st.opMu.Lock()
	defer f.st.opMu.Unlock()
	p := modelconfig.DiagnosticPreviewRequest{Engine: f.request.Engine, ModelID: f.request.ModelID, ProfileKey: f.request.ProfileKey}
	started := time.Now()
	result, err := f.ex.observeDiagnostic(context.Background(), p, modelBufferTestTargetMemory(), diagnosticTestHardware())
	var coded *codedError
	if !modelReadWithinBound(time.Since(started)) || !errors.As(err, &coded) || coded.code != -32010 || coded.message != "Memory measurement conflict: engine lifecycle is busy" || result.Identity.ConfigurationKey != "" {
		t.Fatalf("still-busy diagnostic observation granted proof or lost its refusal: elapsed=%s result=%+v err=%v", time.Since(started), result, err)
	}
}

func TestModelReadReservationEffectsContentionThenSuccess(t *testing.T) {
	f, p := diagnosticReservationFixture(t)
	f.st.opMu.Lock()
	released := make(chan struct{})
	time.AfterFunc(25*time.Millisecond, func() { f.st.opMu.Unlock(); close(released) })
	fence, err := f.ex.diagnosticEffectsFence(context.Background())
	<-released
	if err != nil {
		t.Fatal(err)
	}
	if fence != p.EffectsFence || f.st.running || f.st.proc != nil {
		t.Fatalf("effects observation did not recover the unchanged accepted fence: %q", fence)
	}
}

func TestModelReadReservationEffectsContentionPastBound(t *testing.T) {
	f, _ := diagnosticReservationFixture(t)
	f.st.opMu.Lock()
	defer f.st.opMu.Unlock()
	started := time.Now()
	fence, err := f.ex.diagnosticEffectsFence(context.Background())
	var coded *codedError
	if !modelReadWithinBound(time.Since(started)) || !errors.As(err, &coded) || coded.code != -32010 || coded.message != "Memory measurement conflict: engine lifecycle is busy" || fence != "" {
		t.Fatalf("still-busy reservation observation granted proof or lost its refusal: elapsed=%s fence=%q err=%v", time.Since(started), fence, err)
	}
}

func TestModelReadReservationInspectionContentionThenSuccess(t *testing.T) {
	f, p := diagnosticReservationFixture(t)
	contend, join := contendModelRead(t, f)
	f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
		if path == f.model {
			contend()
		}
		return os.Open(path)
	}
	result, err := f.ex.reserveDiagnostic(context.Background(), p)
	join()
	if err != nil {
		t.Fatal(err)
	}
	if result.Token == "" || result.RunID != p.RunID || result.State != "reserved" || f.st.running || f.st.proc != nil {
		t.Fatalf("reservation lost actual applied identity after read contention: %+v", result)
	}
	if err := f.ex.releaseDiagnostic(diagnosticReleaseParams{p.RunID, result.Token}); err != nil {
		t.Fatal(err)
	}
}

func TestModelReadDiagnosticIdentityRecheckContentionThenSuccess(t *testing.T) {
	for _, name := range []string{"inspection slot", "late runtime fence"} {
		t.Run(name, func(t *testing.T) {
			f, _, op := modelDiagnosticOperationFixture(t, nil)
			ctx := f.ex.modelMemoryContext(op.ctx, op.params.Engine, op.reservation.TargetMemory, false)
			if name == "inspection slot" {
				f.ex.modelInspection.active.Store(true)
				released := make(chan struct{})
				time.AfterFunc(25*time.Millisecond, func() { f.ex.modelInspection.active.Store(false); close(released) })
				defer func() { <-released }()
			} else {
				contend, join := contendModelRead(t, f)
				defer join()
				f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
					if path == f.model {
						contend()
					}
					return os.Open(path)
				}
			}
			result, err := f.ex.diagnosticInspectIdentity(ctx, op)
			if err != nil {
				t.Fatal(err)
			}
			if result.Status != "resolved" || result.ConfigurationKey != op.params.ConfigurationKey || result.ExecutionKey != op.params.ExecutionKey || result.NativeBuildID == "" {
				t.Fatalf("accepted identity recheck did not recover actual current proof: %+v", result)
			}
		})
	}
}

func TestModelReadDiagnosticIdentityRecheckPastBoundKeepsBusy(t *testing.T) {
	f, _, op := modelDiagnosticOperationFixture(t, nil)
	ctx := f.ex.modelMemoryContext(op.ctx, op.params.Engine, op.reservation.TargetMemory, false)
	f.ex.modelInspection.active.Store(true)
	defer f.ex.modelInspection.active.Store(false)
	started := time.Now()
	result, err := f.ex.diagnosticInspectIdentity(ctx, op)
	var coded *codedError
	if !modelReadWithinBound(time.Since(started)) || result.Status != "busy" || result.ConfigurationKey != "" || result.ExecutionKey != "" || !errors.As(err, &coded) || coded.code != -32010 || errors.Is(err, errModelDiagnosticIdentity) || diagnosticFailureReason(err) == "identity-changed" {
		t.Fatalf("past-bound busy inspection was misreported as changed identity or missed its bound: elapsed=%s result=%+v err=%v", time.Since(started), result, err)
	}
}
