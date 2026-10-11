// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/shirou/gopsutil/v4/process"
)

func serviceRecoveryInventory(t *testing.T) (*serviceProcessInventory, serviceProcessIdentity) {
	t.Helper()
	identity := serviceProcessIdentity{PID: 123, Path: filepath.Join(t.TempDir(), "copied-scanner"), Created: 456}
	return &serviceProcessInventory{
		paths: []string{identity.Path},
		known: map[int32]serviceProcessIdentity{identity.PID: identity},
	}, identity
}

func TestServiceRecoveryIncludesRecordedLiveDescendant(t *testing.T) {
	in, identity := serviceRecoveryInventory(t)
	enumerationErr := errors.New("enumeration unavailable")
	scanErr := errors.New("partial inspection failed")
	in.enumerationError = enumerationErr
	in.scanError = scanErr
	calls := 0
	in.recoveryProbe = func(ctx context.Context, pid int32) (bool, error) {
		calls++
		if pid != identity.PID || ctx.Err() != nil {
			t.Fatalf("unexpected recovery probe: pid=%d context=%v", pid, ctx.Err())
		}
		return serviceProcessStoppedWithProbes(ctx, pid,
			func(context.Context, int32) (bool, error) { return true, nil },
			func(context.Context, int32) ([]string, error) { return []string{process.Sleep}, nil })
	}
	scanned, err := in.scanContext(context.Background())
	if len(scanned) != 0 || !errors.Is(err, enumerationErr) || !errors.Is(err, scanErr) {
		t.Fatalf("failed empty scan: identities=%+v error=%v, want no identities and both enumeration and inspection failures", scanned, err)
	}
	owned, err := in.recoveryContext(context.Background(), scanned)
	if err != nil {
		t.Fatalf("recover recorded live descendant: %v", err)
	}
	if calls != 1 || !reflect.DeepEqual(owned, []serviceProcessIdentity{identity}) {
		t.Fatalf("recovered identities=%+v probes=%d, want exact recorded identity %+v and one probe", owned, calls, identity)
	}
}

func TestServiceRecoveryOmitsIndependentlyAbsentIdentity(t *testing.T) {
	in, _ := serviceRecoveryInventory(t)
	calls := 0
	in.recoveryProbe = func(ctx context.Context, pid int32) (bool, error) {
		return serviceProcessStoppedWithProbes(ctx, pid,
			func(context.Context, int32) (bool, error) {
				calls++
				return false, nil
			},
			func(context.Context, int32) ([]string, error) {
				t.Fatal("status queried after independent absence")
				return nil, nil
			})
	}
	owned, err := in.recoveryContext(context.Background(), nil)
	if err != nil || len(owned) != 0 || calls != 1 {
		t.Fatalf("absent recovery: identities=%+v error=%v probes=%d, want empty recovery, nil error and one existence probe", owned, err, calls)
	}
}

func TestServiceRecoveryOmitsKnownZombie(t *testing.T) {
	in, identity := serviceRecoveryInventory(t)
	statusCalls := 0
	in.recoveryProbe = func(ctx context.Context, pid int32) (bool, error) {
		return serviceProcessStoppedWithProbes(ctx, pid,
			func(context.Context, int32) (bool, error) { return true, nil },
			func(ctx context.Context, pid int32) ([]string, error) {
				statusCalls++
				if pid != identity.PID || ctx.Err() != nil {
					t.Fatalf("unexpected status probe: pid=%d context=%v", pid, ctx.Err())
				}
				return []string{process.Sleep, process.Zombie}, nil
			})
	}
	owned, err := in.recoveryContext(context.Background(), nil)
	if err != nil || len(owned) != 0 || statusCalls != 1 {
		t.Fatalf("zombie recovery: identities=%+v error=%v status probes=%d, want empty recovery, nil error and one status probe", owned, err, statusCalls)
	}
}

func TestServiceRecoveryPreservesPartialScanIdentity(t *testing.T) {
	in, missing := serviceRecoveryInventory(t)
	scanned := serviceProcessIdentity{PID: 789, Path: missing.Path, Created: 987}
	in.known[scanned.PID] = scanned
	calls := 0
	in.recoveryProbe = func(ctx context.Context, pid int32) (bool, error) {
		calls++
		if pid != missing.PID {
			t.Fatalf("re-probed scanned identity: pid=%d, want only omitted pid %d", pid, missing.PID)
		}
		return serviceProcessStoppedWithProbes(ctx, pid,
			func(context.Context, int32) (bool, error) { return true, nil }, nil)
	}
	owned, err := in.recoveryContext(context.Background(), []serviceProcessIdentity{scanned})
	if err != nil {
		t.Fatalf("recover partial scan: %v", err)
	}
	if calls != 1 || !reflect.DeepEqual(owned, []serviceProcessIdentity{scanned, missing}) {
		t.Fatalf("partial recovery: identities=%+v probes=%d, want preserved scanned identity %+v followed by recorded identity %+v", owned, calls, scanned, missing)
	}
}

func TestServiceRecoveryRetainsProbeErrors(t *testing.T) {
	probeErr := errors.New("inspection unavailable")
	test := func(name string, present bool, existenceErr error, states []string, statusErr error, wantStatusCalls int) {
		t.Run(name, func(t *testing.T) {
			in, identity := serviceRecoveryInventory(t)
			existenceCalls, statusCalls := 0, 0
			in.recoveryProbe = func(ctx context.Context, pid int32) (bool, error) {
				return serviceProcessStoppedWithProbes(ctx, pid,
					func(context.Context, int32) (bool, error) {
						existenceCalls++
						return present, existenceErr
					},
					func(context.Context, int32) ([]string, error) {
						statusCalls++
						return states, statusErr
					})
			}
			owned, err := in.recoveryContext(context.Background(), nil)
			if !errors.Is(err, probeErr) || !reflect.DeepEqual(owned, []serviceProcessIdentity{identity}) {
				t.Fatalf("ambiguous recovery: identities=%+v error=%v, want exact recorded identity %+v and retained probe error", owned, err, identity)
			}
			if existenceCalls != 1 || statusCalls != wantStatusCalls {
				t.Fatalf("probe calls: existence=%d status=%d, want existence=1 status=%d", existenceCalls, statusCalls, wantStatusCalls)
			}
		})
	}
	test("existence failure preserves apparent presence", true, probeErr, nil, nil, 0)
	test("existence failure preserves apparent absence", false, probeErr, nil, nil, 0)
	test("status failure preserves apparent zombie", true, nil, []string{process.Zombie}, probeErr, 1)
}

func TestServiceRecoveryPreservesAmbiguousStoppedIdentity(t *testing.T) {
	in, identity := serviceRecoveryInventory(t)
	probeErr := errors.New("stopped-state probe unavailable")
	in.recoveryProbe = func(context.Context, int32) (bool, error) { return true, probeErr }
	owned, err := in.recoveryContext(context.Background(), nil)
	if !errors.Is(err, probeErr) || !reflect.DeepEqual(owned, []serviceProcessIdentity{identity}) {
		t.Fatalf("ambiguous stopped recovery: identities=%+v error=%v, want exact recorded identity %+v and retained probe error", owned, err, identity)
	}
}

func TestServiceRecoveryCancellationPreventsFallbackDecisions(t *testing.T) {
	probeErr := errors.New("probe failed while cancelled")
	t.Run("already cancelled preserves only scanned identities", func(t *testing.T) {
		in, recorded := serviceRecoveryInventory(t)
		scanned := serviceProcessIdentity{PID: 789, Path: recorded.Path, Created: 987}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		in.recoveryProbe = func(context.Context, int32) (bool, error) {
			t.Fatal("recovery probed after caller cancellation")
			return false, nil
		}
		owned, err := in.recoveryContext(ctx, []serviceProcessIdentity{scanned})
		if !errors.Is(err, context.Canceled) || !reflect.DeepEqual(owned, []serviceProcessIdentity{scanned}) {
			t.Fatalf("cancelled recovery: identities=%+v error=%v, want only scanned identity %+v and cancellation", owned, err, scanned)
		}
	})
	test := func(name string, cancelDuringStatus bool) {
		t.Run(name, func(t *testing.T) {
			in, _ := serviceRecoveryInventory(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			in.recoveryProbe = func(ctx context.Context, pid int32) (bool, error) {
				return serviceProcessStoppedWithProbes(ctx, pid,
					func(context.Context, int32) (bool, error) {
						if cancelDuringStatus {
							return true, nil
						}
						cancel()
						return false, probeErr
					},
					func(context.Context, int32) ([]string, error) {
						if !cancelDuringStatus {
							t.Fatal("status queried after cancellation during existence probe")
						}
						cancel()
						return []string{process.Zombie}, probeErr
					})
			}
			owned, err := in.recoveryContext(ctx, nil)
			if len(owned) != 0 || !errors.Is(err, context.Canceled) || !errors.Is(err, probeErr) {
				t.Fatalf("cancelled fallback: identities=%+v error=%v, want no added identity and both cancellation and probe failure", owned, err)
			}
		})
	}
	test("cancellation during existence keeps both errors", false)
	test("cancellation during status keeps both errors", true)
}

func TestServiceRecoveryPreservesCallerDeadline(t *testing.T) {
	in, _ := serviceRecoveryInventory(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	in.recoveryProbe = func(ctx context.Context, pid int32) (bool, error) {
		return serviceProcessStoppedWithProbes(ctx, pid,
			func(ctx context.Context, _ int32) (bool, error) {
				<-ctx.Done()
				return false, ctx.Err()
			}, nil)
	}
	owned, err := in.recoveryContext(ctx, nil)
	if len(owned) != 0 || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expired recovery: identities=%+v error=%v, want no fallback identity and caller deadline", owned, err)
	}
}

func TestServiceRecoveryRejectsIdentityOutsideAllowlist(t *testing.T) {
	in, identity := serviceRecoveryInventory(t)
	in.paths = []string{filepath.Join(t.TempDir(), "different-copy")}
	in.recoveryProbe = func(context.Context, int32) (bool, error) {
		t.Fatal("recovery probed an identity outside the copied-path allowlist")
		return false, nil
	}
	owned, err := in.recoveryContext(context.Background(), nil)
	if len(owned) != 0 || err == nil || !strings.Contains(err.Error(), "outside fixture paths") {
		t.Fatalf("unowned recovery: identities=%+v error=%v, want no target and an identity error for %+v", owned, err, identity)
	}
}
