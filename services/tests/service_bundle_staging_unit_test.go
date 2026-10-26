// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"testing"
)

func TestServiceBundleStagingRequiresExitBeforeEveryAttempt(t *testing.T) {
	sharing := errors.New("image release pending")
	notStopped := errors.New("detached scanner is still running")
	var events []string
	checks := 0
	err := serviceBundleMutation(context.Background(), func(context.Context) error {
		events = append(events, "confirm exit")
		checks++
		if checks == 2 {
			return notStopped
		}
		return nil
	}, func() error {
		events = append(events, "stage")
		return sharing
	}, func(err error) bool { return errors.Is(err, sharing) })
	if !errors.Is(err, notStopped) || !errors.Is(err, sharing) {
		t.Fatalf("staging error=%v, want both process and image-release evidence", err)
	}
	if want := []string{"confirm exit", "stage", "confirm exit"}; !reflect.DeepEqual(events, want) {
		t.Fatalf("events=%v, want %v without staging after the live-process finding", events, want)
	}
}

func TestServiceBundleStagingRefusesAmbiguousOwnership(t *testing.T) {
	inspection := errors.New("recorded process cannot be inspected")
	writes := 0
	err := serviceBundleMutation(context.Background(), func(context.Context) error { return inspection }, func() error {
		writes++
		return nil
	}, func(error) bool { return true })
	if writes != 0 || !errors.Is(err, inspection) {
		t.Fatalf("writes=%d error=%v, want no writes and the ownership error", writes, err)
	}
}

func TestServiceBundleStagingRetriesOnlyImageRelease(t *testing.T) {
	sharing := errors.New("image release pending")
	checks, writes := 0, 0
	err := serviceBundleMutation(context.Background(), func(context.Context) error { checks++; return nil }, func() error {
		writes++
		if writes == 1 {
			return sharing
		}
		return nil
	}, func(err error) bool { return errors.Is(err, sharing) })
	if err != nil || checks != 2 || writes != 2 {
		t.Fatalf("checks=%d writes=%d error=%v, want exit-confirmed retry then success", checks, writes, err)
	}
	other := errors.New("disk full")
	writes = 0
	err = serviceBundleMutation(context.Background(), func(context.Context) error { return nil }, func() error {
		writes++
		return other
	}, func(err error) bool { return errors.Is(err, sharing) })
	if writes != 1 || !errors.Is(err, other) {
		t.Fatalf("writes=%d error=%v, want immediate non-sharing failure", writes, err)
	}
}

func TestServiceBundleStagingCancellationBoundsImageReleaseWait(t *testing.T) {
	sharing := errors.New("image release pending")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	writes := 0
	err := serviceBundleMutation(ctx, func(context.Context) error { return nil }, func() error {
		writes++
		cancel()
		return sharing
	}, func(err error) bool { return errors.Is(err, sharing) })
	if !errors.Is(err, context.Canceled) || !errors.Is(err, sharing) || writes != 1 {
		t.Fatalf("writes=%d error=%v, want bounded release wait retaining the image error", writes, err)
	}
}

func TestServiceBundleExitIncludesRecordedOptionalScanner(t *testing.T) {
	service := filepath.Join(t.TempDir(), "service")
	scanner := filepath.Join(filepath.Dir(service), "scanner")
	want := serviceProcessIdentity{PID: 123, Path: scanner, Created: 456}
	in := &serviceProcessInventory{
		paths:            []string{service},
		bundlePaths:      []string{service, scanner},
		known:            map[int32]serviceProcessIdentity{want.PID: want},
		recoveryProbe:    func(context.Context, int32) (bool, error) { return false, nil },
		recoveryIdentity: func(context.Context, int32) (serviceProcessIdentity, error) { return want, nil },
	}
	if err := in.confirmStoppedContext(context.Background()); err == nil {
		t.Fatal("staging accepted a recorded live scanner absent from readiness and the current scan")
	}
	in.recoveryProbe = func(context.Context, int32) (bool, error) { return true, nil }
	in.recoveryIdentity = func(context.Context, int32) (serviceProcessIdentity, error) {
		t.Fatal("a confirmed departed scanner must not be inspected or signalled by reused PID")
		return serviceProcessIdentity{}, nil
	}
	if err := in.confirmStoppedContext(context.Background()); err != nil {
		t.Fatalf("confirmed scanner exit blocks staging: %v", err)
	}
	if want := []string{service, scanner}; !reflect.DeepEqual(in.ownedPaths(), want) {
		t.Fatalf("owned paths=%v, want each required and optional image once: %v", in.ownedPaths(), want)
	}
}
