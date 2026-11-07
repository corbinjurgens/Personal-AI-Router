// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package tests

import (
	"context"
	"errors"
	"os"
	"sync"
	"testing"
	"time"
)

type fixtureProcessLifetime struct {
	mu       sync.Mutex
	expected serviceProcessIdentity
	closed   bool
	exited   bool
}

func retainFixtureProcessLifetime(expected serviceProcessIdentity) (*fixtureProcessLifetime, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return retainFixtureProcessLifetimeWithProbes(expected,
		func() (serviceProcessIdentity, error) { return serviceProcessIdentityAt(ctx, expected.PID) },
		func() (bool, error) { return fixtureProcessIdentityStopped(ctx, expected) })
}

func retainFixtureProcessLifetimeWithProbes(expected serviceProcessIdentity, identify func() (serviceProcessIdentity, error), stopped func() (bool, error)) (*fixtureProcessLifetime, error) {
	current, err := identify()
	if err != nil {
		return nil, err
	}
	if current != expected {
		return nil, errors.New("fixture lifetime identity mismatch")
	}
	exited, err := stopped()
	if err != nil {
		return nil, err
	}
	// A short-lived engine can exit after its exact identity was observed but
	// before capture completes, including while a resumed fixture is torn down.
	// Positive departure evidence already proves this matched lifetime ended.
	return &fixtureProcessLifetime{expected: expected, exited: exited, closed: exited}, nil
}

func (p *fixtureProcessLifetime) close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closed = true
	return nil
}

func (p *fixtureProcessLifetime) stopped(ctx context.Context) (bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.exited {
		return true, nil
	}
	if p.closed {
		return false, errors.New("fixture lifetime released without exit proof")
	}
	stopped, err := fixtureProcessIdentityStopped(ctx, p.expected)
	if err == nil && stopped {
		p.exited, p.closed = true, true
	}
	return stopped, err
}

func (p *fixtureProcessLifetime) wait(timeout time.Duration) (exited bool, err error) {
	defer func() {
		if closeErr := p.close(); closeErr != nil {
			exited = false
			err = errors.Join(err, closeErr)
		}
	}()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	tick := time.NewTicker(25 * time.Millisecond)
	defer tick.Stop()
	for {
		stopped, err := p.stopped(ctx)
		if err != nil || stopped {
			return stopped, err
		}
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case <-tick.C:
		}
	}
}

func TestFixtureProcessLifetime(t *testing.T) {
	identity, err := serviceProcessIdentityAt(context.Background(), int32(os.Getpid()))
	if err != nil {
		t.Fatal(err)
	}
	t.Run("rejects changed identity", func(t *testing.T) {
		changed := identity
		changed.Created++
		if proof, err := retainFixtureProcessLifetime(changed); err == nil || proof != nil {
			t.Fatal("changed identity acquired a lifetime proof")
		}
	})
	t.Run("timeout never proves exit", func(t *testing.T) {
		proof, err := retainFixtureProcessLifetime(identity)
		if err != nil {
			t.Fatal(err)
		}
		if stopped, err := proof.wait(time.Millisecond); stopped || !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("timeout stopped=%t error=%v", stopped, err)
		}
		if stopped, err := proof.stopped(context.Background()); err == nil || stopped {
			t.Fatal("released live lifetime became exit proof")
		}
	})
}

func TestFixtureProcessLifetimeExitDuringCapture(t *testing.T) {
	expected := serviceProcessIdentity{PID: 123, Path: "fixture-engine", Created: 456}
	proof, err := retainFixtureProcessLifetimeWithProbes(expected,
		func() (serviceProcessIdentity, error) { return expected, nil },
		func() (bool, error) { return true, nil })
	if err != nil {
		t.Fatal(err)
	}
	if exited, err := proof.wait(time.Millisecond); err != nil || !exited {
		t.Fatalf("matched process departed during capture: exited=%t error=%v", exited, err)
	}
}

func TestFixtureProcessLifetimeCaptureRefusesUnprovenExit(t *testing.T) {
	expected := serviceProcessIdentity{PID: 123, Path: "fixture-engine", Created: 456}
	inspectionErr := errors.New("process status unavailable")
	proof, err := retainFixtureProcessLifetimeWithProbes(expected,
		func() (serviceProcessIdentity, error) { return expected, nil },
		func() (bool, error) { return false, inspectionErr })
	if proof != nil || !errors.Is(err, inspectionErr) {
		t.Fatalf("unproven exit acquired a lifetime: proof=%v error=%v", proof, err)
	}
}
