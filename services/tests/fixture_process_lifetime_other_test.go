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
	current, err := serviceProcessIdentityAt(ctx, expected.PID)
	if err != nil {
		return nil, err
	}
	if current != expected {
		return nil, errors.New("fixture lifetime identity mismatch")
	}
	stopped, err := switchProcessIdentityStopped(ctx, expected)
	if err != nil {
		return nil, err
	}
	if stopped {
		return nil, errors.New("fixture lifetime was not alive at capture")
	}
	return &fixtureProcessLifetime{expected: expected}, nil
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
	stopped, err := switchProcessIdentityStopped(ctx, p.expected)
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
