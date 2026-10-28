// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package tests

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

type fixtureProcessLifetime struct {
	mu       sync.Mutex
	handle   windows.Handle
	closed   bool
	exited   bool
	closeErr error
}

func retainFixtureProcessLifetime(expected serviceProcessIdentity) (_ *fixtureProcessLifetime, err error) {
	if expected.PID <= 0 {
		return nil, errors.New("invalid fixture process PID")
	}
	handle, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(expected.PID))
	if err != nil {
		return nil, fmt.Errorf("retain fixture lifetime: %w", err)
	}
	retained := false
	defer func() {
		if !retained {
			err = errors.Join(err, windows.CloseHandle(handle))
		}
	}()
	pid, err := windows.GetProcessId(handle)
	if err != nil {
		return nil, err
	}
	var created, end, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(handle, &created, &end, &kernel, &user); err != nil {
		return nil, err
	}
	image := make([]uint16, 32768)
	size := uint32(len(image))
	if err := windows.QueryFullProcessImageName(handle, 0, &image[0], &size); err != nil {
		return nil, err
	}
	path, err := serviceExecutablePath(windows.UTF16ToString(image[:size]))
	if err != nil {
		return nil, err
	}
	if pid != uint32(expected.PID) || path != expected.Path || created.Nanoseconds()/int64(time.Millisecond) != expected.Created {
		return nil, errors.New("fixture lifetime identity mismatch")
	}
	state, err := windows.WaitForSingleObject(handle, 0)
	if err != nil {
		return nil, err
	}
	if state != uint32(windows.WAIT_TIMEOUT) {
		return nil, errors.New("fixture lifetime was not alive at capture")
	}
	retained = true
	return &fixtureProcessLifetime{handle: handle}, nil
}

func (p *fixtureProcessLifetime) closeLocked() error {
	if !p.closed {
		p.closed = true
		p.closeErr = windows.CloseHandle(p.handle)
	}
	return p.closeErr
}

func (p *fixtureProcessLifetime) close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.closeLocked()
}

func (p *fixtureProcessLifetime) observe(milliseconds uint32, finish bool) (bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		if p.closeErr != nil {
			return false, p.closeErr
		}
		if !p.exited {
			return false, errors.New("fixture lifetime released without exit proof")
		}
		return true, nil
	}
	state, err := windows.WaitForSingleObject(p.handle, milliseconds)
	if err == nil && state == uint32(windows.WAIT_TIMEOUT) && !finish {
		return false, nil
	}
	if err == nil {
		switch state {
		case windows.WAIT_OBJECT_0:
			p.exited = true
		case uint32(windows.WAIT_TIMEOUT):
			err = context.DeadlineExceeded
		default:
			err = fmt.Errorf("unexpected fixture lifetime wait result: %d", state)
		}
	}
	err = errors.Join(err, p.closeLocked())
	return p.exited && err == nil, err
}

func (p *fixtureProcessLifetime) stopped(ctx context.Context) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, errors.Join(err, p.close())
	}
	return p.observe(0, false)
}

func (p *fixtureProcessLifetime) wait(timeout time.Duration) (bool, error) {
	milliseconds := timeout.Milliseconds()
	if milliseconds <= 0 || milliseconds >= int64(windows.INFINITE) {
		return false, errors.Join(errors.New("fixture lifetime requires a finite positive wait"), p.close())
	}
	return p.observe(uint32(milliseconds), true)
}

func TestFixtureProcessLifetime(t *testing.T) {
	identity, err := serviceProcessIdentityAt(context.Background(), int32(os.Getpid()))
	if err != nil {
		t.Fatal(err)
	}
	t.Run("rejects changed image", func(t *testing.T) {
		changed := identity
		changed.Path += ".different"
		if proof, err := retainFixtureProcessLifetime(changed); err == nil || proof != nil {
			t.Fatal("changed image acquired a lifetime proof")
		}
	})
	t.Run("rejects changed creation time", func(t *testing.T) {
		changed := identity
		changed.Created++
		if proof, err := retainFixtureProcessLifetime(changed); err == nil || proof != nil {
			t.Fatal("changed creation time acquired a lifetime proof")
		}
	})
	t.Run("timeout closes the retained handle", func(t *testing.T) {
		proof, err := retainFixtureProcessLifetime(identity)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := proof.close(); err != nil {
				t.Error(err)
			}
		})
		handle := proof.handle
		if stopped, err := proof.stopped(context.Background()); err != nil || stopped {
			t.Fatalf("live lifetime stopped=%t error=%v", stopped, err)
		}
		if stopped, err := proof.wait(time.Millisecond); stopped || !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("timeout stopped=%t error=%v", stopped, err)
		}
		if _, err := windows.WaitForSingleObject(handle, 0); !errors.Is(err, windows.ERROR_INVALID_HANDLE) {
			t.Fatalf("timed out lifetime retained its handle: %v", err)
		}
		if stopped, err := proof.stopped(context.Background()); err == nil || stopped {
			t.Fatal("released live lifetime became exit proof")
		}
	})
	t.Run("wait and close errors are not exit proof", func(t *testing.T) {
		proof, err := retainFixtureProcessLifetime(identity)
		if err != nil {
			t.Fatal(err)
		}
		if err := windows.CloseHandle(proof.handle); err != nil {
			t.Fatal(err)
		}
		if stopped, err := proof.wait(time.Millisecond); err == nil || stopped {
			t.Fatal("invalid handle became exit proof")
		}
		if err := proof.close(); !errors.Is(err, windows.ERROR_INVALID_HANDLE) {
			t.Fatalf("close error was lost: %v", err)
		}
	})
}
