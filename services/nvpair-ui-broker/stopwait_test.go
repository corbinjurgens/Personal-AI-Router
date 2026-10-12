// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"sync"
	"syscall"
	"testing"
	"time"
)

// closeRecorder is an io.Closer that records whether Close was called.
type closeRecorder struct {
	mu     sync.Mutex
	closed bool
}

func (c *closeRecorder) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	return nil
}

func (c *closeRecorder) wasClosed() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closed
}

// TestWaitForStdinCloseWaitsForOrderlyExit covers the ordinary path: signal
// shutdown by closing stdin, then let the worker finish on its own. Nothing may
// be signalled while the worker is still inside its grace.
func TestWaitForStdinCloseWaitsForOrderlyExit(t *testing.T) {
	resetTeardownClock()
	stdin := &closeRecorder{}
	done := make(chan struct{})
	returned := make(chan struct{})
	go func() {
		waitForStdinClose("scanner", nil, stdin, done)
		close(returned)
	}()

	select {
	case <-returned:
		t.Fatal("returned before the worker exited")
	case <-time.After(200 * time.Millisecond):
	}
	if !stdin.wasClosed() {
		t.Fatal("did not close stdin to signal shutdown")
	}

	close(done)
	select {
	case <-returned:
	case <-time.After(2 * time.Second):
		t.Fatal("did not return after the worker exited")
	}
}

// TestWaitForStdinCloseGivesUpWithoutProcess is the anti-deadlock guarantee. A
// worker that never exits must not park teardown forever: before this was
// bounded, a worker blocked in write() on a full stderr pipe held the broker here
// until it was force-killed from outside, skipping its remaining shutdown work.
func TestWaitForStdinCloseGivesUpWithoutProcess(t *testing.T) {
	resetTeardownClock()
	stdin := &closeRecorder{}
	done := make(chan struct{})
	defer close(done)

	returned := make(chan struct{})
	go func() {
		waitForStdinClose("scanner", nil, stdin, done)
		close(returned)
	}()

	// No process handle to escalate to, so it must abandon the join after the
	// grace rather than waiting on an exit that is never coming.
	select {
	case <-returned:
	case <-time.After(workerStopGrace + 3*time.Second):
		t.Fatal("never returned for a worker that does not exit; teardown can still deadlock")
	}
}

// TestWaitForStdinCloseKillsWorkerThatIgnoresShutdown checks the escalation
// actually reaches a real process that will not leave on its own.
func TestWaitForStdinCloseKillsWorkerThatIgnoresShutdown(t *testing.T) {
	resetTeardownClock()

	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("test executable: %v", err)
	}
	cmd := exec.Command(executable, shutdownIgnoringChildArg)
	configureSubprocess(cmd)
	cmd.Stderr = os.Stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("stdin pipe: %v", err)
	}
	t.Cleanup(func() {
		if err := stdin.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			t.Errorf("cleanup close stdin: %v", err)
		}
	})
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	t.Cleanup(func() {
		if err := stdout.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			t.Errorf("cleanup close stdout: %v", err)
		}
	})
	if err := cmd.Start(); err != nil {
		t.Fatalf("start: %v", err)
	}

	done := make(chan struct{})
	var waitErr error
	go func() {
		waitErr = cmd.Wait()
		close(done)
	}()

	type childLine struct {
		text string
		err  error
	}
	lines := make(chan childLine, 2)
	readerDone := make(chan struct{})
	go func() {
		defer close(readerDone)
		reader := bufio.NewReader(stdout)
		for range 2 {
			line, err := reader.ReadString('\n')
			lines <- childLine{text: line, err: err}
			if err != nil {
				return
			}
		}
	}()

	var returned chan struct{}
	t.Cleanup(func() {
		select {
		case <-done:
		default:
			if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
				t.Errorf("cleanup kill child %d: %v", cmd.Process.Pid, err)
			}
		}
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Errorf("cleanup did not reap child %d", cmd.Process.Pid)
		}
		if err := stdout.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			t.Errorf("cleanup close stdout: %v", err)
		}
		select {
		case <-readerDone:
		case <-time.After(2 * time.Second):
			t.Error("cleanup did not join child stdout reader")
		}
		if returned != nil {
			select {
			case <-returned:
			case <-time.After(2 * time.Second):
				t.Error("cleanup did not join shutdown waiter")
			}
		}
	})
	readLine := func(want string) {
		t.Helper()
		select {
		case got := <-lines:
			if got.err != nil {
				t.Fatalf("child handshake %q: %v", want, got.err)
			}
			if got.text != want+"\n" {
				t.Fatalf("child handshake = %q, want %q", got.text, want+"\n")
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("child handshake %q timed out", want)
		}
	}
	readLine("ready")
	t.Logf("owned shutdown fixture ready: pid=%d", cmd.Process.Pid)
	select {
	case <-done:
		t.Fatalf("child exited before shutdown: %v", waitErr)
	default:
	}

	started := time.Now()
	returned = make(chan struct{})
	go func() {
		waitForStdinClose("scanner", cmd, stdin, done)
		close(returned)
	}()
	readLine("stdin-closed")

	select {
	case <-returned:
	case <-time.After(workerStopGrace + 4*workerSignalGrace + 5*time.Second):
		t.Fatal("escalation never completed for a worker that ignores shutdown")
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("worker was left running after the join returned")
	}
	var exitErr *exec.ExitError
	if !errors.As(waitErr, &exitErr) {
		t.Fatalf("worker wait = %v, want an unsuccessful native termination", waitErr)
	}
	if runtime.GOOS != "windows" {
		status, ok := exitErr.Sys().(syscall.WaitStatus)
		if !ok || !status.Signaled() || status.Signal() != syscall.SIGKILL {
			t.Fatalf("worker exit = %v, want SIGKILL after ignored SIGTERM", exitErr)
		}
	}
	// POSIX reaches forced kill after both graces; Windows TerminateProcess
	// ends this scanner fixture at the first escalation, after the EOF grace.
	minimum := 3 * time.Second
	if runtime.GOOS != "windows" {
		minimum = 5 * time.Second
	}
	if elapsed := time.Since(started); elapsed < minimum {
		t.Fatalf("worker exited after %v, before its %v shutdown graces", elapsed, minimum)
	}
	t.Logf("owned shutdown fixture reaped before cleanup: pid=%d, exit=%v", cmd.Process.Pid, waitErr)
}

const shutdownIgnoringChildArg = "--nvpair-broker-shutdown-ignoring-child"

// Entered by TestMain before it allocates app-data directories: a killed child
// cannot run the normal TestMain cleanup. This fixture creates no files.
func runShutdownIgnoringChild() {
	if runtime.GOOS != "windows" {
		signal.Ignore(syscall.SIGTERM)
	}
	if _, err := fmt.Fprintln(os.Stdout, "ready"); err != nil {
		os.Exit(2)
	}
	if _, err := io.Copy(io.Discard, os.Stdin); err != nil {
		os.Exit(2)
	}
	if _, err := fmt.Fprintln(os.Stdout, "stdin-closed"); err != nil {
		os.Exit(2)
	}
	for {
		time.Sleep(time.Hour)
	}
}

// TestTeardownBudgetCapsAggregateWait is the guarantee that per-worker graces
// cannot add up. The joins run sequentially, so eleven workers that each hang for
// their own grace would take far longer than the parent waits before killing this
// process — and being killed mid-teardown skips the rest of shutdown.
func TestTeardownBudgetCapsAggregateWait(t *testing.T) {
	resetTeardownClock()
	beginTeardown()

	// Never closed: every worker hangs, the worst case.
	done := make(chan struct{})
	defer close(done)

	const workers = 11
	started := time.Now()
	for i := 0; i < workers; i++ {
		waitForStdinClose("scanner", nil, &closeRecorder{}, done)
	}
	elapsed := time.Since(started)

	unbounded := workers * workerStopGrace
	if elapsed >= unbounded {
		t.Fatalf("aggregate wait %v did not improve on the unbounded %v", elapsed, unbounded)
	}
	// Budget, plus the floor each worker keeps once it is spent, plus slack.
	ceiling := teardownBudget + workers*minWorkerGrace + 2*time.Second
	if elapsed > ceiling {
		t.Fatalf("aggregate wait %v exceeded the budgeted ceiling %v", elapsed, ceiling)
	}
}

// TestGraceForClipsToRemainingBudget checks a worker still gets its own grace when
// the budget is untouched, and is clipped rather than starved once it is spent.
// TestGraceForIgnoresBudgetOutsideTeardown is the regression guard for a
// process-global clock armed by the wrong join. supervisor.Restart stops a worker
// at ordinary runtime — LM Studio port reconciliation does it on every rebind — and
// arming the budget there left it permanently spent, so the real shutdown gave
// every worker the floor and signalled engine-manager almost immediately.
func TestGraceForIgnoresBudgetOutsideTeardown(t *testing.T) {
	resetTeardownClock()

	// A restart-shaped stop, long before any shutdown. It must not be clipped, and
	// it must not arm anything.
	if got := graceFor("lmstudio-proxy"); got != workerStopGrace {
		t.Fatalf("restart-path grace = %v, want its full %v", got, workerStopGrace)
	}
	stdin := &closeRecorder{}
	done := make(chan struct{})
	close(done)
	waitForStdinClose("lmstudio-proxy", nil, stdin, done)

	// The later real teardown must still start with a full budget.
	beginTeardown()
	if got := graceFor(engineManagerWorkerName); got != engineManagerStopGrace {
		t.Fatalf("engine-manager grace after an earlier restart = %v, want its full %v",
			got, engineManagerStopGrace)
	}
}

// TestGraceForReservesEngineManagerAllowance covers join order: engine-manager is
// joined eighth of eleven, so without a reservation the workers ahead of it decide
// how long it gets to stop its engines.
func TestGraceForReservesEngineManagerAllowance(t *testing.T) {
	// Earlier joins have spent the budget down to just over engine-manager's
	// reserve, so an ordinary worker is left with less than the floor.
	setTeardownStart(time.Now().Add(-(teardownBudget - engineManagerStopGrace - 100*time.Millisecond)))

	if got := graceFor("scanner"); got != minWorkerGrace {
		t.Fatalf("ordinary worker grace = %v, want the %v floor once only the reserve is left",
			got, minWorkerGrace)
	}
	// Its reserve survives what the others spent, give or take the clock ticking
	// between these two calls.
	if got := graceFor(engineManagerWorkerName); got < engineManagerStopGrace-50*time.Millisecond {
		t.Fatalf("engine-manager grace = %v, want its reserved %v", got, engineManagerStopGrace)
	}
}

// TestEngineShutdownDeadlineDrawsOnTheBudget checks the one step that used to have
// no deadline at all cannot spend the whole window before a worker is joined.
func TestEngineShutdownDeadlineDrawsOnTheBudget(t *testing.T) {
	resetTeardownClock()
	if got := engineShutdownDeadline(); got != engineStopAllBudget {
		t.Fatalf("deadline outside teardown = %v, want %v", got, engineStopAllBudget)
	}

	setTeardownStart(time.Now().Add(-(teardownBudget - time.Second)))
	got := engineShutdownDeadline()
	if got > time.Second || got < minWorkerGrace {
		t.Fatalf("deadline with 1s of budget left = %v, want about 1s", got)
	}

	setTeardownStart(time.Now().Add(-2 * teardownBudget))
	if got := engineShutdownDeadline(); got != minWorkerGrace {
		t.Fatalf("deadline with a spent budget = %v, want the %v floor", got, minWorkerGrace)
	}
}

func TestGraceForClipsToRemainingBudget(t *testing.T) {
	setTeardownStart(time.Now())

	if got := graceFor("scanner"); got != workerStopGrace {
		t.Fatalf("first worker grace = %v, want its full %v", got, workerStopGrace)
	}
	if got := graceFor(engineManagerWorkerName); got != engineManagerStopGrace {
		t.Fatalf("engine-manager grace = %v, want its full %v", got, engineManagerStopGrace)
	}

	// Pretend teardown began long enough ago that the budget is gone.
	setTeardownStart(time.Now().Add(-2 * teardownBudget))
	if got := graceFor("scanner"); got != minWorkerGrace {
		t.Fatalf("grace with a spent budget = %v, want the %v floor", got, minWorkerGrace)
	}

	// Halfway through the budget a long grace is clipped, not refused.
	setTeardownStart(time.Now().Add(-(teardownBudget - 2*time.Second)))
	got := graceFor(engineManagerWorkerName)
	if got > 2*time.Second || got < time.Second {
		t.Fatalf("clipped engine-manager grace = %v, want about 2s", got)
	}
}

func setTeardownStart(at time.Time) {
	teardownClock.mu.Lock()
	defer teardownClock.mu.Unlock()
	teardownClock.start = at
}

// TestStopGraceForGivesEngineManagerRoom guards the one worker whose shutdown does
// real work: it stops the engine processes it launched, and cutting that short is
// what orphans them.
func TestStopGraceForGivesEngineManagerRoom(t *testing.T) {
	if got := stopGraceFor(engineManagerWorkerName); got != engineManagerStopGrace {
		t.Fatalf("engine-manager grace = %v, want %v", got, engineManagerStopGrace)
	}
	if got := stopGraceFor("scanner"); got != workerStopGrace {
		t.Fatalf("scanner grace = %v, want %v", got, workerStopGrace)
	}
	if engineManagerStopGrace <= workerStopGrace {
		t.Fatal("engine-manager must get more room than an ordinary worker")
	}
	// Otherwise the longest grace is clipped the instant teardown starts and can
	// never actually be granted.
	if engineManagerStopGrace >= teardownBudget {
		t.Fatalf("longest worker grace %v must stay under the teardown budget %v",
			engineManagerStopGrace, teardownBudget)
	}
}
