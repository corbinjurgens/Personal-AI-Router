// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"
	"time"
)

func TestStopAllCancelsInFlightOrphanConfirmation(t *testing.T) {
	// Exercise the existing post-signal confirmation seam without signalling
	// the test PID. An accepted connection positively witnesses confirmation
	// in progress before shutdown; no deadline race or production hook is needed.
	probed := make(chan struct{}, 1)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			select {
			case probed <- struct{}{}:
			default:
			}
		}
	}
	server.Start()
	t.Cleanup(server.Close)
	_, rawPort, err := net.SplitHostPort(server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(rawPort)
	if err != nil {
		t.Fatal(err)
	}
	m := testEngineManifest(fakeEngineBin)
	p := m.Platforms[hostKey()]
	p.Runtime.Port = port
	p.Runtime.Health = nil
	p.Runtime.Stop = &StopSpec{Signal: "term", GraceS: 10}
	m.Platforms[hostKey()] = p
	ex := desiredTestExecutor(t, m, t.TempDir())
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.opMu.Lock()
	ctx, finish := ex.startContext(context.Background(), st)
	confirmed := make(chan error, 1)
	go func() {
		defer st.opMu.Unlock()
		defer finish()
		confirmed <- ex.confirmOrphanStoppedContext(ctx, "fake", os.Getpid(), port, p.Runtime)
	}()
	select {
	case <-probed:
	case <-time.After(3 * time.Second):
		// Always release the simulated in-flight operation on a setup failure.
		st.mu.Lock()
		cancel := st.startCancel
		st.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		t.Fatal("orphan confirmation did not reach the listener")
	}
	stopped := make(chan struct{})
	go func() {
		ex.StopAll()
		close(stopped)
	}()
	select {
	case err := <-confirmed:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("in-flight confirmation = %v, want shutdown cancellation", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("StopAll did not cancel in-flight reclaim confirmation")
	}
	select {
	case <-stopped:
	case <-time.After(3 * time.Second):
		t.Fatal("StopAll remained blocked after reclaim cancellation")
	}
	status, err := ex.Status("fake")
	if err != nil || status.Running || status.Healthy {
		t.Fatalf("shutdown adopted surviving failed reclaim: %+v err=%v", status, err)
	}
	st.mu.Lock()
	fenced, owned := st.reclaimFailed, st.proc != nil
	st.mu.Unlock()
	if !fenced || owned || !portServing(port) {
		t.Errorf("cancelled confirmation: fenced=%v owned=%v serving=%v", fenced, owned, portServing(port))
	}
	reports := ex.Errors()
	if len(reports) != 1 || reports[0].ID != startFailedID("fake") {
		t.Errorf("cancelled reclaim reports = %+v", reports)
	}
}

func TestRestoreEnabledWindowsConsoleOrphanDoesNotSpendGrace(t *testing.T) {
	m, dir, port := restoreOrphanManifest(t)
	p := m.Platforms[hostKey()]
	p.Runtime.Stop = &StopSpec{Signal: "term", GraceS: 10}
	m.Platforms[hostKey()] = p
	orphan := spawnFakeListener(t, p.Runtime.Bin, port)
	ex := restoreOrphanExecutor(t, m, dir)
	if err := ex.desired.set("fake", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	err := ex.RestoreEnabled(context.Background())
	elapsed := time.Since(started)
	if err != nil {
		t.Fatalf("cold restore console orphan: %v", err)
	}
	if elapsed >= 5*time.Second {
		t.Errorf("console orphan restore took %s, must not consume configured 10s graceful wait", elapsed)
	}
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	proc, adopted, running := st.proc, st.adopted, st.running
	st.mu.Unlock()
	if proc == nil || adopted || !running || pidAlive(orphan.Process.Pid) {
		t.Fatalf("cold restore owned=%v adopted=%v running=%v orphanAlive=%v", proc != nil, adopted, running, pidAlive(orphan.Process.Pid))
	}
}

func TestRestoreEnabledWindowsConsoleOrphanCancellation(t *testing.T) {
	for _, scenario := range []string{"caller already cancelled", "manager already shutting down"} {
		t.Run(scenario, func(t *testing.T) {
			m, dir, port := restoreOrphanManifest(t)
			p := m.Platforms[hostKey()]
			p.Runtime.Stop = &StopSpec{Signal: "term", GraceS: 10}
			m.Platforms[hostKey()] = p
			orphan := spawnFakeListener(t, p.Runtime.Bin, port)
			ex := restoreOrphanExecutor(t, m, dir)
			if err := ex.desired.set("fake", true, intentLocal); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if scenario == "caller already cancelled" {
				cancel()
			} else {
				ex.StopAll()
			}
			started := time.Now()
			err := ex.RestoreEnabled(ctx)
			elapsed := time.Since(started)
			if elapsed >= 5*time.Second {
				t.Errorf("cancelled reclaim took %s, must not wait full 10s grace", elapsed)
			}
			if err == nil {
				t.Error("cancelled restore silently reported success")
			}
			// Cancellation predates admission, so no signal may be sent. A
			// killed orphan followed by a cancelled launch is a partial mutation.
			if !pidAlive(orphan.Process.Pid) || !portServing(port) {
				t.Errorf("cancelled restore partially reclaimed orphan: pidAlive=%v serving=%v err=%v", pidAlive(orphan.Process.Pid), portServing(port), err)
			}
			st, stateErr := ex.state("fake")
			if stateErr != nil {
				t.Fatal(stateErr)
			}
			st.mu.Lock()
			proc, running, adopted := st.proc, st.running, st.adopted
			st.mu.Unlock()
			if proc != nil || running || adopted {
				t.Errorf("cancelled reclaim relaunched/adopted: owned=%v running=%v adopted=%v err=%v", proc != nil, running, adopted, err)
			}
			enabled, known, desiredErr := ex.desired.get("fake")
			if desiredErr != nil || !known || !enabled {
				t.Errorf("cancellation changed desired On: enabled=%v known=%v err=%v", enabled, known, desiredErr)
			}
		})
	}
}

func TestCallerCancellationDuringOrphanConfirmation(t *testing.T) {
	// Witness the post-signal confirmation seam without signalling the test
	// PID. Native taskkill cancellation is not exercised here; the console
	// orphan test separately checks actual forceful reclaim and cold launch.
	probed := make(chan struct{}, 1)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			select {
			case probed <- struct{}{}:
			default:
			}
		}
	}
	server.Start()
	t.Cleanup(server.Close)
	_, rawPort, err := net.SplitHostPort(server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(rawPort)
	if err != nil {
		t.Fatal(err)
	}
	m := testEngineManifest(fakeEngineBin)
	p := m.Platforms[hostKey()]
	p.Runtime.Port = port
	p.Runtime.Health = nil
	p.Runtime.Stop = &StopSpec{Signal: "term", GraceS: 10}
	m.Platforms[hostKey()] = p
	ex := desiredTestExecutor(t, m, t.TempDir())
	if err := ex.desired.set("fake", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	callerCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.opMu.Lock()
	ctx, finish := ex.startContext(callerCtx, st)
	confirmed := make(chan error, 1)
	go func() {
		defer st.opMu.Unlock()
		defer finish()
		confirmed <- ex.confirmOrphanStoppedContext(ctx, "fake", os.Getpid(), port, p.Runtime)
	}()
	select {
	case <-probed:
	case <-time.After(3 * time.Second):
		t.Fatal("orphan confirmation did not reach the listener")
	}
	select {
	case err := <-confirmed:
		t.Fatalf("orphan confirmation ended before caller cancellation: %v", err)
	default:
	}
	cancel()
	var confirmationErr error
	select {
	case confirmationErr = <-confirmed:
		if !errors.Is(confirmationErr, context.Canceled) {
			t.Fatalf("in-flight confirmation = %v, want caller cancellation", confirmationErr)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("caller cancellation did not release in-flight reclaim confirmation")
	}
	status, err := ex.Status("fake")
	if err != nil || status.Running || status.Healthy || status.Port != port {
		t.Errorf("cancelled reclaim status = %+v err=%v", status, err)
	}
	st.mu.Lock()
	fenced, owned, running, adopted := st.reclaimFailed, st.proc != nil, st.running, st.adopted
	st.mu.Unlock()
	if !fenced || owned || running || adopted {
		t.Errorf("cancelled reclaim fence=%v owned=%v running=%v adopted=%v", fenced, owned, running, adopted)
	}
	if !pidAlive(os.Getpid()) || !portServing(port) {
		t.Error("confirmation cancellation did not preserve the surviving listener")
	}
	reports := ex.Errors()
	if len(reports) != 1 || reports[0].ID != startFailedID("fake") || reports[0].Message != confirmationErr.Error() {
		t.Errorf("cancelled reclaim reports = %+v, want StartFailed with %v", reports, confirmationErr)
	}
	if enabled, known, desiredErr := ex.desired.get("fake"); desiredErr != nil || !known || !enabled {
		t.Errorf("caller cancellation changed desired On: enabled=%v known=%v err=%v", enabled, known, desiredErr)
	}
}
