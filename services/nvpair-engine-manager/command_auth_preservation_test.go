// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"
	"time"
)

func TestCommandAuthPreservesExistingServer(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
	}{{"unauthorized", http.StatusUnauthorized}, {"forbidden", http.StatusForbidden}} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(tc.status) }))
			defer srv.Close()
			ex, st, started, stopped := newDiagnosisExecutor(t, srv.URL, diagnosisServerPort(t, srv))
			st.plat.Runtime.Ready.TimeoutS = 60
			want := "Fake Engine is running but requires an API key, and PAIR does not have one. Turn off API authentication in Fake Engine's server settings, then try again."
			begin := time.Now()
			if err := ex.Start(context.Background(), "fake"); err == nil || err.Error() != want {
				t.Fatalf("start error = %v, want %q", err, want)
			}
			if elapsed := time.Since(begin); elapsed >= 10*time.Second {
				t.Fatalf("auth failure took %s, want less than 10s", elapsed)
			}
			assertDiagnosisReport(t, ex, startFailedID("fake"), want, "error", "retry")
			for _, marker := range []string{started, stopped} {
				if _, err := os.Stat(marker); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("marker %s exists or cannot be checked: %v", marker, err)
				}
			}
			resp, err := http.Get(srv.URL)
			if err != nil {
				t.Fatalf("listener is unreachable: %v", err)
			}
			resp.Body.Close()
			if resp.StatusCode != tc.status {
				t.Fatalf("listener status = %d, want %d", resp.StatusCode, tc.status)
			}
			st.mu.Lock()
			running, healthy, adopted := st.running, st.healthy, st.adopted
			st.mu.Unlock()
			if running || healthy || adopted {
				t.Fatalf("state: running=%t healthy=%t adopted=%t", running, healthy, adopted)
			}
		})
	}
}

func TestCommandAuthInitiallyRefusedStartsAndCleansUp(t *testing.T) {
	port := diagnosisRefusedPort(t)
	ex, _, started, stopped := newDiagnosisExecutor(t, fmt.Sprintf("http://127.0.0.1:%d/", port), port)
	if err := ex.Start(context.Background(), "fake"); err == nil {
		t.Fatal("start succeeded without a listener")
	}
	if !fileExists(started) || !fileExists(stopped) {
		t.Fatal("expected start and cleanup stop markers")
	}
}

func TestCommandAuthPostStartFailureCleansUp(t *testing.T) {
	var started atomic.Pointer[string]
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if marker := started.Load(); marker != nil && fileExists(*marker) {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	ex, _, marker, stopped := newDiagnosisExecutor(t, srv.URL, diagnosisServerPort(t, srv))
	started.Store(&marker)
	closeDiagnosisServerOnStop(t, srv, stopped)
	if err := ex.Start(context.Background(), "fake"); err == nil {
		t.Fatal("start succeeded after authentication failure")
	}
	if !fileExists(marker) || !fileExists(stopped) {
		t.Fatal("expected start and cleanup stop markers")
	}
}

func TestCommandAuthHealthyServerAdopted(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
	defer srv.Close()
	ex, st, started, stopped := newDiagnosisExecutor(t, srv.URL, diagnosisServerPort(t, srv))
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("adopt healthy server: %v", err)
	}
	st.mu.Lock()
	running, healthy, adopted := st.running, st.healthy, st.adopted
	st.mu.Unlock()
	if !running || !healthy || !adopted || fileExists(started) || fileExists(stopped) {
		t.Fatalf("state: running=%t healthy=%t adopted=%t, start=%t stop=%t", running, healthy, adopted, fileExists(started), fileExists(stopped))
	}
}

func TestCommandWithoutAuthStatusesStillStarts(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusUnauthorized) }))
	ex, st, started, stopped := newDiagnosisExecutor(t, srv.URL, diagnosisServerPort(t, srv))
	st.plat.Runtime.AuthStatuses = nil
	closeDiagnosisServerOnStop(t, srv, stopped)
	if err := ex.Start(context.Background(), "fake"); err == nil {
		t.Fatal("start succeeded against an unready server")
	}
	if !fileExists(started) || !fileExists(stopped) {
		t.Fatal("expected start and cleanup stop markers")
	}
}

func TestCommandAuthCancelledBeforeStart(t *testing.T) {
	port := diagnosisRefusedPort(t)
	ex, st, started, stopped := newDiagnosisExecutor(t, fmt.Sprintf("http://127.0.0.1:%d/", port), port)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := ex.bringUpCommand(ctx, context.Background(), st, "fake", st.plat.Runtime, port, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want cancellation", err)
	}
	if fileExists(started) || fileExists(stopped) {
		t.Fatal("cancelled start ran a command")
	}
}
