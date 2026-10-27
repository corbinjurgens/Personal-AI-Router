// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func diagnosisRefusedPort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("allocate refused port: %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	if err := l.Close(); err != nil {
		t.Fatalf("close refused listener: %v", err)
	}
	return port
}

func newDiagnosisExecutor(t *testing.T, endpoint string, port int) (*Executor, *engineState, string, string) {
	t.Helper()
	dir := t.TempDir()
	started, stopped := filepath.Join(dir, "started"), filepath.Join(dir, "stopped")
	m := testEngineManifest(fakeEngineBin)
	key := runtime.GOOS + "/" + runtime.GOARCH
	p := m.Platforms[key]
	p.Runtime = Runtime{
		Mode: "command", Port: port,
		Start:        [][]string{{fakeEngineBin, "touch", started}},
		Stop:         &StopSpec{Cmd: []string{fakeEngineBin, "touch", stopped}},
		Ready:        &Probe{HTTP: endpoint, Status: http.StatusOK, TimeoutS: 1},
		Health:       &Probe{HTTP: endpoint, Status: http.StatusOK, IntervalS: 1},
		AuthStatuses: []int{http.StatusUnauthorized, http.StatusForbidden},
		AppProcesses: []string{"Fake App"},
	}
	m.Platforms[key] = p
	ex := newTestExecutor(t, m)
	ex.processes = func(context.Context) ([]string, error) { return []string{"Fake App"}, nil }
	st, err := ex.state("fake")
	if err != nil {
		t.Fatalf("engine state: %v", err)
	}
	t.Cleanup(func() {
		st.mu.Lock()
		if st.healthStop != nil {
			st.healthStop()
		}
		st.mu.Unlock()
	})
	return ex, st, started, stopped
}

func diagnosisServerPort(t *testing.T, srv *httptest.Server) int {
	t.Helper()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("parse fake server URL: %v", err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatalf("parse fake server port: %v", err)
	}
	return port
}

// The official fake stop command creates a marker; close the stand-in daemon
// when it appears so cleanup observes a real listener refusal.
func closeDiagnosisServerOnStop(t *testing.T, srv *httptest.Server, stopped string) {
	t.Helper()
	quit, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(5 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-quit:
				return
			case <-ticker.C:
				if fileExists(stopped) {
					srv.Close()
					return
				}
			}
		}
	}()
	t.Cleanup(func() { close(quit); <-done; srv.Close() })
}

func waitDiagnosisReport(t *testing.T, ex *Executor, id string) serviceError {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		for _, report := range ex.reporter.snapshot() {
			if report.ID == id {
				return report
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("missing report %s: %+v", id, ex.reporter.snapshot())
	return serviceError{}
}

func assertDiagnosisReport(t *testing.T, ex *Executor, id, message, severity, action string) {
	t.Helper()
	r := waitDiagnosisReport(t, ex, id)
	if r.Message != message || r.Severity != severity || r.Action != action || r.EngineType != "fake" {
		t.Fatalf("report = %+v, want message %q, severity %q, action %q, engine fake", r, message, severity, action)
	}
}

func TestDiagnoseConnection(t *testing.T) {
	for _, tc := range []struct {
		name        string
		status      int
		images      []string
		listErr     error
		noFields    bool
		noProcesses bool
		want        connectionDiagnosis
	}{
		{name: "401 gives auth required", status: http.StatusUnauthorized, listErr: errors.New("unused lister"), want: diagnosisAuthRequired},
		{name: "403 gives auth required", status: http.StatusForbidden, want: diagnosisAuthRequired},
		{name: "503 with matching process", status: http.StatusServiceUnavailable, images: []string{`C:\Apps\Fake App.EXE`}, want: diagnosisAppRunning},
		{name: "refused with matching process", images: []string{"Fake App"}, want: diagnosisAppRunning},
		{name: "refused with no matching process", images: []string{"Other App"}, want: diagnosisNotRunning},
		{name: "lister error", listErr: errors.New("listing unavailable"), want: diagnosisNone},
		{name: "no declared fields", status: http.StatusUnauthorized, noFields: true, want: diagnosisNone},
		{name: "auth only without auth evidence", status: http.StatusServiceUnavailable, noProcesses: true, want: diagnosisNone},
	} {
		t.Run(tc.name, func(t *testing.T) {
			port := diagnosisRefusedPort(t)
			endpoint := fmt.Sprintf("http://127.0.0.1:%d/", port)
			if tc.status != 0 {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(tc.status) }))
				defer srv.Close()
				endpoint, port = srv.URL, diagnosisServerPort(t, srv)
			}
			ex, st, _, _ := newDiagnosisExecutor(t, endpoint, port)
			calls := 0
			ex.processes = func(context.Context) ([]string, error) { calls++; return tc.images, tc.listErr }
			if tc.noFields {
				st.plat.Runtime.AuthStatuses, st.plat.Runtime.AppProcesses = nil, nil
			}
			if tc.noProcesses {
				st.plat.Runtime.AppProcesses = nil
			}
			got, err := ex.diagnoseConnection(context.Background(), st, port)
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("diagnosis = %v, want %v", got, tc.want)
			}
			if (tc.want == diagnosisAuthRequired || tc.noFields) && calls != 0 {
				t.Fatalf("process lister called %d times without need", calls)
			}
		})
	}
}

func TestStartReportsAppRunningWhenRefused(t *testing.T) {
	port := diagnosisRefusedPort(t)
	ex, _, _, stopped := newDiagnosisExecutor(t, fmt.Sprintf("http://127.0.0.1:%d/", port), port)
	if err := ex.Start(context.Background(), "fake"); err == nil {
		t.Fatal("start succeeded without a listener")
	}
	assertDiagnosisReport(t, ex, startFailedID("fake"), "Can't connect to Fake Engine. Does it have an API key set, or is its server off?", "error", "retry")
	if !fileExists(stopped) {
		t.Fatal("readiness failure did not run cleanup stop")
	}
}

func TestStartCommandFailureIsDiagnosed(t *testing.T) {
	port := diagnosisRefusedPort(t)
	ex, st, _, stopped := newDiagnosisExecutor(t, fmt.Sprintf("http://127.0.0.1:%d/", port), port)
	st.plat.Runtime.Start = [][]string{{fakeEngineBin, "failmark", filepath.Join(t.TempDir(), "failed")}}
	if err := ex.Start(context.Background(), "fake"); err == nil {
		t.Fatal("start succeeded with a failed command")
	}
	assertDiagnosisReport(t, ex, startFailedID("fake"), "Can't connect to Fake Engine. Does it have an API key set, or is its server off?", "error", "retry")
	if !fileExists(stopped) {
		t.Fatal("failed command did not run cleanup stop")
	}
}

func TestStartDiagnosisUnavailableKeepsOriginalError(t *testing.T) {
	for _, scenario := range []string{"no declared fields", "process listing failed"} {
		t.Run(scenario, func(t *testing.T) {
			port := diagnosisRefusedPort(t)
			ex, st, _, _ := newDiagnosisExecutor(t, fmt.Sprintf("http://127.0.0.1:%d/", port), port)
			if scenario == "no declared fields" {
				st.plat.Runtime.AuthStatuses, st.plat.Runtime.AppProcesses = nil, nil
			}
			ex.processes = func(context.Context) ([]string, error) { return nil, errors.New("process listing unavailable") }
			err := ex.Start(context.Background(), "fake")
			want := `engine "fake" did not become ready: not ready after 1s`
			if err == nil || err.Error() != want {
				t.Fatalf("start error = %v, want %q", err, want)
			}
			assertDiagnosisReport(t, ex, startFailedID("fake"), want, "error", "retry")
		})
	}
}

func TestProcessPresenceNeverMakesEngineReady(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusUnauthorized) }))
	ex, _, _, stopped := newDiagnosisExecutor(t, srv.URL, diagnosisServerPort(t, srv))
	closeDiagnosisServerOnStop(t, srv, stopped)
	if err := ex.Start(context.Background(), "fake"); err == nil {
		t.Fatal("process presence made the engine ready")
	}
	status, err := ex.Status("fake")
	if err != nil || status.Running || status.Healthy {
		t.Fatalf("status = %+v, error = %v, want stopped and unhealthy", status, err)
	}
}

func TestHealthFailureNamesAuthRequirement(t *testing.T) {
	var response atomic.Int32
	response.Store(http.StatusOK)
	var started string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if !fileExists(started) {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(int(response.Load()))
	}))
	defer srv.Close()
	ex, _, marker, _ := newDiagnosisExecutor(t, srv.URL, diagnosisServerPort(t, srv))
	started = marker
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	response.Store(http.StatusUnauthorized)
	assertDiagnosisReport(t, ex, unhealthyID("fake"), diagnosisMessage("Fake Engine", diagnosisAuthRequired), "warning", "none")
	response.Store(http.StatusOK)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if len(ex.reporter.snapshot()) == 0 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("health recovery left reports: %+v", ex.reporter.snapshot())
}

func TestAdoptedHealthFailureReportedAfterStatusRead(t *testing.T) {
	var response atomic.Int32
	response.Store(http.StatusOK)
	// Hold the failed health response until Status has changed st.healthy.
	healthEntered, releaseHealth := make(chan struct{}), make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			select {
			case <-healthEntered:
			default:
				close(healthEntered)
			}
			<-releaseHealth
		}
		w.WriteHeader(int(response.Load()))
	}))
	defer srv.Close()
	defer close(releaseHealth)
	ex, st, started, _ := newDiagnosisExecutor(t, srv.URL, diagnosisServerPort(t, srv))
	st.plat.Runtime.Health.HTTP = srv.URL + "/health"
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("adopt: %v", err)
	}
	if fileExists(started) {
		t.Fatal("adoption ran the start command")
	}
	response.Store(http.StatusUnauthorized)
	status, err := ex.Status("fake")
	if err != nil || !status.Running || status.Healthy {
		t.Fatalf("status before health report = %+v, error = %v", status, err)
	}
	select {
	case <-healthEntered:
	case <-time.After(5 * time.Second):
		t.Fatal("health loop did not probe")
	}
	releaseHealth <- struct{}{}
	assertDiagnosisReport(t, ex, unhealthyID("fake"), diagnosisMessage("Fake Engine", diagnosisAuthRequired), "warning", "none")
}

func TestDiagnosisMessage(t *testing.T) {
	for _, tc := range []struct {
		name      string
		diagnosis connectionDiagnosis
		want      string
	}{
		{"auth required", diagnosisAuthRequired, "Fake Engine is running but requires an API key, and PAIR does not have one. Turn off API authentication in Fake Engine's server settings, then try again."},
		{"app running", diagnosisAppRunning, "Can't connect to Fake Engine. Does it have an API key set, or is its server off?"},
		{"not running", diagnosisNotRunning, "Can't connect to Fake Engine, and no Fake Engine process is running. Open Fake Engine, then try again."},
		{"no diagnosis", diagnosisNone, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := diagnosisMessage("Fake Engine", tc.diagnosis); got != tc.want {
				t.Fatalf("message = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestWaitReadyAuthConfirmations(t *testing.T) {
	t.Run("non auth response resets consecutive count", func(t *testing.T) {
		responses := []int{http.StatusUnauthorized, http.StatusForbidden, http.StatusServiceUnavailable, http.StatusUnauthorized, http.StatusForbidden, http.StatusUnauthorized}
		var calls atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			i := int(calls.Add(1)) - 1
			if i >= len(responses) {
				i = len(responses) - 1
			}
			w.WriteHeader(responses[i])
		}))
		defer srv.Close()
		ex, st, _, _ := newDiagnosisExecutor(t, srv.URL, diagnosisServerPort(t, srv))
		st.plat.Runtime.Ready.TimeoutS = 10
		if err := ex.waitReady(context.Background(), st.plat.Runtime.Ready, st.port, st.plat.Runtime.AuthStatuses, candidateCredential{port: st.port}); !errors.Is(err, errAuthRequired) {
			t.Fatalf("waitReady error = %v, want auth required", err)
		}
		if calls.Load() != int32(len(responses)) {
			t.Fatalf("probe calls = %d, want %d", calls.Load(), len(responses))
		}
	})
	t.Run("undeclared auth status keeps timeout", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusUnauthorized) }))
		defer srv.Close()
		ex, st, _, _ := newDiagnosisExecutor(t, srv.URL, diagnosisServerPort(t, srv))
		err := ex.waitReady(context.Background(), st.plat.Runtime.Ready, st.port, nil, candidateCredential{port: st.port})
		if err == nil || err.Error() != "not ready after 1s" {
			t.Fatalf("waitReady error = %v, want original timeout wording", err)
		}
	})
	t.Run("cancellation keeps original cause without process listing", func(t *testing.T) {
		port := diagnosisRefusedPort(t)
		ex, st, _, _ := newDiagnosisExecutor(t, fmt.Sprintf("http://127.0.0.1:%d/", port), port)
		ex.processes = func(context.Context) ([]string, error) {
			t.Fatal("cancelled start attempted process diagnosis")
			return nil, nil
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		err := ex.waitReady(ctx, st.plat.Runtime.Ready, port, st.plat.Runtime.AuthStatuses, candidateCredential{port: port})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("waitReady error = %v, want cancellation", err)
		}
		fallback := fmt.Errorf("readiness: %w", err)
		if got := ex.diagnoseStartFailure(ctx, st, port, err, fallback); got != fallback {
			t.Fatalf("cancelled diagnosis replaced failure: %v", got)
		}
	})
}

func TestProbeStatusPreservesIdentity(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
		want   bool
	}{
		{"matching JSON", http.StatusOK, `{"role":"router"}`, true},
		{"wrong identity", http.StatusOK, `{"role":"model"}`, false},
		{"invalid JSON", http.StatusOK, `{`, false},
		{"auth with invalid JSON", http.StatusUnauthorized, `{`, false},
		{"oversized JSON", http.StatusOK, strings.Repeat(" ", maxProbeJSONBytes+1), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var identityHeader atomic.Bool
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				identityHeader.Store(r.Header.Get(engineIdentityProbeHeader) == "1")
				w.WriteHeader(tc.status)
				if _, err := w.Write([]byte(tc.body)); err != nil {
					t.Errorf("write fake probe response: %v", err)
				}
			}))
			defer srv.Close()
			ex, st, _, _ := newDiagnosisExecutor(t, srv.URL, diagnosisServerPort(t, srv))
			p := st.plat.Runtime.Ready
			p.JSONMatch = &ProbeJSONMatch{Field: "role", Value: "router"}
			ok, status, err := ex.probeStatus(context.Background(), p, st.port, candidateCredential{port: st.port})
			if err != nil {
				t.Fatal(err)
			}
			if ok != tc.want || status != tc.status || !identityHeader.Load() {
				t.Fatalf("probe = (%v, %d), identity header = %v; want (%v, %d), header true", ok, status, identityHeader.Load(), tc.want, tc.status)
			}
			got, err := ex.probe(context.Background(), p, st.port, candidateCredential{port: st.port})
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("boolean probe = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestProbeStatusWithoutHTTPResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
	defer srv.Close()
	port := diagnosisServerPort(t, srv)
	ex, _, _, _ := newDiagnosisExecutor(t, srv.URL, port)
	refused := diagnosisRefusedPort(t)
	for _, tc := range []struct {
		name  string
		probe *Probe
		want  bool
	}{
		{"no probe", nil, true},
		{"TCP reachable", &Probe{TCP: fmt.Sprintf("127.0.0.1:%d", port)}, true},
		{"TCP refused", &Probe{TCP: fmt.Sprintf("127.0.0.1:%d", refused)}, false},
		{"HTTP refused", &Probe{HTTP: fmt.Sprintf("http://127.0.0.1:%d/", refused)}, false},
		{"unresolved placeholder", &Probe{HTTP: "http://127.0.0.1:{missing}/"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ok, status, err := ex.probeStatus(context.Background(), tc.probe, port, candidateCredential{port: port})
			if err != nil {
				t.Fatal(err)
			}
			if ok != tc.want || status != 0 {
				t.Fatalf("probe = (%v, %d), want (%v, 0)", ok, status, tc.want)
			}
		})
	}
}

func TestHealthDiagnosisDiscardsStaleRun(t *testing.T) {
	for _, scenario := range []string{"new generation", "stopped", "cancelled"} {
		t.Run(scenario, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) }))
			defer srv.Close()
			ex, st, _, _ := newDiagnosisExecutor(t, srv.URL, diagnosisServerPort(t, srv))
			entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			defer unblock()
			ex.processes = func(context.Context) ([]string, error) {
				close(entered)
				<-release
				return []string{"Fake App"}, nil
			}
			st.running, st.healthy, st.gen = true, true, 1
			go func() { defer close(done); ex.runHealth(ctx, st, "fake", st.port, 1) }()
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("health diagnosis did not reach lister")
			}
			// Acquiring mu while the lister is blocked proves diagnosis performs
			// its I/O outside the state mutex.
			st.mu.Lock()
			switch scenario {
			case "new generation":
				st.gen++
				st.healthy = true
			case "stopped":
				st.running = false
			case "cancelled":
				cancel()
			}
			st.mu.Unlock()
			unblock()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("stale health loop did not exit")
			}
			if reports := ex.reporter.snapshot(); len(reports) != 0 {
				t.Fatalf("stale diagnosis published: %+v", reports)
			}
			if scenario == "new generation" && !st.healthy {
				t.Fatal("stale diagnosis changed the new run's health")
			}
		})
	}
}
