// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/testbin"
)

func TestStopGrace(t *testing.T) {
	tests := []struct {
		name string
		stop *StopSpec
		want time.Duration
	}{
		{name: "default without stop spec", want: 5 * time.Second},
		{name: "default with zero grace", stop: &StopSpec{Signal: "term"}, want: 5 * time.Second},
		{name: "configured grace", stop: &StopSpec{Signal: "term", GraceS: 10}, want: 10 * time.Second},
		{name: "kill is immediate", stop: &StopSpec{Signal: "kill", GraceS: 10}, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := stopGrace(Runtime{Stop: tt.stop}); got != tt.want {
				t.Fatalf("stopGrace() = %s, want %s", got, tt.want)
			}
		})
	}
}

// spawnFakeListener owns exactly one copied fake-engine child and one waiter.
// Cleanup is registered before readiness/inspection and reaps on every path.
func spawnFakeListener(t *testing.T, binPath string, port int) *exec.Cmd {
	t.Helper()
	cmd := exec.Command(binPath)
	cmd.Env = overrideEnv(map[string]string{
		"FAKE_ADDR":   "127.0.0.1:" + strconv.Itoa(port),
		"OLLAMA_HOST": "127.0.0.1:" + strconv.Itoa(port),
	})
	configureSysProcAttr(cmd)
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	t.Logf("owned listener started: pid=%d port=%d path=%q", cmd.Process.Pid, port, binPath)
	t.Cleanup(func() {
		select {
		case err := <-done:
			t.Logf("owned listener reaped: pid=%d wait=%v", cmd.Process.Pid, err)
			return
		default:
		}
		if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
			t.Errorf("kill owned listener pid=%d: %v", cmd.Process.Pid, err)
		}
		select {
		case err := <-done:
			t.Logf("owned listener cleanup reaped: pid=%d wait=%v", cmd.Process.Pid, err)
		case <-time.After(3 * time.Second):
			t.Errorf("owned listener pid=%d was not reaped within 3s", cmd.Process.Pid)
		}
	})
	waitPortServing(t, port)
	pid, image, ok := pidOnPort(port)
	// TCP reachability can precede the OS inspection tool observing the
	// listener under load. Wait for ownership evidence before exercising Stop.
	deadline := time.Now().Add(10 * time.Second)
	for !ok || pid != cmd.Process.Pid || image == "" {
		if time.Now().After(deadline) {
			break
		}
		time.Sleep(100 * time.Millisecond)
		pid, image, ok = pidOnPort(port)
	}
	if !ok || pid != cmd.Process.Pid || image == "" {
		t.Fatalf("listener inspection = (%d, %q, %v), want owned pid %d and full image", pid, image, ok, cmd.Process.Pid)
	}
	want, err := filepath.EvalSymlinks(binPath)
	if err != nil {
		t.Fatalf("resolve fixture image: %v", err)
	}
	got, err := filepath.EvalSymlinks(image)
	if err != nil || !strings.EqualFold(got, want) {
		t.Fatalf("listener image = %q (canonical %q, err=%v), want %q", image, got, err, want)
	}
	t.Logf("owned listener inspected: pid=%d image=%q canonical=%q", pid, image, got)
	return cmd
}

// TestStopReclaimsOurOrphanOnManagedPort proves a user Stop of an adopted
// engine whose listener is our own managed binary (an orphan left on our
// port) terminates it, stays off across a health interval, and records OFF.
func TestStopReclaimsOurOrphanOnManagedPort(t *testing.T) {
	dir := testbin.Dir(t, "nvpair-engine-manager", "stop-orphan", "fakeorphan")
	bin := filepath.Join(dir, testbin.Exe("fakeorphan"))
	copyFile(t, fakeEngineBin, bin)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	spawnFakeListener(t, bin, port)

	m := testEngineManifest(bin) // Detect resolves st.binPath to bin (our managed binary)
	key := runtime.GOOS + "/" + runtime.GOARCH
	p := m.Platforms[key]
	p.Runtime.Port = port
	p.Runtime.Ready = &Probe{HTTP: "http://127.0.0.1:{port}/", Status: http.StatusOK, TimeoutS: 5}
	p.Runtime.Health = &Probe{HTTP: "http://127.0.0.1:{port}/", Status: http.StatusOK, IntervalS: 1}
	m.Platforms[key] = p

	ex := managedFixtureExecutor(t, m, dir)
	if st, err := ex.Status(m.Engine); err != nil || !st.Running {
		t.Fatalf("adopt orphan: status=%+v err=%v", st, err)
	}

	if err := ex.Stop(m.Engine); err != nil {
		t.Fatalf("stop must reclaim our own orphan, got err=%v", err)
	}
	if portServing(port) {
		t.Fatal("orphan on our managed port must be terminated by stop")
	}
	time.Sleep(1500 * time.Millisecond) // outlast a health interval
	if st, _ := ex.Status(m.Engine); st.Running || st.Healthy {
		t.Fatalf("engine must stay stopped after reclaim, got %+v", st)
	}
	if enabled, known, err := ex.desired.get(m.Engine); err != nil || !known || enabled {
		t.Fatalf("desired state = (%v, %v, %v), want known OFF", enabled, known, err)
	}
}

func TestStopDoesNotReclaimSameExternalImage(t *testing.T) {
	bin := filepath.Join(testbin.Dir(t, "nvpair-engine-manager", "stop-external-same-image", "external-ollama"), testbin.Exe("external-ollama"))
	copyFile(t, fakeEngineBin, bin)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	spawnFakeListener(t, bin, port)

	m := testEngineManifest(bin)
	key := runtime.GOOS + "/" + runtime.GOARCH
	p := m.Platforms[key]
	p.Runtime.Port = port
	m.Platforms[key] = p
	ex := newTestExecutor(t, m) // managed install directory is not bin's directory
	if st, err := ex.Status("fake"); err != nil || !st.Running {
		t.Fatalf("adopt external image: status=%+v err=%v", st, err)
	}
	if err := ex.Stop("fake"); err == nil || !strings.Contains(err.Error(), "external management") {
		t.Fatalf("external same-image stop error = %v", err)
	}
	if !portServing(port) {
		t.Fatal("external same-image listener was terminated")
	}
}

// TestStopDeclinesForeignListenerWithActionableError proves a user Stop of an
// adopted engine whose listener is a genuinely foreign process (a different
// image) is declined with an error naming the offending PID and image, the
// foreign process is left running, and OFF intent is still recorded.
func TestStopDeclinesForeignListenerWithActionableError(t *testing.T) {
	dir := testbin.Dir(t, "nvpair-engine-manager", "stop-managed-foreign", "ourengine")
	ourBin := filepath.Join(dir, testbin.Exe("ourengine"))
	foreignBin := filepath.Join(testbin.Dir(t, "nvpair-engine-manager", "stop-foreign", "foreignengine"), testbin.Exe("foreignengine"))
	copyFile(t, fakeEngineBin, ourBin)
	copyFile(t, fakeEngineBin, foreignBin)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	foreign := spawnFakeListener(t, foreignBin, port)

	m := testEngineManifest(ourBin) // Detect resolves st.binPath to ourBin, not the listener's image
	key := runtime.GOOS + "/" + runtime.GOARCH
	p := m.Platforms[key]
	p.Runtime.Port = port
	m.Platforms[key] = p

	ex := managedFixtureExecutor(t, m, dir)
	if st, err := ex.Status(m.Engine); err != nil || !st.Running {
		t.Fatalf("adopt foreign listener: status=%+v err=%v", st, err)
	}

	err = ex.Stop(m.Engine)
	if err == nil || !strings.Contains(err.Error(), "external management") {
		t.Fatalf("stop error = %v, want external-management decline", err)
	}
	if !strings.Contains(err.Error(), strconv.Itoa(foreign.Process.Pid)) {
		t.Fatalf("decline error must name the offending pid %d: %v", foreign.Process.Pid, err)
	}
	if !strings.Contains(err.Error(), filepath.Base(foreignBin)) {
		t.Fatalf("decline error must name the offending image %q: %v", foreignBin, err)
	}
	if !portServing(port) {
		t.Fatal("a genuinely foreign listener must be left running")
	}
	if !fileExists(ourBin) || !fileExists(foreignBin) {
		t.Fatal("refused stop must preserve managed and external executable files")
	}
	if enabled, known, err := ex.desired.get(m.Engine); err != nil || !known || enabled {
		t.Fatalf("desired state = (%v, %v, %v), want known OFF after a declined stop", enabled, known, err)
	}
}

func TestStopDeclinesLiteralDeletedSuffixListener(t *testing.T) {
	base := t.TempDir()
	managed := filepath.Join(base, "fake", "engine"+exeExt())
	distinct := managed + " (deleted)"
	if err := os.MkdirAll(filepath.Dir(managed), 0o755); err != nil {
		t.Fatal(err)
	}
	copyFile(t, fakeEngineBin, managed)
	copyFile(t, fakeEngineBin, distinct)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	child := spawnFakeListener(t, distinct, port)
	m := testEngineManifest(managed)
	key := runtime.GOOS + "/" + runtime.GOARCH
	p := m.Platforms[key]
	p.Runtime.Port = port
	m.Platforms[key] = p
	reg := NewRegistry()
	reg.engines[m.Engine] = m
	ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, base)
	if st, err := ex.Status(m.Engine); err != nil || !st.Running {
		t.Fatalf("adopt distinct suffixed listener: status=%+v err=%v", st, err)
	}
	err = ex.Stop(m.Engine)
	if err == nil || !strings.Contains(err.Error(), "external management") || !strings.Contains(err.Error(), strconv.Itoa(child.Process.Pid)) || !strings.Contains(err.Error(), " (deleted)") {
		t.Fatalf("distinct literal-suffix stop error=%v, want refusal naming actual pid/image", err)
	}
	if !portServing(port) || !fileExists(managed) || !fileExists(distinct) {
		t.Fatal("refused Stop must preserve the distinct live listener and both executable files")
	}
	if enabled, known, err := ex.desired.get(m.Engine); err != nil || !known || enabled {
		t.Fatalf("desired state=(%v,%v,%v), want known OFF after refusal", enabled, known, err)
	}
}

func TestStopReclaimsManagedLiteralDeletedSuffix(t *testing.T) {
	base := t.TempDir()
	bin := filepath.Join(base, "fake", "engine"+exeExt()+" (deleted)")
	if err := os.MkdirAll(filepath.Dir(bin), 0o755); err != nil {
		t.Fatal(err)
	}
	copyFile(t, fakeEngineBin, bin)
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	spawnFakeListener(t, bin, port)
	m := testEngineManifest(bin)
	key := runtime.GOOS + "/" + runtime.GOARCH
	p := m.Platforms[key]
	p.Runtime.Port = port
	m.Platforms[key] = p
	reg := NewRegistry()
	reg.engines[m.Engine] = m
	ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, base)
	if st, err := ex.Status(m.Engine); err != nil || !st.Running {
		t.Fatalf("adopt managed literal-suffix listener: status=%+v err=%v", st, err)
	}
	if err := ex.Stop(m.Engine); err != nil {
		t.Fatalf("reclaim exact managed literal-suffix image: %v", err)
	}
	if portServing(port) || !fileExists(bin) {
		t.Fatal("Stop must terminate the exact managed listener while preserving its literal-suffix file")
	}
	if enabled, known, err := ex.desired.get(m.Engine); err != nil || !known || enabled {
		t.Fatalf("desired state=(%v,%v,%v), want known OFF", enabled, known, err)
	}
}

func TestStopRejectsAdoptedProcessEngine(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	u, _ := url.Parse(srv.URL)
	port, _ := strconv.Atoi(u.Port())

	m := testEngineManifest(fakeEngineBin)
	p := m.Platforms[runtime.GOOS+"/"+runtime.GOARCH]
	p.Runtime.Port = port
	p.Runtime.Ready = &Probe{HTTP: srv.URL, Status: http.StatusOK}
	m.Platforms[runtime.GOOS+"/"+runtime.GOARCH] = p
	ex := newTestExecutor(t, m)
	if st, err := ex.Status("fake"); err != nil || !st.Running {
		t.Fatalf("adopt external engine: status=%+v err=%v", st, err)
	}
	stateChanged := make(chan EngineStatus, 1)
	ex.emit = func(method string, params any) {
		if method == "engine:state-changed" {
			stateChanged <- params.(EngineStatus)
		}
	}

	err := ex.Stop("fake")
	if err == nil || !strings.Contains(err.Error(), "external management") {
		t.Fatalf("stop error = %v, want actionable external-management error", err)
	}
	if st, _ := ex.Status("fake"); !st.Running || !st.Healthy {
		t.Fatalf("rejected stop must keep the live engine running, got %+v", st)
	}
	select {
	case st := <-stateChanged:
		if !st.Running || !st.Healthy {
			t.Fatalf("rejected stop emitted false terminal state: %+v", st)
		}
	default:
		t.Fatal("rejected stop did not emit the unchanged running state")
	}
}

func TestStopReconcilesAdoptedProcessAfterExternalStop(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	u, _ := url.Parse(srv.URL)
	port, _ := strconv.Atoi(u.Port())
	m := testEngineManifest(fakeEngineBin)
	p := m.Platforms[runtime.GOOS+"/"+runtime.GOARCH]
	p.Runtime.Port = port
	p.Runtime.Ready = &Probe{HTTP: srv.URL, Status: http.StatusOK}
	m.Platforms[runtime.GOOS+"/"+runtime.GOARCH] = p
	ex := newTestExecutor(t, m)
	if st, err := ex.Status("fake"); err != nil || !st.Running {
		t.Fatalf("adopt external engine: status=%+v err=%v", st, err)
	}

	srv.Close()
	if err := ex.Stop("fake"); err == nil {
		t.Fatal("listener closure alone proved external process termination")
	}
	if st, _ := ex.Status("fake"); !st.Running || st.Healthy || st.TerminationVerified || portServing(port) {
		t.Fatalf("closed endpoint lost unknown external ownership: %+v", st)
	}
	state, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	state.mu.Lock()
	state.reclaimPID = os.Getpid()
	state.mu.Unlock()
	ex.reconcilePresenceStateWithProcessProof(t.Context(), "fake", state, true, port, true,
		func(int) (int, string, bool) { return os.Getpid(), "fixture-owner", true },
		func(pid int) bool { return pid == os.Getpid() })
	if err := ex.Stop("fake"); err != nil {
		t.Fatalf("retry after positive external exit: %v", err)
	}
	if st, _ := ex.Status("fake"); st.Running || st.Healthy {
		t.Fatalf("closed external endpoint must reconcile to stopped, got %+v", st)
	}
}

func TestStopKeepsAdoptedProcessRunningWhileEndpointIsUnhealthy(t *testing.T) {
	var unhealthy atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if unhealthy.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	u, _ := url.Parse(srv.URL)
	port, _ := strconv.Atoi(u.Port())
	m := testEngineManifest(fakeEngineBin)
	p := m.Platforms[runtime.GOOS+"/"+runtime.GOARCH]
	p.Runtime.Port = port
	p.Runtime.Ready = &Probe{HTTP: srv.URL, Status: http.StatusOK}
	m.Platforms[runtime.GOOS+"/"+runtime.GOARCH] = p
	ex := newTestExecutor(t, m)
	if st, err := ex.Status("fake"); err != nil || !st.Running {
		t.Fatalf("adopt external engine: status=%+v err=%v", st, err)
	}

	unhealthy.Store(true)
	if err := ex.Stop("fake"); err == nil || !strings.Contains(err.Error(), "external management") {
		t.Fatalf("stop error = %v, want live external-management error", err)
	}
	if st, _ := ex.Status("fake"); !st.Running {
		t.Fatalf("an unhealthy but reachable endpoint must not report stopped, got %+v", st)
	}
}

func TestCommandStopFailureKeepsLiveEngineRunning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	marker := filepath.Join(t.TempDir(), "stop-attempted")
	ex := commandStopTestExecutor(t, srv.URL, []string{fakeEngineBin, "failmark", marker}, 1, false)

	err := ex.Stop("command")
	if err == nil || !strings.Contains(err.Error(), "stop command") {
		t.Fatalf("stop error = %v, want command failure", err)
	}
	if !fileExists(marker) {
		t.Fatal("stop command did not run")
	}
	if st, _ := ex.Status("command"); !st.Running || !st.Healthy {
		t.Fatalf("failed stop must keep the live engine running, got %+v", st)
	}
}

func TestCommandStopFailureSucceedsWhenEndpointIsConfirmedDown(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "stop-attempted")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)
	closed := make(chan struct{})
	go func() {
		for !fileExists(marker) {
			time.Sleep(10 * time.Millisecond)
		}
		srv.Close()
		close(closed)
	}()
	ex := commandStopTestExecutor(t, srv.URL, []string{fakeEngineBin, "failmark", marker}, 2, false)

	if err := ex.Stop("command"); err != nil {
		t.Fatalf("nonzero stop command with a closed endpoint must succeed: %v", err)
	}
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		t.Fatal("test endpoint did not close")
	}
	if st, _ := ex.Status("command"); st.Running || st.Healthy {
		t.Fatalf("closed endpoint must report stopped, got %+v", st)
	}
}

func TestCommandStopWithoutReadinessProbeUsesConfiguredCommand(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "stop-attempted")
	ex := commandStopTestExecutor(t, "http://127.0.0.1:1", []string{fakeEngineBin, "touch", marker}, 1, false)
	st, _ := ex.state("command")
	st.plat.Runtime.Ready = nil

	if err := ex.Stop("command"); err != nil {
		t.Fatalf("stop without readiness probe: %v", err)
	}
	if !fileExists(marker) {
		t.Fatal("configured stop command did not run")
	}
	if status, _ := ex.Status("command"); status.Running || status.Healthy {
		t.Fatalf("legacy no-probe command stop must report stopped, got %+v", status)
	}
}

func TestCommandStopRejectsSuccessWhileEndpointIsLive(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	marker := filepath.Join(t.TempDir(), "stop-attempted")
	ex := commandStopTestExecutor(t, srv.URL, []string{fakeEngineBin, "touch", marker}, 1, false)

	err := ex.Stop("command")
	if err == nil || !strings.Contains(err.Error(), "still serving") {
		t.Fatalf("stop error = %v, want endpoint-still-live error", err)
	}
	if st, _ := ex.Status("command"); !st.Running || !st.Healthy {
		t.Fatalf("live endpoint must remain reported running, got %+v", st)
	}
}

func TestCommandStopRejectsSuccessWhileEndpointIsUnhealthy(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	marker := filepath.Join(t.TempDir(), "stop-attempted")
	ex := commandStopTestExecutor(t, srv.URL, []string{fakeEngineBin, "touch", marker}, 1, false)

	err := ex.Stop("command")
	if err == nil || !strings.Contains(err.Error(), "still serving") {
		t.Fatalf("stop error = %v, want endpoint-still-live error", err)
	}
	if st, _ := ex.Status("command"); !st.Running {
		t.Fatalf("an unhealthy but reachable endpoint must remain running, got %+v", st)
	}
}

func TestCommandStopFailureRestartsHealthMonitoring(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	marker := filepath.Join(t.TempDir(), "stop-attempted")
	ex := commandStopTestExecutor(t, srv.URL, []string{fakeEngineBin, "touch", marker}, 1, false)
	st, _ := ex.state("command")
	st.plat.Runtime.Health = &Probe{HTTP: srv.URL, Status: http.StatusOK, IntervalS: 1}
	t.Cleanup(func() {
		st.mu.Lock()
		if st.healthStop != nil {
			st.healthStop()
		}
		st.mu.Unlock()
		srv.Close()
	})

	if err := ex.Stop("command"); err == nil {
		t.Fatal("expected the live endpoint to reject the stop")
	}
	srv.Close()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if status := ex.snapshot("command", st); status.Running && !status.Healthy {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("health monitoring did not resume after failed stop: %+v", ex.snapshot("command", st))
}

func TestCommandStopReportsStoppedOnlyAfterEndpointCloses(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "stop-attempted")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)
	closed := make(chan struct{})
	go func() {
		for !fileExists(marker) {
			time.Sleep(10 * time.Millisecond)
		}
		srv.Close()
		close(closed)
	}()
	ex := commandStopTestExecutor(t, srv.URL, []string{fakeEngineBin, "touch", marker}, 2, false)

	if err := ex.Stop("command"); err != nil {
		t.Fatalf("stop: %v", err)
	}
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		t.Fatal("test endpoint did not close")
	}
	if st, _ := ex.Status("command"); st.Running || st.Healthy {
		t.Fatalf("closed endpoint must report stopped, got %+v", st)
	}
}

func TestCommandStopRefusesAdoptedExternalOwner(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "unscoped-stop-attempted")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
	t.Cleanup(srv.Close)
	ex := commandStopTestExecutor(t, srv.URL, []string{fakeEngineBin, "touch", marker}, 2, true)
	if err := ex.Stop("command"); err == nil || !strings.Contains(err.Error(), "external management") {
		t.Fatalf("adopted command Stop=%v, want external-ownership refusal", err)
	}
	if _, err := os.Stat(marker); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("unscoped stop command ran or marker could not be inspected: %v", err)
	}
	status, err := ex.Status("command")
	if err != nil || !status.Running || !status.Healthy {
		t.Fatalf("external endpoint state=%+v err=%v", status, err)
	}
	enabled, known, err := ex.desired.get("command")
	if err != nil || !known || enabled {
		t.Fatalf("refused local Stop lost Off intent: enabled=%v known=%v err=%v", enabled, known, err)
	}
}

func commandStopTestExecutor(t *testing.T, readyURL string, stopCmd []string, graceS int, adopted bool) *Executor {
	t.Helper()
	u, err := url.Parse(readyURL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	key := runtime.GOOS + "/" + runtime.GOARCH
	m := &Manifest{
		Engine: "command", DisplayName: "Command Engine", ManifestVersion: 1,
		Platforms: map[string]Platform{
			key: {
				Detect: []string{fakeEngineBin},
				Runtime: Runtime{
					Mode:  "command",
					Port:  port,
					Ready: &Probe{HTTP: readyURL, Status: http.StatusOK},
					Stop:  &StopSpec{Cmd: stopCmd, GraceS: graceS},
				},
			},
		},
	}
	ex := newTestExecutor(t, m)
	st, err := ex.state("command")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.installed = true
	st.running = true
	st.healthy = true
	st.adopted = adopted
	st.port = port
	st.mu.Unlock()
	return ex
}
