// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/testbin"
)

// captured records the executor's emitted notifications so tests can
// assert on them without a live parent. The Reporter's own ring (queried
// via Executor.Errors) captures reported errors even with a nil codec.
type captured struct {
	mu    sync.Mutex
	emits []string
}

func (c *captured) emit(method string, _ any) {
	c.mu.Lock()
	c.emits = append(c.emits, method)
	c.mu.Unlock()
}

func (c *captured) has(method string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, m := range c.emits {
		if m == method {
			return true
		}
	}
	return false
}

func capturingExecutor(t *testing.T, m *Manifest) (*Executor, *captured) {
	t.Helper()
	reg := NewRegistry()
	reg.engines[m.Engine] = m
	c := &captured{}
	ex := NewExecutor(reg, NewReporter(nil), c.emit, t.TempDir())
	ex.detectTimeout = 2 * time.Second // keep failure paths fast in tests
	return ex, c
}

// TestDownloadUnpinned verifies a fetch with no sha256 succeeds (over
// loopback http), allowing an unpinned HTTPS artifact source.
func TestDownloadUnpinned(t *testing.T) {
	payload := []byte("unpinned engine bytes")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(payload)
	}))
	defer srv.Close()
	ex := newTestExecutor(t, llamaTestEngineManifest(fakeEngineBin))
	p, err := ex.download(context.Background(), "fake", &Fetch{URL: srv.URL}) // no SHA256
	if err != nil {
		t.Fatalf("unpinned download should succeed, got %v", err)
	}
	defer os.Remove(p)
	if got, _ := os.ReadFile(p); string(got) != string(payload) {
		t.Fatalf("downloaded content mismatch: %q", got)
	}
}

// TestValidateDownloadURL pins the HTTPS-only policy (loopback http is the
// only plaintext exception — used by the live tests and a LAN mirror).
func TestValidateDownloadURL(t *testing.T) {
	for _, u := range []string{
		"https://example.com/x.zip", "https://example.com/y",
		"http://127.0.0.1:8080/x", "http://localhost/x", "http://[::1]:9/x",
	} {
		if err := validateDownloadURL(u); err != nil {
			t.Errorf("%q should be allowed: %v", u, err)
		}
	}
	for _, u := range []string{
		"http://example.com/x", "http://10.0.0.5/x", "ftp://h/y", "file:///etc/passwd",
	} {
		if err := validateDownloadURL(u); err == nil {
			t.Errorf("%q should be rejected", u)
		}
	}
}

// TestUninstallRetries proves a failing uninstall command is retried
// uninstallRetries times (the mitigation for a process that briefly holds
// files open after it is stopped).
func TestUninstallRetries(t *testing.T) {
	oldR, oldB := uninstallRetries, uninstallBackoff
	uninstallRetries, uninstallBackoff = 3, 5*time.Millisecond
	defer func() { uninstallRetries, uninstallBackoff = oldR, oldB }()

	dir := t.TempDir()
	baseDir := t.TempDir()
	binCopy := filepath.Join(testbin.Dir(t, "nvpair-engine-manager", "uninstall-retries", "fake"), testbin.Exe("fake"))
	copyFile(t, fakeEngineBin, binCopy)
	// Uninstall only runs for an executable inside the managed install directory.
	installed := filepath.Join(baseDir, "fake", testbin.Exe("fake"))
	if err := os.MkdirAll(filepath.Dir(installed), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(installed, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	marker := filepath.Join(dir, "attempts")

	m := llamaTestEngineManifest(installed)
	key := runtime.GOOS + "/" + runtime.GOARCH
	p := m.Platforms[key]
	p.Detect = []string{installed}                                       // installed → uninstall runs the command
	p.Uninstall = &Uninstall{Run: []string{binCopy, "failmark", marker}} // always exits 1, appends one byte
	m.Platforms[key] = p

	reg := NewRegistry()
	reg.engines[m.Engine] = m
	ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, baseDir)
	acceptModelMutations(ex)
	ex.detectTimeout = 2 * time.Second
	err := ex.Uninstall(context.Background(), "fake")
	if err == nil || !strings.Contains(err.Error(), "after 3 attempts") {
		t.Fatalf("expected uninstall failure after 3 attempts, got %v", err)
	}
	if data, _ := os.ReadFile(marker); len(data) != 3 {
		t.Fatalf("expected 3 uninstall attempts, marker has %d (%q)", len(data), data)
	}
}

func hasErr(errs []serviceError, id string) bool {
	for _, e := range errs {
		if e.ID == id {
			return true
		}
	}
	return false
}

func portOf(t *testing.T, rawURL string) int {
	t.Helper()
	u, err := url.Parse(rawURL)
	if err != nil {
		t.Fatal(err)
	}
	p, _ := strconv.Atoi(u.Port())
	return p
}

func waitFor(t *testing.T, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatal("condition not met within timeout")
}

func hostKey() string { return runtime.GOOS + "/" + runtime.GOARCH }

// TestActionReservedPlaceholderWins is the regression guard for the
// security fix: a caller-supplied param must NOT be able to override a
// reserved placeholder like {install_dir}.
func TestActionReservedPlaceholderWins(t *testing.T) {
	m := &Manifest{
		Engine: "fake", DisplayName: "Fake", ManifestVersion: 1,
		Platforms: map[string]Platform{hostKey(): {
			Detect:  []string{fakeEngineBin},
			Runtime: Runtime{Bin: fakeEngineBin},
		}},
		Actions: map[string]Action{"echo": {Cmd: []string{fakeEngineBin, "echo", "{install_dir}", "{model}"}}},
	}
	ex, _ := capturingExecutor(t, m)

	// Malicious params try to override {install_dir}. Reserved must win, so
	// the command echoes the real install directory (not the bogus value) and
	// the model value.
	res, err := ex.Action(context.Background(), "fake", "echo",
		json.RawMessage(`{"install_dir":"definitely-not-the-install-dir","model":"safevalue"}`))
	if err != nil {
		t.Fatalf("expected reserved {install_dir} to win and the command to run, got: %v", err)
	}
	if strings.Contains(string(res), "definitely-not-the-install-dir") {
		t.Fatalf("caller param overrode a reserved placeholder: %s", res)
	}
	if !strings.Contains(string(res), "safevalue") {
		t.Fatalf("unexpected result: %s", res)
	}
}

// TestReadinessTimeoutNoSpuriousExit guards the fix that a readiness
// timeout reports start-failed but NOT a phantom "exited unexpectedly".
func TestReadinessTimeoutNoSpuriousExit(t *testing.T) {
	m := &Manifest{
		Engine: "slow", DisplayName: "Slow", ManifestVersion: 1,
		Platforms: map[string]Platform{hostKey(): {
			Detect: []string{fakeEngineBin},
			Runtime: Runtime{
				Bin:   fakeEngineBin,
				Args:  []string{"noserve"}, // runs but never binds
				Ready: &Probe{TCP: "127.0.0.1:{port}", TimeoutS: 1},
			},
		}},
	}
	ex, _ := capturingExecutor(t, m)
	err := ex.Start(context.Background(), "slow")
	if err == nil || !strings.Contains(err.Error(), "did not become ready") {
		t.Fatalf("expected readiness timeout, got %v", err)
	}
	if !hasErr(ex.Errors(), startFailedID("slow")) {
		t.Fatalf("expected a start-failed error, got %+v", ex.Errors())
	}
	// Give the watcher a chance to (wrongly) fire before asserting silence.
	time.Sleep(400 * time.Millisecond)
	if hasErr(ex.Errors(), exitedID("slow")) {
		t.Fatal("must NOT report 'exited unexpectedly' for a readiness timeout")
	}
	state, _ := ex.state("slow")
	state.mu.Lock()
	proc, running := state.proc, state.running
	state.mu.Unlock()
	if proc != nil || running {
		t.Fatalf("timed-out engine was not cleaned up: proc=%v running=%v", proc != nil, running)
	}
	if st, err := ex.Status("slow"); err != nil || st.Running {
		t.Fatalf("status after timeout = %+v, err=%v; lifecycle lock must be released", st, err)
	}
}

// TestStartWaitsForDelayedReadinessWithinBudget is a reduced-time regression
// for engines whose initialization finishes after several failed probes. Start
// must keep the owned process alive until the configured finite budget expires.
func TestStartWaitsForDelayedReadinessWithinBudget(t *testing.T) {
	m := llamaTestEngineManifest(fakeEngineBin)
	p := m.Platforms[hostKey()]
	p.Runtime.Env["FAKE_START_DELAY"] = "750ms"
	p.Runtime.Ready.TimeoutS = 5
	m.Platforms[hostKey()] = p

	ex, _ := capturingExecutor(t, m)
	t.Cleanup(func() { _ = ex.Stop("fake") })
	startedAt := time.Now()
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("delayed start within readiness budget failed: %v", err)
	}
	if elapsed := time.Since(startedAt); elapsed < 700*time.Millisecond {
		t.Fatalf("fake engine did not exercise delayed readiness: elapsed=%s", elapsed)
	}
	if st, err := ex.Status("fake"); err != nil || !st.Running || !st.Healthy {
		t.Fatalf("delayed start status = %+v, err=%v", st, err)
	}
	if hasErr(ex.Errors(), startFailedID("fake")) {
		t.Fatalf("delayed success retained a start-failed error: %+v", ex.Errors())
	}
}

// TestConcurrentStartIsSerialized guards the per-engine op lock: many
// concurrent Starts must not race or double-spawn. Run under -race.
func TestConcurrentStartIsSerialized(t *testing.T) {
	ex, _ := capturingExecutor(t, llamaTestEngineManifest(fakeEngineBin))
	t.Cleanup(func() { _ = ex.Stop("fake") })
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _ = ex.Start(context.Background(), "fake") }()
	}
	wg.Wait()
	if st, _ := ex.Status("fake"); !st.Running {
		t.Fatalf("expected running after concurrent starts, got %+v", st)
	}
}

func TestInstallChecksumMismatchReported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("payload"))
	}))
	defer srv.Close()
	m := &Manifest{
		Engine: "fake", DisplayName: "Fake", ManifestVersion: 1,
		Platforms: map[string]Platform{hostKey(): {
			Detect:  []string{filepath.Join(t.TempDir(), "never"+exeExt())}, // stays not-installed
			Install: &Install{Fetch: &Fetch{URL: srv.URL, SHA256: "deadbeef"}, Run: []string{fakeEngineBin, "echo", "x"}, Mode: "user"},
			Runtime: Runtime{Bin: fakeEngineBin},
		}},
	}
	ex, c := capturingExecutor(t, m)
	err := ex.Install(context.Background(), "fake")
	if err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("expected checksum mismatch, got %v", err)
	}
	if !hasErr(ex.Errors(), installFailedID("fake")) {
		t.Fatalf("expected install-failed to be reported, got %+v", ex.Errors())
	}
	if !c.has("engine:install-progress") {
		t.Fatal("expected an install-progress notification")
	}
}

func TestUnexpectedExitReported(t *testing.T) {
	ex, _ := capturingExecutor(t, llamaTestEngineManifest(fakeEngineBin))
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	st, _ := ex.Status("fake")
	// The /exit route makes the fake engine os.Exit(1) — a crash.
	_, _ = http.Get(fmt.Sprintf("http://127.0.0.1:%d/exit", st.Port))
	waitFor(t, 6*time.Second, func() bool { s, _ := ex.Status("fake"); return !s.Running })
	// The exit watcher clears running before it reports, so wait for the report.
	waitFor(t, 6*time.Second, func() bool { return hasErr(ex.Errors(), exitedID("fake")) })
}

func TestNormalStopIsSilent(t *testing.T) {
	ex, _ := capturingExecutor(t, llamaTestEngineManifest(fakeEngineBin))
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := ex.Stop("fake"); err != nil {
		t.Fatalf("stop: %v", err)
	}
	time.Sleep(300 * time.Millisecond) // let any watcher fire
	if hasErr(ex.Errors(), exitedID("fake")) {
		t.Fatal("a normal stop must not report 'exited unexpectedly'")
	}
}

func TestRestartProcess(t *testing.T) {
	ex, _ := capturingExecutor(t, llamaTestEngineManifest(fakeEngineBin))
	t.Cleanup(func() { _ = ex.Stop("fake") })
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := ex.Restart(context.Background(), "fake"); err != nil {
		t.Fatalf("restart: %v", err)
	}
	if st, _ := ex.Status("fake"); !st.Running || !st.Healthy {
		t.Fatalf("expected running+healthy after restart, got %+v", st)
	}
}

func TestActionUnknownAndHTTPError(t *testing.T) {
	m := llamaTestEngineManifest(fakeEngineBin)
	m.Actions["err"] = Action{HTTP: &ActionHTTP{Method: "GET", Path: "/api/error"}}
	ex, _ := capturingExecutor(t, m)
	t.Cleanup(func() { _ = ex.Stop("fake") })
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	if _, err := ex.Action(context.Background(), "fake", "nope", nil); err == nil || !strings.Contains(err.Error(), "no action") {
		t.Fatalf("expected unknown-action error, got %v", err)
	}
	if _, err := ex.Action(context.Background(), "fake", "err", nil); err == nil || !strings.Contains(err.Error(), "HTTP 500") {
		t.Fatalf("expected HTTP 500 error, got %v", err)
	}
}

func TestExpandPathForms(t *testing.T) {
	t.Setenv("NVPAIR_TEST_VAR", "xyz")
	for _, tc := range []struct {
		name, goos, input, want string
	}{
		{"windows percent", "windows", "a/%NVPAIR_TEST_VAR%/b", "a/xyz/b"},
		{"windows dollar", "windows", "a/$NVPAIR_TEST_VAR/b", "a/$NVPAIR_TEST_VAR/b"},
		{"windows braced dollar", "windows", "a/${NVPAIR_TEST_VAR}/b", "a/${NVPAIR_TEST_VAR}/b"},
		{"unix dollar", "linux", "a/$NVPAIR_TEST_VAR/b", "a/xyz/b"},
		{"unix braced dollar", "linux", "a/${NVPAIR_TEST_VAR}/b", "a/xyz/b"},
		{"unix percent", "linux", "a/%NVPAIR_TEST_VAR%/b", "a/%NVPAIR_TEST_VAR%/b"},
		{"unix shell parameters", "linux", `tar -xzf "$1" -C "$3"`, `tar -xzf "$1" -C "$3"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := expandPathForOS(tc.input, tc.goos); got != tc.want {
				t.Errorf("expandPathForOS(%q, %q) = %q, want %q", tc.input, tc.goos, got, tc.want)
			}
		})
	}
	if home, err := os.UserHomeDir(); err == nil {
		for _, goos := range []string{"windows", "linux"} {
			if got, want := expandPathForOS("~/sub", goos), filepath.Join(home, "sub"); got != want {
				t.Errorf("%s ~ expansion: got %q, want %q", goos, got, want)
			}
		}
	}
}

// TestBundledManifestsGolden validates the shipped manifests load and
// validate (a typo would otherwise only surface as a runtime fatal).
func TestBundledManifestsGolden(t *testing.T) {
	reg := NewRegistry()
	if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
		t.Fatalf("bundled manifests invalid: %v", err)
	}
	for _, want := range []string{"llamacpp"} {
		m, ok := reg.Get(want)
		if !ok {
			t.Fatalf("missing bundled engine %q (have %v)", want, reg.Names())
		}
		if _, ok := m.Actions["list_models"]; !ok {
			t.Errorf("bundled %q is missing the list_models action", want)
		}
	}
}

func TestDownloadSizeCap(t *testing.T) {
	old := maxDownloadBytes
	maxDownloadBytes = 64
	t.Cleanup(func() { maxDownloadBytes = old })

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(make([]byte, 256)) // larger than the cap
	}))
	defer srv.Close()

	ex, _ := capturingExecutor(t, llamaTestEngineManifest(fakeEngineBin))
	if _, err := ex.download(context.Background(), "x", &Fetch{URL: srv.URL, SHA256: "abc"}); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("expected size-cap error, got %v", err)
	}
}

// ownedHealthManifest runs the fake engine as an owned process whose health
// probe targets healthURL, so a test can flip health independently of the
// process.
func ownedHealthManifest(t *testing.T, healthURL string) *Manifest {
	t.Helper()
	port, err := freePort()
	if err != nil {
		t.Fatal(err)
	}
	return &Manifest{
		Engine: "d", DisplayName: "Daemon", ManifestVersion: 1,
		Platforms: map[string]Platform{hostKey(): {
			Detect: []string{fakeEngineBin},
			Runtime: Runtime{
				Bin:    fakeEngineBin,
				Env:    map[string]string{"FAKE_ADDR": "127.0.0.1:{port}"},
				Port:   port,
				Stop:   &StopSpec{Signal: "term", GraceS: 3},
				Ready:  &Probe{TCP: "127.0.0.1:{port}", TimeoutS: 10},
				Health: &Probe{HTTP: healthURL, Status: 200, IntervalS: 1},
			},
		}},
	}
}

func TestHealthFlipReportsAndClears(t *testing.T) {
	var healthy atomic.Bool
	healthy.Store(true)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if healthy.Load() {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	}))
	defer srv.Close()

	ex, _ := capturingExecutor(t, ownedHealthManifest(t, srv.URL))
	t.Cleanup(func() { _ = ex.Stop("d") })
	if err := ex.Start(context.Background(), "d"); err != nil {
		t.Fatalf("start: %v", err)
	}

	healthy.Store(false)
	waitFor(t, 6*time.Second, func() bool { return hasErr(ex.Errors(), unhealthyID("d")) })
	healthy.Store(true)
	waitFor(t, 6*time.Second, func() bool { return !hasErr(ex.Errors(), unhealthyID("d")) })
}

// TestStopClearsUnhealthy guards that a deliberate stop clears a lingering
// unhealthy error (the same fix-family as clearing exited on start).
func TestStopClearsUnhealthy(t *testing.T) {
	var healthy atomic.Bool
	healthy.Store(true)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if healthy.Load() {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	}))
	t.Cleanup(srv.Close)

	ex, _ := capturingExecutor(t, ownedHealthManifest(t, srv.URL))
	if err := ex.Start(context.Background(), "d"); err != nil {
		t.Fatalf("start: %v", err)
	}
	healthy.Store(false)
	waitFor(t, 6*time.Second, func() bool { return hasErr(ex.Errors(), unhealthyID("d")) })
	if err := ex.Stop("d"); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if hasErr(ex.Errors(), unhealthyID("d")) {
		t.Fatal("stop must clear the lingering unhealthy error")
	}
}
