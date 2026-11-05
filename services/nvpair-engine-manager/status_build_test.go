// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

type statusBuildTransport struct {
	base  http.RoundTripper
	props func(*http.Request) (*http.Response, error)
}

func (s statusBuildTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL.Path == "/props" {
		return s.props(r)
	}
	return s.base.RoundTrip(r)
}

func statusBuildResponse(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}

func TestStatusBuildInfoAfterOwnedReadiness(t *testing.T) {
	m := llamaTestEngineManifest(fakeEngineBin)
	m.Family = "llamacpp"
	ex := newTestExecutor(t, m)
	calls := 0
	ex.client.Transport = statusBuildTransport{base: ex.client.Transport, props: func(r *http.Request) (*http.Response, error) {
		calls++
		if r.Method != http.MethodGet || r.URL.Query().Get("model") != "" || r.URL.Query().Get("autoload") != "false" || r.Header.Get(engineIdentityProbeHeader) != "1" {
			t.Errorf("build observation changed the non-loading request: %s %s", r.Method, r.URL)
		}
		return statusBuildResponse(http.StatusOK, `{"build_info":"b12345-test-build"}`), nil
	}}
	t.Cleanup(func() {
		if err := ex.Stop(m.Engine); err != nil {
			t.Errorf("stop fixture: %v", err)
		}
	})
	if err := ex.Start(context.Background(), m.Engine); err != nil {
		t.Fatal(err)
	}
	st, err := ex.state(m.Engine)
	if err != nil {
		t.Fatal(err)
	}
	wantID := requireNativeBuildID(t, st)
	status, err := ex.Status(m.Engine)
	if err != nil {
		t.Fatal(err)
	}
	if !status.Running || !status.Healthy || status.BuildInfo != "b12345-test-build" || status.BuildID != wantID || calls != 1 {
		t.Fatalf("status=%+v, props calls=%d; want running build b12345-test-build / %s", status, calls, wantID)
	}
	if err := ex.Stop(m.Engine); err != nil {
		t.Fatal(err)
	}
	status, err = ex.Status(m.Engine)
	if err != nil {
		t.Fatal(err)
	}
	if status.Running || status.BuildInfo != "" || status.BuildID != "" {
		t.Fatalf("stopped run retained build: %+v", status)
	}
}

func statusBuildFixture(t *testing.T) (*Executor, *engineState) {
	t.Helper()
	m := llamaTestEngineManifest(fakeEngineBin)
	m.Family = "llamacpp"
	ex := newTestExecutor(t, m)
	st, err := ex.state(m.Engine)
	if err != nil {
		t.Fatal(err)
	}
	nativeBuildTestRuntime(t, st, t.TempDir())
	st.running, st.healthy, st.runID = true, true, "owned-run"
	st.proc = &managedProc{done: make(chan struct{})}
	return ex, st
}

func TestStatusBuildInfoUnavailableClaimLeavesIdentityKnown(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		status     int
	}{
		{"props unavailable", "unavailable", http.StatusServiceUnavailable},
		{"malformed props", "invalid-json", http.StatusOK},
		{"claim omitted", `{}`, http.StatusOK},
		{"control in claim", `{"build_info":"bad\nclaim"}`, http.StatusOK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex, st := statusBuildFixture(t)
			ex.client.Transport = statusBuildTransport{props: func(*http.Request) (*http.Response, error) {
				return statusBuildResponse(tc.status, tc.body), nil
			}}
			ex.observeNativeBuild(context.Background(), st)
			status := ex.snapshot(st.manifest.Engine, st)
			if status.BuildInfo != "" || status.BuildID != requireNativeBuildID(t, st) || !status.Healthy {
				t.Fatalf("invalid/unavailable props must leave the claim unknown without failing readiness: %+v", status)
			}
		})
	}
}

func TestStatusBuildInfoPublicationFenced(t *testing.T) {
	test := func(name string, replace func(*Executor, *engineState)) {
		t.Run(name, func(t *testing.T) {
			ex, st := statusBuildFixture(t)
			entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
			ex.client.Transport = statusBuildTransport{props: func(*http.Request) (*http.Response, error) {
				close(entered)
				<-release
				return statusBuildResponse(http.StatusOK, `{"build_info":"old-run-claim"}`), nil
			}}
			go func() { ex.observeNativeBuild(context.Background(), st); close(done) }()
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				close(release)
				t.Fatal("build observation did not request props")
			}
			replace(ex, st)
			close(release)
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("build observation did not finish")
			}
			status := ex.snapshot(st.manifest.Engine, st)
			if status.BuildInfo != "" || status.BuildID != "" {
				t.Fatalf("obsolete observation published into a stopped/replaced run: %+v", status)
			}
		})
	}
	test("run replaced", func(_ *Executor, st *engineState) {
		st.mu.Lock()
		st.runID = "replacement-run"
		st.mu.Unlock()
	})
	test("stopped during props", func(ex *Executor, st *engineState) { ex.markStopped(st, st.manifest.Engine) })
	test("process exited during props", func(_ *Executor, st *engineState) { close(st.proc.done) })
}

func TestStatusBuildInfoFingerprintFailureLeavesIdentityUnknown(t *testing.T) {
	ex, st := statusBuildFixture(t)
	if err := os.Remove(st.binPath); err != nil {
		t.Fatal(err)
	}
	ex.client.Transport = statusBuildTransport{props: func(*http.Request) (*http.Response, error) {
		return statusBuildResponse(http.StatusOK, `{"build_info":"b12345-test-build"}`), nil
	}}
	ex.observeNativeBuild(context.Background(), st)
	status := ex.snapshot(st.manifest.Engine, st)
	if status.BuildInfo != "b12345-test-build" || status.BuildID != "" || !status.Healthy {
		t.Fatalf("file-read failure must leave identity unknown without failing readiness: %+v", status)
	}
}

func TestStatusBuildInfoInspectionUsesObservedBuild(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	f.st.buildInfo, f.st.buildID = "b12345-running", "sha256:"+strings.Repeat("a", 64)
	r := inspectFixture(t, f)
	if r.NativeBuild != "b12345-running" || r.NativeBuildID != "sha256:"+strings.Repeat("a", 64) {
		t.Fatalf("inspection lost running build: %+v", r)
	}
}

func TestStatusBuildInfoReclaimFailureClearsRun(t *testing.T) {
	ex, st := statusBuildFixture(t)
	st.buildInfo, st.buildID = "previous-build", "previous-id"
	ex.failOrphanReclaim(st, st.manifest.Engine, errors.New("synthetic reclaim failure"))
	if st.runID != "" || st.buildInfo != "" || st.buildID != "" {
		t.Fatalf("failed reclaim retained run/build: %q %q %q", st.runID, st.buildInfo, st.buildID)
	}
}

func TestStatusBuildInfoExternalAdoptionDoesNotClaimInstalledBuild(t *testing.T) {
	ex, st := statusBuildFixture(t)
	st.proc, st.running = nil, false
	st.buildInfo, st.buildID = "previous-owned-build", "previous-owned-id"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	u, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	result, _ := ex.reconcilePresenceStateWithProcessProof(context.Background(), st.manifest.Engine, st, true, port, true,
		func(int) (int, string, bool) { return 12345, "external", true }, func(int) bool { return false })
	if result.Err != nil || !result.Identified || !st.adopted || st.runID == "owned-run" || st.buildInfo != "" || st.buildID != "" {
		t.Fatalf("adoption attributed installed build to an external listener: result=%+v run=%q info=%q id=%q", result, st.runID, st.buildInfo, st.buildID)
	}
}
