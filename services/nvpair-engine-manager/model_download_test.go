// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/modeldownload"
)

func TestModelDownloadFreshProfileProvisionsPrivateJournalDirectory(t *testing.T) {
	base := filepath.Join(t.TempDir(), "fresh", "engine-bin")
	ex := NewExecutor(NewRegistry(), NewReporter(nil), nil, base)
	if ex.downloads.snapshot().RecoveryRequired {
		t.Fatal("fresh profile requires recovery")
	}
	if ex.downloads.copiesComplete || ex.downloads.copyRecoveryRequired {
		t.Fatal("missing historical copy baseline was declared clear")
	}
	info, err := os.Stat(base)
	if err != nil {
		t.Fatal(err)
	}
	if !info.IsDir() || runtime.GOOS != "windows" && info.Mode().Perm() != 0o700 {
		t.Fatalf("journal directory mode = %v", info.Mode())
	}
}

func TestModelDownloadTargetLifecycleAndCancelledQueue(t *testing.T) {
	started, finish := make(chan struct{}), make(chan struct{})
	var release sync.Once
	var effects atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/pull" {
			http.NotFound(w, r)
			return
		}
		effects.Add(1)
		close(started)
		<-finish
		_, _ = fmt.Fprintln(w, `{"status":"success"}`)
	}))
	defer func() { release.Do(func() { close(finish) }); server.Close() }()
	ex := newHTTPPullTestExecutor(t, server, Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/api/pull"}})
	first, err := ex.reserveModelPull(context.Background(), "fake", "first", json.RawMessage(`{"name":"first"}`))
	if err != nil {
		t.Fatal(err)
	}
	firstDone := make(chan error, 1)
	go func() { _, err := first.run(context.Background()); firstDone <- err }()
	<-started
	second, err := ex.reserveModelPull(context.Background(), "fake", "second", json.RawMessage(`{"name":"second"}`))
	if err != nil {
		t.Fatal(err)
	}
	snapshot := ex.downloads.snapshot()
	if len(snapshot.Entries) != 2 {
		t.Fatalf("snapshot = %+v", snapshot)
	}
	for _, entry := range snapshot.Entries {
		if entry.Model == "first" && (entry.ID != first.id || entry.Phase != "in_progress") {
			t.Fatalf("first = %+v", entry)
		}
		if entry.Model == "second" && (entry.ID != second.id || entry.Phase != "queued" || entry.Percent != nil) {
			t.Fatalf("second inherited progress: %+v", entry)
		}
	}
	if _, err := ex.reserveModelPull(context.Background(), "fake", "first", json.RawMessage(`{"name":"first"}`)); !errors.Is(err, errDownloadBusy) {
		t.Fatalf("duplicate = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	secondDone := make(chan error, 1)
	go func() { _, err := second.run(ctx); secondDone <- err }()
	cancel()
	if err := <-secondDone; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled waiter = %v", err)
	}
	if effects.Load() != 1 {
		t.Fatalf("cancelled waiter dispatched: %d", effects.Load())
	}
	release.Do(func() { close(finish) })
	if err := <-firstDone; err != nil {
		t.Fatal(err)
	}
	for _, entry := range ex.downloads.snapshot().Entries {
		if entry.BlocksReadiness {
			t.Fatalf("completed/pre-effect outcomes blocked: %+v", entry)
		}
		if entry.ID == first.id && entry.Phase != "complete" {
			t.Fatalf("terminal = %+v", entry)
		}
		if entry.ID == second.id && (entry.Phase != "failed" || entry.Reason != "interrupted") {
			t.Fatalf("cancel = %+v", entry)
		}
	}
	markers, err := ex.downloads.store.load()
	if err != nil || len(markers) != 0 {
		t.Fatalf("markers = %d/%v", len(markers), err)
	}
}

func TestModelDownloadNDJSONRequiresNativeTerminal(t *testing.T) {
	test := func(name, body, phase, reason string, blocked bool) {
		t.Run(name, func(t *testing.T) {
			var effects atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/api/pull" {
					http.NotFound(w, r)
					return
				}
				effects.Add(1)
				_, _ = fmt.Fprintln(w, body)
			}))
			defer server.Close()
			ex := newHTTPPullTestExecutor(t, server, Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/api/pull"}})
			_, err := ex.PullModelStream(context.Background(), "fake", "demo", json.RawMessage(`{"name":"demo"}`))
			if (err == nil) != (phase == "complete") {
				t.Fatalf("error = %v", err)
			}
			s := ex.downloads.snapshot()
			if err := s.Validate(); err != nil {
				t.Fatal(err)
			}
			if len(s.Entries) != 1 {
				t.Fatalf("entries = %+v", s)
			}
			entry := s.Entries[0]
			if entry.Phase != phase || entry.Reason != reason || entry.BlocksReadiness != blocked {
				t.Fatalf("outcome = %+v", entry)
			}
			recovered := newModelDownloads(newOpID(), ex.baseDir, map[string]string{"fake": engines.ModelNamingExact})
			if len(recovered.snapshot().Entries) != 0 && recovered.snapshot().Entries[0].Phase != "unknown" {
				t.Fatal("restart invented terminal outcome")
			}
			if got := len(recovered.unsafe); (got != 0) != blocked {
				t.Fatalf("recovered unsafe = %d", got)
			}
			if effects.Load() != 1 {
				t.Fatal("recovery caused an engine effect")
			}
		})
	}
	test("explicit success", `{"status":"success"}`, "complete", "", false)
	test("percent one hundred is not completion", `{"status":"pulling","total":100,"completed":100}`, "unknown", "interrupted", true)
	test("explicit native error", `{"error":"private vendor output"}`, "failed", "download_failed", true)
	test("opaque response", "accepted", "unknown", "unsupported", true)
}

func TestModelDownloadPersistenceFailureHasZeroNativeEffects(t *testing.T) {
	var effects atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/pull" {
			http.NotFound(w, r)
			return
		}
		effects.Add(1)
	}))
	defer server.Close()
	ex := newHTTPPullTestExecutor(t, server, Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/api/pull"}})
	ex.downloads.store.replace.Replace = func(string, string) error { return errors.New("injected") }
	if _, err := ex.PullModelStream(context.Background(), "fake", "demo", json.RawMessage(`{"name":"demo"}`)); err == nil {
		t.Fatal("save failure accepted")
	}
	if effects.Load() != 0 {
		t.Fatal("native request preceded durable marker")
	}
	entry := ex.downloads.snapshot().Entries[0]
	if entry.Phase != "failed" || entry.BlocksReadiness || entry.Reason != "persistence_error" {
		t.Fatalf("outcome = %+v", entry)
	}
}

func TestModelDownloadDeleteRequiresForcedUnmaskedAbsence(t *testing.T) {
	var present atomic.Bool
	present.Store(true)
	var forced, ordinary, deletes atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !(r.Method == http.MethodPost && r.URL.Path == "/delete") && !(r.Method == http.MethodGet && r.URL.Path == "/models") {
			http.NotFound(w, r)
			return
		}
		switch r.URL.Path {
		case "/delete":
			deletes.Add(1)
			_, _ = fmt.Fprintln(w, `{"success":true}`)
		case "/models":
			if r.URL.Query().Get("reload") == "1" {
				forced.Add(1)
				if present.Load() {
					_, _ = fmt.Fprintln(w, `{"data":[{"id":"demo","status":{"value":"downloaded"}}]}`)
					return
				}
			} else {
				ordinary.Add(1)
			}
			_, _ = fmt.Fprintln(w, `{"data":[]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	ex := newHTTPPullTestExecutor(t, server, Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/models"}, ProgressProtocol: pullProgressProtocolLlamaCPPModelsSSE})
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.runID = "delete-run"
	st.manifest.Actions["delete_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/delete"}}
	st.manifest.Actions["list_models"] = Action{HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/models"}, Result: &ActionResult{Array: "data", Field: "id"}}
	a := reserveModelDownload(t, ex.downloads, "fake", "demo", engines.ModelNamingExact)
	if _, err := ex.downloads.changeMarker(a, false); err != nil {
		t.Fatal(err)
	}
	if err := ex.downloads.finish(a, "failed", "download_failed", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := ex.ModelDelete(context.Background(), "fake", "demo"); err != nil {
		t.Fatal(err)
	}
	if !ex.downloads.blocked(a.key) || forced.Load() != 1 || ordinary.Load() != 0 {
		t.Fatalf("ordinary absence cleared shards: blocked=%v forced=%d ordinary=%d", ex.downloads.blocked(a.key), forced.Load(), ordinary.Load())
	}
	present.Store(false)
	ex.downloads.store.replace.SyncDir = func(string) error { return errors.New("injected") }
	if _, err := ex.ModelDelete(context.Background(), "fake", "demo"); err != nil {
		t.Fatal(err)
	}
	entry := ex.downloads.snapshot().Entries[0]
	if ex.downloads.blocked(a.key) || entry.BlocksReadiness || entry.Persistence != "uncertain" || deletes.Load() != 2 || forced.Load() != 2 {
		t.Fatalf("proven deletion = %+v; deletes=%d forced=%d", entry, deletes.Load(), forced.Load())
	}
}

func TestModelDownloadDeleteRejectsAmbiguousOrRedirectedAbsence(t *testing.T) {
	for _, tc := range []struct {
		name     string
		body     string
		redirect bool
	}{
		{name: "redirect to ordinary inventory", redirect: true},
		{name: "null identity", body: `{"data":[{"id":"other"},{"id":null}]}`},
		{name: "missing identity", body: `{"data":[{"id":"other"},{}]}`},
		{name: "invalid identity", body: `{"data":[{"id":"other"},{"id":""}]}`},
		{name: "invalid UTF8 identity", body: "{\"data\":[{\"id\":\"other\"},{\"id\":\"\xff\"}]}"},
		{name: "null row", body: `{"data":[{"id":"other"},null]}`},
		{name: "null inventory", body: `{"data":null}`},
		{name: "duplicate identity", body: `{"data":[{"id":"demo","id":"other"}]}`},
		{name: "case variant identity", body: `{"data":[{"ID":"demo","id":"other"}]}`},
		{name: "duplicate inventory", body: `{"data":[{"id":"demo"}],"data":[]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var ordinary atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if !(r.Method == http.MethodPost && r.URL.Path == "/delete") && !(r.Method == http.MethodGet && r.URL.Path == "/models") {
					http.NotFound(w, r)
					return
				}
				if r.URL.Path == "/delete" {
					_, _ = fmt.Fprintln(w, `{"success":true}`)
					return
				}
				if r.URL.Query().Get("reload") != "1" {
					ordinary.Add(1)
					_, _ = fmt.Fprintln(w, `{"data":[]}`)
					return
				}
				if tc.redirect {
					http.Redirect(w, r, "/models", http.StatusFound)
					return
				}
				_, _ = fmt.Fprintln(w, tc.body)
			}))
			defer server.Close()
			ex := newHTTPPullTestExecutor(t, server, Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/models"}, ProgressProtocol: pullProgressProtocolLlamaCPPModelsSSE})
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			st.runID = "delete-run"
			st.manifest.Actions["delete_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/delete"}}
			st.manifest.Actions["list_models"] = Action{HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/models"}, Result: &ActionResult{Array: "data", Field: "id"}}
			a := reserveModelDownload(t, ex.downloads, "fake", "demo", engines.ModelNamingExact)
			if _, err := ex.downloads.changeMarker(a, false); err != nil {
				t.Fatal(err)
			}
			if err := ex.downloads.finish(a, "failed", "download_failed", ""); err != nil {
				t.Fatal(err)
			}
			before := ex.downloads.snapshot()
			if _, err := ex.ModelDelete(context.Background(), "fake", "demo"); err != nil {
				t.Fatal(err)
			}
			if !before.Equal(ex.downloads.snapshot()) || !ex.downloads.blocked(a.key) || ordinary.Load() != 0 {
				t.Fatalf("ambiguous absence changed state: %+v; ordinary reads=%d", ex.downloads.snapshot(), ordinary.Load())
			}
		})
	}
}

func TestModelDownloadDeleteResolutionReservesRevisionBeforePersistence(t *testing.T) {
	prepare := func(t *testing.T) (*modelDownloads, *modelDownloadAttempt) {
		t.Helper()
		d := testModelDownloads(t)
		a := reserveModelDownload(t, d, "llamacpp", "model", engines.ModelNamingExact)
		if _, err := d.changeMarker(a, false); err != nil {
			t.Fatal(err)
		}
		if err := d.finish(a, "failed", "download_failed", ""); err != nil {
			t.Fatal(err)
		}
		return d, a
	}
	t.Run("exhaustion preserves disk and readiness", func(t *testing.T) {
		d, a := prepare(t)
		d.revision = math.MaxUint64
		before := d.snapshot()
		diskBefore, err := os.ReadFile(d.store.path)
		if err != nil {
			t.Fatal(err)
		}
		if d.resolveMarker(a.key, a.entry.ID) {
			t.Fatal("exhausted resolution accepted")
		}
		diskAfter, err := os.ReadFile(d.store.path)
		if err != nil {
			t.Fatal(err)
		}
		if !before.Equal(d.snapshot()) || !bytes.Equal(diskBefore, diskAfter) || !d.blocked(a.key) {
			t.Fatal("exhausted resolution changed disk or state")
		}
	})
	t.Run("progress preserves deletion and active terminal capacity", func(t *testing.T) {
		d, a := prepare(t)
		other := reserveModelDownload(t, d, "ollama", "other", engines.ModelNamingImpliedLatestTag)
		if err := d.update(other, "in_progress", nil); err != nil {
			t.Fatal(err)
		}
		d.revision = math.MaxUint64 - 2
		entered, release := make(chan struct{}), make(chan struct{})
		var once sync.Once
		defer once.Do(func() { close(release) })
		d.store.replace.SyncDir = func(string) error { close(entered); <-release; return nil }
		done := make(chan bool, 1)
		go func() { done <- d.resolveMarker(a.key, a.entry.ID) }()
		<-entered
		percent := 10
		if err := d.update(other, "in_progress", &percent); !errors.Is(err, errDownloadCapacity) {
			t.Fatalf("progress consumed reserved revision: %v", err)
		}
		if err := d.finish(other, "failed", "interrupted", ""); err != nil {
			t.Fatal(err)
		}
		once.Do(func() { close(release) })
		if !<-done || d.blocked(a.key) || d.snapshot().Revision != "18446744073709551615" {
			t.Fatalf("resolution = %+v", d.snapshot())
		}
	})
}

func TestModelDownloadTerminalSnapshotSurvivesSameKeyRetry(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/pull" {
			http.NotFound(w, r)
			return
		}
		_, _ = fmt.Fprintln(w, `{"status":"success"}`)
	}))
	defer server.Close()
	ex := newHTTPPullTestExecutor(t, server, Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/api/pull"}})
	first, err := ex.reserveModelPull(context.Background(), "fake", "demo", json.RawMessage(`{"name":"demo"}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.run(context.Background()); err != nil {
		t.Fatal(err)
	}
	next, err := ex.reserveModelPull(context.Background(), "fake", "demo", json.RawMessage(`{"name":"demo"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer next.cancelUnstarted()
	terminal := first.snapshot()
	if len(terminal.Entries) != 1 || terminal.Entries[0].ID != first.id || terminal.Entries[0].Phase != "complete" {
		t.Fatalf("old stream lost terminal identity: %+v", terminal)
	}
	current := ex.downloads.snapshot()
	if current.Entries[0].ID != next.id || current.Entries[0].Phase != "queued" {
		t.Fatalf("new operation overwritten: %+v", current)
	}
}

func TestModelDownloadBlockedLoadAndExportDispatch(t *testing.T) {
	root := t.TempDir()
	writeOllamaModel(t, root, "demo", 64, 1)
	ex := copyTestExecutor(t, "ollama", root, true)
	st, err := ex.state("ollama")
	if err != nil {
		t.Fatal(err)
	}
	st.manifest.ModelNaming = engines.ModelNamingImpliedLatestTag
	st.manifest.Actions = map[string]Action{"load_model": {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/load"}}}
	var effects atomic.Int32
	ex.client = &http.Client{Transport: &mutationTransport{respond: func(*http.Request) (*http.Response, error) { effects.Add(1); return mutationJSON(http.StatusOK, `{}`) }}}
	a := reserveModelDownload(t, ex.downloads, "ollama", "demo:latest", engines.ModelNamingImpliedLatestTag)
	if _, err := ex.downloads.changeMarker(a, false); err != nil {
		t.Fatal(err)
	}
	if err := ex.downloads.finish(a, "failed", "download_failed", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := ex.Action(context.Background(), "ollama", "load_model", json.RawMessage(`{"model":"demo"}`)); err == nil {
		t.Fatal("raw load dispatched unsafe alias")
	}
	if _, err := ex.ModelLoad(context.Background(), "ollama", "demo"); err == nil {
		t.Fatal("typed load dispatched unsafe alias")
	}
	st.opMu.Lock()
	_, release, loadErr := ex.guardServingLoadLocked(context.Background(), st, "load_model", json.RawMessage(`{"model":"demo"}`))
	if loadErr == nil {
		release()
		st.opMu.Unlock()
		t.Fatal("automatic load admitted unsafe alias")
	}
	st.opMu.Unlock()
	s := &controlServer{exec: ex}
	for _, route := range []string{controlModelFilesPath, controlModelFilePath} {
		r := httptest.NewRequest(http.MethodGet, route+"?engine=ollama&model=demo&path=blobs/unused", nil)
		w := httptest.NewRecorder()
		if route == controlModelFilesPath {
			s.handleModelFiles(w, r)
		} else {
			s.handleModelFile(w, r)
		}
		if w.Code != http.StatusConflict {
			t.Fatalf("export %s = %d", route, w.Code)
		}
	}
	if effects.Load() != 0 {
		t.Fatalf("blocked effects = %d", effects.Load())
	}
}

func testModelDownloads(t *testing.T) *modelDownloads {
	t.Helper()
	return newModelDownloads(newOpID(), t.TempDir(), map[string]string{"ollama": engines.ModelNamingImpliedLatestTag, "llamacpp": engines.ModelNamingExact})
}

func reserveModelDownload(t *testing.T, d *modelDownloads, engine, model, naming string) *modelDownloadAttempt {
	t.Helper()
	a, err := d.reserve(engine, model, naming)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestModelDownloadNamingEquivalence(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "ollama", "demo", engines.ModelNamingImpliedLatestTag)
	if a.entry.Model != "demo" || a.key.model != "demo:latest" {
		t.Fatalf("raw spelling/key = %q/%q", a.entry.Model, a.key.model)
	}
	if _, err := d.reserve("ollama", "demo:latest", engines.ModelNamingImpliedLatestTag); !errors.Is(err, errDownloadBusy) {
		t.Fatalf("duplicate = %v", err)
	}
	if !d.blocked(downloadContentKey{"ollama", "demo:latest"}) {
		t.Fatal("equivalent key was not fenced")
	}
	if _, err := d.changeMarker(a, false); err != nil {
		t.Fatal(err)
	}
	if err := d.finish(a, "failed", "download_failed", ""); err != nil {
		t.Fatal(err)
	}
	b := reserveModelDownload(t, d, "ollama", "demo:latest", engines.ModelNamingImpliedLatestTag)
	if err := d.finish(b, "failed", "interrupted", ""); err != nil {
		t.Fatal(err)
	}
	s := d.snapshot()
	if len(s.Entries) != 1 || s.Entries[0].ID != b.entry.ID || !s.Entries[0].BlocksReadiness {
		t.Fatalf("retry lost earlier unsafe marker: %+v", s)
	}
}

func TestModelDownloadFixedCapacity(t *testing.T) {
	d := testModelDownloads(t)
	for i := 0; i < modeldownload.MaxUnsafeEntries; i++ {
		reserveModelDownload(t, d, "llamacpp", fmt.Sprintf("m%d", i), engines.ModelNamingExact)
	}
	if _, err := d.reserve("llamacpp", "overflow", engines.ModelNamingExact); !errors.Is(err, errDownloadCapacity) {
		t.Fatalf("overflow = %v", err)
	}
	if got := len(d.snapshot().Entries); got != modeldownload.MaxUnsafeEntries {
		t.Fatalf("entries = %d", got)
	}
}

func TestModelDownloadTerminalRetention(t *testing.T) {
	d := testModelDownloads(t)
	for i := 0; i < 100; i++ {
		a := reserveModelDownload(t, d, "llamacpp", fmt.Sprintf("m%d", i), engines.ModelNamingExact)
		if err := d.finish(a, "failed", "interrupted", ""); err != nil {
			t.Fatal(err)
		}
	}
	s := d.snapshot()
	if len(s.Entries) != modeldownload.MaxRecentEntries || len(d.recent) != modeldownload.MaxRecentEntries {
		t.Fatalf("unbounded outcomes: %d/%d", len(s.Entries), len(d.recent))
	}
	for _, e := range s.Entries {
		if e.BlocksReadiness {
			t.Fatal("pre-effect failure invalidated content")
		}
	}
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestModelDownloadGateOrderings(t *testing.T) {
	d := testModelDownloads(t)
	key := downloadContentKey{"llamacpp", "model"}
	reader, err := d.acquireRead(key, false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.acquireWrite("llamacpp"); !errors.Is(err, errDownloadBusy) {
		t.Fatalf("writer behind reader = %v", err)
	}
	other, err := d.acquireWrite("ollama")
	if err != nil {
		t.Fatal(err)
	}
	other()
	reader()
	reader()
	writer, err := d.acquireWrite("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.acquireRead(key, false); !errors.Is(err, errDownloadBusy) {
		t.Fatalf("reader behind writer = %v", err)
	}
	writer()
	writer()
	reader, err = d.acquireRead(key, false)
	if err != nil {
		t.Fatal(err)
	}
	reader()
}

func TestModelDownloadProgressPreservesQueuedTransitionAndTerminals(t *testing.T) {
	d := testModelDownloads(t)
	first := reserveModelDownload(t, d, "llamacpp", "first", engines.ModelNamingExact)
	queued := reserveModelDownload(t, d, "llamacpp", "queued", engines.ModelNamingExact)
	if err := d.update(first, "in_progress", nil); err != nil {
		t.Fatal(err)
	}
	d.revision = math.MaxUint64 - 3
	percent := 10
	if err := d.update(first, "in_progress", &percent); !errors.Is(err, errDownloadCapacity) {
		t.Fatalf("progress consumed the queued transition: %v", err)
	}
	if _, err := d.changeMarker(queued, false); err != nil {
		t.Fatal(err)
	}
	if err := d.update(queued, "in_progress", nil); err != nil {
		t.Fatalf("accepted queued request lost its transition after persistence: %v", err)
	}
	if err := d.finish(first, "failed", "interrupted", ""); err != nil {
		t.Fatal(err)
	}
	if err := d.finish(queued, "failed", "download_failed", ""); err != nil {
		t.Fatal(err)
	}
	if d.snapshot().Revision != "18446744073709551615" {
		t.Fatalf("terminal revisions = %+v", d.snapshot())
	}
}

func TestModelDownloadRevisionCoalescingAndExhaustion(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "llamacpp", "model", engines.ModelNamingExact)
	p := 10
	if err := d.update(a, "in_progress", &p); err != nil {
		t.Fatal(err)
	}
	before := d.snapshot()
	if err := d.update(a, "in_progress", &p); err != nil {
		t.Fatal(err)
	}
	if !before.Equal(d.snapshot()) {
		t.Fatal("unchanged percent advanced revision")
	}
	d.mu.Lock()
	d.revision = math.MaxUint64
	d.mu.Unlock()
	if _, err := d.reserve("llamacpp", "other", engines.ModelNamingExact); !errors.Is(err, errDownloadCapacity) {
		t.Fatalf("exhausted revision = %v", err)
	}
	if d.snapshot().Revision != "18446744073709551615" {
		t.Fatal("revision wrapped")
	}
}
