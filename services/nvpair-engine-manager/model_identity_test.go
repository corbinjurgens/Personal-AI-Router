// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// digestManifest is the fake engine with an Ollama-shaped list_models that
// declares the content digest field, as the bundled Ollama manifest does.
func digestManifest() *Manifest {
	m := testEngineManifest(fakeEngineBin)
	m.Actions["list_models"] = Action{
		HTTP:   &ActionHTTP{Method: "GET", Path: "/api/tags"},
		Result: &ActionResult{Array: "models", Field: "name", Digest: "digest"},
	}
	return m
}

func nameDigest(name string) string {
	sum := sha256.Sum256([]byte(name))
	return hex.EncodeToString(sum[:])
}

func TestModelIdentityFollowsTheOwnedRun(t *testing.T) {
	ex := newTestExecutor(t, digestManifest())
	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })

	if got := ex.modelIdentity(); len(got.Engines) != 0 {
		t.Fatalf("identity of a stopped engine = %+v, want none", got)
	}
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	ex.modelsSweep(ctx, true)
	first := ex.modelIdentity().Engines["fake"]
	if !strings.HasPrefix(first.EngineRunID, "run-") || !contentDigest.MatchString(first.LaunchDigest) {
		t.Fatalf("identity after start = %+v, want a run id and a launch digest", first)
	}
	if want := map[string]string{"llama3.2:1b": nameDigest("llama3.2:1b")}; !reflect.DeepEqual(first.ModelDigests, want) {
		t.Fatalf("model digests = %v, want %v", first.ModelDigests, want)
	}

	if err := ex.Stop("fake"); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if got := ex.modelIdentity(); len(got.Engines) != 0 {
		t.Fatalf("identity after stop = %+v, want none", got)
	}
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("restart: %v", err)
	}
	second := ex.modelIdentity().Engines["fake"]
	if second.EngineRunID == "" || second.EngineRunID == first.EngineRunID {
		t.Fatalf("run id after a restart = %q, want a new one (was %q)", second.EngineRunID, first.EngineRunID)
	}
	// The last listing belongs to the previous run, so none is reported until
	// the new run is listed.
	if len(second.ModelDigests) != 0 {
		t.Fatalf("digests before the new run is listed = %v, want none", second.ModelDigests)
	}
	ex.modelsSweep(ctx, true)
	if got := ex.modelIdentity().Engines["fake"].ModelDigests; len(got) != 1 {
		t.Fatalf("digests after listing the new run = %v, want the model", got)
	}
}

func TestModelIdentityReportsSameNameReplacement(t *testing.T) {
	ex := newTestExecutor(t, digestManifest())
	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	// A residency-only sweep still lists an inventory that carries digests.
	ex.modelsSweep(ctx, true)
	replaced := strings.Repeat("c", 64)
	st, _ := ex.Status("fake")
	body, _ := json.Marshal(map[string]string{"name": "llama3.2:1b", "digest": replaced})
	resp, err := http.Post(fmt.Sprintf("http://127.0.0.1:%d/testctl/digest", st.Port), "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("replace model: %v", err)
	}
	_ = resp.Body.Close()
	ex.modelsSweep(ctx, false)
	if got := ex.modelIdentity().Engines["fake"].ModelDigests["llama3.2:1b"]; got != replaced {
		t.Fatalf("digest after a same-name replacement = %q, want %q", got, replaced)
	}
}

func TestModelIdentityExcludesAdoptedRuns(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			_ = json.NewEncoder(w).Encode(map[string]any{"models": []map[string]string{{"name": "m:latest", "digest": nameDigest("m")}}})
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	u, _ := url.Parse(srv.URL)
	port, _ := strconv.Atoi(u.Port())
	key := runtime.GOOS + "/" + runtime.GOARCH
	m := &Manifest{
		Engine: "ext", DisplayName: "External", ManifestVersion: 1,
		Platforms: map[string]Platform{key: {
			Detect: []string{filepath.Join(t.TempDir(), "missing"+exeExt())},
			Runtime: Runtime{
				Bin:   filepath.Join(t.TempDir(), "does-not-exist"+exeExt()),
				Port:  port,
				Ready: &Probe{HTTP: "http://127.0.0.1:{port}/", Status: 200, TimeoutS: 5},
			},
		}},
		Actions: map[string]Action{"list_models": {
			HTTP:   &ActionHTTP{Method: "GET", Path: "/api/tags"},
			Result: &ActionResult{Array: "models", Field: "name", Digest: "digest"},
		}},
	}
	ex := newTestExecutor(t, m)
	if st, _ := ex.Status("ext"); !st.Running {
		t.Fatalf("external engine not adopted: %+v", st)
	}
	if ex.engineRunID("ext") == "" {
		t.Fatal("an adopted run has no run id")
	}
	ex.modelsSweep(context.Background(), true)
	// The manager did not launch it, so it cannot name its launch.
	if got := ex.modelIdentity(); len(got.Engines) != 0 {
		t.Fatalf("identity of an adopted run = %+v, want none", got)
	}
}

func TestModelIdentityPairsDigestsWithTheirRun(t *testing.T) {
	ex := newTestExecutor(t, digestManifest())
	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	run := ex.engineRunID("fake")
	listed := map[string]string{"llama3.2:1b": nameDigest("llama3.2:1b")}

	ex.storeInventory(nil, map[string]observedDigests{"fake": {runID: "run-earlier", models: listed}}, ex.sweeps.Add(1))
	if got := ex.modelIdentity().Engines["fake"].ModelDigests; len(got) != 0 {
		t.Fatalf("digests read under another run = %v, want none", got)
	}

	newer := ex.sweeps.Add(1)
	older := newer - 1
	ex.storeInventory(nil, map[string]observedDigests{"fake": {runID: run, models: listed}}, newer)
	ex.storeInventory(nil, map[string]observedDigests{"fake": {runID: run, models: map[string]string{"llama3.2:1b": strings.Repeat("d", 64)}}}, older)
	if got := ex.modelIdentity().Engines["fake"].ModelDigests; !reflect.DeepEqual(got, listed) {
		t.Fatalf("digests after an older sweep stored late = %v, want the newer sweep's %v", got, listed)
	}
}

// A run that changes while its inventory request is in flight does not
// inherit that listing's digests.
func TestModelIdentityIgnoresAListingAcrossARunChange(t *testing.T) {
	ex := newTestExecutor(t, digestManifest())
	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	st, _ := ex.Status("fake")
	resp, err := http.Post(fmt.Sprintf("http://127.0.0.1:%d/testctl/tags-delay", st.Port), "application/json", strings.NewReader(`{"ms":800}`))
	if err != nil {
		t.Fatalf("delay the inventory: %v", err)
	}
	_ = resp.Body.Close()
	done := make(chan struct{})
	go func() { ex.modelsSweep(ctx, true); close(done) }()
	time.Sleep(300 * time.Millisecond)
	// The manager observes another incarnation while the listing is pending.
	state, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	state.mu.Lock()
	state.runID = newEngineRunID()
	state.mu.Unlock()
	<-done
	if got := ex.modelIdentity().Engines["fake"].ModelDigests; len(got) != 0 {
		t.Fatalf("the new run inherited digests %v listed for the previous one", got)
	}
}

func TestExtractDigests(t *testing.T) {
	spec := &ActionResult{Array: "models", Field: "name", Digest: "digest"}
	good := strings.Repeat("a", 64)
	for _, tc := range []struct {
		name string
		raw  string
		spec *ActionResult
		want map[string]string
	}{
		{"valid", `{"models":[{"name":"m","digest":"` + good + `"}]}`, spec, map[string]string{"m": good}},
		{"no digest field declared", `{"models":[{"name":"m","digest":"` + good + `"}]}`, &ActionResult{Array: "models", Field: "name"}, nil},
		{"uppercase digest", `{"models":[{"name":"m","digest":"` + strings.ToUpper(good) + `"}]}`, spec, map[string]string{}},
		{"prefixed digest", `{"models":[{"name":"m","digest":"sha256:` + good + `"}]}`, spec, map[string]string{}},
		{"missing digest", `{"models":[{"name":"m"}]}`, spec, map[string]string{}},
		{"conflicting duplicate", `{"models":[{"name":"m","digest":"` + good + `"},{"name":"m","digest":"` + strings.Repeat("b", 64) + `"}]}`, spec, map[string]string{}},
		{"not a listing", `{"models":"x"}`, spec, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := extractDigests(json.RawMessage(tc.raw), tc.spec); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("extractDigests = %#v, want %#v", got, tc.want)
			}
		})
	}
}

// notifications records what an executor emits.
type notifications struct {
	mu   sync.Mutex
	sent []string
	last map[string]any
}

func (n *notifications) emit(method string, params any) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.sent = append(n.sent, method)
	if n.last == nil {
		n.last = map[string]any{}
	}
	n.last[method] = params
}

func (n *notifications) count(method string) int {
	n.mu.Lock()
	defer n.mu.Unlock()
	c := 0
	for _, m := range n.sent {
		if m == method {
			c++
		}
	}
	return c
}

func (n *notifications) identity() modelIdentityParams {
	n.mu.Lock()
	defer n.mu.Unlock()
	id, _ := n.last[modelIdentityMethod].(modelIdentityParams)
	return id
}

func TestPublishModelIdentityEmitsFirstThenOnChange(t *testing.T) {
	var sent notifications
	reg := NewRegistry()
	reg.engines["fake"] = digestManifest()
	ex := NewExecutor(reg, NewReporter(nil), sent.emit, t.TempDir())
	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })

	last := ex.publishModelIdentity(nil)
	if sent.count(modelIdentityMethod) != 1 || len(sent.identity().Engines) != 0 {
		t.Fatalf("first publication = %d sends, %+v; want one empty snapshot", sent.count(modelIdentityMethod), sent.identity())
	}
	last = ex.publishModelIdentity(last)
	if sent.count(modelIdentityMethod) != 1 {
		t.Fatalf("an unchanged snapshot was sent again")
	}
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	ex.publishModelIdentity(last)
	if sent.count(modelIdentityMethod) != 2 || sent.identity().Engines["fake"].EngineRunID == "" {
		t.Fatalf("after a start = %d sends, %+v; want the new run", sent.count(modelIdentityMethod), sent.identity())
	}
}

func TestWatcherReportsTheIdentityOfAStartedEngine(t *testing.T) {
	var sent notifications
	reg := NewRegistry()
	reg.engines["fake"] = digestManifest()
	ex := NewExecutor(reg, NewReporter(nil), sent.emit, t.TempDir())
	// A long tick: only the poke from the state change can drive the report.
	ex.loadedPollInterval = time.Hour
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() { cancel(); _ = ex.Stop("fake") })
	go ex.watchLoaded(ctx)
	ex.pokeLoaded()
	waitFor(t, 10*time.Second, func() bool { return sent.count(modelIdentityMethod) >= 1 })

	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	waitFor(t, 10*time.Second, func() bool {
		id := sent.identity().Engines["fake"]
		return id.EngineRunID != "" && len(id.ModelDigests) == 1
	})
}

// An identity read while the engine runs is sent before that engine's stop,
// never after it, so the broker cannot reinstall a dead run.
func TestModelIdentityIsNeverSentAfterTheStopItPrecedes(t *testing.T) {
	var mu sync.Mutex
	var sent []string
	hold := make(chan struct{})
	holding := false
	emit := func(method string, params any) {
		mu.Lock()
		block := holding && method == modelIdentityMethod
		if method == "engine:state-changed" {
			if st, ok := params.(EngineStatus); ok && !st.Running {
				method += ":stopped"
			}
		}
		sent = append(sent, method)
		mu.Unlock()
		if block {
			<-hold
		}
	}
	reg := NewRegistry()
	reg.engines["fake"] = digestManifest()
	ex := NewExecutor(reg, NewReporter(nil), emit, t.TempDir())
	t.Cleanup(func() { _ = ex.Stop("fake") })
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	mu.Lock()
	holding = true
	mu.Unlock()
	published := make(chan struct{})
	go func() { ex.publishModelIdentity(nil); close(published) }()
	waitFor(t, 10*time.Second, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(sent) > 0 && sent[len(sent)-1] == modelIdentityMethod
	})
	stopped := make(chan struct{})
	go func() { _ = ex.Stop("fake"); close(stopped) }()
	select {
	case <-stopped:
	case <-time.After(2 * time.Second):
	}
	mu.Lock()
	for _, m := range sent {
		if m == "engine:state-changed:stopped" {
			mu.Unlock()
			t.Fatal("the stop was sent while an identity read before it was still being sent")
		}
	}
	holding = false
	mu.Unlock()
	close(hold)
	<-published
	<-stopped
	mu.Lock()
	defer mu.Unlock()
	if !slices.Contains(sent, "engine:state-changed:stopped") {
		t.Fatalf("sent %v, want the stop once the identity was sent", sent)
	}
}

func TestResultDigestIsListModelsOnly(t *testing.T) {
	list := Action{HTTP: &ActionHTTP{Method: "GET", Path: "/api/tags"}, Result: &ActionResult{Array: "models", Field: "name", Digest: "digest"}}
	if err := list.validate("list_models"); err != nil {
		t.Fatalf("list_models with a digest field rejected: %v", err)
	}
	loaded := Action{HTTP: &ActionHTTP{Method: "GET", Path: "/api/ps"}, Result: &ActionResult{Array: "models", Field: "name", Digest: "digest"}}
	if err := loaded.validate("loaded_models"); err == nil || !strings.Contains(err.Error(), "result.digest") {
		t.Fatalf("loaded_models with a digest field = %v, want rejected", err)
	}
	padded := Action{HTTP: &ActionHTTP{Method: "GET", Path: "/api/tags"}, Result: &ActionResult{Array: "models", Field: "name", Digest: " digest"}}
	if err := padded.validate("list_models"); err == nil {
		t.Fatal("a padded digest field name was accepted")
	}
}

func TestOllamaManifestDeclaresContentDigests(t *testing.T) {
	reg := NewRegistry()
	if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
		t.Fatal(err)
	}
	m, ok := reg.Get("ollama")
	if !ok {
		t.Fatal("ollama manifest not loaded")
	}
	if r := m.Actions["list_models"].Result; r == nil || r.Digest != "digest" {
		t.Fatalf("ollama list_models result = %+v, want the digest field", r)
	}
	for _, other := range []string{"lmstudio", "llamacpp"} {
		if om, ok := reg.Get(other); ok {
			if r := om.Actions["list_models"].Result; r != nil && r.Digest != "" {
				t.Errorf("%s declares a content digest %q it does not serve", other, r.Digest)
			}
		}
	}
}
