// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"os"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

// Synthetic owned launch and HTTP readback only; no native engine or model.
func residentNativeFixture(t *testing.T) (llamaServingFixture, *loadedLlamaFake, nodepolicy.ReconcileResidentParams) {
	t.Helper()
	f, server := residentOwnedFixture(t, "resident-model")
	f.ex.nodeProjection.set(nodeProjection{})
	if err := f.ex.desired.set("fake", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	id := f.ex.modelIdentity().Engines["fake"]
	f.ex.storeInventory(map[string][]string{"fake": {"resident-model", "weak"}}, nil, f.ex.sweeps.Add(1))
	p := nodepolicy.ReconcileResidentParams{OperationID: "resident:1", PolicyRevision: "1", DeclaredResidentCount: 1, Engine: "fake", Model: "resident-model",
		Identity:  nodepolicy.StrictModelIdentity{EngineRunID: id.EngineRunID, ModelDigest: id.ModelDigests["resident-model"], SettingsFingerprint: nameDigest("synthetic-settings")},
		TimeoutMs: 60000, Reservation: nodepolicy.ResidentReservationEvidence{ReservationID: "reservation", ProxyRunID: "proxy", ManagerIncarnation: f.ex.switchIncarnation},
		Load: nodepolicy.ResidentLoadOptions{ContextLength: 8192, KeepAliveSeconds: 60}}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	return f, server, p
}

func residentOwnedFixture(t *testing.T, model string) (llamaServingFixture, *loadedLlamaFake) {
	t.Helper()
	return residentOwnedRetentionFixture(t, model, "-1")
}

func residentOwnedRetentionFixture(t *testing.T, model, retention string) (llamaServingFixture, *loadedLlamaFake) {
	t.Helper()
	f := newLlamaServingFixture(t)
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Replace(string(data), "[medium]", "["+model+"]", 1)
	text = strings.Replace(text, "sleep-idle-seconds = 180", "sleep-idle-seconds = "+retention, 1)
	writeServingFixture(t, f.source, []byte(text))
	f.request.Settings.Serving.Models[0].ModelID = model
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.ex.Stop("fake"); err != nil {
			t.Error(err)
		}
	})
	server := attachLoadedLlamaFake(t, f, model)
	reg := loadWithOverrides(t, t.TempDir())
	m, ok := reg.Get("llamacpp")
	if !ok {
		t.Fatal("missing llama manifest")
	}
	f.ex.reg.engines["fake"].Operations = m.Operations
	return f, server
}

func TestResidentAttestedPreflight(t *testing.T) {
	f, server, p := residentNativeFixture(t)
	if got := f.ex.residentCheck(context.Background(), p); got != "" {
		t.Fatalf("current native launch preflight = %q", got)
	}
	st := settingsState(t, f.ex)
	st.mu.Lock()
	st.plat.Runtime.servingBaseDigest = nameDigest("replaced-executable")
	st.mu.Unlock()
	if got := f.ex.residentCheck(context.Background(), p); got != "unqualified" {
		t.Fatalf("stale owned launch preflight = %q, want unqualified", got)
	}
	grants := 0
	f.ex.residentParent = func(context.Context, nodepolicy.DispatchResidentParams) error { grants++; return nil }
	result, err := f.ex.reconcileResident(context.Background(), p)
	if err != nil || result.Outcome != nodepolicy.ResidentNotDispatched || grants != 0 || server.models.Load() != 0 || server.gets.Load() != 0 || server.loads.Load() != 0 {
		t.Fatalf("stale preflight effects: result=%+v err=%v grants=%d models=%d props=%d loads=%d", result, err, grants, server.models.Load(), server.gets.Load(), server.loads.Load())
	}
}

func TestResidentDailyAdapterFinalGrantRevalidatesNativeLaunch(t *testing.T) {
	f := newResidentFixture(t)
	f.grant = func(req residentDispatchRequest) {
		st := f.ex.engines[f.p.Engine]
		st.mu.Lock()
		// Ordinary run/content identity remains current. Only the cold validator
		// detects the changed executable binding at this grant boundary.
		st.plat.Runtime.servingBaseDigest = nameDigest("changed-after-grant")
		st.mu.Unlock()
		if !f.ex.residentIdentity(f.p) {
			t.Fatal("test invalidated the ordinary identity fence")
		}
		f.reply(t, req.ID, nodepolicy.DispatchResidentResult{Dispatched: true})
	}
	result := f.run(t)
	if result.Outcome != nodepolicy.ResidentNotDispatched || f.dispatches != 1 {
		t.Fatalf("post-grant drift result=%+v grants=%d loads=%d reads=%d", result, f.dispatches, f.transport.loads, f.transport.reads)
	}
	f.noCalls(t)
}

func TestResidentDailyAdapterNativeRequirements(t *testing.T) {
	test := func(name, retention string, count int, want string) {
		t.Run(name, func(t *testing.T) {
			f, server := residentOwnedRetentionFixture(t, "resident-model", retention)
			f.ex.nodeProjection.set(nodeProjection{})
			if err := f.ex.desired.set("fake", true, intentLocal); err != nil {
				t.Fatal(err)
			}
			f.ex.storeInventory(map[string][]string{"fake": {"resident-model"}}, nil, f.ex.sweeps.Add(1))
			id := f.ex.modelIdentity().Engines["fake"]
			p := nodepolicy.ReconcileResidentParams{OperationID: "native:1", PolicyRevision: "1", DeclaredResidentCount: count, Engine: "fake", Model: "resident-model",
				Identity:  nodepolicy.StrictModelIdentity{EngineRunID: id.EngineRunID, ModelDigest: id.ModelDigests["resident-model"], SettingsFingerprint: nameDigest("settings")},
				TimeoutMs: 60000, Reservation: nodepolicy.ResidentReservationEvidence{ReservationID: "r", ProxyRunID: "p", ManagerIncarnation: f.ex.switchIncarnation}, Load: nodepolicy.ResidentLoadOptions{ContextLength: 8192, KeepAliveSeconds: 60}}
			if got := f.ex.residentCheck(context.Background(), p); got != want {
				t.Fatalf("preflight=%q, want %q", got, want)
			}
			if want != "" {
				grants := 0
				f.ex.residentParent = func(context.Context, nodepolicy.DispatchResidentParams) error { grants++; return nil }
				result, err := f.ex.reconcileResident(context.Background(), p)
				if err != nil || result.Outcome != nodepolicy.ResidentNotDispatched || grants != 0 {
					t.Fatalf("result=%+v err=%v grants=%d", result, err, grants)
				}
			}
			if server.loads.Load() != 0 || server.models.Load() != 0 || server.gets.Load() != 0 {
				t.Fatal("preflight made HTTP effects")
			}
		})
	}
	test("one resident among two declarations fits", "-1", 1, "")
	test("native idle expiry refuses", "180", 1, "unsupported-load")
	test("resident selection exceeds router capacity", "-1", 2, "unsupported-load")
}

func TestResidentDailyAdapterSettingsStopCancelsLoad(t *testing.T) {
	f := newResidentFixture(t)
	entered, stopped := make(chan struct{}), make(chan struct{})
	f.transport.onLoad = func(req *http.Request) {
		close(entered)
		<-req.Context().Done()
		close(stopped)
	}
	f.transport.loadErr = context.Canceled
	done := make(chan nodepolicy.ReconcileResidentResult, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		result, err := f.ex.reconcileResident(ctx, f.p)
		if err != nil {
			t.Errorf("reconcile: %v", err)
		}
		done <- result
	}()
	<-entered
	release := stopServingLoads(f.ex.engines[f.p.Engine])
	defer release()
	select {
	case <-stopped:
	case <-time.After(3 * time.Second):
		t.Fatal("settings stop did not cancel the registered resident load")
	}
	result := <-done
	if result.Outcome != nodepolicy.ResidentCancelled || result.Observation != nil || f.transport.loads != 1 || f.transport.reads != 0 {
		t.Fatalf("settings-stop result=%+v loads=%d reads=%d", result, f.transport.loads, f.transport.reads)
	}
	st := f.ex.engines[f.p.Engine]
	st.mu.Lock()
	active := len(st.loads.active)
	st.mu.Unlock()
	if active != 0 {
		t.Fatalf("registered loads leaked: %d", active)
	}
}

func TestResidentDailyAdapterUndeclaredLoadBeforeAdmission(t *testing.T) {
	f := newResidentFixture(t)
	st := f.ex.engines[f.p.Engine]
	st.mu.Lock()
	st.plat.Runtime.Serving = st.plat.Runtime.Serving.Clone()
	st.plat.Runtime.Serving.Models = st.plat.Runtime.Serving.Models[1:]
	st.mu.Unlock()
	_, action, body, ok := f.ex.residentLoadCall(f.p)
	if !ok || f.ex.checkServingLoad(st, action, body) == nil {
		t.Fatal("undeclared load passed the pre-admission declaration check")
	}
	if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched || f.dispatches != 0 {
		t.Fatalf("undeclared result=%+v grants=%d", result, f.dispatches)
	}
	f.noCalls(t)
}

func TestResidentDailyAdapterUnsafeShapes(t *testing.T) {
	test := func(name string, modify func(*Manifest)) {
		t.Run(name, func(t *testing.T) {
			f := newResidentFixture(t)
			modify(f.ex.reg.engines[f.p.Engine])
			if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched || f.dispatches != 0 {
				t.Fatalf("unsafe manifest result=%+v grants=%d", result, f.dispatches)
			}
			f.noCalls(t)
		})
	}
	load := func(change func(*Action)) func(*Manifest) {
		return func(m *Manifest) { a := m.Actions["resident_load"]; change(&a); m.Actions["resident_load"] = a }
	}
	read := func(change func(*Action)) func(*Manifest) {
		return func(m *Manifest) { a := m.Actions["loaded_models"]; change(&a); m.Actions["loaded_models"] = a }
	}
	test("load GET", load(func(a *Action) { a.HTTP.Method = http.MethodGet }))
	test("load query override", load(func(a *Action) { a.HTTP.ParamsIn = "query" }))
	test("load missing schema", load(func(a *Action) { a.HTTP.BodySchema = nil }))
	test("load extra inference input", load(func(a *Action) { a.HTTP.BodySchema = json.RawMessage(`{"model":"string","prompt":"string"}`) }))
	test("load wrong schema type", load(func(a *Action) { a.HTTP.BodySchema = json.RawMessage(`{"model":"number"}`) }))
	test("load command", load(func(a *Action) { a.Cmd = []string{"forbidden"} }))
	test("load remove", load(func(a *Action) { a.RemovePath = &ActionRemovePath{} }))
	test("load model resolution", load(func(a *Action) { a.ModelResolution = "lms-get" }))
	test("load progress", load(func(a *Action) { a.ProgressProtocol = "ndjson" }))
	test("different model parameter", func(m *Manifest) {
		b := m.Operations[engines.OperationLoad]
		b.ModelParam = "name"
		m.Operations[engines.OperationLoad] = b
	})
	test("read POST", read(func(a *Action) { a.HTTP.Method = http.MethodPost }))
	test("read query override", read(func(a *Action) { a.HTTP.ParamsIn = "query" }))
	test("read command", read(func(a *Action) { a.Cmd = []string{"forbidden"} }))
	test("read restart", read(func(a *Action) { a.RestartAfter = true }))
	test("read missing result", read(func(a *Action) { a.Result = nil }))
	test("read missing loaded filter", read(func(a *Action) { a.Result.Match = nil }))
	test("read missing extraction field", read(func(a *Action) { a.Result.Field = "" }))
}

func TestResidentDailyAdapterReapplyRequiresFreshObservation(t *testing.T) {
	f, server, p := residentNativeFixture(t)
	st := settingsState(t, f.ex)
	if got := f.ex.residentObservation(context.Background(), p, st); got == nil || !got.Loaded {
		t.Fatalf("initial observation=%+v", got)
	}
	server.mu.Lock()
	server.props = `{"default_generation_settings":{"n_ctx":4096},"total_slots":2}`
	server.mu.Unlock()
	if got := f.ex.residentObservation(context.Background(), p, st); got != nil {
		t.Fatalf("mismatch observation=%+v", got)
	}
	st.mu.Lock()
	fixturePort := st.port
	st.mu.Unlock()
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	// Keep the HTTP fixture attached after apply restores the configured port.
	st.mu.Lock()
	st.port = fixturePort
	st.mu.Unlock()
	id := f.ex.modelIdentity().Engines["fake"]
	p.Identity.EngineRunID, p.Identity.ModelDigest = id.EngineRunID, id.ModelDigests[p.Model]
	before := server.gets.Load()
	if got := f.ex.residentObservation(context.Background(), p, st); got != nil {
		t.Fatalf("unchanged reapply certified old child: %+v", got)
	}
	if server.gets.Load() != before+1 {
		t.Fatal("reapply used cached allocation rather than fresh props")
	}
}

func TestResidentObservedAllocation(t *testing.T) {
	test := func(name, loaded, props string, wantKnown, wantLoaded bool, wantGets int32, withdrawn bool) {
		t.Run(name, func(t *testing.T) {
			f, server, p := residentNativeFixture(t)
			server.mu.Lock()
			server.loaded, server.props = loaded, props
			server.mu.Unlock()
			got := f.ex.residentObservation(context.Background(), p, settingsState(t, f.ex))
			if (got != nil) != wantKnown || got != nil && (got.Loaded != wantLoaded || got.Identity != p.Identity || got.Generation == 0) {
				t.Fatalf("resident observation = %+v, want known=%v loaded=%v", got, wantKnown, wantLoaded)
			}
			if server.models.Load() != 1 || server.gets.Load() != wantGets || server.loads.Load() != 0 {
				t.Fatalf("models=%d props=%d forbidden loads=%d", server.models.Load(), server.gets.Load(), server.loads.Load())
			}
			if id := f.ex.modelIdentity().Engines["fake"]; (len(id.Serving) == 0) != withdrawn {
				t.Fatalf("identity withdrawn=%v, want %v", len(id.Serving) == 0, withdrawn)
			}
		})
	}
	const loaded = `{"data":[{"id":"resident-model","status":{"value":"loaded"}}]}`
	const props = `{"default_generation_settings":{"n_ctx":8192},"total_slots":2}`
	test("matching model-specific allocation", loaded, props, true, true, 1, false)
	test("native context mismatch", loaded, `{"default_generation_settings":{"n_ctx":4096},"total_slots":2}`, false, false, 1, true)
	test("native slot mismatch", loaded, `{"default_generation_settings":{"n_ctx":8192},"total_slots":1}`, false, false, 1, true)
	test("unknown allocation", loaded, `{}`, false, false, 1, true)
	test("missing loaded inventory", `{}`, props, false, false, 0, true)
	test("unknown loaded semantics", `{"data":[{"id":"resident-model","status":{"value":"surprising"}}]}`, props, false, false, 0, true)
	test("explicit empty stays unloaded", `{"data":[]}`, props, true, false, 0, false)
	test("cold launch is not loaded", `{"data":[{"id":"resident-model","status":{"value":"unloaded"}}]}`, props, true, false, 0, false)
}

func TestResidentPostLoadTransportFailureRemainsUnknown(t *testing.T) {
	test := func(name string, readErr error, readStatus int, propsErr error) {
		t.Run(name, func(t *testing.T) {
			f := newResidentFixture(t)
			st := f.ex.engines[f.p.Engine]
			if got := f.ex.residentObservation(context.Background(), f.p, st); got == nil || !got.Loaded {
				t.Fatalf("initial observation=%+v", got)
			}
			f.transport.readErr, f.transport.readStatus, f.transport.propsErr = readErr, readStatus, propsErr
			result := f.run(t)
			if result.Outcome != nodepolicy.ResidentUnknown || result.Observation != nil {
				t.Fatalf("post-load transport failure satisfied residency: %+v", result)
			}
			st.mu.Lock()
			drift, observed := st.plat.Runtime.servingDrift, st.plat.Runtime.servingObserved
			st.mu.Unlock()
			if drift != "" || observed != nil {
				t.Fatalf("transport failure retained proof or latched drift: observed=%+v drift=%q", observed, drift)
			}
			if f.transport.loads != 1 || f.transport.reads != 2 || f.dispatches != 1 {
				t.Fatalf("post-load effects: loads=%d reads=%d grants=%d", f.transport.loads, f.transport.reads, f.dispatches)
			}
			f.transport.readErr, f.transport.readStatus, f.transport.propsErr = nil, 0, nil
			f.p.OperationID = "resident-retry"
			result = f.run(t)
			if result.Outcome != nodepolicy.ResidentLoaded || result.Observation == nil || !result.Observation.Loaded || result.Observation.Identity != f.p.Identity {
				t.Fatalf("fresh retry did not restore residency: %+v", result)
			}
			if f.transport.loads != 2 || f.transport.reads != 3 || f.transport.forbidden != 0 {
				t.Fatalf("retry effects: loads=%d reads=%d forbidden=%d", f.transport.loads, f.transport.reads, f.transport.forbidden)
			}
		})
	}
	test("inventory connection failure", errors.New("inventory unavailable"), 0, nil)
	test("inventory HTTP 201 is not observation success", nil, http.StatusCreated, nil)
	test("inventory HTTP 204 is not observation success", nil, http.StatusNoContent, nil)
	test("props connection failure", nil, 0, errors.New("props unavailable"))
}

func TestResidentObservedAllocationReplacement(t *testing.T) {
	test := func(name string, replace func(*testing.T, llamaServingFixture, *engineState)) {
		t.Run(name, func(t *testing.T) {
			f, server, p := residentNativeFixture(t)
			st := settingsState(t, f.ex)
			server.mu.Lock()
			server.onProps = func() { replace(t, f, st) }
			server.mu.Unlock()
			if got := f.ex.residentObservation(context.Background(), p, st); got != nil {
				t.Fatalf("replaced evidence reported loaded: %+v", got)
			}
			if server.gets.Load() != 1 || server.loads.Load() != 0 {
				t.Fatalf("props=%d forbidden loads=%d", server.gets.Load(), server.loads.Load())
			}
		})
	}
	test("config replaced during observation", func(_ *testing.T, _ llamaServingFixture, st *engineState) {
		st.mu.Lock()
		st.plat.Runtime.Serving = st.plat.Runtime.Serving.Clone()
		st.mu.Unlock()
	})
	test("run replaced during observation", func(_ *testing.T, _ llamaServingFixture, st *engineState) {
		st.mu.Lock()
		st.runID = "replacement-run"
		st.mu.Unlock()
	})
	test("source changed during observation", func(t *testing.T, f llamaServingFixture, _ *engineState) {
		data, err := os.ReadFile(f.source)
		if err != nil {
			t.Fatal(err)
		}
		writeServingFixture(t, f.source, append(data, []byte("\n# replaced source\n")...))
	})
}

type residentTransport struct {
	mu                      sync.Mutex
	loads, forbidden, reads int
	body                    map[string]json.RawMessage
	loaded                  string
	loadErr                 error
	readErr                 error
	readStatus              int
	propsErr                error
	onLoad                  func(*http.Request)
}

type residentDeadlineContext struct {
	context.Context
	done     chan struct{}
	deadline time.Time
}

func newResidentDeadlineContext() (*residentDeadlineContext, func()) {
	ctx := &residentDeadlineContext{Context: context.Background(), done: make(chan struct{}), deadline: time.Now().Add(time.Hour)}
	return ctx, func() { close(ctx.done) }
}

func (ctx *residentDeadlineContext) Deadline() (time.Time, bool) {
	return ctx.deadline, true
}

func (ctx *residentDeadlineContext) Done() <-chan struct{} { return ctx.done }

func (ctx *residentDeadlineContext) Err() error {
	select {
	case <-ctx.done:
		return context.DeadlineExceeded
	default:
		return nil
	}
}

func (r *residentTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	switch req.URL.Path {
	case "/fixture/load":
		r.loads++
		if err := json.NewDecoder(req.Body).Decode(&r.body); err != nil {
			return nil, err
		}
		if _, prompt := r.body["prompt"]; prompt {
			r.forbidden++
		}
		if req.Method != http.MethodPost {
			r.forbidden++
		}
		if r.onLoad != nil {
			r.onLoad(req)
		}
		if r.loadErr != nil {
			return nil, r.loadErr
		}
		return residentJSON(`{}`), nil
	case "/fixture/loaded":
		r.reads++
		if r.readErr != nil {
			return nil, r.readErr
		}
		response := residentJSON(r.loaded)
		if r.readStatus != 0 {
			response.StatusCode = r.readStatus
		}
		return response, nil
	case "/props":
		if req.URL.Query().Get("autoload") != "false" || req.URL.Query().Get("model") != "m" {
			r.forbidden++
			return nil, errors.New("unsafe props")
		}
		if r.propsErr != nil {
			return nil, r.propsErr
		}
		return residentJSON(`{"default_generation_settings":{"n_ctx":8192},"total_slots":2}`), nil
	default:
		// Includes start, pull, delete, copy and arbitrary action endpoints.
		r.forbidden++
		return nil, errors.New("forbidden engine operation")
	}
}

func residentJSON(body string) *http.Response {
	return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}

type residentCodecIO struct{ write func([]byte) (int, error) }

func (*residentCodecIO) Read([]byte) (int, error)       { return 0, io.EOF }
func (rw *residentCodecIO) Write(p []byte) (int, error) { return rw.write(p) }

type residentFixture struct {
	ex           *Executor
	m            *Manager
	p            nodepolicy.ReconcileResidentParams
	transport    *residentTransport
	grant        func(residentDispatchRequest)
	responses    chan Message
	dispatches   int
	contentCalls int
}

func newResidentFixture(t *testing.T) *residentFixture {
	t.Helper()
	native, _ := residentOwnedFixture(t, "m")
	ex := native.ex
	st := settingsState(t, ex)
	mf := st.manifest
	mf.ModelNaming = engines.ModelNamingExact
	mf.Operations = map[string]OperationBinding{engines.OperationLoad: {Action: "resident_load", Params: map[string]json.RawMessage{}}}
	mf.Actions = map[string]Action{
		"resident_load": {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/fixture/load", BodySchema: json.RawMessage(`{"model":"string"}`)}},
		"list_models":   {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/api/tags"}, Result: &ActionResult{Array: "models", Field: "name", Digest: "digest"}},
		"loaded_models": {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/fixture/loaded"}, Result: &ActionResult{Array: "data", Field: "id", Match: &ResultMatch{Field: "status.value", In: []string{"loaded"}}}},
		"pull_model":    {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/forbidden/pull"}},
		"delete_model":  {HTTP: &ActionHTTP{Method: http.MethodDelete, Path: "/forbidden/delete"}},
		"arbitrary":     {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/forbidden/action"}},
	}
	ex.nodeProjection.set(nodeProjection{})
	if err := ex.desired.set("fake", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	ex.storeInventory(map[string][]string{"fake": {"m"}}, nil, ex.sweeps.Add(1))
	ex.settingsHub.Publish([]settings.Snapshot{})
	f := &residentFixture{ex: ex, transport: &residentTransport{loaded: `{"data":[{"id":"m","status":{"value":"loaded"}}]}`}, responses: make(chan Message, 8)}
	ex.client = &http.Client{Transport: f.transport}
	ex.modelMutationParent = func(context.Context, string) error { f.contentCalls++; return errors.New("forbidden content mutation") }
	id := ex.modelIdentity().Engines["fake"]
	f.p = nodepolicy.ReconcileResidentParams{OperationID: "resident-op", PolicyRevision: "1", DeclaredResidentCount: 1, Engine: "fake", Model: "m",
		Identity:  nodepolicy.StrictModelIdentity{EngineRunID: st.runID, SettingsFingerprint: nameDigest("settings-and-launch"), ModelDigest: id.ModelDigests["m"]},
		TimeoutMs: 60000, Reservation: nodepolicy.ResidentReservationEvidence{ReservationID: "reservation-1", ProxyRunID: "proxy-1", ManagerIncarnation: ex.switchIncarnation},
		Load: nodepolicy.ResidentLoadOptions{ContextLength: 8192, KeepAliveSeconds: 120}}
	codec := NewCodec(&residentCodecIO{write: func(raw []byte) (int, error) {
		var msg Message
		if err := json.Unmarshal(raw, &msg); err != nil {
			return 0, err
		}
		if msg.Method == residentDispatchRequestMethod {
			var p residentDispatchRequest
			if err := json.Unmarshal(msg.Params, &p); err != nil {
				return 0, err
			}
			f.dispatches++
			if f.grant != nil {
				f.grant(p)
			} else {
				f.reply(t, p.ID, nodepolicy.DispatchResidentResult{Dispatched: true})
			}
		} else {
			f.responses <- msg
		}
		return len(raw), nil
	}})
	f.m = NewManager(codec, ex, nil, "")
	return f
}

func (f *residentFixture) reply(t *testing.T, id string, result nodepolicy.DispatchResidentResult) {
	t.Helper()
	raw, err := json.Marshal(residentDispatchReply{ID: id, Result: &result})
	if err != nil {
		t.Fatal(err)
	}
	f.m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", Method: residentDispatchReplyMethod, Params: raw})
}

func (f *residentFixture) run(t *testing.T) nodepolicy.ReconcileResidentResult {
	t.Helper()
	result, err := f.ex.reconcileResident(context.Background(), f.p)
	if err != nil {
		t.Fatal(err)
	}
	if err := result.Validate(); err != nil {
		t.Fatalf("invalid result: %v", err)
	}
	return result
}

func (f *residentFixture) noCalls(t *testing.T) {
	t.Helper()
	if f.transport.loads != 0 || f.transport.forbidden != 0 || f.contentCalls != 0 || f.transport.reads != 0 {
		t.Fatalf("refusal made engine calls: load=%d forbidden=%d content=%d read=%d", f.transport.loads, f.transport.forbidden, f.contentCalls, f.transport.reads)
	}
}

func TestResidentContextBoundConfigured(t *testing.T) {
	test := func(name string, context int, envelopes []engines.ServingEnvelope, want bool) {
		t.Run(name, func(t *testing.T) {
			p := nodepolicy.ReconcileResidentParams{Model: "m", Load: nodepolicy.ResidentLoadOptions{ContextLength: context}}
			readback := &engines.ServingReadback{Configured: envelopes}
			if got := residentConfiguredContext(p, engines.ModelNamingImpliedLatestTag, readback); got != want {
				t.Fatalf("configured context accepted=%v, want %v", got, want)
			}
		})
	}
	bound := []engines.ServingEnvelope{{ModelID: "m", PerSlotContextTokens: 4096}}
	test("exact native bound", 4096, bound, true)
	test("bound plus one", 4097, bound, false)
	test("smaller profile cannot shrink native allocation", 2048, bound, false)
	test("absent bound", 4096, nil, false)
	test("zero bound", 4096, []engines.ServingEnvelope{{ModelID: "m", PerSlotContextTokens: 0}}, false)
	test("negative bound", 4096, []engines.ServingEnvelope{{ModelID: "m", PerSlotContextTokens: -1}}, false)
	test("zero request", 0, bound, false)
	test("negative request", -1, bound, false)
	test("maximum integer request", int(^uint(0)>>1), bound, false)
	test("below protocol ceiling above model limit", 8192, bound, false)
	test("normalized collision", 4096, append(append([]engines.ServingEnvelope{}, bound...), engines.ServingEnvelope{ModelID: "m", PerSlotContextTokens: 4096}), false)
	test("different per-model bounds", 4096, append(append([]engines.ServingEnvelope{}, bound...), engines.ServingEnvelope{ModelID: "other", PerSlotContextTokens: 2048}), true)
	test("another model cannot supply bound", 4096, []engines.ServingEnvelope{{ModelID: "other", PerSlotContextTokens: 4096}}, false)
	test("largest int64 native bound", 4096, []engines.ServingEnvelope{{ModelID: "m", PerSlotContextTokens: 1<<63 - 1}}, false)
	if residentConfiguredContext(nodepolicy.ReconcileResidentParams{}, "", nil) {
		t.Fatal("missing cold readback accepted")
	}
}

func TestResidentContextBoundColdPreflight(t *testing.T) {
	test := func(name string, contextLength int, wantBound bool) {
		t.Run(name, func(t *testing.T) {
			cold := newLlamaServingFixture(t)
			applyLlamaServing(t, cold, approveLlamaServing(t, cold))
			f := newResidentFixture(t)
			st := f.ex.engines["fake"]
			st.manifest.Family = "llamacpp"
			// Compare cold bounds independently; never transplant Runtime state.
			coldState := settingsState(t, cold.ex)
			coldState.opMu.Lock()
			readback := cold.ex.launchStateLocked("fake", coldState).ServingReadback
			coldState.opMu.Unlock()
			st.mu.Lock()
			st.plat.Runtime.servingLaunch = nil
			st.mu.Unlock()
			f.p.Load.ContextLength = contextLength
			st.opMu.Lock()
			coldParams := f.p
			coldParams.Model = "medium"
			bounded := residentConfiguredContext(coldParams, st.manifest.ModelNamingOrDefault(), readback)
			reason := f.ex.residentCheck(context.Background(), f.p)
			st.opMu.Unlock()
			if bounded != wantBound {
				t.Fatalf("cold configured context accepted=%v, want %v", bounded, wantBound)
			}
			if reason != "unqualified" {
				t.Fatalf("cold preflight=%q, want unqualified", reason)
			}
			if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched || f.dispatches != 0 {
				t.Fatalf("native resident execution escaped restriction: %+v, grants=%d", result, f.dispatches)
			}
			f.noCalls(t)
			assertNoLlamaSpawn(t, cold)
		})
	}
	// NE identity requires an owned launch; an exact cold bound cannot bypass the earlier identity check.
	test("exact cold allocation lacks owned identity", 8192, true)
	test("native allocation above requested profile", 4096, false)
	test("request exceeds cold allocation", 8193, false)
}

func TestResidentLoadMappingAndObservation(t *testing.T) {
	f := newResidentFixture(t)
	f.grant = func(req residentDispatchRequest) {
		want := nodepolicy.DispatchResidentParams{ReservationID: f.p.Reservation.ReservationID, ProxyRunID: f.p.Reservation.ProxyRunID, ManagerIncarnation: f.ex.switchIncarnation,
			OperationID: f.p.OperationID, PolicyRevision: f.p.PolicyRevision, Engine: f.p.Engine, Model: f.p.Model, Identity: f.p.Identity}
		if !reflect.DeepEqual(req.Params, want) {
			t.Errorf("dispatch params = %+v, want %+v", req.Params, want)
		}
		// RPC waits must not hold inventory, publication or content locks.
		if !f.ex.inventoryMu.TryLock() {
			t.Fatal("inventory lock held")
		}
		f.ex.inventoryMu.Unlock()
		if !f.ex.publishMu.TryLock() {
			t.Fatal("publication lock held")
		}
		f.ex.publishMu.Unlock()
		st := f.ex.engines["fake"]
		select {
		case st.contentMu <- struct{}{}:
			<-st.contentMu
		default:
			t.Fatal("content guard held")
		}
		f.reply(t, req.ID, nodepolicy.DispatchResidentResult{Dispatched: true})
	}
	result := f.run(t)
	if result.Outcome != nodepolicy.ResidentLoaded || result.Observation == nil {
		t.Fatalf("result = %+v", result)
	}
	want := nodepolicy.ResidentLoadedObservation{ManagerIncarnation: f.ex.switchIncarnation, Generation: 1, Engine: "fake", Model: "m", Identity: f.p.Identity, Loaded: true}
	if !reflect.DeepEqual(*result.Observation, want) {
		t.Fatalf("observation = %+v, want %+v", result.Observation, want)
	}
	var body struct {
		Model     string `json:"model"`
		KeepAlive int    `json:"keep_alive"`
		Stream    bool   `json:"stream"`
		Options   struct {
			Context int `json:"num_ctx"`
		} `json:"options"`
	}
	raw, err := json.Marshal(f.transport.body)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	if body.Model != "m" || body.KeepAlive != 0 || body.Options.Context != 0 || body.Stream || len(f.transport.body) != 1 {
		t.Fatalf("load body = %+v", body)
	}
	if f.transport.loads != 1 || f.transport.forbidden != 0 || f.contentCalls != 0 || f.dispatches != 1 {
		t.Fatalf("calls: load=%d forbidden=%d content=%d grants=%d", f.transport.loads, f.transport.forbidden, f.contentCalls, f.dispatches)
	}
	select {
	case <-f.ex.loadedPoke:
	default:
		t.Fatal("watcher not poked")
	}
}

func TestResidentRefusesWithoutSideEffects(t *testing.T) {
	test := func(name, reason string, modify func(*residentFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newResidentFixture(t)
			modify(f)
			if got := f.ex.residentCheck(context.Background(), f.p); got != reason {
				t.Fatalf("reason = %q, want %q", got, reason)
			}
			if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched {
				t.Errorf("result = %+v", result)
			}
			f.noCalls(t)
			if f.dispatches != 0 {
				t.Fatal("refusal asked for dispatch")
			}
		})
	}
	test("unknown projection after restart", "projection-unknown", func(f *residentFixture) { f.ex.nodeProjection = nodeProjectionStore{} })
	test("paused locally edited policy", "paused-local", func(f *residentFixture) { f.ex.nodeProjection.set(nodeProjection{PausedLocally: true}) })
	test("missing installed model", "not-installed", func(f *residentFixture) { f.ex.inventory["fake"] = []string{} })
	test("unknown installed inventory", "not-installed", func(f *residentFixture) { delete(f.ex.inventory, "fake") })
	test("implicit binding", "unsupported-load", func(f *residentFixture) { delete(f.ex.reg.engines["fake"].Operations, engines.OperationLoad) })
	test("restart binding", "unsupported-load", func(f *residentFixture) {
		mf := f.ex.reg.engines["fake"]
		a := mf.Actions["resident_load"]
		a.RestartAfter = true
		mf.Actions["resident_load"] = a
	})
	test("prompt bearing fixed options", "unsupported-load", func(f *residentFixture) {
		f.ex.reg.engines["fake"].Operations[engines.OperationLoad].Params["prompt"] = json.RawMessage(`"forbidden"`)
	})
	test("unexpressible fixed context", "unsupported-load", func(f *residentFixture) {
		f.ex.reg.engines["fake"].Operations[engines.OperationLoad].Params["options"] = json.RawMessage(`{"num_ctx":8}`)
	})
	test("missing load endpoint", "unsupported-load", func(f *residentFixture) {
		f.ex.reg.engines["fake"].Actions["resident_load"].HTTP.Path = ""
	})
	test("new manager incarnation", "unqualified", func(f *residentFixture) { f.p.Reservation.ManagerIncarnation = "old-manager" })
	test("omitted manager incarnation", "unqualified", func(f *residentFixture) { f.p.Reservation.ManagerIncarnation = "" })
	test("stale run", "unqualified", func(f *residentFixture) { f.p.Identity.EngineRunID = "old-run" })
	test("missing launch identity", "unqualified", func(f *residentFixture) { f.ex.engines["fake"].launchDigest = "" })
	test("same digest under another model", "unqualified", func(f *residentFixture) { f.p.Model = "n"; f.ex.inventory["fake"] = []string{"m", "n"} })
	test("different model digest", "unqualified", func(f *residentFixture) { f.p.Identity.ModelDigest = nameDigest("other") })
	test("withdrawn identity", "unqualified", func(f *residentFixture) {
		st := f.ex.engines["fake"]
		st.mu.Lock()
		st.plat.Runtime.servingDrift = "withdrawn"
		st.mu.Unlock()
	})
	test("engine stopped", "engine-off", func(f *residentFixture) { f.ex.engines["fake"].running = false })
	test("peer Off", "engine-off", func(f *residentFixture) {
		if err := f.ex.desired.set("fake", false, intentPeer); err != nil {
			t.Fatal(err)
		}
	})
	test("local Off", "engine-off-local", func(f *residentFixture) {
		if err := f.ex.desired.set("fake", false, intentLocal); err != nil {
			t.Fatal(err)
		}
	})
	test("legacy Off", "engine-off-local", func(f *residentFixture) {
		if err := f.ex.desired.save(desiredStateFile{Engines: map[string]bool{"fake": false}}); err != nil {
			t.Fatal(err)
		}
	})
}

// No sleeps: synchronize on E3's admitted mutation count before releasing opMu.
func residentAwaitMutation(t *testing.T, ex *Executor) {
	t.Helper()
	deadline := time.After(5 * time.Second)
	for {
		ex.switchAdmission.Lock()
		count := ex.switchMutations["fake"]
		ex.switchAdmission.Unlock()
		if count != 0 {
			return
		}
		select {
		case <-deadline:
			t.Fatal("resident did not enter E3")
		default:
			runtime.Gosched()
		}
	}
}

func TestResidentPostLockVeto(t *testing.T) {
	test := func(name string, veto func(*residentFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newResidentFixture(t)
			st := f.ex.engines["fake"]
			st.opMu.Lock()
			done := make(chan nodepolicy.ReconcileResidentResult, 1)
			go func() { done <- f.run(t) }()
			residentAwaitMutation(t, f.ex)
			veto(f)
			st.opMu.Unlock()
			result := <-done
			if result.Outcome != nodepolicy.ResidentNotDispatched {
				t.Errorf("result = %+v", result)
			}
			f.noCalls(t)
		})
	}
	test("projection pause wins", func(f *residentFixture) { f.ex.nodeProjection.set(nodeProjection{PausedLocally: true}) })
	test("Off provenance wins", func(f *residentFixture) {
		if err := f.ex.desired.set("fake", false, intentLocal); err != nil {
			t.Fatal(err)
		}
	})
}

func TestResidentSwitchExclusion(t *testing.T) {
	f := newResidentFixture(t)
	f.grant = func(req residentDispatchRequest) {
		// Exercise E3's actual switch configuration gate while the load is admitted.
		f.ex.switchAdmission.Lock()
		_, err := f.ex.switchConfiguration(desiredStateFile{})
		f.ex.switchAdmission.Unlock()
		var busy *switchBusyError
		if !errors.As(err, &busy) {
			t.Errorf("switch was not excluded: %v", err)
		}
		f.reply(t, req.ID, nodepolicy.DispatchResidentResult{Dispatched: true})
	}
	if result := f.run(t); result.Outcome != nodepolicy.ResidentLoaded {
		t.Fatalf("result = %+v", result)
	}
}

func TestResidentRefusalLogging(t *testing.T) {
	const prompt = "private prompt sentinel"
	const body = "private response body sentinel"
	test := func(name, reason, level string, setup func(*testing.T, *residentFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newResidentFixture(t)
			setup(t, f)
			var logs bytes.Buffer
			previous := slog.Default()
			slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
			t.Cleanup(func() { slog.SetDefault(previous) })

			if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched {
				t.Fatalf("result = %+v", result)
			}
			f.noCalls(t)
			var entry struct {
				Level       string `json:"level"`
				OperationID string `json:"operationId"`
				Engine      string `json:"engine"`
				Model       string `json:"model"`
				Reason      string `json:"reason"`
			}
			// Unmarshal also rejects multiple log lines for the same refusal.
			if err := json.Unmarshal(logs.Bytes(), &entry); err != nil {
				t.Fatalf("decode refusal log: %v", err)
			}
			if entry.Level != level || entry.OperationID != f.p.OperationID || entry.Engine != f.p.Engine || entry.Model != f.p.Model || entry.Reason != reason {
				t.Fatalf("refusal log = %+v, want %s with operation/ref and reason %q", entry, level, reason)
			}
			for _, private := range []string{prompt, body, f.p.Identity.ModelDigest, f.p.Identity.SettingsFingerprint} {
				if strings.Contains(logs.String(), private) {
					t.Fatal("refusal log contains private content")
				}
			}
		})
	}
	test("pre-lock not installed", "not-installed", "INFO", func(_ *testing.T, f *residentFixture) {
		f.ex.inventory["fake"] = nil
	})
	test("final grant reservation revoked", "reservation-revoked", "INFO", func(t *testing.T, f *residentFixture) {
		f.grant = func(req residentDispatchRequest) {
			f.reply(t, req.ID, nodepolicy.DispatchResidentResult{Reason: nodepolicy.DispatchReasonReservationRevoked})
		}
	})
	test("relay error omits private content", "dispatch-relay-failed", "WARN", func(_ *testing.T, f *residentFixture) {
		f.m.residentDispatch.send = func(context.Context, string, any) error { return errors.New(prompt + "\n" + body) }
	})
}

func TestResidentDispatchRefusal(t *testing.T) {
	for _, reason := range []nodepolicy.DispatchResidentReason{"reservation-revoked", "reservation-unknown", "engine-off", "no-fit", "paused-local", "engine-off-local", "not-served", "not-installed", "unqualified", "unsupported-load", "switch-busy", "operation-timeout", "observation-unknown", "projection-unknown"} {
		t.Run(string(reason), func(t *testing.T) {
			f := newResidentFixture(t)
			f.grant = func(req residentDispatchRequest) {
				f.reply(t, req.ID, nodepolicy.DispatchResidentResult{Reason: reason})
			}
			if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched {
				t.Errorf("result = %+v", result)
			}
			f.noCalls(t)
			if len(f.m.residentDispatch.pending) != 0 {
				t.Fatal("pending reply leaked")
			}
		})
	}
}

func TestResidentDispatchFailure(t *testing.T) {
	test := func(name string, setup func(*residentFixture, context.CancelFunc), deadline bool) {
		t.Run(name, func(t *testing.T) {
			f := newResidentFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if deadline {
				var end context.CancelFunc
				ctx, end = context.WithDeadline(ctx, time.Now().Add(time.Millisecond))
				defer end()
			}
			setup(f, cancel)
			result, err := f.ex.reconcileResident(ctx, f.p)
			if err != nil || result.Outcome != nodepolicy.ResidentNotDispatched {
				t.Fatalf("result = %+v, %v", result, err)
			}
			f.noCalls(t)
			if len(f.m.residentDispatch.pending) != 0 {
				t.Fatal("pending reply leaked")
			}
		})
	}
	test("missing parent", func(f *residentFixture, _ context.CancelFunc) { f.ex.residentParent = nil }, false)
	test("send failure", func(f *residentFixture, _ context.CancelFunc) {
		f.m.residentDispatch.send = func(context.Context, string, any) error { return io.ErrClosedPipe }
	}, false)
	test("timeout", func(f *residentFixture, _ context.CancelFunc) { f.grant = func(residentDispatchRequest) {} }, true)
	test("cancel", func(f *residentFixture, cancel context.CancelFunc) {
		f.grant = func(residentDispatchRequest) { cancel() }
	}, false)
	test("queued grant then cancelled", func(f *residentFixture, cancel context.CancelFunc) {
		f.grant = func(req residentDispatchRequest) {
			f.reply(t, req.ID, nodepolicy.DispatchResidentResult{Dispatched: true})
			cancel()
		}
	}, false)
	test("client request cannot reply", func(f *residentFixture, cancel context.CancelFunc) {
		f.grant = func(req residentDispatchRequest) {
			data, err := json.Marshal(residentDispatchReply{ID: req.ID, Result: &nodepolicy.DispatchResidentResult{Dispatched: true}})
			if err != nil {
				t.Fatal(err)
			}
			id := json.RawMessage(`9`)
			go f.m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: residentDispatchReplyMethod, Params: data})
			cancel()
		}
	}, false)
	test("mismatched reply", func(f *residentFixture, cancel context.CancelFunc) {
		f.grant = func(residentDispatchRequest) {
			f.reply(t, "wrong-id", nodepolicy.DispatchResidentResult{Dispatched: true})
			cancel()
		}
	}, false)
	test("malformed reply", func(f *residentFixture, cancel context.CancelFunc) {
		f.grant = func(req residentDispatchRequest) {
			f.m.residentDispatch.reply(json.RawMessage(`{"id":"` + req.ID + `","result":{"dispatched":true,"extra":1}}`))
			cancel()
		}
	}, false)
	test("error plus grant", func(f *residentFixture, _ context.CancelFunc) {
		f.grant = func(req residentDispatchRequest) {
			data, err := json.Marshal(residentDispatchReply{ID: req.ID, Result: &nodepolicy.DispatchResidentResult{Dispatched: true}, Error: &RPCError{Code: -32000, Message: "refused"}})
			if err != nil {
				t.Fatal(err)
			}
			f.m.residentDispatch.reply(data)
		}
	}, false)
}

func TestResidentReadback(t *testing.T) {
	test := func(name, loaded string, loadErr error, want nodepolicy.ResidentOutcome, observation *bool) {
		t.Run(name, func(t *testing.T) {
			f := newResidentFixture(t)
			f.transport.loaded, f.transport.loadErr = loaded, loadErr
			result := f.run(t)
			if result.Outcome != want {
				t.Fatalf("outcome = %s, want %s", result.Outcome, want)
			}
			if observation == nil {
				if result.Observation != nil {
					t.Fatalf("invented observation: %+v", result.Observation)
				}
			} else if result.Observation == nil || result.Observation.Loaded != *observation {
				t.Fatalf("observation = %+v", result.Observation)
			}
			if f.transport.loads != 1 || f.transport.forbidden != 0 {
				t.Fatalf("load=%d forbidden=%d", f.transport.loads, f.transport.forbidden)
			}
			select {
			case <-f.ex.loadedPoke:
			default:
				t.Fatal("failed load did not poke watcher")
			}
		})
	}
	yes, no := true, false
	test("lost response but loaded", `{"data":[{"id":"m","status":{"value":"loaded"}}]}`, io.ErrUnexpectedEOF, nodepolicy.ResidentLoaded, &yes)
	test("missing inventory", `{}`, nil, nodepolicy.ResidentUnknown, nil)
	test("null inventory", `{"data":null}`, nil, nodepolicy.ResidentUnknown, nil)
	test("explicit empty", `{"data":[]}`, nil, nodepolicy.ResidentFailed, &no)
	test("failed load empty", `{"data":[]}`, io.ErrUnexpectedEOF, nodepolicy.ResidentFailed, &no)
	test("failed load missing", `{}`, io.ErrUnexpectedEOF, nodepolicy.ResidentUnknown, nil)
}

func TestResidentDuplicateOperation(t *testing.T) {
	f := newResidentFixture(t)
	first := f.run(t)
	second := f.run(t)
	if !reflect.DeepEqual(first, second) || f.transport.loads != 1 {
		t.Fatalf("duplicate results differ or repeated load: %+v %+v calls=%d", first, second, f.transport.loads)
	}
	f.p.Load.ContextLength++
	if _, err := f.ex.reconcileResident(context.Background(), f.p); err == nil {
		t.Fatal("changed operation payload accepted")
	}
	if f.transport.loads != 1 {
		t.Fatal("conflict loaded again")
	}
}

func TestResidentConcurrentDuplicate(t *testing.T) {
	f := newResidentFixture(t)
	entered, release := make(chan struct{}), make(chan struct{})
	f.grant = func(req residentDispatchRequest) {
		close(entered)
		<-release
		f.reply(t, req.ID, nodepolicy.DispatchResidentResult{Dispatched: true})
	}
	done := make(chan nodepolicy.ReconcileResidentResult, 2)
	go func() { done <- f.run(t) }()
	<-entered
	go func() { done <- f.run(t) }()
	changed := f.p
	changed.Reservation.ReservationID = "other-reservation"
	if _, err := f.ex.reconcileResident(context.Background(), changed); err == nil {
		t.Fatal("in-flight changed payload accepted")
	}
	close(release)
	a, b := <-done, <-done
	if !reflect.DeepEqual(a, b) || a.Outcome != nodepolicy.ResidentLoaded || f.transport.loads != 1 || f.dispatches != 1 {
		t.Fatalf("duplicate results=%+v %+v calls=%d grants=%d", a, b, f.transport.loads, f.dispatches)
	}
}

func TestResidentIdentityWithdrawnDuringLoad(t *testing.T) {
	f := newResidentFixture(t)
	f.transport.onLoad = func(*http.Request) {
		st := f.ex.engines["fake"]
		st.mu.Lock()
		st.plat.Runtime.servingDrift = "withdrawn"
		st.mu.Unlock()
	}
	result := f.run(t)
	if result.Outcome != nodepolicy.ResidentUnknown || result.Observation != nil {
		t.Fatalf("withdrawn identity reported loaded: %+v", result)
	}
}

func TestResidentLoadCancellation(t *testing.T) {
	test := func(name string, deadline bool) {
		t.Run(name, func(t *testing.T) {
			f := newResidentFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var expireDeadline func()
			if deadline {
				ctx, expireDeadline = newResidentDeadlineContext()
			}
			f.transport.onLoad = func(req *http.Request) {
				if deadline {
					expireDeadline()
				} else {
					cancel()
				}
				<-req.Context().Done()
			}
			f.transport.loadErr = context.Canceled
			result, err := f.ex.reconcileResident(ctx, f.p)
			want := nodepolicy.ResidentCancelled
			if deadline {
				want = nodepolicy.ResidentTimeout
			}
			if err != nil || result.Outcome != want || result.Observation != nil || f.transport.loads != 1 {
				t.Fatalf("result=%+v err=%v calls=%d", result, err, f.transport.loads)
			}
		})
	}
	test("cancelled after dispatch", false)
	test("deadline after dispatch", true)
}

func TestResidentLifecycleWaitCancelled(t *testing.T) {
	f := newResidentFixture(t)
	st := f.ex.engines["fake"]
	st.opMu.Lock()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan nodepolicy.ReconcileResidentResult, 1)
	go func() {
		result, err := f.ex.reconcileResident(ctx, f.p)
		if err != nil {
			t.Errorf("reconcile: %v", err)
		}
		done <- result
	}()
	residentAwaitMutation(t, f.ex)
	cancel()
	result := <-done
	st.opMu.Unlock()
	if result.Outcome != nodepolicy.ResidentNotDispatched {
		t.Fatalf("result = %+v", result)
	}
	f.noCalls(t)
}

func TestResidentVetoDuringFinalGrant(t *testing.T) {
	f := newResidentFixture(t)
	f.grant = func(req residentDispatchRequest) {
		f.m.applyNodeProjection(&Message{Params: json.RawMessage(`{"policy":{"revision":"1","roles":{}},"pausedLocally":true}`)})
		f.reply(t, req.ID, nodepolicy.DispatchResidentResult{Dispatched: true})
	}
	if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched {
		t.Errorf("result = %+v", result)
	}
	f.noCalls(t)
}

func TestResidentPendingBounds(t *testing.T) {
	t.Run("relay map", func(t *testing.T) {
		f := newResidentFixture(t)
		f.m.residentDispatch.pending = make(map[string]chan residentDispatchReply)
		for i := 0; i < maxResidentPending; i++ {
			f.m.residentDispatch.pending[strings.Repeat("x", i+1)] = make(chan residentDispatchReply, 1)
		}
		if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched {
			t.Fatalf("result = %+v", result)
		}
		f.noCalls(t)
		if f.dispatches != 0 || len(f.m.residentDispatch.pending) != maxResidentPending {
			t.Fatal("full relay map changed or sent")
		}
	})
	t.Run("operation map", func(t *testing.T) {
		f := newResidentFixture(t)
		f.ex.residents.ops = make(map[string]*residentOperation)
		for i := 0; i < maxResidentOperations; i++ {
			f.ex.residents.ops[strings.Repeat("x", i+1)] = &residentOperation{done: make(chan struct{})}
		}
		if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched {
			t.Fatalf("result = %+v", result)
		}
		f.noCalls(t)
		if len(f.ex.residents.ops) != maxResidentOperations {
			t.Fatal("full operation map changed")
		}
	})
}

func TestResidentMalformedGrantNeverAuthorizesLiveContext(t *testing.T) {
	test := func(name string, envelope func(string) string, clientRequest bool) {
		t.Run(name, func(t *testing.T) {
			f := newResidentFixture(t)
			f.grant = func(req residentDispatchRequest) {
				raw := json.RawMessage(envelope(req.ID))
				msg := &Message{JSONRPC: "2.0", Method: residentDispatchReplyMethod, Params: raw}
				if clientRequest {
					id := json.RawMessage(`10`)
					msg.ID = &id
					go f.m.handleMessage(context.Background(), msg)
				} else {
					f.m.handleMessage(context.Background(), msg)
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
			defer cancel()
			result, err := f.ex.reconcileResident(ctx, f.p)
			if err != nil || result.Outcome != nodepolicy.ResidentNotDispatched {
				t.Errorf("result=%+v err=%v", result, err)
			}
			f.noCalls(t)
			if f.dispatches != 1 || len(f.m.residentDispatch.pending) != 0 {
				t.Fatalf("dispatches=%d pending=%d", f.dispatches, len(f.m.residentDispatch.pending))
			}
		})
	}
	grant := func(id string) string { return `{"id":"` + id + `","result":{"dispatched":true}}` }
	test("client request", grant, true)
	test("mismatched id", func(string) string { return grant("unrelated") }, false)
	test("unknown result field", func(id string) string { return `{"id":"` + id + `","result":{"dispatched":true,"extra":true}}` }, false)
	test("duplicate envelope key", func(id string) string { return `{"id":"` + id + `","id":"` + id + `","result":{"dispatched":true}}` }, false)
	test("case variant envelope key", func(id string) string { return `{"ID":"` + id + `","result":{"dispatched":true}}` }, false)
	test("null error alongside grant", func(id string) string { return `{"id":"` + id + `","result":{"dispatched":true},"error":null}` }, false)
}

func TestResidentOldSweepCannotReplaceNewInventory(t *testing.T) {
	f := newResidentFixture(t)
	sweep := f.ex.sweeps.Add(2)
	f.ex.storeInventory(map[string][]string{"fake": {}}, nil, sweep)
	f.ex.storeInventory(map[string][]string{"fake": {"m"}}, map[string]observedDigests{"fake": {runID: f.p.Identity.EngineRunID, models: map[string]string{"m": f.p.Identity.ModelDigest}}}, sweep-1)
	if got := f.ex.cachedInventory()["fake"]; len(got) != 0 {
		t.Fatalf("old sweep restored installed inventory: %v", got)
	}
	if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched {
		t.Fatalf("result = %+v", result)
	}
	f.noCalls(t)
}

func TestResidentNoContentFallback(t *testing.T) {
	f := newResidentFixture(t)
	f.ex.inventory["fake"] = []string{}
	result := f.run(t)
	f.noCalls(t)
	if result.Outcome != nodepolicy.ResidentNotDispatched {
		t.Fatalf("result = %+v", result)
	}
}

func TestResidentStrictDecode(t *testing.T) {
	test := func(name string, modify func(string) string) {
		t.Run(name, func(t *testing.T) {
			f := newResidentFixture(t)
			raw, err := json.Marshal(f.p)
			if err != nil {
				t.Fatal(err)
			}
			id := json.RawMessage(`1`)
			f.m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: nodepolicy.MethodReconcileResident, Params: json.RawMessage(modify(string(raw)))})
			response := <-f.responses
			if response.Error == nil || response.Error.Code != -32602 {
				t.Fatalf("response = %+v", response)
			}
			f.noCalls(t)
		})
	}
	test("unknown field", func(s string) string {
		return strings.Replace(s, `"timeoutMs":60000`, `"extra":true,"timeoutMs":60000`, 1)
	})
	test("duplicate field", func(s string) string {
		return strings.Replace(s, `"timeoutMs":60000`, `"timeoutMs":1,"timeoutMs":60000`, 1)
	})
	test("null load", func(s string) string {
		return strings.Replace(s, `{"contextLength":8192,"keepAliveSeconds":120}`, `null`, 1)
	})
	test("zero context", func(s string) string { return strings.Replace(s, `"contextLength":8192`, `"contextLength":0`, 1) })
	test("unbounded keep alive", func(s string) string {
		return strings.Replace(s, `"keepAliveSeconds":120`, `"keepAliveSeconds":86401`, 1)
	})
}

func TestResidentSwitchBusy(t *testing.T) {
	f := newResidentFixture(t)
	_, p := switchFixture(t)
	p.SourceEngine, p.TargetEngine, p.ManagerIncarnation = "fake", "other", f.ex.switchIncarnation
	p.ConfigurationFence = "e3v1:" + nameDigest("configuration")
	state, err := f.ex.desired.load()
	if err != nil {
		t.Fatal(err)
	}
	before := state
	t.Cleanup(func() {
		if err := f.ex.desired.save(before); err != nil {
			t.Errorf("restore fixture switch checkpoint: %v", err)
		}
	})
	state.PendingSwitch = &settings.SwitchCheckpoint{SwitchPrepareRequest: p, Phase: settings.SwitchCheckpointReserved}
	if err := f.ex.desired.save(state); err != nil {
		t.Fatal(err)
	}
	if got := f.ex.residentCheck(context.Background(), f.p); got != "switch-busy" {
		t.Fatalf("reason = %s, want switch-busy", got)
	}
	if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched {
		t.Fatalf("result = %+v", result)
	}
	f.noCalls(t)
}
