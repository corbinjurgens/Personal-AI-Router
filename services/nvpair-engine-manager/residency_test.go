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

type residentTransport struct {
	mu                      sync.Mutex
	loads, forbidden, reads int
	body                    map[string]json.RawMessage
	loaded                  string
	loadErr                 error
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
	case "/api/generate":
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
	case "/api/ps":
		r.reads++
		return residentJSON(r.loaded), nil
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
	mf := testEngineManifest("forbidden-resident-engine")
	mf.Engine, mf.Family, mf.ModelNaming = "ollama", "ollama", engines.ModelNamingImpliedLatestTag
	mf.Operations = map[string]OperationBinding{engines.OperationLoad: {Action: "resident_load", Params: map[string]json.RawMessage{"stream": json.RawMessage(`false`)}}}
	mf.Actions = map[string]Action{
		"resident_load": {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/api/generate"}},
		"list_models":   {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/api/tags"}, Result: &ActionResult{Array: "models", Field: "name", Digest: "digest"}},
		"loaded_models": {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/api/ps"}, Result: &ActionResult{Array: "models", Field: "name"}},
		"pull_model":    {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/forbidden/pull"}},
		"delete_model":  {HTTP: &ActionHTTP{Method: http.MethodDelete, Path: "/forbidden/delete"}},
		"arbitrary":     {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/forbidden/action"}},
	}
	ex := newTestExecutor(t, mf)
	st, err := ex.state("ollama")
	if err != nil {
		t.Fatal(err)
	}
	st.running, st.installed, st.proc, st.runID, st.launchDigest = true, true, &managedProc{}, "resident-run", nameDigest("launch")
	ex.nodeProjection.set(nodeProjection{})
	if err := ex.desired.set("ollama", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	ex.storeInventory(map[string][]string{"ollama": {"m:latest"}}, map[string]observedDigests{"ollama": {runID: st.runID, models: map[string]string{"m:latest": nameDigest("model")}}}, 1)
	ex.sweeps.Store(1)
	ex.settingsHub.Publish([]settings.Snapshot{})
	f := &residentFixture{ex: ex, transport: &residentTransport{loaded: `{"models":[{"name":"m:latest"}]}`}, responses: make(chan Message, 8)}
	ex.client = &http.Client{Transport: f.transport}
	ex.modelMutationParent = func(context.Context, string) error { f.contentCalls++; return errors.New("forbidden content mutation") }
	f.p = nodepolicy.ReconcileResidentParams{OperationID: "resident-op", PolicyRevision: "1", Engine: "ollama", Model: "m",
		Identity:  nodepolicy.StrictModelIdentity{EngineRunID: st.runID, SettingsFingerprint: nameDigest("settings-and-launch"), ModelDigest: nameDigest("model")},
		TimeoutMs: 60000, Reservation: nodepolicy.ResidentReservationEvidence{ReservationID: "reservation-1", ProxyRunID: "proxy-1", ManagerIncarnation: ex.switchIncarnation},
		Load: nodepolicy.ResidentLoadOptions{ContextLength: 4096, KeepAliveSeconds: 120}}
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
		st := f.ex.engines["ollama"]
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
	want := nodepolicy.ResidentLoadedObservation{ManagerIncarnation: f.ex.switchIncarnation, Generation: 1, Engine: "ollama", Model: "m:latest", Identity: f.p.Identity, Loaded: true}
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
	if body.Model != "m:latest" || body.KeepAlive != 120 || body.Options.Context != 4096 || body.Stream || len(f.transport.body) != 4 {
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
			if got := f.ex.residentCheck(f.p); got != reason {
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
	test("missing installed model", "not-installed", func(f *residentFixture) { f.ex.inventory["ollama"] = []string{} })
	test("unknown installed inventory", "not-installed", func(f *residentFixture) { delete(f.ex.inventory, "ollama") })
	test("non Ollama", "unsupported-load", func(f *residentFixture) { f.ex.reg.engines["ollama"].Family = "lmstudio" })
	test("implicit binding", "unsupported-load", func(f *residentFixture) { delete(f.ex.reg.engines["ollama"].Operations, engines.OperationLoad) })
	test("restart binding", "unsupported-load", func(f *residentFixture) {
		mf := f.ex.reg.engines["ollama"]
		a := mf.Actions["resident_load"]
		a.RestartAfter = true
		mf.Actions["resident_load"] = a
	})
	test("prompt bearing fixed options", "unsupported-load", func(f *residentFixture) {
		f.ex.reg.engines["ollama"].Operations[engines.OperationLoad].Params["prompt"] = json.RawMessage(`"forbidden"`)
	})
	test("unexpressible fixed context", "unsupported-load", func(f *residentFixture) {
		f.ex.reg.engines["ollama"].Operations[engines.OperationLoad].Params["options"] = json.RawMessage(`{"num_ctx":8}`)
	})
	test("wrong load endpoint", "unsupported-load", func(f *residentFixture) {
		f.ex.reg.engines["ollama"].Actions["resident_load"].HTTP.Path = "/forbidden/action"
	})
	test("new manager incarnation", "unqualified", func(f *residentFixture) { f.p.Reservation.ManagerIncarnation = "old-manager" })
	test("omitted manager incarnation", "unqualified", func(f *residentFixture) { f.p.Reservation.ManagerIncarnation = "" })
	test("stale run", "unqualified", func(f *residentFixture) { f.p.Identity.EngineRunID = "old-run" })
	test("missing launch identity", "unqualified", func(f *residentFixture) { f.ex.engines["ollama"].launchDigest = "" })
	test("same digest under another model", "unqualified", func(f *residentFixture) { f.p.Model = "n"; f.ex.inventory["ollama"] = []string{"m:latest", "n:latest"} })
	test("different model digest", "unqualified", func(f *residentFixture) { f.p.Identity.ModelDigest = nameDigest("other") })
	test("withdrawn identity", "unqualified", func(f *residentFixture) {
		v := f.ex.inventoryDigests["ollama"]
		v.suppressed = true
		f.ex.inventoryDigests["ollama"] = v
	})
	test("engine stopped", "engine-off", func(f *residentFixture) { f.ex.engines["ollama"].running = false })
	test("peer Off", "engine-off", func(f *residentFixture) {
		if err := f.ex.desired.set("ollama", false, intentPeer); err != nil {
			t.Fatal(err)
		}
	})
	test("local Off", "engine-off-local", func(f *residentFixture) {
		if err := f.ex.desired.set("ollama", false, intentLocal); err != nil {
			t.Fatal(err)
		}
	})
	test("legacy Off", "engine-off-local", func(f *residentFixture) {
		if err := f.ex.desired.save(desiredStateFile{Engines: map[string]bool{"ollama": false}}); err != nil {
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
		count := ex.switchMutations["ollama"]
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
			st := f.ex.engines["ollama"]
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
		if err := f.ex.desired.set("ollama", false, intentLocal); err != nil {
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
		f.ex.inventory["ollama"] = nil
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
	test("lost response but loaded", `{"models":[{"name":"m"}]}`, io.ErrUnexpectedEOF, nodepolicy.ResidentLoaded, &yes)
	test("missing inventory", `{}`, nil, nodepolicy.ResidentUnknown, nil)
	test("null inventory", `{"models":null}`, nil, nodepolicy.ResidentUnknown, nil)
	test("explicit empty", `{"models":[]}`, nil, nodepolicy.ResidentFailed, &no)
	test("failed load empty", `{"models":[]}`, io.ErrUnexpectedEOF, nodepolicy.ResidentFailed, &no)
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
		f.ex.inventoryMu.Lock()
		v := f.ex.inventoryDigests["ollama"]
		v.suppressed = true
		v.epoch++
		f.ex.inventoryDigests["ollama"] = v
		f.ex.inventoryMu.Unlock()
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
	st := f.ex.engines["ollama"]
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
	f.ex.storeInventory(map[string][]string{"ollama": {}}, nil, 3)
	f.ex.storeInventory(map[string][]string{"ollama": {"m:latest"}}, map[string]observedDigests{"ollama": {runID: f.p.Identity.EngineRunID, models: map[string]string{"m:latest": f.p.Identity.ModelDigest}}}, 2)
	if got := f.ex.cachedInventory()["ollama"]; len(got) != 0 {
		t.Fatalf("old sweep restored installed inventory: %v", got)
	}
	if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched {
		t.Fatalf("result = %+v", result)
	}
	f.noCalls(t)
}

func TestResidentNoContentFallback(t *testing.T) {
	f := newResidentFixture(t)
	f.ex.inventory["ollama"] = []string{}
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
		return strings.Replace(s, `{"contextLength":4096,"keepAliveSeconds":120}`, `null`, 1)
	})
	test("zero context", func(s string) string { return strings.Replace(s, `"contextLength":4096`, `"contextLength":0`, 1) })
	test("unbounded keep alive", func(s string) string {
		return strings.Replace(s, `"keepAliveSeconds":120`, `"keepAliveSeconds":86401`, 1)
	})
}

func TestResidentSwitchBusy(t *testing.T) {
	f := newResidentFixture(t)
	_, p := switchFixture(t)
	p.SourceEngine, p.TargetEngine, p.ManagerIncarnation = "ollama", "other", f.ex.switchIncarnation
	p.ConfigurationFence = "e3v1:" + nameDigest("configuration")
	state, err := f.ex.desired.load()
	if err != nil {
		t.Fatal(err)
	}
	state.PendingSwitch = &settings.SwitchCheckpoint{SwitchPrepareRequest: p, Phase: settings.SwitchCheckpointReserved}
	if err := f.ex.desired.save(state); err != nil {
		t.Fatal(err)
	}
	if got := f.ex.residentCheck(f.p); got != "switch-busy" {
		t.Fatalf("reason = %s, want switch-busy", got)
	}
	if result := f.run(t); result.Outcome != nodepolicy.ResidentNotDispatched {
		t.Fatalf("result = %+v", result)
	}
	f.noCalls(t)
}
