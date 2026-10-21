// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type loadedLlamaFake struct {
	mu      sync.Mutex
	loaded  string
	props   string
	status  int
	onProps func()
	gets    atomic.Int32
	loads   atomic.Int32
	models  atomic.Int32
	model   string
}

func TestLoadedServingInvalidationConcurrentAdoptedOperations(t *testing.T) {
	test := func(name string, operation func(*testing.T, *Executor)) {
		t.Run(name, func(t *testing.T) {
			ex := settingsExecutor(t, false)
			st := settingsState(t, ex)
			st.mu.Lock()
			st.running, st.adopted = true, true
			st.mu.Unlock()
			defer func() {
				st.mu.Lock()
				st.running, st.adopted = false, false
				st.mu.Unlock()
			}()

			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				for range 10000 {
					ex.pokeLoaded()
				}
			}()
			defer wg.Wait()
			for range 1000 {
				operation(t, ex)
			}
		})
	}
	test("adopted settings read", func(t *testing.T, ex *Executor) {
		t.Helper()
		state, err := ex.LaunchSettings("fake")
		if err != nil {
			t.Fatal(err)
		}
		if state.Editable || !state.Running || !state.Adopted {
			t.Fatalf("adopted launch state = %+v, want running, adopted and uneditable", state)
		}
	})
	test("refused adopted port change", func(t *testing.T, ex *Executor) {
		t.Helper()
		if _, err := ex.SetPort(context.Background(), "fake", 21093); err == nil {
			t.Fatal("adopted engine accepted port mutation")
		}
	})
}

func (s *loadedLlamaFake) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	loaded, props, status, hook := s.loaded, s.props, s.status, s.onProps
	s.mu.Unlock()
	if r.Method != http.MethodGet {
		s.loads.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	switch r.URL.Path {
	case "/models":
		s.models.Add(1)
		fmt.Fprint(w, loaded)
	case "/props":
		s.gets.Add(1)
		// Simulate llama's autoload default: an unsafe query records a load.
		if r.URL.Query().Get("autoload") != "false" || r.URL.Query().Get("model") != s.model {
			s.loads.Add(1)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if hook != nil {
			hook()
		}
		if status == http.StatusFound {
			w.Header().Set("Location", "/models/load")
		}
		w.WriteHeader(status)
		fmt.Fprint(w, props)
	default:
		s.loads.Add(1)
		w.WriteHeader(http.StatusNotFound)
	}
}

func loadedReadbackFixture(t *testing.T, model string) (llamaServingFixture, *loadedLlamaFake) {
	t.Helper()
	f := newLlamaServingFixture(t)
	if model != "medium" {
		data, err := os.ReadFile(f.source)
		if err != nil {
			t.Fatal(err)
		}
		writeServingFixture(t, f.source, []byte(strings.Replace(string(data), "[medium]", "["+model+"]", 1)))
		f.request.Settings.Serving.Models[0].ModelID = model
	}
	applyLlamaServing(t, f, approveLlamaServing(t, f))
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.ex.Stop("fake"); err != nil {
			t.Error(err)
		}
	})
	return f, attachLoadedLlamaFake(t, f, model)
}

func attachLoadedLlamaFake(t *testing.T, f llamaServingFixture, model string) *loadedLlamaFake {
	t.Helper()
	row, err := json.Marshal(map[string]any{"data": []any{map[string]any{"id": model, "status": map[string]string{"value": "loaded"}}}})
	if err != nil {
		t.Fatal(err)
	}
	s := &loadedLlamaFake{loaded: string(row), props: `{"default_generation_settings":{"n_ctx":8192},"total_slots":2}`, status: http.StatusOK, model: model}
	server := httptest.NewServer(s)
	t.Cleanup(server.Close)
	_, portText, err := net.SplitHostPort(server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}
	reg := loadWithOverrides(t, t.TempDir())
	m, ok := reg.Get("llamacpp")
	if !ok {
		t.Fatal("missing llama manifest")
	}
	st := settingsState(t, f.ex)
	st.mu.Lock()
	st.port = port
	st.manifest.Actions = m.Actions
	st.mu.Unlock()
	return s
}

func readbackProof(t *testing.T, f llamaServingFixture, count int) {
	t.Helper()
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	if len(state.ServingReadback.Observed) != count {
		t.Fatalf("readback: %+v", state.ServingReadback)
	}
}

func TestLoadedServingEncodedNameNeverAutoloads(t *testing.T) {
	const model = "org/model:Q4 + &?雪"
	f, server := loadedReadbackFixture(t, model)
	f.ex.sweepLoaded(context.Background(), nil, true)
	value, err := f.ex.currentServingObservation(context.Background(), "fake", model)
	if err != nil || value.PerSlotContextTokens != 8192 || value.Slots != 2 {
		t.Fatalf("per-slot observation: %+v %v", value, err)
	}
	if server.gets.Load() != 1 || server.loads.Load() != 0 {
		t.Fatalf("GETs=%d loads=%d", server.gets.Load(), server.loads.Load())
	}
	readbackProof(t, f, 1)
	state, err := f.ex.LaunchSettings("fake")
	if err != nil || state.ServingReadback.Status != "observed" {
		t.Fatalf("state=%+v err=%v", state.ServingReadback, err)
	}
	if len(f.ex.modelIdentity().Engines["fake"].Serving) != 2 {
		t.Fatal("matching observation withdrew identity")
	}
}

func TestLoadedServingColdAndUnloadClearProof(t *testing.T) {
	f, server := loadedReadbackFixture(t, "medium")
	server.mu.Lock()
	server.loaded = `{"data":[]}`
	server.mu.Unlock()
	f.ex.sweepLoaded(context.Background(), nil, true)
	readbackProof(t, f, 0)
	if server.gets.Load() != 0 || server.loads.Load() != 0 {
		t.Fatal("cold model was probed or loaded")
	}
	if len(f.ex.modelIdentity().Engines["fake"].Serving) != 2 {
		t.Fatal("cold model lost identity")
	}
	server.mu.Lock()
	server.loaded = `{"data":[{"id":"medium","status":{"value":"loaded"}}]}`
	server.mu.Unlock()
	f.ex.sweepLoaded(context.Background(), nil, false)
	readbackProof(t, f, 1)
	server.mu.Lock()
	server.loaded = `{"data":[{"id":"medium","status":{"value":"unloaded"}}]}`
	server.mu.Unlock()
	f.ex.sweepLoaded(context.Background(), nil, false)
	readbackProof(t, f, 0)
	if _, err := f.ex.currentServingObservation(context.Background(), "fake", "medium"); err == nil {
		t.Fatal("unloaded model retained proof")
	}
	if server.gets.Load() != 1 || server.loads.Load() != 0 {
		t.Fatal("unloaded model was probed or loaded")
	}
}

func TestLoadedServingSleepingAndTransitionalModelsAreNotProbed(t *testing.T) {
	for _, status := range []string{"unloaded", "loading", "sleeping", "downloading", "downloaded"} {
		t.Run(status, func(t *testing.T) {
			f, server := loadedReadbackFixture(t, "medium")
			server.mu.Lock()
			server.loaded = fmt.Sprintf(`{"data":[{"id":"medium","status":{"value":%q}}]}`, status)
			server.mu.Unlock()
			f.ex.sweepLoaded(context.Background(), nil, false)
			readbackProof(t, f, 0)
			if server.gets.Load() != 0 || server.loads.Load() != 0 {
				t.Fatal("non-loaded model was probed or woken")
			}
			if len(f.ex.modelIdentity().Engines["fake"].Serving) != 2 {
				t.Fatal("known non-loaded status withdrew identity")
			}
		})
	}
}

func TestLoadedServingMissingEndpointWithdrawsProof(t *testing.T) {
	f, _ := loadedReadbackFixture(t, "medium")
	f.ex.sweepLoaded(context.Background(), nil, false)
	readbackProof(t, f, 1)
	settingsState(t, f.ex).manifest.Actions = nil
	f.ex.sweepLoaded(context.Background(), nil, false)
	readbackProof(t, f, 0)
	if len(f.ex.modelIdentity().Engines["fake"].ModelDigests) != 0 {
		t.Fatal("missing loaded endpoint retained identity")
	}
}

func TestLoadedServingAmbiguousLoadedMappingCannotWake(t *testing.T) {
	f, server := loadedReadbackFixture(t, "medium")
	server.mu.Lock()
	server.loaded = `{"data":[{"id":"medium","status":{"value":"sleeping"}}]}`
	server.mu.Unlock()
	st := settingsState(t, f.ex)
	action := st.manifest.Actions["loaded_models"]
	result := *action.Result
	result.Match = nil
	action.Result = &result
	st.manifest.Actions["loaded_models"] = action
	f.ex.sweepLoaded(context.Background(), nil, false)
	if server.gets.Load() != 0 || server.loads.Load() != 0 {
		t.Fatal("ambiguous mapping probed sleeping model")
	}
	if len(f.ex.modelIdentity().Engines["fake"].ModelDigests) != 0 {
		t.Fatal("ambiguous residency retained identity")
	}
}

func TestLoadedServingInvalidAllocationWithdrawsIdentity(t *testing.T) {
	tests := []struct {
		name, props, loaded string
	}{
		{name: "context mismatch", props: `{"default_generation_settings":{"n_ctx":4096},"total_slots":2}`},
		{name: "slot mismatch", props: `{"default_generation_settings":{"n_ctx":8192},"total_slots":1}`},
		{name: "malformed props", props: `{`},
		{name: "unknown props", props: `{}`},
		{name: "router placeholder", props: `{"default_generation_settings":{"n_ctx":0},"total_slots":2}`},
		{name: "fractional context", props: `{"default_generation_settings":{"n_ctx":8192.5},"total_slots":2}`},
		{name: "missing residency", loaded: `{"data":[{"id":"medium"}]}`},
		{name: "unknown residency", loaded: `{"data":[{"id":"medium","status":{"value":"future"}}]}`},
		{name: "invalid loaded result", loaded: `{"data":null}`},
		{name: "undeclared loaded model", loaded: `{"data":[{"id":"unknown","status":{"value":"loaded"}}]}`},
		{name: "malformed loaded JSON", loaded: `{`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f, server := loadedReadbackFixture(t, "medium")
			f.ex.sweepLoaded(context.Background(), nil, true)
			readbackProof(t, f, 1)
			server.mu.Lock()
			if tc.props != "" {
				server.props = tc.props
			}
			if tc.loaded != "" {
				server.loaded = tc.loaded
			}
			server.mu.Unlock()
			f.ex.sweepLoaded(context.Background(), nil, false)
			readbackProof(t, f, 0)
			id := f.ex.modelIdentity().Engines["fake"]
			if len(id.ModelDigests) != 0 || len(id.Serving) != 0 {
				t.Fatalf("invalid observation retained identity: %+v", id)
			}
			state, err := f.ex.LaunchSettings("fake")
			if err != nil || state.ServingReadback.Status != "mismatch" {
				t.Fatalf("state=%+v err=%v", state.ServingReadback, err)
			}
			if _, err := f.ex.currentServingObservation(context.Background(), "fake", "medium"); err == nil {
				t.Fatal("invalid observation returned proof")
			}
			if server.loads.Load() != 0 {
				t.Fatal("observation triggered load")
			}
			server.mu.Lock()
			server.loaded = `{"data":[{"id":"medium","status":{"value":"loaded"}}]}`
			server.props = `{"default_generation_settings":{"n_ctx":8192},"total_slots":2}`
			server.mu.Unlock()
			f.ex.sweepLoaded(context.Background(), nil, false)
			readbackProof(t, f, 0)
			st := settingsState(t, f.ex)
			st.mu.Lock()
			drift := st.plat.Runtime.servingDrift
			st.mu.Unlock()
			if drift == "" {
				t.Fatal("successful sweep cleared semantic drift")
			}
			// The readback fake uses a separate listener; restore the owned
			// process port before exercising lifecycle restart.
			st.mu.Lock()
			st.port = st.plat.Runtime.Port
			st.mu.Unlock()
			if err := f.ex.Stop("fake"); err != nil {
				t.Fatal(err)
			}
			if err := f.ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			attachLoadedLlamaFake(t, f, "medium")
			f.ex.sweepLoaded(context.Background(), nil, false)
			readbackProof(t, f, 1)
		})
	}
}

// Return valid bytes together with a read failure: even parseable partial
// responses must never become observation evidence.
type readbackBrokenBody struct {
	io.Reader
}

func (b readbackBrokenBody) Read(p []byte) (int, error) {
	n, _ := b.Reader.Read(p)
	return n, io.ErrUnexpectedEOF
}

func (readbackBrokenBody) Close() error { return nil }

func TestLoadedServingTransportFailureRecoversNextSweep(t *testing.T) {
	for _, endpoint := range []string{"/models", "/props"} {
		t.Run(endpoint, func(t *testing.T) {
			for _, failure := range []string{"connection", "500", "timeout", "sweep timeout", "truncated body", "404", "redirect", "201 valid body", "204 empty body"} {
				t.Run(failure, func(t *testing.T) {
					f, server := loadedReadbackFixture(t, "medium")
					f.ex.sweepLoaded(context.Background(), nil, false)
					readbackProof(t, f, 1)
					original := f.ex.client.Transport
					f.ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
						if r.URL.Path != endpoint {
							return original.RoundTrip(r)
						}
						switch failure {
						case "connection":
							return nil, errors.New("connection lost")
						case "timeout", "sweep timeout":
							<-r.Context().Done()
							return nil, r.Context().Err()
						case "truncated body":
							body := `{"data":[{"id":"medium","status":{"value":"loaded"}}]}`
							if endpoint == "/props" {
								body = `{"default_generation_settings":{"n_ctx":8192},"total_slots":2}`
							}
							return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: readbackBrokenBody{strings.NewReader(body)}}, nil
						case "201 valid body":
							body := `{"data":[{"id":"medium","status":{"value":"loaded"}}]}`
							if endpoint == "/props" {
								body = `{"default_generation_settings":{"n_ctx":8192},"total_slots":2}`
							}
							return &http.Response{StatusCode: http.StatusCreated, Body: io.NopCloser(strings.NewReader(body))}, nil
						case "204 empty body":
							return &http.Response{StatusCode: http.StatusNoContent, Body: http.NoBody}, nil
						default:
							status := http.StatusInternalServerError
							if failure == "404" {
								status = http.StatusNotFound
							} else if failure == "redirect" {
								status = http.StatusFound
							}
							header := make(http.Header)
							if failure == "redirect" && endpoint == "/props" {
								header.Set("Location", "/models/load")
							}
							return &http.Response{StatusCode: status, Header: header, Body: io.NopCloser(strings.NewReader("unavailable"))}, nil
						}
					})
					ctx := context.Background()
					if failure == "timeout" {
						var cancel context.CancelFunc
						ctx, cancel = context.WithTimeout(ctx, 100*time.Millisecond)
						defer cancel()
					}
					f.ex.sweepLoaded(ctx, nil, false)
					readbackProof(t, f, 0)
					if _, err := f.ex.currentServingObservation(context.Background(), "fake", "medium"); err == nil {
						t.Fatal("transport failure retained current observation")
					}
					st := settingsState(t, f.ex)
					st.mu.Lock()
					drift := st.plat.Runtime.servingDrift
					st.mu.Unlock()
					if drift != "" {
						t.Fatalf("transport failure latched drift: %s", drift)
					}
					if server.loads.Load() != 0 {
						t.Fatal("failed observation triggered a load")
					}
					f.ex.client.Transport = original
					f.ex.sweepLoaded(context.Background(), nil, false)
					readbackProof(t, f, 1)
					if _, err := f.ex.currentServingObservation(context.Background(), "fake", "medium"); err != nil {
						t.Fatalf("successful sweep did not restore observation: %v", err)
					}
				})
			}
		})
	}
}

func TestLoadedServingRedirectIsTransportUnknown(t *testing.T) {
	for _, endpoint := range []string{"/models", "/props"} {
		t.Run(endpoint, func(t *testing.T) {
			for _, status := range []int{http.StatusMovedPermanently, http.StatusFound, http.StatusTemporaryRedirect} {
				t.Run(http.StatusText(status), func(t *testing.T) {
					f, server := loadedReadbackFixture(t, "medium")
					f.ex.sweepLoaded(context.Background(), nil, false)
					readbackProof(t, f, 1)
					original := f.ex.client.Transport
					redirects := 0
					f.ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
						if r.URL.Path == "/redirected" {
							redirects++
							// A followed redirect would return valid proof, exposing
							// clients that check only the final response's status.
							forward := r.Clone(r.Context())
							forward.URL.Path = endpoint
							return original.RoundTrip(forward)
						}
						if r.URL.Path == endpoint {
							return &http.Response{StatusCode: status, Header: http.Header{"Location": {"/redirected"}}, Body: http.NoBody}, nil
						}
						return original.RoundTrip(r)
					})
					f.ex.sweepLoaded(context.Background(), nil, false)
					st := settingsState(t, f.ex)
					st.mu.Lock()
					drift := st.plat.Runtime.servingDrift
					st.mu.Unlock()
					if redirects != 0 || drift != "" {
						t.Errorf("redirect requests=%d drift=%q, want no redirect or drift latch", redirects, drift)
					}
					readbackProof(t, f, 0)
					if _, err := f.ex.currentServingObservation(context.Background(), "fake", "medium"); err == nil {
						t.Fatal("redirect retained current observation")
					}
					state, err := f.ex.LaunchSettings("fake")
					if err != nil || state.ServingReadback.Status != "unverified" {
						t.Fatalf("readback=%+v err=%v, want unverified", state.ServingReadback, err)
					}
					f.ex.client.Transport = original
					f.ex.sweepLoaded(context.Background(), nil, false)
					readbackProof(t, f, 1)
					if _, err := f.ex.currentServingObservation(context.Background(), "fake", "medium"); err != nil {
						t.Fatalf("next good sweep did not restore observation: %v", err)
					}
					if server.loads.Load() != 0 {
						t.Fatal("observation triggered a load")
					}
				})
			}
		})
	}
}

func TestLoadedServingSemanticEvidenceSurvivesTransportFailure(t *testing.T) {
	test := func(name, semanticModel, props string, semanticFirst bool) {
		t.Run(name, func(t *testing.T) {
			f, server := loadedReadbackFixture(t, "medium")
			f.ex.sweepLoaded(context.Background(), nil, false)
			readbackProof(t, f, 1)
			models := []string{"medium", semanticModel}
			if semanticFirst {
				models[0], models[1] = models[1], models[0]
			}
			rows := make([]map[string]any, 0, len(models))
			for _, model := range models {
				rows = append(rows, map[string]any{"id": model, "status": map[string]string{"value": "loaded"}})
			}
			raw, err := json.Marshal(map[string]any{"data": rows})
			if err != nil {
				t.Fatal(err)
			}
			server.mu.Lock()
			server.loaded = string(raw)
			server.mu.Unlock()
			original := f.ex.client.Transport
			f.ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
				if r.URL.Path == "/props" {
					if r.URL.Query().Get("model") == "medium" {
						return nil, errors.New("connection lost")
					}
					if r.URL.Query().Get("autoload") != "false" {
						t.Error("props request did not disable autoload")
					}
					return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(props))}, nil
				}
				return original.RoundTrip(r)
			})
			f.ex.sweepLoaded(context.Background(), nil, false)
			readbackProof(t, f, 0)
			st := settingsState(t, f.ex)
			st.mu.Lock()
			drift := st.plat.Runtime.servingDrift
			st.mu.Unlock()
			if drift == "" {
				t.Fatal("transport failure masked semantic disagreement")
			}
			f.ex.client.Transport = original
			server.mu.Lock()
			server.loaded = `{"data":[{"id":"medium","status":{"value":"loaded"}}]}`
			server.mu.Unlock()
			f.ex.sweepLoaded(context.Background(), nil, false)
			readbackProof(t, f, 0)
			if _, err := f.ex.currentServingObservation(context.Background(), "fake", "medium"); err == nil {
				t.Fatal("good sweep cleared semantic latch")
			}
			if server.loads.Load() != 0 {
				t.Fatal("observation triggered a load")
			}
			st.mu.Lock()
			st.port = st.plat.Runtime.Port
			st.mu.Unlock()
			if err := f.ex.Stop("fake"); err != nil {
				t.Fatal(err)
			}
			if err := f.ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			attachLoadedLlamaFake(t, f, "medium")
			f.ex.sweepLoaded(context.Background(), nil, false)
			readbackProof(t, f, 1)
		})
	}
	test("undeclared last", "unknown", "", false)
	test("undeclared first", "unknown", "", true)
	test("allocation mismatch last", "weak", `{"default_generation_settings":{"n_ctx":1},"total_slots":1}`, false)
	test("allocation mismatch first", "weak", `{"default_generation_settings":{"n_ctx":1},"total_slots":1}`, true)
	test("invalid props last", "weak", `{`, false)
	test("invalid props first", "weak", `{`, true)
}

func TestLoadedServingFencesReplacementDuringHTTP(t *testing.T) {
	tests := []struct {
		name    string
		replace func(*engineState)
	}{
		{"run", func(st *engineState) { st.runID = newEngineRunID() }},
		{"config", func(st *engineState) { st.plat.Runtime.Serving = st.plat.Runtime.Serving.Clone() }},
		{"process", func(st *engineState) { st.running = false }},
		{"source withdrawal", func(st *engineState) { st.plat.Runtime.servingDrift = "source withdrawn" }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var proc *managedProc
			t.Cleanup(func() {
				if proc == nil {
					return
				}
				select {
				case <-proc.done:
				default:
					t.Error("fake engine was not reaped by fixture cleanup")
				}
			})
			f, server := loadedReadbackFixture(t, "medium")
			st := settingsState(t, f.ex)
			if tc.name == "process" {
				proc = st.proc
				// This case changes the observed flag without stopping its fake
				// process. Restore the fixture's real state before Stop cleanup.
				t.Cleanup(func() {
					st.mu.Lock()
					st.running = true
					st.mu.Unlock()
				})
			}
			server.mu.Lock()
			server.onProps = func() { st.mu.Lock(); defer st.mu.Unlock(); tc.replace(st) }
			server.mu.Unlock()
			f.ex.sweepLoaded(context.Background(), nil, false)
			if _, err := f.ex.currentServingObservation(context.Background(), "fake", "medium"); err == nil {
				t.Fatal("replacement inherited stale observation")
			}
			readbackProof(t, f, 0)
		})
	}
}

func TestLoadedServingSourceChangeWithdrawsProof(t *testing.T) {
	f, _ := loadedReadbackFixture(t, "medium")
	f.ex.sweepLoaded(context.Background(), nil, false)
	readbackProof(t, f, 1)
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	writeServingFixture(t, f.source, append(data, []byte("\n# changed source\n")...))
	f.ex.sweepLoaded(context.Background(), nil, false)
	readbackProof(t, f, 0)
	if len(f.ex.modelIdentity().Engines["fake"].ModelDigests) != 0 {
		t.Fatal("changed source retained identity")
	}
}

func TestLoadedServingStaleMismatchCannotWithdrawReplacement(t *testing.T) {
	f, server := loadedReadbackFixture(t, "medium")
	st := settingsState(t, f.ex)
	server.mu.Lock()
	server.props = `{"default_generation_settings":{"n_ctx":4096},"total_slots":2}`
	server.onProps = func() { st.mu.Lock(); st.plat.Runtime.Serving = st.plat.Runtime.Serving.Clone(); st.mu.Unlock() }
	server.mu.Unlock()
	f.ex.sweepLoaded(context.Background(), nil, false)
	if len(f.ex.modelIdentity().Engines["fake"].Serving) != 2 {
		t.Fatal("stale mismatch withdrew replacement identity")
	}
	readbackProof(t, f, 0)
}

func TestLoadedServingCancellationBoundsHTTP(t *testing.T) {
	f, server := loadedReadbackFixture(t, "medium")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	server.mu.Lock()
	server.onProps = cancel
	server.mu.Unlock()
	f.ex.sweepLoaded(ctx, nil, false)
	readbackProof(t, f, 0)
	if server.loads.Load() != 0 {
		t.Fatal("cancelled observation loaded model")
	}
}

func TestLoadedServingActionAndExpiryInvalidateProof(t *testing.T) {
	f, _ := loadedReadbackFixture(t, "medium")
	f.ex.sweepLoaded(context.Background(), nil, false)
	f.ex.pokeLoaded()
	readbackProof(t, f, 0)
	f.ex.sweepLoaded(context.Background(), nil, false)
	readbackProof(t, f, 1)
	st := settingsState(t, f.ex)
	st.mu.Lock()
	st.plat.Runtime.servingObserved.at = time.Now().Add(-time.Hour)
	st.mu.Unlock()
	readbackProof(t, f, 0)
}

func TestLoadedServingOlderSweepCannotRestoreProof(t *testing.T) {
	f, server := loadedReadbackFixture(t, "medium")
	started, release := make(chan struct{}), make(chan struct{})
	server.mu.Lock()
	server.onProps = func() { close(started); <-release }
	server.mu.Unlock()
	done := make(chan struct{})
	go func() { defer close(done); f.ex.sweepLoaded(context.Background(), nil, false) }()
	<-started
	server.mu.Lock()
	server.onProps = nil
	server.loaded = `{"data":[]}`
	server.mu.Unlock()
	f.ex.sweepLoaded(context.Background(), nil, false)
	close(release)
	<-done
	readbackProof(t, f, 0)
}

// Runtime observation updates use mu independently of opMu. Lifecycle copies
// must synchronize with them even when no serving declaration is configured.
func TestLoadedServingInvalidationConcurrentLifecycle(t *testing.T) {
	for _, operation := range []string{"start", "stop", "failed stop reconciliation"} {
		t.Run(operation, func(t *testing.T) {
			manifest := testEngineManifest(fakeEngineBin)
			for platform, value := range manifest.Platforms {
				value.Runtime.Health = nil
				manifest.Platforms[platform] = value
			}
			ex := newTestExecutor(t, manifest)
			st := settingsState(t, ex)
			stop, done, started := make(chan struct{}), make(chan struct{}), make(chan struct{})
			go func() {
				defer close(done)
				close(started)
				for {
					select {
					case <-stop:
						return
					default:
						ex.pokeLoaded()
					}
				}
			}()
			<-started
			defer func() { close(stop); <-done }()
			switch operation {
			case "start":
				t.Cleanup(ex.StopAll)
				for range 8 {
					if err := ex.Start(context.Background(), "fake"); err != nil {
						t.Fatal(err)
					}
					if err := ex.Stop("fake"); err != nil {
						t.Fatal(err)
					}
				}
			case "stop":
				for range 128 {
					if err := ex.Stop("fake"); err != nil {
						t.Fatal(err)
					}
				}
			case "failed stop reconciliation":
				stopErr := fmt.Errorf("injected fake stop refusal")
				st.opMu.Lock()
				defer st.opMu.Unlock()
				for range 128 {
					if err := ex.reconcileFailedCommandStop(st, "fake", true, stopErr); err != stopErr {
						t.Fatalf("stop refusal = %v, want original refusal", err)
					}
				}
			}
		})
	}
}
