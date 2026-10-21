// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"time"

	"nvpair-shared/engines"
)

type servingObserved struct {
	run    servingRunSnapshot
	values []engines.ServingObservation
	at     time.Time
}

// Capture before loaded_models, so a response from an old process/configuration
// cannot become evidence for its replacement. Caller performs no I/O under mu.
func snapshotLlamaObservation(st *engineState) (servingRunSnapshot, error) {
	s, err := snapshotServingRun(st)
	if err != nil {
		return s, err
	}
	if !llamaServingRuntime(st, s.rt.Serving) {
		return servingRunSnapshot{}, &ServingIdentityRefusal{"Loaded observation is unavailable for this engine."}
	}
	s.rt.servingObserved = nil
	return s, nil
}

// The pinned llama router exposes explicit child status. Missing or unfamiliar
// status is unknown residency, never evidence that all children are cold.
func llamaLoadedSemantics(raw json.RawMessage, loaded []string) bool {
	var result struct {
		Data *[]struct {
			ID     string `json:"id"`
			Status struct {
				Value string `json:"value"`
			} `json:"status"`
		} `json:"data"`
	}
	if json.Unmarshal(raw, &result) != nil || result.Data == nil {
		return false
	}
	seen := make(map[string]bool)
	var names []string
	for _, row := range *result.Data {
		if row.ID == "" || seen[row.ID] {
			return false
		}
		seen[row.ID] = true
		switch row.Status.Value {
		case "loaded":
			names = append(names, row.ID)
		case "unloaded", "loading", "sleeping", "downloading", "downloaded":
		default:
			return false
		}
	}
	return sameStringSet(names, loaded)
}

func (e *Executor) llamaServingProps(ctx context.Context, port int, model string) (engines.ServingObservation, error) {
	query := url.Values{"model": {model}, "autoload": {"false"}}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/props?%s", port, query.Encode()), nil)
	if err != nil {
		return engines.ServingObservation{}, err
	}
	req.Header.Set(engineIdentityProbeHeader, "1")
	// A redirected request can lose autoload=false. Never follow it.
	client := *e.clientFor(0)
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	resp, err := client.Do(req)
	if err != nil {
		return engines.ServingObservation{}, err
	}
	defer resp.Body.Close()
	const limit = 8 * 1024 * 1024
	data, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil || resp.StatusCode != http.StatusOK {
		return engines.ServingObservation{}, fmt.Errorf("loaded allocation read failed")
	}
	if len(data) > limit {
		return engines.ServingObservation{}, &ServingIdentityRefusal{"Loaded allocation exceeds the response bound."}
	}
	var props struct {
		Defaults struct {
			Context int64 `json:"n_ctx"`
		} `json:"default_generation_settings"`
		Slots int64 `json:"total_slots"`
	}
	if json.Unmarshal(data, &props) != nil || props.Defaults.Context <= 0 || props.Slots <= 0 {
		return engines.ServingObservation{}, &ServingIdentityRefusal{"Loaded allocation is unknown."}
	}
	return engines.ServingObservation{ModelID: model, PerSlotContextTokens: props.Defaults.Context, Slots: props.Slots}, nil
}

// Invoked only with the loaded_names response from this sweep. A cold set does
// not query /props. Only semantic failures latch owned-run withdrawal.
func (e *Executor) observeLlamaServing(ctx context.Context, st *engineState, s servingRunSnapshot, sweep uint64, loaded []string, known bool, raw json.RawMessage, transportFailed bool) {
	st.mu.Lock()
	if !s.current(st) || sweep < st.plat.Runtime.servingObservationSweep {
		st.mu.Unlock()
		return
	}
	st.plat.Runtime.servingObservationSweep = sweep
	st.plat.Runtime.servingObserved = nil
	port := st.port
	st.mu.Unlock()
	if transportFailed || ctx.Err() != nil {
		return
	}
	var values []engines.ServingObservation
	reason := ""
	if !known || !llamaLoadedSemantics(raw, loaded) {
		reason = "Loaded model residency is unknown."
	}
	if reason == "" {
		for _, model := range loaded {
			var envelope *engines.ServingEnvelope
			for i := range s.rt.servingLaunch.Envelopes {
				if s.rt.servingLaunch.Envelopes[i].ModelID == model {
					envelope = &s.rt.servingLaunch.Envelopes[i]
					break
				}
			}
			if envelope == nil {
				reason = "Loaded model has no declared serving envelope."
				break
			}
			value, err := e.llamaServingProps(ctx, port, model)
			var semantic *ServingIdentityRefusal
			if err != nil && !errors.As(err, &semantic) {
				return
			}
			if err != nil || value.PerSlotContextTokens != envelope.PerSlotContextTokens || value.Slots != envelope.Slots {
				reason = "Loaded allocation does not match the validated serving envelope."
				break
			}
			values = append(values, value)
		}
	}
	if reason == "" && len(values) > 0 && validateServingSnapshot(ctx, st, s) != nil {
		if ctx.Err() != nil {
			return
		}
		reason = "Serving inputs no longer match the owned launch."
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if !s.current(st) || sweep != st.plat.Runtime.servingObservationSweep || st.plat.Runtime.servingDrift != "" {
		return
	}
	if reason != "" {
		st.plat.Runtime.servingDrift = reason
		return
	}
	st.plat.Runtime.servingObserved = &servingObserved{run: s, values: values, at: time.Now()}
}

// Caller holds st.mu. Expired, withdrawn and replacement-run evidence is absent.
func (e *Executor) servingObservationsLocked(st *engineState) []engines.ServingObservation {
	o := st.plat.Runtime.servingObserved
	if o == nil || !o.run.current(st) || st.plat.Runtime.servingDrift != "" {
		return nil
	}
	budget := e.loadedPollInterval + modelsTimeout
	if budget < modelsTimeout {
		budget = modelsTimeout
	}
	if time.Since(o.at) > budget {
		return nil
	}
	return slices.Clone(o.values)
}

// currentServingObservation is SR's non-loading lookup after a fresh models
// sweep. It revalidates bounded local inputs; it never performs an HTTP probe.
func (e *Executor) currentServingObservation(ctx context.Context, engine, model string) (engines.ServingObservation, error) {
	_, id, err := e.currentServingEnvelope(ctx, engine, model)
	if err != nil {
		return engines.ServingObservation{}, err
	}
	st, err := e.state(engine)
	if err != nil {
		return engines.ServingObservation{}, err
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.runID == id.EngineRunID && st.launchDigest == id.LaunchDigest {
		for _, value := range e.servingObservationsLocked(st) {
			if value.ModelID == model {
				return value, nil
			}
		}
	}
	return engines.ServingObservation{}, &ServingIdentityRefusal{"Model has no fresh loaded allocation observation."}
}
