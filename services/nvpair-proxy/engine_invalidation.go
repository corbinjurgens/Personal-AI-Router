// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"

	"nvpair-shared/nodepolicy"
)

func (p *Proxy) applyInvalidateEngine(params json.RawMessage) (nodepolicy.InvalidateEngineResult, error) {
	var in nodepolicy.InvalidateEngineParams
	if err := json.Unmarshal(params, &in); err != nil {
		return nodepolicy.InvalidateEngineResult{}, err
	}
	p.admission.invalidateEngine(in)
	return nodepolicy.InvalidateEngineResult{OK: true, RunID: p.runID}, nil
}

// invalidateEngine fences held tickets without touching dispatched owners or
// durable capacity markers. Dedupe is per engine/operation within this process;
// the broker replays its switch journal before enabling facades after restart.
func (a *admissionController) invalidateEngine(in nodepolicy.InvalidateEngineParams) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if _, done := a.invalidations[in]; done {
		return
	}
	if a.invalidations == nil {
		a.invalidations = make(map[nodepolicy.InvalidateEngineParams]struct{})
	}
	for k := range a.strict.models {
		if k.engine == in.Engine {
			delete(a.strict.models, k)
			a.strict.generation[k]++
		}
	}
	a.invalidations[in] = struct{}{}
	// StrictReadiness remains withdrawn globally until an authoritative feed
	// exists; removing installed models also removes this engine's eligibility.
	a.signalLocked()
}

func (p *Proxy) applyAdmissionSnapshot(params json.RawMessage) (nodepolicy.AdmissionSnapshotResult, error) {
	var in nodepolicy.AdmissionSnapshotParams
	if err := json.Unmarshal(params, &in); err != nil {
		return nodepolicy.AdmissionSnapshotResult{}, err
	}
	a := p.admission
	a.mu.Lock()
	defer a.mu.Unlock()
	return nodepolicy.AdmissionSnapshotResult{RunID: p.runID, AdmissionState: a.snapshotLocked().state}, nil
}
