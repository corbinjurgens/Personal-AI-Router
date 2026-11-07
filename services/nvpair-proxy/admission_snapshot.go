// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"

	"nvpair-shared/nodepolicy"
)

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
