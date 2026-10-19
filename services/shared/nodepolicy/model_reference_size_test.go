// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
)

func TestServedResidentReferenceSize(t *testing.T) {
	for _, field := range []string{"servedModels", "residentModels"} {
		for _, fill := range []string{"m", "\x00", "<", "界"} {
			t.Run(fmt.Sprintf("%s/%q", field, fill), func(t *testing.T) {
				empty, err := json.Marshal(Role{Engine: "ollama"})
				if err != nil {
					t.Fatal(err)
				}
				unit, err := json.Marshal(fill)
				if err != nil {
					t.Fatal(err)
				}
				n := (maxModelReferenceBytes - len(empty)) / (len(unit) - 2)
				ref := Role{Engine: "ollama", Model: strings.Repeat(fill, n)}
				check := func(wantError bool) {
					t.Helper()
					encoded, err := json.Marshal(map[string][]Role{field: {ref}})
					if err != nil {
						t.Fatal(err)
					}
					var p Policy
					err = json.Unmarshal(encoded, &p)
					if (err != nil) != wantError {
						t.Fatalf("decode error = %v, want error=%v", err, wantError)
					}
					err = validateModelList(field, []Role{ref})
					if (err != nil) != wantError {
						t.Fatalf("validation error = %v, want error=%v", err, wantError)
					}
				}
				check(false)
				ref.Model += fill
				check(true)
			})
		}
	}
}

func TestServedResidentMaximumSizeTransports(t *testing.T) {
	p := Default()
	engine := strings.Repeat("e", engines.MaxDynamicIDLen)
	empty, err := json.Marshal(Role{Engine: engine})
	if err != nil {
		t.Fatal(err)
	}
	for i := range 256 {
		prefix := fmt.Sprintf("%03d", i)
		ref := Role{Engine: engine, Model: prefix + strings.Repeat("m", maxModelReferenceBytes-len(empty)-len(prefix))}
		encoded, err := json.Marshal(ref)
		if err != nil || len(encoded) != maxModelReferenceBytes {
			t.Fatalf("reference bytes=%d, error=%v", len(encoded), err)
		}
		p.ServedModels = append(p.ServedModels, ref)
		p.ResidentModels = append(p.ResidentModels, ref)
		p.Profiles = append(p.Profiles, Profile{Name: prefix, Engine: engine, Model: ref.Model,
			ContextTokens: 2, MaxOutputTokens: 1, MaxConcurrent: 1, MemoryBytes: 1,
			Qualification: &QualificationRef{ID: "q", Revision: "1"}})
	}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	marshal := func(v any) []byte {
		t.Helper()
		raw, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		return raw
	}
	adv := marshal(p.Advertisement())
	// Each list is 256 references * (256 bytes + comma) + brackets - final comma.
	const listBytes = 256*(maxModelReferenceBytes+1) + 1
	if list := marshal(p.ServedModels); len(list) != listBytes {
		t.Fatalf("list bytes=%d, want %d", len(list), listBytes)
	}
	frame := marshal(map[string]any{"jsonrpc": "2.0", "method": "engine:node-projection",
		"params": map[string]any{"policy": p.Advertisement(), "pausedLocally": false}})
	if len(frame)+1 >= jsonrpc.WorkerFrameBytes {
		t.Fatalf("projection frame bytes=%d exceeds worker cap=%d", len(frame)+1, jsonrpc.WorkerFrameBytes)
	}
	// Matches engine-manager policyrelay.go's policyBodyLimit. Other policy
	// fields remain independently variable; this proves the maximum new lists
	// and their required resident profiles fit, not that every policy fits.
	const remotePolicyBodyLimit = 256 << 10
	body := marshal(map[string]any{"policy": p.PeerProjection(), "expectedPersistedRevision": p.Revision})
	if len(body) >= remotePolicyBodyLimit {
		t.Fatalf("remote body bytes=%d exceeds cap=%d", len(body), remotePolicyBodyLimit)
	}
	t.Logf("list=256*(%d+1)+1=%d; advertisement=%d; frame with newline=%d < %d; remote body with both lists and profiles=%d < %d",
		maxModelReferenceBytes, listBytes, len(adv), len(frame)+1, jsonrpc.WorkerFrameBytes, len(body), remotePolicyBodyLimit)
}
