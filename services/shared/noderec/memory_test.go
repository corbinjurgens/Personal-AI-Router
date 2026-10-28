// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestMemoryTelemetryWireRejectsAmbiguousAges(t *testing.T) {
	fact := `{"bytes":"0","source":"fixture","sampledAt":"2026-10-28T00:00:00Z","ageMs":0,"valid":true}`
	decode := func(fact string) error {
		var value MemoryObservation
		return json.Unmarshal([]byte(`{"runId":"run","domains":[{"id":"ram","kind":"system","total":`+fact+`,"free":{"reason":"unavailable","ageMs":0,"valid":false}}]}`), &value)
	}
	if err := decode(fact); err != nil {
		t.Fatalf("known zero rejected: %v", err)
	}
	for _, tc := range []struct{ name, fact string }{
		{"missing age", strings.Replace(fact, `"ageMs":0,`, "", 1)},
		{"null age", strings.Replace(fact, `"ageMs":0`, `"ageMs":null`, 1)},
		{"duplicate age", strings.Replace(fact, `"ageMs":0`, `"ageMs":6000,"ageMs":0`, 1)},
		{"case variant age", strings.Replace(fact, `"ageMs":0`, `"ageMs":6000,"AgeMS":0`, 1)},
		{"wrong case age", strings.Replace(fact, `"ageMs":0`, `"AgeMs":0`, 1)},
		{"escaped duplicate age", strings.Replace(fact, `"ageMs":0`, `"ageMs":6000,"age\u004ds":0`, 1)},
		{"missing validity", strings.Replace(fact, `,"valid":true`, "", 1)},
		{"null validity", strings.Replace(fact, `"valid":true`, `"valid":null`, 1)},
		{"null source", strings.Replace(fact, `"source":"fixture"`, `"source":null`, 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := decode(tc.fact); err == nil {
				t.Fatal("ambiguous fact accepted as a fresh measurement")
			}
		})
	}
}

func TestMemoryTelemetryEncodedBudget(t *testing.T) {
	unknown := MemoryFact{Reason: "unavailable"}
	m := MemoryObservation{RunID: "run", Domains: []MemoryDomain{}}
	for i := 0; i < 64; i++ {
		domain := MemoryDomain{ID: fmt.Sprintf("gpu-%d", i), Kind: "dedicated", Total: unknown, Free: unknown}
		for j := 0; j < 64; j++ {
			domain.DeviceIDs = append(domain.DeviceIDs, fmt.Sprintf("%02d/%02d/", i, j)+strings.Repeat("x", 240))
		}
		m.Domains = append(m.Domains, domain)
	}
	encoded, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) <= MaxMemoryObservationBytes {
		t.Fatal("fixture does not exceed aggregate observation budget")
	}
	if m.Validate() == nil {
		t.Fatal("aggregate alias payload passed validation")
	}
	var decoded MemoryObservation
	if json.Unmarshal(encoded, &decoded) == nil {
		t.Fatal("oversized wire observation accepted")
	}
	for i := range m.Domains {
		m.Domains[i].DeviceIDs = m.Domains[i].DeviceIDs[:1]
	}
	if err := m.Validate(); err != nil {
		t.Fatalf("bounded topology rejected: %v", err)
	}
	frame, err := json.Marshal(struct {
		JSONRPC string        `json:"jsonrpc"`
		Method  string        `json:"method"`
		Params  NodeTelemetry `json:"params"`
	}{"2.0", NotifyNodeTelemetry, NodeTelemetry{HostUUID: "host", Memory: &m}})
	if err != nil {
		t.Fatal(err)
	}
	if len(frame)+1 >= 1<<20 {
		t.Fatal("bounded memory notification exceeds scanner frame")
	}
}

func TestMemoryTelemetryValidation(t *testing.T) {
	fact := MemoryFact{Bytes: "0", Source: "fixture", SampledAt: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC), Valid: true}
	unknown := MemoryFact{Reason: "unavailable"}
	valid := MemoryObservation{RunID: "run", Domains: []MemoryDomain{{ID: "ram", Kind: "system", Total: fact, Free: fact}}}
	if err := valid.Validate(); err != nil {
		t.Fatalf("known zero: %v", err)
	}
	test := func(name string, change func(*MemoryObservation)) {
		t.Run(name, func(t *testing.T) {
			m := valid.Clone()
			change(m)
			if err := m.Validate(); err == nil {
				t.Fatal("expected invalid observation")
			}
		})
	}
	test("noncanonical bytes", func(m *MemoryObservation) { m.Domains[0].Free.Bytes = "00" })
	test("domain limit", func(m *MemoryObservation) { m.Domains = make([]MemoryDomain, 65) })
	test("identifier limit", func(m *MemoryObservation) { m.RunID = strings.Repeat("a", 257) })
	test("control identifier", func(m *MemoryObservation) { m.RunID = "a\x00" })
	test("missing timestamp", func(m *MemoryObservation) { m.Domains[0].Free.SampledAt = time.Time{} })
	test("overflow", func(m *MemoryObservation) { m.Domains[0].Free.Bytes = "18446744073709551616" })
	test("available exceeds total", func(m *MemoryObservation) { m.Domains[0].Free.Bytes = "1" })
	test("negative age", func(m *MemoryObservation) { m.Domains[0].Free.AgeMS = -1 })
	test("duplicate pool", func(m *MemoryObservation) { m.Domains = append(m.Domains, m.Domains[0]) })
	test("overlapping UMA", func(m *MemoryObservation) {
		m.Domains = append(m.Domains, MemoryDomain{ID: "uma", Kind: "shared", Total: unknown, Free: unknown})
	})
	test("duplicate device", func(m *MemoryObservation) { m.Domains[0].DeviceIDs = []string{"gpu", "gpu"} })
	test("unknown inherits timestamp", func(m *MemoryObservation) { m.Domains[0].Free = unknown; m.Domains[0].Free.SampledAt = fact.SampledAt })
}

func TestMemoryTelemetryDecodeRejectsInvalidFacts(t *testing.T) {
	var value MemoryObservation
	err := json.Unmarshal([]byte(`{"runId":"run","domains":[{"id":"ram","kind":"system","total":{"bytes":"00","source":"fixture","sampledAt":"2026-10-28T00:00:00Z","ageMs":0,"valid":true},"free":{"reason":"unavailable","ageMs":0,"valid":false}}]}`), &value)
	if err == nil {
		t.Fatal("malformed decimal survived wire decoder")
	}
}

func TestMemoryTelemetryClone(t *testing.T) {
	m := &MemoryObservation{Domains: []MemoryDomain{{DeviceIDs: []string{"original"}}}}
	clone := m.Clone()
	clone.Domains[0].DeviceIDs[0] = "changed"
	if m.Domains[0].DeviceIDs[0] != "original" {
		t.Fatal("clone aliases device identifiers")
	}
}

func TestMemoryTelemetryEmptyTopologyIsExplicit(t *testing.T) {
	m := &MemoryObservation{RunID: "run", Domains: []MemoryDomain{}}
	clone := m.Clone()
	if clone.Domains == nil || clone.Validate() != nil {
		t.Fatal("clone changed an explicit empty topology into absent data")
	}
	for _, input := range []string{`{"runId":"run"}`, `{"runId":"run","domains":null}`} {
		var decoded MemoryObservation
		if json.Unmarshal([]byte(input), &decoded) == nil {
			t.Fatal("absent topology accepted")
		}
	}
	encoded, err := json.Marshal(clone)
	if err != nil {
		t.Fatal(err)
	}
	var decoded MemoryObservation
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("empty topology round trip: %v", err)
	}
}

func TestMemoryTelemetryAddAge(t *testing.T) {
	m := &MemoryObservation{Domains: []MemoryDomain{{
		Total: MemoryFact{Bytes: "0", Valid: true, AgeMS: 4},
		Free:  MemoryFact{Bytes: "1", Valid: true, AgeMS: 1<<63 - 1},
	}}}
	m.AddAge(1)
	if m.Domains[0].Total.AgeMS != 5 || !m.Domains[0].Total.Valid {
		t.Fatal("known age did not advance")
	}
	if m.Domains[0].Free.Valid || m.Domains[0].Free.Reason != "invalid-age" {
		t.Fatal("overflow did not become unknown")
	}
}
