// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package performancewire

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func memoryTestDigest(b byte) string {
	return strings.Repeat(string(b), 64)
}

func memoryTestSourceJSON() string {
	return `{"hostUuid":"host-a","managerIncarnation":"mgr-1","engineRunId":"run-1","launchDigest":"` +
		memoryTestDigest('a') + `","modelDigest":"` + memoryTestDigest('b') + `"}`
}

func memoryTestSourceField() string {
	return `"source":` + memoryTestSourceJSON()
}

func memoryTestQualifiedJSON() string {
	return `{"version":1,"entries":[{"engine":"ollama","model":"llama3:8b","state":"qualified","uncertainty":[],` +
		memoryTestSourceField() + `,` +
		`"appliedSettings":{"epoch":"epoch-1","revision":7},` +
		`"envelope":{"domain":"dedicated-device","effectiveContextTokens":8192,"concurrentExecutions":2,"offloadMode":"gpu-full"},` +
		`"observed":{"allocatedBytes":9000,"extractorRevision":"ollama-0.12.3"},` +
		`"qualification":{"upperBytes":10000,"id":"mq-1","revision":"3","artifactDigest":"` + memoryTestDigest('c') + `","procedureDigest":"` + memoryTestDigest('d') + `"},` +
		`"provenance":{"procedureId":"mem-proc","procedureRevision":"2","hardwareFingerprint":"gpu-abc","runtimeRevision":"runtime-1","hardwareDescription":"NVIDIA RTX 4090 24GB","driverRevision":"driver-555.42","scope":"operator-attested-current-run"},` +
		`"freshness":{"sampleAgeMs":1200,"qualificationAgeMs":60000,"observationGeneration":42,"readingValidForMs":15000}}],"omittedEntries":0}`
}

func memoryTestAdvisoryJSON() string {
	return `{"version":1,"entries":[{"engine":"ollama","model":"llama3:8b","state":"advisory","uncertainty":["missing-reading"],` +
		memoryTestSourceField() + `,` +
		`"appliedSettings":{"epoch":"epoch-1","revision":7},` +
		`"envelope":{"domain":"dedicated-device","effectiveContextTokens":null,"concurrentExecutions":null,"offloadMode":null},` +
		`"observed":{"allocatedBytes":null,"extractorRevision":null},` +
		`"qualification":{"upperBytes":null,"id":null,"revision":null,"artifactDigest":null,"procedureDigest":null},` +
		`"provenance":{"procedureId":null,"procedureRevision":null,"hardwareFingerprint":null,"runtimeRevision":null,"hardwareDescription":null,"driverRevision":null,"scope":null},` +
		`"freshness":{"sampleAgeMs":null,"qualificationAgeMs":null,"observationGeneration":null,"readingValidForMs":15000}}],"omittedEntries":0}`
}

func memoryTestUnknownJSON() string {
	return `{"version":1,"entries":[{"engine":"ollama","model":"llama3:8b","state":"unknown","uncertainty":["identity-unknown"],` +
		`"source":{"hostUuid":null,"managerIncarnation":null,"engineRunId":null,"launchDigest":null,"modelDigest":null},` +
		`"appliedSettings":{"epoch":null,"revision":null},` +
		`"envelope":{"domain":null,"effectiveContextTokens":null,"concurrentExecutions":null,"offloadMode":null},` +
		`"observed":{"allocatedBytes":null,"extractorRevision":null},` +
		`"qualification":{"upperBytes":null,"id":null,"revision":null,"artifactDigest":null,"procedureDigest":null},` +
		`"provenance":{"procedureId":null,"procedureRevision":null,"hardwareFingerprint":null,"runtimeRevision":null,"hardwareDescription":null,"driverRevision":null,"scope":null},` +
		`"freshness":{"sampleAgeMs":null,"qualificationAgeMs":null,"observationGeneration":null,"readingValidForMs":15000}}],"omittedEntries":0}`
}

func memoryTestMustDecode(t *testing.T, raw string) MemoryEvidence {
	t.Helper()
	m, err := DecodeMemoryEvidence([]byte(raw))
	if err != nil {
		t.Fatalf("decode setup value: %v", err)
	}
	return m
}

func memoryTestMutate(t *testing.T, base, old, new string) string {
	t.Helper()
	out := strings.Replace(base, old, new, 1)
	if out == base {
		t.Fatalf("setup: %q not found in base", old)
	}
	return out
}

func memoryTestExpectError(t *testing.T, raw, caseName string) {
	t.Helper()
	if _, err := DecodeMemoryEvidence([]byte(raw)); err == nil {
		t.Fatalf("%s: malformed memory evidence was accepted", caseName)
	}
}

func memoryTestInt64(v int64) *int64 {
	return &v
}

func memoryTestString(v string) *string {
	return &v
}

func memoryTestAdvisoryEntry(i int) MemoryEvidenceEntry {
	return MemoryEvidenceEntry{
		Engine:      "engine-" + strconv.Itoa(i),
		Model:       "model-" + strconv.Itoa(i),
		State:       MemoryStateAdvisory,
		Uncertainty: []MemoryUncertainty{},
		Freshness:   MemoryFreshness{ReadingValidForMs: MemoryReadingValidForMs},
	}
}

func TestMemoryEvidenceRoundTripStatesAndEmpty(t *testing.T) {
	cases := []struct {
		name        string
		raw         string
		wantEntries int
		wantState   MemoryState
	}{
		{name: "qualified", raw: memoryTestQualifiedJSON(), wantEntries: 1, wantState: MemoryStateQualified},
		{name: "advisory", raw: memoryTestAdvisoryJSON(), wantEntries: 1, wantState: MemoryStateAdvisory},
		{name: "unknown", raw: memoryTestUnknownJSON(), wantEntries: 1, wantState: MemoryStateUnknown},
		{name: "empty", raw: `{"version":1,"entries":[],"omittedEntries":0}`, wantEntries: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			decoded := memoryTestMustDecode(t, tc.raw)
			if len(decoded.Entries) != tc.wantEntries {
				t.Fatalf("decoded %d entries, want %d", len(decoded.Entries), tc.wantEntries)
			}
			if tc.wantEntries == 0 {
				if decoded.Entries == nil {
					t.Fatal("empty entries decoded as null")
				}
			} else {
				if decoded.Entries[0].State != tc.wantState {
					t.Fatalf("state %q, want %q", decoded.Entries[0].State, tc.wantState)
				}
				if decoded.Entries[0].Engine != "ollama" || decoded.Entries[0].Model != "llama3:8b" {
					t.Fatalf("entry identity changed: %q %q", decoded.Entries[0].Engine, decoded.Entries[0].Model)
				}
			}
			encoded, err := EncodeMemoryEvidence(decoded)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			round := memoryTestMustDecode(t, string(encoded))
			if !reflect.DeepEqual(decoded, round) {
				t.Fatalf("round trip changed the envelope")
			}
		})
	}
}

func TestMemoryEvidenceBoundsRetainedEntriesAndOmittedCount(t *testing.T) {
	retained := make([]MemoryEvidenceEntry, MaxMemoryEvidenceEntries)
	for i := range retained {
		retained[i] = memoryTestAdvisoryEntry(i)
	}
	if _, err := EncodeMemoryEvidence(MemoryEvidence{Version: 1, Entries: retained, OmittedEntries: 17}); err != nil {
		t.Fatalf("128 retained entries plus omitted count were rejected: %v", err)
	}
	none := make([]MemoryEvidenceEntry, 0)
	if _, err := EncodeMemoryEvidence(MemoryEvidence{Version: 1, Entries: none, OmittedEntries: MaxMemoryEvidenceEntries + 1}); err != nil {
		t.Fatalf("omittedEntries above the retained cap was rejected: %v", err)
	}
	if _, err := EncodeMemoryEvidence(MemoryEvidence{Version: 1, Entries: none, OmittedEntries: -1}); err == nil {
		t.Fatal("negative omittedEntries was accepted")
	}
}

func TestMemoryEvidenceStaleAdvisoryAgesStayRepresentable(t *testing.T) {
	stale := memoryTestMutate(t, memoryTestAdvisoryJSON(), `"sampleAgeMs":null`, `"sampleAgeMs":60000`)
	stale = memoryTestMutate(t, stale, `"qualificationAgeMs":null`, `"qualificationAgeMs":9223372036854775807`)
	decoded := memoryTestMustDecode(t, stale)
	if decoded.Entries[0].State != MemoryStateAdvisory {
		t.Fatalf("state %q, want advisory", decoded.Entries[0].State)
	}
	if decoded.Entries[0].Freshness.SampleAgeMs == nil || *decoded.Entries[0].Freshness.SampleAgeMs != 60000 {
		t.Fatal("stale advisory sample age was not preserved")
	}
	if decoded.Entries[0].Freshness.QualificationAgeMs == nil || *decoded.Entries[0].Freshness.QualificationAgeMs != 9223372036854775807 {
		t.Fatal("large advisory qualification age was not preserved")
	}
	if decoded.Entries[0].Freshness.Generation != nil {
		t.Fatal("unknown observation generation was not null")
	}
	encoded, err := EncodeMemoryEvidence(decoded)
	if err != nil {
		t.Fatalf("encode stale advisory: %v", err)
	}
	round := memoryTestMustDecode(t, string(encoded))
	if !reflect.DeepEqual(decoded, round) {
		t.Fatal("stale advisory round trip changed the envelope")
	}
}

func TestMemoryEvidenceObservationGenerationNullVersusValue(t *testing.T) {
	base := memoryTestAdvisoryJSON()
	unknown := memoryTestMustDecode(t, base)
	if unknown.Entries[0].Freshness.Generation != nil {
		t.Fatal("null observation generation decoded as a value")
	}
	present := memoryTestMustDecode(t, memoryTestMutate(t, base, `"observationGeneration":null`, `"observationGeneration":42`))
	if present.Entries[0].Freshness.Generation == nil || *present.Entries[0].Freshness.Generation != 42 {
		t.Fatal("positive observation generation was not preserved")
	}
}

func TestMemoryEvidenceMeasuredZeroAndUnknownStayDistinct(t *testing.T) {
	measured := MemoryEvidence{
		Version: 1,
		Entries: []MemoryEvidenceEntry{{
			Engine:      "ollama",
			Model:       "llama3:8b",
			State:       MemoryStateAdvisory,
			Uncertainty: []MemoryUncertainty{},
			Observed:    MemoryObservation{AllocatedBytes: memoryTestInt64(0)},
			Freshness:   MemoryFreshness{ReadingValidForMs: MemoryReadingValidForMs},
		}},
	}
	encoded, err := EncodeMemoryEvidence(measured)
	if err != nil {
		t.Fatalf("encode measured zero: %v", err)
	}
	decoded := memoryTestMustDecode(t, string(encoded))
	if decoded.Entries[0].Observed.AllocatedBytes == nil {
		t.Fatal("measured zero decoded as unknown")
	}
	if *decoded.Entries[0].Observed.AllocatedBytes != 0 {
		t.Fatalf("measured zero changed to %d", *decoded.Entries[0].Observed.AllocatedBytes)
	}

	unknown := measured
	unknown.Entries = []MemoryEvidenceEntry{measured.Entries[0]}
	unknown.Entries[0].Observed.AllocatedBytes = nil
	encoded, err = EncodeMemoryEvidence(unknown)
	if err != nil {
		t.Fatalf("encode unknown: %v", err)
	}
	decoded = memoryTestMustDecode(t, string(encoded))
	if decoded.Entries[0].Observed.AllocatedBytes != nil {
		t.Fatalf("unknown allocation became %d", *decoded.Entries[0].Observed.AllocatedBytes)
	}
}

func TestMemoryEvidenceRejectsStructuralInput(t *testing.T) {
	base := memoryTestQualifiedJSON()
	memoryTestMustDecode(t, base)
	cases := []struct {
		name string
		raw  string
	}{
		{name: "unsupported version", raw: memoryTestMutate(t, base, `"version":1`, `"version":2`)},
		{name: "missing entries", raw: `{"version":1,"omittedEntries":0}`},
		{name: "null entries", raw: `{"version":1,"entries":null,"omittedEntries":0}`},
		{name: "missing omittedEntries", raw: `{"version":1,"entries":[]}`},
		{name: "null omittedEntries", raw: `{"version":1,"entries":[],"omittedEntries":null}`},
		{name: "unknown root key", raw: memoryTestMutate(t, base, `{"version":1`, `{"extra":1,"version":1`)},
		{name: "duplicate root key", raw: memoryTestMutate(t, base, `{"version":1`, `{"version":1,"version":1`)},
		{name: "case variant root key", raw: memoryTestMutate(t, base, `"version":1`, `"Version":1`)},
		{name: "unknown entry key", raw: memoryTestMutate(t, base, `"engine":"ollama"`, `"engine":"ollama","extra":1`)},
		{name: "duplicate entry key", raw: memoryTestMutate(t, base, `"engine":"ollama"`, `"engine":"ollama","engine":"other"`)},
		{name: "case variant entry key", raw: memoryTestMutate(t, base, `"engine":"ollama"`, `"Engine":"ollama"`)},
		{name: "null required entry field", raw: memoryTestMutate(t, base, `"engine":"ollama"`, `"engine":null`)},
		{name: "null uncertainty", raw: memoryTestMutate(t, base, `"uncertainty":[]`, `"uncertainty":null`)},
		{name: "null source group", raw: memoryTestMutate(t, base, memoryTestSourceField(), `"source":null`)},
		{name: "missing nested key", raw: memoryTestMutate(t, base, `"hostUuid":"host-a",`, ``)},
		{name: "missing runtime revision key", raw: memoryTestMutate(t, base, `"runtimeRevision":"runtime-1",`, ``)},
		{name: "missing hardware description key", raw: memoryTestMutate(t, base, `"hardwareDescription":"NVIDIA RTX 4090 24GB",`, ``)},
		{name: "missing driver revision key", raw: memoryTestMutate(t, base, `"driverRevision":"driver-555.42",`, ``)},
		{name: "missing observation generation key", raw: memoryTestMutate(t, base, `"observationGeneration":42,`, ``)},
		{name: "unknown nested key", raw: memoryTestMutate(t, base, `"hostUuid":"host-a"`, `"hostUuid":"host-a","extra":1`)},
		{name: "duplicate nested key", raw: memoryTestMutate(t, base, `"hostUuid":"host-a"`, `"hostUuid":"host-a","hostUuid":"host-b"`)},
		{name: "case variant nested key", raw: memoryTestMutate(t, base, `"hostUuid":"host-a"`, `"HostUuid":"host-a"`)},
		{name: "trailing value", raw: base + `{}`},
		{name: "not an object", raw: `[]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			memoryTestExpectError(t, tc.raw, tc.name)
		})
	}
}

func TestMemoryEvidenceRejectsMalformedStringsAndEnums(t *testing.T) {
	base := memoryTestQualifiedJSON()
	memoryTestMustDecode(t, base)
	longID := strings.Repeat("x", MaxIDBytes+1)
	longModel := strings.Repeat("x", MaxModelBytes+1)
	cases := []struct {
		name string
		raw  string
	}{
		{name: "control character in engine", raw: memoryTestMutate(t, base, `"engine":"ollama"`, `"engine":"oll\u0000ama"`)},
		{name: "control character in model", raw: memoryTestMutate(t, base, `"model":"llama3:8b"`, `"model":"llama3\u0000:8b"`)},
		{name: "oversize engine", raw: memoryTestMutate(t, base, `"engine":"ollama"`, `"engine":"`+longID+`"`)},
		{name: "oversize model", raw: memoryTestMutate(t, base, `"model":"llama3:8b"`, `"model":"`+longModel+`"`)},
		{name: "unsupported state", raw: memoryTestMutate(t, base, `"state":"qualified"`, `"state":"measured"`)},
		{name: "unsupported uncertainty", raw: memoryTestMutate(t, base, `"uncertainty":[]`, `"uncertainty":["other"]`)},
		{name: "duplicate uncertainty", raw: memoryTestMutate(t, base, `"uncertainty":[]`, `"uncertainty":["stale","stale"]`)},
		{name: "unsupported domain", raw: memoryTestMutate(t, base, `"domain":"dedicated-device"`, `"domain":"gpu"`)},
		{name: "unsupported scope", raw: memoryTestMutate(t, base, `"scope":"operator-attested-current-run"`, `"scope":"operator-attested"`)},
		{name: "uppercase digest", raw: memoryTestMutate(t, base, memoryTestDigest('a'), strings.ToUpper(memoryTestDigest('a')))},
		{name: "short digest", raw: memoryTestMutate(t, base, memoryTestDigest('b'), "abc")},
		{name: "leading zero revision", raw: memoryTestMutate(t, base, `"revision":"3"`, `"revision":"03"`)},
		{name: "zero revision", raw: memoryTestMutate(t, base, `"revision":"3"`, `"revision":"0"`)},
		{name: "fractional revision", raw: memoryTestMutate(t, base, `"revision":"3"`, `"revision":"1.0"`)},
		{name: "overflow revision", raw: memoryTestMutate(t, base, `"revision":"3"`, `"revision":"18446744073709551616"`)},
		{name: "space in hardware fingerprint", raw: memoryTestMutate(t, base, `"hardwareFingerprint":"gpu-abc"`, `"hardwareFingerprint":"gpu abc"`)},
		{name: "empty runtime revision", raw: memoryTestMutate(t, base, `"runtimeRevision":"runtime-1"`, `"runtimeRevision":""`)},
		{name: "oversize hardware description", raw: memoryTestMutate(t, base, `"hardwareDescription":"NVIDIA RTX 4090 24GB"`, `"hardwareDescription":"`+longModel+`"`)},
		{name: "control driver revision", raw: memoryTestMutate(t, base, `"driverRevision":"driver-555.42"`, `"driverRevision":"driver\u0000"`)},
		{name: "space in epoch", raw: memoryTestMutate(t, base, `"epoch":"epoch-1"`, `"epoch":"epoch 1"`)},
		{name: "empty offload mode", raw: memoryTestMutate(t, base, `"offloadMode":"gpu-full"`, `"offloadMode":""`)},
		{name: "control extractor revision", raw: memoryTestMutate(t, base, `"extractorRevision":"ollama-0.12.3"`, `"extractorRevision":"ollama\u0000"`)},
		{name: "space in qualification id", raw: memoryTestMutate(t, base, `"id":"mq-1"`, `"id":"mq 1"`)},
		{name: "space in host uuid", raw: memoryTestMutate(t, base, `"hostUuid":"host-a"`, `"hostUuid":"host a"`)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			memoryTestExpectError(t, tc.raw, tc.name)
		})
	}
}

func TestMemoryEvidenceRejectsInvalidNumbers(t *testing.T) {
	base := memoryTestQualifiedJSON()
	memoryTestMustDecode(t, base)
	cases := []struct {
		name string
		raw  string
	}{
		{name: "negative allocation", raw: memoryTestMutate(t, base, `"allocatedBytes":9000`, `"allocatedBytes":-1`)},
		{name: "fractional allocation", raw: memoryTestMutate(t, base, `"allocatedBytes":9000`, `"allocatedBytes":9000.5`)},
		{name: "overflow allocation", raw: memoryTestMutate(t, base, `"allocatedBytes":9000`, `"allocatedBytes":9223372036854775808`)},
		{name: "zero context", raw: memoryTestMutate(t, base, `"effectiveContextTokens":8192`, `"effectiveContextTokens":0`)},
		{name: "negative context", raw: memoryTestMutate(t, base, `"effectiveContextTokens":8192`, `"effectiveContextTokens":-1`)},
		{name: "zero concurrency", raw: memoryTestMutate(t, base, `"concurrentExecutions":2`, `"concurrentExecutions":0`)},
		{name: "fractional concurrency", raw: memoryTestMutate(t, base, `"concurrentExecutions":2`, `"concurrentExecutions":1.5`)},
		{name: "stale sample age", raw: memoryTestMutate(t, base, `"sampleAgeMs":1200`, `"sampleAgeMs":15001`)},
		{name: "overflow sample age", raw: memoryTestMutate(t, base, `"sampleAgeMs":1200`, `"sampleAgeMs":9223372036854775808`)},
		{name: "negative sample age", raw: memoryTestMutate(t, base, `"sampleAgeMs":1200`, `"sampleAgeMs":-1`)},
		{name: "negative qualification age", raw: memoryTestMutate(t, base, `"qualificationAgeMs":60000`, `"qualificationAgeMs":-1`)},
		{name: "overflow qualification age", raw: memoryTestMutate(t, base, `"qualificationAgeMs":60000`, `"qualificationAgeMs":9223372036854775808`)},
		{name: "zero observation generation", raw: memoryTestMutate(t, base, `"observationGeneration":42`, `"observationGeneration":0`)},
		{name: "negative observation generation", raw: memoryTestMutate(t, base, `"observationGeneration":42`, `"observationGeneration":-1`)},
		{name: "fractional observation generation", raw: memoryTestMutate(t, base, `"observationGeneration":42`, `"observationGeneration":1.5`)},
		{name: "overflow observation generation", raw: memoryTestMutate(t, base, `"observationGeneration":42`, `"observationGeneration":18446744073709551616`)},
		{name: "wrong reading window", raw: memoryTestMutate(t, base, `"readingValidForMs":15000`, `"readingValidForMs":14999`)},
		{name: "zero applied revision", raw: memoryTestMutate(t, base, `"revision":7`, `"revision":0`)},
		{name: "fractional applied revision", raw: memoryTestMutate(t, base, `"revision":7`, `"revision":7.5`)},
		{name: "zero upper bound", raw: memoryTestMutate(t, base, `"upperBytes":10000`, `"upperBytes":0`)},
		{name: "negative upper bound", raw: memoryTestMutate(t, base, `"upperBytes":10000`, `"upperBytes":-1`)},
		{name: "overflow upper bound", raw: memoryTestMutate(t, base, `"upperBytes":10000`, `"upperBytes":9223372036854775808`)},
		{name: "fractional version", raw: memoryTestMutate(t, base, `"version":1`, `"version":1.0`)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			memoryTestExpectError(t, tc.raw, tc.name)
		})
	}
}

func TestMemoryEvidenceRejectsUnsafeQualification(t *testing.T) {
	base := memoryTestQualifiedJSON()
	memoryTestMustDecode(t, base)
	cases := []struct {
		name string
		raw  string
	}{
		{name: "advisory with upper", raw: memoryTestMutate(t, base, `"state":"qualified"`, `"state":"advisory"`)},
		{name: "unknown with upper", raw: memoryTestMutate(t, base, `"state":"qualified"`, `"state":"unknown"`)},
		{name: "qualified with uncertainty", raw: memoryTestMutate(t, base, `"uncertainty":[]`, `"uncertainty":["stale"]`)},
		{name: "stale positive bound", raw: memoryTestMutate(t, base, `"sampleAgeMs":1200`, `"sampleAgeMs":15001`)},
		{name: "missing qualification age", raw: memoryTestMutate(t, base, `"qualificationAgeMs":60000`, `"qualificationAgeMs":null`)},
		{name: "missing host binding", raw: memoryTestMutate(t, base, `"hostUuid":"host-a"`, `"hostUuid":null`)},
		{name: "missing launch binding", raw: memoryTestMutate(t, base, memoryTestDigest('a'), `null`)},
		{name: "missing applied epoch", raw: memoryTestMutate(t, base, `"epoch":"epoch-1"`, `"epoch":null`)},
		{name: "missing applied revision", raw: memoryTestMutate(t, base, `"revision":7`, `"revision":null`)},
		{name: "missing domain", raw: memoryTestMutate(t, base, `"domain":"dedicated-device"`, `"domain":null`)},
		{name: "missing context", raw: memoryTestMutate(t, base, `"effectiveContextTokens":8192`, `"effectiveContextTokens":null`)},
		{name: "missing concurrency", raw: memoryTestMutate(t, base, `"concurrentExecutions":2`, `"concurrentExecutions":null`)},
		{name: "missing offload mode", raw: memoryTestMutate(t, base, `"offloadMode":"gpu-full"`, `"offloadMode":null`)},
		{name: "missing allocation", raw: memoryTestMutate(t, base, `"allocatedBytes":9000`, `"allocatedBytes":null`)},
		{name: "missing extractor revision", raw: memoryTestMutate(t, base, `"extractorRevision":"ollama-0.12.3"`, `"extractorRevision":null`)},
		{name: "missing upper", raw: memoryTestMutate(t, base, `"upperBytes":10000`, `"upperBytes":null`)},
		{name: "missing qualification id", raw: memoryTestMutate(t, base, `"id":"mq-1"`, `"id":null`)},
		{name: "missing qualification revision", raw: memoryTestMutate(t, base, `"revision":"3"`, `"revision":null`)},
		{name: "missing artifact digest", raw: memoryTestMutate(t, base, memoryTestDigest('c'), `null`)},
		{name: "missing procedure digest", raw: memoryTestMutate(t, base, memoryTestDigest('d'), `null`)},
		{name: "missing procedure id", raw: memoryTestMutate(t, base, `"procedureId":"mem-proc"`, `"procedureId":null`)},
		{name: "missing procedure revision", raw: memoryTestMutate(t, base, `"procedureRevision":"2"`, `"procedureRevision":null`)},
		{name: "missing hardware fingerprint", raw: memoryTestMutate(t, base, `"hardwareFingerprint":"gpu-abc"`, `"hardwareFingerprint":null`)},
		{name: "missing runtime revision", raw: memoryTestMutate(t, base, `"runtimeRevision":"runtime-1"`, `"runtimeRevision":null`)},
		{name: "missing hardware description", raw: memoryTestMutate(t, base, `"hardwareDescription":"NVIDIA RTX 4090 24GB"`, `"hardwareDescription":null`)},
		{name: "missing driver revision", raw: memoryTestMutate(t, base, `"driverRevision":"driver-555.42"`, `"driverRevision":null`)},
		{name: "missing observation generation", raw: memoryTestMutate(t, base, `"observationGeneration":42`, `"observationGeneration":null`)},
		{name: "missing scope", raw: memoryTestMutate(t, base, `"scope":"operator-attested-current-run"`, `"scope":null`)},
		{name: "wrong scope", raw: memoryTestMutate(t, base, `"scope":"operator-attested-current-run"`, `"scope":"operator-attested"`)},
		{name: "allocation above upper", raw: memoryTestMutate(t, base, `"allocatedBytes":9000`, `"allocatedBytes":10001`)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			memoryTestExpectError(t, tc.raw, tc.name)
		})
	}

	for _, tc := range []struct {
		name string
		raw  string
	}{
		{name: "fresh window boundary", raw: memoryTestMutate(t, base, `"sampleAgeMs":1200`, `"sampleAgeMs":15000`)},
		{name: "large qualification age", raw: memoryTestMutate(t, base, `"qualificationAgeMs":60000`, `"qualificationAgeMs":9223372036854775807`)},
		{name: "allocation equal to upper", raw: memoryTestMutate(t, base, `"allocatedBytes":9000`, `"allocatedBytes":10000`)},
		{name: "zero sample age", raw: memoryTestMutate(t, base, `"sampleAgeMs":1200`, `"sampleAgeMs":0`)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			memoryTestMustDecode(t, tc.raw)
		})
	}
}

func TestMemoryEvidenceAcceptsPartialAdvisoryMetadata(t *testing.T) {
	base := memoryTestAdvisoryJSON()
	decoded := memoryTestMustDecode(t, base)
	if decoded.Entries[0].Qualification.UpperBytes != nil {
		t.Fatal("advisory entry carried an upper bound")
	}
	if decoded.Entries[0].Observed.AllocatedBytes != nil {
		t.Fatal("advisory unknown allocation was not null")
	}
	if decoded.Entries[0].Freshness.Generation != nil {
		t.Fatal("advisory unknown observation generation was not null")
	}
	if decoded.Entries[0].Provenance.RuntimeRevision != nil ||
		decoded.Entries[0].Provenance.HardwareDescription != nil ||
		decoded.Entries[0].Provenance.DriverRevision != nil {
		t.Fatal("advisory entry carried partial provenance text")
	}

	withZero := memoryTestMutate(t, base, `"allocatedBytes":null`, `"allocatedBytes":0`)
	decoded = memoryTestMustDecode(t, withZero)
	if decoded.Entries[0].Observed.AllocatedBytes == nil || *decoded.Entries[0].Observed.AllocatedBytes != 0 {
		t.Fatal("advisory measured zero was not preserved")
	}
	if decoded.Entries[0].Qualification.UpperBytes != nil {
		t.Fatal("advisory entry gained an upper bound")
	}
}

func TestMemoryEvidenceRejectsDuplicateKeysAndOversize(t *testing.T) {
	tooMany := make([]MemoryEvidenceEntry, MaxMemoryEvidenceEntries+1)
	for i := range tooMany {
		tooMany[i] = memoryTestAdvisoryEntry(i)
	}
	if _, err := EncodeMemoryEvidence(MemoryEvidence{Version: 1, Entries: tooMany}); err == nil {
		t.Fatal("too many entries were accepted")
	}

	large := make([]MemoryEvidenceEntry, MaxMemoryEvidenceEntries)
	id := strings.Repeat("i", MaxIDBytes)
	digest := memoryTestDigest('e')
	epoch := strings.Repeat("p", MaxIDBytes)
	offload := strings.Repeat("o", MaxIDBytes)
	extractor := strings.Repeat("r", MaxIDBytes)
	procedure := strings.Repeat("q", MaxIDBytes)
	hardware := strings.Repeat("h", MaxIDBytes)
	domain := MemoryDomainDedicatedDevice
	for i := range large {
		e := memoryTestAdvisoryEntry(i)
		e.Engine = strings.Repeat("e", MaxIDBytes-3) + fmt.Sprintf("%02d", i)
		e.Model = strings.Repeat("m", MaxModelBytes-4) + fmt.Sprintf("%03d", i)
		e.Source = MemorySource{
			HostUUID:           &id,
			ManagerIncarnation: &id,
			EngineRunID:        &id,
			LaunchDigest:       &digest,
			ModelDigest:        &digest,
		}
		e.AppliedSettings = MemoryAppliedSettings{Epoch: &epoch, Revision: memoryTestInt64(1)}
		e.Envelope = MemoryEnvelope{
			Domain:                 &domain,
			EffectiveContextTokens: memoryTestInt64(8192),
			ConcurrentExecutions:   memoryTestInt64(1),
			OffloadMode:            &offload,
		}
		e.Observed = MemoryObservation{AllocatedBytes: memoryTestInt64(1), ExtractorRevision: &extractor}
		e.Qualification = MemoryQualification{
			ID:              &id,
			Revision:        memoryTestString("1"),
			ArtifactDigest:  &digest,
			ProcedureDigest: &digest,
		}
		e.Provenance = MemoryProvenance{
			ProcedureID:         &procedure,
			ProcedureRevision:   memoryTestString("1"),
			HardwareFingerprint: &hardware,
			Scope:               memoryTestScopePointer(),
		}
		large[i] = e
	}
	if _, err := EncodeMemoryEvidence(MemoryEvidence{Version: 1, Entries: large}); err == nil {
		t.Fatal("oversize envelope was accepted")
	}

	oversizeInput := append(bytes.Repeat([]byte(" "), MaxMemoryEvidenceBytes+1), []byte(memoryTestQualifiedJSON())...)
	if _, err := DecodeMemoryEvidence(oversizeInput); err == nil {
		t.Fatal("oversize input was accepted before parsing")
	}

	first := memoryTestAdvisoryEntry(0)
	second := memoryTestAdvisoryEntry(0)
	if _, err := EncodeMemoryEvidence(MemoryEvidence{Version: 1, Entries: []MemoryEvidenceEntry{first, second}}); err == nil {
		t.Fatal("duplicate engine/model/domain keys were accepted")
	}
	d1 := MemoryDomainDedicatedDevice
	d2 := MemoryDomainHostPhysical
	first.Envelope.Domain = &d1
	second.Envelope.Domain = &d2
	if _, err := EncodeMemoryEvidence(MemoryEvidence{Version: 1, Entries: []MemoryEvidenceEntry{first, second}}); err != nil {
		t.Fatalf("different domains should be distinct keys: %v", err)
	}
}

func memoryTestScopePointer() *MemoryQualificationScope {
	scope := MemoryQualificationScopeOperatorAttestedCurrentRun
	return &scope
}

func TestMemoryEvidenceAllJSONPathsRejectOversize(t *testing.T) {
	// Whitespace alone is invalid JSON even with the size guard removed. Use a
	// valid envelope so rejection proves the bound rather than a parse error.
	valid := memoryTestQualifiedJSON()
	oversize := append(bytes.Repeat([]byte(" "), MaxMemoryEvidenceBytes), []byte(valid)...)
	if _, err := DecodeMemoryEvidence(oversize); err == nil {
		t.Fatal("decoder accepted an oversized valid envelope")
	}
	// encoding/json strips surrounding whitespace before invoking UnmarshalJSON;
	// use internal JSON whitespace to exercise that public path's bound too.
	oversize = []byte(strings.Replace(valid, `"entries":`, `"entries":`+strings.Repeat(" ", MaxMemoryEvidenceBytes), 1))
	var decoded MemoryEvidence
	if err := json.Unmarshal(oversize, &decoded); err == nil {
		t.Fatal("direct JSON decoding accepted an oversized valid envelope")
	}
}

func TestMemoryEvidenceRejectsInvalidUTF8(t *testing.T) {
	valid := []byte(memoryTestQualifiedJSON())
	malformed := bytes.Replace(valid, []byte("llama3:8b"), []byte{0xff}, 1)
	if _, err := DecodeMemoryEvidence(malformed); err == nil {
		t.Fatal("decoder replaced malformed model bytes instead of rejecting them")
	}
	var decoded MemoryEvidence
	if err := json.Unmarshal(malformed, &decoded); err == nil {
		t.Fatal("direct JSON decoding replaced malformed model bytes")
	}
}

func TestMemoryEvidenceRejectsKeysInEveryGroup(t *testing.T) {
	base := memoryTestQualifiedJSON()
	groups := []struct{ name, firstKey string }{
		{"appliedSettings", "epoch"},
		{"envelope", "domain"},
		{"observed", "allocatedBytes"},
		{"qualification", "upperBytes"},
		{"provenance", "procedureId"},
		{"freshness", "sampleAgeMs"},
	}
	for _, group := range groups {
		t.Run(group.name, func(t *testing.T) {
			prefix := `"` + group.name + `":{`
			for _, mutation := range []struct{ name, content string }{
				{"unknown", `"unexpected":null,`},
				{"duplicate", `"` + group.firstKey + `":null,`},
				{"case variant", `"` + strings.ToUpper(group.firstKey) + `":null,`},
			} {
				t.Run(mutation.name, func(t *testing.T) {
					raw := memoryTestMutate(t, base, prefix, prefix+mutation.content)
					memoryTestExpectError(t, raw, mutation.name)
				})
			}
		})
	}
}
