// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"nvpair-shared/modelconfig"
)

func strataPlanFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "memoryplan", "strata", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func strataPlanBinding(t *testing.T, b []byte) strataMemoryPlanBinding {
	t.Helper()
	var response strataMemoryPlanResponse
	if err := json.Unmarshal(b, &response); err != nil {
		t.Fatal(err)
	}
	return strataMemoryPlanBinding{Request: strataMemoryPlanRequest{SchemaVersion: 1, Model: strataMemoryPlanModelRequest{ID: response.Model.ID}, Config: response.RequestedConfig, Extensions: strataMemoryPlanRequestExtensions{Strata: response.Extensions.Strata.Requested}}, ExecutableSHA256: response.Engine.ExecutableSHA256, Build: response.Engine.Build, Inputs: response.Extensions.Strata.Inputs, RuntimeSourceSHA256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
}

func TestMemoryPlanStrataRecordings(t *testing.T) {
	for _, tc := range []struct{ name, status string }{
		{"explicit-not-evaluated", "not_evaluated"}, {"indeterminate", "indeterminate"}, {"does-not-fit", "does_not_fit"}, {"minimum-layout-does-not-fit", "does_not_fit"}, {"policy-threshold-shortfall", "does_not_fit"}, {"conditional-allowances", "fits_within_allowances"}, {"after-runtime-indeterminate", "indeterminate"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := strataPlanFixture(t, tc.name)
			binding := strataPlanBinding(t, b)
			// Budgets are private protocol test inputs; production exact-setting requests omit them.
			binding.Request.Budget = fixtureStrataBudget(t, b)
			p, err := decodeStrataMemoryPlan(b, 0, binding)
			if err != nil {
				t.Fatal(err)
			}
			if p.Verdict.Status != tc.status || p.Identity.Cacheable || p.Identity.ContentSHA256 != "" {
				t.Fatalf("incorrect status or assurance: %+v", p)
			}
			if len(p.Devices) != 1 || len(p.Devices[0].Groups) != 6 || p.HostCommit.Kind != "host_commit" {
				t.Fatal("lost resource categories")
			}
			if p.Devices[0].Groups[1].Bytes != "" || p.Devices[0].Groups[1].Quality != "unknown" {
				t.Fatal("invented partition")
			}
			if p.Devices[0].PricedTotalBytes != "130" || p.HostRAM.PricedTotalBytes != "80" || p.HostCommit.PricedTotalBytes != "210" {
				t.Fatal("counted overlapping views")
			}
			if tc.name == "policy-threshold-shortfall" && (p.Devices[0].RequiredWithAllowancesBytes != "200" || p.Devices[0].ShortfallBytes != "50") {
				t.Fatal("policy threshold was added")
			}
			if tc.name == "minimum-layout-does-not-fit" && (p.Resolved != nil || p.ConfigurationComplete) {
				t.Fatal("invented resolved config")
			}
			if err := p.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func fixtureStrataBudget(t *testing.T, b []byte) *strataMemoryPlanBudget {
	t.Helper()
	var r strataMemoryPlanResponse
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	if r.BudgetBasis == nil {
		return nil
	}
	return &strataMemoryPlanBudget{Basis: *r.BudgetBasis, Devices: []strataMemoryPlanDeviceBudget{{ID: 0, Bytes: *r.Resources.Devices[0].BudgetBytes}}, HostRAMBytes: *r.Resources.HostRAM.BudgetBytes, HostCommitBytes: *r.Resources.HostCommit.BudgetBytes}
}

func TestMemoryPlanStrataExitFailures(t *testing.T) {
	for _, name := range []string{"invalid-input", "cleanup-unconfirmed"} {
		t.Run(name, func(t *testing.T) {
			_, err := decodeStrataMemoryPlan(strataPlanFixture(t, name), 2, strataMemoryPlanBinding{})
			if err == nil {
				t.Fatal("accepted failure")
			}
			if name == "cleanup-unconfirmed" && !errors.Is(err, errStrataMemoryPlanCleanup) {
				t.Fatal("lost cleanup refusal")
			}
		})
	}
}

// Mutate decoded JSON tokens, never float64, and keep the expected binding
// independent of each hostile response.
func TestMemoryPlanStrataHostileResponses(t *testing.T) {
	original := strataPlanFixture(t, "explicit-not-evaluated")
	var compact bytes.Buffer
	if err := json.Compact(&compact, original); err != nil {
		t.Fatal(err)
	}
	original = compact.Bytes()
	binding := strataPlanBinding(t, original)
	test := func(name string, mutate func(map[string]json.RawMessage)) {
		t.Run(name, func(t *testing.T) {
			var obj map[string]json.RawMessage
			if err := json.Unmarshal(original, &obj); err != nil {
				t.Fatal(err)
			}
			mutate(obj)
			b, err := json.Marshal(obj)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := decodeStrataMemoryPlan(b, 0, binding); err == nil {
				t.Fatal("accepted hostile response")
			}
		})
	}
	test("wrong model", func(o map[string]json.RawMessage) {
		o["model"] = json.RawMessage(`{"id":"other","content_sha256":null,"asserted_content_sha256":null,"identity_basis":"unverified"}`)
	})
	test("missing required null", func(o map[string]json.RawMessage) { delete(o, "budget_basis") })
	test("missing required false", func(o map[string]json.RawMessage) { delete(o, "cacheable") })
	test("schema fraction", func(o map[string]json.RawMessage) { o["schema_version"] = json.RawMessage("1.0") })
	test("schema boolean", func(o map[string]json.RawMessage) { o["schema_version"] = json.RawMessage("true") })
	test("invented verified content", func(o map[string]json.RawMessage) {
		o["model"] = json.RawMessage(`{"id":"synthetic","content_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","asserted_content_sha256":null,"identity_basis":"unverified"}`)
	})
	replace := func(name, old, new string) {
		test(name, func(o map[string]json.RawMessage) {
			changed := false
			for k, v := range o {
				changed = changed || bytes.Contains(v, []byte(old))
				o[k] = bytes.ReplaceAll(v, []byte(old), []byte(new))
			}
			if !changed {
				t.Fatalf("mutation %q matched no token", name)
			}
		})
	}
	replace("wrong executable digest", binding.ExecutableSHA256, strings.Repeat("b", 64))
	replace("changed explicit setting", `"context":8192`, `"context":4096`)
	replace("wrong backend", `"backend":"cuda"`, `"backend":"cpu"`)
	replace("unknown component", `"weights_canonical":10`, `"unreviewed_component":10`)
	replace("component sum mismatch", `"weights_canonical":10`, `"weights_canonical":11`)
	replace("negative byte", `"weights_canonical":10`, `"weights_canonical":-1`)
	replace("fractional byte", `"weights_canonical":10`, `"weights_canonical":10.5`)
	replace("overflow byte", `"weights_canonical":10`, `"weights_canonical":18446744073709551616`)
	replace("unknown cost cannot disappear", "CUDA context/modules, graph capture, CPU TLS/request scratch, driver and OS residency", "new unclassified native cost")
	replace("wrong stat identity", `"file_id":[1,1]`, `"file_id":[1,99]`)
	replace("wrong category quality", `"quality":"exact"`, `"quality":"measured"`)
	replace("missing category", `"weights":`, `"invented":`)
	replace("optional field spelling", `"kv_type":`, `"KV_TYPE":`)
	replace("allocated exceeds total", `"allocated_bytes":120`, `"allocated_bytes":18446744073709551615`)
	replace("invented partition", `"kv_cache":{"bytes":null,"quality":"unknown"`, `"kv_cache":{"bytes":0,"quality":"exact"`)
	replace("missing nullable bytes", `"bytes":null,`, ``)
	replace("additive peak", `"additive":false`, `"additive":true`)
	replace("private query option", `"--price-only"`, `"--serve"`)
	replace("private query environment", `"CUDA_VISIBLE_DEVICES":"0"`, `"CUDA_VISIBLE_DEVICES":"1"`)
	replace("native settings null integer", `"auto_cache":0`, `"auto_cache":null`)
	replace("native vector null integer", `"layer_blobs":[256]`, `"layer_blobs":[null]`)
	replace("invented physical mapping", `"id":0,"backend":"cuda"`, `"id":0,"backend":"cuda","physical_id":"invented"`)
	t.Run("duplicate keys", func(t *testing.T) {
		b := bytes.Replace(original, []byte(`"kind":`), []byte(`"kind":"engine_memory_plan","kind":`), 1)
		if _, err := decodeStrataMemoryPlan(b, 0, binding); err == nil {
			t.Fatal("accepted duplicate")
		}
	})
	t.Run("multiple frames", func(t *testing.T) {
		if _, err := decodeStrataMemoryPlan(append(append([]byte{}, original...), original...), 0, binding); err == nil {
			t.Fatal("accepted frames")
		}
	})
	t.Run("oversized", func(t *testing.T) {
		if _, err := decodeStrataMemoryPlan(bytes.Repeat([]byte(" "), modelconfig.MaxPayloadBytes+1), 0, binding); err == nil {
			t.Fatal("accepted oversized")
		}
	})
	t.Run("exit zero error", func(t *testing.T) {
		if _, err := decodeStrataMemoryPlan(strataPlanFixture(t, "invalid-input"), 0, binding); err == nil {
			t.Fatal("accepted error frame")
		}
	})
	t.Run("nonzero valid plan", func(t *testing.T) {
		if _, err := decodeStrataMemoryPlan(original, 2, binding); err == nil {
			t.Fatal("accepted nonzero plan")
		}
	})
}

func TestMemoryPlanStrataPrivateTransportSize(t *testing.T) {
	original := strataPlanFixture(t, "explicit-not-evaluated")
	binding := strataPlanBinding(t, original)
	large := append(append([]byte{}, original...), bytes.Repeat([]byte(" "), modelconfig.MaxPayloadBytes)...)
	if _, err := decodeStrataMemoryPlan(large, 0, binding); err != nil {
		t.Fatalf("valid private frame above public ceiling: %v", err)
	}
}

func TestMemoryPlanStrataGiBDecimal(t *testing.T) {
	original := strataPlanFixture(t, "explicit-not-evaluated")
	b := bytes.ReplaceAll(original, []byte("1073741824"), []byte("1073741825"))
	b = bytes.Replace(b, []byte("\"--resident-budget-gib\",\n          \"1\""), []byte("\"--resident-budget-gib\",\n          \"1.000000000931322574615478516\""), 1)
	if bytes.Equal(b, original) || !bytes.Contains(b, []byte("1.000000000931322574615478516")) {
		t.Fatal("fixture mutation failed")
	}
	if _, err := decodeStrataMemoryPlan(b, 0, strataPlanBinding(t, b)); err != nil {
		t.Fatalf("valid Strata decimal: %v", err)
	}
}

func TestMemoryPlanStrataLargeTotals(t *testing.T) {
	original := strataPlanFixture(t, "explicit-not-evaluated")
	binding := strataPlanBinding(t, original)
	var buf bytes.Buffer
	if err := json.Compact(&buf, original); err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()
	// Synthetic once-counted totals fill uint64 without overflow; this is protocol
	// arithmetic, not an engine allocation prediction.
	for _, change := range [][2]string{
		{`"weights_canonical":10`, `"weights_canonical":18446744073709551415`},
		{`"priced_total_bytes":130`, `"priced_total_bytes":18446744073709551535`},
		{`"required_with_allowances_bytes":130`, `"required_with_allowances_bytes":18446744073709551535`},
		{`"allocated_bytes":120`, `"allocated_bytes":18446744073709551525`},
		{`"vram_required":130`, `"vram_required":18446744073709551535`},
		{`"priced_total_bytes":210`, `"priced_total_bytes":18446744073709551615`},
		{`"required_with_allowances_bytes":210`, `"required_with_allowances_bytes":18446744073709551615`},
		{`"commit_required":210`, `"commit_required":18446744073709551615`},
		{`"name":"combined_capacity_model","bytes":210`, `"name":"combined_capacity_model","bytes":18446744073709551615`},
		{`"weights":{"bytes":40`, `"weights":{"bytes":18446744073709551445`},
	} {
		b = bytes.ReplaceAll(b, []byte(change[0]), []byte(change[1]))
	}
	p, err := decodeStrataMemoryPlan(b, 0, binding)
	if err != nil {
		t.Fatal(err)
	}
	if p.HostCommit.PricedTotalBytes != "18446744073709551615" || p.Devices[0].PricedTotalBytes != "18446744073709551535" {
		t.Fatal("large total rounded")
	}
}

func TestMemoryPlanStrataNonzeroAllowance(t *testing.T) {
	original := strataPlanFixture(t, "conditional-allowances")
	var buf bytes.Buffer
	if err := json.Compact(&buf, original); err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()
	for _, change := range [][2]string{
		{`"device_runtime_context":0`, `"device_runtime_context":5`},
		{`"required_with_allowances_bytes":130`, `"required_with_allowances_bytes":135`},
		{`"devices":[{"id":0,"bytes":1000}]`, `"devices":[{"id":0,"bytes":995}]`},
		{`"--plan-vram-free-bytes","1000"`, `"--plan-vram-free-bytes","995"`},
	} {
		b = bytes.ReplaceAll(b, []byte(change[0]), []byte(change[1]))
	}
	binding := strataPlanBinding(t, b)
	binding.Request.Budget = fixtureStrataBudget(t, b)
	p, err := decodeStrataMemoryPlan(b, 0, binding)
	if err != nil {
		t.Fatal(err)
	}
	if p.Devices[0].RequiredWithAllowancesBytes != "135" || p.Devices[0].PricedTotalBytes != "130" {
		t.Fatal("allowance counted incorrectly")
	}
}

func TestMemoryPlanStrataExactInput(t *testing.T) {
	b := strataPlanFixture(t, "explicit-not-evaluated")
	binding := strataPlanBinding(t, b)
	cfg, err := binding.Request.Config.neutral()
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := encodeStrataMemoryPlanRequest(cfg, binding.Request.Model, binding.Request.Extensions.Strata)
	if err != nil {
		t.Fatal(err)
	}
	var got strataMemoryPlanRequest
	if err := strataPlanDecode(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, binding.Request) || got.Budget != nil || len(got.Extensions.Strata.Allowances) != 0 {
		t.Fatalf("request changed: %+v", got)
	}
	reject := func(name string, edit func(*modelconfig.MemoryPlanConfig, *strataMemoryPlanOptions)) {
		t.Run(name, func(t *testing.T) {
			c := cfg.Clone()
			s := binding.Request.Extensions.Strata
			edit(&c, &s)
			if _, err := encodeStrataMemoryPlanRequest(c, binding.Request.Model, s); err == nil {
				t.Fatal("accepted unsupported input")
			}
		})
	}
	reject("automatic prefill", func(c *modelconfig.MemoryPlanConfig, _ *strataMemoryPlanOptions) {
		c.Microbatch = &modelconfig.MemoryPlanMicrobatch{Mode: "auto"}
	})
	reject("automatic cache", func(_ *modelconfig.MemoryPlanConfig, s *strataMemoryPlanOptions) {
		s.CacheSlots = strataMemoryPlanChoice{Auto: "auto"}
	})
	reject("missing device", func(c *modelconfig.MemoryPlanConfig, _ *strataMemoryPlanOptions) { c.Devices = nil })
	reject("missing batch", func(c *modelconfig.MemoryPlanConfig, _ *strataMemoryPlanOptions) { c.BatchSize = nil })
	reject("missing parallelism", func(c *modelconfig.MemoryPlanConfig, _ *strataMemoryPlanOptions) { c.ParallelSlots = nil })
	reject("missing KV", func(c *modelconfig.MemoryPlanConfig, _ *strataMemoryPlanOptions) { c.KVFormat = "" })
	reject("physical selector", func(c *modelconfig.MemoryPlanConfig, _ *strataMemoryPlanOptions) {
		c.Devices[0].Selector = "GPU-unproven"
	})
	reject("parallel slots", func(c *modelconfig.MemoryPlanConfig, _ *strataMemoryPlanOptions) { v := int32(2); c.ParallelSlots = &v })
	reject("caller allowance", func(_ *modelconfig.MemoryPlanConfig, s *strataMemoryPlanOptions) {
		s.Allowances = map[string]uint64{"device_runtime_context": 0}
	})
	reject("credential environment", func(_ *modelconfig.MemoryPlanConfig, s *strataMemoryPlanOptions) {
		s.Environment = map[string]string{"STRATA_API_KEY": "FAKE-secret"}
	})
	reject("unpriced spec", func(_ *modelconfig.MemoryPlanConfig, s *strataMemoryPlanOptions) { s.Spec = 3 })
	reject("prefill borrowing", func(_ *modelconfig.MemoryPlanConfig, s *strataMemoryPlanOptions) { s.NoPrefillBorrow = false })
	reject("fractional MiB reserve", func(_ *modelconfig.MemoryPlanConfig, s *strataMemoryPlanOptions) { s.VRAMReserveBytes = 1 })
}

func TestMemoryPlanStrataUnknownAndZero(t *testing.T) {
	b := strataPlanFixture(t, "conditional-allowances")
	binding := strataPlanBinding(t, b)
	binding.Request.Budget = fixtureStrataBudget(t, b)
	p, err := decodeStrataMemoryPlan(b, 0, binding)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Devices[0].CallerAllowances) != 3 || p.Devices[0].CallerAllowances[0].Bytes != "0" || p.Devices[0].Groups[1].Bytes != "" || p.HostCommit.AllocatedBytes != "" || p.Devices[0].MappedViewBytes != "0" {
		t.Fatal("lost absent/zero distinction")
	}
	if strataPlanBytes(nil) != "" {
		t.Fatal("unknown became zero")
	}
	maximum := uint64(math.MaxUint64)
	if strataPlanBytes(&maximum) != "18446744073709551615" {
		t.Fatal("uint64 lost precision")
	}
	var q strataMemoryPlanQuantity
	if err := strataPlanDecode([]byte(`{"bytes":18446744073709551615,"quality":"exact","basis":"synthetic"}`), &q); err != nil || q.Bytes == nil || *q.Bytes != maximum {
		t.Fatalf("uint64 decode: %+v %v", q, err)
	}
}

func TestMemoryPlanStrataUnclassifiedCost(t *testing.T) {
	original := strataPlanFixture(t, "explicit-not-evaluated")
	binding := strataPlanBinding(t, original)
	var buf bytes.Buffer
	if err := json.Compact(&buf, original); err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()
	b = bytes.ReplaceAll(b, []byte("CUDA context/modules, graph capture, CPU TLS/request scratch, driver and OS residency"), []byte("new native unknown"))
	for _, cost := range []string{"device_transient_peak", "host_ram_transient_peak", "host_commit_transient_peak"} {
		b = bytes.ReplaceAll(b, []byte(`"`+cost+`"]`), []byte(`"`+cost+`","unclassified_native_costs"]`))
	}
	p, err := decodeStrataMemoryPlan(b, 0, binding)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range []modelconfig.MemoryPlanResource{p.Devices[0], p.HostRAM, p.HostCommit} {
		if !slices.Contains(r.UncoveredCosts, "unclassified_native_costs") {
			t.Fatal("unfamiliar unknown cost dropped")
		}
	}
}

func TestMemoryPlanStrataPrivateDataStaysPrivate(t *testing.T) {
	b := strataPlanFixture(t, "explicit-not-evaluated")
	p, err := decodeStrataMemoryPlan(b, 0, strataPlanBinding(t, b))
	if err != nil {
		t.Fatal(err)
	}
	wire, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	for _, private := range []string{"/synthetic/strata", "--price-only", "CUDA_VISIBLE_DEVICES", "weights_canonical", "extensions", "native_components", "STRATA_"} {
		if bytes.Contains(wire, []byte(private)) {
			t.Fatalf("neutral plan leaked %q", private)
		}
	}
	raw := strataPlanFixture(t, "invalid-input")
	raw = bytes.ReplaceAll(raw, []byte("Memory planning refused."), []byte("/private/FAKE-secret"))
	_, err = decodeStrataMemoryPlan(raw, 2, strataMemoryPlanBinding{})
	if !errors.Is(err, errStrataMemoryPlan) || strings.Contains(err.Error(), "FAKE-secret") {
		t.Fatal("unsafe failure diagnostic")
	}
}
