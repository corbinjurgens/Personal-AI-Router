// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"encoding/json"
	"fmt"
	"math/big"
	"strconv"
	"strings"
	"testing"
)

func TestModelBufferObservationRounding(t *testing.T) {
	t.Run("integer native bytes remain inside outward rounded reports", func(t *testing.T) {
		for _, bytes := range []uint64{0, 1, 5242, 5243, 1048575, 1048576, 1048577, (1 << 50) - 1048576} {
			value := fmt.Sprintf("%.2f", float64(bytes)/1048576)
			low, high, err := BufferReportInterval(value)
			if err != nil {
				t.Fatal(err)
			}
			lo, err := strconv.ParseUint(low, 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			hi, err := strconv.ParseUint(high, 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			if bytes < lo || bytes > hi {
				t.Fatalf("%d source bytes excluded from [%s,%s] for %s", bytes, low, high, value)
			}
		}
	})
	for _, value := range []string{"0.00", "0.01", "1.00", "1073741823.99"} {
		t.Run(value, func(t *testing.T) {
			low, high, err := BufferReportInterval(value)
			if err != nil {
				t.Fatal(err)
			}
			q, ok := new(big.Rat).SetString(value)
			if !ok {
				t.Fatal("invalid fixture")
			}
			q.Mul(q, big.NewRat(1048576, 1))
			lo, ok := new(big.Rat).SetString(low)
			if !ok {
				t.Fatal("invalid lower endpoint")
			}
			hi, ok := new(big.Rat).SetString(high)
			if !ok || lo.Cmp(q) > 0 || hi.Cmp(q) < 0 {
				t.Fatalf("interval [%s,%s] excludes %s", low, high, q)
			}
		})
	}
	low, high, err := BufferReportInterval("0.00")
	if err != nil || low != "0" || high != "5243" {
		t.Fatalf("zero report: [%s,%s], %v", low, high, err)
	}
	for _, value := range []string{"-1.00", "NaN", "Inf", "1", "1.000", "01.00", "1.e0", "1073741824.00", "999999999999999999999.00"} {
		t.Run("reject "+value, func(t *testing.T) {
			if _, _, err := BufferReportInterval(value); err == nil {
				t.Fatal("accepted invalid report")
			}
		})
	}
}

func TestModelBufferObservationWire(t *testing.T) {
	value := BufferObservation{Status: "unknown", Reason: "observation-omitted", ConfigurationKey: strings.Repeat("a", 64)}
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	var decoded BufferObservation
	if err := DecodeStrict(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if err := decoded.Validate(); err != nil {
		t.Fatal(err)
	}
	if decoded.ConfigurationKey != value.ConfigurationKey {
		t.Fatal("configuration key lost in roundtrip")
	}
	invalid := decoded
	invalid.ConfigurationKey = "not-a-digest"
	if err := invalid.Validate(); err == nil {
		t.Fatal("invalid key accepted")
	}
	legacy := decoded
	legacy.ConfigurationKey = ""
	if err := legacy.Validate(); err != nil {
		t.Fatalf("keyless diagnostic sample rejected: %v", err)
	}
	if err := DecodeStrict([]byte(`{"status":"unknown","current":false,"rawLog":"secret"}`), &decoded); err == nil {
		t.Fatal("accepted unknown field")
	}
	value.Reasons = []string{"missing-boundary"}
	cloned := value.Clone()
	cloned.Reasons[0] = "unsupported-build"
	if value.Reasons[0] != "missing-boundary" {
		t.Fatal("clone aliases reasons")
	}
	request := RequestedLoad{GPU: &GPU{Mode: "cpu"}}
	value.CapturedOptions = &request
	value.RequestedOptions = &request
	value.OptionUnknowns = []string{"mmap"}
	cloned = value.Clone()
	cloned.CapturedOptions.GPU.Mode = "native-default"
	cloned.RequestedOptions.GPU.Mode = "layers"
	cloned.OptionUnknowns[0] = "load-mode"
	if value.CapturedOptions.GPU.Mode != "cpu" || value.RequestedOptions.GPU.Mode != "cpu" || value.OptionUnknowns[0] != "mmap" {
		t.Fatal("clone aliases captured or requested options")
	}
	result := InspectResult{Engine: "llamacpp", ModelID: "exact", Status: "missing", Estimate: Estimate{Status: "unknown"}, LoadObservation: &value}
	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeInspectResult(encoded); err != nil {
		t.Fatal(err)
	}
}

func TestModelBufferObservationBounds(t *testing.T) {
	t.Run("comparison references its frozen baseline and checked signed interval", func(t *testing.T) {
		o := BufferObservation{Status: "partial", Baseline: []BufferBaseline{{Component: "weights", Status: "estimated", Bytes: "100"}}, Comparisons: []BufferComparison{{Component: "weights", Status: "compatible", EstimatedBytes: "100", LowerBytes: "90", UpperBytes: "110", DeltaLowerBytes: "-10", DeltaUpperBytes: "10"}}}
		if err := o.Validate(); err != nil {
			t.Fatal(err)
		}
		for _, mutate := range []func(*BufferObservation){
			func(o *BufferObservation) { o.Baseline[0].Status = "partial" },
			func(o *BufferObservation) { o.Comparisons[0].EstimatedBytes = "101" },
			func(o *BufferObservation) { o.Comparisons[0].DeltaUpperBytes = "11" },
			func(o *BufferObservation) { o.Comparisons[0].LowerBytes = "111" },
			func(o *BufferObservation) { o.Comparisons[0].Status = "unknown" },
			func(o *BufferObservation) { o.Baseline = append(o.Baseline, o.Baseline[0]) },
		} {
			candidate := o.Clone()
			mutate(&candidate)
			if err := candidate.Validate(); err == nil {
				t.Fatal("accepted inconsistent frozen comparison")
			}
		}
	})
	t.Run("unsupported magnitude retains decimal without a fabricated interval", func(t *testing.T) {
		for _, value := range []string{"1073741824.00", "999999999999999999999.00"} {
			o := BufferObservation{Status: "partial", Rows: []BufferReportRow{{Component: "model", Label: "CPU", Phase: "load", Sequence: "1", Stream: "stdout", Role: "model", ReportedMiB: value, Unit: "MiB", Precision: "0.01", SourceKind: "log-scraped", IntervalStatus: "unknown", Coverage: "unknown"}}}
			if err := o.Validate(); err != nil {
				t.Fatal(err)
			}
			o.Rows[0].LowerBytes = "0"
			if err := o.Validate(); err == nil {
				t.Fatal("unsupported report acquired a zero interval")
			}
		}
	})
	for _, label := range []string{"CPU", "CPU_Mapped", "CUDA_Host", "CUDA0", "CUDA63", "MTL0", "MTL63_Private", "MTL1_Mapped"} {
		if !validBufferLabel(label) {
			t.Fatalf("rejected %s", label)
		}
	}
	for _, label := range []string{"Metal0", "CUDA64", "CUDA00", "CUDA-1", "MTL64", "MTL0_Private_Mapped", "CPU/secret"} {
		if validBufferLabel(label) {
			t.Fatalf("accepted %s", label)
		}
	}
	if err := (BufferObservation{Status: "unknown", Reason: strings.Repeat("x", 257)}).Validate(); err == nil {
		t.Fatal("accepted oversized identifier")
	}
	if err := (BufferObservation{Status: "unknown", Reasons: make([]string, 17)}).Validate(); err == nil {
		t.Fatal("accepted too many reasons")
	}
	if err := (BufferObservation{Status: "unknown", Current: true}).Validate(); err == nil {
		t.Fatal("unknown became current")
	}
}
