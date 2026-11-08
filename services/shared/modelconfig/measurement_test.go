// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestMeasurementDiagnosticProjection(t *testing.T) {
	semantic := strings.Repeat("c", 64)
	base := diagnosticFixture().DiagnosticIdentity
	want := MeasurementFromDiagnostic(semantic, base)
	if !want.Complete || want.Validate() != nil {
		t.Fatalf("complete projection: %+v", want)
	}
	for _, tc := range []struct {
		name string
		edit func(*DiagnosticIdentity)
	}{
		{"CPU", func(i *DiagnosticIdentity) { i.CPUModel = "replacement" }},
		{"RAM", func(i *DiagnosticIdentity) { i.TotalRAMBytes = "137438953472" }},
		{"unused GPU name", func(i *DiagnosticIdentity) { i.GPUs[1].Name = "replacement" }},
		{"GPU capacity", func(i *DiagnosticIdentity) { i.GPUs[1].TotalMemoryBytes = "34359738368" }},
		{"GPU physical identity", func(i *DiagnosticIdentity) { i.GPUs[1].UUID = "replacement"; i.Placement[2].DeviceID = "replacement" }},
		{"build", func(i *DiagnosticIdentity) { i.NativeBuildID = "sha256:" + strings.Repeat("d", 64) }},
		{"realized context", func(i *DiagnosticIdentity) { i.PerSlotContext = 2048 }},
		{"realized slots", func(i *DiagnosticIdentity) { i.Slots = 2 }},
		{"realized placement", func(i *DiagnosticIdentity) { i.Placement = i.Placement[:1] }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			i := base.Clone()
			tc.edit(&i)
			got := MeasurementFromDiagnostic(semantic, i)
			if !got.Complete || got == want {
				t.Fatalf("speed-affecting change reused identity: %+v", got)
			}
		})
	}
	t.Run("order and advisory keys", func(t *testing.T) {
		i := base.Clone()
		slices.Reverse(i.GPUs)
		slices.Reverse(i.Placement)
		i.ConfigurationKey = strings.Repeat("e", 64)
		i.BuildInfo = "same content reported differently"
		if got := MeasurementFromDiagnostic(semantic, i); got != want {
			t.Fatalf("nonsemantic change: %+v != %+v", got, want)
		}
	})
	if got := MeasurementFromDiagnostic(strings.Repeat("d", 64), base); !got.Complete || got == want {
		t.Fatal("semantic configuration omitted")
	}
}

func TestMeasurementUnknownInputs(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*DiagnosticIdentity)
	}{
		{"CPU", func(i *DiagnosticIdentity) { i.CPUModel = "" }},
		{"RAM", func(i *DiagnosticIdentity) { i.TotalRAMBytes = "0" }},
		{"GPU capacity", func(i *DiagnosticIdentity) { i.GPUs[1].TotalMemoryBytes = "0" }},
		{"build", func(i *DiagnosticIdentity) { i.NativeBuildID = "" }},
		{"realization", func(i *DiagnosticIdentity) { i.PerSlotContext = 0 }},
		{"placement", func(i *DiagnosticIdentity) { i.Placement = nil }},
		{"live build readback", func(i *DiagnosticIdentity) { i.BuildInfo = "" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			i := diagnosticFixture().DiagnosticIdentity
			tc.edit(&i)
			if got := MeasurementFromDiagnostic(strings.Repeat("c", 64), i); got != (MeasurementIdentity{}) {
				t.Fatalf("unknown inputs gained equality: %+v", got)
			}
		})
	}
	if got := MeasurementFromDiagnostic("", diagnosticFixture().DiagnosticIdentity); got.Complete {
		t.Fatal("missing semantics completed")
	}
}

func TestMeasurementInspectionWireCloneAndPrivacy(t *testing.T) {
	identity := MeasurementIdentity{Digest: strings.Repeat("a", 64), Complete: true}
	result := InspectResult{Engine: "llamacpp", ModelID: "model", Status: "resolved", Artifacts: diagnosticFixture().Artifacts, Estimate: Estimate{Status: "unknown"}, MeasurementIdentity: &identity}
	raw, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeInspectResult(raw)
	if err != nil || decoded.MeasurementIdentity == nil || *decoded.MeasurementIdentity != identity {
		t.Fatalf("decode: %+v, %v", decoded, err)
	}
	clone := decoded.Clone()
	clone.MeasurementIdentity.Digest = strings.Repeat("b", 64)
	if decoded.MeasurementIdentity.Digest != strings.Repeat("a", 64) {
		t.Fatal("measurement clone aliased")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	var metadata map[string]json.RawMessage
	if err := json.Unmarshal(fields["measurementIdentity"], &metadata); err != nil {
		t.Fatal(err)
	}
	keys := make([]string, 0, len(metadata))
	for key := range metadata {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	if !reflect.DeepEqual(keys, []string{"complete", "digest"}) {
		t.Fatalf("private inputs exposed: %v", keys)
	}
	for _, invalid := range []MeasurementIdentity{{Digest: "partial", Complete: true}, {Digest: strings.Repeat("a", 64)}} {
		result.MeasurementIdentity = &invalid
		raw, err := json.Marshal(result)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := DecodeInspectResult(raw); err == nil {
			t.Fatalf("accepted incomplete claim: %+v", invalid)
		}
	}
	result.MeasurementIdentity = nil
	raw, err = json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err = DecodeInspectResult(raw)
	if err != nil || decoded.MeasurementIdentity != nil {
		t.Fatalf("optional absence: %+v, %v", decoded, err)
	}
}
