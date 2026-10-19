// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"strings"
	"testing"

	"nvpair-shared/enginesettings"
)

func TestInvalidateEngineParams(t *testing.T) {
	test := func(name, engine, operation, reason string, valid bool) {
		t.Run(name, func(t *testing.T) {
			want := InvalidateEngineParams{Engine: engine, OperationID: operation, Reason: reason}
			data, err := json.Marshal(want)
			if err != nil {
				t.Fatal(err)
			}
			var got InvalidateEngineParams
			err = json.Unmarshal(data, &got)
			if (err == nil) != valid {
				t.Fatalf("decode error=%v, valid=%v", err, valid)
			}
			if valid && got != want {
				t.Fatalf("decoded=%+v, want=%+v", got, want)
			}
			if (want.Validate() == nil) != valid {
				t.Fatalf("Validate disagrees with decoding: %+v", want)
			}
		})
	}
	test("built in", "ollama", "switch-1", InvalidationReasonEngineSwitch, true)
	test("variant and request punctuation", "llamacpp-custom", "A._:-09", InvalidationReasonEngineSwitch, true)
	test("maximum request id", "ollama", strings.Repeat("a", enginesettings.MaxSwitchRequestIDBytes), InvalidationReasonEngineSwitch, true)
	test("empty engine", "", "switch-1", InvalidationReasonEngineSwitch, false)
	test("invalid engine syntax", "../ollama", "switch-1", InvalidationReasonEngineSwitch, false)
	test("empty operation", "ollama", "", InvalidationReasonEngineSwitch, false)
	test("oversized operation", "ollama", strings.Repeat("a", enginesettings.MaxSwitchRequestIDBytes+1), InvalidationReasonEngineSwitch, false)
	test("invalid first character", "ollama", "-switch", InvalidationReasonEngineSwitch, false)
	test("non ASCII operation", "ollama", "switch-é", InvalidationReasonEngineSwitch, false)
	test("operation whitespace", "ollama", "switch 1", InvalidationReasonEngineSwitch, false)
	test("operation newline", "ollama", "switch\n", InvalidationReasonEngineSwitch, false)
	test("empty reason", "ollama", "switch-1", "", false)
	test("unknown reason", "ollama", "switch-1", "restart", false)
}

func TestInvalidateEngineStrictJSON(t *testing.T) {
	for name, input := range map[string]string{
		"null":              `null`,
		"array":             `[]`,
		"missing engine":    `{"operationId":"switch-1","reason":"engine-switch"}`,
		"missing operation": `{"engine":"ollama","reason":"engine-switch"}`,
		"missing reason":    `{"engine":"ollama","operationId":"switch-1"}`,
		"null field":        `{"engine":null,"operationId":"switch-1","reason":"engine-switch"}`,
		"wrong type":        `{"engine":"ollama","operationId":1,"reason":"engine-switch"}`,
		"unknown field":     `{"engine":"ollama","operationId":"switch-1","reason":"engine-switch","force":true}`,
		"wrong casing":      `{"Engine":"ollama","operationId":"switch-1","reason":"engine-switch"}`,
		"trailing object":   `{"engine":"ollama","operationId":"switch-1","reason":"engine-switch"}{}`,
	} {
		t.Run(name, func(t *testing.T) {
			var got InvalidateEngineParams
			if err := json.Unmarshal([]byte(input), &got); err == nil {
				t.Fatal("accepted invalid invalidation params")
			}
		})
	}
}

func TestAdmissionSnapshotParams(t *testing.T) {
	if err := json.Unmarshal([]byte(`{}`), &AdmissionSnapshotParams{}); err != nil {
		t.Fatal(err)
	}
	for name, input := range map[string]string{"null": `null`, "array": `[]`, "unknown field": `{"engine":"ollama"}`, "trailing object": `{}{}`} {
		t.Run(name, func(t *testing.T) {
			if err := json.Unmarshal([]byte(input), &AdmissionSnapshotParams{}); err == nil {
				t.Fatal("accepted invalid snapshot params")
			}
		})
	}
}
