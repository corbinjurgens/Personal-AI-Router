// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
)

func TestEngineBindingsCompatibility(t *testing.T) {
	for _, version := range []int{1, 2} {
		t.Run(fmt.Sprintf("version %d absent loads empty", version), func(t *testing.T) {
			p, err := Parse([]byte(fmt.Sprintf(`{"version":%d}`, version)))
			if err != nil {
				t.Fatal(err)
			}
			data, err := json.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(data, &fields); err != nil {
				t.Fatal(err)
			}
			if string(fields["engineBindings"]) != "[]" || p.Version != SchemaVersion {
				t.Fatalf("bindings=%s version=%d", fields["engineBindings"], p.Version)
			}
			parsed, err := Parse(data)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(parsed, p) {
				t.Fatal("policy changed across round trip")
			}
		})
	}
}

func TestEngineBindingsRejectMalformedInput(t *testing.T) {
	test := func(name, raw string) {
		t.Run(name, func(t *testing.T) {
			if _, err := Parse([]byte(raw)); err == nil {
				t.Fatal("strict parse accepted malformed binding")
			}
			if _, _, err := ParseStored([]byte(raw)); err == nil {
				t.Fatal("stored parse silently repaired binding")
			}
		})
	}
	test("null", `{"engineBindings":null}`)
	test("non-list", `{"engineBindings":{}}`)
	test("null reference", `{"engineBindings":[null]}`)
	test("case variant", `{"EngineBindings":[]}`)
	test("unicode case variant", `{"engineBindingſ":[]}`)
	test("repeated list", `{"engineBindings":[],"engineBindings":[]}`)
	test("repeated member", `{"engineBindings":[{"engine":"llamacpp","model":"a","model":"b"}]}`)
	test("repeated engine", `{"engineBindings":[{"engine":"llamacpp-custom","engine":"llamacpp","model":"a"}]}`)
	test("case variant member", `{"engineBindings":[{"engine":"llamacpp","Model":"a"}]}`)
	test("unexpected member", `{"engineBindings":[{"engine":"llamacpp","model":"a","extra":true}]}`)
	test("invalid engine", `{"engineBindings":[{"engine":"../bad","model":"a"}]}`)
	test("empty model", `{"engineBindings":[{"engine":"llamacpp","model":""}]}`)
	test("non-string model", `{"engineBindings":[{"engine":"llamacpp","model":42}]}`)
	test("tier alias", `{"engineBindings":[{"engine":"llamacpp","model":"weak-local"}]}`)
	test("control token", `{"engineBindings":[{"engine":"llamacpp","model":"a\u0000"}]}`)
	test("invalid UTF8", "{\"engineBindings\":[{\"engine\":\"llamacpp\",\"model\":\"\xff\"}]}")
	test("duplicate model across engines", `{"engineBindings":[{"engine":"llamacpp-custom","model":"a"},{"engine":"llamacpp","model":"a"}]}`)
	test("encoded reference bound", `{"engineBindings":[{"engine":"llamacpp","model":"`+strings.Repeat("x", 230)+`"}]}`)
	entries := make([]string, 257)
	for i := range entries {
		entries[i] = fmt.Sprintf(`{"engine":"llamacpp","model":"m%d"}`, i)
	}
	test("entry bound", `{"engineBindings":[`+strings.Join(entries, ",")+`]}`)
}

func TestEngineBindingsTargetNaming(t *testing.T) {
	naming := func(engine string) (string, bool) {
		switch engine {
		case "llamacpp", "llamacpp-custom":
			return engines.ModelNamingExact, true
		default:
			return "", false
		}
	}
	test := func(name string, refs EngineBindings, served ServedModels, valid bool) {
		t.Run(name, func(t *testing.T) {
			p := Default()
			p.EngineBindings, p.ServedModels = refs, served
			if err := p.ValidateEngineBindings(naming); (err == nil) != valid {
				t.Fatalf("valid=%v: %v", valid, err)
			}
		})
	}
	test("same model on two engines overlaps", EngineBindings{{"llamacpp-custom", "m"}, {"llamacpp", "m"}}, nil, false)
	test("case-sensitive names", EngineBindings{{"llamacpp-custom", "M"}, {"llamacpp", "m"}}, nil, true)
	test("different tags", EngineBindings{{"llamacpp-custom", "m:small"}, {"llamacpp", "m"}}, nil, true)
	test("served exact", EngineBindings{{"llamacpp", "m"}}, ServedModels{{"llamacpp", "m"}}, true)
	test("explicit deny all", EngineBindings{{"llamacpp", "m"}}, ServedModels{}, false)
	test("wrong served engine", EngineBindings{{"llamacpp", "m"}}, ServedModels{{"llamacpp-custom", "m"}}, false)
	test("unknown descriptor", EngineBindings{{"unknown", "m"}}, nil, false)
}

func TestEngineBindingsPeerOwnership(t *testing.T) {
	p := Default()
	p.EngineBindings = EngineBindings{{"llamacpp", "Native/Model"}}
	projection := p.PeerProjection()
	projection.EngineBindings[0].Model = "changed"
	if p.EngineBindings[0].Model != "Native/Model" {
		t.Fatal("projection aliases stored bindings")
	}
	projection = p.PeerProjection()
	raw, err := json.Marshal(projection)
	if err != nil {
		t.Fatal(err)
	}
	merged, err := MergePeerSet(p, raw)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(merged.EngineBindings, p.EngineBindings) {
		t.Fatal("merge lost bindings")
	}
	merged.EngineBindings[0].Model = "other"
	if p.EngineBindings[0].Model != "Native/Model" {
		t.Fatal("merge aliases stored bindings")
	}
}

func TestEngineBindingsStoredSanitationKeepsBindings(t *testing.T) {
	p, counts, err := ParseStored([]byte(`{"engineBindings":[{"engine":"llamacpp","model":"Native/Model"}],"profiles":[{"name":"bad","engine":"llamacpp","model":"bad\u0000"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if counts["profiles"] != 1 || !reflect.DeepEqual(p.EngineBindings, EngineBindings{{"llamacpp", "Native/Model"}}) {
		t.Fatalf("bindings=%+v counts=%v", p.EngineBindings, counts)
	}
}
