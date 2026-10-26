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
	test("repeated member", `{"engineBindings":[{"engine":"ollama","model":"a","model":"b"}]}`)
	test("repeated engine", `{"engineBindings":[{"engine":"ollama","engine":"llamacpp","model":"a"}]}`)
	test("case variant member", `{"engineBindings":[{"engine":"ollama","Model":"a"}]}`)
	test("unexpected member", `{"engineBindings":[{"engine":"ollama","model":"a","extra":true}]}`)
	test("invalid engine", `{"engineBindings":[{"engine":"../bad","model":"a"}]}`)
	test("empty model", `{"engineBindings":[{"engine":"ollama","model":""}]}`)
	test("non-string model", `{"engineBindings":[{"engine":"ollama","model":42}]}`)
	test("tier alias", `{"engineBindings":[{"engine":"ollama","model":"weak-local"}]}`)
	test("control token", `{"engineBindings":[{"engine":"ollama","model":"a\u0000"}]}`)
	test("invalid UTF8", "{\"engineBindings\":[{\"engine\":\"ollama\",\"model\":\"\xff\"}]}")
	test("duplicate model across engines", `{"engineBindings":[{"engine":"ollama","model":"a"},{"engine":"llamacpp","model":"a"}]}`)
	test("encoded reference bound", `{"engineBindings":[{"engine":"ollama","model":"`+strings.Repeat("x", 230)+`"}]}`)
	entries := make([]string, 257)
	for i := range entries {
		entries[i] = fmt.Sprintf(`{"engine":"ollama","model":"m%d"}`, i)
	}
	test("entry bound", `{"engineBindings":[`+strings.Join(entries, ",")+`]}`)
}

func TestEngineBindingsTargetNaming(t *testing.T) {
	naming := func(engine string) (string, bool) {
		switch engine {
		case "ollama", "custom-tagged":
			return engines.ModelNamingImpliedLatestTag, true
		case "llamacpp":
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
	test("latest overlaps exact base", EngineBindings{{"ollama", "m:latest"}, {"llamacpp", "m"}}, nil, false)
	test("reverse order overlaps", EngineBindings{{"llamacpp", "m"}, {"ollama", "m:latest"}}, nil, false)
	test("custom descriptor overlaps", EngineBindings{{"custom-tagged", "m:latest"}, {"llamacpp", "m"}}, nil, false)
	test("same engine aliases overlap", EngineBindings{{"ollama", "m"}, {"ollama", "m:latest"}}, nil, false)
	test("case-sensitive names", EngineBindings{{"ollama", "M"}, {"llamacpp", "m"}}, nil, true)
	test("different tags", EngineBindings{{"ollama", "m:small"}, {"llamacpp", "m"}}, nil, true)
	test("served normalized", EngineBindings{{"ollama", "m:latest"}}, ServedModels{{"ollama", "m"}}, true)
	test("binding normalized", EngineBindings{{"ollama", "m"}}, ServedModels{{"ollama", "m:latest"}}, true)
	test("explicit deny all", EngineBindings{{"ollama", "m"}}, ServedModels{}, false)
	test("wrong served engine", EngineBindings{{"ollama", "m"}}, ServedModels{{"llamacpp", "m"}}, false)
	test("unknown descriptor", EngineBindings{{"unknown", "m"}}, nil, false)
}

func TestEngineBindingsPeerOwnership(t *testing.T) {
	p := Default()
	p.EngineBindings = EngineBindings{{"ollama", "Native/Model"}}
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
	p, counts, err := ParseStored([]byte(`{"engineBindings":[{"engine":"ollama","model":"Native/Model"}],"profiles":[{"name":"bad","engine":"ollama","model":"bad\u0000"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if counts["profiles"] != 1 || !reflect.DeepEqual(p.EngineBindings, EngineBindings{{"ollama", "Native/Model"}}) {
		t.Fatalf("bindings=%+v counts=%v", p.EngineBindings, counts)
	}
}
