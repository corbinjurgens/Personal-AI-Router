// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestEngineBindingOffersOptionalValidation(t *testing.T) {
	valid := `{"version":1,"models":[{"engine":"ollama","model":"fixture"}]}`
	test := func(name, extension string, supported bool) {
		t.Run(name, func(t *testing.T) {
			var adv Advertisement
			if err := json.Unmarshal([]byte(`{"revision":"1","roles":{"weak":{"engine":"ollama","model":"actual"}},`+extension+`}`), &adv); err != nil {
				t.Fatal(err)
			}
			if (adv.EngineBindingOffers != nil) != supported || adv.Roles["weak"].Model != "actual" {
				t.Fatalf("offers=%+v canonical role=%+v", adv.EngineBindingOffers, adv.Roles["weak"])
			}
		})
	}
	test("supported", `"engineBindingOffers":`+valid, true)
	test("unsupported version", `"engineBindingOffers":{"version":2,"models":[]}`, false)
	test("malformed optional data", `"engineBindingOffers":5`, false)
	test("null", `"engineBindingOffers":null`, false)
	test("case variant", `"EngineBindingOffers":`+valid, false)
	test("duplicate extension", `"engineBindingOffers":`+valid+`,"engineBindingOffers":`+valid, false)
	test("duplicate reference", `"engineBindingOffers":{"version":1,"models":[{"engine":"ollama","model":"a"},{"engine":"lmstudio","model":"a"}]}`, false)
	test("unknown field", `"engineBindingOffers":{"version":1,"models":[{"engine":"ollama","model":"a"}],"ready":true}`, false)
	test("invalid raw model", "\"engineBindingOffers\":{\"version\":1,\"models\":[{\"engine\":\"ollama\",\"model\":\"\xff\"}]}", false)
	test("escaped surrogate follows existing JSON decoding", `"engineBindingOffers":{"version":1,"models":[{"engine":"ollama","model":"\ud800"}]}`, true)
	test("duplicate model key", `"engineBindingOffers":{"version":1,"models":[{"engine":"ollama","model":"a","model":"b"}]}`, false)
	test("absent", `"future":true`, false)
}

func TestEngineBindingOffersCloneAndRoundTrip(t *testing.T) {
	offers := &EngineBindingOffers{Version: 1, Models: []Role{{Engine: "fixture-engine", Model: "org/model"}}}
	clone := offers.Clone()
	clone.Models[0].Model = "changed"
	if offers.Models[0].Model != "org/model" {
		t.Fatal("clone shares model references")
	}
	raw, err := json.Marshal(Advertisement{Revision: "1", EngineBindingOffers: offers})
	if err != nil {
		t.Fatal(err)
	}
	var got Advertisement
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.EngineBindingOffers, offers) || got.StrictAdmission != nil || got.ServedModels != nil {
		t.Fatalf("roundtrip invented serving evidence: %+v", got)
	}
}
