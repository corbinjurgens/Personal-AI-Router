// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/noderec"
)

func TestDiscoveryFiltersModelIdentities(t *testing.T) {
	identity := `{"repository":"owner/repository","quantization":"Q4_K_M","strength":"declared"}`
	invalidUTF8 := string([]byte{'"', 0xff, '"'})
	raw := []byte(`{"nodes":[{"hostUuid":"peer","models":["first","last"],"modelsByEngine":{"llamacpp":["first","last"]},"modelIdentities":{"llamacpp":{"first":` + identity + `,"":` + identity + `,"bad\u0000name":` + identity + `,"` + strings.Repeat("x", 513) + `":` + identity + `,` + invalidUTF8 + `:` + identity + `,"last":` + identity + `},"llamacpp-custom":{"other":` + identity + `}}}]}`)
	clean, err := filterDiscoveryModelsJSON(raw)
	if err != nil {
		t.Fatalf("filter discovery identities: %v", err)
	}
	var result noderec.GetNodesResult
	if err := json.Unmarshal(clean, &result); err != nil {
		t.Fatalf("decode filtered discovery: %v", err)
	}
	wantIdentity := engines.ModelIdentity{Repository: "owner/repository", Quantization: "Q4_K_M", Strength: "declared"}
	want := map[string]map[string]engines.ModelIdentity{"llamacpp": {"first": wantIdentity, "last": wantIdentity}, "llamacpp-custom": {"other": wantIdentity}}
	if len(result.Nodes) != 1 || !reflect.DeepEqual(result.Nodes[0].ModelIdentities, want) {
		t.Fatalf("filtered identities = %+v, want %+v", result.Nodes, want)
	}
	if !reflect.DeepEqual(result.Nodes[0].Models, []string{"first", "last"}) || !reflect.DeepEqual(result.Nodes[0].ModelsByEngine, map[string][]string{"llamacpp": {"first", "last"}}) {
		t.Fatal("identity filtering changed routing inventory")
	}
}

func TestDiscoveryMalformedIdentityBlockKeepsInventory(t *testing.T) {
	for _, block := range []string{`"invalid"`, `[]`, `{"llamacpp":[]}`, `null`} {
		t.Run(block, func(t *testing.T) {
			clean, err := filterDiscoveryModelsJSON([]byte(`{"nodes":[{"hostUuid":"peer","modelsByEngine":{"llamacpp":["ready"]},"modelIdentities":` + block + `}]}`))
			if err != nil {
				t.Fatalf("optional metadata failed discovery: %v", err)
			}
			var result noderec.GetNodesResult
			if err := json.Unmarshal(clean, &result); err != nil {
				t.Fatalf("decode filtered discovery: %v", err)
			}
			if len(result.Nodes) != 1 || !reflect.DeepEqual(result.Nodes[0].ModelsByEngine, map[string][]string{"llamacpp": {"ready"}}) || len(result.Nodes[0].ModelIdentities) != 0 {
				t.Fatalf("malformed metadata lost valid routing inventory: %+v", result)
			}
		})
	}
}
