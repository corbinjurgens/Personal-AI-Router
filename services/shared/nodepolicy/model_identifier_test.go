// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func TestModelIdentifierValidNames(t *testing.T) {
	for _, model := range []string{"unknown model / tagged:v1", "日本語/模型:latest", " spaces ", strings.Repeat("é", 256), strings.Repeat("x", 512)} {
		if err := ValidateModelIdentifier(model); err != nil {
			t.Fatalf("valid identifier rejected: %v", err)
		}
	}
}

func TestModelIdentifierRejectsInvalidNames(t *testing.T) {
	test := func(name, model string) {
		t.Run(name, func(t *testing.T) {
			if err := ValidateModelIdentifier(model); err == nil {
				t.Fatal("invalid identifier accepted")
			}
		})
	}
	test("empty", "")
	test("513 bytes", strings.Repeat("x", 513))
	test("multibyte over limit", strings.Repeat("é", 257))
	test("invalid UTF-8", string([]byte{0xff}))
	for r := rune(0); r <= 0x9f; r++ {
		if r >= 0x20 && r < 0x7f {
			continue
		}
		test(fmt.Sprintf("control U+%04X", r), "sentinel"+string(r)+"model")
	}
}

func TestModelIdentifierRawJSON(t *testing.T) {
	test := func(name string, raw []byte) {
		t.Run(name, func(t *testing.T) {
			if err := ValidateModelIdentifierJSON(raw); err == nil {
				t.Fatal("invalid raw model accepted")
			}
		})
	}
	test("raw invalid encoding", []byte{'"', 0xff, '"'})
	test("escaped C0", []byte(`"a\nb"`))
	test("escaped C1", []byte(`"a\u0085b"`))
	test("wrong type", []byte(`null`))
	raw, err := json.Marshal(strings.Repeat("é", 256))
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateModelIdentifierJSON(raw); err != nil {
		t.Fatal(err)
	}
}
