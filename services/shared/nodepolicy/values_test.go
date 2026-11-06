// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestRoleNameGrammar(t *testing.T) {
	for _, name := range []string{"weak", "coder", "coder-api", "r1.fast_v2", strings.Repeat("a", 64)} {
		t.Run("accept "+name, func(t *testing.T) {
			if err := ValidateRoleName(name); err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, name := range []string{"", "Coder", "-x", "co:der", strings.Repeat("a", 65)} {
		t.Run("reject "+name, func(t *testing.T) {
			if err := ValidateRoleName(name); err == nil {
				t.Fatalf("accepted %q", name)
			}
		})
	}
	if MaxRoleNameBytes != 64 {
		t.Fatalf("role byte limit = %d", MaxRoleNameBytes)
	}
}

func TestRoleAliasNamespaces(t *testing.T) {
	for _, tc := range []struct {
		name, base string
		mode       AliasMode
	}{
		{"coder", "coder", AliasBare}, {"coder-api", "coder-api", AliasBare},
		{"coder-local", "coder-local", AliasBare}, {"weak-", "weak-", AliasBare},
		{"coder:api", "coder", AliasAPI}, {"coder:local", "coder", AliasLocal},
		{strings.Repeat("a", 64) + ":local", strings.Repeat("a", 64), AliasLocal},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base, mode, ok := ParseAlias(tc.name)
			if !ok || base != tc.base || mode != tc.mode {
				t.Fatalf("alias = %q %q %v", base, mode, ok)
			}
			if got := IsReservedAlias(tc.name); got != (tc.mode != AliasBare) {
				t.Fatalf("reserved alias = %v", got)
			}
		})
	}
	for _, name := range []string{"", "Coder", "coder:fast", "coder:", "coder:API", " coder", strings.Repeat("a", 65) + ":local"} {
		t.Run("invalid alias "+name, func(t *testing.T) {
			base, mode, ok := ParseAlias(name)
			if ok || base != "" || mode != "" || IsReservedAlias(name) {
				t.Fatalf("invalid alias = %q %q %v", base, mode, ok)
			}
		})
	}
	for _, name := range []string{"coder:api", "coder:fast", "coder:", "coder:local"} {
		if !IsReservedProfileKey(name) {
			t.Errorf("profile key %q not reserved", name)
		}
	}
	for _, name := range []string{"weak", "coder", "coder-api", "coder-local", "Coder:api", "unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL"} {
		if IsReservedProfileKey(name) {
			t.Errorf("profile key %q reserved", name)
		}
	}
}

func TestParseNanoUSDAcceptsCanonicalForms(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  int64
	}{
		{name: "zero", input: "0", want: 0},
		{name: "single digit", input: "7", want: 7},
		{name: "two digits", input: "10", want: 10},
		{name: "one dollar", input: "1000000000", want: 1000000000},
		{name: "beyond float64 exactness", input: "9007199254740993", want: 9007199254740993},
		{name: "signed 64-bit maximum", input: "9223372036854775807", want: 9223372036854775807},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := ParseNanoUSD(c.input)
			if err != nil {
				t.Fatalf("ParseNanoUSD(%q) error = %v, want accepted", c.input, err)
			}
			if string(got) != c.input {
				t.Fatalf("ParseNanoUSD(%q) = %q, want the input unchanged", c.input, got)
			}
			exact, err := got.Int64()
			if err != nil {
				t.Fatalf("Int64() error = %v", err)
			}
			if exact != c.want {
				t.Fatalf("Int64() = %d, want %d", exact, c.want)
			}
		})
	}
}

func TestParseNanoUSDRejectsNonCanonicalForms(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		{name: "empty", input: ""},
		{name: "minus sign", input: "-1"},
		{name: "negative zero", input: "-0"},
		{name: "plus sign", input: "+1"},
		{name: "leading space", input: " 1"},
		{name: "trailing space", input: "1 "},
		{name: "trailing newline", input: "1\n"},
		{name: "leading zero", input: "01"},
		{name: "double zero", input: "00"},
		{name: "fraction", input: "0.5"},
		{name: "zero fraction", input: "1.0"},
		{name: "lowercase exponent", input: "1e3"},
		{name: "uppercase exponent", input: "1E3"},
		{name: "digit separator", input: "1_000"},
		{name: "thousands comma", input: "1,000"},
		{name: "hexadecimal", input: "0x10"},
		{name: "non-ASCII digit", input: "١"},
		{name: "one past signed 64-bit maximum", input: "9223372036854775808"},
		{name: "far past signed 64-bit maximum", input: "99999999999999999999"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := ParseNanoUSD(c.input)
			if err == nil {
				t.Fatalf("ParseNanoUSD(%q) = %q, want an error", c.input, got)
			}
		})
	}
}

func TestNanoUSDZeroValueIsInvalid(t *testing.T) {
	var zero NanoUSD
	if _, err := zero.Int64(); err == nil {
		t.Fatal("zero NanoUSD Int64() error = nil, want an error")
	}
	if _, err := json.Marshal(zero); err == nil {
		t.Fatal("zero NanoUSD json.Marshal error = nil, want an error")
	}
}

type moneyDoc struct {
	Amount NanoUSD `json:"amount"`
}

func TestNanoUSDJSONRoundTripsAsString(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  int64
	}{
		{name: "beyond float64 exactness", input: "9007199254740993", want: 9007199254740993},
		{name: "signed 64-bit maximum", input: "9223372036854775807", want: 9223372036854775807},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			amount, err := ParseNanoUSD(c.input)
			if err != nil {
				t.Fatalf("ParseNanoUSD(%q) error = %v", c.input, err)
			}
			data, err := json.Marshal(moneyDoc{Amount: amount})
			if err != nil {
				t.Fatalf("json.Marshal error = %v", err)
			}
			if want := `{"amount":"` + c.input + `"}`; string(data) != want {
				t.Fatalf("json.Marshal = %s, want %s", data, want)
			}
			var back moneyDoc
			if err := json.Unmarshal(data, &back); err != nil {
				t.Fatalf("json.Unmarshal(%s) error = %v", data, err)
			}
			if back.Amount != amount {
				t.Fatalf("round-tripped amount = %q, want %q", back.Amount, amount)
			}
			exact, err := back.Amount.Int64()
			if err != nil {
				t.Fatalf("Int64() error = %v", err)
			}
			if exact != c.want {
				t.Fatalf("Int64() = %d, want %d", exact, c.want)
			}
		})
	}
}

func TestNanoUSDJSONRejectsNumbersAndInvalidStrings(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{name: "integer number", body: `{"amount":9007199254740993}`},
		{name: "fractional number", body: `{"amount":1.5}`},
		{name: "null", body: `{"amount":null}`},
		{name: "boolean", body: `{"amount":true}`},
		{name: "empty string", body: `{"amount":""}`},
		{name: "signed string", body: `{"amount":"-1"}`},
		{name: "leading zero string", body: `{"amount":"01"}`},
		{name: "overflow string", body: `{"amount":"9223372036854775808"}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var doc moneyDoc
			if err := json.Unmarshal([]byte(c.body), &doc); err == nil {
				t.Fatalf("json.Unmarshal(%s) = %q, want an error", c.body, doc.Amount)
			}
		})
	}
}

func TestNanoUSDJSONMarshalRejectsNonCanonicalValue(t *testing.T) {
	cases := []struct {
		name  string
		value NanoUSD
	}{
		{name: "empty zero value", value: ""},
		{name: "leading zero", value: "01"},
		{name: "negative sign", value: "-1"},
		{name: "beyond int64 max", value: "9223372036854775808"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if data, err := json.Marshal(moneyDoc{Amount: c.value}); err == nil {
				t.Fatalf("json.Marshal(%q) = %s, want an error", c.value, data)
			}
		})
	}
}

func TestParseCredentialRefAcceptsReservedNamespace(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		{name: "single letter", input: "NVPAIR_PROVIDER_SECRET_A"},
		{name: "digit only suffix", input: "NVPAIR_PROVIDER_SECRET_0"},
		{name: "underscores and digits", input: "NVPAIR_PROVIDER_SECRET_OPENAI_KEY_2"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := ParseCredentialRef(c.input)
			if err != nil {
				t.Fatalf("ParseCredentialRef(%q) error = %v, want accepted", c.input, err)
			}
			if string(got) != c.input {
				t.Fatalf("ParseCredentialRef(%q) = %q, want the input unchanged", c.input, got)
			}
		})
	}
}

func TestParseCredentialRefRejectsOtherForms(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		{name: "empty", input: ""},
		{name: "prefix only", input: "NVPAIR_PROVIDER_SECRET_"},
		{name: "lowercase prefix", input: "nvpair_provider_secret_OPENAI"},
		{name: "mixed-case prefix", input: "NVPair_PROVIDER_SECRET_OPENAI"},
		{name: "lowercase suffix", input: "NVPAIR_PROVIDER_SECRET_openai"},
		{name: "mixed-case suffix", input: "NVPAIR_PROVIDER_SECRET_OpenAI"},
		{name: "hyphen in suffix", input: "NVPAIR_PROVIDER_SECRET_OPEN-AI"},
		{name: "dot in suffix", input: "NVPAIR_PROVIDER_SECRET_OPEN.AI"},
		{name: "leading space", input: " NVPAIR_PROVIDER_SECRET_A"},
		{name: "trailing space", input: "NVPAIR_PROVIDER_SECRET_A "},
		{name: "trailing newline", input: "NVPAIR_PROVIDER_SECRET_A\n"},
		{name: "non-ASCII suffix", input: "NVPAIR_PROVIDER_SECRET_É"},
		{name: "other namespace", input: "OPENAI_API_KEY"},
		{name: "secret-looking value", input: "sk-live-abc123"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := ParseCredentialRef(c.input)
			if err == nil {
				t.Fatalf("ParseCredentialRef(%q) = %q, want an error", c.input, got)
			}
		})
	}
}

func TestParseCredentialRefErrorDoesNotEchoInput(t *testing.T) {
	const input = "NVPAIR_PROVIDER_SECRET_sk-live-abc123"
	_, err := ParseCredentialRef(input)
	if err == nil {
		t.Fatalf("ParseCredentialRef(%q) error = nil, want an error", input)
	}
	if strings.Contains(err.Error(), "sk-live") || strings.Contains(err.Error(), input) {
		t.Fatalf("error %q echoes the rejected input", err.Error())
	}
}
