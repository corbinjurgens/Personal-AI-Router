// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"strings"
	"testing"
)

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

func TestParseAliasAcceptsNineReservedNames(t *testing.T) {
	cases := []struct {
		name     string
		wantTier string
		wantMode AliasMode
	}{
		{name: "weak", wantTier: "weak", wantMode: AliasBare},
		{name: "weak-api", wantTier: "weak", wantMode: AliasAPI},
		{name: "weak-local", wantTier: "weak", wantMode: AliasLocal},
		{name: "medium", wantTier: "medium", wantMode: AliasBare},
		{name: "medium-api", wantTier: "medium", wantMode: AliasAPI},
		{name: "medium-local", wantTier: "medium", wantMode: AliasLocal},
		{name: "strong", wantTier: "strong", wantMode: AliasBare},
		{name: "strong-api", wantTier: "strong", wantMode: AliasAPI},
		{name: "strong-local", wantTier: "strong", wantMode: AliasLocal},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tier, mode, ok := ParseAlias(c.name)
			if !ok {
				t.Fatalf("ParseAlias(%q) ok = false, want reserved", c.name)
			}
			if tier != c.wantTier || mode != c.wantMode {
				t.Fatalf("ParseAlias(%q) = (%q, %q), want (%q, %q)", c.name, tier, mode, c.wantTier, c.wantMode)
			}
			if !IsReservedAlias(c.name) {
				t.Fatalf("IsReservedAlias(%q) = false, want true", c.name)
			}
		})
	}
}

func TestParseAliasRejectsEverythingElse(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		{name: "empty", input: ""},
		{name: "uppercase tier", input: "Weak"},
		{name: "uppercase suffix", input: "weak-API"},
		{name: "uppercase local suffix", input: "weak-Local"},
		{name: "ollama latest", input: "medium:latest"},
		{name: "leading space", input: " weak"},
		{name: "trailing space", input: "weak "},
		{name: "unknown suffix", input: "weak-cloud"},
		{name: "bare suffix", input: "weak-bare"},
		{name: "dangling hyphen", input: "weak-"},
		{name: "suffix without tier", input: "-api"},
		{name: "local suffix without tier", input: "-local"},
		{name: "double suffix", input: "weak-api-local"},
		{name: "reversed suffix", input: "weak-local-api"},
		{name: "underscore separator", input: "weak_api"},
		{name: "unknown tier", input: "huge"},
		{name: "unknown tier with suffix", input: "huge-api"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tier, mode, ok := ParseAlias(c.input)
			if ok {
				t.Fatalf("ParseAlias(%q) = (%q, %q, true), want not reserved", c.input, tier, mode)
			}
			if tier != "" || mode != "" {
				t.Fatalf("ParseAlias(%q) returned (%q, %q) with ok false, want empty", c.input, tier, mode)
			}
			if IsReservedAlias(c.input) {
				t.Fatalf("IsReservedAlias(%q) = true, want false", c.input)
			}
		})
	}
}
