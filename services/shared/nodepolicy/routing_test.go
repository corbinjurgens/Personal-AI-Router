// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"reflect"
	"strings"
	"testing"
)

// stagedRoutingDocument is a disabled provider configuration: accounts, a
// deployment and API tier entries, none of them enabled.
const stagedRoutingDocument = `{
  "version": 2,
  "revision": "12",
  "tiers": {
    "weak": [
      {"kind": "local", "engine": "ollama", "model": "qwen3:8b"},
      {"kind": "api", "deployment": "free-small"}
    ],
    "strong": [{"kind": "api", "deployment": "free-small"}]
  },
  "routing": {
    "accounts": [
      {"id": "free-acct", "credential": "NVPAIR_PROVIDER_SECRET_FREE", "billing": "free",
       "perRequestCeiling": "0", "dailyCeiling": "0", "monthlyCeiling": "0"},
      {"id": "paid-acct", "credential": "NVPAIR_PROVIDER_SECRET_PAID", "billing": "paid",
       "perRequestCeiling": "1000000", "dailyCeiling": "9007199254740993", "monthlyCeiling": "9223372036854775807"}
    ],
    "deployments": [
      {"id": "free-small", "account": "free-acct", "model": "small-1", "family": "alpha",
       "capabilities": ["tools"], "contextTokens": 32768, "maxOutputTokens": 4096,
       "prices": {"currency": "USD", "unit": "nanoUSDPerMillionTokens", "inputPerMillion": "0", "outputPerMillion": "0"},
       "enabled": false},
      {"id": "paid-big", "account": "paid-acct", "model": "big-1", "family": "beta",
       "contextTokens": 200000, "maxOutputTokens": 8192, "enabled": false}
    ],
    "tiers": {
      "weak": {"bareMode": "local", "remoteAllowed": false, "paidAllowed": false, "allowedFamilies": ["alpha"]}
    },
    "limits": {"queueSeconds": 5, "coldStartSeconds": 10, "maxAttempts": 2, "preOutputBudgetSeconds": 20, "maxOutputTokens": 1024},
    "listener": {"enabled": true, "port": 4321, "allowedOrigins": ["http://localhost:3000", "https://app.example.com:8443"]}
  }
}`

// stagedPolicy parses stagedRoutingDocument, which must be valid.
func stagedPolicy(t *testing.T) Policy {
	t.Helper()
	p, err := Parse([]byte(stagedRoutingDocument))
	if err != nil {
		t.Fatalf("Parse(stagedRoutingDocument): %v", err)
	}
	return p
}

func TestParseAcceptsDisabledProviderConfiguration(t *testing.T) {
	p := stagedPolicy(t)
	if p.Revision != "12" || len(p.Routing.Accounts) != 2 || len(p.Routing.Deployments) != 2 {
		t.Fatalf("revision/accounts/deployments = %q/%d/%d", p.Revision, len(p.Routing.Accounts), len(p.Routing.Deployments))
	}
	if got := p.Routing.Accounts[1].MonthlyCeiling; got != "9223372036854775807" {
		t.Errorf("monthly ceiling = %q, want the signed 64-bit maximum unchanged", got)
	}
	if got := p.Routing.Accounts[1].DailyCeiling; got != "9007199254740993" {
		t.Errorf("daily ceiling = %q, want 9007199254740993 unchanged", got)
	}
	want := []TierEntry{
		{Kind: EntryLocal, Engine: "ollama", Model: "qwen3:8b"},
		{Kind: EntryAPI, Deployment: "free-small"},
	}
	if !reflect.DeepEqual(p.Tiers["weak"], want) {
		t.Errorf("weak entries = %+v, want %+v", p.Tiers["weak"], want)
	}
	if p.Routing.Tiers["weak"].BareMode != BareLocal || p.Routing.Tiers["medium"].BareMode != BareHybrid {
		t.Errorf("tier settings = %+v", p.Routing.Tiers)
	}
	if p.Routing.Limits.MaxAttempts != 2 || !p.Routing.Listener.Enabled || p.Routing.Listener.Port != 4321 {
		t.Errorf("limits/listener = %+v/%+v", p.Routing.Limits, p.Routing.Listener)
	}
}

func TestValidateRoutingRejects(t *testing.T) {
	test := func(name string, mutate func(*Policy), wantErr string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			p := stagedPolicy(t)
			mutate(&p)
			err := p.Validate()
			if err == nil {
				t.Fatalf("Validate() = nil, want an error containing %q", wantErr)
			}
			if !strings.Contains(err.Error(), wantErr) {
				t.Fatalf("Validate() = %q, want it to contain %q", err, wantErr)
			}
		})
	}

	// Duplicates.
	test("duplicate account id", func(p *Policy) {
		p.Routing.Accounts[1].ID = "free-acct"
	}, "duplicate id")
	test("duplicate deployment id", func(p *Policy) {
		p.Routing.Deployments[1].ID = "free-small"
	}, "duplicate id")
	test("shared credential reference", func(p *Policy) {
		p.Routing.Accounts[1].Credential = p.Routing.Accounts[0].Credential
	}, "already used")
	test("duplicate local tier entry", func(p *Policy) {
		p.Tiers["weak"] = append(p.Tiers["weak"], TierEntry{Kind: EntryLocal, Engine: "ollama", Model: "qwen3:8b"})
	}, "duplicate entry")
	test("duplicate api tier entry", func(p *Policy) {
		p.Tiers["strong"] = append(p.Tiers["strong"], TierEntry{Kind: EntryAPI, Deployment: "free-small"})
	}, "duplicate entry")
	test("duplicate allowed family", func(p *Policy) {
		tr := p.Routing.Tiers["weak"]
		tr.AllowedFamilies = []string{"alpha", "alpha"}
		p.Routing.Tiers["weak"] = tr
	}, "duplicate allowed family")

	// Cross references.
	test("deployment names a missing account", func(p *Policy) {
		p.Routing.Deployments[0].Account = "ghost"
	}, `account "ghost" is not defined`)
	test("api entry names a missing deployment", func(p *Policy) {
		p.Tiers["strong"] = []TierEntry{{Kind: EntryAPI, Deployment: "ghost"}}
	}, `deployment "ghost" is not defined`)
	test("api entry with local fields", func(p *Policy) {
		p.Tiers["strong"] = []TierEntry{{Kind: EntryAPI, Deployment: "free-small", Engine: "ollama"}}
	}, "names only a deployment")
	test("local entry with a deployment", func(p *Policy) {
		p.Tiers["strong"] = []TierEntry{{Kind: EntryLocal, Engine: "ollama", Model: "m", Deployment: "free-small"}}
	}, "cannot name a deployment")
	test("entry without a kind", func(p *Policy) {
		p.Tiers["strong"] = []TierEntry{{Engine: "ollama", Model: "m"}}
	}, "kind must be")
	test("routing settings for an unknown tier", func(p *Policy) {
		p.Routing.Tiers["huge"] = TierRouting{BareMode: BareHybrid}
	}, `unknown tier "huge"`)
	test("missing tier settings", func(p *Policy) {
		delete(p.Routing.Tiers, "medium")
	}, "routing.tiers.medium is required")
	test("unknown bare mode", func(p *Policy) {
		p.Routing.Tiers["weak"] = TierRouting{BareMode: "sideways"}
	}, "bareMode")

	// Alias collisions.
	for _, alias := range []string{"weak", "medium", "strong", "weak-api", "medium-api", "strong-api", "weak-local", "medium-local", "strong-local"} {
		test("profile model "+alias, func(p *Policy) {
			p.Profiles = []Profile{{Name: "p", Engine: "ollama", Model: alias}}
		}, "reserved alias")
		test("local tier model "+alias, func(p *Policy) {
			p.Tiers["strong"] = []TierEntry{{Kind: EntryLocal, Engine: "ollama", Model: alias}}
		}, "reserved alias")
		test("deployment model "+alias, func(p *Policy) {
			p.Routing.Deployments[0].Model = alias
		}, "reserved alias")
	}

	// Credentials.
	test("lower-case credential reference", func(p *Policy) {
		p.Routing.Accounts[0].Credential = "nvpair_provider_secret_free"
	}, "credential reference")
	test("mixed-case credential reference", func(p *Policy) {
		p.Routing.Accounts[0].Credential = "NVPAIR_PROVIDER_SECRET_Free"
	}, "credential reference")
	test("credential outside the namespace", func(p *Policy) {
		p.Routing.Accounts[0].Credential = "OPENAI_API_KEY"
	}, "credential reference")
	test("empty credential reference", func(p *Policy) {
		p.Routing.Accounts[0].Credential = ""
	}, "credential reference")

	// Money and currency.
	for name, amount := range map[string]NanoUSD{
		"unset":          "",
		"leading zero":   "01",
		"negative":       "-1",
		"fraction":       "1.5",
		"exponent":       "1e3",
		"padded":         " 1",
		"above signed64": "9223372036854775808",
	} {
		test("ceiling "+name, func(p *Policy) {
			p.Routing.Accounts[1].DailyCeiling = amount
		}, "dailyCeiling")
		test("input price "+name, func(p *Policy) {
			p.Routing.Deployments[0].Prices.InputPerMillion = amount
		}, "inputPerMillion")
		test("output price "+name, func(p *Policy) {
			p.Routing.Deployments[0].Prices.OutputPerMillion = amount
		}, "outputPerMillion")
	}
	test("non-USD currency", func(p *Policy) {
		p.Routing.Deployments[0].Prices.Currency = "EUR"
	}, "prices.currency")
	test("mismatched price unit", func(p *Policy) {
		p.Routing.Deployments[0].Prices.Unit = "USDPerToken"
	}, "prices.unit")
	test("invalid billing class", func(p *Policy) {
		p.Routing.Accounts[0].Billing = "donation"
	}, "billing")

	// Integer bounds.
	test("zero context", func(p *Policy) { p.Routing.Deployments[0].ContextTokens = 0 }, "contextTokens")
	test("output above context", func(p *Policy) { p.Routing.Deployments[0].MaxOutputTokens = 32769 }, "maxOutputTokens")
	test("zero output", func(p *Policy) { p.Routing.Deployments[0].MaxOutputTokens = 0 }, "maxOutputTokens")
	test("unknown capability", func(p *Policy) {
		p.Routing.Deployments[0].Capabilities = []Capability{"telepathy"}
	}, "unknown capability")
	test("invalid family", func(p *Policy) { p.Routing.Deployments[0].Family = "al pha" }, "family")
	test("invalid account id", func(p *Policy) { p.Routing.Accounts[0].ID = "" }, "invalid id")
	test("negative queue", func(p *Policy) { p.Routing.Limits.QueueSeconds = -1 }, "queueSeconds")
	test("negative cold start", func(p *Policy) { p.Routing.Limits.ColdStartSeconds = -1 }, "coldStartSeconds")
	test("queue above the pre-output budget", func(p *Policy) { p.Routing.Limits.QueueSeconds = 21 }, "must not exceed")
	test("cold start above the pre-output budget", func(p *Policy) { p.Routing.Limits.ColdStartSeconds = 21 }, "must not exceed")
	test("zero budget", func(p *Policy) {
		p.Routing.Limits.QueueSeconds, p.Routing.Limits.ColdStartSeconds, p.Routing.Limits.PreOutputBudgetSeconds = 0, 0, 0
	}, "preOutputBudgetSeconds")
	test("budget above ten minutes", func(p *Policy) { p.Routing.Limits.PreOutputBudgetSeconds = 601 }, "preOutputBudgetSeconds")
	test("zero attempts", func(p *Policy) { p.Routing.Limits.MaxAttempts = 0 }, "maxAttempts")
	test("too many attempts", func(p *Policy) { p.Routing.Limits.MaxAttempts = 11 }, "maxAttempts")
	test("zero output limit", func(p *Policy) { p.Routing.Limits.MaxOutputTokens = 0 }, "maxOutputTokens")
	test("listener port below 1024", func(p *Policy) { p.Routing.Listener.Port = 80 }, "port")
	test("listener port above 65535", func(p *Policy) { p.Routing.Listener.Port = 65536 }, "port")
	test("enabled listener without a port", func(p *Policy) { p.Routing.Listener.Port = 0 }, "needs a port")

	// Listener origins.
	for name, origin := range map[string]string{
		"wildcard":       "*",
		"null":           "null",
		"trailing slash": "http://localhost:3000/",
		"path":           "http://localhost:3000/app",
		"query":          "http://localhost:3000?x=1",
		"userinfo":       "http://user@localhost:3000",
		"upper case":     "HTTP://localhost:3000",
		"other scheme":   "ftp://localhost",
		"no host":        "http://",
	} {
		test("origin "+name, func(p *Policy) {
			p.Routing.Listener.AllowedOrigins = []string{origin}
		}, "allowedOrigins")
	}
	test("duplicate origin", func(p *Policy) {
		p.Routing.Listener.AllowedOrigins = []string{"http://localhost:3000", "http://localhost:3000"}
	}, "duplicate allowed origin")

	// Remote and paid execution cannot be enabled before dispatch exists.
	test("remote allowed", func(p *Policy) {
		tr := p.Routing.Tiers["weak"]
		tr.RemoteAllowed = true
		p.Routing.Tiers["weak"] = tr
	}, "cannot be enabled")
	test("paid allowed", func(p *Policy) {
		tr := p.Routing.Tiers["weak"]
		tr.RemoteAllowed, tr.PaidAllowed = true, true
		p.Routing.Tiers["weak"] = tr
	}, "cannot be enabled")
	test("paid without remote", func(p *Policy) {
		tr := p.Routing.Tiers["weak"]
		tr.PaidAllowed = true
		p.Routing.Tiers["weak"] = tr
	}, "paidAllowed requires remoteAllowed")
	test("enabled deployment", func(p *Policy) {
		p.Routing.Deployments[0].Enabled = true
	}, "cannot be enabled")
}

func TestParseRoutingDocumentRefusals(t *testing.T) {
	test := func(name, doc, wantErr string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			_, err := Parse([]byte(doc))
			if err == nil || !strings.Contains(err.Error(), wantErr) {
				t.Fatalf("Parse() err = %v, want it to contain %q", err, wantErr)
			}
		})
	}
	account := func(credential, ceiling string) string {
		return `{"version":2,"routing":{"accounts":[{"id":"a","credential":"` + credential + `","billing":"free",` +
			`"perRequestCeiling":` + ceiling + `,"dailyCeiling":"0","monthlyCeiling":"0"}]}}`
	}
	test("ceiling as a JSON number", account("NVPAIR_PROVIDER_SECRET_A", `5`), "money must be a JSON string")
	test("ceiling as null", account("NVPAIR_PROVIDER_SECRET_A", `null`), "money must be a JSON string")
	test("mixed-case credential", account("NVPAIR_PROVIDER_SECRET_Mixed", `"0"`), "credential reference")
	test("remote allowed", `{"version":2,"routing":{"tiers":{"weak":{"bareMode":"hybrid","remoteAllowed":true,"paidAllowed":false}}}}`, "cannot be enabled")
	test("revision as a number", `{"version":2,"revision":5}`, "parse policy")
}

func TestParseErrorsDoNotEchoCredentialReferences(t *testing.T) {
	const supplied = "NVPAIR_PROVIDER_SECRET_Hunter2Mixed"
	doc := `{"version":2,"routing":{"accounts":[{"id":"a","credential":"` + supplied + `","billing":"free",` +
		`"perRequestCeiling":"0","dailyCeiling":"0","monthlyCeiling":"0"}]}}`
	_, err := Parse([]byte(doc))
	if err == nil {
		t.Fatal("Parse() = nil, want a credential error")
	}
	if strings.Contains(err.Error(), "Hunter2Mixed") {
		t.Fatalf("error echoes the credential reference: %v", err)
	}
}

func TestLocalEntryClassification(t *testing.T) {
	if !(TierEntry{Kind: EntryLocal}).IsLocal() || (TierEntry{Kind: EntryAPI}).IsLocal() || (TierEntry{}).IsLocal() {
		t.Fatal("IsLocal misclassified an entry kind")
	}
}
