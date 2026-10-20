// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestRoutingBufferDefaultsAndRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name, raw string
		want      BufferLimits
	}{
		{"omitted routing", `{}`, BufferLimits{256, 5}},
		{"omitted buffers", `{"routing":{}}`, BufferLimits{256, 5}},
		{"empty buffers", `{"routing":{"buffers":{}}}`, BufferLimits{256, 5}},
		{"capacity only", `{"routing":{"buffers":{"capacityMiB":192}}}`, BufferLimits{192, 5}},
		{"zero wait", `{"routing":{"buffers":{"waitSeconds":0}}}`, BufferLimits{256, 0}},
		{"maximums", `{"routing":{"buffers":{"capacityMiB":1024,"waitSeconds":600}}}`, BufferLimits{1024, 600}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, err := Parse([]byte(tc.raw))
			if err != nil {
				t.Fatal(err)
			}
			if p.Routing.Buffers != tc.want {
				t.Fatalf("buffers = %+v, want %+v", p.Routing.Buffers, tc.want)
			}
			raw, err := json.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			next, err := Parse(raw)
			if err != nil || !reflect.DeepEqual(next, p) {
				t.Fatalf("round trip changed policy: error %v, buffers %+v", err, next.Routing.Buffers)
			}
		})
	}
}

func TestRoutingBufferInvalidExplicitValues(t *testing.T) {
	for _, buffers := range []string{
		`null`, `[]`, `1`, `"settings"`,
		`{"capacityMiB":null}`, `{"waitSeconds":null}`, `{"WaitSeconds":null}`,
		`{"capacityMiB":-1}`, `{"capacityMiB":0}`, `{"capacityMiB":191}`, `{"capacityMiB":1025}`,
		`{"capacityMiB":192.5}`, `{"capacityMiB":"256"}`, `{"capacityMiB":true}`,
		`{"waitSeconds":-1}`, `{"waitSeconds":601}`, `{"waitSeconds":0.5}`, `{"waitSeconds":"0"}`,
	} {
		t.Run(buffers, func(t *testing.T) {
			if _, err := Parse([]byte(`{"routing":{"buffers":` + buffers + `}}`)); err == nil {
				t.Fatal("invalid explicit buffer settings accepted")
			}
		})
	}
}

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

func TestPublicDispatchAvailable(t *testing.T) {
	if PublicDispatchAvailable() {
		t.Fatal("shipped build must keep public dispatch unavailable")
	}
	t.Run("accepts disabled staging", func(t *testing.T) {
		p := stagedPolicy(t)
		completeDeployment(&p.Routing.Deployments[0])
		if err := p.Validate(); err != nil {
			t.Fatalf("disabled complete deployment: %v", err)
		}
	})
	reject := func(name string, edit func(*Policy)) {
		t.Run(name, func(t *testing.T) {
			p := stagedPolicy(t)
			edit(&p)
			if err := p.Validate(); err == nil || !strings.Contains(err.Error(), "cannot be enabled") {
				t.Fatalf("Validate() = %v, want unavailable-dispatch refusal", err)
			}
		})
	}
	reject("rejects enabled deployment", func(p *Policy) {
		completeDeployment(&p.Routing.Deployments[0])
		p.Routing.Deployments[0].Enabled = true
	})
	reject("rejects remote permission", func(p *Policy) {
		tr := p.Routing.Tiers["weak"]
		tr.RemoteAllowed = true
		p.Routing.Tiers["weak"] = tr
	})
	reject("rejects paid permission", func(p *Policy) {
		tr := p.Routing.Tiers["weak"]
		tr.RemoteAllowed, tr.PaidAllowed = true, true
		p.Routing.Tiers["weak"] = tr
	})
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

func TestParseV2RejectsUnknownTierEntryKind(t *testing.T) {
	doc := `{"version":2,"tiers":{"weak":[{"kind":"remote","engine":"ollama","model":"m"}]}}`
	if _, err := Parse([]byte(doc)); err == nil || !strings.Contains(err.Error(), "kind") {
		t.Fatalf("Parse(unknown tier-entry kind) err = %v, want a kind error", err)
	}
}

// A suffixed alias names a fixed mode, so it cannot be configured as a tier.
func TestRoutingRejectsSuffixedAliasAsTierKey(t *testing.T) {
	doc := `{"version":2,"routing":{"tiers":{"weak-api":{"bareMode":"local"}}}}`
	if _, err := Parse([]byte(doc)); err == nil || !strings.Contains(err.Error(), `unknown tier "weak-api"`) {
		t.Fatalf("Parse(suffixed tier key) err = %v, want unknown-tier rejection", err)
	}
}

func TestDefaultAndMigratedTierFamiliesAreEmpty(t *testing.T) {
	checkEmptyFamilies := func(t *testing.T, p Policy) {
		t.Helper()
		for _, tier := range TierOrder {
			if got := p.Routing.Tiers[tier].AllowedFamilies; len(got) != 0 {
				t.Errorf("tier %s allowed families = %v, want none", tier, got)
			}
		}
	}

	t.Run("default", func(t *testing.T) {
		checkEmptyFamilies(t, Default())
	})
	t.Run("migrated v1", func(t *testing.T) {
		p, err := Parse([]byte(`{"version":1,"tiers":{"weak":[{"engine":"ollama","model":"m"}]}}`))
		if err != nil {
			t.Fatalf("Parse(v1): %v", err)
		}
		checkEmptyFamilies(t, p)
	})
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
	test("allowed family no deployment belongs to", func(p *Policy) {
		tr := p.Routing.Tiers["weak"]
		tr.AllowedFamilies = []string{"alpha", "gamma"}
		p.Routing.Tiers["weak"] = tr
	}, `allowed family "gamma" is not the family of any deployment`)
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
	test("enabled complete deployment", func(p *Policy) {
		completeDeployment(&p.Routing.Deployments[0])
		p.Routing.Deployments[0].Enabled = true
	}, "cannot be enabled")

	// Endpoint, protocol, transport and qualification.
	for name, endpoint := range map[string]string{
		"http scheme":             "http://api.example.com/v1/chat/completions",
		"upper-case scheme":       "HTTPS://api.example.com/v1/chat/completions",
		"upper-case host":         "https://API.example.com/v1/chat/completions",
		"userinfo":                "https://user@api.example.com/v1/chat/completions",
		"password":                "https://user:pw@api.example.com/v1/chat/completions",
		"query":                   "https://api.example.com/v1/chat/completions?key=1",
		"empty query":             "https://api.example.com/v1/chat/completions?",
		"fragment":                "https://api.example.com/v1/chat/completions#x",
		"empty fragment":          "https://api.example.com/v1/chat/completions#",
		"percent-encoding":        "https://api.example.com/v1/chat%2Fcompletions",
		"no path":                 "https://api.example.com",
		"root path":               "https://api.example.com/",
		"trailing slash":          "https://api.example.com/v1/",
		"empty segment":           "https://api.example.com/v1//chat",
		"dot segment":             "https://api.example.com/./chat",
		"dot-dot segment":         "https://api.example.com/v1/../chat",
		"single-label host":       "https://api/v1/chat",
		"IPv4 host":               "https://192.168.1.10/v1/chat",
		"public IPv4 host":        "https://8.8.8.8/v1/chat",
		"IPv6 host":               "https://[::1]/v1/chat",
		"localhost":               "https://localhost/v1/chat",
		"localhost subdomain":     "https://api.localhost/v1/chat",
		"mDNS name":               "https://printer.local/v1/chat",
		"internal name":           "https://llm.corp.internal/v1/chat",
		"arpa name":               "https://router.home.arpa/v1/chat",
		"localdomain name":        "https://box.localdomain/v1/chat",
		"implied port":            "https://api.example.com:443/v1/chat",
		"port zero":               "https://api.example.com:0/v1/chat",
		"port above 65535":        "https://api.example.com:65536/v1/chat",
		"leading zero port":       "https://api.example.com:0443/v1/chat",
		"empty port":              "https://api.example.com:/v1/chat",
		"whitespace":              "https://api.example.com/v1/chat completions",
		"trailing newline":        "https://api.example.com/v1/chat\n",
		"backslash":               "https://api.example.com\\v1/chat",
		"label with leading dash": "https://-api.example.com/v1/chat",
		"overlong label":          "https://" + strings.Repeat("a", 64) + ".example.com/v1/chat",
		"overlong host":           "https://" + strings.Repeat("a.", 127) + "com/v1/chat",
		"overlong endpoint":       "https://api.example.com/" + strings.Repeat("a", 2048),
	} {
		test("endpoint "+name, func(p *Policy) { p.Routing.Deployments[0].Endpoint = endpoint }, "endpoint")
	}
	test("unsupported protocol", func(p *Policy) { p.Routing.Deployments[0].Protocol = "openai-responses" }, "protocol")
	test("protocol case variant", func(p *Policy) { p.Routing.Deployments[0].Protocol = "OpenAI-Chat-Completions" }, "protocol")
	for name, c := range map[string]struct {
		mutate func(*Transport)
		field  string
	}{
		"zero connect timeout":     {func(t *Transport) { t.ConnectTimeoutSeconds = 0 }, "connectTimeoutSeconds"},
		"connect above a minute":   {func(t *Transport) { t.ConnectTimeoutSeconds = 61 }, "connectTimeoutSeconds"},
		"zero header timeout":      {func(t *Transport) { t.HeaderTimeoutSeconds = 0 }, "headerTimeoutSeconds"},
		"header above ten minutes": {func(t *Transport) { t.HeaderTimeoutSeconds = 601 }, "headerTimeoutSeconds"},
		"zero idle timeout":        {func(t *Transport) { t.IdleTimeoutSeconds = 0 }, "idleTimeoutSeconds"},
		"idle above ten minutes":   {func(t *Transport) { t.IdleTimeoutSeconds = 601 }, "idleTimeoutSeconds"},
		"zero response bytes":      {func(t *Transport) { t.MaxResponseBytes = 0 }, "maxResponseBytes"},
		"response above 64 MiB":    {func(t *Transport) { t.MaxResponseBytes = 64<<20 + 1 }, "maxResponseBytes"},
		"zero error bytes":         {func(t *Transport) { t.MaxErrorBodyBytes = 0 }, "maxErrorBodyBytes"},
		"error above 1 MiB":        {func(t *Transport) { t.MaxErrorBodyBytes = 1<<20 + 1 }, "maxErrorBodyBytes"},
		"zero event bytes":         {func(t *Transport) { t.MaxEventBytes = 0 }, "maxEventBytes"},
		"event above response":     {func(t *Transport) { t.MaxEventBytes = t.MaxResponseBytes + 1 }, "maxEventBytes"},
	} {
		test("transport "+name, func(p *Policy) {
			completeDeployment(&p.Routing.Deployments[0])
			c.mutate(p.Routing.Deployments[0].Transport)
		}, "transport."+c.field)
	}
	test("qualification without id", func(p *Policy) {
		p.Routing.Deployments[0].Qualification = &QualificationRef{Revision: "1"}
	}, "qualification id")
	test("qualification revision zero", func(p *Policy) {
		p.Routing.Deployments[0].Qualification = &QualificationRef{ID: "q", Revision: "0"}
	}, "qualification revision")
	for _, field := range []string{"endpoint", "protocol", "transport", "qualification", "prices"} {
		test("enabled deployment without "+field, func(p *Policy) {
			d := &p.Routing.Deployments[0]
			completeDeployment(d)
			switch field {
			case "endpoint":
				d.Endpoint = ""
			case "protocol":
				d.Protocol = ""
			case "transport":
				d.Transport = nil
			case "qualification":
				d.Qualification = nil
			case "prices":
				d.Prices = nil
			}
			d.Enabled = true
		}, "an enabled deployment needs "+field)
	}
}

// completeDeployment gives d every field an enabled deployment needs.
func completeDeployment(d *Deployment) {
	d.Endpoint = "https://api.example.com/v1/chat/completions"
	d.Protocol = ProtocolOpenAIChatCompletions
	d.Transport = &Transport{ConnectTimeoutSeconds: 10, HeaderTimeoutSeconds: 60, IdleTimeoutSeconds: 60,
		MaxResponseBytes: 16 << 20, MaxErrorBodyBytes: 64 << 10, MaxEventBytes: 1 << 20}
	d.Qualification = &QualificationRef{ID: "provider-q", Revision: "3"}
	if d.Prices == nil {
		d.Prices = &Prices{Currency: CurrencyUSD, Unit: PriceUnit, InputPerMillion: "100", OutputPerMillion: "200"}
	}
}

func TestCheckEndpoint(t *testing.T) {
	if err := CheckEndpoint("https://example.com/v1/chat/completions"); err != nil {
		t.Fatal(err)
	}
	for _, endpoint := range []string{"https://example.com/chat?", "https://example.com/chat?unknown=x", "https://example.com/chat#", "https://key@example.com/chat"} {
		t.Run(endpoint, func(t *testing.T) {
			if err := CheckEndpoint(endpoint); err == nil {
				t.Fatal("forbidden endpoint accepted")
			}
		})
	}
}

func TestDeploymentEndpointTransportAndQualificationAreAccepted(t *testing.T) {
	for _, endpoint := range []string{
		"https://api.example.com/v1/chat/completions",
		"https://api.example.com:8443/v1/chat/completions",
		"https://api.example.com:65535/a",
		"https://xn--bcher-kva.example/v1/chat",
		"https://a-b.c-d.example.com/openai/deployments/gpt_4.1~x/chat",
		"https://" + strings.Repeat("a", 63) + ".example.com/v1",
	} {
		p := stagedPolicy(t)
		completeDeployment(&p.Routing.Deployments[0])
		p.Routing.Deployments[0].Endpoint = endpoint
		if err := p.Validate(); err != nil {
			t.Errorf("Validate(endpoint %q) = %v, want nil", endpoint, err)
		}
	}

	p := stagedPolicy(t)
	completeDeployment(&p.Routing.Deployments[0])
	tr := p.Routing.Deployments[0].Transport
	tr.ConnectTimeoutSeconds, tr.HeaderTimeoutSeconds, tr.IdleTimeoutSeconds = 60, 600, 600
	tr.MaxResponseBytes, tr.MaxErrorBodyBytes, tr.MaxEventBytes = 64<<20, 1<<20, 64<<20
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	back, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse(complete deployment at every bound) = %v", err)
	}
	if !reflect.DeepEqual(back, p) {
		t.Fatalf("round trip changed the policy:\n got %+v\nwant %+v", back.Routing.Deployments[0], p.Routing.Deployments[0])
	}
}

// A document saved before the endpoint schema existed has none of its fields
// and must keep parsing, marshal without them and round-trip unchanged.
func TestStagedDeploymentWithoutEndpointSchemaRoundTrips(t *testing.T) {
	p := stagedPolicy(t)
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	for _, key := range []string{`"endpoint"`, `"protocol"`, `"transport"`, `"qualification"`} {
		if strings.Contains(string(data), key) {
			t.Errorf("marshaled staged policy contains %s; absent fields must stay absent", key)
		}
	}
	back, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse(round trip) = %v", err)
	}
	if !reflect.DeepEqual(back, p) {
		t.Fatal("round trip changed the staged policy")
	}
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

type deploymentCorpusCase struct {
	Name    string   `json:"name"`
	Path    []string `json:"path"`
	Value   any      `json:"value"`
	Remove  bool     `json:"remove"`
	Enabled bool     `json:"enabled"`
	Valid   bool     `json:"valid"`
}

type deploymentCorpus struct {
	Account    map[string]any         `json:"account"`
	Deployment map[string]any         `json:"deployment"`
	Cases      []deploymentCorpusCase `json:"cases"`
}

func loadDeploymentCorpus(t *testing.T) deploymentCorpus {
	t.Helper()
	data, err := os.ReadFile("testdata/deployment.json")
	if err != nil {
		t.Fatal(err)
	}
	var c deploymentCorpus
	if err := json.Unmarshal(data, &c); err != nil {
		t.Fatal(err)
	}
	return c
}

// The desktop parser consumes this exact corpus; both sides must agree on every
// case. Each mutation is applied to a canonical deployment inside a whole policy
// document, which is then parsed as the broker parses a saved policy.
func TestDeploymentParityCorpus(t *testing.T) {
	for _, tc := range loadDeploymentCorpus(t).Cases {
		t.Run(tc.Name, func(t *testing.T) {
			c := loadDeploymentCorpus(t)
			target := c.Deployment
			for _, key := range tc.Path[:max(len(tc.Path)-1, 0)] {
				child, ok := target[key].(map[string]any)
				if !ok {
					t.Fatalf("corpus path %v does not name an object", tc.Path)
				}
				target = child
			}
			if len(tc.Path) > 0 {
				key := tc.Path[len(tc.Path)-1]
				if tc.Remove {
					delete(target, key)
				} else {
					target[key] = tc.Value
				}
			}
			if tc.Enabled {
				c.Deployment["enabled"] = true
			}
			doc, err := json.Marshal(map[string]any{
				"version": 2,
				"routing": map[string]any{"accounts": []any{c.Account}, "deployments": []any{c.Deployment}},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = Parse(doc)
			if tc.Valid && err != nil {
				t.Fatalf("Parse() = %v, want valid", err)
			}
			if !tc.Valid && err == nil {
				t.Fatal("Parse() = nil, want an error")
			}
		})
	}
}
