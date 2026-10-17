// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"fmt"
	"net/url"
	"regexp"
	"slices"
	"strings"
)

// remoteDispatchAvailable says whether this build can dispatch to a provider
// API. It is false until the hybrid dispatch path exists, so validation refuses
// every setting that would enable remote or paid execution while still
// accepting disabled provider configuration for staging.
const remoteDispatchAvailable = false

// CurrencyUSD is the only denomination routing money is expressed in.
const CurrencyUSD = "USD"

// PriceUnit is the only price unit: integer nano-USD per million tokens.
const PriceUnit = "nanoUSDPerMillionTokens"

// BillingClass says how a provider account is billed.
type BillingClass string

const (
	// BillingFree is a permanently free allowance.
	BillingFree BillingClass = "free"
	// BillingTrial is a trial credit, never implying permanent free access.
	BillingTrial BillingClass = "trial"
	// BillingPaid bills per use.
	BillingPaid BillingClass = "paid"
)

// BareMode is where a bare tier alias ("weak") routes. The suffixed aliases
// ("weak-api", "weak-local") override it per request.
type BareMode string

const (
	// BareHybrid prefers local capacity, then free APIs, then paid APIs.
	BareHybrid BareMode = "hybrid"
	// BareAPI excludes local destinations.
	BareAPI BareMode = "api"
	// BareLocal excludes API destinations.
	BareLocal BareMode = "local"
)

// Routing is the origin-only hybrid routing configuration. It never leaves the
// machine that owns the policy: peers neither see nor change it. Tier
// memberships stay in Policy.Tiers; this section holds the objects they refer
// to and the rules that apply to them.
type Routing struct {
	// Accounts are the provider billing and quota buckets.
	Accounts []Account `json:"accounts"`
	// Deployments are the provider models a tier's API entries refer to.
	Deployments []Deployment `json:"deployments"`
	// Tiers holds the settings of every tier, keyed by tier name.
	Tiers    map[string]TierRouting `json:"tiers"`
	Limits   Limits                 `json:"limits"`
	Listener Listener               `json:"listener"`
}

// Account is one provider billing or quota bucket. It carries the credential
// reference, never the secret, and the money ceilings every deployment on it
// shares.
type Account struct {
	ID string `json:"id"`
	// Credential names the environment variable holding the secret.
	Credential CredentialRef `json:"credential"`
	Billing    BillingClass  `json:"billing"`
	// PerRequestCeiling, DailyCeiling and MonthlyCeiling are the most this
	// instance may spend, in nano-USD. "0" allows no spend.
	PerRequestCeiling NanoUSD `json:"perRequestCeiling"`
	DailyCeiling      NanoUSD `json:"dailyCeiling"`
	MonthlyCeiling    NanoUSD `json:"monthlyCeiling"`
}

// Deployment is one provider model on an account, with the bounds and prices
// the router needs before it may send a request there.
type Deployment struct {
	ID      string `json:"id"`
	Account string `json:"account"`
	// Model is the provider's model identifier.
	Model string `json:"model"`
	// Family groups deployments for per-tier family constraints.
	Family       string       `json:"family"`
	Capabilities []Capability `json:"capabilities,omitempty"`
	// ContextTokens and MaxOutputTokens are the explicit bounds the router
	// computes against. Neither may be unknown.
	ContextTokens   int `json:"contextTokens"`
	MaxOutputTokens int `json:"maxOutputTokens"`
	// Prices is absent while unknown, which keeps paid use off.
	Prices *Prices `json:"prices,omitempty"`
	// Enabled makes the deployment eligible. Only a qualified deployment may be
	// enabled.
	Enabled bool `json:"enabled"`
}

// Prices are a deployment's rates. Currency and Unit are declared so a
// mismatched denomination is refused instead of converted.
type Prices struct {
	Currency string `json:"currency"`
	Unit     string `json:"unit"`
	// InputPerMillion is the maximum uncached input rate.
	InputPerMillion  NanoUSD `json:"inputPerMillion"`
	OutputPerMillion NanoUSD `json:"outputPerMillion"`
}

// TierRouting is one tier's routing settings.
type TierRouting struct {
	// BareMode is where the bare alias for this tier routes.
	BareMode BareMode `json:"bareMode"`
	// RemoteAllowed lets this tier send requests to provider APIs. Paid APIs
	// additionally need PaidAllowed.
	RemoteAllowed bool `json:"remoteAllowed"`
	PaidAllowed   bool `json:"paidAllowed"`
	// AllowedFamilies, when not empty, limits API deployments to these families.
	AllowedFamilies []string `json:"allowedFamilies"`
}

// Limits are the finite bounds on one hybrid request. Zero queue and cold-start
// allowances mean the router never waits for local capacity.
type Limits struct {
	// QueueSeconds is the total time a request may wait for a local slot.
	QueueSeconds int `json:"queueSeconds"`
	// ColdStartSeconds is the total time a request may wait for a model to load.
	ColdStartSeconds int `json:"coldStartSeconds"`
	// MaxAttempts is the total number of attempts across all candidates.
	MaxAttempts int `json:"maxAttempts"`
	// PreOutputBudgetSeconds bounds selection and attempts before any output.
	PreOutputBudgetSeconds int `json:"preOutputBudgetSeconds"`
	// MaxOutputTokens is applied when a request names no output limit.
	MaxOutputTokens int `json:"maxOutputTokens"`
}

// Listener is the unified loopback entry the proxy serves aliases on.
type Listener struct {
	Enabled bool `json:"enabled"`
	// Port is the one port the listener binds. 0 means none is chosen yet.
	Port int `json:"port"`
	// AllowedOrigins are the exact browser origins let through, such as
	// "http://localhost:3000". Requests with any other Origin are denied.
	AllowedOrigins []string `json:"allowedOrigins"`
}

const (
	maxLimitSeconds   = 600
	maxAttempts       = 10
	maxTokenBound     = 1 << 24
	minListenerPort   = 1024
	maxListenerPort   = 65535
	maxAllowedOrigins = 32
)

// routingID is the shape of account, deployment and family identifiers.
var routingID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)

// defaultRouting is the routing section of Default(): no accounts, every tier
// local-preferred with remote and paid execution off, and the finite limits.
func defaultRouting() Routing {
	tiers := make(map[string]TierRouting, len(TierOrder))
	for _, t := range TierOrder {
		tiers[t] = TierRouting{BareMode: BareHybrid, AllowedFamilies: []string{}}
	}
	return Routing{
		Accounts:    []Account{},
		Deployments: []Deployment{},
		Tiers:       tiers,
		Limits: Limits{
			QueueSeconds:           0,
			ColdStartSeconds:       0,
			MaxAttempts:            3,
			PreOutputBudgetSeconds: 30,
			MaxOutputTokens:        4096,
		},
		Listener: Listener{AllowedOrigins: []string{}},
	}
}

// normalize makes equal configurations compare equal after a JSON round trip:
// absent lists become empty and empty capability lists become nil.
func (r *Routing) normalize() {
	if r.Accounts == nil {
		r.Accounts = []Account{}
	}
	if r.Deployments == nil {
		r.Deployments = []Deployment{}
	}
	for i := range r.Deployments {
		r.Deployments[i].Capabilities = nilIfEmpty(r.Deployments[i].Capabilities)
	}
	for t, tr := range r.Tiers {
		if tr.AllowedFamilies == nil {
			tr.AllowedFamilies = []string{}
			r.Tiers[t] = tr
		}
	}
	if r.Listener.AllowedOrigins == nil {
		r.Listener.AllowedOrigins = []string{}
	}
}

func nilIfEmpty[T any](s []T) []T {
	if len(s) == 0 {
		return nil
	}
	return s
}

// validate reports the first problem with the routing section on its own.
// References from tier entries are checked by Policy.Validate.
func (r Routing) validate() error {
	accounts := map[string]bool{}
	credentials := map[CredentialRef]string{}
	for i, a := range r.Accounts {
		if !routingID.MatchString(a.ID) {
			return fmt.Errorf("routing.accounts[%d]: invalid id %q", i, a.ID)
		}
		if accounts[a.ID] {
			return fmt.Errorf("routing.accounts[%d]: duplicate id %q", i, a.ID)
		}
		accounts[a.ID] = true
		if _, err := ParseCredentialRef(string(a.Credential)); err != nil {
			return fmt.Errorf("routing.accounts[%d] %q: %w", i, a.ID, err)
		}
		if other, dup := credentials[a.Credential]; dup {
			return fmt.Errorf("routing.accounts[%d] %q: credential reference is already used by account %q", i, a.ID, other)
		}
		credentials[a.Credential] = a.ID
		switch a.Billing {
		case BillingFree, BillingTrial, BillingPaid:
		default:
			return fmt.Errorf("routing.accounts[%d] %q: billing must be %q, %q or %q", i, a.ID, BillingFree, BillingTrial, BillingPaid)
		}
		for _, c := range []struct {
			field string
			value NanoUSD
		}{
			{"perRequestCeiling", a.PerRequestCeiling},
			{"dailyCeiling", a.DailyCeiling},
			{"monthlyCeiling", a.MonthlyCeiling},
		} {
			if _, err := c.value.Int64(); err != nil {
				return fmt.Errorf("routing.accounts[%d] %q: %s: %w", i, a.ID, c.field, err)
			}
		}
	}

	deployments := map[string]bool{}
	for i, d := range r.Deployments {
		if err := validateDeployment(d, accounts); err != nil {
			return fmt.Errorf("routing.deployments[%d] %q: %w", i, d.ID, err)
		}
		if deployments[d.ID] {
			return fmt.Errorf("routing.deployments[%d]: duplicate id %q", i, d.ID)
		}
		deployments[d.ID] = true
	}

	for t := range r.Tiers {
		if !IsTier(t) {
			return fmt.Errorf("routing.tiers: unknown tier %q (want one of %v)", t, TierOrder)
		}
	}
	for _, t := range TierOrder {
		tr, ok := r.Tiers[t]
		if !ok {
			return fmt.Errorf("routing.tiers.%s is required", t)
		}
		if err := tr.validate(); err != nil {
			return fmt.Errorf("routing.tiers.%s: %w", t, err)
		}
	}
	if err := r.Limits.validate(); err != nil {
		return fmt.Errorf("routing.limits: %w", err)
	}
	if err := r.Listener.validate(); err != nil {
		return fmt.Errorf("routing.listener: %w", err)
	}
	return nil
}

func validateDeployment(d Deployment, accounts map[string]bool) error {
	if !routingID.MatchString(d.ID) {
		return fmt.Errorf("invalid id")
	}
	if !accounts[d.Account] {
		return fmt.Errorf("account %q is not defined", d.Account)
	}
	if d.Model == "" {
		return fmt.Errorf("model is required")
	}
	if IsReservedAlias(d.Model) {
		return fmt.Errorf("model %q is a reserved alias", d.Model)
	}
	if !routingID.MatchString(d.Family) {
		return fmt.Errorf("invalid family %q", d.Family)
	}
	for _, c := range d.Capabilities {
		if !slices.Contains(KnownCapabilities, c) {
			return fmt.Errorf("unknown capability %q", c)
		}
	}
	if d.ContextTokens < 1 || d.ContextTokens > maxTokenBound {
		return fmt.Errorf("contextTokens must be between 1 and %d", maxTokenBound)
	}
	if d.MaxOutputTokens < 1 || d.MaxOutputTokens > d.ContextTokens {
		return fmt.Errorf("maxOutputTokens must be between 1 and contextTokens")
	}
	if d.Prices != nil {
		if d.Prices.Currency != CurrencyUSD {
			return fmt.Errorf("prices.currency must be %q", CurrencyUSD)
		}
		if d.Prices.Unit != PriceUnit {
			return fmt.Errorf("prices.unit must be %q", PriceUnit)
		}
		if _, err := d.Prices.InputPerMillion.Int64(); err != nil {
			return fmt.Errorf("prices.inputPerMillion: %w", err)
		}
		if _, err := d.Prices.OutputPerMillion.Int64(); err != nil {
			return fmt.Errorf("prices.outputPerMillion: %w", err)
		}
	}
	if d.Enabled && !remoteDispatchAvailable {
		return fmt.Errorf("a deployment cannot be enabled: remote execution is not available yet")
	}
	return nil
}

func (t TierRouting) validate() error {
	switch t.BareMode {
	case BareHybrid, BareAPI, BareLocal:
	default:
		return fmt.Errorf("bareMode must be %q, %q or %q", BareHybrid, BareAPI, BareLocal)
	}
	if t.PaidAllowed && !t.RemoteAllowed {
		return fmt.Errorf("paidAllowed requires remoteAllowed")
	}
	if (t.RemoteAllowed || t.PaidAllowed) && !remoteDispatchAvailable {
		return fmt.Errorf("remote and paid execution cannot be enabled: remote execution is not available yet")
	}
	seen := map[string]bool{}
	for _, f := range t.AllowedFamilies {
		if !routingID.MatchString(f) {
			return fmt.Errorf("invalid allowed family %q", f)
		}
		if seen[f] {
			return fmt.Errorf("duplicate allowed family %q", f)
		}
		seen[f] = true
	}
	return nil
}

func (l Limits) validate() error {
	for _, c := range []struct {
		field string
		value int
	}{
		{"queueSeconds", l.QueueSeconds},
		{"coldStartSeconds", l.ColdStartSeconds},
		{"preOutputBudgetSeconds", l.PreOutputBudgetSeconds},
	} {
		if c.value < 0 || c.value > maxLimitSeconds {
			return fmt.Errorf("%s must be between 0 and %d", c.field, maxLimitSeconds)
		}
	}
	if l.PreOutputBudgetSeconds < 1 {
		return fmt.Errorf("preOutputBudgetSeconds must be at least 1")
	}
	if l.QueueSeconds > l.PreOutputBudgetSeconds || l.ColdStartSeconds > l.PreOutputBudgetSeconds {
		return fmt.Errorf("queueSeconds and coldStartSeconds must not exceed preOutputBudgetSeconds")
	}
	if l.MaxAttempts < 1 || l.MaxAttempts > maxAttempts {
		return fmt.Errorf("maxAttempts must be between 1 and %d", maxAttempts)
	}
	if l.MaxOutputTokens < 1 || l.MaxOutputTokens > maxTokenBound {
		return fmt.Errorf("maxOutputTokens must be between 1 and %d", maxTokenBound)
	}
	return nil
}

func (l Listener) validate() error {
	if l.Port != 0 && (l.Port < minListenerPort || l.Port > maxListenerPort) {
		return fmt.Errorf("port must be 0 or between %d and %d", minListenerPort, maxListenerPort)
	}
	if l.Enabled && l.Port == 0 {
		return fmt.Errorf("an enabled listener needs a port")
	}
	if len(l.AllowedOrigins) > maxAllowedOrigins {
		return fmt.Errorf("at most %d allowed origins", maxAllowedOrigins)
	}
	seen := map[string]bool{}
	for _, o := range l.AllowedOrigins {
		if err := checkOrigin(o); err != nil {
			return fmt.Errorf("allowedOrigins %q: %w", o, err)
		}
		if seen[o] {
			return fmt.Errorf("duplicate allowed origin %q", o)
		}
		seen[o] = true
	}
	return nil
}

// checkOrigin accepts only an exact browser origin: lower-case http or https
// scheme, host and optional port, with no path, query, fragment or userinfo.
func checkOrigin(o string) error {
	u, err := url.Parse(o)
	if err != nil {
		return fmt.Errorf("not a valid origin")
	}
	if (u.Scheme != "http" && u.Scheme != "https") || u.Hostname() == "" ||
		u.User != nil || u.Path != "" || u.RawPath != "" || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || u.Opaque != "" ||
		u.String() != o || o != strings.ToLower(o) {
		return fmt.Errorf("must be a lower-case scheme://host[:port] origin")
	}
	return nil
}

// validateTierRefs checks one tier's entries against the routing section.
func (r Routing) validateTierRefs(tier string, entries []TierEntry) error {
	deployments := map[string]bool{}
	for _, d := range r.Deployments {
		deployments[d.ID] = true
	}
	seen := map[string]bool{}
	for i, e := range entries {
		switch e.Kind {
		case EntryLocal:
			if err := checkModelRef(e.Engine, e.Model, e.ContextTokens, e.Capabilities); err != nil {
				return fmt.Errorf("tiers.%s[%d]: %w", tier, i, err)
			}
			if e.Deployment != "" {
				return fmt.Errorf("tiers.%s[%d]: a local entry cannot name a deployment", tier, i)
			}
		case EntryAPI:
			if e.Engine != "" || e.Model != "" || e.ContextTokens != 0 || len(e.Capabilities) > 0 {
				return fmt.Errorf("tiers.%s[%d]: an api entry names only a deployment", tier, i)
			}
			if !deployments[e.Deployment] {
				return fmt.Errorf("tiers.%s[%d]: deployment %q is not defined", tier, i, e.Deployment)
			}
		default:
			return fmt.Errorf("tiers.%s[%d]: kind must be %q or %q", tier, i, EntryLocal, EntryAPI)
		}
		key := entryKey(e)
		if seen[key] {
			return fmt.Errorf("tiers.%s[%d]: duplicate entry", tier, i)
		}
		seen[key] = true
	}
	return nil
}

// entryKey identifies a tier entry for duplicate detection. An API entry is
// its deployment. A local entry is its engine, model, declared context bound
// and capability set (order-insensitive), because entries that differ in bounds
// resolve differently and are not duplicates. Validation and version 1
// migration share it.
func entryKey(e TierEntry) string {
	if e.Kind == EntryAPI {
		return fmt.Sprintf("%s %q", EntryAPI, e.Deployment)
	}
	caps := make([]string, 0, len(e.Capabilities))
	for _, c := range e.Capabilities {
		caps = append(caps, string(c))
	}
	slices.Sort(caps)
	return fmt.Sprintf("%s %q %q %d %q", e.Kind, e.Engine, e.Model, e.ContextTokens, slices.Compact(caps))
}
