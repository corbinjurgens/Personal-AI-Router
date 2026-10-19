// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nodepolicy is the fork's node policy contract: whether this machine
// accepts inference work (availability), how it admits work (resident models,
// concurrency, memory), how idle engines are unloaded or stopped, the model
// profiles it serves, the weak/medium/strong roles it declares for itself, and
// the tiers it routes by.
//
// Schema version 2 adds hybrid routing. A tier entry is tagged: a local entry
// names an engine and model on a cluster node, an API entry names a provider
// deployment. Tier memberships stay in Tiers. The origin-only Routing section
// holds the accounts, deployments, per-tier settings, limits and listener those
// entries refer to; Revision identifies the document. Parse migrates a version 1
// file by tagging its entries local and leaving remote and paid execution off.
// Validation refuses to enable remote or paid execution until a dispatch path
// exists, but accepts disabled provider configuration.
//
// The broker owns the policy file and is the only writer. It pushes the policy
// to the proxy, which enforces admission and resolves tiers. Both import this
// package so the two sides cannot drift. See FORK_DESIGN.md at the repo root.
package nodepolicy

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"

	"nvpair-shared/engines"
)

// FileName is the policy file under the per-user app data dir (appdir.Path).
const FileName = "node-policy.json"

// SchemaVersion is the current policy file version.
const SchemaVersion = 2

// legacySchemaVersion is the previous version Parse still migrates.
const legacySchemaVersion = 1

// Availability is whether this node runs inference work. It only governs work
// executed on this machine: a paused node still routes its own requests to
// other machines.
type Availability string

const (
	// Available accepts work.
	Available Availability = "available"
	// Draining rejects new work while running work finishes or is cancelled.
	// It is never persisted: the broker moves through it on the way to Paused.
	Draining Availability = "draining"
	// Paused rejects all work and never wakes an engine.
	Paused Availability = "paused"
)

// OnActive is what pausing does with work already running on this node.
type OnActive string

const (
	// FinishActive lets running work finish, up to DrainTimeoutSeconds.
	FinishActive OnActive = "finish"
	// CancelActive cancels running work immediately.
	CancelActive OnActive = "cancel"
)

// Capability names a request feature a model must support to be eligible.
type Capability string

const (
	CapTools      Capability = "tools"
	CapVision     Capability = "vision"
	CapStructured Capability = "structured"
	CapEmbeddings Capability = "embeddings"
)

// KnownCapabilities is every capability a profile or tier entry may declare.
var KnownCapabilities = []Capability{CapTools, CapVision, CapStructured, CapEmbeddings}

// Tier names. A tier request uses one of these as its "model" field.
const (
	TierWeak   = "weak"
	TierMedium = "medium"
	TierStrong = "strong"
)

// TierOrder lists the tiers from weakest to strongest.
var TierOrder = []string{TierWeak, TierMedium, TierStrong}

// Policy is the whole node policy file.
type Policy struct {
	Version int `json:"version"`
	// Revision identifies this document: a canonical decimal string, "0" until
	// the broker assigns one.
	Revision     string       `json:"revision"`
	Availability Availability `json:"availability"`
	Pause        PauseConfig  `json:"pause"`
	Admission    Admission    `json:"admission"`
	Idle         IdleConfig   `json:"idle"`
	// Profiles describe the models this node serves. Admission looks a request
	// up by (engine, model); a model with no profile is admitted with the
	// node-wide defaults.
	Profiles []Profile `json:"profiles"`
	// Roles is this node's declaration of what it serves per tier. A node
	// with at least one role is authoritative for itself: routers offer it
	// only its declared role for each tier it names, and nothing for a tier
	// absent from the map. Only a node that declares no role at all receives
	// a router's own tier entries. Routers never read Tiers of another node.
	Roles map[string]Role `json:"roles"`
	// Tiers map each tier name to its candidates, most preferred first. Tiers
	// are resolved by the node a request enters, across every node in the
	// cluster that has a candidate model installed.
	Tiers      map[string][]TierEntry `json:"tiers"`
	TierPolicy TierPolicy             `json:"tierPolicy"`
	// Routing is the origin-only hybrid routing configuration.
	Routing Routing `json:"routing"`
	// PausedBy records who paused the node: "local" or "peer". Empty while
	// available. A paused node with no provenance, such as one paused before
	// the field existed, counts as paused locally. Only node:set-availability
	// writes it; policy:set keeps it.
	PausedBy PauseOrigin `json:"pausedBy,omitempty"`
}

// PauseOrigin says who paused a node.
type PauseOrigin string

const (
	// PausedLocally is a pause made on the node itself.
	PausedLocally PauseOrigin = "local"
	// PausedByPeer is a pause relayed from a paired node.
	PausedByPeer PauseOrigin = "peer"
)

// Role is the engine and model this node serves one tier with. Bounds and
// capabilities come from the node's Profile for the same (engine, model).
type Role struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
}

// PauseConfig controls what pausing does.
type PauseConfig struct {
	OnActive            OnActive `json:"onActive"`
	DrainTimeoutSeconds int      `json:"drainTimeoutSeconds"`
	// UnloadModels unloads every loaded model once the node is drained.
	UnloadModels bool `json:"unloadModels"`
	// StopEngines also stops the managed engines, without changing their saved
	// On/Off intent, and keeps them stopped across restarts while paused.
	StopEngines bool `json:"stopEngines"`
}

// Admission is how this node decides whether to accept a request. Zero values
// disable the corresponding check.
type Admission struct {
	// MaxResidentModels caps how many models may be loaded at once across every
	// engine on this machine. 0 means no limit.
	MaxResidentModels int `json:"maxResidentModels"`
	// MaxConcurrentPerModel caps simultaneous requests per model unless the
	// model's profile sets its own MaxConcurrent. 0 means no limit.
	MaxConcurrentPerModel int `json:"maxConcurrentPerModel"`
	// QueueTimeoutSeconds is the longest a request waits for a slot here before
	// it is rejected so the router can try another machine. The router may ask
	// for less (see AdmissionWaitHeader).
	QueueTimeoutSeconds int `json:"queueTimeoutSeconds"`
	// MemoryBudgetBytes caps the summed MemoryBytes of the profiles of loaded
	// and running models. 0 means no limit. On Apple silicon this is the one
	// unified pool; on a discrete GPU it is the VRAM you want PAIR to use.
	MemoryBudgetBytes int64 `json:"memoryBudgetBytes"`
	// SwitchModels lets admission unload idle models to make room for the
	// requested one. Without it, a request that does not fit is rejected.
	SwitchModels bool `json:"switchModels"`
	// SwitchTimeoutSeconds bounds how long admission waits for an unload to show
	// up in the reported residency before admitting anyway.
	SwitchTimeoutSeconds int `json:"switchTimeoutSeconds"`
}

// IdleConfig is the cross-engine idle policy. Zero disables each step.
type IdleConfig struct {
	// UnloadAfterMinutes unloads an engine's models after this long without a
	// request on that engine.
	UnloadAfterMinutes int `json:"unloadAfterMinutes"`
	// StopEngineAfterMinutes stops an idle engine, keeping its saved intent On so
	// it can be started again on demand.
	StopEngineAfterMinutes int `json:"stopEngineAfterMinutes"`
	// StartOnDemand starts a stopped engine whose saved intent is On when a
	// request for it arrives. An engine saved Off is never started this way,
	// and nothing is started while the node is paused.
	StartOnDemand bool `json:"startOnDemand"`
	// WakeTimeoutSeconds bounds how long a request waits for that start.
	WakeTimeoutSeconds int `json:"wakeTimeoutSeconds"`
}

// Profile describes one model as served on this node.
type Profile struct {
	Name         string `json:"name"`
	Engine       string `json:"engine"`
	Model        string `json:"model"`
	Quantization string `json:"quantization,omitempty"`
	// ContextTokens is the context length this node serves the model with.
	ContextTokens int          `json:"contextTokens,omitempty"`
	Capabilities  []Capability `json:"capabilities,omitempty"`
	// MemoryBytes is a conservative estimate of the memory the model needs when
	// loaded, including its KV cache at ContextTokens. 0 means unknown, which
	// leaves the model out of the memory budget.
	MemoryBytes   int64 `json:"memoryBytes,omitempty"`
	MaxConcurrent int   `json:"maxConcurrent,omitempty"`
	// RequestOptions is merged into every request body for this model before it
	// reaches the engine, e.g. {"options":{"num_ctx":16384}} for Ollama or
	// {"keep_alive":"10m"}. Keys already present in the request win.
	RequestOptions json.RawMessage `json:"requestOptions,omitempty"`
}

// EntryKind tags what a tier entry refers to.
type EntryKind string

const (
	// EntryLocal is an engine and model on a cluster node.
	EntryLocal EntryKind = "local"
	// EntryAPI is a provider deployment in Routing.Deployments.
	EntryAPI EntryKind = "api"
)

// TierEntry is one candidate for a tier, either a local engine and model or a
// provider deployment. Kind says which fields apply; validation refuses an
// entry that sets the other kind's fields.
type TierEntry struct {
	Kind EntryKind `json:"kind"`
	// Engine, Model, ContextTokens and Capabilities describe a local entry.
	Engine        string       `json:"engine,omitempty"`
	Model         string       `json:"model,omitempty"`
	ContextTokens int          `json:"contextTokens,omitempty"`
	Capabilities  []Capability `json:"capabilities,omitempty"`
	// Deployment is the Routing.Deployments id of an API entry.
	Deployment string `json:"deployment,omitempty"`
}

// IsLocal reports whether the entry names a local engine and model.
func (e TierEntry) IsLocal() bool { return e.Kind == EntryLocal }

// TierPolicy controls fallback between tiers.
type TierPolicy struct {
	// AllowStronger lets a stronger tier's model answer when no model in the
	// requested tier is eligible.
	AllowStronger bool `json:"allowStronger"`
	// AllowWeaker lets a weaker tier's model answer as a last resort.
	AllowWeaker bool `json:"allowWeaker"`
	// PreferLoaded ranks candidates already loaded on their node ahead of cold
	// ones. "tier" keeps tier order first and prefers loaded models within each
	// tier; "any" prefers any loaded eligible model over a cold one.
	PreferLoaded string `json:"preferLoaded"`
}

// Default returns the policy a node starts with when it has no policy file.
// It matches upstream behavior except where the fork's goals say otherwise:
// one resident model per machine, with idle models switched out on demand.
func Default() Policy {
	return Policy{
		Version:      SchemaVersion,
		Revision:     "0",
		Availability: Available,
		Pause: PauseConfig{
			OnActive:            FinishActive,
			DrainTimeoutSeconds: 120,
			UnloadModels:        true,
			StopEngines:         false,
		},
		Admission: Admission{
			MaxResidentModels:     1,
			MaxConcurrentPerModel: 0,
			QueueTimeoutSeconds:   30,
			MemoryBudgetBytes:     0,
			SwitchModels:          true,
			SwitchTimeoutSeconds:  60,
		},
		Idle: IdleConfig{
			UnloadAfterMinutes:     0,
			StopEngineAfterMinutes: 0,
			StartOnDemand:          true,
			WakeTimeoutSeconds:     90,
		},
		Profiles: []Profile{},
		Roles:    map[string]Role{},
		Tiers:    map[string][]TierEntry{},
		TierPolicy: TierPolicy{
			AllowStronger: true,
			AllowWeaker:   false,
			PreferLoaded:  "tier",
		},
		Routing: defaultRouting(),
	}
}

var profileName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$`)

// Validate reports the first problem with p, or nil. The broker validates
// every write and refuses an invalid policy; the proxy validates what it is
// pushed and keeps its previous policy on failure.
func (p Policy) Validate() error {
	if p.Version != SchemaVersion {
		return fmt.Errorf("unsupported policy version %d (want %d)", p.Version, SchemaVersion)
	}
	if err := validateRevision(p.Revision); err != nil {
		return err
	}
	switch p.Availability {
	case Available, Paused:
	case Draining:
		return fmt.Errorf("availability %q is transient and cannot be saved", p.Availability)
	default:
		return fmt.Errorf("unknown availability %q", p.Availability)
	}
	switch {
	case p.Availability == Available && p.PausedBy != "":
		return fmt.Errorf("pausedBy must be empty while the node is available")
	case p.PausedBy != "" && p.PausedBy != PausedLocally && p.PausedBy != PausedByPeer:
		return fmt.Errorf("pausedBy must be %q or %q", PausedLocally, PausedByPeer)
	}
	switch p.Pause.OnActive {
	case FinishActive, CancelActive:
	default:
		return fmt.Errorf("pause.onActive must be %q or %q", FinishActive, CancelActive)
	}
	for name, v := range map[string]int{
		"pause.drainTimeoutSeconds":       p.Pause.DrainTimeoutSeconds,
		"admission.maxResidentModels":     p.Admission.MaxResidentModels,
		"admission.maxConcurrentPerModel": p.Admission.MaxConcurrentPerModel,
		"admission.queueTimeoutSeconds":   p.Admission.QueueTimeoutSeconds,
		"admission.switchTimeoutSeconds":  p.Admission.SwitchTimeoutSeconds,
		"idle.unloadAfterMinutes":         p.Idle.UnloadAfterMinutes,
		"idle.stopEngineAfterMinutes":     p.Idle.StopEngineAfterMinutes,
		"idle.wakeTimeoutSeconds":         p.Idle.WakeTimeoutSeconds,
	} {
		if v < 0 {
			return fmt.Errorf("%s must not be negative", name)
		}
	}
	if p.Admission.MemoryBudgetBytes < 0 {
		return fmt.Errorf("admission.memoryBudgetBytes must not be negative")
	}
	seen := map[string]bool{}
	models := map[string]bool{}
	for i, pr := range p.Profiles {
		if !profileName.MatchString(pr.Name) {
			return fmt.Errorf("profiles[%d]: invalid name %q", i, pr.Name)
		}
		if seen[pr.Name] {
			return fmt.Errorf("profiles[%d]: duplicate name %q", i, pr.Name)
		}
		seen[pr.Name] = true
		if err := checkModelRef(pr.Engine, pr.Model, pr.ContextTokens, pr.Capabilities); err != nil {
			return fmt.Errorf("profiles[%d] %q: %w", i, pr.Name, err)
		}
		key := pr.Engine + "\x00" + pr.Model
		if models[key] {
			return fmt.Errorf("profiles[%d] %q: another profile already covers %s %q", i, pr.Name, pr.Engine, pr.Model)
		}
		models[key] = true
		if pr.MemoryBytes < 0 || pr.MaxConcurrent < 0 {
			return fmt.Errorf("profiles[%d] %q: memoryBytes and maxConcurrent must not be negative", i, pr.Name)
		}
		if len(pr.RequestOptions) > 0 {
			var obj map[string]json.RawMessage
			if err := json.Unmarshal(pr.RequestOptions, &obj); err != nil {
				return fmt.Errorf("profiles[%d] %q: requestOptions must be a JSON object", i, pr.Name)
			}
			if _, ok := obj["model"]; ok {
				return fmt.Errorf("profiles[%d] %q: requestOptions may not set model", i, pr.Name)
			}
		}
	}
	for tier, role := range p.Roles {
		if !slices.Contains(TierOrder, tier) {
			return fmt.Errorf("roles: unknown tier %q (want one of %v)", tier, TierOrder)
		}
		if err := checkModelRef(role.Engine, role.Model, 0, nil); err != nil {
			return fmt.Errorf("roles[%s]: %w", tier, err)
		}
	}
	if err := p.Routing.validate(); err != nil {
		return err
	}
	for tier, entries := range p.Tiers {
		if !slices.Contains(TierOrder, tier) {
			return fmt.Errorf("unknown tier %q (want one of %v)", tier, TierOrder)
		}
		if err := p.Routing.validateTierRefs(tier, entries); err != nil {
			return err
		}
	}
	switch p.TierPolicy.PreferLoaded {
	case "tier", "any":
	default:
		return fmt.Errorf(`tierPolicy.preferLoaded must be "tier" or "any"`)
	}
	return nil
}

// checkModelRef accepts a built-in engine or a valid descriptor engine ID. A
// policy may name an engine that only a peer runs, so the ID is checked for
// syntax here and resolved to candidates only where a facade fronts it.
func checkModelRef(engine, model string, contextTokens int, caps []Capability) error {
	if !engines.ValidRoutingID(engine) {
		return fmt.Errorf("unknown engine %q", engine)
	}
	if model == "" {
		return fmt.Errorf("model is required")
	}
	if IsReservedAlias(model) {
		return fmt.Errorf("model %q is a reserved alias", model)
	}
	if contextTokens < 0 {
		return fmt.Errorf("contextTokens must not be negative")
	}
	for _, c := range caps {
		if !slices.Contains(KnownCapabilities, c) {
			return fmt.Errorf("unknown capability %q", c)
		}
	}
	return nil
}

// IsTier reports whether a request's model field names a tier.
func IsTier(model string) bool { return slices.Contains(TierOrder, model) }

// ProfileFor returns the profile covering (engine, model), if any. model must
// already be normalized the way the engine names it.
func (p Policy) ProfileFor(engine, model string) (Profile, bool) {
	for _, pr := range p.Profiles {
		if pr.Engine == engine && pr.Model == model {
			return pr, true
		}
	}
	return Profile{}, false
}

// validateRevision requires the canonical decimal form money uses: "0" or a
// nonzero leading digit, within the signed 64-bit range.
func validateRevision(rev string) error {
	if _, err := ParseNanoUSD(rev); err != nil {
		return errors.New("revision must be a canonical decimal string: 0 or a nonzero leading digit, within the signed 64-bit range")
	}
	return nil
}

// Parse decodes and validates a policy file, filling fields the file omits
// from Default so an older or hand-written file keeps working. A version 1 file
// is migrated: every local setting is kept, its tier entries become local
// entries, and routing and revision take their defaults, so remote and paid
// execution stay off. Parsing the marshaled result of Parse gives the same
// policy.
func Parse(data []byte) (Policy, error) {
	var head struct {
		Version *int `json:"version"`
	}
	if err := json.Unmarshal(data, &head); err != nil {
		return Policy{}, fmt.Errorf("parse policy: %w", err)
	}
	var p Policy
	var err error
	if head.Version != nil && *head.Version == legacySchemaVersion {
		p, err = migrateV1(data)
	} else {
		p = Default()
		if err = json.Unmarshal(data, &p); err != nil {
			err = fmt.Errorf("parse policy: %w", err)
		}
	}
	if err != nil {
		return Policy{}, err
	}
	p.normalize()
	if err := p.Validate(); err != nil {
		return Policy{}, err
	}
	return p, nil
}

// tierEntryV1 is a version 1 tier entry, which had no kind.
type tierEntryV1 struct {
	Engine        string       `json:"engine"`
	Model         string       `json:"model"`
	ContextTokens int          `json:"contextTokens"`
	Capabilities  []Capability `json:"capabilities"`
}

// migrateV1 decodes a version 1 document onto the defaults and returns it as
// version 2.
func migrateV1(data []byte) (Policy, error) {
	doc := struct {
		Policy
		Tiers map[string][]tierEntryV1 `json:"tiers"`
	}{Policy: Default()}
	if err := json.Unmarshal(data, &doc); err != nil {
		return Policy{}, fmt.Errorf("parse policy: %w", err)
	}
	p := doc.Policy
	def := Default()
	p.Version, p.Revision, p.Routing = def.Version, def.Revision, def.Routing
	p.Tiers = make(map[string][]TierEntry, len(doc.Tiers))
	for tier, entries := range doc.Tiers {
		if entries == nil {
			p.Tiers[tier] = nil
			continue
		}
		// Version 1 accepted a repeated entry. An exact repeat (same engine,
		// model, context bound and capabilities) resolved as the first did, so
		// migration keeps the first rather than rejecting the file. Entries that
		// differ in bounds stay separate.
		migrated := make([]TierEntry, 0, len(entries))
		seen := map[string]bool{}
		for _, e := range entries {
			entry := TierEntry{Kind: EntryLocal, Engine: e.Engine, Model: e.Model, ContextTokens: e.ContextTokens, Capabilities: e.Capabilities}
			if key := entryKey(entry); !seen[key] {
				seen[key] = true
				migrated = append(migrated, entry)
			}
		}
		p.Tiers[tier] = migrated
	}
	return p, nil
}

// normalize makes equal policies compare equal after a JSON round trip: absent
// lists become empty, empty capability lists become nil and request options are
// rewritten the way encoding/json writes them, HTML escaping included.
func (p *Policy) normalize() {
	if p.Profiles == nil {
		p.Profiles = []Profile{}
	}
	for i := range p.Profiles {
		pr := &p.Profiles[i]
		pr.Capabilities = nilIfEmpty(pr.Capabilities)
		if len(pr.RequestOptions) > 0 {
			if canonical, err := json.Marshal(pr.RequestOptions); err == nil {
				pr.RequestOptions = canonical
			}
		}
	}
	if p.Roles == nil {
		p.Roles = map[string]Role{}
	}
	if p.Tiers == nil {
		p.Tiers = map[string][]TierEntry{}
	}
	for _, entries := range p.Tiers {
		for i := range entries {
			entries[i].Capabilities = nilIfEmpty(entries[i].Capabilities)
		}
	}
	p.Routing.normalize()
}
