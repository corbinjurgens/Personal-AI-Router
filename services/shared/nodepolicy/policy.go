// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nodepolicy is the fork's node policy contract: whether this machine
// accepts inference work (availability), how it admits work (resident models,
// concurrency, memory), how idle engines are unloaded or stopped, the model
// profiles it serves, and the weak/medium/strong tiers it routes by.
//
// The broker owns the policy file and is the only writer. It pushes the policy
// to the proxy, which enforces admission and resolves tiers. Both import this
// package so the two sides cannot drift. See FORK_DESIGN.md at the repo root.
package nodepolicy

import (
	"encoding/json"
	"fmt"
	"regexp"
	"slices"

	"nvpair-shared/engines"
)

// FileName is the policy file under the per-user app data dir (appdir.Path).
const FileName = "node-policy.json"

// SchemaVersion is the current policy file version.
const SchemaVersion = 1

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
	Version      int          `json:"version"`
	Availability Availability `json:"availability"`
	Pause        PauseConfig  `json:"pause"`
	Admission    Admission    `json:"admission"`
	Idle         IdleConfig   `json:"idle"`
	// Profiles describe the models this node serves. Admission looks a request
	// up by (engine, model); a model with no profile is admitted with the
	// node-wide defaults.
	Profiles []Profile `json:"profiles"`
	// Tiers map each tier name to its candidates, most preferred first. Tiers
	// are resolved by the node a request enters, across every node in the
	// cluster that has a candidate model installed.
	Tiers      map[string][]TierEntry `json:"tiers"`
	TierPolicy TierPolicy             `json:"tierPolicy"`
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

// TierEntry is one candidate model for a tier.
type TierEntry struct {
	Engine        string       `json:"engine"`
	Model         string       `json:"model"`
	ContextTokens int          `json:"contextTokens,omitempty"`
	Capabilities  []Capability `json:"capabilities,omitempty"`
}

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
		Tiers:    map[string][]TierEntry{},
		TierPolicy: TierPolicy{
			AllowStronger: true,
			AllowWeaker:   false,
			PreferLoaded:  "tier",
		},
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
	switch p.Availability {
	case Available, Paused:
	case Draining:
		return fmt.Errorf("availability %q is transient and cannot be saved", p.Availability)
	default:
		return fmt.Errorf("unknown availability %q", p.Availability)
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
	for tier, entries := range p.Tiers {
		if !slices.Contains(TierOrder, tier) {
			return fmt.Errorf("unknown tier %q (want one of %v)", tier, TierOrder)
		}
		for i, e := range entries {
			if err := checkModelRef(e.Engine, e.Model, e.ContextTokens, e.Capabilities); err != nil {
				return fmt.Errorf("tiers.%s[%d]: %w", tier, i, err)
			}
		}
	}
	switch p.TierPolicy.PreferLoaded {
	case "tier", "any":
	default:
		return fmt.Errorf(`tierPolicy.preferLoaded must be "tier" or "any"`)
	}
	return nil
}

func checkModelRef(engine, model string, contextTokens int, caps []Capability) error {
	if _, ok := engines.ByName(engine); !ok {
		return fmt.Errorf("unknown engine %q", engine)
	}
	if model == "" {
		return fmt.Errorf("model is required")
	}
	if slices.Contains(TierOrder, model) {
		return fmt.Errorf("model %q is a tier name", model)
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

// Parse decodes and validates a policy file, filling fields the file omits
// from Default so an older or hand-written file keeps working.
func Parse(data []byte) (Policy, error) {
	p := Default()
	if err := json.Unmarshal(data, &p); err != nil {
		return Policy{}, fmt.Errorf("parse policy: %w", err)
	}
	if p.Profiles == nil {
		p.Profiles = []Profile{}
	}
	if p.Tiers == nil {
		p.Tiers = map[string][]TierEntry{}
	}
	if err := p.Validate(); err != nil {
		return Policy{}, err
	}
	return p, nil
}
