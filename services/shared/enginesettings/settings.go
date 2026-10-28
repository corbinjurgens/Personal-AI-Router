// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package enginesettings is the node-owned engine settings wire contract.
package enginesettings

import (
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"slices"

	"nvpair-shared/engines"
	"nvpair-shared/modelconfig"
)

type Config struct {
	ServerPort int `json:"serverPort"`
	ProxyPort  int `json:"proxyPort"`
	// Arguments and leading environment assignments; excludes the executable
	// and manifest-owned startup subcommand. Snapshot.Format identifies syntax.
	LaunchText string                  `json:"launchText"`
	Serving    *engines.ServingConfig  `json:"serving,omitempty"`
	ModelLoads []modelconfig.ModelLoad `json:"modelLoads,omitempty"`
}

func (c Config) Equal(other Config) bool {
	return c.ServerPort == other.ServerPort && c.ProxyPort == other.ProxyPort && c.LaunchText == other.LaunchText && c.Serving.Equal(other.Serving) && modelconfig.EqualLoads(c.ModelLoads, other.ModelLoads)
}

func (c Config) Clone() Config {
	c.Serving = c.Serving.Clone()
	c.ModelLoads = modelconfig.CloneLoads(c.ModelLoads)
	return c
}

// Canonical validates and sorts model loads, and checks the complete settings
// payload against the existing transport budget.
func (c Config) Canonical() (Config, error) {
	loads, err := modelconfig.CanonicalLoads(c.ModelLoads)
	if err != nil {
		return Config{}, err
	}
	c = c.Clone()
	c.ModelLoads = loads
	if err := modelconfig.CheckPayloadSize(c); err != nil {
		return Config{}, fmt.Errorf("settings: %w", err)
	}
	return c, nil
}

func (s Snapshot) Clone() Snapshot {
	s.Settings = s.Settings.Clone()
	s.Serving = s.Serving.Clone()
	if s.EffectiveSettings != nil {
		c := s.EffectiveSettings.Clone()
		s.EffectiveSettings = &c
	}
	return s
}

// ConfigFromLaunch preserves the complete manager-owned declaration.
func ConfigFromLaunch(s LaunchState, proxyPort int) Config {
	return Config{ServerPort: s.ServerPort, ProxyPort: proxyPort, LaunchText: s.LaunchText, Serving: s.Serving.Clone(), ModelLoads: modelconfig.CloneLoads(s.ModelLoads)}
}

type Request struct {
	NodeID           string                   `json:"nodeId,omitempty"`
	Engine           string                   `json:"engine"`
	ExpectedRevision uint64                   `json:"expectedRevision"`
	RequestID        string                   `json:"requestId,omitempty"`
	Settings         Config                   `json:"settings"`
	ModelChange      *modelconfig.ModelChange `json:"modelChange,omitempty"`
	Resolution       string                   `json:"resolution,omitempty"`
	// Format is supplied only when migrating a saved full-command journal.
	Format string `json:"format,omitempty"`
	// Internal preview guard, overwritten by the broker from the authenticated
	// caller. Client input cannot disable the target's local-only CORS policy.
	PreserveCORS bool `json:"preserveCORS,omitempty"`
}

// WorkerRequest is parent-only operation input. Public Request decoding rejects
// every field in this envelope; none is portable settings or journal material.
type WorkerRequest[T any] struct {
	Request           T                         `json:"request"`
	Memory            *modelconfig.TargetMemory `json:"memory,omitempty"`
	NativeInitAllowed bool                      `json:"nativeInitAllowed"`
	ValidateResources bool                      `json:"validateResources,omitempty"`
	ResourceModel     string                    `json:"resourceModel,omitempty"`
}

func DecodeWorkerRequest[T any](data []byte) (WorkerRequest[T], error) {
	var p WorkerRequest[T]
	if err := modelconfig.DecodeStrict(data, &p); err != nil {
		return p, err
	}
	if len(p.ResourceModel) > 256 || p.ValidateResources != (p.ResourceModel != "") {
		return p, fmt.Errorf("worker resource validation requires an exact selected model")
	}
	if p.Memory != nil {
		if err := p.Memory.Validate(); err != nil {
			return p, err
		}
	}
	return p, nil
}

// ValidateModelChange checks the target baseline and review prerequisites
// before a caller enters the journal or performs runtime effects.
func (r Request) ValidateModelChange(baseline Config, applying bool) error {
	if err := r.Validate(); err != nil {
		return err
	}
	// Full settings is a baseline echo for model loads. Only selected portable
	// modelChange edits can alter them, including on ordinary port/launch writes.
	// This rejects new/replaced caller resolutions, paths, builds and allocations.
	if !modelconfig.EqualLoads(r.Settings.ModelLoads, baseline.ModelLoads) {
		return fmt.Errorf("settings.modelLoads: target-owned baseline required; use modelChange for portable edits")
	}
	if applying && r.RequestID == "" {
		return fmt.Errorf("requestId: required for apply")
	}
	if r.ModelChange == nil {
		return nil
	}
	if !r.Settings.Equal(baseline) {
		return fmt.Errorf("settings: modelChange requires the target baseline")
	}
	if applying && r.ModelChange.PreviewDigest == "" {
		return fmt.Errorf("modelChange.previewDigest: reviewed preview required")
	}
	return nil
}

// Validate checks shape/portable semantics and the COMPLETE request budget on
// every path, including ordinary full settings. Consumers must additionally call
// ValidateModelChange with the fresh target baseline before acceptance/effects.
// These checks do not establish permission, readiness or a valid target preview.
func (r Request) Validate() error {
	if r.ExpectedRevision > MaxSafeCounter {
		return fmt.Errorf("expectedRevision: exceeds exact JSON integer range")
	}
	if !engines.ValidRoutingID(r.Engine) {
		return fmt.Errorf("engine: invalid routing ID")
	}
	if len(r.NodeID) > 256 || len(r.RequestID) > 128 || len(r.Resolution) > 128 || len(r.Format) > 128 {
		return fmt.Errorf("request: identifier exceeds budget")
	}
	if _, err := r.Settings.Canonical(); err != nil {
		return err
	}
	if r.ModelChange != nil {
		if err := r.ModelChange.Validate(); err != nil {
			return err
		}
		if r.ModelChange.Policy != nil {
			if _, selected := r.ModelChange.Policy.Generation["min_p"]; selected && r.Engine != "llamacpp" {
				return fmt.Errorf("modelChange.policy.generation.min_p: llama.cpp only")
			}
		}
	}
	return modelconfig.CheckPayloadSize(r)
}

// DecodeRequest is the public untrusted settings boundary. It does not authorize
// effects. An unchanged Config may echo existing resolved readback; baseline
// comparison in ValidateModelChange must reject any supplied replacement.
// Internal launch/configure traffic and journal readback use their own types.
func DecodeRequest(data []byte) (Request, error) {
	var r Request
	if err := modelconfig.DecodeStrict(data, &r); err != nil {
		return Request{}, err
	}
	if r.PreserveCORS {
		return Request{}, fmt.Errorf("preserveCORS: internal caller guard is not public authority")
	}
	if err := r.Validate(); err != nil {
		return Request{}, err
	}
	c, err := r.Settings.Canonical()
	if err != nil {
		return Request{}, err
	}
	r.Settings = c
	if r.ModelChange != nil {
		change := r.ModelChange.Clone()
		if change.Load != nil {
			slices.Sort(change.Load.Clear)
		}
		r.ModelChange = &change
	}
	return r, nil
}

// DecodeConfig is for target-owned stored/readback settings, which may include
// resolutions. It does not turn a decoded record into caller authority.
func DecodeConfig(data []byte) (Config, error) {
	var c Config
	if err := modelconfig.DecodeStrict(data, &c); err != nil {
		return Config{}, err
	}
	return c.Canonical()
}

func (r Request) Clone() Request {
	r.Settings = r.Settings.Clone()
	if r.ModelChange != nil {
		c := r.ModelChange.Clone()
		r.ModelChange = &c
	}
	return r
}

type Snapshot struct {
	Serving             *engines.ServingReadback `json:"serving,omitempty"`
	NodeID              string                   `json:"nodeId"`
	Engine              string                   `json:"engine"`
	Revision            uint64                   `json:"revision"`
	AppliedRevision     uint64                   `json:"appliedRevision"`
	Sequence            uint64                   `json:"sequence"`
	Epoch               string                   `json:"epoch"`
	Settings            Config                   `json:"settings"`
	EffectiveServerPort int                      `json:"effectiveServerPort"`
	EffectiveProxyPort  int                      `json:"effectiveProxyPort"`
	Running             bool                     `json:"running"`
	Adopted             bool                     `json:"adopted"`
	Editable            bool                     `json:"editable"`
	Reason              string                   `json:"reason,omitempty"`
	Format              string                   `json:"format"`
	Phase               string                   `json:"phase"`
	Error               string                   `json:"error,omitempty"`
	RequestID           string                   `json:"requestId,omitempty"`
	CoupledStage        string                   `json:"coupledStage,omitempty"`
	PolicyRevision      string                   `json:"policyRevision,omitempty"`
	UnresolvedError     string                   `json:"unresolvedError,omitempty"`
	EffectiveSettings   *Config                  `json:"effectiveSettings,omitempty"`
}

type Conflict struct {
	ServerPort int `json:"serverPort"`
	LaunchPort int `json:"launchPort"`
}

type Preview struct {
	Serving           *engines.ServingReadback      `json:"serving,omitempty"`
	Settings          Config                        `json:"settings"`
	Revision          uint64                        `json:"revision"`
	Conflict          *Conflict                     `json:"conflict,omitempty"`
	Errors            map[string]string             `json:"errors,omitempty"`
	Restart           bool                          `json:"restart"`
	Rebind            bool                          `json:"rebind"`
	PolicyRevision    string                        `json:"policyRevision,omitempty"`
	PreviewDigest     string                        `json:"previewDigest,omitempty"`
	Diff              []modelconfig.FieldDiff       `json:"diff,omitempty"`
	RequiredArtifacts []modelconfig.ArtifactSummary `json:"requiredArtifacts,omitempty"`
	Estimate          *modelconfig.Estimate         `json:"estimate,omitempty"`
	Impact            *modelconfig.Impact           `json:"impact,omitempty"`
	Auto              *modelconfig.AutoProposal     `json:"auto,omitempty"`
	// Named to match Runtime's persisted override keys: these carry the same
	// user-supplied launch content, and a log sink redacts them by key. Plain
	// "args"/"env" would collide with every worker-spawn diagnostic.
	Args []string `json:"launch_args"`
	Env  []string `json:"launch_env"`
}

type LaunchState struct {
	ModelLoads      []modelconfig.ModelLoad  `json:"modelLoads,omitempty"`
	Serving         *engines.ServingConfig   `json:"serving,omitempty"`
	ServingReadback *engines.ServingReadback `json:"servingReadback,omitempty"`
	Engine          string                   `json:"engine"`
	ServerPort      int                      `json:"serverPort"`
	EffectivePort   int                      `json:"effectivePort"`
	LaunchText      string                   `json:"launchText"`
	Running         bool                     `json:"running"`
	Adopted         bool                     `json:"adopted"`
	Editable        bool                     `json:"editable"`
	Reason          string                   `json:"reason,omitempty"`
	Format          string                   `json:"format"`
}

type Configure struct {
	Engine      string `json:"engine"`
	Settings    Config `json:"settings"`
	OperationID string `json:"operationId"`
	Resume      bool   `json:"resume,omitempty"`
}

// Relay is a correlated worker-to-parent request. It never blocks a reader.
type Relay struct {
	ID      string                      `json:"id"`
	Method  string                      `json:"method"`
	Caller  string                      `json:"caller,omitempty"`
	Request Request                     `json:"request,omitempty"`
	Inspect *modelconfig.InspectRequest `json:"inspect,omitempty"`
}

// Validate enforces the correlated union. Inspect uses its actual read shape;
// it cannot smuggle a settings mutation beside the read request. Caller is filled
// from authenticated transport by the parent, never accepted as peer authority.
func (r Relay) Validate() error {
	if r.ID == "" || len(r.ID) > 128 || len(r.Caller) > 256 {
		return fmt.Errorf("relay: bounded correlation identity required")
	}
	if r.Method == "inspect" {
		if r.Inspect == nil || !reflect.DeepEqual(r.Request, Request{}) {
			return fmt.Errorf("relay.inspect: inspect request only")
		}
		if err := r.Inspect.Validate(); err != nil {
			return err
		}
	} else {
		if !slices.Contains([]string{"get", "preview", "apply", "rebind"}, r.Method) || r.Inspect != nil {
			return fmt.Errorf("relay: settings method/request mismatch")
		}
		if err := r.Request.Validate(); err != nil {
			return err
		}
	}
	return modelconfig.CheckPayloadSize(r)
}
func DecodeRelay(data []byte) (Relay, error) {
	var r Relay
	if err := modelconfig.DecodeStrict(data, &r); err != nil {
		return Relay{}, err
	}
	if err := r.Validate(); err != nil {
		return Relay{}, err
	}
	if r.Method != "inspect" {
		c, err := r.Request.Settings.Canonical()
		if err != nil {
			return Relay{}, err
		}
		r.Request.Settings = c
	}
	return r, nil
}
func (r Relay) Clone() Relay {
	r.Request = r.Request.Clone()
	if r.Inspect != nil {
		q := *r.Inspect
		r.Inspect = &q
	}
	return r
}

// Existing engine counters retain their numeric wire shape. Refuse values that
// a JavaScript consumer cannot represent, rather than silently rounding them.
// New policy revisions and byte quantities use decimal strings.
const MaxSafeCounter uint64 = 1<<53 - 1

func (r Request) MarshalJSON() ([]byte, error) {
	if r.ExpectedRevision > MaxSafeCounter {
		return nil, fmt.Errorf("expectedRevision: exceeds exact JSON integer range")
	}
	type wire Request
	return json.Marshal(wire(r))
}

func (r *Request) UnmarshalJSON(data []byte) error {
	type wire Request
	var value wire
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	if value.ExpectedRevision > MaxSafeCounter {
		return fmt.Errorf("expectedRevision: exceeds exact JSON integer range")
	}
	*r = Request(value)
	return nil
}

func (s Snapshot) MarshalJSON() ([]byte, error) {
	if s.Revision > MaxSafeCounter || s.AppliedRevision > MaxSafeCounter || s.Sequence > MaxSafeCounter {
		return nil, fmt.Errorf("settings snapshot counter exceeds exact JSON integer range")
	}
	type wire Snapshot
	return json.Marshal(wire(s))
}

func (s *Snapshot) UnmarshalJSON(data []byte) error {
	type wire Snapshot
	var value wire
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	if value.Revision > MaxSafeCounter || value.AppliedRevision > MaxSafeCounter || value.Sequence > MaxSafeCounter {
		return fmt.Errorf("settings snapshot counter exceeds exact JSON integer range")
	}
	*s = Snapshot(value)
	return nil
}

func (p Preview) MarshalJSON() ([]byte, error) {
	if p.Revision > MaxSafeCounter {
		return nil, fmt.Errorf("settings preview revision exceeds exact JSON integer range")
	}
	type wire Preview
	return json.Marshal(wire(p))
}

func (p *Preview) UnmarshalJSON(data []byte) error {
	type wire Preview
	var value wire
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	if value.Revision > MaxSafeCounter {
		return fmt.Errorf("settings preview revision exceeds exact JSON integer range")
	}
	*p = Preview(value)
	return nil
}

func (r Relay) MarshalJSON() ([]byte, error) {
	// Keep the pre-existing settings request Go shape for its callers while
	// making inspect a distinct correlated read, with no mutation payload.
	if r.Method == "inspect" && reflect.DeepEqual(r.Request, Request{}) {
		return json.Marshal(struct {
			ID      string                      `json:"id"`
			Method  string                      `json:"method"`
			Caller  string                      `json:"caller,omitempty"`
			Inspect *modelconfig.InspectRequest `json:"inspect"`
		}{r.ID, r.Method, r.Caller, r.Inspect})
	}
	type wire Relay
	return json.Marshal(wire(r))
}

func (p Preview) Clone() Preview {
	p.Serving = p.Serving.Clone()
	p.Settings = p.Settings.Clone()
	if p.Conflict != nil {
		c := *p.Conflict
		p.Conflict = &c
	}
	p.Errors = maps.Clone(p.Errors)
	p.Diff = slices.Clone(p.Diff)
	for i := range p.Diff {
		p.Diff[i].Before = slices.Clone(p.Diff[i].Before)
		p.Diff[i].After = slices.Clone(p.Diff[i].After)
	}
	p.RequiredArtifacts = slices.Clone(p.RequiredArtifacts)
	if p.Estimate != nil {
		e := p.Estimate.Clone()
		p.Estimate = &e
	}
	if p.Impact != nil {
		i := p.Impact.Clone()
		p.Impact = &i
	}
	if p.Auto != nil {
		a := p.Auto.Clone()
		p.Auto = &a
	}
	p.Args, p.Env = slices.Clone(p.Args), slices.Clone(p.Env)
	return p
}
