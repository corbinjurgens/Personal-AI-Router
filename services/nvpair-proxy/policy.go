// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// The broker → proxy node policy requests (nodepolicy/wire.go). All five are
// process-scoped: the policy is the machine's, not an engine's, so none of
// them is addressed to a facade. Each validates before applying, keeps the
// previous state on invalid input, and answers {"ok":true} or a JSON-RPC
// error. Until the broker sends anything the controller runs on
// nodepolicy.Default() with availability Available.

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

// handlePolicyMethod handles one node policy request, reporting false when the
// method is not one of them.
func (p *Proxy) handlePolicyMethod(msg *Message, method string) bool {
	var err error
	switch method {
	case nodepolicy.MethodSetPolicy:
		err = p.applySetPolicy(msg.Params)
	case nodepolicy.MethodSetAvailability:
		err = p.applySetAvailability(msg.Params)
	case nodepolicy.MethodSetEngineDrain:
		err = p.applySetEngineDrain(msg.Params)
	case nodepolicy.MethodSetResidency:
		err = p.applySetResidency(msg.Params)
	case nodepolicy.MethodSetEngineIntent:
		err = p.applySetEngineIntent(msg.Params)
	default:
		return false
	}
	if err != nil {
		slog.Warn("node policy request rejected", "method", method, "err", err)
		_ = p.codec.RespondError(msg.ID, -32602, err.Error())
		return true
	}
	_ = p.codec.Respond(msg.ID, map[string]bool{"ok": true})
	return true
}

func (p *Proxy) applySetPolicy(params json.RawMessage) error {
	if len(params) == 0 {
		return fmt.Errorf("params must be a node policy object")
	}
	policy, err := nodepolicy.Parse(params)
	if err != nil {
		return err
	}
	p.admission.setPolicy(policy)
	slog.Info("node policy applied",
		"profiles", len(policy.Profiles), "tiers", len(policy.Tiers),
		"max_resident", policy.Admission.MaxResidentModels,
		"switch_models", policy.Admission.SwitchModels)
	return nil
}

func (p *Proxy) applySetAvailability(params json.RawMessage) error {
	var in nodepolicy.SetAvailabilityParams
	if err := decodeStrict(params, &in); err != nil {
		return err
	}
	switch in.State {
	case nodepolicy.Available, nodepolicy.Draining, nodepolicy.Paused:
	default:
		return fmt.Errorf("state must be %q, %q or %q", nodepolicy.Available, nodepolicy.Draining, nodepolicy.Paused)
	}
	p.admission.setAvailability(in.State, in.CancelActive)
	slog.Info("node availability set", "state", in.State, "cancel_active", in.CancelActive)
	return nil
}

func (p *Proxy) applySetEngineDrain(params json.RawMessage) error {
	var in nodepolicy.SetEngineDrainParams
	if err := decodeStrict(params, &in); err != nil {
		return err
	}
	if err := knownEngine(in.Engine); err != nil {
		return err
	}
	p.admission.setDrain(in.Engine, in.Drain)
	slog.Info("engine drain set", "engine", in.Engine, "drain", in.Drain)
	return nil
}

func (p *Proxy) applySetResidency(params json.RawMessage) error {
	var in nodepolicy.SetResidencyParams
	if err := decodeStrict(params, &in); err != nil {
		return err
	}
	if in.LoadedByEngine == nil {
		return fmt.Errorf("loadedByEngine is required")
	}
	for engine, models := range in.LoadedByEngine {
		if err := knownEngine(engine); err != nil {
			return err
		}
		for _, m := range models {
			if strings.TrimSpace(m) == "" {
				return fmt.Errorf("loadedByEngine.%s contains an empty model", engine)
			}
		}
	}
	p.admission.setResidency(in.LoadedByEngine)
	slog.Debug("node residency set", "engines", len(in.LoadedByEngine))
	return nil
}

func (p *Proxy) applySetEngineIntent(params json.RawMessage) error {
	var in nodepolicy.SetEngineIntentParams
	if err := decodeStrict(params, &in); err != nil {
		return err
	}
	if in.EnabledByEngine == nil {
		return fmt.Errorf("enabledByEngine is required")
	}
	for engine := range in.EnabledByEngine {
		if err := knownEngine(engine); err != nil {
			return err
		}
	}
	p.admission.setIntent(in.EnabledByEngine)
	slog.Info("engine intent set", "engines", len(in.EnabledByEngine))
	return nil
}

// decodeStrict decodes a params object, refusing unknown fields so a
// misspelled field is an error rather than a silently ignored setting.
func decodeStrict(params json.RawMessage, v any) error {
	if len(params) == 0 {
		return fmt.Errorf("params are required")
	}
	dec := json.NewDecoder(strings.NewReader(string(params)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("invalid params: %w", err)
	}
	return nil
}

func knownEngine(name string) error {
	if _, ok := engines.ByName(name); !ok {
		return fmt.Errorf("unknown engine %q", name)
	}
	return nil
}

// normalizeEngineModel names a model the way its engine does, so admission,
// residency and profiles agree with nodeAdvertisesModel.
func normalizeEngineModel(engine, model string) string {
	if prof, ok := profileFor(engine); ok {
		return prof.normalizeModel(model)
	}
	return strings.TrimSpace(model)
}

// localEngineHealthy reports whether this node's engine is up, as
// node/set-local-backend last said. An engine with no facade here has no local
// backend the proxy can reach, so it counts as stopped.
func (p *Proxy) localEngineHealthy(engine string) bool {
	f := p.facadeFor(engine)
	if f == nil {
		return false
	}
	_, ok := f.localBackendTarget()
	return ok
}
