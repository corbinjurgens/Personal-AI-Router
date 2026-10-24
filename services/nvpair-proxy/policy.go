// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// The broker → proxy node policy requests (nodepolicy/wire.go). All are
// process-scoped: the policy is the machine's, not an engine's, so none of
// them is addressed to a facade. Each validates before applying, keeps the
// previous state on invalid input, and answers {"ok":true} or a JSON-RPC
// error, with typed acknowledgements and snapshot results where specified. Until the
// broker sends anything the controller runs on the content of
// nodepolicy.Default() with availability Available and no active revision, so
// the broker's first acknowledged revision is the first the proxy holds.

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"strings"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

// bufferSettings checks conversions and old/new body overlap plus the separate
// overflow probe before a policy can change admission or process authority.
func (p *Proxy) bufferSettings(policy nodepolicy.Policy) (int64, time.Duration, error) {
	limits := policy.Routing.Buffers
	mib, seconds := int64(limits.CapacityMiB), int64(limits.WaitSeconds)
	if mib < 0 || mib > math.MaxInt64/(1<<20) || seconds < 0 || seconds > math.MaxInt64/int64(time.Second) {
		return 0, 0, fmt.Errorf("buffer settings exceed integer bounds")
	}
	capacity := mib * (1 << 20)
	individual := p.requestBodyLimit()
	if individual > (math.MaxInt64-1)/2 || capacity < 2*individual+1 {
		return 0, 0, fmt.Errorf("buffer capacity must cover twice the individual body limit plus one byte")
	}
	return capacity, time.Duration(seconds) * time.Second, nil
}

// handlePolicyMethod handles one node policy request, reporting false when the
// method is not one of them.
func (p *Proxy) handlePolicyMethod(msg *Message, method string) bool {
	var err error
	var result any = map[string]bool{"ok": true}
	switch method {
	case nodepolicy.MethodRenewEgressLease:
		result, err = p.renewEgressLease(msg.Params)
	case nodepolicy.MethodRevokeEgressLease:
		result, err = p.revokeEgressLease(msg.Params)
	case nodepolicy.MethodSetPolicy:
		var revision string
		revision, err = p.applySetPolicy(msg.Params)
		p.identityMu.Lock()
		result = nodepolicy.PolicyAck{OriginNodeID: p.originNodeID, Revision: revision, RunID: p.runID}
		p.identityMu.Unlock()
	case nodepolicy.MethodSetAvailability:
		err = p.applySetAvailability(msg.Params)
	case nodepolicy.MethodSetEngineDrain:
		err = p.applySetEngineDrain(msg.Params)
	case nodepolicy.MethodSetResidency:
		err = p.applySetResidency(msg.Params)
	case nodepolicy.MethodInvalidateEngine:
		result, err = p.applyInvalidateEngine(msg.Params)
	case nodepolicy.MethodAdmissionSnapshot:
		result, err = p.applyAdmissionSnapshot(msg.Params)
	case nodepolicy.MethodSetEngineIntent:
		err = p.applySetEngineIntent(msg.Params)
	case nodepolicy.MethodSetModelIdentity:
		err = p.applySetModelIdentity(msg.Params)
	case nodepolicy.MethodPrepareResident:
		var in nodepolicy.PrepareResidentParams
		err = decodeStrict(msg.Params, &in)
		if err == nil {
			result, err = p.admission.prepareResident(p.runID, in)
		}
	case nodepolicy.MethodDispatchResident:
		var in nodepolicy.DispatchResidentParams
		err = decodeStrict(msg.Params, &in)
		if err == nil {
			result, err = p.admission.dispatchResident(p.runID, in)
		}
	case nodepolicy.MethodFinishResident:
		var in nodepolicy.FinishResidentParams
		err = decodeStrict(msg.Params, &in)
		if err == nil {
			result, err = p.admission.finishResident(p.runID, in)
		}
	case nodepolicy.MethodObserveResident:
		var in nodepolicy.ObserveResidentParams
		err = decodeStrict(msg.Params, &in)
		if err == nil {
			result, err = p.admission.observeResident(p.runID, in)
		}
	default:
		return false
	}
	if err != nil {
		slog.Warn("node policy request rejected", "method", method, "err", err)
		_ = p.codec.RespondError(msg.ID, -32602, err.Error())
		return true
	}
	_ = p.codec.Respond(msg.ID, result)
	if (method == nodepolicy.MethodSetPolicy || method == nodepolicy.MethodSetModelIdentity) && p.routingIdentityReady() {
		// The broker reads readiness against the acknowledgement just sent; a
		// new identity can install or withdraw the capability.
		p.reportStrictReadiness()
	}
	return true
}

// applySetModelIdentity replaces the engine identity strict capacity and
// qualification are bound to, rederiving the capability.
func (p *Proxy) applySetModelIdentity(params json.RawMessage) error {
	var in nodepolicy.SetModelIdentityParams
	if err := decodeStrict(params, &in); err != nil {
		return err
	}
	if err := in.Validate(); err != nil {
		return err
	}
	for engine := range in.Engines {
		if err := knownEngine(engine); err != nil {
			return err
		}
	}
	changed := p.admission.setModelIdentity(in.Engines)
	slog.Info("engine identity set", "engines", len(in.Engines), "strict_capability_changed", changed)
	return nil
}

// applySetPolicy activates the policy and returns its revision. A revision
// older than the active one is refused and the previous policy stays.
func (p *Proxy) applySetPolicy(params json.RawMessage) (string, error) {
	var in nodepolicy.SetPolicyParams
	if err := json.Unmarshal(params, &in); err != nil {
		return "", err
	}
	policy := in.Policy
	p.identityMu.Lock()
	if nodepolicy.ValidateRoutingIdentity(in.OriginNodeID, p.runID) != nil ||
		(p.originNodeID != "" && p.originNodeID != in.OriginNodeID) {
		p.identityMu.Unlock()
		return "", fmt.Errorf("proxy origin binding is unavailable or changed")
	}
	token, err := p.beginPublicActivation(policy)
	if err != nil {
		p.identityMu.Unlock()
		return "", err
	}
	p.originNodeID = in.OriginNodeID
	p.identityMu.Unlock()
	// Only one ledger activation may run at once. A newer transition may
	// already have replaced admission while an older durable write finishes.
	// Obsolete waiters must not overwrite the newer account snapshot.
	p.egress.activationMu.Lock()
	defer p.egress.activationMu.Unlock()
	p.egress.mu.Lock()
	current := p.egress.transition == token
	p.egress.mu.Unlock()
	if !current {
		return policy.Revision, nil
	}
	// The account ledger never blocks activation: local routing must follow
	// the policy even when the store is unavailable, and spending fails closed.
	accountErr := p.accounts.applyRouting(policy.Routing, policy.Revision)
	if accountErr != nil {
		slog.Error("provider account ledger unavailable; provider spending is refused", "reason", "ledger-unavailable")
	}
	p.completePublicActivation(token, accountErr)
	p.routing.reconcile()
	slog.Info("node policy applied", "revision", policy.Revision,
		"profiles", len(policy.Profiles), "tiers", len(policy.Tiers),
		"max_resident", policy.Admission.MaxResidentModels,
		"switch_models", policy.Admission.SwitchModels)
	return policy.Revision, nil
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
	clean, err := filterModelInventoryJSON(params, "loadedByEngine")
	if err != nil {
		return err
	}
	var in nodepolicy.SetResidencyParams
	if err := decodeStrict(clean, &in); err != nil {
		return err
	}
	if in.LoadedByEngine == nil {
		return fmt.Errorf("loadedByEngine is required")
	}
	for engine, models := range in.LoadedByEngine {
		if err := knownEngine(engine); err != nil {
			return err
		}
		filtered := make([]string, 0, len(models))
		for _, m := range models {
			if nodepolicy.ValidateModelIdentifier(m) == nil && nodepolicy.ValidateModelIdentifier(p.normalizeEngineModel(engine, m)) == nil {
				filtered = append(filtered, m)
			}
		}
		in.LoadedByEngine[engine] = filtered
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

// knownEngine accepts a built-in engine or a valid descriptor engine ID. Policy
// state may name an engine this process does not front; it then has no
// candidates here rather than invalidating the whole message.
func knownEngine(name string) error {
	if !engines.ValidRoutingID(name) {
		return fmt.Errorf("unknown engine %q", name)
	}
	return nil
}

// normalizeEngineModel names a model the way its engine does, so admission,
// residency and profiles agree with nodeAdvertisesModel.
func normalizeEngineModel(engine, model string) string {
	if nodepolicy.ValidateModelIdentifier(model) != nil {
		return ""
	}
	if prof, ok := profileFor(engine); ok {
		model = prof.normalizeModel(model)
	} else {
		model = strings.TrimSpace(model)
	}
	if nodepolicy.ValidateModelIdentifier(model) != nil {
		return ""
	}
	return model
}

// modelServed is the permission comparison seam. B2 can replace normalization
// here with the shared descriptor naming helper when it lands.
func modelServed(served nodepolicy.ServedModels, engine, model string, normalize func(string, string) string) bool {
	// An omitted model keeps the existing unrestricted-policy behavior; the
	// ingress guard separately refuses an explicitly supplied empty token.
	if model == "" {
		return served == nil
	}
	if nodepolicy.ValidateModelIdentifier(model) != nil {
		return false
	}
	want := normalize(engine, model)
	if nodepolicy.ValidateModelIdentifier(want) != nil {
		return false
	}
	if served == nil {
		return true
	}
	for _, ref := range served {
		if ref.Engine == engine && nodepolicy.ValidateModelIdentifier(ref.Model) == nil && normalize(engine, ref.Model) == want {
			return true
		}
	}
	return false
}

const rejectNotServed nodepolicy.RejectReason = "not-served"

// normalizeEngineModel is normalizeEngineModel extended to the descriptor
// engines this process fronts, whose naming rule lives on their facade. An
// engine without a facade here keeps exact naming.
func (p *Proxy) normalizeEngineModel(engine, model string) string {
	if nodepolicy.ValidateModelIdentifier(model) != nil {
		return ""
	}
	if f := p.facadeFor(engine); f != nil {
		normalized := f.profile.normalizeModel(model)
		if nodepolicy.ValidateModelIdentifier(normalized) != nil {
			return ""
		}
		return normalized
	}
	return normalizeEngineModel(engine, model)
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
