// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"time"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

// bindingOffersLocked reads current target-local evidence. Caller owns
// engineConfigMu, so neither a switch withdrawal nor settings acceptance can
// race registration. No stock port or remembered inventory establishes support.
func (b *Broker) bindingOffersLocked() *nodepolicy.EngineBindingOffers {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	policy := r.current
	eligible := len(policy.EngineBindings) > 0 && !r.recoveryRequired &&
		b.engineSettingsLoaded && b.engineSettingsError == nil &&
		r.live == nodepolicy.Available && !r.managementBlockedLocked() &&
		r.activationLocked(b.policyProxy(), policy.Revision).Activation == activationActive &&
		!r.pendingPush[pushPolicy] && !r.pendingPush[pushAvailability]
	source := r.engineManagerSource
	r.mu.Unlock()
	if !eligible || source == nil || b.pendingSwitchLocked("") != nil {
		return nil
	}
	manager := b.getEngineMgr()
	if manager == nil || !b.engineMgrPairCurrent(manager, source) {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	statuses, err := b.boundStatus(ctx, &boundEngineOperation{manager: manager})
	if err != nil || !b.engineMgrPairCurrent(manager, source) {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.engineManagerSource != source || r.current.Revision != policy.Revision ||
		r.live != nodepolicy.Available || r.managementBlockedLocked() ||
		!b.engineMgrPairCurrent(manager, source) || b.getEngineMgr() != manager ||
		r.pendingPush[pushPolicy] || r.pendingPush[pushAvailability] ||
		r.activationLocked(b.policyProxy(), policy.Revision).Activation != activationActive {
		return nil
	}
	var models []nodepolicy.Role
	for _, ref := range policy.EngineBindings {
		profile, fronted := b.frontedEngineProfile(ref.Engine)
		if profile.Name == ollamaProxyProfile.Name && b.ollamaFacadeIsPendingBackend() {
			continue
		}
		if !fronted || b.engineProxyListenPort(profile) == 0 || r.drains[ref.Engine] ||
			!b.bindingSettingsResolvedLocked(ref.Engine) || !b.bindingPermitted(policy, ref.Engine, ref.Model) {
			continue
		}
		for _, st := range statuses {
			if st.Engine == ref.Engine && *st.Installed && !*st.Locked {
				models = append(models, ref)
			}
		}
	}
	if len(models) == 0 {
		return nil
	}
	return &nodepolicy.EngineBindingOffers{Version: 1, Models: models}
}

func (b *Broker) bindingSettingsResolvedLocked(engine string) bool {
	// A loaded journal with no row means the manager's ordinary configuration,
	// including descriptor engines. A strict qualification fingerprint is not
	// required for concrete preparation. Existing unresolved rows still fence it.
	record := b.engineSettings[engine]
	if record == nil {
		return true
	}
	s := record.Snapshot
	return s.Revision > 0 && s.AppliedRevision == s.Revision && s.Phase != "applying" && s.Phase != "failed"
}

func (b *Broker) bindingPermitted(policy nodepolicy.Policy, engine, model string) bool {
	_, known := b.engineModelNaming(engine)
	if !known {
		return false
	}
	ref := b.modelRef(engine, model)
	if policy.ServedModels == nil {
		return true
	}
	for _, served := range policy.ServedModels {
		if b.modelRef(served.Engine, served.Model) == ref {
			return true
		}
	}
	return false
}

// registerBindingFacadeLocked keeps only a live, evidenced offered listener
// reachable through the ordinary authenticated service allowlist.
func (b *Broker) registerBindingFacadeLocked(profile engineProxyProfile) bool {
	offers := b.bindingOffersLocked()
	if offers != nil {
		for _, ref := range offers.Models {
			if ref.Engine == profile.Name {
				port := b.engineProxyListenPort(profile)
				if port == 0 {
					return false
				}
				b.registerService(noderec.RegisterParams{Service: profile.DiscoveryService, Port: port})
				b.engineProxy(profile).bindingRegistered.Store(true)
				return true
			}
		}
	}
	return false
}

// A withdrawn offer must also withdraw its cold service allowlist entry.
// Healthy publication clears the marker at the existing backend owner.
func (b *Broker) withdrawBindingFacades(offers *nodepolicy.EngineBindingOffers) {
	for _, profile := range b.frontedEngineProfiles() {
		keep := false
		if offers != nil {
			for _, ref := range offers.Models {
				keep = keep || ref.Engine == profile.Name
			}
		}
		if runtime := b.engineProxy(profile); !keep && runtime != nil && runtime.bindingRegistered.Swap(false) {
			b.unregisterService(profile.DiscoveryService)
		}
	}
}

func (b *Broker) advertisementWithdrawnLocked(engine string) bool {
	if b.pendingSwitchLocked(engine) != nil {
		return true
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.drains[engine]
}

func (b *Broker) refreshBindingAdvertisement() {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	configured := len(r.current.EngineBindings) > 0 || r.bindingOffersPublished != nil
	r.mu.Unlock()
	if configured {
		b.refreshStrictAdvertisement()
	}
}
