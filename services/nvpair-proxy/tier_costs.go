// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"maps"
	"time"

	"nvpair-shared/performancewire"
)

func cloneRoutingProfile(p performancewire.RoutingProfile) performancewire.RoutingProfile {
	if p.Prefill != nil {
		r := *p.Prefill
		p.Prefill = &r
	}
	if p.Decode != nil {
		r := *p.Decode
		p.Decode = &r
	}
	if p.LoadP95Ms != nil {
		ms := *p.LoadP95Ms
		p.LoadP95Ms = &ms
	}
	p.Extra = maps.Clone(p.Extra)
	return p
}

func (p *Proxy) setRoutingCosts(c performancewire.RoutingCosts) error {
	if err := c.Validate(); err != nil {
		return err
	}
	p.routingCostsMu.Lock()
	defer p.routingCostsMu.Unlock()
	if c.Revision < p.routingCostsRevision {
		return errors.New("routing costs revision is older than the current revision")
	}
	if c.Revision == p.routingCostsRevision {
		return nil
	}
	profiles := make(map[[6]string]performancewire.RoutingProfile, len(c.Profiles))
	for _, profile := range c.Profiles {
		key := [6]string{profile.Host, profile.Engine, profile.Model, profile.Family, profile.APIPath, profile.Schema}
		profiles[key] = cloneRoutingProfile(profile)
	}
	p.routingCosts, p.routingCostsRevision, p.routingCostsReceived = profiles, c.Revision, time.Now()
	return nil
}

// routingEvidence normalizes through the candidate engine's own facade, not the
// ingress engine's, so the key matches what that engine's sample producer
// records. The returned age advances with this process's monotonic receipt clock.
func (p *Proxy) routingEvidence(host, engine, profile, family, api, schema string) (performancewire.RoutingProfile, int64, bool) {
	model := p.normalizeEngineModel(engine, profile)
	if model == "" {
		return performancewire.RoutingProfile{}, 0, false
	}
	p.routingCostsMu.RLock()
	defer p.routingCostsMu.RUnlock()
	evidence, ok := p.routingCosts[[6]string{host, engine, model, family, api, schema}]
	if !ok {
		return performancewire.RoutingProfile{}, 0, false
	}
	elapsed := max(int64(0), time.Since(p.routingCostsReceived).Milliseconds())
	if evidence.AgeMs > performancewire.RetentionMs || elapsed > performancewire.RetentionMs-evidence.AgeMs {
		return performancewire.RoutingProfile{}, 0, false
	}
	return cloneRoutingProfile(evidence), evidence.AgeMs + elapsed, true
}
