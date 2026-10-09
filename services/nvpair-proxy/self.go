// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// What the proxy knows about its own node, and the dormant self candidate
// that start-on-demand needs.
//
// The broker advertises an engine's service only while the engine is healthy,
// so a stopped engine drops this node out of that engine's discovery overlay
// and resolveCandidates stops offering it. For a request to wake the engine
// (FORK_DESIGN.md §3.3 step 2) the router still needs a self candidate, so the
// proxy remembers its node id and, per engine, the models that engine last
// advertised, and offers a dormant self candidate built from that memory.

import (
	"net"
	"net/url"
	"slices"
	"strconv"

	"nvpair-shared/nodepolicy"
)

// noteSelf records this node's identity and an engine's inventory from a
// discovery snapshot node that resolves to this facade's own listener.
func (p *Proxy) noteSelf(nodeID, engine string, models []string) {
	p.selfMu.Lock()
	defer p.selfMu.Unlock()
	p.selfNodeID = nodeID
	if p.selfModels == nil {
		p.selfModels = map[string][]string{}
	}
	p.selfModels[engine] = append([]string(nil), models...)
}

func (p *Proxy) selfIdentity(engine string) (nodeID string, models []string) {
	p.selfMu.RLock()
	defer p.selfMu.RUnlock()
	return p.selfNodeID, p.selfModels[engine]
}

// noteSelfFromSnapshot finds this node in a fresh discovery overlay. A node is
// this one when one of its addresses resolves to this facade's own listener,
// the same test the self-forward guard applies.
func (f *facade) noteSelfFromSnapshot(nodes []Node) {
	selfPort, aliases := f.selfAddresses()
	for _, n := range nodes {
		for _, hp := range nodeCandidates(n) {
			u := &url.URL{Scheme: "http", Host: hp}
			if isSelfTarget(u, selfPort) || isAnyAliasSelfTarget(u, aliases) {
				f.host.noteSelf(n.ID, f.profile.Name, n.Models)
				break
			}
		}
	}
}

// engineOr returns the candidate's engine, or def for a candidate built
// without one (a test fixture, or a path that predates engine tagging).
func (c candidate) engineOr(def string) string {
	if c.engine != "" {
		return c.engine
	}
	return def
}

// nodeHasLoaded reports whether a node reports model resident for this engine.
func nodeHasLoaded(p engineProfile, n Node, model string) bool {
	want := p.normalizeModel(model)
	return want != "" && slices.ContainsFunc(n.Loaded, func(m string) bool {
		return p.normalizeModel(m) == want
	})
}

// dormantSelfCandidate offers this node's stopped engine for a model it can
// serve once woken: the engine must be saved On, start-on-demand enabled, the
// node available, and the model either last advertised by this engine or
// covered by one of this node's profiles.
func (f *facade) dormantSelfCandidate(model string) (candidate, bool) {
	p := f.host
	if p.localEngineHealthy(f.profile.Name) {
		return candidate{}, false
	}
	if p.admission.currentAvailability() != nodepolicy.Available {
		return candidate{}, false
	}
	pol := p.admission.currentPolicy()
	if on, known := p.admission.engineIntent(f.profile.Name); !known || !on || !pol.Idle.StartOnDemand {
		return candidate{}, false
	}
	selfID, models := p.selfIdentity(f.profile.Name)
	if selfID == "" {
		return candidate{}, false
	}
	want := f.profile.normalizeModel(model)
	served := slices.ContainsFunc(models, func(m string) bool { return f.profile.normalizeModel(m) == want })
	if !served {
		served = slices.ContainsFunc(pol.Profiles, func(pr nodepolicy.Profile) bool {
			return pr.Engine == f.profile.Name && f.profile.normalizeModel(pr.Model) == want
		})
	}
	if !served {
		return candidate{}, false
	}
	// The real address is read again once admission has woken the engine; the
	// last known port only stands in until then.
	port := f.currentLocalBackend().Port
	return candidate{
		id:      selfID,
		url:     &url.URL{Scheme: "http", Host: net.JoinHostPort("127.0.0.1", strconv.Itoa(port))},
		fac:     f,
		engine:  f.profile.Name,
		self:    true,
		dormant: true,
	}, true
}
