// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"maps"
	"math"
	"slices"
	"strconv"
	"time"

	"nvpair-shared/nodepolicy"
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

// tierCostTerms is the expected-completion score: a sum of named, weighted
// terms, all in milliseconds. A later term is one row here plus one input in
// tierTerms.
var tierCostTerms = [...]struct {
	name   string
	weight float64
}{{"service", 1}, {"queue", 1}, {"pressure", 1}, {"cold", 1}}

const (
	// tierDefaultOutputTokens sizes the decode phase of a request without a
	// positive output limit.
	tierDefaultOutputTokens = 512
	// tierUnknownColdMs is the load cost of a cold candidate with no measured load.
	tierUnknownColdMs = 120000
	// tierProbeEvery is the automatic-choice cadence at which an unmeasured
	// idle candidate is tried instead of the cheapest one.
	tierProbeEvery = 16
)

const (
	tierDecisionPinned   = "pinned"
	tierDecisionBaseline = "baseline"
	tierDecisionMeasured = "measured"
	tierDecisionImputed  = "imputed"
	tierDecisionProbe    = "probe"
)

// tierClass is the reservation class a tier round chooses within.
type tierClass struct {
	rank int
	warm bool
}

// tierProbeState is one class's committed automatic choices and the last
// candidate it probed, as "node\x00engine\x00key".
type tierProbeState struct {
	choices uint64
	cursor  string
}

// tierRound carries what a tier round needs beyond its candidates. discovery
// holds every facade's routing generation, captured before the class was
// resolved, so any later routing-relevant discovery change rebuilds the class.
type tierRound struct {
	body      []byte
	route     string
	hybrid    bool
	discovery map[*facade]uint64
}

func (p *Proxy) discoveryGenerations() map[*facade]uint64 {
	facades := p.enabledFacades()
	out := make(map[*facade]uint64, len(facades))
	for _, f := range facades {
		out[f] = f.discovery.RoutingGeneration()
	}
	return out
}

// discoveryCurrent reads only atomics, so it may run under chooser locks.
func (r *tierRound) discoveryCurrent() bool {
	for f, generation := range r.discovery {
		if f.discovery.RoutingGeneration() != generation {
			return false
		}
	}
	return true
}

// tierMember is one class member's scalar inputs. No body, recipe or option
// bytes are retained.
type tierMember struct {
	key      string
	evidence performancewire.RoutingProfile
	measured bool
	ageMs    int64
	sized    bool
	input    int
	output   int
}

// tierFacts are a round's materialized inputs and the source generations they
// were read at. The chooser uses them only while every generation is current.
type tierFacts struct {
	members  []tierMember
	pin      string
	costs    int64
	sizing   uint64
	policy   uint64
	expires  time.Time
	sizedAll bool
	anyRate  bool
}

// tierScore is one scored candidate's terms, in tierCostTerms order.
type tierScore struct {
	NodeID         string
	Engine         string
	Key            string
	Terms          [len(tierCostTerms)]float64
	Total          float64
	Imputed        bool
	EvidenceAgeMs  int64
	PrefillSamples int
	DecodeSamples  int
}

// tierDecision is why a tier round chose its winner. Baseline and pinned
// decisions carry no numeric estimate.
type tierDecision struct {
	Decision  string
	Winner    *tierScore
	RunnerUp  *tierScore
	TierRank  int
	Warm      bool
	ClassSize int
}

func (p *Proxy) routingCostsGeneration() int64 {
	p.routingCostsMu.RLock()
	defer p.routingCostsMu.RUnlock()
	return p.routingCostsRevision
}

// tierFactsFor materializes a class's evidence and destination-effective
// sizing. It runs outside every chooser lock: it reads discovery, the mesh,
// both caches and policy, and transforms scratch bodies. A pin or a class
// with no measured rate skips sizing, because the unchanged chooser decides.
func (p *Proxy) tierFactsFor(ctx context.Context, f *facade, candidates []candidate, model string, round *tierRound) tierFacts {
	facts := tierFacts{costs: p.routingCostsGeneration(), sizing: p.peerSizing.generation.Load(), members: make([]tierMember, len(candidates))}
	f.selectedMu.RLock()
	facts.pin = f.selectedID
	f.selectedMu.RUnlock()
	for i, c := range candidates {
		m := &facts.members[i]
		m.key = c.key
		g := c.fac
		if g == nil {
			g = f
		}
		if q, ok := g.performanceQualification(round.route); ok && c.key != "" {
			m.evidence, m.ageMs, m.measured = p.routingEvidence(c.id, c.engine, c.key, q.EngineFamily, q.APIPath, q.SchemaID)
		}
		facts.anyRate = facts.anyRate || m.measured
		if facts.pin != "" && c.id == facts.pin {
			return facts
		}
	}
	if !facts.anyRate {
		return facts
	}
	p.admission.mu.Lock()
	facts.policy = p.admission.policyGeneration
	profiles := make([]nodepolicy.Profile, len(candidates))
	hasProfile := make([]bool, len(candidates))
	for i, c := range candidates {
		if c.self {
			engine := c.engineOr(f.profile.Name)
			concrete := model
			if c.model != "" {
				concrete = c.model
			}
			profiles[i], hasProfile[i] = p.admission.profileLocked(admKey{engine: engine, model: p.admission.normalize(engine, concrete)})
		}
	}
	p.admission.mu.Unlock()
	facts.sizedAll = true
	for i, c := range candidates {
		m := &facts.members[i]
		var needs requestNeeds
		var expires time.Time
		if c.self {
			// The profile is the one prepareCandidate would apply at this
			// generation; a self candidate without one sends no options.
			var options []byte
			mode := nodepolicy.RequestOptionsDefaults
			if hasProfile[i] {
				options, mode = profiles[i].RequestOptions, profiles[i].RequestOptionsMode.Effective()
			}
			needs, m.sized = tierEffectiveNeeds(ctx, round, c.model, options, mode, round.hybrid)
		} else {
			var recipe nodepolicy.PeerSizingProfile
			recipe, expires, m.sized = p.peerRecipe(c)
			if m.sized {
				needs, m.sized = tierEffectiveNeeds(ctx, round, c.model, recipe.RequestOptions, recipe.RequestOptionsMode.Effective(), true)
			}
		}
		if !m.sized {
			facts.sizedAll = false
			continue
		}
		m.input, m.output = needs.inputTokens, needs.outputTokens
		if !expires.IsZero() && (facts.expires.IsZero() || expires.Before(facts.expires)) {
			facts.expires = expires
		}
	}
	return facts
}

// peerRecipe reads the peer's current admission and advertisement and returns
// its destination recipe only while the private snapshot still matches them.
func (p *Proxy) peerRecipe(c candidate) (nodepolicy.PeerSizingProfile, time.Time, bool) {
	if p.mesh == nil || c.fac == nil || c.key == "" {
		return nodepolicy.PeerSizingProfile{}, time.Time{}, false
	}
	node, ok := discoveredNode(c.fac, c.id)
	if !ok || node.ClusterUUID == "" {
		return nodepolicy.PeerSizingProfile{}, time.Time{}, false
	}
	native := ""
	for _, profile := range node.Profiles {
		if profile.Key == c.key && profile.Engine == c.engine {
			native = profile.Model
			break
		}
	}
	identity, ok := p.mesh.PeerServiceIncarnation(node.ClusterUUID)
	if !ok || native == "" {
		return nodepolicy.PeerSizingProfile{}, time.Time{}, false
	}
	epoch, ok := p.mesh.PeerAdmissionEpoch(node.ClusterUUID)
	if !ok {
		return nodepolicy.PeerSizingProfile{}, time.Time{}, false
	}
	recipe, token, ok := p.peerSizing.recipe(peerSizingQuery{Host: c.id, AdmissionIdentity: identity, AdmissionIncarnation: strconv.FormatUint(epoch, 10),
		PolicyRevision: node.PolicyRevision, Key: c.key, Engine: c.engine, Model: native})
	return recipe, token.Expires, ok
}

// tierEffectiveNeeds sizes the body a destination would execute. Self
// non-hybrid uses prepareCandidate's combined rewrite and options; a peer, and
// a hybrid self, rewrite first and then apply the destination options as
// destination ingress does, and hybrid then removes the validated store
// no-op as strict qualification does. Scratch bodies are charged to a
// temporary account under the request's buffers and released, in reverse
// borrower order, before this returns.
func tierEffectiveNeeds(ctx context.Context, round *tierRound, model string, options []byte, mode nodepolicy.RequestOptionsMode, twoStage bool) (requestNeeds, bool) {
	if owner := inferenceBuffers(ctx); owner != nil {
		scope, err := owner.scope.category(math.MaxInt64)
		if err != nil {
			return requestNeeds{}, false
		}
		scratch := &ownedInferenceBuffers{scope: scope}
		defer scratch.close()
		ctx = context.WithValue(ctx, ownedInferenceContextKey{}, scratch)
	}
	var releases []func()
	defer func() {
		for i := len(releases) - 1; i >= 0; i-- {
			releases[i]()
		}
	}()
	body := round.body
	step := func(model string, defaults []byte, mode nodepolicy.RequestOptionsMode) bool {
		next, release, err := transformBody(ctx, body, model, model != "", defaults, mode)
		if err != nil {
			return false
		}
		releases = append(releases, release)
		body = next
		return true
	}
	if twoStage {
		if !step(model, nil, nodepolicy.RequestOptionsDefaults) || !step("", options, mode) {
			return requestNeeds{}, false
		}
	} else if !step(model, options, mode) {
		return requestNeeds{}, false
	}
	if round.hybrid {
		var err error
		if body, err = stripQualificationStore(ctx, body); err != nil {
			return requestNeeds{}, false
		}
	}
	needs, err := parseRequestNeedsOwned(ctx, body, round.route)
	return needs, err == nil
}

// tierFactsCurrentLocked reports whether facts still describe the sources. It
// runs under the chooser locks and reads only scalars; admissionHeld says the
// caller holds admission.mu, so the policy generation may be compared.
func (p *Proxy) tierFactsCurrentLocked(facts tierFacts, selectedID string, admissionHeld bool) bool {
	if facts.pin != selectedID || facts.costs != p.routingCostsGeneration() || facts.sizing != p.peerSizing.generation.Load() {
		return false
	}
	if admissionHeld && facts.anyRate && facts.policy != p.admission.policyGeneration {
		return false
	}
	return facts.expires.IsZero() || p.peerSizing.clock().Before(facts.expires)
}

// tierChoiceLocked chooses within one tier reservation class. It runs under
// priorityMu with facts already materialized; it transforms and reads nothing.
// A pin, or a class without a measured rate or with any unsized member, keeps
// candidateChoiceLocked's exact result.
func (p *Proxy) tierChoiceLocked(f *facade, candidates []candidate, facts tierFacts, selectedID string) (int, bool, tierDecision) {
	index, reserve := p.candidateChoiceLocked(candidates, selectedID)
	decision := tierDecision{Decision: tierDecisionBaseline, TierRank: candidates[0].tierRank, Warm: candidates[0].warm, ClassSize: len(candidates)}
	if selectedID != "" && candidates[index].id == selectedID {
		decision.Decision = tierDecisionPinned
		return index, reserve, decision
	}
	if !facts.anyRate || !facts.sizedAll || len(facts.members) != len(candidates) {
		return index, reserve, decision
	}
	scores := p.tierScoresLocked(candidates, facts)
	best, second := -1, -1
	for i := range scores {
		switch {
		case best < 0 || scores[i].Total < scores[best].Total:
			best, second = i, best
		case second < 0 || scores[i].Total < scores[second].Total:
			second = i
		}
	}
	index, decision.Decision = best, tierDecisionMeasured
	if scores[best].Imputed {
		decision.Decision = tierDecisionImputed
	}
	if probe, ok := p.tierProbeLocked(f, candidates, facts); ok {
		if probe != best {
			second = best
		}
		index, decision.Decision = probe, tierDecisionProbe
	}
	decision.Winner = &scores[index]
	if second >= 0 && second != index {
		decision.RunnerUp = &scores[second]
	}
	return index, slices.Contains(p.priority, candidates[index].id), decision
}

// tierScoresLocked imputes unknown phase rates from the class's slowest
// measured rate and scores every member.
func (p *Proxy) tierScoresLocked(candidates []candidate, facts tierFacts) []tierScore {
	slowest := func(phase func(performancewire.RoutingProfile) *performancewire.RoutingRate) float64 {
		rate := 0.0
		for _, m := range facts.members {
			if r := phase(m.evidence); m.measured && r != nil && (rate == 0 || r.TokensPerSecond < rate) {
				rate = r.TokensPerSecond
			}
		}
		return rate
	}
	prefillOf := func(e performancewire.RoutingProfile) *performancewire.RoutingRate { return e.Prefill }
	decodeOf := func(e performancewire.RoutingProfile) *performancewire.RoutingRate { return e.Decode }
	slowPrefill, slowDecode := slowest(prefillOf), slowest(decodeOf)
	slowAny := min(slowPrefill, slowDecode)
	if slowAny == 0 {
		slowAny = max(slowPrefill, slowDecode)
	}
	if slowPrefill == 0 {
		slowPrefill = slowAny
	}
	if slowDecode == 0 {
		slowDecode = slowAny
	}
	scores := make([]tierScore, len(candidates))
	for i, c := range candidates {
		m := facts.members[i]
		s := &scores[i]
		s.NodeID, s.Engine, s.Key = c.id, c.engine, m.key
		prefill, decode := slowPrefill, slowDecode
		if m.measured {
			s.EvidenceAgeMs = m.ageMs
		}
		if r := m.evidence.Prefill; m.measured && r != nil {
			prefill, s.PrefillSamples = r.TokensPerSecond, r.Samples
		} else {
			s.Imputed = true
		}
		if r := m.evidence.Decode; m.measured && r != nil {
			decode, s.DecodeSamples = r.TokensPerSecond, r.Samples
		} else {
			s.Imputed = true
		}
		s.Terms = tierTerms(m, c.warm, prefill, decode, p.priorityPending[c.id]+p.priorityReservations[c.id], p.priorityGPUPressure[c.id])
		for t, term := range tierCostTerms {
			s.Total += term.weight * s.Terms[t]
		}
	}
	return scores
}

// tierTerms computes the inputs of tierCostTerms, in its order.
func tierTerms(m tierMember, warm bool, prefill, decode float64, queued, pressure int) [len(tierCostTerms)]float64 {
	output := m.output
	if output == 0 {
		output = tierDefaultOutputTokens
	}
	service := 1000 * (float64(m.input)/prefill + float64(output)/decode)
	cold := 0.0
	if !warm {
		cold = tierUnknownColdMs
		if m.measured && m.evidence.LoadP95Ms != nil {
			cold = float64(*m.evidence.LoadP95Ms)
		}
	}
	return [len(tierCostTerms)]float64{service, float64(queued) * service, float64(pressure) * service, cold}
}

func tierProbeID(c candidate, key string) string {
	return c.id + "\x00" + c.engine + "\x00" + key
}

// tierProbeLocked chooses this turn's probe: on every tierProbeEvery-th
// automatic choice of the class, the next unmeasured idle candidate in
// (node, engine, profile key) order after the class's cursor.
func (p *Proxy) tierProbeLocked(f *facade, candidates []candidate, facts tierFacts) (int, bool) {
	state := f.tierProbes[tierClass{candidates[0].tierRank, candidates[0].warm}]
	if (state.choices+1)%tierProbeEvery != 0 {
		return 0, false
	}
	first, next := -1, -1
	for i, c := range candidates {
		if facts.members[i].measured || p.priorityPending[c.id] != 0 || p.priorityReservations[c.id] != 0 {
			continue
		}
		id := tierProbeID(c, facts.members[i].key)
		if first < 0 || id < tierProbeID(candidates[first], facts.members[first].key) {
			first = i
		}
		if id > state.cursor && (next < 0 || id < tierProbeID(candidates[next], facts.members[next].key)) {
			next = i
		}
	}
	if next < 0 {
		next = first
	}
	return next, next >= 0
}

// commitTierChoiceLocked advances the class's automatic-choice counter, and
// its probe cursor for a probe, at the reservation commit only.
func (f *facade) commitTierChoiceLocked(c candidate, decision tierDecision) {
	switch decision.Decision {
	case tierDecisionMeasured, tierDecisionImputed, tierDecisionProbe:
	default:
		return
	}
	if f.tierProbes == nil {
		f.tierProbes = map[tierClass]tierProbeState{}
	}
	class := tierClass{decision.TierRank, decision.Warm}
	state := f.tierProbes[class]
	state.choices++
	if decision.Decision == tierDecisionProbe {
		state.cursor = tierProbeID(c, decision.Winner.Key)
	}
	f.tierProbes[class] = state
}
