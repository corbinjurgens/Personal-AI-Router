// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"maps"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/performancewire"
)

func proxyRoutingCostsFixture() performancewire.RoutingCosts {
	return performancewire.RoutingCosts{Revision: 2, Profiles: []performancewire.RoutingProfile{{Host: "destination", Engine: "llamacpp", Model: "profile/key", Fingerprint: "fingerprint",
		Family: "llamacpp", APIPath: "/v1/chat/completions", Schema: "llamacpp-b12020-0ec4027a", AgeMs: 100,
		Decode: &performancewire.RoutingRate{Samples: 1, TokensPerSecond: 79, Source: "engine"}}}}
}

func requireRoutingEvidence(t *testing.T, p *Proxy) (performancewire.RoutingProfile, int64) {
	t.Helper()
	e, age, ok := p.routingEvidence("destination", "llamacpp", "profile/key", "llamacpp", "/v1/chat/completions", "llamacpp-b12020-0ec4027a")
	if !ok {
		t.Fatal("current routing evidence missing")
	}
	return e, age
}

func TestRoutingCostsRevisionReplacementAndRejection(t *testing.T) {
	p, _ := policyProxy(t)
	c := proxyRoutingCostsFixture()
	if err := p.setRoutingCosts(c); err != nil {
		t.Fatal(err)
	}
	received := p.routingCostsReceived
	for _, revision := range []int64{1, 2} {
		t.Run(map[int64]string{1: "lower revision rejected", 2: "equal revision no-op"}[revision], func(t *testing.T) {
			next := proxyRoutingCostsFixture()
			next.Revision, next.Profiles[0].Decode.TokensPerSecond = revision, 999
			err := p.setRoutingCosts(next)
			if (revision == 1) != (err != nil) {
				t.Fatalf("revision %d error = %v", revision, err)
			}
			e, _ := requireRoutingEvidence(t, p)
			if e.Decode.TokensPerSecond != 79 || p.routingCostsRevision != 2 || p.routingCostsReceived != received {
				t.Fatalf("repeated revision mutated cache: %+v", e)
			}
		})
	}
	bad := proxyRoutingCostsFixture()
	bad.Revision, bad.Profiles[0].Decode.TokensPerSecond = 3, 0
	if err := p.setRoutingCosts(bad); err == nil {
		t.Fatal("invalid payload accepted")
	}
	e, _ := requireRoutingEvidence(t, p)
	if e.Decode.TokensPerSecond != 79 || p.routingCostsRevision != 2 || p.routingCostsReceived != received {
		t.Fatal("invalid payload changed cache")
	}
	if err := p.setRoutingCosts(performancewire.RoutingCosts{Revision: 3, Profiles: []performancewire.RoutingProfile{}}); err != nil {
		t.Fatal(err)
	}
	if len(p.routingCosts) != 0 || p.routingCostsRevision != 3 {
		t.Fatal("empty full replacement did not withdraw evidence")
	}
}

func TestRoutingCostsExpiresAtLookupWithMonotonicReceiptAge(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		p, _ := policyProxy(t)
		c := proxyRoutingCostsFixture()
		c.Profiles[0].AgeMs = performancewire.RetentionMs - 1000
		if err := p.setRoutingCosts(c); err != nil {
			t.Fatal(err)
		}
		time.Sleep(time.Second)
		_, age := requireRoutingEvidence(t, p)
		if age != performancewire.RetentionMs {
			t.Fatalf("advanced age = %d", age)
		}
		// Equal revision cannot reset the receipt clock or extend freshness.
		if err := p.setRoutingCosts(c); err != nil {
			t.Fatal(err)
		}
		time.Sleep(time.Millisecond)
		if _, _, ok := p.routingEvidence("destination", "llamacpp", "profile/key", "llamacpp", "/v1/chat/completions", "llamacpp-b12020-0ec4027a"); ok {
			t.Fatal("expired evidence survived lookup")
		}
	})
}

func TestRoutingCostsOwnsWholeCacheAndLookupValues(t *testing.T) {
	p, _ := policyProxy(t)
	c := proxyRoutingCostsFixture()
	ms := int64(23)
	c.Profiles[0].Prefill = &performancewire.RoutingRate{Samples: 1, TokensPerSecond: 378, Source: "count"}
	c.Profiles[0].LoadP95Ms, c.Profiles[0].LoadSamples = &ms, 1
	c.Profiles[0].Extra = map[string]float64{"future": 1}
	if err := p.setRoutingCosts(c); err != nil {
		t.Fatal(err)
	}
	c.Profiles[0].Decode.TokensPerSecond = 999
	c.Profiles[0].Prefill.TokensPerSecond = 999
	ms, c.Profiles[0].Extra["future"] = 999, 999
	e, _ := requireRoutingEvidence(t, p)
	if e.Decode.TokensPerSecond != 79 || e.Prefill.TokensPerSecond != 378 || *e.LoadP95Ms != 23 || e.Extra["future"] != 1 {
		t.Fatalf("input retained: %+v", e)
	}
	e.Decode.TokensPerSecond, e.Prefill.TokensPerSecond, *e.LoadP95Ms, e.Extra["future"] = 0, 0, 0, 0
	again, _ := requireRoutingEvidence(t, p)
	if again.Decode.TokensPerSecond != 79 || again.Prefill.TokensPerSecond != 378 || *again.LoadP95Ms != 23 || again.Extra["future"] != 1 {
		t.Fatal("lookup mutated cache")
	}
}

func TestRoutingCostsHandlerRejectsAddressedAndInvalidPayloads(t *testing.T) {
	p, w := policyProxy(t)
	c := proxyRoutingCostsFixture()
	response := rpc(t, p, w, performancewire.MethodRoutingCosts, c)
	if response.Error != nil {
		t.Fatalf("valid feed rejected: %+v", response.Error)
	}
	var result struct {
		Profiles int `json:"profiles"`
	}
	if err := json.Unmarshal(response.Result, &result); err != nil {
		t.Fatal(err)
	}
	if result.Profiles != 1 {
		t.Fatalf("stored profile count = %d", result.Profiles)
	}
	for _, method := range []string{engines.AddressMethod("llamacpp", performancewire.MethodRoutingCosts), performancewire.MethodRoutingCosts} {
		next := proxyRoutingCostsFixture()
		next.Revision = 3
		var params any = next
		if method == performancewire.MethodRoutingCosts {
			params = map[string]any{"revision": 3, "profiles": next.Profiles, "unknown": true}
		}
		response := rpc(t, p, w, method, params)
		if response.Error == nil || response.Error.Code != -32602 {
			t.Fatalf("invalid/addressed method accepted: %+v", response)
		}
		if p.routingCostsRevision != 2 {
			t.Fatal("rejected RPC changed revision")
		}
	}
}

func TestRoutingCostsIndependentOfPriorityAndReservations(t *testing.T) {
	p, _ := policyProxy(t)
	p.priority = []string{"destination"}
	p.priorityPending = map[string]int{"destination": 2}
	p.priorityGPUPressure = map[string]int{"destination": 3}
	p.priorityReservations = map[string]int{"destination": 4}
	p.appliedPriorityGeneration = 7
	if err := p.setRoutingCosts(proxyRoutingCostsFixture()); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p.priority, []string{"destination"}) || !reflect.DeepEqual(p.priorityPending, map[string]int{"destination": 2}) || !reflect.DeepEqual(p.priorityGPUPressure, map[string]int{"destination": 3}) || !reflect.DeepEqual(p.priorityReservations, map[string]int{"destination": 4}) || p.appliedPriorityGeneration != 7 {
		t.Fatal("routing feed changed scheduler or reservation state")
	}
}

func TestRoutingCostsCandidateKeyMatchesSampleProducerAcrossEngines(t *testing.T) {
	w := &recordingWriter{}
	p, ingress, destination := twoEngineProxy(t, w)
	p.noteSelf("destination", destination.profile.Name, []string{"Org/Profile:Q4"})
	p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{destination.profile.Name: performanceIdentity()})
	pol := p.admission.currentPolicy()
	pol.Profiles = []nodepolicy.Profile{{Key: "Org/Profile:Q4", DisplayName: "Candidate profile", Engine: destination.profile.Name, Model: "native-artifact.gguf"}}
	p.admission.setPolicy(pol)
	candidate := candidate{id: "destination", engine: destination.profile.Name, model: "Org/Profile:Q4"}
	key, q, _, ok := destination.performanceFingerprintOwned(candidate.model, "/v1/chat/completions", []byte(`{"model":"native-artifact.gguf","stream":true}`))
	if !ok {
		t.Fatal("destination sample producer refused fixture")
	}
	if key.Host != "destination" || key.Engine != destination.profile.Name || key.Model != "Org/Profile:Q4" || key.Model == pol.Profiles[0].Model || key.Engine == ingress.profile.Name {
		t.Fatalf("sample profile key = %+v", key)
	}
	c := proxyRoutingCostsFixture()
	c.Profiles[0].Host, c.Profiles[0].Engine, c.Profiles[0].Model, c.Profiles[0].Fingerprint = key.Host, key.Engine, key.Model, key.Fingerprint
	c.Profiles[0].Family, c.Profiles[0].APIPath, c.Profiles[0].Schema = q.EngineFamily, q.APIPath, q.SchemaID
	if err := p.setRoutingCosts(c); err != nil {
		t.Fatal(err)
	}
	e, _, ok := p.routingEvidence(candidate.id, candidate.engine, candidate.model, q.EngineFamily, q.APIPath, q.SchemaID)
	if !ok || e.Host != key.Host || e.Engine != key.Engine || e.Model != key.Model || e.Fingerprint != key.Fingerprint {
		t.Fatalf("candidate/sample key mismatch: %+v, %+v", e, key)
	}
	for name, tuple := range map[string][6]string{
		"host":            {"other", candidate.engine, candidate.model, q.EngineFamily, q.APIPath, q.SchemaID},
		"ingress engine":  {key.Host, ingress.profile.Name, candidate.model, q.EngineFamily, q.APIPath, q.SchemaID},
		"native model":    {key.Host, candidate.engine, pol.Profiles[0].Model, q.EngineFamily, q.APIPath, q.SchemaID},
		"case":            {key.Host, candidate.engine, "org/profile:Q4", q.EngineFamily, q.APIPath, q.SchemaID},
		"family":          {key.Host, candidate.engine, candidate.model, "strata", q.APIPath, q.SchemaID},
		"API":             {key.Host, candidate.engine, candidate.model, q.EngineFamily, "/v1/completions", q.SchemaID},
		"schema":          {key.Host, candidate.engine, candidate.model, q.EngineFamily, q.APIPath, "unknown"},
		"invalid profile": {key.Host, candidate.engine, "", q.EngineFamily, q.APIPath, q.SchemaID},
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, ok := p.routingEvidence(tuple[0], tuple[1], tuple[2], tuple[3], tuple[4], tuple[5]); ok {
				t.Fatal("incompatible key matched")
			}
		})
	}
	if !performancewire.RoutingCompatible("llamacpp", "/v1/chat/completions", performanceExtractLlamaSchema) || !performancewire.RoutingCompatible("strata", "/v1/chat/completions", performanceExtractStrataSchema) {
		t.Fatal("routing tuple registry differs from sample extractor schemas")
	}
}

// scoreFixture is one llama.cpp facade whose peers are real pinned mesh
// members advertising the "medium" profile, so destination sizing goes
// through the actual admission, advertisement and snapshot checks.
type scoreFixture struct {
	p        *Proxy
	f        *facade
	costs    int64
	sizing   uint64
	evidence []performancewire.RoutingProfile
}

const scoreRoute = "/v1/chat/completions"

func newScoreFixture(t *testing.T, ids ...string) *scoreFixture {
	t.Helper()
	tc := llamacppCase(t)
	disc := NewDiscovery()
	for _, id := range ids {
		n := nodeFor(t, id, "http://127.0.0.1:1")
		n.Models = []string{"gemma"}
		n.Profiles = []nodepolicy.AdvertisedProfile{{Key: "medium", Engine: tc.profile.Name, Model: "gemma"}}
		n.PolicyRevision = "1"
		disc.AddManual(n)
	}
	p := newTestProxy(tc.profile, NewCodec(&recordingWriter{}), disc, tc.profile.FacadePort)
	fx := &scoreFixture{p: p, f: p.soleFacade()}
	for _, id := range ids {
		pinRecordingPairPeer(t, fx.f, id, newRecordingEngine(t, http.StatusOK))
	}
	p.SetPriority(ids)
	return fx
}

// sizeAll gives every peer a fresh known no-op recipe, or the given options.
func (fx *scoreFixture) sizeAll(t *testing.T, options map[string]json.RawMessage, mode nodepolicy.RequestOptionsMode) {
	t.Helper()
	fx.sizing++
	snapshot := nodepolicy.PeerSizingSnapshot{Revision: fx.sizing, Peers: []nodepolicy.PeerSizing{}}
	nodes := fx.f.discovery.Nodes()
	slices.SortFunc(nodes, func(a, b Node) int { return strings.Compare(a.ID, b.ID) })
	for _, n := range nodes {
		identity, ok := fx.p.mesh.PeerServiceIncarnation(n.ClusterUUID)
		epoch, ok2 := fx.p.mesh.PeerAdmissionEpoch(n.ClusterUUID)
		if !ok || !ok2 {
			t.Fatalf("peer %s lacks admission", n.ID)
		}
		peer := nodepolicy.PeerSizing{Host: n.ID, AdmissionIdentity: identity, AdmissionIncarnation: strconv.FormatUint(epoch, 10), PolicyRevision: n.PolicyRevision, ProxyRunID: "run", Profiles: []nodepolicy.PeerSizingProfile{}}
		for _, profile := range n.Profiles {
			recipe := nodepolicy.PeerSizingProfile{Key: profile.Key, Engine: profile.Engine, Model: profile.Model}
			if o, ok := options[n.ID+"/"+profile.Key]; ok {
				recipe.RequestOptions, recipe.RequestOptionsMode = o, mode
			}
			peer.Profiles = append(peer.Profiles, recipe)
		}
		snapshot.Peers = append(snapshot.Peers, peer)
	}
	if err := fx.p.peerSizing.set(snapshot); err != nil {
		t.Fatal(err)
	}
}

func scoreRate(tps float64) *performancewire.RoutingRate {
	return &performancewire.RoutingRate{TokensPerSecond: tps, Samples: 4, Source: "engine"}
}

// measure feeds a routing-cost replacement: one entry per host for key
// "medium", with prefill/decode rates (0 = phase unmeasured).
func (fx *scoreFixture) measure(t *testing.T, rates map[string][2]float64, load map[string]int64) {
	t.Helper()
	fx.costs++
	c := performancewire.RoutingCosts{Revision: fx.costs, Profiles: []performancewire.RoutingProfile{}}
	for _, host := range slices.Sorted(maps.Keys(rates)) {
		r := rates[host]
		e := performancewire.RoutingProfile{Host: host, Engine: fx.f.profile.Name, Model: fx.p.normalizeEngineModel(fx.f.profile.Name, "medium"), Fingerprint: "fp-" + host,
			Family: "llamacpp", APIPath: scoreRoute, Schema: performanceExtractLlamaSchema, AgeMs: 10}
		if r[0] > 0 {
			e.Prefill = scoreRate(r[0])
		}
		if r[1] > 0 {
			e.Decode = scoreRate(r[1])
		}
		if ms, ok := load[host]; ok {
			e.LoadP95Ms, e.LoadSamples = &ms, 3
		}
		c.Profiles = append(c.Profiles, e)
	}
	if err := fx.p.setRoutingCosts(c); err != nil {
		t.Fatal(err)
	}
}

func (fx *scoreFixture) candidates(warm bool, self string, ids ...string) []candidate {
	out := make([]candidate, 0, len(ids))
	for _, id := range ids {
		out = append(out, candidate{id: id, fac: fx.f, engine: fx.f.profile.Name, model: "medium", key: "medium", warm: warm, self: id == self})
	}
	return out
}

// choose runs one prepared tier round over cands and returns the winner,
// its reservation and the decision.
func (fx *scoreFixture) choose(t *testing.T, body []byte, cands []candidate, prepare candidatePreparation) (string, reservation, tierDecision) {
	t.Helper()
	if prepare == nil {
		prepare = func(context.Context, candidate, nodepolicy.Profile, bool, uint64) error { return nil }
	}
	round := &tierRound{body: body, route: scoreRoute}
	out, held, decision, err := fx.f.reserveRoundPrepared(context.Background(), slices.Clone(cands), round, "medium", prepare)
	if err != nil {
		t.Fatal(err)
	}
	return out[0].id, held, decision
}

// scoreBody is a chat body exactly size bytes long with the given output limit.
func scoreBody(t *testing.T, size, maxTokens int) []byte {
	t.Helper()
	head := `{"model":"medium","max_tokens":` + strconv.Itoa(maxTokens) + `,"messages":[{"role":"user","content":"`
	tail := `"}]}`
	if maxTokens == 0 {
		head = `{"model":"medium","messages":[{"role":"user","content":"`
	}
	pad := size - len(head) - len(tail)
	if pad < 0 {
		t.Fatalf("body size %d too small", size)
	}
	return []byte(head + strings.Repeat("x", pad) + tail)
}

func (fx *scoreFixture) release(held reservation) {
	if held.held {
		fx.p.releaseReservation(held)
	}
}

func TestTierCostAcceptanceIdleFromEveryIngress(t *testing.T) {
	body := scoreBody(t, 8000, 512)
	for _, self := range []string{"", "msi", "joes"} {
		t.Run("ingress "+map[string]string{"": "third node", "msi": "msi", "joes": "joes"}[self], func(t *testing.T) {
			fx := newScoreFixture(t, "msi", "joes")
			fx.sizeAll(t, nil, "")
			fx.measure(t, map[string][2]float64{"msi": {100, 14}, "joes": {378, 79}}, nil)
			winner, held, decision := fx.choose(t, body, fx.candidates(true, self, "msi", "joes"), nil)
			if winner != "joes" || decision.Decision != tierDecisionMeasured || !held.held || held.nodeID != "joes" {
				t.Fatalf("winner %s, decision %+v, held %+v", winner, decision, held)
			}
			if got := decision.Winner.Terms[0]; math.Abs(got-11772.0) > 0.1 {
				t.Fatalf("joes service = %.1f, want 11772.0", got)
			}
			if got := decision.RunnerUp.Terms[0]; math.Abs(got-56571.4) > 0.1 || decision.RunnerUp.NodeID != "msi" {
				t.Fatalf("runner-up = %+v, want msi at 56571.4", decision.RunnerUp)
			}
		})
	}
}

func TestTierCostAcceptanceBusyQueueAndReservations(t *testing.T) {
	body := scoreBody(t, 8000, 512)
	for _, tc := range []struct {
		name        string
		work        int
		reservation bool
		winner      string
		joesTotal   float64
	}{
		{"pending 3 keeps joes", 3, false, "joes", 47088.1},
		{"pending 4 moves to msi", 4, false, "msi", 58860.1},
		{"reservations 3 keep joes", 3, true, "joes", 47088.1},
		{"reservations 4 move to msi", 4, true, "msi", 58860.1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fx := newScoreFixture(t, "msi", "joes")
			fx.sizeAll(t, nil, "")
			fx.measure(t, map[string][2]float64{"msi": {100, 14}, "joes": {378, 79}}, nil)
			fx.p.priorityMu.Lock()
			if tc.reservation {
				fx.p.priorityReservations = map[string]int{"joes": tc.work}
			} else {
				fx.p.priorityPending = map[string]int{"joes": tc.work}
			}
			fx.p.priorityMu.Unlock()
			winner, _, decision := fx.choose(t, body, fx.candidates(true, "", "msi", "joes"), nil)
			joes := decision.Winner
			if winner != "joes" {
				joes = decision.RunnerUp
			}
			if winner != tc.winner || math.Abs(joes.Total-tc.joesTotal) > 0.1 {
				t.Fatalf("winner %s, joes total %.1f, want %s and %.1f", winner, joes.Total, tc.winner, tc.joesTotal)
			}
		})
	}
}

func TestTierCostRequestSizeDecidesBetweenPrefillAndDecodeSpeed(t *testing.T) {
	for _, tc := range []struct {
		name           string
		size, output   int
		winner         string
		xTotal, yTotal float64
	}{
		{"long input favours fast prefill", 40000, 64, "x", 13200, 100800},
		{"long output favours fast decode", 400, 2000, "y", 100100, 26000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fx := newScoreFixture(t, "x", "y")
			fx.sizeAll(t, nil, "")
			fx.measure(t, map[string][2]float64{"x": {1000, 20}, "y": {100, 80}}, nil)
			winner, _, decision := fx.choose(t, scoreBody(t, tc.size, tc.output), fx.candidates(true, "", "x", "y"), nil)
			totals := map[string]float64{decision.Winner.NodeID: decision.Winner.Total, decision.RunnerUp.NodeID: decision.RunnerUp.Total}
			if winner != tc.winner || math.Abs(totals["x"]-tc.xTotal) > 0.1 || math.Abs(totals["y"]-tc.yTotal) > 0.1 {
				t.Fatalf("winner %s, totals %v", winner, totals)
			}
		})
	}
}

// baselineOf is what the unchanged chooser does with the same state.
func baselineOf(fx *scoreFixture, cands []candidate) (string, bool) {
	fx.f.selectedMu.RLock()
	defer fx.f.selectedMu.RUnlock()
	fx.p.priorityMu.Lock()
	defer fx.p.priorityMu.Unlock()
	index, reserve := fx.p.candidateChoiceLocked(cands, fx.f.selectedID)
	return cands[index].id, reserve
}

func TestTierCostBaselineReproducesCandidateChoice(t *testing.T) {
	body := scoreBody(t, 8000, 512)
	fast := map[string][2]float64{"a": {100, 10}, "b": {1000, 100}}
	cases := []struct {
		name     string
		setup    func(t *testing.T, fx *scoreFixture)
		decision string
	}{
		{"no evidence", func(t *testing.T, fx *scoreFixture) { fx.sizeAll(t, nil, "") }, tierDecisionBaseline},
		{"no sizing with fast evidence", func(t *testing.T, fx *scoreFixture) { fx.measure(t, fast, nil) }, tierDecisionBaseline},
		{"one unsized member", func(t *testing.T, fx *scoreFixture) {
			fx.sizeAll(t, nil, "")
			fx.measure(t, fast, nil)
			fx.sizing++
			snapshot := nodepolicy.PeerSizingSnapshot{Revision: fx.sizing, Peers: []nodepolicy.PeerSizing{}}
			if err := fx.p.peerSizing.set(snapshot); err != nil {
				t.Fatal(err)
			}
			fx.sizeAllExcept(t, "b")
		}, tierDecisionBaseline},
		{"expired sizing", func(t *testing.T, fx *scoreFixture) {
			clock := time.Now()
			fx.p.peerSizing.now = func() time.Time { return clock }
			fx.sizeAll(t, nil, "")
			fx.measure(t, fast, nil)
			clock = clock.Add(time.Duration(nodepolicy.PeerSizingFreshnessMs) * time.Millisecond)
		}, tierDecisionBaseline},
		{"pin", func(t *testing.T, fx *scoreFixture) {
			fx.sizeAll(t, nil, "")
			fx.measure(t, fast, nil)
			fx.f.selectedMu.Lock()
			fx.f.selectedID = "a"
			fx.f.selectedMu.Unlock()
		}, tierDecisionPinned},
		{"unlisted candidate", func(t *testing.T, fx *scoreFixture) {
			fx.sizeAll(t, nil, "")
			fx.p.SetPriority([]string{"z"})
		}, tierDecisionBaseline},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fx := newScoreFixture(t, "a", "b")
			fx.p.priorityMu.Lock()
			fx.p.priorityPending = map[string]int{"a": 0, "b": 1}
			fx.p.priorityMu.Unlock()
			tc.setup(t, fx)
			cands := fx.candidates(true, "", "a", "b")
			wantID, wantReserve := baselineOf(fx, cands)
			before := fx.f.tierProbes
			winner, held, decision := fx.choose(t, body, cands, nil)
			if winner != wantID || held.held != wantReserve || decision.Decision != tc.decision || decision.Winner != nil || decision.RunnerUp != nil {
				t.Fatalf("winner %s held %v decision %+v, want %s held %v %s", winner, held.held, decision, wantID, wantReserve, tc.decision)
			}
			if !reflect.DeepEqual(fx.f.tierProbes, before) {
				t.Fatal("a baseline or pinned choice advanced the probe counter")
			}
		})
	}
}

// sizeAllExcept publishes no-op recipes for every peer but one.
func (fx *scoreFixture) sizeAllExcept(t *testing.T, skip string) {
	t.Helper()
	fx.sizeAll(t, nil, "")
	fx.sizing++
	var peers []nodepolicy.PeerSizing
	fx.p.peerSizing.mu.RLock()
	for host, peer := range fx.p.peerSizing.peers {
		if host != skip {
			peers = append(peers, peer)
		}
	}
	fx.p.peerSizing.mu.RUnlock()
	slices.SortFunc(peers, func(a, b nodepolicy.PeerSizing) int { return strings.Compare(a.Host, b.Host) })
	if err := fx.p.peerSizing.set(nodepolicy.PeerSizingSnapshot{Revision: fx.sizing, Peers: peers}); err != nil {
		t.Fatal(err)
	}
}

func TestTierCostColdClassUsesMeasuredLoadAndClassesNeverMix(t *testing.T) {
	t.Run("measured load beats unknown cold start", func(t *testing.T) {
		fx := newScoreFixture(t, "unknown", "measured")
		fx.sizeAll(t, nil, "")
		fx.measure(t, map[string][2]float64{"measured": {200, 20}}, map[string]int64{"measured": 30000})
		winner, _, decision := fx.choose(t, scoreBody(t, 400, 64), fx.candidates(false, "", "unknown", "measured"), nil)
		if winner != "measured" || decision.Decision != tierDecisionMeasured || decision.Winner.Terms[3] != 30000 || decision.RunnerUp.Terms[3] != tierUnknownColdMs || !decision.RunnerUp.Imputed {
			t.Fatalf("winner %s decision %+v runner-up %+v", winner, decision, decision.RunnerUp)
		}
	})
	t.Run("warm slow beats cold fast because classes differ", func(t *testing.T) {
		fx := newScoreFixture(t, "cold", "warm")
		fx.sizeAll(t, nil, "")
		fx.measure(t, map[string][2]float64{"cold": {5000, 500}, "warm": {10, 1}}, map[string]int64{"cold": 1})
		cands := append(fx.candidates(true, "", "warm"), fx.candidates(false, "", "cold")...)
		winner, _, decision := fx.choose(t, scoreBody(t, 400, 64), cands, nil)
		if winner != "warm" || decision.ClassSize != 1 || !decision.Warm {
			t.Fatalf("winner %s decision %+v", winner, decision)
		}
	})
}

func TestTierCostImputesUnknownPhaseFromSlowestMeasured(t *testing.T) {
	fx := newScoreFixture(t, "a", "b", "c")
	fx.sizeAll(t, nil, "")
	// b measured only prefill; c nothing. Decode for b and c is a's 10; c's
	// prefill is the slowest measured prefill, b's 50.
	fx.measure(t, map[string][2]float64{"a": {100, 10}, "b": {50, 0}}, nil)
	_, _, decision := fx.choose(t, scoreBody(t, 4000, 100), fx.candidates(true, "", "a", "b", "c"), nil)
	if decision.Decision != tierDecisionMeasured || decision.Winner.NodeID != "a" || decision.Winner.Imputed {
		t.Fatalf("decision %+v", decision)
	}
	// a: 1000/100 + 100/10 = 20s; b: 1000/50 + 100/10 = 30s; c the same as b.
	if decision.RunnerUp.NodeID != "b" || !decision.RunnerUp.Imputed || math.Abs(decision.RunnerUp.Total-30000) > 0.1 || math.Abs(decision.Winner.Total-20000) > 0.1 {
		t.Fatalf("runner-up %+v winner %+v", decision.RunnerUp, decision.Winner)
	}
}

func TestTierCostProbeCadenceCountsOnlyCommits(t *testing.T) {
	body := scoreBody(t, 400, 64)
	fx := newScoreFixture(t, "measured", "unknown")
	fx.sizeAll(t, nil, "")
	fx.measure(t, map[string][2]float64{"measured": {100, 10}}, nil)
	cands := fx.candidates(true, "", "measured", "unknown")
	var probes []int
	for i := 1; i <= 2*tierProbeEvery; i++ {
		prepare := candidatePreparation(nil)
		if i == tierProbeEvery-1 {
			// The first preparation of this commit finds the sizing feed
			// replaced; that stale attempt must not count.
			stale := true
			prepare = func(context.Context, candidate, nodepolicy.Profile, bool, uint64) error {
				if stale {
					stale = false
					fx.sizeAll(t, nil, "")
				}
				return nil
			}
		}
		winner, held, decision := fx.choose(t, body, cands, prepare)
		fx.release(held)
		if decision.Decision == tierDecisionProbe {
			if winner != "unknown" {
				t.Fatalf("probe chose %s", winner)
			}
			probes = append(probes, i)
		}
	}
	if !reflect.DeepEqual(probes, []int{tierProbeEvery, 2 * tierProbeEvery}) {
		t.Fatalf("probes at %v, want every %dth commit", probes, tierProbeEvery)
	}

	t.Run("busy unknown is never probed", func(t *testing.T) {
		fx := newScoreFixture(t, "measured", "unknown")
		fx.sizeAll(t, nil, "")
		fx.measure(t, map[string][2]float64{"measured": {100, 10}}, nil)
		fx.p.priorityMu.Lock()
		fx.p.priorityPending = map[string]int{"unknown": 1}
		fx.p.priorityMu.Unlock()
		for i := 1; i <= tierProbeEvery; i++ {
			_, held, decision := fx.choose(t, body, fx.candidates(true, "", "measured", "unknown"), nil)
			fx.release(held)
			if decision.Decision == tierDecisionProbe {
				t.Fatalf("probed a busy candidate at commit %d", i)
			}
		}
	})
}

func TestTierCostStaleSourcesRebuildWithoutConsumingReservations(t *testing.T) {
	body := scoreBody(t, 8000, 512)
	for _, tc := range []struct {
		name   string
		change func(t *testing.T, fx *scoreFixture)
	}{
		{"speed feed replaced", func(t *testing.T, fx *scoreFixture) {
			fx.measure(t, map[string][2]float64{"msi": {1000, 140}, "joes": {378, 79}}, nil)
		}},
		{"sizing snapshot replaced", func(t *testing.T, fx *scoreFixture) { fx.sizeAll(t, nil, "") }},
		{"policy-only discovery change on the nonwinning candidate", func(t *testing.T, fx *scoreFixture) {
			n, _ := discoveredNode(fx.f, "msi")
			n.PolicyRevision = "2"
			fx.f.discovery.AddManual(n)
		}},
		{"sizing expires during preparation", func(t *testing.T, fx *scoreFixture) {
			now := time.Now().Add(time.Duration(nodepolicy.PeerSizingFreshnessMs) * time.Millisecond)
			fx.p.peerSizing.now = func() time.Time { return now }
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fx := newScoreFixture(t, "msi", "joes")
			fx.sizeAll(t, nil, "")
			fx.measure(t, map[string][2]float64{"msi": {100, 14}, "joes": {378, 79}}, nil)
			var prepared []string
			var reservedDuringPrep []int
			changed := false
			prepare := func(_ context.Context, c candidate, _ nodepolicy.Profile, _ bool, _ uint64) error {
				prepared = append(prepared, c.id)
				reservedDuringPrep = append(reservedDuringPrep, reservationCount(fx.p, "joes")+reservationCount(fx.p, "msi"))
				if !changed {
					changed = true
					tc.change(t, fx)
				}
				return nil
			}
			resolves := 0
			resolve := func() []candidate { resolves++; return fx.candidates(true, "", "msi", "joes") }
			round := &tierRound{body: body, route: scoreRoute}
			out, held, decision, err := fx.f.reserveRoundPrepared(context.Background(), nil, round, "medium", prepare, resolve)
			if err != nil {
				t.Fatal(err)
			}
			if len(prepared) < 2 || slices.Max(reservedDuringPrep) != 0 {
				t.Fatalf("prepared %v with reservations %v: the stale attempt was not rebuilt or consumed a reservation", prepared, reservedDuringPrep)
			}
			if reservationCount(fx.p, out[0].id) != 1 || !held.held || fx.f.tierProbes[tierClass{0, true}].choices > 1 {
				t.Fatalf("commit state: winner %s held %+v probes %+v decision %+v", out[0].id, held, fx.f.tierProbes, decision)
			}
			if strings.Contains(tc.name, "discovery") && resolves < 2 {
				t.Fatal("discovery change did not re-resolve the class")
			}
		})
	}
}

func TestTierCostEffectiveSizingFollowsDestinationOptions(t *testing.T) {
	client := `{"model":"medium","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function"}],"max_tokens":300}`
	for _, tc := range []struct {
		name    string
		options string
		mode    nodepolicy.RequestOptionsMode
		output  int
	}{
		{"enforced small limit replaces the client's", `{"max_tokens":64}`, nodepolicy.RequestOptionsEnforced, 64},
		{"enforced large limit replaces the client's", `{"max_tokens":2048}`, nodepolicy.RequestOptionsEnforced, 2048},
		{"defaults keep the client's limit", `{"max_tokens":64}`, nodepolicy.RequestOptionsDefaults, 300},
		{"null options are a known no-op", `null`, nodepolicy.RequestOptionsEnforced, 300},
		{"empty options are a known no-op", `{}`, nodepolicy.RequestOptionsDefaults, 300},
		{"numeric token kept verbatim", `{"max_tokens":1.0e3}`, nodepolicy.RequestOptionsEnforced, 300},
		{"escaped key is the same member", `{"max_tokens":128}`, nodepolicy.RequestOptionsEnforced, 128},
	} {
		t.Run(tc.name, func(t *testing.T) {
			round := &tierRound{body: []byte(client), route: scoreRoute}
			needs, ok := tierEffectiveNeeds(context.Background(), round, "medium", json.RawMessage(tc.options), tc.mode, true)
			// The destination receives the origin rewrite and applies its own
			// options exactly as ingress does.
			sent, release, err := transformBody(nil, []byte(client), "medium", true, nil, nodepolicy.RequestOptionsDefaults)
			if err != nil {
				t.Fatal(err)
			}
			defer release()
			effective, releaseEffective, err := transformBody(nil, sent, "", false, json.RawMessage(tc.options), tc.mode)
			if err != nil {
				t.Fatal(err)
			}
			defer releaseEffective()
			want := parseRequestNeeds(effective, scoreRoute)
			if !ok || needs.outputTokens != want.outputTokens || needs.inputTokens != len(effective)/4 {
				t.Fatalf("needs %+v ok %v, destination %+v (%d bytes)", needs, ok, want, len(effective))
			}
			if tc.output != 300 && needs.outputTokens != tc.output && tc.name != "numeric token kept verbatim" {
				t.Fatalf("output %d, want %d", needs.outputTokens, tc.output)
			}
		})
	}

	t.Run("nested num_predict and a key longer than the native name", func(t *testing.T) {
		body := []byte(`{"model":"medium","options":{"num_predict":77},"messages":[{"role":"user","content":"hi"}]}`)
		round := &tierRound{body: body, route: scoreRoute}
		needs, ok := tierEffectiveNeeds(context.Background(), round, "a-much-longer/profile-key", nil, nodepolicy.RequestOptionsDefaults, true)
		if !ok || needs.outputTokens != 77 || needs.inputTokens != (len(body)+len("a-much-longer/profile-key")-len("medium"))/4 {
			t.Fatalf("needs %+v ok %v", needs, ok)
		}
	})

	t.Run("sibling profiles of one model get distinct limits", func(t *testing.T) {
		fx := newScoreFixture(t, "node")
		n, _ := discoveredNode(fx.f, "node")
		n.Profiles = append(n.Profiles, nodepolicy.AdvertisedProfile{Key: "medium-long", Engine: fx.f.profile.Name, Model: "gemma"})
		fx.f.discovery.AddManual(n)
		fx.sizeAll(t, map[string]json.RawMessage{"node/medium": json.RawMessage(`{"max_tokens":64}`), "node/medium-long": json.RawMessage(`{"max_tokens":2048}`)}, nodepolicy.RequestOptionsEnforced)
		cands := []candidate{
			{id: "node", fac: fx.f, engine: fx.f.profile.Name, model: "medium", key: "medium", warm: true},
			{id: "node", fac: fx.f, engine: fx.f.profile.Name, model: "medium-long", key: "medium-long", warm: true},
		}
		fx.measure(t, map[string][2]float64{"node": {100, 10}}, nil)
		facts := fx.p.tierFactsFor(context.Background(), fx.f, cands, "medium", &tierRound{body: []byte(client), route: scoreRoute, discovery: fx.p.discoveryGenerations()})
		if !facts.sizedAll || facts.members[0].output != 64 || facts.members[1].output != 2048 {
			t.Fatalf("members %+v", facts.members)
		}
	})
}

func TestTierCostSelfSizingMatchesPreparedBody(t *testing.T) {
	fx := newScoreFixture(t, "peer")
	pol := fx.p.admission.currentPolicy()
	pol.Profiles = append(pol.Profiles, nodepolicy.Profile{Key: "medium", DisplayName: "Medium", Engine: fx.f.profile.Name, Model: "medium",
		RequestOptions: json.RawMessage(`{"max_tokens":96,"temperature":0.1}`), RequestOptionsMode: nodepolicy.RequestOptionsEnforced})
	fx.p.admission.setPolicy(pol)
	body := []byte(`{"model":"medium","messages":[{"role":"user","content":"hello"}],"max_tokens":300}`)
	fx.measure(t, map[string][2]float64{"self": {100, 10}}, nil)
	cands := []candidate{{id: "self", fac: fx.f, engine: fx.f.profile.Name, model: "medium", key: "medium", warm: true, self: true}}
	facts := fx.p.tierFactsFor(context.Background(), fx.f, cands, "medium", &tierRound{body: body, route: scoreRoute, discovery: fx.p.discoveryGenerations()})
	profile, ok := fx.p.admission.profileLocked(admKey{engine: fx.f.profile.Name, model: fx.p.admission.normalize(fx.f.profile.Name, "medium")})
	if !ok {
		t.Fatal("self profile missing")
	}
	prepared, release, err := transformBody(nil, body, "medium", true, profile.RequestOptions, profile.RequestOptionsMode.Effective())
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	if !facts.sizedAll || facts.members[0].output != 96 || facts.members[0].input != len(prepared)/4 {
		t.Fatalf("self sizing %+v, prepared body %d bytes", facts.members[0], len(prepared))
	}
}

func TestTierCostHybridSizingMatchesStrictQualifier(t *testing.T) {
	q, profile, _ := qualificationFixture(t)
	for _, tc := range []struct {
		name    string
		options string
		mode    nodepolicy.RequestOptionsMode
		body    string
		valid   bool
	}{
		{"input store false under defaults", ``, nodepolicy.RequestOptionsDefaults, `{"model":"fixture-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"store":false}`, true},
		{"profile store false under enforced", `{"store":false,"max_tokens":8}`, nodepolicy.RequestOptionsEnforced, `{"model":"fixture-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16}`, true},
		{"store true is refused", ``, nodepolicy.RequestOptionsDefaults, `{"model":"fixture-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"store":true}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			profile := profile
			profile.RequestOptions, profile.RequestOptionsMode = json.RawMessage(tc.options), tc.mode
			if tc.options == "" {
				profile.RequestOptions = nil
			}
			q.RequestOptions = profile.RequestOptions
			round := &tierRound{body: []byte(tc.body), route: routeChatCompletions, hybrid: true}
			needs, ok := tierEffectiveNeeds(context.Background(), round, "fixture-model", profile.RequestOptions, profile.RequestOptionsMode.Effective(), true)
			effective, _, err := q.validateRequest(profile, routeChatCompletions, []byte(tc.body))
			if !tc.valid {
				if ok || err == nil {
					t.Fatalf("sized %v / qualified %v an invalid strict body", ok, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			want := parseRequestNeeds(effective, routeChatCompletions)
			if !ok || needs.inputTokens != len(effective)/4 || needs.outputTokens != want.outputTokens || strings.Contains(string(effective), "store") {
				t.Fatalf("needs %+v, qualifier %s", needs, effective)
			}
		})
	}
}

func TestTierCostScratchBuffersReleasedOnFailureAndSuccess(t *testing.T) {
	p := NewProxy(NewCodec(&recordingWriter{}))
	r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, scoreRoute, nil))
	used := func() int64 {
		owner.scope.budget.mu.Lock()
		defer owner.scope.budget.mu.Unlock()
		return owner.scope.budget.used
	}
	before := used()
	for _, tc := range []struct {
		name string
		body string
		ok   bool
	}{
		{"valid", `{"model":"medium","messages":[],"max_tokens":4,"store":false}`, true},
		{"invalid strict store", `{"model":"medium","messages":[],"store":true}`, false},
		{"not JSON", `not json`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, ok := tierEffectiveNeeds(r.Context(), &tierRound{body: []byte(tc.body), route: scoreRoute, hybrid: true}, "medium", json.RawMessage(`{"temperature":0.2}`), nodepolicy.RequestOptionsDefaults, true)
			if ok != tc.ok {
				t.Fatalf("ok = %v", ok)
			}
			if now := used(); now != before {
				t.Fatalf("scratch storage retained: %d bytes, was %d", now, before)
			}
		})
	}
	owner.close()
}
