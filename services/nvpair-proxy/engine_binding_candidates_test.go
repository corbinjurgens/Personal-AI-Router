// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"reflect"
	"testing"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/schedulerwire"
)

func bindingOffer(engine, model string) *nodepolicy.EngineBindingOffers {
	return &nodepolicy.EngineBindingOffers{Version: 1, Models: []nodepolicy.Role{{Engine: engine, Model: model}}}
}

func TestEngineBindingSubscribedOfferIsNotInventory(t *testing.T) {
	profile := ollamaCase(t).profile
	n := noderec.DirectoryNode{HostUUID: "cold", IP: "127.0.0.2", Services: map[noderec.ServiceKey]noderec.ServiceStatus{
		profile.DiscoveryService: {Port: 45678}}, Policy: &nodepolicy.Advertisement{Revision: "1", EngineBindingOffers: bindingOffer("ollama", "fixture")}}
	node, ok := subscribedToNode(profile, n)
	if !ok || len(node.Models) != 0 || node.EngineBindingOffers == nil || node.Port != 45678 {
		t.Fatalf("stopped listener projection=%+v present=%v", node, ok)
	}
	if nodeAdvertisesModel(profile, node, "fixture") {
		t.Fatal("old inventory-only router sees an invented owner")
	}
	node.EngineBindingOffers.Models[0].Model = "changed"
	if n.Policy.EngineBindingOffers.Models[0].Model != "fixture" {
		t.Fatal("subscription retained mutable offer references")
	}
	delete(n.Services, profile.DiscoveryService)
	if _, ok := subscribedToNode(profile, n); ok {
		t.Fatal("offer invented an unadvertised listener")
	}
}

func TestEngineBindingCandidatesReadyBeatsSchedulerCold(t *testing.T) {
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "fixture"}}})
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"ready": {"fixture"}, "cold": {}}, nil)
	f := fx.p.soleFacade()
	n, ok := discoveredNode(f, "cold")
	if !ok {
		t.Fatal("missing cold fixture")
	}
	n.PolicyRevision = "1"
	n.EngineBindingOffers = bindingOffer("ollama", "fixture")
	f.discovery.AddManual(n)
	applySnapshot(fx.p, schedulerwire.Priority{Nodes: []string{"cold", "ready"}, Ranks: []schedulerwire.NodeRank{{ID: "cold"}, {ID: "ready", Pending: 100}}})
	test := func(name string, tier, prepared bool) {
		t.Run(name, func(t *testing.T) {
			cands := f.resolveCandidates("fixture")
			if tier {
				cands = f.resolveTierCandidates("weak", requestNeeds{}, false)
			}
			if len(cands) != 2 || cands[0].id != "ready" || !cands[1].preparation {
				t.Fatalf("candidate classes=%+v", cands)
			}
			var held reservation
			if prepared {
				var err error
				cands, held, err = f.reserveRoundPrepared(context.Background(), cands, tier, "fixture", nil)
				if err != nil {
					t.Fatal(err)
				}
			} else {
				cands, held = f.reserveRound(cands, tier)
			}
			defer fx.p.releaseReservation(held)
			if cands[0].id != "ready" {
				t.Fatalf("scheduler promoted cold offer: %+v", cands[0])
			}
		})
	}
	test("concrete reservation", false, false)
	test("concrete prepared reservation", false, true)
	test("tier reservation", true, false)
	test("tier prepared reservation", true, true)
	applySnapshot(fx.p, schedulerwire.Priority{Nodes: []string{"cold"}, Ranks: []schedulerwire.NodeRank{{ID: "cold"}}})
	test("unlisted ready owner beats scheduler offer", false, true)
	f.SetSelected("cold")
	cands, held := f.reserveRound(f.resolveCandidates("fixture"), false)
	defer fx.p.releaseReservation(held)
	if cands[0].id != "cold" || !cands[0].preparation {
		t.Fatal("eligible explicit cold selection was overridden")
	}
	tierCands, tierHeld := f.reserveRound(f.resolveTierCandidates("weak", requestNeeds{}, false), true)
	defer fx.p.releaseReservation(tierHeld)
	if len(tierCands) != 2 || tierCands[0].id != "cold" {
		t.Fatal("eligible explicit cold tier selection was overridden")
	}
}

func TestEngineBindingCandidatesPreserveLoadedPreferenceAndTierFallback(t *testing.T) {
	test := func(name, preferLoaded, want string) {
		t.Run(name, func(t *testing.T) {
			pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
				"weak":   {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "small"}},
				"medium": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "large"}},
			})
			pol.TierPolicy.PreferLoaded = preferLoaded
			pol.TierPolicy.AllowStronger = true
			fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"cold": {}, "ready": {"large"}}, map[string][]string{"ready": {"large"}})
			f := fx.p.soleFacade()
			n, ok := discoveredNode(f, "cold")
			if !ok {
				t.Fatal("missing cold fixture")
			}
			n.PolicyRevision = "1"
			n.EngineBindingOffers = bindingOffer("ollama", "small")
			f.discovery.AddManual(n)
			cands, held := f.reserveRound(f.resolveTierCandidates("weak", requestNeeds{}, false), true)
			defer fx.p.releaseReservation(held)
			if len(cands) != 2 || cands[0].id != want {
				t.Fatalf("loaded/tier preference=%+v want=%s", cands, want)
			}
		})
	}
	test("tier rank stays ahead of replacement class", "tier", "cold")
	test("any loaded model stays ahead of tier rank", "any", "ready")
}

func TestEngineBindingCandidatesAbsentOffersPreserveEntryOrder(t *testing.T) {
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
		"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "first"}, {Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "second"}},
	})
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"a": {"first"}, "b": {"second"}}, nil)
	f := fx.p.soleFacade()
	f.SetSelected("b")
	cands := f.resolveTierCandidates("weak", requestNeeds{}, false)
	if len(cands) != 2 || cands[0].id != "a" || cands[1].id != "b" {
		t.Fatalf("absent bindings changed legacy entry order: %+v", cands)
	}
}

func TestEngineBindingCandidatesRejectConflictingInventory(t *testing.T) {
	p := prProxy(t)
	f := p.soleFacade()
	n := Node{EngineBindingOffers: bindingOffer("lmstudio", "fixture"), Models: []string{"fixture"}}
	if f.bindingConsistent(n.EngineBindingOffers, "fixture") {
		t.Fatal("old-engine inventory overrode bound engine")
	}
	if !f.bindingConsistent(nil, "fixture") || !f.nodeOffersBinding(Node{PolicyRevision: "1", EngineBindingOffers: bindingOffer(f.profile.Name, "fixture")}, "fixture") {
		t.Fatal("absent offers changed ordinary routing or valid native offer lost")
	}
	if f.nodeOffersBinding(Node{PolicyRevision: "1", EngineBindingOffers: &nodepolicy.EngineBindingOffers{Version: 2, Models: []nodepolicy.Role{{Engine: f.profile.Name, Model: "fixture"}}}}, "fixture") {
		t.Fatal("unsupported offers became eligible")
	}
	if !reflect.DeepEqual(n.Models, []string{"fixture"}) {
		t.Fatal("binding eligibility rewrote actual inventory")
	}
}

func TestEngineBindingCandidatesSelfOffUsesCurrentOffer(t *testing.T) {
	test := func(name string, models []string) {
		t.Run(name, func(t *testing.T) {
			p := prProxy(t)
			f := p.soleFacade()
			pol := nodepolicy.Default()
			pol.Revision = "1"
			pol.EngineBindings = nodepolicy.EngineBindings{{Engine: f.profile.Name, Model: "fixture"}}
			p.admission.setPolicy(pol)
			p.admission.setIntent(map[string]bool{f.profile.Name: false})
			n := selfNode("self", 11435, models...)
			n.PolicyRevision = "1"
			n.EngineBindingOffers = bindingOffer(f.profile.Name, "fixture")
			f.discovery.AddManual(n)
			cands := f.resolveCandidates("fixture")
			if len(cands) != 1 || !cands[0].self || !cands[0].preparation || !cands[0].dormant {
				t.Fatalf("bound Off self is unreachable: %+v", cands)
			}
			pol.Revision = "2"
			pol.EngineBindings = nil
			p.admission.setPolicy(pol)
			if got := f.resolveCandidates("fixture"); len(got) != 0 {
				t.Fatalf("stale self offer overrode current policy: %+v", got)
			}
		})
	}
	test("empty actual inventory", nil)
	test("stale positive inventory", []string{"fixture"})
}

func TestEngineBindingCandidatesStrictExcludesOffersWithStaleIdentity(t *testing.T) {
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "q:1b"}}})
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"cold": {}}, nil)
	fx.advertise(t, "cold", map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("q:1b")}, strictCapabilityFor("ollama", "q:1b"))
	f := fx.p.soleFacade()
	n, ok := discoveredNode(f, "cold")
	if !ok {
		t.Fatal("missing cold fixture")
	}
	n.EngineBindingOffers = bindingOffer("ollama", "q:1b")
	f.discovery.AddManual(n)
	if got := f.resolveTierCandidates("weak", requestNeeds{}, false); len(got) != 1 || !got[0].preparation {
		t.Fatalf("concrete-tier offer missing: %+v", got)
	}
	if got := f.hybridLocalCandidates("weak", requestNeeds{}, false); len(got) != 0 {
		t.Fatalf("cold offer became strict capacity: %+v", got)
	}
}
