// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

// setAdvertisedProfilesByRole gives existing single-profile fixtures the same keyed
// profile inventory and role declarations that discovery now carries.
func setAdvertisedProfilesByRole(node *Node, profilesByRole map[string]nodepolicy.AdvertisedProfile) {
	node.Profiles = nil
	node.Roles = nil
	names := make([]string, 0, len(profilesByRole))
	for name := range profilesByRole {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		profile := profilesByRole[name]
		if profile.Key == "" {
			profile.Key = fixtureProfileKey(profile.Model)
		}
		profile.Model = profile.Key
		if profile.DisplayName == "" {
			profile.DisplayName = profile.Key
		}
		if !slices.ContainsFunc(node.Profiles, func(p nodepolicy.AdvertisedProfile) bool { return p.Key == profile.Key }) {
			node.Profiles = append(node.Profiles, profile)
		}
		node.Roles = append(node.Roles, nodepolicy.Role{Name: name, Profiles: []string{profile.Key}})
	}
}

// Public fixture keys must not occupy the role-override namespace.
func fixtureProfileKey(model string) string {
	if nodepolicy.IsReservedProfileKey(model) {
		return "fixture/" + model
	}
	return model
}

// declarePolicyRole preserves qualification facts when attaching a fixture's
// existing policy profile to a role, and creates a keyed profile if absent.
func declarePolicyRole(policy *nodepolicy.Policy, name, engine, model string) {
	key := fixtureProfileKey(model)
	index := slices.IndexFunc(policy.Profiles, func(profile nodepolicy.Profile) bool {
		return profile.Engine == engine && profile.Model == model
	})
	if index < 0 {
		policy.Profiles = append(policy.Profiles, nodepolicy.Profile{
			Key: key, DisplayName: "Role profile", Engine: engine, Model: key,
		})
	} else {
		profile := &policy.Profiles[index]
		profile.Key = key
		profile.Model = key
		if profile.DisplayName == "" {
			profile.DisplayName = "Role profile"
		}
	}
	policy.Roles = []nodepolicy.Role{{Name: name, Profiles: []string{key}}}
}

func TestRoleProfilesResolveOrderedMemberships(t *testing.T) {
	test := func(name string, keys []string, firstModel string, needs requestNeeds, want []tierCand) {
		t.Run(name, func(t *testing.T) {
			pol := nodepolicy.Default()
			pol.TierPolicy.AllowStronger = false
			fx := newTierFixture(t, llamacppCase(t), pol, map[string][]string{
				"peer": {"first-model", "second-model", "router-model"},
			}, map[string][]string{"peer": {"second-model"}})
			f := fx.p.soleFacade()
			n, ok := discoveredNode(f, "peer")
			if !ok {
				t.Fatal("missing fixture peer")
			}
			n.Roles = []nodepolicy.Role{{Name: "weak", Profiles: keys}}
			n.Profiles = []nodepolicy.AdvertisedProfile{
				{Key: "first-model", Engine: "llamacpp", Model: firstModel, ContextTokens: 100},
				{Key: "second-model", Engine: "llamacpp", Model: "second-model", ContextTokens: 10000, Capabilities: []nodepolicy.Capability{nodepolicy.CapTools}},
			}
			f.discovery.AddManual(n)
			if got := resolvedTier(f, "weak", needs, false); !slices.Equal(got, want) {
				t.Fatalf("role candidates = %+v, want %+v", got, want)
			}
		})
	}
	test("warm second leads cold first", []string{"first-model", "second-model"}, "first-model", requestNeeds{}, []tierCand{{"peer", "second-model", 0}, {"peer", "first-model", 0}})
	test("missing first permits second without router fallback", []string{"withheld-key", "second-model"}, "first-model", requestNeeds{}, []tierCand{{"peer", "second-model", 0}})
	test("malformed first permits second without router fallback", []string{"first-model", "second-model"}, "invalid\n", requestNeeds{}, []tierCand{{"peer", "second-model", 0}})
	test("all missing remains authoritative", []string{"withheld-key"}, "first-model", requestNeeds{}, nil)
	test("empty role remains authoritative", []string{}, "first-model", requestNeeds{}, nil)
	test("context uses each member bound", []string{"first-model", "second-model"}, "first-model", requestNeeds{contextTokens: 101}, []tierCand{{"peer", "second-model", 0}})
	test("capabilities use each member facts", []string{"first-model", "second-model"}, "first-model", requestNeeds{caps: []nodepolicy.Capability{nodepolicy.CapTools}}, []tierCand{{"peer", "second-model", 0}})
}

func TestRoleProfilesEmptyDeclarationPreventsOtherTierFallback(t *testing.T) {
	pol := nodepolicy.Default()
	fx := newTierFixture(t, llamacppCase(t), pol, map[string][]string{"peer": {"router-model"}}, nil)
	f := fx.p.soleFacade()
	n, ok := discoveredNode(f, "peer")
	if !ok {
		t.Fatal("missing fixture peer")
	}
	n.Roles = []nodepolicy.Role{{Name: "medium", Profiles: []string{}}}
	f.discovery.AddManual(n)
	if got := resolvedTier(f, "weak", requestNeeds{}, false); len(got) != 0 {
		t.Fatalf("empty declaration reactivated router tiers: %+v", got)
	}
	if got := f.declaredRoles(pol).peerTiers(); !got["medium"] {
		t.Fatalf("empty declaration lost its alias: %+v", got)
	}
}

func TestRoleProfilesMembershipOrderBeforeLexicalOrder(t *testing.T) {
	fx := newTierFixture(t, llamacppCase(t), nodepolicy.Default(), map[string][]string{
		"a": {"a-model", "z-model"}, "b": {"a-model", "z-model"},
	}, nil)
	f := fx.p.soleFacade()
	for _, id := range []string{"a", "b"} {
		n, ok := discoveredNode(f, id)
		if !ok {
			t.Fatalf("missing peer %s", id)
		}
		keys := []string{"z-model", "a-model"}
		if id == "b" {
			keys = []string{"a-model", "z-model"}
		}
		n.Roles = []nodepolicy.Role{{Name: "weak", Profiles: keys}}
		n.Profiles = []nodepolicy.AdvertisedProfile{
			{Key: "a-model", Engine: "llamacpp", Model: "a-model"},
			{Key: "z-model", Engine: "llamacpp", Model: "z-model"},
		}
		f.discovery.AddManual(n)
	}
	want := []tierCand{{"b", "a-model", 0}, {"a", "z-model", 0}, {"a", "a-model", 0}, {"b", "z-model", 0}}
	if got := resolvedTier(f, "weak", requestNeeds{}, false); !slices.Equal(got, want) {
		t.Fatalf("opposing memberships = %+v, want %+v", got, want)
	}
}

func TestRoleProfilesContextUsesCandidateNodeBound(t *testing.T) {
	fx := newTierFixture(t, llamacppCase(t), nodepolicy.Default(), map[string][]string{"small": {"model"}, "large": {"model"}}, nil)
	fx.declare(t, "small", map[string]nodepolicy.AdvertisedProfile{"weak": {Engine: "llamacpp", Model: "model", ContextTokens: 32768}})
	fx.declare(t, "large", map[string]nodepolicy.AdvertisedProfile{"weak": {Engine: "llamacpp", Model: "model", ContextTokens: 131072}})
	test := func(name string, context int, want []tierCand) {
		t.Run(name, func(t *testing.T) {
			if got := resolvedTier(fx.p.soleFacade(), "weak", requestNeeds{contextTokens: context}, false); !slices.Equal(got, want) {
				t.Fatalf("candidates = %+v, want %+v", got, want)
			}
		})
	}
	test("40000 excludes smaller node", 40000, []tierCand{{"large", "model", 0}})
	test("30000 admits both nodes", 30000, []tierCand{{"large", "model", 0}, {"small", "model", 0}})
}

func TestRoleProfilesEqualClassPreservesReverseEngineMembership(t *testing.T) {
	p, local, custom := twoEngineProxy(t, &recordingWriter{})
	for _, f := range []*facade{local, custom} {
		u := newRecordingEngine(t, http.StatusOK)
		model := "a-model"
		if f == custom {
			model = "z-model"
		}
		f.discovery.AddManual(selfNode("self", f.profile.FacadePort, model))
		setBackend(t, f, u.URL, true)
	}
	pol := nodepolicy.Default()
	pol.Profiles = []nodepolicy.Profile{
		{Key: "z-model", DisplayName: "Custom", Engine: "llamacpp-custom", Model: "z-model"},
		{Key: "a-model", DisplayName: "Local", Engine: "llamacpp", Model: "a-model"},
	}
	pol.Roles = []nodepolicy.Role{{Name: "weak", Profiles: []string{"z-model", "a-model"}}}
	p.admission.setPolicy(pol)
	got := local.resolveTierCandidates("weak", requestNeeds{}, true)
	if len(got) != 2 || got[0].engine != "llamacpp-custom" || got[0].model != "z-model" || got[1].engine != "llamacpp" || got[1].model != "a-model" {
		t.Fatalf("reverse engine/model membership order = %+v", got)
	}
}

func TestMalformedRoleDeclarationDoesNotActivateRouterTier(t *testing.T) {
	pol := nodepolicy.Default()
	pol.Roles = nil
	fx := newTierFixture(t, llamacppCase(t), pol, map[string][]string{"peer": {"router-model"}}, nil)
	f := fx.p.soleFacade()
	n, ok := discoveredNode(f, "peer")
	if !ok {
		t.Fatal("missing fixture peer")
	}
	n.Roles = []nodepolicy.Role{{Name: "", Profiles: []string{}}}
	f.discovery.AddManual(n)
	if f.roleRequested("weak") {
		t.Fatal("invalid role name declared a request role")
	}
	rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`)
	if rec.Code != http.StatusNotFound || fx.eng["peer"].hits() != 0 {
		t.Fatalf("malformed declaration activated router tier: status=%d, hits=%d", rec.Code, fx.eng["peer"].hits())
	}
}

func TestRoleProfilesHybridChecksMatchingProfileFacts(t *testing.T) {
	test := func(name string, first nodepolicy.AdvertisedProfile, strictModels []string) {
		t.Run(name, func(t *testing.T) {
			fx := newTierFixture(t, llamacppCase(t), nodepolicy.Default(), map[string][]string{"peer": {"first-model", "second-model"}}, nil)
			f := fx.p.soleFacade()
			n, ok := discoveredNode(f, "peer")
			if !ok {
				t.Fatal("missing fixture peer")
			}
			first.Key = "first-model"
			second := qualifiedProfile("second-model")
			second.Key = "second-model"
			n.Profiles = []nodepolicy.AdvertisedProfile{first, second}
			n.Roles = []nodepolicy.Role{{Name: "weak", Profiles: []string{"first-model", "second-model"}}}
			n.StrictAdmission = strictCapabilityFor("llamacpp", strictModels...)
			f.discovery.AddManual(n)
			want := []tierCand{{"peer", "first-model", 0}, {"peer", "second-model", 0}}
			if got := resolvedTier(f, "weak", requestNeeds{}, false); !slices.Equal(got, want) {
				t.Fatalf("concrete tier candidates = %+v, want %+v", got, want)
			}
			if got := f.hybridLocalCandidates("weak", requestNeeds{}, false); len(got) != 1 || got[0].model != "second-model" {
				t.Fatalf("hybrid candidates must use only second model's qualification and strict capacity: %+v", got)
			}
		})
	}
	test("later qualification cannot qualify first", llamacppProfile("first-model"), []string{"first-model", "second-model"})
	test("later strict capacity cannot admit first", qualifiedProfile("first-model"), []string{"second-model"})
}

func TestRoleProfilesSelfIncludesWithheldLocalProfile(t *testing.T) {
	p, f, _ := twoEngineProxy(t, &recordingWriter{})
	upstream := newRecordingEngine(t, http.StatusOK)
	setBackend(t, f, upstream.URL, true)
	f.discovery.AddManual(selfNode("self", f.profile.FacadePort, "private-model", "public-model"))
	pol := nodepolicy.Default()
	pol.Profiles = []nodepolicy.Profile{
		{Key: "private-model", DisplayName: "Private", Engine: "llamacpp", Model: "private-model", LocalOnly: true},
		{Key: "public-model", DisplayName: "Public", Engine: "llamacpp", Model: "public-model"},
	}
	pol.Roles = []nodepolicy.Role{{Name: "weak", Profiles: []string{"private-model", "public-model"}}}
	if err := pol.Validate(); err != nil {
		t.Fatalf("fixture policy: %v", err)
	}
	p.admission.setPolicy(pol)
	ad := pol.Advertisement()
	if len(ad.Profiles) != 1 || ad.Profiles[0].Key != "public-model" {
		t.Fatalf("local-only profile was not withheld: %+v", ad.Profiles)
	}
	peerEngine := newRecordingEngine(t, http.StatusOK)
	peer := nodeForModel(t, "peer", peerEngine.URL, "private-model")
	peer.Models = append(peer.Models, "public-model")
	peer.Roles, peer.Profiles = ad.Roles, ad.Profiles
	f.discovery.AddManual(peer)
	if got := f.resolveCandidates("private-model"); len(got) == 0 {
		t.Fatalf("fixture has no private-model owner: nodes=%+v self=%+v", f.discovery.Nodes(), f.declaredRoles(pol).self)
	}
	want := []tierCand{{"self", "private-model", 0}, {"peer", "public-model", 0}, {"self", "public-model", 0}}
	if got := resolvedTier(f, "weak", requestNeeds{}, false); !slices.Equal(got, want) {
		t.Fatalf("self and peer visibility = %+v, want %+v", got, want)
	}
}

func TestRoleProfilesFilteredAdvertisementUsesRemainingFirstProfile(t *testing.T) {
	pol := nodepolicy.Default()
	fx := newTierFixture(t, llamacppCase(t), pol, nil, nil)
	f := fx.p.soleFacade()
	upstream := newRecordingEngine(t, http.StatusOK)
	port := nodeFor(t, "peer", upstream.URL).Port
	raw := json.RawMessage(`{"nodes":[{"hostUuid":"peer","ip":"127.0.0.1","services":{"` + string(f.profile.DiscoveryService) + `":{"port":` + fmt.Sprint(port) + `}},"modelsByEngine":{"llamacpp":["second-model","router-model"]},"policy":{"revision":"1","profiles":[{"key":"first-model","displayName":"First","engine":"llamacpp","model":"bad\n"},{"key":"second-model","displayName":"Second","engine":"llamacpp","model":"second-model"}],"roles":[{"name":"weak","profiles":["first-model","second-model"]}]}}]}`)
	f.replaceSubscribed(raw)
	if !f.discovery.Has("peer") {
		t.Fatal("malformed profile discarded the peer inventory")
	}
	n, ok := discoveredNode(f, "peer")
	if !ok {
		t.Fatal("missing decoded fixture peer")
	}
	// This case checks profile filtering, using an explicit loopback manual target.
	f.discovery.AddManual(n)
	if got := f.resolveCandidates("second-model"); len(got) == 0 {
		t.Fatalf("fixture has no second-model owner: nodes=%+v", f.discovery.Nodes())
	}
	if got := resolvedTier(f, "weak", requestNeeds{}, false); len(got) != 1 || got[0].model != "second-model" {
		t.Fatalf("filtered declaration did not select its remaining first profile: %+v", got)
	}
	if !f.declaredRoles(pol).peerTiers()["weak"] {
		t.Fatal("malformed profile discarded role authority")
	}
}

func TestRoleProfilesSelfDetachesProfileFacts(t *testing.T) {
	fx := newTierFixture(t, llamacppCase(t), nodepolicy.Default(), nil, nil)
	pol := nodepolicy.Default()
	pol.Roles = []nodepolicy.Role{{Name: "weak", Profiles: []string{"key"}}}
	pol.Profiles = []nodepolicy.Profile{{Key: "key", DisplayName: "Profile", Engine: "llamacpp", Model: "model", LocalOnly: true, ContextTokens: 4096, MaxOutputTokens: 1024,
		Capabilities:  []nodepolicy.Capability{nodepolicy.CapTools},
		Qualification: &nodepolicy.QualificationRef{ID: "qualification", Revision: "1"},
		ModelIdentity: &engines.ModelIdentity{Strength: "declared", Files: []engines.ModelIdentityFile{{Role: "model", Name: "model.gguf"}}},
	}}
	if err := pol.Validate(); err != nil {
		t.Fatalf("fixture policy: %v", err)
	}
	declarations := fx.p.soleFacade().declaredRoles(pol)
	pol.Profiles[0].Capabilities[0] = nodepolicy.CapVision
	pol.Profiles[0].Qualification.ID = "other-qualification"
	pol.Profiles[0].ModelIdentity.Files[0].Name = "other.gguf"
	facts, ok := declarations.self["weak"]
	if !ok || len(facts) != 1 {
		t.Fatalf("self profile slice = %+v", facts)
	}
	got := facts[0]
	if got.Key != "key" || got.Model != "model" || !slices.Equal(got.Capabilities, []nodepolicy.Capability{nodepolicy.CapTools}) || got.Qualification == nil || got.Qualification.ID != "qualification" {
		t.Fatalf("self profile facts changed: %+v", got)
	}
	if got.ModelIdentity == nil || len(got.ModelIdentity.Files) != 1 || got.ModelIdentity.Files[0].Name != "model.gguf" {
		t.Fatalf("self model identity changed: %+v", got.ModelIdentity)
	}
}

func TestRoleProfilesEveryResolvedMemberDetachesFacts(t *testing.T) {
	roles := []nodepolicy.Role{{Name: "weak", Profiles: []string{"z-model", "a-model"}}}
	profiles := []nodepolicy.AdvertisedProfile{
		{Key: "a-model", Engine: "llamacpp", Model: "a-model", ContextTokens: 32768, Capabilities: []nodepolicy.Capability{nodepolicy.CapTools}, Qualification: &nodepolicy.QualificationRef{ID: "a", Revision: "1"}, ModelIdentity: &engines.ModelIdentity{Files: []engines.ModelIdentityFile{{Name: "a.gguf"}}}},
		{Key: "z-model", Engine: "llamacpp", Model: "z-model", ContextTokens: 131072, Capabilities: []nodepolicy.Capability{nodepolicy.CapTools}, Qualification: &nodepolicy.QualificationRef{ID: "z", Revision: "1"}, ModelIdentity: &engines.ModelIdentity{Files: []engines.ModelIdentityFile{{Name: "z.gguf"}}}},
	}
	got := resolveRoleProfiles(roles, profiles)["weak"]
	roles[0].Profiles[0] = "changed"
	for i := range profiles {
		profiles[i].Capabilities[0] = nodepolicy.CapVision
		profiles[i].Qualification.ID = "changed"
		profiles[i].ModelIdentity.Files[0].Name = "changed"
	}
	if len(got) != 2 {
		t.Fatalf("resolved facts = %+v", got)
	}
	for i, want := range []struct {
		key, model, qualification, file string
		context                         int
	}{
		{"z-model", "z-model", "z", "z.gguf", 131072}, {"a-model", "a-model", "a", "a.gguf", 32768},
	} {
		p := got[i]
		if p.Key != want.key || p.Model != want.model || p.ContextTokens != want.context || !slices.Equal(p.Capabilities, []nodepolicy.Capability{nodepolicy.CapTools}) || p.Qualification == nil || p.Qualification.ID != want.qualification || p.ModelIdentity == nil || len(p.ModelIdentity.Files) != 1 || p.ModelIdentity.Files[0].Name != want.file {
			t.Fatalf("member %d facts changed: %+v", i, p)
		}
	}
}

func TestRoleProfilesSubscriptionDetachesNestedFacts(t *testing.T) {
	profile := llamacppCase(t).profile
	ad := &nodepolicy.Advertisement{
		Revision: "1",
		Roles: []nodepolicy.Role{
			{Name: "strong", Profiles: []string{}},
			{Name: "weak", Profiles: []string{"key", "later-key"}},
		},
		Profiles: []nodepolicy.AdvertisedProfile{{Key: "key", Engine: "llamacpp", Model: "model",
			ModelIdentity: &engines.ModelIdentity{Strength: "declared", GeneralName: "Original", Files: []engines.ModelIdentityFile{{Role: "model", Name: "model.gguf"}}},
			Capabilities:  []nodepolicy.Capability{nodepolicy.CapTools},
			Qualification: &nodepolicy.QualificationRef{ID: "qualification", Revision: "1"}}},
	}
	node, ok := subscribedToNode(profile, noderec.DirectoryNode{
		HostUUID: "peer", IP: "127.0.0.1", Policy: ad,
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{profile.DiscoveryService: {Port: 30001}},
	})
	if !ok {
		t.Fatal("fixture subscription was rejected")
	}
	ad.Roles[0].Name = "medium"
	ad.Roles[1].Profiles[0] = "other-key"
	ad.Profiles[0].Model = "other-model"
	ad.Profiles[0].Capabilities[0] = nodepolicy.CapVision
	ad.Profiles[0].Qualification.ID = "other-qualification"
	ad.Profiles[0].ModelIdentity.GeneralName = "Other"
	ad.Profiles[0].ModelIdentity.Files[0].Name = "other.gguf"
	assert := func(t *testing.T, got Node) {
		t.Helper()
		if len(got.Roles) != 2 || got.Roles[0].Name != "strong" || len(got.Roles[0].Profiles) != 0 || got.Roles[1].Name != "weak" || !slices.Equal(got.Roles[1].Profiles, []string{"key", "later-key"}) || len(got.Profiles) != 1 {
			t.Fatalf("subscription declarations changed: %+v", got)
		}
		pr := got.Profiles[0]
		if pr.Model != "model" || !slices.Equal(pr.Capabilities, []nodepolicy.Capability{nodepolicy.CapTools}) || pr.Qualification == nil || pr.Qualification.ID != "qualification" {
			t.Fatalf("subscription profile facts changed: %+v", pr)
		}
		if pr.ModelIdentity == nil || pr.ModelIdentity.GeneralName != "Original" || len(pr.ModelIdentity.Files) != 1 || pr.ModelIdentity.Files[0].Name != "model.gguf" {
			t.Fatalf("subscription model identity changed: %+v", pr.ModelIdentity)
		}
	}
	assert(t, node)
	d := NewDiscovery()
	d.SetSubscribed([]Node{node})
	node.Roles[1].Profiles[0] = "mutated-after-subscribe"
	node.Profiles[0].Capabilities[0] = nodepolicy.CapVision
	node.Profiles[0].Qualification.ID = "mutated-after-subscribe"
	node.Profiles[0].ModelIdentity.Files[0].Name = "mutated-after-subscribe.gguf"
	nodes := d.Nodes()
	if len(nodes) != 1 {
		t.Fatalf("subscribed nodes = %+v, want one peer", nodes)
	}
	assert(t, nodes[0])
}
