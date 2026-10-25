// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func TestCapacityDropClearsSelectedNode(t *testing.T) {
	profile := anyProfile(t)
	recorder := &recordingWriter{}
	p := newTestProxy(profile, NewCodec(recorder), NewDiscovery(), 11435)
	f := p.soleFacade()
	var nodes []noderec.DirectoryNode
	for i := range maxInventoryNodes {
		nodes = append(nodes, noderec.DirectoryNode{
			HostUUID: fmt.Sprintf("node-%03d", i), IP: "192.0.2.1",
			Services: map[noderec.ServiceKey]noderec.ServiceStatus{profile.DiscoveryService: {Port: 11434}},
		})
	}
	replace := func(nodes []noderec.DirectoryNode) {
		t.Helper()
		raw, err := json.Marshal(noderec.GetNodesResult{Nodes: nodes})
		if err != nil {
			t.Fatal(err)
		}
		f.replaceSubscribed(raw)
	}
	replace(nodes)
	selected := nodes[len(nodes)-1].HostUUID
	f.SetSelected(selected)
	extra := nodes[0]
	extra.HostUUID = "earlier"
	replace(append([]noderec.DirectoryNode{extra}, nodes...))
	if f.SelectedID() != "" || f.discovery.Has(selected) {
		t.Fatal("capacity-dropped node retained its route selection")
	}
	removed, cleared := false, false
	for _, line := range recorder.lines() {
		var msg Message
		if err := json.Unmarshal(line, &msg); err != nil {
			t.Fatal(err)
		}
		switch msg.Method {
		case engines.AddressMethod(profile.Name, "node/removed"):
			var node Node
			if err := json.Unmarshal(msg.Params, &node); err != nil {
				t.Fatal(err)
			}
			removed = removed || node.ID == selected
		case engines.AddressMethod(profile.Name, "node/selection-changed"):
			var result SelectedResult
			if err := json.Unmarshal(msg.Params, &result); err != nil {
				t.Fatal(err)
			}
			cleared = cleared || result.ID == ""
		}
	}
	if !removed || !cleared {
		t.Fatalf("removal notification=%v selection clear=%v", removed, cleared)
	}
	replace(nodes)
	if f.SelectedID() != "" || !f.discovery.Has(selected) {
		t.Fatal("readmission restored a stale selection")
	}
}

func TestUnknownTierDeclarationPreventsRouterFallback(t *testing.T) {
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
		"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "m"}},
	})
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"peer": {"m"}}, nil)
	fx.declare(t, "peer", map[string]nodepolicy.AdvertisedRole{"future": ollamaRole("m")})
	if got := resolvedTier(fx.p.soleFacade(), "weak", requestNeeds{}, false); len(got) != 0 {
		t.Fatalf("unknown tier declaration enabled router fallback: %+v", got)
	}
}

func TestDiscoveryInventoryNodeLimit(t *testing.T) {
	d := NewDiscovery()
	d.AddManual(Node{ID: "manual"})
	nodes := []Node{{ID: "manual"}}
	for i := range maxInventoryNodes + 10 {
		nodes = append(nodes, Node{ID: fmt.Sprintf("node-%03d", i)})
	}
	discovered, _, _ := d.SetSubscribed(nodes)
	if len(d.Nodes()) != maxInventoryNodes || len(discovered) != maxInventoryNodes {
		t.Fatalf("merged=%d discovered=%d, want %d", len(d.Nodes()), len(discovered), maxInventoryNodes)
	}
	if !d.Has("node-254") || d.Has("node-255") || d.AddManual(Node{ID: "overflow"}) {
		t.Fatal("did not retain the first admitted IDs or reject excess manual nodes")
	}
	if d.AddManual(Node{ID: "manual", Host: "updated"}) {
		t.Fatal("an existing manual entry was reported new")
	}
	added, updated, removed := d.SetSubscribed(nodes)
	if len(added)+len(updated)+len(removed) != 0 {
		t.Fatal("identical over-bound snapshot changed admission")
	}
	d.SetSubscribed(nil)
	if len(d.Nodes()) != 1 || !d.AddManual(Node{ID: "replacement"}) {
		t.Fatal("removing subscribed inventory did not release capacity")
	}
}

func TestDiscoveryDuplicateIDsDoNotGrowDiff(t *testing.T) {
	d := NewDiscovery()
	nodes := make([]Node, maxInventoryNodes*2)
	for i := range nodes {
		nodes[i] = Node{ID: "same"}
	}
	discovered, _, _ := d.SetSubscribed(nodes)
	if len(discovered) != 1 || len(d.Nodes()) != 1 {
		t.Fatal("duplicate IDs grew the admitted set or its diff")
	}
}

func TestDiscoveryInventoryModelLimit(t *testing.T) {
	models := make([]string, maxInventoryModels+10)
	served := make(nodepolicy.ServedModels, len(models))
	for i := range models {
		models[i] = fmt.Sprintf("model-%04d", i)
		served[i] = nodepolicy.Role{Engine: "ollama", Model: models[i]}
	}
	for _, source := range []string{"subscribed", "manual"} {
		t.Run(source, func(t *testing.T) {
			d := NewDiscovery()
			node := Node{ID: "node", Models: models, Loaded: models, ServedModels: served}
			if source == "manual" {
				d.AddManual(node)
			} else {
				d.SetSubscribed([]Node{node})
			}
			got := d.Nodes()[0]
			if len(got.Models) != maxInventoryModels || len(got.Loaded) != maxInventoryModels || len(got.ServedModels) != maxInventoryModels {
				t.Fatalf("models=%d loaded=%d served=%d", len(got.Models), len(got.Loaded), len(got.ServedModels))
			}
			if !reflect.DeepEqual(got.Models, models[:maxInventoryModels]) {
				t.Fatal("admission changed retained model order")
			}
			if cap(got.Models) > maxInventoryModels || cap(got.Loaded) > maxInventoryModels || cap(got.ServedModels) > maxInventoryModels {
				t.Fatal("retained over-bound backing storage")
			}
		})
	}
}

func TestSubscribedJSONModelLimit(t *testing.T) {
	profile := anyProfile(t)
	p := testProxy(profile, NewDiscovery(), 11435)
	models := make([]string, maxInventoryModels+1)
	for i := range models {
		models[i] = fmt.Sprintf("model-%04d", i)
	}
	raw, err := json.Marshal(noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{{
		HostUUID: "peer", IP: "192.0.2.1", Models: models,
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{profile.DiscoveryService: {Port: 11434}},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	p.handleMessage(&Message{JSONRPC: "2.0", Method: engines.AddressMethod(profile.Name, noderec.NotifyNodes), Params: raw})
	nodes := p.soleFacade().discovery.Nodes()
	if len(nodes) != 1 || !reflect.DeepEqual(nodes[0].Models, models[:maxInventoryModels]) {
		t.Fatal("JSON discovery intake did not bound the ordinary model inventory")
	}
}

func TestInventoryWarningRateLimit(t *testing.T) {
	previous := slog.Default()
	var output bytes.Buffer
	slog.SetDefault(slog.New(slog.NewTextHandler(&output, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })
	inventoryWarning.Lock()
	inventoryWarning.last = time.Time{}
	inventoryWarning.Unlock()
	warnInventoryLimit()
	warnInventoryLimit()
	if got := strings.Count(output.String(), "level=WARN"); got != 1 {
		t.Fatalf("warnings=%d, want 1", got)
	}
	inventoryWarning.Lock()
	inventoryWarning.last = time.Now().Add(-time.Minute)
	inventoryWarning.Unlock()
	warnInventoryLimit()
	if got := strings.Count(output.String(), "level=WARN"); got != 2 {
		t.Fatalf("warnings after interval=%d, want 2", got)
	}
}

// TestSubscribedOverlayMerge covers the relay-sourced routing overlay: the broker
// pushes the full filtered set, SetSubscribed replaces the overlay wholesale, and
// subscribed nodes merge into Nodes() alongside the separate manual overlay. A
// later snapshot omitting a node drops it while manual nodes survive. The browser
// isn't running, so Nodes() reflects only the overlays.
func TestSubscribedOverlayMerge(t *testing.T) {
	d := NewDiscovery()

	d.AddManual(Node{ID: "m1", Port: 11434, Addresses: []string{"10.0.0.1"}})

	d.SetSubscribed([]Node{{ID: "s1", Port: 11434, Addresses: []string{"10.0.0.2"}, IP: "10.0.0.2"}})

	nodes := d.Nodes()
	ids := map[string]bool{}
	for _, n := range nodes {
		ids[n.ID] = true
	}
	if !ids["m1"] || !ids["s1"] {
		t.Fatalf("Nodes() should include both manual and subscribed entries, got %v", ids)
	}

	// A subsequent snapshot that omits s1 drops it (wholesale replace); the manual
	// overlay is untouched.
	d.SetSubscribed(nil)
	for _, n := range d.Nodes() {
		if n.ID == "s1" {
			t.Fatal("s1 should be gone from Nodes() after a snapshot omitting it")
		}
	}
	if !d.IsManual("m1") {
		t.Fatal("manual node m1 should survive a subscribed-set replace")
	}
}

// TestSubscribedDiff covers the diff SetSubscribed returns so the proxy can emit
// node/discovered|updated|removed for the relay-fed set (the signal the UI uses to
// show which peers run this engine).
func TestSubscribedDiff(t *testing.T) {
	d := NewDiscovery()

	s1 := Node{ID: "s1", Port: 11434, Addresses: []string{"10.0.0.2"}, IP: "10.0.0.2"}
	disc, upd, rem := d.SetSubscribed([]Node{s1})
	if len(disc) != 1 || disc[0].ID != "s1" || len(upd) != 0 || len(rem) != 0 {
		t.Fatalf("first snapshot: want discovered=[s1], got disc=%v upd=%v rem=%v", disc, upd, rem)
	}

	// Same set again: no events.
	disc, upd, rem = d.SetSubscribed([]Node{s1})
	if len(disc)+len(upd)+len(rem) != 0 {
		t.Fatalf("unchanged snapshot should emit nothing, got disc=%v upd=%v rem=%v", disc, upd, rem)
	}

	// Model inventory is routing data; changing it must update the subscribed
	// overlay even when the endpoint is unchanged.
	s1Models := s1
	s1Models.Models = []string{"llama"}
	disc, upd, rem = d.SetSubscribed([]Node{s1Models})
	if len(disc) != 0 || len(upd) != 1 || upd[0].ID != "s1" || len(rem) != 0 {
		t.Fatalf("changed models: want updated=[s1], got disc=%v upd=%v rem=%v", disc, upd, rem)
	}

	// Changed IP: an update.
	s1b := s1Models
	s1b.IP = "10.0.0.9"
	s1b.Addresses = []string{"10.0.0.9"}
	disc, upd, rem = d.SetSubscribed([]Node{s1b})
	if len(disc) != 0 || len(upd) != 1 || upd[0].ID != "s1" || len(rem) != 0 {
		t.Fatalf("changed node: want updated=[s1], got disc=%v upd=%v rem=%v", disc, upd, rem)
	}

	// Dropped from the snapshot: a removal.
	disc, upd, rem = d.SetSubscribed(nil)
	if len(disc) != 0 || len(upd) != 0 || len(rem) != 1 || rem[0].ID != "s1" {
		t.Fatalf("omitted node: want removed=[s1], got disc=%v upd=%v rem=%v", disc, upd, rem)
	}
}

// TestSubscribedToNode covers the DirectoryNode -> routable Node projection:
// this engine's port is read from its own service key, and a node lacking that
// key, or an IP, is rejected.
//
// The two negative fixtures matter for different reasons and the old suites
// carried one each — Ollama rejected a non-engine service, LM Studio rejected
// the other engine. Both run for both engines here, because the second is what
// regresses if the projection ever loses its service-key filter.
func TestSubscribedToNode(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		other := otherEngine(t, tc)

		withService := noderec.DirectoryNode{
			HostUUID: "uuid-a",
			Name:     "host-a",
			IP:       "10.0.0.5",
			Models:   []string{"llama"},
			Services: map[noderec.ServiceKey]noderec.ServiceStatus{
				tc.profile.DiscoveryService: {Port: tc.profile.FacadePort},
			},
		}
		got, ok := subscribedToNode(tc.profile, withService)
		if !ok {
			t.Fatal("node advertising this engine + IP should project")
		}
		if got.ID != "uuid-a" || got.Port != tc.profile.FacadePort || got.IP != "10.0.0.5" ||
			len(got.Models) != 1 || got.Models[0] != "llama" {
			t.Fatalf("unexpected projection: %+v", got)
		}

		noIP := withService
		noIP.IP = ""
		if _, ok := subscribedToNode(tc.profile, noIP); ok {
			t.Fatal("node without IP should not project")
		}

		// A node advertising only a non-engine service (node-info) is not a
		// routing target.
		niOnly := noderec.DirectoryNode{
			Name:     "host-c",
			IP:       "10.0.0.6",
			Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: 14318}},
		}
		if _, ok := subscribedToNode(tc.profile, niOnly); ok {
			t.Fatal("node without this engine's service should not project")
		}

		// Nor is a node running only the *other* engine.
		otherOnly := noderec.DirectoryNode{
			Name: "host-e",
			IP:   "10.0.0.8",
			Services: map[noderec.ServiceKey]noderec.ServiceStatus{
				other.DiscoveryService: {Port: other.FacadePort},
			},
		}
		if _, ok := subscribedToNode(tc.profile, otherOnly); ok {
			t.Fatalf("a node running only %s should not project as a %s target",
				other.DisplayName, tc.profile.DisplayName)
		}

		// Per-engine attribution: a dual-engine node projects ONLY this
		// engine's models, never the cross-engine union, so a model served
		// solely via the other engine is not ranked as an owner here.
		dual := noderec.DirectoryNode{
			HostUUID: "uuid-d",
			Name:     "host-d",
			IP:       "10.0.0.7",
			Models:   []string{"mine", "theirs"},
			ModelsByEngine: map[string][]string{
				tc.profile.Name: {"mine"},
				other.Name:      {"theirs"},
			},
			Services: map[noderec.ServiceKey]noderec.ServiceStatus{
				tc.profile.DiscoveryService: {Port: tc.profile.FacadePort},
			},
		}
		got, ok = subscribedToNode(tc.profile, dual)
		if !ok {
			t.Fatal("dual-engine node advertising this engine should project")
		}
		if len(got.Models) != 1 || got.Models[0] != "mine" {
			t.Fatalf("dual-engine projection Models = %v, want [mine] only", got.Models)
		}
	})
}

// TestSubscribedToNodeKeysByHostUUID: the routable Node keys on the stable
// hostUuid, not the hostname — so routing, scheduledOn, node selection, and the
// scheduler's priority list all survive a PC rename and never conflate two
// same-named machines. Host stays the hostname for display.
func TestSubscribedToNodeKeysByHostUUID(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		const uuid = "11111111-1111-1111-1111-111111111111"
		n := noderec.DirectoryNode{
			HostUUID: uuid,
			Name:     "host-a",
			IP:       "10.0.0.5",
			Services: map[noderec.ServiceKey]noderec.ServiceStatus{
				tc.profile.DiscoveryService: {Port: tc.profile.FacadePort},
			},
		}
		got, ok := subscribedToNode(tc.profile, n)
		if !ok {
			t.Fatal("node advertising this engine + IP should project")
		}
		if got.ID != uuid {
			t.Fatalf("ID = %q, want hostUuid %q", got.ID, uuid)
		}
		if got.Host != "host-a" {
			t.Fatalf("Host = %q, want hostname for display", got.Host)
		}
	})
}

// TestSubscribedCarriesPolicyWithoutChurn covers the proxy overlay hop of the
// roles advertisement: subscribedToNode copies the node's declared roles and
// policy revision onto the routable Node, a node without an advertisement
// declares nothing, and a role edit alone does not announce node/updated.
func TestSubscribedCarriesPolicyWithoutChurn(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		n := noderec.DirectoryNode{
			HostUUID: "host-uuid",
			Name:     "host-a",
			IP:       "10.0.0.5",
			Services: map[noderec.ServiceKey]noderec.ServiceStatus{
				tc.profile.DiscoveryService: {Port: tc.profile.FacadePort},
			},
		}
		undeclared, ok := subscribedToNode(tc.profile, n)
		if !ok {
			t.Fatal("node advertising this engine + IP should project")
		}
		if undeclared.Roles != nil || undeclared.PolicyRevision != "" {
			t.Fatalf("node without an advertisement projected roles %+v at revision %q", undeclared.Roles, undeclared.PolicyRevision)
		}

		roles := map[string]nodepolicy.AdvertisedRole{nodepolicy.TierStrong: {Engine: tc.profile.Name, Model: "big", ContextTokens: 32768}}
		n.Policy = &nodepolicy.Advertisement{Revision: "40", Roles: roles}
		declared, ok := subscribedToNode(tc.profile, n)
		if !ok {
			t.Fatal("declared node should project")
		}
		if !reflect.DeepEqual(declared.Roles, roles) || declared.PolicyRevision != "40" {
			t.Fatalf("projected roles %+v at revision %q, want %+v at 40", declared.Roles, declared.PolicyRevision, roles)
		}
		roles[nodepolicy.TierWeak] = nodepolicy.AdvertisedRole{Engine: tc.profile.Name, Model: "small"}
		if _, aliased := declared.Roles[nodepolicy.TierWeak]; aliased {
			t.Fatal("projected roles alias the directory node's map")
		}

		d := NewDiscovery()
		d.SetSubscribed([]Node{undeclared})
		disc, upd, rem := d.SetSubscribed([]Node{declared})
		if len(disc)+len(upd)+len(rem) != 0 {
			t.Fatalf("a role declaration alone announced changes: disc=%v upd=%v rem=%v", disc, upd, rem)
		}
		nodes := d.Nodes()
		if len(nodes) != 1 || nodes[0].PolicyRevision != "40" {
			t.Fatalf("overlay did not take the declared node: %+v", nodes)
		}
	})
}

// strictTestCapability is a valid capability for model on ollama.
func strictTestCapability(model string) *nodepolicy.StrictAdmissionCapability {
	return &nodepolicy.StrictAdmissionCapability{Version: nodepolicy.StrictAdmissionVersion, Models: []nodepolicy.StrictAdmissionModel{
		{Engine: "ollama", Model: model, MaxConcurrent: 2, Identity: strictRequest().Identity},
	}}
}

// unsupportedStrictCapability is a capability of a future version, decoded the
// way a newer peer's inventory would be.
func unsupportedStrictCapability(t *testing.T) *nodepolicy.StrictAdmissionCapability {
	t.Helper()
	var c nodepolicy.StrictAdmissionCapability
	if err := json.Unmarshal([]byte(`{"version":2,"models":"opaque"}`), &c); err != nil {
		t.Fatal(err)
	}
	if c.State() != nodepolicy.StrictCapabilityUnsupported {
		t.Fatalf("fixture state = %s, want unsupported", c.State())
	}
	return &c
}

// TestSubscribedCarriesStrictCapabilityWithoutChurn covers the proxy overlay hop
// of the strict admission capability: subscribedToNode carries it as decoded,
// whatever its version, a node without one carries none, and a readiness change
// alone does not announce node/updated.
func TestSubscribedCarriesStrictCapabilityWithoutChurn(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		n := noderec.DirectoryNode{
			HostUUID: "host-uuid",
			Name:     "host-a",
			IP:       "10.0.0.5",
			Services: map[noderec.ServiceKey]noderec.ServiceStatus{tc.profile.DiscoveryService: {Port: tc.profile.FacadePort}},
			Policy:   &nodepolicy.Advertisement{Revision: "40"},
		}
		without, ok := subscribedToNode(tc.profile, n)
		if !ok || without.StrictAdmission != nil {
			t.Fatalf("node without a capability projected %+v (ok %v)", without.StrictAdmission, ok)
		}
		for name, capability := range map[string]*nodepolicy.StrictAdmissionCapability{
			"valid":       strictTestCapability("m:latest"),
			"unsupported": unsupportedStrictCapability(t),
		} {
			n.Policy = &nodepolicy.Advertisement{Revision: "40", StrictAdmission: capability}
			with, ok := subscribedToNode(tc.profile, n)
			if !ok || with.StrictAdmission != capability {
				t.Fatalf("%s capability projected as %+v (ok %v), want the decoded value", name, with.StrictAdmission, ok)
			}
			d := NewDiscovery()
			d.SetSubscribed([]Node{without})
			if disc, upd, rem := d.SetSubscribed([]Node{with}); len(disc)+len(upd)+len(rem) != 0 {
				t.Fatalf("a %s capability alone announced changes: disc=%v upd=%v rem=%v", name, disc, upd, rem)
			}
			if nodes := d.Nodes(); len(nodes) != 1 || nodes[0].StrictAdmission != capability {
				t.Fatalf("overlay did not take the %s capability: %+v", name, nodes)
			}
		}
	})
}

// TestStrictModelExcludesPeersWithoutAValidCapability covers the hybrid
// exclusion rule: only a valid capability naming the engine and normalized
// model makes a peer eligible, while every one of these peers stays a
// concrete-routing candidate for the same model.
func TestStrictModelExcludesPeersWithoutAValidCapability(t *testing.T) {
	profile := ollamaOnlyProfile(t)
	model := profile.normalizeModel("m")
	cases := []struct {
		name     string
		policy   *nodepolicy.Advertisement
		eligible bool
	}{
		{"older peer without a policy", nil, false},
		{"policy without a capability", &nodepolicy.Advertisement{Revision: "3"}, false},
		{"unsupported version", &nodepolicy.Advertisement{Revision: "3", StrictAdmission: unsupportedStrictCapability(t)}, false},
		{"valid without that model", &nodepolicy.Advertisement{Revision: "3", StrictAdmission: strictTestCapability("other:latest")}, false},
		{"valid for that model", &nodepolicy.Advertisement{Revision: "3", StrictAdmission: strictTestCapability(model)}, true},
	}
	port := freeTCPPort(t)
	var nodes []Node
	var principals []string
	for i, tc := range cases {
		principal := fmt.Sprintf("principal-%d", i)
		principals = append(principals, principal)
		n, ok := subscribedToNode(profile, noderec.DirectoryNode{
			HostUUID:    fmt.Sprintf("peer-%d", i),
			ClusterUUID: principal,
			Name:        tc.name,
			IP:          fmt.Sprintf("192.0.2.%d", 10+i),
			Services:    map[noderec.ServiceKey]noderec.ServiceStatus{profile.DiscoveryService: {Port: port}},
			Models:      []string{model},
			Policy:      tc.policy,
		})
		if !ok {
			t.Fatalf("%s did not project", tc.name)
		}
		nodes = append(nodes, n)
		t.Run(tc.name, func(t *testing.T) {
			entry, eligible := n.strictModel("ollama", model)
			if eligible != tc.eligible {
				t.Fatalf("eligible = %v, want %v", eligible, tc.eligible)
			}
			if eligible && (entry.Model != model || entry.MaxConcurrent != 2) {
				t.Fatalf("entry = %+v, want %s with maxConcurrent 2", entry, model)
			}
			if _, other := n.strictModel("lmstudio", model); other {
				t.Fatal("an entry for another engine made the node eligible")
			}
		})
	}

	disc := NewDiscovery()
	disc.SetSubscribed(nodes)
	p := testProxy(profile, disc, freeTCPPort(t))
	clusterDir := filepath.Join(t.TempDir(), "cluster")
	clustertrusttest.Join(t, clusterDir, "cluster-xyz", "principal-self", principals...)
	p.mesh = clustertrust.Open(clusterDir)
	got := map[string]bool{}
	for _, c := range p.soleFacade().resolveCandidates("m") {
		got[c.id] = true
	}
	for _, n := range nodes {
		if !got[n.ID] {
			t.Errorf("concrete candidates %v lost %s", got, n.Host)
		}
	}
}
