// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

// strictModelJSON is one valid version-1 capability entry for model on ollama.
func strictModelJSON(model string) string {
	return `{"engine":"ollama","model":"` + model + `","maxConcurrent":1,"identity":{"engineRunId":"engine-run-1","settingsFingerprint":"` +
		strings.Repeat("a", 64) + `","modelDigest":"` + strings.Repeat("b", 64) + `"}}`
}

// validStrictJSON is a valid version-1 capability for model on ollama.
func validStrictJSON(model string) string {
	return `{"version":1,"models":[` + strictModelJSON(model) + `]}`
}

const capabilityRoles = `{"medium":{"engine":"ollama","model":"model","contextTokens":16384}}`

// policyWith is a policy advertisement with the fixture's roles and extra
// top-level members.
func policyWith(extra string) json.RawMessage {
	doc := `{"revision":"31","roles":` + capabilityRoles
	if extra != "" {
		doc += "," + extra
	}
	return json.RawMessage(doc + "}")
}

var capabilityTestRoles = map[string]nodepolicy.AdvertisedRole{nodepolicy.TierMedium: {Engine: "ollama", Model: "model", ContextTokens: 16384}}

func TestDecodeInventoryPolicy(t *testing.T) {
	type want struct {
		err        bool
		policy     bool
		state      nodepolicy.StrictCapabilityState
		keepsRoles bool
		rejected   string
	}
	for _, tc := range []struct {
		name string
		raw  json.RawMessage
		want want
	}{
		{"absent", nil, want{state: nodepolicy.StrictCapabilityAbsent}},
		{"null policy", json.RawMessage("null"), want{state: nodepolicy.StrictCapabilityAbsent}},
		{"no capability", policyWith(""), want{policy: true, state: nodepolicy.StrictCapabilityAbsent, keepsRoles: true}},
		{"valid capability", policyWith(`"strictAdmission":` + validStrictJSON("m:latest")), want{policy: true, state: nodepolicy.StrictCapabilityValid, keepsRoles: true}},
		{"unsupported version kept", policyWith(`"strictAdmission":{"version":2,"future":true}`), want{policy: true, state: nodepolicy.StrictCapabilityUnsupported, keepsRoles: true}},
		{"invalid version 1 dropped", policyWith(`"strictAdmission":{"version":1,"models":[]}`), want{policy: true, state: nodepolicy.StrictCapabilityAbsent, keepsRoles: true, rejected: rejectedCapability}},
		{"unknown version 1 field dropped", policyWith(`"strictAdmission":{"version":1,"models":[` + strictModelJSON("m") + `],"extra":1}`), want{policy: true, state: nodepolicy.StrictCapabilityAbsent, keepsRoles: true, rejected: rejectedCapability}},
		{"invalid envelope dropped", policyWith(`"strictAdmission":{"version":"one"}`), want{policy: true, state: nodepolicy.StrictCapabilityAbsent, keepsRoles: true, rejected: rejectedCapability}},
		{"null capability dropped", policyWith(`"strictAdmission":null`), want{policy: true, state: nodepolicy.StrictCapabilityAbsent, keepsRoles: true, rejected: rejectedCapability}},
		{"case-variant key rejects the advertisement", policyWith(`"StrictAdmission":` + validStrictJSON("m:latest")), want{rejected: rejectedAdvertisement}},
		{"mixed keys reject the advertisement", policyWith(`"strictAdmission":` + validStrictJSON("m:latest") + `,"STRICTADMISSION":{"version":2}`), want{rejected: rejectedAdvertisement}},
		{"invalid roles fail the document", json.RawMessage(`{"revision":"31","roles":[],"strictAdmission":{"version":1,"models":[]}}`), want{err: true}},
		{"invalid revision fails the document", json.RawMessage(`{"revision":31,"roles":{}}`), want{err: true}},
		{"not an object fails the document", json.RawMessage(`[]`), want{err: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			adv, rejected, err := decodeInventoryPolicy(tc.raw)
			if (err != nil) != tc.want.err {
				t.Fatalf("err = %v, want error %v", err, tc.want.err)
			}
			if rejected != tc.want.rejected {
				t.Fatalf("rejected = %q, want %q", rejected, tc.want.rejected)
			}
			if (adv != nil) != tc.want.policy {
				t.Fatalf("advertisement = %+v, want present %v", adv, tc.want.policy)
			}
			if adv == nil {
				return
			}
			if got := adv.StrictAdmission.State(); got != tc.want.state {
				t.Fatalf("capability state = %s, want %s", got, tc.want.state)
			}
			if tc.want.keepsRoles && (adv.Revision != "31" || !reflect.DeepEqual(adv.Roles, capabilityTestRoles)) {
				t.Fatalf("advertisement = revision %q roles %+v, want the fixture's", adv.Revision, adv.Roles)
			}
			// Whatever the scanner keeps must survive the directory, broker and
			// proxy hops, which decode it strictly.
			wire, err := json.Marshal(noderec.DirectoryNode{HostUUID: "peer", Policy: adv})
			if err != nil {
				t.Fatalf("marshal the directory node: %v", err)
			}
			var back noderec.DirectoryNode
			if err := json.Unmarshal(wire, &back); err != nil {
				t.Fatalf("downstream decode of %s: %v", wire, err)
			}
			if back.Policy.StrictAdmission.State() != tc.want.state {
				t.Fatalf("downstream capability state = %s, want %s", back.Policy.StrictAdmission.State(), tc.want.state)
			}
		})
	}
}

// TestDecodeInventoryPolicyKeepsTheSharedDecodingOfOtherFields checks that
// dropping a malformed capability leaves every other member to decode exactly as
// the shared decoder decodes the same document without the capability, keeping
// its document order and duplicate members.
func TestDecodeInventoryPolicyKeepsTheSharedDecodingOfOtherFields(t *testing.T) {
	for _, tc := range []struct{ name, without string }{
		{"later case-folded duplicate wins", `{"revision":"1","Revision":"2","roles":{}`},
		{"roles after a null spelling", `{"revision":"1","roles":null,"Roles":` + capabilityRoles},
		{"null after the roles", `{"revision":"1","Roles":` + capabilityRoles + `,"roles":null`},
		{"exact duplicate", `{"revision":"1","revision":"3","roles":{}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var want nodepolicy.Advertisement
			if err := json.Unmarshal([]byte(tc.without+"}"), &want); err != nil {
				t.Fatal(err)
			}
			for _, placement := range []string{"first", "last"} {
				doc := tc.without + `,"strictAdmission":{"version":1,"models":[]}}`
				if placement == "first" {
					doc = `{"strictAdmission":{"version":1,"models":[]},` + strings.TrimPrefix(tc.without, "{") + "}"
				}
				got, rejected, err := decodeInventoryPolicy(json.RawMessage(doc))
				if err != nil || rejected != rejectedCapability || got == nil {
					t.Fatalf("%s: decode %s = %+v, %q, %v", placement, doc, got, rejected, err)
				}
				if !reflect.DeepEqual(*got, want) {
					t.Fatalf("%s: decoded %+v, want the shared decoding %+v", placement, *got, want)
				}
			}
		})
	}
}

// TestInvalidCapabilityKeepsTheInventory is the concrete-routing half of the
// capability rule on the pinned peer fetch: a peer whose capability is
// malformed, or whose advertisement uses an aliased capability key, keeps its
// model inventory; only hybrid eligibility is lost.
func TestInvalidCapabilityKeepsTheInventory(t *testing.T) {
	for _, tc := range []struct {
		name      string
		policy    json.RawMessage
		wantRoles bool
	}{
		{"malformed capability", policyWith(`"strictAdmission":{"version":1,"models":[]}`), true},
		{"null capability", policyWith(`"strictAdmission":null`), true},
		{"aliased capability key", policyWith(`"StrictAdmission":` + validStrictJSON("m:latest")), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newScannerReverseFixture(t)
			f.direct.Store(true)
			f.policy.Store(&tc.policy)
			node := f.node
			f.d.enrich(&node)
			if !reflect.DeepEqual(node.Models, []string{"model"}) {
				t.Fatalf("models = %v, want [model]", node.Models)
			}
			switch {
			case tc.wantRoles && (node.Policy == nil || !reflect.DeepEqual(node.Policy.Roles, capabilityTestRoles) || node.Policy.StrictAdmission != nil):
				t.Fatalf("policy = %+v, want the roles without a capability", node.Policy)
			case !tc.wantRoles && node.Policy != nil:
				t.Fatalf("policy = %+v, want none", node.Policy)
			}
		})
	}
}

// TestRefreshWithdrawsTheCapability checks that a successful answer without the
// capability replaces the last one rather than keeping it as last-good.
func TestRefreshWithdrawsTheCapability(t *testing.T) {
	d := newRefreshTestDaemon()
	stub, port := startModelsStub(t, http.StatusOK, map[string]any{"models": []string{"a"}, "policy": policyWith(`"strictAdmission":` + validStrictJSON("m:latest"))})
	seedEMNode(d, "peer-P", "127.0.0.1", port)
	if !d.refreshNodeModels("peer-P", "127.0.0.1", port, "") {
		t.Fatal("the inventory was not applied")
	}
	if got, _ := d.dir.get("peer-P"); got.Policy.StrictAdmission.State() != nodepolicy.StrictCapabilityValid {
		t.Fatalf("capability = %+v, want a valid one", got.Policy.StrictAdmission)
	}

	stub.set(http.StatusOK, map[string]any{"models": []string{"a"}, "policy": policyWith("")})
	if !d.refreshNodeModels("peer-P", "127.0.0.1", port, "") {
		t.Fatal("a withdrawn capability alone should report a change")
	}
	if got, _ := d.dir.get("peer-P"); got.Policy == nil || got.Policy.StrictAdmission != nil {
		t.Fatalf("policy after withdrawal = %+v, want the advertisement without a capability", got.Policy)
	}
	d.infoMu.Lock()
	cached := d.lastPolicy["peer-P"]
	d.infoMu.Unlock()
	if cached == nil || cached.StrictAdmission != nil {
		t.Fatalf("cached policy = %+v, want no capability", cached)
	}
}

// TestPeerServicesBootstrapEnrichmentCarriesTheCapability covers capability
// discovery for a peer known only from its authenticated claim, with no mDNS
// record: enrichPeerServices reuses the pinned /v1/models fetch and records the
// capability on the node and in the last-good cache.
func TestPeerServicesBootstrapEnrichmentCarriesTheCapability(t *testing.T) {
	f := newScannerReverseFixture(t)
	f.direct.Store(true)
	served := policyWith(`"strictAdmission":` + validStrictJSON("m:latest"))
	f.policy.Store(&served)
	if _, existed := f.d.dir.removeBrowse("peer"); !existed {
		t.Fatal("fixture peer had no browse record to remove")
	}
	incarnation, ok := f.d.mesh.PeerServiceIncarnation("peer")
	if !ok {
		t.Fatal("fixture has no current incarnation")
	}
	claim := noderec.PeerServicesClaim{Incarnation: incarnation, Source: "cm-first", Sequence: 1, ObservedAt: time.Now().UnixMilli(),
		Descriptor: noderec.PeerServicesDescriptor{HostUUID: "peer", Name: "peer-name", Addresses: []string{"192.0.2.40"}}}
	// Installed directly, as in the roles bootstrap test: descriptor validation
	// refuses the loopback address the fixture serves on.
	f.d.dir.installPeerServices(claim, f.node)
	if !f.d.dir.authenticated("peer") {
		t.Fatal("peer is not held by an authenticated claim")
	}
	f.d.enrichPeerServices(claim, f.node, f.d.dir.routeGeneration("peer"))

	var want nodepolicy.Advertisement
	if err := json.Unmarshal(served, &want); err != nil {
		t.Fatal(err)
	}
	node := peerServicesNode(t, f.d)
	if node.Policy == nil || !reflect.DeepEqual(*node.Policy, want) || node.Policy.StrictAdmission.State() != nodepolicy.StrictCapabilityValid {
		t.Fatalf("bootstrap enrichment policy = %+v, want %+v", node.Policy, want)
	}
	if !reflect.DeepEqual(node.Models, []string{"model"}) {
		t.Fatalf("bootstrap enrichment models = %v, want [model]", node.Models)
	}
	if got := cachedPolicy(f.d, "peer"); got == nil || !reflect.DeepEqual(*got, want) {
		t.Fatalf("cached policy = %+v, want %+v", got, want)
	}
}
