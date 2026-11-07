// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func xaPlatform(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "darwin" {
		t.Skip("XA cross-process acceptance is reserved for Windows PC3; no Mac processes")
	}
	if runtime.GOOS == "windows" {
		if !strings.Contains(os.Getenv("GOFLAGS"), "nvpair-test-exec") || !filepath.IsAbs(os.Getenv("NVPAIR_TEST_BIN_DIR")) {
			t.Fatal("Windows XA fixture requires the configured fixed-path wrapper and NVPAIR_TEST_BIN_DIR")
		}
	}
}

func TestXARemoteConfigurationService(t *testing.T) {
	xaPlatform(t)
	rf := newRemoteServiceFixture(t)
	// Lock verifies transport shutdown, so absent engines need a closed
	// fixture port rather than the service fixture's live HTTP 503 sink.
	jointIsolateEngines(t, rf.configDir, temporaryClusterPort(t))
	rf.start()
	c := rf.readyClient()
	var snap servicePolicySnapshot

	if !t.Run("local grants through the service", func(t *testing.T) {
		assertGrant := func(view managementSnapshot, peer string, allowed bool) {
			t.Helper()
			if view.Version != 1 || view.Revision == "" {
				t.Fatal("service management snapshot lacks a valid version or revision")
			}
			for _, grant := range view.Grants {
				if grant.PeerUUID == peer {
					wantState := "denied"
					if allowed {
						wantState = "allowed"
					}
					if grant.Allowed != allowed || grant.State != wantState {
						t.Fatalf("service grant for %q = %t/%s, want %t/%s", peer, grant.Allowed, grant.State, allowed, wantState)
					}
					return
				}
			}
			t.Fatalf("service management snapshot lacks peer %q", peer)
		}
		view := decodeResult[managementSnapshot](t, *c.mustCall(t, "901", "cluster:get-remote-management", mustMarshal(t, map[string]any{})))
		if view.Version != 1 || view.Revision == "" {
			t.Fatal("initial service management snapshot is unavailable")
		}
		denied := decodeResult[managementSnapshot](t, *c.mustCall(t, "902", "cluster:set-remote-management", mustMarshal(t, map[string]any{
			"peerUuid": "management-service-peer", "allowed": false, "expectedRevision": view.Revision,
		})))
		assertGrant(denied, "management-service-peer", false)
		if denied.Revision == view.Revision {
			t.Fatal("explicit service denial did not advance management revision")
		}
		view = decodeResult[managementSnapshot](t, *c.mustCall(t, "903", "cluster:get-remote-management", mustMarshal(t, map[string]any{})))
		assertGrant(view, "management-service-peer", false)
		if view.Revision != denied.Revision {
			t.Fatal("service management read differs from the accepted denial revision")
		}
		allowed := decodeResult[managementSnapshot](t, *c.mustCall(t, "904", "cluster:set-remote-management", mustMarshal(t, map[string]any{
			"peerUuid": "management-service-allowed", "allowed": true, "expectedRevision": view.Revision,
		})))
		assertGrant(allowed, "management-service-peer", false)
		assertGrant(allowed, "management-service-allowed", true)
		if allowed.Revision == view.Revision {
			t.Fatal("explicit service allowance did not advance management revision")
		}
	}) {
		return
	}

	if !t.Run("local policy edit with roles and served set", func(t *testing.T) {
		before := decodeServicePolicy(t, c.mustCall(t, "905", "policy:get", nil))
		draft := before
		draft.Profiles = append(draft.Profiles, nodepolicy.Profile{Key: "xa-model", DisplayName: "xa-model", Engine: "llamacpp", Model: "xa-model"})
		draft.Roles = []nodepolicy.Role{{Name: "weak", Profiles: []string{"xa-model"}}}
		draft.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "xa-model"}}
		m := c.mustCall(t, "906", "policy:set", mustMarshal(t, map[string]any{"policy": draft, "expectedPersistedRevision": before.Revision}))
		var accepted struct {
			Policy          nodepolicy.Policy `json:"policy"`
			ActiveRevision  string            `json:"activeRevision"`
			Activation      string            `json:"activation"`
			ActivationError string            `json:"activationError,omitempty"`
		}
		if err := json.Unmarshal(m.Result, &accepted); err != nil {
			t.Fatalf("decode service policy:set result: %v", err)
		}
		if err := accepted.Policy.Validate(); err != nil {
			t.Fatalf("service policy:set returned an invalid policy: %v", err)
		}
		if accepted.Policy.Revision == "" || accepted.Policy.Revision == before.Revision || !reflect.DeepEqual(accepted.Policy.Profiles, draft.Profiles) || !reflect.DeepEqual(accepted.Policy.Roles, draft.Roles) || !reflect.DeepEqual(accepted.Policy.ServedModels, draft.ServedModels) || accepted.Activation != "pending" || accepted.ActiveRevision != "" || accepted.ActivationError != "" {
			t.Fatal("service policy edit lacks a new revision, exact roles and served set, or pending activation")
		}
		snap = decodeServicePolicySnapshot(t, c.mustCall(t, "909", "policy:get", nil))
		if !reflect.DeepEqual(snap.Policy, accepted.Policy) || snap.Activation != accepted.Activation || snap.ActiveRevision != accepted.ActiveRevision {
			t.Fatal("service policy:get differs from the accepted policy:set result")
		}
		assertStoredPolicy(t, rf.policyPath(), snap.Policy)
	}) {
		return
	}

	if !t.Run("peers see the service-made policy", func(t *testing.T) {
		for _, peer := range []struct {
			name   string
			client *http.Client
		}{{"allowed", rf.allowed}, {"denied", rf.peer}} {
			data := managementHTTP(t, peer.client, rf.ports.engineControl, http.MethodPost, "/v1/node-policy/get", map[string]any{}, http.StatusOK)
			var view struct {
				Policy nodepolicy.PeerPolicy `json:"policy"`
			}
			if err := json.Unmarshal(data, &view); err != nil {
				t.Fatalf("decode %s peer service policy: %v", peer.name, err)
			}
			if !reflect.DeepEqual(view.Policy, snap.Policy.PeerProjection()) {
				t.Fatalf("%s peer policy differs from the full service-made projection", peer.name)
			}
		}
		before := readPolicyBytes(t, rf.policyPath())
		managementHTTP(t, rf.peer, rf.ports.engineControl, http.MethodPost, "/v1/node-policy/set", map[string]any{
			"policy": snap.Policy.PeerProjection(), "expectedPersistedRevision": snap.Policy.Revision,
		}, http.StatusConflict)
		if string(readPolicyBytes(t, rf.policyPath())) != string(before) {
			t.Fatal("denied peer changed stored service policy bytes")
		}
		assertStoredPolicy(t, rf.policyPath(), snap.Policy)
	}) {
		return
	}

	t.Run("clean stop", func(t *testing.T) {
		assertServiceStoppedReply(t, c.mustCall(t, "910", "service/stop", nil))
		rf.waitChild(rf.service, serviceStopTimeout)
		rf.assertStopped()
	})
}

func TestXARemoteConfigurationStatus(t *testing.T) {
	xaPlatform(t)
	f := newPolicyJointFixture(t)

	t.Run("policy projection", func(t *testing.T) {
		body := f.rpc(t, "engine:remote-policy-get", map[string]any{}).Result
		assertJointPeer(t, body, f.want)
		var view struct {
			Availability nodepolicy.Availability `json:"availability"`
		}
		if err := json.Unmarshal(body, &view); err != nil {
			t.Fatalf("decode remote policy availability: %v", err)
		}
		if view.Availability != nodepolicy.Paused {
			t.Fatalf("remote availability = %q, want paused", view.Availability)
		}
		f.assertPolicy(t)
	})

	t.Run("management grant", func(t *testing.T) {
		m := f.originCall(t, "engine:remote-management-get", map[string]any{})
		if m.Error != nil {
			t.Fatalf("remote management read failed: code=%d", m.Error.Code)
		}
		var view struct {
			Version  int    `json:"version"`
			Allowed  bool   `json:"allowed"`
			Revision string `json:"revision"`
			State    string `json:"state"`
		}
		if err := json.Unmarshal(m.Result, &view); err != nil {
			t.Fatalf("decode remote management grant: %v", err)
		}
		if view.Version != 1 || !view.Allowed || view.Revision == "" || view.State != "allowed" {
			t.Fatalf("remote management grant: version=%d allowed=%t revision_present=%t state=%q, want version 1, allowed, nonempty revision and allowed state", view.Version, view.Allowed, view.Revision != "", view.State)
		}
	})

}

func TestXARemoteConfigurationConflict(t *testing.T) {
	xaPlatform(t)
	f := newPolicyJointFixture(t, jointAvailable)
	before := assertStoredPolicy(t, f.path, f.want)
	params := map[string]any{"policy": jointExpectedPeer(f.want), "expectedPersistedRevision": "0"}
	conflict := "policy conflict: the persisted revision is " + f.want.Revision + ", not the expected 0; reload the policy and review it before saving"
	body := f.http(t, "/v1/node-policy/set", params, http.StatusConflict)
	var refusal struct {
		Error string `json:"error"`
		Code  string `json:"code"`
	}
	if err := json.Unmarshal(body, &refusal); err != nil {
		t.Fatalf("decode HTTP policy conflict: %v", err)
	}
	if refusal.Code != "policy_conflict" || refusal.Error != conflict {
		t.Fatal("HTTP policy conflict differs from the specified code or diagnostic")
	}
	f.assertUnchanged(t, before)
	m := f.originCall(t, "engine:remote-policy-set", params)
	if m.Error == nil || m.Error.Code != -32010 || len(m.Result) != 0 || m.Error.Message != "remote node policy: HTTP 409: "+conflict {
		t.Fatalf("origin policy conflict mismatch: error_present=%t result_bytes=%d", m.Error != nil, len(m.Result))
	}
	f.assertUnchanged(t, before)

	draft := jointExpectedPeer(f.want)
	draft.Idle.UnloadAfterMinutes++
	oldRevision := f.want.Revision
	controlParams := map[string]any{"policy": draft, "expectedPersistedRevision": oldRevision}
	res := f.rpc(t, "engine:remote-policy-set", controlParams)
	var accepted struct {
		Policy nodepolicy.PeerPolicy `json:"policy"`
	}
	if err := json.Unmarshal(res.Result, &accepted); err != nil {
		t.Fatalf("decode accepted policy edit: %v", err)
	}
	if accepted.Policy.Revision == "" || accepted.Policy.Revision == oldRevision {
		t.Fatal("accepted policy edit did not return a new nonempty revision")
	}
	f.want.Idle = draft.Idle
	f.want.Revision = accepted.Policy.Revision
	assertJointPeer(t, res.Result, f.want)
	f.assertPolicy(t)
	beforeRetry := assertStoredPolicy(t, f.path, f.want)

	m = f.originCall(t, "engine:remote-policy-set", controlParams)
	conflict = "policy conflict: the persisted revision is " + f.want.Revision + ", not the expected " + oldRevision + "; reload the policy and review it before saving"
	if m.Error == nil || m.Error.Code != -32010 || len(m.Result) != 0 || m.Error.Message != "remote node policy: HTTP 409: "+conflict {
		t.Fatalf("stale control retry mismatch: error_present=%t result_bytes=%d", m.Error != nil, len(m.Result))
	}
	f.assertUnchanged(t, beforeRetry)
}

func TestXARemoteConfigurationVeto(t *testing.T) {
	xaPlatform(t)
	f := newPolicyJointFixture(t)

	if !t.Run("relayed resume is vetoed", func(t *testing.T) {
		before := assertStoredPolicy(t, f.path, f.want)
		m := f.originCall(t, "engine:remote-availability-set", map[string]any{"state": "available"})
		const message = "remote node policy: HTTP 409: This node is paused. Resume it on this node before managing it remotely."
		if m.Error == nil || m.Error.Code != -32011 || m.Error.Message != message || len(m.Result) != 0 {
			t.Fatalf("relayed resume veto mismatch: error_present=%t result_bytes=%d", m.Error != nil, len(m.Result))
		}
		f.assertUnchanged(t, before)
	}) {
		return
	}

	if !t.Run("remote installation start is vetoed while paused", func(t *testing.T) {
		task1Eventually(t, "remote installation refused by local pause", 15*time.Second, func() bool {
			m := callRPC(t, f.origin.in, f.origin.msgs, "engine:remote-install", map[string]any{"node": "policy-joint-target", "engine": "llamacpp"}, 30*time.Second)
			return m.Error != nil && m.Error.Code == -32011 && m.Error.Message == "remote /v1/engines/install: HTTP 409: This node is paused. Resume it on this node before managing it remotely." && len(m.Result) == 0
		})
	}) {
		return
	}
}

func xaEngineModelsPolicy(t *testing.T, p *interopProcess) *nodepolicy.Advertisement {
	t.Helper()
	m := callRPC(t, p.in, p.msgs, "engine:models", map[string]any{}, 10*time.Second)
	if m.Error != nil {
		t.Fatalf("engine models read failed: code=%d", m.Error.Code)
	}
	var result struct {
		Policy *nodepolicy.Advertisement `json:"policy"`
	}
	if err := json.Unmarshal(m.Result, &result); err != nil {
		t.Fatalf("decode engine models policy: %v", err)
	}
	return result.Policy
}

func xaPeerDraft(t *testing.T, p nodepolicy.Policy) nodepolicy.PeerPolicy {
	t.Helper()
	data, err := json.Marshal(p.PeerProjection())
	if err != nil {
		t.Fatalf("encode peer policy draft: %v", err)
	}
	var draft nodepolicy.PeerPolicy
	if err := json.Unmarshal(data, &draft); err != nil {
		t.Fatalf("decode peer policy draft: %v", err)
	}
	return draft
}

func TestXARemoteConfigurationRoles(t *testing.T) {
	xaPlatform(t)
	f := newManagementFixture(t, jointAvailable)
	j := f.joint

	if !t.Run("allowed peer sets roles", func(t *testing.T) {
		draft := xaPeerDraft(t, j.want)
		draft.Roles = append(draft.Roles, nodepolicy.Role{Name: "xa-role", Profiles: []string{"joint-local"}})
		res := j.rpc(t, "engine:remote-policy-set", map[string]any{"policy": draft, "expectedPersistedRevision": j.want.Revision}).Result
		var accepted struct {
			Policy nodepolicy.PeerPolicy `json:"policy"`
		}
		if err := json.Unmarshal(res, &accepted); err != nil {
			t.Fatalf("decode accepted roles edit: %v", err)
		}
		if !reflect.DeepEqual(accepted.Policy.Profiles, draft.Profiles) || !reflect.DeepEqual(accepted.Policy.Roles, draft.Roles) || accepted.Policy.Revision == "" || accepted.Policy.Revision == j.want.Revision {
			t.Fatal("accepted roles edit differs from the requested roles or lacks a new revision")
		}
		j.want.Profiles = draft.Profiles
		j.want.Roles = draft.Roles
		j.want.Revision = accepted.Policy.Revision
		j.assertPolicy(t)
	}) {
		return
	}

	if !t.Run("target advertises roles", func(t *testing.T) {
		task1Eventually(t, "roles projection", 10*time.Second, func() bool {
			adv := xaEngineModelsPolicy(t, j.target)
			return adv != nil && adv.Revision == j.want.Revision &&
				reflect.DeepEqual(adv.Roles, []nodepolicy.Role{
					{Name: "weak", Profiles: []string{"joint-local"}},
					{Name: "xa-role", Profiles: []string{"joint-local"}},
				}) &&
				reflect.DeepEqual(adv.Profiles, []nodepolicy.AdvertisedProfile{{
					Key: "joint-local", DisplayName: "joint-local", Engine: "llamacpp", Model: "joint-local",
					ContextTokens: 8192, Capabilities: []nodepolicy.Capability{nodepolicy.CapTools},
				}})
		})
	}) {
		return
	}

	if !t.Run("denied peer cannot change roles", func(t *testing.T) {
		before := readPolicyBytes(t, j.path)
		draft := xaPeerDraft(t, j.want)
		draft.Roles = []nodepolicy.Role{{Name: "strong", Profiles: []string{"joint-local"}}}
		managementHTTP(t, f.denied, j.ports[1], http.MethodPost, "/v1/node-policy/set", map[string]any{"policy": draft, "expectedPersistedRevision": j.want.Revision}, http.StatusConflict)
		j.assertUnchanged(t, before)
	}) {
		return
	}

	t.Run("peer cannot inject API tiers alongside roles", func(t *testing.T) {
		draft := xaPeerDraft(t, j.want)
		draft.Roles = append(draft.Roles, nodepolicy.Role{Name: "xa-refused", Profiles: []string{"joint-local"}})
		draft.Tiers["weak"] = []nodepolicy.TierEntry{{Kind: nodepolicy.EntryAPI, Deployment: "free-small"}}
		m := j.originCall(t, "engine:remote-policy-set", map[string]any{"policy": draft, "expectedPersistedRevision": j.want.Revision})
		// The peer projection has no deployment definitions, so the target
		// refuses the injected API entry before merging the role edit. The
		// relay preserves that diagnostic without a transport prefix.
		if m.Error == nil || m.Error.Code != -32602 || m.Error.Message != `tiers.weak[0]: deployment "free-small" is not defined` || len(m.Result) != 0 {
			t.Fatalf("protected tier refusal mismatch: error=%+v result_bytes=%d", m.Error, len(m.Result))
		}
		j.assertPolicy(t)
	})
}

func TestXARemoteConfigurationServing(t *testing.T) {
	t.Run("declarations", func(t *testing.T) {
		xaPlatform(t)
		f := newManagementFixture(t, jointAvailable)
		j := f.joint
		setDeclaration := func(t *testing.T, draft nodepolicy.PeerPolicy) {
			t.Helper()
			res := j.rpc(t, "engine:remote-policy-set", map[string]any{"policy": draft, "expectedPersistedRevision": j.want.Revision}).Result
			var accepted struct {
				Policy nodepolicy.PeerPolicy `json:"policy"`
			}
			if err := json.Unmarshal(res, &accepted); err != nil {
				t.Fatalf("decode accepted model declaration: %v", err)
			}
			if accepted.Policy.Revision == "" || accepted.Policy.Revision == j.want.Revision {
				t.Fatal("accepted model declaration lacks a new revision")
			}
			if !reflect.DeepEqual(accepted.Policy.ServedModels, draft.ServedModels) || !reflect.DeepEqual(accepted.Policy.ResidentModels, draft.ResidentModels) || !reflect.DeepEqual(accepted.Policy.Profiles, draft.Profiles) {
				t.Fatal("accepted model declaration differs from the requested served set, resident set or profiles")
			}
			j.want.Profiles = draft.Profiles
			j.want.ServedModels = draft.ServedModels
			j.want.ResidentModels = draft.ResidentModels
			j.want.Revision = accepted.Policy.Revision
			j.assertPolicy(t)
		}

		if !t.Run("explicit empty served set persists", func(t *testing.T) {
			draft := xaPeerDraft(t, j.want)
			draft.ServedModels = nodepolicy.ServedModels{}
			setDeclaration(t, draft)
			var stored map[string]json.RawMessage
			if err := json.Unmarshal(readPolicyBytes(t, j.path), &stored); err != nil {
				t.Fatalf("decode stored policy fields: %v", err)
			}
			raw, present := stored["servedModels"]
			if !present {
				t.Fatal("stored policy omits explicit empty servedModels")
			}
			var served []json.RawMessage
			if err := json.Unmarshal(raw, &served); err != nil {
				t.Fatalf("decode stored servedModels array: %v", err)
			}
			if served == nil || len(served) != 0 {
				t.Fatal("stored servedModels must be an explicit empty array")
			}
		}) {
			return
		}

		if !t.Run("served subset with resident declaration", func(t *testing.T) {
			draft := xaPeerDraft(t, j.want)
			draft.Profiles = []nodepolicy.Profile{{
				Key: "joint-local", DisplayName: "joint-local", Engine: "llamacpp", Model: "joint-local",
				ContextTokens: 8192, MaxOutputTokens: 1024, MaxConcurrent: 1, MemoryBytes: 1,
				Qualification: &nodepolicy.QualificationRef{ID: "xa-qual", Revision: "1"},
			}}
			draft.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "joint-local"}}
			draft.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "joint-local"}}
			setDeclaration(t, draft)
		}) {
			return
		}

		if !t.Run("target advertises the served set", func(t *testing.T) {
			task1Eventually(t, "served set projection", 10*time.Second, func() bool {
				adv := xaEngineModelsPolicy(t, j.target)
				return adv != nil && adv.Revision == j.want.Revision && reflect.DeepEqual(adv.ServedModels, j.want.ServedModels)
			})
		}) {
			return
		}

		if !t.Run("resident outside served is refused", func(t *testing.T) {
			draft := xaPeerDraft(t, j.want)
			draft.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "other"}}
			if !reflect.DeepEqual(draft.ResidentModels, nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "joint-local"}}) {
				t.Fatal("rejected draft lost its joint-local resident declaration")
			}
			m := j.originCall(t, "engine:remote-policy-set", map[string]any{"policy": draft, "expectedPersistedRevision": j.want.Revision})
			if m.Error == nil || m.Error.Code != -32602 || m.Error.Message != "residentModels must be a subset of servedModels" || len(m.Result) != 0 {
				t.Fatalf("resident subset refusal mismatch: error=%+v result_bytes=%d", m.Error, len(m.Result))
			}
			j.assertPolicy(t)
		}) {
			return
		}

		if !t.Run("denied peer cannot change the served set", func(t *testing.T) {
			before := readPolicyBytes(t, j.path)
			draft := xaPeerDraft(t, j.want)
			draft.ServedModels = nodepolicy.ServedModels{}
			draft.ResidentModels = nil
			managementHTTP(t, f.denied, j.ports[1], http.MethodPost, "/v1/node-policy/set", map[string]any{"policy": draft, "expectedPersistedRevision": j.want.Revision}, http.StatusConflict)
			j.assertUnchanged(t, before)
		}) {
			return
		}

		t.Run("absent served set restores unrestricted declaration", func(t *testing.T) {
			draft := xaPeerDraft(t, j.want)
			draft.ServedModels = nil
			draft.ResidentModels = nil
			setDeclaration(t, draft)
			var stored map[string]json.RawMessage
			if err := json.Unmarshal(readPolicyBytes(t, j.path), &stored); err != nil {
				t.Fatalf("decode unrestricted stored policy fields: %v", err)
			}
			if _, present := stored["servedModels"]; present {
				t.Fatal("unrestricted stored policy still contains servedModels")
			}
			var remote struct {
				Policy map[string]json.RawMessage `json:"policy"`
			}
			res := j.rpc(t, "engine:remote-policy-get", map[string]any{}).Result
			if err := json.Unmarshal(res, &remote); err != nil {
				t.Fatalf("decode unrestricted peer policy fields: %v", err)
			}
			if remote.Policy == nil {
				t.Fatal("unrestricted peer read omitted its policy")
			}
			if _, present := remote.Policy["servedModels"]; present {
				t.Fatal("unrestricted peer policy still contains servedModels")
			}
			task1Eventually(t, "unrestricted served projection", 10*time.Second, func() bool {
				adv := xaEngineModelsPolicy(t, j.target)
				return adv != nil && adv.Revision == j.want.Revision && adv.ServedModels == nil
			})
		})
	})

	t.Run("proxy enforces served set at its facade", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			xaPlatform(t)
		}
		s := newEngineProcessFixture(t)
		s.unblock()
		// Every installed synthetic engine starts on an available node. Routing
		// this profile must leave the other engine's startup lifetime unchanged.
		otherPID := s.pid(t, "llamacpp")
		s.rpcOK(t, "llamacpp-custom-proxy:node/add-manual", map[string]any{"id": "xa-self", "host": "127.0.0.1", "port": s.facadePort, "addresses": []string{"127.0.0.1"}, "models": []string{"switch-model"}})
		s.rpcOK(t, "llamacpp-custom-proxy:node/select", map[string]string{"id": "xa-self"})
		client := &http.Client{Timeout: 30 * time.Second}
		t.Cleanup(client.CloseIdleConnections)
		generate := func() (*http.Response, []byte) {
			t.Helper()
			resp, err := client.Post("http://127.0.0.1:"+strconv.Itoa(s.facadePort)+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"switch-model","stream":false,"messages":[{"role":"user","content":"fixture"}]}`))
			if err != nil {
				t.Fatalf("facade generation request: %v", err)
			}
			body, readErr := io.ReadAll(io.LimitReader(resp.Body, (64<<10)+1))
			closeErr := resp.Body.Close()
			if readErr != nil {
				t.Fatalf("read facade generation response: %v", readErr)
			}
			if closeErr != nil {
				t.Fatalf("close facade generation response: %v", closeErr)
			}
			if len(body) > 64<<10 {
				t.Fatal("facade generation response exceeds 64 KiB")
			}
			return resp, body
		}
		setServed := func(served nodepolicy.ServedModels) {
			t.Helper()
			var current struct {
				Policy nodepolicy.Policy `json:"policy"`
			}
			if err := json.Unmarshal(s.rpcOK(t, "policy:get", map[string]any{}), &current); err != nil {
				t.Fatalf("decode current facade policy: %v", err)
			}
			p := current.Policy
			p.Profiles = []nodepolicy.Profile{{Key: "switch-model", DisplayName: "Serving fixture", Engine: "llamacpp-custom", Model: "switch-model"}}
			p.ServedModels = served
			var accepted struct {
				Policy nodepolicy.Policy `json:"policy"`
			}
			res := s.rpcOK(t, "policy:set", map[string]any{"policy": p, "expectedPersistedRevision": current.Policy.Revision})
			if err := json.Unmarshal(res, &accepted); err != nil {
				t.Fatalf("decode accepted facade policy: %v", err)
			}
			if accepted.Policy.Revision == "" || !reflect.DeepEqual(accepted.Policy.ServedModels, served) {
				t.Fatal("accepted facade policy lacks a revision or differs from the requested served set")
			}
			task1Eventually(t, "policy activation", 15*time.Second, func() bool {
				var view struct {
					Policy         nodepolicy.Policy `json:"policy"`
					ActiveRevision string            `json:"activeRevision"`
					Activation     string            `json:"activation"`
				}
				if err := json.Unmarshal(s.rpcOK(t, "policy:get", map[string]any{}), &view); err != nil {
					t.Fatalf("decode facade policy activation: %v", err)
				}
				return view.Activation == "active" && view.ActiveRevision == accepted.Policy.Revision && view.Policy.Revision == accepted.Policy.Revision
			})
		}
		inferenceHits := func() int {
			s.mu.Lock()
			defer s.mu.Unlock()
			return s.inferences["llamacpp-custom"]
		}
		wantInferenceHits := 0
		assertDone := func(resp *http.Response, body []byte) {
			t.Helper()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("facade generation status=%d, want %d", resp.StatusCode, http.StatusOK)
			}
			var result struct {
				Model   string `json:"model"`
				Choices []struct {
					FinishReason string `json:"finish_reason"`
				} `json:"choices"`
			}
			if err := json.Unmarshal(body, &result); err != nil {
				t.Fatalf("decode facade generation completion: %v", err)
			}
			if result.Model != "switch-model" || len(result.Choices) != 1 || result.Choices[0].FinishReason != "stop" {
				t.Fatal("facade generation did not complete")
			}
			wantInferenceHits++
			if got := inferenceHits(); got != wantInferenceHits {
				t.Fatalf("native inference hits=%d, want %d", got, wantInferenceHits)
			}
		}
		assertNotServed := func(resp *http.Response, body []byte) {
			t.Helper()
			// The origin filters the self candidate by its profile's served model
			// before destination admission. A visible but unserved profile has no
			// eligible node; it must never execute in the synthetic engine.
			if resp.StatusCode != http.StatusBadGateway {
				t.Fatalf("facade refusal status=%d, want %d", resp.StatusCode, http.StatusBadGateway)
			}
			var refusal struct {
				Error string `json:"error"`
			}
			if err := json.Unmarshal(body, &refusal); err != nil {
				t.Fatalf("decode facade admission refusal: %v", err)
			}
			if refusal.Error != "no available node advertises the requested model" || inferenceHits() != wantInferenceHits {
				t.Fatal("unserved profile was not refused before native execution")
			}
		}

		setServed(nil)
		// Catalogs expose configured profiles without consulting native inventory.
		task1Eventually(t, "facade self model inventory", 10*time.Second, func() bool {
			resp, err := client.Get("http://127.0.0.1:" + strconv.Itoa(s.facadePort) + "/v1/models")
			if err != nil {
				t.Fatalf("facade inventory request: %v", err)
			}
			body, readErr := io.ReadAll(io.LimitReader(resp.Body, (64<<10)+1))
			closeErr := resp.Body.Close()
			if readErr != nil || closeErr != nil || len(body) > 64<<10 {
				t.Fatalf("facade inventory response: read_error=%t close_error=%t exceeds_limit=%t", readErr != nil, closeErr != nil, len(body) > 64<<10)
			}
			var inventory struct {
				Models []struct {
					Name string `json:"id"`
				} `json:"data"`
				Error string `json:"error"`
			}
			if err := json.Unmarshal(body, &inventory); err != nil {
				t.Fatalf("decode facade inventory: %v", err)
			}
			if resp.StatusCode == http.StatusServiceUnavailable && inventory.Error == "model inventory unavailable" {
				return false
			}
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("facade inventory status=%d, want %d", resp.StatusCode, http.StatusOK)
			}
			for _, model := range inventory.Models {
				if model.Name == "switch-model" {
					return true
				}
			}
			return false
		})

		task1Eventually(t, "healthy engine answers the configured profile", 15*time.Second, func() bool {
			resp, body := generate()
			if resp.StatusCode == http.StatusBadGateway || resp.StatusCode == http.StatusServiceUnavailable {
				return false
			}
			assertDone(resp, body)
			return true
		})
		setServed(nodepolicy.ServedModels{})
		assertNotServed(generate())
		setServed(nodepolicy.ServedModels{{Engine: "llamacpp-custom", Model: "other"}})
		assertNotServed(generate())
		setServed(nodepolicy.ServedModels{{Engine: "llamacpp-custom", Model: "switch-model"}})
		assertDone(generate())
		s.pid(t, "llamacpp-custom")
		if got := s.pid(t, "llamacpp"); got != otherPID {
			t.Fatalf("profile routing replaced the other engine: pid=%d, want %d", got, otherPID)
		}
	})
}
