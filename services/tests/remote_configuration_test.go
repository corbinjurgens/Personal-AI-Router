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

	"nvpair-shared/engineintent"
	settings "nvpair-shared/enginesettings"
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

	t.Run("switch status", func(t *testing.T) {
		m := f.originCall(t, settings.RemoteSwitchStatusMethod, map[string]any{})
		if m.Error != nil {
			t.Fatalf("remote switch status failed: code=%d", m.Error.Code)
		}
		var remote settings.SwitchStatus
		if err := json.Unmarshal(m.Result, &remote); err != nil {
			t.Fatalf("decode remote switch status: %v", err)
		}
		m = callRPC(t, f.target.in, f.target.msgs, settings.SwitchStatusMethod, map[string]any{}, 10*time.Second)
		if m.Error != nil {
			t.Fatalf("local switch status failed: code=%d", m.Error.Code)
		}
		var local settings.SwitchStatus
		if err := json.Unmarshal(m.Result, &local); err != nil {
			t.Fatalf("decode local switch status: %v", err)
		}
		if remote.ConfigurationRevision == "" || remote.ConfigurationRevision != local.ConfigurationRevision {
			t.Fatalf("switch configuration revision: remote=%q local=%q, want equal nonempty revisions", remote.ConfigurationRevision, local.ConfigurationRevision)
		}
		if remote.ActiveRequestID != "" || remote.Receipt != nil {
			t.Fatalf("remote switch status: active_request_present=%t receipt_present=%t, want neither", remote.ActiveRequestID != "", remote.Receipt != nil)
		}
	})

	t.Run("engine intent", func(t *testing.T) {
		c := subscribeOBIntent(t, f.target)
		assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), obIntent(false, map[string]engineintent.Origin{}))
	})
}

func TestXARemoteConfigurationConflict(t *testing.T) {
	xaPlatform(t)
	f := newPolicyJointFixture(t)
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
		const message = "remote node policy: HTTP 409: this node was paused on the node itself; resume it there"
		if m.Error == nil || m.Error.Code != -32011 || m.Error.Message != message || len(m.Result) != 0 {
			t.Fatalf("relayed resume veto mismatch: error_present=%t result_bytes=%d", m.Error != nil, len(m.Result))
		}
		f.assertUnchanged(t, before)
	}) {
		return
	}

	if !t.Run("remote start is vetoed while paused", func(t *testing.T) {
		task1Eventually(t, "remote start refused by local pause", 15*time.Second, func() bool {
			m := callRPC(t, f.origin.in, f.origin.msgs, "engine:remote-start", map[string]any{"node": "policy-joint-target", "engine": "ollama"}, 30*time.Second)
			return m.Error != nil && m.Error.Code == -32011
		})
	}) {
		return
	}

	t.Run("local resume then local Off vetoes remote start", func(t *testing.T) {
		local := callRPC(t, f.target.in, f.target.msgs, "node:set-availability", map[string]any{"state": "available"}, 15*time.Second)
		if local.Error != nil {
			t.Fatalf("local resume failed: code=%d", local.Error.Code)
		}
		c := subscribeOBIntent(t, f.target)
		c.call(t, "engine:stop", map[string]string{"engine": "ollama"})
		want := obIntent(false, map[string]engineintent.Origin{"ollama": engineintent.Local})
		c.assertIntent(t, want)
		task1Eventually(t, "remote start refused by local Off", 15*time.Second, func() bool {
			m := callRPC(t, f.origin.in, f.origin.msgs, "engine:remote-start", map[string]any{"node": "policy-joint-target", "engine": "ollama"}, 30*time.Second)
			return m.Error != nil && m.Error.Code == -32011
		})
		assertOBIntent(t, decodeOBIntent(t, c.call(t, "engine:intent", nil)), want)
		assertOBStoredIntent(t, filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json"), false, engineintent.Local)
	})
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
	f := newManagementFixture(t)
	j := f.joint

	if !t.Run("allowed peer sets roles", func(t *testing.T) {
		draft := xaPeerDraft(t, j.want)
		draft.Roles = map[string]nodepolicy.Role{"weak": {Engine: "ollama", Model: "joint-local"}}
		res := j.rpc(t, "engine:remote-policy-set", map[string]any{"policy": draft, "expectedPersistedRevision": j.want.Revision}).Result
		var accepted struct {
			Policy nodepolicy.PeerPolicy `json:"policy"`
		}
		if err := json.Unmarshal(res, &accepted); err != nil {
			t.Fatalf("decode accepted roles edit: %v", err)
		}
		if !reflect.DeepEqual(accepted.Policy.Roles, draft.Roles) || accepted.Policy.Revision == "" || accepted.Policy.Revision == j.want.Revision {
			t.Fatal("accepted roles edit differs from the requested roles or lacks a new revision")
		}
		j.want.Roles = draft.Roles
		j.want.Revision = accepted.Policy.Revision
		j.assertPolicy(t)
	}) {
		return
	}

	if !t.Run("target advertises roles", func(t *testing.T) {
		task1Eventually(t, "roles projection", 10*time.Second, func() bool {
			adv := xaEngineModelsPolicy(t, j.target)
			return adv != nil && adv.Revision == j.want.Revision && adv.Roles["weak"].Engine == "ollama" && adv.Roles["weak"].Model == "joint-local"
		})
	}) {
		return
	}

	if !t.Run("denied peer cannot change roles", func(t *testing.T) {
		before := readPolicyBytes(t, j.path)
		draft := xaPeerDraft(t, j.want)
		draft.Roles = map[string]nodepolicy.Role{"strong": {Engine: "ollama", Model: "joint-local"}}
		managementHTTP(t, f.denied, j.ports[1], http.MethodPost, "/v1/node-policy/set", map[string]any{"policy": draft, "expectedPersistedRevision": j.want.Revision}, http.StatusConflict)
		j.assertUnchanged(t, before)
	}) {
		return
	}

	t.Run("peer cannot change tiers alongside roles", func(t *testing.T) {
		draft := xaPeerDraft(t, j.want)
		draft.Tiers["weak"][0].Model = "different-local"
		m := j.originCall(t, "engine:remote-policy-set", map[string]any{"policy": draft, "expectedPersistedRevision": j.want.Revision})
		if m.Error == nil || m.Error.Code != -32602 || m.Error.Message != "remote node policy: HTTP 400: a paired node cannot change the tiers" || len(m.Result) != 0 {
			t.Fatalf("protected tier refusal mismatch: error_present=%t result_bytes=%d", m.Error != nil, len(m.Result))
		}
		j.assertPolicy(t)
	})
}

func TestXARemoteConfigurationServing(t *testing.T) {
	xaPlatform(t)

	t.Run("declarations", func(t *testing.T) {
		f := newManagementFixture(t)
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
			draft.Profiles = append(draft.Profiles, nodepolicy.Profile{
				Name: "xa-resident", Engine: "ollama", Model: "joint-local",
				ContextTokens: 8192, MaxOutputTokens: 1024, MaxConcurrent: 1, MemoryBytes: 1,
				Qualification: &nodepolicy.QualificationRef{ID: "xa-qual", Revision: "1"},
			})
			draft.ServedModels = nodepolicy.ServedModels{{Engine: "ollama", Model: "joint-local"}}
			draft.ResidentModels = nodepolicy.ResidentModels{{Engine: "ollama", Model: "joint-local"}}
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
			draft.ServedModels = nodepolicy.ServedModels{{Engine: "ollama", Model: "other"}}
			if !reflect.DeepEqual(draft.ResidentModels, nodepolicy.ResidentModels{{Engine: "ollama", Model: "joint-local"}}) {
				t.Fatal("rejected draft lost its joint-local resident declaration")
			}
			m := j.originCall(t, "engine:remote-policy-set", map[string]any{"policy": draft, "expectedPersistedRevision": j.want.Revision})
			if m.Error == nil || m.Error.Code != -32602 || m.Error.Message != "remote node policy: HTTP 400: residentModels must be a subset of servedModels" || len(m.Result) != 0 {
				t.Fatalf("resident subset refusal mismatch: error_present=%t result_bytes=%d", m.Error != nil, len(m.Result))
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
		s := newSwitchProcessFixture(t)
		s.unblock()
		s.rpcOK(t, "ollama-proxy:node/add-manual", map[string]any{"id": "xa-self", "host": "127.0.0.1", "port": s.facadePort, "addresses": []string{"127.0.0.1"}, "models": []string{"switch-model"}})
		s.rpcOK(t, "ollama-proxy:node/select", map[string]string{"id": "xa-self"})
		client := &http.Client{Timeout: 30 * time.Second}
		t.Cleanup(client.CloseIdleConnections)
		generate := func() (*http.Response, []byte) {
			t.Helper()
			resp, err := client.Post("http://127.0.0.1:"+strconv.Itoa(s.facadePort)+"/api/generate", "application/json", strings.NewReader(`{"model":"switch-model","stream":false,"prompt":"fixture"}`))
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
		assertDone := func(resp *http.Response, body []byte) {
			t.Helper()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("facade generation status=%d, want %d", resp.StatusCode, http.StatusOK)
			}
			var result struct {
				Done bool `json:"done"`
			}
			if err := json.Unmarshal(body, &result); err != nil {
				t.Fatalf("decode facade generation completion: %v", err)
			}
			if !result.Done {
				t.Fatal("facade generation did not complete")
			}
		}
		assertNotServed := func(resp *http.Response, body []byte) {
			t.Helper()
			if resp.StatusCode != http.StatusServiceUnavailable || resp.Header.Get(nodepolicy.AdmissionHeader) != "not-served" || resp.Header.Get("Retry-After") == "" {
				t.Fatalf("facade refusal status=%d admission=%q retry_after_present=%t", resp.StatusCode, resp.Header.Get(nodepolicy.AdmissionHeader), resp.Header.Get("Retry-After") != "")
			}
			var refusal struct {
				Error  string `json:"error"`
				Code   string `json:"code"`
				Reason string `json:"reason"`
			}
			if err := json.Unmarshal(body, &refusal); err != nil {
				t.Fatalf("decode facade admission refusal: %v", err)
			}
			if refusal.Error != "this node is not accepting the request (not-served)" || refusal.Code != "admission" || refusal.Reason != "not-served" {
				t.Fatal("facade refusal differs from the specified admission error, code or reason")
			}
		}

		// Engine start completes before the broker's advertisement poll supplies
		// the proxy's healthy local endpoint. Wait through the facade itself.
		task1Eventually(t, "facade self model inventory", 10*time.Second, func() bool {
			resp, err := client.Get("http://127.0.0.1:" + strconv.Itoa(s.facadePort) + "/api/tags")
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
					Name string `json:"name"`
				} `json:"models"`
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

		setServed(nil)
		assertDone(generate())
		setServed(nodepolicy.ServedModels{})
		assertNotServed(generate())
		setServed(nodepolicy.ServedModels{{Engine: "ollama", Model: "other"}})
		assertNotServed(generate())
		setServed(nodepolicy.ServedModels{{Engine: "ollama", Model: "switch-model"}})
		assertDone(generate())
		s.pid(t, "ollama")
		s.mu.Lock()
		targetStarts := len(s.starts["llamacpp"])
		s.mu.Unlock()
		if targetStarts != 0 {
			t.Fatalf("llamacpp starts=%d, want none", targetStarts)
		}
	})
}
