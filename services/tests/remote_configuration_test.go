// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
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
