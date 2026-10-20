// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"encoding/json"
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
