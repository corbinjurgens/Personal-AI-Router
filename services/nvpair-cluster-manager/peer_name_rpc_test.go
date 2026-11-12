// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"nvpair-shared/clustertrust"
)

// setPeerNameFixture pairs a viewer and a target over real mTLS listeners.
// The viewer answers the target's post-rename roster push with 500 so no
// background write lands after the test returns.
func setPeerNameFixture(t *testing.T) (viewer, target *clusterPeerFixture) {
	t.Helper()
	viewer, target = newClusterPeerFixture(t, false), newClusterPeerFixture(t, false)
	pairClusterFixtures(t, viewer, target)
	for _, f := range []*clusterPeerFixture{viewer, target} {
		if err := f.m.persistMembersErr(); err != nil {
			t.Fatal(err)
		}
	}
	viewer.rosterReply = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}
	return viewer, target
}

// callSetPeerName sends cluster:set-peer-name through the dispatch, which
// answers from its own goroutine, and waits for the response.
func callSetPeerName(t *testing.T, m *Manager, uuid string, name any) rpcResult {
	t.Helper()
	params, err := json.Marshal(map[string]any{"nodeUuid": uuid, "name": name})
	if err != nil {
		t.Fatal(err)
	}
	out := captureCodec(m)
	id := json.RawMessage(`11`)
	m.handleMessage(&Message{ID: &id, Method: "cluster:set-peer-name", Params: params})
	deadline := time.Now().Add(2 * pairingHTTPTimeout)
	for time.Now().Before(deadline) {
		var res rpcResult
		found := false
		for _, frame := range out.lines(t) {
			if raw, ok := frame["method"]; ok {
				var method string
				_ = json.Unmarshal(raw, &method)
				res.notified = append(res.notified, method)
				continue
			}
			if string(frame["id"]) != "11" {
				continue
			}
			found = true
			res.result = frame["result"]
			if raw, ok := frame["error"]; ok {
				if err := json.Unmarshal(raw, &res.err); err != nil {
					t.Fatal(err)
				}
			}
		}
		if found {
			return res
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("no response to cluster:set-peer-name")
	return rpcResult{}
}

func TestSetPeerNameAllowedRenamesTarget(t *testing.T) {
	viewer, target := setPeerNameFixture(t)
	setTestGrant(t, target.m, viewer.m.identity.NodeUUID, true)
	targetUUID := target.m.identity.NodeUUID

	res := callSetPeerName(t, viewer.m, targetUUID, "Garage PC")
	if res.err != nil {
		t.Fatalf("rename error %+v", res.err)
	}
	var view nodeNameView
	if err := json.Unmarshal(res.result, &view); err != nil {
		t.Fatal(err)
	}
	if view.NodeUUID != targetUUID || view.Name != "Garage PC" || view.CustomName == nil || *view.CustomName != "Garage PC" {
		t.Fatalf("result = %+v, want the target's renamed view", view)
	}
	if got := readIdentityFile(t, target.m).DisplayName; got != "Garage PC" {
		t.Fatalf("target identity.json display_name = %q, want Garage PC", got)
	}
	if n, ok := viewer.m.memberByUUID(targetUUID); !ok || n.Name != "Garage PC" {
		t.Fatalf("viewer member = %+v, want name Garage PC", n)
	}
	if pin, ok := viewer.m.trust.Get(targetUUID); !ok || pin.Name != "Garage PC" {
		t.Fatalf("viewer pin = %+v, want name Garage PC", pin)
	}
	// The pin rename also announces cluster:trust-changed through the trust
	// store's change hook.
	nodesChanged := 0
	for _, method := range res.notified {
		if method == "nodes:changed" {
			nodesChanged++
		}
	}
	if nodesChanged != 1 {
		t.Fatalf("viewer notifications = %v, want one nodes:changed", res.notified)
	}
}

func TestSetPeerNameNullResetsTargetToDefault(t *testing.T) {
	viewer, target := setPeerNameFixture(t)
	setTestGrant(t, target.m, viewer.m.identity.NodeUUID, true)
	targetUUID := target.m.identity.NodeUUID
	if res := callSetPeerName(t, viewer.m, targetUUID, "Temp"); res.err != nil {
		t.Fatalf("set error %+v", res.err)
	}
	res := callSetPeerName(t, viewer.m, targetUUID, nil)
	if res.err != nil {
		t.Fatalf("clear error %+v", res.err)
	}
	want := defaultNodeName(targetUUID)
	var view nodeNameView
	if err := json.Unmarshal(res.result, &view); err != nil {
		t.Fatal(err)
	}
	if view.CustomName != nil || view.Name != want || target.m.selfName() != want {
		t.Fatalf("result = %+v, target name %q, want default %q", view, target.m.selfName(), want)
	}
	if n, _ := viewer.m.memberByUUID(targetUUID); n.Name != want {
		t.Fatalf("viewer member name = %q, want %q", n.Name, want)
	}
}

func TestSetPeerNameDeniedReturnsDenialText(t *testing.T) {
	viewer, target := setPeerNameFixture(t)
	setTestGrant(t, target.m, viewer.m.identity.NodeUUID, false)
	targetUUID := target.m.identity.NodeUUID
	before, _ := viewer.m.memberByUUID(targetUUID)

	res := callSetPeerName(t, viewer.m, targetUUID, "Hijacked")
	if res.err == nil || res.err.Code != codeRemoteManagementDenied ||
		res.err.Message != "Remote management is blocked on this device. Change permission locally on that device." {
		t.Fatalf("response = %+v, want -32012 with the denial text", res.err)
	}
	if target.m.customSelfName() != nil {
		t.Fatal("denied rename changed the target")
	}
	if after, _ := viewer.m.memberByUUID(targetUUID); after.Name != before.Name {
		t.Fatalf("viewer member name changed to %q on a denial", after.Name)
	}
}

func TestSetPeerNameInvalidNameReturnsTargetReason(t *testing.T) {
	viewer, target := setPeerNameFixture(t)
	setTestGrant(t, target.m, viewer.m.identity.NodeUUID, true)
	res := callSetPeerName(t, viewer.m, target.m.identity.NodeUUID, "bad\u0007name")
	if res.err == nil || res.err.Code != codeInvalidParams || res.err.Data["reason"] != "invalid-character" {
		t.Fatalf("response = %+v, want -32602 with reason invalid-character", res.err)
	}
}

func TestSetPeerNameUnreachableTarget(t *testing.T) {
	viewer, target := setPeerNameFixture(t)
	setTestGrant(t, target.m, viewer.m.identity.NodeUUID, true)
	if err := target.listener.Close(); err != nil {
		t.Fatal(err)
	}
	res := callSetPeerName(t, viewer.m, target.m.identity.NodeUUID, "Offline")
	if res.err == nil || res.err.Code != codePeerUnavailable ||
		res.err.Message != "the device is offline or does not trust this node" {
		t.Fatalf("response = %+v, want -32000", res.err)
	}
	if target.m.customSelfName() != nil {
		t.Fatal("unreachable target was renamed")
	}
}

func TestSetPeerNameRejectsSelfAndNonMembers(t *testing.T) {
	viewer, _ := setPeerNameFixture(t)
	for _, tt := range []struct{ name, uuid string }{
		{"self", viewer.m.identity.NodeUUID},
		{"non-member", "66666666-0000-4000-8000-000000000000"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			res := callSetPeerName(t, viewer.m, tt.uuid, "x")
			if res.err == nil || res.err.Code != codeInvalidParams || res.err.Data["field"] != "nodeUuid" {
				t.Fatalf("response = %+v, want -32602 for nodeUuid", res.err)
			}
		})
	}
}

// answerPeerRenameError maps a synthetic target answer and returns the error.
func answerPeerRenameError(t *testing.T, status int, body string) *rpcResult {
	t.Helper()
	m := newTestManager(t)
	out := captureCodec(m)
	id := json.RawMessage(`21`)
	m.answerPeerRename(&Message{ID: &id}, "77777777-0000-4000-8000-000000000000", status, []byte(body))
	for _, frame := range out.lines(t) {
		if string(frame["id"]) != "21" {
			continue
		}
		var res rpcResult
		if err := json.Unmarshal(frame["error"], &res.err); err != nil || res.err == nil {
			t.Fatalf("no error in %v", frame)
		}
		return &res
	}
	t.Fatal("no response")
	return nil
}

func TestSetPeerNameBoundsTargetText(t *testing.T) {
	huge := strings.Repeat("é", 4096) // two bytes per rune
	body, err := json.Marshal(map[string]string{"code": "invalid_name", "reason": strings.Repeat("r", 500), "error": huge})
	if err != nil {
		t.Fatal(err)
	}
	t.Run("400 message and reason are capped on a rune boundary", func(t *testing.T) {
		res := answerPeerRenameError(t, http.StatusBadRequest, string(body))
		if res.err.Code != codeInvalidParams || len(res.err.Message) > peerMessageMaxBytes ||
			!utf8.ValidString(res.err.Message) || !strings.HasPrefix(huge, res.err.Message) || res.err.Message == "" {
			t.Fatalf("message length %d, valid %t; want a non-empty valid prefix of at most %d bytes",
				len(res.err.Message), utf8.ValidString(res.err.Message), peerMessageMaxBytes)
		}
		if got := res.err.Data["reason"]; len(got) != peerReasonMaxBytes {
			t.Fatalf("reason length %d, want %d", len(got), peerReasonMaxBytes)
		}
	})
	t.Run("409 uses the fixed local denial text", func(t *testing.T) {
		res := answerPeerRenameError(t, http.StatusConflict, `{"code":"remote_management_denied","error":"`+strings.Repeat("x", 10000)+`"}`)
		if res.err.Code != codeRemoteManagementDenied || res.err.Message != clustertrust.RemoteManagementDeniedMessage {
			t.Fatalf("denial = %d %q, want -32012 with the fixed text", res.err.Code, res.err.Message)
		}
	})
}
