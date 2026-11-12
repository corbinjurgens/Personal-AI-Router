// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"encoding/pem"
	"os"
	"strings"
	"testing"
)

// nicknameFixture is a clustered manager with one pinned peer member named
// "peer-own".
func nicknameFixture(t *testing.T) (*Manager, string, string, string) {
	t.Helper()
	m := newTestManager(t)
	uuid, certPEM, fp, _ := makeNode(t, "peer")
	pinTrusted(t, m, uuid, certPEM, fp)
	m.upsertMember(&ClusterNode{NodeUUID: uuid, Name: "peer-own", ClusterID: "cluster-1", AdmissionEpoch: 1, State: stateMember})
	if err := m.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	return m, uuid, certPEM, fp
}

type rpcResult struct {
	result json.RawMessage
	err    *struct {
		Code    int               `json:"code"`
		Message string            `json:"message"`
		Data    map[string]string `json:"data"`
	}
	notified []string
}

// callRPC dispatches one request through handleMessage and returns its
// response plus the methods of every notification it emitted.
func callRPC(t *testing.T, m *Manager, method, params string) rpcResult {
	t.Helper()
	out := captureCodec(m)
	id := json.RawMessage(`9`)
	m.handleMessage(&Message{ID: &id, Method: method, Params: json.RawMessage(params)})
	var res rpcResult
	found := false
	for _, frame := range out.lines(t) {
		if method, ok := frame["method"]; ok {
			var name string
			_ = json.Unmarshal(method, &name)
			res.notified = append(res.notified, name)
			continue
		}
		if string(frame["id"]) != "9" {
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
	if !found {
		t.Fatalf("no response to %s", method)
	}
	return res
}

func nicknameParams(t *testing.T, uuid string, nickname any) string {
	t.Helper()
	data, err := json.Marshal(map[string]any{"nodeUuid": uuid, "nickname": nickname})
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func decodeNicknameView(t *testing.T, res rpcResult) peerNicknameView {
	t.Helper()
	if res.err != nil {
		t.Fatalf("unexpected error %+v", res.err)
	}
	var view peerNicknameView
	if err := json.Unmarshal(res.result, &view); err != nil {
		t.Fatal(err)
	}
	return view
}

func readMembersFile(t *testing.T, m *Manager) string {
	t.Helper()
	data, err := os.ReadFile(m.membersPath())
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestSetPeerNicknameSetsAndClears(t *testing.T) {
	m, uuid, _, _ := nicknameFixture(t)

	set := callRPC(t, m, "cluster:set-peer-nickname", nicknameParams(t, uuid, "  Lab box  "))
	view := decodeNicknameView(t, set)
	if view.NodeUUID != uuid || view.Nickname == nil || *view.Nickname != "Lab box" || view.DisplayName != "Lab box" {
		t.Fatalf("set view = %+v, want nickname and displayName %q", view, "Lab box")
	}
	if len(set.notified) != 1 || set.notified[0] != "nodes:changed" {
		t.Fatalf("notifications = %v, want one nodes:changed", set.notified)
	}
	if !strings.Contains(readMembersFile(t, m), `"nickname": "Lab box"`) {
		t.Fatal("nickname not persisted in members.json")
	}

	cleared := decodeNicknameView(t, callRPC(t, m, "cluster:set-peer-nickname", nicknameParams(t, uuid, nil)))
	if cleared.Nickname != nil || cleared.DisplayName != "peer-own" {
		t.Fatalf("cleared view = %+v, want nil nickname and displayName peer-own", cleared)
	}
	if strings.Contains(readMembersFile(t, m), "nickname") {
		t.Fatal("cleared nickname still persisted")
	}
	for _, n := range m.snapshotNodes() {
		if n.NodeUUID == uuid && n.DisplayName != "peer-own" {
			t.Fatalf("snapshot displayName = %q, want peer-own", n.DisplayName)
		}
	}
}

func TestSetPeerNicknameRejectsInvalidRequests(t *testing.T) {
	m, uuid, _, _ := nicknameFixture(t)
	stranger := "55555555-0000-4000-8000-000000000000"
	for _, tt := range []struct {
		name, params, wantField, wantReason string
	}{
		{"invalid nickname", nicknameParams(t, uuid, "bad\u0007name"), "nickname", "invalid-character"},
		{"empty nickname", nicknameParams(t, uuid, "   "), "nickname", "empty"},
		{"non-member", nicknameParams(t, stranger, "x"), "nodeUuid", ""},
		{"self", nicknameParams(t, m.identity.NodeUUID, "x"), "nodeUuid", ""},
		{"missing nickname", `{"nodeUuid":"` + uuid + `"}`, "", ""},
		{"unknown field", `{"nodeUuid":"` + uuid + `","nickname":"x","extra":1}`, "", ""},
		{"number nickname", `{"nodeUuid":"` + uuid + `","nickname":3}`, "", ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			before := readMembersFile(t, m)
			res := callRPC(t, m, "cluster:set-peer-nickname", tt.params)
			if res.err == nil || res.err.Code != codeInvalidParams {
				t.Fatalf("response = %+v, want -32602", res)
			}
			if res.err.Data["field"] != tt.wantField || res.err.Data["reason"] != tt.wantReason {
				t.Fatalf("error data = %v, want field %q reason %q", res.err.Data, tt.wantField, tt.wantReason)
			}
			if len(res.notified) != 0 {
				t.Fatalf("notifications = %v, want none", res.notified)
			}
			if after := readMembersFile(t, m); after != before {
				t.Fatal("rejected request changed members.json")
			}
		})
	}
}

func TestSetPeerNicknameSurvivesRestart(t *testing.T) {
	m, uuid, _, _ := nicknameFixture(t)
	decodeNicknameView(t, callRPC(t, m, "cluster:set-peer-nickname", nicknameParams(t, uuid, "Kept")))
	reopened := reopenManager(t, m)
	n, ok := reopened.memberByUUID(uuid)
	if !ok || n.Nickname != "Kept" {
		t.Fatalf("member after restart = %+v, want nickname Kept", n)
	}
}

func TestSetPeerNicknameSurvivesRepin(t *testing.T) {
	m, uuid, certPEM, fp := nicknameFixture(t)
	decodeNicknameView(t, callRPC(t, m, "cluster:set-peer-nickname", nicknameParams(t, uuid, "Kept")))
	block, _ := pem.Decode([]byte(certPEM))
	if err := m.pinEntry(&RosterEntry{
		NodeUUID: uuid, Name: "peer-renamed", AdmissionEpoch: 2, CertPem: certPEM, CertFingerprint: fp,
	}, block.Bytes); err != nil {
		t.Fatal(err)
	}
	n, ok := m.memberByUUID(uuid)
	if !ok || n.Nickname != "Kept" || n.Name != "peer-renamed" || n.AdmissionEpoch != 2 {
		t.Fatalf("member after repin = %+v, want nickname kept with new name and epoch", n)
	}
}

func TestSetPeerNicknameNeverEntersRoster(t *testing.T) {
	m, uuid, _, _ := nicknameFixture(t)
	decodeNicknameView(t, callRPC(t, m, "cluster:set-peer-nickname", nicknameParams(t, uuid, "Private nickname")))
	data, err := json.Marshal(m.buildLocalRoster())
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "Private nickname") {
		t.Fatal("nickname leaked into the roster sent to peers")
	}
}
