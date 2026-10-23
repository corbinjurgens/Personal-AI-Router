// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"nvpair-tui/rpc"

	tea "github.com/charmbracelet/bubbletea"
)

const permissionDeniedText = "Remote management is blocked on this device. Change permission locally on that device."

func TestRemoteManagementCommandFailure(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want string
	}{
		{"permission", &rpc.RPCError{Code: -32012, Message: permissionDeniedText}, "remote management denied (-32012): " + permissionDeniedText},
		{"local veto", &rpc.RPCError{Code: -32011, Message: "engine is Off locally"}, "local veto (-32011): engine is Off locally"},
		{"conflict", &rpc.RPCError{Code: -32010, Message: "changed"}, "revision conflict (-32010): changed - reload and retry"},
		{"unreachable", errors.New("device unreachable"), "device unreachable"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := commandFailure(tc.err); got != tc.want {
				t.Fatalf("commandFailure = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestLocalManagementCASAdoptsReplyWithoutOptimism(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	v := newNodesView(client)
	v.adoptRemoteManagement(remoteManagementSnapshot{Version: 1, Revision: "9007199254740993", Grants: []remoteManagementGrant{{PeerUUID: "peer", State: "denied"}}})
	cmd := v.setRemoteManagement(v.remoteManagement.Grants[0])
	if v.remoteManagement.Grants[0].Allowed || v.remoteManagement.Grants[0].State != "denied" {
		t.Fatal("permission flipped before reply")
	}
	if !v.remoteManagementSetting {
		t.Fatal("write was not marked pending")
	}
	if duplicate := v.setRemoteManagement(v.remoteManagement.Grants[0]); duplicate != nil {
		t.Fatal("duplicate permission write scheduled")
	}
	request, result := terminalControlsReply(t, broker, cmd, &rpc.Message{Result: []byte(`{"version":1,"revision":"9007199254740994","grants":[{"peerUuid":"peer","allowed":true,"state":"allowed"}]}`)})
	var params map[string]json.RawMessage
	if err := json.Unmarshal(request.Params, &params); err != nil {
		t.Fatal(err)
	}
	want := map[string]json.RawMessage{"peerUuid": json.RawMessage(`"peer"`), "allowed": json.RawMessage(`true`), "expectedRevision": json.RawMessage(`"9007199254740993"`)}
	if request.Method != "cluster:set-remote-management" || !reflect.DeepEqual(params, want) {
		t.Fatalf("CAS frame = %s %s", request.Method, request.Params)
	}
	if cmd := v.Update(result); cmd != nil {
		t.Fatal("successful set scheduled an unnecessary retry")
	}
	if v.remoteManagementSetting || !v.remoteManagement.Grants[0].Allowed || v.remoteManagement.Revision != "9007199254740994" {
		t.Fatalf("reply not adopted: %+v", v.remoteManagement)
	}
}

func TestLocalManagementFailureReadsBackSavedState(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want string
	}{
		{"conflict", &rpc.RPCError{Code: -32010, Message: "revision changed"}, "revision conflict (-32010)"},
		{"refused", &rpc.RPCError{Code: -32602, Message: "invalid peer"}, "invalid peer"},
		{"uncertain", context.DeadlineExceeded, "deadline exceeded"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			v := newNodesView(client)
			v.adoptRemoteManagement(remoteManagementSnapshot{Version: 1, Revision: "41", Grants: []remoteManagementGrant{{PeerUUID: "peer", State: "denied"}}})
			cmd := v.setRemoteManagement(v.remoteManagement.Grants[0])
			if v.remoteManagement.Grants[0].Allowed {
				t.Fatal("optimistic allow before failure")
			}
			var result tea.Msg
			if errors.Is(tc.err, context.DeadlineExceeded) {
				// The transport may lose the reply after the service persisted the change.
				result = remoteManagementSetMsg{writeID: v.remoteManagementWriteID, epoch: v.remoteManagementEpoch, err: tc.err}
			} else {
				var rpcErr *rpc.RPCError
				if !errors.As(tc.err, &rpcErr) {
					t.Fatal("expected RPC refusal")
				}
				_, result = terminalControlsReply(t, broker, cmd, &rpc.Message{Error: rpcErr})
			}
			reload := v.Update(result)
			if !strings.Contains(v.status.render(), tc.want) || !strings.Contains(v.status.render(), "reloading saved permissions") {
				t.Fatalf("failure status: %s", v.status.render())
			}
			if v.remoteManagementFresh || v.remoteManagement.Grants[0].Allowed {
				t.Fatal("failure invented saved permission")
			}
			request, readback := terminalControlsReply(t, broker, reload, &rpc.Message{Result: []byte(`{"version":1,"revision":"43","grants":[{"peerUuid":"peer","allowed":true,"state":"allowed"}]}`)})
			var params map[string]json.RawMessage
			if err := json.Unmarshal(request.Params, &params); err != nil {
				t.Fatal(err)
			}
			if request.Method != "cluster:get-remote-management" || len(params) != 0 {
				t.Fatalf("readback = %s %s", request.Method, request.Params)
			}
			v.Update(readback)
			if !v.remoteManagementFresh || !v.remoteManagement.Grants[0].Allowed || v.remoteManagement.Revision != "43" {
				t.Fatal("readback did not show landed state")
			}
		})
	}
}

func TestLocalManagementPushReplacesSnapshotAndRetiresRead(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	v := newNodesView(client)
	_, old := terminalControlsReply(t, broker, v.remoteManagementCmd(), &rpc.Message{Result: []byte(`{"version":1,"revision":"1","grants":[]}`)})
	reload := v.Update(NotificationMsg{Msg: &rpc.Message{Method: "cluster:remote-management-changed", Params: []byte(`{"version":1,"revision":"2","grants":[{"peerUuid":"peer","allowed":false,"state":"unreviewed"}]}`)}})
	v.Update(old)
	if v.remoteManagement.Revision != "2" || len(v.remoteManagement.Grants) != 1 {
		t.Fatal("old read overwrote push")
	}
	request, msg := terminalControlsReply(t, broker, reload, &rpc.Message{Result: []byte(`{"version":1,"revision":"2","grants":[{"peerUuid":"peer","allowed":false,"state":"unreviewed"}]}`)})
	if request.Method != "cluster:get-remote-management" {
		t.Fatalf("push reload = %s", request.Method)
	}
	v.Update(msg)
}

func TestLocalManagementPushOvertakesWriteReply(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	v := newNodesView(client)
	v.adoptRemoteManagement(remoteManagementSnapshot{Version: 1, Revision: "1", Grants: []remoteManagementGrant{{PeerUUID: "peer", State: "denied"}}})
	_, old := terminalControlsReply(t, broker, v.setRemoteManagement(v.remoteManagement.Grants[0]), &rpc.Message{Result: []byte(`{"version":1,"revision":"2","grants":[{"peerUuid":"peer","allowed":true,"state":"allowed"}]}`)})
	v.Update(NotificationMsg{Msg: &rpc.Message{Method: "cluster:remote-management-changed", Params: []byte(`{"version":1,"revision":"3","grants":[{"peerUuid":"peer","allowed":false,"state":"denied"}]}`)}})
	reload := v.Update(old)
	if v.remoteManagement.Revision != "3" || v.remoteManagement.Grants[0].Allowed {
		t.Fatal("late write reply overwrote newer revoke")
	}
	request, result := terminalControlsReply(t, broker, reload, &rpc.Message{Result: []byte(`{"version":1,"revision":"3","grants":[{"peerUuid":"peer","allowed":false,"state":"denied"}]}`)})
	if request.Method != "cluster:get-remote-management" {
		t.Fatalf("late write triggered %s", request.Method)
	}
	v.Update(result)
}

func TestLocalManagementPreWriteReadCannotUndoSavedRevoke(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	v := newNodesView(client)
	v.adoptRemoteManagement(remoteManagementSnapshot{Version: 1, Revision: "41", Grants: []remoteManagementGrant{{PeerUUID: "peer", Allowed: true, State: "allowed"}}})
	// A read started before the revoke is answered with the pre-revoke state but delivered last.
	_, staleRead := terminalControlsReply(t, broker, v.remoteManagementCmd(), &rpc.Message{Result: []byte(`{"version":1,"revision":"41","grants":[{"peerUuid":"peer","allowed":true,"state":"allowed"}]}`)})
	_, saved := terminalControlsReply(t, broker, v.setRemoteManagement(v.remoteManagement.Grants[0]), &rpc.Message{Result: []byte(`{"version":1,"revision":"42","grants":[{"peerUuid":"peer","allowed":false,"state":"denied"}]}`)})
	if cmd := v.Update(saved); cmd != nil {
		t.Fatal("successful revoke scheduled a reload")
	}
	v.Update(staleRead)
	if v.remoteManagement.Revision != "42" || v.remoteManagement.Grants[0].Allowed {
		t.Fatalf("pre-write read overwrote saved revoke: %+v", v.remoteManagement)
	}
}

func TestLocalManagementSelectionFollowsPeer(t *testing.T) {
	v := newNodesView(nil)
	v.adoptRemoteManagement(remoteManagementSnapshot{Version: 1, Grants: []remoteManagementGrant{{PeerUUID: "b"}, {PeerUUID: "c"}}})
	v.grantCursor = 1
	v.adoptRemoteManagement(remoteManagementSnapshot{Version: 1, Grants: []remoteManagementGrant{{PeerUUID: "a"}, {PeerUUID: "b"}, {PeerUUID: "c"}}})
	if v.remoteManagement.Grants[v.grantCursor].PeerUUID != "c" {
		t.Fatal("snapshot shifted the selected peer")
	}
}

func TestLocalManagementStatesAndLocalDirection(t *testing.T) {
	v := newNodesView(nil)
	v.SetSize(180, 35)
	v.feeds.members = []clusterNode{{NodeUUID: "peer", Name: "Workstation", State: "member"}}
	v.rebuild()
	v.grantsOpen = true
	for _, tc := range []struct {
		state, want string
		allowed     bool
	}{
		{"allowed", "allowed", true}, {"denied", "blocked", false}, {"unreviewed", "unreviewed", false}, {"unavailable", "unavailable", false},
	} {
		t.Run(tc.state, func(t *testing.T) {
			v.adoptRemoteManagement(remoteManagementSnapshot{Version: 1, Revision: "4", Grants: []remoteManagementGrant{{PeerUUID: "peer", Allowed: tc.allowed, State: tc.state}}})
			text := v.View()
			if !strings.Contains(text, "Allow Workstation to manage this device: "+tc.want) || !strings.Contains(text, "THIS device") {
				t.Fatalf("grant display: %s", text)
			}
			if tc.state == "unavailable" && v.handleGrantKey(tea.KeyMsg{Type: tea.KeyEnter}) != nil {
				t.Fatal("unavailable permission offered a write")
			}
		})
	}
}

func TestLocalManagementAttachAndReconnectOnlyRead(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	v := newNodesView(client)
	batch, ok := v.Init()().(tea.BatchMsg)
	if !ok {
		t.Fatal("attach did not batch reads")
	}
	// The permission read is independently scheduled at attach, after the existing feeds.
	request, result := terminalControlsReply(t, broker, batch[len(batch)-1], &rpc.Message{Result: []byte(`{"version":1,"revision":"7","grants":[]}`)})
	if request.Method != "cluster:get-remote-management" {
		t.Fatalf("attach method = %s", request.Method)
	}
	v.Update(result)
	v.Update(DisconnectedMsg{})
	v.Update(result)
	if v.remoteManagementFresh {
		t.Fatal("pre-disconnect read restored permission")
	}
	request, result = terminalControlsReply(t, broker, v.Update(NotificationMsg{Msg: &rpc.Message{Method: "app:ready"}}), &rpc.Message{Result: []byte(`{"version":1,"revision":"8","grants":[]}`)})
	if request.Method != "cluster:get-remote-management" {
		t.Fatalf("reconnect retried mutation: %s", request.Method)
	}
	v.Update(result)
}

func TestRemotePermissionReadbackStates(t *testing.T) {
	for _, tc := range []struct{ state, want string }{
		{"allowed", "allowed"}, {"denied", "blocked"}, {"unreviewed", "unreviewed"}, {"unavailable", "unavailable"}, {"unsupported", "unsupported"},
	} {
		t.Run(tc.state, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			d := newNodeDetail(client, nodeRow{key: "peer", name: "Peer", membership: membershipMember, presence: presenceOnline})
			d.SetSize(180, 40)
			body, err := json.Marshal(remotePermission{Version: 1, Allowed: tc.state == "allowed", Revision: "9007199254740993", State: tc.state})
			if err != nil {
				t.Fatal(err)
			}
			request, result := terminalControlsReply(t, broker, d.remotePermissionCmd(), &rpc.Message{Result: body})
			var params map[string]string
			if err := json.Unmarshal(request.Params, &params); err != nil {
				t.Fatal(err)
			}
			if request.Method != "engine:remote-management-get" || !reflect.DeepEqual(params, map[string]string{"nodeId": "peer"}) {
				t.Fatalf("remote read = %s %s", request.Method, request.Params)
			}
			d.update(result)
			if !strings.Contains(d.View(), "Remote management: "+tc.want) || !strings.Contains(d.View(), "read-only") {
				t.Fatalf("remote permission: %s", d.View())
			}
			for _, binding := range d.Help() {
				if binding.Help().Desc == localGrantsKey.Help().Desc {
					t.Fatal("remote detail offers local setter")
				}
			}
			if cmd, stay := d.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("g")}); cmd != nil || !stay {
				t.Fatal("remote detail opened a permission setter")
			}
		})
	}
}

func TestRemotePermissionFailureAndStaleReadAreUnknown(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "peer", membership: membershipMember, presence: presenceOnline})
	_, old := terminalControlsReply(t, broker, d.remotePermissionCmd(), &rpc.Message{Result: []byte(`{"version":1,"allowed":true,"revision":"1","state":"allowed"}`)})
	d.update(old)
	_, failed := terminalControlsReply(t, broker, d.remotePermissionCmd(), &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "unreachable"}})
	d.update(failed)
	if !strings.Contains(d.remotePermissionLine(), "unknown") {
		t.Fatal("failed read retained allow")
	}
	d.update(old)
	if !strings.Contains(d.remotePermissionLine(), "unknown") {
		t.Fatal("older successful read overwrote current unknown state")
	}
	d.update(DisconnectedMsg{})
	d.update(old)
	if !strings.Contains(d.remotePermissionLine(), "unknown") {
		t.Fatal("late pre-disconnect read restored allow")
	}
	local := newNodeDetail(client, nodeRow{self: true})
	if local.remotePermissionCmd() != nil || local.remotePermissionLine() != "" {
		t.Fatal("local detail made remote permission read")
	}
}

func TestDeniedManagementKeepsStatusReadableAndDoesNotRetry(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "peer", name: "Peer", membership: membershipMember, presence: presenceOnline})
	d.SetSize(180, 40)
	d.engines = []engineStatus{{Engine: "ollama", Installed: true, Running: true, Healthy: true}}
	d.refreshEngines()
	request, denied := terminalControlsReply(t, broker, d.modelCmd("ollama", modelActions["load"], "model"), &rpc.Message{Error: &rpc.RPCError{Code: -32012, Message: permissionDeniedText}})
	if request.Method != "engine:remote-load-model" {
		t.Fatalf("method = %s", request.Method)
	}
	if cmd, _ := d.update(denied); cmd != nil {
		t.Fatal("denied command scheduled retry")
	}
	if !strings.Contains(d.View(), permissionDeniedText) || !strings.Contains(d.View(), "ollama") {
		t.Fatalf("denial hid status: %s", d.View())
	}
	request, status := terminalControlsReply(t, broker, d.enginesCmd(), &rpc.Message{Result: []byte(`{"engines":[{"engine":"ollama","installed":true,"running":true,"healthy":true}]}`)})
	if request.Method != "engine:remote-get-installed" {
		t.Fatalf("status method = %s", request.Method)
	}
	d.update(status)
	if d.enginesStale || len(d.engines) != 1 || !d.engines[0].Running {
		t.Fatal("status unreadable after denial")
	}
	d.handleNotification(&rpc.Message{Method: "service/broker-restarted"})
	d.handleNotification(&rpc.Message{Method: "app:ready"})
	cmd, _ := d.update(TickMsg{})
	batch, ok := cmd().(tea.BatchMsg)
	if !ok {
		t.Fatal("reconnect did not refresh reads")
	}
	for _, read := range batch {
		request, _ := terminalControlsReply(t, broker, read, &rpc.Message{Result: []byte(`{"version":1,"state":"denied","availability":"available"}`)})
		if request.Method != "policy:get" && request.Method != "engine:remote-management-get" {
			t.Fatalf("reconnect retried management: %s", request.Method)
		}
	}
}

func TestPermissionDenialReachesAvailabilityAndSettings(t *testing.T) {
	denial := &rpc.RPCError{Code: -32012, Message: permissionDeniedText}
	a := availabilityState{value: availAvailable}
	a.update(setAvailabilityMsg{err: denial})
	if !strings.Contains(a.err, "remote management denied (-32012)") || a.value != availAvailable {
		t.Fatalf("availability denial = %+v", a)
	}
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "peer", membership: membershipMember})
	reload, _ := d.update(engineSettingsAppliedMsg{engine: "ollama", err: denial})
	if !strings.Contains(d.status.render(), permissionDeniedText) {
		t.Fatalf("settings denial = %s", d.status.render())
	}
	request, _ := terminalControlsReply(t, broker, reload, &rpc.Message{Result: []byte(`{"engine":"ollama"}`)})
	if request.Method != "engine:get-settings" {
		t.Fatalf("denied settings retried mutation: %s", request.Method)
	}
}

func TestGrantedManagementStillReportsLocalVeto(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "peer", name: "Peer", membership: membershipMember, presence: presenceOnline})
	d.remotePermission, d.remotePermissionFresh = remotePermission{Version: 1, Allowed: true, State: "allowed"}, true
	_, result := terminalControlsReply(t, broker, d.modelCmd("ollama", modelActions["load"], "model"), &rpc.Message{Error: &rpc.RPCError{Code: -32011, Message: "engine is Off locally"}})
	if cmd, _ := d.update(result); cmd != nil {
		t.Fatal("local veto scheduled a retry")
	}
	if !strings.Contains(d.status.render(), "local veto (-32011): engine is Off locally") || strings.Contains(d.status.render(), "remote management denied") {
		t.Fatalf("veto = %s", d.status.render())
	}
	if !strings.Contains(d.remotePermissionLine(), "allowed") {
		t.Fatal("local veto rewrote the target permission")
	}
}

func TestLocalGrantControlsKeepPairingVisible(t *testing.T) {
	v := newNodesView(nil)
	v.SetSize(100, 20)
	v.grantsOpen = true
	v.outboundInviteID, v.outboundName, v.outboundPIN = "one", "Peer", "123456"
	if !strings.Contains(v.View(), "invite to Peer is open - PIN 123456") {
		t.Fatal("local permissions hid the active pairing session")
	}
}

func TestPermissionControlsFitShortTerminal(t *testing.T) {
	for _, mode := range []nodesInputMode{nodesInputInviteChoice, nodesInputInviteAddress, nodesInputPin} {
		t.Run(map[nodesInputMode]string{nodesInputInviteChoice: "discovered", nodesInputInviteAddress: "manual", nodesInputPin: "inbound"}[mode], func(t *testing.T) {
			v := newNodesView(nil)
			v.SetSize(80, 9)
			if mode == nodesInputPin {
				v.receiveInvite(clusterInvite{InviteID: "one", FromNodeName: "Peer"})
			}
			v.beginInput(mode, "")
			v.status.info("ready")
			text := v.View()
			if renderedRows(text) > 9 {
				t.Fatalf("choice uses %d rows: %s", renderedRows(text), text)
			}
			const explanation = "This choice applies to the device you are using. Reading status and using models this device serves remain available. Engine locks and node pause still apply."
			if !strings.Contains(strings.Join(strings.Fields(text), " "), explanation) || !strings.Contains(text, "ready") {
				t.Fatalf("consent text or status hidden: %s", text)
			}
		})
	}
}
