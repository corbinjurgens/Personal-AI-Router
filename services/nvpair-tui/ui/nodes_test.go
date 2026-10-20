// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"context"
	"encoding/json"
	"net"
	"strings"
	"testing"
	"time"

	"nvpair-tui/rpc"

	tea "github.com/charmbracelet/bubbletea"
)

func TestReplacementClearsRetiredInboundPIN(t *testing.T) {
	f := newReconnectUIFixture(t)
	v := newNodesView(f.client)
	v.SetSize(140, 35)
	m := New(f.client, nil, []View{v})
	v.receiveInvite(clusterInvite{InviteID: "retired-inbound", FromNodeName: "retired inviting peer"})
	v.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("a")})
	const syntheticPIN = "246810"
	v.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(syntheticPIN)})
	v.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	if v.mode != nodesInputPin || v.input.Value() != syntheticPIN || !v.inboundAllow {
		t.Fatal("inbound acceptance did not retain the typed draft and Allow choice")
	}
	f.replace(t, &m)
	// Do not print input or rendered state: failures may contain a pairing PIN.
	if v.input.Value() != "" || v.inboundAllow {
		t.Error("replacement retained the retired inbound PIN or Allow choice")
	}
	if v.mode == nodesInputPin || v.inbound != nil || v.inboundPrompt() != "" || strings.Contains(v.View(), syntheticPIN) || strings.Contains(v.View(), "retired inviting peer") {
		t.Error("replacement retained an instruction for the retired inbound invitation")
	}
	v.receiveInvite(clusterInvite{InviteID: "fresh-inbound", FromNodeName: "fresh inviting peer"})
	v.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("a")})
	if v.mode != nodesInputPin || v.input.Value() != "" || v.inboundAllow {
		t.Error("fresh invitation did not open with an empty PIN and Block choice")
	}
	if !strings.Contains(v.View(), "(*) Block this peer from changing this device") {
		t.Error("fresh invitation did not render its default Block choice")
	}
}

func TestReplacementRetiresPairingInstructions(t *testing.T) {
	f := newReconnectUIFixture(t)
	v := newNodesView(f.client)
	v.SetSize(140, 35)
	m := New(f.client, nil, []View{v})
	const syntheticPIN = "654321"
	v.handleInviteResult(nodeInviteMsg{
		generation: v.inviteGeneration, name: "retired peer",
		inviteID: "retired-invite", pin: syntheticPIN,
	})
	if !v.status.sticky || !strings.Contains(v.status.render(), syntheticPIN) {
		t.Fatal("outbound invite did not display its persistent pairing instruction")
	}
	v.beginInput(nodesInputManualAddress, "host")
	v.input.SetValue("192.0.2.20")
	f.replace(t, &m)
	// Do not print the rendered text: a failure may contain a pairing PIN.
	if strings.Contains(v.View(), syntheticPIN) || strings.Contains(v.status.render(), "read it to that node") || v.status.sticky {
		t.Error("replacement retained an instruction for the retired pairing session")
	}
	if v.outboundPIN != "" || v.outboundInviteID != "" {
		t.Error("replacement retained retired outbound pairing credentials")
	}
	if v.mode != nodesInputManualAddress || v.input.Value() != "192.0.2.20" {
		t.Error("replacement discarded the unsent manual-address draft")
	}
}

func TestPairingCarriesLocalManagementChoice(t *testing.T) {
	for _, tc := range []struct {
		name, path string
		allow      bool
	}{
		{"discovered defaults to block", "discovered", false},
		{"discovered allows", "discovered", true},
		{"manual defaults to block", "manual", false},
		{"manual allows", "manual", true},
		{"inbound defaults to block", "inbound", false},
		{"inbound allows", "inbound", true},
		{"decline omits choice", "decline", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			v := newNodesView(client)
			v.SetSize(180, 40)
			v.feeds.discovered = []availableNode{{HostUUID: "peer", Name: "Peer", IPAddress: "192.0.2.1"}}
			v.rebuild()
			switch tc.path {
			case "discovered":
				v.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("p")})
			case "manual":
				v.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("n")})
				v.input.SetValue("192.0.2.2:14322")
			case "inbound", "decline":
				v.receiveInvite(clusterInvite{InviteID: "inbound", FromNodeName: "Peer"})
				v.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("a")})
				v.input.SetValue("123456")
			}
			if !strings.Contains(v.View(), "(*) Block this peer from changing this device") {
				t.Fatalf("choice did not default to Block: %s", v.View())
			}
			if tc.allow {
				v.handleKey(tea.KeyMsg{Type: tea.KeyTab})
			}
			var cmd tea.Cmd
			if tc.path == "decline" {
				v.cancelInput()
				cmd = v.respondToInvite(false, "")
			} else {
				cmd = v.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
			}
			request, result := terminalControlsReply(t, broker, cmd, &rpc.Message{Result: []byte(`{"inviteId":"session","state":"pending","pin":"123456"}`)})
			var params map[string]json.RawMessage
			if err := json.Unmarshal(request.Params, &params); err != nil {
				t.Fatal(err)
			}
			choice, present := params["allowRemoteManagement"]
			if tc.path == "decline" {
				if present {
					t.Fatalf("decline carried choice: %s", request.Params)
				}
			} else {
				var allowed bool
				if !present {
					t.Fatal("required allowRemoteManagement omitted")
				}
				if err := json.Unmarshal(choice, &allowed); err != nil {
					t.Fatal(err)
				}
				if allowed != tc.allow {
					t.Fatalf("allowRemoteManagement = %t, want %t", allowed, tc.allow)
				}
			}
			if tc.path == "inbound" || tc.path == "decline" {
				if request.Method != "cluster:respond-to-invite" || string(params["inviteId"]) != `"inbound"` || string(params["accept"]) != map[bool]string{true: "true", false: "false"}[tc.path != "decline"] {
					t.Fatalf("wrong inbound frame: %s %s", request.Method, request.Params)
				}
				if tc.path == "inbound" && string(params["pin"]) != `"123456"` {
					t.Fatalf("PIN lost: %s", request.Params)
				}
			} else {
				if request.Method != "cluster:invite-node" {
					t.Fatalf("method = %s", request.Method)
				}
				wantAddress := `"192.0.2.1"`
				if tc.path == "manual" {
					wantAddress = `"192.0.2.2"`
				}
				if string(params["address"]) != wantAddress {
					t.Fatalf("address = %s", params["address"])
				}
				if tc.path == "discovered" && string(params["nodeId"]) != `"peer"` {
					t.Fatal("discovered target identity lost")
				}
				if tc.path == "manual" && string(params["port"]) != "14322" {
					t.Fatal("manual port lost")
				}
			}
			// Submission freezes the choice even before its reply is delivered.
			v.togglePairingChoice()
			if tc.path == "inbound" && v.inboundAllow != tc.allow {
				t.Fatal("submitted inbound choice changed")
			}
			if (tc.path == "manual" || tc.path == "discovered") && v.outboundAllow != tc.allow {
				t.Fatal("submitted outbound choice changed")
			}
			v.Update(result)
		})
	}
}

func TestPairingChoiceResetsAndRejectsStaleGeneration(t *testing.T) {
	v := newNodesView(nil)
	v.beginInput(nodesInputInviteAddress, "host")
	v.togglePairingChoice()
	oldGeneration := v.inviteGeneration
	v.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if v.outboundAllow {
		t.Fatal("cancelled outbound choice retained Allow")
	}
	v.beginInput(nodesInputInviteAddress, "host")
	if v.outboundAllow {
		t.Fatal("new generation retained Allow")
	}
	v.Update(nodeInviteMsg{generation: oldGeneration, inviteID: "old", pin: "123456"})
	if v.outboundInviteID != "" || v.outboundPIN != "" {
		t.Fatal("stale reply restored old invite")
	}
	v.cancelInput()
	v.receiveInvite(clusterInvite{InviteID: "one", FromNodeName: "First"})
	v.beginInput(nodesInputPin, "PIN")
	v.togglePairingChoice()
	v.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if v.inboundAllow {
		t.Fatal("cancelled PIN editor retained Allow")
	}
	v.beginInput(nodesInputPin, "PIN")
	v.togglePairingChoice()
	v.receiveInvite(clusterInvite{InviteID: "two", FromNodeName: "Second"})
	v.dropInbound("one")
	if v.inboundAllow || v.inbound == nil || v.inbound.InviteID != "two" {
		t.Fatal("next invite did not reset to Block")
	}
	v.Update(pairingResultMsg{inviteID: "one", state: "paired"})
	if v.inbound == nil || v.inbound.InviteID != "two" {
		t.Fatal("stale pairing outcome removed next invite")
	}
}

func TestPendingPINWaitsForExactInvite(t *testing.T) {
	v := newNodesView(nil)
	v.SetSize(100, 30)
	v.Update(inviteReceived("one", "Sender"))
	v.answering = "one"
	if cmd := v.Update(pairingResultMsg{inviteID: "one", from: "Sender", state: "pending", reason: "rollback-pending"}); cmd == nil {
		t.Fatal("pending result did not request exact invite status")
	}
	if v.inbound == nil || v.inbound.InviteID != "one" || v.answering != "one" {
		t.Fatalf("pending prompt/answer guard lost: inbound=%+v answering=%q", v.inbound, v.answering)
	}
	if !strings.Contains(v.inboundPrompt(), "waiting for pairing confirmation") {
		t.Fatalf("pending prompt: %s", v.inboundPrompt())
	}
	if strings.Contains(v.status.render(), "failed") {
		t.Fatalf("pending mislabeled as failed: %s", v.status.render())
	}
	if cmd := v.respondToInvite(true, "123456"); cmd != nil {
		t.Fatal("accepted PIN could be submitted a second time")
	}
	if cmd := v.Update(pairingStatusMsg{inviteID: "one", result: inviteNodeResult{InviteID: "one", State: "pending"}}); cmd != nil {
		t.Fatal("pending poll immediately scheduled an unbounded next poll")
	}
	if cmd := v.Update(TickMsg{}); cmd == nil {
		t.Fatal("shell tick did not schedule pending status retry")
	}
	v.Update(pairingStatusMsg{inviteID: "one", result: inviteNodeResult{InviteID: "stale", State: "paired"}})
	if v.inbound == nil {
		t.Fatal("stale status retired pending invite")
	}
	v.Update(pairingStatusMsg{inviteID: "one", result: inviteNodeResult{InviteID: "one", State: "paired"}})
	if v.inbound != nil || v.answering != "" || v.awaitingPairing {
		t.Fatal("matching paired status did not retire pending state")
	}
	if !strings.Contains(v.status.render(), "paired with Sender") {
		t.Fatalf("terminal status: %s", v.status.render())
	}
	v.Update(pairingResultMsg{inviteID: "one", from: "Sender", state: "pending"})
	if v.awaitingPairing || v.inbound != nil {
		t.Fatal("delayed pending command resurrected terminal invite")
	}
}

func TestPendingPINUnknownAndFailure(t *testing.T) {
	for _, tc := range []struct {
		name   string
		result inviteNodeResult
		err    error
		want   string
	}{
		{"failure", inviteNodeResult{InviteID: "one", State: "failed", Reason: "storage-write-failed"}, nil, "storage-write-failed"},
		{"wrong PIN", inviteNodeResult{InviteID: "one", State: "failed", Reason: "incorrect-pin"}, nil, "wrong PIN"},
		{"expired", inviteNodeResult{InviteID: "one", State: "expired"}, nil, "could not be confirmed"},
		{"unknown", inviteNodeResult{}, &rpc.RPCError{Code: -32001, Message: "unknown inviteId"}, "could not be confirmed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v := newNodesView(nil)
			v.SetSize(100, 30)
			v.Update(inviteReceived("one", "Sender"))
			v.Update(pairingResultMsg{inviteID: "one", from: "Sender", state: "pending"})
			v.Update(pairingStatusMsg{inviteID: "one", result: tc.result, err: tc.err})
			if v.inbound != nil || v.answering != "" {
				t.Fatal("terminal outcome left accepted PIN answerable")
			}
			text := v.status.render()
			if !strings.Contains(text, tc.want) {
				t.Fatalf("outcome text = %q, want %q", text, tc.want)
			}
			if tc.name != "wrong PIN" && strings.Contains(text, "wrong PIN") {
				t.Fatal("non-PIN failure mislabeled as wrong PIN")
			}
		})
	}
}

func TestPendingPINPollUsesExactWireID(t *testing.T) {
	local, remote := net.Pipe()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() { cancel(); _ = local.Close(); _ = remote.Close() })
	client := rpc.NewClient(local, local)
	go client.Run(ctx)
	broker := rpc.NewCodec(remote, remote)
	errCh := make(chan error, 1)
	requestCh := make(chan *rpc.Message, 1)
	go func() {
		req, err := broker.Read()
		if err != nil {
			errCh <- err
			return
		}
		requestCh <- req
		errCh <- broker.Write(&rpc.Message{JSONRPC: "2.0", ID: req.ID, Result: json.RawMessage(`{"inviteId":"one","state":"failed","reason":"storage-write-failed"}`)})
	}()
	v := newNodesView(client)
	v.SetSize(100, 30)
	v.Update(inviteReceived("one", "Sender"))
	cmd := v.Update(pairingResultMsg{inviteID: "one", from: "Sender", state: "pending"})
	if cmd == nil {
		t.Fatal("missing status command")
	}
	v.Update(cmd())
	select {
	case req := <-requestCh:
		var params struct {
			InviteID string `json:"inviteId"`
		}
		if err := json.Unmarshal(req.Params, &params); err != nil {
			t.Fatal(err)
		}
		if req.Method != "cluster:invite-status" || params.InviteID != "one" {
			t.Fatalf("wrong poll request: method=%s params=%s", req.Method, req.Params)
		}
	case <-time.After(time.Second):
		t.Fatal("missing status request")
	}
	if err := <-errCh; err != nil {
		t.Fatal(err)
	}
	if v.inbound != nil || !strings.Contains(v.status.render(), "storage-write-failed") {
		t.Fatalf("matching wire terminal not rendered: %s", v.status.render())
	}
}
