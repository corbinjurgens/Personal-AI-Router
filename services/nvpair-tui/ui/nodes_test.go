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
)

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
