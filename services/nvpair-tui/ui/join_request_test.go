// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"nvpair-tui/rpc"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"
)

func joinKey(value string) tea.KeyMsg {
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(value)}
}

func joinEligibleView(t *testing.T) (*nodesView, *rpc.Codec) {
	t.Helper()
	client, broker := nodeInfoRPCFixture(t)
	v := newNodesView(client)
	v.SetSize(100, 30)
	v.joinFresh = true
	v.joinSnapshot = joinRequestsSnapshot{Supported: true, CanAskToJoin: true}
	v.feeds.discovered = []availableNode{{HostUUID: "foreign-uuid", Name: "Member", IPAddress: "192.0.2.2", Clustered: true}}
	v.rebuild()
	return v, broker
}

func TestJoinRequestForeignRowUsesAdmission(t *testing.T) {
	v, broker := joinEligibleView(t)
	if membershipForeign.invitable() {
		t.Fatal("foreign membership became ordinarily invitable")
	}
	if cmd := v.handleKey(joinKey("p")); cmd != nil || v.mode != nodesInputJoinConfirm {
		t.Fatal("foreign row did not require admission confirmation")
	}
	request, result := terminalControlsReply(t, broker, v.handleKey(tea.KeyMsg{Type: tea.KeyEnter}), &rpc.Message{Result: []byte(`{"requestId":"jr-outgoing","direction":"outgoing","state":"awaiting-approval"}`)})
	var params struct {
		Address, NodeID string
		ReplaceSolo     bool `json:"replaceSoloCluster"`
	}
	if err := json.Unmarshal(request.Params, &params); err != nil {
		t.Fatal(err)
	}
	// NodeID's wire spelling is checked separately because it is not nodeUUID.
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(request.Params, &fields); err != nil {
		t.Fatal(err)
	}
	if request.Method != "cluster:ask-to-join" || params.Address != "192.0.2.2" || string(fields["nodeId"]) != `"foreign-uuid"` || params.ReplaceSolo || len(fields) != 3 {
		t.Fatalf("wrong admission frame: %s %s", request.Method, request.Params)
	}
	msg, ok := result.(joinActionMsg)
	if !ok || msg.err != nil || msg.request.RequestID != "jr-outgoing" || msg.request.State != "awaiting-approval" {
		t.Fatal("admission result was not decoded")
	}
	v.Update(msg)
	if v.joinTrackedID != "jr-outgoing" {
		t.Fatal("request id was not retained separately")
	}
}

func TestJoinRequestManualDirectionAndPort(t *testing.T) {
	v, broker := joinEligibleView(t)
	v.handleKey(joinKey("n"))
	if v.joinByAddress {
		t.Fatal("manual direction did not start with Invite")
	}
	v.input.SetValue("example.test:14322")
	v.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	if !strings.Contains(v.View(), "Direction: Ask to join") {
		t.Fatal("manual direction is not explicit")
	}
	if cmd := v.handleKey(tea.KeyMsg{Type: tea.KeyEnter}); cmd != nil || v.mode != nodesInputJoinConfirm {
		t.Fatal("manual ask skipped review")
	}
	request, _ := terminalControlsReply(t, broker, v.handleKey(tea.KeyMsg{Type: tea.KeyEnter}), &rpc.Message{Result: []byte(`{"requestId":"manual","direction":"outgoing","state":"awaiting-approval"}`)})
	var params struct {
		Address     string
		Port        int
		ReplaceSolo bool `json:"replaceSoloCluster"`
	}
	if err := json.Unmarshal(request.Params, &params); err != nil {
		t.Fatal(err)
	}
	if request.Method != "cluster:ask-to-join" || params.Address != "example.test" || params.Port != 14322 || params.ReplaceSolo {
		t.Fatalf("wrong manual admission frame: %s %s", request.Method, request.Params)
	}
}

func TestJoinRequestManualInviteStillRefusesForeignTarget(t *testing.T) {
	v, _ := joinEligibleView(t)
	v.handleKey(joinKey("n"))
	v.input.SetValue("192.0.2.2")
	if cmd := v.handleKey(tea.KeyMsg{Type: tea.KeyEnter}); cmd != nil || !strings.Contains(v.status.text, "another cluster") {
		t.Fatal("manual Invite became a fallback for a foreign target")
	}
}

func TestJoinRequestSoloReplacementConsent(t *testing.T) {
	v, broker := joinEligibleView(t)
	v.joinSnapshot.RequiresSoloReplacement = true
	v.handleKey(joinKey("p"))
	if !strings.Contains(v.View(), "replaces your solo cluster after approval") {
		t.Fatal("solo replacement was not disclosed")
	}
	v.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if v.mode != nodesInputNone || v.joinWriting {
		t.Fatal("cancelled solo review sent an operation")
	}
	v.handleKey(joinKey("p"))
	request, _ := terminalControlsReply(t, broker, v.handleKey(tea.KeyMsg{Type: tea.KeyEnter}), &rpc.Message{Result: []byte(`{"requestId":"solo","direction":"outgoing","state":"awaiting-approval"}`)})
	var params struct {
		ReplaceSolo bool `json:"replaceSoloCluster"`
	}
	if err := json.Unmarshal(request.Params, &params); err != nil {
		t.Fatal(err)
	}
	if !params.ReplaceSolo {
		t.Fatal("explicit solo consent was not forwarded")
	}
}

func TestJoinRequestRetainedPeersRefused(t *testing.T) {
	v, _ := joinEligibleView(t)
	v.joinSnapshot.CanAskToJoin, v.joinSnapshot.BlockedReason = false, "already-clustered"
	if cmd := v.handleKey(joinKey("p")); cmd != nil || v.mode != nodesInputNone || !strings.Contains(v.status.text, "retained peer relationships") {
		t.Fatal("retained peers did not refuse admission locally")
	}
}

func TestJoinRequestEligibilityChangeRequiresFreshConsent(t *testing.T) {
	v, _ := joinEligibleView(t)
	v.handleKey(joinKey("p"))
	v.joinSnapshot.RequiresSoloReplacement = true
	if cmd := v.handleKey(tea.KeyMsg{Type: tea.KeyEnter}); cmd != nil || !strings.Contains(v.status.text, "eligibility changed") {
		t.Fatal("changed solo status reused stale consent")
	}
}

func TestJoinRequestApprovalGrantChoice(t *testing.T) {
	for _, allow := range []bool{false, true} {
		name := "defaults to block"
		if allow {
			name = "explicit allow"
		}
		t.Run(name, func(t *testing.T) {
			v, broker := joinEligibleView(t)
			v.adoptJoinRequests(joinRequestsSnapshot{Supported: true, Requests: []joinRequest{{RequestID: "review-id", Direction: "incoming", PeerName: "Requester", State: "awaiting-approval"}}})
			if v.mode != nodesInputNone {
				t.Fatal("incoming request stole focus")
			}
			v.handleKey(joinKey("a"))
			if v.mode != nodesInputJoinApproval || v.joinAllow || !strings.Contains(v.View(), "(*) Block this peer") {
				t.Fatal("approval did not default to block")
			}
			if allow {
				v.handleKey(tea.KeyMsg{Type: tea.KeyTab})
			}
			request, _ := terminalControlsReply(t, broker, v.handleKey(tea.KeyMsg{Type: tea.KeyEnter}), &rpc.Message{Result: []byte(`{"requestId":"review-id","direction":"incoming","state":"awaiting-requester","inviteId":"child-invite"}`)})
			var params struct {
				RequestID string `json:"requestId"`
				Accept    bool
				Allow     bool `json:"allowRemoteManagement"`
			}
			if err := json.Unmarshal(request.Params, &params); err != nil {
				t.Fatal(err)
			}
			if request.Method != "cluster:respond-to-join-request" || params.RequestID != "review-id" || !params.Accept || params.Allow != allow {
				t.Fatalf("wrong approval frame: %s %s", request.Method, request.Params)
			}
		})
	}
}

func TestJoinRequestDeclineOmitsGrant(t *testing.T) {
	v, broker := joinEligibleView(t)
	v.adoptJoinRequests(joinRequestsSnapshot{Supported: true, Requests: []joinRequest{{RequestID: "decline-id", Direction: "incoming", State: "awaiting-approval"}}})
	request, _ := terminalControlsReply(t, broker, v.handleKey(joinKey("d")), &rpc.Message{Result: []byte(`{"requestId":"decline-id","direction":"incoming","state":"declined"}`)})
	var params map[string]json.RawMessage
	if err := json.Unmarshal(request.Params, &params); err != nil {
		t.Fatal(err)
	}
	if request.Method != "cluster:respond-to-join-request" || len(params) != 2 || string(params["accept"]) != "false" || string(params["requestId"]) != `"decline-id"` {
		t.Fatalf("wrong decline frame: %s %s", request.Method, request.Params)
	}
}

func TestJoinRequestApprovalCannotRetarget(t *testing.T) {
	v, _ := joinEligibleView(t)
	v.adoptJoinRequests(joinRequestsSnapshot{Supported: true, Requests: []joinRequest{{RequestID: "old", Direction: "incoming", State: "awaiting-approval"}}})
	v.handleKey(joinKey("a"))
	v.adoptJoinRequests(joinRequestsSnapshot{Supported: true, Requests: []joinRequest{{RequestID: "new", Direction: "incoming", State: "awaiting-approval"}}})
	if v.mode != nodesInputNone || v.joinApprovalID != "" || v.joinReviewID != "new" {
		t.Fatal("retired approval was silently retargeted")
	}
	if cmd := v.handleKey(tea.KeyMsg{Type: tea.KeyEnter}); v.joinWriting {
		_ = cmd
		t.Fatal("enter approved an unreviewed replacement")
	}
}

func TestJoinRequestUnrelatedPairingOwnsAcceptKey(t *testing.T) {
	v, _ := joinEligibleView(t)
	v.adoptJoinRequests(joinRequestsSnapshot{Supported: true, Requests: []joinRequest{{RequestID: "admission", Direction: "incoming", State: "awaiting-approval"}}})
	v.receiveInvite(clusterInvite{InviteID: "unrelated-invite", FromNodeName: "Other"})
	v.handleKey(joinKey("a"))
	if v.mode != nodesInputPin || v.joinApprovalID != "" || v.inbound.InviteID != "unrelated-invite" {
		t.Fatal("admission intercepted unrelated displayed PIN prompt")
	}
	if strings.Contains(v.joinPrompt(), "a to review") {
		t.Fatal("two different accept actions were displayed")
	}
}

func TestJoinRequestHydratesRequesterAndDeduplicatesPush(t *testing.T) {
	v, broker := joinEligibleView(t)
	const snapshot = `{"supported":true,"canAskToJoin":false,"requests":[{"requestId":"jr-request","direction":"outgoing","peerName":"Member","state":"pairing","inviteId":"exact-invite"}]}`
	request, msg := terminalControlsReply(t, broker, v.joinRequestsCmd(), &rpc.Message{Result: []byte(snapshot)})
	if request.Method != "cluster:get-join-requests" || string(request.Params) != "{}" {
		t.Fatalf("wrong getter frame: %s %s", request.Method, request.Params)
	}
	v.Update(msg)
	v.handleNotification(&rpc.Message{Method: "cluster:join-requests-changed", Params: []byte(snapshot)})
	if v.joinTrackedID != "jr-request" || v.inbound == nil || v.inbound.InviteID != "exact-invite" || len(v.queued) != 0 {
		t.Fatal("request getter/push confused ids or duplicated the PIN prompt")
	}
	v.handleKey(joinKey("a"))
	if v.mode != nodesInputPin || v.inboundAllow {
		t.Fatal("requester did not enter its own default-block PIN ceremony")
	}
	const syntheticPIN = "135790"
	v.input.SetValue(syntheticPIN)
	frame, result := terminalControlsReply(t, broker, v.handleKey(tea.KeyMsg{Type: tea.KeyEnter}), &rpc.Message{Result: []byte(`{"inviteId":"exact-invite","state":"pending"}`)})
	var params struct {
		InviteID string `json:"inviteId"`
		Pin      string
		Allow    bool `json:"allowRemoteManagement"`
	}
	if err := json.Unmarshal(frame.Params, &params); err != nil {
		t.Fatal("invalid PIN response shape")
	}
	if frame.Method != "cluster:respond-to-invite" || params.InviteID != "exact-invite" || params.Pin != syntheticPIN || params.Allow {
		t.Fatal("requester PIN response used the wrong role, identity or grant")
	}
	cmd := v.Update(result)
	status, _ := terminalControlsReply(t, broker, cmd, &rpc.Message{Result: []byte(`{"inviteId":"exact-invite","state":"pending"}`)})
	var statusParams struct {
		InviteID string `json:"inviteId"`
	}
	if err := json.Unmarshal(status.Params, &statusParams); err != nil {
		t.Fatal(err)
	}
	if status.Method != "cluster:invite-status" || statusParams.InviteID != "exact-invite" || !v.awaitingPairing {
		t.Fatal("requester did not reuse exact-invite confirmation polling")
	}
}

func TestJoinRequestMemberPINUsesExactLocalInvite(t *testing.T) {
	v, broker := joinEligibleView(t)
	id := "member-child"
	cmd := v.adoptJoinRequests(joinRequestsSnapshot{Supported: true, Requests: []joinRequest{{RequestID: "member-request", Direction: "incoming", State: "pairing", InviteID: &id}}})
	const syntheticPIN = "975310"
	frame, msg := terminalControlsReply(t, broker, cmd, &rpc.Message{Result: []byte(`{"inviteId":"member-child","state":"pending","pin":"975310"}`)})
	var params struct {
		InviteID string `json:"inviteId"`
	}
	if err := json.Unmarshal(frame.Params, &params); err != nil {
		t.Fatal(err)
	}
	if frame.Method != "cluster:invite-status" || params.InviteID != "member-child" {
		t.Fatal("member did not retrieve exact local invite")
	}
	v.Update(msg)
	if v.joinPIN != syntheticPIN || !strings.Contains(v.View(), syntheticPIN) || v.inbound != nil {
		t.Fatal("member did not display the PIN in the inviter role")
	}
	v.Update(joinPINMsg{epoch: v.joinEpoch, requestID: "unrelated", inviteID: "member-child", result: inviteNodeResult{InviteID: "member-child", State: "pending"}})
	if v.joinPIN != syntheticPIN {
		t.Fatal("unrelated request cleared member PIN")
	}
}

func TestJoinRequestCancelUsesRequestID(t *testing.T) {
	v, broker := joinEligibleView(t)
	v.adoptJoinRequests(joinRequestsSnapshot{Supported: true, Requests: []joinRequest{{RequestID: "cancel-me", Direction: "outgoing", State: "awaiting-approval"}}})
	frame, _ := terminalControlsReply(t, broker, v.handleKey(joinKey("c")), &rpc.Message{Result: []byte(`{"requestId":"cancel-me","direction":"outgoing","state":"canceled"}`)})
	var params map[string]string
	if err := json.Unmarshal(frame.Params, &params); err != nil {
		t.Fatal(err)
	}
	if frame.Method != "cluster:cancel-join-request" || len(params) != 1 || params["requestId"] != "cancel-me" {
		t.Fatalf("wrong cancel frame: %s %s", frame.Method, frame.Params)
	}
}

func TestJoinRequestTerminalStatesRetireExactPrompt(t *testing.T) {
	for _, state := range []string{"expired", "declined", "canceled", "failed", "rejected", "paired"} {
		t.Run(state, func(t *testing.T) {
			v, _ := joinEligibleView(t)
			id := "child"
			r := joinRequest{RequestID: "tracked", Direction: "outgoing", PeerName: "Member", State: "pairing", InviteID: &id}
			v.adoptJoinRequests(joinRequestsSnapshot{Supported: true, Requests: []joinRequest{r}})
			v.receiveInvite(clusterInvite{InviteID: "unrelated", FromNodeName: "Other"})
			r.State, r.Reason = state, "callback-offline"
			v.adoptJoinRequests(joinRequestsSnapshot{Supported: true, Requests: []joinRequest{r}})
			if v.inbound == nil || v.inbound.InviteID != "unrelated" || len(v.queued) != 0 || !strings.Contains(v.status.text, state) || !strings.Contains(v.status.text, "callback-offline") {
				t.Fatal("terminal state did not retire only its child and explain the service reason")
			}
		})
	}
}

func TestJoinRequestReconnectHydratesAndRejectsOldReplies(t *testing.T) {
	v, _ := joinEligibleView(t)
	v.joinTrackedID, v.joinPIN, v.joinPINID = "previous-request", "synthetic-secret", "old-invite"
	oldEpoch := v.joinEpoch
	v.Update(DisconnectedMsg{})
	if v.joinPIN != "" || v.joinFresh || v.joinTrackedID != "previous-request" {
		t.Fatal("disconnect did not clear credentials and retain request correlation")
	}
	v.Update(joinActionMsg{epoch: oldEpoch, request: joinRequest{RequestID: "stale", Direction: "outgoing"}})
	if v.joinTrackedID != "previous-request" {
		t.Fatal("stale session result was adopted")
	}
	v.Update(joinRequestsMsg{epoch: v.joinEpoch, pushes: v.joinPushes, snapshot: joinRequestsSnapshot{Supported: true, Requests: []joinRequest{{RequestID: "received-without-tui", Direction: "incoming", State: "awaiting-approval"}}}})
	if !v.joinFresh || v.joinReviewID != "received-without-tui" || v.joinTrackedID != "" || !strings.Contains(v.status.text, "status lost") {
		t.Fatal("reconnect did not hydrate passive requests or explain lost status")
	}
}

func TestJoinRequestPushWinsOverOlderGetter(t *testing.T) {
	v, _ := joinEligibleView(t)
	v.handleNotification(&rpc.Message{Method: "cluster:join-requests-changed", Params: []byte(`{"supported":true,"canAskToJoin":false,"blockedReason":"already-clustered","requests":[]}`)})
	v.Update(joinRequestsMsg{epoch: v.joinEpoch, pushes: 0, snapshot: joinRequestsSnapshot{Supported: true, CanAskToJoin: true}})
	if v.joinSnapshot.CanAskToJoin || v.joinSnapshot.BlockedReason != "already-clustered" {
		t.Fatal("old read replaced pushed eligibility")
	}
}

func TestJoinRequestMutationRetiresOlderGetter(t *testing.T) {
	v, _ := joinEligibleView(t)
	v.joinRequestsCmd()
	oldPushes := v.joinPushes
	v.beginJoin("192.0.2.2", "foreign-uuid")
	v.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if v.joinReading || !v.joinWriting {
		t.Fatal("mutation did not retire its preceding read")
	}
	v.joinRequestsCmd()
	v.Update(joinRequestsMsg{epoch: v.joinEpoch, pushes: oldPushes, snapshot: joinRequestsSnapshot{Supported: true}})
	if !v.joinReading || !v.joinSnapshot.CanAskToJoin {
		t.Fatal("preceding read changed the snapshot or cleared the replacement read")
	}
}

func TestJoinRequestPassivePushPreservesDraft(t *testing.T) {
	v, _ := joinEligibleView(t)
	v.beginInput(nodesInputFilter, "filter")
	v.input.SetValue("draft")
	for range 2 {
		v.handleNotification(&rpc.Message{Method: "cluster:join-requests-changed", Params: []byte(`{"supported":true,"requests":[{"requestId":"incoming","direction":"incoming","state":"awaiting-approval"}]}`)})
	}
	if v.mode != nodesInputFilter || v.input.Value() != "draft" || len(v.joinSnapshot.Requests) != 1 || v.joinReviewID != "incoming" {
		t.Fatal("passive or duplicate admission push stole focus or duplicated state")
	}
}

func TestJoinRequestSessionReadySchedulesGetter(t *testing.T) {
	v, broker := joinEligibleView(t)
	v.Update(DisconnectedMsg{})
	cmd := v.Update(sessionReadyMsg{})
	batch, ok := cmd().(tea.BatchMsg)
	if !ok || len(batch) != 8 {
		t.Fatal("replacement session did not schedule all feed and permission reads")
	}
	frame, msg := terminalControlsReply(t, broker, batch[6], &rpc.Message{Result: []byte(`{"supported":true,"canAskToJoin":true,"requests":[{"requestId":"waiting","direction":"incoming","state":"awaiting-approval"}]}`)})
	if frame.Method != "cluster:get-join-requests" || string(frame.Params) != "{}" {
		t.Fatalf("replacement session getter = %s %s", frame.Method, frame.Params)
	}
	v.Update(msg)
	if !v.joinFresh || v.joinReviewID != "waiting" || v.mode != nodesInputNone {
		t.Fatal("replacement session did not hydrate passive incoming review")
	}
}

func TestJoinRequestGetterFailureDoesNotEstablishEligibility(t *testing.T) {
	v, broker := joinEligibleView(t)
	_, msg := terminalControlsReply(t, broker, v.joinRequestsCmd(), &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "cluster-manager unavailable"}})
	v.Update(msg)
	if v.joinFresh || !strings.Contains(v.feedWarning(), "cluster-manager unavailable") {
		t.Fatal("failed getter did not invalidate admission availability")
	}
	if cmd := v.handleKey(joinKey("p")); cmd != nil || v.mode != nodesInputNone {
		t.Fatal("connection failure established admission eligibility")
	}
}

func TestJoinRequestMemberPINFailureIsActionable(t *testing.T) {
	v, _ := joinEligibleView(t)
	id := "exact-child"
	v.adoptJoinRequests(joinRequestsSnapshot{Supported: true, Requests: []joinRequest{{RequestID: "incoming", Direction: "incoming", State: "pairing", InviteID: &id}}})
	v.Update(joinPINMsg{epoch: v.joinEpoch, requestID: "incoming", inviteID: id, err: errors.New("callback-offline")})
	if v.joinPIN != "" || v.joinPINReading || !strings.Contains(v.status.text, "check request status on both devices") {
		t.Fatal("PIN retrieval failure retained a secret or lacked status guidance")
	}
}

func TestJoinRequestMalformedPushRetiresPINAndRetriesRead(t *testing.T) {
	v, broker := joinEligibleView(t)
	id := "member-child"
	oldPIN := v.adoptJoinRequests(joinRequestsSnapshot{Supported: true, Requests: []joinRequest{{RequestID: "member-request", Direction: "incoming", State: "pairing", InviteID: &id}}})
	v.joinPIN = "synthetic-secret"
	v.handleNotification(&rpc.Message{Method: "cluster:join-requests-changed", Params: []byte(`{"requests":"invalid"}`)})
	if v.joinFresh || v.joinPIN != "" || v.joinPINReading || len(v.joinSnapshot.Requests) != 0 || v.joinPrompt() != "" {
		t.Fatal("malformed push retained actionable status or PIN")
	}
	_, stalePIN := terminalControlsReply(t, broker, oldPIN, &rpc.Message{Result: []byte(`{"inviteId":"member-child","state":"pending","pin":"stale-secret"}`)})
	v.Update(stalePIN)
	if v.joinPIN != "" {
		t.Fatal("a late PIN reply restored invalidated state")
	}
	batch, ok := v.Update(manualTickMsg{})().(tea.BatchMsg)
	if !ok || len(batch) != 3 {
		t.Fatal("unavailable admission feed did not schedule a bounded retry")
	}
	frame, msg := terminalControlsReply(t, broker, batch[2], &rpc.Message{Result: []byte(`{"supported":true,"canAskToJoin":true,"requests":[{"requestId":"member-request","direction":"incoming","state":"pairing","inviteId":"member-child"}]}`)})
	if frame.Method != "cluster:get-join-requests" {
		t.Fatal("retry did not read admission state")
	}
	newPIN := v.Update(msg)
	if !v.joinFresh || !v.joinSnapshot.CanAskToJoin || newPIN == nil {
		t.Fatal("successful retry did not restore authoritative eligibility")
	}
	_, freshPIN := terminalControlsReply(t, broker, newPIN, &rpc.Message{Result: []byte(`{"inviteId":"member-child","state":"pending","pin":"fresh-secret"}`)})
	v.Update(freshPIN)
	if v.joinPIN != "fresh-secret" {
		t.Fatal("fresh state did not replace the invalidated PIN read")
	}
}

func TestJoinRequestMalformedPushRetiresRequesterChildPrompt(t *testing.T) {
	v, _ := joinEligibleView(t)
	childID := "requester-child"
	v.adoptJoinRequests(joinRequestsSnapshot{Supported: true, Requests: []joinRequest{{
		RequestID: "requester-request", Direction: "outgoing", State: "pairing", InviteID: &childID,
	}}})
	v.receiveInvite(clusterInvite{InviteID: "ordinary-invite", FromNodeName: "Other"})
	v.beginInput(nodesInputPin, "PIN from the inviting node")
	v.input.SetValue("synthetic-secret")

	v.handleNotification(&rpc.Message{Method: "cluster:join-requests-changed", Params: []byte(`{"requests":"invalid"}`)})

	if v.mode == nodesInputPin || v.input.Value() != "" {
		t.Fatal("invalidated requester child retained its PIN prompt")
	}
	if v.inbound == nil || v.inbound.InviteID != "ordinary-invite" || len(v.queued) != 0 {
		t.Fatal("invalidating requester child did not preserve an unrelated ordinary invite")
	}
}

func TestJoinRequestUnsupportedLocalServicePreservesInvite(t *testing.T) {
	v, broker := joinEligibleView(t)
	_, msg := terminalControlsReply(t, broker, v.joinRequestsCmd(), &rpc.Message{Error: &rpc.RPCError{Code: -32601, Message: "method not found"}})
	v.Update(msg)
	if v.joinSnapshot.Supported || !v.joinFresh {
		t.Fatal("old local service did not become explicitly unsupported")
	}
	if cmd := v.handleKey(joinKey("p")); cmd != nil || v.mode != nodesInputNone || !strings.Contains(v.status.text, "unavailable") {
		t.Fatal("unsupported local service attempted admission")
	}
	v.feeds.discovered[0].Clustered = false
	v.rebuild()
	v.handleKey(joinKey("p"))
	frame, _ := terminalControlsReply(t, broker, v.handleKey(tea.KeyMsg{Type: tea.KeyEnter}), &rpc.Message{Result: []byte(`{"inviteId":"normal","state":"pending"}`)})
	if frame.Method != "cluster:invite-node" {
		t.Fatal("old local service lost ordinary Invite")
	}
}

func TestJoinRequestFailureDoesNotFallBackToInvite(t *testing.T) {
	v, _ := joinEligibleView(t)
	v.joinWriting = true
	v.Update(joinActionMsg{epoch: v.joinEpoch, err: errors.New("unsupported-peer")})
	if v.joinWriting || v.inviteSending || !strings.Contains(v.status.text, "unsupported-peer") || !strings.Contains(v.status.text, "refresh status") {
		t.Fatal("admission failure did not preserve refusal and retry guidance")
	}
}

func TestJoinRequestRenderAtNarrowAndOrdinaryWidths(t *testing.T) {
	for _, width := range []int{40, 100} {
		name := "narrow"
		if width == 100 {
			name = "ordinary"
		}
		t.Run(name, func(t *testing.T) {
			v, _ := joinEligibleView(t)
			v.SetSize(width, 30)
			id := "jr-0123456789abcdef0123456789abcdef"
			v.adoptJoinRequests(joinRequestsSnapshot{Supported: true, Requests: []joinRequest{{RequestID: id, Direction: "incoming", PeerName: "Requester", State: "awaiting-approval"}}})
			v.handleKey(joinKey("a"))
			view := ansi.Strip(v.View())
			if !strings.Contains(view, id) || !strings.Contains(view, "enter to approve; esc to cancel") || !strings.Contains(view, "Block this peer") {
				t.Fatal("approval lost request identity, action or cancel guidance")
			}
			for _, line := range strings.Split(view, "\n") {
				if ansi.StringWidth(line) > width {
					t.Fatal("approval exceeded terminal width")
				}
			}
			v.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
			v.joinSnapshot.CanAskToJoin, v.joinSnapshot.RequiresSoloReplacement = true, true
			v.handleKey(joinKey("p"))
			view = ansi.Strip(v.View())
			if !strings.Contains(view, "192.0.2.2") || !strings.Contains(view, "esc to cancel") || !strings.Contains(view, "solo cluster") {
				t.Fatal("solo confirmation lost target, consequence or cancel guidance")
			}
		})
	}
}
