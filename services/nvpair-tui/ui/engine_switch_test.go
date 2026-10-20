// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"nvpair-shared/enginesettings"
	"nvpair-tui/rpc"
)

func switchDetail(t *testing.T, remote bool) (*nodeDetail, *reconnectUIFixture) {
	t.Helper()
	f := newReconnectUIFixture(t)
	d := newNodeDetail(f.client, nodeRow{key: "node", name: "fixture", self: !remote,
		presence: presenceOnline, membership: membershipMember})
	d.engines = []engineStatus{{Engine: "ollama", Installed: true}, {Engine: "lmstudio", Installed: true}}
	d.refreshEngines()
	d.SetSize(100, 40)
	d.engineSwitch.source, d.engineSwitch.target = "ollama", "lmstudio"
	d.engineSwitch.revision, d.engineSwitch.fresh = "e3v1:"+strings.Repeat("a", 64), true
	d.remotePermissionFresh = true
	d.remotePermission = remotePermission{Version: 1, Allowed: true, State: "allowed"}
	t.Cleanup(d.close)
	if err := f.peer.SetDeadline(time.Now().Add(3 * time.Second)); err != nil {
		t.Fatal(err)
	}
	return d, f
}

func switchResponse(t *testing.T, value any) *rpc.Message {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return &rpc.Message{Result: raw}
}

func switchReceipt(d *nodeDetail, sequence uint64, phase string) enginesettings.SwitchReceipt {
	return enginesettings.SwitchReceipt{NodeID: d.node.key, RequestID: d.engineSwitch.requestID,
		Request: enginesettings.SwitchRequest{NodeID: d.node.key, RequestID: d.engineSwitch.requestID,
			SourceEngine: d.engineSwitch.source, TargetEngine: d.engineSwitch.target},
		Sequence: sequence, Phase: phase, Terminal: enginesettings.IsTerminal(phase),
		Source: enginesettings.SwitchEngineState{Intent: "off"}, Target: enginesettings.SwitchEngineState{Intent: "on"}}
}

func switchMessage(d *nodeDetail, r *enginesettings.SwitchReceipt) detailSwitchMsg {
	return detailSwitchMsg{visit: d.telemetryGen, route: d.telemetryRoute, epoch: d.engineSwitch.epoch,
		transport: clientGeneration(d.client), requestID: d.engineSwitch.requestID,
		status: enginesettings.SwitchStatus{Receipt: r}}
}

func TestSwitchCanonicalFramesAndDuplicateSubmit(t *testing.T) {
	for _, remote := range []bool{false, true} {
		name := "local"
		if remote {
			name = "remote"
		}
		t.Run(name, func(t *testing.T) {
			d, f := switchDetail(t, remote)
			d.engineSwitch.fresh = false
			req, reply := terminalControlsReply(t, f.broker, d.switchStatusCmd(true),
				switchResponse(t, enginesettings.SwitchStatus{ConfigurationRevision: "fresh-revision"}))
			var read enginesettings.SwitchStatusRequest
			if err := json.Unmarshal(req.Params, &read); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchStatusMethod || read.NodeID != d.nodeArg() || read.RequestID != "" {
				t.Fatalf("baseline frame: %+v %s", read, req.Method)
			}
			d.update(reply)
			cmd := d.submitSwitch()
			if cmd == nil {
				t.Fatal("first submit refused")
			}
			identity := d.engineSwitch.requestID
			if d.submitSwitch() != nil {
				t.Fatal("duplicate submit issued a mutation")
			}
			if d.engineSwitch.requestID != identity {
				t.Fatal("duplicate submit replaced pending identity")
			}
			r := switchReceipt(d, 1, enginesettings.SwitchDraining)
			req, reply = terminalControlsReply(t, f.broker, cmd, switchResponse(t, enginesettings.SwitchResult{Receipt: r}))
			var write enginesettings.SwitchRequest
			if err := json.Unmarshal(req.Params, &write); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchMethod || write.NodeID != d.nodeArg() || write.ExpectedRevision != "fresh-revision" || write.RequestID == "" || write.SourceEngine != "ollama" || write.TargetEngine != "lmstudio" {
				t.Fatalf("switch frame: %s %+v", req.Method, write)
			}
			wait, _ := d.update(reply)
			if d.switchStatusCmd(false) != nil {
				t.Fatal("second waiter started")
			}
			req, reply = terminalControlsReply(t, f.broker, wait, switchResponse(t, enginesettings.SwitchStatus{Receipt: &r}))
			if err := json.Unmarshal(req.Params, &read); err != nil {
				t.Fatal(err)
			}
			if read.RequestID != write.RequestID || read.AfterSequence == nil || *read.AfterSequence != 1 {
				t.Fatalf("wait frame: %+v", read)
			}
			next, _ := d.update(reply)
			if next == nil || d.engineSwitch.receipt.Phase != enginesettings.SwitchDraining {
				t.Fatal("unchanged wait was treated as completion")
			}
		})
	}
}

func TestSwitchReceiptIdentityAndSequence(t *testing.T) {
	d, _ := switchDetail(t, false)
	d.engineSwitch.requestID = "request"
	r := switchReceipt(d, 5, enginesettings.SwitchCompleted)
	d.applySwitch(switchMessage(d, &r))
	for _, sequence := range []uint64{5, 4} {
		older := switchReceipt(d, sequence, enginesettings.SwitchDraining)
		d.applySwitch(switchMessage(d, &older))
		if d.engineSwitch.receipt.Phase != enginesettings.SwitchCompleted {
			t.Fatal("duplicate/lower receipt regressed phase")
		}
	}
	for _, identity := range []string{"node", "request"} {
		other := switchReceipt(d, 99, enginesettings.SwitchDraining)
		if identity == "node" {
			other.NodeID = "other"
		} else {
			other.RequestID = "other"
		}
		d.applySwitch(switchMessage(d, &other))
		if d.engineSwitch.receipt.Sequence != 5 {
			t.Fatal("foreign receipt accepted")
		}
	}
}

func TestSwitchStaleReceiptAfterLeavingAndRevisiting(t *testing.T) {
	d, f := switchDetail(t, false)
	d.engineSwitch.requestID = "request"
	r := switchReceipt(d, 1, enginesettings.SwitchCompleted)
	old := switchMessage(d, &r)
	d.close()
	next := newNodeDetail(f.client, d.node)
	t.Cleanup(next.close)
	next.engineSwitch.requestID = "request"
	next.applySwitch(old)
	if next.engineSwitch.receipt != nil {
		t.Fatal("prior visit receipt accepted")
	}
}

func TestSwitchDisconnectMidSwitchCancelsAndFences(t *testing.T) {
	d, f := switchDetail(t, false)
	cmd := d.submitSwitch()
	done := holdReplacementCommand(t, f, cmd, enginesettings.SwitchMethod)
	ctx := d.engineSwitch.ctx
	r := switchReceipt(d, 1, enginesettings.SwitchCompleted)
	late := switchMessage(d, &r)
	d.update(DisconnectedMsg{})
	select {
	case <-ctx.Done():
	default:
		t.Fatal("disconnect did not cancel switch context")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("pending switch RPC leaked")
	}
	d.applySwitch(late)
	if d.engineSwitch.receipt != nil || !d.engineSwitch.unknown || d.engineSwitch.fresh {
		t.Fatal("disconnect accepted late completion or enabled mutation")
	}
	d.policyDisconnected = false
	if d.submitSwitch() != nil {
		t.Fatal("reconnect submitted before baseline")
	}
	req, reply := terminalControlsReply(t, f.broker, d.switchStatusCmd(true), switchResponse(t, enginesettings.SwitchStatus{ConfigurationRevision: "new"}))
	if req.Method != enginesettings.SwitchStatusMethod {
		t.Fatal("reconnect resent switch")
	}
	readback, _ := d.update(reply)
	r.Sequence = 2
	req, _ = terminalControlsReply(t, f.broker, readback, switchResponse(t, enginesettings.SwitchStatus{Receipt: &r}))
	var params enginesettings.SwitchStatusRequest
	if err := json.Unmarshal(req.Params, &params); err != nil {
		t.Fatal(err)
	}
	if params.RequestID != r.RequestID {
		t.Fatal("reconnect lost original request identity")
	}
}

func TestSwitchLostSubmitResponseReadsOriginalID(t *testing.T) {
	d, f := switchDetail(t, false)
	d.engineSwitch.requestID, d.engineSwitch.unknown = "lost", true
	msg := switchMessage(d, nil)
	msg.submit, msg.err = true, context.DeadlineExceeded
	cmd := d.applySwitch(msg)
	r := switchReceipt(d, 3, enginesettings.SwitchFailedInterrupted)
	r.RecoveryRequired = true
	req, reply := terminalControlsReply(t, f.broker, cmd, switchResponse(t, enginesettings.SwitchStatus{Receipt: &r}))
	var read enginesettings.SwitchStatusRequest
	if err := json.Unmarshal(req.Params, &read); err != nil {
		t.Fatal(err)
	}
	if req.Method != enginesettings.SwitchStatusMethod || read.RequestID != "lost" {
		t.Fatalf("lost response resent: %s %+v", req.Method, read)
	}
	d.update(reply)
	if d.engineSwitch.unknown || d.submitSwitch() != nil || !strings.Contains(d.switchView(), "recoveryRequired=true") {
		t.Fatal("recovery outcome missing or mutation enabled")
	}
}

func TestSwitchPermissionAndBlockedDraft(t *testing.T) {
	d, _ := switchDetail(t, true)
	for _, state := range []string{"unknown", "denied", "revoked"} {
		d.remotePermissionFresh = state != "unknown"
		d.remotePermission.Allowed = false
		if d.submitSwitch() != nil {
			t.Fatalf("submitted with %s permission", state)
		}
		if !strings.Contains(d.switchReason(), "management") {
			t.Fatal("permission confused with engine capability")
		}
	}
	if d.engineSwitch.source != "ollama" || d.engineSwitch.target != "lmstudio" {
		t.Fatal("denial lost draft")
	}
}

func TestSwitchTypedRefusalsSurviveReadback(t *testing.T) {
	for _, tc := range []struct {
		code int
		text string
	}{{-32010, "revision conflict"}, {-32011, "local veto"}, {-32012, "remote management denied"}} {
		t.Run(tc.text, func(t *testing.T) {
			d, _ := switchDetail(t, false)
			d.engineSwitch.requestID = "refused"
			msg := switchMessage(d, nil)
			msg.submit, msg.err = true, &rpc.RPCError{Code: tc.code, Message: "fixture"}
			d.applySwitch(msg)
			baseline := switchMessage(d, nil)
			baseline.requestID = ""
			baseline.status.ConfigurationRevision = "new"
			d.applySwitch(baseline)
			if !strings.Contains(d.switchView(), tc.text) || d.engineSwitch.source != "ollama" {
				t.Fatalf("refusal/draft lost: %s", d.switchView())
			}
			if tc.code == -32012 && d.remotePermissionFresh {
				t.Fatal("denial retained stale capability")
			}
		})
	}
}

func TestSwitchRestartRetiresAndRequiresReadback(t *testing.T) {
	d, _ := switchDetail(t, false)
	d.engineSwitch.requestID = "restart"
	d.switchStatusCmd(false)
	ctx := d.engineSwitch.ctx
	r := switchReceipt(d, 1, enginesettings.SwitchCompleted)
	late := switchMessage(d, &r)
	d.handleNotification(&rpc.Message{Method: "service/broker-restarted"})
	if !errors.Is(ctx.Err(), context.Canceled) {
		t.Fatal("restart did not cancel waiter")
	}
	d.applySwitch(late)
	if d.engineSwitch.receipt != nil || d.submitSwitch() != nil {
		t.Fatal("restart accepted old evidence")
	}
	d.handleNotification(&rpc.Message{Method: "app:ready"})
	if d.engineSwitch.fresh {
		t.Fatal("ready enabled writes without baseline")
	}
}

func TestSwitchDraftIdentitySurvivesReorder(t *testing.T) {
	d, _ := switchDetail(t, false)
	d.engineSwitch.open = true
	d.engines[0], d.engines[1] = d.engines[1], d.engines[0]
	d.refreshEngines()
	if d.engineSwitch.source != "ollama" || d.engineSwitch.target != "lmstudio" {
		t.Fatal("reorder changed captured identity")
	}
	if !d.CapturingInput() {
		t.Fatal("switch editor does not own keyboard")
	}
	view := d.View()
	for _, text := range []string{"Reported service build:", "From ollama", "To lmstudio"} {
		if !strings.Contains(view, text) {
			t.Fatalf("missing %q", text)
		}
	}
	d.switchKey(tea.KeyMsg{Type: tea.KeyEsc})
	if d.engineSwitch.open {
		t.Fatal("escape left editor open")
	}
}

func TestSwitchInitialActiveRequestReadback(t *testing.T) {
	d, f := switchDetail(t, false)
	_, reply := terminalControlsReply(t, f.broker, d.switchStatusCmd(true), switchResponse(t,
		enginesettings.SwitchStatus{ConfigurationRevision: "revision", ActiveRequestID: "other-client"}))
	cmd, _ := d.update(reply)
	req, reply := terminalControlsReply(t, f.broker, cmd, &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "receipt belongs to another caller"}})
	var params enginesettings.SwitchStatusRequest
	if err := json.Unmarshal(req.Params, &params); err != nil {
		t.Fatal(err)
	}
	if params.RequestID != "other-client" {
		t.Fatal("initial active receipt was not requested")
	}
	d.update(reply)
	if d.submitSwitch() != nil || d.engineSwitch.active != "other-client" || d.engineSwitch.err == nil {
		t.Fatal("unavailable foreign receipt enabled submission")
	}
}

func TestSwitchNotificationDuringBaselineRequiresNewRead(t *testing.T) {
	d, f := switchDetail(t, false)
	cmd := d.switchStatusCmd(true)
	raw, err := json.Marshal(enginesettings.SwitchChanged{NodeID: d.node.key, RequestID: "elsewhere", Sequence: 4})
	if err != nil {
		t.Fatal(err)
	}
	if d.switchChanged(raw) != nil {
		t.Fatal("notification created second concurrent read")
	}
	_, reply := terminalControlsReply(t, f.broker, cmd, switchResponse(t, enginesettings.SwitchStatus{ConfigurationRevision: "stale"}))
	next, _ := d.update(reply)
	if next == nil || d.engineSwitch.revision == "stale" {
		t.Fatal("stale baseline enabled action after invalidation")
	}
	_, reply = terminalControlsReply(t, f.broker, next, switchResponse(t, enginesettings.SwitchStatus{ConfigurationRevision: "fresh"}))
	d.update(reply)
	if d.engineSwitch.revision != "fresh" || !d.engineSwitch.fresh {
		t.Fatal("fresh baseline unavailable")
	}
}

func TestSwitchRouteChangesCancelWaiters(t *testing.T) {
	for _, change := range []string{"follow", "notification"} {
		t.Run(change, func(t *testing.T) {
			d, f := switchDetail(t, true)
			d.engineSwitch.requestID = "route"
			done := holdReplacementCommand(t, f, d.switchStatusCmd(false), enginesettings.SwitchStatusMethod)
			ctx := d.engineSwitch.ctx
			r := switchReceipt(d, 2, enginesettings.SwitchCompleted)
			old := switchMessage(d, &r)
			if change == "follow" {
				row := d.node
				row.address = "new-address"
				d.followNode(row, true)
			} else {
				raw, err := json.Marshal([]availableNode{{ID: d.node.key, IPAddress: "new-address"}})
				if err != nil {
					t.Fatal(err)
				}
				d.handleNotification(&rpc.Message{Method: "discovery:nodes-changed", Params: raw})
			}
			if ctx.Err() == nil {
				t.Fatal("route change retained switch context")
			}
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("route change leaked waiter")
			}
			d.applySwitch(old)
			if d.engineSwitch.inFlight || d.engineSwitch.receipt != nil {
				t.Fatal("route change accepted late receipt or stuck readback")
			}
		})
	}
}

func TestSwitchTerminalPhasesAndEvidence(t *testing.T) {
	for _, phase := range []string{enginesettings.SwitchFailedDrain, enginesettings.SwitchFailedTargetStart, enginesettings.SwitchFailedInterrupted} {
		t.Run(phase, func(t *testing.T) {
			d, _ := switchDetail(t, false)
			d.engineSwitch.open, d.engineSwitch.requestID = true, "failed"
			r := switchReceipt(d, 9, phase)
			r.Error = &enginesettings.SwitchError{Code: -32000, Kind: "drain", Message: "fixture failure"}
			d.applySwitch(switchMessage(d, &r))
			view := d.switchView()
			for _, want := range []string{phase, "sequence 9", "fixture failure", "saved intent=off", "running=unknown", "termination verified=false"} {
				if !strings.Contains(view, want) {
					t.Fatalf("missing %q in %s", want, view)
				}
			}
		})
	}
}

func TestSwitchPanelFitsNarrowFrame(t *testing.T) {
	d, _ := switchDetail(t, false)
	d.engineSwitch.open, d.engineSwitch.requestID = true, "frame"
	r := switchReceipt(d, 1, enginesettings.SwitchDraining)
	d.engineSwitch.receipt = &r
	d.SetSize(45, 12)
	if view := d.View(); countLines(view) > 12 {
		t.Fatalf("switch panel exceeds height: %s", view)
	}
}

func TestSwitchLocalOffRemainsReachable(t *testing.T) {
	d, f := switchDetail(t, false)
	d.engineSwitch.requestID = "remote-work"
	r := switchReceipt(d, 3, enginesettings.SwitchDraining)
	d.engineSwitch.receipt = &r
	req, _ := terminalControlsReply(t, f.broker, d.lifecycle(&d.engines[0], "stop"), switchResponse(t, map[string]bool{"success": true}))
	if req.Method != "engine:stop" {
		t.Fatalf("local Off method = %s", req.Method)
	}
	if d.engineSwitch.requestID != "remote-work" || d.engineSwitch.receipt.Phase != enginesettings.SwitchDraining {
		t.Fatal("local Off erased switch correlation")
	}
}

func TestSwitchReceiptDenialRefreshesPermission(t *testing.T) {
	d, _ := switchDetail(t, true)
	d.engineSwitch.requestID = "denied"
	r := switchReceipt(d, 2, enginesettings.SwitchFailedValidation)
	r.Error = &enginesettings.SwitchError{Code: -32012, Kind: "validation", Message: "revoked"}
	cmd := d.applySwitch(switchMessage(d, &r))
	if cmd == nil || d.remotePermissionFresh || !strings.Contains(d.switchView(), "remote management denied") {
		t.Fatal("receipt denial did not invalidate permission distinctly")
	}
}

func TestSwitchGenericRPCFailureKeepsOriginalReadback(t *testing.T) {
	d, f := switchDetail(t, true)
	d.engineSwitch.requestID, d.engineSwitch.unknown = "accepted-with-lost-reply", true
	msg := switchMessage(d, nil)
	msg.submit, msg.err = true, &rpc.RPCError{Code: -32000, Message: "response read failed"}
	readback := d.applySwitch(msg)
	if !d.engineSwitch.unknown || d.submitSwitch() != nil {
		t.Fatal("generic RPC failure was treated as definite refusal")
	}
	r := switchReceipt(d, 8, enginesettings.SwitchCompleted)
	req, reply := terminalControlsReply(t, f.broker, readback, switchResponse(t, enginesettings.SwitchStatus{Receipt: &r}))
	var params enginesettings.SwitchStatusRequest
	if err := json.Unmarshal(req.Params, &params); err != nil {
		t.Fatal(err)
	}
	if params.RequestID != r.RequestID {
		t.Fatal("generic failure read baseline instead of original receipt")
	}
	d.update(reply)
	if d.engineSwitch.receipt == nil || d.engineSwitch.unknown {
		t.Fatal("completed original receipt was not recovered")
	}
}
