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

func TestAutomaticEngineSwitchPresentation(t *testing.T) {
	d, _ := switchDetail(t, false)
	d.engineSwitch.requestID = "binding:run:1"
	r := switchReceipt(d, 3, enginesettings.SwitchDraining)
	r.Automatic = true
	d.engineSwitch.receipt = &r
	view := d.switchView()
	for _, want := range []string{"Automatic engine switch", "binding:run:1", "draining", "From receipt", "To receipt"} {
		if !strings.Contains(view, want) {
			t.Fatalf("missing %q in %s", want, view)
		}
	}
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

// Exercise the sent mutation and original-ID absence through the real command
// boundary. The next command must fence that ID, never replay the mutation.
func switchAbsentFence(t *testing.T, d *nodeDetail, f *reconnectUIFixture) (string, tea.Cmd) {
	t.Helper()
	_, reply := terminalControlsReply(t, f.broker, d.submitSwitch(),
		&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "reply lost"}})
	read, _ := d.update(reply)
	id := d.engineSwitch.requestID
	req, reply := terminalControlsReply(t, f.broker, read,
		&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "Unknown switch receipt for this caller."}})
	var p enginesettings.SwitchStatusRequest
	if err := json.Unmarshal(req.Params, &p); err != nil {
		t.Fatal(err)
	}
	if req.Method != enginesettings.SwitchStatusMethod || p.RequestID != id {
		t.Fatalf("original read = %s %+v", req.Method, p)
	}
	fence, _ := d.update(reply)
	if fence == nil || d.submitSwitch() != nil || !d.engineSwitch.outcomeUnknown() {
		t.Fatal("absence must issue one fence while retaining uncertainty")
	}
	return id, fence
}

func TestSwitchFenceAcknowledgementRequiresFreshBaseline(t *testing.T) {
	for _, remote := range []bool{false, true} {
		name := "local"
		if remote {
			name = "remote"
		}
		t.Run(name, func(t *testing.T) {
			d, f := switchDetail(t, remote)
			id, fence := switchAbsentFence(t, d, f)
			req, reply := terminalControlsReply(t, f.broker, fence, switchResponse(t,
				enginesettings.SwitchFenceResult{NodeID: d.node.key, RequestID: id, Fenced: true}))
			var p enginesettings.SwitchFenceRequest
			if err := json.Unmarshal(req.Params, &p); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchFenceMethod || p.NodeID != d.nodeArg() || p.RequestID != id {
				t.Fatalf("fence frame = %s %+v", req.Method, p)
			}
			if d.switchStatusCmd(false) != nil {
				t.Fatal("fence allowed a second waiter")
			}
			baseline, _ := d.update(reply)
			old := switchMessage(d, nil)
			old.submit = true
			if d.engineSwitch.outcomeUnknown() || d.engineSwitch.fresh || d.submitSwitch() != nil ||
				!strings.Contains(d.switchView(), "Not accepted; original request fenced") {
				t.Fatalf("fence acknowledgement evidence/reason: %s / %s", d.switchView(), d.switchReason())
			}
			req, reply = terminalControlsReply(t, f.broker, baseline, switchResponse(t,
				enginesettings.SwitchStatus{ConfigurationRevision: "after-fence"}))
			var read enginesettings.SwitchStatusRequest
			if err := json.Unmarshal(req.Params, &read); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchStatusMethod || read.RequestID != "" {
				t.Fatalf("fresh baseline = %s %+v", req.Method, read)
			}
			if next, _ := d.update(reply); next != nil {
				t.Fatal("fenced ID was read again")
			}
			if reason := d.switchReason(); reason != "" {
				t.Fatalf("fresh baseline blocked: %s", reason)
			}
			if d.engineSwitch.requestID != id {
				t.Fatal("fence evidence lost before deliberate submit")
			}
			cmd := d.submitSwitch()
			newID := d.engineSwitch.requestID
			if newID == id || newID == "" {
				t.Fatal("replacement did not allocate a fresh ID")
			}
			late := switchReceipt(d, 99, enginesettings.SwitchCompleted)
			late.RequestID, late.Request.RequestID = id, id
			old.status.Receipt = &late
			d.applySwitch(old)
			if d.engineSwitch.requestID != newID || d.engineSwitch.receipt != nil || !d.engineSwitch.inFlight {
				t.Fatal("stale original callback changed replacement")
			}
			req, _ = terminalControlsReply(t, f.broker, cmd, &rpc.Message{Error: &rpc.RPCError{Code: -32602, Message: "fixture stops"}})
			var write enginesettings.SwitchRequest
			if err := json.Unmarshal(req.Params, &write); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchMethod || write.RequestID != newID || write.ExpectedRevision != "after-fence" {
				t.Fatalf("replacement frame = %s %+v", req.Method, write)
			}
		})
	}
}

func TestSwitchFenceFailureRetainsOriginalAndRetriesReadback(t *testing.T) {
	for _, scenario := range []string{"lost acknowledgement", "denied", "storage", "malformed", "wrong request", "wrong node", "missing fenced", "null fenced", "null receipt", "unknown field", "both alternatives", "neither alternative"} {
		t.Run(scenario, func(t *testing.T) {
			d, f := switchDetail(t, false)
			id, fence := switchAbsentFence(t, d, f)
			row := map[string]any{"nodeId": d.node.key, "requestId": id, "fenced": true}
			response := switchResponse(t, row)
			switch scenario {
			case "lost acknowledgement":
				response = &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "reply lost"}}
			case "denied":
				response = &rpc.Message{Error: &rpc.RPCError{Code: -32012, Message: "permission denied"}}
			case "storage":
				response = &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "storage uncertain"}}
			case "malformed":
				row["fenced"] = "yes"
			case "wrong request":
				row["requestId"] = "other"
			case "wrong node":
				row["nodeId"] = "other"
			case "missing fenced":
				delete(row, "fenced")
			case "null fenced":
				row["fenced"] = nil
			case "null receipt":
				row["receipt"] = nil
			case "unknown field":
				row["unexpected"] = true
			case "both alternatives":
				row["receipt"] = switchReceipt(d, 1, enginesettings.SwitchDraining)
			case "neither alternative":
				row["fenced"] = false
			}
			if scenario != "lost acknowledgement" && scenario != "denied" && scenario != "storage" {
				response = switchResponse(t, row)
			}
			req, reply := terminalControlsReply(t, f.broker, fence, response)
			if req.Method != enginesettings.SwitchFenceMethod {
				t.Fatalf("method = %s", req.Method)
			}
			if next, _ := d.update(reply); next != nil {
				t.Fatal("failed fence created an automatic retry loop")
			}
			if d.engineSwitch.requestID != id || !d.engineSwitch.outcomeUnknown() || d.submitSwitch() != nil || d.engineSwitch.err == nil {
				t.Fatalf("failed fence resolved original: %s / %s", d.switchView(), d.switchReason())
			}
			_, reply = terminalControlsReply(t, f.broker, d.switchStatusCmd(true), switchResponse(t, enginesettings.SwitchStatus{ConfigurationRevision: "idle"}))
			read, _ := d.update(reply)
			_, reply = terminalControlsReply(t, f.broker, read, &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "Unknown switch receipt for this caller."}})
			retry, _ := d.update(reply)
			req, reply = terminalControlsReply(t, f.broker, retry, switchResponse(t, enginesettings.SwitchFenceResult{NodeID: d.node.key, RequestID: id, Fenced: true}))
			if req.Method != enginesettings.SwitchFenceMethod {
				t.Fatalf("retry = %s", req.Method)
			}
			d.update(reply)
		})
	}
}

func TestSwitchFenceRawWireAmbiguityRetainsUncertainty(t *testing.T) {
	for _, row := range []struct{ name, wire string }{
		{"conflicting case variant flag", `{"nodeId":"node-a","requestId":"lost","fenced":false,"Fenced":true}`},
		{"conflicting duplicate flag", `{"nodeId":"node-a","requestId":"lost","fenced":false,"fenced":true}`},
		{"equal duplicate flag", `{"nodeId":"node-a","requestId":"lost","fenced":true,"fenced":true}`},
		{"escaped duplicate flag", `{"nodeId":"node-a","requestId":"lost","fenced":false,"\u0066enced":true}`},
		{"missing flag", `{"nodeId":"node-a","requestId":"lost"}`},
		{"null flag", `{"nodeId":"node-a","requestId":"lost","fenced":null}`},
		{"null receipt", `{"nodeId":"node-a","requestId":"lost","fenced":true,"receipt":null}`},
		{"case variant flag", `{"nodeId":"node-a","requestId":"lost","Fenced":true}`},
		{"case variant node", `{"NodeId":"node-a","requestId":"lost","fenced":true}`},
		{"case variant request", `{"nodeId":"node-a","RequestId":"lost","fenced":true}`},
		{"case variant receipt", `{"nodeId":"node-a","requestId":"lost","fenced":true,"Receipt":null}`},
	} {
		t.Run(row.name, func(t *testing.T) {
			d, f := switchDetail(t, false)
			d.node.key = "node-a"
			s := &d.engineSwitch
			s.transition(switchTransition{event: switchSubmit, id: "lost"})
			s.fresh = false
			p := enginesettings.SwitchRequest{RequestID: "lost", SourceEngine: s.source,
				TargetEngine: s.target, ExpectedRevision: s.revision}
			req, reply := terminalControlsReply(t, f.broker,
				d.switchCommand(enginesettings.SwitchMethod, p, p.RequestID, true),
				&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "reply lost"}})
			if req.Method != enginesettings.SwitchMethod {
				t.Fatalf("submit method = %s", req.Method)
			}
			read, _ := d.update(reply)
			_, reply = terminalControlsReply(t, f.broker, read,
				&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "Unknown switch receipt for this caller."}})
			fence, _ := d.update(reply)
			if fence == nil {
				t.Fatal("own-ID absence did not request a fence")
			}
			done := make(chan tea.Msg, 1)
			go func() { done <- fence() }()
			req, err := f.broker.Read()
			if err != nil {
				t.Fatal(err)
			}
			var request enginesettings.SwitchFenceRequest
			if err := json.Unmarshal(req.Params, &request); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchFenceMethod || request.RequestID != "lost" || req.ID == nil {
				t.Fatalf("fence frame = %s %+v", req.Method, request)
			}
			// Write raw JSON so duplicate/case-variant keys reach the production
			// decoder unchanged instead of being collapsed by a map fixture.
			frame := []byte(`{"jsonrpc":"2.0","id":` + string(*req.ID) + `,"result":` + row.wire + "}\n")
			if n, err := f.peer.Write(frame); err != nil || n != len(frame) {
				t.Fatalf("write raw fence response: bytes=%d/%d error=%v", n, len(frame), err)
			}
			select {
			case reply = <-done:
			case <-time.After(time.Second):
				t.Fatal("raw fence response did not finish")
			}
			if next, _ := d.update(reply); next != nil {
				t.Fatal("ambiguous fence response started automatic progress")
			}
			if s.requestID != "lost" || !s.outcomeUnknown() || s.state != switchAbsent || s.err == nil ||
				s.fresh || d.submitSwitch() != nil || d.switchReason() == "" || strings.Contains(d.switchView(), "original request fenced") {
				t.Fatalf("ambiguous wire resolved original: %s / %s", d.switchView(), d.switchReason())
			}
		})
	}
}

// The engine-manager's TestSwitchAuthoritativeAbsentHTTPRoundTripPreservesCodeAndMessage
// covers operation_error HTTP translation. Enter this consumer at its actual
// service RPC seam, retaining the exact remote code/message without HTTP-text
// matching or a second implementation of that translation in the TUI.
func TestSwitchRemotePreservedStatusErrorFenceRecovery(t *testing.T) {
	for _, row := range []struct {
		name    string
		code    int
		message string
		fence   bool
	}{
		{"authoritative absence", -32000, "Unknown switch receipt for this caller.", true},
		{"different operation failure", -32000, "Target storage is unavailable.", false},
		{"different refusal code", -32012, "Unknown switch receipt for this caller.", false},
	} {
		t.Run(row.name, func(t *testing.T) {
			d, f := switchDetail(t, true)
			_, reply := terminalControlsReply(t, f.broker, d.submitSwitch(),
				&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "response lost"}})
			read, _ := d.update(reply)
			original := d.engineSwitch.requestID
			req, reply := terminalControlsReply(t, f.broker, read,
				&rpc.Message{Error: &rpc.RPCError{Code: row.code, Message: row.message}})
			var status enginesettings.SwitchStatusRequest
			if err := json.Unmarshal(req.Params, &status); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchStatusMethod || status.NodeID == "" || status.NodeID != d.nodeArg() || status.RequestID != original {
				t.Fatalf("remote original-ID status = %s %+v", req.Method, status)
			}
			wrapped, ok := reply.(detailReply)
			if !ok {
				t.Fatal("missing addressed RPC reply")
			}
			msg, ok := wrapped.msg.(detailSwitchMsg)
			if !ok {
				t.Fatal("missing switch RPC reply")
			}
			var preserved *rpc.RPCError
			if !errors.As(msg.err, &preserved) || preserved.Code != row.code || preserved.Message != row.message {
				t.Fatalf("remote status error changed at client boundary: %v", msg.err)
			}
			fence, _ := d.update(reply)
			if !d.engineSwitch.outcomeUnknown() || d.engineSwitch.requestID != original || d.submitSwitch() != nil {
				t.Fatal("remote status error resolved uncertainty before fence acknowledgement")
			}
			if !row.fence {
				if fence != nil || d.engineSwitch.state != switchUncertain {
					t.Fatal("non-absence error triggered fencing")
				}
				return
			}
			if fence == nil || d.engineSwitch.state != switchAbsent {
				t.Fatal("preserved remote absence did not request a fence")
			}
			req, reply = terminalControlsReply(t, f.broker, fence, switchResponse(t,
				enginesettings.SwitchFenceResult{NodeID: d.node.key, RequestID: original, Fenced: true}))
			var request enginesettings.SwitchFenceRequest
			if err := json.Unmarshal(req.Params, &request); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchFenceMethod || request.NodeID != d.nodeArg() || request.RequestID != original {
				t.Fatalf("remote fence = %s %+v", req.Method, request)
			}
			baseline, _ := d.update(reply)
			if d.engineSwitch.state != switchFenced || d.engineSwitch.fresh || d.submitSwitch() != nil ||
				!strings.Contains(d.switchView(), "Not accepted; original request fenced") {
				t.Fatalf("remote fence acknowledgement = %s / %s", d.switchView(), d.switchReason())
			}
			req, reply = terminalControlsReply(t, f.broker, baseline,
				switchResponse(t, enginesettings.SwitchStatus{ConfigurationRevision: "remote-after-fence"}))
			status = enginesettings.SwitchStatusRequest{}
			if err := json.Unmarshal(req.Params, &status); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchStatusMethod || status.NodeID != d.nodeArg() || status.RequestID != "" {
				t.Fatalf("post-fence baseline = %s %+v", req.Method, status)
			}
			if next, _ := d.update(reply); next != nil {
				t.Fatal("fenced remote ID triggered more automatic work")
			}
			if reason := d.switchReason(); reason != "" {
				t.Fatalf("fresh remote baseline still blocked: %s", reason)
			}
			req, _ = terminalControlsReply(t, f.broker, d.submitSwitch(),
				&rpc.Message{Error: &rpc.RPCError{Code: -32602, Message: "fixture stops"}})
			var replacement enginesettings.SwitchRequest
			if err := json.Unmarshal(req.Params, &replacement); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchMethod || replacement.NodeID != d.nodeArg() || replacement.RequestID == original ||
				replacement.RequestID == "" || replacement.ExpectedRevision != "remote-after-fence" {
				t.Fatalf("remote deliberate replacement = %s %+v", req.Method, replacement)
			}
		})
	}
}

func TestSwitchFenceStaleScopeCannotResolveOriginal(t *testing.T) {
	for _, scope := range []string{"visit", "route", "epoch", "transport"} {
		t.Run(scope, func(t *testing.T) {
			d, f := switchDetail(t, false)
			id, fence := switchAbsentFence(t, d, f)
			_, reply := terminalControlsReply(t, f.broker, fence, switchResponse(t, enginesettings.SwitchFenceResult{NodeID: d.node.key, RequestID: id, Fenced: true}))
			switch scope {
			case "visit":
				d.telemetryGen++
			case "route":
				d.telemetryRoute++
			case "epoch":
				d.retireSwitch()
			case "transport":
				wrapped, ok := reply.(detailReply)
				if !ok {
					t.Fatal("missing detail reply")
				}
				msg, ok := wrapped.msg.(detailSwitchMsg)
				if !ok {
					t.Fatal("missing switch reply")
				}
				msg.transport++
				wrapped.msg = msg
				reply = wrapped
			}
			d.update(reply)
			if !d.engineSwitch.outcomeUnknown() || d.engineSwitch.requestID != id || d.submitSwitch() != nil {
				t.Fatal("stale fence resolved uncertainty")
			}
		})
	}
}

func TestSwitchFenceAcceptanceRaceFollowsOwnedReceipt(t *testing.T) {
	for _, phase := range []string{enginesettings.SwitchDraining, enginesettings.SwitchCompleted} {
		t.Run(phase, func(t *testing.T) {
			d, f := switchDetail(t, true)
			id, fence := switchAbsentFence(t, d, f)
			r := switchReceipt(d, 3, phase)
			r.Version, r.Caller = 1, "local"
			r.ExpectedRevision = "e3v1:" + strings.Repeat("a", 64)
			r.Request.ExpectedRevision, r.AcceptedRevision = r.ExpectedRevision, r.ExpectedRevision
			r.PayloadHash = enginesettings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
			r.AcceptedAt = time.Now().UTC()
			r.UpdatedAt, r.Deadline = r.AcceptedAt, r.AcceptedAt.Add(time.Minute)
			if r.Terminal {
				r.FinishedAt = &r.UpdatedAt
			}
			if err := r.Validate(); err != nil {
				t.Fatal(err)
			}
			_, reply := terminalControlsReply(t, f.broker, fence, switchResponse(t,
				enginesettings.SwitchFenceResult{NodeID: d.node.key, RequestID: id, Receipt: &r}))
			next, _ := d.update(reply)
			if d.engineSwitch.state != switchAccepted || d.engineSwitch.receipt == nil || d.engineSwitch.receipt.Phase != phase ||
				d.engineSwitch.requestID != id || strings.Contains(d.switchView(), "original request fenced") {
				t.Fatalf("acceptance race lost real receipt: %s", d.switchView())
			}
			req, reply := terminalControlsReply(t, f.broker, next, &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "fixture stops"}})
			var p enginesettings.SwitchStatusRequest
			if err := json.Unmarshal(req.Params, &p); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchStatusMethod || (!r.Terminal && (p.RequestID != id || p.AfterSequence == nil || *p.AfterSequence != 3)) ||
				(r.Terminal && p.RequestID != "") {
				t.Fatalf("accepted readback = %s %+v", req.Method, p)
			}
			d.update(reply)
		})
	}
}

func TestSwitchFenceBaselineMustBeCurrentAndAvailable(t *testing.T) {
	for _, scenario := range []string{"failed", "empty revision", "invalidated", "retired"} {
		t.Run(scenario, func(t *testing.T) {
			d, f := switchDetail(t, false)
			id, fence := switchAbsentFence(t, d, f)
			_, reply := terminalControlsReply(t, f.broker, fence, switchResponse(t,
				enginesettings.SwitchFenceResult{NodeID: d.node.key, RequestID: id, Fenced: true}))
			baseline, _ := d.update(reply)
			response := switchResponse(t, enginesettings.SwitchStatus{ConfigurationRevision: "after-fence"})
			switch scenario {
			case "failed":
				response = &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "unavailable"}}
			case "empty revision":
				response = switchResponse(t, enginesettings.SwitchStatus{})
			case "invalidated":
				raw, err := json.Marshal(enginesettings.SwitchChanged{NodeID: d.node.key, RequestID: "foreign", Sequence: 1})
				if err != nil {
					t.Fatal(err)
				}
				if d.switchChanged(raw) != nil {
					t.Fatal("notification added second waiter")
				}
			}
			_, reply = terminalControlsReply(t, f.broker, baseline, response)
			if scenario == "retired" {
				d.retireSwitch()
			}
			d.update(reply)
			if d.engineSwitch.fresh || d.submitSwitch() != nil || d.engineSwitch.state != switchFenced ||
				d.engineSwitch.requestID != id || !strings.Contains(d.switchReason(), "readback") {
				t.Fatalf("unusable post-fence baseline authorized replacement: %s / %s", d.switchView(), d.switchReason())
			}
		})
	}
}

func TestSwitchFenceAcknowledgementTransitionOwnership(t *testing.T) {
	for _, row := range []struct {
		name    string
		state   switchRequestState
		receipt bool
		want    switchRequestState
	}{
		{"absent own ID", switchAbsent, false, switchFenced},
		{"uncertain without absence", switchUncertain, false, switchUncertain},
		{"uncertain accepted ID", switchUncertain, true, switchUncertain},
		{"accepted own ID", switchAccepted, true, switchAccepted},
		{"foreign observed ID", switchObservedOther, false, switchObservedOther},
		{"foreign inactive ID", switchObservedInactive, false, switchObservedInactive},
		{"definite refusal", switchRefused, false, switchRefused},
		{"already fenced", switchFenced, false, switchFenced},
	} {
		t.Run(row.name, func(t *testing.T) {
			r := retainedSwitchRequest{state: row.state, requestID: "original"}
			if row.receipt {
				r.receipt = &enginesettings.SwitchReceipt{RequestID: "original", Sequence: 1}
			}
			previous := r.receipt
			effect := r.transition(switchTransition{event: switchFenceOK})
			if r.state != row.want || r.receipt != previous || r.requestID != "original" {
				t.Fatalf("fence transition = %+v", r)
			}
			wantEffect := switchNoRead
			if row.state == switchAbsent {
				wantEffect = switchReadBaseline
			}
			if effect != wantEffect {
				t.Fatalf("effect = %v, want %v", effect, wantEffect)
			}
		})
	}
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

func TestSwitchCurrentObservationsRefreshAtSameSequence(t *testing.T) {
	for _, phase := range []string{enginesettings.SwitchDraining, enginesettings.SwitchCompleted} {
		t.Run(phase, func(t *testing.T) {
			d, _ := switchDetail(t, false)
			d.engineSwitch.requestID = "request"
			r := switchReceipt(d, 7, phase)
			oldTime := time.Date(2026, 10, 21, 1, 0, 0, 0, time.UTC)
			newTime := oldTime.Add(time.Minute)
			running, stopped := true, false
			msg := switchMessage(d, &r)
			msg.status.Observed = &enginesettings.SwitchObserved{
				Source: enginesettings.SwitchEngineState{Running: &running, ObservedAt: &oldTime},
				Target: enginesettings.SwitchEngineState{Running: &running, ObservedAt: &oldTime},
			}
			d.applySwitch(msg)
			retained := d.engineSwitch.receipt
			duplicate := r
			msg = switchMessage(d, &duplicate)
			msg.status.Observed = &enginesettings.SwitchObserved{
				Source: enginesettings.SwitchEngineState{Running: &stopped, ObservedAt: &newTime},
				Target: enginesettings.SwitchEngineState{Running: &stopped, ObservedAt: &newTime},
			}
			d.applySwitch(msg)
			if d.engineSwitch.receipt != retained {
				t.Fatal("equal sequence replaced the retained receipt")
			}
			for name, state := range map[string]enginesettings.SwitchEngineState{
				"From": d.engineSwitch.observed.Source, "To": d.engineSwitch.observed.Target,
			} {
				if state.Running == nil || *state.Running || state.ObservedAt == nil || !state.ObservedAt.Equal(newTime) {
					t.Fatalf("%s current observation did not refresh: %+v", name, state)
				}
			}
		})
	}
}

func TestSwitchFollowNewActiveAfterPreviousCompletion(t *testing.T) {
	for _, previous := range []string{"completed", "inaccessible other client", "known active other client", "completed local with failed readback", "unresolved local submission", "accepted local submission"} {
		t.Run(previous, func(t *testing.T) {
			d, f := switchDetail(t, false)
			s := &d.engineSwitch
			s.requestID = "previous"
			s.state = switchObservedOther
			if previous == "completed" || previous == "completed local with failed readback" {
				r := switchReceipt(d, 7, enginesettings.SwitchCompleted)
				d.applySwitch(switchMessage(d, &r))
				if previous == "completed local with failed readback" {
					s.state = switchAccepted
					s.err = errors.New("receipt unavailable")
				}
			} else if previous == "known active other client" || previous == "accepted local submission" {
				r := switchReceipt(d, 7, enginesettings.SwitchDraining)
				s.receipt = &r
				if previous == "accepted local submission" {
					s.state = switchAccepted
				}
			} else if previous == "unresolved local submission" {
				s.retainedSwitchRequest = retainedSwitchRequest{}
				cmd := d.submitSwitch()
				_, reply := terminalControlsReply(t, f.broker, cmd, &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "reply lost"}})
				readback, _ := d.update(reply)
				_, reply = terminalControlsReply(t, f.broker, readback,
					&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "receipt unavailable"}})
				d.update(reply)
			}
			originalID := s.requestID
			_, reply := terminalControlsReply(t, f.broker, d.switchStatusCmd(true),
				switchResponse(t, enginesettings.SwitchStatus{ConfigurationRevision: "revision", ActiveRequestID: "new-active"}))
			readback, _ := d.update(reply)
			wantID := "new-active"
			retainLocal := previous == "unresolved local submission" || previous == "accepted local submission"
			if retainLocal {
				wantID = originalID
			}
			if s.requestID != wantID {
				t.Fatalf("retained request = %q, want %q", s.requestID, wantID)
			}
			if !retainLocal && (s.receipt != nil || s.observed != nil) {
				t.Fatal("new active request retained previous evidence")
			}
			req, reply := terminalControlsReply(t, f.broker, readback,
				&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "receipt unavailable"}})
			var params enginesettings.SwitchStatusRequest
			if err := json.Unmarshal(req.Params, &params); err != nil {
				t.Fatal(err)
			}
			wantWait := previous == "accepted local submission"
			if req.Method != enginesettings.SwitchStatusMethod || params.RequestID != wantID || (params.AfterSequence != nil) != wantWait {
				t.Fatalf("request-specific readback: %s %+v, want %q", req.Method, params, wantID)
			}
			d.update(reply)
			if !strings.Contains(d.switchView(), "receipt unavailable") {
				t.Fatal("inaccessible receipt did not display explicit unavailability")
			}
		})
	}
}

func TestSwitchFollowNewActiveAfterDefiniteRefusal(t *testing.T) {
	d, f := switchDetail(t, true)
	cmd := d.submitSwitch()
	_, reply := terminalControlsReply(t, f.broker, cmd,
		&rpc.Message{Error: &rpc.RPCError{Code: -32012, Message: "permission revoked before acceptance"}})
	baseline, _ := d.update(reply)
	if d.engineSwitch.outcomeUnknown() {
		t.Fatal("definite refusal still marked unknown")
	}
	batch, ok := baseline().(tea.BatchMsg)
	if !ok {
		t.Fatal("missing capability and status refresh batch")
	}
	var readback tea.Cmd
	for _, child := range batch {
		req, result := terminalControlsReply(t, f.broker, child, switchResponse(t,
			enginesettings.SwitchStatus{ConfigurationRevision: "revision", ActiveRequestID: "new-active"}))
		next, _ := d.update(result)
		if req.Method == enginesettings.SwitchStatusMethod {
			readback = next
		}
	}
	if readback == nil || d.engineSwitch.requestID != "new-active" {
		t.Fatalf("resolved local refusal prevents active readback: request=%q active=%q readback=%t",
			d.engineSwitch.requestID, d.engineSwitch.active, readback != nil)
	}
	r := switchReceipt(d, 3, enginesettings.SwitchCompleted)
	r.Request.SourceEngine, r.Request.TargetEngine = "lmstudio", "ollama"
	req, reply := terminalControlsReply(t, f.broker, readback,
		switchResponse(t, enginesettings.SwitchStatus{Receipt: &r}))
	var params enginesettings.SwitchStatusRequest
	if err := json.Unmarshal(req.Params, &params); err != nil {
		t.Fatal(err)
	}
	if req.Method != enginesettings.SwitchStatusMethod || params.RequestID != "new-active" || params.AfterSequence != nil {
		t.Fatalf("new active readback: %s %+v", req.Method, params)
	}
	d.update(reply)
	s := &d.engineSwitch
	if s.source != "ollama" || s.target != "lmstudio" || s.refusal == nil {
		t.Fatal("adopting active request lost draft or refusal")
	}
	for _, want := range []string{"remote management denied", "permission revoked before acceptance", "Request: new-active", "From lmstudio To ollama", "phase completed"} {
		if !strings.Contains(d.switchView(), want) {
			t.Fatalf("missing %q in switch view: %s", want, d.switchView())
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
	if d.engineSwitch.receipt != nil || !d.engineSwitch.outcomeUnknown() || d.engineSwitch.fresh {
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
	d.engineSwitch.requestID, d.engineSwitch.state = "lost", switchSubmitting
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
	if d.engineSwitch.outcomeUnknown() || d.submitSwitch() != nil || !strings.Contains(d.switchView(), "recoveryRequired=true") {
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
	}{{-32010, "revision conflict"}, {-32011, "local veto"}, {-32012, "remote management denied"}, {-32013, "engine locked"}} {
		t.Run(tc.text, func(t *testing.T) {
			d, _ := switchDetail(t, false)
			d.engineSwitch.requestID = "refused"
			d.engineSwitch.state = switchSubmitting
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
	d.engineSwitch.requestID, d.engineSwitch.state = "accepted-with-lost-reply", switchSubmitting
	msg := switchMessage(d, nil)
	msg.submit, msg.err = true, &rpc.RPCError{Code: -32000, Message: "response read failed"}
	readback := d.applySwitch(msg)
	if !d.engineSwitch.outcomeUnknown() || d.submitSwitch() != nil {
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
	if d.engineSwitch.receipt == nil || d.engineSwitch.outcomeUnknown() {
		t.Fatal("completed original receipt was not recovered")
	}
}

// Each row lists the expected next state for every event below. Terminal and
// recovery evidence are separate rows because neither may become uncertain.
func TestSwitchRetainedRequestTransitionMatrix(t *testing.T) {
	const (
		n = switchNone
		s = switchSubmitting
		f = switchRefused
		u = switchUncertain
		b = switchAbsent
		a = switchAccepted
		o = switchObservedOther
		i = switchObservedInactive
		x = switchFenced
	)
	events := []struct {
		name  string
		event switchEvent
		id    string
	}{
		{"submit", switchSubmit, "new"},
		{"submit ok", switchSubmitOK, ""},
		{"submit definite refusal", switchSubmitRefused, ""},
		{"submit indeterminate error", switchSubmitLost, ""},
		{"retire restart disconnect route", switchRetire, ""},
		{"baseline none", switchBaseline, ""},
		{"baseline same", switchBaseline, "old"},
		{"baseline different", switchBaseline, "new"},
		{"read ok", switchReadOK, ""},
		{"read absent", switchReadAbsent, ""},
		{"read error", switchReadError, ""},
		{"tab reset", switchReset, ""},
		{"close", switchClose, ""},
	}
	rows := []struct {
		name     string
		state    switchRequestState
		phase    string
		recovery bool
		want     []switchRequestState
	}{
		{"none", n, "", false, []switchRequestState{s, n, n, n, n, n, o, o, n, n, n, n, n}},
		{"submitting", s, "", false, []switchRequestState{s, a, f, u, u, s, s, s, a, s, s, n, n}},
		{"refused", f, "", false, []switchRequestState{s, f, f, f, f, f, f, o, a, f, f, n, n}},
		{"uncertain", u, "", false, []switchRequestState{s, u, u, u, u, u, u, u, a, b, u, n, n}},
		{"absent may still be accepted", b, "", false, []switchRequestState{s, b, b, b, u, b, b, b, a, b, b, n, n}},
		{"accepted active", a, enginesettings.SwitchDraining, false, []switchRequestState{s, a, a, a, u, a, a, a, a, a, a, n, n}},
		{"accepted terminal", a, enginesettings.SwitchCompleted, false, []switchRequestState{s, a, a, a, a, a, a, o, a, a, a, n, n}},
		{"accepted recovery", a, enginesettings.SwitchFailedInterrupted, true, []switchRequestState{s, a, a, a, a, a, a, o, a, a, a, n, n}},
		{"uncertain accepted active", u, enginesettings.SwitchDraining, false, []switchRequestState{s, u, u, u, u, u, u, u, a, u, u, n, n}},
		{"observed inaccessible", o, "", false, []switchRequestState{s, o, o, o, o, i, o, o, o, o, o, n, n}},
		{"observed active", o, enginesettings.SwitchDraining, false, []switchRequestState{s, o, o, o, o, i, o, o, o, o, o, n, n}},
		{"observed terminal", o, enginesettings.SwitchCompleted, false, []switchRequestState{s, o, o, o, o, i, o, o, o, o, o, n, n}},
		{"observed recovery", o, enginesettings.SwitchFailedInterrupted, true, []switchRequestState{s, o, o, o, o, i, o, o, o, o, o, n, n}},
		{"observed inactive", i, "", false, []switchRequestState{s, i, i, i, i, i, o, o, i, i, i, n, n}},
		{"fenced", x, "", false, []switchRequestState{s, x, x, x, x, x, x, o, x, x, x, n, n}},
	}
	for _, row := range rows {
		for i, event := range events {
			t.Run(row.name+"/"+event.name, func(t *testing.T) {
				r := retainedSwitchRequest{state: row.state, requestID: "old"}
				if row.state == switchNone {
					r.requestID = ""
				}
				refusal := errors.New("definite refusal")
				if row.state == switchRefused {
					r.refusal = refusal
				}
				if row.phase != "" {
					r.receipt = &enginesettings.SwitchReceipt{RequestID: "old", Sequence: 7,
						Phase: row.phase, Terminal: enginesettings.IsTerminal(row.phase), RecoveryRequired: row.recovery}
				}
				previous := r.receipt
				incoming := &enginesettings.SwitchReceipt{RequestID: "old", Sequence: 7, Phase: enginesettings.SwitchDraining}
				observed := &enginesettings.SwitchObserved{}
				effect := r.transition(switchTransition{event: event.event, id: event.id,
					receipt: incoming, observed: observed, err: refusal})
				if r.state != row.want[i] {
					t.Fatalf("state = %d, want %d", r.state, row.want[i])
				}
				wantID := "old"
				if row.state == switchNone {
					wantID = ""
				}
				adopt := event.event == switchBaseline && (row.state == switchNone ||
					(event.id == "new" && (row.state == switchRefused || row.state == switchFenced || row.state == switchObservedOther || row.state == switchObservedInactive ||
						(row.state == switchAccepted && previous.Terminal))))
				if adopt {
					wantID = event.id
				}
				if event.event == switchSubmit {
					wantID = "new"
				}
				if event.event == switchReset || event.event == switchClose {
					wantID = ""
				}
				if r.requestID != wantID {
					t.Fatalf("identity = %q, want %q", r.requestID, wantID)
				}
				wantEffect := switchNoRead
				if event.event == switchBaseline && wantID != "" && r.state != switchObservedInactive && r.state != switchFenced && (r.state != switchRefused || event.id == wantID) {
					wantEffect = switchReadRequest
				}
				if event.event == switchSubmitRefused && row.state == switchSubmitting {
					wantEffect = switchReadBaseline
				}
				if event.event == switchReadAbsent && (row.state == switchUncertain && previous == nil || row.state == switchAbsent) {
					wantEffect = switchFenceRequest
				}
				if event.event == switchSubmitLost && row.state == switchSubmitting {
					wantEffect = switchReadRequest
				}
				if effect != wantEffect {
					t.Fatalf("effect = %d, want %d", effect, wantEffect)
				}
				if event.event == switchReadOK || event.event == switchSubmitOK && row.state == switchSubmitting {
					if previous != nil && r.receipt != previous {
						t.Fatal("equal sequence replaced receipt")
					}
					if event.event == switchReadOK && row.state != switchNone && row.state != switchFenced && r.observed != observed {
						t.Fatal("equal sequence lost current observation")
					}
				}
				if event.event == switchRetire && previous != nil && r.receipt != previous {
					t.Fatal("retirement lost receipt")
				}
				if row.state == switchRefused && event.event != switchSubmit && event.event != switchReset && event.event != switchClose && r.refusal != refusal {
					t.Fatal("refusal evidence lost")
				}
				assertSwitchIdleReadbackLiveness(t, r)
			})
		}
	}
}

// Check the recovery property after every matrix edge, rather than merely
// approving each edge independently. Idle baselines cannot resolve a sent local
// request or a retained recoveryRequired receipt; neither may offer a new ID.
func assertSwitchIdleReadbackLiveness(t *testing.T, retained retainedSwitchRequest) {
	t.Helper()
	d, f := switchDetail(t, false)
	d.engineSwitch.retainedSwitchRequest = retained
	localRecovery := retained.state == switchSubmitting || retained.state == switchUncertain || retained.state == switchAbsent ||
		(retained.state == switchAccepted && retained.receipt != nil && (!retained.receipt.Terminal || retained.receipt.RecoveryRequired))
	original := retained.requestID
	for cycle := 0; cycle < 2; cycle++ {
		_, reply := terminalControlsReply(t, f.broker, d.switchStatusCmd(true), switchResponse(t,
			enginesettings.SwitchStatus{ConfigurationRevision: "fresh-idle"}))
		read, _ := d.update(reply)
		if localRecovery && read == nil {
			t.Fatal("local recovery is blocked without offering original-ID readback")
		}
		if read != nil {
			req, result := terminalControlsReply(t, f.broker, read,
				&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "Unknown switch receipt for this caller."}})
			var params enginesettings.SwitchStatusRequest
			if err := json.Unmarshal(req.Params, &params); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchStatusMethod || params.RequestID != original {
				t.Fatalf("recovery frame = %s %+v, want original %q", req.Method, params, original)
			}
			next, _ := d.update(result)
			// A failed fence leaves the same recovery obligation. It must not
			// recurse into an automatic retry loop or permit replacement.
			if next != nil {
				req, result = terminalControlsReply(t, f.broker, next,
					&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "fence acknowledgement lost"}})
				var fence enginesettings.SwitchFenceRequest
				if err := json.Unmarshal(req.Params, &fence); err != nil {
					t.Fatal(err)
				}
				if req.Method != enginesettings.SwitchFenceMethod || fence.RequestID != original || !localRecovery || retained.receipt != nil {
					t.Fatalf("unexpected fence = %s %+v", req.Method, fence)
				}
				if further, _ := d.update(result); further != nil {
					t.Fatal("failed fence created an unbounded automatic retry loop")
				}
			}
		}
	}
	if localRecovery {
		if d.submitSwitch() != nil || d.engineSwitch.requestID != original || d.switchReason() == "" {
			t.Fatal("uncertain local submission/recovery lost its exclusive recovery identity")
		}
	} else if reason := d.switchReason(); reason != "" {
		t.Fatalf("fresh idle reads did not reach a non-blocked view: %q, state=%d", reason, d.engineSwitch.state)
	}
}

func TestSwitchDefiniteRefusalAfterRestartWithNoActive(t *testing.T) {
	d, f := switchDetail(t, true)
	_, reply := terminalControlsReply(t, f.broker, d.submitSwitch(),
		&rpc.Message{Error: &rpc.RPCError{Code: -32012, Message: "refused before acceptance"}})
	refresh, _ := d.update(reply)
	batch, ok := refresh().(tea.BatchMsg)
	if !ok {
		t.Fatal("missing refresh batch")
	}
	for _, child := range batch {
		req, result := terminalControlsReply(t, f.broker, child, switchResponse(t,
			enginesettings.SwitchStatus{ConfigurationRevision: "revision"}))
		if req.Method == enginesettings.SwitchStatusMethod {
			d.update(result)
		}
	}
	d.handleNotification(&rpc.Message{Method: "service/broker-restarted"})
	d.handleNotification(&rpc.Message{Method: "app:ready"})
	_, reply = terminalControlsReply(t, f.broker, d.remotePermissionCmd(), switchResponse(t,
		remotePermission{Version: 1, Allowed: true, State: "allowed"}))
	d.update(reply)
	for i := 0; i < 2; i++ {
		_, reply = terminalControlsReply(t, f.broker, d.switchStatusCmd(true), switchResponse(t,
			enginesettings.SwitchStatus{ConfigurationRevision: "current"}))
		readback, _ := d.update(reply)
		if readback != nil {
			t.Fatal("definite refusal requested a nonexistent receipt")
		}
	}
	if reason := d.switchReason(); reason != "" {
		t.Fatalf("definitely refused identity stranded: %q", reason)
	}
	if d.engineSwitch.state != switchRefused || d.engineSwitch.refusal == nil ||
		!strings.Contains(d.switchView(), "refused before acceptance") || d.engineSwitch.source != "ollama" {
		t.Fatal("retirement lost refusal or draft evidence")
	}
}

func TestSwitchAbsentReceiptPreservesUncertaintyAcrossBaselines(t *testing.T) {
	for _, scenario := range []string{"idle", "active same", "active different", "baseline failed", "baseline empty revision", "retired before baseline", "notification during baseline", "generic error"} {
		t.Run(scenario, func(t *testing.T) {
			d, f := switchDetail(t, false)
			_, reply := terminalControlsReply(t, f.broker, d.submitSwitch(),
				&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "reply lost"}})
			readback, _ := d.update(reply)
			original := d.engineSwitch.requestID
			message := "Unknown switch receipt for this caller."
			if scenario == "generic error" {
				message = "receipt unavailable"
			}
			_, reply = terminalControlsReply(t, f.broker, readback, &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: message}})
			fence, _ := d.update(reply)
			if scenario == "generic error" {
				if fence != nil || d.engineSwitch.state != switchUncertain {
					t.Fatal("generic error established absence")
				}
				return
			}
			if fence == nil || d.engineSwitch.state != switchAbsent || d.submitSwitch() != nil {
				t.Fatal("absence did not require a fence")
			}
			req, reply := terminalControlsReply(t, f.broker, fence, &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "fence unavailable"}})
			if req.Method != enginesettings.SwitchFenceMethod {
				t.Fatalf("absence command = %s", req.Method)
			}
			d.update(reply)
			baseline := d.switchStatusCmd(true)
			status := enginesettings.SwitchStatus{ConfigurationRevision: "fresh"}
			switch scenario {
			case "active same":
				status.ActiveRequestID = original
			case "active different":
				status.ActiveRequestID = "other"
			case "baseline empty revision":
				status.ConfigurationRevision = ""
			}
			response := switchResponse(t, status)
			if scenario == "baseline failed" {
				response = &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "unavailable"}}
			}
			if scenario == "notification during baseline" {
				raw, err := json.Marshal(enginesettings.SwitchChanged{NodeID: d.node.key})
				if err != nil {
					t.Fatal(err)
				}
				d.switchChanged(raw)
			}
			req, reply = terminalControlsReply(t, f.broker, baseline, response)
			var params enginesettings.SwitchStatusRequest
			if err := json.Unmarshal(req.Params, &params); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchStatusMethod || params.RequestID != "" {
				t.Fatalf("expected baseline, got %s %+v", req.Method, params)
			}
			if scenario == "retired before baseline" {
				d.retireSwitch()
			}
			d.update(reply)
			if d.engineSwitch.state == switchRefused || d.switchReason() == "" {
				t.Fatalf("%s incorrectly resolved uncertainty", scenario)
			}
			if scenario == "idle" && (!strings.Contains(d.switchView(), "Not accepted as of ") ||
				!strings.Contains(d.switchView(), "; it may still be accepted") || d.engineSwitch.absentAt.IsZero()) {
				t.Fatal("point-in-time absence is not explained")
			}
			if d.engineSwitch.requestID != original {
				t.Fatal("readback replaced original identity")
			}
		})
	}
}

func TestSwitchLateSubmitReplyCannotUndoRefusal(t *testing.T) {
	d, _ := switchDetail(t, false)
	d.engineSwitch.transition(switchTransition{event: switchSubmit, id: "local"})
	refusal := switchMessage(d, nil)
	refusal.submit, refusal.err = true, &rpc.RPCError{Code: -32012, Message: "refused"}
	d.applySwitch(refusal)
	late := switchReceipt(d, 100, enginesettings.SwitchDraining)
	msg := switchMessage(d, &late)
	msg.submit = true
	d.applySwitch(msg)
	if d.engineSwitch.state != switchRefused || d.engineSwitch.receipt != nil {
		t.Fatal("late submit reply replaced definite refusal")
	}
}

// Port of the r4 probe: foreign receipt inaccessibility must not strand the
// view once the real producer clears its active ID on terminal advancement.
func TestSwitchObservedForeignCompletionDoesNotStrandView(t *testing.T) {
	d, f := switchDetail(t, false)
	_, reply := terminalControlsReply(t, f.broker, d.switchStatusCmd(true), switchResponse(t,
		enginesettings.SwitchStatus{ConfigurationRevision: "revision", ActiveRequestID: "foreign"}))
	read, _ := d.update(reply)
	absent := &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "Unknown switch receipt for this caller."}}
	_, reply = terminalControlsReply(t, f.broker, read, absent)
	d.update(reply)
	if d.engineSwitch.state != switchObservedOther || d.engineSwitch.active != "foreign" || d.submitSwitch() != nil {
		t.Fatal("initial foreign activity was not blocked")
	}
	for cycle := 0; cycle < 2; cycle++ {
		_, reply = terminalControlsReply(t, f.broker, d.switchStatusCmd(true), switchResponse(t,
			enginesettings.SwitchStatus{ConfigurationRevision: "new-revision"}))
		read, _ = d.update(reply)
		if read != nil {
			_, reply = terminalControlsReply(t, f.broker, read, absent)
			d.update(reply)
		}
	}
	if reason := d.switchReason(); reason != "" {
		t.Fatalf("foreign activity finished but view remains blocked: reason=%q retained=%q state=%d active=%q",
			reason, d.engineSwitch.requestID, d.engineSwitch.state, d.engineSwitch.active)
	}
	if !strings.Contains(d.switchView(), "Another client's switch is no longer active; its result is not visible to this client") || d.engineSwitch.receipt != nil {
		t.Fatal("inactive foreign activity lost its diagnostic or invented a result")
	}
	_, reply = terminalControlsReply(t, f.broker, d.submitSwitch(),
		&rpc.Message{Error: &rpc.RPCError{Code: -32602, Message: "fixture stops here"}})
	d.update(reply)
}

// Port of r4's actually-sent mutation/route-cancellation probe. A cancelled
// client wait does not cancel the broker handler or fence delayed acceptance.
func TestSwitchCancelledSubmitAbsenceDoesNotProveNonAcceptance(t *testing.T) {
	for _, later := range []string{"active", "readable after completion"} {
		t.Run(later, func(t *testing.T) {
			d, f := switchDetail(t, false)
			cmd := d.submitSwitch()
			original := d.engineSwitch.requestID
			done := make(chan tea.Msg, 1)
			go func() { done <- cmd() }()
			sent, err := f.broker.Read()
			if err != nil {
				t.Fatal(err)
			}
			if sent.Method != enginesettings.SwitchMethod {
				t.Fatalf("method=%q", sent.Method)
			}
			row := d.node
			row.address = "changed-route"
			d.followNode(row, true)
			select {
			case late := <-done:
				d.update(late)
			case <-time.After(time.Second):
				t.Fatal("cancellation did not return")
			}
			_, reply := terminalControlsReply(t, f.broker, d.switchStatusCmd(true), switchResponse(t,
				enginesettings.SwitchStatus{ConfigurationRevision: "fresh"}))
			read, _ := d.update(reply)
			absent := &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "Unknown switch receipt for this caller."}}
			_, reply = terminalControlsReply(t, f.broker, read, absent)
			fence, _ := d.update(reply)
			req, reply := terminalControlsReply(t, f.broker, fence, &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "fence acknowledgement lost"}})
			if req.Method != enginesettings.SwitchFenceMethod {
				t.Fatalf("absence command = %s", req.Method)
			}
			d.update(reply)
			_, reply = terminalControlsReply(t, f.broker, d.switchStatusCmd(true), switchResponse(t,
				enginesettings.SwitchStatus{ConfigurationRevision: "fresh"}))
			read, _ = d.update(reply)
			if read != nil {
				_, reply = terminalControlsReply(t, f.broker, read, absent)
				fence, _ = d.update(reply)
				req, reply = terminalControlsReply(t, f.broker, fence, &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "fence acknowledgement lost"}})
				if req.Method != enginesettings.SwitchFenceMethod {
					t.Fatalf("absence retry = %s", req.Method)
				}
				d.update(reply)
			}
			if replacement := d.submitSwitch(); replacement != nil {
				req, _ := terminalControlsReply(t, f.broker, replacement,
					&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "test stops here"}})
				var params enginesettings.SwitchRequest
				if err := json.Unmarshal(req.Params, &params); err != nil {
					t.Fatal(err)
				}
				t.Fatalf("already sent submission can still be accepted, but absence sent replacement: original=%q replacement=%q method=%s",
					original, params.RequestID, req.Method)
			}
			if d.engineSwitch.requestID != original || d.engineSwitch.state != switchAbsent ||
				d.engineSwitch.refusal != nil || !strings.Contains(d.switchView(), "; it may still be accepted") {
				t.Fatal("absence falsely resolved acceptance or lost recovery identity")
			}
			status := enginesettings.SwitchStatus{ConfigurationRevision: "later"}
			if later == "active" {
				status.ActiveRequestID = original
			}
			_, reply = terminalControlsReply(t, f.broker, d.switchStatusCmd(true), switchResponse(t, status))
			read, _ = d.update(reply)
			r := switchReceipt(d, 7, enginesettings.SwitchCompleted)
			req, reply = terminalControlsReply(t, f.broker, read, switchResponse(t, enginesettings.SwitchStatus{Receipt: &r}))
			var params enginesettings.SwitchStatusRequest
			if err := json.Unmarshal(req.Params, &params); err != nil {
				t.Fatal(err)
			}
			if req.Method != enginesettings.SwitchStatusMethod || params.RequestID != original {
				t.Fatalf("late recovery read wrong identity: %s %+v", req.Method, params)
			}
			d.update(reply)
			if d.engineSwitch.state != switchAccepted || d.engineSwitch.receipt == nil || d.engineSwitch.outcomeUnknown() {
				t.Fatal("late original receipt not recovered")
			}
		})
	}
}

// Port of review r3's retained-identity probe, with explicit expected reads and
// states rather than a trace. Execute every read to check the public RPC frame.
func TestSwitchRetainedIdentityBaselineMatrix(t *testing.T) {
	rows := []struct {
		name                           string
		state                          switchRequestState
		phase                          string
		retire, inaccessible, recovery bool
		retain                         bool
	}{
		{"empty", switchNone, "", false, false, false, false},
		{"local submitting", switchSubmitting, "", false, false, false, true},
		{"local uncertain", switchUncertain, "", false, false, false, true},
		{"local accepted", switchAccepted, enginesettings.SwitchDraining, false, false, false, true},
		{"local accepted inaccessible", switchAccepted, enginesettings.SwitchDraining, false, true, false, true},
		{"local refused", switchRefused, "", false, false, false, false},
		{"local fenced", switchFenced, "", false, false, false, false},
		{"local refused retired", switchRefused, "", true, false, false, false},
		{"local refused inaccessible", switchRefused, "", false, true, false, false},
		{"local terminal", switchAccepted, enginesettings.SwitchCompleted, false, false, false, false},
		{"local terminal inaccessible", switchAccepted, enginesettings.SwitchCompleted, false, true, false, false},
		{"local terminal recovery", switchAccepted, enginesettings.SwitchFailedInterrupted, false, false, true, false},
		{"observed unknown", switchObservedOther, "", false, false, false, false},
		{"observed active", switchObservedOther, enginesettings.SwitchDraining, false, false, false, false},
		{"observed active inaccessible", switchObservedOther, enginesettings.SwitchDraining, false, true, false, false},
		{"observed terminal", switchObservedOther, enginesettings.SwitchCompleted, false, false, false, false},
		{"observed terminal inaccessible", switchObservedOther, enginesettings.SwitchCompleted, false, true, false, false},
		{"observed terminal recovery", switchObservedOther, enginesettings.SwitchFailedInterrupted, false, false, true, false},
	}
	for _, row := range rows {
		for _, baseline := range []struct{ name, id string }{{"none", ""}, {"same", "old"}, {"different", "new"}} {
			t.Run(row.name+"/"+baseline.name, func(t *testing.T) {
				d, f := switchDetail(t, false)
				s := &d.engineSwitch
				s.state, s.requestID = row.state, "old"
				if row.state == switchNone {
					s.requestID = ""
				}
				if row.state == switchRefused {
					s.refusal = &rpc.RPCError{Code: -32012, Message: "definite refusal"}
				}
				if row.phase != "" {
					r := switchReceipt(d, 7, row.phase)
					r.RecoveryRequired = row.recovery
					s.receipt = &r
				}
				if row.inaccessible {
					msg := switchMessage(d, nil)
					msg.err = &rpc.RPCError{Code: -32000, Message: "receipt inaccessible"}
					d.applySwitch(msg)
				}
				if row.retire {
					d.retireSwitch()
				}
				_, reply := terminalControlsReply(t, f.broker, d.switchStatusCmd(true), switchResponse(t,
					enginesettings.SwitchStatus{ConfigurationRevision: "fresh", ActiveRequestID: baseline.id}))
				next, _ := d.update(reply)
				wantID := "old"
				if row.state == switchNone || baseline.id == "new" && !row.retain {
					wantID = baseline.id
				}
				wantRead := wantID != "" && (row.state != switchRefused || baseline.id != "") &&
					!(row.state == switchFenced && baseline.id != "new") &&
					!(row.state == switchObservedOther && baseline.id == "")
				if s.requestID != wantID || (next != nil) != wantRead {
					t.Fatalf("identity=%q read=%t, want identity=%q read=%t", s.requestID, next != nil, wantID, wantRead)
				}
				if wantRead {
					req, result := terminalControlsReply(t, f.broker, next, &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "receipt inaccessible"}})
					var p enginesettings.SwitchStatusRequest
					if err := json.Unmarshal(req.Params, &p); err != nil {
						t.Fatal(err)
					}
					if req.Method != enginesettings.SwitchStatusMethod || p.RequestID != wantID {
						t.Fatalf("unexpected read frame: %s %+v", req.Method, p)
					}
					d.update(result)
					if !strings.Contains(d.switchView(), "receipt inaccessible") {
						t.Fatal("inaccessibility hidden")
					}
				}
				if row.state == switchRefused && s.refusal == nil {
					t.Fatal("refusal lost")
				}
				if row.recovery && baseline.id != "new" && !(row.state == switchObservedOther && baseline.id == "") && !strings.Contains(d.switchReason(), "recovery required") {
					t.Fatal("terminal recovery no longer blocks")
				}
			})
		}
	}
}

// A locked target is refused before anything is sent; a locked source is not
// an extra prohibition on switching to an unlocked engine.
func TestSwitchLockedTargetIsRefusedBeforeSending(t *testing.T) {
	locked, unlocked := true, false
	test := func(name string, source, target *bool, wantReason string) {
		t.Run(name, func(t *testing.T) {
			d, _ := switchDetail(t, false)
			d.engines[0].Locked, d.engines[1].Locked = source, target
			if got := d.switchReason(); got != wantReason {
				t.Fatalf("switchReason = %q, want %q", got, wantReason)
			}
			cmd := d.submitSwitch()
			if (cmd == nil) != (wantReason != "") {
				t.Fatalf("submit command present = %v with reason %q", cmd != nil, wantReason)
			}
			if wantReason != "" && (d.engineSwitch.state == switchSubmitting || d.engineSwitch.requestID != "") {
				t.Fatalf("refused switch started a request: %+v", d.engineSwitch)
			}
		})
	}
	test("locked target", nil, &locked, engineLockedMessage)
	test("locked source", &locked, &unlocked, "")
	test("unreported lock is not a refusal", nil, nil, "")
}
