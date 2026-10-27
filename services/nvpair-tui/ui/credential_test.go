// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"bytes"
	"encoding/json"
	"log"
	"log/slog"
	"strings"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-tui/rpc"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
)

const credentialTestKey = "test-key-123"

func credentialDetail(t *testing.T, client *rpc.Client) *nodeDetail {
	t.Helper()
	d := newNodeDetail(client, nodeRow{key: "local", self: true, presence: presenceOnline})
	d.engines = []engineStatus{{Engine: "lmstudio", DisplayName: "LM Studio"}, {Engine: "ollama"}}
	d.refreshEngines()
	d.SetSize(160, 50)
	d.credential = detailCredential{engine: "lmstudio", known: true,
		state: engines.CredentialState{Supported: true}}
	t.Cleanup(d.close)
	return d
}

func credentialDraft(t *testing.T, d *nodeDetail) {
	t.Helper()
	if d.editCredential() == nil || d.mode != detailInputCredential {
		t.Fatal("masked field did not open")
	}
	d.input.SetValue(credentialTestKey)
}

func assertCredentialInputDropped(t *testing.T, d *nodeDetail) {
	t.Helper()
	if d.input.Value() != "" || d.mode == detailInputCredential {
		t.Fatal("editable credential survived its lifetime")
	}
	if strings.Contains(d.View(), credentialTestKey) || strings.Contains(d.status.text, credentialTestKey) {
		t.Fatal("credential reached rendered output or status")
	}
}

func assertCredentialRequestDropped(t *testing.T, r *credentialRequest) {
	t.Helper()
	if r == nil {
		t.Fatal("missing request under test")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.params != nil || r.cancel != nil || r.ctx.Err() == nil {
		t.Fatal("request retained parameters or an active context")
	}
}

func TestCredentialRequests(t *testing.T) {
	test := func(name, method, value, response, label string) {
		t.Run(name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			d := credentialDetail(t, client)
			var cmd tea.Cmd
			if method == engines.CredentialStateMethod {
				cmd = d.readCredential()
			} else if value == "" {
				credentialDraft(t, d)
				cmd, _ = d.handleKey(tea.KeyMsg{Type: tea.KeyCtrlD})
			} else {
				credentialDraft(t, d)
				cmd, _ = d.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
			}
			r := d.credential.request
			assertCredentialInputDropped(t, d)
			request, reply := terminalControlsReply(t, broker, cmd, &rpc.Message{Result: []byte(response)})
			var params map[string]string
			if json.Unmarshal(request.Params, &params) != nil {
				t.Fatal("could not decode credential request")
			}
			if request.Method != method || params["engine"] != "lmstudio" {
				t.Fatal("credential request method or engine mismatch")
			}
			if method == engines.CredentialStateMethod {
				if len(params) != 1 {
					t.Fatal("state request has unexpected parameters")
				}
			} else if got, present := params["credential"]; !present || got != value || len(params) != 2 {
				t.Fatal("mutation request has incorrect credential parameters")
			}
			assertCredentialRequestDropped(t, r)
			d.update(reply)
			if d.credential.request != nil || d.credentialLabel() != label {
				t.Fatal("reply did not publish the confirmed credential state")
			}
			assertCredentialInputDropped(t, d)
		})
	}
	test("state", engines.CredentialStateMethod, "", `{"supported":true,"configured":true}`, "Key saved")
	test("Save", engines.SetCredentialMethod, credentialTestKey, `{"supported":true,"configured":true}`, "Key saved")
	test("Clear sends explicit empty credential", engines.SetCredentialMethod, "", `{"supported":true,"configured":false}`, "No key saved")
	test("Save uses service state rather than guessing", engines.SetCredentialMethod, credentialTestKey, `{"supported":true,"configured":false}`, "No key saved")
}

func TestCredentialMaskedInput(t *testing.T) {
	d := credentialDetail(t, nil)
	credentialDraft(t, d)
	if d.input.EchoMode != textinput.EchoPassword || !d.CapturingInput() {
		t.Fatal("credential input is not masked and transient")
	}
	if strings.Contains(d.input.View(), credentialTestKey) || strings.Contains(d.View(), credentialTestKey) {
		t.Fatal("rendered credential input exposed the key")
	}
	if !strings.Contains(d.input.View(), strings.Repeat("*", len(credentialTestKey))) {
		t.Fatal("masked input did not render the entered characters")
	}
	if d.settingsEngine != "" || d.settingsWanted != nil || d.settingsConfirm != nil || d.settingsAwaited != nil {
		t.Fatal("credential entry used the settings lifecycle")
	}
}

func TestCredentialRemoteRefusedBeforeSend(t *testing.T) {
	if method, params, err := setCredentialRequest("peer", "lmstudio", credentialTestKey); err == nil || method != "" || params != nil {
		t.Fatal("remote mutation built a sendable request")
	}
	if method, params, err := credentialStateRequest("peer", "lmstudio"); err == nil || method != "" || params != nil {
		t.Fatal("remote state read built a sendable request")
	}
	client, broker := nodeInfoRPCFixture(t)
	d := credentialDetail(t, client)
	d.node.self = false
	if d.editCredential() != nil || d.readCredential() != nil || d.changeCredential(credentialTestKey) != nil {
		t.Fatal("remote credential action reached command creation")
	}
	if d.credential.request != nil || d.mode == detailInputCredential {
		t.Fatal("remote credential action retained input or a request")
	}
	for _, binding := range d.Help() {
		if binding.Help().Desc == "API key" || binding.Help().Desc == "Clear key" || binding.Help().Desc == "retry key state" {
			t.Fatal("remote help offered credential controls")
		}
	}
	if strings.Contains(d.View(), "API key:") {
		t.Fatal("remote detail displayed a credential control")
	}
	// A subsequent ordinary call is a positive transport control. No refused
	// credential operation may have written a frame ahead of it.
	frame, _ := terminalControlsReply(t, broker,
		call(client, "ping", nil, func(*rpc.Message, error) tea.Msg { return nil }),
		&rpc.Message{Result: []byte(`{}`)})
	if frame.Method != "ping" {
		t.Fatal("a refused credential operation wrote to the transport")
	}
}

func TestCredentialUnknownUnavailableAndRetry(t *testing.T) {
	for _, tc := range []struct {
		name  string
		reply *rpc.Message
	}{
		{"store unavailable", &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: credentialTestKey}}},
		{"missing state", &rpc.Message{Result: []byte(`{}`)}},
		{"null state", &rpc.Message{Result: []byte(`null`)}},
		{"missing configured", &rpc.Message{Result: []byte(`{"supported":true}`)}},
		{"malformed state", &rpc.Message{Result: []byte(`{"supported":true,"configured":"false"}`)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			d := credentialDetail(t, client)
			d.credential = detailCredential{engine: "lmstudio"}
			if d.credentialLabel() != "Credential state unknown" {
				t.Fatal("unknown state was presented as an absent key")
			}
			_, reply := terminalControlsReply(t, broker, d.readCredential(), tc.reply)
			d.update(reply)
			if d.credentialLabel() != "Credential store unavailable" || d.credential.known {
				t.Fatal("failed initial state became a confirmed absent key")
			}
			assertCredentialInputDropped(t, d)
			cmd := d.handleEngineKey(tea.KeyMsg{Type: tea.KeyCtrlR})
			frame, reply := terminalControlsReply(t, broker, cmd, &rpc.Message{Result: []byte(`{"supported":true,"configured":true}`)})
			if frame.Method != engines.CredentialStateMethod {
				t.Fatal("retry repeated a mutation")
			}
			d.update(reply)
			if d.credentialLabel() != "Key saved" || d.credential.unavailable || d.mode != detailInputNone {
				t.Fatal("retry did not recover state without reopening input")
			}
		})
	}
}

func TestCredentialHelpUsesUnshiftedControls(t *testing.T) {
	test := func(name string, state detailCredential, wantEntry bool) {
		t.Run(name, func(t *testing.T) {
			d := credentialDetail(t, nil)
			state.engine = "lmstudio"
			d.credential = state
			var retry, entry, clear bool
			for _, binding := range d.Help() {
				keys := binding.Keys()
				for _, k := range keys {
					if len(k) == 1 && k >= "A" && k <= "Z" {
						t.Fatal("credential help requires Shift for a letter")
					}
				}
				switch binding.Help().Desc {
				case "retry key state":
					retry = len(keys) == 1 && keys[0] == "ctrl+r"
				case "API key":
					entry = len(keys) == 1 && keys[0] == "c"
				case "Clear key":
					clear = len(keys) == 1 && keys[0] == "delete"
				}
			}
			if !retry || entry != wantEntry || clear != wantEntry {
				t.Fatal("credential help does not match the confirmed support state")
			}
		})
	}
	test("unknown", detailCredential{}, false)
	test("unavailable", detailCredential{unavailable: true}, false)
	test("supported", detailCredential{known: true, state: engines.CredentialState{Supported: true}}, true)
	test("unsupported", detailCredential{known: true}, false)
}

func TestCredentialClearFromEnginesPane(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := credentialDetail(t, client)
	d.credential.state.Configured = true
	cmd, handled := d.handleKey(tea.KeyMsg{Type: tea.KeyDelete})
	if !handled {
		t.Fatal("Engines pane did not handle Clear")
	}
	r := d.credential.request
	frame, reply := terminalControlsReply(t, broker, cmd,
		&rpc.Message{Result: []byte(`{"supported":true,"configured":false}`)})
	var params map[string]string
	if json.Unmarshal(frame.Params, &params) != nil {
		t.Fatal("could not decode Clear request")
	}
	value, present := params["credential"]
	if frame.Method != engines.SetCredentialMethod || params["engine"] != "lmstudio" ||
		!present || value != "" || len(params) != 2 {
		t.Fatal("Engines pane Clear did not send the explicit empty credential")
	}
	d.update(reply)
	if d.credentialLabel() != "No key saved" {
		t.Fatal("Clear did not retain the service-confirmed state")
	}
	assertCredentialInputDropped(t, d)
	assertCredentialRequestDropped(t, r)
}

func TestCredentialFailedMutationsKeepConfirmedState(t *testing.T) {
	test := func(name string, configured bool, value string) {
		t.Run(name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			d := credentialDetail(t, client)
			d.credential.state.Configured = configured
			label := d.credentialLabel()
			credentialDraft(t, d)
			cmd := d.changeCredential(value)
			r := d.credential.request
			_, reply := terminalControlsReply(t, broker, cmd, &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: credentialTestKey}})
			d.update(reply)
			if d.credentialLabel() != label || !d.credential.known || !d.credential.unavailable || d.status.kind != toastError {
				t.Fatal("failed mutation replaced confirmed state or hid its failure")
			}
			assertCredentialInputDropped(t, d)
			assertCredentialRequestDropped(t, r)
			if d.credential.request != nil {
				t.Fatal("failed mutation retained a retry request")
			}
		})
	}
	test("failed Set retains absent state", false, credentialTestKey)
	test("failed replace retains saved state", true, credentialTestKey)
	test("failed Delete retains saved state", true, "")
}

func TestCredentialEmptySaveDoesNotClear(t *testing.T) {
	d := credentialDetail(t, nil)
	d.editCredential()
	if cmd, _ := d.handleKey(tea.KeyMsg{Type: tea.KeyEnter}); cmd != nil || d.credential.request != nil {
		t.Fatal("empty Save sent a mutation")
	}
	assertCredentialInputDropped(t, d)
}

func TestCredentialUnsupportedDoesNotOfferEntry(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := credentialDetail(t, client)
	_, reply := terminalControlsReply(t, broker, d.readCredential(), &rpc.Message{Result: []byte(`{"supported":false,"configured":false}`)})
	d.update(reply)
	if d.credentialLabel() != "Credentials not supported" || d.editCredential() != nil || d.changeCredential(credentialTestKey) != nil {
		t.Fatal("unsupported engine offered credential entry")
	}
}

func credentialExitCases() []struct {
	name string
	exit func(*nodeDetail)
} {
	return []struct {
		name string
		exit func(*nodeDetail)
	}{
		{"Escape", func(d *nodeDetail) { d.handleKey(tea.KeyMsg{Type: tea.KeyEsc}) }},
		{"engine changed by inventory", func(d *nodeDetail) {
			d.update(detailEnginesMsg{engines: []engineStatus{{Engine: "ollama"}}})
		}},
		{"node identity changed", func(d *nodeDetail) {
			d.followNode(nodeRow{key: "another-local-node", self: true, presence: presenceOnline}, true)
		}},
		{"node removed", func(d *nodeDetail) { d.followNode(nodeRow{}, false) }},
		{"detail closed", func(d *nodeDetail) { d.close() }},
		{"tab changed", func(d *nodeDetail) {
			v := newNodesView(d.client)
			v.detail = d
			m := New(d.client, nil, []View{v, newLogsView(d.client)})
			m.selectTab(1)
		}},
		{"application cleanup", func(d *nodeDetail) {
			v := newNodesView(d.client)
			v.detail = d
			closeViews([]View{v})
		}},
		{"service disconnected", func(d *nodeDetail) { d.update(DisconnectedMsg{}) }},
		{"broker restarted", func(d *nodeDetail) {
			d.update(NotificationMsg{Msg: &rpc.Message{Method: "service/broker-restarted"}})
		}},
	}
}

func TestCredentialInputExits(t *testing.T) {
	for _, tc := range credentialExitCases() {
		t.Run(tc.name, func(t *testing.T) {
			d := credentialDetail(t, nil)
			credentialDraft(t, d)
			tc.exit(d)
			assertCredentialInputDropped(t, d)
		})
	}
}

func TestCredentialQueuedRequestExits(t *testing.T) {
	for _, tc := range credentialExitCases() {
		t.Run(tc.name, func(t *testing.T) {
			d := credentialDetail(t, nil)
			credentialDraft(t, d)
			cmd := d.saveCredential()
			r := d.credential.request
			tc.exit(d)
			assertCredentialInputDropped(t, d)
			assertCredentialRequestDropped(t, r)
			if d.credential.request == r {
				t.Fatal("detail retained a retired request")
			}
			before := d.credential.state
			d.update(cmd())
			d.update(credentialMsg{request: r, state: engines.CredentialState{Supported: true, Configured: true}})
			if d.credential.state != before || d.mode != detailInputNone {
				t.Fatal("retired queued reply changed state or reopened input")
			}
		})
	}
}

func TestCredentialInFlightRequestExits(t *testing.T) {
	for _, tc := range credentialExitCases() {
		t.Run(tc.name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			d := credentialDetail(t, client)
			credentialDraft(t, d)
			cmd := d.saveCredential()
			r := d.credential.request
			done := make(chan tea.Msg, 1)
			go func() { done <- cmd() }()
			frame, err := broker.Read()
			if err != nil || frame == nil || frame.Method != engines.SetCredentialMethod {
				t.Fatal("in-flight fixture did not receive the mutation")
			}
			tc.exit(d)
			assertCredentialInputDropped(t, d)
			assertCredentialRequestDropped(t, r)
			var reply tea.Msg
			select {
			case reply = <-done:
			case <-time.After(time.Second):
				t.Fatal("cancel did not end CallGeneration")
			}
			before := d.credential.state
			// A response arriving after cancellation must not be applied.
			if broker.Write(&rpc.Message{JSONRPC: "2.0", ID: frame.ID, Result: []byte(`{"supported":true,"configured":true}`)}) != nil {
				t.Fatal("could not deliver late response")
			}
			d.update(reply)
			d.update(credentialMsg{request: r, state: engines.CredentialState{Supported: true, Configured: true}})
			if d.credential.state != before || d.credential.request == r || d.mode != detailInputNone {
				t.Fatal("late response changed the selected state or reopened input")
			}
		})
	}
}

func TestCredentialEngineCursorChangeDropsRequest(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := credentialDetail(t, client)
	credentialDraft(t, d)
	staleCmd := d.saveCredential()
	r := d.credential.request
	cmd := d.handleEngineKey(tea.KeyMsg{Type: tea.KeyDown})
	if e := d.selectedEngine(); e == nil || e.Engine != "ollama" || d.credential.engine != "ollama" || d.credential.known {
		t.Fatal("engine selection did not retire credential state")
	}
	assertCredentialRequestDropped(t, r)
	assertCredentialInputDropped(t, d)
	_, current := terminalControlsReply(t, broker, cmd,
		&rpc.Message{Result: []byte(`{"supported":false,"configured":false}`)})
	d.update(current)
	d.update(staleCmd())
	d.update(credentialMsg{request: r, state: engines.CredentialState{Supported: true, Configured: true}})
	if d.credentialLabel() != "Credentials not supported" || d.mode != detailInputNone {
		t.Fatal("late mutation changed a different engine's state")
	}
}

func TestCredentialLateReplyDoesNotCloseNewInput(t *testing.T) {
	d := credentialDetail(t, nil)
	credentialDraft(t, d)
	stale := d.saveCredential()
	d.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	d.editCredential()
	d.update(stale())
	if d.mode != detailInputCredential || d.input.Value() != "" || d.credential.state.Configured {
		t.Fatal("late reply changed the newly opened input or state")
	}
}

func TestCredentialLateReplyDoesNotAffectReopenedDetail(t *testing.T) {
	first := credentialDetail(t, nil)
	credentialDraft(t, first)
	stale := first.saveCredential()
	r := first.credential.request
	first.close()
	second := credentialDetail(t, nil)
	credentialDraft(t, second)
	second.update(stale())
	second.update(credentialMsg{request: r, state: engines.CredentialState{Supported: true, Configured: true}})
	if second.mode != detailInputCredential || second.input.Value() != credentialTestKey || second.credential.state.Configured {
		t.Fatal("an old detail's reply changed the reopened detail")
	}
	assertCredentialRequestDropped(t, r)
}

func TestCredentialInvalidSaveDropsInputBeforeSend(t *testing.T) {
	d := credentialDetail(t, nil)
	credentialDraft(t, d)
	d.input.SetValue(credentialTestKey + " ")
	if cmd := d.saveCredential(); cmd != nil || d.credential.request != nil {
		t.Fatal("invalid Save created an RPC request")
	}
	assertCredentialInputDropped(t, d)
}

func TestCredentialApplicationExitRoutes(t *testing.T) {
	for _, tc := range []struct {
		name string
		msg  tea.Msg
	}{
		{"interrupt", tea.KeyMsg{Type: tea.KeyCtrlC}},
		{"quit", tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("q")}},
		{"confirmed stop", tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("y")}},
		{"service stop", stopServiceMsg{}},
		{"data reset", wipeDataMsg{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := credentialDetail(t, nil)
			credentialDraft(t, d)
			// Input captures ordinary letters. Pending calls permit the shell's
			// normal quit routes; interrupt also works with the field still open.
			var r *credentialRequest
			if tc.name != "interrupt" {
				d.saveCredential()
				r = d.credential.request
			}
			v := newNodesView(nil)
			v.detail = d
			views := []View{v}
			m := New(nil, nil, views)
			m.confirmingStop = tc.name == "confirmed stop"
			_, cmd := m.Update(tc.msg)
			if cmd == nil {
				t.Fatal("exit route did not request application shutdown")
			}
			if _, ok := cmd().(tea.QuitMsg); !ok {
				t.Fatal("exit route did not produce Quit")
			}
			// ui.Run invokes this after the loop ends, including interrupt/error.
			closeViews(views)
			assertCredentialInputDropped(t, d)
			if r != nil {
				assertCredentialRequestDropped(t, r)
			}
		})
	}
}

func TestCredentialRepliesDoNotExposeKey(t *testing.T) {
	var output bytes.Buffer
	previousLog, previousSlog := log.Writer(), slog.Default()
	log.SetOutput(&output)
	slog.SetDefault(slog.New(slog.NewTextHandler(&output, nil)))
	t.Cleanup(func() { slog.SetDefault(previousSlog); log.SetOutput(previousLog) })
	client, broker := nodeInfoRPCFixture(t)
	d := credentialDetail(t, client)
	credentialDraft(t, d)
	_, reply := terminalControlsReply(t, broker, d.saveCredential(),
		&rpc.Message{Result: []byte(`{"supported":true,"configured":true,"echo":"` + credentialTestKey + `"}`)})
	d.update(reply)
	credentialDraft(t, d)
	_, reply = terminalControlsReply(t, broker, d.saveCredential(),
		&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: credentialTestKey}})
	d.update(reply)
	if strings.Contains(output.String(), credentialTestKey) {
		t.Fatal("credential response reached logs")
	}
	assertCredentialInputDropped(t, d)
}
