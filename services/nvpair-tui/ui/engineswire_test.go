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
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"nvpair-shared/engineintent"
	"nvpair-shared/engines"
	"nvpair-shared/enginesettings"
	"nvpair-tui/rpc"
)

// localRequest builds a local model operation's method and params for an
// engine whose family is its own ID, as a built-in engine's is.
func localRequest(t *testing.T, engine, op, model string) (string, map[string]any) {
	t.Helper()
	act, ok := modelActions[op]
	if !ok {
		t.Fatalf("unknown model operation %q", op)
	}
	return localModelRequest(engine, engine, act, model)
}

// TestLocalModelOperationsUseTypedMethods checks load, eject and delete reach
// engine-manager's typed methods with only the engine and the model, for every
// engine and for a variant the shared table has never heard of. Ollama's
// run_model, stream:false and keep_alive:0 now live in its manifest's
// operation bindings, so a second spelling here could only drift from them.
func TestLocalModelOperationsUseTypedMethods(t *testing.T) {
	want := map[string]string{
		"load": "engine:load-model", "unload": "engine:unload-model", "delete": "engine:delete-model",
	}
	ids := append(engines.Names(), "llamacpp-custom")
	for _, engine := range ids {
		for op, method := range want {
			t.Run(engine+"/"+op, func(t *testing.T) {
				got, params := localRequest(t, engine, op, "owner/model")
				if got != method {
					t.Errorf("method = %q, want %q", got, method)
				}
				wantParams := map[string]any{"engine": engine, "model": "owner/model"}
				if !reflect.DeepEqual(params, wantParams) {
					t.Errorf("params = %v, want %v", params, wantParams)
				}
			})
		}
	}
}

// TestPullSendsBothKeys guards the LM Studio fix: a pull must carry the model
// under BOTH "name" (Ollama's /api/pull body key) and "model" (LM Studio's
// `lms get {model}` CLI placeholder). Sending only "name" silently ran
// `lms get "" --yes`, so the download never reached LM Studio. A pull has no
// typed method yet, so it is still engine:action spelled by family.
func TestPullSendsBothKeys(t *testing.T) {
	for _, engine := range []string{engines.NameOllama, engines.NameLMStudio} {
		method, envelope := localRequest(t, engine, "pull", "owner/model")
		if method != "engine:action" {
			t.Fatalf("%s: method = %q, want engine:action", engine, method)
		}
		if envelope["engine"] != engine || envelope["action"] != "pull_model" {
			t.Fatalf("%s: envelope = %v", engine, envelope)
		}
		params, ok := envelope["params"].(map[string]any)
		if !ok {
			t.Fatalf("%s: params = %T, want map[string]any", engine, envelope["params"])
		}
		if params["name"] != "owner/model" || params["model"] != "owner/model" {
			t.Errorf("%s: pull params = %v, want both name and model set", engine, params)
		}
	}
}

// TestPullDeadlineReportsDetachedNotSilence guards the acknowledgement for a
// long download. A multi-gigabyte pull outlasts the reply deadline while the
// engine keeps working, so a failure would be wrong — but the previous silence
// was too, leaving the operator unable to tell a started download from a
// keystroke that missed.
func TestPullDeadlineReportsDetachedNotSilence(t *testing.T) {
	msg := classifyOpResult("download big-model", "ollama", "pull", context.DeadlineExceeded)
	if msg == nil {
		t.Fatal("a pull that outran its deadline produced no message at all")
	}
	op, ok := msg.(engineOpMsg)
	if !ok {
		t.Fatalf("got %T, want engineOpMsg", msg)
	}
	if op.err != nil {
		t.Errorf("a still-running download was reported as failed: %v", op.err)
	}
	if !op.detached {
		t.Error("a still-running download was reported as complete")
	}
}

// TestDeadlineLeniencyTracksOperationLength checks which operations are excused
// for a slow reply.
//
// The set is not "downloads": it is every operation whose duration is set by how
// much data moves or how slow an engine is to become ready. Ollama's load is
// run_model with streaming off, which does not answer until the model is
// resident, so a large model on cold storage exceeds the deadline routinely —
// and reporting "load failed" at the moment the model finishes loading is worse
// than saying nothing. Quick operations get no such excuse, because a deadline
// there is a real fault.
func TestDeadlineLeniencyTracksOperationLength(t *testing.T) {
	// Only operations engineOps actually declares: the TUI offers no engine
	// update, so listing one here would assert against a path nothing reaches.
	// Start and restart wait on the engine's readiness probe, which the
	// manifests allow up to ten minutes, so a deadline on either is a slow
	// reply rather than a failed start.
	for _, op := range []string{"pull", "load", "install", "uninstall", "start", "restart"} {
		result, ok := classifyOpResult("x", "ollama", op, context.DeadlineExceeded).(engineOpMsg)
		if !ok {
			t.Fatalf("%s: unexpected message type", op)
		}
		if !result.detached {
			t.Errorf("%s timing out was reported as a failure, but it is still running", op)
		}
		if result.err != nil {
			t.Errorf("%s carried an error despite still running: %v", op, result.err)
		}
	}

	for _, op := range []string{"unload", "delete", "stop"} {
		result, ok := classifyOpResult("x", "ollama", op, context.DeadlineExceeded).(engineOpMsg)
		if !ok {
			t.Fatalf("%s: unexpected message type", op)
		}
		if result.detached {
			t.Errorf("%s timing out was excused as still running; a quick operation "+
				"that times out has really failed", op)
		}
		if result.err == nil {
			t.Errorf("%s timing out was reported as success", op)
		}
	}

	// A real error is still an error, however long the operation usually takes.
	result, _ := classifyOpResult("x", "ollama", "pull", errors.New("no such model")).(engineOpMsg)
	if result.detached || result.err == nil {
		t.Errorf("a genuine pull error was not reported: %+v", result)
	}
}

// TestPullSpellsByFamily checks a pull is spelled by the engine's family, not
// its ID: a llama.cpp fork pulls like llama.cpp, with the model under "model"
// alone, since llama.cpp would take a second key as a parameter it does not
// accept; an Ollama-family variant sends Ollama's "name".
func TestPullSpellsByFamily(t *testing.T) {
	pull := modelActions["pull"]
	_, envelope := localModelRequest("llamacpp-custom", engines.NameLlamaCPP, pull, "m:Q4")
	if !reflect.DeepEqual(envelope["params"], map[string]any{"model": "m:Q4"}) {
		t.Errorf("llama.cpp fork pull params = %v, want only the model", envelope["params"])
	}
	_, envelope = localModelRequest("ollama-nightly", engines.NameOllama, pull, "m")
	if params, _ := envelope["params"].(map[string]any); params["name"] != "m" {
		t.Errorf("Ollama-family pull params = %v, want name set", envelope["params"])
	}
	_, envelope = localModelRequest("strata", "strata", pull, "m")
	if !reflect.DeepEqual(envelope["params"], map[string]any{"model": "m"}) {
		t.Errorf("unknown-family pull params = %v, want the conventional model key", envelope["params"])
	}
}

// TestDownloadPromptUsesTheEnginesSpelling is the regression guard for an
// Ollama example offered for every engine. "llama3.2" is not a name LM Studio
// or llama.cpp can download, and llama.cpp's needs a quantization after a colon.
// Examples are per family, so a fork shows its family's.
func TestDownloadPromptUsesTheEnginesSpelling(t *testing.T) {
	for _, e := range engines.All() {
		if _, ok := downloadExamples[e.Name]; !ok {
			t.Errorf("%s has no download example", e.Name)
		}
	}
	if got := downloadPrompt(engines.NameLlamaCPP, "llama.cpp (custom)"); !strings.Contains(got, ":Q4_K_M") {
		t.Errorf("llama.cpp-family prompt %q does not show the quantization suffix", got)
	}
	if got := downloadPrompt("vllm", "vLLM"); got != "model name for vLLM" {
		t.Errorf("an engine with no example got %q", got)
	}
}

// TestDescriptorsGateModelOperations checks an operation the engine's
// descriptor does not declare is refused before anything is sent, while an
// engine with no descriptor here (a peer's variant) keeps the operations the
// interface already offered.
func TestDescriptorsGateModelOperations(t *testing.T) {
	ds := engineDescriptors{
		"strata": {
			Version: engines.DescriptorVersion, Engine: "strata", Family: "strata", Label: "Strata",
			ModelNaming: engines.ModelNamingExact,
			Operations: map[string]engines.OperationCapability{
				engines.OperationLoad: {}, engines.OperationUnload: {},
			},
		},
	}
	if err := ds.unsupported("strata", modelActions["load"]); err != nil {
		t.Errorf("declared load refused: %v", err)
	}
	for _, op := range []string{"delete", "pull"} {
		err := ds.unsupported("strata", modelActions[op])
		if err == nil || !strings.Contains(err.Error(), "Strata does not support") {
			t.Errorf("undeclared %s = %v, want refused naming the engine", op, err)
		}
	}
	if err := ds.unsupported("peer-variant", modelActions["delete"]); err != nil {
		t.Errorf("engine without a descriptor refused: %v", err)
	}
	if got := ds.family("strata"); got != "strata" {
		t.Errorf("family = %q", got)
	}
	if got := ds.family(engines.NameOllama); got != engines.NameOllama {
		t.Errorf("an engine without a descriptor is its own family, got %q", got)
	}
}

// TestLongRunningOpsAreRealOperations keeps the leniency set honest: every
// entry must be an operation the interface can actually issue, or the set
// documents behaviour nothing exercises.
func TestLongRunningOpsAreRealOperations(t *testing.T) {
	for op := range longRunningOps {
		_, isLifecycle := engineOps[op]
		_, isModel := modelActions[op]
		if !isLifecycle && !isModel {
			t.Errorf("longRunningOps names %q, which is neither a lifecycle nor a model operation", op)
		}
	}
}

// TestModelActionsCarryRemoteEquivalents checks every model operation has a
// remote method, since all four are offered on a peer's node.
func TestModelActionsCarryRemoteEquivalents(t *testing.T) {
	for name, act := range modelActions {
		if act.op == "" {
			t.Errorf("%s: no operation name", name)
		}
		if act.remote == "" {
			t.Errorf("%s: no remote method, but model operations are offered on remote nodes", name)
		}
		if act.what == "" {
			t.Errorf("%s: no operator-facing verb", name)
		}
	}
}

// TestLifecycleRemoteCoverageMatchesManager pins which lifecycle operations have
// a remote counterpart. The engine manager has remote install, start, and stop
// but no remote restart, uninstall, or port change — those need process
// ownership on the target host — so those three must be marked local-only or the
// UI would offer an operation that always fails.
func TestLifecycleRemoteCoverageMatchesManager(t *testing.T) {
	for name, op := range engineOps {
		_, hasRemote := remoteEngineMethods[op.method]
		if op.localOnly && hasRemote {
			t.Errorf("%s is marked local-only but a remote method exists", name)
		}
		if !op.localOnly && !hasRemote {
			t.Errorf("%s is offered on remote nodes but has no remote method", name)
		}
	}

	for _, name := range []string{"restart", "uninstall"} {
		if !engineOps[name].localOnly {
			t.Errorf("%s must be local-only: the manager exposes no remote variant", name)
		}
	}
	for _, name := range []string{"install", "start", "stop"} {
		if engineOps[name].localOnly {
			t.Errorf("%s has a remote variant and should not be local-only", name)
		}
	}
}

func terminalControlsReply(t *testing.T, broker *rpc.Codec, cmd tea.Cmd, response *rpc.Message) (*rpc.Message, tea.Msg) {
	t.Helper()
	if cmd == nil {
		t.Fatal("missing command")
	}
	done := make(chan tea.Msg, 1)
	go func() { done <- cmd() }()
	request, err := broker.Read()
	if err != nil {
		t.Fatalf("read request: %v", err)
	}
	response.JSONRPC, response.ID = "2.0", request.ID
	if err := broker.Write(response); err != nil {
		t.Fatalf("write response: %v", err)
	}
	select {
	case result := <-done:
		return request, result
	case <-time.After(time.Second):
		t.Fatal("command did not finish")
		return nil, nil
	}
}

func TestTerminalControlsCanonicalFrames(t *testing.T) {
	for _, tc := range []struct{ name, node, op, method string }{
		{"local load", "", "load", "engine:load-model"},
		{"local eject", "", "unload", "engine:unload-model"},
		{"local delete", "", "delete", "engine:delete-model"},
		{"remote load", "peer", "load", "engine:remote-load-model"},
		{"remote eject", "peer", "unload", "engine:remote-unload-model"},
		{"remote delete", "peer", "delete", "engine:remote-delete-model"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			request, result := terminalControlsReply(t, broker, modelCmd(client, tc.node, "ollama", "ollama", modelActions[tc.op], "model"), &rpc.Message{Result: []byte(`{}`)})
			var params map[string]string
			if err := json.Unmarshal(request.Params, &params); err != nil {
				t.Fatal(err)
			}
			want := map[string]string{"engine": "ollama", "model": "model"}
			if tc.node != "" {
				want["node"] = "peer"
			}
			if request.Method != tc.method || !reflect.DeepEqual(params, want) {
				t.Fatalf("frame = %s %v, want %s %v", request.Method, params, tc.method, want)
			}
			if op, ok := result.(engineOpMsg); !ok || op.err != nil || op.detached {
				t.Fatalf("result = %#v", result)
			}
		})
	}
}

func TestTerminalControlsLifecycleFrames(t *testing.T) {
	for _, tc := range []struct{ name, node, method, want string }{
		{"local start", "", "engine:start", "engine:start"},
		{"remote start", "peer", "engine:start", "engine:remote-start"},
		{"remote stop", "peer", "engine:stop", "engine:remote-stop"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			request, _ := terminalControlsReply(t, broker, engineCmd(client, tc.node, "ollama", tc.method, "start", "start"), &rpc.Message{Result: []byte(`{}`)})
			var params map[string]string
			if err := json.Unmarshal(request.Params, &params); err != nil {
				t.Fatal(err)
			}
			want := map[string]string{"engine": "ollama"}
			if tc.node != "" {
				want["node"] = "peer"
			}
			if request.Method != tc.want || !reflect.DeepEqual(params, want) {
				t.Fatalf("frame = %s %v", request.Method, params)
			}
		})
	}
}

func TestTerminalControlsTypedRefusals(t *testing.T) {
	for _, refusal := range []struct {
		name string
		code int
		want string
	}{
		{"veto", -32011, "local veto (-32011): paused on target"},
		{"conflict", -32010, "revision conflict (-32010): paused on target - reload and retry"},
	} {
		for _, family := range []string{"lifecycle", "model", "settings read", "settings preview", "settings apply", "availability"} {
			t.Run(refusal.name+"/"+family, func(t *testing.T) {
				client, broker := nodeInfoRPCFixture(t)
				var cmd tea.Cmd
				switch family {
				case "lifecycle":
					cmd = engineCmd(client, "peer", "ollama", "engine:start", "start", "start")
				case "model":
					cmd = modelCmd(client, "peer", "ollama", "ollama", modelActions["load"], "model")
				case "settings read":
					cmd = getEngineSettingsCmd(client, "peer", "ollama")
				case "settings preview":
					cmd = previewEngineSettingsCmd(client, enginesettings.Request{NodeID: "peer", Engine: "ollama"})
				case "settings apply":
					cmd = applyEngineSettingsCmd(client, enginesettings.Request{NodeID: "peer", Engine: "ollama"})
				case "availability":
					cmd = setAvailabilityCmd(client, availAvailable)
				}
				_, result := terminalControlsReply(t, broker, cmd, &rpc.Message{Error: &rpc.RPCError{Code: refusal.code, Message: "paused on target"}})
				d := newNodeDetail(client, nodeRow{key: "peer", membership: membershipMember})
				if msg, ok := result.(setAvailabilityMsg); ok {
					a := availabilityState{value: availPaused, pending: availAvailable}
					a.update(msg)
					if a.err != refusal.want || a.value != availPaused || a.pending != "" {
						t.Fatalf("availability = %+v", a)
					}
					return
				}
				d.update(result)
				if d.status.kind != toastError || !strings.Contains(d.status.text, refusal.want) {
					t.Fatalf("status = %+v, want %q", d.status, refusal.want)
				}
				if d.settingsAwaited != nil || d.settingsConfirm != nil {
					t.Fatal("refusal armed settings work")
				}
			})
		}
	}
}

func TestTerminalControlsTargetPolicy(t *testing.T) {
	for _, tc := range []struct {
		name   string
		self   bool
		origin string
	}{{"local", true, "local"}, {"peer", false, "peer"}} {
		t.Run(tc.name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			d := newNodeDetail(client, nodeRow{key: "target", self: tc.self, membership: membershipMember})
			request, reply := terminalControlsReply(t, broker, d.policyCmd(), &rpc.Message{Result: []byte(`{"availability":"paused","policy":{"pausedBy":"` + tc.origin + `"}}`)})
			var params map[string]string
			if err := json.Unmarshal(request.Params, &params); err != nil {
				t.Fatal(err)
			}
			want := map[string]string{}
			if !tc.self {
				want["nodeId"] = "target"
			}
			if request.Method != "policy:get" || !reflect.DeepEqual(params, want) {
				t.Fatalf("policy frame = %s %v", request.Method, params)
			}
			d.update(reply)
			if !d.policyFresh || d.policyLabel() != "Availability: paused (paused by "+tc.origin+")" {
				t.Fatalf("policy = %+v, fresh %v", d.policy, d.policyFresh)
			}
			d.SetSize(100, 40)
			if !strings.Contains(d.View(), "paused by "+tc.origin) {
				t.Fatal("target provenance absent from view")
			}
			d.update(detailReply{node: "other", msg: detailPolicyMsg{policy: targetPolicy{Availability: "available"}}})
			if d.policy.Availability != "paused" {
				t.Fatal("wrong-target reply applied")
			}
			if cmd := d.handleNotification(&rpc.Message{Method: "policy:changed", Params: []byte(`{"nodeId":"other","policy":{"pausedBy":"local"}}`)}); cmd != nil || !d.policyFresh {
				t.Fatal("wrong-target push changed policy")
			}
			d.update(DisconnectedMsg{})
			d.update(reply)
			if d.policyFresh || d.policyLabel() != "Availability unavailable" {
				t.Fatal("late reply restored disconnected target")
			}
		})
	}
}

func TestTerminalControlsPolicyRouteLoss(t *testing.T) {
	d := newNodeDetail(nil, nodeRow{key: "peer", membership: membershipMember})
	late := detailReply{node: "peer", msg: detailPolicyMsg{visit: d.telemetryGen, route: d.policyRoute, policy: targetPolicy{Availability: "available"}}}
	d.followNode(nodeRow{}, false)
	d.followNode(nodeRow{key: "peer", membership: membershipMember}, true)
	d.update(late)
	if d.policyFresh {
		t.Fatal("pre-removal reply restored recreated route")
	}
	d.update(detailPolicyMsg{visit: d.telemetryGen, route: d.policyRoute, err: errors.New("read failed")})
	if d.policyFresh {
		t.Fatal("failed read became fresh")
	}
}

func TestTerminalControlsSavedOff(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		intent                map[string]bool
		origins               map[string]engineintent.Origin
		running, remote, want bool
	}{
		{"local Off stopped", map[string]bool{"ollama": false}, map[string]engineintent.Origin{"ollama": engineintent.Local}, false, false, true},
		{"peer Off running", map[string]bool{"ollama": false}, map[string]engineintent.Origin{"ollama": engineintent.Peer}, true, false, true},
		{"Off without origin", map[string]bool{"ollama": false}, map[string]engineintent.Origin{}, false, false, true},
		{"missing intent", nil, nil, false, false, false},
		{"missing entry", map[string]bool{"other": false}, map[string]engineintent.Origin{"other": engineintent.Local}, false, false, false},
		{"saved On stopped", map[string]bool{"ollama": true}, map[string]engineintent.Origin{}, false, false, false},
		{"no peer intent", map[string]bool{"ollama": false}, map[string]engineintent.Origin{"ollama": engineintent.Peer}, false, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := newNodeDetail(nil, nodeRow{self: !tc.remote})
			d.engines = []engineStatus{{Engine: "ollama", Running: tc.running}}
			d.intent = engineintent.Snapshot{EnabledByEngine: tc.intent, OffByEngine: tc.origins}
			d.refreshEngines()
			rows := d.engineTable.Rows()
			if len(rows) != 1 || strings.Contains(rows[0][0], "saved Off") != tc.want {
				t.Fatalf("rows = %v", rows)
			}
			if rows[0][2] != yesNo(tc.running) {
				t.Fatalf("runtime changed: %v", rows[0])
			}
			wantLabel := "ollama"
			if tc.want {
				wantLabel += " (saved Off)"
			}
			if rows[0][0] != wantLabel {
				t.Fatalf("label = %q, want %q", rows[0][0], wantLabel)
			}
			if !reflect.DeepEqual(d.intent.OffByEngine, tc.origins) {
				t.Fatalf("origins = %v, want %v", d.intent.OffByEngine, tc.origins)
			}
		})
	}
}

func TestTerminalControlsLocalPauseProvenance(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	_, msg := terminalControlsReply(t, broker, policyGetCmd(client), &rpc.Message{Result: []byte(`{"availability":"paused","policy":{"pausedBy":"local"}}`)})
	a := availabilityState{selfID: "me"}
	a.update(msg)
	if a.label() != "Paused (local)" {
		t.Fatalf("label = %q", a.label())
	}
	a.update(availNote("other", availAvailable))
	if a.label() != "Paused (local)" {
		t.Fatal("wrong-target push changed local pause")
	}
	a.update(availNote("me", availPaused))
	if a.pausedBy != "" {
		t.Fatal("provenance-less push retained origin")
	}
	a.update(NotificationMsg{Msg: &rpc.Message{Method: "policy:changed", Params: []byte(`{"nodeId":"me","policy":{"pausedBy":"peer"}}`)}})
	if a.label() != "Paused (peer)" {
		t.Fatalf("policy provenance = %q", a.label())
	}
}

func TestTerminalControlsDescriptorRestrictions(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	request, result := terminalControlsReply(t, broker, descriptorsCmd(client), &rpc.Message{Result: []byte(`{"engines":[{"version":1,"engine":"strata","family":"strata","label":"Strata","modelNaming":"exact","operations":{"load":{}}},{"version":999,"engine":"invalid","family":"strata","label":"Invalid","modelNaming":"exact"}]}`)})
	if request.Method != "engine:descriptors" {
		t.Fatalf("method = %s", request.Method)
	}
	msg, ok := result.(descriptorsMsg)
	if !ok || len(msg.descriptors) != 1 {
		t.Fatalf("descriptors = %#v", result)
	}
	d := newNodeDetail(nil, nodeRow{self: true})
	d.descriptors = msg.descriptors
	if d.refuseModelOp("strata", modelActions["load"]) {
		t.Fatal("declared operation refused")
	}
	if !d.refuseModelOp("strata", modelActions["delete"]) || d.status.kind != toastError {
		t.Fatal("undeclared delete offered")
	}
	if _, present := d.descriptors["invalid"]; present {
		t.Fatal("invalid descriptor accepted")
	}
}

func TestTerminalControlsTimeoutContracts(t *testing.T) {
	load := classifyOpResult("load", "ollama", "load", context.DeadlineExceeded).(engineOpMsg)
	if !load.detached || load.err != nil {
		t.Fatalf("load deadline = %+v", load)
	}
	stop := classifyOpResult("stop", "ollama", "stop", context.DeadlineExceeded).(engineOpMsg)
	if stop.detached || !errors.Is(stop.err, context.DeadlineExceeded) {
		t.Fatalf("stop deadline = %+v", stop)
	}
	if got := commandFailure(errors.New("reload before applying")); got != "reload before applying" {
		t.Fatalf("uncoded error rewritten: %q", got)
	}
}

func TestTerminalControlsRestartRefreshUsesExistingTick(t *testing.T) {
	d := newNodeDetail(nil, nodeRow{key: "self", self: true})
	d.handleNotification(&rpc.Message{Method: "app:ready"})
	if !d.policyRefresh {
		t.Fatal("restart did not request a status refresh")
	}
	cmd, _ := d.update(TickMsg{})
	if cmd == nil || d.policyRefresh {
		t.Fatal("existing tick did not consume the status refresh")
	}
	batch, ok := cmd().(tea.BatchMsg)
	if !ok || len(batch) != 2 {
		t.Fatalf("refresh = %T, want policy and intent reads", batch)
	}
}

func TestTerminalControlsLateIntentAfterRestart(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "self", self: true})
	_, old := terminalControlsReply(t, broker, d.intentCmd(), &rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":false},"offByEngine":{"ollama":"local"}}`)})
	d.handleNotification(&rpc.Message{Method: "service/broker-restarted"})
	d.update(old)
	if d.intent.EnabledByEngine != nil || d.intent.OffByEngine != nil {
		t.Fatal("pre-restart intent restored a saved-Off assertion")
	}
	_, current := terminalControlsReply(t, broker, d.intentCmd(), &rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":false},"offByEngine":{"ollama":"peer"}}`)})
	d.update(current)
	if enabled, present := d.intent.EnabledByEngine["ollama"]; !present || enabled || d.intent.OffByEngine["ollama"] != engineintent.Peer {
		t.Fatal("current intent did not replace the retired read")
	}
}

func TestTerminalControlsLateIntentAfterManagerReady(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "self", self: true})
	_, old := terminalControlsReply(t, broker, d.intentCmd(), &rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":false},"offByEngine":{"ollama":"local"}}`)})
	d.update(old)
	if d.intent.OffByEngine["ollama"] != engineintent.Local {
		t.Fatal("positive control did not retain local origin")
	}
	refresh := d.handleNotification(&rpc.Message{Method: "engine:ready"})
	d.update(old)
	if d.intent.EnabledByEngine != nil || d.intent.OffByEngine != nil {
		t.Fatal("pre-manager-restart reply restored intent")
	}
	if refresh == nil {
		t.Fatal("manager readiness did not issue a fresh intent read")
	}
	request, current := terminalControlsReply(t, broker, refresh, &rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":false},"offByEngine":{"ollama":"peer"}}`)})
	if request.Method != "engine:intent" {
		t.Fatalf("method = %q, want engine:intent", request.Method)
	}
	d.update(current)
	want := engineintent.Snapshot{EnabledByEngine: map[string]bool{"ollama": false}, OffByEngine: map[string]engineintent.Origin{"ollama": engineintent.Peer}}
	if !reflect.DeepEqual(d.intent, want) {
		t.Fatalf("intent = %+v, want %+v", d.intent, want)
	}
}

func TestTerminalControlsIntentSnapshotRead(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "self", self: true})
	request, reply := terminalControlsReply(t, broker, d.intentCmd(), &rpc.Message{Result: []byte(`{
		"enabledByEngine":{"ollama":false,"strata":false,"on":true,"unknown":false},
		"offByEngine":{"ollama":"local","strata":"peer"}
	}`)})
	if request.Method != "engine:intent" {
		t.Fatalf("method = %q, want engine:intent", request.Method)
	}
	d.update(reply)
	want := engineintent.Snapshot{
		EnabledByEngine: map[string]bool{"ollama": false, "strata": false, "on": true, "unknown": false},
		OffByEngine:     map[string]engineintent.Origin{"ollama": engineintent.Local, "strata": engineintent.Peer},
	}
	if !reflect.DeepEqual(d.intent, want) {
		t.Fatalf("intent = %+v, want %+v", d.intent, want)
	}
}

func TestTerminalControlsFailedIntentReadClearsSnapshot(t *testing.T) {
	for _, tc := range []struct {
		name  string
		reply rpc.Message
	}{
		{"RPC error", rpc.Message{Error: &rpc.RPCError{Code: -32603, Message: "read failed"}}},
		{"invalid snapshot shape", rpc.Message{Result: []byte(`[]`)}},
		{"missing origins", rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":false}}`)}},
		{"null origins", rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":false},"offByEngine":null}`)}},
		{"missing enabled map", rpc.Message{Result: []byte(`{"offByEngine":{}}`)}},
		{"invalid enabled value", rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":"false"},"offByEngine":{}}`)}},
		{"invalid origin", rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":false},"offByEngine":{"ollama":"other"}}`)}},
		{"origin for On", rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":true},"offByEngine":{"ollama":"local"}}`)}},
		{"unreported origin", rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":false},"offByEngine":{"other":"peer"}}`)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			d := newNodeDetail(client, nodeRow{key: "self", self: true})
			d.engines = []engineStatus{{Engine: "ollama"}}
			_, initial := terminalControlsReply(t, broker, d.intentCmd(), &rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":false},"offByEngine":{"ollama":"peer"}}`)})
			d.update(initial)
			if d.intent.OffByEngine["ollama"] != engineintent.Peer {
				t.Fatal("positive control did not retain peer origin")
			}
			_, reply := terminalControlsReply(t, broker, d.intentCmd(), &tc.reply)
			scoped, ok := reply.(detailReply)
			if !ok {
				t.Fatalf("reply = %T, want detailReply", reply)
			}
			result, ok := scoped.msg.(detailIntentMsg)
			if !ok || result.err == nil {
				t.Fatalf("result = %+v, want intent error", scoped.msg)
			}
			d.update(reply)
			if d.intent.EnabledByEngine != nil || d.intent.OffByEngine != nil {
				t.Fatalf("failed read retained partial intent: %+v", d.intent)
			}
			if strings.Contains(d.engineTable.Rows()[0][0], "saved Off") {
				t.Fatal("failed read retained saved-Off row")
			}
		})
	}
}

func TestTerminalControlsIntentInvalidationRows(t *testing.T) {
	for _, event := range []string{"disconnect", "broker restart", "intent changed"} {
		t.Run(event, func(t *testing.T) {
			d := newNodeDetail(nil, nodeRow{key: "self", self: true})
			d.SetSize(100, 40)
			d.engines = []engineStatus{{Engine: "ollama", Installed: true}}
			initial := detailIntentMsg{visit: d.telemetryGen, generation: d.intentGeneration, intent: engineintent.Snapshot{
				EnabledByEngine: map[string]bool{"ollama": false},
				OffByEngine:     map[string]engineintent.Origin{"ollama": engineintent.Local},
			}}
			d.update(initial)
			if d.intent.OffByEngine["ollama"] != engineintent.Local {
				t.Fatal("positive control did not retain local origin")
			}
			if !strings.Contains(d.View(), "saved Off") {
				t.Fatal("positive control did not render reported Off")
			}
			switch event {
			case "disconnect":
				d.update(DisconnectedMsg{})
			case "broker restart":
				d.handleNotification(&rpc.Message{Method: "service/broker-restarted"})
			case "intent changed":
				d.handleNotification(&rpc.Message{Method: "engine:intent-changed"})
			}
			d.update(initial)
			if d.intent.EnabledByEngine != nil || d.intent.OffByEngine != nil {
				t.Fatal("intent was not cleared")
			}
			if strings.Contains(d.engineTable.Rows()[0][0], "saved Off") || strings.Contains(d.View(), "saved Off") {
				t.Fatal("retired intent remains in cached rows or rendered view")
			}
		})
	}
}

func TestTerminalControlsPolicyPushPreservesIntentRead(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "self", self: true})
	_, reply := terminalControlsReply(t, broker, d.intentCmd(), &rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":false},"offByEngine":{"ollama":"peer"}}`)})
	d.handleNotification(&rpc.Message{Method: "policy:changed", Params: []byte(`{"nodeId":"self","policy":{"pausedBy":"local"}}`)})
	d.update(reply)
	if enabled, present := d.intent.EnabledByEngine["ollama"]; !present || enabled || d.intent.OffByEngine["ollama"] != engineintent.Peer {
		t.Fatal("unrelated policy push discarded the current intent read")
	}
	_, current := terminalControlsReply(t, broker, d.intentCmd(), &rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":true},"offByEngine":{}}`)})
	d.update(current)
	if enabled, present := d.intent.EnabledByEngine["ollama"]; !present || !enabled || len(d.intent.OffByEngine) != 0 {
		t.Fatal("current On reply was not applied")
	}
}

func TestTerminalControlsIntentReadsUseNewestRequest(t *testing.T) {
	for _, tc := range []struct {
		name    string
		current string
		want    engineintent.Snapshot
	}{
		{"On clears origin", `{"enabledByEngine":{"ollama":true},"offByEngine":{}}`, engineintent.Snapshot{
			EnabledByEngine: map[string]bool{"ollama": true},
			OffByEngine:     map[string]engineintent.Origin{},
		}},
		{"newer origin with unchanged boolean", `{"enabledByEngine":{"ollama":false},"offByEngine":{"ollama":"peer"}}`, engineintent.Snapshot{
			EnabledByEngine: map[string]bool{"ollama": false},
			OffByEngine:     map[string]engineintent.Origin{"ollama": engineintent.Peer},
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			d := newNodeDetail(client, nodeRow{key: "self", self: true})
			_, old := terminalControlsReply(t, broker, d.intentCmd(), &rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":false},"offByEngine":{"ollama":"local"}}`)})
			_, current := terminalControlsReply(t, broker, d.intentCmd(), &rpc.Message{Result: []byte(tc.current)})
			d.update(current)
			d.update(old)
			if !reflect.DeepEqual(d.intent, tc.want) {
				t.Fatalf("older request replaced the newest reported intent: %+v, want %+v", d.intent, tc.want)
			}
		})
	}
}

func TestTerminalControlsReopenedNodeRejectsOldVisit(t *testing.T) {
	for _, family := range []string{"policy", "intent"} {
		t.Run(family, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			row := nodeRow{key: "self", self: true}
			oldDetail := newNodeDetail(client, row)
			oldCmd := oldDetail.policyCmd()
			oldBody := `{"availability":"paused","policy":{"pausedBy":"local"}}`
			if family == "intent" {
				oldCmd = oldDetail.intentCmd()
				oldBody = `{"enabledByEngine":{"ollama":false},"offByEngine":{"ollama":"local"}}`
			}
			_, oldReply := terminalControlsReply(t, broker, oldCmd, &rpc.Message{Result: []byte(oldBody)})
			oldDetail.update(DisconnectedMsg{})
			currentDetail := newNodeDetail(client, row)
			currentCmd := currentDetail.policyCmd()
			currentBody := `{"availability":"available","policy":{}}`
			if family == "intent" {
				currentCmd = currentDetail.intentCmd()
				currentBody = `{"enabledByEngine":{"ollama":false},"offByEngine":{"ollama":"peer"}}`
			}
			_, currentReply := terminalControlsReply(t, broker, currentCmd, &rpc.Message{Result: []byte(currentBody)})
			currentDetail.update(currentReply)
			currentDetail.update(oldReply)
			if family == "policy" {
				if !currentDetail.policyFresh || currentDetail.policy.Availability != "available" || currentDetail.policy.Policy.PausedBy != "" {
					t.Fatalf("previous visit replaced current policy: %+v", currentDetail.policy)
				}
			} else if enabled, present := currentDetail.intent.EnabledByEngine["ollama"]; !present || enabled || currentDetail.intent.OffByEngine["ollama"] != engineintent.Peer {
				t.Fatal("previous visit replaced current intent")
			}
		})
	}
}
