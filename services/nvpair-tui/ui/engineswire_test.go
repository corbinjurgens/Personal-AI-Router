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
	"nvpair-shared/modeldownload"
	"nvpair-tui/rpc"
)

func tuiDownloadSnapshot() *modeldownload.Snapshot {
	return &modeldownload.Snapshot{
		Version: modeldownload.Version, SourceID: strings.Repeat("a", 32), Revision: "7",
		Entries: []modeldownload.Entry{{ID: strings.Repeat("b", 32), Engine: "ollama", Model: "exact/model",
			Phase: modeldownload.PhaseInProgress, BlocksReadiness: true}},
	}
}

func tuiDownloadJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode download fixture: %v", err)
	}
	return data
}

func TestModelDownloadLocalDecodePreservesInventoryWithoutTrustingMalformedSupport(t *testing.T) {
	valid := string(tuiDownloadJSON(t, tuiDownloadSnapshot()))
	test := func(name, extension string, supported bool) {
		t.Run(name, func(t *testing.T) {
			var got modelsResult
			data := `{"modelsByEngine":{"ollama":["exact/model"]},"loadedByEngine":{"ollama":["exact/model"]}` + extension + `}`
			if err := json.Unmarshal([]byte(data), &got); err != nil {
				t.Fatalf("decode inventory: %v", err)
			}
			if got.current != supported || (got.DownloadState != nil) != supported {
				t.Fatalf("support = %v, snapshot = %+v", got.current, got.DownloadState)
			}
			if !reflect.DeepEqual(got.ModelsByEngine["ollama"], []string{"exact/model"}) ||
				!reflect.DeepEqual(got.LoadedByEngine["ollama"], []string{"exact/model"}) {
				t.Fatalf("inventory/residency lost: %+v", got)
			}
		})
	}
	test("strict shared snapshot", `,"downloadState":`+valid, true)
	test("unsupported", "", false)
	test("null", `,"downloadState":null`, false)
	test("missing readiness boolean", `,"downloadState":`+strings.Replace(valid, `,"blocksReadiness":true`, "", 1), false)
	test("unsupported version", `,"downloadState":`+strings.Replace(valid, `"version":1`, `"version":2`, 1), false)
	test("noncanonical revision", `,"downloadState":`+strings.Replace(valid, `"revision":"7"`, `"revision":"07"`, 1), false)
	test("over byte limit", `,"downloadState":{"padding":"`+strings.Repeat("x", modeldownload.MaxSnapshotBytes)+`"}`, false)
}

func TestModelDownloadDiscoveryDecodeRequiresCurrentObservation(t *testing.T) {
	test := func(name, metadata string, current bool) {
		t.Run(name, func(t *testing.T) {
			data := `{"hostUuid":"peer","modelsByEngine":{"ollama":["exact/model"]},"downloadState":` + string(tuiDownloadJSON(t, tuiDownloadSnapshot())) + metadata + `}`
			var node availableNode
			if err := json.Unmarshal([]byte(data), &node); err != nil {
				t.Fatalf("decode discovery: %v", err)
			}
			rows := mergeNodes(nodeFeeds{discovered: []availableNode{node}})
			if len(rows) != 1 || rows[0].downloadState == nil || !rows[0].downloadState.Equal(*tuiDownloadSnapshot()) {
				t.Fatalf("discovery dropped lifecycle: %+v", rows)
			}
			if got := node.modelSnapshot().current; got != current {
				t.Fatalf("current = %v, want %v", got, current)
			}
		})
	}
	test("current", `,"downloadStateCurrent":true`, true)
	test("stale", `,"downloadStateCurrent":false`, false)
	test("unsupported metadata", "", false)
	test("malformed metadata", `,"downloadStateCurrent":"true"`, false)
	test("null metadata", `,"downloadStateCurrent":null`, false)
}

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
		{"locked", -32013, "engine locked (-32013): paused on target"},
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

// TestEngineStatusLockReadback keeps a reported lock exact and an unreported one
// unknown, through decoding and the local state push that replaces a row.
func TestEngineStatusLockReadback(t *testing.T) {
	decode := func(t *testing.T, raw string) engineStatus {
		t.Helper()
		var e engineStatus
		if err := json.Unmarshal([]byte(raw), &e); err != nil {
			t.Fatalf("decode %s: %v", raw, err)
		}
		return e
	}
	test := func(name, raw string, want *bool) {
		t.Run(name, func(t *testing.T) {
			got := decode(t, raw).Locked
			if (got == nil) != (want == nil) || (got != nil && *got != *want) {
				t.Fatalf("locked = %v, want %v", got, want)
			}
		})
	}
	locked, unlocked := true, false
	test("locked", `{"engine":"ollama","installed":true,"locked":true}`, &locked)
	test("unlocked", `{"engine":"ollama","installed":true,"locked":false}`, &unlocked)
	test("absent stays unknown", `{"engine":"ollama","installed":true}`, nil)

	t.Run("push without lock clears a stale lock", func(t *testing.T) {
		d := localDetail()
		d.engines = []engineStatus{{Engine: "ollama", Installed: true, Locked: &locked}}
		d.handleNotification(&rpc.Message{Method: "engine:state-changed", Params: json.RawMessage(`{"engine":"ollama","installed":true}`)})
		if len(d.engines) != 1 || d.engines[0].Locked != nil {
			t.Fatalf("engines after push = %+v, want lock unknown", d.engines)
		}
	})
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
				switch tc.name {
				case "local Off stopped":
					wantLabel += " (saved Off, local)"
				case "peer Off running":
					wantLabel += " (saved Off, by peer)"
				default:
					wantLabel += " (saved Off)"
				}
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

func TestTerminalControlsOffOrigin(t *testing.T) {
	for _, tc := range []struct {
		name    string
		enabled map[string]bool
		origin  engineintent.Origin
		remote  bool
		want    string
	}{
		{"local origin", map[string]bool{"ollama": false}, engineintent.Local, false, "ollama (saved Off, local)"},
		{"peer origin", map[string]bool{"ollama": false}, engineintent.Peer, false, "ollama (saved Off, by peer)"},
		{"missing origin", map[string]bool{"ollama": false}, "", false, "ollama (saved Off)"},
		{"invalid origin", map[string]bool{"ollama": false}, "other", false, "ollama (saved Off)"},
		{"unknown engine", map[string]bool{"other": false}, engineintent.Local, false, "ollama"},
		{"On with stale origin", map[string]bool{"ollama": true}, engineintent.Local, false, "ollama"},
		{"remote row", map[string]bool{"ollama": false}, engineintent.Local, true, "ollama"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := newNodeDetail(nil, nodeRow{self: !tc.remote})
			d.SetSize(100, 40)
			d.engines = []engineStatus{{Engine: "ollama", Installed: true, Running: true, Healthy: true, Port: 11434}}
			d.intent = engineintent.Snapshot{EnabledByEngine: tc.enabled, OffByEngine: map[string]engineintent.Origin{"ollama": tc.origin}}
			d.refreshEngines()
			want := []string{tc.want, "yes", "yes", "yes", "11434"}
			if !tc.remote {
				want = append(want, "?")
			}
			if got := []string(d.engineTable.Rows()[0]); !reflect.DeepEqual(got, want) {
				t.Fatalf("row = %v, want %v", got, want)
			}
			if !strings.Contains(d.View(), tc.want) {
				t.Fatalf("render missing %q:\n%s", tc.want, d.View())
			}
		})
	}

	for _, event := range []string{"On clearing", "failed read", "disconnect", "broker restart", "manager restart", "late reply after newer generation"} {
		t.Run(event, func(t *testing.T) {
			d := newNodeDetail(nil, nodeRow{self: true})
			d.SetSize(100, 40)
			d.engines = []engineStatus{{Engine: "ollama", Installed: true}}
			old := detailIntentMsg{visit: d.telemetryGen, generation: d.intentGeneration, intent: engineintent.Snapshot{
				EnabledByEngine: map[string]bool{"ollama": false},
				OffByEngine:     map[string]engineintent.Origin{"ollama": engineintent.Local},
			}}
			d.update(old)
			if !strings.Contains(d.View(), "ollama (saved Off, local)") {
				t.Fatal("positive control did not render local Off origin")
			}
			want := "ollama"
			switch event {
			case "On clearing":
				d.update(detailIntentMsg{visit: d.telemetryGen, generation: d.intentGeneration, intent: engineintent.Snapshot{
					EnabledByEngine: map[string]bool{"ollama": true}, OffByEngine: map[string]engineintent.Origin{},
				}})
			case "failed read":
				d.update(detailIntentMsg{visit: d.telemetryGen, generation: d.intentGeneration, err: errors.New("read failed")})
			case "disconnect":
				d.update(DisconnectedMsg{})
				d.update(old)
			case "broker restart":
				d.handleNotification(&rpc.Message{Method: "service/broker-restarted"})
				d.update(old)
			case "manager restart":
				d.handleNotification(&rpc.Message{Method: "engine:ready"})
				d.update(old)
			case "late reply after newer generation":
				d.intentCmd()
				d.update(detailIntentMsg{visit: d.telemetryGen, generation: d.intentGeneration, intent: engineintent.Snapshot{
					EnabledByEngine: map[string]bool{"ollama": false}, OffByEngine: map[string]engineintent.Origin{"ollama": engineintent.Peer},
				}})
				d.update(old)
				want = "ollama (saved Off, by peer)"
			}
			if got := d.engineTable.Rows()[0][0]; got != want {
				t.Fatalf("label = %q, want %q", got, want)
			}
			view := d.View()
			if !strings.Contains(view, want) || strings.Contains(view, "saved Off, local") {
				t.Fatalf("stale origin in render:\n%s", view)
			}
		})
	}

	for _, tc := range []struct {
		name  string
		width int
		label string
	}{
		{"narrow terminal", 80, "ollama"},
		{"normal terminal", 100, "ollama"},
		{"long engine label", 140, "Ollama custom engine with a long display name"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := newNodeDetail(nil, nodeRow{self: true})
			d.SetSize(tc.width, 40)
			d.engines = []engineStatus{{Engine: "other"}, {Engine: "ollama", DisplayName: tc.label, Installed: true, Running: true, Port: 11434}}
			d.intent = engineintent.Snapshot{EnabledByEngine: map[string]bool{"ollama": false}, OffByEngine: map[string]engineintent.Origin{"ollama": engineintent.Peer}}
			d.refreshEngines()
			d.engineTable.SetCursor(1)
			d.refreshEngines()
			d.SetSize(tc.width, 40)
			if d.engineTable.Cursor() != 1 || d.selectedEngine().Engine != "ollama" {
				t.Fatal("refresh lost selected engine")
			}
			view := d.engineTable.View()
			for _, text := range []string{"ENGINE", "INSTALLED", "RUNNING", "HEALTHY", "ENGINE PORT", "ollama", "11434"} {
				if text == "ollama" {
					text = tc.label
				}
				if !strings.Contains(view, text) {
					t.Fatalf("render missing %q:\n%s", text, view)
				}
			}
			if tc.width >= 100 && !strings.Contains(view, tc.label+" (saved Off, by peer)") {
				t.Fatalf("render clipped origin:\n%s", view)
			}
			t.Logf("width %d fixture:\n%s", tc.width, view)
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
	batch, ok := refresh().(tea.BatchMsg)
	if !ok || len(batch) != 2 {
		t.Fatal("manager readiness did not refresh intent and canonical models")
	}
	request, current := terminalControlsReply(t, broker, batch[0], &rpc.Message{Result: []byte(`{"enabledByEngine":{"ollama":false},"offByEngine":{"ollama":"peer"}}`)})
	if request.Method != "engine:intent" {
		t.Fatalf("method = %q, want engine:intent", request.Method)
	}
	d.update(current)
	want := engineintent.Snapshot{EnabledByEngine: map[string]bool{"ollama": false}, OffByEngine: map[string]engineintent.Origin{"ollama": engineintent.Peer}}
	if !reflect.DeepEqual(d.intent, want) {
		t.Fatalf("intent = %+v, want %+v", d.intent, want)
	}
	done := make(chan tea.Msg, 1)
	go func() { done <- batch[1]() }()
	for _, method := range []string{"engine:subscribe", "engine:models"} {
		request, err := broker.Read()
		if err != nil {
			t.Fatalf("read model hydration: %v", err)
		}
		if request.Method != method {
			t.Fatalf("model hydration method = %q, want %q", request.Method, method)
		}
		result := json.RawMessage(`{"subscribed":true}`)
		if method == "engine:models" {
			snapshot := tuiDownloadSnapshot()
			snapshot.Entries = []modeldownload.Entry{}
			result = tuiDownloadJSON(t, tuiDownloadInventory(snapshot))
		}
		if err := broker.Write(&rpc.Message{JSONRPC: "2.0", ID: request.ID, Result: result}); err != nil {
			t.Fatalf("reply model hydration: %v", err)
		}
	}
	select {
	case reply := <-done:
		d.update(reply)
	case <-time.After(time.Second):
		t.Fatal("model hydration did not finish")
	}
	if !d.modelReady("ollama", "exact/model") || !reflect.DeepEqual(d.intent, want) {
		t.Fatal("canonical model hydration lost ready content or refreshed intent")
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

func TestTerminalControlsEngineLock(t *testing.T) {
	locked, unlocked := true, false
	lockedEngines := func() []engineStatus {
		return []engineStatus{{Engine: "ollama", Installed: true, Locked: &locked}, {Engine: "lmstudio", Installed: true, Locked: &unlocked}, {Engine: "llamacpp", Installed: true}}
	}

	t.Run("lock shown on self and peer", func(t *testing.T) {
		for _, self := range []bool{true, false} {
			d := newNodeDetail(nil, nodeRow{self: self})
			d.engines = lockedEngines()
			d.refreshEngines()
			rows := d.engineTable.Rows()
			if rows[0][0] != "ollama (locked on this node)" {
				t.Fatalf("self=%v locked label = %q", self, rows[0][0])
			}
		}
	})
	t.Run("absent lock remains unknown", func(t *testing.T) {
		d := newNodeDetail(nil, nodeRow{self: true})
		d.engines = lockedEngines()
		d.refreshEngines()
		rows := d.engineTable.Rows()
		for _, i := range []int{1, 2} {
			if strings.Contains(rows[i][0], "lock") {
				t.Fatalf("row %d label = %q, want no lock claim", i, rows[i][0])
			}
		}
	})
	t.Run("saved Off is not lock", func(t *testing.T) {
		d := localDetail()
		d.engines = []engineStatus{{Engine: "ollama"}, {Engine: "lmstudio", Locked: &locked}}
		d.intent = engineintent.Snapshot{
			EnabledByEngine: map[string]bool{"ollama": false, "lmstudio": false},
			OffByEngine:     map[string]engineintent.Origin{"ollama": engineintent.Local, "lmstudio": engineintent.Local},
		}
		d.refreshEngines()
		rows := d.engineTable.Rows()
		if rows[0][0] != "ollama (saved Off, local)" || rows[1][0] != "lmstudio (locked on this node) (saved Off, local)" {
			t.Fatalf("labels = %q, %q", rows[0][0], rows[1][0])
		}
		if d.engineLocked("ollama") {
			t.Fatal("saved Off was read as a lock")
		}
		// A saved Off engine is still startable; the target decides.
		if cmd := d.lifecycle(&d.engines[0], "start"); cmd == nil {
			t.Fatalf("saved Off start was refused: %s", d.status.render())
		}
	})
	t.Run("locked activation is disabled", func(t *testing.T) {
		d := localDetail()
		d.engines = lockedEngines()
		d.refreshEngines()
		for _, op := range []string{"start", "restart"} {
			if cmd := d.lifecycle(&d.engines[0], op); cmd != nil {
				t.Fatalf("locked %s sent a command", op)
			}
			if d.status.kind != toastError || !strings.Contains(d.status.text, engineLockedMessage) {
				t.Fatalf("%s status = %+v", op, d.status)
			}
			d.status.info("")
		}
		if !d.refuseModelOp("ollama", modelActions["load"]) || !strings.Contains(d.status.text, engineLockedMessage) {
			t.Fatalf("locked load was not refused: %+v", d.status)
		}
		if d.refuseModelOp("lmstudio", modelActions["load"]) {
			t.Fatal("unlocked engine load was refused")
		}
		if d.refuseModelOp("llamacpp", modelActions["load"]) {
			t.Fatal("unreported lock refused a load; the target decides")
		}
	})
	t.Run("locked stop and unload remain available", func(t *testing.T) {
		d := localDetail()
		d.engines = lockedEngines()
		if cmd := d.lifecycle(&d.engines[0], "stop"); cmd == nil {
			t.Fatalf("locked stop was refused: %s", d.status.render())
		}
		for _, op := range []string{"unload", "delete", "pull"} {
			if d.refuseModelOp("ollama", modelActions[op]) {
				t.Fatalf("locked %s was refused: %s", op, d.status.render())
			}
		}
	})
	t.Run("distinct locked paused conflict and denied refusals", func(t *testing.T) {
		seen := map[string]int{}
		for _, code := range []int{-32013, -32011, -32010, -32012, -32000} {
			text := commandFailure(&rpc.RPCError{Code: code, Message: "target text"})
			if !strings.Contains(text, "target text") {
				t.Fatalf("code %d dropped the target text: %q", code, text)
			}
			if prior, dup := seen[text]; dup {
				t.Fatalf("codes %d and %d share %q", prior, code, text)
			}
			seen[text] = code
		}
		if got := commandFailure(&rpc.RPCError{Code: -32000, Message: "projection unavailable"}); strings.Contains(got, "veto") || strings.Contains(got, "locked") {
			t.Fatalf("unknown projection read as a refusal: %q", got)
		}
	})
	t.Run("paused reads remain available", func(t *testing.T) {
		client, broker := nodeInfoRPCFixture(t)
		d := newNodeDetail(client, nodeRow{key: "peer", name: "Peer", membership: membershipMember, presence: presenceOnline})
		d.remotePermission, d.remotePermissionFresh = remotePermission{Version: 1, Allowed: true, State: "allowed"}, true
		snapshot := tuiDownloadSnapshot()
		snapshot.Entries = []modeldownload.Entry{}
		if !d.applyModels(modelsResult{ModelsByEngine: map[string][]string{"ollama": {"model"}}, DownloadState: snapshot, current: true}, true) {
			t.Fatal("ready canonical inventory rejected")
		}
		const pausedText = "This node is paused. Resume it on this node before managing it remotely."
		_, result := terminalControlsReply(t, broker, d.modelCmd("ollama", modelActions["load"], "model"), &rpc.Message{Error: &rpc.RPCError{Code: -32011, Message: pausedText}})
		if cmd, _ := d.update(result); cmd != nil {
			t.Fatal("pause refusal scheduled a retry")
		}
		if !strings.Contains(d.status.render(), pausedText) {
			t.Fatalf("status = %s", d.status.render())
		}
		if d.policyCmd() == nil {
			t.Fatal("a refused mutation stopped status reads")
		}
	})
}

// A locked engine that is also saved Off must still read as locked when the
// narrow engine column clips its label at 80 columns.
func TestLockedEngineRowSurvivesNarrowColumn(t *testing.T) {
	locked := true
	d := localDetail()
	d.SetSize(80, 30)
	d.engines = []engineStatus{{Engine: "ollama", Installed: true, Locked: &locked}, {Engine: "lmstudio", Installed: true}}
	d.intent = engineintent.Snapshot{
		EnabledByEngine: map[string]bool{"ollama": false, "lmstudio": false},
		OffByEngine:     map[string]engineintent.Origin{"ollama": engineintent.Local, "lmstudio": engineintent.Local},
	}
	d.refreshEngines()
	d.engineTable.SetHeight(8)
	rendered := d.engineTable.View()
	var lockedRow, plainRow string
	for _, line := range strings.Split(rendered, "\n") {
		switch {
		case strings.Contains(line, "ollama"):
			lockedRow = line
		case strings.Contains(line, "lmstudio"):
			plainRow = line
		}
	}
	if !strings.Contains(lockedRow, "(locked") {
		t.Fatalf("locked row lost its lock at 80 columns: %q", lockedRow)
	}
	if strings.Contains(plainRow, "locked") || !strings.Contains(plainRow, "(saved Off") {
		t.Fatalf("saved Off row = %q, want provenance without a lock", plainRow)
	}
}
