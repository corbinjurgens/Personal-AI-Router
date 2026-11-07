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
	"nvpair-shared/engines"
	"nvpair-shared/enginesettings"
	"nvpair-shared/modeldownload"
	"nvpair-tui/rpc"
)

func tuiDownloadSnapshot() *modeldownload.Snapshot {
	return &modeldownload.Snapshot{
		Version: modeldownload.Version, SourceID: strings.Repeat("a", 32), Revision: "7",
		Entries: []modeldownload.Entry{{ID: strings.Repeat("b", 32), Engine: "llamacpp-custom", Model: "exact/model",
			Phase: modeldownload.PhaseInProgress, BlocksReadiness: true}},
	}
}

func tuiLlamaCPPDownloadSnapshot() *modeldownload.Snapshot {
	return &modeldownload.Snapshot{
		Version: modeldownload.Version, SourceID: strings.Repeat("a", 32), Revision: "7",
		Entries: []modeldownload.Entry{{ID: strings.Repeat("b", 32), Engine: engines.NameLlamaCPP, Model: "exact/model",
			Phase: modeldownload.PhaseInProgress, BlocksReadiness: true}},
	}
}

func tuiCustomEngine(t *testing.T) engines.Engine {
	t.Helper()
	engine, ok := engines.DynamicEngine("llamacpp-custom", "llama.cpp custom")
	if !ok {
		t.Fatal("invalid custom llama.cpp engine")
	}
	return engine
}

func tuiDownloadJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode download fixture: %v", err)
	}
	return data
}

func TestModelsResultDecodesIdentities(t *testing.T) {
	const valid = `{"repository":"unsloth/LFM2.5-24B-A2B-GGUF","quantization":"UD-Q4_K_XL","quantizationSource":"hf-tag","configuredContextTokens":131072,"files":[{"role":"model","name":"model.gguf","sizeBytes":42}],"strength":"declared"}`
	want := map[string]map[string]engines.ModelIdentity{
		"llamacpp": {"served": {
			Repository: "unsloth/LFM2.5-24B-A2B-GGUF", Quantization: "UD-Q4_K_XL", QuantizationSource: "hf-tag",
			ConfiguredContextTokens: 131072, Strength: "declared",
			Files: []engines.ModelIdentityFile{{Role: "model", Name: "model.gguf", SizeBytes: 42}},
		}},
	}
	decoders := []struct {
		name   string
		decode func([]byte) (modelsResult, error)
	}{
		{"local inventory", func(data []byte) (modelsResult, error) {
			var result modelsResult
			err := json.Unmarshal(data, &result)
			return result, err
		}},
		{"peer discovery", func(data []byte) (modelsResult, error) {
			var node availableNode
			err := json.Unmarshal(data, &node)
			return node.modelSnapshot(), err
		}},
	}
	for _, decoder := range decoders {
		t.Run(decoder.name, func(t *testing.T) {
			test := func(name, metadata string, keepValid bool) {
				t.Run(name, func(t *testing.T) {
					snapshot := tuiLlamaCPPDownloadSnapshot()
					data := `{"models":["served"],"modelsByEngine":{"llamacpp":["served"]},"loadedByEngine":{"llamacpp":["served"]},"downloadState":` +
						string(tuiDownloadJSON(t, snapshot)) + `,"downloadStateCurrent":true` + metadata + `}`
					got, err := decoder.decode([]byte(data))
					if err != nil {
						t.Fatalf("decode inventory: %v", err)
					}
					var expected map[string]map[string]engines.ModelIdentity
					if keepValid {
						expected = want
					}
					if !reflect.DeepEqual(got.ModelIdentities, expected) {
						t.Fatalf("identities = %+v, want %+v", got.ModelIdentities, expected)
					}
					if !reflect.DeepEqual(got.Models, []string{"served"}) ||
						!reflect.DeepEqual(got.ModelsByEngine, map[string][]string{"llamacpp": {"served"}}) ||
						!reflect.DeepEqual(got.LoadedByEngine, map[string][]string{"llamacpp": {"served"}}) {
						t.Fatalf("identity metadata changed inventory/residency: %+v", got)
					}
					if !got.current || got.DownloadState == nil || !got.DownloadState.Equal(*snapshot) {
						t.Fatalf("identity metadata changed download support: %+v", got)
					}
				})
			}
			test("valid identity", `,"modelIdentities":{"llamacpp":{"served":`+valid+`}}`, true)
			test("invalid entries isolated", `,"modelIdentities":{"llamacpp":{
				"served":`+valid+`,
				"bad-type":"invalid",
				"bad-field-type":{"repository":42,"strength":"declared"},
				"bad-repository":{"repository":"/tmp/model","strength":"declared"},
				"bad-digest":{"contentId":"`+strings.Repeat("a", 63)+`","strength":"verified"},
				"bad-file":{"files":[{"role":"model","name":"dir/model.gguf","sizeBytes":1}],"strength":"declared"},
				"negative-context":{"configuredContextTokens":-1,"strength":"declared"},
				"missing-strength":{},
				"null-entry":null
			},"bad-engine":[]}`, true)
			test("absent block", "", false)
			test("null block", `,"modelIdentities":null`, false)
			test("string block", `,"modelIdentities":"invalid"`, false)
			test("array block", `,"modelIdentities":[]`, false)
			test("malformed engine block", `,"modelIdentities":{"llamacpp":"invalid"}`, false)
		})
	}
}

func TestModelDownloadLocalDecodePreservesInventoryWithoutTrustingMalformedSupport(t *testing.T) {
	valid := string(tuiDownloadJSON(t, tuiLlamaCPPDownloadSnapshot()))
	test := func(name, extension string, supported bool) {
		t.Run(name, func(t *testing.T) {
			var got modelsResult
			data := `{"modelsByEngine":{"llamacpp":["exact/model"]},"loadedByEngine":{"llamacpp":["exact/model"]}` + extension + `}`
			if err := json.Unmarshal([]byte(data), &got); err != nil {
				t.Fatalf("decode inventory: %v", err)
			}
			if got.current != supported || (got.DownloadState != nil) != supported {
				t.Fatalf("support = %v, snapshot = %+v", got.current, got.DownloadState)
			}
			if !reflect.DeepEqual(got.ModelsByEngine["llamacpp"], []string{"exact/model"}) ||
				!reflect.DeepEqual(got.LoadedByEngine["llamacpp"], []string{"exact/model"}) {
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
			data := `{"hostUuid":"peer","modelsByEngine":{"llamacpp":["exact/model"]},"downloadState":` + string(tuiDownloadJSON(t, tuiLlamaCPPDownloadSnapshot())) + metadata + `}`
			var node availableNode
			if err := json.Unmarshal([]byte(data), &node); err != nil {
				t.Fatalf("decode discovery: %v", err)
			}
			rows := mergeNodes(nodeFeeds{discovered: []availableNode{node}})
			if len(rows) != 1 || rows[0].downloadState == nil || !rows[0].downloadState.Equal(*tuiLlamaCPPDownloadSnapshot()) {
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
	return localModelRequest(engine, act, model)
}

// TestLocalModelOperationsUseTypedMethods checks load, eject and delete reach
// engine-manager's typed methods with only the engine and the model, for every
// engine and for a variant the shared table has never heard of. Engine-specific
// operation bindings live in manifests, so a second spelling here could only
// drift from them.
func TestLocalModelOperationsUseTypedMethods(t *testing.T) {
	want := map[string]string{
		"load": "engine:load-model", "unload": "engine:unload-model", "delete": "engine:delete-model",
	}
	ids := []string{engines.NameLlamaCPP, tuiCustomEngine(t).Name}
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

// TestPullDeadlineReportsDetachedNotSilence guards the acknowledgement for a
// long download. A multi-gigabyte pull outlasts the reply deadline while the
// engine keeps working, so a failure would be wrong — but the previous silence
// was too, leaving the operator unable to tell a started download from a
// keystroke that missed.
func TestPullDeadlineReportsDetachedNotSilence(t *testing.T) {
	msg := classifyOpResult("download big-model", "llamacpp", "pull", context.DeadlineExceeded)
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
// much data moves or how slow an engine is to become ready. Loading a model
// waits until it is resident, so a large model on cold storage can exceed the
// deadline routinely, and reporting "load failed" when the model finishes is worse
// than saying nothing. Quick operations get no such excuse, because a deadline
// there is a real fault.
func TestDeadlineLeniencyTracksOperationLength(t *testing.T) {
	// Only operations engineOps actually declares: the TUI offers no engine
	// update, so listing one here would assert against a path nothing reaches.
	for _, op := range []string{"pull", "load", "install", "uninstall"} {
		result, ok := classifyOpResult("x", "llamacpp", op, context.DeadlineExceeded).(engineOpMsg)
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

	for _, op := range []string{"unload", "delete"} {
		result, ok := classifyOpResult("x", "llamacpp", op, context.DeadlineExceeded).(engineOpMsg)
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
	result, _ := classifyOpResult("x", "llamacpp", "pull", errors.New("no such model")).(engineOpMsg)
	if result.detached || result.err == nil {
		t.Errorf("a genuine pull error was not reported: %+v", result)
	}
}

// TestLocalPullSendsOnlyTheModel checks a local pull carries the model under
// "model" alone, for a llama.cpp fork and for another descriptor engine: llama.cpp
// would take a second key as a parameter it does not accept.
func TestLocalPullSendsOnlyTheModel(t *testing.T) {
	pull := modelActions["pull"]
	_, envelope := localModelRequest("llamacpp-custom", pull, "m:Q4")
	if !reflect.DeepEqual(envelope["params"], map[string]any{"model": "m:Q4"}) {
		t.Errorf("llama.cpp fork pull params = %v, want only the model", envelope["params"])
	}
	_, envelope = localModelRequest("strata", pull, "m")
	if !reflect.DeepEqual(envelope["params"], map[string]any{"model": "m"}) {
		t.Errorf("descriptor engine pull params = %v, want the conventional model key", envelope["params"])
	}
}

// TestDownloadPromptUsesTheEnginesSpelling is the regression guard for an
// generic example offered for every engine. llama.cpp's example needs a
// quantization after a colon. Examples are per family, so a fork shows its
// family's.
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
	custom := tuiCustomEngine(t)
	ds := engineDescriptors{
		"llamacpp-custom": {
			Version: engines.DescriptorVersion, Engine: custom.Name, Family: engines.NameLlamaCPP, Label: custom.DisplayName,
			ModelNaming: engines.ModelNamingExact,
			Operations: map[string]engines.OperationCapability{
				engines.OperationLoad: {}, engines.OperationUnload: {},
			},
		},
	}
	if err := ds.unsupported("llamacpp-custom", modelActions["load"]); err != nil {
		t.Errorf("declared load refused: %v", err)
	}
	for _, op := range []string{"delete", "pull"} {
		err := ds.unsupported("llamacpp-custom", modelActions[op])
		if err == nil || !strings.Contains(err.Error(), "llama.cpp custom does not support") {
			t.Errorf("undeclared %s = %v, want refused naming the engine", op, err)
		}
	}
	if err := ds.unsupported("peer-variant", modelActions["delete"]); err != nil {
		t.Errorf("engine without a descriptor refused: %v", err)
	}
	if got := ds.family("llamacpp-custom"); got != "llamacpp" {
		t.Errorf("family = %q", got)
	}
	if got := ds.family(engines.NameLlamaCPP); got != engines.NameLlamaCPP {
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
// a remote counterpart. The engine manager exposes remote install but no
// remote uninstall, so uninstall must be marked local-only.
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

	for _, name := range []string{"uninstall"} {
		if !engineOps[name].localOnly {
			t.Errorf("%s must be local-only: the manager exposes no remote variant", name)
		}
	}
	for _, name := range []string{"install"} {
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
			request, result := terminalControlsReply(t, broker, modelCmd(client, tc.node, "llamacpp", modelActions[tc.op], "model"), &rpc.Message{Result: []byte(`{}`)})
			var params map[string]string
			if err := json.Unmarshal(request.Params, &params); err != nil {
				t.Fatal(err)
			}
			want := map[string]string{"engine": "llamacpp", "model": "model"}
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
		{"local install", "", "engine:install", "engine:install"},
		{"remote install", "peer", "engine:install", "engine:remote-install"},
		{"local uninstall", "", "engine:uninstall", "engine:uninstall"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			op := strings.TrimPrefix(tc.method, "engine:")
			request, _ := terminalControlsReply(t, broker, engineCmd(client, tc.node, "llamacpp", tc.method, op, op), &rpc.Message{Result: []byte(`{}`)})
			var params map[string]string
			if err := json.Unmarshal(request.Params, &params); err != nil {
				t.Fatal(err)
			}
			want := map[string]string{"engine": "llamacpp"}
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
					cmd = engineCmd(client, "peer", "llamacpp", "engine:install", "install", "install")
				case "model":
					cmd = modelCmd(client, "peer", "llamacpp", modelActions["load"], "model")
				case "settings read":
					cmd = getEngineSettingsCmd(client, "peer", "llamacpp")
				case "settings preview":
					cmd = previewEngineSettingsCmd(client, enginesettings.Request{NodeID: "peer", Engine: "llamacpp"})
				case "settings apply":
					cmd = applyEngineSettingsCmd(client, enginesettings.Request{NodeID: "peer", Engine: "llamacpp"})
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

func TestEngineStatusTerminationReadback(t *testing.T) {
	for _, tc := range []struct {
		name, raw string
		verified  bool
	}{
		{"positive", `{"engine":"custom","running":false,"terminationVerified":true}`, true},
		{"negative", `{"engine":"custom","running":false,"terminationVerified":false}`, false},
		{"legacy omission", `{"engine":"custom","running":false}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var status engineStatus
			if err := json.Unmarshal([]byte(tc.raw), &status); err != nil {
				t.Fatal(err)
			}
			if status.TerminationVerified != tc.verified {
				t.Fatalf("status=%+v", status)
			}
		})
	}
}

func TestEngineStatusBuildReadback(t *testing.T) {
	var status engineStatus
	if err := json.Unmarshal([]byte(`{"engine":"llamacpp","running":true,"buildInfo":"b12345-test","buildId":"sha256:example"}`), &status); err != nil {
		t.Fatal(err)
	}
	if status.BuildInfo != "b12345-test" || status.BuildID != "sha256:example" {
		t.Fatalf("build readback = %+v", status)
	}
	d := localDetail()
	d.engines = []engineStatus{status}
	d.handleNotification(&rpc.Message{Method: "engine:state-changed", Params: json.RawMessage(`{"engine":"llamacpp","running":false}`)})
	if len(d.engines) != 1 || d.engines[0].BuildInfo != "" || d.engines[0].BuildID != "" {
		t.Fatalf("stopped status retained a stale build: %+v", d.engines)
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
	request, result := terminalControlsReply(t, broker, descriptorsCmd(client), &rpc.Message{Result: []byte(`{"engines":[{"version":1,"engine":"llamacpp-custom","family":"llamacpp","label":"llama.cpp custom","modelNaming":"exact","operations":{"load":{}}},{"version":999,"engine":"invalid","family":"llamacpp","label":"Invalid","modelNaming":"exact"}]}`)})
	if request.Method != "engine:descriptors" {
		t.Fatalf("method = %s", request.Method)
	}
	msg, ok := result.(descriptorsMsg)
	if !ok || len(msg.descriptors) != 1 {
		t.Fatalf("descriptors = %#v", result)
	}
	d := newNodeDetail(nil, nodeRow{self: true})
	d.descriptors = msg.descriptors
	if d.refuseModelOp("llamacpp-custom", modelActions["load"]) {
		t.Fatal("declared operation refused")
	}
	if !d.refuseModelOp("llamacpp-custom", modelActions["delete"]) || d.status.kind != toastError {
		t.Fatal("undeclared delete offered")
	}
	if _, present := d.descriptors["invalid"]; present {
		t.Fatal("invalid descriptor accepted")
	}
}

func TestTerminalControlsTimeoutContracts(t *testing.T) {
	load := classifyOpResult("load", "llamacpp", "load", context.DeadlineExceeded).(engineOpMsg)
	if !load.detached || load.err != nil {
		t.Fatalf("load deadline = %+v", load)
	}
	unload := classifyOpResult("eject", "llamacpp", "unload", context.DeadlineExceeded).(engineOpMsg)
	if unload.detached || !errors.Is(unload.err, context.DeadlineExceeded) {
		t.Fatalf("unload deadline = %+v", unload)
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
}

func TestTerminalControlsModelOperationsAndPause(t *testing.T) {
	t.Run("model operations remain available", func(t *testing.T) {
		d := localDetail()
		d.engines = []engineStatus{{Engine: "llamacpp", Installed: true}}
		for _, op := range []string{"unload", "delete", "pull"} {
			if d.refuseModelOp("llamacpp", modelActions[op]) {
				t.Fatalf("%s was refused: %s", op, d.status.render())
			}
		}
	})
	t.Run("distinct paused conflict and denied refusals", func(t *testing.T) {
		seen := map[string]int{}
		for _, code := range []int{-32011, -32010, -32012, -32000} {
			text := commandFailure(&rpc.RPCError{Code: code, Message: "target text"})
			if !strings.Contains(text, "target text") {
				t.Fatalf("code %d dropped the target text: %q", code, text)
			}
			if prior, dup := seen[text]; dup {
				t.Fatalf("codes %d and %d share %q", prior, code, text)
			}
			seen[text] = code
		}
		if got := commandFailure(&rpc.RPCError{Code: -32000, Message: "projection unavailable"}); strings.Contains(got, "veto") {
			t.Fatalf("unknown projection read as a refusal: %q", got)
		}
	})
	t.Run("paused reads remain available", func(t *testing.T) {
		client, broker := nodeInfoRPCFixture(t)
		d := newNodeDetail(client, nodeRow{key: "peer", name: "Peer", membership: membershipMember, presence: presenceOnline})
		d.remotePermission, d.remotePermissionFresh = remotePermission{Version: 1, Allowed: true, State: "allowed"}, true
		snapshot := tuiLlamaCPPDownloadSnapshot()
		snapshot.Entries = []modeldownload.Entry{}
		if !d.applyModels(modelsResult{ModelsByEngine: map[string][]string{"llamacpp": {"model"}}, DownloadState: snapshot, current: true}, true) {
			t.Fatal("ready canonical inventory rejected")
		}
		const pausedText = "This node is paused. Resume it on this node before managing it remotely."
		_, result := terminalControlsReply(t, broker, d.modelCmd("llamacpp", modelActions["load"], "model"), &rpc.Message{Error: &rpc.RPCError{Code: -32011, Message: pausedText}})
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
