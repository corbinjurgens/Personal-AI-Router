// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
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
