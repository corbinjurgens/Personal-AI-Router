// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"context"
	"errors"
	"fmt"

	"nvpair-shared/engines"
	"nvpair-tui/rpc"

	tea "github.com/charmbracelet/bubbletea"
)

// engineStatus mirrors nvpair-engine-manager's EngineStatus snapshot, the
// element of engine:get-installed and the engine:state-changed payload.
type engineStatus struct {
	Engine      string `json:"engine"`
	DisplayName string `json:"display_name"`
	Installed   bool   `json:"installed"`
	Running     bool   `json:"running"`
	Healthy     bool   `json:"healthy"`
	Port        int    `json:"port"`
	// Locked is the target's durable engine lock. Nil means the target did not
	// report one, as an older target does, and is never read as unlocked.
	Locked *bool `json:"locked,omitempty"`
}

// engineLockedMessage is shown when an activation is refused because the
// engine is locked on its owning node.
const engineLockedMessage = "This engine is locked on this node. Unlock it on this node before starting or loading a model."

// locked reports an authoritative lock. A target that did not report one is
// unknown, not unlocked, so only a present true counts.
func (e engineStatus) locked() bool { return e.Locked != nil && *e.Locked }

func (e engineStatus) label() string {
	if e.DisplayName != "" {
		return e.DisplayName
	}
	return e.Engine
}

// modelsResult mirrors nvpair-engine-manager's ModelsResult, the engine:models
// reply and the engine:models-changed payload. LoadedByEngine names the models
// currently resident in memory, which the engine-manager polls for and pushes —
// so a client never has to poll to know what is loaded.
type modelsResult struct {
	Models         []string            `json:"models"`
	ModelsByEngine map[string][]string `json:"modelsByEngine"`
	LoadedByEngine map[string][]string `json:"loadedByEngine"`
}

// engineOp is one lifecycle request and how to describe it to the operator.
type engineOp struct {
	method string
	what   string
	// localOnly marks an operation the engine manager exposes no remote
	// equivalent for, so it is hidden on a peer's node rather than offered and
	// then failing.
	localOnly bool
}

// engineOps are the lifecycle operations, keyed by the local method name. The
// remote variants take a node and cover a deliberately smaller set: the manager
// has remote install/start/stop but no remote restart, uninstall, or port
// change, because those need process ownership on the target host.
var engineOps = map[string]engineOp{
	"install":   {method: "engine:install", what: "install"},
	"start":     {method: "engine:start", what: "start"},
	"stop":      {method: "engine:stop", what: "stop"},
	"restart":   {method: "engine:restart", what: "restart", localOnly: true},
	"uninstall": {method: "engine:uninstall", what: "uninstall", localOnly: true},
}

// remoteEngineMethods maps a local lifecycle method to its remote counterpart.
var remoteEngineMethods = map[string]string{
	"engine:install": "engine:remote-install",
	"engine:start":   "engine:remote-start",
	"engine:stop":    "engine:remote-stop",
}

// modelAction is one model operation: the engine-manager method that performs
// it on this machine, the remote method that performs it on a peer, and the
// operator-facing verb. Both methods take only an engine and a model, so the
// engine's own action names and body spelling stay with engine-manager, which
// maps them through the manifest's operation bindings.
type modelAction struct {
	op     string
	local  string
	remote string
	what   string
}

var modelActions = map[string]modelAction{
	"load": {op: engines.OperationLoad, local: "engine:load-model", remote: "engine:remote-load-model", what: "load"},
	// "eject" to match the key's own label and the docs; the backend's method
	// keeps its own name. A key labelled eject that reports "unload requested"
	// leaves the operator wondering whether it did something else.
	"unload": {op: engines.OperationUnload, local: "engine:unload-model", remote: "engine:remote-unload-model", what: "eject"},
	"delete": {op: engines.OperationDelete, local: "engine:delete-model", remote: "engine:remote-delete-model", what: "delete"},
	// A local pull has no typed method yet: it is engine:action pull_model,
	// spelled by pullParams.
	"pull": {op: engines.OperationPull, remote: "engine:remote-pull-model", what: "download"},
}

// engineDescriptors is this machine's engine-manager descriptor set
// (engine:descriptors), keyed by engine ID. It says what each engine is and
// which model operations it supports. A peer's engine is looked up here too;
// one that is absent gets no new action, and its own engine-manager decides
// the operations the interface already offered.
type engineDescriptors map[string]engines.Descriptor

// descriptorsMsg carries a fresh descriptor read.
type descriptorsMsg struct {
	descriptors engineDescriptors
}

// descriptorsCmd reads the local engine-manager's descriptors. A descriptor
// that fails validation is dropped rather than guessed at, and a failed read
// leaves the set empty; either way the engine keeps today's behavior.
func descriptorsCmd(client *rpc.Client) tea.Cmd {
	const method = "engine:descriptors"
	return call(client, method, nil, func(msg *rpc.Message, err error) tea.Msg {
		out := engineDescriptors{}
		if err != nil {
			return descriptorsMsg{descriptors: out}
		}
		var r struct {
			Engines []engines.Descriptor `json:"engines"`
		}
		decodeOrLog(method, msg.Result, &r)
		for _, d := range r.Engines {
			if d.Validate() == nil {
				out[d.Engine] = d
			}
		}
		return descriptorsMsg{descriptors: out}
	})
}

// family is the engine's family: its descriptor's, or the engine's own ID when
// there is no descriptor (a built-in engine's ID is its family).
func (ds engineDescriptors) family(engine string) string {
	if d, ok := ds[engine]; ok {
		return d.Family
	}
	return engine
}

// unsupported is an error when the engine's descriptor does not declare the
// operation, and nil when it does or when this machine holds no descriptor for
// the engine.
func (ds engineDescriptors) unsupported(engine string, act modelAction) error {
	d, ok := ds[engine]
	if !ok {
		return nil
	}
	if _, ok := d.Operations[act.op]; ok {
		return nil
	}
	return fmt.Errorf("%s does not support %s", d.Label, act.what)
}

// downloadExamples show what a download name looks like for each engine family.
// The families spell them very differently: an Ollama tag, a Hugging Face repo,
// and a repo with its quantization after a colon.
var downloadExamples = map[string]string{
	engines.NameOllama:   "llama3.2",
	engines.NameLMStudio: "lmstudio-community/Qwen3-8B-GGUF",
	engines.NameLlamaCPP: "ggml-org/gemma-3-1b-it-GGUF:Q4_K_M",
}

// downloadPrompt is the placeholder for the download-by-name field, with an
// example in the engine family's own spelling where there is one.
func downloadPrompt(family, label string) string {
	if example, ok := downloadExamples[family]; ok {
		return fmt.Sprintf("model name for %s (e.g. %s)", label, example)
	}
	return "model name for " + label
}

// pullParams spells a local pull_model for an engine family.
//
// Ollama's pull takes "name" and LM Studio's takes "model", so both are sent
// there. Every other family gets "model" alone, engine-manager's conventional
// parameter: llama.cpp's pull would pass a second key to the engine as a
// parameter it does not take.
func pullParams(family, model string) map[string]any {
	switch family {
	case engines.NameOllama, engines.NameLMStudio:
		return map[string]any{"name": model, "model": model}
	default:
		return map[string]any{"model": model}
	}
}

// localModelRequest is the method and params of a model operation on this
// machine.
func localModelRequest(engine, family string, act modelAction, model string) (string, map[string]any) {
	if act.local == "" {
		return "engine:action", map[string]any{
			"engine": engine, "action": "pull_model", "params": pullParams(family, model),
		}
	}
	return act.local, map[string]any{"engine": engine, "model": model}
}

// engineOpMsg is the outcome of a lifecycle or model command.
type engineOpMsg struct {
	what   string
	engine string
	err    error
	// detached marks a request that outlived its reply deadline but is still
	// running on the engine. Distinct from both success and failure: nothing
	// went wrong, and nothing has finished either.
	detached bool
}

// engineCmd issues a lifecycle request against a node. An empty node means this
// machine and uses the local method; otherwise the remote counterpart is used
// and the node travels in the params.
func engineCmd(client *rpc.Client, node, engine, method, op, what string) tea.Cmd {
	params := map[string]any{"engine": engine}
	if node != "" {
		remote, ok := remoteEngineMethods[method]
		if !ok {
			return func() tea.Msg {
				return engineOpMsg{what: what, engine: engine,
					err: errors.New("not supported on a remote node")}
			}
		}
		method = remote
		params["node"] = node
	}
	return call(client, method, params, func(_ *rpc.Message, err error) tea.Msg {
		return classifyOpResult(what, engine, op, err)
	})
}

// modelCmd issues a model operation against a node.
//
// A download that outlasts callTimeout is reported as detached rather than
// failed: a multi-gigabyte pull routinely exceeds the reply deadline while the
// engine keeps working, and the engine:pull-progress feed carries the real
// outcome. Reporting a failure there would be wrong — but so was the silence
// this replaced, which left the operator with no acknowledgement that the
// download had started at all, and nothing to distinguish it from a keystroke
// that missed.
func modelCmd(client *rpc.Client, node, engine, family string, act modelAction, model string) tea.Cmd {
	what := act.what + " " + model
	// The methods take the operation in the method name, so the engine's own
	// action vocabulary stays with the engine-manager that runs it.
	method := act.remote
	params := map[string]any{"node": node, "engine": engine, "model": model}
	if node == "" {
		method, params = localModelRequest(engine, family, act, model)
	}
	return call(client, method, params, func(_ *rpc.Message, err error) tea.Msg {
		return classifyOpResult(what, engine, act.op, err)
	})
}

// longRunningOps are the operations whose real duration is set by how much data
// has to move or how slow an engine is to become ready, not by the RPC.
//
// A deadline on one of these means the reply was slow, not that the work
// failed — the engine keeps going and reports the true outcome on its progress
// feed. Reporting a failure is actively misleading: the operator sees "load
// failed" at the same moment the model finishes loading.
//
// Ollama's load is `run_model` with streaming off, which does not answer until
// the model is resident and has produced a response, so a large model on cold
// storage exceeds the reply deadline routinely. Install downloads an engine.
//
// Start and restart belong here too. The engine manager answers only once the
// engine passes its readiness probe, and the manifests allow that probe six
// hundred seconds for Ollama and sixty for LM Studio — the desktop app gives
// the same calls a fourteen-minute envelope for exactly this reason. Held to
// the thirty-five-second reply deadline, a cold start behind a slow probe
// reported failure while the engine went on to come up.
//
// Stop, delete, and unload, by contrast, are quick — stop is bounded by the
// manifest's five-second grace — and a deadline there is a real fault worth
// surfacing.
var longRunningOps = map[string]bool{
	"pull":      true,
	"load":      true,
	"install":   true,
	"uninstall": true,
	"start":     true,
	"restart":   true,
}

// classifyOpResult reports an operation as done, failed, or still running.
func classifyOpResult(what, engine, op string, err error) tea.Msg {
	if err != nil && longRunningOps[op] && errors.Is(err, context.DeadlineExceeded) {
		return engineOpMsg{what: what, engine: engine, detached: true}
	}
	return engineOpMsg{what: what, engine: engine, err: err}
}
