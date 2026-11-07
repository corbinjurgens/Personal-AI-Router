// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"nvpair-shared/engines"
	"nvpair-shared/modeldownload"
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
	BuildInfo   string `json:"buildInfo,omitempty"`
	BuildID     string `json:"buildId,omitempty"`
	// Positive stop evidence; omission from an older target is unverified.
	TerminationVerified bool `json:"terminationVerified"`
}

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
	Models          []string                                    `json:"models"`
	ModelsByEngine  map[string][]string                         `json:"modelsByEngine"`
	LoadedByEngine  map[string][]string                         `json:"loadedByEngine"`
	ModelIdentities map[string]map[string]engines.ModelIdentity `json:"modelIdentities,omitempty"`
	DownloadState   *modeldownload.Snapshot                     `json:"downloadState,omitempty"`
	// current qualifies the canonical inventory, not operation completion.
	current bool
}

// UnmarshalJSON preserves inventory/residency even when the extension is
// malformed, while refusing to treat malformed or absent support as ready.
func (r *modelsResult) UnmarshalJSON(data []byte) error {
	type plain modelsResult
	var wire struct {
		*plain
		DownloadState   json.RawMessage `json:"downloadState"`
		ModelIdentities json.RawMessage `json:"modelIdentities"`
	}
	next := modelsResult{}
	wire.plain = (*plain)(&next)
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	next.DownloadState = parseDownloadState(wire.DownloadState)
	next.ModelIdentities = parseModelIdentities(wire.ModelIdentities)
	next.current = next.DownloadState != nil
	*r = next
	return nil
}

// parseModelIdentities isolates optional display metadata failures from the
// inventory and from valid entries belonging to other models or engines.
func parseModelIdentities(data json.RawMessage) map[string]map[string]engines.ModelIdentity {
	var wire map[string]json.RawMessage
	if json.Unmarshal(data, &wire) != nil {
		return nil
	}
	var identities map[string]map[string]engines.ModelIdentity
	for engine, raw := range wire {
		var models map[string]json.RawMessage
		if json.Unmarshal(raw, &models) != nil {
			continue
		}
		for model, raw := range models {
			var identity engines.ModelIdentity
			if json.Unmarshal(raw, &identity) != nil || identity.Validate() != nil {
				continue
			}
			if identities == nil {
				identities = make(map[string]map[string]engines.ModelIdentity)
			}
			if identities[engine] == nil {
				identities[engine] = make(map[string]engines.ModelIdentity)
			}
			identities[engine][model] = identity
		}
	}
	return identities
}

func parseDownloadState(data json.RawMessage) *modeldownload.Snapshot {
	snapshot, err := modeldownload.Parse(data)
	if err != nil {
		return nil
	}
	return &snapshot
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
// remote variant takes a node; uninstall remains local-only.
var engineOps = map[string]engineOp{
	"install":   {method: "engine:install", what: "install"},
	"uninstall": {method: "engine:uninstall", what: "uninstall", localOnly: true},
}

// remoteEngineMethods maps a local lifecycle method to its remote counterpart.
var remoteEngineMethods = map[string]string{
	"engine:install": "engine:remote-install",
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
	// spelled by localModelRequest.
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
// llama.cpp names a repository with its quantization after a colon.
var downloadExamples = map[string]string{
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

// localModelRequest is the method and params of a model operation on this
// machine.
func localModelRequest(engine string, act modelAction, model string) (string, map[string]any) {
	if act.local == "" {
		return "engine:action", map[string]any{
			"engine": engine, "action": "pull_model", "params": map[string]any{"model": model},
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
func modelCmd(client *rpc.Client, node, engine string, act modelAction, model string) tea.Cmd {
	what := act.what + " " + model
	// The methods take the operation in the method name, so the engine's own
	// action vocabulary stays with the engine-manager that runs it.
	method := act.remote
	params := map[string]any{"node": node, "engine": engine, "model": model}
	if node == "" {
		method, params = localModelRequest(engine, act, model)
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
// A load does not answer until the model is resident, so a large model on cold
// storage can exceed the reply deadline. Install downloads an engine.
//
// Delete and unload, by contrast, are quick, and a deadline there is a real
// fault worth surfacing.
var longRunningOps = map[string]bool{
	"pull":      true,
	"load":      true,
	"install":   true,
	"uninstall": true,
}

// classifyOpResult reports an operation as done, failed, or still running.
func classifyOpResult(what, engine, op string, err error) tea.Msg {
	if err != nil && longRunningOps[op] && errors.Is(err, context.DeadlineExceeded) {
		return engineOpMsg{what: what, engine: engine, detached: true}
	}
	return engineOpMsg{what: what, engine: engine, err: err}
}
