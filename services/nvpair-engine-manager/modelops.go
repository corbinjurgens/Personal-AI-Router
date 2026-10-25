// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// modelops.go is the one authority for the semantic model operations: load,
// unload, pull and delete. A manifest binds each operation to one of its
// actions, naming the parameter that carries the model, any fixed parameters,
// and an optional response budget. Local clients (engine:load-model,
// engine:unload-model, engine:delete-model) and remote peers (the ec model
// endpoints) both resolve through here, so no client has to know an engine's
// own action names or body spelling.

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

const (
	loadModelMethod   = "engine:load-model"
	unloadModelMethod = "engine:unload-model"
	deleteModelMethod = "engine:delete-model"
)

// OperationBinding maps one semantic model operation onto a manifest action.
//
//   - Action names the manifest action that performs the operation.
//   - ModelParam is the parameter that carries the model ("model" when empty):
//     the JSON body or query key of an HTTP action, or the {placeholder} of a
//     cmd or remove_path action.
//   - Params are fixed scalar parameters sent alongside the model, such as
//     Ollama's "stream": false or "keep_alive": 0.
//   - ResponseTimeoutS is how long the engine may take to begin answering an
//     HTTP action, in seconds; zero keeps the ordinary 30s budget. It applies to
//     every call of that action, including engine:action, because the engine's
//     latency does not depend on how the call was named.
type OperationBinding struct {
	Action           string                     `json:"action"`
	ModelParam       string                     `json:"model_param,omitempty"`
	Params           map[string]json.RawMessage `json:"params,omitempty"`
	ResponseTimeoutS int                        `json:"response_timeout_s,omitempty"`
}

const defaultModelParam = "model"

// conventionalOperationActions are the action names an operation binds to when
// the manifest does not bind it explicitly, with the model under "model".
var conventionalOperationActions = map[string]string{
	engines.OperationLoad:   "load_model",
	engines.OperationUnload: "unload_model",
	engines.OperationPull:   pullModelAction,
	engines.OperationDelete: "delete_model",
}

var operationParamRe = regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_]*$`)

func (b OperationBinding) modelParam() string {
	if b.ModelParam != "" {
		return b.ModelParam
	}
	return defaultModelParam
}

func (b OperationBinding) responseBudget() time.Duration {
	return time.Duration(b.ResponseTimeoutS) * time.Second
}

// params builds the action params for one call: the fixed params plus the
// model under the bound name.
func (b OperationBinding) params(model string) (json.RawMessage, error) {
	out := make(map[string]json.RawMessage, len(b.Params)+1)
	for name, value := range b.Params {
		out[name] = value
	}
	encoded, err := json.Marshal(model)
	if err != nil {
		return nil, err
	}
	out[b.modelParam()] = encoded
	return json.Marshal(out)
}

// operationBinding returns the binding the manifest declares for op, or the
// conventional binding when the manifest declares none and has the
// conventional action. ok is false for an operation the engine does not offer.
func (m *Manifest) operationBinding(op string) (OperationBinding, bool) {
	if b, ok := m.Operations[op]; ok {
		return b, true
	}
	action, ok := conventionalOperationActions[op]
	if !ok {
		return OperationBinding{}, false
	}
	if _, ok := m.Actions[action]; !ok {
		return OperationBinding{}, false
	}
	return OperationBinding{Action: action}, true
}

// actionResponseBudget is the response budget a declared operation gives the
// action, or zero for the ordinary budget.
func (m *Manifest) actionResponseBudget(action string) time.Duration {
	for _, b := range m.Operations {
		if b.Action == action && b.ResponseTimeoutS > 0 {
			return b.responseBudget()
		}
	}
	return 0
}

// validateOperations checks the declared bindings. Conventional bindings are
// not validated, so a manifest that relies on them loads exactly as it did
// before bindings existed.
func (m *Manifest) validateOperations() error {
	budgets := make(map[string]int, len(m.Operations))
	for _, op := range sortedKeys(m.Operations) {
		b := m.Operations[op]
		if !slices.Contains(engines.Operations(), op) {
			return fmt.Errorf("operations: unknown operation %q (want one of %s)", op, strings.Join(engines.Operations(), ", "))
		}
		act, ok := m.Actions[b.Action]
		if !ok {
			return fmt.Errorf("operation %q: action %q is not declared", op, b.Action)
		}
		// The streaming pull path and its progress adapters belong to the
		// pull_model action, so a pull binding cannot point elsewhere.
		if op == engines.OperationPull && b.Action != pullModelAction {
			return fmt.Errorf("operation %q: action must be %q", op, pullModelAction)
		}
		modelParam := b.modelParam()
		if !operationParamRe.MatchString(modelParam) || allowedPlaceholders[modelParam] {
			return fmt.Errorf("operation %q: model_param %q must be an identifier that is not a runner placeholder", op, modelParam)
		}
		// The lms resolvers and the llama.cpp download protocol read the model
		// from the "model" parameter itself.
		if (act.ModelResolution != "" || act.ProgressProtocol != "") && modelParam != defaultModelParam {
			return fmt.Errorf("operation %q: action %q reads its model from %q, so model_param must be %q", op, b.Action, defaultModelParam, defaultModelParam)
		}
		switch {
		case len(act.Cmd) > 0:
			if !templatesParam(act.Cmd, modelParam) {
				return fmt.Errorf("operation %q: cmd action %q must template {%s}", op, b.Action, modelParam)
			}
		case act.RemovePath != nil && act.ModelResolution == "":
			if !templatesParam([]string{act.RemovePath.Path}, modelParam) {
				return fmt.Errorf("operation %q: remove_path action %q must template {%s} in its path", op, b.Action, modelParam)
			}
		}
		for _, name := range sortedKeys(b.Params) {
			if !operationParamRe.MatchString(name) || allowedPlaceholders[name] || name == modelParam {
				return fmt.Errorf("operation %q: param %q must be an identifier that is neither a runner placeholder nor the model param", op, name)
			}
			var value any
			if err := json.Unmarshal(b.Params[name], &value); err != nil {
				return fmt.Errorf("operation %q: param %q: %w", op, name, err)
			}
			switch value.(type) {
			case string:
			case float64, bool:
				if act.HTTP != nil && act.HTTP.ParamsIn == actionHTTPParamsQuery {
					return fmt.Errorf("operation %q: param %q must be a string for a query-parameter action", op, name)
				}
			default:
				return fmt.Errorf("operation %q: param %q must be a string, number or boolean", op, name)
			}
		}
		if b.ResponseTimeoutS != 0 {
			if op == engines.OperationPull {
				return fmt.Errorf("operation %q: response_timeout_s is not supported; a pull is bounded by its progress watchdog", op)
			}
			if act.HTTP == nil {
				return fmt.Errorf("operation %q: response_timeout_s requires an http action", op)
			}
			if b.ResponseTimeoutS < 1 || b.ResponseTimeoutS > engines.MaxOperationResponseTimeoutS {
				return fmt.Errorf("operation %q: response_timeout_s %d must be between 1 and %d", op, b.ResponseTimeoutS, engines.MaxOperationResponseTimeoutS)
			}
		}
		// The budget belongs to the action, so two operations sharing one must
		// agree on it.
		if prev, ok := budgets[b.Action]; ok && prev != b.ResponseTimeoutS {
			return fmt.Errorf("operation %q: action %q is bound by another operation with a different response_timeout_s", op, b.Action)
		}
		budgets[b.Action] = b.ResponseTimeoutS
	}
	return nil
}

func templatesParam(templates []string, name string) bool {
	token := "{" + name + "}"
	for _, t := range templates {
		if strings.Contains(t, token) {
			return true
		}
	}
	return false
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

// modelActionRequest is the shared body for the typed model operation RPCs and
// the ec model load/unload/delete endpoints.
type modelActionRequest struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
}

func (p *modelActionRequest) UnmarshalJSON(data []byte) error {
	// Supplied tokens must be checked before typed decoding; absent models
	// retain the callers' structural engine/model checks before any effects.
	if err := validateModelRequest(data, false); err != nil {
		return err
	}
	type request modelActionRequest
	var next request
	if err := json.Unmarshal(data, &next); err != nil {
		return err
	}
	*p = modelActionRequest(next)
	return nil
}

// ModelLoad warms a model into engine memory through the engine's load binding.
func (e *Executor) ModelLoad(ctx context.Context, engine, model string) (json.RawMessage, error) {
	return e.modelOperation(ctx, engine, engines.OperationLoad, model)
}

// ModelUnload frees a model from engine memory.
func (e *Executor) ModelUnload(ctx context.Context, engine, model string) (json.RawMessage, error) {
	return e.modelOperation(ctx, engine, engines.OperationUnload, model)
}

// ModelDelete removes a downloaded model from disk when the engine offers it.
func (e *Executor) ModelDelete(ctx context.Context, engine, model string) (json.RawMessage, error) {
	return e.modelOperation(ctx, engine, engines.OperationDelete, model)
}

func (e *Executor) modelOperation(ctx context.Context, engine, op, model string) (json.RawMessage, error) {
	action, params, err := e.modelOperationCall(engine, op, model)
	if err != nil {
		return nil, err
	}
	return e.Action(ctx, engine, action, params)
}

// modelOperationCall resolves an operation to the action and params to run.
func (e *Executor) modelOperationCall(engine, op, model string) (string, json.RawMessage, error) {
	if err := nodepolicy.ValidateModelIdentifier(model); err != nil {
		return "", nil, err
	}
	m, ok := e.reg.Get(engine)
	if !ok {
		return "", nil, fmt.Errorf("unknown engine %q", engine)
	}
	b, ok := m.operationBinding(op)
	if !ok {
		return "", nil, fmt.Errorf("engine %q does not support the model %s operation", engine, op)
	}
	params, err := b.params(model)
	if err != nil {
		return "", nil, err
	}
	return b.Action, params, nil
}

// A cold identifier refusal must not enter the pull-failure reporting pipeline.
type modelIdentifierRefusal struct{ error }

// validateActionModelParams checks only operation-owned identifiers, never
// arbitrary action content. Conventional names identify model actions, but do
// not declare a required HTTP body/query schema. Required keys come from
// explicit bindings or actual template/resolver/progress consumers.
// Validate JSON before decoding can repair UTF-8.
// Bound keys use the exact spelling dispatch consumes. Common model aliases are
// case-insensitive, but cannot satisfy a differently spelled required key.
func (m *Manifest) validateActionModelParams(action string, params json.RawMessage, requireBound bool) error {
	keys := map[string]bool{}
	modelAction := action == pullModelAction
	for _, op := range engines.Operations() {
		if binding, ok := m.operationBinding(op); ok && binding.Action == action {
			modelAction = true
			if _, declared := m.Operations[op]; declared {
				keys[binding.modelParam()] = true
			}
		}
	}
	act := m.Actions[action]
	if act.ModelResolution != "" || act.ProgressProtocol != "" || templatesParam(act.Cmd, defaultModelParam) ||
		(act.RemovePath != nil && templatesParam([]string{act.RemovePath.Root, act.RemovePath.Path}, defaultModelParam)) {
		keys[defaultModelParam] = true
		modelAction = true
	}
	if modelAction && (templatesParam(act.Cmd, "name") ||
		(act.RemovePath != nil && templatesParam([]string{act.RemovePath.Root, act.RemovePath.Path}, "name"))) {
		keys["name"] = true
	}
	if !modelAction {
		return nil
	}
	var fields map[string]json.RawMessage
	if len(params) != 0 && json.Unmarshal(params, &fields) != nil {
		return &modelIdentifierRefusal{fmt.Errorf("model parameters must be a JSON object")}
	}
	for field, raw := range fields {
		_, bound := keys[field]
		common := strings.EqualFold(field, "name") || strings.EqualFold(field, "model")
		if !bound && !common {
			continue
		}
		if err := nodepolicy.ValidateModelIdentifierJSON(raw); err != nil {
			return &modelIdentifierRefusal{err}
		}
		if bound {
			keys[field] = false
		}
	}
	if requireBound {
		for _, missing := range keys {
			if missing {
				return &modelIdentifierRefusal{nodepolicy.ValidateModelIdentifierJSON(nil)}
			}
		}
	}
	return nil
}

// pullParams is the pull_model params for a pull that names only a model.
func (e *Executor) pullParams(engine, model string) (json.RawMessage, error) {
	_, params, err := e.modelOperationCall(engine, engines.OperationPull, model)
	return params, err
}

// loadResponseBudget is the response budget engine's load binding declares on
// this node, zero for the ordinary budget or an engine this node does not
// know. The remote path sizes its wait for a peer's load by it, on the premise
// that one engine ID names one engine on every node.
func (e *Executor) loadResponseBudget(engine string) time.Duration {
	m, ok := e.reg.Get(engine)
	if !ok {
		return 0
	}
	b, ok := m.operationBinding(engines.OperationLoad)
	if !ok {
		return 0
	}
	return b.responseBudget()
}

// runModelOperation answers engine:load-model, engine:unload-model and
// engine:delete-model: the typed local counterparts of the ec model endpoints.
func (m *Manager) runModelOperation(ctx context.Context, msg *Message) {
	var p modelActionRequest
	if !m.parse(msg, &p) {
		return
	}
	if p.Engine == "" || p.Model == "" {
		m.codec.RespondError(msg.ID, -32602, "engine and model are required")
		return
	}
	var (
		res json.RawMessage
		err error
	)
	switch msg.Method {
	case loadModelMethod:
		res, err = m.exec.ModelLoad(ctx, p.Engine, p.Model)
	case unloadModelMethod:
		res, err = m.exec.ModelUnload(ctx, p.Engine, p.Model)
	case deleteModelMethod:
		res, err = m.exec.ModelDelete(ctx, p.Engine, p.Model)
	}
	if err != nil {
		m.codec.RespondError(msg.ID, rpcCode(err), err.Error())
		return
	}
	m.exec.pokeLoaded()
	m.codec.Respond(msg.ID, res)
}

// Typed model requests keep their existing shape; inspect the original field
// as well so malformed UTF-8 cannot become a selectable replacement character.
func validateModelRequest(params json.RawMessage, required bool) error {
	var fields map[string]json.RawMessage
	if json.Unmarshal(params, &fields) != nil {
		return fmt.Errorf("model parameters must be a JSON object")
	}
	present := false
	for field, raw := range fields {
		if strings.EqualFold(field, "model") {
			present = true
			if err := nodepolicy.ValidateModelIdentifierJSON(raw); err != nil {
				return err
			}
		}
	}
	if required && !present {
		return nodepolicy.ValidateModelIdentifierJSON(nil)
	}
	return nil
}
