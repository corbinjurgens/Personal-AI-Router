// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// bindingManifest has one action of each kind an operation can bind.
func bindingManifest() *Manifest {
	m := testEngineManifest(fakeEngineBin)
	m.Actions = map[string]Action{
		"http_load":   {HTTP: &ActionHTTP{Method: "POST", Path: "/load"}},
		"query_erase": {HTTP: &ActionHTTP{Method: "DELETE", Path: "/models", ParamsIn: "query"}},
		"cli_load":    {Cmd: []string{"{cli}", "load", "{model}"}},
		"rm":          {RemovePath: &ActionRemovePath{Root: "{install_dir}", Path: "{install_dir}/{target}"}},
		"lms_get":     {Cmd: []string{"{cli}", "get", "{model}"}, ModelResolution: modelResolutionLMSGet},
		pullModelAction: {
			HTTP:             &ActionHTTP{Method: "POST", Path: "/models"},
			ProgressProtocol: pullProgressProtocolLlamaCPPModelsSSE,
		},
	}
	return m
}

func TestOperationBindingValidation(t *testing.T) {
	raw := func(s string) json.RawMessage { return json.RawMessage(s) }
	cases := []struct {
		name string
		op   string
		b    OperationBinding
		want string // empty: valid
	}{
		{name: "http with defaults and budget", op: "load", b: OperationBinding{Action: "http_load", ModelParam: "target", Params: map[string]json.RawMessage{"wait": raw(`true`), "n": raw(`2`)}, ResponseTimeoutS: 600}},
		{name: "query action with string default", op: "delete", b: OperationBinding{Action: "query_erase", Params: map[string]json.RawMessage{"scope": raw(`"all"`)}}},
		{name: "remove_path templating the param", op: "delete", b: OperationBinding{Action: "rm", ModelParam: "target"}},
		{name: "pull on pull_model", op: "pull", b: OperationBinding{Action: pullModelAction}},
		{name: "unknown operation", op: "warm", b: OperationBinding{Action: "http_load"}, want: "unknown operation"},
		{name: "undeclared action", op: "load", b: OperationBinding{Action: "missing"}, want: `action "missing" is not declared`},
		{name: "pull on another action", op: "pull", b: OperationBinding{Action: "http_load"}, want: `action must be "pull_model"`},
		{name: "model param not an identifier", op: "load", b: OperationBinding{Action: "http_load", ModelParam: "a-b"}, want: "model_param"},
		{name: "model param is a runner placeholder", op: "load", b: OperationBinding{Action: "http_load", ModelParam: "port"}, want: "model_param"},
		{name: "cmd not templating the param", op: "load", b: OperationBinding{Action: "cli_load", ModelParam: "name"}, want: "must template {name}"},
		{name: "remove_path not templating the param", op: "delete", b: OperationBinding{Action: "rm"}, want: "must template {model} in its path"},
		{name: "resolver reads model", op: "pull", b: OperationBinding{Action: pullModelAction, ModelParam: "name"}, want: `model_param must be "model"`},
		{name: "lms resolver reads model", op: "load", b: OperationBinding{Action: "lms_get", ModelParam: "name"}, want: `model_param must be "model"`},
		{name: "default shadows the model", op: "load", b: OperationBinding{Action: "http_load", Params: map[string]json.RawMessage{"model": raw(`"x"`)}}, want: "param \"model\""},
		{name: "default is a runner placeholder", op: "load", b: OperationBinding{Action: "http_load", Params: map[string]json.RawMessage{"bin": raw(`"x"`)}}, want: "param \"bin\""},
		{name: "default is an object", op: "load", b: OperationBinding{Action: "http_load", Params: map[string]json.RawMessage{"o": raw(`{}`)}}, want: "string, number or boolean"},
		{name: "default is null", op: "load", b: OperationBinding{Action: "http_load", Params: map[string]json.RawMessage{"o": raw(`null`)}}, want: "string, number or boolean"},
		{name: "non-string default for a query action", op: "delete", b: OperationBinding{Action: "query_erase", Params: map[string]json.RawMessage{"n": raw(`1`)}}, want: "must be a string"},
		{name: "budget on pull", op: "pull", b: OperationBinding{Action: pullModelAction, ResponseTimeoutS: 60}, want: "progress watchdog"},
		{name: "budget on a cmd action", op: "load", b: OperationBinding{Action: "cli_load", ResponseTimeoutS: 60}, want: "requires an http action"},
		{name: "budget too large", op: "load", b: OperationBinding{Action: "http_load", ResponseTimeoutS: 1801}, want: "between 1 and 1800"},
		{name: "negative budget", op: "load", b: OperationBinding{Action: "http_load", ResponseTimeoutS: -1}, want: "between 1 and 1800"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := bindingManifest()
			m.Operations = map[string]OperationBinding{tc.op: tc.b}
			err := m.Validate()
			if tc.want == "" {
				if err != nil {
					t.Fatalf("Validate: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate error = %v, want it to contain %q", err, tc.want)
			}
		})
	}
}

// A budget belongs to its action, so two operations that bind one action must
// declare the same budget.
func TestOperationsSharingAnActionAgreeOnItsBudget(t *testing.T) {
	m := bindingManifest()
	m.Operations = map[string]OperationBinding{
		"load":   {Action: "http_load", ResponseTimeoutS: 600},
		"unload": {Action: "http_load"},
	}
	if err := m.Validate(); err == nil || !strings.Contains(err.Error(), "different response_timeout_s") {
		t.Fatalf("Validate error = %v, want a conflicting budget error", err)
	}
	m.Operations["unload"] = OperationBinding{Action: "http_load", ResponseTimeoutS: 600}
	if err := m.Validate(); err != nil {
		t.Fatalf("matching budgets rejected: %v", err)
	}
}

// A manifest that declares no bindings keeps the conventional ones for the
// actions it has, so user manifests written before bindings load unchanged.
func TestConventionalOperationBindings(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	m.Actions["load_model"] = Action{HTTP: &ActionHTTP{Method: "POST", Path: "/load"}}
	m.Actions["delete_model"] = Action{Cmd: []string{"rm-model", "{model}"}}
	if err := m.Validate(); err != nil {
		t.Fatal(err)
	}
	for op, want := range map[string]string{"load": "load_model", "delete": "delete_model"} {
		b, ok := m.operationBinding(op)
		if !ok || b.Action != want || b.modelParam() != "model" || b.ResponseTimeoutS != 0 {
			t.Errorf("%s binding = %+v, %v; want conventional %s with model", op, b, ok, want)
		}
	}
	for _, op := range []string{"unload", "pull"} {
		if b, ok := m.operationBinding(op); ok {
			t.Errorf("%s bound to %+v without an action to bind", op, b)
		}
	}
	// A declared binding replaces the conventional one for its operation only.
	m.Actions["warm"] = Action{HTTP: &ActionHTTP{Method: "POST", Path: "/warm"}}
	m.Operations = map[string]OperationBinding{"load": {Action: "warm"}}
	if b, _ := m.operationBinding("load"); b.Action != "warm" {
		t.Errorf("declared load binding = %+v, want warm", b)
	}
	if b, _ := m.operationBinding("delete"); b.Action != "delete_model" {
		t.Errorf("delete binding = %+v, want the conventional one", b)
	}
}

// Bindings load from JSON with the documented spelling.
func TestOperationBindingsDecodeFromJSON(t *testing.T) {
	m := bindingManifest()
	data := []byte(`{"load": {"action": "http_load", "model_param": "target", "params": {"wait": true}, "response_timeout_s": 90}}`)
	if err := json.Unmarshal(data, &m.Operations); err != nil {
		t.Fatal(err)
	}
	if err := m.Validate(); err != nil {
		t.Fatal(err)
	}
	b := m.Operations["load"]
	params, err := b.params("x")
	if err != nil {
		t.Fatal(err)
	}
	if b.Action != "http_load" || b.ResponseTimeoutS != 90 || string(params) != `{"target":"x","wait":true}` {
		t.Fatalf("decoded binding %+v produced params %s", b, params)
	}
}
