// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

type strictFixtureCase struct {
	Name   string                `json:"name"`
	Kind   string                `json:"kind"`
	Path   []string              `json:"path"`
	Value  any                   `json:"value"`
	Remove bool                  `json:"remove"`
	Valid  bool                  `json:"valid"`
	State  StrictCapabilityState `json:"state"`
}

func TestRoutingControlConstantsAndStatuses(t *testing.T) {
	if MethodRoutingCancel != "routing/cancel" || MethodRoutingActive != "routing/active" {
		t.Fatal("routing method wire names changed")
	}
	for _, status := range []string{"not_found", "conflict_committed", "regenerating", "cancelled"} {
		var out RoutingCancelResult
		if err := json.Unmarshal([]byte(`{"status":"`+status+`"}`), &out); err != nil {
			t.Fatal(err)
		}
		if string(out.Status) != status {
			t.Fatalf("status=%s want=%s", out.Status, status)
		}
	}
}

func TestRoutingControlStrictDTOs(t *testing.T) {
	key := `{"originNodeId":"origin","proxyRunId":"run","requestId":"9223372036854775807"}`
	valid := `{"request":` + key + `,"regenerate":false}`
	var in RoutingCancelParams
	if err := json.Unmarshal([]byte(valid), &in); err != nil {
		t.Fatal(err)
	}
	if in.Request.RequestID != "9223372036854775807" || in.Regenerate {
		t.Fatalf("request=%+v", in)
	}
	test := func(name, raw string, target any) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			if err := json.Unmarshal([]byte(raw), target); err == nil {
				t.Fatalf("accepted %s", raw)
			}
		})
	}
	test("cancel null", `null`, &RoutingCancelParams{})
	test("cancel missing key", `{"regenerate":false}`, &RoutingCancelParams{})
	test("cancel missing regenerate", `{"request":`+key+`}`, &RoutingCancelParams{})
	test("cancel null regenerate", `{"request":`+key+`,"regenerate":null}`, &RoutingCancelParams{})
	test("cancel unknown field", `{"request":`+key+`,"regenerate":false,"extra":1}`, &RoutingCancelParams{})
	test("cancel duplicate request", `{"request":`+key+`,"request":`+key+`,"regenerate":false}`, &RoutingCancelParams{})
	test("cancel duplicate nested id", `{"request":{"originNodeId":"origin","proxyRunId":"run","requestId":"1","requestId":"2"},"regenerate":true}`, &RoutingCancelParams{})
	test("cancel nested unknown", `{"request":{"originNodeId":"origin","proxyRunId":"run","requestId":"1","extra":1},"regenerate":true}`, &RoutingCancelParams{})
	for _, id := range []string{`"0"`, `"01"`, `"-1"`, `"1.0"`, `"9223372036854775808"`, `1`, `null`} {
		test("noncanonical id "+id, `{"request":{"originNodeId":"origin","proxyRunId":"run","requestId":`+id+`},"regenerate":true}`, &RoutingCancelParams{})
	}
	test("active null", `null`, &RoutingActiveParams{})
	test("active unknown", `{"request":`+key+`}`, &RoutingActiveParams{})
	test("result unknown status", `{"status":"ok"}`, &RoutingCancelResult{})
	test("result missing status", `{}`, &RoutingCancelResult{})
	test("result null status", `{"status":null}`, &RoutingCancelResult{})
	test("result duplicate status", `{"status":"cancelled","status":"not_found"}`, &RoutingCancelResult{})
	test("snapshot null requests", `{"requests":null}`, &RoutingActiveResult{})
	test("snapshot missing requests", `{}`, &RoutingActiveResult{})
	test("snapshot duplicate requests", `{"requests":[],"requests":[]}`, &RoutingActiveResult{})
	test("snapshot bad key", `{"requests":[{"originNodeId":"origin","proxyRunId":"run","requestId":"01"}]}`, &RoutingActiveResult{})
	test("snapshot duplicate nested id", `{"requests":[{"originNodeId":"origin","proxyRunId":"run","requestId":"1","requestId":"2"}]}`, &RoutingActiveResult{})
	var params RoutingActiveParams
	if err := json.Unmarshal([]byte(`{}`), &params); err != nil {
		t.Fatal(err)
	}
	var active RoutingActiveResult
	if err := json.Unmarshal([]byte(`{"requests":[`+key+`]}`), &active); err != nil {
		t.Fatal(err)
	}
	if len(active.Requests) != 1 || active.Requests[0] != in.Request {
		t.Fatalf("snapshot=%+v", active)
	}
}

type strictFixtures struct {
	Identity map[string]any      `json:"identity"`
	Request  map[string]any      `json:"request"`
	Model    map[string]any      `json:"model"`
	Cases    []strictFixtureCase `json:"cases"`
}

func loadStrictFixtures(t *testing.T) strictFixtures {
	t.Helper()
	data, err := os.ReadFile("testdata/strict-admission.json")
	if err != nil {
		t.Fatal(err)
	}
	var f strictFixtures
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// The desktop suite consumes this exact corpus, including nested mutations.
func TestStrictAdmissionParityCorpus(t *testing.T) {
	for _, tc := range loadStrictFixtures(t).Cases {
		t.Run(tc.Name, func(t *testing.T) {
			f := loadStrictFixtures(t)
			f.Request["identity"] = f.Identity
			f.Model["identity"] = f.Identity
			capability := map[string]any{"version": 1, "models": []any{f.Model}}
			target := map[string]map[string]any{"request": f.Request, "identity": f.Identity, "capability": capability, "model": f.Model}[tc.Kind]
			if target == nil {
				t.Fatalf("unknown fixture kind %q", tc.Kind)
			}
			if len(tc.Path) > 0 {
				for _, key := range tc.Path[:len(tc.Path)-1] {
					child, ok := target[key].(map[string]any)
					if !ok {
						t.Fatalf("invalid fixture path %v", tc.Path)
					}
					target = child
				}
				key := tc.Path[len(tc.Path)-1]
				if tc.Remove {
					delete(target, key)
				} else {
					target[key] = tc.Value
				}
			}
			var input any = f.Request
			if tc.Kind == "capability" || tc.Kind == "model" {
				input = capability
			}
			data, err := json.Marshal(input)
			if err != nil {
				t.Fatal(err)
			}
			if tc.Kind == "capability" || tc.Kind == "model" {
				var got StrictAdmissionCapability
				err = json.Unmarshal(data, &got)
				if (err == nil) != tc.Valid {
					t.Fatalf("decode error = %v, want valid %v", err, tc.Valid)
				}
				if err != nil {
					return
				}
				want := tc.State
				if want == "" {
					want = StrictCapabilityValid
				}
				if got.State() != want {
					t.Fatalf("state = %q, want %q", got.State(), want)
				}
				encoded, err := json.Marshal(got)
				if err != nil {
					t.Fatal(err)
				}
				var again StrictAdmissionCapability
				if err := json.Unmarshal(encoded, &again); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(got, again) {
					t.Fatalf("unstable round trip: %+v / %+v", got, again)
				}
			} else {
				var got StrictAdmissionRequest
				err = json.Unmarshal(data, &got)
				if (err == nil) != tc.Valid {
					t.Fatalf("decode error = %v, want valid %v", err, tc.Valid)
				}
				if err != nil {
					return
				}
				encoded, err := json.Marshal(got)
				if err != nil {
					t.Fatal(err)
				}
				var again StrictAdmissionRequest
				if err := json.Unmarshal(encoded, &again); err != nil {
					t.Fatal(err)
				}
				if got != again {
					t.Fatalf("unstable round trip: %+v / %+v", got, again)
				}
			}
		})
	}
}

func TestStrictAdmissionNumericSpellings(t *testing.T) {
	f := loadStrictFixtures(t)
	f.Request["identity"] = f.Identity
	data, err := json.Marshal(f.Request)
	if err != nil {
		t.Fatal(err)
	}
	for _, spelling := range []string{"1.0", "1e0"} {
		t.Run(spelling, func(t *testing.T) {
			var got StrictAdmissionRequest
			if err := json.Unmarshal([]byte(strings.Replace(string(data), `"version":1`, `"version":`+spelling, 1)), &got); err != nil {
				t.Fatal(err)
			}
			if got.Version != 1 {
				t.Fatalf("version = %d", got.Version)
			}
		})
	}
}

func TestStrictAdmissionUnsupportedRequestError(t *testing.T) {
	err := (StrictAdmissionRequest{Version: 2}).Validate()
	if !errors.Is(err, ErrStrictAdmissionVersion) {
		t.Fatalf("error = %v", err)
	}
}

// decodeStrictContract must reject a duplicate key at its object level before
// the map collapse silently keeps the last occurrence. The duplicate value is
// listed first so a collapsing decode would accept the object.
func TestStrictContractDuplicateObjectKeys(t *testing.T) {
	identity := `{"engineRunId":"engine-1","settingsFingerprint":"` + strings.Repeat("a", 64) + `","modelDigest":"` + strings.Repeat("b", 64) + `"}`
	duplicateIdentity := `{"engineRunId":"rogue-run","engineRunId":"engine-1","settingsFingerprint":"` + strings.Repeat("a", 64) + `","modelDigest":"` + strings.Repeat("b", 64) + `"}`
	prepareResident := func(inner string) string {
		return `{"policyRevision":"1","proxyRunId":"proxy-1","engine":"ollama","model":"m:latest","identity":` + inner + `,"operationId":"resident:1","timeoutMs":60000}`
	}
	reject := func(t *testing.T, err error) {
		t.Helper()
		if err == nil {
			t.Fatal("accepted a contract with a duplicated key")
		}
		if !strings.Contains(err.Error(), "duplicate field") {
			t.Fatalf("error = %v, want a duplicate field error", err)
		}
	}
	t.Run("prepare resident rejects nested duplicate identity field", func(t *testing.T) {
		var got PrepareResidentParams
		err := json.Unmarshal([]byte(prepareResident(duplicateIdentity)), &got)
		reject(t, err)
	})
	t.Run("identity rejects duplicate field", func(t *testing.T) {
		var got StrictModelIdentity
		err := json.Unmarshal([]byte(duplicateIdentity), &got)
		reject(t, err)
	})
	t.Run("control: duplicate-free objects decode", func(t *testing.T) {
		var params PrepareResidentParams
		if err := json.Unmarshal([]byte(prepareResident(identity)), &params); err != nil {
			t.Fatal(err)
		}
		if params.Identity.EngineRunID != "engine-1" {
			t.Fatalf("identity.engineRunId = %q, want engine-1", params.Identity.EngineRunID)
		}
		var direct StrictModelIdentity
		if err := json.Unmarshal([]byte(identity), &direct); err != nil {
			t.Fatal(err)
		}
		if direct.EngineRunID != "engine-1" {
			t.Fatalf("engineRunId = %q, want engine-1", direct.EngineRunID)
		}
	})
}

func TestStrictCapabilityCardinalityAndDuplicates(t *testing.T) {
	f := loadStrictFixtures(t)
	f.Model["identity"] = f.Identity
	data, err := json.Marshal(f.Model)
	if err != nil {
		t.Fatal(err)
	}
	var model StrictAdmissionModel
	if err := json.Unmarshal(data, &model); err != nil {
		t.Fatal(err)
	}
	t.Run("duplicate model", func(t *testing.T) {
		c := StrictAdmissionCapability{Version: 1, Models: []StrictAdmissionModel{model, model}}
		if err := c.Validate(); err == nil {
			t.Fatal("accepted duplicate engine/model")
		}
	})
	t.Run("too many models", func(t *testing.T) {
		c := StrictAdmissionCapability{Version: 1, Models: make([]StrictAdmissionModel, 1025)}
		if err := c.Validate(); err == nil {
			t.Fatal("accepted excess models")
		}
	})
}

type routingHistoryFixture struct {
	Name   string          `json:"name"`
	Record json.RawMessage `json:"record"`
}

func loadRoutingHistoryFixtures(t *testing.T) []routingHistoryFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/routing-request.json")
	if err != nil {
		t.Fatal(err)
	}
	var corpus struct {
		Schema int                     `json:"schema"`
		Cases  []routingHistoryFixture `json:"cases"`
	}
	if err := json.Unmarshal(data, &corpus); err != nil {
		t.Fatal(err)
	}
	wantNames := []string{"rejected then served local", "terminal interrupted API retained accounting", "cancelled with dropped prefix", "recoverable interruption", "actual linked successor"}
	if corpus.Schema != 1 || len(corpus.Cases) != len(wantNames) {
		t.Fatal("unexpected corpus schema/cases")
	}
	for i, name := range wantNames {
		if corpus.Cases[i].Name != name {
			t.Fatalf("case %d name = %q, want %q", i, corpus.Cases[i].Name, name)
		}
	}
	return corpus.Cases
}

func routingHistoryObject(t *testing.T, raw []byte) map[string]any {
	t.Helper()
	d := json.NewDecoder(strings.NewReader(string(raw)))
	d.UseNumber()
	var value map[string]any
	if err := d.Decode(&value); err != nil {
		t.Fatal(err)
	}
	return value
}

func routingHistoryParent(t *testing.T, root map[string]any, path string) (map[string]any, string) {
	t.Helper()
	parts := strings.Split(path, ".")
	var value any = root
	for _, part := range parts[:len(parts)-1] {
		switch parent := value.(type) {
		case map[string]any:
			var ok bool
			value, ok = parent[part]
			if !ok {
				t.Fatalf("missing fixture path %s", path)
			}
		case []any:
			i, err := strconv.Atoi(part)
			if err != nil || i < 0 || i >= len(parent) {
				t.Fatalf("bad fixture index %s", path)
			}
			value = parent[i]
		default:
			t.Fatalf("non-container fixture path %s", path)
		}
	}
	parent, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("non-object fixture parent %s", path)
	}
	return parent, parts[len(parts)-1]
}

func routingHistoryJSON(t *testing.T, value any) []byte {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestRoutingHistoryContract(t *testing.T) {
	fixtures := loadRoutingHistoryFixtures(t)
	const sentinel = "SYNTHETIC_PRIVATE_SENTINEL"
	decode := func(t *testing.T, raw []byte) RoutingHistoryRecord {
		t.Helper()
		got, diag, err := DecodeRoutingHistoryRecord(raw)
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if diag != (RoutingHistoryDiagnostic{}) {
			t.Fatalf("unexpected diagnostic: %+v", diag)
		}
		return got
	}
	reject := func(t *testing.T, raw []byte, category string) {
		t.Helper()
		got, diag, err := DecodeRoutingHistoryRecord(raw)
		if err == nil {
			t.Fatal("accepted malformed history")
		}
		var safe RoutingHistoryDiagnostic
		if !errors.As(err, &safe) || safe.Count != 1 ||
			!slicesContainsHistoryCategory(safe.Category) || (category != "" && safe.Category != category) {
			t.Fatalf("unsafe/wrong category: %v", err)
		}
		if err.Error() != safe.Category || strings.Contains(err.Error(), sentinel) || diag != (RoutingHistoryDiagnostic{}) || !reflect.DeepEqual(got, RoutingHistoryRecord{}) {
			t.Fatal("failure exposed raw text or a partial result")
		}
		prior := decode(t, fixtures[0].Record)
		receiver := decode(t, fixtures[0].Record)
		if err := receiver.UnmarshalJSON(raw); err == nil || !reflect.DeepEqual(receiver, prior) {
			t.Fatal("malformed decode changed the receiver")
		}
	}
	mutate := func(t *testing.T, fixture int, path string, value any, remove bool) []byte {
		t.Helper()
		root := routingHistoryObject(t, fixtures[fixture].Record)
		parent, key := routingHistoryParent(t, root, path)
		if remove {
			delete(parent, key)
		} else {
			parent[key] = value
		}
		return routingHistoryJSON(t, root)
	}
	bad := func(name string, fixture int, path string, value any) {
		t.Helper()
		t.Run(name, func(t *testing.T) { reject(t, mutate(t, fixture, path, value, false), "") })
	}
	good := func(name string, fixture int, path string, value any) {
		t.Helper()
		t.Run(name, func(t *testing.T) { decode(t, mutate(t, fixture, path, value, false)) })
	}

	t.Run("semantic corpus round trip and nullable fidelity", func(t *testing.T) {
		for _, fixture := range fixtures {
			t.Run(fixture.Name, func(t *testing.T) {
				first := decode(t, fixture.Record)
				encoded := routingHistoryJSON(t, first)
				second := decode(t, encoded)
				if !reflect.DeepEqual(first, second) || !reflect.DeepEqual(routingHistoryObject(t, fixture.Record), routingHistoryObject(t, encoded)) {
					t.Fatal("semantic values, required keys or nulls changed")
				}
			})
		}
		local := decode(t, fixtures[0].Record)
		if local.Request.RequestID != "9223372036854775807" || local.Attempts[0].Outcome != "rejected" || local.Attempts[1].Outcome != "committed" ||
			local.Attempts[1].Waits.AdmissionMs != 350 || local.Attempts[1].Waits.FirstByteMs == nil || *local.Attempts[1].Waits.FirstByteMs != 1200 ||
			local.Served == nil || local.Served.Engine == nil || *local.Served.Engine != "llamacpp" {
			t.Fatal("local fixture facts changed")
		}
		api := decode(t, fixtures[1].Record)
		if api.Spend.CeilingNanoUSD == nil || *api.Spend.CeilingNanoUSD != "1000000000" || api.Spend.GrantedNanoUSD == nil || *api.Spend.GrantedNanoUSD != "250000000" ||
			api.Spend.HeldNanoUSD == nil || *api.Spend.HeldNanoUSD != "250000000" || api.Spend.ChargedNanoUSD == nil || *api.Spend.ChargedNanoUSD != "123456789" ||
			api.Spend.Accounting != "retained" || api.Attempts[0].Account == nil || api.Attempts[0].Account.Accounting != "retained" {
			t.Fatal("API accounting fixture changed")
		}
		cancelled := decode(t, fixtures[2].Record)
		if cancelled.DroppedAttempts != 2 || cancelled.Attempts[0].Index != 2 {
			t.Fatal("dropped prefix fixture changed")
		}
		paused := decode(t, fixtures[3].Record)
		linked := decode(t, fixtures[4].Record)
		if !paused.Recoverable || linked.Attempts[1].ContinuationOf == nil || *linked.Attempts[1].ContinuationOf != 0 {
			t.Fatal("synthetic recovery fixture changed")
		}
	})

	t.Run("every required member and nonnullable position", func(t *testing.T) {
		// Literal contract inventory, independent of the production struct tags.
		objects := []struct {
			fixture        int
			prefix         string
			keys, nullable string
		}{
			{0, "", "schema request workload seq requested createdAt endedAt state reason status served recoverable spend droppedAttempts attempts", "workload endedAt status served"},
			{0, "request.", "originNodeId proxyRunId requestId", ""},
			{0, "workload.", "originNodeId proxyRunId engine id attemptIndex attached", ""},
			{0, "requested.", "model tier mode", ""},
			{0, "served.", "kind model nodeId engine deploymentId accountId", "nodeId engine deploymentId accountId"},
			{0, "spend.", "currency ceilingNanoUsd grantedNanoUsd heldNanoUsd chargedNanoUsd accounting", "ceilingNanoUsd grantedNanoUsd heldNanoUsd chargedNanoUsd"},
			{0, "attempts.0.", "index startedAt endedAt destination outcome reason status waits continuationOf account", "endedAt status continuationOf account"},
			{0, "attempts.0.destination.", "kind model nodeId engine deploymentId accountId", "nodeId engine deploymentId accountId"},
			{1, "attempts.0.destination.", "kind model nodeId engine deploymentId accountId", "nodeId engine deploymentId accountId"},
			{0, "attempts.0.waits.", "admissionMs firstByteMs", "firstByteMs"},
			{1, "attempts.0.account.", "state heldNanoUsd chargedNanoUsd accounting", "heldNanoUsd chargedNanoUsd"},
		}
		for _, object := range objects {
			for _, key := range strings.Fields(object.keys) {
				path := object.prefix + key
				t.Run(strconv.Itoa(object.fixture)+" missing "+path, func(t *testing.T) {
					reject(t, mutate(t, object.fixture, path, nil, true), "invalid-shape")
				})
				if !strings.Contains(" "+object.nullable+" ", " "+key+" ") {
					t.Run(strconv.Itoa(object.fixture)+" null "+path, func(t *testing.T) {
						reject(t, mutate(t, object.fixture, path, nil, false), "invalid-shape")
					})
				}
				t.Run(strconv.Itoa(object.fixture)+" wrong type "+path, func(t *testing.T) {
					reject(t, mutate(t, object.fixture, path, []any{sentinel}, false), "invalid-shape")
				})
			}
		}
	})
	good("unavailable account hold", 1, "attempts.0.account.heldNanoUsd", nil)
	good("unavailable account charge", 1, "attempts.0.account.chargedNanoUsd", nil)
	good("unavailable first byte", 1, "attempts.0.waits.firstByteMs", nil)
	good("unavailable request status", 1, "status", nil)
	good("unavailable attempt status", 1, "attempts.0.status", nil)
	good("workload association absent", 0, "workload", nil)

	t.Run("closed enums accept all declared members", func(t *testing.T) {
		for _, value := range []string{"chat", "completion", "embedding", "other"} {
			good("mode "+value, 1, "requested.mode", value)
		}
		for _, value := range []string{"", "weak", "medium", "strong"} {
			good("tier "+value, 1, "requested.tier", value)
		}
		for _, value := range []string{"ok", "cooldown", "frozen", "credential-failure", "exhausted", "probe", "conflict", "unknown"} {
			good("account state "+value, 1, "attempts.0.account.state", value)
		}
		for _, value := range []string{"none", "pending", "recorded", "retained", "failed", "unknown"} {
			good("spend accounting "+value, 1, "spend.accounting", value)
			good("attempt accounting "+value, 1, "attempts.0.account.accounting", value)
		}
		for _, value := range []string{"pending", "committed", "rejected", "failed", "redispatched", "interrupted", "cancelled"} {
			good("outcome "+value, 1, "attempts.0.outcome", value)
		}
		for _, value := range []string{"", "unknown", "client-gone", "cancel", "regenerate", "node-paused", "proxy-shutdown", "proxy-exited", "target-left", "transport", "upstream-status", "no-content", "stream-interrupted", "upstream-error", "admission-paused", "admission-draining", "admission-busy", "admission-no-fit", "admission-engine-off", "admission-wake-timeout", "engine-not-running", "request-invalid", "no-owner", "regenerate-exhausted", "attempts-exhausted", "deadline", "request-ceiling", "account-ceiling", "account-cooldown", "account-frozen", "account-credential", "account-exhausted", "account-conflict", "ledger-unavailable", "deployment-quarantined", "peer-framing"} {
			good("request reason "+value, 1, "reason", value)
			good("attempt reason "+value, 1, "attempts.0.reason", value)
		}
		for _, state := range []string{"routing", "streaming", "completed", "failed", "cancelled"} {
			t.Run("request state "+state, func(t *testing.T) {
				root := routingHistoryObject(t, fixtures[0].Record)
				root["state"] = state
				if state != "completed" {
					root["served"] = nil
				}
				if state == "routing" || state == "streaming" {
					root["endedAt"] = nil
				}
				decode(t, routingHistoryJSON(t, root))
			})
		}
	})
	for _, path := range []string{"state", "reason", "requested.mode", "requested.tier", "spend.accounting", "attempts.0.outcome", "attempts.0.reason", "attempts.0.account.state", "attempts.0.account.accounting", "attempts.0.destination.kind"} {
		bad("reject unknown enum "+path, 1, path, sentinel)
	}
	bad("unsupported schema", 1, "schema", 2)
	bad("currency closed", 1, "spend.currency", "EUR")
	bad("served destination kind closed", 0, "served.kind", sentinel)

	t.Run("numeric domains exact at every position", func(t *testing.T) {
		positions := []struct {
			fixture int
			path    string
		}{
			{1, "schema"}, {1, "seq"}, {1, "createdAt"}, {1, "endedAt"}, {1, "status"}, {1, "droppedAttempts"},
			{0, "workload.attemptIndex"}, {1, "attempts.0.index"}, {1, "attempts.0.startedAt"}, {1, "attempts.0.endedAt"},
			{1, "attempts.0.status"}, {1, "attempts.0.waits.admissionMs"}, {1, "attempts.0.waits.firstByteMs"}, {4, "attempts.1.continuationOf"},
		}
		for _, position := range positions {
			for _, value := range []any{json.Number("-1"), json.Number("0.1"), json.Number("9007199254740992"), json.Number("9007199254740991.0000001"), json.Number("1e999999999999999999999999999"), sentinel} {
				bad("invalid "+position.path+" "+string(routingHistoryJSON(t, value)), position.fixture, position.path, value)
			}
		}
		for _, value := range []json.Number{"9007199254740991", "90071992547409910e-1", "9007199254740991.0", "1e0", "1.0000"} {
			good("exact seq "+string(value), 1, "seq", value)
		}
		good("known zero wait", 1, "attempts.0.waits.admissionMs", json.Number("0e999999999999999999999"))
		good("semantic negative zero wait", 1, "attempts.0.waits.admissionMs", json.Number("-0.0"))
		good("safe wait maximum", 1, "attempts.0.waits.admissionMs", json.Number("9007199254740991"))
		bad("zero seq", 1, "seq", 0)
		for _, path := range []string{"status", "attempts.0.status"} {
			good("status lower "+path, 1, path, 100)
			good("status upper "+path, 1, path, 599)
			bad("status too low "+path, 1, path, 99)
			bad("status too high "+path, 1, path, 600)
		}
	})

	t.Run("money domains at every amount", func(t *testing.T) {
		for _, path := range []string{"spend.ceilingNanoUsd", "spend.grantedNanoUsd", "spend.heldNanoUsd", "spend.chargedNanoUsd", "attempts.0.account.heldNanoUsd", "attempts.0.account.chargedNanoUsd"} {
			for _, value := range []any{"", "01", "-1", "+1", " 1", "1 ", "1.0", "1e0", "9223372036854775808", sentinel, 1, true} {
				bad("money "+path+" "+string(routingHistoryJSON(t, value)), 1, path, value)
			}
			for _, value := range []any{nil, "0", "9223372036854775807"} {
				good("money accepted "+path+" "+string(routingHistoryJSON(t, value)), 1, path, value)
			}
		}
	})

	t.Run("validated metadata", func(t *testing.T) {
		for _, path := range []string{"request.originNodeId", "request.proxyRunId", "workload.originNodeId", "workload.proxyRunId", "workload.engine", "workload.id", "served.nodeId", "served.engine", "attempts.0.destination.nodeId", "attempts.0.destination.engine"} {
			for _, value := range []string{"", "-bad", "white space", "日本語", strings.Repeat("a", 129), "bad\n"} {
				bad("metadata "+path+" "+strconv.Quote(value), 0, path, value)
			}
		}
		for _, path := range []string{"attempts.0.destination.deploymentId", "attempts.0.destination.accountId"} {
			for _, value := range []string{"", "bad/key", strings.Repeat("a", 129)} {
				bad("API identity "+path+" "+strconv.Quote(value), 1, path, value)
			}
			good("API identity max "+path, 1, path, "a"+strings.Repeat("._:-", 31)+"abc")
		}
		for _, value := range []any{"0", "01", "-1", "1.0", "1e0", "9223372036854775808", 1} {
			bad("request id "+string(routingHistoryJSON(t, value)), 1, "request.requestId", value)
		}
		for _, path := range []string{"requested.model", "served.model", "attempts.0.destination.model"} {
			for _, value := range []string{strings.Repeat("a", 513), "bad\nmodel", "bad\x00model", "bad\u0085model"} {
				bad("model "+path+" "+strconv.Quote(value), 0, path, value)
			}
		}
		good("requested UTF8 byte bound", 1, "requested.model", strings.Repeat("é", 256))
		bad("destination model empty", 1, "attempts.0.destination.model", "")
		bad("served model empty", 0, "served.model", "")
		t.Run("raw invalid UTF8", func(t *testing.T) {
			raw := strings.Replace(string(fixtures[1].Record), "synthetic-model", string([]byte{0xff}), 1)
			reject(t, []byte(raw), "invalid-value")
		})
	})

	t.Run("requested model emptiness follows retained lifecycle facts", func(t *testing.T) {
		// Synthetic prevalidation rejection: no attempt, dropped prefix or
		// association exists. Empty model is unavailable validated metadata.
		early := []byte(`{"schema":1,"request":{"originNodeId":"synthetic-origin","proxyRunId":"synthetic-run","requestId":"6"},"workload":null,"seq":1,"requested":{"model":"","tier":"","mode":"other"},"createdAt":1700000050000,"endedAt":1700000050001,"state":"failed","reason":"request-invalid","status":400,"served":null,"recoverable":false,"spend":{"currency":"USD","ceilingNanoUsd":null,"grantedNanoUsd":null,"heldNanoUsd":null,"chargedNanoUsd":null,"accounting":"none"},"droppedAttempts":0,"attempts":[]}`)
		for _, state := range []string{"routing", "failed", "cancelled"} {
			t.Run("prevalidation "+state, func(t *testing.T) {
				root := routingHistoryObject(t, early)
				root["state"] = state
				if state == "routing" {
					root["endedAt"] = nil
					root["reason"] = ""
				} else if state == "cancelled" {
					root["reason"] = "cancel"
				}
				raw := routingHistoryJSON(t, root)
				got := decode(t, raw)
				if got.Requested.Model != "" || got.DroppedAttempts != 0 || len(got.Attempts) != 0 || got.Workload != nil || got.Served != nil {
					t.Fatal("prevalidation record lost its empty model or gained lifecycle facts")
				}
				initial, diag, err := DecodeRoutingHistoryInitial([]byte(`{"requests":[` + string(raw) + `]}`))
				if err != nil || diag != (RoutingHistoryDiagnostic{}) || len(initial.Requests) != 1 || !reflect.DeepEqual(initial.Requests[0], got) {
					t.Fatalf("prevalidation initial decode: %+v %v", diag, err)
				}
				if !reflect.DeepEqual(decode(t, routingHistoryJSON(t, got)), got) {
					t.Fatal("prevalidation empty model changed on round trip")
				}
			})
		}
		for _, fixture := range []int{0, 1, 2, 3, 4} {
			t.Run("validated model required for "+fixtures[fixture].Name, func(t *testing.T) {
				decode(t, fixtures[fixture].Record)
				reject(t, mutate(t, fixture, "requested.model", "", false), "invalid-value")
			})
		}
		t.Run("trimmed prefix still establishes validation", func(t *testing.T) {
			root := routingHistoryObject(t, early)
			root["droppedAttempts"] = 2
			parent, key := routingHistoryParent(t, root, "requested.model")
			parent[key] = "synthetic-model"
			decode(t, routingHistoryJSON(t, root))
			parent[key] = ""
			reject(t, routingHistoryJSON(t, root), "invalid-value")
		})
		t.Run("streaming alone establishes validation", func(t *testing.T) {
			root := routingHistoryObject(t, early)
			root["state"] = "streaming"
			root["endedAt"] = nil
			root["reason"] = ""
			parent, key := routingHistoryParent(t, root, "requested.model")
			parent[key] = "synthetic-model"
			decode(t, routingHistoryJSON(t, root))
			parent[key] = ""
			reject(t, routingHistoryJSON(t, root), "invalid-value")
		})
	})

	t.Run("destination and account exclusivity", func(t *testing.T) {
		for _, path := range []string{"served.nodeId", "served.engine", "attempts.0.destination.nodeId", "attempts.0.destination.engine"} {
			bad("local required "+path, 0, path, nil)
		}
		for _, path := range []string{"served.deploymentId", "served.accountId", "attempts.0.destination.deploymentId", "attempts.0.destination.accountId"} {
			bad("local excludes "+path, 0, path, "synthetic-extra")
		}
		for _, path := range []string{"attempts.0.destination.nodeId", "attempts.0.destination.engine"} {
			bad("API excludes "+path, 1, path, "synthetic-extra")
		}
		for _, path := range []string{"attempts.0.destination.deploymentId", "attempts.0.destination.accountId", "attempts.0.account"} {
			bad("API requires "+path, 1, path, nil)
		}
		account := routingHistoryObject(t, fixtures[1].Record)
		parent, key := routingHistoryParent(t, account, "attempts.0.account")
		bad("local excludes account", 0, "attempts.0.account", parent[key])
	})

	t.Run("terminal delivery recoverability and temporal relations", func(t *testing.T) {
		bad("completed requires served", 0, "served", nil)
		bad("served must match delivered destination", 0, "served.nodeId", "synthetic-other")
		bad("served needs committed attempt", 0, "attempts.1.outcome", "failed")
		for _, fixture := range []int{0, 1, 2} {
			t.Run("terminal ended required "+strconv.Itoa(fixture), func(t *testing.T) { reject(t, mutate(t, fixture, "endedAt", nil, false), "invalid-value") })
			t.Run("terminal unrecoverable "+strconv.Itoa(fixture), func(t *testing.T) { reject(t, mutate(t, fixture, "recoverable", true, false), "invalid-value") })
		}
		bad("active cannot end", 3, "endedAt", json.Number("1700000035000"))
		bad("recoverable must stream", 3, "state", "routing")
		bad("recoverable needs interrupted predecessor", 3, "attempts.0.outcome", "failed")
		bad("recoverable cannot have no attempts", 3, "attempts", []any{})
		bad("linked interruption no longer awaits", 4, "recoverable", true)
		bad("request end before start", 1, "endedAt", 0)
		bad("attempt start before request", 1, "attempts.0.startedAt", 0)
		bad("attempt end before start", 1, "attempts.0.endedAt", 0)
		bad("attempt end beyond request", 1, "attempts.0.endedAt", json.Number("1700000013000"))
		bad("terminal attempt cannot remain open", 1, "attempts.0.endedAt", nil)
		for _, outcome := range []string{"rejected", "failed", "redispatched", "interrupted", "cancelled"} {
			t.Run("ended attempt "+outcome, func(t *testing.T) {
				root := routingHistoryObject(t, fixtures[3].Record)
				root["recoverable"] = false
				p, k := routingHistoryParent(t, root, "attempts.0.outcome")
				p[k] = outcome
				p, k = routingHistoryParent(t, root, "attempts.0.endedAt")
				p[k] = nil
				reject(t, routingHistoryJSON(t, root), "invalid-value")
			})
		}
		t.Run("completed delivery survives failed accounting", func(t *testing.T) {
			root := routingHistoryObject(t, fixtures[1].Record)
			root["state"] = "completed"
			root["reason"] = ""
			p, k := routingHistoryParent(t, root, "attempts.0.destination")
			root["served"] = p[k]
			p, k = routingHistoryParent(t, root, "attempts.0.outcome")
			p[k] = "committed"
			p, k = routingHistoryParent(t, root, "spend.accounting")
			p[k] = "failed"
			p, k = routingHistoryParent(t, root, "attempts.0.account.accounting")
			p[k] = "failed"
			got := decode(t, routingHistoryJSON(t, root))
			if got.State != "completed" || got.Spend.Accounting != "failed" {
				t.Fatal("accounting rewrote delivery")
			}
		})
	})

	t.Run("continuation indexes and workload relations", func(t *testing.T) {
		bad("indexes contiguous from dropped prefix", 0, "attempts.1.index", 3)
		bad("dropped prefix cannot skip retained index", 0, "droppedAttempts", 1)
		bad("continuation must precede successor", 4, "attempts.1.continuationOf", 1)
		bad("continuation cannot target future", 4, "attempts.0.continuationOf", 1)
		bad("continuation requires interrupted predecessor", 4, "attempts.0.outcome", "failed")
		good("producer verified trimmed predecessor", 2, "attempts.0.continuationOf", 1)
		bad("trimmed reference cannot equal first retained", 2, "attempts.0.continuationOf", 2)
		bad("trimmed reference cannot target absent future", 2, "attempts.0.continuationOf", 3)
		bad("workload origin must match", 0, "workload.originNodeId", "synthetic-other")
		bad("workload run must match", 0, "workload.proxyRunId", "synthetic-other")
		bad("workload engine must match local", 0, "workload.engine", "ollama")
		bad("workload attempt must exist", 0, "workload.attemptIndex", 2)
		bad("trimmed workload cannot be attached", 2, "workload.attached", true)
		good("retained local workload attached", 0, "workload.attached", true)
		t.Run("API attempt cannot own workload", func(t *testing.T) {
			root := routingHistoryObject(t, fixtures[1].Record)
			root["workload"] = map[string]any{"originNodeId": "synthetic-origin", "proxyRunId": "synthetic-run", "engine": "llamacpp", "id": "synthetic-job", "attemptIndex": 0, "attached": false}
			reject(t, routingHistoryJSON(t, root), "invalid-value")
		})
	})

	t.Run("raw shape duplicates and safe diagnostics", func(t *testing.T) {
		for _, raw := range []string{"", "null", "[]", "1", `"` + sentinel + `"`, "{}", string(fixtures[0].Record) + " {}", `{"` + sentinel + `":`, `{"schema":1,"schema":1}`, `{"schema":1,"sch\u0065ma":1}`, `{"` + sentinel + `":{"nested":1,"nested":2}}`, `{"` + sentinel + `":{"x":1,"\u0078":2}}`} {
			t.Run(strconv.Quote(raw[:min(len(raw), 80)]), func(t *testing.T) { reject(t, []byte(raw), "") })
		}
		for _, prefix := range []string{"", "request.", "requested.", "workload.", "served.", "spend.", "attempts.0.", "attempts.0.destination.", "attempts.0.waits."} {
			t.Run("unknown subtree duplicate at "+prefix, func(t *testing.T) {
				root := routingHistoryObject(t, fixtures[0].Record)
				p, k := routingHistoryParent(t, root, prefix+sentinel)
				p[k] = json.RawMessage(`{"secret":1,"secr\u0065t":2}`)
				reject(t, routingHistoryJSON(t, root), "duplicate-field")
			})
		}
	})

	t.Run("unknowns stripped at every known projection", func(t *testing.T) {
		for _, item := range []struct {
			fixture  int
			prefixes []string
		}{
			{0, []string{"", "request.", "requested.", "workload.", "served.", "spend.", "attempts.0.", "attempts.0.destination.", "attempts.0.waits.", "attempts.1.", "attempts.1.destination.", "attempts.1.waits."}},
			{1, []string{"", "request.", "requested.", "spend.", "attempts.0.", "attempts.0.destination.", "attempts.0.waits.", "attempts.0.account."}},
		} {
			t.Run(strconv.Itoa(item.fixture), func(t *testing.T) {
				root := routingHistoryObject(t, fixtures[item.fixture].Record)
				for _, prefix := range item.prefixes {
					p, k := routingHistoryParent(t, root, prefix+sentinel)
					p[k] = map[string]any{"unknown-inner": sentinel}
				}
				got, diag, err := DecodeRoutingHistoryRecord(routingHistoryJSON(t, root))
				if err != nil {
					t.Fatal(err)
				}
				if diag.Category != "unknown-fields-stripped" || diag.Count != len(item.prefixes) || !reflect.DeepEqual(got, decode(t, fixtures[item.fixture].Record)) {
					t.Fatalf("wrong stripping/result: %+v", diag)
				}
				if !reflect.DeepEqual(routingHistoryObject(t, routingHistoryJSON(t, got)), routingHistoryObject(t, fixtures[item.fixture].Record)) {
					t.Fatal("unknown key/value survived")
				}
			})
		}
	})
	t.Run("decoded payload owns its metadata and slices", func(t *testing.T) {
		raw := append([]byte(nil), fixtures[0].Record...)
		got := decode(t, raw)
		want := decode(t, fixtures[0].Record)
		for i := range raw {
			raw[i] = 'x'
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatal("input buffer mutation changed retained projection")
		}
		other := decode(t, fixtures[0].Record)
		other.Attempts[1].Outcome = "failed"
		*other.Attempts[1].Destination.Engine = "ollama"
		other.Workload.ID = "synthetic-other"
		if !reflect.DeepEqual(got, want) {
			t.Fatal("decodes share mutable payload storage")
		}
	})

	t.Run("raw bounds before projection", func(t *testing.T) {
		t.Run("record size boundary", func(t *testing.T) {
			raw := append([]byte(nil), fixtures[0].Record...)
			raw = append(raw, []byte(strings.Repeat(" ", 128*1024-len(raw)))...)
			decode(t, raw)
			reject(t, append(raw, ' '), "limit-exceeded")
		})
		for _, depth := range []int{15, 16} {
			t.Run("unknown depth "+strconv.Itoa(depth), func(t *testing.T) {
				raw := strings.TrimSpace(string(fixtures[0].Record))
				raw = raw[:len(raw)-1] + `,"` + sentinel + `":` + strings.Repeat("[", depth) + "0" + strings.Repeat("]", depth) + "}"
				if depth == 16 {
					reject(t, []byte(raw), "limit-exceeded")
				} else {
					_, diag, err := DecodeRoutingHistoryRecord([]byte(raw))
					if err != nil || diag.Count != 1 {
						t.Fatalf("legal depth: %+v %v", diag, err)
					}
				}
			})
		}
		t.Run("unknown members count toward limit", func(t *testing.T) {
			// An early empty routing record has a fixed literal member inventory.
			base := `{"schema":1,"request":{"originNodeId":"o","proxyRunId":"r","requestId":"1"},"workload":null,"seq":1,"requested":{"model":"","tier":"","mode":"other"},"createdAt":0,"endedAt":null,"state":"routing","reason":"","status":null,"served":null,"recoverable":false,"spend":{"currency":"USD","ceilingNanoUsd":null,"grantedNanoUsd":null,"heldNanoUsd":null,"chargedNanoUsd":null,"accounting":"none"},"droppedAttempts":0,"attempts":[]}`
			// 15 root + 3 key + 3 requested + 6 spend = 27 members.
			var extra strings.Builder
			for i := 0; i < 2048-27; i++ {
				extra.WriteString(`,"u` + strconv.Itoa(i) + `":null`)
			}
			raw := base[:len(base)-1] + extra.String() + "}"
			_, diag, err := DecodeRoutingHistoryRecord([]byte(raw))
			if err != nil || diag.Count != 2021 {
				t.Fatalf("member boundary: %+v %v", diag, err)
			}
			reject(t, []byte(raw[:len(raw)-1]+`,"`+sentinel+`":null}`), "limit-exceeded")
		})
		t.Run("attempt cap", func(t *testing.T) {
			root := routingHistoryObject(t, fixtures[1].Record)
			original, ok := root["attempts"].([]any)
			if !ok || len(original) != 1 {
				t.Fatal("invalid attempt seed")
			}
			seed := routingHistoryJSON(t, original[0])
			attempts := make([]any, 0, 33)
			for i := 0; i < 33; i++ {
				a := routingHistoryObject(t, seed)
				a["index"] = i
				attempts = append(attempts, a)
				if i == 31 {
					root["attempts"] = attempts
					decode(t, routingHistoryJSON(t, root))
				}
			}
			root["attempts"] = attempts
			reject(t, routingHistoryJSON(t, root), "limit-exceeded")
		})
	})

	t.Run("initial envelope bounds stripping and atomicity", func(t *testing.T) {
		initial := RoutingHistoryInitial{Requests: []RoutingHistoryRecord{decode(t, fixtures[0].Record), decode(t, fixtures[1].Record)}}
		valid := routingHistoryJSON(t, initial)
		got, diag, err := DecodeRoutingHistoryInitial(valid)
		if err != nil || diag != (RoutingHistoryDiagnostic{}) || !reflect.DeepEqual(got, initial) {
			t.Fatalf("initial roundtrip: %+v %v", diag, err)
		}
		rejectInitial := func(t *testing.T, raw []byte, category string) {
			t.Helper()
			value, diag, err := DecodeRoutingHistoryInitial(raw)
			var safe RoutingHistoryDiagnostic
			if err == nil || !errors.As(err, &safe) || safe.Category != category || safe.Count != 1 || err.Error() != category || diag != (RoutingHistoryDiagnostic{}) || value.Requests != nil {
				t.Fatalf("initial rejection: %+v %v", diag, err)
			}
			receiver := initial
			if err := receiver.UnmarshalJSON(raw); err == nil || !reflect.DeepEqual(receiver, initial) {
				t.Fatal("initial failure changed receiver")
			}
		}
		for _, raw := range []string{`null`, `[]`, `{}`, `{"requests":null}`, `{"requests":{}}`, `{"requests":[null]}`, string(valid) + " {}"} {
			rejectInitial(t, []byte(raw), "invalid-shape")
		}
		rejectInitial(t, []byte(`{"requests":[],"requ\u0065sts":[]}`), "duplicate-field")
		rejectInitial(t, []byte(`{"requests":[],"`+sentinel+`":{"x":1,"x":2}}`), "duplicate-field")
		duplicateRecord := strings.TrimSpace(string(fixtures[0].Record))
		duplicateRecord = duplicateRecord[:len(duplicateRecord)-1] + `,"sch\u0065ma":1}`
		rejectInitial(t, []byte(`{"requests":[`+duplicateRecord+`]}`), "duplicate-field")
		unknown := `{"requests":[` + string(fixtures[0].Record) + `],"` + sentinel + `":{"secret":"` + sentinel + `"}}`
		got, diag, err = DecodeRoutingHistoryInitial([]byte(unknown))
		if err != nil || diag.Category != "unknown-fields-stripped" || diag.Count != 1 || len(got.Requests) != 1 || !reflect.DeepEqual(got.Requests[0], initial.Requests[0]) {
			t.Fatalf("initial stripping: %+v %v", diag, err)
		}
		nestedUnknown := strings.TrimSpace(string(fixtures[0].Record))
		nestedUnknown = nestedUnknown[:len(nestedUnknown)-1] + `,"` + sentinel + `":null}`
		got, diag, err = DecodeRoutingHistoryInitial([]byte(`{"requests":[` + nestedUnknown + `],"` + sentinel + `":null}`))
		if err != nil || diag.Count != 2 || diag.Category != "unknown-fields-stripped" || !reflect.DeepEqual(got.Requests, initial.Requests[:1]) {
			t.Fatalf("nested initial stripping: %+v %v", diag, err)
		}
		empty, _, err := DecodeRoutingHistoryInitial([]byte(`{"requests":[]}`))
		if err != nil || empty.Requests == nil || len(empty.Requests) != 0 {
			t.Fatal("empty requests must stay nonnull")
		}
		many := `{"requests":[` + strings.Repeat(string(fixtures[1].Record)+",", 255) + string(fixtures[1].Record) + "]}"
		got, _, err = DecodeRoutingHistoryInitial([]byte(many))
		if err != nil || len(got.Requests) != 256 {
			t.Fatalf("record cap: %v", err)
		}
		rejectInitial(t, []byte(many[:len(many)-2]+","+string(fixtures[1].Record)+"]}"), "limit-exceeded")
		padded := append([]byte(`{"requests":[]}`), []byte(strings.Repeat(" ", 32*1024*1024-len(`{"requests":[]}`)))...)
		if _, _, err := DecodeRoutingHistoryInitial(padded); err != nil {
			t.Fatalf("legal initial byte cap: %v", err)
		}
		rejectInitial(t, append(padded, ' '), "limit-exceeded")
		record := strings.TrimSpace(string(fixtures[0].Record))
		oversized := record[:len(record)-1] + `,"` + sentinel + `":"` + strings.Repeat("x", 128*1024) + `"}`
		rejectInitial(t, []byte(`{"requests":[`+oversized+`]}`), "limit-exceeded")
		deepRecord := record[:len(record)-1] + `,"` + sentinel + `":` + strings.Repeat("[", 16) + "0" + strings.Repeat("]", 16) + "}"
		rejectInitial(t, []byte(`{"requests":[`+deepRecord+`]}`), "limit-exceeded")
		var memberRecord strings.Builder
		memberRecord.WriteString(record[:len(record)-1])
		for i := 0; i < 2048; i++ {
			memberRecord.WriteString(`,"u` + strconv.Itoa(i) + `":null`)
		}
		memberRecord.WriteString("}")
		rejectInitial(t, []byte(`{"requests":[`+memberRecord.String()+`]}`), "limit-exceeded")
	})
}

func slicesContainsHistoryCategory(category string) bool {
	switch category {
	case "invalid-shape", "invalid-value", "duplicate-field", "limit-exceeded":
		return true
	default:
		return false
	}
}

func TestHistoryIntegerScalarForms(t *testing.T) {
	longOne := "0." + strings.Repeat("0", 66000) + "1E66001"
	if len(longOne) != 66009 {
		t.Fatalf("counterexample token bytes = %d", len(longOne))
	}
	tests := []struct {
		name, token string
		want        int64
		ok          bool
	}{
		{name: "long integral decimal", token: longOne, want: 1, ok: true},
		{name: "long maximum safe", token: "0." + strings.Repeat("0", 66000) + "9007199254740991E66016", want: routingHistorySafeInteger, ok: true},
		{name: "long unsafe integer neighbor", token: "0." + strings.Repeat("0", 66000) + "9007199254740992E66016"},
		{name: "long negative nonzero", token: "-" + longOne},
		{name: "long fractional neighbor", token: "0." + strings.Repeat("0", 66000) + "1E66000"},
		{name: "long exponent bound", token: "0." + strings.Repeat("0", 66000) + "1E131073"},
		{name: "exponent overflow", token: "1E9223372036854775808"},
		{name: "negative exponent overflow", token: "1E-9223372036854775809"},
		{name: "maximum safe", token: "9007199254740991.00e0", want: routingHistorySafeInteger, ok: true},
		{name: "unsafe integer", token: "9007199254740992.00e0"},
		{name: "zero with enormous exponent", token: "0e" + strings.Repeat("9", 66000), ok: true},
		{name: "negative zero with enormous exponent", token: "-0.000e-" + strings.Repeat("9", 66000), ok: true},
		{name: "long zero fraction", token: "0." + strings.Repeat("0", 66000), ok: true},
		{name: "long leading zero exponent", token: "1E" + strings.Repeat("0", 66000) + "1", want: 10, ok: true},
		{name: "positive bound", token: "1e131072"},
		{name: "negative bound", token: "1e-131072"},
		{name: "trim fractional zeroes", token: "12.300e1", want: 123, ok: true},
		{name: "positive exponent sign", token: "1e+3", want: 1000, ok: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !json.Valid([]byte(tt.token)) {
				t.Fatal("numeric fixture is not valid JSON")
			}
			n, ok := historyInteger(tt.token)
			if ok != tt.ok || (ok && n != tt.want) {
				t.Fatalf("string integer = %d / %v, want %d / %v", n, ok, tt.want, tt.ok)
			}
			raw := []byte(tt.token)
			n, ok = historyTransportInteger(raw)
			if ok != tt.ok || (ok && n != tt.want) {
				t.Fatalf("raw integer = %d / %v, want %d / %v", n, ok, tt.want, tt.ok)
			}
			if allocations := testing.AllocsPerRun(1, func() { historyInteger(tt.token); historyTransportInteger(raw) }); allocations != 0 {
				t.Fatalf("scalar conversions allocated: %g", allocations)
			}
		})
	}
}

func TestHistoryRecordAdapterAttemptPresenceAndLimit(t *testing.T) {
	row := RoutingHistoryRecord{Schema: 1, Request: RoutingRequestKey{OriginNodeID: "origin", ProxyRunID: "run", RequestID: "1"},
		Seq: 1, State: "routing", Requested: RoutingHistoryRequested{Mode: "chat"},
		Spend: RoutingHistorySpend{Currency: "USD", Accounting: "none"}, Attempts: []RoutingHistoryAttempt{}}
	if err := row.Validate(); err != nil {
		t.Fatalf("present empty attempts: %v", err)
	}
	row.Attempts = nil
	if err := row.Validate(); err == nil || err.Error() != "invalid-value" {
		t.Fatalf("nil attempts error = %v", err)
	}
	row.Requested.Model = "model"
	row.Attempts = make([]RoutingHistoryAttempt, RoutingHistoryAttempts+1)
	if err := row.Validate(); err == nil || err.Error() != "limit-exceeded" {
		t.Fatalf("attempt count error = %v", err)
	}
}

func TestHistorySemanticRegexpScratch(t *testing.T) {
	// MatchString requests no captures. These anchored deterministic grammars
	// use a scalar one-pass machine, so history does not grow bit-state or
	// NFA queues against C, even for shape-valid 512-byte invalid ID/money.
	for _, re := range []*regexp.Regexp{strictID, nanoUSDCanonical} {
		if reflect.ValueOf(re).Elem().FieldByName("onepass").IsNil() {
			t.Fatalf("semantic regexp needs a new scratch census: %s", re.String())
		}
	}
}
