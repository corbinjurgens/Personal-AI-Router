// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
)

func TestObservationDoesNotInventKnownValues(t *testing.T) {
	test := func(name, input string, wantKnown bool, wantValue *bool, wantError bool) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			var got Observation[bool]
			err := json.Unmarshal([]byte(input), &got)
			if (err != nil) != wantError {
				t.Fatalf("decode error = %v, want error %v", err, wantError)
			}
			if wantError {
				return
			}
			if got.Known != wantKnown || !reflect.DeepEqual(got.Value, wantValue) {
				t.Fatalf("observation = %+v", got)
			}
			encoded, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			var roundtrip Observation[bool]
			if err := json.Unmarshal(encoded, &roundtrip); err != nil {
				t.Fatalf("roundtrip: %v", err)
			}
			if !reflect.DeepEqual(roundtrip, got) {
				t.Fatalf("roundtrip = %+v, want %+v", roundtrip, got)
			}
		})
	}
	no := false
	test("unknown", `{"known":false,"value":null}`, false, nil, false)
	test("known false", `{"known":true,"value":false}`, true, &no, false)
	test("false with value", `{"known":false,"value":false}`, false, nil, true)
	test("known null", `{"known":true,"value":null}`, false, nil, true)
	test("missing value", `{"known":false}`, false, nil, true)
	test("missing known", `{"value":null}`, false, nil, true)
	test("extra field", `{"known":false,"value":null,"remote":true}`, false, nil, true)
	if _, err := json.Marshal(Observation[bool]{Known: true}); err == nil {
		t.Fatal("encoded known without value")
	}
	if _, err := json.Marshal(Known[[]string](nil)); err == nil {
		t.Fatal("encoded known null slice")
	}
}

func TestLocalRequestsRejectUnownedInputs(t *testing.T) {
	gui := GUIFacts{PID: 42, Executable: filepath.Join(packageFixtureTempDir(t), "PAIR.exe"), BuildID: ""}
	encoded, err := json.Marshal(PreviewRequest{CandidatePath: packageFixtureTempDir(t), GUI: gui})
	if err != nil {
		t.Fatalf("encode request: %v", err)
	}
	var preview PreviewRequest
	if err := DecodeRequest(encoded, &preview); err != nil {
		t.Fatalf("valid native request: %v", err)
	}
	test := func(name, field string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(encoded, &fields); err != nil {
				t.Fatal(err)
			}
			fields[field] = json.RawMessage(`"untrusted"`)
			raw, err := json.Marshal(fields)
			if err != nil {
				t.Fatal(err)
			}
			if err := DecodeRequest(raw, &PreviewRequest{}); err == nil {
				t.Fatalf("accepted %s", field)
			}
		})
	}
	test("node selector", "nodeId")
	test("remote caller", "remoteCaller")
	test("destination", "destination")
	test("command", "command")
	test("endpoint", "endpoint")
	if err := DecodeRequest([]byte(strings.Repeat(" ", MaxRequestBytes+1)), &ImpactRequest{}); err == nil {
		t.Fatal("accepted oversized request")
	}
	if err := DecodeRequest([]byte(`{"candidatePath":"relative","gui":null}`), &PreviewRequest{}); err == nil {
		t.Fatal("accepted relative path")
	}
	if err := DecodeRequest([]byte(`{} {}`), &ImpactRequest{}); err == nil {
		t.Fatal("accepted trailing JSON")
	}
	if err := DecodeRequest([]byte(`null`), &ImpactRequest{}); err == nil {
		t.Fatal("accepted null request")
	}
	if err := DecodeRequest([]byte(`{"operationId":"one","operationId":"two"}`), &StatusRequest{}); err == nil {
		t.Fatal("accepted duplicate field")
	}
	start, err := json.Marshal(StartRequest{PreviewID: "preview_1", GUI: gui, Consent: Consent{AllowEngineStop: false}})
	if err != nil {
		t.Fatal(err)
	}
	var got StartRequest
	if err := DecodeRequest(start, &got); err != nil {
		t.Fatal(err)
	}
	if got.Consent.AllowEngineStop {
		t.Fatal("invented engine-stop consent")
	}
	var nested map[string]json.RawMessage
	if err := json.Unmarshal(start, &nested); err != nil {
		t.Fatal(err)
	}
	nested["consent"] = json.RawMessage(`{"allowInterruption":true,"allowEngineStop":true,"nodeId":"peer"}`)
	bad, err := json.Marshal(nested)
	if err != nil {
		t.Fatal(err)
	}
	if err := DecodeRequest(bad, &StartRequest{}); err == nil {
		t.Fatal("accepted nested remote field")
	}
}

func TestUnavailableAndMaintenanceRefusalsDoNotAcknowledgeOwnership(t *testing.T) {
	for _, result := range []Result[BuildApplyOperation]{Unavailable[BuildApplyOperation](), MaintenanceBusy[BuildApplyOperation]()} {
		raw, err := json.Marshal(result)
		if err != nil {
			t.Fatal(err)
		}
		var decoded Result[BuildApplyOperation]
		if err := json.Unmarshal(raw, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded.Available || decoded.Value != nil || decoded.Refusal == nil || decoded.Refusal.Detail == "" {
			t.Fatalf("refusal = %+v", decoded)
		}
	}
	busy := MaintenanceBusy[BuildApplyOperation]()
	if busy.Refusal.Code != RefusalMaintenanceBusy || CodeMaintenanceBusy != -32015 {
		t.Fatalf("busy refusal = %+v", busy)
	}
}

func wireOperationFixture() BuildApplyOperation {
	return BuildApplyOperation{
		OperationID: "operation_1", PreviewID: "preview_1", OwnershipAcknowledged: true,
		Phase: PhasePrepare, ReadbackConfidence: ConfidenceUnknown,
	}
}

func TestBuildApplyStatusRoundtripPreservesStateAndCorrelation(t *testing.T) {
	pending := wireOperationFixture()
	terminal := wireOperationFixture()
	outcome := OutcomeUncertain
	terminal.Outcome = &outcome
	terminal.Failure = &BuildApplyFailure{
		Phase: PhasePrepare, Reason: "readback_unavailable", Detail: "Readback is unavailable.",
		ReadbackConfidence: ConfidenceUnknown, SafeNextAction: "Keep paused and inspect the operation.",
	}
	test := func(name string, status BuildApplyStatus) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			want := Result[BuildApplyStatus]{Available: true, Value: &status}
			raw, err := EncodeStatusResult(want)
			if err != nil {
				t.Fatalf("encode status: %v", err)
			}
			got, err := DecodeStatusResult(raw, "")
			if err != nil {
				t.Fatalf("decode status: %v", err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("status = %#v, want %#v", got, want)
			}
			var envelope map[string]json.RawMessage
			if err := json.Unmarshal(raw, &envelope); err != nil {
				t.Fatalf("decode envelope structure: %v", err)
			}
			if len(envelope) != 3 || string(envelope["available"]) != "true" || string(envelope["refusal"]) != "null" {
				t.Fatalf("envelope = %v", envelope)
			}
			var value map[string]json.RawMessage
			if err := json.Unmarshal(envelope["value"], &value); err != nil {
				t.Fatalf("decode status structure: %v", err)
			}
			var state StatusState
			if err := json.Unmarshal(value["state"], &state); err != nil {
				t.Fatalf("decode status state: %v", err)
			}
			if len(value) != 3 || state != status.State || string(value["startupAllowed"]) != fmt.Sprint(status.StartupAllowed) {
				t.Fatalf("status structure = %v", value)
			}
			if status.Operation == nil {
				if string(value["operation"]) != "null" {
					t.Fatalf("idle operation = %s", value["operation"])
				}
				return
			}
			var operation map[string]json.RawMessage
			if err := json.Unmarshal(value["operation"], &operation); err != nil {
				t.Fatalf("decode operation structure: %v", err)
			}
			var previewID string
			if err := json.Unmarshal(operation["previewId"], &previewID); err != nil {
				t.Fatalf("decode preview correlation: %v", err)
			}
			if previewID != status.Operation.PreviewID {
				t.Fatalf("preview ID = %q, want %q", previewID, status.Operation.PreviewID)
			}
			if _, err := DecodeStatusResult(raw, previewID); err != nil {
				t.Fatalf("matching preview: %v", err)
			}
			if _, err := DecodeStatusResult(raw, "preview_other"); err == nil {
				t.Fatal("accepted unrelated preview")
			}
		})
	}
	test("idle without recorded operation", BuildApplyStatus{State: StatusIdle, StartupAllowed: true})
	test("pending", BuildApplyStatus{State: StatusPending, Operation: &pending})
	test("terminal", BuildApplyStatus{State: StatusTerminal, Operation: &terminal})
	test("closed terminal", BuildApplyStatus{State: StatusTerminal, Operation: &terminal, StartupAllowed: true})
	test("unknown with pending receipt", BuildApplyStatus{State: StatusUnknown, Operation: &pending})
	test("unknown with terminal receipt", BuildApplyStatus{State: StatusUnknown, Operation: &terminal})
}

func TestBuildApplyStatusRejectsMalformedPayloads(t *testing.T) {
	operation, err := json.Marshal(wireOperationFixture())
	if err != nil {
		t.Fatalf("encode operation fixture: %v", err)
	}
	status := func(state, operation string) string {
		return `{"available":true,"value":{"state":"` + state + `","operation":` + operation + `,"startupAllowed":false},"refusal":null}`
	}
	test := func(name, input string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeStatusResult([]byte(input), ""); err == nil {
				t.Fatal("accepted malformed status")
			}
		})
	}
	badOperation := func(name, field string, value json.RawMessage) {
		t.Helper()
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(operation, &fields); err != nil {
			t.Fatalf("decode operation fixture: %v", err)
		}
		if value == nil {
			delete(fields, field)
		} else {
			fields[field] = value
		}
		raw, err := json.Marshal(fields)
		if err != nil {
			t.Fatalf("encode malformed operation: %v", err)
		}
		test(name, status("unknown", string(raw)))
	}
	test("missing state", `{"available":true,"value":{"operation":null},"refusal":null}`)
	test("missing operation", `{"available":true,"value":{"state":"idle"},"refusal":null}`)
	test("missing startup permission", `{"available":true,"value":{"state":"idle","operation":null},"refusal":null}`)
	test("null startup permission", `{"available":true,"value":{"state":"idle","operation":null,"startupAllowed":null},"refusal":null}`)
	test("unknown state name", status("unreadable", string(operation)))
	test("idle carries operation", status("idle", string(operation)))
	test("pending lacks operation", status("pending", "null"))
	test("terminal lacks operation", status("terminal", "null"))
	test("unknown lacks trusted correlation", status("unknown", "null"))
	test("terminal lacks outcome", status("terminal", string(operation)))
	test("pending carries outcome", status("pending", strings.Replace(string(operation), `"outcome":null`, `"outcome":"uncertain"`, 1)))
	test("duplicate state", `{"available":true,"value":{"state":"unknown","state":"idle","operation":null},"refusal":null}`)
	test("duplicate preview correlation", status("pending", strings.Replace(string(operation), `"previewId":"preview_1"`, `"previewId":"preview_1","previewId":"preview_2"`, 1)))
	test("trailing JSON", status("idle", "null")+` {}`)
	test("null output", `null`)
	test("array output", `[]`)
	test("oversized output", strings.Repeat(" ", MaxStatusBytes)+status("idle", "null"))
	test("unavailable carries idle", `{"available":false,"value":{"state":"idle","operation":null},"refusal":{"code":"unavailable","detail":"Unreadable."}}`)
	test("available lacks value", `{"available":true,"value":null,"refusal":null}`)
	test("available carries refusal", `{"available":true,"value":{"state":"idle","operation":null},"refusal":{"code":"unavailable","detail":"Unreadable."}}`)
	test("unavailable lacks refusal", `{"available":false,"value":null,"refusal":null}`)
	test("missing availability", `{"value":null,"refusal":{"code":"unavailable","detail":"Unreadable."}}`)
	test("extra status field", `{"available":true,"value":{"state":"idle","operation":null,"previewId":"preview_1"},"refusal":null}`)
	badOperation("missing preview correlation", "previewId", nil)
	badOperation("null preview correlation", "previewId", json.RawMessage(`null`))
	badOperation("empty preview correlation", "previewId", json.RawMessage(`""`))
	badOperation("path as preview correlation", "previewId", json.RawMessage(`"../preview"`))
	badOperation("oversized preview correlation", "previewId", json.RawMessage(`"`+strings.Repeat("p", 129)+`"`))
	badOperation("missing operation ID", "operationId", nil)
	badOperation("empty operation ID", "operationId", json.RawMessage(`""`))
	badOperation("path as operation ID", "operationId", json.RawMessage(`"operation/1"`))
	badOperation("oversized operation ID", "operationId", json.RawMessage(`"`+strings.Repeat("o", 129)+`"`))
	badOperation("missing acknowledgement", "ownershipAcknowledged", nil)
	badOperation("null acknowledgement", "ownershipAcknowledged", json.RawMessage(`null`))
	badOperation("unknown phase", "phase", json.RawMessage(`"done"`))
	badOperation("unknown outcome", "outcome", json.RawMessage(`"done"`))
	badOperation("unknown confidence", "readbackConfidence", json.RawMessage(`"complete"`))
	badOperation("missing identity facts", "runningService", json.RawMessage(`{}`))
	badOperation("missing fences", "fences", json.RawMessage(`{}`))
	badOperation("unknown observation with value", "saved", json.RawMessage(`{"known":false,"value":false}`))
	badOperation("known observation without value", "saved", json.RawMessage(`{"known":true,"value":null}`))
	badOperation("case variant observation key", "saved", json.RawMessage(`{"Known":false,"value":null}`))
	badOperation("extra observation field", "saved", json.RawMessage(`{"known":false,"value":null,"extra":true}`))
	badOperation("extra operation field", "extra", json.RawMessage(`true`))
}

func TestBuildApplyOperationRejectsContradictoryFailure(t *testing.T) {
	test := func(name string, outcome *Outcome, failure *BuildApplyFailure) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			operation := wireOperationFixture()
			operation.Outcome = outcome
			operation.Failure = failure
			if _, err := json.Marshal(operation); err == nil {
				t.Fatal("encoded contradictory operation")
			}
			// Encode through an alias to independently exercise input validation.
			type wire BuildApplyOperation
			raw, err := json.Marshal(wire(operation))
			if err != nil {
				t.Fatalf("encode malformed fixture: %v", err)
			}
			var decoded BuildApplyOperation
			if err := json.Unmarshal(raw, &decoded); err == nil {
				t.Fatal("decoded contradictory operation")
			}
		})
	}
	failure := BuildApplyFailure{Phase: PhasePrepare, ReadbackConfidence: ConfidenceUnknown}
	succeeded, failed := OutcomeSucceeded, OutcomeFailed
	test("pending carries failure", nil, &failure)
	test("success carries failure", &succeeded, &failure)
	failure.Phase = PhaseSwap
	test("failure phase differs", &failed, &failure)
	failure.Phase = PhasePrepare
	failure.Detail = strings.Repeat("界", MaxDetailBytes/3+1)
	test("failure detail exceeds UTF-8 bound", &failed, &failure)
}

func TestBuildApplyStatusRejectsContradictoryOutput(t *testing.T) {
	pending := wireOperationFixture()
	terminal := wireOperationFixture()
	outcome := OutcomeUncertain
	terminal.Outcome = &outcome
	test := func(name string, status BuildApplyStatus) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			if _, err := EncodeStatusResult(Result[BuildApplyStatus]{Available: true, Value: &status}); err == nil {
				t.Fatal("encoded contradictory status")
			}
		})
	}
	test("idle carries operation", BuildApplyStatus{State: StatusIdle, Operation: &pending})
	test("pending carries terminal outcome", BuildApplyStatus{State: StatusPending, Operation: &terminal})
	test("terminal lacks outcome", BuildApplyStatus{State: StatusTerminal, Operation: &pending})
	test("unknown lacks correlation", BuildApplyStatus{State: StatusUnknown})
	test("invalid state", BuildApplyStatus{State: "unreadable", Operation: &pending})
	test("idle refuses startup", BuildApplyStatus{State: StatusIdle})
	test("pending permits startup", BuildApplyStatus{State: StatusPending, Operation: &pending, StartupAllowed: true})
	test("unknown permits startup", BuildApplyStatus{State: StatusUnknown, Operation: &pending, StartupAllowed: true})
}

func TestBuildApplyStatusEncodingBoundsCompleteOutput(t *testing.T) {
	operation := wireOperationFixture()
	// Valid observed strings can expand during JSON escaping. Bound the encoded
	// envelope rather than assuming per-field bounds bound the command output.
	value := Known(strings.Repeat("<", MaxPathBytes))
	operation.Fences = Fences{
		RuntimeGeneration: value, ServiceInstance: value, BrokerGeneration: value,
		ManagerIncarnation: value, PolicyRevision: value, SettingsRevision: value,
		IntentRevision: value, LocksRevision: value, AvailabilityRevision: value,
	}
	status := BuildApplyStatus{State: StatusPending, Operation: &operation}
	result := Result[BuildApplyStatus]{Available: true, Value: &status}
	if _, err := EncodeStatusResult(result); err == nil {
		t.Fatal("encoded oversized status output")
	}
}

func TestBuildApplyOperationKeepsOpaqueIDsAndOwnershipIndependent(t *testing.T) {
	operation := wireOperationFixture()
	operation.OperationID = strings.Repeat("O", 128)
	operation.PreviewID = strings.Repeat("p_-", 42) + "12"
	operation.OwnershipAcknowledged = false
	raw, err := json.Marshal(operation)
	if err != nil {
		t.Fatalf("encode bounded IDs: %v", err)
	}
	var got BuildApplyOperation
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode bounded IDs: %v", err)
	}
	if !reflect.DeepEqual(got, operation) || got.OwnershipAcknowledged {
		t.Fatalf("operation = %+v", got)
	}
}

func TestBuildApplyStatusRequiresExactPreviewMatch(t *testing.T) {
	operation := wireOperationFixture()
	status := BuildApplyStatus{State: StatusPending, Operation: &operation}
	raw, err := EncodeStatusResult(Result[BuildApplyStatus]{Available: true, Value: &status})
	if err != nil {
		t.Fatalf("encode status fixture: %v", err)
	}
	if _, err := DecodeStatusResult(raw, operation.PreviewID); err != nil {
		t.Fatalf("matching preview: %v", err)
	}
	if _, err := DecodeStatusResult(raw, "preview_other"); err == nil {
		t.Fatal("accepted unrelated preview")
	}
	if _, err := DecodeStatusResult(raw, "../preview"); err == nil {
		t.Fatal("accepted invalid expected preview")
	}
	idle := BuildApplyStatus{State: StatusIdle, StartupAllowed: true}
	raw, err = EncodeStatusResult(Result[BuildApplyStatus]{Available: true, Value: &idle})
	if err != nil {
		t.Fatalf("encode idle fixture: %v", err)
	}
	if _, err := DecodeStatusResult(raw, operation.PreviewID); err == nil {
		t.Fatal("idle satisfied operation correlation")
	}
}

func TestBuildApplyUnavailableStatusDoesNotInventIdle(t *testing.T) {
	want := Unavailable[BuildApplyStatus]()
	raw, err := EncodeStatusResult(want)
	if err != nil {
		t.Fatalf("encode unavailable status: %v", err)
	}
	got, err := DecodeStatusResult(raw, "preview_1")
	if err != nil {
		t.Fatalf("decode unavailable status: %v", err)
	}
	if !reflect.DeepEqual(got, want) || got.Available || got.Value != nil {
		t.Fatalf("unavailable status = %+v", got)
	}
}

func TestBuildApplyStartReplyPreservesAdmittedPreview(t *testing.T) {
	operation := wireOperationFixture()
	want := Result[BuildApplyOperation]{Available: true, Value: &operation}
	raw, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("encode start reply: %v", err)
	}
	var got Result[BuildApplyOperation]
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode start reply: %v", err)
	}
	if !reflect.DeepEqual(got, want) || got.Value.PreviewID != operation.PreviewID {
		t.Fatalf("start reply = %+v", got)
	}
}

func TestBundleLayoutsUseActualPackagingNames(t *testing.T) {
	cases := []struct {
		platform string
		want     Layout
	}{
		{"windows", Layout{Executable: "PAIR.exe", ServiceDir: "resources/cli-bin", ServiceExecutable: "nvpair-service.exe"}},
		{"darwin", Layout{RootName: "NVIDIA PAIR.app", Executable: "Contents/MacOS/NVIDIA PAIR", ServiceDir: "Contents/Resources/cli-bin", ServiceExecutable: "nvpair-service"}},
		{"linux", Layout{Executable: "nvpair", ServiceDir: "resources/cli-bin", ServiceExecutable: "nvpair-service"}},
	}
	for _, tc := range cases {
		t.Run(tc.platform, func(t *testing.T) {
			layout, ok := LayoutFor(tc.platform)
			if !ok || layout != tc.want {
				t.Fatalf("layout = %+v, known %v, want %+v", layout, ok, tc.want)
			}
			root := filepath.Join(packageFixtureTempDir(t), layout.RootName)
			// Tiny inert bytes, never executable candidate code or a build.
			for _, name := range []string{layout.Executable, layout.ServiceDir + "/" + layout.ServiceExecutable} {
				path := filepath.Join(root, filepath.FromSlash(name))
				if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte("inert"), 0600); err != nil {
					t.Fatal(err)
				}
				bytes, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				if string(bytes) != "inert" {
					t.Fatalf("fixture content = %q", bytes)
				}
			}
		})
	}
	if _, ok := LayoutFor("other"); ok {
		t.Fatal("accepted unsupported layout")
	}
}

// Check each meaningful JSON field and its TS type, including nested observations,
// instead of treating method-name coverage as payload parity.
func TestTypeScriptBuildApplyFieldParity(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "desktop", "src", "shared", "types", "build-apply.ts"))
	if err != nil {
		t.Fatal(err)
	}
	types := map[reflect.Type]string{
		reflect.TypeOf(nodepolicy.Availability("")): "NodeAvailability",
		reflect.TypeOf(nodepolicy.OnActive("")):     "'finish' | 'cancel'",
		reflect.TypeOf(Identity{}):                  "BuildIdentity", reflect.TypeOf(GUIFacts{}): "BuildApplyGuiFacts",
		reflect.TypeOf(PreviewRequest{}): "BuildApplyPreviewRequest", reflect.TypeOf(Consent{}): "BuildApplyConsent",
		reflect.TypeOf(StartRequest{}): "BuildApplyStartRequest", reflect.TypeOf(StatusRequest{}): "BuildApplyStatusRequest",
		reflect.TypeOf(GUIReadyRequest{}): "BuildApplyGuiReadyRequest", reflect.TypeOf(Fences{}): "BuildApplyFences",
		reflect.TypeOf(EngineFact{}): "BuildApplyEngineFact", reflect.TypeOf(ModelOperation{}): "BuildApplyModelOperation",
		reflect.TypeOf(SettingsDifference{}): "BuildApplySettingsDifference", reflect.TypeOf(ExpectedSettings{}): "BuildApplyExpectedSettings",
		reflect.TypeOf(InterruptionPolicy{}): "BuildApplyInterruptionPolicy", reflect.TypeOf(RestorePolicy{}): "BuildApplyRestorePolicy",
		reflect.TypeOf(Refusal{}): "BuildApplyRefusal", reflect.TypeOf(MaintenanceState{}): "BuildApplyMaintenanceState",
		reflect.TypeOf(Impact{}): "BuildApplyImpact", reflect.TypeOf(Layout{}): "BuildApplyLayout",
		reflect.TypeOf(BuildApplyPreview{}): "BuildApplyPreview", reflect.TypeOf(BuildApplyFailure{}): "BuildApplyFailure",
		reflect.TypeOf(BuildApplyOperation{}): "BuildApplyOperation", reflect.TypeOf(PrepareRequest{}): "BuildApplyPrepareRequest",
		reflect.TypeOf(FinishRequest{}): "BuildApplyFinishRequest",
		reflect.TypeOf(Phase("")):       "BuildApplyPhase", reflect.TypeOf(Outcome("")): "BuildApplyOutcome",
		reflect.TypeOf(Confidence("")):  "BuildApplyConfidence",
		reflect.TypeOf(StatusState("")): "BuildApplyStatusState",
	}
	var tsType func(reflect.Type) string
	tsType = func(typ reflect.Type) string {
		if name := types[typ]; name != "" {
			return name
		}
		if strings.HasPrefix(typ.Name(), "Observation[") {
			return "Observation<" + tsType(typ.Field(1).Type.Elem()) + ">"
		}
		switch typ.Kind() {
		case reflect.Pointer:
			return tsType(typ.Elem()) + " | null"
		case reflect.Slice:
			return tsType(typ.Elem()) + "[]"
		case reflect.Bool:
			return "boolean"
		case reflect.Int:
			return "number"
		case reflect.String:
			return "string"
		}
		t.Fatalf("unmapped wire type %v", typ)
		return ""
	}
	for typ, name := range types {
		if typ.Kind() != reflect.Struct {
			continue
		}
		t.Run(name, func(t *testing.T) {
			match := regexp.MustCompile("(?s)export interface " + name + "\\s*\\{(.*?)\\}").FindStringSubmatch(string(data))
			if len(match) != 2 {
				t.Fatalf("missing TS interface %s", name)
			}
			fields := map[string]string{}
			for _, line := range strings.Split(strings.TrimSpace(match[1]), "\n") {
				pair := strings.SplitN(strings.TrimSpace(line), ":", 2)
				if len(pair) == 2 {
					fields[pair[0]] = strings.TrimSpace(pair[1])
				}
			}
			want := map[string]string{}
			for i := 0; i < typ.NumField(); i++ {
				field := typ.Field(i)
				want[field.Tag.Get("json")] = tsType(field.Type)
			}
			if !reflect.DeepEqual(fields, want) {
				t.Fatalf("TS fields = %#v, Go fields = %#v", fields, want)
			}
		})
	}
	// The finite phase/outcome/confidence sets must retain parity as well.
	enums := []struct {
		name   string
		values []string
	}{
		{"BuildApplyPhase", []string{string(PhasePreflight), string(PhasePrepare), string(PhaseQuiesce), string(PhaseShutdown), string(PhaseSwap), string(PhaseRestart), string(PhaseVerifyBuild), string(PhaseVerifySettings), string(PhaseFinish)}},
		{"BuildApplyOutcome", []string{string(OutcomeSucceeded), string(OutcomeRefused), string(OutcomeFailed), string(OutcomeInterrupted), string(OutcomeUncertain)}},
		{"BuildApplyConfidence", []string{string(ConfidenceUnknown), string(ConfidenceIncomplete), string(ConfidenceVerified)}},
		{"BuildApplyStatusState", []string{string(StatusIdle), string(StatusPending), string(StatusTerminal), string(StatusUnknown)}},
	}
	for _, enum := range enums {
		match := regexp.MustCompile("(?s)export type " + enum.name + "\\s*=\\s*(.*?)(?:export|$)").FindStringSubmatch(string(data))
		if len(match) != 2 {
			t.Fatalf("missing TS enum %s", enum.name)
		}
		var got []string
		for _, literal := range regexp.MustCompile("'([^']+)'").FindAllStringSubmatch(match[1], -1) {
			got = append(got, literal[1])
		}
		sort.Strings(got)
		sort.Strings(enum.values)
		if !reflect.DeepEqual(got, enum.values) {
			t.Fatalf("%s = %v, want %v", enum.name, got, enum.values)
		}
	}
	status := reflect.TypeOf(BuildApplyStatus{})
	stateField, stateOK := status.FieldByName("State")
	operationField, operationOK := status.FieldByName("Operation")
	startupField, startupOK := status.FieldByName("StartupAllowed")
	if !stateOK || !operationOK || !startupOK || status.NumField() != 3 || startupField.Type.Kind() != reflect.Bool || stateField.Type != reflect.TypeOf(StatusState("")) || operationField.Type != reflect.TypeOf((*BuildApplyOperation)(nil)) {
		t.Fatal("unexpected Go status structure")
	}
	match := regexp.MustCompile(`(?s)export type BuildApplyStatus =\s*(.*?)(?:export|$)`).FindStringSubmatch(string(data))
	if len(match) != 2 {
		t.Fatal("missing TS status union")
	}
	branches := map[string]string{}
	pattern := `\{\s*` + stateField.Tag.Get("json") + `: '([^']+)'; ` + operationField.Tag.Get("json") + `: ([^;]+); ` + startupField.Tag.Get("json") + `: (?:true|false|boolean)\s*\}`
	for _, branch := range regexp.MustCompile(pattern).FindAllStringSubmatch(match[1], -1) {
		branches[branch[1]] = strings.TrimSpace(branch[2])
	}
	wantBranches := map[string]string{
		string(StatusIdle): "null", string(StatusPending): "BuildApplyOperation",
		string(StatusTerminal): "BuildApplyOperation", string(StatusUnknown): "BuildApplyOperation",
	}
	if !reflect.DeepEqual(branches, wantBranches) {
		t.Fatalf("TS status branches = %v, want %v", branches, wantBranches)
	}
	limit := regexp.MustCompile(`BUILD_APPLY_MAX_STATUS_BYTES = (\d+) << (\d+)`).FindStringSubmatch(string(data))
	if len(limit) != 3 {
		t.Fatal("missing TS status size bound")
	}
	base, err := strconv.Atoi(limit[1])
	if err != nil {
		t.Fatalf("decode TS bound base: %v", err)
	}
	shift, err := strconv.Atoi(limit[2])
	if err != nil {
		t.Fatalf("decode TS bound shift: %v", err)
	}
	if base<<shift != MaxStatusBytes {
		t.Fatalf("TS status bound = %d, Go bound = %d", base<<shift, MaxStatusBytes)
	}
}
