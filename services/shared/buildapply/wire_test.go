// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
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
	gui := GUIFacts{PID: 42, Executable: filepath.Join(t.TempDir(), "PAIR.exe"), BuildID: ""}
	encoded, err := json.Marshal(PreviewRequest{CandidatePath: t.TempDir(), GUI: gui})
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
			root := filepath.Join(t.TempDir(), layout.RootName)
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
		reflect.TypeOf(Confidence("")): "BuildApplyConfidence",
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
}
