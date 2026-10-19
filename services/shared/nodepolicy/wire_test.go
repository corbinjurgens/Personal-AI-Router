// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func residentIdentity() StrictModelIdentity {
	return StrictModelIdentity{EngineRunID: "engine-1", SettingsFingerprint: strings.Repeat("a", 64), ModelDigest: strings.Repeat("b", 64)}
}

func residentObservation() ResidentLoadedObservation {
	return ResidentLoadedObservation{ManagerIncarnation: "manager-1", Generation: 9, Engine: "ollama", Model: "m:latest", Identity: residentIdentity(), Loaded: true}
}

type residentWireRejection struct {
	name  string
	field string
	value any
}

// Each fixture checks its own round trip and strict object decoding, then
// exercises the DTO's semantic validation through its wire decoder.
func testResidentWire[T interface{ Validate() error }](t *testing.T, valid T, required string, invalid ...residentWireRejection) {
	t.Helper()
	if err := valid.Validate(); err != nil {
		t.Fatalf("fixture Validate: %v", err)
	}
	data, err := json.Marshal(valid)
	if err != nil {
		t.Fatal(err)
	}
	t.Run("round trip", func(t *testing.T) {
		var again T
		if err := json.Unmarshal(data, &again); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(again, valid) {
			t.Fatalf("round trip = %+v, want %+v", again, valid)
		}
	})
	reject := func(t *testing.T, raw []byte) {
		t.Helper()
		var target T
		if err := json.Unmarshal(raw, &target); err == nil {
			t.Fatalf("accepted invalid contract: %s", raw)
		}
	}
	for _, key := range strings.Fields(required) {
		t.Run("missing "+key, func(t *testing.T) {
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(data, &fields); err != nil {
				t.Fatal(err)
			}
			delete(fields, key)
			raw, err := json.Marshal(fields)
			if err != nil {
				t.Fatal(err)
			}
			reject(t, raw)
		})
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	for key := range fields {
		t.Run("null "+key, func(t *testing.T) {
			var next map[string]json.RawMessage
			if err := json.Unmarshal(data, &next); err != nil {
				t.Fatal(err)
			}
			next[key] = json.RawMessage(`null`)
			raw, err := json.Marshal(next)
			if err != nil {
				t.Fatal(err)
			}
			reject(t, raw)
		})
	}
	for _, tc := range []struct{ name, raw string }{
		{"null object", `null`}, {"array", `[]`}, {"malformed", `{`}, {"trailing object", string(data) + `{}`},
		{"unknown key", `{"extra":true,` + string(data[1:])},
		{"duplicate key", `{"` + strings.Fields(required)[0] + `":0,` + string(data[1:])},
		{"case variant key", `{"` + strings.ToUpper(strings.Fields(required)[0]) + `":0,` + string(data[1:])},
	} {
		t.Run(tc.name, func(t *testing.T) { reject(t, []byte(tc.raw)) })
	}
	for _, tc := range invalid {
		t.Run(tc.name, func(t *testing.T) {
			var next map[string]any
			if err := json.Unmarshal(data, &next); err != nil {
				t.Fatal(err)
			}
			next[tc.field] = tc.value
			raw, err := json.Marshal(next)
			if err != nil {
				t.Fatal(err)
			}
			reject(t, raw)
		})
	}
}

func TestPrepareResidentWire(t *testing.T) {
	valid := PrepareResidentParams{PolicyRevision: "1", ProxyRunID: "proxy-1", ManagerIncarnation: "manager-1", Engine: "ollama", Model: "m:latest", Identity: residentIdentity(), OperationID: "op-1", TimeoutMs: 60_000}
	testResidentWire(t, valid, "policyRevision proxyRunId engine model identity operationId timeoutMs",
		residentWireRejection{"noncanonical revision", "policyRevision", "01"},
		residentWireRejection{"revision overflow", "policyRevision", strings.Repeat("9", 30)},
		residentWireRejection{"empty proxy run", "proxyRunId", ""},
		residentWireRejection{"long proxy run", "proxyRunId", strings.Repeat("p", 129)},
		residentWireRejection{"invalid operation", "operationId", "op/1"},
		residentWireRejection{"long operation", "operationId", strings.Repeat("o", 129)},
		residentWireRejection{"invalid manager", "managerIncarnation", "manager/1"},
		residentWireRejection{"long manager", "managerIncarnation", strings.Repeat("m", 129)},
		residentWireRejection{"invalid engine", "engine", "Bad Engine"},
		residentWireRejection{"long engine", "engine", strings.Repeat("e", 129)},
		residentWireRejection{"empty model", "model", ""},
		residentWireRejection{"untrimmed model", "model", " m"},
		residentWireRejection{"long model", "model", strings.Repeat("m", 1025)},
		residentWireRejection{"reserved alias", "model", TierOrder[0]},
		residentWireRejection{"zero timeout", "timeoutMs", 0},
		residentWireRejection{"negative timeout", "timeoutMs", -1},
		residentWireRejection{"timeout over cap", "timeoutMs", 60_001},
		residentWireRejection{"fractional timeout", "timeoutMs", 0.5},
		residentWireRejection{"empty identity", "identity", StrictModelIdentity{}},
		residentWireRejection{"invalid fingerprint", "identity", StrictModelIdentity{EngineRunID: "e", SettingsFingerprint: "bad", ModelDigest: strings.Repeat("b", 64)}},
		residentWireRejection{"invalid digest", "identity", StrictModelIdentity{EngineRunID: "e", SettingsFingerprint: strings.Repeat("a", 64), ModelDigest: "bad"}},
	)
}

func TestResidentRefusalWire(t *testing.T) {
	testResidentWire(t, ResidentRefusal{Engine: "ollama", Model: "m", Reason: ModelReasonNoFit}, "engine model reason",
		residentWireRejection{"invalid reference", "model", ""}, residentWireRejection{"unknown reason", "reason", "raw engine error"})
}

func TestPrepareResidentResultWire(t *testing.T) {
	refusal := ResidentRefusal{Engine: "ollama", Model: "m", Reason: ModelReasonNoFit}
	t.Run("grant", func(t *testing.T) {
		testResidentWire(t, PrepareResidentResult{ReservationID: "r-1", EvidenceGeneration: 1}, "evidenceGeneration",
			residentWireRejection{"missing grant", "reservationId", ""},
			residentWireRejection{"invalid grant", "reservationId", "r/1"},
			residentWireRejection{"long grant", "reservationId", strings.Repeat("r", 129)},
			residentWireRejection{"refusal cannot reserve", "refusals", []ResidentRefusal{refusal}},
			residentWireRejection{"negative generation", "evidenceGeneration", -1},
			residentWireRejection{"overflow generation", "evidenceGeneration", json.Number("18446744073709551616")})
	})
	t.Run("refusal", func(t *testing.T) {
		testResidentWire(t, PrepareResidentResult{EvidenceGeneration: 1, Refusals: []ResidentRefusal{refusal}}, "evidenceGeneration",
			residentWireRejection{"duplicate refusal", "refusals", []ResidentRefusal{refusal, refusal}},
			residentWireRejection{"too many refusals", "refusals", make([]ResidentRefusal, 257)},
			residentWireRejection{"invalid refusal", "refusals", []ResidentRefusal{{}}})
	})
}

func TestResidentReservationEvidenceWire(t *testing.T) {
	testResidentWire(t, ResidentReservationEvidence{ReservationID: "r-1", ProxyRunID: "p-1", ManagerIncarnation: "m-1"}, "reservationId proxyRunId",
		residentWireRejection{"empty reservation", "reservationId", ""},
		residentWireRejection{"invalid run", "proxyRunId", "p/1"},
		residentWireRejection{"long reservation", "reservationId", strings.Repeat("r", 129)},
		residentWireRejection{"long run", "proxyRunId", strings.Repeat("p", 129)},
		residentWireRejection{"invalid manager", "managerIncarnation", "m/1"},
		residentWireRejection{"long manager", "managerIncarnation", strings.Repeat("m", 129)})
}

func TestReconcileResidentWire(t *testing.T) {
	valid := ReconcileResidentParams{OperationID: "op-1", PolicyRevision: "1", Engine: "ollama", Model: "m:latest", Identity: residentIdentity(), TimeoutMs: 60_000, Reservation: ResidentReservationEvidence{ReservationID: "r-1", ProxyRunID: "p-1"}}
	testResidentWire(t, valid, "operationId policyRevision engine model identity timeoutMs reservation",
		residentWireRejection{"invalid reservation", "reservation", ResidentReservationEvidence{}},
		residentWireRejection{"stale syntax revision", "policyRevision", "01"},
		residentWireRejection{"timeout over cap", "timeoutMs", 60_001},
		residentWireRejection{"zero timeout", "timeoutMs", 0},
		residentWireRejection{"invalid identity", "identity", StrictModelIdentity{}},
		residentWireRejection{"invalid reference", "model", ""},
		residentWireRejection{"invalid operation", "operationId", ""})
}

func TestResidentLoadedObservationWire(t *testing.T) {
	testResidentWire(t, residentObservation(), "managerIncarnation generation engine model identity loaded",
		residentWireRejection{"invalid source", "managerIncarnation", ""},
		residentWireRejection{"long source", "managerIncarnation", strings.Repeat("m", 129)},
		residentWireRejection{"invalid reference", "model", ""},
		residentWireRejection{"invalid identity", "identity", StrictModelIdentity{}},
		residentWireRejection{"negative generation", "generation", -1})
}

func TestReconcileResidentResultWire(t *testing.T) {
	observation := residentObservation()
	testResidentWire(t, ReconcileResidentResult{Outcome: ResidentLoaded, Observation: &observation}, "outcome",
		residentWireRejection{"unknown outcome", "outcome", "success-ish"},
		residentWireRejection{"not dispatched with observation", "outcome", ResidentNotDispatched},
		residentWireRejection{"invalid observation", "observation", ResidentLoadedObservation{}})
	for _, tc := range []struct {
		name        string
		observation *ResidentLoadedObservation
	}{
		{"missing loaded observation", nil},
		{"unloaded observation", &ResidentLoadedObservation{ManagerIncarnation: "m-1", Engine: "ollama", Model: "m", Identity: residentIdentity()}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := (ReconcileResidentResult{Outcome: ResidentLoaded, Observation: tc.observation}).Validate(); err == nil {
				t.Fatal("accepted loaded without proof")
			}
		})
	}
}

func TestFinishResidentWire(t *testing.T) {
	observation := residentObservation()
	testResidentWire(t, FinishResidentParams{ReservationID: "r-1", ProxyRunID: "p-1", OperationID: "op-1", Outcome: ResidentLoaded, Observation: &observation}, "reservationId proxyRunId operationId outcome",
		residentWireRejection{"invalid reservation", "reservationId", ""},
		residentWireRejection{"long reservation", "reservationId", strings.Repeat("r", 129)},
		residentWireRejection{"invalid proxy", "proxyRunId", "p/1"},
		residentWireRejection{"long proxy", "proxyRunId", strings.Repeat("p", 129)},
		residentWireRejection{"invalid operation", "operationId", ""},
		residentWireRejection{"long operation", "operationId", strings.Repeat("o", 129)},
		residentWireRejection{"unknown outcome", "outcome", "other"},
		residentWireRejection{"invalid observation", "observation", ResidentLoadedObservation{}})
}

func TestFinishResidentResultWire(t *testing.T) {
	testResidentWire(t, FinishResidentResult{ReservationID: "r-1", Outcome: ResidentNotDispatched}, "reservationId outcome",
		residentWireRejection{"invalid reservation", "reservationId", ""},
		residentWireRejection{"long reservation", "reservationId", strings.Repeat("r", 129)},
		residentWireRejection{"unknown outcome", "outcome", "other"})
}

func TestModelConfigurationEntryWire(t *testing.T) {
	loaded := true
	valid := ModelConfigurationEntry{Engine: "ollama", Model: "m", DesiredResident: true, ObservedLoaded: &loaded, State: ModelStateSatisfied}
	testResidentWire(t, valid, "engine model desiredResident state",
		residentWireRejection{"invalid reference", "model", ""},
		residentWireRejection{"unknown state", "state", "failed"},
		residentWireRejection{"unknown reason", "reason", "other"},
		residentWireRejection{"satisfied with false observation", "observedLoaded", false},
		residentWireRejection{"satisfied without desire", "desiredResident", false},
		residentWireRejection{"satisfied with reason", "reason", ModelReasonNoFit})
	valid.ObservedLoaded = nil
	if err := valid.Validate(); err == nil {
		t.Fatal("satisfied accepted without observedLoaded")
	}
}

func TestModelConfigurationWire(t *testing.T) {
	entry := ModelConfigurationEntry{Engine: "ollama", Model: "m", DesiredResident: true, State: ModelStateBlocked, Reason: ModelReasonUnqualified}
	testResidentWire(t, ModelConfiguration{PolicyRevision: "1", ObservationGeneration: 2, Models: []ModelConfigurationEntry{entry}}, "policyRevision observationGeneration models",
		residentWireRejection{"invalid revision", "policyRevision", "01"},
		residentWireRejection{"negative generation", "observationGeneration", -1},
		residentWireRejection{"too many entries", "models", make([]ModelConfigurationEntry, 257)},
		residentWireRejection{"invalid entry", "models", []ModelConfigurationEntry{{}}},
		residentWireRejection{"duplicate reference", "models", []ModelConfigurationEntry{entry, entry}})
	if err := (ModelConfiguration{PolicyRevision: "1"}).Validate(); err == nil {
		t.Fatal("accepted nil models")
	}
}

func TestResidentWireBoundsAccepted(t *testing.T) {
	longID := strings.Repeat("i", 128)
	for _, timeout := range []int{1, MaxResidentTimeoutMs} {
		t.Run(fmt.Sprintf("timeout %d", timeout), func(t *testing.T) {
			p := PrepareResidentParams{PolicyRevision: "9223372036854775807", ProxyRunID: longID, ManagerIncarnation: longID, Engine: "ollama", Model: strings.Repeat("m", MaxResidentModelBytes), Identity: residentIdentity(), OperationID: longID, TimeoutMs: timeout}
			testResidentWire(t, p, "policyRevision proxyRunId engine model identity operationId timeoutMs")
		})
	}
	models := make([]ModelConfigurationEntry, MaxModelConfigurationEntries)
	refusals := make([]ResidentRefusal, MaxModelConfigurationEntries)
	for i := range models {
		model := fmt.Sprintf("m%d", i)
		models[i] = ModelConfigurationEntry{Engine: "ollama", Model: model, State: ModelStateUnknown}
		refusals[i] = ResidentRefusal{Engine: "ollama", Model: model, Reason: ModelReasonNoFit}
	}
	t.Run("max status list", func(t *testing.T) {
		testResidentWire(t, ModelConfiguration{PolicyRevision: "0", ObservationGeneration: ^uint64(0), Models: models}, "policyRevision observationGeneration models")
	})
	t.Run("empty status list", func(t *testing.T) {
		testResidentWire(t, ModelConfiguration{PolicyRevision: "0", Models: []ModelConfigurationEntry{}}, "policyRevision observationGeneration models")
	})
	t.Run("max refusal list", func(t *testing.T) {
		testResidentWire(t, PrepareResidentResult{EvidenceGeneration: ^uint64(0), Refusals: refusals}, "evidenceGeneration")
	})
	t.Run("max reservation identity", func(t *testing.T) {
		testResidentWire(t, ResidentReservationEvidence{ReservationID: longID, ProxyRunID: longID, ManagerIncarnation: longID}, "reservationId proxyRunId")
	})
}

func TestResidentWireEnums(t *testing.T) {
	for _, reason := range []ModelConfigurationReason{ModelReasonEngineOff, ModelReasonNoFit, ModelReasonPausedLocal, ModelReasonEngineOffLocal, ModelReasonNotServed, ModelReasonNotInstalled, ModelReasonUnqualified, ModelReasonUnsupportedLoad, ModelReasonSwitchBusy, ModelReasonOperationTimeout, ModelReasonObservationUnknown, ModelReasonProjectionUnknown} {
		t.Run(string(reason), func(t *testing.T) {
			testResidentWire(t, ResidentRefusal{Engine: "ollama", Model: "m", Reason: reason}, "engine model reason")
		})
	}
	for _, state := range []ModelConfigurationState{ModelStateSatisfied, ModelStatePending, ModelStateBlocked, ModelStateUnknown} {
		t.Run(string(state), func(t *testing.T) {
			loaded := true
			testResidentWire(t, ModelConfigurationEntry{Engine: "ollama", Model: "m", DesiredResident: true, ObservedLoaded: &loaded, State: state}, "engine model desiredResident state")
		})
	}
	for _, outcome := range []ResidentOutcome{ResidentLoaded, ResidentNotDispatched, ResidentFailed, ResidentCancelled, ResidentTimeout, ResidentUnknown} {
		t.Run(string(outcome), func(t *testing.T) {
			r := ReconcileResidentResult{Outcome: outcome}
			if outcome == ResidentLoaded {
				observation := residentObservation()
				r.Observation = &observation
			}
			testResidentWire(t, r, "outcome")
		})
	}
	if ModelReasonEngineOff != ModelConfigurationReason(RejectOff) || ModelReasonNoFit != ModelConfigurationReason(RejectNoFit) {
		t.Fatal("admission reason spellings diverged")
	}
}

func TestResidentWirePresence(t *testing.T) {
	for _, tc := range []struct {
		name, raw string
		want      *bool
	}{
		{"unknown observation", `{"engine":"ollama","model":"m","desiredResident":false,"state":"unknown"}`, nil},
		{"explicit unloaded", `{"engine":"ollama","model":"m","desiredResident":false,"observedLoaded":false,"state":"pending"}`, new(bool)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var entry ModelConfigurationEntry
			if err := json.Unmarshal([]byte(tc.raw), &entry); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(entry.ObservedLoaded, tc.want) || entry.DesiredResident {
				t.Fatalf("decoded entry = %+v", entry)
			}
			data, err := json.Marshal(entry)
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(data, &fields); err != nil {
				t.Fatal(err)
			}
			_, present := fields["observedLoaded"]
			if present != (tc.want != nil) {
				t.Fatalf("observedLoaded present = %v", present)
			}
			if string(fields["desiredResident"]) != "false" {
				t.Fatalf("desiredResident = %s", fields["desiredResident"])
			}
		})
	}
	observation := residentObservation()
	observation.Loaded = false
	t.Run("explicit unloaded source observation", func(t *testing.T) {
		testResidentWire(t, observation, "managerIncarnation generation engine model identity loaded")
	})
	t.Run("optional case variant rejected", func(t *testing.T) {
		var entry ModelConfigurationEntry
		if err := json.Unmarshal([]byte(`{"engine":"ollama","model":"m","desiredResident":false,"ObservedLoaded":false,"state":"pending"}`), &entry); err == nil {
			t.Fatal("accepted case variant optional key")
		}
	})
}

func TestResidentNestedValidation(t *testing.T) {
	// Direct construction must reject invalid children as well as JSON decoding.
	for _, tc := range []struct {
		name  string
		value interface{ Validate() error }
	}{
		{"invalid refusal", PrepareResidentResult{Refusals: []ResidentRefusal{{}}}},
		{"too many refusals", PrepareResidentResult{Refusals: make([]ResidentRefusal, MaxModelConfigurationEntries+1)}},
		{"invalid reservation", ReconcileResidentParams{}},
		{"invalid loaded observation", ReconcileResidentResult{Outcome: ResidentUnknown, Observation: &ResidentLoadedObservation{}}},
		{"invalid status entry", ModelConfiguration{PolicyRevision: "1", Models: []ModelConfigurationEntry{{}}}},
		{"empty reason enum", ModelConfigurationReason("")},
		{"empty state enum", ModelConfigurationState("")},
		{"empty outcome enum", ResidentOutcome("")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.value.Validate(); err == nil {
				t.Fatal("Validate accepted invalid value")
			}
		})
	}
}

func TestPrepareResidentWithoutManagerIncarnation(t *testing.T) {
	p := PrepareResidentParams{PolicyRevision: "1", ProxyRunID: "p-1", Engine: "ollama", Model: "m", Identity: residentIdentity(), OperationID: "op-1", TimeoutMs: 1}
	testResidentWire(t, p, "policyRevision proxyRunId engine model identity operationId timeoutMs")
}

func TestResidentMethodNames(t *testing.T) {
	if MethodPrepareResident != "node/prepare-resident" || MethodReconcileResident != "engine:reconcile-resident" || MethodFinishResident != "node/finish-resident" {
		t.Fatal("resident method names changed")
	}
	// Existing wire contracts remain exactly as before this additive change.
	ack, err := json.Marshal(PolicyAck{Revision: "1", RunID: "p-1"})
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(ack, &fields); err != nil {
		t.Fatal(err)
	}
	if len(fields) != 2 || fields["revision"] == nil || fields["runId"] == nil {
		t.Fatalf("PolicyAck fields = %v", fields)
	}
	if MethodSetResidency != "node/set-residency" {
		t.Fatal("set residency method changed")
	}
}

func TestStrictReadinessFences(t *testing.T) {
	for _, tc := range []struct {
		name      string
		value     StrictAdmissionReadiness
		wantError bool
	}{
		{"withdrawal", StrictAdmissionReadiness{PolicyRevision: "0", ProxyRunID: "p-1"}, false},
		{"missing revision", StrictAdmissionReadiness{ProxyRunID: "p-1"}, true},
		{"invalid revision", StrictAdmissionReadiness{PolicyRevision: "01", ProxyRunID: "p-1"}, true},
		{"missing run", StrictAdmissionReadiness{PolicyRevision: "1"}, true},
		{"invalid run", StrictAdmissionReadiness{PolicyRevision: "1", ProxyRunID: "p/1"}, true},
		{"invalid capability", StrictAdmissionReadiness{PolicyRevision: "1", ProxyRunID: "p-1", Capability: &StrictAdmissionCapability{Version: 1}}, true},
		{"unsupported capability", StrictAdmissionReadiness{PolicyRevision: "1", ProxyRunID: "p-1", Capability: &StrictAdmissionCapability{Version: 2}}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.value.Validate()
			if (err != nil) != tc.wantError {
				t.Fatalf("Validate() = %v, want error %v", err, tc.wantError)
			}
		})
	}
}

func TestStrictReadinessWithdrawalOmitsCapability(t *testing.T) {
	r := StrictAdmissionReadiness{PolicyRevision: "12", ProxyRunID: "p-1"}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"policyRevision":"12","proxyRunId":"p-1"}` {
		t.Fatalf("encoded = %s", data)
	}
	var again StrictAdmissionReadiness
	if err := json.Unmarshal(data, &again); err != nil {
		t.Fatal(err)
	}
	if again != r {
		t.Fatalf("round trip = %+v, want %+v", again, r)
	}
}

func TestSetModelIdentityParamsValidate(t *testing.T) {
	digest := strings.Repeat("a", 64)
	valid := func() EngineIdentity {
		return EngineIdentity{EngineRunID: "run-1", SettingsFingerprint: digest, LaunchDigest: digest, ModelDigests: map[string]string{"m:latest": digest}}
	}
	for _, tc := range []struct {
		name    string
		edit    func(*EngineIdentity)
		params  *SetModelIdentityParams
		wantErr string
	}{
		{name: "valid", edit: func(*EngineIdentity) {}},
		{name: "no models", edit: func(i *EngineIdentity) { i.ModelDigests = map[string]string{} }},
		{name: "no engines", params: &SetModelIdentityParams{Engines: map[string]EngineIdentity{}}},
		{name: "missing engines", params: &SetModelIdentityParams{}, wantErr: "engines is required"},
		{name: "missing run", edit: func(i *EngineIdentity) { i.EngineRunID = "" }, wantErr: "engineRunId"},
		{name: "invalid run", edit: func(i *EngineIdentity) { i.EngineRunID = "run/1" }, wantErr: "engineRunId"},
		{name: "uppercase fingerprint", edit: func(i *EngineIdentity) { i.SettingsFingerprint = strings.Repeat("A", 64) }, wantErr: "settingsFingerprint"},
		{name: "short launch digest", edit: func(i *EngineIdentity) { i.LaunchDigest = "abc" }, wantErr: "launchDigest"},
		{name: "missing model map", edit: func(i *EngineIdentity) { i.ModelDigests = nil }, wantErr: "modelDigests is required"},
		{name: "empty model", edit: func(i *EngineIdentity) { i.ModelDigests = map[string]string{" ": digest} }, wantErr: "empty model"},
		{name: "prefixed model digest", edit: func(i *EngineIdentity) { i.ModelDigests = map[string]string{"m": "sha256:" + digest} }, wantErr: "modelDigests"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			params := tc.params
			if params == nil {
				id := valid()
				tc.edit(&id)
				params = &SetModelIdentityParams{Engines: map[string]EngineIdentity{"ollama": id}}
			}
			err := params.Validate()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Validate() = %v, want an error containing %q", err, tc.wantErr)
			}
		})
	}
}
