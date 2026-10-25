// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package enginesettings

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
)

func switchRequestFixture() SwitchRequest {
	return SwitchRequest{NodeID: "node-1", RequestID: "switch-42", SourceEngine: "llamacpp-main",
		TargetEngine: "llamacpp-custom", ExpectedRevision: "e3v1:" + strings.Repeat("a", 64)}
}

func switchReceiptFixture() SwitchReceipt {
	request := switchRequestFixture()
	now := time.Date(2026, 10, 19, 1, 0, 0, 0, time.UTC)
	return SwitchReceipt{Version: SwitchVersion, NodeID: request.NodeID, RequestID: request.RequestID,
		Caller: "local", PayloadHash: SwitchPayloadHash("local", request.NodeID, request), Request: request,
		ExpectedRevision: request.ExpectedRevision, AcceptedRevision: "e3v1:" + strings.Repeat("b", 64),
		Sequence: 1, Phase: SwitchAccepted, AcceptedAt: now, UpdatedAt: now, Deadline: now.Add(SwitchOperationBudget),
		Source: SwitchEngineState{Intent: SwitchIntentOn}, Target: SwitchEngineState{Intent: SwitchIntentOff}}
}

func checkSwitchWire[T any](t *testing.T, value T, keys string) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatalf("decode keys: %v", err)
	}
	got := make([]string, 0, len(fields))
	for key := range fields {
		got = append(got, key)
	}
	want := strings.Fields(keys)
	slices.Sort(got)
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Fatalf("keys = %v, want %v", got, want)
	}
	var decoded T
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("round trip: %v", err)
	}
	if !reflect.DeepEqual(decoded, value) {
		t.Fatalf("round trip = %#v, want %#v", decoded, value)
	}
}

func TestSwitchWireFields(t *testing.T) {
	r := switchReceiptFixture()
	now := r.AcceptedAt
	running, healthy := false, true
	state := SwitchEngineState{Intent: SwitchIntentOff, Running: &running, Healthy: &healthy, ObservedAt: &now, TerminationVerified: true}
	errValue := SwitchError{Code: SwitchOperationErrorCode, Kind: SwitchErrorTargetStart, Message: "Target did not become ready."}
	sequence := uint64(1)
	prepare := SwitchPrepareRequest{OperationID: r.RequestID, Caller: "local", Origin: SwitchOriginLocal,
		SourceEngine: r.Request.SourceEngine, TargetEngine: r.Request.TargetEngine,
		PayloadHash: r.PayloadHash, ConfigurationFence: "fence", Deadline: r.Deadline}
	t.Run("request", func(t *testing.T) {
		checkSwitchWire(t, r.Request, "nodeId requestId sourceEngine targetEngine expectedRevision")
	})
	t.Run("status request", func(t *testing.T) {
		checkSwitchWire(t, SwitchStatusRequest{NodeID: "node-1", RequestID: r.RequestID, AfterSequence: &sequence}, "nodeId requestId afterSequence")
	})
	t.Run("empty status request", func(t *testing.T) { checkSwitchWire(t, SwitchStatusRequest{}, "") })
	t.Run("state", func(t *testing.T) { checkSwitchWire(t, state, "intent running healthy observedAt terminationVerified") })
	t.Run("unknown state", func(t *testing.T) {
		checkSwitchWire(t, SwitchEngineState{Intent: SwitchIntentUnknown}, "intent running healthy observedAt terminationVerified")
	})
	t.Run("error", func(t *testing.T) { checkSwitchWire(t, errValue, "code kind message") })
	t.Run("receipt", func(t *testing.T) {
		checkSwitchWire(t, r, "version nodeId requestId caller payloadHash request expectedRevision acceptedRevision sequence phase terminal acceptedAt updatedAt deadline finishedAt source target error recoveryRequired")
	})
	t.Run("finished receipt", func(t *testing.T) {
		r := switchReceiptFixture()
		r.Terminal, r.Phase, r.FinishedAt, r.Error = true, SwitchFailedTargetStart, &now, &errValue
		checkSwitchWire(t, r, "version nodeId requestId caller payloadHash request expectedRevision acceptedRevision sequence phase terminal acceptedAt updatedAt deadline finishedAt source target error recoveryRequired")
	})
	t.Run("observed", func(t *testing.T) { checkSwitchWire(t, SwitchObserved{Source: state, Target: state}, "source target") })
	t.Run("status", func(t *testing.T) {
		checkSwitchWire(t, SwitchStatus{ConfigurationRevision: r.AcceptedRevision, ActiveRequestID: r.RequestID, Receipt: &r, Observed: &SwitchObserved{Source: state, Target: state}}, "configurationRevision activeRequestId receipt observed")
	})
	t.Run("minimal status", func(t *testing.T) {
		checkSwitchWire(t, SwitchStatus{ConfigurationRevision: r.AcceptedRevision}, "configurationRevision")
	})
	t.Run("result", func(t *testing.T) { checkSwitchWire(t, SwitchResult{Receipt: r}, "receipt") })
	t.Run("journal", func(t *testing.T) {
		checkSwitchWire(t, SwitchJournal{Version: 1, Generation: 1, ActiveRequestID: r.RequestID, Receipts: map[string]SwitchReceipt{r.RequestID: r}}, "version generation activeRequestId receipts")
	})
	t.Run("journal local obligations", func(t *testing.T) {
		checkSwitchWire(t, SwitchJournal{Version: 1, Receipts: map[string]SwitchReceipt{}, LocalObligations: map[string]SwitchLocalObligation{"operation": {SourceOff: true}}}, "version generation activeRequestId receipts localObligations")
		checkSwitchWire(t, SwitchLocalObligation{SourceOff: true}, "sourceOff targetOff pause discharged")
	})
	t.Run("applied pause obligation", func(t *testing.T) {
		checkSwitchWire(t, SwitchLocalObligation{Pause: true, PauseApplied: true}, "sourceOff targetOff pause pauseApplied discharged")
	})
	t.Run("changed", func(t *testing.T) {
		checkSwitchWire(t, SwitchChanged{NodeID: r.NodeID, RequestID: r.RequestID, Sequence: r.Sequence}, "nodeId requestId sequence")
	})
	const prepareKeys = "operationId caller origin sourceEngine targetEngine payloadHash configurationFence managerIncarnation deadline"
	t.Run("prepare", func(t *testing.T) { checkSwitchWire(t, prepare, prepareKeys) })
	t.Run("step", func(t *testing.T) {
		checkSwitchWire(t, SwitchStepRequest{SwitchPrepareRequest: prepare, Step: SwitchStepSourceStop}, prepareKeys+" step")
	})
	t.Run("reconcile", func(t *testing.T) {
		checkSwitchWire(t, SwitchReconcileRequest{SwitchPrepareRequest: prepare}, prepareKeys)
	})
	t.Run("reconcile local veto", func(t *testing.T) {
		veto := SwitchLocalVeto{Engine: "llamacpp"}
		checkSwitchWire(t, SwitchReconcileRequest{SwitchPrepareRequest: prepare, LocalVeto: &veto}, prepareKeys+" localVeto")
		checkSwitchWire(t, veto, "engine pause")
	})
	checkpoint := SwitchCheckpoint{SwitchPrepareRequest: prepare, Phase: SwitchCheckpointReserved, PreSource: state, PreTarget: state, Source: state, Target: state}
	t.Run("checkpoint", func(t *testing.T) {
		checkSwitchWire(t, checkpoint, prepareKeys+" phase preSource preTarget sourceProvenance targetProvenance source target targetCommitted error")
	})
	t.Run("checkpoint operation local Off", func(t *testing.T) {
		cp := checkpoint
		cp.SourceLocalOff, cp.TargetLocalOff = true, true
		checkSwitchWire(t, cp, prepareKeys+" phase preSource preTarget sourceProvenance targetProvenance source target targetCommitted sourceLocalOff targetLocalOff error")
	})
	t.Run("executor result", func(t *testing.T) {
		checkSwitchWire(t, SwitchPrepareResult{Checkpoint: &checkpoint, Observed: SwitchObserved{Source: state, Target: state}, Error: &errValue}, "checkpoint observed error")
	})
	t.Run("configuration entry", func(t *testing.T) {
		checkSwitchWire(t, SwitchConfigurationEngine{Engine: "llamacpp", SettingsRevision: 1, ConfigDigest: "digest", Intent: SwitchIntentOn, Provenance: "local"}, "engine settingsRevision configDigest intent provenance")
	})
	t.Run("configuration snapshot", func(t *testing.T) {
		checkSwitchWire(t, SwitchConfigurationSnapshot{NodeID: "node-1", JournalGeneration: 1, ManagerIncarnation: "manager-1", RegistryRuntimeFingerprint: "registry"}, "nodeId journalGeneration managerIncarnation engines registryRuntimeFingerprint")
	})
}

func TestSwitchPayloadHashBindings(t *testing.T) {
	base := switchRequestFixture()
	want := SwitchPayloadHash("local", "canonical-node", base)
	if !switchPayloadHash.MatchString(want) {
		t.Fatalf("hash format = %q", want)
	}
	cases := []struct {
		name, caller, node string
		change             func(*SwitchRequest)
		same               bool
	}{
		{"repeat", "local", "canonical-node", func(*SwitchRequest) {}, true},
		{"transport alias", "local", "canonical-node", func(r *SwitchRequest) { r.NodeID = "relay-alias" }, true},
		{"caller", "peer", "canonical-node", func(*SwitchRequest) {}, false},
		{"canonical node", "local", "other-node", func(*SwitchRequest) {}, false},
		{"request ID", "local", "canonical-node", func(r *SwitchRequest) { r.RequestID += "2" }, false},
		{"source", "local", "canonical-node", func(r *SwitchRequest) { r.SourceEngine += "2" }, false},
		{"target", "local", "canonical-node", func(r *SwitchRequest) { r.TargetEngine += "2" }, false},
		{"revision", "local", "canonical-node", func(r *SwitchRequest) { r.ExpectedRevision = "e3v1:" + strings.Repeat("c", 64) }, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := base
			tc.change(&r)
			if got := SwitchPayloadHash(tc.caller, tc.node, r); (got == want) != tc.same {
				t.Fatalf("hash equality = %v, want %v", got == want, tc.same)
			}
		})
	}
}

func TestSwitchPayloadHashExcludesReceiptObservations(t *testing.T) {
	r := switchReceiptFixture()
	want := SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
	r.AcceptedAt = r.AcceptedAt.Add(time.Hour)
	r.UpdatedAt = r.UpdatedAt.Add(time.Hour)
	r.Sequence++
	r.Source.Intent = SwitchIntentOff
	if got := SwitchPayloadHash(r.Caller, r.NodeID, r.Request); got != want {
		t.Fatalf("hash = %s, want %s", got, want)
	}
}

func TestSwitchConfigurationToken(t *testing.T) {
	base := SwitchConfigurationSnapshot{NodeID: "node", JournalGeneration: 1, ManagerIncarnation: "manager-1",
		Engines: []SwitchConfigurationEngine{{Engine: "a", SettingsRevision: 1, ConfigDigest: "digest", Intent: SwitchIntentOn, Provenance: "local"}, {Engine: "b"}}, RegistryRuntimeFingerprint: "registry"}
	want := SwitchConfigurationToken(base)
	if !switchRevision.MatchString(want) {
		t.Fatalf("token format = %s", want)
	}
	cases := []struct {
		name   string
		change func(*SwitchConfigurationSnapshot)
		same   bool
	}{
		{"reordered", func(s *SwitchConfigurationSnapshot) { slices.Reverse(s.Engines) }, true},
		{"node", func(s *SwitchConfigurationSnapshot) { s.NodeID += "2" }, false},
		{"generation", func(s *SwitchConfigurationSnapshot) { s.JournalGeneration++ }, false},
		{"incarnation", func(s *SwitchConfigurationSnapshot) { s.ManagerIncarnation += "2" }, false},
		{"engine", func(s *SwitchConfigurationSnapshot) { s.Engines[0].Engine += "2" }, false},
		{"settings revision", func(s *SwitchConfigurationSnapshot) { s.Engines[0].SettingsRevision++ }, false},
		{"digest", func(s *SwitchConfigurationSnapshot) { s.Engines[0].ConfigDigest += "2" }, false},
		{"intent", func(s *SwitchConfigurationSnapshot) { s.Engines[0].Intent = SwitchIntentOff }, false},
		{"provenance", func(s *SwitchConfigurationSnapshot) { s.Engines[0].Provenance = "peer" }, false},
		{"fingerprint", func(s *SwitchConfigurationSnapshot) { s.RegistryRuntimeFingerprint += "2" }, false},
		{"removed entry", func(s *SwitchConfigurationSnapshot) { s.Engines = s.Engines[:1] }, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			snapshot := base
			snapshot.Engines = slices.Clone(base.Engines)
			tc.change(&snapshot)
			before := slices.Clone(snapshot.Engines)
			got := SwitchConfigurationToken(snapshot)
			if (got == want) != tc.same {
				t.Fatalf("token equality = %v, want %v", got == want, tc.same)
			}
			if !reflect.DeepEqual(snapshot.Engines, before) {
				t.Fatal("token mutated input entries")
			}
		})
	}
	t.Run("nil and empty entries", func(t *testing.T) {
		if SwitchConfigurationToken(SwitchConfigurationSnapshot{}) != SwitchConfigurationToken(SwitchConfigurationSnapshot{Engines: []SwitchConfigurationEngine{}}) {
			t.Fatal("nil differs from empty")
		}
	})
}

func TestSwitchTransitions(t *testing.T) {
	legal := map[string][]string{
		SwitchAccepted:       {SwitchValidating, SwitchRejected, SwitchFailedInterrupted},
		SwitchValidating:     {SwitchDraining, SwitchRejected, SwitchFailedValidation, SwitchFailedInterrupted},
		SwitchDraining:       {SwitchSourceStopping, SwitchFailedDrain, SwitchFailedInterrupted},
		SwitchSourceStopping: {SwitchSourceOff, SwitchFailedSourceStop, SwitchFailedInterrupted},
		SwitchSourceOff:      {SwitchTargetStarting, SwitchFailedInterrupted},
		SwitchTargetStarting: {SwitchTargetReady, SwitchFailedTargetStart, SwitchFailedInterrupted},
		SwitchTargetReady:    {SwitchCompleted, SwitchFailedPublication, SwitchFailedInterrupted},
	}
	phases := []string{SwitchAccepted, SwitchValidating, SwitchDraining, SwitchSourceStopping, SwitchSourceOff,
		SwitchTargetStarting, SwitchTargetReady, SwitchCompleted, SwitchFailedValidation, SwitchFailedDrain,
		SwitchFailedSourceStop, SwitchFailedTargetStart, SwitchFailedPublication, SwitchFailedInterrupted, SwitchRejected, "invalid", ""}
	for _, from := range phases {
		t.Run(from, func(t *testing.T) {
			for _, to := range phases {
				want := slices.Contains(legal[from], to)
				if got := ValidTransition(from, to); got != want {
					t.Errorf("%q -> %q = %v, want %v", from, to, got, want)
				}
			}
			wantTerminal := slices.Contains([]string{SwitchCompleted, SwitchFailedValidation, SwitchFailedDrain,
				SwitchFailedSourceStop, SwitchFailedTargetStart, SwitchFailedPublication, SwitchFailedInterrupted, SwitchRejected}, from)
			if IsTerminal(from) != wantTerminal {
				t.Errorf("IsTerminal(%q) = %v, want %v", from, IsTerminal(from), wantTerminal)
			}
		})
	}
}

func TestSwitchCanonicalPayloadEncoding(t *testing.T) {
	request := switchRequestFixture()
	canonical := `[1,"local","canonical-node","switch-42","llamacpp-main","llamacpp-custom","e3v1:` + strings.Repeat("a", 64) + `"]`
	sum := sha256.Sum256([]byte(canonical))
	want := fmt.Sprintf("sha256:%x", sum)
	if got := SwitchPayloadHash("local", "canonical-node", request); got != want {
		t.Fatalf("payload hash = %s, want %s", got, want)
	}
}

func TestSwitchDecodeByteBounds(t *testing.T) {
	raw, err := json.Marshal(switchRequestFixture())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, size := range []int{MaxSwitchRequestBytes, MaxSwitchRequestBytes + 1} {
		t.Run(fmt.Sprintf("request bytes %d", size), func(t *testing.T) {
			data := append(slices.Clone(raw), []byte(strings.Repeat(" ", size-len(raw)))...)
			_, err := DecodeSwitchRequest(data)
			if (err == nil) != (size == MaxSwitchRequestBytes) {
				t.Fatalf("Decode = %v at %d bytes", err, size)
			}
		})
	}
	journal := []byte(`{"version":1,"generation":0,"activeRequestId":"","receipts":{}}`)
	for _, size := range []int{MaxSwitchJournalBytes, MaxSwitchJournalBytes + 1} {
		t.Run(fmt.Sprintf("journal bytes %d", size), func(t *testing.T) {
			data := append(slices.Clone(journal), []byte(strings.Repeat(" ", size-len(journal)))...)
			_, err := DecodeSwitchJournal(data)
			if (err == nil) != (size == MaxSwitchJournalBytes) {
				t.Fatalf("Decode = %v at %d bytes", err, size)
			}
		})
	}
}

func TestSwitchJournalEncodedSizeLimit(t *testing.T) {
	j := SwitchJournal{Version: 1, Generation: 1, Receipts: map[string]SwitchReceipt{}}
	for i := 0; i < MaxSwitchReceipts; i++ {
		r := switchReceiptFixture()
		r.RequestID = fmt.Sprintf("request-%d", i)
		r.Request.RequestID = r.RequestID
		r.Caller = strings.Repeat("a", 8*1024)
		r.PayloadHash = SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
		r.Phase, r.Terminal, r.FinishedAt = SwitchCompleted, true, &r.UpdatedAt
		if err := r.Validate(); err != nil {
			t.Fatalf("individual receipt: %v", err)
		}
		j.Receipts[r.RequestID] = r
	}
	data, err := json.Marshal(j)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if len(data) <= MaxSwitchJournalBytes {
		t.Fatalf("fixture only has %d bytes", len(data))
	}
	if err := j.Validate(); err == nil {
		t.Fatal("accepted oversized journal")
	}
}

func TestSwitchJournalRecoveryFence(t *testing.T) {
	for _, recovery := range []bool{false, true} {
		name := "released terminal receipt"
		if recovery {
			name = "retained recovery fence"
		}
		t.Run(name, func(t *testing.T) {
			r := switchReceiptFixture()
			r.Phase, r.Terminal, r.FinishedAt, r.RecoveryRequired = SwitchFailedInterrupted, true, &r.UpdatedAt, recovery
			r.Error = &SwitchError{Code: SwitchOperationErrorCode, Kind: SwitchErrorInterrupted, Message: "Switch interrupted."}
			j := SwitchJournal{Version: 1, Generation: 1, Receipts: map[string]SwitchReceipt{r.RequestID: r}}
			if recovery {
				j.ActiveRequestID = r.RequestID
			}
			if err := j.Validate(); err != nil {
				t.Fatalf("Validate: %v", err)
			}
		})
	}
}

func TestSwitchEngineStateTerminationEvidence(t *testing.T) {
	now := time.Date(2026, 10, 19, 1, 0, 0, 0, time.UTC)
	running, stopped := true, false
	cases := []struct {
		name  string
		state SwitchEngineState
		valid bool
	}{
		{"verified stopped", SwitchEngineState{Intent: SwitchIntentOff, Running: &stopped, ObservedAt: &now, TerminationVerified: true}, true},
		{"still running", SwitchEngineState{Intent: SwitchIntentOff, Running: &running, ObservedAt: &now, TerminationVerified: true}, false},
		{"missing observation time", SwitchEngineState{Intent: SwitchIntentOff, Running: &stopped, TerminationVerified: true}, false},
		{"zero observation time", SwitchEngineState{Intent: SwitchIntentUnknown, ObservedAt: &time.Time{}}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.state.Validate(); (err == nil) != tc.valid {
				t.Fatalf("Validate = %v, want valid %v", err, tc.valid)
			}
		})
	}
}

func TestSwitchRequestValidation(t *testing.T) {
	cases := []struct {
		name   string
		change func(*SwitchRequest)
		valid  bool
	}{
		{"valid", func(*SwitchRequest) {}, true},
		{"ID limit", func(r *SwitchRequest) { r.RequestID = strings.Repeat("a", MaxSwitchRequestIDBytes) }, true},
		{"ASCII punctuation", func(r *SwitchRequest) { r.RequestID = "a._:-9" }, true},
		{"empty ID", func(r *SwitchRequest) { r.RequestID = "" }, false},
		{"long ID", func(r *SwitchRequest) { r.RequestID = strings.Repeat("a", MaxSwitchRequestIDBytes+1) }, false},
		{"unicode ID", func(r *SwitchRequest) { r.RequestID = "日本" }, false},
		{"leading punctuation", func(r *SwitchRequest) { r.RequestID = "-request" }, false},
		{"ID whitespace", func(r *SwitchRequest) { r.RequestID = "request 1" }, false},
		{"same engines", func(r *SwitchRequest) { r.TargetEngine = r.SourceEngine }, false},
		{"bad source", func(r *SwitchRequest) { r.SourceEngine = "../engine" }, false},
		{"bad target", func(r *SwitchRequest) { r.TargetEngine = "bad engine" }, false},
		{"bad revision", func(r *SwitchRequest) { r.ExpectedRevision = "e3v1:" + strings.Repeat("A", 64) }, false},
		{"short revision", func(r *SwitchRequest) { r.ExpectedRevision = "e3v1:abc" }, false},
		{"wrong version", func(r *SwitchRequest) { r.ExpectedRevision = "e3v2:" + strings.Repeat("a", 64) }, false},
		{"oversized body", func(r *SwitchRequest) { r.NodeID = strings.Repeat("a", MaxSwitchRequestBytes) }, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := switchRequestFixture()
			tc.change(&r)
			if err := r.Validate(); (err == nil) != tc.valid {
				t.Fatalf("Validate = %v, want valid %v", err, tc.valid)
			}
		})
	}
}

func TestSwitchReceiptValidation(t *testing.T) {
	cases := []struct {
		name   string
		change func(*SwitchReceipt)
		valid  bool
	}{
		{"accepted", func(*SwitchReceipt) {}, true},
		{"version", func(r *SwitchReceipt) { r.Version++ }, false},
		{"identity", func(r *SwitchReceipt) { r.NodeID = "different" }, false},
		{"request ID", func(r *SwitchReceipt) { r.RequestID += "2" }, false},
		{"caller", func(r *SwitchReceipt) { r.Caller = "" }, false},
		{"hash", func(r *SwitchReceipt) { r.PayloadHash = "sha256:" + strings.Repeat("0", 64) }, false},
		{"revision", func(r *SwitchReceipt) { r.ExpectedRevision = r.AcceptedRevision }, false},
		{"accepted revision", func(r *SwitchReceipt) { r.AcceptedRevision = "bad" }, false},
		{"sequence", func(r *SwitchReceipt) { r.Sequence = 0 }, false},
		{"phase", func(r *SwitchReceipt) { r.Phase = "bad" }, false},
		{"terminal flag", func(r *SwitchReceipt) { r.Terminal = true }, false},
		{"zero accepted time", func(r *SwitchReceipt) { r.AcceptedAt = time.Time{} }, false},
		{"backward updated time", func(r *SwitchReceipt) { r.UpdatedAt = r.AcceptedAt.Add(-time.Second) }, false},
		{"deadline bound", func(r *SwitchReceipt) { r.Deadline = r.Deadline.Add(time.Nanosecond) }, false},
		{"zero duration", func(r *SwitchReceipt) { r.Deadline = r.AcceptedAt }, false},
		{"premature finish", func(r *SwitchReceipt) { r.FinishedAt = &r.UpdatedAt }, false},
		{"intent", func(r *SwitchReceipt) { r.Source.Intent = "bad" }, false},
		{"unverified termination", func(r *SwitchReceipt) { r.Source.TerminationVerified = true }, false},
		{"unknown observations", func(r *SwitchReceipt) { r.Source.Intent = SwitchIntentUnknown }, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := switchReceiptFixture()
			tc.change(&r)
			if err := r.Validate(); (err == nil) != tc.valid {
				t.Fatalf("Validate = %v, want valid %v", err, tc.valid)
			}
		})
	}
}

func TestSwitchTerminalReceiptValidation(t *testing.T) {
	cases := []struct {
		name     string
		phase    string
		err      *SwitchError
		recovery bool
		valid    bool
	}{
		{"completed", SwitchCompleted, nil, false, true},
		{"completed with recovery", SwitchCompleted, nil, true, false},
		{"failed without error", SwitchFailedDrain, nil, false, false},
		{"failed", SwitchFailedDrain, &SwitchError{Code: SwitchOperationErrorCode, Kind: SwitchErrorDrain, Message: "Drain failed."}, true, true},
		{"rejected", SwitchRejected, &SwitchError{Code: SwitchConfigurationConflictCode, Kind: SwitchErrorConfigurationConflict, Message: "Configuration changed."}, false, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := switchReceiptFixture()
			r.Phase, r.Terminal, r.FinishedAt, r.Error, r.RecoveryRequired = tc.phase, true, &r.UpdatedAt, tc.err, tc.recovery
			if err := r.Validate(); (err == nil) != tc.valid {
				t.Fatalf("Validate = %v, want valid %v", err, tc.valid)
			}
		})
	}
}

func TestSwitchErrorValidation(t *testing.T) {
	cases := []struct {
		name          string
		code          int
		kind, message string
		valid         bool
	}{
		{"message limit", SwitchOperationErrorCode, SwitchErrorStorage, strings.Repeat("a", MaxSwitchErrorMessageBytes), true},
		{"message too long", SwitchOperationErrorCode, SwitchErrorStorage, strings.Repeat("a", MaxSwitchErrorMessageBytes+1), false},
		{"empty message", SwitchOperationErrorCode, SwitchErrorStorage, "", false},
		{"unknown kind", SwitchOperationErrorCode, "other", "error", false},
		{"wrong code", SwitchOperationErrorCode, SwitchErrorRemoteVeto, "error", false},
		{"remote veto", SwitchRemoteVetoCode, SwitchErrorRemoteVeto, "error", true},
		{"engine locked", SwitchEngineLockedCode, SwitchErrorEngineLocked, "error", true},
		{"engine locked with veto code", SwitchRemoteVetoCode, SwitchErrorEngineLocked, "error", false},
		{"remote veto with locked code", SwitchEngineLockedCode, SwitchErrorRemoteVeto, "error", false},
		{"engine locked with operation code", SwitchOperationErrorCode, SwitchErrorEngineLocked, "error", false},
		{"reused ID", SwitchInvalidParamsCode, SwitchErrorReusedID, "error", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := (SwitchError{Code: tc.code, Kind: tc.kind, Message: tc.message}).Validate()
			if (err == nil) != tc.valid {
				t.Fatalf("Validate = %v, want valid %v", err, tc.valid)
			}
		})
	}
}

func TestSwitchJournalValidation(t *testing.T) {
	fixture := func() SwitchJournal {
		r := switchReceiptFixture()
		return SwitchJournal{Version: 1, Generation: 1, ActiveRequestID: r.RequestID, Receipts: map[string]SwitchReceipt{r.RequestID: r}}
	}
	cases := []struct {
		name   string
		change func(*SwitchJournal)
		valid  bool
	}{
		{"active", func(*SwitchJournal) {}, true},
		{"version", func(j *SwitchJournal) { j.Version++ }, false},
		{"generation", func(j *SwitchJournal) { j.Generation = 0 }, false},
		{"missing map", func(j *SwitchJournal) { j.Receipts = nil }, false},
		{"wrong key", func(j *SwitchJournal) { j.Receipts["other"] = j.Receipts[j.ActiveRequestID] }, false},
		{"no active ID", func(j *SwitchJournal) { j.ActiveRequestID = "" }, false},
		{"missing active receipt", func(j *SwitchJournal) { j.ActiveRequestID = "other" }, false},
		{"multiple nonterminal receipts", func(j *SwitchJournal) {
			r := switchReceiptFixture()
			r.RequestID, r.Request.RequestID = "another", "another"
			r.PayloadHash = SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
			j.Receipts[r.RequestID] = r
		}, false},
		{"empty", func(j *SwitchJournal) { j.Generation = 0; j.ActiveRequestID = ""; clear(j.Receipts) }, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			j := fixture()
			tc.change(&j)
			if err := j.Validate(); (err == nil) != tc.valid {
				t.Fatalf("Validate = %v, want valid %v", err, tc.valid)
			}
		})
	}
	t.Run("receipt cap", func(t *testing.T) {
		j := SwitchJournal{Version: 1, Generation: 1, Receipts: map[string]SwitchReceipt{}}
		for i := 0; i < MaxSwitchReceipts; i++ {
			r := switchReceiptFixture()
			r.RequestID = fmt.Sprintf("request-%d", i)
			r.Request.RequestID = r.RequestID
			r.PayloadHash = SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
			r.Phase, r.Terminal, r.FinishedAt = SwitchCompleted, true, &r.UpdatedAt
			j.Receipts[r.RequestID] = r
		}
		if err := j.Validate(); err != nil {
			t.Fatalf("at cap: %v", err)
		}
		j.Receipts["extra"] = switchReceiptFixture()
		if err := j.Validate(); err == nil {
			t.Fatal("accepted too many receipts")
		}
	})
}

func TestSwitchJournalLocalObligationValidation(t *testing.T) {
	test := func(name string, local, orphan, empty bool, valid bool) {
		t.Run(name, func(t *testing.T) {
			r := switchReceiptFixture()
			if !local {
				r.Caller = "peer"
				r.PayloadHash = SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
			}
			id := r.RequestID
			if orphan {
				id = "missing"
			}
			o := SwitchLocalObligation{SourceOff: !empty, TargetOff: !empty}
			j := SwitchJournal{Version: SwitchVersion, Generation: 1, ActiveRequestID: r.RequestID, Receipts: map[string]SwitchReceipt{r.RequestID: r}, LocalObligations: map[string]SwitchLocalObligation{id: o}}
			if err := j.Validate(); (err == nil) != valid {
				t.Fatalf("Validate = %v; want valid %v", err, valid)
			}
		})
	}
	test("both engines on remote operation", false, false, false, true)
	test("local operation cannot own remote veto", true, false, false, false)
	test("obligation requires its receipt", false, true, false, false)
	test("empty obligation refused", false, false, true, false)
}

func TestSwitchJournalPauseAppliedValidation(t *testing.T) {
	test := func(name string, obligation SwitchLocalObligation, valid bool) {
		t.Run(name, func(t *testing.T) {
			r := switchReceiptFixture()
			r.Caller = "peer"
			r.PayloadHash = SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
			j := SwitchJournal{Version: SwitchVersion, Generation: 1, ActiveRequestID: r.RequestID, Receipts: map[string]SwitchReceipt{r.RequestID: r}, LocalObligations: map[string]SwitchLocalObligation{r.RequestID: obligation}}
			data, err := json.Marshal(j)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := DecodeSwitchJournal(data)
			if (err == nil) != valid {
				t.Fatalf("decode = %v; want valid %v", err, valid)
			}
			if valid && decoded.LocalObligations[r.RequestID] != obligation {
				t.Fatalf("obligation changed on reload: %+v", decoded.LocalObligations[r.RequestID])
			}
		})
	}
	test("old unapplied pause without completion field", SwitchLocalObligation{Pause: true}, true)
	test("old discharged pause without completion field", SwitchLocalObligation{Pause: true, Discharged: true}, true)
	test("applied pause", SwitchLocalObligation{Pause: true, PauseApplied: true}, true)
	test("discharged applied pause", SwitchLocalObligation{Pause: true, PauseApplied: true, Discharged: true}, true)
	test("completion without pause", SwitchLocalObligation{SourceOff: true, PauseApplied: true}, false)
}

func TestSwitchStrictDecode(t *testing.T) {
	raw, err := json.Marshal(switchRequestFixture())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	cases := []struct {
		name  string
		data  []byte
		valid bool
	}{
		{"valid", raw, true},
		{"whitespace", append(slices.Clone(raw), []byte(" \n\t")...), true},
		{"unknown", []byte(strings.TrimSuffix(string(raw), "}") + `,"caller":"spoofed"}`), false},
		{"trailing object", append(slices.Clone(raw), []byte(" {}")...), false},
		{"trailing garbage", append(slices.Clone(raw), 'x'), false},
		{"oversized whitespace", append(slices.Clone(raw), []byte(strings.Repeat(" ", MaxSwitchRequestBytes))...), false},
		{"null", []byte("null"), false},
		{"array", []byte("[]"), false},
		{"missing required", []byte("{}"), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := DecodeSwitchRequest(tc.data)
			if (err == nil) != tc.valid {
				t.Fatalf("Decode = %v, want valid %v", err, tc.valid)
			}
		})
	}
}

func TestSwitchStatusDecode(t *testing.T) {
	if _, err := DecodeSwitchStatusRequest([]byte(`{"requestId":"request-1","afterSequence":0}`)); err != nil {
		t.Fatalf("zero sequence: %v", err)
	}
	for _, raw := range []string{`null`, `{"caller":"spoofed"}`, `{"afterSequence":1}`, `{"requestId":"bad id"}`, `{} {}`} {
		if _, err := DecodeSwitchStatusRequest([]byte(raw)); err == nil {
			t.Errorf("accepted status %s", raw)
		}
	}
}

func TestSwitchFenceContract(t *testing.T) {
	checkSwitchWire(t, SwitchFenceRequest{RequestID: "lost"}, "requestId")
	checkSwitchWire(t, SwitchFenceResult{NodeID: "node", RequestID: "lost", Fenced: true}, "nodeId requestId fenced")
	for _, raw := range []string{`{}`, `null`, `{"requestId":"bad id"}`, `{"requestId":"lost","caller":"spoof"}`, `{"requestId":"lost"} {}`, `{"requestId":"` + strings.Repeat("x", 129) + `"}`} {
		if _, err := DecodeSwitchFenceRequest([]byte(raw)); err == nil {
			t.Errorf("accepted fence request %s", raw)
		}
	}
	if _, err := DecodeSwitchFenceRequest([]byte(`{"nodeId":"alias","requestId":"lost"}`)); err != nil {
		t.Fatal(err)
	}
	r := switchReceiptFixture()
	test := func(name string, result SwitchFenceResult, valid bool) {
		t.Run(name, func(t *testing.T) {
			if err := result.Validate(); (err == nil) != valid {
				t.Fatalf("Validate = %v; want valid %v", err, valid)
			}
		})
	}
	test("fenced", SwitchFenceResult{NodeID: r.NodeID, RequestID: r.RequestID, Fenced: true}, true)
	test("accepted", SwitchFenceResult{NodeID: r.NodeID, RequestID: r.RequestID, Receipt: &r}, true)
	test("neither", SwitchFenceResult{NodeID: r.NodeID, RequestID: r.RequestID}, false)
	test("both", SwitchFenceResult{NodeID: r.NodeID, RequestID: r.RequestID, Fenced: true, Receipt: &r}, false)
	test("wrong node", SwitchFenceResult{NodeID: "other", RequestID: r.RequestID, Receipt: &r}, false)
	test("wrong request", SwitchFenceResult{NodeID: r.NodeID, RequestID: "other", Receipt: &r}, false)
	test("no node", SwitchFenceResult{RequestID: r.RequestID, Fenced: true}, false)
	if err := (SwitchError{Code: SwitchConfigurationConflictCode, Kind: SwitchErrorFenced, Message: "Fenced."}).Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestSwitchTombstoneJournalValidation(t *testing.T) {
	test := func(name string, tombstones []SwitchTombstone, receipt *SwitchReceipt, generation uint64, valid bool) {
		t.Run(name, func(t *testing.T) {
			j := SwitchJournal{Version: SwitchVersion, Generation: generation, Receipts: map[string]SwitchReceipt{}, Tombstones: tombstones}
			if receipt != nil {
				j.Receipts[receipt.RequestID], j.ActiveRequestID = *receipt, receipt.RequestID
			}
			raw, err := json.Marshal(j)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := DecodeSwitchJournal(raw); (err == nil) != valid {
				t.Fatalf("decode = %v; want valid %v", err, valid)
			}
		})
	}
	one := SwitchTombstone{Caller: "local", RequestID: "switch-42"}
	r := switchReceiptFixture()
	test("one", []SwitchTombstone{one}, nil, 1, true)
	test("zero generation", []SwitchTombstone{one}, nil, 0, false)
	test("duplicate pair", []SwitchTombstone{one, one}, nil, 1, false)
	test("different owners", []SwitchTombstone{one, {Caller: "peer", RequestID: one.RequestID}}, nil, 1, true)
	test("owned receipt collision", []SwitchTombstone{one}, &r, 1, false)
	test("foreign receipt coexistence", []SwitchTombstone{{Caller: "peer", RequestID: one.RequestID}}, &r, 1, true)
	test("empty caller", []SwitchTombstone{{RequestID: "lost"}}, nil, 1, false)
	test("oversize caller", []SwitchTombstone{{Caller: strings.Repeat("x", 129), RequestID: "lost"}}, nil, 1, false)
	test("bad caller", []SwitchTombstone{{Caller: "bad caller", RequestID: "lost"}}, nil, 1, false)
	test("bad ID", []SwitchTombstone{{Caller: "local", RequestID: "bad id"}}, nil, 1, false)
	many := make([]SwitchTombstone, MaxSwitchTombstones+1)
	for i := range many {
		many[i] = SwitchTombstone{Caller: "local", RequestID: fmt.Sprintf("id-%d", i)}
	}
	test("count bound", many, nil, 1, false)
}

func TestSwitchJournalDecode(t *testing.T) {
	j := SwitchJournal{Version: 1, Receipts: map[string]SwitchReceipt{}}
	raw, err := json.Marshal(j)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if _, err := DecodeSwitchJournal(raw); err != nil {
		t.Fatalf("valid journal: %v", err)
	}
	for _, data := range [][]byte{[]byte(`{"version":1,"receipts":{},"unknown":true}`), append(slices.Clone(raw), []byte(" {}")...), []byte(strings.Repeat(" ", MaxSwitchJournalBytes+1))} {
		if _, err := DecodeSwitchJournal(data); err == nil {
			t.Fatal("accepted invalid journal")
		}
	}
}
