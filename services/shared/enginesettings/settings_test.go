// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package enginesettings

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/modelconfig"
)

func TestConfigCloneAndLaunchReconstruction(t *testing.T) {
	ram, vram := int64(1024), int64(0)
	launch := LaunchState{ServerPort: 8080, LaunchText: "--opaque", Serving: &engines.ServingConfig{Version: 1, Models: []engines.ServingModel{{ModelID: "exact", Memory: engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram}}}}}
	c := ConfigFromLaunch(launch, 8081)
	copy := c.Clone()
	if !c.Equal(copy) || c.ProxyPort != 8081 {
		t.Fatal("launch reconstruction differs")
	}
	*copy.Serving.Models[0].Memory.HostRAMBytes = 2
	if c.Equal(copy) || *launch.Serving.Models[0].Memory.HostRAMBytes != 1024 {
		t.Fatal("config clone aliases launch")
	}
	snapshot := Snapshot{Settings: c, Serving: &engines.ServingReadback{Observed: []engines.ServingObservation{{Slots: 1}}}}
	cloned := snapshot.Clone()
	cloned.Settings.Serving.Models[0].ModelID = "changed"
	cloned.Serving.Observed[0].Slots = 2
	if snapshot.Settings.Serving.Models[0].ModelID != "exact" || snapshot.Serving.Observed[0].Slots != 1 {
		t.Fatal("snapshot clone aliases original")
	}
}

func TestConfigPreservesModelLoads(t *testing.T) {
	slots := int32(2)
	launch := LaunchState{ModelLoads: []modelconfig.ModelLoad{{ModelID: "z", Requested: modelconfig.RequestedLoad{Slots: &slots}}, {ModelID: "a"}}}
	c := ConfigFromLaunch(launch, 8081)
	canonical, err := c.Canonical()
	if err != nil {
		t.Fatal(err)
	}
	if canonical.ModelLoads[0].ModelID != "a" || !canonical.Equal(c) {
		t.Fatal("canonical config changed meaning")
	}
	*canonical.ModelLoads[1].Requested.Slots = 4
	if slots != 2 || canonical.Equal(c) {
		t.Fatal("model load aliases launch")
	}
}

func TestModelChangeRequiresBaselineAndReview(t *testing.T) {
	slots := int32(1)
	baseline := Config{ServerPort: 8080}
	r := Request{Engine: "llamacpp", RequestID: "request-1", Settings: baseline, ModelChange: &modelconfig.ModelChange{ModelID: "m", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &slots}}}}
	if err := r.ValidateModelChange(baseline, false); err != nil {
		t.Fatal(err)
	}
	if err := r.ValidateModelChange(baseline, true); err == nil {
		t.Fatal("accepted unreviewed apply")
	}
	r.ModelChange.PreviewDigest = strings.Repeat("a", 64)
	if err := r.ValidateModelChange(baseline, true); err != nil {
		t.Fatal(err)
	}
	r.Settings.ServerPort++
	if err := r.ValidateModelChange(baseline, true); err == nil {
		t.Fatal("accepted edited baseline")
	}
}

func TestConfigOmitsAbsentModelLoads(t *testing.T) {
	data, err := json.Marshal(Config{ServerPort: 8080, ProxyPort: 8081, LaunchText: "--opaque"})
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"serverPort":8080,"proxyPort":8081,"launchText":"--opaque"}` {
		t.Fatal("absent model loads changed ordinary settings shape")
	}
}

func settingsJSON(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestDecodeRequestRejectsDuplicateUnknownAndAuthorityFields(t *testing.T) {
	for name, input := range map[string]string{
		"duplicate revision":                       `{"engine":"llamacpp","expectedRevision":"1","expectedRevision":"2"}`,
		"unknown root":                             `{"engine":"llamacpp","telemetry":{}}`,
		"unknown setting":                          `{"engine":"llamacpp","settings":{"modelLoad":[]}}`,
		"case folded setting":                      `{"engine":"llamacpp","settings":{"ModelLoads":[]}}`,
		"duplicate load":                           `{"engine":"llamacpp","settings":{"modelLoads":[{"modelId":"m","requested":{"slots":1,"slots":2}}]}}`,
		"unknown resolved option":                  `{"engine":"llamacpp","settings":{"modelLoads":[{"modelId":"m","requested":{},"resolved":{"telemetry":{}}}]}}`,
		"unknown serving through custom decoder":   `{"engine":"llamacpp","settings":{"serving":{"version":1,"models":[],"unknown":true}}}`,
		"duplicate serving through custom decoder": `{"engine":"llamacpp","settings":{"serving":{"version":1,"models":[],"models":[]}}}`,
		"unknown nested policy":                    `{"engine":"llamacpp","modelChange":{"modelId":"m","expectedPolicyRevision":"0","policy":{"qualification":{}}}}`,
		"public internal guard":                    `{"engine":"llamacpp","preserveCORS":true}`,
		"number revision":                          `{"engine":"llamacpp","expectedRevision":9007199254740993}`,
		"noncanonical revision":                    `{"engine":"llamacpp","expectedRevision":"01"}`,
		"negative revision":                        `{"engine":"llamacpp","expectedRevision":"-1"}`,
		"overflow revision":                        `{"engine":"llamacpp","expectedRevision":"18446744073709551616"}`,
		"trailing":                                 `{"engine":"llamacpp"} {}`,
		"root null":                                `null`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeRequest([]byte(input)); err == nil {
				t.Fatal("untrusted boundary accepted invalid request")
			}
		})
	}
}

func TestExistingCountersKeepExactNumericWireShape(t *testing.T) {
	r := Request{Engine: "llamacpp", ExpectedRevision: MaxSafeCounter}
	data := settingsJSON(t, r)
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	var revision uint64
	if err := json.Unmarshal(fields["expectedRevision"], &revision); err != nil || revision != MaxSafeCounter {
		t.Fatalf("expected exact numeric revision, got %s (%v)", fields["expectedRevision"], err)
	}
	got, err := DecodeRequest(data)
	if err != nil || got.ExpectedRevision != r.ExpectedRevision {
		t.Fatalf("revision did not round trip: %v", err)
	}
	snapshot := Snapshot{Revision: MaxSafeCounter, AppliedRevision: MaxSafeCounter, Sequence: MaxSafeCounter}
	var decoded Snapshot
	if err := modelconfig.DecodeStrict(settingsJSON(t, snapshot), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Revision != snapshot.Revision || decoded.AppliedRevision != snapshot.AppliedRevision || decoded.Sequence != snapshot.Sequence {
		t.Fatal("snapshot revision precision changed")
	}
	preview := Preview{Revision: MaxSafeCounter}
	var decodedPreview Preview
	if err := modelconfig.DecodeStrict(settingsJSON(t, preview), &decodedPreview); err != nil || decodedPreview.Revision != preview.Revision {
		t.Fatalf("preview revision did not round trip: %v", err)
	}
	for _, value := range []any{Request{ExpectedRevision: MaxSafeCounter + 1}, Snapshot{Revision: MaxSafeCounter + 1}, Snapshot{Sequence: MaxSafeCounter + 1}, Snapshot{AppliedRevision: MaxSafeCounter + 1}, Preview{Revision: MaxSafeCounter + 1}} {
		if _, err := json.Marshal(value); err == nil {
			t.Fatalf("encoded inexact counter in %T", value)
		}
	}
	for _, value := range []struct {
		body   string
		target any
	}{
		{`{"expectedRevision":9007199254740992}`, &Request{}},
		{`{"revision":9007199254740992}`, &Snapshot{}},
		{`{"appliedRevision":9007199254740992}`, &Snapshot{}},
		{`{"sequence":9007199254740992}`, &Snapshot{}},
		{`{"revision":9007199254740992}`, &Preview{}},
	} {
		if err := json.Unmarshal([]byte(value.body), value.target); err == nil {
			t.Fatalf("decoded inexact counter in %T", value.target)
		}
	}
}

func TestCompleteRequestPayloadBudgetOnEveryPath(t *testing.T) {
	r := Request{Engine: "llamacpp", Settings: Config{}}
	emptySize := len(settingsJSON(t, r))
	r.Settings.LaunchText = strings.Repeat("x", modelconfig.MaxPayloadBytes-emptySize)
	if len(settingsJSON(t, r)) != modelconfig.MaxPayloadBytes {
		t.Fatal("fixture did not reach exact complete request budget")
	}
	if _, err := DecodeRequest(settingsJSON(t, r)); err != nil {
		t.Fatalf("exact 64 KiB request refused: %v", err)
	}
	if err := r.ValidateModelChange(Config{}, false); err != nil {
		t.Fatalf("ordinary exact-boundary request refused: %v", err)
	}
	r.Settings.LaunchText += "x"
	if _, err := r.Settings.Canonical(); err != nil {
		t.Fatalf("fixture must fit Config while exceeding request envelope: %v", err)
	}
	if err := r.ValidateModelChange(Config{}, false); err == nil {
		t.Fatal("ordinary path skipped complete request size check")
	}
	if _, err := DecodeRequest(settingsJSON(t, r)); err == nil {
		t.Fatal("boundary accepted oversized complete request")
	}
	r.Settings.LaunchText = ""
	data := settingsJSON(t, r)
	if _, err := DecodeRequest(append(data, []byte(strings.Repeat(" ", modelconfig.MaxPayloadBytes))...)); err == nil {
		t.Fatal("raw input whitespace bypassed complete transport limit")
	}
}

func TestConfigBudgetIncludesFieldsBesidesModelLoads(t *testing.T) {
	c := Config{LaunchText: strings.Repeat("x", modelconfig.MaxPayloadBytes), ModelLoads: []modelconfig.ModelLoad{{ModelID: "m"}}}
	if _, err := modelconfig.CanonicalLoads(c.ModelLoads); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Canonical(); err == nil {
		t.Fatal("checked loads alone instead of complete Config")
	}
}

func storedModelLoad() modelconfig.ModelLoad {
	return modelconfig.ModelLoad{ModelID: "m", Resolved: &modelconfig.ResolvedLoad{
		PerSlotContext: 8192, TotalContext: 8192, Slots: 1, GPU: modelconfig.GPU{Mode: "cpu"},
		Artifacts:     []modelconfig.ArtifactBinding{{Role: "model", Path: "/models/m.gguf", ResolvedTarget: "/blobs/m", Size: "9007199254740993", MtimeNS: "1780000000000000001", SampleSHA256: strings.Repeat("a", 64)}},
		NativeBuildID: "native-build", SourceInputDigest: strings.Repeat("b", 64), PresetDigest: strings.Repeat("c", 64),
	}}
}

func TestCallerCannotReplaceTargetResolvedAuthority(t *testing.T) {
	baseline := Config{ModelLoads: []modelconfig.ModelLoad{storedModelLoad()}}
	r := Request{Engine: "llamacpp", Settings: baseline.Clone()}
	r.Settings.ProxyPort = 8081
	if err := r.ValidateModelChange(baseline, false); err != nil {
		t.Fatalf("ordinary port edit must preserve target readback: %v", err)
	}
	r.Settings.ModelLoads[0].Resolved.Artifacts[0].Path = "/caller/supplied.gguf"
	if err := r.ValidateModelChange(baseline, false); err == nil {
		t.Fatal("accepted caller replacement of target binding")
	}
	r.Settings = Config{ModelLoads: []modelconfig.ModelLoad{storedModelLoad()}}
	if err := r.ValidateModelChange(Config{}, false); err == nil {
		t.Fatal("accepted caller-introduced resolved authority")
	}
	r.Settings = Config{}
	if err := r.ValidateModelChange(baseline, false); err == nil {
		t.Fatal("ordinary constructor silently dropped model loads")
	}
}

func TestStoredConfigBoundaryValidatesResolvedRecords(t *testing.T) {
	c := Config{ModelLoads: []modelconfig.ModelLoad{storedModelLoad()}}
	got, err := DecodeConfig(settingsJSON(t, c))
	if err != nil || !got.Equal(c) {
		t.Fatalf("stored config round trip failed: %v", err)
	}
	c.ModelLoads[0].Resolved.TotalContext++
	if _, err := DecodeConfig(settingsJSON(t, c)); err == nil {
		t.Fatal("stored decode accepted inconsistent resolution")
	}
}

func TestStoredDecodeRejectsUnknownResolvedFieldOnOtherwiseValidRecord(t *testing.T) {
	c := Config{ModelLoads: []modelconfig.ModelLoad{storedModelLoad()}}
	data := settingsJSON(t, c)
	if _, err := DecodeConfig(data); err != nil {
		t.Fatal(err)
	}
	invalid := strings.Replace(string(data), `"nativeBuildId":`, `"callerTelemetry":{},"nativeBuildId":`, 1)
	if invalid == string(data) {
		t.Fatal("fixture did not insert unknown resolved field")
	}
	if _, err := DecodeConfig([]byte(invalid)); err == nil {
		t.Fatal("stored decode ignored unknown resolved field")
	}
}

func TestRequestRestrictsMinPToExistingLlamaFieldSurface(t *testing.T) {
	input := Request{Engine: "ollama", ModelChange: &modelconfig.ModelChange{ModelID: "m", ExpectedPolicyRevision: "0", Policy: &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"min_p": nil}}}}
	if _, err := DecodeRequest(settingsJSON(t, input)); err == nil {
		t.Fatal("non-llama selected min_p despite existing field surface")
	}
	input.Engine = "llamacpp"
	if _, err := DecodeRequest(settingsJSON(t, input)); err != nil {
		t.Fatal(err)
	}
}

func TestRelayHasDistinctCorrelatedInspectShape(t *testing.T) {
	input := Relay{ID: "correlation-1", Method: "inspect", Caller: "verified-peer", Inspect: &modelconfig.InspectRequest{Engine: "llamacpp", NodeID: "target", ModelID: "exact-model"}}
	got, err := DecodeRelay(settingsJSON(t, input))
	if err != nil || !reflect.DeepEqual(got, input) {
		t.Fatalf("inspect correlation did not round trip: %v", err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(settingsJSON(t, input), &fields); err != nil {
		t.Fatal(err)
	}
	if fields["request"] != nil || fields["inspect"] == nil {
		t.Fatal("inspect masqueraded as engine-settings Request")
	}
	input.Request = Request{Engine: "llamacpp"}
	if _, err := DecodeRelay(settingsJSON(t, input)); err == nil {
		t.Fatal("inspect relay also accepted a settings request")
	}
	input.Request = Request{}
	input.Method = "apply"
	if _, err := DecodeRelay(settingsJSON(t, input)); err == nil {
		t.Fatal("settings relay accepted inspect payload")
	}
}

func TestRelayBudgetIncludesCorrelationEnvelope(t *testing.T) {
	r := Request{Engine: "llamacpp"}
	size := len(settingsJSON(t, r))
	r.Settings.LaunchText = strings.Repeat("x", modelconfig.MaxPayloadBytes-size)
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	relay := Relay{ID: "correlation", Method: "preview", Request: r}
	if _, err := DecodeRelay(settingsJSON(t, relay)); err == nil {
		t.Fatal("relay envelope exceeded transport budget")
	}
}

func TestPreviewCloneDetachesNestedReviewData(t *testing.T) {
	mayInterrupt := false
	p := Preview{Settings: Config{ModelLoads: []modelconfig.ModelLoad{storedModelLoad()}}, Errors: map[string]string{"field": "reason"}, Diff: []modelconfig.FieldDiff{{Field: "slots", Before: json.RawMessage(`1`), After: json.RawMessage(`2`)}}, RequiredArtifacts: []modelconfig.ArtifactSummary{{Size: "1"}}, Estimate: &modelconfig.Estimate{Status: "unknown", Unknowns: []string{"mapping"}, Domains: []modelconfig.MemoryDomain{{ID: "host"}}}, Impact: &modelconfig.Impact{RestartScope: "unknown", MayInterruptActive: &mayInterrupt, AffectedModels: []string{"m"}}, Auto: &modelconfig.AutoProposal{Status: "unavailable", Warnings: []string{"pending owner decision"}}, Args: []string{"arg"}, Env: []string{"key=value"}}
	before := settingsJSON(t, p)
	got := p.Clone()
	got.Settings.ModelLoads[0].Resolved.Artifacts[0].Path = "changed"
	got.Errors["field"] = "changed"
	got.Diff[0].Before[0] = '3'
	got.Diff[0].After[0] = '4'
	got.RequiredArtifacts[0].Size = "2"
	got.Estimate.Unknowns[0] = "changed"
	got.Estimate.Domains[0].ID = "changed"
	*got.Impact.MayInterruptActive = true
	got.Impact.AffectedModels[0] = "changed"
	got.Auto.Warnings[0] = "changed"
	got.Args[0], got.Env[0] = "changed", "changed"
	if !reflect.DeepEqual(before, settingsJSON(t, p)) {
		t.Fatal("preview clone aliases nested review data")
	}
}

func TestSnapshotCloneKeepsDesiredAndEffectiveLoadsIndependent(t *testing.T) {
	launch := LaunchState{ModelLoads: []modelconfig.ModelLoad{storedModelLoad()}}
	c := ConfigFromLaunch(launch, 8081)
	effective := c.Clone()
	s := Snapshot{Settings: c, EffectiveSettings: &effective, CoupledStage: "native-persisted", PolicyRevision: "9007199254740993", UnresolvedError: "policy publication pending"}
	copy := s.Clone()
	copy.Settings.ModelLoads[0].Resolved.Artifacts[0].Path = "changed desired"
	copy.EffectiveSettings.ModelLoads[0].Resolved.Artifacts[0].Path = "changed effective"
	if launch.ModelLoads[0].Resolved.Artifacts[0].Path != "/models/m.gguf" || s.Settings.ModelLoads[0].Resolved.Artifacts[0].Path != "/models/m.gguf" || s.EffectiveSettings.ModelLoads[0].Resolved.Artifacts[0].Path != "/models/m.gguf" {
		t.Fatal("constructor or snapshot clone aliases target authority")
	}
	if copy.CoupledStage != s.CoupledStage || copy.PolicyRevision != s.PolicyRevision || copy.UnresolvedError != s.UnresolvedError {
		t.Fatal("snapshot clone lost coupled state")
	}
}
