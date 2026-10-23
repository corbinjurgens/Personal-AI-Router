// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/buildinfo"
	"nvpair-shared/engines"
	"nvpair-shared/ipc"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/servicectl"
)

// Synthetic private content. None of it may reach either output format.
const (
	sentinelCredential = "NVPAIR_PROVIDER_SECRET_SENTINEL_CREDENTIAL"
	sentinelHost       = "sentinel-origin.example.invalid"
	sentinelPath       = "/private/sentinel/policy/path"
	sentinelModel      = "sentinel-private-model:7b"
	sentinelError      = "SENTINEL-RAW-ERROR-TEXT"
	sentinelUnknown    = "SENTINEL-UNKNOWN-FIELD-VALUE"
	sentinelNotice     = "SENTINEL-NOTIFICATION-PAYLOAD"
)

var diagnoseSentinels = []string{sentinelCredential, sentinelHost, sentinelPath, sentinelModel, sentinelError, sentinelUnknown, sentinelNotice}

const testBuildID = "git:0123456789abcdef0123456789abcdef01234567"

type diagnoseCall struct{ method, params string }

// diagnoseAnswer answers one occurrence (0-based) of a method.
type diagnoseAnswer func(ctx context.Context, occurrence int) (json.RawMessage, error)

// fakeDiagnoseService is an injected caller with scripted answers. A method
// without an answer fails with a raw error carrying a sentinel.
type fakeDiagnoseService struct {
	answers map[string]diagnoseAnswer
	calls   []diagnoseCall
}

func (f *fakeDiagnoseService) call(ctx context.Context, method string, params json.RawMessage) (json.RawMessage, error) {
	occurrence := 0
	for _, c := range f.calls {
		if c.method == method {
			occurrence++
		}
	}
	f.calls = append(f.calls, diagnoseCall{method: method, params: string(params)})
	answer, ok := f.answers[method]
	if !ok {
		return nil, fmt.Errorf("%s: %s %s", method, sentinelError, sentinelPath)
	}
	return answer(ctx, occurrence)
}

func fixed(result json.RawMessage) diagnoseAnswer {
	return func(context.Context, int) (json.RawMessage, error) { return result, nil }
}

// changing answers first on the first occurrence and then on every later one.
func changing(first, later json.RawMessage) diagnoseAnswer {
	return func(_ context.Context, occurrence int) (json.RawMessage, error) {
		if occurrence == 0 {
			return first, nil
		}
		return later, nil
	}
}

func failing(err error) diagnoseAnswer {
	return func(context.Context, int) (json.RawMessage, error) { return nil, err }
}

func mustJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	return data
}

func serviceStatusResult(t *testing.T, brokerRestarts int) json.RawMessage {
	t.Helper()
	return mustJSON(t, statusResult{PID: 4242, BrokerPID: 4243, Clients: 1, StartedAt: "2026-10-23T00:00:00Z",
		Version: "0.2.0", ServiceBuildID: testBuildID, BrokerRestarts: brokerRestarts})
}

func facadeResult(t *testing.T, ready bool, port int) json.RawMessage {
	t.Helper()
	return mustJSON(t, map[string]any{"ready": ready, "port": port})
}

type policyFixture struct {
	revision, activeRevision, activation, availability string
	recovery                                           bool
	generation                                         uint64
	models                                             []nodepolicy.ModelConfigurationEntry
}

func healthyPolicy() policyFixture {
	loaded := true
	return policyFixture{
		revision: "1792285135482000000", activeRevision: "1792285135482000000", activation: "active",
		availability: "available", generation: 7,
		models: []nodepolicy.ModelConfigurationEntry{
			{Engine: engines.NameOllama, Model: sentinelModel, DesiredResident: true, ObservedLoaded: &loaded, State: nodepolicy.ModelStateSatisfied},
			{Engine: engines.NameOllama, Model: sentinelModel + "-second", DesiredResident: true, State: nodepolicy.ModelStateBlocked, Reason: nodepolicy.ModelReasonNoFit},
		},
	}
}

// policyGetResult is a policy:get answer carrying private routing content,
// operational error text and an unknown field beside the allowlisted facts.
func policyGetResult(t *testing.T, p policyFixture) json.RawMessage {
	t.Helper()
	return mustJSON(t, map[string]any{
		"policy": map[string]any{
			"revision": p.revision,
			"routing": map[string]any{"origins": []map[string]any{{
				"baseUrl": "https://" + sentinelHost + "/v1", "credentialRef": sentinelCredential, "configPath": sentinelPath,
			}}},
		},
		"availability":     p.availability,
		"recoveryRequired": p.recovery,
		"activeRevision":   p.activeRevision,
		"activation":       p.activation,
		"activationError":  sentinelError,
		"futureField":      sentinelUnknown,
		"modelConfiguration": nodepolicy.ModelConfiguration{
			PolicyRevision: p.revision, ObservationGeneration: p.generation, Models: p.models,
		},
	})
}

// newHealthyDiagnoseService answers every read of a complete capture.
func newHealthyDiagnoseService(t *testing.T) *fakeDiagnoseService {
	t.Helper()
	f := &fakeDiagnoseService{answers: map[string]diagnoseAnswer{
		methodStatus: fixed(serviceStatusResult(t, 0)),
		"policy:get": fixed(policyGetResult(t, healthyPolicy())),
	}}
	ports := []int{11434, 1234, 8080}
	for i, e := range engines.All() {
		f.answers[e.ComponentName()+":get-status"] = fixed(facadeResult(t, true, ports[i%len(ports)]))
	}
	return f
}

func facadeMethod(t *testing.T, name string) string {
	t.Helper()
	e, ok := engines.ByName(name)
	if !ok {
		t.Fatalf("no built-in engine %q", name)
	}
	return e.ComponentName() + ":get-status"
}

func wantDiagnoseSequence() []diagnoseCall {
	seq := []diagnoseCall{{methodStatus, ""}, {"policy:get", "{}"}}
	for _, e := range engines.All() {
		seq = append(seq, diagnoseCall{e.ComponentName() + ":get-status", ""})
	}
	return append(seq, diagnoseCall{"policy:get", "{}"}, diagnoseCall{methodStatus, ""})
}

// diagnosed is one capture rendered in both formats.
type diagnosed struct {
	report diagnoseReport
	json   string
	text   string
	code   int
}

// runDiagnoseCapture captures once, renders both formats, decodes the JSON
// and checks that no private sentinel reached either.
func runDiagnoseCapture(t *testing.T, ctx context.Context, call diagnoseCaller) diagnosed {
	t.Helper()
	report, code := captureDiagnosis(ctx, call)
	var jsonOut, textOut bytes.Buffer
	writeDiagnoseJSON(&jsonOut, report)
	writeDiagnoseText(&textOut, report)
	var decoded diagnoseReport
	if err := json.Unmarshal(jsonOut.Bytes(), &decoded); err != nil {
		t.Fatalf("decode JSON report: %v\n%s", err, jsonOut.String())
	}
	assertNoSentinels(t, jsonOut.String(), textOut.String())
	assertProbeStatements(t, decoded, textOut.String())
	return diagnosed{report: decoded, json: jsonOut.String(), text: textOut.String(), code: code}
}

func assertNoSentinels(t *testing.T, outputs ...string) {
	t.Helper()
	for _, out := range outputs {
		for _, s := range diagnoseSentinels {
			if strings.Contains(out, s) {
				t.Fatalf("output contains private sentinel %q:\n%s", s, out)
			}
		}
	}
}

func assertProbeStatements(t *testing.T, r diagnoseReport, text string) {
	t.Helper()
	if r.Schema != 1 || r.SnapshotAtomic || r.InferenceProbed || r.PeerConnectivityProbed {
		t.Fatalf("report header = schema %d atomic %v inference %v peer %v, want 1 false false false",
			r.Schema, r.SnapshotAtomic, r.InferenceProbed, r.PeerConnectivityProbed)
	}
	if !strings.Contains(text, "Inference and peer connectivity were not tested.") {
		t.Fatalf("text output lacks the no-probe statement:\n%s", text)
	}
}

func intPtr(v int) *int    { return &v }
func boolPtr(v bool) *bool { return &v }

func knownService(brokerRestarts int) diagnoseService {
	return diagnoseService{Status: diagnoseKnown, PID: intPtr(4242), BrokerPID: intPtr(4243),
		BrokerRestarts: intPtr(brokerRestarts), Version: "0.2.0", BuildID: buildinfo.ID(testBuildID)}
}

func facades(status diagnoseStatus) []diagnoseFacade {
	var out []diagnoseFacade
	for _, e := range engines.All() {
		out = append(out, diagnoseFacade{Engine: e.Name, Status: status})
	}
	return out
}

func boundFacades() []diagnoseFacade {
	out := facades(diagnoseKnown)
	ports := []int{11434, 1234, 8080}
	for i := range out {
		out[i].Ready, out[i].Port = boolPtr(true), ports[i%len(ports)]
	}
	return out
}

func healthyReport() diagnoseReport {
	return diagnoseReport{
		Schema: 1, Complete: true,
		Service: knownService(0),
		Policy: diagnosePolicy{Status: diagnoseKnown, PersistedRevision: "1792285135482000000",
			ActiveRevision: "1792285135482000000", Activation: "active", Availability: "available", RecoveryRequired: boolPtr(false)},
		BuiltinFacades: boundFacades(),
		ResidentConfiguration: diagnoseResidentConfig{Status: diagnoseKnown,
			StateCounts:  map[nodepolicy.ModelConfigurationState]int{nodepolicy.ModelStateSatisfied: 1, nodepolicy.ModelStateBlocked: 1},
			ReasonCounts: map[nodepolicy.ModelConfigurationReason]int{nodepolicy.ModelReasonNoFit: 1}},
	}
}

func assertReport(t *testing.T, got diagnosed, want diagnoseReport, wantCode int) {
	t.Helper()
	if got.code != wantCode {
		t.Fatalf("exit code = %d, want %d\n%s", got.code, wantCode, got.json)
	}
	if !reflect.DeepEqual(got.report, want) {
		t.Fatalf("report mismatch\ngot:  %+v\nwant: %+v\njson:\n%s", got.report, want, got.json)
	}
}

func assertCalls(t *testing.T, got, want []diagnoseCall) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("calls = %v, want %v", got, want)
	}
}

func TestDiagnoseCompleteCapture(t *testing.T) {
	f := newHealthyDiagnoseService(t)
	got := runDiagnoseCapture(t, context.Background(), f.call)
	assertReport(t, got, healthyReport(), diagnoseExitComplete)
	assertCalls(t, f.calls, wantDiagnoseSequence())
	for _, want := range []string{
		"Service: known",
		"pid 4242, broker pid 4243, broker restarts 0",
		"Ollama: known, bound on port 11434",
		"states: satisfied 1, blocked 1",
		"reasons: no-fit 1",
		"Capture: complete.",
	} {
		if !strings.Contains(got.text, want) {
			t.Fatalf("text output lacks %q:\n%s", want, got.text)
		}
	}
}

func TestDiagnoseKnownOperatingStates(t *testing.T) {
	test := func(name string, edit func(*fakeDiagnoseService, *policyFixture), want func(*diagnoseReport)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			f := newHealthyDiagnoseService(t)
			p := healthyPolicy()
			edit(f, &p)
			f.answers["policy:get"] = fixed(policyGetResult(t, p))
			report := healthyReport()
			want(&report)
			got := runDiagnoseCapture(t, context.Background(), f.call)
			assertReport(t, got, report, diagnoseExitComplete)
		})
	}
	unknown := false
	test("paused node with unbound facades", func(f *fakeDiagnoseService, p *policyFixture) {
		p.availability = "paused"
		p.models = []nodepolicy.ModelConfigurationEntry{
			{Engine: engines.NameOllama, Model: sentinelModel, DesiredResident: true, State: nodepolicy.ModelStateBlocked, Reason: nodepolicy.ModelReasonPausedLocal},
			{Engine: engines.NameLlamaCPP, Model: sentinelModel, DesiredResident: true, ObservedLoaded: &unknown, State: nodepolicy.ModelStateUnknown, Reason: nodepolicy.ModelReasonObservationUnknown},
		}
		for _, e := range engines.All() {
			f.answers[e.ComponentName()+":get-status"] = fixed(facadeResult(t, false, 0))
		}
	}, func(r *diagnoseReport) {
		r.Policy.Availability = "paused"
		r.BuiltinFacades = facades(diagnoseKnown)
		for i := range r.BuiltinFacades {
			r.BuiltinFacades[i].Ready = boolPtr(false)
		}
		r.ResidentConfiguration.StateCounts = map[nodepolicy.ModelConfigurationState]int{nodepolicy.ModelStateBlocked: 1, nodepolicy.ModelStateUnknown: 1}
		r.ResidentConfiguration.ReasonCounts = map[nodepolicy.ModelConfigurationReason]int{nodepolicy.ModelReasonPausedLocal: 1, nodepolicy.ModelReasonObservationUnknown: 1}
	})
	test("pending activation", func(_ *fakeDiagnoseService, p *policyFixture) {
		p.activeRevision, p.activation = "1792285135481000000", "pending"
	}, func(r *diagnoseReport) {
		r.Policy.ActiveRevision, r.Policy.Activation = "1792285135481000000", "pending"
	})
	test("activation error without an active revision", func(_ *fakeDiagnoseService, p *policyFixture) {
		p.activeRevision, p.activation = "", "error"
	}, func(r *diagnoseReport) {
		r.Policy.ActiveRevision, r.Policy.Activation = "", "error"
	})
	test("recovery policy", func(_ *fakeDiagnoseService, p *policyFixture) {
		p.recovery = true
	}, func(r *diagnoseReport) {
		r.Policy.RecoveryRequired = boolPtr(true)
	})
	test("no resident models", func(_ *fakeDiagnoseService, p *policyFixture) {
		p.models = []nodepolicy.ModelConfigurationEntry{}
	}, func(r *diagnoseReport) {
		r.ResidentConfiguration.StateCounts = map[nodepolicy.ModelConfigurationState]int{}
		r.ResidentConfiguration.ReasonCounts = map[nodepolicy.ModelConfigurationReason]int{}
	})
	test("relocated facade port", func(f *fakeDiagnoseService, _ *policyFixture) {
		f.answers[facadeMethod(t, engines.NameOllama)] = fixed(facadeResult(t, true, 11500))
	}, func(r *diagnoseReport) {
		r.BuiltinFacades[0].Port = 11500
	})
}

func TestDiagnoseNotRunningStopsAfterFirstDial(t *testing.T) {
	f := newHealthyDiagnoseService(t)
	f.answers[methodStatus] = failing(fmt.Errorf("%w: %s", servicectl.ErrNotRunning, sentinelPath))
	got := runDiagnoseCapture(t, context.Background(), f.call)
	want := diagnoseReport{Schema: 1,
		Service:               diagnoseService{Status: diagnoseNotRunning},
		Policy:                diagnosePolicy{Status: diagnoseNotRunning},
		BuiltinFacades:        facades(diagnoseNotRunning),
		ResidentConfiguration: diagnoseResidentConfig{Status: diagnoseNotRunning},
	}
	assertReport(t, got, want, diagnoseExitNotRunning)
	assertCalls(t, f.calls, []diagnoseCall{{methodStatus, ""}})
}

func TestDiagnoseServiceLostMidCaptureIsIncomplete(t *testing.T) {
	f := newHealthyDiagnoseService(t)
	gone := fmt.Errorf("%w: %s", servicectl.ErrNotRunning, sentinelPath)
	f.answers["policy:get"] = failing(gone)
	f.answers[methodStatus] = func(_ context.Context, occurrence int) (json.RawMessage, error) {
		if occurrence == 0 {
			return serviceStatusResult(t, 0), nil
		}
		return nil, gone
	}
	for _, e := range engines.All() {
		f.answers[e.ComponentName()+":get-status"] = failing(gone)
	}
	got := runDiagnoseCapture(t, context.Background(), f.call)
	want := diagnoseReport{Schema: 1,
		Service:               knownService(0),
		Policy:                diagnosePolicy{Status: diagnoseUnavailable},
		BuiltinFacades:        facades(diagnoseUnavailable),
		ResidentConfiguration: diagnoseResidentConfig{Status: diagnoseUnavailable},
	}
	assertReport(t, got, want, diagnoseExitIncomplete)
	assertCalls(t, f.calls, wantDiagnoseSequence())
	if !strings.Contains(got.text, "Capture: incomplete.") {
		t.Fatalf("text output does not label the capture incomplete:\n%s", got.text)
	}
}

func TestDiagnoseRejectsMalformedValues(t *testing.T) {
	test := func(name, method, result string, want func(*diagnoseReport)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			f := newHealthyDiagnoseService(t)
			f.answers[method] = fixed(json.RawMessage(result))
			report := healthyReport()
			report.Complete = false
			want(&report)
			got := runDiagnoseCapture(t, context.Background(), f.call)
			assertReport(t, got, report, diagnoseExitIncomplete)
			assertCalls(t, f.calls, wantDiagnoseSequence())
		})
	}
	invalidService := func(r *diagnoseReport) { r.Service = diagnoseService{Status: diagnoseInvalid} }
	test("service status without a result", methodStatus, ``, invalidService)
	test("service status missing pid", methodStatus, `{"brokerPid":2,"brokerRestarts":0,"version":"0.2.0"}`, invalidService)
	test("service status pid of the wrong type", methodStatus, `{"pid":"4242","brokerPid":2,"brokerRestarts":0,"version":"0.2.0"}`, invalidService)
	test("service status negative restarts", methodStatus, `{"pid":1,"brokerPid":2,"brokerRestarts":-1,"version":"0.2.0"}`, invalidService)
	test("service status null", methodStatus, `null`, invalidService)

	invalidResident := func(r *diagnoseReport) { r.ResidentConfiguration = diagnoseResidentConfig{Status: diagnoseInvalid} }
	invalidBoth := func(r *diagnoseReport) {
		r.Policy = diagnosePolicy{Status: diagnoseInvalid}
		invalidResident(r)
	}
	// The policy cases below carry a valid, empty resident configuration.
	models := `"modelConfiguration":{"policyRevision":"5","observationGeneration":1,"models":[]}`
	invalidPolicy := func(r *diagnoseReport) {
		r.Policy = diagnosePolicy{Status: diagnoseInvalid}
		r.ResidentConfiguration.StateCounts = map[nodepolicy.ModelConfigurationState]int{}
		r.ResidentConfiguration.ReasonCounts = map[nodepolicy.ModelConfigurationReason]int{}
	}
	test("policy missing recovery flag", "policy:get",
		`{"policy":{"revision":"5"},"availability":"available","activeRevision":"5","activation":"active",`+models+`}`, invalidPolicy)
	test("policy unsupported activation", "policy:get",
		`{"policy":{"revision":"5"},"availability":"available","recoveryRequired":false,"activeRevision":"5","activation":"activating",`+models+`}`, invalidPolicy)
	test("policy unsupported availability", "policy:get",
		`{"policy":{"revision":"5"},"availability":"sleeping","recoveryRequired":false,"activeRevision":"5","activation":"active",`+models+`}`, invalidPolicy)
	test("policy non-canonical revision", "policy:get",
		`{"policy":{"revision":"007"},"availability":"available","recoveryRequired":false,"activeRevision":"5","activation":"active",`+models+`}`, invalidPolicy)
	test("policy recovery flag of the wrong type", "policy:get",
		`{"policy":{"revision":"5"},"availability":"available","recoveryRequired":"no","activeRevision":"5","activation":"active",`+models+`}`, invalidPolicy)
	test("policy result is not an object", "policy:get", `["`+sentinelUnknown+`"]`, invalidBoth)

	policy := `"policy":{"revision":"5"},"availability":"available","recoveryRequired":false,"activeRevision":"5","activation":"active"`
	knownPolicy := func(r *diagnoseReport) {
		r.Policy.PersistedRevision, r.Policy.ActiveRevision = "5", "5"
	}
	test("resident configuration missing", "policy:get", `{`+policy+`}`,
		func(r *diagnoseReport) { knownPolicy(r); invalidResident(r) })
	test("resident unsupported state", "policy:get",
		`{`+policy+`,"modelConfiguration":{"policyRevision":"5","observationGeneration":1,"models":[{"engine":"ollama","model":"`+sentinelModel+`","desiredResident":true,"state":"warming"}]}}`,
		func(r *diagnoseReport) { knownPolicy(r); invalidResident(r) })
	test("resident unsupported reason", "policy:get",
		`{`+policy+`,"modelConfiguration":{"policyRevision":"5","observationGeneration":1,"models":[{"engine":"ollama","model":"`+sentinelModel+`","desiredResident":true,"state":"blocked","reason":"`+sentinelError+`"}]}}`,
		func(r *diagnoseReport) { knownPolicy(r); invalidResident(r) })

	invalidOllama := func(r *diagnoseReport) {
		r.BuiltinFacades[0] = diagnoseFacade{Engine: engines.NameOllama, Status: diagnoseInvalid}
	}
	ollama := facadeMethod(t, engines.NameOllama)
	test("facade missing ready", ollama, `{"port":11434}`, invalidOllama)
	test("facade missing port", ollama, `{"ready":false}`, invalidOllama)
	test("facade port out of range", ollama, `{"ready":true,"port":70000}`, invalidOllama)
	test("ready facade without a port", ollama, `{"ready":true,"port":0}`, invalidOllama)
	test("unbound facade with a port", ollama, `{"ready":false,"port":11434}`, invalidOllama)
	test("facade ready of the wrong type", ollama, `{"ready":"yes","port":11434}`, invalidOllama)
}

func TestDiagnoseDiscardsUnknownFields(t *testing.T) {
	f := newHealthyDiagnoseService(t)
	f.answers[methodStatus] = fixed(json.RawMessage(`{"pid":4242,"brokerPid":4243,"clients":1,"brokerRestarts":0,` +
		`"version":"0.2.0","serviceBuildId":"` + testBuildID + `","futureField":"` + sentinelUnknown + `"}`))
	f.answers[facadeMethod(t, engines.NameOllama)] = fixed(json.RawMessage(`{"ready":true,"port":11434,"address":"` + sentinelHost + `"}`))
	got := runDiagnoseCapture(t, context.Background(), f.call)
	assertReport(t, got, healthyReport(), diagnoseExitComplete)
}

func TestDiagnoseKeepsUnsafeMetadataUnknown(t *testing.T) {
	f := newHealthyDiagnoseService(t)
	f.answers[methodStatus] = fixed(json.RawMessage(`{"pid":4242,"brokerPid":4243,"brokerRestarts":0,` +
		`"version":"0.2.0 \u001b[31m` + sentinelPath + `","serviceBuildId":"` + sentinelError + `"}`))
	got := runDiagnoseCapture(t, context.Background(), f.call)
	want := healthyReport()
	want.Service.Version, want.Service.BuildID = "", ""
	assertReport(t, got, want, diagnoseExitComplete)
	if !strings.Contains(got.text, "version unknown, build unknown") {
		t.Fatalf("text output does not mark the metadata unknown:\n%s", got.text)
	}
}

func TestDiagnoseReportsReadErrorsWithoutTheirText(t *testing.T) {
	f := newHealthyDiagnoseService(t)
	delete(f.answers, facadeMethod(t, engines.NameLMStudio))
	got := runDiagnoseCapture(t, context.Background(), f.call)
	want := healthyReport()
	want.Complete = false
	want.BuiltinFacades[1] = diagnoseFacade{Engine: engines.NameLMStudio, Status: diagnoseUnavailable}
	assertReport(t, got, want, diagnoseExitIncomplete)
}

func TestDiagnoseMarksChangesDuringCapture(t *testing.T) {
	test := func(name string, edit func(*fakeDiagnoseService), want func(*diagnoseReport)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			f := newHealthyDiagnoseService(t)
			edit(f)
			report := healthyReport()
			report.Complete = false
			want(&report)
			got := runDiagnoseCapture(t, context.Background(), f.call)
			assertReport(t, got, report, diagnoseExitIncomplete)
			assertCalls(t, f.calls, wantDiagnoseSequence())
		})
	}
	policyAfter := func(edit func(*policyFixture)) json.RawMessage {
		p := healthyPolicy()
		edit(&p)
		return policyGetResult(t, p)
	}
	test("broker restarted", func(f *fakeDiagnoseService) {
		f.answers[methodStatus] = changing(serviceStatusResult(t, 0), serviceStatusResult(t, 1))
	}, func(r *diagnoseReport) {
		r.Service = diagnoseService{Status: diagnoseChanged}
		r.Policy = diagnosePolicy{Status: diagnoseChanged}
		r.BuiltinFacades = facades(diagnoseChanged)
		r.ResidentConfiguration = diagnoseResidentConfig{Status: diagnoseChanged}
	})
	test("policy revision replaced", func(f *fakeDiagnoseService) {
		f.answers["policy:get"] = changing(policyGetResult(t, healthyPolicy()), policyAfter(func(p *policyFixture) { p.revision = "1792285135483000000" }))
	}, func(r *diagnoseReport) {
		r.Policy = diagnosePolicy{Status: diagnoseChanged}
		r.ResidentConfiguration = diagnoseResidentConfig{Status: diagnoseChanged}
	})
	test("activation settled", func(f *fakeDiagnoseService) {
		pending := policyAfter(func(p *policyFixture) { p.activation = "pending" })
		f.answers["policy:get"] = changing(pending, policyGetResult(t, healthyPolicy()))
	}, func(r *diagnoseReport) {
		r.Policy = diagnosePolicy{Status: diagnoseChanged}
	})
	test("resident observation advanced", func(f *fakeDiagnoseService) {
		f.answers["policy:get"] = changing(policyGetResult(t, healthyPolicy()), policyAfter(func(p *policyFixture) { p.generation++ }))
	}, func(r *diagnoseReport) {
		r.ResidentConfiguration = diagnoseResidentConfig{Status: diagnoseChanged}
	})
	test("final fence unreadable", func(f *fakeDiagnoseService) {
		f.answers[methodStatus] = func(_ context.Context, occurrence int) (json.RawMessage, error) {
			if occurrence == 0 {
				return serviceStatusResult(t, 0), nil
			}
			return nil, errors.New(sentinelError)
		}
	}, func(*diagnoseReport) {})
}

func TestDiagnoseStopsAtTheSharedDeadline(t *testing.T) {
	f := newHealthyDiagnoseService(t)
	var cancelled error
	f.answers[facadeMethod(t, engines.NameLMStudio)] = func(ctx context.Context, _ int) (json.RawMessage, error) {
		<-ctx.Done()
		cancelled = ctx.Err()
		return nil, fmt.Errorf("%s: %w", sentinelError, ctx.Err())
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	got := runDiagnoseCapture(t, ctx, f.call)
	if !errors.Is(cancelled, context.DeadlineExceeded) {
		t.Fatalf("blocked read ended with %v, want the shared deadline", cancelled)
	}
	want := healthyReport()
	want.Complete = false
	for i := 1; i < len(want.BuiltinFacades); i++ {
		want.BuiltinFacades[i] = diagnoseFacade{Engine: want.BuiltinFacades[i].Engine, Status: diagnoseUnavailable}
	}
	assertReport(t, got, want, diagnoseExitIncomplete)
	assertCalls(t, f.calls, wantDiagnoseSequence()[:4])
}

func TestDiagnoseUsageErrorsDoNotDial(t *testing.T) {
	fixture := newDiagnoseFixture(t, func(string) diagnoseReply { return diagnoseReply{drop: true} })
	test := func(name string, args ...string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := run(append([]string{"diagnose"}, args...), &stdout, &stderr)
			if code != diagnoseExitUsage {
				t.Fatalf("exit code = %d, want %d", code, diagnoseExitUsage)
			}
			if stdout.Len() != 0 || stderr.String() != "usage: "+diagnoseUsage+"\n" {
				t.Fatalf("stdout = %q, stderr = %q; want only the usage line on stderr", stdout.String(), stderr.String())
			}
		})
	}
	test("duplicate json flag", "--json", "--json")
	test("unknown flag", "--verbose")
	test("single-dash spelling", "-json")
	test("flag with a value", "--json=true")
	test("positional argument", "now")
	test("positional after json", "--json", "now")
	if calls := fixture.recorded(); len(calls) != 0 {
		t.Fatalf("usage errors dialed the service: %v", calls)
	}
}

// diagnoseReply is how the endpoint fixture answers one request: frames to
// write (the request id substituted for %s) or, with drop, a close without
// answering. No frames and no drop leaves the request unanswered until the
// client closes the connection.
type diagnoseReply struct {
	frames []string
	drop   bool
}

// diagnoseFixture listens on a private endpoint, standing in for the service
// without spawning anything: each connection carries one request, answered by
// a script.
type diagnoseFixture struct {
	t        *testing.T
	listener net.Listener
	reply    func(method string) diagnoseReply

	wg     sync.WaitGroup
	mu     sync.Mutex
	calls  []diagnoseCall
	conns  []net.Conn
	closed int
}

func newDiagnoseFixture(t *testing.T, reply func(method string) diagnoseReply) *diagnoseFixture {
	t.Helper()
	endpoint := testEndpoint(shortTempDir(t))
	t.Setenv(servicectl.EndpointEnv, endpoint)
	l, err := ipc.ListenPrivate(endpoint)
	if err != nil {
		t.Fatalf("listen on private endpoint: %v", err)
	}
	f := &diagnoseFixture{t: t, listener: l, reply: reply}
	f.wg.Add(1)
	go f.accept()
	t.Cleanup(f.close)
	return f
}

func (f *diagnoseFixture) accept() {
	defer f.wg.Done()
	for {
		conn, err := f.listener.Accept()
		if err != nil {
			return
		}
		f.mu.Lock()
		f.conns = append(f.conns, conn)
		f.mu.Unlock()
		f.wg.Add(1)
		go f.serve(conn)
	}
}

func (f *diagnoseFixture) serve(conn net.Conn) {
	defer f.wg.Done()
	defer conn.Close()
	r := bufio.NewReader(conn)
	line, err := r.ReadBytes('\n')
	if err != nil {
		return
	}
	var req jsonrpc.Message
	if err := json.Unmarshal(line, &req); err != nil || req.ID == nil {
		f.record(diagnoseCall{method: "unparsable request"})
		return
	}
	f.record(diagnoseCall{method: req.Method, params: string(req.Params)})
	reply := f.reply(req.Method)
	if reply.drop {
		return
	}
	for _, frame := range reply.frames {
		if strings.Contains(frame, "%s") {
			frame = fmt.Sprintf(frame, string(*req.ID))
		}
		if _, err := io.WriteString(conn, frame+"\n"); err != nil {
			break
		}
	}
	// Wait for the client to close its end; that is the teardown evidence.
	_, _ = io.Copy(io.Discard, r)
	f.mu.Lock()
	f.closed++
	f.mu.Unlock()
}

func (f *diagnoseFixture) record(c diagnoseCall) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, c)
}

func (f *diagnoseFixture) recorded() []diagnoseCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]diagnoseCall(nil), f.calls...)
}

func (f *diagnoseFixture) closedByClient() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.closed
}

// close stops accepting, closes every connection a failed test left open and
// waits for the fixture's goroutines.
func (f *diagnoseFixture) close() {
	_ = f.listener.Close()
	f.mu.Lock()
	for _, c := range f.conns {
		_ = c.Close()
	}
	f.mu.Unlock()
	done := make(chan struct{})
	go func() {
		f.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(waitTimeout):
		f.t.Error("diagnose fixture goroutines did not stop")
	}
}

// healthyReplies answers as a running service does, each response preceded
// by notifications and a reply to another request id.
func healthyReplies(t *testing.T) func(method string) diagnoseReply {
	t.Helper()
	f := newHealthyDiagnoseService(t)
	return func(method string) diagnoseReply {
		answer, ok := f.answers[method]
		if !ok {
			return diagnoseReply{frames: []string{`{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"` + sentinelError + `"}}`}}
		}
		result, err := answer(context.Background(), 0)
		if err != nil {
			t.Errorf("fixture answer for %s: %v", method, err)
			return diagnoseReply{drop: true}
		}
		return diagnoseReply{frames: []string{
			`{"jsonrpc":"2.0","method":"app:ready","params":{"version":"` + sentinelNotice + `"}}`,
			`{"jsonrpc":"2.0","method":"service/log","params":{"text":"` + sentinelNotice + `"}}`,
			`{"jsonrpc":"2.0","id":99,"result":{"text":"` + sentinelNotice + `"}}`,
			`{"jsonrpc":"2.0","id":%s,"result":` + string(result) + `}`,
		}}
	}
}

func waitClosedByClient(t *testing.T, f *diagnoseFixture, want int) {
	t.Helper()
	waitUntil(t, "client connections closed", func() bool { return f.closedByClient() == want })
}

func TestDiagnoseOverPrivateEndpoint(t *testing.T) {
	fixture := newDiagnoseFixture(t, healthyReplies(t))
	var stdout, stderr bytes.Buffer
	code := run([]string{"diagnose", "--json"}, &stdout, &stderr)
	if code != diagnoseExitComplete {
		t.Fatalf("exit code = %d, want %d; stdout:\n%s", code, diagnoseExitComplete, stdout.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
	assertNoSentinels(t, stdout.String())
	var report diagnoseReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode JSON report: %v", err)
	}
	if !reflect.DeepEqual(report, healthyReport()) {
		t.Fatalf("report = %+v, want %+v", report, healthyReport())
	}
	assertCalls(t, fixture.recorded(), wantDiagnoseSequence())
	waitClosedByClient(t, fixture, len(wantDiagnoseSequence()))
}

func TestDiagnoseTextOverPrivateEndpoint(t *testing.T) {
	newDiagnoseFixture(t, healthyReplies(t))
	var stdout, stderr bytes.Buffer
	code := run([]string{"diagnose"}, &stdout, &stderr)
	if code != diagnoseExitComplete || stderr.Len() != 0 {
		t.Fatalf("exit code = %d, stderr = %q; want %d and no stderr", code, stderr.String(), diagnoseExitComplete)
	}
	assertNoSentinels(t, stdout.String())
	for _, want := range []string{"Capture: complete.", "Inference and peer connectivity were not tested."} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("text output lacks %q:\n%s", want, stdout.String())
		}
	}
}

func TestDiagnoseNotRunningAtPrivateEndpoint(t *testing.T) {
	// An endpoint nothing listens on.
	t.Setenv(servicectl.EndpointEnv, testEndpoint(shortTempDir(t)))
	var stdout, stderr bytes.Buffer
	code := run([]string{"diagnose", "--json"}, &stdout, &stderr)
	if code != diagnoseExitNotRunning || stderr.Len() != 0 {
		t.Fatalf("exit code = %d, stderr = %q; want %d and no stderr", code, stderr.String(), diagnoseExitNotRunning)
	}
	var report diagnoseReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode JSON report: %v", err)
	}
	if report.Complete || report.Service.Status != diagnoseNotRunning || report.Policy.Status != diagnoseNotRunning {
		t.Fatalf("report = %+v, want a not-running report", report)
	}
}

func TestDiagnoseEndpointFailures(t *testing.T) {
	test := func(name, method string, reply diagnoseReply, want func(*diagnoseReport)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			healthy := healthyReplies(t)
			fixture := newDiagnoseFixture(t, func(m string) diagnoseReply {
				if m == method {
					return reply
				}
				return healthy(m)
			})
			var stdout, stderr bytes.Buffer
			code := run([]string{"diagnose", "--json"}, &stdout, &stderr)
			if code != diagnoseExitIncomplete || stderr.Len() != 0 {
				t.Fatalf("exit code = %d, stderr = %q; want %d and no stderr", code, stderr.String(), diagnoseExitIncomplete)
			}
			assertNoSentinels(t, stdout.String())
			var report diagnoseReport
			if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
				t.Fatalf("decode JSON report: %v", err)
			}
			wantReport := healthyReport()
			wantReport.Complete = false
			want(&wantReport)
			if !reflect.DeepEqual(report, wantReport) {
				t.Fatalf("report = %+v, want %+v", report, wantReport)
			}
			assertCalls(t, fixture.recorded(), wantDiagnoseSequence())
		})
	}
	oversized := `{"jsonrpc":"2.0","id":%s,"result":{"pad":"` + strings.Repeat("x", jsonrpc.WorkerFrameBytes) + `"}}`
	test("oversized status frame", methodStatus, diagnoseReply{frames: []string{oversized}}, func(r *diagnoseReport) {
		r.Service = diagnoseService{Status: diagnoseInvalid}
	})
	test("connection dropped before the policy answer", "policy:get", diagnoseReply{drop: true}, func(r *diagnoseReport) {
		r.Policy = diagnosePolicy{Status: diagnoseUnavailable}
		r.ResidentConfiguration = diagnoseResidentConfig{Status: diagnoseUnavailable}
	})
	test("error answer for a facade", facadeMethod(t, engines.NameLlamaCPP),
		diagnoseReply{frames: []string{`{"jsonrpc":"2.0","id":%s,"error":{"code":-32000,"message":"` + sentinelError + `"}}`}},
		func(r *diagnoseReport) {
			r.BuiltinFacades[2] = diagnoseFacade{Engine: engines.NameLlamaCPP, Status: diagnoseUnavailable}
		})
}

func TestDiagnoseContextEndClosesTheSlowConnection(t *testing.T) {
	healthy := healthyReplies(t)
	slow := facadeMethod(t, engines.NameOllama)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	fixture := newDiagnoseFixture(t, func(m string) diagnoseReply {
		if m == slow {
			// Leave the request unanswered and end the capture's context,
			// as its deadline would.
			cancel()
			return diagnoseReply{}
		}
		return healthy(m)
	})
	var stdout bytes.Buffer
	code := diagnose(ctx, callRunningServiceWith, true, &stdout)
	if code != diagnoseExitIncomplete {
		t.Fatalf("exit code = %d, want %d", code, diagnoseExitIncomplete)
	}
	if !errors.Is(ctx.Err(), context.Canceled) {
		t.Fatalf("capture returned before its context ended: %v", ctx.Err())
	}
	var report diagnoseReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode JSON report: %v", err)
	}
	for _, f := range report.BuiltinFacades {
		if f.Status != diagnoseUnavailable {
			t.Fatalf("facade %s status = %s, want unavailable after the deadline", f.Engine, f.Status)
		}
	}
	assertCalls(t, fixture.recorded(), wantDiagnoseSequence()[:3])
	// Both answered connections and the unanswered one were closed by the client.
	waitClosedByClient(t, fixture, 3)
}
