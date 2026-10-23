// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"
	"time"

	"nvpair-shared/buildinfo"
	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/servicectl"
)

// `nvpair-service diagnose` reports cached local facts of the running service:
// its own status, the local policy's activation, the built-in engine facades'
// bind state and the resident model configuration counts. It only reads
// service/status, the local policy:get and each built-in facade's get-status
// through the dial-only service connection. It never starts the service, sends
// inference, wakes an engine or contacts a peer, and it prints only validated,
// allowlisted fields: never the policy document, error text, model identifiers,
// paths or addresses.

const diagnoseUsage = "nvpair-service diagnose [--json]"

// diagnoseTimeout bounds the whole capture, as runStatus bounds its one read.
// There are no per-read timeouts that could add up past it.
const diagnoseTimeout = 5 * time.Second

// diagnoseSchema versions the --json report. The report is this command's
// output, not a broker wire.
const diagnoseSchema = 1

// Exit codes: a complete capture, an incomplete one, bad arguments, and no
// service answering the first dial. None of them labels an engine healthy.
const (
	diagnoseExitComplete   = 0
	diagnoseExitIncomplete = 1
	diagnoseExitUsage      = 2
	diagnoseExitNotRunning = 3
)

// diagnoseStatus says how far a report section is established.
type diagnoseStatus string

const (
	diagnoseKnown       diagnoseStatus = "known"
	diagnoseNotRunning  diagnoseStatus = "not-running"
	diagnoseUnavailable diagnoseStatus = "unavailable"
	diagnoseInvalid     diagnoseStatus = "invalid"
	diagnoseChanged     diagnoseStatus = "changed-during-capture"
)

// The policy:get activation values. The broker owns them; this command only
// accepts the ones it sends.
var diagnoseActivations = []string{"active", "pending", "error"}

// policyGetParams asks for this node's own policy: no nodeId, so the broker
// answers from its local state and never relays to a peer.
var policyGetParams = json.RawMessage(`{}`)

// diagnoseReport is the typed report both output formats render. Optional
// members are present only while their section is known.
type diagnoseReport struct {
	Schema                 int                    `json:"schema"`
	Complete               bool                   `json:"complete"`
	SnapshotAtomic         bool                   `json:"snapshotAtomic"`
	InferenceProbed        bool                   `json:"inferenceProbed"`
	PeerConnectivityProbed bool                   `json:"peerConnectivityProbed"`
	Service                diagnoseService        `json:"service"`
	Policy                 diagnosePolicy         `json:"policy"`
	BuiltinFacades         []diagnoseFacade       `json:"builtinFacades"`
	ResidentConfiguration  diagnoseResidentConfig `json:"residentConfiguration"`
}

type diagnoseService struct {
	Status         diagnoseStatus `json:"status"`
	PID            *int           `json:"pid,omitempty"`
	BrokerPID      *int           `json:"brokerPid,omitempty"`
	BrokerRestarts *int           `json:"brokerRestarts,omitempty"`
	Version        string         `json:"version,omitempty"`
	BuildID        buildinfo.ID   `json:"buildId,omitempty"`
}

type diagnosePolicy struct {
	Status            diagnoseStatus `json:"status"`
	PersistedRevision string         `json:"persistedRevision,omitempty"`
	ActiveRevision    string         `json:"activeRevision,omitempty"`
	Activation        string         `json:"activation,omitempty"`
	Availability      string         `json:"availability,omitempty"`
	RecoveryRequired  *bool          `json:"recoveryRequired,omitempty"`
}

// diagnoseFacade is one built-in facade's cached bind state. Ready means the
// proxy reported the facade bound on Port; it says nothing about whether an
// engine or model behind it can answer.
type diagnoseFacade struct {
	Engine string         `json:"engine"`
	Status diagnoseStatus `json:"status"`
	Ready  *bool          `json:"ready,omitempty"`
	Port   int            `json:"port,omitempty"`
}

// diagnoseResidentConfig counts the accepted modelConfiguration entries by
// state and reason, without their engine or model identifiers.
type diagnoseResidentConfig struct {
	Status       diagnoseStatus                              `json:"status"`
	StateCounts  map[nodepolicy.ModelConfigurationState]int  `json:"stateCounts,omitzero"`
	ReasonCounts map[nodepolicy.ModelConfigurationReason]int `json:"reasonCounts,omitzero"`
}

// diagnoseCaller sends one request to the running service and returns its
// result. Production uses callDiagnoseService, which dials without starting
// anything and closes its connection when the answer or the deadline arrives.
type diagnoseCaller func(ctx context.Context, method string, params json.RawMessage) (json.RawMessage, error)

func runDiagnose(args []string, stdout, stderr io.Writer) int {
	asJSON, ok := parseDiagnoseArgs(args)
	if !ok {
		fmt.Fprintln(stderr, "usage: "+diagnoseUsage)
		return diagnoseExitUsage
	}
	ctx, cancel := context.WithTimeout(context.Background(), diagnoseTimeout)
	defer cancel()
	return diagnose(ctx, callDiagnoseService, asJSON, stdout)
}

// parseDiagnoseArgs accepts nothing or a single --json.
func parseDiagnoseArgs(args []string) (asJSON, ok bool) {
	switch {
	case len(args) == 0:
		return false, true
	case len(args) == 1 && args[0] == "--json":
		return true, true
	}
	return false, false
}

func diagnose(ctx context.Context, call diagnoseCaller, asJSON bool, stdout io.Writer) int {
	report, code := captureDiagnosis(ctx, call)
	if asJSON {
		writeDiagnoseJSON(stdout, report)
	} else {
		writeDiagnoseText(stdout, report)
	}
	return code
}

// serviceFacts is the comparable part of service/status.
type serviceFacts struct {
	pid, brokerPID, brokerRestarts int
	version                        string
	buildID                        buildinfo.ID
}

// policyFacts is the allowlisted part of policy:get.
type policyFacts struct {
	persistedRevision, activeRevision, activation, availability string
	recoveryRequired                                            bool
}

// residentFacts is the modelConfiguration fence and its aggregate counts.
type residentFacts struct {
	revision     string
	generation   uint64
	stateCounts  map[nodepolicy.ModelConfigurationState]int
	reasonCounts map[nodepolicy.ModelConfigurationReason]int
}

// captureDiagnosis runs the fixed read sequence under one deadline: service
// status, local policy, each built-in facade, then policy and service status
// again as fences. It does not retry; a fence that moved marks the dependent
// sections changed-during-capture.
func captureDiagnosis(ctx context.Context, call diagnoseCaller) (diagnoseReport, int) {
	builtins := engines.All()
	report := diagnoseReport{Schema: diagnoseSchema, BuiltinFacades: make([]diagnoseFacade, len(builtins))}
	for i, e := range builtins {
		report.BuiltinFacades[i] = diagnoseFacade{Engine: e.Name, Status: diagnoseUnavailable}
	}

	firstService, serviceStatus := readServiceStatus(ctx, call)
	if serviceStatus == diagnoseNotRunning {
		report.Service.Status = diagnoseNotRunning
		report.Policy.Status = diagnoseNotRunning
		report.ResidentConfiguration.Status = diagnoseNotRunning
		for i := range report.BuiltinFacades {
			report.BuiltinFacades[i].Status = diagnoseNotRunning
		}
		return report, diagnoseExitNotRunning
	}
	firstPolicy, policyStatus, firstResident, residentStatus := readLocalPolicy(ctx, call, false)
	for i, e := range builtins {
		report.BuiltinFacades[i] = readFacade(ctx, call, e)
	}
	lastPolicy, lastPolicyStatus, lastResident, lastResidentStatus := readLocalPolicy(ctx, call, true)
	lastService, lastServiceStatus := readServiceStatus(ctx, call)

	fencesHeld := true
	settle := func(status diagnoseStatus, lastStatus diagnoseStatus, same bool) diagnoseStatus {
		if status != diagnoseKnown {
			return status
		}
		if lastStatus != diagnoseKnown {
			fencesHeld = false
			return lastStatus
		}
		if !same {
			return diagnoseChanged
		}
		return status
	}
	if lastServiceStatus == diagnoseKnown {
		serviceStatus = settle(serviceStatus, lastServiceStatus, firstService == lastService)
	} else {
		fencesHeld = false
	}
	policyStatus = settle(policyStatus, lastPolicyStatus, firstPolicy == lastPolicy)
	residentStatus = settle(residentStatus, lastResidentStatus,
		firstResident.revision == lastResident.revision && firstResident.generation == lastResident.generation)
	if policyStatus == diagnoseChanged {
		residentStatus = changedIfKnown(residentStatus)
	}
	if serviceStatus == diagnoseChanged {
		// Every other section came from a broker the service may have replaced.
		policyStatus = changedIfKnown(policyStatus)
		residentStatus = changedIfKnown(residentStatus)
		for i := range report.BuiltinFacades {
			if report.BuiltinFacades[i].Status == diagnoseKnown {
				report.BuiltinFacades[i] = diagnoseFacade{Engine: report.BuiltinFacades[i].Engine, Status: diagnoseChanged}
			}
		}
	}

	report.Service = serviceSection(serviceStatus, firstService)
	report.Policy = policySection(policyStatus, firstPolicy)
	report.ResidentConfiguration = residentSection(residentStatus, firstResident)
	report.Complete = fencesHeld && serviceStatus == diagnoseKnown && policyStatus == diagnoseKnown && residentStatus == diagnoseKnown
	for _, f := range report.BuiltinFacades {
		report.Complete = report.Complete && f.Status == diagnoseKnown
	}
	if !report.Complete {
		return report, diagnoseExitIncomplete
	}
	return report, diagnoseExitComplete
}

func changedIfKnown(status diagnoseStatus) diagnoseStatus {
	if status == diagnoseKnown {
		return diagnoseChanged
	}
	return status
}

// readResult performs one capture read and classifies a failure without
// keeping its text: an oversized or malformed reply is invalid, anything else unavailable.
// Only a refused dial while time remains is not-running.
func readResult(ctx context.Context, call diagnoseCaller, method string, params json.RawMessage) (json.RawMessage, diagnoseStatus) {
	if ctx.Err() != nil {
		return nil, diagnoseUnavailable
	}
	result, err := call(ctx, method, params)
	switch {
	case err == nil:
		return result, diagnoseKnown
	case errors.Is(err, servicectl.ErrNotRunning) && ctx.Err() == nil:
		return nil, diagnoseNotRunning
	case errors.Is(err, bufio.ErrTooLong), errors.Is(err, errInvalidServiceReply):
		return nil, diagnoseInvalid
	}
	return nil, diagnoseUnavailable
}

func readServiceStatus(ctx context.Context, call diagnoseCaller) (serviceFacts, diagnoseStatus) {
	result, status := readResult(ctx, call, methodStatus, nil)
	if status != diagnoseKnown {
		return serviceFacts{}, status
	}
	facts, err := decodeServiceStatus(result)
	if err != nil {
		return serviceFacts{}, diagnoseInvalid
	}
	return facts, diagnoseKnown
}

func decodeServiceStatus(result json.RawMessage) (serviceFacts, error) {
	var v struct {
		PID            *int         `json:"pid"`
		BrokerPID      *int         `json:"brokerPid"`
		BrokerRestarts *int         `json:"brokerRestarts"`
		Version        *string      `json:"version"`
		ServiceBuildID buildinfo.ID `json:"serviceBuildId"`
	}
	if err := json.Unmarshal(result, &v); err != nil {
		return serviceFacts{}, err
	}
	if v.PID == nil || *v.PID <= 0 || v.BrokerPID == nil || *v.BrokerPID < 0 ||
		v.BrokerRestarts == nil || *v.BrokerRestarts < 0 || v.Version == nil {
		return serviceFacts{}, errors.New("service/status is missing a required field")
	}
	facts := serviceFacts{pid: *v.PID, brokerPID: *v.BrokerPID, brokerRestarts: *v.BrokerRestarts, buildID: v.ServiceBuildID}
	// Like the build identity, a version that is not a plain token stays
	// unknown rather than reaching the terminal.
	if plainVersion(*v.Version) {
		facts.version = *v.Version
	}
	return facts, nil
}

// plainVersion accepts a short token of ASCII letters, digits and . _ + -.
func plainVersion(v string) bool {
	if v == "" || len(v) > 64 {
		return false
	}
	for _, c := range v {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || strings.ContainsRune("._+-", c)) {
			return false
		}
	}
	return true
}

// readLocalPolicy reads policy:get and projects it at once. The raw document,
// which can hold private routing configuration, does not outlive this call.
func readLocalPolicy(ctx context.Context, call diagnoseCaller, final bool) (policyFacts, diagnoseStatus, residentFacts, diagnoseStatus) {
	result, status := readResult(ctx, call, "policy:get", policyGetParams)
	if status != diagnoseKnown {
		return policyFacts{}, status, residentFacts{}, status
	}
	policy, policyStatus := policyFacts{}, diagnoseInvalid
	if p, err := decodePolicyFacts(result); err == nil {
		policy, policyStatus = p, diagnoseKnown
	}
	resident, residentStatus := residentFacts{}, diagnoseInvalid
	if r, err := decodeResidentFacts(result); err == nil {
		resident, residentStatus = r, diagnoseKnown
	}
	if final {
		var sections map[string]json.RawMessage
		if json.Unmarshal(result, &sections) == nil && sections != nil {
			if _, ok := sections["policy"]; !ok {
				policyStatus = diagnoseUnavailable
			}
			if _, ok := sections["modelConfiguration"]; !ok {
				residentStatus = diagnoseUnavailable
			}
		}
	}
	return policy, policyStatus, resident, residentStatus
}

func decodePolicyFacts(result json.RawMessage) (policyFacts, error) {
	var v struct {
		Policy *struct {
			Revision *string `json:"revision"`
		} `json:"policy"`
		ActiveRevision   *string `json:"activeRevision"`
		Activation       *string `json:"activation"`
		Availability     *string `json:"availability"`
		RecoveryRequired *bool   `json:"recoveryRequired"`
	}
	if err := json.Unmarshal(result, &v); err != nil {
		return policyFacts{}, err
	}
	if v.Policy == nil || v.Policy.Revision == nil || v.ActiveRevision == nil || v.Activation == nil ||
		v.Availability == nil || v.RecoveryRequired == nil {
		return policyFacts{}, errors.New("policy:get is missing a required field")
	}
	if _, err := nodepolicy.ParseNanoUSD(*v.Policy.Revision); err != nil {
		return policyFacts{}, err
	}
	if *v.ActiveRevision != "" {
		if _, err := nodepolicy.ParseNanoUSD(*v.ActiveRevision); err != nil {
			return policyFacts{}, err
		}
	}
	if !slices.Contains(diagnoseActivations, *v.Activation) {
		return policyFacts{}, errors.New("unknown activation")
	}
	switch nodepolicy.Availability(*v.Availability) {
	case nodepolicy.Available, nodepolicy.Draining, nodepolicy.Paused:
	default:
		return policyFacts{}, errors.New("unknown availability")
	}
	return policyFacts{persistedRevision: *v.Policy.Revision, activeRevision: *v.ActiveRevision,
		activation: *v.Activation, availability: *v.Availability, recoveryRequired: *v.RecoveryRequired}, nil
}

// decodeResidentFacts accepts modelConfiguration only through its own
// validating decoder, which also bounds the entry count, and keeps counts.
func decodeResidentFacts(result json.RawMessage) (residentFacts, error) {
	var v struct {
		ModelConfiguration *nodepolicy.ModelConfiguration `json:"modelConfiguration"`
	}
	if err := json.Unmarshal(result, &v); err != nil {
		return residentFacts{}, err
	}
	if v.ModelConfiguration == nil {
		return residentFacts{}, errors.New("policy:get is missing modelConfiguration")
	}
	facts := residentFacts{
		revision:     v.ModelConfiguration.PolicyRevision,
		generation:   v.ModelConfiguration.ObservationGeneration,
		stateCounts:  map[nodepolicy.ModelConfigurationState]int{},
		reasonCounts: map[nodepolicy.ModelConfigurationReason]int{},
	}
	for _, entry := range v.ModelConfiguration.Models {
		facts.stateCounts[entry.State]++
		if entry.Reason != "" {
			facts.reasonCounts[entry.Reason]++
		}
	}
	return facts, nil
}

func readFacade(ctx context.Context, call diagnoseCaller, e engines.Engine) diagnoseFacade {
	out := diagnoseFacade{Engine: e.Name}
	result, status := readResult(ctx, call, e.ComponentName()+":get-status", nil)
	if status == diagnoseNotRunning {
		status = diagnoseUnavailable
	}
	if status != diagnoseKnown {
		out.Status = status
		return out
	}
	var v struct {
		Ready *bool `json:"ready"`
		Port  *int  `json:"port"`
	}
	if json.Unmarshal(result, &v) != nil || v.Ready == nil || v.Port == nil {
		out.Status = diagnoseInvalid
		return out
	}
	// A ready facade names the port it is bound on; an unbound one reports 0.
	if *v.Ready && (*v.Port < 1 || *v.Port > 65535) || !*v.Ready && *v.Port != 0 {
		out.Status = diagnoseInvalid
		return out
	}
	out.Status, out.Ready, out.Port = diagnoseKnown, v.Ready, *v.Port
	return out
}

func serviceSection(status diagnoseStatus, f serviceFacts) diagnoseService {
	if status == diagnoseNotRunning {
		// A later read that found no service is an incomplete capture, not
		// the first-dial not-running result.
		status = diagnoseUnavailable
	}
	if status != diagnoseKnown {
		return diagnoseService{Status: status}
	}
	return diagnoseService{Status: status, PID: &f.pid, BrokerPID: &f.brokerPID, BrokerRestarts: &f.brokerRestarts,
		Version: f.version, BuildID: f.buildID}
}

func policySection(status diagnoseStatus, f policyFacts) diagnosePolicy {
	if status == diagnoseNotRunning {
		status = diagnoseUnavailable
	}
	if status != diagnoseKnown {
		return diagnosePolicy{Status: status}
	}
	return diagnosePolicy{Status: status, PersistedRevision: f.persistedRevision, ActiveRevision: f.activeRevision,
		Activation: f.activation, Availability: f.availability, RecoveryRequired: &f.recoveryRequired}
}

func residentSection(status diagnoseStatus, f residentFacts) diagnoseResidentConfig {
	if status == diagnoseNotRunning {
		status = diagnoseUnavailable
	}
	if status != diagnoseKnown {
		return diagnoseResidentConfig{Status: status}
	}
	return diagnoseResidentConfig{Status: status, StateCounts: f.stateCounts, ReasonCounts: f.reasonCounts}
}

func writeDiagnoseJSON(w io.Writer, report diagnoseReport) {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(report)
}

// residentStateOrder is the text output's state order.
var residentStateOrder = []nodepolicy.ModelConfigurationState{
	nodepolicy.ModelStateSatisfied, nodepolicy.ModelStatePending, nodepolicy.ModelStateBlocked, nodepolicy.ModelStateUnknown,
}

func writeDiagnoseText(w io.Writer, r diagnoseReport) {
	var b strings.Builder
	line := func(format string, a ...any) { fmt.Fprintf(&b, format+"\n", a...) }

	line("PAIR service diagnosis (cached local state)")
	line("Service: %s", r.Service.Status)
	if r.Service.Status == diagnoseKnown {
		line("  pid %d, broker pid %d, broker restarts %d", *r.Service.PID, *r.Service.BrokerPID, *r.Service.BrokerRestarts)
		line("  version %s, build %s", orUnknown(r.Service.Version), orUnknown(string(r.Service.BuildID)))
	}

	line("Policy: %s", r.Policy.Status)
	if r.Policy.Status == diagnoseKnown {
		line("  persisted revision %s, active revision %s, activation %s",
			r.Policy.PersistedRevision, orNone(r.Policy.ActiveRevision), r.Policy.Activation)
		recovery := "no"
		if *r.Policy.RecoveryRequired {
			recovery = "yes"
		}
		line("  availability %s, recovery required %s", r.Policy.Availability, recovery)
	}

	line("Built-in facades:")
	builtins := engines.All()
	for i, f := range r.BuiltinFacades {
		label := builtins[i].DisplayName
		switch {
		case f.Status != diagnoseKnown:
			line("  %s: %s", label, f.Status)
		case *f.Ready:
			line("  %s: known, bound on port %d", label, f.Port)
		default:
			line("  %s: known, not bound", label)
		}
	}

	line("Resident configuration: %s", r.ResidentConfiguration.Status)
	if r.ResidentConfiguration.Status == diagnoseKnown {
		var states []string
		for _, s := range residentStateOrder {
			if n := r.ResidentConfiguration.StateCounts[s]; n > 0 {
				states = append(states, fmt.Sprintf("%s %d", s, n))
			}
		}
		var reasons []string
		for _, reason := range slices.Sorted(maps.Keys(r.ResidentConfiguration.ReasonCounts)) {
			reasons = append(reasons, fmt.Sprintf("%s %d", reason, r.ResidentConfiguration.ReasonCounts[reason]))
		}
		line("  states: %s", orNone(strings.Join(states, ", ")))
		line("  reasons: %s", orNone(strings.Join(reasons, ", ")))
	}

	if r.Complete {
		line("Capture: complete. Not an atomic snapshot: the start and end checks show only that the compared fields did not change.")
	} else {
		line("Capture: incomplete.")
	}
	line("A bound facade port does not show that an engine or model can answer.")
	line("Inference and peer connectivity were not tested.")
	_, _ = io.WriteString(w, b.String())
}

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}

func orNone(s string) string {
	if s == "" {
		return "none"
	}
	return s
}
