// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"time"

	"nvpair-shared/buildapply"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/servicectl"
)

func (s *Service) handleBuildApply(c *client, msg *jsonrpc.Message) {
	var request any
	switch msg.Method {
	case "service/build-apply-preview":
		request = &buildapply.PreviewRequest{}
	case "service/build-apply-start":
		request = &buildapply.StartRequest{}
	case "service/build-apply-status":
		request = &buildapply.StatusRequest{}
	case "service/build-apply-gui-ready":
		request = &buildapply.GUIReadyRequest{}
	}
	if err := buildapply.DecodeRequest(msg.Params, request); err != nil {
		c.respondError(msg.ID, -32602, err.Error())
		return
	}
	if msg.Method == "service/build-apply-preview" {
		previewRequest, ok := request.(*buildapply.PreviewRequest)
		if !ok {
			c.respondError(msg.ID, -32602, "invalid preview request")
			return
		}
		s.respondBuildApplyPreview(c, msg.ID, *previewRequest)
	} else if r, ok := request.(*buildapply.StatusRequest); ok {
		c.respond(msg.ID, s.buildApplyStatus(*r))
	} else if r, ok := request.(*buildapply.StartRequest); ok {
		if !r.Consent.AllowEngineStop || !r.Consent.AllowInterruption {
			c.respond(msg.ID, s.startBuildApply(c, *r, localBuildApplyAdmission{s: s}))
			return
		}
		// Admission is independent of the requesting connection after capture.
		// The reader remains available to detect disconnect and deliver status.
		var id *json.RawMessage
		if msg.ID != nil {
			copied := append(json.RawMessage(nil), (*msg.ID)...)
			id = &copied
		}
		go func() { c.respond(id, s.startBuildApply(c, *r, localBuildApplyAdmission{s: s})) }()
	} else {
		if r, ok := request.(*buildapply.GUIReadyRequest); ok {
			receipt, err := readBuildApplyReceipt(r.OperationID)
			if err != nil || receipt.GUILaunch.Process == nil {
				c.respond(msg.ID, buildapply.Unavailable[buildapply.BuildApplyOperation]())
				return
			}
			var id *json.RawMessage
			if msg.ID != nil {
				copied := append(json.RawMessage(nil), (*msg.ID)...)
				id = &copied
			}
			go s.respondBuildApplyGUIReady(c, id, *r)
		}
	}
}

// publishBuildApplyChanged reuses the service's existing local-client transport.
func (s *Service) publishBuildApplyChanged(operation buildapply.BuildApplyOperation) {
	s.broadcast("service/build-apply-changed", operation)
}

func isBuildApplyMethod(method string) bool {
	switch method {
	case "service/build-apply-preview", "service/build-apply-start", "service/build-apply-status", "service/build-apply-gui-ready":
		return true
	default:
		return false
	}
}

// The admission slot belongs to this Service, not to a package global or a
// client. Invalidation of a disconnected preview cannot cancel accepted work.
type buildApplyAdmissionState struct {
	previewID string
	done      chan struct{}
	result    buildapply.Result[buildapply.BuildApplyOperation]
}

type buildApplyAdmissionOwner interface {
	Capture(context.Context, buildApplyAdmissionSnapshot, buildapply.StartRequest) (buildApplyPlan, error)
	ReadJournal() (buildapply.JournalState, error)
	Execute(context.Context, buildApplyPlan) buildapply.Result[buildapply.BuildApplyOperation]
}

type buildApplyAdmissionSnapshot struct {
	preview       buildapply.BuildApplyPreview
	gui           buildapply.ProcessIdentity
	candidatePath string
	targetPath    string
	targetSHA256  string
	expires       time.Time
}

type localBuildApplyAdmission struct{ s *Service }

func (o localBuildApplyAdmission) Capture(ctx context.Context, p buildApplyAdmissionSnapshot, r buildapply.StartRequest) (buildApplyPlan, error) {
	return o.s.captureBuildApplyPlan(ctx, p, r)
}
func (o localBuildApplyAdmission) ReadJournal() (buildapply.JournalState, error) {
	return buildapply.ReadJournal()
}

func (o localBuildApplyAdmission) Execute(ctx context.Context, p buildApplyPlan) buildapply.Result[buildapply.BuildApplyOperation] {
	return o.s.executeBuildApplyPlan(ctx, p)
}

func refusedBuildApply(code, detail string) buildapply.Result[buildapply.BuildApplyOperation] {
	return buildapply.Result[buildapply.BuildApplyOperation]{Refusal: &buildapply.Refusal{Code: code, Detail: detail}}
}

func (s *Service) startBuildApply(c *client, r buildapply.StartRequest, owner buildApplyAdmissionOwner) buildapply.Result[buildapply.BuildApplyOperation] {
	// Consent is checked before status reads, locks or even duplicate lookup.
	if !r.Consent.AllowEngineStop || !r.Consent.AllowInterruption {
		return refusedBuildApply("interruption_required", "Applying a whole build requires explicit interruption and engine-stop consent.")
	}
	state, err := owner.ReadJournal()
	if err != nil {
		return refusedBuildApply("recovery_required", "The existing journal cannot be trusted; ordinary startup and apply remain gated.")
	}
	recorded := state.Current
	if recorded == nil {
		recorded = state.LastTerminal
	}
	if recorded != nil && recorded.PreviewID == r.PreviewID {
		o := projectBuildApplyReceipt(*recorded)
		if err := recorded.ValidateOperation(o); err != nil {
			return buildapply.Unavailable[buildapply.BuildApplyOperation]()
		}
		return buildapply.Result[buildapply.BuildApplyOperation]{Available: true, Value: &o}
	}
	if state.Current != nil {
		return buildapply.MaintenanceBusy[buildapply.BuildApplyOperation]()
	}
	s.mu.Lock()
	if active := s.buildApply; active != nil && active.previewID != r.PreviewID {
		select {
		case <-active.done:
			// A completed local admission is only a duplicate-reply cache.
			// The journal above remains the authority after explicit recovery.
			s.buildApply = nil
		default:
			s.mu.Unlock()
			return buildapply.MaintenanceBusy[buildapply.BuildApplyOperation]()
		}
	}
	if active := s.buildApply; active != nil {
		s.mu.Unlock()
		select {
		case <-active.done:
			return active.result
		case <-time.After(buildApplyLiveBudget):
			return refusedBuildApply("read_status", "The same preview is already being admitted; read its correlated status before retrying.")
		}
	}
	p := s.currentBuildApplyPreviewLocked(c, time.Now())
	if p == nil || p.preview.PreviewID != r.PreviewID || !p.preview.ObservationsComplete || !p.preview.Impact.Complete || len(p.preview.Impact.Refusals) != 0 ||
		p.gui.Executable != r.GUI.Executable || p.client != c || !p.preview.CurrentGUI.BuildID.Known || p.preview.CurrentGUI.BuildID.Value == nil || *p.preview.CurrentGUI.BuildID.Value != r.GUI.BuildID {
		s.mu.Unlock()
		return refusedBuildApply("stale_preview", "A complete current preview and the same native GUI are required.")
	}
	snapshot := buildApplyAdmissionSnapshot{preview: p.preview, gui: p.gui, expires: p.expires}
	if p.candidate != nil && p.candidate.Root != nil {
		snapshot.candidatePath = p.candidate.Root.Path
	}
	if p.target != nil && p.target.Root != nil {
		snapshot.targetPath = p.target.Root.Path
		snapshot.targetSHA256 = p.target.SHA256
	}
	admission := &buildApplyAdmissionState{previewID: r.PreviewID, done: make(chan struct{})}
	s.buildApply = admission
	s.mu.Unlock()
	// Capture reopens and validates its own descriptors. No service mutex or
	// transport write lock is held across owner reads.
	ctx, cancel := context.WithTimeout(context.Background(), buildApplyLiveBudget)
	plan, err := owner.Capture(ctx, snapshot, r)
	if err == nil {
		err = ctx.Err()
	}
	cancel()
	if err == nil {
		err = plan.validate(time.Now())
	}
	if err == nil && (plan.receipt.PreviewID != r.PreviewID || plan.receipt.Consent != r.Consent || !reflect.DeepEqual(plan.receipt.Fences, snapshot.preview.Impact.Fences) || !plan.expires.Equal(snapshot.expires) || !snapshot.preview.PackageSHA256.Known || snapshot.preview.PackageSHA256.Value == nil || *snapshot.preview.PackageSHA256.Value != plan.receipt.CandidateSHA256 || !reflect.DeepEqual(plan.expectedService, snapshot.preview.RequestedService) || !reflect.DeepEqual(plan.expectedGUI, snapshot.preview.RequestedGUI)) {
		err = errors.New("captured authority does not match the exact consented preview")
	}
	if plan.candidate != nil {
		defer plan.candidate.Close()
	}
	if plan.current != nil && plan.current != plan.candidate {
		defer plan.current.Close()
	}
	s.mu.Lock()
	if s.currentBuildApplyPreviewLocked(c, time.Now()) != p {
		err = errors.New("preview changed during admission")
	}
	if err != nil {
		admission.result = refusedBuildApply("owner_evidence_unavailable", "Required live owner evidence is missing, changed or outside its admission budget.")
		close(admission.done)
		s.buildApply = nil
		s.mu.Unlock()
		return admission.result
	}
	s.mu.Unlock()
	state, err = owner.ReadJournal()
	s.mu.Lock()
	if err != nil || state.Current != nil {
		admission.result = refusedBuildApply("recovery_required", "The existing operation must be read and explicitly recovered before a new apply.")
		close(admission.done)
		s.buildApply = nil
		s.mu.Unlock()
		if err == nil && state.Current != nil && state.Current.PreviewID == r.PreviewID {
			o := projectBuildApplyReceipt(*state.Current)
			return buildapply.Result[buildapply.BuildApplyOperation]{Available: true, Value: &o}
		}
		return admission.result
	}
	if last := state.LastTerminal; last != nil && last.PreviewID == r.PreviewID {
		operation := projectBuildApplyReceipt(*last)
		admission.result = buildapply.Result[buildapply.BuildApplyOperation]{Available: true, Value: &operation}
		close(admission.done)
		s.mu.Unlock()
		return admission.result
	}
	if s.currentBuildApplyPreviewLocked(c, time.Now()) != p {
		admission.result = refusedBuildApply("stale_preview", "The preview changed before acceptance.")
		close(admission.done)
		s.buildApply = nil
		s.mu.Unlock()
		return admission.result
	}
	// Captured descriptors belong to the plan. Consume the GUI's preview slot
	// before any writable admission, while duplicate replies retain correlation.
	s.invalidateBuildApplyPreviewLocked(c)
	s.mu.Unlock()
	// No client context: a lost reply is recovered using this preview identity.
	ctx, cancel = context.WithDeadline(context.Background(), plan.deadline)
	result := owner.Execute(ctx, plan)
	cancel()
	s.mu.Lock()
	admission.result = result
	close(admission.done)
	s.mu.Unlock()
	return result
}

func projectBuildApplyReceipt(r buildapply.Receipt) buildapply.BuildApplyOperation {
	o := buildapply.BuildApplyOperation{OperationID: r.OperationID, PreviewID: r.PreviewID, Phase: r.Phase, Outcome: r.Outcome,
		OwnershipAcknowledged: r.Handoff == buildapply.HandoffAcknowledged && r.RunnerReady, ReadbackConfidence: buildapply.ConfidenceUnknown, Fences: r.Fences}
	o.Failure = r.Failure
	if r.Progress != nil && r.ValidateOperation(*r.Progress) == nil {
		progress := *r.Progress
		progress.OwnershipAcknowledged = o.OwnershipAcknowledged
		return progress
	}
	if r.Verified != nil && r.ValidateOperation(*r.Verified) == nil {
		return *r.Verified
	}
	// Expected paused settings are not observed enforcement. In particular a
	// readable succeeded receipt alone does not prove the current live process.
	return o
}

func readBuildApplyStatus(read func() (buildapply.JournalState, error), id string) buildapply.Result[buildapply.BuildApplyStatus] {
	state, err := read()
	if err != nil {
		return buildapply.Unavailable[buildapply.BuildApplyStatus]()
	}
	r := state.Current
	if r == nil {
		r = state.LastTerminal
	}
	status := buildapply.BuildApplyStatus{State: buildapply.StatusIdle}
	if r != nil {
		if id != "" && id != r.OperationID {
			return buildapply.Unavailable[buildapply.BuildApplyStatus]()
		}
		o := projectBuildApplyReceipt(*r)
		if err := r.ValidateOperation(o); err != nil {
			return buildapply.Unavailable[buildapply.BuildApplyStatus]()
		}
		status.Operation = &o
		status.State = buildapply.StatusPending
		if r.Outcome != nil {
			status.State = buildapply.StatusTerminal
		}
		if state.TerminalPending {
			status.State = buildapply.StatusUnknown
		}
	} else if id != "" {
		return buildapply.Unavailable[buildapply.BuildApplyStatus]()
	}
	return buildapply.Result[buildapply.BuildApplyStatus]{Available: true, Value: &status}
}

func (s *Service) buildApplyStatus(r buildapply.StatusRequest) buildapply.Result[buildapply.BuildApplyStatus] {
	return readBuildApplyStatus(buildapply.ReadJournal, r.OperationID)
}

// A pending receipt always gates ordinary startup. Successor entry requires a
// separate exact process reservation; an operation ID by itself grants nothing.
func buildApplyStartupGate() error {
	return servicectl.CheckBuildApplyStartup()
}

func (s *Service) buildApplyAllowsActivation(msg *jsonrpc.Message) bool {
	if msg.Method != "node:set-availability" {
		return true
	}
	var request struct {
		NodeID string `json:"nodeId"`
		State  string `json:"state"`
	}
	if json.Unmarshal(msg.Params, &request) != nil || request.State != "available" {
		return true
	}
	if request.NodeID != "" {
		host := readBuildApplyHostID()
		if host.Known && host.Value != nil && request.NodeID != *host.Value {
			return true
		}
	}
	state, err := buildapply.ReadJournal()
	return err == nil && state.Current == nil && !state.TerminalPending
}

func runBuildApply(args []string, out, stderr io.Writer) int {
	if len(args) == 2 && args[0] == "status" && args[1] == "--json" {
		data, err := buildapply.EncodeStatusResult(readBuildApplyStatus(buildapply.ReadJournal, ""))
		if err == nil {
			_, err = out.Write(append(data, '\n'))
		}
		if err != nil {
			return 1
		}
		return 0
	}
	if len(args) == 2 && (args[0] == "recover" || args[0] == "run") && buildapply.ValidOperationID(args[1]) {
		if err := recoverBuildApply(context.Background(), args[1], args[0] == "run"); err != nil {
			// Operational categories only; no raw launch arguments or settings.
			_, _ = io.WriteString(stderr, "nvpair-service: build apply did not verify this boundary; read apply-build status --json before explicit recovery\n")
			return 1
		}
		if args[0] == "recover" {
			_, _ = io.WriteString(out, "Recovery ownership acknowledged; read apply-build status --json for the outcome.\n")
		}
		return 0
	}
	_, _ = io.WriteString(stderr, "usage: nvpair-service apply-build status --json | recover OPERATION_ID\n")
	return 2
}

func recoverBuildApply(ctx context.Context, id string, reserved bool) error {
	return runNativeBuildApply(ctx, id, reserved)
}
