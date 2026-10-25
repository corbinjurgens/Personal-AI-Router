// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"nvpair-shared/enginesettings"
	"nvpair-tui/rpc"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"
)

var detailSwitchKey = key.NewBinding(key.WithKeys("w"), key.WithHelp("w", "switch engine"))

// All mutable switch state belongs to the update loop. Commands capture only
// immutable scope and a cancellable context, independently of performance.
type detailSwitch struct {
	retainedSwitchRequest
	ctx                    context.Context
	cancel                 context.CancelFunc
	epoch                  int
	open                   bool
	field                  int
	source, target         string
	revision, active       string
	fresh, inFlight, dirty bool
	brokerRestart          bool
	started                bool
	err                    error
	viewport               viewport.Model
}

type switchRequestState int

const (
	switchNone switchRequestState = iota
	switchSubmitting
	switchRefused
	switchUncertain
	// Absence is a point-in-time read, not a fence against later acceptance.
	switchAbsent
	switchAccepted
	switchObservedOther
	switchObservedInactive
	// A durable acknowledgement resolves non-acceptance, not engine state.
	switchFenced
)

type switchEvent int

const (
	switchSubmit switchEvent = iota
	switchSubmitOK
	switchSubmitRefused
	switchSubmitLost
	switchRetire
	switchBaseline
	switchReadOK
	switchReadAbsent
	switchReadError
	switchFenceOK
	switchReset
	switchClose
)

type switchTransition struct {
	event    switchEvent
	id       string
	receipt  *enginesettings.SwitchReceipt
	observed *enginesettings.SwitchObserved
	err      error
}

type switchEffect int

const (
	switchNoRead switchEffect = iota
	switchReadRequest
	switchReadBaseline
	switchFenceRequest
)

// Correlation and its evidence have one owner. Transport/UI flags never decide
// whether a retained request was accepted. Refusal is kept as display evidence
// when following another client's activity, and cleared by a deliberate submit.
type retainedSwitchRequest struct {
	state     switchRequestState
	requestID string
	receipt   *enginesettings.SwitchReceipt
	observed  *enginesettings.SwitchObserved
	refusal   error
	absentAt  time.Time
}

func (r *retainedSwitchRequest) unresolved() bool {
	return r.state == switchSubmitting || r.state == switchUncertain || r.state == switchAbsent ||
		(r.state == switchAccepted && r.receipt != nil && !r.receipt.Terminal)
}

func (r *retainedSwitchRequest) outcomeUnknown() bool {
	return r.state == switchUncertain || r.state == switchAbsent ||
		(r.state == switchObservedOther && r.receipt == nil)
}

// transition handles every event affecting retained identity. The caller fences
// visit/route/transport/epoch and validates receipt identity before dispatch.
func (r *retainedSwitchRequest) transition(e switchTransition) switchEffect {
	switch e.event {
	case switchSubmit:
		*r = retainedSwitchRequest{state: switchSubmitting, requestID: e.id}
	case switchSubmitRefused:
		if r.state == switchSubmitting {
			r.state, r.refusal = switchRefused, e.err
			return switchReadBaseline
		}
	case switchSubmitLost:
		if r.state == switchSubmitting {
			r.state = switchUncertain
			return switchReadRequest
		}
	case switchRetire:
		if r.state == switchSubmitting || (r.state == switchAccepted && r.receipt != nil && !r.receipt.Terminal) || r.state == switchAbsent {
			r.state = switchUncertain
		}
	case switchReset, switchClose:
		*r = retainedSwitchRequest{}
	case switchBaseline:
		// Idle releases observed activity, never a sent local identity. Status
		// reads do not fence late acceptance. A replacement may lose the
		// acceptance race, leaving the original able to execute unobserved.
		if (r.state == switchObservedOther || r.state == switchObservedInactive) && e.id == "" {
			r.state = switchObservedInactive
			return switchNoRead
		}
		if r.state == switchObservedInactive && e.id == r.requestID {
			r.state = switchObservedOther
		}
		if r.state == switchNone || (e.id != "" && e.id != r.requestID && !r.unresolved()) {
			r.state, r.requestID = switchNone, e.id
			r.receipt, r.observed = nil, nil
			if e.id != "" {
				r.state = switchObservedOther
			}
		}
		if r.requestID != "" && r.state != switchFenced && (r.state != switchRefused || e.id == r.requestID) {
			return switchReadRequest
		}
	case switchReadAbsent:
		// Never erase an accepted receipt or infer anything about a foreign
		// caller's receipt. Broker receipts are durable across restart.
		if r.state == switchUncertain && r.receipt == nil {
			r.state = switchAbsent
			r.absentAt = time.Now().UTC()
			return switchFenceRequest
		}
		if r.state == switchAbsent {
			r.absentAt = time.Now().UTC()
			return switchFenceRequest
		}
	case switchFenceOK:
		if r.state == switchAbsent && r.receipt == nil {
			r.state = switchFenced
			return switchReadBaseline
		}
	case switchReadError:
		// Read failure changes availability, not acceptance evidence.
	case switchSubmitOK, switchReadOK:
		if r.requestID == "" || r.state == switchFenced {
			return switchNoRead
		}
		if e.event == switchSubmitOK && r.state != switchSubmitting {
			return switchNoRead
		}
		if r.receipt == nil || e.receipt.Sequence > r.receipt.Sequence {
			r.receipt = e.receipt
		}
		if e.event == switchReadOK {
			r.observed = e.observed
		}
		if r.state != switchObservedOther && r.state != switchObservedInactive {
			r.state = switchAccepted
		}
	}
	return switchNoRead
}

type detailSwitchMsg struct {
	visit, route, epoch int
	transport           uint64
	requestID           string
	submit              bool
	fence               bool
	fenceResult         enginesettings.SwitchFenceResult
	status              enginesettings.SwitchStatus
	err                 error
}

func (d *nodeDetail) retireSwitch() {
	s := &d.engineSwitch
	if s.cancel != nil {
		s.cancel()
	}
	s.ctx, s.cancel = nil, nil
	s.epoch++
	s.fresh, s.inFlight, s.dirty = false, false, false
	s.err = nil
	s.transition(switchTransition{event: switchRetire})
}

func (d *nodeDetail) switchConnected() bool {
	return !d.engineSwitch.brokerRestart && !d.policyDisconnected && d.telemetryFound && d.enginesQueryable() && d.node.presence != presenceOffline
}

func (d *nodeDetail) switchCommand(method string, params any, requestID string, submit bool) tea.Cmd {
	s := &d.engineSwitch
	if s.inFlight || !d.switchConnected() {
		return nil
	}
	if s.ctx == nil {
		s.ctx, s.cancel = context.WithCancel(context.Background())
	}
	s.inFlight = true
	s.started = true
	ctx := s.ctx
	stamp := detailSwitchMsg{visit: d.telemetryGen, route: d.telemetryRoute, epoch: s.epoch,
		transport: clientGeneration(d.client), requestID: requestID, submit: submit,
		fence: method == enginesettings.SwitchFenceMethod}
	client := d.client
	return d.own(func() tea.Msg {
		bounded, cancel := context.WithTimeout(ctx, callTimeout)
		defer cancel()
		msg, err := client.CallGeneration(bounded, stamp.transport, method, params)
		stamp.err = err
		if err == nil {
			if stamp.fence {
				stamp.fenceResult, stamp.err = enginesettings.DecodeSwitchFenceResult(msg.Result)
			} else if submit {
				var result enginesettings.SwitchResult
				stamp.err = json.Unmarshal(msg.Result, &result)
				stamp.status.Receipt = &result.Receipt
			} else {
				stamp.err = json.Unmarshal(msg.Result, &stamp.status)
			}
		}
		return stamp
	})
}

func (d *nodeDetail) switchStatusCmd(baseline bool) tea.Cmd {
	s := &d.engineSwitch
	p := enginesettings.SwitchStatusRequest{NodeID: d.nodeArg()}
	if !baseline {
		p.RequestID = s.requestID
		if s.receipt != nil && !s.receipt.Terminal && !s.outcomeUnknown() {
			sequence := s.receipt.Sequence
			p.AfterSequence = &sequence
		}
	}
	return d.switchCommand(enginesettings.SwitchStatusMethod, p, p.RequestID, false)
}

func (d *nodeDetail) switchReason() string {
	s := &d.engineSwitch
	switch {
	case !d.switchConnected():
		return "node or service disconnected"
	case d.remote() && (!d.remotePermissionFresh || d.remotePermission.Version != 1):
		return "remote management permission unavailable"
	case d.remote() && (!d.remotePermission.Allowed || d.remotePermission.State != "allowed"):
		return "remote management denied on target"
	case !s.fresh || s.revision == "":
		return "revision unavailable; readback required"
	case s.inFlight || s.state == switchSubmitting:
		return "switch readback or submission pending"
	case s.outcomeUnknown():
		return "outcome unknown; readback required"
	case s.state != switchObservedInactive && s.receipt != nil && s.receipt.RecoveryRequired:
		return "recovery required; readback does not perform recovery"
	case s.active != "" || (s.state != switchObservedInactive && s.receipt != nil && !s.receipt.Terminal):
		return "switch active; receipt may be unavailable to this caller"
	case s.source == "" || s.target == "" || s.source == s.target:
		return "select different installed From and To engines"
	}
	// A locked target is refused before anything is sent. A locked source is
	// not: leaving a prohibited engine for an unlocked one stays possible.
	if d.engineLocked(s.target) {
		return engineLockedMessage
	}
	ids := d.switchEngines()
	if !slices.Contains(ids, s.source) || !slices.Contains(ids, s.target) || d.enginesStale {
		return "selected engines unavailable"
	}
	return ""
}

func (d *nodeDetail) submitSwitch() tea.Cmd {
	s := &d.engineSwitch
	if reason := d.switchReason(); reason != "" {
		return nil
	}
	s.transition(switchTransition{event: switchSubmit, id: rand.Text()})
	s.err = nil
	s.fresh = false
	p := enginesettings.SwitchRequest{NodeID: d.nodeArg(), RequestID: s.requestID,
		SourceEngine: s.source, TargetEngine: s.target, ExpectedRevision: s.revision}
	return d.switchCommand(enginesettings.SwitchMethod, p, p.RequestID, true)
}

func (d *nodeDetail) applySwitch(msg detailSwitchMsg) tea.Cmd {
	s := &d.engineSwitch
	if msg.visit != d.telemetryGen || msg.route != d.telemetryRoute || msg.epoch != s.epoch ||
		!d.switchConnected() || (d.client != nil && !d.client.IsCurrent(msg.transport)) ||
		(msg.requestID != "" && msg.requestID != s.requestID) ||
		(msg.requestID != "" && s.state == switchFenced) ||
		(msg.fence && (s.state != switchAbsent || s.receipt != nil)) ||
		(msg.submit && s.state != switchSubmitting) {
		return nil
	}
	s.inFlight = false
	if msg.fence && msg.err == nil {
		result := msg.fenceResult
		if err := result.Validate(); err != nil || result.RequestID != s.requestID || !d.switchNodeMatches(result.NodeID) {
			msg.err = errors.New("invalid fence acknowledgement or identity mismatch; outcome unknown")
		} else if result.Fenced {
			s.err = nil
			s.fresh, s.dirty = false, false
			s.transition(switchTransition{event: switchFenceOK})
			return d.switchStatusCmd(true)
		} else {
			// Acceptance won serialization. The fence never cancels that work.
			msg.status.Receipt = result.Receipt
		}
	}
	s.err = msg.err
	if msg.err != nil {
		if msg.requestID == "" {
			s.fresh = false
		}
		var refusal *rpc.RPCError
		if msg.fence {
			// Lost, denied or invalid acknowledgement preserves the original ID.
			// A later readback may recover a receipt or retry the same fence.
			if errors.As(msg.err, &refusal) && refusal.Code == -32012 {
				d.remotePermissionFresh = false
			}
			return nil
		}
		if msg.submit && errors.As(msg.err, &refusal) &&
			(refusal.Code == -32010 || refusal.Code == -32011 || refusal.Code == -32012 || refusal.Code == -32013 || refusal.Code == -32602) {
			s.transition(switchTransition{event: switchSubmitRefused, err: msg.err})
			s.fresh = false
			if refusal.Code == -32012 {
				d.remotePermissionFresh = false
				return tea.Batch(d.remotePermissionCmd(), d.switchStatusCmd(true))
			}
			return d.switchStatusCmd(true)
		}
		// A transport failure cannot establish whether the broker accepted the
		// request. Read it once under its original identity; never resend it.
		if msg.submit {
			s.transition(switchTransition{event: switchSubmitLost})
			return d.switchStatusCmd(false)
		}
		if msg.requestID != "" {
			event := switchReadError
			// The broker's stdio and peer relays send code/message, without
			// SwitchError.Kind. This exact reply establishes only point-in-time
			// absence for our unacknowledged ID, never future non-acceptance.
			if errors.As(msg.err, &refusal) && refusal.Code == enginesettings.SwitchOperationErrorCode &&
				refusal.Message == "Unknown switch receipt for this caller." {
				event = switchReadAbsent
			}
			if s.transition(switchTransition{event: event}) == switchFenceRequest {
				s.fresh = false
				return d.switchCommand(enginesettings.SwitchFenceMethod,
					enginesettings.SwitchFenceRequest{NodeID: d.nodeArg(), RequestID: s.requestID}, s.requestID, false)
			}
		}
		return nil
	}
	if msg.requestID == "" {
		if s.dirty {
			s.dirty = false
			return d.switchStatusCmd(true)
		}
		s.revision, s.active = msg.status.ConfigurationRevision, msg.status.ActiveRequestID
		s.fresh = s.revision != ""
		if !s.fresh {
			return nil
		}
		if s.transition(switchTransition{event: switchBaseline, id: s.active}) == switchReadRequest {
			return d.switchStatusCmd(false)
		}
		return nil
	}
	r := msg.status.Receipt
	if r == nil || r.RequestID != s.requestID || !d.switchNodeMatches(r.NodeID) ||
		r.Request.RequestID != r.RequestID || r.Request.NodeID != r.NodeID {
		s.err = errors.New("receipt unavailable or identity mismatch; outcome unknown")
		if msg.submit {
			s.transition(switchTransition{event: switchSubmitLost})
		}
		return nil
	}
	event := switchReadOK
	if msg.submit {
		event = switchSubmitOK
	}
	s.transition(switchTransition{event: event, receipt: r, observed: msg.status.Observed})
	var permission tea.Cmd
	if s.receipt.Error != nil && s.receipt.Error.Code == -32012 && d.remotePermissionFresh {
		d.remotePermissionFresh = false
		permission = d.remotePermissionCmd()
	}
	if !s.receipt.Terminal {
		return tea.Batch(permission, d.switchStatusCmd(false))
	}
	if msg.submit || s.dirty || s.active == s.requestID || !s.fresh {
		s.dirty = false
		return tea.Batch(permission, d.switchStatusCmd(true))
	}
	return permission
}

func (d *nodeDetail) switchNodeMatches(node string) bool {
	return node != "" && (node == d.node.key || node == d.hostUUID)
}

func (d *nodeDetail) switchChanged(raw []byte) tea.Cmd {
	var hint enginesettings.SwitchChanged
	if json.Unmarshal(raw, &hint) != nil || !d.switchNodeMatches(hint.NodeID) {
		return nil
	}
	s := &d.engineSwitch
	// Notifications only invalidate. The existing ordered service event stream
	// is subscribed before Init reads the baseline.
	s.fresh = false
	if s.inFlight {
		s.dirty = true
		return nil
	}
	return d.switchStatusCmd(true)
}

func (d *nodeDetail) switchEngines() []string {
	var ids []string
	for _, e := range d.engines {
		if e.Installed {
			ids = append(ids, e.Engine)
		}
	}
	return ids
}

func (d *nodeDetail) switchKey(msg tea.KeyMsg) tea.Cmd {
	s := &d.engineSwitch
	switch msg.String() {
	case "esc":
		s.open = false
	case "tab", "left", "right", "h", "l":
		s.field = 1 - s.field
	case "up", "down", "j", "k":
		ids := d.switchEngines()
		if len(ids) == 0 {
			return nil
		}
		selection := &s.source
		if s.field == 1 {
			selection = &s.target
		}
		i := slices.Index(ids, *selection)
		delta := 1
		if msg.String() == "up" || msg.String() == "k" {
			delta = -1
		}
		*selection = ids[(i+delta+len(ids))%len(ids)]
	case "enter":
		return d.submitSwitch()
	case "r":
		d.remotePermissionFresh = false
		return tea.Batch(d.remotePermissionCmd(), d.switchStatusCmd(true))
	default:
		var cmd tea.Cmd
		s.viewport, cmd = s.viewport.Update(msg)
		return cmd
	}
	return nil
}

func switchEvidence(label string, state enginesettings.SwitchEngineState) string {
	boolean := func(v *bool) string {
		if v == nil {
			return "unknown"
		}
		return fmt.Sprint(*v)
	}
	at := "unknown"
	if state.ObservedAt != nil {
		at = state.ObservedAt.Format(time.RFC3339)
	}
	return fmt.Sprintf("%s: saved intent=%s; running=%s healthy=%s observed=%s termination verified=%t",
		label, state.Intent, boolean(state.Running), boolean(state.Healthy), at, state.TerminationVerified)
}

func (d *nodeDetail) switchView() string {
	s := &d.engineSwitch
	if !s.open && s.requestID == "" {
		return "Switch engine: " + d.switchReason()
	}
	lines := []string{fmt.Sprintf("Switch engine on %s: From %s  To %s", d.node.key, s.source, s.target)}
	if s.open {
		field := "From"
		if s.field == 1 {
			field = "To"
		}
		lines = append(lines, "Editing "+field+": tab field; up/down select; pgup/pgdown scroll; r readback; esc close")
		if reason := d.switchReason(); reason != "" {
			lines = append(lines, "Switch unavailable: "+reason)
		} else {
			lines = append(lines, "enter Switch")
		}
	}
	if s.requestID != "" {
		lines = append(lines, "Request: "+s.requestID)
	}
	if s.outcomeUnknown() {
		lines = append(lines, "Outcome unknown; readback required")
	}
	if s.state == switchAbsent {
		lines = append(lines, "Not accepted as of "+s.absentAt.Format(time.RFC3339Nano)+"; it may still be accepted. Original request retained for readback.")
	}
	if s.state == switchFenced {
		lines = append(lines, "Not accepted; original request fenced by the broker.")
		if !s.fresh {
			lines = append(lines, "Fresh revision readback required before another switch.")
		}
	}
	if s.state == switchObservedInactive {
		if s.receipt == nil {
			lines = append(lines, "Another client's switch is no longer active; its result is not visible to this client")
		} else {
			lines = append(lines, "Another client's switch is no longer active; last observed receipt follows")
		}
	}
	if s.err != nil {
		lines = append(lines, commandFailure(s.err))
	}
	if s.refusal != nil && s.err != s.refusal {
		lines = append(lines, commandFailure(s.refusal))
	}
	if r := s.receipt; r != nil {
		if r.Automatic {
			lines = append(lines, "Automatic engine switch")
		}
		lines = append(lines, fmt.Sprintf("Node %s; From %s To %s; sequence %d; phase %s; recoveryRequired=%t",
			r.NodeID, r.Request.SourceEngine, r.Request.TargetEngine, r.Sequence, r.Phase, r.RecoveryRequired))
		if r.Error != nil {
			lines = append(lines, fmt.Sprintf("%s: %s", r.Error.Kind,
				commandFailure(&rpc.RPCError{Code: r.Error.Code, Message: r.Error.Message})))
		}
		lines = append(lines, switchEvidence("From receipt", r.Source), switchEvidence("To receipt", r.Target))
		if s.observed != nil {
			lines = append(lines, switchEvidence("From current", s.observed.Source), switchEvidence("To current", s.observed.Target))
		}
	}
	return strings.Join(lines, "\n")
}

func (d *nodeDetail) switchPanel(identity string) string {
	v := &d.engineSwitch.viewport
	v.Width = max(1, d.width)
	v.Height = max(1, d.height-countLines(identity))
	v.SetContent(ansi.Hardwrap(d.switchView(), v.Width, true))
	return joinLines(identity, v.View())
}
