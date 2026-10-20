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
	ctx                                         context.Context
	cancel                                      context.CancelFunc
	epoch                                       int
	open                                        bool
	field                                       int
	source, target                              string
	revision, active, requestID                 string
	fresh, inFlight, submitting, unknown, dirty bool
	brokerRestart                               bool
	started                                     bool
	localRequest                                bool
	receipt                                     *enginesettings.SwitchReceipt
	observed                                    *enginesettings.SwitchObserved
	err                                         error
	refusal                                     error
	viewport                                    viewport.Model
}

type detailSwitchMsg struct {
	visit, route, epoch int
	transport           uint64
	requestID           string
	submit              bool
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
	s.fresh, s.inFlight, s.submitting, s.dirty = false, false, false, false
	s.err = nil
	if s.requestID != "" {
		s.unknown = true
	}
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
		transport: clientGeneration(d.client), requestID: requestID, submit: submit}
	client := d.client
	return d.own(func() tea.Msg {
		bounded, cancel := context.WithTimeout(ctx, callTimeout)
		defer cancel()
		msg, err := client.CallGeneration(bounded, stamp.transport, method, params)
		stamp.err = err
		if err == nil {
			if submit {
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
		if s.receipt != nil && !s.receipt.Terminal && !s.unknown {
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
	case s.inFlight || s.submitting:
		return "switch readback or submission pending"
	case s.unknown:
		return "outcome unknown; readback required"
	case s.receipt != nil && s.receipt.RecoveryRequired:
		return "recovery required; readback does not perform recovery"
	case s.active != "" || (s.receipt != nil && !s.receipt.Terminal):
		return "switch active; receipt may be unavailable to this caller"
	case s.source == "" || s.target == "" || s.source == s.target:
		return "select different installed From and To engines"
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
	s.requestID = rand.Text()
	s.localRequest = true
	s.receipt, s.observed, s.err = nil, nil, nil
	s.refusal = nil
	s.submitting, s.unknown = true, true
	s.fresh = false
	p := enginesettings.SwitchRequest{NodeID: d.nodeArg(), RequestID: s.requestID,
		SourceEngine: s.source, TargetEngine: s.target, ExpectedRevision: s.revision}
	return d.switchCommand(enginesettings.SwitchMethod, p, p.RequestID, true)
}

func (d *nodeDetail) applySwitch(msg detailSwitchMsg) tea.Cmd {
	s := &d.engineSwitch
	if msg.visit != d.telemetryGen || msg.route != d.telemetryRoute || msg.epoch != s.epoch ||
		!d.switchConnected() || (d.client != nil && !d.client.IsCurrent(msg.transport)) ||
		(msg.requestID != "" && msg.requestID != s.requestID) {
		return nil
	}
	s.inFlight, s.submitting = false, false
	s.err = msg.err
	if msg.err != nil {
		if msg.requestID == "" {
			s.fresh = false
		} else {
			s.unknown = true
		}
		var refusal *rpc.RPCError
		if msg.submit && errors.As(msg.err, &refusal) &&
			(refusal.Code == -32010 || refusal.Code == -32011 || refusal.Code == -32012 || refusal.Code == -32602) {
			s.refusal = msg.err
			s.unknown, s.fresh = false, false
			if refusal.Code == -32012 {
				d.remotePermissionFresh = false
				return tea.Batch(d.remotePermissionCmd(), d.switchStatusCmd(true))
			}
			return d.switchStatusCmd(true)
		}
		// A transport failure cannot establish whether the broker accepted the
		// request. Read it once under its original identity; never resend it.
		if msg.submit {
			return d.switchStatusCmd(false)
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
		// Only an unresolved local submission needs its original identity for
		// recovery. Definite refusals and terminal receipts resolve that duty;
		// observed activity can always follow the currently advertised request.
		unresolvedLocal := s.localRequest && s.refusal == nil && (s.receipt == nil || !s.receipt.Terminal)
		if s.requestID == "" || (s.active != "" && s.active != s.requestID && !unresolvedLocal) {
			s.requestID = s.active
			s.localRequest = false
			s.receipt, s.observed = nil, nil
			s.unknown = s.requestID != ""
		}
		if s.requestID != "" && (s.unknown || s.receipt != nil || s.active == s.requestID) {
			return d.switchStatusCmd(false)
		}
		return nil
	}
	r := msg.status.Receipt
	if r == nil || r.RequestID != s.requestID || !d.switchNodeMatches(r.NodeID) ||
		r.Request.RequestID != r.RequestID || r.Request.NodeID != r.NodeID {
		s.unknown = true
		s.err = errors.New("receipt unavailable or identity mismatch; outcome unknown")
		return nil
	}
	if s.receipt == nil || r.Sequence > s.receipt.Sequence {
		s.receipt = r
	}
	// Runtime observations describe this read, independently of the immutable
	// receipt sequence, including after the transaction has finished.
	if !msg.submit {
		s.observed = msg.status.Observed
	}
	s.unknown = false
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
	if s.unknown {
		lines = append(lines, "Outcome unknown; readback required")
	}
	if s.err != nil {
		lines = append(lines, commandFailure(s.err))
	}
	if s.refusal != nil && s.err != s.refusal {
		lines = append(lines, commandFailure(s.refusal))
	}
	if r := s.receipt; r != nil {
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
