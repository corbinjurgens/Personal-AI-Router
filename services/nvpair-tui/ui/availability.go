// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"encoding/json"
	"strings"
	"time"

	"nvpair-tui/rpc"

	tea "github.com/charmbracelet/bubbletea"
)

// The broker's node availability values.
const (
	availAvailable = "available"
	availDraining  = "draining"
	availPaused    = "paused"
)

// setAvailabilityTimeout bounds node:set-availability. The broker blocks until
// the state is reached and a pause waits for running jobs to drain, so the
// ordinary control-call deadline would report a working pause as failed.
const setAvailabilityTimeout = 30 * time.Minute

// pauseIncompleteLabel is what a node that stays draining after its pause call
// ended shows. Resuming it is a desktop or local RPC action: P only retries.
const pauseIncompleteLabel = "Pause incomplete — P retries"

// pauseIncompleteRecovery is the banner for a node left half-paused by a failed
// call. The terminal has no resume key for that state.
const pauseIncompleteRecovery = "Pause incomplete — P retries; resume from this device's desktop."

// pauseStopFailurePrefix leads the broker's incomplete-shutdown error.
const pauseStopFailurePrefix = "Pause could not stop every engine: "

// availabilityState is the shell's view of this node's pause switch.
type availabilityState struct {
	// value is the last availability the broker reported; empty until known.
	value    string
	pausedBy string
	// pending is the target of a node:set-availability call in flight, or empty.
	pending string
	// err is the last failure, shown in the header until the next attempt.
	err string
	// epoch counts the availability changes and attempts seen so far. A readback
	// issued at one epoch is dropped if another change has arrived since, so a
	// slow read cannot restore a state that has already moved on.
	epoch int
	// selfID and selfUUID identify this node, to filter availability-changed
	// notifications. Both empty means unknown: availability-only pushes are accepted.
	selfID, selfUUID string
}

type policyGetMsg struct {
	availability string
	pausedBy     string
	err          error
	// guarded readbacks carry the epoch they were issued at.
	guarded bool
	epoch   int
}

type setAvailabilityMsg struct {
	target       string
	availability string
	err          error
}

type selfIdentityMsg struct{ id clusterIdentity }

func policyGetCmd(client *rpc.Client) tea.Cmd {
	return call(client, "policy:get", map[string]any{}, func(msg *rpc.Message, err error) tea.Msg {
		if err != nil {
			return policyGetMsg{err: err}
		}
		var r targetPolicy
		decodeOrLog("policy:get", msg.Result, &r)
		return policyGetMsg{availability: r.Availability, pausedBy: r.Policy.PausedBy}
	})
}

// policyReadbackCmd reads the policy back for the availability epoch it was
// issued at.
func policyReadbackCmd(client *rpc.Client, epoch int) tea.Cmd {
	read := policyGetCmd(client)
	return func() tea.Msg {
		msg, ok := read().(policyGetMsg)
		if !ok {
			return nil
		}
		msg.guarded, msg.epoch = true, epoch
		return msg
	}
}

func setAvailabilityCmd(client *rpc.Client, target string) tea.Cmd {
	return callWithin(client, setAvailabilityTimeout, "node:set-availability",
		map[string]any{"state": target}, func(msg *rpc.Message, err error) tea.Msg {
			if err != nil {
				return setAvailabilityMsg{target: target, err: err}
			}
			var r struct {
				Availability string `json:"availability"`
			}
			decodeOrLog("node:set-availability", msg.Result, &r)
			return setAvailabilityMsg{target: target, availability: r.Availability}
		})
}

// incomplete reports a pause call that failed while the node stayed draining.
func (a availabilityState) incomplete() bool {
	return a.err != "" && a.value == availDraining
}

// detail is the failure after the broker's fixed lead-in, wherever the transport
// put it ("rpc error -32000: " precedes it), so the part that differs between
// failures is what a narrow header shows.
func (a availabilityState) detail() string {
	if _, rest, found := strings.Cut(a.err, pauseStopFailurePrefix); found {
		return rest
	}
	return a.err
}

// toggleTarget is the state P asks for: resume when paused, pause otherwise.
func (a availabilityState) toggleTarget() string {
	if a.value == availPaused {
		return availAvailable
	}
	return availPaused
}

// label is the header text, or empty while the state is unknown.
func (a availabilityState) label() string {
	switch {
	case a.pending == availPaused:
		return "Pausing…"
	case a.value == availDraining:
		// Nothing is in flight here, so the node is stuck half-paused.
		return pauseIncompleteLabel
	case a.pending == availAvailable:
		return "Resuming…"
	case a.value == availPaused:
		if a.pausedBy != "" {
			return "Paused (" + a.pausedBy + ")"
		}
		return "Paused"
	case a.value == availAvailable:
		return "Available"
	}
	return ""
}

// notify applies a node:availability-changed notification. Notifications for
// another node are ignored when this node's id is known.
func (a *availabilityState) notify(params json.RawMessage) {
	var p struct {
		NodeID       string `json:"nodeId"`
		Availability string `json:"availability"`
	}
	if !decodeOrLog("node:availability-changed", params, &p) {
		return
	}
	known := a.selfID != "" || a.selfUUID != ""
	if known && p.NodeID != a.selfID && p.NodeID != a.selfUUID {
		return
	}
	a.value = p.Availability
	a.pausedBy = ""
	a.epoch++
}

// update applies the availability-related messages.
func (a *availabilityState) update(msg tea.Msg) {
	switch msg := msg.(type) {
	case selfIdentityMsg:
		a.selfID, a.selfUUID = msg.id.NodeID, msg.id.NodeUUID
	case policyGetMsg:
		if msg.guarded && msg.epoch != a.epoch {
			return
		}
		if msg.err == nil && msg.availability != "" {
			a.value = msg.availability
			a.pausedBy = msg.pausedBy
		} else if msg.err != nil {
			a.value, a.pausedBy = "", ""
		}
	case setAvailabilityMsg:
		a.pending = ""
		a.epoch++
		switch {
		case msg.err != nil:
			a.err = commandFailure(msg.err)
		case msg.availability != "":
			a.value = msg.availability
			a.pausedBy = ""
		default:
			a.value = msg.target
			a.pausedBy = ""
		}
	case NotificationMsg:
		if msg.Msg != nil && msg.Msg.Method == "node:availability-changed" {
			a.notify(msg.Msg.Params)
		}
		if msg.Msg != nil && msg.Msg.Method == "policy:changed" {
			var p struct {
				NodeID string `json:"nodeId"`
				targetPolicy
			}
			if decodeOrLog(msg.Msg.Method, msg.Msg.Params, &p) && p.NodeID != "" && (p.NodeID == a.selfID || p.NodeID == a.selfUUID) {
				a.pausedBy = p.Policy.PausedBy
			}
		}
	}
}
