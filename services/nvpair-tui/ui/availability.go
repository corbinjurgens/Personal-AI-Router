// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"encoding/json"
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

// availabilityState is the shell's view of this node's pause switch.
type availabilityState struct {
	// value is the last availability the broker reported; empty until known.
	value string
	// pending is the target of a node:set-availability call in flight, or empty.
	pending string
	// err is the last failure, shown in the header until the next attempt.
	err string
	// selfID and selfUUID identify this node, to filter availability-changed
	// notifications. Both empty means unknown: every notification is accepted.
	selfID, selfUUID string
}

type policyGetMsg struct {
	availability string
	err          error
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
		var r struct {
			Availability string `json:"availability"`
		}
		decodeOrLog("policy:get", msg.Result, &r)
		return policyGetMsg{availability: r.Availability}
	})
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
	case a.pending == availPaused, a.value == availDraining:
		return "Pausing…"
	case a.pending == availAvailable:
		return "Resuming…"
	case a.value == availPaused:
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
}

// update applies the availability-related messages.
func (a *availabilityState) update(msg tea.Msg) {
	switch msg := msg.(type) {
	case selfIdentityMsg:
		a.selfID, a.selfUUID = msg.id.NodeID, msg.id.NodeUUID
	case policyGetMsg:
		if msg.err == nil && msg.availability != "" {
			a.value = msg.availability
		}
	case setAvailabilityMsg:
		a.pending = ""
		switch {
		case msg.err != nil:
			a.err = msg.err.Error()
		case msg.availability != "":
			a.value = msg.availability
		default:
			a.value = msg.target
		}
	case NotificationMsg:
		if msg.Msg != nil && msg.Msg.Method == "node:availability-changed" {
			a.notify(msg.Msg.Params)
		}
	}
}
