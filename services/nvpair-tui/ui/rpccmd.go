// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"context"
	"time"

	"nvpair-tui/rpc"

	tea "github.com/charmbracelet/bubbletea"
)

// callTimeout bounds a single broker request. It is generous enough to
// cover the broker's slowest relay (the 30s cluster-manager path) plus
// headroom, so a healthy call never times out under us.
const callTimeout = 35 * time.Second

// uiTickInterval drives everything that ages on screen: the relative age
// columns ("12s", "5m") and the expiry of transient status messages. Bubble Tea
// only re-renders in response to a message, so without this the SEEN column sat
// at whatever it read when the last notification happened to arrive.
const uiTickInterval = time.Second

// TickMsg is the shell's periodic re-render pulse, broadcast to every view.
// Views that render relative times need no state to handle it; the redraw alone
// refreshes them.
type TickMsg struct{ Now time.Time }

// uiTick schedules the next pulse. The root model re-arms it on each TickMsg.
func uiTick() tea.Cmd {
	return tea.Tick(uiTickInterval, func(t time.Time) tea.Msg { return TickMsg{Now: t} })
}

// NotificationMsg carries one broker server-push frame into the Bubble
// Tea update loop. Every view receives it.
type NotificationMsg struct{ Msg *rpc.Message }

// DisconnectedMsg invalidates facts owned by a lost service transport.
type DisconnectedMsg struct{}

// sessionReadyMsg refreshes transport-owned facts without restarting UI timers.
type sessionReadyMsg struct{}

type serviceEventMsg struct{ event rpc.Event }
type serviceEventsClosedMsg struct{}

// sessionResult is owned by the shell. View commands keep their ordinary
// message seam; only commands scheduled by the shell receive this envelope.
type sessionResult struct {
	generation uint64
	msg        tea.Msg
}

// sessionInvocation guards work that starts outside the RPC client, such as a
// queued demo dispatcher. Its lifetime is captured when that work is created.
func sessionInvocation(client *rpc.Client, generation uint64, cmd tea.Cmd) tea.Cmd {
	if cmd == nil || client == nil {
		return cmd
	}
	return func() tea.Msg {
		if !client.IsCurrent(generation) {
			return nil
		}
		return cmd()
	}
}

func sessionCommand(client *rpc.Client, generation uint64, cmd tea.Cmd) tea.Cmd {
	if cmd == nil || client == nil {
		return cmd
	}
	return func() tea.Msg {
		msg := cmd()
		switch msg := msg.(type) {
		case tea.BatchMsg:
			cmds := make([]tea.Cmd, len(msg))
			for i, child := range msg {
				cmds[i] = sessionCommand(client, generation, child)
			}
			return tea.BatchMsg(cmds)
		case TickMsg, manualTickMsg, serviceTickMsg, updateCheckDueMsg,
			updateCheckMsg, LogLineMsg, serviceEventMsg, serviceEventsClosedMsg, tea.QuitMsg, sessionResult:
			// These belong to the UI lifetime, including the one timer chain
			// per owner. RPC commands they schedule capture the new generation.
			return msg
		case nil:
			return nil
		default:
			return sessionResult{generation: generation, msg: msg}
		}
	}
}

// LogLineMsg carries one log line into the update loop for the Logs view: the
// broker's and its workers' stderr, and this program's own log.
type LogLineMsg struct{ Line string }

// waitForLog blocks on the next log line, re-arming itself after each one.
//
// The channel is never closed. The broker's stderr ending is not the end of
// the log — this program still writes to it — and a close would make that next
// write panic.
func waitForLog(lines <-chan string) tea.Cmd {
	return func() tea.Msg {
		return LogLineMsg{Line: <-lines}
	}
}

// waitForNotification consumes one ordered service event. The shell re-arms
// this wait after every notification or transient disconnect, until final close.
func waitForNotification(client *rpc.Client) tea.Cmd {
	if client == nil {
		return nil
	}
	return func() tea.Msg {
		event, ok := <-client.Events()
		if !ok {
			return serviceEventsClosedMsg{}
		}
		return serviceEventMsg{event: event}
	}
}

// call issues a broker request on a background goroutine and feeds the
// outcome back into the update loop via decode, which maps the response
// (or error) to a view-specific message.
func call(client *rpc.Client, method string, params any, decode func(*rpc.Message, error) tea.Msg) tea.Cmd {
	return callWithin(client, callTimeout, method, params, decode)
}

// callWithin is call with an explicit budget, for a request whose duration is
// set by the work it starts rather than by the round trip.
//
// callTimeout was reasoned about for control calls, whose slowest leg is a
// thirty-second relay. A request that waits on an engine restarting is not one
// of those, and giving it the same deadline reported a working change as a
// failure half a minute in, while the backend carried on and finished it.
func callWithin(client *rpc.Client, budget time.Duration, method string, params any,
	decode func(*rpc.Message, error) tea.Msg) tea.Cmd {
	generation := clientGeneration(client)
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), budget)
		defer cancel()
		msg, err := client.CallGeneration(ctx, generation, method, params)
		return decode(msg, err)
	}
}

func clientGeneration(client *rpc.Client) uint64 {
	if client == nil {
		return 0
	}
	return client.Generation()
}
