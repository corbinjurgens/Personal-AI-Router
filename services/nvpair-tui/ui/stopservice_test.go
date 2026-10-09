// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"strings"
	"testing"

	"nvpair-tui/rpc"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

func runeKey(s string) tea.KeyMsg { return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)} }

// isQuit reports whether cmd is tea.Quit.
func isQuit(cmd tea.Cmd) bool {
	if cmd == nil {
		return false
	}
	_, ok := cmd().(tea.QuitMsg)
	return ok
}

// TestPlainQuitLeavesTheServiceRunning: q quits without asking for a stop.
func TestPlainQuitLeavesTheServiceRunning(t *testing.T) {
	m := newTestModel(&stubView{title: "A"})
	next, cmd := m.Update(runeKey("q"))
	if !isQuit(cmd) {
		t.Fatal("q did not quit")
	}
	if next.(Model).stopOnExit {
		t.Fatal("plain quit asked to stop the service")
	}
}

// TestStopServiceKeyAsksFirst: Q arms a confirmation in the banner row, y
// confirms, and anything else cancels without quitting.
func TestStopServiceKeyAsksFirst(t *testing.T) {
	m := newTestModel(&stubView{title: "A", rows: 100, width: 10})

	next, cmd := m.Update(runeKey("Q"))
	m = next.(Model)
	if isQuit(cmd) || !m.confirmingStop {
		t.Fatal("Q acted without asking")
	}
	if !strings.Contains(m.banner(), "y to confirm") {
		t.Fatalf("banner = %q, want the confirmation prompt", m.banner())
	}
	// The prompt is chrome: the frame must still fit the terminal exactly.
	if h := lipgloss.Height(m.View()); h != m.height {
		t.Fatalf("frame height %d with the prompt up, want %d", h, m.height)
	}

	next, cmd = m.Update(runeKey("n"))
	m = next.(Model)
	if isQuit(cmd) || m.confirmingStop || m.stopOnExit {
		t.Fatal("a non-confirming key did not cancel cleanly")
	}

	next, _ = m.Update(runeKey("Q"))
	m = next.(Model)
	next, cmd = m.Update(runeKey("y"))
	if !isQuit(cmd) {
		t.Fatal("confirming did not quit")
	}
	if !next.(Model).stopOnExit {
		t.Fatal("confirming did not record the stop")
	}
}

// TestStopBannerFitsTheMinimumWidth keeps the keys to press on screen at every
// supported width.
func TestStopBannerFitsTheMinimumWidth(t *testing.T) {
	for _, w := range []int{minTerminalWidth, 60, 80, 200} {
		m := newTestModel(&stubView{title: "A"})
		m.width = w
		m.confirmingStop = true
		b := m.banner()
		if lipgloss.Width(b) > w {
			t.Fatalf("width %d: banner is %d wide", w, lipgloss.Width(b))
		}
		if !strings.Contains(b, "y") {
			t.Fatalf("width %d: banner %q lost the confirm key", w, b)
		}
	}
}

func TestBrokerRestartClearsReadyUntilNextAppReady(t *testing.T) {
	m := newTestModel(&stubView{title: "A"})
	m.ready = true
	next, _ := m.Update(NotificationMsg{Msg: &rpc.Message{Method: "service/broker-restarted"}})
	if next.(Model).ready {
		t.Fatal("still ready after the broker restarted")
	}
}

// TestServiceTabStopRowRequiresConfirmation: the Service tab's stop row arms
// first and, on y, asks the shell to stop the service.
func TestServiceTabStopRowRequiresConfirmation(t *testing.T) {
	v := newServiceView(nil)
	idx := -1
	for i, it := range v.items {
		if it.action == actionStop {
			idx = i
		}
	}
	if idx < 0 {
		t.Fatal("no stop row")
	}
	v.cursor = idx
	if cmd := v.activate(); cmd != nil {
		t.Fatal("activating the stop row acted immediately")
	}
	cmd := v.handleKey(runeKey("y"))
	if cmd == nil {
		t.Fatal("confirmation produced no command")
	}
	if _, ok := cmd().(stopServiceMsg); !ok {
		t.Fatal("confirmation did not ask to stop the service")
	}

	m := newTestModel(&stubView{title: "A"})
	next, quit := m.Update(stopServiceMsg{})
	if !isQuit(quit) || !next.(Model).stopOnExit {
		t.Fatal("the shell did not quit with a stop on stopServiceMsg")
	}
}
