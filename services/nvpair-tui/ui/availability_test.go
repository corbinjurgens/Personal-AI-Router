// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"errors"
	"strings"
	"testing"

	"nvpair-tui/rpc"

	"github.com/charmbracelet/lipgloss"
)

func availNote(node, state string) NotificationMsg {
	return NotificationMsg{Msg: &rpc.Message{
		Method: "node:availability-changed",
		Params: []byte(`{"nodeId":"` + node + `","availability":"` + state + `","active":0}`),
	}}
}

func TestAvailabilityLabels(t *testing.T) {
	cases := []struct {
		a    availabilityState
		want string
	}{
		{availabilityState{}, ""},
		{availabilityState{value: availAvailable}, "Available"},
		{availabilityState{value: availPaused}, "Paused"},
		{availabilityState{value: availDraining}, "Pause incomplete — P retries"},
		{availabilityState{value: availDraining, pending: availPaused}, "Pausing…"},
		{availabilityState{value: availAvailable, pending: availPaused}, "Pausing…"},
		{availabilityState{value: availPaused, pending: availAvailable}, "Resuming…"},
	}
	for _, tc := range cases {
		if got := tc.a.label(); got != tc.want {
			t.Errorf("%+v: label %q, want %q", tc.a, got, tc.want)
		}
	}
}

func TestAvailabilityNotificationsFilterByNode(t *testing.T) {
	var a availabilityState
	a.update(availNote("other", availPaused)) // identity unknown: accepted
	if a.value != availPaused {
		t.Fatalf("unknown identity should accept all, got %q", a.value)
	}
	a.update(selfIdentityMsg{id: clusterIdentity{NodeID: "me", NodeUUID: "uuid-me"}})
	a.update(availNote("other", availAvailable))
	if a.value != availPaused {
		t.Errorf("another node's change applied: %q", a.value)
	}
	a.update(availNote("me", availDraining))
	if a.value != availDraining {
		t.Errorf("own change ignored: %q", a.value)
	}
	a.update(availNote("uuid-me", availAvailable))
	if a.value != availAvailable {
		t.Errorf("uuid-addressed change ignored: %q", a.value)
	}
}

func TestAvailabilityToggleAndResult(t *testing.T) {
	a := availabilityState{value: availAvailable}
	if got := a.toggleTarget(); got != availPaused {
		t.Errorf("toggle from available = %q", got)
	}
	a.pending = availPaused
	a.update(setAvailabilityMsg{target: availPaused, err: errors.New("boom")})
	if a.pending != "" || a.err != "boom" || a.value != availAvailable {
		t.Errorf("failure state wrong: %+v", a)
	}
	a.pending = availPaused
	a.update(setAvailabilityMsg{target: availPaused, availability: availPaused})
	if a.pending != "" || a.value != availPaused {
		t.Errorf("success state wrong: %+v", a)
	}
	if got := a.toggleTarget(); got != availAvailable {
		t.Errorf("toggle from paused = %q", got)
	}
}

func TestFailedPauseLeavesAnAccurateIncompleteState(t *testing.T) {
	a := availabilityState{value: availAvailable, pending: availPaused}
	a.update(setAvailabilityMsg{target: availPaused, err: &rpc.RPCError{Code: -32000, Message: "Pause could not stop every engine: ollama still running"}})
	// The broker stays draining; its readback replaces the stale value.
	a.update(policyGetMsg{availability: availDraining, guarded: true, epoch: a.epoch})
	if a.value != availDraining || a.pending != "" || !strings.Contains(a.err, "ollama still running") {
		t.Fatalf("state after failed pause = %+v", a)
	}
	if got := a.label(); got != "Pause incomplete — P retries" {
		t.Fatalf("label = %q", got)
	}
	if got := a.toggleTarget(); got != availPaused {
		t.Fatalf("P asks for %q, want a pause retry", got)
	}
	a.pending = a.toggleTarget()
	if got := a.label(); got != "Pausing…" {
		t.Fatalf("retry label = %q", got)
	}
}

func TestIncompletePauseKeepsDetailAndDesktopResumeVisible(t *testing.T) {
	for _, width := range []int{80, 200} {
		m := newTestModel()
		m.width = width
		m.ready, m.brokerVersion = true, "1.2.3"
		// The failure arrives the way a real one does: through the RPC error's
		// text, then the draining readback.
		m.avail = availabilityState{value: availAvailable, pending: availPaused}
		m.avail.update(setAvailabilityMsg{target: availPaused, err: &rpc.RPCError{Code: -32000, Message: "Pause could not stop every engine: ollama still running"}})
		m.avail.update(policyGetMsg{availability: availDraining, guarded: true, epoch: m.avail.epoch})
		// The frame is clipped to the terminal, as View does it.
		header := lipgloss.NewStyle().MaxWidth(width).Render(m.headerView())
		for _, want := range []string{"Pause incomplete — P retries", "ollama still running"} {
			if !strings.Contains(header, want) {
				t.Fatalf("width %d: header %q lacks %q", width, header, want)
			}
		}
		banner := lipgloss.NewStyle().MaxWidth(width).Render(m.banner())
		if !strings.Contains(banner, "Pause incomplete — P retries; resume from this device's desktop.") {
			t.Fatalf("width %d: banner %q lacks the recovery sentence", width, banner)
		}
	}
	m := newTestModel()
	m.avail = availabilityState{value: availAvailable, err: "boom"}
	if m.banner() != "" {
		t.Fatal("an ordinary pause failure showed the incomplete-pause banner")
	}
	m.avail = availabilityState{value: availAvailable, err: "boom"}
	if got := m.headerView(); !strings.Contains(got, "pause failed: boom") {
		t.Fatalf("ordinary failure header = %q", got)
	}
}

func TestStalePauseReadbackIsDropped(t *testing.T) {
	a := availabilityState{value: availAvailable, pending: availPaused}
	a.update(setAvailabilityMsg{target: availPaused, err: errors.New("boom")})
	issued := a.epoch // the readback is issued here and is slow
	// A retry completes before the read result arrives.
	a.pending = availPaused
	a.epoch++
	a.update(setAvailabilityMsg{target: availPaused, availability: availPaused})
	a.update(policyGetMsg{availability: availDraining, guarded: true, epoch: issued})
	if a.value != availPaused || a.label() != "Paused" {
		t.Fatalf("stale draining readback applied: %+v", a)
	}
	if got := a.toggleTarget(); got != availAvailable {
		t.Fatalf("P asks for %q, want resume", got)
	}
	// A notification between issue and result also retires the read.
	b := availabilityState{value: availAvailable}
	b.update(setAvailabilityMsg{target: availPaused, err: errors.New("boom")})
	issued = b.epoch
	b.update(availNote("x", availAvailable))
	b.update(policyGetMsg{availability: availDraining, guarded: true, epoch: issued})
	if b.value != availAvailable {
		t.Fatalf("readback overtaken by a notification applied: %+v", b)
	}
}
