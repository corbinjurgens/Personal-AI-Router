// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"errors"
	"testing"

	"nvpair-tui/rpc"
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
		{availabilityState{value: availDraining}, "Pausing…"},
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
