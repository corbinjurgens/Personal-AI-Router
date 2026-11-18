// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package netmon

import (
	"context"
	"net"
	"sync/atomic"
	"testing"
	"time"
)

func runWakeMonitor(t *testing.T) (*Monitor, *atomic.Int64, context.CancelFunc) {
	t.Helper()
	oldInterval, oldNow := wakeInterval, wallNow
	var seconds atomic.Int64
	started := make(chan struct{}, 1)
	wakeInterval = 5 * time.Millisecond
	wallNow = func() time.Time {
		now := time.Unix(seconds.Load(), 0)
		select {
		case started <- struct{}{}:
		default:
		}
		return now
	}
	ctx, cancel := context.WithCancel(context.Background())
	initial := Enumerate()
	m := &Monitor{snap: initial, fp: fingerprint(initial)}
	done := make(chan struct{})
	go func() { defer close(done); m.loop(ctx, nil) }()
	<-started
	t.Cleanup(func() { cancel(); <-done; wakeInterval, wallNow = oldInterval, oldNow })
	return m, &seconds, cancel
}

func TestMonitorWakeNotifiesWithoutAddressChange(t *testing.T) {
	m, clock, _ := runWakeMonitor(t)
	sub, changed := m.Subscribe(), m.Changed()
	clock.Store(60)
	select {
	case <-changed:
	case <-time.After(time.Second):
		t.Fatal("wake did not close Changed")
	}
	select {
	case <-sub:
	case <-time.After(time.Second):
		t.Fatal("wake did not notify Subscribe")
	}
	select {
	case <-sub:
		t.Fatal("duplicate wake")
	case <-time.After(30 * time.Millisecond):
	}
}

func TestMonitorNoWakeWithinSlack(t *testing.T) {
	m, clock, _ := runWakeMonitor(t)
	ch := m.Changed()
	for _, gap := range []int64{5, 14} {
		clock.Add(gap)
		select {
		case <-ch:
			t.Fatalf("wake within %ds slack", gap)
		case <-time.After(30 * time.Millisecond):
		}
	}
}

func TestMonitorChangedClosesOnAddressChange(t *testing.T) {
	m := &Monitor{fp: "old addresses"}
	ch := m.Changed()
	m.refresh(false)
	select {
	case <-ch:
	default:
		t.Fatal("address change not reported")
	}
	if m.Changed() == ch {
		t.Fatal("generation not replaced")
	}
}

func TestMonitorChangedStaysOpenAfterStop(t *testing.T) {
	m, _, cancel := runWakeMonitor(t)
	ch := m.Changed()
	cancel()
	select {
	case <-ch:
		t.Fatal("Changed closed on stop")
	case <-time.After(200 * time.Millisecond):
	}
}

func TestFingerprintOrderInsensitive(t *testing.T) {
	a := Snapshot{
		LocalIPs: map[string]bool{"10.0.0.1": true, "192.168.1.2": true},
		IfaceV4:  map[int][]net.IP{3: {net.IPv4(10, 0, 0, 1), net.IPv4(10, 0, 0, 2)}},
	}
	b := Snapshot{
		LocalIPs: map[string]bool{"192.168.1.2": true, "10.0.0.1": true},
		IfaceV4:  map[int][]net.IP{3: {net.IPv4(10, 0, 0, 2), net.IPv4(10, 0, 0, 1)}},
	}
	if fingerprint(a) != fingerprint(b) {
		t.Fatalf("fingerprints differ for equal-but-reordered snapshots:\n%q\n%q", fingerprint(a), fingerprint(b))
	}
}

func TestFingerprintDetectsChange(t *testing.T) {
	a := Snapshot{LocalIPs: map[string]bool{"10.0.0.1": true}, IfaceV4: map[int][]net.IP{}}
	b := Snapshot{LocalIPs: map[string]bool{"10.0.0.2": true}, IfaceV4: map[int][]net.IP{}}
	if fingerprint(a) == fingerprint(b) {
		t.Fatal("fingerprint should differ when an IP changes")
	}
}

func TestSnapshotCloneIsIndependent(t *testing.T) {
	orig := Snapshot{
		LocalIPs: map[string]bool{"10.0.0.1": true},
		IfaceV4:  map[int][]net.IP{1: {net.IPv4(10, 0, 0, 1)}},
	}
	cp := orig.clone()
	cp.LocalIPs["10.0.0.2"] = true
	cp.IfaceV4[1][0] = net.IPv4(8, 8, 8, 8)
	if orig.LocalIPs["10.0.0.2"] {
		t.Error("clone shares LocalIPs map with original")
	}
	if orig.IfaceV4[1][0].Equal(net.IPv4(8, 8, 8, 8)) {
		t.Error("clone shares IfaceV4 backing array with original")
	}
}

func TestMonitorSnapshotIPBytesAreIndependent(t *testing.T) {
	tests := []struct {
		name string
		ip   net.IP
	}{
		{name: "IPv4", ip: net.IP{192, 0, 2, 1}},
		{name: "IPv6", ip: net.ParseIP("2001:db8::1")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			want := append(net.IP(nil), tt.ip...)
			mon := &Monitor{snap: Snapshot{IfaceV4: map[int][]net.IP{1: {tt.ip}}}}
			first := mon.Snapshot()
			first.IfaceV4[1][0][0] ^= 0xff
			if got := mon.Snapshot().IfaceV4[1][0]; !got.Equal(want) {
				t.Errorf("fresh snapshot IP = %v, want %v", got, want)
			}
			if got := mon.snap.IfaceV4[1][0]; !got.Equal(want) {
				t.Errorf("internal IP = %v, want %v", got, want)
			}
		})
	}
}

func TestMonitorSnapshotPreservesNilAndEmptyIPs(t *testing.T) {
	mon := &Monitor{snap: Snapshot{IfaceV4: map[int][]net.IP{1: {nil, {}}}}}
	got := mon.Snapshot().IfaceV4[1]
	if got[0] != nil || got[1] == nil || len(got[1]) != 0 {
		t.Errorf("snapshot IPs = %#v, want nil and non-nil empty", got)
	}
}

func TestWatchProvidesInitialSnapshot(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	mon, err := Watch(ctx)
	if err != nil {
		t.Fatalf("Watch: %v", err)
	}
	// Snapshot should match a direct enumeration taken at roughly the same
	// time (interfaces don't change during the test).
	if got, want := fingerprint(mon.Snapshot()), fingerprint(Enumerate()); got != want {
		t.Errorf("monitor snapshot %q != enumerate %q", got, want)
	}
	// Subscribe must hand back a usable channel that closes on cancel.
	ch := mon.Subscribe()
	cancel()
	select {
	case <-ch:
		// closed (or signalled) — both acceptable
	case <-time.After(2 * time.Second):
		t.Error("subscription channel not closed after context cancel")
	}
}
