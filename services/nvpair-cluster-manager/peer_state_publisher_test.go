// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"strings"
	"testing"

	"nvpair-shared/noderec"
)

func testPeerStateFrame(port int) noderec.PeerStateFrame {
	return noderec.PeerStateFrame{
		Descriptor: noderec.PeerServicesDescriptor{HostUUID: "peer-a", Addresses: []string{"192.168.1.5"}, Services: []noderec.PeerService{{Service: noderec.ServiceNodeInfo, Port: port}}},
		RosterHash: strings.Repeat("0", 64),
	}
}

func TestPeerStatePublisherLateSubscriberGetsLatest(t *testing.T) {
	var p peerStatePublisher
	early, closeEarly, ok := p.subscribe()
	if !ok {
		t.Fatal("subscribe refused")
	}
	defer closeEarly()
	select {
	case frame := <-early:
		t.Fatalf("unavailable publisher pre-filled %+v", frame)
	default:
	}
	p.publish(testPeerStateFrame(1))
	late, closeLate, ok := p.subscribe()
	if !ok {
		t.Fatal("subscribe refused")
	}
	defer closeLate()
	for name, ch := range map[string]<-chan noderec.PeerStateFrame{"early": early, "late": late} {
		select {
		case frame := <-ch:
			if frame.Descriptor.Services[0].Port != 1 {
				t.Fatalf("%s reader got %+v", name, frame)
			}
		default:
			t.Fatalf("%s reader has no frame", name)
		}
	}
}

func TestPeerStatePublisherKeepsOnlyLatest(t *testing.T) {
	var p peerStatePublisher
	ch, unsubscribe, _ := p.subscribe()
	defer unsubscribe()
	p.publish(testPeerStateFrame(1))
	p.publish(testPeerStateFrame(2))
	if frame := <-ch; frame.Descriptor.Services[0].Port != 2 {
		t.Fatalf("reader got %+v, want the latest frame", frame)
	}
	select {
	case frame := <-ch:
		t.Fatalf("stale frame retained: %+v", frame)
	default:
	}
}

func TestPeerStatePublisherReaderCap(t *testing.T) {
	var p peerStatePublisher
	for i := range peerStateMaxReaders {
		if _, _, ok := p.subscribe(); !ok {
			t.Fatalf("subscribe %d refused below the cap", i)
		}
	}
	if _, _, ok := p.subscribe(); ok {
		t.Fatal("subscribe above the cap accepted")
	}
}

func TestPeerStatePublisherOfferSkipsUnchanged(t *testing.T) {
	var p peerStatePublisher
	if !p.offer(testPeerStateFrame(1)) {
		t.Fatal("first frame not published")
	}
	ch, unsubscribe, _ := p.subscribe()
	defer unsubscribe()
	<-ch
	if p.offer(testPeerStateFrame(1)) {
		t.Fatal("unchanged frame republished")
	}
	select {
	case frame := <-ch:
		t.Fatalf("unchanged frame delivered: %+v", frame)
	default:
	}
	changedHash := testPeerStateFrame(1)
	changedHash.RosterHash = strings.Repeat("1", 64)
	if !p.offer(changedHash) {
		t.Fatal("roster hash change not published")
	}
	<-ch
	if !p.offer(testPeerStateFrame(2)) {
		t.Fatal("service change not published")
	}
	<-ch
	p.markUnavailable()
	if !p.offer(testPeerStateFrame(2)) {
		t.Fatal("unchanged frame not republished after unavailability")
	}
}

func TestPeerStateRebuildErrorMarksUnavailable(t *testing.T) {
	m := newReverseTestManager(t)
	setDescriptorSelf(t, m)
	setReverseServices(t, m, 1, noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 22441})
	m.rebuildPeerState()
	if !m.peerState.available() {
		t.Fatal("valid descriptor left the publisher unavailable")
	}
	m.setSelfServicesIdentity(nil)
	m.rebuildPeerState()
	if m.peerState.available() {
		t.Fatal("descriptor error left the publisher available")
	}
}
