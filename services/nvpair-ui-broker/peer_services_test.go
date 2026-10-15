// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net"
	"testing"
	"time"

	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

func brokerPeerClaim(t *testing.T) json.RawMessage {
	t.Helper()
	claim := noderec.PeerServicesClaim{Descriptor: noderec.PeerServicesDescriptor{HostUUID: "peer-a", Name: "peer-a", Addresses: []string{"192.168.42.8"}, Services: []noderec.PeerService{{Service: noderec.ServiceNodeInfo, Port: 25441}}}, Incarnation: "original-admission", Source: "current-child", Sequence: 1, ObservedAt: time.Now().UnixMilli()}
	body, err := json.Marshal(claim)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func TestPeerServicesRelayIsPrivateAndSeparateFromLocalRegistrations(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()
	var public bytes.Buffer
	b := &Broker{codec: NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), &public}), regCache: relay.NewRegistrationCache()}
	child := &clusterManagerProcess{}
	// Install without registration replay: this fixture exercises only the
	// private relay and asserts that remote metadata cannot become local state.
	b.clusterMgr = child
	scanner := &scannerProcess{peer: NewPeer(NewCodec(client)), done: make(chan struct{})}
	b.setScanner(scanner)
	done := make(chan struct{})
	go func() { b.relayPeerServices(child, brokerPeerClaim(t)); close(done) }()
	if err := server.SetReadDeadline(time.Now().Add(2 * time.Second)); err != nil {
		t.Fatal(err)
	}
	msg, err := NewCodec(server).Read()
	if err != nil {
		t.Fatal(err)
	}
	<-done
	var claim noderec.PeerServicesClaim
	if err := json.Unmarshal(msg.Params, &claim); err != nil {
		t.Fatal(err)
	}
	if msg.Method != peerServicesIngest || !msg.IsNotification() || claim.Incarnation != "original-admission" || claim.Descriptor.Services[0].Port != 25441 {
		t.Fatalf("private frame=%+v claim=%+v", msg, claim)
	}
	b.forwardClusterManagerNotification(peerServicesEvent, brokerPeerClaim(t))
	if public.Len() != 0 {
		t.Fatalf("private event leaked %d bytes to client", public.Len())
	}
	if len(b.regCache.Snapshot()) != 0 || b.localServicesGeneration != 0 {
		t.Fatalf("remote descriptor mutated local registrations: %+v", b.regCache.Snapshot())
	}
}

func TestPeerServicesRelayRejectsReplacedChild(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()
	b := &Broker{}
	old, current := &clusterManagerProcess{}, &clusterManagerProcess{}
	b.clusterMgr = current
	b.setScanner(&scannerProcess{peer: NewPeer(NewCodec(client)), done: make(chan struct{})})
	b.relayPeerServices(old, brokerPeerClaim(t))
	if err := server.SetReadDeadline(time.Now().Add(30 * time.Millisecond)); err != nil {
		t.Fatal(err)
	}
	var one [1]byte
	n, err := server.Read(one[:])
	if n != 0 {
		t.Fatalf("old child wrote %d bytes", n)
	}
	if timeout, ok := err.(net.Error); !ok || !timeout.Timeout() {
		t.Fatalf("read error=%v, want quiet current scanner pipe", err)
	}
}
