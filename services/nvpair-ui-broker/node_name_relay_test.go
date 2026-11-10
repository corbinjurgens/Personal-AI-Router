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
)

// nodeNameRelayChild starts one end of a pipe as a broker-side Peer and serves
// the other end with respond, recording every request it receives.
func nodeNameRelayChild(t *testing.T, respond func(*Message) json.RawMessage) (*Peer, <-chan *Message) {
	t.Helper()
	brokerConn, childConn := net.Pipe()
	brokerSide := NewPeer(NewCodec(brokerConn))
	childSide := NewPeer(NewCodec(childConn))
	received := make(chan *Message, 16)
	brokerDone, childDone := make(chan struct{}), make(chan struct{})
	go func() { brokerSide.Serve(nil, nil); close(brokerDone) }()
	go func() {
		childSide.Serve(func(msg *Message) {
			received <- msg
			if msg.ID != nil {
				if err := childSide.Respond(msg.ID, respond(msg)); err != nil {
					t.Errorf("respond from child: %v", err)
				}
			}
		}, nil)
		close(childDone)
	}()
	t.Cleanup(func() {
		brokerSide.Close()
		childSide.Close()
		_ = brokerConn.Close()
		_ = childConn.Close()
		for _, done := range []chan struct{}{brokerDone, childDone} {
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Error("node name relay fixture read pump did not stop")
			}
		}
	})
	return brokerSide, received
}

func nodeNameRelayScanner(t *testing.T) (*scannerProcess, <-chan *Message) {
	t.Helper()
	peer, received := nodeNameRelayChild(t, func(*Message) json.RawMessage { return json.RawMessage(`{"ok":true}`) })
	return &scannerProcess{peer: peer}, received
}

// requireSetSelfName waits for the next discovery:set-self-name request,
// skipping other scanner traffic, and checks the name it carries.
func requireSetSelfName(t *testing.T, received <-chan *Message, want string) {
	t.Helper()
	deadline := time.After(2 * time.Second)
	for {
		select {
		case msg := <-received:
			if msg.Method != noderec.MethodSetSelfName {
				continue
			}
			var params noderec.SetSelfNameParams
			if err := json.Unmarshal(msg.Params, &params); err != nil {
				t.Fatalf("decode set-self-name params: %v", err)
			}
			if params.Name != want {
				t.Fatalf("scanner received name %q, want %q", params.Name, want)
			}
			return
		case <-deadline:
			t.Fatalf("scanner never received set-self-name %q", want)
		}
	}
}

func TestNodeNameRelayGetNodeIDNameReachesScanner(t *testing.T) {
	cmPeer, cmReceived := nodeNameRelayChild(t, func(msg *Message) json.RawMessage {
		if msg.Method != "cluster:get-node-id" {
			return json.RawMessage(`{}`)
		}
		return json.RawMessage(`{"nodeUuid":"self-uuid","name":"Desk A","customName":null,"certFingerprint":"","clusterId":""}`)
	})
	scanner, scannerReceived := nodeNameRelayScanner(t)
	b := &Broker{clusterMgr: &clusterManagerProcess{peer: cmPeer}, scanner: scanner}

	b.convergeScannerIdentity()

	if msg := <-cmReceived; msg.Method != "cluster:get-node-id" {
		t.Fatalf("cluster manager received %q, want cluster:get-node-id", msg.Method)
	}
	select {
	case msg := <-scannerReceived:
		if msg.Method != noderec.MethodSetSelfName {
			t.Fatalf("first scanner request = %q, want the name before %s", msg.Method, noderec.MethodReloadIdentity)
		}
		var params noderec.SetSelfNameParams
		if err := json.Unmarshal(msg.Params, &params); err != nil || params.Name != "Desk A" {
			t.Fatalf("set-self-name params = %s (%v), want Desk A", msg.Params, err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("scanner never received set-self-name")
	}
	select {
	case msg := <-scannerReceived:
		if msg.Method != noderec.MethodReloadIdentity {
			t.Fatalf("second scanner request = %q, want %s", msg.Method, noderec.MethodReloadIdentity)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("scanner never received reload-identity")
	}
	b.selfNameMu.Lock()
	uuid, name := b.selfNodeUUID, b.selfNodeName
	b.selfNameMu.Unlock()
	if uuid != "self-uuid" || name != "Desk A" {
		t.Fatalf("cached self = %q %q, want self-uuid Desk A", uuid, name)
	}
}

func TestNodeNameRelayNameChangeReachesClientAndScanner(t *testing.T) {
	scanner, scannerReceived := nodeNameRelayScanner(t)
	var output bytes.Buffer
	b := &Broker{codec: NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), &output}), scanner: scanner}
	params := json.RawMessage(`{"nodeUuid":"self-uuid","name":"Renamed","customName":"Renamed"}`)

	b.forwardClusterManagerNotification("cluster:node-name-changed", params)

	forwarded, err := NewCodec(bytes.NewBuffer(output.Bytes())).Read()
	if err != nil {
		t.Fatalf("read forwarded notification: %v", err)
	}
	if forwarded.Method != "cluster:node-name-changed" || forwarded.ID != nil {
		t.Fatalf("client received %q (id %v), want the cluster:node-name-changed notification", forwarded.Method, forwarded.ID)
	}
	requireJoinRequestRelayJSON(t, forwarded.Params, params)
	requireSetSelfName(t, scannerReceived, "Renamed")
}

func TestNodeNameRelayRespawnedScannerReceivesCachedName(t *testing.T) {
	first, firstReceived := nodeNameRelayScanner(t)
	b := &Broker{scanner: first}
	b.recordSelfName(nodeNameIdentity{NodeUUID: "self-uuid", Name: "Cached"})
	b.pushSelfName()
	requireSetSelfName(t, firstReceived, "Cached")

	replacement, replacementReceived := nodeNameRelayScanner(t)
	b.setScanner(replacement)
	// spawnScanner replays the cached name to every new scanner this way.
	b.pushSelfName()
	requireSetSelfName(t, replacementReceived, "Cached")
}

func TestNodeNameRelaySendsNothingBeforeTheManagerReportsAName(t *testing.T) {
	scanner, received := nodeNameRelayScanner(t)
	b := &Broker{scanner: scanner}
	b.pushSelfName()
	select {
	case msg := <-received:
		t.Fatalf("scanner received %q before any name was reported", msg.Method)
	case <-time.After(100 * time.Millisecond):
	}
}
