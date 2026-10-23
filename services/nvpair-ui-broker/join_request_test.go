// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net"
	"reflect"
	"testing"
	"time"
)

const joinRequestRelayRecord = `{
	"requestId":"0123456789abcdef0123456789abcdef",
	"direction":"outgoing","peerNodeUuid":"selected-member","peerName":"Selected member",
	"peerAddress":"192.168.42.8","peerPort":14321,"state":"awaiting-approval",
	"inviteId":null,"reason":"","createdAt":1000,"expiresAt":301000
}`

const joinRequestRelaySnapshot = `{
	"supported":true,"canAskToJoin":true,"requiresSoloReplacement":false,
	"blockedReason":"","requests":[` + joinRequestRelayRecord + `]
}`

func requireJoinRequestRelayJSON(t *testing.T, got, want json.RawMessage) {
	t.Helper()
	var actual, expected any
	if err := json.Unmarshal(got, &actual); err != nil {
		t.Fatalf("decode relayed JSON: %v", err)
	}
	if err := json.Unmarshal(want, &expected); err != nil {
		t.Fatalf("decode expected JSON: %v", err)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("relay changed payload: got %s, want %s", got, want)
	}
}

func joinRequestRelayBroker(t *testing.T, respond func(*Peer, *Message)) (*Broker, *bytes.Buffer) {
	t.Helper()
	brokerConn, managerConn := net.Pipe()
	manager := NewPeer(NewCodec(managerConn))
	child := &clusterManagerProcess{peer: NewPeer(NewCodec(brokerConn))}
	var output bytes.Buffer
	b := &Broker{codec: NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), &output}), clusterMgr: child}
	childDone, managerDone := make(chan struct{}), make(chan struct{})
	go func() { child.peer.Serve(nil, nil); close(childDone) }()
	go func() {
		manager.Serve(func(msg *Message) { respond(manager, msg) }, nil)
		close(managerDone)
	}()
	t.Cleanup(func() {
		child.peer.Close()
		manager.Close()
		_ = brokerConn.Close()
		_ = managerConn.Close()
		for _, done := range []chan struct{}{childDone, managerDone} {
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Error("join request fixture read pump did not stop")
			}
		}
	})
	return b, &output
}

func joinRequestRelayCall(t *testing.T, b *Broker, output *bytes.Buffer, method, params string) *Message {
	t.Helper()
	id := json.RawMessage(`"local-client-request"`)
	b.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: method, Params: json.RawMessage(params)})
	response, err := NewCodec(bytes.NewBuffer(output.Bytes())).Read()
	if err != nil {
		t.Fatalf("read broker response: %v", err)
	}
	if response.ID == nil || string(*response.ID) != string(id) {
		t.Fatalf("response lost local correlation: %+v", response.ID)
	}
	return response
}

func TestJoinRequestRelayPreservesMethodsAndPayloads(t *testing.T) {
	test := func(name, method, params, result string) {
		t.Run(name, func(t *testing.T) {
			requests := make(chan *Message, 1)
			b, output := joinRequestRelayBroker(t, func(manager *Peer, msg *Message) {
				requests <- msg
				if err := manager.Respond(msg.ID, json.RawMessage(result)); err != nil {
					t.Errorf("respond from manager: %v", err)
				}
			})
			response := joinRequestRelayCall(t, b, output, method, params)
			select {
			case msg := <-requests:
				if msg.Method != method || msg.ID == nil {
					t.Fatalf("manager received method %q, want request %q", msg.Method, method)
				}
				requireJoinRequestRelayJSON(t, msg.Params, json.RawMessage(params))
			case <-time.After(time.Second):
				t.Fatal("join request did not reach manager")
			}
			if response.Error != nil {
				t.Fatalf("relay rejected request: %+v", response.Error)
			}
			requireJoinRequestRelayJSON(t, response.Result, json.RawMessage(result))
		})
	}
	test("snapshot", "cluster:get-join-requests", `{}`, joinRequestRelaySnapshot)
	test("ask with explicit solo consent", "cluster:ask-to-join", `{"address":"192.168.42.8","port":14321,"nodeId":"selected-member","replaceSoloCluster":true}`, joinRequestRelayRecord)
	test("ask without solo replacement", "cluster:ask-to-join", `{"address":"member.example","replaceSoloCluster":false}`, joinRequestRelayRecord)
	test("approve with local grant", "cluster:respond-to-join-request", `{"requestId":"0123456789abcdef0123456789abcdef","accept":true,"allowRemoteManagement":true}`, joinRequestRelayRecord)
	test("approve without local grant", "cluster:respond-to-join-request", `{"requestId":"0123456789abcdef0123456789abcdef","accept":true,"allowRemoteManagement":false}`, joinRequestRelayRecord)
	test("decline", "cluster:respond-to-join-request", `{"requestId":"0123456789abcdef0123456789abcdef","accept":false}`, joinRequestRelayRecord)
	test("cancel exact request", "cluster:cancel-join-request", `{"requestId":"0123456789abcdef0123456789abcdef"}`, joinRequestRelayRecord)
}

func TestJoinRequestRelayPreservesStableErrors(t *testing.T) {
	test := func(name, method, params string, code int, message string) {
		t.Run(name, func(t *testing.T) {
			b, output := joinRequestRelayBroker(t, func(manager *Peer, msg *Message) {
				if err := manager.RespondErrorData(msg.ID, code, message, map[string]string{"reason": "state-unavailable"}); err != nil {
					t.Errorf("respond with manager error: %v", err)
				}
			})
			response := joinRequestRelayCall(t, b, output, method, params)
			if response.Error == nil || response.Error.Code != code || response.Error.Message != message || len(response.Result) != 0 {
				t.Fatalf("relay changed error: result=%s error=%+v", response.Result, response.Error)
			}
			// The generic relay preserves code/message, not error data. Clients
			// obtain eligibility reasons from get-join-requests instead.
			if len(response.Error.Data) != 0 {
				t.Fatalf("generic relay unexpectedly forwarded error data: %s", response.Error.Data)
			}
		})
	}
	test("unknown request", "cluster:cancel-join-request", `{"requestId":"unknown"}`, -32001, "unknown join request")
	test("invalid state", "cluster:respond-to-join-request", `{"requestId":"terminal","accept":false}`, -32002, "invalid join request state")
	test("invalid params", "cluster:ask-to-join", `{"address":"192.168.42.8"}`, -32602, "replaceSoloCluster must be a boolean")
	test("eligibility refusal", "cluster:ask-to-join", `{"address":"192.168.42.8","replaceSoloCluster":false}`, -32004, "state-unavailable")
	test("old local service", "cluster:get-join-requests", `{}`, -32601, "method not found: cluster:get-join-requests")
}

func TestJoinRequestRelayUnavailableManager(t *testing.T) {
	b, output := joinRequestRelayBroker(t, func(_ *Peer, _ *Message) { t.Error("unavailable manager received a request") })
	b.clusterMgr = nil
	response := joinRequestRelayCall(t, b, output, "cluster:get-join-requests", `{}`)
	if response.Error == nil || response.Error.Code != -32000 || response.Error.Message != "nvpair-cluster-manager not available" || len(response.Result) != 0 {
		t.Fatalf("unavailable manager response: result=%s error=%+v", response.Result, response.Error)
	}
}

func TestJoinRequestNotificationForwardsFullSnapshotWithoutSubscription(t *testing.T) {
	test := func(name, snapshot string) {
		t.Run(name, func(t *testing.T) {
			var output bytes.Buffer
			b := &Broker{codec: NewCodec(struct {
				io.Reader
				io.Writer
			}{bytes.NewReader(nil), &output})}
			b.forwardClusterManagerNotification("cluster:join-requests-changed", json.RawMessage(snapshot))
			msg, err := NewCodec(bytes.NewBuffer(output.Bytes())).Read()
			if err != nil {
				t.Fatalf("read join request notification: %v", err)
			}
			if !msg.IsNotification() || msg.Method != "cluster:join-requests-changed" {
				t.Fatalf("unexpected notification: %+v", msg)
			}
			requireJoinRequestRelayJSON(t, msg.Params, json.RawMessage(snapshot))
		})
	}
	test("pending outgoing request", joinRequestRelaySnapshot)
	test("incoming handoff and blocked eligibility", `{
		"supported":true,"canAskToJoin":false,"requiresSoloReplacement":false,
		"blockedReason":"already-clustered","requests":[{
			"requestId":"0123456789abcdef0123456789abcdef","direction":"incoming",
			"peerNodeUuid":"requester","peerName":"Unverified requester",
			"peerAddress":"192.168.42.9","peerPort":14321,"state":"pairing",
			"inviteId":"exact-child-invite","reason":"awaiting-confirmation",
			"createdAt":1000,"expiresAt":301000
		}]
	}`)
	test("empty solo eligibility", `{
		"supported":true,"canAskToJoin":true,"requiresSoloReplacement":true,
		"blockedReason":"","requests":[]
	}`)
}
