// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net"
	"reflect"
	"strconv"
	"testing"
	"time"
)

func modelCopyRelayFixture(t *testing.T, answer func(*Peer, *Message)) (*Broker, *Codec, *engineManagerSource) {
	t.Helper()
	brokerWorker, workerConn := net.Pipe()
	brokerClient, clientConn := net.Pipe()
	worker := NewPeer(NewCodec(workerConn))
	peer := NewPeer(NewCodec(brokerWorker))
	b := &Broker{codec: NewCodec(brokerClient), engineMgr: &rpcWorker{peer: peer}, engineSubscribed: true}
	startTestEngineNotifications(t, b, func() { _ = brokerClient.Close() })
	source := b.activateEngineManagerSource()
	b.bindEngineMgr(b.engineMgr, source)
	pump, workerPump := make(chan struct{}), make(chan struct{})
	go func() { peer.Serve(nil, nil); close(pump) }()
	go func() { worker.Serve(func(msg *Message) { answer(worker, msg) }, nil); close(workerPump) }()
	if err := clientConn.SetReadDeadline(time.Now().Add(3 * time.Second)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		peer.Close()
		worker.Close()
		for _, conn := range []net.Conn{brokerWorker, workerConn, brokerClient, clientConn} {
			_ = conn.Close()
		}
		for _, done := range []chan struct{}{pump, workerPump} {
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Error("copy relay fixture retained a read pump")
			}
		}
	})
	return b, NewCodec(clientConn), source
}

func requireCopyRelayJSON(t *testing.T, actual, expected json.RawMessage) {
	t.Helper()
	var got, want any
	if err := json.Unmarshal(actual, &got); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(expected, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("relay changed JSON: got %s want %s", actual, expected)
	}
}

func TestModelCopyRelayPreservesRequestsAndResults(t *testing.T) {
	test := func(name, method, params, result string) {
		t.Run(name, func(t *testing.T) {
			requests := make(chan *Message, 1)
			b, client, _ := modelCopyRelayFixture(t, func(worker *Peer, msg *Message) {
				requests <- msg
				if err := worker.Respond(msg.ID, json.RawMessage(result)); err != nil {
					t.Errorf("worker response: %v", err)
				}
			})
			id := json.RawMessage(`"copy-client"`)
			b.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: method, Params: json.RawMessage(params)})
			response, err := client.Read()
			if err != nil {
				t.Fatal(err)
			}
			if response.Error != nil || response.ID == nil || string(*response.ID) != string(id) {
				t.Fatalf("response=%+v", response)
			}
			requireCopyRelayJSON(t, response.Result, json.RawMessage(result))
			select {
			case request := <-requests:
				if request.Method != method {
					t.Fatalf("method=%s", request.Method)
				}
				requireCopyRelayJSON(t, request.Params, json.RawMessage(params))
			default:
				t.Fatal("request did not reach engine manager")
			}
		})
	}
	test("copy tuple and qualified result", "engine:copy-model", `{"opId":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","sourceNodeId":"msi-host","destinationNodeId":"joes-host","engine":"e-llama-custom","model":"custom-alias","expectedExportId":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}`, `{"opId":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","result":{"engine":"e-llama-custom","model":"custom-alias","files":2,"filesSkipped":1,"bytesTotal":100,"bytesCopied":50,"layout":"llama-artifacts-v1","exportId":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","durability":"uncertain","requiresConfiguration":true}}`)
	test("caller capabilities", "engine:model-copy-capabilities", `{"nodeId":"joes-host","engine":"e-llama-custom"}`, `{"version":1,"engine":"e-llama-custom","available":true,"installed":true,"modelStore":"llamacpp","exportLayouts":["llamacpp","llama-artifacts-v1"],"importLayouts":["llamacpp","llama-artifacts-v1"],"writable":true,"reason":"","identityBoundRetry":true,"receiptDurability":true}`)
	test("cancel exact operation", "engine:cancel-model-copy", `{"opId":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}`, `{"cancelled":true}`)
	test("active tuple and byte counts", "engine:active-model-copies", `{}`, `{"operations":[{"opId":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","sourceNodeId":"msi-host","destinationNodeId":"joes-host","engine":"llamacpp","model":"alias","progress":{"opId":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","node":"msi-host","sourceNodeId":"msi-host","destinationNodeId":"joes-host","engine":"llamacpp","model":"alias","op":"copy","stage":"downloading","file":"model/a.gguf","bytesDone":50,"bytesTotal":100,"exportId":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}]}`)
}

func TestModelCopyRelayPreservesTypedRefusals(t *testing.T) {
	for _, code := range []int{-32010, -32011, -32012} {
		t.Run(strconv.Itoa(code), func(t *testing.T) {
			b, client, _ := modelCopyRelayFixture(t, func(worker *Peer, msg *Message) {
				if err := worker.RespondError(msg.ID, code, "copy refused"); err != nil {
					t.Errorf("worker refusal: %v", err)
				}
			})
			id := json.RawMessage(`1`)
			b.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: "engine:copy-model", Params: json.RawMessage(`{}`)})
			response, err := client.Read()
			if err != nil {
				t.Fatal(err)
			}
			if response.Error == nil || response.Error.Code != code || response.Error.Message != "copy refused" {
				t.Fatalf("refusal=%+v", response.Error)
			}
		})
	}
}

func TestModelCopyRelayPreservesProgress(t *testing.T) {
	test := func(name, stage string, percent int) {
		t.Run(name, func(t *testing.T) {
			b, client, source := modelCopyRelayFixture(t, func(_ *Peer, _ *Message) { t.Error("progress made an RPC") })
			payload := json.RawMessage(`{"opId":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","node":"msi-host","sourceNodeId":"msi-host","destinationNodeId":"joes-host","engine":"llamacpp","model":"alias","op":"copy","stage":"` + stage + `","percent":` + strconv.Itoa(percent) + `,"bytesDone":100,"bytesTotal":100,"exportId":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}`)
			done := make(chan struct{})
			go func() { b.forwardEngineNotificationFrom(source, "engine:remote-progress", payload); close(done) }()
			message, err := client.Read()
			if err != nil {
				t.Fatal(err)
			}
			if message.Method != "engine:remote-progress" {
				t.Fatalf("method=%s", message.Method)
			}
			requireCopyRelayJSON(t, message.Params, payload)
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("progress forward did not finish")
			}
		})
	}
	test("verified terminal byte counts", "done", 100)
	test("unknown acknowledgement retains operation identity", "unknown", -1)
}
