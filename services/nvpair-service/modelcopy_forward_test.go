// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"io"
	"net"
	"reflect"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
)

func TestModelCopyForwardPreservesPublisherResult(t *testing.T) {
	server, peer := net.Pipe()
	defer peer.Close()
	if err := peer.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	c := newClient(1, server)
	defer c.closeNow()
	brokerRead, brokerWrite := io.Pipe()
	defer brokerRead.Close()
	defer brokerWrite.Close()
	run := &brokerRun{proc: &brokerProc{stdin: brokerWrite}, exited: make(chan struct{})}
	s := &Service{broker: run, clients: map[*client]struct{}{c: {}}, pending: map[int64]*pendingRequest{}}
	id := json.RawMessage(`"copy-client"`)
	params := json.RawMessage(`{"opId":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","sourceNodeId":"source","destinationNodeId":"destination","engine":"llamacpp","model":"owner/repo:Q4_K_M"}`)
	done := make(chan struct{})
	go func() {
		s.forwardRequest(c, &jsonrpc.Message{JSONRPC: "2.0", ID: &id, Method: "engine:copy-model", Params: params})
		close(done)
	}()
	var forwarded jsonrpc.Message
	if err := json.NewDecoder(brokerRead).Decode(&forwarded); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("forwarder did not finish")
	}
	if forwarded.ID == nil || string(*forwarded.ID) == string(id) || forwarded.Method != "engine:copy-model" {
		t.Fatalf("forwarded identity=%+v", forwarded)
	}
	var gotParams, wantParams map[string]any
	if err := json.Unmarshal(forwarded.Params, &gotParams); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(params, &wantParams); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotParams, wantParams) {
		t.Fatalf("copy selector changed: got=%+v want=%+v", gotParams, wantParams)
	}
	result := json.RawMessage(`{"opId":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","result":{"engine":"llamacpp","model":"owner/repo:Q4_K_M","files":1,"filesSkipped":0,"bytesTotal":100,"bytesCopied":100,"publisherComparison":{"status":"mismatch","files":[{"path":"model.gguf","status":"mismatch","provider":"huggingface","repository":"owner/repo","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","file":"model.gguf","algorithm":"sha256","expectedDigest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","actualDigest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","acquisitionBasis":"provider","acquiredAt":"2026-01-01T00:00:00Z"}],"replayed":true}}}`)
	s.routeResponse(&jsonrpc.Message{JSONRPC: "2.0", ID: forwarded.ID, Result: result})
	var response jsonrpc.Message
	if err := json.NewDecoder(peer).Decode(&response); err != nil {
		t.Fatal(err)
	}
	var gotResult, wantResult map[string]any
	if err := json.Unmarshal(response.Result, &gotResult); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(result, &wantResult); err != nil {
		t.Fatal(err)
	}
	if response.ID == nil || string(*response.ID) != string(id) || response.Error != nil || !reflect.DeepEqual(gotResult, wantResult) {
		t.Fatalf("service changed destination observation: %+v", response)
	}
}
