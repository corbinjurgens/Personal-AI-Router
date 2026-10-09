// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"
)

func TestWorkloadsCancelLocalAsksTheProxy(t *testing.T) {
	b, out := newPolicyBroker(t)
	_, proxy := attachFakeProxy(t, b)
	proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
		if method == proxyWorkloadCancel {
			return map[string]bool{"found": true}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	b.handleMessage(clientRequest(1, methodWorkloadsCancel, map[string]any{"originatedFrom": "self-node", "workloadId": "7", "regenerate": true, "engine": "ollama", "runId": "r9"}))
	resp := out.waitFrame(t, "workloads:cancel response", isResponse(1))
	if resp.Error != nil || string(resp.Result) != `{"ok":true}` {
		t.Fatalf("local cancel = %+v %s", resp.Error, resp.Result)
	}
	calls := proxy.snapshot()
	if len(calls) != 1 || calls[0].method != proxyWorkloadCancel {
		t.Fatalf("proxy calls = %v", proxy.methods(""))
	}
	var p map[string]any
	_ = json.Unmarshal(calls[0].params, &p)
	if p["workloadId"] != "7" || p["regenerate"] != true || p["engine"] != "ollama" || p["runId"] != "r9" {
		t.Fatalf("workload/cancel params = %s", calls[0].params)
	}

	b.handleMessage(clientRequest(2, methodWorkloadsCancel, map[string]any{"workloadId": "7"}))
	if resp := out.waitFrame(t, "invalid cancel response", isResponse(2)); resp.Error == nil || resp.Error.Code != -32602 {
		t.Fatalf("cancel without an origin = %+v", resp)
	}
}

func TestWorkloadsCancelForPeerGoesToWorkloadManager(t *testing.T) {
	b, out := newPolicyBroker(t)
	_, proxy := attachFakeProxy(t, b)
	reader, writer := io.Pipe()
	t.Cleanup(func() { _ = reader.Close(); _ = writer.Close() })
	b.setWorkloadMgr(&workloadManagerProcess{stdin: writer, done: make(chan struct{})})
	lines := make(chan string, 1)
	go func() {
		scanner := bufio.NewScanner(reader)
		if scanner.Scan() {
			lines <- scanner.Text()
		}
	}()

	b.handleMessage(clientRequest(1, methodWorkloadsCancel, map[string]any{"originatedFrom": "peer-node", "workloadId": "9"}))
	var frame Message
	select {
	case line := <-lines:
		if err := json.Unmarshal([]byte(line), &frame); err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("cancel never reached the workload-manager")
	}
	if frame.Method != methodWorkloadsCancel || !strings.Contains(string(frame.Params), `"originatedFrom":"peer-node"`) || !strings.Contains(string(frame.Params), `"workloadId":"9"`) {
		t.Fatalf("workload-manager frame = %s %s", frame.Method, frame.Params)
	}
	if resp := out.waitFrame(t, "relayed cancel response", isResponse(1)); resp.Error != nil || string(resp.Result) != `{"ok":true}` {
		t.Fatalf("relayed cancel = %+v %s", resp.Error, resp.Result)
	}
	if n := len(proxy.snapshot()); n != 0 {
		t.Fatal("a peer's job was cancelled through this node's proxy")
	}
}

func TestPeerRelayedCancelActsOnlyOnOrigin(t *testing.T) {
	b, _ := newPolicyBroker(t)
	_, proxy := attachFakeProxy(t, b)
	proxy.setAnswer(func(string, json.RawMessage) (any, string) { return map[string]bool{"found": false}, "" })

	b.forwardWorkloadManagerNotification(methodWorkloadsCancel, json.RawMessage(`{"originatedFrom":"other-node","workloadId":"3"}`))
	time.Sleep(50 * time.Millisecond)
	if n := len(proxy.snapshot()); n != 0 {
		t.Fatal("a node acted on a cancel for a job it did not originate")
	}
	b.forwardWorkloadManagerNotification(methodWorkloadsCancel, json.RawMessage(`{"originatedFrom":"self-node","workloadId":"3","regenerate":true}`))
	calls := proxy.waitFor("workload/cancel on the origin", func(calls []policyCall) bool { return len(calls) == 1 })
	if calls[0].method != proxyWorkloadCancel || !strings.Contains(string(calls[0].params), `"regenerate":true`) {
		t.Fatalf("origin proxy call = %s %s", calls[0].method, calls[0].params)
	}
}
