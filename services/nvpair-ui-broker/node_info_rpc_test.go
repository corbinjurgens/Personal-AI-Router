// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/testbin"
)

func nodeInfoBrokerTestClient(t *testing.T, onScannerRequest func(*Peer, *Message)) (*Broker, *Peer, *Peer) {
	t.Helper()
	clientConn, brokerConn := net.Pipe()
	brokerScannerConn, scannerConn := net.Pipe()
	b := &Broker{codec: NewCodec(brokerConn), startedAt: time.Now()}
	client := NewPeer(NewCodec(clientConn))
	brokerPeer := NewPeer(b.codec)
	scanner := NewPeer(NewCodec(scannerConn))
	sp := &scannerProcess{stdin: brokerScannerConn}
	sp.peer = jsonrpc.NewPeerWithWriteInterrupt(NewCodec(brokerScannerConn), sp.interruptWrite)
	b.setScanner(sp)
	go client.Serve(nil, nil)
	go brokerPeer.Serve(b.handleMessage, nil)
	go sp.peer.Serve(nil, nil)
	go scanner.Serve(func(msg *Message) { onScannerRequest(scanner, msg) }, nil)
	t.Cleanup(func() {
		client.Close()
		brokerPeer.Close()
		sp.peer.Close()
		scanner.Close()
		_ = clientConn.Close()
		_ = brokerConn.Close()
		_ = brokerScannerConn.Close()
		_ = scannerConn.Close()
	})
	return b, client, scanner
}

func brokerNodeInfoCall(t *testing.T, client *Peer, params string) (json.RawMessage, *RPCError) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	result, rpcErr, err := client.Call(ctx, "discovery:get-node-info", json.RawMessage(params))
	if err != nil {
		t.Fatalf("broker RPC transport: %v", err)
	}
	return result, rpcErr
}

func TestNodeInfoRPCRegisteredRelayPreservesFullSample(t *testing.T) {
	const sample = `{"GPUs":[{"name":"peer GPU","vram_bytes":8192,"vram_used_bytes":1024,"utilization_percent":73}],"cpu":{"name":"peer CPU","cores":8,"utilization_percent":42},"memory":{"total_bytes":16384,"used_bytes":4096},"telemetryValid":true,"msSince":125,"hostUuid":"physical-host","clusterUuid":"principal"}`
	_, client, _ := nodeInfoBrokerTestClient(t, func(scanner *Peer, msg *Message) {
		if msg.Method != "discovery:get-node-info" {
			t.Errorf("scanner method = %q", msg.Method)
		}
		var p nodeInfoRPCParams
		if err := json.Unmarshal(msg.Params, &p); err != nil || p.HostUUID != "physical-host" {
			t.Errorf("scanner params = %s: %v", msg.Params, err)
		}
		if err := scanner.Respond(msg.ID, json.RawMessage(sample)); err != nil {
			t.Error(err)
		}
	})
	result, rpcErr := brokerNodeInfoCall(t, client, `{"hostUuid":"physical-host"}`)
	if rpcErr != nil {
		t.Fatalf("broker getter rejected: %+v", rpcErr)
	}
	var got, want map[string]json.RawMessage
	if err := json.Unmarshal(result, &got); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(sample), &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("relay changed or wrapped full sample: got %s want %s", result, sample)
	}
}

func TestNodeInfoRPCRelayPreservesScannerError(t *testing.T) {
	_, client, _ := nodeInfoBrokerTestClient(t, func(scanner *Peer, msg *Message) {
		if err := scanner.RespondErrorData(msg.ID, -32017, "authority lost", map[string]string{"reason": "removed"}); err != nil {
			t.Error(err)
		}
	})
	result, rpcErr := brokerNodeInfoCall(t, client, `{"hostUuid":"peer"}`)
	if rpcErr == nil || rpcErr.Code != -32017 || rpcErr.Message != "authority lost" || len(result) != 0 {
		t.Fatalf("scanner error changed to success/different error: %s %+v", result, rpcErr)
	}
	var data map[string]string
	if err := json.Unmarshal(rpcErr.Data, &data); err != nil || !reflect.DeepEqual(data, map[string]string{"reason": "removed"}) {
		t.Fatalf("scanner error data = %s: %v", rpcErr.Data, err)
	}
}

func TestNodeInfoRPCRelayRejectsInvalidParamsBeforeScanner(t *testing.T) {
	var requests atomic.Int32
	_, client, _ := nodeInfoBrokerTestClient(t, func(scanner *Peer, msg *Message) {
		requests.Add(1)
		if err := scanner.Respond(msg.ID, map[string]bool{"unexpected": true}); err != nil {
			t.Error(err)
		}
	})
	for _, params := range []string{`null`, `{}`, `[]`, `{"hostUuid":false}`, `{"hostUuid":" "}`, `{"hostUuid":"peer","url":"http://127.0.0.1"}`, `{"hostUuid":"peer","port":80}`, `{"hostUuid":"peer","principal":"other"}`} {
		t.Run(params, func(t *testing.T) {
			result, rpcErr := brokerNodeInfoCall(t, client, params)
			if rpcErr == nil || rpcErr.Code != -32602 || len(result) != 0 {
				t.Fatalf("invalid params: result=%s error=%+v", result, rpcErr)
			}
		})
	}
	if requests.Load() != 0 {
		t.Fatalf("invalid params reached scanner %d times", requests.Load())
	}
}

func TestNodeInfoRPCRelayUnavailableWorker(t *testing.T) {
	test := func(name string, change func(*Broker)) {
		t.Run(name, func(t *testing.T) {
			b, client, _ := nodeInfoBrokerTestClient(t, func(_ *Peer, _ *Message) { t.Error("unavailable scanner was called") })
			change(b)
			result, rpcErr := brokerNodeInfoCall(t, client, `{"hostUuid":"peer"}`)
			if rpcErr == nil || rpcErr.Code != -32000 || len(result) != 0 {
				t.Fatalf("unavailable worker succeeded: %s %+v", result, rpcErr)
			}
		})
	}
	test("missing", func(b *Broker) { b.setScanner(nil) })
	test("closed", func(b *Broker) { b.getScanner().peer.Close() })
}

func TestNodeInfoRPCRelayConcurrentCorrelationAndReadPump(t *testing.T) {
	requests := make(chan *Message, 2)
	_, client, scanner := nodeInfoBrokerTestClient(t, func(_ *Peer, msg *Message) { requests <- msg })
	type response struct {
		host   string
		result json.RawMessage
		rpcErr *RPCError
		err    error
	}
	responses := make(chan response, 2)
	for _, host := range []string{"alpha", "beta"} {
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			params, err := json.Marshal(nodeInfoRPCParams{HostUUID: host})
			if err != nil {
				responses <- response{host: host, err: err}
				return
			}
			result, rpcErr, err := client.Call(ctx, "discovery:get-node-info", params)
			responses <- response{host, result, rpcErr, err}
		}()
	}
	var pending []*Message
	for range 2 {
		select {
		case msg := <-requests:
			pending = append(pending, msg)
		case <-time.After(time.Second):
			t.Fatal("concurrent requests did not reach scanner")
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	result, rpcErr, err := client.Call(ctx, "ping", nil)
	if err != nil || rpcErr != nil {
		t.Fatalf("broker read pump blocked: %v %+v", err, rpcErr)
	}
	var pong PingResult
	if err := json.Unmarshal(result, &pong); err != nil || !pong.Pong {
		t.Fatalf("concurrent ping = %s: %v", result, err)
	}
	// Reply out of order: the original local IDs must still receive their sample.
	for i := len(pending) - 1; i >= 0; i-- {
		msg := pending[i]
		var p nodeInfoRPCParams
		if err := json.Unmarshal(msg.Params, &p); err != nil {
			t.Fatal(err)
		}
		if err := scanner.Respond(msg.ID, map[string]string{"hostUuid": p.HostUUID}); err != nil {
			t.Fatal(err)
		}
	}
	seen := make(map[string]bool)
	for range 2 {
		select {
		case response := <-responses:
			var got nodeInfoRPCParams
			if err := json.Unmarshal(response.result, &got); err != nil {
				t.Fatalf("decode %s: %v", response.host, err)
			}
			if response.err != nil || response.rpcErr != nil || got.HostUUID != response.host || seen[response.host] {
				t.Fatalf("mis-correlated response: %+v sample=%+v", response, got)
			}
			seen[response.host] = true
		case <-time.After(time.Second):
			t.Fatal("missing concurrent response")
		}
	}
}

func TestNodeInfoRPCRelayDeadlineDisownsLateReply(t *testing.T) {
	// Allow 2.5 s of scheduling slack over the 1.5 s relay budget, while
	// still requiring a broker response 1 s before the caller's deadline.
	const relayLimit = 4 * time.Second
	requests := make(chan *Message, 2)
	b, client, scanner := nodeInfoBrokerTestClient(t, func(_ *Peer, msg *Message) { requests <- msg })
	type response struct {
		result json.RawMessage
		rpcErr *RPCError
		err    error
	}
	done := make(chan response, 1)
	started := time.Now()
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), relayLimit+time.Second)
		defer cancel()
		result, rpcErr, err := client.Call(ctx, "discovery:get-node-info", json.RawMessage(`{"hostUuid":"old"}`))
		done <- response{result, rpcErr, err}
	}()
	var late *Message
	select {
	case late = <-requests:
	case <-time.After(time.Second):
		t.Fatal("timed request never reached scanner")
	}
	select {
	case got := <-done:
		elapsed := time.Since(started)
		if got.err != nil || got.rpcErr == nil || got.rpcErr.Code != -32000 || len(got.result) != 0 || elapsed < time.Second || elapsed > relayLimit {
			t.Fatalf("relay deadline: %+v elapsed=%s", got, elapsed)
		}
	case <-time.After(relayLimit):
		t.Fatal("relay exceeded deadline")
	}
	select {
	case <-b.getScanner().peer.Done():
		t.Fatal("response-only timeout retired a healthy scanner")
	default:
	}
	if err := scanner.Respond(late.ID, map[string]string{"hostUuid": "old"}); err != nil {
		t.Fatal(err)
	}
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		result, rpcErr, err := client.Call(ctx, "discovery:get-node-info", json.RawMessage(`{"hostUuid":"new"}`))
		done <- response{result, rpcErr, err}
	}()
	select {
	case msg := <-requests:
		if err := scanner.Respond(msg.ID, map[string]string{"hostUuid": "new"}); err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("new request never reached scanner")
	}
	select {
	case got := <-done:
		var sample nodeInfoRPCParams
		if err := json.Unmarshal(got.result, &sample); err != nil || got.err != nil || got.rpcErr != nil || sample.HostUUID != "new" {
			t.Fatalf("late reply contaminated new request: %+v sample=%+v decode=%v", got, sample, err)
		}
	case <-time.After(time.Second):
		t.Fatal("new request did not finish")
	}
}

type nodeInfoWriteProbe struct {
	net.Conn
	started, returned     chan struct{}
	startOnce, returnOnce sync.Once
	writes                atomic.Int32
}

func (p *nodeInfoWriteProbe) Write(data []byte) (int, error) {
	p.writes.Add(1)
	p.startOnce.Do(func() { close(p.started) })
	n, err := p.Conn.Write(data)
	p.returnOnce.Do(func() { close(p.returned) })
	return n, err
}

func TestNodeInfoRPCActiveInputHonorsCallerContext(t *testing.T) {
	scannerInput, scannerConn := net.Pipe()
	t.Cleanup(func() { _ = scannerInput.Close(); _ = scannerConn.Close() })
	probe := &nodeInfoWriteProbe{Conn: scannerInput, started: make(chan struct{}), returned: make(chan struct{})}
	sp := &scannerProcess{stdin: probe}
	sp.peer = jsonrpc.NewPeerWithWriteInterrupt(NewCodec(probe), sp.interruptWrite)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	started := time.Now()
	_, _, err := sp.getNodeInfo(ctx, json.RawMessage(`{"hostUuid":"peer"}`))
	if elapsed := time.Since(started); !errors.Is(err, context.DeadlineExceeded) || elapsed > 250*time.Millisecond || probe.writes.Load() != 1 {
		t.Fatalf("active write outlived caller: err=%v elapsed=%s writes=%d", err, elapsed, probe.writes.Load())
	}
	select {
	case <-probe.returned:
	default:
		t.Fatal("Call abandoned the blocked writer")
	}
	select {
	case <-sp.peer.Done():
	default:
		t.Fatal("interrupted peer was not invalidated")
	}
}

func TestNodeInfoRPCStalledInputRetiresScannerAndUnwindsWrite(t *testing.T) {
	clientConn, brokerConn := net.Pipe()
	scannerInput, scannerConn := net.Pipe()
	probe := &nodeInfoWriteProbe{Conn: scannerInput, started: make(chan struct{}), returned: make(chan struct{})}
	sp := &scannerProcess{stdin: probe}
	sp.peer = jsonrpc.NewPeerWithWriteInterrupt(NewCodec(probe), sp.interruptWrite)
	b := &Broker{codec: NewCodec(brokerConn), startedAt: time.Now()}
	b.setScanner(sp)
	client, brokerPeer := NewPeer(NewCodec(clientConn)), NewPeer(b.codec)
	scannerPumpDone, brokerPumpDone, clientPumpDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
	callerReturned := make(chan struct{})
	go func() { sp.peer.Serve(nil, nil); close(scannerPumpDone) }()
	go func() { brokerPeer.Serve(b.handleMessage, nil); close(brokerPumpDone) }()
	go func() { client.Serve(nil, nil); close(clientPumpDone) }()
	t.Cleanup(func() {
		_ = scannerInput.Close()
		_ = scannerConn.Close()
		_ = brokerConn.Close()
		_ = clientConn.Close()
		for _, done := range []chan struct{}{scannerPumpDone, brokerPumpDone, clientPumpDone, callerReturned} {
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Error("peer read pump did not unwind")
			}
		}
	})
	type response struct {
		result json.RawMessage
		rpcErr *RPCError
		err    error
	}
	done := make(chan response, 1)
	started := time.Now()
	go func() {
		defer close(callerReturned)
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		result, rpcErr, err := client.Call(ctx, "discovery:get-node-info", json.RawMessage(`{"hostUuid":"peer"}`))
		done <- response{result, rpcErr, err}
	}()
	select {
	case <-probe.started:
	case <-time.After(time.Second):
		t.Fatal("scanner input write never started")
	}
	// The scanner deliberately never reads its input, while the broker still
	// answers independent requests from this same client connection.
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	if _, rpcErr, err := client.Call(ctx, "ping", nil); err != nil || rpcErr != nil {
		t.Fatalf("blocked scanner input stalled broker read pump: %v %+v", err, rpcErr)
	}
	select {
	case got := <-done:
		elapsed := time.Since(started)
		if got.err != nil || got.rpcErr == nil || got.rpcErr.Code != -32000 || len(got.result) != 0 || elapsed < time.Second || elapsed > 2500*time.Millisecond {
			t.Fatalf("stalled write deadline: %+v elapsed=%s", got, elapsed)
		}
	case <-time.After(2500 * time.Millisecond):
		t.Fatal("scanner input write outlived the RPC budget")
	}
	select {
	case <-probe.returned:
	default:
		t.Fatal("RPC returned before blocked Write unwound")
	}
	select {
	case <-sp.peer.Done():
	default:
		t.Fatal("partially written scanner peer remains reusable")
	}
	select {
	case <-scannerPumpDone:
	case <-time.After(time.Second):
		t.Fatal("scanner input interruption left its read pump running")
	}
	if _, _, err := sp.getNodeInfo(context.Background(), json.RawMessage(`{"hostUuid":"peer"}`)); !errors.Is(err, jsonrpc.ErrPeerClosed) || probe.writes.Load() != 1 {
		t.Fatalf("retired peer wrote again: err=%v writes=%d", err, probe.writes.Load())
	}
	replacement, _, _ := nodeInfoBrokerTestClient(t, func(scanner *Peer, msg *Message) {
		if err := scanner.Respond(msg.ID, map[string]string{"hostUuid": "recovered"}); err != nil {
			t.Error(err)
		}
	})
	b.setScanner(replacement.getScanner())
	result, rpcErr := brokerNodeInfoCall(t, client, `{"hostUuid":"recovered"}`)
	var sample nodeInfoRPCParams
	if err := json.Unmarshal(result, &sample); err != nil || rpcErr != nil || sample.HostUUID != "recovered" {
		t.Fatalf("replacement scanner did not recover: result=%s error=%+v decode=%v", result, rpcErr, err)
	}
}

func TestNodeInfoRPCQueuedInputDeadlinePreservesScanner(t *testing.T) {
	scannerInput, scannerConn := net.Pipe()
	probe := &nodeInfoWriteProbe{Conn: scannerInput, started: make(chan struct{}), returned: make(chan struct{})}
	sp := &scannerProcess{stdin: probe}
	sp.peer = jsonrpc.NewPeerWithWriteInterrupt(NewCodec(probe), sp.interruptWrite)
	pumpDone := make(chan struct{})
	var scannerPumpDone chan struct{}
	go func() { sp.peer.Serve(nil, nil); close(pumpDone) }()
	t.Cleanup(func() {
		_ = scannerInput.Close()
		_ = scannerConn.Close()
		select {
		case <-pumpDone:
		case <-time.After(time.Second):
			t.Error("scanner read pump did not unwind")
		}
		if scannerPumpDone != nil {
			select {
			case <-scannerPumpDone:
			case <-time.After(time.Second):
				t.Error("resumed scanner fixture did not unwind")
			}
		}
	})
	writerDone := make(chan error, 1)
	writerReturned := make(chan struct{})
	go func() {
		defer close(writerReturned)
		writerDone <- sp.peer.Notify("fixture:holds-write-gate", nil)
	}()
	t.Cleanup(func() {
		_ = scannerInput.Close()
		select {
		case <-writerReturned:
		case <-time.After(time.Second):
			t.Error("gate-holding writer did not unwind")
		}
	})
	select {
	case <-probe.started:
	case <-time.After(time.Second):
		t.Fatal("fixture did not hold the scanner write gate")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	started := time.Now()
	_, _, err := sp.getNodeInfo(ctx, json.RawMessage(`{"hostUuid":"peer"}`))
	if !errors.Is(err, context.DeadlineExceeded) || time.Since(started) > 250*time.Millisecond || probe.writes.Load() != 1 {
		t.Fatalf("queued write escaped deadline: err=%v elapsed=%s writes=%d", err, time.Since(started), probe.writes.Load())
	}
	select {
	case <-sp.peer.Done():
		t.Fatal("queued request cancellation retired scanner")
	default:
	}
	scanner := NewPeer(NewCodec(scannerConn))
	scannerPumpDone = make(chan struct{})
	go func() {
		defer close(scannerPumpDone)
		scanner.Serve(func(msg *Message) {
			if err := scanner.Respond(msg.ID, map[string]string{"hostUuid": "peer"}); err != nil {
				t.Error(err)
			}
		}, nil)
	}()
	select {
	case err := <-writerDone:
		if err != nil {
			t.Fatalf("original writer failed after queue cancellation: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("original writer did not recover after scanner resumed reading")
	}
	recoveryCtx, recoveryCancel := context.WithTimeout(context.Background(), time.Second)
	defer recoveryCancel()
	result, rpcErr, err := sp.getNodeInfo(recoveryCtx, json.RawMessage(`{"hostUuid":"peer"}`))
	var sample nodeInfoRPCParams
	if decodeErr := json.Unmarshal(result, &sample); decodeErr != nil || err != nil || rpcErr != nil || sample.HostUUID != "peer" || probe.writes.Load() != 2 {
		t.Fatalf("queued cancellation contaminated recovery: result=%s err=%v rpc=%+v decode=%v writes=%d", result, err, rpcErr, decodeErr, probe.writes.Load())
	}
}

func TestNodeInfoRPCStalledProcessIsReapedAndReplaced(t *testing.T) {
	// This owned child ignores stdin until replaced. Its responsive mode reads
	// the same newline JSON-RPC pipe startScanner uses in production.
	dir := t.TempDir()
	source := filepath.Join(dir, "scanner.go")
	binary := filepath.Join(testbin.Dir(t, "nvpair-ui-broker", "node-info-rpc", "scanner-fixture"), testbin.Exe("scanner-fixture"))
	const helper = `// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package main

import (
	"encoding/json"
	"os"
	"time"
)

func main() {
	responsive := false
	for _, arg := range os.Args[1:] {
		if arg == "responsive" {
			responsive = true
		}
	}
	if !responsive {
		for {
			time.Sleep(time.Hour)
		}
	}
	decoder, encoder := json.NewDecoder(os.Stdin), json.NewEncoder(os.Stdout)
	for {
		var request map[string]json.RawMessage
		if decoder.Decode(&request) != nil {
			return
		}
		id := request["id"]
		if len(id) == 0 {
			continue
		}
		response := map[string]any{"jsonrpc": "2.0", "id": id, "result": map[string]string{"hostUuid": "peer"}}
		if encoder.Encode(response) != nil {
			return
		}
	}
}
`
	if err := os.WriteFile(source, []byte(helper), 0600); err != nil {
		t.Fatal(err)
	}
	buildCtx, buildCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer buildCancel()
	if output, err := exec.CommandContext(buildCtx, "go", "build", "-o", binary, source).CombinedOutput(); err != nil {
		t.Fatalf("build owned scanner fixture: %v\n%s", err, output)
	}
	sp, err := startScanner(binary, "error", nil, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(sp.Stop)
	params, err := json.Marshal(nodeInfoRPCParams{HostUUID: strings.Repeat("p", 1<<20)})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	started := time.Now()
	_, _, err = sp.getNodeInfo(ctx, params)
	if !errors.Is(err, context.DeadlineExceeded) || time.Since(started) > time.Second {
		t.Fatalf("owned process input write escaped deadline: err=%v elapsed=%s", err, time.Since(started))
	}
	select {
	case <-sp.Done():
	case <-time.After(time.Second):
		t.Fatal("interrupted scanner process was not reaped")
	}
	if sp.cmd.ProcessState == nil || sp.cmd.ProcessState.Success() {
		t.Fatalf("stalled scanner did not terminate unsuccessfully: %v", sp.cmd.ProcessState)
	}
	replacement, err := startScanner(binary, "error", nil, nil, nil, nil, nil, nil, "responsive")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(replacement.Stop)
	// The budget covers the replacement's process startup, unlike the
	// production relay budget, which applies to a running scanner. This test
	// checks recovery, not latency.
	recoveryCtx, recoveryCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer recoveryCancel()
	result, rpcErr, err := replacement.getNodeInfo(recoveryCtx, json.RawMessage(`{"hostUuid":"peer"}`))
	var sample nodeInfoRPCParams
	if decodeErr := json.Unmarshal(result, &sample); decodeErr != nil || err != nil || rpcErr != nil || sample.HostUUID != "peer" {
		t.Fatalf("replacement process did not recover: result=%s err=%v rpc=%+v decode=%v", result, err, rpcErr, decodeErr)
	}
	replacement.Stop()
	select {
	case <-replacement.Done():
	default:
		t.Fatal("replacement process remained after graceful stop")
	}
}
