// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/testbin"
)

// Holds one complete ordinary fixture frame before service input. The next
// broker Write then blocks on its actual stdout pipe while the engine pump
// continues. No service/broker control or reply is replaced by this reader.
type engineResetBlockedReader struct {
	reader           *bufio.Reader
	frame            []byte
	entered, release chan struct{}
	once             sync.Once
}

func (r *engineResetBlockedReader) Read(raw []byte) (int, error) {
	if len(r.frame) == 0 {
		frame, err := r.reader.ReadBytes('\n')
		if err != nil {
			return 0, err
		}
		var msg jsonrpc.Message
		if err := json.Unmarshal(frame, &msg); err != nil {
			return 0, err
		}
		if msg.Method == "engine:pull-progress" {
			r.once.Do(func() { close(r.entered); <-r.release })
		}
		r.frame = frame
	}
	n := copy(raw, r.frame)
	r.frame = r.frame[n:]
	return n, nil
}

func waitEngineResetSignal(t *testing.T, signal <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(5 * time.Second):
		t.Fatalf("engine reset boundary not reached: %s", what)
	}
}

// The broker test image runs the real engine Peer callback, queue and sole
// sender. Compile it from this checkout, never consume a potentially stale
// previously built image. Windows uses the already covered root package image;
// parent-owned gates must remain serialized while this fixture owns that path.
func engineResetBrokerImage(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if root := os.Getenv(testbin.EnvRoot); root != "" {
		if !filepath.IsAbs(root) {
			t.Fatal("NVPAIR_TEST_BIN_DIR must be absolute")
		}
		dir = filepath.Join(root, "go-test", "nvpair-ui-broker")
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}
	image := filepath.Join(dir, testbin.Exe("nvpair-ui-broker.test"))
	brokerDir, err := filepath.Abs(filepath.Join("..", "nvpair-ui-broker"))
	if err != nil {
		t.Fatal(err)
	}
	build := exec.Command("go", "test", "-c", "-o", image, ".")
	build.Dir = brokerDir
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build exact broker relay fixture: %v\n%s", err, output)
	}
	return image
}

func TestEngineNotificationResetBrokerToServiceRecovery(t *testing.T) {
	cmd := exec.Command(engineResetBrokerImage(t), "--engine-notification-service-fixture")
	input, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	output, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	gate := &engineResetBlockedReader{reader: bufio.NewReader(output), entered: make(chan struct{}), release: make(chan struct{})}
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(gate.release) }) }
	var kills atomic.Int32
	h := &historyTestHarness{s: newServiceWithHistory(config{})}
	h.run = &brokerRun{proc: &brokerProc{stdin: input, stdout: gate, pid: cmd.Process.Pid,
		kill: func() error { kills.Add(1); return cmd.Process.Kill() }}, exited: make(chan struct{})}
	h.s.broker = h.run
	h.s.startHistory()
	pumped := make(chan struct{})
	go func() { defer close(pumped); h.s.pumpStdout(h.run) }()
	waited := make(chan struct{})
	var waitErr error
	go func() { waitErr = cmd.Wait(); close(h.run.exited); close(waited) }()
	drained, stderrDone := make(chan struct{}), make(chan struct{})
	var stderrErr error
	go func() {
		defer close(stderrDone)
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			if scanner.Text() == "engine-drained" {
				close(drained)
			}
		}
		stderrErr = scanner.Err()
	}()
	t.Cleanup(func() {
		release()
		for _, p := range h.peers {
			_ = p.conn.Close()
		}
		_ = input.Close()
		select {
		case <-waited:
		case <-time.After(5 * time.Second):
			t.Error("owned relay fixture did not exit after stdin EOF")
			_ = cmd.Process.Kill()
			<-waited
		}
		_ = output.Close()
		_ = stderr.Close()
		close(h.s.stopCh)
		joined := make(chan struct{})
		go func() { h.s.history.wg.Wait(); <-pumped; <-stderrDone; close(joined) }()
		waitEngineResetSignal(t, joined, "service and fixture output borrowers joined")
		if waitErr != nil || (stderrErr != nil && !errors.Is(stderrErr, os.ErrClosed)) {
			t.Errorf("fixture exit/output: %v / %v", waitErr, stderrErr)
		}
	})
	first := h.peer()
	ready := first.next(t)
	if ready.Method != methodAppReady {
		t.Fatalf("initial ready = %+v", ready)
	}
	passive := h.peer()
	if msg := passive.next(t); msg.Method != methodAppReady {
		t.Fatalf("passive attachment ready = %+v", msg)
	}
	first.send(t, `{"jsonrpc":"2.0","id":1,"method":"engine:subscribe"}`)
	if reply := first.next(t); reply.Error != nil || reply.ID == nil || string(*reply.ID) != "1" {
		t.Fatalf("initial subscription = %+v", reply)
	}
	first.send(t, `{"jsonrpc":"2.0","method":"test:overflow"}`)
	waitEngineResetSignal(t, gate.entered, "service stdout reader physically paused")
	waitEngineResetSignal(t, drained, "engine notifications handled while service reader remains paused")
	h.s.mu.Lock()
	beforeReset := len(h.s.clients) == 2 && h.s.subscribed["engine:subscribe"] && h.s.broker == h.run
	var affected []*client
	for c := range h.s.clients {
		affected = append(affected, c)
	}
	h.s.mu.Unlock()
	if !beforeReset {
		t.Fatal("overflow changed service lifetimes before reset could be read")
	}
	for _, p := range []*historyTestPeer{first, passive} {
		if err := p.conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
			t.Fatal(err)
		}
	}
	release()
	for _, p := range []*historyTestPeer{first, passive} {
		for {
			raw, err := p.reader.ReadBytes('\n')
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatalf("affected client was not disconnected: %v", err)
			}
			var msg jsonrpc.Message
			if err := json.Unmarshal(raw, &msg); err != nil {
				t.Fatal(err)
			}
			if msg.Method == notifyEngineNotificationsReset || msg.Method == methodAppReady {
				t.Fatalf("private reset leaked or fabricated ready: %+v", msg)
			}
		}
	}
	for _, c := range affected {
		waitEngineResetSignal(t, c.written, "detached ordinary client writer joined")
	}
	second := h.peer()
	currentReady := second.next(t)
	if currentReady.Method != methodAppReady || !bytes.Equal(currentReady.Params, ready.Params) {
		t.Fatalf("reconnect did not retain actual broker ready: %+v", currentReady)
	}
	second.send(t, `{"jsonrpc":"2.0","id":2,"method":"engine:subscribe"}`)
	if reply := second.next(t); reply.Error != nil || reply.ID == nil || string(*reply.ID) != "2" {
		t.Fatalf("reconnect subscription = %+v", reply)
	}
	second.send(t, `{"jsonrpc":"2.0","id":3,"method":"engine:models"}`)
	state := second.next(t)
	var models struct {
		Models []string `json:"models"`
	}
	if err := json.Unmarshal(state.Result, &models); err != nil {
		t.Fatal(err)
	}
	if state.Error != nil || !reflect.DeepEqual(models.Models, []string{"after-reset"}) {
		t.Fatalf("reconnect initial state = %+v / %+v", state, models)
	}
	second.send(t, `{"jsonrpc":"2.0","id":4,"method":"test:engine-alive"}`)
	engine := second.next(t)
	var alive struct {
		Alive bool `json:"alive"`
	}
	if err := json.Unmarshal(engine.Result, &alive); err != nil {
		t.Fatal(err)
	}
	if engine.Error != nil || !alive.Alive || kills.Load() != 0 {
		t.Fatalf("engine/broker did not survive: %+v / kills=%d", engine, kills.Load())
	}
	select {
	case <-h.run.exited:
		t.Fatal("overflow exited the broker")
	default:
	}
	h.s.mu.Lock()
	recovered := h.s.broker == h.run && !h.s.stopping && h.s.restarts == 0 && h.s.subscribed["engine:subscribe"]
	h.s.mu.Unlock()
	if !recovered {
		t.Fatal("subscription did not recover on the original live broker run")
	}
}

func TestEngineNotificationResetClearsOnlyEngineOwners(t *testing.T) {
	s := newService(config{})
	run := &brokerRun{exited: make(chan struct{})}
	s.broker = run
	s.readyFrame = []byte("actual-ready")
	s.baselines["engine:ready"] = []byte("old-engine-ready")
	s.baselines["ollama-proxy:ready"] = []byte("proxy-ready")
	s.subscribed["engine:subscribe"], s.subscribed["discovery:subscribe"] = true, true
	s.resubscribe = []string{"engine:subscribe", "discovery:subscribe"}
	completed := make(chan *jsonrpc.Message, 1)
	s.pending[1] = &pendingRequest{run: run, method: "engine:subscribe", internal: func(m *jsonrpc.Message) { completed <- m }}
	other := &pendingRequest{run: run, method: "unrelated", internal: func(*jsonrpc.Message) { t.Error("unrelated request was completed by reset") }}
	s.pending[2] = other
	server, peer := net.Pipe()
	defer peer.Close()
	c := newClient(1, server)
	defer c.closeNow()
	s.clients[c] = struct{}{}
	s.pending[3] = &pendingRequest{run: run, method: "engine:subscribe", client: c, origID: json.RawMessage(`3`)}
	s.resetEngineNotifications(run)
	select {
	case msg := <-completed:
		if msg != nil {
			t.Fatal("invalidated subscription succeeded")
		}
	default:
		t.Fatal("reset did not complete the internal pending subscription")
	}
	waitEngineResetSignal(t, c.written, "reset client writer joined")
	s.routeResponse(&jsonrpc.Message{JSONRPC: "2.0", ID: rawID(1), Result: json.RawMessage(`{"subscribed":true}`)})
	s.routeResponse(&jsonrpc.Message{JSONRPC: "2.0", ID: rawID(3), Result: json.RawMessage(`{"subscribed":true}`)})
	// A reader already dispatching as its lifetime was withdrawn can register
	// a request late. Its acknowledgement cannot restore shared cache state.
	s.pending[4] = &pendingRequest{run: run, method: "engine:subscribe", client: c, origID: json.RawMessage(`4`)}
	s.routeResponse(&jsonrpc.Message{JSONRPC: "2.0", ID: rawID(4), Result: json.RawMessage(`{"subscribed":true}`)})
	if s.subscribed["engine:subscribe"] || s.baselines["engine:ready"] != nil || s.pending[1] != nil || s.pending[3] != nil || len(s.clients) != 0 {
		t.Fatal("late subscription acknowledgement restored invalid engine ownership")
	}
	if s.pending[2] != other || !s.subscribed["discovery:subscribe"] || string(s.baselines["ollama-proxy:ready"]) != "proxy-ready" || string(s.readyFrame) != "actual-ready" || !reflect.DeepEqual(s.resubscribe, []string{"discovery:subscribe"}) {
		t.Fatal("reset changed unrelated pending, baseline, ready or subscription ownership")
	}
}

func TestEngineNotificationResetRejectsRetiredRun(t *testing.T) {
	h := newHistoryTestHarness(t)
	p := h.peer()
	h.s.mu.Lock()
	h.s.subscribed["engine:subscribe"] = true
	h.s.mu.Unlock()
	old := &brokerRun{exited: make(chan struct{})}
	h.s.handleBrokerNotification(old, notifyEngineNotificationsReset, []byte("private"))
	h.s.mu.Lock()
	unchanged := h.s.broker == h.run && len(h.s.clients) == 1 && h.s.subscribed["engine:subscribe"]
	h.s.mu.Unlock()
	if !unchanged {
		t.Fatal("retired broker reset current client/subscription")
	}
	h.emit(t, []byte(`{"jsonrpc":"2.0","method":"ordinary:probe"}`))
	if msg := p.next(t); msg.Method != "ordinary:probe" {
		t.Fatalf("stale private reset leaked: %+v", msg)
	}
}

func TestEngineNotificationResetRejectsClientInjection(t *testing.T) {
	h := newHistoryTestHarness(t)
	p := h.peer()
	p.send(t, `{"jsonrpc":"2.0","id":1,"method":"service/engine-notifications-reset"}`)
	if msg := p.next(t); msg.Error == nil || msg.Error.Code != codeMethodNotFound {
		t.Fatalf("private request admitted: %+v", msg)
	}
	p.send(t, `{"jsonrpc":"2.0","method":"service/engine-notifications-reset"}`)
	p.send(t, `{"jsonrpc":"2.0","id":2,"method":"ordinary:probe"}`)
	request := h.request(t)
	if request.Method != "ordinary:probe" {
		t.Fatalf("client reset reached broker: %+v", request)
	}
	h.emit(t, historyTestJSON(t, &jsonrpc.Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`null`)}))
	if msg := p.next(t); msg.Error != nil || msg.ID == nil || string(*msg.ID) != "2" {
		t.Fatalf("client reset altered lifetime: %+v", msg)
	}
}

type engineResetFailingInput struct{ entered, release chan struct{} }

func (w *engineResetFailingInput) Write([]byte) (int, error) {
	close(w.entered)
	<-w.release
	return 0, io.ErrClosedPipe
}
func (w *engineResetFailingInput) Close() error { return nil }

func TestEngineNotificationResetJoinsPendingWriteFailure(t *testing.T) {
	s := newService(config{})
	input := &engineResetFailingInput{entered: make(chan struct{}), release: make(chan struct{})}
	run := &brokerRun{proc: &brokerProc{stdin: input}, exited: make(chan struct{})}
	s.broker = run
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(input.release) }) }
	called := make(chan struct{})
	t.Cleanup(func() { release(); waitEngineResetSignal(t, called, "internal subscribe caller joined") })
	result := make(chan (<-chan *jsonrpc.Message), 1)
	go func() { defer close(called); result <- s.callBroker(run, "engine:subscribe") }()
	waitEngineResetSignal(t, input.entered, "internal subscribe write admitted")
	s.resetEngineNotifications(run)
	release()
	select {
	case response := <-result:
		if msg := <-response; msg != nil {
			t.Fatal("reset subscription succeeded")
		}
		select {
		case <-response:
			t.Fatal("pending subscription completed twice")
		default:
		}
	case <-time.After(5 * time.Second):
		t.Fatal("reset/write-failure pending owner did not join")
	}
	if len(s.pending) != 0 {
		t.Fatal("reset retained pending subscription")
	}
}
