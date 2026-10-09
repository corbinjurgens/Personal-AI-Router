// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/servicectl"
)

const waitTimeout = 5 * time.Second

// fakeBroker stands in for nvpair-ui-broker over in-process pipes. It speaks
// enough of the broker's stdio contract to exercise the service: app:ready on
// start, idempotent subscriptions with a one-time baseline, and shutdown.
type fakeBroker struct {
	gen int

	stdinR  *io.PipeReader
	stdinW  *io.PipeWriter
	stdoutR *io.PipeReader
	stdoutW *io.PipeWriter
	stderrR *io.PipeReader
	stderrW *io.PipeWriter

	// ignoreShutdown makes the broker deaf to shutdown and stdin EOF, so
	// only a kill ends it.
	ignoreShutdown bool

	wmu      sync.Mutex
	exited   bool
	mu       sync.Mutex
	frames   []jsonrpc.Message
	subbed   map[string]bool
	exitOnce sync.Once
	exitCh   chan int
	killed   chan struct{}
}

func newFakeBroker(gen int, ignoreShutdown bool) *fakeBroker {
	b := &fakeBroker{gen: gen, ignoreShutdown: ignoreShutdown, subbed: map[string]bool{},
		exitCh: make(chan int, 1), killed: make(chan struct{})}
	b.stdinR, b.stdinW = io.Pipe()
	b.stdoutR, b.stdoutW = io.Pipe()
	b.stderrR, b.stderrW = io.Pipe()
	return b
}

func (b *fakeBroker) proc() *brokerProc {
	return &brokerProc{
		stdin:  b.stdinW,
		stdout: b.stdoutR,
		stderr: b.stderrR,
		pid:    1000 + b.gen,
		wait:   func() int { return <-b.exitCh },
		kill: func() error {
			b.exitOnce.Do(func() { close(b.killed) })
			b.exit(-1)
			return nil
		},
		closeOutput: func() {
			_ = b.stdoutR.Close()
			_ = b.stderrR.Close()
		},
	}
}

func (b *fakeBroker) exit(code int) {
	b.wmu.Lock()
	defer b.wmu.Unlock()
	if b.exited {
		return
	}
	b.exited = true
	_ = b.stdoutW.Close()
	_ = b.stderrW.Close()
	_ = b.stdinR.Close()
	b.exitCh <- code
}

func (b *fakeBroker) write(format string, a ...any) {
	b.wmu.Lock()
	defer b.wmu.Unlock()
	if b.exited {
		return
	}
	_, _ = fmt.Fprintf(b.stdoutW, format+"\n", a...)
}

func (b *fakeBroker) run() {
	b.write(`{"jsonrpc":"2.0","method":"app:ready","params":{"version":"fake-%d"}}`, b.gen)
	sc := bufio.NewScanner(b.stdinR)
	for sc.Scan() {
		var m jsonrpc.Message
		if json.Unmarshal(sc.Bytes(), &m) != nil {
			continue
		}
		b.mu.Lock()
		b.frames = append(b.frames, m)
		b.mu.Unlock()
		if m.ID == nil {
			continue
		}
		id := string(*m.ID)
		switch m.Method {
		case "echo":
			b.write(`{"jsonrpc":"2.0","id":%s,"result":{"seenId":%q,"params":%s}}`, id, id, orNull(m.Params))
		case "emit":
			b.write(`{"jsonrpc":"2.0","method":"custom:event","params":%s}`, orNull(m.Params))
			b.write(`{"jsonrpc":"2.0","id":%s,"result":null}`, id)
		case "log":
			var p struct{ Text string }
			_ = json.Unmarshal(m.Params, &p)
			b.wmu.Lock()
			if !b.exited {
				_, _ = fmt.Fprintln(b.stderrW, p.Text)
			}
			b.wmu.Unlock()
			b.write(`{"jsonrpc":"2.0","id":%s,"result":null}`, id)
		case "discovery:subscribe", "ollama-proxy:subscribe":
			b.mu.Lock()
			fresh := !b.subbed[m.Method]
			b.subbed[m.Method] = true
			b.mu.Unlock()
			b.write(`{"jsonrpc":"2.0","id":%s,"result":{"subscribed":true}}`, id)
			if fresh && m.Method == "discovery:subscribe" {
				b.write(`{"jsonrpc":"2.0","method":"discovery:nodes-changed","params":[{"id":"node-%d"}]}`, b.gen)
			}
			if fresh && m.Method == "ollama-proxy:subscribe" {
				b.write(`{"jsonrpc":"2.0","method":"ollama-proxy:ready","params":{"port":11434}}`)
			}
		case "crash":
			b.exit(3)
			return
		case "hang":
		case "shutdown":
			if b.ignoreShutdown {
				continue
			}
			b.write(`{"jsonrpc":"2.0","id":%s,"result":null}`, id)
			b.exit(0)
			return
		default:
			b.write(`{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"method not found"}}`, id)
		}
	}
	if b.ignoreShutdown {
		<-b.killed
		return
	}
	b.exit(0)
}

func orNull(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "null"
	}
	return string(raw)
}

// received reports the methods the broker has been sent, in order.
func (b *fakeBroker) received() []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]string, 0, len(b.frames))
	for _, f := range b.frames {
		out = append(out, f.Method)
	}
	return out
}

func (b *fakeBroker) receivedIDs(method string) []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	var out []string
	for _, f := range b.frames {
		if f.Method == method && f.ID != nil {
			out = append(out, string(*f.ID))
		}
	}
	return out
}

// harness runs a Service on a private socket with fake brokers.
type harness struct {
	t        *testing.T
	dir      string
	endpoint string
	svc      *Service
	brokers  chan *fakeBroker
	runErr   chan error
	cancel   context.CancelFunc

	mu             sync.Mutex
	spawned        int
	ignoreShutdown bool
}

type harnessOption func(*harness, *config)

func withIgnoreShutdown() harnessOption {
	return func(h *harness, c *config) {
		h.ignoreShutdown = true
		c.stopGrace = 200 * time.Millisecond
		c.shutdownRequestTimeout = 100 * time.Millisecond
	}
}

// shortTempDir avoids t.TempDir, whose test-named path can overrun the Unix
// socket path limit.
func shortTempDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "nvs")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	return dir
}

func newHarness(t *testing.T, opts ...harnessOption) *harness {
	t.Helper()
	h := &harness{t: t, dir: shortTempDir(t), brokers: make(chan *fakeBroker, 16), runErr: make(chan error, 1)}
	h.endpoint = testEndpoint(h.dir)
	cfg := config{
		endpoint:               h.endpoint,
		version:                "test",
		logPath:                filepath.Join(h.dir, "logs", "broker.log"),
		logMaxSize:             defaultLogMaxSize,
		spawn:                  h.spawn,
		backoff:                func(int) time.Duration { return 10 * time.Millisecond },
		stopGrace:              2 * time.Second,
		shutdownRequestTimeout: 500 * time.Millisecond,
	}
	for _, o := range opts {
		o(h, &cfg)
	}
	h.svc = newService(cfg)
	ctx, cancel := context.WithCancel(context.Background())
	h.cancel = cancel
	go func() { h.runErr <- h.svc.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-h.runErr:
		case <-time.After(10 * time.Second):
			t.Error("service did not stop")
		}
	})
	waitUntil(t, "service listening", func() bool { return alreadyAnswering(context.Background(), h.endpoint) })
	return h
}

func (h *harness) spawn() (*brokerProc, error) {
	h.mu.Lock()
	h.spawned++
	b := newFakeBroker(h.spawned, h.ignoreShutdown)
	h.mu.Unlock()
	go b.run()
	h.brokers <- b
	return b.proc(), nil
}

// broker waits for the next broker the service spawns.
func (h *harness) broker() *fakeBroker {
	h.t.Helper()
	select {
	case b := <-h.brokers:
		return b
	case <-time.After(waitTimeout):
		h.t.Fatal("no broker spawned")
		return nil
	}
}

func waitUntil(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(waitTimeout)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// testClient is a raw JSON-RPC client of the service.
type testClient struct {
	t    *testing.T
	conn io.ReadWriteCloser
	msgs chan *jsonrpc.Message
	// backlog holds messages that arrived while waiting for something else,
	// so a notification racing a response is not lost to whichever wait
	// happened to run first.
	backlog []*jsonrpc.Message
}

func (h *harness) dial() *testClient {
	h.t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), waitTimeout)
	defer cancel()
	conn, err := servicectl.DialEndpoint(ctx, h.endpoint)
	if err != nil {
		h.t.Fatalf("dial: %v", err)
	}
	c := &testClient{t: h.t, conn: conn, msgs: make(chan *jsonrpc.Message, 1024)}
	go func() {
		defer close(c.msgs)
		codec := jsonrpc.NewCodecMaxFrame(conn, jsonrpc.WorkerFrameBytes)
		for {
			m, err := codec.Read()
			if err != nil {
				return
			}
			c.msgs <- m
		}
	}()
	h.t.Cleanup(func() { _ = conn.Close() })
	return c
}

func (c *testClient) send(format string, a ...any) {
	c.t.Helper()
	if _, err := fmt.Fprintf(c.conn, format+"\n", a...); err != nil {
		c.t.Fatalf("send: %v", err)
	}
}

func (c *testClient) request(id, method string, params string) {
	c.t.Helper()
	if params == "" {
		c.send(`{"jsonrpc":"2.0","id":%s,"method":%q}`, id, method)
		return
	}
	c.send(`{"jsonrpc":"2.0","id":%s,"method":%q,"params":%s}`, id, method, params)
}

// next returns the first message matching pred, in arrival order, keeping the
// others for later waits.
func (c *testClient) next(what string, pred func(*jsonrpc.Message) bool) *jsonrpc.Message {
	c.t.Helper()
	for i, m := range c.backlog {
		if pred(m) {
			c.backlog = append(c.backlog[:i:i], c.backlog[i+1:]...)
			return m
		}
	}
	timeout := time.After(waitTimeout)
	for {
		select {
		case m, ok := <-c.msgs:
			if !ok {
				c.t.Fatalf("connection closed while waiting for %s", what)
			}
			if pred(m) {
				return m
			}
			c.backlog = append(c.backlog, m)
		case <-timeout:
			c.t.Fatalf("timed out waiting for %s", what)
		}
	}
}

func (c *testClient) response(id string) *jsonrpc.Message {
	c.t.Helper()
	return c.next("response "+id, func(m *jsonrpc.Message) bool { return m.IsResponse() && string(*m.ID) == id })
}

func (c *testClient) notification(method string) *jsonrpc.Message {
	c.t.Helper()
	return c.next(method, func(m *jsonrpc.Message) bool { return m.IsNotification() && m.Method == method })
}

func (c *testClient) call(id, method, params string) *jsonrpc.Message {
	c.t.Helper()
	c.request(id, method, params)
	return c.response(id)
}

// expectClosed waits for the service to close the connection.
func (c *testClient) expectClosed() {
	c.t.Helper()
	timeout := time.After(waitTimeout)
	for {
		select {
		case _, ok := <-c.msgs:
			if !ok {
				return
			}
		case <-timeout:
			c.t.Fatal("connection still open")
		}
	}
}

// noMessage asserts nothing matching pred arrives within d.
func (c *testClient) noMessage(d time.Duration, pred func(*jsonrpc.Message) bool) {
	c.t.Helper()
	for _, m := range c.backlog {
		if pred(m) {
			c.t.Fatalf("unexpected message: method=%q params=%s", m.Method, m.Params)
		}
	}
	timeout := time.After(d)
	for {
		select {
		case m, ok := <-c.msgs:
			if !ok {
				return
			}
			if pred(m) {
				c.t.Fatalf("unexpected message: method=%q params=%s", m.Method, m.Params)
			}
		case <-timeout:
			return
		}
	}
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

func hasSubstring(s, sub string) bool { return strings.Contains(s, sub) }
