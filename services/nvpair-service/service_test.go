// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/servicectl"
)

func TestTwoClientsShareOneBrokerWithRemappedIDs(t *testing.T) {
	h := newHarness(t)
	b := h.broker()

	c1 := h.dial()
	c2 := h.dial()
	c1.notification("app:ready")
	c2.notification("app:ready")

	// Both clients use the same id; the broker must see two distinct ids and
	// each client must get its own answer back under its own id.
	c1.request("7", "echo", `{"who":"one"}`)
	c2.request("7", "echo", `{"who":"two"}`)
	r1 := c1.response("7")
	r2 := c2.response("7")

	var p1, p2 struct {
		SeenID string `json:"seenId"`
		Params struct {
			Who string `json:"who"`
		} `json:"params"`
	}
	if err := json.Unmarshal(r1.Result, &p1); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(r2.Result, &p2); err != nil {
		t.Fatal(err)
	}
	if p1.Params.Who != "one" || p2.Params.Who != "two" {
		t.Fatalf("responses crossed: c1 got %q, c2 got %q", p1.Params.Who, p2.Params.Who)
	}
	if p1.SeenID == p2.SeenID {
		t.Fatalf("broker saw the same id %s twice; ids were not rewritten", p1.SeenID)
	}
	if ids := b.receivedIDs("echo"); len(ids) != 2 || contains(ids, "7") && ids[0] == ids[1] {
		t.Fatalf("broker ids = %v", ids)
	}
}

func TestStringIDsAreRestored(t *testing.T) {
	h := newHarness(t)
	h.broker()
	c := h.dial()
	r := c.call(`"abc"`, "echo", `{}`)
	if string(*r.ID) != `"abc"` {
		t.Fatalf("response id = %s, want \"abc\"", *r.ID)
	}
}

func TestNotificationsAreBroadcastToEveryClient(t *testing.T) {
	h := newHarness(t)
	h.broker()
	c1 := h.dial()
	c2 := h.dial()

	c1.call("1", "emit", `{"n":1}`)
	for _, c := range []*testClient{c1, c2} {
		m := c.notification("custom:event")
		if string(m.Params) != `{"n":1}` {
			t.Fatalf("params = %s", m.Params)
		}
	}
}

func TestAppReadyIsReplayedToLateClients(t *testing.T) {
	h := newHarness(t)
	h.broker()
	c1 := h.dial()
	c1.notification("app:ready")

	// Long after the broker announced itself, a new client still starts
	// with app:ready, exactly once.
	time.Sleep(50 * time.Millisecond)
	c2 := h.dial()
	m := c2.next("first frame", func(*jsonrpc.Message) bool { return true })
	if m.Method != "app:ready" || !strings.Contains(string(m.Params), "fake-1") {
		t.Fatalf("first frame = %s %s, want app:ready", m.Method, m.Params)
	}
	c2.noMessage(100*time.Millisecond, func(m *jsonrpc.Message) bool { return m.Method == "app:ready" })
}

func TestUnsubscribeIsAnsweredLocally(t *testing.T) {
	h := newHarness(t)
	b := h.broker()
	c := h.dial()

	r := c.call("1", "discovery:unsubscribe", "")
	if r.Error != nil || string(r.Result) != `{"subscribed":false}` {
		t.Fatalf("unsubscribe result = %s err=%v", r.Result, r.Error)
	}
	c.send(`{"jsonrpc":"2.0","method":"engine:unsubscribe"}`)
	c.call("2", "echo", `{}`) // a barrier: everything before it has been handled
	for _, m := range b.received() {
		if strings.HasSuffix(m, ":unsubscribe") {
			t.Fatalf("broker received %s", m)
		}
	}
}

func TestSubscribeIsForwardedAndBaselineReplayed(t *testing.T) {
	h := newHarness(t)
	b := h.broker()
	c1 := h.dial()
	c2 := h.dial()

	c1.call("1", "discovery:subscribe", "")
	c1.notification("discovery:nodes-changed")
	c2.notification("discovery:nodes-changed") // broadcast, not just to c1
	c1.call("2", "ollama-proxy:subscribe", "")
	c1.notification("ollama-proxy:ready")

	// The broker treats the second subscription as redundant and pushes no
	// baseline; the service supplies its cached copies after the answer.
	c3 := h.dial()
	c3.request("9", "discovery:subscribe", "")
	c3.response("9")
	m := c3.notification("discovery:nodes-changed")
	if !strings.Contains(string(m.Params), "node-1") {
		t.Fatalf("replayed baseline = %s", m.Params)
	}
	c3.call("10", "ollama-proxy:subscribe", "")
	c3.notification("ollama-proxy:ready")

	if n := len(b.receivedIDs("discovery:subscribe")); n != 2 {
		t.Fatalf("broker got %d discovery:subscribe, want 2 (forwarded each time)", n)
	}
}

func TestClientShutdownDetachesOnlyThatClient(t *testing.T) {
	h := newHarness(t)
	b := h.broker()
	c1 := h.dial()
	c2 := h.dial()

	r := c1.call("1", "shutdown", "")
	if r.Error != nil || string(r.Result) != "null" {
		t.Fatalf("shutdown result = %s err=%v", r.Result, r.Error)
	}
	c1.expectClosed()

	// The broker is untouched and still serving the other client.
	if got := c2.call("2", "echo", `{}`); got.Error != nil {
		t.Fatalf("echo after detach: %v", got.Error)
	}
	if contains(b.received(), "shutdown") {
		t.Fatal("client shutdown reached the broker")
	}

	// The notification form detaches too and is never forwarded.
	c2.send(`{"jsonrpc":"2.0","method":"shutdown"}`)
	c2.expectClosed()
	c3 := h.dial()
	c3.call("3", "echo", `{}`)
	if contains(b.received(), "shutdown") {
		t.Fatal("shutdown notification reached the broker")
	}
	waitUntil(t, "client count", func() bool { return h.svc.status().Clients == 1 })
}

func TestServiceStatus(t *testing.T) {
	h := newHarness(t)
	h.broker()
	c := h.dial()
	c.notification("app:ready")
	// The harness's readiness probe is a connection too; let its detach land.
	waitUntil(t, "probe connection detached", func() bool { return h.svc.status().Clients == 1 })
	r := c.call("1", "service/status", "")
	var st statusResult
	if err := json.Unmarshal(r.Result, &st); err != nil {
		t.Fatal(err)
	}
	if st.PID != os.Getpid() || st.BrokerPID != 1001 || st.Clients != 1 || st.Version != "test" {
		t.Fatalf("status = %+v", st)
	}
	if _, err := time.Parse(time.RFC3339, st.StartedAt); err != nil {
		t.Fatalf("startedAt %q: %v", st.StartedAt, err)
	}
	if r := c.call("2", "service/nope", ""); r.Error == nil || r.Error.Code != codeMethodNotFound {
		t.Fatalf("unknown service method: %+v", r.Error)
	}
}

func TestBrokerRestartsAfterCrashAndRestoresSubscriptions(t *testing.T) {
	h := newHarness(t)
	b1 := h.broker()
	c := h.dial()
	c.notification("app:ready")
	c.call("1", "discovery:subscribe", "")
	c.notification("discovery:nodes-changed")

	// The crashing request is answered with an error rather than left hanging.
	c.request("2", "crash", "")
	r := c.response("2")
	if r.Error == nil || r.Error.Code != codeUnavailable {
		t.Fatalf("crash response = %+v", r)
	}

	b2 := h.broker()
	restarted := c.notification("service/broker-restarted")
	if !strings.Contains(string(restarted.Params), `"brokerPid":1002`) {
		t.Fatalf("restart params = %s", restarted.Params)
	}
	ready := c.notification("app:ready")
	if !strings.Contains(string(ready.Params), "fake-2") {
		t.Fatalf("app:ready after restart = %s", ready.Params)
	}
	// The new broker was re-subscribed for the client, so its baseline flows.
	m := c.notification("discovery:nodes-changed")
	if !strings.Contains(string(m.Params), "node-2") {
		t.Fatalf("baseline after restart = %s", m.Params)
	}
	waitUntil(t, "resubscribe", func() bool { return contains(b2.received(), "discovery:subscribe") })
	if got := c.call("3", "echo", `{}`); got.Error != nil {
		t.Fatalf("echo after restart: %v", got.Error)
	}
	if st := h.svc.status(); st.BrokerRestarts != 1 || st.BrokerPID != 1002 {
		t.Fatalf("status after restart = %+v", st)
	}
	// A late client gets the new broker's app:ready, not the dead one's.
	late := h.dial()
	if m := late.notification("app:ready"); !strings.Contains(string(m.Params), "fake-2") {
		t.Fatalf("late client app:ready = %s", m.Params)
	}
	_ = b1
}

func TestServiceStopShutsBrokerDownAndExits(t *testing.T) {
	h := newHarness(t)
	b := h.broker()
	c := h.dial()
	other := h.dial()
	c.notification("app:ready")

	r := c.call("1", "service/stop", "")
	if r.Error != nil || string(r.Result) != `{"stopped":true}` {
		t.Fatalf("stop result = %s err=%v", r.Result, r.Error)
	}
	if !contains(b.received(), "shutdown") {
		t.Fatal("broker never received shutdown")
	}
	select {
	case err := <-h.runErr:
		h.runErr <- err // for the cleanup
		if err != nil {
			t.Fatalf("Run returned %v", err)
		}
	case <-time.After(waitTimeout):
		t.Fatal("Run did not return after service/stop")
	}
	other.expectClosed()
	if alreadyAnswering(context.Background(), h.endpoint) {
		t.Fatal("endpoint still answering after stop")
	}
	if _, err := os.Stat(h.endpoint); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("socket file left behind: %v", err)
	}
	select {
	case <-h.brokers:
		t.Fatal("broker was restarted after a requested stop")
	default:
	}
}

func TestStopKillsABrokerThatIgnoresShutdown(t *testing.T) {
	h := newHarness(t, withIgnoreShutdown())
	b := h.broker()
	c := h.dial()
	c.notification("app:ready")

	start := time.Now()
	r := c.call("1", "service/stop", "")
	if r.Error != nil {
		t.Fatalf("stop: %v", r.Error)
	}
	select {
	case <-b.killed:
	default:
		t.Fatal("unresponsive broker was not killed")
	}
	if elapsed := time.Since(start); elapsed < 200*time.Millisecond {
		t.Fatalf("killed after %s, before the grace period", elapsed)
	}
}

func TestSignalStopsTheBrokerCleanly(t *testing.T) {
	h := newHarness(t)
	b := h.broker()
	h.cancel()
	select {
	case err := <-h.runErr:
		h.runErr <- err
	case <-time.After(waitTimeout):
		t.Fatal("Run did not return on cancel")
	}
	if !contains(b.received(), "shutdown") {
		t.Fatal("broker never received shutdown")
	}
}

func TestRequestsFailFastWhileNoBrokerRuns(t *testing.T) {
	dir := shortTempDir(t)
	endpoint := testEndpoint(dir)
	svc := newService(config{
		endpoint: endpoint,
		spawn:    func() (*brokerProc, error) { return nil, errors.New("no broker") },
		backoff:  func(int) time.Duration { return time.Hour },
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- svc.Run(ctx) }()
	defer func() {
		cancel()
		<-done
	}()
	h := &harness{t: t, endpoint: endpoint}
	waitUntil(t, "listening", func() bool { return alreadyAnswering(context.Background(), h.endpoint) })
	c := h.dial()
	r := c.call("1", "echo", `{}`)
	if r.Error == nil || r.Error.Code != codeUnavailable {
		t.Fatalf("echo with no broker = %+v", r)
	}
	if st := c.call("2", "service/status", ""); st.Error != nil {
		t.Fatalf("status with no broker: %v", st.Error)
	}
}

func TestSecondInstanceReportsAlreadyRunning(t *testing.T) {
	h := newHarness(t)
	h.broker()
	second := newService(config{
		endpoint: h.endpoint,
		spawn:    func() (*brokerProc, error) { t.Fatal("second instance spawned a broker"); return nil, nil },
		backoff:  defaultBackoff,
	})
	if err := second.Run(context.Background()); !errors.Is(err, errAlreadyRunning) {
		t.Fatalf("second Run = %v, want errAlreadyRunning", err)
	}
	// The first is unaffected.
	c := h.dial()
	c.call("1", "echo", `{}`)
}

func TestBrokerStderrIsLoggedAndBroadcast(t *testing.T) {
	h := newHarness(t)
	h.broker()
	c := h.dial()
	c.call("1", "log", `{"text":"hello from the broker"}`)
	m := c.notification("service/log")
	var p logParams
	if err := json.Unmarshal(m.Params, &p); err != nil {
		t.Fatal(err)
	}
	if p.Source != "nvpair-ui-broker" || p.Stream != "stderr" || p.Text != "hello from the broker" {
		t.Fatalf("service/log = %+v", p)
	}
	waitUntil(t, "log file line", func() bool {
		data, err := os.ReadFile(filepath.Join(h.dir, "logs", "broker.log"))
		return err == nil && strings.Contains(string(data), "hello from the broker\n")
	})
	if runtime.GOOS != "windows" {
		st, err := os.Stat(filepath.Join(h.dir, "logs", "broker.log"))
		if err != nil {
			t.Fatal(err)
		}
		if mode := st.Mode().Perm(); mode != 0o600 {
			t.Fatalf("log mode = %o, want 600", mode)
		}
	}
}

func TestSlowClientIsDisconnectedNotBlocking(t *testing.T) {
	h := newHarness(t)
	h.broker()
	fast := h.dial()

	// A client that never reads: its connection is opened raw and ignored.
	ctx, cancel := context.WithTimeout(context.Background(), waitTimeout)
	defer cancel()
	slow, err := servicectl.DialEndpoint(ctx, h.endpoint)
	if err != nil {
		t.Fatal(err)
	}
	defer slow.Close()
	waitUntil(t, "slow client attached", func() bool { return h.svc.status().Clients == 2 })

	big := strings.Repeat("x", 4096)
	for i := 0; i < clientQueueFrames+64; i++ {
		h.svc.broadcast("custom:flood", map[string]string{"pad": big})
	}
	waitUntil(t, "slow client dropped", func() bool { return h.svc.status().Clients == 1 })
	if r := fast.call("1", "echo", `{}`); r.Error != nil {
		t.Fatalf("fast client blocked: %v", r.Error)
	}
}

func TestParseRunArgs(t *testing.T) {
	opts, err := parseRunArgs([]string{"--log-level", "debug", "--", "--proxy-engines", "ollama"}, os.Stderr)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--proxy-engines", "ollama", "--log-level", "debug"}
	if strings.Join(opts.brokerArgs, " ") != strings.Join(want, " ") {
		t.Fatalf("broker args = %v, want %v", opts.brokerArgs, want)
	}

	opts, err = parseRunArgs([]string{"--log-level=warn", "--", "--log-level", "error"}, os.Stderr)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(opts.brokerArgs, " ") != "--log-level error" {
		t.Fatalf("explicit broker level overridden: %v", opts.brokerArgs)
	}

	opts, err = parseRunArgs(nil, os.Stderr)
	if err != nil || len(opts.brokerArgs) != 0 {
		t.Fatalf("no args: %v %v", opts, err)
	}

	if _, err := parseRunArgs([]string{"stray"}, os.Stderr); err == nil {
		t.Fatal("stray positional argument accepted")
	}
}

func TestDefaultBackoff(t *testing.T) {
	want := []time.Duration{time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second, 16 * time.Second, 30 * time.Second, 30 * time.Second}
	for i, w := range want {
		if got := defaultBackoff(i + 1); got != w {
			t.Fatalf("backoff(%d) = %s, want %s", i+1, got, w)
		}
	}
}
