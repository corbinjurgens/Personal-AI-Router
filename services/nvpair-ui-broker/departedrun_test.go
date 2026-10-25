// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"io"
	"net"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-ui-broker/workloadstore"
)

func TestDepartedRunHistoryApplyFence(t *testing.T) {
	b, old, _ := historyFixture(t)
	prior := historyRecord("9", 10, 1)
	historySend(t, b, old, "1", prior)
	update := prior
	update.Seq = 2
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	b.routingHistory.beforeApply = func() { close(entered); <-release }
	b.forwardProxyNotificationFrom(old, 0, 0, nodepolicy.NotifyRoutingHistoryWindow, historyMarshal(t, historyManifest(t, "2", update)))
	b.forwardProxyNotificationFrom(old, 0, 0, nodepolicy.NotifyRoutingRequest, historyMarshal(t, update))
	end := historyMarshal(t, nodepolicy.RoutingHistoryWindowEnd{Schema: 1, Generation: "2"})
	go func() {
		b.forwardProxyNotificationFrom(old, 0, 0, nodepolicy.NotifyRoutingHistoryWindowEnd, end)
		close(done)
	}()
	<-entered
	// The admitted callback owns history.mu, but withdrawal needs only
	// workersMu. It must complete before releasing the copy/apply barrier.
	close(old.done)
	b.clearProxyHandles()
	b.setProxyProcessHandle(&proxyProcess{done: make(chan struct{})})
	close(release)
	<-done
	b.routingHistory.beforeApply = nil
	historyRequireRows(t, b, prior)
	h := b.routingHistory
	h.mu.Lock()
	h.retireLocked()
	h.mu.Unlock()
	rows := historyRows(t, b)
	if len(rows) != 1 || rows[0].Seq != 2 || rows[0].Reason != "proxy-exited" || rows[0].State != "failed" || rows[0].Recoverable || rows[0].EndedAt == nil || *rows[0].EndedAt < prior.CreatedAt {
		t.Fatalf("retirement after apply fence = %+v", rows)
	}
	if !reflect.DeepEqual(rows[0].Spend, prior.Spend) {
		t.Fatal("retirement changed observed money")
	}
}

func TestDepartedRunHistoryFrontier(t *testing.T) {
	b, old, _ := historyFixture(t)
	open := historyRecord("9", 10, 4)
	held, charged := nodepolicy.NanoUSD("250000000"), nodepolicy.NanoUSD("123456789")
	open.Spend.HeldNanoUSD, open.Spend.ChargedNanoUSD, open.Spend.Accounting = &held, &charged, "retained"
	terminal := historyTerminal(historyRecord("10", 11, 2))
	historySend(t, b, old, "1", open, terminal)
	b.clearProxyHandles()
	replacement := historyActivate(t, b, "run-b")
	newer := historyRecord("1", 12, 1)
	newer.Request.ProxyRunID = "run-b"
	historySend(t, b, replacement, "1", newer)
	h := b.routingHistory
	h.mu.Lock()
	h.retireLocked()
	h.mu.Unlock()
	rows := historyRows(t, b)
	if len(rows) != 3 || rows[0].Request.ProxyRunID != "run-b" || rows[0].State != "routing" || rows[1].Seq != 2 || rows[1].Reason != "no-owner" || rows[2].Seq != 5 || rows[2].Reason != "proxy-exited" {
		t.Fatalf("frontier scope = %+v", rows)
	}
	retired := rows[2]
	if retired.Spend.HeldNanoUSD == nil || *retired.Spend.HeldNanoUSD != "250000000" || retired.Spend.ChargedNanoUSD == nil || *retired.Spend.ChargedNanoUSD != "123456789" || retired.Spend.Accounting != "retained" || retired.Served != nil {
		t.Fatal("presentation retirement invented delivery or changed accounting")
	}
	historySend(t, b, old, "2", historyRecord("9", 10, 99))
	h.mu.Lock()
	h.retireLocked()
	h.mu.Unlock()
	after := historyRows(t, b)
	if after[2].Seq != retired.Seq || after[0].State != "routing" {
		t.Fatal("late old run or duplicate retirement changed records")
	}
}

func TestDepartedRunHistoryRepeatedCrashesKeepBanks(t *testing.T) {
	b, p, _ := historyFixture(t)
	row := historyRecord("9", 10, 1)
	historySend(t, b, p, "1", row)
	h := b.routingHistory
	banks := map[*routingHistoryBank]bool{h.accepted: true, h.stage: true, h.merge: true, h.initial: true}
	// No history owner turn between crashes: the actual withdrawal frontier
	// must coalesce without pinning a bank or retaining a run list per crash.
	for i := 0; i < 300; i++ {
		b.clearProxyHandles()
		b.setProxyProcessHandle(&proxyProcess{done: make(chan struct{})})
	}
	b.clearProxyHandles()
	h.mu.Lock()
	h.retireLocked()
	h.mu.Unlock()
	for _, bank := range []*routingHistoryBank{h.accepted, h.stage, h.merge, h.initial} {
		if !banks[bank] {
			t.Fatal("departure allocated another window")
		}
	}
	rows := historyRows(t, b)
	if len(rows) != 1 || rows[0].Seq != 2 || rows[0].State != "failed" || h.retiredThrough != b.historyRetiredThrough {
		t.Fatalf("coalesced frontier = %+v", rows)
	}
}

func TestDepartedRunHistoryReadReuseDuringOrdinaryCallbacks(t *testing.T) {
	b, _, _ := historyFixture(t)
	r := &b.routingHistory.readers
	callbacksRelease := make(chan struct{})
	var once sync.Once
	release := func() { once.Do(func() { close(callbacksRelease) }) }
	t.Cleanup(release)
	type oldRead struct {
		codec *Codec
		done  chan struct{}
	}
	var old [4]oldRead
	var inputAddress *byte
	offsetAddress := &r.slots[historyProxyPrivate].offsets[0]
	for i := range old {
		p := historyActivate(t, b, "run-"+strconv.Itoa(i))
		frame := `{"jsonrpc":"2.0","method":"ordinary","params":{"generation":` + strconv.Itoa(i) + "}}\n"
		if err := p.attachRoutingHistoryPeer(b, readWriter{strings.NewReader(frame + frame), io.Discard}, func() {
			t.Error("blocked ordinary callback still owned an active read")
		}); err != nil {
			t.Fatal(err)
		}
		address := &r.slots[historyProxyPrivate].raw[:cap(r.slots[historyProxyPrivate].raw)][0]
		if i == 0 {
			inputAddress = address
		}
		if address != inputAddress || &r.slots[historyProxyPrivate].offsets[0] != offsetAddress {
			t.Fatal("restart allocated another role bank")
		}
		entered, pumpDone := make(chan struct{}), make(chan struct{})
		want, callbacks := i, 0
		p.onNotify = func(method string, params json.RawMessage) {
			callbacks++
			if callbacks == 1 {
				close(entered)
			}
			<-callbacksRelease
			var got struct {
				Generation int `json:"generation"`
			}
			if err := json.Unmarshal(params, &got); err != nil {
				t.Errorf("ordinary owned params: %v", err)
			} else if got.Generation != want || method != "ordinary" {
				t.Errorf("late ordinary callback changed: generation %d, method %s", got.Generation, method)
			}
		}
		go func() {
			p.servePeer()
			if callbacks != 1 {
				t.Errorf("buffered frame escaped retirement: %d callbacks", callbacks)
			}
			close(pumpDone)
		}()
		historyAwaitReadSignal(t, entered)
		withdrawn := make(chan struct{})
		go func() { b.clearProxyHandles(); close(withdrawn) }()
		historyAwaitReadSignal(t, withdrawn)
		historyAwaitReadSignal(t, p.peer.Codec().RetireRead())
		historyRequireReadDetached(t, p.peer.Codec())
		select {
		case <-pumpDone:
			t.Fatal("old ordinary callback was not held across replacement")
		default:
		}
		old[i] = oldRead{p.peer.Codec(), pumpDone}
		// Retained old references cannot read or release a newly bound bank.
		for _, previous := range old[:i+1] {
			r.retire(historyProxyPrivate, previous.codec)
			historyRequireReadDetached(t, previous.codec)
		}
	}
	release()
	for _, previous := range old {
		historyAwaitReadSignal(t, previous.done)
		if _, _, err := previous.codec.ReadWithRelease(); err != jsonrpc.ErrPeerClosed {
			t.Fatalf("old read shell revived: %v", err)
		}
	}
}

func TestDepartedRunHistoryReadWaitsForHistoryCallback(t *testing.T) {
	b, p, _ := historyFixture(t)
	row := historyRecord("9", 10, 1)
	begin := historyMarshal(t, Message{JSONRPC: "2.0", Method: nodepolicy.NotifyRoutingHistoryWindow, Params: historyMarshal(t, historyManifest(t, "1", row))})
	record := historyMarshal(t, Message{JSONRPC: "2.0", Method: nodepolicy.NotifyRoutingRequest, Params: historyMarshal(t, row)})
	end := historyMarshal(t, Message{JSONRPC: "2.0", Method: nodepolicy.NotifyRoutingHistoryWindowEnd, Params: historyMarshal(t, nodepolicy.RoutingHistoryWindowEnd{Schema: 1, Generation: "1"})})
	input := string(begin) + "\n" + string(record) + "\n" + string(end) + "\n"
	if err := p.attachRoutingHistoryPeer(b, readWriter{strings.NewReader(input), io.Discard}, func() {}); err != nil {
		t.Fatal(err)
	}
	p.onNotify = func(method string, params json.RawMessage) { b.forwardProxyNotificationFrom(p, 0, 0, method, params) }
	entered, release, pumpDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	finish := func() { once.Do(func() { close(release) }) }
	t.Cleanup(finish)
	b.routingHistory.beforeApply = func() { close(entered); <-release }
	go func() { p.servePeer(); close(pumpDone) }()
	historyAwaitReadSignal(t, entered)
	withdrawn := make(chan struct{})
	go func() { b.clearProxyHandles(); close(withdrawn) }()
	historyAwaitReadSignal(t, withdrawn)
	codec := p.peer.Codec()
	done := codec.RetireRead()
	historyRequireReadPending(t, done)
	r := &b.routingHistory.readers
	if _, err := r.acquire(historyProxyPrivate, readWriter{strings.NewReader(""), io.Discard}, func() {}, nil); err != jsonrpc.ErrHistoryBusy {
		t.Fatalf("admitted history callback bank rebound: %v", err)
	}
	finish()
	historyAwaitReadSignal(t, done)
	historyAwaitReadSignal(t, pumpDone)
	historyRequireReadDetached(t, codec)
	if _, err := r.acquire(historyProxyPrivate, readWriter{strings.NewReader(""), io.Discard}, func() {}, nil); err != nil {
		t.Fatalf("actual history consumption did not release role: %v", err)
	}
	if b.routingHistory.accepted.count != 0 {
		t.Fatal("old callback committed after withdrawal")
	}
}

func departedRunInput(t *testing.T, id, origin, runID, state, destination string) workloadstore.Incoming {
	t.Helper()
	in := storeIncoming(id, origin, "ollama", runID, state, destination)
	fields := privacyObject(t, in.Info)
	fields["seq"] = json.RawMessage(`7`)
	fields["startedAt"] = json.RawMessage(`2`)
	fields["requestedModel"] = json.RawMessage(`"requested"`)
	fields["requesterId"] = json.RawMessage(`"requester"`)
	if state == "completed" {
		fields["completedAt"] = json.RawMessage(`1`)
	}
	raw, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	in, ok := workloadstore.ParseIncoming(raw)
	if !ok {
		t.Fatal("invalid workload fixture")
	}
	return in
}

func seedDepartedRun(t *testing.T, b *Broker, in workloadstore.Incoming) {
	t.Helper()
	if !b.workloads.Apply(in) {
		t.Fatalf("seed %s rejected", in.ID)
	}
}

func departedRunEnvelope(t *testing.T, info json.RawMessage) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(map[string]json.RawMessage{"workloadInfo": info})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func departedRunRecord(t *testing.T, b *Broker, origin, id, state string) workloadstore.Record {
	t.Helper()
	rec, ok := b.workloads.Get(origin, id)
	if !ok || rec.State != state {
		t.Fatalf("record %s/%s = %+v, present %v, want %s", origin, id, rec, ok, state)
	}
	return rec
}

func departedRunFrames(t *testing.T, out *syncBuffer, method string, want map[string]string) {
	t.Helper()
	frames := out.frames()
	if len(frames) != len(want) {
		t.Fatalf("%s frames = %d, want %d", method, len(frames), len(want))
	}
	seen := make(map[string]bool)
	for _, frame := range frames {
		var env struct {
			Info struct {
				ID    string `json:"id"`
				State string `json:"state"`
				Seq   int64  `json:"seq"`
				Error string `json:"error"`
			} `json:"workloadInfo"`
		}
		if err := json.Unmarshal(frame.Params, &env); err != nil {
			t.Fatal(err)
		}
		state, ok := want[env.Info.ID]
		if !ok || seen[env.Info.ID] || frame.Method != method || env.Info.State != state || env.Info.Seq != 8 || env.Info.Error != "proxy exited before the request finished" {
			t.Fatalf("unexpected %s frame: %+v / %+v", method, frame, env.Info)
		}
		seen[env.Info.ID] = true
	}
}

func waitDepartedRun(t *testing.T, done <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}

// Tests join the actual emitter with their own failure bound, independent of a
// deliberately exhausted product shutdown budget.
func joinDepartedRun(t *testing.T, b *Broker) {
	t.Helper()
	b.proxySupWorkMu.Lock()
	done := b.proxyRetireDone
	b.proxySupWorkMu.Unlock()
	if done != nil {
		waitDepartedRun(t, done, "retirement emitter cleanup")
	}
}

type departedRunObservedPipe struct {
	net.Conn
	entered  chan struct{}
	finished chan struct{}
	once     sync.Once
}

func (p *departedRunObservedPipe) Write(raw []byte) (int, error) {
	p.once.Do(func() { close(p.entered) })
	defer close(p.finished)
	return p.Conn.Write(raw)
}

type departedRunBlockingWriter struct {
	output  *syncBuffer
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func (w *departedRunBlockingWriter) Write(p []byte) (int, error) {
	w.once.Do(func() { close(w.entered) })
	<-w.release
	return w.output.Write(p)
}

func (*departedRunBlockingWriter) Read([]byte) (int, error) { return 0, io.EOF }
func (*departedRunBlockingWriter) Close() error             { return nil }

type departedRunFailedWriter struct{}

func (departedRunFailedWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }
func (departedRunFailedWriter) Read([]byte) (int, error)  { return 0, io.EOF }

func TestDepartedRun(t *testing.T) {
	t.Run("authoritative retirement fans and replays once", func(t *testing.T) {
		b, client, scheduler, manager := privacyBroker(t)
		for _, in := range []workloadstore.Incoming{
			departedRunInput(t, "queued", "host", "r1", "queued", "peer"),
			departedRunInput(t, "running", "host", "r1", "running", "host"),
			departedRunInput(t, "other-run", "host", "r2", "running", "peer"),
			departedRunInput(t, "remote", "peer", "r1", "running", "host"),
			departedRunInput(t, "finished", "host", "r1", "completed", "host"),
		} {
			seedDepartedRun(t, b, in)
		}
		before := time.Now().UnixMilli()
		b.captureDepartedRunWorkloads("test departure")
		joinDepartedRun(t, b)
		after := time.Now().UnixMilli()
		want := map[string]string{"queued": "failed", "running": "failed", "other-run": "failed"}
		for id := range want {
			rec := departedRunRecord(t, b, "host", id, "failed")
			if !rec.Terminal || rec.Inferred || rec.RunID == "" || rec.Seq != 8 || rec.CompletedAt < before || rec.CompletedAt > after {
				t.Fatalf("retired metadata = %+v", rec)
			}
			fields := privacyObject(t, rec.Info)
			for key, expected := range map[string]string{
				"error": `"proxy exited before the request finished"`, "createdAt": `1`,
				"startedAt": `2`, "requesterId": `"requester"`, "requestedModel": `"requested"`,
				"model": `"m"`, "engine": `"ollama"`, "originatedFrom": `"host"`,
			} {
				if string(fields[key]) != expected {
					t.Fatalf("%s %s = %s, want %s", id, key, fields[key], expected)
				}
			}
			wantDestination := "peer"
			if id == "running" {
				wantDestination = "host"
			}
			if rec.ScheduledOn != wantDestination {
				t.Fatalf("%s moved to %s", id, rec.ScheduledOn)
			}
		}
		departedRunRecord(t, b, "peer", "remote", "running")
		departedRunRecord(t, b, "host", "finished", "completed")
		departedRunFrames(t, client, "workloads:upsert", want)
		departedRunFrames(t, scheduler, "workloads:upsert", want)
		departedRunFrames(t, manager, "workload:errored", want)
		b.captureDepartedRunWorkloads("duplicate departure")
		joinDepartedRun(t, b)
		departedRunFrames(t, manager, "workload:errored", want)
		departedRunFrames(t, client, "workloads:upsert", want)
		departedRunFrames(t, scheduler, "workloads:upsert", want)
		replay := b.workloads.ReplayForNode("host", workloadTerminalReplayWindowMs)
		if len(replay) != 3 {
			t.Fatalf("replay = %d, want 3 recent retired records", len(replay))
		}
		for _, rec := range replay {
			if want[rec.ID] != "failed" || rec.State != "failed" || rec.Inferred {
				t.Fatalf("unexpected replay: %+v", rec)
			}
		}
	})

	t.Run("node loss inferred failure becomes authoritative on actual departure", func(t *testing.T) {
		b, client, scheduler, manager := privacyBroker(t)
		path := filepath.Join(t.TempDir(), "retired-history.json")
		b.workloads.WithPersistence(path)
		pp := &proxyProcess{done: make(chan struct{})}
		b.setProxyProcessHandle(pp)
		in := departedRunInput(t, "one", "host", "r1", "running", "peer")
		b.forwardProxyNotificationFrom(pp, 0, 0, "workload:started", departedRunEnvelope(t, in.Info))
		b.failWorkloadsForNode("peer", "peer")
		rec := departedRunRecord(t, b, "host", "one", "failed")
		if !rec.Inferred || rec.Seq != 7 {
			t.Fatalf("node loss did not create the inferred precondition: %+v", rec)
		}
		frames := manager.frames()
		if len(frames) != 1 {
			t.Fatal("inferred failure was broadcast authoritatively")
		}
		last, ok := workloadstore.ParseIncoming(privacyObject(t, frames[0].Params)["workloadInfo"])
		if !ok || last.State != "running" || last.Seq != 7 {
			t.Fatal("manager did not retain its last authoritative running record")
		}
		before := time.Now().UnixMilli()
		crash := newSupervisor(engines.ProxyComponent, defaultRestartPolicy(), nil)
		b.configureProxySupervisorCallbacks(crash)
		close(pp.done)
		crash.onCrash(1)
		joinDepartedRun(t, b)
		rec = departedRunRecord(t, b, "host", "one", "failed")
		if rec.Inferred || !rec.Terminal || rec.Seq != 8 || rec.RunID != "r1" || rec.ScheduledOn != "peer" || rec.CompletedAt < before {
			t.Fatalf("departure left inferred provenance or lost identity/sequence: %+v", rec)
		}
		for _, output := range []struct {
			out    *syncBuffer
			count  int
			method string
		}{
			{client, 3, "workloads:upsert"}, {scheduler, 3, "workloads:upsert"}, {manager, 2, "workload:errored"},
		} {
			frames := output.out.frames()
			if len(frames) != output.count {
				t.Fatalf("%s count = %d, want %d", output.method, len(frames), output.count)
			}
			frame := frames[len(frames)-1]
			info := privacyObject(t, frame.Params)["workloadInfo"]
			last, ok := workloadstore.ParseIncoming(info)
			if !ok || frame.Method != output.method || last.ID != "one" || last.Origin != "host" || last.RunID != "r1" || last.State != "failed" || last.Seq != 8 || last.ScheduledOn != "peer" {
				t.Fatalf("authoritative departure did not converge: %+v", last)
			}
			if string(privacyObject(t, info)["error"]) != `"proxy exited before the request finished"` {
				t.Fatal("wrong retirement error")
			}
		}
		replay := b.workloads.ReplayForNode("host", workloadTerminalReplayWindowMs)
		if len(replay) != 1 || replay[0].Inferred || replay[0].State != "failed" || replay[0].Seq != 8 {
			t.Fatal("authoritative terminal missing from manager restart replay")
		}
		if err := b.workloads.Flush(); err != nil {
			t.Fatal(err)
		}
		restored := workloadstore.New().WithPersistence(path)
		if err := restored.Load(); err != nil {
			t.Fatal(err)
		}
		saved, ok := restored.Get("host", "one")
		if !ok || saved.Inferred || saved.State != "failed" || saved.Seq != 8 {
			t.Fatal("authoritative retirement was not persisted")
		}
	})

	t.Run("replacement run stays live and synthetic frames leave fence alone", func(t *testing.T) {
		b, client, _, _ := privacyBroker(t)
		seedDepartedRun(t, b, departedRunInput(t, "old", "host", "r1", "running", "peer"))
		seedDepartedRun(t, b, departedRunInput(t, "already-live", "host", "r2", "running", "host"))
		pp := &proxyProcess{done: make(chan struct{})}
		b.setProxyProcessHandle(pp)
		live := departedRunInput(t, "live", "host", "r2", "queued", "host")
		b.forwardProxyNotificationFrom(pp, 0, 0, "workload:submitted", departedRunEnvelope(t, live.Info))
		frames := client.frames()
		if len(frames) != 2 {
			t.Fatalf("first event frames = %d, want old failure then new queued", len(frames))
		}
		for index, expected := range []struct{ id, state string }{{"old", "failed"}, {"live", "queued"}} {
			in, ok := workloadstore.ParseIncoming(privacyObject(t, frames[index].Params)["workloadInfo"])
			if !ok || in.ID != expected.id || in.State != expected.state {
				t.Fatalf("event %d = %+v, want %+v", index, in, expected)
			}
		}
		departedRunRecord(t, b, "host", "already-live", "running")
		// This sentinel would be retired by an unnecessary second r2 scan.
		seedDepartedRun(t, b, departedRunInput(t, "sentinel", "host", "r3", "running", "peer"))
		b.forwardProxyNotificationFrom(pp, 0, 0, "workload:started", departedRunEnvelope(t, departedRunInput(t, "live", "host", "r2", "running", "host").Info))
		departedRunRecord(t, b, "host", "sentinel", "running")
		if b.lastLocalRunID != "r2" {
			t.Fatalf("fence = %s, want r2", b.lastLocalRunID)
		}
		// Capture the actual departure before a replacement's record arrives.
		// Keep the emitter at its supervision barrier while publishing the fixture.
		b.proxySupMu.Lock()
		var releaseOnce sync.Once
		releaseCapture := func() { releaseOnce.Do(b.proxySupMu.Unlock) }
		t.Cleanup(func() { releaseCapture(); joinDepartedRun(t, b) })
		b.clearProxyHandles()
		b.captureDepartedRunWorkloads("synthetic departure")
		b.setProxyProcessHandle(&proxyProcess{done: make(chan struct{})})
		b.lastLocalRunID = "r4"
		seedDepartedRun(t, b, departedRunInput(t, "replacement", "host", "r4", "running", "host"))
		releaseCapture()
		joinDepartedRun(t, b)
		departedRunRecord(t, b, "host", "live", "failed")
		departedRunRecord(t, b, "host", "sentinel", "failed")
		departedRunRecord(t, b, "host", "replacement", "running")
		if b.lastLocalRunID != "r4" {
			t.Fatalf("synthetic retirement changed fence to %s", b.lastLocalRunID)
		}
	})

	t.Run("notification failure keeps authoritative terminal and manager forward", func(t *testing.T) {
		b, _, _, manager := privacyBroker(t)
		b.codec = NewCodec(departedRunFailedWriter{})
		seedDepartedRun(t, b, departedRunInput(t, "one", "host", "r1", "running", "host"))
		b.captureDepartedRunWorkloads("failed client output")
		joinDepartedRun(t, b)
		rec := departedRunRecord(t, b, "host", "one", "failed")
		if rec.Inferred || len(b.workloads.ReplayForNode("host", workloadTerminalReplayWindowMs)) != 1 {
			t.Fatal("failed client write lost authoritative replayable outcome")
		}
		departedRunFrames(t, manager, "workload:errored", map[string]string{"one": "failed"})
	})

	t.Run("legacy and manager frames never advance fence", func(t *testing.T) {
		b, _, _, manager := privacyBroker(t)
		b.lastLocalRunID = "r2"
		seedDepartedRun(t, b, departedRunInput(t, "live", "host", "r2", "running", "host"))
		legacy := departedRunInput(t, "legacy", "host", "legacy-run", "queued", "host")
		fields := privacyObject(t, legacy.Info)
		delete(fields, "runId")
		raw, err := json.Marshal(fields)
		if err != nil {
			t.Fatal(err)
		}
		b.routeProxyWorkload("workload:submitted", departedRunEnvelope(t, raw))
		departedRunRecord(t, b, "host", "legacy", "queued")
		b.forwardWorkloadManagerNotification("workloads:upsert", departedRunEnvelope(t, departedRunInput(t, "remote", "peer", "r3", "running", "host").Info))
		departedRunRecord(t, b, "peer", "remote", "running")
		departedRunRecord(t, b, "host", "live", "running")
		if b.lastLocalRunID != "r2" || len(manager.frames()) != 1 {
			t.Fatal("legacy/manager frame advanced fence or rebroadcast remote input")
		}
	})

	t.Run("nil source preserves merge semantics without fencing live process", func(t *testing.T) {
		b, client, scheduler, manager := privacyBroker(t)
		b.setProxyProcessHandle(&proxyProcess{done: make(chan struct{})})
		b.lastLocalRunID = "r2"
		seedDepartedRun(t, b, departedRunInput(t, "live", "host", "r2", "running", "host"))
		b.routeProxyWorkload("workload:submitted", departedRunEnvelope(t, departedRunInput(t, "unscoped", "host", "r1", "queued", "peer").Info))
		departedRunRecord(t, b, "host", "live", "running")
		departedRunRecord(t, b, "host", "unscoped", "queued")
		if b.lastLocalRunID != "r2" || len(client.frames()) != 1 || len(scheduler.frames()) != 1 || len(manager.frames()) != 1 {
			t.Fatal("nil source fenced or changed direct merge semantics")
		}
	})

	t.Run("model only queued repoint is rejected on every proxy output", func(t *testing.T) {
		b, client, scheduler, manager := privacyBroker(t)
		in := departedRunInput(t, "one", "host", "r1", "queued", "host")
		b.routeProxyWorkload("workload:submitted", departedRunEnvelope(t, in.Info))
		fields := privacyObject(t, in.Info)
		fields["model"] = json.RawMessage(`"other-model"`)
		fields["seq"] = json.RawMessage(`8`)
		raw, err := json.Marshal(fields)
		if err != nil {
			t.Fatal(err)
		}
		b.routeProxyWorkload("workload:submitted", departedRunEnvelope(t, raw))
		for _, out := range []*syncBuffer{client, scheduler, manager} {
			if len(out.frames()) != 1 {
				t.Fatal("store-rejected model-only repoint reached a proxy output")
			}
		}
		rec := departedRunRecord(t, b, "host", "one", "queued")
		if string(privacyObject(t, rec.Info)["model"]) != `"m"` || rec.Seq != 7 {
			t.Fatal("same-rank same-placement model-only frame changed store")
		}
	})

	t.Run("rejected frames are never forwarded", func(t *testing.T) {
		b, client, scheduler, manager := privacyBroker(t)
		in := departedRunInput(t, "one", "host", "r2", "running", "host")
		b.routeProxyWorkload("workload:started", departedRunEnvelope(t, in.Info))
		terminal := markWorkloadTerminal(in.Info, "completed", time.UnixMilli(100), "", 8)
		b.routeProxyWorkload("workload:completed", terminal)
		b.routeProxyWorkload("workload:completed", terminal)
		b.routeProxyWorkload("workload:started", departedRunEnvelope(t, in.Info))
		stale := privacyObject(t, in.Info)
		stale["createdAt"] = json.RawMessage(`0`)
		stale["scheduledOn"] = json.RawMessage(`"elsewhere"`)
		raw, err := json.Marshal(stale)
		if err != nil {
			t.Fatal(err)
		}
		b.routeProxyWorkload("workload:errored", markWorkloadTerminal(raw, "failed", time.UnixMilli(200), "stale", 9))
		for _, output := range []*syncBuffer{client, scheduler, manager} {
			if len(output.frames()) != 2 {
				t.Fatalf("got %d frames, want two accepted transitions", len(output.frames()))
			}
		}
		rec := departedRunRecord(t, b, "host", "one", "completed")
		if rec.CompletedAt != 100 || rec.Seq != 8 || rec.ScheduledOn != "host" {
			t.Fatalf("rejected input changed terminal: %+v", rec)
		}
	})

	t.Run("old callback admitted before replacement is rechecked", func(t *testing.T) {
		b, client, _, manager := privacyBroker(t)
		old := &proxyProcess{done: make(chan struct{})}
		replacement := &proxyProcess{done: make(chan struct{})}
		b.setProxyProcessHandle(old)
		seedDepartedRun(t, b, departedRunInput(t, "old", "host", "r1", "running", "host"))
		entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
		var once sync.Once
		releaseOld := func() { once.Do(func() { close(release) }) }
		t.Cleanup(func() { releaseOld(); waitDepartedRun(t, done, "old callback cleanup") })
		var calls atomic.Int32
		b.beforeProxyWorkloadApply = func() {
			if calls.Add(1) == 1 {
				close(entered)
				<-release
			}
		}
		late := departedRunEnvelope(t, departedRunInput(t, "late-old", "host", "r1", "queued", "host").Info)
		go func() { defer close(done); b.forwardProxyNotificationFrom(old, 0, 0, "workload:submitted", late) }()
		waitDepartedRun(t, entered, "callback past initial handle guard")
		close(old.done)
		b.setProxyProcessHandle(replacement)
		b.forwardProxyNotificationFrom(replacement, 0, 0, "workload:submitted", departedRunEnvelope(t, departedRunInput(t, "live", "host", "r2", "queued", "host").Info))
		releaseOld()
		waitDepartedRun(t, done, "late old callback")
		departedRunRecord(t, b, "host", "old", "failed")
		departedRunRecord(t, b, "host", "live", "queued")
		if _, ok := b.workloads.Get("host", "late-old"); ok {
			t.Fatal("late old callback inserted a ghost")
		}
		if b.lastLocalRunID != "r2" || len(client.frames()) != 2 || len(manager.frames()) != 2 {
			t.Fatal("late callback changed fence or propagated")
		}
	})

	t.Run("captured identities preserve intervening terminals and placement", func(t *testing.T) {
		b, client, scheduler, manager := privacyBroker(t)
		for _, id := range []string{"completed", "cancelled", "moved", "removed", "new-generation"} {
			seedDepartedRun(t, b, departedRunInput(t, id, "host", "r1", "running", "host"))
		}
		b.proxySupMu.Lock()
		var once sync.Once
		releaseCapture := func() { once.Do(b.proxySupMu.Unlock) }
		t.Cleanup(func() { releaseCapture(); joinDepartedRun(t, b) })
		b.captureDepartedRunWorkloads("captured")
		apply := func(params json.RawMessage) {
			t.Helper()
			in, ok := workloadstore.ParseIncoming(privacyObject(t, params)["workloadInfo"])
			if !ok {
				t.Fatal("invalid intervening fixture")
			}
			seedDepartedRun(t, b, in)
		}
		for _, state := range []string{"completed", "cancelled"} {
			in := departedRunInput(t, state, "host", "r1", "running", "peer-final")
			apply(markWorkloadTerminal(in.Info, state, time.UnixMilli(200), "actual outcome", 9))
		}
		moved := departedRunInput(t, "moved", "host", "r1", "running", "peer-new")
		movedFields := privacyObject(t, moved.Info)
		movedFields["seq"] = json.RawMessage(`11`)
		movedRaw, err := json.Marshal(movedFields)
		if err != nil {
			t.Fatal(err)
		}
		apply(departedRunEnvelope(t, movedRaw))
		if !b.workloads.Remove("host", "removed") {
			t.Fatal("intervening removal rejected")
		}
		generation := privacyObject(t, departedRunInput(t, "new-generation", "host", "r1", "running", "host").Info)
		generation["createdAt"] = json.RawMessage(`2`)
		generationRaw, err := json.Marshal(generation)
		if err != nil {
			t.Fatal(err)
		}
		apply(departedRunEnvelope(t, generationRaw))
		before := len(client.frames())
		releaseCapture()
		joinDepartedRun(t, b)
		for _, state := range []string{"completed", "cancelled"} {
			rec := departedRunRecord(t, b, "host", state, state)
			if rec.Seq != 9 || rec.CompletedAt != 200 || rec.ScheduledOn != "peer-final" {
				t.Fatalf("capture rewrote accepted terminal: %+v", rec)
			}
		}
		rec := departedRunRecord(t, b, "host", "moved", "failed")
		if rec.Seq != 12 || rec.ScheduledOn != "peer-new" {
			t.Fatalf("capture lost current metadata: %+v", rec)
		}
		departedRunRecord(t, b, "host", "new-generation", "running")
		if _, ok := b.workloads.Get("host", "removed"); ok {
			t.Fatal("capture resurrected removal")
		}
		if len(client.frames()) != before+1 || len(scheduler.frames()) != 1 || len(manager.frames()) != 1 {
			t.Fatal("stale candidates propagated")
		}
	})

	terminalTail := func(name, state string, replaced bool) {
		t.Run(name, func(t *testing.T) {
			b, client, scheduler, manager := privacyBroker(t)
			pp := &proxyProcess{done: make(chan struct{})}
			b.setProxyProcessHandle(pp)
			in := departedRunInput(t, "one", "host", "r1", "running", "host")
			b.forwardProxyNotificationFrom(pp, 0, 0, "workload:started", departedRunEnvelope(t, in.Info))
			entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var once sync.Once
			releaseTail := func() { once.Do(func() { close(release) }) }
			t.Cleanup(func() { releaseTail(); waitDepartedRun(t, done, "terminal callback cleanup") })
			var callbacks atomic.Int32
			b.beforeProxyWorkloadApply = func() {
				if callbacks.Add(1) == 1 {
					close(entered)
					<-release
				}
			}
			method := "workload:completed"
			if state == "cancelled" {
				method = "workload:errored"
			}
			params := markWorkloadTerminal(in.Info, state, time.UnixMilli(200), "actual outcome", 8)
			go func() { defer close(done); b.forwardProxyNotificationFrom(pp, 0, 0, method, params) }()
			waitDepartedRun(t, entered, "delivered terminal past initial source guard")
			close(pp.done)
			if replaced {
				b.clearProxyHandles()
				next := &proxyProcess{done: make(chan struct{})}
				b.setProxyProcessHandle(next)
				b.forwardProxyNotificationFrom(next, 0, 0, "workload:submitted", departedRunEnvelope(t, departedRunInput(t, "replacement", "host", "r2", "queued", "host").Info))
			}
			releaseTail()
			waitDepartedRun(t, done, "delivered terminal after Done")
			if replaced {
				departedRunRecord(t, b, "host", "one", "failed")
				departedRunRecord(t, b, "host", "replacement", "queued")
				if b.lastLocalRunID != "r2" || len(client.frames()) != 3 || len(scheduler.frames()) != 3 || len(manager.frames()) != 3 {
					t.Fatal("withdrawn tail changed replacement fence or output")
				}
				return
			}
			rec := departedRunRecord(t, b, "host", "one", state)
			if rec.Inferred || rec.Seq != 8 || rec.CompletedAt != 200 || b.lastLocalRunID != "r1" {
				t.Fatalf("delivered terminal metadata lost: %+v", rec)
			}
			for _, out := range []*syncBuffer{client, scheduler, manager} {
				frames := out.frames()
				if len(frames) != 2 {
					t.Fatalf("tail notifications = %d, want 2", len(frames))
				}
				last, ok := workloadstore.ParseIncoming(privacyObject(t, frames[1].Params)["workloadInfo"])
				if !ok || last.State != state || last.RunID != "r1" || last.Seq != 8 {
					t.Fatalf("delivered tail was not propagated: %+v", last)
				}
			}
			b.clearProxyHandles()
			b.captureDepartedRunWorkloads("tail already accepted")
			joinDepartedRun(t, b)
			departedRunRecord(t, b, "host", "one", state)
			if len(manager.frames()) != 2 {
				t.Fatal("departure rewrote real terminal")
			}
		})
	}
	terminalTail("completed tail retained before withdrawal", "completed", false)
	terminalTail("cancelled tail retained before withdrawal", "cancelled", false)
	terminalTail("completed tail rejected after replacement", "completed", true)
	terminalTail("cancelled tail rejected after replacement", "cancelled", true)

	t.Run("crash and deliberate stop hooks retire after unlock", func(t *testing.T) {
		for _, action := range []string{"crash", "stop", "shutdown"} {
			t.Run(action, func(t *testing.T) {
				b, _, _, manager := privacyBroker(t)
				seedDepartedRun(t, b, departedRunInput(t, "one", "host", "r1", "queued", "host"))
				switch action {
				case "crash":
					sup := newSupervisor(engines.ProxyComponent, defaultRestartPolicy(), nil)
					b.configureProxySupervisorCallbacks(sup)
					sup.onCrash(1)
				case "stop":
					b.proxySupMu.Lock()
					b.stopProxySupervisorLocked()
					b.proxySupMu.Unlock()
				case "shutdown":
					b.shutdownProxySupervision()
				}
				joinDepartedRun(t, b)
				departedRunRecord(t, b, "host", "one", "failed")
				departedRunFrames(t, manager, "workload:errored", map[string]string{"one": "failed"})
			})
		}
	})

	t.Run("capture before replacement excludes new records", func(t *testing.T) {
		b, _, _, manager := privacyBroker(t)
		seedDepartedRun(t, b, departedRunInput(t, "old", "host", "r1", "running", "host"))
		b.proxySupMu.Lock()
		b.stopProxySupervisorLocked()
		// A replacement's accepted store record can exist before emission.
		// Seeding it directly keeps this fixture free of I/O under proxySupMu.
		pp := &proxyProcess{done: make(chan struct{})}
		b.setProxyProcessHandle(pp)
		b.lastLocalRunID = "r2"
		seedDepartedRun(t, b, departedRunInput(t, "live", "host", "r2", "queued", "host"))
		b.proxySupMu.Unlock()
		joinDepartedRun(t, b)
		departedRunRecord(t, b, "host", "old", "failed")
		departedRunRecord(t, b, "host", "live", "queued")
		if b.lastLocalRunID != "r2" || len(manager.frames()) != 1 {
			t.Fatal("captured stop retired replacement or changed fence")
		}
	})

	t.Run("stop can join monitor after concurrent crash capture", func(t *testing.T) {
		b, _, _, manager := privacyBroker(t)
		seedDepartedRun(t, b, departedRunInput(t, "one", "host", "r1", "running", "host"))
		h := newFakeHandle()
		policy := defaultRestartPolicy()
		policy.maxAttempts = 0
		sup := newSupervisor(engines.ProxyComponent, policy, func() (supervisedHandle, error) { return h, nil })
		b.configureProxySupervisorCallbacks(sup)
		captured := make(chan struct{})
		sup.onExhausted = func(int) { close(captured) }
		sup.onSpawned = nil
		if err := sup.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(sup.Stop)
		b.setProxySupervisor(sup)
		b.proxySupMu.Lock()
		h.crash()
		select {
		case <-captured:
		case <-time.After(3 * time.Second):
			b.proxySupMu.Unlock()
			t.Fatal("monitor crash capture waited for supervision lock/output")
		}
		b.stopProxySupervisorLocked()
		b.proxySupMu.Unlock()
		joinDepartedRun(t, b)
		departedRunFrames(t, manager, "workload:errored", map[string]string{"one": "failed"})
	})

	shutdownBudget := func(name, destination string, armed bool, remaining time.Duration) {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t)
			b, _, _, _ := privacyBroker(t)
			path := filepath.Join(t.TempDir(), "shutdown-history.json")
			b.workloads.WithPersistence(path)
			reader, conn := net.Pipe()
			pipe := &departedRunObservedPipe{Conn: conn, entered: make(chan struct{}), finished: make(chan struct{})}
			if destination == "client" {
				b.codec = NewCodec(pipe)
			} else {
				b.setWorkloadMgr(&workloadManagerProcess{stdin: pipe})
			}
			seedDepartedRun(t, b, departedRunInput(t, "one", "host", "r1", "running", "peer"))
			teardownClock.mu.Lock()
			previous := teardownClock.start
			teardownClock.mu.Unlock()
			t.Cleanup(func() { setTeardownStart(previous) })
			returned := make(chan struct{})
			t.Cleanup(func() {
				// This reader never drains. Closing the transport is the only
				// cleanup escape for both the real emitter and unbounded-wait mutant.
				if err := reader.Close(); err != nil {
					t.Error(err)
				}
				if err := conn.Close(); err != nil {
					t.Error(err)
				}
				waitDepartedRun(t, returned, "bounded shutdown cleanup")
				joinDepartedRun(t, b)
			})
			clockStart := time.Time{}
			if armed {
				clockStart = time.Now().Add(-teardownBudget + remaining)
			}
			setTeardownStart(clockStart)
			bound := time.Second
			if !armed || remaining > workerSignalGrace {
				bound = workerSignalGrace + time.Second
			}
			started := time.Now()
			go func() { defer close(returned); b.shutdownProxySupervision() }()
			waitDepartedRun(t, pipe.entered, "retirement write to never-draining reader")
			select {
			case <-returned:
			case <-time.After(bound):
				t.Fatal("shutdown retirement ignored its capped allowance or remaining shared budget")
			}
			if time.Since(started) > bound {
				t.Fatal("shutdown retirement did not remain bounded")
			}
			select {
			case <-pipe.finished:
				t.Fatal("reader unexpectedly drained before cleanup")
			default:
			}
			if !strings.Contains(bufferText(logs), "proxy workload retirement exhausted its shutdown allowance") {
				t.Fatal("shutdown retirement allowance exhaustion was not reported")
			}
			if _, active := teardownRemaining(); active != armed {
				t.Fatal("retirement wait changed whether the shared teardown clock was armed")
			}
			rec := departedRunRecord(t, b, "host", "one", "failed")
			if rec.Inferred || rec.Seq != 8 {
				t.Fatal("output blockage lost accepted authoritative terminal")
			}
			if err := b.workloads.Flush(); err != nil {
				t.Fatal(err)
			}
			restored := workloadstore.New().WithPersistence(path)
			if err := restored.Load(); err != nil {
				t.Fatal(err)
			}
			saved, ok := restored.Get("host", "one")
			if !ok || saved.State != "failed" || saved.Inferred || saved.Seq != 8 {
				t.Fatal("blocked output lost accepted terminal history")
			}
		})
	}
	shutdownBudget("shutdown unread client uses nearly-spent shared budget", "client", true, 100*time.Millisecond)
	shutdownBudget("shutdown unread manager uses nearly-spent shared budget", "manager", true, 100*time.Millisecond)
	shutdownBudget("shutdown unread client honors exhausted shared budget", "client", true, -time.Millisecond)
	shutdownBudget("shutdown unread manager honors exhausted shared budget", "manager", true, -time.Millisecond)
	shutdownBudget("shutdown unread client caps ample shared budget", "client", true, teardownBudget)
	shutdownBudget("shutdown unread manager caps ample shared budget", "manager", true, teardownBudget)
	shutdownBudget("shutdown unread client caps unarmed clock allowance", "client", false, 0)
	shutdownBudget("shutdown unread manager caps unarmed clock allowance", "manager", false, 0)

	shutdownUnarmedDrain := func(name, destination string) {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t)
			b, client, scheduler, manager := privacyBroker(t)
			reader, conn := net.Pipe()
			pipe := &departedRunObservedPipe{Conn: conn, entered: make(chan struct{}), finished: make(chan struct{})}
			if destination == "client" {
				b.codec = NewCodec(pipe)
			} else {
				b.setWorkloadMgr(&workloadManagerProcess{stdin: pipe})
			}
			seedDepartedRun(t, b, departedRunInput(t, "one", "host", "r1", "running", "peer"))
			teardownClock.mu.Lock()
			previous := teardownClock.start
			teardownClock.mu.Unlock()
			t.Cleanup(func() { setTeardownStart(previous) })
			returned, drained := make(chan struct{}), make(chan struct{})
			readerStarted := false
			t.Cleanup(func() {
				if err := reader.Close(); err != nil {
					t.Error(err)
				}
				if err := conn.Close(); err != nil {
					t.Error(err)
				}
				waitDepartedRun(t, returned, "unarmed shutdown cleanup")
				joinDepartedRun(t, b)
				if readerStarted {
					waitDepartedRun(t, drained, "unarmed reader cleanup")
				}
			})
			setTeardownStart(time.Time{})
			go func() { defer close(returned); b.shutdownProxySupervision() }()
			waitDepartedRun(t, pipe.entered, "unarmed retirement write before drain")
			select {
			case <-returned:
				t.Fatal("unarmed retirement returned before blocked output was released")
			case <-time.After(100 * time.Millisecond):
			}
			var frame struct {
				Method string          `json:"method"`
				Params json.RawMessage `json:"params"`
			}
			var readErr error
			readerStarted = true
			go func() {
				defer close(drained)
				readErr = json.NewDecoder(reader).Decode(&frame)
			}()
			waitDepartedRun(t, returned, "unarmed retirement after drain")
			waitDepartedRun(t, drained, "unarmed terminal frame")
			if readErr != nil {
				t.Fatal(readErr)
			}
			select {
			case <-pipe.finished:
			default:
				t.Fatal("unarmed retirement returned before its output write finished")
			}
			b.proxySupWorkMu.Lock()
			done := b.proxyRetireDone
			b.proxySupWorkMu.Unlock()
			if done != nil {
				select {
				case <-done:
				default:
					t.Fatal("unarmed retirement returned before the drained emitter finished")
				}
			}
			method := "workloads:upsert"
			other := manager
			if destination == "manager" {
				method, other = "workload:errored", client
			}
			info := privacyObject(t, frame.Params)["workloadInfo"]
			in, ok := workloadstore.ParseIncoming(info)
			if !ok || frame.Method != method || in.ID != "one" || in.RunID != "r1" || in.State != "failed" || in.Seq != 8 || string(privacyObject(t, info)["error"]) != `"proxy exited before the request finished"` {
				t.Fatalf("unarmed drain lost the accepted terminal: %+v", frame)
			}
			if len(scheduler.frames()) != 1 || len(other.frames()) != 1 {
				t.Fatal("unarmed retirement did not finish its other fanout")
			}
			if _, active := teardownRemaining(); active {
				t.Fatal("unarmed retirement wait armed the shared teardown clock")
			}
			if strings.Contains(bufferText(logs), "proxy workload retirement exhausted its shutdown allowance") {
				t.Fatal("drained retirement incorrectly reported allowance exhaustion")
			}
		})
	}
	shutdownUnarmedDrain("shutdown unarmed client waits for drain", "client")
	shutdownUnarmedDrain("shutdown unarmed manager waits for drain", "manager")

	t.Run("blocked output leaves supervision and worker locks available", func(t *testing.T) {
		for _, destination := range []string{"client", "manager"} {
			t.Run(destination, func(t *testing.T) {
				b, _, _, _ := privacyBroker(t)
				writer := &departedRunBlockingWriter{output: &syncBuffer{}, entered: make(chan struct{}), release: make(chan struct{})}
				var once sync.Once
				release := func() { once.Do(func() { close(writer.release) }) }
				stopReturned, stopDone, unlock := make(chan struct{}), make(chan struct{}), make(chan struct{})
				var unlockOnce sync.Once
				allowUnlock := func() { unlockOnce.Do(func() { close(unlock) }) }
				t.Cleanup(func() {
					// Always release blocked I/O before joining the stop owner,
					// including when an actual synchronous-stop mutant entered it.
					release()
					allowUnlock()
					waitDepartedRun(t, stopDone, "stop owner cleanup")
					joinDepartedRun(t, b)
				})
				if destination == "client" {
					b.codec = NewCodec(writer)
				} else {
					b.setWorkloadMgr(&workloadManagerProcess{stdin: writer})
				}
				seedDepartedRun(t, b, departedRunInput(t, "one", "host", "r1", "running", "host"))
				go func() {
					defer close(stopDone)
					b.proxySupMu.Lock()
					defer b.proxySupMu.Unlock()
					b.stopProxySupervisorLocked()
					close(stopReturned)
					<-unlock
				}()
				select {
				case <-stopReturned:
				case <-writer.entered:
					t.Fatal("stop synchronously emitted retirement before returning")
				case <-time.After(2 * time.Second):
					t.Fatal("stop did not return within the bounded capture interval")
				}
				select {
				case <-writer.entered:
					t.Fatal("retirement emitted under supervision lock")
				default:
				}
				allowUnlock()
				waitDepartedRun(t, stopDone, "stop owner unlock")
				waitDepartedRun(t, writer.entered, "blocked retirement output")
				if !b.proxySupMu.TryLock() {
					t.Fatal("retirement retained proxySupMu")
				}
				b.proxySupMu.Unlock()
				if !b.workersMu.TryLock() {
					t.Fatal("retirement retained workersMu")
				}
				b.workersMu.Unlock()
				crashes := newSupervisor(engines.ProxyComponent, defaultRestartPolicy(), nil)
				b.configureProxySupervisorCallbacks(crashes)
				b.proxySupWorkMu.Lock()
				emitter := b.proxyRetireDone
				b.proxySupWorkMu.Unlock()
				for range 200 {
					crashes.onCrash(1)
				}
				b.proxySupWorkMu.Lock()
				stable := b.proxyRetireDone == emitter && len(b.proxyRetirePending) == 0 && !b.proxySupWorking && !b.proxySupDirty
				b.proxySupWorkMu.Unlock()
				if !stable {
					t.Fatal("blocked-output crash callbacks grew pending work or occupied supervision")
				}
				// A new listener reconciliation must still stop its supervisor
				// while the old retirement is blocked on output.
				h := newFakeHandle()
				sup := newSupervisor("departed-test", defaultRestartPolicy(), func() (supervisedHandle, error) { return h, nil })
				if err := sup.Start(); err != nil {
					t.Fatal(err)
				}
				t.Cleanup(sup.Stop)
				pp := &proxyProcess{done: make(chan struct{})}
				pp.bringUpDone.Store(true)
				pp.listenerOnly.Store(true)
				b.proxySupMu.Lock()
				b.proxySupPhase = proxySupLive
				b.setProxyProcessHandle(pp)
				b.setProxySupervisor(sup)
				b.proxySupMu.Unlock()
				b.reconcileProxySupervision()
				waitDepartedRun(t, h.stoppedC, "new supervision work during blocked output")
				deadline := time.Now().Add(3 * time.Second)
				for {
					b.proxySupWorkMu.Lock()
					idle := !b.proxySupWorking && !b.proxySupDirty
					b.proxySupWorkMu.Unlock()
					if idle {
						break
					}
					if time.Now().After(deadline) {
						t.Fatal("supervision worker did not finish while output blocked")
					}
					runtime.Gosched()
				}
				release()
				joinDepartedRun(t, b)
				departedRunRecord(t, b, "host", "one", "failed")
			})
		}
	})

	t.Run("repeated captures coalesce without policy work or restart renewal", func(t *testing.T) {
		b, _, _, _ := privacyBroker(t)
		seedDepartedRun(t, b, departedRunInput(t, "one", "host", "r1", "running", "host"))
		b.proxySupExhausted.Store(true)
		b.proxySupDirty = true
		b.proxySupMu.Lock()
		sup := newSupervisor(engines.ProxyComponent, defaultRestartPolicy(), nil)
		b.configureProxySupervisorCallbacks(sup)
		for range 200 {
			sup.onCrash(1)
		}
		b.proxySupWorkMu.Lock()
		pending, done := len(b.proxyRetirePending), b.proxyRetireDone
		dirty, working := b.proxySupDirty, b.proxySupWorking
		b.proxySupWorkMu.Unlock()
		// All retirement emitters are parked at the same lock boundary. Count
		// stack blocks rather than wall-clock scheduling or a sleep as proof.
		stack := make([]byte, 1<<20)
		n := runtime.Stack(stack, true)
		emitters := 0
		for _, block := range strings.Split(string(stack[:n]), "\n\n") {
			if strings.Contains(block, "(*Broker).resumeProxyRetirementsLocked.func1") {
				emitters++
			}
		}
		b.proxySupMu.Unlock()
		joinDepartedRun(t, b)
		if n == len(stack) {
			t.Fatal("emitter stack evidence was truncated")
		}
		if pending != 1 || done == nil || !dirty || working || !b.proxySupExhausted.Load() || emitters != 1 {
			t.Fatalf("capture state pending=%d emitter=%v dirty=%v working=%v exhausted=%v goroutines=%d", pending, done != nil, dirty, working, b.proxySupExhausted.Load(), emitters)
		}
		departedRunRecord(t, b, "host", "one", "failed")
	})
}
