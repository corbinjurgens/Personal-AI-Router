// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"io"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-ui-broker/workloadstore"
)

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
		b.retireDepartedRunWorkloads("", "test departure")
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
		b.retireDepartedRunWorkloads("", "duplicate departure")
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

	t.Run("replacement run stays live and synthetic frames leave fence alone", func(t *testing.T) {
		b, client, _, _ := privacyBroker(t)
		seedDepartedRun(t, b, departedRunInput(t, "old", "host", "r1", "running", "peer"))
		live := departedRunInput(t, "live", "host", "r2", "queued", "host")
		b.routeProxyWorkload("workload:submitted", departedRunEnvelope(t, live.Info))
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
		// This sentinel would be retired by an unnecessary second r2 scan.
		seedDepartedRun(t, b, departedRunInput(t, "sentinel", "host", "r3", "running", "peer"))
		b.routeProxyWorkload("workload:started", departedRunEnvelope(t, departedRunInput(t, "live", "host", "r2", "running", "host").Info))
		departedRunRecord(t, b, "host", "sentinel", "running")
		if b.lastLocalRunID != "r2" {
			t.Fatalf("fence = %s, want r2", b.lastLocalRunID)
		}
		b.retireDepartedRunWorkloads("r2", "synthetic departure")
		departedRunRecord(t, b, "host", "live", "running")
		departedRunRecord(t, b, "host", "sentinel", "failed")
		if b.lastLocalRunID != "r2" {
			t.Fatalf("synthetic retirement changed fence to %s", b.lastLocalRunID)
		}
	})

	t.Run("notification failure keeps authoritative terminal and manager forward", func(t *testing.T) {
		b, _, _, manager := privacyBroker(t)
		b.codec = NewCodec(departedRunFailedWriter{})
		seedDepartedRun(t, b, departedRunInput(t, "one", "host", "r1", "running", "host"))
		b.retireDepartedRunWorkloads("", "failed client output")
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
		t.Cleanup(releaseOld)
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
		// Keep the fixture's old Done open to prove the source identity check;
		// the retained-handle case separately proves the Done check.
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
		b, client, _, manager := privacyBroker(t)
		for _, id := range []string{"completed", "cancelled", "moved", "removed", "new-generation"} {
			seedDepartedRun(t, b, departedRunInput(t, id, "host", "r1", "running", "host"))
		}
		pending := b.departedRunCandidates("", "captured")
		for _, state := range []string{"completed", "cancelled"} {
			in := departedRunInput(t, state, "host", "r1", "running", "peer-final")
			b.emitWorkloadEvent("workloads:upsert", markWorkloadTerminal(in.Info, state, time.UnixMilli(200), "actual outcome", 9))
		}
		moved := departedRunInput(t, "moved", "host", "r1", "running", "peer-new")
		movedFields := privacyObject(t, moved.Info)
		movedFields["seq"] = json.RawMessage(`11`)
		movedRaw, err := json.Marshal(movedFields)
		if err != nil {
			t.Fatal(err)
		}
		b.emitWorkloadEvent("workloads:upsert", departedRunEnvelope(t, movedRaw))
		b.emitWorkloadEvent("workloads:remove", json.RawMessage(`{"workloadId":"removed","originatedFrom":"host"}`))
		generation := privacyObject(t, departedRunInput(t, "new-generation", "host", "r1", "running", "host").Info)
		generation["createdAt"] = json.RawMessage(`2`)
		generationRaw, err := json.Marshal(generation)
		if err != nil {
			t.Fatal(err)
		}
		b.emitWorkloadEvent("workloads:upsert", departedRunEnvelope(t, generationRaw))
		before := len(client.frames())
		b.workloadEmitMu.Lock()
		frames := b.applyDepartedRunWorkloadsLocked(pending)
		for _, frame := range frames {
			b.publishProxyWorkloadLocked(frame.method, frame.params)
		}
		b.workloadEmitMu.Unlock()
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
		if len(frames) != 1 || len(client.frames()) != before+1 || len(manager.frames()) != 1 {
			t.Fatal("stale candidates propagated")
		}
	})

	t.Run("done handle is rechecked even before withdrawal", func(t *testing.T) {
		b, client, _, manager := privacyBroker(t)
		pp := &proxyProcess{done: make(chan struct{})}
		b.setProxyProcessHandle(pp)
		b.lastLocalRunID = "r1"
		seedDepartedRun(t, b, departedRunInput(t, "one", "host", "r1", "running", "host"))
		entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
		var once sync.Once
		releaseCallback := func() { once.Do(func() { close(release) }) }
		t.Cleanup(releaseCallback)
		b.beforeProxyWorkloadApply = func() { close(entered); <-release }
		params := departedRunEnvelope(t, departedRunInput(t, "late", "host", "r2", "queued", "host").Info)
		go func() { defer close(done); b.forwardProxyNotificationFrom(pp, 0, 0, "workload:submitted", params) }()
		waitDepartedRun(t, entered, "callback before Done")
		close(pp.done)
		releaseCallback()
		waitDepartedRun(t, done, "callback after Done")
		departedRunRecord(t, b, "host", "one", "running")
		if b.lastLocalRunID != "r1" || b.workloads.Len() != 1 || len(client.frames()) != 0 || len(manager.frames()) != 0 {
			t.Fatal("exited handle changed fence, store or output")
		}
	})

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
				b.waitProxyRetirements()
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
		b.waitProxyRetirements()
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
		b.waitProxyRetirements()
		departedRunFrames(t, manager, "workload:errored", map[string]string{"one": "failed"})
	})

	t.Run("blocked output leaves supervision and worker locks available", func(t *testing.T) {
		for _, destination := range []string{"client", "manager"} {
			t.Run(destination, func(t *testing.T) {
				b, _, _, _ := privacyBroker(t)
				writer := &departedRunBlockingWriter{output: &syncBuffer{}, entered: make(chan struct{}), release: make(chan struct{})}
				var once sync.Once
				release := func() { once.Do(func() { close(writer.release) }) }
				t.Cleanup(func() { release(); b.waitProxyRetirements() })
				if destination == "client" {
					b.codec = NewCodec(writer)
				} else {
					b.setWorkloadMgr(&workloadManagerProcess{stdin: writer})
				}
				seedDepartedRun(t, b, departedRunInput(t, "one", "host", "r1", "running", "host"))
				b.proxySupMu.Lock()
				b.stopProxySupervisorLocked()
				select {
				case <-writer.entered:
					b.proxySupMu.Unlock()
					t.Fatal("retirement emitted under supervision lock")
				default:
				}
				b.proxySupMu.Unlock()
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
				b.waitProxyRetirements()
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
		b.waitProxyRetirements()
		if n == len(stack) {
			t.Fatal("emitter stack evidence was truncated")
		}
		if pending != 1 || done == nil || !dirty || working || !b.proxySupExhausted.Load() || emitters != 1 {
			t.Fatalf("capture state pending=%d emitter=%v dirty=%v working=%v exhausted=%v goroutines=%d", pending, done != nil, dirty, working, b.proxySupExhausted.Load(), emitters)
		}
		departedRunRecord(t, b, "host", "one", "failed")
	})
}
