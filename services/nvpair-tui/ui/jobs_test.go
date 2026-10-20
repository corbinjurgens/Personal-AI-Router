// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"nvpair-tui/rpc"
)

func TestReplacementProxyReadinessStaysUnknownUntilReported(t *testing.T) {
	f := newReconnectUIFixture(t)
	v := newJobsView(f.client)
	v.SetSize(180, 35)
	m := New(f.client, nil, []View{v})
	t.Cleanup(v.close)
	idx := v.proxy.indexForEngine("ollama")
	if idx < 0 {
		t.Fatal("Ollama proxy missing from the tracker")
	}
	proxyRow := func() string {
		t.Helper()
		rows := strings.Split(v.proxy.strip(), "   ")
		if idx >= len(rows) {
			t.Fatal("proxy strip omitted the tracked facade")
		}
		row := rows[idx]
		if !strings.Contains(v.View(), row) {
			t.Fatal("Jobs did not render the tracked proxy status")
		}
		return row
	}
	assertUnknown := func(stage string, port int) {
		t.Helper()
		row := proxyRow()
		if strings.Contains(row, "down") || (!strings.Contains(row, "unknown") && !strings.Contains(row, "?")) {
			t.Errorf("%s: unconfirmed proxy rendered as %q, want unknown", stage, row)
		}
		if got, usable := v.proxy.portForEngine("ollama"); usable || got != port {
			t.Errorf("%s: endpoint port=%d usable=%v, want %d/false", stage, got, usable, port)
		}
	}
	readStatus := func(reply *rpc.Message) {
		t.Helper()
		if err := f.peer.SetDeadline(time.Now().Add(time.Second)); err != nil {
			t.Fatal(err)
		}
		request, result := terminalControlsReply(t, f.broker,
			sessionCommand(f.client, f.client.Generation(), v.proxy.statusCmd(f.client, idx)), reply)
		if request.Method != "ollama-proxy:get-status" {
			t.Fatalf("proxy status request = %q", request.Method)
		}
		updated, _ := m.Update(result)
		m = updated.(Model)
	}
	readStatus(&rpc.Message{Result: []byte(`{"ready":true,"port":11434}`)})
	if port, usable := v.proxy.portForEngine("ollama"); !usable || port != 11434 {
		t.Fatal("initial successful status did not establish a usable endpoint")
	}
	if err := f.peer.Close(); err != nil {
		t.Fatal(err)
	}
	updated, _ := m.Update(f.event(t))
	m = updated.(Model)
	if !m.disconnected || m.ready {
		t.Fatal("transport loss did not reach the retained shell")
	}
	assertUnknown("disconnected", 0)
	conn, peer := net.Pipe()
	f.peer, f.broker = peer, rpc.NewCodec(peer, peer)
	select {
	case f.dials <- conn:
	case <-time.After(time.Second):
		t.Fatal("client did not dial the replacement")
	}
	if err := f.broker.Write(&rpc.Message{JSONRPC: "2.0", Method: "app:ready", Params: []byte(`{"version":"replacement"}`)}); err != nil {
		t.Fatal(err)
	}
	updated, _ = m.Update(f.event(t))
	m = updated.(Model)
	if m.disconnected || !m.ready {
		t.Fatal("replacement ready did not recover the retained shell")
	}
	assertUnknown("ready before status", 0)
	readStatus(&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "status unavailable"}})
	assertUnknown("failed status read", 0)
	readStatus(&rpc.Message{Result: []byte(`{"ready":false,"port":0}`)})
	if !strings.Contains(proxyRow(), "down") {
		t.Error("explicit ready:false did not render the proxy down")
	}
	if _, usable := v.proxy.portForEngine("ollama"); usable {
		t.Error("explicit ready:false exposed a usable endpoint")
	}
	readStatus(&rpc.Message{Result: []byte(`{"ready":true,"port":12999}`)})
	if port, usable := v.proxy.portForEngine("ollama"); !usable || port != 12999 || !strings.Contains(proxyRow(), ":12999") {
		t.Error("successful ready status did not render its usable endpoint")
	}
	readStatus(&rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "status unavailable"}})
	assertUnknown("failed read after known endpoint", 12999)
	readStatus(&rpc.Message{Result: []byte(`{"ready":false,"port":0}`)})
	if port, usable := v.proxy.portForEngine("ollama"); usable || port != 12999 || !strings.Contains(proxyRow(), "down") {
		t.Error("explicit down status did not preserve the unusable last-known port")
	}
}

// TestJobsHistoryIsBounded is the guard for a leak in a program meant to be left
// running. The broker never tells a client to forget a job, so without a cap
// every job the cluster has ever run accumulates for the life of the process,
// and the "show all" table grows with it.
func TestJobsHistoryIsBounded(t *testing.T) {
	v := newJobsView(nil)
	for i := range maxFinishedJobs + 50 {
		v.upsert(workload{
			ID:             fmt.Sprintf("job-%d", i),
			OriginatedFrom: "node",
			State:          "completed",
		})
	}

	if got := len(v.byKey); got > maxFinishedJobs {
		t.Errorf("kept %d finished jobs, want at most %d", got, maxFinishedJobs)
	}
	if len(v.order) != len(v.byKey) {
		t.Errorf("order (%d) and index (%d) disagree after eviction, so a key leaked",
			len(v.order), len(v.byKey))
	}

	// Eviction is oldest-first, so the most recent job must survive.
	newest := workloadKey(workload{OriginatedFrom: "node", ID: fmt.Sprintf("job-%d", maxFinishedJobs+49)})
	if _, ok := v.byKey[newest]; !ok {
		t.Error("the newest finished job was evicted; eviction is not oldest-first")
	}
}

// TestJobsNeverEvictsActiveWork checks the cap only reclaims finished jobs. An
// in-flight job is the thing the operator is watching, and dropping one would
// make a busy cluster look idle.
func TestJobsNeverEvictsActiveWork(t *testing.T) {
	v := newJobsView(nil)
	v.upsert(workload{ID: "live", OriginatedFrom: "node", State: "running"})
	for i := range maxFinishedJobs + 50 {
		v.upsert(workload{
			ID:             fmt.Sprintf("done-%d", i),
			OriginatedFrom: "node",
			State:          "completed",
		})
	}

	if _, ok := v.byKey[workloadKey(workload{OriginatedFrom: "node", ID: "live"})]; !ok {
		t.Error("a running job was evicted by history trimming")
	}
}

// TestJobsUpsertReplacesRatherThanDuplicating checks a job progressing through
// its states occupies one row, not one per update.
func TestJobsUpsertReplacesRatherThanDuplicating(t *testing.T) {
	v := newJobsView(nil)
	for _, state := range []string{"queued", "running", "completed"} {
		v.upsert(workload{ID: "j1", OriginatedFrom: "node", State: state})
	}

	if len(v.order) != 1 {
		t.Errorf("one job produced %d rows across its state changes", len(v.order))
	}
	if got := v.byKey[workloadKey(workload{OriginatedFrom: "node", ID: "j1"})].State; got != "completed" {
		t.Errorf("state = %q, want the latest", got)
	}
}

// TestJobsKeyIsScopedByOrigin checks two nodes can use the same job id without
// colliding, since ids are only unique to the node that issued them.
func TestJobsKeyIsScopedByOrigin(t *testing.T) {
	v := newJobsView(nil)
	v.upsert(workload{ID: "1", OriginatedFrom: "node-a", State: "running"})
	v.upsert(workload{ID: "1", OriginatedFrom: "node-b", State: "running"})

	if len(v.order) != 2 {
		t.Errorf("same id from two nodes collapsed into %d row(s)", len(v.order))
	}
}

// TestJobsKeyIsScopedByEngineAndRun is the same guard one level down. The ID
// is a counter each engine's facade starts at 1 and every proxy restart resets,
// so a concurrent Ollama and LM Studio job, or a job from before a restart and
// one after it, share an origin and an ID while being different work.
func TestJobsKeyIsScopedByEngineAndRun(t *testing.T) {
	v := newJobsView(nil)
	v.upsert(workload{ID: "1", OriginatedFrom: "node", Engine: "ollama", RunID: "r1", State: "running"})
	v.upsert(workload{ID: "1", OriginatedFrom: "node", Engine: "lmstudio", RunID: "r2", State: "running"})
	v.upsert(workload{ID: "1", OriginatedFrom: "node", Engine: "ollama", RunID: "r3", State: "queued"})

	if len(v.order) != 3 {
		t.Errorf("three distinct jobs sharing an id collapsed into %d row(s)", len(v.order))
	}
}

// TestJobsRemovalDropsEveryGeneration checks a removal takes out every job it
// names. It carries only the origin and ID, so it cannot say which engine or
// run it meant — the broker's own store drops them all, and so must this.
func TestJobsRemovalDropsEveryGeneration(t *testing.T) {
	v := newJobsView(nil)
	v.upsert(workload{ID: "1", OriginatedFrom: "node", Engine: "ollama", RunID: "r1", State: "completed"})
	v.upsert(workload{ID: "1", OriginatedFrom: "node", Engine: "lmstudio", RunID: "r2", State: "completed"})
	v.upsert(workload{ID: "2", OriginatedFrom: "node", Engine: "ollama", RunID: "r1", State: "completed"})

	v.remove(workloadRef{origin: "node", id: "1"})

	if len(v.order) != 1 || len(v.byKey) != 1 {
		t.Fatalf("after removing id 1: %d ordered, %d indexed, want only id 2 left",
			len(v.order), len(v.byKey))
	}
	if w := v.byKey[v.order[0]]; w.ID != "2" {
		t.Errorf("the surviving job is %q, want 2", w.ID)
	}
}

// TestRemovedJobStaysRemovedWhenTheSnapshotLandsLater is the regression guard
// for a job coming back from the dead at startup.
//
// The snapshot reply and the pushes reach the view by different paths, so a
// removal can be handled before a snapshot taken while the job still existed.
// Merging that snapshot re-added the job, and nothing would ever remove it
// again.
func TestRemovedJobStaysRemovedWhenTheSnapshotLandsLater(t *testing.T) {
	v := newJobsView(nil)
	gone := workload{ID: "1", OriginatedFrom: "node", Engine: "ollama", RunID: "r1", State: "completed"}
	kept := workload{ID: "2", OriginatedFrom: "node", Engine: "ollama", RunID: "r1", State: "running"}

	v.remove(workloadRef{origin: "node", id: "1"})
	v.Update(workloadsLoadedMsg{workloads: []workload{gone, kept}})

	if _, back := v.byKey[workloadKey(gone)]; back {
		t.Error("a job removed before the snapshot landed was restored by it")
	}
	if _, ok := v.byKey[workloadKey(kept)]; !ok {
		t.Error("the snapshot's other job was not merged")
	}

	// Once the baseline is in, the list is live and a new job with a reused
	// id is simply new work.
	again := workload{ID: "1", OriginatedFrom: "node", Engine: "ollama", RunID: "r2", State: "running"}
	v.upsert(again)
	if _, ok := v.byKey[workloadKey(again)]; !ok {
		t.Error("a later job reusing a removed id was refused")
	}
	if v.removedEarly != nil {
		t.Error("removals were still being remembered after the baseline landed")
	}
}

// TestNewestJobsLead checks new work is at the top of the table. In arrival
// order, a backlog of older queued jobs pushed each new one below the fold.
func TestNewestJobsLead(t *testing.T) {
	v := newJobsView(nil)
	v.upsert(workload{ID: "old", OriginatedFrom: "n", State: "queued", CreatedAt: 1000})
	v.upsert(workload{ID: "new", OriginatedFrom: "n", State: "queued", CreatedAt: 3000})
	v.upsert(workload{ID: "mid", OriginatedFrom: "n", State: "queued", CreatedAt: 2000})

	rows := v.table.Rows()
	got := []string{rows[0][0], rows[1][0], rows[2][0]}
	if got[0] != "new" || got[1] != "mid" || got[2] != "old" {
		t.Errorf("row order %v, want newest first", got)
	}
}

// TestJobIDTellsSimultaneousJobsApart checks two jobs for the same model, from
// the same node, started at the same moment, do not read as one row twice.
func TestJobIDTellsSimultaneousJobsApart(t *testing.T) {
	cases := map[string]string{"1": "1", "123456": "123456", "burst-1042": "…-1042"}
	for id, want := range cases {
		if got := shortJobID(id); got != want {
			t.Errorf("shortJobID(%q) = %q, want %q", id, got, want)
		}
	}

	v := newJobsView(nil)
	v.upsert(workload{ID: "7", Model: "m", OriginatedFrom: "n", State: "running", CreatedAt: 5})
	v.upsert(workload{ID: "8", Model: "m", OriginatedFrom: "n", State: "running", CreatedAt: 5})
	rows := v.table.Rows()
	if rows[0][0] == rows[1][0] {
		t.Errorf("two different jobs show the same ID %q", rows[0][0])
	}
}

// TestFailedStartupReadsAreRetried checks a subscription or identity read that
// failed is tried again, rather than leaving the table empty, or this
// machine's jobs unnamed, for the rest of the session.
func TestFailedStartupReadsAreRetried(t *testing.T) {
	v := newJobsView(nil)
	v.Update(workloadsSubscribedMsg{err: errFake{}})
	v.Update(jobsIdentityMsg{err: errFake{}})

	var cmd tea.Cmd
	for range proxyPollTicks {
		cmd = v.Update(TickMsg{})
	}
	batch, ok := cmd().(tea.BatchMsg)
	if !ok {
		t.Fatalf("the poll tick returned %T, want a batch", cmd())
	}
	// The proxy refresh, the feed retry, and the identity retry.
	if len(batch) < 3 {
		t.Errorf("the poll tick scheduled %d command(s), want the two retries alongside the proxy read", len(batch))
	}

	// Once both have succeeded, nothing is left to retry.
	v.Update(workloadsSubscribedMsg{})
	v.Update(workloadsLoadedMsg{})
	v.Update(jobsIdentityMsg{id: clusterIdentity{NodeUUID: "self"}})
	if !v.subscribed || !v.baselined || !v.identified {
		t.Errorf("after both reads succeeded: subscribed=%v baselined=%v identified=%v",
			v.subscribed, v.baselined, v.identified)
	}
}

func TestJobsBaselineWaitsForSuccessfulSubscription(t *testing.T) {
	f := newReconnectUIFixture(t)
	f.serve(t, map[string]json.RawMessage{
		"workloads:get-initial": json.RawMessage(`{"workloads":[{"id":"fresh","originatedFrom":"self"}]}`),
	})
	v := newJobsView(f.client)
	ack := v.feedCmd()()
	if f.count("workloads:subscribe") != 1 || f.count("workloads:get-initial") != 0 {
		t.Fatal("baseline was requested before subscription acknowledgement")
	}
	load := v.Update(ack)
	if load == nil {
		t.Fatal("successful subscription did not schedule baseline")
	}
	v.Update(load())
	if !v.baselined || len(v.byKey) != 1 || v.byKey[v.order[0]].ID != "fresh" {
		t.Fatal("acknowledged subscription did not load the fresh job")
	}
	if cmd := v.Update(workloadsSubscribedMsg{}); cmd == nil {
		t.Fatal("successful resubscription did not schedule a fresh baseline")
	}
	if cmd := v.Update(workloadsSubscribedMsg{err: errFake{}}); cmd != nil {
		t.Fatal("failed subscription scheduled an unordered baseline")
	}
}

type observedDemoContext struct {
	context.Context
	starts atomic.Int32
}

func (c *observedDemoContext) Err() error {
	c.starts.Add(1)
	return c.Context.Err()
}

func TestSessionDisconnectStopsDemoWithoutCancellingSubmittedWork(t *testing.T) {
	f := newReconnectUIFixture(t)
	v := newJobsView(f.client)
	runner := v.demo
	submittedContext := runner.ctx
	// A cancelled command context makes this probe safe even if the invocation
	// guard regresses: exec checks Err before starting any dispatcher process.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	observed := &observedDemoContext{Context: ctx}
	runner.ctx = observed
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	runner.executable = executable
	runner.status = demoRunning
	runner.started = time.Now()
	runner.schedule = []demoRequest{{}, {at: time.Hour}}
	v.demoGeneration = 1
	queued := v.Update(TickMsg{})
	if queued == nil || runner.next != 1 {
		t.Fatal("Jobs did not queue the due demo submission")
	}
	m := New(f.client, nil, []View{v})
	f.replace(t, &m)
	if v.demo != runner || runner.status != demoIdle || len(runner.schedule) != 0 || submittedContext.Err() != nil {
		t.Fatal("disconnect retained future demo schedule or cancelled submitted work")
	}
	if msg := queued(); msg != nil || observed.starts.Load() != 0 {
		t.Fatal("queued old-session demo work started on replacement")
	}
	if msg := submitDemoRequest(observed, executable, demoRequest{}, 0)(); msg != nil || observed.starts.Load() != 1 {
		t.Fatal("cancelled-context control did not detect an attempted dispatcher start")
	}
	if cmd := v.Update(TickMsg{}); cmd != nil {
		t.Fatal("stopped demo scheduled another submission")
	}
	started := false
	fresh := sessionInvocation(f.client, 2, func() tea.Msg {
		started = true
		return demoSpawnedMsg{}
	})
	if _, ok := fresh().(demoSpawnedMsg); !ok || !started {
		t.Fatal("replacement session rejected freshly created demo work")
	}
}

var _ View = (*jobsView)(nil)
