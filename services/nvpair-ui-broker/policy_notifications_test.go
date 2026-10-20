// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func TestIntentSubscribedRelayPreservesSnapshot(t *testing.T) {
	b, out := newPolicyBroker(t)
	source := b.activateEngineManagerSource()
	id := json.RawMessage(`1`)
	b.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: "engine:subscribe"})
	raw := json.RawMessage(`{"enabledByEngine":{"ollama":true,"lmstudio":false,"llamacpp":false},"offByEngine":{"lmstudio":"local","llamacpp":"peer"},"relayMarker":"unchanged"}`)
	b.forwardEngineNotificationFrom(source, engineIntentChanged, raw)
	frame := out.waitFrame(t, "subscribed intent snapshot", func(m Message) bool { return m.Method == engineIntentChanged })
	requireIntentRelayPayload(t, frame.Params, raw)
}

// requireIntentRelayPayload compares the complete decoded payload so missing
// maps or a relay that reconstructs only known fields cannot pass.
func requireIntentRelayPayload(t *testing.T, got, want json.RawMessage) {
	t.Helper()
	var actual, expected map[string]json.RawMessage
	if err := json.Unmarshal(got, &actual); err != nil {
		t.Fatalf("decode relayed intent: %v", err)
	}
	if err := json.Unmarshal(want, &expected); err != nil {
		t.Fatalf("decode expected intent: %v", err)
	}
	if _, present := actual["offByEngine"]; !present {
		t.Fatal("relayed intent omits offByEngine")
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("relayed intent = %s, want %s", got, want)
	}
}

// policyChangedEvent is one policy:changed this node announced for itself.
type policyChangedEvent struct {
	Policy nodepolicy.Policy `json:"policy"`
	policyActivation
}

// selfPolicyChanges lists every policy:changed this node announced for itself,
// in the order clients received them.
func selfPolicyChanges(t *testing.T, frames []Message) []policyChangedEvent {
	t.Helper()
	var events []policyChangedEvent
	for _, m := range frames {
		if m.Method != notifyPolicyChanged {
			continue
		}
		var event struct {
			NodeID string `json:"nodeId"`
			policyChangedEvent
		}
		if err := json.Unmarshal(m.Params, &event); err != nil {
			t.Fatalf("decode policy:changed %s: %v", m.Params, err)
		}
		if event.NodeID == "self-node" {
			events = append(events, event.policyChangedEvent)
		}
	}
	return events
}

// requireRevisionsNeverDescend fails when a policy:changed carries an older
// persisted revision than one clients already received.
func requireRevisionsNeverDescend(t *testing.T, events []policyChangedEvent) {
	t.Helper()
	for i := 1; i < len(events); i++ {
		cmp, err := nodepolicy.CompareRevisions(events[i].Policy.Revision, events[i-1].Policy.Revision)
		if err != nil {
			t.Fatalf("compare announced revisions %q and %q: %v", events[i].Policy.Revision, events[i-1].Policy.Revision, err)
		}
		if cmp < 0 {
			t.Fatalf("policy:changed %d announced revision %s after revision %s; announced revisions = %v", i, events[i].Policy.Revision, events[i-1].Policy.Revision, announcedRevisions(events))
		}
	}
}

func announcedRevisions(events []policyChangedEvent) []string {
	out := make([]string, len(events))
	for i, e := range events {
		out[i] = e.Policy.Revision + " " + e.Activation
	}
	return out
}

// requireLastAnnouncementIsStored fails unless the newest policy:changed is
// exactly what policy:get reports: the stored document and its activation.
func requireLastAnnouncementIsStored(t *testing.T, b *Broker, events []policyChangedEvent) {
	t.Helper()
	if len(events) == 0 {
		t.Fatal("no policy:changed was announced")
	}
	last, stored := events[len(events)-1], b.localPolicy()
	if !reflect.DeepEqual(last.Policy, stored.Policy) || last.policyActivation != stored.policyActivation {
		t.Fatalf("the last policy:changed is revision %s %+v, but the stored policy is revision %s %+v; announced revisions = %v",
			last.Policy.Revision, last.policyActivation, stored.Policy.Revision, stored.policyActivation, announcedRevisions(events))
	}
}

// waitUntil polls cond until it holds or the deadline passes.
func waitUntil(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for deadline := time.Now().Add(3 * time.Second); !cond(); {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// receive waits for one value from ch.
func receive[T any](t *testing.T, what string, ch <-chan T) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
	var zero T
	return zero
}

func TestPolicyAnnouncementsFollowTheStoredPolicyUnderConcurrentWriters(t *testing.T) {
	// Save A holds the push slot while the proxy sits on its answer. Save B,
	// a pause and save C each persist a newer revision and announce it before A
	// can, and a replay queues behind them all. Released, A and the pause
	// announce after newer revisions were already announced: they must not take
	// clients back to the document they saved.
	ab := newActivatedBroker(t)
	setActivationWait(ab.b, 4*time.Second)
	releaseA := make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseA) }) }
	t.Cleanup(release)
	ab.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetPolicy {
			var pushed struct {
				Revision string `json:"revision"`
			}
			if json.Unmarshal(params, &pushed) == nil && pushed.Revision == "5000" {
				<-releaseA
			}
		}
		return ab.acks.answer(method, params)
	})

	setA := make(chan policySetResult, 1)
	go func() { setA <- ab.setIdle(t, 5) }()
	ab.proxy.waitFor("save A's push of revision 5000", func(calls []policyCall) bool {
		for _, c := range calls {
			var pushed struct {
				Revision string `json:"revision"`
			}
			if c.method == nodepolicy.MethodSetPolicy && json.Unmarshal(c.params, &pushed) == nil && pushed.Revision == "5000" {
				return true
			}
		}
		return false
	})

	// The later writers cannot get the slot within their bound, so each
	// announces its revision as pending while A still holds it.
	setActivationWait(ab.b, 50*time.Millisecond)
	if b := ab.setIdle(t, 9); b.Policy.Revision != "5001" || b.Activation != activationPending {
		t.Fatalf("save B = revision %s %+v, want revision 5001 pending", b.Policy.Revision, b.policyActivation)
	}
	paused := make(chan error, 1)
	go func() {
		_, err := ab.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally)
		paused <- err
	}()
	waitUntil(t, "the pause to persist revision 5002", func() bool {
		return ab.b.localPolicy().Policy.Revision == "5002"
	})
	c := ab.setIdle(t, 11)
	if c.Policy.Revision != "5003" || c.Activation != activationPending {
		t.Fatalf("save C = revision %s %+v, want revision 5003 pending", c.Policy.Revision, c.policyActivation)
	}
	replayed := make(chan struct{})
	go func() {
		ab.b.replayPolicyToProxy(context.Background(), ab.pp)
		close(replayed)
	}()

	release()
	if a := receive(t, "save A's result", setA); a.Policy.Revision != "5000" {
		t.Fatalf("save A's result describes revision %s, want its own 5000", a.Policy.Revision)
	}
	if err := receive(t, "the pause", paused); err != nil {
		t.Fatalf("pause: %v", err)
	}
	receive(t, "the replay", replayed)
	waitPushesSettled(t, ab.b)
	stored := ab.waitActivation(t, activationActive)
	if stored.Policy.Revision != "5003" || stored.ActiveRevision != "5003" || stored.Policy.Availability != nodepolicy.Paused {
		t.Fatalf("after every writer finished policy:get = revision %s %s %+v, want revision 5003 paused and active", stored.Policy.Revision, stored.Policy.Availability, stored.policyActivation)
	}

	events := selfPolicyChanges(t, ab.out.frames())
	requireRevisionsNeverDescend(t, events)
	requireLastAnnouncementIsStored(t, ab.b, events)
	// Revision 5003 was announced pending, then active: an activation change of
	// the same revision still reaches clients.
	sawPending := false
	for _, e := range events {
		if e.Policy.Revision == "5003" && e.Activation == activationPending {
			sawPending = true
		}
	}
	if !sawPending {
		t.Fatalf("revision 5003 was never announced pending before it was active; announced revisions = %v", announcedRevisions(events))
	}
	requireLastAvailabilityIsLive(t, ab.b, ab.out.frames())
}

// requireLastAvailabilityIsLive fails unless the newest node:availability-changed
// this node announced for itself carries its live availability and active count.
func requireLastAvailabilityIsLive(t *testing.T, b *Broker, frames []Message) {
	t.Helper()
	type availabilityEvent struct {
		NodeID       string                  `json:"nodeId"`
		Availability nodepolicy.Availability `json:"availability"`
		Active       int                     `json:"active"`
	}
	var last *availabilityEvent
	for _, m := range frames {
		if m.Method != notifyAvailabilityChanged {
			continue
		}
		var event availabilityEvent
		if err := json.Unmarshal(m.Params, &event); err != nil {
			t.Fatalf("decode node:availability-changed %s: %v", m.Params, err)
		}
		if event.NodeID == "self-node" {
			last = &event
		}
	}
	if last == nil {
		t.Fatal("no node:availability-changed was announced")
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	live, active := r.live, r.admission.Active
	r.mu.Unlock()
	if last.Availability != live || last.Active != active {
		t.Fatalf("the last node:availability-changed = %s with %d active, want the live %s with %d active", last.Availability, last.Active, live, active)
	}
}

// stallingClient is a client that stops reading at the first frame stall
// selects, until resume is closed.
type stallingClient struct {
	syncBuffer
	stall   func([]byte) bool
	stalled atomic.Bool
	entered chan struct{}
	resume  chan struct{}
}

func (c *stallingClient) Write(p []byte) (int, error) {
	if c.stall(p) && c.stalled.CompareAndSwap(false, true) {
		close(c.entered)
		<-c.resume
	}
	return c.syncBuffer.Write(p)
}

func TestPolicyAnnouncementToASlowClientDoesNotHoldBackSaves(t *testing.T) {
	// While a client is not reading save A's announcement, save B still reaches
	// the disk, policy:get and the proxy; only its announcement waits, and then
	// follows A's.
	client := &stallingClient{entered: make(chan struct{}), resume: make(chan struct{})}
	client.stall = func(frame []byte) bool {
		var m Message
		if json.Unmarshal(frame, &m) != nil || m.Method != notifyPolicyChanged {
			return false
		}
		var event policyChangedEvent
		return json.Unmarshal(m.Params, &event) == nil && event.Policy.Revision == "5000"
	}
	var resumeOnce sync.Once
	resume := func() { resumeOnce.Do(func() { close(client.resume) }) }
	t.Cleanup(resume)
	b := &Broker{codec: NewCodec(client), nodeID: "self-node", clusterDir: filepath.Join(t.TempDir(), "cluster")}
	path := filepath.Join(t.TempDir(), nodepolicy.FileName)
	b.loadNodePolicy(path)
	setClock(b, 5000)
	pp, _, acks := attachAckingProxy(t, b)
	b.replayPolicyToProxy(context.Background(), pp)
	ab := &activatedBroker{b: b, acks: acks}

	setA := make(chan policySetResult, 1)
	go func() { setA <- ab.setIdle(t, 5) }()
	receive(t, "save A's announcement to reach the stalled client", client.entered)

	setB := make(chan error, 1)
	go func() {
		_, err := b.setLocalPolicy(mustJSON(map[string]any{"version": nodepolicy.SchemaVersion, "idle": map[string]any{"unloadAfterMinutes": 9}}), mustJSON("5000"))
		setB <- err
	}()
	// policy:get keeps answering: nothing holds the policy lock across the
	// client write.
	waitUntil(t, "save B to persist revision 5001 while the client is stalled", func() bool {
		got := make(chan string, 1)
		go func() { got <- b.localPolicy().Policy.Revision }()
		select {
		case revision := <-got:
			return revision == "5001"
		case <-time.After(time.Second):
			t.Fatal("policy:get is blocked while a client is stalled")
			return false
		}
	})
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read the policy file: %v", err)
	}
	onDisk, err := nodepolicy.Parse(data)
	if err != nil {
		t.Fatalf("parse the policy file: %v", err)
	}
	if onDisk.Revision != "5001" {
		t.Fatalf("the policy file holds revision %s while the client is stalled, want 5001", onDisk.Revision)
	}
	waitUntil(t, "the proxy to acknowledge revision 5001 while the client is stalled", func() bool {
		return acks.heldRevision() == "5001"
	})

	resume()
	if a := receive(t, "save A's result", setA); a.Policy.Revision != "5000" {
		t.Fatalf("save A's result describes revision %s, want its own 5000", a.Policy.Revision)
	}
	if err := receive(t, "save B's result", setB); err != nil {
		t.Fatalf("save B: %v", err)
	}
	events := selfPolicyChanges(t, client.frames())
	requireRevisionsNeverDescend(t, events)
	requireLastAnnouncementIsStored(t, b, events)
}
