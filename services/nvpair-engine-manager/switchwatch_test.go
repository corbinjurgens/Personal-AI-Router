// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	settings "nvpair-shared/enginesettings"
)

func watchReceipt() settings.SwitchReceipt {
	now := time.Now().UTC()
	request := settings.SwitchRequest{NodeID: "target", RequestID: "request", SourceEngine: "ollama", TargetEngine: "llamacpp", ExpectedRevision: "e3v1:" + strings.Repeat("a", 64)}
	return settings.SwitchReceipt{
		Version: settings.SwitchVersion, NodeID: request.NodeID, RequestID: request.RequestID, Caller: "caller", Request: request,
		PayloadHash: settings.SwitchPayloadHash("caller", request.NodeID, request), ExpectedRevision: request.ExpectedRevision, AcceptedRevision: request.ExpectedRevision,
		Sequence: 1, Phase: settings.SwitchAccepted, AcceptedAt: now, UpdatedAt: now, Deadline: now.Add(settings.SwitchOperationBudget),
		Source: settings.SwitchEngineState{Intent: settings.SwitchIntentUnknown}, Target: settings.SwitchEngineState{Intent: settings.SwitchIntentUnknown},
	}
}

func watchJSON(t *testing.T, receipt settings.SwitchReceipt) json.RawMessage {
	t.Helper()
	if err := receipt.Validate(); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(settings.SwitchResult{Receipt: receipt})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func watchIdentity() switchWatchIdentity {
	return switchWatchIdentity{peer: ecPeer{nodeID: "target", clusterUUID: "principal", port: 1234}, route: clustertrust.PeerServiceRoute{Incarnation: "pin-incarnation"}}
}

func waitWatchEmpty(t *testing.T, watchers *switchWatchers) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		watchers.mu.Lock()
		count := len(watchers.active)
		watchers.mu.Unlock()
		if count == 0 {
			return
		}
		select {
		case <-deadline:
			t.Fatal("watchers did not stop")
		case <-ticker.C:
		}
	}
}

func TestSwitchWatchNewSequencesAndTerminal(t *testing.T) {
	var watchers switchWatchers
	identity := watchIdentity()
	receipt := watchReceipt()
	responses := make(chan json.RawMessage, 4)
	// Unchanged long-poll timeout, progress, duplicate, then terminal.
	responses <- watchJSON(t, receipt)
	receipt.Sequence = 2
	receipt.Phase = settings.SwitchValidating
	responses <- watchJSON(t, receipt)
	responses <- watchJSON(t, receipt)
	receipt.Sequence = 3
	receipt.Phase, receipt.Terminal = settings.SwitchCompleted, true
	receipt.FinishedAt = &receipt.UpdatedAt
	responses <- watchJSON(t, receipt)
	var calls atomic.Int32
	after := make(chan uint64, 4)
	client := switchTestClient(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var params settings.SwitchStatusRequest
		if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
			t.Error(err)
			return
		}
		if r.URL.Path != settings.SwitchStatusPath || params.NodeID != "target" || params.RequestID != "request" || params.AfterSequence == nil {
			t.Errorf("unexpected status request: %+v", params)
			return
		}
		after <- *params.AfterSequence
		select {
		case raw := <-responses:
			if _, err := w.Write(raw); err != nil {
				t.Error(err)
			}
		case <-r.Context().Done():
		}
	}))
	notifications := make(chan settings.SwitchChanged, 4)
	started := watchers.observe(context.Background(), identity, watchJSON(t, watchReceiptForAcceptance(receipt)),
		func(string) (switchWatchIdentity, bool) { return identity, true },
		func(context.Context, ecPeer) (*remoteClient, error) { return client, nil },
		func(method string, value any) error {
			changed, ok := value.(settings.SwitchChanged)
			if method != settings.SwitchChangedNotification || !ok {
				t.Errorf("unexpected notification: %s %+v", method, value)
				return errors.New("bad notification")
			}
			notifications <- changed
			return nil
		})
	if !started {
		t.Fatal("watcher did not start")
	}
	waitWatchEmpty(t, &watchers)
	if calls.Load() != 4 || len(notifications) != 2 {
		t.Fatalf("calls=%d notifications=%d", calls.Load(), len(notifications))
	}
	for _, sequence := range []uint64{2, 3} {
		if got := <-notifications; got != (settings.SwitchChanged{NodeID: "target", RequestID: "request", Sequence: sequence}) {
			t.Fatalf("notification=%+v", got)
		}
	}
	for _, want := range []uint64{1, 1, 2, 2} {
		if got := <-after; got != want {
			t.Fatalf("afterSequence=%d want=%d", got, want)
		}
	}
}

func watchReceiptForAcceptance(r settings.SwitchReceipt) settings.SwitchReceipt {
	r.Sequence, r.Phase, r.Terminal, r.FinishedAt = 1, settings.SwitchAccepted, false, nil
	return r
}

func TestSwitchWatchDedupeBoundAndShutdown(t *testing.T) {
	var watchers switchWatchers
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	identity := watchIdentity()
	lookup := func(string) (switchWatchIdentity, bool) { return identity, true }
	var calls atomic.Int32
	clientFor := func(ctx context.Context, _ ecPeer) (*remoteClient, error) {
		calls.Add(1)
		<-ctx.Done()
		return nil, ctx.Err()
	}
	notify := func(string, any) error { t.Error("unexpected notification"); return nil }
	r := watchReceipt()
	raw := watchJSON(t, r)
	if !watchers.observe(ctx, identity, raw, lookup, clientFor, notify) {
		t.Fatal("first watcher refused")
	}
	if watchers.observe(ctx, identity, raw, lookup, clientFor, notify) {
		t.Fatal("duplicate watcher started")
	}
	for i := 1; i < maxSwitchWatchers; i++ {
		r.RequestID += "x"
		r.Request.RequestID = r.RequestID
		r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
		if !watchers.observe(ctx, identity, watchJSON(t, r), lookup, clientFor, notify) {
			t.Fatalf("slot %d refused", i)
		}
	}
	r.RequestID += "x"
	r.Request.RequestID = r.RequestID
	r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
	if watchers.observe(ctx, identity, watchJSON(t, r), lookup, clientFor, notify) {
		t.Fatal("capacity exceeded")
	}
	cancel()
	waitWatchEmpty(t, &watchers)
	if calls.Load() > maxSwitchWatchers {
		t.Fatalf("too many remote calls: %d", calls.Load())
	}
}

func TestSwitchWatchIdentityChangeStops(t *testing.T) {
	for _, change := range []string{"route", "pin", "disconnect"} {
		t.Run(change, func(t *testing.T) {
			var watchers switchWatchers
			identity := watchIdentity()
			var changed atomic.Bool
			var calls atomic.Int32
			lookup := func(string) (switchWatchIdentity, bool) {
				current := identity
				if changed.Load() {
					if change == "disconnect" {
						return current, false
					}
					if change == "pin" {
						current.route.Incarnation = "replacement"
					} else {
						current.peer.port++
					}
				}
				return current, true
			}
			clientFor := func(context.Context, ecPeer) (*remoteClient, error) {
				calls.Add(1)
				changed.Store(true)
				return nil, errors.New("disconnected")
			}
			if !watchers.observe(context.Background(), identity, watchJSON(t, watchReceipt()), lookup, clientFor, func(string, any) error { t.Error("retired route emitted"); return nil }) {
				t.Fatal("not started")
			}
			waitWatchEmpty(t, &watchers)
			if calls.Load() != 1 {
				t.Fatalf("calls=%d", calls.Load())
			}
		})
	}
}

func TestSwitchWatchDisconnectBackoffStops(t *testing.T) {
	var watchers switchWatchers
	identity := watchIdentity()
	var calls atomic.Int32
	started := time.Now()
	if !watchers.observe(context.Background(), identity, watchJSON(t, watchReceipt()),
		func(string) (switchWatchIdentity, bool) { return identity, true },
		func(context.Context, ecPeer) (*remoteClient, error) {
			calls.Add(1)
			return nil, errors.New("disconnected")
		},
		func(string, any) error { t.Error("disconnect emitted notification"); return nil }) {
		t.Fatal("not started")
	}
	waitWatchEmpty(t, &watchers)
	if calls.Load() != 2 || time.Since(started) < switchWatchBackoff {
		t.Fatalf("calls=%d elapsed=%s", calls.Load(), time.Since(started))
	}
}

func TestSwitchWatchOperationDeadline(t *testing.T) {
	var watchers switchWatchers
	identity := watchIdentity()
	r := watchReceipt()
	r.AcceptedAt = time.Now().Add(-settings.SwitchOperationBudget + 40*time.Millisecond)
	r.UpdatedAt = r.AcceptedAt
	r.Deadline = r.AcceptedAt.Add(settings.SwitchOperationBudget)
	var cancelled atomic.Bool
	if !watchers.observe(context.Background(), identity, watchJSON(t, r),
		func(string) (switchWatchIdentity, bool) { return identity, true },
		func(ctx context.Context, _ ecPeer) (*remoteClient, error) {
			<-ctx.Done()
			cancelled.Store(true)
			return nil, ctx.Err()
		},
		func(string, any) error { t.Error("deadline emitted notification"); return nil }) {
		t.Fatal("not started")
	}
	waitWatchEmpty(t, &watchers)
	if !cancelled.Load() {
		t.Fatal("deadline did not cancel network wait")
	}
}

func TestSwitchWatchDisconnectLeavesOtherWatcherActive(t *testing.T) {
	var watchers switchWatchers
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	identity := watchIdentity()
	lookup := func(string) (switchWatchIdentity, bool) { return identity, true }
	notify := func(string, any) error { t.Error("unexpected notification"); return nil }
	r := watchReceipt()
	blocked := make(chan struct{})
	if !watchers.observe(ctx, identity, watchJSON(t, r), lookup,
		func(ctx context.Context, _ ecPeer) (*remoteClient, error) {
			close(blocked)
			<-ctx.Done()
			return nil, ctx.Err()
		}, notify) {
		t.Fatal("unrelated watcher not started")
	}
	select {
	case <-blocked:
	case <-time.After(time.Second):
		t.Fatal("unrelated watcher did not enter network wait")
	}
	r.RequestID, r.Request.RequestID = "disconnect", "disconnect"
	r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
	var calls atomic.Int32
	if !watchers.observe(ctx, identity, watchJSON(t, r), lookup,
		func(context.Context, ecPeer) (*remoteClient, error) {
			calls.Add(1)
			return nil, errors.New("disconnected")
		}, notify) {
		t.Fatal("disconnect watcher not started")
	}
	deadline := time.After(3 * time.Second)
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		watchers.mu.Lock()
		_, failed := watchers.active[switchWatchKey{"target", "disconnect"}]
		_, other := watchers.active[switchWatchKey{"target", "request"}]
		watchers.mu.Unlock()
		if !other {
			t.Fatal("disconnect retired unrelated watcher")
		}
		if !failed {
			break
		}
		select {
		case <-deadline:
			t.Fatal("disconnect watcher did not stop")
		case <-ticker.C:
		}
	}
	if calls.Load() != 2 || ctx.Err() != nil {
		t.Fatalf("calls=%d parent error=%v", calls.Load(), ctx.Err())
	}
	cancel()
	waitWatchEmpty(t, &watchers)
}
