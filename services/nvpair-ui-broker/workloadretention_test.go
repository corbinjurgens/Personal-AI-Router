// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"testing"
	"time"

	"nvpair-ui-broker/workloadstore"
)

// storeTerminal builds a completed workload finished completedAgo before now.
func storeTerminal(t *testing.T, id, origin string, completedAgo time.Duration) workloadstore.Incoming {
	t.Helper()
	completedAt := time.Now().Add(-completedAgo).UnixMilli()
	raw, err := json.Marshal(map[string]any{
		"id": id, "originatedFrom": origin, "engine": "ollama", "runId": "r1",
		"state": "completed", "scheduledOn": origin,
		"createdAt": completedAt - 1000, "completedAt": completedAt, "model": "m",
	})
	if err != nil {
		t.Fatal(err)
	}
	in, ok := workloadstore.ParseIncoming(raw)
	if !ok {
		t.Fatal("ParseIncoming rejected the fixture")
	}
	return in
}

// TestRetireWorkloadHistoryAnnouncesRemovals: the store's age cap used to drop
// history silently, so a subscribed client kept every finished job until the
// app restarted. Each retired record must now reach the client as a
// workloads:remove, and a record within the caps must not.
func TestRetireWorkloadHistoryAnnouncesRemovals(t *testing.T) {
	var out bytes.Buffer
	b := &Broker{codec: NewCodec(&out), workloads: workloadstore.New(), workloadsSubscribed: true}
	b.workloads.Apply(storeTerminal(t, "old", "peer", workloadstore.DefaultHistoryMaxAge+time.Hour))
	b.workloads.Apply(storeTerminal(t, "recent", "peer", time.Minute))

	b.retireWorkloadHistory()

	var removed []workloadRemoveParams
	sc := bufio.NewScanner(&out)
	for sc.Scan() {
		var frame struct {
			Method string               `json:"method"`
			Params workloadRemoveParams `json:"params"`
		}
		if err := json.Unmarshal(sc.Bytes(), &frame); err != nil {
			t.Fatalf("bad frame %q: %v", sc.Text(), err)
		}
		if frame.Method == "workloads:remove" {
			removed = append(removed, frame.Params)
		}
	}
	want := workloadRemoveParams{WorkloadID: "old", OriginatedFrom: "peer"}
	if len(removed) != 1 || removed[0] != want {
		t.Fatalf("removals = %+v, want exactly %+v", removed, want)
	}
	if _, ok := b.workloads.Get("peer", "recent"); !ok {
		t.Fatal("a record within the caps must stay in the store")
	}

	// A second sweep with nothing left to retire is silent.
	out.Reset()
	b.retireWorkloadHistory()
	if out.Len() != 0 {
		t.Fatalf("second sweep emitted %q, want nothing", out.String())
	}
}

// TestRetireWorkloadHistoryWithoutSubscriberStillPrunes: the store stays bounded
// even when no client has opted in to workload events.
func TestRetireWorkloadHistoryWithoutSubscriberStillPrunes(t *testing.T) {
	var out bytes.Buffer
	b := &Broker{codec: NewCodec(&out), workloads: workloadstore.New()}
	b.workloads.Apply(storeTerminal(t, "old", "peer", workloadstore.DefaultHistoryMaxAge+time.Hour))

	b.retireWorkloadHistory()

	if _, ok := b.workloads.Get("peer", "old"); ok {
		t.Fatal("expired history must be pruned with no subscriber")
	}
	if out.Len() != 0 {
		t.Fatalf("emitted %q to an unsubscribed client", out.String())
	}
}
