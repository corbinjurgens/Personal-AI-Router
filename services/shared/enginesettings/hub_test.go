// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package enginesettings

import (
	"testing"

	"nvpair-shared/engines"
)

func TestFullBaselineCoalescingAndLimits(t *testing.T) {
	h := &Hub{}
	h.Publish([]Snapshot{{Revision: 1}})
	ch, closeSub, ok := h.Subscribe()
	if !ok || (<-ch)[0].Revision != 1 {
		t.Fatal("baseline missing")
	}
	for i := uint64(2); i < 1000; i++ {
		h.Publish([]Snapshot{{Revision: i}})
	}
	if (<-ch)[0].Revision != 999 {
		t.Fatal("slow subscriber missed latest full state")
	}
	closeSub()
	var closeAll []func()
	for i := 0; i < 64; i++ {
		_, close, ok := h.Subscribe()
		if !ok {
			t.Fatal("early subscriber cap")
		}
		closeAll = append(closeAll, close)
	}
	if _, _, ok := h.Subscribe(); ok {
		t.Fatal("unbounded subscribers")
	}
	for _, close := range closeAll {
		close()
	}
	if _, close, ok := h.Subscribe(); !ok {
		t.Fatal("cleanup leaked capacity")
	} else {
		close()
	}
}

func TestHubReadersOwnNestedServing(t *testing.T) {
	ram := int64(1024)
	published := []Snapshot{{Revision: 1,
		Settings: Config{Serving: &engines.ServingConfig{Version: 1, Models: []engines.ServingModel{{ModelID: "exact", Memory: engines.ServingMemory{HostRAMBytes: &ram}}}}},
		Serving:  &engines.ServingReadback{Observed: []engines.ServingObservation{{Slots: 1}}}}}
	h := &Hub{}
	first, closeFirst, _ := h.Subscribe()
	defer closeFirst()
	<-first
	h.Publish(published)
	*published[0].Settings.Serving.Models[0].Memory.HostRAMBytes = 2
	received := <-first
	received[0].Settings.Serving.Models[0].ModelID = "changed"
	received[0].Serving.Observed[0].Slots = 2
	later, closeLater, _ := h.Subscribe()
	defer closeLater()
	baseline := (<-later)[0]
	if baseline.Settings.Serving.Models[0].ModelID != "exact" || *baseline.Settings.Serving.Models[0].Memory.HostRAMBytes != 1024 || baseline.Serving.Observed[0].Slots != 1 {
		t.Fatal("publisher or reader mutation reached a later baseline")
	}
}
