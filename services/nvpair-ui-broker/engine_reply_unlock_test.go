// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"sync"
	"testing"

	settings "nvpair-shared/enginesettings"
)

type engineReplyBarrier struct {
	syncBuffer
	replied, publishing, release chan struct{}
	broker                       *Broker
	operationReleased            bool
}

func (w *engineReplyBarrier) Write(raw []byte) (int, error) {
	var frame Message
	if err := json.Unmarshal(raw, &frame); err != nil {
		return 0, err
	}
	if frame.Method == "engine:settings-changed" {
		close(w.publishing)
		<-w.release
	}
	if frame.ID != nil && string(*frame.ID) == "991" {
		w.operationReleased = w.broker.settingsApplyMu.TryLock()
		if w.operationReleased {
			w.broker.settingsApplyMu.Unlock()
		}
	}
	n, err := w.syncBuffer.Write(raw)
	if frame.ID != nil && string(*frame.ID) == "991" {
		close(w.replied)
	}
	return n, err
}

func TestEnginePortReplyFollowsSettingsOperationRelease(t *testing.T) {
	h := newModelSettingsHarness(t)
	w := &engineReplyBarrier{broker: h.b, replied: make(chan struct{}), publishing: make(chan struct{}), release: make(chan struct{})}
	open := sync.OnceFunc(func() { close(w.release) })
	t.Cleanup(open)
	h.b.codec = NewCodec(w)
	// A changed launch readback makes deferred reconciliation publish settings.
	// Publication drops engineConfigMu but retains settingsApplyMu. Hold that
	// publication to prove the response waits for reconciliation and release.
	h.b.engineSettingsLoaded = true
	h.b.engineSettings = map[string]*engineSettingsRecord{
		"llamacpp": {Snapshot: settings.Snapshot{Engine: "llamacpp", Phase: "idle"}},
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		id := json.RawMessage(`991`)
		h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:set-port", Params: json.RawMessage(`{"engine":"llamacpp","port":19181}`)})
	}()
	defer func() { open(); requireGateOpens(t, done, "relay cleanup") }()
	requireGateOpens(t, w.publishing, "settings reconciliation publication")
	if h.b.settingsApplyMu.TryLock() {
		h.b.settingsApplyMu.Unlock()
		t.Fatal("reconciliation did not retain operation ownership")
	}
	select {
	case <-w.replied:
		t.Fatal("engine:set-port reply preceded reconciliation completion")
	default:
	}
	open()
	requireGateOpens(t, w.replied, "engine:set-port reply after reconciliation")
	if !w.operationReleased {
		t.Fatal("engine:set-port reply preceded settings operation lock release")
	}
	frames := w.frames()
	var reply *Message
	for _, frame := range frames {
		if frame.ID != nil && string(*frame.ID) == "991" {
			reply = &frame
		}
	}
	if reply == nil || reply.Error != nil {
		t.Fatalf("engine:set-port reply = %+v", reply)
	}
}
