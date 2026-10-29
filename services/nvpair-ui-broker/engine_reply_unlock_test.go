// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	settings "nvpair-shared/enginesettings"
)

type engineReplyBarrier struct {
	syncBuffer
	replied, publishing, release chan struct{}
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
	n, err := w.syncBuffer.Write(raw)
	if frame.ID != nil && string(*frame.ID) == "991" {
		close(w.replied)
	}
	return n, err
}

func TestEngineStartReplyReleasesOperationBeforeImmediateSwitch(t *testing.T) {
	h := newSwitchHarness(t)
	p := h.request(t, "immediate-after-start")
	w := &engineReplyBarrier{replied: make(chan struct{}), publishing: make(chan struct{}), release: make(chan struct{})}
	open := sync.OnceFunc(func() { close(w.release) })
	t.Cleanup(open)
	h.b.codec = NewCodec(w)
	// A changed launch readback makes deferred reconciliation publish settings.
	// Publication drops engineConfigMu but retains settingsApplyMu, allowing a
	// real immediate switch to exercise admission while cleanup is held here.
	h.b.engineSettingsLoaded = true
	h.b.engineSettings = map[string]*engineSettingsRecord{
		"llamacpp": {Snapshot: settings.Snapshot{Engine: "llamacpp", Phase: "idle"}},
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		id := json.RawMessage(`991`)
		h.b.relayToEngineNow(&Message{ID: &id, Method: "engine:start", Params: json.RawMessage(`{"engine":"llamacpp"}`)})
	}()
	defer func() { open(); requireGateOpens(t, done, "relay cleanup") }()
	requireGateOpens(t, w.publishing, "settings reconciliation publication")
	select {
	case <-w.replied:
		if h.b.settingsApplyMu.TryLock() {
			h.b.settingsApplyMu.Unlock()
			t.Fatal("reply preceded held reconciliation without operation ownership")
		}
		_, err := h.b.acceptEngineSwitch(context.Background(), p, "")
		t.Fatalf("actual engine:start reply preceded lock release; immediate switch error = %v (kind %s)", err, switchErrorDetail(err, "").Kind)
	default:
	}
	open()
	requireGateOpens(t, w.replied, "engine:start reply after reconciliation")
	frames := w.frames()
	var reply *Message
	for _, frame := range frames {
		if frame.ID != nil && string(*frame.ID) == "991" {
			reply = &frame
		}
	}
	if reply == nil || reply.Error != nil {
		t.Fatalf("engine:start reply = %+v", reply)
	}
	// Reconciliation changed the configuration after p was captured. Admission
	// must reach that revision check and return its durable rejection, not busy.
	result, err := h.b.acceptEngineSwitch(context.Background(), p, "")
	if switchErrorDetail(err, "").Kind != settings.SwitchErrorConfigurationConflict || result.Receipt.Phase != settings.SwitchRejected {
		t.Fatalf("immediate switch after engine:start reply = %+v, %v", result, err)
	}
	h.waitIdle(t)
}
