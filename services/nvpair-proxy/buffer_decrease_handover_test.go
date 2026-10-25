// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"io"
	"testing"
	"testing/synctest"
	"time"
)

func TestBufferDecreaseNumericPreparationWithoutObservationDrains(t *testing.T) {
	root := bufferTestBudget(t, 192<<20, 0)
	ctx, owner := decreaseDrainOwner(t, root)
	prepared, release, err := preparePerformanceBuffers(ctx, []byte(`{"stream":true}`), originStreamRequest{protocol: streamProtocolSSE, declared: true, choices: 1}, performancePrepareLocalNumeric)
	if err != nil {
		t.Fatalf("prepare numeric response: %v", err)
	}
	defer release()
	frames := ownedFrameBudget(prepared)
	if frames == nil {
		t.Fatal("numeric preparation has no frame budget")
	}
	grants := make(map[*bufferLease]int64)
	var preparedBytes int64
	frames.mu.Lock()
	for _, record := range frames.leases {
		if record.lease != nil && record.pending {
			grants[record.lease] = record.bytes
			preparedBytes += record.bytes
		}
	}
	frames.mu.Unlock()
	wantFrameBytes := int64(streamFrameMetadataBytes + 1 + streamFrameNumericBytes + 3*streamFrameReadChunk)
	if len(grants) != 6 || preparedBytes != wantFrameBytes {
		t.Fatalf("pending grants = (%d bytes, %d leases), want (%d, 6)", preparedBytes, len(grants), wantFrameBytes)
	}
	root.mu.Lock()
	used, live := root.used, root.live
	root.mu.Unlock()
	if err := root.reconfigure(1, 0); err != nil {
		t.Fatalf("decrease below live usage: %v", err)
	}
	const first = "data: {\"choices\":[]}\n\n"
	const last = "data: [DONE]\n\n"
	// No performance attempt or Extract callback is installed. The admitted
	// preparation alone determines the reader's parser class.
	r, err := newStreamFrameReader(streamFrameConfig{
		Protocol: streamProtocolSSE, Parent: prepared,
		Upstream: &decreaseDrainBody{frames: [][]byte{[]byte(first), []byte(last)}},
	})
	if err != nil {
		t.Fatalf("construct prepared reader: %v", err)
	}
	defer r.Close()
	if r.parserBytes != streamFrameNumericBytes {
		t.Fatalf("parser before reading = %d, want %d", r.parserBytes, streamFrameNumericBytes)
	}
	if r.budget != frames || r.extract != nil || r.Result().Outcome != streamOutcomeStreaming {
		t.Fatal("reader did not adopt unobserved numeric preparation as a live stream")
	}
	claimed := 0
	frames.mu.Lock()
	for _, record := range frames.leases {
		if record.lease == nil {
			continue
		}
		if n, ok := grants[record.lease]; !ok || n != record.bytes || record.pending {
			frames.mu.Unlock()
			t.Fatal("reader did not claim the original pending grant")
		}
		claimed++
	}
	frames.mu.Unlock()
	if claimed != len(grants) {
		t.Fatalf("claimed grants = %d, want %d", claimed, len(grants))
	}
	bufferAssertTotals(t, root, used, live, 0)
	// Releasing dispatch preparation must leave every claimed reader grant live.
	release()
	if got := frames.inUse(); got != wantFrameBytes {
		t.Fatalf("reader grants after preparation release = %d, want %d", got, wantFrameBytes)
	}
	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("drain after decrease: %v", err)
	}
	if string(got) != first+last {
		t.Fatalf("drained stream = %q, want %q", got, first+last)
	}
	if result := r.Result(); result.Outcome != streamOutcomeCompleted || result.Reason != "" {
		t.Fatalf("drained result = %+v", result)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("close reader: %v", err)
	}
	owner.close()
	if got := frames.inUse(); got != 0 {
		t.Fatalf("frame charges after cleanup = %d, want 0", got)
	}
	bufferAssertTotals(t, root, 0, 0, 0)
}

func TestBufferDecreaseGrowthBorrowsWithoutWaiting(t *testing.T) {
	for _, storage := range []string{"raw frame", "SSE payload"} {
		t.Run(storage, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				const waitSeconds = 5
				root := bufferTestBudget(t, 192<<20, waitSeconds*time.Second)
				ctx, owner := decreaseDrainOwner(t, root)
				prepared, release, err := preparePerformanceBuffers(ctx, []byte(`{"stream":true}`), originStreamRequest{protocol: streamProtocolSSE, declared: true, choices: 1}, performancePrepareLocalNumeric)
				if err != nil {
					t.Fatalf("prepare response: %v", err)
				}
				defer release()
				// A different account makes aggregate pressure waitable: the reader's
				// own live bytes plus a replacement still fit the reduced capacity.
				hold := bufferMustAcquire(t, root, 64<<10)
				const first = "data: {\"choices\":[],\ndata: \"usage\":{}}\n\n"
				const last = "data: [DONE]\n\n"
				r, err := newStreamFrameReader(streamFrameConfig{
					Protocol: streamProtocolSSE, Parent: prepared, IdleTimeout: time.Hour,
					Upstream: &decreaseDrainBody{frames: [][]byte{[]byte(first), []byte(last)}},
				})
				if err != nil {
					t.Fatalf("construct reader: %v", err)
				}
				defer r.Close()
				prefix := make([]byte, 1)
				if n, err := r.Read(prefix); err != nil || n != 1 || prefix[0] != first[0] {
					t.Fatalf("read first byte = (%q, %d, %v)", prefix, n, err)
				}
				root.mu.Lock()
				used, live := root.used, root.live
				root.mu.Unlock()
				capacity := used - 1
				if owner.scope.used+2*streamFrameReadChunk > capacity {
					t.Fatal("fixture would refuse waiting acquisition as an impossible own footprint")
				}
				if err := root.reconfigure(capacity, waitSeconds*time.Second); err != nil {
					t.Fatalf("decrease below live usage: %v", err)
				}
				raw := bytes.Clone(r.buf[:cap(r.buf)])
				payload := bytes.Clone(r.payload[:cap(r.payload)])
				rawStart, payloadStart := &r.buf[:cap(r.buf)][0], &r.payload[:cap(r.payload)][0]
				rawLen, payloadLen := len(r.buf), len(r.payload)
				started := time.Now()
				r.mu.Lock()
				var grew bool
				if storage == "raw frame" {
					grew = r.growLocked(cap(r.buf) + 1)
				} else {
					grew = r.growPayloadLocked(cap(r.payload) + 1)
				}
				r.mu.Unlock()
				if elapsed := time.Since(started); !grew || elapsed != 0 {
					t.Fatalf("growth = %v after %v virtual time, want immediate borrowed growth", grew, elapsed)
				}
				if len(r.buf) != rawLen || len(r.payload) != payloadLen || !bytes.Equal(r.buf, raw[:rawLen]) || !bytes.Equal(r.payload, payload[:payloadLen]) {
					t.Fatal("borrowed growth changed original raw or payload contents")
				}
				if storage == "raw frame" && (&r.buf[:cap(r.buf)][0] == rawStart || cap(r.buf) <= len(raw)) {
					t.Fatal("raw frame did not acquire replacement storage")
				}
				if storage == "SSE payload" && (&r.payload[:cap(r.payload)][0] == payloadStart || cap(r.payload) <= len(payload)) {
					t.Fatal("SSE payload did not acquire replacement storage")
				}
				bufferAssertTotals(t, root, used, live, 0)
				tail, err := io.ReadAll(r)
				if err != nil {
					t.Fatalf("drain unchanged storage: %v", err)
				}
				if string(prefix)+string(tail) != first+last {
					t.Fatalf("stream after refusal = %q", string(prefix)+string(tail))
				}
				if result := r.Result(); result.Outcome != streamOutcomeCompleted || result.Reason != "" {
					t.Fatalf("result after refused growth = %+v", result)
				}
				if err := r.Close(); err != nil {
					t.Fatalf("close reader: %v", err)
				}
				release()
				owner.close()
				hold.close()
				if got := r.budget.inUse(); got != 0 {
					t.Fatalf("frame charges after cleanup = %d, want 0", got)
				}
				bufferAssertTotals(t, root, 0, 0, 0)
			})
		})
	}
}
