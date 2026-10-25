// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Each Read exposes exactly one frame, making the reduction boundary independent
// of transport buffering, scheduling and wall-clock delays.
type decreaseDrainBody struct{ frames [][]byte }

func (b *decreaseDrainBody) Read(p []byte) (int, error) {
	if len(b.frames) == 0 {
		return 0, io.EOF
	}
	n := copy(p, b.frames[0])
	b.frames[0] = b.frames[0][n:]
	if len(b.frames[0]) == 0 {
		b.frames = b.frames[1:]
	}
	return n, nil
}
func (*decreaseDrainBody) Close() error { return nil }

func decreaseDrainOwner(t *testing.T, root *bufferBudget) (context.Context, *ownedInferenceBuffers) {
	t.Helper()
	p := &Proxy{buffers: root}
	r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil))
	lease, err := owner.scope.acquire(r.Context(), 96<<10)
	if err != nil {
		t.Fatal(err)
	}
	owner.retainParser(lease)
	t.Cleanup(owner.close)
	return r.Context(), owner
}

func TestBufferDecreaseAdmittedStreamDrains(t *testing.T) {
	for _, fixture := range []struct {
		name     string
		protocol streamProtocol
		frames   []string
	}{
		{"SSE numeric", streamProtocolSSE, []string{"data: {\"choices\":[],\"usage\":{\"completion_tokens\":2}}\n\n", "data: {\"choices\":[],\"usage\":{\"completion_tokens\":2}}\n\n", "data: [DONE]\n\n"}},
		{"multiline SSE", streamProtocolSSE, []string{"data: {\"choices\":[],\n data-ignored: x\ndata: \"usage\":{\"completion_tokens\":2}}\n\n", "data: {\"choices\":[],\ndata: \"usage\":{\"completion_tokens\":2}}\n\n", "data: [DONE]\n\n"}},
		{"NDJSON numeric", streamProtocolNDJSON, []string{"{\"done\":false,\"eval_count\":3}\n", "{\"done\":false,\"eval_count\":3}\n", "{\"done\":true,\"eval_count\":3}\n"}},
	} {
		for _, boundary := range []string{"after first frame", "before first read", "after admission before reader"} {
			t.Run(fixture.name+"/"+boundary, func(t *testing.T) {
				root := bufferTestBudget(t, 192<<20, 0)
				ctx, owner := decreaseDrainOwner(t, root)
				prepared, release, err := preparePerformanceBuffers(ctx, []byte(`{"model":"m","stream":true}`), originStreamRequest{protocol: fixture.protocol, declared: true, choices: 1}, performancePrepareLocalNumeric)
				if err != nil {
					t.Fatal(err)
				}
				defer release()
				preparationBytes, err := performancePreparationBytes([]byte(`{"model":"m","stream":true}`))
				if err != nil {
					t.Fatal(err)
				}
				responseCharges := []int64{streamFrameMetadataBytes, 1, streamFrameReadChunk, streamFrameNumericBytes, streamFrameReadChunk}
				if fixture.protocol == streamProtocolSSE {
					responseCharges = append(responseCharges, streamFrameReadChunk)
				}
				var responseBytes int64
				for _, charge := range responseCharges {
					responseBytes += charge
				}
				// Request parser, fingerprint scratch and preparation storage remain
				// owned alongside the response grants during handover.
				wantLiveBytes := (96 << 10) + streamFrameNumericBytes + preparationBytes + streamFrameAllowance(fixture.protocol, 1, streamFrameNumericBytes)
				wantLiveLeases := 4
				bufferAssertTotals(t, root, wantLiveBytes, wantLiveLeases, 0)
				body := &decreaseDrainBody{}
				for _, frame := range fixture.frames {
					body.frames = append(body.frames, []byte(frame))
				}
				reduce := func() {
					t.Helper()
					if err := root.reconfigure(1, 0); err != nil {
						t.Fatal(err)
					}
					other, err := root.newRequest(192 << 20)
					if err != nil {
						t.Fatal(err)
					}
					if lease, err := other.tryAcquire(ctx, 1); err == nil {
						lease.close()
						t.Fatal("new admission succeeded above aggregate cap")
					}
				}
				if boundary == "after admission before reader" {
					reduce()
				}
				calls := 0
				qualification := performanceExtractQualification{EngineFamily: "ollama", APIPath: "/v1/chat/completions", Streaming: true}
				if fixture.protocol == streamProtocolNDJSON {
					qualification.APIPath = "/api/generate"
				}
				reader, err := newStreamFrameReader(streamFrameConfig{Protocol: fixture.protocol, Upstream: body, Parent: prepared, Extract: func(payload []byte) performanceExtractMetadata {
					calls++
					return extractPerformancePayload(payload, qualification)
				}})
				if err != nil {
					t.Fatal(err)
				}
				defer reader.Close()
				bufferAssertTotals(t, root, wantLiveBytes, wantLiveLeases, 0)
				wantPendingBytes, wantPendingLeases := int64(0), 0
				ownedBytes := reader.metadataBytes + reader.choiceBytes + reader.scratchBytes + reader.parserBytes + reader.reserved + int64(cap(reader.payload))
				if want := responseBytes - wantPendingBytes; ownedBytes != want {
					t.Fatalf("reader-owned charge = %d, want %d", ownedBytes, want)
				}
				var pendingBytes int64
				pendingLeases := 0
				reader.budget.mu.Lock()
				for _, record := range reader.budget.leases {
					if record.pending {
						pendingBytes += record.bytes
						pendingLeases++
					}
				}
				reader.budget.mu.Unlock()
				if pendingBytes != wantPendingBytes || pendingLeases != wantPendingLeases {
					t.Fatalf("pending response grants = (%d bytes, %d leases), want (%d, %d)", pendingBytes, pendingLeases, wantPendingBytes, wantPendingLeases)
				}
				var got []byte
				if boundary == "after first frame" {
					first := make([]byte, 4096)
					n, err := reader.Read(first)
					if err != nil {
						t.Fatal(err)
					}
					got = append(got, first[:n]...)
					if string(got) != fixture.frames[0] {
						t.Fatalf("first frame = %q", got)
					}
				}
				if boundary != "after admission before reader" {
					reduce()
				}
				tail, err := io.ReadAll(reader)
				if err != nil {
					t.Fatal(err)
				}
				got = append(got, tail...)
				if want := strings.Join(fixture.frames, ""); string(got) != want {
					t.Fatalf("stream = %q, want %q", got, want)
				}
				if result := reader.Result(); result.Outcome != streamOutcomeCompleted || result.Reason != "" {
					t.Fatalf("result = %+v", result)
				}
				wantCalls := len(fixture.frames)
				if fixture.protocol == streamProtocolSSE {
					wantCalls--
				}
				if calls != wantCalls {
					t.Fatalf("numeric observations = %d, want %d", calls, wantCalls)
				}
				wantTokens := int64(2)
				if fixture.protocol == streamProtocolNDJSON {
					wantTokens = 3
				}
				if tokens := reader.Result().Numeric.OutputTokens; tokens == nil || *tokens != wantTokens {
					t.Fatalf("numeric output tokens = %v, want %d", tokens, wantTokens)
				}
				if err := reader.Close(); err != nil {
					t.Fatal(err)
				}
				release()
				owner.close()
				bufferAssertTotals(t, root, 0, 0, 0)
			})
		}
	}
}

func TestBufferDecreaseStreamGrowthDrains(t *testing.T) {
	root := bufferTestBudget(t, 192<<20, 0)
	ctx, owner := decreaseDrainOwner(t, root)
	prepared, release, err := preparePerformanceBuffers(ctx, []byte(`{"stream":true}`), originStreamRequest{protocol: streamProtocolSSE, declared: true, choices: 1}, performancePrepareLocalNumeric)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	first := "data: {\"choices\":[]}\n\n"
	later := "data: {\"padding\":\"" + strings.Repeat("x", int(streamFrameMaxBytes)-64) + "\"}\n\n"
	body := &decreaseDrainBody{frames: [][]byte{[]byte(first), []byte(later), []byte("data: [DONE]\n\n")}}
	r, err := newStreamFrameReader(streamFrameConfig{Protocol: streamProtocolSSE, Upstream: body, Parent: prepared})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	buf := make([]byte, 4096)
	if n, err := r.Read(buf); err != nil || string(buf[:n]) != first {
		t.Fatalf("first read: %q, %v", buf[:n], err)
	}
	if err := root.reconfigure(1, 0); err != nil {
		t.Fatal(err)
	}
	total := root.used
	// Claimed storage survives both preparation and request-owner release.
	release()
	owner.close()
	tail, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	if string(tail) != later+"data: [DONE]\n\n" {
		t.Fatalf("growth output differs: got %d bytes", len(tail))
	}
	if result := r.Result(); result.Outcome != streamOutcomeCompleted || result.Reason != "" {
		t.Fatalf("result = %+v", result)
	}
	if root.used > total {
		t.Fatal("draining stream increased the aggregate charge")
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	release()
	owner.close()
	bufferAssertTotals(t, root, 0, 0, 0)
}

func TestBufferDecreaseFourStreamsDrain(t *testing.T) {
	root := bufferTestBudget(t, 256<<20, 0)
	t.Cleanup(func() { bufferAssertTotals(t, root, 0, 0, 0) })
	var readers []*streamFrameReader
	for i := 0; i < 4; i++ {
		ctx, owner := decreaseDrainOwner(t, root)
		prepared, release, err := preparePerformanceBuffers(ctx, []byte(`{"stream":true}`), originStreamRequest{protocol: streamProtocolSSE, declared: true, choices: 1}, performancePrepareLocalNumeric)
		if err != nil {
			t.Fatalf("stream %d admission: %v", i, err)
		}
		t.Cleanup(release)
		// Include each response envelope in a 50 MiB admitted request footprint.
		padding, err := owner.scope.acquire(ctx, (50<<20)-owner.scope.used)
		if err != nil {
			t.Fatal(err)
		}
		owner.retain(padding)
		first := "data: {\"choices\":[]}\n\n"
		later := fmt.Sprintf("data: {\"padding\":\"%s\"}\n\n", strings.Repeat("x", 32<<10))
		reader, err := newStreamFrameReader(streamFrameConfig{Protocol: streamProtocolSSE, Parent: prepared, Upstream: &decreaseDrainBody{frames: [][]byte{[]byte(first), []byte(later), []byte("data: [DONE]\n\n")}}})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := reader.Close(); err != nil {
				t.Error(err)
			}
		})
		buf := make([]byte, len(first))
		if _, err := io.ReadFull(reader, buf); err != nil || string(buf) != first {
			t.Fatalf("first frame: %q, %v", buf, err)
		}
		readers = append(readers, reader)
	}
	if root.used != 200<<20 {
		t.Fatalf("admitted bytes=%d, want 200 MiB", root.used)
	}
	if err := root.reconfigure(192<<20, 0); err != nil {
		t.Fatal(err)
	}
	if lease, err := root.acquire(context.Background(), 1); err == nil {
		lease.close()
		t.Fatal("new admission succeeded above the reduced capacity")
	} else {
		bufferAssertBusy(t, err)
	}
	for i, reader := range readers {
		t.Run(fmt.Sprintf("stream %d", i+1), func(t *testing.T) {
			out, err := io.ReadAll(reader)
			if err != nil {
				t.Fatal(err)
			}
			want := fmt.Sprintf("data: {\"padding\":\"%s\"}\n\ndata: [DONE]\n\n", strings.Repeat("x", 32<<10))
			if string(out) != want || reader.Result().Outcome != streamOutcomeCompleted {
				t.Fatalf("stream did not complete: bytes=%d result=%+v", len(out), reader.Result())
			}
		})
	}
}

func TestBufferDecreasePreparationSharesStreamCategory(t *testing.T) {
	root := bufferTestBudget(t, 192<<20, 0)
	ctx, _ := decreaseDrainOwner(t, root)
	frames := ownedFrameBudget(ctx)
	if frames == nil || !frames.reserve(streamFrameBudgetBytes) {
		t.Fatal("could not fill shared stream category")
	}
	defer frames.release(streamFrameBudgetBytes)
	other, _ := decreaseDrainOwner(t, root)
	_, release, err := preparePerformanceBuffers(other, []byte(`{"stream":true}`), originStreamRequest{protocol: streamProtocolSSE, declared: true, choices: 1}, performancePrepareLocalNumeric)
	if err == nil {
		release()
		t.Fatal("response preparation bypassed saturated shared stream category")
	}
}

func TestBufferDecreaseCancelledReaderReleasesCharges(t *testing.T) {
	root := bufferTestBudget(t, 192<<20, 0)
	ctx, owner := decreaseDrainOwner(t, root)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	prepared, release, err := preparePerformanceBuffers(ctx, []byte(`{"stream":true}`), originStreamRequest{protocol: streamProtocolSSE, declared: true, choices: 1}, performancePrepareLocalNumeric)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	r, err := newStreamFrameReader(streamFrameConfig{Protocol: streamProtocolSSE, Parent: prepared, Upstream: &decreaseDrainBody{frames: [][]byte{[]byte("data: [DONE]\n\n")}}})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if err := root.reconfigure(1, 0); err != nil {
		t.Fatal(err)
	}
	// The reader owns its claimed response grants independently of the
	// dispatch preparation and request parser owners.
	release()
	owner.close()
	cancel()
	if out, err := io.ReadAll(r); err != nil || len(out) != 0 {
		t.Fatalf("cancelled output = %q, error = %v", out, err)
	}
	if result := r.Result(); result.Outcome != streamOutcomeCancelled {
		t.Fatalf("result = %+v", result)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	bufferAssertTotals(t, root, 0, 0, 0)
}
