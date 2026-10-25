// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

// A failed attempt has no admitted stream to drain. Its replacement must drop
// preparation, resolve the current policy and acquire storage under today's cap.
func TestBufferDecreasePreOutputRetryNewAdmission(t *testing.T) {
	run := func(name string, insufficient bool) {
		t.Run(name, func(t *testing.T) {
			policy := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
				"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "small:1b"}},
			})
			fx := newTierFixture(t, ollamaCase(t), policy, map[string][]string{
				"a": {"small:1b"}, "b": {"large:7b"},
			}, nil)
			root := bufferTestBudget(t, 8<<20, 0)
			fx.p.buffers = root
			queued := make(chan struct{}, 1)
			// Existing allocator clock injection exposes FIFO entry, rather than
			// using elapsed time as evidence that rebuilding has started.
			root.newTimer = func(d time.Duration) *time.Timer {
				timer := time.NewTimer(d)
				select {
				case queued <- struct{}{}:
				default:
				}
				return timer
			}
			received, resume := make(chan struct{}), make(chan struct{})
			var once sync.Once
			release := func() { once.Do(func() { close(resume) }) }
			var firstCalls atomic.Int32
			first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				call := firstCalls.Add(1)
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("read first attempt: %v", err)
					return
				}
				if string(body) != `{"model":"small:1b","stream":true}` {
					t.Errorf("first attempt body = %q", body)
				}
				if call == 1 {
					close(received)
				}
				select {
				case <-resume:
				case <-r.Context().Done():
					return
				}
				// No headers or body have escaped before the activation barrier.
				w.WriteHeader(http.StatusServiceUnavailable)
			}))
			defer first.Close()
			defer release()
			fx.p.soleFacade().discovery.AddManual(nodeForModel(t, "a", first.URL, "small:1b"))
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			recorder, done := httptest.NewRecorder(), make(chan struct{})
			request := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"weak","stream":true}`)).WithContext(ctx)
			go func() {
				defer close(done)
				fx.p.soleFacade().handleHTTP(recorder, request)
			}()
			defer func() {
				cancel()
				release()
				select {
				case <-done:
				case <-time.After(3 * time.Second):
					t.Error("retry handler did not join during cleanup")
				}
			}()
			waitChan(t, received, "first native upstream before output")
			if firstCalls.Load() != 1 || fx.eng["a"].hits() != 0 || fx.eng["b"].hits() != 0 {
				t.Fatal("first dispatch did not exclusively reach the fenced upstream")
			}
			owners := fx.p.activeRoutingRequests()
			if len(owners) != 1 {
				t.Fatalf("routing owners = %d, want one", len(owners))
			}
			control := fx.p.routingRequestControl(owners[0])
			if control == nil {
				t.Fatal("first dispatch has no routing controller")
			}
			control.mu.Lock()
			committed := control.committed
			control.mu.Unlock()
			if committed {
				t.Fatal("fenced attempt already committed output")
			}
			root.mu.Lock()
			firstUsed := root.used
			var firstResponseBytes int64
			if root.streams != nil {
				firstResponseBytes = root.streams.used
			}
			root.mu.Unlock()
			// Manual peers forward NDJSON without local numeric extraction or
			// the decoded-payload scratch used only by SSE.
			wantResponseBytes := int64(streamFrameMetadataBytes + 1 + streamFrameSyntaxBytes + streamFrameReadChunk + 2*streamFrameMaxBytes)
			if firstResponseBytes != wantResponseBytes {
				t.Fatalf("first response preparation = %d bytes, want %d", firstResponseBytes, wantResponseBytes)
			}
			var blocker *bufferLease
			if insufficient {
				blocker = bufferMustAcquire(t, root, wantResponseBytes+(128<<10))
				defer blocker.close()
			}
			fx.p.admission.mu.Lock()
			generation := fx.p.admission.policyGeneration
			fx.p.admission.mu.Unlock()
			// Replacing the tier also proves rebuilding obeys current permission:
			// only b's model is eligible after this real activation transition.
			policy = fx.p.admission.currentPolicy()
			policy.Tiers = map[string][]nodepolicy.TierEntry{
				"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "large:7b"}},
			}
			token, err := fx.p.beginPublicActivation(policy)
			if err != nil {
				t.Fatalf("activate replacement policy: %v", err)
			}
			fx.p.completePublicActivation(token, nil)
			fx.p.admission.mu.Lock()
			advanced := fx.p.admission.policyGeneration
			fx.p.admission.mu.Unlock()
			if advanced != generation+1 {
				t.Fatalf("policy generation = %d, want %d", advanced, generation+1)
			}
			capacity, wait := int64(4<<20), time.Duration(0)
			if insufficient {
				capacity, wait = wantResponseBytes+(256<<10), 3*time.Second
			}
			if err := root.reconfigure(capacity, wait); err != nil {
				t.Fatal(err)
			}
			release()
			if insufficient {
				waitChan(t, queued, "replacement preparation FIFO entry")
				root.mu.Lock()
				used, pending := root.used, root.waiters.Len()
				responseBytes := root.streams.used
				root.mu.Unlock()
				if pending != 1 || responseBytes != 0 || used-wantResponseBytes-(128<<10) > firstUsed-firstResponseBytes {
					t.Fatalf("old preparation retained during rebuild: queued=%d response=%d used=%d first=%d preparation=%d", pending, responseBytes, used, firstUsed, firstResponseBytes)
				}
				fx.p.priorityMu.Lock()
				claims := len(fx.p.priorityReservations)
				fx.p.priorityMu.Unlock()
				publications := notificationsOf(t, fx.w, workloadSubmittedMethod)
				if len(publications) == 0 {
					t.Fatal("missing placement history")
				}
				var last workloadParams
				if err := json.Unmarshal(publications[len(publications)-1], &last); err != nil {
					t.Fatal(err)
				}
				if claims != 0 || last.WorkloadInfo.ScheduledOn != "" || fx.p.admission.currentActive() != 0 || firstCalls.Load() != 1 || fx.eng["b"].hits() != 0 {
					t.Fatalf("unprepared retry placed or dispatched: claims=%d placement=%q active=%d first=%d second=%d", claims, last.WorkloadInfo.ScheduledOn, fx.p.admission.currentActive(), firstCalls.Load(), fx.eng["b"].hits())
				}
				// Deterministically make the pending request individually impossible.
				// reconfigure wakes it with bufferLimitError; no timer expiry is proof.
				if err := root.reconfigure(1, 0); err != nil {
					t.Fatal(err)
				}
			}
			waitChan(t, done, "retry handler completion")
			if firstCalls.Load() != 1 || fx.eng["a"].hits() != 0 {
				t.Fatalf("old permission redispatched: fenced=%d original=%d", firstCalls.Load(), fx.eng["a"].hits())
			}
			if insufficient {
				var result struct {
					Code string `json:"code"`
				}
				if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
					t.Fatalf("decode capacity refusal: %v; body=%q", err, recorder.Body.String())
				}
				if recorder.Code != bufferRequestStatus(&bufferLimitError{}) || result.Code != "inference-buffer" || fx.eng["b"].hits() != 0 {
					t.Fatalf("capacity refusal status=%d code=%q second=%d", recorder.Code, result.Code, fx.eng["b"].hits())
				}
				blocker.close()
			} else {
				var result struct {
					Done bool `json:"done"`
				}
				if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
					t.Fatalf("decode retry response: %v; body=%q", err, recorder.Body.String())
				}
				if recorder.Code != http.StatusOK || !result.Done || fx.eng["b"].hits() != 1 || fx.eng["b"].lastBody() != `{"model":"large:7b","stream":true}` {
					t.Fatalf("retry control status=%d done=%t second=%d body=%q", recorder.Code, result.Done, fx.eng["b"].hits(), fx.eng["b"].lastBody())
				}
			}
			ownedAssertHandlerDrained(t, fx.p, control)
		})
	}
	run("insufficient current capacity refuses replacement", true)
	run("sufficient current capacity dispatches replacement", false)
}
