// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

func preparationPrepare(ctx context.Context, body []byte, request originStreamRequest, mode performancePreparationMode) (context.Context, func(), error) {
	return preparePerformanceBuffers(ctx, body, request, mode)
}

func preparationFramePrepare(ctx context.Context, frames *frameBudget, n int64) error {
	return frames.prepare(ctx, n)
}

type preparationClass struct {
	name     string
	raw      string
	mode     performancePreparationMode
	stream   bool
	numeric  bool
	peer     bool
	ceilings [3]int
}

func preparationClasses() []preparationClass {
	return []preparationClass{
		{name: "self numeric", mode: performancePrepareLocalNumeric, stream: true, numeric: true, ceilings: [3]int{15, 15, 15}},
		{name: "self syntax", mode: performancePrepareLocalSyntax, stream: true, ceilings: [3]int{15, 15, 15}},
		{name: "self nonstream", mode: performancePrepareLocalNumeric, ceilings: [3]int{874, 1166, 4665}},
		{name: "self omitted OpenAI", raw: `{"model":"m"}`, mode: performancePrepareLocalNumeric},
		{name: "peer stream", mode: performancePreparePeer, stream: true, peer: true, ceilings: [3]int{15, 15, 15}},
		{name: "peer nonstream", mode: performancePreparePeer, peer: true, ceilings: [3]int{2047, 2729, 10919}},
	}
}

func preparationBody(stream bool) string {
	if stream {
		return `{"model":"m","stream":true}`
	}
	return `{"model":"m","stream":false}`
}

// Intake uses real ContentLength and the production retained-body, model and
// parser leases. It deliberately does not enter the separate request registry.
func preparationIntake(root *bufferBudget, raw string) (context.Context, *ownedInferenceBuffers, []byte, error) {
	p := &Proxy{buffers: root}
	r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(raw)))
	body, model, err := owner.readBody(r, 1<<20)
	if err != nil {
		owner.close()
		return nil, nil, nil, err
	}
	if model != "m" || r.ContentLength != int64(len(raw)) || cap(body) != len(raw) {
		owner.close()
		return nil, nil, nil, fmt.Errorf("intake model=%q content length=%d body capacity=%d, want m/%d/%d", model, r.ContentLength, cap(body), len(raw), len(raw))
	}
	return r.Context(), owner, body, nil
}

func preparationUsed(root *bufferBudget) (int64, int64) {
	root.mu.Lock()
	defer root.mu.Unlock()
	var frames int64
	if root.streams != nil {
		frames = root.streams.used
	}
	return root.used, frames
}

func TestBufferPreparationChargeClasses(t *testing.T) {
	for _, class := range preparationClasses() {
		t.Run(class.name, func(t *testing.T) {
			root := bufferTestBudget(t, 192<<20, 0)
			raw := class.raw
			if raw == "" {
				raw = preparationBody(class.stream)
			}
			ctx, owner, body, err := preparationIntake(root, raw)
			if err != nil {
				t.Fatalf("intake: %v", err)
			}
			defer owner.close()
			before, _ := preparationUsed(root)
			if want := int64(len(body)+len("m")) + (96 << 10); before != want {
				t.Fatalf("body/model/parser charge=%d, want %d", before, want)
			}
			request, err := parseOriginStreamRequest("/v1/chat/completions", body)
			if err != nil {
				t.Fatalf("parse stream request: %v", err)
			}
			_, release, err := preparationPrepare(ctx, body, request, class.mode)
			if err != nil {
				t.Fatalf("prepare: %v", err)
			}
			defer release()
			wantBase := int64(0)
			if !class.peer {
				fingerprint, err := performancePreparationBytes(body)
				if err != nil {
					t.Fatalf("fingerprint allowance: %v", err)
				}
				wantBase = streamFrameNumericBytes + fingerprint
			}
			wantFrames := int64(0)
			if class.stream {
				parser := int64(streamFrameSyntaxBytes)
				if class.numeric {
					parser = streamFrameNumericBytes
				}
				wantFrames = streamFrameMetadataBytes + 1 + streamFrameReadChunk + parser + 4*streamFrameMaxBytes
			}
			after, frames := preparationUsed(root)
			if after-before != wantBase+wantFrames || frames != wantFrames {
				t.Fatalf("preparation root/category=%d/%d, want %d/%d (base=%d)", after-before, frames, wantBase+wantFrames, wantFrames, wantBase)
			}
			release()
			if used, frames := preparationUsed(root); used != before || frames != 0 {
				t.Fatalf("released preparation root/category=%d/%d, want %d/0", used, frames, before)
			}
			owner.close()
			bufferAssertTotals(t, root, 0, 0, 0)
		})
	}
}

func TestBufferPreparationOnlyEffectiveStreamingRoutesPrepareFrames(t *testing.T) {
	for _, mode := range []performancePreparationMode{performancePrepareLocalNumeric, performancePrepareLocalSyntax, performancePreparePeer} {
		for _, route := range []struct {
			name, path, raw string
			frames          bool
		}{
			{"unknown route", "/v1/embeddings", `{"model":"m","stream":true}`, false},
			{"disabled route", "/v1/chat/completions", `{"model":"m","stream":false}`, false},
			{"omitted OpenAI stream", "/v1/chat/completions", `{"model":"m"}`, false},
			{"null OpenAI stream", "/v1/completions", `{"model":"m","stream":null}`, false},
			{"declared SSE route", "/v1/completions", `{"model":"m","stream":true}`, true},
			{"default NDJSON route", "/api/generate", `{"model":"m"}`, true},
			{"null NDJSON stream", "/api/chat", `{"model":"m","stream":null}`, true},
			{"disabled NDJSON route", "/api/chat", `{"model":"m","stream":false}`, false},
		} {
			t.Run(fmt.Sprintf("%v/%s", mode, route.name), func(t *testing.T) {
				root := bufferTestBudget(t, 192<<20, 0)
				ctx, owner, body, err := preparationIntake(root, route.raw)
				if err != nil {
					t.Fatalf("intake: %v", err)
				}
				defer owner.close()
				request, err := parseOriginStreamRequest(route.path, body)
				if err != nil {
					t.Fatalf("parse: %v", err)
				}
				_, release, err := preparationPrepare(ctx, body, request, mode)
				if err != nil {
					t.Fatalf("prepare: %v", err)
				}
				defer release()
				_, frames := preparationUsed(root)
				if (frames > 0) != route.frames {
					t.Fatalf("frame category=%d, want frames=%v", frames, route.frames)
				}
			})
		}
	}
}

func TestBufferPreparationResponseGrantsFollowEffectiveStream(t *testing.T) {
	for _, test := range []struct {
		name, path, raw, media string
		decreased, protected   bool
	}{
		{name: "auto SSE fresh grants", path: routeChatCompletions, raw: `{"model":"m"}`, media: "text/event-stream"},
		{name: "auto SSE unprotected after decrease", path: routeChatCompletions, raw: `{"model":"m"}`, media: "text/event-stream", decreased: true},
		{name: "declared SSE protected after decrease", path: routeChatCompletions, raw: `{"model":"m","stream":true}`, media: "text/event-stream", decreased: true, protected: true},
		{name: "default NDJSON protected after decrease", path: "/api/generate", raw: `{"model":"m"}`, media: "application/x-ndjson", decreased: true, protected: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := bufferTestBudget(t, 192<<20, 0)
			ctx, owner, body, err := preparationIntake(root, test.raw)
			if err != nil {
				t.Fatalf("intake: %v", err)
			}
			defer owner.close()
			request, err := parseOriginStreamRequest(test.path, body)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			prepared, release, err := preparePerformanceBuffers(ctx, body, request, performancePrepareLocalNumeric)
			if err != nil {
				t.Fatalf("prepare: %v", err)
			}
			defer release()
			before, frames := preparationUsed(root)
			if (frames > 0) != test.protected {
				t.Fatalf("prepaid response bytes=%d, want protection=%v", frames, test.protected)
			}
			if test.decreased {
				if err := root.reconfigure(before-1, 0); err != nil {
					t.Fatalf("decrease: %v", err)
				}
			}
			resp := &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {test.media}}, Body: io.NopCloser(strings.NewReader(""))}
			reader, err := installResponseStream(resp, request, prepared, time.Hour, nil)
			if err != nil {
				t.Fatalf("install auto SSE reader: %v", err)
			}
			if reader == nil {
				t.Fatal("auto SSE response was not detected")
			}
			defer func() {
				if err := reader.Close(); err != nil {
					t.Errorf("close auto SSE reader: %v", err)
				}
			}()
			result := reader.Result()
			if test.decreased && !test.protected {
				if result.Outcome != streamOutcomeInterrupted || result.Reason != streamInterruptReasonBudget {
					t.Fatalf("unprotected auto SSE result=%+v", result)
				}
			} else if result.Outcome != streamOutcomeStreaming {
				t.Fatalf("fresh auto SSE reader=%+v", result)
			}
		})
	}
}

func TestBufferPreparationNativeNDJSONIgnoresChoices(t *testing.T) {
	for _, path := range []string{"/api/chat", "/api/generate"} {
		for _, literal := range []string{"100000000", "184467440737095516160000", `"irrelevant"`} {
			t.Run(path+"/n="+literal, func(t *testing.T) {
				body := []byte(`{"model":"m","n":` + literal + `}`)
				request, err := parseOriginStreamRequest(path, body)
				if err != nil {
					t.Fatalf("native n rejected: %v", err)
				}
				if request.choices != 1 || !request.declared || request.protocol != streamProtocolNDJSON {
					t.Fatalf("native request=%+v", request)
				}
			})
		}
	}
}

func TestBufferPreparationEffectiveProfileStreamErrors(t *testing.T) {
	for _, entry := range []string{"origin", "ingress"} {
		t.Run(entry, func(t *testing.T) {
			profile, ok := profileFor("ollama")
			if !ok {
				t.Fatal("missing ollama profile")
			}
			root := bufferTestBudget(t, 192<<20, 0)
			p := testProxy(profile, NewDiscovery(), profile.FacadePort)
			p.buffers = root
			p.noteSelf("self", profile.Name, []string{"m"})
			p.soleFacade().discovery.AddManual(selfNode("self", profile.FacadePort, "m"))
			if err := p.soleFacade().setLocalBackend(localBackend{Host: "127.0.0.1", Port: 1, Healthy: true}); err != nil {
				t.Fatalf("set local engine: %v", err)
			}
			policy := nodepolicy.Default()
			policy.Profiles = []nodepolicy.Profile{{Name: "m", Engine: profile.Name, Model: "m", RequestOptions: json.RawMessage(`{"stream":"yes"}`)}}
			p.admission.setPolicy(policy)
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m"}`))
			if entry == "origin" {
				p.soleFacade().handleHTTP(recorder, request)
			} else {
				p.soleFacade().serveIngress(recorder, request, "peer")
			}
			var response struct {
				Code  string `json:"code"`
				Error string `json:"error"`
			}
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode refusal: %v", err)
			}
			if recorder.Code != http.StatusBadRequest || response.Code != "invalid-stream" || response.Error != "invalid effective stream flag" {
				t.Fatalf("effective stream refusal status=%d body=%+v", recorder.Code, response)
			}
			if p.admission.active != 0 || len(p.priorityReservations) != 0 {
				t.Fatal("effective stream refusal retained admission or routing reservation")
			}
			bufferAssertTotals(t, root, 0, 0, 0)
		})
	}
}

func TestBufferPreparationRawStreamErrorWire(t *testing.T) {
	for _, entry := range []string{"origin", "ingress", "strict local"} {
		t.Run(entry, func(t *testing.T) {
			notifications := &recordingWriter{}
			profile := anyProfile(t)
			p := newTestProxy(profile, NewCodec(notifications), NewDiscovery(), 0)
			root := bufferTestBudget(t, 192<<20, 0)
			p.buffers = root
			raw := `{"model":"m","stream":"yes"}`
			request := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(raw))
			recorder := httptest.NewRecorder()
			switch entry {
			case "origin":
				p.soleFacade().handleHTTP(recorder, request)
			case "ingress":
				p.soleFacade().serveIngress(recorder, request, "peer")
			case "strict local":
				p.soleFacade().serveStrictLocal(recorder, request, []byte(raw), "m", nodepolicy.StrictAdmissionRequest{}, false)
			}
			var response struct {
				Code  string `json:"code"`
				Error string `json:"error"`
			}
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode refusal: %v", err)
			}
			if recorder.Code != http.StatusBadRequest || response.Code != "invalid-stream" || response.Error != "stream must be a single boolean or null member" {
				t.Fatalf("raw stream refusal status=%d body=%+v", recorder.Code, response)
			}
			if entry == "origin" {
				lines := notifications.lines()
				if len(lines) != 1 {
					t.Fatalf("notifications=%d, want terminal refusal", len(lines))
				}
				var notification struct {
					Method string       `json:"method"`
					Params RequestEvent `json:"params"`
				}
				if err := json.Unmarshal(lines[0], &notification); err != nil {
					t.Fatalf("decode notification: %v", err)
				}
				if notification.Method != engines.AddressMethod(profile.Name, "proxy/request") || notification.Params.Status != http.StatusBadRequest || notification.Params.Error != "invalid stream flag" {
					t.Fatalf("raw stream notification=%+v", notification)
				}
			}
			bufferAssertTotals(t, root, 0, 0, 0)
		})
	}
}

func TestBufferPreparationIngressChoicesErrorWire(t *testing.T) {
	for _, entry := range []string{"ingress", "strict local"} {
		t.Run(entry, func(t *testing.T) {
			p := testProxy(anyProfile(t), NewDiscovery(), 0)
			root := bufferTestBudget(t, 192<<20, 0)
			p.buffers = root
			raw := `{"model":"m","stream":false,"n":184467440737095516160000}`
			request := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(raw))
			recorder := httptest.NewRecorder()
			if entry == "ingress" {
				p.soleFacade().serveIngress(recorder, request, "peer")
			} else {
				p.soleFacade().serveStrictLocal(recorder, request, []byte(raw), "m", nodepolicy.StrictAdmissionRequest{}, false)
			}
			var response struct {
				Code  string `json:"code"`
				Error string `json:"error"`
			}
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode refusal: %v", err)
			}
			if recorder.Code != http.StatusRequestEntityTooLarge || response.Code != "inference-buffer" || response.Error != "inference request could not be prepared" {
				t.Fatalf("choice refusal status=%d body=%+v", recorder.Code, response)
			}
			bufferAssertTotals(t, root, 0, 0, 0)
		})
	}
}

func TestBufferPreparationFrameGrantFIFO(t *testing.T) {
	for _, action := range []string{"release", "cancel", "wait bound"} {
		t.Run(action, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				root := bufferTestBudget(t, 100, time.Hour)
				hold := bufferMustAcquire(t, root, 80)
				headCtx, cancelHead := context.WithCancel(context.Background())
				defer cancelHead()
				head := bufferStartAcquire(t, root, headCtx, 90)
				synctest.Wait()
				if err := root.reconfigure(100, time.Second); err != nil {
					t.Fatalf("set later wait bound: %v", err)
				}
				frames := &frameBudget{limit: 64, scope: bufferMustScope(t, root, 64)}
				ctx, cancel := context.WithCancel(context.Background())
				done := make(chan struct{})
				var grantErr error
				go func() {
					grantErr = preparationFramePrepare(ctx, frames, 10)
					close(done)
				}()
				defer func() {
					cancel()
					<-done
					frames.releasePreparation()
				}()
				synctest.Wait()
				// Twenty aggregate bytes are spare. The ten-byte frame grant must
				// still queue behind the earlier ninety-byte proposal.
				bufferAssertTotals(t, root, 80, 1, 2)
				select {
				case <-done:
					t.Fatalf("frame grant bypassed FIFO: %v", grantErr)
				default:
				}
				switch action {
				case "release":
					hold.close()
					synctest.Wait()
					headLease := head.success(t)
					<-done
					if grantErr != nil {
						t.Fatalf("released frame grant: %v", grantErr)
					}
					bufferAssertTotals(t, root, 100, 2, 0)
					headLease.close()
					if got := frames.inUse(); got != 10 {
						t.Fatalf("prepared frame bytes=%d, want 10", got)
					}
				case "cancel", "wait bound":
					cause := context.Canceled
					if action == "cancel" {
						cancel()
						synctest.Wait()
					} else {
						cause = context.DeadlineExceeded
						bufferAdvance(time.Second)
					}
					<-done
					err := grantErr
					bufferAssertBusy(t, err)
					if !errors.Is(err, cause) {
						t.Fatalf("frame grant error=%v, want wrapped %v", err, cause)
					}
					head.pending(t)
					bufferAssertTotals(t, root, 80, 1, 1)
					cancelHead()
					synctest.Wait()
					_, err = head.result(t)
					bufferAssertBusy(t, err)
					if !errors.Is(err, context.Canceled) {
						t.Fatalf("head cancellation=%v", err)
					}
				}
				frames.releasePreparation()
				hold.close()
				if frames.inUse() != 0 || frames.scope.used != 0 {
					t.Fatal("frame preparation retained a grant or scope reservation")
				}
				bufferAssertTotals(t, root, 0, 0, 0)
			})
		})
	}
}

func TestBufferPreparationHugeChoicesTypedLimit(t *testing.T) {
	for _, literal := range []string{"100000000", "184467440737095516160000"} {
		for _, stream := range []bool{true, false} {
			for _, mode := range []performancePreparationMode{performancePrepareLocalNumeric, performancePrepareLocalSyntax, performancePreparePeer} {
				t.Run(fmt.Sprintf("n=%s/stream=%v/mode=%v", literal, stream, mode), func(t *testing.T) {
					root := bufferTestBudget(t, 192<<20, 0)
					raw := fmt.Sprintf(`{"model":"m","stream":%t,"n":%s}`, stream, literal)
					ctx, owner, body, err := preparationIntake(root, raw)
					if err != nil {
						t.Fatalf("intake: %v", err)
					}
					defer owner.close()
					before, _ := preparationUsed(root)
					request, err := parseOriginStreamRequest("/v1/chat/completions", body)
					if err == nil {
						var release func()
						_, release, err = preparationPrepare(ctx, body, request, mode)
						if release != nil {
							release()
						}
					}
					bufferAssertLimit(t, err)
					if used, frames := preparationUsed(root); used != before || frames != 0 {
						t.Fatalf("rejected choices retained root/category=%d/%d, want %d/0", used, frames, before)
					}
					owner.close()
					bufferAssertTotals(t, root, 0, 0, 0)
				})
			}
		}
	}
}

func TestBufferPreparationHugeChoicesHandlerBeforeDispatch(t *testing.T) {
	for _, literal := range []string{"100000000", "184467440737095516160000"} {
		for _, stream := range []bool{true, false} {
			t.Run(fmt.Sprintf("n=%s/stream=%v", literal, stream), func(t *testing.T) {
				root := bufferTestBudget(t, 192<<20, 0)
				notifications := &recordingWriter{}
				profile := anyProfile(t)
				// Construction attaches a facade without starting it or binding a
				// listener. Empty discovery has no real upstream to contact.
				p := newTestProxy(profile, NewCodec(notifications), NewDiscovery(), 0)
				p.buffers = root
				raw := fmt.Sprintf(`{"model":"m","stream":%t,"n":%s}`, stream, literal)
				recorder := httptest.NewRecorder()
				p.soleFacade().handleHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(raw)))
				if recorder.Code != http.StatusRequestEntityTooLarge {
					t.Fatalf("handler status=%d, want 413", recorder.Code)
				}
				var response struct {
					Code string `json:"code"`
				}
				if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
					t.Fatalf("decode refusal: %v", err)
				}
				if response.Code != "inference-buffer" {
					t.Fatalf("refusal code=%q, want inference-buffer", response.Code)
				}
				lines := notifications.lines()
				if len(lines) != 1 {
					t.Fatalf("notifications=%d, want only terminal refusal", len(lines))
				}
				var notification struct {
					Method string       `json:"method"`
					Params RequestEvent `json:"params"`
				}
				if err := json.Unmarshal(lines[0], &notification); err != nil {
					t.Fatalf("decode notification: %v", err)
				}
				if notification.Method != engines.AddressMethod(profile.Name, "proxy/request") || notification.Params.Status != http.StatusRequestEntityTooLarge || notification.Params.Target != "" || notification.Params.NodeID != "" {
					t.Fatalf("unexpected refusal notification: %+v", notification)
				}
				if notification.Params.Error != "inference request could not be prepared" {
					t.Fatalf("refusal notification error=%q", notification.Params.Error)
				}
				if len(p.activeRoutingRequests()) != 0 || len(p.workloadRequests) != 0 || len(p.priorityReservations) != 0 || p.performance.size != 0 {
					t.Fatal("refusal retained request/workload/reservation/performance effects")
				}
				bufferAssertTotals(t, root, 0, 0, 0)
			})
		}
	}
}

type preparationLive struct {
	owner   *ownedInferenceBuffers
	release func()
	reader  *streamFrameReader
}

func (live *preparationLive) close() error {
	var err error
	if live.reader != nil {
		err = live.reader.Close()
	}
	live.release()
	live.owner.close()
	return err
}

// Never read an upstream byte: successful construction, streaming outcome and
// nonempty allocated storage prove these are LIVE readers held before output,
// rather than metadata-only preparation counts. All leases are logical; only
// the tiny body and initial reader arrays are physically allocated.
func preparationConstruct(root *bufferBudget, class preparationClass) (*preparationLive, error) {
	raw := class.raw
	if raw == "" {
		raw = preparationBody(class.stream)
	}
	ctx, owner, body, err := preparationIntake(root, raw)
	if err != nil {
		return nil, err
	}
	request, err := parseOriginStreamRequest("/v1/chat/completions", body)
	if err != nil {
		owner.close()
		return nil, err
	}
	prepared, release, err := preparationPrepare(ctx, body, request, class.mode)
	if err != nil {
		if release != nil {
			release()
		}
		owner.close()
		return nil, err
	}
	live := &preparationLive{owner: owner, release: release}
	if !class.stream {
		return live, nil
	}
	config := streamFrameConfig{
		Protocol: streamProtocolSSE, Parent: prepared, Choices: request.choices,
		Upstream: io.NopCloser(strings.NewReader("")), IdleTimeout: time.Hour,
		PeerNormalized: class.peer,
	}
	if class.numeric {
		config.Extract = func([]byte) performanceExtractMetadata { return performanceExtractMetadata{} }
	}
	live.reader, err = newStreamFrameReader(config)
	if err == nil && live.reader.Result().Outcome != streamOutcomeStreaming {
		err = fmt.Errorf("constructed reader is terminal: %+v", live.reader.Result())
	}
	if err != nil {
		if closeErr := live.close(); closeErr != nil {
			return nil, fmt.Errorf("construct: %v; cleanup: %w", err, closeErr)
		}
		return nil, err
	}
	return live, nil
}

func TestBufferPreparationMeasuredLiveCeilings(t *testing.T) {
	// These isolated preparations omit the registry's per-live-owner root charge
	// and routing snapshots. They are not admitted HTTP concurrency
	// or RSS measurements. Each row gets a fresh process root and shared 64 MiB
	// stream category. No listener, engine, transport or response output is used.
	for _, class := range preparationClasses() {
		for index, mib := range []int64{192, 256, 1024} {
			t.Run(fmt.Sprintf("%s/%dMiB", class.name, mib), func(t *testing.T) {
				root := bufferTestBudget(t, mib<<20, 0)
				var held []*preparationLive
				defer func() {
					for _, live := range held {
						if err := live.close(); err != nil {
							t.Errorf("close live reader: %v", err)
						}
					}
					bufferAssertTotals(t, root, 0, 0, 0)
				}()
				var perRoot, perFrames, frameCapacity int64
				for {
					before, beforeFrames := preparationUsed(root)
					live, err := preparationConstruct(root, class)
					if err != nil {
						bufferAssertBusy(t, err)
						if after, frames := preparationUsed(root); after != before || frames != beforeFrames {
							t.Fatalf("terminal probe leaked root/category=%d/%d, want %d/%d", after, frames, before, beforeFrames)
						}
						break
					}
					held = append(held, live)
					after, frames := preparationUsed(root)
					if len(held) == 1 {
						perRoot, perFrames = after-before, frames-beforeFrames
					}
					if after-before != perRoot || frames-beforeFrames != perFrames {
						t.Fatalf("nonuniform per-request root/category=%d/%d, want %d/%d", after-before, frames-beforeFrames, perRoot, perFrames)
					}
					if live.reader != nil {
						r := live.reader
						if r.bootstrap != nil || r.choiceBytes != 1 || cap(r.buf) != streamFrameReadChunk || len(r.scratch) != streamFrameReadChunk || cap(r.payload) != streamFrameReadChunk {
							t.Fatal("reader is not fully constructed with one choice and initial 4 KiB arrays")
						}
						wantParser := int64(streamFrameSyntaxBytes)
						if class.numeric {
							wantParser = streamFrameNumericBytes
						}
						if r.parserBytes != wantParser {
							t.Fatalf("reader parser=%d, want %d", r.parserBytes, wantParser)
						}
						frameCapacity += int64(cap(r.buf))
					}
					// A runaway regression is bounded without assuming the expected
					// measured count is also the loop's stopping condition.
					if len(held) > int((mib<<20)/(96<<10))+1 {
						t.Fatal("admission exceeded even the retained parser-only ceiling")
					}
				}
				if len(held) == 0 || perRoot <= 0 {
					t.Fatal("no complete request admitted")
				}
				if !class.stream && perFrames != 0 {
					t.Fatalf("nonstream request charged %d response bytes", perFrames)
				}
				byteCeiling := (mib << 20) / perRoot
				if perFrames > 0 {
					byteCeiling = min(byteCeiling, streamFrameBudgetBytes/perFrames)
				}
				if int64(len(held)) != byteCeiling || (class.ceilings[index] != 0 && len(held) != class.ceilings[index]) {
					t.Fatalf("measured live=%d, byte formula=%d, intake reference=%d; per-request root/category=%d/%d", len(held), byteCeiling, class.ceilings[index], perRoot, perFrames)
				}
				if frameCapacity > 10<<20 {
					t.Fatalf("physical initial frame arrays=%d, exceed 10 MiB", frameCapacity)
				}
				// Exactly one bounded row per class/capacity; no request-count log.
				t.Logf("%s %dMiB: live=%d root/request=%d category/request=%d frame-arrays=%d", class.name, mib, len(held), perRoot, perFrames, frameCapacity)
			})
		}
	}
}

func TestBufferPreparationLogicalCategoryNoPhysicalAllocation(t *testing.T) {
	root := bufferTestBudget(t, 1024<<20, 0)
	scope := bufferMustScope(t, root, math.MaxInt64)
	frames := &frameBudget{limit: streamFrameBudgetBytes, scope: scope}
	defer frames.releasePreparation()
	if err := preparationFramePrepare(context.Background(), frames, streamFrameBudgetBytes); err != nil {
		t.Fatalf("logical category grant: %v", err)
	}
	bufferAssertTotals(t, root, streamFrameBudgetBytes, 1, 0)
	bufferAssertLimit(t, preparationFramePrepare(context.Background(), frames, 1))
	frames.releasePreparation()
	bufferAssertTotals(t, root, 0, 0, 0)
}
