// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	settings "nvpair-shared/enginesettings"
)

const switchApplyFixture = `{"nodeId":"transport-alias","requestId":"switch-42","sourceEngine":"ollama","targetEngine":"llamacpp","expectedRevision":"e3v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}`

func switchTestGate(h pinnedPeerHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) { h(w, r, "pinned-principal") }
}

type switchTestTransport func(*http.Request) (*http.Response, error)

func (f switchTestTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func switchTestClient(handler http.Handler) *remoteClient {
	client := &http.Client{Transport: switchTestTransport(func(r *http.Request) (*http.Response, error) {
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, r)
		return recorder.Result(), nil
	})}
	return &remoteClient{base: "https://pinned-peer.invalid", controlHTTP: client, controlReadyHTTP: client}
}

func TestSwitchRouteTables(t *testing.T) {
	if len(switchRemoteMethods) != 3 || len(switchRouteMethods) != 3 {
		t.Fatalf("unexpected route tables: outbound=%v inbound=%v", switchRemoteMethods, switchRouteMethods)
	}
	for _, tc := range []struct{ remote, path, local string }{
		{settings.RemoteSwitchMethod, settings.SwitchApplyPath, settings.SwitchMethod},
		{settings.RemoteSwitchStatusMethod, settings.SwitchStatusPath, settings.SwitchStatusMethod},
		{settings.RemoteSwitchFenceMethod, settings.SwitchFencePath, settings.SwitchFenceMethod},
	} {
		t.Run(tc.remote, func(t *testing.T) {
			if switchRemoteMethods[tc.remote] != tc.path || switchRouteMethods[tc.path] != tc.local {
				t.Fatalf("route mismatch for %s", tc.remote)
			}
		})
	}
}

func TestSwitchPinnedRoundTrip(t *testing.T) {
	for _, tc := range []struct{ method, path, local, body string }{
		{settings.RemoteSwitchMethod, settings.SwitchApplyPath, settings.SwitchMethod, switchApplyFixture},
		{settings.RemoteSwitchStatusMethod, settings.SwitchStatusPath, settings.SwitchStatusMethod, `{"nodeId":"transport-alias","requestId":"switch-42","afterSequence":3}`},
		{settings.RemoteSwitchFenceMethod, settings.SwitchFencePath, settings.SwitchFenceMethod, `{"nodeId":"transport-alias","requestId":"switch-42"}`},
	} {
		t.Run(tc.method, func(t *testing.T) {
			calls := 0
			target := "target-self"
			if tc.local == settings.SwitchFenceMethod {
				target = "resolved-peer"
			}
			handler := switchHandler(target, switchTestGate, func(_ context.Context, method, caller string, params json.RawMessage) (json.RawMessage, error) {
				calls++
				if method != tc.local || caller != "pinned-principal" {
					t.Fatalf("relay identity: method=%q caller=%q", method, caller)
				}
				var p struct {
					NodeID        string  `json:"nodeId"`
					RequestID     string  `json:"requestId"`
					AfterSequence *uint64 `json:"afterSequence"`
				}
				if err := json.Unmarshal(params, &p); err != nil {
					t.Fatal(err)
				}
				if p.NodeID != target || p.RequestID != "switch-42" {
					t.Fatalf("relay params: %+v", p)
				}
				if tc.local == settings.SwitchStatusMethod && (p.AfterSequence == nil || *p.AfterSequence != 3) {
					t.Fatalf("status sequence lost: %+v", p)
				}
				if tc.local == settings.SwitchFenceMethod {
					return json.Marshal(settings.SwitchFenceResult{NodeID: target, RequestID: p.RequestID, Fenced: true})
				}
				return json.RawMessage(`{"configurationRevision":"test-revision"}`), nil
			})
			client := switchTestClient(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != tc.path || r.Header.Get("Content-Type") != "application/json" {
					t.Fatalf("outbound request: %s %s %v", r.Method, r.URL, r.Header)
				}
				var p struct {
					NodeID string `json:"nodeId"`
				}
				// Inspect without consuming the body the target will decode.
				data, err := readRemoteResponse(r.Body, settings.MaxSwitchRequestBytes)
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(data, &p); err != nil {
					t.Fatal(err)
				}
				if p.NodeID != "resolved-peer" {
					t.Fatalf("outbound node=%q", p.NodeID)
				}
				r.Body = io.NopCloser(strings.NewReader(string(data)))
				handler(w, r)
			}))
			result, err := callRemoteSwitch(context.Background(), tc.method, json.RawMessage(tc.body),
				func(node string) (ecPeer, bool) {
					if node != "transport-alias" {
						t.Fatalf("lookup node=%q", node)
					}
					return ecPeer{nodeID: "resolved-peer", clusterUUID: "pinned-peer"}, true
				}, func(_ context.Context, peer ecPeer) (*remoteClient, error) {
					if peer.clusterUUID != "pinned-peer" {
						t.Fatalf("wrong pin: %+v", peer)
					}
					return client, nil
				})
			if err != nil {
				t.Fatal(err)
			}
			if tc.local == settings.SwitchFenceMethod {
				var fence settings.SwitchFenceResult
				if err := json.Unmarshal(result, &fence); err != nil {
					t.Fatal(err)
				}
				if err := fence.Validate(); err != nil {
					t.Fatal(err)
				}
				if !fence.Fenced || fence.NodeID != target || fence.RequestID != "switch-42" || calls != 1 {
					t.Fatalf("fence = %+v calls=%d", fence, calls)
				}
				return
			}
			var status settings.SwitchStatus
			if err := json.Unmarshal(result, &status); err != nil {
				t.Fatal(err)
			}
			if calls != 1 || status.ConfigurationRevision != "test-revision" {
				t.Fatalf("calls=%d result=%+v", calls, status)
			}
		})
	}
}

func TestSwitchFenceRemoteReplyMustMatchCanonicalRequest(t *testing.T) {
	for _, tc := range []struct{ name, result string }{
		{"wrong node", `{"nodeId":"other","requestId":"lost","fenced":true}`},
		{"wrong ID", `{"nodeId":"peer","requestId":"other","fenced":true}`},
		{"neither outcome", `{"nodeId":"peer","requestId":"lost","fenced":false}`},
		{"null", `null`},
		{"case variant discriminant", `{"nodeId":"peer","requestId":"lost","fenced":false,"Fenced":true}`},
		{"duplicate discriminant", `{"nodeId":"peer","requestId":"lost","fenced":false,"fenced":true}`},
		{"duplicate node", `{"nodeId":"other","nodeId":"peer","requestId":"lost","fenced":true}`},
		{"duplicate request", `{"nodeId":"peer","requestId":"other","requestId":"lost","fenced":true}`},
		{"unknown field", `{"nodeId":"peer","requestId":"lost","fenced":true,"extra":true}`},
		{"null receipt", `{"nodeId":"peer","requestId":"lost","fenced":true,"receipt":null}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := switchTestClient(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(tc.result)) }))
			result, err := callRemoteSwitch(context.Background(), settings.RemoteSwitchFenceMethod, json.RawMessage(`{"nodeId":"alias","requestId":"lost"}`),
				func(string) (ecPeer, bool) { return ecPeer{nodeID: "peer"}, true },
				func(context.Context, ecPeer) (*remoteClient, error) { return client, nil })
			if err == nil || result != nil {
				t.Fatalf("invalid reply accepted: %s %v", result, err)
			}
		})
	}
}

func TestSwitchAuthoritativeAbsentHTTPRoundTripPreservesCodeAndMessage(t *testing.T) {
	const message = "Unknown switch receipt for this caller."
	handler := switchHandler("peer", switchTestGate, func(_ context.Context, method, caller string, raw json.RawMessage) (json.RawMessage, error) {
		if method != settings.SwitchStatusMethod || caller != "pinned-principal" {
			t.Fatalf("wrong status relay: %s %s", method, caller)
		}
		request, err := settings.DecodeSwitchStatusRequest(raw)
		if err != nil {
			return nil, err
		}
		if request.NodeID != "peer" || request.RequestID != "lost" {
			t.Fatalf("status params = %+v", request)
		}
		return nil, &codedError{code: settings.SwitchOperationErrorCode, message: message}
	})
	params := json.RawMessage(`{"nodeId":"peer","requestId":"lost"}`)
	recorder := httptest.NewRecorder()
	handler(recorder, httptest.NewRequest(http.MethodPost, settings.SwitchStatusPath, strings.NewReader(string(params))))
	var body codedBody
	if err := json.Unmarshal(recorder.Body.Bytes(), &body); err != nil {
		t.Fatalf("absence is not a coded envelope: %v; %s", err, recorder.Body.String())
	}
	if recorder.Code != http.StatusConflict || body.Error != message || body.Code != "operation_error" {
		t.Fatalf("absence envelope = %d %+v", recorder.Code, body)
	}
	result, err := switchTestClient(handler).postSwitch(context.Background(), settings.SwitchStatusPath, params)
	var coded *codedError
	if !errors.As(err, &coded) || rpcCode(err) != settings.SwitchOperationErrorCode || err.Error() != message || result != nil {
		t.Fatalf("authoritative absence = %s %v (code %d)", result, err, rpcCode(err))
	}
}

func TestSwitchGenericHTTPFailuresDoNotEstablishAbsence(t *testing.T) {
	const message = "Unknown switch receipt for this caller."
	for _, tc := range []struct {
		name     string
		relayErr error
	}{
		{"uncoded coordinator error", errors.New(message)},
		{"different authoritative operation error", &codedError{code: settings.SwitchOperationErrorCode, message: "Coordinator storage unavailable."}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			handler := switchHandler("peer", switchTestGate, func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
				return nil, tc.relayErr
			})
			result, err := switchTestClient(handler).postSwitch(context.Background(), settings.SwitchStatusPath, json.RawMessage(`{"requestId":"lost"}`))
			if err == nil || err.Error() == message || result != nil {
				t.Fatalf("generic error established absence: %s %v", result, err)
			}
		})
	}
}

func TestSwitchStrictDecode(t *testing.T) {
	for _, route := range []struct{ name, path, method, body string }{
		{"apply", settings.SwitchApplyPath, settings.RemoteSwitchMethod, switchApplyFixture},
		{"status", settings.SwitchStatusPath, settings.RemoteSwitchStatusMethod, `{"nodeId":"target","requestId":"switch-42"}`},
		{"fence", settings.SwitchFencePath, settings.RemoteSwitchFenceMethod, `{"nodeId":"target","requestId":"switch-42"}`},
	} {
		t.Run(route.name, func(t *testing.T) {
			for _, tc := range []struct{ name, body string }{
				{"unknown field", strings.TrimSuffix(route.body, "}") + `,"unexpected":true}`},
				{"spoofed caller", strings.TrimSuffix(route.body, "}") + `,"caller":"spoof"}`},
				{"trailing object", route.body + `{}`},
				{"trailing garbage", route.body + `!`},
				{"oversized", route.body + strings.Repeat(" ", settings.MaxSwitchRequestBytes)},
				{"null", `null`},
			} {
				t.Run(tc.name, func(t *testing.T) {
					handler := switchHandler("self", switchTestGate, func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
						t.Fatal("invalid body reached relay")
						return nil, nil
					})
					recorder := httptest.NewRecorder()
					handler(recorder, httptest.NewRequest(http.MethodPost, route.path, strings.NewReader(tc.body)))
					if recorder.Code != http.StatusBadRequest {
						t.Fatalf("HTTP %d", recorder.Code)
					}
					_, err := callRemoteSwitch(context.Background(), route.method, json.RawMessage(tc.body),
						func(string) (ecPeer, bool) { t.Fatal("invalid body reached lookup"); return ecPeer{}, false }, nil)
					if err == nil || rpcCode(err) != settings.SwitchInvalidParamsCode {
						t.Fatalf("outbound error=%v", err)
					}
				})
			}
		})
	}
}

func TestSwitchHTTPRejections(t *testing.T) {
	for _, tc := range []struct {
		name, method, path string
		status             int
	}{
		{"get apply", http.MethodGet, settings.SwitchApplyPath, http.StatusMethodNotAllowed},
		{"put status", http.MethodPut, settings.SwitchStatusPath, http.StatusMethodNotAllowed},
		{"unknown path", http.MethodPost, "/v1/engine-switch/other", http.StatusNotFound},
		{"executor route", http.MethodPost, "/v1/engine-switch/prepare", http.StatusNotFound},
		{"onward relay", http.MethodPost, "/v1/engine-switch/remote-switch", http.StatusNotFound},
	} {
		t.Run(tc.name, func(t *testing.T) {
			handler := switchHandler("self", switchTestGate, func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
				t.Fatal("rejected HTTP request reached relay")
				return nil, nil
			})
			recorder := httptest.NewRecorder()
			handler(recorder, httptest.NewRequest(tc.method, tc.path, strings.NewReader(switchApplyFixture)))
			if recorder.Code != tc.status {
				t.Fatalf("HTTP %d, want %d", recorder.Code, tc.status)
			}
			if tc.status == http.StatusMethodNotAllowed && recorder.Header().Get("Allow") != http.MethodPost {
				t.Fatal("missing Allow")
			}
		})
	}
}

func TestSwitchCodedErrorsRoundTrip(t *testing.T) {
	for _, code := range []int{settings.SwitchInvalidParamsCode, settings.SwitchConfigurationConflictCode, settings.SwitchRemoteVetoCode, settings.SwitchOperationErrorCode} {
		t.Run(fmt.Sprint(code), func(t *testing.T) {
			for _, path := range []string{settings.SwitchApplyPath, settings.SwitchStatusPath} {
				t.Run(path, func(t *testing.T) {
					handler := switchHandler("self", switchTestGate, func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
						return nil, fmt.Errorf("broker: %w", &codedError{code: code, message: "refused"})
					})
					body := switchApplyFixture
					if path == settings.SwitchStatusPath {
						body = `{}`
					}
					recorder := httptest.NewRecorder()
					handler(recorder, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
					wantHTTP := http.StatusConflict
					if code == settings.SwitchInvalidParamsCode {
						wantHTTP = http.StatusBadRequest
					}
					if recorder.Code != wantHTTP {
						t.Fatalf("HTTP %d, want %d", recorder.Code, wantHTTP)
					}
					if code == settings.SwitchConfigurationConflictCode {
						var wire codedBody
						if err := json.Unmarshal(recorder.Body.Bytes(), &wire); err != nil {
							t.Fatal(err)
						}
						if wire.Code != settings.SwitchErrorConfigurationConflict {
							t.Fatalf("wire code=%q", wire.Code)
						}
					}
					_, err := switchTestClient(handler).postSwitch(context.Background(), path, json.RawMessage(body))
					if err == nil || rpcCode(err) != code {
						t.Fatalf("error=%v code=%d, want %d", err, rpcCode(err), code)
					}
				})
			}
		})
	}
}

func TestSwitchPolicyConflictAliasUnchanged(t *testing.T) {
	if wireCodes[codePolicyConflict] != "policy_conflict" {
		t.Fatal("policy wire alias changed")
	}
	err := remoteStatusError("policy", http.StatusConflict, []byte(`{"error":"refused","code":"policy_conflict"}`))
	if rpcCode(err) != codePolicyConflict {
		t.Fatalf("policy error code=%d", rpcCode(err))
	}
}

func TestSwitchPinGateRefusal(t *testing.T) {
	handler := switchHandler("self", func(pinnedPeerHandler) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) { http.Error(w, "not pinned", http.StatusForbidden) }
	}, func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
		t.Fatal("unpinned caller reached relay")
		return nil, nil
	})
	recorder := httptest.NewRecorder()
	handler(recorder, httptest.NewRequest(http.MethodPost, settings.SwitchApplyPath, strings.NewReader(switchApplyFixture)))
	if recorder.Code != http.StatusForbidden {
		t.Fatalf("HTTP %d", recorder.Code)
	}
}

func TestSwitchUnavailableCoordinator(t *testing.T) {
	for _, tc := range []struct {
		name, target string
		relay        switchBrokerCall
	}{
		{name: "no relay", target: "self"},
		{name: "no target identity", relay: func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
			t.Fatal("unconfigured target reached relay")
			return nil, nil
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			switchHandler(tc.target, switchTestGate, tc.relay)(recorder,
				httptest.NewRequest(http.MethodPost, settings.SwitchApplyPath, strings.NewReader(switchApplyFixture)))
			if recorder.Code != http.StatusServiceUnavailable {
				t.Fatalf("HTTP %d", recorder.Code)
			}
		})
	}
}

func TestSwitchOutboundRefusals(t *testing.T) {
	test := func(name, method, body string, found bool, clientErr error, wantCode, wantLookup, wantClient int) {
		t.Run(name, func(t *testing.T) {
			lookups, clients := 0, 0
			_, err := callRemoteSwitch(context.Background(), method, json.RawMessage(body),
				func(string) (ecPeer, bool) {
					lookups++
					return ecPeer{nodeID: "peer"}, found
				}, func(context.Context, ecPeer) (*remoteClient, error) {
					clients++
					return nil, clientErr
				})
			if err == nil || rpcCode(err) != wantCode || lookups != wantLookup || clients != wantClient {
				t.Fatalf("error=%v code=%d lookups=%d clients=%d", err, rpcCode(err), lookups, clients)
			}
			if clientErr != nil && !errors.Is(err, clientErr) {
				t.Fatalf("client error lost: %v", err)
			}
		})
	}
	test("unknown method", "engine:remote-start", switchApplyFixture, false, nil, -32601, 0, 0)
	test("missing node", settings.RemoteSwitchStatusMethod, `{}`, false, nil, settings.SwitchInvalidParamsCode, 0, 0)
	test("undiscovered peer", settings.RemoteSwitchMethod, switchApplyFixture, false, nil, settings.SwitchOperationErrorCode, 1, 0)
	test("pin factory refusal", settings.RemoteSwitchMethod, switchApplyFixture, true, errors.New("pin revoked"), settings.SwitchOperationErrorCode, 1, 1)
}

func TestSwitchBodyLimitBoundary(t *testing.T) {
	for _, tc := range []struct {
		name         string
		size, status int
	}{
		{"at limit", settings.MaxSwitchRequestBytes, http.StatusOK},
		{"over limit", settings.MaxSwitchRequestBytes + 1, http.StatusBadRequest},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			handler := switchHandler("self", switchTestGate, func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
				calls++
				return json.RawMessage(`{}`), nil
			})
			body := switchApplyFixture + strings.Repeat(" ", tc.size-len(switchApplyFixture))
			recorder := httptest.NewRecorder()
			handler(recorder, httptest.NewRequest(http.MethodPost, settings.SwitchApplyPath, strings.NewReader(body)))
			wantCalls := 0
			if tc.status == http.StatusOK {
				wantCalls = 1
			}
			if recorder.Code != tc.status || calls != wantCalls {
				t.Fatalf("HTTP %d calls=%d, want HTTP %d calls=%d", recorder.Code, calls, tc.status, wantCalls)
			}
		})
	}
}

func TestSwitchRequestCancellationStopsWait(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	handler := switchHandler("self", switchTestGate, func(wait context.Context, _ string, _ string, _ json.RawMessage) (json.RawMessage, error) {
		cancel()
		<-wait.Done()
		return nil, wait.Err()
	})
	request := httptest.NewRequest(http.MethodPost, settings.SwitchApplyPath, strings.NewReader(switchApplyFixture)).WithContext(ctx)
	recorder := httptest.NewRecorder()
	handler(recorder, request)
	if !errors.Is(ctx.Err(), context.Canceled) || recorder.Code != http.StatusConflict {
		t.Fatalf("context=%v HTTP=%d", ctx.Err(), recorder.Code)
	}
}
