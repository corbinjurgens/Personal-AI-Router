// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"log/slog"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
	"nvpair-ui-broker/workloadstore"
)

func captureCredentialLogs(t *testing.T) *syncBuffer {
	t.Helper()
	logs := &syncBuffer{}
	previous, writer, flags, prefix := slog.Default(), log.Writer(), log.Flags(), log.Prefix()
	slog.SetDefault(slog.New(slog.NewTextHandler(logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	log.SetOutput(logs)
	t.Cleanup(func() {
		slog.SetDefault(previous)
		log.SetOutput(writer)
		log.SetFlags(flags)
		log.SetPrefix(prefix)
	})
	return logs
}

func assertCredentialAbsent(t *testing.T, label string, value any) {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("could not encode %s for credential inspection", label)
	}
	if strings.Contains(string(raw), brokerTestKey) {
		t.Fatalf("credential appeared in %s", label)
	}
}

func assertCredentialLogsClean(t *testing.T, logs *syncBuffer) {
	t.Helper()
	logs.mu.Lock()
	leaked := strings.Contains(logs.buf.String(), brokerTestKey)
	logs.mu.Unlock()
	if leaked {
		t.Fatal("credential appeared in logs")
	}
}

// Arbitrary client writer errors must not become relay log payloads.
type credentialReplyWriter struct{ out *syncBuffer }

func (w *credentialReplyWriter) Read(p []byte) (int, error) { return w.out.Read(p) }
func (w *credentialReplyWriter) Write(p []byte) (int, error) {
	n, _ := w.out.Write(p)
	return n, errors.New(brokerTestKey)
}

func TestSetCredentialRelayDoesNotLogKey(t *testing.T) {
	for _, scenario := range []string{"result write failure", "RPC error write failure", "transport error write failure", "unavailable manager write failure"} {
		t.Run(scenario, func(t *testing.T) {
			logs := captureCredentialLogs(t)
			b, out := newPolicyBroker(t)
			b.codec = NewCodec(&credentialReplyWriter{out: out})
			engine := attachFakeEngine(t, b)
			engine.setAnswer(func(string, json.RawMessage) (any, string) {
				if scenario == "RPC error write failure" {
					return nil, "credential store unavailable"
				}
				return engines.CredentialState{Supported: true, Configured: true}, ""
			})
			if scenario == "transport error write failure" {
				b.getEngineMgr().peer.Close()
			}
			if scenario == "unavailable manager write failure" {
				b.setEngineMgr(nil)
			}
			// Synchronous dispatch also waits for the failed-write diagnostic.
			b.relayToEngineNow(clientRequest(1, engines.SetCredentialMethod, engines.SetCredentialParams{Engine: "lmstudio", Credential: brokerTestKey}))
			response := out.waitFrame(t, "credential response", isResponse(1))
			if (response.Error == nil) != (scenario == "result write failure") {
				t.Fatal("incorrect credential reply path exercised")
			}
			assertCredentialAbsent(t, "client response", response)
			assertCredentialLogsClean(t, logs)
		})
	}
}

func TestCredentialRelay(t *testing.T) {
	for _, tc := range []struct {
		name       string
		method     string
		params     any
		configured bool
		failure    string
	}{
		{"set", engines.SetCredentialMethod, engines.SetCredentialParams{Engine: "lmstudio", Credential: brokerTestKey}, true, ""},
		{"clear", engines.SetCredentialMethod, engines.SetCredentialParams{Engine: "lmstudio"}, false, ""},
		{"state", engines.CredentialStateMethod, engines.CredentialParams{Engine: "lmstudio"}, true, ""},
		{"failed write", engines.SetCredentialMethod, engines.SetCredentialParams{Engine: "lmstudio", Credential: brokerTestKey}, false, "credential store unavailable"},
		{"failed delete", engines.SetCredentialMethod, engines.SetCredentialParams{Engine: "lmstudio"}, true, "credential store unavailable"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			logs := captureCredentialLogs(t)
			b, out := newPolicyBroker(t)
			b.store = newDiscoveryStore()
			b.workloads = workloadstore.New()
			b.relayDir = relay.NewDirectory()
			b.regCache = relay.NewRegistrationCache()
			peer, peerOutput := newFakeWorkerPipe(t)
			b.setClusterMgr(&clusterManagerProcess{peer: peer})
			t.Cleanup(func() { b.setClusterMgr(nil) })
			engine := attachFakeEngine(t, b)
			stopNotifications := b.startEngineNotificationRelay(context.Background())
			t.Cleanup(stopNotifications)
			engine.setAnswer(func(method string, _ json.RawMessage) (any, string) {
				switch method {
				case engines.SetCredentialMethod, engines.CredentialStateMethod:
					return engines.CredentialState{Supported: true, Configured: tc.configured}, tc.failure
				case "engine:get-installed":
					return map[string]any{"engines": []map[string]any{{"engine": "lmstudio", "installed": true, "running": true, "healthy": true}}}, ""
				case "engine:models":
					return map[string]any{"models": []string{"fixture-model"}}, ""
				default:
					t.Error("unexpected manager operation")
					return nil, "unexpected operation"
				}
			})
			b.handleMessage(clientRequest(10, "discovery:subscribe", nil))
			b.handleMessage(clientRequest(11, "engine:subscribe", nil))
			b.handleMessage(clientRequest(1, tc.method, tc.params))
			response := out.waitFrame(t, tc.name, isResponse(1))
			if tc.failure != "" {
				if response.Error == nil || response.Error.Code != -32000 || response.Error.Message != tc.failure || len(response.Result) != 0 {
					t.Fatal("failed mutation did not preserve the sanitized store error")
				}
			} else {
				var fields map[string]bool
				if response.Error != nil || json.Unmarshal(response.Result, &fields) != nil || !reflect.DeepEqual(fields, map[string]bool{"supported": true, "configured": tc.configured}) {
					t.Fatal("response was not the exact sanitized credential state")
				}
			}
			calls := engine.snapshot()
			want, err := json.Marshal(tc.params)
			if err != nil || len(calls) != 1 || calls[0].method != tc.method || calls[0].notify || !reflect.DeepEqual(calls[0].params, json.RawMessage(want)) {
				t.Fatal("manager did not receive the exact private request")
			}
			// Exercise snapshots, discovery pushes and service metadata sent
			// toward peers after the credential operation has completed.
			b.handleMessage(clientRequest(12, "discovery:get-nodes", nil))
			b.handleMessage(clientRequest(13, "workloads:get-initial", nil))
			b.handleMessage(clientRequest(14, "engine:get-installed", nil))
			b.handleMessage(clientRequest(15, "engine:models", nil))
			out.waitFrame(t, "installed engine snapshot", isResponse(14))
			out.waitFrame(t, "model snapshot", isResponse(15))
			// Ordinary manager events remain subscribed without introducing a
			// credential event or merging the private request into their state.
			_, source := b.pauseManager()
			b.forwardEngineNotificationFrom(source, "engine:models-changed", json.RawMessage(`{"models":["fixture-model"]}`))
			out.waitFrame(t, "model notification", func(frame Message) bool { return frame.Method == "engine:models-changed" })
			b.registerService(noderec.RegisterParams{Service: noderec.ServiceLMStudio, Port: 25001})
			peerOutput.waitFor("published services", func(calls []policyCall) bool {
				for _, call := range calls {
					var snapshot localServicesSnapshot
					if call.method == setLocalServicesMethod && json.Unmarshal(call.params, &snapshot) == nil && snapshot.Generation == 1 {
						return true
					}
				}
				return false
			})
			frames := out.frames()
			var discoveryPush bool
			for _, frame := range frames {
				discoveryPush = discoveryPush || frame.Method == "discovery:nodes-changed"
				if frame.IsNotification() && strings.HasPrefix(frame.Method, "engine:") && frame.Method != "engine:models-changed" {
					t.Fatal("credential operation emitted an engine notification")
				}
			}
			if !discoveryPush {
				t.Fatal("discovery notification was not exercised")
			}
			assertCredentialAbsent(t, "client snapshots and notifications", frames)
			assertCredentialAbsent(t, "discovery registrations", b.regCache.Snapshot())
			assertCredentialAbsent(t, "discovery nodes", b.store.Snapshot())
			for _, call := range peerOutput.snapshot() {
				assertCredentialAbsent(t, "peer method", call.method)
				assertCredentialAbsent(t, "peer payload", call.params)
			}
			assertCredentialLogsClean(t, logs)
		})
	}
}

func TestCredentialRelayRetainsGenericEngineDispatch(t *testing.T) {
	b, out := newPolicyBroker(t)
	engine := attachFakeEngine(t, b)
	b.handleMessage(clientRequest(1, "engine:fixture-operation", map[string]string{"engine": "fixture"}))
	if out.waitFrame(t, "generic response", isResponse(1)).Error != nil {
		t.Fatal("generic engine relay refused a local request")
	}
	if calls := engine.snapshot(); len(calls) != 1 || calls[0].method != "engine:fixture-operation" {
		t.Fatal("generic private relay was not retained")
	}
}
