// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/modeldownload"
	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

func brokerDownloadSnapshot() *modeldownload.Snapshot {
	percent := 21
	return &modeldownload.Snapshot{Version: modeldownload.Version, SourceID: strings.Repeat("a", 32), Revision: "19", RecoveryRequired: true,
		Entries: []modeldownload.Entry{{ID: strings.Repeat("b", 32), Engine: "llamacpp", Model: "org/model", Phase: modeldownload.PhaseUnknown,
			BlocksReadiness: true, Percent: &percent, Reason: modeldownload.ReasonCleanupUnknown, Persistence: modeldownload.PersistenceUncertain}}}
}

func TestModelDownloadEngineResponseRelay(t *testing.T) {
	test := func(name string, snapshot *modeldownload.Snapshot) {
		t.Run(name, func(t *testing.T) {
			var wire bytes.Buffer
			b := &Broker{codec: NewCodec(&wire)}
			source := b.activateEngineManagerSource()
			_, engine := attachBoundEngineManager(t, b, source)
			want := struct {
				Models         []string                `json:"models"`
				ModelsByEngine map[string][]string     `json:"modelsByEngine"`
				LoadedByEngine map[string][]string     `json:"loadedByEngine"`
				DownloadState  *modeldownload.Snapshot `json:"downloadState,omitempty"`
			}{[]string{"ready"}, map[string][]string{"llamacpp": {"ready"}}, map[string][]string{"llamacpp": {"org/model"}}, snapshot}
			engine.mu.Lock()
			engine.answer = func(method string, params json.RawMessage) (any, string) {
				if method != "engine:models" {
					return nil, "unexpected method"
				}
				return want, ""
			}
			engine.mu.Unlock()
			id := json.RawMessage(`"download-models"`)
			b.relayToEngineNow(&Message{JSONRPC: "2.0", ID: &id, Method: "engine:models", Params: json.RawMessage(`{}`)})
			response, err := b.codec.Read()
			if err != nil {
				t.Fatalf("read relayed models: %v", err)
			}
			if response.Error != nil || response.ID == nil || !bytes.Equal(*response.ID, id) {
				t.Fatalf("wrong response: %+v", response)
			}
			got := want
			got.Models, got.ModelsByEngine, got.LoadedByEngine, got.DownloadState = nil, nil, nil, nil
			if err := json.Unmarshal(response.Result, &got); err != nil {
				t.Fatalf("decode relayed models: %v", err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("generic relay changed lifecycle/inventory/residency: got %+v, want %+v", got, want)
			}
		})
	}
	test("all fields", brokerDownloadSnapshot())
	empty := brokerDownloadSnapshot()
	empty.Entries = []modeldownload.Entry{}
	test("explicit empty", empty)
	test("unsupported remains absent", nil)
}

func TestModelDownloadEngineNotificationRelay(t *testing.T) {
	test := func(name, method string, snapshot *modeldownload.Snapshot) {
		t.Run(name, func(t *testing.T) {
			var wire bytes.Buffer
			b := &Broker{codec: NewCodec(&wire), engineSubscribed: true}
			source := b.activateEngineManagerSource()
			want := map[string]any{"engine": "llamacpp"}
			if method == "engine:pull-progress" {
				want["id"], want["model"], want["op"] = strings.Repeat("b", 32), "org/model", "pull"
				want["stage"], want["percent"], want["downloadState"] = "error", -1, snapshot
			} else {
				want["models"] = map[string]any{
					"models": []string{"ready"}, "modelsByEngine": map[string][]string{"llamacpp": {"ready"}},
					"loadedByEngine": map[string][]string{"llamacpp": {"org/model"}}, "downloadState": snapshot,
				}
			}
			raw, err := json.Marshal(want)
			if err != nil {
				t.Fatalf("encode notification: %v", err)
			}
			b.forwardEngineNotificationFrom(source, method, raw)
			message, err := b.codec.Read()
			if err != nil {
				t.Fatalf("read notification: %v", err)
			}
			if message.Method != method || !message.IsNotification() {
				t.Fatalf("wrong notification: %+v", message)
			}
			var got, expected map[string]json.RawMessage
			if err := json.Unmarshal(message.Params, &got); err != nil {
				t.Fatalf("decode relayed notification: %v", err)
			}
			if err := json.Unmarshal(raw, &expected); err != nil {
				t.Fatalf("decode expected notification: %v", err)
			}
			if !reflect.DeepEqual(got, expected) {
				t.Fatalf("generic notification relay changed fields: %s", message.Params)
			}
		})
	}
	test("pull state all fields", "engine:pull-progress", brokerDownloadSnapshot())
	empty := brokerDownloadSnapshot()
	empty.Entries = []modeldownload.Entry{}
	test("models changed explicit empty", "engine:models-changed", empty)
	test("models changed unknown", "engine:models-changed", nil)
}

func TestModelDownloadDiscoveryInputIsCloned(t *testing.T) {
	current := true
	node := noderec.DirectoryNode{HostUUID: "target", Name: "target", DownloadState: brokerDownloadSnapshot(), DownloadStateCurrent: &current}
	enriched := directoryToEnriched(node)
	store := newDiscoveryStore()
	store.Upsert(enriched, sourceScanner)
	*node.DownloadState.Entries[0].Percent = 50
	*enriched.DownloadState.Entries[0].Percent = 60
	current = false
	got := store.nodes["target"].projected()
	if got.DownloadState == nil || *got.DownloadState.Entries[0].Percent != 21 || got.DownloadStateCurrent == nil || !*got.DownloadStateCurrent {
		t.Fatal("discovery retained caller-owned download snapshot or currentness")
	}
}

func TestModelDownloadDiscoverySourceMerge(t *testing.T) {
	current, stale := true, false
	test := func(name string, state *modeldownload.Snapshot, freshness *bool) {
		t.Run(name, func(t *testing.T) {
			store := newDiscoveryStore()
			store.Upsert(EnrichedNode{HostUUID: "target", ID: "manual", Models: []string{"manual-ready"}}, sourceManual)
			node := noderec.DirectoryNode{HostUUID: "target", Name: "scanner", Models: []string{"canonical-ready"},
				LoadedByEngine: map[string][]string{"llamacpp": {"org/model"}}, DownloadState: state, DownloadStateCurrent: freshness}
			store.Upsert(directoryToEnriched(node), sourceScanner)
			got := store.nodes["target"].projected()
			if !reflect.DeepEqual(got.DownloadState, state) || !reflect.DeepEqual(got.DownloadStateCurrent, freshness) ||
				!reflect.DeepEqual(got.Models, node.Models) || !reflect.DeepEqual(got.LoadedByEngine, node.LoadedByEngine) {
				t.Fatalf("source merge changed lifecycle/canonical inventory/residency: %+v", got)
			}
			if store.Remove("target", sourceScanner) {
				t.Fatal("scanner withdrawal evicted surviving manual source")
			}
			got = store.nodes["target"].projected()
			if got.DownloadState != nil || got.DownloadStateCurrent != nil || !reflect.DeepEqual(got.Models, []string{"manual-ready"}) {
				t.Fatalf("manual survivor inherited scanner lifecycle support: %+v", got)
			}
		})
	}
	test("all fields", brokerDownloadSnapshot(), &current)
	test("stale display", brokerDownloadSnapshot(), &stale)
	empty := brokerDownloadSnapshot()
	empty.Entries = []modeldownload.Entry{}
	test("known empty", empty, &current)
	test("unsupported remains unknown", nil, nil)
}

func TestModelDownloadDiscoveryWire(t *testing.T) {
	current, stale := true, false
	test := func(name string, state *modeldownload.Snapshot, freshness *bool, models []string) {
		t.Run(name, func(t *testing.T) {
			node := noderec.DirectoryNode{HostUUID: "target", Name: "target", IP: "127.0.0.1", Models: models,
				ModelsByEngine: map[string][]string{"llamacpp": models}, LoadedByEngine: map[string][]string{"llamacpp": {"org/model"}},
				DownloadState: state, DownloadStateCurrent: freshness}
			raw, err := json.Marshal(noderec.NodeEvent{Node: node})
			if err != nil {
				t.Fatalf("encode scanner discovery event: %v", err)
			}
			store := newDiscoveryStore()
			directory := relay.NewDirectory()
			scanner := &scannerProcess{store: store, relayDir: directory}
			scanner.handleNotify(noderec.NotifyNodeUpdated, raw)
			internal := directory.Snapshot("")
			if len(internal) != 1 || !reflect.DeepEqual(internal[0].DownloadState, state) || !reflect.DeepEqual(internal[0].DownloadStateCurrent, freshness) {
				t.Fatal("internal discovery parser lost download observation")
			}
			available := store.Snapshot()
			encoded, err := json.Marshal(available)
			if err != nil {
				t.Fatalf("encode available nodes: %v", err)
			}
			var got []AvailableNode
			if err := json.Unmarshal(encoded, &got); err != nil {
				t.Fatalf("decode available nodes: %v", err)
			}
			if len(got) != 1 || !reflect.DeepEqual(got[0].DownloadState, state) || !reflect.DeepEqual(got[0].DownloadStateCurrent, freshness) ||
				!reflect.DeepEqual(got[0].ModelsByEngine, node.ModelsByEngine) || !reflect.DeepEqual(got[0].LoadedByEngine, node.LoadedByEngine) {
				t.Fatalf("discovery output changed lifecycle/inventory/residency: %s", encoded)
			}
			if len(models) == 0 && len(got[0].Models) != 0 {
				t.Fatal("discovery parser manufactured ready names")
			}
			if available[0].DownloadState != nil {
				available[0].DownloadState.Revision = "999"
				*available[0].DownloadStateCurrent = false
				if next := store.Snapshot(); !reflect.DeepEqual(next[0].DownloadState, state) || !reflect.DeepEqual(next[0].DownloadStateCurrent, freshness) {
					t.Fatal("snapshot exposed store-owned lifecycle pointers")
				}
			}
		})
	}
	test("all fields", brokerDownloadSnapshot(), &current, []string{"ready"})
	test("withdrawn names with stale display", brokerDownloadSnapshot(), &stale, []string{})
	empty := brokerDownloadSnapshot()
	empty.Entries = []modeldownload.Entry{}
	test("explicit empty", empty, &current, []string{"ready"})
	test("unsupported", nil, nil, []string{"legacy-ready"})
	test("invalid without prior details", nil, &stale, []string{})
}
