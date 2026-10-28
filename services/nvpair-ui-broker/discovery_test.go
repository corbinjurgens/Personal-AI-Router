// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/errors"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

func TestMemoryTelemetryRetiredSourceCannotRetireReplacementMemory(t *testing.T) {
	b := &Broker{telemetry: newTelemetryCache()}
	b.telemetry.clock = func() (uint64, error) { return 1_000_000_000, nil }
	old, current := &scannerProcess{}, &scannerProcess{}
	b.bindScannerSource(old)
	b.bindScannerSource(current)
	b.telemetry.Upsert(sourceScanner, noderec.NodeTelemetry{HostUUID: "self", Memory: memoryTelemetryFixture(), MemoryHop: noderec.NewMemoryHop("scanner", b.telemetry.clock)}, time.Now())
	old.onMemoryRetired("self")
	if got := b.telemetry.LookupMemory("self", sourceScanner, "producer", time.Now()); got.Reason != "" {
		t.Fatalf("retired source invalidated replacement: %s", got.Reason)
	}
	current.onMemoryRetired("self")
	if got := b.telemetry.LookupMemory("self", sourceScanner, "producer", time.Now()); got.Reason != "invalid" {
		t.Fatalf("live source failed to retire memory: %s", got.Reason)
	}
}

func TestMemoryTelemetryGenerationRetainsSchedulerObservation(t *testing.T) {
	b := &Broker{nodeID: "self", telemetry: newTelemetryCache()}
	b.telemetry.clock = func() (uint64, error) { return 1_000_000_000, nil }
	s := &scannerProcess{}
	b.bindScannerSource(s)
	emit := func(generation string) {
		t.Helper()
		node := noderec.DirectoryNode{HostUUID: "self", Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: 14318}}}
		raw, err := json.Marshal(noderec.NodeEvent{Node: node, MemoryGeneration: &generation})
		if err != nil {
			t.Fatal(err)
		}
		s.handleNotify(noderec.NotifyNodeUpdated, raw)
	}
	emit("1")
	received := time.Unix(1_700_000_000, 0)
	b.telemetry.Upsert(sourceScanner, noderec.NodeTelemetry{HostUUID: "self", GPUUtilizationPct: 95, TelemetryValid: true, MSSince: 100, Memory: memoryTelemetryFixture(), MemoryHop: &noderec.MemoryHop{ForwarderID: "scanner", BootNS: "1000000000"}}, received)
	emit("2")
	got := b.telemetry.Snapshot(received)
	if len(got) != 1 {
		t.Fatal("generation change removed scheduler observation")
	}
	if got[0].GPUUtilizationPct != 95 || !got[0].TelemetryValid || got[0].MSSince != 100 {
		t.Fatalf("generation changed scheduler observation: gpu=%d valid=%v age=%d", got[0].GPUUtilizationPct, got[0].TelemetryValid, got[0].MSSince)
	}
	if got[0].Memory != nil {
		t.Fatal("generation retained retired memory")
	}
	b.telemetry.mu.RLock()
	defer b.telemetry.mu.RUnlock()
	if !b.telemetry.nodes["self"].scanner.receivedAt.Equal(received) {
		t.Fatal("generation reset original receipt")
	}
}

func TestMemoryTelemetryRetiredScannerCallback(t *testing.T) {
	b := &Broker{telemetry: newTelemetryCache()}
	old := &scannerProcess{onTelemetry: func(value noderec.NodeTelemetry) { b.ingestTelemetry(sourceScanner, value) }}
	b.bindScannerSource(old)
	replacement := &scannerProcess{}
	b.bindScannerSource(replacement)
	params, err := json.Marshal(noderec.NodeTelemetry{HostUUID: "peer", Memory: memoryTelemetryFixture()})
	if err != nil {
		t.Fatal(err)
	}
	old.handleNotify(noderec.NotifyNodeTelemetry, params)
	if len(b.telemetry.Snapshot(time.Now())) != 0 {
		t.Fatal("retired scanner callback reached cache")
	}
	old.onExit()
	if !replacement.current() {
		t.Fatal("retired scanner exit retired current worker")
	}
	replacement.onExit()
	if replacement.current() {
		t.Fatal("exited worker remains current")
	}
}

func TestMemoryTelemetryAdmittedScannerCallbackRetiresBeforeMutation(t *testing.T) {
	b := newManualTestBroker()
	b.telemetry.clock = func() (uint64, error) { return 1000000000, nil }
	source := &scannerProcess{onTelemetry: func(value noderec.NodeTelemetry) { b.ingestTelemetry(sourceScanner, value) }}
	b.bindScannerSource(source)
	source.memoryGenerations = map[string]string{"peer": "1"}
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	current := source.current
	source.current = func() bool {
		admitted := current()
		close(entered)
		<-release
		return admitted
	}
	params, err := json.Marshal(noderec.NodeTelemetry{HostUUID: "peer", Memory: memoryTelemetryFixture(), MemoryHop: &noderec.MemoryHop{ForwarderID: "fixture", BootNS: "1000000000"}, MemoryRequest: &noderec.MemoryRequestContext{Generation: "1", RequestBootNS: "1000000000"}})
	if err != nil {
		t.Fatal(err)
	}
	go func() { defer close(done); source.handleNotify(noderec.NotifyNodeTelemetry, params) }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		close(release)
		t.Fatal("callback did not reach admission barrier")
	}
	source.onExit()
	close(release)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("callback did not finish")
	}
	if len(b.telemetry.Snapshot(time.Now())) != 0 {
		t.Fatal("admitted retired callback restored scanner telemetry")
	}
	replacement := &scannerProcess{}
	b.bindScannerSource(replacement)
	replacement.memoryGenerations = map[string]string{"peer": "1"}
	liveDone := make(chan struct{})
	go func() { defer close(liveDone); replacement.handleNotify(noderec.NotifyNodeTelemetry, params) }()
	select {
	case <-liveDone:
	case <-time.After(5 * time.Second):
		t.Fatal("live scanner callback deadlocked during fanout")
	}
	view := b.telemetry.Snapshot(time.Now())
	if len(view) != 1 || view[0].Memory == nil || !view[0].Memory.Domains[0].Free.Valid {
		t.Fatal("live replacement scanner did not apply memory")
	}
}

func TestMemoryTelemetryRetiredManualCallbackCannotRestoreAlias(t *testing.T) {
	b := newManualTestBroker()
	b.telemetry.clock = func() (uint64, error) { return 1000000000, nil }
	callback, retire := b.bindManualNodesSource(&rpcWorker{name: "fixture"})
	status := manualStatus("alias", "127.0.0.1", "peer")
	status.MemoryObservation = memoryTelemetryFixture()
	status.MemoryHop = &noderec.MemoryHop{ForwarderID: "fixture", BootNS: "1000000000"}
	params, err := json.Marshal(status)
	if err != nil {
		t.Fatal(err)
	}
	callback("node/discovered", params)
	if len(b.store.Snapshot()) != 1 || len(b.telemetry.Snapshot(time.Now())) != 1 {
		t.Fatal("live manual callback did not apply")
	}
	b.clearManualNodesState()
	callback("node/updated", params)
	if len(b.store.Snapshot()) != 0 || len(b.telemetry.Snapshot(time.Now())) != 0 {
		t.Fatal("retired manual callback restored node or memory")
	}
	b.manualMu.Lock()
	aliases := len(b.manualNodeKeys)
	b.manualMu.Unlock()
	if aliases != 0 {
		t.Fatal("retired manual callback restored alias")
	}
	live, _ := b.bindManualNodesSource(&rpcWorker{name: "replacement"})
	live("node/discovered", params)
	retire()
	view := b.telemetry.Snapshot(time.Now())
	if len(b.store.Snapshot()) != 1 || len(view) != 1 || view[0].Memory == nil || !view[0].Memory.Domains[0].Free.Valid {
		t.Fatal("live replacement or retired exit lost current manual state")
	}
}

func TestMemoryTelemetryManualRetirementSerializesAdmittedCallback(t *testing.T) {
	b := newManualTestBroker()
	b.telemetry.clock = func() (uint64, error) { return 1000000000, nil }
	callback, retire := b.bindManualNodesSource(&rpcWorker{name: "fixture"})
	entered, release, notified, retired := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
	var releaseOnce, notifyOnce sync.Once
	allow := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(allow)
	b.store.SetOnChange(func() {
		// Fanout may consult worker handles without workersMu reentrancy.
		b.getManualNodes()
		notifyOnce.Do(func() { close(entered); <-release })
	})
	status := manualStatus("alias", "127.0.0.1", "peer")
	status.MemoryObservation = memoryTelemetryFixture()
	status.MemoryHop = &noderec.MemoryHop{ForwarderID: "fixture", BootNS: "1000000000"}
	params, err := json.Marshal(status)
	if err != nil {
		t.Fatal(err)
	}
	go func() { defer close(notified); callback("node/discovered", params) }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("callback did not reach projection barrier")
	}
	if b.manualMu.TryLock() {
		b.manualMu.Unlock()
		t.Fatal("source mutation was not serialized with retirement")
	}
	go func() { defer close(retired); retire() }()
	allow()
	select {
	case <-notified:
	case <-time.After(5 * time.Second):
		t.Fatal("callback did not finish")
	}
	select {
	case <-retired:
	case <-time.After(5 * time.Second):
		t.Fatal("retirement did not finish")
	}
	callback("node/updated", params)
	if len(b.store.Snapshot()) != 0 || len(b.telemetry.Snapshot(time.Now())) != 0 {
		t.Fatal("admitted callback survived retirement")
	}
	b.manualMu.Lock()
	aliases := len(b.manualNodeKeys)
	b.manualMu.Unlock()
	if aliases != 0 {
		t.Fatal("admitted callback left a retired alias")
	}
}

// Consume the manual worker's existing status fields through its real broker
// dispatcher, store mutation hook, subscription and ordinary client response.
func TestModelControlsManualAttributionWireTransitions(t *testing.T) {
	b := newManualTestBroker()
	var clientWire bytes.Buffer
	codec := NewCodec(&clientWire)
	b.codec = codec
	b.store.SetOnChange(b.emitNodesChanged)
	id := json.RawMessage(`"304-manual-subscribe"`)
	b.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: "discovery:subscribe"})
	ack, err := codec.Read()
	if err != nil {
		t.Fatal(err)
	}
	if !ack.IsResponse() || ack.Error != nil {
		t.Fatal("ordinary manual discovery subscription failed")
	}
	initial, err := codec.Read()
	if err != nil {
		t.Fatal(err)
	}
	if initial.Method != "discovery:nodes-changed" || string(initial.Params) != "[]" {
		t.Fatal("fresh subscription did not start with the empty store")
	}
	for _, tc := range []struct {
		name, fields string
		attribution  map[string][]string
		flat         []string
	}{
		{"unknown omitted", ``, nil, nil},
		{"unknown null", `,"ollama_models":null,"lmstudio_models":null`, nil, nil},
		{"Ollama observed empty", `,"ollama_models":[]`, map[string][]string{"ollama": {}}, nil},
		{"Ollama empty repeat", `,"ollama_models":[]`, map[string][]string{"ollama": {}}, nil},
		{"empty to unknown", ``, nil, nil},
		{"LM Studio observed empty", `,"lmstudio_models":[]`, map[string][]string{"lmstudio": {}}, nil},
		{"both observed empty", `,"ollama_models":[],"lmstudio_models":[]`, map[string][]string{"ollama": {}, "lmstudio": {}}, nil},
		{"empty Ollama and useful LM Studio", `,"ollama_models":[],"lmstudio_models":["first","last"]`, map[string][]string{"ollama": {}, "lmstudio": {"first", "last"}}, []string{"first", "last"}},
		{"useful Ollama and empty LM Studio", `,"ollama_models":["first","last"],"lmstudio_models":[]`, map[string][]string{"ollama": {"first", "last"}, "lmstudio": {}}, []string{"first", "last"}},
		{"unknown sibling has no invented key", `,"ollama_models":null,"lmstudio_models":["first","last"]`, map[string][]string{"lmstudio": {"first", "last"}}, []string{"first", "last"}},
		{"populated siblings preserve flat order", `,"ollama_models":["first","shared"],"lmstudio_models":["shared","last"]`, map[string][]string{"ollama": {"first", "shared"}, "lmstudio": {"shared", "last"}}, []string{"first", "shared", "last"}},
		{"return to unknown", ``, nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := json.RawMessage(`{"id":"manual-presence","address":"127.0.0.1","hostUuid":"manual-owner","node_info_port":14318,"ollama_up":true,"ollama_port":11434,"lmstudio_up":true,"lmstudio_port":1234,"node_info_up":true,"telemetryValid":true,"msSince":7` + tc.fields + `}`)
			var workerWire bytes.Buffer
			workerCodec := NewCodec(&workerWire)
			method := "node/updated"
			if tc.name == "unknown omitted" {
				method = "node/discovered"
			}
			if err := workerCodec.Notify(method, raw); err != nil {
				t.Fatal(err)
			}
			frame, err := workerCodec.Read()
			if err != nil {
				t.Fatal(err)
			}
			b.forwardManualNodesNotification(frame.Method, frame.Params)
			status := b.manualNodeStatuses["manual-presence"].status
			if !status.OllamaUp || !status.LMStudioUp || !status.TelemetryValid || status.MSSince != 7 {
				t.Fatal("manual notification changed ordinary status metadata")
			}
			assertClient := func(raw json.RawMessage) {
				t.Helper()
				var nodes []AvailableNode
				if err := json.Unmarshal(raw, &nodes); err != nil {
					t.Fatal(err)
				}
				if len(nodes) != 1 || nodes[0].ID != "manual-presence" || nodes[0].HostUUID != "manual-owner" || nodes[0].IPAddress != "127.0.0.1" || nodes[0].Port != 14318 ||
					!reflect.DeepEqual(nodes[0].ModelsByEngine, tc.attribution) || !reflect.DeepEqual(nodes[0].Models, tc.flat) || nodes[0].LoadedByEngine != nil {
					t.Fatalf("manual status/store/client wire lost inventory or metadata: %s", raw)
				}
				var fields []map[string]json.RawMessage
				if err := json.Unmarshal(raw, &fields); err != nil {
					t.Fatal(err)
				}
				_, present := fields[0]["modelsByEngine"]
				if present != (tc.attribution != nil) {
					t.Fatal("client wire manufactured or omitted manual attribution")
				}
			}
			changed, err := codec.Read()
			if err != nil {
				t.Fatal(err)
			}
			if changed.Method != "discovery:nodes-changed" || !changed.IsNotification() {
				t.Fatal("manual store mutation did not emit the ordinary subscribed snapshot")
			}
			assertClient(changed.Params)
			id := json.RawMessage(`"304-manual-get"`)
			b.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: "discovery:get-nodes"})
			response, err := codec.Read()
			if err != nil {
				t.Fatal(err)
			}
			if !response.IsResponse() || response.Error != nil {
				t.Fatal("ordinary manual get-nodes response failed")
			}
			var result map[string]json.RawMessage
			if err := json.Unmarshal(response.Result, &result); err != nil {
				t.Fatal(err)
			}
			assertClient(result["nodes"])
		})
	}
}

func TestModelControlsBrokerAttributionWire(t *testing.T) {
	for _, tc := range []struct {
		name        string
		raw         string
		attribution map[string][]string
	}{
		{"nil omission retains legacy fallback", `{"hostUuid":"owner","name":"owner","ip":"127.0.0.1","models":["good"]}`, nil},
		{"outer empty remains authoritative", `{"hostUuid":"owner","name":"owner","ip":"127.0.0.1","models":["good"],"modelsByEngine":{}}`, map[string][]string{}},
		{"per engine empty", `{"hostUuid":"owner","name":"owner","ip":"127.0.0.1","models":["good"],"modelsByEngine":{"ollama":[]}}`, map[string][]string{"ollama": {}}},
		{"populated and empty siblings", `{"hostUuid":"owner","name":"owner","ip":"127.0.0.1","models":["good"],"modelsByEngine":{"ollama":["first","last"],"lmstudio":[]}}`, map[string][]string{"ollama": {"first", "last"}, "lmstudio": {}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var node noderec.DirectoryNode
			if err := json.Unmarshal([]byte(tc.raw), &node); err != nil {
				t.Fatalf("decode original raw specimen: %v", err)
			}
			var scannerWire bytes.Buffer
			scannerCodec := NewCodec(&scannerWire)
			if err := scannerCodec.Notify(noderec.NotifyNodeUpdated, noderec.NodeEvent{Node: node}); err != nil {
				t.Fatal(err)
			}
			frame, err := scannerCodec.Read()
			if err != nil {
				t.Fatal(err)
			}
			store := newDiscoveryStore()
			directory := relay.NewDirectory()
			scanner := &scannerProcess{store: store, relayDir: directory}
			scanner.handleNotify(frame.Method, frame.Params)
			internal := directory.Snapshot("")
			if len(internal) != 1 || !reflect.DeepEqual(internal[0].ModelsByEngine, tc.attribution) || !reflect.DeepEqual(internal[0].Models, []string{"good"}) {
				t.Fatal("scanner IPC or internal relay lost attribution or valid flat models")
			}
			var wire bytes.Buffer
			codec := NewCodec(&wire)
			broker := &Broker{codec: codec, store: store}
			assertClient := func(t *testing.T, raw json.RawMessage) {
				t.Helper()
				var nodes []AvailableNode
				if err := json.Unmarshal(raw, &nodes); err != nil {
					t.Fatalf("decode ordinary client nodes: %v", err)
				}
				if len(nodes) != 1 || nodes[0].HostUUID != "owner" || nodes[0].ID != "owner" || nodes[0].IPAddress != "127.0.0.1" || !reflect.DeepEqual(nodes[0].Models, []string{"good"}) || !reflect.DeepEqual(nodes[0].ModelsByEngine, tc.attribution) {
					t.Fatal("store/Snapshot/client JSON changed metadata or attribution")
				}
				var fields []map[string]json.RawMessage
				if err := json.Unmarshal(raw, &fields); err != nil {
					t.Fatal(err)
				}
				_, present := fields[0]["modelsByEngine"]
				if present != (tc.attribution != nil) {
					t.Fatal("client wire collapsed nil and present-empty attribution")
				}
			}
			t.Run("get nodes through ordinary dispatcher and codec", func(t *testing.T) {
				id := json.RawMessage(`"304-get"`)
				broker.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: "discovery:get-nodes"})
				response, err := codec.Read()
				if err != nil {
					t.Fatal(err)
				}
				if !response.IsResponse() || response.Error != nil {
					t.Fatal("ordinary get-nodes response failed")
				}
				var result map[string]json.RawMessage
				if err := json.Unmarshal(response.Result, &result); err != nil {
					t.Fatal(err)
				}
				assertClient(t, result["nodes"])
			})
			t.Run("subscribe through ordinary dispatcher and codec", func(t *testing.T) {
				id := json.RawMessage(`"304-subscribe"`)
				broker.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: "discovery:subscribe"})
				ack, err := codec.Read()
				if err != nil {
					t.Fatal(err)
				}
				if !ack.IsResponse() || ack.Error != nil {
					t.Fatal("ordinary subscribe acknowledgement failed")
				}
				changed, err := codec.Read()
				if err != nil {
					t.Fatal(err)
				}
				if changed.Method != "discovery:nodes-changed" || !changed.IsNotification() {
					t.Fatal("ordinary subscribed snapshot was not delivered")
				}
				assertClient(t, changed.Params)
			})
		})
	}
}

// TestDiscoveryStoreRekeysOnRename verifies that a discovered node changing
// hostname (same host UUID, new mDNS instance name) updates its single
// store entry in place rather than leaving a ghost under the old name.
func TestDiscoveryStoreRekeysOnRename(t *testing.T) {
	s := newDiscoveryStore()
	const uuid = "11111111-1111-1111-1111-111111111111"

	s.Upsert(EnrichedNode{ID: "old-host", HostUUID: uuid}, sourceScanner)
	s.Upsert(EnrichedNode{ID: "new-host", HostUUID: uuid}, sourceScanner)

	snap := s.Snapshot()
	if len(snap) != 1 {
		t.Fatalf("a rename must not duplicate the node: got %d entries, want 1", len(snap))
	}
	if snap[0].ID != "new-host" {
		t.Fatalf("store did not track the new name: got id=%q want %q", snap[0].ID, "new-host")
	}
	if snap[0].HostUUID != uuid {
		t.Fatalf("hostUuid not projected to the wire: got %q want %q", snap[0].HostUUID, uuid)
	}
}

// TestDiscoveryStoreDistinctUUIDsSameName verifies the same-hostname collision
// case: two different machines that share a hostname stay two entries.
func TestDiscoveryStoreDistinctUUIDsSameName(t *testing.T) {
	s := newDiscoveryStore()
	s.Upsert(EnrichedNode{ID: "samename", HostUUID: "aaaaaaaa-0000-0000-0000-000000000001"}, sourceScanner)
	s.Upsert(EnrichedNode{ID: "samename", HostUUID: "bbbbbbbb-0000-0000-0000-000000000002"}, sourceScanner)
	if got := len(s.Snapshot()); got != 2 {
		t.Fatalf("two machines sharing a hostname must not merge: got %d entries, want 2", got)
	}
}

// TestManualToEnrichedHostUUID verifies the manual-node ingestion boundary: a
// manual node keys by the remote's real hostUuid once node-info reports one, and
// by its manual id until then.
func TestManualToEnrichedHostUUID(t *testing.T) {
	withUUID := manualToEnriched(manualNodeStatus{ID: "manual:10.0.0.9", HostUUID: "real-uuid"})
	if withUUID.storeKey() != "real-uuid" {
		t.Fatalf("store key = %q, want the real hostUuid", withUUID.storeKey())
	}
	noUUID := manualToEnriched(manualNodeStatus{ID: "manual:10.0.0.9"})
	if noUUID.storeKey() != "manual:10.0.0.9" {
		t.Fatalf("store key = %q, want the manual id fallback", noUUID.storeKey())
	}
}

// TestDiscoveryStoreRejectsEmptyKey verifies a node with no operational key is
// dropped rather than silently keyed by name.
func TestDiscoveryStoreRejectsEmptyKey(t *testing.T) {
	s := newDiscoveryStore()
	s.Upsert(EnrichedNode{ID: "no-uuid"}, sourceManual) // HostUUID empty
	if got := len(s.Snapshot()); got != 0 {
		t.Fatalf("a node with no hostUuid must be dropped, got %d entries", got)
	}
}

// TestDiscoveryStoreSourceOwnership verifies that a manual node and an mDNS
// node sharing a hostUuid occupy one record with two claims. Removing
// either source keeps the record alive under the survivor; only when no source
// claims it is it forgotten. This is what stops a manual remove (or a
// manual-nodes crash) from evicting a still-live scanner node.
func TestDiscoveryStoreSourceOwnership(t *testing.T) {
	const uuid = "shared-uuid"

	// Manual-remove overlap: scanner + manual both claim uuid; removing the
	// manual claim must leave the scanner's node in the snapshot.
	s := newDiscoveryStore()
	s.Upsert(EnrichedNode{ID: "host", HostUUID: uuid, Port: 14318}, sourceScanner)
	s.Upsert(EnrichedNode{ID: "host", HostUUID: uuid, Port: 14318}, sourceManual)
	if got := len(s.Snapshot()); got != 1 {
		t.Fatalf("shared uuid should be one record, got %d", got)
	}
	s.Remove(uuid, sourceManual)
	if got := len(s.Snapshot()); got != 1 {
		t.Fatalf("removing the manual claim evicted the live scanner node (got %d entries)", got)
	}
	s.Remove(uuid, sourceScanner)
	if got := len(s.Snapshot()); got != 0 {
		t.Fatalf("record should be gone once no source claims it, got %d", got)
	}

	// Symmetric: removing the scanner claim leaves the manual node.
	s = newDiscoveryStore()
	s.Upsert(EnrichedNode{ID: "host", HostUUID: uuid}, sourceScanner)
	s.Upsert(EnrichedNode{ID: "host", HostUUID: uuid}, sourceManual)
	s.Remove(uuid, sourceScanner)
	if got := len(s.Snapshot()); got != 1 {
		t.Fatalf("removing the scanner claim evicted the manual node (got %d entries)", got)
	}
}

// TestDiscoveryStoreScannerProjectionWins: when both sources claim a node, the
// scanner (mDNS) view is projected — it carries trusted/clusterUuid and
// per-engine models the manual probe lacks.
func TestDiscoveryStoreScannerProjectionWins(t *testing.T) {
	const uuid = "shared-uuid"
	s := newDiscoveryStore()
	s.Upsert(EnrichedNode{ID: "manual-host", HostUUID: uuid, Trusted: false}, sourceManual)
	s.Upsert(EnrichedNode{ID: "mdns-host", HostUUID: uuid, Trusted: true}, sourceScanner)
	snap := s.Snapshot()
	if len(snap) != 1 {
		t.Fatalf("want one record, got %d", len(snap))
	}
	if snap[0].ID != "mdns-host" || !snap[0].Trusted {
		t.Fatalf("scanner projection should win when both claim: %+v", snap[0])
	}
	// Drop the scanner claim: the manual projection takes over.
	s.Remove(uuid, sourceScanner)
	snap = s.Snapshot()
	if len(snap) != 1 || snap[0].ID != "manual-host" {
		t.Fatalf("manual projection should surface after scanner removal: %+v", snap)
	}
}

// TestDiscoveryRelayCarriesPolicyAdvertisement covers the broker hop of the
// roles advertisement: the scanner's DirectoryNode.Policy reaches the client
// node list unchanged, and a node without one carries no policy key.
func TestDiscoveryRelayCarriesPolicyAdvertisement(t *testing.T) {
	adv := &nodepolicy.Advertisement{
		Revision: "7",
		Roles:    map[string]nodepolicy.AdvertisedRole{nodepolicy.TierWeak: {Engine: "ollama", Model: "qwen3:4b"}},
		PausedBy: nodepolicy.PausedByPeer,
	}
	s := newDiscoveryStore()
	s.Upsert(directoryToEnriched(noderec.DirectoryNode{HostUUID: "declared", Name: "declared", Policy: adv}), sourceScanner)
	s.Upsert(directoryToEnriched(noderec.DirectoryNode{HostUUID: "undeclared", Name: "undeclared"}), sourceScanner)
	byID := map[string]AvailableNode{}
	for _, n := range s.Snapshot() {
		byID[n.HostUUID] = n
	}
	if got := byID["declared"].Policy; !reflect.DeepEqual(got, adv) {
		t.Fatalf("declared node policy = %+v, want %+v", got, adv)
	}
	raw, err := json.Marshal(byID["undeclared"])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), `"policy"`) {
		t.Fatalf("undeclared node carries a policy key: %s", raw)
	}
}

func newManualTestBroker() *Broker {
	return &Broker{
		manualNodeKeys:     make(map[string]string),
		manualNodeStatuses: make(map[string]manualNodeStatusEntry),
		store:              newDiscoveryStore(),
		telemetry:          newTelemetryCache(),
	}
}

func manualStatus(id, addr, uuid string) manualNodeStatus {
	return manualNodeStatus{ID: id, Address: addr, HostUUID: uuid, NodeInfoPort: 14318}
}

// TestManualAliasesShareKeyUntilLastRemoved covers both removal orders: two
// manual entries (distinct names/addresses for one machine) resolve
// to one HostUUID and share a single sourceManual slot. Removing one alias must
// keep the node — reprojected from the surviving alias's payload so the removed
// alias's address doesn't linger — and only the last removal releases the key.
func TestManualAliasesShareKeyUntilLastRemoved(t *testing.T) {
	const uuid = "shared-uuid"
	for _, tc := range []struct {
		name         string
		removeFirst  string
		survivorID   string
		survivorAddr string
	}{
		{"remove-A-first", "alias-a", "alias-b", "10.0.0.2"},
		{"remove-B-first", "alias-b", "alias-a", "10.0.0.1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := newManualTestBroker()
			b.upsertManualNode(manualStatus("alias-a", "10.0.0.1", uuid))
			b.upsertManualNode(manualStatus("alias-b", "10.0.0.2", uuid))
			if got := len(b.store.Snapshot()); got != 1 {
				t.Fatalf("two aliases for one machine should be one record, got %d", got)
			}

			b.removeManualNode(tc.removeFirst)
			snap := b.store.Snapshot()
			if len(snap) != 1 {
				t.Fatalf("removing one of two aliases evicted the shared node (got %d)", len(snap))
			}
			// The surviving alias must be reprojected: its id and address, not
			// the removed alias's stale payload.
			if snap[0].ID != tc.survivorID || snap[0].IPAddress != tc.survivorAddr {
				t.Fatalf("survivor not reprojected: got id=%q ip=%q, want id=%q ip=%q",
					snap[0].ID, snap[0].IPAddress, tc.survivorID, tc.survivorAddr)
			}

			b.removeManualNode(tc.survivorID)
			if got := len(b.store.Snapshot()); got != 0 {
				t.Fatalf("record should be gone once the last alias left, got %d", got)
			}
		})
	}
}

// TestManualRekeyReprojectsSharedOldKey: when one of two aliases sharing a key
// rekeys onto its own distinct UUID (host replacement), the shared old key must
// stay — reprojected from the alias that still owns it — alongside the rekeyed
// alias's new record.
func TestManualRekeyReprojectsSharedOldKey(t *testing.T) {
	b := newManualTestBroker()
	const shared = "shared-uuid"

	b.upsertManualNode(manualStatus("alias-a", "10.0.0.1", shared))
	b.upsertManualNode(manualStatus("alias-b", "10.0.0.2", shared))

	// alias-a rekeys onto its own UUID; alias-b still owns the shared key.
	b.upsertManualNode(manualStatus("alias-a", "10.0.0.1", "alias-a-uuid"))

	snap := b.store.Snapshot()
	if len(snap) != 2 {
		t.Fatalf("want two records (shared survivor + rekeyed alias), got %d", len(snap))
	}
	// The shared key must now project alias-b (the surviving owner).
	var sharedNode *AvailableNode
	for i := range snap {
		if snap[i].HostUUID == shared {
			sharedNode = &snap[i]
		}
	}
	if sharedNode == nil {
		t.Fatalf("shared key %q dropped after a co-owner rekeyed away", shared)
	}
	if sharedNode.ID != "alias-b" || sharedNode.IPAddress != "10.0.0.2" {
		t.Fatalf("shared key not reprojected from the surviving alias: %+v", *sharedNode)
	}
}

// TestDiscoveryStoreRemoveWrongSourceNoop: removing a source that never claimed
// the record leaves it (and the other source) intact.
func TestDiscoveryStoreRemoveWrongSourceNoop(t *testing.T) {
	const uuid = "u"
	s := newDiscoveryStore()
	s.Upsert(EnrichedNode{ID: "host", HostUUID: uuid}, sourceScanner)
	if s.Remove(uuid, sourceManual) { // manual never claimed it
		t.Fatal("removing an unowned source must report false (nothing removed)")
	}
	if got := len(s.Snapshot()); got != 1 {
		t.Fatalf("removing an unowned source must not drop the record, got %d", got)
	}
}

// TestDiscoveryStoreRemoveReportsFinalClaim: Remove reports whether it dropped
// the FINAL claim, so the node-loss sweep only fires when a node is truly gone.
func TestDiscoveryStoreRemoveReportsFinalClaim(t *testing.T) {
	const uuid = "shared-uuid"
	s := newDiscoveryStore()
	s.Upsert(EnrichedNode{ID: "host", HostUUID: uuid}, sourceScanner)
	s.Upsert(EnrichedNode{ID: "host", HostUUID: uuid}, sourceManual)

	if s.Remove(uuid, sourceScanner) {
		t.Fatal("removing one of two claims must report false (node still present)")
	}
	if s.Remove(uuid, sourceManual) == false {
		t.Fatal("removing the surviving claim must report true (node now gone)")
	}
	if s.Remove(uuid, sourceScanner) {
		t.Fatal("removing from an absent record must report false")
	}
}

// nodeRemovedFrame builds the params of a discovery:node-removed for a node with
// the given display name and stable host UUID.
func nodeRemovedFrame(t *testing.T, name, uuid string) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(noderec.NodeEvent{Node: noderec.DirectoryNode{Name: name, HostUUID: uuid}})
	if err != nil {
		t.Fatalf("marshal node event: %v", err)
	}
	return b
}

// TestHandleNotifyNodeLostOnlyOnFinalClaim: a dual-source node (mDNS + manual)
// whose scanner claim drops is still present via the manual claim, so the
// node-loss sweep must NOT fire — otherwise a still-reachable node's jobs get
// wrongly failed.
func TestHandleNotifyNodeLostOnlyOnFinalClaim(t *testing.T) {
	const uuid = "peer-uuid"
	store := newDiscoveryStore()
	store.Upsert(EnrichedNode{ID: "peer-friendly-name", HostUUID: uuid}, sourceScanner)
	store.Upsert(EnrichedNode{ID: "peer-friendly-name", HostUUID: uuid}, sourceManual)

	var calls int
	sp := &scannerProcess{store: store, onNodeLost: func(string, string) { calls++ }}

	sp.handleNotify(noderec.NotifyNodeRemoved, nodeRemovedFrame(t, "peer-friendly-name", uuid))

	if calls != 0 {
		t.Fatalf("onNodeLost fired %d time(s) for a node still held by a manual claim, want 0", calls)
	}
	if got := len(store.Snapshot()); got != 1 {
		t.Fatalf("dual-source node should remain after one source leaves, got %d entries", got)
	}
}

// TestHandleNotifyNodeLostUsesHostUUID: removing a node's final claim fires the
// sweep with the node's stable HostUUID (what workloads are keyed by), not its
// display hostname — the name != uuid case is exactly the bug.
func TestHandleNotifyNodeLostUsesHostUUID(t *testing.T) {
	const uuid = "peer-uuid"
	const name = "peer-friendly-name"
	store := newDiscoveryStore()
	store.Upsert(EnrichedNode{ID: name, HostUUID: uuid}, sourceScanner)

	var gotUUID, gotName string
	var calls int
	sp := &scannerProcess{store: store, onNodeLost: func(u, n string) { calls++; gotUUID = u; gotName = n }}

	sp.handleNotify(noderec.NotifyNodeRemoved, nodeRemovedFrame(t, name, uuid))

	if calls != 1 {
		t.Fatalf("onNodeLost calls = %d, want 1 (final claim removed)", calls)
	}
	if gotUUID != uuid {
		t.Errorf("onNodeLost uuid = %q, want %q (must be HostUUID, not the hostname)", gotUUID, uuid)
	}
	if gotName != name {
		t.Errorf("onNodeLost name = %q, want %q", gotName, name)
	}
}

func TestMemoryTelemetryScannerGenerationOwner(t *testing.T) {
	b := &Broker{nodeID: "self", telemetry: newTelemetryCache()}
	b.telemetry.clock = func() (uint64, error) { return 1000000000, nil }
	s := &scannerProcess{}
	b.bindScannerSource(s)
	emit := func(method string, value interface{}) {
		t.Helper()
		raw, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		s.handleNotify(method, raw)
	}
	generation := "1"
	node := noderec.DirectoryNode{HostUUID: "self", Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: 12345}}}
	emit(noderec.NotifyNodeDiscovered, noderec.NodeEvent{Node: node, MemoryGeneration: &generation})
	sample := noderec.NodeTelemetry{HostUUID: "self", Memory: memoryTelemetryFixture(), MemoryHop: noderec.NewMemoryHop("worker", b.telemetry.clock), MemoryRequest: &noderec.MemoryRequestContext{Generation: "1", RequestBootNS: "1000000000", VerifiedSelfLoopback: true}}
	now := time.Now().UTC()
	sample.Memory.Domains[0].Total.SampledAt = now
	sample.Memory.Domains[0].Free.SampledAt = now
	emit(noderec.NotifyNodeTelemetry, sample)
	if got := b.lookupLocalMemory("self", "producer", time.Now().UTC()); !got.TargetLocal {
		t.Fatalf("current owner refused: %+v", got)
	}
	emit(noderec.NotifyNodeUpdated, noderec.NodeEvent{Node: node, MemoryGeneration: &generation})
	if got := b.lookupLocalMemory("self", "producer", time.Now().UTC()); !got.TargetLocal {
		t.Fatalf("unchanged directory withdrew authority: %+v", got)
	}
	generation = "2"
	emit(noderec.NotifyNodeUpdated, noderec.NodeEvent{Node: node, MemoryGeneration: &generation})
	emit(noderec.NotifyNodeTelemetry, sample)
	if got := b.lookupLocalMemory("self", "producer", now); got.TargetLocal {
		t.Fatal("old generation authorized")
	}
	sample.MemoryRequest.Generation = generation
	emit(noderec.NotifyNodeTelemetry, sample)
	b.bindScannerSource(&scannerProcess{})
	emit(noderec.NotifyNodeTelemetry, sample)
	if got := b.lookupLocalMemory("self", "producer", now); got.TargetLocal {
		t.Fatal("retired scanner authorized")
	}
}

func TestScannerProcessRoutesAndRemovesTelemetry(t *testing.T) {
	want := noderec.NodeTelemetry{
		HostUUID:          "peer-uuid",
		GPUUtilizationPct: 84,
		TelemetryValid:    true,
		MSSince:           137,
	}
	params, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("marshal telemetry: %v", err)
	}

	var got noderec.NodeTelemetry
	var removed string
	sp := &scannerProcess{
		onTelemetry:        func(value noderec.NodeTelemetry) { got = value },
		onTelemetryRemoved: func(hostUUID string) { removed = hostUUID },
	}
	sp.handleNotify(noderec.NotifyNodeTelemetry, params)
	if got.HostUUID != want.HostUUID || !got.TelemetryValid || got.MSSince != want.MSSince {
		t.Fatalf("routed telemetry = %+v, want %+v", got, want)
	}
	if got.GPUUtilizationPct != 84 {
		t.Fatalf("routed utilization = %d, want 84", got.GPUUtilizationPct)
	}

	sp.handleNotify(noderec.NotifyNodeRemoved, nodeRemovedFrame(t, "peer", "peer-uuid"))
	if removed != "peer-uuid" {
		t.Fatalf("removed telemetry host = %q, want peer-uuid", removed)
	}
}

// A scanner errors:report / errors:clear reaches nvpair-errors the same way a
// supervised worker's does: with the broker's nodeId and a timestamp stamped on
// the report, and the broker's attribution on the clear.
func TestScannerErrorsNotificationsReachTheErrorsPipeline(t *testing.T) {
	b := &Broker{nodeID: "local-node"}
	seen := observeErrors(t, b)
	sp := &scannerProcess{onErrorsNotif: b.dispatchScannerErrorsNotif}

	const id = "node-scanner:node-info-unreachable:peer-uuid"
	sp.handleNotify(methodErrorsReport, json.RawMessage(`{"id":"`+id+`","message":"peer is quiet","severity":"warning","action":"none"}`))
	report := awaitErrorsNotification(t, seen, methodErrorsReport)
	var got errors.ServiceError
	if err := json.Unmarshal(report.Params, &got); err != nil {
		t.Fatal(err)
	}
	if got.ID != id || got.NodeID != "local-node" || got.Timestamp == 0 || got.Severity != "warning" {
		t.Fatalf("forwarded report = %+v", got)
	}

	sp.handleNotify(methodErrorsClear, json.RawMessage(`{"id":"`+id+`"}`))
	clearMsg := awaitErrorsNotification(t, seen, methodErrorsClear)
	var cleared errors.ClearParams
	if err := json.Unmarshal(clearMsg.Params, &cleared); err != nil {
		t.Fatal(err)
	}
	if cleared.ID != id || cleared.ClearedBy != "local-node" {
		t.Fatalf("forwarded clear = %+v", cleared)
	}
}

// A scanner that dies takes its record of what it reported with it, so the
// broker withdraws the errors it forwarded: the ones still standing, not the
// ones the scanner already cleared.
func TestScannerCrashClearsTheErrorsItReported(t *testing.T) {
	b := &Broker{nodeID: "local-node"}
	seen := observeErrors(t, b)
	sp := &scannerProcess{onErrorsNotif: b.dispatchScannerErrorsNotif}

	const standing, recovered = "node-scanner:node-info-unreachable:a", "node-scanner:node-info-unreachable:b"
	sp.handleNotify(methodErrorsReport, json.RawMessage(`{"id":"`+standing+`","message":"quiet"}`))
	sp.handleNotify(methodErrorsReport, json.RawMessage(`{"id":"`+recovered+`","message":"quiet"}`))
	sp.handleNotify(methodErrorsClear, json.RawMessage(`{"id":"`+recovered+`"}`))
	for range 3 {
		select {
		case <-seen:
		case <-time.After(2 * time.Second):
			t.Fatal("scanner notifications did not reach the errors process")
		}
	}

	b.clearScannerErrors()
	msg := awaitErrorsNotification(t, seen, methodErrorsClear)
	var cleared errors.ClearParams
	if err := json.Unmarshal(msg.Params, &cleared); err != nil {
		t.Fatal(err)
	}
	if cleared.ID != standing || cleared.ClearedBy != "local-node" {
		t.Fatalf("crash clear = %+v, want %s", cleared, standing)
	}
	b.clearScannerErrors()
	select {
	case extra := <-seen:
		t.Fatalf("a second crash cleared again: %+v", extra)
	case <-time.After(gateQuietWindow):
	}
}
