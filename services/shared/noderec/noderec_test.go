// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/buildinfo"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/modeldownload"
	"nvpair-shared/nodepolicy"
)

func TestMemoryDirectoryGenerationStrictDecode(t *testing.T) {
	for _, raw := range []string{
		`{"node":{},"memoryGeneration":"1","memoryGeneration":"2"}`,
		`{"node":{},"MemoryGeneration":"1"}`,
		`{"node":{},"memoryGeneration":null}`,
		`{"node":{},"memoryGeneration":"01"}`,
		`{"node":{},"memoryGeneration":"-1"}`,
		`{"node":{},"memoryGeneration":"18446744073709551616"}`,
		`{"node":{},"memoryGeneration":"1","producerFlag":true}`,
	} {
		t.Run(raw, func(t *testing.T) {
			var event NodeEvent
			if err := json.Unmarshal([]byte(raw), &event); err == nil {
				t.Fatal("accepted malformed owner generation")
			}
		})
	}
}

func TestMemoryRequestContextStrictDecode(t *testing.T) {
	for _, raw := range []string{
		`{}`, `{"generation":"1","requestBootNs":"1"}`,
		`{"generation":"1","requestBootNs":"1","verifiedSelfLoopback":false,"verifiedSelfLoopback":true}`,
		`{"generation":"1","requestBootNs":"1","VerifiedSelfLoopback":true}`,
		`{"generation":"01","requestBootNs":"1","verifiedSelfLoopback":true}`,
		`{"generation":"1","requestBootNs":"-1","verifiedSelfLoopback":true}`,
		`{"generation":"1","requestBootNs":"1","verifiedSelfLoopback":null}`,
		`{"generation":"1","requestBootNs":"1","verifiedSelfLoopback":true,"producerFlag":true}`,
	} {
		t.Run(raw, func(t *testing.T) {
			var value MemoryRequestContext
			if err := json.Unmarshal([]byte(raw), &value); err == nil {
				t.Fatal("accepted malformed scanner context")
			}
		})
	}
	var value MemoryRequestContext
	if err := json.Unmarshal([]byte(`{"generation":"0","requestBootNs":"1","verifiedSelfLoopback":false}`), &value); err != nil {
		t.Fatal(err)
	}
}

func TestMemoryTelemetryEnvelopeAllowsObservationBudget(t *testing.T) {
	observation := `{"runId":"producer","domains":[]}`
	// Whitespace is valid wire JSON and counts against the nested decode budget.
	observation = "{" + strings.Repeat(" ", MaxMemoryObservationBytes-len(observation)) + observation[1:]
	var memory MemoryObservation
	if err := json.Unmarshal([]byte(observation), &memory); err != nil {
		t.Fatalf("observation at its own budget is invalid: %v", err)
	}
	envelope := `{"hostUuid":"self","gpuUtilizationPercent":95,"telemetryValid":true,"msSince":100,"memory":` + observation + `}`
	var value NodeTelemetry
	if err := json.Unmarshal([]byte(envelope), &value); err != nil {
		t.Fatalf("valid observation lost in telemetry envelope: %v", err)
	}
	if value.Memory == nil || value.Memory.RunID != "producer" || value.GPUUtilizationPct != 95 || !value.TelemetryValid {
		t.Fatal("envelope round trip lost memory or utilization")
	}
	oversized := strings.Replace(envelope, observation, "{ "+observation[1:], 1)
	if json.Unmarshal([]byte(oversized), &value) == nil {
		t.Fatal("outer envelope bypassed the nested observation budget")
	}
}

func TestMemoryTelemetryEnvelopeStrictDecode(t *testing.T) {
	valid := `{"hostUuid":"self","gpuUtilizationPercent":95,"telemetryValid":true,"msSince":100}`
	for _, tc := range []struct{ name, raw string }{
		{"missing validity", strings.Replace(valid, `,"telemetryValid":true`, "", 1)},
		{"null age", strings.Replace(valid, `"msSince":100`, `"msSince":null`, 1)},
		{"duplicate age", strings.Replace(valid, `"msSince":100`, `"msSince":100,"msSince":0`, 1)},
		{"case alias", strings.Replace(valid, `"msSince":100`, `"MSSince":0`, 1)},
		{"escaped duplicate", strings.Replace(valid, `"msSince":100`, `"msSince":100,"ms\u0053ince":0`, 1)},
		{"unknown flag", strings.Replace(valid, `"msSince":100`, `"msSince":100,"producerFlag":true`, 1)},
		{"trailing value", valid + "{}"},
		{"oversized envelope", "{" + strings.Repeat(" ", MaxMemoryObservationBytes+4096) + valid[1:]},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var value NodeTelemetry
			if json.Unmarshal([]byte(tc.raw), &value) == nil {
				t.Fatal("accepted ambiguous telemetry envelope")
			}
		})
	}
}

func TestEngineBindingOffersDirectoryWire(t *testing.T) {
	node := DirectoryNode{HostUUID: "fixture", Models: []string{}, ModelsByEngine: map[string][]string{"lmstudio": {}},
		Policy: &nodepolicy.Advertisement{Revision: "1", EngineBindingOffers: &nodepolicy.EngineBindingOffers{Version: 1, Models: []nodepolicy.Role{{Engine: "lmstudio", Model: "cold-model"}}}}}
	raw, err := json.Marshal(node)
	if err != nil {
		t.Fatal(err)
	}
	var got DirectoryNode
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Policy, node.Policy) || len(got.Models) != 0 || len(got.EngineModels("lmstudio")) != 0 || len(got.LoadedByEngine) != 0 {
		t.Fatalf("offers became installed inventory: %+v", got)
	}
}

func TestModelDownloadDirectoryWire(t *testing.T) {
	percent := 21
	snapshot := &modeldownload.Snapshot{
		Version: modeldownload.Version, SourceID: strings.Repeat("a", 32), Revision: "17", RecoveryRequired: true,
		Entries: []modeldownload.Entry{{ID: strings.Repeat("b", 32), Engine: "llamacpp", Model: "org/model", Phase: modeldownload.PhaseUnknown,
			BlocksReadiness: true, Percent: &percent, Reason: modeldownload.ReasonInterrupted, Persistence: modeldownload.PersistenceUncertain}},
	}
	test := func(name string, state *modeldownload.Snapshot, current *bool) {
		t.Run(name, func(t *testing.T) {
			node := DirectoryNode{HostUUID: "target", Models: []string{"ready"}, ModelsByEngine: map[string][]string{"llamacpp": {"ready"}},
				LoadedByEngine: map[string][]string{"llamacpp": {"org/model"}}, DownloadState: state, DownloadStateCurrent: current}
			raw, err := json.Marshal(node)
			if err != nil {
				t.Fatalf("encode directory: %v", err)
			}
			var got DirectoryNode
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatalf("decode directory: %v", err)
			}
			if !reflect.DeepEqual(got, node) || !reflect.DeepEqual(got.EngineModels("llamacpp"), []string{"ready"}) {
				t.Fatalf("operation changed inventory, residency or fields: got %+v, want %+v", got, node)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(raw, &fields); err != nil {
				t.Fatalf("decode directory fields: %v", err)
			}
			if (fields["downloadState"] != nil) != (state != nil) || (fields["downloadStateCurrent"] != nil) != (current != nil) {
				t.Fatal("unknown support and explicit freshness were conflated")
			}
		})
	}
	current, stale := true, false
	test("current unsafe and independently loaded", snapshot, &current)
	test("stale details remain visible", snapshot, &stale)
	empty := snapshot.Clone()
	empty.Entries = []modeldownload.Entry{}
	test("explicit empty", &empty, &current)
	test("unsupported", nil, nil)
	test("invalid observation", nil, &stale)
}

func TestModelControlsDirectoryAttributionWire(t *testing.T) {
	for _, tc := range []struct {
		name, raw   string
		attribution map[string][]string
		ollama      []string
	}{
		{"nil retains flat fallback", `{"hostUuid":"owner","models":["good"]}`, nil, []string{"good"}},
		{"outer empty suppresses flat fallback", `{"hostUuid":"owner","models":["good"],"modelsByEngine":{}}`, map[string][]string{}, nil},
		{"per engine known empty", `{"hostUuid":"owner","models":["good"],"modelsByEngine":{"ollama":[]}}`, map[string][]string{"ollama": {}}, []string{}},
		{"populated siblings retain attribution", `{"hostUuid":"owner","models":["good"],"modelsByEngine":{"ollama":["first","last"],"lmstudio":["other"]}}`, map[string][]string{"ollama": {"first", "last"}, "lmstudio": {"other"}}, []string{"first", "last"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var node DirectoryNode
			if err := json.Unmarshal([]byte(tc.raw), &node); err != nil {
				t.Fatalf("decode raw inventory specimen: %v", err)
			}
			node.Name, node.IP, node.Trusted, node.LastSeen = "name", "127.0.0.1", true, 304
			node.Services = map[ServiceKey]ServiceStatus{ServiceOllama: {Port: 24001}}
			assertNode := func(t *testing.T, got DirectoryNode) {
				t.Helper()
				if !reflect.DeepEqual(got.ModelsByEngine, tc.attribution) || !reflect.DeepEqual(got.EngineModels("ollama"), tc.ollama) || !reflect.DeepEqual(got.Models, []string{"good"}) {
					t.Fatal("wire round-trip changed attribution, fallback, order or valid flat inventory")
				}
				if got.HostUUID != "owner" || got.Name != "name" || got.IP != "127.0.0.1" || !got.Trusted || got.LastSeen != 304 || !reflect.DeepEqual(got.Services, node.Services) {
					t.Fatal("wire round-trip changed unrelated node metadata")
				}
			}
			t.Run("direct JSON", func(t *testing.T) {
				encoded, err := json.Marshal(node)
				if err != nil {
					t.Fatal(err)
				}
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(encoded, &fields); err != nil {
					t.Fatal(err)
				}
				_, present := fields["modelsByEngine"]
				if present != (tc.attribution != nil) {
					t.Fatal("attribution omission does not match nil state")
				}
				var decoded DirectoryNode
				if err := json.Unmarshal(encoded, &decoded); err != nil {
					t.Fatal(err)
				}
				assertNode(t, decoded)
			})
			t.Run("NodeEvent notification", func(t *testing.T) {
				var wire bytes.Buffer
				codec := jsonrpc.NewCodec(&wire)
				if err := codec.Notify(NotifyNodeUpdated, NodeEvent{Node: node}); err != nil {
					t.Fatal(err)
				}
				frame, err := codec.Read()
				if err != nil {
					t.Fatal(err)
				}
				if frame.Method != NotifyNodeUpdated || !frame.IsNotification() {
					t.Fatal("node event envelope changed")
				}
				var event NodeEvent
				if err := json.Unmarshal(frame.Params, &event); err != nil {
					t.Fatal(err)
				}
				assertNode(t, event.Node)
			})
			t.Run("subscription snapshot notification", func(t *testing.T) {
				var wire bytes.Buffer
				codec := jsonrpc.NewCodec(&wire)
				if err := codec.Notify(NotifyNodes, GetNodesResult{Nodes: []DirectoryNode{node}}); err != nil {
					t.Fatal(err)
				}
				frame, err := codec.Read()
				if err != nil {
					t.Fatal(err)
				}
				var snapshot GetNodesResult
				if err := json.Unmarshal(frame.Params, &snapshot); err != nil {
					t.Fatalf("decode subscription snapshot: %v", err)
				}
				if frame.Method != NotifyNodes || len(snapshot.Nodes) != 1 {
					t.Fatal("subscription snapshot envelope changed")
				}
				assertNode(t, snapshot.Nodes[0])
			})
			t.Run("get nodes response", func(t *testing.T) {
				var wire bytes.Buffer
				codec := jsonrpc.NewCodec(&wire)
				id := json.RawMessage(`"304"`)
				if err := codec.Respond(&id, GetNodesResult{Nodes: []DirectoryNode{node}}); err != nil {
					t.Fatal(err)
				}
				frame, err := codec.Read()
				if err != nil {
					t.Fatal(err)
				}
				var snapshot GetNodesResult
				if err := json.Unmarshal(frame.Result, &snapshot); err != nil {
					t.Fatalf("decode get-nodes response: %v", err)
				}
				if !frame.IsResponse() || frame.Error != nil || len(snapshot.Nodes) != 1 {
					t.Fatal("get-nodes response envelope changed")
				}
				assertNode(t, snapshot.Nodes[0])
			})
		})
	}
}

func TestDirectoryOptionalServiceBuild(t *testing.T) {
	clean := "git:" + strings.Repeat("a", 40)
	dirty := "git:" + strings.Repeat("b", 64) + "+dirty"
	for _, tc := range []struct {
		name, value string
		want        buildinfo.ID
	}{
		{"clean", `"` + clean + `"`, buildinfo.ID(clean)},
		{"dirty sha256", `"` + dirty + `"`, buildinfo.ID(dirty)},
		{"absent", "", ""},
		{"null", "null", ""},
		{"number", "42", ""},
		{"boolean", "true", ""},
		{"object", `{ "revision": "secret" }`, ""},
		{"array", `[]`, ""},
		{"malformed", `"git:short"`, ""},
		{"uppercase", `"git:` + strings.Repeat("A", 40) + `"`, ""},
		{"oversize", `"git:` + strings.Repeat("a", 75) + `"`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := `{"hostUuid":"host-a","gpus":[{"name":"GPU"}],"cpu":{"cores":8},"memory":{"total_bytes":16384},"models":["m"]`
			if tc.value != "" {
				raw += `,"serviceBuildId":` + tc.value
			}
			raw += `}`
			var got DirectoryNode
			if err := json.Unmarshal([]byte(raw), &got); err != nil {
				t.Fatalf("decode optional build: %v", err)
			}
			if got.ServiceBuildID != tc.want || got.HostUUID != "host-a" || len(got.GPUs) != 1 || got.GPUs[0].Name != "GPU" || got.CPU == nil || got.CPU.Cores != 8 || got.Memory == nil || got.Memory.TotalBytes != 16384 || !reflect.DeepEqual(got.Models, []string{"m"}) {
				t.Fatalf("node = %+v, want build %q and intact identity/inventory", got, tc.want)
			}
			data, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("marshal node: %v", err)
			}
			var wire map[string]json.RawMessage
			if err := json.Unmarshal(data, &wire); err != nil {
				t.Fatalf("decode wire: %v", err)
			}
			_, present := wire["serviceBuildId"]
			if present != (tc.want != "") {
				t.Fatalf("wire build presence = %v, want %v", present, tc.want != "")
			}
		})
	}
}

func TestServedResidentDirectoryRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name   string
		served nodepolicy.ServedModels
	}{
		{name: "absent"}, {name: "empty", served: nodepolicy.ServedModels{}},
		{name: "restricted", served: nodepolicy.ServedModels{{Engine: "ollama", Model: "m"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := nodepolicy.Default()
			p.ServedModels = tc.served
			adv := p.Advertisement()
			node := DirectoryNode{Policy: &adv}
			for range 3 {
				raw, err := json.Marshal(node)
				if err != nil {
					t.Fatal(err)
				}
				var next DirectoryNode
				if err := json.Unmarshal(raw, &next); err != nil {
					t.Fatal(err)
				}
				if next.Policy == nil || !reflect.DeepEqual(next.Policy.ServedModels, tc.served) {
					t.Fatal("directory lost served presence or references")
				}
				node = next
			}
		})
	}
}

func TestEngineModels(t *testing.T) {
	// Attribution present: each engine gets exactly its own list, never the union.
	attributed := DirectoryNode{
		Models: []string{"a", "b", "c"},
		ModelsByEngine: map[string][]string{
			"ollama":   {"a", "b"},
			"lmstudio": {"c"},
		},
	}
	if got := attributed.EngineModels("ollama"); !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Errorf("EngineModels(ollama) = %v, want [a b]", got)
	}
	if got := attributed.EngineModels("lmstudio"); !reflect.DeepEqual(got, []string{"c"}) {
		t.Errorf("EngineModels(lmstudio) = %v, want [c]", got)
	}

	// Attribution present but this engine has no entry: authoritatively empty —
	// NOT the cross-engine union (the whole point of per-engine attribution).
	if got := attributed.EngineModels("llamacpp"); len(got) != 0 {
		t.Errorf("EngineModels(missing engine, attribution present) = %v, want empty", got)
	}

	// No attribution at all (pre-attribution / mixed-version peer): fall back to
	// the flat union so a single-engine consumer doesn't regress to no inventory.
	legacy := DirectoryNode{Models: []string{"a", "b", "c"}}
	if got := legacy.EngineModels("ollama"); !reflect.DeepEqual(got, []string{"a", "b", "c"}) {
		t.Errorf("EngineModels with nil ModelsByEngine = %v, want union [a b c]", got)
	}
}

func TestParseTXT(t *testing.T) {
	txt := []string{
		"v=1", "uuid=host-abc", "cluster-uuid=clu-xyz", "ip=192.168.1.10",
		"ni=14318", "ol=11434", "er=14319", "wl=14320", "cl=14321", "em=14322",
		"unknown=ignored", "bad=notaport",
	}
	r := ParseTXT(txt)
	if r.SchemaVersion != "1" || r.HostUUID != "host-abc" || r.ClusterUUID != "clu-xyz" || r.IP != "192.168.1.10" {
		t.Fatalf("scalar fields wrong: %+v", r)
	}
	if !r.Clustered() {
		t.Error("Clustered() = false, want true (cluster-uuid present)")
	}
	if p, ok := r.Port(ServiceNodeInfo); !ok || p != 14318 {
		t.Errorf("ni port = %d,%v want 14318,true", p, ok)
	}
	if p, ok := r.Port(ServiceCluster); !ok || p != 14321 {
		t.Errorf("cl port = %d,%v want 14321,true", p, ok)
	}
	if p, ok := r.Port(ServiceEngineManager); !ok || p != 14322 {
		t.Errorf("em port = %d,%v want 14322,true", p, ok)
	}
	if _, ok := r.Port(ServiceLMStudio); ok {
		t.Error("lm should be absent")
	}
	// "unknown=" is not a service port; "bad=notaport" is skipped.
	if _, ok := r.Services["unknown"]; ok {
		t.Error("unknown key leaked into Services")
	}
	if _, ok := r.Services["bad"]; ok {
		t.Error("malformed port leaked into Services")
	}
}

// TestTXTEmitsUnknownServiceKey guards the forward-compat path: an unknown/future
// service key must be emitted even when the total service count is at or below
// the number of known keys (a count-based guard would silently drop it).
func TestTXTEmitsUnknownServiceKey(t *testing.T) {
	r := NodeRecord{
		SchemaVersion: "1",
		HostUUID:      "host-abc",
		Services:      map[ServiceKey]int{ServiceNodeInfo: 14318, ServiceKey("zz"): 15000},
	}
	txt := r.TXT()
	joined := strings.Join(txt, ";")
	if !strings.Contains(joined, "zz=15000") {
		t.Fatalf("unknown service key dropped from TXT: %v", txt)
	}
	// And it survives a round-trip.
	if p, ok := ParseTXT(txt).Port(ServiceKey("zz")); !ok || p != 15000 {
		t.Errorf("unknown key round-trip = %d,%v want 15000,true", p, ok)
	}
}

func TestTXTRoundTrip(t *testing.T) {
	orig := NodeRecord{
		SchemaVersion: "1",
		HostUUID:      "host-abc",
		ClusterUUID:   "clu-xyz",
		IP:            "10.0.0.5",
		Services:      map[ServiceKey]int{ServiceOllama: 11434, ServiceErrors: 14319, ServiceNodeInfo: 14318, ServiceEngineManager: 14322},
	}
	txt := orig.TXT()
	// Schema must be first.
	if !strings.HasPrefix(txt[0], "v=") {
		t.Errorf("first TXT entry = %q, want schema", txt[0])
	}
	got := ParseTXT(txt)
	if !reflect.DeepEqual(orig, got) {
		t.Fatalf("round-trip mismatch:\n orig=%+v\n got =%+v", orig, got)
	}
}

func TestTXTDeterministicOrder(t *testing.T) {
	r := NodeRecord{
		HostUUID: "h",
		Services: map[ServiceKey]int{ServiceCluster: 14321, ServiceNodeInfo: 14318, ServiceOllama: 11434},
	}
	// Built twice, identical order (map iteration is randomized, so this guards
	// the deterministic emit).
	if !reflect.DeepEqual(r.TXT(), r.TXT()) {
		t.Fatal("TXT() is not deterministic")
	}
	got := r.TXT()
	want := []string{"v=1", "uuid=h", "ni=14318", "ol=11434", "cl=14321"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("TXT order = %v, want %v", got, want)
	}
}

func TestTXTDefaultsSchema(t *testing.T) {
	r := NodeRecord{HostUUID: "h", Services: map[ServiceKey]int{}}
	if got := r.TXT()[0]; got != "v="+SchemaVersion {
		t.Errorf("missing schema not defaulted: %q", got)
	}
}

func TestTransportPolicy(t *testing.T) {
	cases := []struct {
		svc       ServiceKey
		want      Transport
		mtlsClu   bool // UsesMTLS when target clustered
		mtlsUnclu bool // UsesMTLS when target not clustered
	}{
		{ServiceNodeInfo, TransportPlain, false, false},
		{ServiceOllama, TransportPlain, false, false},
		{ServiceLMStudio, TransportPlain, false, false},
		{ServiceLlamaCPP, TransportPlain, false, false},
		{ServiceEngineManager, TransportPlain, false, false},
		{ServiceErrors, TransportMTLSWhenClustered, true, false},
		{ServiceWorkload, TransportMTLSWhenClustered, true, false},
		{ServiceCluster, TransportSplit, true, false},
	}
	for _, c := range cases {
		if got := c.svc.Transport(); got != c.want {
			t.Errorf("%s Transport = %v, want %v", c.svc, got, c.want)
		}
		if got := c.svc.UsesMTLS(true); got != c.mtlsClu {
			t.Errorf("%s UsesMTLS(clustered) = %v, want %v", c.svc, got, c.mtlsClu)
		}
		if got := c.svc.UsesMTLS(false); got != c.mtlsUnclu {
			t.Errorf("%s UsesMTLS(unclustered) = %v, want %v", c.svc, got, c.mtlsUnclu)
		}
	}
}

func TestNodeInfoAlwaysPlainEvenClustered(t *testing.T) {
	// The subtlest correctness requirement: a clustered node must NOT trick a
	// consumer into dialing node-info over mTLS.
	if ServiceNodeInfo.UsesMTLS(true) {
		t.Fatal("node-info must be plain even when the node is clustered")
	}
}

func TestSubscribeMatches(t *testing.T) {
	n := DirectoryNode{Services: map[ServiceKey]ServiceStatus{
		ServiceOllama:   {Port: 11434},
		ServiceNodeInfo: {Port: 14318},
	}}
	// Empty filter matches everything.
	if !(SubscribeParams{}).Matches(n) {
		t.Error("empty subscribe filter should match all nodes")
	}
	// A service the node has.
	if !(SubscribeParams{Services: []ServiceKey{ServiceOllama}}).Matches(n) {
		t.Error("filter for ol should match a node advertising ol")
	}
	// A service the node lacks.
	if (SubscribeParams{Services: []ServiceKey{ServiceErrors}}).Matches(n) {
		t.Error("filter for er should not match a node without er")
	}
	// Any-of semantics: one present, one absent.
	if !(SubscribeParams{Services: []ServiceKey{ServiceErrors, ServiceNodeInfo}}).Matches(n) {
		t.Error("any-of filter should match when one listed service is present")
	}
}

func TestDirectoryNodeHelpers(t *testing.T) {
	n := DirectoryNode{ClusterUUID: "clu", Services: map[ServiceKey]ServiceStatus{ServiceCluster: {Port: 14321}}}
	if !n.Clustered() {
		t.Error("Clustered() should be true with a cluster-uuid")
	}
	if !n.HasService(ServiceCluster) || n.HasService(ServiceOllama) {
		t.Error("HasService wrong")
	}
	if (DirectoryNode{}).Clustered() {
		t.Error("empty node should not be Clustered")
	}
}

func TestDirectoryNodeJSONRoundTrip(t *testing.T) {
	orig := DirectoryNode{
		HostUUID: "host-1", Name: "host", IP: "192.168.1.10", ClusterUUID: "clu",
		Trusted: true,
		Services: map[ServiceKey]ServiceStatus{
			ServiceOllama:   {Port: 11434, Probe: ProbeReachable},
			ServiceErrors:   {Port: 14319, Probe: ProbeInaccessible},
			ServiceNodeInfo: {Port: 14318},
		},
		GPUs:     []GPUInfo{{Name: "RTX", VramBytes: 1 << 30}},
		CPU:      &CPUInfo{Name: "cpu", Cores: 8},
		Memory:   &MemoryInfo{TotalBytes: 1 << 34},
		LastSeen: 1234567890,
	}
	b, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got DirectoryNode
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !reflect.DeepEqual(orig, got) {
		t.Fatalf("round-trip mismatch:\n orig=%+v\n got =%+v", orig, got)
	}
}

func TestDirectoryNodePolicy(t *testing.T) {
	t.Run("absent policy is omitted", func(t *testing.T) {
		b, err := json.Marshal(DirectoryNode{HostUUID: "host-1"})
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		if strings.Contains(string(b), `"policy"`) {
			t.Fatalf("undeclared node carries a policy key: %s", b)
		}
	})
	t.Run("advertisement round-trips", func(t *testing.T) {
		orig := DirectoryNode{HostUUID: "host-1", Policy: &nodepolicy.Advertisement{
			Revision: "42",
			Roles: map[string]nodepolicy.AdvertisedRole{
				nodepolicy.TierStrong: {Engine: "ollama", Model: "qwen3:32b", ContextTokens: 32768, Capabilities: []nodepolicy.Capability{nodepolicy.CapTools}},
			},
			PausedBy: nodepolicy.PausedLocally,
		}}
		b, err := json.Marshal(orig)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var got DirectoryNode
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if !reflect.DeepEqual(orig, got) {
			t.Fatalf("round-trip mismatch:\n orig=%+v\n got =%+v", orig.Policy, got.Policy)
		}
	})
}

func TestValidateTXTSize(t *testing.T) {
	if err := ValidateTXTSize([]string{"v=1", "ni=14318"}); err != nil {
		t.Errorf("small TXT flagged: %v", err)
	}
	big := "x=" + strings.Repeat("m", 300)
	if err := ValidateTXTSize([]string{"v=1", big}); err == nil {
		t.Error("oversized TXT entry not flagged")
	}
}
