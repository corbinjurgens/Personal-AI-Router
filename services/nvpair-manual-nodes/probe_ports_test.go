// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net/http"
	"reflect"
	"testing"
)

func TestProbePortsAddResponseAndDiscoveryDoNotRestoreDisabledDefaults(t *testing.T) {
	m, rw, _ := newTestManager()
	m.probePorts = probePorts{}
	status := m.addNode(ManualEntry{Address: "127.0.0.1", Name: "fixture"})
	if status.OllamaPort != 0 || status.LMStudioPort != 0 || status.NodeInfoPort != 0 {
		t.Fatalf("initial add response restored default targets: %+v", status)
	}
	frame := readCaptureFrame(t, rw)
	if frame.Method != "node/discovered" {
		t.Fatalf("initial event = %s, want node/discovered", frame.Method)
	}
	var discovered ManualNodeStatus
	if err := json.Unmarshal(frame.Params, &discovered); err != nil {
		t.Fatal(err)
	}
	if discovered.OllamaPort != 0 || discovered.LMStudioPort != 0 || discovered.NodeInfoPort != 0 {
		t.Fatalf("discovered status restored default targets: %+v", discovered)
	}
}

func TestProbePortsProductionDefaults(t *testing.T) {
	m, _, _ := newTestManager()
	if want := (probePorts{11434, 1234, 14318}); m.probePorts != want {
		t.Fatalf("default probe ports = %+v, want %+v", m.probePorts, want)
	}
}

func TestProbePortsUseConfiguredTargetsAndReportThem(t *testing.T) {
	m, _, _ := newTestManager()
	m.probePorts = probePorts{ollama: 25101, lmstudio: 25102, nodeInfo: 25103}
	var targets []string
	m.client.Transport = probePortTransport(func(r *http.Request) (*http.Response, error) {
		targets = append(targets, r.URL.Host+r.URL.Path)
		switch r.URL.Path {
		case "/api/tags":
			return httpJSON(http.StatusOK, `{"models":[{"name":"fixture-model"}]}`)
		case "/v1/models":
			return httpJSON(http.StatusOK, `{"data":[{"id":"fixture-model"}]}`)
		case "/v1/node-info":
			return httpJSON(http.StatusOK, `{"hostUuid":"fixture-node"}`)
		default:
			return httpJSON(http.StatusOK, `{}`)
		}
	})
	tn := &trackedNode{entry: ManualEntry{Address: "127.0.0.1"}}
	m.nodes[nodeID(tn.entry)] = tn
	m.probeNode(tn)
	want := []string{"127.0.0.1:25101/", "127.0.0.1:25101/api/tags", "127.0.0.1:25102/v1/models", "127.0.0.1:25103/v1/node-info"}
	if !reflect.DeepEqual(targets, want) {
		t.Fatalf("probe targets = %v, want %v", targets, want)
	}
	if s := tn.status; !s.OllamaUp || !s.LMStudioUp || !s.NodeInfoUp || s.OllamaPort != 25101 || s.LMStudioPort != 25102 || s.NodeInfoPort != 25103 || s.HostUUID != "fixture-node" {
		t.Fatalf("configured probe status = %+v", s)
	}
}

func TestProbePortsZeroDisablesRequestsWithoutDefaultFallback(t *testing.T) {
	m, _, _ := newTestManager()
	m.probePorts = probePorts{}
	requests := 0
	m.client.Transport = probePortTransport(func(*http.Request) (*http.Response, error) {
		requests++
		return httpJSON(http.StatusOK, `{}`)
	})
	tn := &trackedNode{entry: ManualEntry{Address: "127.0.0.1"}}
	m.nodes[nodeID(tn.entry)] = tn
	m.probeNode(tn)
	if requests != 0 || tn.status.OllamaUp || tn.status.LMStudioUp || tn.status.NodeInfoUp || tn.status.OllamaPort != 0 || tn.status.LMStudioPort != 0 || tn.status.NodeInfoPort != 0 {
		t.Fatalf("disabled probes: requests=%d status=%+v", requests, tn.status)
	}
}

func TestProbePortsTLSHintPreservesExplicitHTTPSAndClient(t *testing.T) {
	m, _, _ := newTestManager()
	m.probePorts = probePorts{}
	var targets []string
	m.tlsClient.Transport = probePortTransport(func(r *http.Request) (*http.Response, error) {
		targets = append(targets, r.URL.String())
		return httpJSON(http.StatusOK, `{"hostUuid":"tls-fixture"}`)
	})
	tn := &trackedNode{entry: ManualEntry{Address: "127.0.0.1", TLSPort: 25104, MTLS: true}}
	m.nodes[nodeID(tn.entry)] = tn
	m.probeNode(tn)
	if !reflect.DeepEqual(targets, []string{"https://127.0.0.1:25104/v1/node-info"}) || !tn.status.NodeInfoUp || !tn.status.TLSEnabled || !tn.status.MTLSRequired || tn.status.NodeInfoPort != 25104 {
		t.Fatalf("TLS probe: targets=%v status=%+v", targets, tn.status)
	}
}

func TestProbePortsValidation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		ports probePorts
		valid bool
	}{
		{"production defaults", defaultProbePorts(), true},
		{"disabled", probePorts{}, true},
		{"private", probePorts{25101, 25102, 25103}, true},
		{"negative Ollama", probePorts{-1, 0, 0}, false},
		{"large LM Studio", probePorts{0, 65536, 0}, false},
		{"negative node-info", probePorts{0, 0, -1}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.ports.validate(); (err == nil) != tc.valid {
				t.Fatalf("validate = %v, valid=%t", err, tc.valid)
			}
		})
	}
}

type probePortTransport func(*http.Request) (*http.Response, error)

func (f probePortTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
