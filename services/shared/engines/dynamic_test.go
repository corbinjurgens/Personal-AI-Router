// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

import (
	"slices"
	"strings"
	"testing"

	"nvpair-shared/noderec/servicekey"
)

func TestValidDynamicID(t *testing.T) {
	for _, id := range []string{"llamacpp-custom", "strata", "errors", "e-x", strings.Repeat("a", MaxDynamicIDLen)} {
		if !ValidDynamicID(id) {
			t.Errorf("ValidDynamicID(%q) = false, want true", id)
		}
	}
	rejected := append(Names(), "", ".", "..", "a/b", "a:b", "a b", strings.Repeat("a", MaxDynamicIDLen+1))
	for _, id := range rejected {
		if ValidDynamicID(id) {
			t.Errorf("ValidDynamicID(%q) = true, want false", id)
		}
	}
}

func TestValidRoutingIDAcceptsBuiltInsAndDynamicIDs(t *testing.T) {
	for _, id := range append(Names(), "llamacpp-custom", "strata") {
		if !ValidRoutingID(id) {
			t.Errorf("ValidRoutingID(%q) = false, want true", id)
		}
	}
	for _, id := range []string{"", "a/b", "a:b", strings.Repeat("a", MaxDynamicIDLen+1)} {
		if ValidRoutingID(id) {
			t.Errorf("ValidRoutingID(%q) = true, want false", id)
		}
	}
}

func TestDynamicEngineIdentity(t *testing.T) {
	e, ok := DynamicEngine("llamacpp-custom", "llama.cpp (custom)")
	if !ok {
		t.Fatal("DynamicEngine rejected a valid id")
	}
	want := Engine{Name: "llamacpp-custom", DisplayName: "llama.cpp (custom)", DiscoveryService: "e-llamacpp-custom", PortFile: "llamacpp-custom-proxy-port.json"}
	if e != want {
		t.Fatalf("DynamicEngine = %+v, want %+v", e, want)
	}
	if e, _ := DynamicEngine("strata", " "); e.DisplayName != "strata" {
		t.Fatalf("blank label display name = %q, want the id", e.DisplayName)
	}
	for _, id := range append(Names(), "", "a/b") {
		if _, ok := DynamicEngine(id, "x"); ok {
			t.Errorf("DynamicEngine(%q) accepted", id)
		}
	}
}

// A dynamic key can never collide with a fixed one, and every carrier of a
// key has room for the longest.
func TestDynamicServiceKeys(t *testing.T) {
	for _, key := range []servicekey.Key{servicekey.NodeInfo, servicekey.Ollama, servicekey.LMStudio, servicekey.LlamaCPP,
		servicekey.Errors, servicekey.Workload, servicekey.Cluster, servicekey.EngineManager, servicekey.EngineControl,
		"e-", "e-a/b", "e-ollama", "x-strata", servicekey.Key("e-" + strings.Repeat("a", MaxDynamicIDLen+1))} {
		if IsDynamicServiceKey(key) {
			t.Errorf("IsDynamicServiceKey(%q) = true, want false", key)
		}
	}
	id, ok := DynamicEngineForServiceKey("e-strata")
	if !ok || id != "strata" {
		t.Fatalf("DynamicEngineForServiceKey(e-strata) = %q, %v", id, ok)
	}
	longest, _ := DynamicEngine(strings.Repeat("a", MaxDynamicIDLen), "")
	if header := "PAIR-REVERSE/1 " + string(longest.DiscoveryService) + "\n"; len(header) > 64 {
		t.Fatalf("longest reverse stream header is %d bytes, want at most 64", len(header))
	}
}

func TestDynamicAddressRoundTrip(t *testing.T) {
	for _, id := range []string{"llamacpp-custom", "strata", "errors", "discovery", "workload"} {
		for _, method := range []string{"ready", "nodes/list", "errors:report"} {
			addressed := AddressMethod(id, method)
			if !strings.HasPrefix(addressed, "e-"+id+":") {
				t.Fatalf("AddressMethod(%q, %q) = %q, want the e- address", id, method, addressed)
			}
			engine, bare := SplitAddressedMethod(addressed)
			if engine != id || bare != method {
				t.Errorf("round trip of %q gave (%q, %q), want (%q, %q)", addressed, engine, bare, id, method)
			}
		}
	}
	// A bare dynamic id is not an address: only the e- form is.
	if engine, bare := SplitAddressedMethod("strata:ready"); engine != "" || bare != "strata:ready" {
		t.Fatalf("unprefixed dynamic id split as (%q, %q)", engine, bare)
	}
}

func TestBuiltInAddressesAreUnchanged(t *testing.T) {
	for _, name := range Names() {
		if got := AddressMethod(name, "ready"); got != name+":ready" {
			t.Errorf("AddressMethod(%q) = %q, want %q", name, got, name+":ready")
		}
	}
}

func frontableDescriptor() Descriptor {
	d := validDescriptor()
	d.Protocols = []string{ProtocolLlamaCPPRouter, ProtocolOpenAIV1}
	return d
}

func TestProxyCanFront(t *testing.T) {
	if err := ProxyCanFront(frontableDescriptor()); err != nil {
		t.Fatalf("llama.cpp fork rejected: %v", err)
	}
	for _, protocols := range [][]string{{ProtocolOpenAIChat}, {ProtocolOllamaAPI}, {ProtocolOpenAIV1, ProtocolAnthropicMessages}} {
		d := frontableDescriptor()
		d.Protocols = protocols
		if err := ProxyCanFront(d); err != nil {
			t.Errorf("protocols %v rejected: %v", protocols, err)
		}
	}
	builtin := frontableDescriptor()
	builtin.Engine, builtin.Protocols = NameOllama, nil
	if err := ProxyCanFront(builtin); err != nil {
		t.Fatalf("built-in engine rejected: %v", err)
	}
	cases := []struct {
		name   string
		mutate func(*Descriptor)
		want   string
	}{
		{"unknown protocol", func(d *Descriptor) { d.Protocols = []string{"grpc"} }, "unknown protocol"},
		{"no protocol", func(d *Descriptor) { d.Protocols = nil }, "no protocol"},
		{"model list only", func(d *Descriptor) { d.Protocols = []string{ProtocolLlamaCPPRouter} }, "no protocol"},
		{"messages only", func(d *Descriptor) { d.Protocols = []string{ProtocolAnthropicMessages} }, "no protocol"},
		{"long id", func(d *Descriptor) { d.Engine = strings.Repeat("a", MaxDynamicIDLen+1) }, "at most"},
		{"bad id", func(d *Descriptor) { d.Engine = "a/b" }, "engine"},
		{"unknown version", func(d *Descriptor) { d.Version = 2 }, "version"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := frontableDescriptor()
			tc.mutate(&d)
			if err := ProxyCanFront(d); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ProxyCanFront error = %v, want %q", err, tc.want)
			}
		})
	}
}

// Every protocol the descriptor vocabulary accepts is one the proxy can front;
// adding a protocol to the vocabulary without its routes would make
// descriptors declaring it fail closed, which this test makes deliberate.
func TestFacadeProtocolsCoverTheVocabulary(t *testing.T) {
	for _, p := range Protocols() {
		if !slices.Contains(FacadeProtocols(), p) {
			t.Errorf("protocol %q has no facade routes", p)
		}
	}
	FacadeProtocols()[0] = "changed"
	if FacadeProtocols()[0] == "changed" {
		t.Fatal("FacadeProtocols aliases the shared table")
	}
}
