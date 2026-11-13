// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"nvpair-shared/engines"
)

func validPeerServices() PeerServicesDescriptor {
	return PeerServicesDescriptor{HostUUID: "peer-a", Addresses: []string{"192.168.1.5"}, Services: []PeerService{{Service: ServiceNodeInfo, Port: 21435}, {Service: ServiceEngineControl, Port: 31421}}}
}

func TestPeerServicesDescriptorRejectsInvalidSnapshot(t *testing.T) {
	test := func(name string, mutate func(*PeerServicesDescriptor)) {
		t.Run(name, func(t *testing.T) {
			d := validPeerServices()
			mutate(&d)
			if err := ValidatePeerServicesDescriptor(d, "peer-a"); err == nil {
				t.Fatal("invalid descriptor accepted")
			}
		})
	}
	test("third party subject", func(d *PeerServicesDescriptor) { d.HostUUID = "peer-b" })
	test("missing initialized services", func(d *PeerServicesDescriptor) { d.Services = nil })
	test("duplicate service", func(d *PeerServicesDescriptor) { d.Services = append(d.Services, d.Services[0]) })
	test("unknown service", func(d *PeerServicesDescriptor) { d.Services[0].Service = "http" })
	test("zero port", func(d *PeerServicesDescriptor) { d.Services[0].Port = 0 })
	test("negative port", func(d *PeerServicesDescriptor) { d.Services[0].Port = -1 })
	test("oversized port", func(d *PeerServicesDescriptor) { d.Services[0].Port = 65536 })
	test("missing address", func(d *PeerServicesDescriptor) { d.Addresses = nil })
	test("too many addresses", func(d *PeerServicesDescriptor) {
		d.Addresses = []string{"10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.5"}
	})
	test("IPv4 loopback", func(d *PeerServicesDescriptor) { d.Addresses = []string{"127.0.0.1"} })
	test("IPv6 loopback", func(d *PeerServicesDescriptor) { d.Addresses = []string{"::1"} })
	test("IPv4 mapped loopback", func(d *PeerServicesDescriptor) { d.Addresses = []string{"::ffff:127.0.0.1"} })
	test("unspecified", func(d *PeerServicesDescriptor) { d.Addresses = []string{"0.0.0.0"} })
	test("multicast", func(d *PeerServicesDescriptor) { d.Addresses = []string{"224.0.0.251"} })
	test("hostname", func(d *PeerServicesDescriptor) { d.Addresses = []string{"localhost"} })
	test("URL", func(d *PeerServicesDescriptor) { d.Addresses = []string{"http://192.168.1.5"} })
	test("duplicate address", func(d *PeerServicesDescriptor) { d.Addresses = []string{"192.168.1.5", "192.168.1.5"} })
}

// A descriptor-backed engine's service key travels on this authenticated
// path; a malformed or excess dynamic key does not.
func TestPeerServicesDescriptorCarriesDynamicEngineKeys(t *testing.T) {
	d := validPeerServices()
	d.Services = append(d.Services, PeerService{Service: "e-llamacpp-custom", Port: 18180}, PeerService{Service: "e-strata", Port: 18181})
	if err := ValidatePeerServicesDescriptor(d, "peer-a"); err != nil {
		t.Fatalf("dynamic engine keys rejected: %v", err)
	}
	for _, key := range []ServiceKey{"e-", "e-llamacpp", "e-a/b", "e-a:b", ServiceKey("e-" + strings.Repeat("a", engines.MaxDynamicIDLen+1))} {
		t.Run(string(key), func(t *testing.T) {
			d := validPeerServices()
			d.Services = append(d.Services, PeerService{Service: key, Port: 18180})
			if err := ValidatePeerServicesDescriptor(d, "peer-a"); err == nil {
				t.Fatal("malformed dynamic key accepted")
			}
		})
	}
	t.Run("too many dynamic keys", func(t *testing.T) {
		d := validPeerServices()
		d.Services = nil
		for i := range engines.MaxDynamicEngines + 1 {
			d.Services = append(d.Services, PeerService{Service: ServiceKey(fmt.Sprintf("e-engine%d", i)), Port: 18180 + i})
		}
		if err := ValidatePeerServicesDescriptor(d, "peer-a"); err == nil {
			t.Fatal("more than MaxDynamicEngines dynamic keys accepted")
		}
		d.Services = d.Services[:engines.MaxDynamicEngines]
		if err := ValidatePeerServicesDescriptor(d, "peer-a"); err != nil {
			t.Fatalf("MaxDynamicEngines dynamic keys rejected: %v", err)
		}
	})
}

func TestPeerServicesCompleteWithdrawal(t *testing.T) {
	d := validPeerServices()
	d.Services = []PeerService{}
	if err := ValidatePeerServicesDescriptor(d, "peer-a"); err != nil {
		t.Fatalf("initialized withdrawal rejected: %v", err)
	}
}

func TestPeerStateFrameDecodeIsStrict(t *testing.T) {
	hash := strings.Repeat("ab", 32)
	body, err := json.Marshal(PeerStateFrame{Descriptor: validPeerServices(), RosterHash: hash})
	if err != nil {
		t.Fatal(err)
	}
	reject := func(name, line, expected string) {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodePeerStateFrame([]byte(line), expected); err == nil {
				t.Fatal("invalid frame accepted")
			}
		})
	}
	reject("truncated JSON", string(body[:len(body)-1]), "peer-a")
	reject("trailing document", string(body)+" {}", "peer-a")
	reject("trailing junk", string(body)+" x", "peer-a")
	reject("oversized", string(body)+strings.Repeat(" ", PeerStateMaxLine), "peer-a")
	reject("unknown frame field", strings.Replace(string(body), `"rosterHash":`, `"generation":1,"rosterHash":`, 1), "peer-a")
	reject("unknown descriptor field", strings.Replace(string(body), `"addresses":`, `"name":"peer host","addresses":`, 1), "peer-a")
	reject("wrong subject", string(body), "peer-b")
	reject("uppercase hash", strings.Replace(string(body), hash, strings.ToUpper(hash), 1), "peer-a")
	reject("short hash", strings.Replace(string(body), hash, hash[:63], 1), "peer-a")
	reject("missing hash", strings.Replace(string(body), `"rosterHash":"`+hash+`"`, `"rosterHash":""`, 1), "peer-a")
	reject("null keepalive", "null", "peer-a")
	t.Run("valid frame", func(t *testing.T) {
		frame, err := DecodePeerStateFrame(body, "peer-a")
		if err != nil {
			t.Fatal(err)
		}
		if frame.RosterHash != hash || frame.Descriptor.HostUUID != "peer-a" || len(frame.Descriptor.Services) != 2 {
			t.Fatalf("decoded frame = %+v", frame)
		}
	})
}
