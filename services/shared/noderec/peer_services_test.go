// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
)

func validPeerServices() PeerServicesDescriptor {
	return PeerServicesDescriptor{HostUUID: "peer-a", Name: "peer host", Addresses: []string{"192.168.1.5"}, Services: []PeerService{{Service: ServiceNodeInfo, Port: 21435}, {Service: ServiceEngineControl, Port: 31421}}}
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
	test("name control character", func(d *PeerServicesDescriptor) { d.Name = "host\nforged" })
}

func TestPeerServicesCompleteWithdrawal(t *testing.T) {
	d := validPeerServices()
	d.Services = []PeerService{}
	if err := ValidatePeerServicesDescriptor(d, "peer-a"); err != nil {
		t.Fatalf("initialized withdrawal rejected: %v", err)
	}
}

type peerServicesReadError struct{ io.Reader }

func (r peerServicesReadError) Read(p []byte) (int, error) {
	n, err := r.Reader.Read(p)
	if err == io.EOF {
		return n, errors.New("response truncated")
	}
	return n, err
}

func TestPeerServicesDecodeRequiresCompleteBoundedDocument(t *testing.T) {
	body, err := json.Marshal(validPeerServices())
	if err != nil {
		t.Fatal(err)
	}
	test := func(name, body string) {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodePeerServices(strings.NewReader(body), "peer-a"); err == nil {
				t.Fatal("invalid document accepted")
			}
		})
	}
	test("truncated JSON", string(body[:len(body)-1]))
	test("trailing document", string(body)+" {}")
	test("trailing junk", string(body)+" x")
	test("oversized", string(body)+strings.Repeat(" ", PeerServicesMaxBytes))
	test("unknown protocol field", strings.Replace(string(body), `"name":`, `"protocol":"plain","name":`, 1))
	t.Run("transport error after complete JSON", func(t *testing.T) {
		if _, err := DecodePeerServices(peerServicesReadError{strings.NewReader(string(body))}, "peer-a"); err == nil {
			t.Fatal("incomplete transport accepted")
		}
	})
	t.Run("complete valid document", func(t *testing.T) {
		d, err := DecodePeerServices(strings.NewReader(string(body)), "peer-a")
		if err != nil {
			t.Fatal(err)
		}
		if d.HostUUID != "peer-a" || len(d.Services) != 2 || d.Services[1].Service != ServiceEngineControl || d.Services[1].Port != 31421 {
			t.Fatalf("decoded descriptor = %+v", d)
		}
	})
}
