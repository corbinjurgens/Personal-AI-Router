// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"crypto/rand"
	"fmt"
	"testing"

	"nvpair-shared/noderec"
)

// Participants receive this value explicitly; no process-wide test registry.
func newFixtureMDNSNamespace(t *testing.T) noderec.MDNSNamespace {
	t.Helper()
	var token [12]byte
	if _, err := rand.Read(token[:]); err != nil {
		t.Fatal(err)
	}
	namespace := noderec.MDNSNamespace{Service: fmt.Sprintf("_pair-%x._tcp", token[:]), Domain: noderec.Domain}
	if err := validateFixtureMDNSNamespace(namespace); err != nil {
		t.Fatal(err)
	}
	t.Logf("fixture mDNS service=%s domain=%s", namespace.Service, namespace.Domain)
	return namespace
}

func validateFixtureMDNSNamespace(namespace noderec.MDNSNamespace) error {
	normalized, err := namespace.Normalize()
	if err != nil {
		return err
	}
	if normalized.Service == noderec.ServiceType {
		return fmt.Errorf("fixture mDNS requires a private service under local: %+v", namespace)
	}
	return nil
}

func fixtureMDNSBrokerArgs(namespace noderec.MDNSNamespace) []string {
	return []string{"--scanner-mdns=true", "--scanner-mdns-service", namespace.Service}
}

func fixtureMDNSScannerArgs(namespace noderec.MDNSNamespace) []string {
	return []string{"--mdns=true", "--mdns-service", namespace.Service}
}

func TestFixtureMDNSNamespaceIsPrivatePerGroup(t *testing.T) {
	first, second := newFixtureMDNSNamespace(t), newFixtureMDNSNamespace(t)
	if first.Service == second.Service {
		t.Fatalf("fixture namespaces collide: %+v %+v", first, second)
	}
	if first.Domain != noderec.Domain || second.Domain != noderec.Domain {
		t.Fatalf("fixture domains must be local: %+v %+v", first, second)
	}
	if err := validateFixtureMDNSNamespace(noderec.MDNSNamespace{Service: first.Service, Domain: "LOCAL."}); err != nil {
		t.Fatalf("normalized local domain rejected: %v", err)
	}
	for _, namespace := range []noderec.MDNSNamespace{
		{}, {Service: noderec.ServiceType, Domain: first.Domain},
		{Service: first.Service, Domain: "fixture.local"}, {Service: first.Service, Domain: "outside.example"},
	} {
		if err := validateFixtureMDNSNamespace(namespace); err == nil {
			t.Fatalf("unsafe namespace accepted: %+v", namespace)
		}
	}
}
