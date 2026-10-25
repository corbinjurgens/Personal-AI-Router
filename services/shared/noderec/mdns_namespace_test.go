// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"strings"
	"testing"
)

func TestMDNSNamespaceNormalization(t *testing.T) {
	for _, input := range []MDNSNamespace{
		{ServiceType, Domain},
		{"_PAIR-Fixture._TCP.", "LOCAL."},
	} {
		got, err := input.Normalize()
		if err != nil {
			t.Fatal(err)
		}
		want := MDNSNamespace{strings.ToLower(strings.TrimSuffix(input.Service, ".")), strings.ToLower(strings.TrimSuffix(input.Domain, "."))}
		if got != want {
			t.Fatalf("namespace = %+v, want %+v", got, want)
		}
	}
}

func TestMDNSNamespaceRejectsInvalidExplicitNames(t *testing.T) {
	for _, tc := range []MDNSNamespace{
		{"", Domain}, {ServiceType, ""}, {"._pair._tcp", Domain},
		{"_pair._udp", Domain}, {"_pair._tcp.extra", Domain}, {"_pair bad._tcp", Domain},
		{"_" + strings.Repeat("a", 63) + "._tcp", Domain},
		{ServiceType, ".local"}, {ServiceType, "fixture..local"},
		{ServiceType, "fixture.local"}, {ServiceType, "example.org"},
		{ServiceType, "fixture.local.."}, {ServiceType, "bad_name.local"},
		{"_K._tcp", Domain}, {ServiceType, "K.local"},
		{ServiceType, "-bad.local"}, {ServiceType, "bad-.local"},
		{ServiceType, strings.Repeat("a", 64) + ".local"},
		{ServiceType, strings.Repeat("longlabel.", 30) + "local"},
	} {
		t.Run(tc.Service+"/"+tc.Domain, func(t *testing.T) {
			if _, err := tc.Normalize(); err == nil {
				t.Fatalf("invalid explicit namespace accepted: %+v", tc)
			}
		})
	}
}
