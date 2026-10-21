// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"reflect"
	"testing"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func TestSubscribedToNodeCarriesPausedBy(t *testing.T) {
	for _, profile := range append(buildProfiles(), strataProfile(t)) {
		t.Run(profile.Name, func(t *testing.T) {
			test := func(name string, policy *nodepolicy.Advertisement, revision string, pausedBy nodepolicy.PauseOrigin) {
				t.Run(name, func(t *testing.T) {
					capability := strictTestCapability("m")
					if policy != nil {
						policy.Roles = map[string]nodepolicy.AdvertisedRole{
							nodepolicy.TierStrong: {Engine: profile.Name, Model: "m", ContextTokens: 32768},
						}
						policy.ServedModels = nodepolicy.ServedModels{{Engine: profile.Name, Model: "m"}}
						policy.StrictAdmission = capability
					}
					n := noderec.DirectoryNode{
						HostUUID:    "host-uuid",
						Name:        "host-a",
						IP:          "10.0.0.5",
						IPs:         []string{"10.0.0.5", "10.0.0.6"},
						ClusterUUID: "cluster-uuid",
						Services: map[noderec.ServiceKey]noderec.ServiceStatus{
							profile.DiscoveryService: {Port: 19001},
						},
						Models:         []string{"m", "other-engine-model"},
						ModelsByEngine: map[string][]string{profile.Name: {"m"}},
						LoadedByEngine: map[string][]string{profile.Name: {"m"}},
						Policy:         policy,
					}
					got, ok := subscribedToNode(profile, n)
					if !ok {
						t.Fatal("node advertising this engine and IP should project")
					}
					want := Node{
						ID:             "host-uuid",
						Host:           "host-a",
						Port:           19001,
						Addresses:      []string{"10.0.0.5", "10.0.0.6"},
						TXT:            []string{"ip=10.0.0.5", "ips=10.0.0.5,10.0.0.6"},
						Models:         []string{"m"},
						Loaded:         []string{"m"},
						IP:             "10.0.0.5",
						ClusterUUID:    "cluster-uuid",
						PolicyRevision: revision,
						PausedBy:       pausedBy,
					}
					if policy != nil {
						want.Roles = map[string]nodepolicy.AdvertisedRole{
							nodepolicy.TierStrong: {Engine: profile.Name, Model: "m", ContextTokens: 32768},
						}
						want.ServedModels = nodepolicy.ServedModels{{Engine: profile.Name, Model: "m"}}
						want.StrictAdmission = capability
					}
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("projected node = %+v, want %+v", got, want)
					}

					data, err := json.Marshal(got)
					if err != nil {
						t.Fatalf("encode projected node: %v", err)
					}
					var wire map[string]json.RawMessage
					if err := json.Unmarshal(data, &wire); err != nil {
						t.Fatalf("decode projected node: %v", err)
					}
					for _, key := range []string{"pausedBy", "PausedBy", "policyRevision", "PolicyRevision"} {
						if value, exists := wire[key]; exists {
							t.Errorf("internal routing metadata %q serialized as %s", key, value)
						}
					}
				})
			}
			test("nil policy", nil, "", "")
			test("revision without pause", &nodepolicy.Advertisement{Revision: "41"}, "41", "")
			test("local pause", &nodepolicy.Advertisement{Revision: "42", PausedBy: nodepolicy.PausedLocally}, "42", nodepolicy.PausedLocally)
			test("peer pause", &nodepolicy.Advertisement{Revision: "43", PausedBy: nodepolicy.PausedByPeer}, "43", nodepolicy.PausedByPeer)
		})
	}
}

func TestPauseHintAnnouncesNoNodeUpdate(t *testing.T) {
	d := NewDiscovery()
	initial := Node{
		ID:             "host-uuid",
		Host:           "host-a",
		Port:           19001,
		Addresses:      []string{"10.0.0.5"},
		TXT:            []string{"ip=10.0.0.5"},
		Models:         []string{"m"},
		IP:             "10.0.0.5",
		PolicyRevision: "40",
	}
	discovered, updated, removed := d.SetSubscribed([]Node{initial})
	if !reflect.DeepEqual(discovered, []Node{initial}) || len(updated) != 0 || len(removed) != 0 {
		t.Fatalf("initial changes: discovered=%+v updated=%+v removed=%+v", discovered, updated, removed)
	}
	if got := d.Nodes(); !reflect.DeepEqual(got, []Node{initial}) {
		t.Fatalf("initial snapshot = %+v, want %+v", got, initial)
	}
	for _, step := range []struct {
		name     string
		revision string
		pausedBy nodepolicy.PauseOrigin
	}{
		{"revision only", "41", ""},
		{"local pause", "42", nodepolicy.PausedLocally},
		{"provenance only", "42", nodepolicy.PausedByPeer},
		{"available again", "43", ""},
	} {
		t.Run(step.name, func(t *testing.T) {
			next := initial
			next.PolicyRevision = step.revision
			next.PausedBy = step.pausedBy
			discovered, updated, removed := d.SetSubscribed([]Node{next})
			if len(discovered) != 0 || len(updated) != 0 || len(removed) != 0 {
				t.Errorf("metadata replacement announced changes: discovered=%+v updated=%+v removed=%+v", discovered, updated, removed)
			}
			if got := d.Nodes(); !reflect.DeepEqual(got, []Node{next}) {
				t.Fatalf("stored snapshot = %+v, want %+v", got, next)
			}
		})
	}
}
