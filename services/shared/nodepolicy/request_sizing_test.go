// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func validPeerSizingJSON() string {
	return `{"revision":3,"peers":[{"host":"node-b","admissionIdentity":"incarnation-b","admissionIncarnation":"7","policyRevision":"12","proxyRunId":"run-b","ageMs":1500,"profiles":[` +
		`{"key":"medium","engine":"llamacpp","model":"gemma-4"},` +
		`{"key":"fast","engine":"llamacpp","model":"qwen-3","requestOptions":{"max_tokens":64,"options":{"num_predict":64}},"requestOptionsMode":"enforced"}]}]}`
}

func TestPeerSizingSnapshotDecodesValidReplacement(t *testing.T) {
	var got PeerSizingSnapshot
	if err := json.Unmarshal([]byte(validPeerSizingJSON()), &got); err != nil {
		t.Fatalf("decode valid snapshot: %v", err)
	}
	if got.Revision != 3 || len(got.Peers) != 1 || len(got.Peers[0].Profiles) != 2 {
		t.Fatalf("decoded snapshot = %+v", got)
	}
	if string(got.Peers[0].Profiles[1].RequestOptions) != `{"max_tokens":64,"options":{"num_predict":64}}` {
		t.Fatalf("request options bytes changed: %s", got.Peers[0].Profiles[1].RequestOptions)
	}
	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var again PeerSizingSnapshot
	if err := json.Unmarshal(encoded, &again); err != nil {
		t.Fatalf("round trip: %v", err)
	}

	t.Run("empty snapshot withdraws everything", func(t *testing.T) {
		var empty PeerSizingSnapshot
		if err := json.Unmarshal([]byte(`{"revision":4,"peers":[]}`), &empty); err != nil || len(empty.Peers) != 0 {
			t.Fatalf("empty snapshot = %+v, %v", empty, err)
		}
	})

	t.Run("known no-op recipes", func(t *testing.T) {
		for _, options := range []string{``, `,"requestOptions":null`, `,"requestOptions":{}`} {
			raw := fmt.Sprintf(`{"key":"medium","engine":"llamacpp","model":"gemma-4"%s}`, options)
			var p PeerSizingProfile
			if err := json.Unmarshal([]byte(raw), &p); err != nil {
				t.Errorf("profile %s: %v", raw, err)
			}
		}
	})

	t.Run("clone detaches option bytes", func(t *testing.T) {
		clone := got.Clone()
		clone.Peers[0].Profiles[1].RequestOptions[2] = 'X'
		if got.Peers[0].Profiles[1].RequestOptions[2] == 'X' {
			t.Fatal("clone shares request option bytes")
		}
	})
}

func TestPeerSizingSnapshotRejectsInvalidFrames(t *testing.T) {
	valid := validPeerSizingJSON()
	cases := []struct {
		name string
		raw  string
	}{
		{"unknown envelope field", strings.Replace(valid, `"revision":3`, `"revision":3,"extra":1`, 1)},
		{"duplicate envelope field", strings.Replace(valid, `"revision":3`, `"revision":3,"revision":4`, 1)},
		{"case-folded duplicate peer field", strings.Replace(valid, `"host":"node-b"`, `"host":"node-b","Host":"node-c"`, 1)},
		{"case-folded field name", strings.Replace(valid, `"host":"node-b"`, `"HOST":"node-b"`, 1)},
		{"unknown profile field", strings.Replace(valid, `"model":"gemma-4"}`, `"model":"gemma-4","localOnly":true}`, 1)},
		{"trailing data", valid + `{}`},
		{"zero revision", strings.Replace(valid, `"revision":3`, `"revision":0`, 1)},
		{"missing peers", `{"revision":3}`},
		{"noncanonical epoch", strings.Replace(valid, `"admissionIncarnation":"7"`, `"admissionIncarnation":"07"`, 1)},
		{"zero epoch", strings.Replace(valid, `"admissionIncarnation":"7"`, `"admissionIncarnation":"0"`, 1)},
		{"invalid policy revision", strings.Replace(valid, `"policyRevision":"12"`, `"policyRevision":"012"`, 1)},
		{"empty proxy run", strings.Replace(valid, `"proxyRunId":"run-b"`, `"proxyRunId":""`, 1)},
		{"negative age", strings.Replace(valid, `"ageMs":1500`, `"ageMs":-1`, 1)},
		{"expired age", strings.Replace(valid, `"ageMs":1500`, `"ageMs":90000`, 1)},
		{"duplicate profile key", strings.Replace(valid, `"key":"fast"`, `"key":"medium"`, 1)},
		{"options set model", strings.Replace(valid, `"max_tokens":64,`, `"Model":"x","max_tokens":64,`, 1)},
		{"enforced ambiguous keys", strings.Replace(valid, `"max_tokens":64,`, `"max_tokens":64,"MAX_TOKENS":1,`, 1)},
		{"options not an object", strings.Replace(valid, `{"max_tokens":64,"options":{"num_predict":64}}`, `[1]`, 1)},
		{"invalid mode", strings.Replace(valid, `"enforced"`, `"strict"`, 1)},
		{"reserved alias model", strings.Replace(valid, `"model":"gemma-4"`, `"model":"medium:local"`, 1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got PeerSizingSnapshot
			if err := json.Unmarshal([]byte(tc.raw), &got); err == nil {
				t.Fatalf("accepted invalid snapshot %s", tc.raw)
			}
		})
	}
}

func TestPeerSizingSnapshotBounds(t *testing.T) {
	peer := func(host string, profiles int) PeerSizing {
		p := PeerSizing{Host: host, AdmissionIdentity: "id-" + host, AdmissionIncarnation: "1", PolicyRevision: "1", ProxyRunID: "run", Profiles: []PeerSizingProfile{}}
		for i := range profiles {
			p.Profiles = append(p.Profiles, PeerSizingProfile{Key: fmt.Sprintf("k%d", i), Engine: "llamacpp", Model: fmt.Sprintf("model-%d", i)})
		}
		return p
	}

	t.Run("duplicate host", func(t *testing.T) {
		s := PeerSizingSnapshot{Revision: 1, Peers: []PeerSizing{peer("a", 1), peer("a", 1)}}
		if err := s.Validate(); err == nil {
			t.Fatal("accepted duplicate host")
		}
	})
	t.Run("too many profiles", func(t *testing.T) {
		s := PeerSizingSnapshot{Revision: 1, Peers: []PeerSizing{peer("a", MaxPeerSizingProfiles+1)}}
		if err := s.Validate(); err == nil {
			t.Fatal("accepted more than the profile bound")
		}
	})
	t.Run("too many peers", func(t *testing.T) {
		s := PeerSizingSnapshot{Revision: 1}
		for i := range MaxPeerSizingPeers + 1 {
			s.Peers = append(s.Peers, peer(fmt.Sprintf("n%d", i), 0))
		}
		if err := s.Validate(); err == nil {
			t.Fatal("accepted more than the peer bound")
		}
	})
	t.Run("encoded byte bound", func(t *testing.T) {
		s := PeerSizingSnapshot{Revision: 1}
		big := `{"pad":"` + strings.Repeat("x", 4000) + `"}`
		for i := range 70 {
			p := peer(fmt.Sprintf("n%d", i), 1)
			p.Profiles[0].RequestOptions = json.RawMessage(big)
			s.Peers = append(s.Peers, p)
		}
		if err := s.Validate(); err == nil {
			t.Fatal("accepted a snapshot over the encoded byte bound")
		}
	})
}

func TestPeerSizingActiveRequestPolicyIsStrict(t *testing.T) {
	var a ActiveRequestPolicy
	if err := json.Unmarshal([]byte(`{"revision":"5","proxyRunId":"run-1"}`), &a); err != nil || a.Revision != "5" || a.ProxyRunID != "run-1" {
		t.Fatalf("valid attestation = %+v, %v", a, err)
	}
	for _, raw := range []string{
		`{"revision":"5"}`,
		`{"revision":"05","proxyRunId":"run-1"}`,
		`{"revision":"5","proxyRunId":"run-1","extra":true}`,
		`{"revision":"5","proxyRunId":"run-1","ProxyRunId":"run-2"}`,
	} {
		if err := json.Unmarshal([]byte(raw), &a); err == nil {
			t.Errorf("accepted invalid attestation %s", raw)
		}
	}
}
