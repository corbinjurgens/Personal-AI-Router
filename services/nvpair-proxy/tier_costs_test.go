// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"reflect"
	"testing"
	"testing/synctest"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/performancewire"
)

func proxyRoutingCostsFixture() performancewire.RoutingCosts {
	return performancewire.RoutingCosts{Revision: 2, Profiles: []performancewire.RoutingProfile{{Host: "destination", Engine: "llamacpp", Model: "profile/key", Fingerprint: "fingerprint",
		Family: "llamacpp", APIPath: "/v1/chat/completions", Schema: "llamacpp-b12020-0ec4027a", AgeMs: 100,
		Decode: &performancewire.RoutingRate{Samples: 1, TokensPerSecond: 79, Source: "engine"}}}}
}

func requireRoutingEvidence(t *testing.T, p *Proxy) (performancewire.RoutingProfile, int64) {
	t.Helper()
	e, age, ok := p.routingEvidence("destination", "llamacpp", "profile/key", "llamacpp", "/v1/chat/completions", "llamacpp-b12020-0ec4027a")
	if !ok {
		t.Fatal("current routing evidence missing")
	}
	return e, age
}

func TestRoutingCostsRevisionReplacementAndRejection(t *testing.T) {
	p, _ := policyProxy(t)
	c := proxyRoutingCostsFixture()
	if err := p.setRoutingCosts(c); err != nil {
		t.Fatal(err)
	}
	received := p.routingCostsReceived
	for _, revision := range []int64{1, 2} {
		t.Run(map[int64]string{1: "lower revision rejected", 2: "equal revision no-op"}[revision], func(t *testing.T) {
			next := proxyRoutingCostsFixture()
			next.Revision, next.Profiles[0].Decode.TokensPerSecond = revision, 999
			err := p.setRoutingCosts(next)
			if (revision == 1) != (err != nil) {
				t.Fatalf("revision %d error = %v", revision, err)
			}
			e, _ := requireRoutingEvidence(t, p)
			if e.Decode.TokensPerSecond != 79 || p.routingCostsRevision != 2 || p.routingCostsReceived != received {
				t.Fatalf("repeated revision mutated cache: %+v", e)
			}
		})
	}
	bad := proxyRoutingCostsFixture()
	bad.Revision, bad.Profiles[0].Decode.TokensPerSecond = 3, 0
	if err := p.setRoutingCosts(bad); err == nil {
		t.Fatal("invalid payload accepted")
	}
	e, _ := requireRoutingEvidence(t, p)
	if e.Decode.TokensPerSecond != 79 || p.routingCostsRevision != 2 || p.routingCostsReceived != received {
		t.Fatal("invalid payload changed cache")
	}
	if err := p.setRoutingCosts(performancewire.RoutingCosts{Revision: 3, Profiles: []performancewire.RoutingProfile{}}); err != nil {
		t.Fatal(err)
	}
	if len(p.routingCosts) != 0 || p.routingCostsRevision != 3 {
		t.Fatal("empty full replacement did not withdraw evidence")
	}
}

func TestRoutingCostsExpiresAtLookupWithMonotonicReceiptAge(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		p, _ := policyProxy(t)
		c := proxyRoutingCostsFixture()
		c.Profiles[0].AgeMs = performancewire.RetentionMs - 1000
		if err := p.setRoutingCosts(c); err != nil {
			t.Fatal(err)
		}
		time.Sleep(time.Second)
		_, age := requireRoutingEvidence(t, p)
		if age != performancewire.RetentionMs {
			t.Fatalf("advanced age = %d", age)
		}
		// Equal revision cannot reset the receipt clock or extend freshness.
		if err := p.setRoutingCosts(c); err != nil {
			t.Fatal(err)
		}
		time.Sleep(time.Millisecond)
		if _, _, ok := p.routingEvidence("destination", "llamacpp", "profile/key", "llamacpp", "/v1/chat/completions", "llamacpp-b12020-0ec4027a"); ok {
			t.Fatal("expired evidence survived lookup")
		}
	})
}

func TestRoutingCostsOwnsWholeCacheAndLookupValues(t *testing.T) {
	p, _ := policyProxy(t)
	c := proxyRoutingCostsFixture()
	ms := int64(23)
	c.Profiles[0].Prefill = &performancewire.RoutingRate{Samples: 1, TokensPerSecond: 378, Source: "count"}
	c.Profiles[0].LoadP95Ms, c.Profiles[0].LoadSamples = &ms, 1
	c.Profiles[0].Extra = map[string]float64{"future": 1}
	if err := p.setRoutingCosts(c); err != nil {
		t.Fatal(err)
	}
	c.Profiles[0].Decode.TokensPerSecond = 999
	c.Profiles[0].Prefill.TokensPerSecond = 999
	ms, c.Profiles[0].Extra["future"] = 999, 999
	e, _ := requireRoutingEvidence(t, p)
	if e.Decode.TokensPerSecond != 79 || e.Prefill.TokensPerSecond != 378 || *e.LoadP95Ms != 23 || e.Extra["future"] != 1 {
		t.Fatalf("input retained: %+v", e)
	}
	e.Decode.TokensPerSecond, e.Prefill.TokensPerSecond, *e.LoadP95Ms, e.Extra["future"] = 0, 0, 0, 0
	again, _ := requireRoutingEvidence(t, p)
	if again.Decode.TokensPerSecond != 79 || again.Prefill.TokensPerSecond != 378 || *again.LoadP95Ms != 23 || again.Extra["future"] != 1 {
		t.Fatal("lookup mutated cache")
	}
}

func TestRoutingCostsHandlerRejectsAddressedAndInvalidPayloads(t *testing.T) {
	p, w := policyProxy(t)
	c := proxyRoutingCostsFixture()
	response := rpc(t, p, w, performancewire.MethodRoutingCosts, c)
	if response.Error != nil {
		t.Fatalf("valid feed rejected: %+v", response.Error)
	}
	var result struct {
		Profiles int `json:"profiles"`
	}
	if err := json.Unmarshal(response.Result, &result); err != nil {
		t.Fatal(err)
	}
	if result.Profiles != 1 {
		t.Fatalf("stored profile count = %d", result.Profiles)
	}
	for _, method := range []string{engines.AddressMethod("llamacpp", performancewire.MethodRoutingCosts), performancewire.MethodRoutingCosts} {
		next := proxyRoutingCostsFixture()
		next.Revision = 3
		var params any = next
		if method == performancewire.MethodRoutingCosts {
			params = map[string]any{"revision": 3, "profiles": next.Profiles, "unknown": true}
		}
		response := rpc(t, p, w, method, params)
		if response.Error == nil || response.Error.Code != -32602 {
			t.Fatalf("invalid/addressed method accepted: %+v", response)
		}
		if p.routingCostsRevision != 2 {
			t.Fatal("rejected RPC changed revision")
		}
	}
}

func TestRoutingCostsIndependentOfPriorityAndReservations(t *testing.T) {
	p, _ := policyProxy(t)
	p.priority = []string{"destination"}
	p.priorityPending = map[string]int{"destination": 2}
	p.priorityGPUPressure = map[string]int{"destination": 3}
	p.priorityReservations = map[string]int{"destination": 4}
	p.appliedPriorityGeneration = 7
	if err := p.setRoutingCosts(proxyRoutingCostsFixture()); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p.priority, []string{"destination"}) || !reflect.DeepEqual(p.priorityPending, map[string]int{"destination": 2}) || !reflect.DeepEqual(p.priorityGPUPressure, map[string]int{"destination": 3}) || !reflect.DeepEqual(p.priorityReservations, map[string]int{"destination": 4}) || p.appliedPriorityGeneration != 7 {
		t.Fatal("routing feed changed scheduler or reservation state")
	}
}

func TestRoutingCostsCandidateKeyMatchesSampleProducerAcrossEngines(t *testing.T) {
	w := &recordingWriter{}
	p, ingress, destination := twoEngineProxy(t, w)
	p.noteSelf("destination", destination.profile.Name, []string{"Org/Profile:Q4"})
	p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{destination.profile.Name: performanceIdentity()})
	pol := p.admission.currentPolicy()
	pol.Profiles = []nodepolicy.Profile{{Key: "Org/Profile:Q4", DisplayName: "Candidate profile", Engine: destination.profile.Name, Model: "native-artifact.gguf"}}
	p.admission.setPolicy(pol)
	candidate := candidate{id: "destination", engine: destination.profile.Name, model: "Org/Profile:Q4"}
	key, q, _, ok := destination.performanceFingerprintOwned(candidate.model, "/v1/chat/completions", []byte(`{"model":"native-artifact.gguf","stream":true}`))
	if !ok {
		t.Fatal("destination sample producer refused fixture")
	}
	if key.Host != "destination" || key.Engine != destination.profile.Name || key.Model != "Org/Profile:Q4" || key.Model == pol.Profiles[0].Model || key.Engine == ingress.profile.Name {
		t.Fatalf("sample profile key = %+v", key)
	}
	c := proxyRoutingCostsFixture()
	c.Profiles[0].Host, c.Profiles[0].Engine, c.Profiles[0].Model, c.Profiles[0].Fingerprint = key.Host, key.Engine, key.Model, key.Fingerprint
	c.Profiles[0].Family, c.Profiles[0].APIPath, c.Profiles[0].Schema = q.EngineFamily, q.APIPath, q.SchemaID
	if err := p.setRoutingCosts(c); err != nil {
		t.Fatal(err)
	}
	e, _, ok := p.routingEvidence(candidate.id, candidate.engine, candidate.model, q.EngineFamily, q.APIPath, q.SchemaID)
	if !ok || e.Host != key.Host || e.Engine != key.Engine || e.Model != key.Model || e.Fingerprint != key.Fingerprint {
		t.Fatalf("candidate/sample key mismatch: %+v, %+v", e, key)
	}
	for name, tuple := range map[string][6]string{
		"host":            {"other", candidate.engine, candidate.model, q.EngineFamily, q.APIPath, q.SchemaID},
		"ingress engine":  {key.Host, ingress.profile.Name, candidate.model, q.EngineFamily, q.APIPath, q.SchemaID},
		"native model":    {key.Host, candidate.engine, pol.Profiles[0].Model, q.EngineFamily, q.APIPath, q.SchemaID},
		"case":            {key.Host, candidate.engine, "org/profile:Q4", q.EngineFamily, q.APIPath, q.SchemaID},
		"family":          {key.Host, candidate.engine, candidate.model, "strata", q.APIPath, q.SchemaID},
		"API":             {key.Host, candidate.engine, candidate.model, q.EngineFamily, "/v1/completions", q.SchemaID},
		"schema":          {key.Host, candidate.engine, candidate.model, q.EngineFamily, q.APIPath, "unknown"},
		"invalid profile": {key.Host, candidate.engine, "", q.EngineFamily, q.APIPath, q.SchemaID},
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, ok := p.routingEvidence(tuple[0], tuple[1], tuple[2], tuple[3], tuple[4], tuple[5]); ok {
				t.Fatal("incompatible key matched")
			}
		})
	}
	if !performancewire.RoutingCompatible("llamacpp", "/v1/chat/completions", performanceExtractLlamaSchema) || !performancewire.RoutingCompatible("strata", "/v1/chat/completions", performanceExtractStrataSchema) {
		t.Fatal("routing tuple registry differs from sample extractor schemas")
	}
}
