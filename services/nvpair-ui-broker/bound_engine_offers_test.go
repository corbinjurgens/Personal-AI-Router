// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

func TestEngineBindingOffersDescriptorDefaultConfiguration(t *testing.T) {
	h := newBindingOfferHarness(t)
	d := testDescriptor("custom", engines.ProtocolOpenAIChat)
	d.ModelNaming = engines.ModelNamingImpliedLatestTag
	h.b.setDynamicEngines([]engines.Descriptor{d}, func(int) bool { return true })
	profile := h.b.dynamicEngineProfiles()[0]
	h.b.setEngineProxyHandle(profile, h.pp)
	h.pp.readyMu.Lock()
	h.pp.facadeState["custom"] = proxyFacadeState{ready: true, port: 18080}
	h.pp.readyMu.Unlock()
	r := h.b.nodePolicyRuntime()
	h.b.engineConfigMu.Lock()
	r.mu.Lock()
	r.current.Revision = "2"
	r.activeRevision = "2"
	r.current.EngineBindings = nodepolicy.EngineBindings{{Engine: "custom", Model: "fixture"}}
	r.current.ServedModels = nodepolicy.ServedModels{{Engine: "custom", Model: "fixture:latest"}}
	r.mu.Unlock()
	h.b.engineConfigMu.Unlock()
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:get-installed" {
			return map[string]any{"engines": []map[string]any{{"engine": "custom", "installed": true, "running": false, "healthy": false, "locked": false}}}, ""
		}
		return base(method, raw)
	})
	h.b.reconcileAdvertiseEngine(profile, nil)
	registrations := h.b.regCache.Snapshot()
	if len(registrations) != 1 || registrations[0].Service != profile.DiscoveryService || registrations[0].Port != 18080 {
		t.Fatalf("descriptor/default configuration lost cold registration: %+v", registrations)
	}
	h.b.engineConfigMu.Lock()
	offers := h.b.bindingOffersLocked()
	h.b.engineConfigMu.Unlock()
	if offers == nil || !reflect.DeepEqual(offers.Models, []nodepolicy.Role{{Engine: "custom", Model: "fixture"}}) {
		t.Fatalf("descriptor naming or native spelling changed: %+v", offers)
	}
}

func TestEngineBindingOffersProjectionContentionKeepsColdRegistration(t *testing.T) {
	h := newBindingOfferHarness(t)
	h.b.reconcileAdvertiseLMStudio(nil)
	h.b.engineConfigMu.Lock()
	// An unrelated advertiser or switch may hold this lock while a strict
	// refresh runs. A missing sample must not remove another owner's service.
	h.b.refreshStrictAdvertisement()
	h.b.engineConfigMu.Unlock()
	if got := h.b.regCache.Snapshot(); len(got) != 1 || got[0].Service != noderec.ServiceLMStudio {
		t.Fatalf("contended projection removed live cold facade: %+v", got)
	}
	h.b.refreshStrictAdvertisement()
	if got := h.b.regCache.Snapshot(); len(got) != 1 || !h.b.lmstudioState().bindingRegistered.Load() {
		t.Fatalf("republished offer has no registered facade: %+v", got)
	}
}

func TestEngineBindingOffersWithdrawDuringManagedBackendMove(t *testing.T) {
	h := newBindingOfferHarness(t)
	h.pp.readyMu.Lock()
	h.pp.facadeState["ollama"] = proxyFacadeState{ready: true, port: 18080}
	h.pp.readyMu.Unlock()
	r := h.b.nodePolicyRuntime()
	h.b.engineConfigMu.Lock()
	r.mu.Lock()
	r.current.Revision, r.activeRevision = "2", "2"
	r.current.EngineBindings = nodepolicy.EngineBindings{{Engine: "ollama", Model: "fixture"}}
	r.mu.Unlock()
	base := h.engine.answer
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:get-installed" {
			return map[string]any{"engines": []map[string]any{{"engine": "ollama", "installed": true, "running": false, "healthy": false, "locked": false}}}, ""
		}
		return base(method, raw)
	})
	registered := h.b.registerBindingFacadeLocked(ollamaProxyProfile)
	h.b.engineConfigMu.Unlock()
	if !registered {
		t.Fatal("fixture did not register stopped bound facade")
	}
	h.b.managedOllamaBackend.Store(1)
	h.b.reconcileAdvertise(nil) // the managed move must skip all engine probing
	h.b.engineConfigMu.Lock()
	offers := h.b.bindingOffersLocked()
	h.b.engineConfigMu.Unlock()
	if offers != nil || len(h.b.regCache.Snapshot()) != 0 {
		t.Fatalf("unresolved backend move retained a cold offer: %+v", offers)
	}
}

func TestEngineBindingOffersAdvertiserPreservesSourceDrain(t *testing.T) {
	h := newBindingOfferHarness(t)
	h.b.reconcileAdvertiseLMStudio(nil)
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.drains["lmstudio"] = true
	r.mu.Unlock()
	h.b.unregisterService(noderec.ServiceLMStudio)
	h.b.reconcileAdvertiseLMStudio(nil)
	if got := h.b.regCache.Snapshot(); len(got) != 0 || h.b.lmstudioState().bindingRegistered.Load() {
		t.Fatalf("advertiser undid source drain: %+v", got)
	}
}

func TestEngineBindingOffersHealthySwitchPublicationRetiresColdMarker(t *testing.T) {
	h := newBindingOfferHarness(t)
	h.b.reconcileAdvertiseLMStudio(nil)
	if err := h.b.switchProxySetLocal(context.Background(), h.pp, "lmstudio", 12345, true); err != nil {
		t.Fatal(err)
	}
	h.b.engineConfigMu.Lock()
	h.b.withdrawBindingFacades(nil)
	h.b.engineConfigMu.Unlock()
	if got := h.b.regCache.Snapshot(); len(got) != 1 || h.b.lmstudioState().bindingRegistered.Load() {
		t.Fatalf("offer withdrawal removed ready-engine registration: %+v", got)
	}
}

func newBindingOfferHarness(t *testing.T) *boundHarness {
	t.Helper()
	h := newBoundHarness(t, false, false)
	h.b.engineConfigMu.Lock()
	h.b.regCache = relay.NewRegistrationCache()
	h.b.engineSettingsLoaded = true
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.EngineBindings = nodepolicy.EngineBindings{{Engine: "lmstudio", Model: "model"}}
	r.mu.Unlock()
	h.b.engineConfigMu.Unlock()
	return h
}

func TestEngineBindingOffersStoppedFacadeRegistration(t *testing.T) {
	h := newBindingOfferHarness(t)
	h.b.reconcileAdvertiseLMStudio(nil) // stopped status short-circuits all health I/O
	registrations := h.b.regCache.Snapshot()
	_, port := h.pp.Status("lmstudio")
	if len(registrations) != 1 || registrations[0].Service != noderec.ServiceLMStudio || registrations[0].Port != port {
		t.Fatalf("stopped facade registrations=%+v actual port=%d", registrations, port)
	}
	var last proxyLocalBackend
	for _, call := range h.proxy.snapshot() {
		if call.method == "lmstudio:node/set-local-backend" {
			if err := json.Unmarshal(call.params, &last); err != nil {
				t.Fatal(err)
			}
		}
	}
	if last.Healthy || last.Port != 0 || last.Engine != "lmstudio" {
		t.Fatalf("cold registration invented backend readiness: %+v", last)
	}
	requireNoBoundEffects(t, h)
}

func TestEngineBindingOffersWithdrawUnknownOrVetoedEvidence(t *testing.T) {
	test := func(name string, change func(*boundHarness)) {
		t.Run(name, func(t *testing.T) {
			h := newBindingOfferHarness(t)
			h.b.reconcileAdvertiseLMStudio(nil)
			h.b.engineConfigMu.Lock()
			before := h.b.bindingOffersLocked()
			h.b.engineConfigMu.Unlock()
			if before == nil {
				t.Fatal("fixture did not offer stopped binding")
			}
			change(h)
			h.b.engineConfigMu.Lock()
			got := h.b.bindingOffersLocked()
			h.b.engineConfigMu.Unlock()
			if got != nil {
				t.Fatalf("withdrawn evidence still offered: %+v", got)
			}
			h.b.refreshStrictAdvertisement()
			if got := h.b.regCache.Snapshot(); len(got) != 0 {
				t.Fatalf("offer withdrawal retained cold service: %+v", got)
			}
			requireNoBoundEffects(t, h)
		})
	}
	policyChange := func(fn func(*policyRuntime)) func(*boundHarness) {
		return func(h *boundHarness) {
			r := h.b.nodePolicyRuntime()
			r.mu.Lock()
			fn(r)
			r.mu.Unlock()
		}
	}
	test("pause", policyChange(func(r *policyRuntime) { r.live = nodepolicy.Paused }))
	test("pending activation", policyChange(func(r *policyRuntime) { r.activeRevision = "" }))
	test("recovery", policyChange(func(r *policyRuntime) { r.recoveryRequired = true }))
	test("source drain", policyChange(func(r *policyRuntime) { r.drains["lmstudio"] = true }))
	test("unresolved settings", func(h *boundHarness) {
		h.b.engineConfigMu.Lock()
		h.b.engineSettingsError = errors.New("settings recovery required")
		h.b.engineConfigMu.Unlock()
	})
	test("pending settings operation", func(h *boundHarness) {
		h.b.engineConfigMu.Lock()
		h.b.engineSettings = map[string]*engineSettingsRecord{"lmstudio": {Snapshot: settings.Snapshot{Revision: 2, AppliedRevision: 1, Phase: "applying"}}}
		h.b.engineConfigMu.Unlock()
	})
	test("manager replacement", policyChange(func(r *policyRuntime) { r.engineManagerSource = &engineManagerSource{} }))
	test("stopped facade", func(h *boundHarness) {
		h.pp.readyMu.Lock()
		delete(h.pp.facadeState, "lmstudio")
		h.pp.readyMu.Unlock()
	})
	test("locked", func(h *boundHarness) {
		base := h.engine.answer
		h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
			if method == "engine:get-installed" {
				return map[string]any{"engines": []map[string]any{{"engine": "lmstudio", "installed": true, "running": false, "healthy": false, "locked": true}}}, ""
			}
			return base(method, raw)
		})
	})
}

func TestEngineBindingOffersProjectionRefreshWithoutStrictChange(t *testing.T) {
	h := newBindingOfferHarness(t)
	h.b.refreshStrictAdvertisement()
	h.engine.waitFor("binding offers projection", func([]policyCall) bool {
		projections := nodeProjections(t, h.engine)
		return len(projections) > 0 && projections[len(projections)-1].EngineBindingOffers != nil
	})
	projections := nodeProjections(t, h.engine)
	if projections[len(projections)-1].EngineBindingOffers == nil {
		t.Fatalf("unchanged strict readiness skipped offers: %+v", projections)
	}
	got := projections[len(projections)-1]
	if got.StrictAdmission != nil || !reflect.DeepEqual(got.EngineBindingOffers.Models, []nodepolicy.Role{{Engine: "lmstudio", Model: "model"}}) {
		t.Fatalf("projection=%+v", got)
	}
}

func TestEngineBindingOffersClearingPublishesWithoutStrictChange(t *testing.T) {
	h := newBindingOfferHarness(t)
	h.b.reconcileAdvertiseLMStudio(nil)
	h.engine.waitFor("initial offers", func([]policyCall) bool {
		projections := nodeProjections(t, h.engine)
		return len(projections) > 0 && projections[len(projections)-1].EngineBindingOffers != nil
	})
	before := len(nodeProjections(t, h.engine))
	r := h.b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.EngineBindings = nil
	r.mu.Unlock()
	h.b.refreshStrictAdvertisement()
	projections := waitProjections(t, h.engine, before+1)
	if got := projections[len(projections)-1]; got.EngineBindingOffers != nil || got.StrictAdmission != nil {
		t.Fatalf("clearing skipped withdrawal or invented strict readiness: %+v", got)
	}
	if got := h.b.regCache.Snapshot(); len(got) != 0 {
		t.Fatalf("clearing retained cold service: %+v", got)
	}
}
