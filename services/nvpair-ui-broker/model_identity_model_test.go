// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

func TestIdentityCurrentModel(t *testing.T) {
	otherDigest := strings.Repeat("e", 64)
	test := func(name, engine, model string, current map[string]string, wantReady bool) {
		t.Run(name, func(t *testing.T) {
			f := newStrictFixture(t)
			capability := strictCapability(t, engine, model)
			r := f.b.nodePolicyRuntime()
			r.mu.Lock()
			r.engineIdentity[engine] = engineManagerIdentity{
				EngineRunID: "engine-run-1", LaunchDigest: launchDigestC,
				ModelDigests: map[string]string{model: modelDigestB},
			}
			r.settingsIdentity[engine] = capability.Models[0].Identity.SettingsFingerprint
			r.mu.Unlock()
			f.advertise(t, capability)

			// Change current inventory while retaining the proxy's held report.
			r.mu.Lock()
			id := r.engineIdentity[engine]
			id.ModelDigests = current
			r.engineIdentity[engine] = id
			joined := r.modelIdentityLocked().Engines[engine]
			r.mu.Unlock()
			if err := joined.Validate(); err != nil {
				t.Fatalf("current identity fixture is invalid: %v", err)
			}
			var want *nodepolicy.StrictAdmissionCapability
			if wantReady {
				want = capability
			}
			if got := f.b.StrictReadiness().Capability; !reflect.DeepEqual(got, want) {
				t.Fatalf("direct readiness = %+v, want %+v", got, want)
			}
			f.b.publishNodeProjection()
			waitAdvertised(t, f, want)
		})
	}
	test("replaced tag cannot borrow another tag digest", "llamacpp", "a:v1", map[string]string{"a:v1": otherDigest, "a:v2": modelDigestB}, false)
	test("unchanged tag with repeated digest", "llamacpp", "a:v2", map[string]string{"a:v1": modelDigestB, "a:v2": modelDigestB}, true)
	test("missing requested model", "llamacpp", "a:v1", map[string]string{"a:v2": modelDigestB}, false)
	test("exact names do not imply latest", "llamacpp", "a", map[string]string{"a:latest": modelDigestB}, false)
	test("exact names trim whitespace", "llamacpp", "org/a", map[string]string{" org/a ": modelDigestB}, true)
	test("reserved alias stays exempt", "llamacpp", "weak:latest", map[string]string{"weak": modelDigestB}, false)
}

func TestIdentityCurrentModelDescriptorSelection(t *testing.T) {
	test := func(name, engine, naming string, install bool, digestModel string, wantReady bool) {
		t.Run(name, func(t *testing.T) {
			f := newStrictFixture(t)
			if install {
				d := testDescriptor(engine, engines.ProtocolOpenAIV1)
				d.ModelNaming = naming
				f.b.setDynamicEngines([]engines.Descriptor{d}, func(int) bool { return true })
			}
			seedStrictIdentity(f.b, engine, digestModel)
			capability := strictCapability(t, engine, "m")
			reportStrict(t, f.b, "0", f.acks.runID, capability)
			var want *nodepolicy.StrictAdmissionCapability
			if wantReady {
				want = capability
			}
			if got := f.b.StrictReadiness().Capability; !reflect.DeepEqual(got, want) {
				t.Fatalf("direct readiness = %+v, want %+v", got, want)
			}
			f.b.publishNodeProjection()
			waitAdvertised(t, f, want)
		})
	}
	test("dynamic exact", "custom-exact", engines.ModelNamingExact, true, "m", true)
	test("dynamic exact does not imply latest", "custom-exact", engines.ModelNamingExact, true, "m:latest", false)
	test("unknown engine", "unknown-engine", "", false, "m", false)
	test("unknown descriptor naming", "custom-unknown", "unknown", true, "m", false)
}
