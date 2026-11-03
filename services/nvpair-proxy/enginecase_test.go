// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// The harness the merged suite runs each engine through.
//
// The two proxies previously had parallel suites that drifted: same-named
// tests with different bodies, and behaviors covered on one side only. A
// single table means a case is either declared to apply to both engines or
// explicitly scoped to one, so a gap has to be a decision rather than an
// oversight.
//
// engineCase carries the fixtures a body cannot derive from the profile — the
// paths a client of that engine actually calls, and the shapes its upstream
// returns.

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"nvpair-shared/engines"
)

type engineCase struct {
	// profile is the engine under test.
	profile engineProfile

	// inferencePath is a path this engine treats as inference, used by bodies
	// that need to drive a routed request.
	inferencePath string

	// nonInferencePath is a real path this engine's clients call that is
	// neither inference nor a model list, so the proxy forwards it verbatim.
	// Bodies use it to prove that model-eligibility filtering and 404
	// failover apply to inference alone. It must stay absent from Routes.
	nonInferencePath string

	// modelListPath is the path whose response the federated list serves.
	modelListPath string

	// emptyModelList is the dialect's empty envelope. It is both what an
	// upstream with no models returns and what the federated list re-emits,
	// which is the assertion: an engine's client must see its own dialect's
	// empty shape rather than a bare 200 or the other engine's.
	emptyModelList string

	// advertisedModel is how a node's discovery inventory spells the model a
	// client reaches with requestedModel. It matches
	// requestedModel exactly.
	advertisedModel string

	// requestedModel is what a client asks for.
	requestedModel string

	// descriptor is the facade/enable descriptor of an engine that is not built
	// in; nil for a built-in engine.
	descriptor *engines.Descriptor
}

func llamacppCase(t *testing.T) engineCase {
	t.Helper()
	p, ok := profileFor(engines.NameLlamaCPP)
	if !ok {
		t.Fatal("llamacpp profile missing")
	}
	return engineCase{
		profile:       p,
		inferencePath: "/v1/chat/completions",
		// llama.cpp's own server-properties endpoint, unlisted in Routes and so
		// forwarded verbatim.
		nonInferencePath: "/props",
		modelListPath:    "/v1/models",
		emptyModelList:   `{"object":"list","data":[]}`,
		advertisedModel:  "qwen3",
		requestedModel:   "qwen3",
	}
}

// customCase is a descriptor engine (a llama.cpp fork registered by manifest)
// that declares the OpenAI v1 and Anthropic messages protocols, so unlike
// llamacpp its model list is served from the upstream's own /v1/models and it
// fronts /v1/messages.
func customCase(t *testing.T) engineCase {
	t.Helper()
	d := customDescriptor()
	d.Protocols = []string{engines.ProtocolOpenAIV1, engines.ProtocolAnthropicMessages}
	p, err := dynamicProfile(d)
	if err != nil {
		t.Fatalf("custom engine profile: %v", err)
	}
	return engineCase{
		profile:          p,
		inferencePath:    "/v1/chat/completions",
		nonInferencePath: "/props",
		modelListPath:    "/v1/models",
		emptyModelList:   `{"object":"list","data":[]}`,
		advertisedModel:  "qwen3-8b",
		requestedModel:   "qwen3-8b",
		descriptor:       &d,
	}
}

// inferenceBody is the request body a client sends for requestedModel. It is
// the one shape both dialects share: a top-level "model" field, which is all
// bufferBodyAndModel reads.
func (tc engineCase) inferenceBody() string {
	return fmt.Sprintf(`{"model":%q}`, tc.requestedModel)
}

// inferenceRequest is a routed inference call for requestedModel, the request
// most bodies in the failover suite need.
func (tc engineCase) inferenceRequest() *http.Request {
	return httptest.NewRequest(http.MethodPost, tc.inferencePath, strings.NewReader(tc.inferenceBody()))
}

// engineCases is every engine the merged suite exercises. A body that runs
// under this is asserting behavior both proxies must share.
func engineCases(t *testing.T) []engineCase {
	t.Helper()
	return []engineCase{llamacppCase(t), customCase(t)}
}

// forEachEngine runs body as a subtest per engine.
func forEachEngine(t *testing.T, body func(t *testing.T, tc engineCase)) {
	t.Helper()
	for _, tc := range engineCases(t) {
		t.Run(tc.profile.Name, func(t *testing.T) { body(t, tc) })
	}
}

// anyProfile is for tests whose subject has no engine input at all — transport
// pooling, cluster trust, server timeouts, priority ordering. They need a
// profile only because NewProxy takes one, and running them per engine would
// double the runtime while proving nothing, since there is no branch for the
// second pass to take. Naming that explicitly keeps them from being mistaken
// for coverage gaps.
func anyProfile(t *testing.T) engineProfile {
	t.Helper()
	return llamacppCase(t).profile
}

// anyCase is anyProfile for a body that still needs an engine's fixtures as a
// vehicle — a routed inference request to reach the streaming path — while its
// subject remains engine-free. The zombie suite is the case in point: it tests
// statusCapture's write and flush deadlines and the terminalOnce guard, none of
// which consult the profile. That the path classifies as inference on both
// engines is already covered by the parameterized workload tests, so a second
// pass here would add real-socket seconds and no coverage.
func anyCase(t *testing.T) engineCase {
	t.Helper()
	return llamacppCase(t)
}

// otherEngine returns an engine that is not tc's, for negative fixtures that
// need to prove this proxy ignores a node running only a sibling engine.
func otherEngine(t *testing.T, tc engineCase) engineProfile {
	t.Helper()
	for _, candidate := range engineCases(t) {
		if candidate.profile.Name != tc.profile.Name {
			return candidate.profile
		}
	}
	t.Fatal("no sibling engine to contrast against")
	return engineProfile{}
}
