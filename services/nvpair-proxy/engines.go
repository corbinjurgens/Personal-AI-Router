// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// This is the only file in the proxy that carries table-driven engine values.
// Everything else reads them from the profile it is handed.
//
// Cross-process identity — Name, DisplayName, DiscoveryService, Namespace,
// FacadePort, EnginePortBase — lives in nvpair-shared/engines, because the
// broker, the scheduler and the TUI need the same values and a second
// declaration here could drift from theirs without failing to compile. What
// remains below is what only the proxy needs to know.
//
// The built-in engines are one fixed entry each here plus one in the broker's
// table. The values that are derivable are derived: the applog component and
// the error-ID prefix both come from Name via ComponentName, so they cannot
// disagree with each other. The persisted-port filename is declared by
// nvpair-shared/engines.
//
// Any other engine is fronted from its descriptor (dynamicProfile): its
// declared protocols select route sets from protocolRoutes, which is the
// proxy's finite, coded protocol registry. Nothing is added here per engine.

import (
	"fmt"
	"slices"
	"strings"

	"nvpair-shared/engines"
)

// routeRole classifies an inbound request path and HTTP method together.
// A POST to /v1/models is not a model list, and a GET to an inference path
// does not emit a workload.
type routeRole int

const (
	// roleInferencePOST is a POST that counts as a cluster workload: it is
	// routed to an owner of the requested model, emits workload lifecycle
	// notifications, and retries a 404 on another owner.
	roleInferencePOST routeRole = iota

	// roleModelListOpenAIGET is the OpenAI /v1/models dialect: the upstream
	// envelope carries a "data" array, each record's identity is its "id",
	// and the federated response is re-emitted as
	// {"object": "list", "data": [...]}.
	roleModelListOpenAIGET
)

// modelNaming is the model identifier convention shared by facade profiles.
type modelNaming int

const (
	// exactID compares identifiers byte for byte.
	exactID modelNaming = iota
)

// route is one classified request path.
type route struct {
	// Path is the client-facing path matched on the facade.
	Path string

	// UpstreamPath optionally rewrites Path for the engine; empty preserves it.
	UpstreamPath string
	Role         routeRole
}

// engineProfile is everything the proxy needs to front one engine.
type engineProfile struct {
	engines.Engine
	Family string

	// StandalonePort is the port the proxy binds when nothing passes --port
	// and no port has been persisted. It must be neither FacadePort (which
	// the engine itself owns) nor a port an engine is relocated onto.
	//
	// Ollama's 11435 currently equals its EnginePortBase, which violates that
	// second half. It is left as-is because it is the value on disk in every
	// existing install; see the plan's port-field note.
	StandalonePort int

	// Routes classifies the paths this engine's clients call. It is NOT an
	// allowlist: an unlisted path is forwarded verbatim, which is how
	// unclassified engine controls and OPTIONS preflights keep working.
	Routes []route

	// ModelNaming is how this engine spells model identifiers.
	ModelNaming modelNaming

	// ReservedPersistedPort is a port that must never be restored from the
	// persisted-port file even if it is stored there, because it belongs to
	// the engine rather than the proxy. Zero means no port is reserved.
	ReservedPersistedPort int
}

// openAIModelListRoutes is the OpenAI-compatible /v1/models list.
var openAIModelListRoutes = []route{
	{Path: "/v1/models", Role: roleModelListOpenAIGET},
}

// llamaCPPBaseRoutes serves both model-list paths through the fleet inventory,
// querying llama.cpp's router endpoint. Its response uses the OpenAI envelope.
var llamaCPPBaseRoutes = []route{
	{Path: "/models", Role: roleModelListOpenAIGET},
	{Path: "/v1/models", UpstreamPath: "/models", Role: roleModelListOpenAIGET},
}

// openAIInferenceRoutes is the OpenAI-compatible inference surface.
var openAIInferenceRoutes = []route{
	{Path: "/v1/chat/completions", Role: roleInferencePOST},
	{Path: "/v1/completions", Role: roleInferencePOST},
	{Path: "/v1/embeddings", Role: roleInferencePOST},
}

// anthropicInferenceRoutes is the Anthropic-compatible inference surface.
var anthropicInferenceRoutes = []route{
	{Path: "/v1/messages", Role: roleInferencePOST},
}

// protocolRoutes is the proxy's protocol registry: the route set each
// descriptor protocol selects for a dynamic facade. It is finite and coded; a
// protocol missing here cannot be fronted. dynamicProfile refuses it before
// a facade is built. TestProtocolRoutesMatchSupportedProtocols pins this registry.
var protocolRoutes = map[string][]route{
	engines.ProtocolOpenAIV1:          slices.Concat(openAIModelListRoutes, openAIInferenceRoutes),
	engines.ProtocolOpenAIChat:        slices.Concat(openAIModelListRoutes, []route{{Path: "/v1/chat/completions", Role: roleInferencePOST}}),
	engines.ProtocolAnthropicMessages: anthropicInferenceRoutes,
	engines.ProtocolLlamaCPPRouter:    llamaCPPBaseRoutes,
}

// dynamicProfile builds the facade profile of a descriptor engine that is not
// built in. It fails closed: a descriptor ProxyCanFront rejects, or a built-in
// engine (whose facade is fixed), has no dynamic profile.
//
// llama.cpp's router list is applied first because it serves /v1/models from
// the router's /models; a later protocol's plain /v1/models must not shadow
// that rewrite. Otherwise routes keep the descriptor's protocol order, and a
// path already classified for its method is not added twice.
func dynamicProfile(d engines.Descriptor) (engineProfile, error) {
	if err := engines.ProxyCanFront(d); err != nil {
		return engineProfile{}, err
	}
	identity, ok := engines.DynamicEngine(d.Engine, d.Label)
	if !ok {
		return engineProfile{}, fmt.Errorf("engine %q is built in; its facade takes no descriptor", d.Engine)
	}
	ordered := slices.Clone(d.Protocols)
	if i := slices.Index(ordered, engines.ProtocolLlamaCPPRouter); i > 0 {
		ordered = slices.Concat([]string{engines.ProtocolLlamaCPPRouter}, slices.Delete(ordered, i, i+1))
	}
	var routes []route
	for _, protocol := range ordered {
		set, ok := protocolRoutes[protocol]
		if !ok {
			return engineProfile{}, fmt.Errorf("engine %q: no routes for protocol %q", d.Engine, protocol)
		}
		for _, r := range set {
			if !slices.ContainsFunc(routes, func(have route) bool {
				return have.Path == r.Path && have.Role.method() == r.Role.method()
			}) {
				routes = append(routes, r)
			}
		}
	}
	return engineProfile{Engine: identity, Family: d.Family, Routes: routes, ModelNaming: exactID}, nil
}

// dynamic reports whether the profile was built from a descriptor rather than
// the built-in table.
func (p engineProfile) dynamic() bool {
	_, builtin := engines.ByName(p.Name)
	return !builtin
}

var profiles = buildProfiles()

func buildProfiles() []engineProfile {
	ollama, _ := engines.ByName("ollama")
	lmstudio, _ := engines.ByName("lmstudio")
	llamacpp, _ := engines.ByName("llamacpp")
	llamaCPPRoutes := slices.Concat(llamaCPPBaseRoutes, openAIInferenceRoutes)

	return []engineProfile{
		{
			Engine:                ollama,
			StandalonePort:        11435,
			Routes:                slices.Concat(openAIModelListRoutes, openAIInferenceRoutes, anthropicInferenceRoutes),
			ModelNaming:           exactID,
			ReservedPersistedPort: 0,
		},
		{
			Engine:                lmstudio,
			StandalonePort:        1234,
			Routes:                slices.Concat(openAIModelListRoutes, openAIInferenceRoutes, anthropicInferenceRoutes),
			ModelNaming:           exactID,
			ReservedPersistedPort: 0,
		},
		{
			Engine:                llamacpp,
			StandalonePort:        8080,
			Routes:                llamaCPPRoutes,
			ModelNaming:           exactID,
			ReservedPersistedPort: 8081,
		},
	}
}

// profileFor resolves the engine named in a facade/enable request.
func profileFor(name string) (engineProfile, bool) {
	for _, p := range profiles {
		if p.Name == name {
			return p, true
		}
	}
	return engineProfile{}, false
}

// engineNames lists the accepted engine ids for a facade/enable rejection.
func engineNames() string {
	names := make([]string, 0, len(profiles))
	for _, p := range profiles {
		names = append(names, p.Name)
	}
	return strings.Join(names, ", ")
}

// routeFor classifies a request. The bool reports whether the path is one this
// engine handles specially; false means forward it verbatim.
func (p engineProfile) routeFor(method, path string) (route, bool) {
	for _, r := range p.Routes {
		// Keep scanning on a method mismatch rather than bailing: a path may
		// legitimately appear twice under different methods, and returning
		// here would make the second entry permanently unreachable — silently
		// forwarded verbatim instead of classified.
		if r.Path != path || r.Role.method() != method {
			continue
		}
		return r, true
	}
	return route{}, false
}

func (p engineProfile) roleFor(method, path string) (routeRole, bool) {
	r, ok := p.routeFor(method, path)
	return r.Role, ok
}

func (r route) upstreamPath() string {
	if r.UpstreamPath != "" {
		return r.UpstreamPath
	}
	return r.Path
}

// method is the HTTP method a role applies to.
func (r routeRole) method() string {
	if r == roleInferencePOST {
		return "POST"
	}
	return "GET"
}

// isModelList reports whether a role serves the federated model list.
func (r routeRole) isModelList() bool {
	return r == roleModelListOpenAIGET
}

// normalizeModel trims surrounding whitespace and otherwise preserves the
// exact identifier used by the engine inventory.
func (p engineProfile) normalizeModel(model string) string {
	return engines.NormalizeModelName(model, engines.ModelNamingExact)
}
