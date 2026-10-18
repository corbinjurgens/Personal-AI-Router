// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// A Descriptor is the sanitized, versioned projection of one engine manifest
// that engine-manager publishes (engine:descriptors). It is what a process
// other than engine-manager may know about an engine: identity, the platforms
// it is defined for, the protocols it speaks, the model operations it supports
// and the coded adapters its family selects.
//
// It deliberately has no field that could carry an executable, a command, an
// environment value, a filesystem path or a credential. Those stay in the
// manifest, which only engine-manager reads. A new field must keep that
// property; the descriptor tests in engine-manager check it against manifests
// that contain all of them.
//
// Engine is the stable engine ID — the key of every engine-manager call, the
// install directory name, and the key of model attribution maps. Family names
// the engine's lineage (a llama.cpp fork has family "llamacpp") and selects the
// shared adapters; it is never a second identity, and two engines of one family
// are still independent engines.
type Descriptor struct {
	// Version is DescriptorVersion. A consumer rejects a descriptor whose
	// version it does not know rather than guessing at its fields.
	Version int    `json:"version"`
	Engine  string `json:"engine"`
	Family  string `json:"family"`
	Label   string `json:"label"`

	// Platforms lists the "<goos>/<goarch>" keys the manifest defines, sorted.
	// Available reports whether the publishing host is one of them.
	Platforms []string `json:"platforms"`
	Available bool     `json:"available"`

	// Protocols are the protocol profiles the engine serves, from Protocols().
	// An engine that declares none claims no inference surface.
	Protocols []string `json:"protocols"`

	// ModelNaming is how the engine spells model identifiers, from
	// ModelNamings().
	ModelNaming string `json:"modelNaming"`

	// ModelStore and Catalog name the coded on-disk store adapter (model copy)
	// and download catalogue the engine's family selects. Empty means the
	// engine has none.
	ModelStore string `json:"modelStore,omitempty"`
	Catalog    string `json:"catalog,omitempty"`

	// Operations holds the semantic model operations the engine supports,
	// keyed by an Operations() name. An absent key is an unsupported operation.
	Operations map[string]OperationCapability `json:"operations"`
}

// OperationCapability is what a client needs to know about one supported model
// operation. How the operation maps onto engine actions stays inside
// engine-manager.
type OperationCapability struct {
	// ResponseTimeoutS is how long, in seconds, the engine may take to begin
	// answering the operation. Zero means the ordinary engine budget.
	ResponseTimeoutS int `json:"responseTimeoutS,omitempty"`
}

// DescriptorVersion is the descriptor shape this package defines.
const DescriptorVersion = 1

// The semantic model operations. Clients ask engine-manager for one of these
// rather than naming an engine's own action and parameter spelling.
const (
	OperationLoad   = "load"
	OperationUnload = "unload"
	OperationPull   = "pull"
	OperationDelete = "delete"
)

// MaxOperationResponseTimeoutS bounds a declared operation response budget. It
// equals engine-manager's ceiling on any single action, which a budget cannot
// usefully exceed.
const MaxOperationResponseTimeoutS = 30 * 60

// The protocol profiles an engine can declare. Each names a finite, coded
// request surface; declaring one is a claim the engine serves it, not a
// request for generic forwarding.
const (
	// ProtocolOllamaAPI is Ollama's native /api surface: generate, chat,
	// embeddings and the /api/tags model list.
	ProtocolOllamaAPI = "ollama-api"
	// ProtocolOpenAIV1 is the OpenAI-compatible chat, completions and
	// embeddings routes with the /v1/models list.
	ProtocolOpenAIV1 = "openai-v1"
	// ProtocolOpenAIChat is the OpenAI-compatible /v1/chat/completions route
	// with /v1/models, without legacy completions or embeddings.
	ProtocolOpenAIChat = "openai-chat"
	// ProtocolAnthropicMessages is the Anthropic-compatible /v1/messages route.
	ProtocolAnthropicMessages = "anthropic-messages"
	// ProtocolLlamaCPPRouter is llama.cpp's router model list at /models,
	// which also answers /v1/models.
	ProtocolLlamaCPPRouter = "llamacpp-router"
)

// The model naming rules an engine can declare.
const (
	// ModelNamingExact: a model is named by exactly the identifier the engine
	// lists.
	ModelNamingExact = "exact"
	// ModelNamingImpliedLatestTag: an untagged name means its ":latest" tag,
	// as in Ollama.
	ModelNamingImpliedLatestTag = "implied-latest-tag"
)

var (
	operations   = []string{OperationLoad, OperationUnload, OperationPull, OperationDelete}
	protocols    = []string{ProtocolOllamaAPI, ProtocolOpenAIV1, ProtocolOpenAIChat, ProtocolAnthropicMessages, ProtocolLlamaCPPRouter}
	modelNamings = []string{ModelNamingExact, ModelNamingImpliedLatestTag}
)

// Operations returns the semantic model operation names.
func Operations() []string { return slices.Clone(operations) }

// Protocols returns the protocol profile names.
func Protocols() []string { return slices.Clone(protocols) }

// ModelNamings returns the model naming rule names.
func ModelNamings() []string { return slices.Clone(modelNamings) }

// validIDRe is the engine ID syntax. An ID is a filesystem path component (the
// per-engine install directory), so it must not contain a separator; "." and
// ".." are rejected separately.
var validIDRe = regexp.MustCompile(`^[a-zA-Z0-9._-]+$`)

// ValidID reports whether id is usable as an engine ID or family name.
func ValidID(id string) bool {
	return validIDRe.MatchString(id) && id != "." && id != ".."
}

// Validate checks a descriptor received across a process boundary. A consumer
// must not act on one that fails: an unknown protocol, operation or version is
// something it cannot honor.
func (d Descriptor) Validate() error {
	if d.Version != DescriptorVersion {
		return fmt.Errorf("descriptor version %d is not supported (want %d)", d.Version, DescriptorVersion)
	}
	if !ValidID(d.Engine) {
		return fmt.Errorf("descriptor engine %q is not a valid engine id", d.Engine)
	}
	if !ValidID(d.Family) {
		return fmt.Errorf("engine %q: family %q is not a valid family name", d.Engine, d.Family)
	}
	if strings.TrimSpace(d.Label) == "" {
		return fmt.Errorf("engine %q: label is required", d.Engine)
	}
	for _, key := range d.Platforms {
		goos, goarch, ok := strings.Cut(key, "/")
		if !ok || goos == "" || goarch == "" || strings.Contains(goarch, "/") {
			return fmt.Errorf("engine %q: platform %q is not \"<goos>/<goarch>\"", d.Engine, key)
		}
	}
	seen := make(map[string]bool, len(d.Protocols))
	for _, p := range d.Protocols {
		if !slices.Contains(protocols, p) {
			return fmt.Errorf("engine %q: unknown protocol %q", d.Engine, p)
		}
		if seen[p] {
			return fmt.Errorf("engine %q: protocol %q is listed twice", d.Engine, p)
		}
		seen[p] = true
	}
	if !slices.Contains(modelNamings, d.ModelNaming) {
		return fmt.Errorf("engine %q: unknown model naming %q", d.Engine, d.ModelNaming)
	}
	for _, adapter := range []string{d.ModelStore, d.Catalog} {
		if adapter != "" && !ValidID(adapter) {
			return fmt.Errorf("engine %q: adapter name %q is not valid", d.Engine, adapter)
		}
	}
	for op, capability := range d.Operations {
		if !slices.Contains(operations, op) {
			return fmt.Errorf("engine %q: unknown operation %q", d.Engine, op)
		}
		if capability.ResponseTimeoutS < 0 || capability.ResponseTimeoutS > MaxOperationResponseTimeoutS {
			return fmt.Errorf("engine %q: operation %q response budget %ds is outside 0-%d", d.Engine, op, capability.ResponseTimeoutS, MaxOperationResponseTimeoutS)
		}
	}
	return nil
}
