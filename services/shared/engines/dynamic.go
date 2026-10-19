// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

import (
	"fmt"
	"slices"
	"strings"

	"nvpair-shared/noderec/servicekey"
)

// A dynamic engine is a descriptor-backed engine that is not built in: a
// llama.cpp fork registered as "llamacpp-custom", or a user manifest such as
// Strata. Its cross-process identity is derived from its ID alone, so adding
// one needs no source change anywhere:
//
//   - its discovery service key is "e-<id>", which no built-in key and no
//     other service can take, because every other key is two letters;
//   - its facade address inside the proxy process is "e-<id>:<method>", so it
//     cannot be mistaken for a process-scoped namespace such as "errors:" or
//     "discovery:" whatever the ID spells;
//   - its persisted facade port lives in "<id>-proxy-port.json".
//
// A dynamic engine has no stock client port. Its facade port is planned by the
// broker from DynamicFacadePortBase and then persisted by the proxy, so it
// stays where it first landed.
//
// Dynamic service keys are carried only over the authenticated peer-services
// path, never in the mDNS TXT record, and reverse transports admit one only
// while the node's broker has it registered.

// MaxDynamicIDLen bounds a dynamic engine ID. The bound keeps "e-<id>" inside
// the reverse transport's stream header ("PAIR-REVERSE/1 e-<id>\n" is under 64
// bytes) and well inside every other carrier of a service key.
const MaxDynamicIDLen = 40

// MaxDynamicEngines bounds how many dynamic engines one node fronts and
// advertises. A peer's service descriptor may carry at most this many dynamic
// service keys beside the fixed ones.
const MaxDynamicEngines = 16

// DynamicFacadePortBase is where the broker begins its search for a dynamic
// engine's facade port when the proxy has none persisted for it. It is clear
// of every built-in facade and backend port and of PAIR's service ports.
const DynamicFacadePortBase = 18180

// dynamicKeyPrefix begins every dynamic service key and facade address.
const dynamicKeyPrefix = "e-"

// facadeProtocols are the protocol profiles nvpair-proxy has coded route
// classifications for. A protocol outside this set cannot be fronted, so a
// descriptor declaring one fails closed rather than being forwarded as
// unclassified traffic.
var facadeProtocols = []string{ProtocolOllamaAPI, ProtocolOpenAIV1, ProtocolOpenAIChat, ProtocolAnthropicMessages, ProtocolLlamaCPPRouter}

// facadeBaseProtocols each give a facade both an inference route and a model
// list. A facade needs at least one; the other facade protocols only add
// routes beside it.
var facadeBaseProtocols = []string{ProtocolOllamaAPI, ProtocolOpenAIV1, ProtocolOpenAIChat}

// FacadeProtocols returns the protocol profiles nvpair-proxy can front.
func FacadeProtocols() []string { return slices.Clone(facadeProtocols) }

// ValidDynamicID reports whether id can name a dynamic engine: a valid engine
// ID within MaxDynamicIDLen that is not a built-in name.
func ValidDynamicID(id string) bool {
	if _, builtin := ByName(id); builtin {
		return false
	}
	return len(id) <= MaxDynamicIDLen && ValidID(id)
}

// ValidRoutingID reports whether id may name an engine in routing, policy,
// admission and scheduling state: a built-in name or a valid dynamic ID. It is
// a syntax rule only. Whether this node can serve the engine is decided by its
// facades, so a policy may name an engine that only a peer runs.
func ValidRoutingID(id string) bool {
	_, builtin := ByName(id)
	return builtin || ValidDynamicID(id)
}

// DynamicEngine derives the cross-process identity of a dynamic engine. Label
// becomes DisplayName and falls back to the ID. It reports false for an ID
// ValidDynamicID rejects, including every built-in name.
func DynamicEngine(id, label string) (Engine, bool) {
	if !ValidDynamicID(id) {
		return Engine{}, false
	}
	if strings.TrimSpace(label) == "" {
		label = id
	}
	return Engine{
		Name:             id,
		DisplayName:      label,
		DiscoveryService: servicekey.Key(dynamicKeyPrefix + id),
		PortFile:         id + "-proxy-port.json",
	}, true
}

// DynamicEngineForServiceKey returns the dynamic engine ID a service key
// names, if it is a well-formed dynamic key.
func DynamicEngineForServiceKey(key servicekey.Key) (string, bool) {
	id, ok := strings.CutPrefix(string(key), dynamicKeyPrefix)
	if !ok || !ValidDynamicID(id) {
		return "", false
	}
	return id, true
}

// IsDynamicServiceKey reports whether key is a well-formed dynamic engine
// service key. Syntax alone admits nothing: a transport still requires the
// owning node to have the service registered.
func IsDynamicServiceKey(key servicekey.Key) bool {
	_, ok := DynamicEngineForServiceKey(key)
	return ok
}

// ProxyCanFront is the single rule for whether nvpair-proxy can front the
// engine d describes — serve it on a facade and route it to peers that
// advertise it. A nil error means it can. The broker applies it before it
// creates a dynamic facade, the proxy applies it before it builds one, and an
// engine switch applies it before it accepts a target.
//
// A built-in engine is always frontable: its facade is fixed and ignores the
// descriptor's protocol list. Any other engine needs a valid bounded dynamic
// ID, only protocols the proxy has coded routes for, and at least one protocol
// that supplies both an inference route and a model list. A descriptor that
// fails Validate is never frontable.
func ProxyCanFront(d Descriptor) error {
	if err := d.Validate(); err != nil {
		return err
	}
	if _, builtin := ByName(d.Engine); builtin {
		return nil
	}
	if !ValidDynamicID(d.Engine) {
		return fmt.Errorf("engine %q: a descriptor engine id must be at most %d characters", d.Engine, MaxDynamicIDLen)
	}
	base := false
	for _, p := range d.Protocols {
		if !slices.Contains(facadeProtocols, p) {
			return fmt.Errorf("engine %q: nvpair-proxy cannot front protocol %q", d.Engine, p)
		}
		if slices.Contains(facadeBaseProtocols, p) {
			base = true
		}
	}
	if !base {
		return fmt.Errorf("engine %q declares no protocol a facade can serve; want one of %v", d.Engine, facadeBaseProtocols)
	}
	return nil
}
