// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Tier routing (FORK_DESIGN.md §3.7): a request whose model is "weak",
// "medium" or "strong" is resolved by the node it enters, to a concrete
// (node, engine, model) chosen from the policy's tier entries.
//
// An OpenAI-compatible route can be served by any engine, so its candidates
// may cross engines; a native route resolves only to its own engine. A
// candidate on another engine is built by that engine's facade in this
// process, from that facade's own discovery overlay: the overlay's node port is
// the peer's advertised service port for that engine, which is the peer's
// facade (proxy) port for it, and its self candidate is that engine's local
// backend. An engine with no facade enabled here therefore contributes no
// candidates — this process has no routing view of it.

import (
	"bytes"
	"encoding/json"
	"slices"
	"sort"

	"nvpair-shared/nodepolicy"
)

// openAICompatibleInference lists the routes any engine can serve, which is
// what lets a tier request on them cross engines.
var openAICompatibleInference = map[string]bool{
	"/v1/chat/completions": true,
	"/v1/completions":      true,
	"/v1/embeddings":       true,
}

// embeddingsRoutes are the inference routes that need an embeddings model.
var embeddingsRoutes = map[string]bool{
	"/v1/embeddings":  true,
	"/api/embed":      true,
	"/api/embeddings": true,
}

// requestNeeds is what a request requires of the model that answers it.
type requestNeeds struct {
	caps          []nodepolicy.Capability
	contextTokens int
}

// names lists the needed capabilities for the no-eligible-model answer.
func (n requestNeeds) names() []string {
	out := make([]string, 0, len(n.caps))
	for _, c := range n.caps {
		out = append(out, string(c))
	}
	return out
}

// parseRequestNeeds reads a buffered request, in either the OpenAI or the
// Ollama shape, for the capabilities and context it needs. A body that is not
// JSON needs nothing beyond its size.
func parseRequestNeeds(body []byte, path string) requestNeeds {
	var probe struct {
		Tools     json.RawMessage `json:"tools"`
		Functions json.RawMessage `json:"functions"`
		Messages  []struct {
			Content json.RawMessage   `json:"content"`
			Images  []json.RawMessage `json:"images"`
		} `json:"messages"`
		Images         []json.RawMessage `json:"images"`
		ResponseFormat *struct {
			Type string `json:"type"`
		} `json:"response_format"`
		Format              json.RawMessage `json:"format"`
		MaxTokens           *int            `json:"max_tokens"`
		MaxCompletionTokens *int            `json:"max_completion_tokens"`
		Options             *struct {
			NumPredict *int `json:"num_predict"`
		} `json:"options"`
	}
	needs := requestNeeds{contextTokens: len(body) / 4}
	_ = json.Unmarshal(body, &probe)

	has := map[nodepolicy.Capability]bool{}
	if nonEmptyArray(probe.Tools) || nonEmptyArray(probe.Functions) {
		has[nodepolicy.CapTools] = true
	}
	if len(probe.Images) > 0 {
		has[nodepolicy.CapVision] = true
	}
	for _, m := range probe.Messages {
		if len(m.Images) > 0 || contentHasImage(m.Content) {
			has[nodepolicy.CapVision] = true
			break
		}
	}
	if probe.ResponseFormat != nil && (probe.ResponseFormat.Type == "json_schema" || probe.ResponseFormat.Type == "json_object") {
		has[nodepolicy.CapStructured] = true
	}
	if f := bytes.TrimSpace(probe.Format); len(f) > 0 && !bytes.Equal(f, []byte("null")) && !bytes.Equal(f, []byte(`""`)) {
		has[nodepolicy.CapStructured] = true
	}
	if embeddingsRoutes[path] {
		has[nodepolicy.CapEmbeddings] = true
	}
	for _, c := range nodepolicy.KnownCapabilities {
		if has[c] {
			needs.caps = append(needs.caps, c)
		}
	}

	switch {
	case probe.MaxTokens != nil && *probe.MaxTokens > 0:
		needs.contextTokens += *probe.MaxTokens
	case probe.MaxCompletionTokens != nil && *probe.MaxCompletionTokens > 0:
		needs.contextTokens += *probe.MaxCompletionTokens
	case probe.Options != nil && probe.Options.NumPredict != nil && *probe.Options.NumPredict > 0:
		needs.contextTokens += *probe.Options.NumPredict
	}
	return needs
}

func nonEmptyArray(raw json.RawMessage) bool {
	var arr []json.RawMessage
	return json.Unmarshal(raw, &arr) == nil && len(arr) > 0
}

// contentHasImage reports whether an OpenAI message content array carries an
// image part. A string content never does.
func contentHasImage(raw json.RawMessage) bool {
	var parts []struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return false
	}
	for _, p := range parts {
		if p.Type == "image_url" || p.Type == "image" || p.Type == "input_image" {
			return true
		}
	}
	return false
}

// tierSearchOrder is the order tiers are searched in for a request: the
// requested tier, then stronger tiers nearest first when allowed, then weaker
// tiers nearest first when allowed.
func tierSearchOrder(tier string, tp nodepolicy.TierPolicy) []string {
	i := slices.Index(nodepolicy.TierOrder, tier)
	if i < 0 {
		return nil
	}
	order := []string{tier}
	if tp.AllowStronger {
		order = append(order, nodepolicy.TierOrder[i+1:]...)
	}
	if tp.AllowWeaker {
		for j := i - 1; j >= 0; j-- {
			order = append(order, nodepolicy.TierOrder[j])
		}
	}
	return order
}

// entryEligible reports whether a tier entry can answer a request with needs.
func entryEligible(e nodepolicy.TierEntry, needs requestNeeds) bool {
	for _, c := range needs.caps {
		if !slices.Contains(e.Capabilities, c) {
			return false
		}
	}
	return e.ContextTokens == 0 || e.ContextTokens >= needs.contextTokens
}

// resolveTierCandidates builds the ordered (node, engine, model) candidates for
// a tier request. Within an entry, candidates keep resolveCandidates' order
// (node/select pin, scheduler priority, ID); across entries the policy's
// preferLoaded decides whether tier rank or warmth leads.
func (f *facade) resolveTierCandidates(tier string, needs requestNeeds, crossEngine bool) []candidate {
	p := f.host
	pol := p.admission.currentPolicy()
	var out []candidate
	seen := map[string]bool{}
	for rank, t := range tierSearchOrder(tier, pol.TierPolicy) {
		for _, e := range pol.Tiers[t] {
			if !entryEligible(e, needs) {
				continue
			}
			g := f
			if e.Engine != f.profile.Name {
				if !crossEngine {
					continue
				}
				if g = p.facadeFor(e.Engine); g == nil {
					continue
				}
			}
			for _, c := range g.resolveCandidates(e.Model) {
				key := c.id + "\x00" + c.engine + "\x00" + g.profile.normalizeModel(e.Model)
				if seen[key] {
					continue
				}
				seen[key] = true
				c.model = e.Model
				c.tierRank = rank
				out = append(out, c)
			}
		}
	}
	anyLoaded := pol.TierPolicy.PreferLoaded == "any"
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if anyLoaded && a.warm != b.warm {
			return a.warm
		}
		if a.tierRank != b.tierRank {
			return a.tierRank < b.tierRank
		}
		if a.warm != b.warm {
			return a.warm
		}
		return false
	})
	return out
}

// reserveRound takes this request's reservation for a fresh round. A tier
// round reserves only among the candidates that share the leading class —
// tier rank and warmth — so load balancing never promotes a fallback tier or
// a cold model over the class the policy prefers.
func (f *facade) reserveRound(cands []candidate, tier bool) ([]candidate, reservation) {
	if !tier || len(cands) == 0 {
		return f.host.reserveCandidate(f, cands)
	}
	n := 1
	for n < len(cands) && cands[n].tierRank == cands[0].tierRank && cands[n].warm == cands[0].warm {
		n++
	}
	_, held := f.host.reserveCandidate(f, cands[:n])
	return cands, held
}

// tierModelEntries appends one synthetic OpenAI model record per configured
// tier to a merged /v1/models list, so a client can pick a tier from it. A
// tier with no entries is not offered, and an upstream model already named
// like a tier is left alone.
func (f *facade) tierModelEntries(seen map[string]modelListItem) []json.RawMessage {
	pol := f.host.admission.currentPolicy()
	var out []json.RawMessage
	for _, t := range nodepolicy.TierOrder {
		if len(pol.Tiers[t]) == 0 {
			continue
		}
		if _, dup := seen[t]; dup {
			continue
		}
		raw, err := json.Marshal(map[string]any{
			"id":       t,
			"object":   "model",
			"created":  0,
			"owned_by": "pair",
		})
		if err != nil {
			continue
		}
		out = append(out, raw)
	}
	return out
}
