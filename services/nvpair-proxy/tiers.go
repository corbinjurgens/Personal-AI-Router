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
	"net/http"
	"slices"
	"sort"
	"strings"

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
// a tier request from the tier's local entries; API entries are skipped until
// hybrid dispatch exists. Within an entry, candidates keep resolveCandidates'
// order (node/select pin, scheduler priority, ID); across entries the policy's
// preferLoaded decides whether tier rank or warmth leads.
func (f *facade) resolveTierCandidates(tier string, needs requestNeeds, crossEngine bool) []candidate {
	p := f.host
	pol := p.admission.currentPolicy()
	var out []candidate
	seen := map[string]bool{}
	for rank, t := range tierSearchOrder(tier, pol.TierPolicy) {
		for _, e := range pol.Tiers[t] {
			if !e.IsLocal() || !entryEligible(e, needs) {
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

// tierModelEntries appends the synthetic alias records to a merged /v1/models
// list, so a client can pick a tier from it. An upstream model already named
// like an alias is left alone.
func (f *facade) tierModelEntries(seen map[string]modelListItem) []json.RawMessage {
	return hybridAliasRecords(f.host.admission.currentPolicy(), seen)
}

// configuredAliases lists the reserved hybrid alias names a policy offers:
// every tier with at least one entry of any kind, each as the bare tier and
// both variants, weakest tier first. A tier whose entries are all API
// deployments is listed like any other. It reads only the policy, so a
// listener with no facade and no upstream can serve it.
func configuredAliases(pol nodepolicy.Policy) []string {
	var out []string
	for _, t := range nodepolicy.TierOrder {
		if len(pol.Tiers[t]) == 0 {
			continue
		}
		out = append(out, t, t+"-api", t+"-local")
	}
	return out
}

// hybridAliasRecords returns one synthetic OpenAI model record per configured
// alias, skipping any id present in skip (which may be nil). The record names
// no policy detail: never a credential, account, endpoint or deployment.
func hybridAliasRecords(pol nodepolicy.Policy, skip map[string]modelListItem) []json.RawMessage {
	var out []json.RawMessage
	for _, id := range configuredAliases(pol) {
		if _, dup := skip[id]; dup {
			continue
		}
		raw, err := json.Marshal(map[string]any{
			"id":       id,
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

// Reasons a request naming a reserved alias is refused. Both answer 400 with
// code "feature_unavailable".
const (
	// aliasReasonNotInferenceReady: the -api and -local aliases have no
	// dispatch path until the hybrid cutover.
	aliasReasonNotInferenceReady = "alias_not_inference_ready"
	// aliasReasonUnsupportedUse: a bare tier named where only a plain tier
	// request on an inference route is supported, or not spelled exactly.
	aliasReasonUnsupportedUse = "unsupported_alias_use"
)

// aliasRejection decides, from the raw requested model, whether a request must
// be refused before any admission, wake, reservation, workload or upstream
// request. It runs on the raw name, before any engine normalization, and is
// the only place that decides it.
//
// Refused: a -api or -local alias on any route; a bare tier on a route that is
// not a classified inference route (the proxy would forward the alias to the
// engine as if it were a model); and a bare tier with surrounding whitespace,
// which the legacy tier path never matched. Allowed: a bare tier exactly
// spelled on a classified inference route, which keeps the legacy local-only
// tier path.
func aliasRejection(model string, inference bool) (reason string, rejected bool) {
	trimmed := strings.TrimSpace(model)
	_, mode, ok := nodepolicy.ParseAlias(trimmed)
	switch {
	case !ok:
		return "", false
	case mode != nodepolicy.AliasBare:
		return aliasReasonNotInferenceReady, true
	case trimmed != model, !inference:
		return aliasReasonUnsupportedUse, true
	}
	return "", false
}

// modelDetailPrefixes are the path prefixes whose remainder names one model, as
// in LM Studio's /v1/models/<id> and /api/v0/models/<id>.
var modelDetailPrefixes = []string{"/v1/models/", "/api/v0/models/"}

// requestAliasRejection is the refusal decision for one request. It applies
// aliasRejection to the body's "model" as the router reads it, and refuses a
// -api or -local alias in any other place an engine could read a model name
// from, because an engine may read a different one than the router does:
//   - every top-level JSON key equal under strings.EqualFold to "model" or
//     "name", counting repeated keys (Ollama's /api/pull, /api/show,
//     /api/delete and /api/create read "name");
//   - every "model" query parameter (llama.cpp's GET /props?model=...);
//   - the remainder of a model-detail path (modelDetailPrefixes).
//
// A bare tier in those places keeps its legacy routing. alias is the name to
// report. The cluster ingress is not covered; it has its own listener.
func requestAliasRejection(r *http.Request, body []byte, model string, inference bool) (alias, reason string, rejected bool) {
	if reason, rejected := aliasRejection(model, inference); rejected {
		return strings.TrimSpace(model), reason, true
	}
	names := jsonModelNames(body)
	names = append(names, r.URL.Query()["model"]...)
	for _, prefix := range modelDetailPrefixes {
		if rest, ok := strings.CutPrefix(r.URL.Path, prefix); ok {
			names = append(names, rest)
		}
	}
	for _, name := range names {
		name = strings.TrimSpace(name)
		if _, mode, ok := nodepolicy.ParseAlias(name); ok && mode != nodepolicy.AliasBare {
			return name, aliasReasonNotInferenceReady, true
		}
	}
	return "", "", false
}

// jsonModelNames returns the string values of every top-level key of a JSON
// object body that equals "model" or "name" under strings.EqualFold. A body
// that is not a JSON object yields none.
func jsonModelNames(body []byte) []string {
	dec := json.NewDecoder(bytes.NewReader(body))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return nil
	}
	var out []string
	for dec.More() {
		tok, err := dec.Token()
		key, isString := tok.(string)
		if err != nil || !isString {
			return out
		}
		var raw json.RawMessage
		if dec.Decode(&raw) != nil {
			return out
		}
		if !strings.EqualFold(key, "model") && !strings.EqualFold(key, "name") {
			continue
		}
		var value string
		if json.Unmarshal(raw, &value) == nil {
			out = append(out, value)
		}
	}
	return out
}

// writeAliasRejection answers a refused alias request with a structured 400.
func writeAliasRejection(w http.ResponseWriter, alias, reason string) {
	body, err := json.Marshal(struct {
		Error  string `json:"error"`
		Code   string `json:"code"`
		Reason string `json:"reason"`
		Model  string `json:"model"`
	}{
		Error:  "hybrid routing alias is not available on this request",
		Code:   "feature_unavailable",
		Reason: reason,
		Model:  strings.TrimSpace(alias),
	})
	if err != nil {
		body = []byte(`{"error":"hybrid routing alias is not available on this request","code":"feature_unavailable"}`)
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(http.StatusBadRequest)
	_, _ = w.Write(body)
}
