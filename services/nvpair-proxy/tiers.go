// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Tier routing (FORK_DESIGN.md §3.7): a request whose model is "weak",
// "medium" or "strong" is resolved by the node it enters, to a concrete
// (node, engine, model) chosen from the roles nodes declare for themselves and,
// for nodes that declare none, the policy's tier entries.
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
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

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
	needs, _ := parseRequestNeedsOwned(nil, body, path)
	return needs
}

func parseRequestNeedsOwned(ctx context.Context, body []byte, path string) (requestNeeds, error) {
	needs := requestNeeds{contextTokens: len(body) / 4}
	has := map[nodepolicy.Capability]bool{}
	if ownedJSONValid(body) && isJSONObject(body) {
		if nonEmptyArray(requestJSONField(body, "tools")) || nonEmptyArray(requestJSONField(body, "functions")) {
			has[nodepolicy.CapTools] = true
		}
		if nonEmptyArray(requestJSONArrayField(body, "images")) {
			has[nodepolicy.CapVision] = true
		}
		vision, err := requestMessagesHaveImage(ctx, body)
		if err != nil {
			return requestNeeds{}, err
		}
		has[nodepolicy.CapVision] = has[nodepolicy.CapVision] || vision
		typeValue := requestNestedString(body, "response_format", "type")
		if jsonStringEquals(typeValue, "json_schema") || jsonStringEquals(typeValue, "json_object") {
			has[nodepolicy.CapStructured] = true
		}
		if format := bytes.TrimSpace(requestJSONField(body, "format")); len(format) > 0 && !bytes.Equal(format, []byte("null")) && !bytes.Equal(format, []byte(`""`)) {
			has[nodepolicy.CapStructured] = true
		}
		for _, n := range []int{requestStructPositiveInt(body, "max_tokens", 0), requestStructPositiveInt(body, "max_completion_tokens", 0), requestNestedPositiveInt(body, "options", "num_predict")} {
			if n > 0 {
				needs.contextTokens += n
				break
			}
		}
	}
	if embeddingsRoutes[path] {
		has[nodepolicy.CapEmbeddings] = true
	}
	for _, c := range nodepolicy.KnownCapabilities {
		if has[c] {
			needs.caps = append(needs.caps, c)
		}
	}

	return needs, nil
}

// Repeated messages arrays decode into reused struct elements in encoding/json.
// An omitted field, null element, or mistyped element retains that element's
// previous fields, including elements hidden by an intermediate shorter array.
// An empty or null array discards the backing storage. Track only the fields
// capability inference reads, borrowing content rather than copying it.
func requestMessagesHaveImage(ctx context.Context, body []byte) (bool, error) {
	maxMessages := 0
	visitObject(body, func(key []byte, start, end int) {
		if !jsonKeyEqual(key, "messages", true) {
			return
		}
		count := 0
		visitArray(body[start:end], func([]byte) { count++ })
		maxMessages = max(maxMessages, count)
	})
	if maxMessages == 0 {
		return false, nil
	}
	type messageState struct {
		content    []byte
		generation uint64
		images     bool
	}
	w := newBodyWorkspace(ctx)
	defer w.close()
	// A borrowed slice header, generation, boolean and alignment total 40
	// bytes per element on the supported 64-bit architectures. One exact-cap
	// vector suffices across every repeated array; no decoder/growth copies.
	if err := w.acquire(40 * int64(maxMessages)); err != nil {
		return false, err
	}
	messages := make([]messageState, maxMessages)
	generation, count := uint64(1), 0
	visitObject(body, func(key []byte, start, end int) {
		if !jsonKeyEqual(key, "messages", true) {
			return
		}
		raw := body[start:end]
		if bytes.Equal(raw, []byte("null")) {
			generation++
			count = 0
			return
		}
		if len(raw) == 0 || raw[0] != '[' {
			return // A type error leaves the old slice and its elements intact.
		}
		count = 0
		visitArray(raw, func(message []byte) {
			state := &messages[count]
			count++
			if state.generation != generation {
				*state = messageState{generation: generation}
			}
			visitObject(message, func(key []byte, start, end int) {
				if jsonKeyEqual(key, "content", true) {
					state.content = message[start:end]
				} else if jsonKeyEqual(key, "images", true) {
					images := message[start:end]
					if bytes.Equal(images, []byte("null")) {
						state.images = false
					} else if len(images) > 0 && images[0] == '[' {
						state.images = false
						visitArray(images, func([]byte) { state.images = true })
					}
				}
			})
		})
		if count == 0 {
			generation++
		}
	})
	for _, message := range messages[:count] {
		if message.images || contentHasImage(message.content) {
			return true, nil
		}
	}
	return false, nil
}

func nonEmptyArray(raw json.RawMessage) bool {
	if !ownedJSONValid(raw) {
		return false
	}
	nonempty := false
	visitArray(raw, func([]byte) { nonempty = true })
	return nonempty
}

// contentHasImage reports whether an OpenAI message content array carries an
// image part. A string content never does.
func contentHasImage(raw json.RawMessage) bool {
	if !ownedJSONValid(raw) {
		return false
	}
	found, valid := false, true
	visitArray(raw, func(part []byte) {
		if bytes.Equal(part, []byte("null")) {
			return
		}
		if !isJSONObject(part) {
			valid = false
			return
		}
		visitObject(part, func(key []byte, start, end int) {
			if jsonKeyEqual(key, "type", true) && part[start] != '"' && !bytes.Equal(part[start:end], []byte("null")) {
				valid = false
			}
		})
		value := requestStructString(part, "type", nil)
		if jsonStringEquals(value, "image_url") || jsonStringEquals(value, "image") || jsonStringEquals(value, "input_image") {
			found = true
		}
	})
	return found && valid
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

// metBy reports whether a model with these declared bounds can answer the
// request: every needed capability present, and either no context bound or
// one at least the estimate. Tier entries and declared roles share it.
func (n requestNeeds) metBy(caps []nodepolicy.Capability, contextTokens int) bool {
	for _, c := range n.caps {
		if !slices.Contains(caps, c) {
			return false
		}
	}
	return contextTokens == 0 || contextTokens >= n.contextTokens
}

// declaredRoles is what the nodes a tier request can reach declare they serve
// per tier. A node that declares at least one role is authoritative for
// itself: it offers exactly its declared role for each tier it names and
// nothing for a tier it omits, and this router's tier entries never apply to
// it. A node that declares nothing receives the router's tier entries.
type declaredRoles struct {
	// self is this node's own declaration, read from its policy and never
	// from the inventory document it publishes.
	self map[string]nodepolicy.AdvertisedRole
	// peers maps a peer's node ID to its advertised declaration. A peer that
	// declares nothing is absent.
	peers map[string]map[string]nodepolicy.AdvertisedRole
}

// declaredRoles collects the declarations visible to a request entering f:
// this node's own roles, and the advertisements in every facade overlay of
// this process. A node's advertisement is the same in every overlay it
// appears in.
func (f *facade) declaredRoles(pol nodepolicy.Policy) declaredRoles {
	d := declaredRoles{
		self:  pol.Advertisement().Roles,
		peers: map[string]map[string]nodepolicy.AdvertisedRole{},
	}
	selfID, _ := f.host.selfIdentity(f.profile.Name)
	for _, g := range append([]*facade{f}, f.host.enabledFacades()...) {
		for _, n := range g.discovery.Nodes() {
			if len(n.Roles) > 0 && (selfID == "" || n.ID != selfID) {
				d.peers[n.ID] = n.Roles
			}
		}
	}
	return d
}

// of returns the declaration of a candidate's node, empty when it declares
// nothing.
func (d declaredRoles) of(c candidate) map[string]nodepolicy.AdvertisedRole {
	if c.self {
		return d.self
	}
	return d.peers[c.id]
}

// forTier lists the distinct (engine, model) pairs declared for tier t by any
// node, ordered by engine and then model so resolution is stable.
func (d declaredRoles) forTier(t string) []nodepolicy.AdvertisedRole {
	var out []nodepolicy.AdvertisedRole
	add := func(roles map[string]nodepolicy.AdvertisedRole) {
		r, ok := roles[t]
		if ok && !slices.ContainsFunc(out, func(o nodepolicy.AdvertisedRole) bool {
			return o.Engine == r.Engine && o.Model == r.Model
		}) {
			out = append(out, nodepolicy.AdvertisedRole{Engine: r.Engine, Model: r.Model})
		}
	}
	add(d.self)
	for _, roles := range d.peers {
		add(roles)
	}
	slices.SortFunc(out, func(a, b nodepolicy.AdvertisedRole) int {
		if c := strings.Compare(a.Engine, b.Engine); c != 0 {
			return c
		}
		return strings.Compare(a.Model, b.Model)
	})
	return out
}

// peerTiers reports every tier at least one peer declares.
func (d declaredRoles) peerTiers() map[string]bool {
	out := map[string]bool{}
	for _, roles := range d.peers {
		for t := range roles {
			out[t] = true
		}
	}
	return out
}

// tierFacade is the facade that builds candidates on engine for a tier request
// entering f: f itself for its own engine, another engine's facade only when
// the route may cross engines, and nil when there is none.
func (f *facade) tierFacade(engine string, crossEngine bool) *facade {
	if engine == f.profile.Name {
		return f
	}
	if !crossEngine {
		return nil
	}
	return f.host.facadeFor(engine)
}

// resolveTierCandidates builds the ordered (node, engine, model) candidates for
// a tier request. For each tier in the search order it takes first the role
// each declaring node names for that tier, then the tier's local entries for
// the nodes that declare nothing (declaredRoles). API entries are skipped
// until hybrid dispatch exists. A declared role, like an entry, resolves only
// on nodes whose inventory has its model. Within a role or an entry,
// candidates keep resolveCandidates' order (node/select pin, scheduler
// priority, ID); across them the policy's preferLoaded decides whether tier
// rank or warmth leads, and declared candidates stay ahead of entries of equal
// rank and warmth.
func (f *facade) resolveTierCandidates(tier string, needs requestNeeds, crossEngine bool) []candidate {
	pol := f.host.admission.currentPolicy()
	declared := f.declaredRoles(pol)
	var out []candidate
	seen := map[string]bool{}
	add := func(c candidate, g *facade, model string, rank int, fromRole bool) {
		key := c.id + "\x00" + c.engine + "\x00" + g.profile.normalizeModel(model)
		if seen[key] {
			return
		}
		seen[key] = true
		c.model = model
		c.tierRank = rank
		c.declared = fromRole
		out = append(out, c)
	}
	for rank, t := range tierSearchOrder(tier, pol.TierPolicy) {
		for _, role := range declared.forTier(t) {
			g := f.tierFacade(role.Engine, crossEngine)
			if g == nil {
				continue
			}
			want := g.profile.normalizeModel(role.Model)
			for _, c := range g.resolveCandidates(role.Model) {
				r, ok := declared.of(c)[t]
				if !ok || r.Engine != role.Engine || g.profile.normalizeModel(r.Model) != want ||
					!needs.metBy(r.Capabilities, r.ContextTokens) {
					continue
				}
				add(c, g, r.Model, rank, true)
			}
		}
		for _, e := range pol.Tiers[t] {
			if !e.IsLocal() || !needs.metBy(e.Capabilities, e.ContextTokens) {
				continue
			}
			g := f.tierFacade(e.Engine, crossEngine)
			if g == nil {
				continue
			}
			for _, c := range g.resolveCandidates(e.Model) {
				if len(declared.of(c)) == 0 {
					add(c, g, e.Model, rank, false)
				}
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

// hybridCandidate is a local candidate eligible for a hybrid request: its
// node's declared role with finite qualification, and the strict entry the
// node advertises for it, whose identity a strict request must carry.
type hybridCandidate struct {
	candidate
	role   nodepolicy.AdvertisedRole
	strict nodepolicy.StrictAdmissionModel
}

// hybridLocalCandidates is the router's role eligibility for a hybrid local
// request: the candidates of resolveTierCandidates, in its order, that a node's
// own role declares (a router's tier entries never make a node hybrid
// eligible), whose advertised role carries finite qualification
// (AdvertisedRole.HybridQualified), and whose node advertises a valid strict
// entry for that model (Node.strictModel; this proxy's own installed
// capability for itself). Everything here is a hint: the destination
// revalidates identity, qualification and the body. Concrete and legacy tier
// routing never call it; hybrid dispatch (Phase 4) is its caller.
func (f *facade) hybridLocalCandidates(tier string, needs requestNeeds, crossEngine bool) []hybridCandidate {
	pol := f.host.admission.currentPolicy()
	declared := f.declaredRoles(pol)
	order := tierSearchOrder(tier, pol.TierPolicy)
	self := Node{StrictAdmission: f.host.admission.StrictReadiness().Capability}
	var out []hybridCandidate
	for _, c := range f.resolveTierCandidates(tier, needs, crossEngine) {
		if c.tierRank >= len(order) {
			continue
		}
		// A node that declares nothing has no role here, so a router tier
		// entry alone never makes it eligible.
		role, ok := declared.of(c)[order[c.tierRank]]
		if !ok || !role.HybridQualified() {
			continue
		}
		node, ok := self, c.self
		if !c.self {
			node, ok = discoveredNode(c.fac, c.id)
		}
		if !ok {
			continue
		}
		m, ok := node.strictModel(c.engine, c.fac.profile.normalizeModel(c.model))
		if !ok {
			continue
		}
		out = append(out, hybridCandidate{candidate: c, role: role, strict: m})
	}
	return out
}

// discoveredNode is the node with id in g's discovery overlay.
func discoveredNode(g *facade, id string) (Node, bool) {
	for _, n := range g.discovery.Nodes() {
		if n.ID == id {
			return n, true
		}
	}
	return Node{}, false
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

func (f *facade) reserveRoundPrepared(ctx context.Context, cands []candidate, tier bool, model string, prepare candidatePreparation, refresh ...func() []candidate) ([]candidate, reservation, error) {
	for {
		f.host.admission.mu.Lock()
		generation := f.host.admission.policyGeneration
		f.host.admission.mu.Unlock()
		if len(refresh) > 0 {
			cands = refresh[0]()
		}
		n := len(cands)
		if tier && n > 0 {
			n = 1
			for n < len(cands) && cands[n].tierRank == cands[0].tierRank && cands[n].warm == cands[0].warm {
				n++
			}
		}
		_, held, err := f.host.reserveCandidatePrepared(ctx, f, cands[:n], model, prepare, &generation)
		if err == errCandidatePreparationStale {
			continue // Re-resolve the policy's candidate class before rebuilding.
		}
		return cands, held, err
	}
}

// tierModelEntries appends the synthetic alias records to a merged /v1/models
// list, so a client can pick a tier from it. An upstream model already named
// like an alias is left alone.
func (f *facade) tierModelEntries(seen map[string]modelListItem) []json.RawMessage {
	pol := f.host.admission.currentPolicy()
	return hybridAliasRecords(pol, f.declaredRoles(pol).peerTiers(), seen)
}

// configuredAliases lists the reserved hybrid alias names offered here: every
// tier that has at least one policy entry of any kind, that this node's own
// roles declare, or that is in peerTiers (the tiers peers visible here
// declare), each as the bare tier and both variants, weakest tier first. A
// tier whose entries are all API deployments is listed like any other.
// peerTiers may be nil: a listener with no facade overlay reads only the
// policy, so only this node's own roles count there.
func configuredAliases(pol nodepolicy.Policy, peerTiers map[string]bool) []string {
	var out []string
	for _, t := range nodepolicy.TierOrder {
		_, own := pol.Roles[t]
		if len(pol.Tiers[t]) == 0 && !own && !peerTiers[t] {
			continue
		}
		out = append(out, t, t+"-api", t+"-local")
	}
	return out
}

// hybridAliasRecords returns one synthetic OpenAI model record per configured
// alias (configuredAliases), skipping any id present in skip (which may be
// nil). The record names no policy detail: never a credential, account,
// endpoint or deployment.
func hybridAliasRecords(pol nodepolicy.Policy, peerTiers map[string]bool, skip map[string]modelListItem) []json.RawMessage {
	var out []json.RawMessage
	for _, id := range configuredAliases(pol, peerTiers) {
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
	if name := jsonModelAlias(body); name != "" {
		return name, aliasReasonNotInferenceReady, true
	}
	for rest := r.URL.RawQuery; rest != ""; {
		var field string
		field, rest, _ = strings.Cut(rest, "&")
		if strings.Contains(field, ";") {
			continue // Preserve URL.Query's rejection of semicolon-bearing fields.
		}
		key, value, _ := strings.Cut(field, "=")
		if queryModelKey(key) {
			decoder := aliasQueryDecoder{raw: value}
			if name := borrowedAlias(decoder.nextRune); name != "" && !decoder.bad {
				return name, aliasReasonNotInferenceReady, true
			}
		}
	}
	for _, prefix := range modelDetailPrefixes {
		if rest, ok := strings.CutPrefix(r.URL.Path, prefix); ok {
			if _, mode, ok := nodepolicy.ParseAlias(strings.TrimSpace(rest)); ok && mode != nodepolicy.AliasBare {
				return strings.TrimSpace(rest), aliasReasonNotInferenceReady, true
			}
		}
	}
	return "", "", false
}

func queryModelKey(raw string) bool {
	decoder := aliasQueryDecoder{raw: raw}
	for _, want := range []byte("model") {
		got, ok := decoder.nextByte()
		if !ok || got != want {
			return false
		}
	}
	_, more := decoder.nextByte()
	return !more && !decoder.bad
}

// The validation guard measures decoded bytes before any alias trimming or
// model normalization. Values remain borrowed, including malformed-prefix
// completed members and query fields whose percent decoding later fails.
func validateRequestIdentifiers(r *http.Request, body []byte) error {
	tooLong := false
	visitCompletedRequestMembers(body, func(key, raw []byte) bool {
		if len(raw) > 0 && raw[0] == '"' && (jsonKeyEqual(key, "model", true) || jsonKeyEqual(key, "name", true)) && requestJSONStringTooLong(raw) {
			tooLong = true
			return false
		}
		return true
	})
	if tooLong {
		return &requestIdentifierError{}
	}
	for rest := r.URL.RawQuery; rest != ""; {
		var field string
		field, rest, _ = strings.Cut(rest, "&")
		if strings.Contains(field, ";") {
			continue
		}
		key, value, _ := strings.Cut(field, "=")
		if !queryModelKey(key) {
			continue
		}
		decoder := aliasQueryDecoder{raw: value}
		n := 0
		for _, more := decoder.nextByte(); more; _, more = decoder.nextByte() {
			if n <= maxRequestIdentifierBytes {
				n++
			}
		}
		if !decoder.bad && n > maxRequestIdentifierBytes {
			return &requestIdentifierError{}
		}
	}
	for _, prefix := range modelDetailPrefixes {
		if rest, ok := strings.CutPrefix(r.URL.Path, prefix); ok && len(rest) > maxRequestIdentifierBytes {
			return &requestIdentifierError{}
		}
	}
	return nil
}

// Completed members retain the old decoder's partial-prefix behavior even
// when a later member is malformed. Only JSON's four whitespace bytes are
// skipped, matching encoding/json rather than Unicode strings.TrimSpace.
func visitCompletedRequestMembers(body []byte, visit func([]byte, []byte) bool) {
	trim := func(raw []byte) []byte { return bytes.Trim(raw, " \t\r\n") }
	rest := trim(body)
	if len(rest) == 0 || rest[0] != '{' {
		return
	}
	rest = trim(rest[1:])
	for len(rest) > 0 && rest[0] != '}' {
		keyEnd, ok := ownedJSONStringEnd(rest, 0)
		if !ok {
			return
		}
		key := rest[:keyEnd]
		rest = trim(rest[keyEnd:])
		if len(rest) == 0 || rest[0] != ':' {
			return
		}
		rest = trim(rest[1:])
		if len(rest) == 0 {
			return
		}
		end := streamJSONValueEnd(rest)
		if rest[0] == '"' {
			var ok bool
			end, ok = ownedJSONStringEnd(rest, 0)
			if !ok {
				return
			}
		}
		if end == 0 || !ownedJSONValid(rest[:end]) {
			return
		}
		if !visit(key, rest[:end]) {
			return
		}
		rest = trim(rest[end:])
		if len(rest) == 0 || rest[0] != ',' {
			return
		}
		rest = trim(rest[1:])
	}
}

func jsonModelAlias(body []byte) string {
	var name string
	visitCompletedRequestMembers(body, func(key, raw []byte) bool {
		if raw[0] != '"' || (!jsonKeyEqual(key, "model", true) && !jsonKeyEqual(key, "name", true)) {
			return true
		}
		pos := 1
		name = borrowedAlias(func() (rune, bool) {
			if pos >= len(raw)-1 {
				return 0, false
			}
			r, next := ownedJSONRune(raw, pos)
			pos = next
			return r, true
		})
		return name == ""
	})
	return name
}

func borrowedAlias(next func() (rune, bool)) string {
	var name [16]byte
	n, trailing, invalid := 0, false, false
	for r, more := next(); more; r, more = next() {
		if unicode.IsSpace(r) {
			trailing = n > 0
			continue
		}
		if trailing || r > 127 || n == len(name) {
			invalid = true
			continue
		}
		name[n], n = byte(r), n+1
	}
	if !invalid {
		for _, alias := range [...]string{"weak-api", "weak-local", "medium-api", "medium-local", "strong-api", "strong-local"} {
			if len(alias) == n {
				matched := true
				for i := range n {
					matched = matched && name[i] == alias[i]
				}
				if matched {
					return alias
				}
			}
		}
	}
	return ""
}

type aliasQueryDecoder struct {
	raw string
	pos int
	bad bool
}

func (d *aliasQueryDecoder) nextByte() (byte, bool) {
	if d.bad || d.pos == len(d.raw) {
		return 0, false
	}
	b := d.raw[d.pos]
	d.pos++
	if b == '+' {
		return ' ', true
	}
	if b != '%' {
		return b, true
	}
	if len(d.raw)-d.pos < 2 {
		d.bad = true
		return 0, false
	}
	var decoded byte
	for range 2 {
		x := d.raw[d.pos]
		d.pos++
		switch {
		case x >= '0' && x <= '9':
			x -= '0'
		case x >= 'a' && x <= 'f':
			x -= 'a' - 10
		case x >= 'A' && x <= 'F':
			x -= 'A' - 10
		default:
			d.bad = true
			return 0, false
		}
		decoded = decoded*16 + x
	}
	return decoded, true
}

func (d *aliasQueryDecoder) nextRune() (rune, bool) {
	var raw [utf8.UTFMax]byte
	n := 0
	for n < len(raw) {
		b, more := d.nextByte()
		if !more {
			if n == 0 {
				return 0, false
			}
			return utf8.RuneError, true
		}
		raw[n], n = b, n+1
		if utf8.FullRune(raw[:n]) {
			r, _ := utf8.DecodeRune(raw[:n])
			return r, true
		}
	}
	return utf8.RuneError, true
}

// writeAliasRejection answers a refused alias request with a structured 400.
func writeAliasRejection(w http.ResponseWriter, alias, reason string) {
	writeInferenceErrorFields(w, http.StatusBadRequest,
		inferenceErrorField{"error", []string{"hybrid routing alias is not available on this request"}},
		inferenceErrorField{"code", []string{"feature_unavailable"}},
		inferenceErrorField{"reason", []string{reason}},
		inferenceErrorField{"model", []string{strings.TrimSpace(alias)}},
	)
}
