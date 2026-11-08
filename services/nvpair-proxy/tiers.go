// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Role routing: a request naming a visibly declared role is resolved by the
// node it enters, to a concrete (node, engine, model) chosen only from the
// roles nodes declare for themselves. Widening follows the requesting node's
// ordered role declarations; API entries supply no local candidates.
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
	"/v1/embeddings": true,
}

// requestNeeds is what a request requires of the model that answers it.
type requestNeeds struct {
	caps          []nodepolicy.Capability
	contextTokens int
	// inputTokens and outputTokens size a destination-effective body for the
	// tier scorer only: input is len(body)/4 and output the first positive
	// limit, or 0 when the request sets none. Admission reads neither.
	inputTokens  int
	outputTokens int
}

// names lists the needed capabilities for the no-eligible-model answer.
func (n requestNeeds) names() []string {
	out := make([]string, 0, len(n.caps))
	for _, c := range n.caps {
		out = append(out, string(c))
	}
	return out
}

// parseRequestNeeds reads a buffered request for the capabilities and context
// it needs. A body that is not
// JSON needs nothing beyond its size.
func parseRequestNeeds(body []byte, path string) requestNeeds {
	needs, _ := parseRequestNeedsOwned(nil, body, path)
	return needs
}

func parseRequestNeedsOwned(ctx context.Context, body []byte, path string) (requestNeeds, error) {
	needs := requestNeeds{contextTokens: len(body) / 4, inputTokens: len(body) / 4}
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
				needs.outputTokens = n
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

// tierSearchOrder follows the requesting node's weakest-to-strongest list:
// requested role, stronger roles, then weaker roles nearest first. A role
// absent from that list never widens.
func tierSearchOrder(tier string, tp nodepolicy.TierPolicy, names []string) []string {
	order := []string{tier}
	i := slices.Index(names, tier)
	if i < 0 {
		return order
	}
	if tp.AllowStronger {
		order = append(order, names[i+1:]...)
	}
	if tp.AllowWeaker {
		for j := i - 1; j >= 0; j-- {
			order = append(order, names[j])
		}
	}
	return order
}

// metBy reports whether a model with these declared bounds can answer the
// request: every needed capability present, and either no context bound or
// one at least the estimate.
func (n requestNeeds) metBy(caps []nodepolicy.Capability, contextTokens int) bool {
	for _, c := range n.caps {
		if !slices.Contains(caps, c) {
			return false
		}
	}
	return contextTokens == 0 || contextTokens >= n.contextTokens
}

// declaredRoles is what the nodes a tier request can reach declare they serve
// per tier. Each node offers only its own declared role memberships.
type declaredRoles struct {
	// self is this node's own declaration, read from its policy and never
	// from the inventory document it publishes.
	self map[string][]nodepolicy.AdvertisedProfile
	// peers maps a peer's node ID to its advertised declaration. A peer that
	// declares nothing is absent.
	peers map[string]map[string][]nodepolicy.AdvertisedProfile
}

// resolveRoleProfiles keeps each role's reachable facts in membership order.
// Empty roles and roles whose keys are all withheld or missing remain present
// so their node's declaration stays authoritative.
func resolveRoleProfiles(roles []nodepolicy.Role, profiles []nodepolicy.AdvertisedProfile) map[string][]nodepolicy.AdvertisedProfile {
	out := make(map[string][]nodepolicy.AdvertisedProfile, len(roles))
	for _, role := range roles {
		out[role.Name] = nil
		for _, key := range role.Profiles {
			for _, profile := range profiles {
				if profile.Key == key && nodepolicy.ValidateModelIdentifier(profile.Model) == nil {
					out[role.Name] = append(out[role.Name], cloneAdvertisedProfile(profile))
					break
				}
			}
		}
	}
	return out
}

// selfDeclaredRoles resolves this node's own policy facts.
func selfDeclaredRoles(pol nodepolicy.Policy) declaredRoles {
	// Self routing uses the whole policy, including local-only profiles that
	// must remain withheld from peer advertisements.
	profiles := make([]nodepolicy.AdvertisedProfile, 0, len(pol.Profiles))
	for _, pr := range pol.Profiles {
		profiles = append(profiles, nodepolicy.AdvertisedProfile{
			Key: pr.Key, DisplayName: pr.DisplayName, Icon: pr.Icon,
			Engine: pr.Engine, Model: pr.Model, ModelIdentity: pr.ModelIdentity,
			ContextTokens: pr.ContextTokens, Capabilities: pr.Capabilities,
			MaxOutputTokens: pr.MaxOutputTokens, Qualification: pr.Qualification,
		})
	}
	return declaredRoles{
		self:  resolveRoleProfiles(pol.Roles, profiles),
		peers: map[string]map[string][]nodepolicy.AdvertisedProfile{},
	}
}

// declaredRoles collects the declarations visible to a request entering f:
// this node's own roles, and the advertisements in every facade overlay of
// this process. A node's advertisement is the same in every overlay it
// appears in.
func (f *facade) declaredRoles(pol nodepolicy.Policy) declaredRoles {
	return f.host.declaredRoles(pol, f)
}

// declaredRoles shares the visible declarations with the unified catalog.
// entry also covers an inference adapter not registered as an enabled facade.
func (p *Proxy) declaredRoles(pol nodepolicy.Policy, entry *facade) declaredRoles {
	d := selfDeclaredRoles(pol)
	selfID, _ := p.selfIdentity("")
	facades := p.enabledFacades()
	if entry != nil && !slices.Contains(facades, entry) {
		facades = append(facades, entry)
	}
	for _, g := range facades {
		for _, n := range g.discovery.Nodes() {
			if len(n.Roles) > 0 && (selfID == "" || n.ID != selfID) {
				d.peers[n.ID] = resolveRoleProfiles(n.Roles, n.Profiles)
			}
		}
	}
	return d
}

// roleRequested reads declarations only, including empty roles. API entries
// and profile facts do not declare roles, and no profile map is built here.
func (f *facade) roleRequested(model string) bool {
	if nodepolicy.ValidateRoleName(model) != nil {
		return false
	}
	for _, role := range f.host.admission.currentPolicy().Roles {
		if role.Name == model {
			return true
		}
	}
	selfID, _ := f.host.selfIdentity(f.profile.Name)
	for _, g := range append([]*facade{f}, f.host.enabledFacades()...) {
		for _, n := range g.discovery.Nodes() {
			if selfID != "" && n.ID == selfID {
				continue
			}
			for _, role := range n.Roles {
				if role.Name == model {
					return true
				}
			}
		}
	}
	return false
}

// concreteModelAdvertised distinguishes an unknown key from a visible profile
// excluded by availability, trust, serving policy or route eligibility.
func (f *facade) concreteModelAdvertised(model string) bool {
	if _, ok := f.host.admission.currentPolicy().ProfileByKey(model); ok {
		return true
	}
	selfID, _ := f.host.selfIdentity("")
	for _, g := range append([]*facade{f}, f.host.enabledFacades()...) {
		for _, n := range g.discovery.Nodes() {
			if selfID != "" && n.ID == selfID {
				continue
			}
			for _, profile := range n.Profiles {
				if profile.Key == model {
					return true
				}
			}
		}
	}
	return false
}

// of returns the declaration of a candidate's node, empty when it declares
// nothing.
func (d declaredRoles) of(c candidate) map[string][]nodepolicy.AdvertisedProfile {
	if c.self {
		return d.self
	}
	return d.peers[c.id]
}

// forTier lists the distinct (engine, profile key) pairs declared for tier t by any
// node, ordered by membership position, then engine and key. Each candidate
// is subsequently placed at its own node's membership position, since a pair
// may occupy different positions on different nodes.
func (d declaredRoles) forTier(t string) []nodepolicy.ModelRef {
	type member struct {
		ref      nodepolicy.ModelRef
		position int
	}
	var members []member
	add := func(roles map[string][]nodepolicy.AdvertisedProfile) {
		for i, r := range roles[t] {
			if nodepolicy.ValidateModelIdentifier(r.Model) == nil {
				members = append(members, member{ref: nodepolicy.ModelRef{Engine: r.Engine, Model: r.Key}, position: i})
			}
		}
	}
	add(d.self)
	var peers []string
	for id := range d.peers {
		peers = append(peers, id)
	}
	slices.Sort(peers)
	for _, id := range peers {
		add(d.peers[id])
	}
	sort.SliceStable(members, func(i, j int) bool {
		a, b := members[i], members[j]
		if a.position != b.position {
			return a.position < b.position
		}
		if c := strings.Compare(a.ref.Engine, b.ref.Engine); c != 0 {
			return c < 0
		}
		return a.ref.Model < b.ref.Model
	})
	var out []nodepolicy.ModelRef
	seen := map[nodepolicy.ModelRef]bool{}
	for _, member := range members {
		if !seen[member.ref] {
			seen[member.ref] = true
			out = append(out, member.ref)
		}
	}
	return out
}

// candidateRoleProfile selects only this candidate's own matching facts. The
// first matching membership that meets the request supplies all its bounds.
func candidateRoleProfile(profiles []nodepolicy.AdvertisedProfile, engine, key string, needs requestNeeds) (nodepolicy.AdvertisedProfile, int, bool) {
	for i, profile := range profiles {
		if profile.Engine == engine && profile.Key == key && needs.metBy(profile.Capabilities, profile.ContextTokens) {
			return profile, i, true
		}
	}
	return nodepolicy.AdvertisedProfile{}, 0, false
}

// roleContextLength is the largest known bound of the role's visible profiles
// whose declaring node is a resolved candidate. Missing keys carry no facts;
// a reachable unbounded profile omits the bound.
func (p *Proxy) roleContextLength(d declaredRoles, t string, reachable func(nodepolicy.AdvertisedProfile, string, bool) bool) int {
	length, unbounded := 0, false
	add := func(roles map[string][]nodepolicy.AdvertisedProfile, nodeID string, self bool) {
		for _, profile := range roles[t] {
			if !reachable(profile, nodeID, self) {
				continue
			}
			if profile.ContextTokens == 0 {
				unbounded = true
			}
			length = max(length, profile.ContextTokens)
		}
	}
	add(d.self, "", true)
	for nodeID, roles := range d.peers {
		add(roles, nodeID, false)
	}
	if unbounded {
		return 0
	}
	return length
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
// a tier request from each node's declared role memberships. A role resolves only
// on nodes with matching inventory. Within a role,
// candidates keep resolveCandidates' order (node/select pin, scheduler
// priority, ID); across them the policy's preferLoaded decides whether tier
// rank or warmth leads. Within equal comparator classes, a node's earlier profile
// memberships lead; pair and owner order break ties at the same position.
func (f *facade) resolveTierCandidates(tier string, needs requestNeeds, crossEngine bool) []candidate {
	pol := f.host.admission.currentPolicy()
	declared := f.declaredRoles(pol)
	var out []candidate
	seen := map[string]bool{}
	add := func(c candidate, model string, rank int) {
		if nodepolicy.ValidateModelIdentifier(model) != nil {
			return
		}
		key := c.id + "\x00" + c.engine + "\x00" + model
		if seen[key] {
			return
		}
		seen[key] = true
		c.model = model
		c.key = model
		c.tierRank = rank
		out = append(out, c)
	}
	for rank, t := range tierSearchOrder(tier, pol.TierPolicy, pol.RoleNames()) {
		type roleCandidate struct {
			candidate candidate
			position  int
		}
		var roles []roleCandidate
		for _, role := range declared.forTier(t) {
			g := f.tierFacade(role.Engine, crossEngine)
			if g == nil {
				continue
			}
			for _, c := range g.resolveCandidates(role.Model) {
				r, position, ok := candidateRoleProfile(declared.of(c)[t], c.engine, role.Model, needs)
				if !ok {
					continue
				}
				c.model = r.Key
				roles = append(roles, roleCandidate{candidate: c, position: position})
			}
		}
		// A pair discovered at another node's first position must not lift
		// this node's later sibling ahead of its own first membership.
		sort.SliceStable(roles, func(i, j int) bool {
			a, b := roles[i], roles[j]
			if a.position != b.position {
				return a.position < b.position
			}
			if a.candidate.engine != b.candidate.engine {
				return a.candidate.engine < b.candidate.engine
			}
			return a.candidate.model < b.candidate.model
		})
		for _, role := range roles {
			c := role.candidate
			add(c, c.model, rank)
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
	role   nodepolicy.AdvertisedProfile
	strict nodepolicy.StrictAdmissionModel
}

// hybridLocalCandidates is the router's role eligibility for a hybrid local
// request: the candidates of resolveTierCandidates, in its order, that a node's
// own role declares, whose advertised role carries finite qualification
// (AdvertisedProfile.HybridQualified), and whose node advertises a valid strict
// entry for that model (Node.strictModel; this proxy's own installed
// capability for itself). Everything here is a hint: the destination
// revalidates identity, qualification and the body. Hybrid dispatch is its caller.
func (f *facade) hybridLocalCandidates(tier string, needs requestNeeds, crossEngine bool) []hybridCandidate {
	pol := f.host.admission.currentPolicy()
	declared := f.declaredRoles(pol)
	order := tierSearchOrder(tier, pol.TierPolicy, pol.RoleNames())
	self := Node{StrictAdmission: f.host.admission.StrictReadiness().Capability}
	var out []hybridCandidate
	for _, c := range f.resolveTierCandidates(tier, needs, crossEngine) {
		if c.dormant {
			continue
		}
		if c.tierRank >= len(order) {
			continue
		}
		role, _, ok := candidateRoleProfile(declared.of(c)[order[c.tierRank]], c.engine, c.model, needs)
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
		m, ok := node.strictModel(c.engine, c.fac.profile.normalizeModel(role.Model))
		if !ok {
			continue
		}
		// Strict qualification and dispatch retain the destination's model
		// identity after the public profile key selected this role member.
		c.model = role.Model
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

// reserveRoundPrepared reserves within the leading class. A non-nil round is
// a tier request: its class is the leading tier rank and warmth, and every
// facade's discovery generation is captured before the class is resolved.
func (f *facade) reserveRoundPrepared(ctx context.Context, cands []candidate, round *tierRound, model string, prepare candidatePreparation, refresh ...func() []candidate) ([]candidate, reservation, tierDecision, error) {
	for {
		f.host.admission.mu.Lock()
		generation := f.host.admission.policyGeneration
		f.host.admission.mu.Unlock()
		if round != nil {
			round.discovery = f.host.discoveryGenerations()
		}
		if len(refresh) > 0 {
			cands = refresh[0]()
		}
		n := len(cands)
		if round != nil && n > 0 {
			n = 1
			for n < len(cands) && cands[n].tierRank == cands[0].tierRank && cands[n].warm == cands[0].warm {
				n++
			}
		}
		_, held, decision, err := f.host.reserveCandidatePrepared(ctx, f, cands[:n], model, prepare, &generation, round)
		if err == errCandidatePreparationStale {
			continue // Re-resolve the policy's candidate class before rebuilding.
		}
		return cands, held, decision, err
	}
}

// tierModelEntries appends the synthetic alias records to a merged /v1/models
// list, so a client can pick a role from it. Role records take precedence over
// concrete records with the same name.
func (f *facade) tierModelEntries() []json.RawMessage {
	pol := f.host.admission.currentPolicy()
	return f.host.hybridAliasRecords(pol, f.declaredRoles(pol))
}

// profileModelEntries lists visible roles first, then distinct profile keys.
// Self facts come from policy, including local-only profiles; peer facts come
// from advertisements, independently of role membership or native inventory.
func (p *Proxy) profileModelEntries(pol nodepolicy.Policy, entry *facade) []json.RawMessage {
	models, _ := p.profileCatalog(pol, entry)
	return models
}

// profileCatalog captures entries and their engines together so a discovery
// update during browser permission checks cannot add an unchecked engine.
func (p *Proxy) profileCatalog(pol nodepolicy.Policy, entry *facade) ([]json.RawMessage, map[string]bool) {
	out := p.hybridAliasRecords(pol, p.declaredRoles(pol, entry))
	engines := map[string]bool{}
	if entry != nil {
		engines[entry.profile.Name] = true
	}
	seen := map[string]bool{}
	for _, raw := range out {
		var record struct {
			ID string `json:"id"`
		}
		if json.Unmarshal(raw, &record) == nil {
			seen[record.ID] = true
		}
	}
	type bounds struct {
		length    int
		unbounded bool
	}
	profiles := map[string]bounds{}
	reachable := p.catalogProfileReachability()
	add := func(profile nodepolicy.AdvertisedProfile, nodeID string, self bool) {
		key, context := profile.Key, profile.ContextTokens
		if nodepolicy.ValidateModelIdentifier(key) != nil {
			return
		}
		engines[profile.Engine] = true
		if seen[key] {
			return
		}
		b := profiles[key]
		if reachable(profile, nodeID, self) {
			b.length = max(b.length, context)
			b.unbounded = b.unbounded || context == 0
		}
		profiles[key] = b
	}
	for _, profile := range pol.Profiles {
		add(nodepolicy.AdvertisedProfile{Key: profile.Key, Engine: profile.Engine, Model: profile.Model, ContextTokens: profile.ContextTokens}, "", true)
	}
	selfID, _ := p.selfIdentity("")
	facades := p.enabledFacades()
	if entry != nil && !slices.Contains(facades, entry) {
		facades = append(facades, entry)
	}
	for _, f := range facades {
		for _, node := range f.discovery.Nodes() {
			if selfID != "" && node.ID == selfID {
				continue
			}
			for _, profile := range node.Profiles {
				add(profile, node.ID, false)
			}
		}
	}
	keys := make([]string, 0, len(profiles))
	for key := range profiles {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		record := map[string]any{"id": key, "object": "model", "created": 0, "owned_by": "pair"}
		b := profiles[key]
		if !b.unbounded && b.length > 0 {
			record["context_length"] = b.length
		}
		raw, err := json.Marshal(record)
		if err == nil {
			out = append(out, raw)
		}
	}
	if out == nil {
		out = []json.RawMessage{}
	}
	return out, engines
}

// configuredAliases lists own roles in policy order, then peer-only roles in
// name order, each followed by its overrides. API entries declare no roles.
func configuredAliases(pol nodepolicy.Policy, peerTiers map[string]bool) []string {
	var out []string
	seen := map[string]bool{}
	add := func(name string) {
		if !seen[name] && nodepolicy.ValidateRoleName(name) == nil {
			seen[name] = true
			out = append(out, name, name+":api", name+":local")
		}
	}
	for _, name := range pol.RoleNames() {
		add(name)
	}
	var peers []string
	for name, declared := range peerTiers {
		if declared && !seen[name] {
			peers = append(peers, name)
		}
	}
	slices.Sort(peers)
	for _, name := range peers {
		add(name)
	}
	return out
}

// catalogProfileReachability resolves each profile key on its engine once and
// retains the declaring-node check used by catalog context bounds.
func (p *Proxy) catalogProfileReachability() func(nodepolicy.AdvertisedProfile, string, bool) bool {
	type catalogNode struct {
		id   string
		self bool
	}
	candidates := map[nodepolicy.ModelRef]map[catalogNode]bool{}
	return func(profile nodepolicy.AdvertisedProfile, nodeID string, self bool) bool {
		ref := nodepolicy.ModelRef{Engine: profile.Engine, Model: profile.Key}
		nodes, resolved := candidates[ref]
		if !resolved {
			nodes = map[catalogNode]bool{}
			if f := p.facadeFor(profile.Engine); f != nil && nodepolicy.ValidateModelIdentifier(profile.Model) == nil {
				for _, c := range f.resolveCandidates(profile.Key) {
					id := c.id
					if c.self {
						id = ""
					}
					nodes[catalogNode{id: id, self: c.self}] = true
				}
			}
			candidates[ref] = nodes
		}
		return nodes[catalogNode{id: nodeID, self: self}]
	}
}

// hybridAliasRecords returns synthetic role records in configured order.
// Context uses reachable declaring-node facts, omitting unknown or unbounded
// bounds. These records take precedence over colliding profile keys.
func (p *Proxy) hybridAliasRecords(pol nodepolicy.Policy, declared declaredRoles) []json.RawMessage {
	reachable := p.catalogProfileReachability()
	lengths := map[string]int{}
	var out []json.RawMessage
	for _, id := range configuredAliases(pol, declared.peerTiers()) {
		record := map[string]any{
			"id":       id,
			"object":   "model",
			"created":  0,
			"owned_by": "pair",
		}
		role, _, _ := nodepolicy.ParseAlias(id)
		length, computed := lengths[role]
		if !computed {
			length = p.roleContextLength(declared, role, reachable)
			lengths[role] = length
		}
		if length > 0 {
			record["context_length"] = length
		}
		raw, err := json.Marshal(record)
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
	// aliasReasonNotInferenceReady: :api and :local require hybrid entry.
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
// A declared role's overrides require hybrid entry. A declared bare role is
// refused on non-inference routes or with surrounding whitespace. Undeclared
// names remain concrete even when they match the alias grammar.
func aliasRejection(model string, inference bool, roleRequested func(string) bool) (reason string, rejected bool) {
	trimmed := strings.TrimSpace(model)
	role, mode, ok := nodepolicy.ParseAlias(trimmed)
	switch {
	case !ok || !roleRequested(role):
		return "", false
	case mode != nodepolicy.AliasBare:
		return aliasReasonNotInferenceReady, true
	case trimmed != model, !inference:
		return aliasReasonUnsupportedUse, true
	}
	return "", false
}

// modelDetailPrefixes are the path prefixes whose remainder names one model, as
// in /v1/models/<id>.
var modelDetailPrefixes = []string{"/v1/models/"}

// requestAliasRejection is the refusal decision for one request. It applies
// aliasRejection to the body's "model" as the router reads it, and refuses a
// :api or :local alias in any other place an engine could read a model name
// from, because an engine may read a different one than the router does:
//   - every top-level JSON key equal under strings.EqualFold to "model" or
//     "name", counting repeated keys;
//   - every "model" query parameter (llama.cpp's GET /props?model=...);
//   - the remainder of a model-detail path (modelDetailPrefixes).
//
// A bare role in those places keeps its existing routing. alias is the name to
// report. The cluster ingress is not covered; it has its own listener.
func requestAliasRejection(r *http.Request, body []byte, model string, inference bool, roleRequested func(string) bool) (alias, reason string, rejected bool) {
	if reason, rejected := aliasRejection(model, inference, roleRequested); rejected {
		return strings.TrimSpace(model), reason, true
	}
	// An undeclared canonical model remains concrete. Other model/name
	// occurrences cannot smuggle hybrid syntax past the router's selector.
	concrete := ""
	if inference && bodyModelMatches(body, model) {
		concrete = model
	}
	return requestAlternateAliasRejection(r, body, concrete)
}

// requestAlternateAliasRejection permits only the canonical model selected by
// the request owner; hybrid syntax at every alternate engine-facing seam is
// refused independently of whether its base role is declared.
func requestAlternateAliasRejection(r *http.Request, body []byte, model string) (alias, reason string, rejected bool) {
	if name := jsonModelAliasExcept(body, model); name != "" {
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
	invalid := false
	visitCompletedRequestMembers(body, func(key, raw []byte) bool {
		if len(raw) > 0 && raw[0] == '"' && (jsonKeyEqual(key, "model", true) || jsonKeyEqual(key, "name", true)) {
			if requestJSONStringTooLong(raw) || nodepolicy.ValidateModelIdentifierJSON(raw) != nil {
				invalid = true
				return false
			}
		}
		return true
	})
	if invalid {
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
		var decoded [maxRequestIdentifierBytes]byte
		n := 0
		for b, more := decoder.nextByte(); more; b, more = decoder.nextByte() {
			if n < len(decoded) {
				decoded[n] = b
			}
			if n <= maxRequestIdentifierBytes {
				n++
			}
		}
		if !decoder.bad && (n > maxRequestIdentifierBytes || nodepolicy.ValidateModelIdentifier(string(decoded[:n])) != nil) {
			return &requestIdentifierError{}
		}
	}
	for _, prefix := range modelDetailPrefixes {
		if rest, ok := strings.CutPrefix(r.URL.Path, prefix); ok && (len(rest) > maxRequestIdentifierBytes || nodepolicy.ValidateModelIdentifier(rest) != nil) {
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
	return jsonModelAliasExcept(body, "")
}

func jsonModelAliasExcept(body []byte, concrete string) string {
	var name string
	visitCompletedRequestMembers(body, func(key, raw []byte) bool {
		if raw[0] != '"' || (!jsonKeyEqual(key, "model", true) && !jsonKeyEqual(key, "name", true)) {
			return true
		}
		if concrete != "" && jsonKeyEqual(key, "model", false) && jsonStringEquals(raw, concrete) {
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
	var name [nodepolicy.MaxRoleNameBytes + len(":local")]byte
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
		alias := string(name[:n])
		if _, mode, ok := nodepolicy.ParseAlias(alias); ok && mode != nodepolicy.AliasBare {
			return alias
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
