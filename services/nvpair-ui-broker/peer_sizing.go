// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sort"
	"strconv"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

// peerSizingRefreshInterval is how often a held observation is re-read. Only a
// new successful authenticated read renews one; heartbeats never do.
const peerSizingRefreshInterval = 30 * time.Second

// peerSizingAuthority is the live mesh admission owner. *clustertrust.Mesh
// satisfies it; tests substitute a controllable one.
type peerSizingAuthority interface {
	Refresh()
	PeerServiceIncarnation(peerUUID string) (string, bool)
	PeerAdmissionEpoch(peerUUID string) (uint64, bool)
}

// peerSizingTarget is one advertised, currently admitted peer as the directory
// describes it: the identity a fetch is captured against.
type peerSizingTarget struct {
	host     string
	cluster  string
	revision string
	profiles []nodepolicy.AdvertisedProfile
}

func (t peerSizingTarget) equal(o peerSizingTarget) bool {
	return t.host == o.host && t.cluster == o.cluster && t.revision == o.revision && slices.EqualFunc(t.profiles, o.profiles, func(a, b nodepolicy.AdvertisedProfile) bool {
		return a.Key == b.Key && a.Engine == b.Engine && a.Model == b.Model
	})
}

// peerSizingAdmission is the authority a fetch was captured under.
type peerSizingAdmission struct {
	identity string
	epoch    uint64
	source   *rpcWorker
}

// peerSizingObservation is one accepted authenticated read. Its age runs from
// fetch start on the monotonic clock, including read latency.
type peerSizingObservation struct {
	target     peerSizingTarget
	admission  peerSizingAdmission
	proxyRunID string
	profiles   []nodepolicy.PeerSizingProfile
	fetchedAt  time.Time
	// serial distinguishes successive successful reads with equal content, so a
	// fresh read is a changed observation even when options match.
	serial uint64
}

// peerSizingRuntime is the origin broker's one background owner of peer sizing
// observations. Relay callbacks only replace the desired set and signal; all
// network work runs on its own goroutine, outside relay, policy and priority
// locks. The projection is process-local and never persisted.
type peerSizingRuntime struct {
	authority peerSizingAuthority
	source    func() *rpcWorker
	fetch     func(ctx context.Context, w *rpcWorker, host string) (json.RawMessage, error)
	deliver   func(ctx context.Context, params json.RawMessage)
	now       func() time.Time
	self      string
	wake      chan struct{}

	mu           sync.Mutex
	desired      map[string]peerSizingTarget
	observations map[string]peerSizingObservation
	serial       uint64
	// The in-flight fetch's captured target and its cancellation, so a
	// directory change that makes it obsolete cancels it.
	inflightHost   string
	inflightTarget peerSizingTarget
	inflight       context.CancelFunc

	// sendMu serializes projection construction and delivery, including spawn
	// replay, so a reordered older snapshot is never sent after a newer one.
	sendMu   sync.Mutex
	sent     nodepolicy.PeerSizingSnapshot
	sentKeys string
}

func newPeerSizingRuntime(self string, authority peerSizingAuthority) *peerSizingRuntime {
	return &peerSizingRuntime{
		authority:    authority,
		now:          time.Now,
		self:         self,
		wake:         make(chan struct{}, 1),
		desired:      map[string]peerSizingTarget{},
		observations: map[string]peerSizingObservation{},
	}
}

func (r *peerSizingRuntime) signal() {
	if r == nil {
		return
	}
	select {
	case r.wake <- struct{}{}:
	default:
	}
}

// startPeerSizing attaches the runtime and runs it with the serve context. The
// returned stop cancels, joins and unregisters the directory subscription.
func (b *Broker) startPeerSizing(ctx context.Context) func() {
	r := newPeerSizingRuntime(b.nodeID, clustertrust.Open(b.clusterDir))
	r.source = b.getEngineMgr
	r.fetch = func(ctx context.Context, w *rpcWorker, host string) (json.RawMessage, error) {
		params, err := json.Marshal(map[string]string{"nodeId": host})
		if err != nil {
			return nil, err
		}
		result, rpcErr, err := w.Call(ctx, remotePolicyMethods[methodPolicyGet], params)
		if err != nil {
			return nil, err
		}
		if rpcErr != nil {
			return nil, fmt.Errorf("RPC %d", rpcErr.Code)
		}
		return result, nil
	}
	r.deliver = b.deliverPeerSizing
	b.peerSizing = r
	var id int
	if b.relayDir != nil {
		sub, subID, err := r.subscribe(b)
		if err != nil {
			slog.Warn("peer sizing directory subscription failed", "err", err)
		} else {
			id = subID
			b.relayDir.Deliver(sub)
		}
	}
	rctx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		r.run(rctx)
	}()
	return func() {
		cancel()
		<-done
		if id != 0 {
			b.relayDir.Unsubscribe(id)
		}
	}
}

func (r *peerSizingRuntime) subscribe(b *Broker) (*relay.Subscriber, int, error) {
	id, sub, err := subscribeRelay(b.relayDir, json.RawMessage(`{}`), r.setDirectory)
	return sub, id, err
}

// setDirectory replaces the desired set from a full directory snapshot. It
// never waits for network work; it cancels a fetch made obsolete by the change.
func (r *peerSizingRuntime) setDirectory(nodes []noderec.DirectoryNode) {
	next := make(map[string]peerSizingTarget, len(nodes))
	for _, n := range nodes {
		if n.HostUUID == "" || n.HostUUID == r.self || n.ClusterUUID == "" || n.Policy == nil || n.Policy.Revision == "" {
			continue
		}
		next[n.HostUUID] = peerSizingTarget{host: n.HostUUID, cluster: n.ClusterUUID, revision: n.Policy.Revision, profiles: slices.Clone(n.Policy.Profiles)}
	}
	r.mu.Lock()
	r.desired = next
	if r.inflight != nil {
		if t, ok := next[r.inflightHost]; !ok || !t.equal(r.inflightTarget) {
			r.inflight()
		}
	}
	r.mu.Unlock()
	r.signal()
}

// peerSizingSweepInterval bounds how late an expiry or authority change seen
// only by the periodic sweep is withdrawn; the proxy also expires on its own.
const peerSizingSweepInterval = 5 * time.Second

func (r *peerSizingRuntime) run(ctx context.Context) {
	ticker := time.NewTicker(peerSizingSweepInterval)
	defer ticker.Stop()
	for {
		r.reconcile(ctx)
		select {
		case <-ctx.Done():
			return
		case <-r.wake:
		case <-ticker.C:
		}
	}
}

// admission reads the live authority for a cluster principal and the current
// engine-manager source.
func (r *peerSizingRuntime) admission(cluster string) (peerSizingAdmission, bool) {
	r.authority.Refresh()
	identity, ok := r.authority.PeerServiceIncarnation(cluster)
	if !ok || identity == "" {
		return peerSizingAdmission{}, false
	}
	epoch, ok := r.authority.PeerAdmissionEpoch(cluster)
	if !ok || epoch == 0 {
		return peerSizingAdmission{}, false
	}
	source := r.source()
	if source == nil {
		return peerSizingAdmission{}, false
	}
	return peerSizingAdmission{identity: identity, epoch: epoch, source: source}, true
}

// reconcile invalidates observations whose directory target, authority or age
// no longer holds, delivers, then refreshes due peers serially in lexical order.
func (r *peerSizingRuntime) reconcile(ctx context.Context) {
	r.invalidate()
	r.send(ctx, false)
	for _, target := range r.due() {
		if ctx.Err() != nil {
			return
		}
		r.refresh(ctx, target)
		r.send(ctx, false)
	}
}

func (r *peerSizingRuntime) invalidate() {
	r.mu.Lock()
	held := make(map[string]peerSizingObservation, len(r.observations))
	for host, obs := range r.observations {
		held[host] = obs
	}
	desired := r.desired
	r.mu.Unlock()
	now := r.now()
	for host, obs := range held {
		target, ok := desired[host]
		keep := ok && target.equal(obs.target) && now.Sub(obs.fetchedAt) < nodepolicy.PeerSizingFreshnessMs*time.Millisecond
		if keep {
			current, admitted := r.admission(target.cluster)
			keep = admitted && current == obs.admission
		}
		if !keep {
			r.mu.Lock()
			if r.observations[host].serial == obs.serial {
				delete(r.observations, host)
			}
			r.mu.Unlock()
		}
	}
}

func (r *peerSizingRuntime) due() []peerSizingTarget {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := r.now()
	var out []peerSizingTarget
	for host, target := range r.desired {
		if obs, ok := r.observations[host]; ok && now.Sub(obs.fetchedAt) < peerSizingRefreshInterval {
			continue
		}
		out = append(out, target)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].host < out[j].host })
	if len(out) > nodepolicy.MaxPeerSizingPeers {
		out = out[:nodepolicy.MaxPeerSizingPeers]
	}
	return out
}

// refresh performs one authenticated read for target. Any failure, malformed,
// unactivated or mismatched reply, or authority change withdraws the peer's
// held recipe immediately.
func (r *peerSizingRuntime) refresh(ctx context.Context, target peerSizingTarget) {
	before, ok := r.admission(target.cluster)
	if !ok {
		r.withdraw(target.host)
		return
	}
	fctx, cancel := context.WithCancel(ctx)
	r.mu.Lock()
	r.inflightHost, r.inflightTarget, r.inflight = target.host, target, cancel
	r.mu.Unlock()
	started := r.now()
	raw, err := r.fetch(fctx, before.source, target.host)
	r.mu.Lock()
	r.inflightHost, r.inflightTarget, r.inflight = "", peerSizingTarget{}, nil
	current, stillDesired := r.desired[target.host]
	r.mu.Unlock()
	cancel()
	if err != nil {
		slog.Debug("peer sizing read failed", "node", target.host, "err", peerSizingErrorClass(err))
		r.withdraw(target.host)
		return
	}
	profiles, runID, err := peerSizingRecipes(raw, target)
	after, admitted := r.admission(target.cluster)
	if err != nil || !stillDesired || !current.equal(target) || !admitted || after != before {
		if err != nil {
			slog.Debug("peer sizing read rejected", "node", target.host, "err", err)
		}
		r.withdraw(target.host)
		return
	}
	r.mu.Lock()
	r.serial++
	r.observations[target.host] = peerSizingObservation{target: target, admission: before, proxyRunID: runID, profiles: profiles, fetchedAt: started, serial: r.serial}
	r.mu.Unlock()
}

func (r *peerSizingRuntime) withdraw(host string) {
	r.mu.Lock()
	delete(r.observations, host)
	r.mu.Unlock()
}

func peerSizingErrorClass(err error) string {
	switch {
	case errors.Is(err, context.Canceled):
		return "cancelled"
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	default:
		return "unavailable"
	}
}

// peerSizingRecipes accepts a peer policy read only when it attests activation
// of exactly the advertised revision, and keeps only the advertised shared
// profiles whose identity matches the read. Withheld, local-only, unmatched
// and invalid profiles stay unknown; nothing else of the policy is retained.
func peerSizingRecipes(raw json.RawMessage, target peerSizingTarget) ([]nodepolicy.PeerSizingProfile, string, error) {
	var read struct {
		Policy struct {
			Revision string               `json:"revision"`
			Profiles []nodepolicy.Profile `json:"profiles"`
		} `json:"policy"`
		ActiveRequestPolicy *nodepolicy.ActiveRequestPolicy `json:"activeRequestPolicy"`
	}
	if err := json.Unmarshal(raw, &read); err != nil {
		return nil, "", errors.New("malformed peer policy read")
	}
	active := read.ActiveRequestPolicy
	if active == nil {
		return nil, "", errors.New("peer policy revision is not active")
	}
	if active.Revision != read.Policy.Revision || read.Policy.Revision != target.revision {
		return nil, "", errors.New("peer policy revision does not match its advertisement")
	}
	byKey := make(map[string]nodepolicy.Profile, len(read.Policy.Profiles))
	for _, p := range read.Policy.Profiles {
		byKey[p.Key] = p
	}
	recipes := []nodepolicy.PeerSizingProfile{}
	for _, advertised := range target.profiles {
		p, ok := byKey[advertised.Key]
		if !ok || p.LocalOnly || p.Engine != advertised.Engine || p.Model != advertised.Model {
			continue
		}
		recipe := nodepolicy.PeerSizingProfile{Key: p.Key, Engine: p.Engine, Model: p.Model, RequestOptions: bytes.Clone(p.RequestOptions), RequestOptionsMode: p.RequestOptionsMode}
		if recipe.Validate() != nil || len(recipes) == nodepolicy.MaxPeerSizingProfiles {
			continue
		}
		recipes = append(recipes, recipe)
	}
	return recipes, active.ProxyRunID, nil
}

// projection builds the bounded snapshot content in lexical host order. A peer
// that cannot fit the byte budget is omitted whole. keys identifies the
// observations it carries, independent of their advancing ages.
func (r *peerSizingRuntime) projection() ([]nodepolicy.PeerSizing, string) {
	r.mu.Lock()
	held := make([]peerSizingObservation, 0, len(r.observations))
	for _, obs := range r.observations {
		held = append(held, obs)
	}
	r.mu.Unlock()
	sort.Slice(held, func(i, j int) bool { return held[i].target.host < held[j].target.host })
	now := r.now()
	peers := []nodepolicy.PeerSizing{}
	var keys bytes.Buffer
	size := len(`{"revision":18446744073709551615,"peers":[]}`)
	for _, obs := range held {
		age := now.Sub(obs.fetchedAt).Milliseconds()
		if age < 0 {
			age = 0
		}
		if age >= nodepolicy.PeerSizingFreshnessMs || len(peers) == nodepolicy.MaxPeerSizingPeers {
			continue
		}
		peer := nodepolicy.PeerSizing{Host: obs.target.host, AdmissionIdentity: obs.admission.identity, AdmissionIncarnation: strconv.FormatUint(obs.admission.epoch, 10), PolicyRevision: obs.target.revision, ProxyRunID: obs.proxyRunID, AgeMs: age, Profiles: obs.profiles}
		if peer.Validate() != nil {
			continue
		}
		encoded, err := json.Marshal(peer)
		if err != nil || size+len(encoded)+1 > nodepolicy.MaxPeerSizingSnapshot {
			continue
		}
		size += len(encoded) + 1
		peers = append(peers, peer)
		fmt.Fprintf(&keys, "%s/%d;", obs.target.host, obs.serial)
	}
	return peers, keys.String()
}

// send delivers the current projection. The revision advances once per changed
// set of observations; a replay resends the current revision with advanced
// ages, which a proxy already holding it ignores.
func (r *peerSizingRuntime) send(ctx context.Context, replay bool) {
	if r == nil || r.deliver == nil {
		return
	}
	r.sendMu.Lock()
	defer r.sendMu.Unlock()
	peers, keys := r.projection()
	if r.sent.Revision != 0 && keys == r.sentKeys && !replay {
		return
	}
	if r.sent.Revision == 0 || keys != r.sentKeys {
		r.sent.Revision++
		r.sentKeys = keys
	}
	r.sent.Peers = peers
	params, err := json.Marshal(r.sent)
	if err != nil {
		return
	}
	r.deliver(ctx, params)
}

// deliverPeerSizing uses the same live-proxy targets and call pattern as the
// scheduler priority and routing cost feeds. Failures carry the engine and an
// operational error only, never recipe contents.
func (b *Broker) deliverPeerSizing(ctx context.Context, params json.RawMessage) {
	for _, target := range b.livePriorityTargets() {
		if _, rpcErr, err := target.proxy.Call(ctx, nodepolicy.MethodSetPeerSizing, params); err != nil {
			slog.Warn("node/set-peer-sizing call failed", "engine", target.engine, "err", err)
		} else if rpcErr != nil {
			slog.Warn("node/set-peer-sizing rejected", "engine", target.engine, "code", rpcErr.Code)
		}
	}
}

// sendPeerSizing replays the current snapshot after a proxy spawn without
// waiting for any fetch.
func (b *Broker) sendPeerSizing(ctx context.Context, replay bool) {
	b.peerSizing.send(ctx, replay)
}
