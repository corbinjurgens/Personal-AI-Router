// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"io"
	"math"
	"sort"
	"strconv"
	"sync"
	"time"

	"nvpair-shared/appdir"
	"nvpair-shared/clustertrust"
	"nvpair-shared/performancewire"
	"nvpair-ui-broker/performancestore"
)

const (
	methodPerformanceGet                = "performance:get"
	methodPerformanceSubscribe          = "performance:subscribe"
	methodPerformanceUnsubscribe        = "performance:unsubscribe"
	notifyPerformanceChanged            = "performance:changed"
	methodPerformanceSample             = "performance:sample"
	methodPerformanceReceived           = "performance:received"
	methodPerformanceStatus             = "performance:status"
	methodPerformancePublish            = "performance:publish"
	performanceSaveInterval             = 5 * time.Second
	performancePeerLimit                = 128
	performanceStaleAfterMs       int64 = 90_000
	performanceDisconnectExpiryMs int64 = 3_600_000
)

// The wire body never ages in place. Offset clocks only increase, including
// under a backwards test/wall clock, and are applied to a decoded projection.
type performancePeer struct {
	identity, epoch          string
	revision                 int64
	digest                   [32]byte
	raw                      []byte
	clock                    time.Time
	reachClock               time.Time
	elapsedMs, lastSeenAgeMs int64
	status                   string
	retired                  bool
}

type performanceRuntime struct {
	mu                                sync.Mutex
	store                             *performancestore.Store
	host                              string
	mesh                              *clustertrust.Mesh
	now                               func() time.Time
	peers                             map[string]*performancePeer
	blocked                           bool // failed quarantine leaves persistence disabled
	persistenceError                  string
	expiredPeers                      uint64
	subscribed                        bool
	frozen                            []byte
	frozenRevision                    int64
	readyRevision                     int64
	readyEpoch                        string
	checkpoint                        bool
	manager                           *workloadManagerProcess
	managerGeneration                 uint64
	checkpointGeneration              uint64
	saveWake, publishWake, changeWake chan struct{}
}

type performanceNode struct {
	MeasuringHost  string                   `json:"measuringHost"`
	Status         string                   `json:"status"`
	LastSeenAgeMs  int64                    `json:"lastSeenAgeMs"`
	Summary        *performancewire.Summary `json:"summary,omitempty"`
	ExpiredEntries int                      `json:"expiredEntries"`
}

type performanceProjection struct {
	Local       performanceNode   `json:"local"`
	Peers       []performanceNode `json:"peers"`
	Persistence struct {
		Degraded           bool                   `json:"degraded"`
		PublicationBlocked bool                   `json:"publicationBlocked"`
		Error              string                 `json:"error,omitempty"`
		Stats              performancestore.Stats `json:"stats"`
	} `json:"persistence"`
	ExpiredPeers uint64 `json:"expiredPeers"`
}

type performanceReceived struct {
	Summary           json.RawMessage `json:"summary"`
	AdmissionIdentity string          `json:"admissionIdentity"`
	SampleElapsedMs   int64           `json:"sampleElapsedMs"`
}

type performanceStatus struct {
	MeasuringHost        string `json:"measuringHost"`
	AdmissionIncarnation string `json:"admissionIncarnation"`
	AdmissionIdentity    string `json:"admissionIdentity"`
	Status               string `json:"status"`
	Revision             int64  `json:"revision"`
	SampleElapsedMs      int64  `json:"sampleElapsedMs"`
	LastSeenAgeMs        int64  `json:"lastSeenAgeMs"`
}

func newPerformanceRuntime(host, path, clusterDir string) *performanceRuntime {
	p := &performanceRuntime{host: host, store: performancestore.New(host, path), mesh: clustertrust.Open(clusterDir),
		now: time.Now, peers: make(map[string]*performancePeer), checkpoint: true,
		saveWake: make(chan struct{}, 1), publishWake: make(chan struct{}, 1), changeWake: make(chan struct{}, 1)}
	if err := p.store.Load(); err != nil {
		p.blocked, p.persistenceError = true, err.Error()
	}
	if path == "" {
		p.blocked = true
		p.persistenceError = "performance history directory unavailable"
	}
	return p
}

func performanceSignal(ch chan struct{}) {
	select {
	case ch <- struct{}{}:
	default:
	}
}

// Initialized before worker readers start. Separate save, pipe and client
// goroutines keep disk/pipe/client backpressure off sample and lifecycle intake.
func (b *Broker) startPerformance(ctx context.Context) func() {
	path, _ := appdir.Path(performancestore.FileName)
	b.performance = newPerformanceRuntime(b.nodeID, path, b.clusterDir)
	p := b.performance
	fctx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(performanceSaveInterval)
		defer ticker.Stop()
		p.runSaves(fctx, ticker.C)
	}()
	go p.publishLoop(fctx)
	go func() {
		ticker := time.NewTicker(performanceSaveInterval)
		defer ticker.Stop()
		p.runSweeps(fctx, ticker.C)
	}()
	go b.performanceChangeLoop(fctx)
	return func() {
		cancel()
		select {
		case <-done:
		case <-time.After(workloadHistoryFlushJoinTimeout):
		}
	}
}

func (p *performanceRuntime) runSaves(ctx context.Context, ticks <-chan time.Time) {
	p.flush()
	for {
		select {
		case <-ctx.Done():
			p.flush()
			return
		case <-ticks:
			p.flush()
		case <-p.saveWake:
			p.mu.Lock()
			checkpoint := p.checkpoint
			p.mu.Unlock()
			if checkpoint {
				p.flush()
			}
		}
	}
}

func (p *performanceRuntime) localEpoch() (string, bool) {
	p.mesh.Refresh()
	epoch, ok := p.mesh.LocalAdmissionEpoch()
	return strconv.FormatUint(epoch, 10), ok && p.mesh.NodeUUID() == p.host
}

// flush is owned by the save loop. Checkpoint, not a synthetic sample, assigns
// a new durable revision on startup, manager restart or local epoch change.
func (p *performanceRuntime) flush() {
	epoch, admitted := p.localEpoch()
	p.mu.Lock()
	checkpoint := p.checkpoint || (admitted && p.readyEpoch != "" && p.readyEpoch != epoch)
	generation, blocked := p.managerGeneration, p.blocked
	p.mu.Unlock()
	if blocked {
		return
	}
	var err error
	if checkpoint {
		err = p.store.Checkpoint()
	} else {
		err = p.store.Save()
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if err != nil {
		p.persistenceError = err.Error()
		p.checkpoint = true
		performanceSignal(p.changeWake)
		return // even an adopted ErrReplaced revision is withheld until a clean checkpoint
	}
	p.persistenceError = ""
	if checkpoint {
		p.checkpointGeneration = generation
		p.checkpoint = p.managerGeneration != generation
		// A startup checkpoint already in progress also covers the first
		// manager. Only a later manager incarnation needs a separate revision.
		if generation == 0 && p.managerGeneration == 1 {
			p.checkpointGeneration, p.checkpoint = 1, false
		}
	}
	if !admitted {
		return
	}
	currentEpoch, current := p.localEpoch()
	if !current || epoch != currentEpoch {
		p.checkpoint = true
		return
	}
	summary, ok := p.store.Published(epoch)
	if !ok {
		return
	}
	if summary.Revision == p.readyRevision {
		if epoch != p.readyEpoch {
			p.checkpoint = true
			performanceSignal(p.saveWake)
		}
		return
	}
	p.readyRevision, p.readyEpoch = summary.Revision, epoch
	performanceSignal(p.publishWake)
	performanceSignal(p.changeWake)
}

func (b *Broker) replayPerformanceToManager(wm *workloadManagerProcess) {
	p := b.performance
	if p == nil || wm == nil {
		return
	}
	p.mu.Lock()
	if p.manager != wm {
		p.manager, p.managerGeneration = wm, p.managerGeneration+1
		if p.managerGeneration == 1 && p.readyRevision > 0 && !p.checkpoint {
			p.checkpointGeneration = 1
			performanceSignal(p.publishWake)
		} else {
			p.checkpoint = true
			performanceSignal(p.saveWake)
		}
	}
	p.mu.Unlock()
}

func (p *performanceRuntime) publication() (*workloadManagerProcess, []byte) {
	p.mu.Lock()
	defer p.mu.Unlock()
	epoch, admitted := p.localEpoch()
	if !admitted || p.blocked || p.checkpoint || p.checkpointGeneration != p.managerGeneration ||
		epoch != p.readyEpoch || p.readyRevision == 0 {
		return nil, nil
	}
	if p.frozenRevision != p.readyRevision {
		// Freeze only when the sender selects the newest durable revision. A
		// pending revision's source ages include its wait behind earlier writes.
		summary, ok := p.store.Published(epoch)
		if !ok || summary.Revision != p.readyRevision {
			return nil, nil
		}
		raw, err := performancewire.EncodeSummary(summary)
		if err != nil {
			return nil, nil
		}
		p.frozen, p.frozenRevision = raw, summary.Revision
	}
	if currentEpoch, ok := p.localEpoch(); !ok || currentEpoch != epoch {
		return nil, nil
	}
	return p.manager, bytes.Clone(p.frozen)
}

func (p *performanceRuntime) publishLoop(ctx context.Context) {
	retry := time.NewTicker(performanceSaveInterval)
	defer retry.Stop()
	p.runPublications(ctx, retry.C)
}

func (p *performanceRuntime) runPublications(ctx context.Context, retries <-chan time.Time) {
	var sentManager *workloadManagerProcess
	var sentRaw []byte
	for {
		select {
		case <-ctx.Done():
			return
		case <-p.publishWake:
		case <-retries:
		}
		wm, raw := p.publication()
		if wm == nil {
			continue
		}
		if wm == sentManager && bytes.Equal(raw, sentRaw) {
			continue // the manager, rather than the broker, owns peer heartbeats
		}
		// No runtime lock is held during a potentially blocked worker pipe write.
		if wm.Forward(methodPerformancePublish, raw) == nil {
			p.mu.Lock()
			if p.manager == wm && p.readyRevision == p.frozenRevision && bytes.Equal(raw, p.frozen) {
				sentManager, sentRaw = wm, raw
			}
			p.mu.Unlock()
		}
	}
}

// ingestPerformanceSample is the future T2 producer boundary. It is deliberately
// process-scoped: one proxy notification records one execution, not one per facade.
func (b *Broker) ingestPerformanceSample(raw json.RawMessage) error {
	if b.performance == nil {
		return errors.New("performance store unavailable")
	}
	var sample performancewire.Sample
	if err := decodePerformance(raw, &sample, performancewire.MaxPayloadBytes); err != nil {
		return err
	}
	if err := b.performance.store.Record(sample); err != nil {
		return err
	}
	performanceSignal(b.performance.changeWake)
	return nil
}

func decodePerformance(raw []byte, out any, limit int) error {
	if len(raw) == 0 || len(raw) > limit || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return errors.New("invalid performance frame size")
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(out); err != nil {
		return err
	}
	if dec.Decode(&struct{}{}) != io.EOF {
		return errors.New("trailing performance data")
	}
	return nil
}

func performanceFields(raw []byte, names ...string) bool {
	var fields map[string]json.RawMessage
	if json.Unmarshal(raw, &fields) != nil || len(fields) != len(names) {
		return false
	}
	for _, name := range names {
		value, ok := fields[name]
		if !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return false
		}
	}
	return true
}

func performanceEpoch(text string) (uint64, bool) {
	epoch, err := strconv.ParseUint(text, 10, 64)
	return epoch, err == nil && epoch > 0 && strconv.FormatUint(epoch, 10) == text
}

func (p *performanceRuntime) peerAuthority(host, epochText, identity string) bool {
	p.mesh.Refresh()
	epoch, canonical := performanceEpoch(epochText)
	current, admitted := p.mesh.PeerServiceIncarnation(host)
	currentEpoch, ok := p.mesh.PeerAdmissionEpoch(host)
	return canonical && admitted && ok && current == identity && currentEpoch == epoch
}

func performanceAdd(a, b int64) int64 {
	if b > math.MaxInt64-a {
		return math.MaxInt64
	}
	return a + b
}

func (state *performancePeer) advance(now time.Time) {
	if elapsed := now.Sub(state.clock).Milliseconds(); elapsed > 0 {
		state.elapsedMs = performanceAdd(state.elapsedMs, elapsed)
		state.clock = state.clock.Add(time.Duration(elapsed) * time.Millisecond)
	}
	if elapsed := now.Sub(state.reachClock).Milliseconds(); elapsed > 0 {
		state.lastSeenAgeMs = performanceAdd(state.lastSeenAgeMs, elapsed)
		state.reachClock = state.reachClock.Add(time.Duration(elapsed) * time.Millisecond)
	}
}

func (p *performanceRuntime) retire(state *performancePeer) {
	state.raw, state.status, state.retired = nil, "unknown", true
}

// consumePerformanceNotification is the sole private workload-manager consumer.
// Authority is checked before mutation and again afterwards; retired records
// retain revision/digest floors so a delayed equal revision cannot resurrect data.
func (b *Broker) consumePerformanceNotification(method string, raw json.RawMessage) {
	p := b.performance
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	switch method {
	case methodPerformanceReceived:
		var in performanceReceived
		if len(raw) > performancewire.MaxPayloadBytes+2048 || !performanceFields(raw, "summary", "admissionIdentity", "sampleElapsedMs") || decodePerformance(raw, &in, performancewire.MaxPayloadBytes+2048) != nil || in.SampleElapsedMs < 0 ||
			in.SampleElapsedMs > performancewire.RetentionMs {
			return
		}
		summary, err := performancewire.DecodeSummary(in.Summary)
		if err != nil || !p.peerAuthority(summary.MeasuringHost, summary.AdmissionIncarnation, in.AdmissionIdentity) {
			return
		}
		body, err := performancewire.EncodeSummary(summary)
		if err != nil {
			return
		}
		state := p.peers[summary.MeasuringHost]
		if state == nil {
			if len(p.peers) >= performancePeerLimit {
				return
			}
			state = &performancePeer{}
			p.peers[summary.MeasuringHost] = state
		}
		if state.epoch != summary.AdmissionIncarnation {
			*state = performancePeer{epoch: summary.AdmissionIncarnation}
		}
		if state.identity != "" && state.identity != in.AdmissionIdentity {
			p.retire(state)
		}
		digest := sha256.Sum256(body)
		if summary.Revision < state.revision || (summary.Revision == state.revision && (state.retired || state.digest != digest)) {
			return
		}
		now := p.now()
		state.advance(now)
		if summary.Revision > state.revision {
			state.raw, state.revision, state.digest = body, summary.Revision, digest
			state.elapsedMs, state.clock, state.retired = in.SampleElapsedMs, now, false
		}
		state.identity = in.AdmissionIdentity
		state.elapsedMs = max(state.elapsedMs, in.SampleElapsedMs)
		state.lastSeenAgeMs, state.status = 0, "fresh"
		state.reachClock = now
		if !p.peerAuthority(summary.MeasuringHost, state.epoch, state.identity) {
			p.retire(state)
		}
	case methodPerformanceStatus:
		var in performanceStatus
		if len(raw) > 2048 || !performanceFields(raw, "measuringHost", "admissionIncarnation", "admissionIdentity", "status", "revision", "sampleElapsedMs", "lastSeenAgeMs") || decodePerformance(raw, &in, 2048) != nil || in.Revision <= 0 || in.SampleElapsedMs < 0 ||
			in.LastSeenAgeMs < 0 {
			return
		}
		if in.Status != "fresh" && in.Status != "stale" && in.Status != "unknown" {
			return
		}
		state := p.peers[in.MeasuringHost]
		if state == nil || state.identity != in.AdmissionIdentity || state.epoch != in.AdmissionIncarnation || state.revision != in.Revision {
			return
		}
		now := p.now()
		state.advance(now)
		state.elapsedMs = max(state.elapsedMs, in.SampleElapsedMs)
		if in.Status == "unknown" || !p.peerAuthority(in.MeasuringHost, in.AdmissionIncarnation, in.AdmissionIdentity) {
			p.retire(state)
		} else if !state.retired {
			state.status, state.lastSeenAgeMs = in.Status, in.LastSeenAgeMs
			state.reachClock = now
			if !p.peerAuthority(in.MeasuringHost, state.epoch, state.identity) {
				p.retire(state)
			}
		}
	default:
		return
	}
	performanceSignal(p.changeWake)
}

func (p *performanceRuntime) sweepLocked() {
	for host, state := range p.peers {
		state.advance(p.now())
		if !p.peerAuthority(host, state.epoch, state.identity) {
			p.retire(state)
		}
		if state.lastSeenAgeMs >= performanceDisconnectExpiryMs && len(state.raw) > 0 {
			p.retire(state)
			p.expiredPeers++
		} else if state.status == "fresh" && state.lastSeenAgeMs >= performanceStaleAfterMs {
			state.status = "stale"
		}
	}
}

func (p *performanceRuntime) sweep() { p.mu.Lock(); defer p.mu.Unlock(); p.sweepLocked() }

func (p *performanceRuntime) runSweeps(ctx context.Context, ticks <-chan time.Time) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticks:
			p.sweep()
			performanceSignal(p.changeWake)
		}
	}
}

func (p *performanceRuntime) projection() performanceProjection {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.sweepLocked()
	out := performanceProjection{Peers: []performanceNode{}, ExpiredPeers: p.expiredPeers}
	local := p.store.Local()
	out.Local = performanceNode{MeasuringHost: local.MeasuringHost, Status: "fresh", Summary: &local}
	if len(local.Entries) == 0 {
		out.Local.Status = "missing"
	}
	out.Persistence.Stats = p.store.Stats()
	out.Persistence.Error, out.Persistence.Degraded, out.Persistence.PublicationBlocked = p.persistenceError, p.persistenceError != "", p.blocked
	hosts := make(map[string]bool)
	for host := range p.peers {
		hosts[host] = true
	}
	for _, host := range p.mesh.PeerUUIDs() {
		if _, ok := p.mesh.PeerServiceIncarnation(host); ok {
			hosts[host] = true
		}
	}
	for host := range hosts {
		node := performanceNode{MeasuringHost: host, Status: "missing"}
		if state := p.peers[host]; state != nil {
			if len(state.raw) > 0 && !p.peerAuthority(host, state.epoch, state.identity) {
				p.retire(state)
			}
			node.Status, node.LastSeenAgeMs = state.status, state.lastSeenAgeMs
			if len(state.raw) > 0 {
				summary, err := performancewire.DecodeSummary(state.raw)
				if err == nil {
					entries := summary.Entries[:0]
					for i := range summary.Entries {
						entry := &summary.Entries[i]
						entry.NewestAgeMs = performanceAdd(entry.NewestAgeMs, state.elapsedMs)
						entry.OldestAgeMs = performanceAdd(entry.OldestAgeMs, state.elapsedMs)
						if entry.NewestAgeMs > summary.Window.MaxAgeMs {
							node.ExpiredEntries++
							continue
						}
						entries = append(entries, *entry)
					}
					summary.Entries = entries
					if p.peerAuthority(host, state.epoch, state.identity) {
						node.Summary = &summary
					} else {
						p.retire(state)
						node.Status = "unknown"
					}
				}
			}
		}
		out.Peers = append(out.Peers, node)
	}
	sort.Slice(out.Peers, func(i, j int) bool { return out.Peers[i].MeasuringHost < out.Peers[j].MeasuringHost })
	return out
}

func (b *Broker) handlePerformanceRequest(msg *Message) bool {
	if msg.Method != methodPerformanceGet && msg.Method != methodPerformanceSubscribe && msg.Method != methodPerformanceUnsubscribe {
		return false
	}
	if !msg.IsRequest() {
		return true
	}
	if len(msg.Params) > 0 && string(msg.Params) != "null" {
		var empty struct{}
		if err := decodePerformance(msg.Params, &empty, 128); err != nil {
			_ = b.codec.RespondError(msg.ID, -32602, "performance methods accept empty params")
			return true
		}
	}
	p := b.performance
	if p == nil {
		_ = b.codec.RespondError(msg.ID, -32000, "performance unavailable")
		return true
	}
	if msg.Method == methodPerformanceGet {
		_ = b.codec.Respond(msg.ID, p.projection())
		return true
	}
	p.mu.Lock()
	p.subscribed = msg.Method == methodPerformanceSubscribe
	subscribed := p.subscribed
	p.mu.Unlock()
	_ = b.codec.Respond(msg.ID, SubscriptionResult{Subscribed: subscribed})
	if subscribed {
		performanceSignal(p.changeWake)
	}
	return true
}

func (b *Broker) performanceChangeLoop(ctx context.Context) {
	p := b.performance
	for {
		select {
		case <-ctx.Done():
			return
		case <-p.changeWake:
		}
		p.mu.Lock()
		subscribed := p.subscribed
		p.mu.Unlock()
		if !subscribed {
			continue
		}
		projection := p.projection()
		p.mu.Lock()
		subscribed = p.subscribed
		p.mu.Unlock()
		if subscribed {
			_ = b.codec.Notify(notifyPerformanceChanged, projection)
		}
	}
}
