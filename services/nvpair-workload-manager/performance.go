// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
	"nvpair-shared/performancewire"
)

const (
	methodPerformancePublish      = "performance:publish"
	methodPerformanceReceived     = "performance:received"
	methodPerformanceStatus       = "performance:status"
	methodPerformanceCapabilities = "performance:capabilities"
	performanceHeartbeat          = 30 * time.Second
	performanceStaleAfter         = 90 * time.Second
	performancePeerLimit          = 128
	performanceSendWorkers        = 4
	performanceProbeAttempts      = 3
	performanceElapsedHeader      = "X-PAIR-Performance-Elapsed-Ms"
)

type performanceReceived struct {
	Summary           performancewire.Summary `json:"summary"`
	AdmissionIdentity string                  `json:"admissionIdentity"`
	SampleElapsedMs   int64                   `json:"sampleElapsedMs"`
}

// sampleElapsedMs includes the sender's publication offset and advances on the
// receiver's monotonic clock, even when heartbeats refresh lastSeenAgeMs.
type performanceStatus struct {
	MeasuringHost        string `json:"measuringHost"`
	AdmissionIncarnation string `json:"admissionIncarnation"`
	AdmissionIdentity    string `json:"admissionIdentity"`
	Status               string `json:"status"`
	Revision             int64  `json:"revision"`
	SampleElapsedMs      int64  `json:"sampleElapsedMs"`
	LastSeenAgeMs        int64  `json:"lastSeenAgeMs"`
}

type performanceReceipt struct {
	mu               sync.Mutex
	identity         string
	epoch            string
	revision         int64
	payload          []byte
	digest           [32]byte
	firstSeen        time.Time
	initialElapsedMs int64
	lastSeen         time.Time
	status           string
	retired          bool
}

type performanceSupport struct {
	identity    string
	generation  uint64
	attempts    int
	supported   bool
	unsupported bool
}

// performanceRelay has one latest outbound replacement and one wake token.
// Its sender and connection pool are separate from the lifecycle FIFO. Only
// short snapshot/map updates hold mu; no HTTP or broker writes hold it.
type performanceRelay struct {
	self           string
	mesh           *clustertrust.Mesh
	emit           func(string, any) error
	now            func() time.Time
	broadcaster    *Broadcaster
	mu             sync.Mutex
	latest         []byte
	latestIdentity string
	latestAt       time.Time
	wake           chan struct{}
	received       map[string]*performanceReceipt
	support        map[string]performanceSupport // sender-loop owned
}

func newPerformanceRelay(self string, mesh *clustertrust.Mesh, emit func(string, any) error) *performanceRelay {
	return &performanceRelay{self: self, mesh: mesh, emit: emit, now: time.Now,
		wake: make(chan struct{}, 1), received: make(map[string]*performanceReceipt), support: make(map[string]performanceSupport)}
}

func canonicalPerformanceEpoch(text string) (uint64, bool) {
	epoch, err := strconv.ParseUint(text, 10, 64)
	return epoch, err == nil && epoch > 0 && strconv.FormatUint(epoch, 10) == text
}

func (p *performanceRelay) localIdentity(summary performancewire.Summary) (string, bool) {
	p.mesh.Refresh()
	epoch, valid := canonicalPerformanceEpoch(summary.AdmissionIncarnation)
	current, ok := p.mesh.LocalAdmissionEpoch()
	// The authenticated peer_services contract binds host UUID to principal.
	// Discovery's legacy host -> clusterUuid aliases cannot grant this identity.
	if !valid || !ok || epoch != current || summary.MeasuringHost != p.self || p.self != p.mesh.NodeUUID() {
		return "", false
	}
	return p.self + ":" + summary.AdmissionIncarnation, true
}

func (p *performanceRelay) publish(raw json.RawMessage) error {
	summary, err := performancewire.DecodeSummary(raw)
	if err != nil {
		return err
	}
	raw, err = performancewire.EncodeSummary(summary)
	if err != nil {
		return err
	}
	identity, ok := p.localIdentity(summary)
	if !ok {
		return errors.New("summary is not from the current local admission")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	sameRevision := false
	if len(p.latest) > 0 && identity == p.latestIdentity {
		old, err := performancewire.DecodeSummary(p.latest)
		if err != nil {
			return err
		}
		if summary.Revision < old.Revision {
			return errors.New("stale local summary revision")
		}
		if summary.Revision == old.Revision && !bytes.Equal(raw, p.latest) {
			return errors.New("local summary changed without a new revision")
		}
		sameRevision = summary.Revision == old.Revision
	}
	if !sameRevision {
		p.latestAt = p.now()
	}
	p.latest, p.latestIdentity = bytes.Clone(raw), identity
	p.signal()
	return nil
}

func (p *performanceRelay) signal() {
	select {
	case p.wake <- struct{}{}:
	default:
	}
}

func (p *performanceRelay) run(ctx context.Context) {
	ticker := time.NewTicker(performanceHeartbeat)
	defer ticker.Stop()
	p.runTicks(ctx, ticker.C)
}

func (p *performanceRelay) runTicks(ctx context.Context, ticks <-chan time.Time) {
	// The timer never waits for a network round. Heartbeats coalesce with
	// publish/discovery wakes, and freshness still advances during blocked I/O.
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case <-ctx.Done():
				return
			case <-p.wake:
				p.sendRound(ctx)
			}
		}
	}()
	defer func() { <-done }()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticks:
			p.sweep()
			p.signal()
		}
	}
}

func (p *performanceRelay) snapshot() []byte {
	raw, _ := p.snapshotForSend()
	return raw
}

func (p *performanceRelay) snapshotForSend() ([]byte, int64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.latest) == 0 {
		return nil, 0
	}
	summary, err := performancewire.DecodeSummary(p.latest)
	identity, ok := p.localIdentity(summary)
	if err != nil || !ok || identity != p.latestIdentity {
		// Withhold dispatch without forgetting this epoch's revision/age anchor.
		// Temporary authority loss cannot make an identical republication new.
		return nil, 0
	}
	elapsed := p.now().Sub(p.latestAt).Milliseconds()
	if elapsed < 0 {
		elapsed = 0
	}
	if elapsed > performancewire.RetentionMs {
		return nil, 0
	}
	return bytes.Clone(p.latest), elapsed
}

func (p *performanceRelay) sendRound(ctx context.Context) {
	if p.broadcaster == nil {
		return
	}
	if p.snapshot() == nil {
		return
	}
	targets := p.broadcaster.peers.targets()
	jobs := make(chan target)
	var wg sync.WaitGroup
	var supportMu sync.Mutex
	next := make(map[string]performanceSupport)
	for range performanceSendWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range jobs {
				// Fail closed on legacy aliases, never trusting a relay mapping to
				// assert that somebody else's host UUID belongs to this principal.
				if t.id != t.uuid {
					continue
				}
				state := p.support[t.uuid]
				_, generation, routeOK := p.broadcaster.peers.targetUUID(t.uuid)
				identity, ok := p.broadcaster.mesh.PeerServiceIncarnation(t.uuid)
				if ok && routeOK {
					if state.identity != identity || state.generation != generation {
						state = performanceSupport{identity: identity, generation: generation}
					}
					state = p.sendPeer(ctx, t, state)
				}
				// Missing authority/route withholds traffic, not the known budget.
				// Only a positively observed identity/route change resets it.
				supportMu.Lock()
				next[t.uuid] = state
				supportMu.Unlock()
			}
		}()
	}
	for _, t := range targets {
		select {
		case jobs <- t:
		case <-ctx.Done():
		}
	}
	close(jobs)
	wg.Wait()
	p.support = next
}

func (p *performanceRelay) sendPeer(ctx context.Context, t target, state performanceSupport) performanceSupport {
	if state.unsupported || (!state.supported && state.attempts >= performanceProbeAttempts) {
		return state
	}
	b := p.broadcaster
	current := func() bool {
		b.mesh.Refresh()
		identity, ok := b.mesh.PeerServiceIncarnation(t.uuid)
		_, generation, routeOK := b.peers.targetUUID(t.uuid)
		return ok && routeOK && identity == state.identity && generation == state.generation
	}
	if !current() {
		return state
	}
	url := "https://" + b.addrs.Prefer(t.id, t.candidates) + eventsPath
	client := b.clients.Client(t.uuid, noderec.ServiceWorkload)
	if !state.supported {
		state.attempts++
		probe := []byte(`{"jsonrpc":"2.0","method":"performance:capabilities","params":{}}`)
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(probe))
		if err != nil {
			return state
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(req)
		if err != nil {
			return state
		}
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, 1025))
		_ = resp.Body.Close()
		if !current() {
			return state
		}
		if resp.StatusCode == http.StatusBadRequest || resp.StatusCode == http.StatusNotImplemented {
			state.unsupported = true
			return state
		}
		var capability struct {
			Version int `json:"version"`
		}
		decoder := json.NewDecoder(bytes.NewReader(body))
		decoder.DisallowUnknownFields()
		if readErr != nil || len(body) > 1024 || resp.StatusCode != http.StatusOK || decoder.Decode(&capability) != nil || decoder.Decode(&struct{}{}) != io.EOF || capability.Version != performancewire.SchemaVersion {
			return state
		}
		state.supported = true
	}
	// Revalidate the queued local admission and peer authority just before
	// dispatch. A changed admission withholds pending bytes until broker republishes.
	raw, elapsed := p.snapshotForSend()
	if raw == nil || !current() {
		return state
	}
	frame, err := json.Marshal(Message{JSONRPC: "2.0", Method: performancewire.MethodSummary, Params: raw})
	if err != nil {
		return state
	}
	headers := http.Header{performanceElapsedHeader: []string{strconv.FormatInt(elapsed, 10)}}
	_ = b.postWithHeaders(ctx, client, url, frame, headers) // one attempt; next heartbeat sends latest
	if !current() {
		return state
	}
	return state
}

func (p *performanceRelay) peerIdentity(r *http.Request, epochText string) (string, string, bool) {
	p.mesh.Refresh()
	principal, pinned := p.mesh.VerifyClientPin(r)
	identity, admitted := p.mesh.PeerServiceIncarnation(principal)
	epoch, canonical := canonicalPerformanceEpoch(epochText)
	current, ok := p.mesh.PeerAdmissionEpoch(principal)
	return principal, identity, pinned && admitted && canonical && ok && current == epoch
}

func (p *performanceRelay) receipt(host string) *performanceReceipt {
	p.mu.Lock()
	defer p.mu.Unlock()
	if state := p.received[host]; state != nil {
		return state
	}
	if len(p.received) >= performancePeerLimit {
		return nil
	}
	state := &performanceReceipt{}
	p.received[host] = state
	return state
}

func (p *performanceRelay) notify(method string, params any) error {
	if p.emit == nil {
		return errors.New("broker unavailable")
	}
	return p.emit(method, params)
}

func (p *performanceRelay) status(host string, state *performanceReceipt, now time.Time, status string) error {
	err := p.notify(methodPerformanceStatus, performanceStatus{MeasuringHost: host,
		AdmissionIncarnation: state.epoch, AdmissionIdentity: state.identity, Status: status,
		Revision: state.revision, SampleElapsedMs: state.initialElapsedMs + now.Sub(state.firstSeen).Milliseconds(), LastSeenAgeMs: now.Sub(state.lastSeen).Milliseconds()})
	if err == nil {
		state.status = status
	}
	return err
}

func (p *performanceRelay) handleSummary(w http.ResponseWriter, r *http.Request, raw json.RawMessage) {
	values := r.Header.Values(performanceElapsedHeader)
	if len(values) != 1 {
		http.Error(w, "invalid performance elapsed time", http.StatusBadRequest)
		return
	}
	elapsed, err := strconv.ParseUint(values[0], 10, 64)
	if err != nil || strconv.FormatUint(elapsed, 10) != values[0] || elapsed > uint64(performancewire.RetentionMs) {
		http.Error(w, "invalid performance elapsed time", http.StatusBadRequest)
		return
	}
	summary, err := performancewire.DecodeSummary(raw)
	if err != nil {
		http.Error(w, "invalid performance summary", http.StatusBadRequest)
		return
	}
	raw, err = performancewire.EncodeSummary(summary)
	if err != nil {
		http.Error(w, "invalid performance summary", http.StatusBadRequest)
		return
	}
	principal, identity, ok := p.peerIdentity(r, summary.AdmissionIncarnation)
	if !ok || principal != summary.MeasuringHost {
		http.Error(w, "performance admission mismatch", http.StatusForbidden)
		return
	}
	state := p.receipt(principal)
	if state == nil {
		http.Error(w, "performance capacity reached", http.StatusServiceUnavailable)
		return
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	p.mu.Lock()
	tracked := p.received[principal] == state
	p.mu.Unlock()
	if !tracked {
		http.Error(w, "performance admission retired", http.StatusServiceUnavailable)
		return
	}
	valid := func() bool {
		currentPrincipal, currentIdentity, ok := p.peerIdentity(r, summary.AdmissionIncarnation)
		return ok && currentPrincipal == principal && currentIdentity == identity
	}
	if !valid() {
		http.Error(w, "performance admission changed", http.StatusForbidden)
		return
	}
	if state.identity != identity {
		// A receiver-local admission/certificate change retires the payload,
		// but cannot lower the sender's revision floor within the same epoch.
		if state.epoch != summary.AdmissionIncarnation {
			state.revision, state.digest = 0, [32]byte{}
		}
		state.identity, state.epoch = identity, summary.AdmissionIncarnation
		state.payload, state.status, state.retired = nil, "", false
	}
	if summary.Revision < state.revision {
		w.WriteHeader(http.StatusOK)
		return
	}
	digest := sha256.Sum256(raw)
	if summary.Revision == state.revision && state.digest != digest {
		http.Error(w, "revision changed contents", http.StatusBadRequest)
		return
	}
	now := p.now()
	if summary.Revision > state.revision {
		if !valid() {
			http.Error(w, "performance admission changed", http.StatusForbidden)
			return
		}
		if err := p.notify(methodPerformanceReceived, performanceReceived{summary, identity, int64(elapsed)}); err != nil {
			http.Error(w, "broker unavailable", http.StatusInternalServerError)
			return
		}
		if !valid() {
			// A completed pipe write cannot be retracted. Explicitly retire its
			// identity so the broker can discard that just-emitted replacement.
			state.revision, state.digest, state.firstSeen, state.lastSeen = summary.Revision, digest, now, now
			state.initialElapsedMs = int64(elapsed)
			_ = p.status(principal, state, now, "unknown")
			state.payload = nil
			http.Error(w, "performance admission changed", http.StatusForbidden)
			return
		}
		state.revision, state.digest, state.payload, state.firstSeen = summary.Revision, digest, bytes.Clone(raw), now
		state.initialElapsedMs = int64(elapsed)
	}
	// A verified retry cannot make samples younger. Sender elapsed time also
	// covers time queued before this receiver's first successful delivery.
	currentElapsed := state.initialElapsedMs + now.Sub(state.firstSeen).Milliseconds()
	if int64(elapsed) > currentElapsed {
		state.initialElapsedMs += int64(elapsed) - currentElapsed
	}
	state.retired = false
	state.lastSeen = now
	if err := p.status(principal, state, now, "fresh"); err != nil {
		http.Error(w, "broker unavailable", http.StatusInternalServerError)
		return
	}
	if !valid() {
		_ = p.status(principal, state, p.now(), "unknown")
		state.payload = nil
		http.Error(w, "performance admission changed", http.StatusForbidden)
		return
	}
	w.WriteHeader(http.StatusOK)
}

func (p *performanceRelay) sweep() {
	p.mu.Lock()
	states := make(map[string]*performanceReceipt, len(p.received))
	for host, state := range p.received {
		states[host] = state
	}
	p.mu.Unlock()
	for host, state := range states {
		state.mu.Lock()
		p.mesh.Refresh()
		identity, ok := p.mesh.PeerServiceIncarnation(host)
		now := p.now()
		if !ok || identity != state.identity {
			if !state.retired && (state.revision == 0 || p.status(host, state, now, "unknown") == nil) {
				state.payload, state.retired = nil, true
			}
			// A missing local pin does not prove a new sender epoch. Keep the
			// bounded revision/digest record across local teardown and recovery.
		} else if state.revision > 0 && now.Sub(state.lastSeen) >= performanceStaleAfter && state.status != "stale" {
			_ = p.status(host, state, now, "stale")
		}
		state.mu.Unlock()
	}
}

func (m *Manager) handleLocalPerformance(msg *Message) {
	err := m.performance.publish(msg.Params)
	if msg.IsRequest() {
		if err != nil {
			_ = m.codec.RespondError(msg.ID, -32602, err.Error())
		} else {
			_ = m.codec.Respond(msg.ID, nil)
		}
	}
}

func (p *performanceRelay) capabilities(w http.ResponseWriter, r *http.Request, raw json.RawMessage) {
	// Capability discovery is authenticated application traffic, not a discovery
	// descriptor extension. The hash brackets the reply just as it brackets data.
	p.mesh.Refresh()
	principal, pinned := p.mesh.VerifyClientPin(r)
	identity, admitted := p.mesh.PeerServiceIncarnation(principal)
	if !pinned || !admitted {
		http.Error(w, "performance admission required", http.StatusForbidden)
		return
	}
	var empty struct{}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if dec.Decode(&empty) != nil || dec.Decode(&struct{}{}) != io.EOF || string(bytes.TrimSpace(raw)) == "null" {
		http.Error(w, "invalid performance capability request", http.StatusBadRequest)
		return
	}
	p.mesh.Refresh()
	current, ok := p.mesh.PeerServiceIncarnation(principal)
	_, pinned = p.mesh.VerifyClientPin(r)
	if !ok || !pinned || current != identity {
		http.Error(w, "performance admission changed", http.StatusForbidden)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprintf(w, `{"version":%d}`, performancewire.SchemaVersion)
}
