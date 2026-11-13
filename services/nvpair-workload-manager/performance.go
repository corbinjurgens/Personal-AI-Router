// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/performancewire"
)

const (
	methodPerformancePublish  = "performance:publish"
	methodPerformanceReceived = "performance:received"
	methodPerformanceStatus   = "performance:status"
	performanceStaleAfter     = 90 * time.Second
	performancePeerLimit      = 128
)

type performanceReceived struct {
	Summary           performancewire.Summary `json:"summary"`
	AdmissionIdentity string                  `json:"admissionIdentity"`
	SampleElapsedMs   int64                   `json:"sampleElapsedMs"`
}

// sampleElapsedMs includes the sender's publication offset and advances on the
// receiver's monotonic clock, even when stream keepalives refresh lastSeenAgeMs.
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
	statusAt         time.Time // when status was last emitted to the broker
	retired          bool
}

// performanceRelay holds the latest local summary, served to pinned peers on
// the performance stream (performance_stream.go), and one receipt per peer
// whose stream this node reads. hub carries the local revision as a change
// token; each stream builds its frame from snapshotForSend at write time. wake
// asks the subscription loop to re-evaluate its peer targets. Only short
// snapshot/map updates hold mu; no HTTP or broker writes hold it.
type performanceRelay struct {
	self           string
	mesh           *clustertrust.Mesh
	emit           func(string, any) error
	now            func() time.Time
	broadcaster    *Broadcaster                    // peer targets and routes
	streams        *clustertrust.PeerServiceClient // subscriber connections; no total timeout
	hub            clustertrust.SnapshotHub[int64]
	mu             sync.Mutex
	latest         []byte
	latestIdentity string
	latestAt       time.Time
	wake           chan struct{}
	received       map[string]*performanceReceipt
}

func newPerformanceRelay(self string, mesh *clustertrust.Mesh, emit func(string, any) error) *performanceRelay {
	p := &performanceRelay{self: self, mesh: mesh, emit: emit, now: time.Now,
		wake: make(chan struct{}, 1), received: make(map[string]*performanceReceipt)}
	p.hub.Limit = performancePeerLimit
	return p
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
	if err := p.record(raw, summary.Revision, identity); err != nil {
		return err
	}
	p.hub.Publish(summary.Revision)
	return nil
}

// record stores an accepted local summary. A republished identical revision
// keeps its publication time, so its sample age keeps advancing.
func (p *performanceRelay) record(raw []byte, revision int64, identity string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	sameRevision := false
	if len(p.latest) > 0 && identity == p.latestIdentity {
		old, err := performancewire.DecodeSummary(p.latest)
		if err != nil {
			return err
		}
		if revision < old.Revision {
			return errors.New("stale local summary revision")
		}
		if revision == old.Revision && !bytes.Equal(raw, p.latest) {
			return errors.New("local summary changed without a new revision")
		}
		sameRevision = revision == old.Revision
	}
	if !sameRevision {
		p.latestAt = p.now()
	}
	p.latest, p.latestIdentity = bytes.Clone(raw), identity
	return nil
}

func (p *performanceRelay) signal() {
	select {
	case p.wake <- struct{}{}:
	default:
	}
}

// snapshotForSend returns the latest local summary and its age in
// milliseconds, or nil while it must be withheld: the local admission changed
// or the publication is past retention.
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
		state.status, state.statusAt = status, now
	}
	return err
}

// applySummary applies one peer summary. authority resolves the sender's
// principal and admission identity for the summary's admission epoch; it is
// called once after decoding and again for every validity recheck. elapsedMs
// is the sender's age of the sample. It returns an HTTP-style status code and
// message; the stream subscriber ends the stream on anything but 200.
func (p *performanceRelay) applySummary(authority func(epoch string) (principal, identity string, ok bool), raw json.RawMessage, elapsedMs int64) (int, string) {
	summary, err := performancewire.DecodeSummary(raw)
	if err != nil {
		return http.StatusBadRequest, "invalid performance summary"
	}
	raw, err = performancewire.EncodeSummary(summary)
	if err != nil {
		return http.StatusBadRequest, "invalid performance summary"
	}
	principal, identity, ok := authority(summary.AdmissionIncarnation)
	if !ok || principal != summary.MeasuringHost {
		return http.StatusForbidden, "performance admission mismatch"
	}
	state := p.receipt(principal)
	if state == nil {
		return http.StatusServiceUnavailable, "performance capacity reached"
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	p.mu.Lock()
	tracked := p.received[principal] == state
	p.mu.Unlock()
	if !tracked {
		return http.StatusServiceUnavailable, "performance admission retired"
	}
	valid := func() bool {
		currentPrincipal, currentIdentity, ok := authority(summary.AdmissionIncarnation)
		return ok && currentPrincipal == principal && currentIdentity == identity
	}
	if !valid() {
		return http.StatusForbidden, "performance admission changed"
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
		return http.StatusOK, ""
	}
	digest := sha256.Sum256(raw)
	if summary.Revision == state.revision && state.digest != digest {
		return http.StatusBadRequest, "revision changed contents"
	}
	now := p.now()
	if summary.Revision > state.revision {
		if !valid() {
			return http.StatusForbidden, "performance admission changed"
		}
		if err := p.notify(methodPerformanceReceived, performanceReceived{summary, identity, elapsedMs}); err != nil {
			return http.StatusInternalServerError, "broker unavailable"
		}
		if !valid() {
			// A completed pipe write cannot be retracted. Explicitly retire its
			// identity so the broker can discard that just-emitted replacement.
			state.revision, state.digest, state.firstSeen, state.lastSeen = summary.Revision, digest, now, now
			state.initialElapsedMs = elapsedMs
			_ = p.status(principal, state, now, "unknown")
			state.payload = nil
			return http.StatusForbidden, "performance admission changed"
		}
		state.revision, state.digest, state.payload, state.firstSeen = summary.Revision, digest, bytes.Clone(raw), now
		state.initialElapsedMs = elapsedMs
	}
	// A verified retry cannot make samples younger. Sender elapsed time also
	// covers time queued before this receiver's first successful delivery.
	currentElapsed := state.initialElapsedMs + now.Sub(state.firstSeen).Milliseconds()
	if elapsedMs > currentElapsed {
		state.initialElapsedMs += elapsedMs - currentElapsed
	}
	state.retired = false
	state.lastSeen = now
	if err := p.status(principal, state, now, "fresh"); err != nil {
		return http.StatusInternalServerError, "broker unavailable"
	}
	if !valid() {
		_ = p.status(principal, state, p.now(), "unknown")
		state.payload = nil
		return http.StatusForbidden, "performance admission changed"
	}
	return http.StatusOK, ""
}

// refreshReceipt is a stream keepalive from host after an accepted frame. It
// refreshes lastSeen and re-asserts "fresh" to the broker at most every
// performanceStatusRefresh, while the receipt is live, authority still holds
// for its admission, and its sample is within retention (the sender withholds
// it past retention, so a keepalive cannot keep it fresh beyond that).
func (p *performanceRelay) refreshReceipt(host string, authority func(epoch string) (principal, identity string, ok bool)) {
	p.mu.Lock()
	state := p.received[host]
	p.mu.Unlock()
	if state == nil {
		return
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.revision == 0 || state.retired || state.payload == nil {
		return
	}
	if principal, identity, ok := authority(state.epoch); !ok || principal != host || identity != state.identity {
		return
	}
	now := p.now()
	if state.initialElapsedMs+now.Sub(state.firstSeen).Milliseconds() > performancewire.RetentionMs {
		return
	}
	state.lastSeen = now
	if state.status == "fresh" && now.Sub(state.statusAt) < performanceStatusRefresh {
		return
	}
	_ = p.status(host, state, now, "fresh")
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
