// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/performancewire"
	"nvpair-ui-broker/performancestore"
)

// These fixtures use certificate files, memory codecs and fake stdin writers.
// No fixture launches a child, opens a listener or accesses real app data.
type brokerPerformanceFixture struct {
	b         *Broker
	dir, path string
	now       time.Time
	identity  string
}

func performanceJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode fixture: %v", err)
	}
	return raw
}

func newBrokerPerformanceFixture(t *testing.T) *brokerPerformanceFixture {
	t.Helper()
	dir := t.TempDir()
	clustertrusttest.Join(t, dir, "cluster", "self", "peer")
	f := &brokerPerformanceFixture{dir: dir, path: filepath.Join(t.TempDir(), performancestore.FileName), now: time.Now()}
	f.admitPeer(t, "peer", 7)
	p := newPerformanceRuntime("self", f.path, dir)
	p.now = func() time.Time { return f.now }
	f.b = &Broker{nodeID: "self", clusterDir: dir, performance: p}
	var ok bool
	f.identity, ok = p.mesh.PeerServiceIncarnation("peer")
	if !ok {
		t.Fatal("fixture must establish authoritative peer admission")
	}
	return f
}

func (f *brokerPerformanceFixture) admitPeer(t *testing.T, host string, epoch uint64) {
	t.Helper()
	path := filepath.Join(f.dir, "trusted", host+".json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read pin: %v", err)
	}
	var pin map[string]json.RawMessage
	if err := json.Unmarshal(raw, &pin); err != nil {
		t.Fatalf("decode pin: %v", err)
	}
	pin["clusterId"] = performanceJSON(t, "cluster")
	pin["admissionEpoch"] = performanceJSON(t, epoch)
	if err := os.WriteFile(path, performanceJSON(t, pin), 0o600); err != nil {
		t.Fatalf("write pin: %v", err)
	}
	member := []map[string]any{{"nodeUuid": host, "state": "member", "clusterId": "cluster", "admissionEpoch": epoch}}
	if err := os.WriteFile(filepath.Join(f.dir, "members.json"), performanceJSON(t, member), 0o600); err != nil {
		t.Fatalf("write roster: %v", err)
	}
}

func performanceTestSample(seq uint64) performancewire.Sample {
	return performancewire.Sample{Scope: performancewire.ScopeLocalExecution, Producer: "proxy-process", Seq: seq,
		Key:         performancewire.ProfileKey{Host: "self", Engine: "ollama", Model: "model", Fingerprint: "profile"},
		Measurement: performancewire.Measurement{Outcome: performancewire.OutcomeCompleted}}
}

func performanceTestSummary(revision int64) performancewire.Summary {
	return performancewire.Summary{Version: 1, MeasuringHost: "peer", AdmissionIncarnation: "7", Revision: revision,
		Window: performancewire.Window{MaxAgeMs: performancewire.RetentionMs, MaxSamplesPerKey: 64},
		Entries: []performancewire.Entry{{Key: performancewire.ProfileKey{Host: "peer", Engine: "ollama", Model: "model", Fingerprint: "profile"},
			Samples: 1, Completed: 1, NewestAgeMs: 100, OldestAgeMs: 100, NewestObservedAtMs: 1000}}}
}

func (f *brokerPerformanceFixture) receive(t *testing.T, summary performancewire.Summary, offset int64) {
	t.Helper()
	raw, err := performancewire.EncodeSummary(summary)
	if err != nil {
		t.Fatalf("encode summary: %v", err)
	}
	f.b.forwardWorkloadManagerNotification(methodPerformanceReceived, performanceJSON(t, performanceReceived{raw, f.identity, offset}))
}

func (f *brokerPerformanceFixture) status(t *testing.T, status string, offset, seen int64) {
	t.Helper()
	f.b.forwardWorkloadManagerNotification(methodPerformanceStatus, performanceJSON(t, performanceStatus{
		MeasuringHost: "peer", AdmissionIncarnation: "7", AdmissionIdentity: f.identity,
		Status: status, Revision: 10, SampleElapsedMs: offset, LastSeenAgeMs: seen}))
}

func requirePerformancePeer(t *testing.T, p *performanceRuntime) performanceNode {
	t.Helper()
	projection := p.projection()
	for _, peer := range projection.Peers {
		if peer.MeasuringHost == "peer" {
			return peer
		}
	}
	t.Fatal("projection is missing peer")
	return performanceNode{}
}

func TestPerformancePublicationRequiresCurrentLocalAdmission(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	p := f.b.performance
	p.manager = &workloadManagerProcess{}
	p.flush()
	_, raw := p.publication()
	summary, err := performancewire.DecodeSummary(raw)
	if err != nil {
		t.Fatalf("decode admitted publication: %v", err)
	}
	if summary.MeasuringHost != "self" || summary.AdmissionIncarnation != "1" || summary.Revision <= 0 {
		t.Fatalf("publication = %+v", summary)
	}
	if err := os.Remove(filepath.Join(f.dir, "admission.json")); err != nil {
		t.Fatalf("remove admission: %v", err)
	}
	if wm, raw := p.publication(); wm != nil || raw != nil {
		t.Fatal("publication survived admission loss")
	}
	clustertrusttest.WriteAdmission(t, f.dir, "cluster", 2)
	if _, raw := p.publication(); raw != nil {
		t.Fatal("old epoch publication survived local readmission")
	}
	p.flush()
	_, next := p.publication()
	updated, err := performancewire.DecodeSummary(next)
	if err != nil {
		t.Fatalf("decode readmission publication: %v", err)
	}
	if updated.Revision <= summary.Revision || updated.AdmissionIncarnation != "2" {
		t.Fatalf("readmission = %+v", updated)
	}
}

func TestPerformancePublicationFreezesSameRevision(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	p := f.b.performance
	p.manager = &workloadManagerProcess{}
	if err := f.b.ingestPerformanceSample(performanceJSON(t, performanceTestSample(1))); err != nil {
		t.Fatal(err)
	}
	p.flush()
	_, first := p.publication()
	time.Sleep(5 * time.Millisecond)
	p.flush()
	_, second := p.publication()
	if !bytes.Equal(first, second) || len(first) == 0 {
		t.Fatal("same revision changed source ages/body")
	}
	clustertrusttest.WriteAdmission(t, f.dir, "cluster", 1)
	_, third := p.publication()
	if !bytes.Equal(first, third) {
		t.Fatal("same epoch refresh changed frozen bytes")
	}
}

func TestPerformanceManagerRestartCheckpointsExactlyOnce(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	p := f.b.performance
	firstManager := &workloadManagerProcess{}
	f.b.replayPerformanceToManager(firstManager)
	p.flush()
	_, first := p.publication()
	a, err := performancewire.DecodeSummary(first)
	if err != nil {
		t.Fatal(err)
	}
	f.b.replayPerformanceToManager(firstManager)
	p.flush()
	if p.store.Stats().PublishedRevision != a.Revision {
		t.Fatal("duplicate replay checkpointed the same manager twice")
	}
	f.b.replayPerformanceToManager(&workloadManagerProcess{})
	if _, raw := p.publication(); raw != nil {
		t.Fatal("manager replay published before checkpoint")
	}
	p.flush()
	_, second := p.publication()
	b, err := performancewire.DecodeSummary(second)
	if err != nil {
		t.Fatal(err)
	}
	if b.Revision <= a.Revision {
		t.Fatalf("restart revision = %d, previous = %d", b.Revision, a.Revision)
	}
}

func TestPerformanceInitialManagerReusesBrokerStartupCheckpoint(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	p := f.b.performance
	p.flush()
	startupRevision := p.store.Stats().PublishedRevision
	f.b.replayPerformanceToManager(&workloadManagerProcess{})
	wm, raw := p.publication()
	summary, err := performancewire.DecodeSummary(raw)
	if err != nil {
		t.Fatal(err)
	}
	if wm == nil || summary.Revision != startupRevision || p.checkpoint || len(p.saveWake) != 0 {
		t.Fatal("first manager duplicated the completed startup checkpoint")
	}
}

func TestPerformanceBrokerRestartPreservesHistoryAndIncreasesRevision(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	p := f.b.performance
	if err := f.b.ingestPerformanceSample(performanceJSON(t, performanceTestSample(1))); err != nil {
		t.Fatal(err)
	}
	p.flush()
	before := p.store.Stats().PublishedRevision
	priorAge := p.store.Local().Entries[0].NewestAgeMs
	restarted := newPerformanceRuntime("self", f.path, f.dir)
	restarted.flush()
	local := restarted.store.Local()
	if len(local.Entries) != 1 || local.Entries[0].Samples != 1 || local.Entries[0].NewestAgeMs < priorAge {
		t.Fatalf("restart aggregates = %+v", local)
	}
	if restarted.store.Stats().PublishedRevision <= before {
		t.Fatal("broker restart reused durable revision")
	}
}

func TestPerformanceSampleHookIsStrictAndDeduplicatesProcessSample(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	raw := performanceJSON(t, performanceTestSample(1))
	if !f.b.routeProcessScopedProxyNotification(methodPerformanceSample, raw) {
		t.Fatal("sample hook was not consumed")
	}
	f.b.routeProcessScopedProxyNotification(methodPerformanceSample, raw)
	if stats := f.b.performance.store.Stats(); stats.Samples != 1 || stats.Duplicates != 1 {
		t.Fatalf("sample stats = %+v", stats)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	fields["unexpected"] = performanceJSON(t, true)
	if err := f.b.ingestPerformanceSample(performanceJSON(t, fields)); err == nil {
		t.Fatal("unknown sample field accepted")
	}
	wrongHost := performanceTestSample(2)
	wrongHost.Key.Host = "peer"
	if err := f.b.ingestPerformanceSample(performanceJSON(t, wrongHost)); err == nil {
		t.Fatal("remote sample entered authoritative local store")
	}
	if err := f.b.ingestPerformanceSample(append(raw, []byte(" {}")...)); err == nil {
		t.Fatal("trailing sample data accepted")
	}
}

func TestPerformancePeerReplacementNeverAddsDuplicateTotals(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	f.receive(t, performanceTestSummary(10), 300)
	f.receive(t, performanceTestSummary(10), 100)
	peer := requirePerformancePeer(t, f.b.performance)
	if peer.Summary == nil || peer.Summary.Entries[0].Samples != 1 || peer.Summary.Entries[0].NewestAgeMs != 400 {
		t.Fatalf("duplicate projection = %+v", peer)
	}
	changed := performanceTestSummary(10)
	changed.Entries[0].Samples, changed.Entries[0].Completed = 2, 2
	f.receive(t, changed, 500)
	f.receive(t, performanceTestSummary(9), 600)
	peer = requirePerformancePeer(t, f.b.performance)
	if peer.Summary.Revision != 10 || peer.Summary.Entries[0].Samples != 1 || peer.Summary.Entries[0].NewestAgeMs != 400 {
		t.Fatalf("forged/stale duplicate replaced summary: %+v", peer.Summary)
	}
	changed.Revision = 11
	f.receive(t, changed, 0)
	peer = requirePerformancePeer(t, f.b.performance)
	if peer.Summary.Revision != 11 || peer.Summary.Entries[0].Samples != 2 {
		t.Fatalf("higher revision not a replacement: %+v", peer.Summary)
	}
}

func TestPerformanceOffsetsAndRollbackNeverMakeSamplesYounger(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	f.receive(t, performanceTestSummary(10), 1000)
	f.now = f.now.Add(10 * time.Second)
	f.status(t, "fresh", 50, 0)
	peer := requirePerformancePeer(t, f.b.performance)
	if peer.Summary.Entries[0].NewestAgeMs != 11_100 || peer.LastSeenAgeMs != 0 {
		t.Fatalf("heartbeat ages = %+v", peer)
	}
	f.now = f.now.Add(-time.Hour)
	f.status(t, "fresh", 1, 0)
	peer = requirePerformancePeer(t, f.b.performance)
	if peer.Summary.Entries[0].NewestAgeMs != 11_100 {
		t.Fatal("rollback lowered sample age")
	}
	f.status(t, "stale", 50_000, 90_000)
	peer = requirePerformancePeer(t, f.b.performance)
	if peer.Status != "stale" || peer.Summary.Entries[0].NewestAgeMs != 50_100 {
		t.Fatalf("status offset = %+v", peer)
	}
	var source performancewire.Summary
	if err := json.Unmarshal(f.b.performance.peers["peer"].raw, &source); err != nil {
		t.Fatal(err)
	}
	if source.Entries[0].NewestAgeMs != 100 {
		t.Fatal("projection rewrote immutable source age")
	}
}

func TestPerformanceFrequentReadsPreserveFractionalElapsedAcrossHeartbeat(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	f.receive(t, performanceTestSummary(10), 0)
	initial := f.now
	for _, elapsed := range []time.Duration{400 * time.Microsecond, 800 * time.Microsecond, 1200 * time.Microsecond} {
		f.now = initial.Add(elapsed)
		_ = f.b.performance.projection()
	}
	peer := requirePerformancePeer(t, f.b.performance)
	if peer.Summary.Entries[0].NewestAgeMs != 101 || peer.LastSeenAgeMs != 1 {
		t.Fatalf("fractional reads lost elapsed time: age=%d seen=%d", peer.Summary.Entries[0].NewestAgeMs, peer.LastSeenAgeMs)
	}
	f.now = f.now.Add(400 * time.Microsecond)
	f.status(t, "fresh", 0, 0)
	f.now = f.now.Add(400 * time.Microsecond)
	peer = requirePerformancePeer(t, f.b.performance)
	if peer.Summary.Entries[0].NewestAgeMs != 102 || peer.LastSeenAgeMs != 0 {
		t.Fatalf("heartbeat reset sample fraction or reachability anchor: age=%d seen=%d", peer.Summary.Entries[0].NewestAgeMs, peer.LastSeenAgeMs)
	}
}

func TestPerformanceReadRetiresRevokedPeerAndRejectsDelayedEvents(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	f.receive(t, performanceTestSummary(10), 0)
	clustertrusttest.RemovePeerPin(t, f.dir, "peer")
	peer := requirePerformancePeer(t, f.b.performance)
	if peer.Status != "unknown" || peer.Summary != nil {
		t.Fatalf("revoked read = %+v", peer)
	}
	f.receive(t, performanceTestSummary(11), 0)
	if f.b.performance.peers["peer"].revision != 10 {
		t.Fatal("delayed event resurrected revoked authority")
	}
}

func TestPerformanceUnknownRetiresEqualRevisionPermanently(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	f.receive(t, performanceTestSummary(10), 0)
	f.status(t, "unknown", 100, 100)
	f.status(t, "fresh", 200, 0)
	f.receive(t, performanceTestSummary(10), 500)
	peer := requirePerformancePeer(t, f.b.performance)
	if peer.Status != "unknown" || peer.Summary != nil {
		t.Fatalf("retired equal revision revived: %+v", peer)
	}
	f.receive(t, performanceTestSummary(11), 500)
	peer = requirePerformancePeer(t, f.b.performance)
	if peer.Status != "fresh" || peer.Summary == nil || peer.Summary.Revision != 11 {
		t.Fatalf("new revision did not recover: %+v", peer)
	}
}

func TestPerformanceCertificateReplacementKeepsSameEpochRevisionFloor(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	f.receive(t, performanceTestSummary(10), 0)
	clustertrusttest.WritePeerPin(t, f.dir, "peer")
	f.admitPeer(t, "peer", 7)
	f.b.performance.mesh.Refresh()
	identity, ok := f.b.performance.mesh.PeerServiceIncarnation("peer")
	if !ok || identity == f.identity {
		t.Fatal("replacement fixture did not change authority")
	}
	f.identity = identity
	f.receive(t, performanceTestSummary(10), 0)
	peer := requirePerformancePeer(t, f.b.performance)
	if peer.Summary != nil || peer.Status != "unknown" {
		t.Fatalf("certificate replacement accepted equal revision: %+v", peer)
	}
	f.receive(t, performanceTestSummary(11), 0)
	if peer = requirePerformancePeer(t, f.b.performance); peer.Summary == nil || peer.Summary.Revision != 11 {
		t.Fatal("replacement rejected higher revision")
	}
}

func TestPerformanceNewSenderEpochResetsRevisionFloor(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	f.receive(t, performanceTestSummary(10), 0)
	f.admitPeer(t, "peer", 8)
	var ok bool
	f.identity, ok = f.b.performance.mesh.PeerServiceIncarnation("peer")
	if !ok {
		t.Fatal("new peer epoch missing")
	}
	next := performanceTestSummary(1)
	next.AdmissionIncarnation = "8"
	f.receive(t, next, 0)
	peer := requirePerformancePeer(t, f.b.performance)
	if peer.Summary == nil || peer.Summary.Revision != 1 || peer.Summary.AdmissionIncarnation != "8" {
		t.Fatalf("new sender epoch = %+v", peer)
	}
}

func TestPerformanceFreshnessAndDisconnectedExpiryKeepTombstone(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	f.receive(t, performanceTestSummary(10), 0)
	f.now = f.now.Add(90 * time.Second)
	if peer := requirePerformancePeer(t, f.b.performance); peer.Status != "stale" || peer.Summary == nil {
		t.Fatalf("stale projection = %+v", peer)
	}
	f.status(t, "fresh", 90_000, 0)
	f.now = f.now.Add(time.Hour - time.Millisecond)
	if peer := requirePerformancePeer(t, f.b.performance); peer.Summary == nil {
		t.Fatal("expired before one hour since heartbeat")
	}
	f.now = f.now.Add(time.Millisecond)
	peer := requirePerformancePeer(t, f.b.performance)
	if peer.Status != "unknown" || peer.Summary != nil || f.b.performance.projection().ExpiredPeers != 1 {
		t.Fatalf("expiry = %+v", peer)
	}
	f.receive(t, performanceTestSummary(10), 0)
	if requirePerformancePeer(t, f.b.performance).Summary != nil {
		t.Fatal("expired equal revision resurrected")
	}
}

func TestPerformanceInvalidPrivateFramesCannotMutateCache(t *testing.T) {
	test := func(name string, change func(map[string]json.RawMessage)) {
		t.Run(name, func(t *testing.T) {
			f := newBrokerPerformanceFixture(t)
			summary := performanceJSON(t, performanceTestSummary(10))
			fields := map[string]json.RawMessage{"summary": summary, "admissionIdentity": performanceJSON(t, f.identity), "sampleElapsedMs": performanceJSON(t, 0)}
			change(fields)
			f.b.consumePerformanceNotification(methodPerformanceReceived, performanceJSON(t, fields))
			if len(f.b.performance.peers) != 0 {
				t.Fatal("invalid private frame mutated cache")
			}
		})
	}
	test("unknown field", func(m map[string]json.RawMessage) { m["extra"] = performanceJSON(t, true) })
	test("missing elapsed", func(m map[string]json.RawMessage) { delete(m, "sampleElapsedMs") })
	test("null elapsed", func(m map[string]json.RawMessage) { m["sampleElapsedMs"] = []byte("null") })
	test("fractional elapsed", func(m map[string]json.RawMessage) { m["sampleElapsedMs"] = []byte("1.5") })
	test("negative elapsed", func(m map[string]json.RawMessage) { m["sampleElapsedMs"] = []byte("-1") })
	test("overflow elapsed", func(m map[string]json.RawMessage) { m["sampleElapsedMs"] = []byte("9223372036854775808") })
	test("expired elapsed", func(m map[string]json.RawMessage) {
		m["sampleElapsedMs"] = performanceJSON(t, performancewire.RetentionMs+1)
	})
	test("wrong identity", func(m map[string]json.RawMessage) { m["admissionIdentity"] = performanceJSON(t, "forged") })
	test("oversize summary", func(m map[string]json.RawMessage) {
		m["summary"] = performanceJSON(t, strings.Repeat("x", performancewire.MaxPayloadBytes+1))
	})
	test("noncanonical epoch", func(m map[string]json.RawMessage) {
		s := performanceTestSummary(10)
		s.AdmissionIncarnation = "07"
		m["summary"] = performanceJSON(t, s)
	})
	test("host alias", func(m map[string]json.RawMessage) {
		s := performanceTestSummary(10)
		s.MeasuringHost = "peer-name"
		s.Entries[0].Key.Host = "peer-name"
		m["summary"] = performanceJSON(t, s)
	})
}

func TestPerformancePeerCapacityPreservesExistingRevisionFloors(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	p := f.b.performance
	for i := 0; i < 128; i++ {
		p.peers[fmt.Sprintf("retired-%d", i)] = &performancePeer{retired: true, status: "unknown", revision: 20}
	}
	f.receive(t, performanceTestSummary(10), 0)
	if len(p.peers) != 128 || p.peers["peer"] != nil || p.peers["retired-0"].revision != 20 {
		t.Fatal("capacity evicted a revision floor or admitted an extra peer")
	}
}

func TestPerformanceInvalidStatusesCannotRefreshOrRetire(t *testing.T) {
	test := func(name string, change func(map[string]json.RawMessage)) {
		t.Run(name, func(t *testing.T) {
			f := newBrokerPerformanceFixture(t)
			f.receive(t, performanceTestSummary(10), 300)
			fields := map[string]json.RawMessage{}
			in := performanceStatus{MeasuringHost: "peer", AdmissionIncarnation: "7", AdmissionIdentity: f.identity, Status: "unknown", Revision: 10, SampleElapsedMs: 500, LastSeenAgeMs: 0}
			if err := json.Unmarshal(performanceJSON(t, in), &fields); err != nil {
				t.Fatal(err)
			}
			change(fields)
			f.b.consumePerformanceNotification(methodPerformanceStatus, performanceJSON(t, fields))
			state := f.b.performance.peers["peer"]
			if state.retired || len(state.raw) == 0 || state.status != "fresh" || state.elapsedMs != 300 {
				t.Fatal("invalid status mutated live cache")
			}
		})
	}
	test("unknown field", func(m map[string]json.RawMessage) { m["extra"] = []byte("true") })
	test("missing status", func(m map[string]json.RawMessage) { delete(m, "status") })
	test("null revision", func(m map[string]json.RawMessage) { m["revision"] = []byte("null") })
	test("stale revision", func(m map[string]json.RawMessage) { m["revision"] = []byte("9") })
	test("negative age", func(m map[string]json.RawMessage) { m["lastSeenAgeMs"] = []byte("-1") })
	test("fractional age", func(m map[string]json.RawMessage) { m["sampleElapsedMs"] = []byte("0.5") })
	test("unsupported status", func(m map[string]json.RawMessage) { m["status"] = performanceJSON(t, "deleted") })
	test("forged identity", func(m map[string]json.RawMessage) { m["admissionIdentity"] = performanceJSON(t, "forged") })
}

func TestPerformanceNoLocalAdmissionCannotAdmitPeerCache(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	if err := os.Remove(filepath.Join(f.dir, "admission.json")); err != nil {
		t.Fatal(err)
	}
	f.receive(t, performanceTestSummary(10), 0)
	if len(f.b.performance.peers) != 0 {
		t.Fatal("peer cache admitted a summary without local admission")
	}
}

func TestPerformanceAgeOverflowSaturates(t *testing.T) {
	state := &performancePeer{clock: time.Unix(1, 0), elapsedMs: math.MaxInt64 - 2, lastSeenAgeMs: math.MaxInt64 - 1}
	state.advance(time.Unix(2, 0))
	if state.elapsedMs != math.MaxInt64 || state.lastSeenAgeMs != math.MaxInt64 || performanceAdd(100, math.MaxInt64) != math.MaxInt64 {
		t.Fatalf("overflow = %+v", state)
	}
}

func TestPerformanceStatusBeyondRetentionStillRetiresPayload(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	f.receive(t, performanceTestSummary(10), 0)
	f.status(t, "stale", performancewire.RetentionMs+10, performancewire.RetentionMs+20)
	state := f.b.performance.peers["peer"]
	if state.status != "stale" || state.elapsedMs != performancewire.RetentionMs+10 {
		t.Fatalf("long stale status rejected: %+v", state)
	}
	f.status(t, "unknown", math.MaxInt64, math.MaxInt64)
	if !state.retired || state.raw != nil || state.status != "unknown" {
		t.Fatalf("long unknown status rejected: %+v", state)
	}
}

func TestPerformanceSaveLoopCoalescesSamplesAndFinalFlushes(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	p := f.b.performance
	ctx, cancel := context.WithCancel(context.Background())
	ticks := make(chan time.Time)
	done := make(chan struct{})
	go func() { defer close(done); p.runSaves(ctx, ticks) }()
	select {
	case <-p.changeWake:
	case <-time.After(time.Second):
		cancel()
		<-done
		t.Fatal("initial checkpoint did not complete")
	}
	before := p.store.Stats().PublishedRevision
	for seq := uint64(1); seq <= 3; seq++ {
		if err := f.b.ingestPerformanceSample(performanceJSON(t, performanceTestSample(seq))); err != nil {
			cancel()
			<-done
			t.Fatal(err)
		}
	}
	if stats := p.store.Stats(); stats.PublishedRevision != before || !stats.Dirty {
		cancel()
		<-done
		t.Fatalf("samples bypassed coalescing timer: %+v", stats)
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("final flush did not finish")
	}
	restarted := performancestore.New("self", f.path)
	if err := restarted.Load(); err != nil {
		t.Fatal(err)
	}
	if stats := restarted.Stats(); stats.Samples != 3 || stats.PublishedRevision <= before {
		t.Fatalf("shutdown lost coalesced samples: %+v", stats)
	}
}

func TestPerformanceCorruptHistoryQuarantinesAndRecovers(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	original := []byte("{not-json contains private material")
	if err := os.WriteFile(f.path, original, 0o600); err != nil {
		t.Fatal(err)
	}
	p := newPerformanceRuntime("self", f.path, f.dir)
	if p.blocked || !p.store.Stats().HistoryReset {
		t.Fatal("successful quarantine did not permit empty recovery")
	}
	matches, err := filepath.Glob(f.path + ".corrupt-*")
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 1 {
		t.Fatalf("quarantine files=%v", matches)
	}
	raw, err := os.ReadFile(matches[0])
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, original) {
		t.Fatal("quarantine changed original bytes")
	}
	f.b.performance = p
	if err := f.b.ingestPerformanceSample(performanceJSON(t, performanceTestSample(1))); err != nil {
		t.Fatal(err)
	}
	p.flush()
	if stats := p.store.Stats(); stats.PublishedRevision <= 0 || stats.Samples != 1 || p.blocked {
		t.Fatalf("recovered history=%+v", stats)
	}
	restarted := newPerformanceRuntime("self", f.path, f.dir)
	if restarted.blocked || restarted.store.Stats().Samples != 1 {
		t.Fatal("recovered history could not restart")
	}
}

func TestPerformanceFailedQuarantineKeepsLocalViewAndOriginal(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	path := filepath.Join(t.TempDir(), "history.json")
	// History must be a regular file; quarantine must never move a directory.
	if err := os.MkdirAll(path, 0o700); err != nil {
		t.Fatal(err)
	}
	original := []byte("keep this file")
	marker := filepath.Join(path, "retained")
	if err := os.WriteFile(marker, original, 0o600); err != nil {
		t.Fatal(err)
	}
	p := newPerformanceRuntime("self", path, f.dir)
	f.b.performance = p
	if !p.blocked || p.store.Stats().HistoryReset {
		t.Fatal("failed quarantine enabled persistence")
	}
	if err := f.b.ingestPerformanceSample(performanceJSON(t, performanceTestSample(1))); err != nil {
		t.Fatal(err)
	}
	p.flush()
	if projection := p.projection(); !projection.Persistence.PublicationBlocked || projection.Local.Summary.Entries[0].Samples != 1 {
		t.Fatalf("failed quarantine projection=%+v", projection)
	}
	raw, err := os.ReadFile(marker)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, original) {
		t.Fatal("failed quarantine overwrote original")
	}
}

type performanceOutput struct {
	mu   sync.Mutex
	buf  bytes.Buffer
	wake chan struct{}
}

func (w *performanceOutput) Write(raw []byte) (int, error) {
	w.mu.Lock()
	n, err := w.buf.Write(raw)
	w.mu.Unlock()
	performanceSignal(w.wake)
	return n, err
}

func (w *performanceOutput) frames(t *testing.T) []Message {
	t.Helper()
	w.mu.Lock()
	raw := bytes.Clone(w.buf.Bytes())
	w.mu.Unlock()
	var frames []Message
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	for scanner.Scan() {
		var frame Message
		if err := json.Unmarshal(scanner.Bytes(), &frame); err != nil {
			t.Fatalf("decode output: %v", err)
		}
		frames = append(frames, frame)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("read output: %v", err)
	}
	return frames
}

func TestPerformanceReadRPCAndSubscriptionBaselineDoNotLeakWorkerFrames(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	out := &performanceOutput{wake: make(chan struct{}, 1)}
	f.b.codec = NewCodec(readWriter{Reader: strings.NewReader(""), Writer: out})
	f.receive(t, performanceTestSummary(10), 100)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); f.b.performanceChangeLoop(ctx) }()
	defer func() { cancel(); <-done }()
	readID, subscribeID, unsubscribeID := json.RawMessage("1"), json.RawMessage("2"), json.RawMessage("3")
	if !f.b.handlePerformanceRequest(&Message{JSONRPC: "2.0", Method: methodPerformanceGet, ID: &readID, Params: json.RawMessage("{}")}) {
		t.Fatal("read RPC not intercepted")
	}
	frames := out.frames(t)
	if len(frames) != 1 || len(frames[0].Result) == 0 {
		t.Fatalf("unsubscribed read frames = %+v", frames)
	}
	var projection performanceProjection
	if err := json.Unmarshal(frames[0].Result, &projection); err != nil {
		t.Fatal(err)
	}
	if projection.Local.Status != "missing" || len(projection.Peers) != 1 || projection.Peers[0].Summary.Entries[0].NewestAgeMs != 200 {
		t.Fatalf("read projection = %+v", projection)
	}
	f.b.handlePerformanceRequest(&Message{JSONRPC: "2.0", Method: methodPerformanceSubscribe, ID: &subscribeID})
	deadline := time.NewTimer(time.Second)
	defer deadline.Stop()
	for {
		frames = out.frames(t)
		if len(frames) >= 3 {
			break
		}
		select {
		case <-out.wake:
		case <-deadline.C:
			t.Fatal("missing subscription baseline")
		}
	}
	if frames[2].Method != notifyPerformanceChanged {
		t.Fatalf("baseline = %+v", frames[2])
	}
	for _, frame := range frames {
		if frame.Method == methodPerformanceReceived || frame.Method == methodPerformanceStatus {
			t.Fatal("private worker frame leaked")
		}
	}
	f.b.handlePerformanceRequest(&Message{JSONRPC: "2.0", Method: methodPerformanceUnsubscribe, ID: &unsubscribeID})
	f.b.performance.mu.Lock()
	subscribed := f.b.performance.subscribed
	f.b.performance.mu.Unlock()
	if subscribed {
		t.Fatal("unsubscribe left stream enabled")
	}
}

type performanceBlockedStdin struct {
	entered     chan struct{}
	release     chan struct{}
	once        sync.Once
	releaseOnce sync.Once
	written     chan []byte
	failWrites  int
}

func (w *performanceBlockedStdin) Write(raw []byte) (int, error) {
	w.once.Do(func() { close(w.entered) })
	<-w.release
	if w.written != nil {
		w.written <- bytes.Clone(raw)
	}
	if w.failWrites > 0 {
		w.failWrites--
		return 0, io.ErrClosedPipe
	}
	return len(raw), nil
}
func (w *performanceBlockedStdin) Close() error { return nil }
func (w *performanceBlockedStdin) unblock()     { w.releaseOnce.Do(func() { close(w.release) }) }

func TestPerformanceSuccessfulPublicationDoesNotRepeatOnRetryTicks(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	w := &performanceBlockedStdin{entered: make(chan struct{}), release: make(chan struct{}), written: make(chan []byte, 8)}
	w.unblock()
	f.b.replayPerformanceToManager(&workloadManagerProcess{stdin: w})
	f.b.performance.flush()
	ctx, cancel := context.WithCancel(context.Background())
	ticks := make(chan time.Time)
	done := make(chan struct{})
	go func() { defer close(done); f.b.performance.runPublications(ctx, ticks) }()
	select {
	case <-w.written:
	case <-time.After(time.Second):
		cancel()
		<-done
		t.Fatal("initial publication missing")
	}
	for i := 0; i < 3; i++ {
		select {
		case ticks <- time.Now():
		case <-time.After(time.Second):
			cancel()
			<-done
			t.Fatal("retry tick not consumed")
		}
	}
	cancel()
	<-done
	if len(w.written) != 0 {
		t.Fatal("successful unchanged revision was republished on retry ticks")
	}
}

func TestPerformanceFailedPublicationRetriesIdenticalFrozenBytes(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	w := &performanceBlockedStdin{entered: make(chan struct{}), release: make(chan struct{}), written: make(chan []byte, 8), failWrites: 1}
	w.unblock()
	f.b.replayPerformanceToManager(&workloadManagerProcess{stdin: w})
	f.b.performance.flush()
	ctx, cancel := context.WithCancel(context.Background())
	ticks := make(chan time.Time)
	done := make(chan struct{})
	go func() { defer close(done); f.b.performance.runPublications(ctx, ticks) }()
	defer func() { cancel(); <-done }()
	var first []byte
	select {
	case first = <-w.written:
	case <-time.After(time.Second):
		t.Fatal("initial failed publication missing")
	}
	select {
	case ticks <- time.Now():
	case <-time.After(time.Second):
		t.Fatal("retry tick not consumed")
	}
	select {
	case second := <-w.written:
		if !bytes.Equal(first, second) {
			t.Fatal("failed publication retried a changed body at the same revision")
		}
	case <-time.After(time.Second):
		t.Fatal("failed publication was not retried")
	}
}

func TestPerformanceExpiryContinuesDuringBlockedClientWrite(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	w := &performanceBlockedStdin{entered: make(chan struct{}), release: make(chan struct{})}
	f.b.codec = NewCodec(readWriter{Reader: strings.NewReader(""), Writer: w})
	f.b.performance.subscribed = true
	f.receive(t, performanceTestSummary(10), 0)
	ctx, cancel := context.WithCancel(context.Background())
	clientDone, sweepDone := make(chan struct{}), make(chan struct{})
	ticks := make(chan time.Time)
	go func() { defer close(clientDone); f.b.performanceChangeLoop(ctx) }()
	go func() { defer close(sweepDone); f.b.performance.runSweeps(ctx, ticks) }()
	defer func() { cancel(); w.unblock(); <-clientDone; <-sweepDone }()
	select {
	case <-w.entered:
	case <-time.After(time.Second):
		t.Fatal("client writer did not block")
	}
	f.now = f.now.Add(time.Hour)
	select {
	case ticks <- f.now:
	case <-time.After(time.Second):
		t.Fatal("independent sweep did not receive tick")
	}
	select {
	case <-f.b.performance.changeWake:
	case <-time.After(time.Second):
		t.Fatal("independent sweep did not complete")
	}
	f.b.performance.mu.Lock()
	retired := f.b.performance.peers["peer"].retired
	payload := len(f.b.performance.peers["peer"].raw)
	expired := f.b.performance.expiredPeers
	f.b.performance.mu.Unlock()
	if !retired || payload != 0 || expired != 1 {
		t.Fatalf("blocked client prevented expiry: retired=%t payload=%d expired=%d", retired, payload, expired)
	}
	if err := f.b.ingestPerformanceSample(performanceJSON(t, performanceTestSample(1))); err != nil {
		t.Fatal(err)
	}
}

func TestPerformanceBlockedPublicationDoesNotBlockSampleIntakeOrCheckpoint(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	w := &performanceBlockedStdin{entered: make(chan struct{}), release: make(chan struct{}), written: make(chan []byte, 8)}
	f.b.replayPerformanceToManager(&workloadManagerProcess{stdin: w})
	f.b.performance.flush()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); f.b.performance.publishLoop(ctx) }()
	defer func() { cancel(); w.unblock(); <-done }()
	select {
	case <-w.entered:
	case <-time.After(time.Second):
		t.Fatal("publisher did not enter fake blocked stdin")
	}
	before := f.b.performance.store.Stats().PublishedRevision
	accepted := make(chan error, 1)
	sample := performanceJSON(t, performanceTestSample(1))
	go func() { accepted <- f.b.ingestPerformanceSample(sample) }()
	select {
	case err := <-accepted:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("blocked publisher blocked sample intake")
	}
	f.b.performance.flush()
	if stats := f.b.performance.store.Stats(); stats.PublishedRevision <= before || stats.Samples != 1 {
		t.Fatalf("blocked publisher blocked persistence: %+v", stats)
	}
	if len(f.b.performance.publishWake) != 1 {
		t.Fatal("latest pending publication did not coalesce to one wake")
	}
	if err := f.b.ingestPerformanceSample(performanceJSON(t, performanceTestSample(2))); err != nil {
		t.Fatal(err)
	}
	f.b.performance.flush()
	latest := f.b.performance.store.Stats().PublishedRevision
	w.unblock()
	for i := 0; i < 2; i++ {
		select {
		case raw := <-w.written:
			var frame Message
			if err := json.Unmarshal(raw, &frame); err != nil {
				t.Fatal(err)
			}
			summary, err := performancewire.DecodeSummary(frame.Params)
			if err != nil {
				t.Fatal(err)
			}
			if i == 0 && summary.Revision != before {
				t.Fatalf("in-flight revision=%d, want %d", summary.Revision, before)
			}
			if i == 1 && (summary.Revision != latest || summary.Entries[0].Samples != 2) {
				t.Fatalf("pending publication did not select latest durable snapshot: %+v", summary)
			}
		case <-time.After(time.Second):
			t.Fatal("pending latest publication was not sent after unblock")
		}
	}
}

func TestPerformanceHistoryFailureKeepsUnsavedLocalView(t *testing.T) {
	f := newBrokerPerformanceFixture(t)
	parent := filepath.Join(t.TempDir(), "file-instead-of-directory")
	p := newPerformanceRuntime("self", filepath.Join(parent, "history.json"), f.dir)
	if err := os.WriteFile(parent, []byte("occupied"), 0o600); err != nil {
		t.Fatal(err)
	}
	f.b.performance = p
	if err := f.b.ingestPerformanceSample(performanceJSON(t, performanceTestSample(1))); err != nil {
		t.Fatal(err)
	}
	p.flush()
	projection := p.projection()
	if !projection.Persistence.Degraded || projection.Local.Summary.Entries[0].Samples != 1 || p.readyRevision != 0 {
		t.Fatalf("failed save projection=%+v", projection)
	}
	if err := os.Remove(parent); err != nil {
		t.Fatal(err)
	}
	p.flush()
	if p.projection().Persistence.Degraded || p.readyRevision == 0 {
		t.Fatal("checkpoint did not recover after save path was repaired")
	}
}

var _ io.WriteCloser = (*performanceBlockedStdin)(nil)
