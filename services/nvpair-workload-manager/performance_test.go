// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/performancewire"
)

// All TestPerformance fixtures use certificate files, recorders or net.Pipe.
// No helper opens a TCP/UDP listener, starts a PAIR binary or runs an engine.
// mu guards now, emitted and failures for tests whose streams run in other
// goroutines; sequential tests may set now directly.
type performanceFixture struct {
	selfDir    string
	peerDir    string
	manager    *Manager
	mu         sync.Mutex
	now        time.Time
	emitted    []Message
	failMethod string
	failures   int
}

func newPerformanceFixture(t *testing.T) *performanceFixture {
	t.Helper()
	selfDir, peerDir := newPinnedPeerDirs(t)
	f := &performanceFixture{selfDir: selfDir, peerDir: peerDir, now: time.Now()}
	f.manager = NewManager(NewCodec(codecNop{}), 0, "uuid-self", selfDir)
	f.manager.performance.now = func() time.Time {
		f.mu.Lock()
		defer f.mu.Unlock()
		return f.now
	}
	f.manager.performance.emit = func(method string, params any) error {
		f.mu.Lock()
		defer f.mu.Unlock()
		if method == f.failMethod && f.failures > 0 {
			f.failures--
			return errors.New("broker write failed")
		}
		raw, err := json.Marshal(params)
		if err != nil {
			return err
		}
		f.emitted = append(f.emitted, Message{Method: method, Params: raw})
		return nil
	}
	t.Cleanup(f.manager.broadcaster.clients.CloseIdle)
	return f
}

func performanceSummary(host string, revision int64) performancewire.Summary {
	return performancewire.Summary{Version: 2, MeasuringHost: host, AdmissionIncarnation: "1", Revision: revision,
		Window: performancewire.Window{MaxAgeMs: performancewire.RetentionMs, MaxSamplesPerKey: 64}, Entries: []performancewire.Entry{}}
}

func performanceEncode(t *testing.T, summary performancewire.Summary) []byte {
	t.Helper()
	raw, err := performancewire.EncodeSummary(summary)
	if err != nil {
		t.Fatalf("encode summary: %v", err)
	}
	return raw
}

func (f *performanceFixture) postRaw(raw []byte) int {
	return f.postRawWithElapsed(raw, 0)
}

// postRawWithElapsed applies one frame from uuid-peer as the stream subscriber
// does, with the authority of a stream opened just now: uuid-peer's admission
// identity at this moment, which must still hold for every recheck.
func (f *performanceFixture) postRawWithElapsed(raw []byte, elapsed int64) int {
	code, _ := f.manager.performance.applySummary(f.authority(), raw, elapsed)
	return code
}

func (f *performanceFixture) authority() func(string) (string, string, bool) {
	mesh := f.manager.mesh
	mesh.Refresh()
	identity, _ := mesh.PeerServiceIncarnation("uuid-peer")
	return f.manager.performance.streamAuthority("uuid-peer", identity, func() bool {
		mesh.Refresh()
		now, ok := mesh.PeerServiceIncarnation("uuid-peer")
		return ok && identity != "" && now == identity
	})
}

func (f *performanceFixture) post(t *testing.T, summary performancewire.Summary) int {
	t.Helper()
	return f.postRaw(performanceEncode(t, summary))
}

// advance moves the receiver's clock while streams may be reading it.
func (f *performanceFixture) advance(d time.Duration) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.now = f.now.Add(d)
}

func (f *performanceFixture) messages() []Message {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]Message(nil), f.emitted...)
}

func (f *performanceFixture) statuses(t *testing.T) []performanceStatus {
	t.Helper()
	var statuses []performanceStatus
	for _, msg := range f.messages() {
		if msg.Method == methodPerformanceStatus {
			var status performanceStatus
			if err := json.Unmarshal(msg.Params, &status); err != nil {
				t.Fatalf("decode status: %v", err)
			}
			statuses = append(statuses, status)
		}
	}
	return statuses
}

func (f *performanceFixture) summaries(t *testing.T) []performanceReceived {
	t.Helper()
	var summaries []performanceReceived
	for _, msg := range f.messages() {
		if msg.Method == methodPerformanceReceived {
			var received performanceReceived
			if err := json.Unmarshal(msg.Params, &received); err != nil {
				t.Fatalf("decode received: %v", err)
			}
			summaries = append(summaries, received)
		}
	}
	return summaries
}

func performanceWrite(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write authority: %v", err)
	}
}

func (f *performanceFixture) peerEpoch(t *testing.T, epoch uint64) {
	t.Helper()
	path := filepath.Join(f.selfDir, "trusted", "uuid-peer.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read pin: %v", err)
	}
	var pin workloadTestPin
	if err := json.Unmarshal(data, &pin); err != nil {
		t.Fatalf("decode pin: %v", err)
	}
	pin.AdmissionEpoch = epoch
	data, err = json.Marshal(pin)
	if err != nil {
		t.Fatalf("encode pin: %v", err)
	}
	performanceWrite(t, path, data)
	performanceWrite(t, filepath.Join(f.selfDir, "members.json"), []byte(`[{"nodeUuid":"uuid-peer","state":"member","clusterId":"test-cluster","admissionEpoch":`+strconv.FormatUint(epoch, 10)+`}]`))
}

func TestPerformanceRejectsForgedAndForwardedMeasuringHosts(t *testing.T) {
	for _, host := range []string{"uuid-self", "uuid-third-party"} {
		t.Run(host, func(t *testing.T) {
			f := newPerformanceFixture(t)
			if code := f.post(t, performanceSummary(host, 1)); code != http.StatusForbidden {
				t.Fatalf("status = %d, want 403", code)
			}
			if len(f.emitted) != 0 {
				t.Fatalf("broker notifications = %d, want none", len(f.emitted))
			}
		})
	}
}

func TestPerformanceRejectsUnpairedAndRevokedPeers(t *testing.T) {
	t.Run("unpinned peer", func(t *testing.T) {
		f := newPerformanceFixture(t)
		if err := os.Remove(filepath.Join(f.selfDir, "trusted", "uuid-peer.json")); err != nil {
			t.Fatalf("remove peer pin: %v", err)
		}
		if code := f.post(t, performanceSummary("uuid-peer", 1)); code != http.StatusForbidden {
			t.Fatalf("status = %d, want 403", code)
		}
		if len(f.emitted) != 0 {
			t.Fatal("unpinned peer reached broker")
		}
	})
	t.Run("revoked roster", func(t *testing.T) {
		f := newPerformanceFixture(t)
		if code := f.post(t, performanceSummary("uuid-peer", 1)); code != http.StatusOK {
			t.Fatalf("initial status = %d", code)
		}
		performanceWrite(t, filepath.Join(f.selfDir, "members.json"), []byte(`[]`))
		if code := f.post(t, performanceSummary("uuid-peer", 2)); code != http.StatusForbidden {
			t.Fatalf("revoked status = %d, want 403", code)
		}
		f.manager.performance.sweep()
		statuses := f.statuses(t)
		if len(statuses) != 2 || statuses[1].Status != "unknown" {
			t.Fatalf("statuses = %+v, want fresh then unknown", statuses)
		}
		state := f.manager.performance.received["uuid-peer"]
		if state == nil || !state.retired || state.payload != nil {
			t.Fatal("revoked payload retained")
		}
		if len(f.summaries(t)) != 1 {
			t.Fatal("revoked summary reached broker")
		}
	})
}

func TestPerformanceCanonicalAdmissionEpochs(t *testing.T) {
	test := func(epoch string, want int) {
		t.Run(epoch, func(t *testing.T) {
			f := newPerformanceFixture(t)
			summary := performanceSummary("uuid-peer", 1)
			summary.AdmissionIncarnation = epoch
			raw, err := json.Marshal(summary)
			if err != nil {
				t.Fatalf("encode untrusted summary: %v", err)
			}
			if code := f.postRaw(raw); code != want {
				t.Fatalf("status = %d, want %d", code, want)
			}
			if len(f.emitted) != 0 {
				t.Fatal("unverified epoch reached broker")
			}
		})
	}
	test("0", http.StatusForbidden)
	test("01", http.StatusForbidden)
	test("+1", http.StatusBadRequest)
	test("-1", http.StatusForbidden)
	test(" 1", http.StatusBadRequest)
	test("1 ", http.StatusBadRequest)
	test("1.0", http.StatusForbidden)
	test("18446744073709551616", http.StatusForbidden)
	test("2", http.StatusForbidden)
}

func TestPerformanceElapsedOffsetAdvancesWithoutDuplicateRejuvenation(t *testing.T) {
	f := newPerformanceFixture(t)
	summary := performanceSummary("uuid-peer", 1)
	summary.Entries = []performancewire.Entry{{Key: performancewire.ProfileKey{Host: "uuid-peer", Engine: "llamacpp", Model: "model", Fingerprint: "profile"}, Samples: 1, Completed: 1, NewestAgeMs: 12_000, OldestAgeMs: 12_000, NewestObservedAtMs: 1700000000000}}
	summary.Entries[0].Applicability = performancewire.ApplicabilityUnknown
	raw := performanceEncode(t, summary)
	post := func(elapsed int64, wantElapsed int64) {
		t.Helper()
		if code := f.postRawWithElapsed(raw, elapsed); code != http.StatusOK {
			t.Fatalf("status = %d, want %d", code, http.StatusOK)
		}
		statuses := f.statuses(t)
		if len(statuses) == 0 {
			t.Fatal("accepted summary did not emit a status")
		}
		status := statuses[len(statuses)-1]
		if status.SampleElapsedMs != wantElapsed || status.LastSeenAgeMs != 0 {
			t.Fatalf("status = %+v, want elapsed %d and current reachability", status, wantElapsed)
		}
	}
	post(17_000, 17_000)
	f.now = f.now.Add(30 * time.Second)
	post(1, 47_000)
	post(70_000, 70_000)
	f.now = f.now.Add(5 * time.Second)
	post(0, 75_000)
	f.now = f.now.Add(90 * time.Second)
	f.manager.performance.sweep()
	statuses := f.statuses(t)
	if status := statuses[len(statuses)-1]; status.Status != "stale" || status.SampleElapsedMs != 165_000 || status.LastSeenAgeMs != 90_000 {
		t.Fatalf("stale status = %+v", status)
	}
	received := f.summaries(t)
	if len(received) != 1 || received[0].SampleElapsedMs != 17_000 || !bytes.Equal(performanceEncode(t, received[0].Summary), raw) {
		t.Fatalf("received = %+v, want one unchanged summary with initial elapsed", received)
	}
}

func TestPerformanceSameCertificateReadmissionRejectsOldEpoch(t *testing.T) {
	f := newPerformanceFixture(t)
	if code := f.post(t, performanceSummary("uuid-peer", 10)); code != http.StatusOK {
		t.Fatalf("initial status = %d", code)
	}
	f.peerEpoch(t, 2)
	if code := f.post(t, performanceSummary("uuid-peer", 11)); code != http.StatusForbidden {
		t.Fatalf("old admission status = %d, want 403", code)
	}
	summary := performanceSummary("uuid-peer", 1)
	summary.AdmissionIncarnation = "2"
	if code := f.post(t, summary); code != http.StatusOK {
		t.Fatalf("readmission status = %d, want 200", code)
	}
	got := f.summaries(t)
	if len(got) != 2 || got[1].Summary.Revision != 1 || got[0].AdmissionIdentity == got[1].AdmissionIdentity {
		t.Fatalf("accepted replacements = %+v", got)
	}
}

func TestPerformanceAdmissionChangeDuringBrokerEmissionRetiresReplacement(t *testing.T) {
	f := newPerformanceFixture(t)
	emit := f.manager.performance.emit
	f.manager.performance.emit = func(method string, params any) error {
		if err := emit(method, params); err != nil {
			return err
		}
		if method == methodPerformanceReceived {
			f.peerEpoch(t, 2)
		}
		return nil
	}
	if code := f.post(t, performanceSummary("uuid-peer", 1)); code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", code)
	}
	statuses := f.statuses(t)
	if len(statuses) != 1 || statuses[0].Status != "unknown" {
		t.Fatalf("statuses = %+v, want unknown only", statuses)
	}
	if len(f.summaries(t)) != 1 || statuses[0].AdmissionIdentity != f.summaries(t)[0].AdmissionIdentity {
		t.Fatal("retirement does not identify emitted replacement")
	}
}

func TestPerformanceStrictPayloadLimits(t *testing.T) {
	test := func(name string, raw func([]byte) []byte) {
		t.Run(name, func(t *testing.T) {
			f := newPerformanceFixture(t)
			body := raw(performanceEncode(t, performanceSummary("uuid-peer", 1)))
			if code := f.postRaw(body); code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", code)
			}
			if len(f.emitted) != 0 {
				t.Fatal("malformed summary reached broker")
			}
		})
	}
	test("unknown field", func(raw []byte) []byte {
		return append(append([]byte{}, raw[:len(raw)-1]...), []byte(`,"unexpected":1}`)...)
	})
	test("params over 128KiB", func(raw []byte) []byte {
		out := append(bytes.Clone(raw[:len(raw)-1]), bytes.Repeat([]byte(" "), (128<<10)+1-len(raw))...)
		return append(out, '}')
	})
	test("trailing JSON", func(raw []byte) []byte { return append(raw, []byte(` {}`)...) })
	test("invalid summary", func(raw []byte) []byte { return bytes.Replace(raw, []byte(`"revision":1`), []byte(`"revision":0`), 1) })
}

func TestPerformanceRelayCarriesConfigurationWithdrawalAndRejectsPrivateMetadata(t *testing.T) {
	f := newPerformanceFixture(t)
	summary := performanceSummary("uuid-peer", 1)
	summary.Entries = []performancewire.Entry{{Key: performancewire.ProfileKey{Host: "uuid-peer", Engine: "llamacpp", Model: "model", Fingerprint: "fp"}, Profile: performancewire.Profile{EngineFamily: "llamacpp"}, Provenance: &performancewire.Provenance{Configuration: nodepolicy.MeasurementIdentity{Complete: true, Digest: strings.Repeat("a", 64)}, EngineFamily: "llamacpp", API: "/v1/chat/completions", ExtractorSchema: "daily-schema"}, Applicability: performancewire.ApplicabilityCurrent, Samples: 1, Completed: 1, NewestObservedAtMs: 1000}}
	if code := f.post(t, summary); code != http.StatusOK {
		t.Fatalf("current summary status = %d", code)
	}
	got := f.summaries(t)
	if len(got) != 1 || got[0].Summary.Entries[0].Applicability != performancewire.ApplicabilityCurrent || got[0].Summary.Entries[0].Provenance.Configuration.Digest != strings.Repeat("a", 64) {
		t.Fatalf("current summary lost metadata: %+v", got)
	}
	summary.Revision = 2
	summary.Entries[0].Applicability = performancewire.ApplicabilityUnknown
	if code := f.post(t, summary); code != http.StatusOK {
		t.Fatalf("withdrawal status = %d", code)
	}
	summary.Revision = 1
	summary.Entries[0].Applicability = performancewire.ApplicabilityCurrent
	if code := f.post(t, summary); code != http.StatusOK {
		t.Fatalf("old retry status = %d", code)
	}
	got = f.summaries(t)
	if len(got) != 2 || got[1].Summary.Revision != 2 || got[1].Summary.Entries[0].Applicability != performancewire.ApplicabilityUnknown {
		t.Fatalf("old retry revived current evidence: %+v", got)
	}
	raw := performanceEncode(t, summary)
	private := bytes.Replace(raw, []byte(`"extractorSchema":"daily-schema"`), []byte(`"extractorSchema":"daily-schema","options":{"secret":"private"}`), 1)
	if bytes.Equal(raw, private) {
		t.Fatal("missing privacy mutation target")
	}
	if code := f.postRaw(private); code != http.StatusBadRequest {
		t.Fatalf("private provenance status = %d", code)
	}
}

func TestPerformanceRejectsV1SummaryBeforeBrokerEmission(t *testing.T) {
	f := newPerformanceFixture(t)
	summary := performanceSummary("uuid-peer", 1)
	summary.Version = 1
	raw, err := json.Marshal(summary)
	if err != nil {
		t.Fatalf("encode v1 summary: %v", err)
	}
	if code := f.postRaw(raw); code != http.StatusBadRequest {
		t.Fatalf("v1 summary status = %d, want %d", code, http.StatusBadRequest)
	}
	if len(f.emitted) != 0 {
		t.Fatalf("v1 summary reached broker: %+v", f.emitted)
	}
}

func TestPerformanceV2ReportedRateReachesBroker(t *testing.T) {
	f := newPerformanceFixture(t)
	summary := performanceSummary("uuid-peer", 1)
	want := performancewire.ReportedRate{Samples: 1, DurationNs: 400_000_000, TokensPerSecond: 55}
	summary.Entries = []performancewire.Entry{{
		Key:     performancewire.ProfileKey{Host: "uuid-peer", Engine: "llamacpp-custom", Model: "model", Fingerprint: "profile"},
		Samples: 1, Completed: 1, NewestObservedAtMs: 1000, EngineDecode: &want,
	}}
	if code := f.post(t, summary); code != http.StatusOK {
		t.Fatalf("v2 summary status = %d, want %d", code, http.StatusOK)
	}
	received := f.summaries(t)
	if len(received) != 1 || received[0].Summary.Version != 2 || len(received[0].Summary.Entries) != 1 ||
		received[0].Summary.Entries[0].EngineDecode == nil || *received[0].Summary.Entries[0].EngineDecode != want || received[0].Summary.Entries[0].Decode != nil {
		t.Fatalf("broker replacements = %+v, want v2 reported decode %+v", received, want)
	}
}

func TestPerformanceDropsStaleRevisionsAndDeduplicatesHeartbeats(t *testing.T) {
	f := newPerformanceFixture(t)
	if code := f.post(t, performanceSummary("uuid-peer", 5)); code != http.StatusOK {
		t.Fatalf("initial status = %d", code)
	}
	f.now = f.now.Add(30 * time.Second)
	if code := f.post(t, performanceSummary("uuid-peer", 5)); code != http.StatusOK {
		t.Fatalf("duplicate status = %d", code)
	}
	if code := f.post(t, performanceSummary("uuid-peer", 4)); code != http.StatusOK {
		t.Fatalf("stale status = %d", code)
	}
	if len(f.summaries(t)) != 1 {
		t.Fatal("heartbeat or stale revision re-emitted aggregate samples")
	}
	statuses := f.statuses(t)
	if len(statuses) != 2 || statuses[1].SampleElapsedMs != 30_000 || statuses[1].LastSeenAgeMs != 0 {
		t.Fatalf("statuses = %+v", statuses)
	}
}

func TestPerformanceStalenessAt90SecondsAndHeartbeatRefresh(t *testing.T) {
	f := newPerformanceFixture(t)
	start := f.now
	if code := f.post(t, performanceSummary("uuid-peer", 1)); code != http.StatusOK {
		t.Fatalf("initial status = %d", code)
	}
	f.now = start.Add(89 * time.Second)
	f.manager.performance.sweep()
	if len(f.statuses(t)) != 1 {
		t.Fatal("summary marked stale before 90 seconds")
	}
	f.now = start.Add(90 * time.Second)
	f.manager.performance.sweep()
	statuses := f.statuses(t)
	if len(statuses) != 2 || statuses[1].Status != "stale" || statuses[1].LastSeenAgeMs != 90_000 {
		t.Fatalf("statuses = %+v", statuses)
	}
	if code := f.post(t, performanceSummary("uuid-peer", 1)); code != http.StatusOK {
		t.Fatalf("heartbeat status = %d", code)
	}
	f.now = start.Add(179 * time.Second)
	f.manager.performance.sweep()
	statuses = f.statuses(t)
	if len(statuses) != 3 || statuses[2].Status != "fresh" || statuses[2].SampleElapsedMs != 90_000 {
		t.Fatalf("heartbeat freshness = %+v", statuses)
	}
	f.now = start.Add(180 * time.Second)
	f.manager.performance.sweep()
	statuses = f.statuses(t)
	if len(statuses) != 4 || statuses[3].Status != "stale" || statuses[3].SampleElapsedMs != 180_000 {
		t.Fatalf("second staleness = %+v", statuses)
	}
	if len(f.summaries(t)) != 1 {
		t.Fatal("freshness changes multiplied samples")
	}
}

func TestPerformanceHeartbeatPreservesSampleAges(t *testing.T) {
	f := newPerformanceFixture(t)
	summary := performanceSummary("uuid-peer", 1)
	summary.Entries = []performancewire.Entry{{Key: performancewire.ProfileKey{Host: "uuid-peer", Engine: "llamacpp", Model: "model", Fingerprint: "profile"}, Samples: 1, Completed: 1, NewestAgeMs: 12_000, OldestAgeMs: 12_000, NewestObservedAtMs: 1700000000000}}
	if code := f.post(t, summary); code != http.StatusOK {
		t.Fatalf("initial status = %d", code)
	}
	f.now = f.now.Add(30 * time.Second)
	if code := f.post(t, summary); code != http.StatusOK {
		t.Fatalf("heartbeat status = %d", code)
	}
	got := f.summaries(t)
	statuses := f.statuses(t)
	if len(got) != 1 || got[0].Summary.Entries[0].NewestAgeMs != 12_000 || got[0].Summary.Entries[0].NewestObservedAtMs != 1700000000000 || statuses[1].SampleElapsedMs != 30_000 {
		t.Fatalf("samples = %+v, statuses = %+v", got, statuses)
	}
}

func TestPerformanceBrokerEmitFailureLeavesRevisionRetryable(t *testing.T) {
	for _, method := range []string{methodPerformanceReceived, methodPerformanceStatus} {
		t.Run(method, func(t *testing.T) {
			f := newPerformanceFixture(t)
			f.failMethod, f.failures = method, 1
			summary := performanceSummary("uuid-peer", 1)
			if code := f.post(t, summary); code != http.StatusInternalServerError {
				t.Fatalf("first status = %d, want 500", code)
			}
			if code := f.post(t, summary); code != http.StatusOK {
				t.Fatalf("retry status = %d, want 200", code)
			}
			if len(f.summaries(t)) != 1 || len(f.statuses(t)) != 1 {
				t.Fatalf("successful emissions = %d summaries, %d statuses", len(f.summaries(t)), len(f.statuses(t)))
			}
		})
	}
}

func TestPerformanceEmptySummaryReplacesPreviousEntries(t *testing.T) {
	f := newPerformanceFixture(t)
	summary := performanceSummary("uuid-peer", 1)
	summary.Entries = []performancewire.Entry{{Key: performancewire.ProfileKey{Host: "uuid-peer", Engine: "llamacpp", Model: "model", Fingerprint: "profile"}, Samples: 1, Completed: 1, NewestObservedAtMs: 1700000000000}}
	if code := f.post(t, summary); code != http.StatusOK {
		t.Fatalf("initial status = %d", code)
	}
	if code := f.post(t, performanceSummary("uuid-peer", 2)); code != http.StatusOK {
		t.Fatalf("replacement status = %d", code)
	}
	got := f.summaries(t)
	if len(got) != 2 || len(got[0].Summary.Entries) != 1 || got[1].Summary.Entries == nil || len(got[1].Summary.Entries) != 0 || got[1].Summary.Revision != 2 {
		t.Fatalf("replacements = %+v", got)
	}
	if len(f.manager.broadcastCh) != 0 || len(f.manager.performance.wake) != 0 {
		t.Fatal("received replacement was gossiped")
	}
}

func performanceAddPeer(m *Manager) {
	m.peers.Replace([]PeerNode{{ID: "uuid-peer", Addresses: []string{"192.0.2.10"}, Port: 14320, TXT: []string{"cluster-uuid=uuid-peer"}}})
}

func performanceWaitDone(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("performance worker did not stop")
	}
}

func TestPerformanceRevisionFloorSurvivesReceiverAuthorityChanges(t *testing.T) {
	test := func(name string, change func(*testing.T, *performanceFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newPerformanceFixture(t)
			if code := f.post(t, performanceSummary("uuid-peer", 5)); code != http.StatusOK {
				t.Fatalf("initial status = %d", code)
			}
			change(t, f)
			f.manager.performance.sweep()
			if code := f.post(t, performanceSummary("uuid-peer", 4)); code != http.StatusOK {
				t.Fatalf("old revision status = %d", code)
			}
			if len(f.summaries(t)) != 1 {
				t.Fatal("receiver authority change reset sender revision floor")
			}
			if code := f.post(t, performanceSummary("uuid-peer", 5)); code != http.StatusOK {
				t.Fatalf("same revision status = %d", code)
			}
			if len(f.summaries(t)) != 1 {
				t.Fatal("same revision heartbeat re-emitted samples after receiver authority change")
			}
			if code := f.post(t, performanceSummary("uuid-peer", 6)); code != http.StatusOK {
				t.Fatalf("new revision status = %d", code)
			}
			if len(f.summaries(t)) != 2 {
				t.Fatal("higher revision did not emit after receiver authority change")
			}
		})
	}
	test("local readmission", func(t *testing.T, f *performanceFixture) {
		performanceWrite(t, filepath.Join(f.selfDir, "admission.json"), []byte(`{"clusterId":"test-cluster","epoch":2,"counter":2,"activated":2}`))
	})
	test("local certificate replacement", func(t *testing.T, f *performanceFixture) {
		cert, key := genLeaf(t, "uuid-self")
		performanceWrite(t, filepath.Join(f.selfDir, "node.crt"), cert)
		performanceWrite(t, filepath.Join(f.selfDir, "node.key"), key)
	})
}

func TestPerformancePublishRequiresCurrentLocalMeasuringIdentity(t *testing.T) {
	for _, host := range []string{"uuid-peer", "legacy-host-alias"} {
		t.Run(host, func(t *testing.T) {
			f := newPerformanceFixture(t)
			if err := f.manager.performance.publish(performanceEncode(t, performanceSummary(host, 1))); err == nil {
				t.Fatal("forged local summary accepted")
			}
			if !performanceWithheld(f.manager.performance) || len(f.manager.performance.wake) != 0 {
				t.Fatal("rejected summary became pending")
			}
		})
	}
	t.Run("canonical epoch", func(t *testing.T) {
		f := newPerformanceFixture(t)
		summary := performanceSummary("uuid-self", 1)
		summary.AdmissionIncarnation = "01"
		if err := f.manager.performance.publish(performanceEncode(t, summary)); err == nil {
			t.Fatal("noncanonical epoch accepted")
		}
	})
	t.Run("pending summary retired on local readmission", func(t *testing.T) {
		f := newPerformanceFixture(t)
		if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 10))); err != nil {
			t.Fatalf("publish: %v", err)
		}
		performanceWrite(t, filepath.Join(f.selfDir, "admission.json"), []byte(`{"clusterId":"test-cluster","epoch":2,"counter":2,"activated":2}`))
		if !performanceWithheld(f.manager.performance) {
			t.Fatal("old local admission kept pending bytes")
		}
		summary := performanceSummary("uuid-self", 1)
		summary.AdmissionIncarnation = "2"
		if err := f.manager.performance.publish(performanceEncode(t, summary)); err != nil {
			t.Fatalf("new local admission publish: %v", err)
		}
	})
}

func TestPerformanceRejectsSameRevisionContentChanges(t *testing.T) {
	f := newPerformanceFixture(t)
	summary := performanceSummary("uuid-peer", 1)
	if code := f.post(t, summary); code != http.StatusOK {
		t.Fatalf("initial status = %d", code)
	}
	summary.Window.MaxSamplesPerKey = 63
	if code := f.post(t, summary); code != http.StatusBadRequest {
		t.Fatalf("changed revision status = %d, want 400", code)
	}
	if len(f.summaries(t)) != 1 || len(f.statuses(t)) != 1 {
		t.Fatal("changed same-revision content was accepted")
	}
}

func TestPerformancePeerStateCapacityIsBounded(t *testing.T) {
	f := newPerformanceFixture(t)
	for i := 0; i < 128; i++ {
		if f.manager.performance.receipt("host-"+strconv.Itoa(i)) == nil {
			t.Fatalf("capacity rejected peer %d", i)
		}
	}
	if f.manager.performance.receipt("overflow") != nil || len(f.manager.performance.received) != 128 {
		t.Fatal("receiver state exceeded 128 peers")
	}
	if code := f.post(t, performanceSummary("uuid-peer", 1)); code != http.StatusServiceUnavailable {
		t.Fatalf("full receiver status = %d, want 503", code)
	}
	if len(f.emitted) != 0 {
		t.Fatal("capacity rejection emitted a summary")
	}
}

func TestPerformanceRetirementStatusRetriesAfterBrokerFailure(t *testing.T) {
	f := newPerformanceFixture(t)
	f.failMethod, f.failures = methodPerformanceStatus, 1
	emit := f.manager.performance.emit
	f.manager.performance.emit = func(method string, params any) error {
		if err := emit(method, params); err != nil {
			return err
		}
		if method == methodPerformanceReceived {
			f.peerEpoch(t, 2)
		}
		return nil
	}
	if code := f.post(t, performanceSummary("uuid-peer", 1)); code != http.StatusForbidden {
		t.Fatalf("changed admission status = %d, want 403", code)
	}
	if len(f.statuses(t)) != 0 {
		t.Fatal("failed retirement write recorded as successful")
	}
	f.manager.performance.sweep()
	statuses := f.statuses(t)
	if len(statuses) != 1 || statuses[0].Status != "unknown" || statuses[0].Revision != 1 {
		t.Fatalf("retirement retry statuses = %+v", statuses)
	}
	state := f.manager.performance.received["uuid-peer"]
	if state == nil || !state.retired || state.payload != nil {
		t.Fatal("retry did not retire old payload")
	}
}

func TestPerformanceSameAuthorityRecoveryCanRetireAgain(t *testing.T) {
	f := newPerformanceFixture(t)
	path := filepath.Join(f.selfDir, "members.json")
	members, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read members: %v", err)
	}
	for revision := int64(1); revision <= 2; revision++ {
		performanceWrite(t, path, members)
		if code := f.post(t, performanceSummary("uuid-peer", revision)); code != http.StatusOK {
			t.Fatalf("revision %d status = %d", revision, code)
		}
		performanceWrite(t, path, []byte(`[]`))
		f.manager.performance.sweep()
		statuses := f.statuses(t)
		if len(statuses) != int(revision)*2 || statuses[len(statuses)-1].Status != "unknown" {
			t.Fatalf("revision %d statuses = %+v; want fresh then unknown each time", revision, statuses)
		}
		if state := f.manager.performance.received["uuid-peer"]; state == nil || state.payload != nil {
			t.Fatal("authority loss retained a payload")
		}
	}
}

func TestPerformanceRevisionFloorSurvivesReceiverPinLoss(t *testing.T) {
	f := newPerformanceFixture(t)
	if code := f.post(t, performanceSummary("uuid-peer", 5)); code != http.StatusOK {
		t.Fatalf("initial status = %d", code)
	}
	pinPath := filepath.Join(f.selfDir, "trusted", "uuid-peer.json")
	pin, err := os.ReadFile(pinPath)
	if err != nil {
		t.Fatalf("read peer pin: %v", err)
	}
	if err := os.Remove(pinPath); err != nil {
		t.Fatalf("remove peer pin: %v", err)
	}
	performanceWrite(t, filepath.Join(f.selfDir, "admission.json"), []byte(`{"clusterId":"test-cluster","epoch":2,"counter":2,"activated":2}`))
	f.manager.performance.sweep()
	performanceWrite(t, pinPath, pin)
	if code := f.post(t, performanceSummary("uuid-peer", 4)); code != http.StatusOK {
		t.Fatalf("lower revision status = %d", code)
	}
	if len(f.summaries(t)) != 1 {
		t.Fatal("receiver pin restoration reset sender revision floor")
	}
	if code := f.post(t, performanceSummary("uuid-peer", 6)); code != http.StatusOK {
		t.Fatalf("higher revision status = %d", code)
	}
	if len(f.summaries(t)) != 2 {
		t.Fatal("higher revision did not recover after receiver pin restoration")
	}
}

func TestPerformancePublicationAnchorSurvivesLocalAuthorityFailure(t *testing.T) {
	f := newPerformanceFixture(t)
	raw := performanceEncode(t, performanceSummary("uuid-self", 5))
	if err := f.manager.performance.publish(raw); err != nil {
		t.Fatalf("publish: %v", err)
	}
	path := filepath.Join(f.selfDir, "admission.json")
	admission, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read local admission: %v", err)
	}
	f.now = f.now.Add(time.Hour)
	performanceWrite(t, path, []byte(`{`))
	if !performanceWithheld(f.manager.performance) {
		t.Fatal("invalid local authority exposed an outbound summary")
	}
	performanceWrite(t, path, admission)
	if err := f.manager.performance.publish(raw); err != nil {
		t.Fatalf("duplicate publish after authority recovery: %v", err)
	}
	got, elapsed := f.manager.performance.snapshotForSend()
	if !bytes.Equal(got, raw) || elapsed != time.Hour.Milliseconds() {
		t.Fatalf("recovered elapsed = %d, want original one-hour publication anchor", elapsed)
	}
	if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 4))); err == nil {
		t.Fatal("authority recovery reset local revision floor")
	}
}

// performanceWithheld reports that the relay has no summary it would send.
func performanceWithheld(p *performanceRelay) bool {
	raw, _ := p.snapshotForSend()
	return raw == nil
}
