// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
	"nvpair-shared/performancewire"
)

// All TestPerformance fixtures use certificate files, recorders or net.Pipe.
// No helper opens a TCP/UDP listener, starts a PAIR binary or runs an engine.
type performanceFixture struct {
	selfDir    string
	peerDir    string
	cert       *x509.Certificate
	manager    *Manager
	now        time.Time
	emitted    []Message
	failMethod string
	failures   int
}

func newPerformanceFixture(t *testing.T) *performanceFixture {
	t.Helper()
	selfDir, peerDir := newPinnedPeerDirs(t)
	f := &performanceFixture{selfDir: selfDir, peerDir: peerDir, now: time.Now()}
	data, err := os.ReadFile(filepath.Join(peerDir, "node.crt"))
	if err != nil {
		t.Fatalf("read peer certificate: %v", err)
	}
	block, _ := pem.Decode(data)
	if block == nil {
		t.Fatal("decode peer certificate")
	}
	f.cert, err = x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Fatalf("parse peer certificate: %v", err)
	}
	f.manager = NewManager(NewCodec(codecNop{}), 0, "uuid-self", selfDir)
	f.manager.performance.now = func() time.Time { return f.now }
	f.manager.performance.emit = func(method string, params any) error {
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
	t.Cleanup(func() {
		f.manager.broadcaster.clients.CloseIdle()
		f.manager.performance.broadcaster.clients.CloseIdle()
	})
	return f
}

func performanceSummary(host string, revision int64) performancewire.Summary {
	return performancewire.Summary{Version: 1, MeasuringHost: host, AdmissionIncarnation: "1", Revision: revision,
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
	return f.postRawWithElapsed(raw, []string{"0"})
}

func (f *performanceFixture) postRawWithElapsed(raw []byte, elapsed []string) int {
	body := append([]byte(`{"jsonrpc":"2.0","method":"performance:summary","params":`), raw...)
	body = append(body, '}')
	req := httptest.NewRequest(http.MethodPost, eventsPath, bytes.NewReader(body))
	for _, value := range elapsed {
		req.Header.Add(performanceElapsedHeader, value)
	}
	req.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{f.cert}}
	w := httptest.NewRecorder()
	f.manager.server.handleEvents(w, req)
	return w.Code
}

func (f *performanceFixture) post(t *testing.T, summary performancewire.Summary) int {
	t.Helper()
	return f.postRaw(performanceEncode(t, summary))
}

func (f *performanceFixture) statuses(t *testing.T) []performanceStatus {
	t.Helper()
	var statuses []performanceStatus
	for _, msg := range f.emitted {
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
	for _, msg := range f.emitted {
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
	t.Run("unpaired certificate", func(t *testing.T) {
		f := newPerformanceFixture(t)
		certPEM, _ := genLeaf(t, "uuid-peer")
		block, _ := pem.Decode(certPEM)
		if block == nil {
			t.Fatal("decode unpaired certificate")
		}
		cert, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			t.Fatalf("parse unpaired certificate: %v", err)
		}
		f.cert = cert
		if code := f.post(t, performanceSummary("uuid-peer", 1)); code != http.StatusForbidden {
			t.Fatalf("status = %d, want 403", code)
		}
		if len(f.emitted) != 0 {
			t.Fatal("unpaired certificate reached broker")
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

func TestPerformanceRejectsMalformedElapsedHeaders(t *testing.T) {
	test := func(name string, values ...string) {
		t.Run(name, func(t *testing.T) {
			f := newPerformanceFixture(t)
			if code := f.postRawWithElapsed(performanceEncode(t, performanceSummary("uuid-peer", 1)), values); code != http.StatusBadRequest {
				t.Fatalf("status = %d, want %d", code, http.StatusBadRequest)
			}
			if len(f.emitted) != 0 || len(f.manager.performance.received) != 0 {
				t.Fatal("invalid elapsed header changed receiver state or reached broker")
			}
		})
	}
	test("missing")
	test("empty", "")
	test("negative", "-1")
	test("positive sign", "+1")
	test("leading zero", "01")
	test("multiple zeroes", "00")
	test("leading space", " 1")
	test("trailing space", "1 ")
	test("fraction", "1.0")
	test("exponent", "1e3")
	test("comma joined", "0, 1")
	test("uint64 overflow", "18446744073709551616")
	test("past retention", strconv.FormatInt(performancewire.RetentionMs+1, 10))
	test("duplicate identical", "0", "0")
	test("duplicate different", "0", "1")
}

func TestPerformanceAcceptsElapsedHeaderBoundaries(t *testing.T) {
	test := func(name string, elapsed int64) {
		t.Run(name, func(t *testing.T) {
			f := newPerformanceFixture(t)
			summary := performanceSummary("uuid-peer", 1)
			if code := f.postRawWithElapsed(performanceEncode(t, summary), []string{strconv.FormatInt(elapsed, 10)}); code != http.StatusOK {
				t.Fatalf("status = %d, want %d", code, http.StatusOK)
			}
			received, statuses := f.summaries(t), f.statuses(t)
			if len(received) != 1 || received[0].SampleElapsedMs != elapsed {
				t.Fatalf("received = %+v, want initial elapsed %d", received, elapsed)
			}
			if len(statuses) != 1 || statuses[0].SampleElapsedMs != elapsed || statuses[0].LastSeenAgeMs != 0 {
				t.Fatalf("statuses = %+v, want initial elapsed %d and current reachability", statuses, elapsed)
			}
		})
	}
	test("zero", 0)
	test("retention", performancewire.RetentionMs)
}

func TestPerformanceElapsedOffsetAdvancesWithoutDuplicateRejuvenation(t *testing.T) {
	f := newPerformanceFixture(t)
	summary := performanceSummary("uuid-peer", 1)
	summary.Entries = []performancewire.Entry{{Key: performancewire.ProfileKey{Host: "uuid-peer", Engine: "llamacpp", Model: "model", Fingerprint: "profile"}, Samples: 1, Completed: 1, NewestAgeMs: 12_000, OldestAgeMs: 12_000, NewestObservedAtMs: 1700000000000}}
	raw := performanceEncode(t, summary)
	post := func(elapsed string, wantElapsed int64) {
		t.Helper()
		if code := f.postRawWithElapsed(raw, []string{elapsed}); code != http.StatusOK {
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
	post("17000", 17_000)
	f.now = f.now.Add(30 * time.Second)
	post("1", 47_000)
	post("70000", 70_000)
	f.now = f.now.Add(5 * time.Second)
	post("0", 75_000)
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

func TestPerformanceStrictPayloadAndOuterFrameLimits(t *testing.T) {
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
	test("outer frame over 1MiB", func(raw []byte) []byte { return append(raw, bytes.Repeat([]byte(" "), (1<<20)+1)...) })
	test("trailing JSON", func(raw []byte) []byte { return append(raw, []byte(` {}`)...) })
	test("invalid summary", func(raw []byte) []byte { return bytes.Replace(raw, []byte(`"revision":1`), []byte(`"revision":0`), 1) })
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

// performancePipeClient uses the real pinned TLS service client on in-memory
// net.Pipe connections. The HTTP peer is a manual request/response loop, not a
// server or listener. Both direct and reverse seams prohibit real network I/O.
func performancePipeClient(t *testing.T, b *Broadcaster, peer *clustertrust.Mesh, handler http.HandlerFunc) func() {
	t.Helper()
	b.clients.CloseIdle()
	var mu sync.Mutex
	var connections []net.Conn
	var wg sync.WaitGroup
	var closed bool
	closeConnections := func() {
		mu.Lock()
		defer mu.Unlock()
		closed = true
		for _, conn := range connections {
			_ = conn.Close()
		}
	}
	b.clients = clustertrust.NewPeerServiceClient(b.mesh, clustertrust.PeerServiceClientOptions{
		PeerClientOptions: clustertrust.PeerClientOptions{Timeout: 2 * time.Second},
		ResolveRoute:      b.resolveRoute, ReuseConnections: true,
		DialContext: func(context.Context, string, string) (net.Conn, error) {
			client, server := net.Pipe()
			mu.Lock()
			if closed {
				mu.Unlock()
				_ = client.Close()
				_ = server.Close()
				return nil, net.ErrClosed
			}
			connections = append(connections, client, server)
			wg.Add(1)
			mu.Unlock()
			go func() {
				defer wg.Done()
				defer server.Close()
				conn := tls.Server(server, peer.ServerTLSConfig())
				if err := conn.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
					t.Errorf("pipe deadline: %v", err)
					return
				}
				if err := conn.Handshake(); err != nil {
					return
				}
				reader := bufio.NewReader(conn)
				for {
					req, err := http.ReadRequest(reader)
					if err != nil {
						return
					}
					state := conn.ConnectionState()
					req.TLS = &state
					if _, ok := peer.VerifyClientPin(req); !ok {
						t.Error("pipe client was not pinned")
						return
					}
					recorder := httptest.NewRecorder()
					handler(recorder, req)
					if err := req.Body.Close(); err != nil {
						t.Errorf("close request body: %v", err)
						return
					}
					response := recorder.Result()
					response.ContentLength = int64(recorder.Body.Len())
					err = response.Write(conn)
					closeErr := response.Body.Close()
					if closeErr != nil {
						t.Errorf("close response body: %v", closeErr)
						return
					}
					if err != nil {
						return
					}
				}
			}()
			return client, nil
		},
		DialReverse: func(context.Context, string, string, noderec.ServiceKey) (net.Conn, error) {
			return nil, errors.New("test reverse path disabled")
		},
	})
	t.Cleanup(func() {
		// net.Pipe is unbuffered: closing both raw ends also releases TLS
		// close-notify writes when a test cancels a blocked HTTP request.
		closeConnections()
		b.clients.CloseIdle()
		wg.Wait()
	})
	return closeConnections
}

func performanceReadFrame(t *testing.T, r *http.Request) Message {
	t.Helper()
	body, err := io.ReadAll(io.LimitReader(r.Body, (1<<20)+1))
	if err != nil {
		t.Errorf("read pipe request: %v", err)
		return Message{}
	}
	var msg Message
	if err := json.Unmarshal(body, &msg); err != nil {
		t.Errorf("decode pipe request: %v", err)
	}
	return msg
}

func performanceAddPeer(m *Manager) {
	m.peers.Replace([]PeerNode{{ID: "uuid-peer", Addresses: []string{"192.0.2.10"}, Port: 14320, TXT: []string{"cluster-uuid=uuid-peer"}}})
}

func performanceWaitFrame(t *testing.T, frames <-chan Message) Message {
	t.Helper()
	select {
	case msg := <-frames:
		return msg
	case <-time.After(2 * time.Second):
		t.Fatal("expected pipe frame did not arrive")
		return Message{}
	}
}

func performanceWaitDone(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("performance worker did not stop")
	}
}

func TestPerformanceCoalescesNewestWholeSnapshotOutsideLifecycleFIFO(t *testing.T) {
	f := newPerformanceFixture(t)
	performanceAddPeer(f.manager)
	var frames []Message
	performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) {
		msg := performanceReadFrame(t, r)
		if msg.Method == methodPerformanceCapabilities {
			_, _ = io.WriteString(w, `{"version":1}`)
			return
		}
		frames = append(frames, msg)
	})
	for revision := int64(1); revision <= 100; revision++ {
		f.manager.handleMessage(&Message{Method: methodPerformancePublish, Params: performanceEncode(t, performanceSummary("uuid-self", revision))})
	}
	if len(f.manager.performance.wake) != 1 || len(f.manager.broadcastCh) != 0 {
		t.Fatal("publishes were not bounded/coalesced separately")
	}
	f.manager.performance.sendRound(context.Background())
	if len(frames) != 1 || frames[0].Method != performancewire.MethodSummary {
		t.Fatalf("frames = %+v", frames)
	}
	got, err := performancewire.DecodeSummary(frames[0].Params)
	if err != nil {
		t.Fatalf("decode sent summary: %v", err)
	}
	if got.Revision != 100 || got.MeasuringHost != "uuid-self" || got.Entries == nil || len(got.Entries) != 0 {
		t.Fatalf("sent summary = %+v", got)
	}
}

func TestPerformanceDelayedFirstSendPreservesPublicationAnchor(t *testing.T) {
	f := newPerformanceFixture(t)
	performanceAddPeer(f.manager)
	type sentSummary struct {
		frame   Message
		elapsed []string
	}
	var sent []sentSummary
	performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) {
		msg := performanceReadFrame(t, r)
		if msg.Method == methodPerformanceCapabilities {
			_, _ = io.WriteString(w, `{"version":1}`)
			return
		}
		sent = append(sent, sentSummary{msg, r.Header.Values(performanceElapsedHeader)})
	})
	summary := performanceSummary("uuid-self", 1)
	summary.Entries = []performancewire.Entry{{Key: performancewire.ProfileKey{Host: "uuid-self", Engine: "llamacpp", Model: "model", Fingerprint: "profile"}, Samples: 1, Completed: 1, NewestAgeMs: 12_000, OldestAgeMs: 12_000, NewestObservedAtMs: 1700000000000}}
	raw := performanceEncode(t, summary)
	if err := f.manager.performance.publish(raw); err != nil {
		t.Fatalf("publish: %v", err)
	}
	f.now = f.now.Add(20 * time.Second)
	if err := f.manager.performance.publish(raw); err != nil {
		t.Fatalf("duplicate publish: %v", err)
	}
	f.now = f.now.Add(15 * time.Second)
	f.manager.performance.sendRound(context.Background())
	f.now = f.now.Add(30 * time.Second)
	f.manager.performance.sendRound(context.Background())
	if len(sent) != 2 {
		t.Fatalf("sent summaries = %d, want 2", len(sent))
	}
	for i, wantElapsed := range []string{"35000", "65000"} {
		if sent[i].frame.Method != performancewire.MethodSummary || !bytes.Equal(sent[i].frame.Params, raw) {
			t.Fatalf("sent frame %d = %+v, want unchanged summary", i, sent[i].frame)
		}
		if len(sent[i].elapsed) != 1 || sent[i].elapsed[0] != wantElapsed {
			t.Fatalf("sent elapsed header %d = %v, want [%s]", i, sent[i].elapsed, wantElapsed)
		}
	}
}

func TestPerformanceSenderWithholdsExpiredPublicationUntilNewRevision(t *testing.T) {
	f := newPerformanceFixture(t)
	performanceAddPeer(f.manager)
	var elapsedHeaders []string
	var revisions []int64
	performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) {
		msg := performanceReadFrame(t, r)
		if msg.Method == methodPerformanceCapabilities {
			_, _ = io.WriteString(w, `{"version":1}`)
			return
		}
		summary, err := performancewire.DecodeSummary(msg.Params)
		if err != nil {
			t.Errorf("decode sent summary: %v", err)
			return
		}
		revisions = append(revisions, summary.Revision)
		elapsedHeaders = append(elapsedHeaders, r.Header.Get(performanceElapsedHeader))
	})
	raw := performanceEncode(t, performanceSummary("uuid-self", 1))
	if err := f.manager.performance.publish(raw); err != nil {
		t.Fatalf("publish: %v", err)
	}
	f.now = f.now.Add(time.Duration(performancewire.RetentionMs) * time.Millisecond)
	f.manager.performance.sendRound(context.Background())
	if len(revisions) != 1 || revisions[0] != 1 || elapsedHeaders[0] != strconv.FormatInt(performancewire.RetentionMs, 10) {
		t.Fatalf("retention boundary revisions = %v, elapsed headers = %v", revisions, elapsedHeaders)
	}
	f.now = f.now.Add(time.Millisecond)
	f.manager.performance.sendRound(context.Background())
	if err := f.manager.performance.publish(raw); err != nil {
		t.Fatalf("expired duplicate publish: %v", err)
	}
	f.manager.performance.sendRound(context.Background())
	if len(revisions) != 1 {
		t.Fatalf("expired publication sent; revisions = %v", revisions)
	}
	if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 2))); err != nil {
		t.Fatalf("new revision publish: %v", err)
	}
	f.manager.performance.sendRound(context.Background())
	if len(revisions) != 2 || revisions[1] != 2 || elapsedHeaders[1] != "0" {
		t.Fatalf("new publication revisions = %v, elapsed headers = %v, want revision 2 elapsed 0", revisions, elapsedHeaders)
	}
}

func TestPerformanceHeartbeat30SecondsAndLateJoinBackfill(t *testing.T) {
	if performanceHeartbeat != 30*time.Second {
		t.Fatalf("heartbeat = %s, want 30s", performanceHeartbeat)
	}
	f := newPerformanceFixture(t)
	frames := make(chan Message, 4)
	performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) {
		msg := performanceReadFrame(t, r)
		if msg.Method == methodPerformanceCapabilities {
			_, _ = io.WriteString(w, `{"version":1}`)
			return
		}
		frames <- msg
	})
	if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 7))); err != nil {
		t.Fatalf("publish: %v", err)
	}
	// No peers yet: consume the pending local publication before discovery.
	<-f.manager.performance.wake
	f.manager.performance.sendRound(context.Background())
	if len(frames) != 0 {
		t.Fatal("publication sent without peers")
	}
	f.manager.peerSource = &performancePeerSource{}
	f.manager.refreshPeers(context.Background())
	if len(f.manager.performance.wake) != 1 {
		t.Fatal("late join did not wake performance sender")
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	ticks := make(chan time.Time)
	go func() { defer close(done); f.manager.performance.runTicks(ctx, ticks) }()
	defer func() { cancel(); performanceWaitDone(t, done) }()
	first := performanceWaitFrame(t, frames)
	ticks <- time.Now().Add(30 * time.Second)
	heartbeat := performanceWaitFrame(t, frames)
	if !bytes.Equal(first.Params, heartbeat.Params) {
		t.Fatal("heartbeat changed source ages or snapshot contents")
	}
	got, err := performancewire.DecodeSummary(heartbeat.Params)
	if err != nil {
		t.Fatalf("decode heartbeat: %v", err)
	}
	if got.Revision != 7 {
		t.Fatalf("heartbeat revision = %d, want 7", got.Revision)
	}
}

type performancePeerSource struct{}

func (*performancePeerSource) Nodes(context.Context) ([]PeerNode, error) {
	return []PeerNode{{ID: "uuid-peer", Addresses: []string{"192.0.2.10"}, Port: 14320, TXT: []string{"cluster-uuid=uuid-peer"}}}, nil
}

func TestPerformanceNewestPublishDuringCapabilityProbeWinsDispatch(t *testing.T) {
	f := newPerformanceFixture(t)
	performanceAddPeer(f.manager)
	frames := make(chan Message, 1)
	performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) {
		msg := performanceReadFrame(t, r)
		if msg.Method == methodPerformanceCapabilities {
			if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 2))); err != nil {
				t.Errorf("new publish: %v", err)
			}
			_, _ = io.WriteString(w, `{"version":1}`)
			return
		}
		frames <- msg
	})
	if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 1))); err != nil {
		t.Fatalf("publish: %v", err)
	}
	f.manager.performance.sendRound(context.Background())
	got, err := performancewire.DecodeSummary(performanceWaitFrame(t, frames).Params)
	if err != nil {
		t.Fatalf("decode dispatched summary: %v", err)
	}
	if got.Revision != 2 {
		t.Fatalf("dispatched revision = %d, want 2", got.Revision)
	}
}

func TestPerformanceUnsupportedPeerNegotiatedOncePerRoute(t *testing.T) {
	f := newPerformanceFixture(t)
	performanceAddPeer(f.manager)
	var probes, summaries atomic.Int32
	performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) {
		msg := performanceReadFrame(t, r)
		if msg.Method == methodPerformanceCapabilities {
			probes.Add(1)
			http.Error(w, "unknown method", http.StatusBadRequest)
			return
		}
		summaries.Add(1)
	})
	if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 1))); err != nil {
		t.Fatalf("publish: %v", err)
	}
	for range 5 {
		f.manager.performance.sendRound(context.Background())
	}
	if probes.Load() != 1 || summaries.Load() != 0 {
		t.Fatalf("probes=%d summaries=%d, want 1/0", probes.Load(), summaries.Load())
	}
	f.manager.peers.Replace([]PeerNode{{ID: "uuid-peer", Addresses: []string{"192.0.2.11"}, Port: 14320, TXT: []string{"cluster-uuid=uuid-peer"}}})
	f.manager.performance.sendRound(context.Background())
	if probes.Load() != 2 || summaries.Load() != 0 {
		t.Fatalf("after route change probes=%d summaries=%d, want 2/0", probes.Load(), summaries.Load())
	}
}

func TestPerformanceCapabilityFailuresHaveBoundedRetryBudget(t *testing.T) {
	f := newPerformanceFixture(t)
	performanceAddPeer(f.manager)
	var probes, summaries atomic.Int32
	performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) {
		msg := performanceReadFrame(t, r)
		if msg.Method == methodPerformanceCapabilities {
			probes.Add(1)
			_, _ = io.WriteString(w, `{"version":1,"unrecognized":true}`)
			return
		}
		summaries.Add(1)
	})
	if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 1))); err != nil {
		t.Fatalf("publish: %v", err)
	}
	for range 8 {
		f.manager.performance.sendRound(context.Background())
	}
	if probes.Load() != 3 || summaries.Load() != 0 {
		t.Fatalf("probes=%d summaries=%d, want 3/0", probes.Load(), summaries.Load())
	}
}

func TestPerformanceBlockedPeerDoesNotDelayLifecycleOrCancel(t *testing.T) {
	f := newPerformanceFixture(t)
	performanceAddPeer(f.manager)
	blocked := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	closePerformance := performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) {
		msg := performanceReadFrame(t, r)
		if msg.Method == methodPerformanceCapabilities {
			_, _ = io.WriteString(w, `{"version":1}`)
			return
		}
		close(blocked)
		<-release
	})
	lifecycle := make(chan Message, 4)
	closeLifecycle := performancePipeClient(t, f.manager.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) { lifecycle <- performanceReadFrame(t, r) })
	if f.manager.broadcaster.clients == f.manager.performance.broadcaster.clients {
		t.Fatal("performance shares lifecycle connection slots")
	}
	ctx, cancel := context.WithCancel(context.Background())
	f.manager.ctx = ctx
	defer cancel()
	done := make(chan struct{})
	if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 1))); err != nil {
		t.Fatalf("publish: %v", err)
	}
	go func() { defer close(done); f.manager.performance.sendRound(ctx) }()
	select {
	case <-blocked:
	case <-time.After(2 * time.Second):
		t.Fatal("performance request did not block")
	}
	fifoDone := make(chan struct{})
	go func() { defer close(fifoDone); f.manager.broadcastLoop(ctx) }()
	defer func() {
		cancel()
		once.Do(func() { close(release) })
		closePerformance()
		closeLifecycle()
		performanceWaitDone(t, done)
		performanceWaitDone(t, fifoDone)
	}()
	f.manager.broadcastFrame("workload:started", json.RawMessage(`{"seq":1}`))
	f.manager.broadcastFrame(MethodRemove, json.RawMessage(`{"seq":2}`))
	first, second := performanceWaitFrame(t, lifecycle), performanceWaitFrame(t, lifecycle)
	if first.Method != "workload:started" || second.Method != MethodRemove {
		t.Fatalf("FIFO methods = %s, %s", first.Method, second.Method)
	}
	f.manager.handleLocalCancel(&Message{Method: MethodCancel, Params: json.RawMessage(`{"originatedFrom":"uuid-peer","workloadId":"1"}`)})
	if msg := performanceWaitFrame(t, lifecycle); msg.Method != MethodCancel {
		t.Fatalf("cancel method = %s", msg.Method)
	}
	select {
	case <-done:
		t.Fatal("performance did not remain blocked while lifecycle/cancel finished")
	default:
	}
}

func TestPerformanceFreshnessSweepRunsWhileNetworkSenderIsBlocked(t *testing.T) {
	f := newPerformanceFixture(t)
	if code := f.post(t, performanceSummary("uuid-peer", 1)); code != http.StatusOK {
		t.Fatalf("receive status = %d", code)
	}
	performanceAddPeer(f.manager)
	blocked, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	closePerformance := performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) {
		msg := performanceReadFrame(t, r)
		if msg.Method == methodPerformanceCapabilities {
			_, _ = io.WriteString(w, `{"version":1}`)
			return
		}
		close(blocked)
		<-release
	})
	stale := make(chan struct{}, 1)
	emit := f.manager.performance.emit
	f.manager.performance.emit = func(method string, params any) error {
		if err := emit(method, params); err != nil {
			return err
		}
		if status, ok := params.(performanceStatus); ok && status.Status == "stale" {
			stale <- struct{}{}
		}
		return nil
	}
	if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 1))); err != nil {
		t.Fatalf("publish: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	ticks, done := make(chan time.Time), make(chan struct{})
	go func() { defer close(done); f.manager.performance.runTicks(ctx, ticks) }()
	defer func() { cancel(); once.Do(func() { close(release) }); closePerformance(); performanceWaitDone(t, done) }()
	select {
	case <-blocked:
	case <-time.After(2 * time.Second):
		t.Fatal("network sender did not block")
	}
	f.now = f.now.Add(90 * time.Second)
	ticks <- f.now
	select {
	case <-stale:
	case <-time.After(time.Second):
		t.Fatal("blocked network prevented 90-second freshness sweep")
	}
	if len(f.manager.performance.wake) != 1 {
		t.Fatal("heartbeat did not queue a coalesced wake while sender blocked")
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

func TestPerformanceAdmissionChangeDuringCapabilityProbePreventsSummaryDispatch(t *testing.T) {
	f := newPerformanceFixture(t)
	performanceAddPeer(f.manager)
	var summaries atomic.Int32
	performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) {
		msg := performanceReadFrame(t, r)
		if msg.Method == methodPerformanceCapabilities {
			f.peerEpoch(t, 2)
			_, _ = io.WriteString(w, `{"version":1}`)
			return
		}
		summaries.Add(1)
	})
	if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 1))); err != nil {
		t.Fatalf("publish: %v", err)
	}
	f.manager.performance.sendRound(context.Background())
	if summaries.Load() != 0 {
		t.Fatal("summary was dispatched using pre-readmission authority")
	}
}

func TestPerformancePublishRequiresCurrentLocalMeasuringIdentity(t *testing.T) {
	for _, host := range []string{"uuid-peer", "legacy-host-alias"} {
		t.Run(host, func(t *testing.T) {
			f := newPerformanceFixture(t)
			if err := f.manager.performance.publish(performanceEncode(t, performanceSummary(host, 1))); err == nil {
				t.Fatal("forged local summary accepted")
			}
			if f.manager.performance.snapshot() != nil || len(f.manager.performance.wake) != 0 {
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
		if f.manager.performance.snapshot() != nil {
			t.Fatal("old local admission kept pending bytes")
		}
		summary := performanceSummary("uuid-self", 1)
		summary.AdmissionIncarnation = "2"
		if err := f.manager.performance.publish(performanceEncode(t, summary)); err != nil {
			t.Fatalf("new local admission publish: %v", err)
		}
	})
}

func TestPerformanceLegacyRelayAliasCannotAuthorizeMeasuringIdentity(t *testing.T) {
	f := newPerformanceFixture(t)
	f.manager.peers.Replace([]PeerNode{{ID: "legacy-host-alias", Addresses: []string{"192.0.2.10"}, Port: 14320, TXT: []string{"cluster-uuid=uuid-peer"}}})
	var calls atomic.Int32
	performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) { calls.Add(1) })
	if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 1))); err != nil {
		t.Fatalf("publish: %v", err)
	}
	f.manager.performance.sendRound(context.Background())
	if calls.Load() != 0 {
		t.Fatal("relay host alias acquired performance identity authority")
	}
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

func TestPerformanceAuthenticatedCapabilityDispatch(t *testing.T) {
	f := newPerformanceFixture(t)
	request := func(params string, authenticated bool) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, eventsPath, strings.NewReader(`{"jsonrpc":"2.0","method":"performance:capabilities","params":`+params+`}`))
		if authenticated {
			req.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{f.cert}}
		}
		w := httptest.NewRecorder()
		f.manager.server.handleEvents(w, req)
		return w
	}
	if w := request(`{}`, false); w.Code != http.StatusForbidden {
		t.Fatalf("unauthenticated capability status = %d", w.Code)
	}
	if w := request(`{"extra":1}`, true); w.Code != http.StatusBadRequest {
		t.Fatalf("unknown capability field status = %d", w.Code)
	}
	w := request(`{}`, true)
	if w.Code != http.StatusOK {
		t.Fatalf("capability status = %d", w.Code)
	}
	var capability struct {
		Version int `json:"version"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &capability); err != nil {
		t.Fatalf("decode capability: %v", err)
	}
	if capability.Version != 1 || len(f.emitted) != 0 {
		t.Fatalf("capability = %+v, emitted=%d", capability, len(f.emitted))
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

func TestPerformanceNegotiationStateCoversAllCurrentAdmittedTargets(t *testing.T) {
	cert, key := genLeaf(t, "uuid-self")
	pins := make(map[string][]byte)
	nodes := make([]PeerNode, 0, 129)
	for i := 0; i < 129; i++ {
		uuid := "uuid-peer-" + strconv.Itoa(i)
		peerCert, _ := genLeaf(t, uuid)
		pins[uuid] = peerCert
		nodes = append(nodes, PeerNode{ID: uuid, Addresses: []string{"192.0.2.10"}, Port: 14320, TXT: []string{"cluster-uuid=" + uuid}})
	}
	dir := setupNode(t, cert, key, pins)
	m := NewManager(NewCodec(codecNop{}), 0, "uuid-self", dir)
	t.Cleanup(func() { m.broadcaster.clients.CloseIdle(); m.performance.broadcaster.clients.CloseIdle() })
	m.peers.Replace(nodes)
	for _, target := range m.peers.targets() {
		identity, ok := m.mesh.PeerServiceIncarnation(target.uuid)
		_, generation, routeOK := m.peers.targetUUID(target.uuid)
		if !ok || !routeOK {
			t.Fatalf("target %s is not admitted", target.uuid)
		}
		// Every target has already negotiated unsupported, so this round must
		// perform no dialing and preserve every unchanged route's budget.
		m.performance.support[target.uuid] = performanceSupport{identity: identity, generation: generation, unsupported: true}
	}
	if err := m.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 1))); err != nil {
		t.Fatalf("publish: %v", err)
	}
	m.performance.sendRound(context.Background())
	if len(m.performance.support) != len(nodes) {
		t.Fatalf("retained negotiation states = %d, want %d", len(m.performance.support), len(nodes))
	}
	for _, state := range m.performance.support {
		if !state.unsupported {
			t.Fatal("unchanged unsupported route lost its negotiation result")
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
	if f.manager.performance.snapshot() != nil {
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

func TestPerformanceNegotiationSurvivesTemporaryPeerAuthorityFailure(t *testing.T) {
	t.Run("unsupported result across failed authority lookup", func(t *testing.T) {
		f := newPerformanceFixture(t)
		performanceAddPeer(f.manager)
		var probes atomic.Int32
		performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) {
			_ = performanceReadFrame(t, r)
			probes.Add(1)
			http.Error(w, "unknown method", http.StatusBadRequest)
		})
		if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 1))); err != nil {
			t.Fatalf("publish: %v", err)
		}
		f.manager.performance.sendRound(context.Background())
		path := filepath.Join(f.selfDir, "members.json")
		members, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read members: %v", err)
		}
		performanceWrite(t, path, []byte(`{`))
		f.manager.performance.sendRound(context.Background())
		performanceWrite(t, path, members)
		f.manager.performance.sendRound(context.Background())
		if probes.Load() != 1 {
			t.Fatalf("unchanged route capability probes = %d, want 1", probes.Load())
		}
	})
	t.Run("probe budget across in-flight authority failure", func(t *testing.T) {
		f := newPerformanceFixture(t)
		performanceAddPeer(f.manager)
		path := filepath.Join(f.selfDir, "members.json")
		members, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read members: %v", err)
		}
		var probes atomic.Int32
		performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) {
			_ = performanceReadFrame(t, r)
			if probes.Add(1) == 1 {
				performanceWrite(t, path, []byte(`{`))
			}
			_, _ = io.WriteString(w, `{`)
		})
		if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 1))); err != nil {
			t.Fatalf("publish: %v", err)
		}
		f.manager.performance.sendRound(context.Background())
		performanceWrite(t, path, members)
		for range 5 {
			f.manager.performance.sendRound(context.Background())
		}
		if probes.Load() != performanceProbeAttempts {
			t.Fatalf("unchanged route capability probes = %d, want %d", probes.Load(), performanceProbeAttempts)
		}
	})
}

func TestPerformanceNegotiationSurvivesTemporaryRouteAmbiguity(t *testing.T) {
	f := newPerformanceFixture(t)
	performanceAddPeer(f.manager)
	var probes atomic.Int32
	performancePipeClient(t, f.manager.performance.broadcaster, clustertrust.Open(f.peerDir), func(w http.ResponseWriter, r *http.Request) {
		_ = performanceReadFrame(t, r)
		probes.Add(1)
		http.Error(w, "unknown method", http.StatusBadRequest)
	})
	if err := f.manager.performance.publish(performanceEncode(t, performanceSummary("uuid-self", 1))); err != nil {
		t.Fatalf("publish: %v", err)
	}
	f.manager.performance.sendRound(context.Background())
	canonical, generation, ok := f.manager.peers.targetUUID("uuid-peer")
	if !ok {
		t.Fatal("initial canonical route unavailable")
	}
	peer := PeerNode{ID: canonical.id, Addresses: []string{"192.0.2.10"}, Port: 14320, TXT: []string{"cluster-uuid=uuid-peer"}}
	alias := peer
	alias.ID = "legacy-alias"
	f.manager.peers.Replace([]PeerNode{peer, alias})
	f.manager.performance.sendRound(context.Background())
	if probes.Load() != 1 {
		t.Fatal("ambiguous route dispatched a probe")
	}
	f.manager.peers.Replace([]PeerNode{peer})
	_, recoveredGeneration, ok := f.manager.peers.targetUUID("uuid-peer")
	if !ok || recoveredGeneration != generation {
		t.Fatal("canonical route changed in ambiguity fixture")
	}
	f.manager.performance.sendRound(context.Background())
	if probes.Load() != 1 {
		t.Fatalf("unchanged canonical route capability probes = %d, want 1", probes.Load())
	}
}
