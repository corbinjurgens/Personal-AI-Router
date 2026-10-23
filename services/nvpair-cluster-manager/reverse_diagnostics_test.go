// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

// recordedLog keeps each record's message and attributes for assertions.
type recordedLog struct {
	mu      sync.Mutex
	records []map[string]string
}

func (r *recordedLog) Enabled(context.Context, slog.Level) bool { return true }
func (r *recordedLog) WithAttrs([]slog.Attr) slog.Handler       { return r }
func (r *recordedLog) WithGroup(string) slog.Handler            { return r }
func (r *recordedLog) Handle(_ context.Context, record slog.Record) error {
	fields := map[string]string{"msg": record.Message, "level": record.Level.String()}
	record.Attrs(func(a slog.Attr) bool {
		fields[a.Key] = a.Value.String()
		return true
	})
	r.mu.Lock()
	r.records = append(r.records, fields)
	r.mu.Unlock()
	return nil
}

func (r *recordedLog) find(msg string, want map[string]string) []map[string]string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var found []map[string]string
	for _, fields := range r.records {
		if fields["msg"] != msg {
			continue
		}
		match := true
		for key, value := range want {
			match = match && fields[key] == value
		}
		if match {
			found = append(found, fields)
		}
	}
	return found
}

func (r *recordedLog) await(t *testing.T, msg string, want map[string]string) map[string]string {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		if found := r.find(msg, want); len(found) > 0 {
			return found[0]
		}
		if time.Now().After(deadline) {
			r.mu.Lock()
			defer r.mu.Unlock()
			t.Fatalf("no %q record with %v; logged %v", msg, want, r.records)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// captureLog routes the default logger to a recording handler for the test.
func captureLog(t *testing.T) *recordedLog {
	t.Helper()
	records := &recordedLog{}
	previous := slog.Default()
	slog.SetDefault(slog.New(records))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return records
}

func TestReverseFailureLogLimitsRepeatsPerPeer(t *testing.T) {
	now := time.Unix(1000, 0)
	log := &reverseFailureLog{interval: 30 * time.Second, now: func() time.Time { return now }}
	note := func(key, text string) (bool, int) { return log.note(key, text) }
	expect := func(what string, wantEmit bool, wantRepeats int, key, text string) {
		t.Helper()
		if emit, repeats := note(key, text); emit != wantEmit || repeats != wantRepeats {
			t.Fatalf("%s: emit=%v repeats=%d, want emit=%v repeats=%d", what, emit, repeats, wantEmit, wantRepeats)
		}
	}
	expect("first failure", true, 0, "outbound p1", "refused")
	for i := 0; i < 5; i++ {
		now = now.Add(time.Second)
		expect("repeat inside the interval", false, 0, "outbound p1", "refused")
	}
	expect("another peer is independent", true, 0, "outbound p2", "refused")
	now = now.Add(24 * time.Second) // 29s after the first line
	expect("repeat just inside the interval", false, 0, "outbound p1", "refused")
	now = now.Add(2 * time.Second)
	expect("repeat after the interval reports the held-back count", true, 6, "outbound p1", "refused")
	now = now.Add(time.Second)
	expect("count restarts after the line", false, 0, "outbound p1", "refused")
	expect("a changed failure is logged at once", true, 0, "outbound p1", "TLS handshake")
	log.clear("outbound p1")
	expect("a failure after success is a first failure", true, 0, "outbound p1", "TLS handshake")
}

func TestReverseFailureLogBoundsTrackedKeys(t *testing.T) {
	log := &reverseFailureLog{}
	for i := 0; i < reverseFailureLogMaxKeys+10; i++ {
		if emit, _ := log.note("inbound 10.0.0."+strconv.Itoa(i), "not pinned"); !emit {
			t.Fatalf("first failure for key %d was not logged", i)
		}
	}
	log.mu.Lock()
	defer log.mu.Unlock()
	if len(log.runs) > reverseFailureLogMaxKeys {
		t.Fatalf("tracked %d keys, want at most %d", len(log.runs), reverseFailureLogMaxKeys)
	}
}

// closedLoopbackPort returns a loopback port nothing listens on.
func closedLoopbackPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	return port
}

func TestReverseDialFailureNamesAddressAndStageOnce(t *testing.T) {
	records := captureLog(t)
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	b.port = closedLoopbackPort(t)
	pairReverseManagers(t, a, b)
	node, ok := a.memberByNodeID(b.identity.NodeUUID)
	if !ok {
		t.Fatal("peer member missing")
	}
	address := net.JoinHostPort("127.0.0.1", strconv.Itoa(b.port))
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	err := a.dialReverseSession(ctx, *node)
	if err == nil || !strings.Contains(err.Error(), address+" dial: ") {
		t.Fatalf("dial error = %v, want the address and its dial stage", err)
	}
	if strings.Contains(err.Error(), "->") {
		t.Fatalf("dial error %q carries the connection's local endpoint, so repeats would differ", err)
	}

	// The connect loop retries about once a second; the same failure is logged once.
	for i := 0; i < 3; i++ {
		a.attemptReverseSession(ctx, *node)
	}
	failed := records.find("reverse session failed", map[string]string{"peer": b.identity.NodeUUID})
	if len(failed) != 1 {
		t.Fatalf("logged %d failures for three identical attempts, want 1: %v", len(failed), failed)
	}
	if failed[0]["direction"] != "outbound" || !strings.Contains(failed[0]["err"], address+" dial: ") || failed[0]["level"] != slog.LevelWarn.String() {
		t.Fatalf("failure record = %v", failed[0])
	}
}

func TestReverseDialFailureNamesTheRejectedUpgrade(t *testing.T) {
	records := captureLog(t)
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	startReverseHTTP(t, b, false)
	// a pins b, but b does not know a: b refuses a's client certificate.
	pinTrusted(t, a, b.identity.NodeUUID, string(b.identity.CertPEM), b.identity.CertFingerprint)
	a.upsertMember(&ClusterNode{NodeUUID: b.identity.NodeUUID, ID: b.identity.NodeID, IPAddress: "127.0.0.1", Port: b.port,
		ClusterID: "cluster-1", AdmissionEpoch: 1, State: stateMember})
	node, ok := a.memberByNodeID(b.identity.NodeUUID)
	if !ok {
		t.Fatal("peer member missing")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	err := a.dialReverseSession(ctx, *node)
	t.Logf("refused upgrade reported as: %v", err)
	address := net.JoinHostPort("127.0.0.1", strconv.Itoa(b.port))
	if err == nil || !strings.Contains(err.Error(), address) {
		t.Fatalf("dial error = %v, want the refused address", err)
	}
	for _, secret := range []string{string(b.identity.CertPEM), "PRIVATE KEY"} {
		if strings.Contains(err.Error(), secret) {
			t.Fatalf("dial error carries certificate or key material: %q", err)
		}
	}
	// The listening side logs why it refused, by the caller's host only.
	refused := records.await(t, "reverse session failed", map[string]string{"direction": "inbound", "err": "client certificate is not a pinned member"})
	if refused["remote"] == "" || refused["peer"] != "" {
		t.Fatalf("inbound refusal record = %v", refused)
	}
}

func TestReverseSessionEstablishmentAndDuplicateAreLogged(t *testing.T) {
	records := captureLog(t)
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	startReverseHTTP(t, a, false)
	pairReverseManagers(t, a, b)
	connectReverseManagers(t, a, b)

	address := net.JoinHostPort("127.0.0.1", strconv.Itoa(a.port))
	dialed := records.await(t, "reverse session established", map[string]string{"direction": "outbound", "peer": a.identity.NodeUUID})
	if dialed["addr"] != address {
		t.Fatalf("outbound record = %v, want addr %s", dialed, address)
	}
	accepted := records.await(t, "reverse session established", map[string]string{"direction": "inbound", "peer": b.identity.NodeUUID})
	if host, _, err := net.SplitHostPort(accepted["remote"]); err != nil || host != "127.0.0.1" {
		t.Fatalf("inbound record = %v, want a loopback remote", accepted)
	}

	// A second session from the same node duplicates the first. That is
	// resolved by the reverse manager, not reported as a failure.
	node, _ := b.memberByNodeID(a.identity.NodeUUID)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	before := len(records.find("reverse session failed", nil))
	b.attemptReverseSession(ctx, *node)
	if after := len(records.find("reverse session failed", nil)); after != before {
		t.Fatalf("a duplicate session was logged as %d failure(s)", after-before)
	}
}

func TestReversePrivateRefusalNamesItsReason(t *testing.T) {
	test := func(name string, prepare func(t *testing.T) (*Manager, string), service noderec.ServiceKey, cause error) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			m, peer := prepare(t)
			startPrivateReverse(t, m)
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			start := time.Now()
			_, err := clustertrust.DialReverse(ctx, filepath.Dir(m.clusterDir), peer, service)
			var unavailable *clustertrust.ReverseUnavailableError
			if !errors.As(err, &unavailable) || unavailable.Route != clustertrust.RouteReverse || !errors.Is(err, cause) {
				t.Fatalf("dial error = %v, want a reverse-route refusal for %v", err, cause)
			}
			if elapsed := time.Since(start); elapsed > 2*time.Second {
				t.Fatalf("refusal took %v, want it reported without waiting out the setup budget", elapsed)
			}
		})
	}
	test("no reverse session", func(t *testing.T) (*Manager, string) {
		a, b := newReverseTestManager(t), newReverseTestManager(t)
		pairReverseManagers(t, a, b)
		return a, b.identity.NodeUUID
	}, noderec.ServiceNodeInfo, clustertrust.ErrReverseNoSession)
	test("peer not admitted", func(t *testing.T) (*Manager, string) {
		return newReverseTestManager(t), "stranger"
	}, noderec.ServiceNodeInfo, clustertrust.ErrReverseAdmission)
	test("private call capacity", func(t *testing.T) (*Manager, string) {
		a, b := newReverseTestManager(t), newReverseTestManager(t)
		pairReverseManagers(t, a, b)
		a.reverse.mu.Lock()
		for i := 0; i < reverseMaxPrivateCalls; i++ {
			a.reverse.calls[fmt.Sprintf("held-%d", i)] = &reversePrivateCall{}
		}
		a.reverse.mu.Unlock()
		return a, b.identity.NodeUUID
	}, noderec.ServiceNodeInfo, clustertrust.ErrReverseCapacity)
	test("remote service refused", func(t *testing.T) (*Manager, string) {
		a, b := newReverseTestManager(t), newReverseTestManager(t)
		startReverseHTTP(t, a, false)
		pairReverseManagers(t, a, b)
		connectReverseManagers(t, a, b) // a registers no local services
		return b, a.identity.NodeUUID
	}, noderec.ServiceNodeInfo, clustertrust.ErrReverseRemoteRefused)
}

func TestReverseDialFailureDoesNotLogPeerResponseBytes(t *testing.T) {
	const payload = "SECRET-RESPONSE-LINE"
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	// A listener that presents b's pinned certificate, then answers the upgrade
	// request with a line that is not HTTP.
	listener, err := tls.Listen("tcp", "127.0.0.1:0", &tls.Config{Certificates: []tls.Certificate{b.identity.Cert}})
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	b.port = listener.Addr().(*net.TCPAddr).Port
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		reader := bufio.NewReader(conn)
		for {
			line, err := reader.ReadString('\n')
			if err != nil || line == "\r\n" {
				break
			}
		}
		conn.Write([]byte(payload + "\r\n"))
	}()
	pairReverseManagers(t, a, b)
	node, ok := a.memberByNodeID(b.identity.NodeUUID)
	if !ok {
		t.Fatal("peer member missing")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	err = a.dialReverseSession(ctx, *node)
	if err == nil || strings.Contains(err.Error(), payload) {
		t.Fatalf("dial error = %v, want a failure that does not repeat the peer's response bytes", err)
	}
	if !strings.Contains(err.Error(), "upgrade response: malformed HTTP response") {
		t.Fatalf("dial error = %v, want the response stage with a fixed description", err)
	}
}
