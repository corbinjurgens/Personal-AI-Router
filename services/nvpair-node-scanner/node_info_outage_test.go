// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	sharederrors "nvpair-shared/errors"
	"nvpair-shared/noderec"
)

// A peer whose node-info address silently drops packets is the failure these
// cover. It cost a full request timeout on every sweep and said so only at debug
// level, so an address that had never once answered was indistinguishable in the
// log from one that was working.

// captureLogs redirects the default logger for one test.
func captureLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	buf := &bytes.Buffer{}
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(buf, &slog.HandlerOptions{Level: slog.LevelInfo})))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return buf
}

// TestNodeInfoOutageIsStatedOnceAtEachEnd pins the reporting to the transitions.
// The sweep repeats for as long as the node is advertised, so reporting every
// attempt is what made this unreadable — and reporting none of them is what hid
// it entirely.
func TestNodeInfoOutageIsStatedOnceAtEachEnd(t *testing.T) {
	logs := captureLogs(t)
	d := &daemon{}

	for range 3 {
		d.noteNodeInfo(noderec.DirectoryNode{HostUUID: "peer-uuid"}, "10.0.0.5:14300", false)
	}
	if got := strings.Count(logs.String(), "node-info is not answering"); got != 1 {
		t.Fatalf("outage reported %d times, want exactly 1:\n%s", got, logs.String())
	}
	if !strings.Contains(logs.String(), "level=WARN") {
		t.Fatalf("an unreachable peer must be visible above debug:\n%s", logs.String())
	}
	if !strings.Contains(logs.String(), "10.0.0.5:14300") {
		t.Fatalf("the report must name where the node was asked:\n%s", logs.String())
	}

	for range 2 {
		d.noteNodeInfo(noderec.DirectoryNode{HostUUID: "peer-uuid"}, "10.0.0.5:14300", true)
	}
	if got := strings.Count(logs.String(), "node-info is answering again"); got != 1 {
		t.Fatalf("recovery reported %d times, want exactly 1:\n%s", got, logs.String())
	}
}

// TestNodeInfoSilenceWhenNothingChanges keeps the steady state quiet: a peer that
// has always answered has nothing to say about itself.
func TestNodeInfoSilenceWhenNothingChanges(t *testing.T) {
	logs := captureLogs(t)
	d := &daemon{}

	for range 5 {
		d.noteNodeInfo(noderec.DirectoryNode{HostUUID: "peer-uuid"}, "10.0.0.5:14300", true)
	}
	if logs.Len() != 0 {
		t.Fatalf("a healthy peer must log nothing:\n%s", logs.String())
	}
}

// TestNodeInfoConnectGivesUpBeforeTheRequestBudget exercises the real dialer's
// connect budget during controlled resolution, without sending DNS or TCP
// packets. It does not measure a kernel SYN drop.
func TestNodeInfoConnectGivesUpBeforeTheRequestBudget(t *testing.T) {
	if nodeInfoDialTimeout >= nodeInfoFetchTimeout {
		t.Fatalf("a connect budget of %v inside a request budget of %v splits nothing",
			nodeInfoDialTimeout, nodeInfoFetchTimeout)
	}

	dialer := nodeInfoDialer()
	var connectAttempted atomic.Bool
	// A static-host answer can bypass DNS; refuse before any socket connects.
	dialer.ControlContext = func(context.Context, string, string, syscall.RawConn) error {
		connectAttempted.Store(true)
		return errors.New("socket connect forbidden in controlled resolver test")
	}
	resolverReached := make(chan struct{})
	resolverStopped := make(chan struct{})
	var reached, stopped sync.Once
	dialer.Resolver = &net.Resolver{
		PreferGo: true,
		Dial: func(ctx context.Context, _, _ string) (net.Conn, error) {
			reached.Do(func() { close(resolverReached) })
			<-ctx.Done()
			stopped.Do(func() { close(resolverStopped) })
			return nil, ctx.Err()
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), nodeInfoFetchTimeout)
	defer cancel()
	start := time.Now()
	conn, err := dialer.DialContext(ctx, "tcp4", "node-info-connect-budget.invalid.:14300")
	elapsed := time.Since(start)
	if connectAttempted.Load() {
		t.Fatal("socket connect guard reached; want the controlled DNS timeout")
	}
	if err == nil {
		if closeErr := conn.Close(); closeErr != nil {
			t.Errorf("close unexpected connection: %v", closeErr)
		}
		t.Fatal("connect succeeded while resolution was held until cancellation")
	}
	select {
	case <-resolverReached:
	default:
		t.Fatal("real DialContext did not reach the controlled resolver")
	}
	select {
	case <-resolverStopped:
	case <-time.After(nodeInfoDialTimeout):
		t.Fatal("controlled resolver did not stop after the dial timed out")
	}
	var timeout net.Error
	if !errors.As(err, &timeout) || !timeout.Timeout() {
		t.Fatalf("connect error = %v, want a timeout", err)
	}
	if ctx.Err() != nil {
		t.Fatalf("connect consumed the request budget: %v", ctx.Err())
	}
	if dialer.Timeout != nodeInfoDialTimeout {
		t.Fatalf("configured connect budget = %v, want %v", dialer.Timeout, nodeInfoDialTimeout)
	}
	if elapsed < nodeInfoDialTimeout || elapsed >= nodeInfoDialTimeout+(nodeInfoFetchTimeout-nodeInfoDialTimeout)/2 {
		t.Fatalf("connect took %v; want the connect budget (%v) with room before the request budget (%v)",
			elapsed, nodeInfoDialTimeout, nodeInfoFetchTimeout)
	}
}

func TestNodeInfoTransportBoundsConnectionsPerHost(t *testing.T) {
	transport := nodeInfoTransport(nil)
	if transport.MaxConnsPerHost != 1 {
		t.Fatalf("MaxConnsPerHost = %d, want 1", transport.MaxConnsPerHost)
	}
	if transport.MaxIdleConnsPerHost != 1 {
		t.Fatalf("MaxIdleConnsPerHost = %d, want 1", transport.MaxIdleConnsPerHost)
	}
}

func TestNodeInfoConcurrentFetchesGetIndependentRequestBudgets(t *testing.T) {
	const responseDelay = 200 * time.Millisecond
	var requests atomic.Int32
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		time.Sleep(responseDelay)
		_ = json.NewEncoder(w).Encode(NodeInfoResponse{HostUUID: "peer-uuid"})
	}))
	var connections atomic.Int32
	server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			connections.Add(1)
		}
	}
	server.Start()
	defer server.Close()

	serverURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("parse server URL: %v", err)
	}
	port, err := strconv.Atoi(serverURL.Port())
	if err != nil {
		t.Fatalf("parse server port: %v", err)
	}
	transport := nodeInfoTransport(nil)
	defer transport.CloseIdleConnections()
	d := &daemon{http: &http.Client{Timeout: 300 * time.Millisecond, Transport: transport}}

	start := make(chan struct{})
	results := make(chan bool, 2)
	var wg sync.WaitGroup
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, ok := d.fetchNodeInfo(serverURL.Hostname(), port)
			results <- ok
		}()
	}
	close(start)
	wg.Wait()
	close(results)

	for ok := range results {
		if !ok {
			t.Fatal("a queued healthy request consumed its timeout before reaching the peer")
		}
	}
	if got := requests.Load(); got != 2 {
		t.Fatalf("server received %d requests, want 2", got)
	}
	if got := connections.Load(); got != 1 {
		t.Fatalf("requests used %d TCP connections, want 1 serialized connection", got)
	}
}

func TestNodeInfoOriginGateHonorsCanceledWaiter(t *testing.T) {
	var gate nodeInfoOriginGate
	release, acquired := gate.acquire(context.Background(), "http://127.0.0.1:14318")
	if !acquired {
		t.Fatal("first origin request did not acquire the gate")
	}
	defer release()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	waitingRelease, waitingAcquired := gate.acquire(ctx, "http://127.0.0.1:14318")
	if waitingAcquired {
		waitingRelease()
		t.Fatal("canceled origin request acquired the occupied gate")
	}
}

func TestNodeInfoErrorBodyIsDrainedForConnectionReuse(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if requests.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte("temporarily unavailable"))
			return
		}
		_ = json.NewEncoder(w).Encode(NodeInfoResponse{HostUUID: "peer-uuid"})
	}))
	var connections atomic.Int32
	server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			connections.Add(1)
		}
	}
	server.Start()
	defer server.Close()

	serverURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("parse server URL: %v", err)
	}
	port, err := strconv.Atoi(serverURL.Port())
	if err != nil {
		t.Fatalf("parse server port: %v", err)
	}
	transport := nodeInfoTransport(nil)
	defer transport.CloseIdleConnections()
	d := &daemon{http: &http.Client{Timeout: time.Second, Transport: transport}}

	if _, ok := d.fetchNodeInfo(serverURL.Hostname(), port); ok {
		t.Fatal("HTTP 503 unexpectedly succeeded")
	}
	info, ok := d.fetchNodeInfo(serverURL.Hostname(), port)
	if !ok || info.HostUUID != "peer-uuid" {
		t.Fatalf("second fetch = %+v,%v, want successful peer response", info, ok)
	}
	if got := connections.Load(); got != 1 {
		t.Fatalf("requests used %d TCP connections, want 1 reused connection", got)
	}
}

// reportingDaemon is a daemon whose codec output is captured and whose directory
// has not yet entered the directory, as is the case for a peer being discovered.
func reportingDaemon(t *testing.T) (*daemon, noderec.DirectoryNode, func() []Message) {
	t.Helper()
	var out bytes.Buffer
	d := &daemon{codec: NewCodec(&out), dir: newDirectory()}
	peer := noderec.DirectoryNode{HostUUID: "peer-uuid", Name: "mac-studio", Trusted: true}
	sent := func() []Message {
		var msgs []Message
		decoder := json.NewDecoder(bytes.NewReader(out.Bytes()))
		for decoder.More() {
			var msg Message
			if err := decoder.Decode(&msg); err != nil {
				t.Fatalf("decode captured frame: %v", err)
			}
			msgs = append(msgs, msg)
		}
		return msgs
	}
	return d, peer, sent
}

func requireErrorsFrames(t *testing.T, msgs []Message, methods ...string) {
	t.Helper()
	if len(msgs) != len(methods) {
		t.Fatalf("sent %d frames %+v, want methods %v", len(msgs), msgs, methods)
	}
	for i, method := range methods {
		if msgs[i].Method != method {
			t.Fatalf("frame %d method = %q, want %q", i, msgs[i].Method, method)
		}
	}
}

// TestNodeInfoOutageIsReportedOnceAndClearedOnRecovery covers the errors view:
// one report when a paired peer stops answering, none while it stays down, one
// clear when it answers again, and nothing further while it stays healthy.
func TestNodeInfoOutageIsReportedOnceAndClearedOnRecovery(t *testing.T) {
	d, peer, sent := reportingDaemon(t)

	for range 3 {
		d.noteNodeInfo(peer, "10.0.0.5:14300", false)
	}
	requireErrorsFrames(t, sent(), "errors:report")
	var report sharederrors.ServiceError
	if err := json.Unmarshal(sent()[0].Params, &report); err != nil {
		t.Fatal(err)
	}
	if report.ID != "node-scanner:node-info-unreachable:peer-uuid" || report.Severity != "warning" || report.Action != "none" {
		t.Fatalf("report = %+v", report)
	}
	if !strings.Contains(report.Message, "mac-studio") || !strings.Contains(report.Message, "10.0.0.5:14300") {
		t.Fatalf("message must name the peer and where it was asked: %q", report.Message)
	}

	for range 2 {
		d.noteNodeInfo(peer, "10.0.0.5:14300", true)
	}
	msgs := sent()
	requireErrorsFrames(t, msgs, "errors:report", "errors:clear")
	var cleared sharederrors.ClearParams
	if err := json.Unmarshal(msgs[1].Params, &cleared); err != nil {
		t.Fatal(err)
	}
	if cleared.ID != report.ID {
		t.Fatalf("clear names %q, want %q", cleared.ID, report.ID)
	}
}

// TestNodeInfoOutageIsClearedOnEviction: forgetting a node takes its report out
// of the errors view. The peer was never in the directory, which is how a peer
// that is already down when it is discovered looks: it is still reported.
func TestNodeInfoOutageIsClearedOnEviction(t *testing.T) {
	d, peer, sent := reportingDaemon(t)

	d.noteNodeInfo(peer, "10.0.0.5:14300", false)
	d.forget(peer.HostUUID)
	requireErrorsFrames(t, sent(), "errors:report", "errors:clear")
}

// TestNodeInfoOutageReportsOnlyPairedRemotePeers: an unpaired LAN node and this
// node's own loopback node-info are not the operator's to act on here.
func TestNodeInfoOutageReportsOnlyPairedRemotePeers(t *testing.T) {
	d, peer, sent := reportingDaemon(t)

	unpaired := peer
	unpaired.Trusted = false
	d.noteNodeInfo(unpaired, "10.0.0.5:14300", false)
	d.noteNodeInfo(peer, "127.0.0.1:14300", false)
	requireErrorsFrames(t, sent())
}

// TestNodeInfoOutageReportsOnceThePeerIsPaired: a peer that was already down when
// it became paired is reported on its next node-info enrichment, and un-pairing clears it.
func TestNodeInfoOutageReportsOnceThePeerIsPaired(t *testing.T) {
	d, peer, sent := reportingDaemon(t)

	unpaired := peer
	unpaired.Trusted = false
	d.noteNodeInfo(unpaired, "10.0.0.5:14300", false)
	d.noteNodeInfo(peer, "10.0.0.5:14300", false)
	requireErrorsFrames(t, sent(), "errors:report")

	d.noteNodeInfo(unpaired, "10.0.0.5:14300", false)
	requireErrorsFrames(t, sent(), "errors:report", "errors:clear")
}
