// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func writeScannerAdmissionPin(t *testing.T, dir, uuid, cluster string, epoch uint64) {
	t.Helper()
	path := filepath.Join(dir, "trusted", uuid+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var pin struct {
		NodeUUID       string `json:"nodeUuid"`
		CertPEM        string `json:"certPem"`
		ClusterID      string `json:"clusterId"`
		AdmissionEpoch uint64 `json:"admissionEpoch"`
	}
	if err := json.Unmarshal(data, &pin); err != nil {
		t.Fatal(err)
	}
	pin.ClusterID, pin.AdmissionEpoch = cluster, epoch
	data, err = json.Marshal(pin)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	member, err := json.Marshal([]map[string]any{{"nodeUuid": uuid, "state": "member", "clusterId": cluster, "admissionEpoch": epoch}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "members.json"), member, 0o600); err != nil {
		t.Fatal(err)
	}
}

type scannerReverseFixture struct {
	d                 *daemon
	node              noderec.DirectoryNode
	direct            atomic.Bool
	reverse           atomic.Bool
	infoCalls         atomic.Int32
	modelCalls        atomic.Int32
	directInfo        atomic.Int32
	directModels      atomic.Int32
	directDials       atomic.Int32
	reverseInfo       atomic.Int32
	reverseModels     atomic.Int32
	reportedHost      atomic.Value
	reportedPrincipal atomic.Pointer[string]
	beforeReply       func(noderec.ServiceKey)
	interrupt         atomic.Bool
	responseEnding    atomic.Int32
	mu                sync.Mutex
	connections       []net.Conn
	wg                sync.WaitGroup
	listeners         []net.Listener

	// policy, when set, is served instead of scannerFixturePolicy.
	policy atomic.Pointer[json.RawMessage]
}

// scannerFixturePolicy is the policy advertisement the fixture's peer serves
// with its model inventory.
var scannerFixturePolicy = &nodepolicy.Advertisement{
	Revision: "31",
	Roles:    map[string]nodepolicy.AdvertisedRole{nodepolicy.TierMedium: {Engine: "ollama", Model: "model", ContextTokens: 16384}},
}

const (
	scannerIncompleteFraming int32 = iota + 1
	scannerAdditionalJSON
	scannerTrailingWhitespace
)

func writeScannerFixtureResponse(t *testing.T, w http.ResponseWriter, body any, ending int32) {
	t.Helper()
	if ending == scannerIncompleteFraming {
		payload, err := json.Marshal(body)
		if err != nil {
			t.Error(err)
			return
		}
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		// The JSON object is complete; only the declared HTTP body is incomplete.
		if _, err := fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s", len(payload)+1, payload); err != nil {
			t.Error(err)
		}
		return
	}
	if err := json.NewEncoder(w).Encode(body); err != nil {
		t.Error(err)
		return
	}
	if ending == scannerAdditionalJSON {
		if _, err := io.WriteString(w, "null"); err != nil {
			t.Error(err)
		}
	} else if ending == scannerTrailingWhitespace {
		if _, err := io.WriteString(w, "\r\n \t"); err != nil {
			t.Error(err)
		}
	}
}

// The private IPC fixture supplies byte streams just as cluster-manager does.
// Inner model TLS and scanner authority are real; session admission itself is
// covered by the cross-process acceptance unit rather than this IPC stub.
func newScannerReverseFixture(t *testing.T) *scannerReverseFixture {
	t.Helper()
	base, err := os.MkdirTemp("", "scanner-reverse-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(base); err != nil {
			t.Error(err)
		}
	})
	dir := filepath.Join(base, "cluster")
	peerDir := t.TempDir()
	clustertrusttest.WriteKeypair(t, dir, "self")
	clustertrusttest.WriteKeypair(t, peerDir, "peer")
	clustertrusttest.WriteAdmission(t, dir, "cluster", 1)
	clustertrusttest.WriteAdmission(t, peerDir, "cluster", 1)
	writePinFromCert(t, dir, "peer", filepath.Join(peerDir, "node.crt"))
	writePinFromCert(t, peerDir, "self", filepath.Join(dir, "node.crt"))
	writeScannerAdmissionPin(t, dir, "peer", "cluster", 1)
	peerMesh := clustertrust.Open(peerDir)
	f := &scannerReverseFixture{d: newMembershipTestDaemon(dir)}
	// Populate the finite logical routes below before publishing this fixture.
	// The transport keeps each original URL, Host and TLS principal unchanged.
	ownedDirect := make(map[string]string)
	dialer := &net.Dialer{}
	f.d.peerDialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		target, known := ownedDirect[address]
		if network != "tcp" || !known {
			return nil, errors.New("scanner fixture refuses unknown direct destination")
		}
		f.directDials.Add(1)
		return dialer.DialContext(ctx, network, target)
	}
	f.reportedHost.Store("peer")
	principal := "peer"
	f.reportedPrincipal.Store(&principal)
	f.reverse.Store(true)
	f.d.http = &http.Client{Timeout: nodeInfoFetchTimeout, Transport: nodeInfoTransport(nil)}
	f.d.modelsHTTP = &http.Client{Timeout: modelsFetchTimeout}
	f.node = noderec.DirectoryNode{HostUUID: "peer", ClusterUUID: "peer", IP: "localhost", Trusted: true,
		Services: make(map[noderec.ServiceKey]noderec.ServiceStatus)}
	info := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.infoCalls.Add(1)
		if r.Method != http.MethodGet || r.URL.Path != "/v1/node-info" || r.Host != net.JoinHostPort(f.node.IP, strconv.Itoa(f.node.Services[noderec.ServiceNodeInfo].Port)) {
			t.Errorf("unexpected node-info request: %s %s host=%s", r.Method, r.URL.Path, r.Host)
		}
		if f.beforeReply != nil {
			f.beforeReply(noderec.ServiceNodeInfo)
		}
		if f.interrupt.Load() {
			conn, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Error(err)
				return
			}
			if _, err := io.WriteString(conn, "HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n{\"hostUuid\":"); err != nil {
				t.Error(err)
			}
			if err := conn.Close(); err != nil {
				t.Error(err)
			}
			return
		}
		reportedHost, ok := f.reportedHost.Load().(string)
		if !ok {
			t.Error("invalid fixture host identity")
			return
		}
		cluster, gpuName := f.reportedPrincipal.Load(), "reverse GPU"
		if reportedHost != "peer" {
			gpuName = "stranger GPU"
		}
		if f.responseEnding.Load() == scannerIncompleteFraming {
			empty := ""
			cluster, gpuName = &empty, "uncommitted GPU"
		}
		writeScannerFixtureResponse(t, w, NodeInfoResponse{HostUUID: reportedHost, ClusterUUID: cluster, GPUs: []GPUInfo{{Name: gpuName, UtilizationPercent: 73}}, CPU: &CPUInfo{Name: "peer CPU", Cores: 8, UtilizationPercent: 42}, Memory: &MemoryInfo{TotalBytes: 8192, UsedBytes: 2048}, TelemetryValid: true, MSSince: 100}, f.responseEnding.Load())
	}))
	models := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.modelCalls.Add(1)
		if uuid, ok := peerMesh.VerifyClientPin(r); !ok || uuid != "self" {
			t.Error("missing pinned inner TLS identity")
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" || r.Host != net.JoinHostPort(f.node.IP, strconv.Itoa(f.node.Services[noderec.ServiceEngineManager].Port)) {
			t.Errorf("unexpected model request: %s %s host=%s", r.Method, r.URL.Path, r.Host)
		}
		if f.beforeReply != nil {
			f.beforeReply(noderec.ServiceEngineManager)
		}
		if f.interrupt.Load() {
			conn, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Error(err)
				return
			}
			if err := conn.Close(); err != nil {
				t.Error(err)
			}
			return
		}
		model := "model"
		if f.responseEnding.Load() == scannerIncompleteFraming {
			model = "uncommitted model"
		}
		var policy any = scannerFixturePolicy
		if raw := f.policy.Load(); raw != nil {
			policy = *raw
		}
		writeScannerFixtureResponse(t, w, map[string]any{"models": []string{model}, "modelsByEngine": map[string][]string{"ollama": {model}}, "loadedByEngine": map[string][]string{"ollama": {model}}, "policy": policy}, f.responseEnding.Load())
	}))
	models.TLS = peerMesh.ServerTLSConfig()
	models.StartTLS()
	t.Cleanup(info.Close)
	t.Cleanup(models.Close)
	targets := map[noderec.ServiceKey]string{noderec.ServiceNodeInfo: info.Listener.Addr().String(), noderec.ServiceEngineManager: models.Listener.Addr().String()}
	track := func(conn net.Conn) {
		f.mu.Lock()
		f.connections = append(f.connections, conn)
		f.mu.Unlock()
	}
	forward := func(conn net.Conn, target string, first io.Reader) {
		upstream, err := net.DialTimeout("tcp", target, time.Second)
		if err != nil {
			t.Error(err)
			return
		}
		track(upstream)
		defer upstream.Close()
		if err := upstream.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
			t.Error(err)
			return
		}
		done := make(chan struct{})
		go func() { _, _ = io.Copy(upstream, first); upstream.Close(); close(done) }()
		_, _ = io.Copy(conn, upstream)
		conn.Close()
		select {
		case <-done:
		case <-time.After(6 * time.Second):
			t.Error("forwarding cleanup timed out")
		}
	}
	start := func(listener net.Listener, serve func(net.Conn)) {
		f.listeners = append(f.listeners, listener)
		f.wg.Add(1)
		go func() {
			defer f.wg.Done()
			for {
				conn, err := listener.Accept()
				if err != nil {
					return
				}
				track(conn)
				f.wg.Add(1)
				go func() {
					defer f.wg.Done()
					defer conn.Close()
					if err := conn.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
						t.Error(err)
						return
					}
					serve(conn)
				}()
			}
		}()
	}
	for _, service := range []noderec.ServiceKey{noderec.ServiceNodeInfo, noderec.ServiceEngineManager} {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		port := listener.Addr().(*net.TCPAddr).Port
		f.node.Services[service] = noderec.ServiceStatus{Port: port}
		for _, host := range []string{"localhost", "192.0.2.40"} {
			ownedDirect[net.JoinHostPort(host, strconv.Itoa(port))] = listener.Addr().String()
		}
		start(listener, func(conn net.Conn) {
			if service == noderec.ServiceNodeInfo {
				f.directInfo.Add(1)
			} else {
				f.directModels.Add(1)
			}
			if f.direct.Load() {
				forward(conn, targets[service], conn)
			}
		})
	}
	private, err := clustertrust.ListenReversePrivate(clustertrust.ReverseDialPath(base))
	if err != nil {
		t.Fatal(err)
	}
	start(private, func(conn net.Conn) {
		reader := bufio.NewReader(conn)
		header, err := reader.ReadBytes('\n')
		if err != nil {
			return
		}
		var request clustertrust.ReverseDialRequest
		if err := json.Unmarshal(header, &request); err != nil {
			t.Error(err)
			return
		}
		if request.Watch {
			if _, err := conn.Write([]byte{0}); err != nil {
				return
			}
			_, _ = io.Copy(io.Discard, reader)
			return
		}
		if request.PeerUUID != "peer" || targets[request.Service] == "" {
			t.Errorf("wrong private target: %+v", request)
			return
		}
		if !f.reverse.Load() {
			return
		}
		if request.Service == noderec.ServiceNodeInfo {
			f.reverseInfo.Add(1)
		} else {
			f.reverseModels.Add(1)
		}
		if _, err := conn.Write([]byte{0}); err != nil {
			return
		}
		forward(conn, targets[request.Service], reader)
	})
	f.d.dir.upsert(f.node)
	t.Cleanup(func() {
		f.d.closePeerClients()
		for _, listener := range f.listeners {
			if err := listener.Close(); err != nil {
				t.Error(err)
			}
		}
		f.mu.Lock()
		for _, conn := range f.connections {
			conn.Close()
		}
		f.mu.Unlock()
		done := make(chan struct{})
		go func() { f.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(6 * time.Second):
			t.Error("fixture cleanup timed out")
		}
		for _, listener := range f.listeners {
			if listener.Addr().Network() != "tcp" {
				continue
			}
			conn, err := net.DialTimeout("tcp", listener.Addr().String(), 100*time.Millisecond)
			if err == nil {
				conn.Close()
				t.Errorf("fixture listener survived cleanup: %s", listener.Addr())
			}
		}
	})
	return f
}

func TestScannerReverseEnrichmentAndDirectRecovery(t *testing.T) {
	f := newScannerReverseFixture(t)
	node := f.node
	f.d.enrich(&node)
	if f.directInfo.Load() != 1 || f.directModels.Load() != 1 || f.reverseInfo.Load() != 1 || f.reverseModels.Load() != 1 {
		t.Fatalf("initial direct/reverse attempts = ni %d/%d em %d/%d, want 1/1 each", f.directInfo.Load(), f.reverseInfo.Load(), f.directModels.Load(), f.reverseModels.Load())
	}
	if len(node.GPUs) != 1 || node.GPUs[0].Name != "reverse GPU" || !reflect.DeepEqual(node.Models, []string{"model"}) || !reflect.DeepEqual(node.ModelsByEngine, map[string][]string{"ollama": {"model"}}) || !reflect.DeepEqual(node.LoadedByEngine, map[string][]string{"ollama": {"model"}}) {
		t.Fatalf("reverse inventory = %+v", node)
	}
	var output bytes.Buffer
	f.d.codec = NewCodec(&output)
	if !f.d.refreshNodeTelemetry(context.Background(), "peer", "localhost", f.node.Services[noderec.ServiceNodeInfo].Port) {
		t.Fatal("reverse telemetry unavailable")
	}
	var message Message
	if err := json.Unmarshal(bytes.TrimSpace(output.Bytes()), &message); err != nil {
		t.Fatal(err)
	}
	var telemetry noderec.NodeTelemetry
	if err := json.Unmarshal(message.Params, &telemetry); err != nil {
		t.Fatal(err)
	}
	if message.Method != noderec.NotifyNodeTelemetry || telemetry.HostUUID != "peer" || !telemetry.TelemetryValid || telemetry.GPUUtilizationPct != 73 || telemetry.MSSince < 100 || telemetry.MSSince >= 3500 {
		t.Fatalf("reverse telemetry = %+v method=%s", telemetry, message.Method)
	}
	f.d.codec = nil
	lastInfoAt := f.d.lastInfoAt["peer"]
	f.reverse.Store(false)
	f.d.enrich(&node)
	if len(node.GPUs) != 1 || node.GPUs[0].Name != "reverse GPU" || !reflect.DeepEqual(node.Models, []string{"model"}) {
		t.Fatalf("route loss discarded last-known inventory: %+v", node)
	}
	if f.d.lastInfoAt["peer"] != lastInfoAt {
		t.Fatal("failed route advanced the last-good freshness timestamp")
	}
	if f.d.refreshNodeTelemetry(context.Background(), "peer", "localhost", f.node.Services[noderec.ServiceNodeInfo].Port) {
		t.Fatal("unavailable route emitted a new telemetry sample")
	}
	if f.d.refreshNodeClusterIdentity("peer", "localhost", f.node.Services[noderec.ServiceNodeInfo].Port) || mustClusterUUID(t, f.d, "peer") != "peer" {
		t.Fatal("route loss changed membership")
	}
	f.direct.Store(true)
	// Readmission changes the route identity and legitimately resets direct health.
	clustertrusttest.WriteAdmission(t, f.d.clusterDir, "cluster", 2)
	beforeInfo, beforeModels := f.directInfo.Load(), f.directModels.Load()
	if _, ok := f.d.fetchPeerNodeInfoWithin(context.Background(), "peer", "peer", "localhost", f.node.Services[noderec.ServiceNodeInfo].Port); !ok {
		t.Fatal("recovered direct node-info failed")
	}
	if inventory, ok := f.d.fetchPeerModels("peer", "localhost", f.node.Services[noderec.ServiceEngineManager].Port, "peer"); !ok || !reflect.DeepEqual(inventory.models, []string{"model"}) {
		t.Fatalf("recovered direct models=%v ok=%v", inventory.models, ok)
	}
	if f.directInfo.Load()-beforeInfo != 1 || f.directModels.Load()-beforeModels != 1 {
		t.Fatal("readmission did not recover the direct paths")
	}
}

func TestScannerDirectPreferenceWithoutReverseSession(t *testing.T) {
	f := newScannerReverseFixture(t)
	f.direct.Store(true)
	f.reverse.Store(false)
	node := f.node
	f.d.enrich(&node)
	if len(node.GPUs) != 1 || !reflect.DeepEqual(node.Models, []string{"model"}) {
		t.Fatalf("direct enrichment failed: %+v", node)
	}
	if f.directInfo.Load() != 1 || f.directModels.Load() != 1 || f.reverseInfo.Load() != 0 || f.reverseModels.Load() != 0 {
		t.Fatal("healthy direct requests used reverse")
	}
	if f.directDials.Load() != 2 {
		t.Fatalf("owned direct dials = %d, want 2 through the scanner transport", f.directDials.Load())
	}
}

func TestScannerFixtureDirectDialRejectsUnknownAndCanceled(t *testing.T) {
	f := newScannerReverseFixture(t)
	address := net.JoinHostPort("localhost", strconv.Itoa(f.node.Services[noderec.ServiceNodeInfo].Port))
	for _, tc := range []struct{ name, network, address string }{
		{"unknown host", "tcp", net.JoinHostPort("192.0.2.41", strconv.Itoa(f.node.Services[noderec.ServiceNodeInfo].Port))},
		{"unknown port", "tcp", "localhost:0"},
		{"unknown network", "udp", address},
	} {
		t.Run(tc.name, func(t *testing.T) {
			conn, err := f.d.peerDialContext(context.Background(), tc.network, tc.address)
			if conn != nil {
				if closeErr := conn.Close(); closeErr != nil {
					t.Error(closeErr)
				}
				t.Fatal("unknown destination returned a connection")
			}
			if err == nil {
				t.Fatal("unknown destination was not refused")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	conn, err := f.d.peerDialContext(ctx, "tcp", address)
	if conn != nil {
		if closeErr := conn.Close(); closeErr != nil {
			t.Error(closeErr)
		}
		t.Fatal("canceled dial returned a connection")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled dial error = %v, want context cancellation", err)
	}
	if f.directDials.Load() != 0 || f.directInfo.Load() != 0 || f.directModels.Load() != 0 || f.reverseInfo.Load() != 0 || f.reverseModels.Load() != 0 {
		t.Fatal("refused or canceled fixture dial reached a listener")
	}
}

func TestScannerDirectInventoryReplyLossDoesNotReplay(t *testing.T) {
	f := newScannerReverseFixture(t)
	f.direct.Store(true)
	f.interrupt.Store(true)
	if _, ok := f.d.fetchPeerModels("peer", "localhost", f.node.Services[noderec.ServiceEngineManager].Port, "peer"); ok {
		t.Fatal("interrupted direct inventory succeeded")
	}
	if f.modelCalls.Load() != 1 || f.directModels.Load() != 1 || f.reverseModels.Load() != 0 {
		t.Fatalf("lost direct response dispatched %d times; direct=%d reverse=%d", f.modelCalls.Load(), f.directModels.Load(), f.reverseModels.Load())
	}
}

func TestScannerReverseRejectsWrongHostIdentity(t *testing.T) {
	f := newScannerReverseFixture(t)
	node := f.node
	f.d.enrichInfoAt(&node, "localhost")
	f.reportedHost.Store("stranger")
	f.d.enrichInfoAt(&node, "localhost")
	if len(node.GPUs) != 1 || node.GPUs[0].Name != "reverse GPU" {
		t.Fatalf("wrong-host read discarded last-good inventory: %+v", node)
	}
	if f.d.refreshNodeTelemetry(context.Background(), "peer", "localhost", f.node.Services[noderec.ServiceNodeInfo].Port) {
		t.Fatal("wrong host emitted telemetry")
	}
	if f.d.refreshNodeClusterIdentity("peer", "localhost", f.node.Services[noderec.ServiceNodeInfo].Port) || mustClusterUUID(t, f.d, "peer") != "peer" {
		t.Fatal("wrong-host read changed membership")
	}
}

func TestScannerReverseReadLossDoesNotReplayInventory(t *testing.T) {
	f := newScannerReverseFixture(t)
	node := f.node
	f.d.enrich(&node)
	f.interrupt.Store(true)
	before := f.modelCalls.Load()
	f.d.enrich(&node)
	if f.modelCalls.Load()-before != 1 {
		t.Fatalf("interrupted inventory dispatched %d times, want 1", f.modelCalls.Load()-before)
	}
	if !reflect.DeepEqual(node.Models, []string{"model"}) || len(node.GPUs) != 1 || node.GPUs[0].Name != "reverse GPU" {
		t.Fatalf("interrupted read lost cache: %+v", node)
	}
	if f.d.refreshNodeClusterIdentity("peer", "localhost", f.node.Services[noderec.ServiceNodeInfo].Port) || mustClusterUUID(t, f.d, "peer") != "peer" {
		t.Fatal("interrupted read changed membership")
	}
	f.interrupt.Store(false)
	if !f.d.refreshNodeTelemetry(context.Background(), "peer", "localhost", f.node.Services[noderec.ServiceNodeInfo].Port) {
		t.Fatal("reverse recovery failed")
	}
}

func TestScannerRejectsResponseAfterAuthorityLoss(t *testing.T) {
	for _, scenario := range []string{"admission", "pin", "pin replacement", "self certificate replacement", "route"} {
		t.Run(scenario, func(t *testing.T) {
			for _, service := range []noderec.ServiceKey{noderec.ServiceNodeInfo, noderec.ServiceEngineManager} {
				t.Run(string(service), func(t *testing.T) {
					f := newScannerReverseFixture(t)
					entered, release := make(chan struct{}), make(chan struct{})
					var releaseOnce sync.Once
					releaseReply := func() { releaseOnce.Do(func() { close(release) }) }
					t.Cleanup(releaseReply)
					f.beforeReply = func(requested noderec.ServiceKey) {
						if requested == service {
							close(entered)
							<-release
						}
					}
					done := make(chan bool, 1)
					go func() {
						if service == noderec.ServiceNodeInfo {
							_, ok := f.d.fetchPeerNodeInfoWithin(context.Background(), "peer", "peer", "localhost", f.node.Services[service].Port)
							done <- ok
						} else {
							_, ok := f.d.fetchPeerModels("peer", "localhost", f.node.Services[service].Port, "peer")
							done <- ok
						}
					}()
					select {
					case <-entered:
					case <-time.After(3 * time.Second):
						releaseReply()
						t.Fatal("read did not reach receiver")
					}
					switch scenario {
					case "admission":
						clustertrusttest.WriteAdmission(t, f.d.clusterDir, "cluster", 2)
					case "pin":
						clustertrusttest.RemovePeerPin(t, f.d.clusterDir, "peer")
					case "pin replacement":
						clustertrusttest.WritePeerPin(t, f.d.clusterDir, "peer")
						writeScannerAdmissionPin(t, f.d.clusterDir, "peer", "cluster", 1)
					case "self certificate replacement":
						clustertrusttest.WriteKeypair(t, f.d.clusterDir, "self")
					case "route":
						f.d.dir.remove("peer")
					}
					releaseReply()
					select {
					case ok := <-done:
						if ok {
							t.Fatal("response accepted after authority changed")
						}
					case <-time.After(3 * time.Second):
						t.Fatal("read did not finish")
					}
				})
			}
		})
	}
}

func TestScannerRejectsIncompleteNodeInfoFraming(t *testing.T) {
	f := newScannerReverseFixture(t)
	node := f.node
	f.d.enrichInfoAt(&node, "localhost")
	if len(node.GPUs) != 1 || node.GPUs[0].Name != "reverse GPU" {
		t.Fatal("missing prior good node-info inventory")
	}
	previousAt := f.d.lastInfoAt["peer"]
	f.responseEnding.Store(scannerIncompleteFraming)
	beforeCalls := f.infoCalls.Load()
	f.d.enrichInfoAt(&node, "localhost")
	if f.infoCalls.Load()-beforeCalls != 1 {
		t.Fatalf("incomplete node-info dispatched %d times, want 1", f.infoCalls.Load()-beforeCalls)
	}
	if len(node.GPUs) != 1 || node.GPUs[0].Name != "reverse GPU" || len(f.d.lastInfo["peer"].GPUs) != 1 || f.d.lastInfo["peer"].GPUs[0].Name != "reverse GPU" {
		t.Fatalf("incomplete body overwrote prior inventory: node=%+v cache=%+v", node.GPUs, f.d.lastInfo["peer"].GPUs)
	}
	if !f.d.nodeInfoDown["peer"] || f.d.lastInfoAt["peer"] != previousAt {
		t.Fatal("incomplete body falsely reported a fresh node-info observation")
	}
	if f.d.refreshNodeClusterIdentity("peer", "localhost", f.node.Services[noderec.ServiceNodeInfo].Port) || mustClusterUUID(t, f.d, "peer") != "peer" {
		t.Fatal("incomplete node-info body changed membership")
	}
	if f.d.refreshNodeTelemetry(context.Background(), "peer", "localhost", f.node.Services[noderec.ServiceNodeInfo].Port) {
		t.Fatal("incomplete body emitted telemetry")
	}
}

func TestScannerRejectsIncompleteInventoryFraming(t *testing.T) {
	test := func(name string, direct bool, wantDirect, wantReverse int32) {
		t.Run(name, func(t *testing.T) {
			f := newScannerReverseFixture(t)
			f.direct.Store(direct)
			node := f.node
			f.d.enrichModelsAt(&node, "localhost")
			f.d.dir.upsert(node)
			if !reflect.DeepEqual(node.Models, []string{"model"}) {
				t.Fatal("missing prior good model inventory")
			}
			f.responseEnding.Store(scannerIncompleteFraming)
			beforeCalls, beforeDirect, beforeReverse := f.modelCalls.Load(), f.directModels.Load(), f.reverseModels.Load()
			if f.d.refreshNodeModels("peer", "localhost", f.node.Services[noderec.ServiceEngineManager].Port, "peer") {
				t.Fatal("incomplete inventory body reported a successful refresh")
			}
			if f.modelCalls.Load()-beforeCalls != 1 || f.directModels.Load()-beforeDirect != wantDirect || f.reverseModels.Load()-beforeReverse != wantReverse {
				t.Fatalf("incomplete inventory attempts: receiver=%d direct=%d reverse=%d, want 1/%d/%d", f.modelCalls.Load()-beforeCalls, f.directModels.Load()-beforeDirect, f.reverseModels.Load()-beforeReverse, wantDirect, wantReverse)
			}
			current, ok := f.d.dir.get("peer")
			if !ok || current.ClusterUUID != "peer" || !current.Trusted {
				t.Fatalf("incomplete inventory changed membership: %+v", current)
			}
			wantModels := []string{"model"}
			wantByEngine := map[string][]string{"ollama": {"model"}}
			if !reflect.DeepEqual(current.Models, wantModels) || !reflect.DeepEqual(current.ModelsByEngine, wantByEngine) || !reflect.DeepEqual(current.LoadedByEngine, wantByEngine) || !reflect.DeepEqual(f.d.lastModels["peer"], wantModels) || !reflect.DeepEqual(f.d.lastModelsByEngine["peer"], wantByEngine) || !reflect.DeepEqual(f.d.lastLoadedByEngine["peer"], wantByEngine) {
				t.Fatalf("incomplete body changed directory/cache inventory: node=%+v cache=%v/%v/%v", current, f.d.lastModels["peer"], f.d.lastModelsByEngine["peer"], f.d.lastLoadedByEngine["peer"])
			}
		})
	}
	test("reverse reply", false, 0, 1)
	test("direct reply cannot fall back to reverse", true, 1, 0)
}

func TestScannerRequiresOneJSONValueAndCleanBodyEnd(t *testing.T) {
	test := func(name string, ending int32, wantOK bool) {
		t.Run(name, func(t *testing.T) {
			f := newScannerReverseFixture(t)
			f.responseEnding.Store(ending)
			if _, ok := f.d.fetchPeerNodeInfoWithin(context.Background(), "peer", "peer", "localhost", f.node.Services[noderec.ServiceNodeInfo].Port); ok != wantOK {
				t.Fatalf("node-info success = %v, want %v", ok, wantOK)
			}
			if _, ok := f.d.fetchPeerModels("peer", "localhost", f.node.Services[noderec.ServiceEngineManager].Port, "peer"); ok != wantOK {
				t.Fatalf("inventory success = %v, want %v", ok, wantOK)
			}
		})
	}
	test("rejects an additional JSON value", scannerAdditionalJSON, false)
	test("accepts trailing whitespace and clean EOF", scannerTrailingWhitespace, true)
}
