// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
)

func pausedAdvertisementPlatform(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "windows" {
		t.Skip("PA process acceptance is reserved for Windows PC3")
	}
	if !filepath.IsAbs(os.Getenv("NVPAIR_TEST_BIN_DIR")) || !strings.Contains(os.Getenv("GOFLAGS"), "nvpair-test-exec") {
		t.Fatal("PA fixture requires the fixed-path nvpair-test-exec wrapper and absolute NVPAIR_TEST_BIN_DIR")
	}
}

// This guarded subscription-scanner role forwards to the production scanner.
// Only private admitted descriptor claims are injected; directory replies and
// all inventory/policy notifications come from the real scanner unchanged.
func runPausedAdvertisementScanner() error {
	var claim noderec.PeerServicesClaim
	if err := json.Unmarshal([]byte(os.Getenv("NVPAIR_TEST_PA_CLAIM")), &claim); err != nil {
		return fmt.Errorf("decode PA claim: %w", err)
	}
	if err := noderec.ValidatePeerServicesDescriptor(claim.Descriptor, claim.Descriptor.HostUUID); err != nil {
		return err
	}
	args := append(os.Args[1:], "--mdns=false")
	cmd := exec.Command(os.Getenv("NVPAIR_TEST_PA_SCANNER"), args...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	lifetime, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	interrupt := context.AfterFunc(lifetime, func() { stdin.Close() })
	defer interrupt()
	wait := make(chan error, 1)
	go func() { wait <- cmd.Wait() }()
	frames := make(chan jsonrpc.Message)
	readErr := make(chan error, 1)
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		codec := jsonrpc.NewCodecMaxFrame(struct {
			io.Reader
			io.Writer
		}{os.Stdin, io.Discard}, jsonrpc.WorkerFrameBytes)
		for {
			frame, err := codec.Read()
			if err != nil {
				readErr <- err
				return
			}
			select {
			case frames <- *frame:
			case <-stop:
				return
			}
		}
	}()
	encoder := json.NewEncoder(stdin)
	refreshClaim := func() error {
		claim.Sequence++
		claim.ObservedAt = time.Now().UnixMilli()
		params, err := json.Marshal(claim)
		if err != nil {
			return err
		}
		return encoder.Encode(jsonrpc.Message{JSONRPC: "2.0", Method: "discovery:peer-services", Params: params})
	}
	// Refresh only authority freshness, never descriptor content or cadence.
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	_, forwardErr := fmt.Fprintf(os.Stderr, "fixture_worker_started pid=%d\n", cmd.Process.Pid)
	if forwardErr == nil {
		forwardErr = refreshClaim()
	}
	forwarding := forwardErr == nil
	for forwarding {
		select {
		case frame := <-frames:
			forwardErr = encoder.Encode(frame)
		case <-ticker.C:
			forwardErr = refreshClaim()
		case forwardErr = <-readErr:
			forwarding = false
		case err := <-wait:
			return fmt.Errorf("PA scanner exited before broker EOF: %v", err)
		case <-lifetime.Done():
			forwardErr = lifetime.Err()
		}
		if forwardErr != nil {
			forwarding = false
		}
	}
	closeErr := stdin.Close()
	select {
	case err := <-wait:
		if err != nil {
			return fmt.Errorf("PA scanner exit: %w", err)
		}
	case <-time.After(time.Second):
		if err := cmd.Process.Kill(); err != nil {
			return fmt.Errorf("kill PA scanner: %w", err)
		}
		select {
		case <-wait:
		case <-time.After(time.Second):
			return fmt.Errorf("PA scanner did not exit after kill")
		}
		return fmt.Errorf("PA scanner required forced cleanup")
	}
	if forwardErr != nil && forwardErr != io.EOF {
		return forwardErr
	}
	if closeErr != nil {
		return closeErr
	}
	_, err = fmt.Fprintf(os.Stderr, "fixture_worker_exited pid=%d exit=0\n", cmd.Process.Pid)
	return err
}

// Adapt only the scanner module's private reverse IPC fixture protocol. The
// outer reverse session is a fixture stub; production scanner admission and
// inner pinned mTLS remain in use. This proves propagation, not outer-session
// authentication, live inference, or engine/model serving.
func pausedAdvertisementReverse(t *testing.T, ctx context.Context, base, incarnation, target string, mesh *clustertrust.Mesh, reverseConnects *atomic.Uint64) {
	t.Helper()
	listener, err := clustertrust.ListenReversePrivate(clustertrust.ReverseDialPath(base))
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	connections := make(map[net.Conn]bool)
	var wg sync.WaitGroup
	track := func(conn net.Conn) {
		mu.Lock()
		connections[conn] = true
		mu.Unlock()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			track(conn)
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer conn.Close()
				deadline, _ := ctx.Deadline()
				if err := conn.SetDeadline(deadline); err != nil {
					t.Error(err)
					return
				}
				reader := bufio.NewReaderSize(conn, clustertrust.ReverseDialMaxHeader)
				header, err := reader.ReadSlice('\n')
				if err != nil {
					t.Errorf("private reverse header: %v", err)
					return
				}
				var request clustertrust.ReverseDialRequest
				if err := json.Unmarshal(header, &request); err != nil {
					t.Error(err)
					return
				}
				if request.CallID == "" {
					t.Error("private reverse call lacks correlation ID")
					return
				}
				if request.Watch {
					if _, err := conn.Write([]byte{0}); err != nil {
						t.Error(err)
						return
					}
					if _, err := io.Copy(io.Discard, reader); err != nil && ctx.Err() == nil {
						t.Errorf("private reverse watch: %v", err)
					}
					return
				}
				mesh.Refresh()
				current, admitted := mesh.PeerServiceIncarnation("pa-peer")
				if request.PeerUUID != "pa-peer" || request.Service != noderec.ServiceEngineManager || !admitted || current != incarnation {
					t.Error("private reverse request lost admitted fixture target")
					return
				}
				upstream, err := net.DialTimeout("tcp", target, time.Second)
				if err != nil {
					t.Error(err)
					return
				}
				track(upstream)
				defer upstream.Close()
				connectionID := reverseConnects.Add(1)
				t.Logf("pa6-propagation private_reverse_connect=%d peer=pa-peer service=em", connectionID)
				if _, err := conn.Write([]byte{0}); err != nil {
					t.Error(err)
					return
				}
				if err := clustertrust.ForwardReverse(ctx, conn, upstream, reader); err != nil && ctx.Err() == nil {
					// The production HTTP client can close a completed or failed
					// response before the opposite copy sees EOF. Receipts and
					// decoded policy, rather than copy termination, prove delivery.
					t.Logf("pa6-propagation reverse stream ended: %v", err)
				}
			}()
		}
	}()
	t.Cleanup(func() {
		if err := listener.Close(); err != nil {
			t.Error(err)
		}
		mu.Lock()
		for conn := range connections {
			// Includes already-closed streams; Close is deliberately idempotent.
			conn.Close()
		}
		mu.Unlock()
		done := make(chan struct{})
		go func() { wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("PA reverse IPC cleanup timed out")
		}
		t.Log("pa6-propagation reverse IPC closed and forwarding drained")
	})
}

func TestPAPausedAdvertisementConvergesWithoutMDNSChange(t *testing.T) {
	pausedAdvertisementPlatform(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	started := time.Now()
	base := shortClusterTempDir(t)
	dir, peerDir := filepath.Join(base, "cluster"), t.TempDir()
	clustertrusttest.WriteKeypair(t, dir, "pa-self")
	clustertrusttest.WriteKeypair(t, peerDir, "pa-peer")
	clustertrusttest.WriteAdmission(t, dir, "pa-cluster", 1)
	clustertrusttest.WriteAdmission(t, peerDir, "pa-cluster", 1)
	task1WriteJSON(t, filepath.Join(dir, "identity.json"), map[string]any{"node_uuid": "pa-self", "created_at": 1})
	task1WriteJSON(t, filepath.Join(peerDir, "identity.json"), map[string]any{"node_uuid": "pa-peer", "created_at": 1})
	writePin := func(destination, source, peer string) {
		t.Helper()
		cert, err := os.ReadFile(filepath.Join(source, "node.crt"))
		if err != nil {
			t.Fatal(err)
		}
		task1WriteJSON(t, filepath.Join(destination, "trusted", peer+".json"), map[string]any{
			"nodeUuid": peer, "certPem": string(cert), "clusterId": "pa-cluster", "admissionEpoch": 1,
		})
		task1WriteJSON(t, filepath.Join(destination, "members.json"), []task1Member{{
			NodeUUID: peer, ClusterID: "pa-cluster", AdmissionEpoch: 1, State: "member",
		}})
	}
	for _, path := range []string{dir, peerDir} {
		if err := os.MkdirAll(filepath.Join(path, "trusted"), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	writePin(dir, peerDir, "pa-peer")
	writePin(peerDir, dir, "pa-self")
	mesh, peerMesh := clustertrust.Open(dir), clustertrust.Open(peerDir)
	incarnation, admitted := mesh.PeerServiceIncarnation("pa-peer")
	if !admitted {
		t.Fatal("PA fixture peer is not currently admitted")
	}
	ports := reserveFixtureServicePorts(t)
	const advertisedHost = "192.0.2.40"
	advertisedAddress := net.JoinHostPort(advertisedHost, strconv.Itoa(ports.engineManagerHTTP))
	// This explicit probe observes documentation-address refusal/timeout only.
	// The scanner's own initial direct attempt remains source-derived evidence.
	probeStarted := time.Now()
	probe, probeErr := (&net.Dialer{Timeout: 2 * time.Second}).DialContext(ctx, "tcp", advertisedAddress)
	if probe != nil {
		if err := probe.Close(); err != nil {
			t.Fatalf("close documentation-address probe: %v", err)
		}
	}
	probeElapsed := time.Since(probeStarted)
	if probeErr == nil {
		t.Fatal("documentation-address probe unexpectedly connected")
	}
	probeOutcome := "refused"
	if !strataContractConnectionRefused(probeErr) {
		var timeout net.Error
		if !errors.As(probeErr, &timeout) || !timeout.Timeout() {
			t.Fatalf("documentation-address probe neither refused nor timed out: %v", probeErr)
		}
		probeOutcome = "timeout"
	}
	if probeElapsed > 3*time.Second {
		t.Fatalf("documentation-address probe exceeded prompt-failure bound: %s", probeElapsed)
	}
	t.Logf("pa6-propagation explicit_direct_probe address=%s outcome=%s elapsed=%s error=%v production_direct_failure=source-derived", advertisedAddress, probeOutcome, probeElapsed, probeErr)
	type receipt struct {
		id             uint64
		status         int
		revision       string
		reverseConnect uint64
	}
	reverseConnects := &atomic.Uint64{}
	var mu sync.Mutex
	policy := nodepolicy.Advertisement{Revision: "5", Roles: map[string]nodepolicy.AdvertisedRole{}}
	status, requestID := http.StatusOK, uint64(0)
	receipts := make(chan receipt, 32)
	resumeEntered := make(chan receipt, 1)
	var resumeGate chan struct{}
	server := &httptest.Server{Listener: listenFixtureLoopback(t), Config: &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		peer, pinned := peerMesh.VerifyClientPin(r)
		if !pinned || peer != "pa-self" || r.Method != http.MethodGet || r.URL.Path != "/v1/models" || r.Host != advertisedAddress {
			t.Error("PA inventory request lost pinned caller, canonical endpoint or descriptor Host")
			http.Error(w, "invalid fixture request", http.StatusForbidden)
			return
		}
		mu.Lock()
		requestID++
		observed := receipt{id: requestID, status: status, revision: policy.Revision, reverseConnect: reverseConnects.Load()}
		snapshot := policy
		gate := resumeGate
		mu.Unlock()
		if gate != nil {
			select {
			case resumeEntered <- observed:
			case <-ctx.Done():
				return
			}
			select {
			case <-gate:
			case <-ctx.Done():
				return
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(observed.status)
		var response any = map[string]any{
			"models": []string{"pa-model"}, "modelsByEngine": map[string][]string{"ollama": {"pa-model"}},
			"loadedByEngine": map[string][]string{"ollama": {"pa-model"}}, "policy": snapshot,
		}
		if observed.status != http.StatusOK {
			response = map[string]string{"error": "fixture refresh failure"}
		}
		if err := json.NewEncoder(w).Encode(response); err != nil {
			t.Errorf("write PA inventory response: %v", err)
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			t.Errorf("flush PA inventory response: %v", err)
			return
		}
		select {
		case receipts <- observed:
		default:
			t.Error("PA response receipt budget exceeded")
		}
	})}}
	server.TLS = peerMesh.ServerTLSConfig()
	server.StartTLS()
	serverPort := server.Listener.Addr().(*net.TCPAddr).Port
	t.Cleanup(func() { server.Close(); assertInteropListenerClosed(t, serverPort) })
	pausedAdvertisementReverse(t, ctx, base, incarnation, server.Listener.Addr().String(), mesh, reverseConnects)
	claim := noderec.PeerServicesClaim{Incarnation: incarnation, Source: "pa6-fixture", Descriptor: noderec.PeerServicesDescriptor{
		HostUUID: "pa-peer", Name: "pa-peer", Addresses: []string{advertisedHost},
		Services: []noderec.PeerService{{Service: noderec.ServiceEngineManager, Port: ports.engineManagerHTTP}},
	}}
	claimJSON := mustMarshal(t, claim)
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	adapter := filepath.Join(testbin.Dir(t, "tests", "subscription-scanner", "subscription-scanner"), testbin.Exe("subscription-scanner"))
	if err := os.WriteFile(adapter, data, 0o700); err != nil {
		t.Fatal(err)
	}
	// Empty private cwd keeps every optional default sibling unresolved.
	args := append([]string{"--scanner-path", adapter, "--cluster-dir", dir}, ports.args()...)
	p := startInteropProcess(t, base, brokerBin, 0, []string{
		"NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + adapter,
		"NVPAIR_TEST_PA_SCANNER=" + scannerBin,
		"NVPAIR_TEST_PA_CLAIM=" + string(claimJSON),
	}, args...)
	interrupt := context.AfterFunc(ctx, func() { p.in.Close() })
	defer interrupt()
	for _, path := range []string{adapter, scannerBin} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		p.paths = append(p.paths, canonical)
	}
	p.ports = []int{ports.nodeInfo, ports.errors, ports.workload, ports.clusterManager, ports.engineManagerHTTP, ports.engineControl}
	t.Logf("pa6-propagation inventory broker=%q adapter=%q scanner=%q model_listener=127.0.0.1:%d direct=192.0.2.40:%d private_reverse=%q", brokerBin, adapter, scannerBin, serverPort, ports.engineManagerHTTP, clustertrust.ReverseDialPath(base))
	var previousReverse uint64
	awaitReceipt := func(after uint64, wantStatus int, revision string, deadline time.Time) receipt {
		t.Helper()
		timer := time.NewTimer(time.Until(deadline))
		defer timer.Stop()
		for {
			select {
			case got := <-receipts:
				if !time.Now().Before(deadline) {
					t.Fatal("PA response receipt deadline exceeded")
				}
				if got.id > after && got.status == wantStatus && got.revision == revision {
					if got.reverseConnect <= previousReverse {
						t.Fatal("PA inventory response lacks a fresh private reverse connection")
					}
					previousReverse = got.reverseConnect
					return got
				}
			case <-timer.C:
				t.Fatalf("PA timed out awaiting new HTTP %d revision %s", wantStatus, revision)
			case <-ctx.Done():
				t.Fatal("PA whole-case deadline exceeded")
			}
		}
	}
	nextID := 0
	readPeer := func(deadline time.Time) availableNode {
		t.Helper()
		if !time.Now().Before(deadline) {
			t.Fatal("PA directory deadline exceeded")
		}
		nextID++
		sendReq(t, p.in, nextID, noderec.MethodGetNodes)
		timer := time.NewTimer(time.Until(deadline))
		defer timer.Stop()
		for {
			select {
			case frame, ok := <-p.msgs:
				if !time.Now().Before(deadline) {
					t.Fatal("PA directory response deadline exceeded")
				}
				if !ok {
					t.Fatal("PA broker stream closed")
				}
				if !idEquals(frame.ID, nextID) {
					continue
				}
				if frame.Error != nil {
					t.Fatalf("PA directory response: %+v", frame.Error)
				}
				var result availableNodesResult
				if err := json.Unmarshal(frame.Result, &result); err != nil {
					t.Fatal(err)
				}
				for _, node := range result.Nodes {
					if node.ID == "pa-peer" {
						return node
					}
				}
				return availableNode{}
			case <-timer.C:
				t.Fatal("PA directory response deadline exceeded")
			case <-ctx.Done():
				t.Fatal("PA whole-case deadline exceeded")
			}
		}
	}
	waitPolicy := func(revision string, paused nodepolicy.PauseOrigin, deadline time.Time) availableNode {
		t.Helper()
		ticker := time.NewTicker(50 * time.Millisecond)
		defer ticker.Stop()
		for {
			node := readPeer(deadline)
			if node.Policy != nil && node.Policy.Revision == revision && node.Policy.PausedBy == paused {
				if !reflect.DeepEqual(node.Models, []string{"pa-model"}) || !reflect.DeepEqual(node.ModelsByEngine, map[string][]string{"ollama": {"pa-model"}}) || !reflect.DeepEqual(node.LoadedByEngine, map[string][]string{"ollama": {"pa-model"}}) {
					t.Fatal("PA policy transition changed model inventory")
				}
				return node
			}
			select {
			case <-ticker.C:
			case <-ctx.Done():
				t.Fatal("PA whole-case deadline exceeded")
			}
		}
	}
	setupDeadline := started.Add(30 * time.Second)
	first := awaitReceipt(0, http.StatusOK, "5", setupDeadline)
	waitPolicy("5", "", setupDeadline)
	p.inventory = &serviceProcessInventory{paths: p.paths, known: make(map[int32]serviceProcessIdentity)}
	owned, err := p.inventory.scan()
	if err != nil || len(owned) != len(p.paths) {
		t.Fatalf("PA process inventory: owned=%+v err=%v want paths=%v", owned, err, p.paths)
	}
	for _, child := range owned {
		if child.Path == p.paths[len(p.paths)-1] {
			p.workerPID = child.PID
		}
	}
	if p.workerPID == 0 {
		t.Fatal("real PA scanner missing from fixture inventory")
	}
	if !time.Now().Before(setupDeadline) {
		t.Fatal("PA routing/setup exceeded 30 seconds")
	}
	t.Logf("pa6-propagation owned_processes=%+v", owned)
	t.Logf("pa6-propagation available5 request=%d private_reverse_connect=%d route_setup=%s reverse_delivery=pinned_HTTP_and_policy direct_failure=source-derived", first.id, first.reverseConnect, time.Since(started))
	wholeDeadline, _ := ctx.Deadline()
	mu.Lock()
	policy.Revision, policy.PausedBy = "6", nodepolicy.PausedLocally
	barrier := requestID
	mu.Unlock()
	paused := awaitReceipt(barrier, http.StatusOK, "6", wholeDeadline)
	waitPolicy("6", nodepolicy.PausedLocally, wholeDeadline)
	t.Logf("pa6-propagation pause6 request=%d private_reverse_connect=%d descriptor=unchanged", paused.id, paused.reverseConnect)
	mu.Lock()
	status, barrier = http.StatusInternalServerError, requestID
	mu.Unlock()
	failed := awaitReceipt(barrier, http.StatusInternalServerError, "6", wholeDeadline)
	// The next real periodic request proves the failed sweep has returned:
	// production sweeps are serialized. Hold its successful resume response
	// until a fresh broker read verifies the retained last-good pause.
	mu.Lock()
	status, policy.Revision, policy.PausedBy, barrier = http.StatusOK, "7", "", requestID
	resumeGate = make(chan struct{})
	gate := resumeGate
	mu.Unlock()
	select {
	case next := <-resumeEntered:
		if next.id <= failed.id || next.status != http.StatusOK || next.revision != "7" || next.reverseConnect <= failed.reverseConnect {
			t.Fatal("PA resume barrier did not follow the acknowledged failed sweep")
		}
	case <-ctx.Done():
		t.Fatal("PA timed out awaiting next periodic resume request")
	}
	retained := readPeer(wholeDeadline)
	if retained.Policy == nil || retained.Policy.Revision != "6" || retained.Policy.PausedBy != nodepolicy.PausedLocally || !reflect.DeepEqual(retained.Models, []string{"pa-model"}) || !reflect.DeepEqual(retained.ModelsByEngine, map[string][]string{"ollama": {"pa-model"}}) || !reflect.DeepEqual(retained.LoadedByEngine, map[string][]string{"ollama": {"pa-model"}}) {
		t.Fatal("PA acknowledged failed refresh did not retain last-good pause6")
	}
	t.Logf("pa6-propagation acknowledged500 request=%d private_reverse_connect=%d retained=pause6", failed.id, failed.reverseConnect)
	close(gate)
	resumed := awaitReceipt(barrier, http.StatusOK, "7", wholeDeadline)
	waitPolicy("7", "", wholeDeadline)
	t.Logf("pa6-propagation resume7 request=%d private_reverse_connect=%d descriptor=unchanged elapsed=%s", resumed.id, resumed.reverseConnect, time.Since(started))
	p.cleanup()
}
