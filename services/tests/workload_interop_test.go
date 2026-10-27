// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Cross-process tests for the nvpair-ui-broker <-> nvpair-workload-manager interop.
//
// Two paths are exercised end-to-end with the real binaries TestMain builds:
//
//   - Inbound (peer -> manager -> broker -> client): a peer's workload
//     event POSTed to the manager's inter-node HTTP endpoint surfaces on the
//     broker's opt-in workloads:* client stream as workloads:upsert.
//   - Outbound (proxy -> broker -> manager -> peer): a real inference
//     request through the supervised ollama-proxy produces a workload event
//     that the broker stamps with the local node id and the manager
//     broadcasts to a peer supplied over discovery:nodes (a stub mTLS server).
//
// Both paths run over cluster mTLS, because that is the only personality the
// inter-node interface has: each test puts the node under test and its stub peer
// into a synthetic cross-pinned cluster (newInterNodeCluster). The non-member case
// — nothing served, nothing broadcast — is covered in cluster_data_plane_test.go.
package tests

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
)

const workloadMgrService = "_nvpair-workload-manager._tcp"

// This test-only stdio adapter supplies the real worker's peer directory through
// discovery:nodes. It preserves the broker's --port argument, so the worker's
// listener agrees with the broker's discovery registration. Worker output is
// forwarded byte-for-byte. The privacy case additionally injects test-only
// worker notifications upstream of the broker's projection, after observing a
// real measured terminal on the existing input seam. Dispatch before TestMain
// so broker worker arguments are not parsed as testing flags and the adapter
// does not build another binary bundle.
func init() {
	if bridge := os.Getenv("NVPAIR_TEST_WORKLOAD_BRIDGE"); bridge != "" && bridge == os.Args[0] {
		if err := runWorkloadInteropBridge(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
}

func runWorkloadInteropBridge() error {
	cmd := exec.Command(os.Getenv("NVPAIR_TEST_WORKLOAD_BINARY"), os.Args[1:]...)
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "fixture_worker_started pid=%d\n", cmd.Process.Pid)
	var outputMu sync.Mutex
	emit := func(raw []byte) error {
		outputMu.Lock()
		defer outputMu.Unlock()
		line := append(append([]byte(nil), raw...), '\n')
		n, err := os.Stdout.Write(line)
		if err == nil && n != len(line) {
			return io.ErrShortWrite
		}
		return err
	}
	outputDone := make(chan error, 1)
	go func() {
		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 64*1024), jsonrpc.WorkerFrameBytes)
		for scanner.Scan() {
			if err := emit(scanner.Bytes()); err != nil {
				outputDone <- err
				return
			}
		}
		outputDone <- scanner.Err()
	}()
	wait := make(chan error, 1)
	go func() {
		outputErr := <-outputDone
		waitErr := cmd.Wait()
		if outputErr != nil {
			wait <- outputErr
		} else {
			wait <- waitErr
		}
	}()
	forward := make(chan error, 1)
	go func() {
		encoder := json.NewEncoder(stdin)
		directory := json.RawMessage(os.Getenv("NVPAIR_TEST_WORKLOAD_NODES"))
		if !json.Valid(directory) {
			forward <- fmt.Errorf("invalid fixture discovery directory")
			return
		}
		if err := encoder.Encode(jsonrpc.Message{JSONRPC: "2.0", Method: noderec.NotifyNodes, Params: directory}); err != nil {
			forward <- err
			return
		}
		decoder := json.NewDecoder(os.Stdin)
		injected := false
		for {
			var frame jsonrpc.Message
			if err := decoder.Decode(&frame); err != nil {
				forward <- err
				return
			}
			if frame.Method == noderec.NotifyNodes {
				frame.Params = directory
			}
			if err := encoder.Encode(frame); err != nil {
				forward <- err
				return
			}
			if !injected && os.Getenv("NVPAIR_TEST_WORKLOAD_PRIVACY") == "1" && frame.Method == "workload:completed" {
				var completed workloadBroadcastFrame
				if err := json.Unmarshal(frame.Params, &completed.Params); err != nil {
					forward <- err
					return
				}
				if completed.Params.WorkloadInfo.Model != workloadBroadcastModel {
					continue
				}
				frames, err := workloadPrivacyInjectionFrames(completed)
				if err != nil {
					forward <- err
					return
				}
				for _, injectedFrame := range frames {
					raw, err := json.Marshal(injectedFrame)
					if err == nil {
						err = emit(raw)
					}
					if err != nil {
						forward <- err
						return
					}
				}
				injected = true
			}
		}
	}()
	select {
	case err := <-wait:
		return fmt.Errorf("workload worker exited before broker EOF: %v", err)
	case err := <-forward:
		closeErr := stdin.Close()
		select {
		case waitErr := <-wait:
			if waitErr != nil {
				return fmt.Errorf("workload worker exit: %w", waitErr)
			}
		case <-time.After(5 * time.Second):
			if killErr := cmd.Process.Kill(); killErr != nil {
				return fmt.Errorf("kill workload worker: %w", killErr)
			}
			select {
			case <-wait:
			case <-time.After(2 * time.Second):
				return fmt.Errorf("workload worker did not exit after kill")
			}
			return fmt.Errorf("workload worker required forced cleanup")
		}
		if err != io.EOF {
			return fmt.Errorf("forward broker input: %w", err)
		}
		if closeErr != nil {
			return closeErr
		}
	}
	fmt.Fprintf(os.Stderr, "fixture_worker_exited pid=%d exit=0\n", cmd.Process.Pid)
	return nil
}

type interopProcess struct {
	cmd       *exec.Cmd
	in        io.WriteCloser
	msgs      <-chan jsonrpc.Message
	output    serviceOutput
	done      chan error
	stopOnce  sync.Once
	ports     []int
	paths     []string
	inventory *serviceProcessInventory
	workerPID int32
	cleanup   func()
}

func startInteropProcess(t *testing.T, base, binary string, port int, extraEnv []string, args ...string) *interopProcess {
	t.Helper()
	canonical, err := canonicalServicePath(binary)
	if err != nil {
		t.Fatal(err)
	}
	p := &interopProcess{cmd: exec.Command(binary, args...), done: make(chan error, 1), paths: []string{canonical}}
	if binary == brokerBin {
		p.cmd, err = fixtureBrokerCommand(binary, args...)
		if err != nil {
			t.Fatalf("broker fixture launch: %v", err)
		}
	}
	if port != 0 {
		p.ports = []int{port}
	}
	p.cmd.Dir = base
	p.cmd.Env = append(os.Environ(), "HOME="+base, "XDG_CONFIG_HOME="+base, "APPDATA="+base, "LOCALAPPDATA="+base, "USERPROFILE="+base)
	if binary == brokerBin || binary == engineMgrBin {
		p.cmd.Env = privateEngineEnv(t, base)
	}
	p.cmd.Env = append(p.cmd.Env, extraEnv...)
	p.in, err = p.cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	out, err := p.cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	p.cmd.Stderr = &p.output
	p.cmd.WaitDelay = 2 * time.Second
	if err := p.cmd.Start(); err != nil {
		t.Fatal(err)
	}
	frames := make(chan jsonrpc.Message, 256)
	p.msgs = frames
	readDone := make(chan error, 1)
	discard := make(chan struct{})
	go func() {
		defer close(frames)
		codec := jsonrpc.NewCodecMaxFrame(struct {
			io.Reader
			io.Writer
		}{out, io.Discard}, jsonrpc.WorkerFrameBytes)
		for {
			frame, err := codec.Read()
			if err != nil {
				readDone <- err
				return
			}
			select {
			case frames <- *frame:
			case <-discard:
			}
		}
	}()
	go func() {
		readErr := <-readDone
		waitErr := p.cmd.Wait()
		if readErr != io.EOF {
			p.done <- fmt.Errorf("stdout: %v; exit: %v", readErr, waitErr)
			return
		}
		p.done <- waitErr
	}()
	t.Logf("fixture_started pid=%d binary=%q port=%d", p.cmd.Process.Pid, binary, port)
	p.cleanup = func() {
		p.stopOnce.Do(func() {
			close(discard)
			if err := p.in.Close(); err != nil {
				t.Errorf("close fixture stdin: %v", err)
			}
			select {
			case err := <-p.done:
				if err != nil {
					t.Errorf("fixture exit: %v; stderr: %s", err, p.output.String())
				}
			case <-time.After(10 * time.Second):
				t.Error("fixture required forced cleanup")
				if err := p.cmd.Process.Kill(); err != nil {
					t.Errorf("kill fixture: %v", err)
				}
				if err := out.Close(); err != nil {
					t.Errorf("close fixture stdout: %v", err)
				}
				select {
				case <-p.done:
				case <-time.After(3 * time.Second):
					t.Error("fixture did not exit after kill")
				}
			}
			if p.inventory == nil {
				p.inventory = &serviceProcessInventory{paths: p.paths, known: make(map[int32]serviceProcessIdentity)}
			}
			owned, err := p.inventory.scan()
			if err != nil || len(owned) != 0 {
				t.Errorf("fixture process cleanup: owned=%+v err=%v", owned, err)
			}
			if p.workerPID != 0 && !strings.Contains(p.output.String(), fmt.Sprintf("fixture_worker_exited pid=%d exit=0", p.workerPID)) {
				t.Errorf("real workload child %d did not report a clean exit", p.workerPID)
			}
			for _, port := range p.ports {
				assertInteropListenerClosed(t, port)
			}
			t.Logf("fixture_cleanup pid=%d owned_processes=%+v listener_ports=%v stderr=%s", p.cmd.Process.Pid, owned, p.ports, p.output.String())
		})
	}
	t.Cleanup(p.cleanup)
	return p
}

func assertInteropListenerClosed(t *testing.T, port int) {
	t.Helper()
	conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), time.Second)
	if err == nil {
		if closeErr := conn.Close(); closeErr != nil {
			t.Errorf("close listener probe: %v", closeErr)
		}
		t.Errorf("fixture listener %d survived cleanup", port)
	}
	t.Logf("fixture_listener_cleanup port=%d closed=%t", port, err != nil)
}

// wlInfo is the subset of the Workload object the interop tests assert on.
type wlInfo struct {
	ID             string `json:"id"`
	Model          string `json:"model"`
	Engine         string `json:"engine"`
	State          string `json:"state"`
	OriginatedFrom string `json:"originatedFrom"`
}

type wlParams struct {
	WorkloadInfo wlInfo `json:"workloadInfo"`
}

// startBrokerProc starts the broker as a NON-MEMBER: it pins an empty cluster dir
// so the broker cannot pick up a real cluster identity from this machine's user
// config. The inter-node data plane is cluster mTLS only, so workers started this
// way serve and broadcast nothing on it — which is exactly what the non-member
// assertions in cluster_data_plane_test.go rely on. Tests that need a working data
// plane use startBrokerProcInCluster.
func startBrokerProc(t *testing.T, args ...string) (io.WriteCloser, <-chan jsonrpc.Message, func()) {
	t.Helper()
	return startBrokerProcInCluster(t, t.TempDir(), args...)
}

// startBrokerProcInCluster starts the broker with clusterDir as its cluster
// identity, so its cluster-scoped workers come up as live members of whatever
// cluster that directory describes (see newInterNodeCluster).
func startBrokerProcInCluster(t *testing.T, clusterDir string, args ...string) (io.WriteCloser, <-chan jsonrpc.Message, func()) {
	t.Helper()
	return startBrokerProcWithPorts(t, clusterDir, reserveFixtureServicePorts(t), args...)
}

func startBrokerProcWithPorts(t *testing.T, clusterDir string, ports fixtureServicePorts, args ...string) (io.WriteCloser, <-chan jsonrpc.Message, func()) {
	t.Helper()
	stdin, msgs, stderr, cleanup := startBrokerWithPorts(t, t.TempDir(), clusterDir, nil, ports, args...)
	go func() {
		for line := range stderr {
			fmt.Fprintln(os.Stderr, line)
		}
	}()
	return stdin, msgs, cleanup
}

// writeRawFrame writes a raw JSON-RPC frame (newline-terminated) to the
// broker. (The package's other writeFrame takes a typed brokerFrame.)
func writeRawFrame(t *testing.T, w io.Writer, frame string) {
	t.Helper()
	if _, err := w.Write([]byte(frame + "\n")); err != nil {
		t.Fatalf("write frame: %v", err)
	}
}

// TestWorkloadManagerInboundRelay drives the inbound path: a peer-origin
// workload event POSTed to the supervised manager's inter-node endpoint
// must surface on the broker's workloads:* client stream as a
// workloads:upsert, but only after the client has subscribed.
func TestWorkloadManagerInboundRelay(t *testing.T) {
	t.Run("fixture isolation controls", testWorkloadFixtureIsolation)
	t.Run("peer event is relayed", testWorkloadManagerInboundRelay)
}

func testWorkloadManagerInboundRelay(t *testing.T) {
	ports := reserveFixtureServicePorts(t)
	endpoint, err := ports.workloadEventsURL()
	if err != nil {
		t.Fatal(err)
	}
	fx := newInterNodeCluster(t)
	stdin, msgs, cleanup := startBrokerProcWithPorts(t, fx.nodeDir, ports,
		"--scanner-path", scannerBin,
		"--workload-manager-path", workloadMgrBin,
	)
	t.Cleanup(cleanup)

	waitForMethod(t, msgs, "app:ready", 10*time.Second)
	t.Log("broker ready")

	// Subscribe first — the stream is opt-in and there's no baseline.
	writeRawFrame(t, stdin, `{"jsonrpc":"2.0","id":1,"method":"workloads:subscribe"}`)
	ack := waitForResponse(t, msgs, 5*time.Second)
	var sr subscriptionResult
	if err := json.Unmarshal(ack.Result, &sr); err != nil || !sr.Subscribed {
		t.Fatalf("workloads:subscribe ack = %s (err %v), want subscribed:true", ack.Result, err)
	}
	t.Log("subscribed to workloads stream")

	// POST a peer workload to the manager's inter-node endpoint over cluster mTLS.
	// The manager binds its planned port in a goroutine, so retry until it accepts.
	const peerEvent = `{"jsonrpc":"2.0","method":"workload:started","params":{"workloadInfo":{"id":"peer-wl-1","model":"llama-3-70b","engine":"trt-llm","state":"running","originatedFrom":"peer-node-A","createdAt":1716998400000,"startedAt":1716998400000,"completedAt":null,"error":null,"requesterId":null}}}`
	postPeerEvent(t, fx.clientAsPeer(t), endpoint, peerEvent, 15*time.Second)
	t.Log("peer workload accepted by manager")

	// The manager translates the peer lifecycle event into workloads:upsert
	// on its stdout; the broker relays it to us.
	upsert := waitForWorkloadEvent(t, msgs, "workloads:upsert", "peer-wl-1", 10*time.Second)
	if upsert.WorkloadInfo.OriginatedFrom != "peer-node-A" {
		t.Errorf("relayed upsert originatedFrom = %q, want %q", upsert.WorkloadInfo.OriginatedFrom, "peer-node-A")
	}
	if upsert.WorkloadInfo.State != "running" {
		t.Errorf("relayed upsert state = %q, want %q", upsert.WorkloadInfo.State, "running")
	}
	t.Log("inbound OK: peer workload relayed to client as workloads:upsert")
}

// TestWorkloadOutOfOrderSuppressed drives the reordering fix end-to-end: a
// peer's terminal (failed) event arriving BEFORE the stale running event for
// the same workload must leave the client's view failed — the late running is
// dropped by the broker's store and never re-emitted — and workloads:get-initial
// reports the authoritative terminal state.
func TestWorkloadOutOfOrderSuppressed(t *testing.T) {
	ports := reserveFixtureServicePorts(t)
	endpoint, err := ports.workloadEventsURL()
	if err != nil {
		t.Fatal(err)
	}
	fx := newInterNodeCluster(t)
	stdin, msgs, cleanup := startBrokerProcWithPorts(t, fx.nodeDir, ports,
		"--scanner-path", scannerBin,
		"--workload-manager-path", workloadMgrBin,
	)
	t.Cleanup(cleanup)

	waitForMethod(t, msgs, "app:ready", 10*time.Second)

	writeRawFrame(t, stdin, `{"jsonrpc":"2.0","id":1,"method":"workloads:subscribe"}`)
	ack := waitForResponse(t, msgs, 5*time.Second)
	var sr subscriptionResult
	if err := json.Unmarshal(ack.Result, &sr); err != nil || !sr.Subscribed {
		t.Fatalf("subscribe ack = %s (err %v), want subscribed:true", ack.Result, err)
	}

	peer := fx.clientAsPeer(t)
	// Same (originatedFrom, id, createdAt) — one workload, two lifecycle events,
	// delivered terminal-first (the inversion the network can produce).
	const failedEvent = `{"jsonrpc":"2.0","method":"workload:errored","params":{"workloadInfo":{"id":"ooo-1","model":"m","engine":"ollama","state":"failed","originatedFrom":"peer-ooo","scheduledOn":"peer-ooo","createdAt":1716998400000,"startedAt":1716998400000,"completedAt":1716998400500,"error":"upstream returned HTTP 400","requesterId":null}}}`
	const runningEvent = `{"jsonrpc":"2.0","method":"workload:started","params":{"workloadInfo":{"id":"ooo-1","model":"m","engine":"ollama","state":"running","originatedFrom":"peer-ooo","scheduledOn":"peer-ooo","createdAt":1716998400000,"startedAt":1716998400000,"completedAt":null,"error":null,"requesterId":null}}}`

	postPeerEvent(t, peer, endpoint, failedEvent, 15*time.Second)
	failed := waitForWorkloadEvent(t, msgs, "workloads:upsert", "ooo-1", 10*time.Second)
	if failed.WorkloadInfo.State != "failed" {
		t.Fatalf("first client upsert state = %q, want failed", failed.WorkloadInfo.State)
	}

	// The stale running arrives after the terminal. The store must reject it,
	// so the client never sees a running upsert for ooo-1.
	postPeerEvent(t, peer, endpoint, runningEvent, 15*time.Second)
	deadline := time.After(3 * time.Second)
	draining := true
	for draining {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatal("client stream closed unexpectedly")
			}
			if msg.Method != "workloads:upsert" {
				continue
			}
			var p wlParams
			if json.Unmarshal(msg.Params, &p) != nil {
				continue
			}
			if p.WorkloadInfo.ID == "ooo-1" && p.WorkloadInfo.State == "running" {
				t.Fatal("stale running after failed was re-emitted to the client — reordering not suppressed")
			}
		case <-deadline:
			draining = false
		}
	}

	// The authoritative baseline reports the terminal state.
	writeRawFrame(t, stdin, `{"jsonrpc":"2.0","id":2,"method":"workloads:get-initial"}`)
	resp := waitForResponse(t, msgs, 5*time.Second)
	var res struct {
		Workloads []wlInfo `json:"workloads"`
	}
	if err := json.Unmarshal(resp.Result, &res); err != nil {
		t.Fatalf("get-initial result: %v\nraw: %s", err, resp.Result)
	}
	found := false
	for _, w := range res.Workloads {
		if w.ID == "ooo-1" {
			found = true
			if w.State != "failed" {
				t.Fatalf("get-initial ooo-1 state = %q, want failed", w.State)
			}
		}
	}
	if !found {
		t.Fatal("get-initial did not include ooo-1")
	}
	t.Log("reordering OK: stale running suppressed; get-initial reports failed")
}

// TestWorkloadFailedOnNodeLoss covers the node-offline cleanup: a peer's
// running workload that the broker is tracking must transition to "failed"
// once that peer drops out of discovery, instead of lingering as a stale "in
// progress" line forever. The workload is marked failed, not removed — it
// stays in the client's history. It drives the full path: a peer workload
// POSTed to the manager's inter-node endpoint surfaces as a tracked
// workloads:upsert; the peer's mDNS record is then withdrawn, the scanner
// evicts it (miss threshold + liveness probe), and the broker synthesizes the
// terminal workloads:upsert(state=failed).
func TestWorkloadFailedOnNodeLoss(t *testing.T) {
	// The peer's mDNS instance name and its stable HostUUID differ on purpose:
	// workloads are stamped/keyed by HostUUID, and the node-loss sweep
	// must match on that UUID, not the display name. Keying by name would leave
	// a UUID-stamped workload running after the (name-distinct) node departs.
	const peerNode = "nodeloss-peer"
	const peerUUID = "nodeloss-peer-uuid"

	ports := reserveFixtureServicePorts(t)
	endpoint, err := ports.workloadEventsURL()
	if err != nil {
		t.Fatal(err)
	}
	fx := newInterNodeCluster(t)
	stdin, msgs, cleanup := startBrokerProcWithPorts(t, fx.nodeDir, ports,
		"--scanner-path", scannerBin,
		"--workload-manager-path", workloadMgrBin,
		"--scanner-mdns=true", "--scanner-mdns-service", fx.namespace.Service,
	)
	t.Cleanup(cleanup)

	waitForMethod(t, msgs, "app:ready", 10*time.Second)
	t.Log("broker ready")

	// Subscribe to the (opt-in) workloads stream before anything is tracked.
	writeRawFrame(t, stdin, `{"jsonrpc":"2.0","id":1,"method":"workloads:subscribe"}`)
	ack := waitForResponse(t, msgs, 5*time.Second)
	var sr subscriptionResult
	if err := json.Unmarshal(ack.Result, &sr); err != nil || !sr.Subscribed {
		t.Fatalf("workloads:subscribe ack = %s (err %v), want subscribed:true", ack.Result, err)
	}
	t.Log("subscribed to workloads stream")

	// Advertise the peer as an identity-only _nvpair-node record (no service
	// ports), so once its advertisement is withdrawn the scanner's liveness
	// probe has nothing to dial and evicts it cleanly. A distinct uuid keeps
	// it from colliding with this host's own record.
	peerTXT := []string{"v=1", "uuid=" + peerUUID, "ip=127.0.0.1"}
	stopAdv := startMDNSAdvertiser(t, mdnsAdvertiserFixture{
		Instance: peerNode, Service: fx.namespace.Service, Domain: fx.namespace.Domain, Port: 14999, TXT: peerTXT, Interface: mdnsFixtureInterface(t).Name,
	})
	t.Logf("advertising peer %q as %s.%s", peerNode, fx.namespace.Service, fx.namespace.Domain)

	// The peer must be in the broker's directory before its removal can
	// trigger a sweep, so wait until discovery has folded it in.
	pollNodeListed(t, stdin, msgs, peerNode, 1000, 30*time.Second, nil)
	t.Logf("peer %q discovered", peerNode)

	// A peer-origin running workload (scheduled on that same peer) POSTed to
	// the manager's inter-node endpoint. The broker relays it to us as a
	// tracked workloads:upsert. Stamped with the peer's HostUUID (not its
	// name), as a real proxy does.
	peerEvent := fmt.Sprintf(`{"jsonrpc":"2.0","method":"workload:started","params":{"workloadInfo":{"id":"wl-nodeloss-1","model":"llama-3-8b","engine":"ollama","state":"running","originatedFrom":%q,"scheduledOn":%q,"createdAt":1716998400000,"startedAt":1716998400000,"completedAt":null,"error":null,"requesterId":null}}}`, peerUUID, peerUUID)
	postPeerEvent(t, fx.clientAsPeer(t), endpoint, peerEvent, 15*time.Second)

	running := waitForWorkloadEvent(t, msgs, "workloads:upsert", "wl-nodeloss-1", 10*time.Second)
	if running.WorkloadInfo.State != "running" {
		t.Fatalf("initial upsert state = %q, want running", running.WorkloadInfo.State)
	}
	t.Log("peer workload tracked as running")

	// Pull the peer off the network. Keep the production scan cadence, browse
	// budget, miss threshold and failed-liveness-probe requirement. Once the
	// peer is evicted, the broker marks the workload it can no longer hear about.
	// The two-minute acceptance bound assumes healthy receive: self is seen on
	// every scan, so each post-departure scan counts toward the 12 misses at 5s
	// cadence (about 60s), leaving room for the final browse, probe and delivery.
	// Intermittent receive can renew empty-scan grace and exceed this estimate.
	stopAdv()
	t.Log("peer advertisement withdrawn; awaiting node eviction + workload failure")

	failed := waitForWorkloadEvent(t, msgs, "workloads:upsert", "wl-nodeloss-1", 2*time.Minute)
	if failed.WorkloadInfo.State != "failed" {
		t.Fatalf("post-eviction upsert state = %q, want failed", failed.WorkloadInfo.State)
	}
	if failed.WorkloadInfo.OriginatedFrom != peerUUID {
		t.Errorf("failed upsert originatedFrom = %q, want %q", failed.WorkloadInfo.OriginatedFrom, peerUUID)
	}
	t.Log("node-loss OK: UUID-stamped workload marked failed after the (name-distinct) peer went offline")
}

// TestWorkloadManagerOutboundBroadcast drives the full outbound chain: a
// real inference request through the supervised ollama-proxy produces a
// workload event that the broker stamps and forwards to the manager, which
// broadcasts it to a discovered peer.
func TestWorkloadManagerOutboundBroadcast(t *testing.T) {
	captureWorkloadBroadcast(t)
}

// The allowlist is deliberately test-owned: decoding into the product's DTO
// would silently grow this privacy contract whenever a private field was added.
type workloadBroadcastFrame struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  struct {
		WorkloadInfo struct {
			ID             string  `json:"id"`
			Model          string  `json:"model"`
			Engine         string  `json:"engine"`
			RunID          string  `json:"runId"`
			Seq            int64   `json:"seq"`
			State          string  `json:"state"`
			OriginatedFrom string  `json:"originatedFrom"`
			ScheduledOn    string  `json:"scheduledOn"`
			CreatedAt      int64   `json:"createdAt"`
			StartedAt      *int64  `json:"startedAt"`
			CompletedAt    *int64  `json:"completedAt"`
			Error          *string `json:"error"`
			RequesterID    *string `json:"requesterId"`
			RequestedModel string  `json:"requestedModel"`
		} `json:"workloadInfo"`
		Resync *bool `json:"resync"`
	} `json:"params"`
}

func TestWorkloadBroadcastCarriesNoRoutingFields(t *testing.T) {
	frames := captureWorkloadBroadcast(t)
	privateValues := workloadBroadcastPrivateValues()
	privateValues["message"] = workloadBroadcastMessage
	privateValues["response"] = workloadBroadcastResponse
	// Search decoded string values, including substrings in otherwise approved
	// fields such as requesterId/error. These exact synthetic markers are private;
	// the legitimate model and node identifiers remain unrestricted.
	var checkStrings func(any, int)
	checkStrings = func(value any, frameIndex int) {
		t.Helper()
		switch value := value.(type) {
		case string:
			for name, marker := range privateValues {
				if strings.Contains(value, marker) {
					t.Fatalf("peer frame %d leaks the private %s marker in a decoded string", frameIndex, name)
				}
			}
		case []any:
			for _, element := range value {
				checkStrings(element, frameIndex)
			}
		case map[string]any:
			for _, element := range value {
				checkStrings(element, frameIndex)
			}
		}
	}
	for i, raw := range frames {
		var decoded any
		if err := json.Unmarshal(raw, &decoded); err != nil {
			t.Fatalf("decode peer frame %d for private-value checks: %v", i, err)
		}
		checkStrings(decoded, i)
		var frame workloadBroadcastFrame
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&frame); err != nil {
			t.Fatalf("peer frame %d violates the workload envelope/nested field allowlist: %v", i, err)
		}
		if err := decoder.Decode(new(json.RawMessage)); err != io.EOF {
			t.Fatalf("peer frame %d has trailing data: %v", i, err)
		}
		if frame.JSONRPC != "2.0" {
			t.Fatalf("peer frame %d jsonrpc=%q, want 2.0", i, frame.JSONRPC)
		}
		if frame.Params.WorkloadInfo.ID == workloadBroadcastMalformedID {
			t.Fatalf("peer frame %d carries the malformed injection", i)
		}
		switch frame.Method {
		case "workload:submitted", "workload:started", "workload:completed", "workload:errored":
		default:
			t.Fatalf("peer frame %d carries non-workload/private method %q", i, frame.Method)
		}
	}
	t.Logf("checked the complete peer capture: %d workload frames", len(frames))
}

const (
	workloadBroadcastModel        = "privacy-routing-request-model"
	workloadBroadcastReadyModel   = "privacy-readiness-model"
	workloadBroadcastNode         = "privacy-upstream-node"
	workloadBroadcastMessage      = "privacy-synthetic-message"
	workloadBroadcastResponse     = "private-response-marker-7c24"
	workloadBroadcastControlID    = "privacy-projection-control"
	workloadBroadcastControlModel = "privacy-projection-model"
	workloadBroadcastMalformedID  = "privacy-malformed-control"
)

func workloadBroadcastPrivateValues() map[string]string {
	return map[string]string{
		"error": "private-error-marker-83b1", "credential": "private-credential-marker-a729",
		"pin": "private-pin-marker-4902", "key": "private-key-marker-d6e8", "unknown": "private-unknown-marker-f315",
	}
}

// Inject only on the test worker's stdout, before the broker's real projection.
// The accepted control has the measured origin/run and a distinct workload ID;
// its replay proves that these inputs were consumed and stored by the broker.
func workloadPrivacyInjectionFrames(completed workloadBroadcastFrame) ([]jsonrpc.Message, error) {
	info := completed.Params.WorkloadInfo
	info.ID, info.Model = workloadBroadcastControlID, workloadBroadcastControlModel
	raw, err := json.Marshal(info)
	if err != nil {
		return nil, err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, err
	}
	private, err := json.Marshal(workloadBroadcastPrivateValues())
	if err != nil {
		return nil, err
	}
	fields["privateRoutingHistory"] = private
	accepted, err := json.Marshal(map[string]any{"workloadInfo": fields, "privateEnvelope": private})
	if err != nil {
		return nil, err
	}
	fields["id"], err = json.Marshal(workloadBroadcastMalformedID)
	if err != nil {
		return nil, err
	}
	fields["state"] = private // A required scalar with the wrong type must be rejected.
	malformed, err := json.Marshal(map[string]any{"workloadInfo": fields})
	if err != nil {
		return nil, err
	}
	return []jsonrpc.Message{
		{JSONRPC: "2.0", Method: "routing/request", Params: private},
		{JSONRPC: "2.0", Method: "privacy:unknown-worker-method", Params: private},
		{JSONRPC: "2.0", Method: "workloads:upsert", Params: malformed},
		{JSONRPC: "2.0", Method: "workloads:upsert", Params: accepted},
	}, nil
}

// captureWorkloadBroadcast keeps the original real proxy -> broker -> manager
// fixture, but first proves peer readiness with a distinct model. The measured
// inference is sent once, after that peer has delivered a terminal frame.
func captureWorkloadBroadcast(t *testing.T) []json.RawMessage {
	t.Helper()
	var measuredRequests atomic.Int32
	ollama := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		var request struct {
			Model    string            `json:"model"`
			Private  map[string]string `json:"private"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode synthetic Ollama request: %v", err)
			http.Error(w, "invalid fixture request", http.StatusBadRequest)
			return
		}
		if (request.Model != workloadBroadcastModel && request.Model != workloadBroadcastReadyModel) ||
			len(request.Messages) != 1 || request.Messages[0].Role != "user" || request.Messages[0].Content != workloadBroadcastMessage {
			t.Error("fake Ollama received unexpected model/messages")
			http.Error(w, "unexpected fixture request", http.StatusBadRequest)
			return
		}
		for name, marker := range workloadBroadcastPrivateValues() {
			if request.Private[name] != marker {
				t.Errorf("fake Ollama did not receive the private %s input marker", name)
				http.Error(w, "missing private fixture input", http.StatusBadRequest)
				return
			}
		}
		if request.Model == workloadBroadcastModel {
			measuredRequests.Add(1)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if err := json.NewEncoder(w).Encode(map[string]any{
			"done": true, "message": map[string]string{"role": "assistant", "content": workloadBroadcastResponse},
			"private": workloadBroadcastPrivateValues(),
		}); err != nil {
			t.Errorf("write fake Ollama completion: %v", err)
		}
	}))
	ollamaPort := ollama.Listener.Addr().(*net.TCPAddr).Port
	t.Cleanup(func() {
		ollama.Close()
		assertInteropListenerClosed(t, ollamaPort)
	})

	// Stub peer: a real cluster peer (cross-pinned with the node under test) that
	// records the workload broadcasts the manager fans out to it. It has to be a
	// cluster peer — the inter-node interface is mTLS only, so an unpinned host is
	// never dialed at all. The fixture feeds the public worker discovery seam,
	// avoiding installed peers and multicast registration/teardown dependencies.
	fx := newInterNodeCluster(t)
	received, peerPort, stopPeer := startWorkloadBroadcastPeer(t, fx)
	base := t.TempDir()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	bridge := filepath.Join(testbin.Dir(t, "tests", "workload-bridge", "workload-bridge"), testbin.Exe("workload-bridge"))
	if err := os.WriteFile(bridge, data, 0o700); err != nil {
		t.Fatalf("copy workload bridge to %q (a previous process may still be running): %v", bridge, err)
	}
	// Reuse the registered empty-directory scanner role. The bridge supplies
	// only the pinned peer to the workload worker; no production scanner runs.
	scanner := filepath.Join(testbin.Dir(t, "tests", "subscription-scanner", "subscription-scanner"), testbin.Exe("subscription-scanner"))
	if err := os.WriteFile(scanner, data, 0o700); err != nil {
		t.Fatalf("copy empty scanner to %q: %v", scanner, err)
	}
	scannerPort := temporaryClusterPort(t)
	ports := reserveFixtureServicePorts(t)
	workerPort := ports.workload
	proxyPort := temporaryClusterPort(t)
	// Persist an explicit startup choice in the broker's temporary journal;
	// changing a proxy port later requires an engine-manager we do not need.
	journal := map[string]any{"ollama": map[string]any{
		"explicit": true,
		"snapshot": map[string]any{
			"engine": "ollama", "revision": 1, "appliedRevision": 1, "phase": "idle", "format": "pair-arguments-v1",
			"settings": map[string]any{"serverPort": ollamaPort, "proxyPort": proxyPort, "launchText": ""},
		},
	}}
	if err := os.WriteFile(filepath.Join(filepath.Dir(fx.nodeDir), "engine-settings-operations.json"), mustMarshal(t, journal), 0o600); err != nil {
		t.Fatal(err)
	}
	directory := noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{{
		HostUUID: "stub-wm-peer-uuid", ClusterUUID: fx.peerUUID, IP: "127.0.0.1",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceWorkload: {Port: peerPort}},
	}}}
	args := append([]string{"--scanner-path", scanner, "--cluster-dir", fx.nodeDir}, ports.args()...)
	args = append(args,
		"--proxy-path", proxyBin, "--proxy-engines", "ollama",
		"--workload-manager-path", bridge,
	)
	p := startInteropProcess(t, base, brokerBin, workerPort, []string{
		"OLLAMA_HOST=",
		"NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + scanner,
		"NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(scannerPort),
		"NVPAIR_TEST_WORKLOAD_BRIDGE=" + bridge,
		"NVPAIR_TEST_WORKLOAD_BINARY=" + workloadMgrBin,
		"NVPAIR_TEST_WORKLOAD_NODES=" + string(mustMarshal(t, directory)),
		"NVPAIR_TEST_WORKLOAD_PRIVACY=1",
	}, args...)
	p.paths = append(p.paths, scanner, proxyBin, bridge, workloadMgrBin)
	for i, path := range p.paths {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		p.paths[i] = canonical
	}
	p.inventory = &serviceProcessInventory{paths: p.paths, known: make(map[int32]serviceProcessIdentity)}
	stdin, msgs := p.in, p.msgs

	waitForMethod(t, msgs, "app:ready", 10*time.Second)
	t.Log("broker ready")

	// The proxy reports its bound port asynchronously; poll ollama-proxy:get-status.
	if readyPort := waitProxyReady(t, stdin, msgs, 15*time.Second); readyPort != proxyPort {
		t.Fatalf("proxy port=%d want fixture port=%d", readyPort, proxyPort)
	}
	p.ports = append(p.ports, proxyPort, scannerPort)
	waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(workerPort)), 10*time.Second)
	waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(scannerPort)), 10*time.Second)
	owned, err := p.inventory.scan()
	if err != nil || len(owned) != len(p.paths) {
		t.Fatalf("fixture process inventory: owned=%+v err=%v want paths=%v", owned, err, p.paths)
	}
	var originalWorker, originalBridge, originalProxy serviceProcessIdentity
	for _, child := range owned {
		if child.Path == p.paths[len(p.paths)-1] {
			p.workerPID = child.PID
			originalWorker = child
		}
		if child.Path == p.paths[len(p.paths)-2] {
			originalBridge = child
		}
		if child.Path == p.paths[len(p.paths)-3] {
			originalProxy = child
		}
	}
	if originalWorker.PID == 0 || originalBridge.PID == 0 || originalProxy.PID == 0 {
		t.Fatal("real workload worker, bridge or proxy missing from fixture inventory")
	}
	t.Logf("fixture_inventory owned=%+v", owned)
	t.Logf("proxy ready on port %d", proxyPort)

	// Register the fake Ollama as a manual node and pin it as the target so
	// the proxy forwards there deterministically.
	writeRawFrame(t, stdin, fmt.Sprintf(`{"jsonrpc":"2.0","id":50,"method":"ollama-proxy:node/add-manual","params":{"id":%q,"host":"127.0.0.1","port":%d,"addresses":["127.0.0.1"],"models":[%q,%q]}}`, workloadBroadcastNode, ollamaPort, workloadBroadcastModel, workloadBroadcastReadyModel))
	if reply := waitForResponse(t, msgs, 5*time.Second); reply.Error != nil {
		t.Fatalf("add manual upstream: %+v", reply.Error)
	}
	writeRawFrame(t, stdin, fmt.Sprintf(`{"jsonrpc":"2.0","id":51,"method":"ollama-proxy:node/select","params":{"id":%q}}`, workloadBroadcastNode))
	if reply := waitForResponse(t, msgs, 5*time.Second); reply.Error != nil {
		t.Fatalf("select upstream: %+v", reply.Error)
	}
	t.Log("fake ollama pinned as proxy target")

	// A completed warmup is the discovery/delivery barrier. All frames,
	// including warmups and teardown, remain in the privacy capture.
	client := &http.Client{Timeout: 3 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	fire := func(model string) {
		t.Helper()
		body := mustMarshal(t, map[string]any{
			"model": model, "messages": []map[string]string{{"role": "user", "content": workloadBroadcastMessage}},
			"private": workloadBroadcastPrivateValues(),
		})
		resp, err := client.Post(
			fmt.Sprintf("http://127.0.0.1:%d/api/chat", proxyPort),
			"application/json",
			bytes.NewReader(body),
		)
		if err != nil {
			t.Fatalf("inference through real proxy: %v", err)
		}
		responseBody, readErr := io.ReadAll(resp.Body)
		closeErr := resp.Body.Close()
		if readErr != nil || closeErr != nil {
			t.Fatalf("read inference response: read=%v close=%v", readErr, closeErr)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("inference status=%d want=200", resp.StatusCode)
		}
		var completion struct {
			Done    bool `json:"done"`
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			Private map[string]string `json:"private"`
		}
		if err := json.Unmarshal(responseBody, &completion); err != nil || !completion.Done {
			t.Fatalf("inference completion done=%t err=%v", completion.Done, err)
		}
		if completion.Message.Role != "assistant" || completion.Message.Content != workloadBroadcastResponse {
			t.Fatal("client did not receive the synthetic private response marker")
		}
		for name, marker := range workloadBroadcastPrivateValues() {
			if completion.Private[name] != marker {
				t.Fatalf("client did not receive the private %s output marker", name)
			}
		}
	}
	var capture []json.RawMessage
	oldPeerConnections := make(map[string]bool)
	read := func(observed workloadBroadcastCapture) workloadBroadcastFrame {
		t.Helper()
		raw := observed.Raw
		capture = append(capture, raw)
		var frame workloadBroadcastFrame
		if err := json.Unmarshal(raw, &frame); err != nil {
			t.Fatalf("decode captured peer frame: %v", err)
		}
		return frame
	}
	waitReady := func() {
		t.Helper()
		fire(workloadBroadcastReadyModel)
		deadline := time.NewTimer(30 * time.Second)
		defer deadline.Stop()
		retry := time.NewTicker(2500 * time.Millisecond)
		defer retry.Stop()
		for {
			select {
			case raw := <-received:
				frame := read(raw)
				info := frame.Params.WorkloadInfo
				if frame.Method == "workload:completed" && info.Model == workloadBroadcastReadyModel && info.State == "completed" {
					return
				}
			case <-retry.C:
				fire(workloadBroadcastReadyModel)
			case <-deadline.C:
				t.Fatal("timed out waiting for pinned peer's completed readiness workload")
			}
		}
	}
	waitReady()
	writeRawFrame(t, stdin, `{"jsonrpc":"2.0","id":52,"method":"workloads:subscribe"}`)
	ack := waitForResponse(t, msgs, 5*time.Second)
	var subscribed subscriptionResult
	if err := json.Unmarshal(ack.Result, &subscribed); err != nil || ack.Error != nil || !subscribed.Subscribed {
		t.Fatalf("subscribe to injection positive control: subscribed=%t decode=%v rpc=%v", subscribed.Subscribed, err, ack.Error)
	}
	fire(workloadBroadcastModel)
	deadline := time.NewTimer(15 * time.Second)
	defer deadline.Stop()
	var id, runID string
	var submittedSeq int64
	var terminalSeq int64
	terminal := false
	for !terminal {
		select {
		case raw := <-received:
			oldPeerConnections[raw.RemoteAddr] = true
			frame := read(raw)
			info := frame.Params.WorkloadInfo
			if info.Model != workloadBroadcastModel {
				continue
			}
			if info.OriginatedFrom != fx.nodeUUID || info.Engine != "ollama" || info.ID == "" || info.RunID == "" {
				t.Fatalf("broadcast identity: origin=%q engine=%q workload=%q run=%q", info.OriginatedFrom, info.Engine, info.ID, info.RunID)
			}
			if id == "" {
				if frame.Method != "workload:submitted" || info.State != "queued" || info.Seq <= 0 {
					t.Fatalf("first measured frame method=%q state=%q seq=%d, want submitted/queued", frame.Method, info.State, info.Seq)
				}
				id, runID, submittedSeq = info.ID, info.RunID, info.Seq
			}
			if info.ID != id || info.RunID != runID {
				t.Fatalf("measured workload identity changed: workload=%q run=%q, want %q/%q", info.ID, info.RunID, id, runID)
			}
			if frame.Method == "workload:errored" {
				t.Fatalf("measured workload failed: state=%q", info.State)
			}
			if frame.Method == "workload:completed" {
				if info.State != "completed" || info.Seq <= submittedSeq || info.ScheduledOn != workloadBroadcastNode || info.Error != nil {
					t.Fatalf("terminal broadcast state=%q seq=%d scheduledOn=%q error=%v", info.State, info.Seq, info.ScheduledOn, info.Error)
				}
				terminal = true
				terminalSeq = info.Seq
			}
		case <-deadline.C:
			t.Fatalf("timed out draining measured workload: submitted=%t terminal=%t", id != "", terminal)
		}
	}
	// Prove private routing data was actually produced during the peer capture.
	assertPrivateHistory := func() {
		neEventually(t, "measured request has local private history", 30*time.Second, func() bool {
			writeRawFrame(t, stdin, `{"jsonrpc":"2.0","id":"privacy-history","method":"routing:get-initial","params":{}}`)
			reply := waitForResponse(t, msgs, 5*time.Second)
			if reply.Error != nil {
				if reply.Error.Code == jsonrpc.HistoryBusyCode {
					return false
				}
				t.Fatalf("private history positive control: %+v", reply.Error)
			}
			initial, diagnostic, err := nodepolicy.DecodeRoutingHistoryInitial(reply.Result)
			if err != nil || diagnostic.Count != 0 {
				t.Fatalf("private history positive control: diagnostic=%+v error=%v", diagnostic, err)
			}
			for _, record := range initial.Requests {
				if record.Request.OriginNodeID == fx.nodeUUID && record.Request.ProxyRunID == runID && record.Request.RequestID == id {
					if record.State != "completed" || record.Workload == nil || record.Workload.Engine != "ollama" || record.Workload.ID != id || record.Workload.ProxyRunID != runID {
						t.Fatalf("private history and measured workload disagree: %+v", record)
					}
					return true
				}
			}
			return false
		})
	}
	// This accepted notification follows the malformed and unknown injections
	// on the same serialized worker stdout. Its presence is the consumption
	// barrier; its later peer replay proves the broker retained the projection.
	injectionDeadline := time.NewTimer(10 * time.Second)
	defer injectionDeadline.Stop()
	injected := false
	for !injected {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatal("broker stream closed before injection control")
			}
			if msg.Method != "workloads:upsert" {
				continue
			}
			var control workloadBroadcastFrame
			if err := json.Unmarshal(msg.Params, &control.Params); err != nil {
				t.Fatalf("decode injection control: %v", err)
			}
			info := control.Params.WorkloadInfo
			if info.ID == workloadBroadcastMalformedID {
				t.Fatal("broker accepted malformed worker injection")
			}
			if info.ID == workloadBroadcastControlID {
				if info.OriginatedFrom != fx.nodeUUID || info.RunID != runID || info.Engine != "ollama" || info.Model != workloadBroadcastControlModel || info.State != "completed" {
					t.Fatal("accepted injection control has the wrong origin/run/engine/model/state")
				}
				injected = true
			}
		case raw := <-received:
			oldPeerConnections[raw.RemoteAddr] = true
			read(raw)
		case <-injectionDeadline.C:
			t.Fatal("broker did not accept the projected worker injection control")
		}
	}

	// Read the RPC baseline only after consuming the notification barrier;
	// waitForResponse deliberately discards unrelated queued notifications.
	assertPrivateHistory()
	// Revalidate path/PID/creation time at the signal edge. The broker must
	// replace the killed real worker and its bridge while retaining this proxy.
	if err := p.inventory.kill(originalWorker); err != nil {
		t.Fatalf("restart exact owned workload worker: %v", err)
	}
	awaitReplacement := func() {
		t.Helper()
		limit := time.NewTimer(30 * time.Second)
		defer limit.Stop()
		tick := time.NewTicker(100 * time.Millisecond)
		defer tick.Stop()
		for {
			current, err := p.inventory.scan()
			if err != nil {
				t.Fatalf("inventory during worker restart: %v", err)
			}
			var worker, bridge serviceProcessIdentity
			proxyUnchanged, oldPresent := false, false
			for _, child := range current {
				if child == originalWorker || child == originalBridge {
					oldPresent = true
				}
				if child.Path == originalWorker.Path {
					worker = child
				}
				if child.Path == originalBridge.Path {
					bridge = child
				}
				if child == originalProxy {
					proxyUnchanged = true
				}
			}
			if !proxyUnchanged {
				t.Fatal("proxy identity changed during workload-worker restart")
			}
			if !oldPresent && worker.PID != 0 && bridge.PID != 0 {
				p.workerPID = worker.PID
				t.Logf("fixture_restart old_worker=%+v new_worker=%+v old_bridge=%+v new_bridge=%+v unchanged_proxy=%+v", originalWorker, worker, originalBridge, bridge, originalProxy)
				return
			}
			select {
			case <-tick.C:
			case <-limit.C:
				t.Fatal("owned workload worker/bridge did not respawn")
			}
		}
	}
	awaitReplacement()
	// Keep the peer available: closing its listener invalidates the replacement
	// worker's pooled connection. A replay must arrive on a new connection after
	// the original worker has exited, so queued predecessor traffic cannot pass.
	replayDeadline := time.NewTimer(75 * time.Second)
	defer replayDeadline.Stop()
	measuredReplay, controlReplay := false, false
	for !measuredReplay || !controlReplay {
		select {
		case raw := <-received:
			frame := read(raw)
			if oldPeerConnections[raw.RemoteAddr] {
				continue
			}
			info := frame.Params.WorkloadInfo
			if info.ID != id && info.ID != workloadBroadcastControlID {
				continue
			}
			// Broker startup rehydration forwards canonical workload frames;
			// only the manager's later heartbeat adds resync:true.
			if info.OriginatedFrom != fx.nodeUUID || info.RunID != runID || info.Engine != "ollama" || frame.Method != "workload:completed" || info.State != "completed" || info.Seq != terminalSeq {
				t.Fatalf("restarted manager replay identity/state: origin=%q run=%q engine=%q method=%q state=%q seq=%d want seq=%d", info.OriginatedFrom, info.RunID, info.Engine, frame.Method, info.State, info.Seq, terminalSeq)
			}
			if info.ID == id && info.Model == workloadBroadcastModel {
				measuredReplay = true
			}
			if info.ID == workloadBroadcastControlID && info.Model == workloadBroadcastControlModel {
				controlReplay = true
			}
		case <-replayDeadline.C:
			t.Fatalf("restarted manager replay missing: measured=%t injected=%t", measuredReplay, controlReplay)
		}
	}
	// Join the producing processes and peer handlers before inspecting the
	// remaining capture. The restarted manager's measured/control reassertions
	// are the positive end barrier; no timing-only absence assertion is used.
	p.cleanup()
	stopPeer()
	for raw := range received {
		read(raw)
	}
	if measuredRequests.Load() != 1 || len(capture) == 0 {
		t.Fatalf("measured upstream requests=%d peer frames=%d, want one request and nonempty capture", measuredRequests.Load(), len(capture))
	}
	t.Logf("outbound controls: origin=%q engine=ollama run=%q workload=%q submitted/terminal delivered, private input/output observed, projected injection and restart replay observed", fx.nodeUUID, runID, id)
	return capture
}

// Preserve the entire wire body: jsonrpc.Message alone would discard unknown
// envelope members before the privacy assertion could inspect them.
type workloadBroadcastCapture struct {
	Raw        json.RawMessage
	RemoteAddr string
}

func startWorkloadBroadcastPeer(t *testing.T, fx *interNodeCluster) (<-chan workloadBroadcastCapture, int, func()) {
	t.Helper()
	received := make(chan workloadBroadcastCapture, 128)
	listener := listenFixtureLoopback(t)
	port := listener.Addr().(*net.TCPAddr).Port
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, ok := fx.peer.VerifyClientPin(r); !ok {
			http.Error(w, "not a pinned cluster peer", http.StatusForbidden)
			return
		}
		if r.Method != http.MethodPost || r.URL.Path != "/v1/workloads/events" {
			t.Errorf("unexpected peer request %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read peer frame: %v", err)
			http.Error(w, "read peer frame", http.StatusBadRequest)
			return
		}
		select {
		case received <- workloadBroadcastCapture{Raw: json.RawMessage(body), RemoteAddr: r.RemoteAddr}:
			w.WriteHeader(http.StatusOK)
		default:
			t.Error("peer capture exceeded its frame budget")
			http.Error(w, "capture full", http.StatusServiceUnavailable)
		}
	})
	var server *http.Server
	var done chan error
	start := func(listener net.Listener) {
		running := &http.Server{Handler: handler}
		exited := make(chan error, 1)
		server, done = running, exited
		go func() { exited <- running.Serve(tls.NewListener(listener, fx.peer.ServerTLSConfig())) }()
	}
	stopServing := func() {
		t.Helper()
		if server == nil {
			return
		}
		running := server
		server = nil
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		if err := running.Shutdown(ctx); err != nil {
			if closeErr := running.Close(); closeErr != nil {
				t.Errorf("close peer: %v", closeErr)
			}
			t.Fatalf("join peer handlers before freezing capture: %v", err)
		}
		select {
		case err := <-done:
			if err != http.ErrServerClosed {
				t.Errorf("peer server exit: %v", err)
			}
		case <-ctx.Done():
			t.Fatal("peer server did not exit before freezing capture")
		}
	}
	start(listener)
	var once sync.Once
	stop := func() {
		once.Do(func() {
			stopServing()
			close(received)
			assertInteropListenerClosed(t, port)
		})
	}
	t.Cleanup(stop)
	t.Logf("fixture_listener peer_port=%d", port)
	return received, port, stop
}

// postPeerEvent POSTs a peer workload frame to the manager's inter-node endpoint
// as a pinned cluster peer, retrying until it returns 200 (the listener comes up
// in a goroutine after the broker spawns the manager) or the timeout elapses. The
// endpoint is cluster mTLS only, so client must carry a cluster identity the node
// under test pins — see interNodeCluster.clientAsPeer.
func postPeerEvent(t *testing.T, client *http.Client, endpoint, body string, timeout time.Duration) {
	t.Helper()
	deadline := time.After(timeout)
	tick := time.NewTicker(300 * time.Millisecond)
	defer tick.Stop()
	for {
		resp, err := postFixturePeerEvent(client, endpoint, body)
		if err == nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
			t.Logf("peer POST returned HTTP %d, retrying", resp.StatusCode)
		}
		select {
		case <-tick.C:
		case <-deadline:
			t.Fatalf("timed out POSTing peer event to %s (last err: %v)", endpoint, err)
		}
	}
}

// Check the target before handing the request to any transport, including one
// with a dialer that could reach an installed service. Synthetic default-port
// controls exercise this path without opening a listener or making a connection.
func postFixturePeerEvent(client *http.Client, endpoint, body string) (*http.Response, error) {
	request, err := http.NewRequest(http.MethodPost, endpoint, strings.NewReader(body))
	if err != nil {
		return nil, err
	}
	port, err := strconv.Atoi(request.URL.Port())
	if err != nil {
		return nil, fmt.Errorf("fixture workload endpoint requires an explicit port: %w", err)
	}
	if err := validateFixturePort(port); err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	return client.Do(request)
}

type workloadFixtureTransport func(*http.Request) (*http.Response, error)

func (transport workloadFixtureTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	return transport(request)
}

// This construction/control regression is nested under the inbound parent so
// the focused six-parent selector includes it. It allocates no real listeners
// and gives its client only an in-memory transport.
func testWorkloadFixtureIsolation(t *testing.T) {
	safe := fixtureServicePorts{
		nodeInfo: 24001, errors: 24002, workload: 24003,
		clusterManager: 24004, engineManagerHTTP: 24005, engineControl: 24006,
	}
	t.Run("launch registration and inbound target agree", func(t *testing.T) {
		args, err := safe.brokerArgs("synthetic-cluster", "--workload-manager-path", "synthetic-worker")
		if err != nil {
			t.Fatal(err)
		}
		flags := make(map[string]string)
		for i := 0; i < len(args); i += 2 {
			if i+1 >= len(args) {
				t.Fatalf("unpaired fixture launch flag: %v", args)
			}
			if _, exists := flags[args[i]]; exists {
				t.Fatalf("duplicate fixture launch flag %s", args[i])
			}
			flags[args[i]] = args[i+1]
		}
		for flag, want := range map[string]string{
			"--node-info-port": "24001", "--errors-port": "24002", "--workload-manager-port": "24003",
			"--cluster-manager-port": "24004", "--engine-manager-http-port": "24005", "--engine-manager-control-port": "24006",
			"--scanner-path": scannerBin, "--cluster-dir": "synthetic-cluster", "--workload-manager-path": "synthetic-worker",
		} {
			if flags[flag] != want {
				t.Fatalf("launch %s = %q, want %q", flag, flags[flag], want)
			}
		}
		// The broker's spawnWorkloadManager passes this flag's port to the
		// worker and registers that same port as ServiceWorkload. This is the
		// expected registration contract; no broker or scanner runs here.
		registered := noderec.RegisterParams{Service: noderec.ServiceWorkload, Port: safe.workload}
		if flags["--workload-manager-port"] != strconv.Itoa(registered.Port) {
			t.Fatalf("launch flags %v disagree with registration %+v", flags, registered)
		}
		endpoint, err := safe.workloadEventsURL()
		if err != nil {
			t.Fatal(err)
		}
		calls := 0
		client := &http.Client{Transport: workloadFixtureTransport(func(request *http.Request) (*http.Response, error) {
			calls++
			if request.URL.Scheme != "https" || request.URL.Hostname() != "127.0.0.1" || request.URL.Port() != strconv.Itoa(registered.Port) || request.URL.Path != "/v1/workloads/events" {
				t.Fatalf("inbound URL %s disagrees with registration %+v", request.URL, registered)
			}
			if request.Method != http.MethodPost || request.Header.Get("Content-Type") != "application/json" {
				t.Fatalf("workload request method=%s headers=%v", request.Method, request.Header)
			}
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("")), Header: make(http.Header), Request: request}, nil
		})}
		response, err := postFixturePeerEvent(client, endpoint, `{}`)
		if err != nil {
			t.Fatal(err)
		}
		if err := response.Body.Close(); err != nil {
			t.Fatal(err)
		}
		if calls != 1 {
			t.Fatalf("transport calls = %d, want 1", calls)
		}
	})
	for port, owner := range installedServicePorts {
		t.Run("rejects installed "+owner, func(t *testing.T) {
			unsafe := safe
			unsafe.workload = port
			if args, err := unsafe.brokerArgs("synthetic-cluster"); err == nil || args != nil {
				t.Fatalf("installed port %d produced launch args %v, err=%v", port, args, err)
			}
			if endpoint, err := unsafe.workloadEventsURL(); err == nil || endpoint != "" {
				t.Fatalf("installed port %d produced inbound URL %q, err=%v", port, endpoint, err)
			}
			calls := 0
			client := &http.Client{Transport: workloadFixtureTransport(func(*http.Request) (*http.Response, error) {
				calls++
				return nil, fmt.Errorf("installed-port control reached transport")
			})}
			// In particular 14320 is only a URL string, never a live negative control.
			endpoint := fmt.Sprintf("https://127.0.0.1:%d/v1/workloads/events", port)
			if response, err := postFixturePeerEvent(client, endpoint, `{}`); err == nil || response != nil || calls != 0 {
				t.Fatalf("installed port %d: response=%v err=%v transport calls=%d, want nil, error, 0", port, response, err, calls)
			}
		})
	}
	t.Run("extra arguments cannot replace the plan", func(t *testing.T) {
		portArgs := safe.args()
		for i := 0; i < len(portArgs); i += 2 {
			name := strings.TrimLeft(portArgs[i], "-")
			for _, override := range [][]string{
				{"--" + name, "14320"}, {"--" + name + "=14320"},
				{"-" + name, "14320"}, {"-" + name + "=14320"},
			} {
				if args, err := safe.brokerArgs("synthetic-cluster", override...); err == nil || args != nil {
					t.Fatalf("override %v produced launch args %v, err=%v", override, args, err)
				}
			}
		}
	})
}

// waitProxyReady polls ollama-proxy:get-status until the Ollama proxy reports ready
// and returns its bound port.
func waitProxyReady(t *testing.T, stdin io.Writer, msgs <-chan jsonrpc.Message, timeout time.Duration) int {
	t.Helper()
	return waitEngineProxyReady(t, "ollama-proxy", stdin, msgs, timeout)
}

// waitLMStudioProxyReady is waitProxyReady for the LM Studio relay namespace.
func waitLMStudioProxyReady(t *testing.T, stdin io.Writer, msgs <-chan jsonrpc.Message, timeout time.Duration) int {
	t.Helper()
	return waitEngineProxyReady(t, "lmstudio-proxy", stdin, msgs, timeout)
}

// engineReadyPollSeq hands every readiness poll a distinct JSON-RPC id. Based
// at 9000 to stay clear of the fixed ids other helpers on this stream use.
var engineReadyPollSeq atomic.Int64

// waitEngineProxyReady polls <namespace>:get-status until that proxy reports
// ready and returns its bound port. The two engines differ only by relay
// namespace, so they share one implementation.
func waitEngineProxyReady(t *testing.T, namespace string, stdin io.Writer, msgs <-chan jsonrpc.Message, timeout time.Duration) int {
	t.Helper()
	method := namespace + ":get-status"
	// Accept only replies to this invocation's own polls. One process hosts
	// both facades and answers get-status for each on this one stream, so a
	// poll's reply can still be in flight when the next wait begins — and
	// every one of them looks alike: a response frame carrying ready and port.
	// Matching on shape alone returns the other engine's port, which is
	// exactly what the tests waiting on both facades assert is different.
	pending := map[int64]bool{}
	poll := func() {
		id := 9000 + engineReadyPollSeq.Add(1)
		pending[id] = true
		writeRawFrame(t, stdin, fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":%q}`, id, method))
	}
	poll()
	deadline := time.After(timeout)
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatalf("broker stream closed waiting for %s", method)
			}
			if msg.ID == nil || msg.Method != "" {
				continue
			}
			var id int64
			if json.Unmarshal(*msg.ID, &id) != nil || !pending[id] {
				continue
			}
			delete(pending, id)
			var st struct {
				Ready bool `json:"ready"`
				Port  int  `json:"port"`
			}
			if json.Unmarshal(msg.Result, &st) == nil && st.Ready {
				return st.Port
			}
		case <-tick.C:
			poll()
		case <-deadline:
			t.Fatalf("timed out (%s) waiting for %s proxy to become ready", timeout, namespace)
		}
	}
}

// waitForWorkloadEvent blocks until a workloads:* notification with the
// given method and workloadInfo.id arrives, returning its parsed params.
func waitForWorkloadEvent(t *testing.T, msgs <-chan jsonrpc.Message, method, id string, timeout time.Duration) wlParams {
	t.Helper()
	to := time.After(timeout)
	for {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatalf("stream closed before %s for %q", method, id)
			}
			if msg.Method != method {
				continue
			}
			var p wlParams
			if err := json.Unmarshal(msg.Params, &p); err != nil {
				t.Fatalf("unmarshal %s: %v\nraw: %s", method, err, msg.Params)
			}
			if p.WorkloadInfo.ID == id {
				return p
			}
		case <-to:
			t.Fatalf("timed out (%s) waiting for %s %q", timeout, method, id)
		}
	}
}
