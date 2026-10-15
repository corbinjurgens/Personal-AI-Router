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
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
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
	"nvpair-shared/noderec"
)

const workloadMgrService = "_nvpair-workload-manager._tcp"

// nodeRecordService is the consolidated discovery record. A node
// advertises ONE of these carrying a compact TXT service map; the broker's
// node-scanner daemon browses it and relays the directory to subscribers.
const nodeRecordService = "_nvpair-node._tcp"

// The broker does not expose a workload worker port override. This test-only
// stdio adapter starts the real worker on an isolated port and supplies its peer
// directory through discovery:nodes. All other broker input and all worker
// output pass through unchanged; it never synthesizes a service response or a
// workload. Dispatch before TestMain so broker worker arguments are not parsed
// as testing flags and the adapter does not build another binary bundle.
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
	args := append(os.Args[1:], "--port", os.Getenv("NVPAIR_TEST_WORKLOAD_PORT"))
	cmd := exec.Command(os.Getenv("NVPAIR_TEST_WORKLOAD_BINARY"), args...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "fixture_worker_started pid=%d\n", cmd.Process.Pid)
	wait := make(chan error, 1)
	go func() { wait <- cmd.Wait() }()
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
	if port != 0 {
		p.ports = []int{port}
	}
	p.cmd.Dir = base
	p.cmd.Env = append(os.Environ(), "HOME="+base, "XDG_CONFIG_HOME="+base, "APPDATA="+base, "LOCALAPPDATA="+base, "USERPROFILE="+base)
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

// startBrokerProc starts the broker with the given extra args and returns
// its stdin, a reader over its stdout JSON-RPC frames, and a cleanup that
// closes stdin and waits for a graceful exit (the broker tears down its
// children with a 2 s grace each, so the wait window is generous).
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
	args = append([]string{"--cluster-dir", clusterDir}, args...)
	cmd := exec.Command(brokerBin, args...)
	cmd.Stderr = os.Stderr

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("broker stdin pipe: %v", err)
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("broker stdout pipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("start broker: %v", err)
	}
	t.Logf("broker started: pid=%d", cmd.Process.Pid)

	ch := startMsgReader(stdoutPipe)
	return stdinPipe, ch, func() {
		stdinPipe.Close()
		done := make(chan error, 1)
		go func() { done <- cmd.Wait() }()
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			cmd.Process.Kill()
			<-done
		}
	}
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
	fx := newInterNodeCluster(t)
	stdin, msgs, cleanup := startBrokerProcInCluster(t, fx.nodeDir,
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
	// The manager binds :14320 in a goroutine, so retry until it accepts.
	const peerEvent = `{"jsonrpc":"2.0","method":"workload:started","params":{"workloadInfo":{"id":"peer-wl-1","model":"llama-3-70b","engine":"trt-llm","state":"running","originatedFrom":"peer-node-A","createdAt":1716998400000,"startedAt":1716998400000,"completedAt":null,"error":null,"requesterId":null}}}`
	postPeerEvent(t, fx.clientAsPeer(t), "https://127.0.0.1:14320/v1/workloads/events", peerEvent, 15*time.Second)
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
	fx := newInterNodeCluster(t)
	stdin, msgs, cleanup := startBrokerProcInCluster(t, fx.nodeDir,
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

	const endpoint = "https://127.0.0.1:14320/v1/workloads/events"
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

	fx := newInterNodeCluster(t)
	stdin, msgs, cleanup := startBrokerProcInCluster(t, fx.nodeDir,
		"--scanner-path", scannerBin,
		"--workload-manager-path", workloadMgrBin,
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
		Instance: peerNode, Service: nodeRecordService, Port: 14999, TXT: peerTXT, Interface: loopbackMDNSInterface(t).Name,
	})
	t.Logf("advertising peer %q as %s", peerNode, nodeRecordService)

	// The peer must be in the broker's directory before its removal can
	// trigger a sweep, so wait until discovery has folded it in.
	pollNodeListed(t, stdin, msgs, peerNode, 1000, 30*time.Second, nil)
	t.Logf("peer %q discovered", peerNode)

	// A peer-origin running workload (scheduled on that same peer) POSTed to
	// the manager's inter-node endpoint. The broker relays it to us as a
	// tracked workloads:upsert. Stamped with the peer's HostUUID (not its
	// name), as a real proxy does.
	peerEvent := fmt.Sprintf(`{"jsonrpc":"2.0","method":"workload:started","params":{"workloadInfo":{"id":"wl-nodeloss-1","model":"llama-3-8b","engine":"ollama","state":"running","originatedFrom":%q,"scheduledOn":%q,"createdAt":1716998400000,"startedAt":1716998400000,"completedAt":null,"error":null,"requesterId":null}}}`, peerUUID, peerUUID)
	postPeerEvent(t, fx.clientAsPeer(t), "https://127.0.0.1:14320/v1/workloads/events", peerEvent, 15*time.Second)

	running := waitForWorkloadEvent(t, msgs, "workloads:upsert", "wl-nodeloss-1", 10*time.Second)
	if running.WorkloadInfo.State != "running" {
		t.Fatalf("initial upsert state = %q, want running", running.WorkloadInfo.State)
	}
	t.Log("peer workload tracked as running")

	// Pull the peer off the network. The scanner holds a node through
	// shared/discovery's full miss threshold (12 scans at 5s, a full minute) with
	// a failed liveness probe on every scan from ~15s in, and only then evicts it,
	// at which point the broker marks the workload it can no longer hear about.
	// The window is deliberately long enough to outlast a node saturated by its
	// own inference load, so this deadline has to clear it with room for the
	// probes and scan jitter.
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
	// Fake Ollama upstream the proxy forwards to (200 on any request).
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"done":true}`))
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
	received, peerPort := fx.startDirectClusterPeer(t)
	base := t.TempDir()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	bridge := filepath.Join(base, "workload-bridge"+filepath.Ext(workloadMgrBin))
	if err := os.WriteFile(bridge, data, 0o700); err != nil {
		t.Fatal(err)
	}
	workerPort := temporaryClusterPort(t)
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
	p := startInteropProcess(t, base, brokerBin, workerPort, []string{
		"OLLAMA_HOST=",
		"NVPAIR_TEST_WORKLOAD_BRIDGE=" + bridge,
		"NVPAIR_TEST_WORKLOAD_BINARY=" + workloadMgrBin,
		"NVPAIR_TEST_WORKLOAD_PORT=" + strconv.Itoa(workerPort),
		"NVPAIR_TEST_WORKLOAD_NODES=" + string(mustMarshal(t, directory)),
	},
		"--cluster-dir", fx.nodeDir,
		"--scanner-path", scannerBin,
		"--proxy-path", proxyBin, "--proxy-engines", "ollama",
		"--workload-manager-path", bridge,
	)
	p.paths = append(p.paths, scannerBin, proxyBin, bridge, workloadMgrBin)
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
	p.ports = append(p.ports, proxyPort)
	waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(workerPort)), 10*time.Second)
	owned, err := p.inventory.scan()
	if err != nil || len(owned) != len(p.paths) {
		t.Fatalf("fixture process inventory: owned=%+v err=%v want paths=%v", owned, err, p.paths)
	}
	for _, child := range owned {
		if child.Path == p.paths[len(p.paths)-1] {
			p.workerPID = child.PID
		}
	}
	if p.workerPID == 0 {
		t.Fatal("real workload worker missing from fixture inventory")
	}
	t.Logf("fixture_inventory owned=%+v", owned)
	t.Logf("proxy ready on port %d", proxyPort)

	// Register the fake Ollama as a manual node and pin it as the target so
	// the proxy forwards there deterministically.
	writeRawFrame(t, stdin, fmt.Sprintf(`{"jsonrpc":"2.0","id":50,"method":"ollama-proxy:node/add-manual","params":{"id":"fake-ollama","host":"127.0.0.1","port":%d,"addresses":["127.0.0.1"],"models":["test-model"]}}`, ollamaPort))
	if reply := waitForResponse(t, msgs, 5*time.Second); reply.Error != nil {
		t.Fatalf("add manual upstream: %+v", reply.Error)
	}
	writeRawFrame(t, stdin, `{"jsonrpc":"2.0","id":51,"method":"ollama-proxy:node/select","params":{"id":"fake-ollama"}}`)
	if reply := waitForResponse(t, msgs, 5*time.Second); reply.Error != nil {
		t.Fatalf("select upstream: %+v", reply.Error)
	}
	t.Log("fake ollama pinned as proxy target")

	// Fire inference requests until the stub peer records a broadcast. The
	// retry loop absorbs worker discovery latency (the manager refreshes its
	// peer set every 5 s and only broadcasts to peers it already knows).
	client := &http.Client{Timeout: 3 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	fire := func() {
		resp, err := client.Post(
			fmt.Sprintf("http://127.0.0.1:%d/api/chat", proxyPort),
			"application/json",
			strings.NewReader(`{"model":"test-model","messages":[]}`),
		)
		if err != nil {
			t.Fatalf("inference through real proxy: %v", err)
		}
		if _, err := io.Copy(io.Discard, resp.Body); err != nil {
			t.Errorf("read inference response: %v", err)
		}
		if err := resp.Body.Close(); err != nil {
			t.Errorf("close inference response: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("inference status=%d want=200", resp.StatusCode)
		}
	}
	fire()

	deadline := time.After(50 * time.Second)
	reqTick := time.NewTicker(2500 * time.Millisecond)
	defer reqTick.Stop()
	for {
		select {
		case msg := <-received:
			if !strings.HasPrefix(msg.Method, "workload:") {
				continue
			}
			var p wlParams
			if err := json.Unmarshal(msg.Params, &p); err != nil {
				t.Fatalf("unmarshal broadcast params: %v\nraw: %s", err, msg.Params)
			}
			if p.WorkloadInfo.Model != "test-model" {
				continue
			}
			if p.WorkloadInfo.OriginatedFrom != fx.nodeUUID {
				t.Errorf("broadcast workload originatedFrom=%q want=%q", p.WorkloadInfo.OriginatedFrom, fx.nodeUUID)
			}
			if p.WorkloadInfo.Engine != "ollama" {
				t.Errorf("broadcast workload engine = %q, want %q", p.WorkloadInfo.Engine, "ollama")
			}
			t.Logf("outbound OK: stub peer received %s for model %q from node %q",
				msg.Method, p.WorkloadInfo.Model, p.WorkloadInfo.OriginatedFrom)
			return
		case <-reqTick.C:
			fire()
		case <-deadline:
			t.Fatal("timed out: stub peer never received a workload broadcast")
		}
	}
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
		resp, err := client.Post(endpoint, "application/json", strings.NewReader(body))
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
