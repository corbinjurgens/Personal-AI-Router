// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Cross-process tests for the two broker-management hotfix legs:
//
//	A. the broker bridges a reachable supervised manual node into
//	   ollama-proxy (node/add-manual), so inference can route to a node
//	   that never appears over mDNS; and
//	B. the broker accepts a client-originated errors:report and forwards it
//	   into nvpair-errors, reflected in the next errors:update.
//
// The broker fixtures use binaries built by TestMain.
package tests

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"nvpair-shared/testbin"
)

// This adapter only changes a real worker's listen port; its stdio protocol
// passes through unchanged, following the workload interop adapter precedent.
// Broker fixtures configure their service ports with the broker's own port
// flags instead (see fixture_ports_test.go); the adapter remains for fixtures
// that start a worker outside that plan.
func init() {
	if bridge := os.Getenv("NVPAIR_TEST_PORT_BRIDGE"); bridge != "" && bridge == os.Args[0] {
		if err := runPortInteropBridge(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
}

func runPortInteropBridge() error {
	args := append(os.Args[1:], "--port", os.Getenv("NVPAIR_TEST_PORT"))
	return runOwnedPortWorker(os.Getenv("NVPAIR_TEST_PORT_BINARY"), args)
}

func newPortInteropBridge(t *testing.T, binary string, port int) (string, []string) {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	bridge := filepath.Join(testbin.Dir(t, "tests", "port-bridge", "port-bridge"), testbin.Exe("port-bridge"))
	if err := copyPortBridge(func() error { return os.WriteFile(bridge, data, 0o700) }, time.Sleep); err != nil {
		t.Fatalf("copy port bridge to %q (a previous process may still be running): %v", bridge, err)
	}
	return bridge, []string{
		"NVPAIR_TEST_PORT_BRIDGE=" + bridge,
		"NVPAIR_TEST_PORT_BINARY=" + binary,
		"NVPAIR_TEST_PORT=" + strconv.Itoa(port),
	}
}

const (
	portBridgeCopyAttempts = 41
	portBridgeCopyDelay    = 50 * time.Millisecond
)

// The adapter is a broker child, so this fixture has no exec.Cmd to wait on.
// Windows can retain its executable lock briefly after process cleanup. Allow
// that lock two seconds to clear, while returning every other copy error at once.
func copyPortBridge(write func() error, sleep func(time.Duration)) error {
	for attempt := 1; ; attempt++ {
		err := write()
		if err == nil || !isPortBridgeSharingViolation(err) || attempt == portBridgeCopyAttempts {
			return err
		}
		sleep(portBridgeCopyDelay)
	}
}

func startClientErrorsBroker(t *testing.T) *interopProcess {
	t.Helper()
	base := t.TempDir()
	ports := reserveFixtureServicePorts(t)
	args := append([]string{"--cluster-dir", filepath.Join(base, "cluster"), "--scanner-path", scannerBin, "--errors-path", errorsBin}, ports.args()...)
	p := startInteropProcess(t, base, brokerBin, ports.errors, nil, args...)
	for _, path := range []string{scannerBin, errorsBin} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		p.paths = append(p.paths, canonical)
	}
	waitForMethod(t, p.msgs, "app:ready", 10*time.Second)
	return p
}

// manualProbePorts are the product ports nvpair-manual-nodes probes on a manual
// node's address (Ollama, LM Studio, node-info). The worker fixes them, so a
// test pointing it at loopback fakes must own all three: otherwise the prober
// reaches whatever else listens there, typically an installed PAIR.
var manualProbePorts = []int{11434, 1234, 14318}

// skipIfManualProbePortsOccupied skips, without binding beside or contacting the
// owner, when any manual probe port is held on IPv4 or IPv6.
func skipIfManualProbePortsOccupied(t *testing.T) {
	t.Helper()
	for _, port := range manualProbePorts {
		skipIfDefaultPortOccupied(t, port, "the loopback manual-node probe fixture")
	}
}

// fakeOllama stands up a minimal Ollama-looking HTTP server on the fixed
// port nvpair-manual-nodes probes (11434): GET / returns 200 (liveness) and
// GET /api/tags returns one model, so a manual node pointed at it reports
// ollama_up=true with a model list. Returns a stop func. Callers first run
// skipIfManualProbePortsOccupied. Skips the test if the port can't be bound.
func fakeOllama(t *testing.T) func() {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:11434")
	if err != nil {
		t.Skipf("cannot bind fake ollama on 127.0.0.1:11434 (%v); skipping", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"models":[{"name":"llama3.2:latest"}]}`)
	})
	srv := &http.Server{Handler: mux}
	go func() { _ = srv.Serve(ln) }()
	return func() { _ = srv.Close() }
}

// Own every endpoint the manual worker probes, including a negative Ollama
// response and a stable node-info identity. Otherwise an installed node-info
// changes the candidate's key from the manual name to the machine's UUID.
func fakeLMStudio(t *testing.T, hostUUID string) {
	t.Helper()
	skipIfManualProbePortsOccupied(t)
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := io.WriteString(w, `{"object":"list","data":[{"id":"qwen2.5-7b-instruct","object":"model"}]}`); err != nil {
			t.Errorf("write fixture LM Studio model list: %v", err)
		}
	})
	nodeInfo := http.NewServeMux()
	nodeInfo.HandleFunc("/v1/node-info", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := io.WriteString(w, fmt.Sprintf(`{"GPUs":[],"hostUuid":%q}`, hostUUID)); err != nil {
			t.Errorf("write fixture node-info: %v", err)
		}
	})
	for _, endpoint := range []struct {
		port    int
		handler http.Handler
	}{
		{1234, mux},
		{11434, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) })},
		{14318, nodeInfo},
	} {
		ln, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(endpoint.port)))
		if err != nil {
			t.Fatalf("bind manual probe fixture port %d: %v", endpoint.port, err)
		}
		srv := &http.Server{Handler: endpoint.handler}
		go func() {
			if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
				t.Errorf("serve manual probe fixture: %v", err)
			}
		}()
		t.Cleanup(func() {
			if err := srv.Close(); err != nil {
				t.Errorf("close manual probe fixture: %v", err)
			}
		})
	}
}

// fakeNodeInfo stands up a minimal node-info server on the fixed port the
// manual prober hits (14318): GET /v1/node-info returns 200 with the given
// hostUuid, so a manual node pointed at it learns its stable UUID. Callers
// first run skipIfManualProbePortsOccupied. Skips the test if the port can't
// be bound.
func fakeNodeInfo(t *testing.T, hostUUID string) func() {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:14318")
	if err != nil {
		t.Skipf("cannot bind fake node-info on 127.0.0.1:14318 (%v); skipping", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/node-info", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, fmt.Sprintf(`{"GPUs":[],"hostUuid":%q}`, hostUUID))
	})
	srv := &http.Server{Handler: mux}
	go func() { _ = srv.Serve(ln) }()
	return func() { _ = srv.Close() }
}

// proxyNode mirrors the proxy's nodes/list element (ollama-proxy Node).
type proxyNode struct {
	ID        string   `json:"id"`
	Host      string   `json:"host"`
	Port      int      `json:"port"`
	Addresses []string `json:"addresses"`
	TXT       []string `json:"txt"`
}

type proxyNodesResult struct {
	Nodes []proxyNode `json:"nodes"`
}

func proxyNodesHas(t *testing.T, raw json.RawMessage, id string) bool {
	t.Helper()
	var res proxyNodesResult
	if err := json.Unmarshal(raw, &res); err != nil {
		return false
	}
	for _, n := range res.Nodes {
		if n.ID == id {
			return true
		}
	}
	return false
}

// TestBrokerBridgesManualNodeIntoProxy is the leg-A repro: with the broker
// supervising both nvpair-manual-nodes and ollama-proxy, a manual node whose
// Ollama is reachable must be bridged into the proxy (node/add-manual) so it
// shows up in ollama-proxy:nodes/list and can be routed to — even though it never
// appears over mDNS.
func TestBrokerBridgesManualNodeIntoProxy(t *testing.T) {
	if portBusy(11435) {
		t.Skip("ollama-proxy port 11435 already in use; skipping")
	}
	skipIfManualProbePortsOccupied(t)
	stopOllama := fakeOllama(t)
	t.Cleanup(stopOllama)

	stdin, msgs, _, cleanup := startBrokerWith(t,
		"--manual-nodes-path", manualNodesBin,
		"--proxy-path", proxyBin, "--proxy-engines", "ollama",
	)
	t.Cleanup(cleanup)

	waitForMethod(t, msgs, "app:ready", 10*time.Second)

	const nodeName = "xproc-manual-proxy-bridge"
	addReq := fmt.Sprintf(`{"jsonrpc":"2.0","id":900,"method":"node/add","params":{"address":"127.0.0.1","name":%q}}`, nodeName) + "\n"
	if _, err := stdin.Write([]byte(addReq)); err != nil {
		t.Fatalf("write node/add: %v", err)
	}

	// Poll the proxy's node set (via the broker's ollama-proxy: relay) until the
	// bridged manual node appears. The first probe fires immediately on
	// node/add; re-probes every 10s re-bridge if the proxy wasn't ready yet.
	deadline := time.After(25 * time.Second)
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	reqID := 901
	sendReq(t, stdin, reqID, "ollama-proxy:nodes/list")
	for {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatal("broker stream closed before the manual node bridged into the proxy")
			}
			if msg.Method == "" && msg.ID != nil && proxyNodesHas(t, msg.Result, nodeName) {
				t.Logf("manual node %q bridged into proxy nodes/list", nodeName)
				return
			}
		case <-ticker.C:
			reqID++
			sendReq(t, stdin, reqID, "ollama-proxy:nodes/list")
		case <-deadline:
			t.Fatalf("timed out waiting for manual node %q in ollama-proxy:nodes/list", nodeName)
		}
	}
}

// TestBrokerBridgesManualNodeUnderLearnedUUID: once a
// manual node's node-info reports its stable hostUuid, the proxy candidate must
// be keyed by that UUID — the same operational key the discovery store and
// scheduler use — so scheduler priority (node/set-priority) and scheduledOn
// resolve to it. The candidate must therefore appear in ollama-proxy:nodes/list under
// the learned hostUuid, not the user-supplied manual name.
func TestBrokerBridgesManualNodeUnderLearnedUUID(t *testing.T) {
	if portBusy(11435) {
		t.Skip("ollama-proxy port 11435 already in use; skipping")
	}
	skipIfManualProbePortsOccupied(t)
	stopOllama := fakeOllama(t)
	t.Cleanup(stopOllama)
	stopNodeInfo := fakeNodeInfo(t, "learned-host-uuid")
	t.Cleanup(stopNodeInfo)

	stdin, msgs, _, cleanup := startBrokerWith(t,
		"--manual-nodes-path", manualNodesBin,
		"--proxy-path", proxyBin, "--proxy-engines", "ollama",
	)
	t.Cleanup(cleanup)

	waitForMethod(t, msgs, "app:ready", 10*time.Second)

	const nodeName = "xproc-manual-learned-uuid"
	addReq := fmt.Sprintf(`{"jsonrpc":"2.0","id":940,"method":"node/add","params":{"address":"127.0.0.1","name":%q}}`, nodeName) + "\n"
	if _, err := stdin.Write([]byte(addReq)); err != nil {
		t.Fatalf("write node/add: %v", err)
	}

	deadline := time.After(25 * time.Second)
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	reqID := 941
	sendReq(t, stdin, reqID, "ollama-proxy:nodes/list")
	for {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatal("broker stream closed before the manual node bridged under its learned UUID")
			}
			if msg.Method == "" && msg.ID != nil {
				if proxyNodesHas(t, msg.Result, "learned-host-uuid") {
					t.Logf("manual node bridged into proxy under learned hostUuid")
					return
				}
				if proxyNodesHas(t, msg.Result, nodeName) {
					t.Fatalf("proxy candidate keyed by manual name %q, not the learned hostUuid (scheduler would not match)", nodeName)
				}
			}
		case <-ticker.C:
			reqID++
			sendReq(t, stdin, reqID, "ollama-proxy:nodes/list")
		case <-deadline:
			t.Fatalf("timed out waiting for manual node under learned hostUuid in ollama-proxy:nodes/list")
		}
	}
}

// TestBrokerBridgesManualNodeIntoLMStudioProxy is the LM Studio counterpart of
// the Ollama bridge test: with the broker supervising both nvpair-manual-nodes and
// lmstudio-proxy, a manual node whose LM Studio is reachable must be bridged
// into lmstudio-proxy (node/add-manual) so it shows up in
// lmstudio-proxy:nodes/list — even though it never appears over mDNS.
func TestBrokerBridgesManualNodeIntoLMStudioProxy(t *testing.T) {
	const hostUUID = "xproc-manual-lmstudio-uuid"
	fakeLMStudio(t, hostUUID)
	base := t.TempDir()
	proxyPort := freePort(t)
	journal := map[string]any{"lmstudio": map[string]any{
		"explicit": true,
		"snapshot": map[string]any{
			"engine": "lmstudio", "revision": 1, "appliedRevision": 1, "phase": "idle", "format": "pair-arguments-v1",
			"settings": map[string]any{"serverPort": freePort(t), "proxyPort": proxyPort, "launchText": ""},
		},
	}}
	if err := os.WriteFile(filepath.Join(base, "engine-settings-operations.json"), mustMarshal(t, journal), 0o600); err != nil {
		t.Fatal(err)
	}
	args := append([]string{
		"--cluster-dir", filepath.Join(base, "cluster"), "--scanner-path", scannerBin,
		"--manual-nodes-path", manualNodesBin,
		"--proxy-path", proxyBin, "--proxy-engines", "lmstudio",
	}, reserveFixtureServicePorts(t).args()...)
	p := startInteropProcess(t, base, brokerBin, proxyPort, nil, args...)
	for _, path := range []string{scannerBin, manualNodesBin, proxyBin} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		p.paths = append(p.paths, canonical)
	}
	stdin, msgs := p.in, p.msgs

	waitForMethod(t, msgs, "app:ready", 10*time.Second)
	if port := waitLMStudioProxyReady(t, stdin, msgs, 10*time.Second); port != proxyPort {
		t.Fatalf("LM Studio proxy port=%d want=%d; stderr: %s", port, proxyPort, p.output.String())
	}

	const nodeName = "xproc-manual-lmstudio-bridge"
	addReq := fmt.Sprintf(`{"jsonrpc":"2.0","id":920,"method":"node/add","params":{"address":"127.0.0.1","name":%q}}`, nodeName) + "\n"
	if _, err := stdin.Write([]byte(addReq)); err != nil {
		t.Fatalf("write node/add: %v", err)
	}

	deadline := time.After(25 * time.Second)
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	reqID := 921
	sendReq(t, stdin, reqID, "lmstudio-proxy:nodes/list")
	for {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatal("broker stream closed before the manual node bridged into lmstudio-proxy")
			}
			if msg.Method == "" && msg.ID != nil {
				if msg.Error != nil {
					t.Fatalf("manual node/proxy RPC id=%s: %+v; stderr: %s", *msg.ID, msg.Error, p.output.String())
				}
				if proxyNodesHas(t, msg.Result, hostUUID) {
					t.Logf("manual node %q bridged under learned UUID %q", nodeName, hostUUID)
					return
				}
			}
		case <-ticker.C:
			reqID++
			sendReq(t, stdin, reqID, "lmstudio-proxy:nodes/list")
		case <-deadline:
			t.Fatalf("timed out waiting for manual node UUID %q in lmstudio-proxy:nodes/list; stderr: %s", hostUUID, p.output.String())
		}
	}
}

// TestBrokerAcceptsClientErrorsReport is the leg-B repro: a client-originated
// errors:report (notification form) must be forwarded into nvpair-errors and
// reflected in the next errors:update — the same as a supervised worker
// emitting one on its stdout.
func TestBrokerAcceptsClientErrorsReport(t *testing.T) {
	p := startClientErrorsBroker(t)
	stdin, msgs := p.in, p.msgs

	const id = "client:leg-b-notif"
	report := fmt.Sprintf(`{"jsonrpc":"2.0","method":"errors:report","params":{"id":%q,"message":"client-synthesized","severity":"error"}}`, id) + "\n"
	if _, err := stdin.Write([]byte(report)); err != nil {
		t.Fatalf("write errors:report notification: %v", err)
	}

	// The broker forwards it into nvpair-errors, which stores it and pushes an
	// errors:update the broker relays back unconditionally.
	waitForErrorsUpdateContaining(t, msgs, id, 10*time.Second)
	t.Logf("client errors:report %q reflected in errors:update", id)
}

// TestBrokerAcceptsClientErrorsReportRequest covers the request form of
// leg B: an id-bearing errors:report is acked with a null result and also
// reflected in errors:update.
func TestBrokerAcceptsClientErrorsReportRequest(t *testing.T) {
	p := startClientErrorsBroker(t)
	stdin, msgs := p.in, p.msgs

	const (
		reqID = 950
		id    = "client:leg-b-req"
	)
	report := fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":"errors:report","params":{"id":%q,"message":"client-synthesized","severity":"error"}}`, reqID, id) + "\n"
	if _, err := stdin.Write([]byte(report)); err != nil {
		t.Fatalf("write errors:report request: %v", err)
	}

	// Expect both the ack (result for reqID, no error) and the errors:update.
	gotAck := false
	gotUpdate := false
	deadline := time.After(10 * time.Second)
	for !(gotAck && gotUpdate) {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatal("broker stream closed before the report was acked and reflected")
			}
			switch {
			case msg.Method == "" && msg.ID != nil:
				if string(*msg.ID) == fmt.Sprintf("%d", reqID) {
					if msg.Error != nil {
						t.Fatalf("errors:report request returned an error: %+v", msg.Error)
					}
					gotAck = true
				}
			case msg.Method == methodErrorsUpdate:
				if errorsListHasID(t, msg.Params, id) {
					gotUpdate = true
				}
			}
		case <-deadline:
			t.Fatalf("timed out (ack=%v update=%v) for errors:report request %q", gotAck, gotUpdate, id)
		}
	}
}
