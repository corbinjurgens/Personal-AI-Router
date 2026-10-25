// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Cross-process test for nvpair-ui-broker's opt-in discovery notification
// stream (discovery:subscribe / discovery:unsubscribe).
//
// A test-owned scanner subprocess emits the shared discovery node-event
// protocol. This isolates the real broker's subscription contract from host
// multicast availability. The scanner module's daemon/directory tests cover
// discovery ingestion; real multicast is exercised by discovery_test.go and
// discovery_relay_test.go. We assert on the broker's stdout JSON-RPC frames:
//
//   - while unsubscribed: the node is reachable via discovery:get-nodes
//     but the broker pushes NO discovery:nodes-changed;
//   - on discovery:subscribe: an ack plus an immediate baseline
//     discovery:nodes-changed snapshot;
//   - while subscribed: a newly-advertised node produces a live
//     discovery:nodes-changed push;
//   - after discovery:unsubscribe: a further newly-advertised node is
//     still discovered (visible via get-nodes) but is NOT pushed.
//
// The broker binary is built by TestMain.
package tests

import (
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
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
)

// availableNode mirrors the broker's external camelCase wire shape (see
// AvailableNode in nvpair-ui-broker).
type availableNode struct {
	ID             string                    `json:"id"`
	Name           string                    `json:"name"`
	IPAddress      string                    `json:"ipAddress"`
	Port           int                       `json:"port"`
	LastSeen       int64                     `json:"lastSeen"`
	Models         []string                  `json:"models,omitempty"`
	ModelsByEngine map[string][]string       `json:"modelsByEngine,omitempty"`
	LoadedByEngine map[string][]string       `json:"loadedByEngine,omitempty"`
	Policy         *nodepolicy.Advertisement `json:"policy,omitempty"`
}

type availableNodesResult struct {
	Nodes []availableNode `json:"nodes"`
}

type subscriptionResult struct {
	Subscribed bool `json:"subscribed"`
}

func startBroker(t *testing.T, extraArgs ...string) (stdin io.WriteCloser, msgs <-chan jsonrpc.Message, cleanup func()) {
	t.Helper()
	args := append([]string{"--scanner-path", scannerBin, "--cluster-dir", t.TempDir()}, extraArgs...)
	cmd := exec.Command(brokerBin, args...)
	cmd.Env = privateEngineEnv(t, t.TempDir())
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
		case <-time.After(5 * time.Second): // broker also tears down the scanner (2s grace)
			cmd.Process.Kill()
			<-done
		}
	}
}

func sendReq(t *testing.T, w io.Writer, id int, method string) {
	t.Helper()
	req := fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":%q}`, id, method) + "\n"
	if _, err := w.Write([]byte(req)); err != nil {
		t.Fatalf("write %s request: %v", method, err)
	}
}

// Dispatch before TestMain so the broker's worker arguments do not become
// testing flags and the scanner fixture does not rebuild the service bundle.
func init() {
	if fixture := os.Getenv("NVPAIR_TEST_SUBSCRIPTION_SCANNER"); fixture != "" && fixture == os.Args[0] {
		if err := runSubscriptionScanner(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
}

func runSubscriptionScanner() error {
	if os.Getenv("NVPAIR_TEST_PA_SCANNER") != "" {
		return runPausedAdvertisementScanner()
	}
	codec := jsonrpc.NewCodec(struct {
		io.Reader
		io.Writer
	}{os.Stdin, os.Stdout})
	nodes := make(map[string]noderec.DirectoryNode)
	var mu sync.Mutex
	ln, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", os.Getenv("NVPAIR_TEST_SUBSCRIPTION_PORT")))
	if err != nil {
		return err
	}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var frame jsonrpc.Message
		if err := json.NewDecoder(r.Body).Decode(&frame); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var event noderec.NodeEvent
		if err := json.Unmarshal(frame.Params, &event); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		mu.Lock()
		defer mu.Unlock()
		switch frame.Method {
		case noderec.NotifyNodeDiscovered, noderec.NotifyNodeUpdated:
			nodes[event.Node.HostUUID] = event.Node
		case noderec.NotifyNodeRemoved:
			delete(nodes, event.Node.HostUUID)
		default:
			http.Error(w, "unexpected fixture event "+frame.Method, http.StatusBadRequest)
			return
		}
		if err := codec.Notify(frame.Method, event); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})}
	serveDone := make(chan error, 1)
	go func() { serveDone <- srv.Serve(ln) }()
	defer srv.Close()
	if err := codec.Notify("ready", map[string]string{"version": "subscription-fixture"}); err != nil {
		return err
	}
	for {
		frame, err := codec.Read()
		if err == io.EOF {
			if err := srv.Close(); err != nil {
				return err
			}
			select {
			case err := <-serveDone:
				if err != http.ErrServerClosed {
					return err
				}
				return nil
			case <-time.After(time.Second):
				return fmt.Errorf("fixture scanner HTTP server did not stop")
			}
		}
		if err != nil {
			return err
		}
		if !frame.IsRequest() {
			continue
		}
		if frame.Method == noderec.MethodGetNodes {
			mu.Lock()
			snapshot := noderec.GetNodesResult{Nodes: make([]noderec.DirectoryNode, 0, len(nodes))}
			for _, node := range nodes {
				snapshot.Nodes = append(snapshot.Nodes, node)
			}
			mu.Unlock()
			if err := codec.Respond(frame.ID, snapshot); err != nil {
				return err
			}
		} else if err := codec.Respond(frame.ID, map[string]bool{"ok": true}); err != nil {
			return err
		}
	}
}

// startSubscriptionBroker starts a broker whose scanner is the test binary's
// mDNS-free stub, with the given service-port plan and extra broker arguments.
func startSubscriptionBroker(t *testing.T, ports fixtureServicePorts, extraArgs ...string) (*interopProcess, func(string, string, int)) {
	t.Helper()
	base := t.TempDir()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	fixture := filepath.Join(testbin.Dir(t, "tests", "subscription-scanner", "subscription-scanner"), testbin.Exe("subscription-scanner"))
	if err := os.WriteFile(fixture, data, 0o700); err != nil {
		t.Fatalf("copy subscription scanner to %q (a previous process may still be running): %v", fixture, err)
	}
	port := freePort(t)
	args := append([]string{"--scanner-path", fixture, "--cluster-dir", filepath.Join(base, "cluster")}, ports.args()...)
	p := startInteropProcess(t, base, brokerBin, port, []string{
		"NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + fixture,
		"NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(port),
	}, append(args, extraArgs...)...)
	canonical, err := canonicalServicePath(fixture)
	if err != nil {
		t.Fatal(err)
	}
	p.paths = append(p.paths, canonical)
	client := &http.Client{Timeout: 2 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), 5*time.Second)
	return p, func(method, instance string, nodeInfoPort int) {
		t.Helper()
		event := noderec.NodeEvent{Node: noderec.DirectoryNode{
			HostUUID: instance + "-uuid", Name: instance, IP: "127.0.0.1",
			Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: nodeInfoPort}},
		}}
		frame := jsonrpc.Message{JSONRPC: "2.0", Method: method, Params: mustMarshal(t, event)}
		resp, err := client.Post(fmt.Sprintf("http://127.0.0.1:%d", port), "application/json", strings.NewReader(string(mustMarshal(t, frame))))
		if err != nil {
			t.Fatalf("publish fixture scanner event: %v; stderr: %s", err, p.output.String())
		}
		body, err := io.ReadAll(resp.Body)
		if closeErr := resp.Body.Close(); closeErr != nil {
			t.Errorf("close scanner event response: %v", closeErr)
		}
		if err != nil || resp.StatusCode != http.StatusOK {
			t.Fatalf("scanner event response status=%d body=%q err=%v; stderr: %s", resp.StatusCode, body, err, p.output.String())
		}
	}
}

func containsNode(nodes []availableNode, id string) bool {
	for _, n := range nodes {
		if n.ID == id {
			return true
		}
	}
	return false
}

// pollNodeListed repeatedly issues discovery:get-nodes until `instance`
// appears in the snapshot. Every discovery:nodes-changed seen in the
// meantime is passed to allowPush; if allowPush returns false the test
// fails (used to assert the stream is silent while unsubscribed).
func pollNodeListed(t *testing.T, stdin io.Writer, msgs <-chan jsonrpc.Message, instance string, baseReqID int, timeout time.Duration, allowPush func([]availableNode) bool) {
	t.Helper()
	deadline := time.After(timeout)
	ticker := time.NewTicker(1500 * time.Millisecond)
	defer ticker.Stop()

	id := baseReqID
	sendReq(t, stdin, id, "discovery:get-nodes")
	for {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatal("broker stream closed unexpectedly")
			}
			switch {
			case msg.Method == "discovery:nodes-changed":
				var nodes []availableNode
				if err := json.Unmarshal(msg.Params, &nodes); err != nil {
					t.Fatalf("unmarshal discovery:nodes-changed: %v\nraw: %s", err, msg.Params)
				}
				if allowPush != nil && !allowPush(nodes) {
					t.Fatalf("unexpected discovery:nodes-changed: %s", msg.Params)
				}
			case msg.Method == "" && msg.ID != nil:
				var res availableNodesResult
				if json.Unmarshal(msg.Result, &res) == nil && containsNode(res.Nodes, instance) {
					return
				}
			}
		case <-ticker.C:
			id++
			sendReq(t, stdin, id, "discovery:get-nodes")
		case <-deadline:
			t.Fatalf("timed out (%s) waiting for %q in discovery:get-nodes", timeout, instance)
		}
	}
}

// waitForPushContaining blocks until a discovery:nodes-changed carrying
// `instance` arrives, failing on timeout.
func waitForPushContaining(t *testing.T, msgs <-chan jsonrpc.Message, instance string, timeout time.Duration) {
	t.Helper()
	to := time.After(timeout)
	for {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatalf("broker stream closed before push containing %q", instance)
			}
			if msg.Method == "discovery:nodes-changed" {
				var nodes []availableNode
				if err := json.Unmarshal(msg.Params, &nodes); err != nil {
					t.Fatalf("unmarshal discovery:nodes-changed: %v\nraw: %s", err, msg.Params)
				}
				if containsNode(nodes, instance) {
					return
				}
			}
		case <-to:
			t.Fatalf("timed out (%s) waiting for discovery:nodes-changed containing %q", timeout, instance)
		}
	}
}

func TestCrossProcessBrokerSubscription(t *testing.T) {
	const (
		inst1 = "xproc-broker-1"
		inst2 = "xproc-broker-2"
		inst3 = "xproc-broker-3"
		port1 = 14401
		port2 = 14402
		port3 = 14403
	)

	p, publish := startSubscriptionBroker(t, reserveFixtureServicePorts(t))
	stdin, msgs := p.in, p.msgs

	waitForMethod(t, msgs, "app:ready", 5*time.Second)
	publish(noderec.NotifyNodeDiscovered, inst1, port1)
	t.Log("broker ready")

	// Phase 1 — unsubscribed: the node must be visible via the on-demand
	// snapshot, but the broker must push NOTHING.
	pollNodeListed(t, stdin, msgs, inst1, 1, 25*time.Second, func([]availableNode) bool { return false })
	t.Log("phase 1 OK: node visible via get-nodes, no push while unsubscribed")

	// Phase 2 — subscribe: expect an ack {subscribed:true} and an
	// immediate baseline snapshot containing inst1.
	sendReq(t, stdin, 100, "discovery:subscribe")
	gotAck, gotBaseline := false, false
	deadline := time.After(8 * time.Second)
	for !(gotAck && gotBaseline) {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatal("broker stream closed waiting for subscribe ack/baseline")
			}
			switch {
			case msg.Method == "" && msg.ID != nil:
				var sr subscriptionResult
				if json.Unmarshal(msg.Result, &sr) == nil {
					if !sr.Subscribed {
						t.Errorf("subscribe ack subscribed=false, want true")
					}
					gotAck = true
				}
			case msg.Method == "discovery:nodes-changed":
				var nodes []availableNode
				if err := json.Unmarshal(msg.Params, &nodes); err != nil {
					t.Fatalf("unmarshal baseline: %v\nraw: %s", err, msg.Params)
				}
				if !containsNode(nodes, inst1) {
					t.Errorf("baseline snapshot missing %q: %s", inst1, msg.Params)
				}
				gotBaseline = true
			}
		case <-deadline:
			t.Fatalf("timed out waiting for subscribe ack=%v baseline=%v", gotAck, gotBaseline)
		}
	}
	t.Log("phase 2 OK: subscribe acked and baseline snapshot received")

	// Phase 3 — subscribed: a newly advertised node must be pushed.
	publish(noderec.NotifyNodeDiscovered, inst2, port2)
	waitForPushContaining(t, msgs, inst2, 25*time.Second)
	t.Log("phase 3 OK: live push received for new node while subscribed")

	// Phase 4 — unsubscribe: expect an ack {subscribed:false}.
	sendReq(t, stdin, 200, "discovery:unsubscribe")
	ackd := false
	deadline = time.After(8 * time.Second)
	for !ackd {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatal("broker stream closed waiting for unsubscribe ack")
			}
			// Discovery:nodes-changed frames may still be in flight here
			// for inst1/inst2 from just before the unsubscribe took
			// effect; ignore them and wait for the ack.
			if msg.Method == "" && msg.ID != nil {
				var sr subscriptionResult
				if json.Unmarshal(msg.Result, &sr) == nil {
					if sr.Subscribed {
						t.Errorf("unsubscribe ack subscribed=true, want false")
					}
					ackd = true
				}
			}
		case <-deadline:
			t.Fatal("timed out waiting for unsubscribe ack")
		}
	}
	t.Log("phase 4 OK: unsubscribe acked")

	// Phase 5 — unsubscribed again: a node advertised now must still be
	// discovered (visible via get-nodes) but must NOT be pushed. Keying
	// the negative assertion on inst3 (which only exists from now on)
	// makes it immune to any pre-unsubscribe push still draining.
	publish(noderec.NotifyNodeDiscovered, inst3, port3)
	pollNodeListed(t, stdin, msgs, inst3, 300, 25*time.Second, func(nodes []availableNode) bool {
		return !containsNode(nodes, inst3)
	})
	t.Log("phase 5 OK: node discovered after unsubscribe with no push — gating confirmed")
}
