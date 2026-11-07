// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

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
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
)

// This personality runs at the existing tests.test image, before TestMain can
// build products. Both the unique selector and the exact self image are needed.
func init() {
	selected := os.Getenv("NVPAIR_TEST_STOP_PAUSE_HELPER")
	if selected == "" || selected != os.Args[0] {
		return
	}
	image, err := os.Executable()
	if err != nil || image != selected {
		fmt.Fprintln(os.Stderr, "stop/Pause fixture self image mismatch", err)
		os.Exit(1)
	}
	if os.Getenv("NVPAIR_TEST_STOP_PAUSE_PORT") == "" {
		// A selected helper must never fall through into TestMain and stage
		// another product bundle while the owning process test is running.
		fmt.Fprintln(os.Stderr, "unsupported stop/Pause helper invocation")
		os.Exit(2)
	}
	if err := runStopPauseModel(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	os.Exit(0)
}

const stopPauseResponseBody = `{"model":"switch-model","choices":[{"index":0,"message":{"role":"assistant","content":"finished"},"finish_reason":"stop"}]}` + "\n"

// Native fake only: production owns lifetime, admission and availability. The
// parent control request holds execution until release or upstream cancellation.
func runStopPauseModel() error {
	ln, err := net.Listen("tcp", "127.0.0.1:"+os.Getenv("NVPAIR_TEST_STOP_PAUSE_PORT"))
	if err != nil {
		return err
	}
	defer ln.Close()
	client := &http.Client{Timeout: 90 * time.Second}
	defer client.CloseIdleConnections()
	get := func(ctx context.Context, endpoint string, into any) error {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			return err
		}
		res, err := client.Do(req)
		if err != nil {
			return err
		}
		defer res.Body.Close()
		if res.StatusCode != http.StatusOK {
			return fmt.Errorf("native fixture control HTTP %d", res.StatusCode)
		}
		if into != nil {
			return json.NewDecoder(res.Body).Decode(into)
		}
		_, err = io.Copy(io.Discard, res.Body)
		return err
	}
	started := os.Getenv("NVPAIR_TEST_SWITCH_CONTROL") + "/started?engine=llamacpp&pid=" + strconv.Itoa(os.Getpid())
	if err := get(context.Background(), started, nil); err != nil {
		return err
	}
	control := os.Getenv("NVPAIR_TEST_STOP_PAUSE_CONTROL")
	var resident atomic.Bool
	resident.Store(true)
	return http.Serve(ln, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/models", "/v1/models":
			state := "unloaded"
			if resident.Load() {
				state = "loaded"
			}
			if err := json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{map[string]any{"id": "switch-model", "status": map[string]string{"value": state}}}}); err != nil {
				return
			}
		case "/models/load", "/models/unload":
			var body struct {
				Model string `json:"model"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Model != "switch-model" {
				http.Error(w, "invalid native model operation", http.StatusBadRequest)
				return
			}
			if err := get(r.Context(), control+r.URL.Path, nil); err != nil {
				http.Error(w, "native control unavailable", http.StatusBadGateway)
				return
			}
			resident.Store(r.URL.Path == "/models/load")
			fmt.Fprintln(w, `{}`)
		case "/v1/chat/completions":
			var body struct {
				Model  string `json:"model"`
				Stream bool   `json:"stream"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Model != "switch-model" {
				http.Error(w, "invalid native inference", http.StatusBadRequest)
				return
			}
			var gate struct {
				ID     int  `json:"id"`
				Hold   bool `json:"hold"`
				Prefix bool `json:"prefix"`
			}
			if err := get(r.Context(), control+"/begin", &gate); err != nil {
				return
			}
			resident.Store(true)
			prefixBytes := 0
			if gate.Prefix && !body.Stream {
				w.Header().Set("Content-Length", strconv.Itoa(len(stopPauseResponseBody)))
				prefixBytes = len(`{"model":"switch-model","choices":[`)
				if _, err := io.WriteString(w, stopPauseResponseBody[:prefixBytes]); err != nil {
					return
				}
				if err := http.NewResponseController(w).Flush(); err != nil {
					return
				}
			}
			if body.Stream {
				w.Header().Set("Content-Type", "text/event-stream")
				if _, err := io.WriteString(w, "data: {\"id\":\"fixture\",\"object\":\"chat.completion.chunk\",\"model\":\"switch-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"},\"finish_reason\":null}]}\n\n"); err != nil {
					return
				}
				if err := http.NewResponseController(w).Flush(); err != nil {
					return
				}
			}
			if gate.Hold {
				if err := get(r.Context(), control+"/hold?id="+strconv.Itoa(gate.ID), nil); err != nil {
					return
				}
			}
			if body.Stream {
				fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
			} else {
				fmt.Fprint(w, stopPauseResponseBody[prefixBytes:])
			}
		default:
			fmt.Fprintln(w, `{"version":"fixture","role":"router"}`)
		}
	}))
}

type stopPauseGate struct {
	release, held, departed chan struct{}
	once                    sync.Once
}

type stopPauseNative struct {
	server         *httptest.Server
	mu             sync.Mutex
	hold           bool
	bodyPrefix     bool
	requests       []*stopPauseGate
	loads, unloads int
	resident       bool
}

func newStopPauseNative(t *testing.T, e *engineProcessFixture) *stopPauseNative {
	t.Helper()
	n := &stopPauseNative{resident: true}
	n.server = newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n.mu.Lock()
		switch r.URL.Path {
		case "/begin":
			gate := &stopPauseGate{release: make(chan struct{}), held: make(chan struct{}), departed: make(chan struct{})}
			n.requests = append(n.requests, gate)
			id, hold, prefix := len(n.requests), n.hold, n.bodyPrefix
			n.mu.Unlock()
			if err := json.NewEncoder(w).Encode(map[string]any{"id": id, "hold": hold, "prefix": prefix}); err != nil {
				t.Errorf("encode native gate: %v", err)
			}
		case "/hold":
			id, err := strconv.Atoi(r.URL.Query().Get("id"))
			if err != nil || id < 1 || id > len(n.requests) {
				n.mu.Unlock()
				http.Error(w, "invalid gate", http.StatusBadRequest)
				return
			}
			gate := n.requests[id-1]
			n.mu.Unlock()
			close(gate.held)
			defer close(gate.departed)
			select {
			case <-gate.release:
			case <-r.Context().Done():
			}
		case "/models/load":
			n.loads++
			n.resident = true
			n.mu.Unlock()
		case "/models/unload":
			n.unloads++
			n.resident = false
			n.mu.Unlock()
		default:
			n.mu.Unlock()
			http.Error(w, "unknown native fixture event", http.StatusNotFound)
		}
	}))
	t.Cleanup(func() {
		n.mu.Lock()
		for _, gate := range n.requests {
			gate.once.Do(func() { close(gate.release) })
		}
		n.mu.Unlock()
		n.server.Close()
	})
	image, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(jointAppDir(e.base), "engines", "llamacpp.json")
	var manifest map[string]json.RawMessage
	task2Decode(t, readPolicyBytes(t, path), &manifest)
	var rt map[string]any
	task2Decode(t, manifest["runtime"], &rt)
	rt["env"] = map[string]string{
		"NVPAIR_TEST_SWITCH_ENGINE": "", "NVPAIR_TEST_SWITCH_CONTROL": e.control.URL,
		"NVPAIR_TEST_STOP_PAUSE_HELPER": image, "NVPAIR_TEST_STOP_PAUSE_PORT": "{port}",
		"NVPAIR_TEST_STOP_PAUSE_CONTROL": n.server.URL,
	}
	manifest["runtime"] = mustMarshal(t, rt)
	var platforms map[string]map[string]json.RawMessage
	task2Decode(t, manifest["platforms"], &platforms)
	for _, platform := range platforms {
		platform["runtime"] = mustMarshal(t, rt)
	}
	manifest["platforms"] = mustMarshal(t, platforms)
	jointWriteJSON(t, path, manifest)
	return n
}

func (n *stopPauseNative) holding(hold bool) { n.mu.Lock(); n.hold = hold; n.mu.Unlock() }

func (n *stopPauseNative) gate(t *testing.T, index int) *stopPauseGate {
	t.Helper()
	var gate *stopPauseGate
	task1Eventually(t, "native inference entered", 15*time.Second, func() bool {
		n.mu.Lock()
		defer n.mu.Unlock()
		if len(n.requests) >= index {
			gate = n.requests[index-1]
		}
		return gate != nil
	})
	stopPauseBarrier(t, gate.held, "native inference held")
	return gate
}

func (n *stopPauseNative) assertCounts(t *testing.T, requests, loads, unloads int) {
	t.Helper()
	n.mu.Lock()
	defer n.mu.Unlock()
	if len(n.requests) != requests || n.loads != loads || n.unloads != unloads {
		t.Fatalf("native inference/load/unload=%d/%d/%d, want %d/%d/%d", len(n.requests), n.loads, n.unloads, requests, loads, unloads)
	}
}

func stopPauseBarrier(t *testing.T, ch <-chan struct{}, description string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(35 * time.Second):
		t.Fatalf("no positive barrier: %s", description)
	}
}

// The stop/Pause cases retain the service and real inventory fixture after
// removal of the binding-only process cases. Native HTTP and discovery are fake.
type stopPauseServiceFixture struct {
	service *remoteServiceFixture
	engine  *engineProcessFixture
	client  *serviceRPCClient
	args    []string
	nodeID  string
}

func newStopPauseServiceFixture(t *testing.T, s *remoteServiceFixture, configure func(*engineProcessFixture)) *stopPauseServiceFixture {
	t.Helper()
	e := prepareEngineProcessFixture(t, s.configDir, s.ports, s.scannerPath)
	if configure != nil {
		configure(e)
	}
	// The service owns a separate config root and cluster root. Its broker's
	// settings journal is beside clusterDir, rather than inside configDir.
	var journal map[string]json.RawMessage
	task2Decode(t, readPolicyBytes(t, filepath.Join(s.configDir, "engine-settings-operations.json")), &journal)
	jointWriteJSON(t, filepath.Join(filepath.Dir(s.clusterDir), "engine-settings-operations.json"), journal)
	facades := map[string]int{"llamacpp-custom": e.facadePort, "llamacpp": e.targetFacadePort}
	for _, name := range engines.Names() {
		if _, exists := facades[name]; !exists {
			facades[name] = freePort(t)
		}
	}
	writePrivateBrokerEngineSettings(t, s.configDir, s.clusterDir, facades)
	// A paired standalone fixture may run concurrently in this test. Reuse the
	// registered manager observation image for this proxy personality, and keep
	// its real proxy in the service bundle so their cleanup inventories differ.
	image, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	e.proxyControl.image = filepath.Join(testbin.Dir(t, "tests", "performance-observations", "observation-manager"), testbin.Exe("observation-manager"))
	jointCopyExecutable(t, image, e.proxyControl.image)
	e.proxyControl.real = s.copyBinary(proxyBin)
	s.proxyPath = e.proxyControl.image
	for _, path := range []string{s.proxyPath, e.proxyControl.real} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		s.inventory.paths = append(s.inventory.paths, canonical)
	}
	s.facadePort = e.targetFacadePort
	s.env = append(s.env, "NVPAIR_TEST_SUBSCRIPTION_SCANNER="+s.scannerPath,
		"NVPAIR_TEST_SUBSCRIPTION_PORT="+strconv.Itoa(e.scannerPort))
	s.env = append(s.env, e.proxyControl.env()...)
	f := &stopPauseServiceFixture{service: s, engine: e, args: s.args(), nodeID: clustertrust.Open(s.clusterDir).NodeUUID()}
	if f.args[len(f.args)-2] != "--proxy-engines" || f.args[len(f.args)-1] != "llamacpp" {
		t.Fatal("service fixture no longer supplies the expected proxy argument")
	}
	f.start(t)
	t.Cleanup(func() {
		e.unblock()
		err := s.cleanupChecked()
		if t.Failed() {
			// The existing owner has joined the stderr pumps; capture before
			// removeBundle deletes the private log directory.
			logStopPauseDecisions(t, s.serviceFixture)
		}
		if err != nil {
			t.Errorf("stop/Pause service cleanup: %v", err)
			return
		}
		s.assertStopped()
		e.mu.Lock()
		identities := append([]serviceProcessIdentity(nil), e.identities...)
		e.mu.Unlock()
		for _, identity := range identities {
			e.assertProcessStopped(t, identity)
		}
		for _, port := range []int{e.sourcePort, e.targetPort, e.facadePort, e.targetFacadePort, e.scannerPort, e.proxyControl.port} {
			assertInteropListenerClosed(t, port)
		}
	})
	return f
}

func logStopPauseDecisions(t *testing.T, s *serviceFixture) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(s.dir, "logs", "broker.log"))
	if err != nil {
		t.Logf("stop/Pause permission decision log read: %v", err)
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.Contains(line, "remote management decision") {
			t.Logf("stop/Pause permission decision: %s", line)
		}
	}
}

func (f *stopPauseServiceFixture) grant(t *testing.T, peer string, allowed bool) {
	t.Helper()
	var before managementSnapshot
	task2Decode(t, f.call(t, "cluster:get-remote-management", map[string]any{}), &before)
	var after managementSnapshot
	task2Decode(t, f.call(t, "cluster:set-remote-management", map[string]any{"peerUuid": peer, "allowed": allowed, "expectedRevision": before.Revision}), &after)
	wantState := "denied"
	if allowed {
		wantState = "allowed"
	}
	for _, grant := range after.Grants {
		if grant.PeerUUID == peer && grant.Allowed == allowed && grant.State == wantState && after.Revision != before.Revision {
			return
		}
	}
	t.Fatal("service did not acknowledge the target-local management grant")
}

func (f *stopPauseServiceFixture) start(t *testing.T) {
	t.Helper()
	f.service.service = f.service.startChild(f.args...)
	f.client = f.service.readyClient()
	f.engine.proxyControl.ready(t, "llamacpp-custom", f.engine.facadePort)
	f.engine.proxyControl.ready(t, "llamacpp", f.engine.targetFacadePort)
	for _, name := range engines.Names() {
		f.call(t, "engine:get-settings", map[string]string{"engine": name})
	}
}

func (f *stopPauseServiceFixture) call(t *testing.T, method string, params any) json.RawMessage {
	t.Helper()
	if reply, handled := f.engine.proxyControl.componentCall(t, method, params); handled {
		if reply.Error != nil {
			t.Fatalf("fixture proxy %s: %+v", method, reply.Error)
		}
		return reply.Result
	}
	return f.client.mustCall(t, strconv.FormatInt(newRPCID(), 10), method, mustMarshal(t, params)).Result
}

// Relay a genuine inventory read through the existing scanner fixture. This is
// deliberately not independent production-scanner enrichment coverage.
func (f *stopPauseServiceFixture) publish(t *testing.T) noderec.DirectoryNode {
	t.Helper()
	var inventory struct {
		Models   []string                  `json:"models"`
		ByEngine map[string][]string       `json:"modelsByEngine"`
		Loaded   map[string][]string       `json:"loadedByEngine"`
		Policy   *nodepolicy.Advertisement `json:"policy"`
	}
	task1Eventually(t, "target-owned inventory and policy", 15*time.Second, func() bool {
		task2Decode(t, f.call(t, "engine:models", nil), &inventory)
		return inventory.Policy != nil
	})
	node := noderec.DirectoryNode{
		HostUUID: f.nodeID, ClusterUUID: f.nodeID, Name: "stop-pause-target-" + inventory.Policy.Revision, IP: "127.0.0.1", Trusted: true,
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{
			privateFixtureEngine(t, "llamacpp-custom").DiscoveryService: {Port: f.engine.facadePort}, noderec.ServiceLlamaCPP: {Port: f.engine.targetFacadePort},
		}, Models: inventory.Models, ModelsByEngine: inventory.ByEngine, LoadedByEngine: inventory.Loaded, Policy: inventory.Policy,
	}
	f.publishDirectory(t, node)
	for _, engine := range []string{"llamacpp-custom", "llamacpp"} {
		if len(inventory.ByEngine[engine]) != 0 {
			f.waitHealthy(t, engine)
		}
	}
	return node
}

func (f *stopPauseServiceFixture) waitHealthy(t *testing.T, engine string) {
	t.Helper()
	port, path := f.engine.targetFacadePort, "/v1/models"
	if engine == "llamacpp-custom" {
		port, path = f.engine.facadePort, "/v1/models"
	}
	if f.service.peer == nil {
		t.Fatal("health barrier requires a pinned peer client")
	}
	task1Eventually(t, "broker publishes healthy local "+engine+" to pinned ingress", 15*time.Second, func() bool {
		response, err := f.service.peer.Get(fmt.Sprintf("https://127.0.0.1:%d%s", port, path))
		if err != nil {
			t.Fatalf("pinned health barrier: %v", err)
		}
		if _, err := io.Copy(io.Discard, response.Body); err != nil {
			t.Fatal(err)
		}
		if err := response.Body.Close(); err != nil {
			t.Fatal(err)
		}
		return response.StatusCode == http.StatusOK
	})
}

func (f *stopPauseServiceFixture) publishDirectory(t *testing.T, node noderec.DirectoryNode) {
	t.Helper()
	frame := jsonrpc.Message{JSONRPC: "2.0", Method: noderec.NotifyNodeUpdated, Params: mustMarshal(t, noderec.NodeEvent{Node: node})}
	client := &http.Client{Timeout: 5 * time.Second}
	defer client.CloseIdleConnections()
	response, err := client.Post(fmt.Sprintf("http://127.0.0.1:%d", f.engine.scannerPort), "application/json", bytes.NewReader(mustMarshal(t, frame)))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(io.Discard, response.Body); err != nil {
		t.Fatal(err)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatalf("scanner publication HTTP %d", response.StatusCode)
	}
	for _, engine := range []string{"llamacpp-custom", "llamacpp"} {
		identity := privateFixtureEngine(t, engine)
		task1Eventually(t, "proxy observed target inventory for "+engine, 10*time.Second, func() bool {
			var result struct {
				Nodes []struct {
					ID     string   `json:"id"`
					Host   string   `json:"host"`
					Port   int      `json:"port"`
					Models []string `json:"models"`
				} `json:"nodes"`
			}
			task2Decode(t, f.call(t, identity.ComponentName()+":nodes/list", nil), &result)
			for _, got := range result.Nodes {
				if got.ID == node.HostUUID && got.Host == node.Name && got.Port == node.Services[identity.DiscoveryService].Port && slices.Equal(got.Models, node.ModelsByEngine[engine]) {
					return true
				}
			}
			return false
		})
	}
}

type stopPauseResponse struct {
	status int
	header http.Header
	body   struct {
		Model   string `json:"model"`
		Code    string `json:"code"`
		Reason  string `json:"reason"`
		Error   string `json:"error"`
		Choices []struct {
			Index   int `json:"index"`
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	err error
}

func stopPauseHTTP(ctx context.Context, client *http.Client, endpoint string) stopPauseResponse {
	var result stopPauseResponse
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(`{"model":"switch-model","stream":false,"messages":[]}`))
	if err != nil {
		result.err = err
		return result
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-PAIR-Peer-Id", "management-service-allowed")
	response, err := client.Do(request)
	if err != nil {
		result.err = err
		return result
	}
	result.status, result.header = response.StatusCode, response.Header.Clone()
	result.err = json.NewDecoder(io.LimitReader(response.Body, 64<<10)).Decode(&result.body)
	if err := response.Body.Close(); result.err == nil {
		result.err = err
	}
	return result
}

func (f *stopPauseServiceFixture) request(t *testing.T, client *http.Client, port int, pinned bool) <-chan stopPauseResponse {
	t.Helper()
	scheme := "http"
	if pinned {
		scheme = "https"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Second)
	t.Cleanup(cancel)
	done := make(chan stopPauseResponse, 1)
	go func() {
		done <- stopPauseHTTP(ctx, client, fmt.Sprintf("%s://127.0.0.1:%d/v1/chat/completions", scheme, port))
	}()
	return done
}

func stopPauseAwait(t *testing.T, done <-chan stopPauseResponse) stopPauseResponse {
	t.Helper()
	select {
	case response := <-done:
		if response.err != nil {
			t.Fatalf("stop/Pause inference transport/decode: %v", response.err)
		}
		return response
	case <-time.After(100 * time.Second):
		t.Fatal("stop/Pause inference exceeded HTTP request bound")
	}
	return stopPauseResponse{}
}

func assertStopPauseSuccess(t *testing.T, response stopPauseResponse, engine, node string) {
	t.Helper()
	if response.status != http.StatusOK || response.body.Error != "" || response.body.Model != "switch-model" || len(response.body.Choices) != 1 || response.body.Choices[0].Index != 0 || response.body.Choices[0].Message.Role != "assistant" || response.body.Choices[0].Message.Content != "finished" || response.body.Choices[0].FinishReason != "stop" || response.header.Get(nodepolicy.EngineHeader) != engine || response.header.Get(nodepolicy.NodeHeader) != node || response.header.Get(nodepolicy.ModelHeader) != "switch-model" {
		t.Fatalf("stop/Pause success HTTP=%d engine=%q node=%q model=%q choices=%d admission=%q code=%q reason=%q error=%q", response.status, response.header.Get(nodepolicy.EngineHeader), response.header.Get(nodepolicy.NodeHeader), response.body.Model, len(response.body.Choices), response.header.Get(nodepolicy.AdmissionHeader), response.body.Code, response.body.Reason, response.body.Error)
	}
}

func (f *stopPauseServiceFixture) assertEffects(t *testing.T, sourceStarts, targetStarts, sourceInference, targetInference int) {
	t.Helper()
	e := f.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.effectError != nil {
		t.Fatalf("fixture effect observation: %v", e.effectError)
	}
	if e.checkSourceExit && e.sourceAliveAtTarget {
		t.Fatal("target started before actual source process exit")
	}
	if len(e.starts["llamacpp-custom"]) != sourceStarts || len(e.starts["llamacpp"]) != targetStarts || e.inferences["llamacpp-custom"] != sourceInference || e.inferences["llamacpp"] != targetInference {
		t.Fatalf("engine effects starts=%v inference=%v, want starts=%d/%d inference=%d/%d", e.starts, e.inferences, sourceStarts, targetStarts, sourceInference, targetInference)
	}
}

type stopPauseFixture struct {
	*stopPauseServiceFixture
	native *stopPauseNative
}

// Keep the service plan occupied while the reused engine fixture allocates
// native/facade ports. Independently released plans can otherwise reuse an ec
// port for a fake model before either process starts. There is no bind retry.
func holdStopPauseServicePorts(t *testing.T, ports fixtureServicePorts) func() {
	t.Helper()
	var listeners []net.Listener
	var once sync.Once
	release := func() {
		once.Do(func() {
			for _, listener := range listeners {
				if err := listener.Close(); err != nil {
					t.Errorf("release service-plan exclusion: %v", err)
				}
			}
		})
	}
	t.Cleanup(release)
	for _, port := range []int{ports.nodeInfo, ports.errors, ports.workload, ports.clusterManager, ports.engineManagerHTTP, ports.engineControl} {
		listener, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
		if err != nil {
			t.Fatalf("reserve private service-plan exclusion: %v", err)
		}
		listeners = append(listeners, listener)
	}
	return release
}

func newStopPauseFixture(t *testing.T) *stopPauseFixture {
	t.Helper()
	s := newRemoteServiceFixture(t)
	release := holdStopPauseServicePorts(t, s.ports)
	var native *stopPauseNative
	f := newStopPauseServiceFixture(t, s, func(e *engineProcessFixture) {
		native = newStopPauseNative(t, e)
		release()
	})
	return &stopPauseFixture{stopPauseServiceFixture: f, native: native}
}

func (f *stopPauseFixture) startModel(t *testing.T) {
	t.Helper()
	f.call(t, "engine:start", map[string]string{"engine": "llamacpp"})
	assertStopPauseNativeLifetime(t, f.engine)
	var inventory struct {
		Loaded map[string][]string `json:"loadedByEngine"`
	}
	task1Eventually(t, "manager reports actual native residency", 15*time.Second, func() bool {
		task2Decode(t, f.call(t, "engine:models", nil), &inventory)
		return slices.Contains(inventory.Loaded["llamacpp"], "switch-model")
	})
	f.publish(t)
}

func assertStopPauseNativeLifetime(t *testing.T, e *engineProcessFixture) {
	t.Helper()
	image, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := canonicalServicePath(image)
	if err != nil {
		t.Fatal(err)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	identities := e.engineIdentities["llamacpp"]
	if e.effectError != nil || len(identities) != 1 {
		t.Fatalf("native lifetime observation=%+v err=%v", identities, e.effectError)
	}
	if identity := identities[0]; identity.PID <= 0 || identity.Created <= 0 || identity.Path != canonical {
		t.Fatalf("native lifetime lacks exact PID/path/creation evidence: %+v", identity)
	}
}

func (f *stopPauseFixture) availability(t *testing.T, state nodepolicy.Availability, mode string) json.RawMessage {
	t.Helper()
	params := map[string]any{"state": state}
	if mode != "" {
		params["pauseMode"] = mode
	}
	raw := f.call(t, "node:set-availability", params)
	var result struct {
		Availability nodepolicy.Availability `json:"availability"`
		PausedBy     nodepolicy.PauseOrigin  `json:"pausedBy"`
	}
	task2Decode(t, raw, &result)
	if result.Availability != state || state == nodepolicy.Available && result.PausedBy != "" {
		t.Fatalf("local availability result=%+v, want %s", result, state)
	}
	return raw
}

func (f *stopPauseFixture) assertPaused(t *testing.T, origin nodepolicy.PauseOrigin) {
	t.Helper()
	// First read after the successful Pause reply: do not poll for a late
	// process exit or issue another stop that could repair premature success.
	f.engine.mu.Lock()
	identities := append([]serviceProcessIdentity(nil), f.engine.identities...)
	f.engine.mu.Unlock()
	for _, identity := range identities {
		stopped, err := f.engine.processStopped(context.Background(), identity)
		if err != nil || !stopped {
			t.Fatalf("Pause completed before native lifetime exited: identity=%+v stopped=%t err=%v", identity, stopped, err)
		}
	}
	snapshot := fetchServicePolicy(t, f.client)
	if snapshot.Availability != nodepolicy.Paused || snapshot.Policy.Availability != nodepolicy.Paused || snapshot.Policy.PausedBy != origin || snapshot.Activation != "active" || snapshot.ActiveRevision != snapshot.Policy.Revision {
		t.Fatalf("target pause readback=%+v", snapshot)
	}
	assertStoredPolicy(t, f.service.policyPath(), snapshot.Policy)
	// Inventory corroborates current-manager state; the exact child lifetimes
	// above establish physical exit. Never manufacture proof with another Sleep.
	var inventory struct {
		Engines []struct {
			Engine              string `json:"engine"`
			Running             *bool  `json:"running"`
			TerminationVerified bool   `json:"terminationVerified"`
		} `json:"engines"`
	}
	task2Decode(t, f.call(t, "engine:get-installed", nil), &inventory)
	seen := make(map[string]bool)
	for _, status := range inventory.Engines {
		if !slices.Contains(privateFixtureEngineNames(), status.Engine) || seen[status.Engine] || status.Running == nil || *status.Running {
			t.Fatalf("current-manager shutdown inventory=%+v", status)
		}
		seen[status.Engine] = true
	}
	if len(seen) != len(privateFixtureEngineNames()) {
		t.Fatalf("current-manager inventory incomplete: %v", seen)
	}
	if fetchServicePolicy(t, f.client).Availability != nodepolicy.Paused {
		t.Fatal("manager proof retired during target readback")
	}
}

func (f *stopPauseFixture) peerHTTP(t *testing.T, client *http.Client, path string, params any, wantStatus int) []byte {
	t.Helper()
	request, err := http.NewRequest(http.MethodPost, fmt.Sprintf("https://127.0.0.1:%d%s", f.service.ports.engineControl, path), bytes.NewReader(mustMarshal(t, params)))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		t.Fatalf("pinned relay %s: %v", path, err)
	}
	data, readErr := io.ReadAll(io.LimitReader(response.Body, 256<<10))
	closeErr := response.Body.Close()
	if readErr != nil || closeErr != nil || response.StatusCode != wantStatus {
		t.Fatalf("pinned relay %s HTTP=%d want=%d read=%v close=%v", path, response.StatusCode, wantStatus, readErr, closeErr)
	}
	return data
}

func stopPauseRefusal(t *testing.T, raw []byte, code string) {
	t.Helper()
	var body struct {
		Code  string `json:"code"`
		Error string `json:"error"`
	}
	task2Decode(t, raw, &body)
	if body.Code != code || body.Error == "" {
		t.Fatalf("refusal=%+v, want %s", body, code)
	}
}

// Each case acquires one service bundle; two-node cases add the already
// registered standalone broker fixture, never another service-bundle slot.
func TestStopPauseControlsProcess(t *testing.T) {
	t.Run("remote immediate and ungranted pause have zero physical effects", testStopPauseRemoteRefusals)
	t.Run("local configured pause fences both managing peers", func(t *testing.T) { testStopPauseFence(t, "configured", false) })
	t.Run("local immediate pause fences both managing peers", func(t *testing.T) { testStopPauseFence(t, "immediate", false) })
	t.Run("managing peer configured pause fences other management", func(t *testing.T) { testStopPauseFence(t, "configured", true) })
	t.Run("peer configured pause and resume use the public relay", testStopPausePeerResume)
	t.Run("active local execution stops and later demand cannot execute", func(t *testing.T) { testStopPauseActive(t, false) })
	t.Run("active pinned peer execution stops without onward routing", func(t *testing.T) { testStopPauseActive(t, true) })
	t.Run("queued local demand is refused without later execution", func(t *testing.T) { testStopPauseQueue(t, false) })
	t.Run("queued pinned peer demand is refused without later execution", func(t *testing.T) { testStopPauseQueue(t, true) })
	t.Run("configured finish preserves execution until explicit completion", func(t *testing.T) { testStopPauseDrain(t, false) })
	t.Run("immediate escalation bypasses the held configured drain", func(t *testing.T) { testStopPauseDrain(t, true) })
	t.Run("paused restart preserves saved Off and resumes only saved On", func(t *testing.T) { testStopPauseRestart(t) })
	t.Run("precommit local pause falls back at the origin", testStopPauseFallback)
	t.Run("committed local stream fails and only a new client request retries", func(t *testing.T) { testStopPauseCommitted(t, false) })
	t.Run("committed peer stream fails at origin without replay", func(t *testing.T) { testStopPauseCommitted(t, true) })
	t.Run("paused origin retains existing remote execution", testStopPauseRemoteExecution)
}

func testStopPausePeerResume(t *testing.T) {
	f, peer := newStopPausePair(t)
	f.grant(t, peer.id, true)
	decode := func(raw json.RawMessage, state nodepolicy.Availability, origin nodepolicy.PauseOrigin, mode nodepolicy.PauseMode) {
		t.Helper()
		var result struct {
			Availability nodepolicy.Availability `json:"availability"`
			PausedBy     nodepolicy.PauseOrigin  `json:"pausedBy"`
			PauseMode    nodepolicy.PauseMode    `json:"pauseMode"`
		}
		task2Decode(t, raw, &result)
		if result.Availability != state || result.PausedBy != origin || result.PauseMode != mode {
			t.Fatalf("peer result=%+v, want %s/%s/%s", result, state, origin, mode)
		}
	}
	pause := peer.engine.rpcOK(t, "engine:remote-availability-set", map[string]string{"nodeId": f.nodeID, "state": "paused", "pauseMode": "configured"})
	decode(pause, nodepolicy.Paused, nodepolicy.PausedByPeer, nodepolicy.PauseConfigured)
	f.assertPaused(t, nodepolicy.PausedByPeer)
	paused := fetchServicePolicy(t, f.client).Policy
	resume := peer.engine.rpcOK(t, "engine:remote-availability-set", map[string]string{"nodeId": f.nodeID, "state": "available"})
	decode(resume, nodepolicy.Available, "", "")
	got := fetchServicePolicy(t, f.client)
	if got.Availability != nodepolicy.Available || got.Policy.Availability != nodepolicy.Available ||
		got.Policy.PausedBy != "" || got.Policy.PauseMode != "" || got.Policy.Revision == paused.Revision {
		t.Fatalf("peer Resume target readback=%+v", got)
	}
	assertStoredPolicy(t, f.service.policyPath(), got.Policy)
}

func testStopPauseRemoteRefusals(t *testing.T) {
	f := newStopPauseFixture(t)
	f.startModel(t)
	f.grant(t, "management-service-peer", true)
	f.grant(t, "management-service-allowed", false)
	before := readPolicyBytes(t, f.service.policyPath())
	intentPath := filepath.Join(jointAppDir(f.service.configDir), "engine-bin", "engine-state.json")
	beforeIntent := readLiveDurableFile(t, intentPath)
	f.native.holding(true)
	client := &http.Client{Timeout: 45 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	demand := f.request(t, client, f.engine.targetFacadePort, false)
	gate := f.native.gate(t, 1)
	// The target's own payload nodeId cannot forge a local caller.
	f.peerHTTP(t, f.service.peer, "/v1/node-availability/set", map[string]any{"nodeId": f.nodeID, "state": "paused", "pauseMode": "immediate"}, http.StatusBadRequest)
	stopPauseRefusal(t, f.peerHTTP(t, f.service.allowed, "/v1/node-availability/set", map[string]any{"state": "paused"}, http.StatusConflict), "remote_management_denied")
	if !bytes.Equal(before, readPolicyBytes(t, f.service.policyPath())) || fetchServicePolicy(t, f.client).Availability != nodepolicy.Available {
		t.Fatal("refused remote requests changed target policy")
	}
	if !bytes.Equal(beforeIntent, readLiveDurableFile(t, intentPath)) {
		t.Fatal("refused remote request changed saved engine state")
	}
	f.native.assertCounts(t, 1, 0, 0)
	select {
	case <-gate.departed:
		t.Fatal("refused remote Pause cancelled existing native execution")
	case <-demand:
		t.Fatal("held request completed during refused remote Pause")
	default:
	}
	f.engine.mu.Lock()
	identity := f.engine.engineIdentities["llamacpp"][0]
	f.engine.mu.Unlock()
	if stopped, err := f.engine.processStopped(context.Background(), identity); err != nil || stopped {
		t.Fatalf("refusal stopped actual model lifetime: stopped=%t err=%v", stopped, err)
	}
	gate.once.Do(func() { close(gate.release) })
	assertStopPauseSuccess(t, stopPauseAwait(t, demand), "llamacpp", f.nodeID)
	f.native.assertCounts(t, 1, 0, 0)
}

func testStopPauseFence(t *testing.T, mode string, peerOrigin bool) {
	var f *stopPauseFixture
	var peer *stopPausePeer
	if peerOrigin {
		f, peer = newStopPausePair(t)
		f.grant(t, peer.id, true)
	} else {
		f = newStopPauseFixture(t)
		f.startModel(t)
	}
	f.grant(t, "management-service-peer", true)
	f.grant(t, "management-service-allowed", true)
	origin := nodepolicy.PausedLocally
	if peerOrigin {
		peer.engine.rpcOK(t, "engine:remote-availability-set", map[string]any{"nodeId": f.nodeID, "state": "paused"})
		origin = nodepolicy.PausedByPeer
	} else {
		f.availability(t, nodepolicy.Paused, mode)
	}
	f.assertPaused(t, origin)
	before := readPolicyBytes(t, f.service.policyPath())
	// The ec registry authorizes before executing mutation handlers. Include
	// nominally harmless remote stop/unload. Availability has its own rule.
	for _, client := range []*http.Client{f.service.peer, f.service.allowed} {
		for _, path := range []string{
			"/v1/engines/start", "/v1/engines/stop", "/v1/engines/install",
			"/v1/models/load", "/v1/models/unload", "/v1/models/pull", "/v1/models/delete", "/v1/models/copy",
			"/v1/node-policy/set",
			"/v1/engine-settings/apply",
		} {
			stopPauseRefusal(t, f.peerHTTP(t, client, path, map[string]any{"engine": "llamacpp", "model": "switch-model", "state": "available"}, http.StatusConflict), "local_veto")
		}
		// Repeating Pause is a successful no-op, including a completed local
		// Halt. Its saved immediate mode and revision must survive unchanged.
		f.peerHTTP(t, client, "/v1/node-availability/set", map[string]any{"nodeId": f.nodeID, "state": "paused"}, http.StatusOK)
		if mode == "immediate" {
			stopPauseRefusal(t, f.peerHTTP(t, client, "/v1/node-availability/set", map[string]any{"nodeId": f.nodeID, "state": "available"}, http.StatusConflict), "local_veto")
		}
		var readback struct {
			Policy       nodepolicy.PeerPolicy   `json:"policy"`
			Availability nodepolicy.Availability `json:"availability"`
		}
		task2Decode(t, f.peerHTTP(t, client, "/v1/node-policy/get", map[string]any{}, http.StatusOK), &readback)
		if readback.Availability != nodepolicy.Paused || readback.Policy.Revision != fetchServicePolicy(t, f.client).Policy.Revision {
			t.Fatalf("observing peer lacks positive target pause readback: %+v", readback)
		}
	}
	if !bytes.Equal(before, readPolicyBytes(t, f.service.policyPath())) {
		t.Fatal("peer mutations changed fenced policy")
	}
	if peerOrigin {
		peer.engine.rpcOK(t, "engine:remote-availability-set", map[string]string{"nodeId": f.nodeID, "state": "paused"})
		if !bytes.Equal(before, readPolicyBytes(t, f.service.policyPath())) {
			t.Fatal("initiating peer repeated Pause changed saved policy")
		}
	}
	for _, method := range []string{"engine:start", "engine:load-model"} {
		params := map[string]string{"engine": "llamacpp"}
		if method == "engine:load-model" {
			params["model"] = "switch-model"
		}
		response, err := f.client.call(strconv.FormatInt(newRPCID(), 10), method, mustMarshal(t, params), serviceCallTimeout)
		if err != nil || response.Error == nil || response.Error.Code != -32011 {
			t.Fatalf("local activation escaped Pause: method=%s response=%+v transport=%v", method, response, err)
		}
	}
	policy := fetchServicePolicy(t, f.client).Policy
	policy.Admission.MaxConcurrentPerModel = 3
	setServicePolicy(t, f.client, policy)
	f.call(t, "engine:stop", map[string]string{"engine": "llamacpp-custom"})
	// Explicit unload can fail because its HTTP server was physically stopped;
	// that transport failure must not be an availability veto or cause a wake.
	unload, err := f.client.call(strconv.FormatInt(newRPCID(), 10), "engine:unload-model", mustMarshal(t, map[string]string{"engine": "llamacpp", "model": "switch-model"}), serviceCallTimeout)
	if err != nil || (unload.Error != nil && unload.Error.Code == -32011) {
		t.Fatalf("local unload was fenced: response=%+v transport=%v", unload, err)
	}
	f.native.assertCounts(t, 0, 0, 0)
	f.availability(t, nodepolicy.Available, "")
	// Round-trip this target's actual public projection. jointExpectedPeer is
	// specific to the policy-joint fixture's seeded tiers and is not a generic
	// projection for this service fixture's policy.
	var current struct {
		Policy nodepolicy.PeerPolicy `json:"policy"`
	}
	task2Decode(t, f.peerHTTP(t, f.service.peer, "/v1/node-policy/get", map[string]any{}, http.StatusOK), &current)
	f.peerHTTP(t, f.service.peer, "/v1/node-policy/set", map[string]any{"policy": current.Policy, "expectedPersistedRevision": current.Policy.Revision}, http.StatusOK)
}

func testStopPauseActive(t *testing.T, peer bool) {
	f := newStopPauseFixture(t)
	f.startModel(t)
	f.native.holding(true)
	var onward atomic.Int32
	trap := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			onward.Add(1)
		}
		http.Error(w, "unexpected onward execution", http.StatusBadGateway)
	}))
	t.Cleanup(trap.Close)
	client := &http.Client{Timeout: 45 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	if peer {
		client = f.service.peer
		f.call(t, "llamacpp-proxy:node/add-manual", map[string]any{"id": "stop-pause-onward", "host": "127.0.0.1", "addresses": []string{"127.0.0.1"}, "port": portOfURL(t, trap.URL), "models": []string{"switch-model"}})
		f.call(t, "llamacpp-proxy:node/select", map[string]string{"id": "stop-pause-onward"})
	}
	done := f.request(t, client, f.engine.targetFacadePort, peer)
	gate := f.native.gate(t, 1)
	f.availability(t, nodepolicy.Paused, "immediate")
	f.assertPaused(t, nodepolicy.PausedLocally)
	stopPauseBarrier(t, gate.departed, "native execution actually cancelled")
	response := stopPauseAwait(t, done)
	if response.status == http.StatusOK {
		t.Fatalf("interrupted precommit request returned success: %+v", response)
	}
	later := stopPauseAwait(t, f.request(t, client, f.engine.targetFacadePort, peer))
	if later.status == http.StatusOK {
		t.Fatal("fenced demand succeeded")
	}
	f.native.assertCounts(t, 1, 0, 0)
	if onward.Load() != 0 {
		t.Fatalf("terminal destination forwarded onward %d times", onward.Load())
	}
}

func testStopPauseQueue(t *testing.T, pinned bool) {
	var f *stopPauseFixture
	var origin *stopPausePeer
	if pinned {
		f, origin = newStopPausePair(t)
		// Keep the real origin's fallback engine locally Off through its owner.
		origin.engine.rpcOK(t, "engine:stop", map[string]string{"engine": "llamacpp"})
	} else {
		f = newStopPauseFixture(t)
		f.startModel(t)
	}
	policy := fetchServicePolicy(t, f.client).Policy
	policy.Admission.MaxConcurrentPerModel = 1
	policy.Admission.QueueTimeoutSeconds = 120
	setServicePolicy(t, f.client, policy)
	f.native.holding(true)
	f.native.mu.Lock()
	f.native.bodyPrefix = true
	f.native.mu.Unlock()
	client := &http.Client{Timeout: 90 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	call := func(method string, params any) json.RawMessage { return f.call(t, method, params) }
	port, originID := f.engine.targetFacadePort, f.nodeID
	if pinned {
		origin.publishTarget(t, f)
		call = func(method string, params any) json.RawMessage { return origin.engine.rpcOK(t, method, params) }
		port, originID = origin.engine.targetFacadePort, origin.id
		call("llamacpp-proxy:node/select", map[string]string{"id": f.nodeID})
	}
	var subscribed struct {
		Subscribed bool `json:"subscribed"`
	}
	task2Decode(t, call("workloads:subscribe", map[string]any{}), &subscribed)
	if !subscribed.Subscribed {
		t.Fatal("origin did not acknowledge public workload subscription")
	}
	workloads := func() []departedRunInfo {
		t.Helper()
		var initial struct {
			Workloads []departedRunInfo `json:"workloads"`
		}
		task2Decode(t, call("workloads:get-initial", nil), &initial)
		return initial.Workloads
	}
	if baseline := workloads(); len(baseline) != 0 {
		t.Fatalf("origin workload baseline is not empty: %+v", baseline)
	}
	waitWorkload := func(state string, prior *departedRunInfo) departedRunInfo {
		t.Helper()
		var found departedRunInfo
		task1Eventually(t, "origin publishes own "+state+" workload", 15*time.Second, func() bool {
			for _, w := range workloads() {
				if prior != nil && w.key() == prior.key() {
					continue
				}
				if w.State != state {
					continue
				}
				if w.ID == "" || w.RunID == "" || w.OriginatedFrom != originID || w.Engine != "llamacpp" || w.Model != "switch-model" || (state == "running" && w.ScheduledOn != f.nodeID) || (state == "queued" && w.ScheduledOn != "" && w.ScheduledOn != f.nodeID) || w.CompletedAt != nil {
					t.Fatalf("invalid own workload identity/placement: %+v", w)
				}
				if (state == "running" && w.StartedAt == nil) || (state == "queued" && w.StartedAt != nil) || (prior != nil && (w.ID == prior.ID || w.RunID != prior.RunID)) {
					t.Fatalf("invalid workload lifecycle: prior=%+v current=%+v", prior, w)
				}
				found = w
				return true
			}
			return false
		})
		return found
	}
	// Retain raw bytes and declared length for the interrupted active response.
	// The shared inference helper deliberately requires a complete JSON body.
	type activeResponse struct {
		status             int
		declared           int64
		raw                []byte
		readErr, decodeErr error
	}
	request, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port), strings.NewReader(`{"model":"switch-model","stream":false,"messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	one := make(chan activeResponse, 1)
	go func() {
		response, err := client.Do(request)
		if err != nil {
			one <- activeResponse{readErr: err}
			return
		}
		raw, readErr := io.ReadAll(response.Body)
		if closeErr := response.Body.Close(); readErr == nil {
			readErr = closeErr
		}
		var body json.RawMessage
		decodeErr := json.NewDecoder(bytes.NewReader(raw)).Decode(&body)
		one <- activeResponse{response.StatusCode, response.ContentLength, raw, readErr, decodeErr}
	}()
	gate := f.native.gate(t, 1)
	running := waitWorkload("running", nil)
	two := f.request(t, client, port, false)
	queued := waitWorkload("queued", &running)
	f.native.assertCounts(t, 1, 0, 0)
	f.availability(t, nodepolicy.Paused, "immediate")
	f.assertPaused(t, nodepolicy.PausedLocally)
	stopPauseBarrier(t, gate.departed, "active request cancelled before queue could execute")
	select {
	case response := <-one:
		if response.status == http.StatusOK {
			// The first body byte commits status (routing.go:297); committed
			// output is not replayed. A truncated 200 is an honest active
			// interruption result only with incomplete body and decode errors.
			if len(response.raw) == 0 || len(response.raw) >= len(stopPauseResponseBody) || !bytes.HasPrefix([]byte(stopPauseResponseBody), response.raw) || !errors.Is(response.readErr, io.ErrUnexpectedEOF) || !errors.Is(response.decodeErr, io.ErrUnexpectedEOF) {
				t.Fatalf("active response was not an incomplete committed prefix: %+v", response)
			}
			if pinned {
				// setPeerStreamFraming uses chunked peer framing for its
				// disposition trailer, removing the native Content-Length.
				if response.declared != -1 {
					t.Fatalf("pinned response Content-Length = %d, want unknown chunked length", response.declared)
				}
			} else if response.declared != int64(len(stopPauseResponseBody)) || int64(len(response.raw)) >= response.declared {
				t.Fatalf("local response did not retain the declared native length: %+v", response)
			}
		}
	case <-time.After(100 * time.Second):
		t.Fatal("active cancellation exceeded HTTP request bound")
	}
	if response := stopPauseAwait(t, two); response.status == http.StatusOK {
		t.Fatal("queued request succeeded after Pause")
	}
	f.native.assertCounts(t, 1, 0, 0)
	if pinned {
		origin.native.assertCounts(t, 0, 0, 0)
	}
	failed := map[departedRunKey]departedRunInfo{}
	task1Eventually(t, "own active and queued workloads fail", 15*time.Second, func() bool {
		clear(failed)
		for _, w := range workloads() {
			if w.key() != running.key() && w.key() != queued.key() {
				t.Fatalf("unexpected workload after Pause: %+v", w)
			}
			if w.State == "running" || w.State == "queued" {
				continue
			}
			if w.State != "failed" || w.CompletedAt == nil || w.Error == nil || w.Model != "switch-model" || (w.key() == queued.key() && w.StartedAt != nil) {
				t.Fatalf("queued/active cancellation outcome=%+v", w)
			}
			failed[w.key()] = w
		}
		return len(failed) == 2
	})
	if !pinned {
		for _, record := range f.history(t, 2) {
			if record.State != "failed" {
				t.Fatalf("queued/active cancellation outcome=%+v", record)
			}
		}
	}
	f.native.holding(false)
	f.availability(t, nodepolicy.Available, "")
	task1Eventually(t, "Resume launches a fresh eligible engine", 30*time.Second, func() bool {
		f.engine.mu.Lock()
		defer f.engine.mu.Unlock()
		return len(f.engine.starts["llamacpp"]) == 2
	})
	f.publish(t)
	if pinned {
		origin.publishTarget(t, f)
		// This origin has no cluster-manager reverse gateway; its direct route
		// may still be in the 30 s probe cooldown after interrupted output.
		deadline := time.Now().Add(45 * time.Second)
		task1Eventually(t, "origin inference transport sees the resumed switch model", 45*time.Second, func() bool {
			remaining := time.Until(deadline)
			if remaining <= 0 {
				return false
			}
			ctx, cancel := context.WithTimeout(context.Background(), min(2*time.Second, remaining))
			defer cancel()
			request, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/v1/models", port), nil)
			if err != nil {
				t.Fatal(err)
			}
			response, err := client.Do(request)
			if err != nil {
				return false
			}
			var models struct {
				Data []struct {
					ID string `json:"id"`
				} `json:"data"`
			}
			decodeErr := json.NewDecoder(io.LimitReader(response.Body, 64<<10)).Decode(&models)
			if err := response.Body.Close(); err != nil {
				t.Fatal(err)
			}
			if response.StatusCode != http.StatusOK || decodeErr != nil {
				return false
			}
			for _, model := range models.Data {
				if model.ID == "switch-model" {
					return true
				}
			}
			return false
		})
	}
	assertStopPauseSuccess(t, stopPauseAwait(t, f.request(t, client, port, false)), "llamacpp", f.nodeID)
	f.native.assertCounts(t, 2, 0, 0)
	if pinned {
		origin.native.assertCounts(t, 0, 0, 0)
	}
	task1Eventually(t, "Resume completes only fresh demand and preserves failures", 15*time.Second, func() bool {
		preserved, completed := 0, 0
		for _, w := range workloads() {
			if previous, ok := failed[w.key()]; ok {
				if !reflect.DeepEqual(w, previous) {
					t.Fatalf("Resume rewrote failed workload: before=%+v after=%+v", previous, w)
				}
				preserved++
			} else if w.State == "completed" && w.CompletedAt != nil && w.OriginatedFrom == originID && w.Engine == "llamacpp" && w.RunID == running.RunID && w.ID != "" && w.Model == "switch-model" && w.ScheduledOn == f.nodeID {
				completed++
			} else if w.State != "running" && w.State != "queued" {
				t.Fatalf("unexpected workload after Resume: %+v", w)
			}
		}
		if completed > 1 {
			t.Fatalf("Resume replayed queued requests: completed=%d", completed)
		}
		return preserved == 2 && completed == 1
	})
	if !pinned {
		completed := 0
		for _, record := range f.history(t, 3) {
			if record.State == "completed" {
				completed++
			}
		}
		if completed != 1 {
			t.Fatalf("Resume replayed queued requests: completed=%d", completed)
		}
	}
}

type stopPauseRPCOutcome struct {
	response *jsonrpc.Message
	err      error
}

func (f *stopPauseFixture) beginPause(t *testing.T) <-chan stopPauseRPCOutcome {
	t.Helper()
	done := make(chan stopPauseRPCOutcome, 1)
	id := strconv.FormatInt(newRPCID(), 10)
	params := mustMarshal(t, map[string]string{"state": "paused"})
	go func() {
		response, err := f.client.call(id, "node:set-availability", params, 90*time.Second)
		done <- stopPauseRPCOutcome{response, err}
	}()
	return done
}

func stopPauseAwaitRPC(t *testing.T, done <-chan stopPauseRPCOutcome, superseded bool) {
	t.Helper()
	select {
	case result := <-done:
		if result.err != nil || result.response == nil {
			t.Fatalf("pause transport: %v", result.err)
		}
		if superseded {
			if result.response.Error == nil || !strings.Contains(result.response.Error.Message, "superseded") {
				t.Fatalf("configured transition was not superseded: %+v", result.response)
			}
		} else if result.response.Error != nil {
			t.Fatalf("configured pause: %+v", result.response.Error)
		}
	case <-time.After(40 * time.Second):
		t.Fatal("pause did not complete after positive release/stop barrier")
	}
}

func testStopPauseDrain(t *testing.T, escalate bool) {
	f := newStopPauseFixture(t)
	f.startModel(t)
	policy := fetchServicePolicy(t, f.client).Policy
	policy.Pause.OnActive, policy.Pause.DrainTimeoutSeconds = nodepolicy.FinishActive, 120
	setServicePolicy(t, f.client, policy)
	f.native.holding(true)
	client := &http.Client{Timeout: 90 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	demand := f.request(t, client, f.engine.targetFacadePort, false)
	gate := f.native.gate(t, 1)
	observer := f.service.readyClient()
	pause := f.beginPause(t)
	task1Eventually(t, "configured draining with held execution", 15*time.Second, func() bool {
		snapshot := fetchServicePolicy(t, observer)
		return snapshot.Availability == nodepolicy.Draining && snapshot.Policy.Availability == nodepolicy.Paused
	})
	f.engine.mu.Lock()
	identity := f.engine.engineIdentities["llamacpp"][0]
	f.engine.mu.Unlock()
	if stopped, err := f.engine.processStopped(context.Background(), identity); err != nil || stopped {
		t.Fatalf("configured finish killed held execution: stopped=%t err=%v", stopped, err)
	}
	select {
	case <-gate.departed:
		t.Fatal("configured finish cancelled held execution")
	case <-pause:
		t.Fatal("configured finish completed before execution release")
	default:
	}
	if escalate {
		// Another local connection can supersede a still-waiting transition.
		result := observer.mustCall(t, strconv.FormatInt(newRPCID(), 10), "node:set-availability", mustMarshal(t, map[string]string{"state": "paused", "pauseMode": "immediate"}))
		if result.Error != nil {
			t.Fatalf("immediate escalation: %+v", result.Error)
		}
		stopPauseBarrier(t, gate.departed, "held execution cancelled without release")
		stopPauseAwaitRPC(t, pause, true)
		if response := stopPauseAwait(t, demand); response.status == http.StatusOK {
			t.Fatal("escalated precommit inference succeeded")
		}
	} else {
		gate.once.Do(func() { close(gate.release) })
		assertStopPauseSuccess(t, stopPauseAwait(t, demand), "llamacpp", f.nodeID)
		stopPauseAwaitRPC(t, pause, false)
	}
	f.assertPaused(t, nodepolicy.PausedLocally)
	f.native.assertCounts(t, 1, 0, 0)
}

func testStopPauseRestart(t *testing.T) {
	f := newStopPauseFixture(t)
	f.startModel(t)
	f.call(t, "engine:stop", map[string]string{"engine": "llamacpp-custom"})
	policy := fetchServicePolicy(t, f.client).Policy
	profile := fixtureResidentProfile()
	// Resident configuration requires a qualification reference. This is only
	// a declaration: install no qualification registry entry, and never use it
	// as admission, loaded-model or physical shutdown evidence.
	profile.Engine, profile.Model, profile.Key, profile.DisplayName = "llamacpp", "switch-model", "switch-model", "stop-pause-resident"
	policy.Profiles = []nodepolicy.Profile{profile}
	policy.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "switch-model"}}
	policy.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "switch-model"}}
	policy = setServicePolicy(t, f.client, policy)
	f.availability(t, nodepolicy.Paused, "immediate")
	f.assertPaused(t, nodepolicy.PausedLocally)
	intentPath := filepath.Join(jointAppDir(f.service.configDir), "engine-bin", "engine-state.json")
	var before struct {
		Engines map[string]bool `json:"engines"`
	}
	task2Decode(t, readLiveDurableFile(t, intentPath), &before)
	if !before.Engines["llamacpp"] || before.Engines["llamacpp-custom"] {
		t.Fatalf("Pause changed On/Off: %+v", before)
	}
	if err := f.service.cleanupChecked(); err != nil {
		t.Fatalf("stop paused service: %v", err)
	}
	f.service.assertStopped()
	f.start(t) // Retain the original dual-facade argv and private manifests.
	task1Eventually(t, "replacement current manager proves startup pause", 30*time.Second, func() bool { return fetchServicePolicy(t, f.client).Availability == nodepolicy.Paused })
	f.assertPaused(t, nodepolicy.PausedLocally)
	got := fetchServicePolicy(t, f.client).Policy
	if !reflect.DeepEqual(got.Profiles, policy.Profiles) || !reflect.DeepEqual(got.ResidentModels, policy.ResidentModels) || !reflect.DeepEqual(got.ServedModels, policy.ServedModels) {
		t.Fatal("paused restart lost model declarations")
	}
	var after struct {
		Engines map[string]bool `json:"engines"`
	}
	task2Decode(t, readLiveDurableFile(t, intentPath), &after)
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("paused restart changed intent: before=%+v after=%+v", before, after)
	}
	f.assertEffects(t, 0, 1, 0, 0)
	f.native.assertCounts(t, 0, 0, 0)
	f.availability(t, nodepolicy.Available, "")
	task1Eventually(t, "eligible saved-On engine resumes", 30*time.Second, func() bool {
		f.engine.mu.Lock()
		defer f.engine.mu.Unlock()
		return len(f.engine.starts["llamacpp"]) == 2
	})
	var status struct {
		Running bool `json:"running"`
	}
	task2Decode(t, f.call(t, "engine:status", map[string]string{"engine": "llamacpp-custom"}), &status)
	if status.Running {
		t.Fatalf("Resume activated saved-Off engine: %+v", status)
	}
	f.assertEffects(t, 0, 2, 0, 0)
	f.native.assertCounts(t, 0, 0, 0)
}

type stopPausePeer struct {
	engine *engineProcessFixture
	native *stopPauseNative
	id     string
}

func (p *stopPausePeer) assertPaused(t *testing.T) {
	t.Helper()
	p.engine.mu.Lock()
	identities := append([]serviceProcessIdentity(nil), p.engine.identities...)
	p.engine.mu.Unlock()
	for _, identity := range identities {
		stopped, err := p.engine.processStopped(context.Background(), identity)
		if err != nil || !stopped {
			t.Fatalf("peer Pause completed before native lifetime exited: identity=%+v stopped=%t err=%v", identity, stopped, err)
		}
	}
	var policy struct {
		Policy       nodepolicy.Policy       `json:"policy"`
		Availability nodepolicy.Availability `json:"availability"`
	}
	task2Decode(t, p.engine.rpcOK(t, "policy:get", nil), &policy)
	if policy.Availability != nodepolicy.Paused || policy.Policy.Availability != nodepolicy.Paused || policy.Policy.PausedBy != nodepolicy.PausedLocally {
		t.Fatalf("peer target pause readback=%+v", policy)
	}
	var inventory struct {
		Engines []struct {
			Engine              string `json:"engine"`
			Running             *bool  `json:"running"`
			TerminationVerified bool   `json:"terminationVerified"`
		} `json:"engines"`
	}
	task2Decode(t, p.engine.rpcOK(t, "engine:get-installed", nil), &inventory)
	if len(inventory.Engines) != len(privateFixtureEngineNames()) {
		t.Fatal("peer current-manager shutdown inventory incomplete")
	}
	seen := make(map[string]bool)
	for _, status := range inventory.Engines {
		if !slices.Contains(privateFixtureEngineNames(), status.Engine) || seen[status.Engine] || status.Running == nil || *status.Running {
			t.Fatalf("peer shutdown inventory=%+v", status)
		}
		seen[status.Engine] = true
	}
}

// Same composition as the existing ready-peer inventory case: one real service
// and one standalone real broker/manager/proxy, with mutual on-disk pins.
func newStopPausePair(t *testing.T) (*stopPauseFixture, *stopPausePeer) {
	t.Helper()
	s := newRemoteServiceFixture(t)
	release := holdStopPauseServicePorts(t, s.ports)
	base := shortClusterTempDir(t)
	cluster := filepath.Join(base, "cluster")
	if err := os.MkdirAll(cluster, 0o700); err != nil {
		t.Fatal(err)
	}
	const peerID = "zz-stop-pause-peer"
	peerCert := mintClusterIdentity(t, cluster, peerID)
	targetCert, err := os.ReadFile(filepath.Join(s.clusterDir, "node.crt"))
	if err != nil {
		t.Fatal(err)
	}
	writeActiveAdmission(t, cluster)
	writePin(t, cluster, "management-service-target", string(targetCert))
	writeMembers(t, cluster, "management-service-target", peerID)
	jointWriteJSON(t, filepath.Join(cluster, "identity.json"), map[string]string{"node_uuid": peerID})
	writePin(t, s.clusterDir, peerID, peerCert)
	writeMembers(t, s.clusterDir, "management-service-target", "management-service-peer", "management-service-allowed", peerID)
	var native *stopPauseNative
	service := newStopPauseServiceFixture(t, s, func(e *engineProcessFixture) {
		native = newStopPauseNative(t, e)
		release()
	})
	f := &stopPauseFixture{stopPauseServiceFixture: service, native: native}
	f.startModel(t)
	ports := reserveFixtureServicePorts(t)
	releasePeer := holdStopPauseServicePorts(t, ports)
	e := prepareEngineProcessFixture(t, base, ports, "")
	peerNative := newStopPauseNative(t, e)
	facades := map[string]int{"llamacpp-custom": e.facadePort, "llamacpp": e.targetFacadePort}
	for _, name := range engines.Names() {
		if _, exists := facades[name]; !exists {
			facades[name] = freePort(t)
		}
	}
	writePrivateBrokerEngineSettings(t, base, cluster, facades)
	releasePeer()
	e.start(t)
	e.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	assertStopPauseNativeLifetime(t, e)
	p := &stopPausePeer{engine: e, native: peerNative, id: peerID}
	var inventory struct {
		Models   []string                  `json:"models"`
		ByEngine map[string][]string       `json:"modelsByEngine"`
		Loaded   map[string][]string       `json:"loadedByEngine"`
		Policy   *nodepolicy.Advertisement `json:"policy"`
	}
	task1Eventually(t, "peer manager reports model inventory and residency", 15*time.Second, func() bool {
		task2Decode(t, e.rpcOK(t, "engine:models", nil), &inventory)
		return inventory.Policy != nil && slices.Contains(inventory.ByEngine["llamacpp"], "switch-model") && slices.Contains(inventory.Loaded["llamacpp"], "switch-model")
	})
	cfg, ok := clustertrust.Open(s.clusterDir).ClientTLSConfig(peerID)
	if !ok {
		t.Fatal("service lacks peer pin")
	}
	pinned := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg}}
	t.Cleanup(pinned.CloseIdleConnections)
	task1Eventually(t, "peer broker publishes healthy native ingress", 15*time.Second, func() bool {
		response, err := pinned.Get(fmt.Sprintf("https://127.0.0.1:%d/v1/models", e.targetFacadePort))
		if err != nil {
			t.Fatal(err)
		}
		_, readErr := io.Copy(io.Discard, response.Body)
		closeErr := response.Body.Close()
		if readErr != nil || closeErr != nil {
			t.Fatalf("peer health read=%v close=%v", readErr, closeErr)
		}
		return response.StatusCode == http.StatusOK
	})
	f.publishDirectory(t, noderec.DirectoryNode{HostUUID: peerID, ClusterUUID: peerID, Name: peerID, IP: "127.0.0.1", Trusted: true,
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{privateFixtureEngine(t, "llamacpp-custom").DiscoveryService: {Port: e.facadePort}, noderec.ServiceLlamaCPP: {Port: e.targetFacadePort}},
		Models:   inventory.Models, ModelsByEngine: inventory.ByEngine, LoadedByEngine: inventory.Loaded, Policy: inventory.Policy})
	p.publishTarget(t, f)
	return f, p
}

func (p *stopPausePeer) publishTarget(t *testing.T, f *stopPauseFixture) {
	t.Helper()
	// Publish the real target's control route through the existing scanner.
	// Outbound RPCs retain the normal pinned client and target broker relay.
	target := f.publish(t)
	target.Services[noderec.ServiceEngineControl] = noderec.ServiceStatus{Port: f.service.ports.engineControl}
	target.Services[noderec.ServiceEngineManager] = noderec.ServiceStatus{Port: f.service.ports.engineManagerHTTP}
	frame := jsonrpc.Message{JSONRPC: "2.0", Method: noderec.NotifyNodeUpdated, Params: mustMarshal(t, noderec.NodeEvent{Node: target})}
	plain := &http.Client{Timeout: 5 * time.Second}
	defer plain.CloseIdleConnections()
	response, err := plain.Post(fmt.Sprintf("http://127.0.0.1:%d", p.engine.scannerPort), "application/json", bytes.NewReader(mustMarshal(t, frame)))
	if err != nil {
		t.Fatal(err)
	}
	_, readErr := io.Copy(io.Discard, response.Body)
	closeErr := response.Body.Close()
	if response.StatusCode != http.StatusOK || readErr != nil || closeErr != nil {
		t.Fatalf("origin scanner publication HTTP=%d read=%v close=%v", response.StatusCode, readErr, closeErr)
	}
	task1Eventually(t, "origin manager reaches target-owned pinned policy relay", 15*time.Second, func() bool {
		got := p.engine.rpc(t, "engine:remote-policy-get", map[string]string{"nodeId": f.nodeID})
		if got.Error != nil {
			return false
		}
		var policy struct {
			Policy nodepolicy.PeerPolicy `json:"policy"`
		}
		task2Decode(t, got.Result, &policy)
		return policy.Policy.Revision == fetchServicePolicy(t, f.client).Policy.Revision
	})
}

func (f *stopPauseFixture) history(t *testing.T, count int) []nodepolicy.RoutingHistoryRecord {
	t.Helper()
	var records []nodepolicy.RoutingHistoryRecord
	task1Eventually(t, "real service terminal routing history", 30*time.Second, func() bool {
		response, err := f.client.call(strconv.FormatInt(newRPCID(), 10), nodepolicy.MethodRoutingGetInitial, json.RawMessage(`{}`), serviceCallTimeout)
		if err != nil {
			t.Fatal(err)
		}
		if response.Error != nil {
			if response.Error.Code == jsonrpc.HistoryBusyCode {
				return false
			}
			t.Fatalf("history: %+v", response.Error)
		}
		initial, diagnostic, err := nodepolicy.DecodeRoutingHistoryInitial(response.Result)
		if err != nil || diagnostic.Count != 0 {
			t.Fatalf("history diagnostic=%+v err=%v", diagnostic, err)
		}
		records = initial.Requests
		if len(records) > count {
			t.Fatalf("hidden replay created extra request histories: %+v", records)
		}
		if len(records) != count {
			return false
		}
		for _, record := range records {
			if record.EndedAt == nil {
				return false
			}
		}
		return true
	})
	return records
}

func testStopPauseFallback(t *testing.T) {
	f, peer := newStopPausePair(t)
	f.native.holding(true)
	// Explicit selection makes the first destination deterministic; a paused
	// candidate is excluded by the existing request owner's fallback path.
	f.call(t, "llamacpp-proxy:node/select", map[string]string{"id": f.nodeID})
	client := &http.Client{Timeout: 90 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	demand := f.request(t, client, f.engine.targetFacadePort, false)
	gate := f.native.gate(t, 1)
	f.availability(t, nodepolicy.Paused, "immediate")
	f.assertPaused(t, nodepolicy.PausedLocally)
	stopPauseBarrier(t, gate.departed, "precommit local execution withdrawn")
	assertStopPauseSuccess(t, stopPauseAwait(t, demand), "llamacpp", peer.id)
	f.native.assertCounts(t, 1, 0, 0)
	peer.native.assertCounts(t, 1, 0, 0)
	records := f.history(t, 1)
	if records[0].State != "completed" || len(records[0].Attempts) != 2 || records[0].Served == nil || records[0].Served.NodeID == nil || *records[0].Served.NodeID != peer.id {
		t.Fatalf("precommit origin fallback history=%+v", records[0])
	}
	first, second := records[0].Attempts[0], records[0].Attempts[1]
	if first.Outcome != "redispatched" || first.Reason != "node-paused" || first.Destination.NodeID == nil || *first.Destination.NodeID != f.nodeID || second.Outcome != "committed" || second.Destination.NodeID == nil || *second.Destination.NodeID != peer.id {
		t.Fatalf("fallback changed owner/attempt cause: %+v", records[0])
	}
}

type stopPauseStream struct {
	reader *bufio.Reader
}

func (f *stopPauseFixture) stream(t *testing.T, client *http.Client) *stopPauseStream {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	t.Cleanup(cancel)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", f.engine.targetFacadePort), strings.NewReader(`{"model":"switch-model","messages":[],"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := response.Body.Close(); err != nil {
			t.Errorf("close stream: %v", err)
		}
	})
	if response.StatusCode != http.StatusOK || !strings.HasPrefix(response.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("stream HTTP=%d content-type=%s", response.StatusCode, response.Header.Get("Content-Type"))
	}
	stream := &stopPauseStream{reader: bufio.NewReader(response.Body)}
	first := stream.event(t)
	var frame struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}
	task2Decode(t, first, &frame)
	if len(frame.Choices) != 1 || frame.Choices[0].Delta.Content != "first" || frame.Choices[0].FinishReason != nil {
		t.Fatalf("first committed event=%+v", frame)
	}
	return stream
}

func (s *stopPauseStream) event(t *testing.T) []byte {
	t.Helper()
	line, err := s.reader.ReadString('\n')
	if err != nil {
		t.Fatalf("read SSE data: %v", err)
	}
	separator, err := s.reader.ReadString('\n')
	if err != nil || separator != "\n" || !strings.HasPrefix(line, "data: ") {
		t.Fatalf("invalid complete SSE frame separator=%q err=%v", separator, err)
	}
	return []byte(strings.TrimSuffix(strings.TrimPrefix(line, "data: "), "\n"))
}

func (s *stopPauseStream) failed(t *testing.T) {
	t.Helper()
	var ending struct {
		Error struct{ Code, Type, Message string } `json:"error"`
	}
	task2Decode(t, s.event(t), &ending)
	if ending.Error.Code != "stream_interrupted" || ending.Error.Type != "server_error" || ending.Error.Message != "stream interrupted: node paused" {
		t.Fatalf("SC-a failure=%+v", ending)
	}
	if terminal := string(s.event(t)); terminal != "[DONE]" {
		t.Fatalf("SC-a terminal=%q", terminal)
	}
	remaining, err := io.ReadAll(s.reader)
	if err != nil || len(remaining) != 0 {
		t.Fatalf("bytes after SC-a terminal=%d err=%v", len(remaining), err)
	}
}

func testStopPauseCommitted(t *testing.T, remote bool) {
	f, peer := newStopPausePair(t)
	selected, executing, alternate := f.nodeID, f.native, peer.native
	if remote {
		selected, executing, alternate = peer.id, peer.native, f.native
	}
	executing.holding(true)
	f.call(t, "llamacpp-proxy:node/select", map[string]string{"id": selected})
	client := &http.Client{Timeout: 90 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	stream := f.stream(t, client)
	gate := executing.gate(t, 1)
	if remote {
		peer.engine.rpcOK(t, "node:set-availability", map[string]string{"state": "paused", "pauseMode": "immediate"})
		peer.assertPaused(t)
	} else {
		f.availability(t, nodepolicy.Paused, "immediate")
		f.assertPaused(t, nodepolicy.PausedLocally)
	}
	stream.failed(t)
	stopPauseBarrier(t, gate.departed, "committed native execution cancelled")
	executing.assertCounts(t, 1, 0, 0)
	alternate.assertCounts(t, 0, 0, 0)
	records := f.history(t, 1)
	failed := records[0]
	if failed.State != "failed" || failed.Served != nil || len(failed.Attempts) != 1 || (failed.Attempts[0].Outcome != "interrupted" && failed.Attempts[0].Outcome != "failed") {
		t.Fatalf("committed interruption history=%+v", failed)
	}
	if failed.Attempts[0].Destination.NodeID == nil || *failed.Attempts[0].Destination.NodeID != selected {
		t.Fatalf("committed attempt changed destination: %+v", failed)
	}
	// Check the independent public workload projection as well as request
	// history. A clean synthetic terminal must never publish completed/cancelled.
	task1Eventually(t, "committed execution publishes failed workload", 15*time.Second, func() bool {
		var initial struct {
			Workloads []departedRunInfo `json:"workloads"`
		}
		task2Decode(t, f.call(t, "workloads:get-initial", nil), &initial)
		if len(initial.Workloads) > 1 {
			t.Fatalf("unexpected extra workloads before client retry: %+v", initial)
		}
		if len(initial.Workloads) == 0 {
			return false
		}
		workload := initial.Workloads[0]
		if workload.State == "running" || workload.State == "queued" {
			return false
		}
		if workload.State != "failed" || workload.CompletedAt == nil || workload.Error == nil || workload.OriginatedFrom != f.nodeID || workload.Engine != "llamacpp" || workload.Model != "switch-model" {
			t.Fatalf("committed workload outcome=%+v", workload)
		}
		return true
	})
	// Only a new HTTP request may use the alternate. The failed history's
	// identity and attempt list must remain unchanged after that new request.
	retryNode := peer.id
	if remote {
		retryNode = f.nodeID
	}
	f.call(t, "llamacpp-proxy:node/select", map[string]string{"id": retryNode})
	assertStopPauseSuccess(t, stopPauseAwait(t, f.request(t, client, f.engine.targetFacadePort, false)), "llamacpp", retryNode)
	executing.assertCounts(t, 1, 0, 0)
	alternate.assertCounts(t, 1, 0, 0)
	records = f.history(t, 2)
	var preserved, retried bool
	for _, record := range records {
		if reflect.DeepEqual(record.Request, failed.Request) {
			preserved = record.State == "failed" && record.Served == nil && reflect.DeepEqual(record.Attempts, failed.Attempts) && reflect.DeepEqual(record.EndedAt, failed.EndedAt)
		} else {
			retried = record.State == "completed" && len(record.Attempts) == 1
		}
	}
	if !preserved || !retried {
		t.Fatalf("client retry rewrote failed request or replayed: %+v", records)
	}
}

func testStopPauseRemoteExecution(t *testing.T) {
	f, peer := newStopPausePair(t)
	peer.native.holding(true)
	f.call(t, "llamacpp-proxy:node/select", map[string]string{"id": peer.id})
	client := &http.Client{Timeout: 90 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	stream := f.stream(t, client)
	gate := peer.native.gate(t, 1)
	f.availability(t, nodepolicy.Paused, "immediate")
	f.assertPaused(t, nodepolicy.PausedLocally)
	select {
	case <-gate.departed:
		t.Fatal("origin Pause cancelled peer execution")
	default:
	}
	gate.once.Do(func() { close(gate.release) })
	var final struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	task2Decode(t, stream.event(t), &final)
	if len(final.Choices) != 1 || final.Choices[0].FinishReason != "stop" || string(stream.event(t)) != "[DONE]" {
		t.Fatalf("existing remote execution did not finish: %+v", final)
	}
	if rest, err := io.ReadAll(stream.reader); err != nil || len(rest) != 0 {
		t.Fatalf("remote stream tail bytes=%d err=%v", len(rest), err)
	}
	peer.native.holding(false)
	assertStopPauseSuccess(t, stopPauseAwait(t, f.request(t, client, f.engine.targetFacadePort, false)), "llamacpp", peer.id)
	f.native.assertCounts(t, 0, 0, 0)
	peer.native.assertCounts(t, 2, 0, 0)
	for _, record := range f.history(t, 2) {
		if record.State != "completed" || len(record.Attempts) != 1 {
			t.Fatalf("remote execution disturbed by origin Pause: %+v", record)
		}
	}
}
