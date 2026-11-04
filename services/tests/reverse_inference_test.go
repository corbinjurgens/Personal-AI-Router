// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

const phase04Model = "phase04"

// The broker boundary supplies discovery and current service registrations.
// Every transport, private gateway and target handler runs its production main.
type phase04Fixture struct {
	control           *task2Fixture
	proxies           map[string]*task1Process
	engines           map[string]*phase04Engine
	stores            map[string]string
	files             map[string]*phase04FileObserver
	engineClusterDirs map[string]string
}

type phase04FixtureOptions struct {
	isolateCopyAuthority bool
}

type phase04Engine struct {
	port     int
	mu       sync.Mutex
	calls    map[string]int
	models   []string
	canceled chan string
	release  chan struct{}
}

// addModel makes the engine's native inventory list one more unloaded model,
// as a router that has the model in its cache does.
func (e *phase04Engine) addModel(id string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.models = append(e.models, id)
}

// inventory is the router's /models body: the resident inference model, then
// every cached model that is not loaded.
func (e *phase04Engine) inventory() map[string]any {
	e.mu.Lock()
	defer e.mu.Unlock()
	rows := []map[string]any{{"id": phase04Model, "status": map[string]string{"value": "loaded"}}}
	for _, id := range e.models {
		rows = append(rows, map[string]any{"id": id, "status": map[string]string{"value": "unloaded"}})
	}
	return map[string]any{"data": rows}
}

func (e *phase04Engine) count(mode string) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.calls[mode]
}

func newPhase04Engine(t *testing.T) *phase04Engine {
	t.Helper()
	e := &phase04Engine{calls: map[string]int{}, canceled: make(chan string, 16), release: make(chan struct{})}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models", "/models":
			if err := json.NewEncoder(w).Encode(e.inventory()); err != nil {
				t.Error(err)
			}
		case "/v1/chat/completions":
			var input struct {
				Model    string `json:"model"`
				Messages []struct {
					Content string `json:"content"`
				} `json:"messages"`
			}
			if err := json.NewDecoder(r.Body).Decode(&input); err != nil || input.Model != phase04Model || len(input.Messages) != 1 {
				t.Errorf("fixture inference model=%q messages=%d decode=%v", input.Model, len(input.Messages), err)
				http.Error(w, "bad input", http.StatusBadRequest)
				return
			}
			e.mu.Lock()
			e.calls[input.Messages[0].Content]++
			e.mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("X-Phase04-Stream", "preserved")
			w.WriteHeader(http.StatusOK)
			if _, err := io.WriteString(w, `data: {"model":"phase04","choices":[{"index":0,"delta":{"content":"first"},"finish_reason":null}]}`+"\n\n"); err != nil {
				return
			}
			w.(http.Flusher).Flush()
			if input.Messages[0].Content == "cancel" || input.Messages[0].Content == "drop" {
				<-r.Context().Done()
				e.canceled <- input.Messages[0].Content
				return
			}
			if input.Messages[0].Content == "direct-drop" {
				select {
				case <-e.release:
				case <-r.Context().Done():
					return
				}
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
			if strings.HasPrefix(input.Messages[0].Content, "concurrent-") {
				select {
				case <-e.release:
				case <-r.Context().Done():
					return
				}
			}
			// A distinct later flush makes buffering observable to the caller.
			time.Sleep(75 * time.Millisecond)
			if _, err := io.WriteString(w, `data: {"model":"phase04","choices":[{"index":0,"delta":{"content":"second"},"finish_reason":"stop"}]}`+"\n\n"); err != nil {
				return
			}
			if _, err := io.WriteString(w, "data: [DONE]\n\n"); err != nil {
				return
			}
			w.(http.Flusher).Flush()
		default:
			if err := json.NewEncoder(w).Encode(map[string]string{"version": "isolated-phase04"}); err != nil {
				t.Error(err)
			}
		}
	}))
	t.Cleanup(srv.Close)
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	e.port, err = strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func newPhase04Fixture(t *testing.T) *phase04Fixture {
	t.Helper()
	return newPhase04FixtureWithOptions(t, phase04FixtureOptions{})
}

func newPhase04FixtureWithOptions(t *testing.T, options phase04FixtureOptions) *phase04Fixture {
	t.Helper()
	f := &phase04Fixture{
		control: &task2Fixture{engines: map[string]*task1Process{}},
		proxies: map[string]*task1Process{}, engines: map[string]*phase04Engine{},
		stores: map[string]string{}, files: map[string]*phase04FileObserver{},
		engineClusterDirs: map[string]string{},
	}
	blocked, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	f.control.blockedPort = blocked.Addr().(*net.TCPAddr).Port
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			conn, err := blocked.Accept()
			if err != nil {
				return
			}
			go func() {
				defer conn.Close()
				_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
				var first [1]byte
				_, _ = conn.Read(first[:])
				if tcp, ok := conn.(*net.TCPConn); ok {
					_ = tcp.SetLinger(0)
				}
			}()
		}
	}()
	t.Cleanup(func() {
		if err := blocked.Close(); err != nil {
			t.Error(err)
		}
		<-done
	})
	for _, uuid := range []string{"phase04-reachable", "phase04-restricted"} {
		base := shortClusterTempDir(t)
		dir := filepath.Join(base, "cluster")
		if err := os.MkdirAll(filepath.Join(dir, "trusted"), 0o700); err != nil {
			t.Fatal(err)
		}
		n := &task1Node{uuid: uuid, base: base, ports: map[noderec.ServiceKey]int{}, observers: map[noderec.ServiceKey]*task1Observer{}}
		n.cert = mintClusterIdentity(t, dir, uuid)
		task1WriteJSON(t, filepath.Join(dir, "identity.json"), map[string]any{"node_uuid": uuid, "created_at": 1})
		task1WriteJSON(t, filepath.Join(dir, "admission.json"), map[string]any{"counter": 1, "activated": 1, "clusterId": task1ClusterID, "epoch": 1})
		for _, service := range []noderec.ServiceKey{noderec.ServiceCluster, noderec.ServiceEngineControl, noderec.ServiceEngineManager, noderec.ServiceNodeInfo, noderec.ServiceLlamaCPP} {
			n.ports[service] = temporaryClusterPort(t)
		}
		if f.control.a == nil {
			f.control.a = n
		} else {
			f.control.b = n
		}
	}
	nodes := []*task1Node{f.control.a, f.control.b}
	for _, n := range nodes {
		peer := f.peer(n)
		task1WriteJSON(t, filepath.Join(n.base, "cluster", "trusted", peer.uuid+".json"), map[string]any{
			"nodeUuid": peer.uuid, "nodeId": peer.uuid, "certPem": peer.cert, "clusterId": task1ClusterID, "admissionEpoch": 1,
		})
		n.mesh = clustertrust.Open(filepath.Join(n.base, "cluster"))
	}
	for _, n := range nodes {
		n.observers[noderec.ServiceCluster] = newTask1Observer(t, n, noderec.ServiceCluster, nodes)
		f.engines[n.uuid] = newPhase04Engine(t)
		f.stores[n.uuid] = filepath.Join(n.base, "model-store")
		config := n.base
		if runtime.GOOS == "darwin" {
			config = filepath.Join(config, "Library", "Application Support")
		}
		manifestDir := filepath.Join(config, "Nvidia Corporation", "Personal AI Router", "engines")
		if err := os.MkdirAll(manifestDir, 0o700); err != nil {
			t.Fatal(err)
		}
		writeDisabledBuiltinEngineOverrides(t, n.base, filepath.Dir(manifestDir))
		port := f.engines[n.uuid].port
		// An empty preset marks the launch as a router whose models come from
		// the HF cache, so the node can export its cached models for copy.
		preset := filepath.Join(n.base, "external", "models.ini")
		if err := os.MkdirAll(filepath.Dir(preset), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(preset, []byte("# synthetic router preset"+string(rune(10))), 0o600); err != nil {
			t.Fatal(err)
		}
		task1WriteJSON(t, filepath.Join(manifestDir, "llamacpp.json"), map[string]any{
			"engine":  "llamacpp",
			"runtime": map[string]any{"port": port, "args": []string{"--models-preset", preset}, "env": map[string]string{"LLAMA_CACHE": f.stores[n.uuid]}, "ready": map[string]any{"http": "http://127.0.0.1:{port}/v1/models", "status": 200, "json_match": nil}},
			"platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{
				"detect": []string{nodeInfoBin}, "models_dir": f.stores[n.uuid], "runtime": map[string]any{"bin": nodeInfoBin, "port": port, "args": []string{"--models-preset", preset}},
			}},
		})
		var members []task1Member
		for _, peer := range nodes {
			members = append(members, task1Member{ID: peer.uuid, NodeUUID: peer.uuid, Name: peer.uuid, IPAddress: "127.0.0.1", Port: f.control.directory(peer).Services[noderec.ServiceCluster].Port, ClusterID: task1ClusterID, AdmissionEpoch: 1, State: "member"})
		}
		task1WriteJSON(t, filepath.Join(n.base, "cluster", "members.json"), members)
		engineClusterDir := filepath.Join(n.base, "cluster")
		if options.isolateCopyAuthority {
			engineClusterDir = phase04CopyAuthorityDirectory(t, n, f.peer(n))
		}
		f.engineClusterDirs[n.uuid] = engineClusterDir
		f.control.engines[n.uuid] = startTask1Process(t, n.base, engineMgrBin, "--cluster-dir", engineClusterDir, "--control-port", strconv.Itoa(n.ports[noderec.ServiceEngineControl]), "--http-port", strconv.Itoa(n.ports[noderec.ServiceEngineManager]), "--loaded-poll-interval", "0")
		completeEngineManagerRecovery(t, func(method string, params any) jsonrpc.Message {
			return f.control.engines[n.uuid].call(t, method, params)
		})
		f.control.engines[n.uuid].call(t, "engine:configured-ports", nil)
		f.files[n.uuid] = newPhase04FileObserver(t, n, f.peer(n))
		p := startTask1Process(t, n.base, proxyBin, "--cluster-dir", filepath.Join(n.base, "cluster"))
		bindProxyFixtureIdentity(t, n.uuid, func(method string, params any) jsonrpc.Message {
			return p.call(t, method, params)
		})
		p.call(t, "facade/enable", map[string]any{"engine": "llamacpp", "port": n.ports[noderec.ServiceLlamaCPP], "ignorePersistedPort": true})
		p.call(t, engines.AddressMethod("llamacpp", "node/set-local-backend"), map[string]any{"engine": "llamacpp", "host": "127.0.0.1", "port": f.engines[n.uuid].port, "healthy": true})
		f.proxies[n.uuid] = p
		n.cm = startTask1Process(t, n.base, clusterMgrBin, "--config-dir", n.base, "--port", strconv.Itoa(n.ports[noderec.ServiceCluster]))
		n.cm.call(t, "cluster:get-node-id", nil)
		n.cm.call(t, "cluster:set-local-services", map[string]any{"generation": 1, "services": []noderec.RegisterParams{
			{Service: noderec.ServiceCluster, Port: n.observers[noderec.ServiceCluster].port},
			{Service: noderec.ServiceEngineControl, Port: f.files[n.uuid].port},
			{Service: noderec.ServiceEngineManager, Port: n.ports[noderec.ServiceEngineManager]},
			{Service: noderec.ServiceLlamaCPP, Port: n.ports[noderec.ServiceLlamaCPP]},
		}})
	}
	task1Eventually(t, "production outer reverse session", 15*time.Second, func() bool {
		return f.control.a.observers[noderec.ServiceCluster].established(f.control.b.uuid) > 0
	})
	for _, n := range nodes {
		peer := f.peer(n)
		directory := f.control.directory(peer)
		port := peer.ports[noderec.ServiceLlamaCPP]
		ec := f.files[peer.uuid].port
		if peer == f.control.b {
			port, ec = f.control.blockedPort, f.control.blockedPort
		}
		directory.Services[noderec.ServiceLlamaCPP] = noderec.ServiceStatus{Port: port, Probe: noderec.ProbeUnreachable}
		directory.Services[noderec.ServiceEngineControl] = noderec.ServiceStatus{Port: ec}
		directory.ModelsByEngine = map[string][]string{"llamacpp": {phase04Model}}
		f.proxies[n.uuid].send(t, engines.AddressMethod("llamacpp", noderec.NotifyNodes), noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{directory}}, false)
		// Destination-owned copy coordination resolves this manager's host ID
		// from the broker directory, independently of its cluster principal.
		self := f.control.directory(n)
		self.Services[noderec.ServiceEngineControl] = noderec.ServiceStatus{Port: f.files[n.uuid].port}
		manager := f.control.engines[n.uuid]
		manager.send(t, noderec.NotifyNodes, noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{self, directory}}, false)
		manager.send(t, "engine:node-projection", map[string]any{
			"policy": nodepolicy.Default().Advertisement(), "pausedLocally": false, "managementBlocked": false,
		}, false)
		// A read on the same stream joins both fixture inputs before copy starts.
		manager.call(t, "engine:configured-ports", nil)
		f.proxies[n.uuid].call(t, engines.AddressMethod("llamacpp", "node/select"), map[string]string{"id": peer.uuid})
	}
	return f
}

func (f *phase04Fixture) peer(n *task1Node) *task1Node {
	if n == f.control.a {
		return f.control.b
	}
	return f.control.a
}

func phase04Request(t *testing.T, ctx context.Context, source *task1Node, mode string) *http.Response {
	t.Helper()
	body, err := json.Marshal(map[string]any{"model": phase04Model, "messages": []map[string]string{{"role": "user", "content": mode}}, "stream": true})
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", source.ports[noderec.ServiceLlamaCPP]), bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 12 * time.Second}).Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

// phase04AssertInterruptedTail checks the stream-continuity contract for an
// interrupted managed stream after its first chunk: the proxy ends it with
// exactly one protocol error frame and a clean EOF, never with more content.
func phase04AssertInterruptedTail(t *testing.T, r io.Reader, label string) {
	t.Helper()
	remainder, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("%s: read error %v, want clean EOF after an error frame (remainder=%q)", label, err, remainder)
	}
	frames := strings.Split(string(remainder), "\n\n")
	if len(frames) != 3 || !strings.HasPrefix(frames[0], "data: ") || frames[1] != "data: [DONE]" || frames[2] != "" {
		t.Fatalf("%s remainder=%q, want one SSE error frame and DONE", label, remainder)
	}
	line := []byte(strings.TrimPrefix(frames[0], "data: "))
	var frame map[string]json.RawMessage
	if err := json.Unmarshal(line, &frame); err != nil {
		t.Fatalf("%s tail %q is not a JSON frame: %v", label, line, err)
	}
	var failure struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	}
	if len(frame) != 1 || json.Unmarshal(frame["error"], &failure) != nil || failure.Message == "" || failure.Type != "server_error" || failure.Code != "stream_interrupted" {
		t.Fatalf("%s tail %q, want a single non-empty error member", label, line)
	}
}

func phase04FirstChunk(t *testing.T, resp *http.Response) *bufio.Reader {
	t.Helper()
	if resp.StatusCode != http.StatusOK || resp.Header.Get("X-Phase04-Stream") != "preserved" || resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("stream response status=%d headers=%v", resp.StatusCode, resp.Header)
	}
	r := bufio.NewReader(resp.Body)
	line, err := r.ReadString('\n')
	if err != nil || line != `data: {"model":"phase04","choices":[{"index":0,"delta":{"content":"first"},"finish_reason":null}]}`+"\n" {
		t.Fatalf("first chunk %q, %v", line, err)
	}
	if line, err := r.ReadString('\n'); err != nil || line != "\n" {
		t.Fatalf("first SSE separator %q, %v", line, err)
	}
	return r
}

func phase04AssertOrdered(t *testing.T, f *phase04Fixture, source *task1Node, mode string) {
	t.Helper()
	target := f.peer(source)
	before := f.engines[target.uuid].count(mode)
	resp := phase04Request(t, context.Background(), source, mode)
	defer resp.Body.Close()
	r := phase04FirstChunk(t, resp)
	line, err := r.ReadString('\n')
	if err != nil || line != `data: {"model":"phase04","choices":[{"index":0,"delta":{"content":"second"},"finish_reason":"stop"}]}`+"\n" {
		t.Fatalf("second chunk %q, %v", line, err)
	}
	if remainder, err := io.ReadAll(r); err != nil || string(remainder) != "\ndata: [DONE]\n\n" {
		t.Fatalf("stream termination %q, %v", remainder, err)
	}
	if got := f.engines[target.uuid].count(mode); got != before+1 {
		t.Fatalf("target dispatches=%d, want %d", got, before+1)
	}
}

func TestReverseInferencePhase04(t *testing.T) {
	f := newPhase04Fixture(t)
	a, b := f.control.a, f.control.b
	t.Run("blocked service remains a discovered eligible model owner", func(t *testing.T) {
		response := f.proxies[a.uuid].call(t, engines.AddressMethod("llamacpp", "nodes/list"), nil)
		var result struct {
			Nodes []struct {
				ID     string   `json:"id"`
				Port   int      `json:"port"`
				Models []string `json:"models"`
			} `json:"nodes"`
		}
		task2Decode(t, response.Result, &result)
		found := false
		for _, node := range result.Nodes {
			if node.ID == b.uuid && node.Port == f.control.blockedPort && reflect.DeepEqual(node.Models, []string{phase04Model}) {
				found = true
			}
		}
		if !found {
			t.Fatalf("restricted model owner absent from overlay: %+v", result)
		}
		phase04AssertOrdered(t, f, a, "ordered")
	})
	t.Run("opposite direction retains direct mTLS streaming", func(t *testing.T) {
		phase04AssertOrdered(t, f, b, "direct")
	})
	testCancellation := func(name string, source, target *task1Node) {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			resp := phase04Request(t, ctx, source, "cancel")
			defer resp.Body.Close()
			r := phase04FirstChunk(t, resp)
			cancel()
			if _, err := io.ReadAll(r); err == nil {
				t.Fatal("canceled exposed stream ended without error")
			}
			select {
			case mode := <-f.engines[target.uuid].canceled:
				if mode != "cancel" {
					t.Fatalf("canceled mode=%q", mode)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("target did not observe caller cancellation")
			}
			if got := f.engines[target.uuid].count("cancel"); got != 1 {
				t.Fatalf("canceled dispatches=%d, want 1", got)
			}
		})
	}
	testCancellation("reverse cancellation after exposed chunk reaches target without replay", a, b)
	testCancellation("direct cancellation after exposed chunk reaches target without replay", b, a)
	t.Run("opposite direct stream drops without replay and new request recovers", func(t *testing.T) {
		resp := phase04Request(t, context.Background(), b, "direct-drop")
		defer resp.Body.Close()
		reader := phase04FirstChunk(t, resp)
		select {
		case f.engines[a.uuid].release <- struct{}{}:
		case <-time.After(5 * time.Second):
			t.Fatal("direct stream stopped before fault")
		}
		phase04AssertInterruptedTail(t, reader, "dropped direct stream")
		phase04AssertOrdered(t, f, b, "direct-recovered")
		if got := f.engines[a.uuid].count("direct-drop"); got != 1 {
			t.Fatalf("exposed direct inference replayed: dispatches=%d", got)
		}
	})
	t.Run("outer session drop terminates exposed stream and new request recovers", func(t *testing.T) {
		resp := phase04Request(t, context.Background(), a, "drop")
		defer resp.Body.Close()
		r := phase04FirstChunk(t, resp)
		outer := a.observers[noderec.ServiceCluster]
		before := outer.established(b.uuid)
		outer.interrupt(b.uuid, true)
		phase04AssertInterruptedTail(t, r, "dropped exposed stream")
		select {
		case mode := <-f.engines[b.uuid].canceled:
			if mode != "drop" {
				t.Fatalf("drop canceled mode=%q", mode)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("dropped stream did not close target request")
		}
		for _, node := range []*task1Node{a, b} {
			var roster struct {
				Nodes []task1Member `json:"nodes"`
			}
			task2Decode(t, node.cm.call(t, "nodes:get-initial", nil).Result, &roster)
			found := false
			for _, member := range roster.Nodes {
				if member.NodeUUID == f.peer(node).uuid && member.ClusterID == task1ClusterID && member.AdmissionEpoch == 1 && member.State == "member" {
					found = true
				}
			}
			if !found {
				t.Fatalf("reverse outage changed membership on %s: %+v", node.uuid, roster)
			}
		}
		outer.interrupt(b.uuid, false)
		task1Eventually(t, "new reverse session", 15*time.Second, func() bool { return outer.established(b.uuid) > before })
		phase04AssertOrdered(t, f, a, "recovered")
		if got := f.engines[b.uuid].count("drop"); got != 1 {
			t.Fatalf("exposed inference replayed: dispatches=%d", got)
		}
	})
	testAuthority := func(name string, source, target *task1Node) {
		t.Run(name, func(t *testing.T) {
			pin := filepath.Join(source.base, "cluster", "trusted", target.uuid+".json")
			original := readLiveDurableFile(t, pin)
			defer func() {
				if err := os.WriteFile(pin, original, 0o600); err != nil {
					t.Error(err)
				}
			}()
			assertRejected := func(t *testing.T, mode string) {
				t.Helper()
				resp := phase04Request(t, context.Background(), source, mode)
				defer resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					t.Fatal("invalid authority accepted inference")
				}
				if got := f.engines[target.uuid].count(mode); got != 0 {
					t.Fatalf("invalid authority reached target %d times", got)
				}
			}
			task1WriteJSON(t, pin, map[string]any{"nodeUuid": target.uuid, "nodeId": target.uuid, "certPem": source.cert, "clusterId": task1ClusterID, "admissionEpoch": 1})
			assertRejected(t, "wrong-pin")
			removeAuthorityFile(t, pin)
			assertRejected(t, "removed-pin")
		})
	}
	testAuthority("wrong and removed reverse pins stop before target handler", a, b)
	testAuthority("wrong and removed direct pins stop before target handler", b, a)
}
