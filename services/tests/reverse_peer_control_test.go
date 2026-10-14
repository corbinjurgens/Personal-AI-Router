// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	errorswire "nvpair-shared/errors"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/noderec"
)

const task1ClusterID = "reverse-task1-cluster"
const task1RosterPath = "/v1/cluster/roster"
const task1RemovePath = "/v1/cluster/members/remove"
const task1UpgradePath = "/cluster/reverse"

// task1Process plays the broker's public stdio role. Every receiver notification
// is retained, including duplicates, rather than hidden by a test-side dedup.
type task1Process struct {
	t        *testing.T
	cmd      *exec.Cmd
	in       io.WriteCloser
	output   serviceOutput
	mu       sync.Mutex
	frames   []jsonrpc.Message
	readErr  error
	done     chan struct{}
	readDone chan struct{}
	waitErr  error
	nextID   int
	stopped  bool
}

func startTask1Process(t *testing.T, base, binary string, args ...string) *task1Process {
	t.Helper()
	p := &task1Process{t: t, cmd: exec.Command(binary, args...), done: make(chan struct{}), readDone: make(chan struct{}), nextID: 1}
	p.cmd.Env = append(os.Environ(), "HOME="+base, "XDG_CONFIG_HOME="+base, "APPDATA="+base, "LOCALAPPDATA="+base, "USERPROFILE="+base)
	var err error
	p.in, err = p.cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	out, err := p.cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	p.cmd.Stderr = &p.output
	if err := p.cmd.Start(); err != nil {
		t.Fatal(err)
	}
	go func() {
		defer close(p.readDone)
		codec := jsonrpc.NewCodecMaxFrame(struct {
			io.Reader
			io.Writer
		}{out, io.Discard}, jsonrpc.WorkerFrameBytes)
		for {
			frame, err := codec.Read()
			p.mu.Lock()
			if err != nil {
				p.readErr = err
				p.mu.Unlock()
				return
			}
			if len(p.frames) >= 4096 {
				p.readErr = fmt.Errorf("fixture frame budget exceeded")
				p.mu.Unlock()
				return
			}
			p.frames = append(p.frames, *frame)
			p.mu.Unlock()
		}
	}()
	go func() { <-p.readDone; p.waitErr = p.cmd.Wait(); close(p.done) }()
	t.Cleanup(p.stop)
	return p
}

func (p *task1Process) send(t *testing.T, method string, params any, request bool) int {
	t.Helper()
	id := p.nextID
	p.nextID++
	frame := map[string]any{"jsonrpc": "2.0", "method": method, "params": params}
	if request {
		frame["id"] = id
	}
	if err := json.NewEncoder(p.in).Encode(frame); err != nil {
		t.Fatal(err)
	}
	return id
}

func (p *task1Process) call(t *testing.T, method string, params any) jsonrpc.Message {
	t.Helper()
	id := p.send(t, method, params, true)
	var response jsonrpc.Message
	task1Eventually(t, method, 10*time.Second, func() bool {
		p.mu.Lock()
		defer p.mu.Unlock()
		for _, f := range p.frames {
			if idEquals(f.ID, id) {
				response = f
				return true
			}
		}
		if p.readErr != nil {
			t.Fatalf("%s stdout: %v; stderr: %s", method, p.readErr, p.output.String())
		}
		return false
	})
	if response.Error != nil {
		t.Fatalf("%s: %+v", method, response.Error)
	}
	return response
}

func (p *task1Process) stop() {
	p.t.Helper()
	if p.stopped {
		return
	}
	p.stopped = true
	if err := p.in.Close(); err != nil {
		p.t.Errorf("close child stdin: %v", err)
	}
	select {
	case <-p.done:
	case <-time.After(10 * time.Second):
		if err := p.cmd.Process.Kill(); err != nil {
			p.t.Errorf("kill child: %v", err)
		}
		<-p.done
		p.t.Errorf("child %s required forced cleanup", filepath.Base(p.cmd.Path))
	}
	if p.waitErr != nil {
		p.t.Errorf("child %s exit: %v; stderr: %s", filepath.Base(p.cmd.Path), p.waitErr, p.output.String())
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.readErr != io.EOF {
		p.t.Errorf("child stdout cleanup: %v", p.readErr)
	}
}

func task1Eventually(t *testing.T, what string, timeout time.Duration, ready func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for !ready() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(25 * time.Millisecond)
	}
}

type task1Member struct {
	ID             string `json:"id"`
	NodeUUID       string `json:"nodeUuid"`
	Name           string `json:"name"`
	IPAddress      string `json:"ipAddress"`
	Port           int    `json:"port"`
	ClusterID      string `json:"clusterId"`
	AdmissionEpoch uint64 `json:"admissionEpoch"`
	State          string `json:"state"`
}

type task1Node struct {
	uuid, base, cert      string
	mesh                  *clustertrust.Mesh
	ports                 map[noderec.ServiceKey]int
	observers             map[noderec.ServiceKey]*task1Observer
	cm, workloads, errors *task1Process
}

type task1RequestKey struct{ peer, method, path string }
type task1ConnKey struct{}

// The observation boundary is an HTTP proxy using ONLY fixture certificates.
// It preserves the caller's exact pinned leaf on the upstream connection and
// never answers a service request itself. All mutations reach real binaries.
// Local reverse service maps point here; B's advertised direct endpoints instead
// point at an accepting-and-rejecting TCP listener. No host firewall is changed.
type task1Observer struct {
	port          int
	server        *http.Server
	mu            sync.Mutex
	counts        map[task1RequestKey]int
	upgrades      map[string][]net.Conn
	readyUpgrades map[string]int
	blockUpgrade  bool
	afterReply    func(string, string)
	problems      []string
}

func (o *task1Observer) count(peer, method, path string) int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.counts[task1RequestKey{peer, method, path}]
}

func (o *task1Observer) interrupt(peer string, blocked bool) {
	o.mu.Lock()
	o.blockUpgrade = blocked
	connections := o.upgrades[peer]
	delete(o.upgrades, peer)
	o.mu.Unlock()
	for _, conn := range connections {
		_ = conn.Close()
	}
}

func (o *task1Observer) established(peer string) int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.readyUpgrades[peer]
}

func newTask1Observer(t *testing.T, target *task1Node, service noderec.ServiceKey, nodes []*task1Node) *task1Observer {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	o := &task1Observer{port: ln.Addr().(*net.TCPAddr).Port, counts: make(map[task1RequestKey]int), upgrades: make(map[string][]net.Conn), readyUpgrades: make(map[string]int)}
	proxies := make(map[string]*httputil.ReverseProxy)
	for _, source := range nodes {
		if source == target {
			continue
		}
		cfg, ok := source.mesh.ClientTLSConfig(target.uuid)
		if !ok {
			t.Fatalf("fixture lacks %s -> %s pin", source.uuid, target.uuid)
		}
		u, err := url.Parse("https://127.0.0.1:" + strconv.Itoa(target.ports[service]))
		if err != nil {
			t.Fatal(err)
		}
		proxy := httputil.NewSingleHostReverseProxy(u)
		transport := &http.Transport{TLSClientConfig: cfg, DisableKeepAlives: true}
		t.Cleanup(transport.CloseIdleConnections)
		proxy.Transport = transport
		proxy.ErrorLog = nil
		proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
			http.Error(w, "fixture upstream unavailable", http.StatusBadGateway)
		}
		peer := source.uuid
		proxy.ModifyResponse = func(response *http.Response) error {
			// Settings notifications are a live stream, not a finite reply.
			if response.Request.URL.Path == "/v1/engine-settings/events" {
				return nil
			}
			if response.StatusCode == http.StatusSwitchingProtocols {
				o.mu.Lock()
				o.readyUpgrades[peer]++
				o.mu.Unlock()
				return nil
			}
			body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
			if err != nil {
				return err
			}
			if err := response.Body.Close(); err != nil {
				return err
			}
			response.Body = io.NopCloser(bytes.NewReader(body))
			o.mu.Lock()
			hook := o.afterReply
			o.mu.Unlock()
			if hook != nil {
				hook(peer, response.Request.URL.Path)
			}
			return nil
		}
		proxies[source.uuid] = proxy
	}
	o.server = &http.Server{
		ConnContext: func(ctx context.Context, conn net.Conn) context.Context {
			return context.WithValue(ctx, task1ConnKey{}, conn)
		},
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			target.mesh.Refresh()
			peer, pinned := target.mesh.VerifyClientPin(r)
			if !pinned {
				http.Error(w, "unpinned fixture caller", http.StatusForbidden)
				return
			}
			proxy := proxies[peer]
			if proxy == nil {
				http.Error(w, "unknown fixture caller", http.StatusForbidden)
				return
			}
			o.mu.Lock()
			o.counts[task1RequestKey{peer, r.Method, r.URL.Path}]++
			if r.URL.Path == task1UpgradePath {
				if o.blockUpgrade {
					o.mu.Unlock()
					http.Error(w, "fixture disconnected", http.StatusServiceUnavailable)
					return
				}
				conn, ok := r.Context().Value(task1ConnKey{}).(net.Conn)
				if !ok {
					o.problems = append(o.problems, "upgrade lacks connection")
					o.mu.Unlock()
					http.Error(w, "missing connection", http.StatusInternalServerError)
					return
				}
				o.upgrades[peer] = append(o.upgrades[peer], conn)
			}
			o.mu.Unlock()
			if r.URL.Path == task1RosterPath {
				body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
				var roster struct {
					Members []struct {
						NodeUUID string `json:"nodeUuid"`
					} `json:"members"`
				}
				if err != nil || json.Unmarshal(body, &roster) != nil || len(roster.Members) == 0 || roster.Members[0].NodeUUID != peer {
					o.mu.Lock()
					o.problems = append(o.problems, "roster was forwarded under another member's identity")
					o.mu.Unlock()
					http.Error(w, "invalid pairwise roster", http.StatusBadRequest)
					return
				}
				r.Body = io.NopCloser(bytes.NewReader(body))
			}
			proxy.ServeHTTP(w, r)
		}),
	}
	serveDone := make(chan error, 1)
	go func() { serveDone <- o.server.Serve(tls.NewListener(ln, target.mesh.ServerTLSConfig())) }()
	t.Cleanup(func() {
		for _, n := range nodes {
			o.interrupt(n.uuid, true)
		}
		if err := o.server.Close(); err != nil {
			t.Error(err)
		}
		if err := <-serveDone; err != http.ErrServerClosed {
			t.Errorf("observer server exit: %v", err)
		}
		o.mu.Lock()
		defer o.mu.Unlock()
		if len(o.problems) != 0 {
			t.Errorf("observation failures: %v", o.problems)
		}
	})
	return o
}

func task1WriteJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func newTask1Nodes(t *testing.T) ([]*task1Node, int) {
	t.Helper()
	blocked, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	blockedPort := blocked.Addr().(*net.TCPAddr).Port
	blockedDone := make(chan struct{})
	go func() {
		defer close(blockedDone)
		for {
			c, err := blocked.Accept()
			if err != nil {
				return
			}
			_ = c.Close()
		}
	}()
	t.Cleanup(func() { _ = blocked.Close(); <-blockedDone })
	nodes := []*task1Node{}
	for _, uuid := range []string{"task1-reachable", "task1-restricted", "task1-third"} {
		base, err := os.MkdirTemp("", "p1u4-")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := os.RemoveAll(base); err != nil {
				t.Error(err)
			}
		})
		dir := filepath.Join(base, "cluster")
		if err := os.MkdirAll(filepath.Join(dir, "trusted"), 0o700); err != nil {
			t.Fatal(err)
		}
		n := &task1Node{uuid: uuid, base: base, ports: make(map[noderec.ServiceKey]int), observers: make(map[noderec.ServiceKey]*task1Observer)}
		n.cert = mintClusterIdentity(t, dir, uuid)
		task1WriteJSON(t, filepath.Join(dir, "identity.json"), map[string]any{"node_uuid": uuid, "created_at": 1})
		task1WriteJSON(t, filepath.Join(dir, "admission.json"), map[string]any{"counter": 1, "activated": 1, "clusterId": task1ClusterID, "epoch": 1})
		for _, service := range []noderec.ServiceKey{noderec.ServiceCluster, noderec.ServiceWorkload, noderec.ServiceErrors} {
			n.ports[service] = temporaryClusterPort(t)
		}
		nodes = append(nodes, n)
	}
	for _, n := range nodes {
		for _, peer := range nodes {
			if n == peer {
				continue
			}
			block, _ := pem.Decode([]byte(peer.cert))
			if block == nil {
				t.Fatal("fixture certificate decode")
			}
			fingerprint := sha256.Sum256(block.Bytes)
			task1WriteJSON(t, filepath.Join(n.base, "cluster", "trusted", peer.uuid+".json"), map[string]any{
				"nodeUuid": peer.uuid, "nodeId": peer.uuid, "name": "stale peer name", "certPem": peer.cert,
				"certFingerprint": "sha256:" + hex.EncodeToString(fingerprint[:]), "clusterId": task1ClusterID, "admissionEpoch": 1,
			})
		}
		n.mesh = clustertrust.Open(filepath.Join(n.base, "cluster"))
	}
	for _, n := range nodes {
		for _, service := range []noderec.ServiceKey{noderec.ServiceCluster, noderec.ServiceWorkload, noderec.ServiceErrors} {
			n.observers[service] = newTask1Observer(t, n, service, nodes)
		}
	}
	for _, n := range nodes {
		var members []task1Member
		for _, peer := range nodes {
			port := peer.observers[noderec.ServiceCluster].port
			if peer == nodes[1] {
				port = blockedPort
			}
			members = append(members, task1Member{ID: peer.uuid, NodeUUID: peer.uuid, Name: "stale peer name", IPAddress: "127.0.0.1", Port: port, ClusterID: task1ClusterID, AdmissionEpoch: 1, State: "member"})
		}
		task1WriteJSON(t, filepath.Join(n.base, "cluster", "members.json"), members)
		n.cm = startTask1Process(t, n.base, clusterMgrBin, "--config-dir", n.base, "--port", strconv.Itoa(n.ports[noderec.ServiceCluster]))
		n.cm.call(t, "cluster:get-node-id", nil)
		var services []noderec.RegisterParams
		for _, service := range []noderec.ServiceKey{noderec.ServiceCluster, noderec.ServiceWorkload, noderec.ServiceErrors} {
			services = append(services, noderec.RegisterParams{Service: service, Port: n.observers[service].port})
		}
		n.cm.call(t, "cluster:set-local-services", map[string]any{"generation": 1, "services": services})
		if n == nodes[2] {
			continue
		}
		n.workloads = startTask1Process(t, n.base, workloadMgrBin, "--cluster-dir", filepath.Join(n.base, "cluster"), "--port", strconv.Itoa(n.ports[noderec.ServiceWorkload]))
		n.errors = startTask1Process(t, n.base, errorsBin, "--cluster-dir", filepath.Join(n.base, "cluster"), "--port", strconv.Itoa(n.ports[noderec.ServiceErrors]), "--peer-sync", "--node-id", n.uuid)
		n.workloads.call(t, "log/set-level", map[string]string{"level": "info"})
		n.errors.call(t, "errors:get-initial", nil)
		waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(n.ports[noderec.ServiceWorkload])), 5*time.Second)
		waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(n.ports[noderec.ServiceErrors])), 5*time.Second)
	}
	return nodes, blockedPort
}

func task1Directory(node *task1Node, blockedPort int) noderec.DirectoryNode {
	services := make(map[noderec.ServiceKey]noderec.ServiceStatus)
	for _, service := range []noderec.ServiceKey{noderec.ServiceCluster, noderec.ServiceWorkload, noderec.ServiceErrors} {
		port := node.observers[service].port
		if node.uuid == "task1-restricted" {
			port = blockedPort
		}
		services[service] = noderec.ServiceStatus{Port: port}
	}
	return noderec.DirectoryNode{HostUUID: node.uuid, ClusterUUID: node.uuid, Name: node.uuid, IP: "127.0.0.1", Services: services}
}

func task1Workload(origin, id, state string, seq int) map[string]any {
	return map[string]any{"workloadInfo": map[string]any{"id": id, "model": "task1-model", "engine": "ollama", "runId": "task1-run", "originatedFrom": origin, "state": state, "seq": seq, "createdAt": 1000}}
}

func task1Upserts(t *testing.T, process *task1Process, origin, id, state string) int {
	t.Helper()
	process.mu.Lock()
	defer process.mu.Unlock()
	count := 0
	for _, f := range process.frames {
		if f.Method != "workloads:upsert" {
			continue
		}
		var params struct {
			WorkloadInfo struct {
				wlInfo
				RunID     string `json:"runId"`
				Seq       int    `json:"seq"`
				CreatedAt int64  `json:"createdAt"`
			} `json:"workloadInfo"`
		}
		if err := json.Unmarshal(f.Params, &params); err != nil {
			t.Fatal(err)
		}
		if params.WorkloadInfo.OriginatedFrom == origin && params.WorkloadInfo.ID == id && params.WorkloadInfo.State == state {
			wantSeq := 1
			if state == "completed" {
				wantSeq = 2
			}
			w := params.WorkloadInfo
			if w.Model != "task1-model" || w.Engine != "ollama" || w.RunID != "task1-run" || w.Seq != wantSeq || w.CreatedAt != 1000 {
				t.Fatalf("delivered workload metadata=%+v want model=task1-model engine=ollama run=task1-run seq=%d createdAt=1000", w, wantSeq)
			}
			count++
		}
	}
	return count
}

func task1Errors(t *testing.T, process *task1Process) []errorswire.ServiceError {
	t.Helper()
	return decodeResult[[]errorswire.ServiceError](t, process.call(t, "errors:get-initial", nil))
}

func task1ErrorUpdates(t *testing.T, process *task1Process, want []errorswire.ServiceError) int {
	t.Helper()
	process.mu.Lock()
	defer process.mu.Unlock()
	count := 0
	for _, frame := range process.frames {
		if frame.Method != "errors:update" {
			continue
		}
		var snapshot []errorswire.ServiceError
		if err := json.Unmarshal(frame.Params, &snapshot); err != nil {
			t.Fatal(err)
		}
		if reflect.DeepEqual(snapshot, want) {
			count++
		}
	}
	return count
}

func TestReversePeerControlTask1(t *testing.T) {
	nodes, blockedPort := newTask1Nodes(t)
	a, b, c := nodes[0], nodes[1], nodes[2]
	aCluster := a.observers[noderec.ServiceCluster]
	bCluster := b.observers[noderec.ServiceCluster]
	bWorkload := b.observers[noderec.ServiceWorkload]
	bErrors := b.observers[noderec.ServiceErrors]
	for _, n := range []*task1Node{a, b} {
		peer := b
		if n == b {
			peer = a
		}
		directory := noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{task1Directory(peer, blockedPort)}}
		n.workloads.send(t, noderec.NotifyNodes, directory, false)
	}

	t.Run("blocked direct endpoints and pairwise roster convergence", func(t *testing.T) {
		cfg, ok := a.mesh.ClientTLSConfig(b.uuid)
		if !ok {
			t.Fatal("missing restricted member pin")
		}
		waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(blockedPort)), time.Second)
		client := &http.Client{Timeout: time.Second, Transport: &http.Transport{TLSClientConfig: cfg}}
		defer client.CloseIdleConnections()
		for _, path := range []string{task1RosterPath, "/v1/workloads/events", "/v1/errors"} {
			response, err := client.Post("https://127.0.0.1:"+strconv.Itoa(blockedPort)+path, "application/json", strings.NewReader("{}"))
			if response != nil {
				response.Body.Close()
			}
			if err == nil {
				t.Fatalf("restricted direct endpoint %s answered", path)
			}
		}
		hostname, err := os.Hostname()
		if err != nil {
			t.Fatal(err)
		}
		// Each member starts with stale peer display names. Merely preserving
		// the seeded three-row roster cannot satisfy convergence below.
		for _, receiver := range nodes {
			initial := decodeResult[struct {
				Nodes []task1Member `json:"nodes"`
			}](t, receiver.cm.call(t, "nodes:get-initial", nil))
			stale := 0
			for _, row := range initial.Nodes {
				if row.NodeUUID != receiver.uuid && row.Name == "stale peer name" {
					stale++
				}
			}
			if stale != 2 {
				t.Fatalf("%s initial stale peers=%d want=2", receiver.uuid, stale)
			}
		}
		for _, receiver := range nodes {
			task1Eventually(t, receiver.uuid+" converged roster", 40*time.Second, func() bool {
				var result struct {
					Nodes []task1Member `json:"nodes"`
				}
				result = decodeResult[struct {
					Nodes []task1Member `json:"nodes"`
				}](t, receiver.cm.call(t, "nodes:get-initial", nil))
				if len(result.Nodes) != 3 {
					return false
				}
				for _, row := range result.Nodes {
					if row.Name != hostname || row.ClusterID != task1ClusterID || row.AdmissionEpoch != 1 || row.State != "member" {
						return false
					}
				}
				for _, sender := range nodes {
					found := false
					for _, row := range result.Nodes {
						if row.NodeUUID == sender.uuid {
							found = true
						}
					}
					if !found {
						return false
					}
					if sender != receiver && receiver.observers[noderec.ServiceCluster].count(sender.uuid, http.MethodPost, task1RosterPath) == 0 {
						return false
					}
				}
				return true
			})
		}
		if aCluster.count(b.uuid, http.MethodGet, task1UpgradePath) == 0 {
			t.Fatal("restricted member never opened outbound reverse session")
		}
		if bCluster.count(a.uuid, http.MethodGet, task1UpgradePath) != 0 {
			t.Fatal("reachable member opened a direct reverse session into restricted member")
		}
		for _, receiver := range nodes {
			for _, sender := range nodes {
				if sender != receiver {
					t.Logf("pairwise_roster source=%s target=%s requests=%d", sender.uuid, receiver.uuid, receiver.observers[noderec.ServiceCluster].count(sender.uuid, http.MethodPost, task1RosterPath))
				}
			}
		}
	})

	t.Run("two way workloads and accepted event dedup after reconnect", func(t *testing.T) {
		// Wait for the workload service's normal discovery tick; the fence request
		// itself cannot force that asynchronous target-set refresh.
		time.Sleep(6 * time.Second)
		a.workloads.send(t, "workload:started", task1Workload(a.uuid, "a-to-b", "running", 1), false)
		b.workloads.send(t, "workload:started", task1Workload(b.uuid, "b-to-a", "running", 1), false)
		task1Eventually(t, "both workload origins", 10*time.Second, func() bool {
			return task1Upserts(t, b.workloads, a.uuid, "a-to-b", "running") == 1 && task1Upserts(t, a.workloads, b.uuid, "b-to-a", "running") == 1
		})
		before := bWorkload.count(a.uuid, http.MethodPost, "/v1/workloads/events")
		cut := make(chan struct{})
		var once sync.Once
		bWorkload.mu.Lock()
		bWorkload.afterReply = func(peer, path string) {
			if path == "/v1/workloads/events" {
				once.Do(func() { aCluster.interrupt(b.uuid, true); close(cut) })
			}
		}
		bWorkload.mu.Unlock()
		frame := task1Workload(a.uuid, "accepted-before-disconnect", "running", 1)
		a.workloads.send(t, "workload:started", frame, false)
		select {
		case <-cut:
		case <-time.After(10 * time.Second):
			t.Fatal("accepted workload did not reach disconnect boundary")
		}
		task1Eventually(t, "accepted receiver event", 5*time.Second, func() bool { return task1Upserts(t, b.workloads, a.uuid, "accepted-before-disconnect", "running") == 1 })
		bWorkload.mu.Lock()
		bWorkload.afterReply = nil
		bWorkload.mu.Unlock()
		upgrades := aCluster.established(b.uuid)
		aCluster.mu.Lock()
		aCluster.blockUpgrade = false
		aCluster.mu.Unlock()
		task1Eventually(t, "new outbound reverse session", 10*time.Second, func() bool { return aCluster.established(b.uuid) > upgrades })
		a.workloads.send(t, "workload:started", frame, false)
		task1Eventually(t, "redelivery after reconnect", 10*time.Second, func() bool { return bWorkload.count(a.uuid, http.MethodPost, "/v1/workloads/events") >= before+2 })
		a.workloads.send(t, "workload:completed", task1Workload(a.uuid, "accepted-before-disconnect", "completed", 2), false)
		task1Eventually(t, "completed workload fence", 10*time.Second, func() bool {
			return task1Upserts(t, b.workloads, a.uuid, "accepted-before-disconnect", "completed") == 1
		})
		if got := task1Upserts(t, b.workloads, a.uuid, "accepted-before-disconnect", "running"); got != 1 {
			t.Fatalf("accepted event mutated receiver %d times, want 1", got)
		}
		b.workloads.send(t, "workload:started", task1Workload(b.uuid, "b-to-a", "running", 1), false)
		b.workloads.send(t, "workload:completed", task1Workload(b.uuid, "b-to-a", "completed", 2), false)
		task1Eventually(t, "opposite direction completion fence", 10*time.Second, func() bool { return task1Upserts(t, a.workloads, b.uuid, "b-to-a", "completed") == 1 })
		if got := task1Upserts(t, a.workloads, b.uuid, "b-to-a", "running"); got != 1 {
			t.Fatalf("opposite direction redelivery mutated receiver %d times, want 1", got)
		}
		t.Logf("workload_reconnect requests=%d running_mutations=1 completed_mutations=1", bWorkload.count(a.uuid, http.MethodPost, "/v1/workloads/events")-before)
	})

	t.Run("two way errors and explicit snapshot recovery", func(t *testing.T) {
		a.errors.send(t, noderec.NotifyNodes, noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{task1Directory(b, blockedPort)}}, false)
		b.errors.send(t, noderec.NotifyNodes, noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{task1Directory(a, blockedPort)}}, false)
		aError := errorswire.ServiceError{ID: "a-error", NodeID: a.uuid, Message: "fixture a", Timestamp: 1000, Severity: "error"}
		bError := errorswire.ServiceError{ID: "b-error", NodeID: b.uuid, Message: "fixture b", Timestamp: 1000, Severity: "warning"}
		a.errors.call(t, "errors:report", aError)
		b.errors.call(t, "errors:report", bError)
		want := []errorswire.ServiceError{aError, bError}
		defer func() {
			if t.Failed() {
				t.Logf("error diagnostic A=%+v B=%+v A_stderr=%s B_stderr=%s", task1Errors(t, a.errors), task1Errors(t, b.errors), a.errors.output.String(), b.errors.output.String())
			}
		}()
		task1Eventually(t, "both error origins", 10*time.Second, func() bool {
			return reflect.DeepEqual(task1Errors(t, a.errors), want) && reflect.DeepEqual(task1Errors(t, b.errors), want)
		})
		cut := make(chan struct{})
		var once sync.Once
		bErrors.mu.Lock()
		bErrors.afterReply = func(peer, path string) {
			if path == "/v1/errors" {
				once.Do(func() { aCluster.interrupt(b.uuid, true); close(cut) })
			}
		}
		bErrors.mu.Unlock()
		aError.Timestamp = 2000
		aError.Message = "accepted before disconnect"
		a.errors.call(t, "errors:report", aError)
		select {
		case <-cut:
		case <-time.After(10 * time.Second):
			t.Fatal("error snapshot did not reach disconnect boundary")
		}
		want = []errorswire.ServiceError{aError, bError}
		task1Eventually(t, "accepted snapshot remains", 5*time.Second, func() bool { return reflect.DeepEqual(task1Errors(t, b.errors), want) })
		before := bErrors.count(a.uuid, http.MethodPost, "/v1/errors")
		a.errors.call(t, "errors:clear", errorswire.ClearParams{ID: aError.ID})
		bError.Timestamp = 2000
		bError.Message = "opposite direction during reverse outage"
		b.errors.call(t, "errors:report", bError)
		want = []errorswire.ServiceError{aError, bError}
		task1Eventually(t, "opposite direction error during outage", 5*time.Second, func() bool { return reflect.DeepEqual(task1Errors(t, a.errors), []errorswire.ServiceError{bError}) })
		// This exceeds the production push timeout: the clear cannot be delivered
		// while B's outbound session is rejected and its direct er is inaccessible.
		time.Sleep(4 * time.Second)
		if got := task1Errors(t, b.errors); !reflect.DeepEqual(got, want) {
			t.Fatalf("disconnected receiver changed without a delivered snapshot: %+v", got)
		}
		if got := bErrors.count(a.uuid, http.MethodPost, "/v1/errors"); got != before {
			t.Fatalf("disconnected snapshot dispatched %d extra requests", got-before)
		}
		bErrors.mu.Lock()
		bErrors.afterReply = nil
		bErrors.mu.Unlock()
		upgrades := aCluster.established(b.uuid)
		aCluster.mu.Lock()
		aCluster.blockUpgrade = false
		aCluster.mu.Unlock()
		task1Eventually(t, "errors reverse reconnect", 10*time.Second, func() bool { return aCluster.established(b.uuid) > upgrades })
		if got := bErrors.count(a.uuid, http.MethodPost, "/v1/errors"); got != before {
			t.Fatalf("lost snapshot replayed on reconnect: requests=%d want=%d", got, before)
		}
		// A new local report is the public explicit full-snapshot trigger. It
		// must evict the cleared A error while preserving B's local origin.
		replacement := errorswire.ServiceError{ID: "a-replacement", NodeID: a.uuid, Message: "explicit next snapshot", Timestamp: 3000}
		a.errors.call(t, "errors:report", replacement)
		want = []errorswire.ServiceError{replacement, bError}
		task1Eventually(t, "explicit full snapshot convergence", 10*time.Second, func() bool {
			return reflect.DeepEqual(task1Errors(t, a.errors), want) && reflect.DeepEqual(task1Errors(t, b.errors), want)
		})
		if got := bErrors.count(a.uuid, http.MethodPost, "/v1/errors"); got != before+1 {
			t.Fatalf("explicit snapshot requests=%d want=%d", got, before+1)
		}
		if got := task1ErrorUpdates(t, b.errors, want); got != 1 {
			t.Fatalf("recovered snapshot mutations=%d want=1", got)
		}
		duplicateReply := make(chan struct{})
		var duplicateOnce sync.Once
		bErrors.mu.Lock()
		bErrors.afterReply = func(peer, path string) {
			if peer == a.uuid && path == "/v1/errors" {
				duplicateOnce.Do(func() { close(duplicateReply) })
			}
		}
		bErrors.mu.Unlock()
		a.errors.call(t, "errors:report", replacement)
		select {
		case <-duplicateReply:
		case <-time.After(10 * time.Second):
			t.Fatal("duplicate snapshot did not complete")
		}
		if got := task1Errors(t, b.errors); !reflect.DeepEqual(got, want) {
			t.Fatalf("duplicate snapshot altered state: %+v", got)
		}
		if got := task1ErrorUpdates(t, b.errors, want); got != 1 {
			t.Fatalf("duplicate snapshot emitted %d meaningful mutations, want 1", got)
		}
		bErrors.mu.Lock()
		bErrors.afterReply = nil
		bErrors.mu.Unlock()
		t.Log("error_snapshot accepted_before_disconnect=true automatic_replay=0 explicit_recovery=1 duplicate_mutations=0 entries=2")
	})

	t.Run("reverse removal loses reply without replay and stays durable", func(t *testing.T) {
		result := decodeResult[struct {
			Removed bool `json:"removed"`
		}](t, a.cm.call(t, "nodes:remove", map[string]string{"nodeUuid": b.uuid}))
		if !result.Removed {
			t.Fatal("initiator did not commit removal")
		}
		task1Eventually(t, "restricted member removed", 10*time.Second, func() bool {
			return decodeResult[cmNodeID](t, b.cm.call(t, "cluster:get-node-id", nil)).ClusterID == ""
		})
		if got := bCluster.count(a.uuid, http.MethodPost, task1RemovePath); got != 1 {
			t.Fatalf("remote removal requests=%d want=1", got)
		}
		if !strings.Contains(a.cm.output.String(), "notify removal to ") {
			t.Fatal("removal reply loss was not observed by initiator")
		}
		admission := readOutboundJSON[struct {
			ClusterID string `json:"clusterId"`
			Epoch     uint64 `json:"epoch"`
		}](t, filepath.Join(b.base, "cluster", "admission.json"))
		if admission.ClusterID != "" || admission.Epoch != 0 {
			t.Fatalf("durable admission retained: %+v", admission)
		}
		if members := readOutboundJSON[[]task1Member](t, filepath.Join(b.base, "cluster", "members.json")); len(members) != 0 {
			t.Fatalf("durable removed roster=%+v", members)
		}
		pins, err := os.ReadDir(filepath.Join(b.base, "cluster", "trusted"))
		if err != nil {
			t.Fatal(err)
		}
		if len(pins) != 0 {
			t.Fatalf("removed member retained %d pins", len(pins))
		}
		b.cm.stop()
		b.cm = startTask1Process(t, b.base, clusterMgrBin, "--config-dir", b.base, "--port", strconv.Itoa(b.ports[noderec.ServiceCluster]))
		if id := decodeResult[cmNodeID](t, b.cm.call(t, "cluster:get-node-id", nil)); id.ClusterID != "" {
			t.Fatalf("restart resurrected removed admission: %+v", id)
		}
		second := decodeResult[struct {
			Removed bool `json:"removed"`
		}](t, a.cm.call(t, "nodes:remove", map[string]string{"nodeUuid": b.uuid}))
		if second.Removed {
			t.Fatal("second removal applied another mutation")
		}
		if got := bCluster.count(a.uuid, http.MethodPost, task1RemovePath); got != 1 {
			t.Fatalf("removal notification replayed: %d", got)
		}
		task1Eventually(t, "third member learns removal", 10*time.Second, func() bool {
			result := decodeResult[struct {
				Nodes []task1Member `json:"nodes"`
			}](t, c.cm.call(t, "nodes:get-initial", nil))
			if len(result.Nodes) != 2 {
				return false
			}
			for _, row := range result.Nodes {
				if row.NodeUUID == b.uuid {
					return false
				}
			}
			return true
		})
		t.Log("removal requests=1 lost_reply=true initiator_commit=true restarted_admission_empty=true second_mutation=false")
	})

	t.Run("third member never relays events and temporary processes stop", func(t *testing.T) {
		for _, service := range []noderec.ServiceKey{noderec.ServiceWorkload, noderec.ServiceErrors} {
			observer := c.observers[service]
			observer.mu.Lock()
			count := len(observer.counts)
			observer.mu.Unlock()
			if count != 0 {
				t.Fatalf("third member received %s events: %d request keys", service, count)
			}
		}
		for _, n := range nodes {
			if n.workloads != nil {
				n.workloads.stop()
				n.errors.stop()
			}
			n.cm.stop()
			for _, port := range n.ports {
				conn, err := net.DialTimeout("tcp", "127.0.0.1:"+strconv.Itoa(port), time.Second)
				if err == nil {
					conn.Close()
					t.Fatalf("temporary service listener %d survived cleanup", port)
				}
			}
		}
		t.Log("cleanup child_exit=0 service_listeners_rejected=true third_member_event_requests=0")
	})
}
