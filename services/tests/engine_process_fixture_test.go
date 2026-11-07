// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
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
	"strconv"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
)

// Use the running test executable, including its fixed windows-test-bin path,
// as the engine image. Dispatch before TestMain, without rebuilding products.
func init() {
	if image := os.Getenv("NVPAIR_TEST_SWITCH_ENGINE"); image != "" && image == os.Args[0] {
		if err := runFixtureEngine(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
}

func runFixtureEngine() error {
	port, err := strconv.Atoi(os.Getenv("NVPAIR_TEST_SWITCH_PORT"))
	if err != nil || validateFixturePort(port) != nil {
		return errors.New("fixture engine requires an isolated port")
	}
	ln, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
	if err != nil {
		return err
	}
	controller := os.Getenv("NVPAIR_TEST_SWITCH_CONTROL")
	name := os.Getenv("NVPAIR_TEST_SWITCH_NAME")
	client := &http.Client{Timeout: 90 * time.Second}
	notify := func(path string) error {
		response, err := client.Get(controller + path + "?engine=" + name + "&pid=" + strconv.Itoa(os.Getpid()))
		if err != nil {
			return err
		}
		defer response.Body.Close()
		if _, err := io.Copy(io.Discard, response.Body); err != nil {
			return err
		}
		if response.StatusCode != http.StatusOK {
			return fmt.Errorf("fixture controller: HTTP %d", response.StatusCode)
		}
		return nil
	}
	if err := notify("/started"); err != nil {
		ln.Close()
		return err
	}
	return http.Serve(ln, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/chat/completions" {
			if err := notify("/inflight"); err != nil {
				http.Error(w, err.Error(), http.StatusBadGateway)
				return
			}
			fmt.Fprintln(w, `{"id":"fixture","object":"chat.completion","model":"switch-model","choices":[{"index":0,"message":{"role":"assistant","content":"finished"},"finish_reason":"stop"}]}`)
			return
		}
		switch r.URL.Path {
		case "/v1/models", "/models":
			if os.Getenv("NVPAIR_TEST_SWITCH_EMPTY_MODELS") == "1" {
				fmt.Fprintln(w, `{"object":"list","data":[]}`)
			} else {
				fmt.Fprintln(w, `{"object":"list","data":[{"id":"switch-model","status":{"value":"unloaded"}}]}`)
			}
		default:
			fmt.Fprintln(w, `{"version":"fixture"}`)
		}
	}))
}

type engineProcessFixture struct {
	proxyControl        *fixtureProxyControl
	base                string
	p                   *interopProcess
	ports               fixtureServicePorts
	sourcePort          int
	targetPort          int
	facadePort          int
	targetFacadePort    int
	control             *httptest.Server
	release             chan struct{}
	releaseOne          sync.Once
	targetRelease       chan struct{}
	targetReleaseOne    sync.Once
	inflight            chan struct{}
	mu                  sync.Mutex
	starts              map[string][]int32
	identities          []serviceProcessIdentity
	engineIdentities    map[string][]serviceProcessIdentity
	lifetimes           map[serviceProcessIdentity]*fixtureProcessLifetime
	effectError         error
	inferences          map[string]int
	failTarget          bool
	checkSourceExit     bool
	sourceAliveAtTarget bool
	scanner             string
	scannerPort         int
}

func newEngineProcessFixture(t *testing.T) *engineProcessFixture {
	t.Helper()
	return newEngineProcessFixtureAt(t, t.TempDir(), reserveFixtureServicePorts(t))
}

func newEngineProcessFixtureAt(t *testing.T, base string, ports fixtureServicePorts) *engineProcessFixture {
	t.Helper()
	f := prepareEngineProcessFixture(t, base, ports, "")
	f.start(t)
	f.rpcOK(t, "engine:install", map[string]any{"engine": "llamacpp-custom"})
	// Initialize every settings baseline before tests capture their revisions.
	for _, engine := range engines.Names() {
		f.rpcOK(t, "engine:get-settings", map[string]string{"engine": engine})
	}
	return f
}

// Preparation is separate from launch so a service-owned broker can supervise
// this same isolated manager/proxy stack without starting a second broker.
// A supplied scanner is an already prepared test-image slot owned by its caller.
func prepareEngineProcessFixture(t *testing.T, base string, ports fixtureServicePorts, preparedScanner string) *engineProcessFixture {
	t.Helper()
	f := &engineProcessFixture{base: base, ports: ports, sourcePort: freePort(t), targetPort: freePort(t), facadePort: freePort(t), targetFacadePort: freePort(t), release: make(chan struct{}), targetRelease: make(chan struct{}), inflight: make(chan struct{}, 1), starts: make(map[string][]int32), engineIdentities: make(map[string][]serviceProcessIdentity), inferences: make(map[string]int)}
	f.lifetimes = make(map[serviceProcessIdentity]*fixtureProcessLifetime)
	f.control = newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pid, err := strconv.ParseInt(r.URL.Query().Get("pid"), 10, 32)
		if err != nil {
			http.Error(w, "invalid fixture PID", http.StatusBadRequest)
			return
		}
		if r.URL.Path == "/started" {
			identity, identityErr := serviceProcessIdentityAt(r.Context(), int32(pid))
			var proof *fixtureProcessLifetime
			if identityErr == nil {
				// Retain the live process before this registration returns and the
				// engine can answer its first readiness probe.
				proof, identityErr = retainFixtureProcessLifetime(identity)
			}
			f.mu.Lock()
			name := r.URL.Query().Get("engine")
			if identityErr != nil {
				f.effectError = errors.Join(f.effectError, identityErr)
			} else {
				if previous := f.lifetimes[identity]; previous != nil {
					if err := proof.close(); err != nil {
						f.effectError = errors.Join(f.effectError, err)
					}
				} else {
					f.lifetimes[identity] = proof
				}
				f.identities = append(f.identities, identity)
				f.engineIdentities[name] = append(f.engineIdentities[name], identity)
			}
			f.starts[name] = append(f.starts[name], int32(pid))
			if name == "llamacpp" && f.checkSourceExit {
				for _, source := range f.engineIdentities["llamacpp-custom"] {
					stopped, err := f.lifetimes[source].stopped(r.Context())
					if err != nil {
						f.effectError = errors.Join(f.effectError, err)
					}
					f.sourceAliveAtTarget = f.sourceAliveAtTarget || !stopped
				}
			}
			fail := name == "llamacpp" && f.failTarget
			f.mu.Unlock()
			if fail {
				http.Error(w, "fixture startup refused", http.StatusServiceUnavailable)
				return
			}
		} else if r.URL.Path == "/inflight" {
			f.mu.Lock()
			name := r.URL.Query().Get("engine")
			f.inferences[name]++
			f.mu.Unlock()
			select {
			case f.inflight <- struct{}{}:
			default:
			}
			release := f.release
			if name == "llamacpp" {
				release = f.targetRelease
			}
			select {
			case <-release:
			case <-r.Context().Done():
			}
		}
	}))
	t.Cleanup(func() { f.unblock(); f.control.Close() })
	t.Cleanup(func() {
		f.mu.Lock()
		defer f.mu.Unlock()
		for identity, proof := range f.lifetimes {
			if err := proof.close(); err != nil {
				t.Errorf("close fixture lifetime %+v: %v", identity, err)
			}
		}
	})
	t.Cleanup(func() {
		f.unblock()
		f.mu.Lock()
		identities := append([]serviceProcessIdentity(nil), f.identities...)
		effectErr := f.effectError
		f.mu.Unlock()
		if effectErr != nil {
			t.Errorf("record fixture engine identity/effect: %v", effectErr)
		}
		// The engine image is the parent test executable. Never scan by that
		// path: signal only controller-recorded PID/path/creation identities.
		for _, identity := range identities {
			stopped, err := f.processStopped(context.Background(), identity)
			if err != nil {
				t.Errorf("inspect fixture engine: %v", err)
				continue
			}
			if !stopped {
				inventory := &serviceProcessInventory{paths: []string{identity.Path}}
				if err := inventory.kill(identity); err != nil && !errors.Is(err, errServiceProcessDeparted) {
					t.Errorf("cleanup fixture engine: %v", err)
					continue
				}
			}
			f.assertProcessStopped(t, identity)
		}
	})
	image, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// Reuse the empty-directory scanner instead of browsing host multicast.
	// The fixed executable role already has a Windows test-bin manifest entry.
	f.scanner = preparedScanner
	if f.scanner == "" {
		f.scanner = filepath.Join(testbin.Dir(t, "tests", "subscription-scanner", "subscription-scanner"), testbin.Exe("subscription-scanner"))
		jointCopyExecutable(t, image, f.scanner)
	}
	f.scannerPort = freePort(t)
	f.proxyControl = newFixtureProxyControl(t)
	jointIsolateEngines(t, f.base, freePort(t))
	for _, engine := range []struct {
		name string
		port int
	}{{"llamacpp-custom", f.sourcePort}, {"llamacpp", f.targetPort}} {
		// Overrides deep-merge with bundled manifests. Clear llama.cpp's
		// router-role predicate: this fixture serves only a generic HTTP probe.
		probe := map[string]any{"http": "http://127.0.0.1:{port}/", "status": http.StatusOK, "timeout_s": 10, "interval_s": 60, "json_match": nil}
		rt := map[string]any{"bin": image, "port": engine.port, "bind": "127.0.0.1", "ready": probe, "health": probe,
			"env":  map[string]string{"NVPAIR_TEST_SWITCH_ENGINE": image, "NVPAIR_TEST_SWITCH_NAME": engine.name, "NVPAIR_TEST_SWITCH_PORT": "{port}", "NVPAIR_TEST_SWITCH_CONTROL": f.control.URL},
			"stop": map[string]any{"signal": "term", "grace_s": 1}}
		jointWriteJSON(t, filepath.Join(jointAppDir(f.base), "engines", engine.name+".json"), privateFixtureManifest(t, engine.name, filepath.Join(f.base, "absent-models", engine.name), image, rt, engines.ProtocolOpenAIV1))
	}
	journal := make(map[string]any)
	for _, facade := range []struct {
		name   string
		server int
		port   int
	}{{"llamacpp-custom", f.sourcePort, f.facadePort}, {"llamacpp", f.targetPort, f.targetFacadePort}} {
		if facade.name == "llamacpp-custom" {
			jointWriteJSON(t, filepath.Join(jointAppDir(f.base), privateFixtureEngine(t, facade.name).PortFile), map[string]int{"port": facade.port})
		}
		// Explicit settings bypass managed default-port ownership. Never bind
		// beside or dial an installed PAIR's default engine facade.
		journal[facade.name] = map[string]any{"explicit": true, "snapshot": map[string]any{
			"engine": facade.name, "revision": 1, "appliedRevision": 1, "phase": "idle", "format": "pair-arguments-v1",
			"settings": map[string]any{"serverPort": facade.server, "proxyPort": facade.port, "launchText": ""},
		}}
	}
	// The broker keeps its settings journal beside --cluster-dir, not in the
	// manager's application directory.
	jointWriteJSON(t, filepath.Join(f.base, "engine-settings-operations.json"), journal)
	return f
}

func (f *engineProcessFixture) unblockSource() { f.releaseOne.Do(func() { close(f.release) }) }

func (f *engineProcessFixture) unblockTarget() {
	f.targetReleaseOne.Do(func() { close(f.targetRelease) })
}

func (f *engineProcessFixture) unblock() { f.unblockSource(); f.unblockTarget() }

func (f *engineProcessFixture) processLifetime(identity serviceProcessIdentity) (*fixtureProcessLifetime, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	proof := f.lifetimes[identity]
	if proof == nil {
		return nil, fmt.Errorf("fixture lifetime was not retained: %+v", identity)
	}
	return proof, nil
}

func (f *engineProcessFixture) processStopped(ctx context.Context, identity serviceProcessIdentity) (bool, error) {
	proof, err := f.processLifetime(identity)
	if err != nil {
		return false, err
	}
	return proof.stopped(ctx)
}

func (f *engineProcessFixture) assertProcessStopped(t *testing.T, identity serviceProcessIdentity) {
	t.Helper()
	proof, err := f.processLifetime(identity)
	if err != nil {
		t.Fatal(err)
	}
	if stopped, err := proof.wait(10 * time.Second); err != nil || !stopped {
		t.Fatalf("inspect engine lifetime %+v: stopped=%t error=%v", identity, stopped, err)
	}
}

// A later process may reuse a departed engine's PID, including another fixture
// engine. Observe the recorded lifetime, just as kill revalidates it before a
// signal; a reused PID is neither a survivor nor authority to signal its owner.
func fixtureProcessIdentityStopped(ctx context.Context, expected serviceProcessIdentity) (bool, error) {
	stopped, err := serviceProcessStopped(ctx, expected.PID)
	if err != nil || stopped {
		return stopped, err
	}
	current, err := serviceProcessIdentityAt(ctx, expected.PID)
	if err != nil {
		// Exit can race the metadata read. Only positive exit evidence resolves
		// that race; an inaccessible live process remains an inspection error.
		departed, exitErr := serviceProcessStopped(ctx, expected.PID)
		if departed && exitErr == nil {
			return true, nil
		}
		return false, errors.Join(err, exitErr)
	}
	return current != expected, nil
}

func (f *engineProcessFixture) start(t *testing.T) {
	t.Helper()
	args := append([]string{"--scanner-path", f.scanner, "--engine-manager-path", engineMgrBin, "--proxy-path", f.proxyControl.image, "--proxy-engines", "llamacpp", "--cluster-dir", filepath.Join(f.base, "cluster")}, f.ports.args()...)
	env := append(f.proxyControl.env(), "NVPAIR_TEST_SUBSCRIPTION_SCANNER="+f.scanner, "NVPAIR_TEST_SUBSCRIPTION_PORT="+strconv.Itoa(f.scannerPort))
	f.p = startInteropProcess(t, f.base, brokerBin, f.ports.engineManagerHTTP, env, args...)
	f.p.ports = append(f.p.ports, f.scannerPort, f.sourcePort, f.targetPort, f.facadePort, f.targetFacadePort, f.proxyControl.port)
	for _, path := range []string{f.scanner, engineMgrBin, proxyBin, f.proxyControl.image} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		f.p.paths = append(f.p.paths, canonical)
	}
	waitForMethod(t, f.p.msgs, "app:ready", 20*time.Second)
	f.proxyControl.ready(t, "llamacpp-custom", f.facadePort)
	f.proxyControl.ready(t, "llamacpp", f.targetFacadePort)
}

func (f *engineProcessFixture) rpc(t *testing.T, method string, params any) jsonrpc.Message {
	t.Helper()
	if reply, handled := f.proxyControl.componentCall(t, method, params); handled {
		return reply
	}
	id := newRPCID()
	rawID := json.RawMessage(strconv.FormatInt(id, 10))
	sendLine(t, f.p.in, jsonrpc.Message{JSONRPC: "2.0", ID: &rawID, Method: method, Params: mustMarshal(t, params)})
	timer := time.NewTimer(35 * time.Second)
	defer timer.Stop()
	for {
		select {
		case msg, ok := <-f.p.msgs:
			if !ok {
				t.Fatalf("broker exited awaiting %s: %s", method, f.p.output.String())
			}
			if msg.ID != nil && string(*msg.ID) == string(rawID) && msg.Method == "" {
				return msg
			}
		case <-timer.C:
			t.Fatalf("timeout awaiting %s: %s", method, f.p.output.String())
		}
	}
}

func (f *engineProcessFixture) rpcOK(t *testing.T, method string, params any) json.RawMessage {
	t.Helper()
	response := f.rpc(t, method, params)
	if response.Error != nil {
		t.Fatalf("%s: %+v", method, response.Error)
	}
	return response.Result
}

func (f *engineProcessFixture) pid(t *testing.T, engine string) int32 {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.starts[engine]) != 1 {
		t.Fatalf("engine %s starts = %v, want exactly one", engine, f.starts[engine])
	}
	return f.starts[engine][0]
}

func assertFixtureProcessStopped(t *testing.T, pid int32) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	tick := time.NewTicker(25 * time.Millisecond)
	defer tick.Stop()
	for {
		stopped, err := serviceProcessStopped(ctx, pid)
		if err != nil {
			t.Fatalf("inspect engine PID %d: %v", pid, err)
		}
		if stopped {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("engine PID %d still exists", pid)
		case <-tick.C:
		}
	}
}

// Only the engine-control client reconnects; the ordinary broker and its fake
// engines stay alive. The existing permission writer supplies pinned identity.
func newEngineClientFixture(t *testing.T) (*engineProcessFixture, *task1Process, func() *http.Client) {
	t.Helper()
	base := shortClusterTempDir(t)
	cluster := filepath.Join(base, "cluster")
	origin := t.TempDir()
	const targetID, callerID = "e4-switch-target", "e4-switch-caller"
	if err := os.MkdirAll(cluster, 0o700); err != nil {
		t.Fatal(err)
	}
	targetCert := mintClusterIdentity(t, cluster, targetID)
	callerCert := mintClusterIdentity(t, origin, callerID)
	for _, identity := range []struct{ dir, id, peer, cert string }{
		{cluster, targetID, callerID, callerCert}, {origin, callerID, targetID, targetCert},
	} {
		writeActiveAdmission(t, identity.dir)
		writePin(t, identity.dir, identity.peer, identity.cert)
		writeMembers(t, identity.dir, targetID, callerID)
		jointWriteJSON(t, filepath.Join(identity.dir, "identity.json"), map[string]string{"node_uuid": identity.id})
	}
	managementGrantFixture(t, cluster, targetID, callerID)
	cm := startTask1Process(t, base, clusterMgrBin, "--config-dir", base, "--port", strconv.Itoa(freePort(t)))
	cm.call(t, "cluster:get-node-id", nil)
	f := newEngineProcessFixtureAt(t, base, reserveFixtureServicePorts(t))
	mesh := clustertrust.Open(origin)
	newClient := func() *http.Client {
		cfg, ok := mesh.ClientTLSConfig(targetID)
		if !ok {
			t.Fatal("engine caller has no pinned target TLS configuration")
		}
		client := &http.Client{Timeout: 35 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg}}
		t.Cleanup(client.CloseIdleConnections)
		return client
	}
	return f, cm, newClient
}

type pinnedEngineProcessFixture struct {
	mesh   *task2Fixture
	f      *engineProcessFixture
	origin *task1Process
	ec     *task1Observer
}

// Keep the existing fixture's production cluster-manager reverse gateway,
// pins, private IPC, and observing inner mTLS proxy. Replace only B's manager
// with a broker that supervises the same manager on those existing ports.
func newPinnedEngineProcessFixture(t *testing.T, reverse bool) *pinnedEngineProcessFixture {
	t.Helper()
	mesh := newTask2Fixture(t, task2WithoutScanner)
	mesh.engines[mesh.b.uuid].stop()
	ports := reserveFixtureServicePorts(t)
	ports.engineManagerHTTP = mesh.b.ports[noderec.ServiceEngineManager]
	ports.engineControl = mesh.b.ports[noderec.ServiceEngineControl]
	f := newEngineProcessFixtureAt(t, mesh.b.base, ports)
	origin := mesh.engines[mesh.a.uuid]
	t.Cleanup(func() {
		// The ordinary interop inventory owns product image paths. Stop the
		// other fixture instances of those images before its final scan.
		origin.stop()
		mesh.stopScanner()
		f.p.cleanup()
	})
	directory := mesh.directory(mesh.b)
	ec := mesh.b.observers[noderec.ServiceEngineControl]
	if !reverse {
		// Close the outer route so direct success cannot silently fall back.
		mesh.a.observers[noderec.ServiceCluster].interrupt(mesh.b.uuid, true)
		directory.Services[noderec.ServiceEngineControl] = noderec.ServiceStatus{Port: ec.port}
	}
	origin.send(t, noderec.NotifyNodes, noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{directory}}, false)
	return &pinnedEngineProcessFixture{mesh: mesh, f: f, origin: origin, ec: ec}
}
