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
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
)

// Use the running test executable, including its fixed windows-test-bin path,
// as the engine image. Dispatch before TestMain, without rebuilding products.
func init() {
	if image := os.Getenv("NVPAIR_TEST_SWITCH_ENGINE"); image != "" && image == os.Args[0] {
		if err := runSwitchEngine(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
}

func runSwitchEngine() error {
	ln, err := net.Listen("tcp", "127.0.0.1:"+os.Getenv("NVPAIR_TEST_SWITCH_PORT"))
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
		if r.URL.Path == "/api/generate" {
			if err := notify("/inflight"); err != nil {
				http.Error(w, err.Error(), http.StatusBadGateway)
				return
			}
			fmt.Fprintln(w, `{"model":"switch-model","done":true,"response":"finished"}`)
			return
		}
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprintln(w, `{"models":[{"name":"switch-model"}]}`)
		case "/v1/models":
			fmt.Fprintln(w, `{"object":"list","data":[{"id":"switch-model"}]}`)
		default:
			fmt.Fprintln(w, `{"version":"fixture"}`)
		}
	}))
}

type switchProcessFixture struct {
	base           string
	p              *interopProcess
	ports          fixtureServicePorts
	sourcePort     int
	targetPort     int
	facadePort     int
	control        *httptest.Server
	release        chan struct{}
	releaseOne     sync.Once
	inflight       chan struct{}
	mu             sync.Mutex
	starts         map[string][]int32
	changes        []settings.SwitchChanged
	holdTarget     bool
	targetStarting chan struct{}
	scanner        string
	scannerPort    int
}

func newSwitchProcessFixture(t *testing.T) *switchProcessFixture {
	t.Helper()
	return newSwitchProcessFixtureAt(t, t.TempDir(), reserveFixtureServicePorts(t))
}

func newSwitchProcessFixtureAt(t *testing.T, base string, ports fixtureServicePorts) *switchProcessFixture {
	t.Helper()
	f := &switchProcessFixture{base: base, ports: ports, sourcePort: freePort(t), targetPort: freePort(t), facadePort: freePort(t), release: make(chan struct{}), inflight: make(chan struct{}, 1), starts: make(map[string][]int32), targetStarting: make(chan struct{}, 1)}
	f.control = newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pid, err := strconv.ParseInt(r.URL.Query().Get("pid"), 10, 32)
		if err != nil {
			http.Error(w, "invalid fixture PID", http.StatusBadRequest)
			return
		}
		if r.URL.Path == "/started" {
			f.mu.Lock()
			name := r.URL.Query().Get("engine")
			f.starts[name] = append(f.starts[name], int32(pid))
			hold := name == "llamacpp" && f.holdTarget
			f.mu.Unlock()
			if hold {
				select {
				case f.targetStarting <- struct{}{}:
				default:
				}
				select {
				case <-f.release:
				case <-r.Context().Done():
				}
			}
		} else if r.URL.Path == "/inflight" {
			select {
			case f.inflight <- struct{}{}:
			default:
			}
			select {
			case <-f.release:
			case <-r.Context().Done():
			}
		}
	}))
	t.Cleanup(func() { f.unblock(); f.control.Close() })
	image, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// Reuse the empty-directory scanner instead of browsing host multicast.
	// The fixed executable role already has a Windows test-bin manifest entry.
	f.scanner = filepath.Join(testbin.Dir(t, "tests", "subscription-scanner", "subscription-scanner"), testbin.Exe("subscription-scanner"))
	jointCopyExecutable(t, image, f.scanner)
	f.scannerPort = freePort(t)
	jointIsolateEngines(t, f.base, freePort(t))
	for _, engine := range []struct {
		name string
		port int
	}{{"ollama", f.sourcePort}, {"llamacpp", f.targetPort}} {
		// Overrides deep-merge with bundled manifests. Clear llama.cpp's
		// router-role predicate: this fixture serves only a generic HTTP probe.
		probe := map[string]any{"http": "http://127.0.0.1:{port}/", "status": http.StatusOK, "timeout_s": 10, "interval_s": 60, "json_match": nil}
		rt := map[string]any{"mode": "process", "bin": image, "port": engine.port, "bind": "127.0.0.1", "ready": probe, "health": probe,
			"env":  map[string]string{"NVPAIR_TEST_SWITCH_ENGINE": image, "NVPAIR_TEST_SWITCH_NAME": engine.name, "NVPAIR_TEST_SWITCH_PORT": "{port}", "NVPAIR_TEST_SWITCH_CONTROL": f.control.URL},
			"stop": map[string]any{"signal": "term", "grace_s": 1}}
		jointWriteJSON(t, filepath.Join(jointAppDir(f.base), "engines", engine.name+".json"), map[string]any{
			"engine": engine.name, "runtime": rt, "platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{image}, "runtime": rt}},
		})
	}
	journal := make(map[string]any)
	for _, facade := range []struct {
		name   string
		server int
		port   int
	}{{"ollama", f.sourcePort, f.facadePort}, {"llamacpp", f.targetPort, freePort(t)}} {
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
	f.start(t)
	f.rpcOK(t, "engine:start", map[string]string{"engine": "ollama"})
	// Initialize every settings baseline before a switch revision is captured.
	// Otherwise the periodic refresh can add an unrelated engine's revision.
	for _, engine := range engines.Names() {
		f.rpcOK(t, "engine:get-settings", map[string]string{"engine": engine})
	}
	return f
}

func (f *switchProcessFixture) unblock() { f.releaseOne.Do(func() { close(f.release) }) }

func (f *switchProcessFixture) start(t *testing.T) {
	t.Helper()
	args := append([]string{"--scanner-path", f.scanner, "--engine-manager-path", engineMgrBin, "--proxy-path", proxyBin, "--proxy-engines", "ollama,llamacpp", "--cluster-dir", filepath.Join(f.base, "cluster")}, f.ports.args()...)
	f.p = startInteropProcess(t, f.base, brokerBin, f.ports.engineManagerHTTP, []string{"OLLAMA_HOST=", "NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + f.scanner, "NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(f.scannerPort)}, args...)
	f.p.ports = append(f.p.ports, f.scannerPort, f.sourcePort, f.targetPort, f.facadePort)
	for _, path := range []string{f.scanner, engineMgrBin, proxyBin} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		f.p.paths = append(f.p.paths, canonical)
	}
	waitForMethod(t, f.p.msgs, "app:ready", 20*time.Second)
	if port := waitProxyReady(t, f.p.in, f.p.msgs, 20*time.Second); port != f.facadePort {
		t.Fatalf("source facade port = %d, want fixture port %d", port, f.facadePort)
	}
	waitEngineProxyReady(t, "llamacpp-proxy", f.p.in, f.p.msgs, 20*time.Second)
}

// Keep switch notifications while correlating RPC replies. The general callRPC
// helper deliberately discards pushes, which would hide the sequence contract.
func (f *switchProcessFixture) rpc(t *testing.T, method string, params any) jsonrpc.Message {
	t.Helper()
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
			if msg.Method == settings.SwitchChangedNotification {
				var change settings.SwitchChanged
				if err := json.Unmarshal(msg.Params, &change); err != nil {
					t.Fatal(err)
				}
				f.changes = append(f.changes, change)
			}
			if msg.ID != nil && string(*msg.ID) == string(rawID) && msg.Method == "" {
				return msg
			}
		case <-timer.C:
			t.Fatalf("timeout awaiting %s: %s", method, f.p.output.String())
		}
	}
}

func (f *switchProcessFixture) rpcOK(t *testing.T, method string, params any) json.RawMessage {
	t.Helper()
	response := f.rpc(t, method, params)
	if response.Error != nil {
		t.Fatalf("%s: %+v", method, response.Error)
	}
	return response.Result
}

func (f *switchProcessFixture) status(t *testing.T, request settings.SwitchStatusRequest) settings.SwitchStatus {
	t.Helper()
	var status settings.SwitchStatus
	if err := json.Unmarshal(f.rpcOK(t, settings.SwitchStatusMethod, request), &status); err != nil {
		t.Fatal(err)
	}
	return status
}

func (f *switchProcessFixture) request(t *testing.T, id string) settings.SwitchRequest {
	t.Helper()
	return settings.SwitchRequest{RequestID: id, SourceEngine: "ollama", TargetEngine: "llamacpp", ExpectedRevision: f.status(t, settings.SwitchStatusRequest{}).ConfigurationRevision}
}

func (f *switchProcessFixture) terminal(t *testing.T, id string) settings.SwitchReceipt {
	t.Helper()
	var after uint64
	deadline := time.Now().Add(45 * time.Second)
	for time.Now().Before(deadline) {
		status := f.status(t, settings.SwitchStatusRequest{RequestID: id, AfterSequence: &after})
		if status.Receipt == nil {
			t.Fatal("status omitted requested receipt")
		}
		if status.Receipt.Terminal {
			return *status.Receipt
		}
		if status.Receipt.Sequence < after {
			t.Fatal("status receipt sequence regressed")
		}
		after = status.Receipt.Sequence
	}
	t.Fatal("switch did not become terminal")
	return settings.SwitchReceipt{}
}

func (f *switchProcessFixture) pid(t *testing.T, engine string) int32 {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.starts[engine]) != 1 {
		t.Fatalf("engine %s starts = %v, want exactly one", engine, f.starts[engine])
	}
	return f.starts[engine][0]
}

func assertSwitchProcessStopped(t *testing.T, pid int32) {
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

func TestEngineSwitchProcessCompletesAndRetries(t *testing.T) {
	f := newSwitchProcessFixture(t)
	sourcePID := f.pid(t, "ollama")
	request := f.request(t, "switch-complete")
	f.rpcOK(t, settings.SwitchMethod, request)
	receipt := f.terminal(t, request.RequestID)
	if receipt.Phase != settings.SwitchCompleted || receipt.Error != nil || receipt.Source.Intent != settings.SwitchIntentOff || receipt.Target.Intent != settings.SwitchIntentOn || !receipt.Source.TerminationVerified {
		t.Fatalf("completed receipt = %+v", receipt)
	}
	assertSwitchProcessStopped(t, sourcePID)
	assertInteropListenerClosed(t, f.sourcePort)
	f.pid(t, "llamacpp")
	var replay settings.SwitchResult
	if err := json.Unmarshal(f.rpcOK(t, settings.SwitchMethod, request), &replay); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(replay.Receipt, receipt) {
		t.Fatalf("identical retry changed terminal receipt: got %+v want %+v", replay.Receipt, receipt)
	}
	var previous uint64
	for _, change := range f.changes {
		if change.RequestID != request.RequestID {
			continue
		}
		if change.NodeID != receipt.NodeID || change.Sequence != previous+1 {
			t.Fatalf("notification = %+v after sequence %d", change, previous)
		}
		previous = change.Sequence
	}
	if previous != receipt.Sequence {
		t.Fatalf("last notification sequence %d, terminal sequence %d", previous, receipt.Sequence)
	}
	var stored struct {
		Engines map[string]bool `json:"engines"`
	}
	data, err := os.ReadFile(filepath.Join(jointAppDir(f.base), "engine-bin", "engine-state.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &stored); err != nil {
		t.Fatal(err)
	}
	if source, known := stored.Engines["ollama"]; !known || source || !stored.Engines["llamacpp"] {
		t.Fatalf("persisted intents = %v", stored.Engines)
	}
}

func TestEngineSwitchProcessRejectsStaleRevision(t *testing.T) {
	f := newSwitchProcessFixture(t)
	request := f.request(t, "switch-stale")
	request.ExpectedRevision = "e3v1:" + strings.Repeat("0", 64)
	response := f.rpc(t, settings.SwitchMethod, request)
	if response.Error == nil || response.Error.Code != settings.SwitchConfigurationConflictCode {
		t.Fatalf("stale revision error = %+v", response.Error)
	}
	f.pid(t, "ollama")
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.starts["llamacpp"]) != 0 {
		t.Fatal("stale request launched target")
	}
}

func TestEngineSwitchProcessRejectsRunningTargetBeforeDrain(t *testing.T) {
	f := newSwitchProcessFixture(t)
	f.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	request := f.request(t, "switch-running-target")
	f.rpcOK(t, settings.SwitchMethod, request)
	receipt := f.terminal(t, request.RequestID)
	if receipt.Phase != settings.SwitchFailedValidation || receipt.Error == nil || receipt.Error.Kind != settings.SwitchErrorTargetRunning {
		t.Fatalf("running target receipt = %+v", receipt)
	}
	for _, change := range f.changes {
		if change.RequestID == request.RequestID && change.Sequence > receipt.Sequence {
			t.Fatal("notification advanced after terminal validation failure")
		}
	}
	for _, engine := range []string{"ollama", "llamacpp"} {
		pid := f.pid(t, engine)
		stopped, err := serviceProcessStopped(context.Background(), pid)
		if err != nil || stopped {
			t.Fatalf("validation stopped %s PID %d: stopped=%t err=%v", engine, pid, stopped, err)
		}
	}
	// A fresh source request must still pass admission after validation fails;
	// process survival alone would not detect a mistakenly installed drain.
	done := f.holdRequest(t)
	f.unblock()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("source remained drained after validation refusal: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("source request did not finish after validation refusal")
	}
}

// Hold an actual request after it reaches the owned source image, rather than
// timing a sleep or assuming a workload notification means dispatch happened.
func (f *switchProcessFixture) holdRequest(t *testing.T) <-chan error {
	t.Helper()
	f.rpcOK(t, "ollama-proxy:node/add-manual", map[string]any{"id": "switch-source", "host": "127.0.0.1", "port": f.sourcePort, "addresses": []string{"127.0.0.1"}, "models": []string{"switch-model"}})
	f.rpcOK(t, "ollama-proxy:node/select", map[string]string{"id": "switch-source"})
	done := make(chan error, 1)
	go func() {
		client := &http.Client{Timeout: 80 * time.Second}
		response, err := client.Post(fmt.Sprintf("http://127.0.0.1:%d/api/generate", f.facadePort), "application/json", strings.NewReader(`{"model":"switch-model","stream":false,"prompt":"fixture"}`))
		if err == nil {
			var result struct {
				Done bool `json:"done"`
			}
			err = json.NewDecoder(response.Body).Decode(&result)
			if response.StatusCode != http.StatusOK || !result.Done {
				err = fmt.Errorf("inference status=%d done=%t", response.StatusCode, result.Done)
			}
			if closeErr := response.Body.Close(); err == nil {
				err = closeErr
			}
		}
		done <- err
	}()
	select {
	case <-f.inflight:
	case err := <-done:
		t.Fatalf("inference ended before dispatch gate: %v", err)
	case <-time.After(10 * time.Second):
		t.Fatal("inference did not reach source process")
	}
	return done
}

func (f *switchProcessFixture) waitDraining(t *testing.T, id string) {
	t.Helper()
	var after uint64
	deadline := time.Now().Add(40 * time.Second)
	for time.Now().Before(deadline) {
		status := f.status(t, settings.SwitchStatusRequest{RequestID: id, AfterSequence: &after})
		if status.Receipt == nil {
			t.Fatal("missing active receipt")
		}
		if status.Receipt.Phase == settings.SwitchDraining {
			return
		}
		if status.Receipt.Terminal {
			t.Fatalf("switch became terminal before drain: %+v", status.Receipt)
		}
		after = status.Receipt.Sequence
	}
	t.Fatal("switch did not reach draining")
}

func TestEngineSwitchProcessDrainsInflightBeforeStop(t *testing.T) {
	f := newSwitchProcessFixture(t)
	sourcePID := f.pid(t, "ollama")
	done := f.holdRequest(t)
	request := f.request(t, "switch-inflight")
	f.rpcOK(t, settings.SwitchMethod, request)
	f.waitDraining(t, request.RequestID)
	stopped, err := serviceProcessStopped(context.Background(), sourcePID)
	if err != nil || stopped {
		t.Fatalf("source stopped with inflight work: stopped=%t err=%v", stopped, err)
	}
	f.unblock()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("inflight request did not finish successfully: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("released request did not finish")
	}
	if receipt := f.terminal(t, request.RequestID); receipt.Phase != settings.SwitchCompleted {
		t.Fatalf("drained receipt = %+v", receipt)
	}
	assertSwitchProcessStopped(t, sourcePID)
}

func TestEngineSwitchProcessBrokerDeathInterruptsWithoutRestore(t *testing.T) {
	f := newSwitchProcessFixture(t)
	sourcePID := f.pid(t, "ollama")
	f.mu.Lock()
	f.holdTarget = true
	f.mu.Unlock()
	request := f.request(t, "switch-interrupted")
	f.rpcOK(t, settings.SwitchMethod, request)
	select {
	case <-f.targetStarting:
	case <-time.After(20 * time.Second):
		t.Fatal("target did not reach uncommitted startup gate")
	}
	status := f.status(t, settings.SwitchStatusRequest{RequestID: request.RequestID})
	if status.Receipt == nil || status.Receipt.Phase != settings.SwitchTargetStarting || status.Receipt.Source.Intent != settings.SwitchIntentOff {
		t.Fatalf("crash gate did not follow durable source Off: %+v", status.Receipt)
	}
	targetPID := f.pid(t, "llamacpp")
	assertSwitchProcessStopped(t, sourcePID)
	old := f.p
	old.stopOnce.Do(func() {
		if err := old.cmd.Process.Kill(); err != nil {
			t.Fatal(err)
		}
		if err := old.in.Close(); err != nil {
			t.Fatal(err)
		}
		select {
		case err := <-old.done:
			if err == nil {
				t.Fatal("killed broker exited successfully")
			}
		case <-time.After(15 * time.Second):
			t.Fatal("killed broker was not reaped")
		}
	})
	assertSwitchProcessStopped(t, int32(old.cmd.Process.Pid))
	assertSwitchProcessStopped(t, sourcePID)
	assertSwitchProcessStopped(t, targetPID)
	assertInteropListenerClosed(t, f.sourcePort)
	// EOF must tear down the old manager/proxy/scanner before a fresh broker
	// can reuse their listeners and configuration. Inspect actual processes.
	inventory := &serviceProcessInventory{paths: old.paths, known: make(map[int32]serviceProcessIdentity)}
	tick := time.NewTicker(25 * time.Millisecond)
	defer tick.Stop()
	deadline := time.NewTimer(15 * time.Second)
	defer deadline.Stop()
	for {
		owned, err := inventory.scan()
		if err != nil {
			t.Fatal(err)
		}
		if len(owned) == 0 {
			break
		}
		select {
		case <-deadline.C:
			t.Fatalf("old broker workers survived broker death: %+v", owned)
		case <-tick.C:
		}
	}
	f.unblock()
	f.start(t)
	receipt := f.terminal(t, request.RequestID)
	if receipt.Phase != settings.SwitchFailedInterrupted || !receipt.RecoveryRequired || receipt.Error == nil || receipt.Error.Kind != settings.SwitchErrorInterrupted {
		t.Fatalf("recovered receipt = %+v", receipt)
	}
	reattached := f.status(t, settings.SwitchStatusRequest{RequestID: request.RequestID, AfterSequence: &receipt.Sequence})
	if reattached.Receipt == nil || !reflect.DeepEqual(*reattached.Receipt, receipt) {
		t.Fatalf("reattach changed interrupted recovery receipt: %+v", reattached)
	}
	// Startup sends the internal restore notification after the proxy port
	// gates. Advertisement starts afterward; observe its backend updates from
	// this fresh broker rather than calling the notification as a public RPC.
	task1Eventually(t, "post-restore startup advertisement", 10*time.Second, func() bool {
		output := f.p.output.String()
		return strings.Contains(output, "local backend updated engine=ollama") &&
			strings.Contains(output, "local backend updated engine=llamacpp")
	})
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.starts["ollama"]) != 1 || len(f.starts["llamacpp"]) != 1 {
		t.Fatalf("recovery/restore started a reserved engine: %v", f.starts)
	}
	assertInteropListenerClosed(t, f.sourcePort)
	assertInteropListenerClosed(t, f.targetPort)
}

func TestEngineSwitchProcessPinnedDirect(t *testing.T) {
	testEngineSwitchPinned(t, false)
}

// Reuse the switch process fixture and the permission writer/pinned identity
// helpers. Only the engine-control client reconnects; the transaction owner and
// its fake engines stay alive. No discovery or vendor engine is needed.
func newSwitchClientFixture(t *testing.T) (*switchProcessFixture, *task1Process, func() *http.Client) {
	t.Helper()
	base := t.TempDir()
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
	f := newSwitchProcessFixtureAt(t, base, reserveFixtureServicePorts(t))
	mesh := clustertrust.Open(origin)
	newClient := func() *http.Client {
		cfg, ok := mesh.ClientTLSConfig(targetID)
		if !ok {
			t.Fatal("switch caller has no pinned target TLS configuration")
		}
		client := &http.Client{Timeout: 35 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg}}
		t.Cleanup(client.CloseIdleConnections)
		return client
	}
	return f, cm, newClient
}

type switchLostResponseTransport struct{ next http.RoundTripper }

func (transport switchLostResponseTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	response, err := transport.next.RoundTrip(request)
	if err != nil {
		return nil, err
	}
	_, readErr := io.Copy(io.Discard, response.Body)
	closeErr := response.Body.Close()
	if readErr != nil || closeErr != nil || response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("lost-response setup: HTTP %d read=%v close=%v", response.StatusCode, readErr, closeErr)
	}
	// Acceptance reached the real broker, but no response reaches the caller.
	return nil, io.ErrUnexpectedEOF
}

func TestE4SwitchClientLostResponseReconnectReadback(t *testing.T) {
	f, _, newClient := newSwitchClientFixture(t)
	client := newClient()
	request := f.request(t, "e4-lost-response")
	f.mu.Lock()
	f.holdTarget = true
	f.mu.Unlock()
	// Deliberately discard the entire apply response, including its receipt.
	// Recovery below knows only the original client-generated request ID.
	losingClient := &http.Client{Timeout: 35 * time.Second, Transport: switchLostResponseTransport{next: client.Transport}}
	apply, err := http.NewRequest(http.MethodPost, fmt.Sprintf("https://127.0.0.1:%d%s", f.ports.engineControl, settings.SwitchApplyPath), strings.NewReader(string(mustMarshal(t, request))))
	if err != nil {
		t.Fatal(err)
	}
	apply.Header.Set("Content-Type", "application/json")
	response, err := losingClient.Do(apply)
	if response != nil || !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("submit did not lose its response: response=%v err=%v", response, err)
	}
	select {
	case <-f.targetStarting:
	case <-time.After(20 * time.Second):
		t.Fatal("switch did not reach held target startup")
	}
	client.CloseIdleConnections()
	client = newClient()
	read := func(after *uint64) settings.SwitchStatus {
		t.Helper()
		var status settings.SwitchStatus
		task2Decode(t, managementHTTP(t, client, f.ports.engineControl, http.MethodPost, settings.SwitchStatusPath,
			settings.SwitchStatusRequest{RequestID: request.RequestID, AfterSequence: after}, http.StatusOK), &status)
		if status.Receipt == nil || status.Receipt.RequestID != request.RequestID || status.Receipt.NodeID != "e4-switch-target" || status.Receipt.Caller != "e4-switch-caller" || status.Receipt.Request.SourceEngine != request.SourceEngine || status.Receipt.Request.TargetEngine != request.TargetEngine || status.Receipt.ExpectedRevision != request.ExpectedRevision {
			t.Fatalf("original-ID readback lost correlation: %+v", status)
		}
		return status
	}
	active := read(nil)
	if active.ActiveRequestID != request.RequestID || active.Receipt.Terminal || active.Receipt.Phase != settings.SwitchTargetStarting {
		t.Fatalf("reattached active receipt = %+v", active)
	}
	f.unblock()
	var terminal settings.SwitchStatus
	after := active.Receipt.Sequence
	task1Eventually(t, "original request completes after client reconnect", 45*time.Second, func() bool {
		terminal = read(&after)
		if terminal.Receipt.Sequence < after {
			t.Fatal("reconnected readback sequence regressed")
		}
		after = terminal.Receipt.Sequence
		return terminal.Receipt.Terminal
	})
	if terminal.Receipt.Phase != settings.SwitchCompleted || terminal.Receipt.Error != nil || terminal.Receipt.RecoveryRequired || !terminal.Receipt.Source.TerminationVerified || terminal.Receipt.Source.Intent != settings.SwitchIntentOff || terminal.Receipt.Target.Intent != settings.SwitchIntentOn {
		t.Fatalf("reconnected completion = %+v", terminal.Receipt)
	}
	client.CloseIdleConnections()
	client = newClient()
	if again := read(&after); !reflect.DeepEqual(again.Receipt, terminal.Receipt) {
		t.Fatalf("terminal receipt changed after another reconnect: %+v", again)
	}
	assertSwitchProcessStopped(t, f.pid(t, "ollama"))
	assertInteropListenerClosed(t, f.sourcePort)
	f.pid(t, "llamacpp") // exactly one start, with no mutation replay
}

func TestE4SwitchClientDenialRevocationVetoConflict(t *testing.T) {
	f, cm, newClient := newSwitchClientFixture(t)
	client := newClient()
	const caller = "e4-switch-caller"
	managementSet(t, cm, caller, false)
	assertDenied := func(id string) {
		t.Helper()
		request := f.request(t, id)
		managementHTTP(t, client, f.ports.engineControl, http.MethodPost, settings.SwitchApplyPath, request, http.StatusConflict)
		var status settings.SwitchStatus
		task2Decode(t, managementHTTP(t, client, f.ports.engineControl, http.MethodPost, settings.SwitchStatusPath, settings.SwitchStatusRequest{}, http.StatusOK), &status)
		if status.ActiveRequestID != "" || status.ConfigurationRevision != request.ExpectedRevision {
			t.Fatalf("denied switch changed configuration or became active: %+v", status)
		}
	}
	assertDenied("e4-denied")
	managementSet(t, cm, caller, true)
	// Read permission while granted, then revoke it before submitting. A
	// previously allowed client snapshot cannot authorize the next mutation.
	var permission struct {
		Allowed bool   `json:"allowed"`
		State   string `json:"state"`
	}
	task2Decode(t, managementHTTP(t, client, f.ports.engineControl, http.MethodGet, "/v1/remote-management", nil, http.StatusOK), &permission)
	if !permission.Allowed || permission.State != "allowed" {
		t.Fatalf("pre-revocation permission = %+v", permission)
	}
	managementSet(t, cm, caller, false)
	assertDenied("e4-revoked")
	managementSet(t, cm, caller, true)
	f.rpcOK(t, "node:set-availability", map[string]string{"state": "paused"})
	assertRefused := func(request settings.SwitchRequest, code int, kind string) {
		t.Helper()
		actual, _, data := managementHTTPResponse(t, client, f.ports.engineControl, http.MethodPost, settings.SwitchApplyPath, request)
		var failure struct {
			Code string `json:"code"`
		}
		task2Decode(t, data, &failure)
		wireCode := kind
		if code == settings.SwitchRemoteVetoCode {
			wireCode = "local_veto"
		}
		if actual != http.StatusConflict || failure.Code != wireCode {
			t.Fatalf("switch refusal status=%d body=%s, want code %d", actual, data, code)
		}
		var status settings.SwitchStatus
		task2Decode(t, managementHTTP(t, client, f.ports.engineControl, http.MethodPost, settings.SwitchStatusPath,
			settings.SwitchStatusRequest{RequestID: request.RequestID}, http.StatusOK), &status)
		if status.Receipt == nil || status.Receipt.RequestID != request.RequestID || !status.Receipt.Terminal || status.Receipt.Phase != settings.SwitchRejected || status.Receipt.Error == nil || status.Receipt.Error.Code != code || status.Receipt.Error.Kind != kind {
			t.Fatalf("typed refusal readback = %+v", status)
		}
	}
	assertRefused(f.request(t, "e4-local-veto"), settings.SwitchRemoteVetoCode, settings.SwitchErrorRemoteVeto)
	f.rpcOK(t, "node:set-availability", map[string]string{"state": "available"})
	stale := f.request(t, "e4-conflict")
	stale.ExpectedRevision = "e3v1:" + strings.Repeat("0", 64)
	assertRefused(stale, settings.SwitchConfigurationConflictCode, settings.SwitchErrorConfigurationConflict)
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.starts["ollama"]) != 1 || len(f.starts["llamacpp"]) != 0 {
		t.Fatalf("refusals changed engine execution: %v", f.starts)
	}
}

func TestEngineSwitchProcessPinnedReverse(t *testing.T) {
	testEngineSwitchPinned(t, true)
}

// pinnedSwitchFixture is node B's broker-supervised switch stack on the
// task2 mesh, driven from node A's engine-manager over the pinned ec route.
type pinnedSwitchFixture struct {
	mesh   *task2Fixture
	f      *switchProcessFixture
	origin *task1Process
	ec     *task1Observer
}

// Keep the existing fixture's production cluster-manager reverse gateway,
// pins, private IPC, and observing inner mTLS proxy. Replace only B's manager
// with a broker that supervises the same manager on those existing ports.
func newPinnedSwitchFixture(t *testing.T, reverse bool) *pinnedSwitchFixture {
	t.Helper()
	mesh := newTask2Fixture(t, task2WithoutScanner)
	mesh.engines[mesh.b.uuid].stop()
	ports := reserveFixtureServicePorts(t)
	ports.engineManagerHTTP = mesh.b.ports[noderec.ServiceEngineManager]
	ports.engineControl = mesh.b.ports[noderec.ServiceEngineControl]
	f := newSwitchProcessFixtureAt(t, mesh.b.base, ports)
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
	return &pinnedSwitchFixture{mesh: mesh, f: f, origin: origin, ec: ec}
}

func (p *pinnedSwitchFixture) remoteStatus(t *testing.T, requestID string) settings.SwitchStatus {
	t.Helper()
	response := p.origin.call(t, settings.RemoteSwitchStatusMethod, settings.SwitchStatusRequest{NodeID: p.mesh.b.uuid, RequestID: requestID})
	if response.Error != nil {
		t.Fatalf("%s: %+v", settings.RemoteSwitchStatusMethod, response.Error)
	}
	var status settings.SwitchStatus
	task2Decode(t, response.Result, &status)
	return status
}

func (p *pinnedSwitchFixture) request(t *testing.T, id string) settings.SwitchRequest {
	t.Helper()
	return settings.SwitchRequest{NodeID: p.mesh.b.uuid, RequestID: id, SourceEngine: "ollama", TargetEngine: "llamacpp", ExpectedRevision: p.remoteStatus(t, "").ConfigurationRevision}
}

func testEngineSwitchPinned(t *testing.T, reverse bool) {
	t.Helper()
	p := newPinnedSwitchFixture(t, reverse)
	mesh, f, origin, ec := p.mesh, p.f, p.origin, p.ec
	var initial settings.SwitchStatus
	task2Decode(t, origin.call(t, settings.RemoteSwitchStatusMethod, settings.SwitchStatusRequest{NodeID: mesh.b.uuid}).Result, &initial)
	request := settings.SwitchRequest{NodeID: mesh.b.uuid, RequestID: "switch-pinned", SourceEngine: "ollama", TargetEngine: "llamacpp", ExpectedRevision: initial.ConfigurationRevision}
	f.mu.Lock()
	f.holdTarget = true
	f.mu.Unlock()
	var accepted settings.SwitchResult
	task2Decode(t, origin.call(t, settings.RemoteSwitchMethod, request).Result, &accepted)
	if accepted.Receipt.NodeID != mesh.b.uuid || accepted.Receipt.Caller != mesh.a.uuid || accepted.Receipt.RequestID != request.RequestID {
		t.Fatalf("remote caller/target identity = %+v", accepted.Receipt)
	}
	select {
	case <-f.targetStarting:
	case <-time.After(20 * time.Second):
		t.Fatal("remote switch did not reach target startup")
	}
	// Leave a nonterminal phase stable until the origin watcher emits progress;
	// foreground status polling must not race the watcher to every sequence.
	task1Eventually(t, "remote progress while target startup is held", 8*time.Second, func() bool {
		origin.mu.Lock()
		defer origin.mu.Unlock()
		for _, frame := range origin.frames {
			if frame.Method != settings.SwitchChangedNotification {
				continue
			}
			var change settings.SwitchChanged
			task2Decode(t, frame.Params, &change)
			if change.NodeID == mesh.b.uuid && change.RequestID == request.RequestID && change.Sequence > accepted.Receipt.Sequence {
				return true
			}
		}
		return false
	})
	f.unblock()
	var terminal settings.SwitchStatus
	task1Eventually(t, "remote terminal switch receipt", 45*time.Second, func() bool {
		task2Decode(t, origin.call(t, settings.RemoteSwitchStatusMethod, settings.SwitchStatusRequest{NodeID: mesh.b.uuid, RequestID: request.RequestID}).Result, &terminal)
		return terminal.Receipt != nil && terminal.Receipt.Terminal
	})
	if terminal.Receipt.Phase != settings.SwitchCompleted || terminal.Receipt.Source.Intent != settings.SwitchIntentOff || terminal.Receipt.Target.Intent != settings.SwitchIntentOn {
		t.Fatalf("remote terminal receipt = %+v", terminal.Receipt)
	}
	after := terminal.Receipt.Sequence - 1
	var status settings.SwitchStatus
	task2Decode(t, origin.call(t, settings.RemoteSwitchStatusMethod, settings.SwitchStatusRequest{NodeID: mesh.b.uuid, RequestID: request.RequestID, AfterSequence: &after}).Result, &status)
	if status.Receipt == nil || !reflect.DeepEqual(status.Receipt, terminal.Receipt) {
		t.Fatal("remote afterSequence status did not return the immutable terminal receipt")
	}
	var replay settings.SwitchResult
	task2Decode(t, origin.call(t, settings.RemoteSwitchMethod, request).Result, &replay)
	if !reflect.DeepEqual(replay.Receipt, *terminal.Receipt) {
		t.Fatal("remote identical retry changed the receipt")
	}
	if ec.count(mesh.a.uuid, http.MethodPost, settings.SwitchApplyPath) != 2 || ec.count(mesh.a.uuid, http.MethodPost, settings.SwitchStatusPath) < 2 {
		t.Fatal("pinned switch apply/status did not reach the observed inner target routes")
	}
	task1Eventually(t, "target-stamped remote switch progress", 10*time.Second, func() bool {
		origin.mu.Lock()
		defer origin.mu.Unlock()
		var sequence uint64
		for _, frame := range origin.frames {
			if frame.Method != settings.SwitchChangedNotification {
				continue
			}
			var change settings.SwitchChanged
			task2Decode(t, frame.Params, &change)
			if change.RequestID != request.RequestID {
				continue
			}
			if change.NodeID != mesh.b.uuid || change.Sequence <= sequence || change.Sequence > terminal.Receipt.Sequence {
				t.Fatalf("invalid remote progress %+v after sequence %d", change, sequence)
			}
			sequence = change.Sequence
		}
		return sequence > 0
	})
	assertSwitchProcessStopped(t, f.pid(t, "ollama"))
	f.pid(t, "llamacpp")
}
