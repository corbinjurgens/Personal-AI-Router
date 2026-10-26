// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/servicectl"
	"nvpair-shared/testbin"
	"nvpair-tui/rpc"
)

// This is the production RPC session against real private service processes.
// View rebootstrap and stale UI-result fencing have separate headless TUI tests;
// this test does not stand in for a terminal emulator or installed desktop run.
func TestServiceRestartReconnect(t *testing.T) {
	const model = "restart-proof-model"
	const engine = "restart-proof"
	const heldToken = "pending-before-service-exit"
	var probeHits, inferenceHits atomic.Int32
	held := make(chan struct{}, 1)
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/ready":
			_, _ = io.WriteString(w, `{}`)
		case r.Method == http.MethodGet && r.URL.Path == "/models":
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": model}}})
		case r.Method == http.MethodPost && r.URL.Path == "/held":
			var params struct {
				Token string `json:"token"`
			}
			if err := json.NewDecoder(r.Body).Decode(&params); err != nil || params.Token != heldToken {
				http.Error(w, "unexpected probe token", http.StatusBadRequest)
				return
			}
			probeHits.Add(1)
			select {
			case held <- struct{}{}:
			default:
			}
			select {
			case <-release:
				_, _ = io.WriteString(w, `{"accepted":true}`)
			case <-r.Context().Done():
			}
		case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
			var params struct {
				Model string `json:"model"`
			}
			if err := json.NewDecoder(r.Body).Decode(&params); err != nil || params.Model != model {
				http.Error(w, "unexpected inference model", http.StatusBadRequest)
				return
			}
			inferenceHits.Add(1)
			_, _ = io.WriteString(w, `{"id":"replacement-proxy-forwarded","choices":[]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(upstream.Close)
	t.Cleanup(func() { close(release) })

	f := newProxyServiceFixture(t)
	f.env = append(f.env, "USERPROFILE="+f.configDir, "OLLAMA_HOST=")
	managerPath := f.copyBinary(engineMgrBin)
	f.inventory.paths = append(f.inventory.paths, managerPath)
	writeRestartProofManifest(t, f, engine, portOfURL(t, upstream.URL))
	bundledPorts := isolateRestartProofBundledEngines(t, f)
	f.start()
	control := f.readyClient()
	oldStatus := serviceStatus(t, control)
	oldOwned := f.scan()
	oldService := restartProofIdentity(t, oldOwned, f.servicePath)
	oldBroker := restartProofIdentity(t, oldOwned, f.brokerPath)
	oldProxy := restartProofIdentity(t, oldOwned, f.proxyPath)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	conn, err := servicectl.DialEndpoint(ctx, f.endpoint)
	if err != nil {
		t.Fatalf("initial production-session dial: %v", err)
	}
	var dialAttempts, actionWrites atomic.Int32
	client := rpc.NewReconnectingClient(&restartProofConnection{ReadWriteCloser: conn, actionWrites: &actionWrites}, func(ctx context.Context) (io.ReadWriteCloser, error) {
		dialAttempts.Add(1)
		conn, err := servicectl.DialEndpoint(ctx, f.endpoint)
		if err != nil {
			return nil, err
		}
		return &restartProofConnection{ReadWriteCloser: conn, actionWrites: &actionWrites}, nil
	})
	retained := client
	runDone := make(chan error, 1)
	go func() { runDone <- client.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		client.Close()
		select {
		case err := <-runDone:
			if !errors.Is(err, context.Canceled) {
				t.Errorf("production session cancellation = %v, want context cancellation", err)
			}
		case <-time.After(serviceCallTimeout):
			t.Error("production session did not exit after cancellation")
		}
	})
	call := func(method string, params any) *rpc.Message {
		t.Helper()
		callCtx, done := context.WithTimeout(ctx, serviceCallTimeout)
		defer done()
		message, err := client.Call(callCtx, method, params)
		if err != nil {
			t.Fatalf("retained production session %s: %v", method, err)
		}
		return message
	}
	await := func(kind rpc.EventKind, method string, generation uint64, deadline time.Time) rpc.Event {
		t.Helper()
		timer := time.NewTimer(time.Until(deadline))
		defer timer.Stop()
		for {
			select {
			case event, ok := <-client.Events():
				if !ok {
					t.Fatalf("production session ended before event kind=%d method=%s generation=%d", kind, method, generation)
				}
				if event.Kind == kind && event.Generation == generation && (method == "" || event.Message != nil && event.Message.Method == method) {
					return event
				}
			case <-timer.C:
				t.Fatalf("missing retained-session event kind=%d method=%s generation=%d", kind, method, generation)
			}
		}
	}
	readPolicy := func() nodepolicy.Policy {
		t.Helper()
		var snapshot servicePolicySnapshot
		if err := json.Unmarshal(call("policy:get", nil).Result, &snapshot); err != nil {
			t.Fatalf("decode retained-session policy: %v", err)
		}
		if err := snapshot.Policy.Validate(); err != nil {
			t.Fatalf("retained-session policy invalid: %v", err)
		}
		return snapshot.Policy
	}
	subscribe := func(generation uint64) {
		t.Helper()
		var result struct {
			Subscribed bool `json:"subscribed"`
		}
		if err := json.Unmarshal(call("discovery:subscribe", nil).Result, &result); err != nil || !result.Subscribed {
			t.Fatalf("discovery subscription not acknowledged: %+v error=%v", result, err)
		}
		await(rpc.EventNotification, "discovery:nodes-changed", generation, time.Now().Add(serviceCallTimeout))
	}
	await(rpc.EventNotification, "app:ready", 1, time.Now().Add(serviceCallTimeout))
	subscribe(1)
	initial := readPolicy()
	initial.Admission.MaxConcurrentPerModel = 3
	initial = setServicePolicy(t, control, initial)
	path := f.policyPath()
	assertStoredPolicy(t, path, initial)
	if got := readPolicy(); !reflect.DeepEqual(got, initial) {
		t.Fatal("initial retained-session policy differs from persisted policy")
	}

	// Identify only our loopback HTTP fixture, without engine:start or a child
	// runtime. The ordinary public model read is the adoption positive control.
	var status struct {
		Running bool `json:"running"`
	}
	if err := json.Unmarshal(call("engine:status", map[string]string{"engine": engine}).Result, &status); err != nil || !status.Running {
		t.Fatalf("private HTTP fixture was not adopted: %+v error=%v", status, err)
	}
	for _, bundled := range engines.All() {
		var status struct {
			Installed bool `json:"installed"`
			Running   bool `json:"running"`
			Port      int  `json:"port"`
		}
		if err := json.Unmarshal(call("engine:status", map[string]string{"engine": bundled.Name}).Result, &status); err != nil || status.Installed || status.Running || status.Port != bundledPorts[bundled.Name] {
			t.Fatalf("bundled engine %s escaped private isolation: %+v error=%v", bundled.Name, status, err)
		}
	}
	var models struct {
		ByEngine     map[string][]string `json:"modelsByEngine"`
		LoadedEngine map[string][]string `json:"loadedByEngine"`
	}
	if err := json.Unmarshal(call("engine:models", nil).Result, &models); err != nil || !reflect.DeepEqual(models.ByEngine[engine], []string{model}) {
		t.Fatalf("public model-read positive control = %+v error=%v", models, err)
	}
	for _, bundled := range engines.All() {
		if len(models.ByEngine[bundled.Name]) != 0 || len(models.LoadedEngine[bundled.Name]) != 0 {
			t.Fatalf("isolated bundled engine %s contributed models: %+v", bundled.Name, models)
		}
	}
	// A private manifest HTTP action carries a token, so background inventory
	// polling cannot be mistaken for this pending, externally observed POST.
	pendingCtx, pendingCancel := context.WithTimeout(ctx, 20*time.Second)
	defer pendingCancel()
	pending := make(chan error, 1)
	pendingDone := make(chan struct{})
	t.Cleanup(func() {
		pendingCancel()
		select {
		case <-pendingDone:
		case <-time.After(serviceCallTimeout):
			t.Error("pending public action goroutine did not exit after cancellation")
		}
	})
	go func() {
		defer close(pendingDone)
		_, err := client.Call(pendingCtx, "engine:action", map[string]any{
			"engine": engine, "action": "restart_probe", "params": map[string]string{"token": heldToken},
		})
		pending <- err
	}()
	select {
	case <-held:
	case err := <-pending:
		t.Fatalf("pending action returned before HTTP barrier: %v", err)
	case <-time.After(4 * time.Second):
		t.Fatal("public pending action never reached the tokenized HTTP barrier")
	}
	select {
	case err := <-pending:
		t.Fatalf("held public action completed before service exit: %v", err)
	default:
	}
	if err := f.inventory.kill(oldService); err != nil {
		t.Fatalf("kill exact retained service identity: %v", err)
	}
	f.waitChild(f.service, serviceCallTimeout)
	await(rpc.EventDisconnected, "", 1, time.Now().Add(serviceCallTimeout))
	select {
	case err := <-pending:
		if err == nil || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			t.Fatalf("interrupted pending action = %v, want immediate transport failure", err)
		}
	case <-time.After(time.Second):
		t.Fatal("service exit did not promptly fail the retained pending action")
	}
	outageCtx, outageCancel := context.WithTimeout(ctx, 3*time.Second)
	defer outageCancel()
	if _, err := client.Call(outageCtx, "policy:get", nil); err == nil {
		t.Fatal("retained session accepted a new call during service outage")
	} else if errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("outage call blocked instead of failing fast: %v", err)
	} else if errors.Is(err, context.Canceled) {
		t.Fatalf("outage call was canceled instead of failing with a transport error: %v", err)
	}
	// Reap only recorded old PID/path/creation identities, keeping the production
	// client alive. cleanupChecked would close clients and invalidate the proof.
	for _, identity := range oldOwned {
		if identity != oldService {
			if err := f.inventory.kill(identity); err != nil {
				t.Fatalf("reap recorded old descendant %+v: %v", identity, err)
			}
		}
	}
	deadline := time.Now().Add(serviceCallTimeout)
	for {
		owned, err := f.inventory.scan()
		if err == nil && len(owned) == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("old owned policy writers did not stop before private persisted mutation: owned=%+v error=%v", owned, err)
		}
		// An inspection failure is ambiguous evidence, not a verdict: an exiting
		// process stays enumerable while it stops and cannot be re-inspected.
		// Only a scan proving no owned process survives may continue the test.
		time.Sleep(25 * time.Millisecond)
	}
	changed := initial
	changed.Revision = strconv.FormatInt(time.Now().UnixNano(), 10)
	changed.Admission.MaxConcurrentPerModel = 7
	if err := changed.Validate(); err != nil {
		t.Fatalf("replacement private policy invalid: %v", err)
	}
	data, err := json.Marshal(changed)
	if err != nil {
		t.Fatalf("encode replacement private policy: %v", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("change stopped fixture's private policy: %v", err)
	}
	assertStoredPolicy(t, path, changed)
	f.start()
	newControl := f.readyClient()
	newStatus := serviceStatus(t, newControl)
	newOwned := f.scan()
	for _, pair := range [][2]serviceProcessIdentity{
		{oldService, restartProofIdentity(t, newOwned, f.servicePath)},
		{oldBroker, restartProofIdentity(t, newOwned, f.brokerPath)},
		{oldProxy, restartProofIdentity(t, newOwned, f.proxyPath)},
	} {
		if pair[0] == pair[1] {
			t.Fatalf("replacement reused old process identity: %+v", pair[0])
		}
	}
	if newStatus.PID == oldStatus.PID || newStatus.BrokerPID == oldStatus.BrokerPID {
		t.Fatalf("replacement service/broker PIDs did not change: old=%+v new=%+v", oldStatus, newStatus)
	}
	await(rpc.EventNotification, "app:ready", 2, time.Now().Add(serviceCallTimeout))
	if client != retained || client.Generation() != 2 || !client.IsCurrent(2) || dialAttempts.Load() == 0 {
		t.Fatal("replacement did not recover through the retained production client")
	}
	subscribe(2)
	if got := readPolicy(); !reflect.DeepEqual(got, changed) {
		t.Fatal("replacement retained-session baseline did not replace the old persisted policy")
	}
	// A new push must reach the retained replacement stream with fresh data.
	pushed := changed
	pushed.Admission.MaxConcurrentPerModel = 9
	pushed = setServicePolicy(t, newControl, pushed)
	pushDeadline := time.Now().Add(serviceCallTimeout)
	for {
		event := await(rpc.EventNotification, "policy:changed", 2, pushDeadline)
		var notification struct {
			Policy nodepolicy.Policy `json:"policy"`
		}
		if err := json.Unmarshal(event.Message.Params, &notification); err != nil {
			t.Fatalf("decode replacement policy push: %v", err)
		}
		if notification.Policy.Revision == pushed.Revision {
			if !reflect.DeepEqual(notification.Policy, pushed) {
				t.Fatal("replacement policy push differs from acknowledged update")
			}
			break
		}
	}
	call("llamacpp-proxy:node/add-manual", map[string]any{
		"id": "restart-proof-upstream", "host": "127.0.0.1", "port": portOfURL(t, upstream.URL),
		"addresses": []string{"127.0.0.1"}, "models": []string{model},
	})
	httpClient := &http.Client{Timeout: 5 * time.Second}
	defer httpClient.CloseIdleConnections()
	response, err := httpClient.Post(fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", f.facadePort), "application/json",
		strings.NewReader(fmt.Sprintf(`{"model":%q,"messages":[]}`, model)))
	if err != nil {
		t.Fatalf("replacement proxy routed HTTP request: %v", err)
	}
	var inference struct {
		ID string `json:"id"`
	}
	decodeErr := json.NewDecoder(response.Body).Decode(&inference)
	closeErr := response.Body.Close()
	if decodeErr != nil || closeErr != nil || response.StatusCode != http.StatusOK || inference.ID != "replacement-proxy-forwarded" || inferenceHits.Load() != 1 {
		t.Fatalf("replacement HTTP inference status=%d id=%q upstream_hits=%d decode=%v close=%v", response.StatusCode, inference.ID, inferenceHits.Load(), decodeErr, closeErr)
	}
	if posts, writes := probeHits.Load(), actionWrites.Load(); posts != 1 || writes != 1 {
		t.Fatalf("interrupted mutation was replayed: tokenized upstream POSTs=%d RPC writes=%d, want exactly 1 each", posts, writes)
	}
	t.Logf("retained_session_recovered generation=2 dial_attempts=%d pending_failed=true mutation_posts=1 replacement_http_hits=1", dialAttempts.Load())
}

// Observe production codec writes without changing framing or replies. Counting
// the public action on every transport catches replay even if a replacement
// worker rejects it before it can reach the HTTP fixture.
type restartProofConnection struct {
	io.ReadWriteCloser
	actionWrites *atomic.Int32
}

func (c *restartProofConnection) Write(data []byte) (int, error) {
	var message rpc.Message
	if err := json.Unmarshal(data, &message); err != nil {
		return 0, fmt.Errorf("inspect production RPC write: %w", err)
	}
	if message.Method == "engine:action" {
		c.actionWrites.Add(1)
	}
	return c.ReadWriteCloser.Write(data)
}

func restartProofIdentity(t *testing.T, owned []serviceProcessIdentity, path string) serviceProcessIdentity {
	t.Helper()
	var found []serviceProcessIdentity
	for _, identity := range owned {
		if identity.Path == path {
			found = append(found, identity)
		}
	}
	if len(found) != 1 {
		t.Fatalf("owned identity for %q: %+v, want exactly one", path, found)
	}
	return found[0]
}

func writeRestartProofManifest(t *testing.T, f *serviceFixture, engine string, port int) {
	t.Helper()
	manifest := map[string]any{
		"engine": engine, "display_name": "Private restart proof", "manifest_version": 1,
		"runtime": map[string]any{
			"mode": "process", "bin": filepath.Join(f.dir, "never-started"), "bind": "127.0.0.1", "port": port,
			"ready":  map[string]any{"http": "http://127.0.0.1:{port}/ready", "status": http.StatusOK, "timeout_s": 1},
			"health": map[string]any{"http": "http://127.0.0.1:{port}/ready", "status": http.StatusOK, "interval_s": 5},
		},
		"platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{}}},
		"actions": map[string]any{
			"list_models": map[string]any{
				"http":   map[string]string{"method": http.MethodGet, "path": "/models"},
				"result": map[string]string{"array": "data", "field": "id"},
			},
			"restart_probe": map[string]any{"http": map[string]string{"method": http.MethodPost, "path": "/held"}},
		},
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("encode private process-mode manifest: %v", err)
	}
	enginesDir := filepath.Join(filepath.Dir(f.policyPath()), "engines")
	if err := os.MkdirAll(enginesDir, 0o700); err != nil {
		t.Fatalf("create private manifest directory: %v", err)
	}
	if err := os.WriteFile(filepath.Join(enginesDir, engine+".json"), data, 0o600); err != nil {
		t.Fatalf("write private manifest: %v", err)
	}
}

// Embedded manifests still load alongside the private custom engine. Redirect
// their probes before either service starts and remove all executable detection
// and command paths so shutdown cannot reach an installed engine either.
func isolateRestartProofBundledEngines(t *testing.T, f *serviceFixture) map[string]int {
	t.Helper()
	ports := make(map[string]int)
	enginesDir := filepath.Join(filepath.Dir(f.policyPath()), "engines")
	for _, engine := range engines.All() {
		unavailable := httptest.NewServer(http.NotFoundHandler())
		// Registered before startChild's cleanup: every owned descendant stops
		// before these listeners close, including on an early test failure.
		t.Cleanup(unavailable.Close)
		port := portOfURL(t, unavailable.URL)
		ports[engine.Name] = port
		privatePath := filepath.Join(f.dir, engine.Name+"-never-started")
		runtimeOverride := map[string]any{
			"port": port, "bin": privatePath, "cli": privatePath,
			"ready":  map[string]any{"http": unavailable.URL + "/unavailable", "status": http.StatusOK, "timeout_s": 1},
			"health": map[string]any{"http": unavailable.URL + "/unavailable", "status": http.StatusOK, "interval_s": 5},
		}
		override := map[string]any{
			"engine": engine.Name, "detect": []string{}, "runtime": runtimeOverride,
			"platforms": map[string]any{
				runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{}, "runtime": runtimeOverride},
			},
		}
		data, err := json.Marshal(override)
		if err != nil {
			t.Fatalf("encode isolated bundled manifest %s: %v", engine.Name, err)
		}
		if err := os.WriteFile(filepath.Join(enginesDir, engine.Name+".json"), data, 0o600); err != nil {
			t.Fatalf("write isolated bundled manifest %s: %v", engine.Name, err)
		}
	}
	return ports
}

// The opt-in deliberately lives outside the ordinary desktop unit project.
// Go alone owns the private service tree; Vitest is an attach-only real client.
func TestServiceRestartReconnectDesktop(t *testing.T) {
	optIn := os.Getenv("PAIR_TEST_DESKTOP_RECONNECT")
	if optIn == "" {
		t.Skip("UNTESTED: desktop real-process restart requires PAIR_TEST_DESKTOP_RECONNECT=1 and explicit local Node/dependencies")
	}
	if optIn != "1" {
		t.Fatal("PAIR_TEST_DESKTOP_RECONNECT must be exactly 1 when supplied")
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatalf("resolve desktop fixture test image: %v", err)
	}
	if runtime.GOOS == "windows" {
		root := os.Getenv(testbin.EnvRoot)
		if !filepath.IsAbs(root) {
			t.Fatal("opted-in Windows desktop proof requires an absolute NVPAIR_TEST_BIN_DIR; no temporary image fallback")
		}
		actual, err := canonicalServicePath(self)
		if err != nil {
			t.Fatalf("resolve current fixed test image: %v", err)
		}
		expected, err := canonicalServicePath(filepath.Join(root, "go-test", "tests", "tests.test.exe"))
		if err != nil || actual != expected {
			t.Fatalf("desktop test image %q is not the approved fixed package image %q: %v", actual, expected, err)
		}
	}
	node := os.Getenv("PAIR_TEST_NODE")
	if !filepath.IsAbs(node) {
		t.Fatal("opted-in desktop proof requires PAIR_TEST_NODE naming the absolute installed Node 25.9.0 executable; no PATH fallback")
	}
	workingDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("resolve tests working directory: %v", err)
	}
	desktop := filepath.Clean(filepath.Join(workingDir, "..", "..", "desktop"))
	vitest := filepath.Join(desktop, "node_modules", "vitest", "vitest.mjs")
	for _, path := range []string{node, vitest, filepath.Join(desktop, "package-lock.json"),
		filepath.Join(desktop, "tests", "service-restart-reconnect", "bridge.test.ts"),
		filepath.Join(desktop, "tests", "service-restart-reconnect", "vitest.config.ts")} {
		info, err := os.Stat(path)
		if err != nil || !info.Mode().IsRegular() {
			t.Fatalf("opted-in desktop prerequisite %q is not a regular local file: %v", path, err)
		}
	}

	const model = "desktop-restart-model"
	const initialModel = "desktop-before-restart-model"
	var actionHits, inferenceHits atomic.Int32
	var replacementInventory atomic.Bool
	held := make(chan struct{}, 1)
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/ready":
			if _, err := io.WriteString(w, `{}`); err != nil {
				t.Errorf("private readiness response: %v", err)
			}
		case r.Method == http.MethodGet && r.URL.Path == "/models":
			name := initialModel
			if replacementInventory.Load() {
				name = model
			}
			if err := json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": name}}}); err != nil {
				t.Errorf("private model response: %v", err)
			}
		case r.Method == http.MethodPost && r.URL.Path == "/held":
			var body struct {
				Token string `json:"token"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Token != "desktop-pending-before-exit" {
				http.Error(w, "invalid private action", http.StatusBadRequest)
				return
			}
			actionHits.Add(1)
			select {
			case held <- struct{}{}:
			default:
			}
			select {
			case <-release:
			case <-r.Context().Done():
			}
		case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
			var body struct {
				Model string `json:"model"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Model != model {
				http.Error(w, "invalid private inference", http.StatusBadRequest)
				return
			}
			inferenceHits.Add(1)
			if _, err := io.WriteString(w, `{"id":"desktop-replacement-forwarded","choices":[]}`); err != nil {
				t.Errorf("private inference response: %v", err)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(upstream.Close)
	t.Cleanup(func() { close(release) })
	f := newProxyServiceFixture(t)
	// Before first startup, replace multicast discovery with the existing empty
	// directory scanner at its already ruled executable path. No new test image.
	jointCopyExecutable(t, self, f.scannerPath)
	privateEnv := f.env[:0]
	for _, entry := range f.env {
		key, _, _ := strings.Cut(entry, "=")
		if !strings.HasPrefix(strings.ToUpper(key), "NVPAIR_TEST_") {
			privateEnv = append(privateEnv, entry)
		}
	}
	f.env = privateEnv
	f.env = append(f.env, "NVPAIR_TEST_SUBSCRIPTION_SCANNER="+f.scannerPath,
		"NVPAIR_TEST_SUBSCRIPTION_PORT="+strconv.Itoa(temporaryClusterPort(t)),
		"USERPROFILE="+f.configDir, "OLLAMA_HOST=")
	managerPath := f.copyBinary(engineMgrBin)
	clusterPath := f.copyBinary(clusterMgrBin)
	f.inventory.paths = append(f.inventory.paths, managerPath, clusterPath)
	writeRestartProofManifest(t, f, "desktop-restart-proof", portOfURL(t, upstream.URL))
	bundledPorts := isolateRestartProofBundledEngines(t, f)
	f.start()
	control := f.readyClient()
	initial := servicePolicy(t, control)
	initial.Admission.MaxConcurrentPerModel = 3
	initial = setServicePolicy(t, control, initial)
	policyPath := f.policyPath()
	assertStoredPolicy(t, policyPath, initial)
	oldOwned := f.scan()
	oldService := restartProofIdentity(t, oldOwned, f.servicePath)
	oldBroker := restartProofIdentity(t, oldOwned, f.brokerPath)
	oldProxy := restartProofIdentity(t, oldOwned, f.proxyPath)
	assertIsolated := func(c *serviceRPCClient, wantModel string) {
		t.Helper()
		failReadiness := func(reason string) {
			t.Helper()
			// The service's output does not include its private broker log. Keep
			// failure diagnostics bounded and identify any omitted older bytes.
			file, err := os.Open(filepath.Join(f.dir, "logs", "broker.log"))
			if err != nil {
				t.Fatalf("%s; open private broker diagnostic: %v", reason, err)
			}
			info, statErr := file.Stat()
			if statErr != nil || !info.Mode().IsRegular() {
				closeErr := file.Close()
				t.Fatalf("%s; inspect private broker diagnostic: stat=%v close=%v", reason, statErr, closeErr)
			}
			length := min(info.Size(), int64(16*1024))
			tail, readErr := io.ReadAll(io.NewSectionReader(file, info.Size()-length, length))
			closeErr := file.Close()
			if readErr != nil || closeErr != nil {
				t.Fatalf("%s; read private broker diagnostic: read=%v close=%v", reason, readErr, closeErr)
			}
			t.Fatalf("%s; private_broker_tail_bytes=%d file_bytes=%d:\n%s", reason, len(tail), info.Size(), tail)
		}
		// app:ready covers the required scanner, not the optional manager's
		// publication/recovery gate. Probe only our private custom engine until
		// the broker actually exposes that worker, on both service incarnations.
		deadline := time.Now().Add(serviceCallTimeout)
		timeout := time.NewTimer(time.Until(deadline))
		defer timeout.Stop()
		tick := time.NewTicker(25 * time.Millisecond)
		defer tick.Stop()
		for {
			remaining := time.Until(deadline)
			if remaining <= 0 {
				failReadiness("desktop private engine-manager did not become available within its readiness deadline")
			}
			message, err := c.call("300", "engine:status", mustMarshal(t, map[string]string{"engine": "desktop-restart-proof"}), remaining)
			if err != nil {
				failReadiness(fmt.Sprintf("desktop private engine-manager readiness RPC: %v", err))
			}
			if time.Now().After(deadline) {
				failReadiness("desktop private engine-manager readiness response exceeded its deadline")
			}
			if message.Error == nil {
				var status struct {
					Engine  string `json:"engine"`
					Running bool   `json:"running"`
					Port    int    `json:"port"`
				}
				if err := json.Unmarshal(message.Result, &status); err != nil || status.Engine != "desktop-restart-proof" || !status.Running || status.Port != portOfURL(t, upstream.URL) {
					failReadiness(fmt.Sprintf("desktop private engine-manager readiness positive control: %+v error=%v", status, err))
				}
				break
			}
			if message.Error.Code != -32000 || message.Error.Message != "engine-manager not available" {
				failReadiness(fmt.Sprintf("desktop private engine-manager unexpected readiness RPC error: %+v", message.Error))
			}
			select {
			case <-timeout.C:
				failReadiness(fmt.Sprintf("desktop private engine-manager remained unavailable: %+v", message.Error))
			case <-tick.C:
			}
		}
		for _, engine := range engines.All() {
			var status struct {
				Installed bool `json:"installed"`
				Running   bool `json:"running"`
				Port      int  `json:"port"`
			}
			message := c.mustCall(t, "301", "engine:status", mustMarshal(t, map[string]string{"engine": engine.Name}))
			if err := json.Unmarshal(message.Result, &status); err != nil || status.Installed || status.Running || status.Port != bundledPorts[engine.Name] {
				t.Fatalf("desktop bundled engine %s escaped private isolation: %+v error=%v", engine.Name, status, err)
			}
		}
		var models struct {
			ByEngine map[string][]string `json:"modelsByEngine"`
			Loaded   map[string][]string `json:"loadedByEngine"`
		}
		message := c.mustCall(t, "302", "engine:models", nil)
		if err := json.Unmarshal(message.Result, &models); err != nil || !reflect.DeepEqual(models.ByEngine["desktop-restart-proof"], []string{wantModel}) {
			t.Fatalf("desktop private model adoption: %+v error=%v", models, err)
		}
		for _, engine := range engines.All() {
			if len(models.ByEngine[engine.Name]) != 0 || len(models.Loaded[engine.Name]) != 0 {
				t.Fatalf("desktop bundled engine %s contributed models", engine.Name)
			}
		}
	}
	assertIsolated(control, initialModel)
	bridge := startDesktopRestartBridge(t, f, node, desktop, vitest, portOfURL(t, upstream.URL))
	bridge.await("attached", 45*time.Second)
	bridge.await("kill", serviceCallTimeout)
	select {
	case <-held:
	case <-time.After(serviceCallTimeout):
		t.Fatal("desktop pending action did not reach the tokenized HTTP barrier")
	}
	if err := f.inventory.kill(oldService); err != nil {
		t.Fatalf("kill exact desktop fixture service: %v", err)
	}
	f.waitChild(f.service, serviceCallTimeout)
	bridge.await("disconnected", 20*time.Second)
	for _, identity := range oldOwned {
		if identity != oldService {
			if err := f.inventory.kill(identity); err != nil {
				t.Fatalf("reap recorded desktop fixture descendant %+v: %v", identity, err)
			}
		}
	}
	deadline := time.Now().Add(serviceCallTimeout)
	for {
		owned, err := f.inventory.scan()
		if err == nil && len(owned) == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("desktop old policy writers not proven stopped: owned=%+v error=%v", owned, err)
		}
		time.Sleep(25 * time.Millisecond)
	}
	changed := initial
	changed.Revision = strconv.FormatInt(time.Now().UnixNano(), 10)
	changed.Admission.MaxConcurrentPerModel = 7
	if err := changed.Validate(); err != nil {
		t.Fatalf("desktop replacement policy invalid: %v", err)
	}
	if err := os.WriteFile(policyPath, mustMarshal(t, changed), 0o600); err != nil {
		t.Fatalf("write stopped desktop fixture policy: %v", err)
	}
	assertStoredPolicy(t, policyPath, changed)
	// Keep availability unchanged: this adopted HTTP fixture is not an engine
	// process the manager owns and can stop for physical pause recovery. A new
	// manager must instead hydrate a changed canonical model inventory.
	replacementInventory.Store(true)
	bridge.await("retrying", 35*time.Second)
	f.start()
	replacement := f.readyClient()
	newOwned := f.scan()
	for _, pair := range [][2]serviceProcessIdentity{
		{oldService, restartProofIdentity(t, newOwned, f.servicePath)},
		{oldBroker, restartProofIdentity(t, newOwned, f.brokerPath)},
		{oldProxy, restartProofIdentity(t, newOwned, f.proxyPath)},
	} {
		if pair[0] == pair[1] {
			t.Fatalf("desktop replacement reused process identity: %+v", pair[0])
		}
	}
	assertIsolated(replacement, model)
	bridge.send("restart")
	bridge.await("hydrated", 40*time.Second)
	pushed := changed
	pushed.Admission.MaxConcurrentPerModel = 9
	pushed = setServicePolicy(t, replacement, pushed)
	assertStoredPolicy(t, policyPath, pushed)
	bridge.send("push")
	bridge.await("pushed", 20*time.Second)
	client := &http.Client{Timeout: 5 * time.Second}
	defer client.CloseIdleConnections()
	response, err := client.Post(fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", f.facadePort), "application/json",
		strings.NewReader(fmt.Sprintf(`{"model":%q,"messages":[]}`, model)))
	if err != nil {
		t.Fatalf("desktop replacement proxy HTTP request: %v", err)
	}
	var inference struct {
		ID string `json:"id"`
	}
	decodeErr := json.NewDecoder(response.Body).Decode(&inference)
	closeErr := response.Body.Close()
	if decodeErr != nil || closeErr != nil || response.StatusCode != http.StatusOK || inference.ID != "desktop-replacement-forwarded" || inferenceHits.Load() != 1 {
		t.Fatalf("desktop replacement inference status=%d id=%q hits=%d decode=%v close=%v", response.StatusCode, inference.ID, inferenceHits.Load(), decodeErr, closeErr)
	}
	bridge.send("detach")
	bridge.await("detached", 15*time.Second)
	// Keep Node alive until the service confirms the detached socket is gone;
	// process exit must not substitute for the production detach's close/join.
	detachDeadline := time.Now().Add(serviceCallTimeout)
	for {
		clients := serviceStatus(t, replacement).Clients
		if clients == 1 {
			break
		}
		if time.Now().After(detachDeadline) {
			t.Fatalf("desktop detach did not join its real socket: clients=%d, want only fixture control", clients)
		}
		time.Sleep(25 * time.Millisecond)
	}
	bridge.send("finish")
	bridge.wait()
	if got := actionHits.Load(); got != 1 {
		t.Fatalf("desktop interrupted action replayed: private POSTs=%d, want 1", got)
	}
	t.Log("desktop_retained_client_recovered finite_retries_exhausted=true pending_failed=true fresh_hydration=true replacement_http_hits=1 mutation_posts=1 detached_and_joined=true")
}

type desktopRestartBridge struct {
	t      *testing.T
	stdin  io.WriteCloser
	stages chan string
	done   chan struct{}
	output *serviceOutput
	err    error
}

// exec owns and joins the stdio copy loops before done closes. Protocol writes
// never block on a consumer, and malformed/overflowed output fails cmd.Wait.
type desktopRestartProtocol struct {
	stages chan string
	line   []byte
}

func (p *desktopRestartProtocol) Write(data []byte) (int, error) {
	for _, value := range data {
		if len(p.line) >= 64*1024 {
			return 0, fmt.Errorf("desktop stdout line exceeded coordination bound")
		}
		if value != '\n' {
			p.line = append(p.line, value)
			continue
		}
		line := strings.TrimSuffix(string(p.line), "\r")
		p.line = p.line[:0]
		if strings.HasPrefix(line, "PAIR_DESKTOP ") {
			select {
			case p.stages <- strings.TrimPrefix(line, "PAIR_DESKTOP "):
			default:
				return 0, fmt.Errorf("desktop coordination stage queue overflow")
			}
		}
	}
	return len(data), nil
}

func startDesktopRestartBridge(t *testing.T, f *serviceFixture, node, desktop, vitest string, upstreamPort int) *desktopRestartBridge {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	bridge := &desktopRestartBridge{t: t, stages: make(chan string, 16), done: make(chan struct{}), output: &serviceOutput{}}
	cmd := exec.CommandContext(ctx, node, vitest, "run", "--config", "tests/service-restart-reconnect/vitest.config.ts", "--configLoader", "native")
	cmd.Dir = desktop
	// Runtime startup and dependency selection are explicit. Never inherit a
	// global loader, NODE_PATH, another Vitest run, or another fixture's selectors.
	for _, entry := range f.env {
		key, _, _ := strings.Cut(entry, "=")
		upper := strings.ToUpper(key)
		if strings.HasPrefix(upper, "NODE_") || strings.HasPrefix(upper, "ESBUILD_") || strings.HasPrefix(upper, "VITEST") || strings.HasPrefix(upper, "TINYPOOL") || strings.HasPrefix(upper, "PAIR_DESKTOP_") {
			continue
		}
		cmd.Env = append(cmd.Env, entry)
	}
	userData := filepath.Join(f.dir, "desktop-user-data")
	cmd.Env = append(cmd.Env, "PAIR_DESKTOP_USER_DATA="+userData, "PAIR_USER_DATA="+userData,
		"PAIR_DESKTOP_PRODUCTS="+filepath.Dir(serviceBin), "PAIR_DESKTOP_ENDPOINT="+f.endpoint,
		"PAIR_DESKTOP_UPSTREAM_PORT="+strconv.Itoa(upstreamPort), "PAIR_DESKTOP_COORDINATION="+filepath.Base(f.dir),
		"NO_COLOR=1")
	cmd.Stdout = io.MultiWriter(bridge.output, &desktopRestartProtocol{stages: bridge.stages})
	cmd.Stderr = bridge.output
	cmd.WaitDelay = 2 * time.Second
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		t.Fatalf("desktop coordination stdin: %v", err)
	}
	bridge.stdin = stdin
	if err := cmd.Start(); err != nil {
		cancel()
		if closeErr := stdin.Close(); closeErr != nil {
			t.Errorf("close unstarted desktop stdin: %v", closeErr)
		}
		t.Fatalf("start explicit local Node/Vitest: %v", err)
	}
	go func() { bridge.err = cmd.Wait(); close(bridge.done) }()
	t.Cleanup(func() {
		if err := stdin.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			t.Errorf("close desktop coordination stdin: %v", err)
		}
		cancel()
		select {
		case <-bridge.done:
		case <-time.After(serviceCallTimeout):
			t.Error("desktop Node process and stdio owners did not join after cancellation")
		}
	})
	return bridge
}

func (b *desktopRestartBridge) await(want string, timeout time.Duration) {
	b.t.Helper()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case got := <-b.stages:
		if got != want {
			b.t.Fatalf("desktop coordination stage=%q want=%q; output: %s", got, want, b.output.String())
		}
	case <-b.done:
		b.t.Fatalf("desktop exited before %s: %v; output: %s", want, b.err, b.output.String())
	case <-timer.C:
		b.t.Fatalf("desktop missing %s within %s; output: %s", want, timeout, b.output.String())
	}
}

func (b *desktopRestartBridge) send(command string) {
	b.t.Helper()
	if _, err := io.WriteString(b.stdin, command+"\n"); err != nil {
		b.t.Fatalf("desktop coordination command %s: %v", command, err)
	}
}

func (b *desktopRestartBridge) wait() {
	b.t.Helper()
	select {
	case <-b.done:
		if b.err != nil {
			b.t.Fatalf("desktop Vitest proof failed: %v; output: %s", b.err, b.output.String())
		}
	case <-time.After(20 * time.Second):
		b.t.Fatalf("desktop did not exit after real detach; output: %s", b.output.String())
	}
}
