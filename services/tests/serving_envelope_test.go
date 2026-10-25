// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/testbin"
)

const (
	neRuntimeArg = "--ne-serving-envelope-runtime"
	neRuntimeEnv = "NVPAIR_NE_SERVING_EXECUTABLE"
	neModel      = "ne-fixture"
	neContext    = 2048
	neOutput     = 128
	neMemory     = 4096
	neLimit      = 64 << 10
	neCallBudget = 35 * time.Second
)

// Dispatch before TestMain: this is the same wrapper-owned tests.test.exe,
// with native engine arguments, rather than another test runner or binary role.
func init() {
	// applyLiteralLaunch puts the two managed native controls before LaunchText.
	// An environment selector alone rejects a malformed child, never runs it.
	if len(os.Args) < 6 || os.Args[5] != neRuntimeArg {
		if os.Getenv(neRuntimeEnv) != "" {
			os.Exit(2)
		}
		return
	}
	image, err := os.Executable()
	if err != nil || runtime.GOOS != "windows" || os.Getenv(neRuntimeEnv) != image || !neFixedTestImage(image) {
		os.Exit(2)
	}
	if err := runNEServingRuntime(os.Args[1:]); err != nil {
		// Arguments, configuration, and synthetic inference bodies stay private.
		fmt.Fprintln(os.Stderr, "NE fake runtime rejected its private launch:", err)
		os.Exit(2)
	}
	os.Exit(0)
}

func neFixedTestImage(image string) bool {
	root := os.Getenv("NVPAIR_TEST_BIN_DIR")
	return filepath.IsAbs(root) && strings.EqualFold(filepath.Clean(image), filepath.Join(root, "go-test", "tests", "tests.test.exe"))
}

type neRuntimeConfig struct {
	Root  string `json:"root"`
	Nonce string `json:"nonce"`
	Port  int    `json:"port"`
}

type neRuntimeStart struct {
	Nonce  string   `json:"nonce"`
	PID    int      `json:"pid"`
	Preset string   `json:"preset"`
	Args   []string `json:"args"`
}

type neRuntimeState struct {
	Nonce         string `json:"nonce"`
	PID           int    `json:"pid"`
	Loaded        bool   `json:"loaded"`
	Chats         int    `json:"chats"`
	Loads         int    `json:"loads"`
	Unloads       int    `json:"unloads"`
	ModelProps    int    `json:"modelProps"`
	Unexpected    int    `json:"unexpected"`
	LeakedHeaders int    `json:"leakedHeaders"`
}

func neSafePort(port int) bool {
	_, protected := installedServicePorts[port]
	return port > 0 && port <= 65535 && !protected
}

func neWithin(root, path string) bool {
	if !filepath.IsAbs(root) || !filepath.IsAbs(path) {
		return false
	}
	rel, err := filepath.Rel(root, path)
	return err == nil && rel != "." && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}

func neDecode(reader io.Reader, value any) error {
	decoder := json.NewDecoder(io.LimitReader(reader, neLimit+1))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		return err
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		return errors.New("private JSON must contain exactly one value")
	}
	return nil
}

func runNEServingRuntime(args []string) (resultErr error) {
	if len(args) < 7 || args[0] != "--host" || args[1] != "127.0.0.1" || args[2] != "--port" || args[4] != neRuntimeArg {
		return errors.New("missing exact managed prefix and dispatch marker")
	}
	options := make(map[string]string)
	for i := 0; i < len(args); i += 2 {
		if i == 4 {
			i++ // Only this position can contain the valueless dispatch marker.
		}
		if i+1 >= len(args) {
			return errors.New("native option is missing its value")
		}
		switch args[i] {
		case "--ne-serving-config", "--host", "--port", "--cors-origins", "--models-preset", "--models-max":
		default:
			return errors.New("unexpected native option")
		}
		if _, duplicate := options[args[i]]; duplicate {
			return errors.New("duplicate native option")
		}
		options[args[i]] = args[i+1]
	}
	if value, present := options["--cors-origins"]; present && value != "" {
		return errors.New("unexpected CORS control")
	}
	configPath := options["--ne-serving-config"]
	if !filepath.IsAbs(configPath) {
		return errors.New("private configuration must be absolute")
	}
	data, err := os.ReadFile(configPath)
	if err != nil || len(data) > neLimit {
		return errors.New("cannot read bounded private configuration")
	}
	var config neRuntimeConfig
	if err := neDecode(bytes.NewReader(data), &config); err != nil {
		return errors.New("invalid private configuration")
	}
	nonce, err := hex.DecodeString(config.Nonce)
	port, portErr := strconv.Atoi(options["--port"])
	if err != nil || len(nonce) != 16 || portErr != nil || !neSafePort(port) || port != config.Port || options["--host"] != "127.0.0.1" || options["--models-max"] != "1" || configPath != filepath.Join(config.Root, "runtime.json") {
		return errors.New("private identity or endpoint mismatch")
	}
	preset := options["--models-preset"]
	snapshotRoot := filepath.Join(jointAppDir(config.Root), "engine-bin", "serving", "llamacpp")
	if !neWithin(snapshotRoot, preset) || !strings.HasPrefix(filepath.Base(filepath.Dir(preset)), "generation-") || filepath.Base(preset) != "models.ini" {
		return errors.New("launch did not select a private generated preset")
	}
	if _, err := neReadPreset(preset, filepath.Join(config.Root, "model.gguf")); err != nil {
		return err
	}
	listener, err := net.Listen("tcp4", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
	if err != nil {
		return errors.New("private runtime bind failed")
	}
	defer func() { resultErr = errors.Join(resultErr, listener.Close()) }()
	start := neRuntimeStart{Nonce: config.Nonce, PID: os.Getpid(), Preset: preset, Args: args}
	encoded, err := json.Marshal(start)
	if err != nil {
		return err
	}
	log, err := os.OpenFile(filepath.Join(config.Root, "starts.jsonl"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return errors.New("cannot record private startup")
	}
	_, writeErr := log.Write(append(encoded, '\n'))
	if err := errors.Join(writeErr, log.Close()); err != nil {
		return errors.New("cannot finish private startup record")
	}
	state := neRuntimeState{Nonce: config.Nonce, PID: os.Getpid()}
	var mu sync.Mutex
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		var response any
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/health":
			response = state
		case r.Method == http.MethodGet && r.URL.Path == "/__ne/control" && r.Header.Get("X-NE-Fixture-Nonce") == config.Nonce:
			response = state
		case r.Method == http.MethodGet && r.URL.Path == "/props" && r.URL.RawQuery == "":
			response = map[string]string{"role": "router"}
		case r.Method == http.MethodGet && r.URL.Path == "/props" && r.URL.Query().Get("model") == neModel && r.URL.Query().Get("autoload") == "false" && state.Loaded:
			state.ModelProps++
			response = map[string]any{"default_generation_settings": map[string]int{"n_ctx": neContext}, "total_slots": 1}
		case r.Method == http.MethodGet && r.URL.Path == "/models":
			status := "unloaded"
			if state.Loaded {
				status = "loaded"
			}
			response = map[string]any{"data": []any{map[string]any{"id": neModel, "status": map[string]string{"value": status}}}}
		case r.Method == http.MethodPost && (r.URL.Path == "/models/load" || r.URL.Path == "/models/unload"):
			var model struct {
				Model string `json:"model"`
			}
			if err := neDecode(r.Body, &model); err != nil || model.Model != neModel {
				state.Unexpected++
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			status := "unloaded"
			state.Loaded = r.URL.Path == "/models/load"
			if state.Loaded {
				state.Loads++
				status = "loaded"
			} else {
				state.Unloads++
			}
			response = map[string]any{"ok": true, "model": neModel, "status": status}
		case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
			state.Chats++
			if r.Header.Get(nodepolicy.StrictAdmissionHeader) != "" {
				state.LeakedHeaders++
			}
			var chat struct {
				Model    string `json:"model"`
				Messages []struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"messages"`
				MaxTokens int  `json:"max_tokens"`
				Stream    bool `json:"stream"`
			}
			if err := neDecode(r.Body, &chat); err != nil || chat.Model != neModel || !state.Loaded || chat.MaxTokens != neOutput || chat.Stream || len(chat.Messages) != 1 {
				state.Unexpected++
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			response = map[string]any{"id": "ne-chat", "object": "chat.completion", "model": neModel, "choices": []any{map[string]any{"index": 0, "message": map[string]string{"role": "assistant", "content": "synthetic reply"}, "finish_reason": "stop"}}}
		default:
			state.Unexpected++
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if err := json.NewEncoder(w).Encode(response); err != nil {
			state.Unexpected++
		}
	})
	server := &http.Server{Handler: handler, ReadHeaderTimeout: 2 * time.Second, ReadTimeout: 5 * time.Second, WriteTimeout: 5 * time.Second, IdleTimeout: 3 * time.Second}
	return server.Serve(listener)
}

// Read only the generated fixture model section, with independent expected
// native values. The fake never interprets or opens model content as an engine.
func neReadPreset(path, model string) (map[string]string, error) {
	data, err := os.ReadFile(path)
	if err != nil || len(data) > neLimit {
		return nil, errors.New("cannot read bounded generated preset")
	}
	section := ""
	values := make(map[string]string)
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = strings.TrimSuffix(strings.TrimPrefix(line, "["), "]")
			continue
		}
		if section != neModel || line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		if _, duplicate := values[key]; !ok || duplicate {
			return nil, errors.New("invalid generated model section")
		}
		values[key] = value
	}
	for key, want := range map[string]string{"ctx-size": "2048", "parallel": "1", "n-predict": "128", "kv-unified": "false", "fit": "off", "load-on-startup": "false", "sleep-idle-seconds": "-1", "model": model} {
		if values[key] != want {
			return nil, fmt.Errorf("generated preset does not preserve finite %s", key)
		}
	}
	return values, nil
}

type neServingFixture struct {
	root, app, image, scanner, source              string
	nonce                                          string
	enginePort, facadePort, scannerPort, inertPort int
	ports                                          fixtureServicePorts
	client                                         *http.Client
	p                                              *interopProcess
	fake                                           *os.Process
	residentRetryInterval                          time.Duration
}

func newNEServingFixture(t *testing.T) *neServingFixture {
	t.Helper()
	if runtime.GOOS != "windows" || !strings.Contains(os.Getenv("GOFLAGS"), "nvpair-test-exec") {
		t.Fatal("NE process acceptance requires the PC3 Windows fixed-path test wrapper")
	}
	image, err := os.Executable()
	if err != nil || !neFixedTestImage(image) {
		t.Fatal("NE fake must reuse the exact fixed tests executable")
	}
	for _, assignment := range os.Environ() {
		key, _, _ := strings.Cut(assignment, "=")
		upper := strings.ToUpper(key)
		if strings.HasPrefix(upper, "LLAMA_") || upper == neRuntimeEnv || (strings.HasPrefix(upper, "NVPAIR_TEST_") && upper != "NVPAIR_TEST_BIN_DIR") || strings.HasPrefix(upper, "NVPAIR_STRATA_CONTRACT_") || strings.HasPrefix(upper, "NVPAIR_OBSERVATION_") {
			// The child harness inherits this test process, then appends its
			// explicit private overrides. Remove ambient launch selectors first:
			// empty native aliases can still act by presence. Setenv registers
			// restoration after all later fixture cleanup; the host shell is untouched.
			t.Setenv(key, "")
			if err := os.Unsetenv(key); err != nil {
				t.Fatalf("remove inherited launch selector %s: %v", key, err)
			}
			if _, present := os.LookupEnv(key); present {
				t.Fatalf("inherited launch selector %s survived fixture isolation", key)
			}
		}
	}
	f := &neServingFixture{root: t.TempDir(), image: image}
	f.app = jointAppDir(f.root)
	f.source = filepath.Join(f.root, "approved.ini")
	identity := make([]byte, 16)
	if _, err := rand.Read(identity); err != nil {
		t.Fatal(err)
	}
	f.nonce = hex.EncodeToString(identity)
	f.client = &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{Proxy: nil}, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	t.Cleanup(f.client.CloseIdleConnections)
	// Hold fake/facade/scanner reservations while allocating the service plan.
	var reservations []net.Listener
	for range 3 {
		listener := listenFixtureLoopback(t)
		reservations = append(reservations, listener)
		defer func() {
			if err := listener.Close(); err != nil {
				t.Errorf("release private port reservation: %v", err)
			}
		}()
	}
	f.enginePort = reservations[0].Addr().(*net.TCPAddr).Port
	f.facadePort = reservations[1].Addr().(*net.TCPAddr).Port
	f.scannerPort = reservations[2].Addr().(*net.TCPAddr).Port
	inert := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) }))
	t.Cleanup(inert.Close)
	f.inertPort = inert.Listener.Addr().(*net.TCPAddr).Port
	f.ports = reserveFixtureServicePorts(t)
	jointIsolateEngines(t, f.root, f.inertPort)
	f.scanner = filepath.Join(testbin.Dir(t, "tests", "subscription-scanner", "subscription-scanner"), testbin.Exe("subscription-scanner"))
	jointCopyExecutable(t, image, f.scanner)
	if err := os.MkdirAll(filepath.Join(f.root, "cluster"), 0o700); err != nil {
		t.Fatal(err)
	}
	rt := map[string]any{
		"mode": "process", "bin": image, "cli": filepath.Join(f.root, "absent-cli"), "port": f.enginePort, "bind": "127.0.0.1",
		"env":    map[string]string{neRuntimeEnv: image},
		"ready":  map[string]any{"http": "http://127.0.0.1:{port}/props", "status": 200, "timeout_s": 10, "json_match": map[string]string{"field": "role", "value": "router"}},
		"health": map[string]any{"http": "http://127.0.0.1:{port}/health", "status": 200, "interval_s": 5},
		"stop":   map[string]any{"signal": "term", "grace_s": 1},
	}
	jointWriteJSON(t, filepath.Join(f.app, "engines", "llamacpp.json"), map[string]any{
		"engine": "llamacpp", "runtime": rt, "models_dir": filepath.Join(f.root, "models"),
		"platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{image}, "runtime": rt, "models_dir": filepath.Join(f.root, "models")}},
	})
	jointWriteJSON(t, filepath.Join(f.root, "runtime.json"), neRuntimeConfig{Root: f.root, Nonce: f.nonce, Port: f.enginePort})
	jointWriteJSON(t, filepath.Join(f.root, "engine-settings-operations.json"), map[string]any{"llamacpp": map[string]any{"explicit": true, "snapshot": settings.Snapshot{
		Engine: "llamacpp", Revision: 1, AppliedRevision: 1, Phase: "idle", Format: "pair-arguments-v1", Settings: settings.Config{ServerPort: f.enginePort, ProxyPort: f.facadePort},
	}}})
	policy := nodepolicy.Default()
	policy.Revision = "1"
	policy.Admission.MaxConcurrentPerModel = 1
	policy.Admission.MemoryBudgetBytes = neMemory
	policy.Idle.StartOnDemand = false
	policy.Profiles = []nodepolicy.Profile{{Name: "ne-process", Engine: "llamacpp", Model: neModel, ContextTokens: neContext, MaxOutputTokens: neOutput, MemoryBytes: neMemory, MaxConcurrent: 1, Qualification: &nodepolicy.QualificationRef{ID: "ne-synthetic", Revision: "1"}}}
	if err := policy.Validate(); err != nil {
		t.Fatalf("invalid finite synthetic policy: %v", err)
	}
	jointWriteJSON(t, filepath.Join(f.app, nodepolicy.FileName), policy)
	t.Cleanup(func() { f.stop(t) })
	return f
}

func (f *neServingFixture) start(t *testing.T) {
	t.Helper()
	if f.p != nil {
		t.Fatal("fixture broker already running")
	}
	// Default sibling resolution must see only this empty private directory.
	for _, worker := range []string{"nvpair-node-info", "nvpair-workload-manager", "nvpair-errors", "nvpair-manual-nodes", "nvpair-node-settings", "nvpair-cluster-manager", "nvpair-job-scheduler"} {
		if _, err := os.Stat(filepath.Join(f.root, testbin.Exe(worker))); !errors.Is(err, os.ErrNotExist) {
			t.Fatal("optional worker was not absent from private cwd")
		}
	}
	f.assertIsolationFiles(t)
	args := append([]string{"--scanner-path", f.scanner, "--engine-manager-path", engineMgrBin, "--proxy-path", proxyBin, "--proxy-engines=llamacpp", "--cluster-dir", filepath.Join(f.root, "cluster"), "--log-level", "info"}, f.ports.args()...)
	args = append(args, f.residentFixtureWaitArgs()...)
	f.p = startInteropProcess(t, f.root, brokerBin, f.ports.engineManagerHTTP, []string{"OLLAMA_HOST=", "NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + f.scanner, "NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(f.scannerPort)}, args...)
	f.p.ports = append(f.p.ports, f.ports.engineControl, f.scannerPort, f.facadePort)
	for _, path := range []string{engineMgrBin, proxyBin, f.scanner} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		f.p.paths = append(f.p.paths, canonical)
	}
	waitForMethod(t, f.p.msgs, "app:ready", neCallBudget)
	if port := waitEngineProxyReady(t, "llamacpp-proxy", f.p.in, f.p.msgs, neCallBudget); port != f.facadePort {
		t.Fatalf("llama facade port=%d, want reserved %d", port, f.facadePort)
	}
	f.assertIsolation(t)
	t.Logf("ne_fixture_products facade_bind=wildcard:%d em_bind=wildcard:%d ec_bind=wildcard:%d clients=loopback", f.facadePort, f.ports.engineManagerHTTP, f.ports.engineControl)
}

func (f *neServingFixture) residentFixtureWaitArgs() []string {
	retry := f.residentRetryInterval
	if retry == 0 {
		retry = time.Second
	}
	return []string{"--test-resident-retry-interval=" + retry.String(), "--test-advertise-interval=250ms"}
}

func (f *neServingFixture) call(t *testing.T, method string, params, result any) {
	t.Helper()
	response := callRPC(t, f.p.in, f.p.msgs, method, params, neCallBudget)
	if response.Error != nil {
		t.Fatalf("%s: code=%d message=%s", method, response.Error.Code, response.Error.Message)
	}
	if result != nil {
		if err := json.Unmarshal(response.Result, result); err != nil {
			t.Fatalf("decode %s: %v", method, err)
		}
	}
}

type neFixtureRuntime struct {
	Bin   string `json:"bin"`
	CLI   string `json:"cli"`
	Port  int    `json:"port"`
	Ready struct {
		HTTP string `json:"http"`
	} `json:"ready"`
	Health struct {
		HTTP string `json:"http"`
	} `json:"health"`
}

type neFixturePlatform struct {
	Detect    []string         `json:"detect"`
	Runtime   neFixtureRuntime `json:"runtime"`
	ModelsDir string           `json:"models_dir"`
}

func (f *neServingFixture) assertRuntimeIsolation(t *testing.T, engine, level string, rt neFixtureRuntime, modelsDir string) {
	t.Helper()
	wantBin, wantCLI := filepath.Join(f.root, "absent-"+engine), filepath.Join(f.root, "absent-"+engine)
	wantPort, wantProbe := f.inertPort, "http://127.0.0.1:{port}/joint-no-engine"
	wantHealth, wantModels := wantProbe, filepath.Join(f.root, "absent-models", engine)
	if engine == "llamacpp" {
		wantBin, wantCLI = f.image, filepath.Join(f.root, "absent-cli")
		wantPort, wantProbe = f.enginePort, "http://127.0.0.1:{port}/props"
		wantHealth, wantModels = "http://127.0.0.1:{port}/health", filepath.Join(f.root, "models")
	}
	if rt.Bin != wantBin || rt.CLI != wantCLI || rt.Port != wantPort || rt.Ready.HTTP != wantProbe || rt.Health.HTTP != wantHealth || modelsDir != wantModels || !neSafePort(rt.Port) {
		t.Fatalf("%s %s isolation mismatch: bin=%t cli=%t port=%t ready=%t health=%t models=%t safe_port=%t", engine, level, rt.Bin == wantBin, rt.CLI == wantCLI, rt.Port == wantPort, rt.Ready.HTTP == wantProbe, rt.Health.HTTP == wantHealth, modelsDir == wantModels, neSafePort(rt.Port))
	}
}

func (f *neServingFixture) assertPlatformIsolation(t *testing.T, engine, level string, platforms map[string]neFixturePlatform) {
	t.Helper()
	platform, ok := platforms[runtime.GOOS+"/"+runtime.GOARCH]
	wantBin := filepath.Join(f.root, "absent-"+engine)
	if engine == "llamacpp" {
		wantBin = f.image
	}
	if !ok || !reflect.DeepEqual(platform.Detect, []string{wantBin}) {
		t.Fatalf("%s %s host detection escaped the private fixture (host_present=%t)", engine, level, ok)
	}
	f.assertRuntimeIsolation(t, engine, level, platform.Runtime, platform.ModelsDir)
}

func (f *neServingFixture) assertIsolationFiles(t *testing.T) {
	t.Helper()
	// Shared defaults and host overrides must both be private before startup.
	// The registry later consumes shared defaults into its platform blocks.
	for _, engine := range engines.Names() {
		data, err := os.ReadFile(filepath.Join(f.app, "engines", engine+".json"))
		if err != nil || len(data) > neLimit {
			t.Fatalf("%s cannot read bounded private isolation override: %v", engine, err)
		}
		var override struct {
			Runtime   neFixtureRuntime             `json:"runtime"`
			ModelsDir string                       `json:"models_dir"`
			Platforms map[string]neFixturePlatform `json:"platforms"`
		}
		if err := json.Unmarshal(data, &override); err != nil {
			t.Fatalf("%s decode private isolation override: %v", engine, err)
		}
		f.assertRuntimeIsolation(t, engine, "raw shared", override.Runtime, override.ModelsDir)
		f.assertPlatformIsolation(t, engine, "raw host", override.Platforms)
	}
}

func (f *neServingFixture) assertIsolation(t *testing.T) {
	t.Helper()
	for _, engine := range engines.Names() {
		var manifest struct {
			Platforms map[string]neFixturePlatform `json:"platforms"`
		}
		f.call(t, "engine:describe", map[string]string{"engine": engine}, &manifest)
		f.assertPlatformIsolation(t, engine, "effective host", manifest.Platforms)
	}
}

func neEventually(t *testing.T, description string, budget time.Duration, check func() bool) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), budget)
	defer cancel()
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		if check() {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("%s did not converge within %s", description, budget)
		case <-ticker.C:
		}
	}
}

func (f *neServingFixture) snapshot(t *testing.T) settings.Snapshot {
	t.Helper()
	var snapshot settings.Snapshot
	f.call(t, "engine:get-settings", settings.Request{Engine: "llamacpp"}, &snapshot)
	return snapshot
}

type neModels struct {
	ByEngine map[string][]string       `json:"modelsByEngine"`
	Loaded   map[string][]string       `json:"loadedByEngine"`
	Policy   *nodepolicy.Advertisement `json:"policy"`
}

func (f *neServingFixture) models(t *testing.T) neModels {
	t.Helper()
	var models neModels
	f.call(t, "engine:models", nil, &models)
	return models
}

func neCapability(models neModels) *nodepolicy.StrictAdmissionModel {
	if models.Policy == nil || models.Policy.StrictAdmission == nil {
		return nil
	}
	for _, model := range models.Policy.StrictAdmission.Models {
		if model.Engine == "llamacpp" && model.Model == neModel {
			return &model
		}
	}
	return nil
}

func (f *neServingFixture) http(method string, port int, path string, body []byte, strict *nodepolicy.StrictAdmissionRequest) (int, []byte, error) {
	if !neSafePort(port) {
		return 0, nil, errors.New("refusing protected HTTP destination")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, method, "http://"+net.JoinHostPort("127.0.0.1", strconv.Itoa(port))+path, bytes.NewReader(body))
	if err != nil {
		return 0, nil, err
	}
	request.Header.Set("X-NE-Fixture-Nonce", f.nonce)
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if strict != nil {
		data, err := json.Marshal(strict)
		if err != nil {
			return 0, nil, err
		}
		request.Header.Set(nodepolicy.StrictAdmissionHeader, string(data))
	}
	response, err := f.client.Do(request)
	if err != nil {
		return 0, nil, err
	}
	data, readErr := io.ReadAll(io.LimitReader(response.Body, neLimit+1))
	if err := errors.Join(readErr, response.Body.Close()); err != nil {
		return 0, nil, err
	}
	if len(data) > neLimit {
		return 0, nil, errors.New("fixture response exceeded bound")
	}
	return response.StatusCode, data, nil
}

func (f *neServingFixture) state() (neRuntimeState, error) {
	status, data, err := f.http(http.MethodGet, f.enginePort, "/__ne/control", nil, nil)
	if err != nil {
		return neRuntimeState{}, err
	}
	var state neRuntimeState
	if status != http.StatusOK || neDecode(bytes.NewReader(data), &state) != nil || state.Nonce != f.nonce || state.PID <= 0 {
		return neRuntimeState{}, errors.New("private runtime did not authenticate")
	}
	return state, nil
}

func (f *neServingFixture) retainRuntime(t *testing.T) {
	t.Helper()
	state, err := f.state()
	if err != nil {
		t.Fatalf("authenticate manager-owned fake: %v", err)
	}
	process, err := os.FindProcess(state.PID)
	if err != nil {
		t.Fatalf("retain authenticated fake handle: %v", err)
	}
	fresh, err := f.state()
	if err != nil || fresh.PID != process.Pid {
		if releaseErr := process.Release(); releaseErr != nil {
			t.Errorf("release unverified handle: %v", releaseErr)
		}
		t.Fatal("fake identity changed while retaining its handle")
	}
	f.fake = process
}

func (f *neServingFixture) stop(t *testing.T) {
	t.Helper()
	if f.p == nil {
		return
	}
	// A fatal assertion may occur between engine:start and handle retention.
	if f.fake == nil {
		if state, err := f.state(); err == nil {
			candidate, openErr := os.FindProcess(state.PID)
			if openErr == nil {
				if fresh, err := f.state(); err == nil && fresh.PID == candidate.Pid {
					f.fake = candidate
				} else if err := candidate.Release(); err != nil {
					t.Errorf("release unverified cleanup handle: %v", err)
				}
			} else {
				t.Errorf("retain authenticated cleanup handle: %v", openErr)
			}
		}
	}
	f.p.cleanup()
	f.p = nil
	if f.fake != nil {
		if state, err := f.state(); err == nil && state.PID == f.fake.Pid {
			// Only a retained handle and fresh nonce/PID handshake authorize a kill.
			if err := f.fake.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
				t.Errorf("terminate authenticated survivor: %v", err)
			}
			t.Error("manager-owned fake survived broker shutdown")
		} else if err != nil && !neConnectionRefused(err) {
			t.Errorf("cannot authenticate fake shutdown: %v", err)
		}
		if err := f.fake.Release(); err != nil {
			t.Errorf("release retained fake handle: %v", err)
		}
		f.fake = nil
	}
	neEventually(t, "fake listener shutdown", 5*time.Second, func() bool {
		_, err := f.state()
		return neConnectionRefused(err)
	})
}

func neConnectionRefused(err error) bool {
	var number syscall.Errno
	return errors.As(err, &number) && (number == syscall.ECONNREFUSED || number == syscall.Errno(10061))
}

var neIdentityLine = regexp.MustCompile(`\[nvpair-engine-manager\] INFO engine identity engine=llamacpp run_id=(run-[0-9a-f]+) launch_digest=([0-9a-f]{64}) models_with_digest=1`)

func (f *neServingFixture) identity(t *testing.T) (string, string) {
	t.Helper()
	var run, launch string
	neEventually(t, "produced llama identity metadata", neCallBudget, func() bool {
		matches := neIdentityLine.FindAllStringSubmatch(f.p.output.String(), -1)
		if len(matches) == 0 {
			return false
		}
		last := matches[len(matches)-1]
		run, launch = last[1], last[2]
		return true
	})
	return run, launch
}

func (f *neServingFixture) starts(t *testing.T) []neRuntimeStart {
	t.Helper()
	file, err := os.Open(filepath.Join(f.root, "starts.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			t.Error(err)
		}
	}()
	scanner := bufio.NewScanner(io.LimitReader(file, 4*neLimit))
	scanner.Buffer(make([]byte, 4096), neLimit)
	var starts []neRuntimeStart
	for scanner.Scan() {
		var start neRuntimeStart
		if err := neDecode(bytes.NewReader(scanner.Bytes()), &start); err != nil {
			t.Fatal(err)
		}
		if start.Nonce != f.nonce || start.PID <= 0 || start.Preset == f.source || !neWithin(filepath.Join(f.app, "engine-bin", "serving", "llamacpp"), start.Preset) {
			t.Fatal("private startup record did not identify generated launch")
		}
		starts = append(starts, start)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return starts
}

func neAssertColdReadback(t *testing.T, snapshot settings.Snapshot, launched bool) engines.ServingEnvelope {
	t.Helper()
	readback := snapshot.Serving
	if readback == nil || readback.Status != "unverified" || len(readback.Configured) != 1 || len(readback.Observed) != 0 {
		t.Fatal("cold declaration was absent or misrepresented as observed runtime proof")
	}
	wantValidated := 0
	if launched {
		wantValidated = 1
	}
	if len(readback.LaunchValidated) != wantValidated || snapshot.Running != launched || snapshot.Adopted {
		t.Fatal("cold owned-launch validation did not follow actual process state")
	}
	envelope := readback.Configured[0]
	if err := envelope.Validate(neModel, envelope.ContentDigest); err != nil {
		t.Fatal(err)
	}
	if envelope.PerSlotContextTokens != neContext || envelope.TotalContextTokens != neContext || envelope.Slots != 1 || envelope.MaxOutputTokens != neOutput || envelope.RouterCapacity != 1 || envelope.Retention != "-1" || envelope.Memory.HostRAMBytes == nil || *envelope.Memory.HostRAMBytes != neMemory || envelope.Memory.VRAMBytes == nil || *envelope.Memory.VRAMBytes != 0 {
		t.Fatal("finite native cold envelope differs from approved fixture")
	}
	if launched && !reflect.DeepEqual(readback.LaunchValidated[0], envelope) {
		t.Fatal("launch-validated envelope differs from configured envelope")
	}
	return envelope
}

func TestNEServingEnvelope(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("NE serving envelope process acceptance is reserved for Windows PC3")
	}
	f := newNEServingFixture(t)
	modelBytes := []byte("NE synthetic model; no model runtime or hardware qualification")
	modelPath := filepath.Join(f.root, "model.gguf")
	if err := os.WriteFile(modelPath, modelBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	source := "[*]\nfit = off\nload-on-startup = false\n\n[" + neModel + "]\nmodel = " + filepath.ToSlash(modelPath) + "\nctx-size = 2048\nparallel = 1\nkv-unified = false\nsleep-idle-seconds = -1\n"
	if err := os.WriteFile(f.source, []byte(source), 0o600); err != nil {
		t.Fatal(err)
	}
	f.start(t)
	before := f.snapshot(t)
	ram, vram := int64(neMemory), int64(0)
	modelHash := sha256.Sum256(modelBytes)
	request := settings.Request{Engine: "llamacpp", ExpectedRevision: before.Revision, RequestID: "ne-apply", Settings: settings.Config{
		ServerPort: f.enginePort, ProxyPort: f.facadePort,
		LaunchText: neRuntimeArg + " --ne-serving-config " + strconv.Quote(filepath.Join(f.root, "runtime.json")) + " --models-preset " + strconv.Quote(f.source) + " --models-max 1",
		Serving:    &engines.ServingConfig{Version: 1, Models: []engines.ServingModel{{ModelID: neModel, MaxOutputTokens: neOutput, Memory: engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram}, Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: hex.EncodeToString(modelHash[:]), Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}}}}}},
	}}
	var preview settings.Preview
	f.call(t, "engine:preview-settings", request, &preview)
	if len(preview.Errors) != 0 || preview.Conflict != nil || preview.Restart || preview.Settings.Serving == nil || len(preview.Settings.Serving.Models) != 1 {
		t.Fatalf("finite preview rejected: errors=%v conflict=%v", preview.Errors, preview.Conflict)
	}
	approved := preview.Settings.Serving.Models[0]
	if approved.ExpectedNativeSourceDigest == "" || len(approved.Artifacts) != 1 || approved.Artifacts[0].Binding == nil {
		t.Fatal("preview did not produce source digest and artifact binding")
	}
	request.Settings = preview.Settings.Clone()
	request.ExpectedRevision = preview.Revision
	var receipt struct {
		Revision uint64 `json:"Revision"`
		Phase    string `json:"Phase"`
	}
	f.call(t, "engine:apply-settings", request, &receipt)
	cold := f.snapshot(t)
	if receipt.Phase != "succeeded" || receipt.Revision <= before.Revision || cold.Revision != receipt.Revision || cold.AppliedRevision != receipt.Revision || !cold.Settings.Equal(preview.Settings) {
		t.Fatal("E apply did not acknowledge and preserve the exact reviewed configuration")
	}
	configuredEnvelope := neAssertColdReadback(t, cold, false)
	if _, err := os.Stat(filepath.Join(f.root, "starts.jsonl")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("preview/cold apply started an engine")
	}
	if _, err := os.Stat(filepath.Join(f.app, "local-qualification.json")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("qualification evidence existed before bootstrap")
	}
	f.call(t, "engine:start", map[string]string{"engine": "llamacpp"}, nil)
	f.retainRuntime(t)
	firstEnvelope := neAssertColdReadback(t, f.snapshot(t), true)
	if !reflect.DeepEqual(configuredEnvelope, firstEnvelope) {
		t.Fatal("starting the generated launch changed its approved cold identity")
	}
	firstRun, firstLaunch := f.identity(t)
	firstStarts := f.starts(t)
	if len(firstStarts) != 1 {
		t.Fatalf("bootstrap starts=%d, want 1", len(firstStarts))
	}
	if _, err := neReadPreset(firstStarts[0].Preset, modelPath); err != nil {
		t.Fatal(err)
	}
	if neCapability(f.models(t)) != nil {
		t.Fatal("broker advertised strict capacity without local qualification")
	}
	state, err := f.state()
	if err != nil || state.Loaded || state.Chats != 0 || state.Loads != 0 || state.ModelProps != 0 || state.Unexpected != 0 {
		t.Fatal("cold identity/readback caused inference, loading, or model props")
	}
	f.stop(t)
	jointWriteJSON(t, filepath.Join(f.app, "local-qualification.json"), map[string]any{"version": 2, "qualifications": []any{map[string]any{
		"id": "ne-synthetic", "revision": "1", "engine": "llamacpp", "model": neModel, "modelDigests": []string{firstEnvelope.ContentDigest}, "launchDigests": []string{firstLaunch},
		"tokenizer": "synthetic-ne-tokenizer", "chatTemplate": "synthetic-ne-template", "contextTokens": neContext, "maxOutputTokens": neOutput, "routes": []string{"/v1/chat/completions"},
		"shapes":     map[string]any{"tools": false, "structuredOutput": []string{}, "reasoningReplay": false},
		"accounting": map[string]any{"method": "utf8-bytes", "requestOverheadTokens": 8, "messageOverheadTokens": 4, "toolOverheadTokens": 0, "schemaOverheadTokens": 0},
		"evidence":   "Synthetic NE process wiring data only; no model, hardware, or native engine qualification was performed.",
	}}})
	f.start(t)
	f.call(t, "engine:start", map[string]string{"engine": "llamacpp"}, nil)
	f.retainRuntime(t)
	secondSnapshot := f.snapshot(t)
	secondEnvelope := neAssertColdReadback(t, secondSnapshot, true)
	secondRun, secondLaunch := f.identity(t)
	starts := f.starts(t)
	if len(starts) != 2 || starts[0].Preset == starts[1].Preset || filepath.Dir(starts[0].Preset) == filepath.Dir(starts[1].Preset) || firstRun == secondRun || firstLaunch != secondLaunch || firstEnvelope.ContentDigest != secondEnvelope.ContentDigest || !reflect.DeepEqual(firstEnvelope, secondEnvelope) || !secondSnapshot.Settings.Equal(cold.Settings) {
		t.Fatal("restart changed qualification identity or reused its snapshot generation/run")
	}
	if _, err := neReadPreset(starts[1].Preset, modelPath); err != nil {
		t.Fatal(err)
	}
	var capability *nodepolicy.StrictAdmissionModel
	neEventually(t, "real broker joined strict capability", neCallBudget, func() bool { capability = neCapability(f.models(t)); return capability != nil })
	if err := capability.Validate(); err != nil {
		t.Fatal(err)
	}
	if capability.MaxConcurrent != 1 || capability.Identity.EngineRunID != secondRun || capability.Identity.ModelDigest != secondEnvelope.ContentDigest || capability.Identity.SettingsFingerprint == "" {
		t.Fatal("broker capability did not join the current manager identity with applied E settings")
	}
	state, err = f.state()
	if err != nil || state.Loaded || state.Chats != 0 || state.Loads != 0 || state.ModelProps != 0 || state.Unexpected != 0 {
		t.Fatal("cold qualification publication caused a load or inference")
	}
	// Existing strict admission requires reported residency, not NE Observed.
	f.call(t, "engine:load-model", map[string]string{"engine": "llamacpp", "model": neModel}, nil)
	neEventually(t, "explicit fake model residency", neCallBudget, func() bool { return reflect.DeepEqual(f.models(t).Loaded["llamacpp"], []string{neModel}) })
	strict := nodepolicy.StrictAdmissionRequest{Version: 1, QueueBudgetMs: 500, ColdStartBudgetMs: 0, PreOutputBudgetMs: 5000,
		Request: nodepolicy.RoutingRequestKey{OriginNodeID: secondSnapshot.NodeID, ProxyRunID: "ne-fixture-origin", RequestID: "1"}, Identity: capability.Identity}
	if err := strict.Validate(); err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"model":"ne-fixture","messages":[{"role":"user","content":"synthetic"}],"max_tokens":128,"stream":false}`)
	var status int
	var data []byte
	requestID := int64(0)
	// The manager's residency response precedes its asynchronous broker/proxy
	// projection. A zero-cold-budget refusal is safe to retry only if the fake
	// confirms that no inference or additional load was dispatched.
	neEventually(t, "warm strict destination Chat", neCallBudget, func() bool {
		requestID++
		strict.Request.RequestID = strconv.FormatInt(requestID, 10)
		status, data, err = f.http(http.MethodPost, f.facadePort, "/v1/chat/completions", body, &strict)
		if err != nil {
			t.Fatalf("qualified strict Chat transport: %v", err)
		}
		if status == http.StatusOK {
			return true
		}
		if status != http.StatusServiceUnavailable && status != http.StatusGatewayTimeout {
			t.Fatalf("qualified strict Chat: status=%d", status)
		}
		pending, err := f.state()
		if err != nil || pending.Chats != 0 || pending.Loads != 1 || pending.Unloads != 0 || pending.Unexpected != 0 {
			t.Fatal("pre-residency refusal caused destination effects")
		}
		return false
	})
	var chat struct {
		ID     string `json:"id"`
		Object string `json:"object"`
		Model  string `json:"model"`
	}
	if err := json.Unmarshal(data, &chat); err != nil || chat.ID != "ne-chat" || chat.Object != "chat.completion" || chat.Model != neModel {
		t.Fatal("strict Chat did not return the fake destination response")
	}
	state, err = f.state()
	if err != nil || state.Chats != 1 || state.Loads != 1 || state.Unloads != 0 || state.LeakedHeaders != 0 || state.Unexpected != 0 {
		t.Fatal("strict Chat had unexpected destination effects or leaked its strict header")
	}
	if err := os.WriteFile(f.source, []byte(source+"\n# controlled source withdrawal\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	// Full inventory owns drift validation at its existing 60-second cadence.
	neEventually(t, "source mismatch and strict withdrawal", 90*time.Second, func() bool {
		models := f.models(t)
		current := f.snapshot(t)
		return models.Policy != nil && neCapability(models) == nil && current.Serving != nil && current.Serving.Status == "mismatch"
	})
	requestID++
	strict.Request.RequestID = strconv.FormatInt(requestID, 10)
	status, _, err = f.http(http.MethodPost, f.facadePort, "/v1/chat/completions", body, &strict)
	if err != nil || status != http.StatusServiceUnavailable {
		t.Fatalf("withdrawn strict Chat: status=%d err=%v", status, err)
	}
	after, err := f.state()
	if err != nil || after.Chats != state.Chats || after.Loads != state.Loads || after.Unloads != state.Unloads || after.Unexpected != 0 || after.LeakedHeaders != 0 {
		t.Fatal("withdrawn identity reached inference or load effects")
	}
	t.Logf("ne_serving_proof generated_launch=true cold_readback=true broker_join=true exact_qualification_reuse=true restart_run_changed=true source_withdrawal=true strict_chats=%d loads=%d", after.Chats, after.Loads)
	f.stop(t)
}
