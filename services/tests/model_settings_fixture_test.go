// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/testbin"
)

func init() {
	if image := os.Getenv("NVPAIR_TEST_MODEL_LMS_IMAGE"); image != "" && image == os.Args[0] {
		if len(os.Args) != 3 || os.Args[1] != "ls" || os.Args[2] != "--json" {
			os.Exit(2)
		}
		data, err := os.ReadFile(os.Getenv("NVPAIR_TEST_MODEL_LMS_ROWS"))
		if err != nil {
			os.Exit(2)
		}
		if _, err := os.Stdout.Write(data); err != nil {
			os.Exit(2)
		}
		os.Exit(0)
	}
	for _, role := range []struct {
		selector string
		run      func() error
	}{{"NVPAIR_TEST_MODEL_ENGINE", runModelSettingsEngine}, {"NVPAIR_TEST_MODEL_MANAGER", runModelSettingsManager}} {
		if image := os.Getenv(role.selector); image != "" && image == os.Args[0] {
			if err := role.run(); err != nil {
				// No frames, launch arguments, model content or inference bodies.
				fmt.Fprintln(os.Stderr, "model-settings fixture failed:", err)
				os.Exit(1)
			}
			os.Exit(0)
		}
	}
}

func modelSettingsControl(path string) (string, error) {
	base, err := url.Parse(os.Getenv("NVPAIR_TEST_MODEL_CONTROL"))
	if err != nil || base.Scheme != "http" || base.Hostname() != "127.0.0.1" {
		return "", errors.New("controller must be private loopback HTTP")
	}
	port, err := strconv.Atoi(base.Port())
	if err != nil || validateFixturePort(port) != nil {
		return "", errors.New("controller must use an allocated fixture port")
	}
	client := &http.Client{Timeout: 2 * time.Minute}
	response, err := client.Get(os.Getenv("NVPAIR_TEST_MODEL_CONTROL") + path)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, 128))
	if err != nil {
		return "", err
	}
	if response.StatusCode != http.StatusOK {
		return "", fmt.Errorf("private controller HTTP %d", response.StatusCode)
	}
	return string(data), nil
}

type modelSettingsManagerInvocation struct {
	Method string
	Engine string
	Action string
}

func (call modelSettingsManagerInvocation) hasEffects() bool {
	switch call.Method {
	case "engine:action", "engine:load-model", "engine:unload-model", "engine:delete-model",
		"engine:start", "engine:stop", "engine:restart", "engine:install", "engine:uninstall",
		"engine:configure-launch", nodepolicy.MethodActivateBoundEngine,
		"engine:copy-model", "engine:remote-copy-model", "engine:remote-pull-model",
		"engine:remote-load-model", "engine:remote-unload-model", "engine:remote-delete-model",
		"engine:remote-start", "engine:remote-stop", "engine:remote-install":
		return true
	}
	return false
}

// The manager bridge runs the real component unchanged. Only transport delivery
// is gated/faulted: it never supplies a settings, policy or recovery result.
func runModelSettingsManager() (resultErr error) {
	// The model-list LAN surface has no bind-host seam and is unused here:
	// these cases inspect over IPC or pinned ec. Disable it through its public
	// flag rather than opening the product's wildcard listener in this fixture.
	args := slices.Clone(os.Args[1:])
	foundHTTP := false
	for i, arg := range args {
		if arg == "--http-port" && i+1 < len(args) {
			args[i+1] = "0"
			foundHTTP = true
			break
		}
	}
	if !foundHTTP {
		return errors.New("private manager launch lacks its explicit HTTP-port argument")
	}
	cmd := exec.Command(os.Getenv("NVPAIR_TEST_MODEL_REAL_MANAGER"), args...)
	cmd.Env, cmd.Stderr = os.Environ(), os.Stderr
	in, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	reaped := false
	defer func() {
		if !reaped {
			closeErr := in.Close()
			killErr := cmd.Process.Kill()
			if errors.Is(killErr, os.ErrProcessDone) {
				killErr = nil
			}
			resultErr = errors.Join(resultErr, closeErr, killErr, cmd.Wait())
		}
	}()
	wait := func() error { err := cmd.Wait(); reaped = true; return err }
	var mu sync.Mutex
	type pendingRequest struct {
		call     modelSettingsManagerInvocation
		readback bool
	}
	requests := map[string]pendingRequest{}
	readbackArmed := map[string]bool{}
	write := json.NewEncoder(in)
	var writeMu sync.Mutex
	writeFrame := func(frame *jsonrpc.Message) error { writeMu.Lock(); defer writeMu.Unlock(); return write.Encode(frame) }
	forward := make(chan error, 1)
	go func() {
		reader := jsonrpc.NewCodecMaxFrame(struct {
			io.Reader
			io.Writer
		}{os.Stdin, io.Discard}, jsonrpc.WorkerFrameBytes)
		for {
			frame, err := reader.Read()
			if err != nil {
				forward <- err
				return
			}
			call := modelSettingsManagerInvocation{Method: frame.Method}
			if (call.Method == "engine:get-launch" || call.hasEffects()) && len(frame.Params) != 0 && !bytes.Equal(frame.Params, []byte("null")) {
				var params struct {
					Engine string `json:"engine"`
					Action string `json:"action"`
				}
				if err := json.Unmarshal(frame.Params, &params); err != nil {
					forward <- err
					return
				}
				call.Engine, call.Action = params.Engine, params.Action
			}
			if call.hasEffects() {
				// Observe attempts before the real manager can refuse an Off
				// load/pull. Every request and result is still forwarded.
				query := url.Values{"method": {call.Method}, "engine": {call.Engine}, "action": {call.Action}}
				if _, err := modelSettingsControl("/manager-invocation?" + query.Encode()); err != nil {
					forward <- err
					return
				}
			}
			if frame.ID != nil && frame.Method != "" {
				mu.Lock()
				request := pendingRequest{call: call}
				if call.Method == "engine:get-launch" && readbackArmed[call.Engine] {
					request.readback = true
					delete(readbackArmed, call.Engine)
				}
				requests[string(*frame.ID)] = request
				mu.Unlock()
			}
			if call.Method == "engine:configure-launch" && call.Engine == "llamacpp" {
				// Keep independent policy replies and availability projection
				// flowing while this one native mutation is held at delivery.
				request := frame
				go func() {
					if _, err := modelSettingsControl("/configure-enter"); err != nil {
						forward <- err
						return
					}
					if err := writeFrame(request); err != nil {
						forward <- err
					}
				}()
				continue
			}
			if err := writeFrame(frame); err != nil {
				forward <- err
				return
			}
		}
	}()
	readDone := make(chan error, 1)
	go func() {
		reader := jsonrpc.NewCodecMaxFrame(struct {
			io.Reader
			io.Writer
		}{out, io.Discard}, jsonrpc.WorkerFrameBytes)
		encoder := json.NewEncoder(os.Stdout)
		for {
			frame, err := reader.Read()
			if err != nil {
				readDone <- err
				return
			}
			var request pendingRequest
			if frame.ID != nil && frame.Method == "" {
				mu.Lock()
				request = requests[string(*frame.ID)]
				delete(requests, string(*frame.ID))
				mu.Unlock()
			}
			configureReply := request.call.Method == "engine:configure-launch" && request.call.Engine == "llamacpp"
			if configureReply || request.readback {
				path := "/readback"
				if configureReply {
					path = "/configure-reply"
				}
				action, err := modelSettingsControl(path)
				if err != nil {
					readDone <- err
					return
				}
				if action == "error" {
					frame.Result = nil
					frame.Error = &jsonrpc.RPCError{Code: -32000, Message: "private readback transport unavailable"}
				}
			}
			if configureReply {
				// Only a new request for this affected engine, issued after its
				// configure reply, can consume the post-configure readback fault.
				// Already pending requests and other engines cannot take it.
				mu.Lock()
				readbackArmed[request.call.Engine] = true
				mu.Unlock()
			}
			if err := encoder.Encode(frame); err != nil {
				readDone <- err
				return
			}
		}
	}()
	select {
	case err := <-forward:
		if err != nil && !errors.Is(err, io.EOF) {
			return err
		}
		if err := in.Close(); err != nil {
			return err
		}
		return wait()
	case err := <-readDone:
		if !errors.Is(err, io.EOF) {
			return err
		}
		return wait()
	}
}

// A synthetic router consumes the actual generated preset but never maps model
// bytes or starts native inference. All model/load/download POSTs are counted.
func runModelSettingsEngine() error {
	image, err := os.Executable()
	if err != nil {
		return err
	}
	actual, err := serviceExecutablePath(image)
	if err != nil {
		return err
	}
	expected, err := serviceExecutablePath(os.Getenv("NVPAIR_TEST_MODEL_ENGINE"))
	if err != nil || actual != expected {
		return errors.New("synthetic router image differs from its fixed test image")
	}
	port, err := strconv.Atoi(os.Getenv("NVPAIR_TEST_MODEL_PORT"))
	if err != nil || validateFixturePort(port) != nil {
		return errors.New("synthetic router requires its private fixture port")
	}
	if _, err := modelSettingsControl("/engine-started?pid=" + strconv.Itoa(os.Getpid())); err != nil {
		return err
	}
	index := slices.Index(os.Args, "--models-preset")
	if index < 0 || index+1 == len(os.Args) {
		return errors.New("missing generated preset")
	}
	preset, err := os.ReadFile(os.Args[index+1])
	if err != nil {
		return errors.New("cannot read private preset")
	}
	if err := os.WriteFile(os.Getenv("NVPAIR_TEST_MODEL_CAPTURE"), preset, 0o600); err != nil {
		return err
	}
	listener, err := net.Listen("tcp", "127.0.0.1:"+os.Getenv("NVPAIR_TEST_MODEL_PORT"))
	if err != nil {
		return err
	}
	return http.Serve(listener, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method != http.MethodGet {
			path := "/mutation"
			if r.URL.Path == "/v1/chat/completions" {
				path = "/inference"
			}
			if _, err := modelSettingsControl(path); err != nil {
				http.Error(w, "private request interrupted", http.StatusServiceUnavailable)
				return
			}
			if path == "/inference" {
				fmt.Fprintln(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"fixture"},"finish_reason":"stop"}]}`)
				return
			}
			http.Error(w, "no synthetic content mutation is supported", http.StatusBadRequest)
			return
		}
		switch r.URL.Path {
		case "/props":
			fmt.Fprintln(w, `{"role":"router"}`)
		case "/models", "/v1/models":
			fmt.Fprintln(w, `{"data":[{"id":"medium-model","status":{"value":"unloaded"}},{"id":"weak-model","status":{"value":"unloaded"}}]}`)
		default:
			fmt.Fprintln(w, `{"status":"ok"}`)
		}
	}))
}

type modelSettingsGate struct {
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func newModelSettingsGate() *modelSettingsGate {
	return &modelSettingsGate{entered: make(chan struct{}, 1), release: make(chan struct{})}
}
func (g *modelSettingsGate) unblock() { g.once.Do(func() { close(g.release) }) }

type modelSettingsProcessFixture struct {
	*switchProcessFixture
	root, preset, capture, manager string
	control                        string
	mu                             sync.Mutex
	configureGate                  *modelSettingsGate
	readbackGate                   *modelSettingsGate
	configureCalls                 int
	managerInvocations             map[modelSettingsManagerInvocation]int
	failReadback                   bool
	failConfigureReply             bool
	mutations, inferences          int
	inference                      *modelSettingsGate
	client                         *http.Client
	callerCluster                  string
	replies                        map[string]jsonrpc.Message
	image                          string
	proxyProbeImage                string
	proxyProbePort                 int
}

func isolateModelSettingsPresetEnv(t *testing.T) {
	t.Helper()
	// Empty has presence semantics in the native option parser. Restore the
	// inherited value at cleanup, but remove it before any artifact read.
	t.Setenv("LLAMA_ARG_MODELS_PRESET", "")
	if err := os.Unsetenv("LLAMA_ARG_MODELS_PRESET"); err != nil {
		t.Fatal(err)
	}
}

type modelSettingsRuntimeCheck struct {
	Bin  string            `json:"bin"`
	Bind string            `json:"bind"`
	Port int               `json:"port"`
	Env  map[string]string `json:"env"`
}

// Inspect the private manifest inputs before any real worker can detect or
// launch an engine. Each enabled image is this exact fixed synthetic executable;
// every other engine is absent in this fixture's private root.
func (f *modelSettingsProcessFixture) assertEngineManifests(t *testing.T) {
	t.Helper()
	if err := f.ports.validate(); err != nil {
		t.Fatal(err)
	}
	seen := map[int]bool{}
	ports := []int{f.ports.nodeInfo, f.ports.errors, f.ports.workload, f.ports.clusterManager, f.ports.engineManagerHTTP, f.ports.engineControl, f.sourcePort, f.targetPort, f.facadePort, f.targetFacadePort, f.scannerPort, portOfURL(t, f.control), portOfURL(t, f.switchProcessFixture.control.URL)}
	if f.proxyProbeImage != "" {
		ports = append(ports, f.proxyProbePort)
	}
	for _, port := range ports {
		if err := validateFixturePort(port); err != nil {
			t.Fatal(err)
		}
		if seen[port] {
			t.Fatal("model settings fixture repeats a private port")
		}
		seen[port] = true
	}
	for _, name := range engines.Names() {
		var manifest struct {
			ModelsDir string                    `json:"models_dir"`
			Runtime   modelSettingsRuntimeCheck `json:"runtime"`
			Platforms map[string]struct {
				Detect    []string                  `json:"detect"`
				ModelsDir string                    `json:"models_dir"`
				Runtime   modelSettingsRuntimeCheck `json:"runtime"`
			} `json:"platforms"`
		}
		modelSettingsDecode(t, readLiveDurableFile(t, filepath.Join(jointAppDir(f.base), "engines", name+".json")), &manifest)
		platform, ok := manifest.Platforms[runtime.GOOS+"/"+runtime.GOARCH]
		if !ok {
			t.Fatal("private engine manifest lacks its active platform")
		}
		expectedImage := filepath.Join(f.base, "absent-"+name)
		expectedRoot := filepath.Join(f.base, "absent-models", name)
		active := name == "llamacpp" || name == "ollama"
		if active {
			expectedImage = f.image
		}
		if name == "llamacpp" {
			expectedRoot = f.root
		}
		if len(platform.Detect) != 1 {
			t.Fatal("private engine detection has an unexpected alternative")
		}
		for _, path := range []string{manifest.Runtime.Bin, platform.Runtime.Bin, platform.Detect[0]} {
			if !modelSettingsSamePath(path, expectedImage) {
				t.Fatalf("engine %s can select an image outside its expected fixed fake", name)
			}
		}
		if !modelSettingsSamePath(manifest.ModelsDir, expectedRoot) || !modelSettingsSamePath(platform.ModelsDir, expectedRoot) {
			t.Fatalf("engine %s can read a non-fixture model root", name)
		}
		if !active {
			continue
		}
		expectedPort := f.targetPort
		selector := "NVPAIR_TEST_MODEL_ENGINE"
		if name == "ollama" {
			expectedPort = f.sourcePort
			selector = "NVPAIR_TEST_SWITCH_ENGINE"
		}
		for _, rt := range []modelSettingsRuntimeCheck{manifest.Runtime, platform.Runtime} {
			if rt.Bind != "127.0.0.1" || rt.Port != expectedPort || !modelSettingsSamePath(rt.Env[selector], f.image) {
				t.Fatalf("engine %s lacks its fixed fake selector/private loopback bind", name)
			}
			if name == "llamacpp" && !modelSettingsSamePath(rt.Env["LLAMA_CACHE"], f.root) {
				t.Fatal("synthetic router can inherit an external model cache")
			}
		}
	}
}

func modelSettingsSamePath(actual, expected string) bool {
	a, err := serviceExecutablePath(actual)
	if err != nil {
		return false
	}
	b, err := serviceExecutablePath(expected)
	return err == nil && a == b
}

func (f *modelSettingsProcessFixture) assertEngineImages(t *testing.T) {
	t.Helper()
	f.switchProcessFixture.mu.Lock()
	defer f.switchProcessFixture.mu.Unlock()
	for engine, identities := range f.engineIdentities {
		for _, identity := range identities {
			if (engine != "llamacpp" && engine != "ollama") || !modelSettingsSamePath(identity.Path, f.image) || identity.PID == int32(os.Getpid()) {
				t.Errorf("launched engine %s is not the expected fixed fake child", engine)
			}
		}
	}
}

func (f *modelSettingsProcessFixture) recordEngineStart(t *testing.T, w http.ResponseWriter, r *http.Request) {
	t.Helper()
	pid, err := strconv.ParseInt(r.URL.Query().Get("pid"), 10, 32)
	if err != nil || pid <= 0 || pid == int64(os.Getpid()) {
		t.Error("invalid synthetic engine child identity")
		http.Error(w, "invalid child identity", http.StatusForbidden)
		return
	}
	identity, err := serviceProcessIdentityAt(r.Context(), int32(pid))
	if err != nil || !modelSettingsSamePath(identity.Path, f.image) {
		t.Error("synthetic engine launched an image other than the expected fixed fake")
		http.Error(w, "unexpected engine image", http.StatusForbidden)
		return
	}
	request, err := http.NewRequestWithContext(r.Context(), http.MethodGet, f.switchProcessFixture.control.URL+"/started?engine=llamacpp&pid="+strconv.FormatInt(pid, 10), nil)
	if err != nil {
		t.Error(err)
		http.Error(w, "invalid private controller request", http.StatusInternalServerError)
		return
	}
	response, err := (&http.Client{Timeout: 2 * time.Minute}).Do(request)
	if err != nil {
		http.Error(w, "private lifetime controller unavailable", http.StatusServiceUnavailable)
		return
	}
	_, readErr := io.Copy(io.Discard, response.Body)
	closeErr := response.Body.Close()
	if readErr != nil || closeErr != nil {
		t.Error(errors.Join(readErr, closeErr))
		http.Error(w, "private controller response failed", http.StatusBadGateway)
		return
	}
	w.WriteHeader(response.StatusCode)
}

func prepareModelSettingsProcessFixture(t *testing.T) *modelSettingsProcessFixture {
	t.Helper()
	image, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS == "windows" {
		root := os.Getenv(testbin.EnvRoot)
		if !filepath.IsAbs(root) || !strings.Contains(os.Getenv("GOFLAGS"), "nvpair-test-exec") {
			t.Fatal("model settings process tests require the registered fixed-path Windows wrapper")
		}
		actual, pathErr := serviceExecutablePath(image)
		expected, expectedErr := serviceExecutablePath(filepath.Join(root, "go-test", "tests", "tests.test.exe"))
		if pathErr != nil || expectedErr != nil || actual != expected {
			t.Fatal("model settings synthetic engine must use the registered services/tests package image")
		}
	}
	isolateModelSettingsPresetEnv(t)
	base := t.TempDir()
	f := &modelSettingsProcessFixture{switchProcessFixture: prepareSwitchProcessFixture(t, base, reserveFixtureServicePorts(t), ""), managerInvocations: map[modelSettingsManagerInvocation]int{}}
	f.image = image
	t.Cleanup(func() { f.assertEngineImages(t) })
	f.root, f.preset, f.capture = filepath.Join(base, "models"), filepath.Join(base, "models", "external.ini"), filepath.Join(base, "capture.ini")
	modelSettingsWrite(t, filepath.Join(f.root, "medium.gguf"), modelSettingsGGUF("blk.0.weight"))
	modelSettingsWrite(t, filepath.Join(f.root, "weak.gguf"), modelSettingsGGUF("blk.1.weight"))
	modelSettingsWrite(t, f.preset, []byte("[*]\nfit = off\nload-on-startup = false\n[medium-model]\nmodel = medium.gguf\nctx-size = 4096\nparallel = 1\ngpu-layers = 0\n[weak-model]\nmodel = weak.gguf\nctx-size = 4096\nparallel = 1\ngpu-layers = 0\n"))
	server := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/engine-started" {
			f.recordEngineStart(t, w, r)
			return
		}
		f.mu.Lock()
		var gate *modelSettingsGate
		action := "forward"
		switch r.URL.Path {
		case "/manager-invocation":
			query := r.URL.Query()
			f.managerInvocations[modelSettingsManagerInvocation{Method: query.Get("method"), Engine: query.Get("engine"), Action: query.Get("action")}]++
		case "/configure-enter":
			f.configureCalls++
			gate = f.configureGate
		case "/configure-reply":
			if f.failConfigureReply {
				action, f.failConfigureReply = "error", false
			}
		case "/readback":
			gate = f.readbackGate
			if f.failReadback {
				action, f.failReadback = "error", false
			}
		case "/mutation":
			f.mutations++
		case "/inference":
			f.inferences++
			gate = f.inference
		default:
			http.Error(w, "unknown controller route", http.StatusNotFound)
			f.mu.Unlock()
			return
		}
		f.mu.Unlock()
		if gate != nil {
			select {
			case gate.entered <- struct{}{}:
			default:
			}
			select {
			case <-gate.release:
			case <-r.Context().Done():
				return
			}
		}
		fmt.Fprint(w, action)
	}))
	f.control = server.URL
	t.Cleanup(func() {
		f.mu.Lock()
		for _, gate := range []*modelSettingsGate{f.configureGate, f.readbackGate, f.inference} {
			if gate != nil {
				gate.unblock()
			}
		}
		f.mu.Unlock()
		server.Close()
	})
	f.manager = filepath.Join(testbin.Dir(t, "tests", "model-settings", "model-settings-manager"), testbin.Exe("model-settings-manager"))
	jointCopyExecutable(t, image, f.manager)
	probe := map[string]any{"http": "http://127.0.0.1:{port}/props", "status": http.StatusOK, "timeout_s": 10, "interval_s": 60, "json_match": map[string]string{"field": "role", "value": "router"}}
	rt := map[string]any{"mode": "process", "bin": image, "port": f.targetPort, "bind": "127.0.0.1", "ready": probe, "health": probe,
		"args": []string{"--models-preset", f.preset}, "editable_launch": map[string]any{"fixed_args": []string{}},
		"env":  map[string]string{"LLAMA_CACHE": f.root, "NVPAIR_TEST_MODEL_ENGINE": image, "NVPAIR_TEST_MODEL_PORT": "{port}", "NVPAIR_TEST_MODEL_CONTROL": f.control, "NVPAIR_TEST_MODEL_CAPTURE": f.capture},
		"stop": map[string]any{"signal": "term", "grace_s": 1}}
	// prepareSwitchProcessFixture replaces its own manifests. Restore BOTH
	// model roots here, before start, rather than relying on HOME isolation.
	jointWriteJSON(t, filepath.Join(jointAppDir(base), "engines", "llamacpp.json"), map[string]any{
		"engine": "llamacpp", "models_dir": f.root, "runtime": rt,
		"platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{image}, "models_dir": f.root, "runtime": rt}},
	})
	var other map[string]any
	otherPath := filepath.Join(jointAppDir(base), "engines", "ollama.json")
	modelSettingsDecode(t, readLiveDurableFile(t, otherPath), &other)
	other["models_dir"] = filepath.Join(base, "absent-models", "ollama")
	platforms, ok := other["platforms"].(map[string]any)
	if !ok {
		t.Fatal("invalid private platform override")
	}
	platform, ok := platforms[runtime.GOOS+"/"+runtime.GOARCH].(map[string]any)
	if !ok {
		t.Fatal("missing private platform override")
	}
	platform["models_dir"] = other["models_dir"]
	jointWriteJSON(t, otherPath, other)
	policy := nodepolicy.Default()
	policy.Idle.UnloadAfterMinutes, policy.Idle.StopEngineAfterMinutes = 0, 0
	policy.Profiles = []nodepolicy.Profile{{Name: "medium", Engine: "llamacpp", Model: "medium-model", RequestOptions: json.RawMessage(`{"temperature":0.2,"top_p":0.9,"opaque":{"integer":9007199254740993}}`)}, {Name: "weak", Engine: "llamacpp", Model: "weak-model", RequestOptions: json.RawMessage(`{"temperature":0.4}`)}}
	policy.Roles = map[string]nodepolicy.Role{"weak": {Engine: "llamacpp", Model: "weak-model"}}
	jointWriteJSON(t, filepath.Join(jointAppDir(base), nodepolicy.FileName), policy)
	return f
}

func newModelSettingsProcessFixture(t *testing.T) *modelSettingsProcessFixture {
	t.Helper()
	f := prepareModelSettingsProcessFixture(t)
	f.startModelSettings(t)
	return f
}

func (f *modelSettingsProcessFixture) enableAdmissionProbe(t *testing.T) {
	t.Helper()
	if f.p != nil || f.proxyProbeImage != "" {
		t.Fatal("admission probe must be enabled once before broker startup")
	}
	f.proxyProbeImage = filepath.Join(testbin.Dir(t, "tests", "performance-observations", "observation-proxy"), testbin.Exe("observation-proxy"))
	jointCopyExecutable(t, f.image, f.proxyProbeImage)
	f.proxyProbePort = freePort(t)
}

func (f *modelSettingsProcessFixture) assertHeldAdmission(t *testing.T) {
	t.Helper()
	if f.proxyProbeImage == "" {
		t.Fatal("admission probe is not enabled")
	}
	client := &http.Client{Timeout: 15 * time.Second, Transport: &http.Transport{Proxy: nil}}
	defer client.CloseIdleConnections()
	var reply jsonrpc.Message
	observationWorkerHTTP(t, client, fmt.Sprintf("http://127.0.0.1:%d", f.proxyProbePort), "/control", jsonrpc.Message{
		JSONRPC: "2.0", Method: nodepolicy.MethodAdmissionSnapshot, Params: mustMarshal(t, nodepolicy.AdmissionSnapshotParams{}),
	}, &reply)
	if reply.Error != nil {
		t.Fatalf("held admission snapshot RPC code=%d", reply.Error.Code)
	}
	var snapshot nodepolicy.AdmissionSnapshotResult
	modelSettingsDecode(t, reply.Result, &snapshot)
	if snapshot.RunID == "" || snapshot.ActiveByEngine == nil {
		t.Fatal("held admission snapshot lacks proxy run or engine counts")
	}
	t.Logf("held_admission runId=%s active=%d activeByEngine.llamacpp=%d queued=%d", snapshot.RunID, snapshot.Active, snapshot.ActiveByEngine["llamacpp"], snapshot.Queued)
	if snapshot.Active != 1 || snapshot.ActiveByEngine["llamacpp"] != 1 || snapshot.Queued != 0 {
		t.Fatal("held synthetic request is not the single admitted llama.cpp request")
	}
}

// Reuse a private store populated by the real content-copy owner. This runs
// before launch and changes only this fixture's manifest source, never a live
// manager, installed store, generated preset or receipt.
func (f *modelSettingsProcessFixture) useCopiedStore(t *testing.T, root string) {
	t.Helper()
	path := filepath.Join(jointAppDir(f.base), "engines", "llamacpp.json")
	var manifest map[string]json.RawMessage
	modelSettingsDecode(t, readLiveDurableFile(t, path), &manifest)
	var platforms map[string]map[string]json.RawMessage
	modelSettingsDecode(t, manifest["platforms"], &platforms)
	var rt map[string]json.RawMessage
	modelSettingsDecode(t, manifest["runtime"], &rt)
	var env map[string]string
	modelSettingsDecode(t, rt["env"], &env)
	env["LLAMA_CACHE"] = root
	rt["env"] = mustMarshal(t, env)
	manifest["runtime"] = mustMarshal(t, rt)
	manifest["models_dir"] = mustMarshal(t, root)
	platforms[runtime.GOOS+"/"+runtime.GOARCH]["runtime"] = manifest["runtime"]
	platforms[runtime.GOOS+"/"+runtime.GOARCH]["models_dir"] = manifest["models_dir"]
	manifest["platforms"] = mustMarshal(t, platforms)
	jointWriteJSON(t, path, manifest)
	f.root = root
}

func (f *modelSettingsProcessFixture) startModelSettings(t *testing.T) {
	t.Helper()
	f.assertEngineManifests(t)
	proxyPath := proxyBin
	env := []string{
		"NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + f.scanner, "NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(f.scannerPort),
		"NVPAIR_TEST_MODEL_MANAGER=" + f.manager, "NVPAIR_TEST_MODEL_REAL_MANAGER=" + engineMgrBin, "NVPAIR_TEST_MODEL_CONTROL=" + f.control,
	}
	if f.proxyProbeImage != "" {
		proxyPath = f.proxyProbeImage
		env = append(env, "NVPAIR_OBSERVATION_PROXY_FIXTURE="+f.proxyProbeImage, "NVPAIR_OBSERVATION_REAL_PROXY="+proxyBin, "NVPAIR_OBSERVATION_PROXY_CONTROL="+strconv.Itoa(f.proxyProbePort))
	}
	args := append([]string{"--scanner-path", f.scanner, "--engine-manager-path", f.manager, "--engine-manager-control-host", "127.0.0.1", "--proxy-path", proxyPath, "--proxy-engines", "ollama,llamacpp", "--cluster-dir", filepath.Join(f.base, "cluster")}, f.ports.args()...)
	f.p = startInteropProcess(t, f.base, brokerBin, f.ports.engineManagerHTTP, env, args...)
	// Registered after process cleanup so a failed assertion cannot leave its
	// own controller barrier blocking EOF and force-kill cleanup.
	t.Cleanup(f.unblockModelSettings)
	for _, path := range []string{f.manager, engineMgrBin, proxyBin, f.scanner} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		f.p.paths = append(f.p.paths, canonical)
	}
	f.p.ports = append(f.p.ports, f.scannerPort, f.sourcePort, f.targetPort, f.facadePort, f.targetFacadePort, f.ports.engineControl)
	if f.proxyProbeImage != "" {
		canonical, err := canonicalServicePath(f.proxyProbeImage)
		if err != nil {
			t.Fatal(err)
		}
		f.p.paths = append(f.p.paths, canonical)
		f.p.ports = append(f.p.ports, f.proxyProbePort)
	}
	waitForMethod(t, f.p.msgs, "app:ready", 30*time.Second)
	waitProxyReady(t, f.p.in, f.p.msgs, 20*time.Second)
	waitEngineProxyReady(t, "llamacpp-proxy", f.p.in, f.p.msgs, 20*time.Second)
	for _, engine := range []string{"ollama", "llamacpp", "lmstudio"} {
		f.rpcOK(t, "engine:get-settings", map[string]string{"engine": engine})
	}
}

func (f *modelSettingsProcessFixture) unblockModelSettings() {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, gate := range []*modelSettingsGate{f.configureGate, f.readbackGate, f.inference} {
		if gate != nil {
			gate.unblock()
		}
	}
}

// Apply replies after its accepted operation runs. Retain that pending reply
// while exercising independent public owners at deterministic effect barriers.
func (f *modelSettingsProcessFixture) begin(t *testing.T, method string, params any) string {
	t.Helper()
	id := strconv.FormatInt(newRPCID(), 10)
	raw := json.RawMessage(id)
	sendLine(t, f.p.in, jsonrpc.Message{JSONRPC: "2.0", ID: &raw, Method: method, Params: mustMarshal(t, params)})
	return id
}

func (f *modelSettingsProcessFixture) await(t *testing.T, id string) jsonrpc.Message {
	t.Helper()
	if f.replies == nil {
		f.replies = map[string]jsonrpc.Message{}
	}
	if reply, ok := f.replies[id]; ok {
		delete(f.replies, id)
		return reply
	}
	timer := time.NewTimer(90 * time.Second)
	defer timer.Stop()
	for {
		select {
		case msg, ok := <-f.p.msgs:
			if !ok {
				t.Fatal("private broker exited before correlated reply")
			}
			if msg.ID != nil && msg.Method == "" {
				key := string(*msg.ID)
				if key == id {
					return msg
				}
				f.replies[key] = msg
			}
		case <-timer.C:
			t.Fatal("private broker did not return correlated reply")
		}
	}
}

func (f *modelSettingsProcessFixture) rpc(t *testing.T, method string, params any) jsonrpc.Message {
	t.Helper()
	response := f.await(t, f.begin(t, method, params))
	f.assertEngineImages(t)
	return response
}

func (f *modelSettingsProcessFixture) rpcOK(t *testing.T, method string, params any) json.RawMessage {
	t.Helper()
	response := f.rpc(t, method, params)
	if response.Error != nil {
		t.Fatalf("%s: code=%d message=%s", method, response.Error.Code, response.Error.Message)
	}
	return response.Result
}

func (f *modelSettingsProcessFixture) finishApply(t *testing.T, id string) {
	t.Helper()
	response := f.await(t, id)
	if response.Error != nil {
		t.Fatalf("accepted Apply failed: code=%d", response.Error.Code)
	}
	var receipt struct {
		Phase    string `json:"phase"`
		Revision uint64 `json:"revision"`
	}
	modelSettingsDecode(t, response.Result, &receipt)
	if receipt.Revision == 0 || (receipt.Phase != "succeeded" && receipt.Phase != "failed" && receipt.Phase != "applying") {
		t.Fatal("accepted Apply did not return an attempt receipt")
	}
}

func (f *modelSettingsProcessFixture) snapshot(t *testing.T) settings.Snapshot {
	t.Helper()
	var result settings.Snapshot
	modelSettingsDecode(t, f.rpcOK(t, "engine:get-settings", map[string]string{"engine": "llamacpp"}), &result)
	return result
}

func (f *modelSettingsProcessFixture) policy(t *testing.T) nodepolicy.Policy {
	t.Helper()
	var result struct {
		Policy nodepolicy.Policy `json:"policy"`
	}
	modelSettingsDecode(t, f.rpcOK(t, "policy:get", map[string]string{}), &result)
	return result.Policy
}

func (f *modelSettingsProcessFixture) request(t *testing.T, id string, patch *modelconfig.PolicyPatch) settings.Request {
	t.Helper()
	state, policy := f.snapshot(t), f.policy(t)
	return settings.Request{Engine: "llamacpp", RequestID: id, ExpectedRevision: state.Revision, Settings: state.Settings.Clone(),
		ModelChange: &modelconfig.ModelChange{ModelID: "medium-model", ExpectedPolicyRevision: policy.Revision,
			Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: 8192}, Slots: modelSettingsInt(2), GPU: &modelconfig.GPU{Mode: "cpu"}}}, Policy: patch}}
}

func (f *modelSettingsProcessFixture) review(t *testing.T, p settings.Request) (settings.Request, settings.Preview) {
	t.Helper()
	var preview settings.Preview
	modelSettingsDecode(t, f.rpcOK(t, "engine:preview-settings", p), &preview)
	if len(preview.Errors) != 0 || preview.Conflict != nil || preview.PreviewDigest == "" {
		t.Fatalf("preview refused: errors=%v conflict=%v", preview.Errors, preview.Conflict)
	}
	p.ModelChange.PreviewDigest = preview.PreviewDigest
	return p, preview
}

func (f *modelSettingsProcessFixture) settled(t *testing.T, id, phase string) settings.Snapshot {
	t.Helper()
	var state settings.Snapshot
	task1Eventually(t, "model settings receipt "+id, 90*time.Second, func() bool {
		state = f.snapshot(t)
		return state.RequestID == id && (state.Phase == "succeeded" || state.Phase == "failed" || state.UnresolvedError != "")
	})
	if state.Phase != phase {
		t.Fatalf("phase=%s stage=%s unresolved=%s, want %s", state.Phase, state.CoupledStage, state.UnresolvedError, phase)
	}
	return state
}

func (f *modelSettingsProcessFixture) setPolicy(t *testing.T, policy nodepolicy.Policy) nodepolicy.Policy {
	t.Helper()
	var result struct {
		Policy nodepolicy.Policy `json:"policy"`
	}
	modelSettingsDecode(t, f.rpcOK(t, "policy:set", map[string]any{"policy": policy, "expectedPersistedRevision": policy.Revision}), &result)
	return result.Policy
}

func (f *modelSettingsProcessFixture) seedModelSettingsPolicyBounds(t *testing.T) {
	t.Helper()
	policy := f.policy(t)
	profile := &policy.Profiles[0]
	profile.ContextTokens, profile.MaxOutputTokens, profile.MaxConcurrent = 4096, 512, 1
	profile.MemoryBytes = 1048576
	profile.RequestOptionsMode = nodepolicy.RequestOptionsEnforced
	// A saved operator reference, never a fabricated qualification registry
	// record or memory observation. These declarations must survive estimates.
	profile.Qualification = &nodepolicy.QualificationRef{ID: "saved-operator-reference", Revision: "7"}
	f.setPolicy(t, policy)
}

func (f *modelSettingsProcessFixture) pinned(t *testing.T) {
	t.Helper()
	cluster := filepath.Join(f.base, "cluster")
	f.callerCluster = t.TempDir()
	if err := os.MkdirAll(cluster, 0o700); err != nil {
		t.Fatal(err)
	}
	const target, caller = "model-settings-target", "model-settings-caller"
	targetCert, callerCert := mintClusterIdentity(t, cluster, target), mintClusterIdentity(t, f.callerCluster, caller)
	for _, entry := range []struct{ dir, id, peer, cert string }{{cluster, target, caller, callerCert}, {f.callerCluster, caller, target, targetCert}} {
		writeActiveAdmission(t, entry.dir)
		writePin(t, entry.dir, entry.peer, entry.cert)
		writeMembers(t, entry.dir, target, caller)
		jointWriteJSON(t, filepath.Join(entry.dir, "identity.json"), map[string]string{"node_uuid": entry.id})
	}
	managementGrantFixture(t, cluster, target, caller)
	mesh := clustertrust.Open(f.callerCluster)
	cfg, ok := mesh.ClientTLSConfig(target)
	if !ok {
		t.Fatal("missing private pinned client configuration")
	}
	f.client = &http.Client{Timeout: 90 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg, Proxy: nil}}
	t.Cleanup(f.client.CloseIdleConnections)
}

func (f *modelSettingsProcessFixture) peer(t *testing.T, method string, p any, status int) []byte {
	t.Helper()
	response, err := f.client.Post(fmt.Sprintf("https://127.0.0.1:%d/v1/engine-settings/%s", f.ports.engineControl, method), "application/json", bytes.NewReader(mustMarshal(t, p)))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, modelconfig.MaxPayloadBytes+1))
	if err != nil || len(data) > modelconfig.MaxPayloadBytes {
		t.Fatal("private peer reply exceeds bound or cannot be read")
	}
	if response.StatusCode != status {
		t.Fatalf("pinned %s HTTP %d, want %d", method, response.StatusCode, status)
	}
	return data
}

func modelSettingsInt(v int32) *int32           { return &v }
func modelSettingsNumber(v string) *json.Number { n := json.Number(v); return &n }
func modelSettingsDecode(t *testing.T, data []byte, out any) {
	t.Helper()
	if err := json.Unmarshal(data, out); err != nil {
		t.Fatal(err)
	}
}
func modelSettingsWrite(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
func modelSettingsWait(t *testing.T, g *modelSettingsGate) {
	t.Helper()
	select {
	case <-g.entered:
	case <-time.After(30 * time.Second):
		t.Fatal("private model settings barrier not reached")
	}
}

func modelSettingsGGUF(tensor string) []byte {
	data := []byte("GGUF")
	u32 := func(v uint32) { data = binary.LittleEndian.AppendUint32(data, v) }
	u64 := func(v uint64) { data = binary.LittleEndian.AppendUint64(data, v) }
	str := func(v string) { u64(uint64(len(v))); data = append(data, v...) }
	u32(3)
	u64(1)
	u64(3)
	str("general.architecture")
	u32(8)
	str("llama")
	str("llama.context_length")
	u32(4)
	u32(32768)
	str("llama.block_count")
	u32(4)
	u32(4)
	str(tensor)
	u32(1)
	u64(32)
	u32(0)
	u64(0)
	for len(data)%32 != 0 {
		data = append(data, 0)
	}
	return append(data, make([]byte, 128)...)
}

// Typed disk reads are only barriers/evidence. Mutations always use public IPC.
type modelSettingsDiskRecord struct {
	Snapshot settings.Snapshot `json:"snapshot"`
	Model    struct {
		Stage     string `json:"stage"`
		Attempted bool   `json:"attempted"`
	} `json:"model"`
	Receipts map[string]struct {
		Hash     string `json:"hash"`
		Revision uint64 `json:"revision"`
		Phase    string `json:"phase"`
	} `json:"receipts"`
}

func (f *modelSettingsProcessFixture) disk(t *testing.T) modelSettingsDiskRecord {
	t.Helper()
	var records map[string]modelSettingsDiskRecord
	modelSettingsDecode(t, readLiveDurableFile(t, filepath.Join(f.base, "engine-settings-operations.json")), &records)
	return records["llamacpp"]
}

type modelSettingsMetadataBaseline struct {
	snapshot           settings.Snapshot
	policy             nodepolicy.Policy
	journal            []byte
	managerInvocations map[modelSettingsManagerInvocation]int
	configureCalls     int
	mutations          int
	inferences         int
}

func (f *modelSettingsProcessFixture) metadataBaseline(t *testing.T) modelSettingsMetadataBaseline {
	t.Helper()
	before := modelSettingsMetadataBaseline{snapshot: f.snapshot(t), policy: f.policy(t),
		journal: readLiveDurableFile(t, filepath.Join(f.base, "engine-settings-operations.json"))}
	if before.snapshot.Running {
		t.Fatal("metadata baseline requires an Off engine")
	}
	f.mu.Lock()
	before.managerInvocations = maps.Clone(f.managerInvocations)
	before.configureCalls, before.mutations, before.inferences = f.configureCalls, f.mutations, f.inferences
	f.mu.Unlock()
	return before
}

func (f *modelSettingsProcessFixture) noMetadataEffects(t *testing.T, before modelSettingsMetadataBaseline) {
	t.Helper()
	after := f.snapshot(t)
	if !after.Settings.Equal(before.snapshot.Settings) || after.Running || after.Revision != before.snapshot.Revision || after.AppliedRevision != before.snapshot.AppliedRevision || after.RequestID != before.snapshot.RequestID || after.Phase != before.snapshot.Phase || after.CoupledStage != before.snapshot.CoupledStage || after.UnresolvedError != before.snapshot.UnresolvedError || !reflect.DeepEqual(after.EffectiveSettings, before.snapshot.EffectiveSettings) || !bytes.Equal(mustMarshal(t, f.policy(t)), mustMarshal(t, before.policy)) {
		t.Fatal("metadata read/preview changed settings authority, runtime or policy")
	}
	// Include every durable receipt and revision, even an otherwise invisible
	// failed attempt. Volatile readback sequence changes are not authority.
	if !bytes.Equal(before.journal, readLiveDurableFile(t, filepath.Join(f.base, "engine-settings-operations.json"))) {
		t.Fatal("metadata read/preview changed the persistent settings journal")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if !maps.Equal(before.managerInvocations, f.managerInvocations) || f.configureCalls != before.configureCalls || f.mutations != before.mutations || f.inferences != before.inferences {
		t.Fatalf("metadata effects: manager attempts=%v baseline=%v configure=%d mutation=%d inference=%d", f.managerInvocations, before.managerInvocations, f.configureCalls, f.mutations, f.inferences)
	}
	f.switchProcessFixture.mu.Lock()
	defer f.switchProcessFixture.mu.Unlock()
	for engine, starts := range f.starts {
		if len(starts) != 0 {
			t.Fatalf("metadata read/preview started synthetic engine %s", engine)
		}
	}
}

func (f *modelSettingsProcessFixture) engineIdentity(t *testing.T, starts int) serviceProcessIdentity {
	t.Helper()
	f.switchProcessFixture.mu.Lock()
	defer f.switchProcessFixture.mu.Unlock()
	identities := f.engineIdentities["llamacpp"]
	if len(identities) != starts || len(f.starts["llamacpp"]) != starts {
		t.Fatalf("synthetic router lifetimes=%d starts=%d, want %d", len(identities), len(f.starts["llamacpp"]), starts)
	}
	return identities[starts-1]
}

func (f *modelSettingsProcessFixture) stopBroker(t *testing.T) {
	t.Helper()
	f.p.cleanup()
	for _, port := range []int{f.targetPort, f.sourcePort} {
		assertInteropListenerClosed(t, port)
	}
}

func (f *modelSettingsProcessFixture) crashBroker(t *testing.T, gate *modelSettingsGate) {
	t.Helper()
	old := f.p
	old.stopOnce.Do(func() {
		if err := old.cmd.Process.Kill(); err != nil {
			t.Fatal(err)
		}
		if err := old.in.Close(); err != nil {
			t.Fatal(err)
		}
		gate.unblock()
		select {
		case err := <-old.done:
			if err == nil {
				t.Fatal("killed broker exited successfully")
			}
		case <-time.After(15 * time.Second):
			t.Fatal("killed broker was not reaped")
		}
	})
	inventory := &serviceProcessInventory{paths: old.paths, known: make(map[int32]serviceProcessIdentity)}
	task1Eventually(t, "crashed broker workers exit", 20*time.Second, func() bool {
		owned, err := inventory.scan()
		if err != nil {
			t.Fatal(err)
		}
		return len(owned) == 0
	})
	for _, port := range old.ports {
		assertInteropListenerClosed(t, port)
	}
}

func (f *modelSettingsProcessFixture) replaceManager(t *testing.T) {
	t.Helper()
	canonical, err := canonicalServicePath(engineMgrBin)
	if err != nil {
		t.Fatal(err)
	}
	inventory := &serviceProcessInventory{paths: []string{canonical}, known: make(map[int32]serviceProcessIdentity)}
	owned, err := inventory.scan()
	if err != nil || len(owned) != 1 {
		t.Fatalf("private real manager inventory count=%d error=%v", len(owned), err)
	}
	stopped, err := modelSettingsReplaceProcess(owned[0], func() error {
		return inventory.kill(owned[0])
	}, 10*time.Second)
	if err != nil || !stopped {
		t.Fatalf("retained manager lifetime did not prove exit: stopped=%t error=%v", stopped, err)
	}
}
