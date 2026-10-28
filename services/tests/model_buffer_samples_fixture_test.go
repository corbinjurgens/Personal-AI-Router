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

	"nvpair-shared/engines"
	"nvpair-shared/modelconfig"
	"nvpair-shared/testbin"
)

const modelBufferFixtureSelector = "NVPAIR_TEST_MODEL_BUFFER_ENGINE"
const modelBufferFixtureControl = "/__model_buffer_fixture/control"

// Dispatch in the registered services/tests package image, before TestMain.
// No fake manager response, dynamic library, GGUF parser or inference runtime.
func init() {
	if image := os.Getenv(modelBufferFixtureSelector); image != "" && image == os.Args[0] {
		if runtime.GOOS != "windows" {
			os.Exit(2)
		}
		if err := runModelBufferFixture(); err != nil {
			fmt.Fprintln(os.Stderr, "model buffer fixture failed")
			os.Exit(2)
		}
		os.Exit(0)
	}
}

func modelBufferFixtureEffect(kind string) error {
	file, err := os.OpenFile(os.Getenv("NVPAIR_TEST_MODEL_BUFFER_EFFECTS"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	_, writeErr := fmt.Fprintln(file, kind)
	return errors.Join(writeErr, file.Close())
}

type modelBufferFixtureModel struct {
	context, slots, port int
	status, build        string
}

func runModelBufferFixture() error {
	image, err := os.Executable()
	if err != nil || !modelSettingsSamePath(image, os.Getenv(modelBufferFixtureSelector)) {
		return errors.New("invalid private image")
	}
	identityCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	identity, identityErr := serviceProcessIdentityAt(identityCtx, int32(os.Getpid()))
	cancel()
	if identityErr != nil || !strings.EqualFold(filepath.Clean(identity.Path), filepath.Clean(os.Getenv(modelBufferFixtureSelector))) {
		return errors.New("actual process identity differs from registered fixed image")
	}
	for _, arg := range os.Args[1:] {
		if arg == "--private-model-devices" || arg == "--version" {
			return errors.Join(modelBufferFixtureEffect("helper-or-version"), errors.New("forbidden fixture operation"))
		}
	}
	index := slices.Index(os.Args, "--models-preset")
	if index < 0 || index+1 >= len(os.Args) {
		return errors.New("missing finite preset")
	}
	file, err := os.Open(os.Args[index+1])
	if err != nil {
		return err
	}
	data, readErr := io.ReadAll(io.LimitReader(file, (1<<20)+1))
	if err := errors.Join(readErr, file.Close()); err != nil {
		return err
	}
	if len(data) > 1<<20 {
		return errors.New("preset budget")
	}
	if err := os.WriteFile(os.Getenv("NVPAIR_TEST_MODEL_CAPTURE"), data, 0o600); err != nil {
		return err
	}
	models := map[string]*modelBufferFixtureModel{}
	section := ""
	scan := bufio.NewScanner(bytes.NewReader(data))
	for scan.Scan() {
		line := strings.TrimSpace(scan.Text())
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = line[1 : len(line)-1]
			continue
		}
		if section != "medium-model" && section != "weak-model" {
			continue
		}
		if models[section] == nil {
			models[section] = &modelBufferFixtureModel{slots: 1, status: "unloaded", build: "b11146-7fe450e"}
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		switch key {
		case "ctx-size":
			models[section].context, err = strconv.Atoi(value)
		case "parallel":
			models[section].slots, err = strconv.Atoi(value)
		}
		if err != nil {
			return err
		}
	}
	if err := scan.Err(); err != nil {
		return err
	}
	for _, model := range models {
		if model.context <= 0 || model.slots <= 0 || model.context%model.slots != 0 {
			return errors.New("nonfinite fixture allocation")
		}
		model.context /= model.slots
	}
	if len(models) != 2 {
		return errors.New("missing exact model identities")
	}
	port, err := strconv.Atoi(os.Getenv("NVPAIR_TEST_MODEL_PORT"))
	if err != nil || validateFixturePort(port) != nil {
		return errors.New("invalid private port")
	}
	listener, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
	if err != nil {
		return err
	}
	defer listener.Close()
	if err := modelBufferFixtureEffect("start"); err != nil {
		return err
	}
	if _, err := modelSettingsControl("/engine-started?pid=" + strconv.Itoa(os.Getpid())); err != nil {
		return err
	}
	var mu sync.Mutex
	nextPort := 31001
	// All pinned frames use one stream, so this fixture establishes real pump
	// order. Native source: tools/server/server-models.cpp at the full pin.
	line := func(format string, args ...any) { fmt.Fprintf(os.Stderr, format+"\n", args...) }
	allocations := func(name string, model *modelBufferFixtureModel, initial bool) {
		line("[%5d] srv   load_model: loading model 'private-fixture-path'", model.port)
		for _, message := range []string{
			"load_tensors: CPU model buffer size = 1.00 MiB", "load_tensors: CPU model buffer size = 1.00 MiB",
			"llama_kv_cache: CPU KV buffer size = 0.00 MiB", "llama_kv_cache: CPU KV buffer size = 0.25 MiB",
			"llama_memory_recurrent: CPU RS buffer size = 0.10 MiB", "llama_context: CPU output buffer size = 0.10 MiB",
			"sched_reserve: reserving ...", "sched_reserve: CPU compute buffer size = 1.00 MiB", "sched_reserve: reserve took 1.00 ms, sched copies = 1",
			"sched_reserve: reserving ...", "sched_reserve: CPU compute buffer size = 0.50 MiB", "sched_reserve: reserve took 1.00 ms, sched copies = 1",
		} {
			line("[%5d] %s", model.port, message)
		}
		if initial {
			line("[%5d] srv         main: model loaded", model.port)
		}
		model.status = "loaded"
	}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		name := r.URL.Query().Get("model")
		model := models[name]
		if r.URL.Path == modelBufferFixtureControl {
			if r.Method != http.MethodPost || r.Header.Get("X-Model-Buffer-Fixture") != os.Getenv("NVPAIR_TEST_MODEL_BUFFER_TOKEN") || model == nil {
				http.Error(w, "invalid private control", http.StatusForbidden)
				return
			}
			switch r.URL.Query().Get("op") {
			case "initial", "reload":
				if model.port != 0 {
					http.Error(w, "existing child", http.StatusConflict)
					return
				}
				model.port = nextPort
				nextPort++
				line("srv         load: spawning server instance with name=%s on port %d", name, model.port)
				allocations(name, model, true)
			case "sleep":
				line("[%5d] srv handle_sleep: server is entering sleeping state", model.port)
				model.status = "sleeping"
			case "wakeup":
				line("[%5d] srv handle_sleep: server is exiting sleeping state", model.port)
				allocations(name, model, false)
			case "exit":
				// server_monitor::run, not operator(): exact pinned exit emitter.
				// https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/tools/server/server-models.cpp#L235-L242
				line("srv          run: instance name=%s exited with status 0", name)
				model.status = "unloaded"
				model.port = 0
			case "oversized":
				line("%s", strings.Repeat("x", 70<<10))
			case "unsupported":
				model.build = "b11147-7fe450e"
			default:
				http.Error(w, "unknown private operation", http.StatusBadRequest)
				return
			}
			fmt.Fprintln(w, `{"ok":true}`)
			return
		}
		if r.Method != http.MethodGet {
			kind := "mutation"
			if r.URL.Path == "/v1/chat/completions" {
				kind = "inference"
			}
			if err := modelBufferFixtureEffect(kind); err != nil {
				http.Error(w, "effect recording failed", http.StatusInternalServerError)
				return
			}
			http.Error(w, "prohibited synthetic operation", http.StatusBadRequest)
			return
		}
		switch r.URL.Path {
		case "/props":
			if name == "" {
				fmt.Fprintln(w, `{"role":"router","build_info":"b11146-7fe450e"}`)
				return
			}
			if model == nil || model.status != "loaded" || r.URL.Query().Get("autoload") != "false" {
				http.Error(w, "non-loading props required", http.StatusConflict)
				return
			}
			if err := json.NewEncoder(w).Encode(map[string]any{"build_info": model.build, "total_slots": model.slots, "default_generation_settings": map[string]int{"n_ctx": model.context}}); err != nil {
				return
			}
		case "/models", "/v1/models":
			var rows []map[string]any
			for _, name := range []string{"medium-model", "weak-model"} {
				rows = append(rows, map[string]any{"id": name, "status": map[string]string{"value": models[name].status}})
			}
			if err := json.NewEncoder(w).Encode(map[string]any{"data": rows}); err != nil {
				return
			}
		default:
			fmt.Fprintln(w, `{"status":"ok"}`)
		}
	})
	server := &http.Server{Handler: handler, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 5 * time.Second, WriteTimeout: 5 * time.Second, MaxHeaderBytes: 4096}
	return server.Serve(listener)
}

type modelBufferProcessFixture struct {
	*modelSettingsProcessFixture
	token, effects, library, history string
}

func modelBufferFixedSnapshot(t *testing.T, directory, image string) (map[string]int64, string) {
	t.Helper()
	entries, err := os.ReadDir(directory)
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]int64{}
	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			t.Fatal(err)
		}
		names[entry.Name()] = info.Size()
	}
	file, err := os.Open(image)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.New()
	_, readErr := io.Copy(digest, file)
	if err := errors.Join(readErr, file.Close()); err != nil {
		t.Fatal(err)
	}
	return names, hex.EncodeToString(digest.Sum(nil))
}

// Exact current modelDeviceLibraryName vocabulary. Together with a nonempty
// source-bound observation, this proves RuntimeSnapshot.Libraries is only the
// inert stub, rather than silently binding unrelated native libraries.
func modelBufferAssertOnlyStub(t *testing.T, root, stub string) {
	t.Helper()
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	var libraries []string
	for _, entry := range entries {
		name := strings.ToLower(entry.Name())
		if !strings.HasSuffix(name, ".dll") && !strings.HasSuffix(name, ".dylib") && !strings.Contains(name, ".so") {
			continue
		}
		for _, prefix := range []string{"ggml", "libggml", "libllama", "llama", "cudart", "cublas", "libcudart", "libcublas"} {
			if strings.HasPrefix(name, prefix) {
				libraries = append(libraries, entry.Name())
				break
			}
		}
	}
	if len(libraries) != 1 || libraries[0] != filepath.Base(stub) {
		t.Fatalf("native source libraries must be exactly the inert stub; found %v", libraries)
	}
	data, err := os.ReadFile(stub)
	if err != nil || bytes.HasPrefix(data, []byte("MZ")) || !bytes.Equal(data, []byte("Synthetic metadata binding only, never a loadable library.\n")) {
		t.Fatalf("stub is not the exact non-PE fixture metadata: %v", err)
	}
}

func newModelBufferProcessFixture(t *testing.T, configure ...func(*modelSettingsProcessFixture)) *modelBufferProcessFixture {
	t.Helper()
	if runtime.GOOS != "windows" {
		t.Skip("PC3 Windows serialized fixed-image acceptance only")
	}
	f := &modelBufferProcessFixture{modelSettingsProcessFixture: prepareModelSettingsProcessFixture(t)}
	identityBytes := make([]byte, 16)
	if _, err := rand.Read(identityBytes); err != nil {
		t.Fatal(err)
	}
	f.token = hex.EncodeToString(identityBytes)
	for _, change := range configure {
		change(f.modelSettingsProcessFixture)
	}
	f.effects = filepath.Join(f.base, "model-buffer-effects.txt")
	f.history = filepath.Join(jointAppDir(f.base), "engine-bin", "model-buffer-samples.json")
	// Existing managed-source validation requires the actual image to belong
	// to the selected install root. A private junction binds that root to the
	// existing registered package directory; the executable path never changes.
	install := filepath.Join(jointAppDir(f.base), "engine-bin", "llamacpp")
	if err := os.MkdirAll(filepath.Dir(install), 0o700); err != nil {
		t.Fatal(err)
	}
	imageDir := filepath.Dir(f.image)
	expectedImage := filepath.Join(os.Getenv(testbin.EnvRoot), "go-test", "tests", "tests.test.exe")
	if !filepath.IsAbs(expectedImage) || !strings.EqualFold(filepath.Clean(expectedImage), filepath.Clean(f.image)) || !strings.Contains(os.Getenv("GOFLAGS"), "nvpair-test-exec") {
		t.Fatal("requires existing registered fixed image and firewall rule via serialized PC3 wrapper")
	}
	fixedBefore, imageBefore := modelBufferFixedSnapshot(t, imageDir, f.image)
	f.library = filepath.Join(imageDir, "libggml-pair-model-buffer-fixture.dll")
	stub := []byte("Synthetic metadata binding only, never a loadable library.\n")
	if bytes.HasPrefix(stub, []byte("MZ")) {
		t.Fatal("fixture metadata stub must be non-PE")
	}
	junctionCreated, stubCreated := false, false
	// Registered after TempDir, before processes: LIFO closes every process,
	// removes only the junction, then the owned stub, then checks fixed state.
	t.Cleanup(func() {
		if junctionCreated {
			if err := os.Remove(install); err != nil {
				t.Errorf("remove private junction itself: %v", err)
			}
		}
		if stubCreated {
			data, err := os.ReadFile(f.library)
			if err != nil || !bytes.Equal(data, stub) {
				t.Errorf("owned stub changed; preserve stale leftover for root at %s: %v", f.library, err)
			} else if err := os.Remove(f.library); err != nil {
				t.Error(err)
			}
		}
		fixedAfter, imageAfter := modelBufferFixedSnapshot(t, imageDir, f.image)
		if !reflect.DeepEqual(fixedBefore, fixedAfter) || imageBefore != imageAfter {
			t.Errorf("fixed directory/image changed: before=%v after=%v imageSHA256=%s/%s", fixedBefore, fixedAfter, imageBefore, imageAfter)
		}
		var delta []string
		for name, size := range fixedBefore {
			if after, ok := fixedAfter[name]; !ok || after != size {
				delta = append(delta, name)
			}
		}
		for name := range fixedAfter {
			if _, ok := fixedBefore[name]; !ok {
				delta = append(delta, name)
			}
		}
		slices.Sort(delta)
		t.Logf("model_buffer_fixed_directory before=%v after=%v delta=%v expected_delta=zero image_sha256_before=%s image_sha256_after=%s", fixedBefore, fixedAfter, delta, imageBefore, imageAfter)
	})
	for _, path := range []string{install, imageDir} {
		if strings.ContainsAny(path, "\"&|<>^%!\r\n") {
			t.Fatal("unsafe fixture junction path")
		}
	}
	// Separate arguments let Go quote each path; a pre-quoted single argument
	// reaches cmd.exe with escaped quotes and mklink rejects the syntax.
	command := exec.Command("cmd.exe", "/d", "/c", "mklink", "/J", install, imageDir)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("private fixed-image junction: %v: %s", err, output)
	}
	junctionCreated = true
	if err := modelBufferSameDirectory(install, imageDir); err != nil {
		t.Fatalf("private install junction must resolve to registered fixed image directory: %v", err)
	}
	library, err := os.OpenFile(f.library, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		t.Fatalf("metadata stub path occupied or unavailable; preserve stale leftover for root at %s: %v", f.library, err)
	}
	stubCreated = true
	_, writeErr := library.Write(stub)
	if err := errors.Join(writeErr, library.Close()); err != nil {
		t.Fatal(err)
	}
	modelBufferAssertOnlyStub(t, install, f.library)
	path := filepath.Join(jointAppDir(f.base), "engines", "llamacpp.json")
	var manifest map[string]any
	modelSettingsDecode(t, readLiveDurableFile(t, path), &manifest)
	platforms, ok := manifest["platforms"].(map[string]any)
	if !ok {
		t.Fatal("missing fixture platforms")
	}
	platform, ok := platforms[runtime.GOOS+"/"+runtime.GOARCH].(map[string]any)
	if !ok {
		t.Fatal("missing fixture platform")
	}
	for _, owner := range []map[string]any{manifest, platform} {
		rt, ok := owner["runtime"].(map[string]any)
		if !ok {
			t.Fatal("missing fixture runtime")
		}
		env, ok := rt["env"].(map[string]any)
		if !ok {
			t.Fatal("missing fixture environment")
		}
		env["NVPAIR_TEST_MODEL_ENGINE"] = ""
		env[modelBufferFixtureSelector] = f.image
		env["NVPAIR_TEST_MODEL_BUFFER_EFFECTS"] = f.effects
		env["NVPAIR_TEST_MODEL_BUFFER_TOKEN"] = f.token
		owner["install"] = map[string]any{"run": []string{"fixture-never-run"}, "artifacts": []map[string]string{{"name": "server", "url": "https://fixture.invalid/releases/download/b11146/native", "sha256": strings.Repeat("a", 64)}}}
	}
	jointWriteJSON(t, path, manifest)
	f.start(t)
	request, _ := f.review(t, f.request(t, "model-buffer-finite", nil))
	f.rpcOK(t, "engine:apply-settings", request)
	f.settled(t, request.RequestID, "succeeded")
	return f
}

func (f *modelBufferProcessFixture) start(t *testing.T) {
	t.Helper()
	f.assertNativeRoot(t)
	f.assertBufferEngineManifests(t)
	args := append([]string{"--scanner-path", f.scanner, "--engine-manager-path", engineMgrBin, "--engine-manager-control-host", "127.0.0.1", "--proxy-path", proxyBin, "--proxy-engines", "ollama,llamacpp", "--cluster-dir", filepath.Join(f.base, "cluster")}, f.ports.args()...)
	f.p = startInteropProcess(t, f.base, brokerBin, f.ports.engineManagerHTTP, []string{"NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + f.scanner, "NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(f.scannerPort)}, args...)
	f.p.ports = append(f.p.ports, f.scannerPort, f.targetPort, f.sourcePort, f.facadePort, f.targetFacadePort, f.ports.engineControl)
	for _, path := range []string{engineMgrBin, proxyBin, f.scanner} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		f.p.paths = append(f.p.paths, canonical)
	}
	waitForMethod(t, f.p.msgs, "app:ready", 30*time.Second)
	waitProxyReady(t, f.p.in, f.p.msgs, 20*time.Second)
	waitEngineProxyReady(t, "llamacpp-proxy", f.p.in, f.p.msgs, 20*time.Second)
}

// The buffer child has its own selector. Keep the existing fixed-image,
// private-model-root and loopback isolation checks without enabling the older
// model-settings child mode beside it.
func (f *modelBufferProcessFixture) assertBufferEngineManifests(t *testing.T) {
	t.Helper()
	if err := f.ports.validate(); err != nil {
		t.Fatal(err)
	}
	seen := map[int]bool{}
	ports := []int{f.ports.nodeInfo, f.ports.errors, f.ports.workload, f.ports.clusterManager, f.ports.engineManagerHTTP, f.ports.engineControl, f.sourcePort, f.targetPort, f.facadePort, f.targetFacadePort, f.scannerPort, portOfURL(t, f.modelSettingsProcessFixture.control), portOfURL(t, f.switchProcessFixture.control.URL)}
	if f.proxyProbeImage != "" {
		ports = append(ports, f.proxyProbePort)
	}
	for _, port := range ports {
		if err := validateFixturePort(port); err != nil {
			t.Fatal(err)
		}
		if seen[port] {
			t.Fatal("buffer fixture repeats a private port")
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
			t.Fatal("private buffer manifest lacks its active platform")
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
			t.Fatal("private buffer detection has an unexpected alternative")
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
		expectedPort, selector := f.targetPort, modelBufferFixtureSelector
		if name == "ollama" {
			expectedPort, selector = f.sourcePort, "NVPAIR_TEST_SWITCH_ENGINE"
		}
		for _, rt := range []modelSettingsRuntimeCheck{manifest.Runtime, platform.Runtime} {
			if rt.Bind != "127.0.0.1" || rt.Port != expectedPort || !modelSettingsSamePath(rt.Env[selector], f.image) {
				t.Fatalf("engine %s lacks its fixed buffer selector/private loopback bind", name)
			}
			if name == "llamacpp" && (!modelSettingsSamePath(rt.Env["LLAMA_CACHE"], f.root) || rt.Env["NVPAIR_TEST_MODEL_ENGINE"] != "" || rt.Env["NVPAIR_TEST_MODEL_BUFFER_TOKEN"] != f.token || rt.Env["NVPAIR_TEST_MODEL_BUFFER_EFFECTS"] != f.effects) {
				t.Fatal("synthetic buffer router can inherit another child mode or external state")
			}
		}
	}
}

// modelBufferSameDirectory compares directory identity. EvalSymlinks does not
// follow Windows junctions, so it returns the link spelling; Stat follows the
// junction and SameFile compares the volume and file index, as the engine
// manager's windowsImageJunction test does.
func modelBufferSameDirectory(link, target string) error {
	got, err := os.Stat(link)
	if err != nil {
		return err
	}
	want, err := os.Stat(target)
	if err != nil {
		return err
	}
	if !os.SameFile(got, want) {
		return fmt.Errorf("%q does not resolve to %q", link, target)
	}
	return nil
}

func (f *modelBufferProcessFixture) assertNativeRoot(t *testing.T) {
	t.Helper()
	root := filepath.Join(jointAppDir(f.base), "engine-bin", "llamacpp")
	if err := modelBufferSameDirectory(root, filepath.Dir(f.image)); err != nil {
		t.Fatalf("junction changed before fixed-image use: %v", err)
	}
	modelBufferAssertOnlyStub(t, root, f.library)
}

func (f *modelBufferProcessFixture) rpcOK(t *testing.T, method string, params any) json.RawMessage {
	t.Helper()
	if method == "engine:start" || method == "engine:restart" {
		f.assertNativeRoot(t)
	}
	return f.modelSettingsProcessFixture.rpcOK(t, method, params)
}

func (f *modelBufferProcessFixture) control(t *testing.T, model, op string) {
	t.Helper()
	request, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d%s?model=%s&op=%s", f.targetPort, modelBufferFixtureControl, model, op), nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("X-Model-Buffer-Fixture", f.token)
	response, err := (&http.Client{Timeout: 10 * time.Second}).Do(request)
	if err != nil {
		t.Fatal(err)
	}
	_, readErr := io.Copy(io.Discard, io.LimitReader(response.Body, 4096))
	closeErr := response.Body.Close()
	if err := errors.Join(readErr, closeErr); err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatalf("private synthetic operation %s: HTTP %d", op, response.StatusCode)
	}
}

func (f *modelBufferProcessFixture) inspect(t *testing.T, model string) modelconfig.InspectResult {
	t.Helper()
	raw := f.rpcOK(t, "engine:inspect-model", modelconfig.InspectRequest{Engine: "llamacpp", ModelID: model})
	value, err := modelconfig.DecodeInspectResult(raw)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func (f *modelBufferProcessFixture) observed(t *testing.T, model, phase string) modelconfig.BufferObservation {
	t.Helper()
	var result modelconfig.BufferObservation
	task1Eventually(t, "real-manager buffer checkpoint "+model+"/"+phase, 90*time.Second, func() bool {
		value := f.inspect(t, model)
		if value.LoadObservation == nil {
			return false
		}
		result = value.LoadObservation.Clone()
		return result.Current && result.LoadPhase == phase && result.Persistence == "saved" && len(result.Rows) > 0
	})
	return result
}
