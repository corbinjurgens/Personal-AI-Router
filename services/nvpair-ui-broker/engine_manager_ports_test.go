// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
)

func TestEngineManagerPortsConfiguration(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		paths                 workerPaths
		wantHTTP, wantControl int
	}{
		{name: "defaults", wantHTTP: 14322, wantControl: 14323},
		{name: "HTTP override", paths: workerPaths{engineMgrHTTPPort: 25122}, wantHTTP: 25122, wantControl: 14323},
		{name: "control override", paths: workerPaths{engineMgrControlPort: 25123}, wantHTTP: 14322, wantControl: 25123},
		{name: "both overrides", paths: workerPaths{engineMgrHTTPPort: 25122, engineMgrControlPort: 25123}, wantHTTP: 25122, wantControl: 25123},
		{name: "range boundaries", paths: workerPaths{engineMgrHTTPPort: 1, engineMgrControlPort: 65535}, wantHTTP: 1, wantControl: 65535},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := NewBroker(nil, tc.paths)
			httpPort, controlPort := b.engineManagerPorts()
			if httpPort != tc.wantHTTP || controlPort != tc.wantControl {
				t.Fatalf("ports = %d/%d, want %d/%d", httpPort, controlPort, tc.wantHTTP, tc.wantControl)
			}
			if err := validateServicePorts(b.servicePorts()); err != nil {
				t.Fatalf("valid configuration rejected: %v", err)
			}
		})
	}
}

func TestEngineManagerPortsCLIRejectsInvalidBeforeWorkerResolution(t *testing.T) {
	binary := buildEngineManagerPortTestBinary(t, "ports-cli", "nvpair-ui-broker", ".", ".")
	for _, tc := range []struct {
		name, httpPort, controlPort, want string
		extra                             []string
	}{
		{name: "node-info collides with HTTP", httpPort: "25122", controlPort: "25123", extra: []string{"--node-info-port", "25122"}, want: "--node-info-port and --engine-manager-http-port must differ (both are 25122)"},
		{name: "control collides with default cluster port", httpPort: "25122", controlPort: "14321", want: "--cluster-manager-port and --engine-manager-control-port must differ (both are 14321)"},
		{name: "errors below range", httpPort: "25122", controlPort: "25123", extra: []string{"--errors-port", "0"}, want: "--errors-port must be between 1 and 65535 (got 0)"},
		{name: "workload on a compatibility facade", httpPort: "25122", controlPort: "25123", extra: []string{"--workload-manager-port", "11434"}, want: "--workload-manager-port 11434 is reserved"},
		{name: "HTTP on a managed backend base", httpPort: "1235", controlPort: "25123", want: "--engine-manager-http-port 1235 is reserved"},
		{name: "zero HTTP", httpPort: "0", controlPort: "25123", want: "--engine-manager-http-port must be between 1 and 65535 (got 0)"},
		{name: "zero control", httpPort: "25122", controlPort: "0", want: "--engine-manager-control-port must be between 1 and 65535 (got 0)"},
		{name: "negative HTTP", httpPort: "-1", controlPort: "25123", want: "--engine-manager-http-port must be between 1 and 65535 (got -1)"},
		{name: "negative control", httpPort: "25122", controlPort: "-1", want: "--engine-manager-control-port must be between 1 and 65535 (got -1)"},
		{name: "HTTP above range", httpPort: "65536", controlPort: "25123", want: "--engine-manager-http-port must be between 1 and 65535 (got 65536)"},
		{name: "control above range", httpPort: "25122", controlPort: "65536", want: "--engine-manager-control-port must be between 1 and 65535 (got 65536)"},
		{name: "equal ports", httpPort: "25122", controlPort: "25122", want: "must differ (both are 25122)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			args := append([]string{
				"--scanner-path", filepath.Join(t.TempDir(), "missing-scanner"),
				"--engine-manager-http-port", tc.httpPort,
				"--engine-manager-control-port", tc.controlPort,
			}, tc.extra...)
			output, err := exec.CommandContext(ctx, binary, args...).CombinedOutput()
			if ctx.Err() != nil {
				t.Fatalf("startup did not fail promptly: %v", ctx.Err())
			}
			if err == nil {
				t.Fatalf("invalid configuration accepted: %s", output)
			}
			if !strings.Contains(string(output), tc.want) || strings.Contains(string(output), "scanner binary:") {
				t.Fatalf("startup diagnostic = %q, want port error %q before worker resolution", output, tc.want)
			}
		})
	}
}

func TestEngineManagerPortsInvalidProgrammaticConfigurationDoesNotSpawn(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		httpPort, controlPort int
	}{
		{name: "invalid HTTP", httpPort: -1, controlPort: 25123},
		{name: "invalid control", httpPort: 25122, controlPort: 65536},
		{name: "equal explicit ports", httpPort: 25122, controlPort: 25122},
		{name: "override collides with default", httpPort: 14323},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := NewBroker(nil, workerPaths{engineMgrHTTPPort: tc.httpPort, engineMgrControlPort: tc.controlPort})
			if err := b.Serve(context.Background()); err == nil || !strings.Contains(err.Error(), "service ports:") {
				t.Fatalf("Serve error = %v, want port configuration failure", err)
			}
			if handle, err := b.spawnEngineMgr(); err == nil || handle != nil || !strings.Contains(err.Error(), "service ports:") {
				t.Fatalf("spawn = %v, %v, want port configuration failure without a handle", handle, err)
			}
			if b.getScanner() != nil || b.getEngineMgr() != nil || len(b.regCache.Snapshot()) != 0 {
				t.Fatal("invalid configuration spawned or registered a worker")
			}
		})
	}
}

func TestEngineManagerPortsSpawnAndScannerReplay(t *testing.T) {
	isolateOllamaHostTestConfig(t)
	engineBinary := buildPrivateEngineManagerPortFixture(t)
	scannerBinary := buildServicePortFixture(t)
	ports := make([]int, 2)
	listeners := make([]net.Listener, 2)
	for i := range ports {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("allocate engine-manager port: %v", err)
		}
		t.Cleanup(func() { _ = listener.Close() })
		listeners[i] = listener
		_, portText, err := net.SplitHostPort(listener.Addr().String())
		if err != nil {
			t.Fatalf("parse allocated address: %v", err)
		}
		ports[i], err = strconv.Atoi(portText)
		if err != nil {
			t.Fatalf("parse allocated port: %v", err)
		}
		if ports[i] == 14322 || ports[i] == 14323 {
			t.Fatalf("allocator returned product default %d; test requires isolated ports", ports[i])
		}
	}
	b := NewBroker(nil, workerPaths{
		scanner: scannerBinary, engineMgr: engineBinary,
		engineMgrHTTPPort: ports[0], engineMgrControlPort: ports[1],
		clusterDir: filepath.Join(t.TempDir(), "cluster"),
	})
	for _, listener := range listeners {
		if err := listener.Close(); err != nil {
			t.Fatalf("release allocated port: %v", err)
		}
	}
	supervisor := newSupervisor("engine-manager", noRestartPolicy(), b.spawnEngineMgr)
	if err := supervisor.Start(); err != nil {
		t.Fatalf("start supervised engine-manager: %v", err)
	}
	t.Cleanup(supervisor.Stop)
	worker := b.getEngineMgr()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	raw, rpcErr, err := worker.peer.Call(ctx, "engine:descriptors", nil)
	if err != nil || rpcErr != nil {
		t.Fatalf("read private engine registry: %v, %v", err, rpcErr)
	}
	var inventory struct {
		Engines []json.RawMessage `json:"engines"`
	}
	if err := json.Unmarshal(raw, &inventory); err != nil {
		t.Fatalf("decode private engine registry: %v", err)
	}
	if len(inventory.Engines) != 0 {
		t.Fatalf("listener fixture loaded engine manifests: %s", raw)
	}
	for _, flag := range []struct {
		name string
		port int
	}{
		{name: "--http-port", port: ports[0]},
		{name: "--control-port", port: ports[1]},
	} {
		found := false
		for i, arg := range worker.cmd.Args {
			if arg == flag.name && i+1 < len(worker.cmd.Args) {
				found = worker.cmd.Args[i+1] == strconv.Itoa(flag.port)
			}
		}
		if !found {
			t.Fatalf("spawn args = %v, want %s %d", worker.cmd.Args, flag.name, flag.port)
		}
	}
	client := &http.Client{Timeout: time.Second}
	deadline := time.Now().Add(10 * time.Second)
	for {
		response, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/v1/models", ports[0]))
		if err == nil {
			closeErr := response.Body.Close()
			if closeErr != nil || response.StatusCode != http.StatusOK {
				t.Fatalf("model listener response = %d, close error = %v", response.StatusCode, closeErr)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("model listener did not bind configured port: %v", err)
		}
		time.Sleep(20 * time.Millisecond)
	}
	connection, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(ports[1])), time.Second)
	if err != nil {
		t.Fatalf("control listener did not bind configured port: %v", err)
	}
	if err := connection.Close(); err != nil {
		t.Fatalf("close control probe: %v", err)
	}
	want := map[noderec.ServiceKey]int{"em": ports[0], "ec": ports[1]}
	cached := map[noderec.ServiceKey]int{}
	for _, registration := range b.regCache.Snapshot() {
		cached[registration.Service] = registration.Port
	}
	if !reflect.DeepEqual(cached, want) {
		t.Fatalf("cached registrations = %v, want %v", cached, want)
	}
	// Each fresh scanner must receive the cached ports without respawning the
	// engine-manager or rebuilding registrations from the product defaults.
	for _, generation := range []string{"initial scanner", "replacement scanner"} {
		t.Run(generation, func(t *testing.T) {
			handle, err := b.spawnScanner()
			if err != nil {
				t.Fatalf("spawn scanner: %v", err)
			}
			defer handle.Stop()
			defer b.setScanner(nil)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			for {
				result, rpcErr, err := b.getScanner().peer.Call(ctx, "test:get-registrations", nil)
				if err != nil || rpcErr != nil {
					t.Fatalf("read scanner registrations: %v, %v", err, rpcErr)
				}
				var got map[noderec.ServiceKey]int
				if err := json.Unmarshal(result, &got); err != nil {
					t.Fatalf("decode scanner registrations: %v", err)
				}
				if reflect.DeepEqual(got, want) {
					break
				}
				if ctx.Err() != nil {
					t.Fatalf("scanner replay = %v, want %v", got, want)
				}
				time.Sleep(10 * time.Millisecond)
			}
		})
	}
}

func TestEngineManagerPortsSettingsReserveConfiguredListeners(t *testing.T) {
	b := NewBroker(nil, workerPaths{engineMgrHTTPPort: 25122, engineMgrControlPort: 25123})
	for _, tc := range []struct {
		name                  string
		serverPort, proxyPort int
	}{
		{name: "HTTP server collision", serverPort: 25122, proxyPort: 25200},
		{name: "HTTP proxy collision", serverPort: 25200, proxyPort: 25122},
		{name: "control server collision", serverPort: 25123, proxyPort: 25200},
		{name: "control proxy collision", serverPort: 25200, proxyPort: 25123},
		{name: "node-info remains reserved", serverPort: 14318, proxyPort: 25200},
		{name: "errors remains reserved", serverPort: 14319, proxyPort: 25200},
		{name: "workload remains reserved", serverPort: 14320, proxyPort: 25200},
		{name: "cluster remains reserved", serverPort: 14321, proxyPort: 25200},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := b.validateSettingsPortsLocked(context.Background(), "ollama", settings.Config{ServerPort: tc.serverPort, ProxyPort: tc.proxyPort}, settings.Snapshot{})
			if err == nil || err.Error() != "a selected port is reserved by a PAIR service or proxy alias" {
				t.Fatalf("settings validation error = %v, want PAIR reservation", err)
			}
		})
	}
}

func TestEngineManagerPortsSettingsReserveTheRoutingListenerPort(t *testing.T) {
	b, _ := newPolicyBroker(t)
	path := filepath.Join(t.TempDir(), nodepolicy.FileName)
	doc := nodepolicy.Default()
	doc.Routing.Listener = nodepolicy.Listener{Enabled: true, Port: 25300, AllowedOrigins: []string{}}
	writePolicyFile(t, path, doc)
	b.loadNodePolicy(path)
	for _, tc := range []struct {
		name                  string
		serverPort, proxyPort int
	}{
		{name: "server port", serverPort: 25300, proxyPort: 25301},
		{name: "proxy port", serverPort: 25301, proxyPort: 25300},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := b.validateSettingsPortsLocked(context.Background(), "ollama", settings.Config{ServerPort: tc.serverPort, ProxyPort: tc.proxyPort}, settings.Snapshot{})
			if err == nil || err.Error() != "a selected port is reserved by a PAIR service or proxy alias" {
				t.Fatalf("settings validation error = %v, want the routing listener port reserved", err)
			}
		})
	}
}

func TestEngineManagerPortsSettingsReleaseUnusedDefaults(t *testing.T) {
	b := NewBroker(nil, workerPaths{engineMgrHTTPPort: 25122, engineMgrControlPort: 25123})
	worker, codec := newTestRPCWorkerPipe(t)
	b.setEngineMgr(worker)
	go func() {
		request, err := codec.Read()
		if err != nil {
			t.Errorf("read configured ports request: %v", err)
			return
		}
		if request.Method != "engine:configured-ports" {
			t.Errorf("request = %q, want engine:configured-ports", request.Method)
		}
		if err := codec.Respond(request.ID, map[string]any{"engines": []any{}}); err != nil {
			t.Errorf("respond with configured ports: %v", err)
		}
	}()
	current := settings.Snapshot{Running: true, EffectiveServerPort: 14322, EffectiveProxyPort: 14323}
	if err := b.validateSettingsPortsLocked(context.Background(), "ollama", settings.Config{ServerPort: 14322, ProxyPort: 14323}, current); err != nil {
		t.Fatalf("unused default listener ports remain reserved: %v", err)
	}
}

// The listener/replay test needs the real main, Manager, Executor and listener
// owners, but no engine inventory. Its generated build overlay changes only
// startup registry construction, so watchLoaded cannot probe bundled engines.
// Bundled-manifest loading remains covered by engine-manager's registry tests.
func buildPrivateEngineManagerPortFixture(t *testing.T) string {
	t.Helper()
	mainPath, err := filepath.Abs("../nvpair-engine-manager/main.go")
	if err != nil {
		t.Fatalf("resolve engine-manager main: %v", err)
	}
	source, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatalf("read engine-manager main: %v", err)
	}
	fixture, err := privateEngineManagerPortFixtureSource(string(source))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	fixturePath := filepath.Join(dir, "main.go")
	if err := os.WriteFile(fixturePath, []byte(fixture), 0600); err != nil {
		t.Fatalf("write private engine-manager main: %v", err)
	}
	overlay, err := json.Marshal(struct {
		Replace map[string]string
	}{Replace: map[string]string{mainPath: fixturePath}})
	if err != nil {
		t.Fatalf("encode engine-manager overlay: %v", err)
	}
	overlayPath := filepath.Join(dir, "overlay.json")
	if err := os.WriteFile(overlayPath, overlay, 0600); err != nil {
		t.Fatalf("write engine-manager overlay: %v", err)
	}
	return buildEngineManagerPortTestBinary(t, "ports-spawn", "nvpair-engine-manager", "../nvpair-engine-manager", ".", "-overlay", overlayPath)
}

func privateEngineManagerPortFixtureSource(source string) (string, error) {
	const startupRegistry = "reg := buildRegistry(manifestDir)"
	if count := strings.Count(source, startupRegistry); count != 1 {
		return "", fmt.Errorf("engine-manager fixture requires exactly one startup registry construction, got %d", count)
	}
	return strings.Replace(source, startupRegistry, "reg := NewRegistry()", 1), nil
}

func TestEngineManagerPortFixtureRegistryIsolation(t *testing.T) {
	source, err := os.ReadFile("../nvpair-engine-manager/main.go")
	if err != nil {
		t.Fatalf("read production engine-manager main: %v", err)
	}
	t.Run("production startup has one replacement site", func(t *testing.T) {
		fixture, err := privateEngineManagerPortFixtureSource(string(source))
		if err != nil {
			t.Fatal(err)
		}
		// Restoring the private constructor must recover the entire production
		// source. Transport, Manager startup and both listener owners stay real.
		if strings.Replace(fixture, "reg := NewRegistry()", "reg := buildRegistry(manifestDir)", 1) != string(source) {
			t.Fatal("fixture overlay changed more than startup registry construction")
		}
	})
	for _, tc := range []struct {
		name   string
		source string
	}{
		{name: "missing construction", source: strings.ReplaceAll(string(source), "reg := buildRegistry(manifestDir)", "reg := NewRegistry()")},
		{name: "duplicate construction", source: string(source) + "\nreg := buildRegistry(manifestDir)\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if fixture, err := privateEngineManagerPortFixtureSource(tc.source); err == nil || fixture != "" {
				t.Fatalf("unsafe overlay returned %d bytes, %v; want rejection without generated source", len(fixture), err)
			}
		})
	}
}

func buildEngineManagerPortTestBinary(t *testing.T, role, program, directory, target string, buildFlags ...string) string {
	t.Helper()
	binary := filepath.Join(testbin.Dir(t, "nvpair-ui-broker", role, program), testbin.Exe(program))
	if err := buildPortTestBinary(binary, directory, target, buildFlags...); err != nil {
		t.Fatal(err)
	}
	return binary
}

func buildPortTestBinary(binary, directory, target string, buildFlags ...string) error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	args := append([]string{"build", "-modcacherw", "-o", binary}, buildFlags...)
	command := exec.CommandContext(ctx, "go", append(args, target)...)
	command.Dir = directory
	// Preserve the real caches while fixtures keep their private app-data roots.
	command.Env = append(os.Environ(), portTestGoEnv...)
	if output, err := command.CombinedOutput(); err != nil {
		return fmt.Errorf("build %s in %s: %w\n%s", target, directory, err, output)
	}
	return nil
}

const engineManagerPortScannerFixture = `// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package main

import (
	"encoding/json"
	"os"
)

func main() {
	decoder, encoder := json.NewDecoder(os.Stdin), json.NewEncoder(os.Stdout)
	registrations := map[string]int{}
	for {
		var request struct {
			ID json.RawMessage
			Method string
			Params struct { Service string; Port int }
		}
		if decoder.Decode(&request) != nil {
			return
		}
		if len(request.ID) == 0 {
			continue
		}
		var result any = registrations
		if request.Method == "discovery:register" {
			registrations[request.Params.Service] = request.Params.Port
			result = map[string]bool{"ok": true}
		}
		if encoder.Encode(map[string]any{"jsonrpc": "2.0", "id": request.ID, "result": result}) != nil {
			return
		}
	}
}
`
