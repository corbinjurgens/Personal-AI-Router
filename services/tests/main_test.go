// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bufio"
	"encoding/json"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/appdir"
	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/testbin"
)

var (
	proxyBin        string
	errorsBin       string
	nodeInfoBin     string
	scannerBin      string
	nodeSettingsBin string
	brokerBin       string
	serviceBin      string
	workloadMgrBin  string
	engineMgrBin    string
	manualNodesBin  string
	clusterMgrBin   string
	schedulerBin    string
	testsConfigBase string
)

func TestMain(m *testing.M) {
	tmpDir, err := os.MkdirTemp("", "nvpair-tests-*")
	if err != nil {
		log.Fatalf("failed to create temp dir: %v", err)
	}

	binDir, cleanupBins, err := testbin.MainDir("tests", "products", "nvpair-tests-*",
		"nvpair-proxy", "nvpair-errors", "nvpair-node-info", "nvpair-node-scanner",
		"nvpair-ui-broker", "nvpair-service", "nvpair-node-settings", "nvpair-workload-manager",
		"nvpair-engine-manager", "nvpair-manual-nodes", "nvpair-cluster-manager", "nvpair-job-scheduler")
	if err != nil {
		os.RemoveAll(tmpDir)
		log.Fatalf("allocate test binaries: %v", err)
	}

	proxyBin = filepath.Join(binDir, testbin.Exe("nvpair-proxy"))
	errorsBin = filepath.Join(binDir, testbin.Exe("nvpair-errors"))
	nodeInfoBin = filepath.Join(binDir, testbin.Exe("nvpair-node-info"))
	scannerBin = filepath.Join(binDir, testbin.Exe("nvpair-node-scanner"))
	brokerBin = filepath.Join(binDir, testbin.Exe("nvpair-ui-broker"))
	serviceBin = filepath.Join(binDir, testbin.Exe("nvpair-service"))
	nodeSettingsBin = filepath.Join(binDir, testbin.Exe("nvpair-node-settings"))
	workloadMgrBin = filepath.Join(binDir, testbin.Exe("nvpair-workload-manager"))
	engineMgrBin = filepath.Join(binDir, testbin.Exe("nvpair-engine-manager"))
	manualNodesBin = filepath.Join(binDir, testbin.Exe("nvpair-manual-nodes"))
	clusterMgrBin = filepath.Join(binDir, testbin.Exe("nvpair-cluster-manager"))
	schedulerBin = filepath.Join(binDir, testbin.Exe("nvpair-job-scheduler"))
	cleanup := func() {
		cleanupBins()
		os.RemoveAll(tmpDir)
	}

	// Build distinct product outputs before redirecting the cache environment.
	builds := []struct {
		name   string
		output string
	}{
		{"nvpair-proxy", proxyBin},
		{"nvpair-errors", errorsBin},
		{"nvpair-node-info", nodeInfoBin},
		{"nvpair-node-scanner", scannerBin},
		{"nvpair-ui-broker", brokerBin},
		{"nvpair-service", serviceBin},
		{"nvpair-node-settings", nodeSettingsBin},
		{"nvpair-workload-manager", workloadMgrBin},
		{"nvpair-engine-manager", engineMgrBin},
		{"nvpair-manual-nodes", manualNodesBin},
		{"nvpair-cluster-manager", clusterMgrBin},
		{"nvpair-job-scheduler", schedulerBin},
	}
	limit := min(runtime.GOMAXPROCS(0), 4)
	slots := make(chan struct{}, limit)
	var workers sync.WaitGroup
	buildErrors := make([]error, len(builds))
	for i, build := range builds {
		slots <- struct{}{}
		workers.Go(func() {
			defer func() { <-slots }()
			log.Printf("building %s...", build.name)
			buildErrors[i] = goBuild(filepath.Join("..", build.name), build.output)
		})
	}
	workers.Wait()
	for i, err := range buildErrors {
		if err != nil {
			cleanup()
			log.Fatalf("build %s: %v", builds[i].name, err)
		}
	}

	// Every child binary inherits this environment, so its persisted state
	// (ports, workloads, settings) lands under tmpDir instead of the
	// developer's real config. These are the variables appdir resolves
	// through on each platform. They are set after the builds, because go
	// derives its module and build caches from these variables.
	// USERPROFILE is redirected too, so Windows engine manifests that
	// resolve vendor CLIs from the user profile, such as LM Studio's
	// lms.exe, cannot reach a real install.
	testsConfigBase = filepath.Join(tmpDir, "config")
	for _, key := range []string{"XDG_CONFIG_HOME", "HOME", "APPDATA", "LOCALAPPDATA", "USERPROFILE"} {
		if err := os.Setenv(key, testsConfigBase); err != nil {
			cleanup()
			log.Fatalf("set %s: %v", key, err)
		}
	}

	// macOS's default temporary prefix can make t.TempDir paths longer than
	// Unix socket addresses permit (notably cluster/reverse.sock). Keep test
	// fixture roots short and private, after builds have used the normal cache
	// environment. Other platforms retain their existing temporary directory.
	var shortTmpDir string
	if runtime.GOOS == "darwin" {
		shortTmpDir, err = os.MkdirTemp("/tmp", "nvp-")
		if err != nil {
			cleanup()
			log.Fatalf("create short test temporary root: %v", err)
		}
		if err := os.Setenv("TMPDIR", shortTmpDir); err != nil {
			os.RemoveAll(shortTmpDir)
			cleanup()
			log.Fatalf("set test temporary root: %v", err)
		}
		log.Printf("using private short test temporary root: %s", shortTmpDir)
	}

	code := m.Run()
	if shortTmpDir != "" {
		os.RemoveAll(shortTmpDir)
	}
	cleanup()
	os.Exit(code)
}

func goBuild(srcDir, output string) error {
	cmd := exec.Command("go", "build", "-o", output, ".")
	cmd.Dir = srcDir
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// Cluster config bases contain private Unix sockets. On Darwin, even a short
// TMPDIR cannot accommodate t.TempDir's full test-name prefix for long cases.
func shortClusterTempDir(t *testing.T) string {
	t.Helper()
	if runtime.GOOS != "darwin" {
		return t.TempDir()
	}
	dir, err := os.MkdirTemp("/tmp", "nvp-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Errorf("remove short cluster fixture directory: %v", err)
		}
	})
	return dir
}

// --- JSON-RPC helpers ---

// The on-wire JSON-RPC frame and error envelope are the shared
// nvpair-shared/jsonrpc types (jsonrpc.Message / jsonrpc.RPCError), so the
// integration tests parse exactly what production emits.

func startMsgReader(r io.Reader) <-chan jsonrpc.Message {
	ch := make(chan jsonrpc.Message, 64)
	go func() {
		defer close(ch)
		scanner := bufio.NewScanner(r)
		scanner.Buffer(make([]byte, 0, 256*1024), 256*1024)
		for scanner.Scan() {
			var msg jsonrpc.Message
			if json.Unmarshal(scanner.Bytes(), &msg) == nil {
				ch <- msg
			}
		}
	}()
	return ch
}

func waitForMethod(t *testing.T, ch <-chan jsonrpc.Message, method string, timeout time.Duration) jsonrpc.Message {
	t.Helper()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		select {
		case msg, ok := <-ch:
			if !ok {
				t.Fatalf("stream closed before receiving %q", method)
			}
			if msg.Method == method {
				return msg
			}
		case <-timer.C:
			t.Fatalf("timed out (%s) waiting for method %q", timeout, method)
		}
	}
}

func waitForResponse(t *testing.T, ch <-chan jsonrpc.Message, timeout time.Duration) jsonrpc.Message {
	t.Helper()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		select {
		case msg, ok := <-ch:
			if !ok {
				t.Fatal("stream closed before receiving response")
			}
			if msg.ID != nil && msg.Method == "" {
				return msg
			}
		case <-timer.C:
			t.Fatal("timed out waiting for JSON-RPC response")
		}
	}
}

// codeFacadeBindFailed mirrors the constant of the same name in nvpair-proxy:
// facade/enable answers with it when the port was taken before it could bind.
const codeFacadeBindFailed = -32010

// requestOnFreePort sends the request that build makes for a free port and
// returns the port the proxy accepted. freePort closes its probe before the
// proxy binds, so another process can take the port first. A request refused
// for that reason is retried on a new port; any other error fails the test.
func requestOnFreePort(t *testing.T, w io.Writer, msgs <-chan jsonrpc.Message, timeout time.Duration, build func(port int) map[string]any) int {
	t.Helper()
	for attempt := 0; attempt < 8; attempt++ {
		port := freePort(t)
		req := build(port)
		sendLine(t, w, req)
		resp := waitForResponse(t, msgs, timeout)
		if resp.Error == nil {
			return port
		}
		if !isBindRace(resp.Error) {
			t.Fatalf("%s failed: %v", req["method"], resp.Error)
		}
	}
	t.Fatal("every probed port was taken before the proxy could bind it")
	return 0
}

// isBindRace reports a request the proxy refused only because the port was
// taken. facade/enable says so with codeFacadeBindFailed; set-port has no
// dedicated code, so its bind error is matched by message.
func isBindRace(e *jsonrpc.RPCError) bool {
	return e.Code == codeFacadeBindFailed || strings.HasPrefix(e.Message, "failed to bind port")
}

// TestLMStudioFacadeChildPersistsUnderPrivateBase proves a real proxy child
// writes its persisted LM Studio port under the TestMain config base, so no
// cross-process test can clobber the developer's saved port.
func TestLMStudioFacadeChildPersistsUnderPrivateBase(t *testing.T) {
	cmd := exec.Command(proxyBin)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = stdin.Close()
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}()
	msgs := startMsgReader(stdout)

	requestOnFreePort(t, stdin, msgs, 10*time.Second, func(port int) map[string]any {
		return map[string]any{
			"jsonrpc": "2.0",
			"id":      1,
			"method":  "facade/enable",
			"params": map[string]any{
				"engine":              "lmstudio",
				"port":                port,
				"ignorePersistedPort": true,
			},
		}
	})

	persistedPort := requestOnFreePort(t, stdin, msgs, 5*time.Second, func(port int) map[string]any {
		return map[string]any{
			"jsonrpc": "2.0",
			"id":      2,
			"method":  engines.AddressMethod("lmstudio", "set-port"),
			"params":  map[string]any{"port": port},
		}
	})

	path, err := appdir.Path("lmstudio-proxy-port.json")
	if err != nil {
		t.Fatal(err)
	}
	rel, err := filepath.Rel(testsConfigBase, path)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		t.Fatalf("lmstudio port path %q is outside test config base %q", path, testsConfigBase)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read persisted lmstudio port: %v", err)
	}
	var saved struct {
		Port int `json:"port"`
	}
	if err := json.Unmarshal(data, &saved); err != nil {
		t.Fatalf("parse persisted lmstudio port: %v", err)
	}
	if saved.Port != persistedPort {
		t.Fatalf("persisted lmstudio port = %d, want %d", saved.Port, persistedPort)
	}
}
