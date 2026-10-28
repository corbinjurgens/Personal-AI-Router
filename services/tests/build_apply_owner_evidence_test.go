// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
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

	"nvpair-shared/buildapply"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/testbin"
)

// TestMain redirects profile variables after its product builds. Retain the
// original toolchain/cache environment for this fixture's later overlay build.
// Product and synthetic children receive only the private launch environment.
var ownerEvidenceBuildEnv = os.Environ()

// Dispatch copied package images before TestMain can build products or replace
// the child's private profile. These selectors exist only in this test package.
func init() {
	for _, role := range []string{"BROKER", "MANAGER", "ENGINE"} {
		if image := os.Getenv("NVPAIR_OWNER_" + role); image != "" && image == os.Args[0] {
			var err error
			if role == "ENGINE" {
				err = runOwnerEvidenceEngine()
			} else {
				err = runOwnerEvidenceBridge(role)
			}
			if err != nil {
				fmt.Fprintln(os.Stderr, "owner-evidence fixture:", err)
				os.Exit(1)
			}
			os.Exit(0)
		}
	}
}

type ownerBridgeStats struct {
	Requests map[string]int `json:"requests"`
	PID      int            `json:"pid"`
}

// The bridge forwards native frames without replacing results. Its private
// /call route is an additional parent of the real stdio transport, not a public
// service method. In particular it does NOT prove the service's private reader.
// Only the broker bridge admits build:apply-impact, never prepare or finish.
func runOwnerEvidenceBridge(role string) (resultErr error) {
	port, err := strconv.Atoi(os.Getenv("NVPAIR_OWNER_" + role + "_PORT"))
	if err != nil || validateFixturePort(port) != nil {
		return errors.New("bridge requires an allocated private port")
	}
	image := os.Getenv("NVPAIR_OWNER_" + role + "_REAL")
	if !filepath.IsAbs(image) || image == os.Args[0] {
		return errors.New("bridge requires its distinct fixed real image")
	}
	args := slices.Clone(os.Args[1:])
	if role == "MANAGER" {
		// The unused LAN model-list surface is wildcard-only. The existing
		// flags disable that listener and automatic engine watcher sweeps.
		args = append(args, "--http-port=0", "--loaded-poll-interval=0")
	}
	cmd := exec.Command(image, args...)
	cmd.Env, cmd.Stderr = os.Environ(), os.Stderr
	in, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return err
	}
	defer listener.Close()
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
	var mu, writeMu sync.Mutex
	stats := ownerBridgeStats{Requests: map[string]int{}, PID: cmd.Process.Pid}
	pending := map[string]chan *jsonrpc.Message{}
	var sequence int
	encoder := json.NewEncoder(in)
	write := func(frame *jsonrpc.Message) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		mu.Lock()
		stats.Requests[frame.Method]++
		mu.Unlock()
		return encoder.Encode(frame)
	}
	stopped := make(chan struct{})
	defer close(stopped)
	mux := http.NewServeMux()
	mux.HandleFunc("/stats", func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if err := json.NewEncoder(w).Encode(stats); err != nil {
			return
		}
	})
	mux.HandleFunc("/call", func(w http.ResponseWriter, r *http.Request) {
		var frame jsonrpc.Message
		if r.Method != http.MethodPost || json.NewDecoder(io.LimitReader(r.Body, jsonrpc.WorkerFrameBytes)).Decode(&frame) != nil ||
			(role == "BROKER" && frame.Method != "build:apply-impact") ||
			(role == "MANAGER" && frame.Method != buildapply.EngineOwnerEvidenceMethod) {
			http.Error(w, "private read only", http.StatusBadRequest)
			return
		}
		mu.Lock()
		sequence++
		// Public clients use numeric service IDs. This string namespace never
		// escapes into the service's pending map or consumes its replies.
		id := json.RawMessage(strconv.Quote(fmt.Sprintf("owner-observer-%d", sequence)))
		ch := make(chan *jsonrpc.Message, 1)
		pending[string(id)] = ch
		mu.Unlock()
		defer func() { mu.Lock(); delete(pending, string(id)); mu.Unlock() }()
		frame.JSONRPC, frame.ID = "2.0", &id
		if err := write(&frame); err != nil {
			http.Error(w, "native transport unavailable", http.StatusServiceUnavailable)
			return
		}
		select {
		case reply := <-ch:
			if err := json.NewEncoder(w).Encode(reply); err != nil {
				return
			}
		case <-stopped:
			http.Error(w, "native child departed", http.StatusServiceUnavailable)
		case <-r.Context().Done():
		}
	})
	server := &http.Server{Handler: mux, ReadHeaderTimeout: serviceCallTimeout}
	defer server.Close()
	done := make(chan error, 3)
	go func() { done <- server.Serve(listener) }()
	go func() {
		codec := jsonrpc.NewCodecMaxFrame(struct {
			io.Reader
			io.Writer
		}{os.Stdin, io.Discard}, jsonrpc.WorkerFrameBytes)
		for {
			frame, err := codec.Read()
			if err != nil {
				done <- err
				return
			}
			if err := write(frame); err != nil {
				done <- err
				return
			}
		}
	}()
	go func() {
		codec := jsonrpc.NewCodecMaxFrame(struct {
			io.Reader
			io.Writer
		}{out, io.Discard}, jsonrpc.WorkerFrameBytes)
		forward := json.NewEncoder(os.Stdout)
		for {
			frame, err := codec.Read()
			if err != nil {
				done <- err
				return
			}
			var ch chan *jsonrpc.Message
			if frame.IsResponse() {
				mu.Lock()
				ch = pending[string(*frame.ID)]
				mu.Unlock()
				// A cancelled private read remains private even after retirement.
				if strings.HasPrefix(string(*frame.ID), `"owner-observer-`) {
					if ch != nil {
						ch <- frame
					}
					continue
				}
			}
			if err := forward.Encode(frame); err != nil {
				done <- err
				return
			}
		}
	}()
	if err := <-done; err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	if err := in.Close(); err != nil {
		return err
	}
	err = cmd.Wait()
	reaped = true
	return err
}

// A pure Go fake records every launch and every HTTP contact at the parent.
// No model bytes, credential store, inference or child processes exist.
func runOwnerEvidenceEngine() error {
	control := os.Getenv("NVPAIR_OWNER_CONTROL")
	client := &http.Client{Timeout: 2 * time.Minute, Transport: &http.Transport{Proxy: nil}}
	defer client.CloseIdleConnections()
	record := func(path string) error {
		response, err := client.Post(control+path, "application/json", bytes.NewReader([]byte(`{}`)))
		if err != nil {
			return err
		}
		defer response.Body.Close()
		if response.StatusCode != http.StatusOK {
			return errors.New("fake observation rejected")
		}
		return nil
	}
	if len(os.Args) == 2 && os.Args[1] == "pull" {
		if err := record("/pull"); err != nil {
			return err
		}
		return errors.New("controlled synthetic download failure")
	}
	port, err := strconv.Atoi(os.Getenv("NVPAIR_OWNER_ENGINE_PORT"))
	if err != nil || validateFixturePort(port) != nil {
		return errors.New("fake engine requires an allocated private port")
	}
	if err := record("/started"); err != nil {
		return err
	}
	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return err
	}
	defer listener.Close()
	return http.Serve(listener, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := record("/contact"); err != nil {
			http.Error(w, "observation unavailable", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"status":"ok","data":[]}`)
	}))
}

type ownerEvidenceProcessFixture struct {
	*serviceFixture
	brokerURL, managerURL    string
	managerImage, fake       string
	enginePorts              map[string]int
	client                   *http.Client
	mu                       sync.Mutex
	starts, contacts         int
	pullEntered, pullRelease chan struct{}
	releasePull              func()
}

// Generated overlay sources are disposable test inputs outside the checkout.
// Like managerFixtureOverlay in engine-manager/main_test.go, this changes only
// registry construction and the credential backend. Evidence production,
// manager identity, owner state, transport and lifecycle remain real.
func ownerEvidenceManagerOverlay(t *testing.T, dir, fake string) string {
	t.Helper()
	replacements := map[string]string{}
	replace := func(name, site, replacement string) {
		t.Helper()
		source, err := filepath.Abs(filepath.Join("..", "nvpair-engine-manager", name))
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(source)
		if err != nil {
			t.Fatal(err)
		}
		text := strings.ReplaceAll(string(data), "\r\n", "\n")
		if strings.Count(text, site) != 1 {
			t.Fatalf("owner fixture overlay site changed in %s", name)
		}
		target := filepath.Join(dir, name)
		if err := os.WriteFile(target, []byte(strings.Replace(text, site, replacement, 1)), 0o600); err != nil {
			t.Fatal(err)
		}
		replacements[source] = target
	}
	replace("main.go", "reg := buildRegistry(manifestDir)", `reg := NewRegistry()
	if err := reg.LoadOverrideDir(manifestDir); err != nil { log.Fatalf("private registry: %v", err) }`)
	replace("credentialstore.go", "var newCredentialBackend = func() credentialBackend { return secretstore.New() }", `var newCredentialBackend = func() credentialBackend { return &ownerFixtureCredentials{} }
type ownerFixtureCredentials struct{}
func (*ownerFixtureCredentials) Set(context.Context, string, string, []byte) error { return secretstore.ErrNotFound }
func (*ownerFixtureCredentials) Get(context.Context, string, string) ([]byte, error) { return nil, secretstore.ErrNotFound }
func (*ownerFixtureCredentials) Delete(context.Context, string, string) error { return secretstore.ErrNotFound }`)
	// This observation is after actual lazy creation, not an inferred absence
	// from PID snapshots or a substituted owner-evidence result.
	replace("executor.go", "e.engines[engine] = st", fmt.Sprintf(`e.engines[engine] = st
	if err := os.WriteFile(filepath.Join(%q, "initialized-"+engine), []byte("created"), 0600); err != nil { panic(err) }`, dir))
	// The pure Go fake never has descendants. Terminate that exact retained
	// process handle and join the real managedProc.done channel, avoiding an
	// unenumerated taskkill child. No owner result or stop proof is substituted.
	// The ordinary lifecycle owner still saves Off, observes exit and emits it.
	replace("proc.go", `force := grace <= 0
	_ = signalPIDContext(ctx, mp.cmd.Process.Pid, force)
	if err := ctx.Err(); err != nil {
		return err
	}
	if !force {
		timer := time.NewTimer(grace)
		defer timer.Stop()
		select {
		case <-mp.done:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		case <-timer.C:
		}
		_ = signalPIDContext(ctx, mp.cmd.Process.Pid, true)
	}
	select {
	case <-mp.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}`, fmt.Sprintf(`if !strings.EqualFold(filepath.Clean(mp.cmd.Path), filepath.Clean(%q)) { return os.ErrPermission }
	if err := mp.cmd.Process.Kill(); err != nil { return err }
	select { case <-mp.done: return nil; case <-ctx.Done(): return ctx.Err() }`, fake))
	path := filepath.Join(dir, "overlay.json")
	jointWriteJSON(t, path, map[string]map[string]string{"Replace": replacements})
	return path
}

func newOwnerEvidenceProcessFixture(t *testing.T) *ownerEvidenceProcessFixture {
	t.Helper()
	if runtime.GOOS != "windows" || !filepath.IsAbs(os.Getenv(testbin.EnvRoot)) {
		t.Fatal("owner process proof requires Windows and the fixed testbin root; compile only elsewhere")
	}
	f := &ownerEvidenceProcessFixture{serviceFixture: newServiceFixture(t), enginePorts: map[string]int{},
		pullEntered: make(chan struct{}, 1), pullRelease: make(chan struct{})}
	f.releasePull = sync.OnceFunc(func() { close(f.pullRelease) })
	f.client = &http.Client{Timeout: 20 * time.Second, Transport: &http.Transport{Proxy: nil}}
	t.Cleanup(f.client.CloseIdleConnections)
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	f.copyExecutable(self, f.scannerPath)
	f.env = ownerEvidencePrivateEnv(f.env)
	f.env = append(f.env, "NVPAIR_TEST_SUBSCRIPTION_SCANNER="+f.scannerPath,
		"NVPAIR_TEST_SUBSCRIPTION_PORT="+strconv.Itoa(temporaryClusterPort(t)))
	control := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/pull" {
			select {
			case f.pullEntered <- struct{}{}:
			default:
			}
			select {
			case <-f.pullRelease:
			case <-r.Context().Done():
			}
			return
		}
		f.mu.Lock()
		defer f.mu.Unlock()
		switch r.URL.Path {
		case "/started":
			f.starts++
		case "/contact":
			f.contacts++
		default:
			http.Error(w, "unknown observation", http.StatusNotFound)
		}
	}))
	t.Cleanup(control.Close)
	t.Cleanup(f.releasePull)
	// All children reuse previously enumerated destinations. The native broker
	// in products/ must receive explicit absent optional-worker paths, otherwise
	// it could find every TestMain product beside its own executable.
	bridge := filepath.Join(testbin.Dir(t, "tests", "model-settings", "model-settings-manager"), testbin.Exe("model-settings-manager"))
	f.fake = filepath.Join(testbin.Dir(t, "tests", "policy-joint", "joint-origin-engine"), testbin.Exe("joint-origin-engine"))
	f.managerImage = filepath.Join(f.binDir, testbin.Exe("nvpair-engine-manager"))
	for _, path := range []string{bridge, f.fake, brokerBin} {
		canonical, err := serviceExecutablePath(path)
		if err != nil {
			t.Fatal(err)
		}
		f.inventory.bundlePaths = append(f.inventory.bundlePaths, canonical)
	}
	bridge = f.copyExecutable(self, bridge)
	f.fake = f.copyExecutable(self, f.fake)
	f.copyExecutable(self, f.brokerPath)
	overlay := ownerEvidenceManagerOverlay(t, f.dir, f.fake)
	if err := f.stageExecutable(func() error {
		cmd := exec.Command("go", "build", "-overlay", overlay, "-o", f.managerImage, ".")
		cmd.Dir = filepath.Join("..", "nvpair-engine-manager")
		cmd.Env = ownerEvidenceBuildEnv
		cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
		return cmd.Run()
	}); err != nil {
		t.Fatalf("build private manager: %v", err)
	}
	f.inventory.paths = append(f.inventory.paths, f.managerImage)
	brokerPort, managerPort := freePort(t), freePort(t)
	f.brokerURL = fmt.Sprintf("http://127.0.0.1:%d", brokerPort)
	f.managerURL = fmt.Sprintf("http://127.0.0.1:%d", managerPort)
	f.env = append(f.env, "NVPAIR_OWNER_BROKER="+f.brokerPath, "NVPAIR_OWNER_BROKER_REAL="+brokerBin,
		"NVPAIR_OWNER_BROKER_PORT="+strconv.Itoa(brokerPort), "NVPAIR_OWNER_MANAGER="+bridge,
		"NVPAIR_OWNER_MANAGER_REAL="+f.managerImage, "NVPAIR_OWNER_MANAGER_PORT="+strconv.Itoa(managerPort),
		"NVPAIR_OWNER_ENGINE="+f.fake, "NVPAIR_OWNER_CONTROL="+control.URL)
	f.additionalBrokerArgs = []string{"--engine-manager-path", bridge, "--engine-manager-control-host", "127.0.0.1", "--proxy-path", filepath.Join(f.dir, "absent-proxy")}
	for _, flag := range []string{"node-info", "workload-manager", "errors", "manual-nodes", "settings", "cluster-manager", "scheduler"} {
		f.additionalBrokerArgs = append(f.additionalBrokerArgs, "--"+flag+"-path", filepath.Join(f.dir, "absent-"+flag))
	}
	// newServiceFixture installed inert overrides for bundled engines. An empty
	// compiled registry plus only these explicit manifests is the whole universe.
	manifestDir := filepath.Join(jointAppDir(f.configDir), "engines")
	entries, err := os.ReadDir(manifestDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if err := os.Remove(filepath.Join(manifestDir, entry.Name())); err != nil {
			t.Fatal(err)
		}
	}
	for _, engine := range []string{"owner-never", "owner-owned", "owner-external"} {
		port := freePort(t)
		f.enginePorts[engine] = port
		probe := map[string]any{"http": "http://127.0.0.1:{port}/health", "status": http.StatusOK, "timeout_s": 5, "interval_s": 60}
		rt := map[string]any{"mode": "process", "bin": f.fake, "bind": "127.0.0.1", "port": port,
			"env": map[string]string{"NVPAIR_OWNER_ENGINE_PORT": "{port}"}, "ready": probe, "health": probe,
			"stop": map[string]any{"signal": "term", "grace_s": 1}}
		jointWriteJSON(t, filepath.Join(manifestDir, engine+".json"), map[string]any{"engine": engine, "display_name": engine, "manifest_version": 1, "model_naming": "exact",
			"operations": map[string]any{"pull": map[string]string{"action": "pull_model", "model_param": "name"}},
			"actions":    map[string]any{"pull_model": map[string]any{"cmd": []string{f.fake, "pull"}}},
			"platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{
				"detect": []string{f.fake}, "models_dir": filepath.Join(f.configDir, "models", engine), "runtime": rt}}})
	}
	// Private preexisting uncertainty is input to the real durable owner. It is
	// not a forged evidence result or a claim of historical completion.
	jointWriteJSON(t, filepath.Join(jointAppDir(f.configDir), "engine-bin", "model-download-state.json"), map[string]any{
		"version": 1, "entries": []map[string]string{{"id": strings.Repeat("d", 32), "engine": "owner-owned",
			"model": "predecessor-download", "contentKey": "predecessor-download", "naming": "exact"}},
		"copies": []any{}, "copiesComplete": false,
	})
	return f
}

// Keep the Go toolchain environment for the overlay build, but never pass
// ambient proxy, engine, credential, preset or test-child selectors to products.
func ownerEvidencePrivateEnv(env []string) []string {
	var clean []string
	for _, entry := range env {
		key, _, ok := strings.Cut(entry, "=")
		upper := strings.ToUpper(key)
		if !ok || strings.Contains(upper, "PROXY") || strings.Contains(upper, "TOKEN") || strings.Contains(upper, "SECRET") ||
			strings.Contains(upper, "PASSWORD") || strings.Contains(upper, "KEY") || strings.Contains(upper, "CREDENTIAL") || strings.Contains(upper, "AUTH") ||
			strings.HasPrefix(upper, "NVPAIR_TEST_") || strings.HasPrefix(upper, "NVPAIR_OWNER_") || strings.HasPrefix(upper, "FAKE_") ||
			strings.HasPrefix(upper, "LLAMA") || strings.HasPrefix(upper, "OLLAMA") || strings.HasPrefix(upper, "STRATA") ||
			strings.HasPrefix(upper, "LMSTUDIO") || strings.HasPrefix(upper, "PAIR_BUILD_APPLY") || strings.HasPrefix(upper, "NVPAIR_BUILD_APPLY_") ||
			strings.HasPrefix(upper, "AWS_") || strings.HasPrefix(upper, "AZURE_") || strings.HasPrefix(upper, "GOOGLE_") || strings.HasPrefix(upper, "HF_") {
			continue
		}
		clean = append(clean, entry)
	}
	return clean
}

func (f *ownerEvidenceProcessFixture) read(t *testing.T, manager bool) (*jsonrpc.Message, error) {
	t.Helper()
	base, method := f.brokerURL, "build:apply-impact"
	if manager {
		base, method = f.managerURL, buildapply.EngineOwnerEvidenceMethod
	}
	request, err := http.NewRequest(http.MethodPost, base+"/call", bytes.NewReader(mustMarshal(t, jsonrpc.Message{
		JSONRPC: "2.0", Method: method, Params: json.RawMessage(`{}`),
	})))
	if err != nil {
		return nil, err
	}
	response, err := f.client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bridge read HTTP %d", response.StatusCode)
	}
	var frame jsonrpc.Message
	err = json.NewDecoder(io.LimitReader(response.Body, jsonrpc.WorkerFrameBytes)).Decode(&frame)
	return &frame, err
}

func (f *ownerEvidenceProcessFixture) impact(t *testing.T) buildapply.ImpactEnvelope {
	t.Helper()
	deadline := time.NewTimer(30 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(25 * time.Millisecond)
	defer tick.Stop()
	var lastErr error
	for {
		frame, err := f.read(t, false)
		if err == nil && frame.Error == nil {
			var result buildapply.Result[buildapply.ImpactEnvelope]
			if err := json.Unmarshal(frame.Result, &result); err != nil {
				t.Fatal(err)
			}
			if result.Available && result.Value != nil && result.Value.ManagerEvidence.Known && result.Value.ManagerEvidence.Value != nil {
				owner := result.Value.ManagerEvidence.Value
				if owner.Universe.Known && owner.StopAuthority.Known && owner.Reservations.Known {
					return *result.Value
				}
			}
		}
		lastErr = err
		select {
		case <-deadline.C:
			t.Fatalf("native broker/manager owner capture unavailable: %v", lastErr)
		case <-tick.C:
		}
	}
}

func (f *ownerEvidenceProcessFixture) stats(t *testing.T, base string) ownerBridgeStats {
	t.Helper()
	response, err := f.client.Get(base + "/stats")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var stats ownerBridgeStats
	if response.StatusCode != http.StatusOK {
		t.Fatalf("bridge stats HTTP %d", response.StatusCode)
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 64*1024)).Decode(&stats); err != nil {
		t.Fatal(err)
	}
	return stats
}

func ownerEvidenceEntry(t *testing.T, e buildapply.EngineOwnerEvidence, engine string) (buildapply.EngineOwnerUniverseEntry, buildapply.EngineOwnerStopAuthority) {
	t.Helper()
	if !e.Universe.Known || e.Universe.Value == nil || !e.StopAuthority.Known || e.StopAuthority.Value == nil {
		t.Fatal("native universe or authority unavailable")
	}
	if len(*e.Universe.Value) != len(*e.StopAuthority.Value) {
		t.Fatal("native authority and universe lengths differ")
	}
	for i, entry := range *e.Universe.Value {
		if entry.Engine == engine {
			authority := (*e.StopAuthority.Value)[i]
			if authority.Engine != engine {
				t.Fatal("native authority not paired with universe")
			}
			return entry, authority
		}
	}
	t.Fatalf("native universe omitted %s", engine)
	return buildapply.EngineOwnerUniverseEntry{}, buildapply.EngineOwnerStopAuthority{}
}

// This target deliberately reports only the reachable process assertions.
// Service-consumer and delayed predecessor-delivery cases are not executed by
// this fixture and must not be counted as Task 6 acceptance.
func TestBuildApplyOwnerEvidenceProcess(t *testing.T) {
	before, err := snapshotInstalledServiceProcesses()
	if err != nil {
		t.Fatal(err)
	}
	f := newOwnerEvidenceProcessFixture(t)
	t.Cleanup(func() {
		if err := f.cleanupChecked(); err != nil {
			t.Error(err)
			return
		}
		after, err := snapshotInstalledServiceProcesses()
		if err != nil {
			t.Error(err)
			return
		}
		if !reflect.DeepEqual(before, after) {
			t.Errorf("installed identity/listener snapshot changed: before=%+v after=%+v", before, after)
		}
	})
	f.start()
	c := f.readyClient()
	var first buildapply.ImpactEnvelope
	t.Run("idle native broker manager read preserves never observed unknown", func(t *testing.T) {
		initializedBefore, err := filepath.Glob(filepath.Join(f.dir, "initialized-*"))
		if err != nil {
			t.Fatal(err)
		}
		first = f.impact(t)
		e := *first.ManagerEvidence.Value
		second := f.impact(t)
		if second.ManagerEvidence.Value.ManagerIncarnation != e.ManagerIncarnation || second.ManagerEvidence.Value.CaptureToken != e.CaptureToken {
			t.Fatal("stable idle captures changed native manager identity or evidence token")
		}
		frame, err := f.read(t, true)
		if err != nil || frame.Error != nil {
			t.Fatalf("direct native owner read: frame=%+v error=%v", frame, err)
		}
		var native buildapply.EngineOwnerEvidence
		if err := json.Unmarshal(frame.Result, &native); err != nil {
			t.Fatal(err)
		}
		if native.ManagerIncarnation != e.ManagerIncarnation || native.CaptureToken != e.CaptureToken {
			t.Fatal("broker source fence differs from the actual native owner document")
		}
		if len(e.ManagerIncarnation) != 32 || e.CaptureToken == "" || e.Complete || first.Impact.Complete {
			t.Fatal("idle capture lacks native identity or claims completeness")
		}
		entry, authority := ownerEvidenceEntry(t, e, "owner-never")
		if !entry.Registered || !entry.Supported.Known || entry.Supported.Value == nil || !*entry.Supported.Value ||
			entry.Installed.Known || entry.Running.Known || entry.TerminationVerified.Known || authority.Scoped.Known || entry.RootsComplete {
			t.Fatal("never-observed supported entry fabricated lifecycle, stop authority or root coverage")
		}
		var names []string
		for _, member := range *e.Universe.Value {
			names = append(names, member.Engine)
			for _, roots := range []buildapply.Observation[[]string]{member.ModelRoots, member.DataRoots} {
				if !roots.Known || roots.Value == nil {
					t.Fatal("explicit private fixture roots disappeared")
				}
				for _, root := range *roots.Value {
					rel, err := filepath.Rel(f.configDir, root)
					if err != nil || !filepath.IsAbs(root) || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
						t.Fatalf("native root escaped private profile: %q error=%v", root, err)
					}
				}
			}
		}
		slices.Sort(names)
		if !slices.Equal(names, []string{"owner-external", "owner-never", "owner-owned"}) {
			t.Fatalf("compiled registry was not restricted to private manifests: %v", names)
		}
		if !e.Reservations.Known || e.Reservations.Value == nil || e.Reservations.Value.UnsafeCopiesComplete {
			t.Fatal("missing historical copy baseline became safe or disappeared")
		}
		initializedAfter, err := filepath.Glob(filepath.Join(f.dir, "initialized-*"))
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(initializedBefore, initializedAfter) {
			t.Fatal("idle evidence lazily initialized an engine")
		}
		f.mu.Lock()
		defer f.mu.Unlock()
		if f.starts != 0 || f.contacts != 0 {
			t.Fatalf("idle read launched/contacted fake engines: starts=%d contacts=%d", f.starts, f.contacts)
		}
	})
	if first.ManagerEvidence.Value == nil {
		t.Fatal("idle capture prerequisite failed")
	}
	t.Run("public relay does not disclose private owner documents", func(t *testing.T) {
		for i, method := range []string{buildapply.EngineOwnerEvidenceMethod, "build:apply-impact"} {
			frame, err := c.call(strconv.Itoa(800+i), method, json.RawMessage(`{}`), serviceCallTimeout)
			if err != nil || frame.Error == nil || frame.Error.Code != -32601 || len(frame.Result) != 0 {
				t.Fatalf("private relay method %s: frame=%+v error=%v", method, frame, err)
			}
		}
	})
	t.Run("owned fake has a retained process lifetime", func(t *testing.T) {
		c.mustCall(t, "810", "engine:start", mustMarshal(t, map[string]string{"engine": "owner-owned"}))
		e := *f.impact(t).ManagerEvidence.Value
		entry, authority := ownerEvidenceEntry(t, e, "owner-owned")
		if !entry.Running.Known || entry.Running.Value == nil || !*entry.Running.Value ||
			!authority.Scoped.Known || authority.Scoped.Value == nil || !*authority.Scoped.Value ||
			!authority.Adopted.Known || authority.Adopted.Value == nil || *authority.Adopted.Value ||
			!authority.ProcessLifetime.Known || authority.ProcessLifetime.Value == nil || authority.ProcessLifetime.Value.EngineRunID == "" {
			t.Fatal("owned fake lacks native scoped process authority")
		}
		lifetime := authority.ProcessLifetime.Value
		if !lifetime.PID.Known || lifetime.PID.Value == nil || !entry.EngineRunID.Known || entry.EngineRunID.Value == nil || lifetime.EngineRunID != *entry.EngineRunID.Value {
			t.Fatal("native owned lifetime has no paired PID and engine run")
		}
		found := slices.ContainsFunc(f.scan(), func(identity serviceProcessIdentity) bool {
			return int(identity.PID) == *lifetime.PID.Value && identity.Path == f.fake
		})
		if !found {
			t.Fatal("owner's retained process is not the fixed synthetic image")
		}
		c.mustCall(t, "811", "engine:stop", mustMarshal(t, map[string]string{"engine": "owner-owned"}))
	})
	t.Run("external fake adoption does not grant process stop authority", func(t *testing.T) {
		cmd := exec.Command(f.fake)
		cmd.Env = append(slices.Clone(f.env), "NVPAIR_OWNER_ENGINE_PORT="+strconv.Itoa(f.enginePorts["owner-external"]))
		child := f.launchChild(cmd)
		// A real start performs the deliberate adoption. Evidence reads never
		// call Status, Detect or a native HTTP endpoint to fill missing facts.
		waitTCP(t, fmt.Sprintf("127.0.0.1:%d", f.enginePorts["owner-external"]), serviceCallTimeout)
		c.mustCall(t, "812", "engine:start", mustMarshal(t, map[string]string{"engine": "owner-external"}))
		entry, authority := ownerEvidenceEntry(t, *f.impact(t).ManagerEvidence.Value, "owner-external")
		if !entry.Running.Known || entry.Running.Value == nil || !*entry.Running.Value ||
			!authority.Adopted.Known || authority.Adopted.Value == nil || !*authority.Adopted.Value ||
			!authority.Scoped.Known || authority.Scoped.Value == nil || *authority.Scoped.Value || authority.ProcessLifetime.Known {
			t.Fatal("adopted external fake received owned-process authority")
		}
		var external serviceProcessIdentity
		for _, identity := range f.scan() {
			if int(identity.PID) == child.cmd.Process.Pid && identity.Path == f.fake {
				external = identity
			}
		}
		if external.PID == 0 {
			t.Fatal("external fake lacks its recorded fixed-image identity")
		}
		refused, err := c.call("813", "engine:stop", mustMarshal(t, map[string]string{"engine": "owner-external"}), serviceCallTimeout)
		if err != nil || refused.Error == nil {
			t.Fatalf("external stop did not refuse: reply=%+v error=%v", refused, err)
		}
		current, err := serviceProcessIdentityAt(t.Context(), external.PID)
		if err != nil || current != external {
			t.Fatalf("manager stopped or replaced the external fake: identity=%+v error=%v", current, err)
		}
		if err := f.inventory.kill(external); err != nil {
			t.Fatal(err)
		}
		f.waitChild(child, serviceCallTimeout)
	})
	t.Run("retained unsafe download blocks broker preflight projection", func(t *testing.T) {
		impact := f.impact(t)
		r := impact.ManagerEvidence.Value.Reservations
		if !r.Known || r.Value == nil || !r.Value.BlocksApply.Known || r.Value.BlocksApply.Value == nil || !*r.Value.BlocksApply.Value {
			t.Fatal("retained unsafe download did not block apply")
		}
		found := false
		for _, op := range r.Value.Operations {
			if op.ID == strings.Repeat("d", 32) {
				found = op.Engine == "owner-owned" && op.Kind == "download" && op.Retained && op.Unsafe && !op.SourceIncarnation.Known
			}
		}
		busy := slices.ContainsFunc(impact.Impact.Refusals, func(r buildapply.Refusal) bool { return r.Code == "operation_busy" })
		if !found || !busy || impact.Impact.Complete {
			t.Fatal("unsafe provenance or broker refusal was lost")
		}
	})
	t.Run("active real reservation blocks before the controlled fake effect completes", func(t *testing.T) {
		defer f.releasePull()
		if err := c.request("814", "engine:action", mustMarshal(t, map[string]any{"engine": "owner-owned", "action": "pull_model", "params": map[string]string{"name": "active-download"}})); err != nil {
			t.Fatal(err)
		}
		select {
		case <-f.pullEntered:
		case <-time.After(30 * time.Second):
			t.Fatal("real download owner did not reach the controlled fixed fake")
		}
		impact := f.impact(t)
		r := impact.ManagerEvidence.Value.Reservations
		if !r.Known || r.Value == nil || !r.Value.BlocksApply.Known || r.Value.BlocksApply.Value == nil || !*r.Value.BlocksApply.Value {
			t.Fatal("active reservation did not block apply")
		}
		active := slices.ContainsFunc(r.Value.Operations, func(op buildapply.EngineOwnerOperation) bool {
			return op.Engine == "owner-owned" && op.Kind == "content_writer" && !op.Retained && op.SourceIncarnation.Known && op.SourceIncarnation.Value != nil && *op.SourceIncarnation.Value == impact.ManagerEvidence.Value.ManagerIncarnation
		})
		if !active || !slices.ContainsFunc(impact.Impact.Refusals, func(r buildapply.Refusal) bool { return r.Code == "operation_busy" }) {
			t.Fatal("active operation lost native source or broker refusal")
		}
		f.releasePull()
		frame, err := c.response("814", serviceCallTimeout)
		if err != nil || frame.Error == nil {
			t.Fatalf("controlled failed effect: reply=%+v error=%v", frame, err)
		}
	})
	t.Run("refused preview and start do not send lifecycle commands", func(t *testing.T) {
		beforeBroker, beforeManager := f.stats(t, f.brokerURL), f.stats(t, f.managerURL)
		preview := c.mustCall(t, "820", "service/build-apply-preview", mustMarshal(t, buildapply.PreviewRequest{
			CandidatePath: filepath.Join(f.dir, "absent-candidate"),
			GUI:           buildapply.GUIFacts{PID: os.Getpid(), Executable: os.Args[0]},
		}))
		var result buildapply.Result[buildapply.BuildApplyPreview]
		if err := json.Unmarshal(preview.Result, &result); err != nil {
			t.Fatal(err)
		}
		if result.Available || result.Value != nil || result.Refusal == nil || result.Refusal.Code != "unsafe_package" {
			t.Fatal("nonpackaged fixture unexpectedly produced an eligible preview")
		}
		start := c.mustCall(t, "821", "service/build-apply-start", mustMarshal(t, buildapply.StartRequest{
			PreviewID: strings.Repeat("a", 32), Consent: buildapply.Consent{AllowEngineStop: true, AllowInterruption: true},
			GUI: buildapply.GUIFacts{PID: os.Getpid(), Executable: os.Args[0]},
		}))
		var operation buildapply.Result[buildapply.BuildApplyOperation]
		if err := json.Unmarshal(start.Result, &operation); err != nil {
			t.Fatal(err)
		}
		if operation.Available || operation.Value != nil || operation.Refusal == nil || operation.Refusal.Code != "stale_preview" {
			t.Fatal("missing preview unexpectedly admitted a build transaction")
		}
		afterBroker, afterManager := f.stats(t, f.brokerURL), f.stats(t, f.managerURL)
		for _, method := range []string{"shutdown", "engine:prepare-shutdown", "engine:stop", "build:apply-prepare", "build:apply-finish"} {
			if beforeBroker.Requests[method] != afterBroker.Requests[method] || beforeManager.Requests[method] != afterManager.Requests[method] {
				t.Fatalf("refused operation sent %s", method)
			}
		}
	})
	t.Run("manager restart replaces native incarnation and withdraws prior running facts", func(t *testing.T) {
		previous := *f.impact(t).ManagerEvidence.Value
		stats := f.stats(t, f.managerURL)
		var identity serviceProcessIdentity
		for _, candidate := range f.scan() {
			if int(candidate.PID) == stats.PID && candidate.Path == f.managerImage {
				identity = candidate
			}
		}
		if identity.PID == 0 {
			t.Fatal("native manager lacks its fixed-image process identity")
		}
		if err := f.inventory.kill(identity); err != nil {
			t.Fatal(err)
		}
		next := *f.impact(t).ManagerEvidence.Value
		if next.ManagerIncarnation == previous.ManagerIncarnation || next.CaptureToken == previous.CaptureToken {
			t.Fatal("replacement manager reused predecessor authority")
		}
		entry, authority := ownerEvidenceEntry(t, next, "owner-never")
		if entry.Running.Known || entry.TerminationVerified.Known || authority.Scoped.Known {
			t.Fatal("replacement fabricated predecessor stop knowledge")
		}
		owned, stop := ownerEvidenceEntry(t, next, "owner-owned")
		if owned.Running.Known || owned.TerminationVerified.Known || stop.Scoped.Known || stop.ProcessLifetime.Known {
			t.Fatal("replacement retained predecessor owned-process or stopped authority")
		}
	})
	if err := f.cleanupChecked(); err != nil {
		t.Fatal(err)
	}
	f.assertStopped()
	t.Log("Unreached: service-owned idle read and reservation preflight, delayed predecessor delivery, swap/runner invocation counters and firewall snapshots. Broker reservation projections are covered; do not treat this target as Task 6 acceptance.")
}
