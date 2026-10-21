// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestE2EOverStdio drives the real engine-manager binary end-to-end
// over JSON-RPC stdio: it discovers an injected "fake" engine, starts
// it (spawning the fake-engine child), runs an action, stops it, and
// shuts down — the same path the supervising broker uses.
func TestE2EOverStdio(t *testing.T) {
	cfg := t.TempDir()
	home := t.TempDir()
	// Drop the test manifest in every location os.UserConfigDir might
	// resolve to, so the child finds it regardless of OS.
	for _, dir := range []string{
		filepath.Join(cfg, "Nvidia Corporation", "Personal AI Router", "engines"),                                    // Windows %LocalAppData%, Linux $XDG_CONFIG_HOME
		filepath.Join(home, "Library", "Application Support", "Nvidia Corporation", "Personal AI Router", "engines"), // macOS
	} {
		writeFakeManifest(t, dir)
	}

	cmd := exec.Command(managerBin)
	cmd.Env = overrideEnv(map[string]string{"APPDATA": cfg, "LOCALAPPDATA": cfg, "XDG_CONFIG_HOME": cfg, "HOME": home})
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

	frames := make(chan frame, 128)
	go readFrames(stdout, frames)

	waitNotify(t, frames, "engine:ready", 5*time.Second)

	send(t, stdin, 1, "engine:get-installed", nil)
	if r := waitResult(t, frames, "1", 5*time.Second); !strings.Contains(string(r), `"engine":"fake"`) {
		t.Fatalf("get-installed did not list the injected engine: %s", r)
	}

	send(t, stdin, 2, "engine:start", map[string]any{"engine": "fake"})
	if r := waitResult(t, frames, "2", 20*time.Second); !strings.Contains(string(r), `"running":true`) {
		t.Fatalf("start did not report running: %s", r)
	}

	send(t, stdin, 3, "engine:action", map[string]any{"engine": "fake", "action": "list_models"})
	if r := waitResult(t, frames, "3", 10*time.Second); !strings.Contains(string(r), "llama3.2") {
		t.Fatalf("action result unexpected: %s", r)
	}

	send(t, stdin, 4, "engine:stop", map[string]any{"engine": "fake"})
	if r := waitResult(t, frames, "4", 10*time.Second); !strings.Contains(string(r), `"running":false`) {
		t.Fatalf("stop did not report stopped: %s", r)
	}

	send(t, stdin, 5, "shutdown", nil)
	waitResult(t, frames, "5", 5*time.Second)
}

// Saving a bundled engine's port through the real worker must preserve the
// rest of its override across a new worker process, not just in-memory state.
func TestE2EPortSavePreservesLaunchOverrides(t *testing.T) {
	cfg, home := t.TempDir(), t.TempDir()
	override := map[string]any{
		"engine": "ollama",
		"runtime": map[string]any{
			"args": []string{"serve", "--custom-option"},
			"env":  map[string]string{"CUSTOM_SETTING": "retained"},
		},
	}
	for _, dir := range []string{
		filepath.Join(cfg, "Nvidia Corporation", "Personal AI Router", "engines"),
		filepath.Join(home, "Library", "Application Support", "Nvidia Corporation", "Personal AI Router", "engines"),
	} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := writeJSONAtomic(filepath.Join(dir, "ollama.json"), override); err != nil {
			t.Fatal(err)
		}
	}
	first := startE2EManager(t, cfg, home)
	send(t, first.stdin, 1, "engine:set-port", map[string]any{"engine": "ollama", "port": 26001})
	var saved EngineStatus
	if err := json.Unmarshal(waitResult(t, first.frames, "1", 10*time.Second), &saved); err != nil || saved.Port != 26001 || saved.Running {
		t.Fatalf("saved status=%+v, err=%v", saved, err)
	}
	first.stop(t)
	second := startE2EManager(t, cfg, home)
	send(t, second.stdin, 1, "engine:describe", map[string]any{"engine": "ollama"})
	var manifest Manifest
	if err := json.Unmarshal(waitResult(t, second.frames, "1", 10*time.Second), &manifest); err != nil {
		t.Fatal(err)
	}
	platform, ok := manifest.HostPlatform()
	if !ok || platform.Runtime.Port != 26001 || len(platform.Runtime.Args) != 2 || platform.Runtime.Args[1] != "--custom-option" || platform.Runtime.Env["CUSTOM_SETTING"] != "retained" {
		t.Fatalf("worker restart lost settings: %+v", platform)
	}
	second.stop(t)
}

func TestE2EDesiredStateAcrossShutdownRPC(t *testing.T) {
	cfg := t.TempDir()
	home := t.TempDir()
	for _, dir := range []string{
		filepath.Join(cfg, "Nvidia Corporation", "Personal AI Router", "engines"),
		filepath.Join(home, "Library", "Application Support", "Nvidia Corporation", "Personal AI Router", "engines"),
	} {
		writeFakeManifest(t, dir)
	}

	first := startE2EManager(t, cfg, home)
	send(t, first.stdin, 1, "engine:start", map[string]any{"engine": "fake"})
	var started EngineStatus
	if err := json.Unmarshal(waitResult(t, first.frames, "1", 20*time.Second), &started); err != nil || !started.Running {
		t.Fatalf("start status=%+v err=%v", started, err)
	}
	send(t, first.stdin, 2, prepareShutdownMethod, nil)
	waitResult(t, first.frames, "2", 15*time.Second)
	if portServing(started.Port) {
		t.Fatalf("engine port %d remained open after shutdown preparation", started.Port)
	}
	first.stop(t)

	second := startE2EManager(t, cfg, home)
	notify(t, second.stdin, restoreEnabledMethod, nil)
	waitNotify(t, second.frames, "engine:state-changed", 20*time.Second)
	send(t, second.stdin, 1, "engine:status", map[string]any{"engine": "fake"})
	if r := waitResult(t, second.frames, "1", 5*time.Second); !strings.Contains(string(r), `"running":true`) {
		t.Fatalf("saved ON state was not restored: %s", r)
	}
	send(t, second.stdin, 2, "engine:stop", map[string]any{"engine": "fake"})
	waitResult(t, second.frames, "2", 10*time.Second)
	second.stop(t)

	third := startE2EManager(t, cfg, home)
	notify(t, third.stdin, restoreEnabledMethod, nil)
	send(t, third.stdin, 1, "engine:status", map[string]any{"engine": "fake"})
	if r := waitResult(t, third.frames, "1", 5*time.Second); !strings.Contains(string(r), `"running":false`) {
		t.Fatalf("explicit OFF state was not preserved: %s", r)
	}
	third.stop(t)
}

// A hard manager death leaves its owned child alive. Saved ON must reclaim
// that exact managed image on the configured port before starting a new child.
func TestE2ERestoreReclaimsOwnedOrphanAfterHardKill(t *testing.T) {
	cfg, home := t.TempDir(), t.TempDir()
	base := cfg
	if runtime.GOOS == "darwin" {
		base = filepath.Join(home, "Library", "Application Support")
	}
	root := filepath.Join(base, "Nvidia Corporation", "Personal AI Router")
	bin := filepath.Join(root, "engine-bin", "fake", filepath.Base(fakeEngineBin))
	if err := os.MkdirAll(filepath.Dir(bin), 0o700); err != nil {
		t.Fatalf("create managed fake install: %v", err)
	}
	copyFile(t, fakeEngineBin, bin)
	port, err := freePort()
	if err != nil {
		t.Fatalf("reserve fake engine port: %v", err)
	}
	pidFile := filepath.Join(t.TempDir(), "fake.pid")
	manifest := testEngineManifest(bin)
	platform := manifest.Platforms[hostKey()]
	platform.Runtime.Port = port
	platform.Runtime.Env["FAKE_PID_FILE"] = pidFile
	manifest.Platforms[hostKey()] = platform
	for _, dir := range []string{
		filepath.Join(cfg, "Nvidia Corporation", "Personal AI Router", "engines"),
		filepath.Join(home, "Library", "Application Support", "Nvidia Corporation", "Personal AI Router", "engines"),
	} {
		writeE2EManifest(t, dir, manifest)
	}

	first := startE2EManager(t, cfg, home)
	send(t, first.stdin, 1, "engine:start", map[string]any{"engine": "fake"})
	var started EngineStatus
	if err := json.Unmarshal(waitResult(t, first.frames, "1", 20*time.Second), &started); err != nil || !started.Running || started.Port != port {
		t.Fatalf("initial start = %+v, err=%v; want running on port %d", started, err, port)
	}
	data, err := os.ReadFile(pidFile)
	if err != nil {
		t.Fatalf("read owned child PID: %v", err)
	}
	oldPID, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil || oldPID <= 0 {
		t.Fatalf("owned child PID = %q, err=%v", data, err)
	}
	t.Cleanup(func() {
		if pidAlive(oldPID) && isOurEngineImage(procImage(oldPID), bin) {
			terminatePID(oldPID, 0)
		}
	})
	first.stopped = true
	if err := first.cmd.Process.Kill(); err != nil {
		t.Fatalf("hard-kill first manager: %v", err)
	}
	_ = first.stdin.Close()
	select {
	case <-first.readerDone:
	case <-time.After(5 * time.Second):
		t.Fatal("first manager reader did not exit after hard kill")
	}
	if err := first.cmd.Wait(); err == nil {
		t.Fatal("hard-killed manager exited successfully")
	}
	if !pidAlive(oldPID) || !portServing(port) {
		t.Fatalf("owned child did not survive hard manager kill: pid=%d alive=%v portServing=%v", oldPID, pidAlive(oldPID), portServing(port))
	}

	second := startE2EManager(t, cfg, home)
	notify(t, second.stdin, restoreEnabledMethod, nil)
	waitNotify(t, second.frames, "engine:state-changed", 20*time.Second)
	send(t, second.stdin, 2, "engine:status", map[string]any{"engine": "fake"})
	var restored EngineStatus
	if err := json.Unmarshal(waitResult(t, second.frames, "2", 5*time.Second), &restored); err != nil || !restored.Running {
		t.Fatalf("restored status = %+v, err=%v; want owned running process", restored, err)
	}
	data, err = os.ReadFile(pidFile)
	if err != nil {
		t.Fatalf("read replacement child PID: %v", err)
	}
	newPID, err := strconv.Atoi(strings.TrimSpace(string(data)))
	listenerPID, image, found := pidOnPort(port)
	if err != nil || newPID <= 0 || newPID == oldPID || !found || listenerPID != newPID || !isOurEngineImage(image, bin) || !pidAlive(newPID) {
		t.Fatalf("cold relaunch PID=%q parsed=%d err=%v; old=%d oldAlive=%v listener=(%d,%q,%v)", data, newPID, err, oldPID, pidAlive(oldPID), listenerPID, image, found)
	}
	waitE2EProcessDead(t, oldPID)
	send(t, second.stdin, 3, "engine:stop", map[string]any{"engine": "fake"})
	waitResult(t, second.frames, "3", 10*time.Second)
	second.stop(t)
	if pidAlive(newPID) || portServing(port) {
		t.Fatalf("replacement survived stop: pid=%d alive=%v portServing=%v", newPID, pidAlive(newPID), portServing(port))
	}
}

// A Unix orphan may remain as a zombie until its new parent reaps it. Require
// absence or an explicit zombie state; a missing image alone is not evidence.
func waitE2EProcessDead(t *testing.T, pid int) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	var state []byte
	var inspectErr error
	for {
		if !pidAlive(pid) {
			return
		}
		if runtime.GOOS != "windows" {
			ctx, cancel := context.WithTimeout(context.Background(), time.Until(deadline))
			state, inspectErr = exec.CommandContext(ctx, "ps", "-o", "stat=", "-p", strconv.Itoa(pid)).Output()
			cancel()
			fields := strings.Fields(string(state))
			if inspectErr == nil && len(fields) == 1 && strings.HasPrefix(fields[0], "Z") {
				return
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("old orphan PID %d is still alive: state=%q inspection error=%v", pid, state, inspectErr)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// TestE2EPrepareShutdownCancelsStartingEngine proves shutdown preparation
// cancels a spawned engine that is still inside its readiness allowance,
// responds while the manager transport remains alive, and joins the child
// before reporting completion.
func TestE2EPrepareShutdownCancelsStartingEngine(t *testing.T) {
	cfg := t.TempDir()
	home := t.TempDir()
	pidFile := filepath.Join(t.TempDir(), "fake.pid")
	manifest := testEngineManifest(fakeEngineBin)
	platform := manifest.Platforms[hostKey()]
	platform.Runtime.Env["FAKE_PID_FILE"] = pidFile
	platform.Runtime.Env["FAKE_START_DELAY"] = "1m"
	platform.Runtime.Ready.TimeoutS = 120
	manifest.Platforms[hostKey()] = platform
	for _, dir := range []string{
		filepath.Join(cfg, "Nvidia Corporation", "Personal AI Router", "engines"),
		filepath.Join(home, "Library", "Application Support", "Nvidia Corporation", "Personal AI Router", "engines"),
	} {
		writeE2EManifest(t, dir, manifest)
	}

	manager := startE2EManager(t, cfg, home)
	send(t, manager.stdin, 1, "engine:start", map[string]any{"engine": "fake"})

	var pid int
	deadline := time.Now().Add(5 * time.Second)
	for pid == 0 && time.Now().Before(deadline) {
		if data, err := os.ReadFile(pidFile); err == nil {
			pid, _ = strconv.Atoi(strings.TrimSpace(string(data)))
		}
		if pid == 0 {
			time.Sleep(25 * time.Millisecond)
		}
	}
	if pid == 0 {
		t.Fatal("fake engine did not publish its PID")
	}
	t.Cleanup(func() {
		if pidAlive(pid) {
			_ = signalPID(pid, true)
		}
	})

	send(t, manager.stdin, 2, prepareShutdownMethod, nil)
	waitResult(t, manager.frames, "2", 10*time.Second)
	if pidAlive(pid) {
		t.Fatalf("managed fake engine PID %d survived shutdown preparation", pid)
	}

	send(t, manager.stdin, 3, "engine:status", map[string]any{"engine": "fake"})
	var stopped EngineStatus
	if err := json.Unmarshal(waitResult(t, manager.frames, "3", 5*time.Second), &stopped); err != nil || stopped.Running {
		t.Fatalf("status after shutdown preparation=%+v err=%v", stopped, err)
	}

	send(t, manager.stdin, 4, "engine:errors", nil)
	var reported struct {
		Errors []serviceError `json:"errors"`
	}
	if err := json.Unmarshal(waitResult(t, manager.frames, "4", 5*time.Second), &reported); err != nil {
		t.Fatalf("decode errors after shutdown preparation: %v", err)
	}
	if hasErr(reported.Errors, startFailedID("fake")) {
		t.Fatalf("shutdown cancellation retained a start-failed error: %+v", reported.Errors)
	}
	manager.stop(t)
}

type e2eManager struct {
	cmd        *exec.Cmd
	stdin      io.WriteCloser
	stdout     io.ReadCloser
	frames     chan frame
	readerDone chan struct{}
	readerStop chan struct{}
	stopped    bool
}

type e2eFrameEnqueue struct {
	id      string
	blocked bool
}

func TestE2EShutdownDrainsSaturatedFrames(t *testing.T) {
	enqueues := make(chan e2eFrameEnqueue)
	manager := startE2EManagerObserved(t, t.TempDir(), t.TempDir(), enqueues)
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	// Unknown requests return frames without detecting or starting any engine.
	// Wait for each response's enqueue before sending another request, so setup
	// never depends on how many requests the OS stdin pipe can hold.
	for id := 1; ; id++ {
		written := make(chan error, 1)
		go func() {
			written <- json.NewEncoder(manager.stdin).Encode(map[string]any{
				"jsonrpc": "2.0", "id": id, "method": "unknown-saturation-request",
			})
		}()
		select {
		case err := <-written:
			if err != nil {
				t.Fatalf("send saturation request %d: %v", id, err)
			}
		case <-deadline.C:
			_ = manager.stdin.Close()
			select {
			case <-written:
			case <-time.After(5 * time.Second):
				t.Fatal("saturation request writer did not stop after stdin closed")
			}
			t.Fatal("timed out writing saturation request")
		}
		select {
		case enqueue := <-enqueues:
			if enqueue.id != strconv.Itoa(id) {
				t.Fatalf("enqueue response ID=%s, want %d", enqueue.id, id)
			}
			if enqueue.blocked {
				if len(manager.frames) != cap(manager.frames) {
					t.Fatal("blocked frame reader did not have a saturated queue")
				}
				t.Logf("saturated frame queue: %d/%d queued, response ID %s blocked", len(manager.frames), cap(manager.frames), enqueue.id)
				manager.stop(t)
				return
			}
		case <-manager.readerDone:
			t.Fatal("frame reader exited before saturation")
		case <-deadline.C:
			t.Fatal("frame reader did not witness saturation")
		}
	}
}

func startE2EManager(t *testing.T, cfg, home string, args ...string) *e2eManager {
	t.Helper()
	return startE2EManagerObserved(t, cfg, home, nil, args...)
}

func startE2EManagerObserved(t *testing.T, cfg, home string, enqueues chan<- e2eFrameEnqueue, args ...string) *e2eManager {
	t.Helper()
	cmd := exec.Command(managerBin, args...)
	cmd.Env = overrideEnv(map[string]string{"APPDATA": cfg, "LOCALAPPDATA": cfg, "XDG_CONFIG_HOME": cfg, "HOME": home})
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
	manager := &e2eManager{
		cmd: cmd, stdin: stdin, stdout: stdout, frames: make(chan frame, 128),
		readerDone: make(chan struct{}), readerStop: make(chan struct{}),
	}
	t.Cleanup(func() {
		if !manager.stopped {
			manager.stopped = true
			if err := manager.shutdown(); err != nil {
				t.Errorf("cleanup manager: %v", err)
			}
		}
	})
	// This parent explicitly acts as a broker that confirms model withdrawal.
	// Consume internal requests before exposing ordinary frames to the test.
	go func() {
		defer close(manager.readerDone)
		sc := bufio.NewScanner(stdout)
		sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for sc.Scan() {
			var f frame
			if err := json.Unmarshal(sc.Bytes(), &f); err != nil {
				t.Errorf("decode manager frame: %v", err)
				return
			}
			if f.Method == nodepolicy.MethodModelMutationRequest {
				var req nodepolicy.ModelMutationRequest
				if err := json.Unmarshal(f.Params, &req); err != nil {
					t.Errorf("decode mutation request: %v", err)
					return
				}
				err := json.NewEncoder(stdin).Encode(map[string]any{
					"jsonrpc": "2.0", "method": nodepolicy.MethodModelMutationReply,
					"params": nodepolicy.ModelMutationReply{RequestID: req.RequestID, Engine: req.Engine, OK: true},
				})
				if err != nil {
					t.Errorf("reply to mutation request: %v", err)
					return
				}
				continue
			}
			enqueue := e2eFrameEnqueue{id: string(f.ID)}
			select {
			case manager.frames <- f:
			default:
				enqueue.blocked = true
			}
			if enqueues != nil && enqueue.id != "" {
				select {
				case enqueues <- enqueue:
				case <-manager.readerStop:
					return
				}
			}
			if enqueue.blocked {
				select {
				case manager.frames <- f:
				case <-manager.readerStop:
					return
				}
			}
		}
		if err := sc.Err(); err != nil {
			select {
			case <-manager.readerStop:
			default:
				t.Errorf("read manager frames: %v", err)
			}
		}
	}()
	waitNotify(t, manager.frames, "engine:ready", 5*time.Second)
	return manager
}

func (m *e2eManager) stop(t *testing.T) {
	t.Helper()
	if m.stopped {
		return
	}
	m.stopped = true
	if err := m.shutdown(); err != nil {
		t.Fatal(err)
	}
}

// shutdown closes stdin and reaps the manager. Wait closes the stdout pipe,
// so it runs only after the frame reader reaches EOF; a reader still
// scanning would report a closed file.
func (m *e2eManager) shutdown() error {
	_ = m.stdin.Close()
	// One Wait owner, after the reader joins, on both normal and forced exits.
	done := make(chan error, 1)
	go func() {
		<-m.readerDone
		done <- m.cmd.Wait()
	}()
	abort := func(reason string) error {
		close(m.readerStop)
		_ = m.stdout.Close()
		_ = m.cmd.Process.Kill()
		select {
		case <-done:
			return errors.New(reason)
		case <-time.After(5 * time.Second):
			return fmt.Errorf("%s; manager reader and process cleanup did not join", reason)
		}
	}
	deadline := time.NewTimer(10 * time.Second)
	defer deadline.Stop()
	reading := true
	for reading {
		select {
		case <-m.frames:
		case <-m.readerDone:
			reading = false
		case <-deadline.C:
			return abort("manager stdout stayed open after stdin closed")
		}
	}
	select {
	case err := <-done:
		if err != nil {
			return fmt.Errorf("manager exit: %w", err)
		}
		return nil
	case <-deadline.C:
		return abort("manager did not exit after stdin closed")
	}
}

func writeFakeManifest(t *testing.T, dir string) {
	writeE2EManifest(t, dir, testEngineManifest(fakeEngineBin))
}

func writeE2EManifest(t *testing.T, dir string, manifest *Manifest) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "fake.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

// overrideEnv returns the current environment with the given keys
// replaced (case-insensitively, for Windows %AppData%).
func overrideEnv(over map[string]string) []string {
	var out []string
	for _, kv := range os.Environ() {
		eq := strings.IndexByte(kv, '=')
		if eq < 0 {
			continue
		}
		key := strings.ToUpper(kv[:eq])
		if _, ok := over[key]; ok {
			continue
		}
		out = append(out, kv)
	}
	for k, v := range over {
		out = append(out, k+"="+v)
	}
	return out
}

type frame struct {
	Params json.RawMessage `json:"params"`
	ID     json.RawMessage `json:"id"`
	Method string          `json:"method"`
	Result json.RawMessage `json:"result"`
	Error  json.RawMessage `json:"error"`
}

func readFrames(r io.Reader, out chan<- frame) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		var f frame
		if err := json.Unmarshal(sc.Bytes(), &f); err != nil {
			continue
		}
		out <- f
	}
}

func send(t *testing.T, w io.Writer, id int, method string, params any) {
	t.Helper()
	msg := map[string]any{"jsonrpc": "2.0", "id": id, "method": method}
	if params != nil {
		msg["params"] = params
	}
	data, _ := json.Marshal(msg)
	if _, err := w.Write(append(data, '\n')); err != nil {
		t.Fatalf("send %s: %v", method, err)
	}
}

func notify(t *testing.T, w io.Writer, method string, params any) {
	t.Helper()
	msg := map[string]any{"jsonrpc": "2.0", "method": method}
	if params != nil {
		msg["params"] = params
	}
	data, _ := json.Marshal(msg)
	if _, err := w.Write(append(data, '\n')); err != nil {
		t.Fatalf("notify %s: %v", method, err)
	}
}

func waitResult(t *testing.T, frames <-chan frame, id string, timeout time.Duration) json.RawMessage {
	t.Helper()
	deadline := time.After(timeout)
	for {
		select {
		case f := <-frames:
			if string(f.ID) != id {
				continue
			}
			if len(f.Error) > 0 && string(f.Error) != "null" {
				t.Fatalf("rpc id %s returned error: %s", id, f.Error)
			}
			return f.Result
		case <-deadline:
			t.Fatalf("timed out waiting for response id %s", id)
			return nil
		}
	}
}

func waitNotify(t *testing.T, frames <-chan frame, method string, timeout time.Duration) {
	t.Helper()
	deadline := time.After(timeout)
	for {
		select {
		case f := <-frames:
			if f.Method == method {
				return
			}
		case <-deadline:
			t.Fatalf("timed out waiting for notification %q", method)
		}
	}
}

// Configure can request a parent rebind while both stdio readers continue
// servicing unrelated messages; literal settings survive a fresh worker.
func TestE2ESettingsRebindRelayAndWorkerReload(t *testing.T) {
	cfg, home := t.TempDir(), t.TempDir()
	fixture := settingsExecutor(t, false)
	manifest, _ := fixture.reg.Get("fake")
	for _, dir := range []string{filepath.Join(cfg, "Nvidia Corporation", "Personal AI Router", "engines"), filepath.Join(home, "Library", "Application Support", "Nvidia Corporation", "Personal AI Router", "engines")} {
		writeE2EManifest(t, dir, manifest)
	}
	manager := startE2EManager(t, cfg, home)
	send(t, manager.stdin, 1, "engine:get-installed", nil)
	waitResult(t, manager.frames, "1", 5*time.Second)
	send(t, manager.stdin, 2, "engine:get-launch", map[string]string{"engine": "fake"})
	var launch settings.LaunchState
	if err := json.Unmarshal(waitResult(t, manager.frames, "2", 5*time.Second), &launch); err != nil {
		t.Fatal(err)
	}
	desired := settings.Config{ServerPort: launch.ServerPort, ProxyPort: 54301, LaunchText: `OLLAMA_ORIGINS="http://localhost" PAIR_TEST_LITERAL="$HOME {port}" ` + launch.LaunchText}
	send(t, manager.stdin, 3, "engine:configure-launch", settings.Configure{Engine: "fake", Settings: desired, OperationID: "operation"})
	var relay settings.Relay
	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()
	waiting := true
	for waiting {
		select {
		case f := <-manager.frames:
			if f.Method == "engine:settings-request" {
				if err := json.Unmarshal(f.Params, &relay); err != nil {
					t.Fatal(err)
				}
				waiting = false
			}
		case <-timer.C:
			t.Fatal("worker never requested parent rebind")
		}
	}
	if relay.Method != "rebind" || relay.Request.RequestID != "operation" || relay.Request.Settings.ProxyPort != 54301 {
		t.Fatalf("bad relay: %+v", relay)
	}
	send(t, manager.stdin, 4, "engine:describe", map[string]string{"engine": "fake"})
	waitResult(t, manager.frames, "4", 5*time.Second)
	notify(t, manager.stdin, "engine:settings-reply", settingsReply{ID: relay.ID, Result: json.RawMessage(`{}`)})
	if err := json.Unmarshal(waitResult(t, manager.frames, "3", 5*time.Second), &launch); err != nil || launch.Running {
		t.Fatalf("configure result %+v %v", launch, err)
	}
	saved := launch.LaunchText
	manager.stop(t)
	restored := startE2EManager(t, cfg, home)
	send(t, restored.stdin, 1, "engine:get-installed", nil)
	waitResult(t, restored.frames, "1", 5*time.Second)
	send(t, restored.stdin, 2, "engine:get-launch", map[string]string{"engine": "fake"})
	if err := json.Unmarshal(waitResult(t, restored.frames, "2", 5*time.Second), &launch); err != nil || launch.LaunchText != saved {
		t.Fatalf("worker reload lost literal settings: %+v %v", launch, err)
	}
	restored.stop(t)
}
