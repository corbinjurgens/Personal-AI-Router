// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func TestSetPortPreservesServingOverrideAndExplicitRemoval(t *testing.T) {
	e := settingsExecutor(t, false)
	serving := servingSettingsFixture()
	port := settingsRequest(t, e).Settings.ServerPort
	if err := e.persistRuntimeServing("fake", port, nil, nil, &serving); err != nil {
		t.Fatal(err)
	}
	if err := e.persistPort("fake", port+1); err != nil {
		t.Fatal(err)
	}
	if !readSettingsOverride(t, e).Serving.Equal(serving) {
		t.Fatal("port-only persistence erased serving")
	}
	path := filepath.Join(e.overrideDir, "fake.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	decoded["custom"] = json.RawMessage(`{"integer":9007199254740993}`)
	if err := writeJSONAtomic(path, decoded); err != nil {
		t.Fatal(err)
	}
	serving = nil
	if err := e.persistRuntimeServing("fake", port, nil, nil, &serving); err != nil {
		t.Fatal(err)
	}
	if readSettingsOverride(t, e).Serving != nil {
		t.Fatal("explicit removal retained serving")
	}
	data, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	var custom struct {
		Integer json.Number `json:"integer"`
	}
	if err := json.Unmarshal(decoded["custom"], &custom); err != nil {
		t.Fatal(err)
	}
	if custom.Integer.String() != "9007199254740993" {
		t.Fatal("unrelated integer changed")
	}
	var override struct {
		Platforms map[string]struct {
			Runtime map[string]json.RawMessage `json:"runtime"`
		} `json:"platforms"`
	}
	if err := json.Unmarshal(data, &override); err != nil {
		t.Fatal(err)
	}
	for _, platform := range override.Platforms {
		if strings.TrimSpace(string(platform.Runtime["serving"])) != "null" {
			t.Fatal("removal cannot mask inherited serving")
		}
	}
}

// loadWithOverrides keeps the real startup merge for bundled metadata and
// launch-description tests. Runtime port fixtures use reloadPortFixture so
// status and background sweeps cannot reach vendor-default listeners.
func loadWithOverrides(t *testing.T, overrideDir string) *Registry {
	t.Helper()
	reg := NewRegistry()
	if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
		t.Fatalf("LoadFS bundled: %v", err)
	}
	if err := reg.LoadOverrideDir(overrideDir); err != nil {
		t.Fatalf("LoadOverrideDir: %v", err)
	}
	return reg
}

func hostPort(t *testing.T, reg *Registry, engine string) int {
	t.Helper()
	m, ok := reg.Get(engine)
	if !ok {
		t.Fatalf("engine %q not in registry", engine)
	}
	p, ok := m.HostPlatform()
	if !ok {
		t.Fatalf("engine %q has no host platform block", engine)
	}
	return p.Runtime.Port
}

// portFixtureManifest is fresh test-only bundled data, not a sanitized vendor
// manifest. Its shared runtime also exercises platform override precedence.
func portFixtureManifest(engine string, port int) map[string]any {
	m := testEngineManifest(fakeEngineBin)
	p := m.Platforms[hostKey()]
	p.Runtime.Port = port
	p.Runtime.Args = []string{"serve"}
	p.Runtime.Env = map[string]string{
		"OLLAMA_HOST": "{host}:{port}", "FAKE_ADDR": "{host}:{port}",
	}
	return map[string]any{
		"engine": engine, "display_name": engine, "manifest_version": 1,
		"runtime": p.Runtime, "actions": m.Actions,
		"platforms": map[string]any{hostKey(): map[string]any{"detect": p.Detect}},
	}
}

// Hold every allocation until the whole set is selected, so old/new/default
// ports cannot accidentally be equal. All listeners are private loopback.
func privateTestPorts(t *testing.T, count int, exclude ...int) []int {
	t.Helper()
	ports, err := privateFixturePorts(count, exclude...)
	if err != nil {
		t.Fatal(err)
	}
	return ports
}

func privateFixturePorts(count int, exclude ...int) (ports []int, err error) {
	var listeners []net.Listener
	defer func() {
		for _, listener := range listeners {
			err = errors.Join(err, listener.Close())
		}
	}()
	exclude = append([]int{11434, 1234, 1235, 8080, 8081}, exclude...)
	for len(ports) < count {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			return nil, err
		}
		listeners = append(listeners, listener)
		port := listener.Addr().(*net.TCPAddr).Port
		if !slices.Contains(exclude, port) {
			ports = append(ports, port)
		}
	}
	return ports, nil
}

func reloadPortFixture(t *testing.T, base *Registry, overrideDir string) *Registry {
	t.Helper()
	reg := NewRegistry()
	for engine, raw := range base.bundledRaw {
		if _, err := reg.addManifest("test-only bundled "+engine, raw); err != nil {
			t.Fatal(err)
		}
		reg.bundledRaw[engine] = append([]byte(nil), raw...)
	}
	if err := reg.LoadOverrideDir(overrideDir); err != nil {
		t.Fatal(err)
	}
	return reg
}

func newPortFixtureExecutor(t *testing.T, overrideDir string) *Executor {
	t.Helper()
	base := NewRegistry()
	ports := privateTestPorts(t, 2)
	for i, engine := range []string{"ollama", "lmstudio"} {
		raw, err := json.Marshal(portFixtureManifest(engine, ports[i]))
		if err != nil {
			t.Fatal(err)
		}
		base.bundledRaw[engine] = raw
	}
	reg := reloadPortFixture(t, base, overrideDir)
	ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())
	ex.overrideDir = overrideDir
	return ex
}

func adoptedEngineFixture(t *testing.T, engine string, port int, rt Runtime) *Executor {
	t.Helper()
	m := testEngineManifest(fakeEngineBin)
	m.Engine = engine
	m.DisplayName = engine
	for key, platform := range m.Platforms {
		platform.Runtime = rt
		m.Platforms[key] = platform
	}
	ex := newTestExecutor(t, m)
	ex.overrideDir = t.TempDir()
	st, err := ex.state(engine)
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.installed = true
	st.running = true
	st.healthy = true
	st.adopted = true
	st.port = port
	st.mu.Unlock()
	return ex
}

func adoptedCommandEngineFixture(t *testing.T, engine string, port int) (*Executor, func() bool, func(int) bool) {
	t.Helper()
	dir := t.TempDir()
	stopMarker := filepath.Join(dir, "stopped-{port}")
	startMarker := filepath.Join(dir, "started-{port}")
	ex := adoptedEngineFixture(t, engine, port, Runtime{
		Mode:  "command",
		Port:  port,
		Start: [][]string{{fakeEngineBin, "touch", startMarker}},
		Stop:  &StopSpec{Cmd: []string{fakeEngineBin, "touch", stopMarker}},
	})
	return ex,
		func() bool { return fileExists(filepath.Join(dir, "stopped-"+strconv.Itoa(port))) },
		func(port int) bool { return fileExists(filepath.Join(dir, "started-"+strconv.Itoa(port))) }
}

func adoptedProcessEngineFixture(t *testing.T, engine string, port int) *Executor {
	t.Helper()
	return adoptedEngineFixture(t, engine, port, Runtime{
		Bin:  fakeEngineBin,
		Port: port,
		Stop: &StopSpec{Signal: "term", GraceS: 3},
	})
}

func adoptedCommandEngineWithoutStopFixture(t *testing.T, engine string, port int) *Executor {
	t.Helper()
	return adoptedEngineFixture(t, engine, port, Runtime{
		Mode:  "command",
		Port:  port,
		Start: [][]string{{fakeEngineBin, "echo", "up"}},
	})
}

// TestOverrideDeepMergePreservesBundled proves a partial port-only override
// file pins runtime.port while inheriting the rest of the bundled manifest
// (the single-source-of-truth merge that backs engine:set-port persistence).
func TestOverrideDeepMergePreservesBundled(t *testing.T) {
	dir := t.TempDir()
	override := []byte(`{"engine":"ollama","runtime":{"port":54321}}`)
	if err := os.WriteFile(filepath.Join(dir, "ollama.json"), override, 0o644); err != nil {
		t.Fatal(err)
	}
	reg := loadWithOverrides(t, dir)

	m, ok := reg.Get("ollama")
	if !ok {
		t.Fatal("ollama not loaded")
	}
	p, ok := m.HostPlatform()
	if !ok {
		t.Fatal("ollama has no host platform")
	}
	if p.Runtime.Port != 54321 {
		t.Errorf("override port: got %d, want 54321", p.Runtime.Port)
	}
	// Bundled fields untouched: the top-level runtime.args ["serve"] is
	// inherited by every platform, and the install fetch URL survives.
	if len(p.Runtime.Args) != 1 || p.Runtime.Args[0] != "serve" {
		t.Errorf("bundled runtime.args not preserved through merge: %v", p.Runtime.Args)
	}
	if p.Install == nil || p.Install.Fetch == nil || p.Install.Fetch.URL == "" {
		t.Errorf("bundled install fetch not preserved through merge: %+v", p.Install)
	}
}

// TestSetPortPersistsAndRestores covers the full engine:set-port contract for
// a bundled engine that isn't running: the chosen port is written as a
// manifest override and a fresh (startup-style) load comes back up on it.
func TestSetPortPersistsAndRestores(t *testing.T) {
	for _, engine := range []string{"ollama", "lmstudio"} {
		t.Run(engine, func(t *testing.T) {
			dir := t.TempDir()
			ex := newPortFixtureExecutor(t, dir)
			bundled := hostPort(t, ex.reg, engine)
			port := privateTestPorts(t, 1, bundled)[0]

			st, err := ex.SetPort(context.Background(), engine, port)
			if err != nil {
				t.Fatalf("SetPort: %v", err)
			}
			if st.Port != port {
				t.Errorf("returned status port: got %d, want %d", st.Port, port)
			}

			// The override file is the minimal port delta.
			data, err := os.ReadFile(filepath.Join(dir, engine+".json"))
			if err != nil {
				t.Fatalf("override file not written: %v", err)
			}
			var got map[string]any
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatalf("override file invalid JSON: %v", err)
			}
			rt, _ := got["runtime"].(map[string]any)
			if rt == nil || int(rt["port"].(float64)) != port {
				t.Errorf("override file runtime.port: got %v, want %d", got["runtime"], port)
			}

			// Restore: a fresh startup-style load comes up on the chosen port.
			if p := hostPort(t, reloadPortFixture(t, ex.reg, dir), engine); p != port {
				t.Errorf("restored port: got %d, want %d", p, port)
			}

			// Reverting to the bundled default removes the override entirely.
			if _, err := ex.SetPort(context.Background(), engine, bundled); err != nil {
				t.Fatalf("SetPort revert: %v", err)
			}
			if _, err := os.Stat(filepath.Join(dir, engine+".json")); !os.IsNotExist(err) {
				t.Errorf("override file should be removed when reverting to default, stat err = %v", err)
			}
			if p := hostPort(t, reloadPortFixture(t, ex.reg, dir), engine); p != bundled {
				t.Errorf("port after revert: got %d, want bundled default %d", p, bundled)
			}
		})
	}
}

// TestSetPortRejectsOutOfRange guards the validation boundary.
func TestSetPortRejectsOutOfRange(t *testing.T) {
	ex := newPortFixtureExecutor(t, t.TempDir())
	for _, bad := range []int{0, -1, 70000} {
		if _, err := ex.SetPort(context.Background(), "ollama", bad); err == nil {
			t.Errorf("SetPort(%d) should have been rejected", bad)
		}
	}
}

func TestSetPortMovesAdoptedCommandEngine(t *testing.T) {
	ports := privateTestPorts(t, 2)
	oldPort, newPort := ports[0], ports[1]
	ex, stopped, started := adoptedCommandEngineFixture(t, "lmstudio", oldPort)
	status, err := ex.SetPort(context.Background(), "lmstudio", newPort)
	if err != nil {
		t.Fatal(err)
	}
	if !stopped() || !started(newPort) {
		t.Fatal("identified command engine was not stopped on the old port and restarted on the new port")
	}
	if status.Port != newPort || !status.Running {
		t.Fatalf("moved status = %+v, want running on %d", status, newPort)
	}
}

func TestSetPortStillRejectsAdoptedProcessEngine(t *testing.T) {
	ports := privateTestPorts(t, 2)
	ex := adoptedProcessEngineFixture(t, "ollama", ports[0])
	if _, err := ex.SetPort(context.Background(), "ollama", ports[1]); err == nil {
		t.Fatal("adopted process engine unexpectedly moved")
	}
	if status, err := ex.Status("ollama"); err != nil || status.Port != ports[0] || !status.Running {
		t.Fatalf("rejected process engine changed: status=%+v err=%v", status, err)
	}
	if _, err := os.Stat(filepath.Join(ex.overrideDir, "ollama.json")); !os.IsNotExist(err) {
		t.Fatalf("rejected process engine persisted an override: %v", err)
	}
}

func TestSetPortRejectsCommandEngineWithoutStopCommand(t *testing.T) {
	ports := privateTestPorts(t, 2)
	ex := adoptedCommandEngineWithoutStopFixture(t, "external", ports[0])
	if _, err := ex.SetPort(context.Background(), "external", ports[1]); err == nil {
		t.Fatal("command engine without official stop unexpectedly moved")
	}
	if status, err := ex.Status("external"); err != nil || status.Port != ports[0] || !status.Running {
		t.Fatalf("rejected command engine changed: status=%+v err=%v", status, err)
	}
	if _, err := os.Stat(filepath.Join(ex.overrideDir, "external.json")); !os.IsNotExist(err) {
		t.Fatalf("rejected command engine persisted an override: %v", err)
	}
}

func TestSetPortRestartsOldPortWhenPersistenceFails(t *testing.T) {
	ports := privateTestPorts(t, 2)
	oldPort, newPort := ports[0], ports[1]
	ex, stopped, started := adoptedCommandEngineFixture(t, "lmstudio", oldPort)
	blocked := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(blocked, []byte("blocked"), 0o644); err != nil {
		t.Fatal(err)
	}
	ex.overrideDir = blocked

	if _, err := ex.SetPort(context.Background(), "lmstudio", newPort); err == nil {
		t.Fatal("expected persistence failure")
	}
	if !stopped() || !started(oldPort) || started(newPort) {
		t.Fatal("persistence failure did not stop and restore the engine on its old port")
	}
	if status, err := ex.Status("lmstudio"); err != nil || status.Port != oldPort || !status.Running {
		t.Fatalf("restored status = %+v err=%v, want running on %d", status, err, oldPort)
	}
}
