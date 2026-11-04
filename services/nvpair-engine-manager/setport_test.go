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
	"strings"
	"testing"

	"nvpair-shared/engines"
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
	p.Runtime.Env = map[string]string{"FAKE_ADDR": "{host}:{port}"}
	return map[string]any{
		"engine": engine, "display_name": engine, "manifest_version": 1,
		"runtime": p.Runtime, "actions": m.Actions,
		"platforms": map[string]any{hostKey(): map[string]any{"detect": p.Detect}},
	}
}

func llamaPortFixtureManifest(engine string, port int) map[string]any {
	m := portFixtureManifest(engine, port)
	rt := m["runtime"].(Runtime)
	rt.Env = map[string]string{"FAKE_ADDR": "{host}:{port}"}
	m["runtime"] = rt
	m["family"] = "llamacpp"
	m["model_naming"] = engines.ModelNamingExact
	m["actions"] = map[string]Action{"list_models": {HTTP: &ActionHTTP{Method: "GET", Path: "/v1/models"}}}
	return m
}

// Synthetic candidates implement only the listener contract; no socket is bound.
type fixturePortCandidate struct {
	port       int
	closeCalls int
	closeErr   error
}

func (c *fixturePortCandidate) Accept() (net.Conn, error) {
	return nil, errors.New("synthetic port candidate cannot accept connections")
}

func (c *fixturePortCandidate) Close() error {
	c.closeCalls++
	return c.closeErr
}

func (c *fixturePortCandidate) Addr() net.Addr {
	return &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: c.port}
}

func TestPrivateFixturePortsRejectsProtectedAndCallerPorts(t *testing.T) {
	candidates := []int{
		11434, 1234, 1235, 8080, 8081, 14318, 14319, 14320,
		24003, 14321, 14322, 14323, 24010, 24001, 24002,
	}
	var opened []*fixturePortCandidate
	calls := 0
	listen := func(network, address string) (net.Listener, error) {
		if network != "tcp" || address != "127.0.0.1:0" {
			t.Fatalf("allocation target = %s %s, want private loopback", network, address)
		}
		for _, candidate := range opened {
			if candidate.closeCalls != 0 {
				t.Fatal("a reservation was released before selection completed")
			}
		}
		if calls == len(candidates) {
			t.Fatal("allocator exhausted its synthetic candidates")
		}
		candidate := &fixturePortCandidate{port: candidates[calls]}
		calls++
		opened = append(opened, candidate)
		return candidate, nil
	}
	excluded := []int{24010}
	ports, err := allocatePrivateFixturePorts(listen, 3, excluded...)
	if err != nil {
		t.Fatalf("allocate synthetic ports: %v", err)
	}
	if !slices.Equal(ports, []int{24003, 24001, 24002}) || calls != len(candidates) {
		t.Fatalf("selection = %v after %d calls, want ordered distinct allowed candidates", ports, calls)
	}
	if !slices.Equal(excluded, []int{24010}) {
		t.Fatal("allocator changed caller exclusions")
	}
	for _, candidate := range opened {
		if candidate.closeCalls != 1 {
			t.Fatalf("candidate %d closed %d times, want once", candidate.port, candidate.closeCalls)
		}
	}
}

func TestPrivateFixturePortsJoinsAllocationAndCleanupErrors(t *testing.T) {
	allocationErr := errors.New("synthetic allocation failure")
	firstCloseErr := errors.New("first synthetic close failure")
	secondCloseErr := errors.New("second synthetic close failure")
	opened := []*fixturePortCandidate{
		{port: 24003, closeErr: firstCloseErr},
		{port: 14318, closeErr: secondCloseErr},
	}
	calls := 0
	listen := func(network, address string) (net.Listener, error) {
		if network != "tcp" || address != "127.0.0.1:0" {
			t.Fatalf("allocation target = %s %s, want private loopback", network, address)
		}
		for _, candidate := range opened {
			if candidate.closeCalls != 0 {
				t.Fatal("cleanup ran before the allocation failed")
			}
		}
		calls++
		if calls > len(opened) {
			return nil, allocationErr
		}
		return opened[calls-1], nil
	}
	ports, err := allocatePrivateFixturePorts(listen, 3)
	if ports != nil || calls != 3 || !errors.Is(err, allocationErr) || !errors.Is(err, firstCloseErr) || !errors.Is(err, secondCloseErr) {
		t.Fatalf("failure = ports %v, calls %d, error %v; want allocation and both cleanup errors", ports, calls, err)
	}
	for _, candidate := range opened {
		if candidate.closeCalls != 1 {
			t.Fatalf("candidate %d closed %d times after failure", candidate.port, candidate.closeCalls)
		}
	}
}

func TestPrivateFixturePortsZeroCountDoesNotAllocate(t *testing.T) {
	listen := func(string, string) (net.Listener, error) {
		t.Fatal("zero-count selection called the listener")
		return nil, errors.New("unexpected synthetic allocation")
	}
	ports, err := allocatePrivateFixturePorts(listen, 0)
	if ports != nil || err != nil {
		t.Fatalf("zero-count selection = %v, %v; want nil without allocation", ports, err)
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

func privateFixturePorts(count int, exclude ...int) ([]int, error) {
	return allocatePrivateFixturePorts(net.Listen, count, exclude...)
}

// Inject only listener allocation so protected candidates can be supplied as
// synthetic TCP addresses without binding them or changing a global listener.
func allocatePrivateFixturePorts(listen func(string, string) (net.Listener, error), count int, exclude ...int) (ports []int, err error) {
	var listeners []net.Listener
	defer func() {
		for _, listener := range listeners {
			err = errors.Join(err, listener.Close())
		}
	}()
	exclude = append([]int{11434, 1234, 1235, 8080, 8081, 14318, 14319, 14320, 14321, 14322, 14323}, exclude...)
	for len(ports) < count {
		listener, err := listen("tcp", "127.0.0.1:0")
		if err != nil {
			return nil, err
		}
		listeners = append(listeners, listener)
		address, ok := listener.Addr().(*net.TCPAddr)
		if !ok {
			return nil, errors.New("fixture allocation requires a TCP address")
		}
		port := address.Port
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

func newLlamaPortFixtureExecutor(t *testing.T, overrideDir string) *Executor {
	t.Helper()
	return portFixtureExecutorFor(t, overrideDir, []string{engines.NameLlamaCPP, portFixtureCustomEngine(t).Name}, llamaPortFixtureManifest)
}

func portFixtureExecutorFor(t *testing.T, overrideDir string, names []string, manifest func(string, int) map[string]any) *Executor {
	t.Helper()
	base := NewRegistry()
	ports := privateTestPorts(t, len(names))
	for i, engine := range names {
		raw, err := json.Marshal(manifest(engine, ports[i]))
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

func portFixtureCustomEngine(t *testing.T) engines.Engine {
	t.Helper()
	engine, ok := engines.DynamicEngine("llamacpp-custom", "llama.cpp custom")
	if !ok {
		t.Fatal("invalid custom fixture engine")
	}
	return engine
}

func adoptedEngineFixture(t *testing.T, engine string, port int, rt Runtime) *Executor {
	t.Helper()
	m := llamaTestEngineManifest(fakeEngineBin)
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

func adoptedProcessEngineFixture(t *testing.T, engine string, port int) *Executor {
	t.Helper()
	return adoptedEngineFixture(t, engine, port, Runtime{
		Bin:  fakeEngineBin,
		Port: port,
		Stop: &StopSpec{Signal: "term", GraceS: 3},
	})
}

// TestOverrideDeepMergePreservesBundled proves a partial port-only override
// file pins runtime.port while inheriting the rest of the bundled manifest
// (the single-source-of-truth merge that backs engine:set-port persistence).
func TestOverrideDeepMergePreservesBundled(t *testing.T) {
	dir := t.TempDir()
	override := []byte(`{"engine":"llamacpp","runtime":{"port":54321}}`)
	if err := os.WriteFile(filepath.Join(dir, "llamacpp.json"), override, 0o644); err != nil {
		t.Fatal(err)
	}
	reg := loadWithOverrides(t, dir)

	m, ok := reg.Get(engines.NameLlamaCPP)
	if !ok {
		t.Fatal("llama.cpp not loaded")
	}
	p, ok := m.HostPlatform()
	if !ok {
		t.Fatal("llama.cpp has no host platform")
	}
	if p.Runtime.Port != 54321 {
		t.Errorf("override port: got %d, want 54321", p.Runtime.Port)
	}
	// Shared launch arguments and the platform install source survive.
	if !slices.Equal(p.Runtime.Args, []string{"--sleep-idle-seconds", "300", "--ctx-size", "8192", "--parallel", "1", "--cache-ram", "512", "--host", "{host}", "--port", "{port}", "--cors-origins", ""}) {
		t.Errorf("bundled runtime.args not preserved through merge: %v", p.Runtime.Args)
	}
	if p.Install == nil || ((p.Install.Fetch == nil || p.Install.Fetch.URL == "") && len(p.Install.Artifacts) == 0) {
		t.Errorf("bundled install fetch not preserved through merge: %+v", p.Install)
	}
}

// TestSetPortPersistsAndRestores covers the full engine:set-port contract for
// a bundled engine that isn't running: the chosen port is written as a
// manifest override and a fresh (startup-style) load comes back up on it.
func TestSetPortPersistsAndRestores(t *testing.T) {
	for _, engine := range []string{engines.NameLlamaCPP, portFixtureCustomEngine(t).Name} {
		t.Run(engine, func(t *testing.T) {
			dir := t.TempDir()
			ex := newLlamaPortFixtureExecutor(t, dir)
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
	ex := newLlamaPortFixtureExecutor(t, t.TempDir())
	for _, bad := range []int{0, -1, 70000} {
		if _, err := ex.SetPort(context.Background(), engines.NameLlamaCPP, bad); err == nil {
			t.Errorf("SetPort(%d) should have been rejected", bad)
		}
	}
}

func TestSetPortStillRejectsAdoptedProcessEngine(t *testing.T) {
	ports := privateTestPorts(t, 2)
	ex := adoptedProcessEngineFixture(t, engines.NameLlamaCPP, ports[0])
	if _, err := ex.SetPort(context.Background(), engines.NameLlamaCPP, ports[1]); err == nil {
		t.Fatal("adopted process engine unexpectedly moved")
	}
	if status, err := ex.Status(engines.NameLlamaCPP); err != nil || status.Port != ports[0] || !status.Running {
		t.Fatalf("rejected process engine changed: status=%+v err=%v", status, err)
	}
	if _, err := os.Stat(filepath.Join(ex.overrideDir, "llamacpp.json")); !os.IsNotExist(err) {
		t.Fatalf("rejected process engine persisted an override: %v", err)
	}
}

// ownedProcessEngineFixture starts the fake engine as an owned process serving
// port, so a test can move it.
func ownedProcessEngineFixture(t *testing.T, engine string, port int) *Executor {
	t.Helper()
	ex := adoptedEngineFixture(t, engine, port, Runtime{
		Bin:   fakeEngineBin,
		Env:   map[string]string{"FAKE_ADDR": "127.0.0.1:{port}"},
		Port:  port,
		Ready: &Probe{TCP: "127.0.0.1:{port}", TimeoutS: 10},
		Stop:  &StopSpec{Signal: "term", GraceS: 3},
	})
	st, err := ex.state(engine)
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.running, st.healthy, st.adopted = false, false, false
	st.mu.Unlock()
	t.Cleanup(ex.StopAll)
	if err := ex.Start(context.Background(), engine); err != nil {
		t.Fatalf("start owned engine: %v", err)
	}
	return ex
}

func TestSetPortMovesOwnedProcessEngine(t *testing.T) {
	ports := privateTestPorts(t, 2)
	oldPort, newPort := ports[0], ports[1]
	engine := portFixtureCustomEngine(t).Name
	ex := ownedProcessEngineFixture(t, engine, oldPort)
	status, err := ex.SetPort(context.Background(), engine, newPort)
	if err != nil {
		t.Fatal(err)
	}
	if status.Port != newPort || !status.Running {
		t.Fatalf("moved status = %+v, want running on %d", status, newPort)
	}
}

func TestSetPortRestartsOldPortWhenPersistenceFails(t *testing.T) {
	ports := privateTestPorts(t, 2)
	oldPort, newPort := ports[0], ports[1]
	engine := portFixtureCustomEngine(t).Name
	ex := ownedProcessEngineFixture(t, engine, oldPort)
	blocked := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(blocked, []byte("blocked"), 0o644); err != nil {
		t.Fatal(err)
	}
	ex.overrideDir = blocked

	if _, err := ex.SetPort(context.Background(), engine, newPort); err == nil {
		t.Fatal("expected persistence failure")
	}
	if status, err := ex.Status(engine); err != nil || status.Port != oldPort || !status.Running {
		t.Fatalf("restored status = %+v err=%v, want running on %d", status, err, oldPort)
	}
}
