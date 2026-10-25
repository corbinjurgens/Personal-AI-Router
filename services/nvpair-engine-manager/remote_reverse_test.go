// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modeldownload"
	"nvpair-shared/noderec"
)

// The fixture supplies authenticated outer-session ownership at Register's
// seam. Real yamux streams forward to the existing pinned inner TLS handler.
// Cross-process acceptance separately covers outer mTLS and private dial IPC.
type remoteReverseFixture struct {
	m                         *Manager
	selfDir, peerDir          string
	server                    *httptest.Server
	directHealthy             atomic.Bool
	directCalls, reverseCalls atomic.Int32
	local, remote             *clustertrust.ReverseManager
	outer                     net.Conn
	node                      noderec.DirectoryNode
}

func newRemoteReverseFixture(t *testing.T, handler func(*clustertrust.Mesh) http.Handler) *remoteReverseFixture {
	t.Helper()
	selfDir, peerDir := pairedPolicyDirs(t)
	mesh, peerMesh := clustertrust.Open(selfDir), clustertrust.Open(peerDir)
	f := &remoteReverseFixture{selfDir: selfDir, peerDir: peerDir}
	f.server = httptest.NewUnstartedServer(handler(peerMesh))
	f.server.TLS = peerMesh.ServerTLSConfig()
	f.server.StartTLS()
	t.Cleanup(f.server.Close)
	out := &syncBuffer{}
	f.m = NewManager(NewCodec(out), &Executor{progress: newProgressHub(), reg: NewRegistry()}, mesh, selfDir)
	host, portText, err := net.SplitHostPort(f.server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}
	f.node = noderec.DirectoryNode{HostUUID: "host-peer", IP: host, ClusterUUID: "uuid-peer",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: port}}}
	f.m.peers.set([]noderec.DirectoryNode{f.node})
	resolve := func(ctx context.Context, _ clustertrust.ReversePeer, service noderec.ServiceKey) (clustertrust.ReverseTarget, error) {
		if service != noderec.ServiceEngineControl {
			return clustertrust.ReverseTarget{}, errors.New("wrong service")
		}
		return clustertrust.ReverseTarget{Protocol: clustertrust.ReverseTLS, Dial: func(ctx context.Context) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp", f.server.Listener.Addr().String())
		}}, nil
	}
	makeOwner := func(uuid string) *clustertrust.ReverseManager {
		owner, err := clustertrust.NewReverseManager(clustertrust.ReverseOptions{LocalUUID: uuid,
			Validate: func(p clustertrust.ReversePeer) bool { return p.Incarnation == "fixture-v1" }, ResolveService: resolve,
			SetupTimeout: time.Second})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			done := make(chan error, 1)
			go func() { done <- owner.Close() }()
			select {
			case err := <-done:
				if err != nil {
					t.Errorf("close reverse owner: %v", err)
				}
			case <-time.After(3 * time.Second):
				t.Error("reverse owner cleanup did not finish")
			}
		})
		return owner
	}
	f.local, f.remote = makeOwner("uuid-self"), makeOwner("uuid-peer")
	f.connect(t)
	adapter := func(budget time.Duration) *clustertrust.PeerServiceClient {
		return clustertrust.NewPeerServiceClient(mesh, clustertrust.PeerServiceClientOptions{
			PeerClientOptions: clustertrust.PeerClientOptions{ResponseHeaderTimeout: budget},
			BaseConfigDir:     filepath.Dir(selfDir), ResolveRoute: f.m.resolveControlRoute, DirectTimeout: 100 * time.Millisecond,
			DialContext: func(ctx context.Context, _, address string) (net.Conn, error) {
				f.directCalls.Add(1)
				if f.directHealthy.Load() {
					return (&net.Dialer{}).DialContext(ctx, "tcp", address)
				}
				// TCP succeeds, but the first TLS exchange fails before HTTP.
				left, right := net.Pipe()
				if err := right.Close(); err != nil {
					t.Errorf("close rejecting TLS peer: %v", err)
				}
				return left, nil
			},
			DialReverse: func(ctx context.Context, base, uuid string, service noderec.ServiceKey) (net.Conn, error) {
				f.reverseCalls.Add(1)
				if base != filepath.Dir(selfDir) || uuid != "uuid-peer" || service != noderec.ServiceEngineControl {
					return nil, errors.New("wrong reverse destination or config directory")
				}
				return f.local.Open(ctx, clustertrust.ReversePeer{UUID: uuid, Incarnation: "fixture-v1"}, service)
			},
		})
	}
	f.m.controlHTTP, f.m.controlReadyHTTP = adapter(remoteResponseHeaderTimeout), adapter(remoteReadyResponseHeaderTimeout)
	t.Cleanup(f.m.controlHTTP.CloseIdle)
	t.Cleanup(f.m.controlReadyHTTP.CloseIdle)
	return f
}

func (f *remoteReverseFixture) connect(t *testing.T) {
	t.Helper()
	left, right := net.Pipe()
	if err := f.local.Register(clustertrust.ReversePeer{UUID: "uuid-peer", Incarnation: "fixture-v1"}, left, "uuid-self", "11111111111111111111111111111111"); err != nil {
		right.Close()
		t.Fatal(err)
	}
	if err := f.remote.Register(clustertrust.ReversePeer{UUID: "uuid-self", Incarnation: "fixture-v1"}, right, "uuid-self", "11111111111111111111111111111111"); err != nil {
		t.Fatal(err)
	}
	f.outer = left
}

func (f *remoteReverseFixture) client(t *testing.T, ctx context.Context) *remoteClient {
	t.Helper()
	peer, ok := f.m.peers.lookup("host-peer")
	if !ok {
		t.Fatal("missing ec peer")
	}
	client, err := f.m.remoteClient(ctx, peer)
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func remoteFixtureContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func remoteFixtureEncode(t *testing.T, w http.ResponseWriter, value any) {
	t.Helper()
	if err := json.NewEncoder(w).Encode(value); err != nil {
		t.Errorf("encode response: %v", err)
	}
}

func TestRemoteControlPrefersDirectAndClosesEachConnection(t *testing.T) {
	var calls atomic.Int32
	f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
		return (&controlServer{mesh: mesh}).requirePin(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			if !r.Close {
				t.Error("control request could retain a replayable idle connection")
			}
			remoteFixtureEncode(t, w, map[string]any{"engines": []EngineStatus{{Engine: "fixture", Running: false, TerminationVerified: true}}})
		}))
	})
	f.directHealthy.Store(true)
	ctx := remoteFixtureContext(t)
	client := f.client(t, ctx)
	for range 3 {
		raw, err := client.getEngines(ctx)
		if err != nil {
			t.Fatal(err)
		}
		var result struct {
			Engines []EngineStatus `json:"engines"`
		}
		if err := json.Unmarshal(raw, &result); err != nil {
			t.Fatal(err)
		}
		if len(result.Engines) != 1 || result.Engines[0].Engine != "fixture" || result.Engines[0].Running || !result.Engines[0].TerminationVerified {
			t.Fatalf("status = %+v", result)
		}
	}
	if calls.Load() != 3 || f.directCalls.Load() != 3 || f.reverseCalls.Load() != 0 {
		t.Fatalf("HTTP/direct/reverse = %d/%d/%d, want 3/3/0", calls.Load(), f.directCalls.Load(), f.reverseCalls.Load())
	}
}

func TestRemoteReverseReadsAndEveryControlRoute(t *testing.T) {
	const (
		downloadID = "22222222222222222222222222222222"
		sourceID   = "11111111111111111111111111111111"
		pullOpID   = "pull-route-operation"
		pullModel  = "fixture-model"
	)
	var mu sync.Mutex
	seen := map[string]int{}
	f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
		return (&controlServer{mesh: mesh}).requirePin(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			seen[r.URL.Path]++
			mu.Unlock()
			if r.URL.Path == controlEnginesPath {
				if r.Method != http.MethodGet {
					t.Error("inventory must be GET")
				}
				remoteFixtureEncode(t, w, map[string]any{"engines": []EngineStatus{{Engine: "fixture"}}})
				return
			}
			if r.Method != http.MethodPost {
				t.Error("control must be POST")
			}
			var body struct {
				Engine string `json:"engine"`
				OpID   string `json:"opId"`
				Model  string `json:"model"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
				return
			}
			if body.Engine != "fixture" {
				t.Errorf("engine = %q", body.Engine)
			}
			if r.URL.Path == controlPullPath {
				if body.OpID != pullOpID || body.Model != pullModel {
					t.Errorf("pull request identity = %+v", body)
				}
				percent := 37
				snapshot := modeldownload.Snapshot{
					Version: modeldownload.Version, SourceID: sourceID, Revision: "1",
					Entries: []modeldownload.Entry{{ID: downloadID, Engine: body.Engine, Model: body.Model, Phase: modeldownload.PhaseInProgress, BlocksReadiness: true, Percent: &percent}},
				}
				remoteFixtureEncode(t, w, streamFrame{
					Type: "progress", OpID: body.OpID, Engine: body.Engine, Op: "pull", ID: downloadID, Model: body.Model,
					Stage: "fixture-stage", Percent: percent, DownloadState: &snapshot,
				})
				terminal := snapshot.Clone()
				terminal.Revision = "2"
				terminal.Entries[0].Phase = modeldownload.PhaseComplete
				terminal.Entries[0].BlocksReadiness = false
				terminal.Entries[0].Percent = nil
				remoteFixtureEncode(t, w, streamFrame{
					Type: "result", OpID: body.OpID, Engine: body.Engine, Op: "pull", ID: downloadID, Model: body.Model,
					DownloadState: &terminal, Result: json.RawMessage(`{"ok":true}`),
				})
				return
			}
			if r.URL.Path == controlInstallPath {
				remoteFixtureEncode(t, w, streamFrame{Type: "progress", Stage: "fixture-stage"})
				remoteFixtureEncode(t, w, streamFrame{Type: "result", Result: json.RawMessage(`{"ok":true}`)})
				return
			}
			remoteFixtureEncode(t, w, map[string]bool{"ok": true})
		}))
	})
	ctx := remoteFixtureContext(t)
	client := f.client(t, ctx)
	if _, err := client.getEngines(ctx); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{controlStartPath, controlStopPath, controlLoadPath, controlUnloadPath, controlDeletePath} {
		t.Run(path, func(t *testing.T) {
			raw, err := client.postJSON(ctx, path, "fixture", map[string]string{"engine": "fixture"})
			if err != nil {
				t.Fatal(err)
			}
			var result struct {
				OK bool `json:"ok"`
			}
			if err := json.Unmarshal(raw, &result); err != nil {
				t.Fatal(err)
			}
			if !result.OK {
				t.Fatal("missing successful control response")
			}
		})
	}
	for _, path := range []string{controlInstallPath, controlPullPath} {
		t.Run(path, func(t *testing.T) {
			assertPullFrame := func(frame streamFrame, revision, phase string) {
				t.Helper()
				if frame.OpID != pullOpID || frame.Engine != "fixture" || frame.Op != "pull" || frame.ID != downloadID || frame.Model != pullModel {
					t.Fatalf("pull frame identity = %+v", frame)
				}
				state := frame.DownloadState
				if state == nil || state.Version != modeldownload.Version || state.SourceID != sourceID || state.Revision != revision || state.RecoveryRequired || len(state.Entries) != 1 {
					t.Fatalf("pull frame download state = %+v", state)
				}
				entry := state.Entries[0]
				if entry.ID != downloadID || entry.Engine != "fixture" || entry.Model != pullModel || entry.Phase != phase || entry.BlocksReadiness != (phase == modeldownload.PhaseInProgress) {
					t.Fatalf("pull frame entry = %+v", entry)
				}
				if phase == modeldownload.PhaseInProgress {
					if frame.Percent != 37 || entry.Percent == nil || *entry.Percent != 37 {
						t.Fatalf("pull progress percent = %d/%v", frame.Percent, entry.Percent)
					}
				} else if entry.Percent != nil {
					t.Fatalf("terminal retained a measured progress percent: %+v", entry)
				}
			}
			progress := 0
			var request any = installRequest{OpID: "install-route-operation", Engine: "fixture"}
			if path == controlPullPath {
				request = pullRequest{OpID: pullOpID, Engine: "fixture", Model: pullModel}
			}
			result, err := client.stream(ctx, path, request, func(frame streamFrame) {
				if frame.Type != "progress" || frame.Stage != "fixture-stage" {
					t.Errorf("progress = %+v", frame)
				}
				if path == controlPullPath {
					assertPullFrame(frame, "1", modeldownload.PhaseInProgress)
				}
				progress++
			})
			if err != nil {
				t.Fatal(err)
			}
			if result.Type != "result" || progress != 1 {
				t.Fatalf("result/progress = %+v/%d", result, progress)
			}
			if path == controlPullPath {
				assertPullFrame(result, "2", modeldownload.PhaseComplete)
			}
			var outcome struct {
				OK bool `json:"ok"`
			}
			if err := json.Unmarshal(result.Result, &outcome); err != nil {
				t.Fatalf("decode streamed result: %v", err)
			}
			if !outcome.OK {
				t.Fatal("missing successful streamed result")
			}
		})
	}
	mu.Lock()
	defer mu.Unlock()
	for _, path := range []string{controlEnginesPath, controlStartPath, controlStopPath, controlLoadPath, controlUnloadPath, controlDeletePath, controlInstallPath, controlPullPath} {
		if seen[path] != 1 {
			t.Errorf("dispatches for %s = %d, want 1", path, seen[path])
		}
	}
	if f.reverseCalls.Load() != 8 || f.directCalls.Load() != 2 {
		t.Fatalf("reverse/direct = %d/%d, want 8/2 (ordinary and readiness routes)", f.reverseCalls.Load(), f.directCalls.Load())
	}
}

func TestRemoteReverseSafeStopAndSettingsPolicyOwners(t *testing.T) {
	exec := settingsExecutor(t, false)
	exec.nodeProjection.set(nodeProjection{ManagementBlocked: false})
	var mu sync.Mutex
	settingsCalls, policyCalls := map[string]int{}, map[string]int{}
	exec.settingsParent = func(_ context.Context, method string, p settings.Request, caller string) (json.RawMessage, error) {
		if caller != "uuid-self" || p.NodeID != "" || p.ExpectedRevision != 7 {
			t.Errorf("settings authority/request = %s/%+v", caller, p)
		}
		mu.Lock()
		settingsCalls[method]++
		mu.Unlock()
		return json.RawMessage(`{"engine":"ollama","revision":8}`), nil
	}
	exec.policyParent = func(_ context.Context, method string, params json.RawMessage, caller string) (json.RawMessage, error) {
		var body policyBody
		if err := json.Unmarshal(params, &body); err != nil {
			t.Error(err)
		}
		if caller != "uuid-self" || body.NodeID != "" {
			t.Errorf("policy authority/request = %s/%+v", caller, body)
		}
		mu.Lock()
		policyCalls[method]++
		mu.Unlock()
		return json.RawMessage(`{"availability":"paused"}`), nil
	}
	f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler { return (&controlServer{exec: exec, mesh: mesh}).mux() })
	grantRemoteManagement(t, f.peerDir, "1", map[string]bool{"uuid-self": true})
	ctx := remoteFixtureContext(t)
	client := f.client(t, ctx)
	raw, err := client.postJSON(ctx, controlStopPath, "fake", stopRequest{Engine: "fake"})
	if err != nil {
		t.Fatal(err)
	}
	var status EngineStatus
	if err := json.Unmarshal(raw, &status); err != nil {
		t.Fatal(err)
	}
	if status.Engine != "fake" || status.Running {
		t.Fatalf("safe stopped engine = %+v", status)
	}
	for _, method := range []string{"get", "preview", "apply"} {
		if _, err := client.postJSON(ctx, settingsPath+method, "ollama", settings.Request{Engine: "ollama", ExpectedRevision: 7}); err != nil {
			t.Fatal(err)
		}
	}
	for _, path := range []string{controlPolicyGetPath, controlPolicySetPath, controlAvailabilitySetPath} {
		if _, err := client.postPolicy(ctx, path, policyBody{State: "paused", Policy: json.RawMessage(`{"revision":4}`)}); err != nil {
			t.Fatal(err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	for _, method := range []string{"get", "preview", "apply"} {
		if settingsCalls[method] != 1 {
			t.Errorf("settings %s count=%d", method, settingsCalls[method])
		}
	}
	for _, method := range []string{"get", "set", "set-availability"} {
		if policyCalls[method] != 1 {
			t.Errorf("policy %s count=%d", method, policyCalls[method])
		}
	}
}

func TestRemoteReverseInvalidAuthorityNeverDispatches(t *testing.T) {
	test := func(name string, invalidate func(*testing.T, *remoteReverseFixture)) {
		t.Run(name, func(t *testing.T) {
			var dispatches atomic.Int32
			f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
				return (&controlServer{mesh: mesh}).requirePin(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					dispatches.Add(1)
					remoteFixtureEncode(t, w, map[string]bool{"ok": true})
				}))
			})
			ctx := remoteFixtureContext(t)
			client := f.client(t, ctx)
			invalidate(t, f)
			if _, err := client.postJSON(ctx, controlStopPath, "fixture", stopRequest{Engine: "fixture"}); err == nil {
				t.Fatal("invalid authority reached control handler")
			}
			if dispatches.Load() != 0 {
				t.Fatalf("mutation dispatched %d times, want zero", dispatches.Load())
			}
		})
	}
	test("removed pin", func(t *testing.T, f *remoteReverseFixture) { clustertrusttest.RemovePeerPin(t, f.selfDir, "uuid-peer") })
	test("left admission", func(t *testing.T, f *remoteReverseFixture) {
		if err := os.Remove(filepath.Join(f.selfDir, "admission.json")); err != nil {
			t.Fatal(err)
		}
	})
	test("wrong pin", func(t *testing.T, f *remoteReverseFixture) {
		clustertrusttest.WriteKeypair(t, f.peerDir, "uuid-peer")
		writePinFromCert(t, f.selfDir, "uuid-peer", filepath.Join(f.peerDir, "node.crt"))
	})
	test("route removed", func(_ *testing.T, f *remoteReverseFixture) { f.m.peers.set(nil) })
	test("disconnect before dispatch", func(t *testing.T, f *remoteReverseFixture) {
		if err := f.outer.Close(); err != nil {
			t.Fatal(err)
		}
	})
}

func TestRemoteMutationInterruptedAfterDispatchIsNeverReplayed(t *testing.T) {
	var dispatches atomic.Int32
	var f *remoteReverseFixture
	f = newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
		return (&controlServer{mesh: mesh}).requirePin(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if _, err := io.Copy(io.Discard, r.Body); err != nil {
				t.Error(err)
				return
			}
			dispatches.Add(1)
			w.Header().Set("Content-Length", "100")
			if _, err := io.WriteString(w, `{"ok":`); err != nil {
				t.Error(err)
				return
			}
			if err := http.NewResponseController(w).Flush(); err != nil {
				t.Error(err)
				return
			}
			if err := f.outer.Close(); err != nil {
				t.Error(err)
			}
		}))
	})
	ctx := remoteFixtureContext(t)
	client := f.client(t, ctx)
	if _, err := client.postJSON(ctx, controlStopPath, "fixture", stopRequest{Engine: "fixture"}); err == nil {
		t.Fatal("interrupted mutation reported success")
	}
	if dispatches.Load() != 1 || f.reverseCalls.Load() != 1 || f.directCalls.Load() != 1 {
		t.Fatalf("dispatch/reverse/direct = %d/%d/%d, want 1/1/1", dispatches.Load(), f.reverseCalls.Load(), f.directCalls.Load())
	}
}

func TestRemoteDispatchedDirectMutationDoesNotFallBack(t *testing.T) {
	test := func(name string, call func(context.Context, *remoteClient) error) {
		t.Run(name, func(t *testing.T) {
			var dispatches atomic.Int32
			f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
				return (&controlServer{mesh: mesh}).requirePin(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if _, err := io.Copy(io.Discard, r.Body); err != nil {
						t.Error(err)
						return
					}
					dispatches.Add(1)
					conn, _, err := http.NewResponseController(w).Hijack()
					if err != nil {
						t.Error(err)
						return
					}
					if err := conn.Close(); err != nil {
						t.Error(err)
					}
				}))
			})
			f.directHealthy.Store(true)
			ctx := remoteFixtureContext(t)
			if err := call(ctx, f.client(t, ctx)); err == nil {
				t.Fatal("lost mutation response reported success")
			}
			if dispatches.Load() != 1 || f.directCalls.Load() != 1 || f.reverseCalls.Load() != 0 {
				t.Fatalf("dispatch/direct/reverse = %d/%d/%d, want 1/1/0", dispatches.Load(), f.directCalls.Load(), f.reverseCalls.Load())
			}
		})
	}
	test("engine stop", func(ctx context.Context, c *remoteClient) error {
		_, err := c.postJSON(ctx, controlStopPath, "fixture", stopRequest{Engine: "fixture"})
		return err
	})
	test("settings apply", func(ctx context.Context, c *remoteClient) error {
		_, err := c.postJSON(ctx, settingsPath+"apply", "fixture", settings.Request{Engine: "fixture", ExpectedRevision: 7})
		return err
	})
	test("policy set", func(ctx context.Context, c *remoteClient) error {
		_, err := c.postPolicy(ctx, controlPolicySetPath, policyBody{Policy: json.RawMessage(`{"revision":4}`)})
		return err
	})
	test("availability set", func(ctx context.Context, c *remoteClient) error {
		_, err := c.postPolicy(ctx, controlAvailabilitySetPath, policyBody{State: "paused"})
		return err
	})
	test("install stream", func(ctx context.Context, c *remoteClient) error {
		_, err := c.stream(ctx, controlInstallPath, installRequest{Engine: "fixture"}, nil)
		return err
	})
	test("pull stream", func(ctx context.Context, c *remoteClient) error {
		_, err := c.stream(ctx, controlPullPath, pullRequest{OpID: "pull-dispatch-operation", Engine: "fixture", Model: "model"}, nil)
		return err
	})
}

func TestRemoteControlRouteTracksDiscoveryAndAdmission(t *testing.T) {
	f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
		return (&controlServer{mesh: mesh}).requirePin(http.NotFound)
	})
	ctx := remoteFixtureContext(t)
	_ = f.client(t, ctx)
	original, ok := f.m.resolveControlRoute("uuid-peer", noderec.ServiceEngineControl)
	if !ok || original.Address != f.server.Listener.Addr().String() || original.Protocol != clustertrust.ReverseTLS || original.Incarnation == "" {
		t.Fatalf("initial route = %+v, ok=%v", original, ok)
	}
	f.m.peers.set([]noderec.DirectoryNode{f.node})
	stable, ok := f.m.resolveControlRoute("uuid-peer", noderec.ServiceEngineControl)
	if !ok || stable != original {
		t.Fatalf("unchanged discovery altered route: %+v -> %+v", original, stable)
	}
	if _, ok := f.m.resolveControlRoute("uuid-peer", noderec.ServiceNodeInfo); ok {
		t.Fatal("ec route authorized another service")
	}
	clustertrusttest.WriteAdmission(t, f.selfDir, "cluster-1", 2)
	readmitted, ok := f.m.resolveControlRoute("uuid-peer", noderec.ServiceEngineControl)
	if !ok || readmitted.Incarnation == original.Incarnation {
		t.Fatal("admission epoch did not change route authority")
	}
	duplicate := f.node
	duplicate.HostUUID = "another-host"
	f.m.peers.set([]noderec.DirectoryNode{f.node, duplicate})
	if _, ok := f.m.resolveControlRoute("uuid-peer", noderec.ServiceEngineControl); ok {
		t.Fatal("ambiguous discovery principal was authorized")
	}
}

func testRemoteResponseBoundaries(t *testing.T, limit int, call func(context.Context, *remoteClient) (json.RawMessage, error)) {
	t.Helper()
	cases := []struct {
		name                         string
		size, declaredExtra          int
		status                       int
		wantOverflow, wantIncomplete bool
	}{
		{name: "complete exact cap", size: limit, status: http.StatusOK},
		{name: "valid JSON prefix plus one excess byte", size: limit + 1, status: http.StatusOK, wantOverflow: true},
		{name: "valid JSON prefix plus excess bytes", size: limit + 17, status: http.StatusOK, wantOverflow: true},
		{name: "valid JSON with incomplete framing below cap", size: 64, declaredExtra: 1, status: http.StatusOK, wantIncomplete: true},
		{name: "valid JSON with incomplete framing at cap", size: limit, declaredExtra: 1, status: http.StatusOK, wantIncomplete: true},
		{name: "HTTP rejection keeps status and diagnostic", size: 64, status: http.StatusConflict},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var dispatches atomic.Int32
			body := `{"ok":true}` + strings.Repeat(" ", tc.size-len(`{"ok":true}`))
			if tc.wantOverflow {
				body = body[:limit] + strings.Repeat("x", tc.size-limit)
			}
			f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
				return (&controlServer{mesh: mesh}).requirePin(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if _, err := io.Copy(io.Discard, r.Body); err != nil {
						t.Error(err)
						return
					}
					dispatches.Add(1)
					w.Header().Set("Content-Length", strconv.Itoa(len(body)+tc.declaredExtra))
					w.WriteHeader(tc.status)
					if _, err := io.WriteString(w, body); err != nil {
						t.Errorf("write response fixture: %v", err)
					}
				}))
			})
			f.directHealthy.Store(true)
			ctx := remoteFixtureContext(t)
			raw, err := call(ctx, f.client(t, ctx))
			if dispatches.Load() != 1 || f.directCalls.Load() != 1 || f.reverseCalls.Load() != 0 {
				t.Fatalf("dispatch/direct/reverse = %d/%d/%d, want 1/1/0", dispatches.Load(), f.directCalls.Load(), f.reverseCalls.Load())
			}
			if tc.wantOverflow || tc.wantIncomplete || tc.status != http.StatusOK {
				if err == nil || len(raw) != 0 {
					t.Fatalf("invalid response reported success: bytes=%d error=%v", len(raw), err)
				}
				if tc.wantIncomplete && !errors.Is(err, io.ErrUnexpectedEOF) {
					t.Fatalf("error = %v, want preserved unexpected EOF", err)
				}
				if tc.wantOverflow && !strings.Contains(err.Error(), "response exceeds "+strconv.Itoa(limit)+"-byte limit") {
					t.Fatalf("error = %v, want configured overflow boundary", err)
				}
				if tc.status != http.StatusOK && !strings.Contains(err.Error(), `HTTP 409: {"ok":true}`) {
					t.Fatalf("error = %v, want original HTTP status and diagnostic", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(raw) != limit {
				t.Fatalf("complete response size=%d, want %d", len(raw), limit)
			}
			var result struct {
				OK bool `json:"ok"`
			}
			if err := json.Unmarshal(raw, &result); err != nil {
				t.Fatal(err)
			}
			if !result.OK {
				t.Fatal("complete JSON response lost its result")
			}
		})
	}
}

func TestRemoteEngineResponseBoundaries(t *testing.T) {
	testRemoteResponseBoundaries(t, 8<<20, func(ctx context.Context, c *remoteClient) (json.RawMessage, error) {
		raw, err := c.postJSON(ctx, controlStopPath, "fixture", stopRequest{Engine: "fixture"})
		if err != nil && !strings.Contains(err.Error(), "HTTP 409") && !strings.Contains(err.Error(), "operation outcome may be uncertain") {
			t.Errorf("mutation error lacks uncertain outcome: %v", err)
		}
		return raw, err
	})
}

func TestRemotePolicyResponseBoundaries(t *testing.T) {
	testRemoteResponseBoundaries(t, 256<<10, func(ctx context.Context, c *remoteClient) (json.RawMessage, error) {
		raw, err := c.postPolicy(ctx, controlPolicySetPath, policyBody{Policy: json.RawMessage(`{"revision":4}`)})
		if err != nil && !strings.Contains(err.Error(), "HTTP 409") && !strings.Contains(err.Error(), "operation outcome may be uncertain") {
			t.Errorf("mutation error lacks uncertain outcome: %v", err)
		}
		return raw, err
	})
}

func TestRemoteInventoryResponseBoundaries(t *testing.T) {
	testRemoteResponseBoundaries(t, 8<<20, func(ctx context.Context, c *remoteClient) (json.RawMessage, error) { return c.getEngines(ctx) })
}

func remoteEventFixture(t *testing.T) (*remoteReverseFixture, <-chan settings.Snapshot, <-chan struct{}, *atomic.Int32) {
	t.Helper()
	exec := settingsExecutor(t, false)
	mutations := &atomic.Int32{}
	exec.settingsParent = func(context.Context, string, settings.Request, string) (json.RawMessage, error) {
		mutations.Add(1)
		return nil, errors.New("event watcher must not invoke settings operations")
	}
	exec.settingsHub.Publish([]settings.Snapshot{{Engine: "ollama", Revision: 7, Epoch: "epoch-peer", Sequence: 9}})
	closed := make(chan struct{}, 8)
	f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
		handler := (&controlServer{exec: exec, mesh: mesh}).mux()
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			handler.ServeHTTP(w, r)
			if r.URL.Path == settingsPath+"events" {
				select {
				case closed <- struct{}{}:
				default:
				}
			}
		})
	})
	emitted := make(chan settings.Snapshot, 8)
	f.m.exec.emit = func(method string, value any) {
		if method != "engine:settings-changed" {
			return
		}
		body, err := json.Marshal(value)
		if err != nil {
			t.Error(err)
			return
		}
		var snapshot settings.Snapshot
		if err := json.Unmarshal(body, &snapshot); err != nil {
			t.Error(err)
			return
		}
		select {
		case emitted <- snapshot:
		default:
			t.Error("too many settings events")
		}
	}
	return f, emitted, closed, mutations
}

func remoteAwaitSnapshot(t *testing.T, events <-chan settings.Snapshot) {
	t.Helper()
	select {
	case snapshot := <-events:
		if snapshot.NodeID != "host-peer" || snapshot.Engine != "ollama" || snapshot.Revision != 7 || snapshot.Epoch != "epoch-peer" || snapshot.Sequence != 9 {
			t.Fatalf("settings event = %+v", snapshot)
		}
	case <-time.After(7 * time.Second):
		t.Fatal("settings event did not arrive")
	}
}

func remoteAwaitClosed(t *testing.T, closed <-chan struct{}) {
	t.Helper()
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		t.Fatal("event handler did not tear down")
	}
}

func TestRemoteSettingsEventsReconnectThroughExistingOwner(t *testing.T) {
	f, events, closed, mutations := remoteEventFixture(t)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	t.Cleanup(cancel)
	watchDone := make(chan struct{})
	go func() { defer close(watchDone); f.m.watchPeerSettings(ctx) }()
	remoteAwaitSnapshot(t, events)
	if err := f.outer.Close(); err != nil {
		t.Fatal(err)
	}
	remoteAwaitClosed(t, closed)
	f.connect(t)
	remoteAwaitSnapshot(t, events)
	cancel()
	remoteAwaitClosed(t, closed)
	select {
	case <-watchDone:
	case <-time.After(3 * time.Second):
		t.Fatal("subscription owner did not stop")
	}
	if mutations.Load() != 0 || f.reverseCalls.Load() != 2 {
		t.Fatalf("settings operations/reverse streams = %d/%d, want 0/2", mutations.Load(), f.reverseCalls.Load())
	}
}

func TestRemoteSettingsEventsRetireChangedAuthority(t *testing.T) {
	test := func(name string, change func(*testing.T, *remoteReverseFixture, context.CancelFunc)) {
		t.Run(name, func(t *testing.T) {
			f, events, closed, mutations := remoteEventFixture(t)
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			t.Cleanup(cancel)
			peer, ok := f.m.peers.lookup("host-peer")
			if !ok {
				t.Fatal("missing peer")
			}
			done := make(chan struct{})
			go func() { defer close(done); f.m.consumeSettingsStream(ctx, peer) }()
			remoteAwaitSnapshot(t, events)
			change(t, f, cancel)
			remoteAwaitClosed(t, closed)
			select {
			case <-done:
			case <-time.After(3 * time.Second):
				t.Fatal("settings consumer did not stop")
			}
			if mutations.Load() != 0 || f.reverseCalls.Load() != 1 {
				t.Fatalf("settings operations/streams = %d/%d, want 0/1", mutations.Load(), f.reverseCalls.Load())
			}
		})
	}
	test("caller cancellation", func(_ *testing.T, _ *remoteReverseFixture, cancel context.CancelFunc) { cancel() })
	test("peer pin removed", func(t *testing.T, f *remoteReverseFixture, _ context.CancelFunc) {
		clustertrusttest.RemovePeerPin(t, f.selfDir, "uuid-peer")
	})
	test("peer certificate replaced", func(t *testing.T, f *remoteReverseFixture, _ context.CancelFunc) {
		clustertrusttest.WriteKeypair(t, f.peerDir, "uuid-peer")
		writePinFromCert(t, f.selfDir, "uuid-peer", filepath.Join(f.peerDir, "node.crt"))
	})
	test("local certificate replaced", func(t *testing.T, f *remoteReverseFixture, _ context.CancelFunc) {
		clustertrusttest.WriteKeypair(t, f.selfDir, "uuid-self")
	})
	test("admission epoch changed", func(t *testing.T, f *remoteReverseFixture, _ context.CancelFunc) {
		clustertrusttest.WriteAdmission(t, f.selfDir, "cluster-1", 2)
	})
	test("endpoint generation changed", func(_ *testing.T, f *remoteReverseFixture, _ context.CancelFunc) {
		changed := f.node
		changed.Services = map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: f.node.Services[noderec.ServiceEngineControl].Port + 1}}
		f.m.peers.set([]noderec.DirectoryNode{changed})
	})
}
