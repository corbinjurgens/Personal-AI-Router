// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	settings "nvpair-shared/enginesettings"
)

func TestRemoteControlRecoveryGateWaitsForCurrentManagerAcknowledgement(t *testing.T) {
	selfDir, peerDir := pairedPolicyDirs(t)
	// The recovery gate is what this test exercises, so the mutating peer
	// needs the grant that otherwise refuses it before the handler runs.
	grantRemoteManagement(t, peerDir, "1", map[string]bool{"uuid-self": true})
	exec := newTestExecutor(t, testEngineManifest(fakeEngineBin))
	t.Cleanup(exec.StopAll)
	if err := exec.desired.set("fake", true, intentLocal); err != nil {
		t.Fatalf("save installed engine On: %v", err)
	}
	mesh := clustertrust.Open(peerDir)
	out := &syncBuffer{}
	m := NewManager(NewCodec(out), exec, mesh, peerDir)
	t.Cleanup(m.controlHTTP.CloseIdle)
	t.Cleanup(m.controlReadyHTTP.CloseIdle)
	// This is the startup projection delivered even while broker recovery is
	// blocked. It must not itself enable the independent direct peer surface.
	m.applyNodeProjection(&Message{Params: json.RawMessage(`{"policy":{"revision":"1","roles":{}},"pausedLocally":false}`)})
	server := httptest.NewUnstartedServer((&controlServer{exec: exec, mesh: mesh}).mux())
	server.TLS = mesh.ServerTLSConfig()
	server.StartTLS()
	t.Cleanup(server.Close)
	peer := clustertrust.Open(selfDir)
	tlsConfig, ok := peer.ClientTLSConfig("uuid-peer")
	if !ok {
		t.Fatal("paired client pin unavailable")
	}
	transport := &http.Transport{TLSClientConfig: tlsConfig}
	t.Cleanup(transport.CloseIdleConnections)
	client := &http.Client{Transport: transport, Timeout: 20 * time.Second}
	request := func(method, path, body string) *http.Response {
		t.Helper()
		req, err := http.NewRequest(method, server.URL+path, strings.NewReader(body))
		if err != nil {
			t.Fatalf("build authenticated control request: %v", err)
		}
		res, err := client.Do(req)
		if err != nil {
			t.Fatalf("authenticated control request: %v", err)
		}
		t.Cleanup(func() { _ = res.Body.Close() })
		return res
	}
	// Broker recovery has not acknowledged success. Every mutating direct
	// route, including stop and install-without-start, remains unavailable.
	for _, path := range []string{controlStartPath, controlStopPath, controlInstallPath,
		controlPullPath, controlLoadPath, controlUnloadPath, controlDeletePath,
		settingsPath + "apply", controlPolicySetPath, controlAvailabilitySetPath, settings.SwitchApplyPath} {
		if res := request(http.MethodPost, path, `{"engine":"fake"}`); res.StatusCode != http.StatusServiceUnavailable {
			t.Fatalf("%s while recovery pending = %d, want 503", path, res.StatusCode)
		}
	}
	if res := request(http.MethodGet, controlEnginesPath, ""); res.StatusCode != http.StatusOK {
		t.Fatalf("read-only inventory while recovery pending = %d, want 200", res.StatusCode)
	}
	if st, err := exec.Status("fake"); err != nil || st.Running {
		t.Fatalf("hidden manager started the saved-On engine: %+v, %v", st, err)
	}
	ack := func(id, incarnation string) Message {
		t.Helper()
		params, err := json.Marshal(map[string]string{"managerIncarnation": incarnation})
		if err != nil {
			t.Fatalf("encode recovery acknowledgement: %v", err)
		}
		rawID := json.RawMessage(id)
		m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &rawID, Method: switchRecoveryReadyMethod, Params: params})
		frames := out.frames()
		if len(frames) == 0 {
			t.Fatal("missing recovery acknowledgement reply")
		}
		return frames[len(frames)-1]
	}
	if reply := ack("1", "old-manager"); reply.Error == nil {
		t.Fatal("a different incarnation opened the recovery gate")
	}
	if res := request(http.MethodPost, controlStartPath, `{"engine":"fake"}`); res.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("start after stale acknowledgement = %d, want 503", res.StatusCode)
	}
	if reply := ack("2", exec.switchIncarnation); reply.Error != nil {
		t.Fatalf("current manager recovery acknowledgement: %+v", reply.Error)
	}
	res := request(http.MethodPost, controlStartPath, `{"engine":"fake"}`)
	if res.StatusCode != http.StatusOK {
		t.Fatalf("start after successful recovery = %d, want 200", res.StatusCode)
	}
	var status EngineStatus
	if err := json.NewDecoder(res.Body).Decode(&status); err != nil {
		t.Fatalf("decode started engine status: %v", err)
	}
	if status.Engine != "fake" || !status.Running {
		t.Fatalf("start after successful recovery returned %+v", status)
	}
}

func TestSwitchRecoveryReadyRetainsGateForUnreleasedCheckpoint(t *testing.T) {
	exec, p := switchFixture(t)
	prepareSwitch(t, exec, p)
	exec.switchRecoveryPending.Store(true)
	if err := exec.completeSwitchRecovery(exec.switchIncarnation); err == nil {
		t.Fatal("recovery acknowledged an unreleased checkpoint")
	}
	if !exec.switchRecoveryPending.Load() {
		t.Fatal("unreleased checkpoint opened the direct peer mutation gate")
	}
}

// postControl drives one ec handler as a paired node's request does.
func postControl(t *testing.T, handler func(http.ResponseWriter, *http.Request), path, body string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
	return rec
}

// requireLocalVeto fails unless rec is the typed 409 of a local veto.
func requireLocalVeto(t *testing.T, what string, rec *httptest.ResponseRecorder) {
	t.Helper()
	var body codedBody
	if rec.Code != http.StatusConflict || json.Unmarshal(rec.Body.Bytes(), &body) != nil || body.Code != "local_veto" || body.Error == "" {
		t.Fatalf("%s = %d %q, want a 409 local_veto refusal", what, rec.Code, rec.Body.String())
	}
}

// vetoTestServer is a control server on a fake-engine executor with its own
// desired-state directory.
func vetoTestServer(t *testing.T, manifest *Manifest) (*controlServer, *Executor) {
	t.Helper()
	exec := newTestExecutor(t, manifest)
	t.Cleanup(exec.StopAll)
	return &controlServer{exec: exec}, exec
}

func TestRemoteStartAndLoadFailClosedBeforeTheFirstProjection(t *testing.T) {
	s, exec := vetoTestServer(t, testEngineManifest(fakeEngineBin))
	for _, c := range []struct {
		what    string
		handler func(http.ResponseWriter, *http.Request)
		path    string
		body    string
	}{
		{"remote start", s.handleStart, controlStartPath, `{"engine":"fake"}`},
		{"remote install with start", s.handleInstall, controlInstallPath, `{"engine":"fake","start":true}`},
		{"remote load", s.handleLoad, controlLoadPath, `{"engine":"fake","model":"tiny"}`},
	} {
		if rec := postControl(t, c.handler, c.path, c.body); rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("%s before the first projection = %d %q, want 503", c.what, rec.Code, rec.Body.String())
		}
	}
	if st, err := exec.Status("fake"); err != nil || st.Running {
		t.Fatalf("a refused remote start ran the engine: %+v %v", st, err)
	}
}

func TestRemoteStartInstallAndLoadAreVetoedWhileLocallyPaused(t *testing.T) {
	s, exec := vetoTestServer(t, testEngineManifest(fakeEngineBin))
	exec.nodeProjection.set(nodeProjection{PausedLocally: true})
	requireLocalVeto(t, "remote start", postControl(t, s.handleStart, controlStartPath, `{"engine":"fake"}`))
	requireLocalVeto(t, "remote install with start", postControl(t, s.handleInstall, controlInstallPath, `{"engine":"fake","start":true}`))
	requireLocalVeto(t, "remote load", postControl(t, s.handleLoad, controlLoadPath, `{"engine":"fake","model":"tiny"}`))
	if st, err := exec.Status("fake"); err != nil || st.Running {
		t.Fatalf("a vetoed remote start ran the engine: %+v %v", st, err)
	}
	if rec := postControl(t, s.handleStop, controlStopPath, `{"engine":"fake"}`); rec.Code != http.StatusOK {
		t.Fatalf("remote stop while paused = %d %q, want it allowed", rec.Code, rec.Body.String())
	}
	if rec := postControl(t, s.handleUnload, controlUnloadPath, `{"engine":"fake","model":"tiny"}`); rec.Code == http.StatusConflict {
		t.Fatalf("remote unload while paused was vetoed: %q", rec.Body.String())
	}
	// The node's own pause never vetoes a local start.
	if err := exec.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("local start while paused: %v", err)
	}
}

func TestRemoteStartIsVetoedByALocalOffUntilALocalStart(t *testing.T) {
	s, exec := vetoTestServer(t, testEngineManifest(fakeEngineBin))
	exec.nodeProjection.set(nodeProjection{})
	if err := exec.Stop("fake"); err != nil {
		t.Fatalf("local stop: %v", err)
	}
	requireLocalVeto(t, "remote start after a local Off", postControl(t, s.handleStart, controlStartPath, `{"engine":"fake"}`))
	requireLocalVeto(t, "remote load after a local Off", postControl(t, s.handleLoad, controlLoadPath, `{"engine":"fake","model":"tiny"}`))
	if rec := postControl(t, s.handleStop, controlStopPath, `{"engine":"fake"}`); rec.Code != http.StatusOK {
		t.Fatalf("remote stop = %d %q", rec.Code, rec.Body.String())
	}
	requireLocalVeto(t, "remote start after a peer stop over a local Off", postControl(t, s.handleStart, controlStartPath, `{"engine":"fake"}`))

	if err := exec.Start(context.Background(), "fake"); err != nil {
		t.Fatalf("local start after a vetoed remote start: %v", err)
	}
	if off, err := exec.desired.offLocally("fake"); err != nil || off {
		t.Fatalf("local start left the local Off: %v %v", off, err)
	}
	if rec := postControl(t, s.handleStop, controlStopPath, `{"engine":"fake"}`); rec.Code != http.StatusOK {
		t.Fatalf("remote stop = %d %q", rec.Code, rec.Body.String())
	}
	if rec := postControl(t, s.handleStart, controlStartPath, `{"engine":"fake"}`); rec.Code != http.StatusOK {
		t.Fatalf("remote start after a remote stop = %d %q, want it allowed", rec.Code, rec.Body.String())
	}
}

func TestLegacyOffWithoutProvenanceVetoesRemoteStart(t *testing.T) {
	s, exec := vetoTestServer(t, testEngineManifest(fakeEngineBin))
	exec.nodeProjection.set(nodeProjection{})
	if err := os.WriteFile(filepath.Join(exec.baseDir, desiredStateFileName), []byte(`{"engines":{"fake":false}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	requireLocalVeto(t, "remote start of a legacy Off", postControl(t, s.handleStart, controlStartPath, `{"engine":"fake"}`))
}

// slowStartManifest is a fake engine that never becomes ready, so a start
// waits in its readiness probe until it is canceled.
func slowStartManifest() *Manifest {
	m := testEngineManifest(fakeEngineBin)
	key := runtime.GOOS + "/" + runtime.GOARCH
	p := m.Platforms[key]
	p.Runtime.Args = []string{"noserve"}
	p.Runtime.Ready.TimeoutS = 60
	m.Platforms[key] = p
	return m
}

// startRemotely runs a remote start in the background, waits until it is
// accepted, and returns the channel its response arrives on.
func startRemotely(t *testing.T, s *controlServer) <-chan *httptest.ResponseRecorder {
	t.Helper()
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		rec := httptest.NewRecorder()
		s.handleStart(rec, httptest.NewRequest(http.MethodPost, controlStartPath, strings.NewReader(`{"engine":"fake"}`)))
		done <- rec
	}()
	deadline := time.Now().Add(5 * time.Second)
	for {
		s.exec.remoteWork.mu.Lock()
		accepted := len(s.exec.remoteWork.ops) == 1
		s.exec.remoteWork.mu.Unlock()
		if accepted {
			return done
		}
		if time.Now().After(deadline) {
			t.Fatal("the remote start was not accepted")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func awaitResponse(t *testing.T, done <-chan *httptest.ResponseRecorder) *httptest.ResponseRecorder {
	t.Helper()
	select {
	case rec := <-done:
		return rec
	case <-time.After(20 * time.Second):
		t.Fatal("the accepted remote start was not canceled")
		return nil
	}
}

func TestLocalVetoCancelsAnAcceptedRemoteStart(t *testing.T) {
	t.Run("local pause", func(t *testing.T) {
		s, exec := vetoTestServer(t, slowStartManifest())
		m := &Manager{exec: exec}
		m.applyNodeProjection(&Message{Params: json.RawMessage(`{"policy":{"revision":"1","roles":{}},"pausedLocally":false}`)})
		done := startRemotely(t, s)
		m.applyNodeProjection(&Message{Params: json.RawMessage(`{"policy":{"revision":"2","roles":{},"pausedBy":"local"},"pausedLocally":true}`)})
		requireLocalVeto(t, "remote start canceled by a local pause", awaitResponse(t, done))
		if enabled, known, err := exec.desired.get("fake"); err != nil || (known && enabled) {
			t.Fatalf("the canceled remote start saved On: %v %v %v", enabled, known, err)
		}
	})
	t.Run("local stop", func(t *testing.T) {
		s, exec := vetoTestServer(t, slowStartManifest())
		exec.nodeProjection.set(nodeProjection{})
		done := startRemotely(t, s)
		if err := exec.Stop("fake"); err != nil {
			t.Fatalf("local stop during a remote start: %v", err)
		}
		requireLocalVeto(t, "remote start canceled by a local stop", awaitResponse(t, done))
		if off, err := exec.desired.offLocally("fake"); err != nil || !off {
			t.Fatalf("the local stop did not save a local Off: %v %v", off, err)
		}
	})
	t.Run("a peer pause does not cancel", func(t *testing.T) {
		_, exec := vetoTestServer(t, testEngineManifest(fakeEngineBin))
		m := &Manager{exec: exec}
		exec.nodeProjection.set(nodeProjection{})
		ctx, done, err := exec.beginRemote(context.Background(), "fake")
		if err != nil {
			t.Fatalf("begin remote work: %v", err)
		}
		defer done()
		m.applyNodeProjection(&Message{Params: json.RawMessage(`{"policy":{"revision":"2","roles":{},"pausedBy":"peer"},"pausedLocally":false}`)})
		if ctx.Err() != nil {
			t.Fatal("a peer pause canceled accepted remote work")
		}
	})
}

func TestRemoteClientMapsTypedRefusalsToRPCCodes(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case controlStartPath:
			writeCodedError(w, codeLocalVeto, nodePausedVeto)
		case controlStopPath:
			http.Error(w, "plain conflict", http.StatusConflict)
		case controlInstallPath:
			w.Header().Set("Content-Type", "application/x-ndjson")
			_ = json.NewEncoder(w).Encode(streamFrame{Type: "error", Message: nodePausedVeto, Code: "local_veto"})
		}
	}))
	defer srv.Close()
	base := http.DefaultTransport.(*http.Transport)
	c := &remoteClient{controlHTTP: newRemoteHTTPClient(base, time.Second), controlReadyHTTP: newRemoteHTTPClient(base, time.Second), base: srv.URL}

	_, err := c.postJSON(context.Background(), controlStartPath, "fake", startRequest{Engine: "fake"})
	if rpcCode(err) != codeLocalVeto || !strings.HasSuffix(err.Error(), nodePausedVeto) {
		t.Fatalf("vetoed remote start = %v (code %d), want %d with the target's reason", err, rpcCode(err), codeLocalVeto)
	}
	_, err = c.postJSON(context.Background(), controlStopPath, "fake", stopRequest{Engine: "fake"})
	if err == nil || rpcCode(err) != -32000 {
		t.Fatalf("untyped 409 = %v (code %d), want an ordinary -32000 failure", err, rpcCode(err))
	}
	_, err = c.stream(context.Background(), controlInstallPath, installRequest{Engine: "fake", Start: true}, nil)
	if rpcCode(err) != codeLocalVeto || err.Error() != nodePausedVeto {
		t.Fatalf("vetoed streamed install = %v (code %d), want %d", err, rpcCode(err), codeLocalVeto)
	}
}
