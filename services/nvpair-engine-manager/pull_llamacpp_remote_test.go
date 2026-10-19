// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	settings "nvpair-shared/enginesettings"
)

func TestLlamaCPPPullRemoteDisconnectStopsDownload(t *testing.T) {
	f := newLlamaCPPPullFixture(t)
	stopped := make(chan struct{})
	f.unload = func(w http.ResponseWriter, _ *http.Request) {
		f.downloading.Store(false)
		if _, err := fmt.Fprint(w, `{"success":true}`); err != nil {
			t.Errorf("write stop response: %v", err)
		}
		close(stopped)
	}
	s := &controlServer{exec: f.ex}
	server := httptest.NewServer(http.HandlerFunc(s.handlePull))
	t.Cleanup(server.Close)
	requestLlamaCPPPullAndDisconnect(t, server.Client(), server.URL, f.started)
	waitLlamaCPPPullSignal(t, stopped)
	if f.downloading.Load() || f.unloads.Load() != 1 {
		t.Fatalf("downloading=%t unloads=%d, want remote disconnect to stop download", f.downloading.Load(), f.unloads.Load())
	}
}

// Exercise the real ec listener, pinned mTLS, request cancellation, and cleanup
// in the compiled manager, without a real engine or model download.
func TestE2ELlamaCPPPullRemoteDisconnectStopsDownload(t *testing.T) {
	f := newLlamaCPPPullFixture(t)
	stopped := make(chan struct{})
	f.unload = func(w http.ResponseWriter, _ *http.Request) {
		f.downloading.Store(false)
		if _, err := fmt.Fprint(w, `{"success":true}`); err != nil {
			t.Errorf("write stop response: %v", err)
		}
		close(stopped)
	}
	serverCert, serverKey := mintLeaf(t, "pull-server")
	clientCert, clientKey := mintLeaf(t, "pull-client")
	serverDir := clusterDirFor(t, serverCert, serverKey, map[string][]byte{"pull-client": clientCert})
	clientDir := clusterDirFor(t, clientCert, clientKey, map[string][]byte{"pull-server": serverCert})
	controlPort, err := freePort()
	if err != nil {
		t.Fatalf("allocate ec port: %v", err)
	}
	state, err := f.ex.state("fake")
	if err != nil {
		t.Fatalf("resolve fake router: %v", err)
	}
	manifest := testEngineManifest(fakeEngineBin)
	manifest.Actions[pullModelAction] = Action{
		HTTP:             &ActionHTTP{Method: http.MethodPost, Path: "/models"},
		ProgressProtocol: pullProgressProtocolLlamaCPPModelsSSE,
	}
	for key, platform := range manifest.Platforms {
		platform.Runtime.Port = state.port
		platform.Runtime.Ready.HTTP = "http://127.0.0.1:{port}/health"
		platform.Runtime.Health = nil
		manifest.Platforms[key] = platform
	}
	cfg, home := t.TempDir(), t.TempDir()
	for _, dir := range []string{
		filepath.Join(cfg, "Nvidia Corporation", "Personal AI Router", "engines"),
		filepath.Join(home, "Library", "Application Support", "Nvidia Corporation", "Personal AI Router", "engines"),
	} {
		writeE2EManifest(t, dir, manifest)
	}
	manager := startE2EManager(t, cfg, home, "--control-port", strconv.Itoa(controlPort), "--cluster-dir", serverDir, "--loaded-poll-interval", "0")
	// The fake router is already listening. Start adopts it using its readiness
	// probe; the child never spawns a real engine or another fake listener.
	send(t, manager.stdin, 1, "engine:start", map[string]string{"engine": "fake"})
	var status EngineStatus
	if err := json.Unmarshal(waitResult(t, manager.frames, "1", 10*time.Second), &status); err != nil {
		t.Fatalf("decode start status: %v", err)
	}
	if !status.Running || status.Port != state.port {
		t.Fatalf("engine status = %+v, want running fake router at %d", status, state.port)
	}
	// This fixture owns the parent pipe. Complete its conservative startup
	// handshake before asking the independently authenticated ec mutation path.
	send(t, manager.stdin, 2, settings.SwitchPrepareMethod, settings.SwitchPrepareRequest{})
	var inspected switchExecutorResult
	if err := json.Unmarshal(waitResult(t, manager.frames, "2", 10*time.Second), &inspected); err != nil {
		t.Fatalf("decode recovery inspection: %v", err)
	}
	if inspected.Error != nil || inspected.Checkpoint != nil || inspected.Configuration.ManagerIncarnation == "" {
		t.Fatalf("unexpected startup recovery state: %+v", inspected)
	}
	send(t, manager.stdin, 3, switchRecoveryReadyMethod, map[string]string{"managerIncarnation": inspected.Configuration.ManagerIncarnation})
	var ready struct {
		ManagerIncarnation string `json:"managerIncarnation"`
	}
	if err := json.Unmarshal(waitResult(t, manager.frames, "3", 10*time.Second), &ready); err != nil {
		t.Fatalf("decode recovery acknowledgement: %v", err)
	}
	if ready.ManagerIncarnation != inspected.Configuration.ManagerIncarnation {
		t.Fatalf("recovery acknowledgement = %+v", ready)
	}
	waitPortServing(t, controlPort)
	tlsConfig, ok := clustertrust.Open(clientDir).ClientTLSConfig("pull-server")
	if !ok {
		t.Fatal("client could not resolve pinned server TLS configuration")
	}
	transport := &http.Transport{TLSClientConfig: tlsConfig}
	t.Cleanup(transport.CloseIdleConnections)
	client := &http.Client{Transport: transport}
	requestLlamaCPPPullAndDisconnect(t, client, fmt.Sprintf("https://127.0.0.1:%d", controlPort), f.started)
	waitLlamaCPPPullSignal(t, stopped)
	if f.downloading.Load() || f.unloads.Load() != 1 {
		t.Fatalf("downloading=%t unloads=%d, want compiled manager to stop download", f.downloading.Load(), f.unloads.Load())
	}
	// Shut the fixture down first: the adopted listener is owned by this test,
	// and manager shutdown must not attempt to terminate the test process.
	f.server.Close()
	manager.stop(t)
}

func requestLlamaCPPPullAndDisconnect(t *testing.T, client *http.Client, baseURL string, started <-chan struct{}) {
	t.Helper()
	body, err := json.Marshal(pullRequest{OpID: "disconnect-test", Engine: "fake", Model: llamaCPPPullTestModel})
	if err != nil {
		t.Fatalf("encode remote pull: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+controlPullPath, bytes.NewReader(body))
	if err != nil {
		t.Fatalf("create remote pull: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("start remote pull: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("remote pull returned HTTP %d", resp.StatusCode)
	}
	waitLlamaCPPPullSignal(t, started)
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("disconnect remote pull: %v", err)
	}
}
