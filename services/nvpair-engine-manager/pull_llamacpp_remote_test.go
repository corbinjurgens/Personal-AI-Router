// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modeldownload"
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
	// The ec listener runs in the compiled manager below, so the pinned pull
	// client needs its grant on disk before it starts.
	grantRemoteManagement(t, serverDir, "1", map[string]bool{"pull-client": true})
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
		// The journal is loaded at executor construction, before an adopted
		// engine could create its install directory. Supply that private root.
		if err := os.MkdirAll(filepath.Join(filepath.Dir(dir), "engine-bin"), 0o700); err != nil {
			t.Fatalf("create fixture download journal directory: %v", err)
		}
	}
	manager := startE2EManager(t, cfg, home, "--control-port", strconv.Itoa(controlPort), "--cluster-dir", serverDir, "--loaded-poll-interval", "0")
	notify(t, manager.stdin, nodeProjectionMethod, nodeProjection{ManagementBlocked: false})
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
		data, err := readRemoteResponse(resp.Body, 64<<10)
		if err != nil {
			t.Fatalf("remote pull returned HTTP %d; read refusal: %v", resp.StatusCode, err)
		}
		t.Fatalf("remote pull returned HTTP %d: %s", resp.StatusCode, data)
	}
	waitLlamaCPPPullSignal(t, started)
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("disconnect remote pull: %v", err)
	}
}

func TestModelDownloadProgressSubscriptionUsesExactIdentity(t *testing.T) {
	hub := newProgressHub()
	events, cancel := hub.subscribeDownload("target-id", "fake", "model-a")
	defer cancel()
	generic, cancelGeneric := hub.subscribe("fake")
	defer cancelGeneric()
	for _, event := range []ProgressEvent{
		{ID: "other-id", Engine: "fake", Model: "model-a", Op: "pull"},
		{ID: "target-id", Engine: "other", Model: "model-a", Op: "pull"},
		{ID: "target-id", Engine: "fake", Model: "model-b", Op: "pull"},
		{ID: "target-id", Engine: "fake", Model: "model-a", Op: "install"},
		{Engine: "fake", Op: "pull"},
	} {
		hub.publish(event)
	}
	select {
	case event := <-events:
		t.Fatalf("unrelated progress reached download: %+v", event)
	default:
	}
	select {
	case <-generic:
	default:
		t.Fatal("generic engine subscription lost advisory events")
	}
	hub.publish(ProgressEvent{ID: "target-id", Engine: "fake", Model: "model-a", Op: "pull", Percent: 37})
	select {
	case event := <-events:
		if event.ID != "target-id" || event.Model != "model-a" || event.Percent != 37 {
			t.Fatalf("download progress = %+v", event)
		}
	default:
		t.Fatal("matching progress was not delivered")
	}
	cancel()
	if _, open := <-events; open {
		t.Fatal("cancelled download subscription remains open")
	}
}

func modelDownloadRemoteTestFrame(kind, revision, phase string) streamFrame {
	const targetID = "22222222222222222222222222222222"
	return streamFrame{
		Type: kind, OpID: "request-op", Engine: "fake", Op: "pull", ID: targetID, Model: "model-a",
		DownloadState: &modeldownload.Snapshot{
			Version: modeldownload.Version, SourceID: "11111111111111111111111111111111", Revision: revision,
			Entries: []modeldownload.Entry{{ID: targetID, Engine: "fake", Model: "model-a", Phase: phase, BlocksReadiness: phase != modeldownload.PhaseComplete}},
		},
	}
}

func TestModelDownloadRemoteDecoderRejectsWrongIdentity(t *testing.T) {
	test := func(name string, change func(*streamFrame)) {
		t.Run(name, func(t *testing.T) {
			queued := modelDownloadRemoteTestFrame("progress", "1", modeldownload.PhaseQueued)
			terminal := modelDownloadRemoteTestFrame("result", "2", modeldownload.PhaseComplete)
			change(&terminal)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				encoder := json.NewEncoder(w)
				for _, frame := range []streamFrame{queued, terminal} {
					if err := encoder.Encode(frame); err != nil {
						t.Errorf("encode stream frame: %v", err)
						return
					}
				}
			}))
			defer server.Close()
			client := &remoteClient{controlHTTP: server.Client(), base: server.URL}
			progress := 0
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			result, err := client.stream(ctx, controlPullPath, pullRequest{OpID: "request-op", Engine: "fake", Model: "model-a"}, func(streamFrame) { progress++ })
			if err == nil || result.Type == "result" || progress != 1 {
				t.Fatalf("result=%+v error=%v forwarded progress=%d, want protocol failure after queued", result, err, progress)
			}
		})
	}
	test("wrong request op ID", func(frame *streamFrame) { frame.OpID = "other-request" })
	test("wrong engine", func(frame *streamFrame) { frame.Engine = "other" })
	test("wrong operation", func(frame *streamFrame) { frame.Op = "install" })
	test("wrong model", func(frame *streamFrame) { frame.Model = "model-b" })
	test("missing target ID", func(frame *streamFrame) { frame.ID = "" })
	test("missing lifecycle support", func(frame *streamFrame) { frame.DownloadState = nil })
	test("target ID replaced", func(frame *streamFrame) {
		frame.ID = "33333333333333333333333333333333"
		frame.DownloadState.Entries[0].ID = frame.ID
	})
	test("target source replaced", func(frame *streamFrame) { frame.DownloadState.SourceID = "33333333333333333333333333333333" })
	test("noncanonical source", func(frame *streamFrame) { frame.DownloadState.SourceID = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" })
	test("noncanonical revision", func(frame *streamFrame) { frame.DownloadState.Revision = "02" })
	test("revision goes backwards", func(frame *streamFrame) { frame.DownloadState.Revision = "0" })
	test("same revision changes state", func(frame *streamFrame) { frame.DownloadState.Revision = "1" })
	test("snapshot has another model", func(frame *streamFrame) { frame.DownloadState.Entries[0].Model = "model-b" })
	test("snapshot omits operation", func(frame *streamFrame) { frame.DownloadState.Entries = []modeldownload.Entry{} })
	test("success with active download", func(frame *streamFrame) {
		frame.DownloadState.Entries[0].Phase = modeldownload.PhaseInProgress
		frame.DownloadState.Entries[0].BlocksReadiness = true
	})
	test("error with completed download", func(frame *streamFrame) { frame.Type = "error" })
	test("invalid progress identity", func(frame *streamFrame) { frame.Type, frame.OpID = "progress", "other-request" })
}

func TestModelDownloadRemoteDecoderPreservesTerminalState(t *testing.T) {
	test := func(name, kind, phase string, wantError bool) {
		t.Run(name, func(t *testing.T) {
			terminal := modelDownloadRemoteTestFrame(kind, "2", phase)
			terminal.Message = "fixture outcome"
			terminal.Result = json.RawMessage(`{"success":true}`)
			if phase == modeldownload.PhaseComplete {
				terminal.DownloadState.Entries[0].Persistence = modeldownload.PersistenceUncertain
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				encoder := json.NewEncoder(w)
				for _, frame := range []streamFrame{modelDownloadRemoteTestFrame("progress", "1", modeldownload.PhaseQueued), terminal} {
					if err := encoder.Encode(frame); err != nil {
						t.Errorf("encode stream frame: %v", err)
						return
					}
				}
			}))
			defer server.Close()
			client := &remoteClient{controlHTTP: server.Client(), base: server.URL}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			result, err := client.stream(ctx, controlPullPath, pullRequest{OpID: "request-op", Engine: "fake", Model: "model-a"}, nil)
			if (err != nil) != wantError || result.ID != terminal.ID || result.Model != terminal.Model || result.DownloadState == nil || !result.DownloadState.Equal(*terminal.DownloadState) {
				t.Fatalf("result=%+v error=%v, want matching %s terminal snapshot", result, err, phase)
			}
		})
	}
	test("completed with persistence qualification", "result", modeldownload.PhaseComplete, false)
	test("failed", "error", modeldownload.PhaseFailed, true)
	test("unknown", "error", modeldownload.PhaseUnknown, true)
}

func TestModelDownloadRemoteDecoderBoundsPullFramesBeforeDecode(t *testing.T) {
	frame := modelDownloadRemoteTestFrame("result", "1", modeldownload.PhaseComplete)
	frame.Result = json.RawMessage(`{"status":"success","padding":"` + strings.Repeat("x", maxPullStreamFrameBytes*2) + `"}`)
	oversized, err := json.Marshal(frame)
	if err != nil {
		t.Fatalf("encode oversized pull frame: %v", err)
	}
	test := func(name, payload string) {
		t.Run(name, func(t *testing.T) {
			reader := strings.NewReader(payload)
			client := &remoteClient{
				base: "http://pull-fixture.invalid",
				controlHTTP: &http.Client{Transport: memoryModelsTransport(func(*http.Request) (*http.Response, error) {
					return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(reader), Header: make(http.Header)}, nil
				})},
			}
			progress := 0
			result, err := client.stream(context.Background(), controlPullPath, pullRequest{OpID: "request-op", Engine: "fake", Model: "model-a"}, func(streamFrame) { progress++ })
			if err == nil || !strings.Contains(err.Error(), "pull frame exceeds") || result.Type != "" || progress != 0 {
				t.Fatalf("oversized pull result=%+v error=%v progress=%d", result, err, progress)
			}
			if consumed := len(payload) - reader.Len(); consumed > maxPullStreamFrameBytes+1 {
				t.Fatalf("read %d bytes before rejecting oversized frame, limit=%d", consumed, maxPullStreamFrameBytes+1)
			}
		})
	}
	test("oversized valid result", string(oversized)+"\n")
	test("oversized unterminated snapshot", `{"type":"progress","downloadState":{"entries":[`+strings.Repeat(" ", maxPullStreamFrameBytes*2))
}

func TestModelDownloadRemoteDecoderAcceptsExactPullFrameLimit(t *testing.T) {
	frame := modelDownloadRemoteTestFrame("result", "1", modeldownload.PhaseComplete)
	raw, err := json.Marshal(frame)
	if err != nil {
		t.Fatalf("encode bounded pull frame: %v", err)
	}
	if len(raw) > maxPullStreamFrameBytes {
		t.Fatal("fixture frame exceeds pull limit before padding")
	}
	payload := string(raw) + strings.Repeat(" ", maxPullStreamFrameBytes-len(raw)) + "\n"
	client := &remoteClient{
		base: "http://pull-fixture.invalid",
		controlHTTP: &http.Client{Transport: memoryModelsTransport(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(payload)), Header: make(http.Header)}, nil
		})},
	}
	result, err := client.stream(context.Background(), controlPullPath, pullRequest{OpID: "request-op", Engine: "fake", Model: "model-a"}, nil)
	if err != nil || result.Type != "result" || result.DownloadState == nil || !result.DownloadState.Equal(*frame.DownloadState) {
		t.Fatalf("exact-limit pull result=%+v error=%v", result, err)
	}
}

func TestModelDownloadRemoteDecoderRejectsOversizedNestedSnapshot(t *testing.T) {
	frame := modelDownloadRemoteTestFrame("result", "1", modeldownload.PhaseComplete)
	raw, err := json.Marshal(frame)
	if err != nil {
		t.Fatalf("encode pull frame: %v", err)
	}
	const prefix = `"downloadState":{`
	if strings.Count(string(raw), prefix) != 1 {
		t.Fatal("fixture is missing its unique download state object")
	}
	payload := strings.Replace(string(raw), prefix, prefix+strings.Repeat(" ", modeldownload.MaxSnapshotBytes), 1) + "\n"
	if len(payload) >= maxPullStreamFrameBytes {
		t.Fatal("oversized nested snapshot must fit within the outer frame limit")
	}
	client := &remoteClient{
		base: "http://pull-fixture.invalid",
		controlHTTP: &http.Client{Transport: memoryModelsTransport(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(payload)), Header: make(http.Header)}, nil
		})},
	}
	result, err := client.stream(context.Background(), controlPullPath, pullRequest{OpID: "request-op", Engine: "fake", Model: "model-a"}, nil)
	if err == nil || !strings.Contains(err.Error(), "byte limit") || result.Type != "" {
		t.Fatalf("oversized nested snapshot result=%+v error=%v", result, err)
	}
}

func TestModelDownloadPullFrameBoundDoesNotApplyToInstall(t *testing.T) {
	frame := streamFrame{Type: "result", Status: &EngineStatus{Installed: true}, Message: strings.Repeat("x", maxPullStreamFrameBytes+1)}
	raw, err := json.Marshal(frame)
	if err != nil {
		t.Fatalf("encode install frame: %v", err)
	}
	client := &remoteClient{
		base: "http://install-fixture.invalid",
		controlHTTP: &http.Client{Transport: memoryModelsTransport(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(bytes.NewReader(raw)), Header: make(http.Header)}, nil
		})},
	}
	result, err := client.stream(context.Background(), controlInstallPath, installRequest{Engine: "fake"}, nil)
	if err != nil || result.Status == nil || !result.Status.Installed || result.Message != frame.Message {
		t.Fatalf("install frame changed under pull-only bound: type=%s error=%v", result.Type, err)
	}
}

func TestModelDownloadInstallDecoderKeepsExistingContract(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		encoder := json.NewEncoder(w)
		for _, frame := range []streamFrame{{Type: "progress", Stage: "installing"}, {Type: "result", Status: &EngineStatus{Installed: true}}} {
			if err := encoder.Encode(frame); err != nil {
				t.Errorf("encode install frame: %v", err)
				return
			}
		}
	}))
	defer server.Close()
	client := &remoteClient{controlHTTP: server.Client(), base: server.URL}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	result, err := client.stream(ctx, controlInstallPath, installRequest{OpID: "install", Engine: "fake"}, nil)
	if err != nil || result.Status == nil || !result.Status.Installed {
		t.Fatalf("install result=%+v error=%v", result, err)
	}
}

type modelDownloadStreamWriter struct {
	recorder    *httptest.ResponseRecorder
	beforeWrite func([]byte) error
}

func (w *modelDownloadStreamWriter) Header() http.Header    { return w.recorder.Header() }
func (w *modelDownloadStreamWriter) WriteHeader(status int) { w.recorder.WriteHeader(status) }
func (w *modelDownloadStreamWriter) Write(data []byte) (int, error) {
	if w.beforeWrite != nil {
		if err := w.beforeWrite(data); err != nil {
			return 0, err
		}
	}
	return w.recorder.Write(data)
}
func (w *modelDownloadStreamWriter) Flush() { w.recorder.Flush() }

func TestModelDownloadRemotePullQueuesBeforeEffectsAndKeepsErrorIdentity(t *testing.T) {
	f := newLlamaCPPPullFixture(t)
	f.start = func(w http.ResponseWriter, _ *http.Request) {
		if _, err := fmt.Fprint(w, `{"success":false}`); err != nil {
			t.Errorf("write rejected pull: %v", err)
		}
	}
	writer := &modelDownloadStreamWriter{recorder: httptest.NewRecorder()}
	var queued streamFrame
	writer.beforeWrite = func(data []byte) error {
		if err := json.Unmarshal(data, &queued); err != nil {
			return err
		}
		select {
		case <-f.started:
			t.Error("target effect preceded initial queued frame")
		default:
		}
		writer.beforeWrite = nil
		return nil
	}
	request := httptest.NewRequest(http.MethodPost, controlPullPath, strings.NewReader(fmt.Sprintf(`{"opId":"request-op","engine":"fake","model":%q}`, llamaCPPPullTestModel)))
	(&controlServer{exec: f.ex}).handlePull(writer, request)
	frames := decodeFrames(t, writer.recorder.Body.String())
	if len(frames) < 2 || queued.Type != "progress" || queued.Stage != "queued" || queued.DownloadState == nil || len(queued.DownloadState.Entries) != 1 || queued.DownloadState.Entries[0].Phase != modeldownload.PhaseQueued {
		t.Fatalf("initial queued=%+v frames=%+v", queued, frames)
	}
	terminal := frames[len(frames)-1]
	if terminal.Type != "error" || terminal.OpID != "request-op" || terminal.ID != queued.ID || terminal.Model != llamaCPPPullTestModel || terminal.DownloadState == nil {
		t.Fatalf("terminal lost download identity: %+v", terminal)
	}
	validator := &pullStreamIdentity{request: pullRequest{OpID: "request-op", Engine: "fake", Model: llamaCPPPullTestModel}}
	for _, frame := range frames {
		if err := validator.validate(frame); err != nil {
			t.Fatalf("invalid emitted frame %+v: %v", frame, err)
		}
	}
}

func TestModelDownloadRemotePullSetupFailureReleasesReservation(t *testing.T) {
	for _, test := range []struct {
		name   string
		writer http.ResponseWriter
	}{
		{"initial write fails", &modelDownloadStreamWriter{recorder: httptest.NewRecorder(), beforeWrite: func([]byte) error { return errors.New("fixture stream disconnected") }}},
		{"streaming unsupported", &struct{ http.ResponseWriter }{httptest.NewRecorder()}},
	} {
		t.Run(test.name, func(t *testing.T) {
			f := newLlamaCPPPullFixture(t)
			request := httptest.NewRequest(http.MethodPost, controlPullPath, strings.NewReader(fmt.Sprintf(`{"opId":"request-op","engine":"fake","model":%q}`, llamaCPPPullTestModel)))
			(&controlServer{exec: f.ex}).handlePull(test.writer, request)
			select {
			case <-f.started:
				t.Fatal("failed stream setup dispatched a target effect")
			default:
			}
			f.ex.downloads.mu.Lock()
			active, unsafe := len(f.ex.downloads.active), len(f.ex.downloads.unsafe)
			f.ex.downloads.mu.Unlock()
			if active != 0 || unsafe != 0 {
				t.Fatalf("active=%d unsafe=%d, want unstarted reservation released", active, unsafe)
			}
			token, err := f.ex.reserveModelPull(context.Background(), "fake", llamaCPPPullTestModel, nil)
			if err != nil {
				t.Fatalf("reserve after failed setup: %v", err)
			}
			token.cancelUnstarted()
		})
	}
}

func TestModelDownloadRemoteProgressCarriesTargetSnapshot(t *testing.T) {
	var output bytes.Buffer
	manager := &Manager{codec: NewCodec(&output)}
	frame := modelDownloadRemoteTestFrame("progress", "7", modeldownload.PhaseInProgress)
	frame.Stage, frame.Percent = "downloading", 37
	manager.remoteProgressFn("request-op", "target-node")(frame)
	var notification struct {
		Method string         `json:"method"`
		Params remoteProgress `json:"params"`
	}
	if err := json.Unmarshal(output.Bytes(), &notification); err != nil {
		t.Fatalf("decode remote progress: %v", err)
	}
	params := notification.Params
	if notification.Method != "engine:remote-progress" || params.Node != "target-node" || params.OpID != "request-op" || params.ID != frame.ID || params.Engine != frame.Engine || params.Op != "pull" || params.Model != frame.Model || params.Stage != frame.Stage || params.Percent != frame.Percent || params.DownloadState == nil || !params.DownloadState.Equal(*frame.DownloadState) {
		t.Fatalf("remote notification=%+v, want request correlation and target-owned snapshot", notification)
	}
}
