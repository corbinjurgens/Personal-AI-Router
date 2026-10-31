// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

type settingsHarness struct {
	b                   *Broker
	applies             atomic.Int32
	proxyRebinds        atomic.Int32
	fail                atomic.Bool
	failBeforeStop      atomic.Bool
	loseProxyOnStop     atomic.Bool
	failResultSave      atomic.Bool
	otherEnginePort     atomic.Int32
	previewPreserveCORS atomic.Bool
	entered, release    chan struct{}
	// launchMu guards the fixture's launch state, which a suspended
	// engine:configure-launch mutates while the reader loop keeps serving.
	launchMu        sync.Mutex
	launchStatus    string
	configureStatus string
	// proxyCalls records the unaddressed requests the fake proxy answered,
	// such as node/set-engine-drain, in arrival order.
	proxyCallsMu sync.Mutex
	proxyCalls   []policyCall
	// settingsReplies receives the engine:settings-reply notifications the
	// broker sends the fake engine-manager.
	settingsReplies chan json.RawMessage
	// reservedPorts belong to the fake workers until test cleanup. Only these
	// reservations may pass the injected check; foreign ports use real preflight.
	reservedPorts sync.Map
}

func (h *settingsHarness) reservePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve settings fixture port: %v", err)
	}
	t.Cleanup(func() {
		if err := ln.Close(); err != nil {
			t.Errorf("close settings fixture port: %v", err)
		}
	})
	port := ln.Addr().(*net.TCPAddr).Port
	h.reservedPorts.Store(port, struct{}{})
	return port
}

func (h *settingsHarness) unaddressedProxyCalls() []policyCall {
	h.proxyCallsMu.Lock()
	defer h.proxyCallsMu.Unlock()
	return append([]policyCall(nil), h.proxyCalls...)
}

func newSettingsHarness(t *testing.T) *settingsHarness {
	t.Helper()
	return newSettingsHarnessForEngine(t, "ollama")
}

func newSettingsHarnessForEngine(t *testing.T, engine string) *settingsHarness {
	t.Helper()
	return newServingSettingsHarness(t, engine, nil)
}

func newServingSettingsHarness(t *testing.T, engine string, serving *engines.ServingConfig) *settingsHarness {
	t.Helper()
	h := &settingsHarness{b: &Broker{codec: NewCodec(&bytes.Buffer{}), nodeID: "target", clusterDir: filepath.Join(t.TempDir(), "cluster")}, settingsReplies: make(chan json.RawMessage, 8)}
	ports := make([]int, 4)
	for i := range ports {
		ports[i] = h.reservePort(t)
	}
	// The workers communicate over pipes and never bind these ports themselves.
	// Keep their reservations across failed starts and proxy moves, so another
	// process cannot turn an unrelated retry into a port-conflict failure.
	h.b.portAvailableCheck = func(port int) bool {
		if _, reserved := h.reservedPorts.Load(port); reserved {
			return true
		}
		return tcpPortAvailable(port)
	}
	worker, codec := newTestRPCWorkerPipe(t)
	h.b.setEngineMgr(worker)
	proxyWorker, proxyCodec := newTestRPCWorkerPipe(t)
	proxy := &proxyProcess{peer: proxyWorker.peer, facadeState: map[string]proxyFacadeState{
		"ollama":   {ready: true, port: ports[1]},
		"lmstudio": {ready: true, port: ports[2]},
		"llamacpp": {ready: true, port: ports[3]},
	}}
	publishTestProxy(h.b, proxy)
	go func() {
		for {
			msg, err := proxyCodec.Read()
			if err != nil {
				return
			}
			if !msg.IsRequest() {
				continue
			}
			engine, method := engines.SplitAddressedMethod(msg.Method)
			if engine == "" {
				h.proxyCallsMu.Lock()
				h.proxyCalls = append(h.proxyCalls, policyCall{method: msg.Method, params: append(json.RawMessage(nil), msg.Params...)})
				h.proxyCallsMu.Unlock()
			}
			if engine == "" || method != "set-port" {
				_ = proxyCodec.Respond(msg.ID, map[string]bool{"ok": true})
				continue
			}
			var p struct {
				Port int `json:"port"`
			}
			if err := json.Unmarshal(msg.Params, &p); err != nil {
				t.Errorf("decode proxy port request: %v", err)
				return
			}
			h.proxyRebinds.Add(1)
			proxy.readyMu.Lock()
			proxy.facadeState[engine] = proxyFacadeState{ready: true, port: p.Port}
			proxy.readyMu.Unlock()
			_ = proxyCodec.Respond(msg.ID, map[string]int{"port": p.Port})
		}
	}()
	launch := settings.LaunchState{Engine: engine, ServerPort: ports[0], EffectivePort: ports[0], LaunchText: "--fixture-option", Running: true, Editable: true, Format: "pair-arguments-v1", Serving: serving.Clone()}
	go func() {
		for {
			msg, err := codec.Read()
			if err != nil {
				return
			}
			if !msg.IsRequest() {
				if msg.Method == "engine:settings-reply" {
					select {
					case h.settingsReplies <- msg.Params:
					default:
						t.Error("too many settings replies")
					}
				}
				continue
			}
			switch msg.Method {
			case "engine:get-launch":
				h.launchMu.Lock()
				current := launch
				current.WorkingSetLimitStatus = h.launchStatus
				h.launchMu.Unlock()
				_ = codec.Respond(msg.ID, current)
			case "engine:configured-ports":
				_ = codec.Respond(msg.ID, map[string]any{"engines": []any{map[string]any{"engine": "lmstudio", "port": h.otherEnginePort.Load()}}})
			case "engine:preview-launch":
				envelope, _ := settings.DecodeWorkerRequest[settings.Request](msg.Params)
				p := envelope.Request
				h.previewPreserveCORS.Store(p.PreserveCORS)
				if p.Format == "pair-launch-v1" {
					p.Settings.LaunchText = strings.TrimPrefix(p.Settings.LaunchText, "managed serve ")
				}
				h.launchMu.Lock()
				restart := launch.Running && (p.Settings.ServerPort != launch.ServerPort || p.Settings.LaunchText != launch.LaunchText)
				h.launchMu.Unlock()
				_ = codec.Respond(msg.ID, settings.Preview{Settings: p.Settings, Restart: restart})
			case "engine:configure-launch":
				// Serve this off the reader loop: a suspended apply must not
				// stop the fixture from answering an unrelated read, which is
				// exactly what the broker's lock split makes possible.
				go func(msg *Message) {
					h.applies.Add(1)
					if h.entered != nil {
						h.entered <- struct{}{}
						<-h.release
					}
					if h.failBeforeStop.Load() {
						if h.loseProxyOnStop.Load() {
							proxy.readyMu.Lock()
							state := proxy.facadeState[engine]
							state.ready = false
							proxy.facadeState[engine] = state
							proxy.readyMu.Unlock()
						}
						_ = codec.RespondError(msg.ID, -32000, "stop failure")
						return
					}
					var p settings.Configure
					_ = json.Unmarshal(msg.Params, &p)
					h.launchMu.Lock()
					launch.ServerPort = p.Settings.ServerPort
					launch.WorkingSetLimitMiB = p.Settings.Clone().WorkingSetLimitMiB
					if h.configureStatus != "" {
						h.launchStatus = h.configureStatus
					}
					launch.WorkingSetLimitStatus = h.launchStatus
					launch.LaunchText = p.Settings.LaunchText
					launch.Serving = p.Settings.Serving.Clone()
					launch.Running = false
					h.launchMu.Unlock()
					if err := h.b.rebindSettingsProxy(p.Engine, p.Settings.ProxyPort); err != nil {
						_ = codec.RespondError(msg.ID, -32000, err.Error())
						return
					}
					if h.fail.Load() {
						_ = codec.RespondError(msg.ID, -32000, "startup failure")
						return
					}
					h.launchMu.Lock()
					launch.EffectivePort = p.Settings.ServerPort
					launch.Running = p.Resume
					current := launch
					h.launchMu.Unlock()
					if h.failResultSave.Load() {
						journal, _ := h.b.engineSettingsPath()
						if err := os.Remove(journal); err != nil {
							t.Error(err)
						}
						if err := os.Mkdir(journal, 0700); err != nil {
							t.Error(err)
						}
					}
					_ = codec.Respond(msg.ID, current)
				}(msg)
			default:
				_ = codec.RespondError(msg.ID, -32601, "unsupported fixture method")
			}
		}
	}()
	return h
}

func (h *settingsHarness) request(t *testing.T) settings.Request {
	t.Helper()
	s, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "ollama"}, "")
	if err != nil {
		t.Fatal(err)
	}
	return settings.Request{Engine: "ollama", ExpectedRevision: s.Revision, RequestID: settingsID(), Settings: s.Settings}
}

func brokerServingFixture() *engines.ServingConfig {
	ram, vram := int64(4096), int64(0)
	return &engines.ServingConfig{Version: 1, Models: []engines.ServingModel{{ModelID: "exact/model", MaxOutputTokens: 128, Memory: engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram}, Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: strings.Repeat("a", 64), Provenance: engines.ServingProvenance{Kind: "hf-lfs-sha256"}, Binding: &engines.ServingBinding{SourcePath: "fixture-link", ResolvedTarget: "fixture-blob", SizeBytes: 64, MtimeNS: "1729000000000000001", SampleSHA256: strings.Repeat("b", 64)}}}}}}
}

func TestWorkingSetLimitStatusReachesPublishedSettings(t *testing.T) {
	h := newSettingsHarnessForEngine(t, "llamacpp")
	var published bytes.Buffer
	h.b.codec = NewCodec(&published)
	h.launchMu.Lock()
	h.launchStatus = "unverified"
	h.launchMu.Unlock()
	unverified, unverifiedErr := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
	if unverifiedErr != nil || unverified.WorkingSetLimitStatus != "unverified" {
		t.Fatalf("unverified snapshot = %+v, error = %v", unverified, unverifiedErr)
	}
	h.launchMu.Lock()
	h.launchStatus = "unenforced-adopted"
	h.launchMu.Unlock()
	snapshot, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil || snapshot.WorkingSetLimitStatus != "unenforced-adopted" {
		t.Fatalf("refresh snapshot = %+v, error = %v", snapshot, err)
	}
	assertPublished := func(want string) {
		t.Helper()
		decoder := json.NewDecoder(bytes.NewReader(published.Bytes()))
		last := ""
		for {
			var message Message
			if err := decoder.Decode(&message); err != nil {
				if err == io.EOF {
					break
				}
				t.Fatal(err)
			}
			if message.Method != "engine:settings-changed" {
				continue
			}
			var payload settings.Snapshot
			if err := json.Unmarshal(message.Params, &payload); err != nil {
				t.Fatal(err)
			}
			if payload.Engine == "llamacpp" {
				last = payload.WorkingSetLimitStatus
			}
		}
		if last != want {
			t.Fatalf("published status = %q, want %q", last, want)
		}
		published.Reset()
	}
	assertPublished("unenforced-adopted")
	h.launchMu.Lock()
	h.configureStatus = "applied"
	h.launchMu.Unlock()
	limit := 256
	request := settings.Request{Engine: "llamacpp", ExpectedRevision: snapshot.Revision, RequestID: settingsID(), Settings: snapshot.Settings.Clone()}
	request.Settings.WorkingSetLimitMiB = &limit
	receipt, err := h.b.applyEngineSettings(context.Background(), request, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("apply receipt = %+v, error = %v", receipt, err)
	}
	// Inspect completion publication before another refresh could mask it.
	assertPublished("applied")
	if got := h.b.engineSettings["llamacpp"].Snapshot; got.WorkingSetLimitStatus != "applied" || got.Settings.WorkingSetLimitMiB == nil || *got.Settings.WorkingSetLimitMiB != 256 {
		t.Fatalf("completion snapshot = %+v", got)
	}
}

func TestSettingsServingReceiptRecoveryAndSwitchPublication(t *testing.T) {
	serving := brokerServingFixture()
	h := newServingSettingsHarness(t, "ollama", serving)
	p := h.request(t)
	if !p.Settings.Serving.Equal(serving) {
		t.Fatal("baseline reconstruction lost serving")
	}
	p.Settings.LaunchText += " --unrelated"
	preview, err := h.b.previewEngineSettings(context.Background(), p, "")
	if err != nil || !preview.Settings.Serving.Equal(serving) {
		t.Fatal("preview lost serving")
	}
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("apply receipt %v: %v", receipt.Phase, err)
	}
	if !h.b.engineSettings["ollama"].Previous.Serving.Equal(serving) {
		t.Fatal("journal previous lost serving")
	}
	h.failBeforeStop.Store(true)
	p = h.request(t)
	p.Settings.LaunchText += " --failed-edit"
	receipt, err = h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "failed" {
		t.Fatal("expected failed receipt")
	}
	h.failBeforeStop.Store(false)
	// Crash after durable acceptance of the next retry, before worker apply.
	record := h.b.engineSettings["ollama"]
	record.Snapshot.Phase = "applying"
	record.Snapshot.RequestID = "interrupted-retry"
	record.Receipts["interrupted-retry"] = settingsReceipt{Revision: record.Snapshot.Revision, Phase: "applying"}
	if err := h.b.saveEngineSettingsLocked(); err != nil {
		t.Fatal(err)
	}
	beforeRecovery := h.applies.Load()
	h.b.engineSettingsLoaded = false
	if !h.b.recoverEngineSettings() {
		t.Fatal("recovery failed")
	}
	if h.applies.Load() != beforeRecovery+1 {
		t.Fatal("interrupted recovery did not replay the accepted settings")
	}
	config, ok := h.b.explicitEngineSettings("ollama")
	if !ok || !config.Serving.Equal(serving) {
		t.Fatal("recovered config lost serving")
	}
	*config.Serving.Models[0].Memory.HostRAMBytes = 1
	again, ok := h.b.explicitEngineSettings("ollama")
	if !ok || !again.Serving.Equal(serving) {
		t.Fatal("recovery config aliases journal")
	}
	if err := h.b.publishSwitchTarget(context.Background(), "ollama"); err != nil {
		t.Fatal(err)
	}
	snapshot, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "ollama"}, "")
	if err != nil || !snapshot.Settings.Serving.Equal(serving) {
		t.Fatal("switch publication/readback lost serving")
	}
	h.b.engineConfigMu.Lock()
	beforeRefresh := h.b.engineSettings["ollama"].Snapshot.Sequence
	h.b.reconcileLegacySettingsLocked()
	afterRefresh := h.b.engineSettings["ollama"].Snapshot.Sequence
	h.b.engineConfigMu.Unlock()
	if beforeRefresh != afterRefresh {
		t.Fatal("equivalent serving republished an unchanged refresh")
	}
	*snapshot.Settings.Serving.Models[0].Memory.VRAMBytes = 2
	if !h.b.engineSettings["ollama"].Snapshot.Settings.Serving.Equal(serving) {
		t.Fatal("readback aliases journal")
	}
	duplicate, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || duplicate != receipt {
		t.Fatal("recovery changed immutable failed receipt")
	}
}

func TestSettingsServingJournalWriteFailureRestoresDeclaration(t *testing.T) {
	serving := brokerServingFixture()
	h := newServingSettingsHarness(t, "ollama", serving)
	p := h.request(t)
	p.Settings.Serving.Models[0].MaxOutputTokens++
	path, err := h.b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(path, 0700); err != nil {
		t.Fatal(err)
	}
	if _, err := h.b.applyEngineSettings(context.Background(), p, ""); err == nil {
		t.Fatal("journal write succeeded unexpectedly")
	}
	if h.applies.Load() != 0 || !h.b.engineSettings["ollama"].Snapshot.Settings.Serving.Equal(serving) {
		t.Fatal("failed journal write changed declaration/runtime")
	}
}

func TestSettingsServingLegacyFingerprintByteStability(t *testing.T) {
	s := settings.Snapshot{Epoch: "fixed-epoch", Revision: 7, AppliedRevision: 7, Phase: "succeeded", Settings: settings.Config{ServerPort: 8080, ProxyPort: 8081, LaunchText: "--opaque"}}
	legacy := sha256.Sum256([]byte(`{"version":1,"epoch":"fixed-epoch","appliedRevision":7,"settings":{"serverPort":8080,"proxyPort":8081,"launchText":"--opaque"}}`))
	fingerprint, ok := settingsFingerprint(s)
	if !ok || fingerprint != hex.EncodeToString(legacy[:]) {
		t.Fatal("unset serving changed pre-NE fingerprint bytes")
	}
	s.Settings.Serving = brokerServingFixture()
	withServing, ok := settingsFingerprint(s)
	if !ok || withServing == fingerprint {
		t.Fatal("serving omitted from applied identity")
	}
}

func TestSettingsCoordinatorRevisionDedupNoopAndFailure(t *testing.T) {
	h := newSettingsHarness(t)
	p := h.request(t)
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil {
		t.Fatal(err)
	}
	if h.applies.Load() != 0 || receipt.Phase != "succeeded" {
		t.Fatal("no-op mutated runtime")
	}
	p = h.request(t)
	p.Settings.LaunchText += " --parallel 2"
	receipt, err = h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Revision != p.ExpectedRevision+1 || receipt.Phase != "succeeded" || h.applies.Load() != 1 {
		t.Fatalf("receipt=%+v applies=%d", receipt, h.applies.Load())
	}
	if _, err = h.b.applyEngineSettings(context.Background(), p, ""); err != nil || h.applies.Load() != 1 {
		t.Fatal("duplicate restarted")
	}
	reused := p
	reused.Settings.LaunchText += " --different"
	if _, err = h.b.applyEngineSettings(context.Background(), reused, ""); err == nil {
		t.Fatal("reused identifier accepted")
	}
	p.RequestID = settingsID()
	if _, err = h.b.applyEngineSettings(context.Background(), p, ""); err == nil {
		t.Fatal("stale revision accepted")
	}
	p = h.request(t)
	p.Settings.LaunchText += " --invalid"
	h.fail.Store(true)
	receipt, err = h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "failed" {
		t.Fatalf("failure receipt %+v %v", receipt, err)
	}
	s := h.b.engineSettings["ollama"].Snapshot
	if !s.Settings.Equal(p.Settings) || s.Running || s.AppliedRevision >= s.Revision || s.Error == "" {
		t.Fatalf("untruthful failed snapshot: %+v", s)
	}
	h.fail.Store(false)
	p = h.request(t)
	if _, err = h.b.applyEngineSettings(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	if s := h.b.engineSettings["ollama"].Snapshot; !s.Running || s.Phase != "succeeded" {
		t.Fatalf("retry lost resume intent: %+v", s)
	}
}

func TestSettingsStoppedRetryPreservesPortReservation(t *testing.T) {
	h := newSettingsHarness(t)
	p := h.request(t)
	p.Settings.LaunchText += " --failed-start"
	h.fail.Store(true)
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "failed" {
		t.Fatalf("startup failure receipt = %+v, %v", receipt, err)
	}
	h.fail.Store(false)
	p = h.request(t)
	before := h.b.engineSettings["ollama"].Snapshot
	if before.Running {
		t.Fatal("failed startup left the engine running")
	}
	// A competing process cannot take the stopped fake engine's port before
	// the retry, even though its worker never binds a real engine listener.
	competitor, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", p.Settings.ServerPort))
	if err == nil {
		if err := competitor.Close(); err != nil {
			t.Fatalf("close competing listener: %v", err)
		}
		t.Fatal("stopped fixture released its server-port reservation")
	}
	if tcpPortAvailable(p.Settings.ServerPort) {
		t.Fatal("production preflight accepted the fixture's occupied loopback port")
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("bind foreign occupied port: %v", err)
	}
	t.Cleanup(func() {
		if err := ln.Close(); err != nil {
			t.Errorf("close foreign occupied port: %v", err)
		}
	})
	conflict := p
	conflict.RequestID = settingsID()
	conflict.Settings.ServerPort = ln.Addr().(*net.TCPAddr).Port
	preview, err := h.b.previewEngineSettings(context.Background(), conflict, "")
	if err != nil {
		t.Fatalf("preview foreign occupied port: %v", err)
	}
	if want := fmt.Sprintf("port %d is already in use", conflict.Settings.ServerPort); preview.Errors["ports"] != want {
		t.Fatalf("port errors = %v, want %q", preview.Errors, want)
	}
	if _, err := h.b.applyEngineSettings(context.Background(), conflict, ""); err == nil {
		t.Fatal("retry accepted a foreign occupied port")
	}
	if after := h.b.engineSettings["ollama"].Snapshot; after.Revision != before.Revision || !after.Settings.Equal(before.Settings) || h.applies.Load() != 1 {
		t.Fatalf("rejected conflict changed settings or runtime: %+v, applies=%d", after, h.applies.Load())
	}
	receipt, err = h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "succeeded" || h.applies.Load() != 2 {
		t.Fatalf("retry on reserved fixture port = %+v, %v, applies=%d", receipt, err, h.applies.Load())
	}
	if after := h.b.engineSettings["ollama"].Snapshot; !after.Running {
		t.Fatal("retry lost resume intent")
	}
}

func TestSettingsFailedApplyRestoresOnlyRunningReadyService(t *testing.T) {
	for _, tc := range []struct {
		name       string
		stopFails  bool
		proxyReady bool
		advertised bool
	}{
		{"stop failed with engine still running", true, true, true},
		{"engine stopped before startup failed", false, true, false},
		{"proxy unavailable", true, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newSettingsHarness(t)
			p := h.request(t)
			oldPort := p.Settings.ProxyPort
			h.b.regCache = relay.NewRegistrationCache()
			h.b.registerService(noderec.RegisterParams{Service: noderec.ServiceOllama, Port: oldPort})
			h.failBeforeStop.Store(tc.stopFails)
			h.fail.Store(true)
			h.loseProxyOnStop.Store(!tc.proxyReady)
			p.Settings.LaunchText += " --new-option"
			receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
			if err != nil || receipt.Phase != "failed" {
				t.Fatalf("expected failed receipt, got %+v, %v", receipt, err)
			}
			registered := h.b.regCache.Snapshot()
			if tc.advertised {
				if len(registered) != 1 || registered[0].Service != noderec.ServiceOllama || registered[0].Port != oldPort {
					t.Fatalf("running engine lost registration: %+v", registered)
				}
			} else if len(registered) != 0 {
				t.Fatalf("unavailable service advertised: %+v", registered)
			}
		})
	}
}

func TestSettingsProxyRebindPushesLivePortToClusterManager(t *testing.T) {
	for _, tc := range []struct {
		engine  string
		service noderec.ServiceKey
	}{
		{engine: "ollama", service: noderec.ServiceOllama},
		{engine: "lmstudio", service: noderec.ServiceLMStudio},
		{engine: "llamacpp", service: noderec.ServiceLlamaCPP},
	} {
		t.Run(tc.engine, func(t *testing.T) {
			h := newSettingsHarnessForEngine(t, tc.engine)
			current, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: tc.engine}, "")
			if err != nil {
				t.Fatalf("get engine settings: %v", err)
			}
			p := settings.Request{Engine: tc.engine, ExpectedRevision: current.Revision,
				RequestID: settingsID(), Settings: current.Settings}
			h.b.regCache = relay.NewRegistrationCache()
			seen := observeLocalServices(t, h.b)
			awaitLocalServices(t, seen, 0)
			h.b.registerService(noderec.RegisterParams{Service: tc.service, Port: p.Settings.ProxyPort})
			awaitLocalServices(t, seen, 1)
			p.Settings.ProxyPort = h.reservePort(t)
			p.Settings.LaunchText += " --rebind-fixture"
			receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
			if err != nil || receipt.Phase != "succeeded" {
				t.Fatalf("proxy port apply = %+v, %v", receipt, err)
			}
			got := awaitLocalServices(t, seen, 3)
			if len(got.Services) != 1 || got.Services[0].Service != tc.service || got.Services[0].Port != p.Settings.ProxyPort {
				t.Fatalf("rebound snapshot = %+v, want %s proxy on %d", got, tc.service, p.Settings.ProxyPort)
			}
			if got.Services[0].Port == p.Settings.ServerPort || h.proxyRebinds.Load() != 1 {
				t.Fatalf("snapshot used raw engine port or missed live rebind: %+v; rebinds=%d", got, h.proxyRebinds.Load())
			}
		})
	}
}

func TestSettingsCoordinatorSerializesAndCancelsQueuedRequests(t *testing.T) {
	h := newSettingsHarness(t)
	p := h.request(t)
	p.Settings.LaunchText += " --first"
	h.entered = make(chan struct{}, 1)
	h.release = make(chan struct{})
	done := make(chan error, 1)
	go func() { _, err := h.b.applyEngineSettings(context.Background(), p, ""); done <- err }()
	<-h.entered
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	other := p
	other.RequestID = settingsID()
	canceled := make(chan error, 1)
	go func() { _, err := h.b.applyEngineSettings(ctx, other, ""); canceled <- err }()
	close(h.release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if err := <-canceled; err == nil {
		t.Fatal("canceled queued mutation accepted")
	}
	if h.applies.Load() != 1 {
		t.Fatal("concurrent request mutated runtime")
	}
}

// An apply stops and restarts an engine, which can run for minutes. Holding the
// journal lock across that froze the other engine's editor and the refresh
// poller for the whole restart.
func TestSettingsApplyDoesNotBlockUnrelatedReads(t *testing.T) {
	h := newSettingsHarness(t)
	p := h.request(t)
	p.Settings.LaunchText += " --busy"
	h.entered = make(chan struct{}, 1)
	h.release = make(chan struct{})
	done := make(chan error, 1)
	go func() { _, err := h.b.applyEngineSettings(context.Background(), p, ""); done <- err }()
	<-h.entered

	read := make(chan error, 1)
	go func() {
		_, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "lmstudio"}, "")
		read <- err
	}()
	select {
	case err := <-read:
		if err != nil {
			t.Fatalf("read during an in-flight apply failed: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("engine settings read blocked behind an in-flight apply")
	}

	close(h.release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestSettingsResultPersistenceFailureReturnsFailedReceipt(t *testing.T) {
	h := newSettingsHarness(t)
	p := h.request(t)
	p.Settings.LaunchText += " --new"
	h.failResultSave.Store(true)
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "failed" || h.b.engineSettings["ollama"].Snapshot.Phase != "failed" {
		t.Fatalf("result persistence failure claimed success: %+v %v", receipt, err)
	}
}

func TestSettingsJournalLoadRebasesOwnerWithoutChangingRecoveryDataOrDisk(t *testing.T) {
	b := &Broker{nodeID: "current-node", clusterDir: filepath.Join(t.TempDir(), "cluster")}
	path, err := b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	journal := []byte(`{
		"ollama": {"snapshot": {"nodeId":"former-node","engine":"ollama","revision":9,"appliedRevision":7,"epoch":"former-epoch","sequence":42,"settings":{"serverPort":11436,"proxyPort":11434,"launchText":"--parallel 2"},"effectiveServerPort":11435,"effectiveProxyPort":11434,"running":true,"adopted":true,"editable":true,"reason":"fixture reason","format":"pair-arguments-v1","phase":"applying","error":"interrupted","requestId":"active-request"},"previous":{"serverPort":11435,"proxyPort":11434,"launchText":"--parallel 1"},"resume":true,"explicit":true,"receipts":{"active-request":{"hash":"active-hash","revision":9,"phase":"applying"},"earlier-request":{"hash":"earlier-hash","revision":7,"phase":"succeeded"}}},
		"lmstudio": {"snapshot": {"engine":"lmstudio","revision":5,"appliedRevision":3,"epoch":"old-epoch","sequence":18,"settings":{"serverPort":1235,"proxyPort":1234,"launchText":"--fixture"},"phase":"failed","error":"startup failure","requestId":"failed-request"},"previous":{"serverPort":1236,"proxyPort":1234},"resume":false,"explicit":false,"receipts":{"failed-request":{"hash":"failed-hash","revision":5,"phase":"failed"}}}
	}`)
	var want map[string]*engineSettingsRecord
	if err := json.Unmarshal(journal, &want); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, journal, 0600); err != nil {
		t.Fatal(err)
	}
	if err := b.loadEngineSettingsLocked(); err != nil {
		t.Fatal(err)
	}
	if b.engineSettingsEpoch == "" || b.engineSettingsEpoch == "former-epoch" || b.engineSettingsEpoch == "old-epoch" {
		t.Fatalf("epoch was not renewed: %q", b.engineSettingsEpoch)
	}
	for _, record := range want {
		record.Snapshot.NodeID = "current-node"
		record.Snapshot.Epoch = b.engineSettingsEpoch
		record.Snapshot.Sequence = 0
	}
	if !reflect.DeepEqual(b.engineSettings, want) {
		t.Fatalf("loaded records differ from preserved recovery data: got %+v, want %+v", b.engineSettings, want)
	}
	loadedBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(loadedBytes, journal) {
		t.Fatal("loading rewrote the journal")
	}
	if err := b.saveEngineSettingsLocked(); err != nil {
		t.Fatal(err)
	}
	savedBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var saved map[string]*engineSettingsRecord
	if err := json.Unmarshal(savedBytes, &saved); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(saved, want) {
		t.Fatalf("normal save did not preserve rebased records: got %+v, want %+v", saved, want)
	}
}

func TestSettingsJournalFailureAndInterruptedRecovery(t *testing.T) {
	h := newSettingsHarness(t)
	p := h.request(t)
	p.Settings.LaunchText += " --new"
	path, _ := h.b.engineSettingsPath()
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(path, 0700); err != nil {
		t.Fatal(err)
	}
	if _, err := h.b.applyEngineSettings(context.Background(), p, ""); err == nil || h.applies.Load() != 0 {
		t.Fatal("runtime changed despite failed journal write")
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	r := h.b.engineSettings["ollama"]
	r.Snapshot.Settings = p.Settings
	r.Snapshot.Revision++
	r.Snapshot.Phase = "applying"
	r.Snapshot.NodeID = "former-node"
	r.Resume = true
	r.Explicit = true
	if err := h.b.saveEngineSettingsLocked(); err != nil {
		t.Fatal(err)
	}
	// Simulate a fresh coordinator after its journal was written but component
	// writes/rebind/start had not completed. The same replay handles later cuts.
	h.b.engineSettingsLoaded = false
	if !h.b.recoverEngineSettings() {
		t.Fatal("recovery refused")
	}
	if h.applies.Load() != 1 || h.b.engineSettings["ollama"].Snapshot.Phase != "succeeded" {
		t.Fatal("accepted operation not recovered")
	}
	if !h.b.recoverEngineSettings() || h.applies.Load() != 1 {
		t.Fatal("completed operation replayed")
	}
	if _, ok := h.b.explicitEngineSettings("ollama"); !ok {
		t.Fatal("explicit startup preference lost")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var saved map[string]*engineSettingsRecord
	if err := json.Unmarshal(data, &saved); err != nil {
		t.Fatal(err)
	}
	if saved["ollama"] == nil || saved["ollama"].Snapshot.NodeID != "target" {
		t.Fatal("recovery did not persist the current settings owner")
	}
}

func TestSettingsPortValidationIncludesStoppedEnginesAndAliases(t *testing.T) {
	h := newSettingsHarness(t)
	p := h.request(t)
	p.Settings.ProxyPort = p.Settings.ServerPort
	preview, err := h.b.previewEngineSettings(context.Background(), p, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(preview.Errors) == 0 {
		t.Fatal("equal ports accepted")
	}
	p = h.request(t)
	p.Settings.ServerPort = engineControlPort
	preview, err = h.b.previewEngineSettings(context.Background(), p, "")
	if err != nil || len(preview.Errors) == 0 {
		t.Fatalf("reserved control port accepted: %v %v", preview, err)
	}
	p = h.request(t)
	h.otherEnginePort.Store(int32(p.Settings.ServerPort))
	preview, err = h.b.previewEngineSettings(context.Background(), p, "")
	if err != nil || len(preview.Errors) == 0 {
		t.Fatalf("stopped engine reservation ignored: %v %v", preview, err)
	}
	h.otherEnginePort.Store(0)
	h.b.ollamaHostAliasMu.Lock()
	h.b.ollamaHostAlias.Port = p.Settings.ProxyPort
	h.b.ollamaHostAliasMu.Unlock()
	preview, err = h.b.previewEngineSettings(context.Background(), p, "")
	if err != nil || len(preview.Errors) == 0 {
		t.Fatalf("proxy alias reservation ignored: %v %v", preview, err)
	}
}

func TestSettingsMigratesLegacyProxyChoiceBeforeManagedDefaults(t *testing.T) {
	h := newSettingsHarness(t)
	path, _ := h.b.engineSettingsPath()
	if err := os.WriteFile(filepath.Join(filepath.Dir(path), "proxy-port.json"), []byte(`{"port":26080}`), 0600); err != nil {
		t.Fatal(err)
	}
	h.b.migrateLegacyEngineSettings()
	config, ok := h.b.explicitEngineSettings("ollama")
	if !ok || config.ProxyPort != 26080 {
		t.Fatalf("legacy choice lost: %+v %v", config, ok)
	}
	if !h.b.prepareExplicitEngineSettings("ollama") || h.b.ollamaState().startupPort.Load() != 26080 || h.b.ollamaState().managedFacade.Load() {
		t.Fatal("automatic startup overrode saved proxy choice")
	}
}
