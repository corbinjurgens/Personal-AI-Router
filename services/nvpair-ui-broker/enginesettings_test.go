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
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

type settingsHarness struct {
	b                   *Broker
	engine              string
	nativeDigest        atomic.Pointer[string]
	typedDigest         atomic.Bool
	applies             atomic.Int32
	proxyRebinds        atomic.Int32
	engineStarts        atomic.Int32
	bindRace            atomic.Bool
	facadeBindRace      atomic.Bool
	facadeUnanswered    atomic.Bool
	serializeLaunch     atomic.Bool
	configureMu         sync.Mutex
	launchReadWaiting   chan struct{}
	fail                atomic.Bool
	failBeforeStop      atomic.Bool
	loseProxyOnStop     atomic.Bool
	dropProxyOnStop     atomic.Bool
	failResultSave      atomic.Bool
	otherEnginePort     atomic.Int32
	previewPreserveCORS atomic.Bool
	entered, release    chan struct{}
	// launchMu guards the fixture's launch state, which a suspended
	// engine:configure-launch mutates while the reader loop keeps serving.
	launchMu        sync.Mutex
	launchStatus    string
	configureStatus string
	launchReadback  *engines.ServingReadback
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

func TestPrepareSettingsSnapshotRejectsUnidentifiedOrInvalidAuthority(t *testing.T) {
	test := func(name string, launch settings.LaunchState, want string) {
		t.Run(name, func(t *testing.T) {
			rb := newRawSnapshotBroker(t)
			rb.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method == "engine:get-launch" {
					return launch, ""
				}
				return nil, "unexpected snapshot request: " + method
			})
			rb.b.engineConfigMu.Lock()
			_, _, err := rb.b.prepareSettingsSnapshotLocked(t.Context(), "llamacpp")
			rb.b.engineConfigMu.Unlock()
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("snapshot authority = %v, want %q", err, want)
			}
			requireRawSnapshotPrivate(t, rb.b)
		})
	}
	test("missing engine", settings.LaunchState{}, "readback did not identify")
	test("other engine", rawSnapshotLaunch("llamacpp-custom"), "readback did not identify")
	duplicate := rawSnapshotLaunch("llamacpp")
	duplicate.ModelLoads = append(duplicate.ModelLoads, rawSnapshotLoad("small-profile", 16384))
	test("duplicate profile load", duplicate, "profileKey: duplicate")
	unresolved := rawSnapshotLaunch("llamacpp")
	unresolved.ModelLoads[0].Resolved.PerSlotContext = 0
	test("invalid finite context", unresolved, "resolved.context")
}

func TestPrepareSettingsSnapshotKeepsNativeLoadsPrivateAndDetached(t *testing.T) {
	rb := newRawSnapshotBroker(t)
	rb.b.engineConfigMu.Lock()
	record, save, err := rb.b.prepareSettingsSnapshotLocked(t.Context(), "llamacpp")
	rb.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if !save || !modelconfig.EqualLoads(record.Snapshot.Settings.ModelLoads, []modelconfig.ModelLoad{rawSnapshotLoad("small-profile", 8192)}) {
		t.Fatalf("private initial native snapshot = %+v, save=%v", record, save)
	}
	record.Snapshot.Settings.ModelLoads[0].Requested.Context.Tokens = 256
	record.Snapshot.Settings.ModelLoads[0].Resolved.PerSlotContext = 256
	record.Snapshot.Settings.ModelLoads[0].Resolved.Artifacts[0].Path = "changed.gguf"
	requireRawSnapshotPrivate(t, rb.b)
	rb.b.engineConfigMu.Lock()
	next, _, err := rb.b.prepareSettingsSnapshotLocked(t.Context(), "llamacpp")
	rb.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if !modelconfig.EqualLoads(next.Snapshot.Settings.ModelLoads, []modelconfig.ModelLoad{rawSnapshotLoad("small-profile", 8192)}) {
		t.Fatalf("private snapshot leaked into next preparation: %+v", next.Snapshot.Settings.ModelLoads)
	}
	requireRawSnapshotPrivate(t, rb.b)
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
	return newSettingsHarnessForEngine(t, "llamacpp")
}

// newLlamaCPPSettingsHarness is the settings harness for an example engine.
func newLlamaCPPSettingsHarness(t *testing.T) *settingsHarness {
	t.Helper()
	return newSettingsHarnessForEngine(t, engines.NameLlamaCPP)
}

func newSettingsHarnessForEngine(t *testing.T, engine string) *settingsHarness {
	t.Helper()
	return newServingSettingsHarness(t, engine, nil)
}

func newServingSettingsHarness(t *testing.T, engine string, serving *engines.ServingConfig) *settingsHarness {
	t.Helper()
	return newSettingsHarnessWithLaunch(t, engine, serving, true)
}

func newSettingsHarnessWithLaunch(t *testing.T, engine string, serving *engines.ServingConfig, running bool) *settingsHarness {
	t.Helper()
	h := &settingsHarness{engine: engine, b: &Broker{codec: NewCodec(&bytes.Buffer{}), nodeID: "target", clusterDir: filepath.Join(t.TempDir(), "cluster")}, settingsReplies: make(chan json.RawMessage, 8), launchReadWaiting: make(chan struct{}, 1)}
	h.b.setDynamicEngines([]engines.Descriptor{customEngineDescriptor()}, func(int) bool { return true })
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
		"llamacpp":        {ready: true, port: ports[3]},
		"llamacpp-custom": {ready: true, port: ports[2]},
	}}
	publishTestProxy(h.b, proxy)
	h.b.engineProxy(mustEngineProxyProfile("llamacpp")).startupPort.Store(int32(ports[3]))
	custom, ok := h.b.frontedEngineProfile("llamacpp-custom")
	if !ok {
		t.Fatal("fixture custom engine is not fronted")
	}
	h.b.engineProxy(custom).startupPort.Store(int32(ports[2]))
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
			if msg.Method == "facade/enable" {
				var spec enableFacadeRequest
				if err := json.Unmarshal(msg.Params, &spec); err != nil {
					t.Errorf("decode facade request: %v", err)
					return
				}
				if h.facadeBindRace.Load() {
					_ = proxyCodec.RespondError(msg.ID, codeFacadeBindFailed, "fixture bind race")
					continue
				}
				proxy.readyMu.Lock()
				state := proxy.facadeState[spec.Engine]
				if !state.ready {
					state = proxyFacadeState{ready: true, port: spec.Port}
					proxy.facadeState[spec.Engine] = state
				}
				proxy.readyMu.Unlock()
				if h.facadeUnanswered.Load() {
					continue
				}
				_ = proxyCodec.Respond(msg.ID, enableFacadeReply{Engine: spec.Engine, Port: state.port})
				continue
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
			if h.bindRace.Load() {
				_ = proxyCodec.RespondError(msg.ID, -32000, "fixture bind race")
				continue
			}
			proxy.readyMu.Lock()
			proxy.facadeState[engine] = proxyFacadeState{ready: true, port: p.Port}
			proxy.readyMu.Unlock()
			_ = proxyCodec.Respond(msg.ID, map[string]int{"port": p.Port})
		}
	}()
	launch := settings.LaunchState{Engine: engine, ServerPort: ports[0], EffectivePort: ports[0], LaunchText: "--fixture-option", Running: running, Editable: true, Format: "pair-arguments-v1", Serving: serving.Clone()}
	if !running {
		launch.EffectivePort = 0
	}
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
			case "engine:diagnostic-run-status":
				// No diagnostic is retained by this settings fixture.
				_ = codec.Respond(msg.ID, diagnosticManagerStatus{State: "none"})
			case "engine:get-launch":
				var request struct {
					Engine string `json:"engine"`
				}
				if err := json.Unmarshal(msg.Params, &request); err != nil {
					t.Error(err)
					return
				}
				if request.Engine == engine && h.serializeLaunch.Load() {
					select {
					case h.launchReadWaiting <- struct{}{}:
					default:
					}
					h.configureMu.Lock()
				}
				h.launchMu.Lock()
				current := launch
				current.WorkingSetLimitStatus = h.launchStatus
				current.ServingReadback = h.launchReadback.Clone()
				h.launchMu.Unlock()
				if request.Engine == engine && h.serializeLaunch.Load() {
					h.configureMu.Unlock()
				}
				current.Engine = request.Engine
				if request.Engine != engine {
					current.ServerPort = int(h.otherEnginePort.Load())
					current.EffectivePort = current.ServerPort
				}
				_ = codec.Respond(msg.ID, current)
			case "engine:configured-ports":
				_ = codec.Respond(msg.ID, map[string]any{"engines": []any{map[string]any{"engine": "llamacpp-custom", "port": h.otherEnginePort.Load()}}})
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
				preview := settings.Preview{Settings: p.Settings, Restart: restart}
				if digest := h.nativeDigest.Load(); digest != nil && p.Settings.Serving != nil {
					// The engine manager's digest rule: a finite or Strata compile
					// refuses an expected digest that does not match, and fills in an
					// absent one. A typed model-load compile instead replaces any
					// nonempty digest with the one it compiled.
					for i := range p.Settings.Serving.Models {
						model := &p.Settings.Serving.Models[i]
						if !h.typedDigest.Load() && model.ExpectedNativeSourceDigest != "" && model.ExpectedNativeSourceDigest != *digest {
							preview.Errors = map[string]string{"serving": "Native source conflict"}
						}
						model.ExpectedNativeSourceDigest = *digest
					}
				}
				_ = codec.Respond(msg.ID, preview)
			case "engine:configure-launch":
				// Serve this off the reader loop: a suspended apply must not
				// stop the fixture from answering an unrelated read, which is
				// exactly what the broker's lock split makes possible.
				go func(msg *Message) {
					h.configureMu.Lock()
					defer h.configureMu.Unlock()
					h.applies.Add(1)
					if h.entered != nil {
						h.entered <- struct{}{}
						<-h.release
					}
					if h.dropProxyOnStop.Load() {
						h.b.clearProxyHandles()
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
					if p.Resume {
						h.engineStarts.Add(1)
					} else {
						launch.EffectivePort = 0
					}
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
	s, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: h.engine}, "")
	if err != nil {
		t.Fatal(err)
	}
	return settings.Request{Engine: h.engine, ExpectedRevision: s.Revision, RequestID: settingsID(), Settings: s.Settings}
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

func TestSettingsServingReceiptRecoveryAndReadback(t *testing.T) {
	serving := brokerServingFixture()
	h := newServingSettingsHarness(t, "llamacpp", serving)
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
	if !h.b.engineSettings["llamacpp"].Previous.Serving.Equal(serving) {
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
	record := h.b.engineSettings["llamacpp"]
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
	config, ok := h.b.explicitEngineSettings("llamacpp")
	if !ok || !config.Serving.Equal(serving) {
		t.Fatal("recovered config lost serving")
	}
	*config.Serving.Models[0].Memory.HostRAMBytes = 1
	again, ok := h.b.explicitEngineSettings("llamacpp")
	if !ok || !again.Serving.Equal(serving) {
		t.Fatal("recovery config aliases journal")
	}
	snapshot, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil || !snapshot.Settings.Serving.Equal(serving) {
		t.Fatal("settings readback lost serving")
	}
	h.b.engineConfigMu.Lock()
	beforeRefresh := h.b.engineSettings["llamacpp"].Snapshot.Sequence
	h.b.reconcileLegacySettingsLocked()
	afterRefresh := h.b.engineSettings["llamacpp"].Snapshot.Sequence
	h.b.engineConfigMu.Unlock()
	if beforeRefresh != afterRefresh {
		t.Fatal("equivalent serving republished an unchanged refresh")
	}
	*snapshot.Settings.Serving.Models[0].Memory.VRAMBytes = 2
	if !h.b.engineSettings["llamacpp"].Snapshot.Settings.Serving.Equal(serving) {
		t.Fatal("readback aliases journal")
	}
	duplicate, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || duplicate != receipt {
		t.Fatal("recovery changed immutable failed receipt")
	}
}

func TestSettingsPreviewReapprovesStaleNativeDigest(t *testing.T) {
	stale, current := strings.Repeat("c", 64), strings.Repeat("d", 64)
	saved := brokerServingFixture()
	saved.Models[0].ExpectedNativeSourceDigest = stale
	h := newServingSettingsHarness(t, "llamacpp", saved)
	h.nativeDigest.Store(&current)

	preview, err := h.b.previewEngineSettings(context.Background(), h.request(t), "")
	if err != nil {
		t.Fatal(err)
	}
	if len(preview.Errors) != 0 {
		t.Fatalf("preview of the saved settings refused: %v", preview.Errors)
	}
	if got := preview.Settings.Serving.Models[0].ExpectedNativeSourceDigest; got != current {
		t.Fatalf("preview digest = %q, want %q", got, current)
	}

	applies := h.applies.Load()
	if _, err := h.b.applyEngineSettings(context.Background(), h.request(t), ""); err == nil {
		t.Fatal("apply with the stale digest was accepted")
	}
	if h.applies.Load() != applies {
		t.Fatal("apply with the stale digest reached the worker")
	}

	reviewed := h.request(t)
	reviewed.Settings = preview.Settings
	receipt, err := h.b.applyEngineSettings(context.Background(), reviewed, "")
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Phase != "succeeded" {
		t.Fatalf("apply with the reviewed digest = %q, want succeeded", receipt.Phase)
	}
	if h.applies.Load() != applies+1 {
		t.Fatal("apply with the reviewed digest did not reach the worker")
	}
}

// An apply must carry the native source digest the preview returned. Apply
// previews internally and would persist that digest, so an empty one would
// otherwise approve source inputs the caller never reviewed.
func TestSettingsApplyRequiresReviewedNativeDigest(t *testing.T) {
	stale, current := strings.Repeat("c", 64), strings.Repeat("d", 64)
	saved := brokerServingFixture()
	saved.Models[0].ExpectedNativeSourceDigest = stale
	h := newServingSettingsHarness(t, "llamacpp", saved)
	h.nativeDigest.Store(&current)

	unreviewed := h.request(t)
	unreviewed.Settings.Serving.Models[0].ExpectedNativeSourceDigest = ""
	applies := h.applies.Load()
	if _, err := h.b.applyEngineSettings(context.Background(), unreviewed, ""); err == nil {
		t.Fatal("apply with an empty digest was accepted")
	}
	if h.applies.Load() != applies {
		t.Fatal("apply with an empty digest reached the worker")
	}
	after, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if got := after.Settings.Serving.Models[0].ExpectedNativeSourceDigest; got != stale {
		t.Fatalf("saved digest = %q, want the refused apply to leave %q", got, stale)
	}

	// A first-time native configuration is reviewed by its preview, and the
	// digest that preview returns is what the apply carries.
	fresh := newServingSettingsHarness(t, "llamacpp", nil)
	fresh.nativeDigest.Store(&current)
	first := fresh.request(t)
	first.Settings.Serving = brokerServingFixture()
	preview, err := fresh.b.previewEngineSettings(context.Background(), first, "")
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("first preview errors=%v err=%v", preview.Errors, err)
	}
	first.Settings = preview.Settings
	first.ExpectedRevision = preview.Revision
	receipt, err := fresh.b.applyEngineSettings(context.Background(), first, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("first-time apply of the reviewed digest = %q: %v", receipt.Phase, err)
	}

	// Without a compiled native source the preview returns no digest, so an
	// apply that changes other settings still needs no review.
	plain := newServingSettingsHarness(t, "llamacpp", brokerServingFixture())
	edit := plain.request(t)
	edit.Settings.LaunchText += " --non-native-edit"
	receipt, err = plain.b.applyEngineSettings(context.Background(), edit, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("non-native apply = %q: %v", receipt.Phase, err)
	}
}

// A typed model-load apply's request digest is not the guard: its internal
// preview replaces a nonempty value with the one it compiled, so the broker
// must compare the request against the preview result, not just refuse an empty
// one.
func TestSettingsApplyRejectsForgedNativeDigest(t *testing.T) {
	forged, stale, current := strings.Repeat("e", 64), strings.Repeat("c", 64), strings.Repeat("d", 64)
	saved := brokerServingFixture()
	saved.Models[0].ExpectedNativeSourceDigest = stale
	h := newServingSettingsHarness(t, "llamacpp", saved)
	h.nativeDigest.Store(&current)
	h.typedDigest.Store(true)

	forgedApply := h.request(t)
	forgedApply.Settings.Serving.Models[0].ExpectedNativeSourceDigest = forged
	applies := h.applies.Load()
	if _, err := h.b.applyEngineSettings(context.Background(), forgedApply, ""); err == nil {
		t.Fatal("apply with a forged digest was accepted")
	} else if !strings.Contains(err.Error(), "native source review is required") {
		t.Fatalf("forged apply refused as %v, want the native source review error", err)
	}
	if h.applies.Load() != applies {
		t.Fatal("apply with a forged digest reached the worker")
	}
	after, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if got := after.Settings.Serving.Models[0].ExpectedNativeSourceDigest; got != stale {
		t.Fatalf("saved digest = %q, want the refused apply to leave %q", got, stale)
	}

	// The preview's returned digest is the review, and echoing it applies.
	preview, err := h.b.previewEngineSettings(context.Background(), h.request(t), "")
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("preview errors=%v err=%v", preview.Errors, err)
	}
	reviewed := h.request(t)
	reviewed.Settings = preview.Settings
	reviewed.ExpectedRevision = preview.Revision
	receipt, err := h.b.applyEngineSettings(context.Background(), reviewed, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("apply of the reviewed digest = %q: %v", receipt.Phase, err)
	}
	if h.applies.Load() != applies+1 {
		t.Fatal("apply of the reviewed digest did not reach the worker")
	}
}

func TestSettingsApplyOfUnchangedSettings(t *testing.T) {
	test := func(name, status string, wantApplies int32) {
		t.Run(name, func(t *testing.T) {
			h := newServingSettingsHarness(t, "llamacpp", brokerServingFixture())
			h.launchMu.Lock()
			h.launchReadback = &engines.ServingReadback{Status: status}
			h.launchMu.Unlock()
			receipt, err := h.b.applyEngineSettings(context.Background(), h.request(t), "")
			if err != nil {
				t.Fatal(err)
			}
			if receipt.Phase != "succeeded" {
				t.Fatalf("receipt phase = %q, want succeeded", receipt.Phase)
			}
			if got := h.applies.Load(); got != wantApplies {
				t.Fatalf("worker applies = %d, want %d", got, wantApplies)
			}
			if got := h.engineStarts.Load(); got != 0 {
				t.Fatalf("engine starts = %d, want 0", got)
			}
		})
	}
	test("reach the worker on a mismatch", "mismatch", 1)
	test("skip the worker on a healthy readback", "unverified", 0)
}

func TestSettingsServingJournalWriteFailureRestoresDeclaration(t *testing.T) {
	test := func(engine string) {
		t.Run(engine, func(t *testing.T) {
			serving := brokerServingFixture()
			h := newServingSettingsHarness(t, engine, serving)
			p := h.request(t)
			p.Settings.Serving.Models[0].MaxOutputTokens++
			before := h.b.engineSettings[engine].clone()
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
			if h.applies.Load() != 0 || h.proxyRebinds.Load() != 0 || !reflect.DeepEqual(h.b.engineSettings[engine], before) {
				t.Fatal("failed journal write changed declaration/runtime")
			}
		})
	}
	test("llamacpp")
	test("llamacpp-custom")
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
	h := newLlamaCPPSettingsHarness(t)
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
	s := h.b.engineSettings["llamacpp"].Snapshot
	if !s.Settings.Equal(p.Settings) || s.Running || s.AppliedRevision >= s.Revision || s.Error == "" {
		t.Fatalf("untruthful failed snapshot: %+v", s)
	}
	h.fail.Store(false)
	p = h.request(t)
	if _, err = h.b.applyEngineSettings(context.Background(), p, ""); err != nil {
		t.Fatal(err)
	}
	if s := h.b.engineSettings["llamacpp"].Snapshot; !s.Running || s.Phase != "succeeded" {
		t.Fatalf("retry lost resume intent: %+v", s)
	}
}

func TestSettingsStoppedRetryPreservesPortReservation(t *testing.T) {
	h := newLlamaCPPSettingsHarness(t)
	p := h.request(t)
	p.Settings.LaunchText += " --failed-start"
	h.fail.Store(true)
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "failed" {
		t.Fatalf("startup failure receipt = %+v, %v", receipt, err)
	}
	h.fail.Store(false)
	p = h.request(t)
	before := h.b.engineSettings["llamacpp"].Snapshot
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
	if after := h.b.engineSettings["llamacpp"].Snapshot; after.Revision != before.Revision || !after.Settings.Equal(before.Settings) || h.applies.Load() != 1 {
		t.Fatalf("rejected conflict changed settings or runtime: %+v, applies=%d", after, h.applies.Load())
	}
	receipt, err = h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "succeeded" || h.applies.Load() != 2 {
		t.Fatalf("retry on reserved fixture port = %+v, %v, applies=%d", receipt, err, h.applies.Load())
	}
	if after := h.b.engineSettings["llamacpp"].Snapshot; !after.Running {
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
			h := newLlamaCPPSettingsHarness(t)
			p := h.request(t)
			oldPort := p.Settings.ProxyPort
			h.b.regCache = relay.NewRegistrationCache()
			h.b.registerService(noderec.RegisterParams{Service: noderec.ServiceLlamaCPP, Port: oldPort})
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
				if len(registered) != 1 || registered[0].Service != noderec.ServiceLlamaCPP || registered[0].Port != oldPort {
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
		{engine: "llamacpp", service: noderec.ServiceLlamaCPP},
		{engine: "llamacpp-custom", service: "e-llamacpp-custom"},
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
			preview, err := h.b.previewEngineSettings(context.Background(), p, "")
			if err != nil || len(preview.Errors) != 0 || preview.Conflict != nil {
				t.Fatalf("own registered facade rejected: %+v, %v", preview, err)
			}
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
	h := newLlamaCPPSettingsHarness(t)
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
		_, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp-custom"}, "")
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
	h := newLlamaCPPSettingsHarness(t)
	p := h.request(t)
	p.Settings.LaunchText += " --new"
	h.failResultSave.Store(true)
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "failed" || h.b.engineSettings["llamacpp"].Snapshot.Phase != "failed" {
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
		"llamacpp": {"snapshot": {"nodeId":"former-node","engine":"llamacpp","revision":9,"appliedRevision":7,"epoch":"former-epoch","sequence":42,"settings":{"serverPort":8082,"proxyPort":8080,"launchText":"--parallel 2"},"effectiveServerPort":8081,"effectiveProxyPort":8080,"running":true,"adopted":true,"editable":true,"reason":"fixture reason","format":"pair-arguments-v1","phase":"applying","error":"interrupted","requestId":"active-request"},"previous":{"serverPort":8081,"proxyPort":8080,"launchText":"--parallel 1"},"resume":true,"explicit":true,"receipts":{"active-request":{"hash":"active-hash","revision":9,"phase":"applying"},"earlier-request":{"hash":"earlier-hash","revision":7,"phase":"succeeded"}}},
		"llamacpp-custom": {"snapshot": {"engine":"llamacpp-custom","revision":5,"appliedRevision":3,"epoch":"old-epoch","sequence":18,"settings":{"serverPort":18081,"proxyPort":18080,"launchText":"--fixture"},"phase":"failed","error":"startup failure","requestId":"failed-request"},"previous":{"serverPort":18082,"proxyPort":18080},"resume":false,"explicit":false,"receipts":{"failed-request":{"hash":"failed-hash","revision":5,"phase":"failed"}}}
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
	h := newLlamaCPPSettingsHarness(t)
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
	r := h.b.engineSettings["llamacpp"]
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
	if h.applies.Load() != 1 || h.b.engineSettings["llamacpp"].Snapshot.Phase != "succeeded" {
		t.Fatal("accepted operation not recovered")
	}
	if !h.b.recoverEngineSettings() || h.applies.Load() != 1 {
		t.Fatal("completed operation replayed")
	}
	if _, ok := h.b.explicitEngineSettings("llamacpp"); !ok {
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
	if saved["llamacpp"] == nil || saved["llamacpp"].Snapshot.NodeID != "target" {
		t.Fatal("recovery did not persist the current settings owner")
	}
}

func TestSettingsPortValidationIncludesStoppedEngines(t *testing.T) {
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
}
