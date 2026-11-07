// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"nvpair-shared/appdir"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
	"nvpair-ui-broker/relay"
)

func newStoppedSettingsHarness(t *testing.T) *settingsHarness {
	t.Helper()
	dir := t.TempDir()
	for _, key := range []string{"LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "HOME"} {
		t.Setenv(key, dir)
	}
	h := newSettingsHarnessWithLaunch(t, "llamacpp", nil, false)
	h.b.codec = NewCodec(&syncBuffer{})
	h.b.regCache = relay.NewRegistrationCache()
	return h
}

func prepareSettingsFacadeRecovery(t *testing.T, h *settingsHarness) *proxyProcess {
	t.Helper()
	profile := mustEngineProxyProfile("llamacpp")
	pp := h.b.settingsProxy("llamacpp")
	if pp == nil {
		t.Fatal("fixture has no initial proxy")
	}
	pp.readyMu.Lock()
	delete(pp.facadeState, "llamacpp")
	pp.readyMu.Unlock()
	h.b.setEngineProxyHandle(profile, nil)
	h.b.proxyPath = "fixture-proxy"
	h.b.proxyEngines = []string{"llamacpp"}
	h.b.proxySupPhase = proxySupLive
	pp.listenerOnly.Store(true)
	pp.bringUpDone.Store(true)
	return pp
}

func prepareSettingsProcessRecovery(t *testing.T, h *settingsHarness) {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(fakeProxyEnv, fakeProxyExactPorts)
	h.b.proxyPath = exe
	h.b.proxyEngines = []string{"llamacpp"}
	h.b.clearProxyHandles()
	h.b.startProxySupervision(false)
	t.Cleanup(h.b.shutdownProxySupervision)
}

func requireStoppedSettings(t *testing.T, h *settingsHarness, port int) settings.Snapshot {
	t.Helper()
	snapshot, err := h.b.getEngineSettings(t.Context(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Running || snapshot.EffectiveServerPort != 0 || h.engineStarts.Load() != 0 {
		t.Fatalf("engine started during repair: snapshot=%+v starts=%d", snapshot, h.engineStarts.Load())
	}
	if snapshot.Settings.ProxyPort != port || snapshot.EffectiveProxyPort != port {
		t.Fatalf("desired/effective proxy ports=%d/%d, want %d/%d", snapshot.Settings.ProxyPort, snapshot.EffectiveProxyPort, port, port)
	}
	for _, entry := range h.b.regCache.Snapshot() {
		if entry.Service == noderec.ServiceLlamaCPP {
			t.Fatalf("stopped engine advertised: %+v", entry)
		}
	}
	return snapshot
}

func TestSettingsUnavailableProxyKeepsConfiguredPortAndEditabilityWithoutJournal(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	profile := mustEngineProxyProfile("llamacpp")
	h.b.engineProxy(profile).startupPort.Store(18080)
	pp := h.b.settingsProxy("llamacpp")
	pp.readyMu.Lock()
	pp.facadeState["llamacpp"] = proxyFacadeState{ready: false, port: 8080}
	pp.readyMu.Unlock()
	h.b.setEngineProxyHandle(profile, nil)
	snapshot, err := h.b.getEngineSettings(t.Context(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if !snapshot.Editable || snapshot.Settings.ProxyPort != 18080 || snapshot.EffectiveProxyPort != 0 {
		t.Fatalf("unavailable initial snapshot=%+v", snapshot)
	}
	if ready, port := pp.Status("llamacpp"); ready || port != 0 {
		t.Fatalf("unready status=%v/%d, want false/0", ready, port)
	}
}

func TestSettingsUnavailableProxyUsesSavedPortWithoutJournal(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	profile := mustEngineProxyProfile("llamacpp")
	path, err := appdir.Path(profile.PortFile)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"port":18080}`), 0600); err != nil {
		t.Fatal(err)
	}
	h.b.engineProxy(profile).startupPort.Store(0)
	h.b.clearProxyHandles()
	snapshot, err := h.b.getEngineSettings(t.Context(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if !snapshot.Editable || snapshot.Settings.ProxyPort != 18080 || snapshot.EffectiveProxyPort != 0 {
		t.Fatalf("saved initial snapshot=%+v", snapshot)
	}
}

func TestSettingsRecoversMissingFacadeWithoutReplacingProcessOrSibling(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	p := h.request(t)
	p.Settings.ProxyPort = h.reservePort(t)
	pp := prepareSettingsFacadeRecovery(t, h)
	_, sibling := pp.Status("llamacpp-custom")
	before := h.b.localPolicy().Policy
	receipt, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("facade recovery receipt=%+v error=%v", receipt, err)
	}
	if h.b.proxyProcessHandle() != pp || h.b.settingsProxy("llamacpp-custom") != pp {
		t.Fatal("facade recovery replaced the shared process or sibling handle")
	}
	if ready, port := pp.Status("llamacpp-custom"); !ready || port != sibling {
		t.Fatalf("sibling changed: ready=%v port=%d, want true/%d", ready, port, sibling)
	}
	if pp.listenerOnly.Load() || !reflect.DeepEqual(before, h.b.localPolicy().Policy) {
		t.Fatal("facade recovery retained listener-only state or changed saved policy")
	}
	if h.proxyRebinds.Load() != 2 {
		t.Fatalf("addressed persistence and Configure rebind calls=%d, want 2", h.proxyRebinds.Load())
	}
	requireStoppedSettings(t, h, p.Settings.ProxyPort)
}

func TestSettingsMissingHandleConvergesAnExistingFacadeOnAnotherPort(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	p := h.request(t)
	p.Settings.ProxyPort = h.reservePort(t)
	pp := h.b.settingsProxy("llamacpp")
	h.b.setEngineProxyHandle(mustEngineProxyProfile("llamacpp"), nil)
	h.b.proxyPath, h.b.proxyEngines, h.b.proxySupPhase = "fixture-proxy", []string{"llamacpp"}, proxySupLive
	receipt, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || receipt.Phase != "succeeded" || h.b.proxyProcessHandle() != pp {
		t.Fatalf("idempotent enable did not converge: receipt=%+v error=%v", receipt, err)
	}
	requireStoppedSettings(t, h, p.Settings.ProxyPort)
}

func TestSettingsRecoveryPreservesReadyAfterUnansweredEnable(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	p := h.request(t)
	p.Settings.ProxyPort = h.reservePort(t)
	pp := prepareSettingsFacadeRecovery(t, h)
	h.facadeUnanswered.Store(true)
	receipt, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("announced-ready recovery=%+v error=%v", receipt, err)
	}
	if h.b.settingsProxy("llamacpp") != pp {
		t.Fatal("unanswered enable withdrew the current ready facade")
	}
	requireStoppedSettings(t, h, p.Settings.ProxyPort)
	// The facade now owns this port. Fresh validation must recognize it as
	// effective rather than reject PAIR's own listener as a foreign occupant.
	h.b.portAvailableCheck = func(port int) bool { return port != p.Settings.ProxyPort }
	h.facadeUnanswered.Store(false)
	p = h.request(t)
	retried, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || retried.Phase != "succeeded" {
		t.Fatalf("retry on PAIR-owned port=%+v error=%v", retried, err)
	}
	requireStoppedSettings(t, h, p.Settings.ProxyPort)
}

func TestSettingsFacadeActivationClearsListenerOnlyWhenPortSetterFails(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	p := h.request(t)
	p.Settings.ProxyPort = h.reservePort(t)
	pp := prepareSettingsFacadeRecovery(t, h)
	h.bindRace.Store(true)
	receipt, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || receipt.Phase != "failed" {
		t.Fatalf("port setter failure=%+v error=%v", receipt, err)
	}
	if pp.listenerOnly.Load() || h.b.proxyProcessHandle() != pp {
		t.Fatal("successful facade activation left the live process listener-only")
	}
	got := h.b.engineSettings["llamacpp"].Snapshot
	if got.EffectiveProxyPort != p.Settings.ProxyPort || h.applies.Load() != 0 {
		t.Fatalf("failed persistence completion=%+v Configure calls=%d", got, h.applies.Load())
	}
}

func TestSettingsRebindRejectsWrongPortAndRetiredProcess(t *testing.T) {
	test := func(name string, retire bool) {
		t.Run(name, func(t *testing.T) {
			h := newStoppedSettingsHarness(t)
			worker, codec := newTestRPCWorkerPipe(t)
			pp := &proxyProcess{peer: worker.peer, facadeState: readyFacade("llamacpp", 18080)}
			h.b.setProxyProcessHandle(pp)
			h.b.setEngineProxyHandle(mustEngineProxyProfile("llamacpp"), pp)
			go func() {
				for {
					msg, err := codec.Read()
					if err != nil {
						return
					}
					if !msg.IsRequest() {
						continue
					}
					if msg.Method != "llamacpp:set-port" {
						if err := codec.Respond(msg.ID, map[string]bool{"ok": true}); err != nil {
							return
						}
						continue
					}
					port := 18081
					if retire {
						port = 18080
						h.b.clearProxyHandles()
					}
					if err := codec.Respond(msg.ID, proxyReadyParams{Port: port}); err != nil {
						t.Errorf("answer port request: %v", err)
					}
					return
				}
			}()
			if err := h.b.rebindSettingsProxy("llamacpp", 18080); err == nil {
				t.Fatal("rebind accepted a different port or a retired process")
			}
		})
	}
	test("different selected port", false)
	test("process retired during reply", true)
}

func TestSettingsIdempotentRebindRestoresReadinessFromCurrentAcknowledgement(t *testing.T) {
	b := &Broker{}
	worker, codec := newTestRPCWorkerPipe(t)
	pp := &proxyProcess{peer: worker.peer}
	publishTestProxy(b, pp)
	go func() {
		msg, err := codec.Read()
		if err != nil {
			t.Errorf("read idempotent set-port: %v", err)
			return
		}
		var request struct {
			Port int `json:"port"`
		}
		if err := json.Unmarshal(msg.Params, &request); err != nil {
			t.Errorf("decode idempotent set-port: %v", err)
			return
		}
		if msg.Method != "llamacpp:set-port" || request.Port != 18080 {
			t.Errorf("idempotent request=%s %+v", msg.Method, request)
		}
		// The real setter persists an already-bound port without a new event.
		if err := codec.Respond(msg.ID, proxyReadyParams{Version: "fixture", Port: 18080}); err != nil {
			t.Errorf("acknowledge idempotent set-port: %v", err)
		}
	}()
	if err := b.rebindSettingsProxy("llamacpp", 18080); err != nil {
		t.Fatal(err)
	}
	if ready, port := pp.Status("llamacpp"); !ready || port != 18080 {
		t.Fatalf("acknowledged facade status=%v/%d, want true/18080", ready, port)
	}
	var baseline proxyReadyParams
	if err := json.Unmarshal(pp.ReadyParams("llamacpp"), &baseline); err != nil {
		t.Fatal(err)
	}
	if baseline.Port != 18080 || baseline.Version != "fixture" {
		t.Fatalf("subscription baseline=%+v", baseline)
	}
}

func TestSettingsSharedConfigureRebindPinsAcceptedPortForRespawn(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	p := h.request(t)
	profile := mustEngineProxyProfile("llamacpp")
	state := h.b.engineProxy(profile)
	state.explicitSettings.Store(false)
	state.startupPort.Store(0)
	// Native model-settings acceptance marks this same journal explicit, then
	// calls Configure directly without ordinary settings Apply's preparation.
	h.b.engineConfigMu.Lock()
	h.b.engineSettings["llamacpp"].Explicit = true
	err := h.b.saveEngineSettingsLocked()
	h.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	var launch settings.LaunchState
	configure := settings.Configure{Engine: "llamacpp", Settings: p.Settings.Clone(), OperationID: "shared-native-configure"}
	if err := h.b.settingsWorkerCall(t.Context(), "engine:configure-launch", configure, &launch); err != nil {
		t.Fatal(err)
	}
	if !state.explicitSettings.Load() || int(state.startupPort.Load()) != p.Settings.ProxyPort {
		t.Fatalf("shared Configure did not pin accepted port: explicit=%v startup=%d want=%d", state.explicitSettings.Load(), state.startupPort.Load(), p.Settings.ProxyPort)
	}
	if launch.Running || h.engineStarts.Load() != 0 {
		t.Fatal("shared Configure started the stopped engine")
	}
	spec := h.b.defaultEngineFacadeSpec(profile)
	if spec.Port != p.Settings.ProxyPort || !spec.IgnorePersistedPort {
		t.Fatalf("respawn plan lost accepted port: %+v", spec)
	}
	h.facadeBindRace.Store(true)
	err = h.b.enableProxyFacadeWithFallback(t.Context(), h.b.settingsProxy("llamacpp"), spec, func(int) int {
		t.Error("accepted native port must not fall back during respawn")
		return p.Settings.ProxyPort + 1
	})
	if !errors.Is(err, errFacadeBindFailed) || int(state.startupPort.Load()) != p.Settings.ProxyPort {
		t.Fatalf("occupied accepted port changed respawn plan: error=%v startup=%d", err, state.startupPort.Load())
	}
}

func TestSettingsFreshRequestRetriesAnUnavailableUnchangedEndpoint(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	p := h.request(t)
	pp := prepareSettingsFacadeRecovery(t, h)
	h.facadeBindRace.Store(true)
	before := h.b.localPolicy().Policy
	failed, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || failed.Phase != "failed" || h.applies.Load() != 0 {
		t.Fatalf("failed recovery=%+v error=%v Configure calls=%d", failed, err, h.applies.Load())
	}
	record := h.b.engineSettings["llamacpp"]
	if !record.Explicit || record.Snapshot.EffectiveProxyPort != 0 || record.Snapshot.Settings.ProxyPort != p.Settings.ProxyPort {
		t.Fatalf("bind-race outcome lost desired or retained effective state: %+v", record)
	}
	state := h.b.engineProxy(mustEngineProxyProfile("llamacpp"))
	if !state.explicitSettings.Load() || int(state.startupPort.Load()) != p.Settings.ProxyPort {
		t.Fatal("failed activation lost the explicit next-spawn plan")
	}
	h.facadeBindRace.Store(false)
	duplicate, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || duplicate != failed || h.applies.Load() != 0 {
		t.Fatalf("duplicate replayed effects: receipt=%+v error=%v", duplicate, err)
	}
	original := p
	p = h.request(t)
	retried, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || retried.Phase != "succeeded" || h.applies.Load() != 1 || h.b.proxyProcessHandle() != pp {
		t.Fatalf("fresh unchanged retry=%+v error=%v calls=%d", retried, err, h.applies.Load())
	}
	if !reflect.DeepEqual(before, h.b.localPolicy().Policy) {
		t.Fatal("retry changed saved policy")
	}
	duplicate, err = h.b.applyEngineSettings(t.Context(), original, "")
	if err != nil || duplicate != failed || h.applies.Load() != 1 {
		t.Fatalf("successful retry rewrote the original receipt: receipt=%+v error=%v", duplicate, err)
	}
	requireStoppedSettings(t, h, p.Settings.ProxyPort)
}

func TestSettingsRebindRaceKeepsAcceptedSettingsAndObservedOldPort(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	p := h.request(t)
	oldPort := p.Settings.ProxyPort
	p.Settings.ProxyPort = h.reservePort(t)
	h.bindRace.Store(true)
	receipt, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || receipt.Phase != "failed" {
		t.Fatalf("bind race receipt=%+v error=%v", receipt, err)
	}
	snapshot, err := h.b.getEngineSettings(t.Context(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Settings.ProxyPort != p.Settings.ProxyPort || snapshot.EffectiveProxyPort != oldPort || !strings.Contains(snapshot.Error, "bind race") {
		t.Fatalf("bind race snapshot=%+v, want accepted %d and observed %d", snapshot, p.Settings.ProxyPort, oldPort)
	}
	if snapshot.Running || h.engineStarts.Load() != 0 {
		t.Fatal("bind race started the engine")
	}
	journal, err := h.b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(journal)
	if err != nil {
		t.Fatal(err)
	}
	var stored map[string]*engineSettingsRecord
	if err := json.Unmarshal(data, &stored); err != nil {
		t.Fatal(err)
	}
	accepted := stored["llamacpp"]
	if accepted == nil || !accepted.Explicit || accepted.Snapshot.Settings.ProxyPort != p.Settings.ProxyPort || accepted.Receipts[p.RequestID].Phase != "failed" {
		t.Fatalf("bind race did not durably retain accepted desired settings and failed receipt: %+v", accepted)
	}
}

func TestSettingsAcceptanceFailureDoesNotChangeProxyStartupPlan(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	p := h.request(t)
	profile := mustEngineProxyProfile("llamacpp")
	state := h.b.engineProxy(profile)
	before := state.startupPort.Load()
	p.Settings.ProxyPort = h.reservePort(t)
	journal, err := h.b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(journal); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(journal, 0700); err != nil {
		t.Fatal(err)
	}
	if _, err := h.b.applyEngineSettings(t.Context(), p, ""); err == nil {
		t.Fatal("acceptance save unexpectedly succeeded")
	}
	if state.startupPort.Load() != before || state.explicitSettings.Load() || h.applies.Load() != 0 || h.proxyRebinds.Load() != 0 {
		t.Fatal("failed acceptance changed runtime or its startup plan")
	}
}

func TestSettingsCompletionClearsEffectivePortWhenProxyDisappears(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	p := h.request(t)
	p.Settings.ProxyPort = h.reservePort(t)
	h.dropProxyOnStop.Store(true)
	receipt, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || receipt.Phase != "failed" {
		t.Fatalf("disappeared proxy receipt=%+v error=%v", receipt, err)
	}
	// Inspect the completion before another settings read can repair stale state.
	got := h.b.engineSettings["llamacpp"].Snapshot
	if got.EffectiveProxyPort != 0 || got.Settings.ProxyPort != p.Settings.ProxyPort || got.Running {
		t.Fatalf("disappeared proxy completion=%+v", got)
	}
	if len(h.b.regCache.Snapshot()) != 0 {
		t.Fatal("disappeared stopped endpoint is still advertised")
	}
}

func TestSettingsProxyRecoveryPreservesSupervisionFences(t *testing.T) {
	test := func(name string, setup func(*Broker), want string) {
		t.Run(name, func(t *testing.T) {
			h := newStoppedSettingsHarness(t)
			profile := mustEngineProxyProfile("llamacpp")
			h.b.clearProxyHandles()
			h.b.proxyPath, h.b.proxyEngines, h.b.proxySupPhase = "fixture-proxy", []string{"llamacpp"}, proxySupLive
			setup(h.b)
			sup := h.b.getProxySupervisor()
			err := h.b.recoverSettingsProxy(t.Context(), profile, 18080)
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("recovery error=%v, want %q", err, want)
			}
			if h.b.getProxySupervisor() != sup || h.b.proxyProcessHandle() != nil || h.applies.Load() != 0 {
				t.Fatal("refused recovery replaced supervision or started work")
			}
		})
	}
	test("starting", func(b *Broker) { b.proxySupPhase = proxySupStarting }, "not available")
	test("closing", func(b *Broker) { b.proxySupClosing.Store(true) }, "not available")
	test("closed", func(b *Broker) { b.proxySupPhase = proxySupClosed }, "not available")
	test("unresolved binary", func(b *Broker) { b.proxyPath = "" }, "disabled")
	test("disabled facade", func(b *Broker) { b.proxyEngines = nil }, "disabled")
	test("existing recovery budget", func(b *Broker) {
		b.setProxySupervisor(newSupervisor("nvpair-proxy", defaultRestartPolicy(), func() (supervisedHandle, error) {
			return nil, fmt.Errorf("unexpected spawn")
		}))
	}, "recovering")
}

func TestSettingsFreshRequestRenewsExhaustedProxySupervision(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	p := h.request(t)
	p.Settings.ProxyPort = 18080
	useSyntheticBrokerPortChecks(t, h.b)
	prepareSettingsProcessRecovery(t, h)
	sup := newSupervisor("nvpair-proxy", noRestartPolicy(), h.b.spawnProxy)
	h.b.configureProxySupervisorCallbacks(sup)
	sup.onExhausted = func(int) { h.b.proxySupExhausted.Store(true) }
	h.b.setProxySupervisor(sup)
	if err := sup.Start(); err != nil {
		// A failed Start has no monitor for shutdown to join.
		h.b.setProxySupervisor(nil)
		t.Fatal(err)
	}
	first := h.b.proxyProcessHandle()
	if first == nil {
		t.Fatal("fixture supervisor did not publish its process")
	}
	if err := first.cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	eventually(t, "proxy supervision exhausting", h.b.proxySupExhausted.Load)
	receipt, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("exhaustion retry=%+v error=%v", receipt, err)
	}
	if h.b.getProxySupervisor() == sup || h.b.proxyProcessHandle() == first {
		t.Fatal("fresh request did not renew exhausted supervision")
	}
	requireStoppedSettings(t, h, 18080)
}

func TestSettingsMissingProcessRecoversWithRoutingDisabled(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	h.b.engineProxy(mustEngineProxyProfile("llamacpp")).startupPort.Store(8080)
	h.b.clearProxyHandles()
	p := h.request(t)
	if p.Settings.ProxyPort != 8080 {
		t.Fatalf("initial desired proxy port=%d, want 8080", p.Settings.ProxyPort)
	}
	p.Settings.ProxyPort = 18080
	// The unavailable old port remains unavailable throughout the operation.
	// Socket occupancy itself is covered by the real-proxy literal-port case.
	h.b.portAvailableCheck = func(port int) bool { return port != 8080 }
	prepareSettingsProcessRecovery(t, h)
	before := h.b.localPolicy().Policy
	if before.Routing.Listener.Enabled {
		t.Fatal("fixture unexpectedly enables routing")
	}
	receipt, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || receipt.Phase != "succeeded" || h.b.getProxySupervisor() == nil {
		t.Fatalf("missing-process recovery=%+v error=%v", receipt, err)
	}
	if !reflect.DeepEqual(before, h.b.localPolicy().Policy) {
		t.Fatal("process recovery changed saved policy")
	}
	if h.b.portAvailable(8080) {
		t.Fatal("recovery required releasing the old port")
	}
	requireStoppedSettings(t, h, 18080)
}

func TestSettingsInterruptedReplayAndRespawnKeepAcceptedExplicitPort(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	p := h.request(t)
	p.Settings.ProxyPort = 18080
	useSyntheticBrokerPortChecks(t, h.b)
	h.failBeforeStop.Store(true)
	receipt, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || receipt.Phase != "failed" {
		t.Fatalf("prepare interrupted record=%+v error=%v", receipt, err)
	}
	h.failBeforeStop.Store(false)
	h.b.engineConfigMu.Lock()
	record := h.b.engineSettings["llamacpp"]
	record.Snapshot.Phase = "applying"
	record.Receipts[p.RequestID] = settingsReceipt{Hash: receipt.Hash, Revision: receipt.Revision, Phase: "applying"}
	err = h.b.saveEngineSettingsLocked()
	h.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	prepareSettingsProcessRecovery(t, h)
	// Simulate a fresh broker, not merely the in-memory accepted port plan.
	h.b.engineSettingsLoaded = false
	h.b.engineSettings = nil
	state := h.b.engineProxy(mustEngineProxyProfile("llamacpp"))
	state.startupPort.Store(8080)
	state.explicitSettings.Store(false)
	if !h.b.recoverEngineSettings() {
		t.Fatal("interrupted journal recovery failed")
	}
	if got := h.b.engineSettings["llamacpp"].Receipts[p.RequestID]; got.Phase != "succeeded" {
		t.Fatalf("replayed receipt=%+v", got)
	}
	requireStoppedSettings(t, h, 18080)
	first := h.b.proxyProcessHandle()
	if err := first.cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	eventually(t, "supervised respawn on accepted port", func() bool {
		pp := h.b.proxyProcessHandle()
		if pp == nil || pp == first {
			return false
		}
		ready, port := pp.Status("llamacpp")
		return pp.bringUpDone.Load() && ready && port == 18080
	})
	requireStoppedSettings(t, h, 18080)
}

func TestSettingsHealthyStoppedEngineEditsEitherPort(t *testing.T) {
	test := func(name string, server bool) {
		t.Run(name, func(t *testing.T) {
			h := newStoppedSettingsHarness(t)
			p := h.request(t)
			before := h.b.proxyProcessHandle()
			if server {
				p.Settings.ServerPort = h.reservePort(t)
			} else {
				p.Settings.ProxyPort = h.reservePort(t)
			}
			receipt, err := h.b.applyEngineSettings(t.Context(), p, "")
			if err != nil || receipt.Phase != "succeeded" {
				t.Fatalf("stopped edit=%+v error=%v", receipt, err)
			}
			got := requireStoppedSettings(t, h, p.Settings.ProxyPort)
			if !got.Settings.Equal(p.Settings) || h.b.proxyProcessHandle() != before {
				t.Fatal("healthy edit changed desired settings or replaced the proxy process")
			}
		})
	}
	test("server port", true)
	test("proxy port", false)
}

func TestSettingsSameEngineReadDuringConfigureDoesNotBlockRebind(t *testing.T) {
	h := newStoppedSettingsHarness(t)
	p := h.request(t)
	p.Settings.ProxyPort = h.reservePort(t)
	prepareSettingsProcessRecovery(t, h)
	h.entered, h.release = make(chan struct{}, 1), make(chan struct{})
	done := make(chan error, 1)
	go func() {
		receipt, err := h.b.applyEngineSettings(t.Context(), p, "")
		if err == nil && receipt.Phase != "succeeded" {
			err = fmt.Errorf("apply phase %q", receipt.Phase)
		}
		done <- err
	}()
	select {
	case <-h.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("Configure did not reach its operation lock")
	}
	// Arm after Apply's initial launch reads, so only the read below can
	// satisfy this barrier while Configure holds the manager operation lock.
	h.serializeLaunch.Store(true)
	// Hold the broker's read mutex while the read waits for that same
	// engine's manager operation lock. Configure's callback must not need it.
	h.b.engineConfigMu.Lock()
	read := make(chan error, 1)
	go func() {
		_, err := h.b.settingsSnapshotLocked(t.Context(), "llamacpp")
		h.b.engineConfigMu.Unlock()
		read <- err
	}()
	select {
	case <-h.launchReadWaiting:
	case <-time.After(5 * time.Second):
		close(h.release)
		t.Fatal("same-engine launch read did not reach the held operation lock")
	}
	close(h.release)
	for _, result := range []<-chan error{done, read} {
		select {
		case err := <-result:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("same-engine read and Configure callback deadlocked")
		}
	}
	requireStoppedSettings(t, h, p.Settings.ProxyPort)
}

func TestSettingsLiteral8080To18080RecoveryKeepsOtherOwnerAndEngineStopped(t *testing.T) {
	runSettingsOccupiedProxyPortRecovery(t, ":8080", 18080)
}

func TestSettingsOccupiedProxyPortRecoveryKeepsOtherOwnerAndEngineStopped(t *testing.T) {
	runSettingsOccupiedProxyPortRecovery(t, ":0", 0)
}

func TestSettingsLoopbackOccupiedProxyPortRecoveryKeepsOtherOwnerAndEngineStopped(t *testing.T) {
	t.Run("IPv4 loopback", func(t *testing.T) {
		runSettingsOccupiedProxyPortRecovery(t, "127.0.0.1:0", 0)
	})
	t.Run("IPv6 loopback", func(t *testing.T) {
		runSettingsOccupiedProxyPortRecovery(t, "[::1]:0", 0)
	})
}

func runSettingsOccupiedProxyPortRecovery(t *testing.T, address string, targetPort int) {
	t.Helper()
	occupant, err := net.Listen("tcp", address)
	if err != nil {
		if address == ":8080" {
			t.Skipf("literal-port fixture cannot own 8080: %v", err)
		}
		if address == "[::1]:0" {
			t.Skipf("IPv6 loopback is unavailable: %v", err)
		}
		t.Fatalf("occupy private proxy port: %v", err)
	}
	oldPort := occupant.Addr().(*net.TCPAddr).Port
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTeapot)
	})}
	served := make(chan error, 1)
	go func() { served <- server.Serve(occupant) }()
	t.Cleanup(func() {
		if err := server.Close(); err != nil {
			t.Errorf("close competing listener: %v", err)
		}
		select {
		case err := <-served:
			if !errors.Is(err, http.ErrServerClosed) {
				t.Errorf("competing HTTP server: %v", err)
			}
		case <-time.After(time.Second):
			t.Error("competing HTTP server did not stop")
		}
	})
	client := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{DisableKeepAlives: true}}
	t.Cleanup(client.CloseIdleConnections)
	occupantHost := "127.0.0.1"
	if ip := occupant.Addr().(*net.TCPAddr).IP; ip.IsLoopback() {
		occupantHost = ip.String()
	}
	askOccupant := func() {
		t.Helper()
		response, err := client.Get("http://" + net.JoinHostPort(occupantHost, fmt.Sprint(oldPort)) + "/v1/models")
		if err != nil {
			t.Fatalf("competing owner was disturbed: %v", err)
		}
		if err := response.Body.Close(); err != nil {
			t.Fatal(err)
		}
		if response.StatusCode != http.StatusTeapot {
			t.Fatalf("competing owner HTTP status=%d, want %d", response.StatusCode, http.StatusTeapot)
		}
	}
	askOccupant()
	if targetPort == 18080 && !tcpPortAvailable(targetPort) {
		t.Skip("literal-port fixture requires free port 18080")
	}
	h := newStoppedSettingsHarness(t)
	h.b.setDynamicEngines(nil, func(int) bool { return true })
	profile := mustEngineProxyProfile("llamacpp")
	h.b.engineProxy(profile).startupPort.Store(int32(oldPort))
	h.b.clearProxyHandles()
	p := h.request(t)
	h.b.engineConfigMu.Lock()
	h.b.engineSettings["llamacpp"].Explicit = true
	err = h.b.saveEngineSettingsLocked()
	h.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(testbin.Dir(t, "tests", "products", "nvpair-proxy"), testbin.Exe("nvpair-proxy"))
	if err := buildPortTestBinary(binary, "../nvpair-proxy", "."); err != nil {
		t.Fatal(err)
	}
	h.b.proxyPath, h.b.proxyEngines = binary, []string{"llamacpp"}
	h.b.clearProxyHandles()
	ctx, cancel := context.WithCancel(t.Context())
	h.b.setServeCtx(ctx)
	t.Cleanup(func() { cancel(); h.b.shutdownProxySupervision() })
	h.b.prepareEnabledFacades()
	h.b.startProxySupervision(true)
	if h.b.proxyProcessHandle() != nil {
		t.Fatal("explicit occupied port unexpectedly started a proxy")
	}
	snapshot, err := h.b.getEngineSettings(t.Context(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if !snapshot.Editable || snapshot.EffectiveProxyPort != 0 || snapshot.Running || h.engineStarts.Load() != 0 {
		t.Fatalf("occupied endpoint settings=%+v starts=%d", snapshot, h.engineStarts.Load())
	}
	askOccupant()
	p = h.request(t)
	if p.Settings.ProxyPort != oldPort {
		t.Fatalf("unavailable desired proxy port=%d, want %d", p.Settings.ProxyPort, oldPort)
	}
	if targetPort == 0 {
		available, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		targetPort = available.Addr().(*net.TCPAddr).Port
		if err := available.Close(); err != nil {
			t.Fatal(err)
		}
	}
	p.Settings.ProxyPort = targetPort
	before := h.b.localPolicy().Policy
	receipt, err := h.b.applyEngineSettings(t.Context(), p, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("occupied-port recovery=%+v error=%v", receipt, err)
	}
	requireStoppedSettings(t, h, targetPort)
	if !reflect.DeepEqual(before, h.b.localPolicy().Policy) {
		t.Fatal("occupied-port recovery changed saved policy")
	}
	response, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/v1/models", targetPort))
	if err != nil {
		t.Fatalf("recovered proxy HTTP endpoint: %v", err)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	// The policy catalog answers independently of native engine readiness.
	// Reading it after binding recovery must leave the local engine stopped.
	if response.StatusCode != http.StatusOK {
		t.Fatalf("policy catalog status=%d, want %d", response.StatusCode, http.StatusOK)
	}
	requireStoppedSettings(t, h, targetPort)
	// Fresh HTTP requests still reach the competing owner on its original port.
	askOccupant()
	portPath, err := appdir.Path(profile.PortFile)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(portPath)
	if err != nil {
		t.Fatal(err)
	}
	var stored struct {
		Port int `json:"port"`
	}
	if err := json.Unmarshal(data, &stored); err != nil {
		t.Fatal(err)
	}
	if stored.Port != targetPort {
		t.Fatalf("component saved port=%d, want %d", stored.Port, targetPort)
	}
}
