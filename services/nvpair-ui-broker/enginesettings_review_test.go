// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"nvpair-shared/appdir"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
)

func TestSettingsRebindAddressesOnlyRequestedFacade(t *testing.T) {
	test := func(engine string) {
		t.Run(engine, func(t *testing.T) {
			h := newSettingsHarness(t)
			p := h.b.engineProxyHandle(mustEngineProxyProfile("llamacpp"))
			before := make(map[string]int)
			for _, candidate := range h.b.frontedEngineProfiles() {
				_, before[candidate.Name] = p.Status(candidate.Name)
			}
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			port := ln.Addr().(*net.TCPAddr).Port
			_ = ln.Close()
			if err := h.b.rebindSettingsProxy(engine, port); err != nil {
				t.Fatal(err)
			}
			before[engine] = port
			for engine, want := range before {
				ready, got := p.Status(engine)
				if !ready || got != want {
					t.Fatalf("%s ready=%v port=%d, want %d", engine, ready, got, want)
				}
			}
		})
	}
	test("llamacpp")
	test("llamacpp-custom")
}

func TestExplicitSettingsBindFailurePreservesChosenPort(t *testing.T) {
	test := func(engine string) {
		t.Run(engine, func(t *testing.T) {
			isolateBrokerTestConfig(t)
			const requested = 25000
			previous := &Broker{
				engineSettings: map[string]*engineSettingsRecord{engine: {
					Explicit: true, Snapshot: settings.Snapshot{Settings: settings.Config{ServerPort: 24999, ProxyPort: requested}},
				}},
			}
			if err := previous.saveEngineSettingsLocked(); err != nil {
				t.Fatal(err)
			}
			path, err := previous.engineSettingsPath()
			if err != nil {
				t.Fatal(err)
			}
			before, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			// A fresh broker must recover from disk, with no runtime state inherited.
			b := &Broker{codec: NewCodec(&bytes.Buffer{}), proxyPath: "test-proxy", proxyEngines: []string{"llamacpp"}}
			b.setDynamicEngines([]engines.Descriptor{customEngineDescriptor()}, func(int) bool { return true })
			profile, ok := b.frontedEngineProfile(engine)
			if !ok {
				t.Fatal("missing fronted profile")
			}
			b.prepareEnabledFacades()
			state := b.engineProxy(profile)
			if !state.explicitSettings.Load() || state.backendPort.Load() != 24999 || state.startupPort.Load() != requested {
				t.Fatalf("explicit settings not restored: explicit=%v server=%d proxy=%d", state.explicitSettings.Load(), state.backendPort.Load(), state.startupPort.Load())
			}
			spec := b.defaultEngineFacadeSpec(profile)
			if _, dynamic := b.dynamicEngineProfile(engine); dynamic {
				spec, err = b.dynamicEngineFacadeSpec(profile)
				if err != nil {
					t.Fatal(err)
				}
				if spec.Descriptor == nil || spec.Descriptor.Engine != engine {
					t.Fatalf("descriptor facade spec = %+v", spec)
				}
			}
			if spec.Engine != engine || spec.Port != requested || !spec.IgnorePersistedPort {
				t.Fatalf("startup spec = %+v, want journal port %d overriding proxy persistence", spec, requested)
			}
			failure := settingsJSON(map[string]any{"code": "bind-failed", "port": requested})
			b.forwardDefaultEngineProxyNotification(profile, profile.addressed("error"), failure)
			if got := b.engineProxy(profile).startupPort.Load(); got != requested {
				t.Fatalf("bind notification changed chosen port to %d", got)
			}
			client, server := net.Pipe()
			t.Cleanup(func() { _ = client.Close(); _ = server.Close() })
			p := &proxyProcess{peer: NewPeer(NewCodec(client))}
			go p.peer.Serve(nil, nil)
			attempts := make(chan enableFacadeRequest, 2)
			serveFacadeEnable(t, server, map[int]bool{requested: true}, attempts)
			err = b.enableProxyFacadeWithFallback(context.Background(), p, spec, func(int) int {
				t.Error("explicit port must not fall back")
				return requested + 1
			})
			if !errors.Is(err, errFacadeBindFailed) {
				t.Fatalf("bind failure = %v, want facade bind failure", err)
			}
			if len(attempts) != 1 {
				t.Fatalf("attempts=%d, want only the chosen port", len(attempts))
			}
			attempt := <-attempts
			if attempt.Engine != engine || attempt.Port != requested || !attempt.IgnorePersistedPort {
				t.Fatalf("bind attempt = %+v, want explicit journal port %d", attempt, requested)
			}
			if got := state.startupPort.Load(); got != requested {
				t.Fatalf("bind failure changed chosen port to %d", got)
			}
			after, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(before, after) {
				t.Fatal("bind failure changed the saved settings journal")
			}
		})
	}
	test("llamacpp")
	test("llamacpp-custom")
}

func TestDescriptorFacadePreparationWithoutExplicitSettingsRetainsPlannedPort(t *testing.T) {
	isolateBrokerTestConfig(t)
	b := &Broker{
		proxyPath: "test-proxy", engineSettingsLoaded: true,
		engineSettings: map[string]*engineSettingsRecord{"llamacpp-custom": {
			Snapshot: settings.Snapshot{Settings: settings.Config{ServerPort: 24999, ProxyPort: 25000}},
		}},
	}
	b.setDynamicEngines([]engines.Descriptor{customEngineDescriptor()}, func(int) bool { return true })
	b.prepareEnabledFacades()
	profile, ok := b.frontedEngineProfile("llamacpp-custom")
	if !ok {
		t.Fatal("missing descriptor profile")
	}
	state := b.engineProxy(profile)
	if state.explicitSettings.Load() || state.backendPort.Load() != 0 {
		t.Fatalf("non-explicit record restored as explicit: explicit=%v server=%d", state.explicitSettings.Load(), state.backendPort.Load())
	}
	spec, err := b.dynamicEngineFacadeSpec(profile)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Port != engines.DynamicFacadePortBase || spec.IgnorePersistedPort || spec.Descriptor == nil {
		t.Fatalf("facade spec = %+v, want planned port %d with proxy persistence enabled", spec, engines.DynamicFacadePortBase)
	}
}

func TestSettingsReservesStoppedProxySavedPort(t *testing.T) {
	test := func(name string, serverPort bool) {
		t.Run(name, func(t *testing.T) {
			h := newSettingsHarness(t)
			request := h.request(t)
			_, reserved := h.b.engineProxyHandle(h.b.dynamicEngineProfiles()[0]).Status("llamacpp-custom")
			h.b.setEngineProxyHandle(h.b.dynamicEngineProfiles()[0], nil)
			h.b.engineSettings["llamacpp-custom"] = &engineSettingsRecord{
				Explicit: true,
				Snapshot: settings.Snapshot{Settings: settings.Config{ProxyPort: reserved}},
			}
			if serverPort {
				request.Settings.ServerPort = reserved
			} else {
				request.Settings.ProxyPort = reserved
			}
			preview, err := h.b.previewEngineSettings(context.Background(), request, "")
			if err != nil {
				t.Fatal(err)
			}
			if len(preview.Errors) == 0 {
				t.Fatalf("accepted stopped proxy's saved port: %+v", preview)
			}
		})
	}
	test("server port cannot reuse saved proxy port", true)
	test("proxy port cannot reuse saved proxy port", false)
}

func TestSettingsSiblingPortConflictDoesNotWrite(t *testing.T) {
	test := func(name, engine, sibling, source string) {
		t.Run(name, func(t *testing.T) {
			isolateBrokerTestConfig(t)
			h := newSettingsHarnessForEngine(t, engine)
			request := h.request(t)
			profile, ok := h.b.frontedEngineProfile(sibling)
			if !ok {
				t.Fatal("missing sibling profile")
			}
			_, port := h.b.settingsProxy(sibling).Status(sibling)
			if source != "live" {
				h.b.setEngineProxyHandle(profile, nil)
			}
			switch source {
			case "journal":
				h.b.engineSettings[sibling] = &engineSettingsRecord{
					Snapshot: settings.Snapshot{Engine: sibling, Settings: settings.Config{ProxyPort: port}},
					Receipts: make(map[string]settingsReceipt),
				}
				if err := h.b.saveEngineSettingsLocked(); err != nil {
					t.Fatal(err)
				}
			case "proxy file":
				path, err := appdir.Path(profile.PortFile)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
					t.Fatal(err)
				}
				data, err := json.Marshal(map[string]int{"port": port})
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			path, err := h.b.engineSettingsPath()
			if err != nil {
				t.Fatal(err)
			}
			before, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			request.Settings.ProxyPort = port
			preview, err := h.b.previewEngineSettings(context.Background(), request, "")
			if err != nil {
				t.Fatal(err)
			}
			want := "a selected port is reserved by another configured proxy"
			if source == "live" {
				want = "a selected port is already used by another proxy"
			}
			if preview.Errors["ports"] != want {
				t.Fatalf("port error = %q, want %q", preview.Errors["ports"], want)
			}
			if _, err := h.b.applyEngineSettings(context.Background(), request, ""); err == nil {
				t.Fatal("conflicting apply succeeded")
			}
			after, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(before, after) || h.applies.Load() != 0 || h.proxyRebinds.Load() != 0 {
				t.Fatal("rejected port conflict changed journal or runtime")
			}
		})
	}
	test("builtin reserves live descriptor", "llamacpp", "llamacpp-custom", "live")
	test("builtin reserves descriptor journal", "llamacpp", "llamacpp-custom", "journal")
	test("builtin reserves descriptor proxy file", "llamacpp", "llamacpp-custom", "proxy file")
	test("descriptor reserves live builtin", "llamacpp-custom", "llamacpp", "live")
	test("descriptor reserves builtin journal", "llamacpp-custom", "llamacpp", "journal")
	test("descriptor reserves builtin proxy file", "llamacpp-custom", "llamacpp", "proxy file")
}

func TestEnabledEngineRestorationSurvivesInvalidSettingsJournal(t *testing.T) {
	b := &Broker{clusterDir: filepath.Join(t.TempDir(), "cluster")}
	path, err := b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("{invalid"), 0600); err != nil {
		t.Fatal(err)
	}
	worker, codec := newTestRPCWorkerPipe(t)
	b.setEngineMgr(worker)
	restored := recordEnabledEngineRestoration(t, codec)
	done := make(chan struct{})
	go func() { b.restoreEnabledEngines(worker); close(done) }()
	for _, want := range []string{engineNodeProjectionMethod, restoreEnabledEnginesMethod} {
		select {
		case method := <-restored:
			if method != want {
				t.Fatalf("restoration method=%q, want %q", method, want)
			}
		case <-time.After(time.Second):
			t.Fatalf("invalid journal suppressed %s", want)
		}
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("enabled-engine restoration did not return")
	}
	if b.engineSettingsError == nil {
		t.Fatal("invalid journal was not reported")
	}
}

func TestSettingsFullCommandJournalMigratesBeforeRecovery(t *testing.T) {
	h := newLlamaCPPSettingsHarness(t)
	request := h.request(t)
	h.b.engineConfigMu.Lock()
	record := h.b.engineSettings["llamacpp"]
	record.Snapshot.Format = "pair-launch-v1"
	record.Snapshot.Settings.LaunchText = "managed serve --fixture-option --future-option"
	record.Snapshot.Phase = "applying"
	record.Resume = true
	oldProxyPort := record.Snapshot.Settings.ProxyPort
	if err := h.b.saveEngineSettingsLocked(); err != nil {
		t.Fatal(err)
	}
	h.b.engineConfigMu.Unlock()
	if !h.b.recoverEngineSettings() {
		t.Fatal("recovery failed")
	}
	snapshot, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "llamacpp"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Format != "pair-arguments-v1" || snapshot.Phase != "succeeded" || snapshot.Settings.LaunchText != "--fixture-option --future-option" || snapshot.Settings.ProxyPort != oldProxyPort || snapshot.Settings.ServerPort != request.Settings.ServerPort {
		t.Fatalf("migration lost accepted configuration: %+v", snapshot)
	}
	if h.applies.Load() != 1 {
		t.Fatal("pending operation was not applied exactly once")
	}
}
