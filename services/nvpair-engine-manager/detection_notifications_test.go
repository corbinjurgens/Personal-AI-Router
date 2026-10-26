// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestConfiguredDetectionCompleteCommandRejectsOccupiedListener(t *testing.T) {
	ex, st := configuredDetectionExecutor(t)
	var response atomic.Int32
	response.Store(http.StatusServiceUnavailable)
	st.port = detectionServer(t, &response)
	startMarker := filepath.Join(t.TempDir(), "start-ran")
	cleanupMarker := filepath.Join(t.TempDir(), "cleanup-ran")
	st.plat.Runtime.Mode = "command"
	st.plat.Runtime.Start = [][]string{{fakeEngineBin, "touch", startMarker}}
	st.plat.Runtime.Stop = &StopSpec{Cmd: []string{fakeEngineBin, "touch", cleanupMarker}, GraceS: 1}
	st.plat.Runtime.Ready.TimeoutS = 1
	if installed, err := ex.Detect("fake"); err != nil || !installed {
		t.Fatalf("complete evidence = %v, %v", installed, err)
	}
	if err := ex.Start(context.Background(), "fake"); err == nil || !strings.Contains(err.Error(), "occupied") {
		t.Errorf("start against unidentified listener = %v, want occupied refusal", err)
	}
	for _, path := range []string{startMarker, cleanupMarker} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Errorf("command executed over unidentified listener (%s): %v", filepath.Base(path), err)
		}
	}
}

// Collect typed notifications independently of the snapshot returned by a poll.
// The mutex also covers an owned process's asynchronous exit publication.
func recordDetectionStates(t *testing.T, ex *Executor) func() []EngineStatus {
	t.Helper()
	var mu sync.Mutex
	var events []EngineStatus
	ex.emit = func(method string, params any) {
		if method != "engine:state-changed" {
			return
		}
		status, ok := params.(EngineStatus)
		if !ok {
			t.Errorf("state notification type = %T", params)
			return
		}
		mu.Lock()
		events = append(events, status)
		mu.Unlock()
	}
	return func() []EngineStatus {
		mu.Lock()
		defer mu.Unlock()
		out := append([]EngineStatus(nil), events...)
		events = nil
		return out
	}
}

func TestConfiguredDetectionInstallationNotifications(t *testing.T) {
	test := func(name string, start func(*testing.T, *Executor, *engineState)) {
		t.Run(name, func(t *testing.T) {
			ex, st := configuredDetectionExecutor(t)
			start(t, ex, st)
			takeEvents := recordDetectionStates(t, ex)
			poll := func(wantInstalled bool) {
				t.Helper()
				got, err := ex.Status("fake")
				if err != nil || got.Installed != wantInstalled || !got.Running || !got.Healthy {
					t.Fatalf("status = %+v, %v, want installed %v", got, err, wantInstalled)
				}
				listed := ex.GetInstalled()
				if len(listed) != 1 || listed[0].Installed != wantInstalled || !listed[0].Running || !listed[0].Healthy {
					t.Fatalf("get-installed = %+v, want installed %v", listed, wantInstalled)
				}
			}
			assertTransition := func(wantInstalled bool) {
				t.Helper()
				events := takeEvents()
				if len(events) != 1 || events[0].Installed != wantInstalled || !events[0].Running || !events[0].Healthy {
					t.Fatalf("transition notifications = %+v, want one installed=%v running healthy snapshot", events, wantInstalled)
				}
			}
			poll(true)
			if events := takeEvents(); len(events) != 0 {
				t.Fatalf("unchanged initial polls notified: %+v", events)
			}
			config := filepath.Join(st.installDir, "config.json")
			removeDetectionFile(t, config)
			poll(false)
			assertTransition(false)
			poll(false)
			if events := takeEvents(); len(events) != 0 {
				t.Fatalf("unchanged incomplete polls notified: %+v", events)
			}
			writeDetectionFile(t, config)
			// GetInstalled owns the restoration refresh, then Status observes it.
			listed := ex.GetInstalled()
			if len(listed) != 1 || !listed[0].Installed {
				t.Fatalf("restored get-installed = %+v", listed)
			}
			poll(true)
			assertTransition(true)
			poll(true)
			if events := takeEvents(); len(events) != 0 {
				t.Fatalf("unchanged restored polls notified: %+v", events)
			}
		})
	}
	test("owned", func(t *testing.T, ex *Executor, _ *engineState) {
		if err := ex.Start(context.Background(), "fake"); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := ex.Stop("fake"); err != nil {
				t.Errorf("cleanup: %v", err)
			}
		})
	})
	test("adopted", func(t *testing.T, ex *Executor, st *engineState) {
		var response atomic.Int32
		response.Store(http.StatusOK)
		st.port = detectionServer(t, &response)
		if got, err := ex.Status("fake"); err != nil || !got.Installed || !got.Running || !st.adopted {
			t.Fatalf("initial adoption = %+v, %v", got, err)
		}
	})
}

func TestConfiguredDetectionNotificationCombinesReadinessAndInstallation(t *testing.T) {
	ex, st := configuredDetectionExecutor(t)
	var response atomic.Int32
	response.Store(http.StatusOK)
	st.port = detectionServer(t, &response)
	if got, err := ex.Status("fake"); err != nil || !got.Installed || !got.Running {
		t.Fatalf("initial adoption = %+v, %v", got, err)
	}
	takeEvents := recordDetectionStates(t, ex)
	removeDetectionFile(t, filepath.Join(st.installDir, "config.json"))
	response.Store(http.StatusServiceUnavailable)
	if got, err := ex.Status("fake"); err != nil || got.Installed || got.Healthy || !got.Running {
		t.Fatalf("unhealthy incomplete adoption = %+v, %v", got, err)
	}
	if events := takeEvents(); len(events) != 1 || events[0].Installed || events[0].Healthy || !events[0].Running {
		t.Fatalf("combined transition notifications = %+v", events)
	}
	writeDetectionFile(t, filepath.Join(st.installDir, "config.json"))
	response.Store(http.StatusOK)
	listed := ex.GetInstalled()
	if len(listed) != 1 || !listed[0].Installed || !listed[0].Healthy || !listed[0].Running {
		t.Fatalf("restored adoption = %+v", listed)
	}
	if events := takeEvents(); len(events) != 1 || !events[0].Installed || !events[0].Healthy || !events[0].Running {
		t.Fatalf("combined restoration notifications = %+v", events)
	}
}

func TestDetectionServiceOnlyAdoptionPollsDoNotNotify(t *testing.T) {
	ex, st := configuredDetectionExecutor(t)
	st.plat.DetectRequired = nil
	st.plat.Detect = []string{filepath.Join(t.TempDir(), "absent-executable")}
	var response atomic.Int32
	response.Store(http.StatusOK)
	st.port = detectionServer(t, &response)
	if got, err := ex.Status("fake"); err != nil || !got.Installed || !got.Running || !st.adopted {
		t.Fatalf("service-only adoption = %+v, %v", got, err)
	}
	takeEvents := recordDetectionStates(t, ex)
	for range 3 {
		if got, err := ex.Status("fake"); err != nil || !got.Installed || !got.Running || !got.Healthy {
			t.Fatalf("service-only status = %+v, %v", got, err)
		}
		listed := ex.GetInstalled()
		if len(listed) != 1 || !listed[0].Installed || !listed[0].Running || !listed[0].Healthy {
			t.Fatalf("service-only get-installed = %+v", listed)
		}
	}
	if events := takeEvents(); len(events) != 0 {
		t.Fatalf("unchanged service-only adoption emitted notifications: %+v", events)
	}
}
