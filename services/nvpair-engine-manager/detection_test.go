// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func configuredDetectionExecutor(t *testing.T) (*Executor, *engineState) {
	t.Helper()
	m := testEngineManifest(fakeEngineBin)
	m.ManifestVersion = 2
	key := runtime.GOOS + "/" + runtime.GOARCH
	p := m.Platforms[key]
	p.DetectRequired = []string{"{install_dir}/server.py", "{install_dir}/config.json"}
	p.Runtime.Health = nil
	m.Platforms[key] = p
	if err := m.Validate(); err != nil {
		t.Fatal(err)
	}
	ex := newTestExecutor(t, m)
	st, err := ex.state(m.Engine)
	if err != nil {
		t.Fatal(err)
	}
	writeDetectionFile(t, filepath.Join(st.installDir, "server.py"))
	writeDetectionFile(t, filepath.Join(st.installDir, "config.json"))
	return ex, st
}

func writeDetectionFile(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("synthetic detection evidence"), 0o600); err != nil {
		t.Fatal(err)
	}
}

func removeDetectionFile(t *testing.T, path string) {
	t.Helper()
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
}

func detectionServer(t *testing.T, status *atomic.Int32) int {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(int(status.Load()))
	}))
	t.Cleanup(srv.Close)
	address, ok := srv.Listener.Addr().(*net.TCPAddr)
	if !ok {
		t.Fatalf("listener address = %T", srv.Listener.Addr())
	}
	return address.Port
}

func TestConfiguredDetectionRequiresEveryFile(t *testing.T) {
	test := func(name string, change func(*testing.T, *engineState), want bool) {
		t.Run(name, func(t *testing.T) {
			ex, st := configuredDetectionExecutor(t)
			change(t, st)
			installed, err := ex.Detect("fake")
			if err != nil || installed != want {
				t.Fatalf("Detect = %v, %v, want %v", installed, err, want)
			}
			if st.binPath != fakeEngineBin {
				t.Fatalf("binary = %q, want interpreter %q", st.binPath, fakeEngineBin)
			}
			if descriptor := st.manifest.descriptor(runtime.GOOS, runtime.GOARCH); !descriptor.Available {
				t.Fatal("installation completeness changed platform availability")
			}
		})
	}
	test("complete", func(*testing.T, *engineState) {}, true)
	test("server missing", func(t *testing.T, st *engineState) {
		removeDetectionFile(t, filepath.Join(st.installDir, "server.py"))
	}, false)
	test("config missing", func(t *testing.T, st *engineState) {
		removeDetectionFile(t, filepath.Join(st.installDir, "config.json"))
	}, false)
	test("directory is not evidence", func(t *testing.T, st *engineState) {
		path := filepath.Join(st.installDir, "config.json")
		removeDetectionFile(t, path)
		if err := os.Mkdir(path, 0o700); err != nil {
			t.Fatal(err)
		}
	}, false)
	test("unresolved directory placeholder", func(_ *testing.T, st *engineState) {
		st.plat.DetectRequired = []string{"{models_dir}/config.json"}
	}, false)
	test("blank rejected even without load validation", func(_ *testing.T, st *engineState) {
		st.plat.DetectRequired = []string{" "}
	}, false)
	test("empty requirement keeps executable detection", func(_ *testing.T, st *engineState) {
		st.plat.DetectRequired = []string{}
	}, true)
}

func TestConfiguredDetectionRequiresExecutableAlternative(t *testing.T) {
	for _, name := range []string{"absent", "directory", "blank"} {
		t.Run(name, func(t *testing.T) {
			ex, st := configuredDetectionExecutor(t)
			path := filepath.Join(t.TempDir(), "interpreter")
			if name == "directory" {
				if err := os.Mkdir(path, 0o700); err != nil {
					t.Fatal(err)
				}
			}
			if name == "blank" {
				path = " "
			}
			st.plat.Detect = []string{path}
			if installed, err := ex.Detect("fake"); err != nil || installed || st.binPath != "" {
				t.Fatalf("Detect = %v, %v, binary %q", installed, err, st.binPath)
			}
		})
	}
}

func TestConfiguredDetectionEnvironmentReferences(t *testing.T) {
	file := filepath.Join(t.TempDir(), "config.json")
	writeDetectionFile(t, file)
	test := func(name, value string, want bool) {
		t.Run(name, func(t *testing.T) {
			t.Setenv("PAIR_DETECTION_ROOT", value)
			template := "$PAIR_DETECTION_ROOT" + file
			if runtime.GOOS == "windows" {
				template = "%PAIR_DETECTION_ROOT%" + file
			}
			if got := detectionFile(template, nil); (got != "") != want {
				t.Fatalf("resolved = %q, want present %v", got, want)
			}
		})
	}
	// If an empty value were expanded permissively, the existing absolute file
	// suffix would be mistaken for the declared evidence.
	test("empty value fails closed", "", false)
	t.Run("unset value fails closed", func(t *testing.T) {
		t.Setenv("PAIR_DETECTION_UNSET", "restore after test")
		if err := os.Unsetenv("PAIR_DETECTION_UNSET"); err != nil {
			t.Fatal(err)
		}
		template := "$PAIR_DETECTION_UNSET" + file
		if runtime.GOOS == "windows" {
			template = "%PAIR_DETECTION_UNSET%" + file
		}
		if got := detectionFile(template, nil); got != "" {
			t.Fatalf("unset reference resolved to %q", got)
		}
	})
	t.Run("configured value", func(t *testing.T) {
		t.Setenv("PAIR_DETECTION_FILE", file)
		template := "${PAIR_DETECTION_FILE}"
		if runtime.GOOS == "windows" {
			template = "%PAIR_DETECTION_FILE%"
		}
		if got := detectionFile(template, nil); got != file {
			t.Fatalf("resolved = %q, want %q", got, file)
		}
	})
}

func TestConfiguredDetectionReadinessCannotCompleteInstallation(t *testing.T) {
	test := func(name, mode string) {
		t.Run(name, func(t *testing.T) {
			ex, st := configuredDetectionExecutor(t)
			removeDetectionFile(t, filepath.Join(st.installDir, "config.json"))
			var status atomic.Int32
			status.Store(http.StatusOK)
			st.port = detectionServer(t, &status)
			st.plat.Runtime.Mode = mode
			marker := filepath.Join(t.TempDir(), "installer-ran")
			st.plat.Install = &Install{Script: []string{fakeEngineBin, "touch", marker}}
			got, err := ex.Status("fake")
			if err != nil || got.Installed || !got.Running || !got.Healthy {
				t.Fatalf("ready but incomplete status = %+v, %v", got, err)
			}
			if err := ex.Start(context.Background(), "fake"); err == nil || !strings.Contains(err.Error(), "not installed") {
				t.Fatalf("start incomplete installation = %v", err)
			}
			if err := ex.Install(context.Background(), "fake"); err == nil || !strings.Contains(err.Error(), "occupied") {
				t.Fatalf("install over listener = %v", err)
			}
			if _, err := os.Stat(marker); !os.IsNotExist(err) {
				t.Fatalf("installer ran over occupied listener: %v", err)
			}
			if installedByPAIR(st.installDir) {
				t.Fatal("readiness claimed installation ownership")
			}
			listed := ex.GetInstalled()
			if len(listed) != 1 || listed[0].Installed || !listed[0].Running {
				t.Fatalf("get-installed = %+v", listed)
			}
		})
	}
	test("process listener", "process")
	test("command listener", "command")
}

func TestConfiguredDetectionAdoptedFileDisappearance(t *testing.T) {
	ex, st := configuredDetectionExecutor(t)
	var status atomic.Int32
	status.Store(http.StatusOK)
	st.port = detectionServer(t, &status)
	if got, err := ex.Status("fake"); err != nil || !got.Installed || !got.Running || !st.adopted {
		t.Fatalf("initial adoption = %+v, %v", got, err)
	}
	removeDetectionFile(t, filepath.Join(st.installDir, "config.json"))
	if got, err := ex.Status("fake"); err != nil || got.Installed || !got.Running || !got.Healthy || st.binPath != fakeEngineBin {
		t.Fatalf("adopted after file disappearance = %+v, %v, binary %q", got, err, st.binPath)
	}
	status.Store(http.StatusServiceUnavailable)
	if got, err := ex.Status("fake"); err != nil || got.Installed || !got.Running || got.Healthy {
		t.Fatalf("transient readiness failure = %+v, %v", got, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	presence := ex.reconcilePresence(ctx, "fake", st, false, st.port, false)
	if !presence.Occupied || st.installed || !st.running || st.binPath != fakeEngineBin {
		t.Fatalf("indeterminate listener = %+v, state %+v", presence, ex.snapshot("fake", st))
	}
}

func TestConfiguredDetectionOccupiedForeignListener(t *testing.T) {
	ex, st := configuredDetectionExecutor(t)
	removeDetectionFile(t, filepath.Join(st.installDir, "config.json"))
	var status atomic.Int32
	status.Store(http.StatusServiceUnavailable)
	st.port = detectionServer(t, &status)
	marker := filepath.Join(t.TempDir(), "installer-ran")
	st.plat.Install = &Install{Script: []string{fakeEngineBin, "touch", marker}}
	if err := ex.Install(context.Background(), "fake"); err == nil || !strings.Contains(err.Error(), "occupied") {
		t.Fatalf("install over foreign listener = %v", err)
	}
	if err := ex.Start(context.Background(), "fake"); err == nil {
		t.Fatal("incomplete installation started over foreign listener")
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("installer executed: %v", err)
	}
	if got, err := ex.Status("fake"); err != nil || got.Installed || got.Running {
		t.Fatalf("foreign listener status = %+v, %v", got, err)
	}
}

func TestConfiguredDetectionEvidenceDisappearsDuringReadiness(t *testing.T) {
	test := func(name string, action func(*testing.T, *Executor), owned bool) {
		t.Run(name, func(t *testing.T) {
			ex, st := configuredDetectionExecutor(t)
			if owned {
				t.Cleanup(func() {
					if err := ex.Stop("fake"); err != nil {
						t.Errorf("cleanup: %v", err)
					}
				})
			}
			if !owned {
				var status atomic.Int32
				status.Store(http.StatusOK)
				st.port = detectionServer(t, &status)
			}
			original := ex.client.Transport
			var once sync.Once
			ex.client.Transport = healthProbeTransport(func(request *http.Request) (*http.Response, error) {
				response, err := original.RoundTrip(request)
				if err == nil && response.StatusCode == http.StatusOK {
					once.Do(func() { removeDetectionFile(t, filepath.Join(st.installDir, "config.json")) })
				}
				return response, err
			})
			action(t, ex)
			if st.installed {
				t.Fatal("readiness hid disappearing mandatory evidence")
			}
			if owned && (st.proc != nil || st.running) {
				t.Fatal("failed owned readiness left a process running")
			}
		})
	}
	start := func(t *testing.T, ex *Executor) {
		t.Helper()
		if err := ex.Start(context.Background(), "fake"); err == nil || !strings.Contains(err.Error(), "not installed") {
			t.Fatalf("start with disappearing evidence = %v", err)
		}
	}
	test("status adoption", func(t *testing.T, ex *Executor) {
		if got, err := ex.Status("fake"); err != nil || got.Installed || !got.Running {
			t.Fatalf("status = %+v, %v", got, err)
		}
	}, false)
	test("start adoption", start, false)
	test("owned readiness cleans up", start, true)
}

func TestConfiguredDetectionInstallRequiresCompleteEvidence(t *testing.T) {
	test := func(name, created string, wantError bool) {
		t.Run(name, func(t *testing.T) {
			ex, st := configuredDetectionExecutor(t)
			config := filepath.Join(st.installDir, "config.json")
			removeDetectionFile(t, config)
			st.plat.Runtime.Ready = nil
			st.plat.Install = &Install{Script: []string{fakeEngineBin, "touch", filepath.Join(st.installDir, created)}}
			ex.detectTimeout = 0
			err := ex.Install(context.Background(), "fake")
			if (err != nil) != wantError {
				t.Fatalf("install = %v, want error %v", err, wantError)
			}
			wantInstalled := !wantError
			if installedByPAIR(st.installDir) != wantInstalled || st.installed != wantInstalled {
				t.Fatalf("installed = %v, marker = %v", st.installed, installedByPAIR(st.installDir))
			}
		})
	}
	test("installer supplies missing evidence", "config.json", false)
	test("successful command leaves evidence missing", "unrelated-file", true)
}

func TestConfiguredDetectionOwnedProcessRemainsStoppable(t *testing.T) {
	ex, st := configuredDetectionExecutor(t)
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ex.Stop("fake"); err != nil {
			t.Errorf("cleanup: %v", err)
		}
	})
	st.mu.Lock()
	proc := st.proc
	st.mu.Unlock()
	removeDetectionFile(t, filepath.Join(st.installDir, "config.json"))
	if got, err := ex.Status("fake"); err != nil || got.Installed || !got.Running || !got.Healthy {
		t.Fatalf("owned incomplete status = %+v, %v", got, err)
	}
	if err := ex.Start(context.Background(), "fake"); err == nil || !strings.Contains(err.Error(), "not installed") {
		t.Fatalf("already-running start bypassed required evidence: %v", err)
	}
	if err := ex.Stop("fake"); err != nil {
		t.Fatal(err)
	}
	select {
	case <-proc.done:
	default:
		t.Fatal("owned process survived stop")
	}
}

func managedDetectionExecutable(t *testing.T, st *engineState) string {
	t.Helper()
	bin := filepath.Join(st.installDir, "interpreter"+exeExt())
	copyFile(t, fakeEngineBin, bin)
	st.plat.Detect = []string{bin}
	st.plat.Runtime.Bin = bin
	st.modelsDir = filepath.Join(t.TempDir(), "models")
	st.plat.Uninstall = &Uninstall{Remove: []string{st.installDir}}
	if err := writeInstallMarker(st.installDir, "fake"); err != nil {
		t.Fatal(err)
	}
	return bin
}

func TestConfiguredDetectionUninstallRemovesIncompleteOwnedProcess(t *testing.T) {
	ex, st := configuredDetectionExecutor(t)
	bin := managedDetectionExecutable(t, st)
	if err := ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ex.Stop("fake"); err != nil {
			t.Errorf("cleanup: %v", err)
		}
	})
	removeDetectionFile(t, filepath.Join(st.installDir, "config.json"))
	if err := ex.Uninstall(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(bin); !os.IsNotExist(err) {
		t.Fatalf("managed executable survived: %v", err)
	}
	if got, err := ex.Status("fake"); err != nil || got.Installed || got.Running || installedByPAIR(st.installDir) {
		t.Fatalf("uninstall status = %+v, %v", got, err)
	}
}

func TestConfiguredDetectionUninstallRetainsSurvivingExecutable(t *testing.T) {
	test := func(name string, removeBefore bool) {
		t.Run(name, func(t *testing.T) {
			ex, st := configuredDetectionExecutor(t)
			bin := managedDetectionExecutable(t, st)
			config := filepath.Join(st.installDir, "config.json")
			if removeBefore {
				removeDetectionFile(t, config)
			}
			st.plat.Uninstall.Remove = []string{config}
			ex.detectTimeout = 0
			if err := ex.Uninstall(context.Background(), "fake"); err == nil || !strings.Contains(err.Error(), "still detected") {
				t.Fatalf("partial uninstall = %v", err)
			}
			if _, err := os.Stat(bin); err != nil {
				t.Fatalf("test must retain the executable: %v", err)
			}
			if !installedByPAIR(st.installDir) || st.installed {
				t.Fatalf("partial uninstall marker = %v, installed = %v", installedByPAIR(st.installDir), st.installed)
			}
		})
	}
	test("evidence missing before uninstall", true)
	test("uninstall deletes only mandatory evidence", false)
}

func TestConfiguredDetectionUninstallRefusesExternalExecutable(t *testing.T) {
	ex, st := configuredDetectionExecutor(t)
	removeDetectionFile(t, filepath.Join(st.installDir, "config.json"))
	st.plat.Uninstall = &Uninstall{Remove: []string{st.installDir}}
	if err := writeInstallMarker(st.installDir, "fake"); err != nil {
		t.Fatal(err)
	}
	if err := ex.Uninstall(context.Background(), "fake"); err == nil || !strings.Contains(err.Error(), "outside NVPAIR") {
		t.Fatalf("external incomplete uninstall = %v", err)
	}
	if !installedByPAIR(st.installDir) {
		t.Fatal("external executable refusal cleared ownership marker")
	}
}

func TestBundledDetectionAlternatives(t *testing.T) {
	reg := NewRegistry()
	if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
		t.Fatal(err)
	}
	for _, engine := range reg.Names() {
		m, ok := reg.Get(engine)
		if !ok {
			t.Fatalf("missing bundled manifest %s", engine)
		}
		for _, key := range sortedKeys(m.Platforms) {
			p := m.Platforms[key]
			t.Run(engine+"/"+key, func(t *testing.T) {
				if m.ManifestVersion != 1 || len(p.DetectRequired) != 0 || len(p.Detect) == 0 {
					t.Fatalf("bundled installation contract changed: version %d, detect %v, required %v", m.ManifestVersion, p.Detect, p.DetectRequired)
				}
				// Rebind every declared alternative to inert regular files. These
				// cases never inspect a real engine path on any host platform.
				alternatives := make([]string, len(p.Detect))
				for i := range alternatives {
					alternatives[i] = filepath.Join(t.TempDir(), fmt.Sprintf("alternative-%d", i))
				}
				p.Detect = alternatives
				copy := *m
				copy.Platforms = map[string]Platform{runtime.GOOS + "/" + runtime.GOARCH: p}
				ex := newTestExecutor(t, &copy)
				for i, path := range alternatives {
					t.Run(fmt.Sprintf("alternative %d alone", i+1), func(t *testing.T) {
						writeDetectionFile(t, path)
						if installed, err := ex.Detect(engine); err != nil || !installed {
							t.Fatalf("alternative alone = %v, %v", installed, err)
						}
						st, err := ex.state(engine)
						if err != nil {
							t.Fatal(err)
						}
						if st.binPath != path {
							t.Fatalf("selected binary = %q, want %q", st.binPath, path)
						}
						removeDetectionFile(t, path)
					})
				}
				for _, path := range alternatives {
					writeDetectionFile(t, path)
				}
				if installed, err := ex.Detect(engine); err != nil || !installed {
					t.Fatalf("all alternatives = %v, %v", installed, err)
				}
				st, err := ex.state(engine)
				if err != nil {
					t.Fatal(err)
				}
				if st.binPath != alternatives[0] {
					t.Fatalf("first match = %q, want %q", st.binPath, alternatives[0])
				}
			})
		}
	}
}

func TestBundledDetectionAdoptionCompatibility(t *testing.T) {
	reg := NewRegistry()
	if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
		t.Fatal(err)
	}
	for _, engine := range reg.Names() {
		m, ok := reg.Get(engine)
		if !ok {
			t.Fatalf("missing bundled manifest %s", engine)
		}
		for _, platform := range sortedKeys(m.Platforms) {
			t.Run(engine+"/"+platform, func(t *testing.T) {
				p := m.Platforms[platform]
				p.Detect = []string{filepath.Join(t.TempDir(), "absent-executable")}
				if p.Runtime.modeOrDefault() == "command" {
					// The existing command-mode contract requires its CLI even
					// when the API is already live. It is never executed here.
					writeDetectionFile(t, p.Detect[0])
				}
				p.Runtime.Health = nil
				copy := *m
				copy.Platforms = map[string]Platform{runtime.GOOS + "/" + runtime.GOARCH: p}
				ex := newTestExecutor(t, &copy)
				st, err := ex.state(engine)
				if err != nil {
					t.Fatal(err)
				}
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					if _, err := w.Write([]byte(`{"role":"router"}`)); err != nil {
						t.Errorf("write readiness: %v", err)
					}
				}))
				t.Cleanup(srv.Close)
				address, ok := srv.Listener.Addr().(*net.TCPAddr)
				if !ok {
					t.Fatalf("listener address = %T", srv.Listener.Addr())
				}
				st.port = address.Port
				if got, err := ex.Status(engine); err != nil || !got.Installed || !got.Running || !got.Healthy {
					t.Fatalf("bundled adoption = %+v, %v", got, err)
				}
				if err := ex.Start(context.Background(), engine); err != nil {
					t.Fatalf("adopted start = %v", err)
				}
				if err := ex.Install(context.Background(), engine); err != nil {
					t.Fatalf("adopted install = %v", err)
				}
				if st.proc != nil || installedByPAIR(st.installDir) {
					t.Fatal("adoption spawned a process or claimed installation")
				}
			})
		}
	}
}
