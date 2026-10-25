// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Render the legacy full command only for migration and argv regression tests.
func (command launchCommand) text() (string, error) {
	return formatLaunchParts(command.Env, append([]string{command.Bin}, command.Args...))
}

func TestResolvedLaunchMatchesBundledEngines(t *testing.T) {
	reg := loadWithOverrides(t, t.TempDir())
	vars := map[string]string{"host": "127.0.0.1", "port": "12345", "cli": "/test path/lms", "install_dir": "/test path", "models_dir": "/test models"}
	for _, engine := range []string{"ollama", "lmstudio", "llamacpp"} {
		manifest, ok := reg.Get(engine)
		if !ok {
			t.Fatalf("missing bundled engine %q", engine)
		}
		// Exercise every platform even when running on Windows or macOS, so
		// Linux-only launch environment requirements are covered locally too.
		for platformKey, platform := range manifest.Platforms {
			t.Run(engine+"/"+platformKey, func(t *testing.T) {
				var launch launchCommand
				var err error
				var want []string
				if engine == "ollama" {
					launch, err = resolveProcessLaunch(platform.Runtime, "/test path/ollama", vars)
					want = []string{"OLLAMA_HOST=127.0.0.1:12345", "/test path/ollama", "serve"}
					if strings.HasPrefix(platformKey, "linux/") {
						want = append([]string{"LD_LIBRARY_PATH=/test path/lib/ollama"}, want...)
					}
				} else if engine == "lmstudio" {
					launch, err = resolveCommandLaunch(platform.Runtime.Start[0], vars)
					want = []string{"/test path/lms", "server", "start", "--port", "12345", "--bind", "127.0.0.1"}
				} else {
					var bin string
					bin, err = resolvePlaceholders(platform.Runtime.Bin, vars)
					if err == nil {
						launch, err = resolveProcessLaunch(platform.Runtime, bin, vars)
					}
					want = []string{"LLAMA_CACHE=/test models", bin, "--sleep-idle-seconds", "300", "--ctx-size", "8192", "--parallel", "1", "--cache-ram", "512", "--host", "127.0.0.1", "--port", "12345", "--cors-origins", ""}
					if strings.HasPrefix(platformKey, "linux/") {
						want = append([]string{"LD_LIBRARY_PATH=/test path"}, want...)
					}
				}
				if err != nil {
					t.Fatal(err)
				}
				text, err := launch.text()
				if err != nil {
					t.Fatal(err)
				}
				got, err := parseLaunchText(text)
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Fatalf("launch description differs from executable inputs: got %#v, want %#v, error %v", got, want, err)
				}
			})
		}
	}
	if _, changed := vars["bin"]; changed {
		t.Fatal("process builder mutated caller's resolution context")
	}
}

func TestLaunchEnvironmentFormattingIsDeterministic(t *testing.T) {
	launch := launchCommand{Bin: "engine", Env: map[string]string{"Z_SETTING": "z", "A_SETTING": "a b"}}
	text, err := launch.text()
	if err != nil || text != `A_SETTING="a b" Z_SETTING="z" engine` {
		t.Fatalf("unstable environment order or quoting: %q, %v", text, err)
	}
}

func TestCustomLlamaPresetFixedArgs(t *testing.T) {
	reg := loadWithOverrides(t, t.TempDir())
	manifest, ok := reg.Get("llamacpp")
	if !ok {
		t.Fatal("missing bundled llama.cpp manifest")
	}
	rt := manifest.Platforms["linux/amd64"].Runtime
	fixed := slices.Clone(rt.EditableLaunch.FixedArgs)
	managed := []string{"--host", "127.0.0.1", "--port", "23456"}
	vars := map[string]string{"host": "127.0.0.1", "port": "23456", "models_dir": "/test models", "install_dir": "/test engine"}
	test := func(name string, args, extraFixed, wantFixed []string) {
		t.Run(name, func(t *testing.T) {
			runtime := rt
			policy := *rt.EditableLaunch
			policy.FixedArgs = append(slices.Clone(fixed), extraFixed...)
			runtime.EditableLaunch = &policy
			runtime.LaunchArgs = &args
			launch, err := resolveProcessLaunch(runtime, "/test engine/llama-server", vars)
			if err != nil {
				t.Fatal(err)
			}
			want := append(slices.Clone(wantFixed), managed...)
			want = append(want, args...)
			if !slices.Equal(launch.Args, want) {
				t.Fatalf("argv = %q, want %q", launch.Args, want)
			}
			if err := validateEffectiveLaunch(runtime, launch, vars["host"], vars["port"]); err != nil {
				t.Fatalf("effective launch rejected: %v", err)
			}
			text, err := launch.argumentText(runtime)
			if err != nil {
				t.Fatal(err)
			}
			parsed, err := parseLaunchText(text)
			if err != nil {
				t.Fatal(err)
			}
			// Explicit environment assignments precede the editable arguments.
			for len(parsed) > 0 && !strings.HasPrefix(parsed[0], "-") {
				parsed = parsed[1:]
			}
			if !slices.Equal(parsed, want[len(wantFixed):]) {
				t.Fatalf("editable argv = %q, want %q", parsed, want[len(wantFixed):])
			}
		})
	}
	test("preset owns model defaults", []string{"--models-preset", "/test models/preset.ini"}, nil, nil)
	test("equals preset", []string{"--models-preset=/test models/preset.ini"}, nil, nil)
	test("native underscore preset", []string{"--models_preset", "/test models/preset.ini"}, nil, nil)
	test("user context", []string{"--models-preset", "preset.ini", "--ctx-size", "131072"}, nil, nil)
	test("user equals context", []string{"--models-preset=preset.ini", "--ctx-size=32768"}, nil, nil)
	test("user short context", []string{"--models-preset", "preset.ini", "-c", "32768"}, nil, nil)
	test("non-preset defaults unchanged", []string{"--model", "model.gguf"}, nil, fixed)
	test("non-preset explicit context unchanged", []string{"--ctx-size", "32768"}, nil, fixed)
	test("preset text in option value is literal", []string{"--alias", "--models-preset"}, nil, fixed)
	test("preset text in log filename is literal", []string{"--log-file", "--models-preset"}, nil, fixed)
	test("equals preset text in log filename is literal", []string{"--log-file", "--models-preset=preset.ini"}, nil, fixed)
	test("model option text in fixed filename is literal", []string{"--models-preset", "preset.ini"}, []string{"--log-file", "--ctx-size"}, []string{"--log-file", "--ctx-size"})
	test("switch before preset", []string{"--log-disable", "--models-preset", "preset.ini"}, nil, nil)
	test("two-value option before preset", []string{"--control-vector-layer-range", "1", "32", "--models-preset", "preset.ini"}, nil, nil)
	test("router fixed args retained", []string{"--models-preset", "preset.ini"}, []string{"--models-max", "2", "--log-disable"}, []string{"--models-max", "2", "--log-disable"})
	test("user router option wins", []string{"--models-preset", "preset.ini", "--models-max=3"}, []string{"--models-max", "2", "--log-disable"}, []string{"--log-disable"})
	test("user non-envelope option wins", []string{"--models-preset", "preset.ini", "--log-file", "user.log"}, []string{"--log-file", "bundled.log"}, nil)
	test("preset in fixed args", []string{"--ctx-size", "32768"}, []string{"--models-preset", "preset.ini"}, []string{"--models-preset", "preset.ini"})
	if !slices.Equal(rt.EditableLaunch.FixedArgs, fixed) {
		t.Fatal("custom launch mutated manifest defaults")
	}
}

func TestCommandLaunchRejectsEmptyExecutable(t *testing.T) {
	for _, template := range [][]string{{"", "start"}, {"{cli}", "start"}} {
		if _, err := resolveCommandLaunch(template, map[string]string{"cli": ""}); err == nil {
			t.Fatal("empty executable must fail rather than skip the start command")
		}
	}
}

// The child receives the parser's output directly. This catches escaping
// differences in the real Windows/POSIX argv path, not just parser symmetry.
func TestLaunchTextReachesChildLiterally(t *testing.T) {
	path := filepath.Join(t.TempDir(), "captured args.json")
	want := []string{"", "two words", "日本語", `C:\new\tools\`, `\\server\share`, `a"b`, "$HOME", "%USERPROFILE%", "{port}", "$(ignored)", "line\nbreak"}
	tokens := append([]string{fakeEngineBin, "captureargs", path}, want...)
	text, err := formatLaunchText(tokens)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := parseLaunchText(text)
	if err != nil {
		t.Fatal(err)
	}
	proc, err := startManagedProc(parsed[0], parsed[1:], nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { proc.stop(0) })
	select {
	case <-proc.done:
	case <-time.After(10 * time.Second):
		t.Fatal("argument-capture child did not exit")
	}
	assertCapturedArgs(t, path, want)
}

func TestLifecycleUsesResolvedLaunchBuilder(t *testing.T) {
	for _, mode := range []string{"process", "command"} {
		t.Run(mode, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "captured args.json")
			port, err := freePort()
			if err != nil {
				t.Fatal(err)
			}
			manifest := testEngineManifest(fakeEngineBin)
			for key, platform := range manifest.Platforms {
				platform.Runtime = Runtime{Mode: mode, Bin: fakeEngineBin, Port: port,
					Args:  []string{"{host}", "{port}", "two words", ""},
					Start: [][]string{{fakeEngineBin, "captureargs", path, "{host}", "{port}", "two words", ""}},
				}
				if mode == "process" {
					// Keep the process alive through readiness and durable PID save.
					// The normal server records argv before opening its listener.
					platform.Runtime.Env = map[string]string{"FAKE_START_LOG": path, "OLLAMA_HOST": "127.0.0.1:{port}"}
					platform.Runtime.Ready = &Probe{TCP: "127.0.0.1:{port}"}
				}
				manifest.Platforms[key] = platform
			}
			ex := newTestExecutor(t, manifest)
			if err := ex.Start(context.Background(), "fake"); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = ex.Stop("fake") })
			assertCapturedArgs(t, path, []string{"127.0.0.1", strconv.Itoa(port), "two words", ""})
		})
	}
}

func assertCapturedArgs(t *testing.T, path string, want []string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("child received %#v; want %#v", got, want)
	}
}
