// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"

	"nvpair-shared/modelconfig"
)

func TestLlamaProjectorDisableLaunch(t *testing.T) {
	test := func(name, text, envName, envValue string) {
		t.Run(name, func(t *testing.T) {
			args, err := parseLaunchText(text)
			if err != nil {
				t.Fatal(err)
			}
			env := map[string]string{}
			if envName != "" {
				env, err = literalEnvironment(args)
				if err != nil {
					t.Fatal(err)
				}
				args = nil
			}
			command := launchCommand{Args: args, Env: env}
			cli, nativeEnv, err := llamaLaunchOptions(command, nil)
			if err != nil {
				t.Fatalf("disable projector: %v", err)
			}
			if envName == "" {
				if cli["mmproj-auto"] != "false" {
					t.Fatalf("CLI projector selection=%q, want false", cli["mmproj-auto"])
				}
			} else {
				want := envValue
				if envName == "LLAMA_ARG_NO_MMPROJ_AUTO" {
					want = "false"
				}
				if nativeEnv["mmproj-auto"] != want {
					t.Fatalf("environment projector selection=%q, want %q", nativeEnv["mmproj-auto"], want)
				}
			}
			child := llamaSnapshotCommand(command, "snapshot.ini")
			if !slices.Equal(child.Args, append(slices.Clone(args), "--log-verbosity", "4", "--models-preset", "snapshot.ini")) {
				t.Fatalf("native arguments changed: %v", child.Args)
			}
			if envName != "" {
				if value, exists := child.Env[envName]; !exists || value != envValue {
					t.Fatalf("native environment %s=%q, exists=%v", envName, value, exists)
				}
			}
		})
	}
	test("no projector flag", "--no-mmproj", "", "")
	test("no automatic projector flag", "--no-mmproj-auto", "", "")
	test("negative environment presence", "LLAMA_ARG_NO_MMPROJ_AUTO=", "LLAMA_ARG_NO_MMPROJ_AUTO", "")
	test("negative environment ignores value", "LLAMA_ARG_NO_MMPROJ_AUTO=false", "LLAMA_ARG_NO_MMPROJ_AUTO", "false")
	test("positive environment false", "LLAMA_ARG_MMPROJ_AUTO=false", "LLAMA_ARG_MMPROJ_AUTO", "false")
	test("positive environment zero", "LLAMA_ARG_MMPROJ_AUTO=0", "LLAMA_ARG_MMPROJ_AUTO", "0")
	test("positive environment off", "LLAMA_ARG_MMPROJ_AUTO=off", "LLAMA_ARG_MMPROJ_AUTO", "off")
}

func TestLlamaProjectorDisablePreset(t *testing.T) {
	test := func(name, key, value, want string) {
		t.Run(name, func(t *testing.T) {
			sections, err := parseLlamaServingPreset([]byte("[*]\n" + key + " = " + value + "\n"))
			if err != nil {
				t.Fatal(err)
			}
			if got := sections["*"]["mmproj-auto"]; got != want {
				t.Fatalf("projector selection=%q, want %q", got, want)
			}
		})
	}
	test("no projector", "no-mmproj", "true", "false")
	test("no automatic projector", "no-mmproj-auto", "true", "false")
	test("negative enabled uses native truthiness", "no-mmproj", "enabled", "false")
	test("canonical false", "mmproj-auto", "false", "false")
	test("canonical zero", "mmproj-auto", "0", "0")
	test("canonical off", "mmproj-auto", "off", "off")
}

func TestLlamaProjectorEnableRefused(t *testing.T) {
	t.Run("launch", func(t *testing.T) {
		for _, text := range []string{
			"--mmproj-auto", "--mmproj projector.gguf", "--mmproj-url https://example.invalid/projector.gguf",
			"LLAMA_ARG_MMPROJ_AUTO=true", "LLAMA_ARG_MMPROJ_AUTO=1", "LLAMA_ARG_MMPROJ_AUTO=on", "LLAMA_ARG_MMPROJ_AUTO=enabled",
			"LLAMA_ARG_MMPROJ=projector.gguf", "LLAMA_ARG_MMPROJ_URL=https://example.invalid/projector.gguf",
		} {
			t.Run(text, func(t *testing.T) {
				args, err := parseLaunchText(text)
				if err != nil {
					t.Fatal(err)
				}
				env := map[string]string{}
				if strings.HasPrefix(text, "LLAMA_ARG_") {
					env, err = literalEnvironment(args)
					if err != nil {
						t.Fatal(err)
					}
					args = nil
				}
				if _, _, err := llamaLaunchOptions(launchCommand{Args: args, Env: env}, nil); err == nil {
					t.Fatal("projector enabling launch accepted")
				}
			})
		}
	})
	t.Run("preset", func(t *testing.T) {
		for _, option := range []string{"mmproj-auto = true", "mmproj-auto = on", "mmproj-auto = 1", "mmproj-auto = enabled", "no-mmproj = false", "no-mmproj-auto = off", "no-mmproj = native-value", "mmproj = projector.gguf", "mmproj-url = https://example.invalid/projector.gguf"} {
			t.Run(option, func(t *testing.T) {
				if _, err := parseLlamaServingPreset([]byte("[*]\n" + option + "\n")); err == nil {
					t.Fatal("projector enabling preset accepted")
				}
			})
		}
	})
	t.Run("typed extras", func(t *testing.T) {
		test := func(key, value string) {
			t.Run(key+" "+value, func(t *testing.T) {
				if err := validateModelExtraOptions("model", map[string]string{key: value}); err == nil {
					t.Fatal("projector enabling extra accepted")
				}
			})
		}
		test("mmproj-auto", "true")
		test("mmproj-auto", "on")
		test("mmproj-auto", "1")
		test("mmproj-auto", "enabled")
		test("mmproj", "projector.gguf")
		test("mmproj-url", "https://example.invalid/projector.gguf")
	})
}

func TestTypedLlamaProjectorDisableExtra(t *testing.T) {
	for _, value := range []string{"false", "0", "off"} {
		t.Run(value, func(t *testing.T) {
			request := modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "cpu"}, ExtraOptions: map[string]string{"mmproj-auto": value}}
			layers := int32(4)
			options, _, err := typedLlamaModelOptions("model", request, &layers)
			if err != nil {
				t.Fatal(err)
			}
			data, err := typedLlamaPreset(map[string]map[string]string{"model": options})
			if err != nil {
				t.Fatal(err)
			}
			sections, err := parseLlamaServingPreset(data)
			if err != nil {
				t.Fatal(err)
			}
			if sections["model"]["mmproj-auto"] != value {
				t.Fatalf("native preset changed extra: %v", sections["model"])
			}
		})
	}
}

func TestModelLoadConversionWildcardProjectorDisable(t *testing.T) {
	f := newModelLoadFixture(t)
	data, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	data = []byte(strings.Replace(string(data), "[*]\n", "[*]\nno-mmproj = true\n", 1))
	writeTestFile(t, f.source, data)
	conversion, err := f.ex.previewTypedLlamaConversion(context.Background(), f.st, f.command, f.rt)
	if err != nil {
		t.Fatalf("whole-preset conversion review: %v", err)
	}
	if len(conversion.Runtime.ModelLoads) != 2 {
		t.Fatalf("conversion lost profiles: %+v", conversion.Runtime.ModelLoads)
	}
	for _, load := range conversion.Runtime.ModelLoads {
		if !reflect.DeepEqual(load.Requested.ExtraOptions, map[string]string{"mmproj-auto": "false"}) {
			t.Fatalf("profile %s lost wildcard disable: %v", load.ProfileKey, load.Requested.ExtraOptions)
		}
	}
	prepared, err := compilePreparedServing(f.st, typedFixtureCommand(t, f, conversion.Runtime), conversion.Runtime, false)
	if err != nil {
		t.Fatal(err)
	}
	sections, err := parseLlamaServingPreset(prepared.Snapshot)
	if err != nil {
		t.Fatal(err)
	}
	for _, profile := range []string{"weak-model", "medium-model"} {
		if sections[profile]["mmproj-auto"] != "false" {
			t.Fatalf("native snapshot lost %s projector disable: %v", profile, sections[profile])
		}
	}
	unchanged, err := os.ReadFile(f.source)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(data, unchanged) || f.st.running {
		t.Fatal("conversion review changed the source or started the engine")
	}
}
