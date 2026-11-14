// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"nvpair-shared/modelconfig"
)

func init() {
	if len(os.Args) == 2 && os.Args[1] == "--help" && os.Getenv("PAIR_NATIVE_HELP_SPAWN_MARKER") != "" {
		file, err := os.OpenFile(os.Getenv("PAIR_NATIVE_HELP_SPAWN_MARKER"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
		if err != nil {
			os.Exit(2)
		}
		if _, err := file.WriteString("spawned\n"); err != nil {
			os.Exit(2)
		}
		if err := file.Close(); err != nil {
			os.Exit(2)
		}
		if os.Getenv("PAIR_NATIVE_HELP_FAILURE") == "1" {
			os.Exit(2)
		}
		data, err := os.ReadFile(os.Getenv("PAIR_NATIVE_HELP_FILE"))
		if err != nil {
			os.Exit(2)
		}
		if _, err := os.Stdout.Write(data); err != nil {
			os.Exit(2)
		}
		os.Exit(0)
	}
	if len(os.Args) != 3 || os.Args[1] != "native-help-test-child" {
		return
	}
	switch os.Args[2] {
	case "timeout":
		time.Sleep(30 * time.Second)
	case "oversize":
		if _, err := os.Stdout.WriteString(strings.Repeat("x", nativeHelpBytes+1)); err != nil {
			os.Exit(2)
		}
	case "stderr oversize":
		if _, err := os.Stderr.WriteString(strings.Repeat("x", nativeHelpBytes+1)); err != nil {
			os.Exit(2)
		}
	case "failure":
		os.Exit(2)
	default:
		if _, err := os.Stdout.WriteString("--spec-type {none,draft-mtp,ngram-mod}  speculative type\n--cache-prompt, --no-cache-prompt  prompt cache\n--moe-cache-mib N  cache size\n"); err != nil {
			os.Exit(2)
		}
	}
	os.Exit(0)
}

const nativeOptionsFixtureHelp = "--spec-type {none,draft-mtp,ngram-mod}  speculative type\n--cache-prompt, --no-cache-prompt  prompt cache\n--moe-cache-mib N  cache size\n"

// Alias spacing and value forms from llama.cpp 0ec4027a2 common/arg.cpp.
const nativeHelpPaddedAliases = `-dev,  --device <dev1,dev2,..>  devices to use
-sm,   --split-mode {none,layer,row,tensor}  split mode
-ts,   --tensor-split N0,N1,N2,...  split proportions
-mg,   --main-gpu INDEX  main GPU
-m,    --model FNAME  model file, with --example only in prose
`

// Keep the existing package executable, including Windows' fixed-path wrapper.
// A uniquely named inert library gives each fixture its own content identity;
// the pure-Go --help child never loads it or opens a listener.
func nativeOptionsFixture(t *testing.T, e *Executor, st *engineState) (string, string) {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	library, err := os.CreateTemp(filepath.Dir(executable), "libggml-native-help-*.so")
	if err != nil {
		t.Fatal(err)
	}
	path := library.Name()
	if _, err := library.WriteString(t.Name() + path); err != nil {
		_ = library.Close()
		t.Fatal(err)
	}
	if err := library.Close(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Remove(path); err != nil {
			t.Error(err)
		}
	})
	marker := filepath.Join(t.TempDir(), "native-help-spawns")
	t.Setenv("PAIR_NATIVE_HELP_SPAWN_MARKER", marker)
	helpFile := filepath.Join(t.TempDir(), "native-help.txt")
	writeTestFile(t, helpFile, []byte(nativeOptionsFixtureHelp))
	t.Setenv("PAIR_NATIVE_HELP_FILE", helpFile)
	t.Setenv("PAIR_NATIVE_HELP_FAILURE", "")
	st.binPath, st.plat.Runtime.Bin = executable, executable
	st.plat.Detect, st.plat.DetectRequired = []string{executable}, nil
	e.nodeProjection.set(nodeProjection{})
	return marker, path
}

func requireNativeHelpSpawns(t *testing.T, marker string, want int) {
	t.Helper()
	data, err := os.ReadFile(marker)
	if os.IsNotExist(err) && want == 0 {
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(string(data), "spawned\n"); got != want {
		t.Fatalf("native help spawns = %d, want %d", got, want)
	}
}

func TestNativeOptionsSelectedBuildCache(t *testing.T) {
	e, st := new(Executor), nativeBuildTestState(t)
	marker, library := nativeOptionsFixture(t, e, st)
	want := []modelconfig.NativeOption{
		{Name: "cache-prompt", Kind: "switch"},
		{Name: "moe-cache-mib", Kind: "number"},
		{Name: "no-cache-prompt", Kind: "switch"},
		{Name: "spec-type", Kind: "choice", Choices: []string{"none", "draft-mtp", "ngram-mod"}},
	}
	check := func() {
		t.Helper()
		got, known := e.nativeOptions(context.Background(), st)
		if !known || !reflect.DeepEqual(got, want) {
			t.Fatalf("selected native options = %+v, known=%v", got, known)
		}
	}
	check()
	check()
	requireNativeHelpSpawns(t, marker, 1)
	info, err := os.Stat(st.binPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chtimes(st.binPath, info.ModTime(), info.ModTime()); err != nil {
			t.Error(err)
		}
	})
	touch := info.ModTime().Add(time.Hour)
	if err := os.Chtimes(st.binPath, touch, touch); err != nil {
		t.Fatal(err)
	}
	check()
	requireNativeHelpSpawns(t, marker, 1)
	data, err := os.ReadFile(library)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, library, data)
	check()
	requireNativeHelpSpawns(t, marker, 1)
	writeTestFile(t, library, append(data, []byte("changed native library")...))
	check()
	requireNativeHelpSpawns(t, marker, 2)
	// A populated cache still cannot publish hints or run a child while paused.
	e.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	if got, known := e.nativeOptions(context.Background(), st); known || got != nil {
		t.Fatalf("paused cached options = %+v, known=%v", got, known)
	}
	requireNativeHelpSpawns(t, marker, 2)
}

func TestNativeOptionsLaunchWarningsCoverDefaultsAndEverySection(t *testing.T) {
	preset := []byte("[*]\nfit = off\nkv-unified = false\n[first]\nmodel = one.gguf\nctx-size = 4096\n[second]\nmodel = two.gguf\ncache-ram = 512\nspec-type = draft-mtp\n")
	options := []modelconfig.NativeOption{{Name: "spec-type", Kind: "choice", Choices: []string{"none", "draft-mtp"}}}
	command := launchCommand{Args: []string{"--cors-origins=*", "--spec-type=draft-mtp"}}
	got, err := unreportedNativeLaunchOptions(options, command, preset, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"cache-ram", "cors-origins", "ctx-size", "fit", "kv-unified", "log-verbosity", "model", "models-preset"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("launch warnings = %v, want %v", got, want)
	}
	if got, err := unreportedNativeLaunchOptions(nil, command, preset, nil); err != nil || got != nil {
		t.Fatalf("unknown help emitted warnings = %v, err=%v", got, err)
	}
}

func TestNativeHelpHints(t *testing.T) {
	got := parseNativeHelp(`usage: llama-server
  --spec-type {none,draft-mtp,ngram-mod}  speculative type
  --cache-prompt, --no-cache-prompt  prompt cache
  --moe-cache-mib N  cache size
  -m, --model FNAME  model file
  -ngl, --gpu-layers, --n-gpu-layers N  GPU layers
  --ui-config, --webui-config JSON  UI defaults
  --spec-draft-threads, -td, --threads-draft N  draft threads
  --color auto|on|off  color mode
  --verbose  verbose output with --example in prose
  prose mentions --ignored
`)
	want := []modelconfig.NativeOption{
		{Name: "cache-prompt", Kind: "switch"},
		{Name: "color", Kind: "choice", Choices: []string{"auto", "on", "off"}},
		{Name: "gpu-layers", Kind: "number"},
		{Name: "model", Kind: "text"},
		{Name: "moe-cache-mib", Kind: "number"},
		{Name: "n-gpu-layers", Kind: "number"},
		{Name: "no-cache-prompt", Kind: "switch"},
		{Name: "spec-draft-threads", Kind: "number"},
		{Name: "spec-type", Kind: "choice", Choices: []string{"none", "draft-mtp", "ngram-mod"}},
		{Name: "threads-draft", Kind: "number"},
		{Name: "ui-config", Kind: "text"},
		{Name: "verbose", Kind: "switch"},
		{Name: "webui-config", Kind: "text"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("hints = %#v, want %#v", got, want)
	}
	if parseNativeHelp("unrecognized output") != nil {
		t.Fatal("unrecognized output was known")
	}
}

func TestNativeHelpPaddedAliasHints(t *testing.T) {
	got := parseNativeHelp(nativeHelpPaddedAliases)
	want := []modelconfig.NativeOption{
		{Name: "device", Kind: "text"},
		{Name: "main-gpu", Kind: "text"},
		{Name: "model", Kind: "text"},
		{Name: "split-mode", Kind: "choice", Choices: []string{"none", "layer", "row", "tensor"}},
		{Name: "tensor-split", Kind: "text"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("padded alias hints = %#v, want %#v", got, want)
	}
	warnings, err := unreportedNativeLaunchOptions(got, launchCommand{Args: []string{"--device=CPU", "--split-mode=layer", "--tensor-split=1,1", "--main-gpu=0"}}, nil, nil)
	if err != nil || len(warnings) != 0 {
		t.Fatalf("reported padded aliases produced warnings = %v, err=%v", warnings, err)
	}
}

func TestNativeHelpWrappedProse(t *testing.T) {
	// Pinned formatter uses column 40 for help continuations. Removed draft
	// aliases mention these live numeric flags on wrapped lines afterwards.
	help := "--spec-ngram-mod-n-min N                  minimum\n" +
		"--spec-ngram-mod-n-max N                  maximum\n" +
		"--draft, --draft-n, --draft-max N         the argument has been removed. use --spec-draft-n-max or\n" +
		strings.Repeat(" ", 40) + "--spec-ngram-mod-n-max\n" +
		strings.Repeat(" ", 40) + "--spec-ngram-mod-n-min\n" +
		strings.Repeat(" ", 40) + "--spec-ngram-*-size-n or --spec-ngram-mod-n-match\n" +
		"--spec-ngram-*-size-n or --spec-ngram-mod-n-match\n"
	want := []modelconfig.NativeOption{
		{Name: "draft", Kind: "number"},
		{Name: "draft-max", Kind: "number"},
		{Name: "draft-n", Kind: "number"},
		{Name: "spec-ngram-mod-n-max", Kind: "number"},
		{Name: "spec-ngram-mod-n-min", Kind: "number"},
	}
	if got := parseNativeHelp(help); !reflect.DeepEqual(got, want) {
		t.Fatalf("wrapped prose changed hints = %#v, want %#v", got, want)
	}
	if got := parseNativeHelp(strings.Repeat(" ", 40) + "--only-a-prose-reference\n"); got != nil {
		t.Fatalf("prose-only help became known: %#v", got)
	}
}

func TestNativeHelpCommandBounds(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"valid", "timeout", "oversize", "stderr oversize", "failure"} {
		t.Run(mode, func(t *testing.T) {
			start := time.Now()
			output, err := runNativeHelpCommand(context.Background(), executable, []string{"native-help-test-child", mode}, os.Environ())
			if mode == "valid" {
				if err != nil {
					t.Fatal(err)
				}
				got := parseNativeHelp(output)
				if len(got) != 4 || got[2].Name != "no-cache-prompt" || got[3].Kind != "choice" {
					t.Fatalf("help = %+v", got)
				}
			} else if err == nil || output != "" {
				t.Fatalf("bounded failure: output bytes=%d err=%v", len(output), err)
			}
			if time.Since(start) > nativeHelpBudget+2*time.Second {
				t.Fatal("native help exceeded wall clock ceiling and scheduling margin")
			}
		})
	}
}

func TestNativeOptionsCacheContentIdentity(t *testing.T) {
	st := nativeBuildTestState(t)
	var cache nativeHelpCache
	reads := 0
	read := func() []modelconfig.NativeOption {
		reads++
		return []modelconfig.NativeOption{{Name: "spec-type", Kind: "choice", Choices: []string{"none", "draft-mtp"}}}
	}
	first := requireNativeBuildID(t, st)
	got := cache.get(context.Background(), first, read)
	got[0].Choices[0] = "mutated"
	if again := cache.get(context.Background(), first, read); reads != 1 || again[0].Choices[0] != "none" {
		t.Fatalf("cache reads=%d options=%+v", reads, again)
	}
	info, err := os.Stat(st.binPath)
	if err != nil {
		t.Fatal(err)
	}
	touch := info.ModTime().Add(time.Hour)
	if err := os.Chtimes(st.binPath, touch, touch); err != nil {
		t.Fatal(err)
	}
	touched := requireNativeBuildID(t, st)
	cache.get(context.Background(), touched, read)
	if touched != first || reads != 1 {
		t.Fatalf("mtime invalidated cache: reads=%d", reads)
	}
	writeTestFile(t, filepath.Join(st.installDir, "libggml-base.dylib"), []byte("changed library bytes"))
	changed := requireNativeBuildID(t, st)
	cache.get(context.Background(), changed, read)
	if changed == first || reads != 2 {
		t.Fatalf("bytes did not refresh cache: reads=%d", reads)
	}
	writeTestFile(t, st.binPath, []byte("changed executable bytes"))
	cache.get(context.Background(), requireNativeBuildID(t, st), read)
	if reads != 3 {
		t.Fatalf("executable bytes did not refresh cache: reads=%d", reads)
	}
}

func TestNativeOptionsCacheRemembersFailure(t *testing.T) {
	var cache nativeHelpCache
	reads := 0
	read := func() []modelconfig.NativeOption { reads++; return nil }
	for range 2 {
		if got := cache.get(context.Background(), "failed-build", read); got != nil {
			t.Fatal("failed build known")
		}
	}
	if reads != 1 {
		t.Fatalf("failed build ran %d times", reads)
	}
}

func TestNativeOptionsPausedNoSpawn(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "spawned")
	t.Setenv("PAIR_NATIVE_HELP_SPAWN_MARKER", marker)
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	e := new(Executor)
	e.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	st := nativeBuildTestState(t)
	st.binPath = executable
	if got, known := e.nativeOptions(context.Background(), st); known || got != nil {
		t.Fatalf("paused options = %+v, known=%v", got, known)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("paused helper spawned: stat error=%v", err)
	}
	// Exercise the actual spawn boundary independently of runtime resolution.
	ctx := e.activationDispatchContext(context.Background(), "llamacpp")
	if output, err := runNativeHelpCommand(ctx, executable, []string{"--help"}, os.Environ()); err == nil || output != "" {
		t.Fatalf("paused command boundary: output=%q err=%v", output, err)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("paused command boundary spawned: stat error=%v", err)
	}
}
