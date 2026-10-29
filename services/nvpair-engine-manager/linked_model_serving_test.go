// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"nvpair-shared/modelconfig"
)

func TestLinkedModelServingBinding(t *testing.T) {
	t.Run("unchanged noncanonical preset paths", func(t *testing.T) {
		for _, mode := range []string{"command", "environment"} {
			t.Run(mode, func(t *testing.T) {
				f := newLlamaServingFixture(t)
				applyLlamaServing(t, f, approveLlamaServing(t, f))
				st := settingsState(t, f.ex)
				command, err := launchForState(st, st.port)
				if err != nil {
					t.Fatal(err)
				}
				index := slices.Index(command.Args, "--models-preset")
				if index < 0 || index+1 >= len(command.Args) {
					t.Fatal("preset argument missing")
				}
				path := filepath.Dir(f.source) + string(os.PathSeparator) + "." + string(os.PathSeparator) + filepath.Base(f.source)
				if mode == "command" {
					command.Args[index+1] = path
				} else {
					command.Args = append(command.Args[:index:index], command.Args[index+2:]...)
					command.Env["LLAMA_ARG_MODELS_PRESET"] = path
				}
				rt := st.plat.Runtime
				prepared, err := compileLlamaServing(command, rt, false)
				if err != nil {
					t.Fatal(err)
				}
				rt.Serving = prepared.Config
				if err := recheckPreparedServing(st, command, rt, prepared); err != nil {
					t.Fatalf("unchanged preset rejected: %v", err)
				}
			})
		}
	})
	t.Run("auto library pre-spawn", func(t *testing.T) {
		f, frozen, _ := autoTestFrozenLaunch(t)
		rt := f.st.plat.Runtime
		rt.ModelLoads = modelconfig.CloneLoads(frozen.Settings.ModelLoads)
		command := typedFixtureCommand(t, f, rt)
		prepared, err := compilePreparedServing(f.st, command, rt, true)
		if err != nil {
			t.Fatal(err)
		}
		library, ok := prepared.evidence.typed.libraries["libggml-base.dylib"]
		if !ok {
			t.Fatal("prepared library evidence missing")
		}
		replacement := filepath.Join(t.TempDir(), "replacement")
		writeServingFixture(t, replacement, []byte("replacement native library"))
		reads, otherReads, closes := 0, 0, 0
		substitute := false
		access := servingFileAccess{open: func(name string) (servingArtifactFile, error) {
			counter := &otherReads
			if substitute && name == library.configured {
				name, counter = replacement, &reads
			}
			file, err := os.Open(name)
			if err != nil {
				return nil, err
			}
			return &countedLinkedFile{File: file, reads: counter, closes: &closes}, nil
		}}
		if err := recheckPreparedServingAccess(f.st, command, rt, prepared, access); err != nil || otherReads == 0 {
			t.Fatalf("positive recheck: reads=%d error=%v", otherReads, err)
		}
		substitute = true
		if err := recheckPreparedServingAccess(f.st, command, rt, prepared, access); err == nil {
			t.Fatal("substituted library accepted")
		}
		if reads != 0 {
			t.Fatalf("replacement library read: %d", reads)
		}
	})
	t.Run("preset and artifact pre-spawn", func(t *testing.T) {
		f := newLlamaServingFixture(t)
		preview := approveLlamaServing(t, f)
		applyLlamaServing(t, f, preview)
		st := settingsState(t, f.ex)
		command, err := launchForState(st, st.port)
		if err != nil {
			t.Fatal(err)
		}
		prepared, err := compileLlamaServing(command, st.plat.Runtime, true)
		if err != nil {
			t.Fatal(err)
		}
		for _, path := range []string{f.source, filepath.Join(f.root, "medium.gguf")} {
			t.Run(filepath.Base(path), func(t *testing.T) {
				replacement := filepath.Join(t.TempDir(), "replacement")
				writeServingFixture(t, replacement, []byte("replacement"))
				reads, otherReads, closes := 0, 0, 0
				substitute := false
				access := servingFileAccess{open: func(name string) (servingArtifactFile, error) {
					counter := &otherReads
					if substitute && name == path {
						name, counter = replacement, &reads
					}
					file, err := os.Open(name)
					if err != nil {
						return nil, err
					}
					return &countedLinkedFile{File: file, reads: counter, closes: &closes}, nil
				}}
				if err := recheckPreparedServingAccess(st, command, st.plat.Runtime, prepared, access); err != nil || otherReads == 0 {
					t.Fatalf("positive recheck: reads=%d err=%v", otherReads, err)
				}
				substitute = true
				if err := recheckPreparedServingAccess(st, command, st.plat.Runtime, prepared, access); err == nil {
					t.Fatal("replacement accepted")
				}
				if reads != 0 {
					t.Fatalf("replacement read before rejection: %d", reads)
				}
			})
		}
	})
	t.Run("typed executable pre-spawn", func(t *testing.T) {
		f := newModelLoadFixture(t)
		rt := convertModelLoadFixture(t, f)
		command := typedFixtureCommand(t, f, rt)
		prepared, err := compilePreparedServing(f.st, command, rt, false)
		if err != nil {
			t.Fatal(err)
		}
		rt.ModelLoads = prepared.ModelLoads
		replacement := filepath.Join(t.TempDir(), "replacement")
		writeServingFixture(t, replacement, []byte("replacement executable"))
		reads, otherReads, closes := 0, 0, 0
		substitute := false
		access := servingFileAccess{open: func(name string) (servingArtifactFile, error) {
			counter := &otherReads
			if substitute && name == command.Bin {
				name, counter = replacement, &reads
			}
			file, err := os.Open(name)
			if err != nil {
				return nil, err
			}
			return &countedLinkedFile{File: file, reads: counter, closes: &closes}, nil
		}}
		if err := recheckPreparedServingAccess(f.st, command, rt, prepared, access); err != nil || otherReads == 0 {
			t.Fatalf("positive recheck: reads=%d err=%v", otherReads, err)
		}
		substitute = true
		if err := recheckPreparedServingAccess(f.st, command, rt, prepared, access); err == nil {
			t.Fatal("replacement accepted")
		}
		if reads != 0 {
			t.Fatalf("replacement executable read: %d", reads)
		}
	})
	t.Run("selected export retains artifact binding", func(t *testing.T) {
		f := newLlamaServingFixture(t)
		settingsState(t, f.ex).modelsDir = t.TempDir()
		applyLlamaServing(t, f, approveLlamaServing(t, f))
		path := filepath.Join(f.root, "medium.gguf")
		replacement := filepath.Join(t.TempDir(), "replacement")
		writeServingFixture(t, replacement, []byte("replacement"))
		opens, trigger, forbidden, otherReads, closes := 0, 0, 0, 0, 0
		access := servingFileAccess{open: func(name string) (servingArtifactFile, error) {
			counter := &otherReads
			if name == path {
				opens++
				if opens == trigger {
					name, counter = replacement, &forbidden
				}
			}
			file, err := os.Open(name)
			if err != nil {
				return nil, err
			}
			return &countedLinkedFile{File: file, reads: counter, closes: &closes}, nil
		}}
		if _, err := f.ex.modelSourceFilesWithAccess(context.Background(), "fake", "medium", access); err != nil {
			t.Fatal(err)
		}
		if opens == 0 || otherReads == 0 {
			t.Fatalf("export positive control did not read selected file: opens=%d reads=%d", opens, otherReads)
		}
		trigger, opens = opens, 0
		if _, err := f.ex.modelSourceFilesWithAccess(context.Background(), "fake", "medium", access); err == nil {
			t.Fatal("substituted export artifact accepted")
		}
		if opens != trigger || forbidden != 0 {
			t.Fatalf("export substitution: opens=%d trigger=%d forbidden reads=%d", opens, trigger, forbidden)
		}
	})
	t.Run("auto native library", func(t *testing.T) {
		f := newModelLoadFixture(t)
		autoTestNativeSource(t, &f)
		command := typedFixtureCommand(t, f, f.st.plat.Runtime)
		executable, err := captureSource(command.Bin, false)
		if err != nil {
			t.Fatal(err)
		}
		reads, closes := 0, 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			file, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			return &countedLinkedFile{File: file, reads: &reads, closes: &closes}, nil
		}}
		_, libraries, err := modelReviewedNativeSourceExpected(context.Background(), f.st, command, access, executable, nil)
		if err != nil || reads == 0 {
			t.Fatalf("initial native source: reads=%d error=%v", reads, err)
		}
		library, ok := libraries["libggml-base.dylib"]
		if !ok {
			t.Fatal("library evidence missing")
		}
		if err := os.Rename(library.configured, library.configured+".old"); err != nil {
			t.Fatal(err)
		}
		writeServingFixture(t, library.configured, []byte("replacement library"))
		reads = 0
		if _, _, err := modelReviewedNativeSourceExpected(context.Background(), f.st, command, access, executable, libraries); err == nil {
			t.Fatal("replacement library accepted")
		}
		if reads != 0 {
			t.Fatalf("replacement library read: %d", reads)
		}
	})
}
