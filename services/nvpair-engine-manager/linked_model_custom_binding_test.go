// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"nvpair-shared/modelconfig"
)

func customBindingLoad(t *testing.T, model, path string) modelconfig.ModelLoad {
	t.Helper()
	binding, err := servingFileBinding(path)
	if err != nil {
		t.Fatal(err)
	}
	load := modelconfig.ModelLoad{ModelID: model, Resolved: &modelconfig.ResolvedLoad{
		PerSlotContext: 4096, TotalContext: 4096, Slots: 1, GPU: modelconfig.GPU{Mode: "cpu"}, NativeBuildID: "synthetic-build",
		SourceInputDigest: strings.Repeat("a", 64), PresetDigest: strings.Repeat("b", 64),
		Artifacts: []modelconfig.ArtifactBinding{{Role: "model", Path: binding.SourcePath, ResolvedTarget: binding.ResolvedTarget, Size: strconv.FormatInt(binding.SizeBytes, 10), MtimeNS: binding.MtimeNS, SampleSHA256: binding.SampleSHA256}},
	}}
	if err := load.Validate(); err != nil {
		t.Fatal(err)
	}
	return load
}

func TestLinkedModelCustomBinding(t *testing.T) {
	t.Run("preset replacement during HF recheck", func(t *testing.T) {
		root := t.TempDir()
		const model = "owner/model:Q4_K_M"
		writeLlamaCache(t, root, "owner/model", strings.Repeat("a", 40), map[string][]byte{"model-Q4_K_M.gguf": []byte("cached model")})
		preset := filepath.Join(t.TempDir(), "models.ini")
		writeTestFile(t, preset, []byte("[unrelated]\nmodel = unrelated.gguf\n"))
		ex := copyTestExecutor(t, "llamacpp", root, false)
		st, store, err := ex.modelStoreFor("llamacpp")
		if err != nil {
			t.Fatal(err)
		}
		st.plat.Runtime.Args = []string{"--models-preset", preset}
		snapshot, command, err := copyLaunchSnapshot(st)
		if err != nil {
			t.Fatal(err)
		}
		list, _, err := resolveLlamaArtifacts(context.Background(), snapshot, command, store, model, servingFileAccess{})
		if err != nil {
			t.Fatal(err)
		}
		changed := false
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			if !changed && strings.Contains(path, string(filepath.Separator)+"refs"+string(filepath.Separator)) {
				changed = true
				if err := os.Rename(preset, preset+"-old"); err != nil {
					return nil, err
				}
				if err := os.WriteFile(preset, []byte("["+model+"]\nmodel = new.gguf\n"), 0600); err != nil {
					return nil, err
				}
			}
			return os.Open(path)
		}}
		if _, _, err := resolveLlamaArtifactsExpected(context.Background(), snapshot, command, store, model, access, list); err == nil {
			t.Fatal("accepted preset replaced during HF recheck")
		}
		if !changed {
			t.Fatal("HF ref read was not reached")
		}
	})
	type fixture struct {
		state           *engineState
		store           engineModelStore
		command         launchCommand
		preset, receipt string
		artifact        string
	}
	newAgreement := func(t *testing.T) fixture {
		t.Helper()
		root := t.TempDir()
		const model = "custom"
		data := []byte("shared custom model")
		list := modelFileList{Engine: "llamacpp", Model: model, Layout: llamaArtifactsLayout, Files: []modelFile{{Role: "model", Path: "model/shared.gguf", Size: int64(len(data)), SHA256: sha256Hex(data)}}}
		id, err := artifactExportID(list)
		if err != nil {
			t.Fatal(err)
		}
		list.ExportID = id
		base := copyAliasDir(root, model)
		artifact := filepath.Join(base, id, "model", "shared.gguf")
		writeTestFile(t, artifact, data)
		receipt := filepath.Join(base, "copy.json")
		receiptBytes, err := json.Marshal(copyReceipt{Version: 1, List: list})
		if err != nil {
			t.Fatal(err)
		}
		writeTestFile(t, receipt, receiptBytes)
		preset := filepath.Join(t.TempDir(), "models.ini")
		writeTestFile(t, preset, []byte("[custom]\nmodel = "+artifact+"\n"))
		ex := copyTestExecutor(t, "llamacpp", root, false)
		state, store, err := ex.modelStoreFor("llamacpp")
		if err != nil {
			t.Fatal(err)
		}
		state.plat.Runtime.Args = []string{"--models-preset", preset}
		state.plat.Runtime.ModelLoads = []modelconfig.ModelLoad{customBindingLoad(t, model, artifact)}
		snapshot, command, err := copyLaunchSnapshot(state)
		if err != nil {
			t.Fatal(err)
		}
		return fixture{snapshot, store, command, preset, receipt, artifact}
	}
	resolve := func(t *testing.T, f fixture) modelFileList {
		t.Helper()
		list, selected, err := resolveLlamaArtifacts(context.Background(), f.state, f.command, f.store, "custom", servingFileAccess{})
		if err != nil || !selected {
			t.Fatalf("agreeing configured sources: selected=%v err=%v", selected, err)
		}
		return list
	}
	countAccess := func(reads, closes *int) servingFileAccess {
		return servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			return &countedLinkedFile{File: f, reads: reads, closes: closes}, nil
		}}
	}
	replace := func(t *testing.T, path string) {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Rename(path, path+"-old"); err != nil {
			t.Fatal(err)
		}
		writeTestFile(t, path, data)
		if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
			t.Fatal(err)
		}
	}
	t.Run("retains every agreeing preset receipt and typed binding", func(t *testing.T) {
		f := newAgreement(t)
		list := resolve(t, f)
		if list.source == nil || len(list.source.selectors) != 2 || len(list.source.artifacts) != 3 || len(list.Files) != 1 || list.Files[0].configured != f.artifact {
			t.Fatalf("incomplete agreement evidence: %+v", list.source)
		}
		current, selected, err := resolveLlamaArtifactsExpected(context.Background(), f.state, f.command, f.store, "custom", servingFileAccess{}, list)
		if err != nil || !selected || current.binding != list.binding {
			t.Fatalf("unchanged expected resolution: selected=%v err=%v", selected, err)
		}
	})
	testReplacement := func(name string, path func(fixture) string) {
		t.Run(name, func(t *testing.T) {
			f := newAgreement(t)
			list := resolve(t, f)
			replace(t, path(f))
			reads, closes := 0, 0
			current, _, err := resolveLlamaArtifactsExpected(context.Background(), f.state, f.command, f.store, "custom", countAccess(&reads, &closes), list)
			if err == nil || len(current.Files) != 0 || reads != 0 {
				t.Fatalf("replacement: files=%d reads=%d err=%v", len(current.Files), reads, err)
			}
		})
	}
	testReplacement("same-stat preset replacement refuses before all reads", func(f fixture) string { return f.preset })
	testReplacement("same-stat receipt replacement refuses before all reads", func(f fixture) string { return f.receipt })
	testReplacement("same-stat artifact replacement refuses before all reads", func(f fixture) string { return f.artifact })
	t.Run("fresh capture can accept identical replacement while old evidence refuses", func(t *testing.T) {
		f := newAgreement(t)
		old := resolve(t, f)
		replace(t, f.preset)
		fresh := resolve(t, f)
		if sameCopyFile(old.source.selectors[0].file.info, fresh.source.selectors[0].file.info) {
			t.Fatal("replacement retained old object identity")
		}
		if _, _, err := resolveLlamaArtifactsExpected(context.Background(), f.state, f.command, f.store, "custom", servingFileAccess{}, old); err == nil {
			t.Fatal("old authority accepted the fresh object")
		}
	})
	t.Run("configuration drift refuses before reads", func(t *testing.T) {
		f := newAgreement(t)
		list := resolve(t, f)
		f.command.Args = append(f.command.Args, "--ctx-size", "8192")
		reads, closes := 0, 0
		if _, _, err := resolveLlamaArtifactsExpected(context.Background(), f.state, f.command, f.store, "custom", countAccess(&reads, &closes), list); err == nil || reads != 0 {
			t.Fatalf("configuration drift: reads=%d err=%v", reads, err)
		}
	})
	t.Run("missing expected evidence refuses before reads", func(t *testing.T) {
		f := newAgreement(t)
		list := resolve(t, f)
		list.source = nil
		reads, closes := 0, 0
		if _, _, err := resolveLlamaArtifactsExpected(context.Background(), f.state, f.command, f.store, "custom", countAccess(&reads, &closes), list); err == nil || reads != 0 {
			t.Fatalf("missing snapshot: reads=%d err=%v", reads, err)
		}
	})
	t.Run("HF fallback retains checked selectors and directories", func(t *testing.T) {
		f := newAgreement(t)
		const model = "owner/model:Q4_K_M"
		writeLlamaCache(t, f.store.root, "owner/model", strings.Repeat("a", 40), map[string][]byte{"model-Q4_K_M.gguf": []byte("cached model")})
		// This preset selects another model, but its contents still affected the
		// decision to use the HF fallback and remain part of the old evidence.
		list, selected, err := resolveLlamaArtifacts(context.Background(), f.state, f.command, f.store, model, servingFileAccess{})
		if err != nil || selected || list.source == nil || len(list.source.selectors) != 2 || len(list.source.directories) == 0 {
			t.Fatalf("HF fallback evidence: selected=%v source=%+v err=%v", selected, list.source, err)
		}
		if _, _, err := resolveLlamaArtifactsExpected(context.Background(), f.state, f.command, f.store, model, servingFileAccess{}, list); err != nil {
			t.Fatal(err)
		}
		replace(t, list.source.selectors[1].file.configured)
		reads, closes := 0, 0
		if _, _, err := resolveLlamaArtifactsExpected(context.Background(), f.state, f.command, f.store, model, countAccess(&reads, &closes), list); err == nil || reads != 0 {
			t.Fatalf("old HF selector: reads=%d err=%v", reads, err)
		}
	})
	t.Run("disagreeing typed source refuses", func(t *testing.T) {
		f := newAgreement(t)
		other := filepath.Join(t.TempDir(), "other.gguf")
		writeTestFile(t, other, []byte("other model"))
		f.state.plat.Runtime.ModelLoads = []modelconfig.ModelLoad{customBindingLoad(t, "custom", other)}
		if _, _, err := resolveLlamaArtifacts(context.Background(), f.state, f.command, f.store, "custom", servingFileAccess{}); err == nil {
			t.Fatal("disagreeing sources accepted")
		}
	})
	t.Run("typed expected open rejects a substituted object before sampling", func(t *testing.T) {
		f := newAgreement(t)
		f.command.Args = nil
		f.state.plat.Runtime.Args = nil
		if err := os.Remove(f.receipt); err != nil {
			t.Fatal(err)
		}
		list, selected, err := resolveLlamaArtifacts(context.Background(), f.state, f.command, f.store, "custom", servingFileAccess{})
		if err != nil || selected {
			t.Fatalf("typed-only capture: preset=%v err=%v", selected, err)
		}
		replacement := filepath.Join(t.TempDir(), "replacement.gguf")
		writeTestFile(t, replacement, []byte("forbidden replacement"))
		reads, closes, opens := 0, 0, 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			if path != f.artifact {
				t.Fatalf("unexpected typed source open %q", path)
			}
			file, err := os.Open(replacement)
			if err != nil {
				return nil, err
			}
			opens++
			return &countedLinkedFile{File: file, reads: &reads, closes: &closes}, nil
		}}
		if _, _, err := resolveLlamaArtifactsExpected(context.Background(), f.state, f.command, f.store, "custom", access, list); err == nil || reads != 0 || opens == 0 || opens != closes {
			t.Fatalf("substituted typed handle: reads=%d opens=%d closes=%d err=%v", reads, opens, closes, err)
		}
	})
	t.Run("receipt symlink still refuses before external content", func(t *testing.T) {
		f := newAgreement(t)
		external := filepath.Join(t.TempDir(), "copy.json")
		if err := os.Rename(f.receipt, external); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(external, f.receipt); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		reads, closes := 0, 0
		access := countAccess(&reads, &closes)
		if _, err := readCopyReceiptWithAccess(context.Background(), f.store, "custom", access); err == nil || reads != 0 {
			t.Fatalf("linked receipt: reads=%d err=%v", reads, err)
		}
	})
	t.Run("outside-store custom capture and expected recheck need no native inventory", func(t *testing.T) {
		f := newCustomCopyFixture(t)
		state, store, err := f.source.modelStoreFor("llamacpp")
		if err != nil {
			t.Fatal(err)
		}
		list, err := f.source.modelSourceFiles(context.Background(), "llamacpp", reportedCopyAlias)
		if err != nil {
			t.Fatal(err)
		}
		if pathWithinRoot(store.root, list.Files[0].abs) {
			t.Fatal("outside-store fixture became confined")
		}
		snapshot, command, err := copyLaunchSnapshot(state)
		if err != nil {
			t.Fatal(err)
		}
		access := servingFileAccess{recognize: func(context.Context, string) (engineInventory, error) {
			t.Fatal("configured custom source requested native inventory")
			return engineInventory{}, nil
		}}
		if _, _, err := resolveLlamaArtifactsExpected(context.Background(), snapshot, command, store, reportedCopyAlias, access, list); err != nil {
			t.Fatal(err)
		}
		if err := f.source.recheckCopyExport(context.Background(), list); err != nil {
			t.Fatal(err)
		}
		replace(t, f.preset)
		if err := f.source.recheckCopyExport(context.Background(), list); err == nil {
			t.Fatal("copy export recheck refreshed replaced preset")
		}
	})
	t.Run("saved typed metadata drift refuses before samples", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "external.gguf")
		writeTestFile(t, path, []byte("original"))
		load := customBindingLoad(t, "custom", path)
		writeTestFile(t, path, []byte("changed-size"))
		reads, closes := 0, 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			return &countedLinkedFile{File: f, reads: &reads, closes: &closes}, nil
		}}
		if _, err := resolvedLoadArtifacts(context.Background(), engineModelStore{engine: "llamacpp", root: t.TempDir()}, "custom", load, access); err == nil {
			t.Fatal("stale typed binding accepted")
		}
		if reads != 0 {
			t.Fatalf("replacement content reads = %d, want 0", reads)
		}
	})
}
