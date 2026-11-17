// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"nvpair-shared/modelconfig"
)

func importTestExecutor(t *testing.T) (*Executor, engineModelStore) {
	t.Helper()
	ex := copyTestExecutor(t, "llamacpp", t.TempDir(), true)
	ex.baseDir = t.TempDir()
	_, store, err := ex.modelStoreFor("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	return ex, store
}

func fileDigest(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func TestImportModelCopiesAFileIntoABundle(t *testing.T) {
	plentyOfDisk(t)
	ex, store := importTestExecutor(t)
	dir := t.TempDir()
	file := filepath.Join(dir, "Mine-Q4_K_M.gguf")
	writeSourceGGUF(t, file)
	writeSourceGGUF(t, filepath.Join(dir, "mmproj-mine.gguf"))
	before := fileDigest(t, file)

	res, err := ex.importModel(context.Background(), importParams{Engine: "llamacpp", Path: file, ID: "Mine-Q4_K_M"})
	if err != nil {
		t.Fatalf("import: %v", err)
	}
	if res.ID != "Mine-Q4_K_M" || res.Layout != llamaArtifactsLayout || !validSHA256(res.ExportID) {
		t.Fatalf("result = %+v", res)
	}
	copied := filepath.Join(copyAliasDir(store.root, res.ID), res.ExportID, "model", "Mine-Q4_K_M.gguf")
	if fileDigest(t, copied) != before {
		t.Fatal("bundle file differs from the source")
	}
	if fileDigest(t, file) != before {
		t.Fatal("import changed the source file")
	}
	receipt, err := readCopyReceipt(store, res.ID)
	if err != nil || receipt.List.ExportID != res.ExportID || len(receipt.List.Files) != 1 || receipt.List.Files[0].SHA256 != before {
		t.Fatalf("receipt = %+v, %v", receipt, err)
	}

	// The receipt is a resolver candidate, so a profile on the ID binds the copy.
	st, _, err := ex.modelStoreFor("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, command, err := copyLaunchSnapshot(st)
	if err != nil {
		t.Fatal(err)
	}
	list, _, err := resolveLlamaArtifacts(context.Background(), snapshot, command, store, res.ID, servingFileAccess{})
	if err != nil || len(list.Files) != 1 || list.Files[0].configured != copied {
		t.Fatalf("resolved %+v, %v", list.Files, err)
	}

	again, err := ex.importModel(context.Background(), importParams{Engine: "llamacpp", Path: file, ID: "Mine-Q4_K_M"})
	if err != nil || again.ExportID != res.ExportID {
		t.Fatalf("repeat import = %+v, %v", again, err)
	}
	other := filepath.Join(t.TempDir(), "Other.gguf")
	writeSourceGGUF(t, other)
	if _, err := ex.importModel(context.Background(), importParams{Engine: "llamacpp", Path: other, ID: "Mine-Q4_K_M"}); err == nil || !strings.Contains(err.Error(), "conflict") {
		t.Fatalf("a different file under the same name = %v", err)
	}
}

func TestImportModelFromASourceFolder(t *testing.T) {
	plentyOfDisk(t)
	ex, store := importTestExecutor(t)
	studio := t.TempDir()
	writeSourceGGUF(t, filepath.Join(studio, "pub", "repo", "Big-00001-of-00002.gguf"))
	writeSourceGGUF(t, filepath.Join(studio, "pub", "repo", "Big-00002-of-00002.gguf"))
	ollama := t.TempDir()
	writeSourceOllamaModel(t, ollama, "registry.ollama.ai/library/tiny/latest", map[string]string{ollamaModelLayer: "GGUF ollama blob"})
	ex.modelSources = []modelSource{
		{name: "ollama", layout: sourceLayoutOllama, root: ollama},
		{name: "lmstudio", layout: sourceLayoutGGUFTree, root: studio},
	}
	for source, want := range map[string][]string{
		"lmstudio/pub/repo/Big": {"model/Big-00001-of-00002.gguf", "model/Big-00002-of-00002.gguf"},
		"ollama/tiny@latest":    {"model/tiny-latest.gguf"},
	} {
		t.Run(source, func(t *testing.T) {
			id := strings.NewReplacer("/", "-", "@", "-").Replace(strings.SplitN(source, "/", 2)[1])
			res, err := ex.importModel(context.Background(), importParams{Engine: "llamacpp", SourceID: source, ID: id})
			if err != nil {
				t.Fatalf("import: %v", err)
			}
			receipt, err := readCopyReceipt(store, id)
			if err != nil {
				t.Fatal(err)
			}
			var got []string
			for _, f := range receipt.List.Files {
				got = append(got, f.Path)
			}
			if strings.Join(got, ",") != strings.Join(want, ",") || receipt.List.ExportID != res.ExportID {
				t.Fatalf("bundle files = %v, want %v", got, want)
			}
		})
	}
}

func TestImportModelRefusals(t *testing.T) {
	ex, store := importTestExecutor(t)
	file := filepath.Join(t.TempDir(), "Mine.gguf")
	writeSourceGGUF(t, file)
	for name, p := range map[string]importParams{
		"both inputs":    {Engine: "llamacpp", Path: file, SourceID: "lmstudio/x", ID: "Mine"},
		"no input":       {Engine: "llamacpp", ID: "Mine"},
		"unknown source": {Engine: "llamacpp", SourceID: "lmstudio/missing", ID: "Mine"},
		"bad name":       {Engine: "llamacpp", Path: file, ID: "Mine;rm"},
		"not a model":    {Engine: "llamacpp", Path: filepath.Join(t.TempDir(), "x.gguf"), ID: "Mine"},
		"repo name":      {Engine: "llamacpp", Path: file, ID: "owner/repo"},
		"repo and tag":   {Engine: "llamacpp", Path: file, ID: "owner/repo:Q4_K_M"},
		"linked prefix":  {Engine: "llamacpp", Path: file, ID: "local/Mine"},
		"ollama prefix":  {Engine: "llamacpp", Path: file, ID: "ollama/mine@latest"},
		"studio prefix":  {Engine: "llamacpp", Path: file, ID: "lmstudio/pub/repo/Mine"},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := ex.importModel(context.Background(), p); err == nil {
				t.Fatal("import accepted")
			}
		})
	}
	t.Run("insufficient disk", func(t *testing.T) {
		prev := diskFreeBytes
		diskFreeBytes = func(string) (uint64, error) { return 0, nil }
		t.Cleanup(func() { diskFreeBytes = prev })
		if _, err := ex.importModel(context.Background(), importParams{Engine: "llamacpp", Path: file, ID: "Mine"}); err == nil {
			t.Fatal("import ran without disk space")
		}
		if _, err := os.Stat(copyAliasDir(store.root, "Mine")); err == nil {
			entries, _ := os.ReadDir(copyAliasDir(store.root, "Mine"))
			for _, entry := range entries {
				if entry.Name() == "copy.json" {
					t.Fatal("a receipt was written without disk space")
				}
			}
		}
	})
}

// An imported bundle appears in the source listing as a store copy, so the
// Models tab can show it, and engine:delete-model removes it once no profile
// uses it. An unfinished copy, with no receipt, is not listed.
func TestCopyBundlesAreListedAndDeletable(t *testing.T) {
	plentyOfDisk(t)
	ex, store := importTestExecutor(t)
	file := filepath.Join(t.TempDir(), "Mine-00001-of-00002.gguf")
	writeSourceGGUF(t, file)
	writeSourceGGUF(t, filepath.Join(filepath.Dir(file), "Mine-00002-of-00002.gguf"))
	res, err := ex.importModel(context.Background(), importParams{Engine: "llamacpp", Path: file, ID: "Mine"})
	if err != nil {
		t.Fatalf("import: %v", err)
	}
	unfinished := filepath.Join(copyAliasDir(store.root, "Other"), res.ExportID, "model", "Other.gguf.part")
	writeSourceFile(t, unfinished, []byte("partial"))

	listing := ex.sourceModelsList(context.Background(), "llamacpp")
	var rows []sourceModelInfo
	for _, row := range listing.Models {
		if row.Source == copySourceName {
			rows = append(rows, row)
		}
	}
	if len(rows) != 1 || rows[0].ID != "Mine" || rows[0].Name != "Mine" || rows[0].Shards != 2 || rows[0].SizeBytes <= 0 || rows[0].HasProjector {
		t.Fatalf("copy rows = %+v", rows)
	}
	data, err := json.Marshal(listing)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), filepath.ToSlash(store.root)) || strings.Contains(string(data), strings.ReplaceAll(store.root, `\`, `\\`)) {
		t.Fatalf("listing leaks a host path: %s", data)
	}

	st, err := ex.state("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	st.plat.Runtime.ModelLoads = []modelconfig.ModelLoad{{ProfileKey: "chat", ModelID: "Mine"}}
	if _, err := ex.ModelDelete(context.Background(), "llamacpp", "Mine"); err == nil || !strings.Contains(err.Error(), "used by profile(s) chat") {
		t.Fatalf("delete while a profile uses it = %v", err)
	}
	if _, err := readCopyReceipt(store, "Mine"); err != nil {
		t.Fatalf("refused delete removed the receipt: %v", err)
	}
	st.plat.Runtime.ModelLoads = nil
	if _, err := ex.ModelDelete(context.Background(), "llamacpp", "Mine"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if _, err := os.Stat(copyAliasDir(store.root, "Mine")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("bundle directory remains: %v", err)
	}
	for _, row := range ex.sourceModelsList(context.Background(), "llamacpp").Models {
		if row.Source == copySourceName {
			t.Fatalf("deleted bundle still listed: %+v", row)
		}
	}
}
