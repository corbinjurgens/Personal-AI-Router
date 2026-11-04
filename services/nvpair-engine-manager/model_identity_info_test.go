// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/modelconfig"
)

func identityFixture(t *testing.T) inspectionFixture {
	t.Helper()
	f := newInspectionFixture(t, 2<<20)
	f.ex.inventory = map[string][]string{"llamacpp": {"exact-alias"}}
	return f
}

func resolvedIdentity(t *testing.T, f inspectionFixture) engines.ModelIdentity {
	t.Helper()
	f.ex.modelIdentities.reconcile(context.Background(), true, time.Now())
	identity, ok := f.ex.modelIdentities.snapshot()["llamacpp"][f.request.ModelID]
	if !ok {
		t.Fatal("identity missing")
	}
	return identity
}

func identityHFPath(t *testing.T, f inspectionFixture) (string, string, string) {
	t.Helper()
	revision, digest := strings.Repeat("a", 40), strings.Repeat("b", 64)
	repo := filepath.Join(f.root, "models--unsloth--LFM2.5-24B-A2B-GGUF")
	if err := os.MkdirAll(filepath.Join(repo, "blobs"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(repo, "snapshots", revision), 0700); err != nil {
		t.Fatal(err)
	}
	blob := filepath.Join(repo, "blobs", digest)
	writeInspectionGGUF(t, blob, "llama", 0.25, 2<<20)
	path := filepath.Join(repo, "snapshots", revision, "LFM2.5-24B-A2B-UD-Q4_K_XL.gguf")
	if err := os.Symlink(blob, path); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	return path, revision, digest
}

func TestModelIdentityInfoHFCache(t *testing.T) {
	f := identityFixture(t)
	path, revision, digest := identityHFPath(t, f)
	f.request.ModelID = "unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL"
	if err := validateLlamaModelID(f.request.ModelID); err == nil {
		t.Fatal("strict serving validation accepted a noncanonical native ID")
	}
	f.ex.inventory["llamacpp"] = []string{f.request.ModelID}
	writeTestFile(t, f.preset, []byte("["+f.request.ModelID+"]\nmodel = "+filepath.ToSlash(path)+"\n"))
	// Check computation independently of the resolver's exact-ID admission.
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	list := modelFileList{Files: []modelFile{{Path: filepath.Base(path), Role: "model", Size: info.Size(), configured: path, abs: filepath.Join(filepath.Dir(filepath.Dir(filepath.Dir(path))), "blobs", digest), info: info}}}
	direct, ok := f.ex.modelIdentityFromArtifacts("llamacpp", f.request.ModelID, list, 0)
	if !ok || direct.Quantization != "UD-Q4_K_XL" || direct.Repository != "unsloth/LFM2.5-24B-A2B-GGUF" || direct.Revision != revision || direct.Strength != "verified" {
		t.Fatalf("direct identity = %+v, valid %v", direct, ok)
	}
	id := resolvedIdentity(t, f)
	if id.Repository != "unsloth/LFM2.5-24B-A2B-GGUF" || id.Revision != revision || id.Quantization != "UD-Q4_K_XL" || id.QuantizationSource != "hf-tag" || id.Strength != "verified" || id.ContentID == "" || len(id.Files) != 1 || id.Files[0].SHA256 != digest {
		t.Fatalf("identity = %+v", id)
	}
}

func TestModelIdentityInfoArtifactIDStillRequiresSafeSection(t *testing.T) {
	for _, id := range []string{"*", " id", "id ", "bad[id]", "bad#id", "bad;id", "bad\x00id", "bad\nid"} {
		t.Run(id, func(t *testing.T) {
			if err := validateLlamaArtifactModelID(id); err == nil {
				t.Fatal("artifact lookup accepted an unsafe section ID")
			}
		})
	}
}

func TestModelIdentityInfoPresetSection(t *testing.T) {
	f := identityFixture(t)
	path, revision, _ := identityHFPath(t, f)
	f.request.ModelID = "lfm2.5-24b-a2b-msi-ctx131072"
	f.ex.inventory["llamacpp"] = []string{f.request.ModelID}
	writeTestFile(t, f.preset, []byte("[*]\nctx-size = 65536\nparallel = 2\n["+f.request.ModelID+"]\nmodel = "+filepath.ToSlash(path)+"\nctx-size = 131072\nparallel = 1\n"))
	id := resolvedIdentity(t, f)
	if id.ConfiguredContextTokens != 131072 || id.Repository != "unsloth/LFM2.5-24B-A2B-GGUF" || id.Revision != revision || id.Quantization != "UD-Q4_K_XL" || id.QuantizationSource != "file-name" {
		t.Fatalf("identity = %+v", id)
	}
}

func TestModelIdentityInfoPlainFile(t *testing.T) {
	f := identityFixture(t)
	id := resolvedIdentity(t, f)
	if id.Strength != "declared" || id.ContentID != "" || id.TotalBytes != 2<<20 || len(id.Files) != 1 || id.Files[0].Name != "main.gguf" || id.Files[0].SizeBytes != 2<<20 || id.Architecture != "llama" || id.TrainedContextTokens != 4096 {
		t.Fatalf("identity = %+v", id)
	}
	copy := f.ex.modelIdentities.snapshot()
	value := copy["llamacpp"]["exact-alias"]
	value.Files[0].Name = "changed"
	delete(copy, "llamacpp")
	if got := f.ex.modelIdentities.snapshot()["llamacpp"]["exact-alias"].Files[0].Name; got != "main.gguf" {
		t.Fatalf("snapshot aliased: %s", got)
	}
}

func TestModelContentIDMatchesServingFormula(t *testing.T) {
	digest := strings.Repeat("a", 64)
	model := engines.ServingModel{Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: strings.ToUpper(digest), Binding: &engines.ServingBinding{SizeBytes: 123}}}}
	// Literal pre-extraction JSON, including field order and no newline.
	data := `{"version":"pair-serving-content-v1","artifacts":[{"role":"model","sizeBytes":123,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}`
	sum := sha256.Sum256([]byte(data))
	want := hex.EncodeToString(sum[:])
	if got := servingContentDigest(model); got != want {
		t.Fatalf("serving digest = %s, want %s", got, want)
	}
	if got := contentDigestOf([]contentArtifact{{"model", 123, digest}}); got != want {
		t.Fatalf("content digest = %s, want %s", got, want)
	}
}

func TestModelIdentityInfoCacheAndFailures(t *testing.T) {
	f := identityFixture(t)
	m := f.ex.modelIdentities
	now := time.Now()
	m.reconcile(context.Background(), false, now)
	old := m.snapshot()["llamacpp"]["exact-alias"]
	if len(old.Files) != 1 {
		t.Fatal("missing initial identity")
	}
	writeTestFile(t, f.preset, []byte("[exact-alias]\n"))
	m.reconcile(context.Background(), false, now.Add(time.Minute))
	if len(m.snapshot()["llamacpp"]) != 1 {
		t.Fatal("cache was not retained before expiry")
	}
	f.ex.modelInspection.active.Store(true)
	m.reconcile(context.Background(), true, now.Add(time.Minute))
	f.ex.modelInspection.active.Store(false)
	if len(m.snapshot()["llamacpp"]) != 1 {
		t.Fatal("busy inspection dropped cached identity")
	}
	f.st.opMu.Lock()
	m.reconcile(context.Background(), true, now.Add(time.Minute))
	f.st.opMu.Unlock()
	if len(m.snapshot()["llamacpp"]) != 1 {
		t.Fatal("busy configuration dropped cached identity")
	}
	f.ex.downloads.mu.Lock()
	f.ex.downloads.writers["llamacpp"] = true
	f.ex.downloads.mu.Unlock()
	m.reconcile(context.Background(), true, now.Add(time.Minute))
	if len(m.snapshot()["llamacpp"]) != 1 {
		t.Fatal("busy download dropped cached identity")
	}
	f.ex.downloads.mu.Lock()
	delete(f.ex.downloads.writers, "llamacpp")
	f.ex.downloads.mu.Unlock()
	m.reconcile(context.Background(), false, now.Add(10*time.Minute))
	if len(m.snapshot()["llamacpp"]) != 0 {
		t.Fatal("resolution error retained expired identity")
	}
	writeTestFile(t, f.preset, []byte("[exact-alias]\nmodel = main.gguf\n"))
	m.reconcile(context.Background(), true, now)
	writeTestFile(t, f.preset, []byte("[exact-alias]\n"))
	f.st.mu.Lock()
	f.st.runID = "new-run"
	f.st.mu.Unlock()
	m.reconcile(context.Background(), false, now.Add(time.Minute))
	if len(m.snapshot()["llamacpp"]) != 0 {
		t.Fatal("run change reused stale identity")
	}
}

func TestModelIdentityInfoConfiguredContext(t *testing.T) {
	f := identityFixture(t)
	t.Setenv("LLAMA_ARG_CTX_SIZE", "8192")
	t.Setenv("LLAMA_ARG_PARALLEL", "2")
	writeTestFile(t, f.preset, []byte("[*]\nctx-size = 65536\n[exact-alias]\nmodel = main.gguf\nctx-size = 131072\n"))
	f.st.plat.Runtime.Args = append(f.st.plat.Runtime.Args, "--ctx-size", "32768", "--parallel", "4")
	id := resolvedIdentity(t, f)
	if id.ConfiguredContextTokens != 8192 {
		t.Fatalf("merged context = %d", id.ConfiguredContextTokens)
	}
	snapshot, err := inspectionSnapshot(f.st, "llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	snapshot.state.plat.Runtime.ModelLoads = []modelconfig.ModelLoad{{ModelID: "exact-alias", Resolved: &modelconfig.ResolvedLoad{PerSlotContext: 12288}}}
	if got := modelIdentityConfiguredContext(snapshot, "exact-alias", modelFileList{}); got != 12288 {
		t.Fatalf("typed context = %d", got)
	}
}

func TestModelIdentityInfoFactsAndModelRoles(t *testing.T) {
	f := identityFixture(t)
	data, _ := inspectionGGUF("llama", 0.25, "tensor")
	var extra []byte
	str := func(v string) {
		extra = binary.LittleEndian.AppendUint64(extra, uint64(len(v)))
		extra = append(extra, v...)
	}
	for _, field := range []struct{ key, value string }{{"general.name", "Readable model"}, {"general.size_label", "24B"}, {"general.repo_url", "https://huggingface.co/owner/repo/"}} {
		str(field.key)
		extra = binary.LittleEndian.AppendUint32(extra, 8)
		str(field.value)
	}
	str("general.file_type")
	extra = binary.LittleEndian.AppendUint32(extra, 4)
	extra = binary.LittleEndian.AppendUint32(extra, 15)
	binary.LittleEndian.PutUint64(data[16:24], 9)
	data = append(append(data[:24:24], extra...), data[24:]...)
	data = append(data, make([]byte, 1024)...)
	writeTestFile(t, f.model, data)
	id := resolvedIdentity(t, f)
	if id.GeneralName != "Readable model" || id.SizeLabel != "24B" || id.Repository != "owner/repo" || id.Quantization != "Q4_K_M" || id.QuantizationSource != "gguf-file-type" {
		t.Fatalf("facts identity = %+v", id)
	}
	info, err := os.Stat(f.model)
	if err != nil {
		t.Fatal(err)
	}
	digest := strings.Repeat("a", 64)
	list := modelFileList{Files: []modelFile{{Path: "model/main.gguf", Role: "model", Size: info.Size(), SHA256: digest, configured: f.model, abs: f.model, info: info}, {Path: "draft/draft.gguf", Role: "draft", Size: 99}}}
	got, ok := f.ex.modelIdentityFromArtifacts("llamacpp", "alias", list, 0)
	if !ok || got.TotalBytes != info.Size() || got.ContentID != contentDigestOf([]contentArtifact{{"model", info.Size(), digest}}) || got.Architecture != "" || got.GeneralName != "" {
		t.Fatalf("roles/facts error identity = %+v, valid %v", got, ok)
	}
}

func TestModelIdentityInfoRefreshInvalidatesAndPrunes(t *testing.T) {
	f := identityFixture(t)
	m := f.ex.modelIdentities
	resolvedIdentity(t, f)
	writeTestFile(t, f.preset, []byte("[exact-alias]\nmodel = main.gguf\nctx-size = 16384\n"))
	m.reconcile(context.Background(), true, time.Now())
	if got := m.snapshot()["llamacpp"]["exact-alias"].ConfiguredContextTokens; got != 16384 {
		t.Fatalf("refresh retained stale context: %d", got)
	}
	writeTestFile(t, f.preset, []byte("[exact-alias]\n"))
	m.reconcile(context.Background(), true, time.Now())
	if len(m.snapshot()["llamacpp"]) != 0 {
		t.Fatal("forced resolution retained missing model")
	}
	writeTestFile(t, f.preset, []byte("[exact-alias]\nmodel = main.gguf\n"))
	resolvedIdentity(t, f)
	f.ex.inventoryMu.Lock()
	f.ex.inventory = map[string][]string{}
	f.ex.inventoryMu.Unlock()
	m.reconcile(context.Background(), false, time.Now())
	if len(m.snapshot()) != 0 {
		t.Fatal("removed inventory retained identity")
	}
}

func TestModelIdentityInfoCancellationDuringRead(t *testing.T) {
	f := identityFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) { cancel(); return os.Open(path) }
	f.ex.modelIdentities.reconcile(ctx, true, time.Now())
	if ctx.Err() == nil {
		t.Fatal("read seam was not reached")
	}
	if len(f.ex.modelIdentities.snapshot()) != 0 {
		t.Fatal("cancelled resolution published identity")
	}
	if f.ex.modelInspection.active.Load() {
		t.Fatal("cancellation retained inspection gate")
	}
	f.ex.downloads.mu.Lock()
	readers := f.ex.downloads.readers["llamacpp"]
	f.ex.downloads.mu.Unlock()
	if readers != 0 {
		t.Fatalf("cancellation retained read admission: %d", readers)
	}
}

func TestModelIdentityInfoRefreshAndLifetime(t *testing.T) {
	f := identityFixture(t)
	m := f.ex.modelIdentities
	for i := 0; i < 100; i++ {
		m.refresh()
	}
	if len(m.poke) != 1 {
		t.Fatalf("refresh queue = %d", len(m.poke))
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	done := make(chan struct{})
	go func() { m.run(ctx); close(done) }()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("identity owner ignored lifetime cancellation")
	}
	if len(m.snapshot()) != 0 {
		t.Fatal("cancelled owner performed work")
	}
}
