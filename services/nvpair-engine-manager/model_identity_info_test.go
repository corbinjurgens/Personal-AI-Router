// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
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
	modelReads := 0
	f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
		if path == f.model {
			modelReads++
		}
		return os.Open(path)
	}
	m.reconcile(context.Background(), false, now.Add(time.Minute))
	if len(m.snapshot()["llamacpp"]) != 1 || modelReads != 0 {
		t.Fatalf("unchanged cache was not reused before expiry: %d model reads", modelReads)
	}
	writeTestFile(t, f.preset, []byte("[exact-alias]\n"))
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
	m.reconcile(context.Background(), false, now.Add(2*time.Minute))
	if len(m.snapshot()["llamacpp"]) != 0 {
		t.Fatal("an edited preset left the cached identity in place before expiry")
	}
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
	if len(f.st.contentMu) != 0 {
		t.Fatal("cancellation retained content semaphore")
	}
}

func TestModelIdentityInfoYieldsToReservedMutation(t *testing.T) {
	f := identityFixture(t)
	m := f.ex.modelIdentities
	resolvedIdentity(t, f)
	old := m.entries["llamacpp"][f.request.ModelID]
	f.ex.downloads.mu.Lock()
	generation := f.ex.downloads.authorityGeneration
	f.ex.downloads.mu.Unlock()
	opens := 0
	f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
		opens++
		return os.Open(path)
	}
	// A mutation owns this semaphore before requesting its writer lease.
	f.st.contentMu <- struct{}{}
	entry, keep := m.resolve(context.Background(), "llamacpp", f.request.ModelID, old, true, true, time.Now())
	<-f.st.contentMu
	if opens != 0 || !keep || !reflect.DeepEqual(entry, old) {
		t.Fatalf("reserved mutation did not preserve cached identity without reads: opens=%d keep=%t equal=%t", opens, keep, reflect.DeepEqual(entry, old))
	}
	f.ex.downloads.mu.Lock()
	readers, current := f.ex.downloads.readers["llamacpp"], f.ex.downloads.authorityGeneration
	f.ex.downloads.mu.Unlock()
	if readers != 0 || current != generation || f.ex.modelInspection.active.Load() {
		t.Fatalf("yield changed admission: readers=%d generation=%d want=%d inspection=%t", readers, current, generation, f.ex.modelInspection.active.Load())
	}
	if _, keep := m.resolve(context.Background(), "llamacpp", f.request.ModelID, old, true, true, time.Now()); !keep || opens == 0 {
		t.Fatalf("released mutation did not permit resolution: keep=%t opens=%d", keep, opens)
	}
}

func TestModelIdentityInfoSerializesReadWithMutation(t *testing.T) {
	f := identityFixture(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	opened, release, resolved := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
		once.Do(func() { close(opened) })
		select {
		case <-release:
			return os.Open(path)
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	go func() {
		f.ex.modelIdentities.reconcile(ctx, true, time.Now())
		close(resolved)
	}()
	select {
	case <-opened:
	case <-ctx.Done():
		t.Fatal("identity resolver did not reach artifact barrier")
	}
	// This fixture has no manager goroutines or other readers: the lease at
	// this barrier belongs to the background identity resolver alone.
	f.ex.downloads.mu.Lock()
	readers := f.ex.downloads.readers["llamacpp"]
	writer := f.ex.downloads.writers["llamacpp"]
	active, unsafe, recovery := len(f.ex.downloads.active), len(f.ex.downloads.unsafe), f.ex.downloads.recoveryRequired
	f.ex.downloads.mu.Unlock()
	if readers != 1 || writer || active != 0 || unsafe != 0 || recovery {
		t.Errorf("identity barrier admission: readers=%d writer=%t active=%d unsafe=%d recovery=%t", readers, writer, active, unsafe, recovery)
	}
	select {
	case f.st.contentMu <- struct{}{}:
		<-f.st.contentMu
		t.Error("identity reader did not serialize with content mutations")
	default:
	}
	withdrawals := 0
	f.ex.modelMutationParent = func(context.Context, string) error { withdrawals++; return nil }
	mutated := make(chan error, 1)
	go func() {
		mutated <- f.ex.withModelMutation(ctx, "llamacpp", func() error {
			f.ex.downloads.mu.Lock()
			defer f.ex.downloads.mu.Unlock()
			if f.ex.downloads.readers["llamacpp"] != 0 || !f.ex.downloads.writers["llamacpp"] {
				return fmt.Errorf("mutation did not acquire exclusive content admission")
			}
			return nil
		})
	}()
	close(release)
	select {
	case err := <-mutated:
		if err != nil {
			t.Fatalf("mutation behind passive identity read: %v", err)
		}
	case <-ctx.Done():
		t.Fatal("mutation did not complete after identity read")
	}
	select {
	case <-resolved:
	case <-ctx.Done():
		t.Fatal("identity resolution did not finish")
	}
	if withdrawals != 1 || len(f.st.contentMu) != 0 {
		t.Fatalf("mutation completion: withdrawals=%d content reservations=%d", withdrawals, len(f.st.contentMu))
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

func TestModelIdentityInfoRemembersFailures(t *testing.T) {
	f := identityFixture(t)
	m := f.ex.modelIdentities
	writeTestFile(t, f.preset, []byte("[exact-alias]\n"))
	reads := 0
	f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
		if path == f.preset {
			reads++
		}
		return os.Open(path)
	}
	attempt := func(force bool, at time.Time) int {
		t.Helper()
		before := reads
		m.reconcile(context.Background(), force, at)
		if len(m.snapshot()["llamacpp"]) != 0 {
			t.Fatal("failed resolution published an identity")
		}
		return reads - before
	}
	now := time.Now()
	if attempt(false, now) == 0 {
		t.Fatal("first resolution did not read the preset")
	}
	if n := attempt(false, now.Add(30*time.Second)); n != 0 {
		t.Fatalf("failed resolution repeated within the retry interval: %d preset reads", n)
	}
	retried := now.Add(modelIdentityFailureRetry)
	if attempt(false, retried) == 0 {
		t.Fatal("failed resolution was not retried after the interval")
	}
	f.ex.downloads.mu.Lock()
	err := f.ex.downloads.advanceLocked()
	f.ex.downloads.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if attempt(false, retried.Add(time.Second)) == 0 {
		t.Fatal("download change did not retry the failed resolution")
	}
	if attempt(true, retried.Add(2*time.Second)) == 0 {
		t.Fatal("explicit refresh did not retry the failed resolution")
	}
	writeTestFile(t, f.preset, []byte("[exact-alias]\nmodel = main.gguf\n"))
	m.reconcile(context.Background(), false, retried.Add(2*modelIdentityFailureRetry))
	if len(m.snapshot()["llamacpp"]) != 1 {
		t.Fatal("repaired model was not resolved after the retry interval")
	}
}

// starveInspection runs the identity owner continuously, refreshing it as
// often as it can, against background reads that are slow and always fail.
func starveInspection(t *testing.T, f inspectionFixture) {
	t.Helper()
	f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
		file, err := os.Open(path)
		if err != nil || !identityBackgroundCaller() {
			return file, err
		}
		return slowFailingFile{file}, nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); f.ex.modelIdentities.run(ctx) }()
	go func() {
		defer wg.Done()
		for ctx.Err() == nil {
			f.ex.modelIdentities.refresh()
			time.Sleep(time.Millisecond)
		}
	}()
	t.Cleanup(func() { cancel(); wg.Wait() })
	deadline := time.Now().Add(5 * time.Second)
	for !f.ex.modelInspection.active.Load() {
		if time.Now().After(deadline) {
			t.Fatal("identity owner never took the inspection gate")
		}
		time.Sleep(time.Millisecond)
	}
}

// The seam cannot see its caller's context; the stack names the background owner.
func identityBackgroundCaller() bool {
	buf := make([]byte, 64<<10)
	return bytes.Contains(buf[:runtime.Stack(buf, false)], []byte("(*modelIdentityInfo).resolveOnce"))
}

type slowFailingFile struct{ *os.File }

func (f slowFailingFile) ReadAt([]byte, int64) (int, error) {
	time.Sleep(100 * time.Millisecond)
	return 0, errors.New("slow artifact read failed")
}

func TestModelIdentityInfoYieldsToRequestedInspection(t *testing.T) {
	f := identityFixture(t)
	starveInspection(t, f)
	for i := 0; i < 3; i++ {
		if result := inspectFixture(t, f); result.Status != "resolved" {
			t.Fatalf("inspection %d behind the identity owner: %s %s", i, result.Status, result.Reason)
		}
	}
}

func TestModelIdentityInfoYieldsToRequestedPreview(t *testing.T) {
	f := newAppliedKeyFixture(t)
	f.ex.inventoryMu.Lock()
	f.ex.inventory = map[string][]string{"llamacpp": {f.request.ModelID}}
	f.ex.inventoryMu.Unlock()
	request := nativeOptionsPreviewRequest(t, f, nil)
	request.ModelChange.Load.Set.Context = &modelconfig.Context{Mode: "fixed", Tokens: 4096}
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: modelTargetMemoryOwner{target: modelBufferTestTargetMemory()}})
	starveInspection(t, f)
	for i := 0; i < 3; i++ {
		result, err := f.ex.previewModelLoadChange(ctx, f.st, request)
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Errors) != 0 {
			t.Fatalf("preview %d behind the identity owner: %v", i, result.Errors)
		}
	}
}

func TestModelIdentityInfoRefreshFollowsInventoryChanges(t *testing.T) {
	f := identityFixture(t)
	store := func(models ...string) bool {
		t.Helper()
		return f.ex.storeInventory(map[string][]string{"llamacpp": models}, map[string]observedDigests{"llamacpp": modelInventoryObservation(t, f.ex, "llamacpp")}, f.ex.sweeps.Add(1))
	}
	if !store("exact-alias", "other") {
		t.Fatal("an added model was not reported as an inventory change")
	}
	// A peer reading /v1/models repeats the same sweep result.
	if store("exact-alias", "other") {
		t.Fatal("an unchanged inventory was reported as a change")
	}
	if !store("exact-alias") {
		t.Fatal("a removed model was not reported as an inventory change")
	}
}

func TestModelIdentityInfoNoticesFileReplacedInPlace(t *testing.T) {
	f := identityFixture(t)
	m := f.ex.modelIdentities
	now := time.Now()
	m.reconcile(context.Background(), false, now)
	before := m.snapshot()["llamacpp"]["exact-alias"].TotalBytes
	writeInspectionGGUF(t, f.model, "llama", 0.25, 3<<20)
	m.reconcile(context.Background(), false, now.Add(time.Minute))
	if after := m.snapshot()["llamacpp"]["exact-alias"].TotalBytes; after == before || after != 3<<20 {
		t.Fatalf("replaced model kept its stale identity: before=%d after=%d", before, after)
	}
}
