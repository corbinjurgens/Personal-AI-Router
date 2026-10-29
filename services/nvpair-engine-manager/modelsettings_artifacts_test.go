// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/gguffacts"
	"nvpair-shared/modelconfig"
)

type inspectionFixture struct {
	ex                  *Executor
	st                  *engineState
	root, preset, model string
	request             modelconfig.InspectRequest
	directoryEnd        int64
}

// Every fixture is synthetic. Tensor bytes are sparse zeros and never decoded.
func inspectionGGUF(architecture string, sampler float64, tensorName string) ([]byte, int64) {
	data := []byte("GGUF")
	u32 := func(v uint32) { data = binary.LittleEndian.AppendUint32(data, v) }
	u64 := func(v uint64) { data = binary.LittleEndian.AppendUint64(data, v) }
	str := func(v string) { u64(uint64(len(v))); data = append(data, v...) }
	u32(3)
	u64(1)
	u64(5)
	str("general.architecture")
	u32(8)
	str(architecture)
	str(architecture + ".context_length")
	u32(4)
	u32(4096)
	str(architecture + ".block_count")
	u32(4)
	u32(4)
	str("general.sampling.temp")
	u32(12)
	u64(math.Float64bits(sampler))
	str("tokenizer.chat_template")
	u32(8)
	str("private synthetic template")
	str(tensorName)
	u32(1)
	u64(32)
	u32(0)
	u64(0)
	end := int64(len(data))
	for len(data)%32 != 0 {
		data = append(data, 0)
	}
	return data, end
}

func writeInspectionGGUF(t *testing.T, path, architecture string, sampler float64, size int64) int64 {
	t.Helper()
	name := "blk.0.weight"
	if parts := llamaShardName.FindStringSubmatch(path); parts != nil {
		name = "blk." + strings.TrimLeft(parts[2], "0") + ".weight"
	}
	data, end := inspectionGGUF(architecture, sampler, name)
	if size < int64(len(data))+128 {
		t.Fatal("synthetic size does not contain the tensor")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Truncate(size); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return end
}

type inspectionForbiddenHTTP struct{ t *testing.T }

func (r inspectionForbiddenHTTP) RoundTrip(*http.Request) (*http.Response, error) {
	r.t.Error("inspection attempted HTTP")
	return nil, errors.New("HTTP forbidden")
}

func newInspectionFixture(t *testing.T, size int64) inspectionFixture {
	t.Helper()
	t.Setenv("LLAMA_ARG_MODELS_PRESET", "")
	// An empty native environment option still has presence semantics. Restore
	// the caller's value at cleanup, but leave only the CLI preset in this fixture.
	if err := os.Unsetenv("LLAMA_ARG_MODELS_PRESET"); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	ex := copyTestExecutor(t, "llamacpp", root, false)
	st, err := ex.state("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	preset, model := filepath.Join(root, "models.ini"), filepath.Join(root, "main.gguf")
	end := writeInspectionGGUF(t, model, "llama", 0.25, size)
	writeTestFile(t, preset, []byte("[exact-alias]\nmodel = main.gguf\n"))
	st.plat.Runtime.Args = []string{"--models-preset", preset}
	ex.client = &http.Client{Transport: inspectionForbiddenHTTP{t}}
	ex.processes = func(context.Context) ([]string, error) {
		t.Error("inspection attempted process inventory")
		return nil, errors.New("process inventory forbidden")
	}
	ex.modelMutationParent = func(context.Context, string) error {
		t.Error("inspection attempted content mutation")
		return errors.New("mutation forbidden")
	}
	ex.emit = func(string, any) { t.Error("inspection emitted a mutation/runtime notification") }
	return inspectionFixture{ex, st, root, preset, model, modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "exact-alias"}, end}
}

func inspectFixture(t *testing.T, f inspectionFixture) modelconfig.InspectResult {
	t.Helper()
	result, err := f.ex.InspectModel(context.Background(), f.request)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

// Cache fixtures opt into exact native membership; configured preset, receipt
// and typed sources continue to exercise their own authority with the engine off.
func inspectionNativeMembership(t *testing.T, f inspectionFixture) *int {
	t.Helper()
	return copyTestNativeModels(t, f.ex, f.request.Engine, f.request.ModelID)
}

func TestModelSettingsArtifactsExactAliasOffAndNoApproval(t *testing.T) {
	f := newInspectionFixture(t, 2<<20)
	// Inspection must not compile the other declaration or require its files,
	// approved samples, native context, hashes or qualification evidence.
	f.st.plat.Runtime.Serving = &engines.ServingConfig{Version: 1, Models: []engines.ServingModel{{ModelID: "other-missing"}}}
	writeTestFile(t, f.preset, []byte("[exact-alias]\nmodel = main.gguf\n[other-missing]\nmodel = absent.gguf\n"))
	before, err := os.ReadFile(f.preset)
	if err != nil {
		t.Fatal(err)
	}
	result := inspectFixture(t, f)
	if result.Status != "resolved" || len(result.Artifacts) != 1 || result.Artifacts[0].Role != "model" {
		t.Fatalf("inspection=%+v", result)
	}
	if result.Facts.Architecture != "llama" || result.Facts.ContextTokens == nil || *result.Facts.ContextTokens != 4096 || result.Facts.Dimensions["tensor.bytes"] != "128" {
		t.Fatalf("facts=%+v", result.Facts)
	}
	seen := map[string]bool{}
	for _, option := range result.Options {
		if seen[option.Field] {
			t.Fatalf("duplicate descriptor %s", option.Field)
		}
		seen[option.Field] = true
		if option.Field == "generation.temperature" && (string(option.Default) != "0.25" || !strings.HasPrefix(option.DefaultSource, "embedded GGUF")) {
			t.Fatalf("recommendation=%+v", option)
		}
	}
	if f.st.running || f.st.gen != 0 || f.st.plat.Runtime.Serving.Models[0].Artifacts != nil || len(f.st.plat.Runtime.ModelLoads) != 0 {
		t.Fatal("read changed runtime or authority")
	}
	after, err := os.ReadFile(f.preset)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("inspection rewrote source preset")
	}
	raw, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte(f.root)) || bytes.Contains(raw, []byte("private synthetic template")) || bytes.Contains(raw, []byte("blk.0.weight")) {
		t.Fatal("public inspection disclosed paths/template/tensor names")
	}
	f.request.ModelID = "exact-alias-suffix"
	if result := inspectFixture(t, f); result.Status == "resolved" {
		t.Fatal("inexact alias selected another section")
	}
}

type inspectionRead struct {
	path   string
	offset int64
	bytes  int
}
type inspectionCountFile struct {
	*os.File
	reads          *[]inspectionRead
	opened, closed *int
	onRead         func(int64, int)
}

func (f *inspectionCountFile) ReadAt(p []byte, offset int64) (int, error) {
	if f.onRead != nil {
		f.onRead(offset, len(p))
	}
	n, err := f.File.ReadAt(p, offset)
	*f.reads = append(*f.reads, inspectionRead{f.Name(), offset, n})
	return n, err
}
func (f *inspectionCountFile) Close() error {
	*f.closed++
	return f.File.Close()
}

func TestModelSettingsFactsReadAccounting(t *testing.T) {
	for _, size := range []int64{4096, 2 << 20} {
		t.Run(strconv.FormatInt(size, 10), func(t *testing.T) {
			f := newInspectionFixture(t, size)
			var reads []inspectionRead
			opened, closed := 0, 0
			f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
				file, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				opened++
				return &inspectionCountFile{File: file, reads: &reads, opened: &opened, closed: &closed}, nil
			}
			if result := inspectFixture(t, f); result.Status != "resolved" {
				t.Fatalf("inspection=%+v", result)
			}
			usage := f.ex.modelInspection.lastUsage
			var total uint64
			for _, read := range reads {
				total += uint64(read.bytes)
				if read.path != f.model {
					continue
				}
				header := read.offset+int64(read.bytes) <= f.directoryEnd
				sample := read.offset >= 0 && read.offset+int64(read.bytes) <= min(size, servingSampleSize) || read.offset >= size-servingSampleSize && read.offset+int64(read.bytes) <= size
				if size <= 2*servingSampleSize {
					sample = read.offset >= 0 && read.offset+int64(read.bytes) <= size
				}
				if !header && !sample {
					t.Fatalf("read outside header or permitted samples: %+v", read)
				}
			}
			if opened != closed || usage.HeaderBytes != uint64(f.directoryEnd) || total != usage.HeaderBytes+usage.SampleBytes+usage.SourceBytes {
				t.Fatalf("open=%d closed=%d usage=%+v actual=%d", opened, closed, usage, total)
			}
			preset, err := os.ReadFile(f.preset)
			if err != nil {
				t.Fatal(err)
			}
			wantSamples := 2*uint64(min(size, 2*servingSampleSize)) + 6*uint64(len(preset))
			if usage.SampleBytes != wantSamples || usage.SourceBytes != 2*uint64(len(preset)) {
				t.Fatalf("sample accounting=%+v wantSamples=%d", usage, wantSamples)
			}
			reads = nil
			if result := inspectFixture(t, f); result.Status != "resolved" {
				t.Fatalf("cached=%+v", result)
			}
			usage = f.ex.modelInspection.lastUsage
			if usage.HeaderBytes != 0 || usage.SampleBytes != wantSamples || opened != closed {
				t.Fatalf("cache hit skipped binding checks or read GGUF again: %+v", usage)
			}
		})
	}
}

func TestModelSettingsFactsCacheDetachedAndSourceRechecked(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	first := inspectFixture(t, f)
	if first.Status != "resolved" {
		t.Fatalf("first=%+v", first)
	}
	first.Facts.Dimensions["tensor.bytes"] = "999"
	first.Artifacts[0].Identity = "changed by caller"
	second := inspectFixture(t, f)
	if second.Facts.Dimensions["tensor.bytes"] != "128" || second.Artifacts[0].Identity == first.Artifacts[0].Identity {
		t.Fatal("caller mutated cached facts")
	}
	info, err := os.Stat(f.model)
	if err != nil {
		t.Fatal(err)
	}
	writeInspectionGGUF(t, f.model, "llama", 0.75, info.Size())
	if err := os.Chtimes(f.model, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	third := inspectFixture(t, f)
	if third.Status != "resolved" || f.ex.modelInspection.lastUsage.HeaderBytes == 0 || third.Artifacts[0].Identity == second.Artifacts[0].Identity {
		t.Fatalf("same-stat sample change reused facts: %+v", third)
	}
	if err := os.Remove(f.model); err != nil {
		t.Fatal(err)
	}
	if result := inspectFixture(t, f); result.Status == "resolved" {
		t.Fatal("cache hit hid missing source")
	}
}

func TestLinkedModelInspectionBinding(t *testing.T) {
	testReplacement := func(name, source, replace string, warm bool) {
		t.Run(name, func(t *testing.T) {
			f := newInspectionFixture(t, 4096)
			// Initial preset resolution samples, reads, resamples, then checks
			// its binding after sampling the artifact. Replace at the next open,
			// after final resolution has checked the original snapshot metadata.
			selector, selectorOpensBeforeFinal := f.preset, 4
			if source == "native cache" {
				f.st.plat.Runtime.Args = nil
				f.request.ModelID = "owner/repository:Q4_K_M"
				commit := strings.Repeat("a", 40)
				repo := llamaRepoDir(f.root, "owner/repository")
				selector = filepath.Join(repo, "refs", "main")
				writeTestFile(t, selector, []byte(commit))
				f.model = filepath.Join(repo, "snapshots", commit, "model-Q4_K_M.gguf")
				writeInspectionGGUF(t, f.model, "llama", 0.25, 4096)
				inspectionNativeMembership(t, f)
				// Initial HF resolution performs two complete checked ref reads.
				selectorOpensBeforeFinal = 6
			}
			if warm {
				if result := inspectFixture(t, f); result.Status != "resolved" {
					t.Fatalf("cache warmup=%+v", result)
				}
			}
			var reads []inspectionRead
			opened, closed, selectorOpens, forbidden := 0, 0, 0, 0
			replaced := false
			replacedPath := selector
			if replace == "artifact" {
				replacedPath = f.model
			}
			f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
				if path == selector {
					selectorOpens++
					if selectorOpens == selectorOpensBeforeFinal+1 {
						if err := os.Rename(replacedPath, replacedPath+".old"); err != nil {
							t.Fatal(err)
						}
						if replace == "artifact" {
							writeInspectionGGUF(t, replacedPath, "llama", 0.75, 4096)
						} else if source == "native cache" {
							writeTestFile(t, replacedPath, []byte(strings.Repeat("a", 40)))
						} else {
							writeTestFile(t, replacedPath, []byte("[exact-alias]\nmodel = main.gguf\n"))
						}
						replaced = true
					}
				}
				file, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				opened++
				return &inspectionCountFile{File: file, reads: &reads, opened: &opened, closed: &closed, onRead: func(int64, int) {
					if path == replacedPath && replaced {
						forbidden++
					}
				}}, nil
			}
			if result := inspectFixture(t, f); result.Status == "resolved" || !replaced || forbidden != 0 || len(result.Artifacts) != 0 {
				t.Fatalf("replacement binding: result=%+v replaced=%v forbidden=%d", result, replaced, forbidden)
			}
			if opened != closed {
				t.Fatalf("handles opened=%d closed=%d", opened, closed)
			}
			if warm && f.ex.modelInspection.lastUsage.HeaderBytes != 0 {
				t.Fatal("warm facts were decoded again")
			}
		})
	}
	testReplacement("cold preset replacement", "preset", "selector", false)
	testReplacement("warm preset replacement", "preset", "selector", true)
	testReplacement("cold preset artifact replacement", "preset", "artifact", false)
	testReplacement("warm preset artifact replacement", "preset", "artifact", true)
	testReplacement("cold ref replacement", "native cache", "selector", false)
	testReplacement("warm ref replacement", "native cache", "selector", true)
	testReplacement("cold cache artifact replacement", "native cache", "artifact", false)
	testReplacement("warm cache artifact replacement", "native cache", "artifact", true)

	testWithdrawal := func(name string, warm, beforeInitial bool) {
		t.Run(name, func(t *testing.T) {
			f := newInspectionFixture(t, 4096)
			f.st.plat.Runtime.Args = nil
			f.request.ModelID = "owner/repository:Q4_K_M"
			commit := strings.Repeat("a", 40)
			repo := llamaRepoDir(f.root, "owner/repository")
			writeTestFile(t, filepath.Join(repo, "refs", "main"), []byte(commit))
			writeInspectionGGUF(t, filepath.Join(repo, "snapshots", commit, "model-Q4_K_M.gguf"), "llama", 0.25, 4096)
			inspectionNativeMembership(t, f)
			if warm {
				if result := inspectFixture(t, f); result.Status != "resolved" {
					t.Fatalf("cache warmup=%+v", result)
				}
			}
			native := f.ex.client.Transport
			calls, forbidden, opened, closed := 0, 0, 0, 0
			withdrawn := false
			f.ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
				calls++
				if beforeInitial || calls == 2 {
					withdrawn = true
					return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"models":[]}`))}, nil
				}
				return native.RoundTrip(request)
			})
			var reads []inspectionRead
			f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
				file, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				opened++
				return &inspectionCountFile{File: file, reads: &reads, opened: &opened, closed: &closed, onRead: func(int64, int) {
					if withdrawn {
						forbidden++
					}
				}}, nil
			}
			result := inspectFixture(t, f)
			if result.Status == "resolved" || !withdrawn || forbidden != 0 || len(result.Artifacts) != 0 || opened != closed {
				t.Fatalf("withdrawn recognition: result=%+v calls=%d withdrawn=%v forbidden=%d opens=%d closes=%d", result, calls, withdrawn, forbidden, opened, closed)
			}
			if beforeInitial && opened != 0 {
				t.Fatalf("unrecognized source opened %d files", opened)
			}
			if !beforeInitial && warm && f.ex.modelInspection.lastUsage.HeaderBytes != 0 {
				t.Fatal("warm facts were decoded again")
			}
		})
	}
	testWithdrawal("cold initial recognition withdrawal", false, true)
	testWithdrawal("warm initial recognition withdrawal", true, true)
	testWithdrawal("cold final recognition withdrawal", false, false)
	testWithdrawal("warm final recognition withdrawal", true, false)

	for _, warm := range []bool{false, true} {
		name := "cold native exit during final resolution"
		if warm {
			name = "warm native exit during final resolution"
		}
		t.Run(name, func(t *testing.T) {
			f := newInspectionFixture(t, 4096)
			f.st.plat.Runtime.Args = nil
			f.request.ModelID = "owner/repository:Q4_K_M"
			commit := strings.Repeat("a", 40)
			repo := llamaRepoDir(f.root, "owner/repository")
			writeTestFile(t, filepath.Join(repo, "refs", "main"), []byte(commit))
			f.model = filepath.Join(repo, "snapshots", commit, "model-Q4_K_M.gguf")
			writeInspectionGGUF(t, f.model, "llama", 0.25, 4096)
			inspectionNativeMembership(t, f)
			if warm && inspectFixture(t, f).Status != "resolved" {
				t.Fatal("cache warmup failed")
			}
			opens, target := 0, 3
			if warm {
				target = 2
			}
			f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
				if path == f.model {
					opens++
					if opens == target {
						f.st.mu.Lock()
						f.st.running, f.st.runID = false, ""
						f.st.mu.Unlock()
					}
				}
				return os.Open(path)
			}
			if result := inspectFixture(t, f); opens < target || result.Status == "resolved" || len(result.Artifacts) != 0 {
				t.Fatalf("native exit: opens=%d result=%+v", opens, result)
			}
		})
	}

	t.Run("replacement at decoder open", func(t *testing.T) {
		f := newInspectionFixture(t, 4096)
		var reads []inspectionRead
		opened, closed, modelOpens, forbidden := 0, 0, 0, 0
		replaced := false
		f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
			if path == f.model {
				modelOpens++
				if modelOpens == 2 {
					if err := os.Rename(path, path+".old"); err != nil {
						t.Fatal(err)
					}
					writeInspectionGGUF(t, path, "llama", 0.75, 4096)
					replaced = true
				}
			}
			file, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			opened++
			return &inspectionCountFile{File: file, reads: &reads, opened: &opened, closed: &closed, onRead: func(int64, int) {
				if path == f.model && replaced {
					forbidden++
				}
			}}, nil
		}
		result := inspectFixture(t, f)
		if result.Status == "resolved" || !replaced || forbidden != 0 || opened != closed || len(f.ex.modelInspection.entries) != 0 {
			t.Fatalf("decoder replacement: result=%+v replaced=%v forbidden=%d opens=%d closes=%d", result, replaced, forbidden, opened, closed)
		}
	})
}

func TestModelSettingsFactsCacheLRUExpiryAndByteCap(t *testing.T) {
	now := time.Unix(1000, 0)
	s := modelInspectionState{now: func() time.Time { return now }}
	facts := gguffacts.Facts{Shards: []gguffacts.Shard{{Role: gguffacts.RoleMain, Numeric: map[string]gguffacts.Numeric{"llama.block_count": {Type: 4, Bits: 4}}}}, Usage: gguffacts.Usage{RetainedBytes: 1024}}
	for i := 0; i < modelInspectionCacheEntries; i++ {
		s.remember(strconv.Itoa(i), strconv.Itoa(i), facts)
	}
	clone, ok := s.cached("0", "0")
	if !ok {
		t.Fatal("initial cache miss")
	}
	clone.Shards[0].Numeric["llama.block_count"] = gguffacts.Numeric{Type: 4, Bits: 999}
	s.remember("new", "new", facts)
	if _, ok := s.cached("1", "1"); ok {
		t.Fatal("LRU entry survived eviction")
	}
	retained, ok := s.cached("0", "0")
	if !ok || retained.Shards[0].Numeric["llama.block_count"].Bits != 4 {
		t.Fatal("LRU hit lost or clone mutated cache")
	}
	now = now.Add(modelInspectionCacheTTL)
	if _, ok := s.cached("0", "0"); ok || len(s.entries) != 0 || s.bytes != 0 {
		t.Fatal("expired cache retained entries")
	}
	facts.Usage.RetainedBytes = modelInspectionCacheBytes / 2
	s.remember("large-one", "one", facts)
	s.remember("large-two", "two", facts)
	if len(s.entries) != 1 || s.bytes > modelInspectionCacheBytes || s.entries[0].key != "large-two" {
		t.Fatal("retained byte cap ignored entry overhead")
	}
	facts.Usage.RetainedBytes = modelInspectionCacheBytes
	s.remember("oversized", "oversized", facts)
	if len(s.entries) != 1 {
		t.Fatal("oversized facts entered cache")
	}
	facts.Usage.RetainedBytes = 1024
	s.remember("changed", "two", facts)
	if _, ok := s.cached("changed", "two"); !ok || len(s.entries) != 1 {
		t.Fatal("changed source did not invalidate older facts")
	}
}

func TestModelSettingsArtifactsMainDraftAndCompanion(t *testing.T) {
	t.Run("ordered configured main and draft shards", func(t *testing.T) {
		f := newInspectionFixture(t, 4096)
		for _, name := range []string{"main-00001-of-00002.gguf", "main-00002-of-00002.gguf"} {
			writeInspectionGGUF(t, filepath.Join(f.root, name), "llama", 0.25, 4096)
		}
		writeInspectionGGUF(t, filepath.Join(f.root, "draft.gguf"), "qwen2", 0.99, 4096)
		writeTestFile(t, f.preset, []byte("[exact-alias]\nmodel = main-00001-of-00002.gguf\nmodel-draft = draft.gguf\n"))
		result := inspectFixture(t, f)
		if result.Status != "resolved" || len(result.Artifacts) != 3 || result.Artifacts[0].Role != "model" || result.Artifacts[1].Role != "model" || result.Artifacts[2].Role != "draft" || result.Facts.Architecture != "llama" || result.Facts.Dimensions["tensor.bytes"] != "256" {
			t.Fatalf("roles/facts=%+v", result)
		}
		for _, option := range result.Options {
			if option.Field == "generation.temperature" && string(option.Default) != "0.25" {
				t.Fatal("draft sampler replaced main recommendation")
			}
		}
	})
	t.Run("canonical cache known companion", func(t *testing.T) {
		f := newInspectionFixture(t, 4096)
		f.st.plat.Runtime.Args = nil
		f.request.ModelID = "owner/repository:Q4_K_M"
		inspectionNativeMembership(t, f)
		commit := strings.Repeat("a", 40)
		repo := llamaRepoDir(f.root, "owner/repository")
		writeTestFile(t, filepath.Join(repo, "refs", "main"), []byte(commit))
		writeInspectionGGUF(t, filepath.Join(repo, "snapshots", commit, "model-Q4_K_M.gguf"), "llama", 0.25, 4096)
		writeInspectionGGUF(t, filepath.Join(repo, "snapshots", commit, "mmproj-f16.gguf"), "clip", 0.99, 4096)
		result := inspectFixture(t, f)
		if result.Status != "resolved" || len(result.Artifacts) != 2 || result.Artifacts[1].Role != "companion" || result.Facts.Architecture != "llama" || result.Facts.Dimensions["tensor.bytes"] != "128" {
			t.Fatalf("companion=%+v", result)
		}
	})
}

func TestModelSettingsArtifactsUnsafeMissingAndAmbiguous(t *testing.T) {
	for _, scenario := range []string{"missing shard", "duplicate main/draft", "partial directory", "missing exact alias", "ambiguous receipt"} {
		t.Run(scenario, func(t *testing.T) {
			f := newInspectionFixture(t, 4096)
			switch scenario {
			case "missing shard":
				writeInspectionGGUF(t, filepath.Join(f.root, "main-00001-of-00002.gguf"), "llama", 0.25, 4096)
				writeTestFile(t, f.preset, []byte("[exact-alias]\nmodel = main-00001-of-00002.gguf\n"))
			case "duplicate main/draft":
				writeTestFile(t, f.preset, []byte("[exact-alias]\nmodel = main.gguf\nmodel-draft = main.gguf\n"))
			case "partial directory":
				writeInspectionGGUF(t, filepath.Join(f.root, "pull.partial", "main.gguf"), "llama", 0.25, 4096)
				writeTestFile(t, f.preset, []byte("[exact-alias]\nmodel = pull.partial/main.gguf\n"))
			case "missing exact alias":
				f.request.ModelID = "exact-Alias"
			case "ambiguous receipt":
				list := modelFileList{Engine: "llamacpp", Model: f.request.ModelID, Layout: llamaArtifactsLayout, Files: []modelFile{{Role: "model", Path: "model/copied.gguf", Size: 4096, SHA256: strings.Repeat("b", 64)}}}
				id, err := artifactExportID(list)
				if err != nil {
					t.Fatal(err)
				}
				list.ExportID = id
				base := copyAliasDir(f.root, f.request.ModelID)
				writeInspectionGGUF(t, filepath.Join(base, id, "model", "copied.gguf"), "llama", 0.25, 4096)
				data, err := json.Marshal(copyReceipt{Version: 1, List: list})
				if err != nil {
					t.Fatal(err)
				}
				writeTestFile(t, filepath.Join(base, "copy.json"), data)
			}
			result := inspectFixture(t, f)
			if result.Status == "resolved" || len(result.Artifacts) != 0 || f.ex.modelInspection.lastUsage.HeaderBytes != 0 {
				t.Fatalf("unsafe source decoded: %+v", result)
			}
			if scenario == "ambiguous receipt" && result.Status != "ambiguous" {
				t.Fatalf("ambiguous state=%s", result.Status)
			}
		})
	}
}

func TestModelSettingsArtifactsDownloadFencesAndNilOwners(t *testing.T) {
	for _, scenario := range []string{"active", "unsafe", "recovery", "writer", "nil registry", "nil downloads", "operation mutation"} {
		t.Run(scenario, func(t *testing.T) {
			f := newInspectionFixture(t, 4096)
			key, err := downloadKey("llamacpp", f.request.ModelID, "exact")
			if err != nil {
				t.Fatal(err)
			}
			switch scenario {
			case "active":
				f.ex.downloads.active[key] = &modelDownloadAttempt{}
			case "unsafe":
				f.ex.downloads.unsafe[key] = modelDownloadMarker{}
			case "recovery":
				f.ex.downloads.recoveryRequired = true
			case "writer":
				f.ex.downloads.writers["llamacpp"] = true
			case "nil registry":
				f.ex.reg = nil
			case "nil downloads":
				f.ex.downloads = nil
			case "operation mutation":
				f.st.opMu.Lock()
				defer f.st.opMu.Unlock()
			}
			f.ex.modelInspection.open = func(string) (servingArtifactFile, error) {
				t.Error("fenced inspection opened a file")
				return nil, errors.New("file read forbidden")
			}
			result := inspectFixture(t, f)
			if result.Status != "busy" && result.Status != "unavailable" {
				t.Fatalf("fenced=%+v", result)
			}
		})
	}
}

func TestModelSettingsArtifactsSharedSourcesUseEngineWideFence(t *testing.T) {
	test := func(name, role string, active bool) {
		t.Run(name, func(t *testing.T) {
			f := newInspectionFixture(t, 4096)
			canonical := "owner/repository:Q4_K_M"
			commit := strings.Repeat("a", 40)
			repo := llamaRepoDir(f.root, "owner/repository")
			writeTestFile(t, filepath.Join(repo, "refs", "main"), []byte(commit))
			shared := filepath.Join(repo, "snapshots", commit, "model-Q4_K_M.gguf")
			writeInspectionGGUF(t, shared, "llama", 0.25, 4096)
			switch role {
			case "model":
				writeTestFile(t, f.preset, []byte("[exact-alias]\nmodel = "+filepath.ToSlash(shared)+"\n"))
			case "draft":
				writeTestFile(t, f.preset, []byte("[exact-alias]\nmodel = main.gguf\nmodel-draft = "+filepath.ToSlash(shared)+"\n"))
			case "companion":
				f.st.plat.Runtime.Args = nil
				f.request.ModelID = canonical
				inspectionNativeMembership(t, f)
				shared = filepath.Join(repo, "snapshots", commit, "mmproj-f16.gguf")
				writeInspectionGGUF(t, shared, "clip", 0.25, 4096)
				canonical = "shared-projector-alias"
			}
			if result := inspectFixture(t, f); result.Status != "resolved" {
				t.Fatalf("shared-source positive control: status=%s reason=%s", result.Status, result.Reason)
			}
			key, err := downloadKey("llamacpp", canonical, "exact")
			if err != nil {
				t.Fatal(err)
			}
			if active {
				f.ex.downloads.active[key] = &modelDownloadAttempt{}
			} else {
				f.ex.downloads.unsafe[key] = modelDownloadMarker{}
			}
			f.ex.modelInspection.open = func(string) (servingArtifactFile, error) {
				t.Error("alternate-key fence allowed artifact I/O")
				return nil, errors.New("read forbidden")
			}
			result := inspectFixture(t, f)
			if result.Status != "busy" || len(result.Artifacts) != 0 || f.ex.modelInspection.lastUsage.HeaderBytes != 0 {
				t.Fatalf("alternate-key fence: status=%s artifacts=%d", result.Status, len(result.Artifacts))
			}
		})
	}
	test("active canonical model behind alias", "model", true)
	test("unsafe canonical model behind alias", "model", false)
	test("active shared draft", "draft", true)
	test("unsafe shared draft", "draft", false)
	test("active shared companion", "companion", true)
	test("unsafe shared companion", "companion", false)
}

func TestModelSettingsArtifactsReceiptExportIgnoresUnrelatedServingDeclaration(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	f.request.ModelID = "receipt-only"
	// This declaration belongs to another preset alias and has no approval.
	// It must not be compiled to re-export an independently committed receipt.
	f.st.plat.Runtime.Serving = &engines.ServingConfig{Version: 1, Models: []engines.ServingModel{{ModelID: "exact-alias"}}}
	data, err := os.ReadFile(f.model)
	if err != nil {
		t.Fatal(err)
	}
	list := modelFileList{Engine: "llamacpp", Model: f.request.ModelID, Layout: llamaArtifactsLayout, Files: []modelFile{{Role: "model", Path: "model/copied.gguf", Size: int64(len(data)), SHA256: sha256Hex(data)}}}
	id, err := artifactExportID(list)
	if err != nil {
		t.Fatal(err)
	}
	list.ExportID = id
	base := copyAliasDir(f.root, f.request.ModelID)
	writeTestFile(t, filepath.Join(base, id, "model", "copied.gguf"), data)
	receipt, err := json.Marshal(copyReceipt{Version: 1, List: list})
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(base, "copy.json"), receipt)
	if result := inspectFixture(t, f); result.Status != "resolved" {
		t.Fatalf("receipt inspection: status=%s reason=%s", result.Status, result.Reason)
	}
	exported, err := f.ex.modelSourceFiles(context.Background(), "llamacpp", f.request.ModelID)
	if err != nil {
		t.Fatal(err)
	}
	if exported.Model != f.request.ModelID || len(exported.Files) != 1 || exported.Files[0].asserted != list.Files[0].SHA256 || exported.Files[0].configured != filepath.Join(base, id, "model", "copied.gguf") {
		t.Fatalf("receipt export did not retain committed source: %+v", exported)
	}
}

func TestModelSettingsFactsConcurrentBusyAndCooperativeCancel(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	entered, continueRead := make(chan struct{}), make(chan struct{})
	var once sync.Once
	opened, closed := 0, 0
	var reads []inspectionRead
	f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		opened++
		return &inspectionCountFile{File: file, reads: &reads, opened: &opened, closed: &closed, onRead: func(int64, int) {
			once.Do(func() {
				close(entered)
				select {
				case <-continueRead:
				case <-ctx.Done():
				}
			})
		}}, nil
	}
	done := make(chan error, 1)
	go func() { _, err := f.ex.InspectModel(ctx, f.request); done <- err }()
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("inspection never entered bounded I/O")
	}
	result := inspectFixture(t, f)
	if result.Status != "busy" {
		t.Fatalf("concurrent inspection queued: %+v", result)
	}
	// The existing reader gate excludes an actual writer throughout the scan.
	if release, err := f.ex.downloads.acquireWrite("llamacpp"); err == nil {
		release()
		t.Error("reader gate released before inspection finished")
	}
	cancel()
	close(continueRead)
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation=%v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("inspection did not cooperate with cancellation")
	}
	if opened != closed || f.ex.modelInspection.active.Load() || len(f.ex.modelInspection.entries) != 0 {
		t.Fatalf("cancel leaked files/scan/cache: open=%d close=%d", opened, closed)
	}
	if release, err := f.ex.downloads.acquireWrite("llamacpp"); err != nil {
		t.Fatalf("reader gate leaked: %v", err)
	} else {
		release()
	}
}

func TestModelSettingsArtifactsSourceAndConfigurationRaces(t *testing.T) {
	for _, scenario := range []string{"generation", "launch", "preset", "gate source", "gate revision", "gate unsafe", "artifact replacement"} {
		t.Run(scenario, func(t *testing.T) {
			f := newInspectionFixture(t, 4096)
			// Populate the cache first; racing source checks must run on hits too.
			if result := inspectFixture(t, f); result.Status != "resolved" {
				t.Fatalf("warmup=%+v", result)
			}
			var once sync.Once
			modelOpens := 0
			f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
				if path == f.model {
					modelOpens++
				}
				if path == f.model && modelOpens == 2 {
					once.Do(func() {
						switch scenario {
						case "generation":
							f.st.mu.Lock()
							f.st.gen++
							f.st.mu.Unlock()
						case "launch":
							f.st.opMu.Lock()
							f.st.mu.Lock()
							f.st.plat.Runtime.Args = append(f.st.plat.Runtime.Args, "--threads", "3")
							f.st.mu.Unlock()
							f.st.opMu.Unlock()
						case "preset":
							writeTestFile(t, f.preset, []byte("# changed\n[exact-alias]\nmodel = main.gguf\n"))
						case "gate source":
							f.ex.downloads.mu.Lock()
							f.ex.downloads.source = "replaced"
							f.ex.downloads.mu.Unlock()
						case "gate revision":
							f.ex.downloads.mu.Lock()
							f.ex.downloads.revision++
							f.ex.downloads.mu.Unlock()
						case "gate unsafe":
							key, err := downloadKey("llamacpp", f.request.ModelID, "exact")
							if err != nil {
								t.Fatal(err)
							}
							f.ex.downloads.mu.Lock()
							f.ex.downloads.unsafe[key] = modelDownloadMarker{}
							f.ex.downloads.mu.Unlock()
						case "artifact replacement":
							if err := os.Rename(f.model, f.model+".old"); err != nil {
								t.Fatal(err)
							}
							writeInspectionGGUF(t, f.model, "llama", 0.75, 4096)
						}
					})
				}
				return os.Open(path)
			}
			if result := inspectFixture(t, f); result.Status == "resolved" {
				t.Fatalf("race accepted: %+v", result)
			}
		})
	}
}

func TestModelSettingsArtifactsScanAndArtifactBudgets(t *testing.T) {
	t.Run("directory entry budget before visit", func(t *testing.T) {
		root := t.TempDir()
		writeTestFile(t, filepath.Join(root, "one"), []byte("x"))
		writeTestFile(t, filepath.Join(root, "two"), []byte("x"))
		scan := modelArtifactScan{entries: modelArtifactScanEntries - 1}
		visited := 0
		err := scan.directory(context.Background(), root, func(os.DirEntry) error { visited++; return nil })
		if !errors.Is(err, errModelArtifactLimit) || visited != 1 || scan.entries != modelArtifactScanEntries {
			t.Fatalf("scan=%+v visited=%d err=%v", scan, visited, err)
		}
	})
	t.Run("shard cap before allocation", func(t *testing.T) {
		paths, _, err := llamaConsumedPaths(filepath.Join(t.TempDir(), "models.ini"), map[string]string{"model": "main-00001-of-01025.gguf"}, "alias", maxCopyArtifacts)
		if err == nil || len(paths) != 0 {
			t.Fatal("excessive shard set allocated")
		}
	})
	t.Run("cancellation before entry visit", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		scan := modelArtifactScan{}
		if err := scan.directory(ctx, t.TempDir(), func(os.DirEntry) error { t.Error("cancelled visit"); return nil }); !errors.Is(err, context.Canceled) {
			t.Fatalf("scan cancellation=%v", err)
		}
	})
}

func TestModelSettingsFactsUnknownSamplersAndMainConflicts(t *testing.T) {
	for _, sampler := range []gguffacts.Numeric{{Type: 12, Bits: math.Float64bits(-1)}, {Type: 12, Bits: math.Float64bits(math.NaN())}, {Type: 4, Array: []uint64{1}}, {Type: 7, Bits: 1}} {
		facts := gguffacts.Facts{Shards: []gguffacts.Shard{{Role: gguffacts.RoleMain, Architecture: "llama", Numeric: map[string]gguffacts.Numeric{"general.sampling.temp": sampler}, Templates: []gguffacts.Template{{Key: "tokenizer.chat_template"}}}}}
		result, recommendations := projectInspectionFacts(facts)
		if len(recommendations) != 0 || result.ContextTokens != nil || result.LayerCount != nil || result.TemplatePresent == nil || !*result.TemplatePresent {
			t.Fatalf("invented missing facts/recommendations: %+v %+v", result, recommendations)
		}
	}
	facts := gguffacts.Facts{Shards: []gguffacts.Shard{
		{Role: gguffacts.RoleMain, Architecture: "llama", Numeric: map[string]gguffacts.Numeric{"llama.block_count": {Type: 4, Bits: 4}}},
		{Role: gguffacts.RoleMain, Architecture: "qwen2", Numeric: map[string]gguffacts.Numeric{"qwen2.block_count": {Type: 4, Bits: 8}}},
	}}
	result, options := projectInspectionFacts(facts)
	if result.Architecture != "" || result.LayerCount != nil || len(options) != 0 {
		t.Fatalf("conflicting main architecture leaked: %+v", result)
	}
}

func TestModelSettingsArtifactsSnapshotClonesStoredLoads(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	layers := int32(3)
	f.st.plat.Runtime.ModelLoads = []modelconfig.ModelLoad{{ModelID: "exact-alias", Requested: modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "layers", Layers: &layers}}, Resolved: &modelconfig.ResolvedLoad{Artifacts: []modelconfig.ArtifactBinding{{Path: f.model}}}}}
	snapshot, err := inspectionSnapshot(f.st, "llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	f.st.plat.Runtime.ModelLoads[0].Resolved.Artifacts[0].Path = "mutated"
	layers = 99
	load := snapshot.state.plat.Runtime.ModelLoads[0]
	if load.Resolved.Artifacts[0].Path != f.model || *load.Requested.GPU.Layers != 3 {
		t.Fatal("stored model loads were shallow copied")
	}
}

func TestModelSettingsFactsProjectionBoundedUnsignedDimensions(t *testing.T) {
	array := make([]uint64, 4096)
	for i := range array {
		array[i] = uint64(i)
	}
	facts := gguffacts.Facts{Shards: []gguffacts.Shard{{Role: gguffacts.RoleMain, Architecture: "llama", Numeric: map[string]gguffacts.Numeric{
		"llama.attention.head_count": {Type: 10, Array: array}, "llama.context_length": {Type: 10, Bits: math.MaxUint64}, "llama.block_count": {Type: 11, Bits: math.MaxUint64},
	}}}}
	result, _ := projectInspectionFacts(facts)
	if len(result.Dimensions) > 256 || result.ContextTokens != nil || result.LayerCount != nil {
		t.Fatalf("unbounded or narrowed dimensions: %+v", result)
	}
	for key, value := range result.Dimensions {
		if _, err := strconv.ParseUint(value, 10, 64); err != nil {
			t.Fatalf("dimension %s=%s: %v", key, value, err)
		}
	}
}

func TestModelSettingsArtifactsConfiguredSourcePreservedByCopyResolver(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	before, err := os.ReadFile(f.model)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := inspectionSnapshot(f.st, "llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	read, _, err := resolveLlamaArtifacts(context.Background(), snapshot.state, snapshot.command, snapshot.store, f.request.ModelID, servingFileAccess{})
	if err != nil {
		t.Fatal(err)
	}
	copyList, err := f.ex.modelSourceFiles(context.Background(), "llamacpp", f.request.ModelID)
	if err != nil {
		t.Fatal(err)
	}
	if read.binding != copyList.binding || len(read.Files) != len(copyList.Files) || copyList.ExportID != "" {
		t.Fatal("copy and inspection diverged before hash ownership")
	}
	for i, file := range read.Files {
		copied := copyList.Files[i]
		if !sameCopyFile(file.info, copied.info) {
			t.Fatalf("artifact %d has different file identity, size or modification time", i)
		}
		// os.FileInfo may include access times and platform-internal state that
		// change during reads. Compare its source identity above, not its internals.
		file.info, copied.info = nil, nil
		if !reflect.DeepEqual(file, copied) || copied.SHA256 != "" {
			t.Fatalf("artifact %d differs before hash ownership", i)
		}
	}
	after, err := os.ReadFile(f.model)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("read resolver changed source")
	}
}

func TestModelSettingsFactsCacheSourceContextChanges(t *testing.T) {
	for _, source := range []string{"preset bytes", "download incarnation", "engine generation"} {
		t.Run(source, func(t *testing.T) {
			f := newInspectionFixture(t, 4096)
			if result := inspectFixture(t, f); result.Status != "resolved" {
				t.Fatalf("first=%+v", result)
			}
			switch source {
			case "preset bytes":
				writeTestFile(t, f.preset, []byte("# fresh source\n[exact-alias]\nmodel = main.gguf\n"))
			case "download incarnation":
				f.ex.downloads.source = "fresh-incarnation"
			case "engine generation":
				f.st.gen++
			}
			if result := inspectFixture(t, f); result.Status != "resolved" || f.ex.modelInspection.lastUsage.HeaderBytes == 0 || len(f.ex.modelInspection.entries) != 1 {
				t.Fatalf("source context cache invalidation=%+v", result)
			}
		})
	}
}

func TestModelSettingsArtifactsCanonicalCacheMissingShard(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	f.st.plat.Runtime.Args = nil
	f.request.ModelID = "owner/repository:Q4_K_M"
	inspectionNativeMembership(t, f)
	commit := strings.Repeat("a", 40)
	repo := llamaRepoDir(f.root, "owner/repository")
	writeTestFile(t, filepath.Join(repo, "refs", "main"), []byte(commit))
	writeInspectionGGUF(t, filepath.Join(repo, "snapshots", commit, "model-Q4_K_M-00001-of-00003.gguf"), "llama", 0.25, 4096)
	writeInspectionGGUF(t, filepath.Join(repo, "snapshots", commit, "model-Q4_K_M-00003-of-00003.gguf"), "llama", 0.25, 4096)
	if result := inspectFixture(t, f); result.Status == "resolved" || f.ex.modelInspection.lastUsage.HeaderBytes != 0 {
		t.Fatalf("noncontiguous cache decoded: %+v", result)
	}
}

func TestModelSettingsFactsDescriptorRangesUseExistingPolicy(t *testing.T) {
	facts := gguffacts.Facts{Shards: []gguffacts.Shard{{Role: gguffacts.RoleMain, Architecture: "llama", Numeric: map[string]gguffacts.Numeric{
		"general.sampling.top_p": {Type: 12, Bits: math.Float64bits(2)}, "general.sampling.top_k": {Type: 10, Bits: math.MaxUint64}, "general.sampling.penalty_repeat": {Type: 12, Bits: math.Float64bits(0)}, "general.sampling.min_p": {Type: 12, Bits: math.Float64bits(0.125)},
	}}}}
	_, recommendations := projectInspectionFacts(facts)
	options := mergeModelInspectionOptions(recommendations, modelSettingsExtraOptions())
	defaults := map[string]string{}
	for _, option := range options {
		if strings.HasPrefix(option.DefaultSource, "embedded GGUF") {
			defaults[option.Field] = string(option.Default)
		}
	}
	if !reflect.DeepEqual(defaults, map[string]string{"generation.min_p": "0.125"}) {
		t.Fatalf("unsupported sampler values became defaults: %+v", defaults)
	}
}

func TestModelSettingsArtifactsDiscoveryLimitIsStructured(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	writeTestFile(t, f.preset, []byte("[exact-alias]\nmodel = main-00001-of-01025.gguf\n"))
	result := inspectFixture(t, f)
	if result.Status != "unknown" || len(result.Artifacts) != 0 || f.ex.modelInspection.lastUsage.HeaderBytes != 0 {
		t.Fatalf("discovery limit=%+v", result)
	}
}

func TestModelSettingsArtifactsStoredBindingsReenterResolver(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	f.st.plat.Runtime.Args = nil
	binding, err := servingFileBinding(f.model)
	if err != nil {
		t.Fatal(err)
	}
	f.st.plat.Runtime.ModelLoads = []modelconfig.ModelLoad{{ModelID: f.request.ModelID, Resolved: &modelconfig.ResolvedLoad{
		PerSlotContext: 4096, TotalContext: 4096, Slots: 1, GPU: modelconfig.GPU{Mode: "cpu"}, NativeBuildID: "synthetic-build",
		SourceInputDigest: strings.Repeat("a", 64), PresetDigest: strings.Repeat("b", 64),
		Artifacts: []modelconfig.ArtifactBinding{{Role: "model", Path: binding.SourcePath, ResolvedTarget: binding.ResolvedTarget, Size: strconv.FormatInt(binding.SizeBytes, 10), MtimeNS: binding.MtimeNS, SampleSHA256: binding.SampleSHA256}},
	}}}
	if err := f.st.plat.Runtime.ModelLoads[0].Validate(); err != nil {
		t.Fatalf("typed fixture: %v", err)
	}
	result := inspectFixture(t, f)
	if result.Status != "resolved" || len(result.Artifacts) != 1 || result.Facts.Architecture != "llama" {
		t.Fatalf("typed binding=%+v", result)
	}
	writeInspectionGGUF(t, f.model, "llama", 0.5, 4096)
	if result := inspectFixture(t, f); result.Status == "resolved" {
		t.Fatal("stale stored binding was silently renewed")
	}
}

func TestModelSettingsArtifactsCacheSymlinkConfinement(t *testing.T) {
	for _, scenario := range []string{"confined blob", "escaping blob"} {
		t.Run(scenario, func(t *testing.T) {
			f := newInspectionFixture(t, 4096)
			f.st.plat.Runtime.Args = nil
			f.request.ModelID = "owner/repository:Q4_K_M"
			inspectionNativeMembership(t, f)
			repo, commit := llamaRepoDir(f.root, "owner/repository"), strings.Repeat("a", 40)
			writeTestFile(t, filepath.Join(repo, "refs", "main"), []byte(commit))
			target := filepath.Join(repo, "blobs", strings.Repeat("b", 64))
			if scenario == "escaping blob" {
				target = filepath.Join(t.TempDir(), "external.gguf")
			}
			writeInspectionGGUF(t, target, "llama", 0.25, 4096)
			link := filepath.Join(repo, "snapshots", commit, "model-Q4_K_M.gguf")
			if err := os.MkdirAll(filepath.Dir(link), 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(target, link); err != nil {
				t.Fatal(err)
			}
			result := inspectFixture(t, f)
			if result.Status != "resolved" {
				t.Fatalf("selected HF blob refused: %+v", result)
			}
			if scenario == "confined blob" {
				if result.Artifacts[0].Identity == filepath.Base(target) || !strings.Contains(result.Artifacts[0].Assurance, "not full") {
					t.Fatal("asserted blob name became full verification")
				}
			} else {
				_, client := modelFileServer(t, f.ex, nil)
				ctx := context.Background()
				list, err := client.modelFiles(ctx, "llamacpp", f.request.ModelID, nil)
				if err != nil {
					t.Fatal(err)
				}
				if len(list.Files) != 1 {
					t.Fatalf("files=%v", list.Files)
				}
				body, _, err := client.modelFile(ctx, "llamacpp", f.request.ModelID, list.Files[0].Path, list.ExportID, 0)
				if err != nil {
					t.Fatal(err)
				}
				got, err := io.ReadAll(body)
				if err != nil {
					t.Fatal(err)
				}
				if err := body.Close(); err != nil {
					t.Fatal(err)
				}
				want, err := os.ReadFile(target)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got, want) {
					t.Fatal("outside-bound HTTP bytes differ")
				}
				counter := &linkedHashCounts{}
				server := &controlServer{exec: f.ex, access: servingFileAccess{open: func(path string) (servingArtifactFile, error) {
					file, err := os.Open(path)
					if err != nil {
						return nil, err
					}
					return &linkedHashFile{File: file, counts: counter, forbidden: true}, nil
				}}}
				query := url.Values{"engine": {"llamacpp"}, "model": {f.request.ModelID}, "path": {"unlisted.gguf"}, "exportId": {list.ExportID}}
				response := httptest.NewRecorder()
				server.handleModelFile(response, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+query.Encode(), nil))
				if response.Code != http.StatusNotFound || counter.bytes.Load() != 0 {
					t.Fatalf("unlisted status=%d reads=%d", response.Code, counter.bytes.Load())
				}
				replacement := filepath.Join(t.TempDir(), "replacement.gguf")
				writeInspectionGGUF(t, replacement, "llama", 0.75, 4096)
				if err := os.Remove(link); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(replacement, link); err != nil {
					t.Fatal(err)
				}
				query.Set("path", list.Files[0].Path)
				response = httptest.NewRecorder()
				server.handleModelFile(response, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+query.Encode(), nil))
				if response.Code != http.StatusConflict || counter.bytes.Load() != 0 {
					t.Fatalf("stale export status=%d reads=%d", response.Code, counter.bytes.Load())
				}
			}
		})
	}
}

func TestModelSettingsArtifactsDuplicateAndCaseCollision(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	one, two := filepath.Join(f.root, "one", "model.gguf"), filepath.Join(f.root, "two", "MODEL.gguf")
	writeInspectionGGUF(t, one, "llama", 0.25, 4096)
	writeInspectionGGUF(t, two, "llama", 0.25, 4096)
	makeFile := func(path, name string) modelFile {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		return modelFile{Role: "model", Path: "model/" + name, configured: path, abs: path, info: info, Size: info.Size()}
	}
	list := modelFileList{Files: []modelFile{makeFile(one, "model.gguf"), makeFile(two, "MODEL.gguf")}}
	if err := validateResolvedArtifacts(list); !errors.Is(err, errNoArtifactBinding) {
		t.Fatalf("case collision=%v", err)
	}
	list.Files[1] = list.Files[0]
	if err := validateResolvedArtifacts(list); !errors.Is(err, errNoArtifactBinding) {
		t.Fatalf("duplicate=%v", err)
	}
}

func TestModelSettingsFactsDecodeFailureClosesFiles(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	writeTestFile(t, f.model, bytes.Repeat([]byte("invalidGGUF"), 100))
	opened, closed := 0, 0
	var reads []inspectionRead
	f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		opened++
		return &inspectionCountFile{File: file, reads: &reads, opened: &opened, closed: &closed}, nil
	}
	result := inspectFixture(t, f)
	if result.Status != "unknown" || opened != closed || len(f.ex.modelInspection.entries) != 0 {
		t.Fatalf("decode failure=%+v opens=%d closes=%d", result, opened, closed)
	}
}

func TestModelSettingsArtifactsPresetAndReceiptReadLimits(t *testing.T) {
	for _, source := range []string{"preset", "receipt"} {
		t.Run(source, func(t *testing.T) {
			f := newInspectionFixture(t, 4096)
			path := f.preset
			if source == "receipt" {
				path = filepath.Join(copyAliasDir(f.root, f.request.ModelID), "copy.json")
				f.st.plat.Runtime.Args = nil
			}
			if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
				t.Fatal(err)
			}
			file, err := os.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := file.Truncate(servingSmallFileLimit + 1); err != nil {
				_ = file.Close()
				t.Fatal(err)
			}
			if err := file.Close(); err != nil {
				t.Fatal(err)
			}
			result := inspectFixture(t, f)
			if result.Status == "resolved" || f.ex.modelInspection.lastUsage.SourceBytes != 0 || f.ex.modelInspection.lastUsage.HeaderBytes != 0 {
				t.Fatalf("oversized source read: %+v usage=%+v", result, f.ex.modelInspection.lastUsage)
			}
		})
	}
}

func TestModelSettingsFactsReceiptAllReadsAccounted(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	f.st.plat.Runtime.Args = nil
	list := modelFileList{Engine: "llamacpp", Model: f.request.ModelID, Layout: llamaArtifactsLayout, Files: []modelFile{{Role: "model", Path: "model/copied.gguf", Size: 4096, SHA256: strings.Repeat("b", 64)}}}
	id, err := artifactExportID(list)
	if err != nil {
		t.Fatal(err)
	}
	list.ExportID = id
	base := copyAliasDir(f.root, f.request.ModelID)
	writeInspectionGGUF(t, filepath.Join(base, id, "model", "copied.gguf"), "llama", 0.25, 4096)
	data, err := json.Marshal(copyReceipt{Version: 1, List: list})
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(base, "copy.json"), data)
	var reads []inspectionRead
	opened, closed := 0, 0
	f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		opened++
		return &inspectionCountFile{File: file, reads: &reads, opened: &opened, closed: &closed}, nil
	}
	result := inspectFixture(t, f)
	usage := f.ex.modelInspection.lastUsage
	var total uint64
	for _, read := range reads {
		total += uint64(read.bytes)
	}
	if result.Status != "resolved" || opened != closed || usage.SourceBytes != 2*uint64(len(data)) || total != usage.SourceBytes+usage.SampleBytes+usage.HeaderBytes {
		t.Fatalf("receipt=%+v usage=%+v actual=%d opens=%d closes=%d", result, usage, total, opened, closed)
	}
}

func TestModelSettingsFactsRecommendationMergePreservesDescriptorAuthority(t *testing.T) {
	standard := modelSettingsExtraOptions()
	recommendations := []modelconfig.OptionDescriptor{
		{Field: "generation.top_p", Default: json.RawMessage("2"), DefaultSource: "embedded invalid"},
		{Field: "generation.repeat_penalty", Default: json.RawMessage("0"), DefaultSource: "embedded invalid"},
		{Field: "generation.min_p", Default: json.RawMessage("0.125"), DefaultSource: "embedded valid"},
		{Field: "reasoning", Default: json.RawMessage(`"unsupported"`), DefaultSource: "embedded invalid"},
	}
	merged := mergeModelInspectionOptions(recommendations, standard)
	if len(merged) != len(standard) {
		t.Fatal("recommendation changed descriptor set")
	}
	for i, option := range merged {
		if err := option.Validate(); err != nil {
			t.Fatalf("descriptor %s: %v", option.Field, err)
		}
		if option.Field == "generation.min_p" {
			if string(option.Default) != "0.125" || option.DefaultSource != "embedded valid" || !reflect.DeepEqual(option.Minimum, standard[i].Minimum) || !reflect.DeepEqual(option.Maximum, standard[i].Maximum) || option.Kind != standard[i].Kind || option.Label != standard[i].Label {
				t.Fatalf("valid default lost authority: %+v", option)
			}
		} else if !reflect.DeepEqual(option, standard[i]) {
			t.Fatalf("invalid or unsupported recommendation changed descriptor: %+v", option)
		}
	}
}

func TestModelSettingsFactsCanonicalReadsAccounted(t *testing.T) {
	f := newInspectionFixture(t, 4096)
	f.st.plat.Runtime.Args = nil
	f.request.ModelID = "owner/repository:Q4_K_M"
	inspectionNativeMembership(t, f)
	repo, commit := llamaRepoDir(f.root, "owner/repository"), strings.Repeat("a", 40)
	writeTestFile(t, filepath.Join(repo, "refs", "main"), []byte(commit))
	writeInspectionGGUF(t, filepath.Join(repo, "snapshots", commit, "model-Q4_K_M.gguf"), "llama", 0.25, 4096)
	var reads []inspectionRead
	opened, closed := 0, 0
	f.ex.modelInspection.open = func(path string) (servingArtifactFile, error) {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		opened++
		return &inspectionCountFile{File: file, reads: &reads, opened: &opened, closed: &closed}, nil
	}
	result := inspectFixture(t, f)
	usage := f.ex.modelInspection.lastUsage
	var total uint64
	for _, read := range reads {
		total += uint64(read.bytes)
	}
	// Each initial/final resolver reads the ref for selection and rechecks its
	// exact bytes after collecting artifacts.
	if result.Status != "resolved" || opened != closed || usage.SourceBytes != 4*uint64(len(commit)) || total != usage.SourceBytes+usage.SampleBytes+usage.HeaderBytes || usage.HeaderBytes != uint64(f.directoryEnd) {
		t.Fatalf("canonical reads: status=%s usage=%+v actual=%d open=%d close=%d", result.Status, usage, total, opened, closed)
	}
}

func TestModelSettingsFactsPublicArchitectureRejectsFreeText(t *testing.T) {
	for _, architecture := range []string{"/private/model", `C:\private\model`, "private template text\n"} {
		facts := gguffacts.Facts{Shards: []gguffacts.Shard{{Role: gguffacts.RoleMain, Architecture: architecture, Numeric: map[string]gguffacts.Numeric{architecture + ".block_count": {Type: 4, Bits: 4}}}}}
		result, _ := projectInspectionFacts(facts)
		if result.Architecture != "" || result.LayerCount != nil {
			t.Fatalf("free text published: %+v", result)
		}
		for key := range result.Dimensions {
			if strings.Contains(key, architecture) {
				t.Fatalf("free-text dimension published: %s", key)
			}
		}
	}
}

func TestModelSettingsFactsPublicReplyBudgetIncludesDescriptors(t *testing.T) {
	result := modelconfig.InspectResult{Engine: "llamacpp", ModelID: "exact-alias", Status: "resolved", Facts: &modelconfig.Facts{Status: "partial"}}
	artifact := modelconfig.ArtifactSummary{Role: "model", Identity: strings.Repeat("a", 64), Size: "4096", Assurance: "local head/tail samples and file identity; not full content verification"}
	result.Artifacts = []modelconfig.ArtifactSummary{artifact}
	fits, err := inspectionReplyFits(result)
	if err != nil || !fits {
		t.Fatalf("ordinary complete reply: fits=%v err=%v", fits, err)
	}
	result.Artifacts = make([]modelconfig.ArtifactSummary, maxCopyArtifacts)
	for i := range result.Artifacts {
		result.Artifacts[i] = artifact
	}
	fits, err = inspectionReplyFits(result)
	if err != nil || fits {
		t.Fatalf("oversized complete reply: fits=%v err=%v", fits, err)
	}
}
