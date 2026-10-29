// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"nvpair-shared/modelconfig"
)

func TestLinkedModelRecognitionBoundary(t *testing.T) {
	t.Setenv("LLAMA_ARG_MODELS_PRESET", "")
	type reads struct {
		opens, readAt, streams, readDir, closes int
	}
	track := func() (servingFileAccess, *reads) {
		c := &reads{}
		return servingFileAccess{
			open: func(path string) (servingArtifactFile, error) {
				f, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				c.opens++
				return &countedLinkedFile{File: f, reads: &c.readAt, streams: &c.streams, closes: &c.closes}, nil
			},
			openDir: func(path string) (servingDirectory, error) {
				f, err := os.Open(path)
				if err != nil {
					return nil, err
				}
				c.opens++
				return &countedLinkedDir{File: f, reads: &c.readDir, closes: &c.closes}, nil
			},
		}, c
	}
	type fixture struct {
		ex            *Executor
		engine, model string
		files         map[string]int64
	}
	ollama := func(t *testing.T) fixture {
		t.Helper()
		root := t.TempDir()
		fx := writeOllamaModel(t, filepath.Join(root, "models"), "demo", 128, 1)
		files := map[string]int64{fx.manifestRel: int64(len(fx.manifest))}
		for path, data := range fx.blobs {
			files[path] = int64(len(data))
		}
		return fixture{copyTestExecutor(t, "ollama", root, true), "ollama", "demo", files}
	}
	llama := func(t *testing.T) fixture {
		t.Helper()
		root := t.TempDir()
		const repo = "owner/demo-GGUF"
		writeLlamaCache(t, root, repo, strings.Repeat("a", 40), map[string][]byte{"demo-Q4_K_M.gguf": testBytes(128, 2)})
		return fixture{copyTestExecutor(t, "llamacpp", root, true), "llamacpp", repo + ":Q4_K_M", map[string]int64{"demo-Q4_K_M.gguf": 128}}
	}
	assertFiles := func(t *testing.T, f fixture, list modelFileList) {
		t.Helper()
		if list.Engine != f.engine || list.Model != f.model || list.Layout != "" || len(list.Files) != len(f.files) {
			t.Fatalf("native listing identity or set changed: %+v", list)
		}
		seen := make(map[string]bool)
		for _, file := range list.Files {
			size, ok := f.files[file.Path]
			if !ok || seen[file.Path] || file.Size != size || file.configured == "" || file.abs == "" || file.info == nil {
				t.Errorf("unexpected or incomplete native file: %+v", file)
			}
			seen[file.Path] = true
		}
	}
	assertRejected := func(t *testing.T, f fixture, calls *int) {
		t.Helper()
		*calls = 0
		access, c := track()
		list, err := f.ex.modelSourceFilesWithAccess(context.Background(), f.engine, f.model, access)
		if err == nil || len(list.Files) != 0 {
			t.Errorf("unrecognized native source returned files: files=%d error=%v", len(list.Files), err)
		}
		if c.readAt != 0 || c.streams != 0 || c.readDir != 0 {
			t.Errorf("content accessed before exact recognition: ReadAt=%d stream=%d ReadDir=%d", c.readAt, c.streams, c.readDir)
		}
		if c.opens != c.closes {
			t.Errorf("source handles leaked: opened=%d closed=%d", c.opens, c.closes)
		}
		if *calls != 1 {
			t.Errorf("fresh selected-engine recognition calls=%d, want 1", *calls)
		}
	}
	testNative := func(name string, setup func(*testing.T) fixture) {
		t.Run(name, func(t *testing.T) {
			t.Run("recognized native control", func(t *testing.T) {
				f := setup(t)
				calls := copyTestNativeModels(t, f.ex, f.engine, f.model)
				observation := f.ex.readEngineInventory(context.Background(), f.engine)
				if got := observation.membership(f.model); got != inventoryPresent {
					t.Fatalf("native positive control membership=%v", got)
				}
				*calls = 0
				access, c := track()
				list, err := f.ex.modelSourceFilesWithAccess(context.Background(), f.engine, f.model, access)
				if err != nil {
					t.Fatalf("recognized native source: %v", err)
				}
				assertFiles(t, f, list)
				_, store, err := f.ex.modelStoreFor(f.engine)
				if err != nil {
					t.Fatal(err)
				}
				source := list.source
				if source == nil || source.inventory == nil || source.engine != f.engine || source.root != store.root ||
					source.run != observation.runID || source.configuration != observation.launchDigest ||
					!bytes.Equal(source.recognition, observation.raw) || source.inventory.membership(f.model) != inventoryPresent ||
					source.inventory.modelsDir != observation.modelsDir || source.inventory.epoch != observation.epoch || len(source.artifacts) != len(list.Files) {
					t.Fatalf("source lost its exact native observation, run, configuration or artifact set: %+v", source)
				}
				wantCalls := 1
				if f.engine == "ollama" {
					wantCalls = 2
				}
				if *calls != wantCalls {
					t.Errorf("fresh selected-engine recognition calls=%d, want %d (Ollama includes final-publication recheck)", *calls, wantCalls)
				}
				if c.readAt == 0 || c.opens != c.closes || (f.engine == "llamacpp" && c.readDir == 0) {
					t.Fatalf("native control did not exercise closed instrumented readers: %+v", c)
				}
			})
			t.Run("withdrawn recognition requires a fresh read", func(t *testing.T) {
				f := setup(t)
				copyTestNativeModels(t, f.ex, f.engine, f.model)
				copyTestSourceListing(t, f.ex, f.engine, f.model)
				calls := copyTestNativeModels(t, f.ex, f.engine)
				assertRejected(t, f, calls)
			})
			reject := func(name, body string, want inventoryMembership) {
				t.Run(name, func(t *testing.T) {
					f := setup(t)
					calls := copyTestNativeModels(t, f.ex, f.engine, f.model)
					st, err := f.ex.state(f.engine)
					if err != nil {
						t.Fatal(err)
					}
					// Exercise the existing native readiness extraction, not a
					// download read fence that would reject before recognition.
					st.manifest.Actions["pull_model"] = Action{ProgressProtocol: pullProgressProtocolLlamaCPPModelsSSE}
					original := f.ex.client.Transport
					f.ex.client.Transport = memoryModelsTransport(func(req *http.Request) (*http.Response, error) {
						response, err := original.RoundTrip(req)
						if err != nil {
							return nil, err
						}
						if err := response.Body.Close(); err != nil {
							return nil, err
						}
						if body == "" {
							return nil, errors.New("synthetic native recognition failure")
						}
						response.Body = io.NopCloser(strings.NewReader(strings.ReplaceAll(body, "MODEL", f.model)))
						return response, nil
					})
					if got := f.ex.readEngineInventory(context.Background(), f.engine).membership(f.model); got != want {
						t.Fatalf("native refusal fixture membership=%v, want %v", got, want)
					}
					assertRejected(t, f, calls)
				})
			}
			reject("absent", `{"models":[]}`, inventoryAbsent)
			reject("unready", `{"models":[{"name":"MODEL","status":{"value":"downloading"}}]}`, inventoryUnready)
			reject("ambiguous", `{"models":[{"name":"MODEL","status":{"value":"unloaded"}},{"name":"MODEL","status":{"value":"unloaded"}}]}`, inventoryAmbiguous)
			reject("failed recognition", "", inventoryFailed)
			reject("malformed recognition", `{"models":null}`, inventoryFailed)

			t.Run("wrong engine", func(t *testing.T) {
				f := setup(t)
				calls := copyTestNativeModels(t, f.ex, f.engine)
				st, err := f.ex.state(f.engine)
				if err != nil {
					t.Fatal(err)
				}
				other := *st.manifest
				other.Engine = f.engine + "-other"
				other.Actions = map[string]Action{"list_models": {
					HTTP:   &ActionHTTP{Method: http.MethodGet, Path: "/other/models"},
					Result: &ActionResult{Array: "models", Field: "name"},
				}}
				f.ex.reg.engines[other.Engine] = &other
				otherState, err := f.ex.state(other.Engine)
				if err != nil {
					t.Fatal(err)
				}
				otherState.running, otherState.runID, otherState.port = true, "other-run", 18081
				original, otherCalls := f.ex.client.Transport, 0
				f.ex.client.Transport = memoryModelsTransport(func(req *http.Request) (*http.Response, error) {
					if req.URL.Path != "/other/models" {
						return original.RoundTrip(req)
					}
					otherCalls++
					body := fmt.Sprintf(`{"models":[{"name":%q}]}`, f.model)
					return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}, nil
				})
				if got := f.ex.readEngineInventory(context.Background(), other.Engine).membership(f.model); got != inventoryPresent {
					t.Fatalf("other engine did not recognize the exact model: %v", got)
				}
				if got := f.ex.readEngineInventory(context.Background(), f.engine).membership(f.model); got != inventoryAbsent {
					t.Fatalf("selected engine unexpectedly recognizes model: %v", got)
				}
				otherCalls = 0
				assertRejected(t, f, calls)
				if otherCalls != 0 {
					t.Fatalf("source admission queried unrelated engine %d times", otherCalls)
				}
			})
		})
	}
	testNative("ollama", ollama)
	testNative("llama HF fallback", llama)

	t.Run("LM Studio native row needs no running HTTP inventory", func(t *testing.T) {
		root := t.TempDir()
		const model, path = "owner/demo/demo-Q4_K_M", "owner/demo/demo-Q4_K_M.gguf"
		writeTestFile(t, filepath.Join(root, filepath.FromSlash(path)), testBytes(128, 3))
		ex := copyTestExecutor(t, "lmstudio", root, true)
		ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
			t.Error("LM Studio source attempted an additional HTTP inventory gate")
			return nil, errors.New("unexpected HTTP inventory")
		})
		original, calls := lmsListModels, 0
		lmsListModels = func(_ context.Context, owner *Executor, _ string) ([]lmsListEntry, error) {
			if owner != ex {
				t.Fatal("LM Studio listing used another executor")
			}
			calls++
			return []lmsListEntry{{ModelKey: model, Path: path}}, nil
		}
		t.Cleanup(func() { lmsListModels = original })
		list, err := ex.modelSourceFiles(context.Background(), "lmstudio", model)
		if err != nil {
			t.Fatalf("native LM Studio row: %v", err)
		}
		assertFiles(t, fixture{ex, "lmstudio", model, map[string]int64{path: 128}}, list)
		if calls != 2 {
			t.Fatalf("native row queries=%d, want 2 including final-publication row-drift check", calls)
		}
	})

	testCustom := func(name string, configure func(*testing.T, *Executor, string, string)) {
		t.Run(name, func(t *testing.T) {
			root, external := t.TempDir(), t.TempDir()
			const model = "owner/custom-GGUF:Q4_K_M"
			artifact := filepath.Join(external, "custom.gguf")
			writeTestFile(t, artifact, testBytes(128, 4))
			// A conflicting native cache makes falling through observable.
			writeLlamaCache(t, root, "owner/custom-GGUF", strings.Repeat("b", 40), map[string][]byte{"cached-Q4_K_M.gguf": testBytes(256, 5)})
			ex := copyTestExecutor(t, "llamacpp", root, true)
			ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
				t.Error("configured custom source attempted native recognition")
				return nil, errors.New("unexpected native recognition")
			})
			configure(t, ex, model, artifact)
			st, err := ex.state("llamacpp")
			if err != nil {
				t.Fatal(err)
			}
			if st.running || st.plat.Runtime.Serving != nil {
				t.Fatal("custom control requires an off engine without Runtime.Serving")
			}
			access, c := track()
			list, err := ex.modelSourceFilesWithAccess(context.Background(), "llamacpp", model, access)
			if err != nil {
				t.Fatalf("configured custom source: %v", err)
			}
			if list.Layout != llamaArtifactsLayout || len(list.Files) != 1 || list.Files[0].Size != 128 || list.Files[0].Path != "model/custom.gguf" {
				t.Fatalf("custom authority fell through to native cache: %+v", list)
			}
			if c.readAt == 0 || c.opens != c.closes {
				t.Fatalf("custom control did not exercise closed instrumented readers: %+v", c)
			}
		})
	}
	testCustom("off-engine preset authority", func(t *testing.T, ex *Executor, model, artifact string) {
		preset := filepath.Join(filepath.Dir(artifact), "models.ini")
		writeTestFile(t, preset, []byte("["+model+"]\nmodel = custom.gguf\n"))
		st, err := ex.state("llamacpp")
		if err != nil {
			t.Fatal(err)
		}
		// Use one effective option owner, including when the inherited
		// LLAMA_ARG_MODELS_PRESET is present but empty.
		st.plat.Runtime.Env = map[string]string{"LLAMA_ARG_MODELS_PRESET": preset}
	})
	testCustom("off-engine typed authority", func(t *testing.T, ex *Executor, model, artifact string) {
		binding, err := servingFileBinding(artifact)
		if err != nil {
			t.Fatal(err)
		}
		load := modelconfig.ModelLoad{ModelID: model, Resolved: &modelconfig.ResolvedLoad{
			PerSlotContext: 4096, TotalContext: 4096, Slots: 1, GPU: modelconfig.GPU{Mode: "cpu"},
			NativeBuildID: "synthetic-build", SourceInputDigest: strings.Repeat("c", 64), PresetDigest: strings.Repeat("d", 64),
			Artifacts: []modelconfig.ArtifactBinding{{Role: "model", Path: artifact, ResolvedTarget: binding.ResolvedTarget,
				Size: strconv.FormatInt(binding.SizeBytes, 10), MtimeNS: binding.MtimeNS, SampleSHA256: binding.SampleSHA256}},
		}}
		if err := load.Validate(); err != nil {
			t.Fatalf("typed authority fixture: %v", err)
		}
		st, err := ex.state("llamacpp")
		if err != nil {
			t.Fatal(err)
		}
		st.plat.Runtime.ModelLoads = []modelconfig.ModelLoad{load}
	})
	testCustom("off-engine receipt authority", func(t *testing.T, ex *Executor, model, artifact string) {
		data, err := os.ReadFile(artifact)
		if err != nil {
			t.Fatal(err)
		}
		list := modelFileList{Engine: "llamacpp", Model: model, Layout: llamaArtifactsLayout,
			Files: []modelFile{{Role: "model", Path: "model/custom.gguf", Size: int64(len(data)), SHA256: sha256Hex(data)}}}
		list.ExportID, err = artifactExportID(list)
		if err != nil {
			t.Fatal(err)
		}
		_, store, err := ex.modelStoreFor("llamacpp")
		if err != nil {
			t.Fatal(err)
		}
		base := copyAliasDir(store.root, model)
		writeTestFile(t, filepath.Join(base, list.ExportID, "model", "custom.gguf"), data)
		receipt, err := json.Marshal(copyReceipt{Version: 1, List: list})
		if err != nil {
			t.Fatal(err)
		}
		writeTestFile(t, filepath.Join(base, "copy.json"), receipt)
	})
}
