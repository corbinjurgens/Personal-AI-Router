// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/performancewire"
)

type memoryModelsTransport func(*http.Request) (*http.Response, error)

func (f memoryModelsTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

// This fixture owns no process or listener. Every engine request must be one
// of the two existing inventory GETs; mutating endpoints fail immediately.
func memoryModelsFixture(t *testing.T) (*memoryEvidenceHarness, *string) {
	t.Helper()
	h := newMemoryEvidenceHarness(t)
	manifest := h.ex.engines["ollama"].manifest
	manifest.Actions = map[string]Action{
		"list_models":   {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/api/tags"}, Result: &ActionResult{Array: "models", Field: "name", Digest: "digest"}},
		"loaded_models": {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/api/ps"}, Result: &ActionResult{Array: "models", Field: "name"}},
	}
	loaded := memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
	h.ex.client = &http.Client{Transport: memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
		if request.Method != http.MethodGet {
			t.Errorf("forbidden engine method %s", request.Method)
			return nil, fmt.Errorf("forbidden engine method")
		}
		body := ""
		switch request.URL.Path {
		case "/api/tags":
			body = fmt.Sprintf(`{"models":[{"name":%q,"digest":%q}]}`, memoryTestModel, h.digest)
		case "/api/ps":
			body = loaded
		default:
			t.Errorf("forbidden engine endpoint %s", request.URL.Path)
			return nil, fmt.Errorf("forbidden engine endpoint")
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}, nil
	})}
	h.ex.modelMutationParent = func(context.Context, string) error {
		t.Error("forbidden model-content mutation")
		return fmt.Errorf("forbidden model-content mutation")
	}
	return h, &loaded
}

func modelsMemoryEntry(t *testing.T, result ModelsResult) performancewire.MemoryEvidenceEntry {
	t.Helper()
	if !reflect.DeepEqual(result.Models, []string{memoryTestModel}) ||
		!reflect.DeepEqual(result.ByEngine["ollama"], []string{memoryTestModel}) ||
		!reflect.DeepEqual(result.LoadedByEngine["ollama"], []string{memoryTestModel}) {
		t.Fatalf("memory processing changed model names: %+v", result)
	}
	if result.MemoryEvidence == nil {
		t.Fatal("memoryEvidence is absent")
	}
	if err := result.MemoryEvidence.Validate(); err != nil {
		t.Fatalf("invalid public evidence: %v", err)
	}
	return memoryEntry(t, result.MemoryEvidence.Entries)
}

func TestModelsMemoryEvidencePublicUnknownZeroAndQualified(t *testing.T) {
	test := func(name, raw string, artifact bool, wantAllocated *int64, wantQualified bool) {
		t.Run(name, func(t *testing.T) {
			h, loaded := memoryModelsFixture(t)
			*loaded = raw
			if artifact {
				h.writeArtifact(t, h.artifact("2"))
			}
			entry := modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))
			if !sameMemoryInt64(entry.Observed.AllocatedBytes, wantAllocated) {
				t.Fatalf("allocation = %v, want %v", entry.Observed.AllocatedBytes, wantAllocated)
			}
			if (memoryUpper(entry) != nil) != wantQualified {
				t.Fatalf("upper = %v, qualified wanted %v; uncertainty=%v", memoryUpper(entry), wantQualified, entry.Uncertainty)
			}
		})
	}
	test("absent authority remains advisory", memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192), false, memoryValue[int64](9000), false)
	test("actual measured zero", memoryPS(memoryTestModel, memoryTestDigest, 0, 8192), true, memoryValue[int64](0), true)
	test("missing allocation stays unknown", memoryPSWithoutSize(memoryTestModel, memoryTestDigest), true, nil, false)
	test("synthetic qualified upper", memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192), true, memoryValue[int64](9000), true)
}

func TestModelsMemoryEvidenceInvalidInputPreservesNames(t *testing.T) {
	test := func(name string, setup func(*testing.T, *memoryEvidenceHarness, *string)) {
		t.Run(name, func(t *testing.T) {
			h, loaded := memoryModelsFixture(t)
			setup(t, h, loaded)
			entry := modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))
			if memoryUpper(entry) != nil {
				t.Fatal("invalid memory input qualified")
			}
		})
	}
	test("malformed artifact", func(t *testing.T, h *memoryEvidenceHarness, loaded *string) {
		h.writeArtifactBytes(t, []byte(`{"version":`))
	})
	test("oversize artifact", func(t *testing.T, h *memoryEvidenceHarness, loaded *string) {
		h.writeArtifactBytes(t, []byte(strings.Repeat("x", maxMemoryQualificationBytes+1)))
	})
	test("ambiguous allocation", func(t *testing.T, h *memoryEvidenceHarness, loaded *string) {
		h.writeArtifact(t, h.artifact("2"))
		*loaded = strings.Replace(*loaded, `"size_vram":9000`, `"size_vram":9000,"size_vram":9001`, 1)
	})
	test("oversize loaded response", func(t *testing.T, h *memoryEvidenceHarness, loaded *string) {
		h.writeArtifact(t, h.artifact("2"))
		*loaded = strings.TrimSuffix(*loaded, "}") + `,"extra":"` + strings.Repeat("x", performancewire.MaxMemoryEvidenceBytes) + `"}`
	})
}

func TestModelsMemoryEvidenceReviewAmbiguousSharedDigest(t *testing.T) {
	h, loaded := memoryModelsFixture(t)
	h.writeArtifact(t, h.artifact("2"))
	target := strings.TrimSuffix(strings.TrimPrefix(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192), `{"models":[`), `]}`)
	*loaded = fmt.Sprintf(`{"models":[{"name":"other:latest","model":"other:latest","digest":%q,"size_vram":1,"size_vram":2,"context_length":8192},%s]}`, memoryTestDigest, target)
	result := h.ex.ModelsResult(context.Background())
	if result.MemoryEvidence == nil {
		t.Fatal("memory evidence missing")
	}
	for _, entry := range result.MemoryEvidence.Entries {
		if entry.Model == memoryTestModel {
			if memoryUpper(entry) != nil {
				t.Fatalf("target sharing digest with ambiguous sibling qualified with upper %d", *memoryUpper(entry))
			}
			return
		}
	}
	t.Fatal("target evidence missing")
}

func TestModelsMemoryEvidenceAmbiguousResponsePreservesInventory(t *testing.T) {
	const otherDigest = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
	target := strings.TrimSuffix(strings.TrimPrefix(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192), `{"models":[`), `]}`)
	other := func(name, digest string) string {
		return fmt.Sprintf(`{"name":%q,"model":%q,"digest":%q,"size_vram":1,"context_length":8192}`, name, name, digest)
	}
	test := func(name, siblings string, wantLoaded []string, wantQualified bool) {
		t.Run(name, func(t *testing.T) {
			h, loaded := memoryModelsFixture(t)
			h.writeArtifact(t, h.artifact("2"))
			*loaded = `{"models":[` + siblings + "," + target + "]}"
			var reads atomic.Int32
			original := h.ex.client.Transport
			h.ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
				reads.Add(1)
				return original.RoundTrip(request)
			})
			result := h.ex.ModelsResult(context.Background())
			if !reflect.DeepEqual(result.Models, []string{memoryTestModel}) ||
				!reflect.DeepEqual(result.ByEngine, map[string][]string{"ollama": {memoryTestModel}}) ||
				!reflect.DeepEqual(result.LoadedByEngine, map[string][]string{"ollama": wantLoaded}) {
				t.Fatalf("memory ambiguity changed installed/loaded inventory: %+v", result)
			}
			if reads.Load() != 2 {
				t.Fatalf("engine requests = %d, want only the existing inventory GETs", reads.Load())
			}
			if result.MemoryEvidence == nil {
				t.Fatal("memory evidence missing")
			}
			if err := result.MemoryEvidence.Validate(); err != nil {
				t.Fatalf("invalid public memory evidence: %v", err)
			}
			found := false
			for _, entry := range result.MemoryEvidence.Entries {
				if !wantQualified && (memoryUpper(entry) != nil || entry.Observed.AllocatedBytes != nil) {
					t.Fatalf("ambiguous response published allocation or upper for %q: %+v", entry.Model, entry)
				}
				if entry.Model == memoryTestModel {
					found = true
					if (memoryUpper(entry) != nil) != wantQualified {
						t.Fatalf("target upper = %v, want qualified %v; uncertainty=%v", memoryUpper(entry), wantQualified, entry.Uncertainty)
					}
					if wantQualified && (entry.Observed.AllocatedBytes == nil || *entry.Observed.AllocatedBytes != 9000) {
						t.Fatal("clean unique-digest positive lost its allocation")
					}
				}
			}
			if !found {
				t.Fatal("target evidence missing")
			}
		})
	}
	wantLoaded := []string{"other:latest", memoryTestModel}
	test("duplicate allocation with shared digest",
		fmt.Sprintf(`{"name":"other:latest","model":"other:latest","digest":%q,"size_vram":1,"size_vram":2,"context_length":8192}`, memoryTestDigest), wantLoaded, false)
	test("duplicate context with shared digest",
		fmt.Sprintf(`{"name":"other:latest","model":"other:latest","digest":%q,"size_vram":1,"context_length":4096,"context_length":8192}`, memoryTestDigest), wantLoaded, false)
	test("duplicate identity with shared digest",
		fmt.Sprintf(`{"name":"other:latest","name":"other:latest","model":"other:latest","digest":%q,"size_vram":1,"context_length":8192}`, memoryTestDigest), wantLoaded, false)
	test("repeated name hides shared digest",
		other("other:latest", memoryTestDigest)+","+other("other:latest", otherDigest), []string{"other:latest", "other:latest", memoryTestModel}, false)
	test("repeated normalized name hides shared digest",
		other("other", memoryTestDigest)+","+other("other:latest", otherDigest), []string{"other", "other:latest", memoryTestModel}, false)
	test("duplicate distinct digest fields hide target ownership",
		fmt.Sprintf(`{"name":"other:latest","model":"other:latest","digest":%q,"digest":%q,"size_vram":1,"context_length":8192}`, otherDigest, memoryTestDigest), wantLoaded, false)
	test("clean same-digest aliases", other("other:latest", memoryTestDigest), wantLoaded, false)
	test("clean unique-digest positive", other("other:latest", otherDigest), wantLoaded, true)
}

func TestModelsMemoryEvidenceNewQueriesRefreshOnlyAllocationAge(t *testing.T) {
	h, _ := memoryModelsFixture(t)
	h.writeArtifact(t, h.artifact("2"))
	first := modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))
	h.clock = h.clock.Add(20 * time.Second)
	second := modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))
	if memoryUpper(second) == nil || second.Freshness.SampleAgeMs == nil || *second.Freshness.SampleAgeMs != 0 {
		t.Fatalf("new successful query did not refresh allocation: %+v", second)
	}
	if first.Freshness.Generation == nil || second.Freshness.Generation == nil || *second.Freshness.Generation <= *first.Freshness.Generation {
		t.Fatal("real query did not advance observation generation")
	}
	if first.Freshness.QualificationAgeMs == nil || second.Freshness.QualificationAgeMs == nil || *second.Freshness.QualificationAgeMs-*first.Freshness.QualificationAgeMs != 20_000 {
		t.Fatal("new query rejuvenated qualification age")
	}
}

func TestModelsMemoryEvidencePreIOAnchorIncludesReadDelay(t *testing.T) {
	test := func(name string, delay time.Duration, wantQualified bool, wantAge int64) {
		t.Run(name, func(t *testing.T) {
			h, _ := memoryModelsFixture(t)
			h.writeArtifact(t, h.artifact("2"))
			original := h.ex.client.Transport
			h.ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
				response, err := original.RoundTrip(request)
				if request.URL.Path == "/api/ps" {
					// Buffer the existing source response first, then advance
					// time before the sweep can process or verify it.
					h.clock = h.clock.Add(delay)
				}
				return response, err
			})
			entry := modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))
			if (memoryUpper(entry) != nil) != wantQualified {
				t.Fatalf("upper = %v, want qualified %v; uncertainty=%v", memoryUpper(entry), wantQualified, entry.Uncertainty)
			}
			if entry.Freshness.SampleAgeMs == nil || *entry.Freshness.SampleAgeMs != wantAge {
				t.Fatalf("sample age = %v, want %d ms", entry.Freshness.SampleAgeMs, wantAge)
			}
			if entry.Observed.AllocatedBytes == nil || *entry.Observed.AllocatedBytes != 9000 {
				t.Fatal("delayed source response lost its allocation")
			}
		})
	}
	test("fresh buffered response includes elapsed query time", 4*time.Second, true, 4000)
	test("buffered response exceeds fixed validity", 16*time.Second, false, 16000)
}

func TestModelsMemoryEvidenceWithdrawalCannotRevive(t *testing.T) {
	h, loaded := memoryModelsFixture(t)
	original := h.writeArtifact(t, h.artifact("2"))
	if memoryUpper(modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))) == nil {
		t.Fatal("positive control failed")
	}
	*loaded = memoryPS(memoryTestModel, memoryTestDigest, 11_000, 8192)
	if memoryUpper(modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))) != nil {
		t.Fatal("exceedance retained upper")
	}
	*loaded = memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
	assertWithdrawn := func() {
		t.Helper()
		if memoryUpper(modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))) != nil {
			t.Fatal("contradicted authority revived")
		}
	}
	assertWithdrawn()
	h.removeArtifact(t)
	assertWithdrawn()
	h.writeArtifactBytes(t, original)
	assertWithdrawn()
	*loaded = `{"models":[]}`
	h.ex.ModelsResult(context.Background())
	*loaded = memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
	assertWithdrawn()
	h.writeArtifact(t, h.artifact("1"))
	assertWithdrawn()
	conflict := h.artifact("2")
	(*conflict.Records)[0].Qualification.UpperBytes = memoryValue[int64](20_000)
	h.writeArtifact(t, conflict)
	assertWithdrawn()
	h.writeArtifact(t, h.artifact("3"))
	if memoryUpper(modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))) == nil {
		t.Fatal("higher reviewed document did not recover")
	}
}

func TestModelsResultDecodeDiscardsInvalidMemoryEvidence(t *testing.T) {
	test := func(name, evidence string) {
		t.Run(name, func(t *testing.T) {
			payload := `{"models":["m"],"modelsByEngine":{"ollama":["m"]},"loadedByEngine":{"ollama":["m"]},"policy":{"revision":"21","roles":{}},"memoryEvidence":` + evidence + `}`
			result := ModelsResult{MemoryEvidence: &performancewire.MemoryEvidence{Version: 2}}
			if err := json.Unmarshal([]byte(payload), &result); err != nil {
				t.Fatalf("decode inventory with invalid optional evidence: %v", err)
			}
			want := ModelsResult{
				Models:         []string{"m"},
				ByEngine:       map[string][]string{"ollama": {"m"}},
				LoadedByEngine: map[string][]string{"ollama": {"m"}},
				Policy:         &nodepolicy.Advertisement{Revision: "21", Roles: map[string]nodepolicy.AdvertisedRole{}},
			}
			if !reflect.DeepEqual(result, want) {
				t.Fatalf("decoded result = %+v, want %+v", result, want)
			}
		})
	}
	test("unsupported version", `{"version":2,"entries":[],"omittedEntries":0}`)
	test("missing required field", `{"version":1,"entries":[]}`)
	test("wrong envelope type", `[]`)
	test("wrong entries type", `{"version":1,"entries":{},"omittedEntries":0}`)
	test("invalid entry", `{"version":1,"entries":[{}],"omittedEntries":0}`)
	test("unknown envelope field", `{"version":1,"entries":[],"omittedEntries":0,"extra":true}`)
	test("oversize envelope", `{"version":1,"entries":[],"omittedEntries":0,"extra":"`+strings.Repeat("x", performancewire.MaxMemoryEvidenceBytes)+`"}`)
}

func TestModelsResultDecodeOptionalMemoryEvidence(t *testing.T) {
	test := func(name, payload string, want *performancewire.MemoryEvidence) {
		t.Run(name, func(t *testing.T) {
			result := ModelsResult{MemoryEvidence: &performancewire.MemoryEvidence{Version: 2}}
			if err := json.Unmarshal([]byte(payload), &result); err != nil {
				t.Fatalf("decode optional evidence: %v", err)
			}
			if !reflect.DeepEqual(result.Models, []string{"m"}) {
				t.Fatalf("models = %v, want [m]", result.Models)
			}
			if !reflect.DeepEqual(result.MemoryEvidence, want) {
				t.Fatalf("evidence = %+v, want %+v", result.MemoryEvidence, want)
			}
		})
	}
	test("valid replaces previous evidence", `{"models":["m"],"memoryEvidence":{"version":1,"entries":[],"omittedEntries":0}}`, &performancewire.MemoryEvidence{Version: 1, Entries: []performancewire.MemoryEvidenceEntry{}})
	test("absent clears previous evidence", `{"models":["m"]}`, nil)
	test("null clears previous evidence", `{"models":["m"],"memoryEvidence":null}`, nil)
}

func TestModelsResultDecodeRejectsMalformedInventory(t *testing.T) {
	test := func(name, payload string) {
		t.Run(name, func(t *testing.T) {
			var result ModelsResult
			if err := json.Unmarshal([]byte(payload), &result); err == nil {
				t.Fatalf("decode malformed inventory %s succeeded", payload)
			}
		})
	}
	test("invalid JSON", `{"models":["m"],"memoryEvidence":`)
	test("wrong result type", `[]`)
	test("wrong models type", `{"models":{},"memoryEvidence":{"version":2}}`)
	test("wrong attribution type", `{"models":["m"],"modelsByEngine":{"ollama":1}}`)
	test("wrong residency type", `{"models":["m"],"loadedByEngine":{"ollama":1}}`)
	test("wrong policy type", `{"models":["m"],"policy":[]}`)
}

func TestModelsMemoryEvidenceRPCCodec(t *testing.T) {
	h, _ := memoryModelsFixture(t)
	h.writeArtifact(t, h.artifact("2"))
	var output bytes.Buffer
	manager := &Manager{exec: h.ex, codec: NewCodec(&output)}
	id := json.RawMessage(`1`)
	manager.runModels(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:models"})
	var message Message
	if err := json.Unmarshal(output.Bytes(), &message); err != nil {
		t.Fatalf("decode RPC: %v", err)
	}
	var result ModelsResult
	if err := json.Unmarshal(message.Result, &result); err != nil {
		t.Fatalf("decode RPC models result: %v", err)
	}
	if memoryUpper(modelsMemoryEntry(t, result)) == nil || !message.IsResponse() {
		t.Fatal("actual engine:models response lost qualified evidence")
	}
}

func TestModelsMemoryEvidenceWatchPublishesBaselineChangesAndWithdrawal(t *testing.T) {
	h, loaded := memoryModelsFixture(t)
	h.writeArtifact(t, h.artifact("2"))
	h.ex.loadedPollInterval = time.Hour
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	results := make(chan modelsChangedParams, 4)
	var writeMu sync.Mutex
	var output bytes.Buffer
	codec := NewCodec(&output)
	h.ex.emit = func(method string, params any) {
		if method != modelsChangedMethod {
			return
		}
		writeMu.Lock()
		defer writeMu.Unlock()
		output.Reset()
		if err := codec.Notify(method, params); err != nil {
			t.Errorf("encode models notification: %v", err)
			return
		}
		var message Message
		if err := json.Unmarshal(output.Bytes(), &message); err != nil {
			t.Errorf("decode notification: %v", err)
			return
		}
		var result modelsChangedParams
		if err := json.Unmarshal(message.Params, &result); err != nil {
			t.Errorf("decode notification params: %v", err)
			return
		}
		results <- result
	}
	done := make(chan struct{})
	go func() { defer close(done); h.ex.watchLoaded(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Error("watcher did not stop")
		}
	})
	next := func() performancewire.MemoryEvidenceEntry {
		t.Helper()
		h.ex.pokeLoaded()
		select {
		case result := <-results:
			if result.Engine != "ollama" {
				t.Fatalf("trigger engine = %q", result.Engine)
			}
			return modelsMemoryEntry(t, result.Models)
		case <-time.After(5 * time.Second):
			t.Fatal("missing existing models notification")
			return performancewire.MemoryEvidenceEntry{}
		}
	}
	if memoryUpper(next()) == nil {
		t.Fatal("first memory baseline was not published")
	}
	*loaded = memoryPS(memoryTestModel, memoryTestDigest, 9500, 8192)
	changed := next()
	if changed.Observed.AllocatedBytes == nil || *changed.Observed.AllocatedBytes != 9500 {
		t.Fatal("unchanged-name allocation update was not published")
	}
	h.removeArtifact(t)
	if memoryUpper(next()) != nil {
		t.Fatal("unchanged-name withdrawal was not published")
	}
}

func TestModelsMemoryEvidenceSourceChangesAcrossRead(t *testing.T) {
	test := func(name string, change func(*memoryEvidenceHarness)) {
		t.Run(name, func(t *testing.T) {
			h, _ := memoryModelsFixture(t)
			h.writeArtifact(t, h.artifact("2"))
			original := h.ex.client.Transport
			h.ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
				response, err := original.RoundTrip(request)
				if request.URL.Path == "/api/ps" {
					change(h)
				}
				return response, err
			})
			entry := modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))
			if memoryUpper(entry) != nil {
				t.Fatal("source changed across read but qualified")
			}
			if entry.Source.EngineRunID != nil || entry.Provenance.Scope != nil || entry.Freshness.Generation != nil {
				t.Fatalf("response across source change retained run provenance: %+v", entry)
			}
		})
	}
	test("host disappears", func(h *memoryEvidenceHarness) { h.host = "" })
	test("manager incarnation changes", func(h *memoryEvidenceHarness) { h.ex.switchIncarnation = "manager-replacement" })
	test("engine run changes", func(h *memoryEvidenceHarness) {
		state := h.ex.engines["ollama"]
		state.mu.Lock()
		state.runID = "run-replacement"
		state.mu.Unlock()
	})
	test("owned identity disappears", func(h *memoryEvidenceHarness) {
		state := h.ex.engines["ollama"]
		state.mu.Lock()
		state.proc = nil
		state.mu.Unlock()
	})
	test("content epoch changes", func(h *memoryEvidenceHarness) {
		h.ex.inventoryMu.Lock()
		observed := h.ex.inventoryDigests["ollama"]
		observed.epoch++
		observed.models = map[string]string{memoryTestModel: strings.Repeat("d", 64)}
		h.ex.inventoryDigests["ollama"] = observed
		h.ex.inventoryMu.Unlock()
	})
	test("applied settings change", func(h *memoryEvidenceHarness) {
		next := h.settings
		next.Revision++
		next.AppliedRevision++
		h.ex.settingsHub.Publish([]settings.Snapshot{next})
	})
}

func TestModelsMemoryEvidenceOlderCompletionCannotPublishUpper(t *testing.T) {
	h, _ := memoryModelsFixture(t)
	h.writeArtifact(t, h.artifact("2"))
	started, release := make(chan struct{}), make(chan struct{})
	var reads atomic.Int32
	original := h.ex.client.Transport
	h.ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
		response, err := original.RoundTrip(request)
		if request.URL.Path == "/api/ps" && reads.Add(1) == 1 {
			close(started)
			select {
			case <-release:
			case <-request.Context().Done():
				return nil, request.Context().Err()
			}
		}
		return response, err
	})
	done := make(chan ModelsResult, 1)
	go func() { done <- h.ex.ModelsResult(context.Background()) }()
	var releaseOnce sync.Once
	releaseRead := func() { releaseOnce.Do(func() { close(release) }) }
	defer releaseRead()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("old sweep did not reach read barrier")
	}
	if memoryUpper(modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))) == nil {
		t.Fatal("newer sweep positive control failed")
	}
	// Release once, while ensuring failure cleanup also releases the waiter.
	releaseRead()
	select {
	case result := <-done:
		if memoryUpper(modelsMemoryEntry(t, result)) != nil {
			t.Fatal("older completion published an upper")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("old sweep did not complete")
	}
}

func TestModelsMemoryEvidenceTwoEngineContradictionWithdrawsWholeDocument(t *testing.T) {
	h, _ := memoryModelsFixture(t)
	const other = "ollama-variant"
	manifest := &Manifest{Engine: other, DisplayName: "Variant", Family: "ollama", ModelNaming: h.ex.engines["ollama"].manifest.ModelNaming,
		Actions: map[string]Action{
			"list_models":   {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/other/tags"}, Result: &ActionResult{Array: "models", Field: "name", Digest: "digest"}},
			"loaded_models": {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/other/ps"}, Result: &ActionResult{Array: "models", Field: "name"}},
		}}
	h.ex.reg.engines[other] = manifest
	h.ex.engines[other] = &engineState{manifest: manifest, plat: &Platform{}, running: true, runID: "other-run", launchDigest: memoryTestLaunch, proc: &managedProc{}}
	h.ex.inventoryDigests[other] = observedDigests{runID: "other-run", models: map[string]string{memoryTestModel: memoryTestDigest}, epoch: 9}
	otherSettings := h.settings
	otherSettings.Engine = other
	h.ex.settingsHub.Publish([]settings.Snapshot{h.settings, otherSettings})
	document := h.artifact("2")
	record := h.record()
	record.Source.Engine = memoryValue(other)
	record.Source.EngineRunID = memoryValue("other-run")
	*document.Records = append(*document.Records, record)
	h.writeArtifact(t, document)
	allocated := int64(11_000)
	original := h.ex.client.Transport
	h.ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path == "/other/tags" || request.URL.Path == "/other/ps" {
			if request.Method != http.MethodGet {
				return nil, fmt.Errorf("forbidden method")
			}
			body := memoryPS(memoryTestModel, memoryTestDigest, allocated, 8192)
			return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}, nil
		}
		return original.RoundTrip(request)
	})
	assertAll := func(wantQualified bool) {
		t.Helper()
		result := h.ex.ModelsResult(context.Background())
		if result.MemoryEvidence == nil || len(result.MemoryEvidence.Entries) != 2 {
			t.Fatalf("evidence = %+v", result.MemoryEvidence)
		}
		for _, entry := range result.MemoryEvidence.Entries {
			if (memoryUpper(entry) != nil) != wantQualified {
				t.Fatalf("%s upper = %v, want qualified %v", entry.Engine, memoryUpper(entry), wantQualified)
			}
		}
	}
	assertAll(false)
	allocated = 9000
	assertAll(false)
	document.Revision = memoryValue("3")
	h.writeArtifact(t, document)
	assertAll(true)
}

func TestModelsMemoryEvidenceActualFamilyAndNaming(t *testing.T) {
	test := func(name, family string, wantQualified bool) {
		t.Run(name, func(t *testing.T) {
			h, _ := memoryModelsFixture(t)
			manifest := h.ex.engines["ollama"].manifest
			manifest.Family = family
			manifest.Engine = "dynamic-variant"
			h.ex.reg.engines[manifest.Engine] = manifest
			delete(h.ex.reg.engines, "ollama")
			h.ex.engines[manifest.Engine] = h.ex.engines["ollama"]
			delete(h.ex.engines, "ollama")
			h.ex.inventoryDigests[manifest.Engine] = h.ex.inventoryDigests["ollama"]
			delete(h.ex.inventoryDigests, "ollama")
			h.settings.Engine = manifest.Engine
			h.ex.settingsHub.Publish([]settings.Snapshot{h.settings})
			document := h.artifact("2")
			(*document.Records)[0].Source.Engine = memoryValue(manifest.Engine)
			h.writeArtifact(t, document)
			result := h.ex.ModelsResult(context.Background())
			if result.MemoryEvidence == nil {
				t.Fatal("public evidence missing")
			}
			entry := memoryEntry(t, result.MemoryEvidence.Entries)
			if (memoryUpper(entry) != nil) != wantQualified {
				t.Fatalf("family %q upper = %v, qualified wanted %v", family, memoryUpper(entry), wantQualified)
			}
		})
	}
	test("owned Ollama variant", "ollama", true)
	test("unsupported actual family despite Ollama artifact", "unsupported", false)
}

func TestExtractStrings(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		spec *ActionResult
		want []string
	}{
		{
			name: "ollama tags shape",
			raw:  `{"models":[{"name":"llama3:8b","model":"llama3:8b"},{"name":"qwen:0.5b"}]}`,
			spec: &ActionResult{Array: "models", Field: "name"},
			want: []string{"llama3:8b", "qwen:0.5b"},
		},
		{
			name: "lmstudio v1/models shape",
			raw:  `{"object":"list","data":[{"id":"phi-3"},{"id":"gemma-2b"}]}`,
			spec: &ActionResult{Array: "data", Field: "id"},
			want: []string{"phi-3", "gemma-2b"},
		},
		{
			name: "lmstudio native models shape",
			raw:  `{"models":[{"key":"phi-3","loaded_instances":[]},{"key":"gemma-2b","loaded_instances":[]}]}`,
			spec: &ActionResult{Array: "models", Field: "key"},
			want: []string{"phi-3", "gemma-2b"},
		},
		{
			name: "missing array yields nothing",
			raw:  `{"other":[]}`,
			spec: &ActionResult{Array: "models", Field: "name"},
			want: nil,
		},
		{
			name: "elements missing the field are skipped",
			raw:  `{"models":[{"name":"a"},{"other":"b"},{"name":""}]}`,
			spec: &ActionResult{Array: "models", Field: "name"},
			want: []string{"a"},
		},
		{
			name: "non-object response yields nothing",
			raw:  `["a","b"]`,
			spec: &ActionResult{Array: "models", Field: "name"},
			want: nil,
		},
		{
			name: "string-in match keeps only matching rows",
			raw:  `{"data":[{"id":"a","state":"loaded"},{"id":"b","state":"not-loaded"},{"id":"c","state":"loaded"}]}`,
			spec: &ActionResult{Array: "data", Field: "id", Match: &ResultMatch{Field: "state", In: []string{"loaded"}}},
			want: []string{"a", "c"},
		},
		{
			name: "match: rows missing the state field are excluded",
			raw:  `{"data":[{"id":"a","state":"loaded"},{"id":"b"}]}`,
			spec: &ActionResult{Array: "data", Field: "id", Match: &ResultMatch{Field: "state", In: []string{"loaded"}}},
			want: []string{"a"},
		},
		{
			name: "nested string-in match excludes missing and malformed paths",
			raw:  `{"data":[{"id":"a","status":{"value":"loaded"}},{"id":"b","status":{"value":"not-loaded"}},{"id":"c","status":{}},{"id":"d","status":"loaded"}]}`,
			spec: &ActionResult{Array: "data", Field: "id", Match: &ResultMatch{Field: "status.value", In: []string{"loaded"}}},
			want: []string{"a"},
		},
		{
			name: "match with no accepted values yields nothing",
			raw:  `{"data":[{"id":"a","state":"loaded"}]}`,
			spec: &ActionResult{Array: "data", Field: "id", Match: &ResultMatch{Field: "state", In: []string{"other"}}},
			want: nil,
		},
		{
			name: "lmstudio v1 nonempty loaded_instances keeps only loaded rows",
			raw:  `{"models":[{"key":"a","loaded_instances":[{"id":"a"}]},{"key":"b","loaded_instances":[]},{"key":"c","loaded_instances":[{"id":"c"}]}]}`,
			spec: &ActionResult{Array: "models", Field: "key", Match: &ResultMatch{Field: "loaded_instances", Nonempty: true}},
			want: []string{"a", "c"},
		},
		{
			name: "nonempty match: missing or non-array field is excluded",
			raw:  `{"models":[{"key":"a","loaded_instances":[{"id":"a"}]},{"key":"b"},{"key":"c","loaded_instances":"bad"}]}`,
			spec: &ActionResult{Array: "models", Field: "key", Match: &ResultMatch{Field: "loaded_instances", Nonempty: true}},
			want: []string{"a"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := extractStrings(json.RawMessage(tc.raw), tc.spec)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("extractStrings = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestExtractStringsResultDistinguishesEmptyFromUnknown(t *testing.T) {
	spec := &ActionResult{Array: "models", Field: "key"}
	if got, ok := extractStringsResult(json.RawMessage(`{"models":[]}`), spec); !ok || got == nil || len(got) != 0 {
		t.Fatalf("explicit empty inventory = (%v, %v), want non-nil empty, true", got, ok)
	}
	for _, raw := range []string{
		`{}`,
		`{"models":null}`,
		`{"models":{}}`,
		`{"models":[{}]}`,
		`{"models":[null]}`,
		`not-json`,
	} {
		if got, ok := extractStringsResult(json.RawMessage(raw), spec); ok || got != nil {
			t.Errorf("invalid inventory %q = (%v, %v), want nil, false", raw, got, ok)
		}
	}
}

// TestModels drives Models() against the running fake engine: a running engine
// with a Result-bearing list_models action contributes its models; a stopped
// engine contributes nothing.
func TestModels(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	m.Actions["list_models"] = Action{
		HTTP:   &ActionHTTP{Method: "GET", Path: "/api/tags"},
		Result: &ActionResult{Array: "models", Field: "name"},
	}
	ex := newTestExecutor(t, m)
	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })

	// Stopped: nothing queryable.
	if got := ex.Models(ctx); len(got) != 0 {
		t.Fatalf("Models() on stopped engine = %v, want empty", got)
	}

	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	got := ex.Models(ctx)
	want := []string{"llama3.2:1b"} // fake engine's seeded model
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Models() on running engine = %v, want %v", got, want)
	}

	if err := ex.Stop("fake"); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if got := ex.Models(ctx); len(got) != 0 {
		t.Fatalf("Models() after stop = %v, want empty", got)
	}
}

// TestModelsResult drives ModelsResult() against the running fake engine: the
// flat union and the per-engine attribution are both derived from one sweep,
// and a stopped engine yields neither.
func TestModelsResult(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	m.Actions["list_models"] = Action{
		HTTP:   &ActionHTTP{Method: "GET", Path: "/api/tags"},
		Result: &ActionResult{Array: "models", Field: "name"},
	}
	m.Actions["delete_model"] = Action{HTTP: &ActionHTTP{Method: "DELETE", Path: "/api/delete"}}
	ex := newTestExecutor(t, m)
	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })

	// Stopped: empty union, no attribution map.
	res := ex.ModelsResult(ctx)
	if len(res.Models) != 0 {
		t.Fatalf("ModelsResult().Models on stopped engine = %v, want empty", res.Models)
	}
	if len(res.ByEngine) != 0 {
		t.Fatalf("ModelsResult().ByEngine on stopped engine = %v, want empty", res.ByEngine)
	}

	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	res = ex.ModelsResult(ctx)
	wantModels := []string{"llama3.2:1b"} // fake engine's seeded model
	if !reflect.DeepEqual(res.Models, wantModels) {
		t.Fatalf("ModelsResult().Models = %v, want %v", res.Models, wantModels)
	}
	wantByEngine := map[string][]string{"fake": {"llama3.2:1b"}}
	if !reflect.DeepEqual(res.ByEngine, wantByEngine) {
		t.Fatalf("ModelsResult().ByEngine = %v, want %v", res.ByEngine, wantByEngine)
	}

	// Deleting the last model is a successful empty inventory, not an unknown
	// response: preserve the engine key so consumers can clear stale state.
	if _, err := ex.Action(ctx, "fake", "delete_model", json.RawMessage(`{"name":"llama3.2:1b"}`)); err != nil {
		t.Fatalf("delete last model: %v", err)
	}
	res = ex.ModelsResult(ctx)
	if len(res.Models) != 0 {
		t.Fatalf("ModelsResult().Models after last delete = %v, want empty", res.Models)
	}
	if want := map[string][]string{"fake": {}}; !reflect.DeepEqual(res.ByEngine, want) {
		t.Fatalf("ModelsResult().ByEngine after last delete = %v, want %v", res.ByEngine, want)
	}
}

// loadedActionManifest is testEngineManifest with both list_models (/api/tags)
// and loaded_models (/api/ps) actions declared, so the executor surfaces
// LoadedByEngine from the fake engine's resident set.
func loadedActionManifest() *Manifest {
	m := testEngineManifest(fakeEngineBin)
	m.Actions["list_models"] = Action{
		HTTP:   &ActionHTTP{Method: "GET", Path: "/api/tags"},
		Result: &ActionResult{Array: "models", Field: "name"},
	}
	m.Actions["loaded_models"] = Action{
		HTTP:   &ActionHTTP{Method: "GET", Path: "/api/ps"},
		Result: &ActionResult{Array: "models", Field: "name"},
	}
	return m
}

// setLoaded drives the fake engine's resident set via its /testctl/loaded
// endpoint, standing in for an explicit load/unload or a TTL eviction.
func setLoaded(t *testing.T, ex *Executor, names []string) {
	t.Helper()
	st, err := ex.Status("fake")
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	body, _ := json.Marshal(map[string][]string{"names": names})
	url := fmt.Sprintf("http://127.0.0.1:%d/testctl/loaded", st.Port)
	resp, err := http.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("set loaded: %v", err)
	}
	_ = resp.Body.Close()
}

// TestModelsResultLoaded covers the loaded surface: a running engine that
// declares loaded_models reports its resident subset in LoadedByEngine, and an
// eviction leaves the engine key present with an empty list ("running, nothing
// loaded") while the installed list is unchanged.
func TestModelsResultLoaded(t *testing.T) {
	ex := newTestExecutor(t, loadedActionManifest())
	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })

	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	res := ex.ModelsResult(ctx)
	if want := map[string][]string{"fake": {"llama3.2:1b"}}; !reflect.DeepEqual(res.LoadedByEngine, want) {
		t.Fatalf("LoadedByEngine = %v, want %v", res.LoadedByEngine, want)
	}

	// Evict everything: the key stays with an empty list.
	setLoaded(t, ex, nil)
	res = ex.ModelsResult(ctx)
	if want := map[string][]string{"fake": {}}; !reflect.DeepEqual(res.LoadedByEngine, want) {
		t.Fatalf("LoadedByEngine after evict = %v, want %v", res.LoadedByEngine, want)
	}
	if want := []string{"llama3.2:1b"}; !reflect.DeepEqual(res.Models, want) {
		t.Fatalf("Models after evict = %v, want %v (residency must not change the installed list)", res.Models, want)
	}
}

func TestWatcherPublishesFirstEmptyLoadedInventory(t *testing.T) {
	reg := NewRegistry()
	reg.engines["fake"] = loadedActionManifest()
	updates := make(chan modelsChangedParams, 2)
	ex := NewExecutor(reg, NewReporter(nil), func(method string, params any) {
		if method == modelsChangedMethod {
			update, ok := params.(modelsChangedParams)
			if ok {
				updates <- update
			}
		}
	}, t.TempDir())
	ex.loadedPollInterval = time.Hour
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() { cancel(); _ = ex.Stop("fake") })
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start fake engine: %v", err)
	}
	setLoaded(t, ex, nil)
	go ex.watchLoaded(ctx)
	ex.pokeLoaded()
	select {
	case update := <-updates:
		loaded, known := update.Models.LoadedByEngine["fake"]
		if update.Engine != "fake" || !known || loaded == nil || len(loaded) != 0 {
			t.Fatalf("initial loaded inventory = %+v", update)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("first empty loaded inventory was not published")
	}
}

// TestModelsResultNoLoadedActionOmitsKey confirms an engine with no loaded_models
// action contributes no LoadedByEngine key (the map stays nil), so the field is
// omitted on the wire for engines that don't support loaded reporting.
func TestModelsResultNoLoadedActionOmitsKey(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	m.Actions["list_models"] = Action{
		HTTP:   &ActionHTTP{Method: "GET", Path: "/api/tags"},
		Result: &ActionResult{Array: "models", Field: "name"},
	}
	// No loaded_models action.
	ex := newTestExecutor(t, m)
	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	if res := ex.ModelsResult(ctx); res.LoadedByEngine != nil {
		t.Fatalf("LoadedByEngine = %v, want nil (no loaded_models action)", res.LoadedByEngine)
	}
}

// TestSweepLoadedSeedsThenEmitsOnChange covers the watcher's core diff without
// goroutines or timers: the first sweep publishes its baseline, a subsequent
// identical sweep reports no change, and an eviction carries a fresh result.
func TestSweepLoadedSeedsThenEmitsOnChange(t *testing.T) {
	ex := newTestExecutor(t, loadedActionManifest())
	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}

	// Seed sweep: baseline is {fake:[llama3.2:1b]}.
	_, prev, _ := ex.sweepLoaded(ctx, nil, true)
	if want := map[string][]string{"fake": {"llama3.2:1b"}}; !reflect.DeepEqual(prev, want) {
		t.Fatalf("seed baseline = %v, want %v", prev, want)
	}

	// No residency change -> no engine reported changed.
	changed, prev, _ := ex.sweepLoaded(ctx, prev, true)
	if len(changed) != 0 {
		t.Fatalf("unchanged sweep reported %v, want none", changed)
	}

	// Evict everything -> fake changes; payload carries the empty loaded set.
	setLoaded(t, ex, nil)
	changed, _, res := ex.sweepLoaded(ctx, prev, true)
	if !reflect.DeepEqual(changed, []string{"fake"}) {
		t.Fatalf("changed = %v, want [fake]", changed)
	}
	if want := map[string][]string{"fake": {}}; !reflect.DeepEqual(res.LoadedByEngine, want) {
		t.Fatalf("pushed LoadedByEngine = %v, want %v", res.LoadedByEngine, want)
	}
}

// TestSweepLoadedRetainsLastGoodOnTransientMiss covers the anti-churn guard: an
// engine that drops out of a sweep (a transient loaded_models miss, indistinct
// from a stop) is NOT reported as changed and keeps its last-good baseline, so a
// single blip can't emit a spurious drop-then-re-add pair.
func TestSweepLoadedRetainsLastGoodOnTransientMiss(t *testing.T) {
	ex := newTestExecutor(t, loadedActionManifest())
	prev := map[string][]string{"fake": {"llama3.2:1b"}}
	// The engine isn't started, so ModelsResult reports it neither running nor
	// queryable: LoadedByEngine has no "fake" key this sweep.
	changed, next, _ := ex.sweepLoaded(context.Background(), prev, true)
	if len(changed) != 0 {
		t.Fatalf("a disappeared engine reported %v changed, want none", changed)
	}
	if want := map[string][]string{"fake": {"llama3.2:1b"}}; !reflect.DeepEqual(next, want) {
		t.Fatalf("baseline after miss = %v, want last-good %v", next, want)
	}
}

func TestChangedEngines(t *testing.T) {
	tests := []struct {
		name      string
		prev, cur map[string][]string
		want      []string
	}{
		{name: "no change", prev: map[string][]string{"a": {"x"}}, cur: map[string][]string{"a": {"x"}}, want: nil},
		{name: "reorder is not a change", prev: map[string][]string{"a": {"x", "y"}}, cur: map[string][]string{"a": {"y", "x"}}, want: nil},
		{name: "value changed", prev: map[string][]string{"a": {"x"}}, cur: map[string][]string{"a": {"x", "y"}}, want: []string{"a"}},
		{name: "new key reported", prev: map[string][]string{}, cur: map[string][]string{"a": {"x"}}, want: []string{"a"}},
		{name: "disappeared key not reported", prev: map[string][]string{"a": {"x"}, "b": {"y"}}, cur: map[string][]string{"a": {"x"}}, want: nil},
		{name: "present-empty vs nil is not a change", prev: map[string][]string{"a": {}}, cur: map[string][]string{"a": nil}, want: nil},
		{name: "multiple changed sorted", prev: nil, cur: map[string][]string{"b": {"1"}, "a": {"2"}}, want: []string{"a", "b"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := changedEngines(tc.prev, tc.cur); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("changedEngines = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestSameStringSet(t *testing.T) {
	tests := []struct {
		name string
		a, b []string
		want bool
	}{
		{name: "nil equals empty", a: nil, b: []string{}, want: true},
		{name: "same order", a: []string{"x", "y"}, b: []string{"x", "y"}, want: true},
		{name: "different order", a: []string{"x", "y"}, b: []string{"y", "x"}, want: true},
		{name: "different length", a: []string{"x"}, b: []string{"x", "y"}, want: false},
		{name: "different elements", a: []string{"x"}, b: []string{"y"}, want: false},
		{name: "duplicates matter", a: []string{"x", "x"}, b: []string{"x", "y"}, want: false},
		{name: "same multiset with dups", a: []string{"x", "x", "y"}, b: []string{"y", "x", "x"}, want: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := sameStringSet(tc.a, tc.b); got != tc.want {
				t.Fatalf("sameStringSet(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

// TestSameReadRequestMatchesShippedManifests pins which bundled engines answer
// list_models and loaded_models from one endpoint, so ModelsResult fetches it
// once per sweep for them and twice only where the requests really differ.
func TestSameReadRequestMatchesShippedManifests(t *testing.T) {
	reg := NewRegistry()
	if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
		t.Fatalf("load bundled manifests: %v", err)
	}
	want := map[string]bool{"lmstudio": true, "llamacpp": true, "ollama": false}
	for name, shared := range want {
		mf, ok := reg.Get(name)
		if !ok {
			t.Fatalf("bundled manifest %q missing", name)
		}
		if got := sameReadRequest(mf.Actions["list_models"], mf.Actions["loaded_models"]); got != shared {
			t.Errorf("%s: sameReadRequest = %v, want %v", name, got, shared)
		}
	}
}

func TestSameReadRequestRejectsDifferingOrSideEffectingActions(t *testing.T) {
	get := func(path string) Action { return Action{HTTP: &ActionHTTP{Method: "GET", Path: path}} }
	restart := get("/m")
	restart.RestartAfter = true
	cases := []struct {
		name string
		a, b Action
		want bool
	}{
		{"identical GET", get("/m"), get("/m"), true},
		{"different path", get("/m"), get("/n"), false},
		{"different method", get("/m"), Action{HTTP: &ActionHTTP{Method: "POST", Path: "/m"}}, false},
		{"restart after", get("/m"), restart, false},
		{"cmd action", Action{Cmd: []string{"{cli}", "ls"}}, Action{Cmd: []string{"{cli}", "ls"}}, false},
	}
	for _, tc := range cases {
		if got := sameReadRequest(tc.a, tc.b); got != tc.want {
			t.Errorf("%s: sameReadRequest = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestModelsResultSharedEndpoint: when both actions hit the same endpoint, the
// single response still yields both the installed and the loaded views.
func TestModelsResultSharedEndpoint(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	m.Actions["list_models"] = Action{
		HTTP:   &ActionHTTP{Method: "GET", Path: "/api/tags"},
		Result: &ActionResult{Array: "models", Field: "name"},
	}
	m.Actions["loaded_models"] = Action{
		HTTP:   &ActionHTTP{Method: "GET", Path: "/api/tags"},
		Result: &ActionResult{Array: "models", Field: "name"},
	}
	ex := newTestExecutor(t, m)
	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	res := ex.ModelsResult(ctx)
	want := map[string][]string{"fake": {"llama3.2:1b"}}
	if !reflect.DeepEqual(res.ByEngine, want) {
		t.Fatalf("ByEngine = %v, want %v", res.ByEngine, want)
	}
	if !reflect.DeepEqual(res.LoadedByEngine, want) {
		t.Fatalf("LoadedByEngine = %v, want %v", res.LoadedByEngine, want)
	}
}

// TestSweepReusesInventoryBetweenReconciles: between full sweeps the watcher
// checks residency only, so an engine with separate list and loaded endpoints
// keeps its last inventory, and the next full sweep picks up the change.
func TestSweepReusesInventoryBetweenReconciles(t *testing.T) {
	m := loadedActionManifest()
	m.Actions["delete_model"] = Action{HTTP: &ActionHTTP{Method: "DELETE", Path: "/api/delete"}}
	ex := newTestExecutor(t, m)
	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}

	if res := ex.modelsSweep(ctx, true); !reflect.DeepEqual(res.ByEngine, map[string][]string{"fake": {"llama3.2:1b"}}) {
		t.Fatalf("full sweep ByEngine = %v", res.ByEngine)
	}
	if _, err := ex.Action(ctx, "fake", "delete_model", json.RawMessage(`{"name":"llama3.2:1b"}`)); err != nil {
		t.Fatalf("delete: %v", err)
	}

	res := ex.modelsSweep(ctx, false)
	if !reflect.DeepEqual(res.ByEngine, map[string][]string{"fake": {"llama3.2:1b"}}) {
		t.Fatalf("residency-only sweep ByEngine = %v, want the cached inventory", res.ByEngine)
	}
	if _, ok := res.LoadedByEngine["fake"]; !ok {
		t.Fatal("residency-only sweep must still report the loaded set")
	}

	if res := ex.modelsSweep(ctx, true); !reflect.DeepEqual(res.ByEngine, map[string][]string{"fake": {}}) {
		t.Fatalf("full sweep after delete ByEngine = %v, want empty", res.ByEngine)
	}
}

// TestSweepQueriesInventoryOnCacheMiss: a residency-only sweep with nothing
// cached (first sweep, or the engine just restarted) still reads the inventory.
func TestSweepQueriesInventoryOnCacheMiss(t *testing.T) {
	ex := newTestExecutor(t, loadedActionManifest())
	ctx := context.Background()
	t.Cleanup(func() { _ = ex.Stop("fake") })
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatalf("start: %v", err)
	}
	if res := ex.modelsSweep(ctx, false); !reflect.DeepEqual(res.ByEngine, map[string][]string{"fake": {"llama3.2:1b"}}) {
		t.Fatalf("ByEngine = %v, want the queried inventory", res.ByEngine)
	}
}

// TestModelsResultCarriesAndWithdrawsTheStrictCapability covers the
// engine-manager hop of the strict admission capability: the inventory serves
// the capability the broker last pushed, on /v1/models and engine:models, and a
// push without one withdraws it instead of keeping the previous value.
func TestModelsResultCarriesAndWithdrawsTheStrictCapability(t *testing.T) {
	var out bytes.Buffer
	ex := NewExecutor(NewRegistry(), NewReporter(nil), func(string, any) {}, t.TempDir())
	m := NewManager(NewCodec(&out), ex, nil, "")
	ready := nodepolicy.Advertisement{
		Revision: "21",
		Roles:    map[string]nodepolicy.AdvertisedRole{},
		StrictAdmission: &nodepolicy.StrictAdmissionCapability{Version: nodepolicy.StrictAdmissionVersion, Models: []nodepolicy.StrictAdmissionModel{{
			Engine: "ollama", Model: "m:latest", MaxConcurrent: 2,
			Identity: nodepolicy.StrictModelIdentity{EngineRunID: "engine-run-1", SettingsFingerprint: strings.Repeat("a", 64), ModelDigest: strings.Repeat("b", 64)},
		}}},
	}
	pushProjection(t, m, string(mustMarshal(t, map[string]any{"policy": ready, "pausedLocally": false, "managementBlocked": false})))
	fromHTTP, fromRPC, _, _ := servedPolicy(t, m, &out)
	for name, got := range map[string]*nodepolicy.Advertisement{"/v1/models": fromHTTP, "engine:models": fromRPC} {
		if got == nil || got.StrictAdmission.State() != nodepolicy.StrictCapabilityValid || !reflect.DeepEqual(got.StrictAdmission.Models, ready.StrictAdmission.Models) {
			t.Fatalf("%s policy = %+v, want the pushed capability %+v", name, got, ready.StrictAdmission)
		}
	}

	withdrawn := ready
	withdrawn.StrictAdmission = nil
	pushProjection(t, m, string(mustMarshal(t, map[string]any{"policy": withdrawn, "pausedLocally": false, "managementBlocked": false})))
	fromHTTP, fromRPC, _, _ = servedPolicy(t, m, &out)
	for name, got := range map[string]*nodepolicy.Advertisement{"/v1/models": fromHTTP, "engine:models": fromRPC} {
		if got == nil || got.Revision != "21" || got.StrictAdmission != nil {
			t.Fatalf("%s policy after withdrawal = %+v, want revision 21 without a capability", name, got)
		}
	}
}
