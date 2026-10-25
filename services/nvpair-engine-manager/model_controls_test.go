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
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modeldownload"
	"nvpair-shared/nodepolicy"
)

func TestModelControlsInventoryFailureDiagnostics(t *testing.T) {
	for _, tc := range []struct {
		name                                    string
		shared, failList, failLoaded, transport bool
	}{
		{"shared HTTP failure", true, true, true, false},
		{"separate list HTTP failure", false, true, false, false},
		{"separate loaded HTTP failure", false, false, true, false},
		{"shared transport failure", true, true, true, true},
		{"separate transport failures", false, true, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, _ := memoryModelsFixture(t)
			ex := h.ex
			st := ex.engines["ollama"]
			st.port = 24001
			if tc.shared {
				act := st.manifest.Actions["loaded_models"]
				act.HTTP = &ActionHTTP{Method: http.MethodGet, Path: "/api/tags"}
				st.manifest.Actions["loaded_models"] = act
			}
			for _, action := range []string{"list_models", "loaded_models"} {
				act := st.manifest.Actions[action]
				httpAction := *act.HTTP
				httpAction.Path += "?token=j304-url-secret"
				act.HTTP = &httpAction
				st.manifest.Actions[action] = act
			}
			healthy := *st.manifest
			healthy.Engine = "healthy"
			ex.reg.engines["healthy"] = &healthy
			ex.engines["healthy"] = &engineState{manifest: &healthy, plat: &Platform{}, running: true, port: 24002, proc: &managedProc{}}
			phase := "seed"
			ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
				if r.Method != http.MethodGet || (r.URL.Host != "127.0.0.1:24001" && r.URL.Host != "127.0.0.1:24002") ||
					(r.URL.Path != "/api/tags" && r.URL.Path != "/api/ps") {
					t.Errorf("unexpected inventory request: %s %s", r.Method, r.URL)
					return nil, errors.New("unexpected inventory request")
				}
				if r.URL.Host == "127.0.0.1:24002" {
					return mutationJSON(http.StatusOK, `{"models":[{"name":"sibling"}]}`)
				}
				failing := phase == "failure" && ((r.URL.Path == "/api/tags" && tc.failList) || (r.URL.Path == "/api/ps" && tc.failLoaded))
				if failing {
					if tc.transport {
						return nil, errors.New("j304-transport-secret https://credential:private@fixture.invalid/secret")
					}
					return mutationJSON(http.StatusServiceUnavailable, "j304-body-secret")
				}
				if phase == "empty" {
					return mutationJSON(http.StatusOK, `{"models":[{"name":""}]}`)
				}
				return mutationJSON(http.StatusOK, `{"models":[{"name":"seed"}]}`)
			})
			_, baseline, seed := ex.sweepLoaded(context.Background(), nil, false)
			if !reflect.DeepEqual(seed.ByEngine["ollama"], []string{"seed"}) || !reflect.DeepEqual(baseline["ollama"], []string{"seed"}) {
				t.Fatal("inventory/loaded positive did not seed the existing owner")
			}
			var capture bytes.Buffer
			previous := slog.Default()
			slog.SetDefault(slog.New(slog.NewJSONHandler(&capture, &slog.HandlerOptions{Level: slog.LevelDebug})))
			t.Cleanup(func() { slog.SetDefault(previous) })
			slog.Debug("j304 diagnostic capture enabled")
			phase = "failure"
			_, next, result := ex.sweepLoaded(context.Background(), baseline, false)
			if !reflect.DeepEqual(result.ByEngine["healthy"], []string{"sibling"}) || !reflect.DeepEqual(result.LoadedByEngine["healthy"], []string{"sibling"}) ||
				!reflect.DeepEqual(next["ollama"], []string{"seed"}) {
				t.Fatal("inventory failure lost healthy sibling or last-good loaded baseline")
			}
			_, listPresent := result.ByEngine["ollama"]
			_, loadedPresent := result.LoadedByEngine["ollama"]
			if listPresent == tc.failList || loadedPresent == tc.failLoaded {
				t.Fatal("failed action manufactured known-empty or removed its successful sibling action")
			}
			seen := map[string]int{}
			marker := false
			decoder := json.NewDecoder(bytes.NewReader(capture.Bytes()))
			for {
				var record struct {
					Message  string `json:"msg"`
					Level    string `json:"level"`
					Engine   string `json:"engine"`
					Action   string `json:"action"`
					Category string `json:"category"`
					Status   int    `json:"status"`
				}
				if err := decoder.Decode(&record); err == io.EOF {
					break
				} else if err != nil {
					t.Fatalf("decode actual debug capture: %v", err)
				}
				if record.Message == "j304 diagnostic capture enabled" && record.Level == "DEBUG" {
					marker = true
					continue
				}
				if record.Engine == "ollama" && (record.Action == "list_models" || record.Action == "loaded_models") {
					category, status := "http-status", http.StatusServiceUnavailable
					if tc.transport {
						category, status = "action-failed", 0
					}
					if record.Message != "engine:models "+record.Action+" failed" || record.Level != "DEBUG" || record.Category != category || record.Status != status {
						t.Fatalf("unsafe or unclassified inventory diagnostic: %+v", record)
					}
					seen[record.Action]++
				}
			}
			wantList, wantLoaded := 0, 0
			if tc.failList {
				wantList = 1
			}
			if tc.failLoaded {
				wantLoaded = 1
			}
			if !marker || seen["list_models"] != wantList || seen["loaded_models"] != wantLoaded {
				t.Fatalf("enabled capture missed the actual failure owner: marker=%v actions=%v", marker, seen)
			}
			for _, secret := range []string{"j304-body-secret", "j304-transport-secret", "j304-url-secret", "credential:private", "fixture.invalid/secret"} {
				if strings.Contains(capture.String(), secret) {
					t.Fatalf("inventory diagnostics leaked sentinel %q", secret)
				}
			}
			phase = "empty"
			_, _, empty := ex.sweepLoaded(context.Background(), next, false)
			if empty.ByEngine["ollama"] == nil || len(empty.ByEngine["ollama"]) != 0 || empty.LoadedByEngine["ollama"] == nil || len(empty.LoadedByEngine["ollama"]) != 0 ||
				!reflect.DeepEqual(empty.ByEngine["healthy"], []string{"sibling"}) {
				t.Fatal("post-failure all-invalid success lost known-empty or useful sibling")
			}
		})
	}
}

// These tests reuse the existing in-memory inventory fixture and HTTP seam.
// They never start an engine, manager, catalog server, listener or command.
func modelControlsExecutor(t *testing.T) *Executor {
	t.Helper()
	h := newMemoryEvidenceHarness(t)
	st := h.ex.engines["ollama"]
	st.manifest.ModelNaming = engines.ModelNamingExact
	st.manifest.Actions = map[string]Action{
		"warm":          {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/load"}},
		"cool":          {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/unload"}},
		"erase":         {HTTP: &ActionHTTP{Method: http.MethodDelete, Path: "/delete"}},
		pullModelAction: {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/pull"}},
		"opaque":        {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/opaque"}},
		"list_models":   {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/list"}, Result: &ActionResult{Array: "models", Field: "name"}},
		"loaded_models": {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/loaded"}, Result: &ActionResult{Array: "models", Field: "name"}},
	}
	st.manifest.Operations = map[string]OperationBinding{
		engines.OperationLoad:   {Action: "warm", ModelParam: "target"},
		engines.OperationUnload: {Action: "cool", ModelParam: "target"},
		engines.OperationDelete: {Action: "erase", ModelParam: "target"},
		engines.OperationPull:   {Action: pullModelAction, ModelParam: "target"},
	}
	h.ex.client = &http.Client{Transport: memoryModelsTransport(func(*http.Request) (*http.Response, error) {
		t.Error("unexpected engine request")
		return nil, errors.New("unexpected engine request")
	})}
	h.ex.modelMutationParent = func(context.Context, string) error {
		t.Error("unexpected model mutation")
		return errors.New("unexpected model mutation")
	}
	return h.ex
}

func modelControlsJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode fixture: %v", err)
	}
	return raw
}

func modelControlsPullEntry(t *testing.T, event ProgressEvent, source, id, model string) modeldownload.Entry {
	t.Helper()
	if id == "" || event.ID != id || event.Engine != "ollama" || event.Model != model || event.Op != "pull" || event.Message != "" {
		t.Fatalf("pull progress lost target identity or exposed a message: %+v", event)
	}
	state := event.DownloadState
	if state == nil {
		t.Fatal("pull progress omitted download state")
	}
	if err := state.Validate(); err != nil {
		t.Fatalf("invalid pull download state: %v", err)
	}
	if state.SourceID != source || state.RecoveryRequired || len(state.Entries) != 1 {
		t.Fatalf("unexpected pull snapshot: %+v", state)
	}
	entry := state.Entries[0]
	if entry.ID != id || entry.Engine != event.Engine || entry.Model != model {
		t.Fatalf("snapshot entry does not match progress: %+v", entry)
	}
	return entry
}

func TestModelDownloadPullIdentityFollowsDeclaredBinding(t *testing.T) {
	test := func(name string, binding *OperationBinding, params, want string) {
		t.Run(name, func(t *testing.T) {
			manifest := &Manifest{Actions: map[string]Action{pullModelAction: {}}}
			if binding != nil {
				manifest.Operations = map[string]OperationBinding{engines.OperationPull: *binding}
			}
			model, err := manifest.pullProgressModel(json.RawMessage(params))
			if err != nil || model != want {
				t.Fatalf("pull identity=%q error=%v, want %q", model, err, want)
			}
		})
	}
	test("explicit target overrides conventional aliases", &OperationBinding{Action: pullModelAction, ModelParam: "target"}, `{"target":"actual","name":"other","model":"another"}`, "actual")
	test("explicit default model overrides name", &OperationBinding{Action: pullModelAction}, `{"model":"actual","name":"other"}`, "actual")
	test("explicit name overrides model", &OperationBinding{Action: pullModelAction, ModelParam: "name"}, `{"name":"actual","model":"other"}`, "actual")
	test("custom key retains exact casing", &OperationBinding{Action: pullModelAction, ModelParam: "TARGET"}, `{"TARGET":"actual","target":"opaque","name":"other"}`, "actual")
	test("conventional name alias remains supported", nil, `{"name":"actual"}`, "actual")
	test("conventional name keeps alias precedence", nil, `{"name":"actual","model":"other"}`, "actual")
	test("conventional model remains supported", nil, `{"model":"actual"}`, "actual")
}

func TestModelDownloadFailedBoundPullFencesActualNativeModel(t *testing.T) {
	ex := modelControlsExecutor(t)
	acceptModelMutations(ex)
	root := t.TempDir()
	t.Setenv("OLLAMA_MODELS", root)
	params := json.RawMessage(`{"target":"actual","name":"other"}`)
	var effects int
	var artifact ollamaFixture
	ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
		effects++
		body, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
		if r.Method != http.MethodPost || r.URL.Path != "/pull" || !bytes.Equal(body, params) {
			return nil, errors.New("pull changed its native dispatch or caller params")
		}
		var native struct{ Target string }
		if err := json.Unmarshal(body, &native); err != nil {
			return nil, err
		}
		artifact = writeOllamaModel(t, root, native.Target, 16, 3)
		return mutationJSON(http.StatusOK, `{"error":"fixture native failure after content write"}`)
	})
	result, err := ex.Action(context.Background(), "ollama", pullModelAction, params)
	if err == nil || result != nil || effects != 1 || artifact.manifestRel == "" {
		t.Fatalf("failed native pull result=%s error=%v effects=%d artifact=%+v", result, err, effects, artifact)
	}
	snapshot := ex.downloads.snapshot()
	if err := snapshot.Validate(); err != nil {
		t.Fatalf("invalid failed pull snapshot: %v", err)
	}
	if len(snapshot.Entries) != 1 {
		t.Fatalf("failed pull entries=%+v", snapshot)
	}
	entry := snapshot.Entries[0]
	if entry.Model != "actual" || entry.Phase != modeldownload.PhaseFailed || entry.Reason != modeldownload.ReasonDownloadFailed || !entry.BlocksReadiness {
		t.Fatalf("failed native model was not fenced: %+v", entry)
	}
	markers, err := ex.downloads.store.load()
	if err != nil {
		t.Fatalf("read failed pull journal: %v", err)
	}
	marker, found := markers[downloadContentKey{"ollama", "actual"}]
	if !found || len(markers) != 1 || marker.ID != entry.ID || marker.Model != "actual" {
		t.Fatalf("journal does not identify native target: %+v", markers)
	}
	if _, err := ex.ModelLoad(context.Background(), "ollama", "actual"); !errors.Is(err, errDownloadBusy) {
		t.Fatalf("typed Load bypassed actual-model fence: %v", err)
	}
	if _, err := ex.Action(context.Background(), "ollama", "warm", params); !errors.Is(err, errDownloadBusy) {
		t.Fatalf("raw Load bypassed actual-model fence: %v", err)
	}
	server := &controlServer{exec: ex}
	query := modelQueryValues("ollama", "actual")
	listing := httptest.NewRecorder()
	server.handleModelFiles(listing, httptest.NewRequest(http.MethodGet, controlModelFilesPath+"?"+query.Encode(), nil))
	query.Set("path", artifact.manifestRel)
	file := httptest.NewRecorder()
	server.handleModelFile(file, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+query.Encode(), nil))
	if listing.Code != http.StatusConflict || listing.Flushed || file.Code != http.StatusConflict || file.Flushed || effects != 1 {
		t.Fatalf("failed model escaped export/Load fence: listing=%d file=%d native effects=%d", listing.Code, file.Code, effects)
	}
	release, err := ex.acquireModelRead("ollama", "other", false)
	if err != nil {
		t.Fatalf("unrelated name alias was incorrectly fenced: %v", err)
	}
	release()
}

func TestModelDownloadTypedPullRejectsMismatchWithDeclaredTargetBeforeEffects(t *testing.T) {
	ex := modelControlsExecutor(t)
	var effects, withdrawals, notifications int
	ex.modelMutationParent = func(context.Context, string) error { withdrawals++; return nil }
	ex.emit = func(string, any) { notifications++ }
	ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
		effects++
		return mutationJSON(http.StatusOK, `{"status":"success"}`)
	})
	before := ex.downloads.snapshot()
	result, err := ex.PullModelStream(context.Background(), "ollama", "other", json.RawMessage(`{"target":"actual","name":"other"}`))
	if err == nil || err.Error() != "pull model does not match action parameters" || result != nil ||
		effects != 0 || withdrawals != 0 || notifications != 0 || !before.Equal(ex.downloads.snapshot()) || len(ex.Errors()) != 0 {
		t.Fatalf("typed mismatch crossed reservation/effect boundary: result=%s error=%v effects=%d withdrawals=%d notifications=%d", result, err, effects, withdrawals, notifications)
	}
	markers, err := ex.downloads.store.load()
	if err != nil || len(markers) != 0 {
		t.Fatalf("typed mismatch created unsafe content: %+v, %v", markers, err)
	}
}

func modelControlsInvalidIdentifiers() []struct{ name, model string } {
	return []struct{ name, model string }{
		{"empty", ""},
		{"leading newline", "\nprivate-model"},
		{"trailing tab", "private-model\t"},
		{"embedded newline", "private\nmodel"},
		{"NUL", "private\x00model"},
		{"DEL", "private\x7fmodel"},
		{"C1 lower bound", "private\u0080model"},
		{"C1 upper bound", "private\u009fmodel"},
		{"invalid UTF8", "private\xffmodel"},
		{"513 bytes", strings.Repeat("s", 513)},
		{"513 bytes before trim", " " + strings.Repeat("s", 512)},
		{"513 multibyte bytes", strings.Repeat("界", 171)},
	}
}

func modelControlsWantRefusal(t *testing.T, err error, model string) {
	t.Helper()
	want := nodepolicy.ValidateModelIdentifier(model)
	if want == nil || err == nil || err.Error() != want.Error() {
		t.Fatalf("refusal = %v, want fixed shared error %v", err, want)
	}
}

func TestModelControlsCatalogRawBeforeTrimAndFallback(t *testing.T) {
	for _, tc := range modelControlsInvalidIdentifiers() {
		if tc.model == "" {
			continue // An absent optional primary identifier may use its alias.
		}
		t.Run(tc.name, func(t *testing.T) {
			if got := normalizeOllamaRows([]ollamaLibraryRow{{Name: tc.model, Model: "fallback"}, {Name: "sibling"}}); len(got) != 1 || got[0].ID != "sibling" {
				t.Fatalf("Ollama retained a malformed row or lost sibling: %+v", got)
			}
			if got := normalizeHFRows([]hfModelRow{{ID: tc.model, ModelID: "owner/fallback"}, {ID: "owner/sibling"}}); len(got) != 1 || got[0].ID != "owner/sibling" {
				t.Fatalf("HF retained a malformed row or lost sibling: %+v", got)
			}
			bad, good := modelControlsLlamaRow(tc.model), modelControlsLlamaRow("owner/sibling")
			bad.ModelID = "owner/fallback"
			if got := normalizeLlamaCPPRows([]hfModelRow{bad, good}); len(got) != 1 || got[0].ID != "owner/sibling:Q4_K_M" {
				t.Fatalf("llama retained a malformed row or lost sibling: %+v", got)
			}
		})
	}
	for _, tc := range []struct{ name, primary, fallback, want string }{
		{"ordinary spaces", "  valid  ", "", "valid"},
		{"empty primary", "", "fallback", "fallback"},
		{"space primary", " ", "fallback", "fallback"},
		{"512 bytes", strings.Repeat("a", 512), "", strings.Repeat("a", 512)},
		{"512 multibyte bytes", strings.Repeat("界", 170) + "aa", "", strings.Repeat("界", 170) + "aa"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ollama := normalizeOllamaRows([]ollamaLibraryRow{{Name: tc.primary, Model: tc.fallback}})
			hf := normalizeHFRows([]hfModelRow{{ID: tc.primary, ModelID: tc.fallback}})
			if len(ollama) != 1 || ollama[0].ID != tc.want || len(hf) != 1 || hf[0].ID != tc.want {
				t.Fatalf("valid identifier changed: Ollama=%+v HF=%+v", ollama, hf)
			}
		})
	}
}

func modelControlsLlamaRow(id string) hfModelRow {
	return hfModelRow{ID: id, Gated: json.RawMessage("false"), Tags: []string{"gguf"}, PipelineTag: "text-generation", LastModified: "2026-10-24T00:00:00Z", Siblings: []hfSibling{{RFilename: "model-Q4_K_M.gguf"}}}
}

func TestModelControlsCatalogDerivedIdentifierLimit(t *testing.T) {
	for _, tc := range []struct {
		name   string
		length int
		want   bool
	}{
		{"512 final bytes", 512, true},
		{"513 final bytes", 513, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			repo := "o/" + strings.Repeat("a", tc.length-len("o/:Q4_K_M"))
			got, ok := llamaCPPModel(modelControlsLlamaRow(repo))
			if ok != tc.want || (ok && len(got.ID) != tc.length) {
				t.Fatalf("derived identifier: ok=%v bytes=%d, want ok=%v bytes=%d", ok, len(got.ID), tc.want, tc.length)
			}
		})
	}
}

func TestModelControlsCatalogJSONSkipsIndividualInvalidFields(t *testing.T) {
	for _, tc := range []struct{ name, raw string }{
		{"escaped newline", `"\nprivate-model"`},
		{"DEL", `"private\u007fmodel"`},
		{"C1", `"private\u0085model"`},
		{"invalid UTF8", "\"private\xffmodel\""},
		{"wrong type", `123`},
		{"null", `null`},
		{"513 bytes", `"` + strings.Repeat("s", 513) + `"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var ollama []ollamaLibraryRow
			if err := json.Unmarshal([]byte(`[{"name":`+tc.raw+`,"model":"fallback"},{"name":"sibling"}]`), &ollama); err != nil {
				t.Fatalf("decode Ollama listing: %v", err)
			}
			if got := normalizeOllamaRows(ollama); len(got) != 1 || got[0].ID != "sibling" {
				t.Fatalf("Ollama filtering = %+v", got)
			}
			var hf []hfModelRow
			if err := json.Unmarshal([]byte(`[{"id":`+tc.raw+`,"modelId":"owner/fallback"},{"id":"owner/sibling"}]`), &hf); err != nil {
				t.Fatalf("decode HF listing: %v", err)
			}
			if got := normalizeHFRows(hf); len(got) != 1 || got[0].ID != "owner/sibling" {
				t.Fatalf("HF filtering = %+v", got)
			}
		})
	}
}

func TestModelControlsInventoryAndDigestFiltering(t *testing.T) {
	spec := &ActionResult{Array: "models", Field: "name", Digest: "digest"}
	for _, tc := range modelControlsInvalidIdentifiers() {
		t.Run(tc.name, func(t *testing.T) {
			// Keep invalid UTF-8 in its raw string field instead of letting the
			// fixture encoder itself replace the malformed byte.
			rawName := modelControlsJSON(t, tc.model)
			if tc.name == "invalid UTF8" {
				rawName = json.RawMessage("\"private\xffmodel\"")
			}
			raw := json.RawMessage(`{"models":[{"name":` + string(rawName) + `,"digest":"` + memoryTestDigest + `"},{"name":"sibling","digest":"` + memoryTestDigest + `"}]}`)
			got, ok := extractStringsResult(raw, spec)
			if !ok || !reflect.DeepEqual(got, []string{"sibling"}) {
				t.Fatalf("inventory = %v, %v; want sibling, known", got, ok)
			}
			if got := extractDigests(raw, spec); !reflect.DeepEqual(got, map[string]string{"sibling": memoryTestDigest}) {
				t.Fatalf("digest keys = %v", got)
			}
			allBad := json.RawMessage(`{"models":[{"name":` + string(rawName) + `,"digest":"` + memoryTestDigest + `"}]}`)
			if got, ok := extractStringsResult(allBad, spec); !ok || got == nil || len(got) != 0 {
				t.Fatalf("filtered inventory = %v, %v; want non-nil known-empty", got, ok)
			}
			filtered := *spec
			filtered.Match = &ResultMatch{Field: "kind", In: []string{"loaded"}}
			if got, ok := extractStringsResult(raw, &filtered); !ok || len(got) != 0 {
				t.Fatalf("row match no longer excludes unclassified models: %v, %v", got, ok)
			}
		})
	}
	for _, model := range []string{strings.Repeat("a", 512), strings.Repeat("界", 170) + "aa", " spaced "} {
		raw := modelControlsJSON(t, map[string]any{"models": []map[string]string{{"name": model, "digest": memoryTestDigest}}})
		if got, ok := extractStringsResult(raw, spec); !ok || !reflect.DeepEqual(got, []string{model}) || extractDigests(raw, spec)[model] != memoryTestDigest {
			t.Fatalf("valid inventory or digest changed: %v, %v", got, ok)
		}
	}
}

func TestModelControlsInventoryCacheAndKnownEmptyRefresh(t *testing.T) {
	ex := modelControlsExecutor(t)
	var listingCalls atomic.Int32
	var body atomic.Value
	body.Store(`{"models":[{"name":"sibling"}]}`)
	ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodGet || (r.URL.Path != "/list" && r.URL.Path != "/loaded") {
			return nil, fmt.Errorf("unexpected inventory request")
		}
		if r.URL.Path == "/list" {
			listingCalls.Add(1)
		}
		return mutationJSON(http.StatusOK, body.Load().(string))
	})
	first := ex.modelsSweep(context.Background(), true)
	if !reflect.DeepEqual(first.Models, []string{"sibling"}) {
		t.Fatalf("seed inventory = %+v", first)
	}
	body.Store(`{"models":[{"name":"private\nmodel"}]}`)
	second := ex.modelsSweep(context.Background(), true)
	for _, inventory := range []map[string][]string{second.ByEngine, second.LoadedByEngine} {
		models, present := inventory["ollama"]
		if !present || models == nil || len(models) != 0 {
			t.Fatalf("all-bad presence = %v, want known-empty", inventory)
		}
	}
	if len(second.Models) != 0 {
		t.Fatalf("last-good names survived refresh: %v", second.Models)
	}
	before := listingCalls.Load()
	third := ex.modelsSweep(context.Background(), false)
	if listingCalls.Load() != before || third.ByEngine["ollama"] == nil || len(third.ByEngine["ollama"]) != 0 {
		t.Fatalf("empty inventory cache missed or revived last-good: %+v", third)
	}
	ex.inventoryMu.Lock()
	ex.inventory["ollama"] = []string{"private\nmodel", strings.Repeat("x", 513), strings.Repeat("v", 512)}
	ex.inventoryMu.Unlock()
	if got := ex.cachedInventory()["ollama"]; !reflect.DeepEqual(got, []string{strings.Repeat("v", 512)}) {
		t.Fatalf("cached identifier filtering = %v", got)
	}
}

func TestModelControlsCatalogCacheReturnBoundaries(t *testing.T) {
	valid := CatalogModel{ID: strings.Repeat("v", 512), Name: strings.Repeat("v", 512)}
	badID := CatalogModel{ID: "private\nmodel", Name: "sibling"}
	badName := CatalogModel{ID: "sibling", Name: strings.Repeat("x", 513)}
	rows := []CatalogModel{badID, valid, badName}
	for _, tc := range []struct {
		name            string
		fetched, failed time.Time
	}{
		{"fresh", time.Now(), time.Time{}},
		{"failure backoff", time.Now().Add(-2 * catalogCacheTTL), time.Now()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cache := &listingCache{models: rows, fetched: tc.fetched, failed: tc.failed, fetch: func(context.Context) ([]CatalogModel, error) {
				t.Error("cache hit attempted a fetch")
				return nil, errors.New("unexpected fetch")
			}}
			got, at, err := cache.get(context.Background())
			if err != nil || !at.Equal(tc.fetched) || !reflect.DeepEqual(got, []CatalogModel{valid}) {
				t.Fatalf("cache return = %+v, %v, %v", got, at, err)
			}
		})
	}
	for _, target := range []struct{ platform, arch string }{{"darwin", "arm64"}, {"windows", "amd64"}} {
		if got := filterForTarget(rows, target.platform, target.arch); !reflect.DeepEqual(got, []CatalogModel{valid}) {
			t.Fatalf("target return retained invalid ID/name: %+v", got)
		}
	}
	c := newCatalogService()
	c.ollamaOnce.Do(func() { c.ollama = rows })
	got, err := c.Catalog(context.Background(), "ollama", "windows", "amd64", "")
	if err != nil || !reflect.DeepEqual(got.Models, []CatalogModel{valid}) {
		t.Fatalf("embedded cached return = %+v, %v", got, err)
	}
}

func TestModelControlsCatalogFilteredRefreshReplacesLastGood(t *testing.T) {
	for _, source := range []string{"lmstudio", "llamacpp", "search"} {
		t.Run(source, func(t *testing.T) {
			var calls int
			cache := &listingCache{label: source, models: []CatalogModel{{ID: "old", Name: "old"}}, fetched: time.Now().Add(-2 * catalogCacheTTL)}
			cache.fetch = func(context.Context) ([]CatalogModel, error) {
				calls++
				switch source {
				case "lmstudio":
					return normalizeHFRows([]hfModelRow{{ID: "\nprivate-model", ModelID: "fallback"}}), nil
				case "llamacpp":
					return normalizeLlamaCPPRows([]hfModelRow{modelControlsLlamaRow("\nowner/model")}), nil
				default:
					return []CatalogModel{{ID: "private\nmodel", Name: "private\nmodel"}}, nil
				}
			}
			for range 2 {
				got, at, err := cache.get(context.Background())
				if err != nil || at.IsZero() || got == nil || len(got) != 0 || !cache.failed.IsZero() {
					t.Fatalf("filtered refresh = %+v, %v, %v; failed=%v", got, at, err, cache.failed)
				}
			}
			if calls != 1 {
				t.Fatalf("known-empty cache made %d fetches, want one", calls)
			}
		})
	}
}

func TestModelControlsCatalogFailureStillKeepsUsableCache(t *testing.T) {
	cache := &listingCache{label: "fake", models: []CatalogModel{{ID: "sibling", Name: "sibling"}, {ID: "bad\nname", Name: "bad\nname"}}, fetched: time.Now().Add(-2 * catalogCacheTTL), fetch: func(context.Context) ([]CatalogModel, error) { return nil, errors.New("fixture fetch failure") }}
	got, _, err := cache.get(context.Background())
	if err != nil || !reflect.DeepEqual(got, []CatalogModel{{ID: "sibling", Name: "sibling"}}) || cache.failed.IsZero() {
		t.Fatalf("failed refresh changed stale-cache contract: %+v, %v", got, err)
	}
}

// Done is evaluated when get enters its in-flight wait. This makes the
// coalescing assertion deterministic without sleeps or production hooks.
type modelControlsWaitingContext struct {
	context.Context
	waiting chan struct{}
	once    sync.Once
}

func (c *modelControlsWaitingContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.waiting) })
	return c.Context.Done()
}

func TestModelControlsCatalogCoalescedRefreshIsKnownEmpty(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	started, release := make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	cache := &listingCache{label: "fake", models: []CatalogModel{{ID: "old", Name: "old"}}, fetched: time.Now().Add(-2 * catalogCacheTTL)}
	cache.fetch = func(ctx context.Context) ([]CatalogModel, error) {
		if calls.Add(1) == 1 {
			close(started)
		}
		select {
		case <-release:
			return []CatalogModel{{ID: "private\nmodel", Name: "private\nmodel"}}, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	type result struct {
		models []CatalogModel
		err    error
	}
	results := make(chan result, 2)
	go func() { models, _, err := cache.get(ctx); results <- result{models, err} }()
	select {
	case <-started:
	case <-ctx.Done():
		t.Fatal("fake catalog fetch did not start")
	}
	waiter := &modelControlsWaitingContext{Context: ctx, waiting: make(chan struct{})}
	go func() { models, _, err := cache.get(waiter); results <- result{models, err} }()
	select {
	case <-waiter.waiting:
	case <-ctx.Done():
		t.Fatal("second caller did not enter coalesced wait")
	}
	close(release)
	for range 2 {
		select {
		case got := <-results:
			if got.err != nil || got.models == nil || len(got.models) != 0 {
				t.Fatalf("coalesced refresh = %+v, %v", got.models, got.err)
			}
		case <-ctx.Done():
			t.Fatal("coalesced result did not settle")
		}
	}
	if calls.Load() != 1 {
		t.Fatalf("coalesced refresh made %d fetches", calls.Load())
	}
}

func TestModelControlsTypedOperationsRejectBeforeEffects(t *testing.T) {
	for _, tc := range modelControlsInvalidIdentifiers() {
		t.Run(tc.name, func(t *testing.T) {
			for _, op := range []struct {
				name string
				call func(*Executor, string) (json.RawMessage, error)
			}{
				{"load", func(ex *Executor, model string) (json.RawMessage, error) {
					return ex.ModelLoad(context.Background(), "ollama", model)
				}},
				{"unload", func(ex *Executor, model string) (json.RawMessage, error) {
					return ex.ModelUnload(context.Background(), "ollama", model)
				}},
				{"delete", func(ex *Executor, model string) (json.RawMessage, error) {
					return ex.ModelDelete(context.Background(), "ollama", model)
				}},
				{"pull", func(ex *Executor, model string) (json.RawMessage, error) {
					return ex.PullModelStream(context.Background(), "ollama", model, nil)
				}},
			} {
				t.Run(op.name, func(t *testing.T) {
					ex := modelControlsExecutor(t)
					before := ex.inventoryDigests["ollama"]
					ex.emit = func(string, any) { t.Error("rejection emitted a notification") }
					result, err := op.call(ex, tc.model)
					modelControlsWantRefusal(t, err, tc.model)
					if result != nil || !reflect.DeepEqual(before, ex.inventoryDigests["ollama"]) {
						t.Fatal("rejection returned a value or changed identity state")
					}
				})
			}
		})
	}
}

func TestModelControlsBoundActionsRejectRawBeforeEffects(t *testing.T) {
	for _, tc := range []struct{ name, raw string }{
		{"leading newline", `"\nprivate-model"`},
		{"DEL", `"private\u007fmodel"`},
		{"C1", `"private\u0085model"`},
		{"invalid UTF8", "\"private\xffmodel\""},
		{"513 bytes", `"` + strings.Repeat("s", 513) + `"`},
		{"number", `23`},
		{"null", `null`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, action := range []string{"warm", "cool", "erase", pullModelAction} {
				t.Run(action, func(t *testing.T) {
					ex := modelControlsExecutor(t)
					ex.emit = func(string, any) { t.Error("rejection emitted a notification") }
					before := ex.inventoryDigests["ollama"]
					result, err := ex.Action(context.Background(), "ollama", action, json.RawMessage(`{"target":`+tc.raw+`}`))
					want := nodepolicy.ValidateModelIdentifierJSON(json.RawMessage(tc.raw))
					if err == nil || want == nil || err.Error() != want.Error() || result != nil {
						t.Fatalf("Action refusal = %s, %v; want fixed %v", result, err, want)
					}
					if !reflect.DeepEqual(before, ex.inventoryDigests["ollama"]) {
						t.Fatal("invalid action entered mutation")
					}
				})
			}
		})
	}
}

func TestModelControlsValidBoundActionsPreserveRequestAndPermissions(t *testing.T) {
	for _, action := range []string{"warm", "cool", "erase", pullModelAction, "opaque"} {
		t.Run(action, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			var calls, mutations int
			ex.modelMutationParent = func(context.Context, string) error { mutations++; return nil }
			params := modelControlsJSON(t, map[string]string{"target": strings.Repeat("v", 512), "prompt": "opaque\ncontent\u0085"})
			if action == "opaque" {
				params = json.RawMessage(`{"model":"opaque\ncontent","name":"opaque\u0085content"}`)
			}
			ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				body, err := io.ReadAll(r.Body)
				if err != nil {
					return nil, err
				}
				if !bytes.Equal(body, params) {
					return nil, errors.New("valid action bytes changed")
				}
				if action == pullModelAction {
					return mutationJSON(http.StatusOK, `{"accepted":true,"status":"success"}`)
				}
				return mutationJSON(http.StatusOK, `{"accepted":true}`)
			})
			result, err := ex.Action(context.Background(), "ollama", action, params)
			var response struct {
				Accepted bool   `json:"accepted"`
				Status   string `json:"status"`
			}
			if err != nil {
				t.Fatalf("valid Action: %v", err)
			}
			if err := json.Unmarshal(result, &response); err != nil || !response.Accepted || calls != 1 {
				t.Fatalf("valid action result=%s calls=%d decode=%v", result, calls, err)
			}
			if action == pullModelAction && response.Status != "success" {
				t.Fatalf("pull omitted native success terminal: %+v", response)
			}
			wantMutations := 0
			if action == "erase" || action == pullModelAction {
				wantMutations = 1
			}
			if mutations != wantMutations {
				t.Fatalf("mutation permissions changed: %d, want %d", mutations, wantMutations)
			}
		})
	}
}

func TestModelControlsValidIdentifierStillRequiresMutationPermission(t *testing.T) {
	ex := modelControlsExecutor(t)
	denied := errors.New("fixture mutation permission denied")
	var admissions int
	ex.modelMutationParent = func(context.Context, string) error { admissions++; return denied }
	_, err := ex.ModelDelete(context.Background(), "ollama", strings.Repeat("v", 512))
	if !errors.Is(err, denied) || admissions != 1 {
		t.Fatalf("valid identifier bypassed mutation permission: %v, admissions=%d", err, admissions)
	}
}

func TestModelControlsTyped512UsesCustomOperationKey(t *testing.T) {
	for _, operation := range engines.Operations() {
		t.Run(operation, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			model := strings.Repeat("v", 512)
			action, raw, err := ex.modelOperationCall("ollama", operation, model)
			if err != nil {
				t.Fatalf("resolve valid model: %v", err)
			}
			var params map[string]string
			if err := json.Unmarshal(raw, &params); err != nil {
				t.Fatalf("decode resolved params: %v", err)
			}
			binding, ok := ex.engines["ollama"].manifest.operationBinding(operation)
			if !ok || action != binding.Action || !reflect.DeepEqual(params, map[string]string{"target": model}) {
				t.Fatalf("binding changed: %s %+v", action, params)
			}
		})
	}
}

func TestModelControlsPullExplicitParametersRejectBeforeMutation(t *testing.T) {
	for _, key := range []string{"target", "model", "name"} {
		t.Run(key, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			bad := strings.Repeat("s", 513)
			params := modelControlsJSON(t, map[string]string{"target": "valid", key: bad})
			result, err := ex.PullModelStream(context.Background(), "ollama", "valid", params)
			modelControlsWantRefusal(t, err, bad)
			if result != nil {
				t.Fatalf("invalid pull returned %s", result)
			}
		})
	}
}

func TestModelControlsManagerPullRefusesBeforeProgressAndReporter(t *testing.T) {
	for _, tc := range []struct{ name, params string }{
		{"name before fallback", `{"name":"\nprivate-model","model":"valid"}`},
		{"secondary model", `{"name":"valid","model":"private\u0085model"}`},
		{"bound key", `{"target":"private\u007fmodel"}`},
		{"invalid UTF8", "{\"target\":\"private\xffmodel\"}"},
		{"case folded alias invalid UTF8", "{\"NAME\":\"private\xffmodel\"}"},
		{"513 bytes", `{"target":"` + strings.Repeat("s", 513) + `"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			ex.emit = func(string, any) { t.Error("invalid manager request emitted progress/identity") }
			var output bytes.Buffer
			manager := &Manager{exec: ex, codec: NewCodec(&output)}
			id := json.RawMessage("1")
			params := json.RawMessage(`{"engine":"ollama","action":"pull_model","params":` + tc.params + `}`)
			manager.runAction(context.Background(), &Message{ID: &id, Params: params})
			var response Message
			if err := json.NewDecoder(&output).Decode(&response); err != nil {
				t.Fatalf("decode manager response: %v", err)
			}
			if response.Error == nil || response.Error.Code != -32602 || len(response.Result) != 0 {
				t.Fatalf("manager response = %+v", response)
			}
			if strings.Contains(response.Error.Message, "private") || strings.Contains(response.Error.Message, strings.Repeat("s", 32)) {
				t.Fatalf("identifier leaked in error: %q", response.Error.Message)
			}
			if len(ex.Errors()) != 0 {
				t.Fatalf("invalid model reached pull failure reporter: %+v", ex.Errors())
			}
		})
	}
}

func TestModelControlsManagerPullAccepts512CustomKey(t *testing.T) {
	ex := modelControlsExecutor(t)
	acceptModelMutations(ex)
	model := strings.Repeat("v", 512)
	var calls int
	ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
		calls++
		var body map[string]string
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			return nil, err
		}
		if !reflect.DeepEqual(body, map[string]string{"target": model}) {
			return nil, errors.New("custom pull key changed")
		}
		return mutationJSON(http.StatusOK, `{"status":"success"}`)
	})
	var output bytes.Buffer
	manager := &Manager{exec: ex, codec: NewCodec(&output)}
	id := json.RawMessage("1")
	manager.runAction(context.Background(), &Message{ID: &id, Params: modelControlsJSON(t, map[string]any{"engine": "ollama", "action": pullModelAction, "params": map[string]string{"target": model}})})
	var response Message
	if err := json.NewDecoder(&output).Decode(&response); err != nil {
		t.Fatalf("decode manager response: %v", err)
	}
	if response.Error != nil || calls != 1 {
		t.Fatalf("valid manager pull = %+v calls=%d", response, calls)
	}
}

func TestModelControlsParamsOnlyPullPreserves512Identifier(t *testing.T) {
	for _, key := range []string{"target", "name", "model"} {
		t.Run(key, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			ex.engines["ollama"].manifest.Operations[engines.OperationPull] = OperationBinding{Action: pullModelAction, ModelParam: key}
			acceptModelMutations(ex)
			model := strings.Repeat("v", 512)
			var calls int
			ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
				calls++
				return mutationJSON(http.StatusOK, `{"status":"success"}`)
			})
			result, err := ex.PullModelStream(context.Background(), "ollama", "", modelControlsJSON(t, map[string]string{key: model}))
			if err != nil || calls != 1 || !json.Valid(result) {
				t.Fatalf("params-only pull = %s, %v, calls=%d", result, err, calls)
			}
		})
	}
}

func TestModelControlsCopyAndSourceQueriesRejectBeforeEffects(t *testing.T) {
	for _, tc := range modelControlsInvalidIdentifiers() {
		t.Run(tc.name, func(t *testing.T) {
			var ex *Executor
			source := &fakeModelSource{}
			result, err := ex.CopyModelFrom(context.Background(), source, "ollama", tc.model, func(copyProgress) { t.Error("invalid copy emitted progress") })
			modelControlsWantRefusal(t, err, tc.model)
			if result != (copyModelResult{}) || source.listed != 0 || len(source.fetchedPaths()) != 0 {
				t.Fatal("invalid copy reached source or returned a result")
			}
			var remote *remoteClient
			if _, err := remote.modelFiles(context.Background(), "ollama", tc.model, nil); err == nil {
				t.Fatal("remote file listing accepted invalid model")
			}
			body, start, err := remote.modelFile(context.Background(), "ollama", tc.model, "file", "", 0)
			modelControlsWantRefusal(t, err, tc.model)
			if body != nil || start != 0 {
				t.Fatal("invalid file request returned a value")
			}
			for _, path := range []string{controlModelFilesPath, controlModelFilePath} {
				q := url.Values{"engine": {"ollama"}, "model": {tc.model}, "path": {"file"}}
				req := httptest.NewRequest(http.MethodGet, path+"?"+q.Encode(), nil)
				recorder := httptest.NewRecorder()
				server := &controlServer{}
				if path == controlModelFilesPath {
					server.handleModelFiles(recorder, req)
				} else {
					server.handleModelFile(recorder, req)
				}
				if recorder.Code != http.StatusBadRequest || strings.Contains(recorder.Body.String(), "private") || strings.Contains(recorder.Body.String(), strings.Repeat("s", 32)) {
					t.Fatalf("source query refusal = %d %q", recorder.Code, recorder.Body.String())
				}
			}
		})
	}
}

func TestModelControlsSourceQueryAccepts512Unchanged(t *testing.T) {
	model := strings.Repeat("界", 170) + "aa"
	for _, method := range []string{http.MethodGet, http.MethodHead} {
		req := httptest.NewRequest(method, controlModelFilesPath+"?"+modelQueryValues("ollama", model).Encode(), nil)
		recorder := httptest.NewRecorder()
		engine, got, ok := modelQuery(recorder, req)
		if !ok || engine != "ollama" || got != model {
			t.Fatalf("valid query changed: %q %q %v", engine, got, ok)
		}
	}
}

func TestModelControlsRemoteManagerRejectsBeforePeerLookup(t *testing.T) {
	for _, method := range []string{"engine:remote-load-model", "engine:remote-unload-model", "engine:remote-delete-model", "engine:remote-copy-model", "engine:remote-pull-model"} {
		t.Run(method, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			var output bytes.Buffer
			// No peer directory or transport: crossing preflight cannot succeed.
			manager := &Manager{exec: ex, codec: NewCodec(&output)}
			id := json.RawMessage("1")
			msg := &Message{ID: &id, Method: method, Params: json.RawMessage(`{"node":"private-peer","engine":"ollama","model":"private\u0085model"}`)}
			if method == "engine:remote-copy-model" {
				manager.runRemoteCopy(context.Background(), msg)
			} else {
				manager.runRemote(context.Background(), msg)
			}
			var response Message
			if err := json.NewDecoder(&output).Decode(&response); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if response.Error == nil || response.Error.Code != -32602 || strings.Contains(response.Error.Message, "private") {
				t.Fatalf("remote model escaped preflight: %+v", response)
			}
		})
	}
}

func TestModelControlsTypedJSONRefusesOriginalInvalidUTF8(t *testing.T) {
	raw := json.RawMessage("{\"engine\":\"ollama\",\"model\":\"private\xffmodel\"}")
	var request modelActionRequest
	if err := json.Unmarshal(raw, &request); err == nil {
		t.Fatal("typed decode repaired invalid model UTF-8")
	}
	var pull pullRequest
	if err := json.Unmarshal(raw, &pull); err == nil {
		t.Fatal("pull decode repaired invalid model UTF-8")
	}
	for _, op := range []string{"load", "unload", "delete"} {
		recorder := httptest.NewRecorder()
		server := &controlServer{}
		server.handleModelAction(recorder, httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(raw)), op)
		if recorder.Code != http.StatusBadRequest {
			t.Fatalf("invalid model reached %s owner: HTTP %d", op, recorder.Code)
		}
	}
}

func TestModelControlsPullFailureReporterRejectsInvalidModel(t *testing.T) {
	ex := modelControlsExecutor(t)
	ex.emit = func(string, any) { t.Error("invalid failure identifier was published") }
	bad := "private\nmodel"
	want := nodepolicy.ValidateModelIdentifier(bad)
	if got := ex.reportPullFailed("ollama", bad, errors.New("fixture failure")); got != want.Error() || len(ex.Errors()) != 0 {
		t.Fatalf("invalid failure reached reporter: %q, %+v", got, ex.Errors())
	}
}

func TestModelControlsHTTPPullCustomKeyRefusalDoesNotReportFailure(t *testing.T) {
	ex := modelControlsExecutor(t)
	ex.emit = func(string, any) { t.Error("invalid HTTP pull emitted an engine notification") }
	server := &controlServer{exec: ex}
	params := modelControlsJSON(t, map[string]any{"engine": "ollama", "model": "valid", "params": map[string]string{"target": strings.Repeat("s", 513)}})
	recorder := httptest.NewRecorder()
	server.handlePull(recorder, httptest.NewRequest(http.MethodPost, controlPullPath, bytes.NewReader(params)))
	var frame streamFrame
	if err := json.NewDecoder(recorder.Body).Decode(&frame); err != nil {
		t.Fatalf("decode pull refusal frame: %v", err)
	}
	want := nodepolicy.ValidateModelIdentifier(strings.Repeat("s", 513))
	if frame.Type != "error" || frame.Message != want.Error() || len(ex.Errors()) != 0 {
		t.Fatalf("custom key rejection reached reporter or lost fixed error: %+v, %+v", frame, ex.Errors())
	}
}

func TestModelControlsTyped512ReachesOnlyFakeEngine(t *testing.T) {
	for _, op := range []string{engines.OperationLoad, engines.OperationUnload, engines.OperationDelete} {
		t.Run(op, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			acceptModelMutations(ex)
			model := strings.Repeat("v", 512)
			var calls int
			ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				var params map[string]string
				if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
					return nil, err
				}
				if !reflect.DeepEqual(params, map[string]string{"target": model}) {
					return nil, errors.New("typed model or custom key changed")
				}
				return mutationJSON(http.StatusOK, `{"accepted":true}`)
			})
			var err error
			switch op {
			case engines.OperationLoad:
				_, err = ex.ModelLoad(context.Background(), "ollama", model)
			case engines.OperationUnload:
				_, err = ex.ModelUnload(context.Background(), "ollama", model)
			case engines.OperationDelete:
				_, err = ex.ModelDelete(context.Background(), "ollama", model)
			}
			if err != nil || calls != 1 {
				t.Fatalf("typed operation = %v, calls=%d", err, calls)
			}
		})
	}
}

func TestModelControlsLoadedRowsKeepValidSiblingsAndMatch(t *testing.T) {
	spec := &ActionResult{Array: "models", Field: "id", Match: &ResultMatch{Field: "status.value", In: []string{"loaded"}}}
	raw := json.RawMessage(`{"models":[{"id":"private\u0085model","status":{"value":"loaded"}},{"id":"sibling","status":{"value":"loaded"}},{"id":"unloaded","status":{"value":"unloaded"}}]}`)
	got, ok := extractStringsResult(raw, spec)
	if !ok || !reflect.DeepEqual(got, []string{"sibling"}) {
		t.Fatalf("loaded filter = %+v, %v", got, ok)
	}
}

func TestModelControlsCatalogInvalidSecondaryCannotHideBehindPrimary(t *testing.T) {
	if got := normalizeOllamaRows([]ollamaLibraryRow{{Name: "valid", Model: "private\nmodel"}}); len(got) != 0 {
		t.Fatalf("Ollama retained invalid alias: %+v", got)
	}
	if got := normalizeHFRows([]hfModelRow{{ID: "valid", ModelID: "private\u0085model"}}); len(got) != 0 {
		t.Fatalf("HF retained invalid alias: %+v", got)
	}
	row := modelControlsLlamaRow("owner/valid")
	row.ModelID = "private\nmodel"
	if _, ok := llamaCPPModel(row); ok {
		t.Fatal("llama retained invalid alias")
	}
}

func modelControlsServing(model string) *engines.ServingConfig {
	ram, vram := int64(4096), int64(0)
	return &engines.ServingConfig{Version: 1, Models: []engines.ServingModel{{ModelID: model, MaxOutputTokens: 128,
		Memory:    engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram},
		Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: memoryTestDigest, Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}, Binding: &engines.ServingBinding{SourcePath: "fixture", ResolvedTarget: "fixture", SizeBytes: 100, MtimeNS: "1", SampleSHA256: memoryTestDigest}}},
	}}}
}

func TestModelControlsServingDeclarationOwnersRejectBeforeDerivation(t *testing.T) {
	for _, tc := range modelControlsInvalidIdentifiers() {
		t.Run(tc.name, func(t *testing.T) {
			config := modelControlsServing(tc.model)
			want := nodepolicy.ValidateModelIdentifier(tc.model)
			if got := servingDeclarationErrors(config)["serving.models.0.modelId"]; got != want.Error() {
				t.Fatalf("declaration error = %q, want %q", got, want)
			}
			platform := &Platform{Runtime: Runtime{Bin: "unused", Serving: config}}
			if err := platform.validate("windows/amd64"); err == nil || strings.Contains(err.Error(), "private") {
				t.Fatalf("platform accepted or echoed invalid model: %v", err)
			}
			ex := modelControlsExecutor(t)
			st := ex.engines["ollama"]
			preview := ex.previewLaunch(st, settings.Request{Settings: settings.Config{Serving: config}}, func(launchCommand, Runtime) (*llamaServingLaunch, error) {
				t.Error("invalid declaration reached compiler")
				return nil, errors.New("unexpected compiler")
			})
			if preview.Errors["serving.models.0.modelId"] != want.Error() {
				t.Fatalf("preview error = %v", preview.Errors)
			}
			if _, err := compileLlamaServing(launchCommand{}, platform.Runtime, false); err == nil || strings.Contains(err.Error(), "private") {
				t.Fatalf("llama compiler accepted or echoed invalid model: %v", err)
			}
			if _, err := deriveStrataServing(launchCommand{}, platform.Runtime); err == nil || strings.Contains(err.Error(), "private") {
				t.Fatalf("Strata derivation accepted or echoed invalid model: %v", err)
			}
			if err := validateLlamaModelID(tc.model); err == nil || strings.Contains(err.Error(), "private") {
				t.Fatalf("native model validator accepted or echoed invalid model: %v", err)
			}
			var nilExecutor *Executor
			_, _, err := nilExecutor.currentServingEnvelope(context.Background(), "ollama", tc.model)
			var refusal *ServingIdentityRefusal
			if !errors.As(err, &refusal) || strings.Contains(err.Error(), "private") {
				t.Fatalf("cold envelope acceptance = %v", err)
			}
		})
	}
}

func TestModelControlsServingEnvelopePublicationBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name, declaration, envelope string
		valid                       bool
	}{
		{"512 bytes", strings.Repeat("v", 512), strings.Repeat("v", 512), true},
		{"513 declaration", strings.Repeat("v", 513), "valid", false},
		{"513 envelope", "valid", strings.Repeat("v", 513), false},
		{"raw newline declaration", "\nprivate-model", "valid", false},
		{"C1 envelope", "valid", "private\u0085model", false},
		{"invalid UTF8 envelope", "valid", "private\xffmodel", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			config := modelControlsServing(tc.declaration)
			envelope := engines.ServingEnvelope{ModelID: tc.envelope, PerSlotContextTokens: 8192, TotalContextTokens: 8192, Slots: 1, MaxOutputTokens: 128, Memory: config.Models[0].Memory, NativeSourceDigest: memoryTestDigest}
			prepared := &llamaServingLaunch{Config: config, Envelopes: []engines.ServingEnvelope{envelope}}
			ex := modelControlsExecutor(t)
			st := ex.engines["ollama"]
			st.plat.Runtime.Serving, st.plat.Runtime.servingLaunch = config, prepared
			id := ex.modelIdentity().Engines["ollama"]
			if tc.valid {
				if len(servingDeclarationErrors(config)) != 0 || validateLlamaModelID(tc.declaration) != nil || len(id.ModelDigests) != 1 || id.Serving[tc.declaration].ModelID != tc.declaration {
					t.Fatalf("valid serving ID failed publication: %+v", id)
				}
			} else if len(servingEnvelopes(prepared)) != 0 || len(id.ModelDigests) != 0 || len(id.Serving) != 0 {
				t.Fatalf("invalid serving identifier published: %+v", id)
			}
		})
	}
}

func TestModelControlsDigestCachePublicationFiltersInvalidKeys(t *testing.T) {
	ex := modelControlsExecutor(t)
	valid := strings.Repeat("v", 512)
	ex.inventoryDigests["ollama"] = observedDigests{runID: memoryTestRun, models: map[string]string{valid: memoryTestDigest, "private\nmodel": memoryTestDigest, strings.Repeat("s", 513): memoryTestDigest}}
	if got := ex.modelIdentity().Engines["ollama"].ModelDigests; !reflect.DeepEqual(got, map[string]string{valid: memoryTestDigest}) {
		t.Fatalf("digest cache retained bad keys: %+v", got)
	}
}

func TestModelControlsServingJSONBeforeDecoderAndManifestMerge(t *testing.T) {
	for _, tc := range []struct{ name, raw string }{
		{"newline", `"\nprivate-model"`},
		{"invalid UTF8", "\"private\xffmodel\""},
		{"513 bytes", `"` + strings.Repeat("s", 513) + `"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			serving := `{"models":[{"modelId":` + tc.raw + `}]}`
			for _, manifest := range []string{
				`{"runtime":{"serving":` + serving + `},"platforms":{"windows/amd64":{}}}`,
				`{"platforms":{"windows/amd64":{"runtime":{"serving":` + serving + `}}}}`,
			} {
				if _, err := applyPlatformDefaults([]byte(manifest)); err == nil || strings.Contains(err.Error(), "private") {
					t.Fatalf("manifest merge accepted or echoed model: %v", err)
				}
				if _, err := NewRegistry().mergeOntoBundled([]byte(manifest)); err == nil || strings.Contains(err.Error(), "private") {
					t.Fatalf("override merge accepted or echoed model: %v", err)
				}
			}
			var output bytes.Buffer
			manager := &Manager{codec: NewCodec(&output)}
			id := json.RawMessage("1")
			var request settings.Request
			if manager.parse(&Message{ID: &id, Params: json.RawMessage(`{"settings":{"serving":` + serving + `}}`)}, &request) {
				t.Fatal("settings typed decoder repaired invalid model")
			}
			if _, err := parseStrataServingConfig([]byte(`{"model_name":` + tc.raw + `}`)); err == nil || strings.Contains(err.Error(), "private") {
				t.Fatalf("Strata model_name accepted or echoed: %v", err)
			}
		})
	}
	valid := strings.Repeat("v", 512)
	config, err := parseStrataServingConfig(modelControlsJSON(t, map[string]string{"model_name": valid}))
	if err != nil || config.ModelName != valid {
		t.Fatalf("512-byte Strata model_name = %q, %v", config.ModelName, err)
	}
}

func TestModelControlsOtherServingConstraintsRemainActive(t *testing.T) {
	config := modelControlsServing("valid")
	config.Version = 2
	if len(servingDeclarationErrors(config)) == 0 {
		t.Fatal("identifier validation replaced declaration validation")
	}
	if err := validateLlamaModelID("noncanonical:q4_k_m"); err == nil {
		t.Fatal("identifier validation replaced native canonicalization constraint")
	}
}

func TestModelControlsSettingsHTTPValidatesCompleteRawRequestBeforeRelay(t *testing.T) {
	request := func(model string) string {
		return `{"nodeId":"ignored-peer","engine":"ollama","settings":{"serving":{"models":[{"modelId":` + model + `}]}}}`
	}
	for _, tc := range []struct {
		name, body, model string
		accepted          bool
	}{
		{name: "original invalid UTF8", body: request("\"private\xffmodel\"")},
		{name: "escaped control", body: request(`"private\nmodel"`)},
		{name: "513 bytes", body: request(`"` + strings.Repeat("s", 513) + `"`)},
		{name: "unknown root field", body: `{"engine":"ollama","private":true}`},
		{name: "unknown settings field", body: `{"engine":"ollama","settings":{"private":true}}`},
		{name: "trailing document", body: request(`"valid"`) + `{}`},
		{name: "malformed tail", body: request(`"valid"`) + `x`},
		{name: "overflow after valid prefix", body: request(`"valid"`) + strings.Repeat(" ", 64*1024)},
		{name: "deliberate replacement character", body: request(`"private\ufffdmodel"`), model: "private\ufffdmodel", accepted: true},
		{name: "512 bytes", body: request(`"` + strings.Repeat("v", 512) + `"`), model: strings.Repeat("v", 512), accepted: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			var relays int
			ex.settingsParent = func(_ context.Context, method string, p settings.Request, caller string) (json.RawMessage, error) {
				relays++
				if !tc.accepted || method != "get" || caller != "pinned-peer" || p.NodeID != "" || p.Engine != "ollama" ||
					p.Settings.Serving == nil || len(p.Settings.Serving.Models) != 1 || p.Settings.Serving.Models[0].ModelID != tc.model {
					t.Errorf("unexpected settings relay: method=%s caller=%s request=%+v", method, caller, p)
				}
				return json.RawMessage(`{"accepted":true}`), nil
			}
			server := &controlServer{exec: ex, mesh: clustertrust.Open(t.TempDir())}
			recorder := httptest.NewRecorder()
			server.settingsHandler("get")(recorder, httptest.NewRequest(http.MethodPost, settingsPath+"get", strings.NewReader(tc.body)), "pinned-peer")
			if !tc.accepted {
				if recorder.Code != http.StatusBadRequest || relays != 0 || strings.Contains(recorder.Body.String(), "private") || strings.Contains(recorder.Body.String(), strings.Repeat("s", 32)) {
					t.Fatalf("raw settings refusal: HTTP %d relays=%d body=%q", recorder.Code, relays, recorder.Body.String())
				}
				return
			}
			var result struct{ Accepted bool }
			if err := json.NewDecoder(recorder.Body).Decode(&result); err != nil {
				t.Fatalf("decode settings result: %v", err)
			}
			if recorder.Code != http.StatusOK || relays != 1 || !result.Accepted {
				t.Fatalf("valid settings request: HTTP %d relays=%d result=%+v", recorder.Code, relays, result)
			}
		})
	}
}

func TestModelControlsBoundActionsRequireModelPresenceBeforeEffects(t *testing.T) {
	for _, tc := range []struct{ name, params string }{
		{"absent params", ""},
		{"root null", "null"},
		{"empty object", "{}"},
		{"unrelated content", `{"prompt":"opaque\ntext"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, action := range []string{"warm", "cool", "erase", pullModelAction} {
				t.Run(action, func(t *testing.T) {
					ex := modelControlsExecutor(t)
					ex.emit = func(string, any) { t.Error("missing model emitted a notification") }
					before := ex.inventoryDigests["ollama"]
					result, err := ex.Action(context.Background(), "ollama", action, json.RawMessage(tc.params))
					want := nodepolicy.ValidateModelIdentifierJSON(nil)
					if err == nil || err.Error() != want.Error() || result != nil || !reflect.DeepEqual(before, ex.inventoryDigests["ollama"]) || len(ex.Errors()) != 0 {
						t.Fatalf("missing bound model: result=%s error=%v", result, err)
					}
				})
			}
		})
	}
}

func TestModelControlsUnboundActionPreservesAbsentAndOpaqueParams(t *testing.T) {
	for _, tc := range []struct{ name, params string }{
		{"absent params", ""},
		{"root null", "null"},
		{"empty object", "{}"},
		{"opaque model fields", `{"model":null,"name":"opaque\ntext","target":"opaque\u0085text"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			var calls int
			ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				var body []byte
				if r.Body != nil {
					var err error
					body, err = io.ReadAll(r.Body)
					if err != nil {
						return nil, err
					}
				}
				want := tc.params
				if want == "null" {
					want = ""
				}
				if string(body) != want {
					return nil, errors.New("opaque params changed")
				}
				return mutationJSON(http.StatusOK, `{"accepted":true}`)
			})
			result, err := ex.Action(context.Background(), "ollama", "opaque", json.RawMessage(tc.params))
			if err != nil || calls != 1 || !json.Valid(result) {
				t.Fatalf("unbound action: result=%s error=%v calls=%d", result, err, calls)
			}
		})
	}
}

func TestModelControlsRemoteRequiredModelsRejectAbsentBeforePeerLookup(t *testing.T) {
	for _, tc := range []struct{ name, params string }{
		{"root null", "null"},
		{"empty object", "{}"},
		{"absent model", `{"node":"private-peer","engine":"ollama"}`},
		{"null model", `{"node":"private-peer","engine":"ollama","model":null}`},
		{"empty model", `{"node":"private-peer","engine":"ollama","model":""}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, method := range []string{"engine:remote-load-model", "engine:remote-unload-model", "engine:remote-delete-model", "engine:remote-copy-model", "engine:remote-pull-model"} {
				t.Run(method, func(t *testing.T) {
					var output bytes.Buffer
					manager := &Manager{codec: NewCodec(&output)}
					id := json.RawMessage("1")
					msg := &Message{ID: &id, Method: method, Params: json.RawMessage(tc.params)}
					if method == "engine:remote-copy-model" {
						manager.runRemoteCopy(context.Background(), msg)
					} else {
						manager.runRemote(context.Background(), msg)
					}
					var response Message
					if err := json.NewDecoder(&output).Decode(&response); err != nil {
						t.Fatalf("decode missing-model refusal: %v", err)
					}
					if response.Error == nil || response.Error.Code != -32602 || strings.Contains(response.Error.Message, "private") {
						t.Fatalf("missing remote model reached peer lookup: %+v", response)
					}
				})
			}
		})
	}
}

func TestModelControlsPullRejectsSuppliedEmptyOrNullModelBeforeEffects(t *testing.T) {
	for _, rawModel := range []string{`null`, `""`} {
		t.Run(rawModel, func(t *testing.T) {
			raw := json.RawMessage(`{"node":"private-peer","engine":"ollama","model":` + rawModel + `,"params":{"target":"valid"}}`)
			var request pullRequest
			if err := json.Unmarshal(raw, &request); err == nil {
				t.Fatal("pull decoder accepted supplied invalid optional model")
			}
			server := &controlServer{}
			recorder := httptest.NewRecorder()
			server.handlePull(recorder, httptest.NewRequest(http.MethodPost, controlPullPath, bytes.NewReader(raw)))
			if recorder.Code != http.StatusBadRequest || strings.Contains(recorder.Body.String(), "private") {
				t.Fatalf("supplied invalid model reached HTTP execution: %d %q", recorder.Code, recorder.Body.String())
			}
			var output bytes.Buffer
			manager := &Manager{codec: NewCodec(&output)}
			id := json.RawMessage("1")
			manager.runRemote(context.Background(), &Message{ID: &id, Method: "engine:remote-pull-model", Params: raw})
			var response Message
			if err := json.NewDecoder(&output).Decode(&response); err != nil {
				t.Fatalf("decode initiating refusal: %v", err)
			}
			if response.Error == nil || response.Error.Code != -32602 || strings.Contains(response.Error.Message, "private") {
				t.Fatalf("supplied invalid remote model reached peer lookup: %+v", response)
			}
		})
	}
}

func TestModelControlsHTTPPullRejectsBeforeStreamAdmission(t *testing.T) {
	for _, tc := range []struct{ name, params string }{
		{"custom control", `{"target":"private\nmodel"}`},
		{"custom invalid UTF8", "{\"target\":\"private\xffmodel\"}"},
		{"null bound field", `{"target":null}`},
		{"empty params", `{}`},
		{"opaque params without model", `{"prompt":"opaque\ntext"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			ex.emit = func(string, any) { t.Error("invalid HTTP pull emitted a notification") }
			before := ex.inventoryDigests["ollama"]
			server := &controlServer{exec: ex}
			recorder := httptest.NewRecorder()
			raw := json.RawMessage(`{"engine":"ollama","model":"valid","params":` + tc.params + `}`)
			server.handlePull(recorder, httptest.NewRequest(http.MethodPost, controlPullPath, bytes.NewReader(raw)))
			var frame streamFrame
			if err := json.NewDecoder(recorder.Body).Decode(&frame); err != nil {
				t.Fatalf("decode cold pull refusal: %v", err)
			}
			ex.progress.mu.Lock()
			admissions, subscribers := ex.progress.next, len(ex.progress.subs)
			ex.progress.mu.Unlock()
			if recorder.Code != http.StatusBadRequest || recorder.Flushed || recorder.Header().Get("Content-Type") != "application/json" ||
				admissions != 0 || subscribers != 0 || frame.Type != "error" || frame.Message == "" || strings.Contains(frame.Message, "private") ||
				len(ex.Errors()) != 0 || !reflect.DeepEqual(before, ex.inventoryDigests["ollama"]) {
				t.Fatalf("cold refusal crossed stream/effect boundary: HTTP %d flushed=%v admissions=%d subs=%d frame=%+v", recorder.Code, recorder.Flushed, admissions, subscribers, frame)
			}
		})
	}
}

func TestModelControlsCatalogCompleteBodyBeforeCacheReplacement(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		accepted   bool
	}{
		{name: "malformed suffix", body: `[]x`},
		{name: "second document", body: `[] {}`},
		{name: "overflow after array", body: `[]` + strings.Repeat(" ", maxCatalogBody-1)},
		{name: "truncated array", body: `[{"id":"owner/valid"}`},
		{name: "null document", body: `null`},
		{name: "complete empty array", body: `[]`, accepted: true},
		{name: "complete all-bad array", body: `[{"id":"private\nmodel"}]`, accepted: true},
		{name: "exact size bound", body: `[]` + strings.Repeat(" ", maxCatalogBody-2), accepted: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			catalog := newCatalogService()
			catalog.baseURL = "https://catalog.invalid/models"
			var calls int
			catalog.client = &http.Client{Transport: memoryModelsTransport(func(*http.Request) (*http.Response, error) {
				calls++
				return mutationJSON(http.StatusOK, tc.body)
			})}
			rows, err := catalog.fetchHF(context.Background(), url.Values{})
			if (err == nil) != tc.accepted || (!tc.accepted && rows != nil) {
				t.Fatalf("complete-body preflight: rows=%+v error=%v", rows, err)
			}
			previous := []CatalogModel{{ID: "owner/previous", Name: "owner/previous"}}
			previousAt := time.Now().Add(-2 * catalogCacheTTL)
			catalog.lmStudio.models, catalog.lmStudio.fetched = previous, previousAt
			got, at, err := catalog.lmStudio.get(context.Background())
			if err != nil {
				t.Fatalf("refresh with last-good cache: %v", err)
			}
			if tc.accepted {
				if got == nil || len(got) != 0 || !at.After(previousAt) || !catalog.lmStudio.failed.IsZero() {
					t.Fatalf("valid filtered refresh was not authoritative empty: models=%+v at=%v", got, at)
				}
			} else if !reflect.DeepEqual(got, previous) || !at.Equal(previousAt) || catalog.lmStudio.failed.IsZero() {
				t.Fatalf("incomplete body replaced last-good: models=%+v at=%v", got, at)
			}
			if _, _, err := catalog.lmStudio.get(context.Background()); err != nil || calls != 2 {
				t.Fatalf("cache/backoff missed after refresh: error=%v calls=%d", err, calls)
			}
		})
	}
}

func TestModelControlsHTTPParamsOnlyForwardingPreserves512Identifier(t *testing.T) {
	ex := modelControlsExecutor(t)
	model := strings.Repeat("v", 512)
	params := modelControlsJSON(t, map[string]string{"target": model, "prompt": "opaque\ntext"})
	var output bytes.Buffer
	manager := &Manager{exec: ex, codec: NewCodec(&output)}
	id := json.RawMessage("1")
	var p remoteParam
	if !manager.parse(&Message{ID: &id, Method: "engine:remote-pull-model", Params: modelControlsJSON(t, remoteParam{Node: "peer", Engine: "ollama", Params: params})}, &p) {
		t.Fatalf("params-only initiating preflight refused valid request: %s", output.String())
	}
	// This is the producer in runRemote, including its omitted optional Model.
	wire := modelControlsJSON(t, pullRequest{OpID: "forwarded-op", Engine: p.Engine, Model: p.Model, Params: p.Params})
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(wire, &fields); err != nil {
		t.Fatalf("decode forwarded wire body: %v", err)
	}
	if _, supplied := fields["model"]; supplied {
		t.Fatal("forwarding supplied an empty optional model")
	}
	var decoded pullRequest
	if err := json.Unmarshal(wire, &decoded); err != nil {
		t.Fatalf("destination decode of forwarded params-only request: %v", err)
	}
	if decoded.Model != "" || !bytes.Equal(decoded.Params, params) {
		t.Fatalf("forwarding changed optional model or params: %+v", decoded)
	}
	var calls, mutations int
	ex.modelMutationParent = func(context.Context, string) error { mutations++; return nil }
	ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
		calls++
		body, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
		if !bytes.Equal(body, params) {
			return nil, errors.New("forwarded params changed at engine boundary")
		}
		return mutationJSON(http.StatusOK, `{"status":"success"}`)
	})
	server := &controlServer{exec: ex}
	recorder := httptest.NewRecorder()
	server.handlePull(recorder, httptest.NewRequest(http.MethodPost, controlPullPath, bytes.NewReader(wire)))
	decoder := json.NewDecoder(recorder.Body)
	var terminal streamFrame
	for {
		var frame streamFrame
		if err := decoder.Decode(&frame); err == io.EOF {
			break
		} else if err != nil {
			t.Fatalf("decode forwarded pull stream: %v", err)
		}
		terminal = frame
	}
	var result struct{ Status string }
	if err := json.Unmarshal(terminal.Result, &result); err != nil {
		t.Fatalf("decode terminal pull result: %v", err)
	}
	if recorder.Code != http.StatusOK || !recorder.Flushed || terminal.Type != "result" || terminal.OpID != "forwarded-op" ||
		result.Status != "success" || calls != 1 || mutations != 1 || ex.progress.next != 1 || len(ex.progress.subs) != 0 || len(ex.Errors()) != 0 {
		t.Fatalf("valid forwarded pull lost stream semantics: HTTP %d terminal=%+v calls=%d mutations=%d", recorder.Code, terminal, calls, mutations)
	}
}

func TestModelControlsForwardedPullUsesDestinationBindingBeforeEffects(t *testing.T) {
	for _, tc := range []struct {
		name, model string
		accepted    bool
	}{
		{name: "invalid destination-only model", model: "private\nmodel"},
		{name: "original invalid UTF8 destination-only model", model: "private\xffmodel"},
		{name: "valid destination-only model", model: strings.Repeat("v", 512), accepted: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			initiator := modelControlsExecutor(t)
			destination := modelControlsExecutor(t)
			destination.engines["ollama"].manifest.Operations[engines.OperationPull] = OperationBinding{Action: pullModelAction, ModelParam: "remote_target"}
			rawModel := modelControlsJSON(t, tc.model)
			if tc.name == "original invalid UTF8 destination-only model" {
				rawModel = json.RawMessage("\"private\xffmodel\"")
			}
			params := json.RawMessage(`{"target":"opaque\ntext","remote_target":` + string(rawModel) + `,"prompt":"opaque\ntext"}`)
			var output bytes.Buffer
			manager := &Manager{exec: initiator, codec: NewCodec(&output)}
			id := json.RawMessage("1")
			var p remoteParam
			if !manager.parse(&Message{ID: &id, Method: "engine:remote-pull-model", Params: modelControlsJSON(t, remoteParam{Node: "peer", Engine: "ollama", Params: params})}, &p) {
				t.Fatalf("initiator inferred destination-only binding: %s", output.String())
			}
			wire := modelControlsJSON(t, pullRequest{OpID: "different-binding", Engine: p.Engine, Model: p.Model, Params: p.Params})
			var decoded pullRequest
			if err := json.Unmarshal(wire, &decoded); err != nil {
				t.Fatalf("destination decode of actual forwarded shape: %v", err)
			}
			if decoded.Model != "" || !bytes.Equal(decoded.Params, params) {
				t.Fatal("forwarded params-only shape changed")
			}
			var calls, mutations, notifications int
			destination.modelMutationParent = func(context.Context, string) error { mutations++; return nil }
			destination.emit = func(string, any) { notifications++ }
			destination.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				body, err := io.ReadAll(r.Body)
				if err != nil {
					return nil, err
				}
				if !bytes.Equal(body, params) {
					return nil, errors.New("destination changed opaque or custom params")
				}
				return mutationJSON(http.StatusOK, `{"status":"success"}`)
			})
			server := &controlServer{exec: destination}
			recorder := httptest.NewRecorder()
			server.handlePull(recorder, httptest.NewRequest(http.MethodPost, controlPullPath, bytes.NewReader(wire)))
			if !tc.accepted {
				var refusal streamFrame
				if err := json.NewDecoder(recorder.Body).Decode(&refusal); err != nil {
					t.Fatalf("decode destination refusal: %v", err)
				}
				want := nodepolicy.ValidateModelIdentifierJSON(rawModel)
				if recorder.Code != http.StatusBadRequest || recorder.Flushed || calls != 0 || mutations != 0 || notifications != 0 ||
					destination.progress.next != 0 || len(destination.progress.subs) != 0 || len(destination.Errors()) != 0 ||
					refusal.Type != "error" || refusal.Message != want.Error() {
					t.Fatalf("destination-only invalid model crossed effect boundary: HTTP %d refusal=%+v calls=%d mutations=%d notifications=%d", recorder.Code, refusal, calls, mutations, notifications)
				}
				return
			}
			decoder := json.NewDecoder(recorder.Body)
			var terminal streamFrame
			for {
				var frame streamFrame
				if err := decoder.Decode(&frame); err == io.EOF {
					break
				} else if err != nil {
					t.Fatalf("decode destination pull stream: %v", err)
				}
				terminal = frame
			}
			var result struct{ Status string }
			if err := json.Unmarshal(terminal.Result, &result); err != nil {
				t.Fatalf("decode destination terminal result: %v", err)
			}
			if recorder.Code != http.StatusOK || !recorder.Flushed || terminal.Type != "result" || terminal.OpID != "different-binding" ||
				result.Status != "success" || calls != 1 || mutations != 1 || destination.progress.next != 1 || len(destination.progress.subs) != 0 || len(destination.Errors()) != 0 {
				t.Fatalf("valid destination custom pull: HTTP %d terminal=%+v calls=%d mutations=%d", recorder.Code, terminal, calls, mutations)
			}
		})
	}
}

func TestModelControlsRemotePullExplicitCommonFieldsRejectBeforePeerLookup(t *testing.T) {
	for _, key := range []string{"name", "model", "NAME", "MODEL"} {
		t.Run(key, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			ex.emit = func(string, any) { t.Error("invalid common model emitted a notification") }
			var output bytes.Buffer
			manager := &Manager{exec: ex, codec: NewCodec(&output)}
			id := json.RawMessage("1")
			params := modelControlsJSON(t, map[string]string{"target": "valid", "remote_target": "valid", key: "private\nmodel"})
			manager.runRemote(context.Background(), &Message{ID: &id, Method: "engine:remote-pull-model", Params: modelControlsJSON(t, remoteParam{Node: "private-peer", Engine: "ollama", Params: params})})
			var response Message
			if err := json.NewDecoder(&output).Decode(&response); err != nil {
				t.Fatalf("decode initiating common-field refusal: %v", err)
			}
			want := nodepolicy.ValidateModelIdentifier("private\nmodel")
			if response.Error == nil || response.Error.Code != -32602 || response.Error.Message != want.Error() || len(ex.Errors()) != 0 {
				t.Fatalf("invalid explicit common model reached peer lookup: %+v", response)
			}
		})
	}
}

func TestModelControlsHTTPPullRequiresDestinationCustomBindingBeforeEffects(t *testing.T) {
	for _, tc := range []struct{ name, model, params string }{
		{"name cannot replace bound target", "", `{"name":"valid"}`},
		{"model cannot replace bound target", "", `{"model":"valid"}`},
		{"top-level model cannot replace bound target", "valid", `{"name":"valid","model":"valid"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			before := ex.inventoryDigests["ollama"]
			var calls, mutations, notifications int
			ex.modelMutationParent = func(context.Context, string) error { mutations++; return nil }
			ex.emit = func(string, any) { notifications++ }
			ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
				calls++
				return nil, errors.New("missing bound model reached engine transport")
			})
			wire := modelControlsJSON(t, pullRequest{Engine: "ollama", Model: tc.model, Params: json.RawMessage(tc.params)})
			server := &controlServer{exec: ex}
			recorder := httptest.NewRecorder()
			server.handlePull(recorder, httptest.NewRequest(http.MethodPost, controlPullPath, bytes.NewReader(wire)))
			var refusal streamFrame
			if err := json.NewDecoder(recorder.Body).Decode(&refusal); err != nil {
				t.Fatalf("decode missing-binding refusal: %v", err)
			}
			ex.progress.mu.Lock()
			admissions, subscribers := ex.progress.next, len(ex.progress.subs)
			ex.progress.mu.Unlock()
			want := nodepolicy.ValidateModelIdentifierJSON(nil)
			if recorder.Code != http.StatusBadRequest || recorder.Flushed || recorder.Header().Get("Content-Type") != "application/json" ||
				refusal.Type != "error" || refusal.Message != want.Error() || calls != 0 || mutations != 0 || notifications != 0 ||
				admissions != 0 || subscribers != 0 || len(ex.Errors()) != 0 || !reflect.DeepEqual(before, ex.inventoryDigests["ollama"]) {
				t.Fatalf("missing destination binding crossed effect boundary: HTTP %d refusal=%+v calls=%d mutations=%d notifications=%d admissions=%d subscribers=%d", recorder.Code, refusal, calls, mutations, notifications, admissions, subscribers)
			}
		})
	}
}

func TestModelControlsActionRequiresExactCustomKeyBeforeEffects(t *testing.T) {
	for _, tc := range []struct{ name, action, form string }{
		{"load body", "warm", "body"},
		{"unload body", "cool", "body"},
		{"delete body", "erase", "body"},
		{"pull body", pullModelAction, "body"},
		{"delete query", "erase", "query"},
		{"delete command", "erase", "command"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			st := ex.engines["ollama"]
			if tc.form == "query" {
				st.manifest.Actions[tc.action].HTTP.ParamsIn = actionHTTPParamsQuery
			} else if tc.form == "command" {
				st.manifest.Actions[tc.action] = Action{Cmd: []string{"unreachable-f6-child", "{target}"}}
			}
			before, beforeProc, beforeRun := ex.inventoryDigests["ollama"], st.proc, st.runID
			var calls, mutations, notifications int
			ex.emit = func(string, any) { notifications++ }
			ex.modelMutationParent = func(context.Context, string) error {
				mutations++
				return errors.New("fixture refuses mutation before any command can launch")
			}
			ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
				calls++
				return nil, errors.New("missing exact key reached engine HTTP")
			})
			result, err := ex.Action(context.Background(), "ollama", tc.action, json.RawMessage(`{"TARGET":"valid","name":"valid"}`))
			want := nodepolicy.ValidateModelIdentifierJSON(nil)
			if err == nil || err.Error() != want.Error() || result != nil || calls != 0 || mutations != 0 || notifications != 0 ||
				ex.progress.next != 0 || len(ex.progress.subs) != 0 || len(ex.Errors()) != 0 ||
				!reflect.DeepEqual(before, ex.inventoryDigests["ollama"]) || st.proc != beforeProc || st.runID != beforeRun {
				t.Fatalf("case variant satisfied exact binding: result=%s error=%v calls=%d mutations=%d notifications=%d", result, err, calls, mutations, notifications)
			}
		})
	}
}

func TestModelControlsPullRequiresExactBoundKeyBeforeEffects(t *testing.T) {
	for _, boundary := range []string{"manager", "direct pull", "destination HTTP", "default destination HTTP", "uppercase custom destination HTTP", "name destination HTTP", "name invalid common destination HTTP"} {
		t.Run(boundary, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			before := ex.inventoryDigests["ollama"]
			var calls, mutations, notifications int
			ex.emit = func(string, any) { notifications++ }
			ex.modelMutationParent = func(context.Context, string) error { mutations++; return errors.New("unexpected mutation") }
			ex.client.Transport = memoryModelsTransport(func(*http.Request) (*http.Response, error) {
				calls++
				return nil, errors.New("missing exact key reached engine HTTP")
			})
			params := json.RawMessage(`{"TARGET":"valid","name":"valid"}`)
			earlyPlainRefusal := ""
			if boundary == "default destination HTTP" {
				ex.engines["ollama"].manifest.Operations[engines.OperationPull] = OperationBinding{Action: pullModelAction}
				params = json.RawMessage(`{"MODEL":"valid","name":"valid"}`)
			} else if boundary == "uppercase custom destination HTTP" {
				ex.engines["ollama"].manifest.Operations[engines.OperationPull] = OperationBinding{Action: pullModelAction, ModelParam: "TARGET"}
				params = json.RawMessage(`{"target":"valid","name":"valid"}`)
			} else if boundary == "name destination HTTP" {
				ex.engines["ollama"].manifest.Operations[engines.OperationPull] = OperationBinding{Action: pullModelAction, ModelParam: "name"}
				params = json.RawMessage(`{"NAME":"valid","model":"valid"}`)
			} else if boundary == "name invalid common destination HTTP" {
				ex.engines["ollama"].manifest.Operations[engines.OperationPull] = OperationBinding{Action: pullModelAction, ModelParam: "name"}
				params = modelControlsJSON(t, map[string]string{"name": strings.Repeat("v", 512), "MODEL": "private\nmodel"})
				earlyPlainRefusal = "invalid JSON body\n"
			}
			want := nodepolicy.ValidateModelIdentifierJSON(nil)
			switch boundary {
			case "manager":
				var output bytes.Buffer
				manager := &Manager{exec: ex, codec: NewCodec(&output)}
				id := json.RawMessage("1")
				manager.runAction(context.Background(), &Message{ID: &id, Params: modelControlsJSON(t, actionParam{Engine: "ollama", Action: pullModelAction, Params: params})})
				var response Message
				if err := json.NewDecoder(&output).Decode(&response); err != nil {
					t.Fatalf("decode manager exact-key refusal: %v", err)
				}
				if response.Error == nil || response.Error.Code != -32602 || response.Error.Message != want.Error() || len(response.Result) != 0 {
					t.Fatalf("manager accepted missing exact key: %+v", response)
				}
			case "direct pull":
				result, err := ex.PullModelStream(context.Background(), "ollama", "", params)
				if err == nil || err.Error() != want.Error() || result != nil {
					t.Fatalf("direct pull accepted missing exact key: result=%s error=%v", result, err)
				}
			case "destination HTTP", "default destination HTTP", "uppercase custom destination HTTP", "name destination HTTP", "name invalid common destination HTTP":
				server := &controlServer{exec: ex}
				recorder := httptest.NewRecorder()
				wire := modelControlsJSON(t, pullRequest{Engine: "ollama", Params: params})
				server.handlePull(recorder, httptest.NewRequest(http.MethodPost, controlPullPath, bytes.NewReader(wire)))
				if earlyPlainRefusal != "" {
					body := recorder.Body.String()
					if recorder.Code != http.StatusBadRequest || recorder.Flushed || body != earlyPlainRefusal || strings.Contains(body, "private\nmodel") {
						t.Errorf("early common-field refusal: HTTP %d flushed=%v", recorder.Code, recorder.Flushed)
					}
				} else {
					var refusal streamFrame
					if err := json.NewDecoder(recorder.Body).Decode(&refusal); err != nil {
						t.Fatalf("decode destination exact-key refusal: %v", err)
					}
					if recorder.Code != http.StatusBadRequest || recorder.Flushed || refusal.Type != "error" || refusal.Message != want.Error() {
						t.Fatalf("destination accepted missing exact key: HTTP %d refusal=%+v", recorder.Code, refusal)
					}
				}
			}
			if calls != 0 || mutations != 0 || notifications != 0 || ex.progress.next != 0 || len(ex.progress.subs) != 0 ||
				len(ex.Errors()) != 0 || !reflect.DeepEqual(before, ex.inventoryDigests["ollama"]) {
				t.Fatalf("missing exact key crossed effects: calls=%d mutations=%d notifications=%d admissions=%d", calls, mutations, notifications, ex.progress.next)
			}
		})
	}
}

func TestModelControlsActionPreservesOpaqueCustomCaseVariant(t *testing.T) {
	for _, tc := range []struct{ name, action, form string }{
		{"load body", "warm", "body"},
		{"unload body", "cool", "body"},
		{"delete body", "erase", "body"},
		{"pull body", pullModelAction, "body"},
		{"delete query", "erase", "query"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			if tc.form == "query" {
				ex.engines["ollama"].manifest.Actions[tc.action].HTTP.ParamsIn = actionHTTPParamsQuery
			}
			model, opaque := strings.Repeat("v", 512), "opaque\ntext\u0085"
			params := modelControlsJSON(t, map[string]string{"target": model, "TARGET": opaque})
			var calls, mutations int
			ex.modelMutationParent = func(context.Context, string) error { mutations++; return nil }
			ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				if tc.form == "query" {
					if !reflect.DeepEqual(r.URL.Query(), url.Values{"target": {model}, "TARGET": {opaque}}) {
						return nil, errors.New("exact or opaque query key changed")
					}
				} else {
					body, err := io.ReadAll(r.Body)
					if err != nil {
						return nil, err
					}
					if !bytes.Equal(body, params) {
						return nil, errors.New("exact or opaque body key changed")
					}
				}
				if tc.action == pullModelAction {
					return mutationJSON(http.StatusOK, `{"accepted":true,"status":"success"}`)
				}
				return mutationJSON(http.StatusOK, `{"accepted":true}`)
			})
			result, err := ex.Action(context.Background(), "ollama", tc.action, params)
			if err != nil {
				t.Fatalf("valid exact model with opaque variant: %v", err)
			}
			var response struct {
				Accepted bool
				Status   string
			}
			if err := json.Unmarshal(result, &response); err != nil {
				t.Fatalf("decode action result: %v", err)
			}
			if tc.action == pullModelAction && response.Status != "success" {
				t.Fatalf("pull omitted native success terminal: %+v", response)
			}
			wantMutations := 0
			if tc.action == "erase" || tc.action == pullModelAction {
				wantMutations = 1
			}
			if !response.Accepted || calls != 1 || mutations != wantMutations || ex.progress.next != 0 || len(ex.Errors()) != 0 {
				t.Fatalf("opaque variant altered valid action: result=%+v calls=%d mutations=%d", response, calls, mutations)
			}
		})
	}
}

func TestModelControlsPullPreservesOpaqueCustomCaseVariant(t *testing.T) {
	for _, boundary := range []string{"manager", "direct pull", "destination HTTP", "uppercase custom destination HTTP"} {
		t.Run(boundary, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			params := modelControlsJSON(t, map[string]string{"target": strings.Repeat("v", 512), "TARGET": "opaque\ntext\u0085"})
			if boundary == "uppercase custom destination HTTP" {
				ex.engines["ollama"].manifest.Operations[engines.OperationPull] = OperationBinding{Action: pullModelAction, ModelParam: "TARGET"}
				params = modelControlsJSON(t, map[string]string{"TARGET": strings.Repeat("v", 512), "target": "opaque\ntext\u0085"})
			}
			var calls, mutations, progress int
			ex.modelMutationParent = func(context.Context, string) error { mutations++; return nil }
			ex.emit = func(method string, _ any) {
				if method == "engine:pull-progress" {
					progress++
				}
			}
			ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				body, err := io.ReadAll(r.Body)
				if err != nil {
					return nil, err
				}
				if !bytes.Equal(body, params) {
					return nil, errors.New("pull changed exact or opaque custom field")
				}
				return mutationJSON(http.StatusOK, `{"status":"success"}`)
			})
			var result json.RawMessage
			wantAdmissions := 0
			switch boundary {
			case "manager":
				var output bytes.Buffer
				manager := &Manager{exec: ex, codec: NewCodec(&output)}
				id := json.RawMessage("1")
				manager.runAction(context.Background(), &Message{ID: &id, Params: modelControlsJSON(t, actionParam{Engine: "ollama", Action: pullModelAction, Params: params})})
				var response Message
				if err := json.NewDecoder(&output).Decode(&response); err != nil {
					t.Fatalf("decode manager pull result: %v", err)
				}
				if response.Error != nil {
					t.Fatalf("manager scanned opaque custom variant: %+v", response.Error)
				}
				result = response.Result
			case "direct pull":
				var err error
				result, err = ex.PullModelStream(context.Background(), "ollama", "", params)
				if err != nil {
					t.Fatalf("direct pull scanned opaque custom variant: %v", err)
				}
			case "destination HTTP", "uppercase custom destination HTTP":
				wantAdmissions = 1
				server := &controlServer{exec: ex}
				recorder := httptest.NewRecorder()
				wire := modelControlsJSON(t, pullRequest{Engine: "ollama", Params: params})
				server.handlePull(recorder, httptest.NewRequest(http.MethodPost, controlPullPath, bytes.NewReader(wire)))
				decoder := json.NewDecoder(recorder.Body)
				var terminal streamFrame
				for {
					var frame streamFrame
					if err := decoder.Decode(&frame); err == io.EOF {
						break
					} else if err != nil {
						t.Fatalf("decode destination stream: %v", err)
					}
					terminal = frame
				}
				if recorder.Code != http.StatusOK || !recorder.Flushed || terminal.Type != "result" {
					t.Fatalf("destination scanned opaque custom variant: HTTP %d terminal=%+v", recorder.Code, terminal)
				}
				result = terminal.Result
			}
			var response struct{ Status string }
			if err := json.Unmarshal(result, &response); err != nil {
				t.Fatalf("decode pull success: %v", err)
			}
			if response.Status != "success" || calls != 1 || mutations != 1 || progress == 0 || ex.progress.next != wantAdmissions || len(ex.progress.subs) != 0 || len(ex.Errors()) != 0 {
				t.Fatalf("opaque variant altered valid pull: result=%+v calls=%d mutations=%d progress=%d admissions=%d", response, calls, mutations, progress, ex.progress.next)
			}
		})
	}
}

func TestModelControlsPullCommonAliasesCannotReplaceExactDefaultKey(t *testing.T) {
	manifest := &Manifest{
		Actions:    map[string]Action{pullModelAction: {}},
		Operations: map[string]OperationBinding{engines.OperationPull: {Action: pullModelAction}},
	}
	for _, tc := range []struct{ name, params string }{
		{"uppercase model", `{"MODEL":"valid"}`},
		{"mixed-case model with progress name", `{"Model":"valid","name":"valid"}`},
		{"name alias only", `{"NAME":"valid"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model, err := manifest.pullProgressModel(json.RawMessage(tc.params))
			want := nodepolicy.ValidateModelIdentifierJSON(nil)
			if err == nil || err.Error() != want.Error() || model != "" {
				t.Fatalf("common alias replaced exact default consumer key: model=%q error=%v", model, err)
			}
		})
	}
	model := strings.Repeat("v", 512)
	params := modelControlsJSON(t, map[string]string{"model": model, "MODEL": model, "NAME": model})
	if got, err := manifest.pullProgressModel(params); err != nil || got != model {
		t.Fatalf("exact default key with valid common aliases: model=%q error=%v", got, err)
	}
	for _, key := range []string{"MODEL", "NAME"} {
		t.Run("invalid common "+key, func(t *testing.T) {
			params := modelControlsJSON(t, map[string]string{"model": model, key: "private\nmodel"})
			got, err := manifest.pullProgressModel(params)
			want := nodepolicy.ValidateModelIdentifier("private\nmodel")
			if err == nil || err.Error() != want.Error() || got != "" {
				t.Fatalf("invalid common alias escaped validation: model=%q error=%v", got, err)
			}
		})
	}
}

func TestModelControlsConventionalHTTPActionsPreserveCallerParams(t *testing.T) {
	for _, tc := range []struct{ name, action, params string }{
		{"model-free load", "load_model", ""},
		{"model-free load with informational schema", "load_model", "null"},
		{"name-only delete body", "delete_model", `{"name":"tiny","prompt":"opaque\ntext"}`},
		{"name-only delete query", "delete_model", `{"name":"tiny"}`},
		{"name-only pull action", pullModelAction, `{"name":"tiny"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			manifest := ex.engines["ollama"].manifest
			manifest.Operations = nil
			manifest.Actions["load_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/load"}}
			manifest.Actions["delete_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodDelete, Path: "/delete"}}
			query := tc.name == "name-only delete query"
			if query {
				manifest.Actions[tc.action].HTTP.ParamsIn = actionHTTPParamsQuery
			}
			if tc.name == "model-free load with informational schema" {
				manifest.Actions[tc.action].HTTP.BodySchema = json.RawMessage(`{"model":"string"}`)
			}
			// Typed synthesis still uses the conventional model key. It does
			// not make that key mandatory in the generic action's raw params.
			op := engines.OperationLoad
			if tc.action == "delete_model" {
				op = engines.OperationDelete
			} else if tc.action == pullModelAction {
				op = engines.OperationPull
			}
			action, raw, err := ex.modelOperationCall("ollama", op, "typed-model")
			if err != nil {
				t.Fatalf("resolve conventional typed call: %v", err)
			}
			var typedParams map[string]string
			if err := json.Unmarshal(raw, &typedParams); err != nil {
				t.Fatalf("decode typed params: %v", err)
			}
			if action != tc.action || !reflect.DeepEqual(typedParams, map[string]string{"model": "typed-model"}) {
				t.Fatalf("conventional typed binding changed: action=%s params=%v", action, typedParams)
			}
			var calls, mutations int
			ex.modelMutationParent = func(context.Context, string) error { mutations++; return nil }
			ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				var body []byte
				if r.Body != nil {
					var err error
					body, err = io.ReadAll(r.Body)
					if err != nil {
						return nil, err
					}
				}
				if query {
					if len(body) != 0 || !reflect.DeepEqual(r.URL.Query(), url.Values{"name": {"tiny"}}) {
						return nil, errors.New("name-only query changed")
					}
				} else {
					want := tc.params
					if want == "null" {
						want = ""
					}
					if string(body) != want {
						return nil, errors.New("generic action body changed")
					}
				}
				if tc.action == pullModelAction {
					return mutationJSON(http.StatusOK, `{"accepted":true,"status":"success"}`)
				}
				return mutationJSON(http.StatusOK, `{"accepted":true}`)
			})
			result, err := ex.Action(context.Background(), "ollama", tc.action, json.RawMessage(tc.params))
			if err != nil {
				t.Fatalf("conventional action: %v", err)
			}
			var response struct {
				Accepted bool
				Status   string
			}
			if err := json.Unmarshal(result, &response); err != nil {
				t.Fatalf("decode generic action result: %v", err)
			}
			if tc.action == pullModelAction && response.Status != "success" {
				t.Fatalf("pull omitted native success terminal: %+v", response)
			}
			wantMutations := 1
			if tc.action == "load_model" {
				wantMutations = 0
			}
			if !response.Accepted || calls != 1 || mutations != wantMutations || len(ex.Errors()) != 0 {
				t.Fatalf("generic action result=%+v calls=%d mutations=%d", response, calls, mutations)
			}
		})
	}
}

func TestModelControlsConventionalNamePullStreamsAtEntryBoundaries(t *testing.T) {
	for _, boundary := range []string{"manager", "direct pull", "destination HTTP"} {
		t.Run(boundary, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			manifest := ex.engines["ollama"].manifest
			manifest.Operations = nil
			model := strings.Repeat("v", 512)
			params := modelControlsJSON(t, map[string]string{"name": model, "TARGET": "opaque\ntext"})
			if got, err := manifest.pullProgressModel(params); err != nil || got != model {
				t.Fatalf("name-only progress model=%q error=%v", got, err)
			}
			both := modelControlsJSON(t, map[string]string{"name": model, "model": "secondary"})
			if got, err := manifest.pullProgressModel(both); err != nil || got != model {
				t.Fatalf("name-before-model progress changed: model=%q error=%v", got, err)
			}
			var calls, mutations int
			ex.modelMutationParent = func(context.Context, string) error { mutations++; return nil }
			var progress []ProgressEvent
			ex.emit = func(method string, value any) {
				if method == "engine:pull-progress" {
					fields, ok := value.(map[string]any)
					if !ok {
						t.Errorf("pull progress type=%T", value)
						return
					}
					raw, err := json.Marshal(fields)
					if err != nil {
						t.Errorf("encode progress: %v", err)
						return
					}
					var event ProgressEvent
					if err := json.Unmarshal(raw, &event); err != nil {
						t.Errorf("decode progress: %v", err)
						return
					}
					progress = append(progress, event)
				}
			}
			ex.client.Transport = memoryModelsTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				body, err := io.ReadAll(r.Body)
				if err != nil {
					return nil, err
				}
				if !bytes.Equal(body, params) {
					return nil, errors.New("name-only pull body changed")
				}
				return mutationJSON(http.StatusOK, "{\"status\":\"pulling\",\"total\":100,\"completed\":40}\n{\"status\":\"success\"}\n")
			})
			var result json.RawMessage
			wantAdmissions := 0
			switch boundary {
			case "manager":
				var output bytes.Buffer
				manager := &Manager{exec: ex, codec: NewCodec(&output)}
				id := json.RawMessage("1")
				manager.runAction(context.Background(), &Message{ID: &id, Params: modelControlsJSON(t, actionParam{Engine: "ollama", Action: pullModelAction, Params: params})})
				var response Message
				if err := json.NewDecoder(&output).Decode(&response); err != nil {
					t.Fatalf("decode name-only manager result: %v", err)
				}
				if response.Error != nil {
					t.Fatalf("manager refused name-only pull: %+v", response.Error)
				}
				result = response.Result
			case "direct pull":
				var err error
				result, err = ex.PullModelStream(context.Background(), "ollama", "", params)
				if err != nil {
					t.Fatalf("direct name-only pull: %v", err)
				}
			case "destination HTTP":
				wantAdmissions = 1
				server := &controlServer{exec: ex}
				recorder := httptest.NewRecorder()
				wire := modelControlsJSON(t, pullRequest{OpID: "name-only", Engine: "ollama", Params: params})
				server.handlePull(recorder, httptest.NewRequest(http.MethodPost, controlPullPath, bytes.NewReader(wire)))
				decoder := json.NewDecoder(recorder.Body)
				var terminal streamFrame
				var streamed []ProgressEvent
				for {
					var frame streamFrame
					if err := decoder.Decode(&frame); err == io.EOF {
						break
					} else if err != nil {
						t.Fatalf("decode name-only stream: %v", err)
					}
					if frame.OpID != "name-only" || frame.ID == "" || frame.Engine != "ollama" || frame.Op != "pull" || frame.Model != model || frame.DownloadState == nil {
						t.Fatalf("name-only stream lost target identity: %+v", frame)
					}
					if frame.Type == "progress" {
						streamed = append(streamed, ProgressEvent{Engine: frame.Engine, Op: frame.Op, Stage: frame.Stage, Percent: frame.Percent, Message: frame.Message, ID: frame.ID, Model: frame.Model, DownloadState: frame.DownloadState})
					}
					terminal = frame
				}
				if recorder.Code != http.StatusOK || !recorder.Flushed || terminal.Type != "result" || terminal.OpID != "name-only" || !reflect.DeepEqual(streamed, progress) {
					t.Fatalf("name-only stream: HTTP %d terminal=%+v streamed=%+v", recorder.Code, terminal, streamed)
				}
				// The stream's explicit queued frame replaces the queued local
				// event emitted before its exact-ID subscription was installed.
				if len(progress) == 0 || progress[len(progress)-1].DownloadState == nil || terminal.ID != progress[0].ID || !terminal.DownloadState.Equal(*progress[len(progress)-1].DownloadState) {
					t.Fatalf("stream terminal does not match local lifecycle: %+v", terminal)
				}
				result = terminal.Result
			}
			var response struct{ Status string }
			if err := json.Unmarshal(result, &response); err != nil {
				t.Fatalf("decode name-only pull result: %v", err)
			}
			wantProgress := []struct {
				stage, phase string
				percent      int
				blocks       bool
			}{
				{"queued", modeldownload.PhaseQueued, 0, true},
				{"pulling", modeldownload.PhaseInProgress, 0, true},
				{"pulling", modeldownload.PhaseInProgress, 40, true},
				{"success", modeldownload.PhaseComplete, 100, false},
			}
			if len(progress) != len(wantProgress) {
				t.Fatalf("pull progress=%+v, want queued, pulling, measured progress and terminal", progress)
			}
			var previousRevision uint64
			for i, want := range wantProgress {
				event := progress[i]
				entry := modelControlsPullEntry(t, event, ex.switchIncarnation, progress[0].ID, model)
				if event.Stage != want.stage || event.Percent != want.percent {
					t.Fatalf("pull progress[%d]=%+v, want stage=%s percent=%d", i, event, want.stage, want.percent)
				}
				wantEntry := modeldownload.Entry{ID: event.ID, Engine: "ollama", Model: model, Phase: want.phase, BlocksReadiness: want.blocks}
				if want.percent == 40 {
					percent := 40
					wantEntry.Percent = &percent
				}
				if !reflect.DeepEqual(entry, wantEntry) {
					t.Fatalf("pull entry[%d]=%+v, want %+v", i, entry, wantEntry)
				}
				revision, err := strconv.ParseUint(event.DownloadState.Revision, 10, 64)
				if err != nil || revision <= previousRevision {
					t.Fatalf("pull revision=%q after %d: %v", event.DownloadState.Revision, previousRevision, err)
				}
				previousRevision = revision
			}
			if response.Status != "success" || calls != 1 || mutations != 1 ||
				ex.progress.next != wantAdmissions || len(ex.progress.subs) != 0 || len(ex.Errors()) != 0 {
				t.Fatalf("name-only pull result=%+v progress=%+v calls=%d mutations=%d", response, progress, calls, mutations)
			}
		})
	}
}

func TestModelControlsConventionalNamePullStoppedEngineReportsTerminalError(t *testing.T) {
	ex := modelControlsExecutor(t)
	ex.engines["ollama"].manifest.Operations = nil
	ex.engines["ollama"].running = false
	acceptModelMutations(ex)
	var progress []ProgressEvent
	ex.emit = func(method string, value any) {
		if method == "engine:pull-progress" {
			fields, ok := value.(map[string]any)
			if !ok {
				t.Errorf("pull progress type=%T", value)
				return
			}
			raw, err := json.Marshal(fields)
			if err != nil {
				t.Errorf("encode stopped pull progress: %v", err)
				return
			}
			var event ProgressEvent
			if err := json.Unmarshal(raw, &event); err != nil {
				t.Errorf("decode stopped pull progress: %v", err)
				return
			}
			progress = append(progress, event)
		}
	}
	var output bytes.Buffer
	manager := &Manager{exec: ex, codec: NewCodec(&output)}
	id := json.RawMessage("1")
	manager.runAction(context.Background(), &Message{ID: &id, Params: json.RawMessage(`{"engine":"ollama","action":"pull_model","params":{"name":"tiny"}}`)})
	var response Message
	if err := json.NewDecoder(&output).Decode(&response); err != nil {
		t.Fatalf("decode stopped-engine response: %v", err)
	}
	want := formatEnginePullError("Ollama", errors.New(`engine "ollama" is not running`))
	if len(progress) != 2 {
		t.Fatalf("stopped pull progress=%+v, want queued and failed", progress)
	}
	downloadID := progress[0].ID
	queued := modelControlsPullEntry(t, progress[0], ex.switchIncarnation, downloadID, "tiny")
	failed := modelControlsPullEntry(t, progress[1], ex.switchIncarnation, downloadID, "tiny")
	if progress[0].Stage != "queued" || progress[0].Percent != 0 ||
		!reflect.DeepEqual(queued, modeldownload.Entry{ID: downloadID, Engine: "ollama", Model: "tiny", Phase: modeldownload.PhaseQueued, BlocksReadiness: true}) {
		t.Fatalf("stopped pull did not queue exact model: %+v", progress[0])
	}
	if progress[1].Stage != "error" || progress[1].Percent != -1 ||
		!reflect.DeepEqual(failed, modeldownload.Entry{ID: downloadID, Engine: "ollama", Model: "tiny", Phase: modeldownload.PhaseFailed, Reason: modeldownload.ReasonDownloadFailed}) ||
		progress[0].DownloadState.Equal(*progress[1].DownloadState) || !progress[1].DownloadState.Equal(ex.downloads.snapshot()) {
		t.Fatalf("stopped pull did not publish fixed terminal state: %+v", progress[1])
	}
	reports := ex.Errors()
	if response.Error == nil || response.Error.Code != -32000 || response.Error.Message != want ||
		len(reports) != 1 || reports[0].Message != want || reports[0].ModelName != "tiny" || reports[0].Operation != "pull" || reports[0].Action != "retry" {
		t.Fatalf("stopped name-only pull response=%+v progress=%v reports=%+v", response, progress, reports)
	}
}

func TestModelControlsConventionalActionsRejectSuppliedCommonIdentifiers(t *testing.T) {
	for _, tc := range []struct{ name, raw string }{
		{"null", `null`},
		{"empty", `""`},
		{"wrong type", `123`},
		{"C0", `"private\nmodel"`},
		{"DEL", `"private\u007fmodel"`},
		{"C1", `"private\u0085model"`},
		{"original invalid UTF8", "\"private\xffmodel\""},
		{"513 bytes", `"` + strings.Repeat("s", 513) + `"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, boundary := range []string{"load action", "delete action", "pull action", "manager pull", "direct pull", "destination pull"} {
				t.Run(boundary, func(t *testing.T) {
					for _, key := range []string{"name", "MODEL"} {
						t.Run(key, func(t *testing.T) {
							ex := modelControlsExecutor(t)
							manifest := ex.engines["ollama"].manifest
							manifest.Operations = nil
							manifest.Actions["load_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/load"}}
							manifest.Actions["delete_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodDelete, Path: "/delete"}}
							ex.emit = func(string, any) { t.Error("invalid common model emitted a notification") }
							before := ex.inventoryDigests["ollama"]
							params := json.RawMessage(`{"model":"valid","` + key + `":` + tc.raw + `}`)
							want := nodepolicy.ValidateModelIdentifierJSON(json.RawMessage(tc.raw))
							switch boundary {
							case "load action", "delete action", "pull action", "direct pull":
								var result json.RawMessage
								var err error
								if boundary == "direct pull" {
									result, err = ex.PullModelStream(context.Background(), "ollama", "valid", params)
								} else {
									action := pullModelAction
									if boundary == "load action" {
										action = "load_model"
									} else if boundary == "delete action" {
										action = "delete_model"
									}
									result, err = ex.Action(context.Background(), "ollama", action, params)
								}
								if err == nil || err.Error() != want.Error() || result != nil {
									t.Fatalf("supplied common model refusal: result=%s error=%v want=%v", result, err, want)
								}
							case "manager pull":
								var output bytes.Buffer
								manager := &Manager{exec: ex, codec: NewCodec(&output)}
								id := json.RawMessage("1")
								manager.runAction(context.Background(), &Message{ID: &id, Params: modelControlsJSON(t, actionParam{Engine: "ollama", Action: pullModelAction, Params: params})})
								var response Message
								if err := json.NewDecoder(&output).Decode(&response); err != nil {
									t.Fatalf("decode common-field refusal: %v", err)
								}
								if response.Error == nil || response.Error.Code != -32602 || response.Error.Message != want.Error() || len(response.Result) != 0 {
									t.Fatalf("manager common-field refusal=%+v", response)
								}
							case "destination pull":
								server := &controlServer{exec: ex}
								recorder := httptest.NewRecorder()
								wire := modelControlsJSON(t, pullRequest{Engine: "ollama", Params: params})
								server.handlePull(recorder, httptest.NewRequest(http.MethodPost, controlPullPath, bytes.NewReader(wire)))
								if recorder.Code != http.StatusBadRequest || recorder.Flushed || recorder.Body.String() != "invalid JSON body\n" {
									t.Fatalf("destination common-field refusal: HTTP %d body=%q", recorder.Code, recorder.Body.String())
								}
							}
							if len(ex.Errors()) != 0 || ex.progress.next != 0 || len(ex.progress.subs) != 0 || !reflect.DeepEqual(before, ex.inventoryDigests["ollama"]) {
								t.Fatal("invalid supplied common field crossed progress/report/mutation boundary")
							}
						})
					}
				})
			}
		})
	}
}

func TestModelControlsConventionalConsumersRequireExactModelBeforeEffects(t *testing.T) {
	for _, tc := range []struct {
		name, action string
		definition   Action
	}{
		{"command model placeholder", "delete_model", Action{Cmd: []string{"unreachable-compatibility-child", "{model}"}}},
		{"unbound command model placeholder", "opaque", Action{Cmd: []string{"unreachable-compatibility-child", "{model}"}}},
		{"command name placeholder", "delete_model", Action{Cmd: []string{"unreachable-compatibility-child", "{name}"}}},
		{"remove path model placeholder", "delete_model", Action{RemovePath: &ActionRemovePath{Root: "unused", Path: "unused/{model}"}}},
		{"unbound remove path model placeholder", "opaque", Action{RemovePath: &ActionRemovePath{Root: "unused", Path: "unused/{model}"}}},
		{"model resolver", pullModelAction, Action{Cmd: []string{"unreachable-compatibility-child"}, ModelResolution: modelResolutionLMSGet}},
		{"progress protocol", pullModelAction, Action{HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/models"}, ProgressProtocol: pullProgressProtocolLlamaCPPModelsSSE}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex := modelControlsExecutor(t)
			manifest := ex.engines["ollama"].manifest
			manifest.Operations = nil
			manifest.Actions[tc.action] = tc.definition
			ex.emit = func(string, any) { t.Error("missing consumer key emitted a notification") }
			before := ex.inventoryDigests["ollama"]
			result, err := ex.Action(context.Background(), "ollama", tc.action, json.RawMessage(`{"MODEL":"valid","NAME":"valid"}`))
			want := nodepolicy.ValidateModelIdentifierJSON(nil)
			if err == nil || err.Error() != want.Error() || result != nil || len(ex.Errors()) != 0 || ex.progress.next != 0 ||
				!reflect.DeepEqual(before, ex.inventoryDigests["ollama"]) {
				t.Fatalf("missing consumer key reached effects: result=%s error=%v", result, err)
			}
		})
	}
}

func TestModelControlsConventionalLoadRetainsServingDeclarationGuard(t *testing.T) {
	ex := modelControlsExecutor(t)
	st := ex.engines["ollama"]
	st.manifest.Operations = nil
	st.manifest.Actions["load_model"] = Action{HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/load"}}
	st.plat.Runtime.Serving = modelControlsServing("valid")
	ex.emit = func(string, any) { t.Error("missing serving model emitted a notification") }
	for _, params := range []string{"", `{"MODEL":"valid","name":"valid"}`} {
		result, err := ex.Action(context.Background(), "ollama", "load_model", json.RawMessage(params))
		var fieldError *ServingFieldError
		if !errors.As(err, &fieldError) || fieldError.Field != "model" || fieldError.ModelID != "" || result != nil || len(ex.Errors()) != 0 {
			t.Fatalf("conventional serving guard: result=%s error=%v", result, err)
		}
	}
}

func TestModelControlsTypedMissingFieldsKeepStructuralErrors(t *testing.T) {
	for _, tc := range []struct{ name, params, want string }{
		{"root null", `null`, "engine and model are required"},
		{"empty object", `{}`, "engine and model are required"},
		{"absent model", `{"engine":"ollama"}`, "engine and model are required"},
		{"absent engine", `{"model":"valid"}`, "engine and model are required"},
		{"supplied null model", `{"model":null}`, "invalid params: model is required"},
		{"supplied empty model", `{"model":""}`, "invalid params: model is required"},
		{"supplied wrong type", `{"model":{"private":"value"}}`, "invalid params: model must be a string"},
		{"supplied invalid UTF8", "{\"model\":\"private\xffmodel\"}", "invalid params: model must be valid UTF-8"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var output bytes.Buffer
			manager := &Manager{codec: NewCodec(&output)}
			id := json.RawMessage("1")
			manager.runModelOperation(context.Background(), &Message{ID: &id, Method: loadModelMethod, Params: json.RawMessage(tc.params)})
			var response Message
			if err := json.NewDecoder(&output).Decode(&response); err != nil {
				t.Fatalf("decode structural refusal: %v", err)
			}
			if response.Error == nil || response.Error.Code != -32602 || response.Error.Message != tc.want || len(response.Result) != 0 {
				t.Fatalf("typed structural refusal=%+v want=%q", response, tc.want)
			}
		})
	}
}

func TestModelControlsCopyMissingFieldsKeepOrderBeforePeerLookup(t *testing.T) {
	for _, tc := range []struct{ name, params, want string }{
		{"root null", `null`, "node is required"},
		{"empty object", `{}`, "node is required"},
		{"node before model", `{"engine":"ollama"}`, "node is required"},
		{"engine before model", `{"node":"private-peer"}`, "engine is required"},
		{"model before lookup", `{"node":"private-peer","engine":"ollama"}`, "model is required"},
		{"supplied null model", `{"model":null}`, "model is required"},
		{"supplied empty model", `{"model":""}`, "model is required"},
		{"supplied invalid UTF8", "{\"model\":\"private\xffmodel\"}", "model must be valid UTF-8"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var output bytes.Buffer
			// Nil peer directory/executor: every case must stop before lookup,
			// client construction, progress, copy source or store access.
			manager := &Manager{codec: NewCodec(&output)}
			id := json.RawMessage("1")
			manager.runRemoteCopy(context.Background(), &Message{ID: &id, Method: "engine:remote-copy-model", Params: json.RawMessage(tc.params)})
			var response Message
			if err := json.NewDecoder(&output).Decode(&response); err != nil {
				t.Fatalf("decode copy structural refusal: %v", err)
			}
			if response.Error == nil || response.Error.Code != -32602 || response.Error.Message != tc.want || len(response.Result) != 0 {
				t.Fatalf("copy structural refusal=%+v want=%q", response, tc.want)
			}
		})
	}
}
