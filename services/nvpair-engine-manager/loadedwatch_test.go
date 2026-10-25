// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/modeldownload"
)

func modelDownloadWatch(t *testing.T, ex *Executor) <-chan modelsChangedParams {
	t.Helper()
	ex.loadedPollInterval = time.Hour
	updates := make(chan modelsChangedParams, 16)
	ex.emit = func(method string, params any) {
		if method != modelsChangedMethod {
			return
		}
		// Notifications must release the content gate before entering a relay
		// which may synchronously obtain another lifecycle snapshot.
		ex.downloads.snapshot()
		data, err := json.Marshal(params)
		if err != nil {
			t.Errorf("encode model notification: %v", err)
			return
		}
		var update modelsChangedParams
		if err := json.Unmarshal(data, &update); err != nil {
			t.Errorf("decode model notification: %v", err)
			return
		}
		updates <- update
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); ex.watchLoaded(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Error("model watcher did not stop")
		}
	})
	return updates
}

func nextModelDownloadUpdate(t *testing.T, updates <-chan modelsChangedParams) modelsChangedParams {
	t.Helper()
	select {
	case update := <-updates:
		if update.Engine != "ollama" || update.Models.DownloadState == nil {
			t.Fatalf("notification lost engine or target snapshot: %+v", update)
		}
		if err := update.Models.DownloadState.Validate(); err != nil {
			t.Fatalf("notification has invalid lifecycle: %v", err)
		}
		return update
	case <-time.After(5 * time.Second):
		t.Fatal("missing model notification")
		return modelsChangedParams{}
	}
}

func TestModelDownloadWatcherPublishesLifecycleWithoutLoadedChanges(t *testing.T) {
	ex := modelDownloadInventoryExecutor(t)
	ex.engines["ollama"].running = false
	updates := modelDownloadWatch(t, ex)
	ex.pokeLoaded()
	baseline := nextModelDownloadUpdate(t, updates)
	if len(baseline.Models.DownloadState.Entries) != 0 {
		t.Fatalf("initial lifecycle is not explicitly empty: %+v", baseline)
	}
	attempt := reserveModelDownload(t, ex.downloads, "ollama", "demo", engines.ModelNamingImpliedLatestTag)
	ex.pokeLoaded()
	queued := nextModelDownloadUpdate(t, updates)
	entry := queued.Models.DownloadState.Entries[0]
	if entry.ID != attempt.entry.ID || entry.Model != "demo" || entry.Phase != modeldownload.PhaseQueued || !entry.BlocksReadiness {
		t.Fatalf("queued publication lost target operation: %+v", entry)
	}
	if len(queued.Models.LoadedByEngine) != 0 || len(queued.Models.ByEngine) != 0 {
		t.Fatalf("stopped engine gained residency or ready inventory: %+v", queued)
	}
	if err := ex.downloads.finish(attempt, modeldownload.PhaseFailed, modeldownload.ReasonInterrupted, ""); err != nil {
		t.Fatalf("finish cancelled waiter: %v", err)
	}
	ex.pokeLoaded()
	terminal := nextModelDownloadUpdate(t, updates)
	entry = terminal.Models.DownloadState.Entries[0]
	if entry.ID != attempt.entry.ID || entry.Phase != modeldownload.PhaseFailed || entry.BlocksReadiness {
		t.Fatalf("lifecycle-only terminal was not published: %+v", entry)
	}
	fresh := ex.ModelsResult(context.Background())
	if fresh.DownloadState == nil || !fresh.DownloadState.Equal(*terminal.Models.DownloadState) {
		t.Fatalf("fresh client disagrees with pushed lifecycle: %+v", fresh.DownloadState)
	}
}

func TestModelDownloadWatcherPublishesInventoryWithoutLoadedChanges(t *testing.T) {
	ex := modelDownloadInventoryExecutor(t)
	var removed atomic.Bool
	original := ex.client.Transport
	ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path == "/api/tags" && removed.Load() {
			return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"models":[]}`))}, nil
		}
		return original.RoundTrip(request)
	})
	updates := modelDownloadWatch(t, ex)
	ex.pokeLoaded()
	baseline := nextModelDownloadUpdate(t, updates)
	if !reflect.DeepEqual(baseline.Models.Models, []string{"demo:latest", "ready:latest"}) {
		t.Fatalf("missing ready inventory baseline: %+v", baseline)
	}
	removed.Store(true)
	ex.pokeLoaded()
	changed := nextModelDownloadUpdate(t, updates)
	if len(changed.Models.Models) != 0 || !reflect.DeepEqual(changed.Models.ByEngine["ollama"], []string{}) ||
		!reflect.DeepEqual(changed.Models.LoadedByEngine, baseline.Models.LoadedByEngine) ||
		!changed.Models.DownloadState.Equal(*baseline.Models.DownloadState) {
		t.Fatalf("inventory-only change lost independent loaded/lifecycle facts: %+v", changed)
	}
}
