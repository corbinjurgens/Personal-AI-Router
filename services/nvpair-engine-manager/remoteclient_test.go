// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRemoteClientModelFilesUseOrdinaryHeaderBudget(t *testing.T) {
	test := func(name string, call func(context.Context, *remoteClient) error) {
		t.Run(name, func(t *testing.T) {
			release := make(chan struct{})
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				select {
				case <-release:
				case <-r.Context().Done():
				}
			}))
			defer func() {
				close(release)
				srv.Close()
			}()
			base := http.DefaultTransport.(*http.Transport)
			client := &remoteClient{
				controlHTTP:      newRemoteHTTPClient(base, 20*time.Millisecond),
				controlReadyHTTP: newRemoteHTTPClient(base, time.Second),
				base:             srv.URL,
			}
			ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
			defer cancel()
			err := call(ctx, client)
			if err == nil || !strings.Contains(err.Error(), "timeout awaiting response headers") {
				t.Fatalf("model request error = %v, want ordinary header timeout", err)
			}
		})
	}
	test("listing", func(ctx context.Context, client *remoteClient) error {
		_, err := client.modelFiles(ctx, "ollama", "tiny", nil)
		return err
	})
	test("download", func(ctx context.Context, client *remoteClient) error {
		body, _, err := client.modelFile(ctx, "ollama", "tiny", "blobs/fixture", 0)
		if body != nil {
			err = errors.Join(err, body.Close())
		}
		return err
	})
}

func TestRemoteStartUsesReadinessHeaderBudget(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"engine":"remote-only","running":true,"healthy":true}`))
	}))
	defer srv.Close()

	base := http.DefaultTransport.(*http.Transport)
	c := &remoteClient{
		controlHTTP:      newRemoteHTTPClient(base, 20*time.Millisecond),
		controlReadyHTTP: newRemoteHTTPClient(base, time.Second),
		base:             srv.URL,
	}
	if _, err := c.postJSON(context.Background(), controlStartPath, "remote-only", startRequest{Engine: "remote-only"}); err != nil {
		t.Fatalf("remote start was cut off by the ordinary response-header budget: %v", err)
	}
	if _, err := c.postJSON(context.Background(), controlStopPath, "ollama", stopRequest{Engine: "ollama"}); err == nil ||
		!strings.Contains(err.Error(), "timeout awaiting response headers") {
		t.Fatalf("ordinary remote call error = %v, want the ordinary bounded response-header timeout", err)
	}
}

// A remote Ollama load can withhold headers while a cold model loads. A remote
// LM Studio delete can likewise restart the peer's engine before it replies.
// Both need the longer response-header budget; other model calls remain bounded.
// The load budget comes from the bundled manifests' load bindings.
func TestRemoteSlowModelOperationsUseReadinessHeaderBudget(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	base := http.DefaultTransport.(*http.Transport)
	c := &remoteClient{
		controlHTTP:      newRemoteHTTPClient(base, 20*time.Millisecond),
		controlReadyHTTP: newRemoteHTTPClient(base, time.Second),
		base:             srv.URL,
		loadBudget:       bundledTestExecutor(t).loadResponseBudget,
	}
	ollama := modelActionRequest{Engine: "ollama", Model: "qwen2.5"}
	if _, err := c.postJSON(context.Background(), controlLoadPath, "ollama", ollama); err != nil {
		t.Fatalf("remote load was cut off by the ordinary response-header budget: %v", err)
	}
	lmstudio := modelActionRequest{Engine: "lmstudio", Model: "qwen2.5"}
	if _, err := c.postJSON(context.Background(), controlDeletePath, "lmstudio", lmstudio); err != nil {
		t.Fatalf("remote delete was cut off by the ordinary response-header budget: %v", err)
	}
	if _, err := c.postJSON(context.Background(), controlLoadPath, "lmstudio", lmstudio); err == nil ||
		!strings.Contains(err.Error(), "timeout awaiting response headers") {
		t.Fatalf("remote LM Studio load error = %v, want the ordinary bounded response-header timeout", err)
	}
	if _, err := c.postJSON(context.Background(), controlUnloadPath, "ollama", ollama); err == nil ||
		!strings.Contains(err.Error(), "timeout awaiting response headers") {
		t.Fatalf("remote unload error = %v, want the ordinary bounded response-header timeout", err)
	}
}

func TestRemoteReadinessBudgetCoversEngineStartupAllowance(t *testing.T) {
	if remoteReadyResponseHeaderTimeout <= remoteResponseHeaderTimeout {
		t.Fatalf("readiness budget %s must exceed the ordinary budget %s",
			remoteReadyResponseHeaderTimeout, remoteResponseHeaderTimeout)
	}
	ex := bundledTestExecutor(t)
	cases := []struct {
		path   string
		engine string
		want   bool
	}{
		{controlStartPath, "ollama", true},
		{controlDeletePath, "lmstudio", true},
		{controlLoadPath, "ollama", true},
		{controlLoadPath, "lmstudio", false},
		{controlLoadPath, "llamacpp", false},
		{controlLoadPath, "peer-only", false},
		{controlStopPath, "ollama", false},
		{controlUnloadPath, "ollama", false},
		{controlEnginesPath, "", false},
	}
	for _, tc := range cases {
		if got := waitsForEngineReadiness(tc.path, ex.loadResponseBudget(tc.engine)); got != tc.want {
			t.Errorf("waitsForEngineReadiness(%q, budget of %q) = %v, want %v", tc.path, tc.engine, got, tc.want)
		}
	}
}

func TestRemoteStartHeaderWaitRemainsBounded(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-release
		_, _ = w.Write([]byte(`{"engine":"ollama"}`))
	}))
	defer func() {
		close(release)
		srv.Close()
	}()

	base := http.DefaultTransport.(*http.Transport)
	c := &remoteClient{
		controlHTTP:      newRemoteHTTPClient(base, time.Second),
		controlReadyHTTP: newRemoteHTTPClient(base, 50*time.Millisecond),
		base:             srv.URL,
	}
	started := time.Now()
	_, err := c.postJSON(context.Background(), controlStartPath, "ollama", startRequest{Engine: "ollama"})
	if err == nil || !strings.Contains(err.Error(), "timeout awaiting response headers") {
		t.Fatalf("remote start error = %v, want bounded response-header timeout", err)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("remote start took %s, want a prompt bounded failure", elapsed)
	}
}
