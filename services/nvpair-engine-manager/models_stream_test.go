// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/performancewire"
)

// modelsBuilder is a controllable ModelsResult source for the publisher.
type modelsBuilder struct {
	mu     sync.Mutex
	res    ModelsResult
	builds int
}

func (b *modelsBuilder) set(res ModelsResult) {
	b.mu.Lock()
	b.res = res
	b.mu.Unlock()
}

func (b *modelsBuilder) build(context.Context) ModelsResult {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.builds++
	return b.res
}

func (b *modelsBuilder) buildCount() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.builds
}

// freshnessEntry returns a memory-evidence entry with the per-build freshness
// fields set to age and generation.
func freshnessEntry(age, generation uint64) performancewire.MemoryEvidenceEntry {
	sampleAge := int64(age)
	qualificationAge := int64(age * 2)
	return performancewire.MemoryEvidenceEntry{
		Engine: "llamacpp",
		Model:  "demo",
		Freshness: performancewire.MemoryFreshness{
			SampleAgeMs:        &sampleAge,
			QualificationAgeMs: &qualificationAge,
			Generation:         &generation,
		},
	}
}

// modelsStreamFixture serves the em TLS mux for an executor whose publisher is
// fed by builder, and returns a pinned peer's client and the server URL.
type modelsStreamFixture struct {
	url      string
	pinned   *http.Client
	stranger *http.Client
	plain    *http.ServeMux
	pub      *modelsPublisher
}

func newModelsStreamFixture(t *testing.T, builder *modelsBuilder) *modelsStreamFixture {
	t.Helper()
	selfCert, selfKey := mintLeaf(t, "uuid-self")
	peerCert, peerKey := mintLeaf(t, "uuid-peer")
	strangerCert, strangerKey := mintLeaf(t, "uuid-stranger")
	selfMesh := clustertrust.Open(clusterDirFor(t, selfCert, selfKey, map[string][]byte{"uuid-peer": peerCert}))
	peerMesh := clustertrust.Open(clusterDirFor(t, peerCert, peerKey, map[string][]byte{"uuid-self": selfCert}))
	strangerMesh := clustertrust.Open(clusterDirFor(t, strangerCert, strangerKey, map[string][]byte{"uuid-self": selfCert}))

	exec := &Executor{modelsEvents: newModelsPublisher()}
	plain, tlsMux := emMuxes(exec, selfMesh)
	srv := httptest.NewUnstartedServer(tlsMux)
	srv.TLS = selfMesh.ServerTLSConfig()
	srv.StartTLS()
	t.Cleanup(srv.Close)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		exec.modelsEvents.run(ctx, builder.build)
	}()
	t.Cleanup(func() {
		cancel()
		<-done
	})

	client := func(m *clustertrust.Mesh) *http.Client {
		cfg, ok := m.ClientTLSConfig("uuid-self")
		if !ok {
			t.Fatal("a mesh pinning uuid-self must build a client for it")
		}
		return &http.Client{Transport: &http.Transport{TLSClientConfig: cfg}}
	}
	return &modelsStreamFixture{url: srv.URL, pinned: client(peerMesh), stranger: client(strangerMesh), plain: plain, pub: exec.modelsEvents}
}

// openStream opens the models stream and returns a channel of its lines.
func (f *modelsStreamFixture) openStream(t *testing.T) <-chan string {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, f.url+modelsEventsPath, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := f.pinned.Do(req)
	if err != nil {
		t.Fatalf("open models stream: %v", err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("models stream status = %d, want 200", resp.StatusCode)
	}
	if got := resp.Header.Get("Cache-Control"); got != "no-store" {
		t.Fatalf("Cache-Control = %q, want no-store", got)
	}
	lines := make(chan string, 16)
	go func() {
		defer close(lines)
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			lines <- scanner.Text()
		}
	}()
	return lines
}

// nextFrame returns the next non-keepalive line within d.
func nextFrame(t *testing.T, lines <-chan string, d time.Duration) (string, bool) {
	t.Helper()
	deadline := time.After(d)
	for {
		select {
		case line, open := <-lines:
			if !open {
				t.Fatal("models stream ended")
			}
			if line != "null" {
				return line, true
			}
		case <-deadline:
			return "", false
		}
	}
}

func decodeModels(t *testing.T, line string) ModelsResult {
	t.Helper()
	var res ModelsResult
	if err := json.Unmarshal([]byte(line), &res); err != nil {
		t.Fatalf("decode frame %q: %v", line, err)
	}
	return res
}

// waitBuilds waits until the publisher has built at least n times.
func waitBuilds(t *testing.T, builder *modelsBuilder, n int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for builder.buildCount() < n {
		if time.Now().After(deadline) {
			t.Fatalf("publisher built %d times, want %d", builder.buildCount(), n)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestModelsStream(t *testing.T) {
	t.Run("pinned peer gets the current document then changes only", func(t *testing.T) {
		builder := &modelsBuilder{}
		builder.set(ModelsResult{Models: []string{"a"}, ByEngine: map[string][]string{"llamacpp": {"a"}}})
		f := newModelsStreamFixture(t, builder)
		waitBuilds(t, builder, 1)
		lines := f.openStream(t)
		line, ok := nextFrame(t, lines, 2*time.Second)
		if !ok {
			t.Fatal("no current document on connect")
		}
		if got := decodeModels(t, line).Models; len(got) != 1 || got[0] != "a" {
			t.Fatalf("first frame models = %v, want [a]", got)
		}

		// An unchanged rebuild publishes nothing.
		f.pub.signal()
		waitBuilds(t, builder, 2)
		if line, ok := nextFrame(t, lines, 500*time.Millisecond); ok {
			t.Fatalf("unchanged rebuild produced a frame: %s", line)
		}

		builder.set(ModelsResult{Models: []string{"a", "b"}, ByEngine: map[string][]string{"llamacpp": {"a", "b"}}})
		f.pub.signal()
		line, ok = nextFrame(t, lines, 2*time.Second)
		if !ok {
			t.Fatal("changed inventory produced no frame after signal")
		}
		if got := decodeModels(t, line).Models; len(got) != 2 {
			t.Fatalf("second frame models = %v, want [a b]", got)
		}
	})

	t.Run("memory evidence freshness alone publishes nothing", func(t *testing.T) {
		builder := &modelsBuilder{}
		withEntry := func(entry performancewire.MemoryEvidenceEntry) ModelsResult {
			return ModelsResult{Models: []string{"demo"}, MemoryEvidence: &performancewire.MemoryEvidence{Version: 1, Entries: []performancewire.MemoryEvidenceEntry{entry}}}
		}
		builder.set(withEntry(freshnessEntry(10, 1)))
		f := newModelsStreamFixture(t, builder)
		waitBuilds(t, builder, 1)
		lines := f.openStream(t)
		line, ok := nextFrame(t, lines, 2*time.Second)
		if !ok {
			t.Fatal("no current document on connect")
		}
		// The fixture entry is not valid evidence, so check the wire bytes
		// rather than the strict decoder.
		if !strings.Contains(line, `"sampleAgeMs":10,"qualificationAgeMs":20,"observationGeneration":1`) {
			t.Fatalf("published frame must carry the full, un-normalized evidence: %s", line)
		}

		builder.set(withEntry(freshnessEntry(900, 7)))
		f.pub.signal()
		waitBuilds(t, builder, 2)
		if line, ok := nextFrame(t, lines, 500*time.Millisecond); ok {
			t.Fatalf("freshness-only change produced a frame: %s", line)
		}

		changed := freshnessEntry(900, 7)
		changed.State = performancewire.MemoryStateQualified
		builder.set(withEntry(changed))
		f.pub.signal()
		if _, ok := nextFrame(t, lines, 2*time.Second); !ok {
			t.Fatal("a memory status change produced no frame")
		}
	})

	t.Run("keepalives arrive at the configured interval", func(t *testing.T) {
		saved := modelsStreamKeepalive
		modelsStreamKeepalive = 50 * time.Millisecond
		t.Cleanup(func() { modelsStreamKeepalive = saved })
		builder := &modelsBuilder{}
		builder.set(ModelsResult{Models: []string{"a"}})
		f := newModelsStreamFixture(t, builder)
		waitBuilds(t, builder, 1)
		lines := f.openStream(t)
		if _, ok := nextFrame(t, lines, 2*time.Second); !ok {
			t.Fatal("no current document on connect")
		}
		keepalives := 0
		deadline := time.After(time.Second)
		for keepalives < 3 {
			select {
			case line := <-lines:
				if line != "null" {
					t.Fatalf("unexpected frame %q", line)
				}
				keepalives++
			case <-deadline:
				t.Fatalf("got %d keepalives in 1s at a 50ms interval", keepalives)
			}
		}
	})

	t.Run("unpinned TLS peer is refused", func(t *testing.T) {
		builder := &modelsBuilder{}
		f := newModelsStreamFixture(t, builder)
		resp, err := f.stranger.Get(f.url + modelsEventsPath)
		if err != nil {
			t.Fatalf("stranger request: %v", err)
		}
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusForbidden {
			t.Fatalf("stranger status = %d, want 403", resp.StatusCode)
		}
	})

	t.Run("plain loopback mux does not serve the stream", func(t *testing.T) {
		builder := &modelsBuilder{}
		f := newModelsStreamFixture(t, builder)
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, modelsEventsPath, nil)
		req.RemoteAddr = "127.0.0.1:51234"
		f.plain.ServeHTTP(rec, req)
		if rec.Code != http.StatusNotFound {
			t.Fatalf("plain %s status = %d, want 404", modelsEventsPath, rec.Code)
		}
	})

	t.Run("non-GET is refused", func(t *testing.T) {
		builder := &modelsBuilder{}
		f := newModelsStreamFixture(t, builder)
		resp, err := f.pinned.Post(f.url+modelsEventsPath, "application/json", strings.NewReader("{}"))
		if err != nil {
			t.Fatalf("POST: %v", err)
		}
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusMethodNotAllowed {
			t.Fatalf("POST status = %d, want 405", resp.StatusCode)
		}
	})
}

func TestModelsStreamSignalNeverBlocks(t *testing.T) {
	pub := newModelsPublisher()
	for range 10 {
		pub.signal()
	}
	var nilPub *modelsPublisher
	nilPub.signal()
}
