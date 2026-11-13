// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
)

func shortenEnginesKeepalive(t *testing.T, d time.Duration) {
	t.Helper()
	previous := enginesStreamKeepalive
	enginesStreamKeepalive = d
	t.Cleanup(func() { enginesStreamKeepalive = previous })
}

// enginesList is a controllable status list for the publisher under test.
type enginesList struct {
	mu   sync.Mutex
	list []EngineStatus
}

func (l *enginesList) set(list ...EngineStatus) {
	l.mu.Lock()
	l.list = list
	l.mu.Unlock()
}

func (l *enginesList) get() []EngineStatus {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]EngineStatus(nil), l.list...)
}

func newEnginesTestExecutor(t *testing.T, list *enginesList) *Executor {
	t.Helper()
	exec := NewExecutor(NewRegistry(), NewReporter(nil), func(string, any) {}, t.TempDir())
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); exec.enginesEvents.run(ctx, list.get) }()
	t.Cleanup(func() { cancel(); <-done })
	return exec
}

func TestEnginesStreamDefaults(t *testing.T) {
	if enginesStreamKeepalive != 10*time.Second || enginesStreamIdle != 25*time.Second ||
		enginesRebuildEvery != 10*time.Second || enginesRebuildDelay != 250*time.Millisecond {
		t.Fatalf("keepalive/idle/rebuild/delay = %v/%v/%v/%v, want 10s/25s/10s/250ms",
			enginesStreamKeepalive, enginesStreamIdle, enginesRebuildEvery, enginesRebuildDelay)
	}
}

func TestEnginesStreamEmitStateSignalsPublisher(t *testing.T) {
	exec := settingsExecutor(t, false)
	select {
	case <-exec.enginesEvents.signal:
	default:
	}
	exec.emitState("fake")
	select {
	case <-exec.enginesEvents.signal:
	default:
		t.Fatal("engine:state-changed did not signal the engine status publisher")
	}
}

func TestEngineEventsStream(t *testing.T) {
	shortenEnginesKeepalive(t, 50*time.Millisecond)
	a, certA := settingsMesh(t, "a")
	b, certB := settingsMesh(t, "b")
	stranger, _ := settingsMesh(t, "stranger")
	settingsPin(t, a, "b", certB)
	for _, dir := range []string{b, stranger} {
		settingsPin(t, dir, "a", certA)
	}
	mesh := clustertrust.Open(a)
	list := &enginesList{}
	list.set(EngineStatus{Engine: "llamacpp", DisplayName: "llama.cpp", Installed: true})
	exec := newEnginesTestExecutor(t, list)
	server := httptest.NewUnstartedServer((&controlServer{exec: exec, mesh: mesh}).mux())
	server.TLS = mesh.ServerTLSConfig()
	server.StartTLS()
	defer server.Close()
	client := func(dir string) *http.Client {
		config, ok := clustertrust.Open(dir).ClientTLSConfig("a")
		if !ok {
			t.Fatal("pin unavailable")
		}
		transport := &http.Transport{TLSClientConfig: config}
		t.Cleanup(transport.CloseIdleConnections)
		return &http.Client{Transport: transport}
	}

	t.Run("unpinned caller is refused", func(t *testing.T) {
		res, err := client(stranger).Get(server.URL + controlEnginesPath + "/events")
		if err != nil {
			t.Fatal(err)
		}
		_ = res.Body.Close()
		if res.StatusCode != http.StatusForbidden {
			t.Fatalf("status = %d, want 403", res.StatusCode)
		}
	})

	t.Run("pinned caller gets the list, then changes and keepalives", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, server.URL+controlEnginesPath+"/events", nil)
		if err != nil {
			t.Fatal(err)
		}
		res, err := client(b).Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		if res.StatusCode != http.StatusOK || res.Header.Get("Cache-Control") != "no-store" {
			t.Fatalf("status/cache = %d/%q, want 200/no-store", res.StatusCode, res.Header.Get("Cache-Control"))
		}
		lines := make(chan string, 64)
		go func() {
			defer close(lines)
			scan := bufio.NewScanner(res.Body)
			for scan.Scan() {
				lines <- scan.Text()
			}
		}()
		// next returns the next frame, counting keepalives skipped on the way;
		// "" means none arrived within wait.
		next := func(wait time.Duration) (string, int) {
			t.Helper()
			keepalives := 0
			deadline := time.After(wait)
			for {
				select {
				case line, open := <-lines:
					if !open {
						t.Fatal("stream ended")
					}
					if line == "null" {
						keepalives++
						continue
					}
					return line, keepalives
				case <-deadline:
					return "", keepalives
				}
			}
		}
		decode := func(line string) []EngineStatus {
			t.Helper()
			var frame struct {
				Engines []EngineStatus `json:"engines"`
			}
			if err := json.Unmarshal([]byte(line), &frame); err != nil {
				t.Fatalf("decode frame %q: %v", line, err)
			}
			return frame.Engines
		}
		first, _ := next(2 * time.Second)
		if got := decode(first); len(got) != 1 || got[0].Engine != "llamacpp" || got[0].Running {
			t.Fatalf("first frame = %+v", got)
		}
		// An unchanged rebuild publishes nothing; only keepalives arrive.
		exec.enginesEvents.poke()
		if line, keepalives := next(time.Second); line != "" || keepalives < 3 {
			t.Fatalf("unchanged rebuild: frame %q, keepalives %d (want none, at least 3)", line, keepalives)
		}
		list.set(EngineStatus{Engine: "llamacpp", DisplayName: "llama.cpp", Installed: true, Running: true, Healthy: true})
		exec.enginesEvents.poke()
		changed, _ := next(2 * time.Second)
		if got := decode(changed); len(got) != 1 || !got[0].Running || !got[0].Healthy {
			t.Fatalf("changed frame = %+v", got)
		}
	})
}

func TestRemoteEngineCacheIsPerPeer(t *testing.T) {
	var cache remoteEngineCache
	live := true
	cache.peers = map[string]*remoteEnginesEntry{
		"peer-a": {engines: json.RawMessage(`[{"engine":"a"}]`), lineAt: time.Now(), current: func() bool { return live }},
		"peer-b": {engines: json.RawMessage(`[{"engine":"b"}]`), lineAt: time.Now(), current: func() bool { return true }},
		"stale":  {engines: json.RawMessage(`[]`), lineAt: time.Now().Add(-enginesStreamIdle - time.Second), current: func() bool { return true }},
		"empty":  {lineAt: time.Now(), current: func() bool { return true }},
	}
	for node, want := range map[string]string{"peer-a": `{"engines":[{"engine":"a"}]}`, "peer-b": `{"engines":[{"engine":"b"}]}`} {
		got, err := cache.answer(node)
		if err != nil || string(got) != want {
			t.Errorf("answer(%s) = %s, %v; want %s", node, got, err, want)
		}
	}
	for _, node := range []string{"stale", "empty", "missing"} {
		if _, err := cache.answer(node); err != errRemoteEnginesUnavailable {
			t.Errorf("answer(%s) error = %v, want unavailable", node, err)
		}
	}
	live = false
	if _, err := cache.answer("peer-a"); err != errRemoteEnginesUnavailable {
		t.Errorf("answer after route change error = %v, want unavailable", err)
	}
	if _, err := cache.answer("peer-b"); err != nil {
		t.Errorf("other peer affected by route change: %v", err)
	}
}

// remoteGetInstalled sends engine:remote-get-installed and returns the reply.
func remoteGetInstalled(t *testing.T, m *Manager, out *syncBuffer, id int) (json.RawMessage, string) {
	t.Helper()
	raw := json.RawMessage(mustJSONInt(id))
	m.runRemote(context.Background(), &Message{JSONRPC: "2.0", ID: &raw, Method: "engine:remote-get-installed",
		Params: json.RawMessage(`{"node":"host-peer"}`)})
	for _, frame := range out.frames() {
		if frame.ID != nil && string(*frame.ID) == string(raw) {
			if frame.Error != nil {
				return nil, frame.Error.Message
			}
			return frame.Result, ""
		}
	}
	t.Fatalf("no reply for request %d", id)
	return nil, ""
}

func mustJSONInt(id int) []byte {
	data, _ := json.Marshal(id)
	return data
}

func TestEnginesStreamPeerCacheAndNotification(t *testing.T) {
	shortenEnginesKeepalive(t, 50*time.Millisecond)
	list := &enginesList{}
	list.set(EngineStatus{Engine: "llamacpp", DisplayName: "llama.cpp", Installed: true})
	exec := newEnginesTestExecutor(t, list)
	var fetches, streams atomic.Int32
	f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
		handler := (&controlServer{exec: exec, mesh: mesh}).mux()
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case controlEnginesPath:
				fetches.Add(1)
			case controlEnginesPath + "/events":
				streams.Add(1)
			}
			handler.ServeHTTP(w, r)
		})
	})
	out := &syncBuffer{}
	f.m.codec = NewCodec(out)
	type change struct {
		NodeID  string         `json:"nodeId"`
		Engines []EngineStatus `json:"engines"`
	}
	changes := make(chan change, 8)
	f.m.exec.emit = func(method string, value any) {
		if method != "engine:remote-installed-changed" {
			return
		}
		data, err := json.Marshal(value)
		if err != nil {
			t.Error(err)
			return
		}
		var c change
		if err := json.Unmarshal(data, &c); err != nil {
			t.Error(err)
			return
		}
		changes <- c
	}
	awaitChange := func() change {
		t.Helper()
		select {
		case c := <-changes:
			return c
		case <-time.After(5 * time.Second):
			t.Fatal("engine:remote-installed-changed did not arrive")
			return change{}
		}
	}
	if _, msg := remoteGetInstalled(t, f.m, out, 1); msg != errRemoteEnginesUnavailable.Error() {
		t.Fatalf("before the stream: error %q, want unavailable", msg)
	}

	peer, ok := f.m.peers.lookup("host-peer")
	if !ok {
		t.Fatal("missing peer")
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); f.m.consumeEnginesStream(ctx, peer) }()
	t.Cleanup(func() { cancel(); <-done })

	first := awaitChange()
	if first.NodeID != "host-peer" || len(first.Engines) != 1 || first.Engines[0].Engine != "llamacpp" || first.Engines[0].Running {
		t.Fatalf("first notification = %+v", first)
	}
	result, msg := remoteGetInstalled(t, f.m, out, 2)
	var reply struct {
		Engines []EngineStatus `json:"engines"`
	}
	if msg != "" || json.Unmarshal(result, &reply) != nil || len(reply.Engines) != 1 || reply.Engines[0].Engine != "llamacpp" {
		t.Fatalf("cached reply = %s, %q", result, msg)
	}
	if fetches.Load() != 0 || streams.Load() != 1 {
		t.Fatalf("GET /v1/engines / streams = %d/%d, want 0/1", fetches.Load(), streams.Load())
	}

	// An identical rebuild notifies nothing; a change notifies once.
	exec.enginesEvents.poke()
	select {
	case c := <-changes:
		t.Fatalf("unchanged list notified: %+v", c)
	case <-time.After(600 * time.Millisecond):
	}
	list.set(EngineStatus{Engine: "llamacpp", DisplayName: "llama.cpp", Installed: true, Running: true})
	exec.enginesEvents.poke()
	if c := awaitChange(); len(c.Engines) != 1 || !c.Engines[0].Running {
		t.Fatalf("change notification = %+v", c)
	}

	// A changed route makes the answer-time check fail before the stream ends.
	clustertrusttest.RemovePeerPin(t, f.selfDir, "uuid-peer")
	if _, msg := remoteGetInstalled(t, f.m, out, 3); msg != errRemoteEnginesUnavailable.Error() {
		t.Fatalf("after pin removal: error %q, want unavailable", msg)
	}
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("engine status consumer did not stop after pin removal")
	}
	f.m.remoteEngines.mu.Lock()
	remaining := len(f.m.remoteEngines.peers)
	f.m.remoteEngines.mu.Unlock()
	if remaining != 0 {
		t.Fatalf("cache entries after stream end = %d, want 0", remaining)
	}
	if fetches.Load() != 0 {
		t.Fatalf("GET /v1/engines = %d, want 0", fetches.Load())
	}
}
