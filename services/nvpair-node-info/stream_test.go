// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

const mib = uint64(1) << 20

// baseNodeInfo is a fully populated sample the threshold cases mutate.
func baseNodeInfo() NodeInfoResponse {
	cluster := "cluster-a"
	return NodeInfoResponse{
		MemoryObservation: &noderec.MemoryObservation{
			RunID: "run-1",
			Domains: []noderec.MemoryDomain{{
				ID:    "system",
				Kind:  "shared",
				Total: noderec.MemoryFact{Valid: true},
				Free:  noderec.MemoryFact{Valid: true},
			}},
		},
		ServiceBuildID: "build-1",
		GPUs: []GPUInfo{{
			Name:               "GPU",
			PhysicalID:         "pci:1",
			VramBytes:          16 << 30,
			VramUsedBytes:      4 << 30,
			UtilizationPercent: 40,
		}},
		CPU:            &CPUInfo{Name: "CPU", Cores: 8, UtilizationPercent: 30},
		Memory:         &MemoryInfo{TotalBytes: 4 << 30, UsedBytes: 2 << 30},
		TelemetryValid: true,
		MSSince:        100,
		HostUUID:       "host-a",
		ClusterUUID:    &cluster,
	}
}

func TestNodeInfoChanged(t *testing.T) {
	empty, other := "", "cluster-b"
	memory32GiB := func(r *NodeInfoResponse) { r.Memory.TotalBytes = 32 << 30 }
	vram4GiB := func(r *NodeInfoResponse) { r.GPUs[0].VramBytes, r.GPUs[0].VramUsedBytes = 4<<30, 1<<30 }
	cases := []struct {
		name   string
		setup  func(*NodeInfoResponse) // applied to both samples
		mutate func(*NodeInfoResponse) // applied to the candidate
		want   bool
	}{
		{"unchanged sample", nil, func(*NodeInfoResponse) {}, false},
		{"age only", nil, func(r *NodeInfoResponse) { r.MSSince = 900 }, false},
		{"age reaches the stale bound", nil, func(r *NodeInfoResponse) { r.MSSince = 5000 }, true},
		{"stale age grows", func(r *NodeInfoResponse) { r.MSSince = 6000 }, func(r *NodeInfoResponse) { r.MSSince = 7000 }, true},
		{"stale age while telemetry invalid", func(r *NodeInfoResponse) { r.TelemetryValid, r.MSSince = false, 6000 }, func(r *NodeInfoResponse) { r.MSSince = 7000 }, false},
		{"GPU utilization +4", nil, func(r *NodeInfoResponse) { r.GPUs[0].UtilizationPercent += 4 }, false},
		{"GPU utilization +5", nil, func(r *NodeInfoResponse) { r.GPUs[0].UtilizationPercent += 5 }, true},
		{"GPU utilization -5", nil, func(r *NodeInfoResponse) { r.GPUs[0].UtilizationPercent -= 5 }, true},
		{"CPU utilization +9", nil, func(r *NodeInfoResponse) { r.CPU.UtilizationPercent += 9 }, false},
		{"CPU utilization +10", nil, func(r *NodeInfoResponse) { r.CPU.UtilizationPercent += 10 }, true},
		{"CPU utilization -10", nil, func(r *NodeInfoResponse) { r.CPU.UtilizationPercent -= 10 }, true},
		{"memory +63 MiB on 4 GiB", nil, func(r *NodeInfoResponse) { r.Memory.UsedBytes += 63 * mib }, false},
		{"memory +64 MiB on 4 GiB", nil, func(r *NodeInfoResponse) { r.Memory.UsedBytes += 64 * mib }, true},
		{"memory -64 MiB on 4 GiB", nil, func(r *NodeInfoResponse) { r.Memory.UsedBytes -= 64 * mib }, true},
		{"memory +300 MiB on 32 GiB", memory32GiB, func(r *NodeInfoResponse) { r.Memory.UsedBytes += 300 * mib }, false},
		{"memory +330 MiB on 32 GiB", memory32GiB, func(r *NodeInfoResponse) { r.Memory.UsedBytes += 330 * mib }, true},
		{"VRAM +63 MiB on 4 GiB", vram4GiB, func(r *NodeInfoResponse) { r.GPUs[0].VramUsedBytes += 63 * mib }, false},
		{"VRAM +64 MiB on 4 GiB", vram4GiB, func(r *NodeInfoResponse) { r.GPUs[0].VramUsedBytes += 64 * mib }, true},
		{"VRAM +160 MiB on 16 GiB", nil, func(r *NodeInfoResponse) { r.GPUs[0].VramUsedBytes += 160 * mib }, false},
		{"VRAM +164 MiB on 16 GiB", nil, func(r *NodeInfoResponse) { r.GPUs[0].VramUsedBytes += 164 * mib }, true},
		{"VRAM -164 MiB on 16 GiB", nil, func(r *NodeInfoResponse) { r.GPUs[0].VramUsedBytes -= 164 * mib }, true},
		{"telemetryValid flip", nil, func(r *NodeInfoResponse) { r.TelemetryValid = false }, true},
		{"clusterUuid value to absent", nil, func(r *NodeInfoResponse) { r.ClusterUUID = nil }, true},
		{"clusterUuid value to empty", nil, func(r *NodeInfoResponse) { r.ClusterUUID = &empty }, true},
		{"clusterUuid value to other", nil, func(r *NodeInfoResponse) { r.ClusterUUID = &other }, true},
		{"serviceBuildId", nil, func(r *NodeInfoResponse) { r.ServiceBuildID = "build-2" }, true},
		{"hostUuid", nil, func(r *NodeInfoResponse) { r.HostUUID = "host-b" }, true},
		{"GPU count", nil, func(r *NodeInfoResponse) { r.GPUs = append(r.GPUs, GPUInfo{Name: "GPU 2"}) }, true},
		{"GPU name", nil, func(r *NodeInfoResponse) { r.GPUs[0].Name = "Other" }, true},
		{"GPU physical ID", nil, func(r *NodeInfoResponse) { r.GPUs[0].PhysicalID = "" }, true},
		{"GPU VRAM total", nil, func(r *NodeInfoResponse) { r.GPUs[0].VramBytes = 8 << 30 }, true},
		{"CPU absent", nil, func(r *NodeInfoResponse) { r.CPU = nil }, true},
		{"CPU name", nil, func(r *NodeInfoResponse) { r.CPU.Name = "Other" }, true},
		{"CPU cores", nil, func(r *NodeInfoResponse) { r.CPU.Cores = 16 }, true},
		{"memory absent", nil, func(r *NodeInfoResponse) { r.Memory = nil }, true},
		{"memory total", nil, func(r *NodeInfoResponse) { r.Memory.TotalBytes = 8 << 30 }, true},
		{"memory observation absent", nil, func(r *NodeInfoResponse) { r.MemoryObservation = nil }, true},
		{"memory observation run ID", nil, func(r *NodeInfoResponse) { r.MemoryObservation.RunID = "run-2" }, true},
		{"memory observation domain ID", nil, func(r *NodeInfoResponse) { r.MemoryObservation.Domains[0].ID = "other" }, true},
		{"memory observation domain added", nil, func(r *NodeInfoResponse) {
			r.MemoryObservation.Domains = append(r.MemoryObservation.Domains, noderec.MemoryDomain{ID: "gpu"})
		}, true},
		{"memory observation total valid", nil, func(r *NodeInfoResponse) { r.MemoryObservation.Domains[0].Total.Valid = false }, true},
		{"memory observation free valid", nil, func(r *NodeInfoResponse) { r.MemoryObservation.Domains[0].Free.Valid = false }, true},
		{"memory observation bytes only", nil, func(r *NodeInfoResponse) { r.MemoryObservation.Domains[0].Free.Bytes = "123" }, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			published, candidate := baseNodeInfo(), baseNodeInfo()
			if tc.setup != nil {
				tc.setup(&published)
				tc.setup(&candidate)
			}
			tc.mutate(&candidate)
			if got := nodeInfoChanged(published, candidate); got != tc.want {
				t.Errorf("nodeInfoChanged = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestNodeInfoChangedReplay publishes a recorded sequence of consecutive
// one-second bodies (captured on a development build) through the threshold
// rule and expects fewer frames than samples.
func TestNodeInfoChangedReplay(t *testing.T) {
	file, err := os.Open("testdata/node-info-sequence.txt")
	if err != nil {
		t.Fatalf("open sequence: %v", err)
	}
	defer file.Close()
	var samples, frames int
	var published NodeInfoResponse
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64<<10), 1<<20)
	for scanner.Scan() {
		var candidate NodeInfoResponse
		if err := json.Unmarshal(scanner.Bytes(), &candidate); err != nil {
			t.Fatalf("sample %d: %v", samples+1, err)
		}
		if samples == 0 || nodeInfoChanged(published, candidate) {
			published = candidate
			frames++
		}
		samples++
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("read sequence: %v", err)
	}
	if samples < 20 {
		t.Fatalf("sequence holds %d samples, want at least 20", samples)
	}
	if frames >= samples {
		t.Fatalf("published %d frames for %d samples, want fewer frames than samples", frames, samples)
	}
	t.Logf("%d frames for %d samples", frames, samples)
}

// streamPublisher starts a publisher whose body reports one GPU at the given
// utilization, with a 50 ms keepalive, both undone at cleanup.
func streamPublisher(t *testing.T, utilization *atomic.Uint32) *nodeInfoPublisher {
	t.Helper()
	keepalive := nodeInfoStreamKeepalive
	nodeInfoStreamKeepalive = 50 * time.Millisecond
	t.Cleanup(func() { nodeInfoStreamKeepalive = keepalive })
	pub := &nodeInfoPublisher{body: func() []byte {
		body, err := json.Marshal(NodeInfoResponse{
			GPUs:           []GPUInfo{{Name: "GPU", VramBytes: 16 << 30, UtilizationPercent: utilization.Load()}},
			TelemetryValid: true,
			HostUUID:       "stream-host",
		})
		if err != nil {
			t.Errorf("marshal body: %v", err)
		}
		return body
	}}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		pub.run(ctx)
	}()
	t.Cleanup(func() {
		cancel()
		<-done
	})
	return pub
}

// streamLines reads NDJSON lines from body into a channel that closes at EOF.
func streamLines(body io.Reader) <-chan string {
	lines := make(chan string, 64)
	go func() {
		defer close(lines)
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64<<10), 1<<20)
		for scanner.Scan() {
			lines <- scanner.Text()
		}
	}()
	return lines
}

// nextFrame returns the next non-null line, failing after timeout or at EOF.
func nextFrame(t *testing.T, lines <-chan string, timeout time.Duration) NodeInfoResponse {
	t.Helper()
	deadline := time.After(timeout)
	for {
		select {
		case line, open := <-lines:
			if !open {
				t.Fatal("stream ended before a frame arrived")
			}
			if line == "null" {
				continue
			}
			var frame NodeInfoResponse
			if err := json.Unmarshal([]byte(line), &frame); err != nil {
				t.Fatalf("decode frame %q: %v", line, err)
			}
			return frame
		case <-deadline:
			t.Fatalf("no frame within %s", timeout)
		}
	}
}

// signalWriter is a ResponseWriter for direct handler calls whose stream must
// stay open: it reports the first write and keeps the status code.
type signalWriter struct {
	header http.Header
	status atomic.Int32
	wrote  chan struct{}
	once   sync.Once
}

func newSignalWriter() *signalWriter {
	return &signalWriter{header: make(http.Header), wrote: make(chan struct{})}
}

func (w *signalWriter) Header() http.Header { return w.header }
func (w *signalWriter) WriteHeader(code int) {
	w.status.CompareAndSwap(0, int32(code))
}
func (w *signalWriter) Write(p []byte) (int, error) {
	w.WriteHeader(http.StatusOK)
	w.once.Do(func() { close(w.wrote) })
	return len(p), nil
}
func (w *signalWriter) Flush() {}

func TestNodeInfoStream(t *testing.T) {
	t.Run("unclustered stream sends a frame then only threshold changes", func(t *testing.T) {
		var utilization atomic.Uint32
		utilization.Store(40)
		pub := streamPublisher(t, &utilization)
		mux := http.NewServeMux()
		mux.HandleFunc("/v1/node-info/events", nodeInfoStreamHandler(clustertrust.Open(""), pub))
		srv := httptest.NewServer(mux)
		t.Cleanup(srv.Close)

		ctx, cancel := context.WithCancel(context.Background())
		t.Cleanup(cancel)
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL+"/v1/node-info/events", nil)
		if err != nil {
			t.Fatalf("new request: %v", err)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("subscribe: %v", err)
		}
		t.Cleanup(func() { resp.Body.Close() })
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
		if got := resp.Header.Get("Cache-Control"); got != "no-store" {
			t.Errorf("Cache-Control = %q, want no-store", got)
		}
		lines := streamLines(resp.Body)
		first := nextFrame(t, lines, 3*time.Second)
		if first.HostUUID != "stream-host" || len(first.GPUs) != 1 || first.GPUs[0].UtilizationPercent != 40 {
			t.Fatalf("first frame = %+v, want stream-host at 40 %%", first)
		}

		// Within the threshold: only keepalives across more than one publish tick.
		utilization.Store(42)
		quiet := time.After(nodeInfoPublishEvery + 500*time.Millisecond)
		nulls := 0
	within:
		for {
			select {
			case line, open := <-lines:
				if !open {
					t.Fatal("stream ended during the quiet window")
				}
				if line != "null" {
					t.Fatalf("got frame %q for a 2-point change, want only null keepalives", line)
				}
				nulls++
			case <-quiet:
				break within
			}
		}
		if nulls == 0 {
			t.Fatal("no keepalive arrived during the quiet window")
		}

		utilization.Store(52)
		if frame := nextFrame(t, lines, 3*time.Second); frame.GPUs[0].UtilizationPercent != 52 {
			t.Fatalf("second frame utilization = %d, want 52", frame.GPUs[0].UtilizationPercent)
		}
	})

	t.Run("per-address cap refuses the ninth stream from one LAN address", func(t *testing.T) {
		var utilization atomic.Uint32
		pub := streamPublisher(t, &utilization)
		handler := nodeInfoStreamHandler(clustertrust.Open(""), pub)
		ctx, cancel := context.WithCancel(context.Background())
		var wg sync.WaitGroup
		t.Cleanup(func() {
			cancel()
			wg.Wait()
		})
		open := func(remoteAddr string) *signalWriter {
			t.Helper()
			req := httptest.NewRequest(http.MethodGet, "/v1/node-info/events", nil).WithContext(ctx)
			req.RemoteAddr = remoteAddr
			w := newSignalWriter()
			wg.Add(1)
			go func() {
				defer wg.Done()
				handler(w, req)
			}()
			select {
			case <-w.wrote:
			case <-time.After(3 * time.Second):
				t.Fatalf("stream from %s wrote nothing", remoteAddr)
			}
			return w
		}
		for i := range nodeInfoStreamsPerAddress {
			if w := open("192.0.2.10:" + strconv.Itoa(1000+i)); w.status.Load() != http.StatusOK {
				t.Fatalf("stream %d status = %d, want 200", i+1, w.status.Load())
			}
		}
		if w := open("192.0.2.10:9999"); w.status.Load() != http.StatusServiceUnavailable {
			t.Fatalf("ninth stream status = %d, want 503", w.status.Load())
		}
		if w := open("192.0.2.11:1"); w.status.Load() != http.StatusOK {
			t.Fatalf("other address status = %d, want 200", w.status.Load())
		}
		for i := range nodeInfoStreamsPerAddress + 1 {
			if w := open("127.0.0.1:" + strconv.Itoa(1000+i)); w.status.Load() != http.StatusOK {
				t.Fatalf("loopback stream %d status = %d, want 200", i+1, w.status.Load())
			}
		}
	})

	t.Run("total cap refuses streams beyond the hub limit", func(t *testing.T) {
		var utilization atomic.Uint32
		pub := streamPublisher(t, &utilization)
		pub.hub.Limit = 1
		_, cancelReader, ok := pub.hub.Subscribe()
		if !ok {
			t.Fatal("first hub reader refused")
		}
		t.Cleanup(cancelReader)
		rec := httptest.NewRecorder()
		nodeInfoStreamHandler(clustertrust.Open(""), pub)(rec, httptest.NewRequest(http.MethodGet, "/v1/node-info/events", nil))
		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("status = %d, want 503", rec.Code)
		}
	})

	t.Run("non-GET is refused", func(t *testing.T) {
		var utilization atomic.Uint32
		pub := streamPublisher(t, &utilization)
		rec := httptest.NewRecorder()
		nodeInfoStreamHandler(clustertrust.Open(""), pub)(rec, httptest.NewRequest(http.MethodPost, "/v1/node-info/events", nil))
		if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != http.MethodGet {
			t.Fatalf("status = %d Allow = %q, want 405 GET", rec.Code, rec.Header().Get("Allow"))
		}
	})

	t.Run("cluster-gated stream admits pinned peers and ends when the pin goes", func(t *testing.T) {
		aCert, aKey := genLeaf(t, "uuid-a")
		bCert, bKey := genLeaf(t, "uuid-b")
		cCert, cKey := genLeaf(t, "uuid-c")
		dCert, _ := genLeaf(t, "uuid-d")
		// A pins B and D, so removing B's pin leaves A clustered. C pins A,
		// but A does not pin C.
		dirA := setupNode(t, aCert, aKey, map[string][]byte{"uuid-b": bCert, "uuid-d": dCert})
		meshA := clustertrust.Open(dirA)
		meshB := clustertrust.Open(setupNode(t, bCert, bKey, map[string][]byte{"uuid-a": aCert}))
		meshC := clustertrust.Open(setupNode(t, cCert, cKey, map[string][]byte{"uuid-a": aCert}))

		var utilization atomic.Uint32
		pub := streamPublisher(t, &utilization)
		handler := nodeInfoStreamHandler(meshA, pub)
		mux := http.NewServeMux()
		mux.HandleFunc("/v1/node-info/events", handler)
		srv := httptest.NewUnstartedServer(mux)
		srv.TLS = meshA.ServerTLSConfig()
		srv.StartTLS()
		t.Cleanup(srv.Close)

		subscribe := func(m *clustertrust.Mesh) *http.Response {
			t.Helper()
			cfg, ok := m.ClientTLSConfig("uuid-a")
			if !ok {
				t.Fatal("no client config for uuid-a")
			}
			client := &http.Client{Transport: &http.Transport{TLSClientConfig: cfg}}
			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL+"/v1/node-info/events", nil)
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			resp, err := client.Do(req)
			if err != nil {
				t.Fatalf("subscribe: %v", err)
			}
			t.Cleanup(func() { resp.Body.Close() })
			return resp
		}

		if resp := subscribe(meshC); resp.StatusCode != http.StatusForbidden {
			t.Fatalf("unpinned client status = %d, want 403", resp.StatusCode)
		}
		plain := httptest.NewRecorder()
		handler(plain, httptest.NewRequest(http.MethodGet, "/v1/node-info/events", nil))
		if plain.Code != http.StatusForbidden {
			t.Fatalf("plain-HTTP client status = %d, want 403", plain.Code)
		}

		resp := subscribe(meshB)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("pinned client status = %d, want 200", resp.StatusCode)
		}
		lines := streamLines(resp.Body)
		nextFrame(t, lines, 3*time.Second)

		if err := os.Remove(filepath.Join(dirA, "trusted", "uuid-b.json")); err != nil {
			t.Fatalf("remove pin: %v", err)
		}
		deadline := time.After(time.Second)
		for {
			select {
			case _, open := <-lines:
				if !open {
					return
				}
			case <-deadline:
				t.Fatal("stream still open 1 s after its pin was removed")
			}
		}
	})
}
