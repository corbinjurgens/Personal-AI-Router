// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func init() {
	if image := os.Getenv("NVPAIR_TEST_MEMORY_NODE_INFO"); image != "" && image == os.Args[0] && os.Getenv("NVPAIR_TEST_MODEL_ENGINE") == "" {
		if err := runModelMemoryNodeInfo(); err != nil {
			fmt.Fprintln(os.Stderr, "private memory producer:", err)
			os.Exit(1)
		}
		os.Exit(0)
	}
}

// Only the producer is fake. Its HTTP answer traverses the real scanner's
// generation and boot-clock checks, NodeTelemetry and the broker's local cache.
// The executable is the already registered services/tests image.
func runModelMemoryNodeInfo() error {
	values := map[string]string{}
	for i := 1; i+1 < len(os.Args); i += 2 {
		values[os.Args[i]] = os.Args[i+1]
	}
	port, err := strconv.Atoi(values["--port"])
	if err != nil || validateFixturePort(port) != nil || values["--node-id"] == "" {
		return errors.New("missing private port or identity")
	}
	listener, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
	if err != nil {
		return err
	}
	client := &http.Client{Timeout: 5 * time.Second}
	defer client.CloseIdleConnections()
	server := &http.Server{ReadHeaderTimeout: time.Second, Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/node-info" {
			http.NotFound(w, r)
			return
		}
		response, err := client.Get(os.Getenv("NVPAIR_TEST_MODEL_CONTROL") + "/memory-node-info?host=" + url.QueryEscape(values["--node-id"]))
		if err != nil {
			http.Error(w, "private producer unavailable", http.StatusServiceUnavailable)
			return
		}
		defer response.Body.Close()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(response.StatusCode)
		if _, err := io.Copy(w, io.LimitReader(response.Body, noderec.MaxMemoryObservationBytes)); err != nil {
			fmt.Fprintln(os.Stderr, "private memory response failed")
		}
	})}
	done := make(chan error, 1)
	go func() { done <- server.Serve(listener) }()
	_, readErr := io.Copy(io.Discard, os.Stdin)
	closeErr := server.Close()
	serveErr := <-done
	if errors.Is(serveErr, http.ErrServerClosed) {
		serveErr = nil
	}
	return errors.Join(readErr, closeErr, serveErr)
}

type modelMemoryProcessSource struct {
	observation *noderec.MemoryObservation
	requests    int
}

func (f *modelSettingsProcessFixture) enableMemory(t *testing.T, pools ...noderec.MemoryDomain) {
	t.Helper()
	f.scanner = scannerBin
	f.memory = &modelMemoryProcessSource{observation: &noderec.MemoryObservation{RunID: "private-memory-run-1", Domains: pools}}
	// Phase 15 restores installed engines on available nodes. A configured
	// pause is the real current way to keep inspection fixtures stopped.
	path := filepath.Join(jointAppDir(f.base), nodepolicy.FileName)
	var policy nodepolicy.Policy
	modelSettingsDecode(t, readLiveDurableFile(t, path), &policy)
	policy.Availability, policy.PauseMode = nodepolicy.Paused, nodepolicy.PauseConfigured
	jointWriteJSON(t, path, policy)
}

func modelMemoryPool(id, kind, total, available string, devices ...string) noderec.MemoryDomain {
	fact := func(value string) noderec.MemoryFact {
		return noderec.MemoryFact{Bytes: value, Source: "private-memory-fixture", SampledAt: time.Now().UTC(), Valid: true}
	}
	return noderec.MemoryDomain{ID: id, Kind: kind, DeviceIDs: devices, Total: fact(total), Free: fact(available)}
}

func (f *modelSettingsProcessFixture) serveMemoryNodeInfo(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.memory == nil {
		http.Error(w, "private producer not configured", http.StatusServiceUnavailable)
		return
	}
	f.memory.requests++
	value := struct {
		HostUUID string                     `json:"hostUuid"`
		GPUs     []map[string]string        `json:"GPUs"`
		Memory   *noderec.MemoryObservation `json:"memoryObservation,omitempty"`
	}{HostUUID: r.URL.Query().Get("host"), Memory: f.memory.observation.Clone(), GPUs: []map[string]string{}}
	// Equal display names intentionally carry no physical joining authority.
	if value.Memory != nil {
		for i := range value.Memory.Domains {
			pool := &value.Memory.Domains[i]
			pool.Total.SampledAt = time.Now().UTC().Add(-time.Duration(pool.Total.AgeMS) * time.Millisecond)
			pool.Free.SampledAt = time.Now().UTC().Add(-time.Duration(pool.Free.AgeMS) * time.Millisecond)
			if pool.Kind == "dedicated" {
				value.GPUs = append(value.GPUs, map[string]string{"name": "identical private GPU"})
			}
		}
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(value); err != nil {
		fmt.Fprintln(os.Stderr, "private memory encoding failed")
	}
}

func (f *modelSettingsProcessFixture) inspectMemory(t *testing.T) modelconfig.InspectResult {
	t.Helper()
	var value modelconfig.InspectResult
	modelSettingsDecode(t, f.rpcOK(t, "engine:inspect-model", modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "medium-model"}), &value)
	return value
}

func (f *modelSettingsProcessFixture) waitMemory(t *testing.T, capacity string) modelconfig.InspectResult {
	t.Helper()
	var value modelconfig.InspectResult
	task1Eventually(t, "scanner-backed target memory", 30*time.Second, func() bool {
		value = f.inspectMemory(t)
		return slices.ContainsFunc(value.Estimate.Domains, func(d modelconfig.MemoryDomain) bool { return d.CapacityBytes == capacity && d.RemainingValidityMS > 0 })
	})
	return value
}

func assertModelMemoryProvenance(t *testing.T, value modelconfig.Estimate, kind, total, available string) modelconfig.MemoryDomain {
	t.Helper()
	index := slices.IndexFunc(value.Domains, func(d modelconfig.MemoryDomain) bool { return d.Kind == kind })
	if index < 0 {
		t.Fatalf("missing %s pool: %+v", kind, value.Domains)
	}
	d := value.Domains[index]
	if d.CapacityBytes != total || d.FreeBytes != available || d.CapacitySource != "private-memory-fixture" || d.FreeSource != d.CapacitySource || d.CapacityTimestamp == "" || d.FreeTimestamp == "" || d.SampleID == "" || d.RemainingValidityMS <= 0 || d.RemainingValidityMS > 5000 {
		t.Fatalf("lost target memory provenance: %+v", d)
	}
	return d
}

// Reuse the existing admitted reverse-transport fixture. The observer forwards
// every request to this fixture's real manager and never supplies a result.
func modelMemoryReversePeer(t *testing.T, f *modelSettingsProcessFixture, nodes []*task1Node, blockedPort int) *task1Observer {
	t.Helper()
	origin, target := nodes[0], nodes[1]
	target.ports[noderec.ServiceEngineControl] = f.ports.engineControl
	observer := newTask1Observer(t, target, noderec.ServiceEngineControl, nodes)
	target.cm.call(t, "cluster:set-local-services", map[string]any{"generation": 2, "services": []noderec.RegisterParams{
		{Service: noderec.ServiceCluster, Port: target.observers[noderec.ServiceCluster].port},
		{Service: noderec.ServiceEngineControl, Port: observer.port},
	}})
	managementSet(t, target.cm, origin.uuid, true)
	task1Eventually(t, "memory reverse route", 20*time.Second, func() bool {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		conn, err := clustertrust.DialReverse(ctx, origin.base, target.uuid, noderec.ServiceEngineControl)
		if err != nil {
			return false
		}
		if err := conn.Close(); err != nil {
			t.Fatal(err)
		}
		return true
	})
	adapter := clustertrust.NewPeerServiceClient(origin.mesh, clustertrust.PeerServiceClientOptions{
		BaseConfigDir: origin.base, PeerClientOptions: clustertrust.PeerClientOptions{Timeout: 15 * time.Second},
		ResolveRoute: func(uuid string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
			origin.mesh.Refresh()
			incarnation, ok := origin.mesh.PeerServiceIncarnation(uuid)
			return clustertrust.PeerServiceRoute{Incarnation: incarnation, Address: net.JoinHostPort("127.0.0.1", strconv.Itoa(blockedPort)), Protocol: clustertrust.ReverseTLS, Generation: 1}, ok && uuid == target.uuid && service == noderec.ServiceEngineControl
		},
	})
	t.Cleanup(adapter.CloseIdle)
	f.client = adapter.Client(target.uuid, noderec.ServiceEngineControl)
	f.peerPort = blockedPort
	return observer
}

func TestModelMemoryTelemetryProcess(t *testing.T) {
	t.Run("reverse peer inspection and preview retain target memory", func(t *testing.T) {
		nodes, blocked := newTask1Nodes(t)
		f := prepareModelSettingsProcessFixtureAt(t, nodes[1].base)
		f.enableMemory(t, modelMemoryPool("ram", "system", "17179869184", "8589934592"))
		f.startModelSettings(t)
		f.waitMemory(t, "17179869184")
		observer := modelMemoryReversePeer(t, f, nodes, blocked)
		var result modelconfig.InspectResult
		modelSettingsDecode(t, f.peer(t, "inspect", modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "medium-model"}, http.StatusOK), &result)
		assertModelMemoryProvenance(t, result.Estimate, "host", "17179869184", "8589934592")
		var preview settings.Preview
		modelSettingsDecode(t, f.peer(t, "preview", f.request(t, "reverse-memory-preview", nil), http.StatusOK), &preview)
		if preview.Estimate == nil || len(preview.Errors) != 0 {
			t.Fatal("reverse preview lost target estimate")
		}
		assertModelMemoryProvenance(t, *preview.Estimate, "host", "17179869184", "8589934592")
		for _, path := range []string{"inspect", "preview"} {
			if observer.count(nodes[0].uuid, http.MethodPost, "/v1/engine-settings/"+path) == 0 {
				t.Fatalf("%s bypassed the authenticated reverse route", path)
			}
		}
	})
	t.Run("local and pinned inspection and preview use scanner memory", func(t *testing.T) {
		f := prepareModelSettingsProcessFixture(t)
		f.pinned(t)
		f.enableMemory(t, modelMemoryPool("ram", "system", "17179869184", "8589934592"))
		f.startModelSettings(t)
		before := f.metadataBaseline(t)
		local := f.waitMemory(t, "17179869184")
		first := assertModelMemoryProvenance(t, local.Estimate, "host", "17179869184", "8589934592")
		var peer modelconfig.InspectResult
		params := modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "medium-model"}
		modelSettingsDecode(t, f.peer(t, "inspect", params, http.StatusOK), &peer)
		assertModelMemoryProvenance(t, peer.Estimate, "host", "17179869184", "8589934592")
		request := f.request(t, "memory-preview", nil)
		_, localPreview := f.review(t, request)
		var peerPreview settings.Preview
		modelSettingsDecode(t, f.peer(t, "preview", request, http.StatusOK), &peerPreview)
		for _, preview := range []settings.Preview{localPreview, peerPreview} {
			if preview.Estimate == nil || len(preview.Errors) != 0 {
				t.Fatal("manual preview lost its advisory estimate")
			}
			assertModelMemoryProvenance(t, *preview.Estimate, "host", "17179869184", "8589934592")
		}
		// Public callers cannot smuggle origin capacity through either boundary.
		forged := map[string]any{"engine": "llamacpp", "modelId": "medium-model", "memory": map[string]string{"capacityBytes": "999999999999"}}
		if reply := f.rpc(t, "engine:inspect-model", forged); reply.Error == nil || reply.Error.Code != -32602 {
			t.Fatal("public inspection accepted origin capacity")
		}
		f.peer(t, "inspect", forged, http.StatusBadRequest)
		var forgedPreview map[string]json.RawMessage
		modelSettingsDecode(t, mustMarshal(t, request), &forgedPreview)
		forgedPreview["memory"] = mustMarshal(t, map[string]string{"capacityBytes": "999999999999"})
		if reply := f.rpc(t, "engine:preview-settings", forgedPreview); reply.Error == nil || reply.Error.Code != -32602 {
			t.Fatal("public preview accepted origin capacity")
		}
		f.peer(t, "preview", forgedPreview, http.StatusBadRequest)
		// A replaced producer has new observation authority, including on an
		// explicit repeat of Preview (the public Recalculate operation).
		f.mu.Lock()
		f.memory.observation = &noderec.MemoryObservation{RunID: "private-memory-run-2", Domains: []noderec.MemoryDomain{modelMemoryPool("ram", "system", "34359738368", "4294967296")}}
		f.mu.Unlock()
		replaced := f.waitMemory(t, "34359738368")
		second := assertModelMemoryProvenance(t, replaced.Estimate, "host", "34359738368", "4294967296")
		if first.SampleID == second.SampleID {
			t.Fatal("producer replacement retained old sample authority")
		}
		_, recalculated := f.review(t, request)
		assertModelMemoryProvenance(t, *recalculated.Estimate, "host", "34359738368", "4294967296")
		f.mu.Lock()
		for i := range f.memory.observation.Domains {
			f.memory.observation.Domains[i].Total.AgeMS = 6000
			f.memory.observation.Domains[i].Free.AgeMS = 6000
		}
		f.mu.Unlock()
		task1Eventually(t, "stale observation loses fit", 20*time.Second, func() bool {
			value := f.inspectMemory(t)
			return value.Estimate.CapacityFit == "unknown" && value.Estimate.FreeFit == "unknown"
		})
		f.mu.Lock()
		f.memory.observation = nil
		f.mu.Unlock()
		task1Eventually(t, "removed sample has no capacity", 20*time.Second, func() bool {
			value := f.inspectMemory(t)
			return value.Estimate.CapacityFit == "unknown" && !slices.ContainsFunc(value.Estimate.Domains, func(d modelconfig.MemoryDomain) bool { return d.CapacityBytes != "" || d.FreeBytes != "" })
		})
		f.noMetadataEffects(t, before)
	})
	t.Run("UMA is one shared pool", func(t *testing.T) {
		f := prepareModelSettingsProcessFixture(t)
		f.enableMemory(t, modelMemoryPool("uma", "shared", "17179869184", "8589934592", "registry:private-gpu"))
		f.startModelSettings(t)
		value := f.waitMemory(t, "17179869184")
		assertModelMemoryProvenance(t, value.Estimate, "shared", "17179869184", "8589934592")
		if len(value.Estimate.Domains) != 1 {
			t.Fatalf("UMA was counted in separate pools: %+v", value.Estimate.Domains)
		}
	})
	t.Run("two identical GPUs never gain ordinal placement authority", func(t *testing.T) {
		f := prepareModelSettingsProcessFixture(t)
		f.enableMemory(t, modelMemoryPool("ram", "system", "17179869184", "8589934592"),
			modelMemoryPool("gpu-a", "dedicated", "4294967296", "2147483648", "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "pci:0000:01:00.0"),
			modelMemoryPool("gpu-b", "dedicated", "8589934592", "4294967296", "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "pci:0000:02:00.0"))
		f.startModelSettings(t)
		f.waitMemory(t, "17179869184")
		request := f.request(t, "two-gpu-preview", nil)
		request.ModelChange.Load.Set.GPU = &modelconfig.GPU{Mode: "layers", Layers: modelSettingsInt(1)}
		check := func() {
			t.Helper()
			_, preview := f.review(t, request)
			if preview.Estimate == nil || preview.Estimate.CapacityFit != "unknown" || preview.Estimate.FreeFit != "unknown" {
				t.Fatal("ambiguous GPU placement acquired a fit verdict")
			}
			for _, load := range preview.Settings.ModelLoads {
				if load.Resolved != nil && load.Resolved.DeviceBinding != nil {
					t.Fatal("same-name GPU order manufactured a binding")
				}
			}
		}
		check()
		f.mu.Lock()
		slices.Reverse(f.memory.observation.Domains[1:])
		f.memory.observation.RunID = "reversed-gpu-order"
		f.mu.Unlock()
		task1Eventually(t, "reversed topology reached scanner", 20*time.Second, func() bool {
			return slices.ContainsFunc(f.inspectMemory(t).Estimate.Domains, func(d modelconfig.MemoryDomain) bool { return strings.Contains(d.SampleID, "reversed-gpu-order") })
		})
		check()
	})
}
