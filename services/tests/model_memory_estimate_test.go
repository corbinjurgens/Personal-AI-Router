// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
	"nvpair-shared/testbin"
)

func applyModelMemoryConfiguration(t *testing.T, f *modelSettingsProcessFixture, id string) modelconfig.InspectResult {
	t.Helper()
	f.waitMemory(t, "17179869184")
	request, _ := f.review(t, f.request(t, id, nil))
	f.rpcOK(t, "engine:apply-settings", request)
	f.settled(t, request.RequestID, "succeeded")
	value := inspectAppliedModelMemory(t, f)
	if len(value.ConfigurationKey) != 64 || !strings.HasPrefix(value.NativeBuildID, "sha256:") || !slices.Equal(value.ConfigurationPools, []string{"host"}) {
		t.Fatalf("applied CPU identity or pool unavailable: key=%q reason=%q build=%q pools=%v", value.ConfigurationKey, value.ConfigurationKeyReason, value.NativeBuildID, value.ConfigurationPools)
	}
	return value
}

func inspectAppliedModelMemory(t *testing.T, f *modelSettingsProcessFixture) modelconfig.InspectResult {
	t.Helper()
	var value modelconfig.InspectResult
	// Resume also refreshes model identities through the manager's shared
	// inspection slot. A busy reply has no applied key to reuse in a statement.
	task1Eventually(t, "applied model memory inspection available", 15*time.Second, func() bool {
		modelSettingsDecode(t, f.rpcOK(t, "engine:inspect-model", modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "medium-model", ProfileKey: "medium-model"}), &value)
		return value.Status != "busy"
	})
	return value
}

func modelMemoryStorePath(f *modelSettingsProcessFixture) string {
	return filepath.Join(jointAppDir(f.base), "engine-bin", "model-memory-profiles.json")
}

// Reuse the existing scanner directory fixture for the origin's peer locator.
// The origin broker/manager and target broker/manager remain real. Target
// memory still comes solely from its real scanner and local telemetry cache.
func modelMemoryOrigin(t *testing.T, base, target string, targetPort int) *task1Process {
	t.Helper()
	jointIsolateEngines(t, base, freePort(t))
	image, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	scanner := filepath.Join(testbin.Dir(t, "tests", "subscription-scanner", "subscription-scanner"), testbin.Exe("subscription-scanner"))
	jointCopyExecutable(t, image, scanner)
	scannerPort := freePort(t)
	ports := reserveFixtureServicePorts(t)
	args := append(ports.args(), "--scanner-path", scanner, "--engine-manager-path", engineMgrBin, "--engine-manager-control-host", "127.0.0.1", "--proxy-engines=", "--cluster-dir", filepath.Join(base, "cluster"))
	origin := startTask1ProcessWithEnv(t, base, brokerBin, []string{"NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + scanner, "NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(scannerPort)}, args...)
	task1Eventually(t, "origin broker ready", 20*time.Second, func() bool {
		origin.mu.Lock()
		defer origin.mu.Unlock()
		return slices.ContainsFunc(origin.frames, func(frame jsonrpc.Message) bool { return frame.Method == "app:ready" })
	})
	waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(scannerPort)), 10*time.Second)
	client := &http.Client{Timeout: 5 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	event := noderec.NodeEvent{Node: noderec.DirectoryNode{HostUUID: target, ClusterUUID: target, Name: "private-memory-target", IP: "127.0.0.1", Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: targetPort}}}}
	frame := jsonrpc.Message{JSONRPC: "2.0", Method: noderec.NotifyNodeDiscovered, Params: mustMarshal(t, event)}
	egressProcessHTTP(t, client, http.MethodPost, scannerPort, "/", mustMarshal(t, frame), http.StatusOK)
	last := "no reply"
	t.Cleanup(func() {
		if t.Failed() {
			t.Logf("origin target inspection: %s", last)
		}
	})
	task1Eventually(t, "origin broker dispatches target inspection", 20*time.Second, func() bool {
		reply := origin.callResponse(t, "engine:inspect-model", modelconfig.InspectRequest{Engine: "llamacpp", NodeID: target, ModelID: "medium-model", ProfileKey: "medium-model"})
		if reply.Error != nil {
			last = fmt.Sprintf("code=%d message=%s", reply.Error.Code, reply.Error.Message)
			return false
		}
		var result modelconfig.InspectResult
		modelSettingsDecode(t, reply.Result, &result)
		last = fmt.Sprintf("node=%s key=%s reason=%s", result.NodeID, result.ConfigurationKey, result.ConfigurationKeyReason)
		return result.NodeID == target && result.ConfigurationKey != ""
	})
	return origin
}

func assertModelMemoryStatement(t *testing.T, f *modelSettingsProcessFixture, value modelconfig.InspectResult, amount string) {
	t.Helper()
	index := slices.IndexFunc(value.MemoryProfiles, func(p modelconfig.MemoryProfile) bool {
		return p.Current && p.ConfigurationKey == value.ConfigurationKey
	})
	if index < 0 {
		t.Fatalf("statement history=%+v", value.MemoryProfiles)
	}
	p := value.MemoryProfiles[index]
	if !p.Current || p.ConfigurationKey != value.ConfigurationKey || p.HostBytes != amount || p.NativeBuildID != value.NativeBuildID || p.NativeBuildID == "" || value.Estimate.EstimateSource == nil || value.Estimate.EstimateSource.Kind != "user-stated" {
		t.Fatalf("statement did not bind target configuration/build: %+v", p)
	}
	var records []struct {
		Engine  string                    `json:"engine"`
		ModelID string                    `json:"modelId"`
		Profile modelconfig.MemoryProfile `json:"profile"`
	}
	modelSettingsDecode(t, readLiveDurableFile(t, modelMemoryStorePath(f)), &records)
	stored := -1
	for i, record := range records {
		if record.Profile.ConfigurationKey == p.ConfigurationKey {
			stored = i
			break
		}
	}
	if stored < 0 || records[stored].Engine != "llamacpp" || records[stored].ModelID != "medium-model" || records[stored].Profile.Current || records[stored].Profile.HostBytes != amount || records[stored].Profile.NativeBuildID != p.NativeBuildID {
		t.Fatal("target store differs from the stated-memory reply")
	}
}

func assertModelMemoryRedacted(t *testing.T, f *modelSettingsProcessFixture, value modelconfig.InspectResult) {
	t.Helper()
	data := mustMarshal(t, value)
	for _, forbidden := range [][]byte{[]byte(f.root), []byte(strings.ReplaceAll(f.root, `\`, `\\`)), []byte(f.preset), modelSettingsGGUF("blk.0.weight"), []byte("private-prompt-must-not-leak")} {
		if bytes.Contains(data, forbidden) {
			t.Fatal("inspection reply leaked paths, content or a prompt")
		}
	}
	// Inspect the real combined worker log, not a test-created projection.
	if strings.Contains(f.p.output.String(), "private-prompt-must-not-leak") || strings.Contains(f.p.output.String(), f.root) || strings.Contains(f.p.output.String(), f.preset) || strings.Contains(f.p.output.String(), string(modelSettingsGGUF("blk.0.weight"))) {
		t.Fatal("worker log leaked prompt or raw fixture content")
	}
}

func TestModelMemoryEstimateProcess(t *testing.T) {
	t.Run("ranged basis and local and direct stated figures stay target owned", func(t *testing.T) {
		f := prepareModelSettingsProcessFixture(t)
		f.pinned(t)
		modelSettingsWrite(t, filepath.Join(f.root, "medium.gguf"), modelSettingsGGUF("blk.0.weight", [2]string{"tokenizer.chat_template", "private-prompt-must-not-leak"}))
		f.enableMemory(t, modelMemoryPool("ram", "system", "17179869184", "8589934592"))
		f.startModelSettings(t)
		value := applyModelMemoryConfiguration(t, f, "estimate-process-config")
		priced := 0
		for _, row := range value.Estimate.Components {
			if row.Bytes == "" {
				continue
			}
			low, lowErr := strconv.ParseUint(row.LowBytes, 10, 64)
			expected, expectedErr := strconv.ParseUint(row.Bytes, 10, 64)
			high, highErr := strconv.ParseUint(row.HighBytes, 10, 64)
			if lowErr != nil || expectedErr != nil || highErr != nil || low > expected || expected > high || row.Basis == "" {
				t.Fatalf("invalid ranged row: %+v", row)
			}
			priced++
		}
		if priced == 0 {
			t.Fatal("inspection returned no priced rows")
		}
		set := modelconfig.SetStatedMemoryRequest{Engine: "llamacpp", ModelID: "medium-model", ConfigurationKey: value.ConfigurationKey, HostBytes: "1073741824"}
		modelSettingsDecode(t, f.rpcOK(t, "engine:set-stated-memory", set), &value)
		assertModelMemoryStatement(t, f, value, set.HostBytes)
		assertModelMemoryRedacted(t, f, value)
		// Peer statements retain the existing target Pause and management fences.
		before := readLiveDurableFile(t, modelMemoryStorePath(f))
		f.peer(t, "set-stated-memory", set, http.StatusConflict)
		if !bytes.Equal(before, readLiveDurableFile(t, modelMemoryStorePath(f))) {
			t.Fatal("paused peer statement changed target history")
		}
		f.rpcOK(t, "node:set-availability", map[string]string{"state": "available"})
		task1Eventually(t, "private engine available", 30*time.Second, func() bool { return f.snapshot(t).Running })
		value = inspectAppliedModelMemory(t, f)
		set.ConfigurationKey = value.ConfigurationKey
		origin := modelMemoryOrigin(t, filepath.Dir(f.callerCluster), "model-settings-target", f.ports.engineControl)
		set.HostBytes = "2147483648"
		set.NodeID = "model-settings-target"
		reply := origin.callResponse(t, "engine:set-stated-memory", set)
		if reply.Error != nil {
			t.Fatalf("origin broker direct statement: %+v", reply.Error)
		}
		modelSettingsDecode(t, reply.Result, &value)
		set.NodeID = ""
		assertModelMemoryStatement(t, f, value, set.HostBytes)
		if _, err := os.Stat(filepath.Join(jointAppDir(filepath.Dir(f.callerCluster)), "engine-bin", "model-memory-profiles.json")); !os.IsNotExist(err) {
			t.Fatalf("origin acquired a target statement store: %v", err)
		}
		// A changed applied configuration retains history but never reuses it as
		// current, measured evidence or a warning lift.
		request := f.request(t, "estimate-process-new-context", nil)
		request.ModelChange.Load.Set.Context = &modelconfig.Context{Mode: "fixed", Tokens: 4096}
		request, _ = f.review(t, request)
		f.rpcOK(t, "engine:apply-settings", request)
		f.settled(t, request.RequestID, "succeeded")
		changed := inspectAppliedModelMemory(t, f)
		old := slices.IndexFunc(changed.MemoryProfiles, func(p modelconfig.MemoryProfile) bool { return p.ConfigurationKey == set.ConfigurationKey })
		if changed.ConfigurationKey == set.ConfigurationKey || old < 0 || changed.MemoryProfiles[old].Current || changed.Estimate.EstimateSource == nil || changed.Estimate.EstimateSource.Kind != "pair-estimate" {
			t.Fatal("configuration change reused an earlier statement as current evidence")
		}
		before = readLiveDurableFile(t, modelMemoryStorePath(f))
		if reply := f.rpc(t, "engine:set-stated-memory", set); reply.Error == nil || reply.Error.Code != -32010 {
			t.Fatal("stale configuration key accepted a new statement")
		}
		if !bytes.Equal(before, readLiveDurableFile(t, modelMemoryStorePath(f))) {
			t.Fatal("stale-key refusal changed history")
		}
		assertModelMemoryRedacted(t, f, changed)
		f.rpcOK(t, "node:set-availability", map[string]string{"state": "paused", "pauseMode": "immediate"})
	})
	t.Run("reverse peer writes only the target store and honors revoked grant", func(t *testing.T) {
		nodes, blocked := newTask1Nodes(t)
		f := prepareModelSettingsProcessFixtureAt(t, nodes[1].base)
		modelSettingsWrite(t, filepath.Join(f.root, "medium.gguf"), modelSettingsGGUF("blk.0.weight", [2]string{"tokenizer.chat_template", "private-prompt-must-not-leak"}))
		f.enableMemory(t, modelMemoryPool("ram", "system", "17179869184", "8589934592"))
		f.startModelSettings(t)
		value := applyModelMemoryConfiguration(t, f, "reverse-estimate-config")
		f.rpcOK(t, "node:set-availability", map[string]string{"state": "available"})
		task1Eventually(t, "private reverse target available", 30*time.Second, func() bool { return f.snapshot(t).Running })
		value = inspectAppliedModelMemory(t, f)
		observer := modelMemoryReversePeer(t, f, nodes, blocked)
		origin := modelMemoryOrigin(t, nodes[0].base, nodes[1].uuid, blocked)
		set := modelconfig.SetStatedMemoryRequest{Engine: "llamacpp", NodeID: nodes[1].uuid, ModelID: "medium-model", ConfigurationKey: value.ConfigurationKey, HostBytes: "3221225472"}
		reply := origin.callResponse(t, "engine:set-stated-memory", set)
		if reply.Error != nil {
			t.Fatalf("origin broker reverse statement: %+v", reply.Error)
		}
		modelSettingsDecode(t, reply.Result, &value)
		assertModelMemoryStatement(t, f, value, set.HostBytes)
		if observer.count(nodes[0].uuid, http.MethodPost, "/v1/engine-settings/set-stated-memory") == 0 {
			t.Fatal("statement bypassed real reverse transport")
		}
		if _, err := os.Stat(filepath.Join(jointAppDir(nodes[0].base), "engine-bin", "model-memory-profiles.json")); !os.IsNotExist(err) {
			t.Fatalf("reverse origin acquired a target statement store: %v", err)
		}
		before := readLiveDurableFile(t, modelMemoryStorePath(f))
		managementSet(t, nodes[1].cm, nodes[0].uuid, false)
		if reply := origin.callResponse(t, "engine:set-stated-memory", set); reply.Error == nil || reply.Error.Code != -32012 {
			t.Fatalf("origin broker lost revoked target grant: %+v", reply.Error)
		}
		if !bytes.Equal(before, readLiveDurableFile(t, modelMemoryStorePath(f))) {
			t.Fatal("revoked reverse statement changed target store")
		}
		assertModelMemoryRedacted(t, f, value)
		f.rpcOK(t, "node:set-availability", map[string]string{"state": "paused", "pauseMode": "immediate"})
	})
	t.Run("GPU-selection keys stay unpriced on a two-GPU target", func(t *testing.T) {
		f := prepareModelSettingsProcessFixture(t)
		f.enableMemory(t, modelMemoryPool("ram", "system", "17179869184", "8589934592"),
			modelMemoryPool("gpu-a", "dedicated", "4294967296", "2147483648", "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "pci:0000:01:00.0"),
			modelMemoryPool("gpu-b", "dedicated", "8589934592", "4294967296", "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "pci:0000:02:00.0"))
		f.startModelSettings(t)
		f.waitMemory(t, "17179869184")
		for _, option := range []struct{ key, value string }{{"device", "CUDA1"}, {"split-mode", "layer"}, {"tensor-split", "1,1"}, {"main-gpu", "1"}} {
			t.Run(option.key, func(t *testing.T) {
				request := f.request(t, "unpriced-"+option.key, nil)
				request.ModelChange.Load.Set.ExtraOptions = map[string]string{option.key: option.value}
				request, preview := f.review(t, request)
				index := slices.IndexFunc(preview.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == "medium-model" })
				if index < 0 || preview.Settings.ModelLoads[index].Requested.ExtraOptions[option.key] != option.value {
					t.Fatal("accepted GPU-selection option was discarded")
				}
				if preview.Estimate == nil || preview.Estimate.CapacityFit != "unknown" || preview.Estimate.FreeFit != "unknown" {
					t.Fatal("GPU-selection placement gained a priced comparison")
				}
				f.rpcOK(t, "engine:apply-settings", request)
				f.settled(t, request.RequestID, "succeeded")
				value := inspectAppliedModelMemory(t, f)
				if value.ConfigurationKey != "" || value.ConfigurationKeyReason != "gpu-selection" || value.ExecutionKey == "" || len(value.ConfigurationPools) != 0 || value.Estimate.CapacityFit != "unknown" || value.Estimate.FreeFit != "unknown" || value.Estimate.EstimateSource != nil && value.Estimate.EstimateSource.Kind == "user-stated" {
					t.Fatal("GPU-selection inspection exposed finite statement or placement authority")
				}
				if value.LoadObservation != nil && (value.LoadObservation.Current || value.LoadObservation.SampleID != "") {
					t.Fatal("GPU-selection placement gained a buffer observation")
				}
			})
		}
	})
}
