// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"reflect"
	"slices"
	"testing"
	"time"

	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func TestModelDiagnosticHardwareBindsPhysicalIDsRatherThanArrayOrder(t *testing.T) {
	b, _ := newPolicyBroker(t)
	identity := diagnosticIdentityFixture()
	scanner := seedDiagnosticHardware(t, b, identity, time.Now())
	scanner.mu.Lock()
	base := scanner.answer
	scanner.mu.Unlock()
	scanner.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		body, err := base(method, raw)
		value, ok := body.(map[string]any)
		if !ok {
			return nil, "node-info fixture shape changed"
		}
		gpus, ok := value["GPUs"].([]noderec.GPUInfo)
		if !ok {
			return nil, "GPU fixture shape changed"
		}
		gpus = slices.Clone(gpus)
		slices.Reverse(gpus)
		value["GPUs"] = gpus
		return value, err
	})
	hardware, err := b.diagnosticTargetHardware(context.Background(), b.modelTargetMemory())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(hardware, diagnosticIdentityHardware(identity)) {
		t.Fatalf("hardware joined by order: %+v", hardware)
	}
}

func TestModelDiagnosticHardwareRefusesUnknownOrChangedAssociation(t *testing.T) {
	for _, scenario := range []string{"host", "producer", "missing physical ID", "omitted unused GPU", "duplicate GPU", "capacity", "stale full sample"} {
		t.Run(scenario, func(t *testing.T) {
			b, _ := newPolicyBroker(t)
			scanner := seedDiagnosticHardware(t, b, diagnosticIdentityFixture(), time.Now())
			scanner.mu.Lock()
			base := scanner.answer
			scanner.mu.Unlock()
			scanner.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				body, err := base(method, raw)
				value, ok := body.(map[string]any)
				if !ok {
					return nil, "node-info fixture shape changed"
				}
				gpus, ok := value["GPUs"].([]noderec.GPUInfo)
				if !ok {
					return nil, "GPU fixture shape changed"
				}
				gpus = slices.Clone(gpus)
				memory, ok := value["memoryObservation"].(*noderec.MemoryObservation)
				if !ok {
					return nil, "memory fixture shape changed"
				}
				memory = memory.Clone()
				switch scenario {
				case "host":
					value["hostUuid"] = "other"
				case "producer":
					memory.RunID = "replacement"
				case "missing physical ID":
					gpus[1].PhysicalID = ""
				case "omitted unused GPU":
					gpus = gpus[:1]
				case "duplicate GPU":
					gpus[1].PhysicalID = gpus[0].PhysicalID
				case "capacity":
					gpus[1].VramBytes++
				case "stale full sample":
					memory.Domains[0].Total.SampledAt = time.Now().Add(-10 * time.Second)
				}
				value["GPUs"], value["memoryObservation"] = gpus, memory
				return value, err
			})
			if _, err := b.diagnosticTargetHardware(context.Background(), b.modelTargetMemory()); err == nil {
				t.Fatal("unknown hardware association accepted")
			}
		})
	}
}

func TestModelDiagnosticResidentReloadConsentUsesPolicyAndPreservesPause(t *testing.T) {
	b, _ := newPolicyBroker(t)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.Profiles = []nodepolicy.Profile{{Key: "resident-profile", Engine: "llamacpp", Model: "resident.gguf"}, {Key: "transient-profile", Engine: "llamacpp", Model: "transient.gguf"}}
	r.current.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "resident.gguf"}}
	r.mu.Unlock()
	want := []modelconfig.DiagnosticModelRef{{Engine: "llamacpp", ModelID: "resident.gguf", ProfileKey: "resident-profile"}}
	if got := b.diagnosticResidentReloads(); !reflect.DeepEqual(got, want) {
		t.Fatalf("resident consent=%+v", got)
	}
	r.mu.Lock()
	r.current.Availability = nodepolicy.Paused
	r.current.PauseMode = nodepolicy.PauseConfigured
	r.mu.Unlock()
	if got := b.diagnosticResidentReloads(); len(got) != 0 {
		t.Fatalf("paused resident falsely promised reload: %+v", got)
	}
}

func TestModelDiagnosticSharedHardwareBindsGPUCapacityToHostRAM(t *testing.T) {
	b, _ := newPolicyBroker(t)
	now := time.Now()
	memory := diagnosticMemoryFixture(now, 9000)
	memory.Domains[0].ID, memory.Domains[0].Kind, memory.Domains[0].DeviceIDs = "unified", "shared", []string{"metal-device"}
	seedDiagnosticTelemetry(t, b, now, memory)
	peer, scanner := newFakeWorkerPipe(t)
	owner := &scannerProcess{peer: peer}
	b.workersMu.Lock()
	b.scanner, b.scannerSource = owner, owner
	b.workersMu.Unlock()
	scanner.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		return map[string]any{"hostUuid": b.nodeID, "GPUs": []noderec.GPUInfo{{PhysicalID: "metal-device", Name: "Unified GPU"}}, "cpu": noderec.CPUInfo{Name: "fixture CPU"}, "memory": noderec.MemoryInfo{TotalBytes: 10000}, "memoryObservation": memory, "telemetryValid": true, "msSince": 0}, ""
	})
	hardware, err := b.diagnosticTargetHardware(context.Background(), b.modelTargetMemory())
	if err != nil {
		t.Fatal(err)
	}
	if len(hardware.GPUs) != 1 || hardware.GPUs[0].TotalMemoryBytes != "10000" || hardware.TotalRAMBytes != "10000" {
		t.Fatalf("shared hardware=%+v", hardware)
	}
}

func TestModelDiagnosticDedicatedHardwareAcceptsUUIDAndPCIAliases(t *testing.T) {
	b, _ := newPolicyBroker(t)
	identity := diagnosticIdentityFixture()
	scanner := seedDiagnosticHardware(t, b, identity, time.Now())
	scanner.mu.Lock()
	base := scanner.answer
	scanner.mu.Unlock()
	body, errText := base("discovery:get-node-info", settingsJSON(nodeInfoRPCParams{HostUUID: b.nodeID}))
	if errText != "" {
		t.Fatal(errText)
	}
	value, ok := body.(map[string]any)
	if !ok {
		t.Fatal("node-info fixture shape changed")
	}
	memory, ok := value["memoryObservation"].(*noderec.MemoryObservation)
	if !ok {
		t.Fatal("memory fixture shape changed")
	}
	memory = memory.Clone()
	memory.Domains[1].DeviceIDs = append(memory.Domains[1].DeviceIDs, "0000:01:00.0")
	memory.Domains[2].DeviceIDs = append(memory.Domains[2].DeviceIDs, "0000:02:00.0")
	seedDiagnosticTelemetry(t, b, time.Now(), memory)
	value["memoryObservation"] = memory
	// Missing utilization must not invalidate independently current memory.
	value["telemetryValid"], value["msSince"] = false, int64(60000)
	scanner.setAnswer(func(string, json.RawMessage) (any, string) { return value, "" })
	hardware, err := b.diagnosticTargetHardware(context.Background(), b.modelTargetMemory())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(hardware, diagnosticIdentityHardware(identity)) {
		t.Fatalf("physical aliases changed hardware: %+v", hardware)
	}
}
