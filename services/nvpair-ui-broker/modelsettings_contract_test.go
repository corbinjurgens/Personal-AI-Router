// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

func TestModelSettingsContractInspectionInjectsOnlyTargetSelfMemory(t *testing.T) {
	b := &Broker{nodeID: "target", telemetry: newTelemetryCache(), scannerSource: &scannerProcess{}}
	b.telemetry.clock = func() (uint64, error) { return 1000000000, nil }
	now := time.Now().UTC()
	fact := noderec.MemoryFact{Valid: true, Bytes: "8589934592", Source: "synthetic-target", SampledAt: now}
	sample := func(host, bytes string) noderec.NodeTelemetry {
		f := fact
		f.Bytes = bytes
		return noderec.NodeTelemetry{HostUUID: host, Memory: &noderec.MemoryObservation{RunID: "run-" + host, Domains: []noderec.MemoryDomain{{ID: "ram", Kind: "system", Total: f, Free: f}}}, MemoryHop: noderec.NewMemoryHop("scanner", b.telemetry.clock), MemoryRequest: &noderec.MemoryRequestContext{Generation: "1", RequestBootNS: "1000000000", VerifiedSelfLoopback: true}}
	}
	b.telemetry.Upsert(sourceScanner, sample("origin", "999999999999"), now)
	b.telemetry.Upsert(sourceScanner, sample("target", fact.Bytes), now)
	worker, codec := newTestRPCWorkerPipe(t)
	b.setEngineMgr(worker)
	done := make(chan error, 1)
	go func() {
		msg, err := codec.Read()
		if err != nil {
			done <- err
			return
		}
		input, err := settings.DecodeWorkerRequest[modelconfig.InspectRequest](msg.Params)
		if err != nil {
			done <- err
			return
		}
		if input.Memory == nil || input.Memory.HostID != "target" || input.Memory.Observation.RunID != "run-target" || input.Memory.Observation.Domains[0].Free.Bytes != fact.Bytes || input.Request.NodeID != "" {
			done <- fmt.Errorf("wrong target-owned evidence: %+v", input)
			return
		}
		done <- codec.Respond(msg.ID, modelconfig.InspectResult{Engine: input.Request.Engine, ModelID: input.Request.ModelID, Status: "unavailable", Estimate: modelconfig.Estimate{Status: "unknown"}})
	}()
	result, err := b.inspectModelSettings(context.Background(), modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "fixture/model"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if result.NodeID != "target" {
		t.Fatalf("target readback=%+v", result)
	}
}

func TestModelSettingsContractInspectionDoesNotInitializeJournal(t *testing.T) {
	b := &Broker{nodeID: "target", clusterDir: filepath.Join(t.TempDir(), "cluster")}
	worker, codec := newTestRPCWorkerPipe(t)
	b.setEngineMgr(worker)
	done := make(chan error, 1)
	go func() {
		msg, err := codec.Read()
		if err != nil {
			done <- err
			return
		}
		envelope, err := settings.DecodeWorkerRequest[modelconfig.InspectRequest](msg.Params)
		if err != nil {
			done <- err
			return
		}
		p := envelope.Request
		if msg.Method != "engine:inspect-model" || p.NodeID != "" || p.ModelID != "fixture/model" {
			done <- fmt.Errorf("unexpected inspection request: %s %+v", msg.Method, p)
			return
		}
		result := modelconfig.InspectResult{Engine: p.Engine, ModelID: p.ModelID, Status: "unavailable", Estimate: modelconfig.Estimate{Status: "unavailable"}}
		done <- codec.Respond(msg.ID, result)
	}()
	result, err := b.inspectModelSettings(context.Background(), modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "fixture/model"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if result.NodeID != "target" || result.Status != "unavailable" || result.Estimate.Status != "unavailable" {
		t.Fatalf("invalid inspection result: %+v", result)
	}
	if b.engineSettingsLoaded || b.engineSettings != nil {
		t.Fatal("inspection initialized settings journal")
	}
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := modelconfig.DecodeInspectResult(data); err != nil {
		t.Fatal(err)
	}
}

func TestModelSettingsContractWorkerInspectionRefusalsKeepTheirCodes(t *testing.T) {
	for _, code := range []int{-32602, -32010, -32011, -32012, -32014} {
		t.Run(fmt.Sprint(code), func(t *testing.T) {
			b := &Broker{nodeID: "target"}
			worker, codec := newTestRPCWorkerPipe(t)
			b.setEngineMgr(worker)
			done := make(chan error, 1)
			go func() {
				msg, err := codec.Read()
				if err != nil {
					done <- err
					return
				}
				done <- codec.RespondError(msg.ID, code, "target refusal")
			}()
			_, err := b.inspectModelSettings(context.Background(), modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "fixture/model"}, "")
			if err == nil || err.Error() != "target refusal" || engineSettingsErrorCode(err) != code {
				t.Fatalf("worker code lost: error=%v code=%d", err, engineSettingsErrorCode(err))
			}
			if err := <-done; err != nil {
				t.Fatal(err)
			}
			if b.engineSettingsLoaded {
				t.Fatal("refused inspection initialized journal")
			}
		})
	}
}

func TestModelSettingsContractUnreviewedChangeHasNoEffects(t *testing.T) {
	for _, method := range []string{"preview", "apply"} {
		t.Run(method, func(t *testing.T) {
			b := &Broker{clusterDir: filepath.Join(t.TempDir(), "cluster")}
			p := settings.Request{Engine: "llamacpp", RequestID: "fixture-request", ModelChange: &modelconfig.ModelChange{ModelID: "fixture/model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Clear: []string{"context"}}}}
			_, err := b.dispatchEngineSettings(context.Background(), method, p, "")
			if err == nil || policyErrorCode(err) != -32602 {
				t.Fatalf("unreviewed change: error=%v code=%d", err, policyErrorCode(err))
			}
			if b.engineSettingsLoaded || b.engineSettings != nil {
				t.Fatal("refused change initialized settings journal")
			}
			path, err := b.engineSettingsPath()
			if err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(path); !os.IsNotExist(err) {
				t.Fatalf("journal exists or cannot be inspected: %v", err)
			}
		})
	}
}

func TestModelSettingsContractRejectsMalformedInspectionBeforeWorker(t *testing.T) {
	b := &Broker{}
	_, err := b.inspectModelSettings(context.Background(), modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "bad\nmodel"}, "")
	if err == nil || policyErrorCode(err) != -32602 {
		t.Fatalf("invalid inspection: %v", err)
	}
	if b.engineSettingsLoaded {
		t.Fatal("inspection initialized settings journal")
	}
}
