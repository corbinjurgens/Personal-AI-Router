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

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
)

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
		p, err := modelconfig.DecodeInspectRequest(msg.Params)
		if err != nil {
			done <- err
			return
		}
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
