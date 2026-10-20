// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"encoding/json"
	"net/http"
	"reflect"
	"testing"

	"nvpair-shared/noderec"
)

func TestStrataContractRemoteUnsupported(t *testing.T) {
	strataContractPlatform(t)
	a, b := strataContractNew(t, true), strataContractNew(t, true)
	const uuidA, uuidB = "strata-contract-a", "strata-contract-b"
	const nodeB = "strata-contract-peer-b"
	certA := mintClusterIdentity(t, a.ClusterDir, uuidA)
	certB := mintClusterIdentity(t, b.ClusterDir, uuidB)
	writeActiveAdmission(t, a.ClusterDir)
	writeActiveAdmission(t, b.ClusterDir)
	writePin(t, a.ClusterDir, uuidB, certB)
	writePin(t, b.ClusterDir, uuidA, certA)
	writeMembers(t, a.ClusterDir, uuidA, uuidB)
	writeMembers(t, b.ClusterDir, uuidA, uuidB)
	managementGrantFixture(t, b.ClusterDir, uuidB, uuidA)
	// Keep the selected control port distinct from both not-yet-bound runtimes.
	for attempt := 0; attempt < 8; attempt++ {
		port := strataContractPort(t)
		if port != a.Port && port != b.Port {
			b.ControlPort = port
			break
		}
	}
	if b.ControlPort == 0 {
		t.Fatal("could not allocate a distinct private control port")
	}
	b.strataContractStartManager(t)
	b.strataContractStartRuntime(t)
	a.strataContractStartManager(t)
	a.strataContractStartRuntime(t)
	a.Manager.strataContractNotify(t, "discovery:nodes", noderec.GetNodesResult{
		Nodes: []noderec.DirectoryNode{{
			HostUUID: nodeB, Name: nodeB, IP: "127.0.0.1", ClusterUUID: uuidB, Trusted: true,
			Services: map[noderec.ServiceKey]noderec.ServiceStatus{
				noderec.ServiceEngineControl: {Port: b.ControlPort},
			},
		}},
	})
	// A successful peer read is a prerequisite: TLS, discovery and membership
	// failures must not satisfy either capability rejection below.
	var inventory struct {
		Engines []strataContractStatus `json:"engines"`
	}
	a.Manager.strataContractOK(t, "engine:remote-get-installed", map[string]string{"node": nodeB}, &inventory)
	var strata []strataContractStatus
	for _, status := range inventory.Engines {
		if status.Engine == "strata" {
			strata = append(strata, status)
		}
	}
	want := []strataContractStatus{{Engine: "strata", Installed: true, Running: true, Healthy: true, Port: b.Port}}
	if !reflect.DeepEqual(strata, want) {
		t.Fatalf("authenticated peer Strata inventory = %+v, want %+v", strata, want)
	}

	assertPreserved := func(t *testing.T, f *strataContractFixture, files map[string]string) {
		t.Helper()
		control := f.strataContractControl(t, 0)
		wantModels := map[string]string{
			"Org/Strata-Loaded:Q4_K_M":   "loaded",
			"Org/Strata-Unloaded:Q5_K_M": "unloaded",
			"Org/Strata-Loading:Q6_K":    "loading",
			"Org/Strata-NoStatus:Q8_0":   "",
		}
		if !reflect.DeepEqual(control.Models, wantModels) || control.NextLoadStatus != 0 {
			t.Fatalf("runtime state changed: models=%v nextLoadStatus=%d", control.Models, control.NextLoadStatus)
		}
		for _, request := range control.Requests {
			if request.Method != http.MethodGet || (request.Path != "/health" && request.Path != "/v1/models") || request.Model != "" {
				t.Fatalf("runtime received a request beyond health/inventory reads: %+v", request)
			}
		}
		// Remote rejection permits no private artifact changes or exclusions.
		if got := strataContractArtifacts(t, f.Root); !reflect.DeepEqual(got, files) {
			t.Fatalf("private fixture files changed; model copy/delete must create no files: got %v, want %v", got, files)
		}
		f.strataContractWaitStatus(t, true)
	}
	params := map[string]string{"node": nodeB, "engine": "strata", "model": "Org/Strata-Loaded:Q4_K_M"}

	t.Run("remote delete propagates peer capability rejection", func(t *testing.T) {
		filesA, filesB := strataContractArtifacts(t, a.Root), strataContractArtifacts(t, b.Root)
		response := a.Manager.strataContractCall(t, "engine:remote-delete-model", params)
		const diagnostic = `remote /v1/models/delete: HTTP 500: engine "strata" does not support the model delete operation`
		if response.Error == nil || response.Error.Code != -32000 || response.Error.Message != diagnostic {
			t.Fatalf("remote delete error = %+v, want code -32000 and %q", response.Error, diagnostic)
		}
		assertPreserved(t, a, filesA)
		assertPreserved(t, b, filesB)
	})

	t.Run("remote copy rejects destination without a model store", func(t *testing.T) {
		filesA, filesB := strataContractArtifacts(t, a.Root), strataContractArtifacts(t, b.Root)
		const diagnostic = `model copy is not supported for this engine: "strata"`
		type copyProgress struct {
			OpID    string `json:"opId"`
			Node    string `json:"node"`
			Engine  string `json:"engine"`
			Op      string `json:"op"`
			Stage   string `json:"stage"`
			Percent int    `json:"percent"`
			Message string `json:"message"`
		}
		response, notification := a.Manager.strataContractCallAndNotification(t,
			"engine:remote-copy-model", params, "engine:remote-progress", func(data json.RawMessage) bool {
				var progress copyProgress
				strataContractDecode(t, data, &progress)
				return progress.Node == nodeB && progress.Engine == "strata" && progress.Op == "copy" && progress.Stage == "error"
			})
		if response.Error == nil || response.Error.Code != -32000 || response.Error.Message != diagnostic {
			t.Fatalf("remote copy error = %+v, want code -32000 and %q", response.Error, diagnostic)
		}
		var progress copyProgress
		strataContractDecode(t, notification.Params, &progress)
		if progress.OpID == "" {
			t.Fatal("terminal copy progress has no operation ID")
		}
		want := copyProgress{OpID: progress.OpID, Node: nodeB, Engine: "strata", Op: "copy", Stage: "error", Percent: -1, Message: diagnostic}
		if progress != want {
			t.Fatalf("terminal copy progress = %+v, want %+v", progress, want)
		}
		// This rejects the destination store before any source-file transfer.
		assertPreserved(t, a, filesA)
		assertPreserved(t, b, filesB)
	})
}
