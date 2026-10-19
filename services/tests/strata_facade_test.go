// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

// Relay real manager state explicitly: this fixture starts no broker or scanner.
func strataContractU4Front(t *testing.T) (*strataContractFixture, *strataContractProxy) {
	t.Helper()
	fixture := strataContractNew(t, true)
	fixture.strataContractStartManager(t)
	descriptor := fixture.strataContractDescriptor(t)
	fixture.strataContractStartRuntime(t)
	var models strataContractModels
	fixture.Manager.strataContractOK(t, "engine:models", nil, &models)
	wantInventory := []string{
		"Org/Strata-Loaded:Q4_K_M", "Org/Strata-Loading:Q6_K",
		"Org/Strata-NoStatus:Q8_0", "Org/Strata-Unloaded:Q5_K_M",
	}
	inventory := slices.Clone(models.ModelsByEngine["strata"])
	slices.Sort(inventory)
	if !slices.Equal(inventory, wantInventory) || !slices.Equal(models.LoadedByEngine["strata"], []string{"Org/Strata-Loaded:Q4_K_M"}) {
		t.Fatalf("manager inventory/residency = %v / %v", inventory, models.LoadedByEngine["strata"])
	}

	proxy := strataContractStartProxy(t)
	proxy.strataContractEnable(t, descriptor)
	data, err := os.ReadFile(filepath.Join(proxy.AppDir, "strata-proxy-port.json"))
	if err != nil {
		t.Fatalf("read private facade port: %v", err)
	}
	var persisted struct {
		Port int `json:"port"`
	}
	strataContractDecode(t, data, &persisted)
	if persisted.Port != proxy.Port {
		t.Fatalf("persisted facade port = %d, want %d", persisted.Port, proxy.Port)
	}

	proxy.RPC.strataContractOK(t, "e-strata:node/set-local-backend", map[string]any{
		"engine": "strata", "host": "127.0.0.1", "port": fixture.Port, "healthy": true,
	}, nil)
	proxy.RPC.strataContractNotify(t, "e-strata:discovery:nodes", noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{{
		HostUUID: "strata-contract-self", Name: "strata-contract-self", IP: "127.0.0.1",
		Services:       map[noderec.ServiceKey]noderec.ServiceStatus{"e-strata": {Port: proxy.Port}},
		ModelsByEngine: models.ModelsByEngine,
		LoadedByEngine: models.LoadedByEngine,
	}}})
	// This response fences the preceding discovery notification before HTTP.
	var nodes struct {
		Nodes []struct {
			ID     string   `json:"id"`
			Port   int      `json:"port"`
			Models []string `json:"models"`
		} `json:"nodes"`
	}
	proxy.RPC.strataContractOK(t, "e-strata:nodes/list", nil, &nodes)
	if len(nodes.Nodes) != 1 {
		t.Fatalf("facade nodes = %d, want one local node", len(nodes.Nodes))
	}
	node := nodes.Nodes[0]
	slices.Sort(node.Models)
	if node.ID != "strata-contract-self" || node.Port != proxy.Port || !slices.Equal(node.Models, wantInventory) {
		t.Fatalf("facade local node = %+v", node)
	}
	proxy.RPC.strataContractOK(t, nodepolicy.MethodSetResidency, nodepolicy.SetResidencyParams{LoadedByEngine: models.LoadedByEngine}, nil)
	return fixture, proxy
}

func strataContractU4Chat(t *testing.T, fixture *strataContractFixture, proxy *strataContractProxy) {
	t.Helper()
	before := strataContractCount(fixture.strataContractControl(t, 0), http.MethodPost, "/v1/chat/completions", strataContractLoadedModel)
	status, header, body := strataContractHTTP(t, http.MethodPost, proxy.Port, "/v1/chat/completions", map[string]any{
		"model": strataContractLoadedModel, "messages": []any{}, "stream": false,
	})
	mediaType, _, err := mime.ParseMediaType(header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" || status != http.StatusOK {
		t.Fatalf("JSON chat status=%d media type=%q parse error=%v", status, mediaType, err)
	}
	var got, want map[string]any
	strataContractDecode(t, body, &got)
	strataContractDecode(t, []byte(`{
		"id":"strata-contract","object":"chat.completion","model":"Org/Strata-Loaded:Q4_K_M",
		"choices":[{"index":0,"message":{"role":"assistant","content":"fixture reply"},"finish_reason":"stop"}]
	}`), &want)
	if !reflect.DeepEqual(got, want) {
		t.Fatal("JSON chat did not preserve the exact model and canned completion")
	}
	after := strataContractCount(fixture.strataContractControl(t, 0), http.MethodPost, "/v1/chat/completions", "Org/Strata-Loaded:Q4_K_M")
	if after != before+1 {
		t.Fatalf("runtime exact-model chat count = %d, want %d", after, before+1)
	}
}

func TestStrataContractFacade(t *testing.T) {
	strataContractPlatform(t)
	fixture, proxy := strataContractU4Front(t)

	t.Run("serves exact model inventory", func(t *testing.T) {
		before := strataContractCount(fixture.strataContractControl(t, 0), http.MethodGet, "/v1/models", "")
		status, _, body := strataContractHTTP(t, http.MethodGet, proxy.Port, "/v1/models", nil)
		var list struct {
			Object string `json:"object"`
			Data   []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		strataContractDecode(t, body, &list)
		var ids []string
		for _, row := range list.Data {
			ids = append(ids, row.ID)
		}
		slices.Sort(ids)
		want := []string{"Org/Strata-Loaded:Q4_K_M", "Org/Strata-Loading:Q6_K", "Org/Strata-NoStatus:Q8_0", "Org/Strata-Unloaded:Q5_K_M"}
		if status != http.StatusOK || list.Object != "list" || !slices.Equal(ids, want) {
			t.Fatalf("model list status=%d object=%q ids=%v", status, list.Object, ids)
		}
		// Background manager reads may also occur; require the facade's read.
		if after := strataContractCount(fixture.strataContractControl(t, 0), http.MethodGet, "/v1/models", ""); after <= before {
			t.Fatal("model list did not reach the owned runtime")
		}
	})
	t.Run("routes exact model JSON chat", func(t *testing.T) {
		strataContractU4Chat(t, fixture, proxy)
	})
	t.Run("routes exact model SSE through terminal marker", func(t *testing.T) {
		before := strataContractCount(fixture.strataContractControl(t, 0), http.MethodPost, "/v1/chat/completions", strataContractLoadedModel)
		status, header, body := strataContractHTTP(t, http.MethodPost, proxy.Port, "/v1/chat/completions", map[string]any{
			"model": strataContractLoadedModel, "messages": []any{}, "stream": true,
		})
		mediaType, _, err := mime.ParseMediaType(header.Get("Content-Type"))
		if err != nil || mediaType != "text/event-stream" || status != http.StatusOK {
			t.Fatalf("SSE status=%d media type=%q parse error=%v", status, mediaType, err)
		}
		var chunks []map[string]any
		done := false
		scanner := bufio.NewScanner(bytes.NewReader(body))
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}
			if done || !strings.HasPrefix(line, "data: ") {
				t.Fatal("unexpected SSE field or data after terminal marker")
			}
			payload := strings.TrimPrefix(line, "data: ")
			if payload == "[DONE]" {
				done = true
				continue
			}
			var chunk map[string]any
			strataContractDecode(t, []byte(payload), &chunk)
			chunks = append(chunks, chunk)
		}
		if err := scanner.Err(); err != nil {
			t.Fatalf("read SSE: %v", err)
		}
		var want []map[string]any
		strataContractDecode(t, []byte(`[
			{"id":"strata-contract","object":"chat.completion.chunk","model":"Org/Strata-Loaded:Q4_K_M","choices":[{"index":0,"delta":{"role":"assistant","content":"fixture reply"},"finish_reason":null}]},
			{"id":"strata-contract","object":"chat.completion.chunk","model":"Org/Strata-Loaded:Q4_K_M","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
		]`), &want)
		if !done || !reflect.DeepEqual(chunks, want) {
			t.Fatal("SSE did not preserve content, exact model, stop chunk and terminal marker")
		}
		if after := strataContractCount(fixture.strataContractControl(t, 0), http.MethodPost, "/v1/chat/completions", "Org/Strata-Loaded:Q4_K_M"); after != before+1 {
			t.Fatalf("runtime SSE chat count = %d, want %d", after, before+1)
		}
	})

	countChats := func(control strataContractControl) int {
		count := 0
		for _, request := range control.Requests {
			if request.Method == http.MethodPost && request.Path == "/v1/chat/completions" {
				count += request.Count
			}
		}
		return count
	}
	reject := func(name, model string) {
		t.Run(name, func(t *testing.T) {
			before := countChats(fixture.strataContractControl(t, 0))
			status, _, _ := strataContractHTTP(t, http.MethodPost, proxy.Port, "/v1/chat/completions", map[string]any{
				"model": model, "messages": []any{},
			})
			if status != http.StatusBadGateway {
				t.Fatalf("inexact model status = %d, want 502", status)
			}
			if after := countChats(fixture.strataContractControl(t, 0)); after != before {
				t.Fatalf("rejected model reached runtime: chat count %d -> %d", before, after)
			}
		})
	}
	reject("rejects unadvertised model ID", "Org/Strata-Absent:Q4_K_M")
	reject("rejects different model case", "org/strata-loaded:q4_k_m")
	reject("rejects appended latest tag", "Org/Strata-Loaded:Q4_K_M:latest")

	passThrough := func(name, path string) {
		t.Run(name, func(t *testing.T) {
			// An unadvertised model also proves this path bypasses chat routing.
			const model = "Org/Strata-Unadvertised:Q4_K_M"
			before := strataContractCount(fixture.strataContractControl(t, 0), http.MethodPost, path, model)
			status, _, body := strataContractHTTP(t, http.MethodPost, proxy.Port, path, map[string]any{"model": model})
			var response map[string]string
			strataContractDecode(t, body, &response)
			if status != http.StatusNotFound || !reflect.DeepEqual(response, map[string]string{"error": "fixture unsupported path", "path": path}) {
				t.Fatalf("unclassified %s did not preserve runtime 404 JSON (status=%d)", path, status)
			}
			if after := strataContractCount(fixture.strataContractControl(t, 0), http.MethodPost, path, model); after != before+1 {
				t.Fatalf("runtime %s exact method/path/model count = %d, want %d", path, after, before+1)
			}
		})
	}
	passThrough("passes legacy completions to runtime", "/v1/completions")
	passThrough("passes embeddings to runtime", "/v1/embeddings")
	passThrough("passes messages to runtime", "/v1/messages")
	passThrough("passes responses to runtime", "/v1/responses")
}

func TestStrataContractRequiresLocalQualification(t *testing.T) {
	strataContractPlatform(t)
	fixture, proxy := strataContractU4Front(t)
	qualificationPath := filepath.Join(proxy.AppDir, "local-qualification.json")
	if _, err := os.Stat(qualificationPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("qualification evidence must be absent before policy: %v", err)
	}
	frame := fixture.Manager.strataContractNotification(t, "engine:model-identity", func(raw json.RawMessage) bool {
		var identity strataContractIdentity
		strataContractDecode(t, raw, &identity)
		_, present := identity.Engines["strata"]
		return present
	})
	var observed strataContractIdentity
	strataContractDecode(t, frame.Params, &observed)
	actual := observed.Engines["strata"]
	if actual.EngineRunID == "" || len(actual.LaunchDigest) != 64 || len(actual.ModelDigests) != 0 {
		t.Fatal("real manager must identify its owned run without manufacturing model digests")
	}

	// These values are synthetic inputs, not model or hardware evidence. A
	// complete identity prevents missing identity from masking the evidence gate.
	identity := nodepolicy.SetModelIdentityParams{Engines: map[string]nodepolicy.EngineIdentity{"strata": {
		EngineRunID: "synthetic-strata-contract-run", SettingsFingerprint: strings.Repeat("a", 64),
		LaunchDigest: strings.Repeat("b", 64), ModelDigests: map[string]string{"Org/Strata-Loaded:Q4_K_M": strings.Repeat("c", 64)},
	}}}
	if err := identity.Validate(); err != nil {
		t.Fatalf("synthetic identity is invalid: %v", err)
	}
	proxy.RPC.strataContractOK(t, nodepolicy.MethodSetModelIdentity, identity, nil)
	policy := nodepolicy.Default()
	policy.Revision = "41"
	policy.Admission.MaxConcurrentPerModel = 1
	policy.Profiles = []nodepolicy.Profile{{
		Name: "strata-contract", Engine: "strata", Model: "Org/Strata-Loaded:Q4_K_M",
		ContextTokens: 4096, MaxOutputTokens: 1024, MaxConcurrent: 1,
		Qualification: &nodepolicy.QualificationRef{ID: "synthetic-missing-evidence", Revision: "1"},
	}}
	if err := policy.Validate(); err != nil {
		t.Fatalf("qualification policy is invalid: %v", err)
	}
	var ack nodepolicy.PolicyAck
	proxy.RPC.strataContractOK(t, nodepolicy.MethodSetPolicy, policy, &ack)
	if ack.Revision != "41" || ack.RunID == "" {
		t.Fatalf("policy acknowledgment = %+v, want revision 41 and a process run ID", ack)
	}
	readyFrame := proxy.RPC.strataContractNotification(t, nodepolicy.NotifyStrictReadiness, func(raw json.RawMessage) bool {
		var report nodepolicy.StrictAdmissionReadiness
		strataContractDecode(t, raw, &report)
		return report.PolicyRevision == ack.Revision && report.ProxyRunID == ack.RunID
	})
	var readiness nodepolicy.StrictAdmissionReadiness
	strataContractDecode(t, readyFrame.Params, &readiness)
	if err := readiness.Validate(); err != nil {
		t.Fatalf("strict readiness report is invalid: %v", err)
	}
	if readiness.Capability != nil {
		t.Fatal("strict capability was granted without local qualification evidence")
	}
	strataContractU4Chat(t, fixture, proxy)
	if _, err := os.Stat(qualificationPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("ordinary chat must not create qualification evidence: %v", err)
	}
}
