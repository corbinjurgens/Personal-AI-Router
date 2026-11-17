// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"nvpair-shared/modelconfig"
)

func TestSourceModelsListingIsPathFree(t *testing.T) {
	root := t.TempDir()
	writeSourceGGUF(t, filepath.Join(root, "pub", "repo", "Model-Q4_K_M.gguf"))
	writeSourceGGUF(t, filepath.Join(root, "pub", "repo", "mmproj-model.gguf"))
	ex := linkedTestExecutor(t)
	ex.modelSources = []modelSource{{name: "lmstudio", layout: sourceLayoutGGUFTree, root: root}}
	linkDir := t.TempDir()
	writeSourceGGUF(t, filepath.Join(linkDir, "Split-00001-of-00002.gguf"))
	writeSourceGGUF(t, filepath.Join(linkDir, "Split-00002-of-00002.gguf"))
	id, err := ex.linkModel("llamacpp", filepath.Join(linkDir, "Split-00001-of-00002.gguf"))
	if err != nil {
		t.Fatal(err)
	}

	res := ex.sourceModelsList(context.Background(), "llamacpp")
	if res.Error != "" || len(res.Models) != 2 {
		t.Fatalf("listing = %+v", res)
	}
	studio, linked := res.Models[0], res.Models[1]
	if studio.ID != "lmstudio/pub/repo/Model-Q4_K_M" || studio.Source != "lmstudio" || studio.Name != "pub/repo/Model-Q4_K_M" || !studio.HasProjector || studio.Shards != 1 || studio.SizeBytes <= 0 {
		t.Fatalf("LM Studio row = %+v", studio)
	}
	if linked.ID != id || linked.Source != "local" || linked.Name != "Split-00001-of-00002.gguf" || linked.Shards != 2 || linked.HasProjector {
		t.Fatalf("linked row = %+v", linked)
	}
	data, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	for _, dir := range []string{root, linkDir} {
		if strings.Contains(string(data), filepath.ToSlash(dir)) || strings.Contains(string(data), strings.ReplaceAll(dir, `\`, `\\`)) {
			t.Fatalf("listing leaks a host path: %s", data)
		}
	}
}

func TestUnlinkModelRefusedWhileAProfileUsesIt(t *testing.T) {
	ex := copyTestExecutor(t, "llamacpp", t.TempDir(), false)
	ex.baseDir = t.TempDir()
	file := filepath.Join(t.TempDir(), "Model.gguf")
	writeSourceGGUF(t, file)
	id, err := ex.linkModel("llamacpp", file)
	if err != nil {
		t.Fatal(err)
	}
	st, err := ex.state("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	st.plat.Runtime.ModelLoads = []modelconfig.ModelLoad{{ProfileKey: "chat", ModelID: id}}
	err = ex.unlinkUnused(context.Background(), "llamacpp", id)
	if err == nil || !strings.Contains(err.Error(), "used by profile(s) chat") {
		t.Fatalf("unlink while used = %v", err)
	}
	st.plat.Runtime.ModelLoads = nil
	if err := ex.unlinkUnused(context.Background(), "llamacpp", id); err != nil {
		t.Fatalf("unlink after the profile is gone: %v", err)
	}
}

func TestSourceModelsControlRoute(t *testing.T) {
	root := t.TempDir()
	writeSourceGGUF(t, filepath.Join(root, "pub", "repo", "Model.gguf"))
	ex := linkedTestExecutor(t)
	ex.modelSources = []modelSource{{name: "lmstudio", layout: sourceLayoutGGUFTree, root: root}}
	s := &controlServer{exec: ex}
	var route *controlRoute
	for _, r := range s.routes() {
		if r.path == controlSourceModelsPath {
			route = &r
		}
	}
	if route == nil || route.class != controlRead || route.method != http.MethodGet {
		t.Fatalf("source model route = %+v", route)
	}

	rec := httptest.NewRecorder()
	s.handleSourceModels(rec, httptest.NewRequest(http.MethodGet, controlSourceModelsPath+"?engine=llamacpp", nil))
	var res sourceModelsResult
	if rec.Code != http.StatusOK || json.Unmarshal(rec.Body.Bytes(), &res) != nil || len(res.Models) != 1 || res.Models[0].ID != "lmstudio/pub/repo/Model" {
		t.Fatalf("route reply %d %s", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	s.handleSourceModels(rec, httptest.NewRequest(http.MethodGet, controlSourceModelsPath, nil))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("missing engine answered %d", rec.Code)
	}
}
