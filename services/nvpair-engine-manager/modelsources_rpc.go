// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// modelsources_rpc.go answers the methods that list, link and unlink models
// served from outside the store. The listing is path-free and readable by any
// pinned peer; linking and unlinking name local paths, so they act only on the
// node that receives them.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

const controlSourceModelsPath = "/v1/models/sources"

// sourceModelInfo is one outside-the-store model as clients see it: no host
// paths, only what identifies and sizes it.
type sourceModelInfo struct {
	ID string `json:"id"`
	// Source is "ollama", "lmstudio" or "local" (a linked file).
	Source string `json:"source"`
	// Name is what the source calls the model: Ollama's name@tag, LM Studio's
	// relative path, or a linked file's base name.
	Name         string `json:"name"`
	SizeBytes    int64  `json:"sizeBytes"`
	Shards       int    `json:"shards"`
	HasProjector bool   `json:"hasProjector"`
}

// sourceModelsResult is the engine:source-models reply. Error reports a linked
// list that could not be read; the folder sources are still listed.
type sourceModelsResult struct {
	Models []sourceModelInfo `json:"models"`
	Error  string            `json:"error,omitempty"`
}

func (e *Executor) sourceModelsList(ctx context.Context, engine string) sourceModelsResult {
	res := sourceModelsResult{Models: []sourceModelInfo{}}
	e.linked.mu.Lock()
	if _, err := e.linkedEntries(); err != nil {
		res.Error = err.Error()
	}
	e.linked.mu.Unlock()
	for _, model := range e.sourceModelsFor(ctx, engine) {
		if model.format != sourceFormatGGUF {
			continue
		}
		source, name, _ := strings.Cut(model.id, "/")
		if source == linkedSourceName {
			name = filepath.Base(model.path)
		}
		info := sourceModelInfo{ID: model.id, Source: source, Name: name, HasProjector: model.mmproj != ""}
		for _, shard := range sourceModelShards(model) {
			if stat, err := os.Stat(shard); err == nil {
				info.SizeBytes += stat.Size()
			}
			info.Shards++
		}
		res.Models = append(res.Models, info)
	}
	return res
}

// runSourceModels answers engine:source-models for this node or, through the
// control surface, for a peer.
func (m *Manager) runSourceModels(ctx context.Context, msg *Message) {
	var p struct {
		NodeID string `json:"nodeId"`
		Engine string `json:"engine"`
	}
	if !m.parse(msg, &p) {
		return
	}
	if p.NodeID == "" || p.Engine == "" {
		m.codec.RespondError(msg.ID, -32602, "nodeId and engine are required")
		return
	}
	self, err := m.copySelfID()
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	if p.NodeID == self {
		m.codec.Respond(msg.ID, m.exec.sourceModelsList(ctx, p.Engine))
		return
	}
	client, err := m.copyClient(ctx, p.NodeID)
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	res, err := client.sourceModels(ctx, p.Engine)
	m.respondOrErr(msg, res, err)
}

// runLinkModel answers engine:link-model on this node.
func (m *Manager) runLinkModel(msg *Message) {
	var p struct {
		Engine string `json:"engine"`
		Path   string `json:"path"`
	}
	if !m.parse(msg, &p) {
		return
	}
	if p.Engine == "" || p.Path == "" {
		m.codec.RespondError(msg.ID, -32602, "engine and path are required")
		return
	}
	id, err := m.exec.linkModel(p.Engine, p.Path)
	m.respondOrErr(msg, map[string]string{"id": id}, err)
}

// runUnlinkModel answers engine:unlink-model on this node, refusing while a
// profile still serves the linked file.
func (m *Manager) runUnlinkModel(ctx context.Context, msg *Message) {
	var p struct {
		Engine string `json:"engine"`
		ID     string `json:"id"`
	}
	if !m.parse(msg, &p) {
		return
	}
	if p.Engine == "" || p.ID == "" {
		m.codec.RespondError(msg.ID, -32602, "engine and id are required")
		return
	}
	m.respondOrErr(msg, struct{}{}, m.exec.unlinkUnused(ctx, p.Engine, p.ID))
}

func (e *Executor) unlinkUnused(ctx context.Context, engine, id string) error {
	st, err := e.state(engine)
	if err != nil {
		return err
	}
	snapshot, err := readOnlyInspectionSnapshot(ctx, st, engine)
	if err != nil {
		return err
	}
	var users []string
	for _, load := range snapshot.state.plat.Runtime.ModelLoads {
		if load.ModelID == id {
			users = append(users, load.ProfileKey)
		}
	}
	if len(users) > 0 {
		return modelSettingsParamsError(fmt.Errorf("%s is used by profile(s) %s; remove or change those profiles first.", id, strings.Join(users, ", ")))
	}
	return e.unlinkModel(engine, id)
}

func (s *controlServer) handleSourceModels(w http.ResponseWriter, r *http.Request) {
	q, err := url.ParseQuery(r.URL.RawQuery)
	if err != nil || len(q) != 1 || len(q["engine"]) != 1 || q.Get("engine") == "" {
		http.Error(w, "engine is required", http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(s.exec.sourceModelsList(r.Context(), q.Get("engine")))
}

func (c *remoteClient) sourceModels(ctx context.Context, engine string) (sourceModelsResult, error) {
	var res sourceModelsResult
	ctx, cancel := context.WithTimeout(ctx, remoteResponseHeaderTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+controlSourceModelsPath+"?engine="+url.QueryEscape(engine), nil)
	if err != nil {
		return res, err
	}
	resp, err := c.controlClient(false).Do(req)
	if err != nil {
		return res, err
	}
	defer resp.Body.Close()
	data, err := readRemoteResponse(resp.Body, maxCopyControlBody)
	if err != nil {
		return res, err
	}
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusMethodNotAllowed {
		return res, fmt.Errorf("device does not list models outside its store")
	}
	if resp.StatusCode != http.StatusOK {
		return res, remoteStatusError("source models", resp.StatusCode, data)
	}
	if err := json.Unmarshal(data, &res); err != nil || res.Models == nil {
		return res, fmt.Errorf("unrecognized source model listing")
	}
	return res, nil
}
