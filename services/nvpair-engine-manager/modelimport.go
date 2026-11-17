// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// modelimport.go answers engine:import-model: it copies a GGUF file on this
// node, or a model from one of its model sources, into the engine's own store
// as a custom-artifact bundle. The copy is the peer-copy path with a local file
// source, so it gets the same free-space check, locks, full-hash verification
// and receipt; the receipt then lets a profile name the model by its new ID.
// The original file is only read.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// importParams is the engine:import-model request: exactly one of Path (a file
// the operator picked) or SourceID (a model from engine:source-models).
type importParams struct {
	Engine   string `json:"engine"`
	Path     string `json:"path"`
	SourceID string `json:"sourceId"`
	ID       string `json:"id"`
}

// importResult is the engine:import-model reply.
type importResult struct {
	ID         string         `json:"id"`
	Layout     string         `json:"layout"`
	ExportID   string         `json:"exportId"`
	Durability copyDurability `json:"durability"`
}

// localModelSource serves files on this host to the copy driver under the
// custom layout's logical names.
type localModelSource struct {
	list  modelFileList
	hosts map[string]string // logical path -> host path
}

func (s *localModelSource) modelFiles(_ context.Context, engine, model string, _ func(streamFrame)) (modelFileList, error) {
	if engine != s.list.Engine || model != s.list.Model {
		return modelFileList{}, errModelNotFound
	}
	return s.list, nil
}

func (s *localModelSource) modelFile(_ context.Context, engine, model, path, exportID string, offset int64) (io.ReadCloser, int64, error) {
	host, ok := s.hosts[path]
	if !ok || engine != s.list.Engine || model != s.list.Model || exportID != s.list.ExportID {
		return nil, 0, errModelNotFound
	}
	f, err := os.Open(host)
	if err != nil {
		return nil, 0, err
	}
	if offset > 0 {
		if _, err := f.Seek(offset, io.SeekStart); err != nil {
			f.Close()
			return nil, 0, err
		}
	}
	return f, offset, nil
}

// importFiles are the host files an import copies, model shards in order.
// Custom bundles carry model and draft files only, so a projector beside the
// model is not imported.
func (e *Executor) importFiles(ctx context.Context, p importParams) ([]string, error) {
	switch {
	case (p.Path == "") == (p.SourceID == ""):
		return nil, modelSettingsParamsError(errors.New("Name exactly one of path or sourceId."))
	case p.Path != "":
		if _, err := checkLinkableFile(p.Path); err != nil {
			return nil, err
		}
		return sourceModelShards(sourceModel{path: filepath.Clean(p.Path)}), nil
	default:
		for _, model := range e.sourceModelsFor(ctx, p.Engine) {
			if model.id == p.SourceID && model.format == sourceFormatGGUF {
				return sourceModelShards(model), nil
			}
		}
		return nil, modelSettingsParamsError(fmt.Errorf("%s is not a model this node serves from outside its store", p.SourceID))
	}
}

// importModel copies the files into a custom bundle named p.ID.
func (e *Executor) importModel(ctx context.Context, p importParams) (importResult, error) {
	if err := validateLlamaModelID(p.ID); err != nil || strings.ContainsAny(p.ID, sourcePresetUnsafe) {
		return importResult{}, modelSettingsParamsError(errors.New("Choose a model name of letters, digits and - _ . / @ only."))
	}
	// The resolver would also bind a source model or a Hugging Face download
	// by these names, and refuse the pair as ambiguous.
	if _, _, err := splitLlamaModel(p.ID); err == nil || isSourceModelID(p.ID) {
		return importResult{}, modelSettingsParamsError(errors.New("Choose a name that is not owner/repo (a Hugging Face name) and does not start with ollama/, lmstudio/ or local/."))
	}
	hosts, err := e.importFiles(ctx, p)
	if err != nil {
		return importResult{}, err
	}
	src := &localModelSource{list: modelFileList{Engine: p.Engine, Model: p.ID, Layout: llamaArtifactsLayout}, hosts: map[string]string{}}
	for _, host := range hosts {
		file, err := hashLocalFile(ctx, host)
		if err != nil {
			return importResult{}, fmt.Errorf("read %s: %w", filepath.Base(host), err)
		}
		file.Role, file.Path = "model", "model/"+bundleFileName(host, p.ID)
		src.list.Files = append(src.list.Files, file)
		src.hosts[file.Path] = host
	}
	src.list.ExportID, err = artifactExportID(src.list)
	if err != nil {
		return importResult{}, modelSettingsParamsError(err)
	}
	res, err := e.copyModelFromPublisher(ctx, src, p.Engine, p.ID, "", nil, nil, nil, nil)
	if err != nil {
		return importResult{}, err
	}
	return importResult{ID: p.ID, Layout: res.Layout, ExportID: res.ExportID, Durability: res.Durability}, nil
}

// hashLocalFile reads a file once for its size and full SHA-256.
func hashLocalFile(ctx context.Context, host string) (modelFile, error) {
	f, err := os.Open(host)
	if err != nil {
		return modelFile{}, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, &contextReader{ctx: ctx, r: f})
	if err != nil {
		return modelFile{}, err
	}
	return modelFile{Size: n, SHA256: hex.EncodeToString(h.Sum(nil))}, nil
}

// contextReader stops a long read when its context ends.
type contextReader struct {
	ctx context.Context
	r   io.Reader
}

func (c *contextReader) Read(p []byte) (int, error) {
	if err := c.ctx.Err(); err != nil {
		return 0, err
	}
	return c.r.Read(p)
}

// runImportModel answers engine:import-model on this node. It returns when the
// copy has finished, which can take minutes for a large file.
func (m *Manager) runImportModel(ctx context.Context, msg *Message) {
	var p importParams
	if !m.parse(msg, &p) {
		return
	}
	if p.Engine == "" || p.ID == "" {
		m.codec.RespondError(msg.ID, -32602, "engine and id are required")
		return
	}
	res, err := m.exec.importModel(ctx, p)
	m.respondOrErr(msg, res, err)
}
