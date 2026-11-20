// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"crypto/sha256"
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/wailsapp/wails/v3/pkg/application"
)

//go:embed build/overview/asset-manifest.json build/overview/renderer
var overviewAssets embed.FS

const (
	rendererPath = "build/overview/renderer"
	manifestPath = "build/overview/asset-manifest.json"
)

type assetManifest struct {
	SHA256 map[string]string `json:"sha256"`
}

func newRendererHandler(assets fs.FS) (http.Handler, error) {
	manifestBytes, err := fs.ReadFile(assets, manifestPath)
	if err != nil {
		return nil, err
	}
	var manifest assetManifest
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		return nil, fmt.Errorf("decode renderer manifest: %w", err)
	}
	if _, ok := manifest.SHA256["index.html"]; !ok {
		return nil, fmt.Errorf("renderer manifest is missing index.html")
	}

	renderer, err := fs.Sub(assets, rendererPath)
	if err != nil {
		return nil, err
	}
	matched := 0
	err = fs.WalkDir(renderer, ".", func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		content, err := fs.ReadFile(renderer, path)
		if err != nil {
			return err
		}
		if digest := fmt.Sprintf("%x", sha256.Sum256(content)); digest != manifest.SHA256[path] {
			return fmt.Errorf("renderer asset %q does not match the manifest", path)
		}
		matched++
		return nil
	})
	if err != nil {
		return nil, err
	}
	if matched != len(manifest.SHA256) {
		return nil, fmt.Errorf("renderer is missing files declared in the manifest")
	}

	// Serve exact manifest paths, without directory listings or an index fallback.
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/")
		if path == "" {
			path = "index.html"
		}
		if _, ok := manifest.SHA256[path]; !ok {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		content, err := fs.ReadFile(renderer, path)
		if err != nil {
			http.Error(w, "renderer asset unavailable", http.StatusInternalServerError)
			return
		}
		http.ServeContent(w, r, path, time.Time{}, bytes.NewReader(content))
	}), nil
}

// newOverviewBridge selects the renderer input and its native transport.
func newOverviewBridge(mode, scene, order string) (string, application.AssetOptions, error) {
	assets := application.AssetOptions{}
	switch mode {
	case "replay":
	case "live":
		return "", assets, fmt.Errorf("live mode is not implemented")
	default:
		return "", assets, fmt.Errorf("unknown Overview mode %q", mode)
	}
	switch scene {
	case "idle", "shared-node", "running-12", "running-13":
	default:
		return "", assets, fmt.Errorf("unknown replay scene %q", scene)
	}
	if order != "" && order != "nodes-first" {
		return "", assets, fmt.Errorf("unknown replay order %q", order)
	}

	handler, err := newRendererHandler(overviewAssets)
	if err != nil {
		return "", assets, err
	}
	query := url.Values{"mode": {mode}, "scene": {scene}}
	if order != "" {
		query.Set("order", order)
	}
	// Replay uses only the shared renderer. Skip Wails' built-in runtime routes.
	assets.Middleware = func(http.Handler) http.Handler { return handler }
	assets.DisableLogging = true
	return "/?" + query.Encode(), assets, nil
}
