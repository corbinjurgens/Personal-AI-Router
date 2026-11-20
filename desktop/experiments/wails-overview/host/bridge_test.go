// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"testing/fstest"
	"unsafe"

	"github.com/wailsapp/wails/v3/pkg/application"
)

func TestReplayServesManifestIdenticalAssets(t *testing.T) {
	manifestBytes, err := overviewAssets.ReadFile("build/overview/asset-manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	var manifest assetManifest
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		t.Fatal(err)
	}
	if _, ok := manifest.SHA256["index.html"]; !ok {
		t.Fatal("manifest is missing index.html")
	}
	handler := replayHandler(t)
	assertAsset := func(path, digest string) {
		t.Run(path, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, path, nil))
			if response.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d", response.Code, http.StatusOK)
			}
			if got := fmt.Sprintf("%x", sha256.Sum256(response.Body.Bytes())); got != digest {
				t.Fatalf("served SHA-256 = %s, want %s", got, digest)
			}
		})
	}
	assertAsset("/?mode=replay&scene=running-13", manifest.SHA256["index.html"])
	for path, digest := range manifest.SHA256 {
		assertAsset("/"+path, digest)
	}
}

func TestReplayServesOnlyRendererFiles(t *testing.T) {
	handler := replayHandler(t)
	for _, path := range []string{
		"/asset-manifest.json",
		"/renderer/index.html",
		"/build/overview/renderer/index.html",
		"/assets/",
		"/assets/missing.js",
		"/wails/runtime.js",
		"/assets/../index.html",
	} {
		t.Run(path, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, path, nil))
			if response.Code != http.StatusNotFound {
				t.Errorf("status = %d, want %d", response.Code, http.StatusNotFound)
			}
		})
	}
}

func TestRendererRejectsInvalidEmbeddedBundle(t *testing.T) {
	const manifest = `{"sha256":{"index.html":"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824","main.js":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}}`
	test := func(name string, index string, arrange func(fstest.MapFS), wantError string) {
		t.Run(name, func(t *testing.T) {
			files := fstest.MapFS{
				"build/overview/asset-manifest.json": {Data: []byte(manifest)},
				"build/overview/renderer/index.html": {Data: []byte(index)},
				"build/overview/renderer/main.js":    {},
			}
			arrange(files)
			_, err := newRendererHandler(files)
			if err == nil || err.Error() != wantError {
				t.Errorf("error = %v, want %q", err, wantError)
			}
		})
	}
	test("changed bytes", "HELLO", func(fstest.MapFS) {},
		`renderer asset "index.html" does not match the manifest`)
	test("extra asset", "hello", func(files fstest.MapFS) {
		files["build/overview/renderer/extra.js"] = &fstest.MapFile{}
	}, `renderer asset "extra.js" does not match the manifest`)
	test("missing asset", "hello", func(files fstest.MapFS) {
		delete(files, "build/overview/renderer/main.js")
	}, "renderer is missing files declared in the manifest")
}

func replayHandler(t *testing.T) http.Handler {
	t.Helper()
	_, assets, err := newOverviewBridge("replay", "running-13", "")
	if err != nil {
		t.Fatal(err)
	}
	upstream := http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("replay request reached the Wails runtime middleware")
	})
	return assets.Middleware(upstream)
}

func TestShowOverviewWindow(t *testing.T) {
	options := application.WebviewWindowOptions{Name: "overview", URL: "/?mode=replay&scene=shared-node"}
	testNewWindow := func(name string, existing *testWindow) {
		t.Run(name, func(t *testing.T) {
			windows := &testWindowManager{existing: existing}
			showOverviewWindow(windows, options)
			if windows.created != 1 || windows.newURL != "/?mode=replay&scene=shared-node" {
				t.Fatalf("created %d windows with URL %q, want one fresh replay window", windows.created, windows.newURL)
			}
			if existing != nil && (existing.focused || existing.restored) {
				t.Error("reused a closed window")
			}
		})
	}
	testNewWindow("registry empty after close", nil)
	testNewWindow("registry still contains destroyed window", &testWindow{})
	t.Run("live window is restored without creating another", func(t *testing.T) {
		current := &testWindow{native: new(byte)}
		windows := &testWindowManager{existing: current}
		showOverviewWindow(windows, options)
		if windows.created != 0 || !current.focused || !current.restored {
			t.Errorf("created = %d, focused = %t, restored = %t, want 0, true, true",
				windows.created, current.focused, current.restored)
		}
	})
}

type testWindow struct {
	application.Window
	native   *byte
	focused  bool
	restored bool
}

func (w *testWindow) NativeWindow() unsafe.Pointer { return unsafe.Pointer(w.native) }
func (w *testWindow) Focus()                       { w.focused = true }
func (w *testWindow) UnMinimise()                  { w.restored = true }

type testWindowManager struct {
	existing *testWindow
	created  int
	newURL   string
}

func (m *testWindowManager) GetByName(name string) (application.Window, bool) {
	if name != "overview" || m.existing == nil {
		return nil, false
	}
	return m.existing, true
}

func (m *testWindowManager) NewWithOptions(options application.WebviewWindowOptions) *application.WebviewWindow {
	m.created++
	m.newURL = options.URL
	return nil
}
