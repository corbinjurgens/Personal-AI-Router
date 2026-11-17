// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
)

const catalogFilesRevision = "f9c28bcd85737ffc5aef028638d3341d49869c27"

// catalogFilesServer stands in for the provider's revision endpoint, answering
// owner/Model-GGUF with body and counting requests.
func catalogFilesServer(t *testing.T, status int, body string) (*catalogService, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/catalog/owner/Model-GGUF/revision/main", func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Query().Get("blobs") != "true" {
			t.Errorf("listing requested without blobs: %s", r.URL.RawQuery)
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	c := newCatalogService()
	c.baseURL = srv.URL + "/catalog"
	return c, &calls
}

func catalogFilesBody(private bool, gated string, siblings string) string {
	return fmt.Sprintf(`{"id":"owner/Model-GGUF","sha":%q,"private":%t,"gated":%s,"siblings":[%s]}`,
		catalogFilesRevision, private, gated, siblings)
}

const catalogFilesSiblings = `{"rfilename":"README.md","size":10},
{"rfilename":"x-Q4_K_M.gguf","size":400},
{"rfilename":"x-Q8_0-00001-of-00002.gguf","size":500},
{"rfilename":"x-Q8_0-00002-of-00002.gguf","size":300,"lfs":{"size":300}},
{"rfilename":"mmproj-x-f16.gguf","size":90},
{"rfilename":"x.imatrix.gguf","size":5}`

func TestCatalogFilesNormalizesRepoForms(t *testing.T) {
	for _, input := range []string{
		"owner/Model-GGUF",
		"  owner/Model-GGUF  ",
		"owner/Model-GGUF:Q8_0",
		"https://huggingface.co/owner/Model-GGUF",
		"https://huggingface.co/owner/Model-GGUF/tree/main?x=1#readme",
		"http://huggingface.co/owner/Model-GGUF",
		"huggingface.co/owner/Model-GGUF/blob/main/x-Q4_K_M.gguf",
		"https://hf.co/owner/Model-GGUF",
		"hf.co/owner/Model-GGUF",
	} {
		t.Run(input, func(t *testing.T) {
			got, err := normalizeCatalogRepo(input)
			if err != nil || got != "owner/Model-GGUF" {
				t.Fatalf("normalizeCatalogRepo(%q) = %q, %v", input, got, err)
			}
		})
	}
}

func TestCatalogFilesRefusesMalformedInputWithoutFetch(t *testing.T) {
	c, calls := catalogFilesServer(t, http.StatusOK, catalogFilesBody(false, "false", catalogFilesSiblings))
	for name, input := range map[string]string{
		"too long":       "owner/" + strings.Repeat("m", maxCatalogRepoInput),
		"empty":          "  ",
		"one segment":    "Model-GGUF",
		"path traversal": "../etc/passwd",
		"other host":     "https://example.com/owner/Model-GGUF",
		"link no name":   "https://huggingface.co/owner",
	} {
		t.Run(name, func(t *testing.T) {
			_, err := c.CatalogFiles(context.Background(), "llamacpp", input)
			if !errors.Is(err, errCatalogRepoInput) {
				t.Fatalf("input accepted or misreported: %v", err)
			}
		})
	}
	if calls.Load() != 0 {
		t.Fatalf("malformed input reached the provider %d times", calls.Load())
	}
}

func TestCatalogFilesRefusesClosedRepos(t *testing.T) {
	for name, body := range map[string]string{
		"gated":   catalogFilesBody(false, `"manual"`, catalogFilesSiblings),
		"private": catalogFilesBody(true, "false", catalogFilesSiblings),
		"unknown": `{"id":"owner/Model-GGUF","sha":"` + catalogFilesRevision + `","private":false,"siblings":[]}`,
	} {
		t.Run(name, func(t *testing.T) {
			c, _ := catalogFilesServer(t, http.StatusOK, body)
			_, err := c.CatalogFiles(context.Background(), "llamacpp", "owner/Model-GGUF")
			if err == nil || !strings.Contains(err.Error(), "private or gated") {
				t.Fatalf("closed repository listed: %v", err)
			}
		})
	}
	t.Run("missing", func(t *testing.T) {
		c, _ := catalogFilesServer(t, http.StatusUnauthorized, `{"error":"Invalid username or password."}`)
		_, err := c.CatalogFiles(context.Background(), "llamacpp", "owner/Model-GGUF")
		if err == nil || !strings.Contains(err.Error(), "Repository not found") {
			t.Fatalf("missing repository misreported: %v", err)
		}
	})
}

func TestCatalogFilesGroupsQuantizationsAndCaches(t *testing.T) {
	c, calls := catalogFilesServer(t, http.StatusOK, catalogFilesBody(false, "false", catalogFilesSiblings))
	res, err := c.CatalogFiles(context.Background(), "llama-cpp", "https://huggingface.co/owner/Model-GGUF")
	if err != nil {
		t.Fatalf("list files: %v", err)
	}
	want := []CatalogQuantFile{
		{Quantization: "Q4_K_M", Name: "owner/Model-GGUF:Q4_K_M", SizeBytes: 400, Shards: 1},
		{Quantization: "Q8_0", Name: "owner/Model-GGUF:Q8_0", SizeBytes: 800, Shards: 2},
	}
	if res.Repo != "owner/Model-GGUF" || res.Revision != catalogFilesRevision || res.FetchedAt == "" || !reflect.DeepEqual(res.Files, want) {
		t.Fatalf("listing = %+v", res)
	}
	if _, err := c.CatalogFiles(context.Background(), "llamacpp", "owner/model-gguf"); err != nil {
		t.Fatalf("cached listing: %v", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("provider read %d times within the cache lifetime, want 1", calls.Load())
	}
}

func TestCatalogFilesRefusesOtherEngines(t *testing.T) {
	c, calls := catalogFilesServer(t, http.StatusOK, catalogFilesBody(false, "false", catalogFilesSiblings))
	if _, err := c.CatalogFiles(context.Background(), "ollama", "owner/Model-GGUF"); err == nil || calls.Load() != 0 {
		t.Fatalf("non-llama.cpp engine listed: err=%v calls=%d", err, calls.Load())
	}
}

func TestGGUFFileQuantization(t *testing.T) {
	for name, want := range map[string]string{
		"gemma-3-1b-it-Q4_K_M":      "Q4_K_M",
		"Qwen3-8B-UD-Q4_K_XL":       "UD-Q4_K_XL",
		"model.q8_0":                "Q8_0",
		"model_IQ4_XS":              "IQ4_XS",
		"Llama-3.2-1B-Instruct-f16": "F16",
		"gpt-oss-20b-MXFP4":         "MXFP4",
		"BF16":                      "BF16",
	} {
		if got, ok := ggufFileQuantization(name); !ok || got != want {
			t.Errorf("ggufFileQuantization(%q) = %q, %t; want %q", name, got, ok, want)
		}
	}
	if got, ok := ggufFileQuantization("model-final"); ok {
		t.Errorf("unlabelled file reported quantization %q", got)
	}
}
