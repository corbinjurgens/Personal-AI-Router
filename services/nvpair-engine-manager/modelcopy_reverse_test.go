// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
)

type reverseCopyFixture struct {
	peer    *remoteReverseFixture
	dst     *Executor
	model   ollamaFixture
	root    string
	weights string
}

func newReverseCopyFixture(t *testing.T, wrap func(http.Handler) http.Handler) *reverseCopyFixture {
	t.Helper()
	srcDir, dstDir := t.TempDir(), t.TempDir()
	src := copyTestExecutor(t, "ollama", srcDir, true)
	f := &reverseCopyFixture{
		dst:   copyTestExecutor(t, "ollama", dstDir, true),
		model: writeOllamaModel(t, filepath.Join(srcDir, "models"), "tiny", 500_000, 5),
		root:  filepath.Join(dstDir, "models"),
	}
	for path, data := range f.model.blobs {
		if len(data) == 500_000 {
			f.weights = path
		}
	}
	if f.weights == "" {
		t.Fatal("missing weights fixture")
	}
	f.peer = newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
		s := &controlServer{exec: src, mesh: mesh}
		mux := http.NewServeMux()
		mux.HandleFunc(controlModelFilesPath, s.handleModelFiles)
		mux.HandleFunc(controlModelFilePath, s.handleModelFile)
		var handler http.Handler = mux
		if wrap != nil {
			handler = wrap(handler)
		}
		return s.requirePin(handler.ServeHTTP)
	})
	return f
}

func (f *reverseCopyFixture) assertPublished(t *testing.T) {
	t.Helper()
	for path, want := range f.model.blobs {
		got, err := os.ReadFile(filepath.Join(f.root, filepath.FromSlash(path)))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) || sha256Hex(got) != sha256Hex(want) {
			t.Fatalf("published blob %s differs from source", path)
		}
	}
	got, err := os.ReadFile(filepath.Join(f.root, filepath.FromSlash(f.model.manifestRel)))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, f.model.manifest) {
		t.Fatal("published manifest differs from source")
	}
	assertNoPartFiles(t, f.root)
}

func TestModelCopyReversePreservesPublicationOrder(t *testing.T) {
	test := func(name string, direct bool) {
		t.Run(name, func(t *testing.T) {
			var f *reverseCopyFixture
			var mu sync.Mutex
			var fetched []string
			f = newReverseCopyFixture(t, func(next http.Handler) http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path == controlModelFilePath {
						path := r.URL.Query().Get("path")
						mu.Lock()
						fetched = append(fetched, path)
						mu.Unlock()
						if !r.Close {
							t.Error("file request retained a replayable connection")
						}
						if path == f.model.manifestRel {
							for rel, want := range f.model.blobs {
								got, err := os.ReadFile(filepath.Join(f.root, filepath.FromSlash(rel)))
								if err != nil || !bytes.Equal(got, want) {
									t.Errorf("manifest requested before verified blob %s: %v", rel, err)
								}
							}
						}
						if _, err := os.Stat(filepath.Join(f.root, filepath.FromSlash(f.model.manifestRel))); !errors.Is(err, os.ErrNotExist) {
							t.Errorf("manifest visible during download: %v", err)
						}
					}
					next.ServeHTTP(w, r)
				})
			})
			f.peer.directHealthy.Store(direct)
			ctx := remoteFixtureContext(t)
			result, err := f.dst.CopyModelFrom(ctx, f.peer.client(t, ctx), "ollama", "tiny", nil)
			if err != nil {
				t.Fatal(err)
			}
			if result.Files != 3 || result.FilesSkipped != 0 || result.BytesCopied != result.BytesTotal {
				t.Fatalf("copy result = %+v", result)
			}
			f.assertPublished(t)
			mu.Lock()
			defer mu.Unlock()
			if len(fetched) != 3 || fetched[2] != f.model.manifestRel {
				t.Fatalf("file request order = %q, want two blobs then manifest", fetched)
			}
			if direct {
				if f.peer.directCalls.Load() != 4 || f.peer.reverseCalls.Load() != 0 {
					t.Fatalf("direct/reverse = %d/%d, want 4/0", f.peer.directCalls.Load(), f.peer.reverseCalls.Load())
				}
			} else if f.peer.reverseCalls.Load() != 4 {
				t.Fatalf("reverse calls = %d, want listing and three file requests", f.peer.reverseCalls.Load())
			}
		})
	}
	test("restricted source uses reverse", false)
	test("reachable source uses direct", true)
}

func TestModelCopyReverseResumesDroppedBodyWithNewRange(t *testing.T) {
	var f *reverseCopyFixture
	var mu sync.Mutex
	var ranges []string
	f = newReverseCopyFixture(t, func(next http.Handler) http.Handler {
		cut := interruptAfter("sha256-"+sha256Hex(testBytes(500_000, 5)), 200_000)(next)
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == controlModelFilePath && r.URL.Query().Get("path") == f.weights {
				mu.Lock()
				ranges = append(ranges, r.Header.Get("Range"))
				mu.Unlock()
				if r.Header.Get("Range") != "" {
					part, err := os.ReadFile(filepath.Join(f.root, filepath.FromSlash(f.weights)) + ".part")
					if err != nil || !bytes.Equal(part, f.model.blobs[f.weights][:200_000]) {
						t.Errorf("resumed partial = %d bytes, error = %v", len(part), err)
					}
					if _, err := os.Stat(filepath.Join(f.root, filepath.FromSlash(f.model.manifestRel))); !errors.Is(err, os.ErrNotExist) {
						t.Errorf("manifest visible before resumed file verification: %v", err)
					}
				}
			}
			cut.ServeHTTP(w, r)
		})
	})
	ctx := remoteFixtureContext(t)
	if _, err := f.dst.CopyModelFrom(ctx, f.peer.client(t, ctx), "ollama", "tiny", nil); err != nil {
		t.Fatal(err)
	}
	f.assertPublished(t)
	mu.Lock()
	defer mu.Unlock()
	if len(ranges) != 2 || ranges[0] != "" || ranges[1] != "bytes=200000-" {
		t.Fatalf("Range requests = %q, want fresh request then bytes=200000-", ranges)
	}
	if f.peer.reverseCalls.Load() != 5 {
		t.Fatalf("reverse requests = %d, want listing, three files and one owner retry", f.peer.reverseCalls.Load())
	}
}

func TestModelCopyReverseCancellationKeepsPartialWithoutVisibility(t *testing.T) {
	var f *reverseCopyFixture
	var calls atomic.Int32
	blocked := make(chan struct{})
	finished := make(chan struct{})
	f = newReverseCopyFixture(t, func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != controlModelFilePath || r.URL.Query().Get("path") != f.weights {
				next.ServeHTTP(w, r)
				return
			}
			calls.Add(1)
			w.Header().Set("Content-Length", "500000")
			if _, err := w.Write(f.model.blobs[f.weights][:65536]); err != nil {
				t.Error(err)
				return
			}
			if err := http.NewResponseController(w).Flush(); err != nil {
				t.Error(err)
				return
			}
			close(blocked)
			<-r.Context().Done()
			close(finished)
		})
	})
	ctx, cancel := context.WithCancel(remoteFixtureContext(t))
	defer cancel()
	client := f.peer.client(t, ctx)
	dest := filepath.Join(f.root, filepath.FromSlash(f.weights))
	file := plannedFile{modelFile: modelFile{Path: f.weights, Size: 500_000, SHA256: sha256Hex(f.model.blobs[f.weights])}, dest: dest}
	// Cancel only once bytes reached the copy owner's .part, independently of
	// HTTP buffering and the throttled user-facing progress notifications.
	_, _, err := downloadPlannedFile(ctx, client, "ollama", "tiny", file, func(n int64) {
		if n >= 65536 {
			cancel()
		}
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("copy error = %v, want cancellation", err)
	}
	select {
	case <-blocked:
	default:
		t.Fatal("source never streamed partial bytes")
	}
	select {
	case <-finished:
	case <-remoteFixtureContext(t).Done():
		t.Fatal("cancellation did not reach source handler")
	}
	part, err := os.ReadFile(dest + ".part")
	if err != nil || !bytes.Equal(part, f.model.blobs[f.weights][:65536]) {
		t.Fatalf("retained partial = %d bytes, error = %v", len(part), err)
	}
	for _, path := range []string{dest, filepath.Join(f.root, filepath.FromSlash(f.model.manifestRel))} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("unverified file visible at %s: %v", path, err)
		}
	}
	if calls.Load() != 1 || f.peer.reverseCalls.Load() != 1 {
		t.Fatalf("file/reverse calls = %d/%d, want 1/1 without replay", calls.Load(), f.peer.reverseCalls.Load())
	}
}

func TestModelCopyReverseStalePartialRetriesFreshAfterHashMismatch(t *testing.T) {
	var f *reverseCopyFixture
	var mu sync.Mutex
	var ranges []string
	f = newReverseCopyFixture(t, func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == controlModelFilePath && r.URL.Query().Get("path") == f.weights {
				mu.Lock()
				ranges = append(ranges, r.Header.Get("Range"))
				mu.Unlock()
			}
			next.ServeHTTP(w, r)
		})
	})
	const offset = 32768
	writeTestFile(t, filepath.Join(f.root, filepath.FromSlash(f.weights))+".part", bytes.Repeat([]byte{0}, offset))
	ctx := remoteFixtureContext(t)
	if _, err := f.dst.CopyModelFrom(ctx, f.peer.client(t, ctx), "ollama", "tiny", nil); err != nil {
		t.Fatal(err)
	}
	f.assertPublished(t)
	mu.Lock()
	defer mu.Unlock()
	if len(ranges) != 2 || ranges[0] != "bytes="+strconv.Itoa(offset)+"-" || ranges[1] != "" {
		t.Fatalf("Range requests = %q, want stale resume then one fresh hash retry", ranges)
	}
}

func TestModelCopyReverseFailureBeforeManifestKeepsOnlyVerifiedData(t *testing.T) {
	var f *reverseCopyFixture
	var refused atomic.Int32
	f = newReverseCopyFixture(t, func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == controlModelFilePath && r.URL.Query().Get("path") == f.model.manifestRel {
				refused.Add(1)
				http.Error(w, "fixture refused final visibility", http.StatusForbidden)
				return
			}
			next.ServeHTTP(w, r)
		})
	})
	ctx := remoteFixtureContext(t)
	if _, err := f.dst.CopyModelFrom(ctx, f.peer.client(t, ctx), "ollama", "tiny", nil); err == nil {
		t.Fatal("manifest refusal reported success")
	}
	if refused.Load() != 1 {
		t.Fatalf("visibility requests = %d, want one permanent refusal", refused.Load())
	}
	for path, want := range f.model.blobs {
		got, err := os.ReadFile(filepath.Join(f.root, filepath.FromSlash(path)))
		if err != nil || !bytes.Equal(got, want) {
			t.Fatalf("verified data %s missing or changed: %v", path, err)
		}
	}
	if _, err := os.Stat(filepath.Join(f.root, filepath.FromSlash(f.model.manifestRel))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed copy published manifest: %v", err)
	}
}

func TestModelCopyReverseInvalidPinNeverReachesFileHandlers(t *testing.T) {
	test := func(name string, invalidate func(*testing.T, *remoteReverseFixture)) {
		t.Run(name, func(t *testing.T) {
			var calls atomic.Int32
			f := newReverseCopyFixture(t, func(next http.Handler) http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls.Add(1)
					next.ServeHTTP(w, r)
				})
			})
			ctx := remoteFixtureContext(t)
			client := f.peer.client(t, ctx)
			invalidate(t, f.peer)
			if _, err := client.modelFiles(ctx, "ollama", "tiny", nil); err == nil {
				t.Fatal("invalid pin allowed file listing")
			}
			body, _, err := client.modelFile(ctx, "ollama", "tiny", f.weights, 0)
			if body != nil {
				if closeErr := body.Close(); closeErr != nil {
					t.Error(closeErr)
				}
			}
			if err == nil || calls.Load() != 0 {
				t.Fatalf("file error = %v, handler calls = %d, want refusal before handler", err, calls.Load())
			}
		})
	}
	test("removed pin", func(t *testing.T, f *remoteReverseFixture) {
		clustertrusttest.RemovePeerPin(t, f.selfDir, "uuid-peer")
	})
	test("wrong pin", func(t *testing.T, f *remoteReverseFixture) {
		clustertrusttest.WriteKeypair(t, f.peerDir, "uuid-peer")
		writePinFromCert(t, f.selfDir, "uuid-peer", filepath.Join(f.peerDir, "node.crt"))
	})
}

func TestModelCopyReverseDispatchedDirectGETDoesNotReplay(t *testing.T) {
	test := func(name string, call func(context.Context, *remoteClient) error) {
		t.Run(name, func(t *testing.T) {
			var calls atomic.Int32
			f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
				return (&controlServer{mesh: mesh}).requirePin(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					calls.Add(1)
					conn, _, err := http.NewResponseController(w).Hijack()
					if err != nil {
						t.Error(err)
						return
					}
					if err := conn.Close(); err != nil {
						t.Error(err)
					}
				}))
			})
			f.directHealthy.Store(true)
			ctx := remoteFixtureContext(t)
			if err := call(ctx, f.client(t, ctx)); err == nil {
				t.Fatal("dispatched GET with no response reported success")
			}
			if calls.Load() != 1 || f.directCalls.Load() != 1 || f.reverseCalls.Load() != 0 {
				t.Fatalf("HTTP/direct/reverse = %d/%d/%d, want 1/1/0 without replay", calls.Load(), f.directCalls.Load(), f.reverseCalls.Load())
			}
		})
	}
	test("file listing", func(ctx context.Context, client *remoteClient) error {
		_, err := client.modelFiles(ctx, "ollama", "tiny", nil)
		return err
	})
	test("file download", func(ctx context.Context, client *remoteClient) error {
		body, _, err := client.modelFile(ctx, "ollama", "tiny", "blobs/fixture", 0)
		if body != nil {
			err = errors.Join(err, body.Close())
		}
		return err
	})
}

func TestModelCopyReverseListingProgressAndFileBytesStreamBeforeCompletion(t *testing.T) {
	listingRelease, fileRelease := make(chan struct{}), make(chan struct{})
	defer close(listingRelease)
	defer close(fileRelease)
	data := testBytes(65536, 9)
	f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
		return (&controlServer{mesh: mesh}).requirePin(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case controlModelFilesPath:
				remoteFixtureEncode(t, w, streamFrame{Type: "progress", Stage: "hashing", Message: "fixture.gguf"})
				if err := http.NewResponseController(w).Flush(); err != nil {
					t.Error(err)
					return
				}
				select {
				case <-listingRelease:
				case <-r.Context().Done():
					return
				}
				list, err := json.Marshal(modelFileList{Engine: "lmstudio", Model: "fixture", Files: []modelFile{{Path: "fixture.gguf", Size: int64(len(data)), SHA256: sha256Hex(data)}}})
				if err != nil {
					t.Error(err)
					return
				}
				remoteFixtureEncode(t, w, streamFrame{Type: "result", Result: list})
			case controlModelFilePath:
				w.Header().Set("Content-Length", strconv.Itoa(len(data)))
				if _, err := w.Write(data[:4096]); err != nil {
					t.Error(err)
					return
				}
				if err := http.NewResponseController(w).Flush(); err != nil {
					t.Error(err)
					return
				}
				select {
				case <-fileRelease:
				case <-r.Context().Done():
					return
				}
				if _, err := w.Write(data[4096:]); err != nil {
					t.Error(err)
				}
			default:
				http.NotFound(w, r)
			}
		}))
	})
	ctx := remoteFixtureContext(t)
	client := f.client(t, ctx)
	if client.controlHTTP.Timeout != 0 {
		t.Fatal("model copy has a total client timeout")
	}
	progress := make(chan streamFrame, 1)
	listingDone := make(chan error, 1)
	go func() {
		list, err := client.modelFiles(ctx, "lmstudio", "fixture", func(frame streamFrame) { progress <- frame })
		if err == nil && (len(list.Files) != 1 || list.Files[0].SHA256 != sha256Hex(data)) {
			err = errors.New("listing differs from source")
		}
		listingDone <- err
	}()
	select {
	case frame := <-progress:
		if frame.Stage != "hashing" || frame.Message != "fixture.gguf" {
			t.Fatalf("listing progress = %+v", frame)
		}
	case <-ctx.Done():
		t.Fatal("listing progress was buffered until completion")
	}
	select {
	case err := <-listingDone:
		t.Fatalf("listing completed before source released result: %v", err)
	default:
	}
	select {
	case listingRelease <- struct{}{}:
	case <-ctx.Done():
		t.Fatal("source did not receive listing release")
	}
	select {
	case err := <-listingDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal("listing did not complete after release")
	}
	body, offset, err := client.modelFile(ctx, "lmstudio", "fixture", "fixture.gguf", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer body.Close()
	if offset != 0 {
		t.Fatalf("file offset = %d, want zero", offset)
	}
	prefix := make([]byte, 4096)
	if _, err := io.ReadFull(body, prefix); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(prefix, data[:4096]) {
		t.Fatal("streamed prefix differs from source")
	}
	select {
	case fileRelease <- struct{}{}:
	case <-ctx.Done():
		t.Fatal("source did not receive file release")
	}
	rest, err := io.ReadAll(body)
	if err != nil || !bytes.Equal(append(prefix, rest...), data) {
		t.Fatalf("streamed body differs from source: %v", err)
	}
	if f.reverseCalls.Load() != 2 {
		t.Fatalf("reverse calls = %d, want listing and file", f.reverseCalls.Load())
	}
}
