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
	peer     *remoteReverseFixture
	dst      *Executor
	model    string
	files    map[string][]byte
	repoDir  string // destination hub-cache repository
	snapshot string // destination snapshot directory
	root     string // destination store root
	weights  string // the model GGUF, which the destination publishes last
	listing  modelFileList
}

func newReverseCopyFixture(t *testing.T, wrap func(http.Handler) http.Handler) *reverseCopyFixture {
	t.Helper()
	const repo, commit = "owner/tiny-GGUF", "0123456789abcdef0123456789abcdef01234567"
	srcDir, dstDir := t.TempDir(), t.TempDir()
	f := &reverseCopyFixture{
		model:   repo + ":Q4_K_M",
		weights: "tiny-Q4_K_M.gguf",
		files: map[string][]byte{
			"tiny-Q4_K_M.gguf":      testBytes(500_000, 5),
			"mmproj-tiny-Q8_0.gguf": testBytes(5_000, 6),
		},
		repoDir: llamaRepoDir(dstDir, repo),
		root:    dstDir,
	}
	f.snapshot = filepath.Join(f.repoDir, "snapshots", commit)
	writeLlamaCache(t, srcDir, repo, commit, f.files)
	src := copyTestExecutor(t, "llamacpp", srcDir, true)
	f.dst = copyTestExecutor(t, "llamacpp", dstDir, true)
	copyTestNativeModels(t, src, "llamacpp", f.model)
	f.listing = copyTestSourceListing(t, src, "llamacpp", f.model)
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

// notPublished fails the test when the destination cache already lists the
// model: its GGUF or its refs/main is visible.
func (f *reverseCopyFixture) notPublished(t *testing.T, context string) {
	t.Helper()
	for _, path := range []string{filepath.Join(f.snapshot, f.weights), filepath.Join(f.repoDir, "refs", "main")} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Errorf("%s visible %s: %v", path, context, err)
		}
	}
}

func (f *reverseCopyFixture) assertPublished(t *testing.T) {
	t.Helper()
	for path, want := range f.files {
		got, err := os.ReadFile(filepath.Join(f.snapshot, filepath.FromSlash(path)))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) || sha256Hex(got) != sha256Hex(want) {
			t.Fatalf("published file %s differs from source", path)
		}
	}
	if _, err := os.Stat(filepath.Join(f.repoDir, "refs", "main")); err != nil {
		t.Fatalf("published cache has no ref: %v", err)
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
						if path == f.weights {
							for rel, want := range f.files {
								if rel == f.weights {
									continue
								}
								got, err := os.ReadFile(filepath.Join(f.snapshot, filepath.FromSlash(rel)))
								if err != nil || !bytes.Equal(got, want) {
									t.Errorf("model requested before verified file %s: %v", rel, err)
								}
							}
						}
						f.notPublished(t, "during download")
					}
					next.ServeHTTP(w, r)
				})
			})
			f.peer.directHealthy.Store(direct)
			ctx := remoteFixtureContext(t)
			result, err := f.dst.CopyModelFrom(ctx, f.peer.client(t, ctx), "llamacpp", f.model, nil)
			if err != nil {
				t.Fatal(err)
			}
			if result.Files != 2 || result.FilesSkipped != 0 || result.BytesCopied != result.BytesTotal {
				t.Fatalf("copy result = %+v", result)
			}
			f.assertPublished(t)
			mu.Lock()
			defer mu.Unlock()
			if len(fetched) != 2 || fetched[1] != f.weights {
				t.Fatalf("file request order = %q, want the projector then the model", fetched)
			}
			if direct {
				if f.peer.directCalls.Load() != 3 || f.peer.reverseCalls.Load() != 0 {
					t.Fatalf("direct/reverse = %d/%d, want 3/0", f.peer.directCalls.Load(), f.peer.reverseCalls.Load())
				}
			} else if f.peer.reverseCalls.Load() != 3 {
				t.Fatalf("reverse calls = %d, want listing and two file requests", f.peer.reverseCalls.Load())
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
		cut := interruptAfter("tiny-Q4_K_M.gguf", 200_000)(next)
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == controlModelFilePath && r.URL.Query().Get("path") == f.weights {
				mu.Lock()
				ranges = append(ranges, r.Header.Get("Range"))
				mu.Unlock()
				if r.Header.Get("Range") != "" {
					part, err := os.ReadFile(filepath.Join(f.snapshot, filepath.FromSlash(f.weights)) + ".part")
					if err != nil || !bytes.Equal(part, f.files[f.weights][:200_000]) {
						t.Errorf("resumed partial = %d bytes, error = %v", len(part), err)
					}
					f.notPublished(t, "before resumed file verification")
				}
			}
			cut.ServeHTTP(w, r)
		})
	})
	ctx := remoteFixtureContext(t)
	if _, err := f.dst.CopyModelFrom(ctx, f.peer.client(t, ctx), "llamacpp", f.model, nil); err != nil {
		t.Fatal(err)
	}
	f.assertPublished(t)
	mu.Lock()
	defer mu.Unlock()
	if len(ranges) != 2 || ranges[0] != "" || ranges[1] != "bytes=200000-" {
		t.Fatalf("Range requests = %q, want fresh request then bytes=200000-", ranges)
	}
	if f.peer.reverseCalls.Load() != 4 {
		t.Fatalf("reverse requests = %d, want listing, two files and one owner retry", f.peer.reverseCalls.Load())
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
			if _, err := w.Write(f.files[f.weights][:65536]); err != nil {
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
	dest := filepath.Join(f.snapshot, filepath.FromSlash(f.weights))
	file := plannedFile{modelFile: modelFile{Path: f.weights, Size: 500_000, SHA256: sha256Hex(f.files[f.weights])}, dest: dest, root: f.root, exportID: f.listing.ExportID}
	// Cancel only once bytes reached the copy owner's .part, independently of
	// HTTP buffering and the throttled user-facing progress notifications.
	_, _, err := downloadPlannedFile(ctx, client, "llamacpp", f.model, file, func(n int64) {
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
	if err != nil || !bytes.Equal(part, f.files[f.weights][:65536]) {
		t.Fatalf("retained partial = %d bytes, error = %v", len(part), err)
	}
	f.notPublished(t, "after cancellation")
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
	writeTestFile(t, filepath.Join(f.snapshot, filepath.FromSlash(f.weights))+".part", bytes.Repeat([]byte{0}, offset))
	ctx := remoteFixtureContext(t)
	if _, err := f.dst.CopyModelFrom(ctx, f.peer.client(t, ctx), "llamacpp", f.model, nil); err != nil {
		t.Fatal(err)
	}
	f.assertPublished(t)
	mu.Lock()
	defer mu.Unlock()
	if len(ranges) != 2 || ranges[0] != "bytes="+strconv.Itoa(offset)+"-" || ranges[1] != "" {
		t.Fatalf("Range requests = %q, want stale resume then one fresh hash retry", ranges)
	}
}

func TestModelCopyReverseFailureBeforeFinalFileKeepsOnlyVerifiedData(t *testing.T) {
	var f *reverseCopyFixture
	var refused atomic.Int32
	f = newReverseCopyFixture(t, func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == controlModelFilePath && r.URL.Query().Get("path") == f.weights {
				refused.Add(1)
				http.Error(w, "fixture refused final visibility", http.StatusForbidden)
				return
			}
			next.ServeHTTP(w, r)
		})
	})
	ctx := remoteFixtureContext(t)
	if _, err := f.dst.CopyModelFrom(ctx, f.peer.client(t, ctx), "llamacpp", f.model, nil); err == nil {
		t.Fatal("model refusal reported success")
	}
	if refused.Load() != 1 {
		t.Fatalf("visibility requests = %d, want one permanent refusal", refused.Load())
	}
	for path, want := range f.files {
		if path == f.weights {
			continue
		}
		got, err := os.ReadFile(filepath.Join(f.snapshot, filepath.FromSlash(path)))
		if err != nil || !bytes.Equal(got, want) {
			t.Fatalf("verified data %s missing or changed: %v", path, err)
		}
	}
	f.notPublished(t, "after a failed copy")
}

func TestModelCopyReverseInvalidPinNeverReachesFileHandlers(t *testing.T) {
	test := func(name string, invalidate func(*testing.T, *remoteReverseFixture)) {
		t.Run(name, func(t *testing.T) {
			var calls atomic.Int32
			source := namedCustomCopyFixture(t, "tiny", 500_000, 5)
			list := sourceFromExecutor(t, source.source, "llamacpp", "tiny").list
			f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
				control := &controlServer{exec: source.source, mesh: mesh}
				return control.requirePin(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls.Add(1)
					control.mux().ServeHTTP(w, r)
				}))
			})
			ctx := remoteFixtureContext(t)
			client := f.client(t, ctx)
			invalidate(t, f)
			if _, err := client.modelFiles(ctx, "llamacpp", "tiny", nil); err == nil {
				t.Fatal("invalid pin allowed file listing")
			}
			body, _, err := client.modelFile(ctx, "llamacpp", "tiny", "model/model.gguf", list.ExportID, 0)
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
		_, err := client.modelFiles(ctx, "llamacpp", "tiny", nil)
		return err
	})
	test("file download", func(ctx context.Context, client *remoteClient) error {
		body, _, err := client.modelFile(ctx, "llamacpp", "tiny", "model/fixture.gguf", "", 0)
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
				list, err := json.Marshal(modelFileList{Engine: "llamacpp", Model: "fixture", Files: []modelFile{{Path: "fixture.gguf", Size: int64(len(data)), SHA256: sha256Hex(data)}}})
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
		list, err := client.modelFiles(ctx, "llamacpp", "fixture", func(frame streamFrame) { progress <- frame })
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
	body, offset, err := client.modelFile(ctx, "llamacpp", "fixture", "fixture.gguf", "", 0)
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
