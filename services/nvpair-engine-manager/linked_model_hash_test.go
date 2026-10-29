// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// These counters observe content access itself, including the old streaming
// path. A later error or an absent progress frame cannot hide forbidden reads.
type linkedHashCounts struct {
	opens, readAt, streams, bytes, forbiddenBytes, closes atomic.Int64
}

type linkedHashFile struct {
	*os.File
	counts     *linkedHashCounts
	forbidden  bool
	beforeRead func()
	afterRead  func(int, int64)
	stat       func() (os.FileInfo, error)
}

func (f *linkedHashFile) record(n int) {
	f.counts.bytes.Add(int64(n))
	if f.forbidden {
		f.counts.forbiddenBytes.Add(int64(n))
	}
}

func (f *linkedHashFile) ReadAt(p []byte, off int64) (int, error) {
	f.counts.readAt.Add(1)
	if f.beforeRead != nil {
		f.beforeRead()
	}
	n, err := f.File.ReadAt(p, off)
	f.record(n)
	if f.afterRead != nil {
		f.afterRead(n, off)
	}
	return n, err
}

func (f *linkedHashFile) Read(p []byte) (int, error) {
	f.counts.streams.Add(1)
	n, err := f.File.Read(p)
	f.record(n)
	return n, err
}

func (f *linkedHashFile) WriteTo(w io.Writer) (int64, error) {
	return io.Copy(w, struct{ io.Reader }{f})
}

func (f *linkedHashFile) Stat() (os.FileInfo, error) {
	if f.stat != nil {
		return f.stat()
	}
	return f.File.Stat()
}

func (f *linkedHashFile) Close() error {
	f.counts.closes.Add(1)
	return f.File.Close()
}

func TestLinkedModelHashBinding(t *testing.T) {
	fixture := func(t *testing.T, data []byte) (sourceEvidence, string) {
		t.Helper()
		dir := t.TempDir()
		path, replacement := filepath.Join(dir, "model"), filepath.Join(dir, "replacement")
		writeTestFile(t, path, data)
		original, err := captureSource(path, false)
		if err != nil {
			t.Fatal(err)
		}
		writeTestFile(t, replacement, bytes.Repeat([]byte("x"), len(data)))
		if err := os.Chtimes(replacement, original.info.ModTime(), original.info.ModTime()); err != nil {
			t.Fatal(err)
		}
		other, err := captureSource(replacement, false)
		if err != nil {
			t.Fatal(err)
		}
		if other.size != original.size || other.mtime != original.mtime || os.SameFile(original.info, other.info) {
			t.Fatal("replacement fixture must have identical size/mtime and a distinct identity")
		}
		return original, replacement
	}
	replace := func(t *testing.T, e sourceEvidence, replacement string) {
		t.Helper()
		if err := os.Rename(e.configured, e.configured+".old"); err != nil {
			t.Fatal(err)
		}
		if err := os.Rename(replacement, e.configured); err != nil {
			t.Fatal(err)
		}
	}
	observe := func(e sourceEvidence, choose func(string) string, configure func(*linkedHashFile)) (*hashCache, *linkedHashCounts) {
		counts := &linkedHashCounts{}
		cache := &hashCache{access: servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			if choose != nil {
				path = choose(path)
			}
			file, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			counts.opens.Add(1)
			observed := &linkedHashFile{File: file, counts: counts}
			info, err := file.Stat()
			if err != nil {
				observed.Close()
				return nil, err
			}
			observed.forbidden = !os.SameFile(e.info, info)
			if configure != nil {
				configure(observed)
			}
			return observed, nil
		}}}
		return cache, counts
	}
	assertCounts := func(t *testing.T, counts *linkedHashCounts, wantBytes, wantOpens int64) {
		t.Helper()
		if counts.bytes.Load() != wantBytes || counts.opens.Load() != wantOpens || counts.closes.Load() != wantOpens || counts.streams.Load() != 0 || counts.forbiddenBytes.Load() != 0 {
			t.Errorf("bytes=%d opens=%d closes=%d streams=%d forbidden bytes=%d; want bytes=%d opens=closes=%d and no stream/forbidden reads",
				counts.bytes.Load(), counts.opens.Load(), counts.closes.Load(), counts.streams.Load(), counts.forbiddenBytes.Load(), wantBytes, wantOpens)
		}
		if wantBytes > 0 && counts.readAt.Load() == 0 {
			t.Error("positive control did not exercise ReadAt")
		}
		if wantBytes == 0 && counts.readAt.Load() != 0 {
			t.Errorf("unexpected ReadAt attempts: %d", counts.readAt.Load())
		}
	}
	listFor := func(e sourceEvidence) modelFileList {
		return modelFileList{Files: []modelFile{{Path: "model", abs: e.target, configured: e.configured, info: e.info, Size: e.size}}}
	}
	assertUnpublished := func(t *testing.T, c *hashCache, list modelFileList) {
		t.Helper()
		if len(c.done) != 0 || len(c.inflight) != 0 || list.ExportID != "" || list.Files[0].SHA256 != "" {
			t.Errorf("failed hash published state: cache=%v inflight=%v export=%q digest=%q", c.done, c.inflight, list.ExportID, list.Files[0].SHA256)
		}
	}

	stale := func(name, prefilled, asserted string, warm bool) {
		t.Run(name, func(t *testing.T) {
			e, replacement := fixture(t, []byte("original"))
			c, counts := observe(e, nil, nil)
			var wantBytes, wantOpens int64
			if warm {
				if got, err := c.sumBound(context.Background(), e, nil); err != nil || got != sha256Hex([]byte("original")) {
					t.Fatalf("warm positive control: sum=%q error=%v", got, err)
				}
				wantBytes, wantOpens = e.size, 1
			}
			replace(t, e, replacement)
			list := listFor(e)
			list.Files[0].SHA256, list.Files[0].asserted = prefilled, asserted
			progress := 0
			if err := c.hashModelFiles(context.Background(), &list, func(string, int) { progress++ }); err == nil {
				t.Error("stale binding was accepted")
			}
			if progress != 0 {
				t.Errorf("replacement was hashed: %d progress callbacks", progress)
			}
			if list.Files[0].SHA256 != prefilled || list.ExportID != "" {
				t.Error("stale binding published a digest or export")
			}
			assertCounts(t, counts, wantBytes, wantOpens)
		})
	}
	stale("cold replacement", "", "", false)
	stale("prefilled replacement", "asserted-digest", "", false)
	stale("asserted replacement", "", sha256Hex([]byte("original")), false)
	stale("warm replacement", "", "", true)
	stale("warm prefilled replacement", sha256Hex([]byte("original")), "", true)

	for _, prefilled := range []bool{false, true} {
		t.Run(fmt.Sprintf("native capture same-stat replacement prefilled=%t", prefilled), func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "model")
			writeTestFile(t, path, []byte("original"))
			target, info, err := confinedRegularFile(dir, path)
			if err != nil {
				t.Fatal(err)
			}
			// Do not call SameFile on info here: the native capture owner must
			// freeze Windows' lazy identity before the pathname is replaced.
			e := sourceEvidence{configured: path, target: target, info: info, size: info.Size(), mtime: info.ModTime().UnixNano()}
			replacement := filepath.Join(dir, "replacement")
			writeTestFile(t, replacement, []byte("replaced"))
			if err := os.Chtimes(replacement, info.ModTime(), info.ModTime()); err != nil {
				t.Fatal(err)
			}
			replace(t, e, replacement)
			c, counts := observe(e, nil, nil)
			list := listFor(e)
			if prefilled {
				list.Files[0].SHA256 = sha256Hex([]byte("original"))
			}
			if err := c.hashModelFiles(context.Background(), &list, nil); err == nil {
				t.Error("native capture accepted a same-stat replacement")
			}
			assertCounts(t, counts, 0, 0)
		})
	}

	t.Run("replacement cache cannot authorize an old request", func(t *testing.T) {
		e, replacement := fixture(t, []byte("original"))
		replace(t, e, replacement)
		c, counts := observe(e, nil, nil)
		// remember has legitimate current-destination authority, not authority
		// to refresh the older source request supplied below.
		c.remember(e.configured, sha256Hex([]byte("xxxxxxxx")))
		list := listFor(e)
		if err := c.hashModelFiles(context.Background(), &list, nil); err == nil {
			t.Error("new cache entry refreshed stale requesting evidence")
		}
		assertCounts(t, counts, 0, 0)
	})

	missing := func(name string, change func(*modelFile)) {
		t.Run(name, func(t *testing.T) {
			e, _ := fixture(t, []byte("original"))
			c, counts := observe(e, nil, nil)
			list := listFor(e)
			change(&list.Files[0])
			if err := c.hashModelFiles(context.Background(), &list, nil); err == nil {
				t.Error("missing or inconsistent original evidence was accepted")
			}
			assertCounts(t, counts, 0, 0)
		})
	}
	missing("missing FileInfo cannot recapture current authority", func(f *modelFile) { f.info = nil })
	missing("prefilled digest requires FileInfo", func(f *modelFile) { f.info, f.SHA256 = nil, "asserted-digest" })
	missing("missing configured path", func(f *modelFile) { f.configured = "" })
	missing("missing resolved target", func(f *modelFile) { f.abs = "" })
	missing("listed size disagrees with evidence", func(f *modelFile) { f.Size++ })

	t.Run("configured and resolved names must retain their original relationship", func(t *testing.T) {
		e, replacement := fixture(t, []byte("original"))
		c, counts := observe(e, nil, nil)
		list := listFor(e)
		list.Files[0].abs = replacement
		if err := c.hashModelFiles(context.Background(), &list, nil); err == nil {
			t.Error("unrelated resolved target accepted with original FileInfo")
		}
		assertCounts(t, counts, 0, 0)
	})

	t.Run("substituted checked handle closes before any read", func(t *testing.T) {
		e, replacement := fixture(t, []byte("original"))
		c, counts := observe(e, func(string) string { return replacement }, nil)
		list := listFor(e)
		if err := c.hashModelFiles(context.Background(), &list, nil); err == nil {
			t.Error("substituted opened object was accepted")
		}
		assertCounts(t, counts, 0, 1)
		assertUnpublished(t, c, list)
	})

	t.Run("cold stream and warm hit preserve digest and close the same handle", func(t *testing.T) {
		data := bytes.Repeat([]byte("model"), (2<<20)/5+1)
		e, _ := fixture(t, data)
		c, counts := observe(e, nil, nil)
		list := listFor(e)
		list.Files[0].asserted = sha256Hex(data)
		if err := c.hashModelFiles(context.Background(), &list, nil); err != nil || list.Files[0].SHA256 != sha256Hex(data) {
			t.Fatalf("cold hash: digest=%q error=%v", list.Files[0].SHA256, err)
		}
		list = listFor(e)
		if err := c.hashModelFiles(context.Background(), &list, nil); err != nil || list.Files[0].SHA256 != sha256Hex(data) {
			t.Fatalf("warm hash: digest=%q error=%v", list.Files[0].SHA256, err)
		}
		assertCounts(t, counts, e.size, 1)
		if counts.readAt.Load() < 2 {
			t.Error("streaming positive control did not span multiple reads")
		}
	})

	t.Run("unchanged prefilled digest is validated without reading", func(t *testing.T) {
		e, _ := fixture(t, []byte("original"))
		c, counts := observe(e, nil, nil)
		list := listFor(e)
		list.Files[0].SHA256, list.Files[0].asserted = sha256Hex([]byte("original")), sha256Hex([]byte("original"))
		if err := c.hashModelFiles(context.Background(), &list, nil); err != nil {
			t.Fatal(err)
		}
		assertCounts(t, counts, 0, 0)
	})

	mismatch := func(name string, prefilled bool) {
		t.Run(name, func(t *testing.T) {
			e, _ := fixture(t, []byte("original"))
			c, counts := observe(e, nil, nil)
			list := listFor(e)
			list.Files[0].asserted = sha256Hex([]byte("wrong"))
			var wantBytes, wantOpens int64
			if prefilled {
				list.Files[0].SHA256 = sha256Hex([]byte("original"))
			} else {
				wantBytes, wantOpens = e.size, 1
			}
			if err := c.hashModelFiles(context.Background(), &list, nil); err == nil {
				t.Error("asserted digest disagreement was accepted")
			}
			if !prefilled && list.Files[0].SHA256 != "" {
				t.Error("unaccepted digest was published in the list")
			}
			assertCounts(t, counts, wantBytes, wantOpens)
		})
	}
	mismatch("streamed digest must match asserted digest", false)
	mismatch("prefilled digest must match asserted digest", true)

	t.Run("empty native sidecar hashes and closes without content reads", func(t *testing.T) {
		e, _ := fixture(t, nil)
		c, counts := observe(e, nil, nil)
		list := listFor(e)
		if err := c.hashModelFiles(context.Background(), &list, nil); err != nil || list.Files[0].SHA256 != sha256Hex(nil) {
			t.Fatalf("empty sidecar: digest=%q error=%v", list.Files[0].SHA256, err)
		}
		assertCounts(t, counts, 0, 1)
	})

	t.Run("changed source between chunks refuses before another read", func(t *testing.T) {
		data := bytes.Repeat([]byte("o"), 2<<20)
		e, _ := fixture(t, data)
		c, counts := observe(e, nil, func(f *linkedHashFile) {
			f.afterRead = func(n int, off int64) {
				if off == 0 {
					changed := e.info.ModTime().Add(time.Second)
					if err := os.Chtimes(e.configured, changed, changed); err != nil {
						t.Fatal(err)
					}
				}
			}
		})
		list := listFor(e)
		if err := c.hashModelFiles(context.Background(), &list, nil); err == nil {
			t.Error("changed source was accepted")
		}
		assertCounts(t, counts, 1<<20, 1)
		if counts.readAt.Load() != 1 {
			t.Errorf("ReadAt attempts=%d, want only the original chunk", counts.readAt.Load())
		}
		assertUnpublished(t, c, list)
	})

	t.Run("final opened handle identity is checked independently of path", func(t *testing.T) {
		e, replacement := fixture(t, []byte("original"))
		other, err := captureSource(replacement, false)
		if err != nil {
			t.Fatal(err)
		}
		c, counts := observe(e, nil, func(f *linkedHashFile) {
			read := false
			f.afterRead = func(int, int64) { read = true }
			f.stat = func() (os.FileInfo, error) {
				if read {
					return other.info, nil
				}
				return f.File.Stat()
			}
		})
		list := listFor(e)
		if err := c.hashModelFiles(context.Background(), &list, nil); err == nil {
			t.Error("changed final opened identity was accepted")
		}
		assertCounts(t, counts, e.size, 1)
		assertUnpublished(t, c, list)
	})

	t.Run("final original evidence is checked after the last chunk", func(t *testing.T) {
		e, _ := fixture(t, []byte("original"))
		c, counts := observe(e, nil, func(f *linkedHashFile) {
			// Keep the handle's reported metadata unchanged to isolate the
			// original configured-path check from the final handle check.
			f.stat = func() (os.FileInfo, error) { return e.info, nil }
		})
		_, err := c.sumBound(context.Background(), e, func(int64) {
			changed := e.info.ModTime().Add(time.Second)
			if err := os.Chtimes(e.configured, changed, changed); err != nil {
				t.Fatal(err)
			}
		})
		if err == nil {
			t.Error("last-chunk change was accepted")
		}
		assertCounts(t, counts, e.size, 1)
		assertUnpublished(t, c, listFor(e))
	})

	t.Run("short read cannot publish a partial digest", func(t *testing.T) {
		e, _ := fixture(t, []byte("original"))
		c, counts := observe(e, nil, func(f *linkedHashFile) {
			f.beforeRead = func() {
				if err := os.Truncate(e.configured, 1); err != nil {
					t.Fatal(err)
				}
			}
		})
		list := listFor(e)
		if err := c.hashModelFiles(context.Background(), &list, nil); err == nil {
			t.Error("partial digest was accepted")
		}
		assertCounts(t, counts, 1, 1)
		assertUnpublished(t, c, list)
	})

	t.Run("canceled request does not open even an unchanged source", func(t *testing.T) {
		e, _ := fixture(t, []byte("original"))
		c, counts := observe(e, nil, nil)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := c.sumBound(ctx, e, nil); !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled hash: %v", err)
		}
		assertCounts(t, counts, 0, 0)
	})

	t.Run("cancellation after the last chunk closes without caching", func(t *testing.T) {
		e, _ := fixture(t, []byte("original"))
		c, counts := observe(e, nil, nil)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		if got, err := c.sumBound(ctx, e, func(int64) { cancel() }); !errors.Is(err, context.Canceled) || got != "" {
			t.Errorf("last-chunk cancellation: sum=%q error=%v", got, err)
		}
		assertCounts(t, counts, e.size, 1)
		assertUnpublished(t, c, listFor(e))
	})

	// Inject completed work only for boundary cases that need a deterministic
	// replacement between joining an existing call and receiving its result.
	inflightReplacement := func(name string, resultErr error) {
		t.Run(name, func(t *testing.T) {
			e, replacement := fixture(t, []byte("original"))
			c, counts := observe(e, nil, nil)
			call := &hashCall{done: make(chan struct{}), info: e.info, sum: sha256Hex([]byte("original")), err: resultErr}
			key := hashEntry{path: e.target, size: e.size, mtime: e.mtime}
			c.inflight = map[hashEntry]*hashCall{key: call}
			call.onWait = func() {
				replace(t, e, replacement)
				delete(c.inflight, key)
				close(call.done)
			}
			list := listFor(e)
			if err := c.hashModelFiles(context.Background(), &list, nil); err == nil {
				t.Error("inflight result or retry refreshed stale authority")
			}
			assertCounts(t, counts, 0, 0)
			assertUnpublished(t, c, list)
		})
	}
	inflightReplacement("inflight success validates the waiting request again", nil)
	inflightReplacement("canceled leader retry refuses a same-stat new object", context.Canceled)
	inflightReplacement("expired leader retry refuses a same-stat new object", context.DeadlineExceeded)

	t.Run("inflight object mismatch refuses without reading", func(t *testing.T) {
		e, replacement := fixture(t, []byte("original"))
		other, err := captureSource(replacement, false)
		if err != nil {
			t.Fatal(err)
		}
		c, counts := observe(e, nil, nil)
		c.inflight = map[hashEntry]*hashCall{{path: e.target, size: e.size, mtime: e.mtime}: {done: make(chan struct{}), info: other.info}}
		if _, err := c.sumBound(context.Background(), e, nil); err == nil {
			t.Error("unrelated in-flight object accepted")
		}
		assertCounts(t, counts, 0, 0)
	})

	concurrent := func(name string, cancelLeader, replaceObject, cancelWaiter bool) {
		t.Run(name, func(t *testing.T) {
			data := []byte("original")
			e, replacement := fixture(t, data)
			entered, release := make(chan struct{}), make(chan struct{})
			var first, releaseOnce sync.Once
			c, counts := observe(e, nil, func(f *linkedHashFile) {
				f.beforeRead = func() {
					first.Do(func() { close(entered); <-release })
				}
			})
			type result struct {
				sum string
				err error
			}
			leaderCtx, stopLeader := context.WithCancel(context.Background())
			defer stopLeader()
			leader := make(chan result, 1)
			leaderDone := make(chan struct{})
			go func() {
				defer close(leaderDone)
				sum, err := c.sumBound(leaderCtx, e, nil)
				leader <- result{sum, err}
			}()
			t.Cleanup(func() {
				releaseOnce.Do(func() { close(release) })
				<-leaderDone
			})
			select {
			case <-entered:
			case got := <-leader:
				t.Fatalf("leader returned before the streaming positive control: sum=%q error=%v", got.sum, got.err)
			}
			joined, resume := make(chan struct{}), make(chan struct{})
			var resumeOnce sync.Once
			t.Cleanup(func() { resumeOnce.Do(func() { close(resume) }) })
			key := hashEntry{path: e.target, size: e.size, mtime: e.mtime}
			c.mu.Lock()
			c.inflight[key].onWait = func() { close(joined); <-resume }
			c.mu.Unlock()
			waiterCtx, stopWaiter := context.WithCancel(context.Background())
			defer stopWaiter()
			waiter := make(chan result, 1)
			waiterDone := make(chan struct{})
			go func() {
				defer close(waiterDone)
				sum, err := c.sumBound(waiterCtx, e, nil)
				waiter <- result{sum, err}
			}()
			t.Cleanup(func() {
				releaseOnce.Do(func() { close(release) })
				resumeOnce.Do(func() { close(resume) })
				<-waiterDone
			})
			select {
			case <-joined:
			case got := <-waiter:
				t.Fatalf("waiter did not join the in-flight hash: sum=%q error=%v", got.sum, got.err)
			}
			if cancelLeader {
				stopLeader()
			}
			if cancelWaiter {
				stopWaiter()
				resumeOnce.Do(func() { close(resume) })
				got := <-waiter
				if !errors.Is(got.err, context.Canceled) || got.sum != "" {
					t.Errorf("canceled waiter: sum=%q error=%v", got.sum, got.err)
				}
			}
			releaseOnce.Do(func() { close(release) })
			lead := <-leader
			if cancelLeader {
				if !errors.Is(lead.err, context.Canceled) || lead.sum != "" {
					t.Errorf("canceled leader: sum=%q error=%v", lead.sum, lead.err)
				}
			} else if lead.err != nil || lead.sum != sha256Hex(data) {
				t.Errorf("leader: sum=%q error=%v", lead.sum, lead.err)
			}
			if replaceObject {
				// The first handle has closed, so this replacement is portable
				// to Windows and cannot depend on Unix unlink-open semantics.
				replace(t, e, replacement)
			}
			resumeOnce.Do(func() { close(resume) })
			if !cancelWaiter {
				got := <-waiter
				if replaceObject {
					if got.err == nil || got.sum != "" {
						t.Errorf("stale waiter: sum=%q error=%v", got.sum, got.err)
					}
				} else if got.err != nil || got.sum != sha256Hex(data) {
					t.Errorf("unchanged waiter: sum=%q error=%v", got.sum, got.err)
				}
			}
			wantOpens := int64(1)
			if cancelLeader && !replaceObject && !cancelWaiter {
				wantOpens++
			}
			assertCounts(t, counts, wantOpens*e.size, wantOpens)
			if cancelLeader && replaceObject {
				assertUnpublished(t, c, listFor(e))
			}
		})
	}
	concurrent("unchanged concurrent requests share one stream", false, false, false)
	concurrent("unchanged waiter retries a canceled leader", true, false, false)
	concurrent("real canceled leader cannot refresh waiting authority", true, true, false)
	concurrent("canceled waiter leaves the leader usable", false, false, true)

	t.Run("path wrappers hash current destination objects and remembered IDs stay pinned", func(t *testing.T) {
		data := []byte("original")
		e, replacement := fixture(t, data)
		c, counts := observe(e, nil, nil)
		if got, err := hashFile(context.Background(), e.configured, nil); err != nil || got != sha256Hex(data) {
			t.Fatalf("uncached wrapper: sum=%q error=%v", got, err)
		}
		if got, err := c.sum(context.Background(), e.configured, nil); err != nil || got != sha256Hex(data) {
			t.Fatalf("cached wrapper: sum=%q error=%v", got, err)
		}
		assertCounts(t, counts, e.size, 1)
		// Do not inspect this entry with SameFile before replacing the path.
		// remember itself must pin Windows' otherwise lazy file identity.
		remembered := &hashCache{}
		remembered.remember(e.configured, sha256Hex(data))
		replace(t, e, replacement)
		if got, err := hashFile(context.Background(), e.configured, nil); err != nil || got != sha256Hex([]byte("xxxxxxxx")) {
			t.Fatalf("new current object wrapper: sum=%q error=%v", got, err)
		}
		if got, err := remembered.sum(context.Background(), e.configured, nil); err != nil || got != sha256Hex([]byte("xxxxxxxx")) {
			t.Fatalf("same-stat destination replacement kept remembered digest: sum=%q error=%v", got, err)
		}
		remembered.remember(e.configured, "verified destination digest")
		if got, err := remembered.sum(context.Background(), e.configured, nil); err != nil || got != "verified destination digest" {
			t.Fatalf("unchanged destination was rehashed: sum=%q error=%v", got, err)
		}
	})

	t.Run("path wrappers preserve cancellation and missing-file errors", func(t *testing.T) {
		e, _ := fixture(t, []byte("original"))
		c, counts := observe(e, nil, nil)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		for name, hash := range map[string]func(context.Context, string, func(int64)) (string, error){
			"hashFile": hashFile,
			"sum":      c.sum,
		} {
			t.Run(name, func(t *testing.T) {
				if _, err := hash(ctx, e.configured, nil); !errors.Is(err, context.Canceled) {
					t.Errorf("canceled wrapper: %v", err)
				}
				if _, err := hash(context.Background(), e.configured+".missing", nil); !errors.Is(err, os.ErrNotExist) {
					t.Errorf("missing file wrapper: %v", err)
				}
			})
		}
		assertCounts(t, counts, 0, 0)
	})
}
