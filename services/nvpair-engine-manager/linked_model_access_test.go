// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

type countedLinkedFile struct {
	*os.File
	reads      *int
	closes     *int
	streams    *int
	beforeRead func()
	afterRead  func(int, int64)
	statError  error
}

type countedLinkedDir struct {
	*os.File
	reads, closes *int
	beforeRead    func(int)
	afterRead     func([]os.DirEntry, error)
}

func (f *countedLinkedDir) ReadDir(n int) ([]os.DirEntry, error) {
	*f.reads++
	if f.beforeRead != nil {
		f.beforeRead(n)
	}
	entries, err := f.File.ReadDir(n)
	if f.afterRead != nil {
		f.afterRead(entries, err)
	}
	return entries, err
}

func (f *countedLinkedDir) Close() error {
	*f.closes++
	return f.File.Close()
}

func (f *countedLinkedFile) ReadAt(p []byte, off int64) (int, error) {
	*f.reads++
	if f.beforeRead != nil {
		f.beforeRead()
	}
	n, err := f.File.ReadAt(p, off)
	if f.afterRead != nil {
		f.afterRead(n, off)
	}
	return n, err
}

func (f *countedLinkedFile) Read(p []byte) (int, error) {
	if f.streams != nil {
		*f.streams++
	} else {
		*f.reads++
	}
	return f.File.Read(p)
}

func (f *countedLinkedFile) WriteTo(w io.Writer) (int64, error) {
	// Do not let os.File.WriteTo bypass the streaming counter.
	return io.Copy(w, struct{ io.Reader }{f})
}

func (f *countedLinkedFile) Stat() (os.FileInfo, error) {
	if f.statError != nil {
		return nil, f.statError
	}
	return f.File.Stat()
}

func (f *countedLinkedFile) Close() error {
	*f.closes++
	return f.File.Close()
}

func TestLinkedModelCheckedAccess(t *testing.T) {
	type counts struct {
		opens, reads, streams, closes int
	}
	track := func(t *testing.T, choose func(int, string) string, configure func(int, *countedLinkedFile)) (servingFileAccess, *counts) {
		t.Helper()
		c := &counts{}
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			c.opens++
			if choose != nil {
				path = choose(c.opens, path)
			}
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			observed := &countedLinkedFile{File: f, reads: &c.reads, streams: &c.streams, closes: &c.closes}
			if configure != nil {
				configure(c.opens, observed)
			}
			return observed, nil
		}}
		return access, c
	}
	trackDirectory := func(t *testing.T, choose func(string) string, configure func(*countedLinkedDir)) (servingFileAccess, *counts) {
		t.Helper()
		c := &counts{}
		access := servingFileAccess{openDir: func(path string) (servingDirectory, error) {
			c.opens++
			if choose != nil {
				path = choose(path)
			}
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			observed := &countedLinkedDir{File: f, reads: &c.reads, closes: &c.closes}
			if configure != nil {
				configure(observed)
			}
			return observed, nil
		}}
		return access, c
	}
	assertClosed := func(t *testing.T, c *counts) {
		t.Helper()
		if c.closes != c.opens || c.streams != 0 {
			t.Errorf("opens=%d closes=%d direct stream reads=%d", c.opens, c.closes, c.streams)
		}
	}
	fixture := func(t *testing.T, data []byte) (string, string, sourceEvidence) {
		t.Helper()
		dir := t.TempDir()
		path, other := filepath.Join(dir, "selected"), filepath.Join(dir, "unselected")
		for _, p := range []string{path, other} {
			if err := os.WriteFile(p, data, 0o600); err != nil {
				t.Fatal(err)
			}
		}
		e, err := captureSource(path, false)
		if err != nil {
			t.Fatal(err)
		}
		return path, other, e
	}
	changeTime := func(t *testing.T, e sourceEvidence) {
		t.Helper()
		changed := e.info.ModTime().Add(time.Second)
		if err := os.Chtimes(e.configured, changed, changed); err != nil {
			t.Fatal(err)
		}
	}
	linkDirectory := func(t *testing.T, link, target string) {
		t.Helper()
		if runtime.GOOS == "windows" {
			output, err := exec.Command("cmd.exe", "/d", "/c", "mklink", "/J", link, target).CombinedOutput()
			if err != nil {
				t.Fatalf("create junction: %v: %s", err, output)
			}
			return
		}
		if err := os.Symlink(target, link); err != nil {
			t.Skipf("directory symlink unavailable: %v", err)
		}
	}

	t.Run("substituted open refuses without reading and closes", func(t *testing.T) {
		dir := t.TempDir()
		path, other := filepath.Join(dir, "source"), filepath.Join(dir, "other")
		for _, p := range []string{path, other} {
			if err := os.WriteFile(p, []byte("content"), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		reads, closes := 0, 0
		access := servingFileAccess{open: func(string) (servingArtifactFile, error) {
			f, err := os.Open(other)
			if err != nil {
				return nil, err
			}
			return &countedLinkedFile{File: f, reads: &reads, closes: &closes}, nil
		}}
		if _, err := access.binding(context.Background(), path); err == nil {
			t.Error("substitution accepted")
		}
		if reads != 0 || closes != 1 {
			t.Errorf("reads=%d closes=%d", reads, closes)
		}
	})

	t.Run("directory substitution refuses before enumeration", func(t *testing.T) {
		root := t.TempDir()
		path, other := filepath.Join(root, "source"), filepath.Join(root, "other")
		for _, p := range []string{path, other} {
			if err := os.Mkdir(p, 0o700); err != nil {
				t.Fatal(err)
			}
		}
		e, err := captureSource(path, true)
		if err != nil {
			t.Fatal(err)
		}
		reads, closes := 0, 0
		access := servingFileAccess{openDir: func(string) (servingDirectory, error) {
			f, err := os.Open(other)
			if err != nil {
				return nil, err
			}
			return &countedLinkedDir{File: f, reads: &reads, closes: &closes}, nil
		}}
		if _, err := access.readDirectory(context.Background(), e, 10); err == nil {
			t.Error("substitution accepted")
		}
		if reads != 0 || closes != 1 {
			t.Errorf("enumerations=%d closes=%d", reads, closes)
		}
	})

	t.Run("directory control and cancellation", func(t *testing.T) {
		path := t.TempDir()
		if err := os.WriteFile(filepath.Join(path, "entry"), nil, 0o600); err != nil {
			t.Fatal(err)
		}
		e, err := captureSource(path, true)
		if err != nil {
			t.Fatal(err)
		}
		access, c := trackDirectory(t, nil, nil)
		defer assertClosed(t, c)
		entries, err := access.readDirectory(context.Background(), e, 1)
		if err != nil || len(entries) != 1 {
			t.Fatalf("entries=%v err=%v", entries, err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := access.readDirectory(ctx, e, 1); err == nil {
			t.Error("cancelled enumeration succeeded")
		}
	})

	t.Run("empty selector and bounded streaming", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "selector")
		if err := os.WriteFile(path, nil, 0o600); err != nil {
			t.Fatal(err)
		}
		access, c := track(t, nil, nil)
		defer assertClosed(t, c)
		data, _, err := access.sourceLimited(context.Background(), path, 4<<20, true)
		if err != nil || len(data) != 0 {
			t.Fatalf("empty selector: %v, %v", data, err)
		}
		if _, _, err := access.source(context.Background(), path); err == nil {
			t.Error("empty serving source accepted")
		}
	})
	t.Run("missing expected evidence refuses before opening", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "selector")
		if err := os.WriteFile(path, []byte("selector"), 0600); err != nil {
			t.Fatal(err)
		}
		reads, closes, opens := 0, 0, 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			opens++
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			return &countedLinkedFile{File: f, reads: &reads, closes: &closes}, nil
		}}
		_, err := access.bindingExpected(context.Background(), path, "", nil)
		if err == nil {
			t.Error("missing expected evidence accepted")
		}
		if opens != 0 || reads != 0 {
			t.Errorf("opened %d and read %d times before refusal", opens, reads)
		}
		if closes != opens {
			t.Errorf("closed %d of %d handles", closes, opens)
		}
	})

	t.Run("oversized source refuses before sampling", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "selector")
		if err := os.WriteFile(path, make([]byte, servingSmallFileLimit+1), 0600); err != nil {
			t.Fatal(err)
		}
		reads, closes, opens := 0, 0, 0
		access := servingFileAccess{open: func(path string) (servingArtifactFile, error) {
			opens++
			f, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			return &countedLinkedFile{File: f, reads: &reads, closes: &closes}, nil
		}}
		_, _, err := access.source(context.Background(), path)
		if err == nil {
			t.Error("oversized source accepted")
		}
		if reads != 0 {
			t.Errorf("sampled oversized source %d times", reads)
		}
		if closes != opens {
			t.Errorf("closed %d of %d handles", closes, opens)
		}
	})

	beforeOpen := func(name string, mutate func(*testing.T, string, string, sourceEvidence)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			path, other, e := fixture(t, []byte("selected bytes"))
			mutate(t, path, other, e)
			access, c := track(t, nil, nil)
			defer assertClosed(t, c)
			if _, err := access.bindingExpected(context.Background(), path, e.target, e.info); err == nil {
				t.Error("changed expected binding accepted")
			}
			if _, _, err := access.sourceExpected(context.Background(), e, servingSmallFileLimit, false); err == nil {
				t.Error("changed expected source accepted")
			}
			if f, err := access.checkedOpen(context.Background(), path, e.target, e.info); err == nil {
				if err := f.Close(); err != nil {
					t.Error(err)
				}
				t.Error("changed checked open accepted")
			}
			if c.opens != 0 || c.reads != 0 {
				t.Errorf("opened %d and read %d times before refusal", c.opens, c.reads)
			}
		})
	}
	beforeOpen("replacement identity refuses before open", func(t *testing.T, path, other string, _ sourceEvidence) {
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.Rename(other, path); err != nil {
			t.Fatal(err)
		}
	})
	beforeOpen("changed metadata refuses before open", func(t *testing.T, _, _ string, e sourceEvidence) {
		changeTime(t, e)
	})
	beforeOpen("equal metadata replacement refuses before open", func(t *testing.T, path, other string, e sourceEvidence) {
		if err := os.WriteFile(other, []byte("replaced bytes"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.Rename(other, path); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, e.info.ModTime(), e.info.ModTime()); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("equal directory metadata replacement refuses before enumeration", func(t *testing.T) {
		root := t.TempDir()
		path, replacement := filepath.Join(root, "selected"), filepath.Join(root, "replacement")
		for _, p := range []string{path, replacement} {
			if err := os.Mkdir(p, 0o700); err != nil {
				t.Fatal(err)
			}
		}
		e, err := captureSource(path, true)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.Rename(replacement, path); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, e.info.ModTime(), e.info.ModTime()); err != nil {
			t.Fatal(err)
		}
		access, c := trackDirectory(t, nil, nil)
		defer assertClosed(t, c)
		if _, err := access.readDirectory(context.Background(), e, 0); err == nil {
			t.Error("replacement directory accepted under original evidence")
		}
		if c.opens != 0 || c.reads != 0 {
			t.Errorf("opened %d and enumerated %d times before refusal", c.opens, c.reads)
		}
	})
	beforeOpen("changed type refuses before open", func(t *testing.T, path, _ string, _ sourceEvidence) {
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(path, 0o700); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("target mismatch refuses even when object identity matches", func(t *testing.T) {
		path, other, e := fixture(t, []byte("selector"))
		wrongTarget, err := resolveExistingPath(other)
		if err != nil {
			t.Fatal(err)
		}
		e.target = wrongTarget
		access, c := track(t, nil, nil)
		defer assertClosed(t, c)
		if _, _, err := access.sourceExpected(context.Background(), e, 4<<20, true); err == nil {
			t.Error("mismatched expected target accepted")
		}
		if _, err := access.bindingExpected(context.Background(), path, wrongTarget, e.info); err == nil {
			t.Error("binding ignored expected target")
		}
		if c.opens != 0 || c.reads != 0 {
			t.Errorf("target mismatch opened=%d read=%d", c.opens, c.reads)
		}
	})

	substitutedOpen := func(name string, at int) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			path, other, _ := fixture(t, []byte("selected bytes"))
			forbidden := 0
			access, c := track(t, func(n int, selected string) string {
				if n == at {
					return other
				}
				return selected
			}, func(n int, f *countedLinkedFile) {
				if n == at {
					f.reads, f.streams = &forbidden, &forbidden
				}
			})
			defer assertClosed(t, c)
			if _, _, err := access.source(context.Background(), path); err == nil {
				t.Error("substituted source accepted")
			}
			if c.opens != at || forbidden != 0 {
				t.Errorf("opens=%d want=%d forbidden ReadAt/stream calls=%d", c.opens, at, forbidden)
			}
		})
	}
	substitutedOpen("first sample substituted open closes without reading", 1)
	substitutedOpen("second stream substituted open closes without reading", 2)
	substitutedOpen("final sample substituted open closes without reading", 3)

	t.Run("initial source metadata is captured before the first sample", func(t *testing.T) {
		path, _, e := fixture(t, []byte("selector"))
		access, c := track(t, nil, func(_ int, _ *countedLinkedFile) {
			changeTime(t, e)
		})
		defer assertClosed(t, c)
		if _, _, err := access.source(context.Background(), path); err == nil {
			t.Error("initial metadata change accepted")
		}
		if c.opens != 1 || c.reads != 0 {
			t.Errorf("opens=%d reads=%d before initial metadata refusal", c.opens, c.reads)
		}
	})

	midRead := func(name string, at int) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			path, _, e := fixture(t, bytes.Repeat([]byte("a"), 3*servingSampleSize))
			changed, forbidden := false, 0
			access, c := track(t, nil, func(n int, f *countedLinkedFile) {
				f.beforeRead = func() {
					if changed {
						forbidden++
					}
				}
				if n == at {
					f.afterRead = func(int, int64) {
						if !changed {
							changeTime(t, e)
							changed = true
						}
					}
				}
			})
			defer assertClosed(t, c)
			if _, _, err := access.source(context.Background(), path); err == nil {
				t.Error("mid-read metadata change accepted")
			}
			if !changed || c.opens != at || forbidden != 0 {
				t.Errorf("mutation reached=%v opens=%d want=%d forbidden calls=%d", changed, c.opens, at, forbidden)
			}
		})
	}
	midRead("metadata drift stops the next sample ReadAt", 1)
	midRead("metadata drift stops the next streaming ReadAt", 2)
	midRead("metadata drift during final sample refuses publication", 3)

	t.Run("final sample detects same-stat content change after stream", func(t *testing.T) {
		path, _, e := fixture(t, []byte("selected bytes"))
		changed := false
		access, c := track(t, nil, func(n int, f *countedLinkedFile) {
			if n == 2 {
				f.afterRead = func(n int, offset int64) {
					if offset+int64(n) != e.size {
						return
					}
					if err := os.WriteFile(path, []byte("mutated! bytes"), 0o600); err != nil {
						t.Fatal(err)
					}
					if err := os.Chtimes(path, e.info.ModTime(), e.info.ModTime()); err != nil {
						t.Fatal(err)
					}
					changed = true
				}
			}
		})
		defer assertClosed(t, c)
		usage := &servingReadUsage{}
		access.usage = usage
		if _, _, err := access.source(context.Background(), path); err == nil {
			t.Error("same-stat changed source accepted")
		}
		if !changed || c.opens != 3 || usage.SourceBytes != uint64(e.size) || usage.SampleBytes != 2*uint64(e.size) {
			t.Errorf("changed=%v opens=%d usage=%+v", changed, c.opens, usage)
		}
	})

	t.Run("caller four MiB limit permits selector beyond serving limit", func(t *testing.T) {
		data := bytes.Repeat([]byte("s"), 4<<20)
		path, _, e := fixture(t, data)
		access, c := track(t, nil, nil)
		defer assertClosed(t, c)
		usage := &servingReadUsage{}
		access.usage = usage
		got, binding, err := access.sourceExpected(context.Background(), e, 4<<20, true)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, data) || binding.SizeBytes != int64(len(data)) || c.opens != 3 {
			t.Fatalf("selector bytes or binding differ: bytes=%d binding=%+v opens=%d", len(got), binding, c.opens)
		}
		if usage.SourceBytes != uint64(len(data)) || usage.SampleBytes != 4*servingSampleSize || usage.HeaderBytes != 0 {
			t.Errorf("usage=%+v", usage)
		}
		before := *c
		if _, _, err := access.source(context.Background(), path); err == nil {
			t.Error("four MiB selector accepted as serving source")
		}
		if *c != before {
			t.Errorf("serving size refusal performed content access: before=%+v after=%+v", before, *c)
		}
	})

	t.Run("empty sidecar allowed but serving and artifact bindings remain nonempty", func(t *testing.T) {
		path, _, e := fixture(t, nil)
		access, c := track(t, nil, nil)
		defer assertClosed(t, c)
		data, binding, err := access.sourceExpected(context.Background(), e, 0, true)
		if err != nil || len(data) != 0 || binding == nil || binding.SizeBytes != 0 {
			t.Fatalf("empty sidecar data=%v binding=%+v error=%v", data, binding, err)
		}
		if c.opens != 3 || c.reads != 0 {
			t.Errorf("empty sidecar opens=%d reads=%d", c.opens, c.reads)
		}
		before := *c
		if _, _, err := access.source(context.Background(), path); err == nil {
			t.Error("empty serving source accepted")
		}
		if _, err := access.binding(context.Background(), path); err == nil {
			t.Error("empty model artifact accepted")
		}
		if *c != before {
			t.Error("empty serving or artifact refusal opened or read content")
		}
	})

	t.Run("missing expected selector record never recaptures authority", func(t *testing.T) {
		_, _, e := fixture(t, []byte("selector"))
		access, c := track(t, nil, nil)
		defer assertClosed(t, c)
		for _, expected := range []sourceEvidence{{}, {configured: e.configured}, {configured: e.configured, target: e.target}, {configured: e.configured, target: e.target, info: e.info}} {
			if _, _, err := access.sourceExpected(context.Background(), expected, 4<<20, true); err == nil {
				t.Error("incomplete selector evidence accepted")
			}
		}
		if _, err := access.bindingExpected(context.Background(), e.configured, "", e.info); err == nil {
			t.Error("missing expected target accepted")
		}
		if f, err := access.checkedOpen(context.Background(), e.configured, e.target, nil); err == nil {
			if err := f.Close(); err != nil {
				t.Error(err)
			}
			t.Error("missing expected opened identity accepted")
		}
		if c.opens != 0 || c.reads != 0 {
			t.Errorf("missing evidence opened=%d read=%d", c.opens, c.reads)
		}
	})

	t.Run("final file metadata validation runs after the last sample read", func(t *testing.T) {
		path, _, e := fixture(t, []byte("selector"))
		changed := false
		access, c := track(t, nil, func(n int, f *countedLinkedFile) {
			if n == 3 {
				f.afterRead = func(int, int64) {
					changeTime(t, e)
					changed = true
				}
			}
		})
		defer assertClosed(t, c)
		if _, _, err := access.source(context.Background(), path); err == nil {
			t.Error("changed final file metadata published")
		}
		if !changed || c.opens != 3 || c.reads != 3 {
			t.Errorf("changed=%v opens=%d reads=%d, want final sample reached", changed, c.opens, c.reads)
		}
	})

	t.Run("opened stat failure closes without reading", func(t *testing.T) {
		path, _, _ := fixture(t, []byte("selector"))
		access, c := track(t, nil, func(_ int, f *countedLinkedFile) {
			f.statError = errors.New("synthetic stat failure")
		})
		defer assertClosed(t, c)
		if _, _, err := access.source(context.Background(), path); err == nil {
			t.Error("failed opened stat accepted")
		}
		if c.opens != 1 || c.reads != 0 {
			t.Errorf("opens=%d reads=%d", c.opens, c.reads)
		}
	})

	cancelRead := func(name string, at int) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			path, _, _ := fixture(t, bytes.Repeat([]byte("a"), 3*servingSampleSize))
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			cancelled, forbidden := false, 0
			access, c := track(t, nil, func(n int, f *countedLinkedFile) {
				f.beforeRead = func() {
					if cancelled {
						forbidden++
					}
				}
				if n == at {
					f.afterRead = func(int, int64) {
						cancelled = true
						cancel()
					}
				}
			})
			defer assertClosed(t, c)
			if _, _, err := access.source(ctx, path); err == nil {
				t.Error("cancelled read succeeded")
			}
			if !cancelled || c.opens != at || forbidden != 0 {
				t.Errorf("cancelled=%v opens=%d want=%d forbidden reads=%d", cancelled, c.opens, at, forbidden)
			}
		})
	}
	cancelRead("cancellation during sample closes and stops reads", 1)
	cancelRead("cancellation during stream closes and stops reads", 2)
	cancelRead("cancellation during final sample closes and stops reads", 3)

	t.Run("cancelled operations do not open handles", func(t *testing.T) {
		path, _, e := fixture(t, []byte("selector"))
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		access, c := track(t, nil, nil)
		defer assertClosed(t, c)
		if _, _, err := access.source(ctx, path); !errors.Is(err, context.Canceled) {
			t.Errorf("source error=%v", err)
		}
		if _, err := access.bindingExpected(ctx, path, e.target, e.info); !errors.Is(err, context.Canceled) {
			t.Errorf("binding error=%v", err)
		}
		if c.opens != 0 || c.reads != 0 {
			t.Errorf("cancelled file opens=%d reads=%d", c.opens, c.reads)
		}
	})

	t.Run("cancellation after second open closes without streaming", func(t *testing.T) {
		path, _, _ := fixture(t, []byte("selector"))
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		access, c := track(t, nil, func(n int, _ *countedLinkedFile) {
			if n == 2 {
				cancel()
			}
		})
		defer assertClosed(t, c)
		if _, _, err := access.source(ctx, path); err == nil {
			t.Error("cancelled reopened source accepted")
		}
		if c.opens != 2 || c.reads != 1 {
			t.Errorf("opens=%d reads=%d, want only the first sample read", c.opens, c.reads)
		}
	})

	t.Run("linked file and directory retain target evidence before open", func(t *testing.T) {
		root := t.TempDir()
		first, second, link := filepath.Join(root, "first"), filepath.Join(root, "second"), filepath.Join(root, "selected")
		for _, dir := range []string{first, second} {
			if err := os.Mkdir(dir, 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "selector"), []byte("selector"), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		linkDirectory(t, link, first)
		fileEvidence, err := captureSource(filepath.Join(link, "selector"), false)
		if err != nil {
			t.Fatal(err)
		}
		dirEvidence, err := captureSource(link, true)
		if err != nil {
			t.Fatal(err)
		}
		access, c := track(t, nil, nil)
		defer assertClosed(t, c)
		got, _, err := access.sourceExpected(context.Background(), fileEvidence, 4<<20, true)
		if err != nil || string(got) != "selector" {
			t.Fatalf("linked selector=%q error=%v", got, err)
		}
		dirAccess, dc := trackDirectory(t, nil, nil)
		defer assertClosed(t, dc)
		entries, err := dirAccess.readDirectory(context.Background(), dirEvidence, 1)
		if err != nil || len(entries) != 1 || entries[0].Name() != "selector" {
			t.Fatalf("linked entries=%v error=%v", entries, err)
		}
		before, dirBefore := *c, *dc
		if err := os.Remove(link); err != nil {
			t.Fatal(err)
		}
		linkDirectory(t, link, second)
		if _, _, err := access.sourceExpected(context.Background(), fileEvidence, 4<<20, true); err == nil {
			t.Error("retargeted file accepted")
		}
		if _, err := dirAccess.readDirectory(context.Background(), dirEvidence, 1); err == nil {
			t.Error("retargeted directory accepted")
		}
		if *c != before || *dc != dirBefore {
			t.Errorf("retargeted source opened or read: file=%+v directory=%+v", c, dc)
		}
	})

	t.Run("directory metadata admission precedes open", func(t *testing.T) {
		path := t.TempDir()
		e, err := captureSource(path, true)
		if err != nil {
			t.Fatal(err)
		}
		changeTime(t, e)
		access, c := trackDirectory(t, nil, nil)
		defer assertClosed(t, c)
		if _, err := access.readDirectory(context.Background(), e, 1); err == nil {
			t.Error("changed directory metadata accepted")
		}
		if _, err := access.readDirectory(context.Background(), sourceEvidence{directory: true}, 1); err == nil {
			t.Error("missing directory evidence accepted")
		}
		if c.opens != 0 || c.reads != 0 {
			t.Errorf("directory opened=%d enumerated=%d before admission", c.opens, c.reads)
		}
	})

	t.Run("directory opener returning regular file refuses before enumeration", func(t *testing.T) {
		path, other, _ := fixture(t, []byte("selector"))
		e, err := captureSource(filepath.Dir(path), true)
		if err != nil {
			t.Fatal(err)
		}
		access, c := trackDirectory(t, func(string) string { return other }, nil)
		defer assertClosed(t, c)
		if _, err := access.readDirectory(context.Background(), e, 2); err == nil {
			t.Error("regular file enumerated as directory")
		}
		if c.opens != 1 || c.reads != 0 {
			t.Errorf("opens=%d enumerations=%d", c.opens, c.reads)
		}
	})

	t.Run("directory metadata changed by open refuses before first enumeration", func(t *testing.T) {
		e, err := captureSource(t.TempDir(), true)
		if err != nil {
			t.Fatal(err)
		}
		access, c := trackDirectory(t, nil, func(_ *countedLinkedDir) {
			changeTime(t, e)
		})
		defer assertClosed(t, c)
		if _, err := access.readDirectory(context.Background(), e, 0); err == nil {
			t.Error("opened directory metadata change accepted")
		}
		if c.opens != 1 || c.reads != 0 {
			t.Errorf("opens=%d enumerations=%d", c.opens, c.reads)
		}
	})

	t.Run("directory bound returns no partial entries", func(t *testing.T) {
		path := t.TempDir()
		for _, name := range []string{"a", "b", "c", "d"} {
			if err := os.WriteFile(filepath.Join(path, name), nil, 0o600); err != nil {
				t.Fatal(err)
			}
		}
		e, err := captureSource(path, true)
		if err != nil {
			t.Fatal(err)
		}
		returned := 0
		access, c := trackDirectory(t, nil, func(f *countedLinkedDir) {
			f.beforeRead = func(n int) {
				if n <= 0 || n > 3 {
					t.Errorf("unbounded ReadDir(%d)", n)
				}
			}
			f.afterRead = func(entries []os.DirEntry, _ error) { returned += len(entries) }
		})
		defer assertClosed(t, c)
		entries, err := access.readDirectory(context.Background(), e, 2)
		if err == nil || entries != nil || returned != 3 || c.reads != 1 {
			t.Errorf("partial=%v returned=%d reads=%d error=%v", entries, returned, c.reads, err)
		}
	})

	t.Run("zero directory bound probes only one entry", func(t *testing.T) {
		path := t.TempDir()
		if err := os.WriteFile(filepath.Join(path, "entry"), nil, 0o600); err != nil {
			t.Fatal(err)
		}
		e, err := captureSource(path, true)
		if err != nil {
			t.Fatal(err)
		}
		access, c := trackDirectory(t, nil, func(f *countedLinkedDir) {
			f.beforeRead = func(n int) {
				if n != 1 {
					t.Errorf("zero-bound ReadDir(%d), want 1", n)
				}
			}
		})
		defer assertClosed(t, c)
		if entries, err := access.readDirectory(context.Background(), e, 0); err == nil || entries != nil {
			t.Errorf("zero bound returned %v, error=%v", entries, err)
		}
	})

	t.Run("empty directory accepts zero bound and maximum integer bound", func(t *testing.T) {
		e, err := captureSource(t.TempDir(), true)
		if err != nil {
			t.Fatal(err)
		}
		access, c := trackDirectory(t, nil, func(f *countedLinkedDir) {
			f.beforeRead = func(n int) {
				if n <= 0 || n > 128 {
					t.Errorf("unbounded ReadDir(%d)", n)
				}
			}
		})
		defer assertClosed(t, c)
		for _, limit := range []int{0, int(^uint(0) >> 1)} {
			entries, err := access.readDirectory(context.Background(), e, limit)
			if err != nil || len(entries) != 0 {
				t.Errorf("limit=%d entries=%v error=%v", limit, entries, err)
			}
		}
		if c.opens != 2 || c.reads != 2 {
			t.Errorf("opens=%d enumerations=%d", c.opens, c.reads)
		}
	})

	directoryDrift := func(name string, final bool) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			path := t.TempDir()
			if err := os.WriteFile(filepath.Join(path, "entry"), nil, 0o600); err != nil {
				t.Fatal(err)
			}
			e, err := captureSource(path, true)
			if err != nil {
				t.Fatal(err)
			}
			changed, forbidden := false, 0
			access, c := trackDirectory(t, nil, func(f *countedLinkedDir) {
				f.beforeRead = func(int) {
					if changed {
						forbidden++
					}
				}
				f.afterRead = func(_ []os.DirEntry, err error) {
					if !changed && (!final || err == io.EOF) {
						changeTime(t, e)
						changed = true
					}
				}
			})
			defer assertClosed(t, c)
			entries, err := access.readDirectory(context.Background(), e, 1)
			if err == nil || entries != nil || !changed || forbidden != 0 {
				t.Errorf("entries=%v changed=%v forbidden=%d error=%v", entries, changed, forbidden, err)
			}
		})
	}
	directoryDrift("directory drift stops further enumeration", false)
	directoryDrift("final directory identity check refuses publication", true)

	t.Run("directory cancellation closes handle and stops enumeration", func(t *testing.T) {
		path := t.TempDir()
		e, err := captureSource(path, true)
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		access, c := trackDirectory(t, nil, func(f *countedLinkedDir) {
			f.afterRead = func([]os.DirEntry, error) { cancel() }
		})
		defer assertClosed(t, c)
		if entries, err := access.readDirectory(ctx, e, 0); !errors.Is(err, context.Canceled) || entries != nil {
			t.Errorf("entries=%v cancellation error=%v", entries, err)
		}
		if c.opens != 1 || c.reads != 1 {
			t.Errorf("opens=%d enumerations=%d", c.opens, c.reads)
		}
		before := *c
		if _, err := access.readDirectory(ctx, e, 0); !errors.Is(err, context.Canceled) {
			t.Errorf("already cancelled error=%v", err)
		}
		if *c != before {
			t.Error("already cancelled operation opened or enumerated directory")
		}
	})

	t.Run("snapshot carrier stays private on the existing wire", func(t *testing.T) {
		_, _, e := fixture(t, []byte("selector"))
		list := modelFileList{Engine: "engine", Model: "model", Files: []modelFile{}}
		before, err := json.Marshal(list)
		if err != nil {
			t.Fatal(err)
		}
		list.source = &sourceSnapshot{engine: "private engine", run: "private run", configuration: "private configuration", root: e.configured,
			recognition: []byte("private membership"), selectors: []selectorEvidence{{bytes: []byte("private selector"), file: e}},
			directories: []sourceEvidence{e}, artifacts: []modelFile{{Path: "private artifact", info: e.info, configured: e.configured, abs: e.target}}}
		after, err := json.Marshal(list)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(before, after) {
			t.Errorf("private snapshot changed wire: before=%s after=%s", before, after)
		}
	})
}
