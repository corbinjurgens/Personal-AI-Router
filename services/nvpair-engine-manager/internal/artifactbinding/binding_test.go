// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package artifactbinding

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"

	"nvpair-shared/modelconfig"
)

func syntheticEvidence(t *testing.T, data []byte) Evidence {
	t.Helper()
	p := filepath.Join(t.TempDir(), "synthetic")
	if err := os.WriteFile(p, data, 0600); err != nil {
		t.Fatal(err)
	}
	e, err := Capture(p)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestSampleHistoricalIdentity(t *testing.T) {
	data := append(bytes.Repeat([]byte("a"), SampleSize), bytes.Repeat([]byte("b"), 2*SampleSize)...)
	e := syntheticEvidence(t, data)
	var count uint64
	b, err := (Access{SampleBytes: &count}).Sample(context.Background(), e, false)
	if err != nil {
		t.Fatal(err)
	}
	expected := sha256.Sum256(append(append([]byte(nil), data[:SampleSize]...), data[len(data)-SampleSize:]...))
	if b.SourcePath != e.Configured || b.ResolvedTarget != e.Target || b.SizeBytes != e.Size || b.MtimeNS != strconv.FormatInt(e.Mtime, 10) || b.SampleSHA256 != hex.EncodeToString(expected[:]) || count != 2*SampleSize {
		t.Fatalf("historical first/last identity changed: %+v read=%d", b, count)
	}
}

func TestSummariesHistoricalProjection(t *testing.T) {
	bindings := []modelconfig.ArtifactBinding{{Role: "model", Path: "synthetic-link", ResolvedTarget: "synthetic-target", Size: "123", MtimeNS: "456", SampleSHA256: "sample"}, {Role: "draft", Path: "draft", ResolvedTarget: "draft-target", Size: "789", MtimeNS: "987", SampleSHA256: "other"}}
	var expected []modelconfig.ArtifactSummary
	for _, b := range bindings {
		data, err := json.Marshal(b)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		expected = append(expected, modelconfig.ArtifactSummary{Role: b.Role, Identity: hex.EncodeToString(sum[:]), Size: b.Size})
	}
	if got := Summaries(bindings); !reflect.DeepEqual(got, expected) {
		t.Fatalf("ordered JSON/SHA projection changed: %+v", got)
	}
	for _, input := range [][]modelconfig.ArtifactBinding{nil, {}} {
		if got := Summaries(input); got == nil || len(got) != 0 {
			t.Fatalf("nil/empty semantics changed: %#v", got)
		}
	}
}

type watchedFile struct {
	File
	closed *bool
	read   func()
}

func (f watchedFile) Close() error { *f.closed = true; return f.File.Close() }
func (f watchedFile) ReadAt(p []byte, off int64) (int, error) {
	n, err := f.File.ReadAt(p, off)
	if f.read != nil {
		f.read()
	}
	return n, err
}

func TestCheckedOpenRejectsSubstitutedHandle(t *testing.T) {
	e := syntheticEvidence(t, []byte("original"))
	other := syntheticEvidence(t, []byte("substitute"))
	closed := false
	a := Access{Open: func(string) (File, error) {
		f, err := os.Open(other.Configured)
		if err != nil {
			return nil, err
		}
		return watchedFile{File: f, closed: &closed}, nil
	}}
	if f, err := a.CheckedOpen(context.Background(), e); err == nil || f != nil || !closed {
		t.Fatalf("substitution accepted or not closed: f=%v err=%v closed=%t", f, err, closed)
	}
}

func TestSampleRejectsDriftAndCloses(t *testing.T) {
	e := syntheticEvidence(t, bytes.Repeat([]byte("a"), 3*SampleSize))
	closed := false
	a := Access{Open: func(p string) (File, error) {
		f, err := os.Open(p)
		if err != nil {
			return nil, err
		}
		return watchedFile{File: f, closed: &closed, read: func() {
			if err := os.WriteFile(p, []byte("changed"), 0600); err != nil {
				t.Fatal(err)
			}
		}}, nil
	}}
	if _, err := a.Sample(context.Background(), e, false); err == nil || !closed {
		t.Fatalf("drift accepted or not closed: err=%v closed=%t", err, closed)
	}
	if err := Validate(e); err == nil {
		t.Fatal("recheck refreshed original evidence")
	}
}

func TestSampleCancellationAndEmpty(t *testing.T) {
	e := syntheticEvidence(t, nil)
	if _, err := (Access{}).Sample(context.Background(), e, false); err == nil {
		t.Fatal("nonempty requirement ignored")
	}
	if _, err := (Access{}).Sample(context.Background(), e, true); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := (Access{}).Sample(ctx, e, true); err != context.Canceled {
		t.Fatalf("got %v", err)
	}
}

func TestReaderRetainsOriginalEvidence(t *testing.T) {
	e := syntheticEvidence(t, []byte("original"))
	f, err := (Access{}).CheckedOpen(context.Background(), e)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := os.WriteFile(e.Configured, []byte("different-sized"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := (Reader{context.Background(), f, e}).ReadAt(make([]byte, 1), 0); err == nil || err == io.EOF {
		t.Fatalf("read drift accepted: %v", err)
	}
}

func TestSampleFinalRecheckAndCancellation(t *testing.T) {
	test := func(name string, cancelRead bool) {
		t.Run(name, func(t *testing.T) {
			e := syntheticEvidence(t, []byte("one read"))
			closed := false
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			a := Access{Open: func(p string) (File, error) {
				f, err := os.Open(p)
				if err != nil {
					return nil, err
				}
				return watchedFile{File: f, closed: &closed, read: func() {
					if cancelRead {
						cancel()
					} else {
						if err := os.WriteFile(p, []byte("changed after last read"), 0600); err != nil {
							t.Fatal(err)
						}
					}
				}}, nil
			}}
			if _, err := a.Sample(ctx, e, false); err == nil || !closed {
				t.Fatalf("last-read change/cancel accepted or handle leaked: err=%v closed=%t", err, closed)
			}
		})
	}
	test("final original evidence recheck", false)
	test("cancel after read", true)
}
