// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package gguffacts

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"
)

type fixture struct {
	bytes.Buffer
	order binary.ByteOrder
}

func TestDecodeTokenShiftCount(t *testing.T) {
	for _, arch := range []string{"rwkv6", "rwkv7"} {
		t.Run(arch, func(t *testing.T) {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 0, 3)
			f.key("general.architecture", 8)
			f.str(arch)
			f.key(arch+".token_shift_count", 4)
			f.u32(3)
			f.key("other.token_shift_count", 4)
			f.u32(99)
			facts, err := Decode(context.Background(), []Artifact{artifact(&f, aligned(f.Len()))})
			if err != nil {
				t.Fatal(err)
			}
			if got := facts.Shards[0].Numeric[arch+".token_shift_count"]; got.Type != 4 || got.Bits != 3 || got.Array != nil {
				t.Fatalf("shift=%+v", got)
			}
			if _, ok := facts.Shards[0].Numeric["other.token_shift_count"]; ok {
				t.Fatal("retained foreign namespace")
			}
		})
	}
}

func TestDecodeTokenShiftCountUsesNumericLimits(t *testing.T) {
	t.Run("string dimension rejected", func(t *testing.T) {
		f := fixture{order: binary.LittleEndian}
		f.header(3, 0, 2)
		f.key("general.architecture", 8)
		f.str("rwkv6")
		f.key("rwkv6.token_shift_count", 8)
		f.str("3")
		if _, err := Decode(context.Background(), []Artifact{artifact(&f, aligned(f.Len()))}); !errors.Is(err, ErrInvalid) {
			t.Fatalf("error=%v", err)
		}
	})
	t.Run("array ceiling retained", func(t *testing.T) {
		f := fixture{order: binary.LittleEndian}
		f.header(3, 0, 2)
		f.key("general.architecture", 8)
		f.str("rwkv7")
		f.key("rwkv7.token_shift_count", 9)
		f.u32(4)
		f.u64(productionLimits.numeric + 1)
		if _, err := Decode(context.Background(), []Artifact{artifact(&f, aligned(f.Len()))}); !errors.Is(err, ErrLimit) {
			t.Fatalf("error=%v", err)
		}
	})
}

func TestTensorRangesAndTruncation(t *testing.T) {
	for _, tc := range []struct {
		name string
		make func(*fixture)
	}{
		{"overlap", func(f *fixture) { f.header(3, 2, 0); f.tensor("a", 0, 16, 0); f.tensor("b", 0, 16, 32) }},
		{"offset alignment", func(f *fixture) { f.header(3, 1, 0); f.tensor("a", 0, 1, 1) }},
		{"duplicate tensor", func(f *fixture) { f.header(3, 2, 0); f.tensor("a", 0, 1, 0); f.tensor("a", 0, 1, 32) }},
		{"size overflow", func(f *fixture) {
			f.header(3, 1, 0)
			f.str("a")
			f.u32(2)
			f.u64(1 << 62)
			f.u64(16)
			f.u32(0)
			f.u64(0)
		}},
		{"invalid alignment", func(f *fixture) { f.header(3, 0, 1); f.key("general.alignment", 4); f.u32(3) }},
		{"truncated", func(f *fixture) { f.header(3, 0, 1); f.key("x", 8); f.u64(12); f.WriteString("short") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := fixture{order: binary.LittleEndian}
			tc.make(&f)
			if _, err := Decode(context.Background(), []Artifact{artifact(&f, aligned(f.Len())+512)}); err == nil {
				t.Fatal("accepted invalid data")
			}
		})
	}
}

func TestSparseFile(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.header(3, 1, 0)
	f.tensor("weight", 0, 16, 0)
	file, err := os.Create(filepath.Join(t.TempDir(), "fixture.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := file.Close(); err != nil {
			t.Error(err)
		}
	})
	if _, err = file.Write(f.Bytes()); err != nil {
		t.Fatal(err)
	}
	size := int64(8 << 20)
	if err = file.Truncate(size); err != nil {
		t.Fatal(err)
	}
	sentinel := []byte("untouched tensor payload")
	if _, err := file.WriteAt(sentinel, aligned(f.Len())); err != nil {
		t.Fatal(err)
	}
	r := &countingReader{r: file, ceiling: int64(f.Len())}
	facts, err := Decode(context.Background(), []Artifact{{Role: RoleDraft, Reader: r, Size: size}})
	if err != nil {
		t.Fatal(err)
	}
	if facts.Shards[0].Role != RoleDraft || r.end != int64(f.Len()) || r.denied != 0 || facts.Usage.ScannedHeaderBytes != uint64(f.Len()) {
		t.Fatalf("facts=%+v reads=%+v", facts, r)
	}
	got := make([]byte, len(sentinel))
	if _, err := file.ReadAt(got, aligned(f.Len())); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, sentinel) {
		t.Fatalf("tensor sentinel changed: %q", got)
	}
}

func (f *fixture) u32(v uint32) { var b [4]byte; f.order.PutUint32(b[:], v); f.Write(b[:]) }
func (f *fixture) u64(v uint64) { var b [8]byte; f.order.PutUint64(b[:], v); f.Write(b[:]) }
func (f *fixture) str(s string) { f.u64(uint64(len(s))); f.WriteString(s) }
func (f *fixture) header(version uint32, tensors, metadata uint64) {
	f.WriteString("GGUF")
	f.u32(version)
	f.u64(tensors)
	f.u64(metadata)
}
func (f *fixture) key(k string, typ uint32) { f.str(k); f.u32(typ) }
func (f *fixture) tensor(name string, typ uint32, dim, offset uint64) {
	f.str(name)
	f.u32(1)
	f.u64(dim)
	f.u32(typ)
	f.u64(offset)
}

type ceilingReader struct {
	r   io.ReaderAt
	end int64
}

func (r ceilingReader) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 || off+int64(len(p)) > r.end {
		return 0, errors.New("read past directory")
	}
	return r.r.ReadAt(p, off)
}
func artifact(f *fixture, size int64) Artifact {
	return Artifact{Role: RoleMain, Reader: ceilingReader{bytes.NewReader(f.Bytes()), int64(f.Len())}, Size: size}
}
func aligned(n int) int64 { return int64((n + 31) &^ 31) }

func TestDecodeEndianAndDirectoryBoundary(t *testing.T) {
	for _, order := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
		for _, version := range []uint32{2, 3} {
			f := fixture{order: order}
			f.header(version, 1, 3)
			f.key("llama.context_length", 4)
			f.u32(4096)
			f.key("general.architecture", 8)
			f.str("llama")
			f.key("tokenizer.chat_template", 8)
			f.str("secret template")
			f.tensor("blk.3.weight", 2, 32, 0)
			facts, err := Decode(context.Background(), []Artifact{artifact(&f, aligned(f.Len())+18)})
			if err != nil {
				t.Fatal(err)
			}
			s := facts.Shards[0]
			if s.Version != version || s.Tensors[0].Bytes != 18 || *s.Tensors[0].Layer != 3 || s.Templates[0].SHA256 != sha256.Sum256([]byte("secret template")) {
				t.Fatalf("bad facts: %+v", s)
			}
			if v, ok := s.Numeric["llama.context_length"].Uint64(); !ok || v != 4096 {
				t.Fatal(v, ok)
			}
			clone := facts.Clone()
			clone.Shards[0].Tensors[0].Dimensions[0] = 99
			if s.Tensors[0].Dimensions[0] != 32 || clone.RetainedSize() != facts.RetainedSize() {
				t.Fatal("clone aliases or lost accounting")
			}
		}
	}
}

func TestArraysCountsAndPrivacy(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.header(3, 0, 4)
	f.key("tokenizer.ggml.tokens", 9)
	f.u32(8)
	f.u64(2)
	f.str("secret")
	f.str("other")
	f.key("tokenizer.ggml.merges", 9)
	f.u32(8)
	f.u64(0)
	f.key("llama.attention.head_count", 9)
	f.u32(4)
	f.u64(2)
	f.u32(8)
	f.u32(16)
	f.key("general.architecture", 8)
	f.str("llama")
	facts, err := Decode(context.Background(), []Artifact{artifact(&f, int64(f.Len()))})
	if err != nil {
		t.Fatal(err)
	}
	s := facts.Shards[0]
	if !s.Vocabulary.Present || s.Vocabulary.Value != 2 || !s.Merges.Present || s.Merges.Value != 0 || len(s.Numeric["llama.attention.head_count"].Array) != 2 {
		t.Fatalf("%+v", s)
	}
}

func TestRejectInvalidAndBudgets(t *testing.T) {
	base := func() fixture { f := fixture{order: binary.LittleEndian}; f.header(3, 0, 0); return f }
	tests := []struct {
		name  string
		make  func() fixture
		limit func(*limits)
	}{
		{"version", func() fixture { f := fixture{order: binary.LittleEndian}; f.header(4, 0, 0); return f }, nil},
		{"metadata count", func() fixture { f := fixture{order: binary.LittleEndian}; f.header(3, 0, 2); return f }, func(l *limits) { l.metadata = 1 }},
		{"tensor count", func() fixture { f := fixture{order: binary.LittleEndian}; f.header(3, 2, 0); return f }, func(l *limits) { l.tensors = 1 }},
		{"scanned", base, func(l *limits) { l.scanned = 23 }},
		{"retained", base, func(l *limits) { l.retained = 1 }},
		{"name", func() fixture {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 0, 1)
			f.str("toolong")
			return f
		}, func(l *limits) { l.name = 3 }},
		{"string", func() fixture {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 0, 1)
			f.key("x", 8)
			f.str("abcd")
			return f
		}, func(l *limits) { l.str = 3 }},
		{"array", func() fixture {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 0, 1)
			f.key("x", 9)
			f.u32(4)
			f.u64(2)
			return f
		}, func(l *limits) { l.skipped = 1 }},
		{"numeric array", func() fixture {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 0, 1)
			f.key("llama.context_length", 9)
			f.u32(4)
			f.u64(2)
			return f
		}, func(l *limits) { l.numeric = 1 }},
		{"nested array", func() fixture {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 0, 1)
			f.key("x", 9)
			f.u32(9)
			f.u64(1)
			return f
		}, nil},
		{"invalid type", func() fixture {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 1, 0)
			f.tensor("a", 4, 32, 0)
			return f
		}, nil},
		{"quant row", func() fixture {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 1, 0)
			f.tensor("a", 2, 31, 0)
			return f
		}, nil},
		{"duplicate", func() fixture {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 0, 2)
			f.key("x", 4)
			f.u32(0)
			f.key("x", 4)
			f.u32(0)
			return f
		}, nil},
		{"utf8", func() fixture {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 0, 1)
			f.key("x", 8)
			f.str("\xff")
			return f
		}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := tt.make()
			l := productionLimits
			if tt.limit != nil {
				tt.limit(&l)
			}
			if _, err := decode(context.Background(), []Artifact{artifact(&f, aligned(f.Len())+1024)}, l); err == nil {
				t.Fatal("accepted invalid fixture")
			}
		})
	}
	if MaxArtifacts != 1024 || MaxMetadataPerShard != 65536 || MaxTensorsPerSet != 262144 || MaxSkippedArray != 1048576 || MaxNumericArray != 4096 || MaxNameBytes != 256 || MaxStringBytes != 1<<20 || MaxRetainedBytes != 64<<20 || MaxScannedHeaderBytes != 128<<20 {
		t.Fatal("production limits changed")
	}
}

func TestCancellationAndAggregate(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.header(3, 0, 0)
	a := artifact(&f, int64(f.Len()))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Decode(ctx, []Artifact{a}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	l := productionLimits
	l.scanned = 47
	if _, err := decode(context.Background(), []Artifact{a, a}, l); !errors.Is(err, ErrLimit) {
		t.Fatal(err)
	}
	l = productionLimits
	l.artifacts = 1
	if _, err := decode(context.Background(), []Artifact{a, a}, l); !errors.Is(err, ErrLimit) {
		t.Fatal(err)
	}
}

// A rejected read is counted as well: success or ErrLimit alone cannot prove
// that the parser avoided an oversized request to ReaderAt.
type countingReader struct {
	r          io.ReaderAt
	ceiling    int64
	end        int64
	bytes      uint64
	calls      int
	denied     int
	maxRequest int
	after      func()
}

func (r *countingReader) ReadAt(p []byte, off int64) (int, error) {
	r.calls++
	if len(p) > r.maxRequest {
		r.maxRequest = len(p)
	}
	end := off + int64(len(p))
	if off < 0 || end < off || end > r.ceiling {
		r.denied++
		return 0, errors.New("read exceeded witness ceiling")
	}
	if end > r.end {
		r.end = end
	}
	r.bytes += uint64(len(p))
	n, err := r.r.ReadAt(p, off)
	if r.after != nil {
		r.after()
	}
	return n, err
}

func countedArtifact(f *fixture, ceiling int64) (Artifact, *countingReader) {
	r := &countingReader{r: bytes.NewReader(f.Bytes()), ceiling: ceiling}
	return Artifact{Role: RoleMain, Reader: r, Size: int64(f.Len())}, r
}

func requireLimitAt(t *testing.T, f *fixture, lim limits, ceiling int64) {
	t.Helper()
	a, r := countedArtifact(f, ceiling)
	got, err := decode(context.Background(), []Artifact{a}, lim)
	if !errors.Is(err, ErrLimit) || len(got.Shards) != 0 {
		t.Fatalf("facts=%+v error=%v, want limit without partial facts", got, err)
	}
	if r.denied != 0 || r.end > ceiling {
		t.Fatalf("excess read before limit: %+v", r)
	}
}

func requireDecode(t *testing.T, f *fixture, lim limits) Facts {
	t.Helper()
	a, r := countedArtifact(f, int64(f.Len()))
	facts, err := decode(context.Background(), []Artifact{a}, lim)
	if err != nil {
		t.Fatal(err)
	}
	if r.denied != 0 || r.end != int64(f.Len()) || facts.Usage.ScannedHeaderBytes != uint64(f.Len()) {
		t.Fatalf("incomplete or excess header reads: usage=%+v reader=%+v", facts.Usage, r)
	}
	return facts
}

func TestBudgetCountBoundaries(t *testing.T) {
	t.Run("metadata per shard", func(t *testing.T) {
		f := fixture{order: binary.LittleEndian}
		f.header(3, 0, 2)
		f.key("a", 0)
		f.WriteByte(1)
		f.key("b", 0)
		f.WriteByte(2)
		lim := productionLimits
		lim.metadata = 2
		requireDecode(t, &f, lim)
		lim.metadata = 1
		requireLimitAt(t, &f, lim, 24)
	})
	t.Run("tensors across set", func(t *testing.T) {
		f := fixture{order: binary.LittleEndian}
		f.header(3, 2, 0)
		f.tensor("a", 0, 1, 0)
		f.tensor("b", 0, 1, 32)
		lim := productionLimits
		lim.tensors = 2
		a, r := countedArtifact(&f, int64(f.Len()))
		a.Size = aligned(f.Len()) + 36
		facts, err := decode(context.Background(), []Artifact{a}, lim)
		if err != nil || facts.Usage.TensorCount != 2 || r.denied != 0 {
			t.Fatalf("facts=%+v error=%v reads=%+v", facts, err, r)
		}
		lim.tensors = 1
		requireLimitAt(t, &f, lim, 24)
	})
	t.Run("artifacts before result allocation", func(t *testing.T) {
		f := fixture{order: binary.LittleEndian}
		f.header(3, 0, 0)
		a, r := countedArtifact(&f, 24)
		lim := productionLimits
		lim.artifacts = 2
		facts, err := decode(context.Background(), []Artifact{a, a}, lim)
		if err != nil || len(facts.Shards) != 2 {
			t.Fatalf("facts=%+v error=%v", facts, err)
		}
		r.calls = 0
		lim.artifacts = 1
		if _, err := decode(context.Background(), []Artifact{a, a}, lim); !errors.Is(err, ErrLimit) || r.calls != 0 {
			t.Fatalf("error=%v reads=%d", err, r.calls)
		}
	})
}

func TestBudgetArrayBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name, key string
		typ       uint32
		retained  bool
	}{
		{"skipped numeric", "discard", 4, false},
		{"skipped strings", "tokenizer.ggml.tokens", 8, false},
		{"retained numeric", "general.sampling.top_k", 4, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 0, 1)
			f.key(tc.key, 9)
			f.u32(tc.typ)
			f.u64(2)
			declarationEnd := int64(f.Len())
			if tc.typ == 8 {
				f.str("first")
				f.str("second")
			} else {
				f.u32(7)
				f.u32(11)
			}
			lim := productionLimits
			lim.numeric, lim.skipped = 2, 2
			facts := requireDecode(t, &f, lim)
			if tc.retained && !reflect.DeepEqual(facts.Shards[0].Numeric[tc.key].Array, []uint64{7, 11}) {
				t.Fatalf("numeric array=%+v", facts.Shards[0].Numeric)
			}
			if tc.retained {
				lim.numeric = 1
			} else {
				lim.skipped = 1
			}
			requireLimitAt(t, &f, lim, declarationEnd)
		})
	}
}

func TestBudgetStringBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name string
		key  bool
	}{
		{"metadata key", true},
		{"metadata string", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 0, 1)
			ceiling := int64(32)
			if tc.key {
				f.key("abcd", 0)
				f.WriteByte(0)
			} else {
				f.key("x", 8)
				ceiling = int64(f.Len() + 8)
				f.str("abcd")
			}
			lim := productionLimits
			if tc.key {
				lim.name = 4
			} else {
				lim.str = 4
			}
			requireDecode(t, &f, lim)
			if tc.key {
				lim.name--
			} else {
				lim.str--
			}
			requireLimitAt(t, &f, lim, ceiling)
		})
	}
	t.Run("tensor name", func(t *testing.T) {
		f := fixture{order: binary.LittleEndian}
		f.header(3, 1, 0)
		f.tensor("abcd", 0, 1, 0)
		lim := productionLimits
		lim.name = 4
		a, r := countedArtifact(&f, int64(f.Len()))
		a.Size = aligned(f.Len()) + 4
		if _, err := decode(context.Background(), []Artifact{a}, lim); err != nil || r.denied != 0 {
			t.Fatalf("error=%v reads=%+v", err, r)
		}
		lim.name = 3
		requireLimitAt(t, &f, lim, 32)
	})
}

func TestBudgetBytesBeforeAllocationAndRead(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.header(3, 0, 1)
	f.key("general.sampling.top_k", 9)
	f.u32(4)
	f.u64(3)
	arrayStart := int64(f.Len())
	for i := 0; i < 3; i++ {
		f.u32(uint32(i))
	}
	facts := requireDecode(t, &f, productionLimits)
	lim := productionLimits
	lim.retained = facts.RetainedSize()
	requireDecode(t, &f, lim)
	lim.retained--
	requireLimitAt(t, &f, lim, arrayStart)
	lim = productionLimits
	lim.scanned = uint64(f.Len())
	requireDecode(t, &f, lim)
	lim.scanned--
	requireLimitAt(t, &f, lim, arrayStart)

	// Both byte counters reject a first-over charge without changing their
	// state, independently of payload allocation or the fixture's other limits.
	for _, budget := range []uint64{MaxRetainedBytes, MaxScannedHeaderBytes} {
		d := decoder{ctx: context.Background(), lim: productionLimits, size: budget + 1}
		if budget == MaxRetainedBytes {
			if err := d.charge(budget); err != nil || d.usage.RetainedBytes != budget {
				t.Fatalf("at retained ceiling: %+v %v", d.usage, err)
			}
			if err := d.charge(1); !errors.Is(err, ErrLimit) || d.usage.RetainedBytes != budget {
				t.Fatalf("over retained ceiling: %+v %v", d.usage, err)
			}
		} else {
			if err := d.advance(budget); err != nil || d.usage.ScannedHeaderBytes != budget {
				t.Fatalf("at scanned ceiling: %+v %v", d.usage, err)
			}
			if err := d.advance(1); !errors.Is(err, ErrLimit) || d.off != budget || d.usage.ScannedHeaderBytes != budget {
				t.Fatalf("over scanned ceiling: %+v %v", d.usage, err)
			}
		}
	}
}

func TestProductionLimitDeclarations(t *testing.T) {
	// These are full production declarations, not scaled down count limits.
	// At-ceiling counts proceed to the file-bound check with no count-driven
	// allocation; first-over counts stop at the declaration before any body I/O.
	for _, tc := range []struct {
		name    string
		ceiling uint64
		tensors bool
	}{
		{"metadata", MaxMetadataPerShard, false},
		{"tensors", MaxTensorsPerSet, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, n := range []uint64{tc.ceiling, tc.ceiling + 1, math.MaxUint64} {
				f := fixture{order: binary.LittleEndian}
				if tc.tensors {
					f.header(3, n, 0)
				} else {
					f.header(3, 0, n)
				}
				a, r := countedArtifact(&f, 24)
				_, err := Decode(context.Background(), []Artifact{a})
				want := error(ErrLimit)
				if n == tc.ceiling {
					want = io.ErrUnexpectedEOF
				}
				if !errors.Is(err, want) || r.denied != 0 || r.end != 24 {
					t.Fatalf("count=%d error=%v want=%v reads=%+v", n, err, want, r)
				}
			}
		})
	}

	t.Run("artifacts", func(t *testing.T) {
		f := fixture{order: binary.LittleEndian}
		f.header(3, 0, 0)
		a, r := countedArtifact(&f, 24)
		artifacts := make([]Artifact, MaxArtifacts+1)
		for i := range artifacts {
			artifacts[i] = a
		}
		facts, err := Decode(context.Background(), artifacts[:MaxArtifacts])
		if err != nil || len(facts.Shards) != MaxArtifacts {
			t.Fatalf("shards=%d error=%v", len(facts.Shards), err)
		}
		r.calls = 0
		if _, err := Decode(context.Background(), artifacts); !errors.Is(err, ErrLimit) || r.calls != 0 {
			t.Fatalf("error=%v reads=%d", err, r.calls)
		}
	})
	for _, tc := range []struct {
		name, key string
		ceiling   uint64
	}{
		{"skipped array", "ignored", MaxSkippedArray},
		{"numeric array", "general.sampling.top_k", MaxNumericArray},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 0, 1)
			f.key(tc.key, 9)
			f.u32(0)
			countOffset := f.Len()
			f.u64(tc.ceiling)
			declarationEnd := int64(f.Len())
			f.Write(make([]byte, int(tc.ceiling)))
			requireDecode(t, &f, productionLimits)
			f.order.PutUint64(f.Bytes()[countOffset:countOffset+8], tc.ceiling+1)
			requireLimitAt(t, &f, productionLimits, declarationEnd)
			f.order.PutUint64(f.Bytes()[countOffset:countOffset+8], math.MaxUint64)
			requireLimitAt(t, &f, productionLimits, declarationEnd)
		})
	}
	for _, tc := range []struct {
		name string
		key  bool
	}{
		{"name bytes", true},
		{"string bytes", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 0, 1)
			limit := MaxNameBytes
			if !tc.key {
				f.key("ignored", 8)
				limit = MaxStringBytes
			}
			lengthOffset := f.Len()
			f.str(strings.Repeat("a", limit))
			if tc.key {
				f.u32(0)
				f.WriteByte(0)
			}
			requireDecode(t, &f, productionLimits)
			f.order.PutUint64(f.Bytes()[lengthOffset:lengthOffset+8], uint64(limit+1))
			requireLimitAt(t, &f, productionLimits, int64(lengthOffset+8))
			f.order.PutUint64(f.Bytes()[lengthOffset:lengthOffset+8], math.MaxUint64)
			requireLimitAt(t, &f, productionLimits, int64(lengthOffset+8))
		})
	}
}

func TestAggregateBudgetsAcrossRoles(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.header(3, 1, 0)
	f.tensor("w", 0, 1, 0)
	size := aligned(f.Len()) + 4
	one, err := Decode(context.Background(), []Artifact{artifact(&f, size)})
	if err != nil {
		t.Fatal(err)
	}
	roles := []Role{RoleMain, RoleMain, RoleDraft, RoleCompanion}
	for _, budget := range []string{"tensor", "scanned", "retained"} {
		t.Run(budget, func(t *testing.T) {
			var readers []*countingReader
			var artifacts []Artifact
			for i, role := range roles {
				shard := fixture{order: binary.LittleEndian}
				shard.header(3, 1, 0)
				shard.tensor(strconv.Itoa(i), 0, 1, 0)
				a, r := countedArtifact(&shard, int64(shard.Len()))
				a.Role, a.Size = role, size
				artifacts = append(artifacts, a)
				readers = append(readers, r)
			}
			lim := productionLimits
			switch budget {
			case "tensor":
				lim.tensors = uint64(len(roles))
			case "scanned":
				lim.scanned = uint64(len(roles)) * one.Usage.ScannedHeaderBytes
			case "retained":
				lim.retained = uint64(len(roles)) * one.RetainedSize()
			}
			facts, err := decode(context.Background(), artifacts, lim)
			if err != nil || len(facts.Shards) != len(roles) {
				t.Fatalf("at aggregate boundary: facts=%+v error=%v", facts, err)
			}
			for i, role := range roles {
				if facts.Shards[i].Role != role {
					t.Fatalf("shard %d role=%s want=%s", i, facts.Shards[i].Role, role)
				}
				readers[i].calls, readers[i].end = 0, 0
			}
			switch budget {
			case "tensor":
				lim.tensors--
			case "scanned":
				lim.scanned--
			case "retained":
				lim.retained--
			}
			facts, err = decode(context.Background(), artifacts, lim)
			if !errors.Is(err, ErrLimit) || len(facts.Shards) != 0 {
				t.Fatalf("cross-role aggregate bypass: facts=%+v error=%v", facts, err)
			}
			for i, r := range readers {
				if r.denied != 0 {
					t.Fatalf("shard %d made an excess read: %+v", i, r)
				}
			}
			if budget == "tensor" && readers[len(readers)-1].end != 24 {
				t.Fatalf("aggregate tensor count was not rejected at header: %+v", readers[len(readers)-1])
			}
		})
	}
}

func TestPrimitiveRepresentations(t *testing.T) {
	values := []struct {
		typ  uint32
		bits uint64
	}{
		{0, 255}, {1, 0x80}, {2, 65535}, {3, 0x8000},
		{4, math.MaxUint32}, {5, 0x80000000}, {6, 0x80000000},
		{7, 1}, {10, math.MaxUint64}, {11, 0x8000000000000000},
		{12, 0x8000000000000000},
	}
	for _, order := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
		t.Run(order.String(), func(t *testing.T) {
			for _, value := range values {
				t.Run(strconv.FormatUint(uint64(value.typ), 10), func(t *testing.T) {
					f := fixture{order: order}
					f.header(3, 0, 2)
					f.key("general.sampling.top_k", value.typ)
					var raw [8]byte
					order.PutUint64(raw[:], value.bits)
					w := width(value.typ)
					if order == binary.BigEndian {
						f.Write(raw[8-w:])
					} else {
						f.Write(raw[:w])
					}
					f.key("general.sampling.temp", 9)
					f.u32(value.typ)
					f.u64(1)
					if order == binary.BigEndian {
						f.Write(raw[8-w:])
					} else {
						f.Write(raw[:w])
					}
					facts := requireDecode(t, &f, productionLimits)
					scalar := facts.Shards[0].Numeric["general.sampling.top_k"]
					array := facts.Shards[0].Numeric["general.sampling.temp"]
					if scalar.Type != ValueType(value.typ) || scalar.Bits != value.bits || scalar.Array != nil || array.Type != ValueType(value.typ) || !reflect.DeepEqual(array.Array, []uint64{value.bits}) {
						t.Fatalf("scalar=%+v array=%+v want=%+v", scalar, array, value)
					}
				})
			}
		})
	}
}

func TestNumericConversions(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   Numeric
		want float64
		ok   bool
	}{
		{"float32", Numeric{Type: 6, Bits: uint64(math.Float32bits(0.5))}, 0.5, true},
		{"float64", Numeric{Type: 12, Bits: math.Float64bits(0.125)}, 0.125, true},
		{"negative integer", Numeric{Type: 1, Bits: 255}, -1, true},
		{"exact positive integer", Numeric{Type: 10, Bits: 1 << 53}, 1 << 53, true},
		{"lossy integer", Numeric{Type: 10, Bits: 1<<53 + 1}, 0, false},
		{"nan", Numeric{Type: 12, Bits: math.Float64bits(math.NaN())}, 0, false},
		{"infinity", Numeric{Type: 6, Bits: uint64(math.Float32bits(float32(math.Inf(1))))}, 0, false},
		{"boolean is not a recommendation", Numeric{Type: 7, Bits: 1}, 0, false},
		{"array is not scalar", Numeric{Type: 4, Array: []uint64{}}, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := tc.in.Float64()
			if ok != tc.ok || ok && got != tc.want {
				t.Fatalf("Float64=%v,%v want=%v,%v", got, ok, tc.want, tc.ok)
			}
		})
	}
	for _, typ := range []ValueType{1, 3, 5, 11} {
		if _, ok := (Numeric{Type: typ, Bits: math.MaxUint64}).Uint64(); ok {
			t.Fatalf("negative type %d converted to unsigned", typ)
		}
	}
	if n, ok := (Numeric{Type: 10, Bits: math.MaxUint64}).Uint64(); !ok || n != math.MaxUint64 {
		t.Fatalf("lost integer precision: %d %v", n, ok)
	}
}

func TestMetadataNamespacePrivacyAndClone(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.header(3, 1, 9)
	f.key("other.context_length", 4)
	f.u32(999)
	f.key("llama.attention.head_count", 9)
	f.u32(4)
	f.u64(2)
	f.u32(8)
	f.u32(16)
	f.key("llama.context_length", 4)
	f.u32(4096)
	f.key("llama.private", 8)
	f.str("unrestricted metadata must be discarded")
	f.key("tokenizer.ggml.tokens", 9)
	f.u32(8)
	f.u64(1)
	f.str("private token")
	f.key("tokenizer.ggml.merges", 9)
	f.u32(8)
	f.u64(1)
	f.str("private merge")
	f.key("tokenizer.chat_template", 8)
	f.str("private default template")
	f.key("tokenizer.chat_template.tool_use", 8)
	f.str("private named template")
	f.key("general.architecture", 8)
	f.str("llama")
	f.tensor("blk.2.weight", 0, 1, 0)
	facts, err := Decode(context.Background(), []Artifact{artifact(&f, aligned(f.Len())+4)})
	if err != nil {
		t.Fatal(err)
	}
	s := facts.Shards[0]
	if len(s.Numeric) != 2 || s.Numeric["llama.context_length"].Bits != 4096 || len(s.Templates) != 2 || s.Templates[0].SHA256 != sha256.Sum256([]byte("private default template")) || s.Templates[1].SHA256 != sha256.Sum256([]byte("private named template")) || s.Vocabulary != (Count{true, 1}) || s.Merges != (Count{true, 1}) {
		t.Fatalf("unexpected retained metadata: %+v", s)
	}
	clone := facts.Clone()
	clone.Shards[0].Numeric["llama.attention.head_count"].Array[0] = 999
	delete(clone.Shards[0].Numeric, "llama.context_length")
	clone.Shards[0].Templates[0].Key = "changed"
	clone.Shards[0].Tensors[0].Dimensions[0] = 999
	*clone.Shards[0].Tensors[0].Layer = 999
	if s.Numeric["llama.attention.head_count"].Array[0] != 8 || s.Numeric["llama.context_length"].Bits != 4096 || s.Templates[0].Key != "tokenizer.chat_template" || s.Tensors[0].Dimensions[0] != 1 || *s.Tensors[0].Layer != 2 || clone.RetainedSize() != facts.RetainedSize() {
		t.Fatal("clone shares mutable collections or lost its accounting")
	}
}

func TestEmptyNumericArrayRemainsArrayAfterClone(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.header(3, 0, 1)
	f.key("general.sampling.top_k", 9)
	f.u32(4)
	f.u64(0)
	facts := requireDecode(t, &f, productionLimits).Clone()
	n := facts.Shards[0].Numeric["general.sampling.top_k"]
	if n.Array == nil || len(n.Array) != 0 {
		t.Fatalf("empty array became scalar: %+v", n)
	}
}

func TestAllocationChargeCoversDescriptorsAndCapacity(t *testing.T) {
	// The constants cover 32-bit and 64-bit layouts, including allocation
	// rounding, duplicate maps, and the retained tensor slice. String staging
	// and Numeric map entries are separately charged by the parser.
	if unsafe.Sizeof(Shard{})+64 > 256 || unsafe.Sizeof(Tensor{})+64 > 192 || unsafe.Sizeof(Template{}) > 48 {
		t.Fatal("descriptor charge must be updated for this layout")
	}
	f := fixture{order: binary.LittleEndian}
	f.header(3, 0, 3)
	for _, key := range []string{"tokenizer.chat_template", "tokenizer.chat_template.a", "tokenizer.chat_template.b"} {
		f.key(key, 8)
		f.str("")
	}
	facts := requireDecode(t, &f, productionLimits)
	// Artifact descriptors, duplicate map entries, staged keys, and each
	// explicitly allocated template backing array (capacity 1, 2, then 4).
	want := uint64(256 + 3*128 + (1+2+4)*48 + 3*16)
	for _, key := range []string{"tokenizer.chat_template", "tokenizer.chat_template.a", "tokenizer.chat_template.b"} {
		want += uint64(2*len(key) + 32)
	}
	if facts.RetainedSize() != want || cap(facts.Shards[0].Templates) != 4 {
		t.Fatalf("retained=%d want=%d capacity=%d", facts.RetainedSize(), want, cap(facts.Shards[0].Templates))
	}
}

func TestCancellationBetweenBoundedReads(t *testing.T) {
	for _, tc := range []struct {
		name string
		typ  uint32
	}{
		{"skipped primitive array", 9},
		{"discarded string", 8},
		{"hashed template", 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 0, 1)
			key := "ignored"
			if tc.name == "hashed template" {
				key = "tokenizer.chat_template"
			}
			f.key(key, tc.typ)
			if tc.typ == 9 {
				f.u32(0)
			}
			f.u64(8192)
			body := int64(f.Len())
			f.WriteString(strings.Repeat("a", 8192))
			a, r := countedArtifact(&f, body+4096)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r.after = func() {
				if r.end == body+4096 {
					cancel()
				}
			}
			facts, err := Decode(ctx, []Artifact{a})
			if !errors.Is(err, context.Canceled) || len(facts.Shards) != 0 || r.end != body+4096 || r.denied != 0 || r.maxRequest > 4096 {
				t.Fatalf("facts=%+v error=%v reads=%+v", facts, err, r)
			}
		})
	}
	t.Run("header read", func(t *testing.T) {
		f := fixture{order: binary.LittleEndian}
		f.header(3, 0, 0)
		a, r := countedArtifact(&f, 8)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		r.after = cancel
		if _, err := Decode(ctx, []Artifact{a}); !errors.Is(err, context.Canceled) || r.calls != 1 || r.denied != 0 {
			t.Fatalf("error=%v reads=%+v", err, r)
		}
	})
	t.Run("expired deadline before allocation", func(t *testing.T) {
		ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
		defer cancel()
		if _, err := Decode(ctx, nil); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("error=%v", err)
		}
	})
}

func TestHostileDeclarationsAndMalformedTypes(t *testing.T) {
	for _, tc := range []struct {
		name string
		body func(*fixture)
	}{
		{"unknown scalar", func(f *fixture) { f.header(3, 0, 1); f.key("x", 13) }},
		{"unknown empty array", func(f *fixture) { f.header(3, 0, 1); f.key("x", 9); f.u32(13); f.u64(0) }},
		{"nested empty array", func(f *fixture) { f.header(3, 0, 1); f.key("x", 9); f.u32(9); f.u64(0) }},
		{"invalid boolean", func(f *fixture) { f.header(3, 0, 1); f.key("x", 7); f.WriteByte(2) }},
		{"invalid boolean array", func(f *fixture) { f.header(3, 0, 1); f.key("x", 9); f.u32(7); f.u64(1); f.WriteByte(2) }},
		{"array alignment", func(f *fixture) { f.header(3, 0, 1); f.key("general.alignment", 9); f.u32(4); f.u64(0) }},
		{"empty architecture", func(f *fixture) { f.header(3, 0, 1); f.key("general.architecture", 8); f.str("") }},
		{"template array", func(f *fixture) { f.header(3, 0, 1); f.key("tokenizer.chat_template", 9); f.u32(8); f.u64(0) }},
		{"tokens scalar", func(f *fixture) { f.header(3, 0, 1); f.key("tokenizer.ggml.tokens", 8); f.str("private") }},
		{"empty key", func(f *fixture) { f.header(3, 0, 1); f.key("", 0); f.WriteByte(1) }},
		{"key utf8", func(f *fixture) { f.header(3, 0, 1); f.key("\xff", 0); f.WriteByte(1) }},
		{"key embedded nul", func(f *fixture) { f.header(3, 0, 1); f.key("a\x00b", 0); f.WriteByte(1) }},
		{"architecture embedded nul", func(f *fixture) { f.header(3, 0, 1); f.key("general.architecture", 8); f.str("llama\x00other") }},
		{"string utf8", func(f *fixture) { f.header(3, 0, 1); f.key("x", 8); f.str("\xc3") }},
		{"zero dimensions", func(f *fixture) {
			f.header(3, 1, 0)
			f.str("w")
			f.u32(0)
			f.u64(1)
			f.u32(0)
			f.u64(0)
		}},
		{"too many dimensions", func(f *fixture) { f.header(3, 1, 0); f.str("w"); f.u32(5); f.Write(make([]byte, 52)) }},
		{"negative shape", func(f *fixture) { f.header(3, 1, 0); f.tensor("w", 0, math.MaxUint64, 0) }},
		{"zero shape", func(f *fixture) { f.header(3, 1, 0); f.tensor("w", 0, 0, 0) }},
		{"tensor utf8", func(f *fixture) { f.header(3, 1, 0); f.tensor("\xff", 0, 1, 0) }},
		{"tensor embedded nul", func(f *fixture) { f.header(3, 1, 0); f.tensor("a\x00b", 0, 1, 0) }},
		{"quantized row cannot borrow next dimension", func(f *fixture) {
			f.header(3, 1, 0)
			f.str("w")
			f.u32(2)
			f.u64(16)
			f.u64(2)
			f.u32(2)
			f.u64(0)
		}},
		{"byte size overflow", func(f *fixture) { f.header(3, 1, 0); f.tensor("w", 28, 1<<62, 0) }},
		{"huge offset", func(f *fixture) { f.header(3, 1, 0); f.tensor("w", 0, 1, math.MaxUint64-31) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := fixture{order: binary.LittleEndian}
			tc.body(&f)
			// Enough zero header bytes keep malformed-type validation distinct
			// from the inexpensive minimum-size declaration preflight.
			f.Write(make([]byte, 64))
			a, r := countedArtifact(&f, int64(f.Len()))
			a.Size += 512
			if facts, err := Decode(context.Background(), []Artifact{a}); !errors.Is(err, ErrInvalid) || len(facts.Shards) != 0 || r.denied != 0 {
				t.Fatalf("facts=%+v error=%v reads=%+v", facts, err, r)
			}
		})
	}
}

func TestTruncationAtEveryDirectoryByte(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.header(3, 1, 2)
	f.key("general.architecture", 8)
	f.str("llama")
	f.key("llama.attention.head_count", 9)
	f.u32(4)
	f.u64(2)
	f.u32(8)
	f.u32(16)
	f.tensor("w", 2, 32, 0)
	for n := 0; n < f.Len(); n++ {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			r := &countingReader{r: bytes.NewReader(f.Bytes()[:n]), ceiling: int64(f.Len())}
			// Declared size is deliberately larger than the short reader, so
			// this also exercises short reads, not only size preflight.
			a := Artifact{Role: RoleMain, Reader: r, Size: aligned(f.Len()) + 18}
			if _, err := Decode(context.Background(), []Artifact{a}); !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("error=%v want short-read failure", err)
			}
		})
	}
}

func TestUTF8AcrossStreamingChunks(t *testing.T) {
	for _, value := range []string{strings.Repeat("a", 4095) + "\u65e5\U0001f600", strings.Repeat("a", 4095) + "\xff"} {
		f := fixture{order: binary.LittleEndian}
		f.header(3, 0, 1)
		f.key("tokenizer.chat_template", 8)
		f.str(value)
		facts, err := Decode(context.Background(), []Artifact{artifact(&f, int64(f.Len()))})
		if strings.HasSuffix(value, "\xff") {
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("invalid utf8 error=%v", err)
			}
		} else if err != nil || facts.Shards[0].Templates[0].SHA256 != sha256.Sum256([]byte(value)) {
			t.Fatalf("split utf8 digest facts=%+v error=%v", facts, err)
		}
	}
}

func TestArtifactRoleOrderAndOwnership(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.header(3, 0, 0)
	a, r := countedArtifact(&f, 24)
	for _, roles := range [][]Role{{RoleDraft, RoleMain}, {RoleCompanion, RoleDraft}, {"projector"}} {
		artifacts := make([]Artifact, len(roles))
		for i, role := range roles {
			artifacts[i] = a
			artifacts[i].Role = role
		}
		if _, err := Decode(context.Background(), artifacts); !errors.Is(err, ErrInvalid) || r.calls != 0 {
			t.Fatalf("roles=%v error=%v reads=%d", roles, err, r.calls)
		}
	}
	for _, invalid := range []Artifact{{Role: RoleMain, Size: 0}, {Role: RoleMain, Reader: r, Size: -1}} {
		if _, err := Decode(context.Background(), []Artifact{invalid}); !errors.Is(err, ErrInvalid) || r.calls != 0 {
			t.Fatalf("invalid artifact error=%v reads=%d", err, r.calls)
		}
	}
}

func TestDuplicateTensorAcrossShardsWithinRole(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.header(3, 1, 0)
	f.tensor("w", 0, 1, 0)
	a := artifact(&f, aligned(f.Len())+4)
	if _, err := Decode(context.Background(), []Artifact{a, a}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("duplicate main-shard name error=%v", err)
	}
	draft := a
	draft.Role = RoleDraft
	if facts, err := Decode(context.Background(), []Artifact{a, draft}); err != nil || len(facts.Shards) != 2 {
		t.Fatalf("independent main and draft names: facts=%+v error=%v", facts, err)
	}
}

func TestHeaderMagicAndVersionRejection(t *testing.T) {
	for _, tc := range []struct {
		name, magic string
		version     uint32
	}{
		{"reversed magic", "FUGG", 3},
		{"unrecognized magic", "XXXX", 3},
		{"zero version", "GGUF", 0},
		{"v1", "GGUF", 1},
		{"future version", "GGUF", 4},
		{"ambiguous byte order", "GGUF", 0x00020000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := fixture{order: binary.LittleEndian}
			f.WriteString(tc.magic)
			f.u32(tc.version)
			f.u64(0)
			f.u64(0)
			a, r := countedArtifact(&f, 8)
			if _, err := Decode(context.Background(), []Artifact{a}); !errors.Is(err, ErrInvalid) || r.denied != 0 || r.end != 8 {
				t.Fatalf("error=%v reads=%+v", err, r)
			}
		})
	}
}

func TestProductionTensorNameBoundary(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.header(3, 1, 0)
	f.tensor(strings.Repeat("a", MaxNameBytes), 0, 1, 0)
	a, r := countedArtifact(&f, int64(f.Len()))
	a.Size = aligned(f.Len()) + 4
	if _, err := Decode(context.Background(), []Artifact{a}); err != nil || r.denied != 0 {
		t.Fatalf("at name ceiling: error=%v reads=%+v", err, r)
	}
	f.order.PutUint64(f.Bytes()[24:32], MaxNameBytes+1)
	requireLimitAt(t, &f, productionLimits, 32)
	f.order.PutUint64(f.Bytes()[24:32], math.MaxUint64)
	requireLimitAt(t, &f, productionLimits, 32)
}

func TestRetainedChargeBeforeNumericAllocation(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.u32(4)
	f.u64(3)
	f.u32(1)
	f.u32(2)
	f.u32(3)
	key := "general.sampling.top_k"
	// Requested capacity is 3*8; charged storage is rounded to 4*8.
	want := uint64(128 + len(key) + 32 + 16)
	for _, limit := range []uint64{want, want - 1} {
		r := &countingReader{r: bytes.NewReader(f.Bytes()), ceiling: int64(f.Len())}
		lim := productionLimits
		lim.retained = limit
		d := decoder{ctx: context.Background(), lim: lim, r: r, size: uint64(f.Len()), order: f.order}
		s := Shard{Numeric: make(map[string]Numeric)}
		err := d.metadata(&s, key, 9)
		if limit == want {
			if err != nil || d.usage.RetainedBytes != want || !reflect.DeepEqual(s.Numeric[key].Array, []uint64{1, 2, 3}) {
				t.Fatalf("at boundary: shard=%+v usage=%+v error=%v", s, d.usage, err)
			}
		} else if !errors.Is(err, ErrLimit) || d.usage.RetainedBytes != 0 || len(s.Numeric) != 0 || r.end != 12 || r.denied != 0 {
			t.Fatalf("allocated/read body before budget: shard=%+v usage=%+v error=%v reads=%+v", s, d.usage, err, r)
		}
	}
}

func TestTemplateCapacityChargedBeforeGrowth(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.str("")
	r := &countingReader{r: bytes.NewReader(f.Bytes()), ceiling: int64(f.Len())}
	lim := productionLimits
	lim.retained = 48 + 16 - 1
	d := decoder{ctx: context.Background(), lim: lim, r: r, size: uint64(f.Len()), order: f.order}
	s := Shard{}
	if err := d.metadata(&s, "tokenizer.chat_template", 8); !errors.Is(err, ErrLimit) || len(s.Templates) != 0 || cap(s.Templates) != 0 || d.usage.RetainedBytes != 0 || r.denied != 0 {
		t.Fatalf("growth exceeded retained budget: shard=%+v usage=%+v error=%v", s, d.usage, err)
	}
}

func TestKnownQuantizedLayoutSizes(t *testing.T) {
	// Independent examples from the pinned GGML_QUANT_SIZES reference use
	// multidimensional shapes so row-block and higher-dimension products both
	// contribute. These include ordinary, K, IQ, ternary and packed float types.
	for _, tc := range []struct {
		name       string
		typ        uint32
		row, bytes uint64
	}{
		{"f32", 0, 1, 4}, {"f16", 1, 1, 2}, {"q4_0", 2, 32, 18},
		{"q5_1", 7, 32, 24}, {"q8_1", 9, 32, 36}, {"q2_k", 10, 256, 84},
		{"q6_k", 14, 256, 210}, {"q8_k", 15, 256, 292},
		{"iq1_s", 19, 256, 50}, {"iq3_s", 21, 256, 110}, {"iq1_m", 29, 256, 56},
		{"bf16", 30, 1, 2}, {"tq1_0", 34, 256, 54}, {"tq2_0", 35, 256, 66},
		{"mxfp4", 39, 32, 17}, {"nvfp4", 40, 64, 36}, {"q1_0", 41, 128, 18}, {"q2_0", 42, 64, 18},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := fixture{order: binary.LittleEndian}
			f.header(3, 1, 0)
			f.str("w")
			f.u32(3)
			f.u64(tc.row * 2)
			f.u64(3)
			f.u64(5)
			f.u32(tc.typ)
			f.u64(0)
			facts, err := Decode(context.Background(), []Artifact{artifact(&f, aligned(f.Len())+int64(tc.bytes*30))})
			if err != nil || facts.Shards[0].Tensors[0].Bytes != tc.bytes*30 {
				t.Fatalf("facts=%+v error=%v want bytes=%d", facts, err, tc.bytes*30)
			}
		})
	}
	for _, typ := range []uint32{4, 5, 31, 32, 33, 36, 37, 38, 43, math.MaxUint32} {
		f := fixture{order: binary.LittleEndian}
		f.header(3, 1, 0)
		f.tensor("w", typ, 256, 0)
		_, err := Decode(context.Background(), []Artifact{artifact(&f, aligned(f.Len())+4096)})
		if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "unsupported GGML type") {
			t.Fatalf("type %d error=%v", typ, err)
		}
	}
}

func TestTensorFileBoundaryAndCustomAlignment(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.header(3, 2, 1)
	f.key("general.alignment", 4)
	f.u32(64)
	f.tensor("a", 0, 1, 0)
	f.tensor("b", 0, 1, 64)
	data := int64((f.Len() + 63) &^ 63)
	size := data + 68
	a, r := countedArtifact(&f, int64(f.Len()))
	a.Size = size
	facts, err := Decode(context.Background(), []Artifact{a})
	if err != nil || facts.Shards[0].DataOffset != uint64(data) || facts.Shards[0].Alignment != 64 || r.denied != 0 || r.end != int64(f.Len()) {
		t.Fatalf("exact file boundary: facts=%+v error=%v reads=%+v", facts, err, r)
	}
	a.Size--
	if _, err := Decode(context.Background(), []Artifact{a}); !errors.Is(err, ErrInvalid) || r.denied != 0 {
		t.Fatalf("one byte beyond file: error=%v reads=%+v", err, r)
	}
}

func TestUnboundDimensionsRemainUnknown(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.header(3, 0, 1)
	f.key("llama.context_length", 4)
	f.u32(4096)
	facts := requireDecode(t, &f, productionLimits)
	if facts.Shards[0].Architecture != "" || len(facts.Shards[0].Numeric) != 0 || facts.Shards[0].Vocabulary.Present || facts.Shards[0].Merges.Present {
		t.Fatalf("inferred missing architecture or tokenizer metadata: %+v", facts.Shards[0])
	}
}

func TestRetainedChargeBeforeNameAllocation(t *testing.T) {
	f := fixture{order: binary.LittleEndian}
	f.str("name")
	want := uint64(2*len("name") + 32)
	for _, ceiling := range []uint64{want, want - 1} {
		r := &countingReader{r: bytes.NewReader(f.Bytes()), ceiling: int64(f.Len())}
		lim := productionLimits
		lim.retained = ceiling
		d := decoder{ctx: context.Background(), lim: lim, r: r, size: uint64(f.Len()), order: f.order}
		name, _, err := d.text(MaxNameBytes, true, false)
		if ceiling == want {
			if err != nil || name != "name" || d.usage.RetainedBytes != want {
				t.Fatalf("at boundary: name=%q usage=%+v error=%v", name, d.usage, err)
			}
		} else if !errors.Is(err, ErrLimit) || name != "" || r.end != 8 || r.denied != 0 || d.usage.RetainedBytes != 0 {
			t.Fatalf("allocated/read name body before budget: name=%q usage=%+v error=%v reads=%+v", name, d.usage, err, r)
		}
	}
}

func TestRetainedChargeBeforeDescriptorsAndMaps(t *testing.T) {
	t.Run("result descriptors", func(t *testing.T) {
		f := fixture{order: binary.LittleEndian}
		f.header(3, 0, 0)
		lim := productionLimits
		lim.retained = 256
		requireDecode(t, &f, lim)
		lim.retained--
		requireLimitAt(t, &f, lim, 0)
	})
	t.Run("maps and tensor descriptors", func(t *testing.T) {
		f := fixture{order: binary.LittleEndian}
		f.header(3, 1, 1)
		f.key("x", 0)
		f.WriteByte(0)
		f.tensor("w", 0, 1, 0)
		lim := productionLimits
		// At the map/descriptor charge, the next name charge is the first
		// limit. One less stops at the header before constructing the maps.
		lim.retained = 256 + 128 + 192
		a, r := countedArtifact(&f, 32)
		a.Size = aligned(f.Len()) + 4
		if _, err := decode(context.Background(), []Artifact{a}, lim); !errors.Is(err, ErrLimit) || r.end != 32 || r.denied != 0 {
			t.Fatalf("at descriptor charge: error=%v reads=%+v", err, r)
		}
		lim.retained--
		requireLimitAt(t, &f, lim, 24)
	})
}
