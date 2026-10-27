// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package gguffacts describes bounded facts decoded from supplied GGUF artifacts.
package gguffacts

import (
	"io"
	"math"
)

// These production ceilings are immutable and apply before allocation. Array
// ceilings apply per array; tensor, retained and scanned ceilings apply to the
// entire supplied set, including main, draft and companion artifacts.
const (
	MaxMetadataPerShard   = 65536
	MaxArtifacts          = 1024
	MaxTensorsPerSet      = 262144
	MaxSkippedArray       = 1048576
	MaxNumericArray       = 4096
	MaxNameBytes          = 256
	MaxStringBytes        = 1 << 20
	MaxRetainedBytes      = 64 << 20
	MaxScannedHeaderBytes = 128 << 20
)

// ReaderRevision changes whenever the decoded facts or validation changes.
const ReaderRevision = 1

type Role string

const (
	RoleMain      Role = "main"
	RoleDraft     Role = "draft"
	RoleCompanion Role = "companion"
)

// Artifact is an already-open, caller-owned reader and its declared file size.
// Input order is preserved and must group main, then draft, then companion
// artifacts (groups may be absent). The decoder neither opens nor closes readers.
type Artifact struct {
	Role   Role
	Reader io.ReaderAt
	Size   int64
}

// ValueType is the GGUF numeric value type ID. Bits preserves the original
// integer, floating-point or boolean representation without precision loss.
type ValueType uint32

type Numeric struct {
	Type  ValueType
	Bits  uint64
	Array []uint64 // nil for a scalar; a non-nil empty slice is an empty array.
}

// Uint64 returns an exact nonnegative integer scalar.
func (n Numeric) Uint64() (uint64, bool) {
	if n.Array != nil {
		return 0, false
	}
	switch n.Type {
	case 0, 2, 4, 10:
		return n.Bits, true
	case 1:
		v := int8(n.Bits)
		if v >= 0 {
			return uint64(v), true
		}
	case 3:
		v := int16(n.Bits)
		if v >= 0 {
			return uint64(v), true
		}
	case 5:
		v := int32(n.Bits)
		if v >= 0 {
			return uint64(v), true
		}
	case 11:
		v := int64(n.Bits)
		if v >= 0 {
			return uint64(v), true
		}
	}
	return 0, false
}

// Float64 returns a scalar only when its conversion is exact and finite.
func (n Numeric) Float64() (float64, bool) {
	if n.Array != nil {
		return 0, false
	}
	var v float64
	switch n.Type {
	case 6:
		v = float64(math.Float32frombits(uint32(n.Bits)))
	case 12:
		v = math.Float64frombits(n.Bits)
	case 0, 2, 4, 10:
		if n.Bits > 1<<53 {
			return 0, false
		}
		v = float64(n.Bits)
	case 1:
		v = float64(int8(n.Bits))
	case 3:
		v = float64(int16(n.Bits))
	case 5:
		v = float64(int32(n.Bits))
	case 11:
		i := int64(n.Bits)
		if i < -(1<<53) || i > 1<<53 {
			return 0, false
		}
		v = float64(i)
	default:
		return 0, false
	}
	return v, !math.IsNaN(v) && !math.IsInf(v, 0)
}

// Tensor holds a validated directory entry; Offset is relative to DataOffset.
// Layer is present only when the tensor name identifies a layer.
type Tensor struct {
	Name       string
	Type       uint32
	Dimensions []uint64
	Offset     uint64
	Bytes      uint64
	Layer      *uint64
}

type Count struct {
	Present bool
	Value   uint64
}

// Template retains a digest, never template contents. Key identifies a named
// template, including the default key, and is bounded like metadata keys.
type Template struct {
	Key    string
	SHA256 [32]byte
}

// Shard contains only selected facts. Numeric keys are limited by the decoder
// to its documented architecture/dimension and sampler allowlist; this is not
// an unrestricted metadata store. Missing keys and counts mean unknown.
type Shard struct {
	Role         Role
	Version      uint32
	Architecture string
	Alignment    uint32
	DataOffset   uint64
	Numeric      map[string]Numeric
	Vocabulary   Count
	Merges       Count
	Templates    []Template
	Tensors      []Tensor
}

// Usage reports separately budgeted decoder work and retained facts. Scanned
// bytes are logical header/directory bytes, including skipped ranges. Binding
// samples are external to this package and are not included in these figures.
type Usage struct {
	ScannedHeaderBytes uint64
	RetainedBytes      uint64
	TensorCount        uint64
}

type Facts struct {
	Shards []Shard
	Usage  Usage
}
