// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package gguffacts

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

var ErrLimit = errors.New("gguf resource limit")
var ErrInvalid = errors.New("invalid or unsupported gguf")

type limits struct{ artifacts, metadata, tensors, skipped, numeric, name, str, retained, scanned uint64 }

var productionLimits = limits{MaxArtifacts, MaxMetadataPerShard, MaxTensorsPerSet, MaxSkippedArray, MaxNumericArray, MaxNameBytes, MaxStringBytes, MaxRetainedBytes, MaxScannedHeaderBytes}

type decoder struct {
	ctx       context.Context
	lim       limits
	usage     Usage
	r         io.ReaderAt
	size, off uint64
	order     binary.ByteOrder
	names     map[string]struct{} // Tensor names in the current role's shard set.
}

func (d *decoder) charge(n uint64) error {
	if err := d.ctx.Err(); err != nil {
		return err
	}
	if d.usage.RetainedBytes > d.lim.retained || n > d.lim.retained-d.usage.RetainedBytes {
		return ErrLimit
	}
	d.usage.RetainedBytes += n
	return nil
}

// require checks a complete declared range before any allocation or read for it.
func (d *decoder) require(n uint64) error {
	if err := d.ctx.Err(); err != nil {
		return err
	}
	if d.usage.ScannedHeaderBytes > d.lim.scanned || n > d.lim.scanned-d.usage.ScannedHeaderBytes {
		return ErrLimit
	}
	if d.off > d.size || n > d.size-d.off {
		return io.ErrUnexpectedEOF
	}
	return nil
}

func (d *decoder) advance(n uint64) error {
	if err := d.require(n); err != nil {
		return err
	}
	d.off += n
	d.usage.ScannedHeaderBytes += n
	return nil
}

func (d *decoder) read(p []byte) error {
	start := d.off
	if err := d.advance(uint64(len(p))); err != nil {
		return err
	}
	if start > math.MaxInt64 {
		return ErrInvalid
	}
	n, err := d.r.ReadAt(p, int64(start))
	if e := d.ctx.Err(); e != nil {
		return e
	}
	if n != len(p) {
		if err != nil {
			return err
		}
		return io.ErrUnexpectedEOF
	}
	return err
}

func (d *decoder) number(n int) (uint64, error) {
	var b [8]byte
	if err := d.read(b[:n]); err != nil {
		return 0, err
	}
	switch n {
	case 1:
		return uint64(b[0]), nil
	case 2:
		return uint64(d.order.Uint16(b[:2])), nil
	case 4:
		return uint64(d.order.Uint32(b[:4])), nil
	case 8:
		return d.order.Uint64(b[:]), nil
	}
	return 0, ErrInvalid
}

// text streams and validates UTF-8 without retaining discarded strings.
func (d *decoder) text(max uint64, keep bool, digest bool) (string, [32]byte, error) {
	var sum [32]byte
	n, err := d.number(8)
	if err != nil {
		return "", sum, err
	}
	if n > max {
		return "", sum, ErrLimit
	}
	if err := d.require(n); err != nil {
		return "", sum, err
	}
	if keep {
		// Account for both the staging byte slice and the resulting string.
		if err := d.charge(n*2 + 32); err != nil {
			return "", sum, err
		}
	}
	var out []byte
	if keep {
		out = make([]byte, 0, int(n))
	}
	h := sha256.New()
	var buf [4096 + utf8.UTFMax]byte
	carry := 0
	for n > 0 {
		part := n
		if part > 4096 {
			part = 4096
		}
		if err := d.read(buf[carry : carry+int(part)]); err != nil {
			return "", sum, err
		}
		if keep {
			out = append(out, buf[carry:carry+int(part)]...)
		}
		if digest {
			_, _ = h.Write(buf[carry : carry+int(part)])
		}
		n -= part
		end := carry + int(part)
		i := 0
		for i < end {
			if !utf8.FullRune(buf[i:end]) && n > 0 {
				break
			}
			r, width := utf8.DecodeRune(buf[i:end])
			if r == utf8.RuneError && width == 1 {
				return "", sum, ErrInvalid
			}
			i += width
		}
		carry = copy(buf[:], buf[i:end])
	}
	if carry != 0 {
		return "", sum, ErrInvalid
	}
	if digest {
		copy(sum[:], h.Sum(nil))
	}
	return string(out), sum, nil
}

func width(t uint32) int {
	switch t {
	case 0, 1, 7:
		return 1
	case 2, 3:
		return 2
	case 4, 5, 6:
		return 4
	case 10, 11, 12:
		return 8
	}
	return 0
}

var dimensionKeys = map[string]bool{
	"vocab_size": true, "context_length": true, "embedding_length": true, "embedding_length_out": true, "block_count": true, "leading_dense_block_count": true, "feed_forward_length": true, "expert_feed_forward_length": true, "expert_shared_feed_forward_length": true, "expert_count": true, "expert_used_count": true, "expert_shared_count": true, "expert_group_count": true, "expert_group_used_count": true, "expert_latent_length": true, "experts_per_group": true, "moe_every_n_layers": true, "moe_latent_size": true, "nextn_predict_layers": true, "full_attention_interval": true,
	"attention.head_count": true, "attention.head_count_kv": true, "attention.key_length": true, "attention.value_length": true, "attention.key_length_swa": true, "attention.value_length_swa": true, "attention.key_length_mla": true, "attention.value_length_mla": true, "attention.kv_lora_rank": true, "attention.q_lora_rank": true, "attention.sliding_window": true, "attention.sliding_window_pattern": true, "attention.recurrent_layers": true, "attention.shared_kv_layers": true,
	"ssm.conv_kernel": true, "ssm.inner_size": true, "ssm.state_size": true, "ssm.time_step_rank": true, "ssm.group_count": true, "shortconv.l_cache": true, "wkv.head_size": true, "rope.dimension_count": true, "rope.dimension_count_swa": true, "rope.dimension_sections": true, "rope.scaling.original_context_length": true, "rope.scaling.factor": true,
}
var samplerKeys = map[string]bool{"general.sampling.top_k": true, "general.sampling.top_p": true, "general.sampling.min_p": true, "general.sampling.temp": true, "general.sampling.xtc_probability": true, "general.sampling.xtc_threshold": true, "general.sampling.penalty_last_n": true, "general.sampling.penalty_repeat": true, "general.sampling.mirostat": true, "general.sampling.mirostat_tau": true, "general.sampling.mirostat_eta": true}

func candidate(key string) bool {
	if samplerKeys[key] {
		return true
	}
	i := strings.IndexByte(key, '.')
	return i > 0 && dimensionKeys[key[i+1:]]
}

// Decode parses only the supplied artifacts, with one cooperative deadline and
// shared resource ceilings. It never reads tensor data or directory padding.
func Decode(ctx context.Context, artifacts []Artifact) (Facts, error) {
	return decode(ctx, artifacts, productionLimits)
}

func decode(ctx context.Context, artifacts []Artifact, lim limits) (Facts, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	d := decoder{ctx: ctx, lim: lim}
	if err := ctx.Err(); err != nil {
		return Facts{}, err
	}
	if uint64(len(artifacts)) > lim.artifacts {
		return Facts{}, ErrLimit
	}
	// Validate roles before allocating the result, preserving caller ownership.
	lastRole := 0
	for _, a := range artifacts {
		if err := ctx.Err(); err != nil {
			return Facts{}, err
		}
		rank := 0
		switch a.Role {
		case RoleMain:
			rank = 1
		case RoleDraft:
			rank = 2
		case RoleCompanion:
			rank = 3
		}
		if a.Reader == nil || a.Size < 0 || rank == 0 || rank < lastRole {
			return Facts{}, ErrInvalid
		}
		lastRole = rank
	}
	if err := d.charge(uint64(len(artifacts)) * 256); err != nil {
		return Facts{}, err
	}
	f := Facts{Shards: make([]Shard, len(artifacts))}
	for i, a := range artifacts {
		if i == 0 || a.Role != artifacts[i-1].Role {
			d.names = make(map[string]struct{})
		}
		d.r, d.size, d.off = a.Reader, uint64(a.Size), 0
		s, err := d.shard(a.Role)
		if err != nil {
			return Facts{}, fmt.Errorf("shard %d: %w", i, err)
		}
		f.Shards[i] = s
	}
	f.Usage = d.usage
	if err := ctx.Err(); err != nil {
		return Facts{}, err
	}
	return f, nil
}

func (d *decoder) shard(role Role) (Shard, error) {
	s := Shard{Role: role, Alignment: 32}
	var head [8]byte
	if err := d.read(head[:]); err != nil {
		return s, err
	}
	if string(head[:4]) != "GGUF" {
		return s, ErrInvalid
	}
	le, be := binary.LittleEndian.Uint32(head[4:]), binary.BigEndian.Uint32(head[4:])
	if le == 2 || le == 3 {
		d.order = binary.LittleEndian
		s.Version = le
	} else if be == 2 || be == 3 {
		d.order = binary.BigEndian
		s.Version = be
	} else {
		return s, ErrInvalid
	}
	nt, err := d.number(8)
	if err != nil {
		return s, err
	}
	nm, err := d.number(8)
	if err != nil {
		return s, err
	}
	if nm > d.lim.metadata || nt > d.lim.tensors-d.usage.TensorCount {
		return s, ErrLimit
	}
	// A nonempty key plus the smallest scalar is at least 14 bytes; a
	// nonempty tensor name with one dimension is at least 33. Check these
	// lower bounds before count-sized maps or slices. Production counts make
	// these products safe; internal test limits may only lower the ceilings.
	if err = d.require(nm*14 + nt*33); err != nil {
		return s, err
	}
	// Charge conservative map buckets and keys for duplicate detection before allocation.
	if err = d.charge(nm*128 + nt*192); err != nil {
		return s, err
	}
	seen := make(map[string]struct{}, int(nm))
	s.Numeric = make(map[string]Numeric)
	for i := uint64(0); i < nm; i++ {
		key, _, e := d.text(d.lim.name, true, false)
		if e != nil {
			return s, e
		}
		if _, ok := seen[key]; ok || key == "" || strings.IndexByte(key, 0) >= 0 {
			return s, ErrInvalid
		}
		seen[key] = struct{}{}
		t, e := d.number(4)
		if e != nil {
			return s, e
		}
		if e = d.metadata(&s, key, uint32(t)); e != nil {
			return s, e
		}
	}
	for key := range s.Numeric {
		if err := d.ctx.Err(); err != nil {
			return s, err
		}
		if !samplerKeys[key] && (s.Architecture == "" || !strings.HasPrefix(key, s.Architecture+".")) {
			delete(s.Numeric, key)
		}
	}
	if err = d.require(nt * 33); err != nil {
		return s, err
	}
	s.Tensors = make([]Tensor, int(nt))
	d.usage.TensorCount += nt
	for i := range s.Tensors {
		t, e := d.tensor()
		if e != nil {
			return s, e
		}
		if _, ok := d.names[t.Name]; ok {
			return s, ErrInvalid
		}
		d.names[t.Name] = struct{}{}
		s.Tensors[i] = t
	}
	data := d.off
	if nt > 0 {
		if uint64(s.Alignment)-1 > math.MaxUint64-data {
			return s, ErrInvalid
		}
		data = (data + uint64(s.Alignment) - 1) &^ (uint64(s.Alignment) - 1)
	}
	if data > d.size {
		return s, io.ErrUnexpectedEOF
	}
	s.DataOffset = data
	var end uint64
	for _, t := range s.Tensors {
		if err := d.ctx.Err(); err != nil {
			return s, err
		}
		if t.Offset%uint64(s.Alignment) != 0 || t.Offset < end || t.Offset > d.size-data || t.Bytes > d.size-data-t.Offset {
			return s, ErrInvalid
		}
		end = t.Offset + t.Bytes
	}
	return s, nil
}

func (d *decoder) metadata(s *Shard, key string, typ uint32) error {
	array := typ == 9
	var count uint64 = 1
	if array {
		x, e := d.number(4)
		if e != nil {
			return e
		}
		typ = uint32(x)
		count, e = d.number(8)
		if e != nil {
			return e
		}
		if typ == 9 {
			return ErrInvalid
		}
	}
	isTemplate := key == "tokenizer.chat_template" || strings.HasPrefix(key, "tokenizer.chat_template.")
	if (key == "general.architecture" || isTemplate) && (array || typ != 8) {
		return ErrInvalid
	}
	retain := candidate(key)
	if array {
		if retain && count > d.lim.numeric || !retain && count > d.lim.skipped {
			return ErrLimit
		}
	}
	if key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges" {
		if !array || typ != 8 {
			return ErrInvalid
		}
		if key == "tokenizer.ggml.tokens" {
			s.Vocabulary = Count{true, count}
		} else {
			s.Merges = Count{true, count}
		}
	}
	if typ == 8 {
		if retain || key == "general.alignment" {
			return ErrInvalid
		}
		for i := uint64(0); i < count; i++ {
			architecture := key == "general.architecture" && !array
			template := isTemplate
			max := d.lim.str
			if architecture && d.lim.name < max {
				max = d.lim.name
			}
			v, sum, e := d.text(max, architecture, template)
			if e != nil {
				return e
			}
			if architecture {
				if len(v) > int(d.lim.name) || v == "" || strings.IndexByte(v, 0) >= 0 {
					return ErrInvalid
				}
				s.Architecture = v
			}
			if template {
				if len(s.Templates) == cap(s.Templates) {
					capacity := 2 * cap(s.Templates)
					if capacity == 0 {
						capacity = 1
					}
					// Charge the full new backing array, including unused
					// capacity, before growth. Earlier charges are not refunded.
					if e = d.charge(uint64(capacity)*48 + 16); e != nil {
						return e
					}
					templates := make([]Template, len(s.Templates), capacity)
					copy(templates, s.Templates)
					s.Templates = templates
				}
				s.Templates = append(s.Templates, Template{key, sum})
			}
		}
		return nil
	}
	w := width(typ)
	if w == 0 {
		return ErrInvalid
	}
	if key == "general.architecture" || (key == "general.alignment" && (array || typ != 4)) {
		return ErrInvalid
	}
	if count > math.MaxUint64/uint64(w) {
		return ErrInvalid
	}
	bytes := count * uint64(w)
	if err := d.require(bytes); err != nil {
		return err
	}
	if !retain && key != "general.alignment" {
		remaining := bytes
		var buf [4096]byte
		for remaining > 0 {
			n := remaining
			if n > uint64(len(buf)) {
				n = uint64(len(buf))
			}
			if e := d.read(buf[:int(n)]); e != nil {
				return e
			}
			if typ == 7 {
				for _, value := range buf[:int(n)] {
					if value > 1 {
						return ErrInvalid
					}
				}
			}
			remaining -= n
		}
		return nil
	}
	var v Numeric
	v.Type = ValueType(typ)
	if retain {
		// Round backing-storage charges up to a power of two. Go allocation
		// classes can round beyond the requested slice capacity; charging only
		// len*8 plus a descriptor would undercount large numeric arrays.
		storage := uint64(0)
		if array && count > 0 {
			storage = 1
			for storage < count {
				storage *= 2
			}
			storage *= 8
		}
		if err := d.charge(128 + uint64(len(key)) + storage + 16); err != nil {
			return err
		}
		if array {
			v.Array = make([]uint64, int(count))
		}
	}
	for i := uint64(0); i < count; i++ {
		bits, e := d.number(w)
		if e != nil {
			return e
		}
		if typ == 7 && bits > 1 {
			return ErrInvalid
		}
		if key == "general.alignment" {
			if bits == 0 || bits&(bits-1) != 0 {
				return ErrInvalid
			}
			s.Alignment = uint32(bits)
		}
		if retain {
			if array {
				v.Array[i] = bits
			} else {
				v.Bits = bits
			}
		}
	}
	if retain {
		s.Numeric[key] = v
	}
	return nil
}

// GGML type ID -> elements per block, bytes per block, from the pinned MIT
// ggml type traits and GGML_QUANT_SIZES reference documented in README.md.
// Removed on-disk repacking formats are intentionally absent.
var tensorTypes = map[uint32][2]uint64{
	0: {1, 4}, 1: {1, 2}, 2: {32, 18}, 3: {32, 20},
	6: {32, 22}, 7: {32, 24}, 8: {32, 34}, 9: {32, 36},
	10: {256, 84}, 11: {256, 110}, 12: {256, 144}, 13: {256, 176},
	14: {256, 210}, 15: {256, 292}, 16: {256, 66}, 17: {256, 74},
	18: {256, 98}, 19: {256, 50}, 20: {32, 18}, 21: {256, 110},
	22: {256, 82}, 23: {256, 136}, 24: {1, 1}, 25: {1, 2},
	26: {1, 4}, 27: {1, 8}, 28: {1, 8}, 29: {256, 56},
	30: {1, 2}, 34: {256, 54}, 35: {256, 66}, 39: {32, 17},
	40: {64, 36}, 41: {128, 18}, 42: {64, 18},
}

func (d *decoder) tensor() (Tensor, error) {
	var t Tensor
	name, _, err := d.text(d.lim.name, true, false)
	if err != nil {
		return t, err
	}
	if name == "" || strings.IndexByte(name, 0) >= 0 {
		return t, ErrInvalid
	}
	t.Name = name
	n, err := d.number(4)
	if err != nil {
		return t, err
	}
	if n == 0 || n > 4 {
		return t, ErrInvalid
	}
	if err = d.require(n*8 + 12); err != nil {
		return t, err
	}
	if err = d.charge(n*8 + 16); err != nil {
		return t, err
	}
	t.Dimensions = make([]uint64, int(n))
	elements := uint64(1)
	for i := range t.Dimensions {
		x, e := d.number(8)
		if e != nil {
			return t, e
		}
		if x == 0 || x > math.MaxInt64 {
			return t, ErrInvalid
		}
		if elements > math.MaxInt64/x {
			return t, ErrInvalid
		}
		elements *= x
		t.Dimensions[i] = x
	}
	typ, err := d.number(4)
	if err != nil {
		return t, err
	}
	t.Type = uint32(typ)
	layout, ok := tensorTypes[t.Type]
	if !ok {
		return t, fmt.Errorf("%w: unsupported GGML type %d", ErrInvalid, t.Type)
	}
	if t.Dimensions[0]%layout[0] != 0 {
		return t, ErrInvalid
	}
	size := t.Dimensions[0] / layout[0]
	if size > math.MaxUint64/layout[1] {
		return t, ErrInvalid
	}
	size *= layout[1]
	for _, x := range t.Dimensions[1:] {
		if size > math.MaxUint64/x {
			return t, ErrInvalid
		}
		size *= x
	}
	t.Bytes = size
	t.Offset, err = d.number(8)
	if err != nil {
		return t, err
	}
	if strings.HasPrefix(name, "blk.") {
		layer, _, ok := strings.Cut(name[len("blk."):], ".")
		if ok {
			if n, e := strconv.ParseUint(layer, 10, 64); e == nil {
				if err = d.charge(16); err != nil {
					return t, err
				}
				t.Layer = new(uint64)
				*t.Layer = n
			}
		}
	}
	return t, nil
}

// RetainedSize is the conservative allocation charge made by Decode.
func (f Facts) RetainedSize() uint64 { return f.Usage.RetainedBytes }

// Clone detaches all mutable collections from the source facts.
func (f Facts) Clone() Facts {
	out := f
	out.Shards = make([]Shard, len(f.Shards))
	for i, s := range f.Shards {
		out.Shards[i] = s
		dst := &out.Shards[i]
		dst.Numeric = make(map[string]Numeric, len(s.Numeric))
		for k, v := range s.Numeric {
			if v.Array != nil {
				v.Array = append(make([]uint64, 0, len(v.Array)), v.Array...)
			}
			dst.Numeric[k] = v
		}
		dst.Templates = append([]Template(nil), s.Templates...)
		dst.Tensors = make([]Tensor, len(s.Tensors))
		for j, t := range s.Tensors {
			dst.Tensors[j] = t
			dst.Tensors[j].Dimensions = append([]uint64(nil), t.Dimensions...)
			if t.Layer != nil {
				v := *t.Layer
				dst.Tensors[j].Layer = &v
			}
		}
	}
	return out
}
