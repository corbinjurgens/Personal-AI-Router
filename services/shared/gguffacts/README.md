<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GGUF facts

This package decodes selected GGUF v2/v3 metadata and tensor directories from
caller-supplied readers and declared sizes. It does not open files, discover
shards, read directory padding or read tensor data. The caller owns closing
readers, artifact resolution, content fencing and sampled binding checks. Those
binding samples are separate I/O and may include tensor bytes. A cooperative
five-second deadline checks cancellation between bounded reads; it cannot
interrupt an operating-system read already in progress.

Supply artifacts in main, draft, companion order, omitting absent groups.
Readers and sizes stay caller-owned. A projector is a companion, never inferred
to be a main or draft model. Duplicate tensor names are rejected within each
role's shard set; independent roles may use identical names. This is a facts
reader, not a model-family detector, shard resolver or native capability check.

Production limits are fixed in `types.go`:

| Resource | Inclusive ceiling | Scope |
| --- | ---: | --- |
| Supplied artifacts | 1024 | Entire supplied set |
| Metadata entries | 65536 | Each shard |
| Tensor entries | 262144 | Entire supplied set |
| Skipped array elements | 1048576 | Each array |
| Retained numeric array elements | 4096 | Each array |
| Key and tensor name bytes | 256 | Each UTF-8 name |
| Metadata string bytes | 1048576 | Each string; architecture additionally limited to 256 bytes |
| Conservative retained charge | 67108864 (64 MiB) | Entire supplied set |
| Logical scanned header/directory bytes | 134217728 (128 MiB) | Entire supplied set |

The tensor, retained and scanned budgets include main, draft and companion
shards together. Keys, tensor names and architecture names must be nonempty
UTF-8 without embedded NUL bytes. Every count and length is checked before count-sized
allocation. Minimum directory/file sizes are checked before allocating maps or
tensor descriptors; numeric payload ranges are checked before numeric array
allocation. Unsupported or excessive declarations stop without reading their
payload, except that read-ahead (below) can already hold the start of a later
item. Internal lower limits let tests exercise complete economical fixtures
at exact boundaries as well as full production declarations.

Retained accounting is a conservative cumulative allocation charge, not a
measurement of Go heap use. It includes result/shard descriptors (352 bytes per
artifact), duplicate-key map storage (128 bytes per declared metadata entry),
tensor descriptors and per-role duplicate-name maps (192 bytes per tensor),
staged and retained name/architecture/identity bytes (twice the length plus 32 bytes),
numeric map entries (128 bytes plus key length plus 16 bytes) and numeric
backing storage rounded up to a power of two. Tensor dimension storage includes
16 bytes of rounding allowance, and layer pointers charge 16 bytes. Template
arrays grow with explicit capacities; the full new capacity times 48 bytes plus
16 bytes is charged before each allocation, including unused capacity. Earlier
charges are not refunded, including temporary duplicate keys, staging strings
and discarded foreign architecture candidates. Fixed bounded streaming scratch
buffers are not retained facts. `Clone` detaches mutable collections;
`RetainedSize` preserves this conservative charge for cache sizing.

Scanned bytes count every logical byte from each artifact's magic through its
last tensor descriptor (or the last metadata value when there are no tensors),
including discarded strings and skipped array contents. Reads are sequential,
each byte is read once, and reads stop at the directory end. Once a string's
length, an array's count or a value's size has been checked, a read extends
ahead by up to 64 KiB, but never past the smallest encoding of the declared
items not yet decoded: each remaining array element (8 bytes for a string),
metadata key (14 bytes) and tensor descriptor (33 bytes) still has to follow.
A tokenizer vocabulary is hundreds of thousands of values, and one read per
value is one system call each, which on Windows took longer than the deadline.
Read-ahead also stays within the scanned-byte ceiling. Alignment is computed
without reading padding; tensor ranges are validated arithmetically against the
declared file size. Tensor offsets must be aligned, nonoverlapping and ordered.
No tensor payload is read, including for quantized row validation. External
binding head/tail samples have a separate I/O budget and must be accounted in
addition to these header bytes, even if the sample and header ranges overlap.
This package's `Usage` never includes those samples or claims a full-file hash.

The operation has one five-second cooperative context deadline across the
entire set, with checks between entries and before and after bounded reads.
There are no parser goroutines to abandon. A caller-supplied `ReaderAt` that
blocks can exceed the deadline until it returns. Errors include `ErrLimit`,
`ErrInvalid` (with an explicit unsupported GGML type diagnostic), I/O errors
and context errors. Facts are returned only on success. Missing metadata
remains absent.

`Shard.General` retains scalar string keys `general.name`, `general.basename`,
`general.size_label`, `general.quantized_by` and `general.repo_url`, each bounded
to 256 bytes of UTF-8 without NUL. Oversized or invalid identity strings and
mistyped values (including arrays) are consumed and ignored. The global string,
array, scanned and retained resource ceilings still apply, and I/O failures
still fail decoding. `general.file_type` retains only a scalar uint32 (type 4),
with a 16-byte pointer allocation charge; other types leave it absent. These
optional identity rules do not relax validation of other metadata.

Selected numeric keys are explicitly listed in `reader.go`, restricted to the
reported architecture or the sampler keys. Before `general.architecture`
arrives, only candidates with an allowlisted suffix are retained, under the
same budgets; afterward, foreign prefixes are removed. Numeric bits retain the
GGUF primitive type and exact representation. `Uint64` returns only nonnegative
integer scalars; `Float64` returns only exact, finite scalar conversions. Token
and merge strings are discarded after UTF-8 validation, retaining only their
counts. Chat templates are hashed while streaming; only key/presence and SHA256
survive. Tensor descriptors retain names, shapes, type, offset, calculated bytes
and syntactically recognized `blk.N` layer indices. Other metadata is validated
and discarded, with no unrestricted metadata map in the returned facts.

Format and type-size references are the MIT-licensed llama.cpp sources at
commit `7fe450e19305b828c199d602c23a8337aaa1f03b`: `ggml/include/gguf.h`,
`ggml/src/gguf.cpp`, `ggml/include/ggml.h`,
`ggml/src/ggml.c` type traits, `ggml/src/ggml-common.h` block definitions,
`gguf-py/gguf/gguf_reader.py`, and `gguf-py/gguf/constants.py`.
The sources are copyright (c) 2023-2026 The ggml authors, licensed under MIT
([pinned license](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/LICENSE)).
The decoder is independently implemented from the format and numeric layout
facts; no upstream parser implementation is copied and no parser dependency,
mmap, subprocess or network parser is used. The package's implementation is
Apache-2.0 as marked in each source file.

Synthetic tests use counting readers that record attempted excess reads, exact
boundary fixtures and aggregate cross-shard/role limits. The real sparse-file
fixture is generated in `t.TempDir`, is smaller than 16 MiB, leaves a sentinel
in the tensor region and asserts the actual read ceiling is the directory end.
No model fixtures are committed.
