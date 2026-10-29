<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Strata pure memory-plan protocol fixtures

Hand-built synthetic JSON based on read-only source at Strata
`37115e0bfbd4ffd051984b3176d6b56011e7f848`: `serve/memory_plan.py`,
`serve/test_memory_plan.py`, `serve/launch_plan.py`, and `docs/MEMORY_PLAN.md`.
No Strata code, Python, native transport, engine, GPU query, or credential
lookup was executed to create these files. These are protocol specimens,
not captured runtime output or memory accuracy oracles. This follows the
owner's explicit hand-built-JSON override of the plan's capture procedure.

The synthetic native component prices are deliberately arbitrary: each
device component is 10 bytes (total 130), each host component is 10 bytes
(total 80), and host commit is the separate combined capacity model (210).
Groups and combined views overlap; no sum of those views is an allocation
formula. Source-owned protected reserves and targets remain distinct.
Finite request: context 8192, prefill 1024, int8 KV, serial batch/parallel 1,
CUDA ordinal 0, hybrid, spec 2, no prefill borrowing, cache slots 2.
Minimum-layout scenario uses automatic prefill/cache and no resolved config.

| File | Synthetic scenario | Exit |
| --- | --- | --- |
| explicit-not-evaluated.json | Exact settings, no capacities or allowances | 0 |
| indeterminate.json | Capacities cover prices, additional costs unbounded | 0 |
| does-not-fit.json | Explicit zero device capacity | 0 |
| minimum-layout-does-not-fit.json | Minimum automatic layout exceeds zero device capacity | 0 |
| policy-threshold-shortfall.json | 130 priced, 200 policy threshold, 150 capacity, 50 shortfall | 0 |
| conditional-allowances.json | Explicit zero asserted bounds for every required unknown cost | 0 |
| after-runtime-indeterminate.json | Initial context already paid; late costs still uncovered | 0 |
| invalid-input.json | Sanitized refusal, no plan | 2 |
| cleanup-unconfirmed.json | Distinct cleanup refusal, no plan | 2 |

Raw files are immutable mapper inputs. Hostile/numeric/nonzero-allowance
variants are constructed in Go tests without rewriting these files.
Private expected identities in mapper tests come from the specimen.
Production callers must instead supply independently admitted identities
through `strataMemoryPlanBinding`; Task 4 owns that preparation and recheck.

Deterministic private root: `/synthetic/strata`. To reproduce synthetic
identity bytes in later process fixtures: metadata files `profile.bin`,
`index.txt`, `native_experts.txt` contain ASCII `synthetic metadata` followed
by LF (19 bytes); `dense.bin` contains ASCII `unread tensor payload` (21 bytes).
`model.gguf` is the 24-byte little-endian GGUF header (magic GGUF, version 3,
zero tensors, zero metadata pairs), followed by ASCII `unread tensors`
(38 bytes total). Header SHA-256 is recorded; full model content is unverified.
`synthetic.exe` contains this exact ASCII line plus LF and is never executed:

```text
Not executable: STRATA_BUILD/1:{"schema_version":1,"version":"fixture","commit":"synthetic","dirty":true,"built_at":"2026-10-29T00:00:00Z","cuda":"synthetic","backend":"cuda"}
```

Later process fakes may rebind only private requested executable/cwd/pack/
native-model/profile paths, input paths/file IDs, and matching query path
positions/cwd. Preserve bytes, hashes, config, build, accounting, qualities
and verdict; assert changes stay within that allowlist and keep identity
mismatch negatives. Runtime-source digest is independently computed by PAIR,
not a Strata response field. No process fake or production provider is
activated by this task.
