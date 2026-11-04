<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance extractor fixtures

All fixtures are synthetic documentation recordings or derived cases, retrieved
2026-10-20. They contain no real user content, installed-engine capture, MSI
execution evidence or native qualification. No approved versioned capture was
available to this task. These fixtures prove extraction, not installed-runtime
qualification. Content strings are empty; model identifiers and timestamps
are removed. JSON and stream fixtures carry their licensing/provenance here so
their protocol bytes remain valid.

## Pinned daily-engine schemas

The following fixtures are synthetic source-derived protocol examples, retrieved
2026-10-20. Every numeric example is invented; none is a native capture. Prompts,
generated text, reasoning text and tool arguments/results are absent. Empty
message/delta objects and empty reasoning strings retain placement evidence only.

- Schema `llamacpp-b12020-0ec4027a` binds llama.cpp b12020 at
  `0ec4027a2a0183de87c007069adddc06e665f7de`:
  [stats serialization](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/tools/server/server-common.cpp),
  [decode-step basis and per-token times](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/tools/server/server-common.h),
  [Chat serialization](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/tools/server/server-task.cpp).
  `llamacpp-chat-stream-usage.txt` places timings on the final empty-choices
  usage object. `llamacpp-chat-stream-no-usage.txt` places them on the finish
  object, with a tool finish and empty reasoning event. Both precede `[DONE]`.
  `llamacpp-chat-complete.json` is the corresponding single-object form.
  Output is 11, native decode work is 10 steps, duration is 250.5 ms, and
  the reported rate is approximately `10000 / 250.5` tokens/s.
- Schema `strata-f6726098` binds the owner's Strata fork at
  `f6726098e0545f9ffd4c392f72b20cc1045cd409`:
  [`request_timings`, `openai_chunks`, `openai_collect` and internal MCP rounds](https://github.com/corbinjurgens/Strata/blob/f6726098e0545f9ffd4c392f72b20cc1045cd409/serve/server.py).
  `strata-chat-stream.txt` places usage/timings on the final tool-finish
  object regardless of the usage option; `strata-chat-complete.json` is its
  collected shape. `strata-native-count-diff.json` deliberately reports
  output 20, duration 400 ms and rate 55 tokens/s. Its internal native count
  differs from visible output and is never reconstructed.
- `daily-invalid.json` is a synthetic case table containing already separated
  JSON payloads and typed expected metadata. It preserves duplicate keys and
  escaped spellings, invalid counts/durations, contradictory counters/rates,
  absent native timing and a nonfinite numeric string. Inline cases cover
  invalid nonfinite JSON syntax, explicit zero, schema rejection, optional
  reciprocal timing, internal MCP markers and exact decimal boundaries.

Counts remain integer-only. Daily milliseconds use bounded exact decimal
scaling with nearest-nanosecond rounding, halves upward; values above the phase
limit are rejected before rounding. Decimal lexemes are limited to 64 bytes and
exponents to magnitude 308. Rates stay finite, nonnegative and within T1 limits.
llama rates use a relative tolerance of `1e-6` plus half-nanosecond interval
uncertainty. Strata reciprocal checks use the overlapping rounding intervals
of 0.05 tokens/s and 0.0005 ms; prompt checks also allow 0.05 ms duration
uncertainty. Missing auxiliary rates leave independent prompt pairs intact.
Contradictions clear the affected phase, preserving unrelated output counts.
Strata MCP markers disqualify phase timing without returning their contents.

Daily schemas preserve `EngineDecodeNs` and the reported native rate separately
from output count and leave legacy `DecodeNs` unknown. Usage-only responses do
not establish evaluated prompt work or native speed. Pinned schema selection
is required for both streaming and pure nonstream extraction. Native MSI/PC3
captures and installed-version qualification remain pending; these fixtures
provide no host, completion, integration or deployment evidence.
