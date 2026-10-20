<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance extractor fixtures

All fixtures are synthetic documentation recordings or derived cases, retrieved
2026-10-20. They contain no real user content, installed-engine capture, MSI
execution evidence or native qualification. No approved versioned capture was
available to this task. Content strings are empty; model identifiers and timestamps
are removed. JSON and stream fixtures carry their licensing/provenance here so
their protocol bytes remain valid.

- `ollama-complete.json`: sanitized response example in the default 200 response
  block of [Generate a chat message](https://docs.ollama.com/api/chat).
  Numeric fields are unchanged: output 18, decode 52,479,709 ns, load
  101,397,084 ns, prompt evaluation 13,074,791 ns and total 174,560,334 ns.
  The filename denotes a complete terminal payload, not nonstreaming qualification.
- `ollama-stream.txt`: synthetic NDJSON placement with an empty partial response
  followed by the sanitized default 200 response example from
  [Generate a response](https://docs.ollama.com/api/generate), whose numeric
  fields match the chat example. The extractor receives each JSON payload directly.
- `ollama-openai-complete.json` and `ollama-openai-stream.txt`: synthetic
  serializations of the published Ollama v0.12.6
  [TestToUsage example](https://github.com/ollama/ollama/blob/v0.12.6/openai/openai_test.go#L103-L122)
  (prompt 10, completion 20, total 30), using the `Usage`, `ChatCompletion`
  and `ChatCompletionChunk` shapes in that version's
  [OpenAI conversion](https://github.com/ollama/ollama/blob/v0.12.6/openai/openai.go).
  Empty content/choices and SSE placement are synthetic structural scaffolding.
  The tag identifies source provenance, not an installed engine version.
- `missing-usage.json` and `malformed.txt`: synthetic absent-usage and truncated
  JSON cases. Inline invalid, duplicate, limit and zero examples in the tests are
  also synthetic.

The native references describe nanosecond decode/load durations and uncached
prompt evaluation duration. Only the latter duration is preserved as `PrefillNs`;
prompt/cache token counts stay unknown because installed-version cache semantics
are unproven. `total_duration` never supplies a proxy latency or decode denominator.
Compatibility usage establishes output count only; timing extensions are ignored.
Native nonstreaming throughput remains unqualified.

Stream fixtures document terminal placement only. Test code selects their JSON
payloads without installing another production framing or terminal detector.
SC's reader remains the sole completion authority. Repeated extraction is pure;
attempt uniqueness and publication belong to Phase B.

Numeric fields accept integer JSON tokens within T1 limits. Duplicate fields
(including escaped keys) invalidate that field; duplicate usage containers
invalidate usage. Malformed JSON invalidates the payload. A bad optional number
leaves unrelated valid numbers intact. An impossible decode pair retains its
valid count and suppresses its duration. Explicit zero stays distinct from absent.
