<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution plan: B extension — qualification for the OpenCode request shapes actually used

Planned against `fork/groundwork` at commit `584e99242562df1cb19d2c7a318b7342e03cb7d2` (newer than 17727f1; the files this plan touches are unchanged between the two). Path shorthand: P/ = `services/nvpair-proxy/`, N/ = `services/shared/`, E/ = `services/nvpair-engine-manager/`.

## Goal

A node's local qualification evidence states, per qualified (engine, model, launch), which OpenCode request shapes it was measured for: function tools, `parallel_tool_calls`, reasoning replay (`reasoning_content` on assistant messages), `reasoning_effort`, and structured output. A strict `/v1/chat/completions` request that uses a shape the installed qualification does not declare is answered 400 `strict-qualification` before any wake, unload, slot or inference, naming the field only. A request using only declared shapes is accounted (including the bytes of a replayed `reasoning_content` and of `reasoning_effort`) and forwarded with the profile's defaults merged in, exactly as today. A profile that declares a routing capability the qualification does not cover is not installed. Synthetic, redacted fixtures mirror the bodies the OpenCode client (Vercel AI SDK, `@ai-sdk/openai-compatible`) sends, and a fixture-driven test pins which are accepted and which are refused under each shape declaration. Docs and the proxy spec describe the new field and rules. Nothing in the JSON-RPC surface, the broker, engine-manager, desktop or `services/tests` changes.

## Change scope

**Intent.** Close the gap between "the evidence file says this model was qualified" and "the request uses tools / reasoning / structured output the template may not render", for the shapes OpenCode actually emits, and nothing else.

**Scope: Bounded improvement.** May change: P/`local_qualification.go` (artifact schema version, `shapes`, request gating and accounting), P/`readiness_contract.go` (one extra install condition), P/`local_qualification_test.go`, P/`testdata/local-qualification.json`, new P/`testdata/opencode-shapes.json`, new P/`opencode_shapes_test.go`, `docs/local-qualification.mdx`, P/`spec.md`.

**Basis.** The current validator (P/`local_qualification.go:242-545`) accepts `tools`, `tool_choice`, assistant `tool_calls`, `tool` messages and `json_object`/`json_schema` formats for every qualification, with no evidence that the chat template renders tools or schemas. It rejects `reasoning_content` on assistant messages (`decodeExact` on `chatMessage`, lines 378-384 and 403), `parallel_tool_calls` and `reasoning_effort` (not in `chatRequestKeys`, 242-245) for every qualification, including one measured with them. Both directions are wrong for a qualification-driven boundary: the evidence, not the code, should say which shapes hold.

**Compatibility.** The artifact document version moves from 1 to 2 and `shapes` becomes required. No real evidence file exists (`docs/local-qualification.mdx:14-19`: "No node ships with qualification evidence"), so nothing deployed is invalidated. A version-1 document is rejected as a whole, as any other version is today (`local_qualification.go:174-176`). No wire shape (`nodepolicy`), JSON-RPC method, advertisement field or desktop bridge changes, so `npm --prefix desktop run service-contracts:check` must pass unchanged.

**Non-goals.**
- Vision/image parts, audio, files, `n`, `logprobs`, `user`, `verbosity`, `store`, `chat_template_kwargs`, context overrides: stay rejected (not OpenCode shapes this fork needs; `user`/`verbosity` are sent by the AI SDK only when the OpenCode provider options set them).
- Advertising shapes to peers or routers (no `AdvertisedRole` change). The destination validates every request itself (`docs/local-qualification.mdx:104-106`).
- Engine-side probing of template capabilities (llama.cpp `/props`, Strata). Shape support is operator evidence bound to the launch digest, like the tokenizer and template names already are.
- Producing model content digests for llama.cpp or Strata. Today `E/manifests/llamacpp.json:98-101` and `E/testdata/strata/strata.json:29` declare no `digest` in `list_models`, so `E/model_identity.go:161-170` reports no model digests for them and `P/readiness_contract.go` cannot install any llama.cpp or Strata model. That prerequisite is outside this plan (see Owner questions).
- Changing the inactive Chat adapter library (E; P/`chat_adapter.go`), the Ollama path, or `chatRequestKeys` itself.

## Context the executor needs

Verified from the code at `584e992`:

- **Validator entry.** `(*localQualification).validateRequest(profile, route, body)` (P/`local_qualification.go:251-309`) is called by `qualifyStrict` (P/`readiness_contract.go:212-227`), which `serveStrictLocal` (P/`ingress.go:318-335`) calls before `AdmitStrict`; a `*strictQualificationError` becomes 400 with code `strict-qualification` (P/`ingress.go:331-335`). Reasons name fields and bounds only (P/`local_qualification.go:222-233`).
- **Top-level allowlist.** `chatRequestKeys` (P/`local_qualification.go:242-245`) = `model, messages, tools, tool_choice, response_format, max_tokens, max_completion_tokens, stream, stream_options, temperature, top_p, stop, seed, frequency_penalty, presence_penalty`. P/`chat_adapter.go:446` builds its own key list as `chatRequestKeys` plus `"parallel_tool_calls"` and serializes the effective body by iterating that list; adding `parallel_tool_calls` to `chatRequestKeys` would duplicate the key there. Do not modify `chatRequestKeys`.
- **Current acceptance per shape** (all unconditional, independent of evidence):
  - `tools`: array of `{type:"function", function:{name, description, parameters(object), strict}}`, any length including empty (P/`local_qualification.go:471-496`), each accounted `toolOverheadTokens + compactLen(tool)`.
  - `tool_choice`: `"none"|"auto"|"required"` or `{type:"function",function:{name}}`, accepted without `tools` (497-516).
  - messages: `{role, content, name, tool_calls, tool_call_id}` only (378-384, `decodeExact` at 403); roles `system|user|assistant|tool` (406-410); `tool_calls` only on assistant (429-431); `tool_call_id` only on tool (423-428); null content only on assistant with tool_calls (416-418); content string or array of `{type:"text",text}` parts (446-469; image parts rejected at 463).
  - `response_format`: `text` (0), `json_object` (`schemaOverheadTokens`), `json_schema` with nonempty name and object schema (520-545).
  - scalars: `temperature, top_p, frequency_penalty, presence_penalty` numbers; `seed` integer; `stream` boolean; `stream_options` exactly `{include_usage: bool}`, not tied to `stream`; `stop` string or ≤4 strings (311-352).
  - output: one of `max_tokens`/`max_completion_tokens` required, equal if both (356-376); ceiling `min(profile, qualification)` (286-288); input+output ≤ `min(profile.ContextTokens, q.ContextTokens)` (305-307).
  - Rejected today regardless of evidence: `parallel_tool_calls`, `reasoning_effort`, `user`, `verbosity`, `n`, `options`, any message field beyond the five above (so `reasoning_content`), any non-text content part.
- **Artifact.** `qualificationDocumentVersion = 1` (39); `localQualification` struct (65-80); `validate()` (98-156); `parseQualificationDocument` uses `DisallowUnknownFields` (161-190). Fixture: P/`testdata/local-qualification.json` (`{"artifact": {...}, "cases": [...]}`), driven by `TestQualificationFixtureCases` (P/`local_qualification_test.go:130-160`) through `qualificationFixture` (103-128), which builds the profile from the qualification. `TestQualificationDocumentRejects` (228-290) has a case `"another version"` expecting `{"version":2,...}` to be rejected (251). `testQualification` (25-35) is the in-memory qualification every strict harness test installs through `installQualified` (49-75).
- **Install conditions.** `qualifiedModelLocked` (P/`readiness_contract.go:125-153`): registry lookup for same engine and normalized model, identity present, launch digest and model digest covered, `requestOptions` equal, profile bounds ≤ qualified. `sameInstalledQualification` (169-172) compares the qualification pointer and the profile's bounds and defaults. `TestStrictCapabilityDerivation` (P/`local_qualification_test.go:310`) covers these conditions.
- **Profile capabilities.** `nodepolicy.Capability` with `tools|vision|structured|embeddings` (N/`nodepolicy/policy.go:71-81`) on `Profile.Capabilities` (344). Tier routing derives `needs` from the body: a body with `tools` needs `CapTools` (P/`tiers.go:88`) and is routed only to entries declaring it (P/`tiers.go:307-314`). Nothing ties a profile's capabilities to its qualification today.
- **HTTP test helpers.** `strictHTTPFixture`, `strictHTTPRequestWith`, `serveStrictHTTP` (P/`admission_http_test.go:28-60`); `newRecordingEngine` (280); `recordingEngine.hits()`/`lastBody()`; `TestStrictHTTPRejectsUnqualifiedRequestsBeforeSideEffects` (P/`local_qualification_test.go:460-519`) is the pattern for "rejected before side effects" on both `ingress` and `self` paths.
- **Engines.** Engine routing ids are validated by `engines.ValidRoutingID` (N/`engines/dynamic.go:79`); the built-in llama.cpp id is `llamacpp` (N/`engines/engines.go:141`); Strata is a descriptor engine with protocol `openai-chat` (`docs/strata.mdx:18-84`). llama.cpp per-model preset options other than the envelope table in E/`serving_llamacpp.go:74-180` (so `jinja`, `reasoning-format`, `chat-template-kwargs`) pass through to the preset unchanged; the launch digest covers executable, args and explicit env (E/`model_identity.go:68-85`).
- **Docs to update.** `docs/local-qualification.mdx`: example artifact 28-54, field table 56-68, accounting 73-86, validation list 127-145, digest paragraph 167-171. P/`spec.md:179-222` "Local qualification and identity (B)".
- **Existing tests pass** on this tree: `cd services/nvpair-proxy && go test ./...` → `ok`.

**What OpenCode sends — facts versus assumptions.**

Facts, read from `vercel/ai` `main` on 2026-10-20, package `@ai-sdk/openai-compatible` (`src/chat/openai-compatible-chat-language-model.ts`, `convert-to-openai-compatible-chat-messages.ts`, `openai-compatible-prepare-tools.ts`, `openai-compatible-chat-language-model-options.ts`, `src/openai-compatible-provider.ts`):

| Field | When the SDK sends it |
| --- | --- |
| `model`, `messages` | always |
| `max_tokens` | when the caller sets `maxOutputTokens` (never `max_completion_tokens`) |
| `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `seed` | when the caller sets them |
| `stream: true` | always on the streaming path |
| `stream_options: {"include_usage": true}` | only when the provider was created with `includeUsage: true`; otherwise absent |
| `tools` | when the caller passes tools: `{type:"function", function:{name, description, parameters, strict?}}`, `strict` only if set; `parameters` is the tool's JSON Schema |
| `tool_choice` | only when the caller sets `toolChoice`: `"auto"|"none"|"required"` or `{type:"function",function:{name}}`; never without tools |
| `parallel_tool_calls` | never |
| `response_format` | only for a JSON response format: `{type:"json_schema", json_schema:{schema, strict, name, description}}` when `supportsStructuredOutputs` is true, else `{type:"json_object"}` |
| `reasoning_effort` | only when provider option `reasoningEffort` is set, or the call's `reasoning` option is a custom string |
| `user`, `verbosity` | only when provider options `user` / `textVerbosity` are set |
| unknown provider-option keys | spread into the body as extra top-level fields (a configured `options` object can inject any field) |
| system message | `{role:"system", content:<string>}` |
| user message | `content` is a string for a single text part, otherwise an array of parts: `{type:"text",text}`, images as `{type:"image_url", image_url:{url}}`, PDFs as `{type:"file", ...}` |
| assistant message | `content` is the concatenated text, or `null` when there are tool calls and no text; `tool_calls[] = {id, type:"function", function:{name, arguments:<JSON string>}}`; **`reasoning_content: <string>` is emitted whenever the history holds reasoning parts** |
| tool message | one per tool result: `{role:"tool", tool_call_id, content:<string>}` (an array of parts only when `supportsMultiPartToolContent`) |
| response parsing | `delta.reasoning_content ?? delta.reasoning` becomes reasoning; `tool_calls` deltas keyed by `index` |

Assumptions to confirm on MSI (the OpenCode repository itself was not readable from the planning environment: raw GitHub paths answered 404 and the GitHub API 403):

1. OpenCode's custom provider uses `@ai-sdk/openai-compatible` (configured by `npm` in `opencode.json`) and therefore the table above.
2. OpenCode sets `maxOutputTokens` on every call, so `max_tokens` is present. If it ever omits it, the profile's `requestOptions` can supply `{"max_tokens": N}`: `mergeDefaults` runs before `outputBound` (P/`local_qualification.go:267`, 282).
3. Whether OpenCode creates the provider with `includeUsage: true` (then `stream_options` appears; accepted either way).
4. Whether OpenCode sets `toolChoice` (then `tool_choice` appears; accepted under the tools shape).
5. Whether OpenCode keeps reasoning parts in the replayed history, so `reasoning_content` appears on assistant messages. This is the "reasoning replay" the owner milestone names; the plan supports both answers through evidence.
6. Whether OpenCode's `options`/variants configuration sets `reasoningEffort` for the local model (then `reasoning_effort` appears).
7. Which temperature/top_p values OpenCode uses for the local model (accepted as numbers either way).
8. Whether the owner's `opencode.json` sets per-model `options` that the SDK would spread into the body (any such key is rejected by the allowlist; the owner removes it or the plan is extended).
9. Whether attachments (images) are ever sent to the local model (rejected; out of scope).

Could not verify: whether the llama.cpp preset snapshot path is content-derived, i.e. whether a changed `reasoning-format` or `jinja` line in a preset changes the launch digest. The plan does not depend on it: `shapes` is evidence the operator records for a launch digest, as `chatTemplate` already is.

## Decisions (settled — do not revisit)

- Evidence, not code, decides shapes: a shape is accepted only when the installed qualification declares it. Not "accept tools everywhere as today", because a template that does not render tools makes the accounting and the output meaningless. Not "strip undeclared fields and forward", because the engine must receive the request the client sent (P/`body.go:9-12`) and silent stripping hides the mismatch.
- Only OpenCode shapes are added: `tools` (incl. `tool_choice`, assistant `tool_calls`, `tool` role), `parallel_tool_calls`, `reasoning_content` replay, `reasoning_effort`, structured output. Not vision, `user`, `verbosity`, `n`, logprobs: unused by this fork's client, and each would widen the engine surface the evidence must cover.
- No wire, RPC, advertisement or desktop change. Not "advertise shapes to routers", because hybrid dispatch is not wired yet and the destination revalidates every request anyway.
- `chatRequestKeys` stays as is; the E adapter is untouched.
- Commits use `git commit --no-signoff`; new Go tests listen on `127.0.0.1:0` (the `httptest` helpers already do); new files carry the two-line SPDX header; copy says "engine", never "backend"; no prompt, body or key is logged.

## Decisions made while writing this plan

1. **Artifact version 2 with a required `shapes` object.** Rejecting version 1 is deliberate: a version-1 document implicitly claimed tools for every entry. No real version-1 file exists.
2. **Five boolean shapes, all required**: `tools`, `parallelToolCalls`, `reasoningReplay`, `reasoningEffort`, `structuredOutput`. `parallelToolCalls: true` requires `tools: true`. Booleans, not per-field lists, because each shape is one template/engine fact.
3. **`tools` must be a nonempty array and `tool_choice` requires `tools`** (the SDK never sends either otherwise; the E adapter already enforces both).
4. **`reasoning_content` is accepted only on assistant messages, only as a string, only under `reasoningReplay`, and its bytes count as input** (the template renders it when replay is on). Under any other role or when the shape is off it is rejected, not dropped.
5. **`reasoning_effort` must be a nonempty string of at most 32 bytes**, accounted as its bytes, under `reasoningEffort`.
6. **A profile's declared capabilities must be covered by the qualification** or the model is not installed: `tools` → `shapes.tools`; `structured` → `shapes.structuredOutput`; `vision` or `embeddings` → never installed (no strict shape serves them). Otherwise tier routing would send tool requests to a node whose strict path refuses them.
7. **Fixtures are synthetic**: field names and nesting mirror the SDK output above; every string is placeholder text, tool names are generic, ids are `call_1`-style. No real prompt, path, host or user data. Captured bodies from MSI are never committed; the operator compares their shape against the fixture by eye.
8. `testQualification` (test helper) declares `tools: true` and the other shapes `false`, so the existing harness tests keep their meaning; the shared fixture artifact declares `tools` and `structuredOutput` true (its cases use both).

## Tasks

### Task 1 — The artifact declares its qualified shapes (schema version 2)

**Files:** P/`local_qualification.go`, P/`local_qualification_test.go`, P/`testdata/local-qualification.json`

**Depends on:** none

**Complexity:** mechanical

**Do:**

1. In P/`local_qualification.go` set `qualificationDocumentVersion = 2` and update the comment at line 38 to say the version was raised when `shapes` became required.
2. Add the type and field. Field order in the struct: after `Routes`, before `Accounting`.

```go
// qualificationShapes names the OpenCode request shapes the evidence covers.
// Every field is required so an artifact cannot claim a shape by omission.
type qualificationShapes struct {
	// Tools covers function tools, tool_choice, assistant tool_calls and tool
	// messages.
	Tools *bool `json:"tools"`
	// ParallelToolCalls covers the parallel_tool_calls field; needs Tools.
	ParallelToolCalls *bool `json:"parallelToolCalls"`
	// ReasoningReplay covers reasoning_content on assistant messages.
	ReasoningReplay *bool `json:"reasoningReplay"`
	// ReasoningEffort covers the reasoning_effort field.
	ReasoningEffort *bool `json:"reasoningEffort"`
	// StructuredOutput covers response_format json_object and json_schema.
	StructuredOutput *bool `json:"structuredOutput"`
}

func (s qualificationShapes) validate() error {
	for name, v := range map[string]*bool{"tools": s.Tools, "parallelToolCalls": s.ParallelToolCalls,
		"reasoningReplay": s.ReasoningReplay, "reasoningEffort": s.ReasoningEffort, "structuredOutput": s.StructuredOutput} {
		if v == nil {
			return fmt.Errorf("shapes.%s must be set", name)
		}
	}
	if *s.ParallelToolCalls && !*s.Tools {
		return errors.New("shapes.parallelToolCalls needs shapes.tools")
	}
	return nil
}

func (s qualificationShapes) tools() bool            { return s.Tools != nil && *s.Tools }
func (s qualificationShapes) parallelToolCalls() bool { return s.ParallelToolCalls != nil && *s.ParallelToolCalls }
func (s qualificationShapes) reasoningReplay() bool   { return s.ReasoningReplay != nil && *s.ReasoningReplay }
func (s qualificationShapes) reasoningEffort() bool   { return s.ReasoningEffort != nil && *s.ReasoningEffort }
func (s qualificationShapes) structuredOutput() bool  { return s.StructuredOutput != nil && *s.StructuredOutput }
```

   In `localQualification` add `Shapes qualificationShapes \`json:"shapes"\``. In `validate()` (after the routes loop, before the accounting check) add:

```go
	if err := q.Shapes.validate(); err != nil {
		return err
	}
```

3. Update the package comment (lines 6-18) to list "the request shapes it covers (shapes)" where it lists what a qualification names.
4. P/`testdata/local-qualification.json`: in `artifact`, set `"version": 2` and add to the single qualification, after `"routes"`:

```json
        "shapes": {
          "tools": true,
          "parallelToolCalls": false,
          "reasoningReplay": false,
          "reasoningEffort": false,
          "structuredOutput": true
        },
```

5. P/`local_qualification_test.go`:
   - `testQualification` (25-35): add `Shapes: qualificationShapes{Tools: ptr(true), ParallelToolCalls: ptr(false), ReasoningReplay: ptr(false), ReasoningEffort: ptr(false), StructuredOutput: ptr(false)}` and add the file-local helper `func ptr[T any](v T) *T { return &v }` (no such helper exists in the package's test files at `584e992`).
   - `TestQualificationDocumentRejects` (228-290): in `valid()` add `"shapes": map[string]any{"tools": true, "parallelToolCalls": false, "reasoningReplay": false, "reasoningEffort": false, "structuredOutput": false}`; in `doc()` write `"version": 2`; change the `"another version"` case to `{"version":1,"qualifications":[]}` still expecting `"version"`; add cases:
     - `{"no shapes", func() []byte { e := valid(); delete(e, "shapes"); return doc(e) }, "shapes."}`
     - `{"partial shapes", func() []byte { e := valid(); e["shapes"] = map[string]any{"tools": true}; return doc(e) }, "must be set"}`
     - `{"parallel without tools", func() []byte { e := valid(); e["shapes"] = map[string]any{"tools": false, "parallelToolCalls": true, "reasoningReplay": false, "reasoningEffort": false, "structuredOutput": false}; return doc(e) }, "needs shapes.tools"}`
     - `{"unknown shape", func() []byte { e := valid(); e["shapes"] = map[string]any{"tools": true, "parallelToolCalls": false, "reasoningReplay": false, "reasoningEffort": false, "structuredOutput": false, "vision": true}; return doc(e) }, "unknown field"}`
   - `TestLoadQualificationRegistry` (291-306) needs no change: its only literal document is the invalid `{"version":1,"qualifications":[{"id":"q"}]}`, which stays rejected.
6. Run `gofmt -l services/nvpair-proxy` and fix any listed file.

**Verify:**
- Mac: `cd services/nvpair-proxy && go vet ./... && go test -count=1 ./...` passes; `go test -race -count=1 -run 'Qualification|Strict' ./...` passes.
- `grep -n '"version": 2' services/nvpair-proxy/testdata/local-qualification.json` prints one line.
- `node scripts/spdx-headers.mjs` exits 0.
- PC3 (Windows gate, no `-race`): `cd services\nvpair-proxy && go test -count=1 ./...` passes.

### Task 2 — Strict validation gates and accounts the OpenCode shapes by evidence

**Files:** P/`local_qualification.go`, P/`local_qualification_test.go`, P/`testdata/local-qualification.json`

**Depends on:** Task 1

**Complexity:** judgment

**Do:**

1. P/`local_qualification.go`: add, next to `chatRequestKeys`:

```go
// shapeRequestKeys are top-level fields a qualified chat request may send only
// when the qualification declares the shape that covers them. They are kept
// apart from chatRequestKeys, which chat_adapter.go extends on its own.
var shapeRequestKeys = []string{"parallel_tool_calls", "reasoning_effort"}
```

2. In `validateRequest`, replace the allowlist loop (262-266) with:

```go
	for key := range request {
		if !slices.Contains(chatRequestKeys, key) && !slices.Contains(shapeRequestKeys, key) {
			return nil, strictBound{}, rejectBody("field %q is not supported", key)
		}
	}
```

   After `checkChatScalars(fields)` (279-281) add `if err := q.checkShapes(fields); err != nil { return nil, strictBound{}, err }`, and change `accountMessages(fields["messages"], a)` to `accountMessages(fields["messages"], a, q.Shapes)`, `accountTools(fields, a)` to `accountTools(fields, a, q.Shapes)`, `accountResponseFormat(fields["response_format"], a)` to `accountResponseFormat(fields["response_format"], a, q.Shapes)`. After the response-format accounting, add the `reasoning_effort` bytes:

```go
	if raw, ok := fields["reasoning_effort"]; ok {
		var effort string
		_ = json.Unmarshal(raw, &effort) // shape checked by checkShapes
		bound.input += len(effort)
	}
```

3. Add `checkShapes` (validates the shape-gated top-level fields on the effective body; runs before accounting so a rejected shape never reaches the engine):

```go
// checkShapes refuses top-level fields whose shape the qualification does not
// declare, and checks their values. Reasons name fields only.
func (q *localQualification) checkShapes(f map[string]json.RawMessage) error {
	_, hasTools := f["tools"]
	if raw, ok := f["parallel_tool_calls"]; ok {
		if !q.Shapes.parallelToolCalls() {
			return rejectBody("field %q is not qualified for this model", "parallel_tool_calls")
		}
		if !hasTools {
			return rejectBody("parallel_tool_calls requires tools")
		}
		var v bool
		if json.Unmarshal(raw, &v) != nil {
			return rejectBody("parallel_tool_calls must be a boolean")
		}
	}
	if raw, ok := f["reasoning_effort"]; ok {
		if !q.Shapes.reasoningEffort() {
			return rejectBody("field %q is not qualified for this model", "reasoning_effort")
		}
		var v string
		if json.Unmarshal(raw, &v) != nil || v == "" || len(v) > 32 {
			return rejectBody("reasoning_effort must be a short nonempty string")
		}
	}
	return nil
}
```

4. `accountTools(f, a, shapes qualificationShapes)`:
   - When `tools` is present: if `!shapes.tools()` return `rejectBody("tools are not qualified for this model")`; the decoded array must be nonempty, else `rejectBody("tools must be a nonempty array")`. Keep the per-tool checks and accounting.
   - When `tool_choice` is present: if `tools` is absent return `rejectBody("tool_choice requires tools")` (the shape check above already covers the shape, since tools is present). Keep the existing value checks and accounting.
5. `accountMessages(raw, a, shapes qualificationShapes)`:
   - Extend `chatMessage` with `ReasoningContent *string \`json:"reasoning_content"\`` (a non-string value fails `decodeExact` and keeps the existing "unsupported shape" rejection).
   - Directly after the role switch (406-410), before any other check on the message: if `msg.Role == "tool" && !shapes.tools()` return `rejectBody("messages[%d]: tool messages are not qualified for this model", i)`; if `len(msg.ToolCalls) > 0 && !shapes.tools()` return `rejectBody("messages[%d].tool_calls are not qualified for this model", i)`. The existing ownership checks (423-431) stay unchanged after them.
   - `reasoning_content`: if `msg.ReasoningContent != nil`: when `msg.Role != "assistant"` return `rejectBody("messages[%d].reasoning_content belongs to an assistant message", i)`; when `!shapes.reasoningReplay()` return `rejectBody("messages[%d].reasoning_content is not qualified for this model", i)`; otherwise `n += len(*msg.ReasoningContent)`.
   - Null content: keep the rule at 416-418 unchanged (null content needs tool_calls; a replayed reasoning_content alone does not make null content valid, because the SDK sends `null` only with tool calls).
6. `accountResponseFormat(raw, a, shapes qualificationShapes)`: for `json_object` and `json_schema`, when `!shapes.structuredOutput()` return `rejectBody("response_format %s is not qualified for this model", format.Type)`; `text` stays accepted. Keep the accounting.
7. Update the package comment and the comment on `chatRequestKeys` to mention `shapeRequestKeys`.
8. P/`testdata/local-qualification.json` cases (the artifact declares `tools` and `structuredOutput`; add after the existing cases, keeping the existing ones and their numbers unchanged):

```json
    {
      "name": "tool_choice without tools",
      "body": {"model": "fixture-model", "messages": [{"role": "user", "content": "hi"}], "tool_choice": "auto", "max_tokens": 8},
      "reject": "tool_choice requires tools"
    },
    {
      "name": "empty tools",
      "body": {"model": "fixture-model", "messages": [{"role": "user", "content": "hi"}], "tools": [], "max_tokens": 8},
      "reject": "nonempty array"
    },
    {
      "name": "parallel tool calls not qualified",
      "body": {"model": "fixture-model", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}], "parallel_tool_calls": false, "max_tokens": 8},
      "reject": "\"parallel_tool_calls\" is not qualified"
    },
    {
      "name": "reasoning replay not qualified",
      "body": {"model": "fixture-model", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok", "reasoning_content": "thinking"}], "max_tokens": 8},
      "reject": "reasoning_content is not qualified"
    },
    {
      "name": "reasoning effort not qualified",
      "body": {"model": "fixture-model", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low", "max_tokens": 8},
      "reject": "\"reasoning_effort\" is not qualified"
    },
    {
      "name": "reasoning content on a user message",
      "body": {"model": "fixture-model", "messages": [{"role": "user", "content": "hi", "reasoning_content": "x"}], "max_tokens": 8},
      "reject": "belongs to an assistant message"
    },
    {
      "name": "user field",
      "body": {"model": "fixture-model", "messages": [{"role": "user", "content": "hi"}], "user": "u", "max_tokens": 8},
      "reject": "field \"user\" is not supported"
    }
```

9. P/`local_qualification_test.go`: add `TestQualificationShapesAcceptDeclaredFields`: build `q := testQualification("shapes", "ollama", "m")` with all five shapes true and `ToolOverheadTokens: 16`; profile as in `installQualified`; call `q.validateRequest` on the body

```json
{"model":"m","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"ok","reasoning_content":"thinking"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"parallel_tool_calls":false,"reasoning_effort":"low","max_tokens":16}
```

   and assert no error, `bound.output == 16`, and `bound.input == 8 + (4+4+2) + (4+9+2+8) + (16 + len(compact tool JSON)) + 3`, where the compact tool JSON is `{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}` (74 bytes), so `bound.input == 8 + 10 + 23 + 90 + 3 == 134`. Then flip each shape to false in turn (five sub-tests) and assert `validateRequest` returns a `*strictQualificationError` whose message contains `not qualified`.
10. `gofmt -l services/nvpair-proxy` must print nothing.

**Verify:**
- Mac: `cd services/nvpair-proxy && go vet ./... && go test -count=1 ./...` passes; `go test -race -count=1 -run 'Qualification|Strict|Chat' ./...` passes (the `Chat` tests prove `chat_adapter.go` is unaffected).
- `go test -count=1 -run 'TestQualificationFixtureCases' -v ./... | grep -c '^    --- PASS'` prints the number of fixture cases (17 existing + 7 new = 24).
- PC3: `go test -count=1 ./...` in P passes.

### Task 3 — A profile's routing capabilities must be covered by its qualification

**Files:** P/`readiness_contract.go`, P/`local_qualification_test.go`

**Depends on:** Task 2

**Complexity:** mechanical

**Do:**

1. P/`readiness_contract.go`, in `qualifiedModelLocked` after the `requestOptions`/bounds check (the `sameJSONObject(...) || pr.ContextTokens > ... || pr.MaxOutputTokens > ...` condition), add:

```go
	if !shapesCoverCapabilities(q.Shapes, pr.Capabilities) {
		return nodeStrictModel{}, nil, false
	}
```

   and the helper:

```go
// shapesCoverCapabilities reports whether every routing capability the
// profile declares is a shape the qualification was measured for. Vision and
// embeddings have no strict shape, so a profile declaring them is never
// installed; tier routing would otherwise send requests here that the strict
// path refuses.
func shapesCoverCapabilities(s qualificationShapes, caps []nodepolicy.Capability) bool {
	for _, c := range caps {
		switch c {
		case nodepolicy.CapTools:
			if !s.tools() {
				return false
			}
		case nodepolicy.CapStructured:
			if !s.structuredOutput() {
				return false
			}
		default:
			return false
		}
	}
	return true
}
```

2. Update the comment on `refreshStrictCapabilityLocked` (72-83) to add "the profile's capabilities are shapes the qualification covers" to the list of conditions.
3. P/`local_qualification_test.go`, `TestStrictCapabilityDerivation` (310-439). Its table holds `{name string; edit func(*admissionController)}` cases, each run on `newStrictHarness(t)` and expected to leave the model not installed. Add two cases:
   - `{"capability structured not covered", func(a *admissionController) { p := a.currentPolicy(); p.Profiles[0].Capabilities = []nodepolicy.Capability{nodepolicy.CapStructured}; a.setPolicy(p) }}` (the harness qualification declares `structuredOutput: false`).
   - `{"capability vision never covered", func(a *admissionController) { p := a.currentPolicy(); p.Profiles[0].Capabilities = []nodepolicy.Capability{nodepolicy.CapVision}; a.setPolicy(p) }}`.
   Add one positive sub-test after the existing `"matching inputs install the identity"` run: `t.Run("covered capability stays installed", ...)` that sets `Capabilities = []nodepolicy.Capability{nodepolicy.CapTools}` the same way and asserts `installed(h.a)` returns `ok == true`.

**Verify:**
- Mac: `cd services/nvpair-proxy && go vet ./... && go test -count=1 ./...` passes; `go test -race -count=1 -run 'StrictCapability|Qualification' ./...` passes.
- PC3: `go test -count=1 ./...` in P passes.

### Task 4 — Synthetic OpenCode fixtures and the before-side-effects test

**Files:** P/`testdata/opencode-shapes.json` (new), P/`opencode_shapes_test.go` (new)

**Depends on:** Task 2

**Complexity:** judgment

**Do:**

1. Create P/`testdata/opencode-shapes.json` with this structure. Every value is synthetic; do not paste any captured body. The artifact holds two qualifications on `"engine": "llamacpp"`, `"model": "fixture/model:Q4"`: `opencode-full` (all shapes true) and `opencode-plain` (all false). Both: `modelDigests` `["3333…"]` (64 × `3`), `launchDigests` `["4444…"]`, no `requestOptions`, `tokenizer` `"synthetic-byte-level-bpe"`, `chatTemplate` `"synthetic-jinja-tools"` / `"synthetic-jinja-plain"`, `contextTokens` 8192, `maxOutputTokens` 4096, `routes` `["/v1/chat/completions"]`, accounting `utf8-bytes` with overheads 8/4/16/16, `evidence` a sentence stating it is synthetic and qualifies nothing real.

   Each case has `name`, `qualification` (id), `body`, and either `inputTokens` + `outputTokens` or `reject`. Cases (bodies mirror the SDK shapes from the Context table):

   - `opencode first turn with tools` (full; accept): `model`, `max_tokens: 4096`, `temperature: 0`, `messages` = system string + user string, `tools` = one function `read_file` with `description` and `parameters` `{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}`, `stream: true`, `stream_options: {"include_usage": true}`.
   - `opencode tool round trip` (full; accept): user string; assistant `content: null` with one `tool_calls` entry `{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}}`; `{"role":"tool","tool_call_id":"call_1","content":"file contents"}`; same single tool; `stream: true`; `max_tokens: 4096`.
   - `opencode reasoning replay` (full; accept): user `"hi"`, assistant `{"content":"ok","reasoning_content":"thinking"}`, `max_tokens: 16`. Expected `inputTokens` 41 (8 + (4+4+2) + (4+9+2+8)), `outputTokens` 16.
   - `opencode reasoning effort` (full; accept): user `"hi"`, `reasoning_effort: "low"`, `max_tokens: 16`. Expected `inputTokens` 21 (8 + 10 + 3).
   - `opencode parallel tool calls` (full; accept): as the first turn plus `parallel_tool_calls: false`.
   - `plain refuses tools` (plain; reject `"tools are not qualified"`): the first-turn body.
   - `plain refuses tool history` (plain; reject `"not qualified"`): the round-trip body without `tools`.
   - `plain refuses reasoning replay` (plain; reject `"reasoning_content is not qualified"`).
   - `plain refuses reasoning effort` (plain; reject `"\"reasoning_effort\" is not qualified"`).
   - `plain refuses structured output` (plain; reject `"json_schema is not qualified"`): `response_format` `{"type":"json_schema","json_schema":{"name":"response","strict":true,"schema":{"type":"object"}}}`.
   - `plain accepts text only` (plain; accept): system + user strings, `stream: true`, `stream_options`, `max_tokens: 4096`.
   - `image attachment refused` (full; reject `"only text parts"`): user content array with a `text` part and `{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}`.
   - `provider option passthrough refused` (full; reject `"field \"chat_template_kwargs\" is not supported"`): body with `chat_template_kwargs: {"enable_thinking": false}`.
   - `verbosity refused` (full; reject `"field \"verbosity\" is not supported"`).

   For the accept cases whose bytes are not given above, compute `inputTokens` by the documented formula (request overhead + per message: overhead + role + content + name + tool_call_id + reasoning_content bytes + compact JSON of each tool call; per tool: overhead + compact JSON; plus compact `tool_choice`; plus schema overhead + name + description + compact schema; plus `reasoning_effort` bytes) and write the arithmetic in a `"note"` string on the case (the decoder ignores unknown case fields). The test fails if the number is wrong, so run it once, compare with the hand sum, and fix the hand sum or the body, never copy the test's number without the arithmetic.

2. Create P/`opencode_shapes_test.go` (SPDX header, `package main`) with:
   - `TestOpenCodeShapeFixtures`: load the file, `parseQualificationDocument(doc.Artifact)`, for each case look up its qualification, build the profile as `qualificationFixture` does (same bounds, same `RequestOptions`, `Qualification` ref), call `validateRequest(profile, routeChatCompletions, body)` and assert exactly like `TestQualificationFixtureCases` (130-160). Define a local `openCodeCase` struct (`Name`, `Qualification`, `Body json.RawMessage`, `InputTokens`, `OutputTokens`, `Reject`); do not change `qualificationCase`. Assert `len(doc.Cases) == openCodeFixtureCases` with `const openCodeFixtureCases = 14`.
   - `TestStrictHTTPRefusesUnqualifiedShapesBeforeSideEffects`: for `path` in `ingress`, `self`: build `p := strictHTTPFixture(t)`-style proxy with `configureStrict`, a `newRecordingEngine`, `setBackend(..., false)` and a cold, stopped engine exactly as `TestStrictHTTPRejectsUnqualifiedRequestsBeforeSideEffects` (460-519) does; the installed `testQualification` has `tools: true` and the other shapes false. Send (a) a body with `reasoning_content` on an assistant message, (b) a body with `response_format` `{"type":"json_object"}`, (c) a body with `reasoning_effort`. Assert status 400, code `strict-qualification`, `e.hits() == 0`, no active tickets, no markers, and no `admission/wake` or `admission/unload` notification, as that test does. Then send the fixture's `opencode tool round trip` body (tools shape on) with the engine healthy (`setBackend(..., true)`) and assert 200 and `e.hits() == 1` and that `e.lastBody()` contains `"tool_call_id":"call_1"`.

**Verify:**
- Mac: `cd services/nvpair-proxy && go vet ./... && go test -count=1 -run 'OpenCode|StrictHTTP' -v ./...` passes and lists every fixture case as a passing sub-test; `go test -race -count=1 ./...` passes.
- `go test -count=1 -run 'TestOpenCodeShapeFixtures' -v ./... | grep -c '^    --- PASS'` prints 14.
- `node scripts/spdx-headers.mjs` exits 0.
- PC3: `go test -count=1 ./...` in P passes.

### Task 5 — Docs and spec describe shapes

**Files:** `docs/local-qualification.mdx`, P/`spec.md`

**Depends on:** Tasks 2, 3, 4

**Complexity:** mechanical

**Do:**

1. `docs/local-qualification.mdx`:
   - Example artifact (28-54): `"version": 2`; add after `"routes"`:
     ```json
         "shapes": {
           "tools": true,
           "parallelToolCalls": false,
           "reasoningReplay": false,
           "reasoningEffort": false,
           "structuredOutput": false
         },
     ```
   - Field table (56-68): add a row `| \`shapes\` | The OpenCode request shapes the evidence covers: \`tools\` (function tools, \`tool_choice\`, assistant \`tool_calls\`, \`tool\` messages), \`parallelToolCalls\` (needs \`tools\`), \`reasoningReplay\` (\`reasoning_content\` on assistant messages), \`reasoningEffort\`, \`structuredOutput\` (\`json_object\` and \`json_schema\`). All five are required. A shape is accepted only when declared. |`
   - Accounting paragraph (73-86): after "the bytes of its role, name, tool call id and text content" add "and replayed `reasoning_content`", and after "plus the compact `tool_choice`" add ", plus the bytes of `reasoning_effort`".
   - "Installing strict capacity" list (114-120): add a bullet "- the profile's declared capabilities are shapes the qualification covers (`tools` → `shapes.tools`, `structured` → `shapes.structuredOutput`; `vision` and `embeddings` are never covered)."
   - "Validating a strict request" (127-145): in step 3 add ", a shape the qualification does not declare (tools, `parallel_tool_calls`, `reasoning_content`, `reasoning_effort`, a structured `response_format`)"; add a sentence after the list: "`tools` must be nonempty and `tool_choice` and `parallel_tool_calls` require `tools`; `reasoning_content` belongs to assistant messages only."
   - Add a short section `## OpenCode request shapes` before `## Not covered here` with the fact table from this plan's Context (the SDK rows only, without the assumption list), one sentence that the shapes were chosen from it, and one sentence that `services/nvpair-proxy/testdata/opencode-shapes.json` holds synthetic examples of each and that captured bodies are never committed or logged.
   - Under `## Not covered here` add "- Model content digests for llama.cpp and Strata: their `list_models` declares no `digest`, so no model of theirs is installed until engine-manager reports one."
   - Line 15-19 stays as is (hybrid still disabled).
2. P/`spec.md` section "Local qualification and identity (B)" (179-222): in the paragraph starting "Only the `/v1/chat/completions` shape is supported" replace "allowlisted top-level fields, text content, function tools, `json_object`/`json_schema` formats" with "allowlisted top-level fields plus `parallel_tool_calls` and `reasoning_effort` under their shapes, text content, function tools and `reasoning_content` replay under their shapes, `json_object`/`json_schema` formats under `structuredOutput`"; add a sentence: "The artifact (version 2) declares `shapes {tools, parallelToolCalls, reasoningReplay, reasoningEffort, structuredOutput}`; a shape the installed qualification does not declare is refused with 400 `strict-qualification` naming the field; a profile whose capabilities are not covered by the shapes is not installed." In the accounting sentence add "replayed `reasoning_content` and `reasoning_effort` bytes".
3. Do not edit `desktop/docs/services-api.md`, `CHANGELOG.md` or `services/versions.json`.

**Verify:**
- Mac: `node scripts/spdx-headers.mjs` exits 0; `git diff --stat` shows only the two files; `grep -n '"version": 2' docs/local-qualification.mdx` prints one line; `grep -c 'shapes' services/nvpair-proxy/spec.md` ≥ 2.
- Mac: `npm --prefix desktop run service-contracts:check` passes (no RPC change was made; this proves it).

## Verification (whole-plan)

- Mac: `cd services/nvpair-proxy && go vet ./... && go test -race -count=1 ./...` → `ok`. `cd services/shared && go test -count=1 ./...` → `ok` (untouched, sanity). `node scripts/spdx-headers.mjs` → exit 0. `npm --prefix desktop run service-contracts:check` → passes unchanged.
- PC3 (Windows gate, serialized, no `-race`): `cd services\nvpair-proxy && go test -count=1 ./...` → `ok`. No `services/tests` case changed; the existing Windows `services/tests` gate runs unchanged.
- MSI (owner, outside the executor's tasks; confirms the assumptions list): run OpenCode once against the llama.cpp facade with a loopback request capture that is not PAIR (PAIR never logs bodies), and compare the captured field names and message shapes, not their content, with `testdata/opencode-shapes.json`. Record which of assumptions 1-9 hold. Any top-level field or message field not in the fixture means the owner answers the corresponding owner question before the shape is added; the captured bodies are discarded, not committed.
- Reading `plans/b-opencode-shapes.md` is not part of the product build; the `plans/` branch is never merged.

## Do not

- Do not add `parallel_tool_calls` or `reasoning_effort` to `chatRequestKeys`; P/`chat_adapter.go:446` extends that list itself and serializes by iterating it.
- Do not strip or rewrite undeclared fields; reject them.
- Do not accept `tools: []`, `tool_choice` without `tools`, or `reasoning_content` outside assistant messages, even when the shape is on.
- Do not change any `nodepolicy` wire type, `AdvertisedRole`, JSON-RPC method, broker, engine-manager, desktop file, `services/tests`, `desktop/docs/services-api.md`, `CHANGELOG.md` or `services/versions.json`.
- Do not log, print in a test failure, or write to a fixture any real request body, prompt, path, host, user or key. Test failure messages may print synthetic fixture bodies only.
- Do not use type casts in TS (no TS is touched), legacy fallbacks, or the word "backend" in new copy.
- Do not run `-race` on PC3.

## If stuck

- A `TestQualificationFixtureCases` or `TestOpenCodeShapeFixtures` byte count disagrees with the hand sum: recompute with `json.Compact` semantics (no insignificant whitespace; strings keep their escapes as written) and remember `len(role)` is counted per message and `messageOverheadTokens` is 4 in both fixtures. Fix the fixture, not the accounting, unless the accounting diverges from `docs/local-qualification.mdx:73-86`; then stop and report.
- `TestChat*` tests in `chat_adapter_test.go` fail after Task 2: `chatRequestKeys` was modified; revert that and use `shapeRequestKeys`.
- `go test` in P fails on a test unrelated to qualification on the clean tree: re-run once; if it fails again, report it as pre-existing with the output and continue only with tasks whose Verify does not depend on it.
- A `ptr` helper already exists in the package tests under another name: reuse it; do not define a second one.
- The `-race` run on Mac exceeds 10 minutes: run `go test -race -run 'Qualification|Strict|OpenCode' ./...` and report that the full race run was not completed.
- After two failed attempts at any task, stop and report the diff and the failing output instead of widening the change.

## Owner questions

1. **Is reasoning replay enabled in your OpenCode configuration for the local model, and is the model run with a reasoning format (`reasoning-format`/`jinja` in its llama.cpp preset)?** This decides whether the MSI evidence declares `reasoningReplay: true` (assistant `reasoning_content` accepted and counted as input) or `false` (refused). The code supports both; the evidence file picks one.
2. **Does OpenCode ever send `reasoning_effort` to the local model (a `reasoningEffort`/variant setting in `opencode.json`)?** If never, the `reasoningEffort` shape and `shapeRequestKeys` entry can be dropped to keep the accepted surface smaller; keeping it is harmless only while every evidence file sets it `false`.
3. **A profile that declares capability `tools` (or `structured`) while its qualification lacks the shape is not installed at all (plan default), rather than installed for non-tool requests only.** Confirm, or choose the narrower behaviour (install, refuse only the uncovered shape per request), which routes tool requests to a node that will refuse them.
4. **llama.cpp and Strata report no model content digests today, so no model of theirs can be installed strictly (`E/manifests/llamacpp.json:98-101`, `E/model_identity.go:161-170`).** This plan does not add them. Confirm that digest reporting is tracked in another package (NE Task 3 or a follow-up), or ask for a separate plan.
