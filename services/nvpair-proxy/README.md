<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Engine Proxy

A discovery-aware HTTP reverse proxy that fronts every enabled inference engine.
It runs no mDNS browse of its own: routing targets come from the broker's
discovery relay (each facade sends `discovery:subscribe` for its engine's
service key and replaces its routing overlay from each pushed `discovery:nodes`
snapshot) plus user-added manual nodes. It forwards HTTP requests to the
selected node, aggregates the model-list routes across candidate nodes, and
exposes a bidirectional JSON-RPC 2.0 control channel over stdio (or an IPC
socket).

**One process, one facade per engine.** The process starts with no engine and no
listener; the broker asks for each engine's facade with `facade/enable`, which
carries that engine's port. A flag cannot express this,
because the broker plans a different port for each engine.

They share a process on purpose. Between scheduler snapshots a facade takes
short-lived reservations for work it has dispatched, and those live in the
process — a process per engine split that picture, so simultaneous bursts on
different engines could pick the same node believing it idle.

The cost is shared fate for the **process**: a crash takes every facade down and
the supervisor restarts them together. Smaller failures are contained to one
engine — a lost bind race, a port that cannot be re-listened, or a panic while
handling a request. Each of those used to end the process, which was fine when
the process was one engine.

`spec.md` is normative; where it and this file disagree, it wins. Everything
below is shared behavior unless a section says otherwise.

Inference requests name a visible profile key or a declared role. A role wins
over an equal profile key. Other names return HTTP 404 with
`no profile or role named <name>`, even when native inventory or model sources
advertise them. `/v1/models` lists visible profile keys and declared roles with
their `:api` and `:local` aliases, each once. Files without profiles remain
visible in the desktop Models tab but are not routable. Strict, public and
hybrid qualification requirements still apply.

Request model and tier-alias identifiers are limited to 256 decoded UTF-8
bytes, measured before trimming whitespace or engine normalization. This applies
to body model/name identifiers, model query parameters and model-detail paths.
An identifier above the limit returns HTTP 400 before routing resolution,
admission, preparation or workload placement. Escapes count as their decoded
UTF-8 bytes. The independent request-body limit remains 64 MiB by default;
body or required inference-storage overflow returns 413, and buffer contention
returns 503. Bounded normalized identifiers and ordinary workload/cancellation
serialization are outside the inference byte root; live canonical registry
entries and `routing/active` copies/encoding are charged to it. Queued identifiers and finished-model
timestamps own small copies rather than retaining request storage.

## Origin-local routing history

One process-lifetime copier reads bounded, typed request facts captured after
canonical registration and before body preparation. It runs independently of
inference and publishes frozen windows to the local broker. Requests refused
before registration are outside history. No prompt, response body, raw error,
credential or provider configuration is retained in this projection.

The private notifications `routing/history-window`, `routing/request` (one full
record), and `routing/history-window-end` form one generation. The broker
stages the complete window, validates its already acknowledged process/origin/run
binding, commits it, then replies with `routing/history-accepted`. Unacknowledged
windows replay on the single five-second reconciliation timer with the same
generation. These methods never reach clients, peers, managers or the scheduler;
clients use the broker's `routing:subscribe`, `routing:unsubscribe`,
`routing:get-initial` and `routing:changed` surface through `nvpair-service`.

History keeps the newest 256 requests and the newest 32 attempts per request,
preserving original indexes and `droppedAttempts`. A model-less local attempt
retains its actual node/engine with `destination.model=null`; its presentation
label is `model not specified`. API destinations require a model.
Money is an exact decimal nano-USD string or null, never inferred from the
trimmed attempt list. See [the history contract](spec.md#1252-origin-local-routing-history)
for ordering, waits, workload linkage, amount meanings and capacity ownership.

History is memory-only. Proxy death can lose unsent facts and incomplete
windows; the broker preserves previously accepted rows and retires that run's
open presentation records. Broker restart clears history. Diagnostics neither
replay inference nor invent delivery or charges. Desktop/TUI consumer banks are
wired; desktop Settings/card presentation and the TUI Routing tab remain
separate original Tasks 4/5. This does not enable public providers or SC-b
same-model continuation.

## Local execution measurements

The destination records each actual local inference dispatch, including a failed
engine dial, through the existing unaddressed `performance:sample` notification.
Self routing, ordinary peer ingress and strict local dispatch use the same
producer. Peer forwarding at the origin, control requests and admission refusals
create no local execution sample. Strict completion retains its existing
uncertain-capacity reconciliation rules.

Streaming measurements consume typed numeric metadata from the existing SC
response reader. Its completion verdict remains authoritative. The llama.cpp
and Strata Chat adapters preserve engine-reported decode rates and intervals
separately from visible output tokens. Nonstream bodies pass through without
numeric observation. No first token time is inferred from headers, body bytes
or final timings.
The aggregate `Entry.outputTokens` is the total of known engine-reported output
counts from successful, SC-complete local executions in that entry. It is
qualified independently of decode rates: usage-only completion can contribute
a count while rates remain unknown. The field is absent when no qualifying
count is known; an explicit zero is a measured zero. Native decode intervals
and rates remain separate from this visible-output total.
Unobserved responses count as completed only after the accepted response body
reaches upstream EOF and delivery succeeds; protocol completion stays unknown.

Samples require the joined engine run, settings, launch and model-content
identity plus the measuring host. One proxy-local fingerprint includes that
identity, effective request options, API and timing schema. Missing identity,
unknown overrides or identity changes during execution suppress the sample.
Descriptor family selects the reviewed extractor schema; native daily-engine
acceptance still requires trustworthy installed identities and engine evidence.

A process-wide FIFO holds at most 128 samples. Sequence numbers are assigned at
enqueue in finalization order across facades. One sender writes notifications
outside the queue lock; saturation drops telemetry with bounded counters rather
than waiting on the codec. Normal shutdown cancellation stops the sender
between writes or while waiting for the codec write gate. The parent session
interrupts an active write when its caller is cancelled or its 60-second
notification budget expires, then closes both transport directions. The sender
is not joined during shutdown.

## Build

Strict destination admission is opt-in through one `X-PAIR-Strict-Admission`
JSON header on paired ingress or local dispatch. It shares the process-wide
controller with concrete admission. Invalid/duplicate envelopes return 400;
unavailable/off and cold-residency refusals return 503, occupied capacity 429.
Queue and cold allowances consume the same monotonic pre-output cap. Zero cold
allowance causes no wake/unload and excludes unloaded models and pending unloads.
Warm-only tickets recheck residency and pending unloads before dispatch; loss
rejects with 503 and releases the undispatched slot. Once cold work is needed,
its deadline bounds every later wait, including capacity waits. Exhaustion rejects
before allocating a ticket.

After each acknowledged `node/set-policy` or `node/set-model-identity` the proxy
reports its strict readiness to the broker (`admission/strict-readiness`), naming
the acknowledged revision and run and carrying the installed capability. Each
peer's advertised capability arrives on its discovery node and is kept as internal
routing metadata. Hybrid role eligibility (`hybridLocalCandidates`) offers only a
node's own declared role with finite qualification on a node advertising a valid
capability for its engine and normalized model; peers without one keep concrete
routing. The common request owner's hybrid policy branch now calls this filter,
preserving tier candidate order, and shares attempts, strict queue/cold allowances
and one pre-output deadline across destinations. Concrete HTTP inference still
passes `concreteDispatchPolicy`. The loopback routing listener enters the same
HTTP request lifecycle with `hybridDispatchPolicy`, buffers under the effective
individual body limit and accepts one exact tier alias. Qualified local and peer
destinations receive the selected concrete model. Public selection remains disabled by
`PublicDispatchAvailable()`. See [Hybrid routing](../../docs/hybrid-routing.mdx)
for selection bounds and listener lifetime.

The capability is derived, never set: a profile's model is installed only when the
profile names a local qualification that `local-qualification.json` in the shared
`appdir` directory holds for the same engine and model, the broker's engine identity
(`node/set-model-identity`) carries a launch digest and model content digest the
qualification covers, the profile's options and effective mode match it, its
bounds are within it, and effective concurrency is finite. Without that evidence nothing is installed and
strict requests fail closed. A strict request is validated against the
qualification (finite output bound, accounted input plus output within the served
context, supported request shape, no request-level context override) before any
wake, unload, slot or inference, and the engine receives it with the profile's
options applied under the profile's precedence. See `docs/local-qualification.mdx`.
Concrete requests retain their existing admission and retry behavior.

Automatic residency uses the same admission owner and qualified finite bounds.
`servedModels` permits on-demand work; `residentModels` declares the desired
loaded subset, without proving that it is loaded. The proxy reserves the whole
desired set against resident/in-flight memory and model limits, without eviction.
For an NE envelope, profile context equals native per-slot context, qualification
covers native context/output, and effective concurrency equals native slots.
The manager requires explicit model-only POST load and safe loaded-model GET
bindings, native retention `"-1"`, and router capacity for the revision-bound
engine resident set. No request keep-alive field changes native retention.
Local pause/Off vetoes still apply; reconciliation cannot start an Off engine.
The broker coordinates internal prepare/dispatch/finish/observe calls and publishes
model-configuration status; desktop clients do not call these admission methods.
A definite terminal manager load failure is published as `unknown`/`load-failed`,
including retained finish/recovery readbacks. The reason does not establish
unloaded inventory or free capacity, and leaves hold cleanup, expiry and retry
deadlines unchanged. A transient proxy `engine-off` refusal stays `pending`
without a reason while the current manager source reports the engine running.
PG keeps route-policy ownership. Unknown load/allocation evidence and restart
are not proof of satisfaction; fresh manager-owned observed allocation is required.
An interrupted load without a terminal manager reply retains its resident hold
and reports `unknown`/`operation-timeout` until the next policy edit or a proxy
restart. After proxy or manager replacement, fresh observed allocation from the
current manager can report it `satisfied` while the hold remains. Engines
adopted from a previous manager run cannot provide confirmed resident
allocation and stay `unknown`.
See `docs/local-qualification.mdx` for adapter requirements and pending real-host
acceptance, including retention, capacity, loaded limits and restart recovery.

Before strict inference bytes, the proxy durably writes a slot to
`proxy-strict-capacity.json` in the shared `appdir` directory. Cancellation,
transport failure and HTTP EOF retain uncertain capacity across restart; HTTP
completion alone is not qualified termination. Authoritative reconciliation is
required to clear it. No timeout releases quarantine. Storage failures prevent
dispatch or retain the slot and log the failure. Corrupt marker documents disable
strict admission without restoring partial state: version and slots are required
and nonnull, with positive unique IDs, built-in engines and valid normalized models.
All `X-PAIR-*` request headers
and `X-NVPAIR-Engine-Identity-Probe` are stripped before local engine forwarding.

```bash
go build -o nvpair-proxy .
```

## Usage

```
nvpair-proxy [flags]
```

Then, over the control channel, for each engine to front:

```json
{"jsonrpc":"2.0","id":1,"method":"facade/enable",
 "params":{"engine":"llamacpp","port":8080,"ignorePersistedPort":true}}
```

The response is `{"engine":"llamacpp","port":8080}`, reporting the port actually
bound — enable is a request, and the child's persisted-port restore can override
the port asked for. Enabling an engine that is already up is an idempotent
success, so a redelivered enable never tears down a working listener.

### Flags

Only process-scoped settings are flags. Anything per-engine is a `facade/enable`
parameter, because one flag cannot carry multiple engines' plans.

| Flag | Default | Description |
|------|---------|-------------|
| `--ipc` | *(empty — use stdio)* | Path to a Unix domain socket or Windows named pipe for IPC |
| `--cluster-dir` | *(empty)* | Cluster trust directory (`node.crt`/`node.key` plus trusted pins). Enables the LAN mTLS inference ingress while this node is a cluster member; empty means no ingress and no peer candidates. |
| `--max-request-bytes` | `67108864` (64 MiB) | Largest request body the proxy buffers so a failed attempt can be replayed on another node. A larger body is answered with `413` and never dispatched. |
| `--log-level` | *(`$NVPAIR_LOG_LEVEL`, else `info`)* | Initial log level: `debug`, `info`, `warn`, or `error`. Changeable at runtime with `log/set-level`. |
| `--version` | | Print version and exit |

### `facade/enable` parameters

| Field | Default | Description |
|-------|---------|-------------|
| `engine` | *(required)* | Which engine to front: the built-in `llamacpp`, or a descriptor engine ID together with `descriptor`. An unknown name without a descriptor is rejected with the accepted values. |
| `port` | per engine, see below | HTTP listen port for request forwarding. Must be 1–65535, or omitted for the engine's standalone default. `0` means "the default" rather than "pick an ephemeral port", and any other out-of-range value is rejected, because the facade announces the requested port in its `ready` notification and the broker would be told `0`. |
| `ignorePersistedPort` | `false` | Use `port` even when a saved port exists (used by broker-managed startup) |
| `descriptor` | *(none)* | The engine's `nvpair-shared/engines.Descriptor`, required for a descriptor engine and refused for a built-in one. The facade is built from its protocols (see below); a descriptor that `engines.ProxyCanFront` rejects, or one naming another engine, is refused. A descriptor engine also needs an explicit `port`. |

### What differs per engine

Everything engine-specific is one entry in `engines.go`, plus the shared
identity in `nvpair-shared/engines`.

### Descriptor engines

An engine that is not built in, such as a llama.cpp fork registered as
`llamacpp-custom` or a Strata user manifest, needs no entry here. Its facade
profile is built from its descriptor: each declared protocol selects a coded
route set (`protocolRoutes` in `engines.go`), with `llamacpp-router`'s
`/v1/models` rewrite applied before a plain `/v1/models`. `openai-v1` or
`openai-chat` must be among them, because each supplies an
inference route and a model list; `anthropic-messages` and `llamacpp-router`
only add routes. Model identifiers match exactly. The engine has no standalone
or reserved port.
Its discovery service key is `e-<id>`, its facade-scoped messages are addressed
`e-<id>:<method>`, and the port its facade first binds is persisted in
`<id>-proxy-port.json` so it stays there across restarts. Peers that advertise
the same engine ID are routed over the pinned direct and reverse paths like a
built-in engine. Node policy, drain, residency, intent and strict capacity
accept any `engines.ValidRoutingID`; an engine with no facade here simply has no
local candidates.

| | `"engine":"llamacpp"` |
|---|---|
| Facade id — error-ID prefix and broker relay namespace | `llamacpp-proxy` |
| Discovery service key | `lc` |

The **log component, supervisor label, and TUI health crash key are not in that
table**: they name the process (`nvpair-proxy`), not a facade, because one
process hosts every engine and there is no single engine to attribute them to.
Facade-scoped log records carry an `engine` field instead. The supervisor label
and the health crash key are matched against each other, so they move together
— see `nvpair-shared/engines` and `spec.md` §9.

| | `"engine":"llamacpp"` |
|---|---|
| Engine's own client-facing port | 8080 |
| Where PAIR relocates the engine | 8081 |
| Standalone port, used when `port` is omitted | 8080 |
| Persisted-port file (declared, not derived) | `llamacpp-proxy-port.json` |
| Model-list routes | `GET /models`, `GET /v1/models` (OpenAI; browser permission probes use upstream `/models`) |
| Inference routes | `/completion`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` |
| Model naming | identifiers compared byte for byte |

The route table is a **classifier, not an allowlist**. An unlisted path is
forwarded verbatim, including engine controls and `OPTIONS` preflights.

The broker enables table-engine facades by default. Local llama.cpp-compatible
clients use `8080`, while the managed `llama-server` stays on `8081`.
`--proxy-engines` can restrict a standalone broker or TUI to a subset.

### HTTP Reverse Proxy

A facade listens on its enabled port and routes inference through declared
roles and profile keys, including llama.cpp's native `POST /completion`.
On the llama.cpp facade, `GET /models` and `GET /v1/models` return the same
policy-derived role and profile-key catalog, even when a node is selected.
Point your client at the proxy and it handles routing.

**Cluster ingress.** The listener carries two personalities, demultiplexed by
each connection's first byte. Plaintext HTTP is accepted only from loopback; a
LAN caller is refused. When `--cluster-dir` shows this node is a cluster member,
the same listener also terminates cluster mTLS: a peer whose client certificate
matches one of this node's pins is forwarded straight to the local engine
reported by `node/set-local-backend`, and is never re-routed onward to another
node. Inference arriving this way is admitted first (see "Node policy" below).
Model-list reads use the same policy-derived catalog as local clients.
Membership and pins are re-derived per request, so joining or leaving a
cluster needs no restart.

**Peer transport.** Cluster candidates use the shared pinned peer/service
adapter with the engine's advertised facade endpoint (`lc` for llama.cpp, `e-<id>` for a descriptor engine),
including runtime port changes. It prefers direct mTLS, then uses the existing
authenticated reverse session after a bounded failure before HTTP request
dispatch. The stream still carries inner mTLS to the receiving facade; it does
not enter that facade's loopback plaintext routing path. Current admission,
pins, endpoint generation and the receiver's local policy remain authoritative.
A failed direct probe alone does not remove a pinned model owner from the
discovery overlay. Self and manual targets retain their existing transports.

Reverse transport preserves streamed status, headers, ordered body bytes,
backpressure and cancellation without buffering the complete response. It does
not replay a request after dispatch or splice an interrupted response. After
response bytes have been exposed, an interrupted stream terminates that attempt;
new requests can use a restored session. The proxy's existing dispatch policy
continues to own retries before its commitment point. No listener is added.

**Origin stream integrity (SC-a).** After the first body byte commits an
inference attempt, the origin forwards complete SSE events
with their original bytes and delimiters. Supported routes are OpenAI
`/v1/chat/completions` and `/v1/completions` with `stream:true` or an SSE response
content type. Framing uses the effective body
dispatched by each attempt, after its profile defaults. Explicit `stream:false`
keeps the raw non-streaming path. A single `stream:null` is accepted without rewriting
request bytes and has the same framing meaning as omission on that route.
OpenAI's omitted flag defaults false; SSE content type
can still select management. Other routes, including Responses and Anthropic, remain
unmanaged even when they return SSE.

Paired ingress reports its effective framing settings in the internal response
header `X-PAIR-Stream-Framing: v1;<mode>;<choices>`, where mode is `declared`,
`disabled` or OpenAI `auto`, and choices is a positive decimal integer. The
destination derives it after its own defaults, overriding engine-supplied
values. The origin requires valid metadata on successful peer responses to
these supported inference routes, consumes it before body commit, and strips
it from client headers and trailers. Direct engine responses never supply
authoritative peer metadata. Older peers without the header are incompatible:
the attempt fails with `peer framing metadata missing (version-incompatible peer)`
and existing pre-commit failover tries another candidate. Invalid metadata has
the corresponding `peer framing metadata invalid` reason. If attempts run out,
the existing explicit upstream error reaches the client and the workload fails;
no peer body bytes are exposed. Error statuses, unsupported routes and
non-inference requests retain their existing handling.

Supported inference routes explicitly request `Accept-Encoding: identity` on
both origin and destination engine dispatch, including when the client asks
for gzip. An engine that still compresses a managed stream fails closed.

Successful managed streaming requires protocol completion: OpenAI `[DONE]`, or
all requested choices' finish reasons followed by clean EOF.
The next complete valid event must arrive within 60 seconds, including SSE
heartbeat comments; partial byte drips do not refresh that idle interval.
Frames use volatile memory only, with a 1 MiB raw-frame limit. Frame reservations
compete with all other inference storage for the configured process-wide buffer
root across facades. Each request's frame scope remains limited to 64 MiB.
No response content is logged or persisted, and no whole-response prefix is retained.

Effectively enabled streams prepay a bounded reader envelope, including raw-frame
and SSE payload replacement overlap up to the frame limit. Growth replaces one
array at a time, so SSE prepays three 1 MiB arrays rather than two replacement
pairs together. Allocations borrow that lease and cannot exceed it. Each aggregate acquisition
uses the root's FIFO queue and configured inference-buffer wait; there is no
independent process-wide stream cutoff. Local non-streams retain telemetry
preparation and peer non-streams add none. Admitted readers can grow within that envelope and drain
after a cap decrease; new admissions and pre-output retries require current capacity. See the
spec's preparation ceiling table and drain conditions, including protocol
defaults and responses that acquire fresh reader grants.

An interrupted/malformed stream, idle expiry, overflow, exhausted frame budget,
empty declared stream or wrong streaming media type ends with a protocol error.
The origin discards a torn trailing frame, emits an OpenAI error event followed
by `[DONE]`, and finishes HTTP framing normally.
Already forwarded upstream error objects remain failures without a duplicate
error. HTTP status remains committed (typically 200); the workload instead ends
`failed` through `workload:errored`, with a bounded operational reason and node.
Client cancellation/disconnect and failed client writes remain `cancelled`.
An unmanaged inference copy abort also fails the workload, but its arbitrary
protocol receives no synthetic error or framing repair. Managed bodies discard
upstream Content-Length and integrity metadata that would no longer describe
the body, while retaining endpoint content type, routing metadata and
engine-owned CORS permissions.

One `routingSpend` owns acceptance, attempts, cancellation and raw-byte
commitment; one `responseSink` owns delivery and distinguishes genuine output,
synthetic endings, protocol completion and successful delivery. Parsing and
reader allocation cannot move the first accepted raw-byte commit point or
authorize fallback after it. A normalized peer response ends with the internal
`X-PAIR-Stream-Framing` trailer `v1;completed` or
`v1;interrupted;<reason>`. The origin validates and strips it, so a synthetic
`[DONE]` from a destination cannot make an interrupted peer workload completed.
Missing or invalid disposition is a peer interruption, not completion.

Before commitment, a paused local attempt can be excluded and retried on an
eligible destination within the existing budgets. A peer destination is terminal
and leaves fallback to the origin. Pause does not cancel work already executing
elsewhere or stop this router's listeners.

SC-a never dispatches again after commit. A pause, peer or engine loss mid-stream
ends a managed OpenAI stream with an error frame plus `[DONE]`, not a pause,
until SC-b. Endings are best effort when delivery
fails. Qualified same-model continuation
(SC-b) and real-engine/native acceptance remain pending. Origin-local history
is available to consumer banks independently; its presentation remains separate.
Destination ingress remains local-only with no onward dispatch;
recognizing a terminal frame does not resolve an uncertain strict admission
hold. Non-streaming whole-body recovery remains outside SC-a. It does not splice
responses or continue at a paid provider.

**Persisted port.** A port chosen at runtime via `set-port` is saved to the
per-user data dir (`%LocalAppData%\Nvidia Corporation\Personal AI Router` on
Windows, `~/.config/Nvidia Corporation/Personal AI Router` on Linux) and
**restored when the facade is enabled**, taking precedence over the requested
`port`/the default — so the proxy comes back up where it was last put. Each
engine has its own file, so moving one proxy never moves another.
`ignorePersistedPort` deliberately bypasses that restoration for a
broker-coordinated start. Delete the file (or
`set-port` back to the default) to revert.

Port saves use the shared durable-file replacement helper. A bind failure or
save failure before replacement keeps the old listener and saved port. If the
new file has already replaced the old one but durability cannot be confirmed
(`ErrReplaced`), the proxy keeps the new listener and port, emits `ready`, and
still reports the save error. Memory and disk therefore agree in either case.

**Browser clients (CORS).** PAIR does not enable CORS or add default browser permissions. Configure origins through the engine; its built-in defaults and user configuration remain authoritative. Ordinary forwarded responses preserve the upstream status, body, and CORS headers, including missing headers. A denial is never replaced with a successful OPTIONS response or retried to find permission elsewhere. Proxy-generated errors carry their actual status without CORS permission headers, so browser JavaScript may see a generic CORS failure while curl and diagnostics show the real error.

A browser preflight (OPTIONS with Origin and Access-Control-Request-Method) queries every currently routable target, with concurrency eight, a ten-second query deadline, and no redirects. A single target's response is relayed. Multiple responding targets must all permit the requested origin, method, and headers; PAIR grants only their shared permissions. Credentials require unanimous explicit support. Synthesized preflights allow browsers to cache the result for 60 seconds; PAIR itself does not cache decisions. A policy denial returns 403. Unavailable targets are skipped; if none can answer, the proxy returns 502. Ordinary OPTIONS requests retain normal routing. Preflights do not create inference jobs or reserve scheduler capacity. Paired ingress forwards only to its local engine.

Profile catalogs with an Origin header require browser permission from the entry facade's engine and every engine contributing visible profiles. These checks forward the caller's origin and end-to-end headers, excluding Authorization and Cookie so credentials are not shared across engines. Multi-target preflights apply the same credential filtering. Every responding node of each contributing engine must return a valid list and permit sharing: one denial returns 403 and one invalid list returns 502. An invalid-list error retains the combined CORS permissions when every contributing engine allows the origin. Unavailable nodes are skipped within each engine, with 502 returned if any contributing engine has no responders. Successful checks combine origin/credential permissions and Vary requirements, but native model records never enter the profile catalog. Peer-ingress permission checks use only this node's local engines, without forwarding catalog reads onward. Requests without Origin use policy and advertised profiles without querying engine lists. Engines without CORS support remain unavailable to cross-origin browser clients through PAIR.

Both engine facades use nvpair-shared/cors. Neither facade reads engine environment variables or parses launch commands to determine CORS policy.

One limit is outside the proxy's control: current Chromium-based browsers gate a request from a public origin to a local or loopback address behind the user's [Local Network Access](https://chromestatus.com/feature/5152728072060928) permission, which replaced the old server-side opt-in header. No header the proxy sends can grant that. A hosted page needs the permission plus a `fetch(url, { targetAddressSpace: 'loopback' })` annotation; a page served from the local machine is unaffected.

### Node selection

- **Eligibility**: Before routing model-bearing inference, the proxy keeps only
  nodes declaring the exact requested profile key for this engine. Self uses
  policy profiles; peers use advertised profiles. `ServedModels` restricts the
  profile's native model, compared using the engine's naming rule. Unprofiled
  names return `404`; if a known profile has no routable owner,
  the proxy returns a local `502`.
- **Auto**: When no eligible node is explicitly selected, the proxy follows
  `node/set-priority`, then discovered nodes in stable ID order.
- **Priority (scheduler-driven)**: The Job Scheduler ranks the cluster
  least-loaded-first by pending workload plus smoothed GPU pressure and, via
  `nvpair-ui-broker`, pushes the ordered node list with those per-node counts to
  this proxy with `node/set-priority`. Auto routing sends the request to the
  listed node carrying the least estimated load. See
  [`nvpair-job-scheduler`](../nvpair-job-scheduler/README.md).
- **Manual**: `node/select` pins traffic to a specific node. A manual pin
  **overrides the priority list only when that node is eligible** for the
  requested model.
- **Failover**: If the selected node disappears from the discovery set, the
  proxy falls back to auto-select and emits `node/selection-changed`. A
  transport error or retryable status, including a model `404` from an
  advertised owner with stale inventory, steps to the next eligible owner.
  Once a round's owners are used up, an inference request re-resolves and keeps
  trying under a dispatch budget and a wall-clock deadline, and it commits to a
  node at the first byte of response body rather than at its headers. `spec.md`
  §5.1–§5.4 is normative for the bounds, the commit point, and the statuses.
- **Paused peers**: For inference, including requests without a `model` member
  and tier requests, `resolveRouteCandidates` leaves a remote node out of the
  result set when its advertised pause hint `Node.PausedBy` is non-empty. Self
  and dormant self (judged by their own admission), manual nodes (no hint),
  model lists, preflights and other non-inference requests are not filtered. The
  hint is a snapshot refreshed on a nominal 15-second cadence and kept when a
  fetch fails, so the destination's ordinary non-strict admission `503`, with
  the header
  `X-PAIR-Admission: paused` and `Retry-After: 30`, wins over a stale hint. The
  router can try another candidate before committing response bytes while its
  dispatch budget permits; on its final permitted attempt it relays the `503`.
  This is a fast admission refusal, not a promise that another node can serve.
  Strict hybrid admission also refuses a non-available destination, using its
  `strict-admission` error rather than the ordinary pause headers.
  When hints exclude
  every remote candidate and no self candidate remains, a concrete request gets
  `502` and a tier request `404`. See [spec §5](spec.md#5-request-path),
  [§12.2](spec.md#122-admission) and
  [Node availability](../../docs/node-availability.mdx).

  Source path: the broker's engine-addressed `discovery:nodes` snapshot enters
  `replaceSubscribed` in [`facade.go`](facade.go); `subscribedToNode` in
  [`proxy.go`](proxy.go) copies `DirectoryNode.Policy.PausedBy` into `Node.PausedBy`.
  [`Discovery.SetSubscribed`](discovery.go) replaces that internal snapshot even
  when only the hint changed. Such a change emits no outward proxy `node/updated`,
  and it does not revoke an already-resolved attempt. Eligibility is checked by
  `resolveRouteCandidates`; destination refusal and retry are owned by
  [`admission.go`](admission.go) and [`routing.go`](routing.go).

### Node policy: admission, tiers, cancel (fork)

A profile names its engine. Concrete model names use the entry engine's
inventory and eligible dormant self candidates; ordinary admission, on-demand
wake and retryable failover retain their existing owners and bounds.

The broker pushes this node's policy (`nvpair-shared/nodepolicy`) with
process-scoped requests — `node/set-policy`, `node/set-availability`,
`node/set-engine-drain`, `node/set-residency`, `node/set-engine-intent` — each
validated and refused with the previous state kept when invalid. All but
`node/set-policy` answer `{"ok":true}`; `node/set-policy` answers an
acknowledgement `{revision, runId}` of the policy revision it activated and the
process's run id, which is how the broker learns that a persisted revision is
active. It is the only request that activates a revision: the broker follows a
persisted availability change with a `node/set-policy` of the same revision. A
`node/set-policy` older than the active revision is refused as out of order and
the previous policy stays. Until the first request arrives the proxy runs on `nodepolicy.Default()` with no active revision.

**Inference buffers.** `routing.buffers.capacityMiB` accepts 192..1024 MiB
(default 256), and `routing.buffers.waitSeconds` accepts 0..600 seconds
(default 5; 0 disables waiting). These bound owned inference allocation
capacity process-wide, including retained bodies, rewrites, parsing and response
storage across facades and paired ingress; they do not bound process RSS or
engine memory. Storage is charged before allocation and through the last
borrower, including overlapping old/new arrays during growth. Activation
requires `capacityBytes >= 2 * effectiveIndividualLimitBytes + 1` (the
individual body limit defaults to 64 MiB); otherwise `node/set-policy` keeps
the previous active settings.

Each canonical live registry entry also holds a nonwaiting root grant for its
metadata and owned key bytes until final cleanup, even with no body or facade.
Failed admission returns 503 before body/effects and leaves the request counter
unchanged. There is no separate live-entry count limit. Cancellation of an
existing owner needs no additional grant; cancellation alone does not release
its metadata. Both registry and workload-index map backing have their own
grants from this same root, retained across partial drains and reused by later
insertions. Empty backing is discarded and released, including an empty
workload index with live canonical requests. Deletion never rebuilds or waits
for a grant. Insertions periodically rebuild under an overlapping old/new
backing grant to bound churn; they retain no terminal rows or per-removed-row
leases. `routing/active` first reserves its key clones and overlapping
result/envelope encoding storage, including the caller's ID and newline, from
the same root. Capacity exhaustion or a reply exceeding the existing 1 MiB IPC
frame capability returns a bounded control error before copying. The single
snapshot lease remains held until response I/O succeeds or fails, including
when an original entry is removed concurrently.

Aggregate acquisitions wait in FIFO order, bounded to 1,024 waiter records,
with head-of-line blocking: a small request cannot bypass a larger head.
Each wait observes cancellation and the earlier of its snapshotted allowance
and context deadline, including the overall pre-output deadline. An oversized
body is 413 immediately once its size is known; aggregate busy, full queue or
wait expiry is 503. An impossible per-request footprint is also 413. A live
cap decrease leaves existing storage to drain, possibly above the new cap;
impossible queued proposals become 413, and new/growing allocations and retries
must fit current capacity. Existing wait deadlines do not restart. Stream
grants and immediate category refusals are detailed in `spec.md` §§5.2 and
12.1. Settings persist in node policy and activate live through the broker.
The desktop's Node policy JSON editor supports them; the TUI exposes no buffer
controls. [Hybrid routing](../../docs/hybrid-routing.mdx) includes an edit example.

The internal process-scoped `node/admission-snapshot` request takes an explicit
`{}` and returns `{runId, active, activeByEngine, queued, lastActivityMs}`.
It copies current counts under the admission mutex, including held and
quarantined slots. When settings drain waits for active work, it requires fresh
readback after the drain acknowledgement from that same proxy run; a cached
zero is insufficient.

`spec.md` §12 is normative; in short:

**Model inference settings:** `requestOptionsMode` is `defaults` when omitted:
request values win. In `enforced` mode, configured scalar and array values
replace request values and objects merge recursively, without changing
unconfigured request fields. Ambiguous configured keys refuse with HTTP `400`
and a fixed message before engine contact. Qualification compares effective mode
and options; a mismatch invalidates installed strict capacity. Option comparison
uses the same precision-preserving comparator as the target policy lock: equal
numeric values such as `0.1` and `0.10` match, while distinct high-precision values
do not. Absent options, defaults-mode `null` and `{}` match; object key order is ignored and array
order matters. See [qualification binding](../../docs/local-qualification.mdx#installing-strict-capacity).
The serving node owns this processing: a request routed through the Mac to PC3 uses PC3's
profile, not the Mac's peer-candidate defaults. See the
[broker contract](../nvpair-ui-broker/README.md#model-inference-settings-and-target-owned-locks)
for the independent `localOnly` edit lock, desktop controls and precedence example.

- **Admission.** One controller for the whole process decides whether
  inference may run on this machine: in the cluster ingress before forwarding a
  peer's request, and before dispatching a local request to this node's own
  engine. It checks availability and engine drains, wakes a stopped engine
  saved On (`admission/wake`, then waits for `node/set-local-backend` healthy),
  limits concurrency per model, keeps the resident model set and the memory
  budget, unloading idle models via `admission/unload` when `switchModels` is
  set, and merges the model profile's `requestOptions` into the body. A refusal
  is `503` with `X-PAIR-Admission: <reason>` and `Retry-After`; a router fails
  over on it. Waits are bounded by `X-PAIR-Admission-Wait`, which the router
  sets to `0` while a round has other candidates. It reports
  `admission/state` (active, queued, last activity per engine) to the broker.
- **Roles.** Roles are user-created names, ordered weakest to strongest on each
  node, with no built-ins, defaults, protected names or cluster role store.
  A request for a visibly declared role is resolved to a concrete model from
  the roles nodes declare, filtered by what the request needs (tools, vision,
  structured output, embeddings, context) and ordered by tier and by whether the model is
  already loaded. Each node (a peer's `policy`
  advertisement, or this node's own `roles`) is authoritative for itself: for a
  tier it declares it offers its ordered profile list, with each profile's own
  bounds, and for a tier it omits it offers nothing. Roles are ordered
  `{name, profiles}` records; their profile keys resolve against that node's
  keyed facts. Self includes local-only profiles, while peer facts withhold
  them. Missing keys are skipped without hiding later members; empty or fully
  unreachable lists still declare a role. Local tier entries are retired;
  a node with no declarations supplies no local role candidates. Widening
  follows the requesting node's own role order: requested role, stronger roles
  nearest first with `allowStronger`, then weaker roles nearest first with
  `allowWeaker`. A peer-only role absent from that order never widens.
  `preferLoaded` remains this router's policy.
  Within equal rank, warmth, preparation and selected-pin classes, earlier
  list positions lead. Each candidate uses its own node's membership position,
  capabilities and context, even when another node lists the same pair earlier.
  Hybrid qualification also uses the selected engine/model's own profile facts.
  OpenAI-compatible routes may
  be served by another engine; native routes stay on their own. The body's
  `model` is rewritten in place. Nothing eligible is a `404` naming the needs.
  Any role may have `api` entries naming provider deployments from the
  policy's origin-only `routing` section. Public selection requires both that
  entry and a `routing.tiers` rule allowing remote use. A deployment names its
  `endpoint`
  (the full `https` request URL with a lower-case public DNS host and a path;
  no IP literal, local name, userinfo, query, fragment, percent-encoding or
  explicit `:443`), its `protocol` (`openai-chat-completions` only), its
  finite `transport` bounds (connect, header and idle timeouts, response,
  error-body and streamed-event byte caps) and its provider `qualification`
  revision; enabling a deployment requires all four and prices. Peers never
  see any of them. API configuration is private to the origin node; remote role
  editing cannot inspect or change it. Public dispatch is disabled in this
  build: enabled deployments and remote or paid permissions cannot be saved.
  Naming or selecting an API route does not activate dispatch or bypass its
  permission owner; API overrides are listed and refused without provider calls.
- **Hybrid aliases.** Role names match `^[a-z0-9][a-z0-9._-]{0,63}$`.
  Exact `<role>:api` and `<role>:local` override the bare role's mode.
  Hyphens are ordinary characters: `coder-api` is an ordinary name.
  Every public profile key with a valid role-name prefix followed by a colon
  is reserved, regardless of declaration or suffix. Native/provider model and
  artifact identifiers are exempt; otherwise valid 512-byte profile keys remain
  valid. Classification requires a visible declaration; API entries alone do
  not declare a role. An unknown named model returns `404` with
  `no profile or role named <name>`, distinct from a declared role's no-eligible
  model failure. `/v1/models` lists self roles in policy order, then peer-only
  names sorted by name, each followed by `:api` and `:local`, including empty
  roles. Synthetic role records replace colliding concrete records. When no upstream
  answers and roles are declared the
  synthetic records are still returned instead of `503`, and without
  configured tiers the statuses are unchanged.
  All variants of a role alias carry `context_length`, the largest known
  context among that role's visible resolved profiles across self and peers
  whose declaring node is returned by its engine facade's inference candidate
  resolver for that model. Self facts require a self candidate; peer facts
  require that exact peer, so another node's inventory cannot qualify them.
  Each catalog resolves each engine/model pair once and computes each tier's
  bound once, sharing results across alias variants only until that catalog
  returns. These resolver checks can refresh mesh state and start background
  target-address confirmation. These observation effects are accepted; catalog
  checks introduce no policy, engine or preparation action.
  It is omitted when none is known or any such profile is unbounded
  (`contextTokens: 0`). Missing or withheld keys supply no facts; router tier
  entries do not supply this field. This is a catalog contract, not live
  validation: client
  acceptance remains pending N8. Typed profile context comes from the broker's
  resolved per-slot launch context; native-fit realized-context agreement
  remains pending N7. On the concrete facade boundary a request is answered `400`
  `{"error":...,"code":"feature_unavailable","reason":"alias_not_inference_ready"|"unsupported_alias_use","model":"<alias>"}`
  before the CORS preflight, candidate resolution, admission, wake,
  reservation, workload events or any engine request when it names a declared
  role's `:api` or `:local` override in body `model`, or override syntax in an
  alternate selector: a top-level JSON body key equal under case folding
  to `model` or `name` (repeated keys included), a `model` query parameter, or
  the remainder of a `/v1/models/` path; or when it names
  a bare declared role in the body `model` on a route that is not a classified inference
  route (Responses, an unclassified native or passthrough route), or with
  surrounding whitespace. Nothing else is checked: nested keys, other fields and
  other paths are not. A bare tier in `name`, a query parameter or a model path
  keeps its existing behavior. An undeclared canonical body model remains concrete.
  A bare declared role exactly spelled in `model` on a classified inference
  route takes the facade's local role path. Unified listener inference uses
  the hybrid entry below. Cluster ingress has its own listener boundary.
- **Routing listener.** When the active policy sets `routing.listener.enabled`
  with a `port`, the process itself (not an engine facade) binds
  `127.0.0.1:<port>` and serves the hybrid aliases with no engine running. It is
  off by default, loopback only, and never advertised or exposed to the LAN. It
  follows `node/set-policy`: a changed port binds first, serves with the same
  HTTP server, then closes the old acceptance socket. Existing connections
  keep their contexts. Disable cancels the server context and bounds shutdown
  to three seconds before forced close; re-enable creates a fresh server.
  The broker's replay after a proxy restart re-establishes it. A port already
  in use preserves the working listener during rebind; an initial collision
  leaves none. It never picks another port (including a port a facade of this
  process or another socket holds on a wildcard address or `[::1]`, and a
  facade is likewise refused the listener's port). A collision is reported as
  the service error
  `nvpair-proxy:routing-listener-blocked` (`errors:report`, cleared when the
  listener binds or is disabled); the policy still activates, and replaying it
  retries the bind. It serves only:
  - `GET /v1/models`: the same synthetic role records and ordering over self
    and visible peer declarations as facade catalogs. `context_length` uses
    the same declaring-node eligibility and unknown/unbounded omission rule,
    including self's local-only profiles when eligible. No upstream request
    is made.
  - Every inference route any engine profile declares (`POST /v1/chat/completions`,
    `/v1/completions`, `/v1/embeddings`, `/v1/messages`): enters the common inference
    owner after the Host, Origin, path, method, query and preflight gates.
    Owned buffering uses the effective individual request limit; all top-level
    model identifiers are checked before selection. One exact model member
    is required (`400` `unsupported-alias` otherwise); duplicate or case-folded
    model members return `400` `ambiguous-model`. Nested model values remain
    payload. Visible declarations distinguish roles from concrete names.
    Bare aliases follow `bareMode`; `:local` excludes APIs and `:api`
    excludes local destinations. With public dispatch disabled, an API-only
    request has no eligible destination (`404`) and does not fall back locally.
    Other supported aliases select only qualified declared roles and carry the
    strict destination envelope on peer hops. A query string, including a bare
    `?`, is `400` `query_not_allowed`, before body reads or request registration.

  `serveHybridInference` in `routing.go` selects a compatible enabled facade
  deterministically for the driver's dialect and existing notification ID
  sequence. It creates no facade listener or discovery subscription. With no
  compatible facade, a request-local empty adapter is pinned to no local
  candidates even if a facade is enabled while the body is arriving. With public
  dispatch unavailable it emits no workload, so its notification ID cannot
  collide with a real facade's workload. Rejections and successful requests
  unwind the same buffers, cancellation and canonical registry as facade
  inference. Rebind preserves committed streams; disable/shutdown cancels them
  and forces socket close within the three-second listener grace. Parent I/O
  retains its separate codec/transport lifetime; cancellation alone does not
  interrupt `codec.Read`.

  Other paths are `404` and other methods `405`. `Host` must be `127.0.0.1` or
  `localhost` with the accepted connection's local port, else `403`
  `host_not_allowed`. Connections surviving a rebind still require their old
  port in `Host`. A request with
  an `Origin` header is `403` `origin_not_allowed` unless the header is
  exactly one of the policy's `routing.listener.allowedOrigins`; for those,
  responses carry `Access-Control-Allow-Origin` and a preflight (`OPTIONS` with
  `Access-Control-Request-Method`) is answered `204` for the route's own method
  and the headers `Content-Type` and `Authorization` only, else `403`
  `preflight_not_allowed`. Engine facades and their CORS behavior are unchanged.
- **Routing metadata.** Routed inference responses carry `X-PAIR-Model`,
  `X-PAIR-Engine`, `X-PAIR-Node`, and `X-PAIR-Tier` for a tier request.
  Workloads carry the concrete `model` and, for a tier request,
  `requestedModel`.
- **Cancel.** `workload/cancel {originatedFrom, runId, engine, workloadId, regenerate?}`
  requires the complete target at every hop and cancels an in-flight workload
  this node originated. Missing/malformed keys are invalid params; a valid key
  for another origin/run or an ended workload finds nothing. Records missing
  a run ID are displayable but not cancellable. Before the response commits,
  `regenerate` re-dispatches it to another node; otherwise, or after the
  commit, it ends as `cancelled`.
- **Provider account ledger.** The proxy is the only writer of provider
  account state, kept in `proxy-accounts.json` under the app data directory and
  guarded by an exclusive lock on `proxy-accounts.lock` (`flock` on macOS and
  Linux, `LockFileEx` on Windows) held for the process lifetime, so a second
  proxy on the same data directory refuses it. Both files are created only once
  an activated policy names a provider account. The store is one versioned,
  checksummed document replaced through `durablefile` on every change, and a
  damaged store is never rewritten. It records money holds and reconciliation
  in exact nano-USD with each hold's price, policy revision and qualification,
  spend per UTC day and month, credential-reference bindings that outlive
  account deletion, cooldowns with one half-open probe, credential-failure
  suppression, freezes, and quarantined deployment identities. The last configured
  credential reference persists through retirement so delayed obsolete-reference
  failures cannot replace its failure; re-adding with a different reference is
  an ordinary reference edit. Reconciled holds
  remain retained and count against admission capacity until both the account
  outcome and qualification classification are durably recorded. Every failure
  refuses spending and never blocks policy activation or local routing. No
  user-traffic path calls it. G's `runPublicChatAttempt` is the sole intended
  production wrapper around the private `runChatAccountAttempt` library runner;
  the wrapper remains unwired from HTTP ingress. See `spec.md` sections 12.7–12.9.

### Public egress lease (G; disabled)

The trusted broker parent alone sends `node/renew-egress-lease`
`{runId, revision, sequence, notAfterNanos}` and `node/revoke-egress-lease`
`{runId, sequence}`. These process-scoped requests are not desktop or peer
capabilities. A renewal answers `{runId, revision, sequence, nowNanos}`; a revoke
answers `{runId, sequence}`. A successful wire acknowledgement grants no public
permission by itself.

The broker renews once per second only for its exact persisted desired revision
and acknowledged active proxy run, with policy and availability pushes complete
and no activation, recovery or withdrawal uncertainty. A zero deadline is an
anchor-only clock sample; later grants use that run's proxy monotonic clock and
expire no later than issuance plus five seconds, even after pipe delay. At
`now >= notAfter` permission is refused without grace. Sequence high-water
survives policy changes and revocation within a run; a replacement inherits no
lease. Credential-reference replacement disables renewal and attempts a bounded
revoke before stopping/joining the old proxy, even if the revoke fails.

`remoteDispatchAvailable=false` in `shared/nodepolicy/routing.go` remains the
single build availability owner; `PublicDispatchAvailable()` only reads it.
There is no environment variable, flag or second enable bit. Disabled provider
configuration can be staged; enabled deployments and remote/paid tier permissions
are rejected. Production public traffic remains zero.

I's advisory `publicAttemptPermitted(tier, snapshot)` can skip a candidate before
body construction or qualification, but each attempt must enter
`runPublicChatAttempt` with its request-bound lifecycle. D durably reserves and E
prepares/accounts; G checks permission early, after reserve and immediately before
`http.Client.Do`, where the same guard accepts I's still-live attempt. Refusal
after reservation retains the hold. I owns tier/mode selection, qualification,
request identity, cumulative allowance, deadline, cancellation and output
commitment; it must not call the raw runner/transport or cache a successful guard.

See `spec.md` §12.9 and [Public egress lease](../../docs/public-egress-lease.mdx)
for the clock formula, refusal stages, activation prerequisites and acceptance
limits. Real-provider activation, external generation and process acceptance
remain separate gates.

### Strict Chat adapter library (E; inactive)

`chatQualification.prepare` and `parseResponse` implement bounded, stateless text
Chat Completions, including qualified function tools, structured output and SSE.
Acceptance is the intersection of the strict grammar and a supplied qualification;
unknown/stateful fields, multimodal input, opaque reasoning, Responses, Anthropic
Messages, native engine protocols and embeddings are unsupported. Input accounting
covers the final body and qualified framing overhead, with finite output/context
bounds. `newPublicClient` uses the configured HTTPS URL, public-address-checked
numeric dials and verified TLS; it rejects redirects and compressed responses,
ignores environment proxies, and never forwards raw provider headers or trailers.

`runChatAccountAttempt` ties one send to a pinned account snapshot, selected-reference
resolver, mandatory guard, `publicAttemptLifecycle` and finite byte-budget hooks.
It neither selects targets nor retries. G supplies the production guard through
`runPublicChatAttempt`; E preserves the final pre-send refusal sentinel for G's
typed result. Ledger outcome, money and qualification completion remain independent;
incomplete or untrusted responses retain their money hold.

This adds no route or RPC exposure, configured provider or real qualification.
`remoteDispatchAvailable` remains false. G's lease and wrapper are present; F must
supply real evidence and freshness, B finite local qualification, and I complete
public ingress/attempt/cancellation wiring and the aggregate buffer owner. D's native
lock/crash-window acceptance and owner recovery controls remain activation gates.
Synthetic profiles and successful test guards activate nothing. The normative
matrix, limits and completion rules are in `spec.md` section 12.8.

### IPC Transport

By default the proxy communicates over **stdin/stdout** using newline-delimited
JSON-RPC 2.0 (one message per line). All diagnostic logging goes to **stderr**.

For environments where stdout may conflict with the host process (e.g. Electron),
pass `--ipc` to redirect the JSON-RPC channel to a named socket or pipe. The
parent process should create and listen on the endpoint before spawning the
proxy.

```bash
# Default: stdin/stdout
nvpair-proxy

# Unix domain socket
nvpair-proxy --ipc /tmp/nvpair-proxy.sock

# Windows named pipe
nvpair-proxy --ipc \\.\pipe\nvpair-proxy
```

## JSON-RPC 2.0 Protocol

The control channel is identical for every engine; only the broker's relay
prefix differs. The generated method surface — every request and notification,
with payload shapes — is in
[`desktop/docs/services-api.md`](../../desktop/docs/services-api.md), produced by
`npm --prefix desktop run service-contracts:write`. Read that rather than a
hand-maintained copy, which is how the two per-engine READMEs drifted.

The proxy emits `ready` on startup carrying its bound port, which is the
authoritative source of where it is listening — do not assume a port before it
reports one.

## Shutdown

Closing stdin (observed as EOF) drains the HTTP server and exits. The broker
owns the lifecycle: `shutdown` relayed from a client is refused.

## Adding an engine

Add one entry to `nvpair-shared/engines` for the cross-process identity, one to
`engines.go` here for the proxy-only values, and one to the broker's
`engineproxy.go` for its ownership. See the "Adding a new engine" section of the
proxy-unification plan for the full checklist, including the one-liners that are
still per-engine (a `noderec.ServiceKey`, the TUI health label, the desktop
engine-type constants, and an engine-manager manifest).
