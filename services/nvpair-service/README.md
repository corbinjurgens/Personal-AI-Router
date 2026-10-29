<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-service

## Local staged-build contract

`service/build-apply-preview`, `service/build-apply-start`,
`service/build-apply-status` and `service/build-apply-gui-ready` are local
service methods. Preview reads the current broker's private `build:apply-impact`
envelope with a five-second budget. The envelope carries public impact and
private manager evidence; only public impact enters the preview. Manager model
and data roots, including declared install roots, are joined with service
profile roots before either package is inspected. Missing root coverage or a
changed manager identity refuses before inspection. A second read checks the
same manager capture, roots and broker fences after package validation.

The service's production owner-read adapter also works while idle, without a
preview slot. It retains broker-run binding, cancellation and pending-request
cleanup, and projects only facts supported by the manager document. It cannot
claim identity membership, owned worker lifetimes, saved/enforced Pause or
physical stop. Production start stays refused: a read does not provide the
atomic `WithLocked` acceptance owner, and those other proofs are still absent.
Unknown historical copy safety and unrepresented external data roots also
refuse. Preview performs validation and read-only inspection but never starts
the candidate or reserves maintenance.
The registered `service/build-apply-changed` operation notification uses the
existing service connection and native desktop notification dispatcher.

Native main owns the candidate picker, GUI PID/executable and embedded GUI
build stamp. Start accepts only the preview ID, interruption consent and native
GUI facts. No method accepts a remote node/caller, command, renderer-selected
endpoint or destination. GUI-ready acknowledges a new GUI; it is unavailable
until the transaction owner independently checks the expected process and path.
The early `--pair-build-apply=OPERATION_ID` entry handles successor GUI
acknowledgement before ordinary startup. Windows shortcut command mode likewise
runs before ordinary single-instance, connector or updater work.

The shared contract separates current GUI, bundled service, requested GUI and
service, attached running service, measured SHA-256, source revision and dirty
state. Unknown observations use `{known:false,value:null}`. Saved, enforced and
physically stopped results are independent. The accepted restore policy finishes
while paused, preserves intent/locks and newer pauses, requires explicit local
Resume, and performs no automatic rollback. Preview validation is read-only;
transaction admission still requires the missing production acceptance owner.

Build identity inspection uses standard `debug/buildinfo.Read` on an already
opened bounded regular-file `io.SectionReader`, the root-approved equivalent
of the planned `ReadFile`. This avoids reopening a substituted path. Descriptor
and path identity/size/modification checks bracket parsing; a read budget and
object-header allocation bounds limit malformed metadata. Supported inspection
formats are current 64-bit little-endian PE, ELF and thin Mach-O. A missing VCS
revision or modification field stays independently unknown; it never means
clean. The existing `git:<revision>[+dirty]` ID requires both fields. Artifact
hashes and the Vite-embedded GUI stamp do not fill missing Go VCS metadata or
authenticate a publisher. Native linked-worktree/standalone packaging evidence
is a separate validation gate.

The per-user, long-running owner of `nvpair-ui-broker`. It spawns the broker
over stdio exactly as a front end used to, restarts it if it dies, and lets any
number of clients attach to it and detach from it over a local socket or named
pipe. Quitting a client leaves the service, and the inference it serves,
running. Stopping the service is an explicit request.

```text
Electron app ─┐                        ┌─ nvpair-ui-broker ─┬─ nvpair-proxy
nvpair-tui ───┼─ local socket / pipe ── nvpair-service      ├─ nvpair-engine-manager
another client┘   (attach / detach)    (owns broker stdio)  └─ … every other worker
```

The broker speaks newline-delimited JSON-RPC 2.0 over
stdio to a single parent, which is now this process.

## Engine notification overflow recovery

The broker's bounded engine-event sender reports a delivery gap through private
`service/engine-notifications-reset` once its output drains. The service
intercepts this notification before ordinary caching/broadcast and accepts it
only from its current live broker run. Clients cannot invoke or inject it.

Engine notifications are broadcast to every attachment, so all attached clients
can have missed state. The service withdraws those registrations, closes them
through the existing detach/closeNow owner, clears the acknowledged
`engine:subscribe` and cached `engine:ready`, and invalidates pending engine
subscriptions so a late acknowledgement cannot restore the old cache. Internal
subscription callers are completed once. Other pending requests, subscriptions
and baselines retain their existing owners; accepted broker work is not
cancelled. Detach does not release a history writer's borrowed backing early.

The broker and engine-manager stay running. The service retains the actual
broker `app:ready`, rather than synthesizing another ready event, so reconnecting
clients use their existing subscribe and initial-state reads. The reset is
consumed here and never exposed as a desktop/TUI API. Broker and service must be
updated together for this recovery path.

## Origin-local routing history relay

Production constructs the history-aware service before starting broker
supervision or accepting clients. It owns one broker history subscription and
one cache of the newest 256 records; the broker remains the accepted-history
authority. Local clients use `routing:subscribe {}`, `routing:unsubscribe {}` and
`routing:get-initial {}` (reply shape `{requests: RoutingHistoryRecord[]}`),
and subscribed clients receive full-record `routing:changed` notifications.
Subscription state is independent of other domains; subscribe before initial
hydration and merge concurrent pushes by complete origin/run/request key and seq.

Private `routing/history-window`, `routing/request`,
`routing/history-window-end` and `routing/history-accepted` never reach clients,
peers, managers or scheduler and cannot be injected through this endpoint.
Electron's `routing-history:ready`, `routing-history:pull` and
`routing-history:consumed` controls belong to its main/preload/renderer boundary;
they are not service methods. TUI consumes the public service wire directly.

There is no subscriber cap. Existing client registrations carry at most 32 bytes
of history scalar cursor/flags each. At most eight history writer leases run
concurrently, including one possible global initial transfer. Waiting clients
advance fairly through the existing registry without a per-client data bank or
waiter goroutine. A pinned initial or unavailable writer returns fixed
`-32096`/`routing history busy; retry`; desktop/TUI retain validated rows and
stay loading, with one retry timer at 250/500/1000 ms then one second until
progress or cancellation. Other RPC and inference continue.
Actual write completion, not logical timeout,
releases borrowed storage.

Raw history is guarded before generic parsing/logging; unknown fields are
stripped and recursive duplicates rejected. Record/private values are bounded
at 128 KiB, initial values at 32 MiB and physical newline frames at 33,558,528
bytes. Ordinary service messages retain their 8 MiB logical cap. History's
service capacity `3W+8P+2H+M+S+C` is 152,281,088 bytes, plus client scalars;
this bounds admitted representations, not process RSS. Native producer facts,
ordinary queues and baseline runtime memory are separate. See the
[capacity and semantic contract](../nvpair-proxy/spec.md#1252-origin-local-routing-history)
for the constants, the 32-attempt limit, exact money, waits and workload
association rules.

History is memory-only. Broker restart clears history and resets the relay;
client disconnect clears its consumer generation, and reconnect resubscribes
and fetches a fresh baseline. Proxy death preserves previously accepted rows
but can lose unsent/partial windows; retirement proves no delivery or charge.
Model-less local attempts preserve node/engine and null model (`model not specified`).
Desktop/TUI consumer banks are activated; original Tasks 4/5 presentation remains
separate. The relay adds no history cancel action or SC-b/public-provider support.

## Endpoint

| Platform | Endpoint | Access |
|---|---|---|
| Linux, macOS | `<appdir>/service.sock`, or `$TMPDIR/nvpair-<uid>/service.sock` when that path is longer than a socket path allows | Unix socket, mode `0600`; the fallback directory must be `0700` and owned by the user |
| Windows | `\\.\pipe\nvpair-service-<username>` | DACL granting only the current user's SID; remote clients rejected |

`<appdir>` is `nvpair-shared/appdir.Dir()`. `NVPAIR_SERVICE_ENDPOINT` overrides
the endpoint for both the service and `nvpair-shared/servicectl` clients, which
is how tests use a private socket. The Windows descriptor comes from
`ipc.ListenPrivate`; the default named-pipe DACL would let Everyone read.

**Single instance.** On Linux and macOS the service takes an exclusive lock on
`<endpoint>.lock` before anything else, then dials the endpoint. If the lock is
held or something answers, it prints `nvpair-service: already running` and exits
with status 0. Only then is a leftover socket file removed as stale. On Windows
the pipe's first-instance flag does the same job.

## Running

```sh
nvpair-service                              # serve; broker found beside this binary
nvpair-service --log-level debug            # also passed to the broker
nvpair-service -- --proxy-engines ollama    # everything after -- goes to the broker
nvpair-service status                       # print service/status of the running service
nvpair-service diagnose [--json]            # bounded local report; see Diagnosis
nvpair-service call <method> [json-params]  # send any request through the service and print the result
nvpair-service stop                         # service/stop: stop the broker cleanly, then exit
nvpair-service --version
```

`policy:set` is conditional: it must carry `expectedPersistedRevision`, the
`policy.revision` of the `policy:get` the new policy is based on. Read the
policy first, edit it, then send it back with that revision:

```sh
nvpair-service call policy:get '{}'
nvpair-service call policy:set '{"expectedPersistedRevision":"1792285135482000000","policy":{...}}'
```

A set without the field is refused with `-32602`. A set whose revision is no
longer the persisted one is refused with a `policy conflict:` error that names
the current revision and changes nothing; call `policy:get` again, review the
policy and resubmit.

### Diagnosis

```sh
nvpair-service diagnose          # readable report
nvpair-service diagnose --json   # the same report as JSON
```

`diagnose` answers common local support questions without printing the policy
document. It dials the running service and never starts one. Within one
five-second deadline it reads, in this order and nothing else:

1. `service/status`;
2. the local `policy:get` with empty params (no `nodeId`, so nothing is relayed
   to a peer);
3. `<engine>-proxy:get-status` for each built-in engine facade, in the order of
   the shared engine table;
4. `policy:get` and `service/status` again, as end-of-capture checks.

Each of these answers from state the service and broker already hold. The
command sends no inference, wakes no engine, probes no peer, subscribes to
nothing and changes no configuration. It reports only validated, allowlisted
fields: never the policy's routing or provider configuration, credential
references, error text, model identifiers, paths, hostnames or addresses.
`call policy:get` prints the complete policy, including its private routing
configuration.

| Section | Fields when `known` |
|---|---|
| `service` | `pid`, `brokerPid` (`0` while no broker runs), `brokerRestarts`, `version` and `buildId` when they are plain identifiers |
| `policy` | `persistedRevision`, `activeRevision` (absent when the proxy has activated none), `activation` (`active`, `pending` or `error`), `availability` (`available`, `draining` or `paused`), `recoveryRequired` |
| `builtinFacades` | Per engine: `ready`, and `port` when ready. The port is the one the facade actually bound, which can differ from the engine's stock port |
| `residentConfiguration` | `stateCounts` and `reasonCounts` over the accepted `modelConfiguration` entries |

Every section has a `status`: `known`, `not-running`, `unavailable` (the read
failed or the deadline ended first), `invalid` (the answer was malformed,
oversized or outside the accepted values) or `changed-during-capture`. A
field is present only while its section is `known`; missing values are never
shown as `false` or `0`. Unknown fields in an answer are discarded.
`modelConfiguration` is accepted only through its own strict decoder, so an
unknown field inside it makes the section `invalid`.

The JSON report is a command result with `"schema": 1`, not a broker wire. It
always carries `"snapshotAtomic": false`, `"inferenceProbed": false` and
`"peerConnectivityProbed": false`, and the text report ends with
`Inference and peer connectivity were not tested.`

| Exit | Meaning |
|---|---|
| `0` | Complete capture: every section `known` and the end-of-capture checks matched. Paused, pending, error and recovery states are complete captures |
| `1` | Incomplete capture: a section is not `known`, an end-of-capture check could not be read, or a checked value changed. The service vanishing after the first read is also `1` |
| `2` | Invalid arguments; nothing is dialed |
| `3` | No service answered the first dial; nothing else is read |

The capture is not a transaction and is never retried. The end-of-capture
checks compare the service and broker process ids, `brokerRestarts` and
`buildId`; the policy's revisions, `activation`, `availability` and
`recoveryRequired`; and `modelConfiguration`'s `policyRevision` and
`observationGeneration`. A change marks the sections that depend on it
`changed-during-capture`: a service or broker change marks every section. A
match shows only that those fields did not change, so `snapshotAtomic` stays
`false`.

Limits of what it reports:

- Only the built-in facades are listed; facades of engines defined only by a
  descriptor are not.
- A facade bound on a port is not evidence that an engine or model behind it
  can answer inference. No URL is printed for that reason.
- A `satisfied` resident entry describes an observed load, not a successful
  inference request.
- It does not diagnose peer reachability, firewalls, VPNs or provider
  credentials, and it keeps no history.

A node with a pending policy activation, running on the recovery policy, with
Ollama's facade relocated to another port (synthetic values):

```text
PAIR service diagnosis (cached local state)
Service: known
  pid 4242, broker pid 4243, broker restarts 0
  version 0.2.0, build git:0123456789abcdef0123456789abcdef01234567
Policy: known
  persisted revision 1792285135482000000, active revision 1792285135481000000, activation pending
  availability available, recovery required yes
Built-in facades:
  Ollama: known, bound on port 11500
  LM Studio: known, not bound
  llama.cpp: known, not bound
Resident configuration: known
  states: blocked 1
  reasons: no-fit 1
Capture: complete. Not an atomic snapshot: the start and end checks show only that the compared fields did not change.
A bound facade port does not show that an engine or model can answer.
Inference and peer connectivity were not tested.
```

The same capture with `--json`:

```json
{
  "schema": 1,
  "complete": true,
  "snapshotAtomic": false,
  "inferenceProbed": false,
  "peerConnectivityProbed": false,
  "service": {"status": "known", "pid": 4242, "brokerPid": 4243, "brokerRestarts": 0, "version": "0.2.0", "buildId": "git:0123456789abcdef0123456789abcdef01234567"},
  "policy": {"status": "known", "persistedRevision": "1792285135482000000", "activeRevision": "1792285135481000000", "activation": "pending", "availability": "available", "recoveryRequired": true},
  "builtinFacades": [
    {"engine": "ollama", "status": "known", "ready": true, "port": 11500},
    {"engine": "lmstudio", "status": "known", "ready": false},
    {"engine": "llamacpp", "status": "known", "ready": false}
  ],
  "residentConfiguration": {"status": "known", "stateCounts": {"blocked": 1}, "reasonCounts": {"no-fit": 1}}
}
```

The command prints this JSON indented one field per line. When the broker
restarts during the capture, every section reads `changed-during-capture`,
`complete` is `false` and the exit status is `1`; run the command again.

Clients normally start it themselves: `servicectl.ConnectOrStart` dials the
endpoint and, when nothing answers, starts the service detached — its own
session on Unix (`Setsid`), `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
CREATE_NO_WINDOW` on Windows, stdio on the null device, working directory set to
the binary's directory — and keeps dialing for up to ten seconds. The service
inherits the client's whole environment, including provider credentials in the
reserved `NVPAIR_PROVIDER_SECRET_` namespace.

Windows ignores `CREATE_NO_WINDOW` when combined with `DETACHED_PROCESS`;
`DETACHED_PROCESS` prevents the service from creating or inheriting a console.
The shared `servicectl` launcher owns these flags for both client starts and
the Windows autostart starter.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--broker-path <path>` | `nvpair-ui-broker[.exe]` beside this executable | Broker binary to supervise. There is no PATH fallback |
| `--endpoint <path>` | see [Endpoint](#endpoint) | Socket or pipe to listen on |
| `--log-dir <dir>` | `<appdir>/logs` | Where `broker.log` and `service.log` are written |
| `--log-level <level>` | `$NVPAIR_LOG_LEVEL` or `info` | The service's own level. When given explicitly it is also passed to the broker as `--log-level`, unless the broker arguments set one |
| `--version` | | Print version and exit |
| `-- <args…>` | | Passed to the broker verbatim, for example the same arguments the desktop app gives it |

The broker is started from its own directory with that directory as its
working directory, which is where it resolves its sibling workers. It gets its
own process group, so Ctrl+C in a terminal running the service reaches only the
service, which then stops the broker in order. `SIGINT` and `SIGTERM` stop the
service the same way `service/stop` does; `SIGHUP` is ignored. The broker
inherits the service's whole environment, including the
`NVPAIR_PROVIDER_SECRET_` namespace, because it passes the configured credential
references on to the proxy.

### Autostart

```sh
nvpair-service autostart enable [--systemd] [service flags] [-- broker args]
nvpair-service autostart status
nvpair-service autostart disable
```

Off unless enabled. `enable` records this executable's resolved path and the
arguments after `enable`, and takes effect at the next login:

| Platform | Entry |
|---|---|
| Windows | `HKCU\Software\Microsoft\Windows\CurrentVersion\Run\NVPAIRService` |
| macOS | `~/Library/LaunchAgents/com.nvidia.pair.service.plist` (`RunAtLoad`, no `KeepAlive`, so `service/stop` sticks until the next login) |
| Linux | `$XDG_CONFIG_HOME/autostart/nvpair-service.desktop` (`~/.config` when unset) |

On Windows, the Run value invokes this executable's internal `autostart run`
command with the recorded service flags and broker arguments. The starter
validates those arguments and starts the ordinary service command from its own
executable directory through the same shared detached launcher as clients,
with stdio on the null device and
`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`. The daemon has
no console and outlives the starter. A console window may still flash briefly
while Windows starts the console-subsystem starter, before its Go code runs.

The starter returns once the child process is created; successful return does
not establish service or broker readiness. Clients still wait for the broker's
`app:ready`. Ordinary foreground execution and the `status`, `call` and `stop`
CLI commands retain their console output, exit codes and waiting behavior.
The internal starter is refused on other platforms.

Existing Windows Run entries are not rewritten automatically. After upgrading,
rerun `autostart enable` with the same service flags and broker arguments to
refresh an existing entry to use the starter.
`autostart status` prints the recorded command; a refreshed Windows entry
includes `autostart run` immediately after the executable path. Before the
next login after downgrading to a binary without the internal starter, rerun
`autostart enable` with that older binary and the same flags and broker
arguments to replace the entry it cannot run.

The detached child's startup errors are not printed in the starter's terminal.
For diagnosis, run `nvpair-service` directly in a terminal with the same service
flags and broker arguments, then inspect `service.log` and `broker.log` in the
configured log directory. A missing service or log directory can mean startup
failed before logging was initialized.

The Linux entry follows the XDG autostart specification, so it runs when a
desktop session starts. A headless machine has no desktop session to process
it; use `--systemd` there (Linux only, an error elsewhere):

```sh
nvpair-service autostart enable --systemd [service flags] [-- broker args]
```

This writes the systemd user unit
`$XDG_CONFIG_HOME/systemd/user/nvpair-service.service` (`~/.config` when
unset) with `ExecStart` set to the absolute binary path plus the same flags and
broker arguments, `Restart=on-failure`, and `WantedBy=default.target`. It then
runs `systemctl --user daemon-reload` and
`systemctl --user enable nvpair-service.service`. If `systemctl` is missing or
fails, the unit file is kept and the exact commands are printed to run by hand.
Start it now with `systemctl --user start nvpair-service.service`. A user unit
starts at login unless lingering is on; to start at boot with no login, run
`loginctl enable-linger $USER` once.

`disable` removes whichever of the XDG entry and the unit exist (running
`systemctl --user disable` best-effort), and `status` reports both. `systemctl`
itself runs without the `NVPAIR_PROVIDER_SECRET_` namespace.

An autostarted service receives the login session's environment, not one PAIR
builds. That environment is the place to configure provider credentials; GUI
and terminal clients can then attach with a clean environment.

## Broker supervision

- The broker is spawned when the service starts. If it exits without having
  been asked to, it is restarted after a backoff of 1 s, doubling to at most
  30 s; a broker that stayed up for a minute starts the backoff over.
- Requests still waiting on a broker that exits are answered with error
  `-32000` (`broker exited before answering <method>`), so no client waits
  forever. While no broker is running, forwarded requests fail the same way
  (`broker is not running`).
- Once the new broker is running, every client gets
  `service/broker-restarted`. When it sends `app:ready`, the service re-sends
  every `*:subscribe` the previous broker had acknowledged, so client streams
  resume without each client having to notice the restart.

## Multiplexing

- **Requests.** Each client request's `id` is replaced by a service-unique
  numeric id before it reaches the broker. The response goes back to the client
  that asked, under its original id (numbers, strings, anything JSON).
- **Notifications from the broker** are broadcast to every attached client.
- **`app:ready`** is cached and replayed as the first frame to a client that
  attaches later. A broker restart clears the cache until the new broker sends
  its own.
- **`*:subscribe`** is forwarded every time; the broker treats a repeat as
  idempotent. Because the broker pushes a stream's baseline only on a fresh
  subscription, the service caches the latest `discovery:nodes-changed` and
  `<prefix>:ready` (an engine proxy's readiness) and replays them to a client
  whose subscription the broker considered redundant, right after its
  acknowledgement.
- **`*:unsubscribe`** is answered locally with `{"subscribed": false}` and never
  forwarded: another client may still depend on the stream. The broker keeps
  sending it; the client ignores what it no longer wants.
- **`shutdown`** from a client is answered locally with `null`, and that
  client is then detached. The broker is not stopped. The notification form
  detaches the client too and is never forwarded.
- **Notifications from a client** (such as `log/set-level` sent without an id)
  are forwarded to the broker unchanged.
- Each client has a bounded write queue. A client that stops reading and falls
  4096 frames behind is disconnected rather than allowed to stall the others or
  silently miss frames; reconnecting gets it a fresh `app:ready`.
- Frames may be up to `jsonrpc.WorkerFrameBytes` (8 MiB) in either direction,
  the same cap as every other hop on the broker's path.

## JSON-RPC methods (client → service)

Everything under `service/` is handled by the service and never reaches the
broker. Any other method is forwarded as described above.

### `service/status`

```json
{"jsonrpc":"2.0","id":1,"method":"service/status"}
```

```json
{"jsonrpc":"2.0","id":1,"result":{"pid":4242,"brokerPid":4243,"clients":2,"startedAt":"2026-10-09T08:00:00Z","version":"0.1.0","brokerRestarts":0}}
```

| Field | Description |
|---|---|
| `pid` | The service's process id |
| `brokerPid` | The running broker's process id, or `0` while none is running |
| `clients` | Attached connections, including the caller |
| `startedAt` | When the service started (RFC 3339, UTC) |
| `version` | The service binary's stamped version |
| `brokerRestarts` | Unexpected broker exits that were followed by a restart |

### `service/stop`

Stops the broker cleanly and then exits. The sequence is the one `nvpair-tui`
used when it owned the broker: send the broker `shutdown`, wait up to 2 s for
its answer, close its stdin, then wait up to 18 s for it to exit before killing
it. The broker tears down its workers in its own order (proxy first, then
engines, then the rest). The answer, `{"stopped": true}`, is sent once the
broker has exited; then every client is disconnected and the endpoint removed.

```json
{"jsonrpc":"2.0","id":2,"method":"service/stop"}
```

A caller should allow at least 25 s for the answer.

### `shutdown` / `*:unsubscribe`

Answered locally; see [Multiplexing](#multiplexing).

## JSON-RPC notifications (service → client)

### `service/broker-restarted`

Sent to every client once a replacement broker has been spawned after an
unexpected exit. Its `app:ready` follows.

```json
{"jsonrpc":"2.0","method":"service/broker-restarted","params":{"brokerPid":4250,"restarts":1}}
```

### `service/log`

One per line of broker stderr, which carries the broker's own log and every
worker's.

```json
{"jsonrpc":"2.0","method":"service/log","params":{"source":"nvpair-ui-broker","stream":"stderr","text":"[nvpair-ui-broker] INFO …"}}
```

A client must redact these before storing or showing them, exactly as it
redacts child stderr when it owns the process. A line longer than 1 MiB is
truncated.

## Logs

- `<appdir>/logs/broker.log` receives every broker stderr line. It rotates at
  10 MiB to `broker.log.1`, keeping one old generation.
- `<appdir>/logs/service.log` is the service's own log, rotated the same way,
  and also written to stderr (the null device when started detached).

Both files are created with mode `0600`.

## Build & test

Built by `services/build.sh` / `build.bat` and the desktop's
`npm run build:modular-binaries`, and staged beside the broker.

```sh
cd nvpair-service
go test ./...
```

The tests drive a real service over a private socket with in-process fake
brokers (multiplexing, id remapping, broadcast, `app:ready` replay, subscription
baselines, unsubscribe and shutdown handling, broker restart, clean and forced
stop, single instance, stale sockets, slow clients, and log rotation), plus one
test that runs the real exec-based spawner against the test binary acting as a
broker. The `diagnose` tests drive the command through an injected caller and
through a scripted listener on a private endpoint in the test process; they
start no service, broker or other process.

### Real-service lifecycle and policy integration

`services/tests` also runs the service, broker and scanner as separate real
processes against each other. From the repository root:

```sh
cd services/tests
go test -count=1 -timeout 10m -run '^TestService(Lifecycle|Policy)' -v .
go test -count=3 -timeout 10m -run '^TestService(Lifecycle|Policy)' -v .
go test -count=1 -timeout 10m -run '^(TestServiceLifecycle|TestServicePolicy|TestServiceInspectionError|TestServiceRecovery|TestServiceRPCReader)' -v .
```

`-count=1` disables the test cache; `-count=3` repeats every named case in one
test process to check for state leaks. The third command adds the
inspection-error, process-recovery and RPC-reader regressions to the lifecycle
selection. A skipped case leaves that behavior unverified. The lifecycle cases
assert, against real processes:

- Two clients call `policy:get` under the same request id and each receives its
  own reply.
- One client detaching leaves the other functional.
- All clients detaching and a later reconnect preserve the service, broker and
  scanner identities.
- `service/stop` replies `{"stopped": true}` and the owned service exits.
- A second service launched at the same endpoint exits without replacing the
  first.
- CLI `status` prints the structured `service/status` snapshot, and after the
  service stops `status` exits 3.
- CLI `stop` allows the full broker shutdown grace.
- Cleanup started before readiness is awaited, and cleanup after the owned
  service is killed, each leave no owned process and a rejected endpoint.
- A failed process scan or a failed empty enumeration retains its error, does
  not report cleanup as verified, and still stops every recorded owned
  descendant while leaving the bundle in place.
- A malformed RPC frame, a reader that cannot close, or a reader that exceeds
  its notification retention budget makes cleanup fail visibly and preserves
  the bundle until the injected failure is disposed of.

The policy cases each use fresh fixture state and complete policies fetched
through `policy:get`. They assert that:

- A valid concurrency update returns the complete policy, notifies an observer
  with the fixture's node identity, writes the platform-resolved disposable
  `appdir` policy file, and survives a service and broker restart.
- Invalid `tierPolicy.preferLoaded` input returns a structured JSON-RPC error
  and preserves both the complete policy/live snapshot and exact stored bytes.
- Pausing returns and publishes the final paused state to another connection,
  persists it across restart, and resuming returns, publishes and stores the
  available state. The observer waits for the final state while retaining the
  intermediate draining event.
- A `policy:set` based on a revision another client has since replaced returns a
  `policy conflict:` error naming the persisted revision and preserves the
  complete policy/live snapshot and exact stored bytes; a set based on the
  reloaded revision then succeeds. A `policy:set` without
  `expectedPersistedRevision` returns `-32602` and changes neither.
- With a copied `nvpair-proxy` fronting ollama (the proxy variant of the
  fixture; the Ollama facade picks a free port), `policy:get` and the `policy:set`
  result report `activation` `active` with `activeRevision` equal to the
  persisted `policy.revision` after a set, a pause and a resume.
- CLI `call policy:get '{}'` transports the complete updated policy through
  the fixture's private endpoint.
- Detaching the command client leaves the updated policy readable on the
  surviving connection without replacing the service or broker.

The inspection-error, process-recovery and RPC-reader regressions are
deterministic and in-memory: they drive the fixture's cleanup inspection,
recovery and frame-reader helpers with injected failures and in-memory
transports, so they assert which errors are retained and which identities are
recovered without launching a process or opening an endpoint. The lifecycle
cases instead run real copied processes, so the verified process absence and
retained bundles they check come from the fixture's own process inventory and
endpoint. The proxy variant of the fixture also copies `nvpair-proxy` into the
bundle and passes `--proxy-path <copied proxy> --proxy-engines ollama` instead of
`--proxy-engines=`.

Each lifecycle case copies only `nvpair-service`, `nvpair-ui-broker` and
`nvpair-node-scanner` into a fresh temporary bundle, then launches the copied
service with the copied broker, an endpoint private to the bundle
(`NVPAIR_SERVICE_ENDPOINT`; a Unix socket in the bundle, or a unique
`\\.\pipe\nvpair-test-<id>` on Windows), disposable
`HOME`/`XDG_CONFIG_HOME`/`APPDATA`/`LOCALAPPDATA`, and broker arguments
`--scanner-path <copied scanner> --cluster-dir <empty directory> --proxy-engines=`. The scanner is the only explicitly supplied worker and is
required; the other optional workers are absent from the bundle, and their
absence is non-fatal. No engine management or inference listener is present, so
the fixture operates no engine, model, autostart registration or pairing.
Readiness is a broker `app:ready` received over the endpoint. Cleanup identifies owned
processes by exact copied executable path through
`github.com/shirou/gopsutil/v4/process`, revalidates PID, path and creation time
before any fallback kill, and retains the bundle whenever cleanup could not be
verified.

The scanner still performs real network discovery, so run the lifecycle
commands on a controlled network and never assert which peers it finds.

The lifecycle and policy cases prove JSON-RPC multiplexing, process lifetime,
policy persistence, and availability control flow. Availability is exercised
without a proxy or engines; it does not prove destination rejection, live
draining, unloading, or Off/wake behavior. Those require the existing proxy
and broker suites and real-engine validation. These integration cases
do not exercise real inference, an engine or a model, GPU scheduling, the
desktop or TUI attach paths, login startup, paired or multi-machine routing, or
memory use, and the fixture runs without a proxy. Run them natively on macOS
and Windows; a cross-compiled binary does not establish named-pipe or
process-lifetime behavior.
