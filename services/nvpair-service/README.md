<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-service

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

The broker is unchanged: it still speaks newline-delimited JSON-RPC 2.0 over
stdio to a single parent, which is now this process.

## Endpoint

| Platform | Endpoint | Access |
|---|---|---|
| Linux, macOS | `<appdir>/service.sock` | Unix socket, mode `0600` |
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
nvpair-service stop                         # service/stop: stop the broker cleanly, then exit
nvpair-service --version
```

Clients normally start it themselves: `servicectl.ConnectOrStart` dials the
endpoint and, when nothing answers, starts the service detached — its own
session on Unix (`Setsid`), `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
CREATE_NO_WINDOW` on Windows, stdio on the null device, working directory set to
the binary's directory — and keeps dialing for up to ten seconds.

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
service the same way `service/stop` does; `SIGHUP` is ignored.

### Autostart

```sh
nvpair-service autostart enable [service flags] [-- broker args]
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

The Linux entry follows the XDG autostart specification, so it runs when a
desktop session starts. A headless machine with no desktop session does not
process it; there, start the service from a systemd user unit or the shell
profile instead.

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
broker.
