<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA Personal AI Router — backend services

These are the Go services behind PAIR. They **discover other PAIR nodes** on the
local network: each node advertises itself over mDNS as one consolidated
`_nvpair-node._tcp` record, and peers learn from that record which services a
node offers and where to reach them.

What a discovered node can actually serve is a separate question, answered after
discovery. A node may be running [Ollama](https://ollama.com/), LM Studio,
llama.cpp, any combination, or none, and its model inventory is fetched over
HTTP from its engine-manager rather than crammed into mDNS TXT records, which
are too small to carry it.

Locally, each node exposes compatibility proxies — Ollama-compatible and
OpenAI-compatible — so an unmodified client on that machine can reach any capable
node in the cluster. Routing requires a compatible engine and the requested
model. Explicit [model engine bindings](../docs/model-engine-bindings.mdx) also
allow bounded preparation of a stopped engine before fresh inventory checks and
ordinary admission. Ready equivalent candidates lead cold offers; offers do not
establish inventory, engine health or strict readiness.

Standalone, the Ollama proxy defaults to `http://localhost:11435`; under the
broker it serves the Ollama-compatible `http://localhost:11434` when that port is
free. If an inherited `OLLAMA_HOST` names a different local HTTP port, the broker
can serve that loopback-only address from the same router when it is free —
`localhost` is claimed on IPv4 and IPv6 together, and remote or HTTPS targets are
never intercepted.

The llama.cpp facade is enabled by default alongside Ollama and LM Studio. It
exposes OpenAI-compatible traffic on `http://localhost:8080` while the managed
router runs on `8081`; a safe fallback is reported when the facade port is
occupied. The desktop and TUI expose its managed lifecycle, downloads,
inventory, and load/unload actions. `--proxy-engines` can still restrict the
facades a standalone broker or TUI starts.

PAIR ships a graphical UI alongside these services. The UI attaches to
**`nvpair-service`**, starting it when needed; the service owns the broker, which
orchestrates the workers and exposes a newline-delimited JSON-RPC 2.0 API over
stdio (or a Unix socket / Windows named pipe via `--ipc`). For supported platforms
and engines, see the [root README](../README.md#what-is-supported).

## Architecture

The Go binaries below share one process owner chain. `nvpair-ui-broker`
supervises the workers, all spawned at startup; only the scanner is required,
and a missing optional binary leaves the broker running without that capability.
`nvpair-proxy` is one worker hosting a facade per enabled engine.
`nvpair-service` is the per-user owner of the broker. Desktop and terminal
clients attach over its local socket or named pipe and detach without stopping
it. Worker communication uses newline-delimited JSON-RPC 2.0 over stdio or,
optionally, a Unix socket / Windows named pipe.

| Binary | Role |
| --- | --- |
| `nvpair-ui-broker` | Parent service and JSON-RPC API surface used by the bundled UI and other clients. Supervises workers, relays consolidated discovery, and coordinates routing and scheduling. |
| `nvpair-proxy` | The engine HTTP reverse proxy: one process hosting a facade per enabled engine, each enabled over `facade/enable` after spawn. Serves Ollama-native and OpenAI-compatible traffic, selects inventory owners or supported binding offers, and applies fresh admission, failover and scheduler priorities. |
| `nvpair-node-info` | Local HTTP service on `:14318` exposing GPU, CPU, and memory inventory at `/v1/node-info`. |
| `nvpair-node-scanner` | Consolidated discovery daemon. Advertises and browses `_nvpair-node._tcp`, maintains the node directory, and enriches peers with hardware and model information over HTTP. |
| `nvpair-manual-nodes` | Manages user-added nodes that don't appear via mDNS; probes them every 10 s. |
| `nvpair-workload-manager` | Cluster-wide workload relay. |
| `nvpair-errors` | Service-errors datastore + cross-node sync. |
| `nvpair-engine-manager` | Config-driven control plane for local inference engines, model inventory, and trusted remote engine operations. |
| `nvpair-node-settings` | Typed key-value store for per-node preferences. |
| `nvpair-cluster-manager` | Node identity, PIN pairing, and the trusted-node store. |
| `nvpair-job-scheduler` | Responsive scheduler combining total node queue depth across engines with smoothed GPU pressure. |
| `nvpair-service` | Per-user owner of `nvpair-ui-broker`. Restarts it if it dies, multiplexes any number of clients onto its stdio over `<appdir>/service.sock` or a per-user named pipe, and keeps running after they detach. Also registers itself to start at login (`autostart`). |
| `nvpair-tui` | Terminal interface for headless and SSH operation; attaches to `nvpair-service`, starting it when needed. |

Shared code lives in the local `shared/` Go module (imported as `nvpair-shared/…`, replaced via `replace nvpair-shared => ../shared`). It provides logging, wire types, JSON-RPC and IPC, discovery records, mDNS, network monitoring, stable node identity, application data paths, and cluster trust helpers.

The mDNS responder is our own rather than the host's, because Windows ships
none. Its receive and per-interface send sockets bind UDP 5353 as RFC 6762
requires. Socket reuse lets them coexist with sibling PAIR processes and with a
system responder — `avahi-daemon` on Linux or Bonjour where present — without
configuration.

The broker feeds every accepted local or peer workload transition plus compact
GPU telemetry to the scheduler. Queued and running work is counted by destination
node across Ollama, LM Studio, and llama.cpp together. Fresh maximum-GPU
utilization is smoothed into pressure 0–3; missing or stale telemetry is
neutral. Rankings use `pending + gpuPressure`, and each facade adds local
reservations before choosing, so bursts spread without waiting for workload
feedback.

## Repository layout

```
nvpair-ui-broker/        Parent service / JSON-RPC API surface
nvpair-proxy/            Engine routing proxy, one process hosting a facade per engine
nvpair-node-info/        Local GPU-inventory HTTP service
nvpair-node-scanner/     Consolidated _nvpair-node._tcp discovery daemon
nvpair-manual-nodes/     Manual-node manager
nvpair-workload-manager/ Cluster workload relay
nvpair-errors/           Service-errors datastore + sync
nvpair-engine-manager/   Local and remote inference-engine control plane
nvpair-node-settings/    Per-node preferences store
nvpair-cluster-manager/  Node pairing / trust service
nvpair-job-scheduler/    Cluster job scheduler
nvpair-tui/              Terminal interface for headless / SSH operation
nvpair-service/          Per-user service that owns the broker for its clients
shared/                   Shared Go module (nvpair-shared/…)
eap-noob/                 EAP-NOOB implementation used by cluster pairing
tests/                    Cross-process integration tests (separate go.mod)
versions.json             Single source of truth for every component version
build.bat                 Builds all thirteen binaries (Windows)
build.sh                  Builds all thirteen binaries (Linux)
VERSIONING.md             SemVer rules and version-bump workflow
```

Every directory above is its own Go module and holds its own tests beside the
source it covers. See [Testing](#testing).

## Requirements

- [Go](https://go.dev/dl/) 1.25 or newer.
- [`jq`](https://jqlang.org/) on `PATH` — the build scripts use it to parse `versions.json`. Install via `winget install jqlang.jq` / `choco install jq` / `scoop install jq` on Windows, `sudo apt install jq` on Debian/Ubuntu, `sudo dnf install jq` on Fedora/RHEL, or `brew install jq` on macOS.
- No C toolchain, GUI, or webkit dependencies. Every module is pure Go (`cgo` is not used).

## Building

Always use the platform-appropriate build script, run from this `services/`
directory.

On Windows:

```powershell
.\build.bat
```

On Linux and macOS:

```bash
./build.sh
```

Both scripts read `versions.json`, build all thirteen Go binaries with `-X main.Version=…` ldflags, and stage them together in `services/build/bin/`.

Do **not** build individual components by hand without also copying their binaries into `build/bin/`: the broker will silently keep using the older binary there.

The staged binaries in `build/bin/` are what you run locally. Installable builds
come from the
[releases page](https://github.com/NVIDIA/Personal-AI-Router/releases).

## Running

Normal installations launch the bundled UI, which starts the broker from the same
installation directory.

To drive a services-only build interactively, run `nvpair-tui` — it attaches to
`nvpair-service`, starting it (and through it the broker) when it is not
running, so nothing else needs to be running. Quitting the TUI leaves the
service running; `nvpair-service stop` or `nvpair-tui --stop-service` stops it. See
[Using the PAIR terminal interface](../docs/terminal-interface.mdx).

For backend development or direct API access, launch the broker yourself; it
supervises workers that it expects as siblings in the same `build/bin/`
directory:

```powershell
.\build\bin\nvpair-ui-broker.exe                                              # Windows
```

```bash
./build/bin/nvpair-ui-broker                                                  # Linux
```

The bundled UI connects to the broker over its JSON-RPC API (stdio or `--ipc`).
Other clients can use the same contract. See `nvpair-ui-broker/README.md` for
the API surface.

## Debugging

Every binary respects two inputs for its starting log level, resolved before any other work happens. Precedence: `--log-level` flag > `NVPAIR_LOG_LEVEL` env var > default (`info`). The broker passes its own resolved level to every subprocess it spawns (`--log-level <level>`), so setting either before launch turns on debug output from the very first line of all processes.

```powershell
# Windows / PowerShell
$env:NVPAIR_LOG_LEVEL = "debug"
& ".\build\bin\nvpair-ui-broker.exe"
& ".\build\bin\nvpair-ui-broker.exe" --log-level debug
```

```bash
# Linux
NVPAIR_LOG_LEVEL=debug ./build/bin/nvpair-ui-broker
./build/bin/nvpair-ui-broker --log-level debug
```

Accepted values: `debug`, `info`, `warn`, `error`. The level can also be changed live over the broker's `log/set-level` JSON-RPC method (not persisted across restarts), so for launch-time issues use the env var or the flag.

## Testing

There are two layers, and while you are editing a component you want the first
one.

### Per-component tests

Every component keeps its tests beside its source, and each is its own Go module,
so run them from that component's directory:

```bash
cd nvpair-engine-manager
go test ./...
```

These are fast and are what you should run on each edit. Substitute any component
directory — `shared/` and `eap-noob/` have their own tests too:

```bash
cd shared
go test ./...
```

**Every one of the thirteen binaries has tests**, as do `shared/` and
`eap-noob/`. Depth varies with how much behaviour a component carries:
`nvpair-engine-manager` and `nvpair-cluster-manager` have the largest suites,
while a component with one test file may still hold twenty test functions in it.
File count is a poor proxy for coverage, so read the suite rather than counting
files.

### Windows fixed test paths

Windows Firewall rules match exact executable paths. To keep plain `go test`
off changing `go-build` paths, set `NVPAIR_TEST_BIN_DIR` to an absolute directory
in your current shell, then run `scripts/windows-test-bin.ps1 install` from the
repository root. This non-elevated command builds `bin/nvpair-test-exec.exe`
beneath that directory and prints the user environment settings to apply:
`NVPAIR_TEST_BIN_DIR` and `GOFLAGS` with an appended `-exec=<wrapper>` flag.
It does not apply those settings. Preserve the printed quoting for paths with
spaces, and open a new shell after applying them.

With these settings, ordinary `go test` in any checkout runs package binaries at
`<root>\go-test\<module>\<module>[-<package-path-with-dashes>].test.exe`.
`go run` passes through unchanged; `go build` and `go vet` ignore `-exec`.
The wrapper passes through non-test executables, packages outside a service
module, and an unset or relative root. On other platforms it uses `.test`
and only stages binaries when an absolute root is set.

Use `scripts/windows-test-bin.ps1 list` to inventory paths and `rules` from an
elevated shell to create the corresponding inbound block rules. Rerun `rules`
after adding packages or testbin manifest entries. The existing `run` command
still compiles one package with `go test -c` and runs its fixed binary directly.
Keep machine-wide test invocations serialized: fixture programs are shared too,
and `run` does not acquire the wrapper lock.

The wrapper locks each package destination for up to 30 minutes, including the
child's lifetime. After a crash, remove a leftover `.lock` file only after
checking no process uses that destination. Failed replacement retries for five
seconds before reporting an earlier process may still be running. On Windows
a kill-on-close job object cleans up the child and assigned descendants when
the wrapper exits; the child starts suspended until assigned to the job.
On other platforms cancellation kills the direct child only.

### Cross-process tests

`tests/` has its own `go.mod` and exercises real binaries talking to each other.
Its `TestMain` builds the workers it needs into a temp directory first, so you do
not have to build beforehand:

```bash
cd tests
go test ./...
```

Expect this to take a few minutes. Run it before opening a pull request, and
after any change to a JSON-RPC method, payload, or notification.

Fixture broker launches, including custom service/broker argument lists, must
pass the complete `fixtureServicePorts` plan. `fixtureBrokerCommand` and
`fixtureServiceCommand` reject missing, zero, repeated or installed-default
service ports before constructing a command. The service plan covers listener
arguments and registrations; remote workload destinations retain their peer's
explicit advertised `wl` port. Manual probe fixtures use private HTTP servers
and startup probe overrides; unused legs are disabled with zero. Ordinary
broker fixtures forward `--scanner-mdns=false` before scanner startup.

Tests that assert real multicast discovery explicitly retain mDNS. The real
scanner and each test's advertisers/browsers share a freshly generated private
service type under the supported `local` domain. A private domain is unsupported
by zeroconf v1.0.0's receive matching. The unchanged OS-hostname SRV/A target
does not admit another service's PTR/SRV/TXT records. Broker/service launch guards
refuse multicast reenablement without a private service; direct scanner launches
use the same check and default to multicast off. This covers broker advertisement, ghost-node
eviction, model enrichment/refresh, mDNS node-loss, workload peer discovery and
rehydration, reverse node-info/control discovery, and standalone multicast
fixtures. The Windows fixture's query, reply and goodbye names use that same
service under `local`. No global per-test registry or replacement discovery
directory is used.

Namespace and listener separation do not establish universal beside-installed
isolation. Configured/authenticated peer endpoints, engine-manager's configured
runtime checks, HTTP redirects and transport proxy settings still require their
own endpoint evidence. Native Windows multicast and beside-installed acceptance
must verify real discovery, identity, updates/removals and absence of ambient
contact; TLS refusal alone is insufficient.

#### Remote configuration acceptance (XA)

`tests/remote_configuration_test.go` is the acceptance suite for remote
configuration between paired nodes: status reads, revision conflicts,
local veto, roles, served and resident model declarations, served-set
enforcement at a proxy facade, idempotent engine switches, and the same
calls through `nvpair-service`. It drives real binaries through public
JSON-RPC methods with fake engines only and never starts an installed engine.
It skips on macOS and, on Windows, requires the fixed-path wrapper above.
On Linux, the cases run unguarded with real binaries. In this fork, run these
process cases only on the designated Windows gate machine, one area at a time:

```bash
cd tests
go test -v -count=1 -run 'TestXARemoteConfigurationStatus$' .
go test -v -count=1 -run 'TestXARemoteConfigurationConflict$' .
go test -v -count=1 -run 'TestXARemoteConfigurationVeto$' .
go test -v -count=1 -run 'TestXARemoteConfigurationRoles$' .
go test -v -count=1 -run 'TestXARemoteConfigurationServing$' .
go test -v -count=1 -run 'TestXARemoteConfigurationSwitch$' .
go test -v -count=1 -run 'TestXARemoteConfigurationService$' .
```

`go test -v -count=1 -run '^TestXARemoteConfiguration(Status|Conflict|Veto|Roles|Serving|Switch|Service)$' .` runs all XA cases. On macOS,
`go vet ./...` and `go test -list 'TestXARemoteConfiguration' .` are the
only checks; the cases themselves do not run there. Network isolation and
protected-port repairs are coordinator-owned prerequisites; existing
fixtures alone do not establish hermetic execution. Automatic resident
execution belongs to SR Task 6b after NE Task 4 and SR 5a–5c; XA consumes
that evidence. Live MSI/PC3 OpenCode evidence remains a separate manual
checklist; passing fake XA cases does not close the daily milestone.

### Tests that skip themselves

Some tests need something the machine may not have, and skip rather than fail:

- Broker fixtures reserve their own ephemeral ports for the six broker-owned
  services and pass them through the broker's port flags, so they do not use
  an installed PAIR's `14318`–`14323`. Tests that still need a product default
  port, such as the default engine facade
  ports (`11434`, `1234`, `8080`), skip when that port is held on IPv4 or IPv6,
  so quit PAIR before running them, or expect gaps.
- Ordinary fixture scanners disable mDNS. Cases explicitly testing multicast
  browse and advertise a private per-test service under `local`.
- Broker advertiser health checks require a current running endpoint from
  engine-manager; an absent manager or failed status RPC does not trigger a
  guessed stock-port probe. Separately, a supported binding offer can keep a real
  stopped-engine facade registered while engine health stays false. That listener
  permits preparation/refusal, not a readiness claim. Configured engine targets
  still need separate review before beside-installed execution.
- Tests needing a real mDNS responder skip when one is unavailable.
- Live engine tests only run when you opt in:

```bash
NVPAIR_LIVE_OLLAMA=1 go test ./...        # drives a real Ollama install
NVPAIR_LIVE_LMSTUDIO=1 go test ./...      # drives a real LM Studio install
```

Those two install, start, stop, and pull against the engines actually on your
machine. Run them deliberately, on a machine you do not mind changing.

A skip is not a pass. If you are relying on a test, check it actually ran.

### Model engine binding proof status

The service path supports persisted bindings, guarded automatic E4 replacement,
stopped-engine offers and bounded HTTP preparation. The desktop editor uses the
existing policy draft and Save/CAS flow. Joined `TestEngineBindingProcess`
acceptance remains pending: focused Task 7 runs are reported green, but the
combined run intermittently refuses reverse pinned target acceptance. See the
[binding guide's pending process procedure](../docs/model-engine-bindings.mdx#client-and-process-proof-status)
for planned Windows selectors and required drain/effect/response, authority,
recovery and cleanup evidence. Its
[native matrix](../docs/model-engine-bindings.mdx#native-acceptance-matrix) is
entirely pending, not observed. Component or fake-engine process results cannot establish
installed-engine or multi-machine acceptance.

### Controller-selected model COPY

`TestModelCopyControllerProcess` in `tests/model_copy_controller_test.go` uses
three private controller/source/destination identities, production broker and
engine-manager handlers, an empty-directory scanner, and tiny synthetic artifacts.
The controller has no model store. The source reader is authenticated as the
destination, and catalog/download egress is refused and counted.

The selector covers custom and canonical layouts, direct/reverse transport,
positive-byte interruption, explicit nonzero-Range retry, cancellation, permission
and pin refusals, source replacement, destination conflicts, failed mutation
acknowledgement and no automatic replay after a lost acknowledgement. Run its
plain and race variants in the machine's serialized fixed-path test lane:

```sh
go test ./... -run '^TestModelCopyControllerProcess$' -count=1 -v
go test -race ./... -run '^TestModelCopyControllerProcess$' -count=1 -v
```

Windows focused plain/race runs at base `44e76c6b` passed on 2026-10-26 with
14 named terminals each and no failures or skips. After review strengthened
the transport-drop oracle, that leaf and its parent passed plain/race again;
module vet exited zero. The combined whole-suite run remains separate.
The fixture checks actual
loopback ec binds; existing cluster/inventory listeners retain their production
binds on private ports and registered images. Its transport counters and absent
controller store do not observe every attempted local artifact filesystem call.
Strict zero-attempt controller IO remains an instrumentation requirement.
Native acceptance separately verifies installed builds, exact source bindings,
live pins/grants, destination hashes/receipts, explicit interruption/retry/cancel,
controller IO instrumentation and UI behavior. COPY does not authorize engine
startup, model loading or inference, or clone source serving configuration.

## Versioning

`services/versions.json` is the single source of truth for every component version and the `services` suite version. The build scripts read it and stamp each binary via `-ldflags "-X main.Version=..."`. Declare the bumps your change needs in the `pair-release-intent:v1` block in your pull-request description — automation writes `versions.json`, so do not edit it by hand — and describe any user-facing change there so it reaches the changelog. [`VERSIONING.md`](VERSIONING.md) has the bump rules.

You can verify a built binary's stamped version at any time:

```powershell
.\build\bin\nvpair-proxy.exe --version                                     # Windows
```

```bash
./build/bin/nvpair-proxy --version                                         # Linux
```

## Further reading

- `VERSIONING.md` — SemVer bump rules and how a release is declared.
- `bom.md` — bill of materials for a given release.
- Per-component `README.md` files in each subdirectory.
- [Architecture](../docs/architecture.mdx) — how these services fit together at
  runtime, including node-to-node transport and discovery.
