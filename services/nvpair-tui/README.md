<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-tui

A terminal UI for running and supervising the NVPAIR fleet on a **headless
machine over SSH**, where the bundled graphical UI cannot run. That is its
purpose: it is an operations tool for hosts without a desktop, not a replacement
for the graphical UI, and it does not cover every operation the desktop does.

It attaches to [`nvpair-service`](../nvpair-service/README.md), the per-user
process that owns `nvpair-ui-broker`, and starts that service detached when it is
not running; the broker in turn supervises the worker subprocesses. Quitting the
TUI detaches and leaves the service, and the inference it serves, running.
Stopping the service is an explicit action (`Q`, or `--stop-service`).

This file is the component reference. For task-oriented usage instructions, see
[Using the PAIR terminal interface](../../docs/terminal-interface.mdx).

## What it does

On Nodes, `p` invites a standalone target or offers **Ask to join** for a row
in another cluster. `n` accepts an address and uses `up` / `down` to choose
Invite or Ask to join (initially Invite). Ask to join requires a supported local
service and peer, authoritative standalone eligibility, and explicit consent
before replacing a solo cluster; retained peer relationships cannot move.

Admission requests hydrate through `cluster:get-join-requests` on attach and
reconnect and reconcile `cluster:join-requests-changed` snapshots. Incoming
requests remain passive until `a` reviews the displayed request or `d` declines
it. Approval chooses remote management of the admitting device (Block by
default); the member retrieves the exact child invite's PIN locally and reads
it to the requester. The requester enters that PIN through ordinary pairing
with its own independent management choice. Request IDs never substitute for
invite IDs. `c` cancels an outgoing request. Expiry, refusal and lost status
come from cluster-manager; lost status requires checking membership before
retrying. Unsupported services retain normal Invite without reversing it into
an invite to a clustered device. No new worker, transport or firewall rule is
started by this flow. See the terminal guide for the complete ceremony.

`nvpair-tui` is a JSON-RPC 2.0 client of `nvpair-ui-broker`, reached through
`nvpair-service` (newline-delimited JSON over the service's Unix socket or named
pipe, which the service multiplexes onto the broker's stdio). It consumes the
broker's notification stream and renders a tabbed, keyboard-driven dashboard
built with [Bubble Tea](https://github.com/charmbracelet/bubbletea).

The tab set is machine-first: a node is the unit an operator reasons about, so
everything specific to one machine hangs off its row rather than living in a tab
of its own.

| Tab | Purpose |
| --- | --- |
| **Nodes** | Every machine PAIR knows about — discovered, added by hand, or paired into the cluster — merged into one table. Reachability (`STATUS`) and membership (`CLUSTER`) are separate columns because they are independent facts. `enter` opens the node's detail screen; `p` pairs, `n` pairs by address, `f` finds by address, `c` cancels a pairing request you sent, `r` removes — un-pairing a member asks you to confirm, dropping a hand-added entry does not — `a`/`d` answer an inbound pairing request, `l` leaves the cluster (with a confirmation), `/` filters by name or address. The keys follow the words on screen: everything here is "pair", so `p` starts one and `a` accepts one. A filtered table says so, and the cluster summary still counts every node rather than the visible ones. |
| **Jobs** | Inference work across the cluster (`workloads:get-initial` plus the live `workloads:upsert` / `workloads:remove` stream), headed by the proxy endpoints local clients connect to. `FROM` and `RAN ON` are the job's `originatedFrom` and `scheduledOn` nodes. `a` toggles finished work. `t` starts or stops the Inference Demo — a sixty-second burst of synthetic traffic through those endpoints, which is why it lives here rather than with the service controls: the ports it needs are already on this tab and the jobs it produces land in the table below. |
| **Service** | Broker version and uptime (`ping`), a row per supervised worker derived from `supervisor:subprocess-crashed:*` errors, the cluster name, the fleet log level (a picker over the four `applog` levels), a confirmed stop of the service (the same as `Q`), and a confirmed data reset. The reset uninstalls the engines PAIR installed, stops the service, and deletes the data directory — engine files live outside that directory when a vendor installer chose their location — and keeps every engine's downloaded models. Ports are not here — they live on the node detail screen beside the engine each one serves. `force-ports` and `cluster-auto-sync` are persisted by `nvpair-node-settings` but not offered: nothing currently acts on either. |
| **Errors** | The service-error datastore (`errors:get-initial` plus live `errors:update`); `c` clears the selected entry. Only entries this node reported are clearable: `errors:clear` is delete-by-id on the receiving node and cross-node propagation is unbuilt (`shared/errors` stamps `ClearedBy` for it and ignores it), so clearing a peer's entry is reverted by the next sync. The broker acknowledges the relay rather than the outcome, so the reply cannot be used to detect it — the key is withdrawn for a peer's entry instead, naming the node to clear it from. Node ids are resolved to names, and a line under the table carries the selected entry's engine, operation, model, and suggested action. |
| **Logs** | The broker's and workers' stderr, relayed by the service as `service/log` (lines from before this session attached are in `<appdir>/logs/broker.log`), with a substring filter (`/`), a follow toggle (`t`, for tail — `f` belongs to the viewport's paging), and save-to-file (`s`). |

Diagnostics come last, errors before logs, which is the order you consult them
in. The **Errors** tab carries its active count in its own label (`Errors (2)`),
so the tab bar is the indicator and nothing extra has to be learned to notice a
problem from another tab.

Errors were briefly an overlay on a dedicated key instead. That needed a global
binding, and every candidate was either a letter that shadowed a view's own verb
or a digit that looked like a tab number without being one — so it became the
tab it was already pretending to be.

### Update notice

A newer published release is announced in a row under the tab bar, checked
shortly after startup and every six hours against the public releases feed.

It belongs to the shell rather than to the **Service** tab: the operator this is
for is the one who lives on **Nodes** or **Jobs** and has no reason to open
**Service**. `ctrl+x` dismisses it on every tab at once, and a release newer
than the dismissed one brings it back — that version was never acknowledged.
The key is `ctrl+x` because the views between them bind `a` through `y` and the
table and viewport add the paging keys; the shell handles its own bindings
before the active view sees them, so a global letter would silently shadow a
verb.

Two layout constraints, both regression-tested. The row comes out of
`contentHeight()`, or it is a row the shell then deletes from the bottom of
whichever view is showing — which is where every view keeps its messages. And
the line is assembled longest-first against the real width, dropping the URL and
then the version detail, because the frame is clamped to the terminal and the
rightmost text is the dismiss hint: the only key that closes it.

It compares `ui.ReleaseVersion`, stamped by both build paths from
`desktop/package.json`, against the feed's latest stable tag. That is the
release number users install and the one the tags are named for — this
component's own version and the services suite version describe parts of the
build and mean nothing to the comparison. Drafts and prereleases are ignored.

Awareness only: nothing is downloaded or installed, because this client resolves
the service beside its own executable, the service runs the broker from the same
directory, and that broker spawns the worker set from it too — replacing "the client" means swapping every binary
atomically while they serve inference, and a partial swap leaves a new client
driving old workers across a JSON-RPC contract that may have changed. Silent on
failure, skipped for an unstamped build, and disabled by
`NVPAIR_NO_UPDATE_CHECK`. A desktop-app install needs none of this: `nvpair-tui`
ships in `cli-bin` and the app's updater replaces it.

### Node detail

Pairing from **Nodes**, by discovered node or by address, and accepting an
inbound PIN both offer **Remote management of this device**. `tab` switches
allow/block, with block selected initially. This grants the named peer authority
over the device running this TUI; the other device chooses independently.
On Nodes, `g` opens this device's local permission list; up/down selects a peer,
space/enter toggles permission, `r` reloads and `esc` returns. Saves use the
reported revision and authoritative readback, including after errors; they are
not automatically retried. `cluster_auto_sync` remains an inert stored
preference and grants no authority. An engine lock and a node pause on the target
still apply; revocation prevents new acceptance while already accepted work can
finish or reconcile.

`enter` on a node opens a full-screen drill-down with two panes, switched with
`h`/`l`:

- **Engines** — install (`i`), start (`s`), stop (`x`), and, on this machine
  only, restart (`r`), uninstall (`u`, confirmed with `y`), the engine's own port (`e`, via
  `engine:set-port`), and the client-facing port of the proxy fronting it
  (`p`, via `<prefix>:set-port`). Both ports are shown per engine because they
  are easily confused and were previously configured on different tabs.
- **Models** — the inventory per engine with loaded state, plus browse-and-download
  (`p`), download by name (`n`), load (`enter`), eject (`e`), and delete (`d`,
  confirmed with `y`). Every destructive key arms on the first press and acts
  only on `y`, against the target captured at arm time — these lists re-sort
  under the cursor whenever a download finishes or a peer republishes.

`p` opens a catalog browser over the engine's downloadable models, served by the
backend's `engine:catalog`. Sorting (`o`) is local. So is filtering (`/`) for a
source that cannot search, such as Ollama's list of thousands of entries. For
one whose reply says it is `searchable` — llama.cpp's, a slice of Hugging Face —
`/` sends the query upstream instead, and `c` returns to the browse list.

Both panes work on remote cluster peers through the engine manager's
`engine:remote-*` methods. Restart, uninstall, and the port change need process
ownership on the target host, so they are hidden on a peer rather than offered
and then failed.

Load, eject and delete use the canonical `engine:load-model`,
`engine:unload-model` and `engine:delete-model` methods and their remote
counterparts. The validated `engine:descriptors` projection restricts model
operations to those declared by each engine.

The detail header reports the selected target's live availability and reported
`policy.pausedBy` provenance. Failed reads, disconnection and loss of cluster
admission mark availability unavailable; another target's replies and pushes
cannot refresh it. The local header also retains reported pause provenance.
Engine rows continue to show actual installed, running and healthy state.
A false entry in the current local `enabledByEngine` snapshot adds
` (saved Off, local)` when `offByEngine` reports `local`, or
` (saved Off, by peer)` when it reports `peer`. Without a valid reported origin,
the label remains ` (saved Off)`. A stopped engine alone does not establish saved
Off. These labels appear only on this machine's rows; there is no remote intent
projection. Origin identifies a class relative to the engine-owning device,
not a particular peer, and does not change lifecycle or permission decisions.

Plain Off, an engine lock and a node pause are different facts. Saved Off is
ordinary intent and does not stop a granted peer from starting the engine. A
lock is reported by the engine-owning node as `locked` on its status; when it is
true the row adds ` (locked on this node)`, on this machine and on a paired
target. A target that reports no `locked` field, as an older one does, is shown
without a lock claim and is never read as unlocked; saved Off is never read as a
lock. A locked engine refuses start, restart, model load and being a switch
target before anything is sent, with `This engine is locked on this node. Unlock
it on this node before starting or loading a model.` Stop, unload, delete,
download, settings and reads stay available, and a locked switch source adds no
prohibition on switching to an unlocked engine. The target stays authoritative
against a stale readback. The terminal has no lock or unlock key and exposes no
remote setter; locks are changed on the engine's own device.

Lifecycle, model, settings and availability refusals retain the target's message
and RPC code. `-32012` means remote management is blocked and must be changed
locally on the target. Its permission readback in remote detail is read-only;
unknown, unavailable or unsupported is not a grant. Reads and permitted
inference remain available. Downgraded pre-PG targets cannot enforce PG.
`-32011` means a local veto, including a paused target, which must be resumed on
that node first; `-32013` means the engine is locked on its node; `-32010`
means a revision conflict requiring reload and retry. An unavailable projection
on the target is an ordinary retryable operational error, not a veto. Refusals do not retry or resume work automatically.
Existing long-operation timeouts still report continuing work rather than
success, while quick-operation timeouts remain failures.

A remote node's models come from the discovery snapshot, which the broker
enriches from each peer's engine manager — no extra request. This machine's come
from `engine:models` and stay live through `engine:models-changed`.

#### Engine switching

Node detail offers `w` / **switch engine**. Use `tab` for **From**/**To**,
up/down to select installed engines, `enter` to **Switch**, `r` for status and
permission readback, and `esc` to close the panel. Blocked submission shows its
reason, including a locked target engine. Existing local pause/Off controls remain
reachable. The TUI invokes the
existing E3 `engine:switch` and
`engine:switch-status` methods through the broker, with a peer `nodeId` only for
a peer; it never orchestrates separate stop/start calls.

Readback keeps node/request identity and receipt sequence, reported phase,
typed errors and `recoveryRequired`. Saved intent, runtime observations and
termination evidence are separate; intermediate acknowledgement/readiness is
not completion. Remote permission denial (`-32012`), local veto (`-32011`) and
revision conflict (`-32010`) retain distinct meanings. Failed or unavailable
reads do not grant permission or establish success. After a lost response,
read the original ID rather than resubmitting; readback performs no recovery,
rollback or source restart.

Correlation is visit-scoped. An absent sent request can remain unresolved
indefinitely: the broker does not fence/tombstone an ID it has not accepted.
Closing the panel retains the ID; leaving node detail clears it. Reopening an
idle node can permit a fresh submission, but forgetting the old ID does not
resolve its outcome or prevent late acceptance. Reconnect/restart alone does
not establish abandonment. See [Switch Engine](../../docs/terminal-interface.mdx#switch-engine)
for the operator workflow and recovery limitations.

The detail screen polls GPU, CPU and memory only for the open node, with the
existing two-second interval between reads. Paired remote detail uses
`discovery:get-node-info {hostUuid}` through the existing
local service RPC client; Go owns admitted endpoint selection and the pinned
direct/reverse route. Self and unpaired discovery retain direct HTTP reads;
manual TLS telemetry retains the manual-node worker's metrics. A paired read
needs a resolved physical host identity, without requiring a direct address.

The getter returns a fresh full sample within its 1500 ms backend budget.
RPC, decode, timeout or identity failures mark telemetry unavailable while
preserving membership. Recovery refreshes the values; a failed paired read
does not fall back to unpaired HTTP. Detail generations, current host identity
and self route intent reject late results after a screen or route transition.
Client timeout disowns the result locally rather than sending wire cancellation.

## Keys

- `tab` / `shift+tab` or the digits `1`-`5` — switch tabs
- `?` — full help
- `ctrl+x` — dismiss the update notice, while one is showing
- `q` / `ctrl+c` — quit; the service, the broker, and inference keep running
- `P` — pause or resume this node (`node:set-availability`, run in the
  background). The header shows `Available`, `Pausing…`, `Paused`, or
  `Pause incomplete — P retries`, read from `policy:get` at startup and kept
  current by `node:availability-changed`. A pause that fails after the node began
  draining shows `Pause incomplete — P retries; resume from this device's desktop.`
  with the error; `P` retries the pause, and resuming that state is a desktop or
  local RPC action, not a terminal key. It is
  the shifted key because lower-case `p` is "pair" / "proxy port" in the views
- `Q` — stop the service and quit, after a `y` confirmation shown in the notice
  row. The service stops the broker cleanly (the broker tears its workers down
  proxy first, then engines), so inference on this machine stops for every
  client
- Per-tab keys appear in the footer. While editing a field (port, PIN, address,
  model name) every key goes to the field until `enter` or `esc`.

`h` / `l` and the arrows are deliberately **not** bound to tab switching: they
move within content, and the node detail screen needs them for its panes.

## Running

`nvpair-tui` attaches to the running `nvpair-service`. When none is running it
starts `nvpair-service` from next to its own executable (the installed `bin/`
layout), detached, and waits up to ten seconds for it to answer. Override the
binary with `--service-path`:

```sh
nvpair-tui                                   # attach, starting the service if needed
nvpair-tui --service-path /opt/nvpair/bin/nvpair-service
nvpair-tui --stop-service                    # stop the running service and exit
nvpair-tui --log-level debug                 # own logging (to stderr)
nvpair-tui --appearance light                # if the colours come out wrong
nvpair-tui --version
```

Logging goes to stderr (the broker's logs are shown inside the **Logs**
tab, not on the terminal), so it never corrupts the full-screen UI.

The RPC session retains one client handle after an established transport is
lost. It fails pending requests, rejects requests while disconnected and retries
attachment to the same endpoint with cancellable delays capped at two seconds.
These retries only dial; they never start a replacement service. Requests are
never replayed. Detach and an explicit service stop suppress further attachment.
The log stream and its service/log filter survive transport replacement.

Transport events carry a monotonically increasing generation in wire order,
including disconnects. `CallGeneration` rejects commands captured for an old
transport, and `IsCurrent` lets consumers reject obsolete asynchronous results.
The shell consumes that event stream and marks the service disconnected on loss.
Once a replacement reports ready, existing views resubscribe and reread their
baselines once for that transport. Old rows, permissions and snapshot precedence
are cleared; unavailable readings remain unknown. Local filters, presentation
choices and unsent text survive, while pairing sessions and destructive
confirmations are retired. Commands capture their transport when created, and
the shell rejects obsolete results even if an old RPC completed before recovery.
Performance cleanup cannot unsubscribe a replacement transport. UI timers retain
one chain per owner rather than restarting with every ready notification.

A service the TUI starts gets no broker arguments: the broker resolves its
workers beside itself and inherits the environment the service was started
with, including `NVPAIR_LOG_LEVEL` and any provider credentials in the reserved
`NVPAIR_PROVIDER_SECRET_` namespace. Inference Demo children get neither that
namespace nor `INFERENCE_DISPATCHER_*` variables. To run the service with other broker
arguments, start it yourself first (`nvpair-service -- <broker args>`) or
register it with `nvpair-service autostart enable`.

### Colours

Every colour is a `lipgloss.AdaptiveColor` with a light and a dark variant,
chosen from the terminal's background. Text drawn *on* one of those colours has
to adapt with it: a fixed foreground over an adaptive background is legible in
one terminal and not the other, which is how the selected table row came to be
black on dark blue for anyone using a light theme. `TestTextOnAnAdaptiveBackgroundAdaptsToo`
pins the pairing.

Detection asks the terminal for its background and reads the reply from stdin.
lipgloss does that lazily, the first time an adaptive colour resolves, which is
during the first render — after Bubble Tea has taken the terminal and started
its own reader, so the answer goes to that reader and the query learns nothing.
It is therefore forced at startup instead, while stdin is still ours, and the
result cached behind lipgloss's `sync.Once`.

That query costs nothing on a terminal that answers and five seconds on one
that does not, since termenv's timeout is a constant. It runs alongside
attaching to the service for that reason, and is joined immediately before the first render. It
cannot be abandoned early: the query owns the terminal until it returns.

termenv declines to ask at all under `screen`, `tmux`, or `TERM=dumb`, which can
be attached to several terminals at once. Those fall back to assuming dark, and
`--appearance light|dark` is the answer.

## Architecture

```
nvpair-tui (this process)
├── servicelink.go     attach to (or start) nvpair-service, relay service/log, stop it
├── rpc/               JSON-RPC 2.0 codec + id-matching client
└── ui/                Bubble Tea root model, one file per tab, plus:
    ├── table.go       shared column layout (accounts for bubbles' cell padding)
    ├── toast.go       transient status lines that expire on their own
    ├── nodesmodel.go  merges the discovery, cluster, and manual feeds
    └── nodedetail.go  the per-node engines + models drill-down
        │ Unix socket / named pipe (newline-delimited JSON-RPC 2.0)
        ▼
   nvpair-service ──stdio──► nvpair-ui-broker ──► nvpair-node-scanner, nvpair-proxy, ... (workers)
```

Quitting closes the connection, which the service treats as a detach. `Q` and
`--stop-service` send `service/stop` instead: the service sends the broker
`shutdown`, closes its stdin, and waits for it to tear its own workers down
before exiting, so stopping leaves no orphans. The data reset stops the service
the same way and deletes the data directory only once the service has gone.

## Build & test

Built by the repo's top-level `build.bat` / `build.sh` (stamped via
`-X main.Version` from `versions.json`) and staged in `build/bin/` alongside the
other binaries. Standalone:

```sh
cd nvpair-tui
go build ./...
go test ./...
```
