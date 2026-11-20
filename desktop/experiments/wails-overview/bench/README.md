<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Overview benchmark tooling

Use a booked host and collect one GUI arm at a time against the same warmed
service and model. Freeze the comparison table, parity checks and primary,
tray and active-use scenes before recording. Keep stock Electron, matched
Electron and Wails separate. Collect service-only idle and warmed inference
controls before the GUI runs. Alternate arm order across at least three timed
pairs per decisive scene. Interleave same-arm runs on the same machine for
noise. Keep a separate trace for short UI interactions.

## Capture

Run from the repository root. Set the example paths, names and patterns to the
packaged host under test and write to the designated evidence directory. Use
the same interval within a comparison. Capture for 340 seconds to leave room
for a five-minute steady window such as `[30000, 330000]` in elapsed milliseconds.

Windows PowerShell 5.1, with English performance counter names:

```powershell
$bench = 'desktop/experiments/wails-overview/bench'
$evidence = 'C:\BenchEvidence\overview'
$gui = 'C:\Path\To\PackagedHost.exe'
$model = '\\runner\.exe(?:[" ]|$)'
& "$bench/sample-windows.ps1" -GuiPath $gui -ModelPattern $model -IntervalSeconds 5 -DurationSeconds 340 |
    Set-Content -Encoding UTF8 "$evidence\visible-01.jsonl"
node "$bench/compare.mjs" --validate-samples "$evidence\visible-01.jsonl"
```

The Windows sampler reads raw performance data with one CIM query per class
and computes rates from consecutive passes, so it has no `Get-Counter`
two-collection delay. On White (Windows 11 10.0.26300, about 340 processes)
one pass took about 0.5 s with the GUI, service and model groups selected, and
a 30-second run at `-IntervalSeconds 1` kept every sample within 22 ms of its
slot. One second is the shortest supported interval. Check slot coverage on
each new host before relying on it.

`--validate-samples` rejects a sample in which a classified process has no
`memoryBytes`: GUI `main`, `renderer` and `gpu`, every service role and the
model `engine`. Only unclassified GUI helpers (`gui/other`) may omit it.
`compare.mjs` applies the same rule to every raw file in a manifest.

Pass `-GuiName @('PAIR', 'AnotherMain')` for image names without `.exe`.
`-HelperPattern` can include verified out-of-tree helpers. On Windows the
helper, renderer, GPU and model expressions match executable path and command
line in memory. The default renderer and GPU expressions recognize Chromium
`--type=` flags. Supply specific expressions for WebView2 when its command
lines and process tree provide ownership evidence.

macOS with Node on `PATH`:

```bash
bench=desktop/experiments/wails-overview/bench
evidence=/absolute/path/to/overview-evidence
gui=/Applications/PackagedHost.app/Contents/MacOS/PackagedHost
bash "$bench/sample-macos.sh" --gui-path "$gui" --renderer ' Helper \(Renderer\)$' \
  --gpu ' Helper \(GPU\)$' --model '/runner$' --interval 5 --duration 340 \
  > "$evidence/visible-01.jsonl"
node "$bench/compare.mjs" --validate-samples "$evidence/visible-01.jsonl"
```

Repeat `--gui-name` for exact basenames or `--gui-path` for exact executable paths.
Use `--gui-pid` for a main process observed at sampler startup. On macOS, pass
each independently verified detached helper as
`--helper '12345:2026-11-20T14:03:01Z:renderer'`. The fields are the PID, the
UTC start time from `TZ=UTC LC_ALL=C ps -p PID -o lstart=` converted to ISO 8601
at whole-second precision, and the role (`renderer`, `gpu` or `other`). Retain
the ownership evidence. If a launchd-parented helper restarts, identify its
replacement before making another decisive capture. The sampler only matches
the declared incarnation. `--helper-pattern` can inspect detached regex
matches during discovery. Any helper admitted by that pattern without other
ownership evidence marks the sample `attributionClear: false`, which invalidates
the steady run.

The macOS helper, renderer, GPU and model expressions match the `ps` executable
name/path field. On either OS, selectors seed groups and the collector follows
their descendants. Use `-ServicePattern` or `--service` to adjust the service
image-name selector. For service-only controls, omit the GUI selectors. The
model selector takes precedence when a model engine belongs to the service
process tree. Role expressions classify selected helpers and leave the
remainder in `gui/other`.

The samplers only observe. They omit paths, command lines, request payloads,
machine names and model names from output. A missed timer slot stays missing;
inspect timestamps and window coverage before treating a run as representative.
A process that exits during a pass is omitted from that sample, because its
counters are missing and its PID could already name another process.
Processes that finish between samples or that already detached before observation
require separate attribution evidence.

## Counters

Each JSONL line is a schema 1 sample with `os`, UTC `timestamp`, monotonic
`elapsedMs`, `processes` and `system`. Each process has a PID, UTC
`createdUtc`, a `group` and a `role`. Unknown or unavailable counters are
`null`. A PID with a new creation time is a new process. Rate fields
(`cpuMsPerSec`, `cpuPercentOneCore`, `gpuEnginePercent` and Windows paging rates)
are `null` in a process's first sample, because they need a previous pass.

| Field                                 | Windows                                                                                                                                                                                                                                                                                                                         | macOS                                                                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `memoryBytes`                         | Raw **Working Set - Private** from `Win32_PerfRawData_Counters_ProcessV2` (Process V2), whose instance name `<image>:<pid>` gives the PID. Without Process V2, `Win32_PerfRawData_PerfProc_Process` and its **ID Process** counter. Counters join a process only when its PID and creation time match before and after the pass | `top -l 1 -stats pid,mem` **MEM**, the process physical footprint, rounded at the precision printed by `top`                      |
| `commitBytes`                         | Raw **Private Bytes** from the same process class, private committed address space                                                                                                                                                                                                                                              | Unknown                                                                                                                           |
| `cpuMsPerSec`                         | Delta of CIM **KernelModeTime + UserModeTime** divided by actual elapsed time                                                                                                                                                                                                                                                   | Delta of cumulative **ps -o time=** divided by actual elapsed time                                                                |
| `cpuPercentOneCore`                   | Delta of raw **% Processor Time** over the delta of its 100 ns timestamp, with 100% equal to one busy core                                                                                                                                                                                                                      | Unknown. Use the cumulative CPU delta                                                                                             |
| `gpuEnginePercent`                    | Sum over the PID's `Win32_PerfRawData_GPUPerformanceCounters_GPUEngine` instances of the raw **Utilization Percentage** delta over its 100 ns timestamp delta. Can exceed 100                                                                                                                                                   | Unknown                                                                                                                           |
| `gpuDedicatedBytes`, `gpuSharedBytes` | Per-PID sums of raw **Dedicated Usage** and **Shared Usage** from `GPUProcessMemory`                                                                                                                                                                                                                                            | Unknown                                                                                                                           |
| System memory and paging              | Raw **Available Bytes**, and **Pages Input/sec** and **Pages Output/sec** as deltas over the performance-counter time, from `Win32_PerfRawData_PerfOS_Memory`, and OS page size                                                                                                                                                 | `vm_stat` page size, cumulative page-ins, page-outs, swap-ins and swap-outs, and `memory_pressure -Q` system-wide free percentage |

Windows private working sets exclude shareable pages. macOS physical footprint
is the kernel's per-task charge, which can include compressed memory and
charged shared allocations. Adding processes' physical footprints adds their
reported task charges without deduplicating shared backing or measuring the
system-wide physical delta. On macOS, `ps -o lstart=` resolves creation time
to one second. GPU counters can include shared allocations or report no
instance for an idle process. If ambiguous GPU or webview ownership could
reverse a result, set `attributionClear` to false.

## Analyze

Use [the synthetic manifest](fixtures/manifest.json) as the schema example.
It contains one pair per OS and produces an inconclusive verdict. The tests
generate a complete synthetic matrix with independent capture timestamps.
Use a separate raw file for each run. The analyzer rejects every occurrence
of a repeated capture identity across pairs and arms. The identity hashes the
normalized parsed observations, with process rows ordered by PID, so a
different filename, UTF-8 BOM, line ending or process row order does not make
a capture independent. The `captureSha256` field retains
the SHA-256 of the original bytes for provenance.
Record before and after fences for each
run using the observed service and broker PID and creation time, redacted
executable/build identities (for example executable hashes), observed broker
restart count and a redacted fingerprint of the effective model settings.
Fence the entire pair as well. Preserve the observations supporting each fence
and the evidence for settings, output parity and usable current data.

Declare `primaryScene`, `trayScene` and `activeScene`. Each run names its
platform, redacted `host`, `scene`, `arm`, `pair`, increasing
capture `order` on that machine, raw `file` relative to the manifest,
inclusive `windowMs`, `intervalMs`, `inputMode` and `requiredGuiRoles`. Set
`inputMode` to `replay` for a fixed GUI input stream and `live` for live GUI
input. A pair needs the same mode and sampling interval in both runs. Use a
separate manifest per mode to obtain separate verdicts.

The sampling schedule starts at elapsed zero. Every scheduled slot inside
`windowMs` needs exactly one sample within 10% of `intervalMs` of its scheduled
time. The window must contain at least two slots. Collect padding before and
after the window so boundary slots can tolerate timer jitter.

Within each run, every steady sample must contain the same process incarnations
(PID and canonical UTC creation time) for GUI `main`, `renderer` and `gpu`, all
service roles and model engines. An appearance, disappearance or replacement
invalidates the run and excludes its pair, with a reason naming the group and
role. `gui/other` helpers may come and go, and their reported counters still
contribute to GUI totals when present.

Two capture caveats follow from this rule:

- Boundary samples count as steady even when timer jitter places them slightly
  outside `windowMs`. Let required processes settle before the earliest
  allowed first sample, and keep them running past the final pass.
  Otherwise a start or exit just outside the literal window still excludes
  the pair.
- `ps` on macOS reports creation times to the second. If a PID is reused
  within the same second, the replacement looks like the same incarnation, so
  this rule cannot detect it. Treat a renderer or helper restart seen in the
  host's own logs or process trees as an invalid run, even when the analyzer
  accepts it.

Use
`kind: "comparison"` for matched Electron versus Wails,
`kind: "control"` for a same-arm noise pair and
`kind: "stock-reference"` for stock versus matched Electron. Pair names are
local to the machine, scene and kind. Set `requiredGroup: "service"` for
service-only idle or `"model"` when a model engine must be present in a
separate scene. Every primary and active-use steady sample requires GUI main,
a local model engine and all declared GUI roles together. The default group
outside those scenes requires GUI main and the declared roles. Declare whether
parity, attribution, usable current data and active-use conditions were
actually observed.

Use redacted request IDs. Record a `measurements.inferenceTimings` entry with
`requestId`, positive integer `attempt` and at least one of `firstTokenMs`
and `tokensPerSecond`. Record its corroborating
`measurements.inferenceRequests` entry with the same ID and attempt,
`success: true`, `singleAttempt: true`, and the sampled local model
`localModel: { pid, createdUtc }`. The analyzer matches process incarnations
by PID and UTC instant at up to seven fractional digits. Equivalent trailing
zeros match across request evidence, samples and fences. Retain the redacted
route, model, node, terminal completion and single-attempt evidence. The
analyzer excludes and counts unkeyed, unmatched, failed, ambiguous and
non-local timings. A declared failed request or a completion percentage below
100 invalidates the run. Keep the other per-observation measurements as
numeric arrays.

```bash
node "$bench/compare.mjs" "$evidence/manifest.json" \
  > "$evidence/summary.json" 2> "$evidence/summary.md"
node --test "$bench/"
```

The analyzer rejects malformed counter rows, missing required helpers,
unverified parity or attribution, and changed fences within or across paired
runs. It records PID reuse separately. It reports the median of each steady
window for group and role totals, system counters and external measurements.
CPU deltas exclude samples without prior identity history, with the valid
count reported. macOS paging rates are derived from adjacent cumulative
observations. Raw counter differences are candidate minus baseline. Decision
deltas are positive when the candidate improves. Relative inference deltas
use that pair's baseline as the denominator.

Noise is the maximum absolute delta among valid, interleaved same-arm controls
from the same scene, machine, fence, input mode and sampling interval. Each
comparison arm needs its own matching controls with measurements for the
metric.

A visible memory win requires every valid pair to favor Wails and a median
benefit above both 100 MiB and observed noise. The alternative requires every
valid primary pair to favor Wails for the same GUI demand metric (CPU ms/s or
GPU engine utilization) and the same outcome. Both median improvements must
exceed their observed noise. The outcome also needs at least 5% higher
tokens/s, 10% lower first-token latency, or 5 percentage points fewer missed
presentations. For decisive CPU demand, every steady slot after the first
needs complete counters for the selected GUI group. GPU demand needs complete
steady-window utilization counters. The report lists the criteria that
passed in `winCriteria`.

Guardrail tolerances are 100 MiB for visible memory, 5 percentage points
for missed frames, 5% for throughput, 10% for first-token latency, 250 ms for
warm reopen, and 50 MiB for tray memory. Each decisive pair must stay within
the tolerance, a fixed ceiling. A breach that the observed noise can explain
makes the result inconclusive; a median loss beyond both the tolerance and
the noise is a regression. The
`withinRegressionTolerance` field is null when measurements or controls are
missing. Mixed visible memory deltas within the limit still allow the GUI
demand and outcome criterion to win. The overall OS verdict requires at
least three valid visible and tray comparisons, plus decisive guardrail and
active-use results. Treat a Windows-only result as pending macOS confirmation.
Review the stock reference separately when attributing gains to a narrower
workload. Keep cold open, completion, attribution, native trace and installer
size evidence with the output.
