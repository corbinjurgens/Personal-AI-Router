<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Application overhead sampler

Requires Windows PowerShell 5.1 or newer. Run synthetic verification with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/performance/app-overhead.Tests.ps1
```

PowerShell 7 (`pwsh`) is optional and can run the same test file.

Dot-sourcing `app-overhead.ps1` defines functions and starts no measurement.
`Invoke-AppOverheadSample` requires a separately approved controller, explicit
process authority, an absolute new output file outside the checkout/vault on a
local fixed drive (no UNC/mapped drive or reparse-point ancestors),
source revision, state annotation (`UNKNOWN` when unverified), logical CPU
count, and lease acquisition/release and competition callbacks. Callbacks must
be bounded, metadata-only, and throw on unavailable controls. Acquisition must
either return owning the lease or throw without retaining it. Release must
complete even after failure. No installed trial or GUI transition is authorized
by this tooling. Paired scenario repetitions and attribution remain pending.

Authority records name exact PID, image, CIM creation time, parent PID and role
(`Application`, `RuntimeModel`, or `Observer`). The controller must verify each
root image and parent authority, including the GUI, Electron children, service,
broker, workers and native descendants; the sampler does not infer ownership
from executable names. Use `Get-OverheadProcessSnapshot` for both authority
baseline and subsequent samples: creation timestamps always come from CIM and
use UTC round-trip precision. Never compare them to .NET `Process.StartTime`.
Any disappearance, PID reuse, changed image/parent or added descendant aborts.
Snapshot equality cannot prove absence of transient processes between samples.

JSONL stores the baseline and each completed sample before checking abort,
including failed identity fields, wall/monotonic time, cumulative CPU, working
set, private bytes, handles and observer CPU cost. Only allowlisted metadata is
serialized; controllers must not put secrets into provenance or annotations.
The synchronous schedule has 31 samples at nominal 0..60 seconds, no timer or
background job. Each capture must finish within one second of its nominal slot;
missed slots abort rather than producing catch-up samples. The default CIM query
uses `-OperationTimeoutSec 1 -ErrorAction Stop`. Capture/control callbacks must
themselves be bounded; there is no detached cancellation worker. CIM errors abort
with `CaptureFailedOrTimedOut`; typed timeout exceptions use `CaptureTimeout`.
Aborted
trials are invalid. Abort records name an allowlisted reason/stage and failed
index without exception text. Metrics records use the interval between capture
completion timestamps, independently of the nominal schedule. Capturing process
metadata is not atomic; those timestamps approximate each snapshot's counters.
Lease release and writer disposal run in `finally`.

CPU delta / elapsed is utilization of one core; dividing again by logical CPUs
gives machine utilization. Metrics remain per PID, with separate authority roles
for application, runtime/model and observer. Working sets overlap shared pages
and cannot be summed into physical RAM. Private bytes describe commit rather
than residency; sleeping does not prove that model memory was unmapped.
No command lines, prompts, profiles, RPC, network calls, GUI/lifecycle actions,
process termination, or persistent scheduling are used. Output files are the
only sampler writes. This source implementation supplies no live CPU/RAM result.
