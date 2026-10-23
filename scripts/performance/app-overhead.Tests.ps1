# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. "$PSScriptRoot/app-overhead.ps1"
$script:passed = 0
function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
    $script:passed++
}
function New-SyntheticProcess {
    param([long]$Id = 10, [long]$Parent = 1, [double]$Cpu = 2)
    [pscustomobject]@{ Pid = $Id; ParentPid = $Parent; Image = 'C:\synthetic\pair.exe'
        CreatedUtc = '2026-10-23T00:00:00.0000000Z'; CpuSeconds = $Cpu
        WorkingSetBytes = 100; PrivateBytes = 80; Handles = 5; Role = 'Application'
        CommandLine = 'DO-NOT-RECORD'; Config = 'DO-NOT-RECORD' }
}
function Invoke-SyntheticTrial {
    param([string]$Path, [string]$Mode = 'Stable')
    $state = @{ Tick = 0.0; Captures = 0; Acquired = 0; Released = 0 }
    $capture = {
        $state.Captures++
        if ($Mode -eq 'CaptureFailure' -and $state.Captures -eq 3) { throw 'DO-NOT-RECORD' }
        if ($Mode -eq 'BaselineLatency' -and $state.Captures -eq 1) { $state.Tick += 1 }
        if ($Mode -eq 'FinalOverrun' -and $state.Captures -eq 31) { $state.Tick += 5 }
        $p = New-SyntheticProcess -Cpu (2 + $state.Tick)
        if ($Mode -eq 'Reuse' -and $state.Captures -eq 3) { $p.CreatedUtc = '2026-10-24T00:00:00.0000000Z' }
        $p
    }.GetNewClosure()
    try {
        Invoke-AppOverheadSample -Authority @(New-SyntheticProcess) -OutputPath $Path -SourcePin 'synthetic-revision' `
            -ControllerProvenance 'synthetic-only' -StateAnnotation 'UNKNOWN' -LogicalCpus 8 `
            -AcquireLease ({ $state.Acquired++ }.GetNewClosure()) -ReleaseLease ({ $state.Released++ }.GetNewClosure()) `
            -Capture $capture -Monotonic ({ $state.Tick }.GetNewClosure()) `
            -Wait ({ param($seconds) $state.Tick += $seconds }.GetNewClosure()) `
            -ObserverCpu ({ $state.Captures * 0.01 }.GetNewClosure()) `
            -WallTime { '2026-10-23T00:00:00.0000000Z' } `
            -Competition ({ if ($Mode -eq 'Competition' -and $state.Captures -eq 2) { 'synthetic-build' } }.GetNewClosure())
    } catch {
        if ($Mode -in @('Stable', 'BaselineLatency')) { throw }
    }
    Assert-True ($state.Acquired -eq 1 -and $state.Released -eq 1) "Lease cleanup failed for $Mode"
    # Opening exclusively also proves the writer was disposed on success/error.
    $handle = [IO.File]::Open($Path, 'Open', 'ReadWrite', 'None')
    $handle.Dispose()
    $raw = [IO.File]::ReadAllText($Path)
    Assert-True (-not $raw.Contains('DO-NOT-RECORD')) "Privacy failed for $Mode"
    @([IO.File]::ReadAllLines($Path) | ForEach-Object { $_ | ConvertFrom-Json })
}

$temporary = Join-Path ([IO.Path]::GetTempPath()) ('pair-overhead-synthetic-' + [guid]::NewGuid())
[void][IO.Directory]::CreateDirectory($temporary)
try {
    $base = New-SyntheticProcess
    $same = New-SyntheticProcess -Cpu 4
    Assert-True (@(Compare-OverheadIdentity @($base) @($same)).Count -eq 0) 'Stable identity rejected'
    $metric = @(Measure-OverheadDelta @($base) @($same) 2 8)[0]
    Assert-True ($metric.CpuSeconds -eq 2 -and $metric.OneCorePercent -eq 100 -and $metric.MachinePercent -eq 12.5) 'CPU arithmetic'
    foreach ($field in @('CreatedUtc', 'Image', 'ParentPid')) {
        $changed = New-SyntheticProcess
        if ($field -eq 'ParentPid') { $changed.ParentPid = 9 } else { $changed.$field = 'different' }
        Assert-True (@(Compare-OverheadIdentity @($base) @($changed) | Where-Object Field -EQ $field).Count -eq 1) "Identity field $field not rejected"
    }
    Assert-True (@(Compare-OverheadIdentity @($base) @()).Count -eq 1) 'Disappearance not rejected'
    Assert-True (@(Compare-OverheadIdentity @($base) @((New-SyntheticProcess -Id 20))).Count -eq 1) 'Replacement not rejected'
    $children = @($same, (New-SyntheticProcess -Id 20 -Parent 10), (New-SyntheticProcess -Id 30 -Parent 20))
    Assert-True (@(Compare-OverheadIdentity @($base) $children | Where-Object Field -EQ 'AddedChild').Count -eq 2) 'Transitive added children not rejected'
    $cim = [pscustomobject]@{ ProcessId = 10; ParentProcessId = 1; ExecutablePath = $base.Image
        CreationDate = [datetime]::new(2026, 10, 23, 0, 0, 0, [DateTimeKind]::Utc)
        KernelModeTime = 10000000; UserModeTime = 10000000; WorkingSetSize = 100; PrivatePageCount = 80; HandleCount = 5 }
    $normalized = ConvertTo-OverheadProcess $cim
    Assert-True ($normalized.CreatedUtc -ceq $base.CreatedUtc -and $normalized.CpuSeconds -eq 2) 'Canonical CIM conversion'
    $stable = @(Invoke-SyntheticTrial (Join-Path $temporary 'stable.jsonl'))
    $samples = @($stable | Where-Object Kind -EQ 'Sample')
    Assert-True ($samples.Count -eq 31 -and $samples[0].ElapsedSeconds -eq 0 -and $samples[30].ElapsedSeconds -eq 60) 'Bounded sample schedule'
    Assert-True ($stable[-1].Kind -eq 'Complete' -and $stable[-1].Valid) 'Completion marker missing'
    $metrics = @($stable | Where-Object Kind -EQ 'Metrics')
    Assert-True ($samples[-1].ObserverCpuSeconds -gt 0 -and $metrics[-1].Metrics[0].MachinePercent -eq 12.5) 'Observer/metric record missing'
    $latency = @(Invoke-SyntheticTrial (Join-Path $temporary 'latency.jsonl') 'BaselineLatency')
    $firstMetric = @($latency | Where-Object Kind -EQ 'Metrics')[0]
    Assert-True ($firstMetric.ElapsedSeconds -eq 1 -and $firstMetric.Metrics[0].OneCorePercent -eq 100) 'Baseline capture latency biased CPU denominator'
    foreach ($mode in @('Reuse', 'CaptureFailure', 'Competition')) {
        $records = @(Invoke-SyntheticTrial (Join-Path $temporary "$mode.jsonl") $mode)
        Assert-True ($records[0].Kind -eq 'Sample' -and $records[0].Index -eq 0) "Baseline not retained for $mode"
        Assert-True ($records[-1].Kind -eq 'Abort' -and -not $records[-1].Valid -and $records[-1].CompletedSamples -eq 2) "Abort retention for $mode"
        $reasons = @{ Reuse = 'IdentityChanged'; CaptureFailure = 'CaptureRead'; Competition = 'CompetitionDetected' }
        Assert-True ($records[-1].Reason -eq $reasons[$mode] -and $records[-1].FailedIndex -eq 2) "Abort reason/index for $mode"
        if ($mode -eq 'Reuse') {
            Assert-True ($records[-2].IdentityFailures[0].Field -eq 'CreatedUtc') 'Failed identity fields not retained'
        }
    }
    $overrun = @(Invoke-SyntheticTrial (Join-Path $temporary 'overrun.jsonl') 'FinalOverrun')
    Assert-True ($overrun[-1].Kind -eq 'Abort' -and $overrun[-1].Reason -eq 'ScheduleInvalid' -and $overrun[-1].CompletedSamples -eq 30 -and $overrun[-1].FailedIndex -eq 30) 'Final capture overrun accepted'
    Assert-True ($overrun[-2].Kind -eq 'Sample' -and $overrun[-2].CaptureMonotonicSeconds -eq 65) 'Overrun capture not retained'
    $source = [IO.File]::ReadAllText("$PSScriptRoot/app-overhead.ps1")
    Assert-True ($source -notmatch '(?im)^\s*(Start-Process|Stop-Process|Invoke-WebRequest|Invoke-RestMethod|Register-ObjectEvent|Start-Job)\b') 'Unexpected operation or persistent timer'
    Write-Output "PASS: $script:passed assertions; synthetic fixtures only; no live measurement."
} finally {
    # Only this newly created synthetic directory is removed.
    [IO.Directory]::Delete($temporary, $true)
}
