# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

Set-StrictMode -Version Latest

function ConvertTo-OverheadProcess {
    param([Parameter(Mandatory)]$Process)
    # Both baseline and samples pass through this same CIM-only conversion.
    [pscustomobject]@{
        Pid = [long]$Process.ProcessId
        ParentPid = [long]$Process.ParentProcessId
        Image = [string]$Process.ExecutablePath
        CreatedUtc = $(if ($null -ne $Process.CreationDate) { $Process.CreationDate.ToUniversalTime().ToString('o') } else { '' })
        CpuSeconds = ([double]$Process.KernelModeTime + [double]$Process.UserModeTime) / 10000000
        WorkingSetBytes = [long]$Process.WorkingSetSize
        PrivateBytes = [long]$Process.PrivatePageCount
        Handles = [long]$Process.HandleCount
    }
}

function Get-OverheadProcessSnapshot {
    # Deliberately select metadata only: never query CommandLine or profiles.
    @(Get-CimInstance -ClassName Win32_Process -OperationTimeoutSec 1 -ErrorAction Stop -Property ProcessId, ParentProcessId,
        ExecutablePath, CreationDate, KernelModeTime, UserModeTime, WorkingSetSize,
        PrivatePageCount, HandleCount | ForEach-Object { ConvertTo-OverheadProcess $_ })
}

function Resolve-OverheadOutputPath {
    param([string]$Path, [string]$Repository, [string]$Vault,
        [scriptblock]$DriveType = { param($Root) ([IO.DriveInfo]::new($Root)).DriveType })
    # Reject UNC/device paths before any filesystem operation can touch them.
    if ($Path -notmatch '^[A-Za-z]:[\\/]') { throw 'Output must be on a local fixed drive.' }
    $full = [IO.Path]::GetFullPath($Path)
    if ((& $DriveType ([IO.Path]::GetPathRoot($full))) -ne [IO.DriveType]::Fixed) { throw 'Output drive is not fixed.' }
    foreach ($excluded in @($Repository, $Vault)) {
        if ([string]::IsNullOrWhiteSpace($excluded)) { throw 'Repository and vault roots are required.' }
        $root = [IO.Path]::GetFullPath($excluded).TrimEnd('\', '/')
        if ($full.Equals($root, [StringComparison]::OrdinalIgnoreCase) -or
            $full.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Output is inside the repository or vault.' }
    }
    # A junction could otherwise redirect an apparently safe local path into an
    # excluded root or a share. Fail closed rather than resolving it by access.
    $ancestor = [IO.Path]::GetDirectoryName($full)
    while ($ancestor) {
        if ([IO.Directory]::Exists($ancestor) -and
            (([IO.File]::GetAttributes($ancestor) -band [IO.FileAttributes]::ReparsePoint) -ne 0)) { throw 'Output ancestors must not be reparse points.' }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
    $full
}

function Compare-OverheadIdentity {
    param([array]$Baseline, [array]$Current)
    $failures = @()
    foreach ($expected in $Baseline) {
        $matches = @($Current | Where-Object Pid -EQ $expected.Pid)
        if ($matches.Count -ne 1) {
            $failures += [pscustomobject]@{ Pid = $expected.Pid; Field = 'Pid'; Expected = $expected.Pid; Actual = @($matches | Select-Object -ExpandProperty Pid) }
            continue
        }
        foreach ($field in @('Image', 'CreatedUtc', 'ParentPid')) {
            if ($expected.$field -cne $matches[0].$field) {
                $failures += [pscustomobject]@{ Pid = $expected.Pid; Field = $field; Expected = $expected.$field; Actual = $matches[0].$field }
            }
        }
    }
    # Include descendants transitively, so a newly added runtime is not hidden
    # behind another newly added process. Never silently adopt new children.
    $known = @($Baseline.Pid)
    do {
        $added = @($Current | Where-Object { $_.ParentPid -in $known -and $_.Pid -notin $known })
        foreach ($process in $added) {
            $known += $process.Pid
            $failures += [pscustomobject]@{ Pid = $process.Pid; Field = 'AddedChild'; Expected = $null
                Actual = ($process | Select-Object Pid, ParentPid, Image, CreatedUtc) }
        }
    } while ($added.Count -gt 0)
    $failures
}

function Measure-OverheadDelta {
    param([array]$Baseline, [array]$Current, [double]$ElapsedSeconds, [int]$LogicalCpus)
    if ($ElapsedSeconds -le 0 -or $LogicalCpus -le 0) { throw 'Invalid elapsed time or logical CPU count.' }
    if (@(Compare-OverheadIdentity $Baseline $Current).Count) { throw 'Unstable process identity.' }
    foreach ($first in $Baseline) {
        $last = @($Current | Where-Object Pid -EQ $first.Pid)[0]
        $cpu = $last.CpuSeconds - $first.CpuSeconds
        if ($cpu -lt 0) { throw 'Cumulative CPU decreased.' }
        [pscustomobject]@{
            Pid = $first.Pid; CpuSeconds = $cpu
            OneCorePercent = 100 * $cpu / $ElapsedSeconds
            MachinePercent = 100 * $cpu / $ElapsedSeconds / $LogicalCpus
            WorkingSetBytes = $last.WorkingSetBytes
            PrivateBytes = $last.PrivateBytes; Handles = $last.Handles
        }
    }
}

function Invoke-AppOverheadSample {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][array]$Authority,
        [Parameter(Mandatory)][string]$OutputPath,
        [Parameter(Mandatory)][string]$SourcePin,
        [Parameter(Mandatory)][string]$ControllerProvenance,
        [Parameter(Mandatory)][string]$StateAnnotation,
        [Parameter(Mandatory)][int]$LogicalCpus,
        [Parameter(Mandatory)][scriptblock]$AcquireLease,
        [Parameter(Mandatory)][scriptblock]$ReleaseLease,
        [Parameter(Mandatory)][scriptblock]$Competition,
        [scriptblock]$Capture = { Get-OverheadProcessSnapshot },
        [scriptblock]$Monotonic = { [diagnostics.stopwatch]::GetTimestamp() / [double][diagnostics.stopwatch]::Frequency },
        [scriptblock]$WallTime = { [datetime]::UtcNow.ToString('o') },
        [scriptblock]$Wait = { param($Seconds) Start-Sleep -Seconds $Seconds },
        [scriptblock]$ObserverCpu = { [diagnostics.process]::GetCurrentProcess().TotalProcessorTime.TotalSeconds }
    )
    $OutputPath = Resolve-OverheadOutputPath $OutputPath ([IO.Path]::GetFullPath("$PSScriptRoot/../..")) $env:WASABI_VAULT
    if ($LogicalCpus -le 0 -or $Authority.Count -eq 0) { throw 'Explicit authority and logical CPU count required.' }
    $ids = @()
    foreach ($entry in $Authority) {
        if ($entry.Role -notin @('Application', 'RuntimeModel', 'Observer') -or
            $entry.Pid -le 0 -or $entry.Pid -in $ids -or
            [string]::IsNullOrWhiteSpace($entry.Image) -or [string]::IsNullOrWhiteSpace($entry.CreatedUtc)) {
            throw 'Invalid explicit process authority.'
        }
        $ids += $entry.Pid
    }
    # CreateNew prevents overwriting an earlier trial, including failed trials.
    $stream = [IO.File]::Open($OutputPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false))
    $leased = $false
    $completed = 0
    $baseline = @()
    $reason = 'LeaseAcquire'
    $index = -1
    try {
        & $AcquireLease
        $leased = $true
        $reason = 'ObserverRead'
        $observerStart = & $ObserverCpu
        $reason = 'ClockRead'
        $start = & $Monotonic
        $lastTime = -1.0
        for ($index = 0; $index -le 30; $index++) {
            $reason = 'ScheduleWait'
            $remaining = $start + 2 * $index - (& $Monotonic)
            if ($remaining -gt 0) { & $Wait $remaining }
            $now = & $Monotonic
            $elapsed = $now - $start
            $reason = 'ScheduleInvalid'
            if ($elapsed -lt $lastTime -or $elapsed -gt 62) { throw 'Schedule overrun or nonmonotonic clock.' }
            $reason = 'SlotMissed'
            if ($elapsed -gt 2 * $index + 1) { throw 'Sample missed its scheduled slot.' }
            $reason = 'CompetitionRead'
            $competitionNames = @(& $Competition)
            $reason = 'CaptureRead'
            $all = @(& $Capture)
            $capturedAt = & $Monotonic
            $captureElapsed = $capturedAt - $start
            $failures = @(Compare-OverheadIdentity $Authority $all)
            $selected = @($all | Where-Object Pid -In $ids)
            # Explicit allowlist avoids serializing arbitrary capture properties.
            $safe = @($selected | ForEach-Object {
                [pscustomobject]@{ Pid = $_.Pid; ParentPid = $_.ParentPid; Image = $_.Image; CreatedUtc = $_.CreatedUtc
                    CpuSeconds = $_.CpuSeconds; WorkingSetBytes = $_.WorkingSetBytes; PrivateBytes = $_.PrivateBytes; Handles = $_.Handles }
            })
            if ($index -eq 0) { $baseline = $safe; $baselineAt = $capturedAt }
            $metricElapsed = $capturedAt - $baselineAt
            $record = [ordered]@{ Kind = 'Sample'; Index = $index; WallUtc = (& $WallTime); MonotonicSeconds = $now
                ElapsedSeconds = $elapsed; LogicalCpus = $LogicalCpus; SourcePin = $SourcePin
                CaptureMonotonicSeconds = $capturedAt; MetricElapsedSeconds = $metricElapsed
                ControllerProvenance = $ControllerProvenance; StateAnnotation = $StateAnnotation
                Authority = @($Authority | Select-Object Pid, ParentPid, Image, CreatedUtc, Role)
                Processes = $safe; IdentityFailures = $failures; Competition = $competitionNames
                ObserverCpuSeconds = ((& $ObserverCpu) - $observerStart); Metrics = @() }
            $reason = 'RecordWrite'
            # Persist capture metadata before any subsequent reducer failure.
            $writer.WriteLine(($record | ConvertTo-Json -Depth 12 -Compress))
            $writer.Flush()
            $reason = 'IdentityChanged'
            if ($failures.Count) { throw 'Identity changed; trial invalid.' }
            $reason = 'CompetitionDetected'
            if ($competitionNames.Count) { throw 'Competition detected; trial invalid.' }
            $reason = 'SlotMissed'
            if ($captureElapsed -gt 2 * $index + 1) { throw 'Capture missed its scheduled slot.' }
            $reason = 'ScheduleInvalid'
            if ($captureElapsed -lt $elapsed -or $captureElapsed -gt 62) { throw 'Capture overrun or nonmonotonic clock.' }
            $lastTime = $captureElapsed
            if ($index -gt 0) {
                $reason = 'MetricInvalid'
                $metrics = @(Measure-OverheadDelta $baseline $safe $metricElapsed $LogicalCpus)
                $reason = 'RecordWrite'
                $writer.WriteLine((@{ Kind = 'Metrics'; Index = $index; ElapsedSeconds = $metricElapsed; Metrics = $metrics } | ConvertTo-Json -Depth 6 -Compress))
                $writer.Flush()
            }
            $completed++
        }
        $writer.WriteLine((@{ Kind = 'Complete'; Samples = $completed; Valid = $true } | ConvertTo-Json -Compress))
        $writer.Flush()
    } catch {
        if ($reason -eq 'CaptureRead') {
            if ($_.Exception -is [TimeoutException]) { $reason = 'CaptureTimeout' }
            else { $reason = 'CaptureFailedOrTimedOut' }
        }
        # Do not serialize exception messages: injected controllers may contain
        # sensitive context. The failed sample already holds identity fields.
        $writer.WriteLine((@{ Kind = 'Abort'; CompletedSamples = $completed; Valid = $false; Reason = $reason; FailedIndex = $index } | ConvertTo-Json -Compress))
        $writer.Flush()
        throw
    } finally {
        try { if ($leased) { & $ReleaseLease } } finally { $writer.Dispose() }
    }
}
