# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

[CmdletBinding()]
param(
    [string[]]$GuiPath = @(),
    [string[]]$GuiName = @(),
    [string]$HelperPattern = '(?!)',
    [string]$RendererPattern = '--type=(renderer|webview)(\s|$)',
    [string]$GpuPattern = '--type=gpu-process(\s|$)',
    [string]$ServicePattern = '^nvpair-(service|ui-broker|cluster-manager|engine-manager|errors|job-scheduler|manual-nodes|node-info|node-scanner|node-settings|proxy|workload-manager)$',
    [string]$ModelPattern = '(?!)',
    [ValidateRange(1,3600)][int]$IntervalSeconds = 1,
    [ValidateRange(1,86400)][int]$DurationSeconds = 300
)
$ErrorActionPreference = 'Stop'

function Get-ProcessSnapshot {
    $snapshot = @{}
    foreach ($item in Get-CimInstance Win32_Process -Property ProcessId,ParentProcessId,
            CreationDate,ExecutablePath,Name,CommandLine,KernelModeTime,UserModeTime) {
        if (!$item.CreationDate -or !$item.ProcessId) { continue }
        $start = $item.CreationDate.ToUniversalTime()
        $snapshot[[int]$item.ProcessId] = [pscustomobject]@{
            pid = [int]$item.ProcessId
            parent = [int]$item.ParentProcessId
            start = $start
            createdUtc = $start.ToString("yyyy-MM-ddTHH:mm:ss.ffffff'Z'")
            image = [string]$item.ExecutablePath
            name = [IO.Path]::GetFileNameWithoutExtension($item.Name)
            matchText = "$($item.ExecutablePath) $($item.CommandLine)"
            cpuSeconds = ([double]$item.KernelModeTime + [double]$item.UserModeTime) / 10000000
        }
    }
    return $snapshot
}

function Get-GuiRole($row) {
    if ($row.matchText -match $GpuPattern) { return 'gpu' }
    if ($row.matchText -match $RendererPattern) { return 'renderer' }
    return 'other'
}

function Add-Descendants($snapshot, $selected) {
    do {
        $added = $false
        foreach ($row in $snapshot.Values) {
            if ($selected.ContainsKey($row.pid) -or !$selected.ContainsKey($row.parent) -or
                $snapshot[$row.parent].start -gt $row.start) { continue }
            $owner = $selected[$row.parent].group
            $role = switch ($owner) {
                'gui' { Get-GuiRole $row }
                'service' { 'worker' }
                'model' { 'engine' }
            }
            $selected[$row.pid] = @{group=$owner; role=$role}
            $added = $true
        }
    } while ($added)
}

# Raw performance data comes from one CIM query per class. Get-Counter cooks rate
# counters from two collections a second apart, which made one pass take 10-15 s.
function Read-Raw([string]$class, [string[]]$properties) {
    try { return @(Get-CimInstance -ClassName $class -Property $properties) } catch { return @() }
}

# Rate of a raw counter between this pass and the previous pass of the same key.
# The first observation of a key, a counter reset or a missing value is unknown.
function Get-Rate($previous, $key, $value, $time, [double]$scale) {
    $rate = $null
    $last = $previous[$key]
    if ($null -ne $last -and $null -ne $value -and $time -gt $last.time -and $value -ge $last.value) {
        $rate = ($value - $last.value) * $scale / ($time - $last.time)
    }
    return $rate
}

# Process V2 names instances '<image>:<pid>' and has no ID Process counter. The
# legacy Process set names instances '<image>#<n>' and carries ID Process.
function Read-ProcessCounters {
    $byPid = @{}
    $ambiguous = @{}
    $properties = @('Name', 'WorkingSetPrivate', 'PrivateBytes', 'PercentProcessorTime', 'Timestamp_Sys100NS')
    if ($processV2) {
        $rows = Read-Raw 'Win32_PerfRawData_Counters_ProcessV2' $properties
    } else {
        $rows = Read-Raw 'Win32_PerfRawData_PerfProc_Process' ($properties + 'IDProcess')
    }
    foreach ($row in $rows) {
        if ($processV2) {
            if ($row.Name -notmatch ':(\d+)$') { continue }
            $id = [int]$Matches[1]
        } else {
            $id = [int]$row.IDProcess
        }
        if ($id -le 0) { continue }
        if ($byPid.ContainsKey($id)) { $ambiguous[$id] = $true }
        $byPid[$id] = $row
    }
    foreach ($id in $ambiguous.Keys) { $byPid.Remove($id) }
    return $byPid
}

function Read-GpuCounters {
    $engines = @{}
    foreach ($row in Read-Raw 'Win32_PerfRawData_GPUPerformanceCounters_GPUEngine' @('Name', 'UtilizationPercentage', 'Timestamp_Sys100NS')) {
        if ($row.Name -notmatch '(?:^|_)pid_(\d+)(?:_|$)') { continue }
        $id = [int]$Matches[1]
        if (!$engines.ContainsKey($id)) { $engines[$id] = @() }
        $engines[$id] += $row
    }
    $memory = @{}
    foreach ($row in Read-Raw 'Win32_PerfRawData_GPUPerformanceCounters_GPUProcessMemory' @('Name', 'DedicatedUsage', 'SharedUsage')) {
        if ($row.Name -notmatch '(?:^|_)pid_(\d+)(?:_|$)') { continue }
        $id = [int]$Matches[1]
        if (!$memory.ContainsKey($id)) { $memory[$id] = @{dedicated=0.0; shared=0.0} }
        $memory[$id].dedicated += [double]$row.DedicatedUsage
        $memory[$id].shared += [double]$row.SharedUsage
    }
    return @{engines=$engines; memory=$memory}
}

function Read-SystemCounters($previous) {
    $values = @{availableBytes=$null; pageInPerSec=$null; pageOutPerSec=$null}
    $row = Read-Raw 'Win32_PerfRawData_PerfOS_Memory' @('AvailableBytes', 'PagesInputPersec', 'PagesOutputPersec',
        'Timestamp_PerfTime', 'Frequency_PerfTime') | Select-Object -First 1
    if ($row -and $row.Frequency_PerfTime -gt 0) {
        $time = [double]$row.Timestamp_PerfTime
        $scale = [double]$row.Frequency_PerfTime
        $values.availableBytes = [double]$row.AvailableBytes
        $values.pageInPerSec = Get-Rate $previous 'pageIn' ([double]$row.PagesInputPersec) $time $scale
        $values.pageOutPerSec = Get-Rate $previous 'pageOut' ([double]$row.PagesOutputPersec) $time $scale
        $previous['pageIn'] = @{value=[double]$row.PagesInputPersec; time=$time}
        $previous['pageOut'] = @{value=[double]$row.PagesOutputPersec; time=$time}
    }
    return @{
        availableBytes=$values.availableBytes; pageInPerSec=$values.pageInPerSec
        pageOutPerSec=$values.pageOutPerSec; pageSizeBytes=[Environment]::SystemPageSize
        pageInCount=$null; pageOutCount=$null; swapInCount=$null; swapOutCount=$null; pressureFreePercent=$null
    }
}

$processV2 = [bool](Get-CimClass -ClassName 'Win32_PerfRawData_Counters_ProcessV2' -ErrorAction SilentlyContinue)
$known = @{}
$previous = @{}
$previousMs = 0
$previousRaw = @{}
$systemRaw = @{}
$intervalMs = $IntervalSeconds * 1000
$durationMs = $DurationSeconds * 1000
$clock = [Diagnostics.Stopwatch]::StartNew()
for ($next = 0; $next -lt $durationMs;) {
    $wait = $next - $clock.ElapsedMilliseconds
    if ($wait -gt 0) { Start-Sleep -Milliseconds $wait }
    $elapsedMs = $clock.Elapsed.TotalMilliseconds
    if ($next -gt 0 -and $elapsedMs -ge $durationMs) { break }
    $timestamp = [DateTime]::UtcNow
    $before = Get-ProcessSnapshot
    $selected = @{$PID = @{group='observer'}}
    foreach ($row in $before.Values) {
        if ($row.pid -eq $PID) { continue }
        $key = '{0}/{1}' -f $row.pid,$row.createdUtc
        if ($row.matchText -match $ModelPattern) {
            $selected[$row.pid] = @{group='model'; role='engine'}
        } elseif ($row.name -match $ServicePattern) {
            $role = switch ($row.name) {
                'nvpair-service' { 'service' }
                'nvpair-ui-broker' { 'broker' }
                default { 'worker' }
            }
            $selected[$row.pid] = @{group='service'; role=$role}
        } elseif (($row.image -and $GuiPath -contains $row.image) -or $GuiName -contains $row.name) {
            $role = Get-GuiRole $row
            if ($role -eq 'other' -and $row.matchText -notmatch '--type=') { $role = 'main' }
            $selected[$row.pid] = @{group='gui'; role=$role}
        } elseif ($row.matchText -match $HelperPattern) {
            $selected[$row.pid] = @{group='gui'; role=(Get-GuiRole $row)}
        } elseif ($known.ContainsKey($key)) {
            $selected[$row.pid] = $known[$key]
        }
    }
    Add-Descendants $before $selected
    $perf = Read-ProcessCounters
    $gpu = Read-GpuCounters
    $system = Read-SystemCounters $systemRaw
    $after = Get-ProcessSnapshot
    $known = @{}
    $current = @{}
    $raw = @{}
    $rows = @(foreach ($id in $selected.Keys) {
        $attribution = $selected[$id]
        if ($attribution.group -eq 'observer') { continue }
        $row = $before[$id]
        $key = '{0}/{1}' -f $id,$row.createdUtc
        # A process that exited during the pass has no counters for this sample.
        # Its PID could already name another process, so the row is omitted.
        if (!$after.ContainsKey($id) -or $after[$id].createdUtc -cne $row.createdUtc) { continue }
        $cpu = $null
        if ($previous.ContainsKey($key) -and $row.cpuSeconds -ge $previous[$key]) {
            $cpu = ($row.cpuSeconds - $previous[$key]) * 1000000 / ($elapsedMs - $previousMs)
        }
        $known[$key] = $attribution
        $current[$key] = $row.cpuSeconds
        $memory = $null; $commit = $null; $cpuPercent = $null
        if ($perf.ContainsKey($id)) {
            $counter = $perf[$id]
            $memory = [double]$counter.WorkingSetPrivate
            $commit = [double]$counter.PrivateBytes
            $time = [double]$counter.Timestamp_Sys100NS
            $cpuPercent = Get-Rate $previousRaw "$key/cpu" ([double]$counter.PercentProcessorTime) $time 100
            $raw["$key/cpu"] = @{value=[double]$counter.PercentProcessorTime; time=$time}
        }
        $gpuPercent = $null
        if ($gpu.engines.ContainsKey($id)) {
            $gpuPercent = 0.0
            foreach ($engine in $gpu.engines[$id]) {
                $engineKey = "$key/gpu/$($engine.Name)"
                $time = [double]$engine.Timestamp_Sys100NS
                $rate = Get-Rate $previousRaw $engineKey ([double]$engine.UtilizationPercentage) $time 100
                $raw[$engineKey] = @{value=[double]$engine.UtilizationPercentage; time=$time}
                if ($null -eq $rate -or $null -eq $gpuPercent) { $gpuPercent = $null } else { $gpuPercent += $rate }
            }
        }
        $gpuMemory = $gpu.memory[$id]
        @{
            pid=$id; createdUtc=$row.createdUtc; group=$attribution.group; role=$attribution.role
            memoryBytes=$memory; commitBytes=$commit
            cpuMsPerSec=$cpu; cpuPercentOneCore=$cpuPercent
            gpuEnginePercent=$gpuPercent
            gpuDedicatedBytes=$(if ($gpuMemory) { $gpuMemory.dedicated } else { $null })
            gpuSharedBytes=$(if ($gpuMemory) { $gpuMemory.shared } else { $null })
        }
    })
    # Processes may appear after the initial wall timestamp but before CIM enumerates them.
    foreach ($row in $rows) {
        $creation = $before[$row.pid].start
        if ($creation -gt $timestamp) { $timestamp = $creation }
    }
    @{
        schema=1; os='windows'; elapsedMs=$elapsedMs; timestamp=$timestamp.ToString("yyyy-MM-ddTHH:mm:ss.ffffff'Z'")
        processes=$rows; system=$system
    } | ConvertTo-Json -Depth 5 -Compress
    $previousMs = $elapsedMs
    $previous = $current
    $previousRaw = $raw
    $next += $intervalMs
    while ($next -lt $clock.ElapsedMilliseconds) { $next += $intervalMs }
}
