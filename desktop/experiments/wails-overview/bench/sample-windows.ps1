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

function Read-Counters([string[]]$paths) {
    try {
        # A disappearing instance can leave valid sibling samples in the response.
        $query = Get-Counter -Counter $paths -ErrorAction SilentlyContinue
        if ($query) { $query.CounterSamples }
    } catch {
        # Unsupported counters are unknown.
    }
}

function Counter-Value($sample) {
    if ($null -ne $sample -and $sample.Status -in @(0, 1) -and
        $sample.CookedValue -ge 0 -and ![double]::IsInfinity($sample.CookedValue)) {
        return $sample.CookedValue
    }
    return $null
}

function Read-ProcessCounters($category) {
    $instances = @{}
    $paths = @('ID Process', 'Working Set - Private', 'Private Bytes', '% Processor Time') |
        ForEach-Object { "\$category(*)\$_" }
    foreach ($sample in Read-Counters $paths) {
        if ($sample.Path -notmatch '\\([^\\]+)$') { continue }
        $name = $Matches[1]
        $instance = $sample.Path.Substring(0, $sample.Path.LastIndexOf('\'))
        if (!$instances.ContainsKey($instance)) { $instances[$instance] = @{} }
        $instances[$instance][$name] = Counter-Value $sample
    }
    $byPid = @{}
    $ambiguous = @{}
    foreach ($item in $instances.Values) {
        if ($item['ID Process'] -gt 0) {
            $id = [int]$item['ID Process']
            if ($byPid.ContainsKey($id)) { $ambiguous[$id] = $true }
            $byPid[$id] = $item
        }
    }
    foreach ($id in $ambiguous.Keys) { $byPid.Remove($id) }
    return $byPid
}

function Read-GpuCounters {
    $byPid = @{}
    $paths = @('\GPU Engine(*)\Utilization Percentage',
        '\GPU Process Memory(*)\Dedicated Usage', '\GPU Process Memory(*)\Shared Usage')
    foreach ($path in $paths) {
        foreach ($sample in Read-Counters @($path)) {
            if ($sample.InstanceName -notmatch '(?:^|_)pid_(\d+)(?:_|$)') { continue }
            $id = [int]$Matches[1]
            if (!$byPid.ContainsKey($id)) { $byPid[$id] = @{} }
            $name = $path.Substring($path.LastIndexOf('\') + 1)
            $value = Counter-Value $sample
            $row = $byPid[$id]
            if (!$row.ContainsKey($name)) { $row[$name] = $value }
            elseif ($null -eq $value -or $null -eq $row[$name]) { $row[$name] = $null }
            else { $row[$name] += $value }
        }
    }
    return $byPid
}

function Read-SystemCounters {
    $values = @{}
    $paths = @('\Memory\Available Bytes', '\Memory\Pages Input/sec', '\Memory\Pages Output/sec')
    foreach ($sample in Read-Counters $paths) {
        $values[$sample.Path.Substring($sample.Path.LastIndexOf('\') + 1)] = Counter-Value $sample
    }
    return @{
        availableBytes=$values['Available Bytes']; pageInPerSec=$values['Pages Input/sec']
        pageOutPerSec=$values['Pages Output/sec']; pageSizeBytes=[Environment]::SystemPageSize
        pageInCount=$null; pageOutCount=$null; swapInCount=$null; swapOutCount=$null; pressureFreePercent=$null
    }
}

$category = 'Process'
if (Get-Counter -ListSet 'Process V2' -ErrorAction SilentlyContinue) { $category = 'Process V2' }
$known = @{}
$previous = @{}
$previousMs = 0
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
    $perf = Read-ProcessCounters $category
    $gpuCounters = Read-GpuCounters
    $after = Get-ProcessSnapshot
    $known = @{}
    $current = @{}
    $rows = @(foreach ($id in $selected.Keys) {
        $attribution = $selected[$id]
        if ($attribution.group -eq 'observer') { continue }
        $row = $before[$id]
        $key = '{0}/{1}' -f $id,$row.createdUtc
        $stable = $after.ContainsKey($id) -and $after[$id].createdUtc -ceq $row.createdUtc
        $cpu = $null
        if ($previous.ContainsKey($key) -and $row.cpuSeconds -ge $previous[$key]) {
            $cpu = ($row.cpuSeconds - $previous[$key]) * 1000000 / ($elapsedMs - $previousMs)
        }
        $known[$key] = $attribution
        $current[$key] = $row.cpuSeconds
        $processPerf = @{}
        $processGpu = @{}
        if ($stable -and $perf.ContainsKey($id)) { $processPerf = $perf[$id] }
        if ($stable -and $gpuCounters.ContainsKey($id)) { $processGpu = $gpuCounters[$id] }
        @{
            pid=$id; createdUtc=$row.createdUtc; group=$attribution.group; role=$attribution.role
            memoryBytes=$processPerf['Working Set - Private']; commitBytes=$processPerf['Private Bytes']
            cpuMsPerSec=$cpu; cpuPercentOneCore=$processPerf['% Processor Time']
            gpuEnginePercent=$processGpu['Utilization Percentage']; gpuDedicatedBytes=$processGpu['Dedicated Usage']
            gpuSharedBytes=$processGpu['Shared Usage']
        }
    })
    # Processes may appear after the initial wall timestamp but before CIM enumerates them.
    foreach ($row in $rows) {
        $creation = $before[$row.pid].start
        if ($creation -gt $timestamp) { $timestamp = $creation }
    }
    @{
        schema=1; os='windows'; elapsedMs=$elapsedMs; timestamp=$timestamp.ToString("yyyy-MM-ddTHH:mm:ss.ffffff'Z'")
        processes=$rows; system=(Read-SystemCounters)
    } | ConvertTo-Json -Depth 5 -Compress
    $previousMs = $elapsedMs
    $previous = $current
    $next += $intervalMs
    while ($next -lt $clock.ElapsedMilliseconds) { $next += $intervalMs }
}
