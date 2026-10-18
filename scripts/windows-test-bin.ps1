# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

[CmdletBinding()]
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('list', 'rules', 'run')][string]$Command,
    [switch]$Remove,
    [string]$Module,
    [string]$Package = '.',
    [string]$Run,
    [string]$Timeout = '10m'
)

$ErrorActionPreference = 'Stop'
$root = $env:NVPAIR_TEST_BIN_DIR
if (!$root -or $root -notmatch '^(?:[A-Za-z]:[\\/]|\\\\[^\\/]+[\\/][^\\/]+(?:[\\/]|$))') {
    throw 'NVPAIR_TEST_BIN_DIR must be set to an absolute path.'
}
$root = [IO.Path]::GetFullPath($root)
$services = Join-Path (Split-Path $PSScriptRoot -Parent) 'services'

function Get-TestPackages([IO.DirectoryInfo]$ModuleDir) {
    $lines = & go -C $ModuleDir.FullName list -f '{{if or .TestGoFiles .XTestGoFiles}}{{.Dir}}{{end}}' ./...
    if ($LASTEXITCODE -ne 0) { throw "go list failed for $($ModuleDir.Name)" }
    foreach ($line in $lines) {
        if (!$line) { continue }
        $relative = $line.Substring($ModuleDir.FullName.Length).TrimStart('\', '/').Replace('\', '/')
        $name = $ModuleDir.Name
        $pkg = '.'
        if ($relative) {
            $name += '-' + $relative.Replace('/', '-')
            $pkg = './' + $relative
        }
        [pscustomobject]@{
            Package = $pkg
            Directory = $line
            Path = Join-Path $root "go-test\$($ModuleDir.Name)\$name.test.exe"
        }
    }
}

if ($Command -eq 'run') {
    if (!$Module -or $Module -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'Specify a services module with -Module.' }
    $moduleDir = Get-Item -LiteralPath (Join-Path $services $Module)
    if (!(Test-Path -LiteralPath (Join-Path $moduleDir.FullName 'go.mod'))) { throw 'Module must contain go.mod.' }
    $pkg = $Package.Replace('\', '/').TrimEnd('/')
    if ($pkg -ne '.' -and !$pkg.StartsWith('./')) { $pkg = './' + $pkg }
    $selected = @(Get-TestPackages $moduleDir | Where-Object { $_.Package -eq $pkg })
    if ($selected.Count -ne 1) { throw "Expected one package with tests matching $Package in $Module." }
    $target = $selected[0]
    New-Item -ItemType Directory -Force -Path (Split-Path $target.Path -Parent) | Out-Null
    & go -C $moduleDir.FullName test -c -o $target.Path $target.Package
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $testArgs = @('-test.count=1', '-test.v', "-test.timeout=$Timeout")
    if ($Run) { $testArgs += @('-test.run', $Run) }
    Push-Location -LiteralPath $target.Directory
    try {
        & $target.Path @testArgs
        $code = $LASTEXITCODE
    } finally { Pop-Location }
    exit $code
}

$relativePrograms = & go -C (Join-Path $services 'shared') run ./testbin/cmd/testbin-programs
if ($LASTEXITCODE -ne 0) { throw 'Could not read test binary manifest.' }
$paths = @($relativePrograms | ForEach-Object { Join-Path $root ($_.Replace('/', '\') + '.exe') })
foreach ($moduleDir in Get-ChildItem -LiteralPath $services -Directory) {
    if (Test-Path -LiteralPath (Join-Path $moduleDir.FullName 'go.mod')) {
        $paths += @(Get-TestPackages $moduleDir | ForEach-Object { $_.Path })
    }
}
$paths = @($paths | Sort-Object -Unique)
if (!$paths.Count) { throw 'No test programs found.' }
if ($Command -eq 'list') {
    $paths
    exit 0
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'The rules command requires an elevated PowerShell.'
}
$changed = 0
$skipped = 0
foreach ($path in $paths) {
    $name = 'pair-test-bin ' + $path.Substring($root.TrimEnd('\').Length + 1)
    $existing = @(Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -eq $name })
    if ($Remove) {
        if ($existing.Count) { $existing | Remove-NetFirewallRule; $changed++ } else { $skipped++ }
    } elseif ($existing.Count) {
        $skipped++
    } else {
        New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Block -Program $path -Enabled True -Profile Any | Out-Null
        $changed++
    }
}
if ($Remove) { "Removed: $changed; skipped: $skipped" } else { "Created: $changed; skipped: $skipped" }
