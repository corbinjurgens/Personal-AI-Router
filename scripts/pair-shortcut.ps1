# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateSet('create', 'status', 'remove')]
    [string]$Command,
    [Parameter(Mandatory)]
    [string]$AppPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

try {
    if ($AppPath -notmatch '^(?:[A-Za-z]:\\|\\\\[^\\]+\\[^\\]+\\)') {
        throw 'AppPath must be an absolute Windows path.'
    }
    $item = Get-Item -LiteralPath $AppPath -Force
    if ($item.PSIsContainer -or $item.Name -ine 'PAIR.exe' -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'AppPath must name an existing regular PAIR.exe file.'
    }
    $process = Start-Process -FilePath $item.FullName -ArgumentList "--pair-shortcut=$Command" -Wait -PassThru
    exit $process.ExitCode
} catch {
    [Console]::Error.WriteLine("PAIR shortcut command failed: $($_.Exception.Message)")
    exit 1
}
