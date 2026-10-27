# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

Describe 'pair-shortcut launcher adapter' {
    BeforeEach {
        $script:appPath = 'C:\Candidate with spaces\PAIR.exe'
        Mock Get-Item {
            [pscustomobject]@{
                FullName = 'C:\Candidate with spaces\PAIR.exe'
                Name = 'PAIR.exe'
                PSIsContainer = $false
                Attributes = [IO.FileAttributes]::Normal
            }
        }
        Mock Start-Process { [pscustomobject]@{ ExitCode = 0 } }
    }

    It 'passes the explicit path and only the requested fixed flag' {
        & "$PSScriptRoot\pair-shortcut.ps1" create -AppPath $script:appPath
        $LASTEXITCODE | Should Be 0
        Assert-MockCalled Get-Item -Times 1 -Exactly -Scope It -ParameterFilter { $LiteralPath -eq $script:appPath }
        Assert-MockCalled Start-Process -Times 1 -Exactly -Scope It -ParameterFilter {
            $FilePath -eq $script:appPath -and $ArgumentList -eq '--pair-shortcut=create' -and $Wait -and $PassThru
        }
    }

    It 'passes status and preserves a product failure code' {
        Mock Start-Process { [pscustomobject]@{ ExitCode = 69 } }
        & "$PSScriptRoot\pair-shortcut.ps1" status -AppPath $script:appPath
        $LASTEXITCODE | Should Be 69
        Assert-MockCalled Start-Process -Times 1 -Exactly -Scope It -ParameterFilter { $ArgumentList -eq '--pair-shortcut=status' }
    }

    It 'passes the remove flag' {
        & "$PSScriptRoot\pair-shortcut.ps1" remove -AppPath $script:appPath
        $LASTEXITCODE | Should Be 0
        Assert-MockCalled Start-Process -Times 1 -Exactly -Scope It -ParameterFilter { $ArgumentList -eq '--pair-shortcut=remove' }
    }

    It 'rejects relative paths before consulting the filesystem' {
        & "$PSScriptRoot\pair-shortcut.ps1" create -AppPath 'PAIR.exe'
        $LASTEXITCODE | Should Be 1
        Assert-MockCalled Get-Item -Times 0 -Exactly -Scope It
        Assert-MockCalled Start-Process -Times 0 -Exactly -Scope It
    }

    It 'rejects another executable' {
        Mock Get-Item { [pscustomobject]@{ FullName = $script:appPath; Name = 'other.exe'; PSIsContainer = $false; Attributes = [IO.FileAttributes]::Normal } }
        & "$PSScriptRoot\pair-shortcut.ps1" create -AppPath $script:appPath
        $LASTEXITCODE | Should Be 1
        Assert-MockCalled Start-Process -Times 0 -Exactly -Scope It
    }

    It 'rejects a directory named PAIR.exe' {
        Mock Get-Item { [pscustomobject]@{ FullName = $script:appPath; Name = 'PAIR.exe'; PSIsContainer = $true; Attributes = [IO.FileAttributes]::Directory } }
        & "$PSScriptRoot\pair-shortcut.ps1" create -AppPath $script:appPath
        $LASTEXITCODE | Should Be 1
        Assert-MockCalled Start-Process -Times 0 -Exactly -Scope It
    }

    It 'reports a missing file without launching anything' {
        Mock Get-Item { throw 'File not found' }
        & "$PSScriptRoot\pair-shortcut.ps1" create -AppPath $script:appPath
        $LASTEXITCODE | Should Be 1
        Assert-MockCalled Start-Process -Times 0 -Exactly -Scope It
    }
}
