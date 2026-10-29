# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

. (Join-Path $PSScriptRoot '../profile-windows-gui-build.ps1')

Describe 'Windows build profile observer' {
    BeforeEach {
        $script:temp = Join-Path ([IO.Path]::GetFullPath($env:TEMP)) ('pair-profile-test-' + [guid]::NewGuid().ToString('N'))
        [void](New-Item -ItemType Directory -Path $script:temp)
        $script:child = Join-Path $script:temp 'child.ps1'
        @'
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Values)
ConvertTo-Json -InputObject @($Values) -Compress
'@ | Set-Content -LiteralPath $script:child -Encoding UTF8
        $script:exe = (Get-Command powershell.exe).Source
    }
    AfterEach { Remove-Item -LiteralPath $script:temp -Recurse -Force -ErrorAction SilentlyContinue }

    It 'preserves original argument values and records timing and raw hashes' {
        $sourcePath=Join-Path $PSScriptRoot '../profile-windows-gui-build.ps1'
        $sourceHash=(Get-FileHash -LiteralPath $sourcePath).Hash
        $values = @('', 'two words', 'a"b', 'trailing slash \', 'slashes\\"quote')
        $args = [string[]](@('-NoProfile','-File',$script:child) + $values)
        $stage = [pscustomobject]@{name='argv';executable=$script:exe;arguments=$args;cwd=$script:temp;environment=[pscustomobject]@{};timeoutSeconds=30}
        $receipt = Invoke-ProfileStage $stage $script:temp -Snapshot { @() }
        $receipt.failure | Should BeNullOrEmpty
        $actual = Get-Content (Join-Path $script:temp 'argv.stdout') -Raw | ConvertFrom-Json
        $actual.Count | Should Be $values.Count
        for ($i=0; $i -lt $values.Count; $i++) { $actual[$i] | Should BeExactly $values[$i]; $receipt.arguments[$i+3] | Should BeExactly $values[$i] }
        $receipt.elapsedMilliseconds | Should BeGreaterThan 0
        ([DateTimeOffset]::Parse($receipt.endUtc) -ge [DateTimeOffset]::Parse($receipt.startUtc)) | Should Be $true
        foreach ($raw in $receipt.raw) {
            $path = Join-Path $script:temp $raw.name
            $raw.bytes | Should Be (Get-Item $path).Length
            $raw.sha256 | Should Be (Get-FileHash $path -Algorithm SHA256).Hash
        }
        (Get-FileHash -LiteralPath $sourcePath).Hash | Should BeExactly $sourceHash
        $receipt.observerProcessCpuSeconds | Should Not BeLessThan 0
    }

    It 'bounds raw output and reports overflow' {
        "[Console]::Out.Write(('x' * 200000))" | Set-Content -LiteralPath $script:child
        $stage = [pscustomobject]@{name='overflow';executable=$script:exe;arguments=[string[]]@('-NoProfile','-File',$script:child);cwd=$script:temp;environment=[pscustomobject]@{};timeoutSeconds=30}
        $receipt = Invoke-ProfileStage $stage $script:temp -StreamLimit 4096 -Snapshot { @() }
        $receipt.failure | Should Be 'raw stream limit exceeded'
        (Get-Item (Join-Path $script:temp 'overflow.stdout')).Length | Should Be 4096
    }

    It 'stops the recipe after a nonzero stage' {
        'exit 7' | Set-Content -LiteralPath $script:child
        $next = Join-Path $script:temp 'next.ps1'
        "[IO.File]::WriteAllText('$($script:temp.Replace("'","''"))\ran','yes')" | Set-Content -LiteralPath $next
        $steps = @('first','second') | ForEach-Object { [pscustomobject]@{name=$_;executable=$script:exe;arguments=[string[]]@('-NoProfile','-File',$(if ($_ -eq 'first') {$script:child} else {$next}));cwd=$script:temp;environment=[pscustomobject]@{};timeoutSeconds=30;requireCleanAfter=$false} }
        $recipe = [pscustomobject]@{expectedHead='fake';cacheState='test';toolchain=[pscustomobject]@{node=$script:exe};steps=$steps}
        $recipePath = Join-Path $script:temp 'recipe.json'
        $recipe | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $recipePath
        Mock Get-ProfileSourceState { [pscustomobject]@{head='fake';indexTree='fake';status=@();lockSha256='fake'} }
        Mock Get-ProfileToolchain { [pscustomobject]@{identity='fake';probeElapsedMilliseconds=0} }
        $evidence = Join-Path $script:temp 'evidence'
        { Invoke-WindowsGuiBuildProfile -Repository $PSScriptRoot -Vault (Join-Path $script:temp 'vault') -RecipePath $recipePath -EvidencePath $evidence -ExpectedRecipeSha256 (Get-FileHash $recipePath).Hash } | Should Throw
        (Test-Path (Join-Path $script:temp 'ran')) | Should Be $false
        (Test-Path (Join-Path $evidence 'second.json')) | Should Be $false
        $summary = Get-Content (Join-Path $evidence 'summary.json') -Raw | ConvertFrom-Json
        $summary.stages[0].exitCode | Should Be 7
        $summary.failure | Should Match 'child exited nonzero'
    }

    It 'selects descendants by creation identity and rejects recycled parents' {
        $root = [pscustomobject]@{pid=1;parentPid=0;createdUtc='2026-01-01T00:00:00Z'}
        $child = [pscustomobject]@{pid=2;parentPid=1;createdUtc='2026-01-01T00:00:01Z'}
        $old = [pscustomobject]@{pid=3;parentPid=1;createdUtc='2025-01-01T00:00:00Z'}
        $known = @{'1/2026-01-01T00:00:00Z'=$root}
        @(Update-ProfileProcessTree $known @($child,$old,$root)).Count | Should Be 2
        $reused = [pscustomobject]@{pid=1;parentPid=0;createdUtc='2026-02-01T00:00:00Z'}
        $newChild = [pscustomobject]@{pid=4;parentPid=1;createdUtc='2026-02-01T00:00:01Z'}
        @(Update-ProfileProcessTree $known @($reused,$newChild)).Count | Should Be 0
    }

    It 'reports gaps and unknown resource coverage' {
        $quality = Get-ProfileSampleQuality @() @([pscustomobject]@{reason='unavailable'})
        $quality.complete | Should Be $false
        $quality.gaps[0].reason | Should Be 'unavailable'
        $quality.shortLivedProcesses | Should Match 'unknown'
        $quality.physicalMemory | Should Match 'unknown'
        $quality.sampledPrivateBytesPeak | Should BeNullOrEmpty
    }

    It 'preserves UTF-8 recipe arguments and the declared environment in toolchain probes' {
        @'
param([string]$Value)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::Out.Write($Value + '|' + $env:PAIR_PROFILE_SYNTHETIC)
'@ | Set-Content -LiteralPath $script:child -Encoding UTF8
        $unicode = -join ([char]0x65e5,[char]0x672c,[char]0xe9)
        $step=[pscustomobject]@{name='unicode';executable=$script:exe;arguments=@('-NoProfile','-File',$script:child,$unicode)
            cwd=$script:temp;environment=[pscustomobject]@{PATH='synthetic-declared-path';PAIR_PROFILE_SYNTHETIC='isolated'};timeoutSeconds=30;requireCleanAfter=$false}
        $recipe=[pscustomobject]@{expectedHead='fake';cacheState='synthetic';toolchain=[pscustomobject]@{node=$script:exe};inputs=@();steps=@($step)}
        $recipePath=Join-Path $script:temp 'utf8.json'
        [IO.File]::WriteAllText($recipePath,($recipe | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
        Mock Get-ProfileSourceState { [pscustomobject]@{head='fake';indexTree='fake';status=@();lockSha256='fake'} }
        Mock Get-ProfileToolchain {
            param($Toolchain,$Evidence,$Prefix,$WorkingDirectory,$Environment)
            $script:probedPath=$Environment.PATH
            $script:probedMarker=$Environment.PAIR_PROFILE_SYNTHETIC
            [pscustomobject]@{identity='synthetic';probeElapsedMilliseconds=0}
        }
        $parentPath=$env:PATH
        $evidence=Join-Path $script:temp 'unicode-evidence'
        Invoke-WindowsGuiBuildProfile -Repository $PSScriptRoot -Vault (Join-Path $script:temp 'vault') -RecipePath $recipePath -EvidencePath $evidence -ExpectedRecipeSha256 (Get-FileHash $recipePath).Hash | Out-Null
        (Get-Content -LiteralPath (Join-Path $evidence 'unicode.stdout') -Raw -Encoding UTF8) | Should BeExactly ($unicode+'|isolated')
        $script:probedPath | Should BeExactly ((Split-Path $script:exe)+';synthetic-declared-path')
        $script:probedMarker | Should BeExactly 'isolated'
        $env:PATH | Should BeExactly $parentPath
        $env:PAIR_PROFILE_SYNTHETIC | Should BeNullOrEmpty
    }

    It 'rejects every provenance artifact collision before execution' {
        Mock Get-ProfileSourceState { [pscustomobject]@{head='fake';indexTree='fake';status=@();lockSha256='fake'} }
        foreach ($name in @('recipe','summary','source-before','check-first-node','check-first-npm')) {
            $steps=@('first',$name) | ForEach-Object { [pscustomobject]@{name=$_;executable=$script:exe;arguments=@('-NoProfile','-File',$script:child)
                cwd=$script:temp;environment=[pscustomobject]@{};timeoutSeconds=30;requireCleanAfter=$false} }
            $recipe=[pscustomobject]@{expectedHead='fake';cacheState='synthetic';toolchain=@{node=$script:exe};inputs=@();steps=$steps}
            $path=Join-Path $script:temp ($name+'.json')
            $recipe | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $path -Encoding UTF8
            { Invoke-WindowsGuiBuildProfile -Repository $PSScriptRoot -Vault (Join-Path $script:temp 'vault') -RecipePath $path -EvidencePath (Join-Path $script:temp ('out-'+$name)) -ExpectedRecipeSha256 (Get-FileHash $path).Hash } | Should Throw 'artifact namespace collision'
            (Test-Path -LiteralPath (Join-Path $script:temp ('out-'+$name))) | Should Be $false
        }
    }

    It 'keeps CPU intervals bound to creation identity instead of lifetime sums' {
        $previous=@{}
        $first=[pscustomobject]@{pid=1;createdUtc='2026-01-01T00:00:00Z';image='synthetic';cpuSeconds=10}
        $second=[pscustomobject]@{pid=1;createdUtc=$first.createdUtc;image='synthetic';cpuSeconds=10.5}
        $reuse=[pscustomobject]@{pid=1;createdUtc='2026-01-02T00:00:00Z';image='synthetic';cpuSeconds=2}
        @(Get-ProfileCpuDeltas $previous @($first) 250)[0].deltaSeconds | Should BeNullOrEmpty
        $interval=@(Get-ProfileCpuDeltas $previous @($second) 250)[0]
        $interval.deltaSeconds | Should Be 0.5
        $interval.cpuPercentOneCore | Should Be 200
        @(Get-ProfileCpuDeltas $previous @() 250).Count | Should Be 0
        @(Get-ProfileCpuDeltas $previous @($second) 250)[0].deltaSeconds | Should BeNullOrEmpty
        @(Get-ProfileCpuDeltas $previous @($reuse) 250)[0].deltaSeconds | Should BeNullOrEmpty
    }

    It 'contains a surviving descendant even when all process samples miss its parent' {
        $grand=Join-Path $script:temp 'grand.ps1'
        $middle=Join-Path $script:temp 'middle.ps1'
        $identity=Join-Path $script:temp 'grand.json'
        $escapedIdentity=$identity.Replace("'","''")
        @"
[pscustomobject]@{pid=`$PID;createdUtc=(Get-Process -Id `$PID).StartTime.ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath '$escapedIdentity'
Start-Sleep -Seconds 30
"@ | Set-Content -LiteralPath $grand -Encoding UTF8
        "Start-Process -WindowStyle Hidden -FilePath '$($script:exe.Replace("'","''"))' -ArgumentList @('-NoProfile','-File','`"$($grand.Replace("'","''"))`"') | Out-Null" | Set-Content -LiteralPath $middle -Encoding UTF8
        @"
`$p=Start-Process -WindowStyle Hidden -PassThru -FilePath '$($script:exe.Replace("'","''"))' -ArgumentList @('-NoProfile','-File','`"$($middle.Replace("'","''"))`"')
`$p.WaitForExit()
`$deadline=[DateTime]::UtcNow.AddSeconds(10)
while (!(Test-Path -LiteralPath '$escapedIdentity')) { if ([DateTime]::UtcNow -gt `$deadline) {exit 9}; Start-Sleep -Milliseconds 20 }
"@ | Set-Content -LiteralPath $script:child -Encoding UTF8
        try {
            $stage=[pscustomobject]@{name='descendant';executable=$script:exe;arguments=@('-NoProfile','-File',$script:child);cwd=$script:temp;environment=[pscustomobject]@{};timeoutSeconds=15}
            $receipt=Invoke-ProfileStage $stage $script:temp -Snapshot { @() }
            $receipt.failure | Should Be 'descendant survived stage'
            (Test-Path -LiteralPath $identity) | Should Be $true
            $record=Get-Content -LiteralPath $identity -Raw | ConvertFrom-Json
            $live=Get-Process -Id $record.pid -ErrorAction SilentlyContinue
            if ($live) { $live.StartTime.ToUniversalTime().ToString('o') | Should Not Be $record.createdUtc }
        } finally {
            if (Test-Path -LiteralPath $identity) {
                $record=Get-Content -LiteralPath $identity -Raw | ConvertFrom-Json
                $live=Get-Process -Id $record.pid -ErrorAction SilentlyContinue
                if ($live -and $live.StartTime.ToUniversalTime().ToString('o') -ceq $record.createdUtc) { $live.Kill();[void]$live.WaitForExit(5000) }
            }
        }
    }

    It 'rejects evidence inside the repository or vault' {
        { Resolve-ProfileEvidencePath (Join-Path $PSScriptRoot 'evidence') $PSScriptRoot $script:temp } | Should Throw
        { Resolve-ProfileEvidencePath (Join-Path $script:temp 'evidence') $PSScriptRoot $script:temp } | Should Throw
        Resolve-ProfileEvidencePath (Join-Path $script:temp 'external') $PSScriptRoot (Join-Path $script:temp 'vault') | Should BeExactly (Join-Path $script:temp 'external')
    }

    It 'retains metric failure diagnostics and never starts dependent work' {
        'Start-Sleep -Seconds 10' | Set-Content -LiteralPath $script:child
        $steps=@('metrics','dependent') | ForEach-Object { [pscustomobject]@{name=$_;executable=$script:exe
            arguments=@('-NoProfile','-File',$script:child);cwd=$script:temp;environment=[pscustomobject]@{};timeoutSeconds=30;requireCleanAfter=$false} }
        $recipe=[pscustomobject]@{expectedHead='fake';cacheState='synthetic';toolchain=@{node=$script:exe};inputs=@();steps=$steps}
        $path=Join-Path $script:temp 'metric-recipe.json'
        $recipe | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $path -Encoding UTF8
        Mock Get-ProfileSourceState { [pscustomobject]@{head='fake';indexTree='fake';status=@();lockSha256='fake'} }
        Mock Get-ProfileToolchain { [pscustomobject]@{identity='synthetic';probeElapsedMilliseconds=0} }
        Mock Get-ProfileProcessSnapshot { throw 'synthetic metric permission denied' }
        $evidence=Join-Path $script:temp 'metric-evidence'
        { Invoke-WindowsGuiBuildProfile -Repository $PSScriptRoot -Vault (Join-Path $script:temp 'vault') -RecipePath $path -EvidencePath $evidence -ExpectedRecipeSha256 (Get-FileHash $path).Hash } | Should Throw 'process metrics unavailable'
        $receipt=Get-Content -LiteralPath (Join-Path $evidence 'metrics.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        $receipt.quality.gaps[0].reason | Should Match 'synthetic metric permission denied'
        (Test-Path -LiteralPath (Join-Path $evidence 'dependent.json')) | Should Be $false
        (Get-Content -LiteralPath (Join-Path $evidence 'summary.json') -Raw -Encoding UTF8 | ConvertFrom-Json).failure | Should Match 'process metrics unavailable'
    }
}
