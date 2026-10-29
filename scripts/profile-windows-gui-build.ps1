# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file defines helpers only. Dot-sourcing it never starts a process.

function ConvertTo-ProfileWindowsArgument {
    param([Parameter(Mandatory)][AllowEmptyString()][string]$Value)
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    $result = '"'
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') { $slashes++; continue }
        if ($character -eq '"') {
            $result += ('\' * (2 * $slashes + 1)) + '"'
        } else {
            $result += ('\' * $slashes) + $character
        }
        $slashes = 0
    }
    $result + ('\' * (2 * $slashes)) + '"'
}

function Resolve-ProfileEvidencePath {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Repository,
        [Parameter(Mandatory)][string]$Vault)
    if ($Path -notmatch '^[A-Za-z]:[\\/]') { throw 'Evidence must be on a local drive.' }
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    if (([IO.DriveInfo]::new([IO.Path]::GetPathRoot($full))).DriveType -ne [IO.DriveType]::Fixed) { throw 'Evidence drive must be fixed.' }
    foreach ($excluded in @($Repository, $Vault)) {
        if ([string]::IsNullOrWhiteSpace($excluded)) { throw 'Repository and vault roots are required.' }
        $root = [IO.Path]::GetFullPath($excluded).TrimEnd('\', '/')
        if ($full.Equals($root, [StringComparison]::OrdinalIgnoreCase) -or $full.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Evidence is inside an excluded root.' }
    }
    $current = $full
    while ($current) {
        if (Test-Path -LiteralPath (Join-Path $current '.git')) { throw 'Evidence is inside a repository.' }
        if ((Test-Path -LiteralPath $current) -and (([IO.File]::GetAttributes($current) -band [IO.FileAttributes]::ReparsePoint) -ne 0)) { throw 'Evidence path contains a reparse point.' }
        $current = [IO.Path]::GetDirectoryName($current)
    }
    return $full
}

function Initialize-ProfileStreamCapture {
    if ('PairBuildProfile.Capture' -as [type]) { return }
    # Separate byte pumps avoid pipe deadlock and preserve exact raw bytes. The
    # bound applies before writing, independently to each stream.
    Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Threading.Tasks;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
namespace PairBuildProfile {
    // Start suspended, contain, then resume. Sampling is never the owner of
    // descendants, including those whose intermediate parent exits unseen.
    public sealed class Child : IDisposable {
        [StructLayout(LayoutKind.Sequential)] struct Security { public int Length; public IntPtr Descriptor; public int Inherit; }
        [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)] struct Startup {
            public int Size; public string Reserved, Desktop, Title;
            public int X,Y,XSize,YSize,XChars,YChars,Fill,Flags; public short Show,ReservedSize;
            public IntPtr ReservedBytes, Input, Output, Error;
        }
        [StructLayout(LayoutKind.Sequential)] struct ProcessInfo { public IntPtr Process,Thread; public int Pid,Tid; }
        [StructLayout(LayoutKind.Sequential)] struct Limits {
            public long ProcessTime, JobTime; public uint Flags; public UIntPtr MinWorking,MaxWorking;
            public uint ActiveLimit; public UIntPtr Affinity; public uint Priority,Scheduling;
        }
        [StructLayout(LayoutKind.Sequential)] struct Io { public ulong A,B,C,D,E,F; }
        [StructLayout(LayoutKind.Sequential)] struct Extended { public Limits Basic; public Io Counters; public UIntPtr ProcessMemory,JobMemory,PeakProcess,PeakJob; }
        [StructLayout(LayoutKind.Sequential)] struct Accounting {
            public long User,Kernel,PeriodUser,PeriodKernel; public uint Faults,Total,Active,Terminated;
        }
        [DllImport("kernel32.dll",SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr a,string n);
        [DllImport("kernel32.dll",SetLastError=true)] static extern bool SetInformationJobObject(IntPtr j,int c,ref Extended i,int l);
        [DllImport("kernel32.dll",SetLastError=true)] static extern bool QueryInformationJobObject(IntPtr j,int c,out Accounting i,int l,IntPtr n);
        [DllImport("kernel32.dll",SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr j,IntPtr p);
        [DllImport("kernel32.dll",SetLastError=true)] static extern bool TerminateJobObject(IntPtr j,uint c);
        [DllImport("kernel32.dll",SetLastError=true)] static extern bool TerminateProcess(IntPtr p,uint c);
        [DllImport("kernel32.dll",SetLastError=true)] static extern uint WaitForSingleObject(IntPtr p,uint ms);
        [DllImport("kernel32.dll",SetLastError=true)] static extern bool GetExitCodeProcess(IntPtr p,out uint code);
        [DllImport("kernel32.dll",SetLastError=true)] static extern uint ResumeThread(IntPtr t);
        [DllImport("kernel32.dll",SetLastError=true)] static extern bool CloseHandle(IntPtr h);
        [DllImport("kernel32.dll",SetLastError=true)] static extern bool CreatePipe(out IntPtr r,out IntPtr w,ref Security s,int n);
        [DllImport("kernel32.dll",SetLastError=true)] static extern bool SetHandleInformation(IntPtr h,uint mask,uint flags);
        [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern IntPtr CreateFile(string n,uint a,uint s,ref Security sec,uint d,uint f,IntPtr t);
        [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern bool CreateProcess(string app,System.Text.StringBuilder cmd,IntPtr pa,IntPtr ta,bool inherit,uint flags,IntPtr env,string cwd,ref Startup si,out ProcessInfo pi);
        IntPtr job, processHandle;
        public Process Process;
        public Stream Output, Error;
        public Child(ProcessStartInfo info, string commandLine) {
            IntPtr readOut=IntPtr.Zero,writeOut=IntPtr.Zero,readErr=IntPtr.Zero,writeErr=IntPtr.Zero,input=IntPtr.Zero,env=IntPtr.Zero;
            ProcessInfo pi = new ProcessInfo();
            try {
                job=CreateJobObject(IntPtr.Zero,null); if(job==IntPtr.Zero) Fail();
                var limits=new Extended(); limits.Basic.Flags=0x2000;
                if(!SetInformationJobObject(job,9,ref limits,Marshal.SizeOf(limits))) Fail();
                var security=new Security { Length=Marshal.SizeOf(typeof(Security)), Inherit=1 };
                if(!CreatePipe(out readOut,out writeOut,ref security,0) || !CreatePipe(out readErr,out writeErr,ref security,0)) Fail();
                if(!SetHandleInformation(readOut,1,0) || !SetHandleInformation(readErr,1,0)) Fail();
                input=CreateFile("NUL",0x80000000,3,ref security,3,0,IntPtr.Zero);
                if(input==new IntPtr(-1)) Fail();
                var keys=new System.Collections.Generic.List<string>();
                foreach(string key in info.EnvironmentVariables.Keys) keys.Add(key);
                keys.Sort(StringComparer.OrdinalIgnoreCase);
                var block=new System.Text.StringBuilder();
                foreach(string key in keys) block.Append(key).Append('=').Append(info.EnvironmentVariables[key]).Append('\0');
                block.Append('\0'); env=Marshal.StringToHGlobalUni(block.ToString());
                var startup=new Startup { Size=Marshal.SizeOf(typeof(Startup)), Flags=0x100, Input=input, Output=writeOut, Error=writeErr };
                if(!CreateProcess(info.FileName,new System.Text.StringBuilder(commandLine),IntPtr.Zero,IntPtr.Zero,true,0x08000404,env,info.WorkingDirectory,ref startup,out pi)) Fail();
                if(!AssignProcessToJobObject(job,pi.Process)) Fail();
                Process=Process.GetProcessById(pi.Pid);
                processHandle=pi.Process; pi.Process=IntPtr.Zero;
                Output=new FileStream(new SafeFileHandle(readOut,true),FileAccess.Read); readOut=IntPtr.Zero;
                Error=new FileStream(new SafeFileHandle(readErr,true),FileAccess.Read); readErr=IntPtr.Zero;
                if(ResumeThread(pi.Thread)==0xffffffff) Fail();
            } catch { if(pi.Process!=IntPtr.Zero) TerminateProcess(pi.Process,1); Dispose(); throw; }
            finally {
                foreach(var handle in new[]{readOut,writeOut,readErr,writeErr,input,pi.Process,pi.Thread})
                    if(handle!=IntPtr.Zero && handle!=new IntPtr(-1)) CloseHandle(handle);
                if(env!=IntPtr.Zero) Marshal.FreeHGlobal(env);
            }
        }
        static void Fail() { throw new Win32Exception(Marshal.GetLastWin32Error()); }
        public uint Active {
            get { Accounting value; if(!QueryInformationJobObject(job,1,out value,Marshal.SizeOf(typeof(Accounting)),IntPtr.Zero)) Fail(); return value.Active; }
        }
        public bool Exited { get { return WaitForSingleObject(processHandle,0)==0; } }
        public bool Wait(int milliseconds) { return WaitForSingleObject(processHandle,(uint)milliseconds)==0; }
        public uint ExitCode { get { uint code; if(!GetExitCodeProcess(processHandle,out code)) Fail(); return code; } }
        public void Stop() {
            if(job==IntPtr.Zero) return;
            if(!TerminateJobObject(job,1)) Fail();
            var clock=Stopwatch.StartNew();
            while(Active!=0) { if(clock.ElapsedMilliseconds>5000) throw new IOException("Job did not become empty."); System.Threading.Thread.Sleep(10); }
        }
        public void Dispose() {
            if(job!=IntPtr.Zero) { CloseHandle(job); job=IntPtr.Zero; }
            if(processHandle!=IntPtr.Zero) { CloseHandle(processHandle); processHandle=IntPtr.Zero; }
            if(Output!=null) Output.Dispose(); if(Error!=null) Error.Dispose();
            if(Process!=null) Process.Dispose();
        }
    }
    public sealed class Capture {
        public volatile bool Overflow;
        public volatile string Failure;
        public Task Completion;
        public Capture(Stream source, string path, long limit) {
            Completion = Task.Run(async () => {
                try {
                    using (var output = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read)) {
                        var buffer = new byte[8192]; long written = 0; int size;
                        while ((size = await source.ReadAsync(buffer, 0, buffer.Length)) != 0) {
                            int keep = (int)Math.Min(size, limit - written);
                            if (keep > 0) { await output.WriteAsync(buffer, 0, keep); written += keep; }
                            if (keep != size) Overflow = true;
                        }
                    }
                } catch (Exception error) { Failure = error.GetType().Name; }
            });
        }
    }
}
'@
}

function Get-ProfileProcessSnapshot {
    @(Get-CimInstance Win32_Process -OperationTimeoutSec 1 -ErrorAction Stop -Property ProcessId,
        ParentProcessId, CreationDate, ExecutablePath, KernelModeTime, UserModeTime,
        PrivatePageCount, WorkingSetSize, Priority | ForEach-Object {
        [pscustomobject]@{
            pid = [int]$_.ProcessId; parentPid = [int]$_.ParentProcessId
            createdUtc = $(if ($_.CreationDate) { $_.CreationDate.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.ffffff'Z'") } else { '' })
            image = [string]$_.ExecutablePath
            cpuSeconds = ([double]$_.KernelModeTime + [double]$_.UserModeTime) / 10000000
            privateBytes = [long]$_.PrivatePageCount; workingSetBytes = [long]$_.WorkingSetSize
            priority = [int]$_.Priority
        }
    })
}

function Update-ProfileProcessTree {
    param([hashtable]$Known, [array]$Snapshot)
    # Parent PID alone is insufficient. A parent's observed lifetime must cover
    # the child's creation, and a recycled parent PID must never gain children.
    $selected = @{}
    foreach ($item in $Snapshot) {
        if (!$item.createdUtc) { continue }
        $key = '{0}/{1}' -f $item.pid, $item.createdUtc
        if ($Known.ContainsKey($key)) { $selected[$key] = $item }
    }
    do {
        $added = $false
        foreach ($item in $Snapshot) {
            if (!$item.createdUtc) { continue }
            $key = '{0}/{1}' -f $item.pid, $item.createdUtc
            if ($selected.ContainsKey($key)) { continue }
            foreach ($parent in @($selected.Values)) {
                if ($item.parentPid -eq $parent.pid -and
                    [DateTimeOffset]::Parse($item.createdUtc) -ge [DateTimeOffset]::Parse($parent.createdUtc)) {
                    $item | Add-Member -NotePropertyName parentIdentity -NotePropertyValue ('{0}/{1}' -f $parent.pid,$parent.createdUtc) -Force
                    $selected[$key] = $item; $Known[$key] = $item; $added = $true; break
                }
            }
        }
    } while ($added)
    foreach ($key in @($selected.Keys)) { $Known[$key] = $selected[$key] }
    @($selected.Values)
}

function Get-ProfileSampleQuality {
    param([array]$Samples, [array]$Gaps)
    [pscustomobject]@{
        intervalMilliseconds = 250; peaks = 'sampled lower bounds'
        shortLivedProcesses = 'unknown: processes may start and exit between samples'
        complete = $false
        gaps = @($Gaps)
        sampledPrivateBytesPeak = $(if ($Samples.Count) { ($Samples | Measure-Object privateBytes -Maximum).Maximum } else { $null })
        sampledWorkingSetBytesPeak = $(if ($Samples.Count) { ($Samples | Measure-Object workingSetBytes -Maximum).Maximum } else { $null })
        physicalMemory = 'unknown: working sets can share pages'
    }
}

function Write-ProfileJson {
    param([string]$Path, $Value)
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($false))
}

function Get-ProfileCpuDeltas {
    param([hashtable]$Previous, [array]$Current, [double]$IntervalMilliseconds)
    $seen=@{}
    foreach ($item in $Current) { if ($item.createdUtc) { $seen['{0}/{1}' -f $item.pid,$item.createdUtc]=$true } }
    foreach ($key in @($Previous.Keys)) { if (!$seen.ContainsKey($key)) { $Previous.Remove($key) } }
    foreach ($item in $Current) {
        $key = '{0}/{1}' -f $item.pid,$item.createdUtc
        $delta = $null
        if ($item.createdUtc -and $Previous.ContainsKey($key) -and $IntervalMilliseconds -gt 0) {
            $difference = $item.cpuSeconds - $Previous[$key]
            if ($difference -ge 0) { $delta = $difference }
        }
        [pscustomobject]@{identity=$key;image=$item.image;cpuSeconds=$item.cpuSeconds;deltaSeconds=$delta
            cpuPercentOneCore=$(if ($null -ne $delta) { $delta * 100000 / $IntervalMilliseconds } else { $null })}
        if ($item.createdUtc) { $Previous[$key] = $item.cpuSeconds }
    }
}

function Invoke-ProfileStage {
    param([Parameter(Mandatory)]$Stage, [Parameter(Mandatory)][string]$Evidence,
        [ValidateRange(1,16777216)][long]$StreamLimit = 16777216,
        [scriptblock]$Snapshot = { Get-ProfileProcessSnapshot })
    $ErrorActionPreference='Stop'
    Initialize-ProfileStreamCapture
    $start = [DateTimeOffset]::UtcNow
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $observer = [Diagnostics.Stopwatch]::new()
    $self = [Diagnostics.Process]::GetCurrentProcess()
    $selfCpu = $self.TotalProcessorTime.TotalSeconds
    $process = $null; $child = $null
    $previousCpu = @{}; $observations=0
    $known = @{}; $samples = [Collections.Generic.List[object]]::new()
    $gaps = [Collections.Generic.List[object]]::new()
    $failure = $null; $exitCode = $null; $rootCpu = $null
    $stdout = $null; $stderr = $null
    $outPath = Join-Path $Evidence ($Stage.name + '.stdout')
    $errPath = Join-Path $Evidence ($Stage.name + '.stderr')
    try {
        $info = [Diagnostics.ProcessStartInfo]::new()
        $info.FileName = $Stage.executable
        $info.Arguments = (@($Stage.arguments | ForEach-Object { ConvertTo-ProfileWindowsArgument $_ }) -join ' ')
        $info.WorkingDirectory = $Stage.cwd
        $info.UseShellExecute = $false; $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
        foreach ($property in $Stage.environment.PSObject.Properties) {
            if ($null -eq $property.Value) { $info.EnvironmentVariables.Remove($property.Name) }
            else { $info.EnvironmentVariables[$property.Name] = [string]$property.Value }
        }
        $commandLine = (ConvertTo-ProfileWindowsArgument $info.FileName) + ' ' + $info.Arguments
        $child = [PairBuildProfile.Child]::new($info,$commandLine)
        $process = $child.Process
        $root = [pscustomobject]@{pid=$process.Id; parentPid=$PID; createdUtc=$process.StartTime.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.ffffff'Z'")
            image=$Stage.executable; cpuSeconds=0; privateBytes=0; workingSetBytes=0; priority=$null}
        $known['{0}/{1}' -f $root.pid,$root.createdUtc] = $root
        $stdout = [PairBuildProfile.Capture]::new($child.Output, $outPath, $StreamLimit)
        $stderr = [PairBuildProfile.Capture]::new($child.Error, $errPath, $StreamLimit)
        $last = 0.0; $nextSample = 0.0
        while ($true) {
            if ($clock.Elapsed.TotalMilliseconds -ge $nextSample) {
                $now = $clock.Elapsed.TotalMilliseconds
                if ($now - $last -gt 500) { $gaps.Add([pscustomobject]@{atMilliseconds=$now;intervalMilliseconds=$now-$last;reason='sampling delayed'}) }
                $observer.Start()
                try {
                    $all = @(& $Snapshot)
                    $observations += $all.Count
                    if ($all.Count -gt 4096 -or $observations -gt 1000000) { throw 'process observation bound exceeded' }
                    $owned = @(Update-ProfileProcessTree $known $all)
                    $cpu = @(Get-ProfileCpuDeltas $previousCpu $all ($now-$last))
                    if ($previousCpu.Count -gt 65536) { throw 'CPU identity bound exceeded' }
                    if ($samples.Count -ge 8192 -or $known.Count -gt 4096) { throw 'process sample bound exceeded' }
                    $samples.Add([pscustomobject]@{milliseconds=$now;processes=$owned
                        privateBytes=($owned | Measure-Object privateBytes -Sum).Sum
                        workingSetBytes=($owned | Measure-Object workingSetBytes -Sum).Sum
                        cpu=$cpu;competing=$all.Count-$owned.Count
                        ownedIdentities=@($owned | ForEach-Object { '{0}/{1}' -f $_.pid,$_.createdUtc })})
                } catch {
                    if ($gaps.Count -lt 8192) { $gaps.Add([pscustomobject]@{atMilliseconds=$now;reason=$_.Exception.Message}) }
                    $failure='process metrics unavailable'
                    $previousCpu.Clear()
                    if ($samples.Count -ge 8192 -or $known.Count -gt 4096 -or $previousCpu.Count -gt 65536 -or
                        $all.Count -gt 4096 -or $observations -gt 1000000) { $failure='process sample bound exceeded' }
                }
                finally { $observer.Stop() }
                $last = $now; $nextSample = $now + 250
            }
            if ($failure) { break }
            if ($stdout.Overflow -or $stderr.Overflow) { $failure = 'raw stream limit exceeded'; break }
            if ($stdout.Failure -or $stderr.Failure) { $failure = 'raw stream capture failed'; break }
            if ($clock.Elapsed.TotalSeconds -gt $Stage.timeoutSeconds) { $failure = 'stage deadline exceeded'; break }
            if ($child.Exited) {
                if ($child.Active -gt 0) { $failure='descendant survived stage';break }
                if ($stdout.Completion.IsCompleted -and $stderr.Completion.IsCompleted) { break }
            }
            if ($child.Exited) { Start-Sleep -Milliseconds 25 }
            else { [void]$child.Wait(25) }
        }
        if (!$child.Exited -or $failure) { $child.Stop();[void]$child.Wait(5000) }
        if (!$stdout.Completion.Wait(5000) -or !$stderr.Completion.Wait(5000)) { $failure = 'stream drain deadline exceeded' }
        if ($stdout.Overflow -or $stderr.Overflow) { $failure = 'raw stream limit exceeded' }
        if ($stdout.Failure -or $stderr.Failure) { $failure = 'raw stream capture failed' }
        if ($child.Exited) { $exitCode = $child.ExitCode; $rootCpu = $process.TotalProcessorTime.TotalSeconds }
        if ($exitCode -ne 0 -and !$failure) { $failure = 'child exited nonzero' }
    } catch { $failure = $_.Exception.Message; if ($child) { $child.Stop() } }
    finally {
        if ($child) { $child.Dispose() }
        $clock.Stop();$self.Refresh()
        $observerCpu = $self.TotalProcessorTime.TotalSeconds - $selfCpu
        $self.Dispose()
    }
    $records = @($known.Values)
    $receipt = [pscustomobject]@{
        name=$Stage.name; executable=$Stage.executable; executableSha256=(Get-FileHash -LiteralPath $Stage.executable -Algorithm SHA256).Hash
        arguments=@($Stage.arguments); cwd=$Stage.cwd; environmentDelta=$Stage.environment
        startUtc=$start.ToString('o');endUtc=[DateTimeOffset]::UtcNow.ToString('o');elapsedMilliseconds=$clock.Elapsed.TotalMilliseconds
        exitCode=$exitCode;failure=$failure;rootCpuSeconds=$rootCpu
        observedTreeCpuSecondsLowerBound=($records | Measure-Object cpuSeconds -Sum).Sum
        observerSamplingMilliseconds=$observer.Elapsed.TotalMilliseconds
        observerProcessCpuSeconds=$observerCpu
        observerCpuScope='whole host process delta, including pumps; other work in this process is a confounder'
        processedBytesAndObjects='unknown unless supplied by the canonical stage receipt'
        quality=(Get-ProfileSampleQuality @($samples.ToArray()) @($gaps.ToArray()))
        processes=$records;samples=@($samples.ToArray())
        raw=@(foreach ($path in @($outPath,$errPath)) { if (Test-Path -LiteralPath $path) {
            [pscustomobject]@{name=[IO.Path]::GetFileName($path);bytes=(Get-Item -LiteralPath $path).Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash}
        } })
    }
    Write-ProfileJson (Join-Path $Evidence ($Stage.name + '.json')) $receipt
    $receipt
}

function Get-ProfileSourceState {
    param([string]$Repository)
    $ErrorActionPreference='Stop'
    $head = & git --no-optional-locks -C $Repository rev-parse HEAD
    if ($LASTEXITCODE) { throw 'Cannot resolve source HEAD.' }
    $indexPath = & git --no-optional-locks -C $Repository rev-parse --path-format=absolute --git-path index
    if ($LASTEXITCODE) { throw 'Cannot locate source index.' }
    $index = (Get-FileHash -LiteralPath $indexPath -Algorithm SHA256).Hash
    $status = @(& git --no-optional-locks -c core.fsmonitor=false -C $Repository status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE) { throw 'Cannot inspect source cleanliness.' }
    $lock = Join-Path $Repository 'desktop/package-lock.json'
    [pscustomobject]@{head=[string]$head;indexTree=[string]$index;status=$status
        lockSha256=(Get-FileHash -LiteralPath $lock -Algorithm SHA256).Hash}
}

function Get-ProfileToolchain {
    param($Toolchain, [string]$Evidence, [string]$Prefix, [string]$WorkingDirectory, $Environment)
    $ErrorActionPreference='Stop'
    if (![IO.Path]::IsPathRooted($Toolchain.node) -or ![IO.Path]::IsPathRooted($Toolchain.npmCli) -or
        !(Test-Path -LiteralPath $Toolchain.node -PathType Leaf) -or !(Test-Path -LiteralPath $Toolchain.npmCli -PathType Leaf)) {
        throw 'Toolchain needs absolute existing portable node and npm-cli paths.'
    }
    $node = Invoke-ProfileStage ([pscustomobject]@{name="$Prefix-node";executable=$Toolchain.node
        arguments=@('-p','JSON.stringify({version:process.version,execPath:process.execPath,arch:process.arch})')
        cwd=$WorkingDirectory;environment=$Environment;timeoutSeconds=30}) $Evidence
    if ($node.failure) { throw 'Node toolchain probe failed.' }
    $actual = Get-Content -LiteralPath (Join-Path $Evidence "$Prefix-node.stdout") -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($actual.version -cne 'v26.8.1' -or $actual.arch -cne 'x64' -or
        ![string]::Equals($actual.execPath,$Toolchain.node,[StringComparison]::OrdinalIgnoreCase)) { throw 'Portable Node identity/version mismatch.' }
    $npm = Invoke-ProfileStage ([pscustomobject]@{name="$Prefix-npm";executable=$Toolchain.node
        arguments=@($Toolchain.npmCli,'--version');cwd=$WorkingDirectory;environment=$Environment;timeoutSeconds=30}) $Evidence
    if ($npm.failure -or (Get-Content -LiteralPath (Join-Path $Evidence "$Prefix-npm.stdout") -Raw -Encoding UTF8).Trim() -cne '11.19.0') { throw 'Portable npm version mismatch.' }
    [pscustomobject]@{node=$actual;nodeSha256=(Get-FileHash -LiteralPath $Toolchain.node -Algorithm SHA256).Hash
        npmVersion='11.19.0';npmCliSha256=(Get-FileHash -LiteralPath $Toolchain.npmCli -Algorithm SHA256).Hash
        probeElapsedMilliseconds=$node.elapsedMilliseconds+$npm.elapsedMilliseconds}
}

function Invoke-WindowsGuiBuildProfile {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Repository, [Parameter(Mandatory)][string]$Vault,
        [Parameter(Mandatory)][string]$RecipePath, [Parameter(Mandatory)][string]$EvidencePath,
        [Parameter(Mandatory)][string]$ExpectedRecipeSha256)
    $ErrorActionPreference='Stop'
    $evidence = Resolve-ProfileEvidencePath $EvidencePath $Repository $Vault
    if (Test-Path -LiteralPath $evidence) { throw 'Evidence directory must be new.' }
    if (!(Test-Path -LiteralPath $RecipePath -PathType Leaf) -or (Get-Item -LiteralPath $RecipePath).Length -gt 1048576) { throw 'Recipe must be an existing file of at most 1 MiB.' }
    $recipeHash = (Get-FileHash -LiteralPath $RecipePath -Algorithm SHA256).Hash
    if ($recipeHash -cne $ExpectedRecipeSha256.ToUpperInvariant()) { throw 'Recipe hash differs from the released recipe.' }
    $recipe = Get-Content -LiteralPath $RecipePath -Raw -Encoding UTF8 | ConvertFrom-Json
    $source = Get-ProfileSourceState $Repository
    if ($source.head -cne $recipe.expectedHead -or $source.status.Count) { throw 'Source must be exactly the released clean revision.' }
    if (!$recipe.cacheState -or !$recipe.toolchain -or !$recipe.steps.Count) { throw 'Recipe needs cache state, toolchain and steps.' }
    foreach ($inputFile in @($recipe.inputs)) {
        if ($null -eq $inputFile) { continue }
        if (![IO.Path]::IsPathRooted($inputFile.path) -or !(Test-Path -LiteralPath $inputFile.path -PathType Leaf) -or
            (Get-FileHash -LiteralPath $inputFile.path -Algorithm SHA256).Hash -cne $inputFile.sha256.ToUpperInvariant()) {
            throw 'Frozen recipe input differs.'
        }
    }
    $names = @{}
    $artifacts = @{recipe=$true;summary=$true;'source-before'=$true}
    foreach ($stage in $recipe.steps) {
        if ($stage.name -notmatch '^[a-z0-9][a-z0-9-]{0,63}$' -or $names.ContainsKey($stage.name)) { throw 'Invalid or repeated stage name.' }
        $names[$stage.name]=$true
        foreach ($artifact in @($stage.name,('check-'+$stage.name+'-node'),('check-'+$stage.name+'-npm'),($stage.name+'.source'))) {
            if ($artifacts.ContainsKey($artifact)) { throw 'Stage artifact namespace collision.' }
            $artifacts[$artifact]=$true
        }
        if (![IO.Path]::IsPathRooted($stage.executable) -or !(Test-Path -LiteralPath $stage.executable -PathType Leaf) -or
            ![IO.Path]::IsPathRooted($stage.cwd) -or !(Test-Path -LiteralPath $stage.cwd -PathType Container) -or
            $stage.arguments -isnot [array] -or $stage.timeoutSeconds -lt 1 -or $stage.timeoutSeconds -gt 7200 -or
            $null -eq $stage.environment) { throw 'Stage requires executable, cwd, arguments, environment delta and bounded deadline.' }
        foreach ($arg in $stage.arguments) { if ($arg -isnot [string] -or $arg.Contains([char]0)) { throw 'Arguments must be strings without NUL.' } }
    }
    [void](New-Item -ItemType Directory -Path $evidence -ErrorAction Stop)
    Copy-Item -LiteralPath $RecipePath -Destination (Join-Path $evidence 'recipe.json') -ErrorAction Stop
    Write-ProfileJson (Join-Path $evidence 'source-before.json') $source
    $clock = [Diagnostics.Stopwatch]::StartNew(); $receipts = @(); $failure = $null; $toolchains = @()
    try {
        foreach ($stage in $recipe.steps) {
            # Only this child's environment changes. Parent/user PATH is untouched.
            $portablePath = [IO.Path]::GetDirectoryName($recipe.toolchain.node)
            $basePath = $env:PATH
            if ($stage.environment.PSObject.Properties['PATH']) {
                $basePath = [string]$stage.environment.PATH
            }
            $stage.environment | Add-Member -NotePropertyName PATH -NotePropertyValue ($portablePath+';'+$basePath) -Force
            $toolchains += Get-ProfileToolchain $recipe.toolchain $evidence ('check-'+$stage.name) $stage.cwd $stage.environment
            $receipt = Invoke-ProfileStage $stage $evidence
            $receipts += $receipt
            if ($receipt.failure) { throw ('Stage {0}: {1}' -f $stage.name,$receipt.failure) }
            if ($stage.requireCleanAfter) {
                $after = Get-ProfileSourceState $Repository
                Write-ProfileJson (Join-Path $evidence ($stage.name + '.source.json')) $after
                if ($after.head -cne $source.head -or $after.indexTree -cne $source.indexTree -or $after.status.Count) { throw 'Stage changed tracked or nonignored source.' }
            }
        }
    } catch { $failure = $_.Exception.Message }
    finally {
        $clock.Stop()
        Write-ProfileJson (Join-Path $evidence 'summary.json') ([pscustomobject]@{
            source=$source;recipeSha256=$recipeHash;toolchainDeclaration=$recipe.toolchain;verifiedToolchains=$toolchains;cacheState=$recipe.cacheState
            wallMilliseconds=$clock.Elapsed.TotalMilliseconds;serialStageMilliseconds=($receipts | Measure-Object elapsedMilliseconds -Sum).Sum
            stages=@($receipts | Select-Object name,startUtc,endUtc,elapsedMilliseconds,exitCode,failure)
            failure=$failure;attribution='sampled process tree and serial stage edges; network/IO wait and missed processes unknown'
            humanGaps='not attributed to child CPU; wall includes observer and stage coordination'
            toolchainProbeMilliseconds=($toolchains | Measure-Object probeElapsedMilliseconds -Sum).Sum
        })
    }
    if ($failure) { throw $failure }
    $evidence
}
