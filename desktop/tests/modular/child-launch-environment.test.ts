// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { SpawnOptions, ExecFileOptions } from 'node:child_process'
import type { SupportedPlatform } from '@/shared/types/platform'
import { ensureNvpairOnPath } from '@/electron/nvpair-command'
import { spawnWipeScript } from '@/electron/run-wipe-script'
import { macPrivilege } from '@/electron/services/mac-privilege-service'
import { serviceConnectionDeps } from '@/electron/service-bridge/service-connection'

let platform: SupportedPlatform = 'win32'
const launches: {
    executable: string
    args: readonly string[]
    options: SpawnOptions | ExecFileOptions
}[] = []
let unrefs = 0
const parentEnv = {
    ...process.env,
    NVPAIR_PROVIDER_SECRET_TEST: 'test-sentinel',
    nvpair_provider_secret_test: 'lowercase-sentinel',
    C2_UNRELATED: 'kept'
}

vi.mock('@/shared/utils/platform', () => ({ currentPlatform: () => platform }))
vi.mock('electron', () => ({
    app: { isPackaged: true, getPath: () => '/fake/user-data', getAppPath: () => '/fake/app' }
}))
vi.mock('fs', () => ({
    default: {
        existsSync: () => true,
        mkdirSync: () => {},
        readFileSync: () => '',
        writeFileSync: () => {}
    },
    existsSync: () => true
}))
vi.mock('child_process', () => ({
    spawn: (executable: string, args: readonly string[], options: SpawnOptions) => {
        launches.push({ executable, args, options })
        return {
            on: () => {},
            unref: () => {
                unrefs++
            }
        }
    },
    execFile: (
        executable: string,
        args: readonly string[],
        options: ExecFileOptions,
        callback: (error: Error | null, stdout: string) => void
    ) => {
        launches.push({ executable, args, options })
        callback(null, '{"registration":"enabled","reachable":true}')
    }
}))

beforeEach(() => {
    launches.length = 0
    unrefs = 0
    platform = 'win32'
    // A plain object preserves both case variants even on Windows.
    vi.stubGlobal('process', {
        ...process,
        env: { ...parentEnv },
        resourcesPath: '/fake/resources'
    })
})
afterEach(() => vi.unstubAllGlobals())

function expectFilteredLaunch(): void {
    expect(launches).toHaveLength(1)
    const env = launches[0]?.options.env
    expect(env !== undefined).toBe(true)
    expect(Object.hasOwn(env ?? {}, 'NVPAIR_PROVIDER_SECRET_TEST')).toBe(false)
    expect(Object.hasOwn(env ?? {}, 'nvpair_provider_secret_test')).toBe(false)
    expect(env?.C2_UNRELATED === 'kept').toBe(true)
    expect(Object.hasOwn(process.env, 'NVPAIR_PROVIDER_SECRET_TEST')).toBe(true)
    expect(Object.hasOwn(process.env, 'nvpair_provider_secret_test')).toBe(true)
}

describe('Electron child environment boundaries', () => {
    it('filters the PowerShell PATH update', () => {
        ensureNvpairOnPath()
        expectFilteredLaunch()
        expect(launches[0]?.executable).toBe('powershell')
    })

    it('filters the Windows wipe and relaunch script', () => {
        spawnWipeScript({ waitPid: 123, relaunchExecPath: '/fake/app.exe' })
        expectFilteredLaunch()
        expect(launches[0]?.executable).toContain('wipe-app-data.cmd')
        expect(launches[0]?.options).toMatchObject({ shell: true, detached: true, stdio: 'ignore' })
        expect(unrefs).toBe(1)
    })

    it('filters the Bash wipe and relaunch script', () => {
        platform = 'linux'
        spawnWipeScript({ waitPid: 123, relaunchExecPath: '/fake/app' })
        expectFilteredLaunch()
        expect(launches[0]?.executable).toBe('bash')
        expect(launches[0]?.args[0]).toContain('wipe-app-data.sh')
        expect(unrefs).toBe(1)
    })

    it('filters the macOS privilege control tool', async () => {
        platform = 'darwin'
        await macPrivilege.status()
        expectFilteredLaunch()
        expect(launches[0]?.executable).toContain('nvpair-helper-ctl')
        expect(launches[0]?.args).toEqual(['status'])
    })

    it('retains inherited credentials when production dependencies start the service', () => {
        serviceConnectionDeps('/fake/nvpair-service', ['--ipc']).startService()
        expect(launches).toHaveLength(1)
        const launch = launches[0]
        expect(launch?.executable).toBe('/fake/nvpair-service')
        expect(launch?.args).toEqual(['--', '--ipc'])
        expect(Object.hasOwn(launch?.options ?? {}, 'env')).toBe(false)
        expect(launch?.options).toMatchObject({
            detached: true,
            stdio: 'ignore',
            windowsHide: true
        })
        expect(Object.hasOwn(process.env, 'NVPAIR_PROVIDER_SECRET_TEST')).toBe(true)
        expect(Object.hasOwn(process.env, 'nvpair_provider_secret_test')).toBe(true)
        expect(process.env.C2_UNRELATED === 'kept').toBe(true)
        expect(unrefs).toBe(1)
    })
})
