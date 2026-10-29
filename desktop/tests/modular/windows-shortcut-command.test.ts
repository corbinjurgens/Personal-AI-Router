// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import path from 'path'
import { afterEach, describe, expect, it, vi } from 'vitest'

const calls = vi.hoisted(() => ({
    exit: vi.fn<(code: number) => void>(),
    lock: vi.fn<() => boolean>(() => false),
    connector: vi.fn<() => void>(),
    service: vi.fn<() => void>(),
    updater: vi.fn<() => void>(),
    window: vi.fn<() => void>(),
    buildApply: vi.fn<(argv: readonly string[]) => Promise<number | null>>()
}))

vi.mock('electron', () => ({
    app: {
        isPackaged: true,
        getPath: (name: string) => name === 'appData' ? 'C:\\User' : name === 'temp' ? 'C:\\Temp' : 'C:\\PAIR\\PAIR.exe',
        exit: calls.exit,
        requestSingleInstanceLock: calls.lock
    },
    shell: {
        readShortcutLink: () => ({ target: 'C:\\PAIR\\PAIR.exe', cwd: 'C:\\PAIR', icon: 'C:\\PAIR\\PAIR.exe', iconIndex: 0, description: 'Launch NVIDIA PAIR', appUserModelId: 'com.nvidia.nvpair', args: '' }),
        writeShortcutLink: () => true
    }
}))
vi.mock('@/shared/utils/platform', () => ({ currentPlatform: () => 'win32' }))
vi.mock('@/shared/utils/log', () => ({ createStructuredLogger: () => ({ info: vi.fn(), warn: vi.fn() }), initFileLogger: vi.fn() }))
vi.mock('@/electron/globals', () => ({ initPlatform: vi.fn() }))
vi.mock('@/electron/path', () => ({ ElectronPathProvider: vi.fn(), setPaths: vi.fn() }))
vi.mock('@/electron/build-apply', () => ({ handleBuildApplyEntry: calls.buildApply }))
vi.mock('@/electron/connector', () => ({ initializeConnector: calls.connector }))
vi.mock('@/electron/service-bridge/modular-supervisor', () => ({ ModularSupervisor: calls.service }))
vi.mock('@/electron/updater', () => ({ initializeUpdater: calls.updater }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: calls.window }))

const originalArgv = process.argv

afterEach(() => {
    process.argv = originalArgv
    vi.restoreAllMocks()
    vi.clearAllMocks()
    vi.resetModules()
})

describe('packaged shortcut command entry', () => {
    it.each(['create', 'status', 'remove'])('%s exits before normal startup when a GUI already owns the lock', async command => {
        process.argv = ['PAIR.exe', `--pair-shortcut=${command}`]
        vi.spyOn(fs, 'realpathSync').mockImplementation(value => value === 'C:\\Temp' ? 'C:\\Temp' : 'C:\\PAIR\\PAIR.exe')
        const fileStat = fs.statSync(path.join(process.cwd(), 'package.json'))
        vi.spyOn(fs, 'statSync').mockReturnValue(fileStat)
        vi.spyOn(fs, 'lstatSync').mockReturnValue(fileStat)
        vi.spyOn(fs, 'existsSync').mockReturnValue(false)
        vi.spyOn(fs, 'unlinkSync').mockImplementation(() => {})
        vi.spyOn(console, 'log').mockImplementation(() => {})
        vi.spyOn(console, 'error').mockImplementation(() => {})

        await import('@/electron/index')
        await vi.waitFor(() => expect(calls.exit).toHaveBeenCalledWith(0))
        expect(calls.lock).not.toHaveBeenCalled()
        expect(calls.buildApply).not.toHaveBeenCalled()
        expect(calls.connector).not.toHaveBeenCalled()
        expect(calls.service).not.toHaveBeenCalled()
        expect(calls.updater).not.toHaveBeenCalled()
        expect(calls.window).not.toHaveBeenCalled()
    })
})
