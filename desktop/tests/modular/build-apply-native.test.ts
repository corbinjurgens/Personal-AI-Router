// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import log from 'electron-log'

const calls = vi.hoisted(() => {
    const logger: { current: Pick<typeof log, 'error'> | null } = { current: null }
    return {
        logger,
        install: vi.fn<() => void>(),
        stop: vi.fn<() => Promise<void>>(),
        check: vi.fn<() => Promise<null>>(),
        download: vi.fn<() => Promise<string[]>>(),
        on: vi.fn<(event: string, callback: (value: Error) => void) => void>(),
        packaged: true
    }
})
vi.mock('electron', () => ({
    app: {
        get isPackaged() {
            return calls.packaged
        }
    },
    BrowserWindow: { getAllWindows: () => [] }
}))
vi.mock('electron-updater', () => ({
    default: {
        autoUpdater: {
            get logger() {
                return calls.logger.current
            },
            set logger(value: Pick<typeof log, 'error'> | null) {
                calls.logger.current = value
            },
            quitAndInstall: calls.install,
            checkForUpdates: calls.check,
            downloadUpdate: calls.download,
            on: calls.on
        }
    }
}))
vi.mock('@/electron/connector', () => ({ destroyConnector: calls.stop }))
vi.mock('@/electron/updater/update-notification', () => ({
    notifyUpdateAvailable: vi.fn<(version: string) => void>()
}))
vi.mock('@/electron/updater/pending-update-store', () => ({
    clearPendingUpdate: vi.fn<() => void>(),
    readPendingUpdateVersion: vi.fn<() => string | null>(),
    writePendingUpdateVersion: vi.fn<(version: string) => void>()
}))

import {
    checkForUpdates,
    downloadUpdate,
    getUpdateStatus,
    initializeUpdater,
    quitAndInstallUpdate
} from '@/electron/updater'

let directory = ''
const resources = Object.getOwnPropertyDescriptor(process, 'resourcesPath')
beforeEach(() => {
    vi.useFakeTimers()
    directory = fs.mkdtempSync(path.join(os.tmpdir(), 'pair-build-updater-'))
    Object.defineProperty(process, 'resourcesPath', { configurable: true, value: directory })
    calls.packaged = true
    calls.check.mockResolvedValue(null)
    calls.download.mockResolvedValue([])
})
afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
    vi.restoreAllMocks()
    fs.rmSync(directory, { recursive: true })
    if (resources) Object.defineProperty(process, 'resourcesPath', resources)
    else Reflect.deleteProperty(process, 'resourcesPath')
})

it('refuses direct installation without stopping the service or launching the installer', async () => {
    await quitAndInstallUpdate()

    expect(calls.stop).not.toHaveBeenCalled()
    expect(calls.install).not.toHaveBeenCalled()
    expect(getUpdateStatus()).toMatchObject({
        phase: 'error',
        error: expect.stringContaining('unavailable')
    })
})

describe('distribution eligibility', () => {
    it('skips development scheduling and gives the explicit unconfigured result', async () => {
        calls.packaged = false
        initializeUpdater()
        expect(vi.getTimerCount()).toBe(0)
        await checkForUpdates()
        await downloadUpdate()
        expect(getUpdateStatus().error).toBe(
            'Distribution updates are not configured for this build'
        )
        expect(calls.check).not.toHaveBeenCalled()
        expect(calls.download).not.toHaveBeenCalled()
        expect(calls.on).not.toHaveBeenCalled()
    })
    it('does not let the former dev mock environment bypass unconfigured Check or Download', async () => {
        calls.packaged = false
        vi.stubEnv('PAIR_MOCK_UPDATER', 'true')
        await checkForUpdates()
        await downloadUpdate()
        expect(getUpdateStatus().error).toBe(
            'Distribution updates are not configured for this build'
        )
        expect(calls.check).not.toHaveBeenCalled()
        expect(calls.download).not.toHaveBeenCalled()
        vi.unstubAllEnvs()
    })
    it('quietly skips absent packaged configuration without timers or updater calls', async () => {
        const previous = getUpdateStatus()
        initializeUpdater()
        initializeUpdater()
        expect(getUpdateStatus()).toEqual(previous)
        expect(vi.getTimerCount()).toBe(0)
        expect(calls.on).not.toHaveBeenCalled()
        await checkForUpdates()
        await downloadUpdate()
        expect(getUpdateStatus().error).toBe(
            'Distribution updates are not configured for this build'
        )
        expect(calls.check).not.toHaveBeenCalled()
        expect(calls.download).not.toHaveBeenCalled()
    })
    it('allows configured packaged discovery and download, while installation still refuses', async () => {
        fs.writeFileSync(
            path.join(directory, 'app-update.yml'),
            'provider: generic\nurl: https://example.invalid/updates\n'
        )
        initializeUpdater()
        expect(calls.on).toHaveBeenCalled()
        expect(vi.getTimerCount()).toBe(2)
        await checkForUpdates()
        await downloadUpdate()
        await quitAndInstallUpdate()
        expect(calls.check).toHaveBeenCalledOnce()
        expect(calls.download).toHaveBeenCalledOnce()
        expect(calls.stop).not.toHaveBeenCalled()
        expect(calls.install).not.toHaveBeenCalled()
        const errorListener = calls.on.mock.calls.find(([event]) => event === 'error')?.[1]
        if (!errorListener) throw new Error('Missing updater error listener')
        const report = vi.spyOn(log, 'error').mockImplementation(() => {})
        const malformed = Object.assign(new Error('Malformed updater configuration'), {
            name: 'YAMLException'
        })
        calls.check.mockImplementationOnce(async () => {
            // AppUpdater's constructor installs this before the app listener.
            calls.logger.current?.error(`Error: ${malformed.stack || malformed.message}`)
            errorListener(malformed)
            throw malformed
        })
        await checkForUpdates()
        expect(getUpdateStatus().error).toContain('invalid or unreadable')
        expect(report).toHaveBeenCalledOnce()
        expect(vi.getTimerCount()).toBe(0)
        const registrations = calls.on.mock.calls.length
        initializeUpdater()
        expect(calls.on).toHaveBeenCalledTimes(registrations)
        expect(vi.getTimerCount()).toBe(0)
    })
    it.each(['YAMLException', 'TypeError'])(
        'uses the existing updater loader for malformed present configuration (%s) and cancels automatic retries',
        async name => {
            fs.writeFileSync(path.join(directory, 'app-update.yml'), 'provider: [')
            calls.check.mockRejectedValue(
                Object.assign(new Error('Invalid update configuration'), { name })
            )
            await checkForUpdates()
            expect(getUpdateStatus().error).toContain('invalid or unreadable')
            expect(calls.check).toHaveBeenCalledOnce()
            expect(calls.download).not.toHaveBeenCalled()
            expect(vi.getTimerCount()).toBe(0)
        }
    )
    it('reports unreadable present configuration distinctly from absence through the updater loader', async () => {
        fs.writeFileSync(path.join(directory, 'app-update.yml'), 'provider: generic')
        calls.check.mockRejectedValue(
            Object.assign(new Error('Cannot read configuration'), { code: 'EACCES' })
        )
        await checkForUpdates()
        expect(getUpdateStatus().error).toContain('invalid or unreadable')
        expect(calls.check).toHaveBeenCalledOnce()
        expect(calls.on).not.toHaveBeenCalled()
    })
})
