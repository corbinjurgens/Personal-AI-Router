// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import path from 'path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { initPlatform } from '@/electron/globals'
import { MODULAR_DEFAULT_LOG_LEVEL } from '@/shared/constants/modular-runtime'
import { currentPlatform } from '@/shared/utils/platform'
import { createTmpUserData } from '../fixtures/tmpdir'

vi.mock('@/shared/utils/platform', async importOriginal => {
    const actual = await importOriginal<typeof import('@/shared/utils/platform')>()
    return { ...actual, currentPlatform: vi.fn(actual.currentPlatform) }
})

const nativePlatform = currentPlatform()

describe('legacy UI config migration', () => {
    const userData = createTmpUserData()
    const legacy = path.join(userData.dir, 'ui-config.json')
    const current = path.join(userData.dir, 'configs', 'ui-config.json')
    const preferences = { firstRun: false, trayMode: 'menu', modularLogLevel: 'debug' }

    beforeEach(() => {
        fs.mkdirSync(userData.dir, { recursive: true })
        vi.mocked(currentPlatform).mockReturnValue(nativePlatform)
    })
    afterEach(() => vi.restoreAllMocks())

    async function instance() {
        vi.resetModules()
        initPlatform({
            getUserData: () => userData.dir,
            getTemp: () => userData.dir,
            getResourcesPath: () => userData.dir
        })
        return vi.importActual<typeof import('@/electron/config/ui-config')>(
            '@/electron/config/ui-config'
        )
    }

    async function verifyRetry() {
        const retry = await instance()
        retry.loadUiConfig()
        expect(retry.isFirstRun()).toBe(false)
        expect(retry.getTrayMode()).toBe('menu')
        expect(retry.getModularLogLevel()).toBe('debug')
        expect(JSON.parse(fs.readFileSync(current, 'utf8'))).toMatchObject(preferences)
        expect(fs.existsSync(legacy)).toBe(nativePlatform === 'win32')
    }

    it.each(['write', 'rename'])(
        'retains legacy preferences after a %s failure and retries',
        async operation => {
            fs.writeFileSync(legacy, JSON.stringify(preferences))
            const config = await instance()
            const failure =
                operation === 'write'
                    ? vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {
                          throw new Error('write failed')
                      })
                    : vi.spyOn(fs, 'renameSync').mockImplementation(() => {
                          throw new Error('rename failed')
                      })

            config.loadUiConfig()
            failure.mockRestore()

            expect(JSON.parse(fs.readFileSync(legacy, 'utf8'))).toEqual(preferences)
            expect(fs.existsSync(current)).toBe(false)
            await verifyRetry()
        }
    )

    it('retains legacy preferences when the file sync fails and retries', async () => {
        fs.writeFileSync(legacy, JSON.stringify(preferences))
        const config = await instance()
        const sync = vi.spyOn(fs, 'fsyncSync').mockImplementation(() => {
            throw new Error('file sync failed')
        })

        expect(() => config.loadUiConfig()).not.toThrow()
        expect(sync).toHaveBeenCalledOnce()
        sync.mockRestore()

        expect(JSON.parse(fs.readFileSync(legacy, 'utf8'))).toEqual(preferences)
        expect(fs.existsSync(current)).toBe(false)
        expect(config.getTrayMode()).toBe('menu')
        await verifyRetry()
    })

    it.skipIf(nativePlatform === 'win32').each(['configs', 'userData'])(
        'retains legacy preferences when post-rename %s sync fails',
        async directory => {
            fs.writeFileSync(legacy, JSON.stringify(preferences))
            const config = await instance()
            const fsync = fs.fsyncSync
            const sync = vi.spyOn(fs, 'fsyncSync').mockImplementationOnce(fsync)
            if (directory === 'userData') sync.mockImplementationOnce(fsync)
            sync.mockImplementationOnce(() => {
                throw new Error('directory sync failed')
            })

            expect(() => config.loadUiConfig()).not.toThrow()
            expect(sync).toHaveBeenCalledTimes(directory === 'configs' ? 2 : 3)
            sync.mockRestore()

            expect(JSON.parse(fs.readFileSync(legacy, 'utf8'))).toEqual(preferences)
            expect(JSON.parse(fs.readFileSync(current, 'utf8'))).toMatchObject(preferences)
            expect(config.getTrayMode()).toBe('menu')
            const reload = await instance()
            reload.loadUiConfig()
            expect(reload.getTrayMode()).toBe('menu')
            expect(fs.existsSync(legacy)).toBe(true)
        }
    )

    it.skipIf(nativePlatform === 'win32')('flushes before unlinking legacy', async () => {
        fs.writeFileSync(legacy, JSON.stringify(preferences))
        const config = await instance()
        const operations: string[] = []
        const files = new Map<number, string>()
        const { openSync, fsyncSync, closeSync, renameSync, unlinkSync } = fs
        vi.spyOn(fs, 'openSync').mockImplementation((file, flags, mode) => {
            const fd = openSync(file, flags, mode)
            files.set(fd, String(file))
            return fd
        })
        vi.spyOn(fs, 'fsyncSync').mockImplementation(fd => {
            fsyncSync(fd)
            operations.push(`sync ${files.get(fd)}`)
        })
        vi.spyOn(fs, 'closeSync').mockImplementation(fd => {
            closeSync(fd)
            operations.push(`close ${files.get(fd)}`)
        })
        vi.spyOn(fs, 'renameSync').mockImplementation((from, to) => {
            renameSync(from, to)
            operations.push('rename')
        })
        vi.spyOn(fs, 'unlinkSync').mockImplementation(file => {
            unlinkSync(file)
            operations.push('unlink')
        })

        config.loadUiConfig()

        expect(operations).toEqual([
            `sync ${current}.tmp`,
            `close ${current}.tmp`,
            'rename',
            `sync ${path.dirname(current)}`,
            `close ${path.dirname(current)}`,
            `sync ${userData.dir}`,
            `close ${userData.dir}`,
            'unlink'
        ])
        expect(JSON.parse(fs.readFileSync(current, 'utf8'))).toMatchObject(preferences)
        expect(fs.existsSync(legacy)).toBe(false)
    })

    it('retains legacy on Windows without metadata durability confirmation', async () => {
        vi.mocked(currentPlatform).mockReturnValue('win32')
        fs.writeFileSync(legacy, JSON.stringify(preferences))
        const config = await instance()

        config.loadUiConfig()

        expect(JSON.parse(fs.readFileSync(current, 'utf8'))).toMatchObject(preferences)
        expect(JSON.parse(fs.readFileSync(legacy, 'utf8'))).toEqual(preferences)
        expect(config.getTrayMode()).toBe('menu')
    })

    it('merges defaults when legacy preferences omit newer fields', async () => {
        fs.writeFileSync(legacy, JSON.stringify({ trayMode: 'menu' }))
        const config = await instance()

        config.loadUiConfig()

        expect(config.isFirstRun()).toBe(false)
        expect(config.isMacHelperSetupComplete()).toBe(false)
        expect(config.getModularLogLevel()).toBe(MODULAR_DEFAULT_LOG_LEVEL)
        expect(JSON.parse(fs.readFileSync(current, 'utf8'))).toEqual({
            firstRun: false,
            macHelperSetupComplete: false,
            modularLogLevel: MODULAR_DEFAULT_LOG_LEVEL,
            trayMode: 'menu'
        })
    })

    it.each(['write', 'rename'])('keeps setters best effort on %s failure', async operation => {
        const config = await instance()
        const failure =
            operation === 'write'
                ? vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {
                      throw new Error('write failed')
                  })
                : vi.spyOn(fs, 'renameSync').mockImplementation(() => {
                      throw new Error('rename failed')
                  })

        expect(() => {
            config.completeFirstRun()
            config.setModularLogLevel('debug')
            config.setMacHelperSetupComplete(true)
            config.setTrayMode('menu')
        }).not.toThrow()
        failure.mockRestore()

        expect(config.isFirstRun()).toBe(false)
        expect(config.getModularLogLevel()).toBe('debug')
        expect(config.isMacHelperSetupComplete()).toBe(true)
        expect(config.getTrayMode()).toBe('menu')
        expect(fs.existsSync(current)).toBe(false)
    })
})
