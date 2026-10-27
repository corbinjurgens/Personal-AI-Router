// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { mkdtempSync, readFileSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { Console } from 'node:console'
import electronLog from 'electron-log/node'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
    createStructuredLogger,
    flushLogs,
    getStructuredLogFilePath,
    initFileLogger
} from '@/shared/utils/log'

const directories: string[] = []

afterEach(() => {
    vi.restoreAllMocks()
    for (const directory of directories.splice(0))
        rmSync(directory, { recursive: true, force: true })
})

function initializeLogger() {
    const directory = mkdtempSync(join(tmpdir(), 'pair-test-stdout-'))
    directories.push(directory)
    initFileLogger({
        getUserData: () => directory,
        getTemp: () => directory,
        getResourcesPath: () => directory
    })
    return createStructuredLogger('stdout-test')
}

function readEntries(): object[] {
    return readFileSync(getStructuredLogFilePath(), 'utf8')
        .trim()
        .split('\n')
        .map(line => JSON.parse(line))
}

describe('logger console mirror', () => {
    it('contains repeated stderr EPIPE errors from the electron-log console transport', async () => {
        const logger = initializeLogger()
        const console = new Console({ stdout: process.stdout, stderr: process.stderr })
        const updaterLog = electronLog.create({ logId: 'stderr-pipe-test' })
        updaterLog.transports.file.level = false
        // Vitest replaces the global console. Use Node's console with the real
        // inherited streams at electron-log's existing console transport seam.
        updaterLog.transports.console.writeFn = ({ message }) => console.error(...message.data)
        const pipeError = Object.assign(new Error('broken pipe'), { code: 'EPIPE' })
        const emitted = vi.fn<() => void>()
        vi.spyOn(process.stderr, 'write').mockImplementation(() => {
            queueMicrotask(() => {
                process.stderr.emit('error', pipeError)
                emitted()
            })
            return false
        })
        vi.spyOn(process.stdout, 'write').mockReturnValue(true)

        expect(() => updaterLog.error('first updater failure')).not.toThrow()
        await new Promise<void>(resolve => queueMicrotask(resolve))
        expect(emitted).toHaveBeenCalledTimes(1)
        logger.info({ sublevel: 'test', message: 'after first stderr error' })
        expect(() => updaterLog.error('second updater failure')).not.toThrow()
        await new Promise<void>(resolve => queueMicrotask(resolve))
        expect(emitted).toHaveBeenCalledTimes(2)
        logger.info({ sublevel: 'test', message: 'after second stderr error' })
        await flushLogs(1000)

        expect(readEntries()).toEqual([
            expect.objectContaining({ message: 'after first stderr error', level: 'info' }),
            expect.objectContaining({ message: 'after second stderr error', level: 'info' })
        ])
    })

    it('tolerates synchronous stderr write failure through console.error', async () => {
        const logger = initializeLogger()
        const console = new Console({ stdout: process.stdout, stderr: process.stderr })
        vi.spyOn(process.stderr, 'write').mockImplementation(() => {
            throw Object.assign(new Error('closed stderr'), { code: 'EPIPE' })
        })
        vi.spyOn(process.stdout, 'write').mockReturnValue(true)

        expect(() => console.error('closed inherited pipe')).not.toThrow()
        logger.info({ sublevel: 'test', message: 'file remains writable' })
        await flushLogs(1000)
        expect(readEntries()).toEqual([
            expect.objectContaining({ message: 'file remains writable', level: 'info' })
        ])
    })
    it('preserves console formatting and structured file logging', async () => {
        const logger = initializeLogger()
        const write = vi.spyOn(process.stdout, 'write').mockReturnValue(true)

        logger.info({ sublevel: 'test', message: 'healthy', data: { count: 1 } })
        await flushLogs(1000)

        expect(write).toHaveBeenCalledWith(
            expect.stringMatching(/\(stdout-test\)\s+> healthy \{"count":1\}\n$/)
        )
        expect(readEntries()).toEqual([
            expect.objectContaining({
                source: 'stdout-test',
                level: 'info',
                message: 'healthy',
                data: { count: 1 }
            })
        ])
    })

    it('keeps file logging when stdout writes throw synchronously', async () => {
        const logger = initializeLogger()
        vi.spyOn(process.stdout, 'write').mockImplementation(() => {
            throw new Error('closed stdout')
        })

        expect(() => logger.info({ sublevel: 'test', message: 'first' })).not.toThrow()
        expect(() => logger.warn({ sublevel: 'test', message: 'second' })).not.toThrow()
        await flushLogs(1000)

        expect(readEntries()).toEqual([
            expect.objectContaining({ message: 'first', level: 'info' }),
            expect.objectContaining({ message: 'second', level: 'warn' })
        ])
    })

    it('handles asynchronous pipe errors and continues writing the file', async () => {
        const logger = initializeLogger()
        const pipeError = Object.assign(new Error('broken pipe'), { code: 'EPIPE' })
        const emitted = vi.fn<() => void>()
        vi.spyOn(process.stdout, 'write').mockImplementation(() => {
            queueMicrotask(() => {
                process.stdout.emit('error', pipeError)
                emitted()
            })
            return false
        })

        logger.info({ sublevel: 'test', message: 'before pipe error' })
        await new Promise<void>(resolve => queueMicrotask(resolve))
        expect(emitted).toHaveBeenCalledTimes(1)
        logger.error({ sublevel: 'test', message: 'after pipe error' })
        await flushLogs(1000)

        expect(emitted).toHaveBeenCalledTimes(2)
        expect(readEntries()).toEqual([
            expect.objectContaining({ message: 'before pipe error', level: 'info' }),
            expect.objectContaining({ message: 'after pipe error', level: 'error' })
        ])
    })
})
