// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PassThrough, type Duplex } from 'stream'
import { describe, expect, it, vi } from 'vitest'
import {
    connectOrStart,
    type ConnectOrStartDeps
} from '@/electron/service-bridge/service-connection'

vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        verbose: vi.fn()
    })
}))

const ENDPOINT = '/run/user/1000/service.sock'

/** Fake deps with a virtual clock that only `sleep` advances. */
function harness(options: { answersAfterStarts?: number; answersAfterAttempts?: number }) {
    const socket = new PassThrough()
    let clock = 0
    let started = 0
    const attempts: string[] = []
    const deps: ConnectOrStartDeps = {
        connect: vi.fn((endpoint: string): Promise<Duplex> => {
            attempts.push(endpoint)
            const answers =
                (options.answersAfterStarts !== undefined &&
                    started >= options.answersAfterStarts) ||
                (options.answersAfterAttempts !== undefined &&
                    started > 0 &&
                    attempts.length >= options.answersAfterAttempts)
            return answers ? Promise.resolve(socket) : Promise.reject(new Error('ECONNREFUSED'))
        }),
        startService: vi.fn(() => {
            started++
        }),
        sleep: vi.fn((ms: number) => {
            clock += ms
            return Promise.resolve()
        }),
        now: () => clock
    }
    return { deps, socket, attempts, elapsed: () => clock }
}

describe('connectOrStart', () => {
    it('attaches to a running service without starting one', async () => {
        const h = harness({ answersAfterStarts: 0 })

        await expect(connectOrStart({ endpoint: ENDPOINT }, h.deps)).resolves.toBe(h.socket)

        expect(h.deps.startService).not.toHaveBeenCalled()
        expect(h.deps.sleep).not.toHaveBeenCalled()
        expect(h.attempts).toEqual([ENDPOINT])
    })

    it('starts the service when nothing answers, then retries until it does', async () => {
        const h = harness({ answersAfterAttempts: 5 })

        await expect(connectOrStart({ endpoint: ENDPOINT }, h.deps)).resolves.toBe(h.socket)

        expect(h.deps.startService).toHaveBeenCalledOnce()
        // One dial before the start, then four more at 100 ms intervals.
        expect(h.attempts).toHaveLength(5)
        expect(h.deps.sleep).toHaveBeenCalledTimes(3)
        expect(h.elapsed()).toBe(300)
    })

    it('dials the same endpoint every time', async () => {
        const h = harness({ answersAfterAttempts: 3 })

        await connectOrStart({ endpoint: ENDPOINT }, h.deps)

        expect(new Set(h.attempts)).toEqual(new Set([ENDPOINT]))
    })

    it('does not start a second service while retrying', async () => {
        const h = harness({ answersAfterAttempts: 20 })

        await connectOrStart({ endpoint: ENDPOINT }, h.deps)

        expect(h.deps.startService).toHaveBeenCalledOnce()
    })

    it('gives up after about ten seconds', async () => {
        const h = harness({})

        await expect(connectOrStart({ endpoint: ENDPOINT }, h.deps)).rejects.toThrow(
            /did not answer on \/run\/user\/1000\/service\.sock within 10000 ms: ECONNREFUSED/
        )

        expect(h.deps.startService).toHaveBeenCalledOnce()
        expect(h.elapsed()).toBeGreaterThanOrEqual(9_900)
        expect(h.elapsed()).toBeLessThanOrEqual(10_000)
    })

    it('honours a custom timeout and interval', async () => {
        const h = harness({})

        await expect(
            connectOrStart({ endpoint: ENDPOINT, timeoutMs: 1_000, retryIntervalMs: 250 }, h.deps)
        ).rejects.toThrow(/within 1000 ms/)

        expect(h.elapsed()).toBe(1_000)
    })

    it('does not dial or wait when the service cannot be launched', async () => {
        const h = harness({})
        h.deps.startService = vi.fn(() => {
            throw new Error('spawn ENOENT')
        })

        await expect(connectOrStart({ endpoint: ENDPOINT }, h.deps)).rejects.toThrow('spawn ENOENT')

        expect(h.attempts).toHaveLength(1)
        expect(h.deps.sleep).not.toHaveBeenCalled()
    })
})
