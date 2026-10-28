// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ExecFileOptions } from 'node:child_process'
import type { BuildApplyOperation, Observation } from '@/shared/types/build-apply'
import { readBuildApplyStatus, requireBuildApplyStartup } from '@/electron/build-apply/status'

const fake = vi.hoisted(() => {
    const calls: string[][] = []
    return { output: '', error: false, calls }
})
vi.mock('node:child_process', () => ({
    execFile: (
        binary: string,
        args: string[],
        options: ExecFileOptions,
        callback: (error: Error | null, stdout: string) => void
    ) => {
        fake.calls.push([binary, ...args])
        expect(options).toMatchObject({
            cwd: '/inert',
            timeout: 10_000,
            maxBuffer: 128 << 10,
            windowsHide: true
        })
        callback(fake.error ? new Error('failed') : null, fake.output)
    }
}))

beforeEach(() => {
    fake.calls.length = 0
    fake.error = false
    fake.output = JSON.stringify({
        available: true,
        value: { state: 'idle', operation: null, startupAllowed: true },
        refusal: null
    })
})

function operation(outcome: BuildApplyOperation['outcome']): BuildApplyOperation {
    const unknown: Observation<never> = { known: false, value: null }
    const identity = { buildId: unknown, sourceRevision: unknown, dirty: unknown, sha256: unknown }
    const fences = {
        runtimeGeneration: unknown,
        serviceInstance: unknown,
        brokerGeneration: unknown,
        managerIncarnation: unknown,
        policyRevision: unknown,
        settingsRevision: unknown,
        intentRevision: unknown,
        locksRevision: unknown,
        availabilityRevision: unknown
    }
    return {
        operationId: 'operation',
        previewId: 'preview',
        ownershipAcknowledged: true,
        phase: 'prepare',
        outcome,
        failure: null,
        saved: unknown,
        enforced: unknown,
        physicalStop: unknown,
        remainsPaused: unknown,
        runningService: identity,
        runningGui: identity,
        readbackConfidence: 'unknown',
        fences
    }
}

describe('no-start status command', () => {
    it('executes the bounded service status protocol and permits idle', async () => {
        await expect(requireBuildApplyStartup('/inert/nvpair-service')).resolves.toBeUndefined()
        expect(fake.calls).toEqual([['/inert/nvpair-service', 'apply-build', 'status', '--json']])
    })
    it.each(['failure', 'malformed', 'unavailable', 'contradictory', 'oversized'])(
        'refuses %s',
        async scenario => {
            if (scenario === 'failure') fake.error = true
            if (scenario === 'malformed') fake.output = '{'
            if (scenario === 'unavailable')
                fake.output = JSON.stringify({
                    available: false,
                    value: null,
                    refusal: { code: 'unavailable', detail: 'Unreadable' }
                })
            if (scenario === 'contradictory')
                fake.output = JSON.stringify({
                    available: true,
                    value: { state: 'idle', operation: null, startupAllowed: false },
                    refusal: null
                })
            if (scenario === 'oversized') fake.output = ' '.repeat((128 << 10) + 1)
            await expect(requireBuildApplyStartup('/inert/nvpair-service')).rejects.toThrow(
                'prevents service lifecycle'
            )
        }
    )
    it('returns an unavailable result on command failure', async () => {
        fake.error = true
        expect(await readBuildApplyStatus('/inert/nvpair-service')).toMatchObject({
            available: false,
            value: null
        })
    })
    it.each([
        { state: 'pending', operation: operation(null), startupAllowed: false },
        { state: 'unknown', operation: operation(null), startupAllowed: false },
        { state: 'terminal', operation: operation('failed'), startupAllowed: false },
        { state: 'terminal', operation: operation('succeeded'), startupAllowed: true }
    ])('uses the authority for $state with startupAllowed=$startupAllowed', async status => {
        fake.output = JSON.stringify({ available: true, value: status, refusal: null })
        if (status.startupAllowed)
            await expect(requireBuildApplyStartup('/inert/nvpair-service')).resolves.toBeUndefined()
        else
            await expect(requireBuildApplyStartup('/inert/nvpair-service')).rejects.toThrow(
                'prevents service lifecycle'
            )
    })
})
