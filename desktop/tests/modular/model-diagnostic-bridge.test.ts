// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import type { JsonObject, JsonValue } from '@/electron/service-bridge/json-rpc-client'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import type { DiagnosticPreview, DiagnosticRun } from '@/shared/types/model-diagnostic'
import type { MeasuredProfile } from '@/shared/types/model-config'

const mocks = vi.hoisted(() => ({
    supervisor: {
        callProcess: vi.fn<ReturnType<typeof getModularSupervisor>['callProcess']>(),
        hasProcess: vi.fn<ReturnType<typeof getModularSupervisor>['hasProcess']>(),
        callProxy: vi.fn<ReturnType<typeof getModularSupervisor>['callProxy']>(),
        sendProcess: vi.fn<ReturnType<typeof getModularSupervisor>['sendProcess']>(),
        reportError: vi.fn<ReturnType<typeof getModularSupervisor>['reportError']>()
    }
}))

vi.mock('@/electron/service-bridge/modular-supervisor', () => ({
    getModularSupervisor: () => mocks.supervisor
}))
vi.mock('@/electron/service-bridge/modular-state', () => ({
    getModularBridgeState: () => ({ getSelfId: () => 'local-node' }),
    isUpstreamUnreachableError: () => false,
    parseServiceErrors: () => []
}))
vi.mock('@/electron/service-bridge/model-catalog', () => ({ getEngineHubModels: vi.fn() }))
vi.mock('@/ui/stores/pending-actions.store', () => ({ usePendingActionsStore: {} }))
vi.mock('@/ui/stores/node-configuration.store', () => ({ refreshNodeManagement: vi.fn() }))

import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'
import {
    DIAGNOSTIC_CANCEL_TIMEOUT_MS,
    diagnosticStartParams,
    diagnosticRunParams,
    parseDiagnosticPreview,
    parseDiagnosticRun,
    parseDiagnosticStatus
} from '@/electron/service-bridge/model-diagnostic'
import { JsonRpcResponseError } from '@/electron/service-bridge/json-rpc-client'
import { createEngineApi } from '@/ui/api/engine-api'

const target = { engine: 'llamacpp', modelId: 'exact/model', profileKey: 'model-profile' }
const key = 'a'.repeat(64)
const at = '2026-11-07T12:00:00.123456789Z'

function preview(): DiagnosticPreview {
    return {
        ...target,
        configurationKey: key,
        nativeBuildId: `sha256:${'b'.repeat(64)}`,
        buildInfo: '',
        cpuModel: 'Fixture CPU',
        totalRamBytes: '18446744073709551615',
        gpus: [
            { uuid: 'gpu-a', name: 'GPU A', totalMemoryBytes: '8589934592' },
            { uuid: 'gpu-b', name: 'GPU B', totalMemoryBytes: '8589934592' }
        ],
        artifacts: [{ role: 'model', identity: 'artifact', size: '4096', assurance: 'sampled' }],
        requested: { context: { mode: 'fixed', tokens: 4096 }, slots: 1 },
        perSlotContext: 0,
        slots: 0,
        placement: [],
        previewId: 'review',
        lowPromptTokens: 0,
        highPromptTokens: 0,
        generateTokens: 128,
        timeLimitSeconds: 1200,
        unloadModels: [target],
        reloadedAfter: [target],
        drainEngines: ['llamacpp'],
        startsEngine: true,
        warnings: ['Other programs may affect this measurement.'],
        expiresAt: at
    }
}

function run(): DiagnosticRun {
    return {
        ...target,
        runId: 'run',
        configurationKey: key,
        phase: 'reserving',
        outcome: 'running',
        startedAt: at,
        elapsedMs: 0
    }
}

function measured(): MeasuredProfile {
    const identity = preview()
    return {
        configurationKey: key,
        nativeBuildId: identity.nativeBuildId,
        buildInfo: 'fixture build',
        cpuModel: identity.cpuModel,
        totalRamBytes: identity.totalRamBytes,
        gpus: identity.gpus,
        artifacts: identity.artifacts,
        requested: identity.requested,
        perSlotContext: 4096,
        slots: 1,
        placement: [{ domainId: 'host', kind: 'host' }],
        runId: 'run',
        recordedAt: at,
        outcome: 'completed',
        buildPin: 'fixture pin',
        pools: [
            {
                domainId: 'host',
                kind: 'host',
                baselineAvailableBytes: '8192',
                afterLoadAvailableBytes: '4096',
                lowMinAvailableBytes: '4000',
                highMinAvailableBytes: '3000',
                peakUsedBytes: '5192',
                source: 'fixture',
                samples: 5,
                intervalMs: 2000
            }
        ],
        low: {
            coverage: 'partial',
            requests: [{ targetPromptTokens: 512 }],
            wallMs: 0
        },
        high: {
            coverage: 'complete',
            requests: [
                {
                    targetPromptTokens: 3952,
                    promptTokens: 3952,
                    cachedTokens: 0,
                    promptMs: 100,
                    promptPerSecond: 39520,
                    generatedTokens: 128,
                    decodeMs: 1000,
                    decodePerSecond: 127,
                    firstTokenMs: 100,
                    endMs: 1100
                }
            ],
            overlapped: true,
            overlapWindowSampled: true,
            combinedGeneratedPerSecond: 116.36,
            wallMs: 1100
        },
        bufferSampleId: 'buffer',
        notes: ['Sampling can miss short peaks.']
    }
}

describe('diagnostic public decoding', () => {
    it('preserves unknown runtime allocation in the consent preview', () => {
        expect(parseDiagnosticPreview(preview())).toEqual(preview())
    })

    it('preserves known prompt targets and every consent effect', () => {
        const value = {
            ...preview(),
            perSlotContext: 4096,
            slots: 1,
            lowPromptTokens: 512,
            highPromptTokens: 3952
        }
        expect(parseDiagnosticPreview(value)).toEqual(value)
    })

    it('preserves unpriced GPU-selection identity and multiple device placements', () => {
        const { configurationKey, ...value } = preview()
        expect(configurationKey).toBe(key)
        const unpriced = {
            ...value,
            executionKey: key,
            configurationKeyReason: 'gpu-selection',
            requested: { extraOptions: { device: 'GPU-A,GPU-B', 'tensor-split': '1,1' } },
            placement: [
                { domainId: 'device-a', kind: 'device', deviceId: 'gpu-a' },
                { domainId: 'device-b', kind: 'device', deviceId: 'gpu-b' }
            ]
        }
        expect(parseDiagnosticPreview(unpriced)).toEqual(unpriced)
    })

    it('preserves the complete public run and measured result notification', () => {
        const value = {
            ...run(),
            phase: 'done',
            outcome: 'completed',
            elapsedMs: 10000,
            failedPhase: 'high',
            reason: 'Coverage note',
            result: measured()
        }
        expect(parseDiagnosticRun(value)).toEqual(value)
    })

    it('keeps failed results without inventing realized runtime values', () => {
        const result = measured()
        result.outcome = 'failed'
        result.perSlotContext = 0
        result.slots = 0
        result.buildInfo = ''
        result.placement = []
        result.pools = []
        delete result.high
        delete result.low
        expect(
            parseDiagnosticRun({ ...run(), phase: 'done', outcome: 'failed', result }).result
        ).toEqual(result)
    })

    it('rejects completed results without realized allocation', () => {
        expect(() =>
            parseDiagnosticRun({
                ...run(),
                outcome: 'completed',
                result: { ...measured(), perSlotContext: 0, slots: 0, placement: [], pools: [] }
            })
        ).toThrow()
    })

    it.each([
        { name: 'private manager authority', patch: { token: 'private' } },
        { name: 'private manager observation', patch: { observation: {} } },
        { name: 'unknown phase', patch: { phase: 'accepted' } },
        { name: 'unknown outcome', patch: { outcome: 'success' } },
        { name: 'oversized reason', patch: { reason: 'x'.repeat(257) } },
        { name: 'fractional elapsed', patch: { elapsedMs: 1.5 } },
        { name: 'invalid timestamp', patch: { startedAt: '2026-02-30T00:00:00Z' } },
        { name: 'two identity keys', patch: { executionKey: key } }
    ] satisfies Array<{ name: string; patch: JsonObject }>)('rejects $name', ({ patch }) => {
        expect(() => parseDiagnosticRun({ ...run(), ...patch })).toThrow()
    })

    it('rejects malformed consent effects and prompt targets', () => {
        expect(() =>
            parseDiagnosticPreview({
                ...preview(),
                warnings: Array.from({ length: 65 }, () => 'warning')
            })
        ).toThrow()
        expect(() => parseDiagnosticPreview({ ...preview(), highPromptTokens: 3952 })).toThrow()
        expect(() => parseDiagnosticPreview({ ...preview(), totalRamBytes: '01' })).toThrow()
    })

    it('rejects a measured result belonging to another run', () => {
        expect(() =>
            parseDiagnosticRun({
                ...run(),
                outcome: 'completed',
                result: { ...measured(), runId: 'another-run' }
            })
        ).toThrow()
    })

    it('decodes absent active status as null', () => {
        expect(parseDiagnosticStatus(null)).toBeNull()
        expect(diagnosticRunParams({})).toEqual({})
    })

    it('requires explicit consent and a run ID for cancellation', () => {
        expect(() =>
            diagnosticStartParams({ ...target, previewId: 'review', consent: false })
        ).toThrow()
        expect(() => diagnosticRunParams({}, true)).toThrow()
    })
})

describe('diagnostic bridge relay', () => {
    beforeEach(() => mocks.supervisor.callProcess.mockReset())

    it('relays preview and accepts the pending reservation from start', async () => {
        mocks.supervisor.callProcess.mockResolvedValueOnce(preview()).mockResolvedValueOnce(run())
        await expect(
            handleServiceBridgeInvoke('engines:diagnostic-preview', target)
        ).resolves.toEqual(preview())
        const request = { ...target, previewId: 'review', consent: true }
        await expect(
            handleServiceBridgeInvoke('engines:diagnostic-start', request)
        ).resolves.toEqual(run())
        expect(mocks.supervisor.callProcess.mock.calls).toEqual([
            ['broker', 'engine:diagnostic-preview', target],
            ['broker', 'engine:diagnostic-start', request]
        ])
    })

    it('looks up the active run without an ID', async () => {
        mocks.supervisor.callProcess.mockResolvedValueOnce(run())
        await expect(handleServiceBridgeInvoke('engines:diagnostic-status', {})).resolves.toEqual(
            run()
        )
        expect(mocks.supervisor.callProcess).toHaveBeenCalledWith(
            'broker',
            'engine:diagnostic-status',
            {}
        )
    })

    it('waits for cancellation cleanup and preserves cleanup-failed', async () => {
        let complete: ((value: JsonValue) => void) | undefined
        mocks.supervisor.callProcess.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    complete = resolve
                })
        )
        const pending = handleServiceBridgeInvoke('engines:diagnostic-cancel', { runId: 'run' })
        let settled = false
        void pending.then(() => {
            settled = true
        })
        await Promise.resolve()
        expect(settled).toBe(false)
        expect(mocks.supervisor.callProcess).toHaveBeenCalledWith(
            'broker',
            'engine:diagnostic-cancel',
            { runId: 'run' },
            DIAGNOSTIC_CANCEL_TIMEOUT_MS
        )
        if (!complete) throw new Error('Cancellation call did not reach the owner.')
        const held = {
            ...run(),
            outcome: 'cleanup-failed',
            phase: 'restoring',
            reason: 'still held'
        }
        complete(held)
        await expect(pending).resolves.toEqual(held)
    })

    it('rejects another run returned by status', async () => {
        mocks.supervisor.callProcess.mockResolvedValueOnce({ ...run(), runId: 'another' })
        await expect(
            handleServiceBridgeInvoke('engines:diagnostic-status', { runId: 'run' })
        ).rejects.toThrow('identity mismatch')
    })

    it('retains the broker refusal code', async () => {
        const error = new JsonRpcResponseError(-32010, 'diagnostic busy')
        mocks.supervisor.callProcess.mockRejectedValueOnce(error)
        await expect(handleServiceBridgeInvoke('engines:diagnostic-preview', target)).rejects.toBe(
            error
        )
    })
})

it('exposes the coordinated renderer diagnostic methods and subscription', async () => {
    const unsubscribe = vi.fn<() => void>()
    const transport = {
        connected: true,
        invoke: vi.fn<ServiceTransport['invoke']>().mockResolvedValue(null),
        subscribePush: vi.fn<ServiceTransport['subscribePush']>().mockReturnValue(unsubscribe),
        onConnect: vi.fn<ServiceTransport['onConnect']>(),
        onDisconnect: vi.fn<ServiceTransport['onDisconnect']>(),
        onAuthFailure: vi.fn<ServiceTransport['onAuthFailure']>(),
        destroy: vi.fn<ServiceTransport['destroy']>()
    } satisfies ServiceTransport
    const api = createEngineApi(transport)
    const start = { ...target, previewId: 'review', consent: true }
    await api.diagnosticPreview(target)
    await api.diagnosticStart(start)
    await api.diagnosticStatus({ nodeId: 'local-node' })
    await api.diagnosticCancel({ nodeId: 'local-node', runId: 'run' })
    const changed = vi.fn<(value: DiagnosticRun) => void>()
    expect(api.onDiagnosticChanged(changed)).toBe(unsubscribe)
    expect(transport.invoke.mock.calls).toEqual([
        ['engines:diagnostic-preview', target],
        ['engines:diagnostic-start', start],
        ['engines:diagnostic-status', { nodeId: 'local-node' }],
        ['engines:diagnostic-cancel', { nodeId: 'local-node', runId: 'run' }]
    ])
    expect(transport.subscribePush).toHaveBeenCalledWith('engines:diagnostic-changed', changed)
})
