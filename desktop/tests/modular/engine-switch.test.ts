// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from 'vitest'
import {
    parseEngineSwitchResult,
    parseEngineSwitchStatus,
    parseEngineSwitchChanged,
    parseEngineSwitchFenceResult
} from '@/electron/service-bridge/engine-switch'
import type { JsonObject, JsonValue } from '@/electron/service-bridge/json-rpc-client'
import type { SwitchReceipt, SwitchPhase, SwitchFenceResult } from '@/shared/types/engine-switch'
import type { PreloadServiceTransport } from '@/shared/types/service-bridge'
import { ServiceCallError, serviceErrorCode, serviceTransport } from '@/ui/api/service-transport'
import { createEngineApi } from '@/ui/api/engine-api'
import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { JsonRpcResponseError } from '@/electron/service-bridge/json-rpc-client'
import { MODULAR_ENGINE_LIFECYCLE_CALL_TIMEOUT_MS } from '@/shared/constants/modular-runtime'
import type { createOverviewWindow } from '@/electron/window'

vi.mock('electron', () => ({ app: { isPackaged: false, getAppPath: () => process.cwd() } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn<typeof createOverviewWindow>() }))
vi.mock('@/electron/config/ui-config', () => ({ isFirstRun: () => false }))
vi.mock('@/electron/service-bridge/node-info-poller', () => ({
    startNodeInfoPoller: vi.fn<() => void>(),
    stopNodeInfoPoller: vi.fn<() => void>()
}))

afterEach(() => vi.restoreAllMocks())

const revision = `e3v1:${'a'.repeat(64)}`
const observed = {
    intent: 'off',
    running: false,
    healthy: false,
    observedAt: '2026-10-19T01:00:00Z',
    terminationVerified: true
}

describe('engine switch fence', () => {
    const request = { nodeId: 'node-a', requestId: 'switch-1' }
    const fenced = { ...request, fenced: true } satisfies SwitchFenceResult
    it('parses confirmed fencing and a receipt that won acceptance', () => {
        expect(parseEngineSwitchFenceResult(fenced, request)).toEqual(fenced)
        const accepted = { ...request, fenced: false, receipt: fixture() }
        expect(parseEngineSwitchFenceResult(accepted, request)).toEqual(accepted)
    })
    it('accepts a canonical node returned for a discovery alias', () => {
        const alias = { nodeId: 'discovery-alias', requestId: request.requestId }
        expect(parseEngineSwitchFenceResult(fenced, alias)).toEqual(fenced)
        const accepted = { ...request, fenced: false, receipt: fixture() }
        expect(parseEngineSwitchFenceResult(accepted, alias)).toEqual(accepted)
    })
    const invalid: [string, JsonValue][] = [
        ['missing flag', { ...request }],
        ['neither alternative', { ...request, fenced: false }],
        ['both alternatives', { ...fenced, receipt: fixture() }],
        ['null receipt', { ...request, fenced: false, receipt: null }],
        ['wrong request', { ...fenced, requestId: 'other' }],
        ['empty canonical node', { ...fenced, nodeId: '' }],
        [
            'receipt identity mismatch',
            { ...request, fenced: false, receipt: { ...fixture(), nodeId: 'other' } }
        ],
        ['unknown field', { ...fenced, success: true }],
        ['malformed receipt', { ...request, fenced: false, receipt: {} }]
    ]
    it.each(invalid)('rejects %s without claiming resolution', (_name, value) => {
        expect(() => parseEngineSwitchFenceResult(value, request)).toThrow(
            'Invalid engine switch response.'
        )
    })
    it.each([undefined, 'node-a', 'discovery-alias'])(
        'calls the broker fence with exact payload for node %s',
        async nodeId => {
            vi.spyOn(getModularBridgeState(), 'getSelfId').mockReturnValue('node-a')
            const payload = { ...(nodeId === undefined ? {} : { nodeId }), requestId: 'switch-1' }
            const call = vi.spyOn(getModularSupervisor(), 'callProcess').mockResolvedValue(fenced)
            await expect(
                handleServiceBridgeInvoke('engines:switch-fence', payload)
            ).resolves.toEqual(fenced)
            expect(call).toHaveBeenCalledExactlyOnceWith(
                'broker',
                'engine:switch-fence',
                payload,
                MODULAR_ENGINE_LIFECYCLE_CALL_TIMEOUT_MS
            )
        }
    )
    it('preserves an owned receipt from the broker fence', async () => {
        const accepted = { ...request, fenced: false, receipt: fixture() }
        vi.spyOn(getModularSupervisor(), 'callProcess').mockResolvedValue(accepted)
        await expect(handleServiceBridgeInvoke('engines:switch-fence', request)).resolves.toEqual(
            accepted
        )
    })
    it.each([-32010, -32012, -32000])('propagates broker refusal %s', async code => {
        const refusal = new JsonRpcResponseError(code, 'fence refused')
        vi.spyOn(getModularSupervisor(), 'callProcess').mockRejectedValue(refusal)
        await expect(handleServiceBridgeInvoke('engines:switch-fence', request)).rejects.toBe(
            refusal
        )
    })
    it('rejects a mismatched broker result', async () => {
        vi.spyOn(getModularSupervisor(), 'callProcess').mockResolvedValue({
            ...fenced,
            requestId: 'other'
        })
        await expect(handleServiceBridgeInvoke('engines:switch-fence', request)).rejects.toThrow(
            'Invalid engine switch response.'
        )
    })
    it('forwards the typed renderer fence and result without changing engine status', async () => {
        const statuses = getModularBridgeState().getEngineInitialState().statuses
        const invoke = vi
            .fn<
                PreloadServiceTransport['invoke']
            >(async () => ({ success: false, error: 'unexpected invoke' }))
            .mockResolvedValue({ success: true, data: fenced })
        const preload: PreloadServiceTransport = {
            connected: true,
            invoke,
            invokeHistory: vi.fn<PreloadServiceTransport['invokeHistory']>(() => {
                throw new Error('Unexpected history invoke')
            }),
            subscribePush: () => () => {},
            onConnect: () => () => {},
            onDisconnect: () => () => {},
            onAuthFailure: () => () => {},
            destroy: () => {}
        }
        const api = createEngineApi(serviceTransport(preload))
        await expect(api.fenceSwitch(request)).resolves.toEqual(fenced)
        expect(invoke).toHaveBeenCalledExactlyOnceWith('engines:switch-fence', request)
        expect(getModularBridgeState().getEngineInitialState().statuses).toEqual(statuses)
        invoke.mockResolvedValue({ success: false, error: 'denied', code: -32012 })
        await expect(api.fenceSwitch(request)).rejects.toSatisfy(
            (error: Error) =>
                error instanceof ServiceCallError && serviceErrorCode(error) === -32012
        )
    })
})
function fixture(): JsonObject {
    return {
        version: 1,
        nodeId: 'node-a',
        requestId: 'switch-1',
        caller: 'local',
        payloadHash: `sha256:${'b'.repeat(64)}`,
        request: {
            nodeId: 'node-a',
            requestId: 'switch-1',
            sourceEngine: 'ollama',
            targetEngine: 'llama-cpp',
            expectedRevision: revision
        },
        expectedRevision: revision,
        acceptedRevision: revision,
        sequence: 1,
        phase: 'accepted',
        terminal: false,
        acceptedAt: '2026-10-19T01:00:00Z',
        updatedAt: '2026-10-19T01:00:00Z',
        deadline: '2026-10-19T01:15:00Z',
        finishedAt: null,
        source: { ...observed },
        target: { ...observed },
        error: null,
        recoveryRequired: false
    }
}

describe('engine switch receipt parser', () => {
    const phases: SwitchPhase[] = [
        'accepted',
        'validating',
        'draining',
        'source-stopping',
        'source-off',
        'target-starting',
        'target-ready',
        'completed',
        'failed-validation',
        'failed-drain',
        'failed-source-stop',
        'failed-target-start',
        'failed-publication',
        'failed-interrupted',
        'rejected'
    ]
    it.each(phases)('accepts the complete %s receipt', phase => {
        const terminal =
            phase === 'completed' || phase === 'rejected' || phase.startsWith('failed-')
        const row = {
            ...fixture(),
            phase,
            terminal,
            finishedAt: terminal ? observed.observedAt : null,
            error:
                terminal && phase !== 'completed'
                    ? { code: -32000, kind: 'interrupted', message: 'interrupted' }
                    : null
        }
        expect(parseEngineSwitchResult({ receipt: row }).receipt).toEqual(row)
    })
    const rejected: [string, JsonValue | undefined][] = [
        ['missing', undefined],
        ['null', null],
        ['array', []],
        ['wrong version', { ...fixture(), version: 2 }],
        ['unknown field', { ...fixture(), surprise: true }],
        ['missing field', { ...fixture(), caller: null }],
        ['unsafe sequence', { ...fixture(), sequence: Number.MAX_SAFE_INTEGER + 1 }],
        ['zero sequence', { ...fixture(), sequence: 0 }],
        ['unknown phase', { ...fixture(), phase: 'starting' }],
        ['terminal mismatch', { ...fixture(), terminal: true }],
        ['request mismatch', { ...fixture(), requestId: 'different' }],
        ['revision mismatch', { ...fixture(), expectedRevision: `e3v1:${'c'.repeat(64)}` }],
        ['bad hash', { ...fixture(), payloadHash: 'bad' }],
        ['bad timestamp', { ...fixture(), updatedAt: 'yesterday' }],
        ['backward time', { ...fixture(), updatedAt: '2026-10-19T00:00:00Z' }],
        ['long deadline', { ...fixture(), deadline: '2026-10-19T02:00:00Z' }],
        ['missing observation', { ...fixture(), source: { ...observed, observedAt: null } }],
        ['invalid observation flag', { ...fixture(), target: { ...observed, running: 'no' } }],
        ['invalid intent', { ...fixture(), source: { ...observed, intent: 'enabled' } }],
        [
            'bad error code',
            { ...fixture(), error: { code: -32000, kind: 'remote_veto', message: 'refused' } }
        ],
        [
            'unknown error kind',
            { ...fixture(), error: { code: -32000, kind: 'other', message: 'refused' } }
        ]
    ]
    it.each(rejected)('rejects %s', (_name, row) => {
        expect(() => parseEngineSwitchResult({ receipt: row ?? null })).toThrow(
            'Invalid engine switch response'
        )
    })
    it.each(Object.keys(fixture()))('requires the %s receipt field', key => {
        const row = fixture()
        delete row[key]
        expect(() => parseEngineSwitchResult({ receipt: row })).toThrow()
    })
    it('accepts registered variant IDs without a built-in engine allowlist', () => {
        const row = fixture()
        const request = {
            nodeId: 'node-a',
            requestId: 'switch-1',
            sourceEngine: 'Ollama_1.0',
            targetEngine: 'custom.engine-2',
            expectedRevision: revision
        }
        expect(parseEngineSwitchResult({ receipt: { ...row, request } }).receipt.request).toEqual(
            request
        )
    })
    it('preserves explicit unknown runtime observations', () => {
        const source = {
            intent: 'unknown',
            running: null,
            healthy: null,
            observedAt: null,
            terminationVerified: false
        }
        expect(
            parseEngineSwitchResult({ receipt: { ...fixture(), source } }).receipt.source
        ).toEqual(source)
    })
    it.each([
        [-32602, 'invalid_request'],
        [-32602, 'reused_id'],
        [-32010, 'configuration_conflict'],
        [-32010, 'fenced'],
        [-32011, 'remote_veto'],
        [-32013, 'engine_locked'],
        [-32012, 'validation'],
        [-32000, 'validation'],
        [-32000, 'busy']
    ])('preserves error code %s and kind %s', (code, kind) => {
        const error = { code, kind, message: 'refused' }
        expect(parseEngineSwitchResult({ receipt: { ...fixture(), error } }).receipt.error).toEqual(
            error
        )
    })
})

describe('engine switch status and notification parsers', () => {
    it('accepts remote management denial with validation kind in status', () => {
        const error = { code: -32012, kind: 'validation', message: 'Remote management denied' }
        expect(
            parseEngineSwitchStatus({
                configurationRevision: revision,
                receipt: { ...fixture(), error }
            }).receipt?.error
        ).toEqual(error)
    })
    it.each([
        [-32012, 'remote_veto'],
        [-32013, 'remote_veto'],
        [-32011, 'engine_locked'],
        [-32000, 'engine_locked'],
        [-32013, 'validation'],
        [-32012, 'busy'],
        [-32010, 'validation'],
        [-32000, 'configuration_conflict'],
        [-32000, 'fenced'],
        [-32602, 'validation']
    ])('rejects invalid code %s and kind %s in result and status', (code, kind) => {
        const value = { ...fixture(), error: { code, kind, message: 'invalid pair' } }
        expect(() => parseEngineSwitchResult({ receipt: value })).toThrow(
            'Invalid engine switch response.'
        )
        expect(() =>
            parseEngineSwitchStatus({ configurationRevision: revision, receipt: value })
        ).toThrow('Invalid engine switch response.')
    })
    it('accepts an idle status and separately observed states', () => {
        expect(parseEngineSwitchStatus({ configurationRevision: revision })).toEqual({
            configurationRevision: revision
        })
        const status = {
            configurationRevision: revision,
            activeRequestId: 'switch-1',
            receipt: fixture(),
            observed: { source: observed, target: observed }
        }
        expect(parseEngineSwitchStatus(status)).toEqual(status)
    })
    const badStatuses: JsonValue[] = [
        {},
        { configurationRevision: 'bad' },
        { configurationRevision: revision, receipt: null },
        { configurationRevision: revision, observed: {} }
    ]
    it.each(badStatuses)('rejects malformed status %j', row => {
        expect(() => parseEngineSwitchStatus(row)).toThrow()
    })
    it('preserves remote notification identity and sequence', () => {
        const change = { nodeId: 'remote-node', requestId: 'switch-1', sequence: 42 }
        expect(parseEngineSwitchChanged(change)).toEqual(change)
    })
    const badChanges: JsonValue[] = [
        null,
        {},
        { nodeId: '', requestId: 'switch-1', sequence: 1 },
        { nodeId: 'n', requestId: 'bad id', sequence: 1 },
        { nodeId: 'n', requestId: 's', sequence: -1 },
        { nodeId: 'n', requestId: 's', sequence: 1, receipt: {} }
    ]
    it.each(badChanges)('rejects malformed notification %j', row => {
        expect(() => parseEngineSwitchChanged(row)).toThrow()
    })
})

describe('typed engine switch transport', () => {
    it.each([-32013, -32012, -32010, -32011, -32000, -32602])(
        'preserves numeric refusal %s without classifying message text',
        async code => {
            const preload: PreloadServiceTransport = {
                connected: true,
                invoke: async () => ({ success: false, error: '-32011 misleading text', code }),
                invokeHistory: vi.fn<PreloadServiceTransport['invokeHistory']>(() => {
                    throw new Error('Unexpected history invoke in ordinary transport fixture')
                }),
                subscribePush: () => () => {},
                onConnect: () => () => {},
                onDisconnect: () => () => {},
                onAuthFailure: () => () => {},
                destroy: () => {}
            }
            const api = createEngineApi(serviceTransport(preload))
            await expect(
                api.switchStatus({ requestId: 'switch-1', afterSequence: 1 })
            ).rejects.toSatisfy(
                (error: Error) =>
                    error instanceof ServiceCallError && serviceErrorCode(error) === code
            )
        }
    )
    it('forwards switch requests, status sequence floors and push subscriptions', async () => {
        const result = parseEngineSwitchResult({ receipt: fixture() })
        const invoke = vi.fn<PreloadServiceTransport['invoke']>(async () => ({
            success: false,
            error: 'fixture refusal'
        }))
        const subscribePush = vi.fn<PreloadServiceTransport['subscribePush']>(() => () => {})
        const preload: PreloadServiceTransport = {
            connected: true,
            invoke,
            invokeHistory: vi.fn<PreloadServiceTransport['invokeHistory']>(() => {
                throw new Error('Unexpected history invoke in ordinary transport fixture')
            }),
            subscribePush,
            onConnect: () => () => {},
            onDisconnect: () => () => {},
            onAuthFailure: () => () => {},
            destroy: () => {}
        }
        const api = createEngineApi(serviceTransport(preload))
        const receipt: SwitchReceipt = result.receipt
        await expect(api.switchEngine(receipt.request)).rejects.toThrow('fixture refusal')
        expect(invoke).toHaveBeenCalledWith('engines:switch', receipt.request)
        await expect(
            api.switchStatus({ requestId: receipt.requestId, afterSequence: receipt.sequence })
        ).rejects.toThrow('fixture refusal')
        expect(invoke).toHaveBeenLastCalledWith('engines:switch-status', {
            requestId: 'switch-1',
            afterSequence: 1
        })
        const callback = vi.fn<Parameters<typeof api.onSwitchChanged>[0]>()
        api.onSwitchChanged(callback)
        expect(subscribePush).toHaveBeenCalledWith('engines:switch-changed', callback)
    })
})
