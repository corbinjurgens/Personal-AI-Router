// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { SwitchChanged, SwitchReceipt, SwitchRequest } from '@/shared/types/engine-switch'
import type { PreloadServiceTransport } from '@/shared/types/service-bridge'
import type { NodePolicyDocument } from '@/shared/types/node-policy'
import type { IClusterApi, INodePolicyApi } from '@/ui/api/pair-api'
import { createEngineApi } from '@/ui/api/engine-api'
import { serviceTransport } from '@/ui/api/service-transport'
import { useEngineSwitchStore, watchEngineSwitch } from '@/ui/stores/engine-switch.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import {
    useNodeConfigurationStore,
    watchNodeManagement
} from '@/ui/stores/node-configuration.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { emptyEngineStatus } from '@/shared/utils/engines'
import { parseEngineSwitchStatus } from '@/electron/service-bridge/engine-switch'

const revision = `e3v1:${'a'.repeat(64)}`
const request: SwitchRequest = {
    nodeId: 'node-a',
    requestId: 'other-client',
    sourceEngine: 'ollama',
    targetEngine: 'llamacpp',
    expectedRevision: revision
}
function receipt(changes: Partial<SwitchReceipt> = {}): SwitchReceipt {
    const state = {
        intent: 'off',
        running: false,
        healthy: false,
        observedAt: '2026-10-20T00:00:00Z',
        terminationVerified: true
    } satisfies SwitchReceipt['source']
    return {
        version: 1,
        nodeId: 'node-a',
        requestId: request.requestId,
        caller: 'local',
        payloadHash: `sha256:${'b'.repeat(64)}`,
        request,
        expectedRevision: revision,
        acceptedRevision: revision,
        sequence: 1,
        phase: 'accepted',
        terminal: false,
        acceptedAt: '2026-10-20T00:00:00Z',
        updatedAt: '2026-10-20T00:00:00Z',
        deadline: '2026-10-20T00:15:00Z',
        finishedAt: null,
        source: state,
        target: state,
        error: null,
        recoveryRequired: false,
        ...changes
    }
}
function deferred<T>() {
    let resolve: (value: T) => void = () => {
        throw new Error('Deferred not initialized')
    }
    let reject: (error: Error) => void = () => {
        throw new Error('Deferred not initialized')
    }
    const promise = new Promise<T>((finish, fail) => {
        resolve = finish
        reject = fail
    })
    return { promise, resolve, reject }
}
const invoke = vi.fn<PreloadServiceTransport['invoke']>()
const readPolicy = vi.fn<INodePolicyApi['get']>()
const stopListener = vi.fn<() => void>()
let change: (value: SwitchChanged) => void
let releaseDefault: () => void
const stops: Array<() => void> = []
const store = useEngineSwitchStore
async function flush() {
    for (let i = 0; i < 12; i++) await Promise.resolve()
}
async function idle() {
    invoke.mockResolvedValueOnce({ success: true, data: { configurationRevision: revision } })
    stops.push(watchEngineSwitch('node-a'))
    await flush()
}
async function active(value = receipt()) {
    invoke.mockResolvedValueOnce({
        success: true,
        data: { configurationRevision: revision, activeRequestId: 'other-client' }
    })
    invoke.mockResolvedValueOnce({
        success: true,
        data: { configurationRevision: revision, receipt: value }
    })
    stops.push(watchEngineSwitch('node-a'))
    await flush()
}

beforeEach(() => {
    vi.useFakeTimers()
    const pending = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
    releaseDefault = () => pending.resolve({ success: false, error: 'test transport closed' })
    invoke.mockReset().mockReturnValue(pending.promise)
    stopListener.mockReset()
    const api = createEngineApi(
        serviceTransport({
            connected: true,
            invoke,
            invokeHistory: vi.fn<PreloadServiceTransport['invokeHistory']>(() => {
                throw new Error('Unexpected history invoke in ordinary transport fixture')
            }),
            subscribePush: () => stopListener,
            onConnect: () => stopListener,
            onDisconnect: () => stopListener,
            onAuthFailure: () => stopListener,
            destroy: () => {}
        })
    )
    vi.spyOn(api, 'onSwitchChanged').mockImplementation(callback => {
        change = callback
        return stopListener
    })
    readPolicy.mockReset().mockRejectedValue(new Error('Policy unavailable'))
    vi.stubGlobal('window', { pairApi: { engines: api, nodePolicy: { get: readPolicy } } })
    useConnectionStore.setState({ connected: true, selfId: 'node-a' })
    store.setState({ entries: {} })
    usePendingActionsStore.setState({ switches: new Map() })
    useEngineStatusStore.setState({ statusByNode: new Map() })
})
afterEach(async () => {
    stops.splice(0).forEach(stop => stop())
    releaseDefault()
    await flush()
    vi.clearAllTimers()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
})

describe('node-scoped engine switch store', () => {
    it('Refresh waits for a fresh permission read after an older pre-grant read', async () => {
        useConnectionStore.setState({ selfId: 'controller' })
        const older = deferred<Awaited<ReturnType<IClusterApi['getRemoteManagementForNode']>>>()
        const granted = deferred<Awaited<ReturnType<IClusterApi['getRemoteManagementForNode']>>>()
        const read = vi
            .fn<IClusterApi['getRemoteManagementForNode']>()
            .mockReturnValueOnce(older.promise)
            .mockReturnValueOnce(granted.promise)
        vi.stubGlobal('window', {
            pairApi: { ...window.pairApi, cluster: { getRemoteManagementForNode: read } }
        })
        stops.push(watchNodeManagement('node-a', 'node-a'))
        await flush()
        invoke.mockResolvedValue({ success: true, data: { configurationRevision: revision } })
        stops.push(watchEngineSwitch('node-a'))
        let completed = false
        const refresh = store
            .getState()
            .refresh('node-a')
            .then(result => {
                completed = true
                return result
            })
        older.resolve({ version: 1, state: 'unreviewed', allowed: false, revision: '5' })
        await flush()
        expect(read).toHaveBeenCalledTimes(2)
        expect(completed).toBe(false)

        granted.resolve({ version: 1, state: 'allowed', allowed: true, revision: '6' })

        expect(await refresh).toBe(true)
        expect(useNodeConfigurationStore.getState().management.get('node-a')).toMatchObject({
            state: 'allowed',
            revision: '6'
        })
    })

    it('refreshes a permission granted while connected without a denial receipt or reconnect', async () => {
        useConnectionStore.setState({ selfId: 'controller' })
        const read = vi.fn<IClusterApi['getRemoteManagementForNode']>().mockResolvedValue({
            version: 1,
            state: 'unreviewed',
            allowed: false,
            revision: '5'
        })
        vi.stubGlobal('window', {
            pairApi: {
                ...window.pairApi,
                cluster: { getRemoteManagementForNode: read }
            }
        })
        stops.push(watchNodeManagement('node-a', 'node-a'))
        await useNodeConfigurationStore.getState().readManagement('node-a', 'node-a')
        await idle()
        expect(useNodeConfigurationStore.getState().management.get('node-a')?.state).toBe(
            'unreviewed'
        )
        read.mockResolvedValue({ version: 1, state: 'allowed', allowed: true, revision: '6' })
        invoke.mockResolvedValueOnce({ success: true, data: { configurationRevision: revision } })

        expect(await store.getState().refresh('node-a')).toBe(true)

        expect(useNodeConfigurationStore.getState().management.get('node-a')).toMatchObject({
            state: 'allowed',
            revision: '6'
        })
        expect(useConnectionStore.getState().connected).toBe(true)
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            0
        )
    })

    it('reads fresh policy for a retained veto receipt without resubmitting the switch', async () => {
        readPolicy.mockResolvedValue({
            policy: JSON.stringify({ availability: 'paused', pausedBy: 'peer' }),
            revision: '5',
            activation: null,
            recoveryRequired: false,
            availability: 'draining'
        })
        await active(
            receipt({
                terminal: true,
                phase: 'failed-validation',
                error: { code: -32011, kind: 'remote_veto', message: 'refused' }
            })
        )
        expect(readPolicy).toHaveBeenCalledExactlyOnceWith(undefined)
        expect(store.getState().entries['node-a'].refusalPolicy?.availability).toBe('draining')
        expect(store.getState().entries['node-a'].receipt?.source).toEqual(receipt().source)
        expect(store.getState().entries['node-a'].receipt?.target).toEqual(receipt().target)
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            0
        )
    })

    describe('a refusal whose classification read is outstanding', () => {
        const paused: NodePolicyDocument = {
            policy: JSON.stringify({ availability: 'paused', pausedBy: 'peer' }),
            revision: '5',
            activation: null,
            recoveryRequired: false,
            availability: 'paused'
        }
        async function refuse() {
            await idle()
            const policy = deferred<NodePolicyDocument>()
            readPolicy.mockReturnValue(policy.promise)
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision }
            })
            invoke.mockResolvedValueOnce({ success: false, error: 'refused', code: -32011 })
            const submitting = store.getState().submit('node-a', 'ollama', 'llamacpp')
            await flush()
            return { policy, submitting }
        }

        it('releases the controls before the read settles, then applies a guarded late result', async () => {
            const { policy, submitting } = await refuse()
            expect(readPolicy).toHaveBeenCalledTimes(1)
            expect(usePendingActionsStore.getState().switches.has('node-a')).toBe(false)
            expect(store.getState().entries['node-a'].refused).toBe(
                'Target refused: refused before acceptance: refused (-32011).'
            )
            policy.resolve(paused)
            await submitting
            expect(store.getState().entries['node-a'].refused).toBe(
                'Node paused: refused before acceptance: refused (-32011).'
            )
        })

        it('drops a late result once the request is no longer the retained one', async () => {
            const { policy, submitting } = await refuse()
            store.setState({
                entries: {
                    'node-a': {
                        ...store.getState().entries['node-a'],
                        request: undefined,
                        refused: undefined
                    }
                }
            })
            policy.resolve(paused)
            await submitting
            expect(store.getState().entries['node-a'].refused).toBeUndefined()
            expect(store.getState().entries['node-a'].refusalPolicy).toBeUndefined()
        })
    })

    it.each([
        { name: 'lock', code: -32013, message: 'engine is locked', label: 'Engine locked' },
        {
            name: 'authoritative pause',
            code: -32011,
            message: 'refused',
            availability: 'paused',
            label: 'Node paused'
        },
        {
            name: 'authoritative draining',
            code: -32011,
            message: 'refused',
            availability: 'draining',
            label: 'Node paused'
        },
        {
            name: 'bounded pause message without readback',
            code: -32011,
            message: 'This node is paused. Resume it on this node before managing it remotely.',
            label: 'Node paused'
        },
        {
            name: 'older target saved Off',
            code: -32011,
            message: 'engine was turned off locally',
            availability: 'available',
            label: 'Target refused'
        },
        {
            name: 'scoped local stop',
            code: -32011,
            message: 'local stop cancelled this switch',
            availability: 'available',
            label: 'Target refused'
        },
        { name: 'unavailable policy', code: -32011, message: 'refused', label: 'Target refused' },
        {
            name: 'permission',
            code: -32012,
            message: 'permission denied',
            label: 'Remote management permission denied'
        },
        {
            name: 'conflict',
            code: -32010,
            message: 'revision changed',
            label: 'Configuration conflict'
        }
    ] satisfies {
        name: string
        code: number
        message: string
        availability?: NodePolicyDocument['availability']
        label: string
    }[])(
        'classifies $name from fresh target evidence without resending',
        async ({ code, message, availability, label }) => {
            await idle()
            vi.spyOn(useNodeConfigurationStore.getState(), 'readManagement').mockResolvedValue()
            if (availability)
                readPolicy.mockResolvedValue({
                    policy: JSON.stringify({ availability, pausedBy: 'peer' }),
                    revision: '5',
                    activation: null,
                    recoveryRequired: false,
                    availability
                })
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision }
            })
            invoke.mockResolvedValueOnce({ success: false, error: message, code })
            await store.getState().submit('node-a', 'ollama', 'llamacpp')
            expect(store.getState().entries['node-a'].refused).toBe(
                `${label}: refused before acceptance: ${message} (${code}).`
            )
            expect(store.getState().entries['node-a'].unknown).toBe(false)
            expect(readPolicy).toHaveBeenCalledTimes(code === -32011 ? 1 : 0)
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision }
            })
            await store.getState().refresh('node-a')
            await vi.advanceTimersByTimeAsync(1000)
            expect(
                invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')
            ).toHaveLength(1)
        }
    )

    it('does not submit a known locked target and allows leaving a locked source', async () => {
        await idle()
        useEngineStatusStore.setState({
            statusByNode: new Map([
                [
                    'node-a',
                    new Map([
                        ['llama-cpp', { ...emptyEngineStatus('node-a', 'llama-cpp'), locked: true }]
                    ])
                ]
            ])
        })
        await store.getState().submit('node-a', 'ollama', 'llamacpp')
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            0
        )
        invoke.mockResolvedValueOnce({ success: true, data: { configurationRevision: revision } })
        invoke.mockResolvedValueOnce({ success: false, error: 'revision changed', code: -32010 })
        await store.getState().submit('node-a', 'llamacpp', 'ollama')
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toEqual([
            [
                'engines:switch',
                expect.objectContaining({ sourceEngine: 'llamacpp', targetEngine: 'ollama' })
            ]
        ])
    })

    it.each([
        { outcome: 'observed', sequence: 1 },
        { outcome: 'error', sequence: 1 },
        { outcome: 'missing', sequence: 1 },
        { outcome: 'advanced receipt', sequence: 2 }
    ])(
        'r2 older equal-sequence wait $outcome cannot replace newer refresh evidence',
        async ({ outcome, sequence }) => {
            const waiting = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
            const first = receipt()
            const old = { source: first.source, target: first.target }
            const fresh = {
                source: {
                    ...first.source,
                    running: true,
                    observedAt: '2026-10-21T00:00:00Z'
                },
                target: first.target
            }
            const baseline = {
                success: true,
                data: { configurationRevision: revision, activeRequestId: request.requestId }
            } satisfies Awaited<ReturnType<PreloadServiceTransport['invoke']>>
            invoke.mockResolvedValueOnce(baseline)
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision, receipt: first, observed: old }
            })
            invoke.mockReturnValueOnce(waiting.promise)
            stops.push(watchEngineSwitch('node-a'))
            await flush()
            const retained = store.getState().entries['node-a'].receipt
            invoke.mockResolvedValueOnce(baseline)
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision, receipt: first, observed: fresh }
            })
            await store.getState().refresh('node-a')
            expect(store.getState().entries['node-a'].observed).toEqual(fresh)
            expect(store.getState().entries['node-a'].receipt).toBe(retained)
            if (outcome === 'error') waiting.reject(new Error('older wait failed'))
            else
                waiting.resolve({
                    success: true,
                    data: {
                        configurationRevision: revision,
                        ...(outcome === 'observed' || outcome === 'advanced receipt'
                            ? { receipt: receipt({ sequence }), observed: old }
                            : {})
                    }
                })
            await flush()
            expect(store.getState().entries['node-a'].observed).toEqual(fresh)
            expect(store.getState().entries['node-a'].unavailable).toBeUndefined()
            expect(store.getState().entries['node-a'].receipt?.sequence).toBe(sequence)
            if (sequence === first.sequence)
                expect(store.getState().entries['node-a'].receipt).toBe(retained)
            const calls = invoke.mock.calls.length
            await vi.advanceTimersByTimeAsync(0)
            expect(invoke.mock.calls.length).toBe(calls + 1)
            expect(invoke).toHaveBeenLastCalledWith('engines:switch-status', {
                nodeId: 'node-a',
                requestId: request.requestId,
                afterSequence: sequence
            })
        }
    )
    it('permission refusal delivered after remount must not strand its original ID', async () => {
        useConnectionStore.setState({ selfId: 'self' })
        const permission = vi
            .spyOn(useNodeConfigurationStore.getState(), 'readManagement')
            .mockImplementation(async () => {
                useNodeConfigurationStore.setState({
                    management: new Map([['node-a', { state: 'allowed', revision: '2' }]])
                })
            })
        const sent = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
        invoke.mockImplementation(async (channel, payload) => {
            if (channel === 'engines:switch') return sent.promise
            if (typeof payload === 'object' && payload !== null && 'requestId' in payload)
                return {
                    success: false,
                    code: -32000,
                    error: 'Unknown switch receipt for this caller.'
                }
            return { success: true, data: { configurationRevision: revision } }
        })
        stops.push(watchEngineSwitch('node-a'))
        await flush()
        const submit = store.getState().submit('node-a', 'ollama', 'llamacpp')
        await flush()
        const original = store.getState().entries['node-a'].request
        expect(original).toBeDefined()
        stops.splice(0).forEach(stop => stop())
        stops.push(watchEngineSwitch('node-a'))
        await flush()
        sent.resolve({
            success: false,
            error: 'permission revoked before acceptance',
            code: -32012
        })
        await submit
        expect(store.getState().entries['node-a'].unknown).toBe(false)
        // Initial/remounted status refresh and submit's refresh now also read permission.
        expect(permission).toHaveBeenCalledTimes(5)
        expect(store.getState().entries['node-a'].refused).toContain('refused before acceptance')
        expect(store.getState().entries['node-a'].request).toBe(original)
        expect(store.getState().reserved('node-a', 'ollama')).toBe(false)
        expect(store.getState().reserved('node-a', 'llamacpp')).toBe(false)
        for (let round = 0; round < 3; round++) await store.getState().refresh('node-a')
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            1
        )
        await store.getState().submit('node-a', 'ollama', 'llamacpp')
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            2
        )
        expect(store.getState().entries['node-a'].request?.requestId).not.toBe(original?.requestId)
    })
    it('unchanged wait refreshes separate current observations', async () => {
        const waiting = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
        const first = receipt()
        const old = { source: first.source, target: first.target }
        const fresh = {
            source: { ...first.source, running: true, observedAt: '2026-10-21T00:00:00Z' },
            target: first.target
        }
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: revision, activeRequestId: request.requestId }
        })
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: revision, receipt: first, observed: old }
        })
        invoke.mockReturnValueOnce(waiting.promise)
        stops.push(watchEngineSwitch('node-a'))
        await flush()
        const retained = store.getState().entries['node-a'].receipt
        expect(store.getState().entries['node-a'].observed).toEqual(old)
        waiting.resolve({
            success: true,
            data: { configurationRevision: revision, receipt: first, observed: fresh }
        })
        await flush()
        expect(store.getState().entries['node-a'].observed).toEqual(fresh)
        expect(store.getState().entries['node-a'].receipt).toBe(retained)
    })
    it.each(['accepted receipt', 'typed refusal'])(
        'retains a late %s without restarting retired view reads',
        async outcome => {
            await idle()
            const sent = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision }
            })
            invoke.mockReturnValueOnce(sent.promise)
            const submit = store.getState().submit('node-a', 'ollama', 'llamacpp')
            await flush()
            const original = store.getState().entries['node-a'].request
            if (!original) throw new Error('Missing submitted request')
            stops.splice(0).forEach(stop => stop())
            const calls = invoke.mock.calls.length
            sent.resolve(
                outcome === 'typed refusal'
                    ? { success: false, error: 'refused', code: -32011 }
                    : {
                          success: true,
                          data: {
                              receipt: receipt({ request: original, requestId: original.requestId })
                          }
                      }
            )
            await submit
            await vi.advanceTimersByTimeAsync(1)
            expect(store.getState().entries['node-a'].unknown).toBe(false)
            expect(store.getState().entries['node-a'].revision).toBeUndefined()
            expect(invoke).toHaveBeenCalledTimes(calls)
        }
    )
    it.each([
        { outcome: 'typed refusal', newer: 'request' },
        { outcome: 'typed refusal', newer: 'receipt' },
        { outcome: 'accepted receipt', newer: 'request' },
        { outcome: 'accepted receipt', newer: 'receipt' }
    ])('ignores late $outcome after retirement with a newer $newer', async ({ outcome, newer }) => {
        await idle()
        const sent = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
        invoke.mockResolvedValueOnce({ success: true, data: { configurationRevision: revision } })
        invoke.mockReturnValueOnce(sent.promise)
        const submit = store.getState().submit('node-a', 'ollama', 'llamacpp')
        await flush()
        const original = store.getState().entries['node-a'].request
        if (!original) throw new Error('Missing submitted request')
        stops.splice(0).forEach(stop => stop())
        const retainedRequest = newer === 'request' ? request : original
        const retained = {
            request: retainedRequest,
            receipt: receipt({
                request: retainedRequest,
                requestId: retainedRequest.requestId,
                sequence: 9
            }),
            unknown: false
        }
        store.setState({ entries: { 'node-a': retained } })
        sent.resolve(
            outcome === 'typed refusal'
                ? { success: false, error: 'refused', code: -32011 }
                : {
                      success: true,
                      data: {
                          receipt: receipt({ request: original, requestId: original.requestId })
                      }
                  }
        )
        await submit
        expect(store.getState().entries['node-a']).toBe(retained)
    })
    it.each([
        { name: 'wrong node', change: { nodeId: 'node-b' } },
        { name: 'wrong request', change: { requestId: 'unrelated' } },
        { name: 'lower sequence', change: { sequence: 0 } }
    ])('rejects observations from a wait with $name', async ({ change: changes }) => {
        const waiting = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: revision, activeRequestId: request.requestId }
        })
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: revision, receipt: receipt() }
        })
        invoke.mockReturnValueOnce(waiting.promise)
        stops.push(watchEngineSwitch('node-a'))
        await flush()
        const before = store.getState().entries['node-a']
        const invalid = receipt(changes)
        waiting.resolve({
            success: true,
            data: {
                configurationRevision: revision,
                receipt: invalid,
                observed: { source: invalid.source, target: invalid.target }
            }
        })
        await flush()
        expect(store.getState().entries['node-a']).toBe(before)
    })
    it.each([
        { phase: 'completed', terminal: true, outcome: 'rejects', read: 'refresh' },
        { phase: 'completed', terminal: true, outcome: 'missing receipt', read: 'notification' },
        { phase: 'draining', terminal: false, outcome: 'rejects', read: 'notification' },
        { phase: 'draining', terminal: false, outcome: 'missing receipt', read: 'refresh' }
    ] satisfies Array<{
        phase: SwitchReceipt['phase']
        terminal: boolean
        outcome: string
        read: string
    }>)(
        'preserves newer $phase evidence from $read when the same-request wait $outcome',
        async ({ phase, terminal, outcome, read }) => {
            const old = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision, activeRequestId: request.requestId }
            })
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision, receipt: receipt() }
            })
            invoke.mockReturnValueOnce(old.promise)
            stops.push(watchEngineSwitch('node-a'))
            await flush()
            expect(invoke).toHaveBeenLastCalledWith('engines:switch-status', {
                nodeId: 'node-a',
                requestId: 'other-client',
                afterSequence: 1
            })
            invoke.mockResolvedValueOnce({
                success: true,
                data: {
                    configurationRevision: revision,
                    activeRequestId: terminal ? undefined : request.requestId
                }
            })
            invoke.mockResolvedValueOnce({
                success: true,
                data: {
                    configurationRevision: revision,
                    receipt: receipt({ sequence: 2, phase, terminal })
                }
            })
            if (read === 'notification') {
                change({ nodeId: 'node-a', requestId: request.requestId, sequence: 2 })
                await flush()
            } else await store.getState().refresh('node-a')
            const authoritative = store.getState().entries['node-a']
            expect(authoritative.receipt).toMatchObject({ sequence: 2, phase, terminal })
            expect(authoritative.revision).toBe(revision)
            expect(authoritative.unavailable).toBeUndefined()
            expect(invoke).toHaveBeenCalledTimes(5)
            if (outcome === 'rejects') old.reject(new Error('old sequence-1 wait failed'))
            else old.resolve({ success: true, data: { configurationRevision: revision } })
            await flush()
            expect(store.getState().entries['node-a']).toBe(authoritative)
            expect(authoritative.unknown).toBe(false)
            await vi.advanceTimersByTimeAsync(1)
            expect(invoke).toHaveBeenCalledTimes(terminal ? 5 : 6)
            if (!terminal)
                expect(invoke).toHaveBeenLastCalledWith('engines:switch-status', {
                    nodeId: 'node-a',
                    requestId: 'other-client',
                    afterSequence: 2
                })
            expect(
                invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')
            ).toHaveLength(0)
        }
    )
    it.each(['rejects', 'missing receipt'])(
        'reports a current wait that %s without newer evidence and stops until refresh',
        async outcome => {
            const waiting = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision, activeRequestId: request.requestId }
            })
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision, receipt: receipt() }
            })
            invoke.mockReturnValueOnce(waiting.promise)
            stops.push(watchEngineSwitch('node-a'))
            await flush()
            if (outcome === 'rejects') waiting.reject(new Error('current wait failed'))
            else waiting.resolve({ success: true, data: { configurationRevision: revision } })
            await flush()
            expect(store.getState().entries['node-a'].unavailable).toBe(
                outcome === 'rejects'
                    ? 'Switch receipt unavailable: current wait failed'
                    : 'Switch receipt unavailable. Refresh readback.'
            )
            expect(store.getState().entries['node-a'].receipt).toMatchObject({
                sequence: 1,
                terminal: false
            })
            await vi.advanceTimersByTimeAsync(1)
            expect(invoke).toHaveBeenCalledTimes(3)
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision, activeRequestId: request.requestId }
            })
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision, receipt: receipt() }
            })
            await store.getState().refresh('node-a')
            expect(store.getState().entries['node-a'].unavailable).toBeUndefined()
            expect(invoke).toHaveBeenCalledTimes(6)
            expect(invoke).toHaveBeenLastCalledWith('engines:switch-status', {
                nodeId: 'node-a',
                requestId: 'other-client',
                afterSequence: 1
            })
        }
    )
    it.each(['rejects', 'resolves without receipt'])(
        'ignores A wait that %s after B becomes current',
        async outcome => {
            const old = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision, activeRequestId: request.requestId }
            })
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision, receipt: receipt() }
            })
            invoke.mockReturnValueOnce(old.promise)
            stops.push(watchEngineSwitch('node-a'))
            await flush()
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision }
            })
            invoke.mockResolvedValueOnce({
                success: true,
                data: {
                    configurationRevision: revision,
                    receipt: receipt({ sequence: 2, phase: 'completed', terminal: true })
                }
            })
            await store.getState().refresh('node-a')
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision }
            })
            invoke.mockResolvedValueOnce({
                success: true,
                data: {
                    configurationRevision: revision,
                    receipt: receipt({ sequence: 2, phase: 'completed', terminal: true })
                }
            })
            invoke.mockImplementationOnce(async (_channel, payload) => {
                const sent = store.getState().entries['node-a'].request
                if (!sent || payload !== sent) throw new Error('Missing B request')
                return {
                    success: true,
                    data: {
                        receipt: receipt({
                            request: sent,
                            requestId: sent.requestId,
                            phase: 'draining'
                        })
                    }
                }
            })
            const waitingB = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
            invoke.mockReturnValueOnce(waitingB.promise)
            await store.getState().submit('node-a', 'llamacpp', 'ollama')
            const before = store.getState().entries['node-a']
            expect(before.request?.requestId).not.toBe(request.requestId)
            expect(before.receipt?.phase).toBe('draining')
            if (outcome === 'rejects') old.reject(new Error('old A transport timeout'))
            else old.resolve({ success: true, data: { configurationRevision: revision } })
            await flush()
            expect(store.getState().entries['node-a']).toBe(before)
            waitingB.resolve({
                success: true,
                data: { configurationRevision: revision, receipt: before.receipt }
            })
            await flush()
            await vi.advanceTimersByTimeAsync(1)
            expect(
                invoke.mock.calls.filter(
                    ([channel, payload]) =>
                        channel === 'engines:switch-status' &&
                        typeof payload === 'object' &&
                        payload !== null &&
                        'requestId' in payload &&
                        payload.requestId === before.request?.requestId &&
                        'afterSequence' in payload &&
                        payload.afterSequence === 1
                )
            ).toHaveLength(2)
        }
    )
    it.each<SwitchReceipt['phase']>(['completed', 'failed-interrupted'])(
        'keeps %s readback authoritative when submit later rejects',
        async phase => {
            await idle()
            const sent = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision }
            })
            invoke.mockReturnValueOnce(sent.promise)
            const submitted = store.getState().submit('node-a', 'ollama', 'llamacpp')
            await flush()
            const currentRequest = store.getState().entries['node-a'].request
            if (!currentRequest) throw new Error('Missing submitted request')
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision }
            })
            invoke.mockResolvedValueOnce({
                success: true,
                data: {
                    configurationRevision: revision,
                    receipt: receipt({
                        request: currentRequest,
                        requestId: currentRequest.requestId,
                        sequence: 2,
                        phase,
                        terminal: true,
                        recoveryRequired: phase === 'failed-interrupted',
                        error:
                            phase === 'failed-interrupted'
                                ? { code: -32000, kind: 'interrupted', message: 'interrupted' }
                                : null
                    })
                }
            })
            change({ nodeId: 'node-a', requestId: currentRequest.requestId, sequence: 2 })
            await flush()
            expect(usePendingActionsStore.getState().switches.has('node-a')).toBe(false)
            expect(store.getState().reserved('node-a', 'ollama')).toBe(
                phase === 'failed-interrupted'
            )
            expect(store.getState().reserved('node-a', 'llamacpp')).toBe(
                phase === 'failed-interrupted'
            )
            const authoritative = store.getState().entries['node-a']
            expect(authoritative.unknown).toBe(false)
            expect(authoritative.receipt?.phase).toBe(phase)
            sent.reject(new Error('submit reply lost'))
            await submitted
            expect(store.getState().entries['node-a']).toBe(authoritative)
            expect(authoritative.unavailable).toBeUndefined()
            expect(
                invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')
            ).toHaveLength(1)
        }
    )
    it('reads permission denial through the real status parser and refreshes capability', async () => {
        const read = vi
            .spyOn(useNodeConfigurationStore.getState(), 'readManagement')
            .mockResolvedValue()
        const denied = receipt({
            phase: 'failed-validation',
            terminal: true,
            finishedAt: '2026-10-20T00:00:00Z',
            error: { code: -32012, kind: 'validation', message: 'Remote management denied' }
        })
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: revision, activeRequestId: request.requestId }
        })
        invoke.mockImplementationOnce(async () => ({
            success: true,
            data: parseEngineSwitchStatus(
                JSON.parse(JSON.stringify({ configurationRevision: revision, receipt: denied }))
            )
        }))
        stops.push(watchEngineSwitch('node-a'))
        await flush()
        expect(store.getState().entries['node-a'].receipt?.error).toEqual({
            code: -32012,
            kind: 'validation',
            message: 'Remote management denied'
        })
        expect(store.getState().entries['node-a'].unavailable).toBeUndefined()
        expect(read).toHaveBeenCalledWith('node-a', undefined)
    })
    it('subscribes before baseline and reference counts cleanup', async () => {
        await idle()
        expect(
            vi.mocked(window.pairApi.engines.onSwitchChanged).mock.invocationCallOrder[0]
        ).toBeLessThan(invoke.mock.invocationCallOrder[0])
        const second = watchEngineSwitch('node-a')
        expect(invoke).toHaveBeenCalledTimes(1)
        stops[0]()
        stops[0]()
        expect(stopListener).not.toHaveBeenCalled()
        second()
        expect(stopListener).toHaveBeenCalledTimes(2)
        expect(store.getState().entries['node-a'].revision).toBeUndefined()
    })
    it('sends one deliberate request with fresh revision and exact source and target', async () => {
        await idle()
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: 'fresh-revision' }
        })
        const sent = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
        invoke.mockReturnValueOnce(sent.promise)
        const first = store.getState().submit('node-a', 'ollama', 'llamacpp')
        const second = store.getState().submit('node-a', 'llamacpp', 'ollama')
        await flush()
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toEqual([
            [
                'engines:switch',
                {
                    nodeId: 'node-a',
                    requestId: expect.any(String),
                    sourceEngine: 'ollama',
                    targetEngine: 'llamacpp',
                    expectedRevision: 'fresh-revision'
                }
            ]
        ])
        const correlation = store.getState().entries['node-a'].request
        if (!correlation) throw new Error('Missing request')
        sent.resolve({
            success: true,
            data: {
                receipt: receipt({
                    request: correlation,
                    requestId: correlation.requestId,
                    expectedRevision: 'fresh-revision'
                })
            }
        })
        await Promise.all([first, second])
        expect(store.getState().entries['node-a'].receipt?.phase).toBe('accepted')
        expect(store.getState().reserved('node-a', 'ollama')).toBe(true)
        expect(store.getState().reserved('node-a', 'llamacpp')).toBe(true)
        expect(store.getState().reserved('node-a', 'unrelated')).toBe(false)
    })
    it('refuses locally when fresh readback is unavailable', async () => {
        await idle()
        invoke.mockResolvedValueOnce({ success: false, error: 'offline' })
        await store.getState().submit('node-a', 'ollama', 'llamacpp')
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            0
        )
        expect(store.getState().entries['node-a'].unavailable).toContain('readback unavailable')
    })
    it('discovers another client active request and waits with afterSequence', async () => {
        await active()
        expect(store.getState().entries['node-a'].own).toBe(false)
        expect(store.getState().entries['node-a'].receipt?.requestId).toBe('other-client')
        expect(invoke).toHaveBeenLastCalledWith('engines:switch-status', {
            nodeId: 'node-a',
            requestId: 'other-client',
            afterSequence: 1
        })
        await store.getState().submit('node-a', 'ollama', 'llamacpp')
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            0
        )
    })
    it('shows busy and unavailable when another caller receipt cannot be read', async () => {
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: revision, activeRequestId: 'foreign' }
        })
        invoke.mockResolvedValueOnce({ success: false, error: 'unknown receipt' })
        stops.push(watchEngineSwitch('node-a'))
        await flush()
        expect(store.getState().entries['node-a'].activeRequestId).toBe('foreign')
        expect(store.getState().entries['node-a'].unavailable).toContain('unknown receipt')
        await store.getState().submit('node-a', 'ollama', 'llamacpp')
        expect(invoke).toHaveBeenCalledTimes(2)
    })
    it.each(['node mismatch', 'request mismatch'])('ignores %s receipt', async mismatch => {
        const wrong = mismatch === 'node mismatch' ? { nodeId: 'node-b' } : { requestId: 'wrong' }
        await active(receipt(wrong))
        expect(store.getState().entries['node-a'].receipt).toBeUndefined()
        expect(store.getState().entries['node-a'].unavailable).toContain('receipt unavailable')
    })
    it('keeps duplicate and out-of-order receipts idempotent', async () => {
        const waiting = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: revision, activeRequestId: 'other-client' }
        })
        invoke.mockResolvedValueOnce({
            success: true,
            data: {
                configurationRevision: revision,
                receipt: receipt({ sequence: 4, phase: 'draining' })
            }
        })
        invoke.mockReturnValueOnce(waiting.promise)
        stops.push(watchEngineSwitch('node-a'))
        await flush()
        const original = store.getState().entries['node-a'].receipt
        waiting.resolve({
            success: true,
            data: {
                configurationRevision: revision,
                receipt: receipt({ sequence: 4, phase: 'target-ready' })
            }
        })
        await flush()
        expect(store.getState().entries['node-a'].receipt).toBe(original)
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: revision, receipt: receipt({ sequence: 2 }) }
        })
        await vi.advanceTimersByTimeAsync(1)
        expect(store.getState().entries['node-a'].receipt).toBe(original)
    })
    it('unchanged bounded waits repeat without fabricating completion or parallel waits', async () => {
        const waiting = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: revision, activeRequestId: 'other-client' }
        })
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: revision, receipt: receipt() }
        })
        invoke.mockReturnValueOnce(waiting.promise)
        stops.push(watchEngineSwitch('node-a'))
        await flush()
        expect(invoke).toHaveBeenCalledTimes(3)
        waiting.resolve({
            success: true,
            data: { configurationRevision: revision, receipt: receipt() }
        })
        await flush()
        await vi.advanceTimersByTimeAsync(1)
        expect(invoke).toHaveBeenCalledTimes(4)
        expect(store.getState().entries['node-a'].receipt?.phase).toBe('accepted')
        expect(store.getState().entries['node-a'].receipt?.terminal).toBe(false)
    })
    it('newer notification invalidates an old baseline and is never itself a receipt', async () => {
        const old = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
        invoke.mockReturnValueOnce(old.promise)
        stops.push(watchEngineSwitch('node-a'))
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: 'new-revision' }
        })
        change({ nodeId: 'node-a', requestId: 'unrelated', sequence: 99 })
        await flush()
        old.resolve({ success: true, data: { configurationRevision: revision } })
        await flush()
        expect(store.getState().entries['node-a'].revision).toBe('new-revision')
        expect(store.getState().entries['node-a'].receipt).toBeUndefined()
        change({ nodeId: 'node-b', requestId: 'unrelated', sequence: 100 })
        expect(invoke).toHaveBeenCalledTimes(2)
    })
    it('disconnect mid-wait retires callbacks; reconnect reads before enabling submission', async () => {
        const waiting = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: revision, activeRequestId: 'other-client' }
        })
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: revision, receipt: receipt() }
        })
        invoke.mockReturnValueOnce(waiting.promise)
        stops.push(watchEngineSwitch('node-a'))
        await flush()
        useConnectionStore.setState({ connected: false })
        waiting.resolve({
            success: true,
            data: {
                configurationRevision: revision,
                receipt: receipt({ phase: 'completed', terminal: true, sequence: 9 })
            }
        })
        await flush()
        expect(store.getState().entries['node-a'].receipt?.phase).toBe('accepted')
        expect(store.getState().entries['node-a'].revision).toBeUndefined()
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: 'reconnected' }
        })
        useConnectionStore.setState({ connected: true })
        expect(store.getState().entries['node-a'].revision).toBeUndefined()
        await flush()
        expect(store.getState().entries['node-a'].revision).toBe('reconnected')
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            0
        )
    })
    it('last unsubscribe retires delayed baseline and wait callbacks', async () => {
        const delayed = deferred<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>()
        invoke.mockReturnValueOnce(delayed.promise)
        const stop = watchEngineSwitch('node-a')
        stop()
        delayed.resolve({ success: true, data: { configurationRevision: revision } })
        await flush()
        expect(store.getState().entries['node-a'].revision).toBeUndefined()
        expect(invoke).toHaveBeenCalledTimes(1)
    })
    it.each([undefined, -32000])(
        'indeterminate submit failure %s retains original recovery through absence and remount',
        async code => {
            await idle()
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision }
            })
            invoke.mockResolvedValueOnce({ success: false, error: 'lost response', code })
            await store.getState().submit('node-a', 'ollama', 'llamacpp')
            const original = store.getState().entries['node-a'].request
            if (!original) throw new Error('Missing original identity')
            expect(store.getState().entries['node-a'].unknown).toBe(true)
            expect(store.getState().entries['node-a'].unavailable).toContain('outcome unknown')
            stops.splice(0).forEach(stop => stop())
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision }
            })
            invoke.mockResolvedValueOnce({
                success: false,
                code: -32000,
                error: 'Unknown switch receipt for this caller.'
            })
            stops.push(watchEngineSwitch('node-a'))
            await flush()
            expect(store.getState().entries['node-a'].unknown).toBe(true)
            expect(store.getState().entries['node-a'].request).toBe(original)
            expect(store.getState().reserved('node-a', 'ollama')).toBe(true)
            expect(store.getState().reserved('node-a', 'llamacpp')).toBe(true)
            const beforeRetry = invoke.mock.calls.length
            await store.getState().submit('node-a', 'ollama', 'llamacpp')
            expect(invoke).toHaveBeenCalledTimes(beforeRetry)
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision }
            })
            invoke.mockResolvedValueOnce({
                success: true,
                data: {
                    configurationRevision: revision,
                    receipt: receipt({
                        request: original,
                        requestId: original.requestId,
                        phase: 'completed',
                        terminal: true,
                        sequence: 8
                    })
                }
            })
            await store.getState().refresh('node-a')
            expect(invoke).toHaveBeenLastCalledWith('engines:switch-status', {
                nodeId: 'node-a',
                requestId: original.requestId
            })
            expect(store.getState().entries['node-a'].receipt?.phase).toBe('completed')
            expect(store.getState().entries['node-a'].unknown).toBe(false)
            expect(store.getState().entries['node-a'].refused).toBeUndefined()
            expect(
                invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')
            ).toHaveLength(1)
        }
    )
    it.each([
        [-32012, 'Remote management permission denied'],
        [-32013, 'Engine locked'],
        [-32011, 'Target refused'],
        [-32010, 'Configuration conflict'],
        [-32602, 'Invalid switch request']
    ])(
        'resolves typed refusal %s across readback and remount without stranding the draft',
        async (code, label) => {
            await idle()
            const permission = vi
                .spyOn(useNodeConfigurationStore.getState(), 'readManagement')
                .mockResolvedValue()
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision }
            })
            invoke.mockResolvedValueOnce({ success: false, error: 'refused', code: Number(code) })
            await store.getState().submit('node-a', 'ollama', 'llamacpp')
            const original = store.getState().entries['node-a'].request
            expect(original).toMatchObject({ sourceEngine: 'ollama', targetEngine: 'llamacpp' })
            expect(store.getState().entries['node-a'].unknown).toBe(false)
            expect(store.getState().entries['node-a'].refused).toContain(
                `${label}: refused before acceptance: refused`
            )
            expect(store.getState().reserved('node-a', 'ollama')).toBe(false)
            expect(store.getState().reserved('node-a', 'llamacpp')).toBe(false)
            expect(permission).toHaveBeenCalledTimes(code === -32012 ? 1 : 0)
            // The audit probe's unknown-receipt response must never be needed for
            // a definitely refused ID, even after repeated readback and remount.
            for (let round = 0; round < 3; round++) {
                invoke.mockResolvedValueOnce({
                    success: true,
                    data: { configurationRevision: revision }
                })
                await store.getState().refresh('node-a')
            }
            stops.splice(0).forEach(stop => stop())
            await idle()
            expect(store.getState().entries['node-a'].request).toBe(original)
            expect(store.getState().entries['node-a'].refused).toContain(String(label))
            expect(store.getState().entries['node-a'].unavailable).toBeUndefined()
            expect(
                invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')
            ).toHaveLength(1)
            expect(
                invoke.mock.calls.filter(
                    ([channel, payload]) =>
                        channel === 'engines:switch-status' &&
                        typeof payload === 'object' &&
                        payload !== null &&
                        'requestId' in payload
                )
            ).toHaveLength(0)
            invoke.mockResolvedValueOnce({
                success: true,
                data: { configurationRevision: revision }
            })
            invoke.mockResolvedValueOnce({
                success: false,
                error: String(label),
                code: Number(code)
            })
            await store.getState().submit('node-a', 'ollama', 'llamacpp')
            expect(
                invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')
            ).toHaveLength(2)
            expect(store.getState().entries['node-a'].request?.requestId).not.toBe(
                original?.requestId
            )
        }
    )
    it('terminal failure and recoveryRequired remain independent and reserve both engines', async () => {
        await active(
            receipt({
                terminal: true,
                phase: 'failed-interrupted',
                recoveryRequired: true,
                error: { code: -32000, kind: 'interrupted', message: 'broker restarted' }
            })
        )
        expect(store.getState().entries['node-a'].receipt?.error?.message).toBe('broker restarted')
        expect(store.getState().entries['node-a'].receipt?.recoveryRequired).toBe(true)
        expect(store.getState().reserved('node-a', 'ollama')).toBe(true)
        expect(invoke).toHaveBeenCalledTimes(2)
    })
    it('permission revocation before send prevents mutation', async () => {
        await idle()
        useConnectionStore.setState({ selfId: 'self' })
        await flush()
        const read = vi
            .spyOn(useNodeConfigurationStore.getState(), 'readManagement')
            .mockImplementation(async () => {
                useNodeConfigurationStore.setState({
                    management: new Map([['node-a', { state: 'denied', revision: '2' }]])
                })
            })
        await store.getState().submit('node-a', 'ollama', 'llamacpp')
        expect(read).toHaveBeenCalledWith('node-a', 'node-a')
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            0
        )
    })
})
