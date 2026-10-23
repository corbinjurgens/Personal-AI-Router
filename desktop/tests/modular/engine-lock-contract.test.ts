// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

type Handler = (event: { sender: { getURL: () => string } }, ...args: unknown[]) => Promise<unknown>

const ipc = vi.hoisted(() => ({ handlers: new Map<string, Handler>() }))

vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => process.cwd() },
    BrowserWindow: { getAllWindows: () => [], fromWebContents: () => ({}) },
    ipcMain: {
        handle: (channel: string, handler: Handler) => ipc.handlers.set(channel, handler)
    }
}))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))

import { safeHandle } from '@/electron/ipc/safe-handle'
import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'
import { JsonRpcResponseError } from '@/electron/service-bridge/json-rpc-client'
import type { JsonValue } from '@/electron/service-bridge/json-rpc-client'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import {
    ENGINE_LOCKED_CODE,
    LOCAL_VETO_CODE,
    POLICY_CONFLICT_CODE,
    REMOTE_MANAGEMENT_DENIED_CODE
} from '@/shared/constants/service-errors'
import type { EngineStatusData, EngineType } from '@/shared/types/engines'
import type { IpcResult } from '@/shared/types/ipc-channels'
import type { PreloadServiceTransport } from '@/shared/types/service-bridge'
import type { WsInvokeResponse } from '@/shared/types/ws-channels'
import { ServiceCallError, serviceErrorCode, serviceTransport } from '@/ui/api/service-transport'

const SELF = 'engine-lock-self'
const PEER = 'engine-lock-peer'

afterEach(() => {
    vi.restoreAllMocks()
})

describe('engine lock command dispatch', () => {
    function spySupervisor() {
        const supervisor = getModularSupervisor()
        return {
            send: vi.spyOn(supervisor, 'sendProcess').mockReturnValue(undefined),
            report: vi.spyOn(supervisor, 'reportError').mockReturnValue(undefined)
        }
    }

    it.each([true, false])('sends locked=%s to this device as engine:set-lock', async locked => {
        getModularBridgeState().setSelfId(SELF)
        const { send, report } = spySupervisor()
        await handleServiceBridgeInvoke('engine:command', {
            command: 'setLock',
            engineType: 'lm-studio',
            nodeId: SELF,
            locked
        })
        expect(send).toHaveBeenCalledOnce()
        const [name, method, params, , observeResponse] = send.mock.calls[0]
        expect([name, method, params, observeResponse]).toEqual([
            'broker',
            'engine:set-lock',
            { engine: 'lmstudio', locked },
            true
        ])
        expect(report).not.toHaveBeenCalled()
    })

    it('reports a refused lock change without claiming success', async () => {
        getModularBridgeState().setSelfId(SELF)
        const { send, report } = spySupervisor()
        await handleServiceBridgeInvoke('engine:command', {
            command: 'setLock',
            engineType: 'ollama',
            nodeId: SELF,
            locked: true
        })
        const onFailure = send.mock.calls[0][3]
        onFailure?.('-32000: Engine locking is not available yet.', true)
        expect(report).toHaveBeenCalledWith(
            'Failed to lock ollama: -32000: Engine locking is not available yet.',
            'error',
            `engine-cmd:setLock:${SELF}:ollama`,
            { nodeId: SELF, engineType: 'ollama' }
        )
    })

    it('refuses a lock change for another device before sending', async () => {
        getModularBridgeState().setSelfId(SELF)
        const { send, report } = spySupervisor()
        await handleServiceBridgeInvoke('engine:command', {
            command: 'setLock',
            engineType: 'ollama',
            nodeId: PEER,
            locked: true
        })
        expect(send).not.toHaveBeenCalled()
        expect(report).toHaveBeenCalledWith(
            'Engine locks can only be changed on the device running the engine.',
            'warning',
            `engine-cmd:setLock:${PEER}:ollama`,
            { nodeId: PEER, engineType: 'ollama' }
        )
    })

    it('refuses a lock change without a requested state', async () => {
        getModularBridgeState().setSelfId(SELF)
        const { send, report } = spySupervisor()
        await handleServiceBridgeInvoke('engine:command', {
            command: 'setLock',
            engineType: 'ollama',
            nodeId: SELF
        })
        expect(send).not.toHaveBeenCalled()
        expect(report).toHaveBeenCalledOnce()
    })
})

describe('engine lock status round trips', () => {
    const state = getModularBridgeState()

    beforeAll(() => {
        state.setSelfId(SELF)
        state.handleNotification({
            source: 'lmstudio-proxy',
            method: 'node/discovered',
            params: {
                id: PEER,
                host: PEER,
                port: 1234,
                addresses: ['192.0.2.191'],
                ip: '192.0.2.191'
            }
        })
    })

    function status(nodeId: string, engineType: EngineType): EngineStatusData | undefined {
        return state
            .getEngineInitialState()
            .statuses.find(row => row.nodeId === nodeId && row.engineType === engineType)
    }
    function expectLock(row: EngineStatusData | undefined, locked: boolean | undefined): void {
        expect(row).toBeDefined()
        if (locked === undefined) expect(row).not.toHaveProperty('locked')
        else expect(row).toHaveProperty('locked', locked)
    }
    function engineStatus(engine: string, locked: JsonValue | undefined): JsonValue {
        return {
            engine,
            installed: true,
            running: false,
            healthy: false,
            port: 1235,
            ...(locked === undefined ? {} : { locked })
        }
    }
    const cases: [string, JsonValue | undefined, boolean | undefined][] = [
        ['locked', true, true],
        ['unlocked', false, false],
        ['absent', undefined, undefined],
        ['not a boolean', 'yes', undefined],
        ['null', null, undefined]
    ]

    it.each(cases)('keeps a local %s report through facts and a pending op', (_n, raw, want) => {
        state.setSelfId(SELF)
        state.applyEngineManagerStatus(engineStatus('llamacpp', raw))
        expectLock(status(SELF, 'llama-cpp'), want)
        state.beginLocalEngineOp('llama-cpp', 'starting')
        expectLock(status(SELF, 'llama-cpp'), want)
        state.applyEngineManagerStatus(engineStatus('llamacpp', raw))
    })

    it('replaces a local lock with an older report rather than keeping it', () => {
        state.setSelfId(SELF)
        state.applyEngineManagerStatus(engineStatus('llamacpp', true))
        state.applyEngineManagerStatus(engineStatus('llamacpp', undefined))
        expectLock(status(SELF, 'llama-cpp'), undefined)
    })

    it.each(cases)('keeps a remote %s report through facts and a pending op', (_n, raw, want) => {
        state.setSelfId(SELF)
        state.applyRemoteEngineFacts(PEER, { engines: [engineStatus('llamacpp', raw)] })
        expectLock(status(PEER, 'llama-cpp'), want)
        state.beginRemoteEngineOp(PEER, 'llama-cpp', 'starting')
        expectLock(status(PEER, 'llama-cpp'), want)
        state.clearPendingRemoteEngineOp(PEER, 'llama-cpp')
    })

    it.each(cases)('keeps a remote %s operation result', (_n, raw, want) => {
        state.setSelfId(SELF)
        state.applyRemoteEngineFacts(PEER, { engines: [engineStatus('llamacpp', !want)] })
        state.applyRemoteEngineStatusResult(PEER, 'llama-cpp', engineStatus('llamacpp', raw))
        expectLock(status(PEER, 'llama-cpp'), want)
    })

    it('makes local and remote locks unknown on invalidation and keeps other facts', () => {
        state.setSelfId(SELF)
        state.applyEngineManagerStatus(engineStatus('llamacpp', false))
        state.applyRemoteEngineFacts(PEER, { engines: [engineStatus('llamacpp', true)] })
        state.invalidateEngineLocks()
        expectLock(status(SELF, 'llama-cpp'), undefined)
        expectLock(status(PEER, 'llama-cpp'), undefined)
        expect(status(SELF, 'llama-cpp')).toMatchObject({ processStatus: 'stopped' })
        expect(status(PEER, 'llama-cpp')).toMatchObject({ processStatus: 'stopped' })
    })

    it('forgets a remote lock when a refresh no longer reports the engine', () => {
        state.setSelfId(SELF)
        state.applyRemoteEngineFacts(PEER, { engines: [engineStatus('llamacpp', true)] })
        state.applyRemoteEngineFacts(PEER, { engines: [] })
        expect(status(PEER, 'llama-cpp')).toBeUndefined()
    })
})

describe('engine lock refusal codes', () => {
    async function invokeThrowing(error: Error): Promise<unknown> {
        safeHandle('service-bridge:invoke', () => {
            throw error
        })
        const handler = ipc.handlers.get('service-bridge:invoke')
        if (!handler) throw new Error('no handler registered')
        return handler(
            { sender: { getURL: () => 'file:///index.html' } },
            { channel: 'engines:switch-status' }
        )
    }
    function preloadAnswering(
        result: IpcResult<WsInvokeResponse<'engines:switch-status'>>
    ): PreloadServiceTransport {
        return {
            connected: true,
            invoke: () => Promise.resolve(result),
            subscribePush: () => () => {},
            onConnect: () => () => {},
            onDisconnect: () => () => {},
            onAuthFailure: () => () => {},
            destroy: () => {}
        }
    }

    it.each([
        ENGINE_LOCKED_CODE,
        LOCAL_VETO_CODE,
        REMOTE_MANAGEMENT_DENIED_CODE,
        POLICY_CONFLICT_CODE
    ])('keeps code %s distinct from main process to renderer', async code => {
        await expect(
            invokeThrowing(new JsonRpcResponseError(code, 'target refusal'))
        ).resolves.toEqual({ success: false, error: `${code}: target refusal`, code })
        const failure = await serviceTransport(
            preloadAnswering({ success: false, error: 'target refusal', code })
        )
            .invoke('engines:switch-status', {})
            .catch((err: unknown) => err)
        expect(failure).toBeInstanceOf(ServiceCallError)
        expect(serviceErrorCode(failure)).toBe(code)
        expect(failure).toHaveProperty('message', 'target refusal')
    })

    it('defines the lock code as -32013', () => {
        expect([
            ENGINE_LOCKED_CODE,
            LOCAL_VETO_CODE,
            REMOTE_MANAGEMENT_DENIED_CODE,
            POLICY_CONFLICT_CODE
        ]).toEqual([-32013, -32011, -32012, -32010])
    })
})
