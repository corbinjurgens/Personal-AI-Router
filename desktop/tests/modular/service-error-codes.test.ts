// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from 'vitest'

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
import { JsonRpcResponseError } from '@/electron/service-bridge/json-rpc-client'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import {
    ENGINE_LOCKED_CODE,
    LOCAL_VETO_CODE,
    REMOTE_SETTING_LOCKED_CODE,
    engineLockedMessage,
    localVetoMessage
} from '@/shared/constants/service-errors'
import type { IpcResult } from '@/shared/types/ipc-channels'
import type { PreloadServiceTransport } from '@/shared/types/service-bridge'
import type { WsInvokeResponse } from '@/shared/types/ws-channels'
import { ServiceCallError, serviceErrorCode, serviceTransport } from '@/ui/api/service-transport'

const VETO_DETAIL = 'remote /v1/engines/start: HTTP 409: this node was paused on the node itself'

afterEach(() => {
    vi.restoreAllMocks()
})

describe('service error codes across the IPC boundary', () => {
    async function invokeThrowing(error: Error): Promise<unknown> {
        safeHandle('service-bridge:invoke', () => {
            throw error
        })
        const handler = ipc.handlers.get('service-bridge:invoke')
        if (!handler) throw new Error('no handler registered')
        return handler(
            { sender: { getURL: () => 'file:///index.html' } },
            { channel: 'policy:set' }
        )
    }

    it('keeps a service refusal’s JSON-RPC code beside its message', async () => {
        await expect(
            invokeThrowing(new JsonRpcResponseError(LOCAL_VETO_CODE, 'paused on the node itself'))
        ).resolves.toEqual({
            success: false,
            error: '-32011: paused on the node itself',
            code: LOCAL_VETO_CODE
        })
    })

    it('carries no code for a failure that is not a service refusal', async () => {
        await expect(invokeThrowing(new Error('-32011 in plain text'))).resolves.toEqual({
            success: false,
            error: '-32011 in plain text'
        })
    })

    it('preserves the inference-settings lock code through the typed IPC and renderer transport', async () => {
        await expect(
            invokeThrowing(new JsonRpcResponseError(REMOTE_SETTING_LOCKED_CODE, 'locked profile'))
        ).resolves.toEqual({
            success: false,
            error: '-32014: locked profile',
            code: REMOTE_SETTING_LOCKED_CODE
        })
        const transport = serviceTransport(
            preloadAnswering({
                success: false,
                error: 'locked profile',
                code: REMOTE_SETTING_LOCKED_CODE
            })
        )
        await expect(
            transport.invoke('policy:set', {
                nodeId: 'peer',
                policy: '{}',
                expectedPersistedRevision: '5'
            })
        ).rejects.toMatchObject({
            name: 'Error',
            message: 'locked profile',
            code: REMOTE_SETTING_LOCKED_CODE
        })
        expect(
            serviceErrorCode(new ServiceCallError('locked profile', REMOTE_SETTING_LOCKED_CODE))
        ).toBe(REMOTE_SETTING_LOCKED_CODE)
        expect(serviceErrorCode(new Error('-32014: locked profile'))).toBeUndefined()
    })

    function preloadAnswering(
        result: IpcResult<WsInvokeResponse<'policy:set'>>
    ): PreloadServiceTransport {
        return {
            connected: true,
            invoke: () => Promise.resolve(result),
            invokeHistory: vi.fn<PreloadServiceTransport['invokeHistory']>(() => {
                throw new Error('Unexpected history invoke in ordinary transport fixture')
            }),
            subscribePush: () => () => {},
            onConnect: () => () => {},
            onDisconnect: () => () => {},
            onAuthFailure: () => () => {},
            destroy: () => {}
        }
    }

    it('rejects a failed renderer invoke with the service code', async () => {
        const transport = serviceTransport(
            preloadAnswering({ success: false, error: 'vetoed', code: LOCAL_VETO_CODE })
        )
        const failure = await transport
            .invoke('policy:set', { policy: '{}', expectedPersistedRevision: '1' })
            .catch((err: unknown) => err)
        expect(failure).toBeInstanceOf(ServiceCallError)
        expect(serviceErrorCode(failure)).toBe(LOCAL_VETO_CODE)
        expect(failure).toHaveProperty('message', 'vetoed')
    })

    it('leaves the code undefined when the failure has none', async () => {
        const transport = serviceTransport(preloadAnswering({ success: false, error: 'offline' }))
        const failure = await transport
            .invoke('policy:set', { policy: '{}', expectedPersistedRevision: '1' })
            .catch((err: unknown) => err)
        expect(serviceErrorCode(failure)).toBeUndefined()
        expect(serviceErrorCode(new Error('-32011: vetoed'))).toBeUndefined()
    })

    it('resolves a successful renderer invoke with its data', async () => {
        const snapshot = { policy: '{}', revision: '2', activation: null }
        const transport = serviceTransport(preloadAnswering({ success: true, data: snapshot }))
        await expect(
            transport.invoke('policy:set', { policy: '{}', expectedPersistedRevision: '1' })
        ).resolves.toEqual(snapshot)
    })
})

describe('remote engine actions refused by the target', () => {
    function refuse(code: number, detail = VETO_DETAIL) {
        const supervisor = getModularSupervisor()
        vi.spyOn(supervisor, 'callProcess').mockRejectedValue(
            new JsonRpcResponseError(code, detail)
        )
        vi.spyOn(supervisor, 'refreshRemoteEngineStatus').mockResolvedValue(undefined)
        return {
            supervisor,
            report: vi.spyOn(supervisor, 'reportError').mockReturnValue(undefined)
        }
    }
    /** The report a typed refusal produces: the explanation, then the target's text. */
    function refusal(explanation: string, code: number, detail = VETO_DETAIL): string {
        return `${explanation} Target: ${code}: ${detail}`
    }

    it('explains a vetoed remote start', async () => {
        const { supervisor, report } = refuse(LOCAL_VETO_CODE)
        await supervisor.toggleEngineRemote('peer-node', 'ollama', 'ollama', false)
        expect(report).toHaveBeenCalledWith(
            refusal(localVetoMessage('start ollama on peer-node'), LOCAL_VETO_CODE),
            'error',
            'engine-remote-toggle:peer-node:ollama'
        )
    })

    it('explains a vetoed remote model load', async () => {
        const { supervisor, report } = refuse(LOCAL_VETO_CODE)
        await supervisor.modelActionRemote('peer-node', 'ollama', 'ollama', 'qwen3:4b', 'loadModel')
        expect(report.mock.calls[0][0]).toBe(
            refusal(localVetoMessage('load qwen3:4b on peer-node'), LOCAL_VETO_CODE)
        )
    })

    it('explains a vetoed remote install with start and offers no retry', async () => {
        const { supervisor, report } = refuse(LOCAL_VETO_CODE)
        await supervisor.installEngineRemote('peer-node', 'ollama', 'ollama', true)
        expect(report.mock.calls[0][0]).toBe(
            refusal(localVetoMessage('install and start ollama on peer-node'), LOCAL_VETO_CODE)
        )
        expect(report.mock.calls[0][3]).not.toHaveProperty('action')
    })

    it.each([
        [LOCAL_VETO_CODE, localVetoMessage],
        [ENGINE_LOCKED_CODE, engineLockedMessage]
    ])('applies refusal %s to every remote mutation without a retry', async (code, explain) => {
        const { supervisor, report } = refuse(code)
        await supervisor.toggleEngineRemote('peer-node', 'ollama', 'ollama', true)
        await supervisor.modelActionRemote('peer-node', 'ollama', 'ollama', 'm', 'unloadModel')
        await supervisor.modelActionRemote('peer-node', 'ollama', 'ollama', 'm', 'deleteModel')
        await supervisor.pullModelRemote('peer-node', 'ollama', 'ollama', 'm')
        await supervisor.installEngineRemote('peer-node', 'ollama', 'ollama', false)
        expect(report.mock.calls.map(call => call[0])).toEqual([
            refusal(explain('stop ollama on peer-node'), code),
            refusal(explain('unload m on peer-node'), code),
            refusal(explain('delete m on peer-node'), code),
            refusal(explain('download m on peer-node'), code),
            refusal(explain('install ollama on peer-node'), code)
        ])
        expect(report.mock.calls[3][3]).not.toHaveProperty('action')
        expect(report.mock.calls[4][3]).not.toHaveProperty('action')
    })

    it('names a lock refusal for start and load', async () => {
        const detail = 'remote /v1/engines/start: HTTP 409: This engine is locked on this node.'
        const { supervisor, report } = refuse(ENGINE_LOCKED_CODE, detail)
        await supervisor.toggleEngineRemote('peer-node', 'ollama', 'ollama', false)
        await supervisor.modelActionRemote('peer-node', 'ollama', 'ollama', 'm', 'loadModel')
        expect(report.mock.calls.map(call => call[0])).toEqual([
            refusal(engineLockedMessage('start ollama on peer-node'), ENGINE_LOCKED_CODE, detail),
            refusal(engineLockedMessage('load m on peer-node'), ENGINE_LOCKED_CODE, detail)
        ])
    })

    it.each([
        [
            'an older target’s saved Off',
            'remote /v1/engines/start: HTTP 409: ollama was turned off on the node itself; start it there'
        ],
        [
            'a local stop that cancelled the operation',
            'ollama was turned off on the node itself; start it there'
        ]
    ])('does not present %s as a pause', async (_name, detail) => {
        const { supervisor, report } = refuse(LOCAL_VETO_CODE, detail)
        await supervisor.toggleEngineRemote('peer-node', 'ollama', 'ollama', false)
        const message = report.mock.calls[0][0]
        expect(message).toBe(
            refusal(localVetoMessage('start ollama on peer-node'), LOCAL_VETO_CODE, detail)
        )
        expect(message).not.toMatch(/paused/i)
    })

    it('bounds the target’s text', async () => {
        const { supervisor, report } = refuse(LOCAL_VETO_CODE, 'x'.repeat(5000))
        await supervisor.toggleEngineRemote('peer-node', 'ollama', 'ollama', false)
        const explanation = localVetoMessage('start ollama on peer-node')
        expect(report.mock.calls[0][0]).toHaveLength(`${explanation} Target: `.length + 1024)
    })

    it('reports any other refusal as before, by code and not by text', async () => {
        const { supervisor, report } = refuse(-32000)
        await supervisor.toggleEngineRemote('peer-node', 'ollama', 'ollama', false)
        expect(report.mock.calls[0][0]).toBe(
            `Failed to start ollama on peer-node: -32000: ${VETO_DETAIL}`
        )
        await supervisor.installEngineRemote('peer-node', 'ollama', 'ollama', true)
        expect(report.mock.calls[1][3]).toHaveProperty('action', 'retry')
    })
})
