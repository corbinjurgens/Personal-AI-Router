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
import { LOCAL_VETO_CODE, localVetoMessage } from '@/shared/constants/service-errors'
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

    function preloadAnswering(
        result: IpcResult<WsInvokeResponse<'policy:set'>>
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

describe('remote engine actions refused by the local veto', () => {
    function refuse(code: number) {
        const supervisor = getModularSupervisor()
        vi.spyOn(supervisor, 'callProcess').mockRejectedValue(
            new JsonRpcResponseError(code, VETO_DETAIL)
        )
        return {
            supervisor,
            report: vi.spyOn(supervisor, 'reportError').mockReturnValue(undefined)
        }
    }

    it('explains a vetoed remote start', async () => {
        const { supervisor, report } = refuse(LOCAL_VETO_CODE)
        await supervisor.toggleEngineRemote('peer-node', 'ollama', 'ollama', false)
        expect(report).toHaveBeenCalledWith(
            localVetoMessage('start ollama on peer-node'),
            'error',
            'engine-remote-toggle:peer-node:ollama'
        )
    })

    it('explains a vetoed remote model load', async () => {
        const { supervisor, report } = refuse(LOCAL_VETO_CODE)
        await supervisor.modelActionRemote('peer-node', 'ollama', 'ollama', 'qwen3:4b', 'loadModel')
        expect(report.mock.calls[0][0]).toBe(localVetoMessage('load qwen3:4b on peer-node'))
    })

    it('explains a vetoed remote install with start and offers no retry', async () => {
        const { supervisor, report } = refuse(LOCAL_VETO_CODE)
        await supervisor.installEngineRemote('peer-node', 'ollama', 'ollama', true)
        expect(report.mock.calls[0][0]).toBe(
            localVetoMessage('install and start ollama on peer-node')
        )
        expect(report.mock.calls[0][3]).not.toHaveProperty('action')
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
