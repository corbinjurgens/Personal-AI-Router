// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { duplexPair } from 'node:stream'
import { createInterface } from 'node:readline'
import { expect, it, vi } from 'vitest'
import type { JsonObject, JsonValue } from '@/electron/service-bridge/json-rpc-client'
import { createPairTransport } from '@/preload/api/pair-transport.api'
import type {
    PreloadServiceTransport,
    ServiceBridgeInvokeRequest
} from '@/shared/types/service-bridge'
import type { WsInvokeChannel } from '@/shared/types/ws-channels'
import { useWorkloadsStore } from '@/ui/stores/workloads.store'
import { bindOverviewWindow } from '../../../experiments/wails-overview/electron/window-session'
import { liveOverviewInput } from '../../../experiments/wails-overview/ui/host-input'
import {
    connectOverview,
    type OverviewStatus
} from '../../../experiments/wails-overview/ui/overview-session'

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] }, ipcRenderer: {} }))

const refused = {
    'nodes:remove-member': true,
    'cluster:get-join-requests': true,
    'cluster:ask-to-join': true,
    'cluster:respond-to-join-request': true,
    'cluster:cancel-join-request': true,
    'cluster:invite-node': true,
    'cluster:invite-status': true,
    'cluster:respond-to-invite': true,
    'cluster:set-remote-management': true,
    'cluster:set-node-name': true,
    'cluster:set-peer-name': true,
    'cluster:set-peer-nickname': true,
    'cluster:cancel-invite': true,
    'cluster:abandon-if-solo': true,
    'engines:inspect-model': true,
    'engines:diagnostic-preview': true,
    'engines:diagnostic-start': true,
    'engines:diagnostic-status': true,
    'engines:diagnostic-cancel': true,
    'engines:set-stated-memory': true,
    'engines:clear-stated-memory': true,
    'engines:preview-settings': true,
    'engines:apply-settings': true,
    'engines:reset-settings-record': true,
    'engine:command': true,
    'engine:copy-model': true,
    'engine:cancel-model-copy': true,
    'engine:active-model-copies': true,
    'engine:model-copy-capabilities': true,
    'engine:search-hub': true,
    'engine:catalog-files': true,
    'engine:source-models': true,
    'engine:link-model': true,
    'engine:unlink-model': true,
    'engine:import-model': true,
    'errors:clear': true,
    'workloads:cancel': true,
    'routing:cancel': true,
    'routing:active': true,
    'routing:subscribe': true,
    'routing:unsubscribe': true,
    'routing:get-initial': true,
    'routing-history:pull': true,
    'routing-history:consumed': true,
    'policy:set': true,
    'node:set-availability': true
} satisfies Record<
    Exclude<
        WsInvokeChannel,
        | 'app:get-initial'
        | 'cluster:get-initial'
        | 'nodes:get-initial'
        | 'engines:get-initial'
        | 'workloads:get-initial'
        | 'discovery:get-nodes'
        | 'errors:get-initial'
        | 'cluster:get-remote-management'
        | 'engine:remote-management-get'
        | 'engines:get-settings'
        | 'policy:get'
    >,
    true
>

function recordingService() {
    const [client, peer] = duplexPair()
    const calls: string[] = []
    const unexpected: string[] = []
    let failWorkloads = false
    let closed = false
    let self = 'fixture-first'
    const responses = new Map<string, JsonValue>([
        ['nodes:get-initial', { nodes: [] }],
        ['settings/get-cluster-friendly-name', { value: '' }],
        ['discovery:subscribe', { subscribed: true }],
        ['discovery:get-nodes', { nodes: [] }],
        ['llamacpp-proxy:subscribe', { subscribed: true }],
        ['llamacpp-proxy:nodes/list', { nodes: [] }],
        ['llamacpp-proxy:get-status', { ready: true, port: 8080 }],
        ['engine:subscribe', { subscribed: true }],
        ['engine:models', { models: [] }],
        ['engine:descriptors', { engines: [] }],
        ['engine:get-installed', { engines: [] }],
        ['errors:get-initial', []],
        ['workloads:subscribe', { subscribed: true }],
        ['cluster:get-remote-management', { version: 1, revision: '', grants: [] }],
        ['policy:get', {}],
        ['performance:subscribe', { subscribed: true }],
        ['performance:get', {}]
    ])
    const workloads = {
        workloads: [
            {
                id: 'retained',
                originatedFrom: self,
                engine: 'llamacpp',
                model: 'fixture-model',
                state: 'completed'
            }
        ]
    }
    function send(message: JsonObject) {
        peer.write(`${JSON.stringify({ jsonrpc: '2.0', ...message })}\n`)
    }
    const lines = createInterface({ input: peer })
    lines.on('line', line => {
        const request: JsonObject = JSON.parse(line)
        if (typeof request.method !== 'string' || typeof request.id !== 'number') return
        const method = request.method
        calls.push(method)
        if (method === 'cluster:get-node-id') {
            send({ id: request.id, result: { nodeUuid: self, name: self } })
        } else if (method === 'workloads:get-initial') {
            if (failWorkloads)
                send({ id: request.id, error: { code: -32000, message: 'Fixture read failed' } })
            else send({ id: request.id, result: workloads })
        } else {
            const result = responses.get(method)
            if (result === undefined) {
                unexpected.push(method)
                send({
                    id: request.id,
                    error: { code: -32601, message: 'Fixture refused request' }
                })
            } else send({ id: request.id, result })
        }
    })
    client.on('close', () => {
        closed = true
    })
    return {
        dial: async () => client,
        ready: () => send({ method: 'app:ready' }),
        restart() {
            failWorkloads = true
            self = 'fixture-second'
            send({ method: 'service/broker-restarted' })
            send({ method: 'app:ready' })
        },
        calls,
        unexpected,
        isClosed: () => closed,
        stop() {
            lines.close()
            client.destroy()
            peer.destroy()
        }
    }
}

function fakeWindow() {
    type WindowIpc = Parameters<typeof bindOverviewWindow>[0]
    type Invoke = Parameters<WindowIpc['register']>[0]
    let invoke: Invoke | undefined
    const generations: (number | null)[] = []
    let onGeneration: ((transport: PreloadServiceTransport | null) => void) | undefined
    const register = vi.fn<WindowIpc['register']>(dispatch => {
        invoke = dispatch
        return () => {
            invoke = undefined
        }
    })
    async function request(
        channel: ServiceBridgeInvokeRequest['channel'],
        generation: number | null,
        payload?: ServiceBridgeInvokeRequest['payload']
    ) {
        if (!invoke) throw new Error('Fixture window has no IPC handler')
        const envelope = { channel, payload, generation }
        return invoke(envelope)
    }
    const stopRenderer = connectOverview(
        liveOverviewInput({
            platform: 'MacOS',
            onGeneration(listener) {
                onGeneration = listener
                return () => {
                    onGeneration = undefined
                }
            }
        }),
        status => {
            statuses.push(status)
        }
    )
    function generation(id: number | null) {
        generations.push(id)
        onGeneration?.(
            id === null
                ? null
                : createPairTransport({
                      async invoke(_channel: string, envelope: ServiceBridgeInvokeRequest) {
                          try {
                              return {
                                  success: true,
                                  data: await request(envelope.channel, id, envelope.payload)
                              }
                          } catch (error) {
                              return {
                                  success: false,
                                  error: error instanceof Error ? error.message : String(error)
                              }
                          }
                      },
                      on: vi.fn(),
                      removeListener: vi.fn()
                  })
        )
    }
    return { register, generation, generations, request, stopRenderer }
}

const statuses: OverviewStatus[] = []

it('refuses every non-allowlisted request, retires IPC on close, and resnapshots after restart and reopen', async () => {
    const service = recordingService()
    const window = fakeWindow()
    const close = bindOverviewWindow(window, service.dial)
    let stopSecondWindow: (() => void) | undefined
    let secondService: ReturnType<typeof recordingService> | undefined
    try {
        service.ready()
        await vi.waitFor(() => expect(statuses).toEqual(['connecting', 'ready']))
        expect([...useWorkloadsStore.getState().workloads.values()]).toMatchObject([
            { id: 'retained', originatedFrom: 'fixture-first', state: 'completed' }
        ])
        const first = window.generations[0]
        expect(first).toEqual(expect.any(Number))
        const calls = [...service.calls]
        for (const channel of getRefusedChannels(refused)) {
            await expect(window.request(channel, first)).rejects.toThrow(
                'outside the read allowlist'
            )
        }
        expect(service.calls).toEqual(calls)
        expect(service.unexpected).toEqual([])

        service.restart()
        await vi.waitFor(() =>
            expect(statuses).toEqual([
                'connecting',
                'ready',
                'unavailable',
                'connecting',
                'unavailable'
            ])
        )
        const second = window.generations.at(-1)
        expect(second).toEqual(expect.any(Number))
        expect(second).not.toBe(first)
        expect(useWorkloadsStore.getState().workloads.size).toBe(0)
        expect(service.calls.filter(method => method === 'workloads:get-initial')).toHaveLength(2)
        await expect(window.request('workloads:get-initial', first)).rejects.toThrow('retired')
        expect(service.unexpected).toEqual([])
        close()
        window.stopRenderer()
        await vi.waitFor(() => expect(service.isClosed()).toBe(true))
        await expect(window.request('app:get-initial', second ?? null)).rejects.toThrow(
            'no IPC handler'
        )

        secondService = recordingService()
        const reopened = fakeWindow()
        const closeReopened = bindOverviewWindow(reopened, secondService.dial)
        stopSecondWindow = () => {
            closeReopened()
            reopened.stopRenderer()
        }
        secondService.ready()
        await vi.waitFor(() => expect(statuses.at(-1)).toBe('ready'))
        const newGeneration = reopened.generations[0]
        expect(newGeneration).toEqual(expect.any(Number))
        expect(newGeneration).not.toBe(second)
        expect(secondService.calls).toContain('workloads:get-initial')
        expect(secondService.unexpected).toEqual([])
    } finally {
        close()
        window.stopRenderer()
        stopSecondWindow?.()
        service.stop()
        secondService?.stop()
        statuses.length = 0
    }
})

function getRefusedChannels<C extends WsInvokeChannel>(channels: Record<C, true>): C[] {
    const keys: C[] = []
    for (const channel in channels) {
        if (Object.hasOwn(channels, channel)) keys.push(channel)
    }
    return keys
}
