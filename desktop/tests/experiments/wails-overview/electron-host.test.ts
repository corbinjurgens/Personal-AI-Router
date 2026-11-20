// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { duplexPair } from 'node:stream'
import { createInterface } from 'node:readline'
import { EventEmitter } from 'node:events'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, beforeAll, expect, it, vi } from 'vitest'
import type { JsonObject, JsonValue } from '@/electron/service-bridge/json-rpc-client'
import type { IpcResult } from '@/shared/types/ipc-channels'
import type { ServiceBridgeInvokeRequest } from '@/shared/types/service-bridge'
import type { WsInvokeChannel } from '@/shared/types/ws-channels'
import { useWorkloadsStore } from '@/ui/stores/workloads.store'
import type { createProjectionRealm } from '../../../experiments/wails-overview/electron/projection-module.js'
import '../../../experiments/wails-overview/electron/preload'
import type { ProjectionPush } from '../../../experiments/wails-overview/electron/projection-push'
import { bindOverviewWindow } from '../../../experiments/wails-overview/electron/window-session'
import {
    liveOverviewInput,
    type LiveOverviewHost
} from '../../../experiments/wails-overview/ui/host-input'
import {
    connectOverview,
    type OverviewStatus
} from '../../../experiments/wails-overview/ui/overview-session'

type WindowIpc = Parameters<typeof bindOverviewWindow>[0]
type Invoke = Parameters<WindowIpc['register']>[1]
type InvokeResult = IpcResult<Awaited<ReturnType<Invoke>>>
type Envelope = ServiceBridgeInvokeRequest & { generation: number | null }

const nativeIpc = vi.hoisted(
    (): { events?: EventEmitter; handler?: { channel: string; invoke: Invoke } } => ({})
)

vi.mock('electron', () => {
    nativeIpc.events = new EventEmitter()
    return {
        ipcRenderer: Object.assign(nativeIpc.events, {
            async invoke(channel: string, envelope: Envelope): Promise<InvokeResult> {
                if (channel !== 'service-bridge:invoke')
                    throw new Error(`Unexpected IPC: ${channel}`)
                const handler = nativeIpc.handler
                if (!handler || handler.channel !== channel)
                    throw new Error('Fixture window has no IPC handler')
                try {
                    return { success: true, data: await handler.invoke(envelope) }
                } catch (error) {
                    return { success: false, error: String(error) }
                }
            }
        }),
        contextBridge: {
            exposeInMainWorld(key: string, host: LiveOverviewHost) {
                Object.defineProperty(window, key, { value: host, configurable: true })
            }
        }
    }
})

let createRealm: typeof createProjectionRealm
let bundleDir: string
beforeAll(() => {
    bundleDir = mkdtempSync(join(tmpdir(), 'pair-overview-projection-'))
    const script = join(__dirname, 'build-projection-fixture.mjs')
    execFileSync(process.execPath, [script, bundleDir], { timeout: 12_000 })
    const load = createRequire(script)
    createRealm = load(join(bundleDir, 'projection-module.js')).createProjectionRealm
}, 15_000)

afterAll(() => {
    if (bundleDir) rmSync(bundleDir, { recursive: true, force: true })
})

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
    let workloadRead: 'respond' | 'hold' | 'fail' = 'respond'
    let heldWorkloadRead: number | undefined
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
    function workload(id: string, state: string) {
        return { id, originatedFrom: self, engine: 'llamacpp', model: 'fixture-model', state }
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
            if (workloadRead === 'fail')
                send({ id: request.id, error: { code: -32000, message: 'Fixture read failed' } })
            else if (workloadRead === 'hold') heldWorkloadRead = request.id
            else
                send({ id: request.id, result: { workloads: [workload('retained', 'completed')] } })
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
        restart(nextSelf: string, read: typeof workloadRead) {
            workloadRead = read
            self = nextSelf
            send({ method: 'service/broker-restarted' })
            send({ method: 'app:ready' })
        },
        hasHeldBaseline: () => heldWorkloadRead !== undefined,
        sendReplacementBaseline() {
            if (heldWorkloadRead === undefined)
                throw new Error('Fixture baseline was not requested')
            send({
                id: heldWorkloadRead,
                result: {
                    workloads: [workload('replacement', 'completed'), workload('live', 'queued')]
                }
            })
            heldWorkloadRead = undefined
        },
        upsert(id: string) {
            send({ method: 'workloads:upsert', params: { workloadInfo: workload(id, 'running') } })
        },
        remove(id: string) {
            send({ method: 'workloads:remove', params: { workloadId: id, originatedFrom: self } })
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

function fakeWindow(statuses: OverviewStatus[]) {
    const generations: (number | null)[] = []
    const host = window.__PAIR_OVERVIEW_HOST__
    if (!host) throw new Error('Overview preload did not expose its host')
    const register = vi.fn<WindowIpc['register']>((channel, invoke) => {
        nativeIpc.handler = { channel, invoke }
        return () => {
            nativeIpc.handler = undefined
        }
    })
    async function request(
        channel: ServiceBridgeInvokeRequest['channel'],
        generation: number | null,
        payload?: ServiceBridgeInvokeRequest['payload']
    ) {
        if (!nativeIpc.handler) throw new Error('Fixture window has no IPC handler')
        const envelope = { channel, payload, generation }
        return nativeIpc.handler.invoke(envelope)
    }
    const stopRenderer = connectOverview(liveOverviewInput(host), status => statuses.push(status))
    return {
        register,
        send(...[channel, message]: Parameters<WindowIpc['send']>) {
            if (channel === 'overview-experiment:generation') generations.push(message)
            nativeIpc.events?.emit(channel, {}, message)
        },
        generations,
        request,
        stopRenderer
    }
}

const currentWorkloads = () =>
    [...useWorkloadsStore.getState().workloads.values()].sort((a, b) => a.id.localeCompare(b.id))

it('maps real preload IPC and resnapshots between ordered live upsert and remove events', async () => {
    const statuses: OverviewStatus[] = []
    const window = fakeWindow(statuses)
    const service = recordingService()
    const { close } = bindOverviewWindow(window, service.dial, createRealm)
    let stopSecondWindow: (() => void) | undefined
    let secondService: ReturnType<typeof recordingService> | undefined
    try {
        service.ready()
        await vi.waitFor(() => expect(statuses).toEqual(['unavailable', 'connecting', 'ready']))
        expect(currentWorkloads()).toMatchObject([
            { id: 'retained', originatedFrom: 'fixture-first', state: 'completed' }
        ])
        const first = window.generations[0] ?? null
        expect(first).toEqual(expect.any(Number))
        const calls = [...service.calls]
        for (const channel of getRefusedChannels(refused)) {
            await expect(window.request(channel, first)).rejects.toThrow(
                'outside the read allowlist'
            )
        }
        expect(service.calls).toEqual(calls)
        expect(service.unexpected).toEqual([])

        service.upsert('before-restart')
        await vi.waitFor(() =>
            expect(currentWorkloads()).toMatchObject([
                { id: 'before-restart', originatedFrom: 'fixture-first', state: 'running' },
                { id: 'retained' }
            ])
        )

        service.restart('fixture-second', 'hold')
        await vi.waitFor(() => expect(service.hasHeldBaseline()).toBe(true))
        expect(currentWorkloads()).toEqual([])
        service.upsert('live')
        await vi.waitFor(() =>
            expect(currentWorkloads()).toMatchObject([
                { id: 'live', originatedFrom: 'fixture-second', state: 'running' }
            ])
        )
        service.sendReplacementBaseline()
        await vi.waitFor(() => expect(statuses.at(-1)).toBe('ready'))
        expect(currentWorkloads()).toMatchObject([
            { id: 'live', originatedFrom: 'fixture-second', state: 'running' },
            { id: 'replacement', originatedFrom: 'fixture-second', state: 'completed' }
        ])
        service.remove('live')
        await vi.waitFor(() =>
            expect(currentWorkloads()).toMatchObject([
                { id: 'replacement', originatedFrom: 'fixture-second', state: 'completed' }
            ])
        )
        const second = window.generations.at(-1)
        expect(second).toEqual(expect.any(Number))
        expect(second).not.toBe(first)
        await expect(window.request('workloads:get-initial', first)).rejects.toThrow('retired')

        service.restart('fixture-third', 'fail')
        await vi.waitFor(() =>
            expect(statuses).toEqual([
                'unavailable',
                'connecting',
                'ready',
                'unavailable',
                'connecting',
                'ready',
                'unavailable',
                'connecting',
                'unavailable'
            ])
        )
        const third = window.generations.at(-1)
        expect(third).toEqual(expect.any(Number))
        expect(third).not.toBe(second)
        expect(useWorkloadsStore.getState().workloads.size).toBe(0)
        expect(service.calls.filter(method => method === 'workloads:get-initial')).toHaveLength(3)
        expect(service.unexpected).toEqual([])
        close()
        window.stopRenderer()
        await vi.waitFor(() => expect(service.isClosed()).toBe(true))
        await expect(window.request('app:get-initial', third ?? null)).rejects.toThrow(
            'no IPC handler'
        )

        secondService = recordingService()
        const reopened = fakeWindow(statuses)
        const { close: closeReopened } = bindOverviewWindow(
            reopened,
            secondService.dial,
            createRealm
        )
        stopSecondWindow = () => {
            closeReopened()
            reopened.stopRenderer()
        }
        secondService.ready()
        await vi.waitFor(() => expect(statuses.at(-1)).toBe('ready'))
        const newGeneration = reopened.generations[0]
        expect(newGeneration).toEqual(expect.any(Number))
        expect(newGeneration).not.toBe(third)
        expect(secondService.calls).toContain('workloads:get-initial')
        expect(secondService.unexpected).toEqual([])
    } finally {
        close()
        window.stopRenderer()
        stopSecondWindow?.()
        service.stop()
        secondService?.stop()
    }
})

function projectionFixture(proxyStatus: JsonValue, model?: string) {
    const responses = new Map<string, JsonValue>([
        ['cluster:get-node-id', { nodeUuid: 'fixture-node' }],
        ['discovery:subscribe', { subscribed: true }],
        ['discovery:get-nodes', { nodes: [] }],
        ['llamacpp-proxy:subscribe', { subscribed: true }],
        ['llamacpp-proxy:nodes/list', { nodes: [] }],
        ['llamacpp-proxy:get-status', proxyStatus],
        ['engine:subscribe', { subscribed: true }],
        [
            'engine:models',
            model ? { models: [model], modelsByEngine: { llamacpp: [model] } } : { models: [] }
        ],
        ['engine:descriptors', { engines: [] }],
        [
            'engine:get-installed',
            { engines: [{ engine: 'llamacpp', installed: true, running: false }] }
        ],
        ['performance:subscribe', { subscribed: true }],
        ['performance:get', {}],
        ['workloads:subscribe', { subscribed: true }],
        ['workloads:get-initial', { workloads: [] }]
    ])
    return {
        async call(method: string) {
            const response = responses.get(method)
            if (response === undefined) throw new Error(`Unexpected fixture method: ${method}`)
            return response
        }
    }
}

it.each<{ label: string; status: JsonValue }>([
    { label: 'unready', status: { ready: false, port: 0 } },
    { label: 'malformed', status: { port: 54302 } },
    { label: 'ready with a malformed port', status: { ready: true, port: 54302.5 } }
])('starts fresh when the replacement proxy status is $label', async ({ status }) => {
    const push = vi.fn<ProjectionPush>()
    const first = createRealm(projectionFixture({ ready: true, port: 54301 }, 'old-model'), push)
    try {
        const initial = await first.invoke('engines:get-initial', undefined)
        expect(initial.statuses).toEqual([
            {
                engineType: 'llama-cpp',
                nodeId: 'fixture-node',
                processStatus: 'stopped',
                enginePort: null,
                proxyPort: 54301
            }
        ])
        expect(initial.models).toMatchObject([
            { engineType: 'llama-cpp', nodeId: 'fixture-node', models: [{ name: 'old-model' }] }
        ])
        first.project({
            source: 'broker',
            method: 'engine:install-progress',
            params: { engine: 'llamacpp', stage: 'installing' }
        })
        expect(push).toHaveBeenCalledWith({
            channel: 'engines:state-changed',
            payload: {
                nodeId: 'fixture-node',
                engineType: 'llama-cpp',
                status: {
                    nodeId: 'fixture-node',
                    engineType: 'llama-cpp',
                    processStatus: 'installing',
                    enginePort: null,
                    proxyPort: 54301
                }
            }
        })
    } finally {
        first.retire()
    }

    const replacement = createRealm(projectionFixture(status), () => {})
    try {
        const initial = await replacement.invoke('engines:get-initial', undefined)
        expect(initial.statuses).toEqual([
            {
                engineType: 'llama-cpp',
                nodeId: 'fixture-node',
                processStatus: 'stopped',
                enginePort: null,
                proxyPort: null
            }
        ])
        expect(initial.models).toEqual([
            { engineType: 'llama-cpp', nodeId: 'fixture-node', models: [] }
        ])
    } finally {
        replacement.retire()
    }
})

it('rejects a retired workload read that finishes after its replacement is ready', async () => {
    const held = Promise.withResolvers<JsonValue>()
    const requested = Promise.withResolvers<void>()
    const fixture = projectionFixture({ ready: false, port: 0 })
    const first = createRealm(
        {
            async call(method) {
                if (method !== 'workloads:get-initial') return fixture.call(method)
                requested.resolve()
                return held.promise
            }
        },
        () => {}
    )
    const oldRead = first.invoke('workloads:get-initial', undefined).then(
        () => 'resolved',
        error => String(error)
    )
    await requested.promise
    first.retire()

    const replacement = createRealm(fixture, () => {})
    try {
        expect(await replacement.invoke('workloads:get-initial', undefined)).toEqual({})
        held.resolve({ workloads: [{ id: 'retired', engine: 'llamacpp', state: 'running' }] })
        expect(await oldRead).toBe('Error: Overview generation retired')
        expect(await replacement.invoke('workloads:get-initial', undefined)).toEqual({})
    } finally {
        held.resolve({ workloads: [] })
        replacement.retire()
    }
})

function getRefusedChannels<C extends WsInvokeChannel>(channels: Record<C, true>): C[] {
    const keys: C[] = []
    for (const channel in channels) {
        if (Object.hasOwn(channels, channel)) keys.push(channel)
    }
    return keys
}
