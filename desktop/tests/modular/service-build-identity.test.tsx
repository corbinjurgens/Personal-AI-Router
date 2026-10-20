// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { JsonValue } from '@/shared/types/json'
import type { ServiceStatus, ServiceVersions } from '@/shared/types/ipc-channels'
import { parseServiceBuildId } from '@/shared/utils/service-build'
import { brokerNodesValue, sameNode, toNodeItem } from '@/electron/service-bridge/node-projection'
import { JsonRpcSocketClient } from '@/electron/service-bridge/json-rpc-client'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import VersionsCard from '@/ui/components/ServiceSettings/VersionsCard'
import NodeCardDetails from '@/ui/components/NodeList/NodeCardDetails'
import type { NodeItem } from '@/shared/types/nodes'

const pushes = vi.hoisted(() => ({ emit: vi.fn<(channel: string) => void>() }))
vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => process.cwd() },
    BrowserWindow: { getAllWindows: () => [] }
}))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn<() => void>() }))
vi.mock('@/electron/service-bridge/broadcaster', () => ({ emitBridgePush: pushes.emit }))
vi.mock('@/electron/config/ui-config', () => ({ isFirstRun: () => false }))
vi.mock('@/ui/api/bootstrap', () => ({ isElectron: true, getPairTransport: () => null }))
vi.mock('@/ui/components/NodeList/NodeEnginesInline', () => ({ default: () => null }))
vi.mock('@/ui/components/NodeList/NodeEngineSettings', () => ({ default: () => null }))
vi.mock('@/ui/components/NodeList/NodeDeclaredRoles', () => ({ default: () => null }))
vi.mock('@/ui/components/NodeList/NodePerformance', () => ({ default: () => null }))
vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn<() => void>(),
        warn: vi.fn<() => void>(),
        error: vi.fn<() => void>(),
        verbose: vi.fn<() => void>()
    })
}))

const SHA1 = 'git:0123456789abcdef0123456789abcdef01234567'
const SHA256 = `git:${'abcdef01'.repeat(8)}`
const DIRTY = `${SHA1}+dirty`

function deferred<T>() {
    let resolve: (value: T) => void = () => {
        throw new Error('Promise not initialized')
    }
    const promise = new Promise<T>(done => {
        resolve = done
    })
    return { promise, resolve }
}

describe('service build grammar', () => {
    it.each([SHA1, DIRTY, SHA256, `${SHA256}+dirty`])('accepts %s', value => {
        expect(parseServiceBuildId(value)).toBe(value)
    })
    const invalid: (JsonValue | undefined)[] = [
        undefined,
        null,
        false,
        42,
        {},
        [],
        '',
        'v1.2.3',
        SHA1.toUpperCase(),
        `${SHA1}\n`,
        `${SHA1}\r\n`,
        ` ${SHA1}`,
        `${SHA1} `,
        `${SHA1}+clean`,
        `git:${'a'.repeat(39)}`,
        `git:${'a'.repeat(41)}`,
        `git:${'a'.repeat(63)}`,
        `git:${'a'.repeat(65)}`,
        `git:${'a'.repeat(1000)}`,
        `git:${'ａ'.repeat(40)}`
    ]
    it.each(invalid)('drops invalid metadata %j', value => {
        expect(parseServiceBuildId(value)).toBeUndefined()
    })
})

function baseline(serviceBuildId?: JsonValue) {
    return {
        hostUuid: 'build-peer',
        name: 'peer',
        ipAddress: '192.0.2.9',
        port: 14318,
        lastSeen: 100,
        models: ['model-a'],
        modelsByEngine: { ollama: ['model-a'] },
        policy: { revision: '7', roles: { medium: { engine: 'ollama', model: 'model-a' } } },
        trusted: true,
        clustered: true,
        serviceBuildId
    }
}

function publish(serviceBuildId?: JsonValue): void {
    const node = baseline(serviceBuildId)
    // Missing metadata must be a valid JSON object too.
    const { serviceBuildId: build, ...fields } = node
    getModularBridgeState().handleNotification({
        source: 'broker',
        method: 'discovery:nodes-changed',
        params: {
            nodes: [{ ...fields, ...(build === undefined ? {} : { serviceBuildId: build }) }]
        }
    })
}

describe('node build projection', () => {
    beforeEach(() => {
        getModularBridgeState().handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: { nodes: [] }
        })
        pushes.emit.mockClear()
    })

    it('drops only malformed metadata, preserving identity, hardware, models and policy facts', () => {
        publish(SHA1)
        const state = getModularBridgeState()
        state.mergeNodeInfoResponse('build-peer', {
            GPUs: [{ name: 'GPU A', vram_bytes: 1024 }],
            cpu: { name: 'CPU A', cores: 4 },
            memory: { total_bytes: 4096 }
        })
        const before = state.getNodesInitial().nodes['build-peer']
        publish({ malformed: true })
        const after = state.getNodesInitial().nodes['build-peer']
        expect(after).toEqual({ ...before, serviceBuildId: undefined })
        expect(after.name).toBe('peer')
        expect(after.topology.gpus[0]?.name).toBe('GPU A')
        expect(state.getAvailableNodes()[0]).toMatchObject({ trusted: true, clustered: true })
        expect(state.getAvailableNodes()[0].policy).toEqual({
            revision: '7',
            roles: { medium: { engine: 'ollama', model: 'model-a' } }
        })
        const parsed = brokerNodesValue({ nodes: [{ ...baseline(), serviceBuildId: [] }] })[0]
        expect(parsed.models).toEqual(['model-a'])
        expect(parsed.modelsByEngine).toEqual({ ollama: ['model-a'] })
    })

    it('compares and pushes a build-only change', () => {
        const left = brokerNodesValue({ nodes: [{ ...baseline(SHA1), serviceBuildId: SHA1 }] })[0]
        const right = { ...left, serviceBuildId: DIRTY }
        expect(sameNode(left, { ...left })).toBe(true)
        expect(sameNode(left, right)).toBe(false)
        expect(toNodeItem(right, null).serviceBuildId).toBe(DIRTY)
        publish(SHA1)
        pushes.emit.mockClear()
        publish(DIRTY)
        expect(pushes.emit).toHaveBeenCalledWith(
            'nodes:upsert',
            expect.objectContaining({ serviceBuildId: DIRTY })
        )
    })

    it('preserves across proxy and performance updates, then clears on a missing broker baseline', () => {
        publish(SHA1)
        const state = getModularBridgeState()
        state.handleNotification({
            source: 'ollama-proxy',
            method: 'node/updated',
            params: { id: 'build-peer', host: 'peer', ip: '192.0.2.9', port: 11434 }
        })
        expect(state.getNodesInitial().nodes['build-peer'].serviceBuildId).toBe(SHA1)
        state.handleNotification({
            source: 'broker',
            method: 'performance:changed',
            params: {
                local: {
                    measuringHost: 'local-build-peer',
                    status: 'unknown',
                    lastSeenAgeMs: 0,
                    expiredEntries: 0
                },
                peers: [
                    {
                        measuringHost: 'build-peer',
                        status: 'unknown',
                        lastSeenAgeMs: 10,
                        expiredEntries: 0
                    }
                ],
                expiredPeers: 0,
                persistence: {
                    degraded: false,
                    publicationBlocked: false,
                    stats: {
                        Keys: 0,
                        Samples: 0,
                        Dirty: false,
                        PublishedRevision: 0,
                        HistoryReset: false,
                        EvictedKeys: 0,
                        EvictedSamples: 0,
                        ExpiredSamples: 0,
                        Duplicates: 0
                    }
                }
            }
        })
        expect(state.getNodesInitial().nodes['build-peer'].performance?.status).toBe('unknown')
        expect(state.getNodesInitial().nodes['build-peer'].serviceBuildId).toBe(SHA1)
        publish()
        expect(state.getNodesInitial().nodes['build-peer'].serviceBuildId).toBeUndefined()
    })
})

describe('reported node build row', () => {
    const node: NodeItem = {
        id: 'display-peer',
        name: 'Display peer',
        status: 'offline',
        ipAddress: '192.0.2.10',
        port: 14318,
        allIpAddresses: ['192.0.2.10'],
        os: 'Windows',
        topology: { cpu: { model: 'CPU', cores: 4, threads: 4 }, gpus: [], ram: 4096, storage: [] }
    }
    it('labels an offline reported value as cached', () => {
        const markup = renderToStaticMarkup(
            <NodeCardDetails node={{ ...node, serviceBuildId: DIRTY }} />
        )
        expect(markup).toContain('Reported service build: ')
        expect(markup).toContain(DIRTY)
        expect(markup).toContain('(cached; node offline)')
    })
    it('shows unknown without inventing a build', () => {
        const markup = renderToStaticMarkup(<NodeCardDetails node={node} />)
        expect(markup).toContain('Reported service build: --')
    })
})

class FakeClient extends JsonRpcSocketClient {
    override call = vi.fn<JsonRpcSocketClient['call']>()
    constructor() {
        const refuse = (): Promise<never> => Promise.reject(new Error('No real connection allowed'))
        super('broker', refuse, undefined, refuse)
    }
}

describe('attached service build read', () => {
    const supervisor = getModularSupervisor()
    let clients: Map<string, JsonRpcSocketClient>
    let client: FakeClient
    beforeEach(() => {
        client = new FakeClient()
        clients = new Map([['broker', client]])
        Reflect.set(supervisor, 'processes', clients)
        vi.spyOn(supervisor, 'ready', 'get').mockReturnValue(true)
    })
    afterEach(() => {
        clients.clear()
        vi.restoreAllMocks()
    })

    it('reads running status independently of bundled semantic versions and releases listeners', async () => {
        client.call.mockResolvedValue({ serviceBuildId: DIRTY, version: '1.2.3' })
        expect(await supervisor.getServiceBuildId()).toBe(DIRTY)
        expect(client.call).toHaveBeenCalledWith('service/status')
        expect(client.listenerCount('disconnected')).toBe(0)
        expect(client.listenerCount('exit')).toBe(0)
    })

    it('does not start or query an unattached service', async () => {
        clients.clear()
        expect(await supervisor.getServiceBuildId()).toBeUndefined()
        expect(client.call).not.toHaveBeenCalled()
    })

    const unknownStatuses: JsonValue[] = [{}, { serviceBuildId: 1 }, { serviceBuildId: 'v1.2.3' }]
    it.each(unknownStatuses)('keeps unknown status unknown: %j', async result => {
        client.call.mockResolvedValue(result)
        expect(await supervisor.getServiceBuildId()).toBeUndefined()
    })

    it('keeps failed status unknown and removes request listeners', async () => {
        client.call.mockRejectedValue(new Error('Connection ended'))
        expect(await supervisor.getServiceBuildId()).toBeUndefined()
        expect(client.listenerCount('disconnected')).toBe(0)
        expect(client.listenerCount('broker-restarted')).toBe(0)
        expect(client.listenerCount('exit')).toBe(0)
    })

    it('drops a late reply from a replaced client', async () => {
        const reply = deferred<JsonValue>()
        client.call.mockReturnValue(reply.promise)
        const reading = supervisor.getServiceBuildId()
        clients.set('broker', new FakeClient())
        reply.resolve({ serviceBuildId: SHA1 })
        expect(await reading).toBeUndefined()
    })

    it('drops a pre-disconnect reply even when the same client is ready again', async () => {
        const reply = deferred<JsonValue>()
        client.call.mockReturnValue(reply.promise)
        const reading = supervisor.getServiceBuildId()
        client.emit('disconnected')
        reply.resolve({ serviceBuildId: SHA1 })
        expect(await reading).toBeUndefined()
        expect(client.listenerCount('disconnected')).toBe(0)
        client.call.mockResolvedValue({ serviceBuildId: DIRTY })
        expect(await supervisor.getServiceBuildId()).toBe(DIRTY)
    })
})

describe('About running service build', () => {
    const getVersions = vi.fn<Window['windowApi']['service']['getVersions']>()
    const unsubscribe = vi.fn<() => void>()
    let notify: (status: ServiceStatus) => void = () => {}
    const onStatusChanged = vi.fn<Window['windowApi']['service']['onStatusChanged']>(listener => {
        notify = listener
        return unsubscribe
    })
    const bundled: ServiceVersions = {
        appVersion: '9.8.7',
        modularServices: '1.2.3',
        binaries: [{ name: 'nvpair-service', version: '4.5.6' }],
        licenseType: ''
    }
    let container: HTMLDivElement
    let root: Root
    let mounted: boolean
    beforeEach(() => {
        vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
        getVersions.mockReset().mockResolvedValue({ ...bundled, serviceBuildId: SHA1 })
        unsubscribe.mockClear()
        onStatusChanged.mockClear()
        Object.defineProperty(window, 'windowApi', {
            configurable: true,
            value: { service: { getVersions, onStatusChanged } }
        })
        container = document.createElement('div')
        document.body.append(container)
        root = createRoot(container)
        mounted = true
    })
    afterEach(async () => {
        if (mounted) await act(async () => root.unmount())
        container.remove()
        vi.unstubAllGlobals()
    })
    async function render(): Promise<void> {
        await act(async () => root.render(<VersionsCard />))
    }
    async function status(connectorStatus: ServiceStatus['connectorStatus']): Promise<void> {
        await act(async () => notify({ connectorStatus, weSpawned: false }))
    }
    function runningRow(): string {
        const label = Array.from(container.querySelectorAll('*')).find(
            element => element.textContent === 'Running service build'
        )
        if (!label?.parentElement) throw new Error('Missing running build row')
        return label.parentElement.textContent ?? ''
    }

    it('shows running build without a v prefix beside independent bundled versions', async () => {
        await render()
        expect(runningRow()).toBe(`Running service build${SHA1}`)
        expect(container.textContent).toContain('Servicesv1.2.3')
        expect(container.textContent).toContain('nvpair-servicev4.5.6')
    })
    it('shows unknown on absent metadata and IPC errors', async () => {
        getVersions.mockResolvedValue(bundled)
        await render()
        expect(runningRow()).toBe('Running service build--')
        getVersions.mockRejectedValue(new Error('Disconnected'))
        await status('connected')
        expect(runningRow()).toBe('Running service build--')
    })
    it('clears on disconnect and fences late version replies across reconnect', async () => {
        await render()
        const old = deferred<ServiceVersions>()
        getVersions.mockReturnValueOnce(old.promise)
        await status('connected')
        await status('disconnected')
        expect(runningRow()).toBe('Running service build--')
        getVersions.mockResolvedValue({ ...bundled, serviceBuildId: DIRTY })
        await status('connected')
        expect(runningRow()).toBe(`Running service build${DIRTY}`)
        await act(async () => old.resolve({ ...bundled, serviceBuildId: SHA1 }))
        expect(runningRow()).toBe(`Running service build${DIRTY}`)
    })
    it('unsubscribes and ignores replies after unmount', async () => {
        const reply = deferred<ServiceVersions>()
        getVersions.mockReturnValue(reply.promise)
        await render()
        await act(async () => root.unmount())
        mounted = false
        expect(unsubscribe).toHaveBeenCalledTimes(1)
        await act(async () => reply.resolve({ ...bundled, serviceBuildId: SHA1 }))
        expect(container.textContent).toBe('')
    })
})
