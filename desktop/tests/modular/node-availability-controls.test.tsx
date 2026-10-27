// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type {
    NodeAvailabilityChange,
    NodeAvailabilityRequest,
    NodePolicyDocument
} from '@/shared/types/node-policy'
import type { AvailableNode } from '@/shared/types/cluster'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi, type INodePolicyApi } from '@/ui/api/pair-api'
import NodeAvailabilityControls, {
    AvailabilityControls
} from '@/ui/components/NodeList/NodeAvailabilityControls'
import {
    initializeNodeAvailability,
    useNodeConfigurationStore,
    watchNodeAvailability
} from '@/ui/stores/node-configuration.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'

const getPolicy = vi.fn<INodePolicyApi['get']>()
const setAvailability = vi.fn<INodePolicyApi['setAvailability']>()
const changes = new Set<(change: NodeAvailabilityChange) => void>()
const stops: Array<() => void> = []
let root: Root
let container: HTMLDivElement
let cleanup: () => void

function documentFor(
    availability: NodePolicyDocument['availability'],
    revision = '1',
    pausedBy?: 'local' | 'peer'
): NodePolicyDocument {
    return {
        policy: JSON.stringify({ revision, availability, pausedBy }),
        revision,
        activation: null,
        availability,
        pausedBy,
        recoveryRequired: false
    }
}

function advertisement(revision: string, pausedBy?: 'local' | 'peer'): AvailableNode {
    return {
        id: 'peer',
        name: 'Peer',
        ipAddress: '127.0.0.1',
        port: 1,
        lastSeen: 0,
        trusted: true,
        clustered: true,
        policy: { revision, pausedBy, roles: {} }
    }
}

function button(label: string): HTMLButtonElement {
    const result = Array.from(container.querySelectorAll('button')).find(
        item => item.getAttribute('aria-label') === label || item.textContent?.trim() === label
    )
    if (!result) throw new Error(`Missing ${label}`)
    return result
}

async function tick(ms = 2_000): Promise<void> {
    await act(async () => {
        await vi.advanceTimersByTimeAsync(ms)
    })
}

async function watch(
    id = 'peer',
    nodeId: string | undefined = id === 'self' ? undefined : 'peer'
): Promise<void> {
    await act(async () => {
        stops.push(watchNodeAvailability(id, nodeId, true))
    })
}

beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => true })
    getPolicy.mockReset().mockResolvedValue(documentFor('available'))
    setAvailability.mockReset().mockResolvedValue({ availability: 'paused' })
    changes.clear()
    const transport: ServiceTransport = {
        connected: true,
        invoke: vi.fn<ServiceTransport['invoke']>(),
        subscribePush: vi.fn<ServiceTransport['subscribePush']>(() => () => {}),
        onConnect: vi.fn<ServiceTransport['onConnect']>(() => () => {}),
        onDisconnect: vi.fn<ServiceTransport['onDisconnect']>(() => () => {}),
        onAuthFailure: vi.fn<ServiceTransport['onAuthFailure']>(() => () => {}),
        destroy: vi.fn<ServiceTransport['destroy']>()
    }
    const api = createPairApi(transport)
    api.nodePolicy.get = getPolicy
    api.nodePolicy.setAvailability = setAvailability
    api.nodePolicy.onAvailabilityChanged = callback => {
        changes.add(callback)
        return () => changes.delete(callback)
    }
    api.cluster.getRemoteManagementForNode = vi
        .fn<typeof api.cluster.getRemoteManagementForNode>()
        .mockResolvedValue({ version: 1, revision: '1', allowed: true, state: 'allowed' })
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    useConnectionStore.setState({ connected: true, selfId: 'self' })
    useDiscoveredNodesStore.setState({ nodes: [] })
    useNodeConfigurationStore.setState({
        availability: new Map(),
        management: new Map(),
        editors: new Map(),
        policyDrafts: new Map()
    })
    cleanup = initializeNodeAvailability()
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    stops.splice(0).forEach(stop => stop())
    cleanup()
    container.remove()
    vi.useRealTimers()
    vi.unstubAllGlobals()
})

describe('node availability presentation', () => {
    it('sends the exact configured target and has no mounted immediate action', async () => {
        await act(async () =>
            root.render(
                <NodeAvailabilityControls targetId="peer" nodeId="peer" name="Peer" online />
            )
        )
        await act(async () => button('Pause node: Peer').click())
        expect(setAvailability).toHaveBeenCalledExactlyOnceWith({
            nodeId: 'peer',
            state: 'paused',
            pauseMode: 'configured'
        })
        expect(container.textContent).not.toContain('Stop local models now')
        expect(container.textContent).toContain('routing elsewhere continues')
        expect(container.textContent).toContain('new client request')
    })

    it.each([
        { name: 'offline', online: false, connected: true, known: true, allowed: true },
        { name: 'disconnected', online: true, connected: false, known: true, allowed: true },
        { name: 'unknown', online: true, connected: true, known: false, allowed: true },
        { name: 'permission loading', online: true, connected: true, known: true, allowed: false }
    ])('makes no mutation for $name', async ({ online, connected, known, allowed }) => {
        const onRequest = vi.fn<(request: NodeAvailabilityRequest) => void>()
        await act(async () =>
            root.render(
                <AvailabilityControls
                    name="Peer"
                    nodeId="peer"
                    local={false}
                    enabled={online && connected && allowed}
                    entry={{
                        current: known,
                        value: known ? 'available' : undefined,
                        reading: false,
                        stale: false
                    }}
                    onRequest={onRequest}
                    onRefresh={() => {}}
                />
            )
        )
        await act(async () => button('Pause node: Peer').click())
        expect(button('Pause node: Peer').disabled).toBe(true)
        expect(onRequest).not.toHaveBeenCalled()
        expect(setAvailability).not.toHaveBeenCalled()
    })

    it('isolates immediate escalation and permits local Resume during a configured drain', async () => {
        const onRequest = vi.fn<(request: NodeAvailabilityRequest) => void>()
        const onImmediate = vi.fn<(request: NodeAvailabilityRequest) => void>()
        await act(async () =>
            root.render(
                <AvailabilityControls
                    name="This PC"
                    local
                    enabled
                    entry={{
                        current: true,
                        value: 'draining',
                        reading: false,
                        stale: false,
                        pending: 'paused'
                    }}
                    onRequest={onRequest}
                    onImmediate={onImmediate}
                    onRefresh={() => {}}
                />
            )
        )
        await act(async () => {
            button('Stop local models now').click()
            button('Resume node: This PC').click()
        })
        expect(onImmediate).toHaveBeenCalledExactlyOnceWith({
            state: 'paused',
            pauseMode: 'immediate'
        })
        expect(onRequest).toHaveBeenCalledExactlyOnceWith({ nodeId: undefined, state: 'available' })
        expect(setAvailability).not.toHaveBeenCalled()
    })

    it.each(['draining', 'paused'] satisfies NodePolicyDocument['availability'][])(
        'offers no remote Resume or repeated Pause while %s',
        async value => {
            const onRequest = vi.fn<(request: NodeAvailabilityRequest) => void>()
            await act(async () =>
                root.render(
                    <AvailabilityControls
                        name="Peer"
                        nodeId="peer"
                        local={false}
                        enabled
                        entry={{
                            current: true,
                            value,
                            pausedBy: 'peer',
                            reading: false,
                            stale: false
                        }}
                        onRequest={onRequest}
                        onRefresh={() => {}}
                        onImmediate={onRequest}
                    />
                )
            )
            expect(container.textContent).toContain('Resume on that device')
            expect(container.textContent).toContain('Paused by a managing peer')
            expect(container.querySelector('[aria-label="Resume node: Peer"]')).toBeNull()
            expect(container.textContent).not.toContain('Stop local models now')
            await act(async () => button('Pause node: Peer').click())
            expect(onRequest).not.toHaveBeenCalled()
        }
    )

    it.each([240, 600])('keeps accessible text actions and wrapping at %spx', async width => {
        container.style.width = `${width}px`
        await act(async () =>
            root.render(
                <NodeAvailabilityControls targetId="self" name="A device with a long name" online />
            )
        )
        const pause = button('Pause node: A device with a long name')
        pause.focus()
        expect(document.activeElement).toBe(pause)
        expect(pause.className).toContain('whitespace-normal')
        expect(
            container.querySelector('[aria-label="Availability for A device with a long name"]')
                ?.className
        ).toContain('min-w-0')
    })
})

describe('authoritative availability lifecycle', () => {
    it('hydrates the new self target rather than a closed former self card', async () => {
        await watch('self', undefined)
        stops.pop()?.()
        useConnectionStore.setState({ selfId: 'new-self' })
        getPolicy.mockResolvedValue(documentFor('draining', '2', 'peer'))
        await act(async () => {
            stops.push(watchNodeAvailability('new-self', undefined, true))
        })
        expect(useNodeConfigurationStore.getState().availability.get('new-self')).toMatchObject({
            value: 'draining',
            current: true
        })
        expect(useNodeConfigurationStore.getState().availability.get('self')?.current).toBe(false)
    })

    it('coalesces repeated manual Refresh within the two-second read interval', async () => {
        await watch()
        await act(async () => {
            void useNodeConfigurationStore.getState().readAvailability('peer', true)
            void useNodeConfigurationStore.getState().readAvailability('peer', true)
        })
        expect(getPolicy).toHaveBeenCalledTimes(1)
        await tick(1_999)
        expect(getPolicy).toHaveBeenCalledTimes(1)
        await tick(1)
        expect(getPolicy).toHaveBeenCalledTimes(2)
    })

    it('does not reuse an old client read or revision after the API session is replaced', async () => {
        let finish: ((document: NodePolicyDocument) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        await watch()
        stops.pop()?.()
        const freshGet = vi.fn<INodePolicyApi['get']>().mockResolvedValue(documentFor('available'))
        const api = {
            ...window.pairApi,
            nodePolicy: { ...window.pairApi.nodePolicy, get: freshGet }
        }
        Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
        await watch()
        expect(freshGet).toHaveBeenCalledExactlyOnceWith('peer')
        if (!finish) throw new Error('Missing old client read')
        await act(async () => finish?.(documentFor('paused', '99', 'peer')))
        expect(useNodeConfigurationStore.getState().availability.get('peer')).toMatchObject({
            value: 'available',
            revision: '1',
            current: true
        })
    })

    it('shares the pending action and operational error when a view closes and reopens', async () => {
        await watch('self', undefined)
        let reject: ((error: Error) => void) | undefined
        setAvailability.mockImplementationOnce(
            () =>
                new Promise((_resolve, fail) => {
                    reject = fail
                })
        )
        await act(async () => {
            void useNodeConfigurationStore.getState().requestAvailability('self', 'paused')
        })
        stops.pop()?.()
        getPolicy.mockResolvedValue(documentFor('draining', '2', 'local'))
        await watch('self', undefined)
        await tick()
        await act(async () =>
            useNodeConfigurationStore.getState().requestAvailability('self', 'paused')
        )
        expect(setAvailability).toHaveBeenCalledTimes(1)
        if (!reject) throw new Error('Missing pending pause')
        await act(async () => reject?.(new Error('Shutdown incomplete')))
        await tick()
        expect(useNodeConfigurationStore.getState().availability.get('self')).toMatchObject({
            pending: undefined,
            value: 'draining'
        })
        expect(useNodeConfigurationStore.getState().availability.get('self')?.error).toContain(
            'Shutdown incomplete'
        )
    })

    it('stops hidden polling and refreshes on visible activation without a mutation', async () => {
        getPolicy.mockResolvedValue(documentFor('draining'))
        await watch()
        Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' })
        await act(async () => document.dispatchEvent(new Event('visibilitychange')))
        await tick(10_000)
        expect(getPolicy).toHaveBeenCalledTimes(1)
        getPolicy.mockResolvedValue(documentFor('available', '2'))
        Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' })
        await act(async () => document.dispatchEvent(new Event('visibilitychange')))
        expect(getPolicy).toHaveBeenCalledTimes(2)
        expect(useNodeConfigurationStore.getState().availability.get('peer')).toMatchObject({
            value: 'available',
            current: true
        })
        expect(setAvailability).not.toHaveBeenCalled()
    })

    it.each(['visible activation', 'reconnect'])(
        'retains pending Pause ownership and its failure through %s readback',
        async refresh => {
            await act(async () =>
                root.render(<NodeAvailabilityControls targetId="self" name="This PC" online />)
            )
            let reject: ((error: Error) => void) | undefined
            setAvailability.mockImplementationOnce(
                () =>
                    new Promise((_resolve, fail) => {
                        reject = fail
                    })
            )
            await act(async () => button('Pause node: This PC').click())
            if (refresh === 'visible activation') {
                Object.defineProperty(document, 'visibilityState', {
                    configurable: true,
                    value: 'hidden'
                })
                await act(async () => document.dispatchEvent(new Event('visibilitychange')))
                await tick()
                Object.defineProperty(document, 'visibilityState', {
                    configurable: true,
                    value: 'visible'
                })
                await act(async () => document.dispatchEvent(new Event('visibilitychange')))
            } else {
                await act(async () => useConnectionStore.setState({ connected: false }))
                await tick()
                await act(async () => useConnectionStore.setState({ connected: true }))
            }
            expect(getPolicy).toHaveBeenCalledTimes(2)
            expect(useNodeConfigurationStore.getState().availability.get('self')).toMatchObject({
                current: true,
                value: 'available',
                pending: 'paused'
            })
            expect(button('Pause node: This PC').disabled).toBe(true)
            await act(async () => button('Pause node: This PC').click())
            await act(async () =>
                useNodeConfigurationStore.getState().requestAvailability('self', 'paused')
            )
            expect(setAvailability).toHaveBeenCalledTimes(1)
            if (!reject) throw new Error('Missing pending Pause')
            await act(async () => reject?.(new Error('Owned engine shutdown failed')))
            await tick()
            expect(useNodeConfigurationStore.getState().availability.get('self')).toMatchObject({
                current: true,
                value: 'available',
                pending: undefined
            })
            expect(container.querySelector('[role="alert"]')?.textContent).toContain(
                'Owned engine shutdown failed'
            )
            expect(setAvailability).toHaveBeenCalledTimes(1)
        }
    )

    it('reads current remote availability after a delayed Pause notification follows target-local Resume', async () => {
        await watch()
        getPolicy.mockResolvedValue(documentFor('available', '3'))
        await act(async () => useDiscoveredNodesStore.setState({ nodes: [advertisement('3')] }))
        await tick()
        expect(useNodeConfigurationStore.getState().availability.get('peer')).toMatchObject({
            value: 'available',
            revision: '3',
            current: true
        })
        await act(async () => {
            changes.forEach(listener =>
                listener({ nodeId: 'peer', availability: 'paused', pausedBy: 'peer' })
            )
        })
        expect(useNodeConfigurationStore.getState().availability.get('peer')).toMatchObject({
            value: 'available',
            revision: '3',
            current: false
        })
        await tick()
        expect(getPolicy).toHaveBeenCalledTimes(3)
        expect(getPolicy).toHaveBeenLastCalledWith('peer')
        expect(useNodeConfigurationStore.getState().availability.get('peer')).toMatchObject({
            value: 'available',
            revision: '3',
            current: true,
            pausedBy: undefined
        })
        await tick(10_000)
        expect(getPolicy).toHaveBeenCalledTimes(3)
        expect(setAvailability).not.toHaveBeenCalled()
    })

    it('coalesces a burst of provenance hints without treating any hint as stopped', async () => {
        await watch()
        getPolicy.mockResolvedValue(documentFor('draining', '2', 'peer'))
        await act(async () => {
            useDiscoveredNodesStore.setState({ nodes: [advertisement('2', 'local')] })
            useDiscoveredNodesStore.setState({ nodes: [advertisement('2', 'peer')] })
            useDiscoveredNodesStore.setState({ nodes: [advertisement('2', 'local')] })
        })
        expect(getPolicy).toHaveBeenCalledTimes(1)
        expect(useNodeConfigurationStore.getState().availability.get('peer')?.current).toBe(false)
        await tick(1_999)
        expect(getPolicy).toHaveBeenCalledTimes(1)
        await tick(1)
        expect(getPolicy).toHaveBeenCalledTimes(2)
        expect(useNodeConfigurationStore.getState().availability.get('peer')?.value).toBe(
            'draining'
        )
    })

    it('coalesces readers and observes another client Pause and target-local Resume without a local mutation or push', async () => {
        await watch()
        await watch()
        expect(getPolicy).toHaveBeenCalledTimes(1)
        getPolicy.mockResolvedValue(documentFor('draining', '2', 'peer'))
        await act(async () =>
            useDiscoveredNodesStore.setState({ nodes: [advertisement('2', 'peer')] })
        )
        expect(useNodeConfigurationStore.getState().availability.get('peer')?.value).toBe(
            'available'
        )
        expect(useNodeConfigurationStore.getState().availability.get('peer')?.current).toBe(false)
        await tick()
        expect(useNodeConfigurationStore.getState().availability.get('peer')).toMatchObject({
            value: 'draining',
            pausedBy: 'peer'
        })
        getPolicy.mockResolvedValue(documentFor('paused', '2', 'peer'))
        await tick()
        expect(useNodeConfigurationStore.getState().availability.get('peer')?.value).toBe('paused')
        getPolicy.mockResolvedValue(documentFor('available', '3'))
        await act(async () => useDiscoveredNodesStore.setState({ nodes: [advertisement('3')] }))
        await tick()
        expect(useNodeConfigurationStore.getState().availability.get('peer')).toMatchObject({
            value: 'available',
            current: true
        })
        expect(getPolicy).toHaveBeenCalledTimes(4)
        expect(setAvailability).not.toHaveBeenCalled()
    })

    it('bounds draining readback and requires manual Refresh after sixty seconds', async () => {
        getPolicy.mockResolvedValue(documentFor('draining'))
        await watch()
        await tick(60_000)
        expect(getPolicy).toHaveBeenCalledTimes(30)
        expect(useNodeConfigurationStore.getState().availability.get('peer')).toMatchObject({
            value: 'draining',
            current: false,
            stale: true
        })
        await tick(10_000)
        expect(getPolicy).toHaveBeenCalledTimes(30)
        getPolicy.mockResolvedValue(documentFor('paused', '2'))
        await act(async () => useNodeConfigurationStore.getState().readAvailability('peer', true))
        expect(useNodeConfigurationStore.getState().availability.get('peer')).toMatchObject({
            value: 'paused',
            current: true,
            stale: false
        })
    })

    it('keeps one physical read in flight and stales even a hung read at the bound', async () => {
        let finish: ((document: NodePolicyDocument) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        await watch()
        await tick(60_000)
        expect(getPolicy).toHaveBeenCalledTimes(1)
        expect(useNodeConfigurationStore.getState().availability.get('peer')?.stale).toBe(true)
        await act(async () => finish?.(documentFor('paused')))
        expect(useNodeConfigurationStore.getState().availability.get('peer')?.current).toBe(false)
    })

    it('rejects an older revision and a read overtaken by a push', async () => {
        await watch()
        getPolicy.mockResolvedValue(documentFor('paused', '1', 'peer'))
        await act(async () =>
            useDiscoveredNodesStore.setState({ nodes: [advertisement('2', 'peer')] })
        )
        await tick()
        expect(useNodeConfigurationStore.getState().availability.get('peer')?.current).toBe(false)
        let finish: ((document: NodePolicyDocument) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        await tick()
        await act(async () => {
            changes.forEach(listener =>
                listener({ nodeId: 'peer', availability: 'draining', pausedBy: 'local' })
            )
            finish?.(documentFor('available', '2'))
        })
        expect(useNodeConfigurationStore.getState().availability.get('peer')).toMatchObject({
            value: 'available',
            current: false
        })
        getPolicy.mockResolvedValue(documentFor('draining', '2', 'local'))
        await tick()
        expect(useNodeConfigurationStore.getState().availability.get('peer')).toMatchObject({
            value: 'draining',
            pausedBy: 'local',
            current: true
        })
    })

    it('discards late replies on disconnect and reads again after reconnect without resending', async () => {
        let finish: ((document: NodePolicyDocument) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        await watch()
        await act(async () => useConnectionStore.setState({ connected: false }))
        await act(async () => finish?.(documentFor('paused')))
        expect(useNodeConfigurationStore.getState().availability.get('peer')?.current).toBe(false)
        await act(async () => useConnectionStore.setState({ connected: true }))
        await tick()
        expect(useNodeConfigurationStore.getState().availability.get('peer')).toMatchObject({
            value: 'available',
            current: true
        })
        expect(setAvailability).not.toHaveBeenCalled()
    })

    it('retains a lost-reply error and accepts only authoritative readback', async () => {
        await watch('self', undefined)
        setAvailability.mockRejectedValueOnce(new Error('Reply lost'))
        getPolicy.mockResolvedValue(documentFor('draining', '2', 'local'))
        await act(async () =>
            useNodeConfigurationStore.getState().requestAvailability('self', 'paused')
        )
        expect(useNodeConfigurationStore.getState().availability.get('self')?.error).toContain(
            'Reply lost'
        )
        expect(useNodeConfigurationStore.getState().availability.get('self')?.current).toBe(false)
        await tick()
        expect(useNodeConfigurationStore.getState().availability.get('self')).toMatchObject({
            value: 'draining',
            current: true
        })
        expect(setAvailability).toHaveBeenCalledTimes(1)
    })

    it('allows local Resume to supersede a pending configured drain and ignores its late completion', async () => {
        await watch('self', undefined)
        let finish: ((value: { availability: 'paused' }) => void) | undefined
        setAvailability.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        await act(async () => {
            void useNodeConfigurationStore.getState().requestAvailability('self', 'paused')
        })
        await act(async () =>
            changes.forEach(listener =>
                listener({ nodeId: 'self', availability: 'draining', pausedBy: 'local' })
            )
        )
        getPolicy.mockResolvedValue(documentFor('available', '3'))
        await act(async () =>
            useNodeConfigurationStore.getState().requestAvailability('self', 'available')
        )
        await tick()
        await act(async () => finish?.({ availability: 'paused' }))
        expect(setAvailability.mock.calls.map(call => call[0])).toEqual([
            { nodeId: undefined, state: 'paused', pauseMode: 'configured' },
            { nodeId: undefined, state: 'available' }
        ])
        expect(useNodeConfigurationStore.getState().availability.get('self')).toMatchObject({
            value: 'available',
            current: true
        })
    })
})
