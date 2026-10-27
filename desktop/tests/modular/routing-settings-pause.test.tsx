// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { LOCAL_VETO_CODE } from '@/shared/constants/service-errors'
import type { NodeAvailabilityChange, NodePolicyDocument } from '@/shared/types/node-policy'
import type { NodeItem } from '@/shared/types/nodes'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi, type INodePolicyApi } from '@/ui/api/pair-api'
import { ServiceCallError } from '@/ui/api/service-transport'
import RoutingSettings from '@/ui/components/RoutingSettings/RoutingSettings'
import { useConnectionStore } from '@/ui/stores/connection.store'
import {
    initializeNodeAvailability,
    useNodeConfigurationStore
} from '@/ui/stores/node-configuration.store'
import { useNodesStore } from '@/ui/stores/nodes.store'

vi.mock('@/ui/components/RoutingSettings/RoleEditor', () => ({ default: () => null }))
vi.mock('@/ui/components/RoutingSettings/BulkNodeConfiguration', () => ({ default: () => null }))

function documentFor(
    availability: NodePolicyDocument['availability'],
    revision = '1',
    pausedBy = 'local'
): NodePolicyDocument {
    return {
        policy: JSON.stringify({ version: 2, revision, availability, pausedBy }),
        revision,
        activation: null,
        recoveryRequired: false,
        availability
    }
}

const getPolicy = vi.fn<INodePolicyApi['get']>()
const setAvailability = vi.fn<INodePolicyApi['setAvailability']>()
let availabilityChanged: ((change: NodeAvailabilityChange) => void) | undefined
let root: Root
let container: HTMLDivElement
let cleanupAvailability: () => void

async function readback(): Promise<void> {
    await act(async () => {
        await vi.advanceTimersByTimeAsync(2_000)
    })
}

function button(label: string): HTMLButtonElement {
    const found = Array.from(container.querySelectorAll('button')).find(
        item => item.textContent?.trim() === label || item.getAttribute('aria-label') === label
    )
    if (!found) throw new Error(`Missing ${label} button`)
    return found
}

async function mount(availability: NodePolicyDocument['availability'] = 'draining') {
    getPolicy.mockResolvedValue(documentFor(availability))
    await act(async () => root.render(<RoutingSettings />))
}

beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => true })
    getPolicy.mockReset()
    setAvailability.mockReset().mockResolvedValue({ availability: 'paused' })
    availabilityChanged = undefined
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
        availabilityChanged = callback
        return () => {
            availabilityChanged = undefined
        }
    }
    api.cluster.getRemoteManagementForNode = vi
        .fn<typeof api.cluster.getRemoteManagementForNode>()
        .mockResolvedValue({ version: 1, revision: '1', allowed: true, state: 'allowed' })
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    useConnectionStore.setState({ connected: true, selfId: 'self' })
    useNodesStore.setState({ nodes: new Map() })
    useNodeConfigurationStore.setState({
        availability: new Map(),
        editors: new Map(),
        management: new Map(),
        policyDrafts: new Map()
    })
    cleanupAvailability = initializeNodeAvailability()
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    cleanupAvailability()
    container.remove()
    useNodeConfigurationStore.setState({
        availability: new Map(),
        editors: new Map(),
        management: new Map(),
        policyDrafts: new Map()
    })
    vi.unstubAllGlobals()
    vi.useRealTimers()
})

describe('routing settings pause recovery', () => {
    it('reads the draining fence after a physical pause failure and preserves its error', async () => {
        await mount('available')
        setAvailability.mockRejectedValueOnce(new Error('Owned engine did not stop.'))
        getPolicy.mockResolvedValue(documentFor('draining', '2'))

        await act(async () => button('Pause node').click())
        await readback()

        expect(setAvailability).toHaveBeenCalledExactlyOnceWith({
            state: 'paused',
            nodeId: undefined,
            pauseMode: 'configured'
        })
        expect(getPolicy).toHaveBeenCalledTimes(2)
        expect(container.textContent).toContain('Owned engine did not stop.')
        expect(button('Retry pause').disabled).toBe(false)
        expect(button('Resume').disabled).toBe(false)
        expect(container.textContent).toContain(
            'Pause unloads models and stops controlled engines.'
        )
    })

    it('retries an incomplete pause and adopts the paused readback', async () => {
        await mount()
        getPolicy.mockResolvedValue(documentFor('paused', '2'))

        await act(async () => button('Retry pause').click())
        await readback()

        expect(setAvailability).toHaveBeenCalledExactlyOnceWith({
            state: 'paused',
            nodeId: undefined,
            pauseMode: 'configured'
        })
        expect(button('Resume').disabled).toBe(false)
        expect(container.textContent).not.toContain('Retry pause')
        expect(container.textContent).toContain('Paused')
    })

    it('retains both recovery choices and the operational error after retry failure', async () => {
        await mount()
        setAvailability.mockRejectedValueOnce(new Error('Shutdown remains incomplete.'))

        await act(async () => button('Retry pause').click())
        await readback()

        expect(getPolicy).toHaveBeenCalledTimes(2)
        expect(container.textContent).toContain('Shutdown remains incomplete.')
        expect(button('Retry pause').disabled).toBe(false)
        expect(button('Resume').disabled).toBe(false)
    })

    it('waits for authoritative policy before displaying a successful resume', async () => {
        await mount()
        setAvailability.mockResolvedValueOnce({ availability: 'available' })
        let finishRead: ((value: NodePolicyDocument) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finishRead = resolve
                })
        )

        await act(async () => button('Resume').click())
        await readback()

        expect(setAvailability).toHaveBeenCalledExactlyOnceWith({
            state: 'available',
            nodeId: undefined
        })
        expect(container.textContent).toContain('Availability unconfirmed')
        expect(button('Resume').disabled).toBe(true)
        if (!finishRead) throw new Error('Missing resume policy readback')
        await act(async () => finishRead?.(documentFor('available', '2')))
        expect(button('Pause node').disabled).toBe(false)
        expect(container.textContent).not.toContain('Retry pause')
    })

    it('retains draining after a failed resume and refreshes its policy', async () => {
        await mount()
        setAvailability.mockRejectedValueOnce(new Error('Resume was superseded by another pause.'))

        await act(async () => button('Resume').click())
        await readback()

        expect(getPolicy).toHaveBeenCalledTimes(2)
        expect(container.textContent).toContain('Resume was superseded by another pause.')
        expect(button('Retry pause').disabled).toBe(false)
        expect(button('Resume').disabled).toBe(false)
    })

    it('preserves the action error when the authoritative refresh also fails', async () => {
        await mount()
        setAvailability.mockRejectedValueOnce(new Error('Pause could not stop the engine.'))
        getPolicy.mockRejectedValueOnce(new Error('Policy readback disconnected.'))

        await act(async () => button('Retry pause').click())
        await readback()

        expect(container.textContent).toContain('Pause could not stop the engine.')
        expect(container.textContent).toContain('Policy readback disconnected.')
        expect(button('Retry pause').disabled).toBe(true)
        expect(button('Resume').disabled).toBe(true)
        expect(button('Refresh').disabled).toBe(false)
    })

    it('restores local draining recovery choices from reconnect readback without writing', async () => {
        await mount()
        await act(async () => useConnectionStore.setState({ connected: false }))
        expect(button('Retry pause').disabled).toBe(true)
        expect(button('Resume').disabled).toBe(true)

        await act(async () => useConnectionStore.setState({ connected: true }))

        expect(getPolicy).toHaveBeenCalledTimes(2)
        expect(button('Retry pause').disabled).toBe(false)
        expect(button('Resume').disabled).toBe(false)
        expect(setAvailability).not.toHaveBeenCalled()
    })

    it('does not replace newer draining evidence with an obsolete resume readback', async () => {
        await mount()
        setAvailability.mockResolvedValueOnce({ availability: 'available' })
        let finishRead: ((value: NodePolicyDocument) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finishRead = resolve
                })
        )
        await act(async () => button('Resume').click())
        await readback()
        if (!availabilityChanged || !finishRead)
            throw new Error('Missing availability observer/read')

        await act(async () => {
            availabilityChanged?.({ nodeId: 'self', availability: 'draining' })
            finishRead?.(documentFor('available', '2'))
        })

        expect(button('Retry pause').disabled).toBe(false)
        expect(button('Resume').disabled).toBe(false)
        expect(container.textContent).toContain('Pausing')
        expect(
            useNodeConfigurationStore.getState().editors.get('self')?.baseline?.availability
        ).toBe('draining')
    })

    it('fences peer writes until fresh draining readback and retains its role draft through an older read', async () => {
        const peer: NodeItem = {
            id: 'peer',
            name: 'Peer',
            status: 'active',
            ipAddress: '127.0.0.1',
            port: 1,
            allIpAddresses: [],
            os: 'Linux',
            topology: {
                cpu: { model: 'CPU', cores: 4, threads: 4 },
                gpus: [],
                ram: 4096,
                storage: []
            }
        }
        useNodesStore.setState({ nodes: new Map([['peer', peer]]) })
        await mount('available')
        await act(async () => button('Select node').click())
        const option = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
            item => item.textContent?.trim() === 'Peer'
        )
        if (!option) throw new Error('Missing peer menu item')
        await act(async () => option.click())
        const draft = { weak: { engine: 'ollama', model: 'unsent-role' } }
        useNodeConfigurationStore.getState().setDraft('peer', draft)
        const baselinePolicy = useNodeConfigurationStore.getState().editors.get('peer')
            ?.baseline?.policy
        let finishRead: ((value: NodePolicyDocument) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finishRead = resolve
                })
        )
        await act(async () => button('Reload').click())
        if (!availabilityChanged || !finishRead)
            throw new Error('Missing availability observer/read')

        await act(async () => availabilityChanged?.({ nodeId: 'peer', availability: 'draining' }))

        expect(useNodeConfigurationStore.getState().editors.get('peer')).toMatchObject({
            baseline: { availability: 'available', policy: baselinePolicy },
            draft
        })
        expect(useNodeConfigurationStore.getState().availability.get('peer')?.current).toBe(false)
        expect(button('Save').disabled).toBe(true)
        await act(async () => finishRead?.(documentFor('available', '0')))
        expect(useNodeConfigurationStore.getState().availability.get('peer')?.current).toBe(false)
        expect(button('Save').disabled).toBe(true)
        getPolicy.mockResolvedValue(documentFor('draining', '2'))
        await readback()
        expect(useNodeConfigurationStore.getState().availability.get('peer')).toMatchObject({
            value: 'draining',
            current: true,
            revision: '2'
        })
        expect(useNodeConfigurationStore.getState().editors.get('peer')).toMatchObject({
            baseline: { availability: 'draining' },
            draft
        })
        expect(container.querySelector('[aria-label="Resume node: Peer"]')).toBeNull()
        expect(button('Save').disabled).toBe(true)
        expect(setAvailability).not.toHaveBeenCalled()
    })

    it('retains pending Resume through reconnect and reads current draining state after completion', async () => {
        await mount()
        let finishResume: ((value: { availability: 'available' }) => void) | undefined
        setAvailability.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finishResume = resolve
                })
        )
        await act(async () => button('Resume').click())
        await act(async () => useConnectionStore.setState({ connected: false }))
        await act(async () => useConnectionStore.setState({ connected: true }))
        await readback()
        expect(useNodeConfigurationStore.getState().availability.get('self')).toMatchObject({
            value: 'draining',
            current: true,
            pending: 'available'
        })
        expect(button('Retry pause').disabled).toBe(true)
        expect(button('Resume').disabled).toBe(true)
        if (!finishResume) throw new Error('Missing pending Resume request')

        await act(async () => finishResume?.({ availability: 'available' }))
        await readback()

        expect(button('Retry pause').disabled).toBe(false)
        expect(button('Resume').disabled).toBe(false)
        expect(container.textContent).toContain('Pausing')
        expect(useNodeConfigurationStore.getState().availability.get('self')).toMatchObject({
            value: 'draining',
            current: true,
            pending: undefined
        })
        expect(setAvailability).toHaveBeenCalledExactlyOnceWith({
            nodeId: undefined,
            state: 'available'
        })
    })

    it('shows the target refusal detail when a policy save is vetoed', async () => {
        await mount('available')
        const detail = 'This node is paused. Resume it on this node before managing it remotely.'
        window.pairApi.nodePolicy.set = vi
            .fn<INodePolicyApi['set']>()
            .mockRejectedValue(new ServiceCallError(detail, LOCAL_VETO_CODE))
        await act(async () => button('Save').click())

        expect(container.textContent).toContain('the target refused the operation')
        expect(container.textContent).toContain(detail)
    })

    it.each([
        { availability: 'paused', pausedBy: 'local' },
        { availability: 'paused', pausedBy: 'peer' },
        { availability: 'draining', pausedBy: 'peer' }
    ] satisfies { availability: NodePolicyDocument['availability']; pausedBy: string }[])(
        'keeps $pausedBy-origin $availability peer views fenced with reads available',
        async ({ availability, pausedBy }) => {
            const peer: NodeItem = {
                id: 'peer',
                name: 'Peer',
                status: 'active',
                ipAddress: '127.0.0.1',
                port: 1,
                allIpAddresses: [],
                os: 'Linux',
                topology: {
                    cpu: { model: 'CPU', cores: 4, threads: 4 },
                    gpus: [],
                    ram: 4096,
                    storage: []
                }
            }
            useNodesStore.setState({ nodes: new Map([['peer', peer]]) })
            await mount('available')
            getPolicy.mockResolvedValue(documentFor(availability, '2', pausedBy))
            await act(async () => button('Select node').click())
            const option = Array.from(
                document.querySelectorAll<HTMLElement>('[role="menuitem"]')
            ).find(item => item.textContent?.trim() === 'Peer')
            if (!option) throw new Error('Missing peer menu item')
            await act(async () => option.click())

            expect(container.querySelector('[aria-label="Resume node: Peer"]')).toBeNull()
            expect(button('Save').disabled).toBe(true)
            expect(button('Reload').disabled).toBe(false)
            expect(container.textContent).toContain(
                'This node is paused. Resume it on that node before making changes.'
            )
            await act(async () => button('Pause node').click())
            await act(async () => button('Reload').click())
            expect(setAvailability).not.toHaveBeenCalled()
            expect(getPolicy).toHaveBeenLastCalledWith('peer')
        }
    )
})
