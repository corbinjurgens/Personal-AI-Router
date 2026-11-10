// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act, useState } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import type { NodeItem } from '@/shared/types/nodes'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import type { IWindowActionApi } from '@/preload/api/window.api'
import type { OverviewCommand } from '@/shared/types/overview'
import { createPairApi } from '@/ui/api/pair-api'
import Settings from '@/ui/components/Settings'
import NodeCardDetails from '@/ui/components/NodeList/NodeCardDetails'
import ActiveNode from '@/ui/components/ClusterSettings/ActiveNode'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useOverviewCommands } from '@/ui/hooks/useOverviewCommands'
import NodeMemorySettings from '@/ui/components/RoutingSettings/NodeMemorySettings'

// Engine operations are outside this navigation contract; keep the real engine body.
vi.mock('@/ui/components/BackendRow/BackendRow', () => ({
    BackendRow: ({ nodeId }: { nodeId: string }) => {
        const [confirming, setConfirming] = useState(false)
        return (
            <div>
                Engine target {nodeId}
                <button onClick={() => setConfirming(true)}>Review engine action</button>
                {confirming && <span>Pending engine action for {nodeId}</span>}
            </div>
        )
    }
}))

let root: Root
let container: HTMLDivElement
const originalWindowApi = Object.getOwnPropertyDescriptor(window, 'windowApi')

it('edits idle and capacity in the policy draft while preserving unrelated numeric tokens', async () => {
    let policy =
        '{"opaque":9007199254740993,"idle":{"wakeTimeoutSeconds":90},"admission":{"memoryBudgetBytes":0}}'
    function MemoryEditor() {
        const [text, setText] = useState(policy)
        return (
            <NodeMemorySettings
                text={text}
                disabled={false}
                onChange={next => {
                    policy = next
                    setText(next)
                }}
            />
        )
    }
    await act(async () => root.render(<MemoryEditor />))
    const capacity = container.querySelector(
        'input[aria-label="Models that can stay loaded at once"]'
    )
    const idle = container.querySelector(
        'input[aria-label="Release model memory after N idle minutes"]'
    )
    if (!(capacity instanceof HTMLInputElement) || !(idle instanceof HTMLInputElement))
        throw new Error('Missing memory settings')
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
    if (!setter) throw new Error('Missing native input setter')
    await act(async () => {
        setter.call(capacity, '2')
        capacity.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await act(async () => {
        setter.call(idle, '0')
        idle.dispatchEvent(new Event('input', { bubbles: true }))
    })
    expect(JSON.parse(policy)).toMatchObject({
        idle: { unloadAfterMinutes: 0, wakeTimeoutSeconds: 90 },
        admission: { maxResidentModels: 2, memoryBudgetBytes: 0 }
    })
    expect(policy).toContain('9007199254740993')
    expect(container.textContent).toContain('Never')
})

function node(id: string, name: string): NodeItem {
    return {
        id,
        name,
        status: 'active',
        ipAddress: '127.0.0.1',
        port: 1,
        allIpAddresses: [],
        os: 'Windows',
        topology: { cpu: { model: 'CPU', cores: 4, threads: 4 }, gpus: [], ram: 4096, storage: [] }
    }
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => true })
    vi.stubGlobal(
        'ResizeObserver',
        class {
            observe = vi.fn<ResizeObserver['observe']>()
            disconnect = vi.fn<ResizeObserver['disconnect']>()
        }
    )
    const transport: ServiceTransport = {
        connected: false,
        invoke: vi.fn<ServiceTransport['invoke']>().mockRejectedValue(new Error('Disconnected')),
        subscribePush: () => () => {},
        onConnect: () => () => {},
        onDisconnect: () => () => {},
        onAuthFailure: () => () => {},
        destroy: () => {}
    }
    Object.defineProperty(window, 'pairApi', {
        configurable: true,
        value: createPairApi(transport)
    })
    useConnectionStore.setState({ connected: false, selfId: 'local' })
    useNodesStore.setState({
        nodes: new Map([
            ['local', node('local', 'Local A')],
            ['peer', node('peer', 'Peer B')]
        ])
    })
    useOverviewUiStore.setState({
        activeTab: 'overview',
        settingsSubTab: 'cluster',
        settingsNodeId: null
    })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    if (originalWindowApi) Object.defineProperty(window, 'windowApi', originalWindowApi)
    else Reflect.deleteProperty(window, 'windowApi')
    vi.unstubAllGlobals()
})

it('focuses Settings and Node settings for B and renders the engines for B', async () => {
    useOverviewUiStore.getState().focusNodeSettings('peer')
    expect(useOverviewUiStore.getState()).toMatchObject({
        activeTab: 'settings',
        settingsSubTab: 'routing',
        settingsNodeId: 'peer'
    })
    await act(async () => root.render(<Settings />))
    expect(container.textContent).toContain('Node settings')
    expect(container.textContent).toContain('Settings for Peer B')
    expect(container.textContent).toContain('Engines')
    expect(container.textContent).toContain('Engine target peer')
    expect(container.textContent).not.toContain('Engine target local')
})

function NavigationSurface() {
    useOverviewCommands()
    const activeTab = useOverviewUiStore(state => state.activeTab)
    return activeTab === 'settings' ? (
        <Settings />
    ) : (
        <NodeCardDetails node={node('peer', 'Peer B')} />
    )
}

it('opens the chosen node page with one card settings click', async () => {
    await act(async () => root.render(<NavigationSurface />))
    expect(container.textContent).not.toContain('Engine target peer')
    const settings = container.querySelector<HTMLButtonElement>('[aria-label="Node settings"]')
    if (!settings) throw new Error('Missing card settings button')
    expect(settings.hasAttribute('aria-expanded')).toBe(false)
    await act(async () => settings.click())
    expect(useOverviewUiStore.getState()).toMatchObject({
        activeTab: 'settings',
        settingsSubTab: 'routing',
        settingsNodeId: 'peer'
    })
    expect(container.textContent).toContain('Settings for Peer B')
    expect(container.textContent).toContain('Engine target peer')
    expect(container.textContent).not.toContain('Engine target local')
})

it('opens the requested node from the tray overview command boundary', async () => {
    let receive: ((command: OverviewCommand) => void) | undefined
    const unsubscribe = vi.fn<() => void>()
    const onOverviewCommand = vi.fn<IWindowActionApi['onOverviewCommand']>(callback => {
        receive = callback
        return unsubscribe
    })
    const overviewReady = vi.fn<IWindowActionApi['overviewReady']>().mockResolvedValue()
    Object.defineProperty(window, 'windowApi', {
        configurable: true,
        value: { window: { onOverviewCommand, overviewReady } }
    })
    await act(async () => root.render(<NavigationSurface />))
    expect(overviewReady).toHaveBeenCalledExactlyOnceWith()
    if (!receive) throw new Error('Missing overview command subscription')
    await act(async () => receive?.({ type: 'focus-node', nodeId: 'peer' }))
    expect(useOverviewUiStore.getState()).toMatchObject({
        activeTab: 'settings',
        settingsSubTab: 'routing',
        settingsNodeId: 'peer'
    })
    expect(container.textContent).toContain('Settings for Peer B')
    expect(container.textContent).toContain('Engine target peer')
    await act(async () => root.render(null))
    expect(unsubscribe).toHaveBeenCalledExactlyOnceWith()
})

it('Cluster Edit node targets the stable node identity', async () => {
    await act(async () =>
        root.render(
            <ActiveNode
                node={{
                    nodeUuid: 'peer',
                    name: 'Peer B',
                    displayName: 'Peer B',
                    ipAddress: '127.0.0.1',
                    port: 1,
                    clusterId: 'cluster',
                    state: 'member',
                    joinedAt: null,
                    lastSeen: null
                }}
            />
        )
    )
    const menu = container.querySelector<HTMLButtonElement>('[aria-label="Open Peer B menu"]')
    if (!menu) throw new Error('Missing cluster node menu')
    await act(async () => menu.click())
    const edit = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
        item => item.textContent?.trim() === 'Edit node'
    )
    if (!edit) throw new Error('Missing Edit node action')
    await act(async () => edit.click())
    expect(useOverviewUiStore.getState()).toMatchObject({
        activeTab: 'settings',
        settingsSubTab: 'routing',
        settingsNodeId: 'peer'
    })
    await act(async () => root.render(<Settings />))
    expect(container.textContent).toContain('Settings for Peer B')
    expect(container.textContent).toContain('Engine target peer')
})

it('the node picker switches the shared target to local A', async () => {
    useOverviewUiStore.getState().focusNodeSettings('peer')
    await act(async () => root.render(<Settings />))
    const picker = container.querySelector<HTMLButtonElement>('[aria-label="Select node"]')
    if (!picker) throw new Error('Missing node picker')
    await act(async () => picker.click())
    const local = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
        item => item.textContent?.trim() === 'This PC'
    )
    if (!local) throw new Error('Missing local picker option')
    await act(async () => local.click())
    expect(useOverviewUiStore.getState().settingsNodeId).toBeNull()
    expect(container.textContent).toContain('Settings for Local A')
    expect(container.textContent).toContain('Engine target local')
    expect(container.textContent).not.toContain('Engine target peer')
})

it('keeps the requested peer target when its directory entry disappears', async () => {
    useOverviewUiStore.getState().focusNodeSettings('peer')
    await act(async () => root.render(<Settings />))
    await act(async () =>
        useNodesStore.setState({ nodes: new Map([['local', node('local', 'Local A')]]) })
    )
    expect(useOverviewUiStore.getState().settingsNodeId).toBe('peer')
    expect(container.textContent).toContain('Settings for peer')
    expect(container.textContent).toContain('Engine target peer')
    expect(container.textContent).not.toContain('Settings for Local A')
    expect(container.textContent).not.toContain('Engine target local')
})

it('updates an already mounted settings page when focus changes to the local UUID', async () => {
    useOverviewUiStore.getState().focusNodeSettings('peer')
    await act(async () => root.render(<Settings />))
    await act(async () => useOverviewUiStore.getState().focusNodeSettings('local'))
    expect(container.textContent).toContain('Settings for Local A')
    expect(container.textContent).toContain('Engine target local')
    expect(container.textContent).not.toContain('Engine target peer')
})

it('clears pending engine row state when a mounted page focuses another node', async () => {
    useOverviewUiStore.getState().focusNodeSettings('local')
    await act(async () => root.render(<Settings />))
    const review = Array.from(container.querySelectorAll('button')).find(
        button => button.textContent === 'Review engine action'
    )
    if (!review) throw new Error('Missing engine action')
    await act(async () => review.click())
    expect(container.textContent).toContain('Pending engine action for local')
    await act(async () => useOverviewUiStore.getState().focusNodeSettings('peer'))
    expect(container.textContent).toContain('Engine target peer')
    expect(container.textContent).not.toContain('Pending engine action')
})

it('keeps routing history on Cluster and node settings on the selected node page', async () => {
    useOverviewUiStore.getState().focusNodeSettings('peer')
    await act(async () => root.render(<Settings />))
    expect(container.textContent).not.toContain('Configure several nodes')
    expect(container.textContent).not.toContain('Routing history')
    const cluster = Array.from(container.querySelectorAll<HTMLElement>('*')).find(
        item => item.textContent === 'Cluster' && item.children.length === 0
    )
    if (!cluster) throw new Error('Missing Cluster navigation')
    await act(async () => cluster.click())
    expect(useOverviewUiStore.getState().settingsSubTab).toBe('cluster')
    expect(container.textContent).not.toContain('Configure several nodes')
    expect(container.textContent).toContain('Routing history')
    expect(container.textContent).not.toContain('Settings for Peer B')
    expect(container.textContent).not.toContain('Engine target peer')
    expect(container.textContent).not.toContain('Engine target local')
})
