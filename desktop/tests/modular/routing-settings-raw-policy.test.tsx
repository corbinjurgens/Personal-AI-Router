// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { INVALID_PARAMS_CODE } from '@/shared/constants/service-errors'
import type { NodePolicyDocument } from '@/shared/types/node-policy'
import type { NodeItem } from '@/shared/types/nodes'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi, type INodePolicyApi } from '@/ui/api/pair-api'
import { ServiceCallError } from '@/ui/api/service-transport'
import RoutingSettings from '@/ui/components/RoutingSettings/RoutingSettings'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { useNodesStore } from '@/ui/stores/nodes.store'

vi.mock('@/ui/components/RoutingSettings/RoleEditor', () => ({ default: () => null }))
vi.mock('@/ui/components/NodeList/NodeEngineSettings', () => ({ default: () => null }))

const BASE = '9007199254740993'
const peer: NodeItem = {
    id: 'peer',
    name: 'Peer device',
    status: 'active',
    ipAddress: '127.0.0.1',
    port: 1,
    allIpAddresses: [],
    os: 'Linux',
    topology: {
        cpu: { model: 'Fixture CPU', cores: 4, threads: 4 },
        gpus: [],
        ram: 4096,
        storage: []
    }
}
const getPolicy = vi.fn<INodePolicyApi['get']>()
const setPolicy = vi.fn<INodePolicyApi['set']>()
let root: Root
let container: HTMLDivElement

function saved(revision = BASE): NodePolicyDocument {
    return {
        policy: JSON.stringify({ version: 2, revision, profiles: [], roles: [] }),
        revision,
        availability: 'available',
        activation: { state: 'active', activeRevision: revision },
        recoveryRequired: false
    }
}

function editor(): HTMLTextAreaElement {
    const found = container.querySelector('textarea[aria-label="Node policy JSON"]')
    if (!(found instanceof HTMLTextAreaElement)) throw new Error('Missing policy editor')
    return found
}

function button(label: string): HTMLButtonElement {
    const found = Array.from(container.querySelectorAll('button')).find(
        item => item.textContent?.trim() === label || item.getAttribute('aria-label') === label
    )
    if (!found) throw new Error(`Missing button: ${label}`)
    return found
}

async function click(label: string) {
    await act(async () => button(label).click())
}

async function edit(text: string) {
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set
    if (!setter) throw new Error('Missing textarea value setter')
    await act(async () => {
        setter.call(editor(), text)
        editor().dispatchEvent(new Event('input', { bubbles: true }))
    })
}

async function pickPeer() {
    await click('Select node')
    const option = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
        item => item.textContent?.trim() === peer.name
    )
    if (!option) throw new Error('Missing peer option')
    await act(async () => option.click())
}

async function mount() {
    getPolicy.mockResolvedValue(saved())
    await act(async () => root.render(<RoutingSettings />))
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => false })
    getPolicy.mockReset()
    setPolicy.mockReset().mockRejectedValue(new Error('Unexpected policy write'))
    const api = createPairApi({
        connected: true,
        invoke: vi.fn<ServiceTransport['invoke']>().mockRejectedValue(new Error('Unexpected RPC')),
        subscribePush: vi.fn<ServiceTransport['subscribePush']>(() => () => {}),
        onConnect: vi.fn<ServiceTransport['onConnect']>(() => () => {}),
        onDisconnect: vi.fn<ServiceTransport['onDisconnect']>(() => () => {}),
        onAuthFailure: vi.fn<ServiceTransport['onAuthFailure']>(() => () => {}),
        destroy: vi.fn<ServiceTransport['destroy']>()
    })
    api.nodePolicy.get = getPolicy
    api.nodePolicy.set = setPolicy
    api.nodePolicy.onPolicyChanged = () => () => {}
    api.cluster.getRemoteManagementForNode = vi
        .fn<typeof api.cluster.getRemoteManagementForNode>()
        .mockResolvedValue({ version: 1, revision: '1', allowed: true, state: 'allowed' })
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    useConnectionStore.setState({ connected: true, selfId: 'self' })
    useOverviewUiStore.setState({ settingsNodeId: null, settingsProfileKey: null })
    useNodesStore.setState({ nodes: new Map([['peer', peer]]) })
    useNodeConfigurationStore.setState({
        availability: new Map(),
        editors: new Map(),
        management: new Map(),
        policyDrafts: new Map()
    })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.unstubAllGlobals()
})

describe('routing settings raw policy', () => {
    it('submits malformed text with its loaded CAS base and retains the exact text and validation error', async () => {
        await mount()
        const malformed = '  { "roles": [\n  '
        await edit(malformed)
        setPolicy.mockRejectedValueOnce(
            new ServiceCallError('Node policy must be valid JSON.', INVALID_PARAMS_CODE)
        )
        await click('Save')

        expect(setPolicy).toHaveBeenCalledExactlyOnceWith({
            nodeId: undefined,
            policy: malformed,
            expectedPersistedRevision: BASE
        })
        expect(editor().value).toBe(malformed)
        expect(container.textContent).toContain('Node policy must be valid JSON.')
        expect(container.textContent).toContain('Your policy text is kept.')
    })

    it('clears the previous target draft and refuses Save while the new target read is pending or fails', async () => {
        await mount()
        await edit('  { "roles": [\n  ')
        let rejectRead: ((error: Error) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise((_, reject) => {
                    rejectRead = reject
                })
        )
        await pickPeer()

        expect(editor().value).toBe('')
        expect(button('Save').disabled).toBe(true)
        await click('Save')
        if (!rejectRead) throw new Error('Missing peer read')
        await act(async () => rejectRead?.(new Error('Peer policy unreachable')))
        await click('Save')

        expect(setPolicy).not.toHaveBeenCalled()
        expect(getPolicy).toHaveBeenLastCalledWith('peer')
        expect(editor().value).toBe('')
        expect(container.textContent).toContain('Peer policy unreachable')
    })

    it('ignores a completed read for the previous target after selecting a peer', async () => {
        await mount()
        let finishRead: ((value: NodePolicyDocument) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finishRead = resolve
                })
        )
        await click('Reload')
        const peerPolicy = saved('42')
        getPolicy.mockResolvedValue(peerPolicy)
        await pickPeer()
        if (!finishRead) throw new Error('Missing previous target read')
        await act(async () => finishRead?.(saved('9007199254740994')))

        expect(editor().value).toBe(peerPolicy.policy)
        expect(setPolicy).not.toHaveBeenCalled()
    })
})
