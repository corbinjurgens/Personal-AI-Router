// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { JsonValue } from '@/shared/types/json'
import type { NodePolicyDocument, NodePolicySnapshot } from '@/shared/types/node-policy'
import type { NodeRoles } from '@/shared/types/node-roles'
import type { NodeItem, NodeStatus } from '@/shared/types/nodes'
import type { IClusterApi, INodePolicyApi } from '@/ui/api/pair-api'
import { REMOTE_MANAGEMENT_DENIED_CODE } from '@/shared/constants/service-errors'
import { ServiceCallError } from '@/ui/api/service-transport'
import BulkNodeConfiguration from '@/ui/components/RoutingSettings/BulkNodeConfiguration'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import {
    BULK_CONCURRENCY,
    targetState,
    useNodeConfigurationStore,
    type RoleEditorEntry
} from '@/ui/stores/node-configuration.store'
import { useNodesStore } from '@/ui/stores/nodes.store'

const WEAK = { engine: 'ollama', model: 'qwen3:4b' }
const ROLES: NodeRoles = { weak: WEAK }

function policyText(revision: string, extra: Record<string, JsonValue> = {}): string {
    return JSON.stringify({ version: 2, revision, availability: 'available', ...extra }, null, 2)
}

function snapshot(revision: string, extra: Record<string, JsonValue> = {}): NodePolicySnapshot {
    return { policy: policyText(revision, extra), revision, activation: null }
}

function policyDocument(
    revision: string,
    extra: Record<string, JsonValue> = {}
): NodePolicyDocument {
    return { ...snapshot(revision, extra), availability: 'available', recoveryRequired: false }
}

/** The document a successful save returns: the request's policy at the next revision. */
function persisted(request: { policy: string; expectedPersistedRevision: string }) {
    const revision = String(BigInt(request.expectedPersistedRevision) + 1n)
    return { policy: request.policy, revision, activation: null }
}

const get = vi.fn<INodePolicyApi['get']>()
const set = vi.fn<INodePolicyApi['set']>()
const getRemoteManagementForNode = vi.fn<IClusterApi['getRemoteManagementForNode']>()

beforeEach(() => {
    get.mockReset()
    set.mockReset()
    getRemoteManagementForNode.mockReset()
    getRemoteManagementForNode.mockResolvedValue({
        version: 1,
        state: 'allowed',
        allowed: true,
        revision: '1'
    })
    Object.defineProperty(window, 'pairApi', {
        configurable: true,
        value: { nodePolicy: { get, set }, cluster: { getRemoteManagementForNode } }
    })
    useNodeConfigurationStore.setState({
        editors: new Map(),
        management: new Map(),
        bulk: { selected: [], mode: 'common', common: {} }
    })
})

function entry(targetId: string) {
    const found = useNodeConfigurationStore.getState().editors.get(targetId)
    if (!found) throw new Error(`no draft for ${targetId}`)
    return found
}

const store = () => useNodeConfigurationStore.getState()

const targets = (...ids: string[]) => ids.map(id => ({ targetId: id, nodeId: id }))

describe('bulk fan-out', () => {
    it('never has more than four per-node operations in flight', async () => {
        const ids = Array.from({ length: 10 }, (_, index) => `node-${index}`)
        const active = new Set<string>()
        let most = 0
        const later = () => new Promise(resolve => setTimeout(resolve, 0))
        get.mockImplementation(async nodeId => {
            active.add(nodeId ?? '')
            most = Math.max(most, active.size)
            await later()
            return policyDocument('5')
        })
        set.mockImplementation(async request => {
            await later()
            active.delete(request.nodeId ?? '')
            return persisted(request)
        })

        await store().applyAll(targets(...ids), ROLES)

        expect(BULK_CONCURRENCY).toBe(4)
        expect(most).toBe(4)
        expect(set).toHaveBeenCalledTimes(10)
        expect(ids.every(id => entry(id).phase === 'applied')).toBe(true)
    })

    it('substitutes the common roles into each node’s own document with its own revision', async () => {
        get.mockImplementation(async nodeId =>
            policyDocument(nodeId === 'a' ? '5' : '9', { admission: { maxResidentModels: 1 } })
        )
        set.mockImplementation(async request => persisted(request))

        await store().applyAll(targets('a', 'b'), ROLES)

        const sent = new Map(set.mock.calls.map(([request]) => [request.nodeId, request]))
        expect(sent.get('a')?.expectedPersistedRevision).toBe('5')
        expect(sent.get('b')?.expectedPersistedRevision).toBe('9')
        expect(JSON.parse(sent.get('b')?.policy ?? '{}')).toEqual({
            ...JSON.parse(policyText('9', { admission: { maxResidentModels: 1 } })),
            roles: { weak: { engine: 'ollama', model: 'qwen3:4b' } }
        })
    })

    it('keeps the nodes that succeeded when another fails, with no rollback', async () => {
        get.mockResolvedValue(policyDocument('5'))
        set.mockImplementation(async request => {
            if (request.nodeId === 'b') throw new Error('-32602: roles[weak]: unknown engine')
            return persisted(request)
        })

        await store().applyAll(targets('a', 'b', 'c'), ROLES)

        expect(entry('a').phase).toBe('applied')
        expect(entry('b').phase).toBe('rejected')
        expect(entry('c').phase).toBe('applied')
        // One send per node, and nothing sent afterwards to undo a or c.
        expect(set).toHaveBeenCalledTimes(3)
        expect(entry('a').baseline?.revision).toBe('6')
    })

    it('marks a node already holding the roles applied without sending', async () => {
        get.mockResolvedValue(policyDocument('5', { roles: { weak: WEAK } }))
        await store().apply('a', 'a', ROLES)
        expect(set).not.toHaveBeenCalled()
        expect(entry('a').phase).toBe('applied')
    })

    it('retains successful rows and unsent drafts in a mixed permission refusal and unreachable attempt', async () => {
        get.mockImplementation(async nodeId => {
            if (nodeId === 'offline') throw new Error('device is offline')
            return policyDocument('5')
        })
        set.mockImplementation(async request => {
            if (request.nodeId === 'denied')
                throw new ServiceCallError('not granted by target', REMOTE_MANAGEMENT_DENIED_CODE)
            return persisted(request)
        })
        store().adopt('offline', snapshot('4'))
        store().setDraft('offline', ROLES)
        await store().applyAll(targets('allowed', 'denied', 'offline'), ROLES)
        expect(entry('allowed').phase).toBe('applied')
        expect(entry('allowed').baseline?.revision).toBe('6')
        expect(entry('denied').phase).toBe('rejected')
        expect(entry('denied').message).toContain('Remote management permission')
        expect(entry('denied').draft).toEqual(ROLES)
        expect(entry('offline').phase).toBe('unreachable')
        expect(entry('offline').draft).toEqual(ROLES)
        expect(set.mock.calls.map(([request]) => request.nodeId)).toEqual(['allowed', 'denied'])
        expect(get.mock.calls.map(([nodeId]) => nodeId)).toEqual(['allowed', 'denied', 'offline'])
    })

    it('rejects denied and unsupported readbacks without sending and keeps their drafts', async () => {
        get.mockResolvedValue(policyDocument('5'))
        set.mockImplementation(async request => persisted(request))
        getRemoteManagementForNode.mockImplementation(async nodeId => {
            if (nodeId === 'denied')
                return { version: 1, state: 'denied', allowed: false, revision: '2' }
            if (nodeId === 'unsupported')
                return { version: 1, state: 'unsupported', allowed: false, revision: '' }
            return { version: 1, state: 'allowed', allowed: true, revision: '2' }
        })
        for (const id of ['denied', 'unsupported', 'allowed']) {
            store().adopt(id, snapshot('4'))
            store().setDraft(id, ROLES)
        }

        await store().applyAll(targets('denied', 'unsupported', 'allowed'))

        expect(set.mock.calls.map(([request]) => request.nodeId)).toEqual(['allowed'])
        expect(get.mock.calls.map(([nodeId]) => nodeId)).toEqual(['allowed'])
        for (const id of ['denied', 'unsupported']) {
            expect(entry(id).phase).toBe('rejected')
            expect(entry(id).message).toBe(store().management.get(id)?.message)
            expect(entry(id).draft).toEqual(ROLES)
            expect(entry(id).baseline?.revision).toBe('4')
        }
        expect(entry('denied').message).toContain('blocked remote management')
        expect(entry('unsupported').message).toContain('does not support')
        expect(entry('allowed').phase).toBe('applied')
        expect(entry('allowed').baseline?.revision).toBe('6')
    })

    it('applies local roles without requiring remote management capability', async () => {
        get.mockResolvedValue(policyDocument('5'))
        set.mockImplementation(async request => persisted(request))
        useNodeConfigurationStore.setState({
            management: new Map([['self', { state: 'denied', revision: '2' }]])
        })

        await store().applyAll([{ targetId: 'self', nodeId: undefined }], ROLES)

        expect(getRemoteManagementForNode).not.toHaveBeenCalled()
        expect(set).toHaveBeenCalledTimes(1)
        expect(entry('self').phase).toBe('applied')
    })
})

describe('per-node outcomes', () => {
    beforeEach(() => {
        get.mockResolvedValue(policyDocument('5'))
    })

    it('shows a conflict from -32010 with the node read back, and resends nothing', async () => {
        get.mockResolvedValueOnce(policyDocument('5'))
        get.mockResolvedValueOnce(
            policyDocument('6', { roles: { weak: { engine: 'llamacpp', model: 'g' } } })
        )
        set.mockRejectedValue(new Error('-32010: the persisted revision is 6, not the expected 5'))

        await store().apply('a', 'a', ROLES)

        expect(entry('a').phase).toBe('conflict')
        expect(entry('a').latest?.revision).toBe('6')
        expect(set).toHaveBeenCalledTimes(1)
    })

    it('shows a local veto from -32011', async () => {
        set.mockRejectedValue(new Error('-32011: this node was paused on the node itself'))
        await store().apply('a', 'a', ROLES)
        expect(entry('a').phase).toBe('vetoed')
        expect(get).toHaveBeenCalledTimes(1)
    })

    it.each([
        'node policy unavailable: device is offline or does not support it',
        'engine manager unavailable',
        'dial tcp 192.0.2.5:14318: connect: connection refused'
    ])('is unreachable when the send never reached the node: %s', async message => {
        set.mockRejectedValue(new Error(`-32000: ${message}`))
        await store().apply('a', 'a', ROLES)
        expect(entry('a').phase).toBe('unreachable')
        // Nothing reached the node, so there is nothing to read back.
        expect(get).toHaveBeenCalledTimes(1)
        expect(set).toHaveBeenCalledTimes(1)
    })

    it('is unreachable, with the draft kept, when the node cannot be read first', async () => {
        store().adopt('a', snapshot('5'))
        store().setRole('a', 'weak', ROLES.weak)
        get.mockRejectedValue(new Error('-32000: node policy unavailable: device is offline'))

        await store().apply('a', 'a')

        expect(entry('a').phase).toBe('unreachable')
        expect(entry('a').draft).toEqual(ROLES)
        expect(entry('a').baseline?.revision).toBe('5')
        expect(set).not.toHaveBeenCalled()
    })

    it('is unknown when the reply is lost after sending and the read back fails', async () => {
        get.mockResolvedValueOnce(policyDocument('5'))
        get.mockRejectedValueOnce(new Error('ui-broker policy:get timed out'))
        set.mockRejectedValue(new Error('ui-broker policy:set timed out'))

        await store().apply('a', 'a', ROLES)

        expect(entry('a').phase).toBe('unknown')
        expect(set).toHaveBeenCalledTimes(1)
    })

    it('settles an unknown outcome by reading back before any resend', async () => {
        get.mockResolvedValueOnce(policyDocument('5'))
        get.mockRejectedValueOnce(new Error('ui-broker policy:get timed out'))
        set.mockRejectedValue(new Error('ui-broker policy:set timed out'))
        await store().apply('a', 'a', ROLES)

        // The lost save did land: the retry finds the roles and sends nothing.
        get.mockResolvedValueOnce(policyDocument('6', { roles: { weak: WEAK } }))
        await store().apply('a', 'a', ROLES)

        expect(entry('a').phase).toBe('applied')
        expect(entry('a').baseline?.revision).toBe('6')
        expect(set).toHaveBeenCalledTimes(1)
    })

    it('keeps a per-node conflict until the user resolves it', async () => {
        store().adopt('a', snapshot('5'))
        store().setRole('a', 'weak', ROLES.weak)
        get.mockResolvedValue(
            policyDocument('6', { roles: { weak: { engine: 'llamacpp', model: 'g' } } })
        )

        await store().apply('a', 'a')
        expect(entry('a').phase).toBe('conflict')
        await store().apply('a', 'a')
        expect(entry('a').phase).toBe('conflict')
        expect(set).not.toHaveBeenCalled()

        set.mockImplementation(async request => persisted(request))
        store().keepDraftOnLatest('a')
        await store().apply('a', 'a')
        expect(set).toHaveBeenCalledTimes(1)
        expect(set.mock.calls[0][0].expectedPersistedRevision).toBe('6')
    })

    it('keeps one request id through an attempt and takes a new one for a retry', async () => {
        let release: (document: NodePolicyDocument) => void = () => {}
        get.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    release = resolve
                })
        )
        let reply: (snapshot: NodePolicySnapshot) => void = () => {}
        set.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    reply = resolve
                })
        )

        const attempt = store().apply('a', 'a', ROLES)
        expect(entry('a').phase).toBe('reading')
        const id = entry('a').requestId
        expect(id).toMatch(/^[0-9a-f-]{36}$/)

        await vi.waitFor(() => expect(get).toHaveBeenCalledTimes(1))
        release(policyDocument('5'))
        await vi.waitFor(() => expect(entry('a').phase).toBe('sending'))
        expect(entry('a').requestId).toBe(id)

        reply(snapshot('6'))
        await attempt
        expect(entry('a').phase).toBe('applied')
        expect(entry('a').requestId).toBe(id)

        get.mockResolvedValue(policyDocument('6'))
        set.mockRejectedValue(new Error('-32011: vetoed'))
        await store().apply('a', 'a', { strong: { engine: 'ollama', model: 'x' } })
        expect(entry('a').requestId).not.toBe(id)
    })
})

describe('advertised', () => {
    it('follows this node’s discovery of the target revision', () => {
        const applied: RoleEditorEntry = {
            baseline: { policy: '', revision: '6', roles: {} },
            draft: {},
            phase: 'applied'
        }
        expect(targetState(applied, undefined)).toBe('applied')
        expect(targetState(applied, { revision: '5', roles: {} })).toBe('applied')
        expect(targetState(applied, { revision: '6', roles: {} })).toBe('advertised')
        expect(targetState(applied, { revision: '12', roles: {} })).toBe('advertised')
        expect(targetState({ ...applied, phase: 'unknown' }, { revision: '12', roles: {} })).toBe(
            'unknown'
        )
        expect(targetState(undefined, { revision: '12', roles: {} })).toBe('draft')
    })
})

describe('panel', () => {
    const topology: NodeItem['topology'] = {
        cpu: { model: 'CPU', cores: 4, threads: 4 },
        gpus: [],
        ram: 4096,
        storage: []
    }

    function node(id: string, status: NodeStatus = 'active'): NodeItem {
        return {
            id,
            name: `host-${id}`,
            status,
            ipAddress: '192.0.2.1',
            port: 1,
            allIpAddresses: [],
            topology,
            os: 'Linux'
        }
    }

    function peers(...nodes: NodeItem[]) {
        useNodesStore.setState({
            nodes: new Map(
                [node('self'), ...nodes].map((item): [string, NodeItem] => [item.id, item])
            )
        })
    }

    let container: HTMLDivElement
    let root: Root

    beforeEach(async () => {
        vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
        vi.stubGlobal('CSS', { supports: () => true })
        useConnectionStore.setState({ selfId: 'self' })
        container = document.createElement('div')
        document.body.append(container)
        root = createRoot(container)
    })

    afterEach(async () => {
        await act(async () => root.unmount())
        container.remove()
        useNodesStore.setState({ nodes: new Map() })
        useDiscoveredNodesStore.setState({ nodes: [] })
        vi.unstubAllGlobals()
    })

    const render = () => act(async () => root.render(createElement(BulkNodeConfiguration)))

    function button(label: string): HTMLButtonElement {
        const found = Array.from(container.querySelectorAll('button')).find(candidate =>
            candidate.textContent?.startsWith(label)
        )
        if (!found) throw new Error(`no button ${label}`)
        return found
    }

    it('keeps an offline node’s draft unsent, including after it reconnects', async () => {
        get.mockResolvedValue(policyDocument('5'))
        set.mockImplementation(async request => persisted(request))
        peers(node('a', 'offline'))
        store().setBulk({ selected: ['a'], common: ROLES })
        await render()

        expect(button('Send to').textContent).toBe('Send to 0 nodes')
        expect(button('Send to').disabled).toBe(true)
        expect(container.textContent).toContain('not sent when the node returns')
        expect(get).not.toHaveBeenCalled()

        await act(async () => peers(node('a')))
        // The node is read for its preview, never sent to.
        await vi.waitFor(() => expect(entry('a').phase).toBe('draft'))
        expect(set).not.toHaveBeenCalled()
        expect(store().bulk.common).toEqual(ROLES)

        await act(async () => button('Send to').click())
        await vi.waitFor(() => expect(entry('a').phase).toBe('applied'))
        expect(set).toHaveBeenCalledTimes(1)
    })

    it('retries only the selected failures', async () => {
        const unreadable = new Set<string>()
        get.mockImplementation(async nodeId => {
            if (unreadable.has(nodeId ?? '')) throw new Error('ui-broker policy:get timed out')
            return policyDocument('5')
        })
        set.mockImplementation(async request => {
            if (request.nodeId === 'b') throw new Error('-32000: engine manager unavailable')
            if (request.nodeId === 'c') {
                unreadable.add('c')
                throw new Error('ui-broker policy:set timed out')
            }
            return persisted(request)
        })
        peers(node('a'), node('b'), node('c'))
        store().setBulk({ selected: ['a', 'b', 'c'], common: ROLES })
        await render()
        await vi.waitFor(() => expect(button('Send to').textContent).toBe('Send to 3 nodes'))

        await act(async () => button('Send to').click())
        await vi.waitFor(() => expect(entry('c').phase).toBe('unknown'))
        expect(entry('a').phase).toBe('applied')
        expect(entry('b').phase).toBe('unreachable')

        // The user leaves c out of the retry.
        unreadable.clear()
        set.mockImplementation(async request => persisted(request))
        get.mockClear()
        set.mockClear()
        await act(async () => store().setBulk({ selected: ['a', 'b'] }))
        expect(button('Retry selected failures').textContent).toBe('Retry selected failures (1)')

        await act(async () => button('Retry selected failures').click())
        await vi.waitFor(() => expect(entry('b').phase).toBe('applied'))
        expect(get.mock.calls.map(([nodeId]) => nodeId)).toEqual(['b'])
        expect(set.mock.calls.map(([request]) => request.nodeId)).toEqual(['b'])
        expect(entry('c').phase).toBe('unknown')
    })

    it('shows a node as advertised once discovery reports its saved revision', async () => {
        get.mockResolvedValue(policyDocument('5'))
        set.mockImplementation(async request => persisted(request))
        peers(node('a'))
        store().setBulk({ selected: ['a'], common: ROLES })
        await render()
        await vi.waitFor(() => expect(button('Send to').disabled).toBe(false))
        await act(async () => button('Send to').click())
        await vi.waitFor(() => expect(container.textContent).toContain('Saved at revision 6'))

        const discovered = (revision: string) =>
            useDiscoveredNodesStore.setState({
                nodes: [
                    {
                        id: 'a',
                        name: 'host-a',
                        ipAddress: '192.0.2.1',
                        port: 1,
                        lastSeen: 1,
                        trusted: true,
                        clustered: true,
                        policy: { revision, roles: ROLES }
                    }
                ]
            })
        await act(async () => discovered('5'))
        expect(container.textContent).not.toContain('Advertised')
        await act(async () => discovered('6'))
        expect(container.textContent).toContain('Advertised at revision 6')
    })
})
