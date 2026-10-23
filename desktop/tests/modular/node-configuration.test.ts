// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))

import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import {
    INVALID_PARAMS_CODE,
    LOCAL_VETO_CODE,
    POLICY_CONFLICT_CODE,
    REMOTE_MANAGEMENT_DENIED_CODE
} from '@/shared/constants/service-errors'
import { ServiceCallError } from '@/ui/api/service-transport'
import type { EngineModels, ModelItem } from '@/shared/types/engines'
import type { JsonValue } from '@/shared/types/json'
import type { NodePolicyDocument, NodePolicySnapshot } from '@/shared/types/node-policy'
import type { IClusterApi, INodePolicyApi } from '@/ui/api/pair-api'
import NodeDeclaredRoles from '@/ui/components/NodeList/NodeDeclaredRoles'
import RoleEditor from '@/ui/components/RoutingSettings/RoleEditor'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import {
    inventoryChoices,
    NODE_PAUSED_MESSAGE,
    policyRefusal,
    policyRoles,
    roleDifferences,
    suggestRoles,
    useNodeConfigurationStore,
    withRoles
} from '@/ui/stores/node-configuration.store'

const SELF = 'self'
const PEER = 'peer'

function model(name: string, downloaded = true): ModelItem {
    return {
        name,
        size: 0,
        downloaded,
        status: 'idle',
        parameterSize: '',
        quantization: '',
        family: '',
        digest: '',
        sizeVram: null,
        expiresAt: null,
        expiry: '10m',
        capabilities: []
    }
}

const inventory: EngineModels[] = [
    { nodeId: PEER, engineType: 'ollama', models: [model('qwen3:4b'), model('gone', false)] },
    { nodeId: PEER, engineType: 'llama-cpp', models: [model('gemma-3-27b')] }
]

/** A peer projection: device fields, local tier entries and roles. */
function policyText(revision: string, extra: Record<string, JsonValue> = {}): string {
    return JSON.stringify(
        {
            version: 2,
            revision,
            availability: 'available',
            admission: { maxResidentModels: 1 },
            tiers: {
                weak: [
                    { kind: 'local', engine: 'ollama', model: 'missing' },
                    { kind: 'local', engine: 'ollama', model: 'qwen3:4b' }
                ],
                strong: [{ kind: 'api', deployment: 'cloud' }]
            },
            ...extra
        },
        null,
        2
    )
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

const get = vi.fn<INodePolicyApi['get']>()
const set = vi.fn<INodePolicyApi['set']>()
const management = vi.fn<IClusterApi['getRemoteManagementForNode']>()

beforeEach(() => {
    get.mockReset()
    set.mockReset()
    management
        .mockReset()
        .mockResolvedValue({ version: 1, allowed: true, state: 'allowed', revision: '1' })
    Object.defineProperty(window, 'pairApi', {
        configurable: true,
        value: { nodePolicy: { get, set }, cluster: { getRemoteManagementForNode: management } }
    })
    useNodeConfigurationStore.setState({ editors: new Map() })
})

function entry(targetId = PEER) {
    const found = useNodeConfigurationStore.getState().editors.get(targetId)
    if (!found) throw new Error(`no role draft for ${targetId}`)
    return found
}

describe('role substitution', () => {
    it.each<{ name: string; policy: Record<string, JsonValue> }>([
        { name: 'local pause', policy: { availability: 'paused', pausedBy: 'local' } },
        { name: 'peer pause', policy: { availability: 'paused', pausedBy: 'peer' } },
        { name: 'draining', policy: { availability: 'draining', pausedBy: 'peer' } }
    ])('keeps a remote role draft without sending during $name', async ({ policy }) => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5', policy))
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })

        await store.save(PEER, PEER)
        await store.save(PEER, PEER)

        expect(set).not.toHaveBeenCalled()
        expect(entry()).toMatchObject({
            phase: 'vetoed',
            message: NODE_PAUSED_MESSAGE,
            draft: { weak: { engine: 'ollama', model: 'qwen3:4b' } }
        })
    })

    it('uses live draining readback even while the saved document is available', async () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, { ...policyDocument('5'), availability: 'draining' })
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        await store.save(PEER, PEER)
        expect(set).not.toHaveBeenCalled()
        expect(entry().phase).toBe('vetoed')
    })

    it('does not treat locked-engine model evidence as a node-wide pause', async () => {
        const store = useNodeConfigurationStore.getState()
        const locked: NodePolicySnapshot = {
            ...snapshot('5'),
            modelConfiguration: {
                policyRevision: '5',
                observationGeneration: 1,
                models: [
                    {
                        engine: 'ollama',
                        model: 'qwen3:4b',
                        desiredResident: true,
                        state: 'blocked',
                        reason: 'engine-locked'
                    }
                ]
            }
        }
        store.adopt(PEER, locked)
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        set.mockResolvedValue(snapshot('6'))
        await store.save(PEER, PEER)
        expect(set).toHaveBeenCalledTimes(1)
        expect(entry().phase).toBe('applied')
    })

    it('permits local role configuration while paused', async () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(SELF, snapshot('5', { availability: 'paused', pausedBy: 'peer' }))
        store.setRole(SELF, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        set.mockResolvedValue(
            snapshot('6', {
                availability: 'paused',
                roles: { weak: { engine: 'ollama', model: 'qwen3:4b' } }
            })
        )
        await store.save(SELF, undefined)
        expect(set).toHaveBeenCalledTimes(1)
        expect(entry(SELF).phase).toBe('applied')
    })

    it('keeps canonical pause options returned by the server after a roles-only save', async () => {
        const store = useNodeConfigurationStore.getState()
        const legacyPause = {
            stopEngines: false,
            unloadModels: false,
            onActive: 'drain',
            drainTimeoutSeconds: 12
        }
        const canonicalPause = { ...legacyPause, stopEngines: true, unloadModels: true }
        store.adopt(PEER, snapshot('5', { pause: legacyPause }))
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        set.mockResolvedValue(
            snapshot('6', {
                pause: canonicalPause,
                roles: { weak: { engine: 'ollama', model: 'qwen3:4b' } }
            })
        )
        await store.save(PEER, PEER)
        expect(JSON.parse(set.mock.calls[0][0].policy).pause).toEqual(legacyPause)
        expect(JSON.parse(entry().baseline?.policy ?? '{}').pause).toEqual(canonicalPause)
        expect(entry().draft).toEqual(entry().baseline?.roles)
    })

    it('keeps a refused draft after resume until preview and an explicit revision-checked save', async () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        set.mockRejectedValueOnce(new ServiceCallError('pause accepted', LOCAL_VETO_CODE))
        await store.save(PEER, PEER)
        store.adopt(PEER, snapshot('6'))
        expect(set).toHaveBeenCalledTimes(1)
        expect(entry().draft.weak?.model).toBe('qwen3:4b')

        get.mockResolvedValue(policyDocument('6'))
        await store.save(PEER, PEER)
        expect(set).toHaveBeenCalledTimes(1)
        expect(entry().phase).toBe('draft')
        set.mockResolvedValue(
            snapshot('7', { roles: { weak: { engine: 'ollama', model: 'qwen3:4b' } } })
        )
        await store.save(PEER, PEER)
        expect(set).toHaveBeenCalledTimes(2)
        expect(set.mock.calls[1][0].expectedPersistedRevision).toBe('6')
    })

    it.each(['setRole', 'setDraft'])(
        'keeps the veto fence when %s edits a refused draft',
        async edit => {
            const store = useNodeConfigurationStore.getState()
            store.adopt(PEER, snapshot('5'))
            store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
            set.mockRejectedValueOnce(new ServiceCallError('pause accepted', LOCAL_VETO_CODE))
            await store.save(PEER, PEER)

            const changed = { engine: 'ollama', model: 'changed-model' }
            if (edit === 'setRole') store.setRole(PEER, 'weak', changed)
            else store.setDraft(PEER, { weak: changed })
            expect(entry().phase).toBe('vetoed')

            get.mockResolvedValue(policyDocument('6', { availability: 'paused', pausedBy: 'peer' }))
            await store.save(PEER, PEER)

            expect(get).toHaveBeenCalledTimes(1)
            expect(set).toHaveBeenCalledTimes(1)
            expect(entry()).toMatchObject({
                phase: 'vetoed',
                message: NODE_PAUSED_MESSAGE,
                draft: { weak: changed },
                baseline: { revision: '6' }
            })
        }
    )

    it('retains live draining after an identical policy push omits availability', async () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, { ...policyDocument('5'), availability: 'draining' })
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        store.adopt(PEER, snapshot('5'))

        expect(entry().baseline?.availability).toBe('draining')
        await store.save(PEER, PEER)
        expect(set).not.toHaveBeenCalled()
        expect(entry().phase).toBe('vetoed')
    })

    it('allows a newer saved available revision to clear old draining readback', async () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, { ...policyDocument('5'), availability: 'draining' })
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        store.adopt(PEER, snapshot('6'))
        expect(entry().baseline?.availability).toBeUndefined()
        expect(set).not.toHaveBeenCalled()

        set.mockResolvedValue(snapshot('7'))
        await store.save(PEER, PEER)
        expect(set).toHaveBeenCalledTimes(1)
        expect(set.mock.calls[0][0].expectedPersistedRevision).toBe('6')
    })

    it('replaces only roles and keeps every other field of the target document', () => {
        const text = withRoles(
            policyText('5', { roles: { weak: { engine: 'ollama', model: 'a' } } }),
            {
                strong: { engine: 'llamacpp', model: 'gemma-3-27b' }
            }
        )
        const sent = JSON.parse(text)
        const original = JSON.parse(policyText('5'))
        expect(sent).toEqual({
            ...original,
            roles: { strong: { engine: 'llamacpp', model: 'gemma-3-27b' } }
        })
    })

    it('sends the substituted document with the target revision to a peer', async () => {
        set.mockResolvedValue(
            snapshot('6', { roles: { medium: { engine: 'ollama', model: 'qwen3:4b' } } })
        )
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.setRole(PEER, 'medium', { engine: 'ollama', model: 'qwen3:4b' })

        const saved = await store.save(PEER, PEER)

        expect(set).toHaveBeenCalledTimes(1)
        const request = set.mock.calls[0][0]
        expect(request.nodeId).toBe(PEER)
        expect(request.expectedPersistedRevision).toBe('5')
        expect(JSON.parse(request.policy)).toEqual({
            ...JSON.parse(policyText('5')),
            roles: { medium: { engine: 'ollama', model: 'qwen3:4b' } }
        })
        expect(saved?.revision).toBe('6')
        expect(entry().phase).toBe('applied')
        expect(entry().baseline?.revision).toBe('6')
    })

    it('omits the node id for this PC', async () => {
        set.mockResolvedValue(snapshot('2'))
        const store = useNodeConfigurationStore.getState()
        store.adopt(SELF, snapshot('1'))
        store.setRole(SELF, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        await store.save(SELF, undefined)
        expect(set.mock.calls[0][0].nodeId).toBeUndefined()
        expect(set.mock.calls[0][0].expectedPersistedRevision).toBe('1')
    })
})

describe('unset tiers', () => {
    it('reads a document without roles as declaring nothing', () => {
        expect(policyRoles(policyText('1'))).toEqual({})
    })

    it('drops a tier set back to unset from the document sent', async () => {
        set.mockResolvedValue(snapshot('4'))
        const store = useNodeConfigurationStore.getState()
        store.adopt(
            PEER,
            snapshot('3', {
                roles: {
                    weak: { engine: 'ollama', model: 'qwen3:4b' },
                    strong: { engine: 'llamacpp', model: 'gemma-3-27b' }
                }
            })
        )
        store.setRole(PEER, 'weak', undefined)
        await store.save(PEER, PEER)
        expect(JSON.parse(set.mock.calls[0][0].policy).roles).toEqual({
            strong: { engine: 'llamacpp', model: 'gemma-3-27b' }
        })
    })

    it('ignores unknown tiers and incomplete roles', () => {
        expect(
            policyRoles(
                JSON.stringify({
                    roles: {
                        weak: { engine: 'ollama' },
                        tiny: { engine: 'ollama', model: 'x' },
                        medium: { engine: 'ollama', model: 'y' }
                    }
                })
            )
        ).toEqual({ medium: { engine: 'ollama', model: 'y' } })
    })
})

describe('conflicts', () => {
    it.each([
        '-32000: policy conflict: the persisted revision is 6, not the expected 5',
        '-32010: the persisted revision is 6, not the expected 5'
    ])('reads the target back and shows the diff without retrying: %s', async message => {
        set.mockRejectedValue(new Error(message))
        get.mockResolvedValue(
            policyDocument('6', { roles: { weak: { engine: 'llamacpp', model: 'gemma-3-27b' } } })
        )
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })

        expect(await store.save(PEER, PEER)).toBeNull()

        expect(set).toHaveBeenCalledTimes(1)
        expect(get).toHaveBeenCalledTimes(1)
        expect(get).toHaveBeenCalledWith(PEER)
        const { phase, latest, draft } = entry()
        expect(phase).toBe('conflict')
        expect(latest?.revision).toBe('6')
        expect(roleDifferences(draft, latest?.roles ?? {})).toEqual([
            {
                tier: 'weak',
                mine: { engine: 'ollama', model: 'qwen3:4b' },
                theirs: { engine: 'llamacpp', model: 'gemma-3-27b' }
            }
        ])
        // A save while the conflict stands sends nothing.
        await store.save(PEER, PEER)
        expect(set).toHaveBeenCalledTimes(1)
    })

    it('sends again only after the user bases the draft on the latest', async () => {
        set.mockRejectedValueOnce(new Error('-32000: policy conflict: stale'))
        set.mockResolvedValue(snapshot('7'))
        get.mockResolvedValue(policyDocument('6'))
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        await store.save(PEER, PEER)
        store.keepDraftOnLatest(PEER)
        expect(entry().phase).toBe('draft')
        await store.save(PEER, PEER)
        expect(set).toHaveBeenCalledTimes(2)
        expect(set.mock.calls[1][0].expectedPersistedRevision).toBe('6')
    })

    it('marks a draft stale when a newer document changes the roles', () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        store.adopt(PEER, snapshot('6', { roles: { weak: { engine: 'ollama', model: 'z' } } }))
        expect(entry().phase).toBe('conflict')
    })

    it('rebases a draft when a newer document leaves the roles alone', () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        store.adopt(PEER, snapshot('6', { availability: 'paused' }))
        expect(entry().phase).toBe('draft')
        expect(entry().baseline?.revision).toBe('6')
        expect(entry().draft.weak?.model).toBe('qwen3:4b')
    })

    it('shows a local veto and reads nothing back', async () => {
        set.mockRejectedValue(new Error('-32011: this node was paused on the node itself'))
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        await store.save(PEER, PEER)
        expect(entry().phase).toBe('vetoed')
        expect(get).not.toHaveBeenCalled()
    })

    it('reports an unconfirmed outcome when the read back fails', async () => {
        set.mockRejectedValue(new Error('connection lost'))
        get.mockRejectedValue(new Error('connection lost'))
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        await store.save(PEER, PEER)
        expect(entry().phase).toBe('unknown')
        expect(set).toHaveBeenCalledTimes(1)
    })

    it('classifies a refusal by its JSON-RPC code, not by its text', () => {
        expect(policyRefusal('the persisted revision is 6', POLICY_CONFLICT_CODE)).toBe('conflict')
        expect(policyRefusal('paused on the node itself', LOCAL_VETO_CODE)).toBe('vetoed')
        expect(policyRefusal('roles[weak]: unknown engine', INVALID_PARAMS_CODE)).toBe('rejected')
        expect(
            policyRefusal('device has blocked remote management', REMOTE_MANAGEMENT_DENIED_CODE)
        ).toBe('rejected')
        expect(policyRefusal('-32011: policy conflict: misleading text', -32000)).toBe('other')
        expect(policyRefusal('-32000: engine manager unavailable', -32000)).toBe('unreachable')
    })

    it('shows a coded local veto whose message names no code', async () => {
        set.mockRejectedValue(new ServiceCallError('paused on the node itself', LOCAL_VETO_CODE))
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        await store.save(PEER, PEER)
        expect(entry().phase).toBe('vetoed')
        expect(get).not.toHaveBeenCalled()
    })

    it('does not read a code out of the text of a coded failure', async () => {
        set.mockRejectedValue(new ServiceCallError('-32011: the policy could not be saved', -32000))
        get.mockResolvedValue(policyDocument('5'))
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        await store.save(PEER, PEER)
        expect(entry().phase).toBe('rejected')
        expect(get).toHaveBeenCalledTimes(1)
    })

    it('falls back to the message only when a failure carries no code', () => {
        expect(policyRefusal('-32602: roles[weak]: unknown engine "x"')).toBe('rejected')
        expect(policyRefusal('-32011: vetoed')).toBe('vetoed')
        expect(policyRefusal('-32012: permission denied')).toBe('rejected')
        expect(policyRefusal('-32000: the policy could not be saved')).toBe('other')
        expect(policyRefusal('-32000: engine manager unavailable')).toBe('unreachable')
        expect(policyRefusal('ui-broker policy:set timed out')).toBe('other')
    })

    it('keeps the draft and reads nothing back when the node was not reached', async () => {
        set.mockRejectedValue(
            new Error('-32000: node policy unavailable: device is offline or does not support it')
        )
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        await store.save(PEER, PEER)
        expect(entry().phase).toBe('unreachable')
        expect(entry().draft.weak?.model).toBe('qwen3:4b')
        expect(get).not.toHaveBeenCalled()
    })

    it('keeps a permission-refused draft rejected independently of local veto and misleading transport text', async () => {
        set.mockRejectedValue(
            new ServiceCallError('device is offline -32011', REMOTE_MANAGEMENT_DENIED_CODE)
        )
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.setRole(PEER, 'weak', { engine: 'ollama', model: 'qwen3:4b' })
        await store.save(PEER, PEER)
        expect(entry().phase).toBe('rejected')
        expect(entry().message).toContain('Remote management permission')
        expect(entry().draft.weak).toEqual({ engine: 'ollama', model: 'qwen3:4b' })
        expect(entry().baseline?.revision).toBe('5')
        expect(get).not.toHaveBeenCalled()
        expect(set).toHaveBeenCalledTimes(1)
    })
})

describe('inventory choices', () => {
    it('offers only installed models of that node, in engine-manager spelling', () => {
        expect(inventoryChoices(inventory).map(({ engine, model }) => ({ engine, model }))).toEqual(
            [
                { engine: 'llamacpp', model: 'gemma-3-27b' },
                { engine: 'ollama', model: 'qwen3:4b' }
            ]
        )
    })

    it('suggests the first installed local tier entry per tier and leaves the rest unset', () => {
        const choices = inventoryChoices(inventory)
        expect(suggestRoles(choices, [policyText('1')])).toEqual({
            weak: { engine: 'ollama', model: 'qwen3:4b' }
        })
    })

    it('falls back to the later document for a tier the first leaves unmatched', () => {
        const choices = inventoryChoices(inventory)
        const own = JSON.stringify({
            tiers: { strong: [{ kind: 'local', engine: 'llamacpp', model: 'gemma-3-27b' }] }
        })
        expect(suggestRoles(choices, [policyText('1'), own])).toEqual({
            weak: { engine: 'ollama', model: 'qwen3:4b' },
            strong: { engine: 'llamacpp', model: 'gemma-3-27b' }
        })
    })
})

describe('declared roles on the node card', () => {
    it('carries the advertisement through the broker snapshot and keeps it across a proxy update', () => {
        const state = getModularBridgeState()
        const brokerNode = {
            hostUuid: 'uuid-roles-1',
            name: 'roles-host-1',
            ipAddress: '192.0.2.41',
            port: 14318,
            trusted: true,
            clustered: true,
            policy: {
                revision: '9',
                roles: {
                    strong: { engine: 'llamacpp', model: 'gemma-3-27b', contextTokens: 8192 }
                },
                pausedBy: 'local'
            }
        }
        state.handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: { nodes: [brokerNode] }
        })
        state.handleNotification({
            source: 'ollama-proxy',
            method: 'node/discovered',
            params: { id: 'uuid-roles-1', host: 'roles-host-1', port: 11434, ip: '192.0.2.41' }
        })
        const available = () => state.getAvailableNodes().find(node => node.id === 'uuid-roles-1')
        expect(available()?.policy).toEqual({
            revision: '9',
            roles: { strong: { engine: 'llamacpp', model: 'gemma-3-27b' } },
            pausedBy: 'local'
        })

        const { policy: _dropped, ...withoutPolicy } = brokerNode
        void _dropped
        state.handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: { nodes: [withoutPolicy] }
        })
        expect(available()).not.toHaveProperty('policy')
    })

    describe('rendering', () => {
        let container: HTMLDivElement
        let root: Root
        beforeEach(() => {
            vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
            container = document.createElement('div')
            document.body.append(container)
            root = createRoot(container)
        })
        afterEach(async () => {
            await act(async () => root.unmount())
            container.remove()
            useDiscoveredNodesStore.setState({ nodes: [] })
            useEngineModelsStore.setState({ models: new Map() })
            vi.unstubAllGlobals()
        })

        const discovered = (roles: Record<string, { engine: string; model: string }> | null) =>
            useDiscoveredNodesStore.setState({
                nodes: [
                    {
                        id: PEER,
                        name: 'peer',
                        ipAddress: '192.0.2.5',
                        port: 1,
                        lastSeen: 1,
                        trusted: true,
                        clustered: true,
                        ...(roles ? { policy: { revision: '3', roles } } : {})
                    }
                ]
            })

        it('lists each declared tier', async () => {
            discovered({
                weak: { engine: 'ollama', model: 'qwen3:4b' },
                strong: { engine: 'llamacpp', model: 'gemma-3-27b' }
            })
            await act(async () => root.render(createElement(NodeDeclaredRoles, { nodeId: PEER })))
            expect(container.textContent).toContain('weak: qwen3:4b (Ollama)')
            expect(container.textContent).toContain('strong: gemma-3-27b (llama.cpp)')
            expect(container.textContent).not.toContain('medium')
        })

        it('shows nothing for a node that declares no role', async () => {
            discovered(null)
            await act(async () => root.render(createElement(NodeDeclaredRoles, { nodeId: PEER })))
            expect(container.textContent).toBe('')
            await act(async () => discovered({}))
            expect(container.textContent).toBe('')
        })

        it('shows a peer-origin pause with local-resume guidance only for a paired node', async () => {
            vi.stubGlobal('CSS', { supports: () => true })
            const paused = { availability: 'paused', pausedBy: 'peer' }
            useNodeConfigurationStore.getState().adopt(PEER, snapshot('5', paused))
            useNodeConfigurationStore.getState().adopt(SELF, snapshot('5', paused))
            const onSaved = vi.fn()

            await act(async () =>
                root.render(createElement(RoleEditor, { targetId: PEER, nodeId: PEER, onSaved }))
            )
            expect(container.textContent).toContain('PausedBy: peer')
            expect(container.textContent).toContain(NODE_PAUSED_MESSAGE)
            expect(container.textContent).toContain('unset tiers are not served')

            await act(async () =>
                root.render(
                    createElement(RoleEditor, { targetId: SELF, nodeId: undefined, onSaved })
                )
            )
            expect(container.textContent).not.toContain(NODE_PAUSED_MESSAGE)
        })
    })
})
