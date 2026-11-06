// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { customEngine, customEngineDescriptor } from './engine-fixtures'
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
    REMOTE_MANAGEMENT_DENIED_CODE,
    REMOTE_SETTING_LOCKED_CODE
} from '@/shared/constants/service-errors'
import { ServiceCallError } from '@/ui/api/service-transport'
import type { EngineModels, ModelItem } from '@/shared/types/engines'
import type { JsonValue } from '@/shared/types/json'
import type { NodePolicyDocument, NodePolicySnapshot } from '@/shared/types/node-policy'
import type { NodeRoles } from '@/shared/types/node-roles'
import type { IClusterApi, INodePolicyApi } from '@/ui/api/pair-api'
import NodeDeclaredRoles from '@/ui/components/NodeList/NodeDeclaredRoles'
import RoleEditor from '@/ui/components/RoutingSettings/RoleEditor'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
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
const roleProfiles = [
    { key: 'qwen-fast', displayName: 'Qwen', engine: 'llamacpp', model: 'qwen3:4b' },
    { key: 'gemma-3-27b', displayName: 'Gemma', engine: 'llamacpp-custom', model: 'gemma-3-27b' }
]

function model(name: string, downloaded = true): ModelItem {
    return {
        name,
        size: 0,
        downloaded,
        status: 'idle',
        parameterSize: '',
        family: '',
        sizeVram: null,
        expiresAt: null,
        expiry: '10m',
        capabilities: []
    }
}

const inventory: EngineModels[] = [
    { nodeId: PEER, engineType: 'llama-cpp', models: [model('qwen3:4b'), model('gone', false)] },
    { nodeId: PEER, engineType: customEngine, models: [model('gemma-3-27b')] }
]

/** A peer projection with explicit profiles and no implicit role declarations. */
function policyText(revision: string, extra: Record<string, JsonValue> = {}): string {
    return JSON.stringify(
        {
            version: 2,
            revision,
            availability: 'available',
            admission: { maxResidentModels: 1 },
            profiles: roleProfiles,
            tiers: {},
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
    it('retains the saved pause mode in the policy baseline', () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(
            PEER,
            snapshot('5', { availability: 'paused', pausedBy: 'local', pauseMode: 'immediate' })
        )
        expect(entry().baseline?.pauseMode).toBe('immediate')
        store.adopt(
            PEER,
            snapshot('6', { availability: 'paused', pausedBy: 'local', pauseMode: 'configured' })
        )
        expect(entry().baseline?.pauseMode).toBe('configured')
    })

    it.each<{ name: string; policy: Record<string, JsonValue> }>([
        { name: 'local pause', policy: { availability: 'paused', pausedBy: 'local' } },
        { name: 'peer pause', policy: { availability: 'paused', pausedBy: 'peer' } },
        { name: 'draining', policy: { availability: 'draining', pausedBy: 'peer' } }
    ])('keeps a remote role draft without sending during $name', async ({ policy }) => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5', policy))
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')

        await store.save(PEER, PEER)
        await store.save(PEER, PEER)

        expect(set).not.toHaveBeenCalled()
        expect(entry()).toMatchObject({
            phase: 'vetoed',
            message: NODE_PAUSED_MESSAGE,
            draft: [{ name: 'weak', profiles: ['qwen-fast'] }]
        })
    })

    it('uses live draining readback even while the saved document is available', async () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, { ...policyDocument('5'), availability: 'draining' })
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
        await store.save(PEER, PEER)
        expect(set).not.toHaveBeenCalled()
        expect(entry().phase).toBe('vetoed')
    })

    it('permits local role configuration while paused', async () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(SELF, snapshot('5', { availability: 'paused', pausedBy: 'peer' }))
        store.addRole(SELF, 'weak')
        store.setRole(SELF, 'weak', 'qwen-fast')
        set.mockResolvedValue(
            snapshot('6', {
                availability: 'paused',
                roles: [{ name: 'weak', profiles: ['qwen-fast'] }]
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
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
        set.mockResolvedValue(
            snapshot('6', {
                pause: canonicalPause,
                roles: [{ name: 'weak', profiles: ['qwen-fast'] }]
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
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
        set.mockRejectedValueOnce(new ServiceCallError('pause accepted', LOCAL_VETO_CODE))
        await store.save(PEER, PEER)
        store.adopt(PEER, snapshot('6'))
        expect(set).toHaveBeenCalledTimes(1)
        expect(entry().draft.find(role => role.name === 'weak')?.profiles[0]).toBe('qwen-fast')

        get.mockResolvedValue(policyDocument('6'))
        await store.save(PEER, PEER)
        expect(set).toHaveBeenCalledTimes(1)
        expect(entry().phase).toBe('draft')
        set.mockResolvedValue(snapshot('7', { roles: [{ name: 'weak', profiles: ['qwen-fast'] }] }))
        await store.save(PEER, PEER)
        expect(set).toHaveBeenCalledTimes(2)
        expect(set.mock.calls[1][0].expectedPersistedRevision).toBe('6')
    })

    it.each(['setRole', 'setDraft'])(
        'keeps the veto fence when %s edits a refused draft',
        async edit => {
            const store = useNodeConfigurationStore.getState()
            store.adopt(PEER, snapshot('5'))
            store.addRole(PEER, 'weak')
            store.setRole(PEER, 'weak', 'qwen-fast')
            set.mockRejectedValueOnce(new ServiceCallError('pause accepted', LOCAL_VETO_CODE))
            await store.save(PEER, PEER)

            const changed = 'changed-model'
            if (edit === 'setRole') store.setRole(PEER, 'weak', changed)
            else store.setDraft(PEER, [{ name: 'weak', profiles: [changed] }])
            expect(entry().phase).toBe('vetoed')

            get.mockResolvedValue(policyDocument('6', { availability: 'paused', pausedBy: 'peer' }))
            await store.save(PEER, PEER)

            expect(get).toHaveBeenCalledTimes(1)
            expect(set).toHaveBeenCalledTimes(1)
            expect(entry()).toMatchObject({
                phase: 'vetoed',
                message: NODE_PAUSED_MESSAGE,
                draft: [{ name: 'weak', profiles: [changed] }],
                baseline: { revision: '6' }
            })
        }
    )

    it('retains live draining after an identical policy push omits availability', async () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, { ...policyDocument('5'), availability: 'draining' })
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
        store.adopt(PEER, snapshot('5'))

        expect(entry().baseline?.availability).toBe('draining')
        await store.save(PEER, PEER)
        expect(set).not.toHaveBeenCalled()
        expect(entry().phase).toBe('vetoed')
    })

    it('allows a newer saved available revision to clear old draining readback', async () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, { ...policyDocument('5'), availability: 'draining' })
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
        store.adopt(PEER, snapshot('6'))
        expect(entry().baseline?.availability).toBeUndefined()
        expect(set).not.toHaveBeenCalled()

        set.mockResolvedValue(snapshot('7'))
        await store.save(PEER, PEER)
        expect(set).toHaveBeenCalledTimes(1)
        expect(set.mock.calls[0][0].expectedPersistedRevision).toBe('6')
    })

    it('replaces only roles and keeps every other field of the target document', () => {
        const text = withRoles(policyText('5', { roles: [{ name: 'weak', profiles: ['a'] }] }), [
            { name: 'strong', profiles: ['gemma-3-27b'] }
        ])
        const sent = JSON.parse(text)
        const original = JSON.parse(policyText('5'))
        expect(sent).toEqual({
            ...original,
            roles: [{ name: 'strong', profiles: ['gemma-3-27b'] }]
        })
    })

    it('sends the substituted document with the target revision to a peer', async () => {
        set.mockResolvedValue(
            snapshot('6', { roles: [{ name: 'medium', profiles: ['qwen-fast'] }] })
        )
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.addRole(PEER, 'medium')
        store.setRole(PEER, 'medium', 'qwen-fast')

        const saved = await store.save(PEER, PEER)

        expect(set).toHaveBeenCalledTimes(1)
        const request = set.mock.calls[0][0]
        expect(request.nodeId).toBe(PEER)
        expect(request.expectedPersistedRevision).toBe('5')
        expect(JSON.parse(request.policy)).toEqual({
            ...JSON.parse(policyText('5')),
            roles: [{ name: 'medium', profiles: ['qwen-fast'] }]
        })
        expect(saved?.revision).toBe('6')
        expect(entry().phase).toBe('applied')
        expect(entry().baseline?.revision).toBe('6')
    })

    it('omits the node id for this PC', async () => {
        set.mockResolvedValue(snapshot('2'))
        const store = useNodeConfigurationStore.getState()
        store.adopt(SELF, snapshot('1'))
        store.addRole(SELF, 'weak')
        store.setRole(SELF, 'weak', 'qwen-fast')
        await store.save(SELF, undefined)
        expect(set.mock.calls[0][0].nodeId).toBeUndefined()
        expect(set.mock.calls[0][0].expectedPersistedRevision).toBe('1')
    })
})

describe('empty and arbitrary roles', () => {
    it('reads a document without roles as declaring nothing', () => {
        expect(policyRoles(policyText('1'))).toEqual([])
    })

    it('keeps an emptied role in its saved position in the document sent', async () => {
        set.mockResolvedValue(snapshot('4'))
        const store = useNodeConfigurationStore.getState()
        store.adopt(
            PEER,
            snapshot('3', {
                roles: [
                    { name: 'weak', profiles: ['qwen-fast'] },
                    { name: 'strong', profiles: ['gemma-3-27b'] }
                ]
            })
        )
        store.setRole(PEER, 'weak', undefined)
        await store.save(PEER, PEER)
        expect(JSON.parse(set.mock.calls[0][0].policy).roles).toEqual([
            { name: 'weak', profiles: [] },
            { name: 'strong', profiles: ['gemma-3-27b'] }
        ])
    })

    it('keeps arbitrary names in declaration order and empty roles after filtering invalid keys', () => {
        expect(
            policyRoles(
                JSON.stringify({
                    roles: [
                        { name: 'weak', profiles: [''] },
                        { name: 'tiny', profiles: ['x'] },
                        { name: 'Invalid', profiles: ['x'] },
                        { name: 'medium', profiles: ['y'] }
                    ]
                })
            )
        ).toEqual([
            { name: 'weak', profiles: [] },
            { name: 'tiny', profiles: ['x'] },
            { name: 'medium', profiles: ['y'] }
        ])
    })
})

describe('conflicts', () => {
    it.each([
        '-32000: policy conflict: the persisted revision is 6, not the expected 5',
        '-32010: the persisted revision is 6, not the expected 5'
    ])('reads the target back and shows the diff without retrying: %s', async message => {
        set.mockRejectedValue(new Error(message))
        get.mockResolvedValue(
            policyDocument('6', { roles: [{ name: 'weak', profiles: ['gemma-3-27b'] }] })
        )
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')

        expect(await store.save(PEER, PEER)).toBeNull()

        expect(set).toHaveBeenCalledTimes(1)
        expect(get).toHaveBeenCalledTimes(1)
        expect(get).toHaveBeenCalledWith(PEER)
        const { phase, latest, draft } = entry()
        expect(phase).toBe('conflict')
        expect(latest?.revision).toBe('6')
        expect(roleDifferences(draft, latest?.roles ?? [])).toEqual([
            {
                tier: 'weak',
                mine: ['qwen-fast'],
                theirs: ['gemma-3-27b']
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
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
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
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
        store.adopt(PEER, snapshot('6', { roles: [{ name: 'weak', profiles: ['z'] }] }))
        expect(entry().phase).toBe('conflict')
    })

    it('rebases a draft when a newer document leaves the roles alone', () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
        store.adopt(PEER, snapshot('6', { availability: 'paused' }))
        expect(entry().phase).toBe('draft')
        expect(entry().baseline?.revision).toBe('6')
        expect(entry().draft.find(role => role.name === 'weak')?.profiles[0]).toBe('qwen-fast')
    })

    it('shows a local veto and reads nothing back', async () => {
        set.mockRejectedValue(new Error('-32011: this node was paused on the node itself'))
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
        await store.save(PEER, PEER)
        expect(entry().phase).toBe('vetoed')
        expect(get).not.toHaveBeenCalled()
    })

    it('reports an unconfirmed outcome when the read back fails', async () => {
        set.mockRejectedValue(new Error('connection lost'))
        get.mockRejectedValue(new Error('connection lost'))
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
        await store.save(PEER, PEER)
        expect(entry().phase).toBe('unknown')
        expect(set).toHaveBeenCalledTimes(1)
    })

    it('classifies a refusal by its JSON-RPC code, not by its text', () => {
        expect(policyRefusal('device is offline -32010', REMOTE_SETTING_LOCKED_CODE)).toBe(
            'rejected'
        )
        expect(policyRefusal('-32014: locked profile')).toBe('other')
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
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
        await store.save(PEER, PEER)
        expect(entry().phase).toBe('vetoed')
        expect(get).not.toHaveBeenCalled()
    })

    it('does not read a code out of the text of a coded failure', async () => {
        set.mockRejectedValue(new ServiceCallError('-32011: the policy could not be saved', -32000))
        get.mockResolvedValue(policyDocument('5'))
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
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
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
        await store.save(PEER, PEER)
        expect(entry().phase).toBe('unreachable')
        expect(entry().draft.find(role => role.name === 'weak')?.profiles[0]).toBe('qwen-fast')
        expect(get).not.toHaveBeenCalled()
    })

    it('keeps a permission-refused draft rejected independently of local veto and misleading transport text', async () => {
        set.mockRejectedValue(
            new ServiceCallError('device is offline -32011', REMOTE_MANAGEMENT_DENIED_CODE)
        )
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5'))
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
        await store.save(PEER, PEER)
        expect(entry().phase).toBe('rejected')
        expect(entry().message).toContain('Remote management permission')
        expect(entry().draft.find(role => role.name === 'weak')?.profiles).toEqual(['qwen-fast'])
        expect(entry().baseline?.revision).toBe('5')
        expect(get).not.toHaveBeenCalled()
        expect(set).toHaveBeenCalledTimes(1)
    })

    it('keeps a whole-policy lock refusal definite and preserves the role draft and settings baseline', async () => {
        set.mockRejectedValue(
            new ServiceCallError('device is offline -32010', REMOTE_SETTING_LOCKED_CODE)
        )
        const profiles = [
            {
                engine: 'llamacpp-custom',
                model: 'm',
                localOnly: true,
                requestOptionsMode: 'enforced',
                requestOptions: { temperature: 1 }
            }
        ]
        const store = useNodeConfigurationStore.getState()
        store.adopt(PEER, snapshot('5', { profiles }))
        store.addRole(PEER, 'weak')
        store.setRole(PEER, 'weak', 'qwen-fast')
        await store.save(PEER, PEER)
        expect(entry().phase).toBe('rejected')
        expect(entry().draft.find(role => role.name === 'weak')?.profiles).toEqual(['qwen-fast'])
        expect(entry().baseline?.policy).toBe(policyText('5', { profiles }))
        expect(entry().baseline?.revision).toBe('5')
        expect(get).not.toHaveBeenCalled()
        expect(set).toHaveBeenCalledTimes(1)
    })
})

describe('inventory choices', () => {
    it('suggests only target declarations in their order and preserves empty memberships', () => {
        const choices = inventoryChoices(inventory, roleProfiles)
        const roles = [
            { name: 'vision', profiles: ['gemma-3-27b'] },
            { name: 'coder', profiles: [] },
            { name: 'chat', profiles: ['qwen-fast'] }
        ] satisfies NodeRoles
        const policy = policyText('1', { roles })
        const suggested = suggestRoles(choices, [policy])
        expect(suggested).toEqual([
            { name: 'vision', profiles: ['gemma-3-27b'] },
            { name: 'coder', profiles: [] },
            { name: 'chat', profiles: ['qwen-fast'] }
        ])
        suggested[0].profiles.push('draft-only')
        expect(JSON.parse(policy).roles).toEqual(roles)
    })

    it('offers only installed models of that node, in engine-manager spelling', () => {
        expect(
            inventoryChoices(inventory, roleProfiles).map(({ engine, model }) => ({
                engine,
                model
            }))
        ).toEqual([
            { engine: 'llamacpp-custom', model: 'gemma-3-27b' },
            { engine: 'llamacpp', model: 'qwen3:4b' }
        ])
    })

    it('does not create roles from retired local tier entries or API entries', () => {
        const choices = inventoryChoices(inventory, roleProfiles)
        const policy = policyText('1', {
            tiers: {
                coder: [{ kind: 'local', engine: 'llamacpp', model: 'qwen3:4b' }],
                vision: [{ kind: 'api', deployment: 'cloud' }]
            }
        })
        expect(suggestRoles(choices, [policy])).toEqual([])
    })

    it("does not import another node's roles into the target suggestions", () => {
        const choices = inventoryChoices(inventory, roleProfiles)
        const target = policyText('1', { roles: [{ name: 'coder', profiles: [] }] })
        const own = policyText('2', {
            roles: [{ name: 'vision', profiles: ['gemma-3-27b'] }]
        })
        expect(suggestRoles(choices, [target, own])).toEqual([{ name: 'coder', profiles: [] }])
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
                roles: [{ name: 'strong', profiles: ['gemma-3-27b'] }],
                profiles: [{ ...roleProfiles[1], contextTokens: 8192 }],
                pausedBy: 'local',
                pauseMode: 'immediate'
            }
        }
        state.handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: { nodes: [brokerNode] }
        })
        state.handleNotification({
            source: 'llamacpp-proxy',
            method: 'node/discovered',
            params: { id: 'uuid-roles-1', host: 'roles-host-1', port: 8080, ip: '192.0.2.41' }
        })
        const available = () => state.getAvailableNodes().find(node => node.id === 'uuid-roles-1')
        expect(available()?.policy).toEqual({
            revision: '9',
            roles: [{ name: 'strong', profiles: ['gemma-3-27b'] }],
            profiles: [{ ...roleProfiles[1], contextTokens: 8192 }],
            pausedBy: 'local',
            pauseMode: 'immediate'
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
            vi.stubGlobal('CSS', { supports: () => false, escape: (value: string) => value })
            for (const method of ['show', 'showModal']) {
                Object.defineProperty(HTMLDialogElement.prototype, method, {
                    configurable: true,
                    value: function (this: HTMLDialogElement) {
                        this.open = true
                    }
                })
            }
            Object.defineProperty(HTMLDialogElement.prototype, 'close', {
                configurable: true,
                value: function (this: HTMLDialogElement) {
                    this.open = false
                }
            })
            useConnectionStore.setState({ connected: true })
            useNodeConfigurationStore.setState({ management: new Map() })
            useEngineStatusStore.setState({
                descriptors: new Map([[customEngine, customEngineDescriptor]])
            })
        })
        afterEach(async () => {
            await act(async () => root.unmount())
            container.remove()
            useDiscoveredNodesStore.setState({ nodes: [] })
            useEngineModelsStore.setState({ models: new Map() })
            useEngineStatusStore.setState({ descriptors: new Map() })
            useConnectionStore.setState({ connected: false })
            vi.unstubAllGlobals()
        })

        const discovered = (roles: NodeRoles | null, profiles = roleProfiles) =>
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
                        ...(roles ? { policy: { revision: '3', roles, profiles } } : {})
                    }
                ]
            })

        function button(label: string, scope: ParentNode = container) {
            const found = Array.from(scope.querySelectorAll('button')).find(
                item => item.getAttribute('aria-label') === label || item.textContent === label
            )
            if (!found) throw new Error(`Missing button: ${label}`)
            return found
        }

        async function click(label: string, scope: ParentNode = container) {
            await act(async () => button(label, scope).click())
        }

        function nameForm(label: string) {
            const found = Array.from(container.querySelectorAll('form')).find(
                item => item.getAttribute('aria-label') === label
            )
            if (!found) throw new Error(`Missing form: ${label}`)
            return found
        }

        async function inputName(label: string, value: string) {
            const fieldLabel = Array.from(container.querySelectorAll('label')).find(
                item => item.textContent?.trim() === label
            )
            const field = fieldLabel && document.getElementById(fieldLabel.htmlFor)
            const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
            if (!(field instanceof HTMLInputElement) || !setter)
                throw new Error(`Missing role name input: ${label}`)
            await act(async () => {
                setter.call(field, value)
                field.dispatchEvent(new Event('input', { bubbles: true }))
            })
        }

        function confirmation() {
            const found = document.querySelector<HTMLDialogElement>('dialog[open]')
            if (!found) throw new Error('Missing role confirmation dialog')
            return found
        }

        async function renderEditor(targetId = SELF, nodeId?: string) {
            const onSaved = vi.fn<(saved: NodePolicySnapshot) => void>()
            await act(async () =>
                root.render(createElement(RoleEditor, { targetId, nodeId, onSaved }))
            )
            return onSaved
        }

        function roleApiPolicy() {
            return {
                roles: [
                    { name: 'chat', profiles: ['qwen-fast'] },
                    { name: 'coder', profiles: ['gemma-3-27b'] },
                    { name: 'vision', profiles: [] }
                ],
                tiers: {
                    coder: [{ kind: 'api', deployment: 'cloud' }],
                    vision: [{ kind: 'api', deployment: 'spare' }]
                },
                routing: {
                    accounts: [
                        {
                            id: 'account',
                            credential: 'NVPAIR_PROVIDER_SECRET_FIXTURE',
                            billing: 'free',
                            perRequestCeiling: '0',
                            dailyCeiling: '0',
                            monthlyCeiling: '0'
                        }
                    ],
                    deployments: [
                        {
                            id: 'cloud',
                            account: 'account',
                            model: 'provider-model:v2',
                            family: 'primary',
                            contextTokens: 8192,
                            maxOutputTokens: 1024,
                            enabled: false
                        },
                        {
                            id: 'spare',
                            account: 'account',
                            model: 'provider-vision',
                            family: 'vision',
                            contextTokens: 4096,
                            maxOutputTokens: 512,
                            enabled: false
                        }
                    ],
                    tiers: {
                        coder: {
                            bareMode: 'local',
                            remoteAllowed: false,
                            paidAllowed: false,
                            allowedFamilies: ['primary']
                        },
                        vision: {
                            bareMode: 'hybrid',
                            remoteAllowed: false,
                            paidAllowed: false,
                            allowedFamilies: ['vision']
                        }
                    }
                }
            }
        }

        it('starts with no roles and explains role order and the disabled API boundary', async () => {
            useNodeConfigurationStore.getState().adopt(SELF, snapshot('5'))
            await renderEditor()

            expect(container.textContent).toContain('No roles declared')
            expect(container.querySelectorAll('[role="group"]')).toHaveLength(0)
            expect(button('Save roles').disabled).toBe(true)
            expect(container.textContent).toContain(
                'Roles, weakest first. Allow stronger tries roles further down this list; Allow weaker tries roles further up.'
            )
            expect(container.textContent).toContain(
                'An API route needs an API entry for the role and a routing rule that allows remote execution.'
            )
            expect(container.textContent).toContain(
                'API overrides are listed, but public API dispatch is disabled in this build and requests are refused.'
            )
            expect(container.textContent).toContain(
                'Enabled deployments and remote or paid permissions cannot be saved.'
            )
            expect(set).not.toHaveBeenCalled()
        })

        it('holds visible mutation controls when a role save is unconfirmed', async () => {
            useNodeConfigurationStore.getState().adopt(SELF, snapshot('5', roleApiPolicy()))
            set.mockRejectedValueOnce(new Error('connection lost'))
            get.mockRejectedValueOnce(new Error('connection lost'))
            await renderEditor()
            await click('Move role coder up')
            await click('Save roles')
            expect(entry(SELF).phase).toBe('unknown')
            const draft = entry(SELF).draft
            for (const label of [
                'Add role',
                'Rename role coder',
                'Remove role coder',
                'Save roles'
            ]) {
                expect(button(label).disabled).toBe(true)
                await click(label)
            }
            expect(container.querySelector('form')).toBeNull()
            expect(document.querySelector('dialog[open]')).toBeNull()
            expect(entry(SELF).draft).toEqual(draft)
            expect(button('Discard role changes').disabled).toBe(false)
            expect(set).toHaveBeenCalledTimes(1)
            expect(get).toHaveBeenCalledTimes(1)
        })

        it.each([
            { value: '', error: 'Role name must match' },
            { value: 'Coder', error: 'Role name must match' },
            { value: 'co:der', error: 'Role name must match' },
            { value: 'x'.repeat(65), error: 'Role name must match' },
            { value: 'chat', error: 'Role chat already exists on this node.' },
            {
                value: 'qwen-fast',
                error: 'Role qwen-fast cannot have the same name as a profile key on this node.'
            }
        ])('rejects an invalid new role live: $value', async ({ value, error }) => {
            const roles = [{ name: 'chat', profiles: [] }]
            useNodeConfigurationStore.getState().adopt(SELF, snapshot('5', { roles }))
            await renderEditor()
            await click('Add role')
            await inputName('New role name', value)

            const form = nameForm('Add role')
            expect(form.textContent).toContain(error)
            expect(button('Add role', form).disabled).toBe(true)
            await click('Add role', form)
            expect(entry(SELF).draft).toEqual([{ name: 'chat', profiles: [] }])
            expect(set).not.toHaveBeenCalled()
        })

        it('appends a valid role after live correction and saves its empty membership list', async () => {
            const store = useNodeConfigurationStore.getState()
            store.adopt(SELF, snapshot('5', { roles: [{ name: 'chat', profiles: ['qwen-fast'] }] }))
            const saved = snapshot('6', {
                roles: [
                    { name: 'chat', profiles: ['qwen-fast'] },
                    { name: 'coder', profiles: [] }
                ]
            })
            set.mockResolvedValue(saved)
            const onSaved = await renderEditor()
            await click('Add role')
            await inputName('New role name', 'Coder')
            expect(button('Add role', nameForm('Add role')).disabled).toBe(true)
            await inputName('New role name', 'coder')
            expect(nameForm('Add role').textContent).not.toContain('Role name must match')
            expect(button('Add role', nameForm('Add role')).disabled).toBe(false)
            await click('Add role', nameForm('Add role'))

            expect(container.querySelector('[aria-label="coder role"]')?.textContent).toContain(
                'No profiles assigned'
            )
            expect(set).not.toHaveBeenCalled()
            await click('Save roles')
            expect(set).toHaveBeenCalledTimes(1)
            expect(set.mock.calls[0][0]).toEqual({
                nodeId: undefined,
                expectedPersistedRevision: '5',
                policy: expect.any(String)
            })
            expect(JSON.parse(set.mock.calls[0][0].policy)).toEqual({
                ...JSON.parse(policyText('5')),
                roles: [
                    { name: 'chat', profiles: ['qwen-fast'] },
                    { name: 'coder', profiles: [] }
                ]
            })
            expect(onSaved).toHaveBeenCalledTimes(1)
            expect(onSaved).toHaveBeenCalledWith(saved)
            expect(container.textContent).toContain('Saved at revision 6')
        })

        it.each([
            { value: '', error: 'Role name must match' },
            { value: 'Code', error: 'Role name must match' },
            { value: 'co:de', error: 'Role name must match' },
            { value: 'x'.repeat(65), error: 'Role name must match' },
            { value: 'chat', error: 'Role chat already exists on this node.' },
            {
                value: 'qwen-fast',
                error: 'Role qwen-fast cannot have the same name as a profile key on this node.'
            }
        ])('rejects an invalid rename live: $value', async ({ value, error }) => {
            const roles = [
                { name: 'chat', profiles: [] },
                { name: 'coder', profiles: ['gemma-3-27b'] }
            ]
            useNodeConfigurationStore.getState().adopt(SELF, snapshot('5', { roles }))
            await renderEditor()
            await click('Rename role coder')
            await inputName('New name for role coder', value)

            const form = nameForm('Rename role coder')
            expect(form.textContent).toContain(error)
            expect(button('Rename role', form).disabled).toBe(true)
            await click('Rename role', form)
            expect(document.querySelector('dialog[open]')).toBeNull()
            expect(entry(SELF).draft).toEqual(roles)
            expect(set).not.toHaveBeenCalled()
        })

        it('renames in place after confirmation and preserves every membership and unrelated field', async () => {
            const roles = [
                { name: 'chat', profiles: [] },
                { name: 'coder', profiles: ['gemma-3-27b', 'qwen-fast'] },
                { name: 'vision', profiles: [] }
            ]
            useNodeConfigurationStore.getState().adopt(SELF, snapshot('5', { roles }))
            const renamed = [
                { name: 'chat', profiles: [] },
                { name: 'code', profiles: ['gemma-3-27b', 'qwen-fast'] },
                { name: 'vision', profiles: [] }
            ]
            set.mockResolvedValue(snapshot('6', { roles: renamed }))
            await renderEditor()
            await click('Rename role coder')
            await inputName('New name for role coder', 'Code')
            expect(button('Rename role', nameForm('Rename role coder')).disabled).toBe(true)
            await inputName('New name for role coder', 'code')
            expect(button('Rename role', nameForm('Rename role coder')).disabled).toBe(false)
            await click('Rename role', nameForm('Rename role coder'))
            expect(confirmation().textContent).toContain(
                'Rename role coder to code? Its position and profile memberships will be kept.'
            )
            expect(entry(SELF).draft).toEqual(roles)
            expect(set).not.toHaveBeenCalled()
            await click('Rename role', confirmation())
            expect(entry(SELF).draft).toEqual(renamed)
            expect(set).not.toHaveBeenCalled()
            await click('Save roles')

            expect(set).toHaveBeenCalledTimes(1)
            expect(set.mock.calls[0][0].expectedPersistedRevision).toBe('5')
            expect(JSON.parse(set.mock.calls[0][0].policy)).toEqual({
                ...JSON.parse(policyText('5')),
                roles: renamed
            })
        })

        it('names a removable role in its confirmation and saves no roles after removing the last one', async () => {
            const roles = [{ name: 'weak', profiles: ['qwen-fast'] }]
            useNodeConfigurationStore.getState().adopt(SELF, snapshot('5', { roles }))
            set.mockResolvedValue(snapshot('6', { roles: [] }))
            await renderEditor()
            await click('Remove role weak')
            expect(confirmation().textContent).toContain('Remove role weak?')
            expect(confirmation().textContent).toContain(
                'Remove role weak and its profile memberships? The profiles will be kept.'
            )
            await click('Cancel', confirmation())
            expect(entry(SELF).draft).toEqual(roles)
            expect(set).not.toHaveBeenCalled()
            await click('Remove role weak')
            await click('Remove role', confirmation())
            expect(container.textContent).toContain('No roles declared')
            expect(set).not.toHaveBeenCalled()
            await click('Save roles')

            expect(set).toHaveBeenCalledTimes(1)
            expect(set.mock.calls[0][0].expectedPersistedRevision).toBe('5')
            expect(JSON.parse(set.mock.calls[0][0].policy)).toEqual({
                ...JSON.parse(policyText('5')),
                roles: []
            })
        })

        it('moves whole roles up and down and saves their chosen order', async () => {
            useNodeConfigurationStore.getState().adopt(
                SELF,
                snapshot('5', {
                    roles: [
                        { name: 'chat', profiles: ['qwen-fast'] },
                        { name: 'coder', profiles: ['gemma-3-27b'] },
                        { name: 'vision', profiles: [] }
                    ]
                })
            )
            const reordered = [
                { name: 'coder', profiles: ['gemma-3-27b'] },
                { name: 'vision', profiles: [] },
                { name: 'chat', profiles: ['qwen-fast'] }
            ]
            set.mockResolvedValue(snapshot('6', { roles: reordered }))
            await renderEditor()
            expect(button('Move role chat up').disabled).toBe(true)
            expect(button('Move role vision down').disabled).toBe(true)
            await click('Move role chat down')
            await click('Move role vision up')
            expect(
                Array.from(container.querySelectorAll('[role="group"]'), group =>
                    group.getAttribute('aria-label')
                )
            ).toEqual(['coder role', 'vision role', 'chat role'])
            expect(set).not.toHaveBeenCalled()
            await click('Save roles')
            expect(set).toHaveBeenCalledTimes(1)
            expect(set.mock.calls[0][0].expectedPersistedRevision).toBe('5')
            expect(JSON.parse(set.mock.calls[0][0].policy)).toEqual({
                ...JSON.parse(policyText('5')),
                roles: reordered
            })
        })

        it('warns about a visible peer profile key without blocking the role save', async () => {
            discovered(
                [],
                [
                    {
                        key: 'coder',
                        displayName: 'Remote coding profile',
                        engine: 'llamacpp',
                        model: 'remote'
                    }
                ]
            )
            useNodeConfigurationStore.getState().adopt(SELF, snapshot('5'))
            set.mockResolvedValue(snapshot('6', { roles: [{ name: 'coder', profiles: [] }] }))
            await renderEditor()
            await click('Add role')
            await inputName('New role name', 'coder')
            expect(nameForm('Add role').textContent).toContain(
                'Role coder hides the profile coder on peer.'
            )
            expect(button('Add role', nameForm('Add role')).disabled).toBe(false)
            await click('Add role', nameForm('Add role'))
            expect(container.textContent).toContain('Role coder hides the profile coder on peer.')
            await click('Save roles')
            expect(set).toHaveBeenCalledTimes(1)
            expect(set.mock.calls[0][0].expectedPersistedRevision).toBe('5')
            expect(JSON.parse(set.mock.calls[0][0].policy).roles).toEqual([
                { name: 'coder', profiles: [] }
            ])
        })

        it('confirms the API move and renames both API maps in one revision-checked policy save', async () => {
            const fields = roleApiPolicy()
            useNodeConfigurationStore.getState().adopt(SELF, snapshot('5', fields))
            const expected = {
                ...JSON.parse(policyText('5', fields)),
                roles: [
                    { name: 'chat', profiles: ['qwen-fast'] },
                    { name: 'code', profiles: ['gemma-3-27b'] },
                    { name: 'vision', profiles: [] }
                ],
                tiers: {
                    code: [{ kind: 'api', deployment: 'cloud' }],
                    vision: [{ kind: 'api', deployment: 'spare' }]
                },
                routing: {
                    ...fields.routing,
                    tiers: {
                        code: {
                            bareMode: 'local',
                            remoteAllowed: false,
                            paidAllowed: false,
                            allowedFamilies: ['primary']
                        },
                        vision: fields.routing.tiers.vision
                    }
                }
            }
            set.mockResolvedValue({
                policy: JSON.stringify({ ...expected, revision: '6' }),
                revision: '6',
                activation: null
            })
            await renderEditor()
            await click('Rename role coder')
            await inputName('New name for role coder', 'code')
            await click('Rename role', nameForm('Rename role coder'))
            expect(confirmation().textContent).toContain(
                "Rename role coder to code? Its position and profile memberships will be kept. Also moves this node's API entry for coder."
            )
            expect(set).not.toHaveBeenCalled()
            await click('Rename role', confirmation())
            expect(set).not.toHaveBeenCalled()
            expect(entry(SELF).baseline?.policy).toBe(policyText('5', fields))
            await click('Save roles')
            expect(set).toHaveBeenCalledTimes(1)
            expect(set.mock.calls[0][0].expectedPersistedRevision).toBe('5')
            expect(JSON.parse(set.mock.calls[0][0].policy)).toEqual(expected)
            expect(get).not.toHaveBeenCalled()
        })

        it('confirms the API deletion and removes both API maps in one revision-checked policy save', async () => {
            const fields = roleApiPolicy()
            useNodeConfigurationStore.getState().adopt(SELF, snapshot('5', fields))
            const expected = {
                ...JSON.parse(policyText('5', fields)),
                roles: [
                    { name: 'chat', profiles: ['qwen-fast'] },
                    { name: 'vision', profiles: [] }
                ],
                tiers: { vision: [{ kind: 'api', deployment: 'spare' }] },
                routing: { ...fields.routing, tiers: { vision: fields.routing.tiers.vision } }
            }
            set.mockResolvedValue({
                policy: JSON.stringify({ ...expected, revision: '6' }),
                revision: '6',
                activation: null
            })
            await renderEditor()
            await click('Remove role coder')
            expect(confirmation().textContent).toContain(
                "Remove role coder and its profile memberships? The profiles will be kept. Also deletes this node's API entry for coder."
            )
            await click('Cancel', confirmation())
            expect(entry(SELF).draft).toEqual(fields.roles)
            expect(entry(SELF).baseline?.policy).toBe(policyText('5', fields))
            expect(set).not.toHaveBeenCalled()
            await click('Remove role coder')
            await click('Remove role', confirmation())
            expect(set).not.toHaveBeenCalled()
            await click('Save roles')
            expect(set).toHaveBeenCalledTimes(1)
            expect(set.mock.calls[0][0].expectedPersistedRevision).toBe('5')
            expect(JSON.parse(set.mock.calls[0][0].policy)).toEqual(expected)
            expect(get).not.toHaveBeenCalled()
        })

        it.each([
            { name: 'deployment entries on both names', source: 'tiers', destination: 'tiers' },
            {
                name: 'source deployment and destination routing',
                source: 'tiers',
                destination: 'routing'
            },
            {
                name: 'source routing and destination deployment',
                source: 'routing',
                destination: 'tiers'
            }
        ])('refuses a local API collision live: $name', async ({ source, destination }) => {
            const fields = roleApiPolicy()
            const tiers: Record<string, JsonValue> = {}
            const routingTiers: Record<string, JsonValue> = {}
            if (source === 'tiers') tiers.coder = fields.tiers.coder
            else routingTiers.coder = fields.routing.tiers.coder
            if (destination === 'tiers') tiers.code = fields.tiers.vision
            else routingTiers.code = fields.routing.tiers.vision
            const policy = { ...fields, tiers, routing: { ...fields.routing, tiers: routingTiers } }
            useNodeConfigurationStore.getState().adopt(SELF, snapshot('5', policy))
            await renderEditor()
            await click('Rename role coder')
            await inputName('New name for role coder', 'code')

            const form = nameForm('Rename role coder')
            expect(form.textContent).toMatch(/\bcoder\b/)
            expect(form.textContent).toMatch(/\bcode\b/)
            expect(form.textContent).toMatch(/another name/i)
            expect(button('Rename role', form).disabled).toBe(true)
            expect(document.querySelector('dialog[open]')).toBeNull()
            expect(entry(SELF).draft).toEqual(fields.roles)
            expect(entry(SELF).baseline?.policy).toBe(policyText('5', policy))
            expect(set).not.toHaveBeenCalled()
        })

        it.each([
            { name: 'empty source', sourcePresent: false, destinationPresent: false },
            {
                name: 'empty destination permits source move',
                sourcePresent: true,
                destinationPresent: false
            },
            {
                name: 'empty source retains destination',
                sourcePresent: false,
                destinationPresent: true
            }
        ])(
            'renames with empty API tier keys: $name',
            async ({ sourcePresent, destinationPresent }) => {
                const fields = roleApiPolicy()
                const policy = {
                    ...fields,
                    tiers: {
                        coder: sourcePresent ? fields.tiers.coder : [],
                        code: destinationPresent ? fields.tiers.vision : []
                    },
                    routing: { ...fields.routing, tiers: {} }
                }
                const expected = {
                    ...JSON.parse(policyText('5', policy)),
                    roles: [
                        { name: 'chat', profiles: ['qwen-fast'] },
                        { name: 'code', profiles: ['gemma-3-27b'] },
                        { name: 'vision', profiles: [] }
                    ],
                    tiers: {
                        code: sourcePresent
                            ? fields.tiers.coder
                            : destinationPresent
                              ? fields.tiers.vision
                              : []
                    }
                }
                useNodeConfigurationStore.getState().adopt(SELF, snapshot('5', policy))
                set.mockResolvedValue({
                    policy: JSON.stringify({ ...expected, revision: '6' }),
                    revision: '6',
                    activation: null
                })
                await renderEditor()
                await click('Rename role coder')
                await inputName('New name for role coder', 'code')
                expect(button('Rename role', nameForm('Rename role coder')).disabled).toBe(false)
                await click('Rename role', nameForm('Rename role coder'))
                if (sourcePresent)
                    expect(confirmation().textContent).toContain(
                        "Also moves this node's API entry for coder."
                    )
                else expect(confirmation().textContent).not.toContain('Also moves')
                await click('Rename role', confirmation())
                expect(set).not.toHaveBeenCalled()
                await click('Save roles')
                expect(set).toHaveBeenCalledTimes(1)
                expect(set.mock.calls[0][0].expectedPersistedRevision).toBe('5')
                expect(JSON.parse(set.mock.calls[0][0].policy)).toEqual(expected)
            }
        )

        it('does not describe an empty source tier key as an API deletion', async () => {
            useNodeConfigurationStore.getState().adopt(
                SELF,
                snapshot('5', {
                    roles: [{ name: 'coder', profiles: [] }],
                    tiers: { coder: [] }
                })
            )
            await renderEditor()
            await click('Remove role coder')
            expect(confirmation().textContent).toContain(
                'Remove role coder and its profile memberships?'
            )
            expect(confirmation().textContent).not.toContain('Also deletes')
            await click('Cancel', confirmation())
            expect(entry(SELF).draft).toEqual([{ name: 'coder', profiles: [] }])
            expect(set).not.toHaveBeenCalled()
        })

        it.each([
            { name: 'rename without private maps', rename: true, privateMaps: false },
            { name: 'rename with private maps', rename: true, privateMaps: true },
            { name: 'remove without private maps', rename: false, privateMaps: false },
            { name: 'remove with private maps', rename: false, privateMaps: true }
        ])(
            'keeps remote API configuration private and unchanged: $name',
            async ({ rename, privateMaps }) => {
                const fields = roleApiPolicy()
                const policy = privateMaps
                    ? {
                          ...fields,
                          tiers: { ...fields.tiers, code: fields.tiers.vision },
                          routing: {
                              ...fields.routing,
                              tiers: { ...fields.routing.tiers, code: fields.routing.tiers.vision }
                          }
                      }
                    : { roles: fields.roles }
                useNodeConfigurationStore.getState().adopt(PEER, snapshot('5', policy))
                discovered(fields.roles)
                useDiscoveredNodesStore.setState(state => ({
                    nodes: state.nodes.map(node => ({ ...node, name: 'Studio PC' }))
                }))
                const roles = rename
                    ? [
                          { name: 'chat', profiles: ['qwen-fast'] },
                          { name: 'code', profiles: ['gemma-3-27b'] },
                          { name: 'vision', profiles: [] }
                      ]
                    : [fields.roles[0], fields.roles[2]]
                const expected = { ...JSON.parse(policyText('5', policy)), roles }
                set.mockResolvedValue({
                    policy: JSON.stringify({ ...expected, revision: '6' }),
                    revision: '6',
                    activation: null
                })
                await renderEditor(PEER, PEER)
                await click(rename ? 'Rename role coder' : 'Remove role coder')
                if (rename) {
                    await inputName('New name for role coder', 'code')
                    await click('Rename role', nameForm('Rename role coder'))
                }
                expect(confirmation().textContent).toContain(
                    "API entries on Studio PC can't be seen from here and stay as they are."
                )
                expect(confirmation().textContent).not.toContain('Also moves')
                expect(confirmation().textContent).not.toContain('Also deletes')
                await click('Cancel', confirmation())
                expect(entry().draft).toEqual(fields.roles)
                expect(set).not.toHaveBeenCalled()
                await click(rename ? 'Rename role coder' : 'Remove role coder')
                if (rename) {
                    await inputName('New name for role coder', 'code')
                    await click('Rename role', nameForm('Rename role coder'))
                }
                await click(rename ? 'Rename role' : 'Remove role', confirmation())
                expect(set).not.toHaveBeenCalled()
                await click('Save roles')
                expect(set).toHaveBeenCalledTimes(1)
                expect(set.mock.calls[0][0]).toEqual({
                    nodeId: PEER,
                    expectedPersistedRevision: '5',
                    policy: expect.any(String)
                })
                expect(JSON.parse(set.mock.calls[0][0].policy)).toEqual(expected)
            }
        )

        it('keeps full accessible names and titles for roles differing only at byte 64', async () => {
            const first = `${'a'.repeat(63)}x`
            const second = `${'a'.repeat(63)}y`
            useNodeConfigurationStore.getState().adopt(
                SELF,
                snapshot('5', {
                    roles: [
                        { name: first, profiles: [] },
                        { name: second, profiles: [] }
                    ]
                })
            )
            await renderEditor()
            expect(
                Array.from(container.querySelectorAll('[role="group"]'), group => ({
                    name: group.getAttribute('aria-label'),
                    title: group.getAttribute('title')
                }))
            ).toEqual([
                { name: `${first} role`, title: first },
                { name: `${second} role`, title: second }
            ])
            expect(button(`Rename role ${first}`).title).toBe(`Rename role ${first}`)
            expect(button(`Rename role ${second}`).title).toBe(`Rename role ${second}`)
            await click(`Rename role ${second}`)
            expect(nameForm(`Rename role ${second}`).querySelector('input')?.value).toBe(second)
            expect(
                nameForm(`Rename role ${second}`).querySelector(`[title="${second}"]`)
            ).not.toBeNull()
            expect(set).not.toHaveBeenCalled()
        })

        it('keeps draft renames and removals when suggesting from the target declarations', async () => {
            useEngineModelsStore.setState({
                models: new Map([
                    [
                        `${SELF}:llama-cpp`,
                        { nodeId: SELF, engineType: 'llama-cpp', models: [model('qwen3:4b')] }
                    ]
                ])
            })
            useNodeConfigurationStore.getState().adopt(
                SELF,
                snapshot('5', {
                    roles: [
                        { name: 'coder', profiles: [] },
                        { name: 'vision', profiles: ['qwen-fast'] }
                    ]
                })
            )
            set.mockResolvedValue(
                snapshot('6', { roles: [{ name: 'chat', profiles: ['qwen-fast'] }] })
            )
            await renderEditor()
            await click('Remove role coder')
            await click('Remove role', confirmation())
            await click('Rename role vision')
            await inputName('New name for role vision', 'chat')
            await click('Rename role', nameForm('Rename role vision'))
            await click('Rename role', confirmation())
            expect(button('Suggest roles').disabled).toBe(false)
            await click('Suggest roles')
            expect(entry(SELF).draft).toEqual([{ name: 'chat', profiles: ['qwen-fast'] }])
            expect(container.querySelector('[aria-label="coder role"]')).toBeNull()
            expect(container.querySelector('[aria-label="vision role"]')).toBeNull()
            expect(set).not.toHaveBeenCalled()
            await click('Save roles')
            expect(set).toHaveBeenCalledTimes(1)
            expect(set.mock.calls[0][0].expectedPersistedRevision).toBe('5')
            expect(JSON.parse(set.mock.calls[0][0].policy).roles).toEqual([
                { name: 'chat', profiles: ['qwen-fast'] }
            ])
        })

        it('keeps role controls and a dirty save disabled when the peer denies management', async () => {
            management.mockResolvedValue({
                version: 1,
                allowed: false,
                state: 'denied',
                revision: '2'
            })
            const store = useNodeConfigurationStore.getState()
            store.adopt(PEER, snapshot('5', { roles: [{ name: 'coder', profiles: [] }] }))
            store.addRole(PEER, 'chat')
            await renderEditor(PEER, PEER)

            expect(management).toHaveBeenCalledWith(PEER)
            expect(button('Add role').disabled).toBe(true)
            expect(button('Rename role coder').disabled).toBe(true)
            expect(button('Remove role coder').disabled).toBe(true)
            expect(button('Move role coder down').disabled).toBe(true)
            expect(button('Save roles').disabled).toBe(true)
            await click('Add role')
            await click('Save roles')
            expect(container.querySelector('form')).toBeNull()
            expect(entry(PEER).draft).toEqual([
                { name: 'coder', profiles: [] },
                { name: 'chat', profiles: [] }
            ])
            expect(set).not.toHaveBeenCalled()
        })

        it('disables role changes and keeps the draft when the connection drops', async () => {
            const store = useNodeConfigurationStore.getState()
            store.adopt(SELF, snapshot('5', { roles: [{ name: 'coder', profiles: [] }] }))
            store.addRole(SELF, 'chat')
            await renderEditor()
            expect(button('Save roles').disabled).toBe(false)
            await act(async () => useConnectionStore.setState({ connected: false }))

            expect(button('Add role').disabled).toBe(true)
            expect(button('Rename role coder').disabled).toBe(true)
            expect(button('Remove role coder').disabled).toBe(true)
            expect(button('Move role coder down').disabled).toBe(true)
            expect(button('Save roles').disabled).toBe(true)
            await click('Save roles')
            expect(entry(SELF).draft).toEqual([
                { name: 'coder', profiles: [] },
                { name: 'chat', profiles: [] }
            ])
            expect(set).not.toHaveBeenCalled()
        })

        it('moves and removes profile memberships and saves their order with the loaded revision', async () => {
            vi.stubGlobal('CSS', { supports: () => true })
            useConnectionStore.setState({ connected: true })
            const store = useNodeConfigurationStore.getState()
            store.adopt(
                SELF,
                snapshot('5', {
                    roles: [
                        { name: 'strong', profiles: ['qwen-fast', 'gemma-3-27b'] },
                        { name: 'weak', profiles: [] }
                    ]
                })
            )
            set.mockResolvedValue(
                snapshot('6', {
                    roles: [
                        { name: 'strong', profiles: ['gemma-3-27b', 'qwen-fast'] },
                        { name: 'weak', profiles: [] }
                    ]
                })
            )
            await act(async () =>
                root.render(
                    createElement(RoleEditor, {
                        targetId: SELF,
                        nodeId: undefined,
                        onSaved: vi.fn()
                    })
                )
            )
            const click = async (label: string) => {
                const button = Array.from(container.querySelectorAll('button')).find(
                    button =>
                        button.getAttribute('aria-label') === label || button.textContent === label
                )
                if (!button) throw new Error(`Missing button: ${label}`)
                await act(async () => button.click())
            }
            expect(container.textContent).toContain('No profiles assigned')
            expect(
                container
                    .querySelector('[aria-label="Move Qwen up in strong"]')
                    ?.hasAttribute('disabled')
            ).toBe(true)
            await click('Move Gemma up in strong')
            expect(entry(SELF).draft).toEqual([
                { name: 'strong', profiles: ['gemma-3-27b', 'qwen-fast'] },
                { name: 'weak', profiles: [] }
            ])
            await click('Move Gemma down in strong')
            await click('Remove Qwen from strong')
            await click('Remove Gemma from strong')
            expect(entry(SELF).draft).toEqual([
                { name: 'strong', profiles: [] },
                { name: 'weak', profiles: [] }
            ])
            const add = async (name: string) => {
                const trigger = container.querySelector<HTMLButtonElement>(
                    '[aria-label="Add profile to strong"]'
                )
                if (!trigger) throw new Error('Missing Add profile selector')
                await act(async () => trigger.click())
                const option = Array.from(
                    document.querySelectorAll<HTMLElement>('[role="option"]')
                ).find(item => item.textContent?.startsWith(name))
                if (!option) throw new Error(`Missing profile option: ${name}`)
                await act(async () => option.click())
            }
            await add('Gemma')
            await add('Qwen')
            expect(container.textContent).toContain('Gemma')
            await click('Save roles')
            expect(set).toHaveBeenCalledTimes(1)
            const request = set.mock.calls[0][0]
            expect(request.expectedPersistedRevision).toBe('5')
            expect(JSON.parse(request.policy).roles).toEqual([
                { name: 'strong', profiles: ['gemma-3-27b', 'qwen-fast'] },
                { name: 'weak', profiles: [] }
            ])
            useConnectionStore.setState({ connected: false })
        })

        it('lists each declared tier', async () => {
            discovered([
                { name: 'weak', profiles: ['qwen-fast'] },
                { name: 'strong', profiles: ['gemma-3-27b'] }
            ])
            await act(async () => root.render(createElement(NodeDeclaredRoles, { nodeId: PEER })))
            expect(
                Array.from(container.querySelectorAll('button'), button =>
                    button.getAttribute('aria-label')
                )
            ).toEqual(['weak: qwen-fast', 'strong: gemma-3-27b'])
            expect(
                container.querySelectorAll('[title="Repository unknown: no source establishes it"]')
            ).toHaveLength(2)
            expect(container.textContent).not.toContain('medium')
        })

        it('labels roles from this node and engine, keeps declaration order and updates identity pushes', async () => {
            const served = 'preset-ctx131072'
            const peerModel: ModelItem = {
                ...model(served),
                identity: {
                    repository: 'unsloth/LFM2.5-24B-A2B-GGUF',
                    quantization: 'UD-Q4_K_XL',
                    quantizationSource: 'hf-tag',
                    strength: 'declared'
                }
            }
            useEngineModelsStore.setState({
                models: new Map<string, EngineModels>([
                    [
                        `${SELF}:llama-cpp`,
                        {
                            nodeId: SELF,
                            engineType: 'llama-cpp',
                            models: [
                                {
                                    ...peerModel,
                                    identity: { repository: 'wrong/node', strength: 'declared' }
                                }
                            ]
                        }
                    ],
                    [
                        `${PEER}:llama-cpp`,
                        { nodeId: PEER, engineType: 'llama-cpp', models: [peerModel] }
                    ],
                    [
                        `${PEER}:${customEngine}`,
                        {
                            nodeId: PEER,
                            engineType: customEngine,
                            models: [
                                {
                                    ...peerModel,
                                    identity: { repository: 'wrong/engine', strength: 'declared' }
                                }
                            ]
                        }
                    ]
                ])
            })
            discovered(
                [
                    { name: 'strong', profiles: [served] },
                    { name: 'weak', profiles: ['gemma-3-27b'] }
                ],
                [
                    ...roleProfiles,
                    { key: served, displayName: served, engine: 'llamacpp', model: served }
                ]
            )
            await act(async () => root.render(createElement(NodeDeclaredRoles, { nodeId: PEER })))
            const members = () => Array.from(container.querySelectorAll('button'))
            expect(members().map(member => member.getAttribute('aria-label'))).toEqual([
                'strong: preset-ctx131072',
                'weak: gemma-3-27b'
            ])
            expect(members()[0].textContent).toBe('strongunsloth/LFM2.5-24B-A2B-GGUFUD-Q4_K_XL')
            expect(members()[1].textContent).toBe('weakUnknownUnknown')
            expect(container.textContent).not.toContain('wrong/')
            await act(async () => {
                const models = new Map(useEngineModelsStore.getState().models)
                models.set(`${PEER}:llama-cpp`, {
                    nodeId: PEER,
                    engineType: 'llama-cpp',
                    models: [
                        {
                            ...peerModel,
                            identity: { repository: 'updated/model', strength: 'declared' }
                        }
                    ]
                })
                useEngineModelsStore.setState({ models })
            })
            expect(members()[0].textContent).toBe('strongupdated/modelUnknown')
        })

        it('reports a declaration-order conflict without claiming matching roles', async () => {
            vi.stubGlobal('CSS', { supports: () => true })
            const roles = [
                { name: 'strong', profiles: ['gemma-3-27b'] },
                { name: 'medium', profiles: [] }
            ] satisfies NodeRoles
            const store = useNodeConfigurationStore.getState()
            store.adopt(SELF, snapshot('5', { roles }))
            store.setDraft(SELF, [...roles].reverse())
            set.mockRejectedValue(new Error('-32000: policy conflict: stale'))
            get.mockResolvedValue(policyDocument('6', { roles }))
            await store.save(SELF, undefined)
            expect(entry(SELF).phase).toBe('conflict')
            await act(async () =>
                root.render(
                    createElement(RoleEditor, {
                        targetId: SELF,
                        nodeId: undefined,
                        onSaved: vi.fn()
                    })
                )
            )
            expect(container.textContent).toContain(
                'medium: yours empty, node empty; position yours 1, node 2'
            )
            expect(container.textContent).toContain('position yours 2, node 1')
            expect(container.textContent).not.toContain('Its roles match this draft')
        })

        it('distinguishes same-model siblings in a membership-order conflict', async () => {
            vi.stubGlobal('CSS', { supports: () => true })
            useEngineModelsStore.setState({
                models: new Map([
                    [
                        `${SELF}:llama-cpp`,
                        {
                            nodeId: SELF,
                            engineType: 'llama-cpp',
                            models: ['fast-8k', 'fast-32k'].map(key => ({
                                ...model(key),
                                identity: {
                                    repository: 'owner/same',
                                    quantization: 'Q4_K_M',
                                    strength: 'declared'
                                }
                            }))
                        }
                    ]
                ])
            })
            const profiles = ['fast-8k', 'fast-32k'].map(key => ({
                key,
                displayName: 'Same label',
                engine: 'llamacpp',
                model: key,
                modelIdentity: {
                    repository: 'owner/same',
                    quantization: 'Q4_K_M',
                    strength: 'declared'
                }
            }))
            const store = useNodeConfigurationStore.getState()
            store.adopt(
                SELF,
                snapshot('5', { profiles, roles: [{ name: 'strong', profiles: ['fast-8k'] }] })
            )
            store.setDraft(SELF, [{ name: 'strong', profiles: ['fast-8k', 'fast-32k'] }])
            set.mockRejectedValue(new Error('-32000: policy conflict: stale'))
            get.mockResolvedValue(
                policyDocument('6', {
                    profiles,
                    roles: [{ name: 'strong', profiles: ['fast-32k', 'fast-8k'] }]
                })
            )
            await store.save(SELF, undefined)
            expect(entry(SELF).phase).toBe('conflict')
            await act(async () =>
                root.render(
                    createElement(RoleEditor, {
                        targetId: SELF,
                        nodeId: undefined,
                        onSaved: vi.fn()
                    })
                )
            )
            expect(container.textContent).toContain('yours Same label (fast-8k)')
            expect(container.textContent).toContain('node Same label (fast-32k)')
        })

        it('uses the edited node inventory for its role select label', async () => {
            vi.stubGlobal('CSS', { supports: () => true })
            const served = 'preset-ctx131072'
            const configured: ModelItem = {
                ...model(served),
                identity: {
                    repository: 'unsloth/LFM2.5-24B-A2B-GGUF',
                    quantization: 'UD-Q4_K_XL',
                    quantizationSource: 'hf-tag',
                    strength: 'declared'
                }
            }
            useEngineModelsStore.setState({
                models: new Map<string, EngineModels>([
                    [
                        `${PEER}:llama-cpp`,
                        { nodeId: PEER, engineType: 'llama-cpp', models: [configured] }
                    ],
                    [
                        `${SELF}:llama-cpp`,
                        {
                            nodeId: SELF,
                            engineType: 'llama-cpp',
                            models: [
                                {
                                    ...configured,
                                    identity: { repository: 'other/node', strength: 'declared' }
                                }
                            ]
                        }
                    ]
                ])
            })
            useNodeConfigurationStore.getState().adopt(
                PEER,
                snapshot('5', {
                    roles: [{ name: 'strong', profiles: [served] }],
                    profiles: [
                        { key: served, displayName: served, engine: 'llamacpp', model: served }
                    ]
                })
            )
            await act(async () =>
                root.render(
                    createElement(RoleEditor, { targetId: PEER, nodeId: PEER, onSaved: vi.fn() })
                )
            )
            expect(container.textContent).toContain(served)
            expect(container.textContent).toContain('unsloth/LFM2.5-24B-A2B-GGUF')
            expect(container.textContent).toContain('UD-Q4_K_XL')
            expect(container.textContent).not.toContain('other/node')
        })

        it('shows nothing for a node that declares no role', async () => {
            discovered(null)
            await act(async () => root.render(createElement(NodeDeclaredRoles, { nodeId: PEER })))
            expect(container.textContent).toBe('')
            await act(async () => discovered([]))
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
            expect(container.textContent).toContain('A role can be empty.')

            await act(async () =>
                root.render(
                    createElement(RoleEditor, { targetId: SELF, nodeId: undefined, onSaved })
                )
            )
            expect(container.textContent).not.toContain(NODE_PAUSED_MESSAGE)
        })
    })
})
