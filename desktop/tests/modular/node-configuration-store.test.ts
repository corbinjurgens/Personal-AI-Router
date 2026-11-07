// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { RemoteManagementCapability } from '@/shared/types/cluster'
import type { ClusterInitialSnapshot } from '@/shared/types/bootstrap'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import type {
    EngineDescriptor,
    EngineModels,
    EngineType,
    ModelIdentity
} from '@/shared/types/engines'
import type { IClusterApi, INodePolicyApi } from '@/ui/api/pair-api'
import type { NodePolicyDocument, NodePolicySnapshot } from '@/shared/types/node-policy'
import type { NodeRoles } from '@/shared/types/node-roles'
import { getPairTransport } from '@/ui/api/bootstrap'
import { createEngineApi } from '@/ui/api/engine-api'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import {
    inventoryChoices,
    roleIdentity,
    roleItems,
    roleLabel,
    useNodeConfigurationStore,
    watchNodeManagement
} from '@/ui/stores/node-configuration.store'

vi.mock('@/ui/api/bootstrap', () => ({ getPairTransport: vi.fn() }))

const read = vi.fn<IClusterApi['getRemoteManagementForNode']>()
const policyGet = vi.fn<INodePolicyApi['get']>()
const policySet = vi.fn<INodePolicyApi['set']>()
const connects = new Set<() => void>()
const disconnects = new Set<() => void>()
let connected = true
const transport: ServiceTransport = {
    get connected() {
        return connected
    },
    invoke: vi.fn<ServiceTransport['invoke']>(),
    subscribePush: vi.fn<ServiceTransport['subscribePush']>(),
    onConnect: callback => {
        connects.add(callback)
        return () => {
            connects.delete(callback)
        }
    },
    onDisconnect: callback => {
        disconnects.add(callback)
        return () => {
            disconnects.delete(callback)
        }
    },
    onAuthFailure: () => () => {},
    destroy: vi.fn<ServiceTransport['destroy']>()
}

function deferred<T>() {
    let resolve: (value: T) => void = () => {
        throw new Error('deferred not initialized')
    }
    const promise = new Promise<T>(finish => {
        resolve = finish
    })
    return { promise, resolve }
}

function capability(
    state: RemoteManagementCapability['state'],
    revision = '5'
): RemoteManagementCapability {
    return { version: 1, state, allowed: state === 'allowed', revision }
}

const cleanups: Array<() => void> = []
const store = () => useNodeConfigurationStore.getState()

const orderedRoles: NodeRoles = [
    { name: 'fast', profiles: [] },
    { name: 'coder', profiles: ['code-small', 'code-large'] },
    { name: 'deep', profiles: ['code-large'] }
]
const apiEntries = {
    coder: [{ kind: 'api', deployment: 'cloud-code' }],
    ops: [{ kind: 'api', deployment: 'cloud-ops' }]
}
const apiRouting = {
    coder: { bareMode: 'local', remoteAllowed: false },
    ops: { bareMode: 'local', remoteAllowed: false }
}

function roleSnapshot(
    revision = '5',
    roles: NodeRoles = orderedRoles,
    tiers: Record<string, Array<{ kind: string; deployment: string }>> = apiEntries,
    routingTiers: Record<string, { bareMode: string; remoteAllowed: boolean }> = apiRouting
): NodePolicySnapshot {
    return {
        revision,
        activation: null,
        policy: `{
  "revision": "${revision}",
  "availability": "available",
  "opaque": { "number": 1.2300e+09, "escaped": "\\u0061" },
  "profiles": [
    {"key":"code-small","engine":"llamacpp","model":"small"},
    {"key":"code-large","engine":"llamacpp","model":"large"}
  ],
  "roles": ${JSON.stringify(roles)},
  "tiers": ${JSON.stringify(tiers)},
  "routing": {"tiers": ${JSON.stringify(routingTiers)}, "opaque": [1, 2]},
  "admission": { "maxResidentModels": 1 }
}`
    }
}

function roleDocument(revision = '5', roles: NodeRoles = orderedRoles): NodePolicyDocument {
    return { ...roleSnapshot(revision, roles), availability: 'available', recoveryRequired: false }
}

function renamedRoleDocument(): NodePolicyDocument {
    return {
        ...roleSnapshot(
            '6',
            [
                orderedRoles[0],
                { name: 'code', profiles: ['code-small', 'code-large'] },
                orderedRoles[2]
            ],
            { code: apiEntries.coder, ops: apiEntries.ops },
            { code: apiRouting.coder, ops: apiRouting.ops }
        ),
        availability: 'available',
        recoveryRequired: false
    }
}

function roleEntry(targetId = 'peer') {
    const entry = store().editors.get(targetId)
    if (!entry) throw new Error(`Missing role editor for ${targetId}`)
    return entry
}

describe('user ordered role drafts', () => {
    beforeEach(() => {
        store().adopt('peer', roleSnapshot())
        policySet.mockResolvedValue(roleSnapshot('6'))
    })

    it('appends coder as an empty declaration and saves it with the baseline revision', async () => {
        store().adopt('empty', roleSnapshot('2', []))
        store().addRole('empty', 'coder')
        expect(roleEntry('empty').draft).toEqual([{ name: 'coder', profiles: [] }])

        await store().save('empty', undefined)

        expect(policySet).toHaveBeenCalledTimes(1)
        expect(policySet.mock.calls[0][0]).toMatchObject({ expectedPersistedRevision: '2' })
        expect(policySet.mock.calls[0][0].nodeId).toBeUndefined()
        expect(JSON.parse(policySet.mock.calls[0][0].policy).roles).toEqual([
            { name: 'coder', profiles: [] }
        ])
    })

    it('saves an empty role list after removing every declaration', async () => {
        store().removeRole('peer', 'fast')
        store().removeRole('peer', 'coder')
        store().removeRole('peer', 'deep')
        await store().save('peer', undefined)

        expect(policySet).toHaveBeenCalledTimes(1)
        expect(JSON.parse(policySet.mock.calls[0][0].policy).roles).toEqual([])
    })

    it('renames coder in place and moves both API maps in one revision checked save', async () => {
        expect(store().roleApiEntryPresent('peer', 'coder')).toBe(true)
        expect(store().roleApiEntryPresent('peer', 'fast')).toBe(false)
        store().renameRole('peer', 'coder', 'code')
        expect(roleEntry().draft).toEqual([
            { name: 'fast', profiles: [] },
            { name: 'code', profiles: ['code-small', 'code-large'] },
            { name: 'deep', profiles: ['code-large'] }
        ])
        await store().save('peer', undefined)

        expect(policySet).toHaveBeenCalledTimes(1)
        const request = policySet.mock.calls[0][0]
        expect(request).toMatchObject({ nodeId: undefined, expectedPersistedRevision: '5' })
        const saved = JSON.parse(request.policy)
        expect(saved.roles).toEqual([
            { name: 'fast', profiles: [] },
            { name: 'code', profiles: ['code-small', 'code-large'] },
            { name: 'deep', profiles: ['code-large'] }
        ])
        expect(saved.tiers).toEqual({
            code: [{ kind: 'api', deployment: 'cloud-code' }],
            ops: [{ kind: 'api', deployment: 'cloud-ops' }]
        })
        expect(saved.routing).toEqual({
            tiers: {
                code: { bareMode: 'local', remoteAllowed: false },
                ops: { bareMode: 'local', remoteAllowed: false }
            },
            opaque: [1, 2]
        })
        expect(request.policy).toContain('"opaque": { "number": 1.2300e+09, "escaped": "\\u0061" }')
        expect(request.policy).toContain('"admission": { "maxResidentModels": 1 }')
    })

    it('removes a role and both API entries without changing the other declarations', async () => {
        store().removeRole('peer', 'coder')
        await store().save('peer', undefined)

        expect(policySet).toHaveBeenCalledTimes(1)
        expect(policySet.mock.calls[0][0]).toMatchObject({ expectedPersistedRevision: '5' })
        const saved = JSON.parse(policySet.mock.calls[0][0].policy)
        expect(saved.roles).toEqual([
            { name: 'fast', profiles: [] },
            { name: 'deep', profiles: ['code-large'] }
        ])
        expect(saved.tiers).toEqual({ ops: [{ kind: 'api', deployment: 'cloud-ops' }] })
        expect(saved.routing.tiers).toEqual({ ops: { bareMode: 'local', remoteAllowed: false } })
        expect(saved.profiles).toEqual([
            { key: 'code-small', engine: 'llamacpp', model: 'small' },
            { key: 'code-large', engine: 'llamacpp', model: 'large' }
        ])
    })

    it('moves a role down and up while preserving memberships and API entries', async () => {
        store().moveRole('peer', 'coder', 1)
        expect(roleEntry().draft.map(role => role.name)).toEqual(['fast', 'deep', 'coder'])
        store().moveRole('peer', 'fast', 1)
        store().moveRole('peer', 'coder', -1)
        await store().save('peer', undefined)

        const saved = JSON.parse(policySet.mock.calls[0][0].policy)
        expect(saved.roles).toEqual([
            { name: 'deep', profiles: ['code-large'] },
            { name: 'coder', profiles: ['code-small', 'code-large'] },
            { name: 'fast', profiles: [] }
        ])
        expect(saved.tiers).toEqual(apiEntries)
        expect(saved.routing.tiers).toEqual(apiRouting)
    })

    it('tracks API provenance through chained renames and removal', async () => {
        store().renameRole('peer', 'coder', 'code')
        expect(store().roleApiEntryPresent('peer', 'code')).toBe(true)
        store().renameRole('peer', 'code', 'review')
        expect(store().roleApiEntryPresent('peer', 'review')).toBe(true)
        store().removeRole('peer', 'review')
        await store().save('peer', undefined)

        expect(policySet).toHaveBeenCalledTimes(1)
        const saved = JSON.parse(policySet.mock.calls[0][0].policy)
        expect(saved.tiers).toEqual({ ops: [{ kind: 'api', deployment: 'cloud-ops' }] })
        expect(saved.routing.tiers).toEqual({ ops: { bareMode: 'local', remoteAllowed: false } })
        expect(saved.roles.map((role: { name: string }) => role.name)).toEqual(['fast', 'deep'])
    })

    it('discards API edit provenance along with a renamed draft', async () => {
        store().renameRole('peer', 'coder', 'code')
        store().discard('peer')
        expect(roleEntry().draft).toEqual(orderedRoles)
        expect(store().roleApiEntryPresent('peer', 'coder')).toBe(true)
        store().removeRole('peer', 'fast')
        await store().save('peer', undefined)

        const saved = JSON.parse(policySet.mock.calls[0][0].policy)
        expect(saved.roles).toEqual([
            { name: 'coder', profiles: ['code-small', 'code-large'] },
            { name: 'deep', profiles: ['code-large'] }
        ])
        expect(saved.tiers).toEqual(apiEntries)
        expect(saved.routing.tiers).toEqual(apiRouting)
    })

    it('keeps a conflicted API rename unsent until explicitly rebased on readback', async () => {
        policySet.mockRejectedValueOnce(new Error('-32010: policy conflict: stale revision'))
        policyGet.mockResolvedValue(roleDocument('7'))
        store().renameRole('peer', 'coder', 'code')
        expect(await store().save('peer', undefined)).toBeNull()
        expect(roleEntry().phase).toBe('conflict')
        expect(roleEntry().latest?.revision).toBe('7')
        expect(JSON.parse(roleEntry().latest?.policy ?? '{}').tiers).toEqual(apiEntries)
        await store().save('peer', undefined)
        expect(policySet).toHaveBeenCalledTimes(1)
        expect(policyGet).toHaveBeenCalledWith(undefined)

        store().keepDraftOnLatest('peer')
        await store().save('peer', undefined)
        expect(policySet).toHaveBeenCalledTimes(2)
        const request = policySet.mock.calls[1][0]
        expect(request.expectedPersistedRevision).toBe('7')
        expect(JSON.parse(request.policy).tiers).toEqual({
            code: [{ kind: 'api', deployment: 'cloud-code' }],
            ops: [{ kind: 'api', deployment: 'cloud-ops' }]
        })
        expect(JSON.parse(request.policy).routing.tiers).toEqual({
            code: { bareMode: 'local', remoteAllowed: false },
            ops: { bareMode: 'local', remoteAllowed: false }
        })
    })

    it('keeps a paused target rename as a draft without sending either API map', async () => {
        store().adopt('paused', { ...roleSnapshot(), availability: 'draining' })
        store().renameRole('paused', 'coder', 'code')
        expect(await store().save('paused', 'paused')).toBeNull()
        expect(roleEntry('paused').phase).toBe('vetoed')
        expect(roleEntry('paused').draft[1]).toEqual({
            name: 'code',
            profiles: ['code-small', 'code-large']
        })
        expect(policySet).not.toHaveBeenCalled()
    })

    it('does not resend an uncertain API rename and retains the complete draft', async () => {
        policySet.mockRejectedValueOnce(new Error('connection lost'))
        policyGet.mockRejectedValueOnce(new Error('connection lost'))
        store().renameRole('peer', 'coder', 'code')
        expect(await store().save('peer', undefined)).toBeNull()

        expect(policySet).toHaveBeenCalledTimes(1)
        expect(policyGet).toHaveBeenCalledTimes(1)
        expect(roleEntry().phase).toBe('unknown')
        expect(roleEntry().baseline?.revision).toBe('5')
        expect(roleEntry().draft[1]).toEqual({
            name: 'code',
            profiles: ['code-small', 'code-large']
        })
        expect(store().roleApiEntryPresent('peer', 'code')).toBe(true)
    })

    it('confirms a lost rename reply from matching roles and both API maps without resending', async () => {
        const saved = renamedRoleDocument()
        policySet.mockRejectedValueOnce(new Error('connection lost'))
        policyGet.mockResolvedValueOnce(saved)
        store().renameRole('peer', 'coder', 'code')

        expect(await store().save('peer', undefined)).toEqual(saved)
        expect(policySet).toHaveBeenCalledTimes(1)
        expect(policyGet).toHaveBeenCalledExactlyOnceWith(undefined)
        expect(roleEntry().phase).toBe('applied')
        expect(roleEntry().baseline?.policy).toBe(saved.policy)
        expect(roleEntry().roleEdits).toBeUndefined()
    })

    it.each(['tiers', 'routing'])(
        'does not confirm matching roles with mismatched %s API entries',
        async map => {
            const latest = renamedRoleDocument()
            const fields = JSON.parse(latest.policy)
            if (map === 'tiers') fields.tiers = apiEntries
            else fields.routing.tiers = apiRouting
            latest.policy = JSON.stringify(fields)
            policySet.mockRejectedValueOnce(new Error('connection lost'))
            policyGet.mockResolvedValueOnce(latest)
            store().renameRole('peer', 'coder', 'code')
            const draft = roleEntry().draft
            const edits = roleEntry().roleEdits

            expect(await store().save('peer', undefined)).toBeNull()
            expect(roleEntry().phase).toBe('conflict')
            expect(roleEntry().baseline?.revision).toBe('5')
            expect(roleEntry().latest?.policy).toBe(latest.policy)
            expect(roleEntry().draft).toEqual(draft)
            expect(roleEntry().roleEdits).toEqual(edits)
            expect(await store().save('peer', undefined)).toBeNull()
            expect(policySet).toHaveBeenCalledTimes(1)
            expect(policyGet).toHaveBeenCalledTimes(1)
        }
    )

    it.each([true, false])(
        'reloads an unknown rename without replay, confirmed=%s',
        async confirmed => {
            policySet.mockRejectedValueOnce(new Error('connection lost'))
            policyGet.mockRejectedValueOnce(new Error('connection lost'))
            store().renameRole('peer', 'coder', 'code')
            await store().save('peer', undefined)
            expect(roleEntry().phase).toBe('unknown')
            const draft = roleEntry().draft
            const edits = roleEntry().roleEdits
            store().addRole('peer', 'new-role')
            store().renameRole('peer', 'code', 'review')
            store().removeRole('peer', 'code')
            store().moveRole('peer', 'code', 1)
            expect(await store().save('peer', undefined)).toBeNull()
            expect(roleEntry().draft).toEqual(draft)
            expect(roleEntry().roleEdits).toEqual(edits)
            expect(policySet).toHaveBeenCalledTimes(1)
            const latest = confirmed ? renamedRoleDocument() : roleDocument()
            policyGet.mockResolvedValueOnce(latest)

            await store().read('peer', undefined)
            expect(policyGet).toHaveBeenCalledTimes(2)
            expect(policySet).toHaveBeenCalledTimes(1)
            expect(roleEntry().phase).toBe(confirmed ? 'applied' : 'rejected')
            if (confirmed) {
                expect(roleEntry().baseline?.policy).toBe(latest.policy)
                expect(roleEntry().roleEdits).toBeUndefined()
            } else {
                expect(roleEntry().draft).toEqual(draft)
                expect(roleEntry().roleEdits).toEqual(edits)
                expect(roleEntry().message).toContain('did not confirm')
            }
        }
    )

    it('refuses explicit conflict rebasing when the readback adds destination API configuration', async () => {
        const latest = {
            ...roleSnapshot(
                '7',
                orderedRoles,
                { ...apiEntries, code: [{ kind: 'api', deployment: 'cloud-review' }] },
                apiRouting
            ),
            availability: 'available',
            recoveryRequired: false
        } satisfies NodePolicyDocument
        policySet.mockRejectedValueOnce(new Error('-32010: policy conflict: stale revision'))
        policyGet.mockResolvedValueOnce(latest)
        store().renameRole('peer', 'coder', 'code')
        await store().save('peer', undefined)
        const before = roleEntry()
        expect(before.phase).toBe('conflict')

        store().keepDraftOnLatest('peer')
        expect(roleEntry().baseline).toEqual(before.baseline)
        expect(roleEntry().draft).toEqual(before.draft)
        expect(roleEntry().roleEdits).toEqual(before.roleEdits)
        expect(roleEntry().latest).toEqual(before.latest)
        expect(roleEntry().message).toMatch(/coder.*code.*another name/i)
        expect(await store().save('peer', undefined)).toBeNull()
        expect(policySet).toHaveBeenCalledTimes(1)
    })

    it('coalesces a pending save per target while keeping another target independent', async () => {
        const pending = deferred<NodePolicySnapshot>()
        policySet.mockReturnValueOnce(pending.promise)
        store().adopt('other', roleSnapshot('9'))
        store().renameRole('peer', 'coder', 'code')
        store().removeRole('other', 'coder')
        const saving = store().save('peer', undefined)
        expect(await store().save('peer', undefined)).toBeNull()
        await store().save('other', undefined)

        expect(policySet).toHaveBeenCalledTimes(2)
        expect(policySet.mock.calls[0][0]).toMatchObject({
            nodeId: undefined,
            expectedPersistedRevision: '5'
        })
        expect(policySet.mock.calls[1][0]).toMatchObject({
            nodeId: undefined,
            expectedPersistedRevision: '9'
        })
        expect(JSON.parse(policySet.mock.calls[1][0].policy).tiers).toEqual({
            ops: [{ kind: 'api', deployment: 'cloud-ops' }]
        })
        pending.resolve(
            roleSnapshot('6', [
                { name: 'fast', profiles: [] },
                { name: 'code', profiles: ['code-small', 'code-large'] },
                { name: 'deep', profiles: ['code-large'] }
            ])
        )
        await saving
        expect(roleEntry().phase).toBe('applied')
    })
})

describe('local API configuration rename collisions', () => {
    const deployment = [{ kind: 'api', deployment: 'cloud-code' }]
    const destination = [{ kind: 'api', deployment: 'cloud-review' }]
    const rule = { bareMode: 'local', remoteAllowed: false }

    const collisions: Array<{
        label: string
        tiers: Parameters<typeof roleSnapshot>[2]
        routing: Parameters<typeof roleSnapshot>[3]
    }> = [
        {
            label: 'both deployment maps',
            tiers: { coder: deployment, review: destination },
            routing: {}
        },
        { label: 'both routing maps', tiers: {}, routing: { coder: rule, review: rule } },
        {
            label: 'source deployment and destination routing',
            tiers: { coder: deployment },
            routing: { review: rule }
        },
        {
            label: 'source routing and destination deployment',
            tiers: { review: destination },
            routing: { coder: rule }
        }
    ]
    it.each(collisions)(
        'refuses $label before changing draft or API lineage',
        async ({ tiers, routing }) => {
            const baseline = roleSnapshot('5', orderedRoles, tiers, routing)
            store().adopt('peer', baseline)
            const before = roleEntry()
            const message = store().roleRenameError('peer', 'coder', 'review')
            expect(message).toContain('coder')
            expect(message).toContain('review')
            expect(message).toMatch(/another name/i)

            store().renameRole('peer', 'coder', 'review')

            expect(roleEntry().message).toBe(message)
            expect(roleEntry().draft).toEqual(before.draft)
            expect(roleEntry().roleEdits).toEqual(before.roleEdits)
            expect(roleEntry().baseline?.policy).toBe(baseline.policy)
            expect(store().roleApiEntryPresent('peer', 'coder')).toBe(true)
            expect(store().roleApiEntryPresent('peer', 'review')).toBe(true)
            expect(policySet).not.toHaveBeenCalled()
            // Save an independent change to observe both configurations after refusal.
            store().moveRole('peer', 'fast', 1)
            policySet.mockResolvedValue(roleSnapshot('6'))
            await store().save('peer', undefined)
            expect(policySet).toHaveBeenCalledTimes(1)
            const request = policySet.mock.calls[0][0]
            expect(request.expectedPersistedRevision).toBe('5')
            expect(JSON.parse(request.policy).tiers).toEqual(tiers)
            expect(JSON.parse(request.policy).routing.tiers).toEqual(routing)
            expect(JSON.parse(request.policy).roles).toEqual([
                orderedRoles[1],
                orderedRoles[0],
                orderedRoles[2]
            ])
        }
    )

    const nonCollisions: Array<{
        label: string
        tiers: Parameters<typeof roleSnapshot>[2]
        routing: Parameters<typeof roleSnapshot>[3]
        expectedTiers: Parameters<typeof roleSnapshot>[2]
        expectedRouting: Parameters<typeof roleSnapshot>[3]
        sourcePresent: boolean
    }> = [
        {
            label: 'empty source retains destination',
            tiers: { coder: [], review: destination },
            routing: { review: rule },
            expectedTiers: { review: destination },
            expectedRouting: { review: rule },
            sourcePresent: false
        },
        {
            label: 'empty destination receives source',
            tiers: { coder: deployment, review: [] },
            routing: { coder: rule },
            expectedTiers: { review: deployment },
            expectedRouting: { review: rule },
            sourcePresent: true
        },
        {
            label: 'neither configured deletes old empty key',
            tiers: { coder: [], review: [] },
            routing: {},
            expectedTiers: { review: [] },
            expectedRouting: {},
            sourcePresent: false
        },
        {
            label: 'empty source and destination routing',
            tiers: { coder: [] },
            routing: { review: rule },
            expectedTiers: {},
            expectedRouting: { review: rule },
            sourcePresent: false
        },
        {
            label: 'source routing and empty destination',
            tiers: { review: [] },
            routing: { coder: rule },
            expectedTiers: {},
            expectedRouting: { review: rule },
            sourcePresent: true
        }
    ]
    it.each(nonCollisions)(
        'allows $label in validation and saved policy',
        async ({ tiers, routing, expectedTiers, expectedRouting, sourcePresent }) => {
            store().adopt('peer', roleSnapshot('5', orderedRoles, tiers, routing))
            expect(store().roleApiEntryPresent('peer', 'coder')).toBe(sourcePresent)
            expect(store().roleRenameError('peer', 'coder', 'review')).toBeUndefined()
            store().renameRole('peer', 'coder', 'review')
            expect(roleEntry().draft[1]).toEqual({
                name: 'review',
                profiles: ['code-small', 'code-large']
            })
            policySet.mockResolvedValue(roleSnapshot('6'))
            await store().save('peer', undefined)
            expect(policySet).toHaveBeenCalledTimes(1)
            const request = policySet.mock.calls[0][0]
            expect(request.expectedPersistedRevision).toBe('5')
            expect(JSON.parse(request.policy).tiers).toEqual(expectedTiers)
            expect(JSON.parse(request.policy).routing.tiers).toEqual(expectedRouting)
            expect(request.policy).toContain('"number": 1.2300e+09, "escaped": "\\u0061"')
        }
    )

    it('deletes an empty old key on local removal without claiming configuration exists', async () => {
        store().adopt('peer', roleSnapshot('5', orderedRoles, { coder: [] }, {}))
        expect(store().roleApiEntryPresent('peer', 'coder')).toBe(false)
        store().removeRole('peer', 'coder')
        policySet.mockResolvedValue(roleSnapshot('6'))
        await store().save('peer', undefined)
        expect(policySet).toHaveBeenCalledTimes(1)
        const saved = JSON.parse(policySet.mock.calls[0][0].policy)
        expect(saved.tiers).toEqual({})
        expect(saved.routing.tiers).toEqual({})
        expect(saved.roles).toEqual([orderedRoles[0], orderedRoles[2]])
    })

    it('preserves exact private entry tokens through a chained local rename', async () => {
        const baseline = roleSnapshot('5', orderedRoles, {}, {})
        const entrySource =
            '[{"kind":"api","deployment":"cloud-code","opaque":9007199254740993,"decimal":1.2300e+09,"escaped":"\\u0061"}]'
        const routingSource = '{"bareMode":"local","remoteAllowed":false,"opaque":9.9900e-7}'
        baseline.policy = baseline.policy
            .replace('"tiers": {}', `"tiers": {"coder":${entrySource}}`)
            .replace('"routing": {"tiers": {}', `"routing": {"tiers": {"coder":${routingSource}}`)
        expect(baseline.policy).toContain(entrySource)
        expect(baseline.policy).toContain(routingSource)
        store().adopt('peer', baseline)
        store().renameRole('peer', 'coder', 'code')
        expect(store().roleRenameError('peer', 'code', 'review')).toBeUndefined()
        store().renameRole('peer', 'code', 'review')
        policySet.mockResolvedValue(roleSnapshot('6'))
        await store().save('peer', undefined)
        expect(policySet).toHaveBeenCalledTimes(1)
        const request = policySet.mock.calls[0][0]
        expect(request.expectedPersistedRevision).toBe('5')
        expect(request.policy).toContain(`"review":${entrySource}`)
        expect(request.policy).toContain(`"review":${routingSource}`)
        expect(Object.keys(JSON.parse(request.policy).tiers)).toEqual(['review'])
        expect(Object.keys(JSON.parse(request.policy).routing.tiers)).toEqual(['review'])
        expect(request.policy).toContain('"number": 1.2300e+09, "escaped": "\\u0061"')
    })

    it('uses retained destination configuration for subsequent unsaved collisions and removal', async () => {
        store().adopt(
            'peer',
            roleSnapshot('5', orderedRoles, { coder: [], review: destination }, { ops: rule })
        )
        store().renameRole('peer', 'coder', 'review')
        expect(store().roleApiEntryPresent('peer', 'review')).toBe(true)
        const before = roleEntry()
        expect(store().roleRenameError('peer', 'review', 'ops')).toMatch(/another name/i)
        store().renameRole('peer', 'review', 'ops')
        expect(roleEntry().draft).toEqual(before.draft)
        expect(roleEntry().roleEdits).toEqual(before.roleEdits)
        store().removeRole('peer', 'review')
        expect(store().roleApiEntryPresent('peer', 'review')).toBe(false)
        store().renameRole('peer', 'fast', 'review')
        expect(store().roleRenameError('peer', 'review', 'ops')).toBeUndefined()
        store().renameRole('peer', 'review', 'ops')
        policySet.mockResolvedValue(roleSnapshot('6'))
        await store().save('peer', undefined)
        expect(policySet).toHaveBeenCalledTimes(1)
        const request = policySet.mock.calls[0][0]
        expect(request.expectedPersistedRevision).toBe('5')
        expect(JSON.parse(request.policy).tiers).toEqual({})
        expect(JSON.parse(request.policy).routing.tiers).toEqual({ ops: rule })
        expect(JSON.parse(request.policy).roles).toEqual([
            { name: 'ops', profiles: [] },
            orderedRoles[2]
        ])
    })
})

describe('remote role edits preserve supplied private policy source', () => {
    it.each(['rename', 'remove'])(
        '%s sends only declaration changes without API intent',
        async action => {
            const baseline = roleSnapshot()
            const privateTiers =
                '"tiers": { "coder": [{"kind":"api","deployment":"cloud-code","opaque":1.2300e+09}], "review": [{"kind":"api","deployment":"cloud-review"}] }'
            const privateRouting =
                '"routing": { "tiers": {"coder":{"bareMode":"local","remoteAllowed":false,"opaque":"\\u0061"}}, "opaque": [1, 2] }'
            baseline.policy = baseline.policy
                .replace(`"tiers": ${JSON.stringify(apiEntries)}`, privateTiers)
                .replace(
                    `"routing": {"tiers": ${JSON.stringify(apiRouting)}, "opaque": [1, 2]}`,
                    privateRouting
                )
            expect(baseline.policy).toContain(privateTiers)
            expect(baseline.policy).toContain(privateRouting)
            store().adopt('peer', baseline)
            if (action === 'rename') {
                expect(store().roleRenameError('peer', 'coder', 'review', 'peer')).toBeUndefined()
                store().renameRole('peer', 'coder', 'review', 'peer')
            } else {
                store().removeRole('peer', 'coder', 'peer')
            }
            expect(roleEntry().roleEdits ?? []).toEqual([])
            policySet.mockResolvedValue(roleSnapshot('6'))
            await store().save('peer', 'peer')
            expect(policySet).toHaveBeenCalledTimes(1)
            const request = policySet.mock.calls[0][0]
            expect(Object.keys(request).sort()).toEqual([
                'expectedPersistedRevision',
                'nodeId',
                'policy'
            ])
            expect(request.nodeId).toBe('peer')
            expect(request.expectedPersistedRevision).toBe('5')
            expect(request.policy).toContain(privateTiers)
            expect(request.policy).toContain(privateRouting)
            expect(request.policy).toContain('"number": 1.2300e+09, "escaped": "\\u0061"')
            const expectedRoles =
                action === 'rename'
                    ? [
                          orderedRoles[0],
                          { name: 'review', profiles: ['code-small', 'code-large'] },
                          orderedRoles[2]
                      ]
                    : [orderedRoles[0], orderedRoles[2]]
            expect(JSON.parse(request.policy).roles).toEqual(expectedRoles)
            expect(request.policy).toBe(
                baseline.policy.replace(JSON.stringify(orderedRoles), JSON.stringify(expectedRoles))
            )
        }
    )
})

describe('role identity labels', () => {
    const descriptors = new Map<EngineType, EngineDescriptor>()
    const role = { engine: 'llamacpp', model: 'preset-ctx131072' }
    const identity: ModelIdentity = {
        repository: 'unsloth/LFM2.5-24B-A2B-GGUF',
        quantization: 'UD-Q4_K_XL',
        quantizationSource: 'hf-tag',
        strength: 'declared'
    }

    it('shows the identity label and engine separated by a middle dot', () => {
        expect(roleLabel(role, descriptors, identity)).toBe(
            'LFM2.5 24B A2B (UD-Q4_K_XL) · llama.cpp'
        )
    })

    it('formats the model without an identity while retaining the declared engine', () => {
        expect(
            roleLabel({ engine: 'llamacpp', model: 'owner/model-GGUF:Q4_K_M' }, descriptors)
        ).toBe('model (Q4_K_M) · llama.cpp')
        expect(
            roleLabel({ engine: 'unregistered', model: 'lmstudio/model-GGUF' }, descriptors)
        ).toBe('model · unregistered')
    })

    it('uses only the role engine and exact model when resolving inventory metadata', () => {
        const inventory: EngineModels[] = [
            {
                nodeId: 'peer',
                engineType: 'llama-cpp',
                models: [
                    {
                        name: role.model,
                        identity,
                        size: 0,
                        downloaded: true,
                        status: 'idle',
                        parameterSize: '',
                        family: '',
                        sizeVram: null,
                        expiresAt: null,
                        expiry: '10m',
                        capabilities: []
                    }
                ]
            }
        ]
        expect(roleIdentity(role, inventory)).toEqual(identity)
        expect(roleIdentity({ ...role, engine: 'other' }, inventory)).toBeUndefined()
        expect(roleIdentity({ ...role, model: 'different' }, inventory)).toBeUndefined()
        const profile = { ...role, key: 'long-context', displayName: 'Long context' }
        const choices = inventoryChoices(inventory, [profile])
        expect(roleItems(choices, profile.key, descriptors, inventory)).toEqual([
            { value: '', children: 'Unset' },
            {
                value: 'long-context',
                children: 'LFM2.5 24B A2B (UD-Q4_K_XL) · llama.cpp'
            }
        ])
        expect(roleItems(choices, profile.key, descriptors, undefined)[1].children).toBe(
            'preset-ctx131072 · llama.cpp'
        )
    })
})

function clusterSnapshot(certFingerprint = 'A'): ClusterInitialSnapshot {
    return {
        identity: {
            nodeUuid: 'self',
            nodeId: 'self.local',
            name: 'Self',
            certFingerprint,
            clusterId: 'home'
        },
        info: { clusterId: 'home', isClustered: true, clusterFriendlyName: 'Home' },
        members: [
            {
                nodeUuid: 'peer',
                id: 'peer.local',
                name: 'Peer',
                ipAddress: '192.0.2.1',
                port: 1,
                clusterId: 'home',
                state: 'member',
                joinedAt: 1,
                lastSeen: 1
            }
        ],
        pendingInvites: []
    }
}

beforeEach(() => {
    connected = true
    vi.mocked(getPairTransport).mockReturnValue(transport)
    read.mockReset().mockResolvedValue(capability('allowed'))
    policyGet.mockReset()
    policySet.mockReset()
    vi.mocked(transport.invoke).mockReset().mockResolvedValue(null)
    vi.stubGlobal('window', {
        pairApi: {
            nodePolicy: { get: policyGet, set: policySet },
            cluster: {
                getRemoteManagementForNode: read,
                getRemoteManagement: vi.fn().mockResolvedValue({
                    version: 1,
                    revision: '1',
                    grants: [{ peerUuid: 'peer', allowed: false, state: 'denied' }]
                })
            }
        }
    })
    useClusterInvitationsStore.getState().cleanup()
    useNodeConfigurationStore.setState({ management: new Map(), editors: new Map() })
})

afterEach(() => {
    cleanups.splice(0).forEach(stop => stop())
    useClusterInvitationsStore.getState().cleanup()
    vi.unstubAllGlobals()
})

describe('shared target management capability', () => {
    const staleStates: RemoteManagementCapability['state'][] = ['unreviewed', 'denied']
    it.each(staleStates)(
        'rereads an open %s permission after a successful remote management command',
        async state => {
            read.mockResolvedValueOnce(capability(state))
            cleanups.push(watchNodeManagement('peer', 'peer'))
            await store().readManagement('peer', 'peer')
            expect(store().management.get('peer')?.state).toBe(state)

            createEngineApi(transport).install('llama-cpp', 'peer')

            await vi.waitFor(() => expect(store().management.get('peer')?.state).toBe('allowed'))
            expect(read).toHaveBeenCalledTimes(2)
            expect(connects.size).toBe(1)
        }
    )

    it('does not refresh permission after a failed management command or create an unwatched target', async () => {
        read.mockResolvedValue(capability('denied'))
        cleanups.push(watchNodeManagement('peer', 'peer'))
        await store().readManagement('peer', 'peer')
        vi.mocked(transport.invoke).mockRejectedValueOnce(new Error('Remote management denied'))
        const api = createEngineApi(transport)
        api.install('llama-cpp', 'peer')
        await Promise.resolve()
        await Promise.resolve()
        api.install('llama-cpp', 'unwatched')
        await Promise.resolve()
        await Promise.resolve()
        expect(read).toHaveBeenCalledTimes(1)
        expect(store().management.get('peer')?.state).toBe('denied')
        expect(store().management.has('unwatched')).toBe(false)
    })

    it('coalesces mutations during an older read into one fresh capability read', async () => {
        const old = deferred<RemoteManagementCapability>()
        read.mockReturnValueOnce(old.promise)
        cleanups.push(watchNodeManagement('peer', 'peer'))
        const initial = store().readManagement('peer', 'peer')
        const api = createEngineApi(transport)
        api.install('llama-cpp', 'peer')
        api.install('llama-cpp', 'peer')
        await Promise.resolve()
        await Promise.resolve()
        expect(read).toHaveBeenCalledTimes(1)

        old.resolve(capability('unreviewed'))
        await initial
        await store().readManagement('peer', 'peer')

        expect(read).toHaveBeenCalledTimes(2)
        expect(store().management.get('peer')?.state).toBe('allowed')
    })

    it('fences an old capability after same-UUID removal and readmission', async () => {
        const cluster = useClusterInvitationsStore.getState()
        const snapshot = clusterSnapshot()
        cluster.hydrate(snapshot)
        const old = deferred<RemoteManagementCapability>()
        read.mockReturnValueOnce(old.promise)
        cleanups.push(watchNodeManagement('peer', 'peer'))
        const pending = store().readManagement('peer', 'peer')
        await Promise.resolve()
        read.mockResolvedValue(capability('denied', '1'))
        cluster.hydrate({ ...snapshot, members: [] })
        expect(store().management.get('peer')?.state).toBe('unreachable')
        cluster.hydrate({
            ...snapshot,
            members: snapshot.members.map(member => ({ ...member, joinedAt: 2 }))
        })
        await vi.waitFor(() =>
            expect(store().management.get('peer')).toMatchObject({ state: 'denied', revision: '1' })
        )
        old.resolve(capability('allowed', '99'))
        await pending
        expect(read).toHaveBeenCalledTimes(2)
        expect(store().management.get('peer')).toMatchObject({ state: 'denied', revision: '1' })
    })

    it('fences an old caller capability when the local certificate changes', async () => {
        const cluster = useClusterInvitationsStore.getState()
        const snapshot = clusterSnapshot()
        cluster.hydrate(snapshot)
        const old = deferred<RemoteManagementCapability>()
        read.mockReturnValueOnce(old.promise)
        cleanups.push(watchNodeManagement('peer', 'peer'))
        const pending = store().readManagement('peer', 'peer')
        await Promise.resolve()
        read.mockResolvedValue(capability('denied', '1'))
        // Reuse the exact member array: the identity itself must fence authority.
        cluster.hydrate({
            ...snapshot,
            identity: { ...snapshot.identity, certFingerprint: 'B' }
        })
        await vi.waitFor(() =>
            expect(store().management.get('peer')).toMatchObject({ state: 'denied', revision: '1' })
        )
        old.resolve(capability('allowed', '99'))
        await pending
        expect(read).toHaveBeenCalledTimes(2)
        expect(store().management.get('peer')).toMatchObject({ state: 'denied', revision: '1' })
    })

    it('does not issue a queued capability read after cluster cleanup', async () => {
        const cluster = useClusterInvitationsStore.getState()
        cluster.hydrate(clusterSnapshot())
        cleanups.push(watchNodeManagement('peer', 'peer'))
        cluster.cleanup()
        await Promise.resolve()
        await Promise.resolve()
        expect(read).not.toHaveBeenCalled()
        expect(store().management.get('peer')?.state).toBe('unreachable')
    })

    it('does not reread capability for name and last-seen changes alone', async () => {
        const cluster = useClusterInvitationsStore.getState()
        const snapshot = clusterSnapshot()
        cluster.hydrate(snapshot)
        cleanups.push(watchNodeManagement('peer', 'peer'))
        await store().readManagement('peer', 'peer')
        cluster.hydrate({
            ...snapshot,
            members: snapshot.members.map(member => ({ ...member, name: 'New name', lastSeen: 2 }))
        })
        await Promise.resolve()
        expect(read).toHaveBeenCalledTimes(1)
        expect(store().management.get('peer')?.state).toBe('allowed')
    })

    it('shares one target read and transport lifecycle between consumers', async () => {
        const first = watchNodeManagement('peer', 'peer')
        const second = watchNodeManagement('peer', 'peer')
        cleanups.push(first, second)
        await store().readManagement('peer', 'peer')
        expect(read).toHaveBeenCalledExactlyOnceWith('peer')
        expect(store().management.get('peer')).toMatchObject({ state: 'allowed', revision: '5' })
        expect(connects.size).toBe(1)
        expect(disconnects.size).toBe(1)
        first()
        first()
        expect(connects.size).toBe(1)
        expect(store().management.get('peer')?.state).toBe('allowed')
        second()
        expect(connects.size).toBe(0)
        expect(disconnects.size).toBe(0)
        expect(store().management.get('peer')?.state).toBe('unreachable')
    })

    it.each(['denied', 'unreviewed', 'unavailable', 'unsupported'])(
        'retains %s as a refusal with an explanation',
        async state => {
            if (
                state !== 'denied' &&
                state !== 'unreviewed' &&
                state !== 'unavailable' &&
                state !== 'unsupported'
            )
                throw new Error('invalid test state')
            read.mockResolvedValue(capability(state, state === 'unsupported' ? '' : '5'))
            await store().readManagement('peer', 'peer')
            expect(store().management.get('peer')?.state).toBe(state)
            expect(store().management.get('peer')?.message).toBeTruthy()
        }
    )

    it('never promotes contradictory or revisionless remote allow data', async () => {
        read.mockResolvedValueOnce({ ...capability('denied'), allowed: true })
        await store().readManagement('peer', 'peer')
        expect(store().management.get('peer')?.state).toBe('unavailable')
        read.mockResolvedValueOnce(capability('allowed', ''))
        await store().readManagement('peer', 'peer')
        expect(store().management.get('peer')?.state).toBe('unavailable')
    })

    it('reports transport errors as unreachable rather than unsupported', async () => {
        read.mockRejectedValue(new Error('device offline'))
        await store().readManagement('peer', 'peer')
        expect(store().management.get('peer')).toMatchObject({
            state: 'unreachable',
            message: 'device offline'
        })
    })

    it('uses an explicit local target without requesting any remote grant', async () => {
        await store().readManagement('self', undefined)
        expect(store().management.get('self')).toMatchObject({ state: 'local', revision: '' })
        expect(read).not.toHaveBeenCalled()
    })

    it('fences an old read after disconnect and reads again on reconnect without mutations', async () => {
        const old = deferred<RemoteManagementCapability>()
        read.mockReturnValueOnce(old.promise)
        cleanups.push(watchNodeManagement('peer', 'peer'))
        const pending = store().readManagement('peer', 'peer')
        await Promise.resolve()
        expect(store().management.get('peer')?.message).toContain('Checking')
        connected = false
        disconnects.forEach(callback => callback())
        expect(store().management.get('peer')?.state).toBe('unreachable')
        connected = true
        read.mockResolvedValue(capability('denied', '1'))
        connects.forEach(callback => callback())
        await store().readManagement('peer', 'peer')
        old.resolve(capability('allowed', '99'))
        await pending
        expect(store().management.get('peer')).toMatchObject({ state: 'denied', revision: '1' })
        expect(read).toHaveBeenCalledTimes(2)
    })

    it('fences a read after the last watcher leaves and an old cleanup cannot detach its replacement', async () => {
        const old = deferred<RemoteManagementCapability>()
        read.mockReturnValueOnce(old.promise)
        const stop = watchNodeManagement('slot', 'old-peer')
        cleanups.push(stop)
        const pending = store().readManagement('slot', 'old-peer')
        await Promise.resolve()
        const replacement = watchNodeManagement('slot', 'new-peer')
        cleanups.push(replacement)
        read.mockResolvedValue(capability('denied', '1'))
        stop()
        await store().readManagement('slot', 'new-peer')
        old.resolve(capability('allowed', '50'))
        await pending
        expect(store().management.get('slot')).toMatchObject({ state: 'denied', revision: '1' })
        expect(connects.size).toBe(1)
        replacement()
        expect(connects.size).toBe(0)
    })

    it('discards an allow arriving after the last watcher closes', async () => {
        const old = deferred<RemoteManagementCapability>()
        read.mockReturnValue(old.promise)
        const stop = watchNodeManagement('closed', 'closed')
        cleanups.push(stop)
        const pending = store().readManagement('closed', 'closed')
        stop()
        old.resolve(capability('allowed'))
        await pending
        expect(store().management.get('closed')?.state).toBe('unreachable')
    })

    it('a read after definite refusal refreshes capability without keeping an old allow', async () => {
        await store().readManagement('peer', 'peer')
        read.mockResolvedValue(capability('denied', '6'))
        const refresh = store().readManagement('peer', 'peer')
        expect(store().management.get('peer')?.state).toBe('loading')
        await refresh
        expect(store().management.get('peer')?.state).toBe('denied')
    })
})
