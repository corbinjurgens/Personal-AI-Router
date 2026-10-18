// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Role drafts for the role editor, one per node, for the app session.
 *
 * `policy:set` replaces the whole document and fills omitted fields with
 * defaults, so a role edit never sends a document of its own: it takes the
 * target's own projection as last read, substitutes `roles` only, and sends it
 * back checked against that target's revision. A refused or uncertain save is
 * read back and shown; nothing is retried automatically.
 */
import { create } from 'zustand'
import type { EngineDescriptor, EngineModels, EngineType } from '@/shared/types/engines'
import type { JsonValue } from '@/shared/types/json'
import type { NodePolicySnapshot } from '@/shared/types/node-policy'
import type { NodeRole, NodeRoles, RoleTier } from '@/shared/types/node-roles'
import { engineLabel, engineManagerName, engineTypeFromManagerName } from '@/shared/utils/engines'
import getErrorString from '@/shared/utils/get-error-string'
import {
    pauseOriginValue,
    parseNodeRoles,
    ROLE_TIERS,
    sameNodeRoles
} from '@/shared/utils/node-roles'

type JsonMap = { [key: string]: JsonValue }

function policyObject(text: string): JsonMap {
    let parsed: JsonValue
    try {
        parsed = JSON.parse(text)
    } catch {
        return {}
    }
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return {}
    return parsed
}

/** The roles a policy document declares. */
export function policyRoles(policy: string): NodeRoles {
    return parseNodeRoles(policyObject(policy).roles)
}

/** Whether the document says the node was paused on the node itself. */
export function pausedLocally(policy: string): boolean {
    const body = policyObject(policy)
    if (body.availability !== 'paused') return false
    // A pause with no recorded origin predates the field and counts as local.
    return pauseOriginValue(body.pausedBy) !== 'peer'
}

/**
 * The target's own document with `roles` replaced by `roles` and every other
 * field kept as read. Unset tiers are omitted, which leaves them undeclared.
 */
export function withRoles(policy: string, roles: NodeRoles): string {
    const body = policyObject(policy)
    const declared: JsonMap = {}
    for (const tier of ROLE_TIERS) {
        const role = roles[tier]
        if (role) declared[tier] = { engine: role.engine, model: role.model }
    }
    return JSON.stringify({ ...body, roles: declared }, null, 2)
}

/** One installed model a role can name, in engine-manager spelling. */
interface RoleChoice extends NodeRole {
    /** Stable key for a select item. */
    key: string
}

/** A role as the user reads it: the model and the engine that serves it. */
export function roleLabel(
    role: NodeRole,
    descriptors: ReadonlyMap<EngineType, EngineDescriptor>
): string {
    const engine = engineTypeFromManagerName(role.engine)
    const name = engine ? engineLabel(engine, descriptors.get(engine)) : role.engine
    return `${role.model} (${name})`
}

export function roleKey(role: NodeRole): string {
    return JSON.stringify([role.engine, role.model])
}

/** The roles a node can be given: the models its inventory reports as installed. */
export function inventoryChoices(inventory: EngineModels[]): RoleChoice[] {
    const choices = new Map<string, RoleChoice>()
    for (const list of inventory) {
        const engine = engineManagerName(list.engineType)
        for (const model of list.models) {
            if (!model.downloaded || model.name === '') continue
            const role = { engine, model: model.name }
            choices.set(roleKey(role), { ...role, key: roleKey(role) })
        }
    }
    return Array.from(choices.values()).sort(
        (left, right) =>
            left.model.localeCompare(right.model) || left.engine.localeCompare(right.engine)
    )
}

/**
 * A suggestion for each tier: the first local tier entry, in the given
 * documents' order, whose engine and model the node has installed. A tier with
 * no installed entry stays unset. It is only offered, never applied by itself.
 */
export function suggestRoles(choices: RoleChoice[], policies: string[]): NodeRoles {
    const installed = new Set(choices.map(choice => choice.key))
    const roles: NodeRoles = {}
    for (const tier of ROLE_TIERS) {
        for (const policy of policies) {
            if (roles[tier]) break
            const tiers = policyObject(policy).tiers
            if (tiers === null || typeof tiers !== 'object' || Array.isArray(tiers)) continue
            const entries = tiers[tier]
            if (!Array.isArray(entries)) continue
            for (const entry of entries) {
                if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) continue
                if (entry.kind !== 'local') continue
                if (typeof entry.engine !== 'string' || typeof entry.model !== 'string') continue
                const role = { engine: entry.engine, model: entry.model }
                if (installed.has(roleKey(role))) {
                    roles[tier] = role
                    break
                }
            }
        }
    }
    return roles
}

/** A tier whose role differs between two documents. */
interface RoleDifference {
    tier: RoleTier
    mine?: NodeRole
    theirs?: NodeRole
}

export function roleDifferences(mine: NodeRoles, theirs: NodeRoles): RoleDifference[] {
    return ROLE_TIERS.filter(
        tier =>
            mine[tier]?.engine !== theirs[tier]?.engine || mine[tier]?.model !== theirs[tier]?.model
    ).map(tier => ({ tier, mine: mine[tier], theirs: theirs[tier] }))
}

/** How the service refused a save. The bridge carries the JSON-RPC code ahead of the message. */
type Refusal = 'conflict' | 'vetoed' | 'rejected' | 'other'

export function policyRefusal(message: string): Refusal {
    if (/-32010\b/.test(message) || message.includes('policy conflict')) return 'conflict'
    if (/-32011\b/.test(message)) return 'vetoed'
    if (/-32602\b/.test(message)) return 'rejected'
    return 'other'
}

/** Where one node's role edit stands (the bulk UX contract's per-node states). */
type RoleEditPhase =
    | 'draft'
    | 'sending'
    | 'applied'
    | 'conflict'
    | 'vetoed'
    | 'rejected'
    | 'unknown'

interface RoleBaseline {
    policy: string
    revision: string
    roles: NodeRoles
}

interface RoleEditorEntry {
    /** The target's own document the draft substitutes into. */
    baseline: RoleBaseline
    draft: NodeRoles
    phase: RoleEditPhase
    /** The document read back after a conflict, for the diff. */
    latest?: RoleBaseline
    /** The service's message for the last refused or uncertain save. */
    message?: string
}

function baselineOf(snapshot: Pick<NodePolicySnapshot, 'policy' | 'revision'>): RoleBaseline {
    return {
        policy: snapshot.policy,
        revision: snapshot.revision,
        roles: policyRoles(snapshot.policy)
    }
}

function isOlder(revision: string, than: string): boolean {
    return BigInt(revision) < BigInt(than)
}

interface NodeConfigurationState {
    /** Keyed by the broker's node id; this PC's is the self id. */
    editors: Map<string, RoleEditorEntry>
    /**
     * Record the target's document as read or reported. A newer document
     * replaces the baseline when the draft is untouched or its roles did not
     * change; otherwise the draft is stale and shown against it.
     */
    adopt(targetId: string, snapshot: Pick<NodePolicySnapshot, 'policy' | 'revision'>): void
    setRole(targetId: string, tier: RoleTier, role: NodeRole | undefined): void
    /** Replace the draft, for an offered suggestion the user accepted. */
    setDraft(targetId: string, roles: NodeRoles): void
    /** After a conflict: keep the draft and base it on the document read back. */
    keepDraftOnLatest(targetId: string): void
    /** Drop the draft; after a conflict, take the document read back. */
    discard(targetId: string): void
    /**
     * Send the draft to the target. `nodeId` is omitted for this PC. Resolves
     * with the persisted document on success, and null otherwise.
     */
    save(targetId: string, nodeId: string | undefined): Promise<NodePolicySnapshot | null>
}

export const useNodeConfigurationStore = create<NodeConfigurationState>((set, get) => {
    function update(targetId: string, change: (entry: RoleEditorEntry) => RoleEditorEntry): void {
        const entry = get().editors.get(targetId)
        if (!entry) return
        const editors = new Map(get().editors)
        editors.set(targetId, change(entry))
        set({ editors })
    }

    return {
        editors: new Map(),

        adopt: (targetId, snapshot) => {
            const entry = get().editors.get(targetId)
            const next = baselineOf(snapshot)
            const editors = new Map(get().editors)
            if (!entry) {
                editors.set(targetId, { baseline: next, draft: next.roles, phase: 'draft' })
                set({ editors })
                return
            }
            if (entry.phase === 'sending') return
            if (isOlder(next.revision, entry.baseline.revision)) return
            if (
                next.revision === entry.baseline.revision &&
                next.policy === entry.baseline.policy
            ) {
                return
            }
            const edited = !sameNodeRoles(entry.draft, entry.baseline.roles)
            if (!edited) {
                editors.set(targetId, {
                    baseline: next,
                    draft: next.roles,
                    // "Saved" names the revision this editor saved, not a later one.
                    phase:
                        entry.phase === 'applied' && next.revision === entry.baseline.revision
                            ? 'applied'
                            : 'draft'
                })
            } else if (sameNodeRoles(next.roles, entry.baseline.roles)) {
                // Only other fields moved: the substitution keeps them, so the
                // draft simply rebases.
                editors.set(targetId, { ...entry, baseline: next })
            } else {
                editors.set(targetId, {
                    ...entry,
                    phase: 'conflict',
                    latest: next,
                    message: undefined
                })
            }
            set({ editors })
        },

        setRole: (targetId, tier, role) =>
            update(targetId, entry => {
                const draft: NodeRoles = { ...entry.draft }
                if (role) draft[tier] = role
                else delete draft[tier]
                return { ...entry, draft, phase: entry.phase === 'conflict' ? 'conflict' : 'draft' }
            }),

        setDraft: (targetId, roles) =>
            update(targetId, entry => ({
                ...entry,
                draft: { ...roles },
                phase: entry.phase === 'conflict' ? 'conflict' : 'draft'
            })),

        keepDraftOnLatest: targetId =>
            update(targetId, entry =>
                entry.latest
                    ? { baseline: entry.latest, draft: entry.draft, phase: 'draft' }
                    : entry
            ),

        discard: targetId =>
            update(targetId, entry => {
                const baseline = entry.latest ?? entry.baseline
                return { baseline, draft: baseline.roles, phase: 'draft' }
            }),

        save: async (targetId, nodeId) => {
            const entry = get().editors.get(targetId)
            if (!entry || entry.phase === 'sending' || entry.phase === 'conflict') return null
            const base = entry.baseline
            update(targetId, current => ({ ...current, phase: 'sending', message: undefined }))
            try {
                const result = await window.pairApi.nodePolicy.set({
                    nodeId,
                    policy: withRoles(base.policy, entry.draft),
                    expectedPersistedRevision: base.revision
                })
                const saved = baselineOf(result)
                update(targetId, () => ({ baseline: saved, draft: saved.roles, phase: 'applied' }))
                return result
            } catch (err) {
                const message = getErrorString(err)
                const refusal = policyRefusal(message)
                if (refusal === 'vetoed' || refusal === 'rejected') {
                    update(targetId, current => ({ ...current, phase: refusal, message }))
                    return null
                }
                // A conflict, or an outcome the reply does not settle: read
                // the target back and show what it holds. Never resend.
                const latest = await window.pairApi.nodePolicy.get(nodeId).catch(() => null)
                if (latest === null) {
                    update(targetId, current => ({ ...current, phase: 'unknown', message }))
                } else if (refusal === 'conflict' || latest.revision !== base.revision) {
                    update(targetId, current => ({
                        ...current,
                        phase: 'conflict',
                        latest: baselineOf(latest),
                        message
                    }))
                } else {
                    update(targetId, current => ({ ...current, phase: 'rejected', message }))
                }
                return null
            }
        }
    }
})
