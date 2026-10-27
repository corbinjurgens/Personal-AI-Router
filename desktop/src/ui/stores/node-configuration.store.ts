// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Node configuration drafts for the app session. Routing and model setup share
 * one whole-policy draft per node; the role editor and bulk panel retain their
 * focused role drafts alongside it. None of these drafts are persisted here.
 *
 * `policy:set` replaces the whole document and fills omitted fields with
 * defaults, so a role edit never sends a document of its own: it takes the
 * target's own projection as last read, substitutes `roles` only, and sends it
 * back checked against that target's revision. A refused or uncertain save is
 * read back and shown; nothing is retried automatically.
 *
 * Bulk configuration is a bounded fan-out of the same per-node operation. There
 * is no cluster-wide document: each target keeps its own baseline, result and
 * readback, and a node that is offline keeps an unsent draft.
 */
import { create } from 'zustand'
import type { EngineDescriptor, EngineModels, EngineType } from '@/shared/types/engines'
import type { JsonValue } from '@/shared/types/json'
import type { NodeAvailability, NodePolicySnapshot } from '@/shared/types/node-policy'
import type {
    NodePolicyAdvertisement,
    NodeRole,
    NodeRoles,
    PauseOrigin,
    RoleTier
} from '@/shared/types/node-roles'
import {
    INVALID_PARAMS_CODE,
    LOCAL_VETO_CODE,
    POLICY_CONFLICT_CODE,
    REMOTE_MANAGEMENT_DENIED_CODE,
    REMOTE_SETTING_LOCKED_CODE,
    localVetoMessage
} from '@/shared/constants/service-errors'
import { getPairTransport } from '@/ui/api/bootstrap'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { engineLabel, engineManagerName, engineTypeFromManagerName } from '@/shared/utils/engines'
import { serviceErrorCode } from '@/ui/api/service-transport'
import getErrorString from '@/shared/utils/get-error-string'
import { replaceJsonFieldSource } from '@/shared/utils/json-source'
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

/** Pause fences peer writes regardless of who requested it. */
export function nodePaused(policy: string, availability?: NodeAvailability): boolean {
    const body = policyObject(policy)
    return (
        availability === 'paused' ||
        availability === 'draining' ||
        body.availability === 'paused' ||
        body.availability === 'draining'
    )
}

export const NODE_PAUSED_MESSAGE =
    'This node is paused. Resume it on that node before making changes.'

/**
 * The target's own document with `roles` replaced by `roles` and every other
 * field kept as read. Unset tiers are omitted, which leaves them undeclared.
 */
export function withRoles(policy: string, roles: NodeRoles): string {
    const declared: JsonMap = {}
    for (const tier of ROLE_TIERS) {
        const role = roles[tier]
        if (role) declared[tier] = { engine: role.engine, model: role.model }
    }
    return replaceJsonFieldSource(policy, 'roles', JSON.stringify(declared))
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

/** The select value for an unset tier. */
export const UNSET_ROLE = 'unset'

/**
 * Select items for one tier: unset, then each installed choice. A declared
 * model the node no longer reports stays visible as the current value.
 */
export function roleItems(
    choices: RoleChoice[],
    current: NodeRole | undefined,
    descriptors: ReadonlyMap<EngineType, EngineDescriptor>
): { value: string; children: string }[] {
    const items = [
        { value: UNSET_ROLE, children: 'Unset' },
        ...choices.map(choice => ({ value: choice.key, children: roleLabel(choice, descriptors) }))
    ]
    if (current && !choices.some(choice => choice.key === roleKey(current))) {
        items.push({
            value: roleKey(current),
            children: `${roleLabel(current, descriptors)}, not installed`
        })
    }
    return items
}

/** The role a select value names, or undefined for unset or a value no longer offered. */
export function chosenRole(choices: RoleChoice[], value: string): NodeRole | undefined {
    const choice = choices.find(candidate => candidate.key === value)
    return choice ? { engine: choice.engine, model: choice.model } : undefined
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

/** How the service refused a save. */
type Refusal = 'conflict' | 'vetoed' | 'rejected' | 'unreachable' | 'other'

/**
 * Failures raised before the request reached the target: no engine manager to
 * relay through, a node that is not a reachable peer, or a connection that was
 * never made. A timeout or a reply lost after sending is not one of these.
 */
const UNREACHABLE =
    /engine manager unavailable|not a discovered ec peer|device is offline|is not running|connection refused|no route to host|network is unreachable/i

/**
 * Classify a refusal by the service's JSON-RPC `code`. Only a failure without
 * one (a call that never got a service's answer) is matched on its message.
 */
export function policyRefusal(message: string, code?: number): Refusal {
    if (code === POLICY_CONFLICT_CODE) return 'conflict'
    if (code === LOCAL_VETO_CODE) return 'vetoed'
    if (code === REMOTE_MANAGEMENT_DENIED_CODE) return 'rejected'
    if (code === REMOTE_SETTING_LOCKED_CODE) return 'rejected'
    if (code === INVALID_PARAMS_CODE) return 'rejected'
    if (code === undefined) {
        if (/-32010\b/.test(message) || message.includes('policy conflict')) return 'conflict'
        if (/-32011\b/.test(message)) return 'vetoed'
        if (/-32012\b/.test(message)) return 'rejected'
        if (/-32602\b/.test(message)) return 'rejected'
    }
    if (UNREACHABLE.test(message)) return 'unreachable'
    return 'other'
}

/** Where one node's role edit stands (the bulk UX contract's per-node states). */
type RoleEditPhase =
    | 'draft'
    | 'reading'
    | 'sending'
    | 'applied'
    | 'conflict'
    | 'vetoed'
    | 'rejected'
    | 'unreachable'
    | 'unknown'

interface RoleBaseline {
    policy: string
    revision: string
    roles: NodeRoles
    availability?: NodeAvailability
    pausedBy?: PauseOrigin
}

type PolicyBaselineInput = Pick<NodePolicySnapshot, 'policy' | 'revision'> & {
    availability?: NodeAvailability
}

export interface RoleEditorEntry {
    /**
     * The target's own document the draft substitutes into. Absent only while
     * the first read of the target is in flight or has failed; the draft then
     * holds nothing the user chose.
     */
    baseline?: RoleBaseline
    draft: NodeRoles
    phase: RoleEditPhase
    /** The document read back after a conflict, for the diff. */
    latest?: RoleBaseline
    /** The service's message for the last refused or uncertain save. */
    message?: string
    /**
     * Names one bulk attempt, from its read through its send and read back. A
     * retry is a new attempt. `policy:set` takes no request id, so it stays on
     * this side; the expected revision is what keeps a resend from landing twice.
     */
    requestId?: string
}

function baselineOf(snapshot: PolicyBaselineInput): RoleBaseline {
    return {
        policy: snapshot.policy,
        revision: snapshot.revision,
        roles: policyRoles(snapshot.policy),
        availability: snapshot.availability,
        pausedBy: pauseOriginValue(policyObject(snapshot.policy).pausedBy)
    }
}

function isOlder(revision: string, than: string): boolean {
    return BigInt(revision) < BigInt(than)
}

/** A target's state in the bulk panel: its edit phase, or `advertised` (see {@link targetState}). */
export type TargetState = RoleEditPhase | 'advertised'

/** The states a user can retry. */
export const FAILED_STATES: ReadonlySet<TargetState> = new Set<TargetState>([
    'conflict',
    'rejected',
    'vetoed',
    'unreachable',
    'unknown'
])

/**
 * A target's bulk state. An applied save becomes `advertised` once this node's
 * own discovery reports the target at that revision or a later one; until then
 * the target has persisted the roles but routers here may not use them yet. A
 * node never edited here is a draft.
 */
export function targetState(
    entry: RoleEditorEntry | undefined,
    advertised: NodePolicyAdvertisement | undefined
): TargetState {
    if (!entry) return 'draft'
    if (
        entry.phase === 'applied' &&
        entry.baseline &&
        advertised &&
        /^\d+$/.test(advertised.revision) &&
        !isOlder(advertised.revision, entry.baseline.revision)
    ) {
        return 'advertised'
    }
    return entry.phase
}

/** One target of a bulk send. */
export interface BulkTarget {
    /** The broker's id for the node. */
    targetId: string
    /** The id `policy:*` takes; undefined for this PC. */
    nodeId: string | undefined
}

/** The most per-node operations a bulk send keeps in flight. */
export const BULK_CONCURRENCY = 4

/** Run `operation` for every item, never more than `limit` at once. */
async function runBounded<T>(
    items: readonly T[],
    limit: number,
    operation: (item: T) => Promise<void>
): Promise<void> {
    let next = 0
    const worker = async () => {
        while (next < items.length) {
            const item = items[next]
            next += 1
            await operation(item)
        }
    }
    await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker))
}

/** Same roles on every selected node, or a draft of its own for each. */
type BulkMode = 'common' | 'per-node'

/** The bulk panel's choices, kept for the app session with the drafts. */
interface BulkSelection {
    /** Selected targets, by the broker's node id. */
    selected: string[]
    mode: BulkMode
    /** The roles every selected node is given in `common` mode. */
    common: NodeRoles
}

interface NodeConfigurationState {
    /** Whole-policy drafts are session-only and have one visible editor owner. */
    policyDrafts: Map<string, PolicyDraft>
    claimPolicyDraft(targetId: string, owner: PolicyDraft['owner'], discard?: boolean): boolean
    editPolicyDraft(targetId: string, owner: PolicyDraft['owner'], text: string): void
    /** Pushes preserve the draft's CAS base; a refresh/readback confirms conflicts. */
    observePolicyDraft(
        targetId: string,
        snapshot: PolicyDraftSnapshot,
        source?: 'read' | 'push'
    ): void
    observePolicyAvailability(targetId: string, availability: NodeAvailability): void
    readPolicyDraft(targetId: string, owner: PolicyDraft['owner'], nodeId?: string): Promise<void>
    savePolicyDraft(targetId: string, owner: PolicyDraft['owner'], nodeId?: string): Promise<void>
    invalidatePolicyDraft(targetId: string): void
    management: Map<string, NodeManagementEntry>
    readManagement(targetId: string, nodeId: string | undefined, fresh?: boolean): Promise<void>
    disconnectManagement(targetId: string): void
    /** Keyed by the broker's node id; this PC's is the self id. */
    editors: Map<string, RoleEditorEntry>
    /**
     * Record the target's document as read or reported. A newer document
     * replaces the baseline when the draft is untouched or its roles did not
     * change; otherwise the draft is stale and shown against it.
     */
    adopt(targetId: string, snapshot: PolicyBaselineInput): void
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
    /** Take a first baseline for a target that has none, for its preview. */
    read(targetId: string, nodeId: string | undefined): Promise<void>
    /**
     * One bulk attempt on one target: read the target's own document, then send
     * the draft over it. With `roles`, those roles are substituted into the
     * document just read (the same roles on every node); without, the target's
     * own draft is checked against its baseline, and a change to its roles
     * stops at a conflict. A target already holding the roles is applied
     * without a send. Never throws; the outcome is the entry's phase.
     */
    apply(targetId: string, nodeId: string | undefined, roles?: NodeRoles): Promise<void>
    /**
     * {@link apply} for each target, at most {@link BULK_CONCURRENCY} at once.
     * `onSettled` is told as each target's attempt ends, for progress.
     */
    applyAll(
        targets: readonly BulkTarget[],
        roles?: NodeRoles,
        onSettled?: (target: BulkTarget) => void
    ): Promise<void>
    bulk: BulkSelection
    setBulk(change: Partial<BulkSelection>): void
}

type PolicyDraftSnapshot = NodePolicySnapshot & {
    availability?: NodeAvailability
    recoveryRequired?: boolean
}

interface PolicyDraft {
    owner: 'Routing' | 'Model Manager'
    text: string
    baseline?: PolicyDraftSnapshot
    saved?: PolicyDraftSnapshot
    phase: 'draft' | 'reading' | 'saving' | 'saved' | 'conflict' | 'error' | 'unknown'
    message?: string
    failure?: { code?: number; detail: string }
    generation: number
    /** Cleared on disconnect, so cached authority never enables a mutation. */
    current: boolean
    /** Notifications can precede the first saved policy snapshot. */
    availabilityObservation?: { value: NodeAvailability }
}

interface NodeManagementEntry {
    state:
        | 'local'
        | 'loading'
        | 'allowed'
        | 'denied'
        | 'unreviewed'
        | 'unavailable'
        | 'unsupported'
        | 'unreachable'
    revision: string
    message?: string
}

const MANAGEMENT_MESSAGES: Partial<Record<NodeManagementEntry['state'], string>> = {
    loading: 'Checking remote management permission on this device.',
    denied: 'This device has blocked remote management. Allow it on that device to make changes.',
    unreviewed:
        'This device has not granted remote management. Allow it on that device to make changes.',
    unavailable:
        'Remote management permission is unavailable. Changes are disabled until it can be confirmed.',
    unsupported:
        'This device does not support remote management permissions. Update it before making remote changes.',
    unreachable:
        'Device disconnected or unreachable. Reconnect to confirm remote management permission.'
}

interface ManagementTarget {
    nodeId: string | undefined
    generation: number
    consumers: number
    request: Promise<void> | null
    refreshAfterRead: boolean
}

const managementTargets = new Map<string, ManagementTarget>()
let managementUnsubs: Array<() => void> = []
let managementContextUnsubscribe: (() => void) | null = null

/** Successful remote mutations refresh open editors without creating new watches. */
export function refreshNodeManagement(nodeId: string | undefined): void {
    if (!nodeId) return
    for (const [id, target] of managementTargets) {
        if (target.consumers === 0 || target.nodeId !== nodeId) continue
        void useNodeConfigurationStore.getState().readManagement(id, nodeId, true)
    }
}

function managementTarget(targetId: string, nodeId: string | undefined): ManagementTarget {
    const existing = managementTargets.get(targetId)
    if (existing && existing.nodeId === nodeId) return existing
    if (existing) useNodeConfigurationStore.getState().disconnectManagement(targetId)
    const target: ManagementTarget = {
        nodeId,
        generation: 0,
        consumers: 0,
        request: null,
        refreshAfterRead: false
    }
    managementTargets.set(targetId, target)
    return target
}

/** Open editors share target reads and transport listeners; reconnect only reads. */
export function watchNodeManagement(targetId: string, nodeId: string | undefined): () => void {
    const target = managementTarget(targetId, nodeId)
    target.consumers++
    if (!managementContextUnsubscribe)
        managementContextUnsubscribe = useClusterInvitationsStore.subscribe((state, previous) => {
            if (state.remoteManagementContext === previous.remoteManagementContext) return
            for (const [id, watched] of managementTargets) {
                if (watched.consumers === 0 || watched.nodeId === undefined) continue
                useNodeConfigurationStore.getState().disconnectManagement(id)
                if (
                    state.remoteManagementContext !== null &&
                    state.members.some(
                        member => member.nodeUuid === watched.nodeId && member.state === 'member'
                    )
                )
                    void useNodeConfigurationStore.getState().readManagement(id, watched.nodeId)
            }
        })
    if (managementUnsubs.length === 0) {
        const transport = getPairTransport()
        if (transport)
            managementUnsubs = [
                transport.onDisconnect(() => {
                    for (const id of managementTargets.keys())
                        useNodeConfigurationStore.getState().disconnectManagement(id)
                }),
                transport.onConnect(() => {
                    for (const [id, watched] of managementTargets)
                        if (watched.consumers > 0) {
                            useNodeConfigurationStore.getState().disconnectManagement(id)
                            void useNodeConfigurationStore
                                .getState()
                                .readManagement(id, watched.nodeId)
                        }
                })
            ]
    }
    if (target.consumers === 1)
        void useNodeConfigurationStore.getState().readManagement(targetId, nodeId)
    let stopped = false
    return () => {
        if (stopped) return
        stopped = true
        target.consumers--
        if (target.consumers === 0 && managementTargets.get(targetId) === target) {
            useNodeConfigurationStore.getState().disconnectManagement(targetId)
            managementTargets.delete(targetId)
        }
        if (![...managementTargets.values()].some(watched => watched.consumers > 0)) {
            managementUnsubs.forEach(stop => stop())
            managementUnsubs = []
            managementContextUnsubscribe?.()
            managementContextUnsubscribe = null
        }
    }
}

/** Subscribe before the first read; closing or reconnecting invalidates late replies. */
export function watchModelSettingsDraft(targetId: string, nodeId: string | undefined): () => void {
    const store = () => useNodeConfigurationStore.getState()
    const offPolicy = window.pairApi.nodePolicy.onPolicyChanged(change => {
        if (change.nodeId === targetId) store().observePolicyDraft(targetId, change)
    })
    const offAvailability = window.pairApi.nodePolicy.onAvailabilityChanged(change => {
        if (change.nodeId === targetId)
            store().observePolicyAvailability(targetId, change.availability)
    })
    const offConnection = useConnectionStore.subscribe((state, previous) => {
        if (state.connected === previous.connected) return
        store().invalidatePolicyDraft(targetId)
        if (state.connected) void store().readPolicyDraft(targetId, 'Model Manager', nodeId)
    })
    if (useConnectionStore.getState().connected)
        void store().readPolicyDraft(targetId, 'Model Manager', nodeId)
    return () => {
        offPolicy()
        offAvailability()
        offConnection()
        if (store().policyDrafts.get(targetId)?.owner === 'Model Manager')
            store().invalidatePolicyDraft(targetId)
    }
}

export const useNodeConfigurationStore = create<NodeConfigurationState>((set, get) => {
    function putPolicy(targetId: string, entry: PolicyDraft): void {
        const policyDrafts = new Map(get().policyDrafts)
        policyDrafts.set(targetId, entry)
        set({ policyDrafts })
    }
    function update(targetId: string, change: (entry: RoleEditorEntry) => RoleEditorEntry): void {
        const entry = get().editors.get(targetId)
        if (!entry) return
        put(targetId, change(entry))
    }

    function put(targetId: string, entry: RoleEditorEntry): void {
        const editors = new Map(get().editors)
        editors.set(targetId, entry)
        set({ editors })
    }

    return {
        policyDrafts: new Map(),

        claimPolicyDraft: (targetId, owner, discard = false) => {
            const entry = get().policyDrafts.get(targetId)
            if (entry?.phase === 'saving') return false
            if (!discard && entry) {
                if (entry.owner === owner) return true
                if (entry.text !== (entry.baseline?.policy ?? '')) return false
            }
            putPolicy(targetId, {
                owner,
                text: entry?.saved?.policy ?? '',
                baseline: entry?.saved,
                saved: entry?.saved,
                phase: 'draft',
                generation: (entry?.generation ?? 0) + 1,
                current: false
            })
            return true
        },

        editPolicyDraft: (targetId, owner, text) => {
            const entry = get().policyDrafts.get(targetId)
            if (!entry || entry.owner !== owner || entry.phase === 'saving') return
            putPolicy(targetId, { ...entry, text })
        },

        observePolicyDraft: (targetId, snapshot, source = 'push') => {
            const entry = get().policyDrafts.get(targetId)
            if (!entry || (entry.saved && isOlder(snapshot.revision, entry.saved.revision))) return
            const dirty = entry.text !== (entry.baseline?.policy ?? '')
            putPolicy(targetId, {
                ...entry,
                saved: { ...entry.saved, ...snapshot },
                ...(dirty
                    ? snapshot.revision !== entry.baseline?.revision
                        ? {
                              phase:
                                  source === 'read' && entry.phase !== 'saving'
                                      ? 'conflict'
                                      : entry.phase,
                              message:
                                  entry.failure === undefined
                                      ? 'The saved policy changed. Your draft keeps its original revision.'
                                      : entry.message
                          }
                        : {}
                    : { text: snapshot.policy, baseline: snapshot }),
                current: true
            })
        },

        observePolicyAvailability: (targetId, availability) => {
            const entry = get().policyDrafts.get(targetId)
            if (!entry) return
            putPolicy(targetId, {
                ...entry,
                availabilityObservation: { value: availability },
                saved: entry.saved ? { ...entry.saved, availability } : undefined
            })
        },

        invalidatePolicyDraft: targetId => {
            const entry = get().policyDrafts.get(targetId)
            if (entry)
                putPolicy(targetId, {
                    ...entry,
                    generation: entry.generation + 1,
                    current: false,
                    phase: entry.phase === 'saving' ? 'unknown' : 'draft',
                    message:
                        entry.phase === 'saving'
                            ? 'Save outcome unknown. Read the target before deciding whether to retry.'
                            : entry.message
                })
        },

        readPolicyDraft: async (targetId, owner, nodeId) => {
            if (!get().claimPolicyDraft(targetId, owner)) return
            const entry = get().policyDrafts.get(targetId)
            if (!entry) return
            const generation = entry.generation + 1
            putPolicy(targetId, { ...entry, generation, phase: 'reading' })
            try {
                const snapshot = await window.pairApi.nodePolicy.get(nodeId)
                const current = get().policyDrafts.get(targetId)
                if (current?.generation !== generation || current.owner !== owner) return
                get().observePolicyDraft(
                    targetId,
                    {
                        ...snapshot,
                        availability:
                            current.availabilityObservation !== entry.availabilityObservation &&
                            current.availabilityObservation
                                ? current.availabilityObservation.value
                                : current.saved !== entry.saved &&
                                    current.saved?.availability !== undefined
                                  ? current.saved.availability
                                  : snapshot.availability
                    },
                    'read'
                )
                const observed = get().policyDrafts.get(targetId)
                if (observed)
                    putPolicy(targetId, {
                        ...observed,
                        phase:
                            observed.saved?.revision !== observed.baseline?.revision &&
                            observed.text !== (observed.baseline?.policy ?? '')
                                ? 'conflict'
                                : 'draft'
                    })
            } catch (error) {
                const current = get().policyDrafts.get(targetId)
                if (current?.generation === generation && current.owner === owner)
                    putPolicy(targetId, {
                        ...current,
                        phase: 'error',
                        current: false,
                        message: getErrorString(error)
                    })
            }
        },

        savePolicyDraft: async (targetId, owner, nodeId) => {
            const entry = get().policyDrafts.get(targetId)
            if (
                !entry ||
                entry.owner !== owner ||
                !entry.baseline ||
                !entry.current ||
                !useConnectionStore.getState().connected ||
                entry.phase === 'reading' ||
                entry.phase === 'saving' ||
                entry.phase === 'conflict' ||
                entry.phase === 'unknown' ||
                (nodeId !== undefined &&
                    (get().management.get(targetId)?.state !== 'allowed' ||
                        nodePaused(entry.saved?.policy ?? '', entry.saved?.availability)))
            )
                return
            const generation = entry.generation + 1
            putPolicy(targetId, {
                ...entry,
                generation,
                phase: 'saving',
                message: undefined,
                failure: undefined
            })
            try {
                const result = await window.pairApi.nodePolicy.set({
                    nodeId,
                    policy: entry.text,
                    expectedPersistedRevision: entry.baseline.revision
                })
                const current = get().policyDrafts.get(targetId)
                if (current?.generation !== generation || current.owner !== owner) return
                if (current.saved && isOlder(result.revision, current.saved.revision)) {
                    putPolicy(targetId, {
                        ...current,
                        phase: 'conflict',
                        message:
                            'A newer policy was reported during Save. Discard and reload to review it.'
                    })
                    return
                }
                const saved = { ...current.saved, ...result }
                putPolicy(targetId, {
                    ...current,
                    text: result.policy,
                    baseline: saved,
                    saved,
                    phase: 'saved',
                    message:
                        result.activation?.state === 'pending'
                            ? 'Saved, activation pending.'
                            : result.activation?.state === 'error'
                              ? `Saved, activation error: ${result.activation.error ?? 'not enforced'}`
                              : 'Saved. Generation settings affect subsequent requests.'
                })
                get().adopt(targetId, saved)
            } catch (error) {
                const current = get().policyDrafts.get(targetId)
                if (current?.generation !== generation || current.owner !== owner) return
                const code = serviceErrorCode(error)
                const detail = getErrorString(error)
                const refusal = policyRefusal(detail, code)
                const message =
                    code === REMOTE_SETTING_LOCKED_CODE
                        ? 'Generation settings are locked on the target. Your draft is kept.'
                        : code === REMOTE_MANAGEMENT_DENIED_CODE
                          ? 'This device refused remote management permission. Your policy draft is kept. Change the permission on that device before trying again.'
                          : code === LOCAL_VETO_CODE
                            ? `${localVetoMessage('save this policy')} ${detail}`
                            : refusal === 'other'
                              ? `Save outcome unknown. Your policy text is kept. ${detail}`
                              : `The target ${refusal === 'unreachable' ? 'is disconnected or unreachable' : 'refused this policy'}. Your policy text is kept. ${detail}`
                putPolicy(targetId, {
                    ...current,
                    phase:
                        refusal === 'conflict'
                            ? 'conflict'
                            : refusal === 'other'
                              ? 'unknown'
                              : 'error',
                    message,
                    failure: { code, detail }
                })
                if (code === REMOTE_MANAGEMENT_DENIED_CODE)
                    void get().readManagement(targetId, nodeId, true)
                if (
                    code === REMOTE_SETTING_LOCKED_CODE ||
                    code === REMOTE_MANAGEMENT_DENIED_CODE ||
                    code === LOCAL_VETO_CODE
                )
                    return
                // Readback is observation only. A lost reply never triggers a resend.
                try {
                    const beforeRead = get().policyDrafts.get(targetId)?.saved
                    const latest = await window.pairApi.nodePolicy.get(nodeId)
                    const afterRead = get().policyDrafts.get(targetId)
                    if (afterRead?.generation !== generation) return
                    get().observePolicyDraft(
                        targetId,
                        {
                            ...latest,
                            availability:
                                afterRead.saved !== beforeRead &&
                                afterRead.saved?.availability !== undefined
                                    ? afterRead.saved.availability
                                    : latest.availability
                        },
                        'read'
                    )
                    const observed = get().policyDrafts.get(targetId)
                    if (observed && observed.saved?.revision === entry.baseline.revision)
                        putPolicy(targetId, {
                            ...observed,
                            phase: 'error',
                            message:
                                refusal === 'other'
                                    ? `The saved policy is unchanged. Your policy text is kept. Service message: ${detail}`
                                    : message
                        })
                } catch {
                    // Keep the original refusal or uncertain outcome and its draft.
                    const unread = get().policyDrafts.get(targetId)
                    if (unread?.generation === generation)
                        putPolicy(targetId, {
                            ...unread,
                            current: false,
                            message: `Could not confirm whether the save took effect: ${detail} Your policy text is kept. Reload to see the saved policy.`
                        })
                }
            }
        },

        management: new Map(),

        disconnectManagement: targetId => {
            const target = managementTargets.get(targetId)
            if (target) {
                target.generation++
                target.request = null
                target.refreshAfterRead = false
            }
            const management = new Map(get().management)
            management.set(targetId, {
                state: 'unreachable',
                revision: '',
                message: MANAGEMENT_MESSAGES.unreachable
            })
            set({ management })
        },

        readManagement: (targetId, nodeId, fresh = false) => {
            const target = managementTarget(targetId, nodeId)
            const currentGeneration = target.generation
            const current = () =>
                managementTargets.get(targetId) === target &&
                target.generation === currentGeneration
            if (target.request) {
                if (!fresh) return target.request
                target.refreshAfterRead = true
                // The pending read's finalizer starts the coalesced fresh read.
                return target.request.then(() =>
                    current() ? (target.request ?? undefined) : undefined
                )
            }
            const putManagement = (entry: NodeManagementEntry) => {
                const management = new Map(get().management)
                management.set(targetId, {
                    ...entry,
                    message: entry.message ?? MANAGEMENT_MESSAGES[entry.state]
                })
                set({ management })
            }
            if (nodeId === undefined) {
                putManagement({ state: 'local', revision: '' })
                return Promise.resolve()
            }
            if (getPairTransport()?.connected === false) {
                get().disconnectManagement(targetId)
                return Promise.resolve()
            }
            const previous = get().management.get(targetId)
            putManagement({ state: 'loading', revision: previous?.revision ?? '' })
            target.request = Promise.resolve()
                .then(() =>
                    current() ? window.pairApi.cluster.getRemoteManagementForNode(nodeId) : null
                )
                .then(capability => {
                    if (!current() || !capability) return
                    if (
                        capability.revision &&
                        previous?.revision &&
                        BigInt(capability.revision) < BigInt(previous.revision)
                    ) {
                        putManagement({ state: 'unavailable', revision: previous.revision })
                        return
                    }
                    const consistent = capability.allowed === (capability.state === 'allowed')
                    const state =
                        !consistent || (capability.state === 'allowed' && !capability.revision)
                            ? 'unavailable'
                            : capability.state
                    putManagement({ state, revision: capability.revision })
                })
                .catch(error => {
                    if (current())
                        putManagement({
                            state: 'unreachable',
                            revision: '',
                            message: getErrorString(error)
                        })
                })
                .finally(() => {
                    if (!current()) return
                    target.request = null
                    if (target.refreshAfterRead) {
                        target.refreshAfterRead = false
                        if (target.consumers > 0) void get().readManagement(targetId, nodeId)
                    }
                })
            return target.request
        },

        editors: new Map(),

        adopt: (targetId, snapshot) => {
            const entry = get().editors.get(targetId)
            const next = baselineOf(snapshot)
            const editors = new Map(get().editors)
            if (!entry?.baseline) {
                editors.set(targetId, { baseline: next, draft: next.roles, phase: 'draft' })
                set({ editors })
                return
            }
            if (entry.phase === 'sending') return
            if (isOlder(next.revision, entry.baseline.revision)) return
            if (
                next.availability === undefined &&
                next.revision === entry.baseline.revision &&
                next.policy === entry.baseline.policy
            ) {
                // Policy pushes omit live availability. The same saved
                // document cannot invalidate a draining policy:get readback.
                next.availability = entry.baseline.availability
            }
            if (
                next.revision === entry.baseline.revision &&
                next.policy === entry.baseline.policy &&
                next.availability === entry.baseline.availability
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
                return {
                    ...entry,
                    draft,
                    phase:
                        entry.phase === 'conflict' || entry.phase === 'vetoed'
                            ? entry.phase
                            : 'draft'
                }
            }),

        setDraft: (targetId, roles) =>
            update(targetId, entry => ({
                ...entry,
                draft: { ...roles },
                phase:
                    entry.phase === 'conflict' || entry.phase === 'vetoed' ? entry.phase : 'draft'
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
                return baseline ? { baseline, draft: baseline.roles, phase: 'draft' } : entry
            }),

        save: async (targetId, nodeId) => {
            const entry = get().editors.get(targetId)
            const base = entry?.baseline
            if (!entry || !base || entry.phase === 'sending' || entry.phase === 'conflict') {
                return null
            }
            if (nodeId !== undefined && nodePaused(base.policy, base.availability)) {
                update(targetId, current => ({
                    ...current,
                    phase: 'vetoed',
                    message: NODE_PAUSED_MESSAGE
                }))
                return null
            }
            if (entry.phase === 'vetoed') {
                // A refusal keeps its draft. Read again for the preview; a
                // later explicit save uses that revision, never this attempt.
                try {
                    const latest = await window.pairApi.nodePolicy.get(nodeId)
                    get().adopt(targetId, latest)
                    update(targetId, current => ({
                        ...current,
                        phase:
                            current.phase === 'conflict'
                                ? 'conflict'
                                : nodeId !== undefined &&
                                    nodePaused(latest.policy, latest.availability)
                                  ? 'vetoed'
                                  : 'draft',
                        message:
                            nodeId !== undefined && nodePaused(latest.policy, latest.availability)
                                ? NODE_PAUSED_MESSAGE
                                : undefined
                    }))
                } catch (err) {
                    update(targetId, current => ({ ...current, message: getErrorString(err) }))
                }
                return null
            }
            update(targetId, current => ({ ...current, phase: 'sending', message: undefined }))
            try {
                const result = await window.pairApi.nodePolicy.set({
                    nodeId,
                    policy: withRoles(base.policy, entry.draft),
                    expectedPersistedRevision: base.revision
                })
                const saved = baselineOf(result)
                update(targetId, current => ({
                    baseline: saved,
                    draft: saved.roles,
                    phase: 'applied',
                    requestId: current.requestId
                }))
                return result
            } catch (err) {
                const message = getErrorString(err)
                const code = serviceErrorCode(err)
                const refusal = policyRefusal(message, code)
                const explanation =
                    code === REMOTE_MANAGEMENT_DENIED_CODE ||
                    (code === undefined && /-32012\b/.test(message))
                        ? `Remote management permission was refused by this device. ${message}`
                        : refusal === 'vetoed'
                          ? NODE_PAUSED_MESSAGE
                          : message
                // Vetoed, rejected and unreachable are definite: the target
                // refused it, or it never got there. The draft is kept.
                if (refusal === 'vetoed' || refusal === 'rejected' || refusal === 'unreachable') {
                    update(targetId, current => ({
                        ...current,
                        phase: refusal,
                        message: explanation
                    }))
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
        },

        read: async (targetId, nodeId) => {
            const prior = get().editors.get(targetId)
            if (prior?.baseline || prior?.phase === 'reading') return
            put(targetId, { draft: {}, phase: 'reading' })
            try {
                const latest = await window.pairApi.nodePolicy.get(nodeId)
                // Another reader may have taken a baseline meanwhile; it stands.
                if (get().editors.get(targetId)?.baseline) return
                const next = baselineOf(latest)
                put(targetId, { baseline: next, draft: next.roles, phase: 'draft' })
            } catch (err) {
                put(targetId, { draft: {}, phase: 'unreachable', message: getErrorString(err) })
            }
        },

        apply: async (targetId, nodeId, roles) => {
            const prior = get().editors.get(targetId)
            if (prior?.phase === 'reading' || prior?.phase === 'sending') return
            // Without roles of its own to send, an unread target is only read.
            if (roles === undefined && !prior?.baseline) return get().read(targetId, nodeId)
            const requestId = crypto.randomUUID()
            put(targetId, {
                ...(prior ?? { draft: {} }),
                phase: 'reading',
                message: undefined,
                requestId
            })
            if (nodeId !== undefined) {
                await get().readManagement(targetId, nodeId)
                const management = get().management.get(targetId)
                if (management?.state !== 'allowed') {
                    update(targetId, current => ({
                        ...current,
                        draft: roles ?? current.draft,
                        phase: 'rejected',
                        message: management?.message ?? MANAGEMENT_MESSAGES.unavailable
                    }))
                    return
                }
            }
            let latest: NodePolicySnapshot
            try {
                latest = await window.pairApi.nodePolicy.get(nodeId)
            } catch (err) {
                // Nothing was sent: the draft stays for an explicit retry.
                update(targetId, current => ({
                    ...current,
                    phase: 'unreachable',
                    message: getErrorString(err)
                }))
                return
            }
            const next = baselineOf(latest)
            const draft = roles ?? get().editors.get(targetId)?.draft ?? {}
            if (sameNodeRoles(draft, next.roles)) {
                // Also settles an unknown outcome whose save did land.
                put(targetId, { baseline: next, draft, phase: 'applied', requestId })
                return
            }
            if (roles !== undefined) {
                put(targetId, { baseline: next, draft, phase: 'reading', requestId })
            } else {
                get().adopt(targetId, latest)
                update(targetId, current => ({ ...current, requestId }))
            }
            const entry = get().editors.get(targetId)
            if (!entry?.baseline || entry.phase === 'conflict') return
            if (sameNodeRoles(entry.draft, entry.baseline.roles)) {
                // The draft was never edited and simply followed the node.
                update(targetId, current => ({ ...current, phase: 'draft' }))
                return
            }
            if (nodeId !== undefined && nodePaused(next.policy, next.availability)) {
                // Both pause origins and an incomplete drain fence every peer.
                update(targetId, current => ({
                    ...current,
                    phase: 'vetoed',
                    message: NODE_PAUSED_MESSAGE
                }))
                return
            }
            await get().save(targetId, nodeId)
        },

        applyAll: (targets, roles, onSettled) =>
            runBounded(targets, BULK_CONCURRENCY, async target => {
                await get().apply(target.targetId, target.nodeId, roles)
                onSettled?.(target)
            }),

        bulk: { selected: [], mode: 'common', common: {} },

        setBulk: change => set({ bulk: { ...get().bulk, ...change } })
    }
})
