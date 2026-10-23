// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState, type ReactNode } from 'react'
import {
    Badge,
    Button,
    Checkbox,
    Flex,
    ProgressBar,
    SegmentedControl,
    Select,
    Stack,
    Table,
    Text,
    type BadgeProps,
    type TableRowDefinition
} from '@nvidia/foundations-react-core'
import { useShallow } from 'zustand/react/shallow'
import type { EngineDescriptor, EngineType } from '@/shared/types/engines'
import type {
    NodePolicyAdvertisement,
    NodeRole,
    NodeRoles,
    RoleTier
} from '@/shared/types/node-roles'
import type { NodeItem } from '@/shared/types/nodes'
import { ROLE_TIERS, sameNodeRoles } from '@/shared/utils/node-roles'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import {
    chosenRole,
    FAILED_STATES,
    inventoryChoices,
    NODE_PAUSED_MESSAGE,
    roleDifferences,
    roleItems,
    roleKey,
    roleLabel,
    targetState,
    UNSET_ROLE,
    useNodeConfigurationStore,
    type BulkTarget,
    type RoleEditorEntry,
    type TargetState
} from '@/ui/stores/node-configuration.store'
import { useNodesStore } from '@/ui/stores/nodes.store'

const STATE_BADGES: Record<TargetState, { label: string; color: BadgeProps['color'] }> = {
    draft: { label: 'Draft', color: 'gray' },
    reading: { label: 'Reading', color: 'blue' },
    sending: { label: 'Sending', color: 'blue' },
    applied: { label: 'Applied', color: 'teal' },
    advertised: { label: 'Advertised', color: 'green' },
    conflict: { label: 'Conflict', color: 'yellow' },
    rejected: { label: 'Rejected', color: 'red' },
    vetoed: { label: 'Vetoed', color: 'yellow' },
    unreachable: { label: 'Unreachable', color: 'red' },
    unknown: { label: 'Outcome unknown', color: 'yellow' }
}

type Descriptors = ReadonlyMap<EngineType, EngineDescriptor>

function describeRole(role: NodeRole | undefined, descriptors: Descriptors): string {
    return role ? roleLabel(role, descriptors) : 'unset'
}

function describeRoles(roles: NodeRoles, descriptors: Descriptors): string {
    return ROLE_TIERS.map(tier => `${tier}: ${describeRole(roles[tier], descriptors)}`).join('\n')
}

/** One paired node as the panel shows it. */
interface TargetRow {
    node: NodeItem
    offline: boolean
    state: TargetState
    advertisement?: NodePolicyAdvertisement
}

/** What the state means for this target, in a sentence or two. */
function stateDetail(
    row: TargetRow,
    entry: RoleEditorEntry | undefined,
    descriptors: Descriptors
): string {
    const message = entry?.message ? ` Service message: ${entry.message}` : ''
    switch (row.state) {
        case 'draft':
            return row.offline
                ? 'Offline. The draft stays here and is not sent when the node returns.'
                : ''
        case 'reading':
            return "Reading the node's own policy."
        case 'sending':
            return 'Sending.'
        case 'applied':
            return `Saved at revision ${entry?.baseline?.revision ?? ''}; this PC's discovery has not reported it yet.`
        case 'advertised':
            return `Advertised at revision ${row.advertisement?.revision ?? ''}.`
        case 'conflict': {
            const latest = entry?.latest
            if (!entry || !latest) return `The node's policy changed before the send.${message}`
            const differences = roleDifferences(entry.draft, latest.roles)
            const diff = differences
                .map(
                    difference =>
                        `${difference.tier}: yours ${describeRole(difference.mine, descriptors)}, node ${describeRole(difference.theirs, descriptors)}`
                )
                .join('\n')
            return `The node's policy changed to revision ${latest.revision}, so nothing was sent.\n${diff || 'Its roles match this draft.'}${message}`
        }
        case 'rejected':
            return `The node refused these roles; change the draft.${message}`
        case 'vetoed':
            return `${NODE_PAUSED_MESSAGE}${entry?.baseline?.pausedBy ? ` PausedBy: ${entry.baseline.pausedBy}.` : ''}`
        case 'unreachable':
            return `Not sent; the draft is kept.${message}`
        case 'unknown':
            return `Sent, but the reply was lost and the node could not be read back. Retry reads it first and never resends blindly.${message}`
    }
}

/**
 * Roles for several paired nodes at once: the same roles on each, or a draft
 * per node. Each target is read, previewed, sent and read back on its own, a
 * few at a time; one failing leaves the others as they landed.
 */
export default function BulkNodeConfiguration() {
    const selfId = useConnectionStore(state => state.selfId)
    const peers = useNodesStore(
        useShallow(state => Array.from(state.nodes.values()).filter(node => node.id !== selfId))
    )
    const editors = useNodeConfigurationStore(state => state.editors)
    const bulk = useNodeConfigurationStore(state => state.bulk)
    const discovered = useDiscoveredNodesStore(state => state.nodes)
    const descriptors = useEngineStatusStore(state => state.descriptors)
    const models = useEngineModelsStore(state => state.models)
    const [progress, setProgress] = useState<{ done: number; total: number } | null>(null)
    const store = useNodeConfigurationStore.getState()

    const selected = useMemo(() => new Set(bulk.selected), [bulk.selected])
    const perNode = bulk.mode === 'per-node'

    // A selected node that is reachable is read for its preview. Reading
    // never sends; nothing here sends without the user pressing a button.
    useEffect(() => {
        for (const node of peers) {
            if (selected.has(node.id) && node.status !== 'offline' && !editors.has(node.id)) {
                void useNodeConfigurationStore.getState().read(node.id, node.id)
            }
        }
    }, [peers, selected, editors])

    const choicesFor = (ids: ReadonlySet<string>) =>
        inventoryChoices(Array.from(models.values()).filter(list => ids.has(list.nodeId)))
    const commonChoices = choicesFor(selected)

    const rows: TargetRow[] = peers.map(node => {
        const advertisement = discovered.find(candidate => candidate.id === node.id)?.policy
        return {
            node,
            offline: node.status === 'offline',
            state: targetState(editors.get(node.id), advertisement),
            advertisement
        }
    })

    const running = progress !== null
    const ready = rows.filter(
        row =>
            selected.has(row.node.id) &&
            !row.offline &&
            row.state !== 'reading' &&
            row.state !== 'sending'
    )
    const toSend = perNode
        ? ready.filter(row => {
              const entry = editors.get(row.node.id)
              return (
                  entry?.baseline !== undefined &&
                  entry.phase !== 'conflict' &&
                  !sameNodeRoles(entry.draft, entry.baseline.roles)
              )
          })
        : ready
    const toRetry = ready.filter(row => FAILED_STATES.has(row.state))

    const run = async (targetRows: TargetRow[]) => {
        const targets: BulkTarget[] = targetRows.map(row => ({
            targetId: row.node.id,
            nodeId: row.node.id
        }))
        setProgress({ done: 0, total: targets.length })
        try {
            await store.applyAll(targets, perNode ? undefined : bulk.common, () =>
                setProgress(current => current && { ...current, done: current.done + 1 })
            )
        } finally {
            setProgress(null)
        }
    }

    const toggle = (id: string, checked: boolean) =>
        store.setBulk({
            selected: checked
                ? [...bulk.selected.filter(other => other !== id), id]
                : bulk.selected.filter(other => other !== id)
        })

    const setCommon = (tier: RoleTier, value: string) => {
        const common: NodeRoles = { ...bulk.common }
        if (value === UNSET_ROLE) delete common[tier]
        else {
            const role = chosenRole(commonChoices, value)
            if (!role) return
            common[tier] = role
        }
        store.setBulk({ common })
    }

    const setOwn = (id: string, tier: RoleTier, value: string) => {
        if (value === UNSET_ROLE) {
            store.setRole(id, tier, undefined)
            return
        }
        const role = chosenRole(choicesFor(new Set([id])), value)
        if (role) store.setRole(id, tier, role)
    }

    const currentCell = (row: TargetRow): string => {
        const baseline = editors.get(row.node.id)?.baseline
        if (baseline) return describeRoles(baseline.roles, descriptors)
        if (row.advertisement) {
            return `${describeRoles(row.advertisement.roles, descriptors)}\n(as advertised)`
        }
        return row.offline ? 'Unknown while offline' : 'Not read yet'
    }

    const newCell = (row: TargetRow): ReactNode => {
        const entry = editors.get(row.node.id)
        if (!perNode) {
            const installed = new Set(choicesFor(new Set([row.node.id])).map(choice => choice.key))
            return ROLE_TIERS.map(tier => {
                const role = bulk.common[tier]
                const missing = role && !installed.has(roleKey(role)) ? ', not installed here' : ''
                return `${tier}: ${describeRole(role, descriptors)}${missing}`
            }).join('\n')
        }
        if (!entry?.baseline) {
            return row.offline ? 'Read the node while it is online to edit its roles.' : ''
        }
        const choices = choicesFor(new Set([row.node.id]))
        const busy = entry.phase === 'reading' || entry.phase === 'sending'
        return (
            <Stack gap="1">
                {ROLE_TIERS.map(tier => (
                    <Select
                        key={tier}
                        size="small"
                        aria-label={`${row.node.name} ${tier} role`}
                        items={roleItems(choices, entry.draft[tier], descriptors)}
                        value={entry.draft[tier] ? roleKey(entry.draft[tier]) : UNSET_ROLE}
                        onValueChange={(value: string) => setOwn(row.node.id, tier, value)}
                        disabled={busy || running}
                    />
                ))}
            </Stack>
        )
    }

    const stateCell = (row: TargetRow): ReactNode => {
        const entry = editors.get(row.node.id)
        const badge = STATE_BADGES[row.state]
        const detail = stateDetail(row, entry, descriptors)
        return (
            <Stack gap="1">
                <Badge color={badge.color} kind="solid">
                    {badge.label}
                </Badge>
                {detail && (
                    <Text kind="body/regular/sm" className="text-subtle-color whitespace-pre-line">
                        {detail}
                    </Text>
                )}
                {entry?.requestId && (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {`Request ${entry.requestId}`}
                    </Text>
                )}
                {perNode && row.state === 'conflict' && entry?.latest && (
                    <Flex gap="1" wrap="wrap">
                        <Button
                            kind="secondary"
                            size="tiny"
                            onClick={() => store.keepDraftOnLatest(row.node.id)}
                        >
                            Keep my roles on the latest
                        </Button>
                        <Button
                            kind="secondary"
                            size="tiny"
                            onClick={() => store.discard(row.node.id)}
                        >
                            Use the node&apos;s roles
                        </Button>
                    </Flex>
                )}
            </Stack>
        )
    }

    const tableRows: TableRowDefinition[] = rows.map(row => ({
        id: row.node.id,
        selected: selected.has(row.node.id),
        cells: [
            {
                children: (
                    <Checkbox
                        aria-label={`Select ${row.node.name}`}
                        checked={selected.has(row.node.id)}
                        disabled={running}
                        onCheckedChange={checked => toggle(row.node.id, checked === true)}
                    />
                )
            },
            {
                children: (
                    <Stack gap="1">
                        <Text kind="body/regular/sm">{row.node.name}</Text>
                        {row.offline && (
                            <Badge color="gray" kind="outline">
                                Offline
                            </Badge>
                        )}
                    </Stack>
                )
            },
            {
                children: (
                    <Text kind="body/regular/sm" className="whitespace-pre-line">
                        {currentCell(row)}
                    </Text>
                )
            },
            {
                children: selected.has(row.node.id) ? (
                    perNode ? (
                        newCell(row)
                    ) : (
                        <Text kind="body/regular/sm" className="whitespace-pre-line">
                            {newCell(row)}
                        </Text>
                    )
                ) : (
                    ''
                )
            },
            { children: stateCell(row) }
        ]
    }))

    return (
        <Stack gap="4">
            <Text kind="body/semibold/md">Configure several nodes</Text>
            <Text kind="body/regular/sm" className="text-subtle-color">
                Each selected node is read and sent its roles on its own, a few at a time. A node
                that fails does not undo the others, and nothing is retried or sent to an offline
                node unless you ask.
            </Text>
            {peers.length === 0 ? (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    No paired nodes.
                </Text>
            ) : (
                <>
                    <SegmentedControl
                        size="small"
                        aria-label="Roles for the selected nodes"
                        items={[
                            { value: 'common', children: 'Same roles on each' },
                            { value: 'per-node', children: 'Per node' }
                        ]}
                        value={bulk.mode}
                        onValueChange={value =>
                            store.setBulk({ mode: value === 'per-node' ? 'per-node' : 'common' })
                        }
                    />
                    {!perNode && (
                        <Stack gap="2">
                            {ROLE_TIERS.map(tier => (
                                <Flex key={tier} align="center" gap="3" wrap="wrap">
                                    <Text
                                        kind="body/regular/sm"
                                        className="text-subtle-color w-16 capitalize"
                                    >
                                        {tier}
                                    </Text>
                                    <div className="min-w-0 flex-1">
                                        <Select
                                            size="small"
                                            aria-label={`Common ${tier} role`}
                                            items={roleItems(
                                                commonChoices,
                                                bulk.common[tier],
                                                descriptors
                                            )}
                                            value={
                                                bulk.common[tier]
                                                    ? roleKey(bulk.common[tier])
                                                    : UNSET_ROLE
                                            }
                                            onValueChange={(value: string) =>
                                                setCommon(tier, value)
                                            }
                                            disabled={running}
                                        />
                                    </div>
                                </Flex>
                            ))}
                        </Stack>
                    )}
                    <Table
                        density="compact"
                        layout="auto"
                        columns={['Selected', 'Node', 'Current roles', 'New roles', 'State']}
                        rows={tableRows}
                    />
                    {progress && (
                        <ProgressBar
                            aria-label={`${progress.done} of ${progress.total} nodes done`}
                            value={progress.total ? (progress.done / progress.total) * 100 : 100}
                        />
                    )}
                    <Flex gap="2" justify="end">
                        <Button
                            kind="secondary"
                            size="small"
                            disabled={running || toRetry.length === 0}
                            onClick={() => void run(toRetry)}
                        >
                            {`Retry selected failures (${toRetry.length})`}
                        </Button>
                        <Button
                            kind="primary"
                            color="brand"
                            size="small"
                            disabled={running || toSend.length === 0}
                            aria-busy={running}
                            onClick={() => void run(toSend)}
                        >
                            {`Send to ${toSend.length} ${toSend.length === 1 ? 'node' : 'nodes'}`}
                        </Button>
                    </Flex>
                </>
            )}
        </Stack>
    )
}
