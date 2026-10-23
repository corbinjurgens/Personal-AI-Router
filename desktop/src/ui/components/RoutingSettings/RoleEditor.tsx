// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from 'react'
import { Badge, Button, Flex, Select, Stack, Text } from '@nvidia/foundations-react-core'
import { useShallow } from 'zustand/react/shallow'
import type { NodePolicySnapshot } from '@/shared/types/node-policy'
import type { NodeRole, RoleTier } from '@/shared/types/node-roles'
import { ROLE_TIERS, sameNodeRoles } from '@/shared/utils/node-roles'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import {
    chosenRole,
    inventoryChoices,
    nodePaused,
    NODE_PAUSED_MESSAGE,
    roleDifferences,
    roleItems,
    roleKey,
    roleLabel,
    suggestRoles,
    UNSET_ROLE,
    useNodeConfigurationStore,
    watchNodeManagement
} from '@/ui/stores/node-configuration.store'

const TIER_LABELS: Record<RoleTier, string> = {
    weak: 'Weak',
    medium: 'Medium',
    strong: 'Strong'
}

interface RoleEditorProps {
    /** The broker's id for the node; this PC's is the self id. */
    targetId: string
    /** The id `policy:*` takes; undefined for this PC. */
    nodeId: string | undefined
    /** The persisted document as saved by this editor, for the policy editor to adopt. */
    onSaved: (saved: NodePolicySnapshot) => void
}

/** Weak, medium and strong roles for one node, chosen from that node's installed models. */
export default function RoleEditor({ targetId, nodeId, onSaved }: RoleEditorProps) {
    const entry = useNodeConfigurationStore(state => state.editors.get(targetId))
    const management = useNodeConfigurationStore(state => state.management.get(targetId))
    const connected = useConnectionStore(state => state.connected)
    const readOnly = !connected || (nodeId !== undefined && management?.state !== 'allowed')
    useEffect(() => watchNodeManagement(targetId, nodeId), [targetId, nodeId])
    const descriptors = useEngineStatusStore(state => state.descriptors)
    const inventory = useEngineModelsStore(
        useShallow(state =>
            Array.from(state.models.values()).filter(models => models.nodeId === targetId)
        )
    )
    const choices = useMemo(() => inventoryChoices(inventory), [inventory])
    const [suggesting, setSuggesting] = useState(false)
    const [noSuggestion, setNoSuggestion] = useState(false)

    const baseline = entry?.baseline
    const paused =
        nodeId !== undefined &&
        baseline !== undefined &&
        nodePaused(baseline.policy, baseline.availability)

    const suggest = useCallback(async () => {
        if (!baseline || readOnly) return
        setSuggesting(true)
        setNoSuggestion(false)
        try {
            // This PC's own tier entries are the router's current guesses; a
            // paired node's document comes first.
            const own = nodeId ? await window.pairApi.nodePolicy.get().catch(() => null) : null
            const policies = own ? [baseline.policy, own.policy] : [baseline.policy]
            const roles = suggestRoles(choices, policies)
            if (Object.keys(roles).length === 0) setNoSuggestion(true)
            else useNodeConfigurationStore.getState().setDraft(targetId, roles)
        } finally {
            setSuggesting(false)
        }
    }, [baseline, nodeId, choices, targetId, readOnly])

    const save = useCallback(async () => {
        if (readOnly) return
        const saved = await useNodeConfigurationStore.getState().save(targetId, nodeId)
        if (saved) onSaved(saved)
    }, [targetId, nodeId, onSaved, readOnly])

    if (!entry || !baseline) return null

    const { draft, phase } = entry
    const edited = !sameNodeRoles(draft, baseline.roles)
    // A bulk attempt on this node reads before it sends; both hold the draft.
    const sending = phase === 'sending' || phase === 'reading'
    const store = useNodeConfigurationStore.getState()

    const pick = (tier: RoleTier, value: string) => {
        if (value === UNSET_ROLE) {
            store.setRole(targetId, tier, undefined)
            return
        }
        const role = chosenRole(choices, value)
        if (role) store.setRole(targetId, tier, role)
    }

    const differences = entry.latest ? roleDifferences(draft, entry.latest.roles) : []
    const describe = (role: NodeRole | undefined) => (role ? roleLabel(role, descriptors) : 'unset')

    return (
        <Stack gap="2">
            <Flex align="center" gap="3" wrap="wrap">
                <Text kind="body/semibold/sm">Roles</Text>
                {phase === 'applied' && (
                    <Badge color="green" kind="solid">
                        {`Saved at revision ${baseline.revision}`}
                    </Badge>
                )}
                {paused && (
                    <Badge color="gray" kind="solid">
                        Paused{baseline.pausedBy ? `; PausedBy: ${baseline.pausedBy}` : ''}
                    </Badge>
                )}
            </Flex>
            <Text kind="body/regular/sm" className="text-subtle-color">
                Declaring any tier makes this node authoritative: routers send it only its declared
                tiers, and its unset tiers are not served.
            </Text>
            {paused && (
                <Text kind="body/regular/sm" role="status">
                    {NODE_PAUSED_MESSAGE}
                </Text>
            )}
            {nodeId !== undefined && readOnly && (
                <Text kind="body/regular/sm" role="status">
                    {management?.message || 'Reading remote management permission for this device.'}
                </Text>
            )}
            {ROLE_TIERS.map(tier => (
                <Flex key={tier} align="center" gap="3" wrap="wrap">
                    <Text kind="body/regular/sm" className="text-subtle-color w-16">
                        {TIER_LABELS[tier]}
                    </Text>
                    <div className="min-w-0 flex-1">
                        <Select
                            size="small"
                            aria-label={`${TIER_LABELS[tier]} role`}
                            items={roleItems(choices, draft[tier], descriptors)}
                            value={draft[tier] ? roleKey(draft[tier]) : UNSET_ROLE}
                            onValueChange={(value: string) => pick(tier, value)}
                            disabled={sending || readOnly}
                        />
                    </div>
                </Flex>
            ))}
            {choices.length === 0 && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    This node reports no installed models yet.
                </Text>
            )}
            {noSuggestion && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    No tier entry names a model installed on this node.
                </Text>
            )}
            {phase === 'conflict' && entry.latest && (
                <InlineErrorBanner
                    severity="warning"
                    message={[
                        `The node's policy changed to revision ${entry.latest.revision} after these roles were loaded, so they were not sent.`,
                        differences.length === 0
                            ? 'Its roles match this draft; only other settings changed.'
                            : differences
                                  .map(
                                      difference =>
                                          `${TIER_LABELS[difference.tier]}: yours ${describe(difference.mine)}, node ${describe(difference.theirs)}`
                                  )
                                  .join('\n'),
                        entry.message ? `Service message: ${entry.message}` : ''
                    ]
                        .filter(Boolean)
                        .join('\n')}
                >
                    <Button
                        kind="secondary"
                        size="tiny"
                        onClick={() => store.keepDraftOnLatest(targetId)}
                    >
                        Keep my roles on the latest
                    </Button>
                    <Button kind="secondary" size="tiny" onClick={() => store.discard(targetId)}>
                        Use the node&apos;s roles
                    </Button>
                </InlineErrorBanner>
            )}
            {phase === 'vetoed' && (
                <InlineErrorBanner severity="warning" message={NODE_PAUSED_MESSAGE} />
            )}
            {phase === 'rejected' && entry.message && (
                <InlineErrorBanner severity="error" message={entry.message} />
            )}
            {phase === 'unreachable' && (
                <InlineErrorBanner
                    severity="warning"
                    message={`The node could not be reached, so the roles were not sent. Your draft is kept. ${entry.message ?? ''}`}
                />
            )}
            {phase === 'unknown' && (
                <InlineErrorBanner
                    severity="warning"
                    message={`Could not confirm whether the roles were saved: ${entry.message ?? ''} Reload to see the node's roles.`}
                />
            )}
            <Flex gap="2" justify="end">
                <Button
                    kind="secondary"
                    size="small"
                    disabled={sending || readOnly || suggesting || choices.length === 0}
                    aria-busy={suggesting}
                    onClick={() => void suggest()}
                >
                    Suggest roles
                </Button>
                <Button
                    kind="secondary"
                    size="small"
                    disabled={sending || !edited || phase === 'conflict'}
                    onClick={() => store.discard(targetId)}
                >
                    Discard role changes
                </Button>
                <Button
                    kind="primary"
                    color="brand"
                    size="small"
                    disabled={sending || readOnly || paused || !edited || phase === 'conflict'}
                    aria-busy={sending}
                    onClick={() => void save()}
                >
                    Save roles
                </Button>
            </Flex>
        </Stack>
    )
}
