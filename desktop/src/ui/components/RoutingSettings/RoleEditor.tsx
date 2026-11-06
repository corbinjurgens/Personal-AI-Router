// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from 'react'
import { Badge, Button, Flex, Select, Stack, Text } from '@nvidia/foundations-react-core'
import { useShallow } from 'zustand/react/shallow'
import type { NodePolicySnapshot } from '@/shared/types/node-policy'
import type { RoleTier } from '@/shared/types/node-roles'
import { sameNodeRoles } from '@/shared/utils/node-roles'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { ModelIdentityTags } from '@/ui/components/ModelIdentity/ModelIdentityTags'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import {
    chosenRole,
    inventoryChoices,
    nodePaused,
    NODE_PAUSED_MESSAGE,
    roleDifferences,
    roleIdentity,
    roleItems,
    policyRoleProfiles,
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
    const profiles = useMemo(
        () => policyRoleProfiles(entry?.baseline?.policy ?? '{}'),
        [entry?.baseline?.policy]
    )
    const choices = useMemo(() => inventoryChoices(inventory, profiles), [inventory, profiles])
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
            if (roles.length === 0) setNoSuggestion(true)
            else
                useNodeConfigurationStore.getState().setDraft(
                    targetId,
                    baseline.roles.map(role => {
                        const suggestion = roles.find(suggestion => suggestion.name === role.name)
                        return role.profiles.length > 0 ? role : (suggestion ?? role)
                    })
                )
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
    // Reading and sending both hold the draft.
    const sending = phase === 'sending' || phase === 'reading'
    const store = useNodeConfigurationStore.getState()

    const differences = entry.latest ? roleDifferences(draft, entry.latest.roles) : []
    const describeProfile = (key: string, policy: string) => {
        const profile = policyRoleProfiles(policy).find(profile => profile.key === key)
        return profile
            ? `${profile.displayName} (${key}): ${roleLabel(profile, descriptors, roleIdentity(profile, inventory))}`
            : key
    }
    const describe = (keys: string[] | undefined, policy: string) =>
        keys === undefined
            ? 'undeclared'
            : keys.length === 0
              ? 'empty'
              : keys.map(key => describeProfile(key, policy)).join(', ')

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
            {draft.map(role => (
                <Stack key={role.name} gap="2" aria-label={`${TIER_LABELS[role.name]} role`}>
                    <Text kind="body/regular/sm" className="text-subtle-color w-16">
                        {TIER_LABELS[role.name]}
                    </Text>
                    {role.profiles.length === 0 && <Text>No profiles assigned</Text>}
                    {role.profiles.map((key, index) => {
                        const profile = profiles.find(profile => profile.key === key)
                        const label = profile?.displayName ?? key
                        return (
                            <Flex key={key} align="center" gap="2" wrap="wrap" aria-label={label}>
                                <Text title={label}>{label}</Text>
                                <ModelIdentityTags
                                    identity={
                                        profile?.modelIdentity ??
                                        (profile ? roleIdentity(profile, inventory) : undefined)
                                    }
                                />
                                <Button
                                    size="tiny"
                                    kind="secondary"
                                    aria-label={`Move ${label} up in ${role.name}`}
                                    disabled={sending || readOnly || index === 0}
                                    onClick={() =>
                                        store.moveRoleProfile(targetId, role.name, index, -1)
                                    }
                                >
                                    Move up
                                </Button>
                                <Button
                                    size="tiny"
                                    kind="secondary"
                                    aria-label={`Move ${label} down in ${role.name}`}
                                    disabled={
                                        sending || readOnly || index === role.profiles.length - 1
                                    }
                                    onClick={() =>
                                        store.moveRoleProfile(targetId, role.name, index, 1)
                                    }
                                >
                                    Move down
                                </Button>
                                <Button
                                    size="tiny"
                                    kind="secondary"
                                    aria-label={`Remove ${label} from ${role.name}`}
                                    disabled={sending || readOnly}
                                    onClick={() =>
                                        store.removeRoleProfile(targetId, role.name, key)
                                    }
                                >
                                    Remove
                                </Button>
                            </Flex>
                        )
                    })}
                    <div className="min-w-0 flex-1">
                        <Select
                            size="small"
                            placeholder="Add profile"
                            aria-label={`Add profile to ${TIER_LABELS[role.name]}`}
                            items={roleItems(
                                profiles.filter(profile => !role.profiles.includes(profile.key)),
                                undefined,
                                descriptors
                            )
                                .filter(item => item.value !== UNSET_ROLE)
                                .map(item => {
                                    const profile = profiles.find(
                                        profile => profile.key === item.value
                                    )
                                    return {
                                        value: item.value,
                                        children: (
                                            <Stack gap="1">
                                                <Text title={profile?.displayName}>
                                                    {profile?.displayName}
                                                </Text>
                                                <ModelIdentityTags
                                                    identity={
                                                        profile?.modelIdentity ??
                                                        (profile
                                                            ? roleIdentity(profile, inventory)
                                                            : undefined)
                                                    }
                                                />
                                            </Stack>
                                        )
                                    }
                                })}
                            value={UNSET_ROLE}
                            onValueChange={(value: string) => {
                                const key = chosenRole(profiles, value)
                                if (key) store.addRoleProfile(targetId, role.name, key)
                            }}
                            disabled={sending || readOnly}
                        />
                    </div>
                </Stack>
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
                            ? sameNodeRoles(draft, entry.latest.roles)
                                ? 'Its roles match this draft; only other settings changed.'
                                : 'The role declaration order differs from this draft.'
                            : differences
                                  .map(
                                      difference =>
                                          `${TIER_LABELS[difference.tier]}: yours ${describe(difference.mine, baseline.policy)}, node ${describe(difference.theirs, entry.latest?.policy ?? '{}')}`
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
