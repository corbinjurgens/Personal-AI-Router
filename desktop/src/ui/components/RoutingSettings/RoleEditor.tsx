// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from 'react'
import {
    Badge,
    Button,
    Flex,
    FormField,
    Select,
    Stack,
    Text,
    TextInput
} from '@nvidia/foundations-react-core'
import { useShallow } from 'zustand/react/shallow'
import type { NodePolicySnapshot } from '@/shared/types/node-policy'
import { roleNameError, sameNodeRoles } from '@/shared/utils/node-roles'
import { ConfirmModal } from '@/ui/components/ConfirmModal'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { ModelIdentityTags } from '@/ui/components/ModelIdentity/ModelIdentityTags'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
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

interface RoleEditorProps {
    /** The broker's id for the node; this PC's is the self id. */
    targetId: string
    /** The id `policy:*` takes; undefined for this PC. */
    nodeId: string | undefined
    /** The persisted document as saved by this editor, for the policy editor to adopt. */
    onSaved: (saved: NodePolicySnapshot) => void
}

/** Ordered role declarations and profile memberships for one node. */
export default function RoleEditor({ targetId, nodeId, onSaved }: RoleEditorProps) {
    const entry = useNodeConfigurationStore(state => state.editors.get(targetId))
    const management = useNodeConfigurationStore(state => state.management.get(targetId))
    const connected = useConnectionStore(state => state.connected)
    const peers = useDiscoveredNodesStore(state => state.nodes)
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
    const [noSuggestion, setNoSuggestion] = useState(false)
    const [nameForm, setNameForm] = useState<{
        targetId: string
        originalName?: string
        value: string
    } | null>(null)
    const [confirmation, setConfirmation] = useState<{
        targetId: string
        name: string
        newName?: string
    } | null>(null)

    const baseline = entry?.baseline
    const roleDraft = entry?.draft
    const paused =
        nodeId !== undefined &&
        baseline !== undefined &&
        nodePaused(baseline.policy, baseline.availability)

    const suggest = useCallback(() => {
        if (!baseline || !roleDraft || readOnly) return
        const roles = suggestRoles(choices, [baseline.policy])
        const next = roleDraft.map(role => {
            const suggestion = roles.find(suggestion => suggestion.name === role.name)
            return role.profiles.length > 0 ? role : (suggestion ?? role)
        })
        setNoSuggestion(sameNodeRoles(next, roleDraft))
        if (!sameNodeRoles(next, roleDraft))
            useNodeConfigurationStore.getState().setDraft(targetId, next)
    }, [baseline, roleDraft, choices, targetId, readOnly])

    const save = useCallback(async () => {
        if (readOnly) return
        const saved = await useNodeConfigurationStore.getState().save(targetId, nodeId)
        if (saved) onSaved(saved)
    }, [targetId, nodeId, onSaved, readOnly])

    if (!entry || !baseline) return null

    const { draft, phase } = entry
    const edited = !sameNodeRoles(draft, baseline.roles) || (entry.roleEdits?.length ?? 0) > 0
    // An uncertain save holds mutations until readback or discard resolves it.
    const sending = phase === 'sending' || phase === 'reading'
    const mutationLocked = sending || phase === 'unknown'
    const store = useNodeConfigurationStore.getState()
    const form = nameForm?.targetId === targetId ? nameForm : null
    const pending = confirmation?.targetId === targetId ? confirmation : null
    const validateName = (name: string, originalName?: string) => {
        const error = roleNameError(name)
        if (error) return error
        if (draft.some(role => role.name === name && role.name !== originalName))
            return `Role ${name} already exists on this node.`
        if (profiles.some(profile => profile.key === name))
            return `Role ${name} cannot have the same name as a profile key on this node.`
        if (originalName !== undefined && name !== originalName)
            return store.roleRenameError(targetId, originalName, name, nodeId)
        return undefined
    }
    const nameError = form ? validateName(form.value, form.originalName) : undefined
    const shadowWarnings = (name: string) =>
        peers
            .filter(
                peer =>
                    peer.id !== targetId &&
                    peer.policy?.profiles.some(profile => profile.key === name)
            )
            .map(peer => (
                <Text key={peer.id} kind="body/regular/sm" role="status" className="break-all">
                    {`Role ${name} hides the profile ${name} on ${peer.name}.`}
                </Text>
            ))

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
                <Text kind="body/semibold/sm">
                    Roles, weakest first. Allow stronger tries roles further down this list; Allow
                    weaker tries roles further up.
                </Text>
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
                Create roles here, then assign this node&apos;s profiles. A role can be empty.
            </Text>
            <Text kind="body/regular/sm" className="text-subtle-color">
                An API route needs an API entry for the role and a routing rule that allows remote
                execution. API overrides are listed, but public API dispatch is disabled in this
                build and requests are refused. Enabled deployments and remote or paid permissions
                cannot be saved.
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
            {!form && (
                <Button
                    kind="secondary"
                    size="small"
                    disabled={mutationLocked || readOnly}
                    onClick={() => setNameForm({ targetId, value: '' })}
                >
                    Add role
                </Button>
            )}
            {form && (
                <form
                    aria-label={
                        form.originalName === undefined
                            ? 'Add role'
                            : `Rename role ${form.originalName}`
                    }
                    onSubmit={event => {
                        event.preventDefault()
                        if (mutationLocked || readOnly || nameError) return
                        if (form.originalName === undefined) {
                            store.addRole(targetId, form.value)
                            setNameForm(null)
                        } else if (
                            form.value !== form.originalName &&
                            draft.some(role => role.name === form.originalName)
                        ) {
                            setConfirmation({
                                targetId,
                                name: form.originalName,
                                newName: form.value
                            })
                        }
                    }}
                >
                    <Stack gap="2">
                        <FormField
                            name="roleName"
                            slotLabel={
                                form.originalName === undefined
                                    ? 'New role name'
                                    : `New name for role ${form.originalName}`
                            }
                            status={nameError ? 'error' : undefined}
                            slotError={nameError}
                            slotHelp="Use up to 64 lowercase letters, digits, dots, underscores or hyphens; start with a letter or digit."
                        >
                            <TextInput
                                size="small"
                                value={form.value}
                                title={form.value}
                                onValueChange={value => setNameForm({ ...form, value })}
                                disabled={mutationLocked || readOnly}
                            />
                        </FormField>
                        {!nameError && shadowWarnings(form.value)}
                        <Flex gap="2">
                            <Button
                                type="submit"
                                kind="primary"
                                size="small"
                                disabled={
                                    mutationLocked ||
                                    readOnly ||
                                    nameError !== undefined ||
                                    form.value === form.originalName
                                }
                            >
                                {form.originalName === undefined ? 'Add role' : 'Rename role'}
                            </Button>
                            <Button
                                type="button"
                                kind="secondary"
                                size="small"
                                onClick={() => setNameForm(null)}
                            >
                                Cancel
                            </Button>
                        </Flex>
                    </Stack>
                </form>
            )}
            {draft.length === 0 && <Text>No roles declared</Text>}
            {draft.map((role, roleIndex) => (
                <Stack
                    key={role.name}
                    gap="2"
                    role="group"
                    aria-label={`${role.name} role`}
                    title={role.name}
                >
                    <Flex align="center" gap="2" wrap="wrap">
                        <Text
                            kind="body/semibold/sm"
                            title={role.name}
                            className="min-w-0 break-all"
                        >
                            {role.name}
                        </Text>
                        <Button
                            size="tiny"
                            kind="secondary"
                            aria-label={`Rename role ${role.name}`}
                            title={`Rename role ${role.name}`}
                            disabled={mutationLocked || readOnly}
                            onClick={() =>
                                setNameForm({
                                    targetId,
                                    originalName: role.name,
                                    value: role.name
                                })
                            }
                        >
                            Rename
                        </Button>
                        <Button
                            size="tiny"
                            kind="secondary"
                            aria-label={`Remove role ${role.name}`}
                            title={`Remove role ${role.name}`}
                            disabled={mutationLocked || readOnly}
                            onClick={() => setConfirmation({ targetId, name: role.name })}
                        >
                            Remove role
                        </Button>
                        <Button
                            size="tiny"
                            kind="secondary"
                            aria-label={`Move role ${role.name} up`}
                            title={`Move role ${role.name} up`}
                            disabled={mutationLocked || readOnly || roleIndex === 0}
                            onClick={() => store.moveRole(targetId, role.name, -1)}
                        >
                            Move up
                        </Button>
                        <Button
                            size="tiny"
                            kind="secondary"
                            aria-label={`Move role ${role.name} down`}
                            title={`Move role ${role.name} down`}
                            disabled={mutationLocked || readOnly || roleIndex === draft.length - 1}
                            onClick={() => store.moveRole(targetId, role.name, 1)}
                        >
                            Move down
                        </Button>
                    </Flex>
                    {shadowWarnings(role.name)}
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
                                    disabled={mutationLocked || readOnly || index === 0}
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
                                        mutationLocked ||
                                        readOnly ||
                                        index === role.profiles.length - 1
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
                                    disabled={mutationLocked || readOnly}
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
                            aria-label={`Add profile to ${role.name}`}
                            title={`Add profile to ${role.name}`}
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
                            disabled={mutationLocked || readOnly}
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
                    No saved declaration adds a profile to this draft.
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
                                  .map(difference => {
                                      const position =
                                          difference.mineIndex !== undefined &&
                                          difference.theirsIndex !== undefined
                                              ? `; position yours ${difference.mineIndex + 1}, node ${difference.theirsIndex + 1}`
                                              : ''
                                      return `${difference.tier}: yours ${describe(difference.mine, baseline.policy)}, node ${describe(difference.theirs, entry.latest?.policy ?? '{}')}${position}`
                                  })
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
            {phase === 'draft' && entry.message && (
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
                    disabled={mutationLocked || readOnly || choices.length === 0}
                    onClick={suggest}
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
                    disabled={
                        mutationLocked || readOnly || paused || !edited || phase === 'conflict'
                    }
                    aria-busy={sending}
                    onClick={() => void save()}
                >
                    Save roles
                </Button>
            </Flex>
            {pending && (
                <ConfirmModal
                    open
                    onOpenChange={open => {
                        if (!open) setConfirmation(null)
                    }}
                    title={
                        pending.newName === undefined
                            ? `Remove role ${pending.name}?`
                            : `Rename role ${pending.name}?`
                    }
                    message={[
                        pending.newName === undefined
                            ? `Remove role ${pending.name} and its profile memberships? The profiles will be kept.`
                            : `Rename role ${pending.name} to ${pending.newName}? Its position and profile memberships will be kept.`,
                        nodeId !== undefined
                            ? `API entries on ${peers.find(peer => peer.id === targetId)?.name || targetId} can't be seen from here and stay as they are.`
                            : store.roleApiEntryPresent(targetId, pending.name)
                              ? pending.newName === undefined
                                  ? `Also deletes this node's API entry for ${pending.name}.`
                                  : `Also moves this node's API entry for ${pending.name}.`
                              : ''
                    ]
                        .filter(Boolean)
                        .join(' ')}
                    confirmLabel={pending.newName === undefined ? 'Remove role' : 'Rename role'}
                    confirmColor={pending.newName === undefined ? 'danger' : 'brand'}
                    onConfirm={() => {
                        if (
                            mutationLocked ||
                            readOnly ||
                            !draft.some(role => role.name === pending.name)
                        )
                            return
                        if (pending.newName === undefined) {
                            store.removeRole(targetId, pending.name, nodeId)
                            if (form?.originalName === pending.name) setNameForm(null)
                        } else if (!validateName(pending.newName, pending.name)) {
                            store.renameRole(targetId, pending.name, pending.newName, nodeId)
                            setNameForm(null)
                        }
                    }}
                />
            )}
        </Stack>
    )
}
