// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react'
import { Button, FormField, Select, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { ModelSettingsIdentity } from '@/shared/types/model-config'
import type { EngineSettingsTarget } from '@/shared/types/engine-settings'
import {
    isValidProfileKey,
    newProfileKeyStatus,
    parseModelSettings
} from '@/shared/utils/node-policy-model-settings'
import {
    engineSettingsKey,
    modelSettingsKey,
    useEngineSettingsStore
} from '@/ui/stores/engine-settings.store'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { portableLoadPatch } from '@/shared/utils/node-policy-model-copy'

export default function ModelProfileControls({
    text,
    savedText,
    identity,
    remote,
    disabled,
    target,
    policyRevision,
    newProfile,
    onIdentityChange,
    onReview
}: {
    text: string
    savedText: string | null
    identity: ModelSettingsIdentity
    remote: boolean
    disabled: boolean
    target?: EngineSettingsTarget
    policyRevision: string
    newProfile: boolean
    onIdentityChange: (identity: ModelSettingsIdentity) => void
    onReview: () => void
}) {
    const [confirmRemove, setConfirmRemove] = useState(false)
    const parsed = useMemo(() => parseModelSettings(text), [text])
    const saved = useMemo(
        () => (savedText === null ? null : parseModelSettings(savedText)),
        [savedText]
    )
    const profile =
        !newProfile && parsed.ok
            ? parsed.value.find(item => item.key === identity.profileKey)
            : undefined
    const stored =
        !newProfile && saved?.ok
            ? saved.value.find(item => item.key === identity.profileKey)
            : undefined
    const keyStatus = newProfileKeyStatus(text, identity.profileKey)
    const key = target
        ? modelSettingsKey(
              target,
              newProfile && keyStatus !== 'available' ? undefined : identity.profileKey
          )
        : ''
    const draft = useEngineSettingsStore(state => state.modelDrafts[key])
    const snapshot = useEngineSettingsStore(state =>
        target ? state.entries[engineSettingsKey(target)]?.snapshot : undefined
    )
    const [displayName, setDisplayName] = useState(profile?.displayName ?? identity.model)
    const [icon, setIcon] = useState(profile?.icon ?? '')
    const locked = remote && (!saved?.ok || stored?.localOnly === true)
    const keyValid = isValidProfileKey(identity.profileKey ?? '')
    const duplicate = newProfile && keyStatus === 'duplicate'
    const nameValid =
        displayName.length > 0 &&
        displayName.isWellFormed() &&
        new TextEncoder().encode(displayName).length <= 128 &&
        !/[\p{Cc}]/u.test(displayName)
    const blocked =
        disabled ||
        locked ||
        !target ||
        !snapshot?.editable ||
        !parsed.ok ||
        !!draft?.uncertain ||
        !!draft?.busy ||
        !!draft?.change?.remove
    const stageMetadata = () => {
        if (blocked || !target || !keyValid || duplicate || !nameValid) return
        useEngineSettingsStore.getState().editModel(key, {
            ...draft?.change,
            modelId: identity.model,
            profileKey: identity.profileKey,
            expectedPolicyRevision: policyRevision,
            // A new typed profile inherits defaults; an empty load patch is invalid.
            ...(newProfile && !draft?.change?.load
                ? { load: portableLoadPatch({}, ['context']) }
                : {}),
            policy: { ...draft?.change?.policy, displayName, icon }
        })
        onReview()
    }
    const typed =
        snapshot?.settings.modelLoads?.some(load => load.profileKey === identity.profileKey) ===
        true
    return (
        <Stack gap="3">
            <Text kind="body/regular/sm">
                Saved generation lock:{' '}
                {!saved?.ok
                    ? 'Unknown, reload the target policy'
                    : stored?.localOnly
                      ? 'Only this device can edit generation settings'
                      : 'Unlocked'}
            </Text>
            {newProfile ? (
                <FormField
                    slotLabel="Profile key"
                    slotHelp="Use a unique profile key. Names with a valid role name before a colon are reserved."
                    status={!keyValid || duplicate ? 'error' : undefined}
                    slotError={
                        duplicate
                            ? 'This profile key already exists.'
                            : !keyValid
                              ? 'Enter a valid profile key.'
                              : undefined
                    }
                >
                    <TextInput
                        aria-label="Profile key"
                        value={identity.profileKey ?? ''}
                        disabled={disabled || !!draft?.attempt}
                        onValueChange={profileKey => onIdentityChange({ ...identity, profileKey })}
                    />
                </FormField>
            ) : (
                <Text kind="body/regular/sm">
                    Draft profile: {profile?.displayName ?? identity.profileKey}
                </Text>
            )}
            <FormField
                slotLabel="Display name"
                status={!nameValid ? 'error' : undefined}
                slotError={
                    !nameValid
                        ? 'Enter a display name of up to 128 UTF-8 bytes without control characters.'
                        : undefined
                }
            >
                <TextInput
                    aria-label="Display name"
                    value={displayName}
                    disabled={blocked}
                    onValueChange={setDisplayName}
                />
            </FormField>
            <FormField slotLabel="Profile icon">
                <Select
                    aria-label="Profile icon"
                    value={icon}
                    disabled={blocked}
                    items={[
                        { value: '', children: 'No icon' },
                        { value: 'chat-single', children: 'Chat' },
                        { value: 'code', children: 'Code' },
                        { value: 'cube', children: 'Cube' },
                        { value: 'document', children: 'Document' },
                        { value: 'image', children: 'Image' }
                    ]}
                    onValueChange={setIcon}
                />
            </FormField>
            <Button
                size="small"
                disabled={blocked || !keyValid || duplicate || !nameValid || !identity.model}
                onClick={stageMetadata}
            >
                {newProfile ? 'Create profile' : 'Review profile metadata'}
            </Button>
            {profile &&
                (confirmRemove ? (
                    <>
                        <Text kind="body/regular/sm">
                            Remove profile {profile.displayName}?{' '}
                            {typed
                                ? 'Removes its load, role memberships, served permission and residency for this profile key.'
                                : `Removes its role memberships and residency for native model ${profile.model}. Served permission stays in place.`}{' '}
                            Model files stay in place. Preview and apply the removal below.
                        </Text>
                        <Button
                            size="small"
                            disabled={blocked || !snapshot?.editable}
                            onClick={() => {
                                if (!target || blocked) return
                                useEngineSettingsStore.getState().editModel(key, {
                                    modelId: identity.model,
                                    profileKey: identity.profileKey,
                                    expectedPolicyRevision: policyRevision,
                                    remove: true
                                })
                                setConfirmRemove(false)
                                onReview()
                            }}
                        >
                            Confirm remove profile
                        </Button>
                        <Button
                            size="small"
                            kind="tertiary"
                            onClick={() => setConfirmRemove(false)}
                        >
                            Keep profile
                        </Button>
                    </>
                ) : (
                    <Button
                        size="small"
                        kind="secondary"
                        disabled={blocked}
                        onClick={() => setConfirmRemove(true)}
                    >
                        Remove profile
                    </Button>
                ))}
            {locked && (
                <InlineErrorBanner
                    severity="info"
                    message="Generation settings are locked or the target policy is unavailable. Reload or edit this profile on that device."
                />
            )}
            <Text kind="body/regular/sm">
                Generation preferences do not establish readiness. Keep loaded and strict readiness
                require existing qualified finite bounds and target-side evidence. Setup does not
                create qualifications.
            </Text>
        </Stack>
    )
}
