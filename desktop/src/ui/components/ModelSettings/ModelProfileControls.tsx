// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react'
import { Button, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { NodeModelReference } from '@/shared/types/node-policy'
import {
    createModelSettingsProfile,
    parseModelSettings,
    removeModelSettingsProfile
} from '@/shared/utils/node-policy-model-settings'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'

export default function ModelProfileControls({
    text,
    savedText,
    identity,
    onTextChange,
    remote,
    disabled
}: {
    text: string
    savedText: string | null
    identity: NodeModelReference
    onTextChange: (text: string) => void
    remote: boolean
    disabled: boolean
}) {
    const [name, setName] = useState('')
    const [message, setMessage] = useState<string | null>(null)
    const [confirmRemove, setConfirmRemove] = useState(false)
    const parsed = useMemo(() => parseModelSettings(text), [text])
    const saved = useMemo(
        () => (savedText === null ? null : parseModelSettings(savedText)),
        [savedText]
    )
    const same = (item: NodeModelReference) =>
        item.engine === identity.engine && item.model === identity.model
    const profile = parsed.ok ? parsed.value.find(same) : undefined
    const stored = saved?.ok ? saved.value.find(same) : undefined
    const locked = remote && (!saved?.ok || stored?.localOnly === true)
    const removal = profile ? removeModelSettingsProfile(text, identity) : null
    const apply = (result: ReturnType<typeof createModelSettingsProfile>) => {
        if (disabled || locked) return
        if (!result.ok) setMessage(result.error.message)
        else {
            onTextChange(result.value)
            setMessage(null)
            setConfirmRemove(false)
        }
    }
    return (
        <Stack gap="3">
            <Text kind="body/regular/sm">
                Saved generation lock:{' '}
                {!saved?.ok
                    ? 'Unknown, reload the target policy'
                    : !stored
                      ? 'No saved profile'
                      : stored.localOnly
                        ? 'Only this device can edit generation settings'
                        : 'Unlocked'}
            </Text>
            {!profile && locked && (
                <InlineErrorBanner
                    severity="info"
                    message={
                        saved?.ok
                            ? 'Generation settings are locked in the target’s saved policy. Edit this profile on that device.'
                            : 'Reload the target’s saved policy before creating a profile.'
                    }
                />
            )}
            {profile ? (
                <>
                    <Text kind="body/regular/sm">Draft profile: {profile.name}</Text>
                    {removal && !removal.ok && (
                        <InlineErrorBanner severity="warning" message={removal.error.message} />
                    )}
                    {confirmRemove ? (
                        <>
                            <Text kind="body/regular/sm">
                                Remove profile {profile.name}? Model files and other model
                                references stay in place.
                            </Text>
                            <Button
                                size="small"
                                disabled={disabled || locked || !removal?.ok}
                                onClick={() => removal && apply(removal)}
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
                            disabled={disabled || locked || !removal?.ok}
                            onClick={() => setConfirmRemove(true)}
                        >
                            Remove profile
                        </Button>
                    )}
                </>
            ) : (
                <>
                    <FormField
                        slotLabel="Profile name"
                        slotHelp="Use letters, digits, dots, underscores, colons, slashes or hyphens. Start with a letter or digit."
                    >
                        <TextInput
                            aria-label="Profile name"
                            value={name}
                            onValueChange={setName}
                            disabled={disabled || locked || !parsed.ok}
                        />
                    </FormField>
                    <Button
                        size="small"
                        disabled={disabled || locked || !parsed.ok}
                        onClick={() => apply(createModelSettingsProfile(text, identity, name))}
                    >
                        Create profile
                    </Button>
                </>
            )}
            {message && <InlineErrorBanner message={message} />}
            <Text kind="body/regular/sm">
                Generation preferences do not establish readiness. Keep loaded and strict readiness
                require existing qualified finite bounds and target-side evidence. Setup does not
                create qualifications.
            </Text>
        </Stack>
    )
}
