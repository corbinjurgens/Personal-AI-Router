// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from 'react'
import { Checkbox, FormField, Select, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import {
    applyModelSettings,
    MODEL_OPTION_FIELDS,
    parseModelSettings
} from '@/shared/utils/node-policy-model-settings'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'

/** Shares the routing editor's JSON draft, target and revision-checked Save. */
export default function ModelSettingsEditor({
    text,
    onTextChange,
    savedText,
    nodeName,
    remote,
    disabled,
    onValidityChange
}: {
    text: string
    onTextChange: (text: string) => void
    savedText: string | null
    nodeName: string
    remote: boolean
    disabled: boolean
    onValidityChange: (valid: boolean) => void
}) {
    const draft = useMemo(() => parseModelSettings(text), [text])
    const saved = useMemo(
        () => (savedText === null ? null : parseModelSettings(savedText)),
        [savedText]
    )
    const [picked, setPicked] = useState('')
    const [inputs, setInputs] = useState<{
        text: string
        profile: string
        fields: Record<string, { value: string; message?: string }>
    }>({ text: '', profile: '', fields: {} })
    const profiles = draft.ok ? draft.value : []
    const profile = profiles.find(item => item.key === picked) ?? profiles[0]
    const fields = inputs.text === text && inputs.profile === profile?.key ? inputs.fields : {}
    const hasErrors = Object.values(fields).some(field => field.message !== undefined)
    useEffect(() => onValidityChange(!hasErrors), [hasErrors, onValidityChange])

    // Only target-reported settings establish a peer lock. Raw draft edits cannot lift it.
    const stored =
        saved?.ok && profile ? saved.value.find(item => item.key === profile.key) : undefined
    const lockUnknown = remote && (!saved || !saved.ok)
    const locked = remote && stored?.localOnly === true
    const readOnly = disabled || lockUnknown || locked
    const apply = (
        edit: Parameters<typeof applyModelSettings>[2],
        field: string,
        value: string
    ) => {
        if (!profile || readOnly) return
        const result = applyModelSettings(text, profile, edit)
        if (!result.ok) {
            setInputs({
                text,
                profile: profile.key,
                fields: { ...fields, [field]: { value, message: result.error.message } }
            })
            return
        }
        const remaining = { ...fields }
        delete remaining[field]
        // Keep valid input spelling as well, so typing a decimal such as "0." can continue.
        if (edit.kind === 'option') remaining[field] = { value }
        setInputs({ text: result.value, profile: profile.key, fields: remaining })
        onTextChange(result.value)
    }

    return (
        <Stack gap="3">
            <Text kind="body/semibold/sm">Model inference settings</Text>
            <Text kind="body/regular/sm">Editing settings on {nodeName}</Text>
            <Text kind="body/regular/sm" className="text-subtle-color">
                These settings share the policy draft and Save button below. Output length belongs
                to the request; maxOutputTokens is a strict ceiling. Other options and unsupported
                engines remain in Advanced policy JSON.
            </Text>
            {!draft.ok && <InlineErrorBanner severity="warning" message={draft.error.message} />}
            {draft.ok && profiles.length === 0 && (
                <Text kind="body/regular/sm" role="status">
                    No supported profiles in this policy. Add a llama.cpp or LM Studio profile in
                    Advanced policy JSON. Ollama options stay in Advanced policy JSON.
                </Text>
            )}
            {profile && (
                <>
                    <FormField slotLabel="Model profile">
                        <Select
                            size="small"
                            attributes={{ SelectTrigger: { 'aria-label': 'Model profile' } }}
                            items={profiles.map(item => ({
                                value: item.key,
                                children: `${item.model} (${item.engine})`
                            }))}
                            value={profile.key}
                            onValueChange={(value: string) => setPicked(value)}
                            disabled={disabled}
                        />
                    </FormField>
                    {locked && (
                        <InlineErrorBanner
                            severity="info"
                            message={`Locked on ${nodeName}. Change this profile's inference settings on that device.`}
                        />
                    )}
                    {lockUnknown && (
                        <InlineErrorBanner
                            severity="warning"
                            message="Reload the target's saved policy before editing inference settings."
                        />
                    )}
                    <Checkbox
                        slotLabel="Only this device can edit"
                        checked={remote ? stored?.localOnly === true : profile.localOnly}
                        disabled={disabled || remote}
                        onCheckedChange={value => {
                            if (remote || disabled || typeof value !== 'boolean') return
                            apply({ kind: 'localOnly', value }, 'localOnly', String(value))
                        }}
                    />
                    <FormField
                        slotLabel="Request overrides"
                        slotHelp="Allow request overrides uses these values as defaults. Force PAIR values replaces configured leaves while keeping unrelated request fields."
                    >
                        <Select
                            size="small"
                            attributes={{ SelectTrigger: { 'aria-label': 'Request overrides' } }}
                            items={[
                                { value: 'defaults', children: 'Allow request overrides' },
                                { value: 'enforced', children: 'Force PAIR values' }
                            ]}
                            value={fields.mode?.value ?? profile.mode}
                            onValueChange={(value: string) =>
                                apply({ kind: 'mode', value }, 'mode', value)
                            }
                            disabled={readOnly}
                        />
                    </FormField>
                    {fields.mode?.message && <InlineErrorBanner message={fields.mode.message} />}
                    {MODEL_OPTION_FIELDS.filter(
                        field => field.key !== 'min_p' || profile.engine === 'llamacpp'
                    ).map(field => {
                        const value = profile.options[field.key]
                        const unsupported = value !== undefined && typeof value !== 'number'
                        const input = fields[field.key]
                        const error = input?.message
                        return (
                            <FormField
                                key={field.key}
                                slotLabel={field.label}
                                slotHelp={
                                    unsupported
                                        ? 'This value uses Advanced policy JSON. Clear or replace it here to use a number.'
                                        : field.help
                                }
                                status={error ? 'error' : undefined}
                                slotError={error}
                            >
                                <TextInput
                                    size="small"
                                    aria-label={field.label}
                                    inputMode="decimal"
                                    value={
                                        input?.value ??
                                        (typeof value === 'number' ? String(value) : '')
                                    }
                                    placeholder="Not configured"
                                    onValueChange={(value: string) =>
                                        apply(
                                            { kind: 'option', key: field.key, value },
                                            field.key,
                                            value
                                        )
                                    }
                                    disabled={readOnly}
                                />
                            </FormField>
                        )
                    })}
                </>
            )}
        </Stack>
    )
}
