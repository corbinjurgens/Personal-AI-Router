// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import { jsonFieldSource, replaceJsonFieldSource } from '@/shared/utils/json-source'
import type { JsonValue } from '@/shared/types/json'

export default function NodeMemorySettings({
    text,
    onChange,
    disabled
}: {
    text: string
    onChange: (text: string) => void
    disabled: boolean
}) {
    let valid = false
    try {
        const parsed: JsonValue = JSON.parse(text)
        valid = parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)
    } catch {
        /* The JSON editor owns invalid drafts. */
    }
    const fields = [
        {
            section: 'admission',
            key: 'maxResidentModels',
            label: 'Models that can stay loaded at once',
            fallback: '1',
            min: 0
        },
        {
            section: 'idle',
            key: 'unloadAfterMinutes',
            label: 'Release model memory after N idle minutes',
            fallback: '5',
            min: 0
        }
    ]
    return (
        <Stack gap="3">
            {fields.map(field => {
                const value = valid
                    ? (jsonFieldSource(text, [field.section, field.key]) ?? field.fallback)
                    : ''
                return (
                    <FormField key={field.key} slotLabel={field.label}>
                        <TextInput
                            aria-label={field.label}
                            type="number"
                            min={field.min}
                            step={1}
                            value={value}
                            disabled={disabled || !valid}
                            onChange={event => {
                                const input = event.target.value
                                const number = Number(input)
                                if (!input || !Number.isSafeInteger(number) || number < field.min)
                                    return
                                const section = jsonFieldSource(text, [field.section]) ?? '{}'
                                try {
                                    onChange(
                                        replaceJsonFieldSource(
                                            text,
                                            field.section,
                                            replaceJsonFieldSource(
                                                section,
                                                field.key,
                                                String(number)
                                            )
                                        )
                                    )
                                } catch {
                                    /* Keep malformed sections in the JSON editor for repair. */
                                }
                            }}
                        />
                        {field.key === 'unloadAfterMinutes' && (
                            <Text kind="body/regular/sm">
                                {value === '0' ? 'Never' : '0 means Never'}
                            </Text>
                        )}
                    </FormField>
                )
            })}
        </Stack>
    )
}
