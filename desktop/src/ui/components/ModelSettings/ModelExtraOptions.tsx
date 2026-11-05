// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useId, useRef, useState } from 'react'
import {
    Button,
    FormField,
    Select,
    Stack,
    Switch,
    Text,
    TextInput
} from '@nvidia/foundations-react-core'
import type { InspectModelResult, RequestedLoad } from '@/shared/types/model-config'

type Row = { id: string; key: string; value: string }

function syntaxError(rows: Row[]): string | undefined {
    if (rows.length > 32) return 'Extra options: use at most 32 entries.'
    const keys = new Set<string>()
    for (const [index, row] of rows.entries()) {
        const label = `Extra option ${index + 1}`
        if (!/^[a-z0-9][a-z0-9-]{0,63}$/.test(row.key))
            return `${label}: enter a key of 1 to 64 lowercase letters, digits or hyphens, starting with a letter or digit.`
        if (keys.has(row.key)) return `${label}: duplicate key ${row.key}.`
        keys.add(row.key)
        if (/[\uD800-\uDFFF]/u.test(row.value)) return `${label}: enter valid UTF-8 text.`
        if (new TextEncoder().encode(row.value).byteLength > 1024)
            return `${label}: the value must be at most 1024 UTF-8 bytes.`
        if (/\p{Cc}/u.test(row.value)) return `${label}: control characters are not allowed.`
        if (/[#;$%{}`]/.test(row.value))
            return `${label}: the value cannot contain # ; $ % { } or a backtick.`
    }
    return undefined
}

export default function ModelExtraOptions({
    descriptor,
    value,
    nativeOptions,
    disabled,
    reset,
    onChange
}: {
    descriptor: NonNullable<InspectModelResult['options']>[number]
    value?: RequestedLoad['extraOptions']
    nativeOptions?: InspectModelResult['nativeOptions']
    disabled: boolean
    reset: boolean
    onChange: (options: RequestedLoad['extraOptions'], error?: string) => void
}) {
    const [draftRows, setDraftRows] = useState<Row[]>()
    const nextId = useRef(0)
    const suggestionsId = useId()
    const rows =
        draftRows ??
        Object.entries(value ?? {}).map(([key, optionValue]) => ({
            id: `saved:${key}`,
            key,
            value: optionValue
        }))
    useEffect(() => {
        if (reset) setDraftRows(undefined)
    }, [reset])
    const change = (next: Row[]) => {
        setDraftRows(next)
        const error = syntaxError(next)
        onChange(
            next.length ? Object.fromEntries(next.map(row => [row.key, row.value])) : undefined,
            error
        )
    }
    const update = (index: number, patch: Partial<Row>) =>
        change(rows.map((row, position) => (position === index ? { ...row, ...patch } : row)))
    const error = syntaxError(rows)
    return (
        <Stack gap="2">
            <Text kind="body/semibold/sm">{descriptor.label}</Text>
            <Text kind="body/regular/sm">
                Extra options are passed to the engine. PAIR does not price all option costs.{' '}
                {descriptor.reason}
            </Text>
            <datalist id={suggestionsId}>
                {nativeOptions?.map(option => (
                    <option key={option.name} value={option.name} />
                ))}
            </datalist>
            {rows.map((row, index) => {
                const hint = nativeOptions?.find(option => option.name === row.key)
                const label = `Extra option value ${index + 1}`
                return (
                    <Stack key={row.id} gap="2">
                        <FormField slotLabel={`Extra option key ${index + 1}`}>
                            <TextInput
                                aria-label={`Extra option key ${index + 1}`}
                                list={suggestionsId}
                                value={row.key}
                                disabled={disabled}
                                onValueChange={key => {
                                    const hint = nativeOptions?.find(option => option.name === key)
                                    update(index, {
                                        key,
                                        value:
                                            hint?.kind === 'switch'
                                                ? String(['true', 'on', '1'].includes(row.value))
                                                : row.value
                                    })
                                }}
                            />
                        </FormField>
                        <FormField slotLabel={label}>
                            {hint?.kind === 'switch' ? (
                                <Switch
                                    aria-label={label}
                                    checked={['true', 'on', '1'].includes(row.value)}
                                    disabled={disabled}
                                    onCheckedChange={checked =>
                                        update(index, { value: String(checked) })
                                    }
                                />
                            ) : hint?.kind === 'choice' ? (
                                <Select
                                    aria-label={label}
                                    value={row.value}
                                    disabled={disabled}
                                    items={[...new Set([row.value, ...(hint.choices ?? [])])].map(
                                        choice => ({
                                            value: choice,
                                            children: choice || 'Choose a value'
                                        })
                                    )}
                                    onValueChange={optionValue =>
                                        update(index, { value: optionValue })
                                    }
                                />
                            ) : (
                                <TextInput
                                    aria-label={label}
                                    type={hint?.kind === 'number' ? 'number' : 'text'}
                                    value={row.value}
                                    disabled={disabled}
                                    onValueChange={optionValue =>
                                        update(index, { value: optionValue })
                                    }
                                />
                            )}
                        </FormField>
                        <Button
                            kind="tertiary"
                            size="small"
                            disabled={disabled}
                            onClick={() => change(rows.filter((_, position) => position !== index))}
                        >
                            Remove extra option {index + 1}
                        </Button>
                    </Stack>
                )
            })}
            {error && (
                <Text role="alert" kind="body/regular/sm">
                    {error}
                </Text>
            )}
            <Button
                kind="secondary"
                disabled={disabled || rows.length >= 32}
                onClick={() =>
                    change([...rows, { id: `draft:${nextId.current++}`, key: '', value: '' }])
                }
            >
                Add extra option
            </Button>
            <Button kind="tertiary" size="small" disabled={disabled} onClick={() => change([])}>
                Clear extra options
            </Button>
        </Stack>
    )
}
