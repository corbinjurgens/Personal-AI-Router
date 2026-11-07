// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react'
import { Button, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { EngineSettingsTarget } from '@/shared/types/engine-settings'
import type { InspectModelResult, SetStatedMemoryRequest } from '@/shared/types/model-config'
import { POLICY_CONFLICT_CODE } from '@/shared/constants/service-errors'
import getErrorString from '@/shared/utils/get-error-string'
import { serviceErrorCode } from '@/ui/api/service-transport'
import { useEngineSettingsStore } from '@/ui/stores/engine-settings.store'

const gib = 1073741824n
const maximumBytes = 18446744073709551615n
type Pool = 'host' | 'device' | 'shared'

function toGiB(bytes?: string): string {
    if (bytes === undefined) return ''
    const value = BigInt(bytes)
    const fraction = ((value % gib) * 5n ** 30n).toString().padStart(30, '0').replace(/0+$/, '')
    return `${value / gib}${fraction ? `.${fraction}` : ''}`
}

function toBytes(value: string): string | undefined {
    if (!/^[0-9]{1,20}(\.[0-9]{1,30})?$/.test(value)) return undefined
    const [whole, fraction = ''] = value.split('.')
    const scale = 10n ** BigInt(fraction.length)
    const numerator = (BigInt(whole) * scale + BigInt(fraction || '0')) * gib
    if (numerator % scale !== 0n) return undefined
    const result = numerator / scale
    return result <= maximumBytes ? result.toString() : undefined
}

const labels: Record<Pool, string> = {
    host: 'Host memory (GiB)',
    device: 'Device memory (GiB)',
    shared: 'Shared UMA memory (GiB)'
}

export default function ModelMemoryStatedFigure({
    inspection,
    target,
    profileKey,
    disabled = false
}: {
    inspection?: InspectModelResult
    target: EngineSettingsTarget
    profileKey?: string
    disabled?: boolean
}) {
    const saved = inspection?.memoryProfiles?.find(
        profile => profile.current && profile.configurationKey === inspection.configurationKey
    )
    const [values, setValues] = useState<Record<Pool, string>>({ host: '', device: '', shared: '' })
    const [pending, setPending] = useState(false)
    const [message, setMessage] = useState<string>()
    const operation = useRef({ token: 0 })
    const identity = JSON.stringify([
        target.nodeId,
        target.engine,
        profileKey,
        inspection?.modelId,
        inspection?.configurationKey
    ])
    useEffect(() => {
        const currentOperation = operation.current
        currentOperation.token++
        setValues({
            host: toGiB(saved?.hostBytes),
            device: toGiB(saved?.deviceBytes),
            shared: toGiB(saved?.sharedBytes)
        })
        setPending(false)
        setMessage(undefined)
        return () => {
            currentOperation.token++
        }
    }, [identity, saved])
    const pools = inspection?.configurationPools ?? []
    const mutate = async (clear: boolean) => {
        if (disabled || pending || !inspection?.configurationKey || !pools.length) return
        const request: SetStatedMemoryRequest & EngineSettingsTarget = {
            ...target,
            modelId: inspection.modelId,
            ...(profileKey === undefined ? {} : { profileKey }),
            configurationKey: inspection.configurationKey
        }
        if (!clear) {
            for (const pool of pools) {
                const bytes = toBytes(values[pool])
                if (bytes === undefined) {
                    setMessage(
                        'Enter each required GiB amount as an exact whole number of bytes within the supported range.'
                    )
                    return
                }
                if (pool === 'host') request.hostBytes = bytes
                else if (pool === 'device') request.deviceBytes = bytes
                else request.sharedBytes = bytes
            }
        }
        const token = ++operation.current.token
        setPending(true)
        setMessage(undefined)
        try {
            await useEngineSettingsStore.getState().saveStatedMemory(request, clear)
        } catch (error) {
            if (operation.current.token === token)
                setMessage(
                    serviceErrorCode(error) === POLICY_CONFLICT_CODE
                        ? 'Settings changed; reinspect and try again.'
                        : getErrorString(error)
                )
        } finally {
            if (operation.current.token === token) setPending(false)
        }
    }
    if (!inspection?.configurationKey)
        return (
            <Text kind="body/regular/sm">
                Stated figure unavailable:{' '}
                {inspection?.configurationKeyReason ?? 'Configuration key unavailable'}.
            </Text>
        )
    return (
        <Stack gap="2">
            <Text kind="body/semibold/sm">I know how much this uses</Text>
            <Text kind="body/regular/sm">
                For the applied configuration. Your figure (not verified by PAIR).
            </Text>
            {saved && <Text kind="body/regular/sm">Saved: {saved.recordedAt}</Text>}
            {pools.map(pool => (
                <FormField key={pool} slotLabel={labels[pool]}>
                    <TextInput
                        aria-label={labels[pool]}
                        inputMode="decimal"
                        value={values[pool]}
                        disabled={disabled || pending}
                        onValueChange={value =>
                            setValues(previous => ({ ...previous, [pool]: value }))
                        }
                    />
                </FormField>
            ))}
            {!pools.length && (
                <Text kind="body/regular/sm">
                    Required memory pools unavailable. Reinspect before saving.
                </Text>
            )}
            <Button
                size="small"
                disabled={disabled || pending || !pools.length}
                onClick={() => void mutate(false)}
            >
                Save figure
            </Button>
            <Button
                size="small"
                kind="secondary"
                disabled={disabled || pending || !saved}
                onClick={() => void mutate(true)}
            >
                Clear figure
            </Button>
            {message && (
                <Text role="alert" kind="body/regular/sm">
                    {message}
                </Text>
            )}
        </Stack>
    )
}
