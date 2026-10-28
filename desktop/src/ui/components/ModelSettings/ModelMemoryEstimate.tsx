// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core'
import type { Estimate } from '@/shared/types/model-config'
import { formatBytes } from '@/ui/utils/formatters'

function bytes(value?: string): string {
    return value === undefined ? 'Unknown' : formatBytes(Number(value))
}

function comparison(
    required: string | undefined,
    available: string | undefined,
    capacity: string | undefined,
    complete: boolean
): string {
    if (!complete || required === undefined || available === undefined || capacity === undefined)
        return 'Unknown'
    const need = BigInt(required)
    const amount = BigInt(available)
    const reserve = BigInt(capacity) / 10n + 536870912n
    if (need > amount) return 'Exceeds'
    if (need + reserve > amount) return 'Tight'
    return 'Fits'
}

export default function ModelMemoryEstimate({
    estimate,
    stale = false
}: {
    estimate?: Estimate
    stale?: boolean
}) {
    const domains = estimate?.domains ?? []
    const components = estimate?.components ?? []
    const missing = [
        'weights',
        'attention-cache',
        'recurrent-state',
        'compute-workspace',
        'prompt-cache',
        'companions'
    ].filter(name => !components.some(component => component.name === name))
    const shared = domains.filter(domain => domain.kind === 'shared')
    const independent = shared.length === 0 || (shared.length === 1 && domains.length === 1)
    const complete =
        estimate?.status === 'estimated' &&
        !estimate.unknowns?.length &&
        missing.length === 0 &&
        independent &&
        domains.every(domain => domain.kind !== 'device' || !!domain.deviceId) &&
        components.every(component =>
            component.status === 'excluded'
                ? !!component.reason && (component.bytes === undefined || component.bytes === '0')
                : component.status === 'estimated' &&
                  component.bytes !== undefined &&
                  domains.some(domain => domain.id === component.domainId)
        )
    return (
        <Stack gap="2">
            <Text kind="body/semibold/sm">Advisory memory estimate{stale ? ' (stale)' : ''}</Text>
            <Text kind="body/regular/sm">
                {estimate?.status ?? 'Unknown'} · Source: {estimate?.source ?? 'Unknown'} · Build:{' '}
                {estimate?.buildId ?? 'Unknown'} · Estimate time: {estimate?.timestamp ?? 'Unknown'}
            </Text>
            <Text kind="body/regular/sm">
                Memory source: {estimate?.memorySource ?? 'Unknown'} · Free memory sampled:{' '}
                {estimate?.memoryTimestamp ?? 'Unknown'}. Selected model estimate; simultaneous
                residents are not credited as released.
            </Text>
            {missing.length > 0 && (
                <Text kind="body/regular/sm">
                    Unknown: missing component costs or explicit exclusions: {missing.join(', ')}.
                </Text>
            )}
            {!independent && (
                <Text kind="body/regular/sm">
                    Unknown: shared UMA mapping overlaps other pools. Only the shared pool is
                    displayed; capacities are not added.
                </Text>
            )}
            {domains.length === 0 && <Text kind="body/regular/sm">Memory domains: Unknown.</Text>}
            {(shared.length ? shared.slice(0, 1) : domains).map(domain => {
                const mapped = domain.kind !== 'device' || !!domain.deviceId
                const sum = components
                    .filter(
                        component =>
                            component.status === 'estimated' && component.domainId === domain.id
                    )
                    .reduce((total, component) => total + BigInt(component.bytes ?? '0'), 0n)
                const known =
                    !!complete &&
                    mapped &&
                    !stale &&
                    domain.requiredBytes !== undefined &&
                    sum === BigInt(domain.requiredBytes)
                const reserve =
                    domain.capacityBytes === undefined
                        ? undefined
                        : (BigInt(domain.capacityBytes) / 10n + 536870912n).toString()
                return (
                    <Text key={domain.id} kind="body/regular/sm">
                        {domain.kind === 'shared'
                            ? 'Shared UMA pool'
                            : domain.kind === 'device'
                              ? `Device ${domain.deviceId ?? 'mapping unknown'}`
                              : 'Host'}{' '}
                        ({domain.id}): required {bytes(domain.requiredBytes)} · capacity{' '}
                        {bytes(domain.capacityBytes)} (
                        {comparison(
                            domain.requiredBytes,
                            domain.capacityBytes,
                            domain.capacityBytes,
                            known
                        )}
                        ) · free {bytes(domain.freeBytes)} (
                        {comparison(
                            domain.requiredBytes,
                            domain.freeBytes,
                            domain.capacityBytes,
                            known && !!estimate.memoryTimestamp
                        )}
                        ) · reserve {bytes(reserve)} (10% of capacity plus 512 MiB)
                    </Text>
                )
            })}
            {estimate?.components?.map((component, index) => (
                <Text key={`${component.name}:${index}`} kind="body/regular/sm">
                    {component.name} ({component.domainId ?? 'unmapped'}): {component.status} ·{' '}
                    {bytes(component.bytes)}
                    {component.reason ? ` · ${component.reason}` : ''}
                </Text>
            ))}
            {estimate?.assumptions?.map((item, index) => (
                <Text key={index} kind="body/regular/sm">
                    Assumption: {item}
                </Text>
            ))}
            {estimate?.unknowns?.map((item, index) => (
                <Text key={index} kind="body/regular/sm">
                    Unknown: {item}
                </Text>
            ))}
        </Stack>
    )
}
