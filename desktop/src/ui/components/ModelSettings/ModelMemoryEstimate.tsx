// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core'
import type { Estimate } from '@/shared/types/model-config'
import { formatBytes } from '@/ui/utils/formatters'

function bytes(value?: string): string {
    return value === undefined ? 'Unknown' : formatBytes(Number(value))
}

function amount(expected?: string, low?: string, high?: string): string {
    return `${bytes(expected)} (up to ${bytes(high)}; range ${bytes(low)} to ${bytes(high)})`
}

function comparison(fit: Estimate['capacityFit'], stale: boolean): string {
    if (stale) return 'Unknown'
    switch (fit) {
        case 'fits':
            return 'Fits'
        case 'tight':
            return 'Tight: may not fit'
        case 'exceeds':
            return 'Exceeds available memory'
        default:
            return 'Unknown'
    }
}

function sourceLabel(kind?: NonNullable<Estimate['estimateSource']>['kind']): string {
    switch (kind) {
        case 'pair-estimate':
            return 'Estimate'
        case 'user-stated':
            return 'Your figure'
        case 'measured':
            return 'Measured'
        case 'engine-report':
            return 'Engine report'
        default:
            return 'Unknown'
    }
}

export default function ModelMemoryEstimate({
    estimate,
    stale = false
}: {
    estimate?: Estimate
    stale?: boolean
}) {
    const domains = estimate?.domains ?? []
    const shared = domains.filter(domain => domain.kind === 'shared')
    const source = estimate?.estimateSource
    return (
        <Stack gap="2">
            <Text kind="body/semibold/sm">Advisory memory estimate{stale ? ' (stale)' : ''}</Text>
            <Text kind="body/regular/sm">
                {sourceLabel(source?.kind)} · {estimate?.status ?? 'Unknown'} · Source:{' '}
                {estimate?.source ?? 'Unknown'} · Build: {estimate?.buildId ?? 'Unknown'} · Estimate
                time: {estimate?.timestamp ?? 'Unknown'}
            </Text>
            <Text kind="body/regular/sm">
                Provider: {source?.provider ?? 'Unknown'} · Revision:{' '}
                {source?.revision ?? 'Unknown'} · Basis: {source?.basis ?? 'Unknown'}
            </Text>
            <Text kind="body/regular/sm">
                Memory source: {estimate?.memorySource ?? 'Unknown'} · Available memory sampled:{' '}
                {estimate?.memoryTimestamp ?? 'Unknown'}. Selected model estimate; simultaneous
                residents are not credited as released.
            </Text>
            <Text kind="body/regular/sm">
                Service comparison: capacity ({comparison(estimate?.capacityFit, stale)}) ·
                available ({comparison(estimate?.freeFit, stale)})
            </Text>
            {stale && (
                <Text kind="body/regular/sm">
                    Memory evidence is stale. Comparisons are Unknown; retained amounts are
                    historical.
                </Text>
            )}
            {shared.length > 0 && domains.length > 1 && (
                <Text kind="body/regular/sm">
                    Shared UMA memory is displayed once. Other pool capacities are not added.
                </Text>
            )}
            {domains.length === 0 && <Text kind="body/regular/sm">Memory domains: Unknown.</Text>}
            {(shared.length ? shared.slice(0, 1) : domains).map(domain => (
                <Stack key={domain.id} gap="1">
                    <Text kind="body/regular/sm">
                        {domain.kind === 'shared'
                            ? 'Shared UMA pool'
                            : domain.kind === 'device'
                              ? `Device ${domain.deviceId ?? 'mapping unknown'}`
                              : 'Host'}{' '}
                        ({domain.id}): {sourceLabel(source?.kind)} expected{' '}
                        {amount(
                            domain.requiredBytes,
                            domain.requiredLowBytes,
                            domain.requiredHighBytes
                        )}{' '}
                        · capacity {bytes(domain.capacityBytes)} (
                        {comparison(domain.capacityFit, stale)}) · available{' '}
                        {bytes(domain.freeBytes)} ({comparison(domain.freeFit, stale)}) · service
                        reserve {bytes(domain.reserveBytes)}
                    </Text>
                    <Text kind="body/regular/sm">
                        Accounting source: {domain.source ?? 'Unknown'} · Sample:{' '}
                        {domain.sampleId ?? 'Unknown'} · Capacity source:{' '}
                        {domain.capacitySource ?? 'Unknown'} · Capacity sampled:{' '}
                        {domain.capacityTimestamp ?? 'Unknown'} · Available source:{' '}
                        {domain.freeSource ?? 'Unknown'} · Available sampled:{' '}
                        {domain.freeTimestamp ?? 'Unknown'} · Remaining validity:{' '}
                        {stale
                            ? 'Expired'
                            : domain.remainingValidityMs === undefined
                              ? 'Unknown'
                              : `${domain.remainingValidityMs} ms`}
                    </Text>
                    {domain.kind === 'device' && !domain.deviceId && (
                        <Text kind="body/regular/sm">
                            Unknown: physical device mapping unavailable.
                        </Text>
                    )}
                    {domain.unknowns?.map((item, index) => (
                        <Text key={index} kind="body/regular/sm">
                            {item.startsWith('not priced')
                                ? 'Not included in this estimate'
                                : 'Unknown'}{' '}
                            ({domain.id}): {item}
                        </Text>
                    ))}
                </Stack>
            ))}
            {estimate?.components?.map((component, index) => (
                <Text key={`${component.name}:${index}`} kind="body/regular/sm">
                    {component.name} ({component.domainId ?? 'unmapped'}): {component.status} ·
                    expected {amount(component.bytes, component.lowBytes, component.highBytes)} ·
                    Basis: {component.basis ?? 'Unknown'}
                    {component.basis === 'heuristic' ? ' (uncalibrated)' : ''}
                    {component.reason ? ` · ${component.reason}` : ''}
                </Text>
            ))}
            {!!source?.unpricedCosts?.length && (
                <Stack gap="1">
                    <Text kind="body/semibold/sm">Not included in this estimate</Text>
                    {source.unpricedCosts.map((item, index) => (
                        <Text key={index} kind="body/regular/sm">
                            Not priced: {item}
                        </Text>
                    ))}
                </Stack>
            )}
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
