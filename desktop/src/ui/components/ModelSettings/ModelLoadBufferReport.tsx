// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Table, Text } from '@nvidia/foundations-react-core'
import type { InspectModelResult } from '@/shared/types/model-config'

export default function ModelLoadBufferReport({
    inspection,
    stale = false
}: {
    inspection?: InspectModelResult
    stale?: boolean
}) {
    const observation = inspection?.loadObservation
    const supported = observation && observation.status !== 'unknown'
    const rows = supported ? (observation.rows ?? []) : []
    const baseline = supported ? (observation.baseline ?? []) : []
    return (
        <Stack gap="2" style={{ overflowWrap: 'anywhere' }}>
            <Text kind="body/semibold/sm">
                Estimated versus native-reported buffers for this load
            </Text>
            <Text kind="body/regular/sm">
                Rounded allocation reports, not physical GPU usage, exact native bytes or a process
                total.
            </Text>
            {!supported ? (
                <Text kind="body/regular/sm">
                    Unknown ({observation?.reason?.replaceAll('-', ' ') ?? 'observation omitted'}).
                </Text>
            ) : (
                <>
                    <Text kind="body/regular/sm">
                        {observation.current && !stale
                            ? 'Current load at inspection'
                            : 'Historical load comparison'}
                        {stale ? ' (inspection stale; current state Unknown)' : ''} ·{' '}
                        {observation.status}
                        {' · '}
                        {inspection.modelId} · Load time: {observation.recordedAt ?? 'Unknown'}
                        {' · '}Phase: {observation.loadPhase ?? 'Unknown'}
                    </Text>
                    <Text kind="body/regular/sm">
                        Save:{' '}
                        {observation.persistence === 'uncertain'
                            ? 'Durability uncertain (replacement landed)'
                            : (observation.persistence ?? 'Unknown')}
                        .
                        {observation.persistence === 'failed'
                            ? ' Sample save failed; shown from memory.'
                            : ''}
                        {observation.persistence === 'pending'
                            ? ' Sample persistence pending or omitted.'
                            : ''}
                        {observation.persistence === 'history-unavailable'
                            ? ' Local history unavailable.'
                            : ''}
                    </Text>
                    <Text kind="body/regular/sm">
                        Upstream pin: {observation.upstreamPin ?? 'Unknown'} · Sample:{' '}
                        {observation.sampleId ?? 'Unknown'}. Recognized self-reported build;
                        clean-source provenance is unverified.
                    </Text>
                    <Text kind="body/regular/sm">
                        Frozen baseline for captured load, independent of the edited draft and Auto
                        proposal. Estimator: {observation.estimatorRevision ?? 'Unknown'}. Captured
                        options: {JSON.stringify(observation.capturedOptions ?? {})}. Requested
                        options: {JSON.stringify(observation.requestedOptions ?? {})}. Unresolved
                        options: {observation.optionUnknowns?.join(', ') || 'None reported'}.
                    </Text>
                    <Table
                        density="compact"
                        layout="auto"
                        columns={[
                            'Component / buffer',
                            'Device / pool',
                            'Estimate for this load',
                            'Native-reported size',
                            'Status / coverage'
                        ]}
                        rows={[
                            ...rows.map(row => ({
                                id: `frame:${row.stream ?? ''}:${row.sequence}`,
                                cells: [
                                    `${row.component} / ${row.label}`,
                                    row.poolIdentity
                                        ? `Pool ${row.poolIdentity}${row.deviceBinding ? `; binding ${row.deviceBinding}` : ''}`
                                        : 'Unmapped',
                                    row.estimatedBytes === undefined
                                        ? 'Unknown per buffer'
                                        : `${row.estimatedBytes} estimated bytes`,
                                    `${row.reportedMiB} MiB (precision ${row.precision}; log-scraped)${row.intervalStatus === 'bounded' ? `; rounded interval [${row.lowerBytes}, ${row.upperBytes}] bytes` : '; byte interval Unknown'}`,
                                    `${row.coverage}; ${row.role}; ${row.phase}`
                                ]
                            })),
                            ...baseline.map(component => {
                                const comparison = observation.comparisons?.find(
                                    item => item.component === component.component
                                )
                                return {
                                    id: `baseline:${component.component}`,
                                    cells: [
                                        `${component.component} (component)`,
                                        'Component scope',
                                        component.bytes === undefined
                                            ? `Unknown (${component.status})`
                                            : `${component.bytes} estimated bytes (${component.status})`,
                                        comparison?.status === 'compatible'
                                            ? `Rounded interval [${comparison.lowerBytes}, ${comparison.upperBytes}] bytes`
                                            : 'Unknown compatible scope',
                                        comparison?.status === 'compatible'
                                            ? `Compatible; reported minus estimated [${comparison.deltaLowerBytes}, ${comparison.deltaUpperBytes}] bytes`
                                            : 'Unknown comparison'
                                    ]
                                }
                            })
                        ]}
                    />
                    <Text kind="body/regular/sm">
                        Coverage: {observation.reason?.replaceAll('-', ' ') ?? 'partial'}
                        {observation.reasons?.length ? `; ${observation.reasons.join(', ')}` : ''}.
                        Unreported allocations are Unknown. No fit, qualification or calibration is
                        inferred.
                    </Text>
                </>
            )}
        </Stack>
    )
}
