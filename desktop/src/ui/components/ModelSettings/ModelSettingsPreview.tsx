// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core'
import type {
    EngineSettingsPreview,
    EngineSettingsSnapshot,
    EngineSettingsTarget
} from '@/shared/types/engine-settings'
import type { ArtifactSummary, InspectModelResult } from '@/shared/types/model-config'
import ModelMemoryEstimate from './ModelMemoryEstimate'
import ModelLoadBufferReport from './ModelLoadBufferReport'
import ModelMemoryStatedFigure from './ModelMemoryStatedFigure'

function artifactLabel(role: string): string {
    switch (role) {
        case 'model':
            return 'Main model file'
        case 'draft':
            return 'Draft model file'
        case 'companion':
            return 'Companion file'
        default:
            return `Artifact file (${role})`
    }
}
function Artifacts({ artifacts }: { artifacts?: ArtifactSummary[] }) {
    return (
        <>
            {artifacts?.map((artifact, index) => (
                <Text key={`${artifact.role}:${index}`} kind="body/regular/sm">
                    {artifactLabel(artifact.role)}: {artifact.identity} · {artifact.size} bytes ·{' '}
                    {artifact.assurance}
                </Text>
            ))}
        </>
    )
}

export default function ModelSettingsPreview({
    inspection,
    preview,
    snapshot,
    profileKey,
    reviewed = true,
    stale = false,
    target,
    disabled = false
}: {
    inspection?: InspectModelResult
    preview?: EngineSettingsPreview
    snapshot?: EngineSettingsSnapshot
    profileKey?: string
    reviewed?: boolean
    stale?: boolean
    target?: EngineSettingsTarget
    disabled?: boolean
}) {
    const impact = preview?.impact
    const estimate = preview ? (preview.auto?.estimate ?? preview.estimate) : inspection?.estimate
    const desired = snapshot?.settings.modelLoads?.find(load => load.profileKey === profileKey)
    const applied = snapshot?.effectiveSettings?.modelLoads?.find(
        load => load.profileKey === profileKey
    )
    const observed = snapshot?.serving?.observed?.find(load => load.modelId === profileKey)
    const convertedLoads =
        preview?.settings.modelLoads?.filter(
            load =>
                load.resolved?.resolverRevision === 'external-conversion/llama' ||
                (load.resolved?.resolverRevision?.startsWith('auto/') &&
                    /^[0-9a-f]{64}:/.test(load.resolved.reviewedInputId ?? ''))
        ) ?? []
    const conversion =
        snapshot && !snapshot.settings.modelLoads?.length && convertedLoads.length > 0
    const matchingInspection =
        inspection &&
        (!snapshot ||
            (snapshot.engine === inspection.engine &&
                snapshot.nodeId === (inspection.nodeId ?? '')))
            ? inspection
            : undefined
    return (
        <Stack gap="2">
            {conversion && (
                <Text role="status" kind="body/regular/sm">
                    Saving converts this engine&apos;s hand-written preset into{' '}
                    {convertedLoads.length} profiles.
                </Text>
            )}
            <Text kind="body/semibold/sm">Target review</Text>
            <Text kind="body/regular/sm">
                Artifact: {inspection?.status ?? 'Unknown'} · Source:{' '}
                {inspection?.source ?? 'Unknown'} · Assurance: {inspection?.assurance ?? 'Unknown'}
            </Text>
            <ModelMemoryEstimate estimate={estimate} stale={stale} />
            {target && (
                <ModelMemoryStatedFigure
                    key={`${target.nodeId}:${target.engine}:${profileKey ?? ''}:${matchingInspection?.modelId ?? ''}:${matchingInspection?.configurationKey ?? ''}`}
                    target={target}
                    inspection={matchingInspection}
                    profileKey={profileKey}
                    disabled={disabled}
                />
            )}
            <ModelLoadBufferReport inspection={matchingInspection} stale={!snapshot} />
            {preview?.auto && (
                <>
                    <Text kind="body/regular/sm">
                        Auto proposal: {preview.auto.status} · Requested context:{' '}
                        {JSON.stringify(preview.auto.requested.context ?? 'Inherited')} · Requested
                        constraints: {JSON.stringify(preview.auto.requested)}
                    </Text>
                    {preview.auto.resolved ? (
                        <Text kind="body/regular/sm">
                            Feasible context: {preview.auto.resolved.perSlotContext} tokens per
                            request · Total: {preview.auto.resolved.totalContext} · Slots:{' '}
                            {preview.auto.resolved.slots} · Placement:{' '}
                            {JSON.stringify(preview.auto.resolved.gpu)}. CPU placement can be
                            slower.
                        </Text>
                    ) : (
                        <Text kind="body/regular/sm">
                            Auto resolution unavailable. Manual settings remain usable.
                        </Text>
                    )}
                    {preview.auto.warnings?.map((warning, index) => (
                        <Text key={index} kind="body/regular/sm">
                            {warning}
                        </Text>
                    ))}
                </>
            )}
            <Artifacts artifacts={preview?.requiredArtifacts ?? inspection?.artifacts} />
            <Text kind="body/regular/sm">
                Facts: {inspection?.facts?.status ?? 'Unknown'} · Estimate:{' '}
                {estimate?.status ?? 'Unknown'}
            </Text>
            <Text kind="body/regular/sm">
                Desired settings revision: {snapshot?.revision ?? 'Unknown'} · Applied settings
                revision: {snapshot?.appliedRevision ?? 'Unknown'} · Observed engine:{' '}
                {snapshot ? (snapshot.running ? 'Running' : 'Off') : 'Unknown'}
            </Text>
            <Text kind="body/regular/sm">
                Desired load preferences:{' '}
                {snapshot && inspection ? JSON.stringify(desired?.requested ?? {}) : 'Unknown'}
            </Text>
            <Text kind="body/regular/sm">
                Applied load preferences:{' '}
                {snapshot?.effectiveSettings && inspection
                    ? JSON.stringify(applied?.requested ?? {})
                    : 'Unknown'}
            </Text>
            <Text kind="body/regular/sm">
                Observed model allocation:{' '}
                {observed
                    ? `${observed.perSlotContextTokens} tokens per request, ${observed.slots} slots`
                    : 'Unknown'}
            </Text>
            <Text kind="body/regular/sm">
                Serving readback stage: {snapshot?.serving?.status ?? 'Unknown'} · Stage:{' '}
                {snapshot?.coupledStage ?? snapshot?.phase ?? 'Unknown'}
            </Text>
            {snapshot?.unresolvedError && (
                <Text kind="body/regular/sm">Unresolved: {snapshot.unresolvedError}</Text>
            )}
            {snapshot?.error && (
                <Text kind="body/regular/sm">Operation error: {snapshot.error}</Text>
            )}
            {preview && (
                <>
                    <Text kind="body/regular/sm">
                        {reviewed ? 'Reviewed' : 'Advisory, not reviewed'} settings revision:{' '}
                        {preview.revision} · Policy revision: {preview.policyRevision ?? 'Unknown'}{' '}
                        · Digest: {preview.previewDigest ?? 'Unavailable'}
                    </Text>
                    {preview.diff?.map((diff, index) => (
                        <Text key={`${diff.field}:${index}`} kind="body/regular/sm">
                            {diff.field}: {JSON.stringify(diff.before) ?? 'Inherited'} →{' '}
                            {JSON.stringify(diff.after) ?? 'Inherited'}
                        </Text>
                    ))}
                    <Text kind="body/regular/sm">
                        {impact?.restartScope === 'whole-router' ||
                        impact?.restartScope === 'engine' ||
                        impact?.restartScope === 'whole-engine'
                            ? 'Restarts the whole llama.cpp engine'
                            : `Restart scope: ${impact?.restartScope ?? 'Unknown'}`}
                    </Text>
                    <Text kind="body/regular/sm">
                        Affected models:{' '}
                        {impact?.affectedModels === undefined
                            ? 'Unknown'
                            : impact.affectedModels.length
                              ? impact.affectedModels.join(', ')
                              : 'None reported'}
                    </Text>
                    <Text kind="body/regular/sm">
                        Drain bound:{' '}
                        {impact?.drainTimeoutSeconds === undefined
                            ? 'Unknown'
                            : `${impact.drainTimeoutSeconds} seconds`}
                        .{' '}
                        {impact?.mayInterruptActive === undefined
                            ? 'Active request interruption: Unknown.'
                            : impact.mayInterruptActive
                              ? 'Active requests can be interrupted after the reported drain bound.'
                              : 'No active request interruption reported.'}
                    </Text>
                    <Text kind="body/regular/sm">
                        Active work policy: {impact?.activeWorkPolicy ?? 'Unknown'}
                    </Text>
                    <Text kind="body/regular/sm">
                        Off behavior:{' '}
                        {impact?.staysOff === undefined
                            ? 'Unknown'
                            : impact.staysOff
                              ? 'Off remains Off. Save does not load a model.'
                              : 'See target impact.'}
                    </Text>
                    {impact?.unknowns?.map(reason => (
                        <Text key={reason} kind="body/regular/sm">
                            Unknown: {reason}
                        </Text>
                    ))}
                </>
            )}
        </Stack>
    )
}
