// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core'
import type { EngineSettingsPreview, EngineSettingsSnapshot } from '@/shared/types/engine-settings'
import type { ArtifactSummary, InspectModelResult } from '@/shared/types/model-config'

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
    snapshot
}: {
    inspection?: InspectModelResult
    preview?: EngineSettingsPreview
    snapshot?: EngineSettingsSnapshot
}) {
    const impact = preview?.impact
    const desired = snapshot?.settings.modelLoads?.find(
        load => load.modelId === inspection?.modelId
    )
    const applied = snapshot?.effectiveSettings?.modelLoads?.find(
        load => load.modelId === inspection?.modelId
    )
    const observed = snapshot?.serving?.observed?.find(load => load.modelId === inspection?.modelId)
    return (
        <Stack gap="2">
            <Text kind="body/semibold/sm">Target review</Text>
            <Text kind="body/regular/sm">
                Artifact: {inspection?.status ?? 'Unknown'} · Source:{' '}
                {inspection?.source ?? 'Unknown'} · Assurance: {inspection?.assurance ?? 'Unknown'}
            </Text>
            <Artifacts artifacts={preview?.requiredArtifacts ?? inspection?.artifacts} />
            <Text kind="body/regular/sm">
                Facts: {inspection?.facts?.status ?? 'Unknown'} · Estimate:{' '}
                {preview?.estimate?.status ?? inspection?.estimate.status ?? 'Unknown'}
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
                        Reviewed settings revision: {preview.revision} · Policy revision:{' '}
                        {preview.policyRevision ?? 'Unknown'} · Digest:{' '}
                        {preview.previewDigest ?? 'Unavailable'}
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
