// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Button, Stack, Text } from '@nvidia/foundations-react-core'
import type { InspectModelResult, MeasuredProfile } from '@/shared/types/model-config'
import type { DiagnosticPreviewRequest } from '@/shared/types/model-diagnostic'
import { engineTypeFromManagerName } from '@/shared/utils/engines'
import { ConfirmModal } from '@/ui/components/ConfirmModal'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { useEngineSettingsStore, watchModelDiagnostic } from '@/ui/stores/engine-settings.store'
import { formatBytes } from '@/ui/utils/formatters'

function bytes(value?: string): string {
    return value === undefined ? 'Not sampled' : formatBytes(Number(value))
}

function rate(value?: number): string {
    return value === undefined ? 'Not reported' : `${value.toFixed(1)} tokens/s`
}

function MeasurementResult({
    result,
    nativeBuildId,
    current
}: {
    result: MeasuredProfile
    nativeBuildId?: string
    current?: boolean
}) {
    const earlierBuild = !!nativeBuildId && nativeBuildId !== result.nativeBuildId
    return (
        <Stack gap="2">
            <Text kind="body/semibold/sm">
                Measured on this machine, {result.recordedAt}, build {result.buildPin}
            </Text>
            {earlierBuild && (
                <Text kind="body/regular/sm">
                    Earlier build: measured on {result.buildPin}. This is historical evidence;
                    measure the current build to check this configuration.
                </Text>
            )}
            {current === false && !earlierBuild && (
                <Text kind="body/regular/sm">
                    Historical measurement; configuration or machine identity is no longer current.
                </Text>
            )}
            {current === true && !earlierBuild && (
                <Text kind="body/regular/sm">
                    {result.outcome === 'completed'
                        ? 'Current measurement for this configuration.'
                        : 'Current configuration and machine identity.'}
                </Text>
            )}
            {current === undefined && !earlierBuild && (
                <Text kind="body/regular/sm">
                    Measurement currentness awaits inspection of this machine and configuration.
                </Text>
            )}
            {result.executionKey && (
                <Text kind="body/regular/sm">
                    Diagnostic only ({result.configurationKeyReason}). No memory estimate or
                    estimate comparison is available for this configuration.
                </Text>
            )}
            {result.outcome !== 'completed' && (
                <Text kind="body/regular/sm">
                    Did not complete: {result.failedPhase ?? 'Unknown phase'} -{' '}
                    {result.reason ?? result.outcome}
                </Text>
            )}
            {result.pools.map(pool => (
                <Stack key={pool.domainId} gap="1">
                    <Text kind="body/semibold/sm">
                        {pool.kind} pool {pool.domainId}: Measured peak {bytes(pool.peakUsedBytes)}
                    </Text>
                    <Text kind="body/regular/sm">
                        Baseline available: {bytes(pool.baselineAvailableBytes)}. Sampled peak,
                        {pool.samples} samples, interval {pool.intervalMs} ms, source {pool.source}.
                        Sampling can miss brief peaks.
                    </Text>
                </Stack>
            ))}
            {(['low', 'high'] satisfies Array<'low' | 'high'>).map(name => {
                const phase = result[name]
                if (!phase)
                    return (
                        <Text key={name} kind="body/regular/sm">
                            {name === 'low' ? 'Short prompt' : 'Full context'}: Not sampled.
                        </Text>
                    )
                return (
                    <Stack key={name} gap="1">
                        {phase.requests.map((request, index) => (
                            <Text key={index} kind="body/regular/sm">
                                {name === 'low' ? 'Short prompt' : 'Full context'} (
                                {request.targetPromptTokens} tokens), request {index + 1}: actual
                                prompt {request.promptTokens ?? 'Not reported'} tokens; prompt rate{' '}
                                {rate(request.promptPerSecond)}; decode rate{' '}
                                {rate(request.decodePerSecond)}; generated{' '}
                                {request.generatedTokens ?? 'Not reported'} tokens.
                            </Text>
                        ))}
                        <Text kind="body/regular/sm">
                            Combined throughput: {rate(phase.combinedGeneratedPerSecond)}.{' '}
                            {phase.coverage === 'partial'
                                ? 'Partial coverage.'
                                : 'Complete coverage.'}
                            {name === 'high' &&
                                !phase.overlapWindowSampled &&
                                ' Concurrent overlap was not sampled.'}
                        </Text>
                    </Stack>
                )
            })}
            {result.notes.map((note, index) => (
                <Text key={index} kind="body/regular/sm">
                    {note}
                </Text>
            ))}
        </Stack>
    )
}

/** Measures only an inspected, applied profile; drafts never become launch requests. */
export default function ModelMemoryDiagnostic({
    nodeId,
    engine,
    modelId,
    profileKey,
    inspection,
    applied,
    open = true,
    disabled = false
}: {
    nodeId: string
    engine: string
    modelId: string
    profileKey?: string
    inspection?: InspectModelResult
    applied: boolean
    open?: boolean
    disabled?: boolean
}) {
    const [consentOpen, setConsentOpen] = useState(false)
    const connected = useConnectionStore(state => state.connected)
    const entry = useEngineSettingsStore(state => state.diagnostics[nodeId])
    const engineType = engineTypeFromManagerName(engine)
    const descriptor = useEngineStatusStore(state =>
        engineType ? state.descriptors.get(engineType) : undefined
    )
    const family = engine === 'llamacpp' || descriptor?.family === 'llamacpp'
    const applicable = !!(
        family &&
        applied &&
        profileKey &&
        inspection?.engine === engine &&
        inspection.modelId === modelId &&
        (!inspection.nodeId || inspection.nodeId === nodeId) &&
        (inspection.configurationKey ||
            (inspection.executionKey &&
                (inspection.configurationKeyReason === 'native-fit' ||
                    inspection.configurationKeyReason === 'gpu-selection')))
    )
    const identity = JSON.stringify([
        nodeId,
        engine,
        modelId,
        profileKey,
        applicable,
        disabled,
        inspection?.configurationKey,
        inspection?.executionKey
    ])
    useEffect(() => {
        if (open) return watchModelDiagnostic(nodeId)
        return undefined
    }, [nodeId, open])
    useEffect(() => {
        setConsentOpen(false)
        if (open && applicable && !disabled && connected && profileKey)
            void useEngineSettingsStore
                .getState()
                .previewDiagnostic({ nodeId, engine, modelId, profileKey })
        // The identity includes every applied input; service refusals remain service-owned.
    }, [identity, connected, applicable, disabled, nodeId, engine, modelId, profileKey, open])
    const run = entry?.run
    const active = run?.outcome === 'running' || run?.outcome === 'cleanup-failed'
    const preview = entry?.preview
    const matchingPreview =
        applicable &&
        preview?.engine === engine &&
        preview.modelId === modelId &&
        preview.profileKey === profileKey &&
        preview.configurationKey === inspection?.configurationKey &&
        preview.executionKey === inspection?.executionKey
    const ready = !!(
        connected &&
        entry?.current &&
        !disabled &&
        !active &&
        !entry?.operation &&
        matchingPreview &&
        preview &&
        Date.parse(preview.expiresAt) > Date.now()
    )
    const currentRun =
        run?.engine === engine && run.modelId === modelId && run.profileKey === profileKey
    useEffect(() => {
        if (open && connected && applied && profileKey && currentRun && !active && !disabled)
            void useEngineSettingsStore
                .getState()
                .inspectModel({ nodeId, engine }, modelId, profileKey)
    }, [
        connected,
        applied,
        profileKey,
        currentRun,
        active,
        disabled,
        open,
        run?.runId,
        run?.result?.recordedAt,
        nodeId,
        engine,
        modelId
    ])
    const stored = inspection?.memoryProfiles?.filter(profile => profile.measured) ?? []
    const result = currentRun ? run.result : undefined
    const resultProfile = stored.find(profile => profile.measured?.runId === result?.runId)
    if (!applicable && !run && !stored.length) return null
    const request: DiagnosticPreviewRequest | undefined = profileKey
        ? { nodeId, engine, modelId, profileKey }
        : undefined
    return (
        <Stack gap="3">
            <Text kind="body/semibold/sm">Memory measurement</Text>
            {applicable && (
                <>
                    <Button
                        kind="secondary"
                        size="small"
                        disabled={!ready}
                        onClick={() => setConsentOpen(true)}
                    >
                        Measure this configuration
                    </Button>
                    {disabled && (
                        <Text kind="body/regular/sm">
                            Save or discard draft changes before measuring.
                        </Text>
                    )}
                    {entry?.refusal && (
                        <Text role="status" kind="body/regular/sm">
                            {entry.refusal}
                        </Text>
                    )}
                    {!active && !entry?.operation && request && (
                        <Button
                            kind="tertiary"
                            size="small"
                            disabled={!connected || disabled}
                            onClick={() =>
                                void useEngineSettingsStore.getState().previewDiagnostic(request)
                            }
                        >
                            Refresh measurement preview
                        </Button>
                    )}
                </>
            )}
            {entry?.operation === 'preview' && (
                <Text role="status" kind="body/regular/sm">
                    Checking measurement availability...
                </Text>
            )}
            {entry?.operation === 'start' && (
                <Text role="status" kind="body/regular/sm">
                    Starting measurement...
                </Text>
            )}
            {run && active && (
                <>
                    <Text role="status" kind="body/regular/sm">
                        Measuring memory: {run.phase}, {Math.floor(run.elapsedMs / 1000)} s elapsed.
                        {!currentRun && ` Profile ${run.profileKey} (${run.engine}).`}
                        {!entry?.current && ' Reconnecting to measurement status.'}
                    </Text>
                    {run.outcome === 'cleanup-failed' && (
                        <Text role="alert" kind="body/regular/sm">
                            Cleanup failed. {run.reason} Engines remain held until cleanup succeeds.
                        </Text>
                    )}
                    <Button
                        kind="secondary"
                        size="small"
                        disabled={
                            !connected ||
                            !entry?.current ||
                            (!!entry.operation && entry.operation !== 'start')
                        }
                        onClick={() =>
                            void useEngineSettingsStore.getState().cancelDiagnostic(nodeId)
                        }
                    >
                        {run.outcome === 'cleanup-failed' ? 'Retry cleanup' : 'Cancel measurement'}
                    </Button>
                </>
            )}
            {entry?.message && (
                <Text role="status" kind="body/regular/sm">
                    {entry.message}
                </Text>
            )}
            {currentRun && !active && run.outcome !== 'completed' && !result && (
                <Text kind="body/regular/sm">
                    Did not complete: {run.failedPhase ?? run.phase} - {run.reason ?? run.outcome}
                </Text>
            )}
            {result && (
                <MeasurementResult
                    result={result}
                    nativeBuildId={inspection?.nativeBuildId}
                    current={resultProfile?.current}
                />
            )}
            {stored.map(
                profile =>
                    profile.measured &&
                    profile.measured.runId !== result?.runId && (
                        <MeasurementResult
                            key={profile.measured.runId}
                            result={profile.measured}
                            nativeBuildId={inspection?.nativeBuildId}
                            current={profile.current}
                        />
                    )
            )}
            <ConfirmModal
                open={consentOpen}
                onOpenChange={setConsentOpen}
                title="Measure this configuration"
                confirmLabel="Measure"
                confirmDisabled={!ready}
                onConfirm={() => {
                    if (ready && request && preview)
                        void useEngineSettingsStore.getState().startDiagnostic({
                            ...request,
                            consent: true,
                            previewId: preview.previewId
                        })
                }}
                message={
                    preview && matchingPreview ? (
                        <Stack gap="2">
                            <Text kind="body/regular/sm">
                                Profile {preview.profileKey}, model {preview.modelId} (
                                {preview.engine}).
                            </Text>
                            <Text kind="body/regular/sm">
                                This uses as much memory as the configuration needs at full context.
                                If it does not fit, the engine can stop; PAIR will record that.
                            </Text>
                            <Text kind="body/regular/sm">
                                Models unloaded:{' '}
                                {preview.unloadModels
                                    .map(
                                        model =>
                                            `${model.engine}: ${model.modelId} (${model.profileKey})`
                                    )
                                    .join(', ') || 'None'}
                                .
                            </Text>
                            <Text kind="body/regular/sm">
                                Models reloaded afterwards:{' '}
                                {preview.reloadedAfter
                                    .map(
                                        model =>
                                            `${model.engine}: ${model.modelId} (${model.profileKey})`
                                    )
                                    .join(', ') || 'None'}
                                .
                            </Text>
                            <Text kind="body/regular/sm">
                                Engines held: {preview.drainEngines.join(', ') || 'None'}. New work
                                cannot start during measurement.
                            </Text>
                            <Text kind="body/regular/sm">
                                {preview.startsEngine
                                    ? `Starts ${preview.engine} for this measurement and stops it afterwards.`
                                    : 'The measured engine is already running.'}{' '}
                                Saved pause state stays unchanged.
                            </Text>
                            <Text kind="body/regular/sm">
                                Duration bound: {preview.timeLimitSeconds} seconds.
                            </Text>
                            <Text kind="body/regular/sm">
                                {preview.highPromptTokens > 0
                                    ? `Planned targets: short prompt ${preview.lowPromptTokens} tokens; full context ${preview.highPromptTokens} tokens; ${preview.generateTokens} generated tokens per request, ${preview.plannedSlots} slots. Actual allocation is verified after load.`
                                    : `${preview.configurationKeyReason === 'native-fit' ? 'Native-fit context' : 'Context'} and prompt targets are determined after load.`}
                            </Text>
                            <Text kind="body/regular/sm">
                                Bound to the hardware on this machine, build {preview.nativeBuildId}
                                , model artifacts and realized context and placement. Other GPU
                                activity can affect available memory and sampled peaks.
                            </Text>
                            <Text kind="body/regular/sm">
                                CPU: {preview.cpuModel}. RAM: {bytes(preview.totalRamBytes)}. GPUs:{' '}
                                {preview.gpus
                                    .map(
                                        gpu =>
                                            `${gpu.name} (${gpu.uuid}), ${bytes(gpu.totalMemoryBytes)}`
                                    )
                                    .join('; ') || 'None'}
                                .
                            </Text>
                            {preview.warnings.map((warning, index) => (
                                <Text key={index} kind="body/regular/sm">
                                    {warning}
                                </Text>
                            ))}
                            {preview.executionKey && (
                                <Text kind="body/regular/sm">
                                    This unpriced configuration produces diagnostic results only,
                                    without estimates or Auto selection.
                                </Text>
                            )}
                        </Stack>
                    ) : (
                        'Refresh the measurement preview before consenting.'
                    )
                }
            />
        </Stack>
    )
}
