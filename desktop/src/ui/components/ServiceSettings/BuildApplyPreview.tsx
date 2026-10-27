// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import {
    Button,
    Checkbox,
    Flex,
    ModalContent,
    ModalDialog,
    ModalHeading,
    ModalRoot,
    SidePanel,
    Stack,
    Text
} from '@nvidia/foundations-react-core'
import type {
    BuildApplyPreview as Preview,
    BuildIdentity,
    Observation
} from '@/shared/types/build-apply'

function observationLabel<T>(value: Observation<T>): string {
    return value.known
        ? Array.isArray(value.value)
            ? value.value.join(', ') || 'None'
            : String(value.value)
        : 'Unknown'
}

export function ObservedValue<T>({ value }: { value: Observation<T> }) {
    return observationLabel(value)
}

function Identity({ label, value }: { label: string; value: BuildIdentity }) {
    return (
        <div className="min-w-0 wrap-anywhere">
            <Text kind="body/semibold/sm">{label}</Text>
            <div>Build: {observationLabel(value.buildId)}</div>
            <div>
                Source: {observationLabel(value.sourceRevision)}; modified:{' '}
                {observationLabel(value.dirty)}
            </div>
            <div>SHA-256: {observationLabel(value.sha256)}</div>
        </div>
    )
}

export default function BuildApplyPreview({
    preview,
    deviceName,
    observedAt,
    open,
    onOpenChange,
    busy,
    onApply
}: {
    preview: Preview
    deviceName: string
    observedAt: string
    open: boolean
    onOpenChange: (open: boolean) => void
    busy: boolean
    onApply: (allowEngineStop: boolean) => void
}) {
    const [confirm, setConfirm] = useState(false)
    const [allowEngineStop, setAllowEngineStop] = useState(false)
    const [now, setNow] = useState(Date.now())
    useEffect(() => {
        setConfirm(false)
        setAllowEngineStop(false)
        const timer = setInterval(() => setNow(Date.now()), 1000)
        return () => clearInterval(timer)
    }, [preview])
    const impact = preview.impact
    const expiry = Date.parse(preview.expiresAt)
    const expired = !Number.isFinite(expiry) || now >= expiry
    const identities = [
        preview.currentGui,
        preview.bundledService,
        preview.runningService,
        preview.requestedGui,
        preview.requestedService
    ]
    const complete =
        preview.observationsComplete &&
        impact.complete &&
        preview.hostId.known &&
        preview.packageSha256.known &&
        preview.targetLayout.known &&
        identities.every(
            value =>
                value.buildId.known &&
                value.sourceRevision.known &&
                value.dirty.known &&
                value.sha256.known
        ) &&
        Object.values(impact.fences).every(value => value.known) &&
        Object.values(impact.expectedSettings).every(value => value.known) &&
        impact.maintenance.buildApplyActive.known &&
        !impact.maintenance.buildApplyActive.value &&
        impact.maintenance.modelSettingsActive.known &&
        !impact.maintenance.modelSettingsActive.value &&
        impact.engines.known &&
        impact.engines.value.every(engine =>
            Object.values(engine).every(value => typeof value === 'string' || value.known)
        ) &&
        impact.activeWork.known &&
        impact.modelOperations.known &&
        !impact.modelOperations.value.some(operation => operation.blocksApply) &&
        impact.settingsDifferences.known &&
        impact.availability.known &&
        impact.pauseOrigin.known &&
        impact.interruptionPolicy.onActive.known &&
        impact.interruptionPolicy.drainTimeoutSeconds.known &&
        impact.restorePolicy.remainPaused &&
        impact.restorePolicy.requireLocalResume &&
        impact.restorePolicy.preserveIntentAndLocks &&
        impact.restorePolicy.preserveNewerPause &&
        !impact.restorePolicy.automaticRollback &&
        impact.refusals.length === 0
    const canApply = complete && !expired && !busy
    return (
        <>
            <SidePanel
                id="build-preview"
                open={open}
                onOpenChange={value => {
                    setConfirm(false)
                    onOpenChange(value)
                }}
                modal={false}
                bordered
                slotHeading={`Preview build on ${deviceName}`}
                slotFooter={
                    <Flex gap="2" wrap="wrap">
                        <Button kind="secondary" onClick={() => onOpenChange(false)}>
                            Close preview
                        </Button>
                        <Button
                            kind="primary"
                            color="brand"
                            disabled={!canApply}
                            aria-describedby="build-preview-eligibility"
                            onClick={() => {
                                setAllowEngineStop(false)
                                setConfirm(true)
                            }}
                        >
                            Apply build and restart
                        </Button>
                    </Flex>
                }
            >
                <Stack gap="4" className="min-w-0 wrap-anywhere">
                    <Text kind="body/regular/sm">
                        Observed {observedAt}. Expires {preview.expiresAt}. {preview.platform} /{' '}
                        {preview.architecture}. Local host: {observationLabel(preview.hostId)}.
                    </Text>
                    <div id="build-preview-eligibility" role="status">
                        {expired
                            ? 'Preview expired. Choose the staged build again and renew consent.'
                            : complete
                              ? 'Impact observations complete.'
                              : 'Impact observations incomplete or refused. Apply is unavailable.'}
                    </div>
                    {impact.refusals.map(refusal => (
                        <div key={refusal.code} role="alert">
                            {refusal.code}: {refusal.detail}
                        </div>
                    ))}
                    <Identity label="Current GUI" value={preview.currentGui} />
                    <Identity label="Bundled service" value={preview.bundledService} />
                    <Identity label="Attached running service" value={preview.runningService} />
                    <Identity label="Requested GUI" value={preview.requestedGui} />
                    <Identity label="Requested service" value={preview.requestedService} />
                    <div>
                        Selected package SHA-256: {observationLabel(preview.packageSha256)}. Hashes
                        identify selected bytes, not publisher authenticity. Only a complete
                        same-user writable local bundle is eligible.
                    </div>
                    <div>
                        Target layout:{' '}
                        {preview.targetLayout.known
                            ? `${preview.targetLayout.value.rootName} / ${preview.targetLayout.value.executable} / ${preview.targetLayout.value.serviceDir}`
                            : 'Unknown'}
                    </div>
                    <div>
                        Availability: {observationLabel(impact.availability)}. Pause origin:{' '}
                        {observationLabel(impact.pauseOrigin)}. Active work:{' '}
                        {observationLabel(impact.activeWork)}.
                    </div>
                    <div>
                        Interruption: {observationLabel(impact.interruptionPolicy.onActive)} active
                        work; drain bound{' '}
                        {observationLabel(impact.interruptionPolicy.drainTimeoutSeconds)} seconds.
                        Engines stop and models unload. Work admitted before the pause fence is
                        included. Committed streams may fail; seamless replay is not promised.
                    </div>
                    <div>
                        Engines:{' '}
                        {impact.engines.known
                            ? impact.engines.value.map(engine => (
                                  <div key={engine.engine}>
                                      {engine.engine}: active {observationLabel(engine.active)},
                                      running {observationLabel(engine.running)}, loaded models{' '}
                                      {observationLabel(engine.loadedModels)}, saved intent{' '}
                                      {observationLabel(engine.desiredState)}, locked{' '}
                                      {observationLabel(engine.locked)}, scoped stop authority{' '}
                                      {observationLabel(engine.scopedStopAuthority)}
                                  </div>
                              ))
                            : 'Unknown'}
                    </div>
                    <div>
                        Model operations:{' '}
                        {impact.modelOperations.known
                            ? impact.modelOperations.value
                                  .map(
                                      operation =>
                                          `${operation.engine}: ${operation.kind}${operation.blocksApply ? ' (blocks apply)' : ''}`
                                  )
                                  .join(', ') || 'None'
                            : 'Unknown'}
                    </div>
                    <div>
                        Canonical settings:{' '}
                        {impact.settingsDifferences.known
                            ? impact.settingsDifferences.value.map(difference => (
                                  <div key={difference.field}>
                                      {difference.field}: stored{' '}
                                      {observationLabel(difference.stored)}, effective{' '}
                                      {observationLabel(difference.effective)}, persisted{' '}
                                      {observationLabel(difference.persisted)}
                                  </div>
                              ))
                            : 'Unknown'}
                    </div>
                    <div>
                        Expected stopEngines:{' '}
                        {observationLabel(impact.expectedSettings.stopEngines)}; unloadModels:{' '}
                        {observationLabel(impact.expectedSettings.unloadModels)}. All other policy,
                        settings, intent and locks must be preserved. Build applied; node remains
                        paused. Ordinary local Resume is required. A pre-existing or newer Pause is
                        preserved. No automatic rollback.
                    </div>
                </Stack>
            </SidePanel>
            <ModalRoot id="build-apply-confirm" open={confirm && open} onOpenChange={setConfirm}>
                <ModalDialog>
                    <ModalContent className="no-drag-elements max-content-modal min-w-0 wrap-anywhere">
                        <ModalHeading>Apply build and restart on {deviceName}?</ModalHeading>
                        <Stack gap="4">
                            <Text kind="body/regular/sm">
                                This interrupts the app and service, stops engines and unloads
                                models under the previewed policy. The node remains paused after
                                verified application. Your source, profiles and model stores are
                                preserved.
                            </Text>
                            <label className="flex gap-2">
                                <Checkbox
                                    checked={allowEngineStop}
                                    onCheckedChange={value => setAllowEngineStop(value === true)}
                                />
                                Allow interruption and engine stop
                            </label>
                            {!allowEngineStop && (
                                <div id="build-consent-error" role="status">
                                    Full service restart requires engine stop, even if engines
                                    currently appear stopped.
                                </div>
                            )}
                            <Flex gap="2" wrap="wrap" justify="end">
                                <Button kind="secondary" onClick={() => setConfirm(false)}>
                                    Cancel
                                </Button>
                                <Button
                                    kind="primary"
                                    color="danger"
                                    disabled={!canApply || !allowEngineStop}
                                    aria-describedby={
                                        !allowEngineStop
                                            ? 'build-consent-error'
                                            : 'build-preview-eligibility'
                                    }
                                    onClick={() => {
                                        setConfirm(false)
                                        onApply(allowEngineStop)
                                    }}
                                >
                                    Apply build and restart
                                </Button>
                            </Flex>
                        </Stack>
                    </ModalContent>
                </ModalDialog>
            </ModalRoot>
        </>
    )
}
