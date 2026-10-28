// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Stack, Text } from '@nvidia/foundations-react-core'
import type { BuildApplyOperation } from '@/shared/types/build-apply'
import { ObservedValue } from './BuildApplyPreview'

export default function BuildApplyResult({
    operation,
    onRead,
    busy,
    uncertain = false
}: {
    operation: BuildApplyOperation
    onRead: () => void
    busy: boolean
    uncertain?: boolean
}) {
    const success =
        !uncertain &&
        operation.outcome === 'succeeded' &&
        operation.phase === 'finish' &&
        operation.failure === null &&
        operation.readbackConfidence === 'verified' &&
        [operation.runningService, operation.runningGui].every(
            identity =>
                identity.buildId.known &&
                identity.sourceRevision.known &&
                identity.dirty.known &&
                identity.sha256.known
        ) &&
        operation.remainsPaused.known &&
        operation.remainsPaused.value &&
        operation.saved.known &&
        operation.saved.value &&
        operation.enforced.known &&
        operation.enforced.value &&
        operation.physicalStop.known &&
        operation.physicalStop.value
    const recovery =
        uncertain || (operation.outcome !== null && !success && operation.outcome !== 'refused')
    const command = /^[a-zA-Z0-9_-]{1,128}$/.test(operation.operationId)
        ? `nvpair-service apply-build recover ${operation.operationId}`
        : null
    // Copy only allowlisted operational facts. Service detail/expected policy is
    // deliberately excluded from clipboard diagnostics.
    const diagnostics = JSON.stringify({
        operationId: operation.operationId,
        phase: operation.phase,
        outcome: operation.outcome,
        ownershipAcknowledged: operation.ownershipAcknowledged,
        readbackConfidence: operation.readbackConfidence,
        reason: operation.failure?.reason,
        bytesChanged: operation.failure?.bytesChanged,
        settingsChanged: operation.failure?.settingsChanged,
        runningService: operation.runningService,
        runningGui: operation.runningGui
    })
    return (
        <Stack
            gap="2"
            className="min-w-0 wrap-anywhere"
            role="status"
            aria-label="Build application result"
        >
            <Text kind="body/semibold/md">
                {uncertain
                    ? 'Build status is uncertain'
                    : success
                      ? 'Build applied; node remains paused'
                      : operation.outcome === 'succeeded'
                        ? 'Build readback incomplete'
                        : `Build application: ${operation.outcome ?? 'in progress'}`}
            </Text>
            <div>
                Operation {operation.operationId}. Phase:{' '}
                {operation.failure?.phase ?? operation.phase}. Ownership:{' '}
                {operation.ownershipAcknowledged ? 'acknowledged' : 'not acknowledged'}.
            </div>
            <div>
                Saved: <ObservedValue value={operation.saved} />. Enforced:{' '}
                <ObservedValue value={operation.enforced} />. Physical stop:{' '}
                <ObservedValue value={operation.physicalStop} />. Paused:{' '}
                <ObservedValue value={operation.remainsPaused} />.
            </div>
            <div>
                Known running service: <ObservedValue value={operation.runningService.buildId} />.
                Known running GUI: <ObservedValue value={operation.runningGui.buildId} />. Settings
                readback: {operation.failure?.readbackConfidence ?? operation.readbackConfidence}.
            </div>
            <div>
                Bytes changed:{' '}
                {operation.failure ? (
                    <ObservedValue value={operation.failure.bytesChanged} />
                ) : (
                    'Unknown'
                )}
                . Settings changed:{' '}
                {operation.failure ? (
                    <ObservedValue value={operation.failure.settingsChanged} />
                ) : (
                    'Unknown'
                )}
                .
            </div>
            {operation.failure && (
                <div role="alert">
                    {operation.failure.reason}: {operation.failure.detail}
                    <br />
                    Safe next action: {operation.failure.safeNextAction}
                </div>
            )}
            {success && (
                <div>
                    Ordinary explicit Resume on this device is required. Saved intent and locks
                    remain preserved.
                </div>
            )}
            {recovery && command && (
                <div>
                    Explicit recovery with the trusted service executable: <code>{command}</code>.
                    Recovery reconciles this operation; it does not automatically roll back. Retain
                    the previous bundle and selected candidate.
                </div>
            )}
            <Button kind="secondary" size="small" disabled={busy} onClick={onRead}>
                Read build status
            </Button>
            <Button
                kind="tertiary"
                size="small"
                onClick={() => {
                    void window.windowApi.window.copyToClipboard(diagnostics)
                }}
            >
                Copy operational diagnostics
            </Button>
        </Stack>
    )
}
