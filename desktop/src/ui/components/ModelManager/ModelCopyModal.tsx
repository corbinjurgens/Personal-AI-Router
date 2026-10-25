// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from 'react'
import {
    Button,
    Flex,
    FormField,
    ModalContent,
    ModalDialog,
    ModalHeading,
    ModalRoot,
    ProgressBar,
    Select,
    Stack,
    Text
} from '@nvidia/foundations-react-core'
import type { EngineModels, EngineProgress, EngineType } from '@/shared/types/engines'
import type { ModelCopyCapabilities, ModelCopyRequest } from '@/shared/types/model-copy'
import { engineManagerName } from '@/shared/utils/engines'
import { isEngineCopyInProgress } from '@/shared/utils/engine-progress'
import { ServiceCallError } from '@/ui/api/service-transport'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { formatBytes, formatCopyProgressLabel } from '@/ui/utils/formatters'
import {
    LOCAL_VETO_CODE,
    POLICY_CONFLICT_CODE,
    REMOTE_MANAGEMENT_DENIED_CODE
} from '@/shared/constants/service-errors'

function failure(error: Error): string {
    if (error instanceof ServiceCallError) {
        if (error.code === -32601)
            return 'Unsupported: this device needs a PAIR version with model copy support.'
        if (error.code === REMOTE_MANAGEMENT_DENIED_CODE)
            return `Permission denied: ${error.message}`
        if (error.code === LOCAL_VETO_CODE)
            return `Device unavailable for changes: ${error.message}`
        if (error.code === POLICY_CONFLICT_CODE)
            return `Conflict: ${error.message} Existing content will not be overwritten.`
    }
    return `Copy availability unknown: ${error.message}`
}

function capabilityReason(
    cap: ModelCopyCapabilities,
    engine: string,
    destination: boolean
): string {
    if (cap.version !== 1 || cap.engine !== engine)
        return 'Unsupported model copy capability version or engine.'
    if (!cap.available) return cap.reason || 'This engine is unsupported on this device.'
    if (!cap.installed) return 'Install this engine on this device before copying.'
    if (!cap.modelStore) return 'This engine has no supported model store adapter.'
    if (destination && !cap.writable)
        return cap.reason || 'Destination write permission is unavailable.'
    if ((destination ? cap.importLayouts : cap.exportLayouts).length === 0)
        return 'This device has no supported copy layout.'
    return ''
}

function isCopySourceReady(
    inventory: EngineModels | undefined,
    model: string,
    progress: Iterable<EngineProgress>
): boolean {
    const snapshot = inventory?.downloadState
    if (!inventory?.downloadStateCurrent || !snapshot || snapshot.recoveryRequired) return false
    const row = inventory.models.find(row => row.name === model)
    const entry = snapshot.entries.find(
        entry => entry.engine === engineManagerName(inventory.engineType) && entry.model === model
    )
    if (
        !row?.downloaded ||
        row.downloadState?.blocksReadiness ||
        row.downloadState?.phase === 'queued' ||
        row.downloadState?.phase === 'in_progress' ||
        entry?.blocksReadiness ||
        entry?.phase === 'queued' ||
        entry?.phase === 'in_progress'
    )
        return false
    for (const p of progress) {
        if (
            p.nodeId !== inventory.nodeId ||
            p.engineType !== inventory.engineType ||
            p.model !== model ||
            p.operation !== 'pull' ||
            !p.downloadState
        )
            continue
        if (p.downloadSourceId !== snapshot.sourceId || !p.downloadRevision) return false
        if (BigInt(p.downloadRevision) < BigInt(snapshot.revision)) continue
        if (
            p.downloadRevision !== snapshot.revision ||
            p.downloadStateCurrent !== true ||
            p.downloadState.blocksReadiness ||
            p.downloadState.phase === 'queued' ||
            p.downloadState.phase === 'in_progress'
        )
            return false
    }
    return true
}

/** A draft selection only; operation state and terminal qualifications live in the progress store. */
export function ModelCopyModal({
    open,
    onOpenChange,
    engineType,
    initialSource,
    initialModel,
    initialOpId
}: {
    open: boolean
    onOpenChange: (open: boolean) => void
    engineType: EngineType
    initialSource: string
    initialModel: string
    initialOpId?: string
}) {
    const engine = engineManagerName(engineType)
    const connected = useConnectionStore(state => state.connected)
    const selfId = useConnectionStore(state => state.selfId)
    const nodes = useNodesStore(state => state.nodes)
    const progress = useEngineProgressStore(state => state.progress)
    const models = useEngineModelsStore(state => state.models)
    const initialCopy = Array.from(progress.values()).find(
        p => p.opId === initialOpId && p.operation === 'copy'
    )
    const [source, setSource] = useState(initialCopy?.sourceNodeId ?? initialSource)
    const [model, setModel] = useState(initialCopy?.model ?? initialModel)
    const [destination, setDestination] = useState(initialCopy?.destinationNodeId ?? '')
    const [opId, setOpId] = useState(initialOpId)
    const [inventory, setInventory] = useState<{
        source: string
        value?: EngineModels
        error?: string
    }>()
    const [capabilities, setCapabilities] = useState<{
        key: string
        source?: ModelCopyCapabilities
        destination?: ModelCopyCapabilities
        reason: string
    }>()
    const [error, setError] = useState('')
    const [checking, setChecking] = useState(false)
    const [cancelling, setCancelling] = useState(false)
    // Synchronous guard also fences retained callbacks before React commits disabled controls.
    const submitting = useRef(false)
    const selectionGeneration = useRef(0)
    const copy = Array.from(progress.values()).find(
        p =>
            p.operation === 'copy' &&
            p.opId === opId &&
            p.sourceNodeId === source &&
            p.destinationNodeId === destination &&
            p.engineType === engineType &&
            p.model === model
    )
    const active = isEngineCopyInProgress(copy)
    const locked = active || checking
    const retry = !!copy && ['error', 'cancelled', 'unknown'].includes(copy.status)
    const expectedExportId = retry ? (copy.exportId ?? copy.expectedExportId) : undefined
    const key = JSON.stringify([source, destination, engine, model, expectedExportId])
    const name = (id: string) => nodes.get(id)?.name ?? id
    const online = (id: string) => !!id && (id === selfId || nodes.get(id)?.status === 'active')
    const sourceOnline = online(source)
    const destinationOnline = online(destination)
    const invalidateSelection = useCallback(() => {
        selectionGeneration.current++
    }, [])

    useEffect(() => {
        if (!open || !connected) return
        let current = true
        setInventory({ source })
        void window.pairApi.engines
            .getInitialState()
            .then(snapshot => {
                if (!current) return
                setInventory({
                    source,
                    value: snapshot.models.find(
                        row => row.nodeId === source && row.engineType === engineType
                    )
                })
            })
            .catch(err => {
                if (current)
                    setInventory({
                        source,
                        error: err instanceof Error ? err.message : 'Inventory unavailable'
                    })
            })
        return () => {
            current = false
        }
    }, [open, connected, source, engineType])

    // Store pushes supersede the selection's read without fetching any artifact/catalog data.
    const observed = models.get(`${source}:${engineType}`)
    const sourceInventory = observed ?? (inventory?.source === source ? inventory.value : undefined)
    const readyModels =
        sourceInventory?.models.filter(row =>
            isCopySourceReady(sourceInventory, row.name, progress.values())
        ) ?? []

    const readCapabilities = useCallback(async (): Promise<{
        source?: ModelCopyCapabilities
        destination?: ModelCopyCapabilities
        reason: string
    }> => {
        if (!connected) return { reason: 'Disconnected. Reconnect to check copy availability.' }
        if (!source || !destination)
            return { reason: 'Choose a source and an independent destination device.' }
        if (source === destination)
            return { reason: 'Source and destination must be different devices.' }
        if (!destinationOnline) return { reason: 'Destination device is offline.' }
        if (!sourceOnline && !expectedExportId) return { reason: 'Source device is offline.' }
        // A constrained retry may verify an already committed receipt with its source offline.
        // The destination still rechecks caller authority before it examines that receipt.
        const [from, to] = await Promise.allSettled([
            window.pairApi.engines.modelCopyCapabilities({ nodeId: source, engine }),
            window.pairApi.engines.modelCopyCapabilities({ nodeId: destination, engine })
        ])
        if (to.status === 'rejected')
            return {
                reason:
                    to.reason instanceof Error
                        ? failure(to.reason)
                        : 'Destination capability unknown.'
            }
        const target = to.value
        const targetReason = capabilityReason(target, engine, true)
        if (targetReason) return { destination: target, reason: targetReason }
        if (expectedExportId && (!target.identityBoundRetry || !target.receiptDurability))
            return {
                destination: target,
                reason: 'Destination does not support identity-bound receipt verification. Retry cannot safely confirm this copy.'
            }
        if (from.status === 'rejected') {
            if (expectedExportId && !sourceOnline) return { destination: target, reason: '' }
            return {
                destination: target,
                reason:
                    from.reason instanceof Error
                        ? failure(from.reason)
                        : 'Source capability unknown.'
            }
        }
        const origin = from.value
        const reason =
            capabilityReason(origin, engine, false) ||
            (!origin.exportLayouts.some(layout => target.importLayouts.includes(layout))
                ? 'Source and destination have no compatible copy layout.'
                : '')
        return { source: origin, destination: target, reason }
    }, [connected, source, destination, engine, expectedExportId, sourceOnline, destinationOnline])

    useEffect(() => {
        if (!open) return
        let current = true
        const generation = ++selectionGeneration.current
        setCapabilities(undefined)
        void readCapabilities().then(value => {
            if (current && generation === selectionGeneration.current)
                setCapabilities({ key, ...value })
        })
        return () => {
            current = false
            invalidateSelection()
        }
        // Device changes and reconnects invalidate capabilities; grants are checked again on submit.
    }, [open, key, readCapabilities, invalidateSelection])

    const reason = !connected
        ? 'Disconnected. Reconnect to check copy availability.'
        : capabilities?.key === key
          ? capabilities.reason
          : 'Checking device copy capabilities…'
    const inventoryReady = readyModels.some(row => row.name === model)
    const canSubmit =
        !!source &&
        !!destination &&
        !!model &&
        !reason &&
        (inventoryReady || !!expectedExportId) &&
        !locked &&
        (!copy || retry)

    async function submit() {
        if (!canSubmit || submitting.current) return
        submitting.current = true
        setChecking(true)
        setError('')
        const generation = selectionGeneration.current
        let submittedOpId: string | undefined
        try {
            const fresh = await readCapabilities()
            if (generation !== selectionGeneration.current) return
            setCapabilities({ key, ...fresh })
            if (fresh.reason) {
                setError(fresh.reason)
                return
            }
            if (!expectedExportId) {
                const snapshot = await window.pairApi.engines.getInitialState()
                if (generation !== selectionGeneration.current) return
                const current = snapshot.models.find(
                    row => row.nodeId === source && row.engineType === engineType
                )
                // A read can resolve after a newer canonical push or lifecycle
                // observation. Recheck live stores synchronously before dispatch.
                const canonical = useEngineModelsStore
                    .getState()
                    .models.get(`${source}:${engineType}`)
                if (
                    !isCopySourceReady(current, model, snapshot.activeProgress) ||
                    !isCopySourceReady(
                        canonical ?? current,
                        model,
                        useEngineProgressStore.getState().progress.values()
                    ) ||
                    (canonical?.downloadState &&
                        (current?.downloadState?.sourceId !== canonical.downloadState.sourceId ||
                            BigInt(current.downloadState.revision) <
                                BigInt(canonical.downloadState.revision)))
                ) {
                    setError(
                        'This model is no longer ready in the source inventory. Refresh the selection.'
                    )
                    return
                }
            }
            const request: ModelCopyRequest = {
                opId: crypto.randomUUID().replaceAll('-', ''),
                sourceNodeId: source,
                destinationNodeId: destination,
                engine,
                model,
                ...(expectedExportId ? { expectedExportId } : {})
            }
            submittedOpId = request.opId
            setOpId(request.opId)
            // The bridge publishes progress and retains verified completion, even after this modal closes.
            await window.pairApi.engines.copyModel(request)
        } catch (err) {
            // A refusal before bridge admission has no new progress entry. Keep
            // the previous retry selection, including its learned export ID.
            if (
                submittedOpId &&
                !Array.from(useEngineProgressStore.getState().progress.values()).some(
                    p => p.opId === submittedOpId
                )
            )
                setOpId(opId)
            setError(
                err instanceof Error
                    ? failure(err)
                    : 'Copy outcome unknown. Reconnect before retrying.'
            )
        } finally {
            submitting.current = false
            setChecking(false)
        }
    }

    async function cancel() {
        if (!copy?.opId || !active || cancelling) return
        setCancelling(true)
        setError('')
        try {
            await window.pairApi.engines.cancelModelCopy(copy.opId)
        } catch (err) {
            setError(err instanceof Error ? failure(err) : 'Cancellation could not be confirmed.')
        } finally {
            setCancelling(false)
        }
    }

    const deviceItems = Array.from(nodes.values()).map(node => ({
        value: node.id,
        children: `${node.name}${node.id === selfId ? ' (this device)' : ''}${node.status === 'offline' ? ' — offline' : ''}`
    }))
    if (selfId && !nodes.has(selfId))
        deviceItems.unshift({ value: selfId, children: 'This device' })
    const change = () => {
        invalidateSelection()
        setOpId(undefined)
        setError('')
    }

    return (
        <ModalRoot open={open} onOpenChange={onOpenChange}>
            <ModalDialog>
                <ModalContent className="no-drag-elements max-content-modal">
                    <ModalHeading>Copy model</ModalHeading>
                    <Stack gap="4">
                        <Text kind="body/regular/sm">Engine: {engineType}</Text>
                        <FormField slotLabel="Source device" required>
                            <Select
                                aria-label="Source device"
                                value={source}
                                disabled={locked}
                                items={deviceItems.map(item => ({
                                    ...item,
                                    disabled: item.value === destination
                                }))}
                                onValueChange={(value: string) => {
                                    if (submitting.current || active) return
                                    change()
                                    setSource(value)
                                    setModel('')
                                }}
                            />
                        </FormField>
                        <FormField
                            slotLabel="Model"
                            required
                            slotHelp={
                                inventory?.source === source && inventory.error
                                    ? inventory.error
                                    : !sourceInventory?.downloadStateCurrent
                                      ? 'Source inventory unknown or unavailable.'
                                      : undefined
                            }
                        >
                            <Select
                                aria-label="Model"
                                value={model}
                                disabled={locked || !online(source)}
                                placeholder="Choose an installed model"
                                items={readyModels.map(row => ({
                                    value: row.name,
                                    children: row.name
                                }))}
                                onValueChange={(value: string) => {
                                    if (submitting.current || active) return
                                    change()
                                    setModel(value)
                                }}
                            />
                        </FormField>
                        <FormField slotLabel="Destination device" required>
                            <Select
                                aria-label="Destination device"
                                value={destination}
                                disabled={locked}
                                placeholder="Choose a destination device"
                                items={deviceItems.map(item => ({
                                    ...item,
                                    disabled: item.value === source
                                }))}
                                onValueChange={(value: string) => {
                                    if (submitting.current || active) return
                                    change()
                                    setDestination(value)
                                }}
                            />
                        </FormField>
                        {source && destination && model && (
                            <Text kind="body/regular/sm">
                                Copy {model} from {name(source)} to {name(destination)}. The
                                original stays on {name(source)}.
                                {source !== selfId && destination !== selfId
                                    ? ' Transfers directly between these devices.'
                                    : ''}
                            </Text>
                        )}
                        {reason && !active && (!copy || retry) && (
                            <Text kind="body/regular/sm" role="status">
                                {reason}
                            </Text>
                        )}
                        {copy && (
                            <Stack gap="2" aria-live="polite">
                                <Text kind="body/regular/sm">
                                    {model} · {name(source)} → {name(destination)} ·{' '}
                                    {formatCopyProgressLabel(copy)}
                                </Text>
                                {active &&
                                    (copy.percent === undefined ? (
                                        <ProgressBar
                                            aria-label="Model copy progress"
                                            kind="indeterminate"
                                        />
                                    ) : (
                                        <ProgressBar
                                            aria-label="Model copy progress"
                                            value={copy.percent}
                                        />
                                    ))}
                                {copy.completed !== undefined && (
                                    <Text kind="body/regular/sm">
                                        {formatBytes(copy.completed)}
                                        {copy.total !== undefined
                                            ? ` / ${formatBytes(copy.total)}`
                                            : ''}
                                    </Text>
                                )}
                                {copy.message && <Text kind="body/regular/sm">{copy.message}</Text>}
                            </Stack>
                        )}
                        {retry && (
                            <Text kind="body/regular/sm">
                                {expectedExportId
                                    ? 'Retry verifies the same export identity. A verified destination receipt can confirm completion with the source offline; otherwise the source must still provide that exact export.'
                                    : 'Outcome may be unknown. Retry resolves this name again; identical content cannot be guaranteed without an export identity.'}
                            </Text>
                        )}
                        {error && (
                            <Text kind="body/regular/sm" role="alert">
                                {error}
                            </Text>
                        )}
                        <Flex justify="end" gap="2">
                            <Button kind="secondary" onClick={() => onOpenChange(false)}>
                                Close
                            </Button>
                            {active && (
                                <Button
                                    kind="secondary"
                                    disabled={cancelling}
                                    onClick={() => void cancel()}
                                >
                                    Cancel copy
                                </Button>
                            )}
                            <Button
                                kind="primary"
                                color="brand"
                                disabled={!canSubmit}
                                onClick={() => void submit()}
                            >
                                {retry ? 'Retry copy' : 'Copy model'}
                            </Button>
                        </Flex>
                    </Stack>
                </ModalContent>
            </ModalDialog>
        </ModalRoot>
    )
}
