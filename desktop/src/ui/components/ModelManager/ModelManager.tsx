// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useMemo, useState } from 'react'
import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core'
import type { BackendInfo } from '@/ui/types/engine-info'
import { engineCapabilities } from '@/ui/constants/engine-capabilities'
import { useEngineDescriptor } from '@/ui/stores/engine-status.store'
import { formatModelDisplayName } from '@/ui/utils/format-model-display-name'
import { ModelExpiry, type ModelItem } from '@/shared/types/engines'
import { emptyEngineModels, engineManagerName } from '@/shared/utils/engines'

import { ConfirmModal } from '@/ui/components/ConfirmModal'
import { ModelHubModal } from '@/ui/components/ModelHub/ModelHubModal'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { isEnginePullInProgress } from '@/shared/utils/engine-progress'

import ModelRow from './ModelRow'
import { TransientModelStatusRow } from './TransientModelStatusRow'
import type { ModelEntry } from '@/ui/types/model-hub'
import { ModelCopyModal } from './ModelCopyModal'
import { formatCopyProgressLabel } from '@/ui/utils/formatters'
import ModelSettingsPanel from '@/ui/components/ModelSettings/ModelSettingsPanel'
import { useConnectionStore } from '@/ui/stores/connection.store'

export function ModelManager({ backend, nodeId }: { backend: BackendInfo; nodeId: string }) {
    const [openModelHubModal, setOpenModelHubModal] = useState(false)
    const [modelPendingDelete, setModelPendingDelete] = useState<string | null>(null)
    const [configureModel, setConfigureModel] = useState<string | null>(null)
    const selfId = useConnectionStore(state => state.selfId)
    const policyEngine = engineManagerName(backend.type)
    const canConfigure = policyEngine === 'llamacpp' || policyEngine === 'lmstudio'
    const [copySelection, setCopySelection] = useState<{
        source: string
        model: string
        opId?: string
    } | null>(null)
    const nodes = useNodesStore(state => state.nodes)
    const progress = useEngineProgressStore(state => state.progress)
    const copies = Array.from(progress.values()).filter(
        p =>
            p.operation === 'copy' &&
            p.engineType === backend.type &&
            (p.sourceNodeId === nodeId || p.destinationNodeId === nodeId)
    )
    const observedModels = useEngineModelsStore(state =>
        state.models.get(`${nodeId}:${backend.type}`)
    )
    const observation = useMemo(
        () => observedModels ?? emptyEngineModels(nodeId, backend.type),
        [observedModels, nodeId, backend.type]
    )
    const caps = engineCapabilities(backend.type, useEngineDescriptor(backend.type))

    const backendType = backend.type
    const getProgress = useEngineProgressStore(state => state.getProgress)

    /**
     * Track only this engine's pull-progress keys so we re-render only when a
     * pull for THIS engine changes — not every progress tick cluster-wide.
     */
    const pullProgressFingerprint = useEngineProgressStore(state => {
        const parts: string[] = []
        for (const p of state.progress.values()) {
            if (p.nodeId === nodeId && p.engineType === backend.type && p.operation === 'pull') {
                parts.push(JSON.stringify(p))
            }
        }
        return parts.join('|')
    })

    /**
     * Re-render when an optimistic model action for this engine begins/clears;
     * the per-model pending action is read below via getState().
     */
    const modelPendingFingerprint = usePendingActionsStore(state => {
        const prefix = `${nodeId}:${backend.type}:model:`
        const parts: string[] = []
        for (const [key, p] of state.pending) {
            if (key.startsWith(prefix)) parts.push(`${p.model ?? ''}:${p.action}`)
        }
        return parts.join('|')
    })
    void modelPendingFingerprint

    const displayName = useCallback(
        (name: string) => formatModelDisplayName(name, backend.type),
        [backend.type]
    )

    // Canonical models already contain inventory, residency and lifecycle rows.
    // Progress may arrive first, so merge only exact model keys without making
    // an operation event certify content readiness.
    const models = (() => {
        const rows = new Map<string, ModelItem>()
        const supplied =
            observation.models.length > 0 || observation.downloadState
                ? observation.models
                : backend.models
        for (const model of supplied) {
            rows.set(model.name, {
                ...model,
                downloaded:
                    observation.downloadStateCurrent === true &&
                    !!observation.downloadState &&
                    !observation.downloadState.recoveryRequired &&
                    model.downloaded,
                downloadStateCurrent: observation.downloadStateCurrent
            })
        }
        for (const p of useEngineProgressStore.getState().getProgressForNode(nodeId)) {
            if (
                p.engineType !== backend.type ||
                p.operation !== 'pull' ||
                !p.model ||
                !p.downloadState
            )
                continue
            const snapshot = observation.downloadState
            if (
                snapshot &&
                (snapshot.sourceId !== p.downloadSourceId ||
                    (p.downloadRevision && BigInt(p.downloadRevision) < BigInt(snapshot.revision)))
            )
                continue
            const existing = rows.get(p.model)
            rows.set(p.model, {
                name: p.model,
                size: 0,
                status: 'idle',
                parameterSize: '',
                quantization: '',
                family: '',
                digest: '',
                sizeVram: null,
                expiresAt: null,
                expiry: '10m',
                capabilities: [],
                ...existing,
                downloaded:
                    existing?.downloaded === true &&
                    snapshot?.revision === p.downloadRevision &&
                    p.downloadStateCurrent === true &&
                    !p.downloadState.blocksReadiness,
                downloadState: p.downloadState,
                downloadStateCurrent: p.downloadStateCurrent
            })
        }
        return Array.from(rows.values()).sort((a, b) =>
            displayName(a.name).localeCompare(displayName(b.name))
        )
    })()

    const transientModel = models.find(
        m => m.status === 'loading' || m.status === 'ejecting' || m.status === 'pulling'
    )

    // pullProgressFingerprint triggers re-renders when progress changes;
    // the actual data is read from the store below.
    void pullProgressFingerprint

    // Only an engine that restarts to pick a deletion up (LM Studio today)
    // confirms first, because the restart interrupts in-flight inference.
    // Ollama and every other engine delete straight away — this stays false for
    // them, which `tests/modular/delete-model-restart.test.ts` pins against the
    // engine-manager manifests. The restart itself belongs to the engine
    // manager, not to this click.
    const confirmBeforeDelete = caps.restartsOnModelDelete ?? false

    const actionAllowed = useCallback(
        (modelName: string, action: string) => {
            const current = useEngineModelsStore.getState().getModels(nodeId, backendType)
            const model = current.models.find(row => row.name === modelName)
            const progress = useEngineProgressStore
                .getState()
                .getProgress(nodeId, backendType, 'pull', modelName)
            const entry = current.downloadState?.entries.find(
                row => row.engine === engineManagerName(backendType) && row.model === modelName
            )
            if (
                usePendingActionsStore.getState().getModelPending(nodeId, backendType, modelName) ||
                isEnginePullInProgress(progress) ||
                entry?.phase === 'queued' ||
                entry?.phase === 'in_progress'
            )
                return false
            const ready =
                current.downloadStateCurrent === true &&
                !!current.downloadState &&
                !current.downloadState.recoveryRequired &&
                model?.downloaded === true &&
                !entry?.blocksReadiness &&
                !progress.downloadState?.blocksReadiness &&
                (!progress.downloadState ||
                    (progress.downloadStateCurrent === true &&
                        progress.downloadSourceId === current.downloadState.sourceId &&
                        progress.downloadRevision === current.downloadState.revision))
            switch (action) {
                case 'load':
                    return (
                        caps.hasLoad &&
                        ready &&
                        model?.status !== 'loaded' &&
                        backend.processStatus === 'running'
                    )
                case 'copy':
                    return caps.hasModelCopy && ready
                case 'eject':
                    return (
                        caps.hasEject &&
                        model?.status === 'loaded' &&
                        backend.processStatus === 'running'
                    )
                case 'delete':
                    return caps.hasDeleteModel
                default:
                    return false
            }
        },
        [nodeId, backendType, caps, backend.processStatus]
    )

    const handleAction = useCallback(
        (modelName: string, action: string) => {
            if (action === 'configure') {
                if (canConfigure) setConfigureModel(modelName)
                return
            }
            if (!actionAllowed(modelName, action)) return
            switch (action) {
                case 'load':
                    window.pairApi.engines.loadModel(backendType, nodeId, modelName)
                    break
                case 'eject':
                    window.pairApi.engines.unloadModel(backendType, nodeId, modelName)
                    break
                case 'copy':
                    setCopySelection({ source: nodeId, model: modelName })
                    break
                case 'delete':
                    if (confirmBeforeDelete) {
                        setModelPendingDelete(modelName)
                        break
                    }
                    window.pairApi.engines.deleteModel(backendType, nodeId, modelName)
                    break
            }
        },
        [backendType, confirmBeforeDelete, nodeId, actionAllowed, canConfigure]
    )

    // `ConfirmModal` closes (`onOpenChange(false)`) before it calls `onConfirm`,
    // which clears `modelPendingDelete`. This still reads the right model: the
    // callback closes over the value from the render that showed the modal, and
    // React's state update does not mutate that binding.
    const handleConfirmDelete = useCallback(() => {
        if (modelPendingDelete && actionAllowed(modelPendingDelete, 'delete')) {
            window.pairApi.engines.deleteModel(backendType, nodeId, modelPendingDelete)
        }
    }, [backendType, modelPendingDelete, nodeId, actionAllowed])

    const handleExpiryChange = useCallback(
        (modelName: string, value: ModelExpiry) => {
            window.pairApi.engines.setModelExpiry(backendType, nodeId, modelName, value)
        },
        [backendType, nodeId]
    )

    const transientPullProgress = transientModel
        ? getProgress(nodeId, backend.type, 'pull', transientModel.name)
        : undefined

    const hasModelSearchOnlyWhenRunning = caps.hasModelSearchOnlyWhenRunning
    const modelOpsWhenStopped = caps.modelOpsWhenStopped
    const isRunning = useMemo(() => backend.processStatus === 'running', [backend.processStatus])
    // An engine that cannot download a model offers no hub at all.
    const supportsSearch = useMemo(
        () => caps.hasPull && (!hasModelSearchOnlyWhenRunning || isRunning),
        [caps.hasPull, hasModelSearchOnlyWhenRunning, isRunning]
    )

    const handleDownload = useCallback(
        (entries: ModelEntry[]) => {
            if (!entries || entries.length === 0) return

            setOpenModelHubModal(false)

            for (const entry of entries) {
                if (entry.name) {
                    window.pairApi.engines.pullModel(backendType, nodeId, entry.name)
                }
            }
        },
        [backendType, nodeId]
    )

    return (
        <Stack gap="4">
            {transientModel && (
                <TransientModelStatusRow
                    transientModel={transientModel}
                    displayName={displayName}
                    pullProgress={transientPullProgress}
                />
            )}

            {models.length === 0 && observation.downloadStateCurrent !== true && (
                <Text kind="body/regular/sm" className="text-subtle-color pl-2">
                    Download status unknown · Disconnected or download status unavailable
                </Text>
            )}
            {(isRunning || modelOpsWhenStopped) &&
                models.length === 0 &&
                observation.downloadStateCurrent === true && (
                    <Text kind="body/regular/sm" className="text-subtle-color pl-2">
                        No model/weight files downloaded
                    </Text>
                )}
            {!modelOpsWhenStopped && models.length === 0 && backend.processStatus === 'stopped' && (
                <Text kind="body/regular/sm" className="text-subtle-color pl-2">
                    Start engine to see models
                </Text>
            )}

            {models.length > 0 && (
                <Stack gap="1">
                    {models
                        .filter(model => model !== transientModel)
                        .map(model => (
                            <ModelRow
                                key={model.name}
                                model={model}
                                isRunning={isRunning}
                                capabilities={caps}
                                progress={getProgress(nodeId, backend.type, 'pull', model.name)}
                                pendingAction={usePendingActionsStore
                                    .getState()
                                    .getModelPending(nodeId, backendType, model.name)}
                                canCopy={caps.hasModelCopy}
                                canConfigure={canConfigure}
                                displayName={displayName}
                                onAction={handleAction}
                                onExpiryChange={handleExpiryChange}
                            />
                        ))}
                </Stack>
            )}

            {copies.map(copy => (
                <Stack key={copy.opId} gap="1">
                    <Text kind="body/regular/sm">
                        {copy.model} ·{' '}
                        {nodes.get(copy.sourceNodeId ?? '')?.name ?? copy.sourceNodeId}
                        {' → '}
                        {nodes.get(copy.destinationNodeId ?? '')?.name ?? copy.destinationNodeId}
                        {' · '}
                        {formatCopyProgressLabel(copy)}
                    </Text>
                    {copy.message && <Text kind="body/regular/sm">{copy.message}</Text>}
                    <Button
                        kind="tertiary"
                        size="small"
                        onClick={() =>
                            setCopySelection({
                                source: copy.sourceNodeId ?? '',
                                model: copy.model ?? '',
                                opId: copy.opId
                            })
                        }
                    >
                        View copy
                    </Button>
                </Stack>
            ))}
            <ModelCopyModal
                key={`${backendType}:${copySelection?.source ?? ''}:${copySelection?.model ?? ''}:${copySelection?.opId ?? ''}`}
                open={copySelection !== null}
                onOpenChange={open => {
                    if (!open) setCopySelection(null)
                }}
                engineType={backendType}
                initialSource={copySelection?.source ?? nodeId}
                initialModel={copySelection?.model ?? ''}
                initialOpId={copySelection?.opId}
            />

            {configureModel !== null && (
                <ModelSettingsPanel
                    key={`${nodeId}:${backendType}:${configureModel}`}
                    targetId={nodeId}
                    nodeId={nodeId === selfId ? undefined : nodeId}
                    nodeName={nodes.get(nodeId)?.name ?? (nodeId === selfId ? 'This PC' : nodeId)}
                    identity={{ engine: engineManagerName(backendType), model: configureModel }}
                    open
                    onOpenChange={open => {
                        if (!open) setConfigureModel(null)
                    }}
                />
            )}

            <ConfirmModal
                open={modelPendingDelete !== null}
                onOpenChange={open => {
                    if (!open) setModelPendingDelete(null)
                }}
                title="Delete model?"
                message={`Deleting ${displayName(modelPendingDelete ?? '')} requires ${backend.displayName} to be restarted.`}
                confirmLabel="Delete and restart"
                confirmColor="danger"
                onConfirm={handleConfirmDelete}
            />

            {supportsSearch && (
                <Flex justify="end">
                    <Button
                        size="small"
                        kind="primary"
                        color="brand"
                        onClick={() => setOpenModelHubModal(true)}
                    >
                        Add model
                    </Button>
                    <ModelHubModal
                        engine={openModelHubModal ? backend.type : null}
                        onOpenChange={setOpenModelHubModal}
                        onSubmit={handleDownload}
                        downloadedModels={models}
                    />
                </Flex>
            )}
        </Stack>
    )
}
