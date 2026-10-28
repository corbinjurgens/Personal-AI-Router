// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { create } from 'zustand'
import type {
    EngineSettingsRequest,
    EngineSettingsPreview,
    EngineSettingsSnapshot,
    EngineSettingsTarget
} from '@/shared/types/engine-settings'
import type { InspectModelResult, ModelChange } from '@/shared/types/model-config'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { nodePaused, useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import getErrorString from '@/shared/utils/get-error-string'
import { modelPolicyPatch } from '@/shared/utils/node-policy-model-copy'
import { serviceErrorCode } from '@/ui/api/service-transport'
import {
    INVALID_PARAMS_CODE,
    LOCAL_VETO_CODE,
    POLICY_CONFLICT_CODE,
    REMOTE_MANAGEMENT_DENIED_CODE,
    REMOTE_SETTING_LOCKED_CODE
} from '@/shared/constants/service-errors'

export const engineSettingsKey = (target: EngineSettingsTarget): string =>
    `${target.nodeId}:${target.engine}`

interface SettingsEntry {
    snapshot?: EngineSettingsSnapshot
    unavailable?: string
    arrival: number
}
interface SettingsStore {
    entries: Record<string, SettingsEntry>
    receive: (snapshot: EngineSettingsSnapshot) => void
    disconnect: (nodeId?: string) => void
    reload: (target: EngineSettingsTarget) => Promise<void>
    apply: (request: EngineSettingsRequest) => Promise<void>
    modelDrafts: Record<string, ModelDraft>
    editModel: (key: string, change: ModelChange) => void
    discardModel: (key: string) => void
    invalidateModel: (key: string) => void
    inspectModel: (target: EngineSettingsTarget, modelId: string) => Promise<void>
    previewModel: (
        target: EngineSettingsTarget,
        change: ModelChange,
        authority: string,
        advisoryOnly?: boolean
    ) => Promise<void>
    applyModel: (key: string, authority: string) => Promise<void>
    readModelOutcome: (key: string) => Promise<void>
}

interface ModelDraft {
    change?: ModelChange
    inspection?: InspectModelResult
    advisory?: EngineSettingsPreview
    review?: {
        request: EngineSettingsRequest
        preview: EngineSettingsPreview
        baseline: string
        authority: string
        authorityState: string
        inspection: string
    }
    busy?: boolean
    previewOperation?: string
    message?: string
    attempt?: EngineSettingsRequest
    submittedPolicy?: {
        owner: 'Routing' | 'Model Manager'
        text: string
        revision: string
    }
    uncertain?: boolean
    generation: number
}
interface ModelPreviewOwner {
    key: string
    generation: number
    advisoryOnly: boolean
    authority: string
    authorityState: string
    pending?: {
        key: string
        generation: number
        advisoryOnly: boolean
        authority: string
        authorityState: string
        run: () => Promise<void>
        finish: () => void
    }
}
// Transport calls cannot be aborted. Keep their ownership until they settle,
// even when a draft is logically cancelled, and coalesce its next request.
let modelPreview: ModelPreviewOwner | undefined
export const modelSettingsKey = (target: EngineSettingsTarget, modelId: string): string =>
    `${engineSettingsKey(target)}:${modelId}`

function invalidateDraft(draft?: ModelDraft): ModelDraft {
    return {
        ...draft,
        review: undefined,
        advisory: undefined,
        generation: (draft?.generation ?? 0) + 1,
        ...(draft?.previewOperation ? { busy: false, previewOperation: undefined } : {})
    }
}

function baselineIdentity(snapshot: EngineSettingsSnapshot | undefined): string {
    if (!snapshot) return ''
    return JSON.stringify([
        snapshot.epoch,
        snapshot.revision,
        snapshot.policyRevision,
        snapshot.editable,
        snapshot.adopted,
        snapshot.settings,
        snapshot.effectiveSettings,
        snapshot.phase,
        snapshot.requestId,
        snapshot.coupledStage
    ])
}

function inspectionIdentity(inspection: InspectModelResult): string {
    // Native build/preset/input identities are also bound by baselineIdentity's
    // target-owned resolved loads. Advisory estimates and display metadata are
    // deliberately absent: their refresh does not establish changed artifacts.
    return JSON.stringify([
        inspection.engine,
        inspection.nodeId,
        inspection.modelId,
        inspection.source,
        inspection.assurance,
        inspection.artifacts?.map(artifact => [
            artifact.role,
            artifact.identity,
            artifact.size,
            artifact.assurance
        ])
    ])
}

function correlateModelOutcome(draft: ModelDraft, snapshot: EngineSettingsSnapshot): ModelDraft {
    const attempt = draft.attempt
    if (
        !attempt?.requestId ||
        attempt.requestId !== snapshot.requestId ||
        attempt.nodeId !== snapshot.nodeId ||
        attempt.engine !== snapshot.engine
    )
        return draft
    const attemptedChange = { ...attempt.modelChange }
    delete attemptedChange.previewDigest
    const completed =
        snapshot.phase === 'succeeded' &&
        (!draft.change || JSON.stringify(draft.change) === JSON.stringify(attemptedChange))
    return {
        ...draft,
        uncertain: snapshot.phase === 'applying',
        ...(completed ? { change: undefined, review: undefined } : {}),
        message: `Operation ${attempt.requestId}: ${snapshot.phase}. ${snapshot.unresolvedError || snapshot.error || ''} ${completed ? 'Completed edits cleared.' : 'Unsubmitted draft kept.'} Inspect and review before another model operation.`
    }
}

function modelAuthority(target: EngineSettingsTarget): string {
    const connection = useConnectionStore.getState()
    const store = useNodeConfigurationStore.getState()
    const permission = store.management.get(target.nodeId)
    const policy = store.policyDrafts.get(target.nodeId)
    return JSON.stringify([
        connection.connected,
        connection.selfId,
        useClusterInvitationsStore.getState().remoteManagementContext,
        permission?.state,
        permission?.revision,
        policy?.current,
        policy?.saved?.revision,
        policy?.saved?.availability,
        policy?.availabilityObservation
    ])
}

async function authorizeModel(
    request: EngineSettingsRequest,
    current: () => boolean
): Promise<void> {
    if (!useConnectionStore.getState().connected) throw new Error('Device disconnected.')
    const local = request.nodeId === useConnectionStore.getState().selfId
    const policy = await window.pairApi.nodePolicy.get(local ? undefined : request.nodeId)
    if (!current()) throw new Error('Preview expired. Your draft is kept.')
    if (policy.revision !== request.modelChange?.expectedPolicyRevision)
        throw new Error('Policy changed. Preview again; your draft is kept.')
    if (!local) {
        if (useClusterInvitationsStore.getState().remoteManagementContext === null)
            throw new Error('Destination authority is unavailable.')
        const permission = useNodeConfigurationStore.getState().management.get(request.nodeId)
        const fresh = await window.pairApi.cluster.getRemoteManagementForNode(request.nodeId)
        if (!current()) throw new Error('Preview expired. Your draft is kept.')
        if (
            permission?.state !== 'allowed' ||
            fresh.state !== 'allowed' ||
            !fresh.allowed ||
            fresh.revision !== permission.revision
        )
            throw new Error('Current destination permission is required.')
        if (nodePaused(policy.policy, policy.availability))
            throw new Error('Destination is paused.')
    }
}

export const useEngineSettingsStore = create<SettingsStore>((set, get) => ({
    entries: {},
    modelDrafts: {},
    editModel: (key, change) =>
        set(state => ({
            modelDrafts: {
                ...state.modelDrafts,
                [key]: {
                    ...invalidateDraft(state.modelDrafts[key]),
                    change: structuredClone(change)
                }
            }
        })),
    discardModel: key =>
        set(state => ({
            modelDrafts: {
                ...state.modelDrafts,
                [key]: {
                    ...invalidateDraft(state.modelDrafts[key]),
                    change: undefined
                }
            }
        })),
    invalidateModel: key =>
        set(state => ({
            modelDrafts: {
                ...state.modelDrafts,
                [key]: invalidateDraft(state.modelDrafts[key])
            }
        })),
    inspectModel: async (target, modelId) => {
        const key = modelSettingsKey(target, modelId)
        get().invalidateModel(key)
        const generation = get().modelDrafts[key].generation
        try {
            const inspection = await window.pairApi.engines.inspectModel({ ...target, modelId })
            if (
                inspection.engine !== target.engine ||
                inspection.modelId !== modelId ||
                (inspection.nodeId && inspection.nodeId !== target.nodeId)
            )
                throw new Error('Inspection target mismatch.')
            if (get().modelDrafts[key]?.generation !== generation) return
            set(state => ({
                modelDrafts: {
                    ...state.modelDrafts,
                    [key]: {
                        ...state.modelDrafts[key],
                        inspection,
                        message: inspection.reason
                    }
                }
            }))
        } catch (error) {
            if (get().modelDrafts[key]?.generation !== generation) return
            set(state => ({
                modelDrafts: {
                    ...state.modelDrafts,
                    [key]: {
                        ...state.modelDrafts[key],
                        inspection: undefined,
                        message: getErrorString(error)
                    }
                }
            }))
        }
    },
    previewModel: async (target, change, authority, advisoryOnly = false) => {
        // The wire contract rejects an empty selected policy patch, even when
        // the same operation includes a valid load edit.
        change = structuredClone(change)
        if (change.policy) {
            if (!Object.keys(change.policy.generation ?? {}).length) delete change.policy.generation
            if (!Object.keys(change.policy.roles ?? {}).length) delete change.policy.roles
            if (!Object.values(change.policy).some(value => value !== undefined))
                delete change.policy
        }
        const key = modelSettingsKey(target, change.modelId)
        const draft = get().modelDrafts[key]
        const owner = modelPreview
        const authorityState = modelAuthority(target)
        const explicit = owner?.pending ?? owner
        // Background estimates must not displace a user-requested review,
        // including a destination review while the source editor stays mounted.
        if (
            advisoryOnly &&
            [owner, owner?.pending].some(
                request =>
                    request &&
                    !request.advisoryOnly &&
                    get().modelDrafts[request.key]?.generation === request.generation
            )
        )
            return
        if (
            advisoryOnly &&
            JSON.stringify(draft?.change) === JSON.stringify(change) &&
            ((draft?.review?.authority === authority &&
                draft.review.authorityState === authorityState) ||
                (explicit &&
                    explicit.key === key &&
                    !explicit.advisoryOnly &&
                    explicit.generation === draft?.generation &&
                    explicit.authority === authority &&
                    explicit.authorityState === authorityState))
        )
            return
        if ((draft?.busy && !draft.previewOperation) || draft?.uncertain) return
        get().editModel(key, change)
        const generation = get().modelDrafts[key].generation
        if (owner) {
            if (owner.key !== key) get().invalidateModel(owner.key)
            if (owner.pending && owner.pending.key !== key) get().invalidateModel(owner.pending.key)
            owner.pending?.finish()
            return new Promise<void>(finish => {
                owner.pending = {
                    key,
                    generation,
                    advisoryOnly,
                    authority,
                    authorityState,
                    finish,
                    run: async () => {
                        if (modelAuthority(target) !== authorityState) {
                            get().invalidateModel(key)
                            return
                        }
                        await get().previewModel(target, change, authority, advisoryOnly)
                    }
                }
                set(state => ({
                    modelDrafts: {
                        ...state.modelDrafts,
                        [key]: {
                            ...state.modelDrafts[key],
                            busy: true,
                            previewOperation: crypto.randomUUID()
                        }
                    }
                }))
            })
        }
        const active: ModelPreviewOwner = {
            key,
            generation,
            advisoryOnly,
            authority,
            authorityState
        }
        modelPreview = active
        const operation = crypto.randomUUID()
        const current = () =>
            get().modelDrafts[key]?.generation === generation &&
            get().modelDrafts[key]?.previewOperation === operation &&
            modelAuthority(target) === authorityState
        const put = (value: Partial<ModelDraft>) => {
            if (
                get().modelDrafts[key]?.generation === generation &&
                get().modelDrafts[key]?.previewOperation === operation
            )
                set(state => ({
                    modelDrafts: {
                        ...state.modelDrafts,
                        [key]: { ...state.modelDrafts[key], ...value }
                    }
                }))
        }
        set(state => ({
            modelDrafts: {
                ...state.modelDrafts,
                [key]: {
                    ...state.modelDrafts[key],
                    previewOperation: operation,
                    busy: true,
                    message: undefined
                }
            }
        }))
        try {
            await get().reload(target)
            if (!current()) return
            if (JSON.stringify(get().modelDrafts[key]?.change) !== JSON.stringify(change))
                throw new Error('Draft changed during preview. Review again.')
            if (get().modelDrafts[key]?.previewOperation !== operation) return
            const entry = get().entries[engineSettingsKey(target)]
            if (!entry?.snapshot || entry.unavailable || !entry.snapshot.editable)
                throw new Error(
                    entry?.unavailable ?? entry?.snapshot?.reason ?? 'Settings unavailable.'
                )
            const snapshot = entry.snapshot
            const request: EngineSettingsRequest = {
                ...target,
                expectedRevision: snapshot.revision,
                settings: structuredClone(snapshot.settings),
                modelChange: structuredClone(change)
            }
            await authorizeModel(request, current)
            if (!current()) return
            const inspection = await window.pairApi.engines.inspectModel({
                ...target,
                modelId: change.modelId
            })
            if (!current()) return
            if (
                inspection.engine !== target.engine ||
                inspection.modelId !== change.modelId ||
                (inspection.nodeId && inspection.nodeId !== target.nodeId)
            )
                throw new Error('Inspection target mismatch.')
            const preview = await window.pairApi.engines.previewSettings(request)
            if (!current()) return
            if (
                modelAuthority(target) !== authorityState ||
                baselineIdentity(get().entries[engineSettingsKey(target)]?.snapshot) !==
                    baselineIdentity(snapshot)
            )
                throw new Error('Target changed during preview. Review again.')
            if (
                preview.revision !== snapshot.revision ||
                (preview.policyRevision !== undefined &&
                    preview.policyRevision !== change.expectedPolicyRevision)
            )
                throw new Error('Preview revisions do not match the target. Review again.')
            const saved = snapshot.settings.modelLoads?.find(
                load => load.modelId === change.modelId
            )?.requested
            const context = change.load?.clear?.includes('context')
                ? undefined
                : (change.load?.set.context ?? saved?.context)
            const gpu = change.load?.clear?.includes('gpu')
                ? undefined
                : (change.load?.set.gpu ?? saved?.gpu)
            const automatic = !!change.load && (context?.mode === 'auto' || gpu?.mode === 'auto')
            if (
                advisoryOnly ||
                Object.keys(preview.errors).length ||
                !preview.previewDigest ||
                preview.policyRevision !== change.expectedPolicyRevision ||
                (automatic &&
                    (!preview.auto?.resolved ||
                        preview.auto.status === 'unavailable' ||
                        preview.auto.estimate.status !== 'estimated' ||
                        !!preview.auto.estimate.unknowns?.length ||
                        !preview.auto.reviewedInputId ||
                        !preview.auto.resolverRevision?.startsWith('auto/')))
            ) {
                put({
                    inspection,
                    review: undefined,
                    advisory: preview,
                    message:
                        Object.values(preview.errors).join('; ') ||
                        (advisoryOnly
                            ? undefined
                            : 'A reviewed Auto resolution, digest and current revisions are required.')
                })
                return
            }
            put({
                inspection,
                review: {
                    request: {
                        ...request,
                        modelChange: { ...change, previewDigest: preview.previewDigest }
                    },
                    preview,
                    baseline: baselineIdentity(snapshot),
                    authority,
                    authorityState,
                    inspection: inspectionIdentity(inspection)
                }
            })
        } catch (error) {
            put({ review: undefined, message: getErrorString(error) })
        } finally {
            put({ busy: false, previewOperation: undefined })
            modelPreview = undefined
            const pending = active.pending
            if (pending) {
                if (get().modelDrafts[pending.key]?.generation === pending.generation) {
                    // Release the queued marker before re-entering this owner.
                    set(state => ({
                        modelDrafts: {
                            ...state.modelDrafts,
                            [pending.key]: {
                                ...state.modelDrafts[pending.key],
                                busy: false,
                                previewOperation: undefined
                            }
                        }
                    }))
                    void pending.run().finally(pending.finish)
                } else pending.finish()
            }
        }
    },
    applyModel: async (key, authority) => {
        const draft = get().modelDrafts[key]
        const review = draft?.review
        if (!review || draft.busy || draft.uncertain) return
        const generation = draft.generation
        const request = { ...structuredClone(review.request), requestId: crypto.randomUUID() }
        const put = (value: Partial<ModelDraft>) =>
            set(state => ({
                modelDrafts: {
                    ...state.modelDrafts,
                    [key]: { ...state.modelDrafts[key], ...value }
                }
            }))
        const current = () => {
            const entry = get().entries[engineSettingsKey(request)]
            return (
                !entry?.unavailable &&
                baselineIdentity(entry?.snapshot) === review.baseline &&
                get().modelDrafts[key]?.generation === generation &&
                review.authority === authority &&
                modelAuthority(request) === review.authorityState
            )
        }
        let dispatched = false
        put({ busy: true })
        try {
            if (!current()) throw new Error('Preview expired. Your draft is kept.')
            await authorizeModel(request, current)
            const inspection = await window.pairApi.engines.inspectModel({
                engine: request.engine,
                nodeId: request.nodeId,
                modelId: request.modelChange?.modelId ?? ''
            })
            if (!current() || inspectionIdentity(inspection) !== review.inspection)
                throw new Error('Artifact, build or target changed. Preview again.')
            dispatched = true
            const policyDraft = useNodeConfigurationStore
                .getState()
                .policyDrafts.get(request.nodeId)
            const selected =
                policyDraft?.baseline && request.modelChange
                    ? modelPolicyPatch(policyDraft.baseline.policy, policyDraft.text, {
                          engine: request.engine,
                          model: request.modelChange.modelId
                      })
                    : undefined
            const submittedGeneration = request.modelChange?.policy?.generation ?? {}
            const draftGeneration = selected?.ok ? (selected.value.generation ?? {}) : {}
            const submittedPolicy =
                policyDraft?.baseline &&
                selected?.ok &&
                (Object.keys(draftGeneration).length > 0 ||
                    selected.value.requestOptionsMode !== undefined) &&
                selected.value.requestOptionsMode ===
                    request.modelChange?.policy?.requestOptionsMode &&
                Object.keys(draftGeneration).length === Object.keys(submittedGeneration).length &&
                Object.entries(draftGeneration).every(([field, value]) =>
                    Object.entries(submittedGeneration).some(
                        ([key, submitted]) => key === field && submitted === value
                    )
                )
                    ? {
                          owner: policyDraft.owner,
                          text: policyDraft.text,
                          revision: policyDraft.baseline.revision
                      }
                    : undefined
            put({
                attempt: request,
                submittedPolicy,
                review: undefined,
                uncertain: true,
                message: `Awaiting operation ${request.requestId}.`
            })
            await window.pairApi.engines.applySettings(request)
            // An acceptance receipt is not applied state. Correlate a full readback by request ID.
            await get().readModelOutcome(key)
        } catch (error) {
            const code = serviceErrorCode(error)
            const refused =
                code !== undefined &&
                [
                    INVALID_PARAMS_CODE,
                    LOCAL_VETO_CODE,
                    POLICY_CONFLICT_CODE,
                    REMOTE_MANAGEMENT_DENIED_CODE,
                    REMOTE_SETTING_LOCKED_CODE
                ].includes(code)
            put({
                review: undefined,
                ...(refused ? { uncertain: false } : {}),
                message: `${dispatched && !refused ? 'Outcome unknown. ' : refused ? 'Refused. ' : ''}${getErrorString(error)}`
            })
            if (dispatched && !refused) await get().readModelOutcome(key)
        } finally {
            put({ busy: false })
        }
    },
    readModelOutcome: async key => {
        const attempt = get().modelDrafts[key]?.attempt
        if (!attempt) return
        await get().reload(attempt)
        const entry = get().entries[engineSettingsKey(attempt)]
        const snapshot = entry?.snapshot
        const correlated =
            !entry?.unavailable && snapshot && snapshot.requestId === attempt.requestId
        set(state => ({
            modelDrafts: {
                ...state.modelDrafts,
                [key]: {
                    ...(correlated && snapshot
                        ? correlateModelOutcome(state.modelDrafts[key], snapshot)
                        : {
                              ...state.modelDrafts[key],
                              uncertain: true,
                              message: `Outcome unknown for ${attempt.requestId}. Read status before retry; no automatic resend.`
                          })
                }
            }
        }))
        const submitted = get().modelDrafts[key]?.submittedPolicy
        if (correlated && snapshot?.phase === 'succeeded' && submitted) {
            try {
                const local = attempt.nodeId === useConnectionStore.getState().selfId
                const policy = await window.pairApi.nodePolicy.get(
                    local ? undefined : attempt.nodeId
                )
                if (
                    get().modelDrafts[key]?.attempt !== attempt ||
                    policy.revision !== snapshot.policyRevision
                )
                    return
                const store = useNodeConfigurationStore.getState()
                const current = store.policyDrafts.get(attempt.nodeId)
                const unchanged =
                    current?.owner === submitted.owner &&
                    current.text === submitted.text &&
                    current.baseline?.revision === submitted.revision
                store.observePolicyDraft(attempt.nodeId, policy, 'read')
                // Adopt only the exact submitted draft. Later edits keep their
                // original CAS base and the existing explicit conflict flow.
                if (
                    unchanged &&
                    useNodeConfigurationStore.getState().policyDrafts.get(attempt.nodeId)?.saved
                        ?.revision === policy.revision
                ) {
                    store.claimPolicyDraft(attempt.nodeId, submitted.owner, true)
                    store.observePolicyDraft(attempt.nodeId, policy, 'read')
                }
                set(state => ({
                    modelDrafts: {
                        ...state.modelDrafts,
                        [key]: { ...state.modelDrafts[key], submittedPolicy: undefined }
                    }
                }))
            } catch (error) {
                set(state => ({
                    modelDrafts: {
                        ...state.modelDrafts,
                        [key]: {
                            ...state.modelDrafts[key],
                            message: `Operation succeeded; policy readback unavailable. Draft kept: ${getErrorString(error)}`
                        }
                    }
                }))
            }
        }
    },
    receive: snapshot => {
        const key = engineSettingsKey(snapshot)
        const current = get().entries[key]
        if (
            current?.snapshot?.epoch === snapshot.epoch &&
            (current.snapshot.sequence > snapshot.sequence ||
                (current.snapshot.sequence === snapshot.sequence && !current.unavailable))
        )
            return
        set(state => ({
            modelDrafts: Object.fromEntries(
                Object.entries(state.modelDrafts).map(([draftKey, draft]) => [
                    draftKey,
                    draftKey.startsWith(`${key}:`) &&
                    baselineIdentity(current?.snapshot) !== baselineIdentity(snapshot)
                        ? correlateModelOutcome(invalidateDraft(draft), snapshot)
                        : draft
                ])
            ),
            entries: {
                ...state.entries,
                [key]: { snapshot: structuredClone(snapshot), arrival: (current?.arrival ?? 0) + 1 }
            }
        }))
    },
    disconnect: nodeId =>
        set(state => ({
            modelDrafts: Object.fromEntries(
                Object.entries(state.modelDrafts).map(([key, draft]) => [
                    key,
                    !nodeId || key.startsWith(`${nodeId}:`) ? invalidateDraft(draft) : draft
                ])
            ),
            entries: Object.fromEntries(
                Object.entries(state.entries).map(([key, entry]) => [
                    key,
                    !nodeId || key.startsWith(`${nodeId}:`)
                        ? {
                              ...entry,
                              unavailable: 'Device disconnected. Reconnect to edit settings.',
                              arrival: entry.arrival + 1
                          }
                        : entry
                ])
            )
        })),
    reload: async target => {
        const key = engineSettingsKey(target)
        const arrival = get().entries[key]?.arrival ?? 0
        try {
            const snapshot = await window.pairApi.engines.getSettings(target)
            // A push/disconnect after this read began wins, including a new authority epoch.
            if ((get().entries[key]?.arrival ?? 0) !== arrival) return
            get().receive({ ...snapshot, nodeId: target.nodeId })
            set(state => ({
                entries: {
                    ...state.entries,
                    [key]: { ...state.entries[key], unavailable: undefined }
                }
            }))
        } catch (error) {
            if ((get().entries[key]?.arrival ?? 0) !== arrival) return
            set(state => ({
                entries: {
                    ...state.entries,
                    [key]: {
                        ...state.entries[key],
                        arrival,
                        unavailable: `Settings unavailable: ${getErrorString(error)}`
                    }
                }
            }))
        }
    },
    apply: async request => {
        try {
            // Receipts acknowledge operations; only a full snapshot updates
            // durable state, and engines:settings-changed delivers that on its
            // own. Re-reading here would just race the push.
            await window.pairApi.engines.applySettings(request)
        } catch (error) {
            // A transport failure leaves no push to wait for, so read back
            // whatever the device can still report — including nothing.
            await get().reload(request)
            throw error
        }
    }
}))

let consumers = 0
let unsubscribe: Array<() => void> = []
/** Subscribe before fetching a baseline. All open editors share one push listener. */
export function watchEngineSettings(): () => void {
    if (consumers++ === 0)
        unsubscribe = [
            window.pairApi.engines.onSettingsChanged(snapshot =>
                useEngineSettingsStore.getState().receive(snapshot)
            ),
            window.pairApi.engines.onSettingsDisconnected(({ nodeId }) =>
                useEngineSettingsStore.getState().disconnect(nodeId)
            ),
            useConnectionStore.subscribe((state, previous) => {
                if (state.connected !== previous.connected || state.selfId !== previous.selfId)
                    useEngineSettingsStore.getState().disconnect()
            }),
            useClusterInvitationsStore.subscribe((state, previous) => {
                if (state.remoteManagementContext !== previous.remoteManagementContext)
                    for (const key of Object.keys(useEngineSettingsStore.getState().modelDrafts))
                        useEngineSettingsStore.getState().invalidateModel(key)
            }),
            useNodeConfigurationStore.subscribe((state, previous) => {
                const store = useEngineSettingsStore.getState()
                const nodes = new Set([
                    ...state.management.keys(),
                    ...previous.management.keys(),
                    ...state.policyDrafts.keys(),
                    ...previous.policyDrafts.keys()
                ])
                for (const nodeId of nodes) {
                    const permission = state.management.get(nodeId)
                    const oldPermission = previous.management.get(nodeId)
                    const policy = state.policyDrafts.get(nodeId)
                    const oldPolicy = previous.policyDrafts.get(nodeId)
                    if (
                        permission?.state !== oldPermission?.state ||
                        permission?.revision !== oldPermission?.revision ||
                        (policy !== oldPolicy &&
                            (policy?.current !== oldPolicy?.current ||
                                policy?.saved?.revision !== oldPolicy?.saved?.revision))
                    )
                        for (const key of Object.keys(store.modelDrafts))
                            if (key.startsWith(`${nodeId}:`)) store.invalidateModel(key)
                }
            })
        ]
    // Closing the last editor says nothing about any device, so cached
    // snapshots stay as they are. `disconnect` is reserved for a real
    // engines:settings-disconnected push and for losing the service
    // connection. An editor unmounts whenever its engine starts installing,
    // so flagging every node here would mislabel the whole cluster.
    return () => {
        if (--consumers === 0) {
            unsubscribe.forEach(stop => stop())
            unsubscribe = []
        }
    }
}
