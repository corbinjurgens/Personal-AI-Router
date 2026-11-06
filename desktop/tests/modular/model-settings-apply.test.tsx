// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type {
    EngineSettingsPreview,
    EngineSettingsSnapshot,
    EngineSettingsTarget
} from '@/shared/types/engine-settings'
import type { ArtifactSummary, InspectModelResult, ModelChange } from '@/shared/types/model-config'
import type { NodePolicyDocument } from '@/shared/types/node-policy'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi } from '@/ui/api/pair-api'
import type { IEngineApi } from '@/ui/api/engine-api'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import {
    modelSettingsKey,
    useEngineSettingsStore,
    watchEngineSettings
} from '@/ui/stores/engine-settings.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { applyModelSettings } from '@/shared/utils/node-policy-model-settings'
import { ServiceCallError } from '@/ui/api/service-transport'
import { POLICY_CONFLICT_CODE } from '@/shared/constants/service-errors'
import {
    modelPolicyPatch,
    modelRoutingState,
    portableLoadPatch
} from '@/shared/utils/node-policy-model-copy'

const target: EngineSettingsTarget = { nodeId: 'self', engine: 'llamacpp' }
const key = modelSettingsKey(target, 'model')
const policy: NodePolicyDocument = {
    policy: '{"revision":"7","profiles":[{"key":"model","displayName":"test","engine":"llamacpp","model":"model"}],"opaque":9007199254740993}',
    revision: '7',
    activation: null,
    availability: 'available',
    recoveryRequired: false
}
const change: ModelChange = {
    profileKey: 'model',
    modelId: 'model',
    expectedPolicyRevision: '7',
    load: { set: { slots: 2 } }
}
function snapshot(extra: Partial<EngineSettingsSnapshot> = {}): EngineSettingsSnapshot {
    return {
        ...target,
        revision: 3,
        appliedRevision: 3,
        sequence: 1,
        epoch: 'authority',
        settings: {
            serverPort: 1,
            proxyPort: 2,
            launchText: 'target-only',
            modelLoads: [{ profileKey: 'other', modelId: 'other', requested: { slots: 3 } }]
        },
        effectiveServerPort: 1,
        effectiveProxyPort: 2,
        running: false,
        adopted: false,
        editable: true,
        reason: '',
        format: '',
        phase: 'idle',
        error: '',
        requestId: '',
        policyRevision: '7',
        ...extra
    }
}
function inspection(extra: Partial<InspectModelResult> = {}): InspectModelResult {
    return {
        engine: target.engine,
        nodeId: target.nodeId,
        modelId: 'model',
        status: 'unavailable',
        reason: 'Facts unavailable; descriptors may still be supported.',
        estimate: { status: 'unknown' },
        options: [
            {
                field: 'slots',
                label: 'Slots',
                kind: 'integer',
                minimum: 1,
                maximum: 8,
                portable: true,
                supported: true
            }
        ],
        ...extra
    }
}
class EngineFake implements Pick<
    IEngineApi,
    'getSettings' | 'inspectModel' | 'previewSettings' | 'applySettings'
> {
    getSettings = vi.fn<IEngineApi['getSettings']>().mockResolvedValue(snapshot())
    inspectModel = vi.fn<IEngineApi['inspectModel']>().mockResolvedValue(inspection())
    previewSettings = vi.fn<IEngineApi['previewSettings']>().mockResolvedValue({
        settings: { serverPort: 99, proxyPort: 98, launchText: 'foreign-preview-resolution' },
        revision: 3,
        policyRevision: '7',
        previewDigest: 'reviewed',
        errors: {},
        restart: true,
        rebind: false
    })
    applySettings = vi
        .fn<IEngineApi['applySettings']>()
        .mockResolvedValue({ revision: 4, phase: 'applying' })
}
let engine: EngineFake
function deferred<T>() {
    let finish: ((value: T) => void) | undefined
    const promise = new Promise<T>(resolve => {
        finish = resolve
    })
    return {
        promise,
        resolve: (value: T) => {
            if (!finish) throw new Error('Deferred promise was not initialized')
            finish(value)
        }
    }
}
beforeEach(() => {
    engine = new EngineFake()
    const transport: ServiceTransport = {
        connected: true,
        invoke: vi.fn<ServiceTransport['invoke']>(),
        subscribePush: () => () => {},
        onConnect: () => () => {},
        onDisconnect: () => () => {},
        onAuthFailure: () => () => {},
        destroy: () => {}
    }
    const api = createPairApi(transport)
    Object.assign(api.engines, engine)
    api.nodePolicy.get = vi.fn<typeof api.nodePolicy.get>().mockResolvedValue(policy)
    api.nodePolicy.set = vi.fn<typeof api.nodePolicy.set>()
    api.cluster.getRemoteManagementForNode = vi
        .fn<typeof api.cluster.getRemoteManagementForNode>()
        .mockResolvedValue({ version: 1, revision: '1', state: 'allowed', allowed: true })
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    useConnectionStore.setState({ connected: true, selfId: 'self' })
    useClusterInvitationsStore.setState({ remoteManagementContext: 1 })
    useNodeConfigurationStore.setState({
        policyDrafts: new Map(),
        management: new Map([['peer', { state: 'allowed', revision: '1' }]])
    })
    useEngineSettingsStore.setState({ entries: {}, modelDrafts: {} })
    useEngineSettingsStore.getState().receive(snapshot())
})

describe('reviewed model operations', () => {
    it('retains the refused digest and draft until explicit recalculation without retrying', async () => {
        const store = useEngineSettingsStore.getState()
        await store.previewModel(target, change, 'authority')
        const reviewed = useEngineSettingsStore.getState().modelDrafts[key].review?.request
        engine.applySettings.mockRejectedValueOnce(
            new ServiceCallError('Reviewed memory is stale. Recalculate.', POLICY_CONFLICT_CODE)
        )
        await store.applyModel(key, 'authority')
        const refused = useEngineSettingsStore.getState().modelDrafts[key]
        expect(engine.applySettings).toHaveBeenCalledTimes(1)
        expect(engine.applySettings.mock.calls[0]?.[0].modelChange?.previewDigest).toBe('reviewed')
        expect(refused.change).toEqual(change)
        expect(refused.retainedReview?.request).toEqual(reviewed)
        expect(refused.review).toBeUndefined()
        expect(refused.uncertain).toBe(false)
        expect(refused.message).toContain('Reviewed memory is stale. Recalculate.')
        await store.applyModel(key, 'authority')
        await store.previewModel(target, change, 'authority', true)
        expect(engine.applySettings).toHaveBeenCalledTimes(1)
        expect(engine.previewSettings).toHaveBeenCalledTimes(1)
        engine.previewSettings.mockResolvedValueOnce({
            settings: snapshot().settings,
            revision: 3,
            policyRevision: '7',
            previewDigest: 'explicit-replacement',
            errors: {},
            restart: false,
            rebind: false
        })
        await store.previewModel(target, change, 'authority')
        expect(
            useEngineSettingsStore.getState().modelDrafts[key].review?.preview.previewDigest
        ).toBe('explicit-replacement')
        expect(useEngineSettingsStore.getState().modelDrafts[key].retainedReview).toBeUndefined()
    })
    it('reviews and applies fit and raw options as one existing model operation', async () => {
        const store = useEngineSettingsStore.getState()
        const nativeChange: ModelChange = {
            profileKey: 'model',
            modelId: 'model',
            expectedPolicyRevision: '7',
            load: {
                set: {
                    fit: 'on',
                    fitTarget: '512',
                    fitCtx: 4096,
                    extraOptions: { 'spec-type': 'draft-mtp' }
                }
            }
        }
        store.editModel(key, nativeChange)
        await store.previewModel(target, nativeChange, 'authority')
        expect(engine.previewSettings.mock.calls[0]?.[0].modelChange).toEqual(nativeChange)
        await store.applyModel(key, 'authority')
        expect(engine.applySettings).toHaveBeenCalledOnce()
        expect(engine.applySettings.mock.calls[0]?.[0].modelChange).toEqual({
            ...nativeChange,
            previewDigest: 'reviewed'
        })
        expect(window.pairApi.nodePolicy.set).not.toHaveBeenCalled()
    })
    it('retains service option errors as advisory evidence without application authority', async () => {
        engine.previewSettings.mockResolvedValue({
            settings: snapshot().settings,
            revision: 3,
            policyRevision: '7',
            previewDigest: 'refused',
            errors: { 'extraOptions.port': 'PAIR owns port.' },
            restart: false,
            rebind: false
        })
        const invalidChange: ModelChange = {
            ...change,
            load: { set: { extraOptions: { port: '9999' } } }
        }
        const store = useEngineSettingsStore.getState()
        store.editModel(key, invalidChange)
        await store.previewModel(target, invalidChange, 'authority')
        expect(useEngineSettingsStore.getState().modelDrafts[key].advisory?.errors).toEqual({
            'extraOptions.port': 'PAIR owns port.'
        })
        expect(useEngineSettingsStore.getState().modelDrafts[key].review).toBeUndefined()
        await store.applyModel(key, 'authority')
        expect(engine.applySettings).not.toHaveBeenCalled()
    })
    it('preserves an explicit destination review when a source advisory arrives', async () => {
        const response = deferred<EngineSettingsPreview>()
        const started = deferred<void>()
        engine.previewSettings.mockImplementationOnce(() => {
            started.resolve()
            return response.promise
        })
        const store = useEngineSettingsStore.getState()
        const explicit = store.previewModel(target, change, 'authority')
        await started.promise
        const sourceTarget: EngineSettingsTarget = { ...target, nodeId: 'peer' }
        expect(modelSettingsKey(sourceTarget, change.modelId)).not.toBe(key)
        const advisory = store.previewModel(sourceTarget, change, 'source-authority', true)
        response.resolve({
            settings: snapshot().settings,
            revision: 3,
            policyRevision: '7',
            previewDigest: 'destination',
            errors: {},
            restart: false,
            rebind: false
        })
        await Promise.all([explicit, advisory])
        expect(engine.previewSettings).toHaveBeenCalledTimes(1)
        expect(
            useEngineSettingsStore.getState().modelDrafts[key].review?.preview.previewDigest
        ).toBe('destination')
    })
    it('does not replace an explicit review with an advisory for the same input', async () => {
        const response = deferred<EngineSettingsPreview>()
        const started = deferred<void>()
        engine.previewSettings.mockImplementationOnce(() => {
            started.resolve()
            return response.promise
        })
        const store = useEngineSettingsStore.getState()
        const explicit = store.previewModel(target, change, 'authority')
        await started.promise
        const advisory = store.previewModel(target, change, 'authority', true)
        response.resolve({
            settings: snapshot().settings,
            revision: 3,
            policyRevision: '7',
            previewDigest: 'explicit',
            errors: {},
            restart: false,
            rebind: false
        })
        await Promise.all([explicit, advisory])
        expect(engine.previewSettings).toHaveBeenCalledTimes(1)
        expect(
            useEngineSettingsStore.getState().modelDrafts[key].review?.preview.previewDigest
        ).toBe('explicit')
    })
    it('coalesces bursts behind one in-flight request and runs only the latest input', async () => {
        const response = deferred<EngineSettingsPreview>()
        const started = deferred<void>()
        engine.previewSettings.mockImplementationOnce(() => {
            started.resolve()
            return response.promise
        })
        const store = useEngineSettingsStore.getState()
        const first = store.previewModel(target, change, 'authority', true)
        await started.promise
        const pending = [3, 4, 5, 6].map(slots =>
            store.previewModel(target, { ...change, load: { set: { slots } } }, 'authority', true)
        )
        const inFlightReads = engine.getSettings.mock.calls.length
        response.resolve({
            settings: snapshot().settings,
            revision: 3,
            errors: {},
            restart: false,
            rebind: false
        })
        await Promise.all([first, ...pending])
        expect(inFlightReads).toBe(1)
        expect(engine.previewSettings).toHaveBeenCalledTimes(2)
        expect(engine.previewSettings.mock.calls[1]?.[0].modelChange?.load?.set.slots).toBe(6)
    })
    it('stops an obsolete inspection before dispatching Preview', async () => {
        const result = deferred<InspectModelResult>()
        const started = deferred<void>()
        engine.inspectModel.mockImplementationOnce(() => {
            started.resolve()
            return result.promise
        })
        const store = useEngineSettingsStore.getState()
        const pending = store.previewModel(target, change, 'authority')
        await started.promise
        store.invalidateModel(key)
        result.resolve(inspection())
        await pending
        expect(engine.previewSettings).not.toHaveBeenCalled()
    })
    it.each(['settings', 'authorization'])(
        'stops invalidated work after the awaited %s stage',
        async stage => {
            const started = deferred<void>()
            const settings = deferred<EngineSettingsSnapshot>()
            const authorization = deferred<NodePolicyDocument>()
            if (stage === 'settings')
                engine.getSettings.mockImplementationOnce(() => {
                    started.resolve()
                    return settings.promise
                })
            else
                vi.mocked(window.pairApi.nodePolicy.get).mockImplementationOnce(() => {
                    started.resolve()
                    return authorization.promise
                })
            const store = useEngineSettingsStore.getState()
            const pending = store.previewModel(target, change, 'authority')
            await started.promise
            store.invalidateModel(key)
            settings.resolve(snapshot())
            authorization.resolve(policy)
            await pending
            expect(engine.inspectModel).not.toHaveBeenCalled()
            expect(engine.previewSettings).not.toHaveBeenCalled()
        }
    )
    it('keeps a queued explicit review ahead of a same-input advisory', async () => {
        const response = deferred<EngineSettingsPreview>()
        const started = deferred<void>()
        engine.previewSettings.mockImplementationOnce(() => {
            started.resolve()
            return response.promise
        })
        const store = useEngineSettingsStore.getState()
        const first = store.previewModel(target, change, 'authority', true)
        await started.promise
        const explicit = store.previewModel(target, change, 'authority')
        const advisory = store.previewModel(target, change, 'authority', true)
        response.resolve({
            settings: snapshot().settings,
            revision: 3,
            errors: {},
            restart: false,
            rebind: false
        })
        await Promise.all([first, explicit, advisory])
        expect(engine.previewSettings).toHaveBeenCalledTimes(2)
        expect(
            useEngineSettingsStore.getState().modelDrafts[key].review?.preview.previewDigest
        ).toBe('reviewed')
    })
    it('discards an invalidated coalesced input before starting another request', async () => {
        const response = deferred<EngineSettingsPreview>()
        const started = deferred<void>()
        engine.previewSettings.mockImplementationOnce(() => {
            started.resolve()
            return response.promise
        })
        const store = useEngineSettingsStore.getState()
        const first = store.previewModel(target, change, 'authority', true)
        await started.promise
        const pending = store.previewModel(
            target,
            { ...change, load: { set: { slots: 4 } } },
            'authority',
            true
        )
        store.invalidateModel(key)
        response.resolve({
            settings: snapshot().settings,
            revision: 3,
            errors: {},
            restart: false,
            rebind: false
        })
        await Promise.all([first, pending])
        expect(engine.getSettings).toHaveBeenCalledTimes(1)
        expect(engine.previewSettings).toHaveBeenCalledTimes(1)
        expect(useEngineSettingsStore.getState().modelDrafts[key].review).toBeUndefined()
    })
    it('coalesces target changes behind the same request owner', async () => {
        const response = deferred<EngineSettingsPreview>()
        const started = deferred<void>()
        engine.previewSettings.mockImplementationOnce(() => {
            started.resolve()
            return response.promise
        })
        engine.inspectModel.mockImplementation(async request =>
            inspection({ modelId: request.modelId })
        )
        const store = useEngineSettingsStore.getState()
        const first = store.previewModel(target, change, 'old authority')
        await started.promise
        const next = store.previewModel(
            target,
            { ...change, modelId: 'other', profileKey: 'other' },
            'new authority'
        )
        const reads = engine.getSettings.mock.calls.length
        response.resolve({
            settings: snapshot().settings,
            revision: 3,
            errors: {},
            restart: false,
            rebind: false
        })
        await Promise.all([first, next])
        expect(reads).toBe(1)
        expect(engine.previewSettings).toHaveBeenCalledTimes(2)
        expect(useEngineSettingsStore.getState().modelDrafts[key].review).toBeUndefined()
        expect(
            useEngineSettingsStore.getState().modelDrafts[modelSettingsKey(target, 'other')].review
                ?.authority
        ).toBe('new authority')
    })
    it('recalculates Auto without saving and refuses an incomplete proposal even with a digest', async () => {
        const automatic: ModelChange = {
            ...change,
            load: { set: { context: { mode: 'auto', floor: 4096, cap: 8192 } } }
        }
        engine.previewSettings.mockResolvedValue({
            settings: snapshot().settings,
            revision: 3,
            policyRevision: '7',
            previewDigest: 'digest',
            errors: {},
            restart: false,
            rebind: false,
            auto: {
                status: 'unavailable',
                requested: automatic.load?.set ?? {},
                estimate: { status: 'partial' }
            }
        })
        const store = useEngineSettingsStore.getState()
        await store.previewModel(target, automatic, 'authority')
        await store.previewModel(target, automatic, 'authority')
        expect(engine.previewSettings).toHaveBeenCalledTimes(2)
        expect(useEngineSettingsStore.getState().modelDrafts[key].review).toBeUndefined()
        expect(
            useEngineSettingsStore.getState().entries['self:llamacpp'].snapshot?.settings
        ).toEqual(snapshot().settings)
        await store.applyModel(key, 'authority')
        expect(engine.applySettings).not.toHaveBeenCalled()
    })
    it('keeps the newest advisory when an older preview finishes late', async () => {
        const old = deferred<EngineSettingsPreview>()
        const started = deferred<void>()
        engine.previewSettings
            .mockImplementationOnce(() => {
                started.resolve()
                return old.promise
            })
            .mockResolvedValueOnce({
                settings: snapshot().settings,
                revision: 3,
                policyRevision: '7',
                errors: {},
                restart: false,
                rebind: false,
                estimate: { status: 'unknown', source: 'new' }
            })
        const store = useEngineSettingsStore.getState()
        const first = store.previewModel(target, change, 'authority', true)
        await started.promise
        const next = { ...change, load: { set: { slots: 4 } } }
        const pending = store.previewModel(target, next, 'authority', true)
        old.resolve({
            settings: snapshot().settings,
            revision: 3,
            policyRevision: '7',
            errors: {},
            restart: false,
            rebind: false,
            estimate: { status: 'unknown', source: 'old' }
        })
        await Promise.all([first, pending])
        expect(useEngineSettingsStore.getState().modelDrafts[key].advisory?.estimate?.source).toBe(
            'new'
        )
        expect(useEngineSettingsStore.getState().modelDrafts[key].review).toBeUndefined()
    })
    it('retains an unavailable Auto advisory without authorizing Apply, then invalidates it', async () => {
        const automatic: ModelChange = {
            ...change,
            load: { set: { context: { mode: 'auto', floor: 4096, cap: 8192 } } }
        }
        engine.previewSettings.mockResolvedValueOnce({
            settings: snapshot().settings,
            revision: 3,
            policyRevision: '7',
            errors: { context: 'Auto unavailable' },
            restart: false,
            rebind: false,
            auto: {
                status: 'unavailable',
                requested: automatic.load?.set ?? {},
                estimate: { status: 'unknown' }
            }
        })
        const store = useEngineSettingsStore.getState()
        await store.previewModel(target, automatic, 'authority')
        expect(store.modelDrafts[key]).toBeUndefined()
        expect(useEngineSettingsStore.getState().modelDrafts[key].advisory?.auto?.status).toBe(
            'unavailable'
        )
        expect(useEngineSettingsStore.getState().modelDrafts[key].review).toBeUndefined()
        await store.applyModel(key, 'authority')
        expect(engine.applySettings).not.toHaveBeenCalled()
        store.editModel(key, change)
        expect(useEngineSettingsStore.getState().modelDrafts[key].advisory).toBeUndefined()
    })
    it('omits empty policy patches and inspects an untyped profile without a saved-key selector before Apply', async () => {
        const store = useEngineSettingsStore.getState()
        await store.previewModel(
            target,
            { ...change, policy: { generation: {}, roles: {} } },
            'authority'
        )
        expect(engine.previewSettings.mock.calls[0]?.[0].modelChange?.policy).toBeUndefined()
        await store.applyModel(key, 'authority')
        expect(engine.inspectModel.mock.calls).toEqual([
            [{ ...target, modelId: 'model' }],
            [{ ...target, modelId: 'model' }]
        ])
        expect(engine.applySettings.mock.calls[0]?.[0].modelChange?.policy).toBeUndefined()
        expect(engine.applySettings.mock.calls[0]?.[0].modelChange?.profileKey).toBe('model')
    })
    it('inspects the selected saved key during review and fresh Apply validation', async () => {
        const saved = snapshot({
            sequence: 2,
            settings: {
                ...snapshot().settings,
                modelLoads: [
                    { profileKey: 'sibling', modelId: 'model', requested: { slots: 5 } },
                    { profileKey: 'model', modelId: 'model', requested: { slots: 1 } }
                ]
            }
        })
        engine.getSettings.mockResolvedValue(saved)
        const store = useEngineSettingsStore.getState()
        store.receive(saved)
        await store.previewModel(target, change, 'authority')
        await store.applyModel(key, 'authority')
        expect(engine.inspectModel.mock.calls).toEqual([
            [{ ...target, modelId: 'model', profileKey: 'model' }],
            [{ ...target, modelId: 'model', profileKey: 'model' }]
        ])
        expect(engine.applySettings).toHaveBeenCalledOnce()
        expect(engine.applySettings.mock.calls[0]?.[0].modelChange).toMatchObject({
            profileKey: 'model',
            modelId: 'model',
            load: { set: { slots: 2 } }
        })
    })
    it.each([false, true])(
        'reconciles completed policy edits while preserving later edits=%s',
        async laterEdit => {
            const edited = policy.policy.replace(
                '"model":"model"',
                '"model":"model","requestOptions":{"temperature":0.3}'
            )
            const saved = {
                ...policy,
                revision: '8',
                policy: edited.replace('"revision":"7"', '"revision":"8"')
            }
            const store = useNodeConfigurationStore.getState()
            store.claimPolicyDraft('self', 'Model Manager')
            store.observePolicyDraft('self', policy, 'read')
            store.editPolicyDraft('self', 'Model Manager', edited)
            engine.applySettings.mockImplementation(async request => {
                if (laterEdit)
                    store.editPolicyDraft('self', 'Model Manager', edited.replace('0.3', '0.4'))
                window.pairApi.nodePolicy.get = vi
                    .fn<typeof window.pairApi.nodePolicy.get>()
                    .mockResolvedValue(saved)
                engine.getSettings.mockResolvedValue(
                    snapshot({
                        revision: 4,
                        sequence: 2,
                        policyRevision: '8',
                        requestId: request.requestId,
                        phase: 'succeeded'
                    })
                )
                return { revision: 4, phase: 'succeeded' }
            })
            await useEngineSettingsStore
                .getState()
                .previewModel(
                    target,
                    { ...change, policy: { generation: { temperature: 0.3 } } },
                    'authority'
                )
            await useEngineSettingsStore.getState().applyModel(key, 'authority')
            const draft = useNodeConfigurationStore.getState().policyDrafts.get('self')
            expect(draft?.saved?.revision).toBe('8')
            if (laterEdit) {
                expect(draft?.baseline?.revision).toBe('7')
                expect(draft?.text).toContain('0.4')
                expect(draft?.phase).toBe('conflict')
            } else {
                expect(draft?.baseline?.revision).toBe('8')
                expect(draft?.text).toBe(saved.policy)
                expect(draft?.current).toBe(true)
            }
        }
    )
    it.each(['snapshot', 'authority', 'disconnect'])(
        'releases an invalidated pending %s preview without clearing a newer preview’s busy ownership',
        async reason => {
            const first = deferred<EngineSettingsPreview>()
            const second = deferred<EngineSettingsPreview>()
            const firstStarted = deferred<void>()
            const secondStarted = deferred<void>()
            const response: EngineSettingsPreview = {
                settings: snapshot().settings,
                revision: 3,
                policyRevision: '7',
                previewDigest: 'new review',
                errors: {},
                restart: true,
                rebind: false
            }
            engine.previewSettings.mockImplementationOnce(() => {
                firstStarted.resolve()
                return first.promise
            })
            engine.previewSettings.mockImplementationOnce(() => {
                secondStarted.resolve()
                return second.promise
            })
            const stop = watchEngineSettings()
            try {
                const store = useEngineSettingsStore.getState()
                const pending = store.previewModel(target, change, 'first authority')
                await firstStarted.promise
                if (reason === 'snapshot') {
                    const changed = snapshot({ sequence: 2, revision: 4 })
                    engine.getSettings.mockResolvedValue(changed)
                    store.receive(changed)
                    response.revision = 4
                } else if (reason === 'authority')
                    useClusterInvitationsStore.setState({ remoteManagementContext: 2 })
                else store.disconnect('self')
                expect(useEngineSettingsStore.getState().modelDrafts[key]).toMatchObject({
                    busy: false,
                    change
                })
                const fresh = store.previewModel(target, change, 'new authority')
                first.resolve({ ...response, previewDigest: 'obsolete' })
                await pending
                await secondStarted.promise
                expect(useEngineSettingsStore.getState().modelDrafts[key].busy).toBe(true)
                second.resolve(response)
                await fresh
                expect(useEngineSettingsStore.getState().modelDrafts[key]).toMatchObject({
                    busy: false,
                    change,
                    review: { preview: { previewDigest: 'new review' }, authority: 'new authority' }
                })
            } finally {
                stop()
            }
        }
    )
    it('keeps Apply unavailable when a supported descriptor returns a not-ready preview', async () => {
        engine.previewSettings.mockRejectedValue(new Error('modelChange: application is not ready'))
        await useEngineSettingsStore.getState().previewModel(target, change, 'authority')
        expect(useEngineSettingsStore.getState().modelDrafts[key]).toMatchObject({
            change,
            message: 'modelChange: application is not ready'
        })
        expect(useEngineSettingsStore.getState().modelDrafts[key].review).toBeUndefined()
        await useEngineSettingsStore.getState().applyModel(key, 'authority')
        expect(engine.applySettings).not.toHaveBeenCalled()
    })
    it('sends one combined operation with target baseline, both revisions and reviewed digest', async () => {
        const combined = {
            ...change,
            policy: { generation: { temperature: 0.3 }, roles: { weak: true } }
        }
        await useEngineSettingsStore.getState().previewModel(target, combined, 'authority')
        await useEngineSettingsStore.getState().applyModel(key, 'authority')
        expect(engine.applySettings).toHaveBeenCalledTimes(1)
        expect(window.pairApi.nodePolicy.set).not.toHaveBeenCalled()
        const request = engine.applySettings.mock.calls[0]?.[0]
        expect(request).toMatchObject({
            expectedRevision: 3,
            settings: snapshot().settings,
            modelChange: { ...combined, previewDigest: 'reviewed' },
            requestId: expect.any(String)
        })
        expect(request?.settings.launchText).toBe('target-only')
    })
    it.each([
        ['engine revision', { revision: 4 }],
        ['policy revision', { policyRevision: '8' }],
        ['authority epoch', { epoch: 'replacement' }],
        ['permission', { editable: false }],
        [
            'artifact/native source',
            { settings: { serverPort: 1, proxyPort: 2, launchText: 'changed build/source' } }
        ]
    ] satisfies Array<[string, Partial<EngineSettingsSnapshot>]>)(
        'invalidates on %s while retaining the draft',
        async (_name, update) => {
            await useEngineSettingsStore.getState().previewModel(target, change, 'authority')
            useEngineSettingsStore.getState().receive(snapshot({ sequence: 2, ...update }))
            expect(useEngineSettingsStore.getState().modelDrafts[key].review).toBeUndefined()
            expect(useEngineSettingsStore.getState().modelDrafts[key].change).toEqual(change)
            await useEngineSettingsStore.getState().applyModel(key, 'authority')
            expect(engine.applySettings).not.toHaveBeenCalled()
        }
    )
    it('invalidates review on editing and disconnect without dropping dirty fields', async () => {
        const store = useEngineSettingsStore.getState()
        await store.previewModel(target, change, 'authority')
        store.editModel(key, { ...change, load: { set: { slots: 4 } } })
        store.disconnect('self')
        await store.applyModel(key, 'authority')
        expect(engine.applySettings).not.toHaveBeenCalled()
        expect(useEngineSettingsStore.getState().modelDrafts[key].change?.load?.set.slots).toBe(4)
    })
    it('refuses a changed permission authority token before dispatch', async () => {
        await useEngineSettingsStore.getState().previewModel(target, change, 'permission 1')
        await useEngineSettingsStore.getState().applyModel(key, 'permission 2')
        expect(engine.applySettings).not.toHaveBeenCalled()
    })
    it('requires a new review when artifact/build identity changes after preview', async () => {
        await useEngineSettingsStore.getState().previewModel(target, change, 'authority')
        engine.inspectModel.mockResolvedValue(inspection({ source: 'replacement native build' }))
        await useEngineSettingsStore.getState().applyModel(key, 'authority')
        expect(engine.applySettings).not.toHaveBeenCalled()
        expect(useEngineSettingsStore.getState().modelDrafts[key].message).toContain(
            'Artifact, build or target changed'
        )
    })
    it('does not expire review for estimate timestamps, optional estimates or display-only facts refresh', async () => {
        engine.inspectModel.mockResolvedValue(
            inspection({
                estimate: {
                    status: 'estimated',
                    timestamp: 'first',
                    memoryTimestamp: 'first',
                    hostBytes: '1024'
                }
            })
        )
        await useEngineSettingsStore.getState().previewModel(target, change, 'authority')
        engine.inspectModel.mockResolvedValue(
            inspection({
                reason: 'Refreshed label',
                estimate: { status: 'unknown', timestamp: 'second', memoryTimestamp: 'second' }
            })
        )
        await useEngineSettingsStore.getState().applyModel(key, 'authority')
        expect(engine.applySettings).toHaveBeenCalledOnce()
    })
    it('refuses changed artifact identity even when advisory estimates stay identical', async () => {
        const artifact: ArtifactSummary = {
            role: 'model',
            identity: 'old artifact',
            size: '10',
            assurance: 'bound'
        }
        engine.inspectModel.mockResolvedValue(inspection({ artifacts: [artifact] }))
        await useEngineSettingsStore.getState().previewModel(target, change, 'authority')
        engine.inspectModel.mockResolvedValue(
            inspection({ artifacts: [{ ...artifact, identity: 'new artifact' }] })
        )
        await useEngineSettingsStore.getState().applyModel(key, 'authority')
        expect(engine.applySettings).not.toHaveBeenCalled()
    })
    it('invalidates review on an actual target native build identity change', async () => {
        const native = snapshot()
        native.settings.modelLoads = [
            {
                profileKey: 'model',
                modelId: 'model',
                requested: {},
                resolved: {
                    perSlotContext: 8192,
                    totalContext: 8192,
                    slots: 1,
                    gpu: { mode: 'cpu' },
                    options: {},
                    artifacts: [],
                    nativeBuildId: 'native-build-one',
                    sourceInputDigest: 'input',
                    presetDigest: 'preset'
                }
            }
        ]
        engine.getSettings.mockResolvedValue(native)
        useEngineSettingsStore.getState().receive({ ...native, sequence: 2 })
        await useEngineSettingsStore.getState().previewModel(target, change, 'authority')
        const replacement = structuredClone(native)
        const resolved = replacement.settings.modelLoads?.[0].resolved
        if (!resolved) throw new Error('Missing native build fixture')
        resolved.nativeBuildId = 'native-build-two'
        useEngineSettingsStore.getState().receive({ ...replacement, sequence: 3 })
        await useEngineSettingsStore.getState().applyModel(key, 'authority')
        expect(engine.applySettings).not.toHaveBeenCalled()
    })
    it('rechecks live authority context during an awaited apply read even when allowed/revision return unchanged', async () => {
        const peer = { ...target, nodeId: 'peer' }
        const peerKey = modelSettingsKey(peer, 'model')
        engine.getSettings.mockResolvedValue(snapshot({ nodeId: 'peer' }))
        engine.inspectModel.mockResolvedValue(inspection({ nodeId: 'peer' }))
        useEngineSettingsStore.getState().receive(snapshot({ nodeId: 'peer' }))
        await useEngineSettingsStore.getState().previewModel(peer, change, 'same captured token')
        const started = deferred<void>()
        let finish: ((value: NodePolicyDocument) => void) | undefined
        window.pairApi.nodePolicy.get = vi
            .fn<typeof window.pairApi.nodePolicy.get>()
            .mockImplementation(
                () =>
                    new Promise<NodePolicyDocument>(resolve => {
                        finish = resolve
                        started.resolve()
                    })
            )
        const applying = useEngineSettingsStore
            .getState()
            .applyModel(peerKey, 'same captured token')
        await started.promise
        useClusterInvitationsStore.setState({ remoteManagementContext: 2 })
        if (!finish) throw new Error('Apply did not reach its read')
        finish(policy)
        await applying
        expect(engine.applySettings).not.toHaveBeenCalled()
        expect(useEngineSettingsStore.getState().modelDrafts[peerKey].change).toEqual(change)
    })
    it('reads back a lost response by request ID and clears unchanged completed edits without resending', async () => {
        engine.applySettings.mockImplementation(async request => {
            engine.getSettings.mockResolvedValue(
                snapshot({
                    revision: 4,
                    sequence: 2,
                    requestId: request.requestId,
                    phase: 'succeeded'
                })
            )
            throw new Error('reply lost')
        })
        await useEngineSettingsStore.getState().previewModel(target, change, 'authority')
        await useEngineSettingsStore.getState().applyModel(key, 'authority')
        expect(engine.applySettings).toHaveBeenCalledTimes(1)
        expect(useEngineSettingsStore.getState().modelDrafts[key]).toMatchObject({
            uncertain: false
        })
        expect(useEngineSettingsStore.getState().modelDrafts[key].change).toBeUndefined()
        expect(useEngineSettingsStore.getState().modelDrafts[key].message).toContain('succeeded')
    })
    it('keeps edits made after dispatch when a lost response is read back as succeeded', async () => {
        const later = { ...change, load: { set: { slots: 4 } } }
        engine.applySettings.mockImplementation(async request => {
            useEngineSettingsStore.getState().editModel(key, later)
            engine.getSettings.mockResolvedValue(
                snapshot({
                    revision: 4,
                    sequence: 2,
                    requestId: request.requestId,
                    phase: 'succeeded'
                })
            )
            throw new Error('reply lost')
        })
        await useEngineSettingsStore.getState().previewModel(target, change, 'authority')
        await useEngineSettingsStore.getState().applyModel(key, 'authority')
        expect(useEngineSettingsStore.getState().modelDrafts[key]).toMatchObject({
            change: later,
            uncertain: false
        })
        expect(engine.applySettings).toHaveBeenCalledOnce()
    })
    it('blocks retry until readback can correlate the uncertain request', async () => {
        engine.applySettings.mockRejectedValue(new Error('reply lost'))
        const store = useEngineSettingsStore.getState()
        await store.previewModel(target, change, 'authority')
        await store.applyModel(key, 'authority')
        const previews = engine.previewSettings.mock.calls.length
        await store.previewModel(target, change, 'authority')
        expect(engine.previewSettings).toHaveBeenCalledTimes(previews)
        expect(useEngineSettingsStore.getState().modelDrafts[key].uncertain).toBe(true)
        expect(useEngineSettingsStore.getState().modelDrafts[key].message).toContain(
            'Outcome unknown'
        )
        expect(engine.applySettings).toHaveBeenCalledTimes(1)
    })
    it('accepts an explicit local failed-operation supersession only through a fresh preview', async () => {
        const failed = snapshot({
            phase: 'failed',
            requestId: 'failed-op',
            coupledStage: 'policy',
            unresolvedError: 'reconcile'
        })
        engine.getSettings.mockResolvedValue(failed)
        useEngineSettingsStore.getState().receive({ ...failed, sequence: 2 })
        await useEngineSettingsStore
            .getState()
            .previewModel(target, { ...change, supersedesRequestId: 'failed-op' }, 'authority')
        expect(engine.previewSettings.mock.calls[0]?.[0].modelChange?.supersedesRequestId).toBe(
            'failed-op'
        )
        expect(engine.applySettings).not.toHaveBeenCalled()
    })
    it('permits load-only preview beside a saved generation lock and refuses remote paused destination', async () => {
        const peer = { ...target, nodeId: 'peer' }
        engine.getSettings.mockResolvedValue(snapshot({ nodeId: 'peer' }))
        engine.inspectModel.mockResolvedValue(inspection({ nodeId: 'peer' }))
        window.pairApi.nodePolicy.get = vi
            .fn<typeof window.pairApi.nodePolicy.get>()
            .mockResolvedValue({
                ...policy,
                policy: policy.policy.replace(
                    '"displayName":"test"',
                    '"localOnly":true,"displayName":"test"'
                )
            })
        useEngineSettingsStore.getState().receive(snapshot({ nodeId: 'peer' }))
        await useEngineSettingsStore.getState().previewModel(peer, change, 'authority')
        expect(engine.previewSettings).toHaveBeenCalledOnce()
        window.pairApi.nodePolicy.get = vi
            .fn<typeof window.pairApi.nodePolicy.get>()
            .mockResolvedValue({ ...policy, availability: 'paused' })
        await useEngineSettingsStore.getState().previewModel(peer, change, 'authority')
        expect(engine.previewSettings).toHaveBeenCalledOnce()
        expect(
            useEngineSettingsStore.getState().modelDrafts[modelSettingsKey(peer, 'model')].message
        ).toContain('paused')
    })
})

describe('memory observation lifetime', () => {
    beforeEach(() => vi.useFakeTimers())
    afterEach(() => vi.useRealTimers())
    const estimate = {
        status: 'estimated',
        capacityFit: 'fits',
        freeFit: 'fits',
        domains: [
            {
                id: 'host',
                kind: 'host',
                capacityBytes: '16000',
                freeBytes: '12000',
                requiredBytes: '1000',
                capacityFit: 'fits',
                freeFit: 'fits',
                source: 'fixture',
                sampleId: 'unchanged-sample',
                remainingValidityMs: 5000
            }
        ]
    } satisfies NonNullable<InspectModelResult['estimate']>

    it('deducts the complete inspection request and expires both comparisons with no refresh', async () => {
        const response = deferred<InspectModelResult>()
        engine.inspectModel.mockReturnValueOnce(response.promise)
        const store = useEngineSettingsStore.getState()
        const pending = store.inspectModel(target, 'model', 'model')
        vi.advanceTimersByTime(1500)
        response.resolve(inspection({ estimate }))
        await pending
        expect(
            useEngineSettingsStore.getState().modelDrafts[key].inspection?.estimate?.domains?.[0]
        ).toMatchObject({ remainingValidityMs: 3500, capacityFit: 'fits', freeFit: 'fits' })
        vi.setSystemTime(new Date('2000-01-01T00:00:00Z'))
        vi.advanceTimersByTime(3499)
        expect(useEngineSettingsStore.getState().modelDrafts[key].memoryStale).toBe(false)
        vi.advanceTimersByTime(1)
        const expired = useEngineSettingsStore.getState().modelDrafts[key]
        expect(expired.inspection?.estimate).toMatchObject({
            capacityFit: 'unknown',
            freeFit: 'unknown'
        })
        expect(expired.inspection?.estimate?.domains?.[0]).toMatchObject({
            capacityFit: 'unknown',
            freeFit: 'unknown',
            remainingValidityMs: 0,
            sampleId: 'unchanged-sample',
            capacityBytes: '16000',
            freeBytes: '12000'
        })
        expect(expired.memoryStale).toBe(true)
        expect(engine.inspectModel).toHaveBeenCalledTimes(1)
        expect(engine.previewSettings).not.toHaveBeenCalled()
        expect(engine.applySettings).not.toHaveBeenCalled()
    })

    it('keeps a fresh applied inspection separate from a historical retained review', async () => {
        const store = useEngineSettingsStore.getState()
        engine.previewSettings.mockResolvedValueOnce({
            settings: snapshot().settings,
            revision: 3,
            policyRevision: '7',
            previewDigest: 'historical',
            errors: {},
            restart: false,
            rebind: false,
            estimate
        })
        await store.previewModel(target, change, 'authority')
        vi.advanceTimersByTime(1000)
        engine.inspectModel.mockResolvedValueOnce(inspection({ estimate }))
        await store.inspectModel(target, 'model', 'model')
        const draft = useEngineSettingsStore.getState().modelDrafts[key]
        expect(draft.inspection?.estimate.capacityFit).toBe('fits')
        expect(draft.inspection?.estimate.domains?.[0].remainingValidityMs).toBe(5000)
        expect(draft.retainedReview?.preview.estimate?.capacityFit).toBe('unknown')
        expect(draft.retainedReview?.request.modelChange?.previewDigest).toBe('historical')
        expect(draft.review).toBeUndefined()
    })

    it('deducts preview request time while retaining the original reviewed digest after expiry', async () => {
        const response = deferred<EngineSettingsPreview>()
        const started = deferred<void>()
        engine.previewSettings.mockImplementationOnce(() => {
            started.resolve()
            return response.promise
        })
        const pending = useEngineSettingsStore.getState().previewModel(target, change, 'authority')
        await started.promise
        vi.advanceTimersByTime(5000)
        response.resolve({
            settings: snapshot().settings,
            revision: 3,
            policyRevision: '7',
            previewDigest: 'original',
            errors: {},
            restart: false,
            rebind: false,
            estimate
        })
        await pending
        const reviewed = useEngineSettingsStore.getState().modelDrafts[key]
        expect(reviewed.review?.preview.estimate).toMatchObject({
            capacityFit: 'unknown',
            freeFit: 'unknown'
        })
        expect(reviewed.review?.request.modelChange?.previewDigest).toBe('original')
        expect(reviewed.change).toEqual(change)
        expect(engine.previewSettings).toHaveBeenCalledTimes(1)
    })

    it('keeps hidden and disconnected observations historical after reconnect and rejects late hidden responses', async () => {
        const store = useEngineSettingsStore.getState()
        const stop = watchEngineSettings()
        try {
            engine.inspectModel.mockResolvedValue(inspection({ estimate }))
            await store.inspectModel(target, 'model', 'model')
            document.dispatchEvent(new Event('visibilitychange'))
            expect(
                useEngineSettingsStore.getState().modelDrafts[key].inspection?.estimate?.capacityFit
            ).toBe('unknown')
            const late = deferred<InspectModelResult>()
            engine.inspectModel.mockReturnValueOnce(late.promise)
            const pending = store.inspectModel(target, 'model', 'model')
            document.dispatchEvent(new Event('visibilitychange'))
            late.resolve(inspection({ estimate }))
            await pending
            expect(useEngineSettingsStore.getState().modelDrafts[key].inspection).toBeUndefined()
            await store.inspectModel(target, 'model', 'model')
            store.disconnect('self')
            await store.reload(target)
            expect(
                useEngineSettingsStore.getState().modelDrafts[key].inspection?.estimate?.capacityFit
            ).toBe('unknown')
            expect(engine.previewSettings).not.toHaveBeenCalled()
        } finally {
            stop()
        }
    })
})

describe('selected policy and portable load patches', () => {
    const identity = { engine: 'llamacpp', model: 'model', profileKey: 'model' }
    it.each(['llamacpp-custom', 'custom-engine'])(
        'selects the raw profile after a preceding %s profile',
        engine => {
            const baseline = policy.policy.replace(
                '"profiles":[',
                `"profiles":[{"key":"other","displayName":"other","engine":"${engine}","model":"other","opaque":9007199254740993},`
            )
            const edited = applyModelSettings(baseline, identity, {
                kind: 'option',
                key: 'temperature',
                value: '0.2'
            })
            if (!edited.ok) throw new Error(edited.error.message)
            expect(modelPolicyPatch(baseline, edited.value, identity)).toEqual({
                ok: true,
                value: { generation: { temperature: 0.2 } }
            })
            expect(edited.value).toContain('"opaque":9007199254740993')
            expect(
                modelPolicyPatch(
                    baseline,
                    edited.value.replaceAll('9007199254740993', '9007199254740992'),
                    identity
                ).ok
            ).toBe(false)
        }
    )
    it('accepts reordered selected additions and retains exact unrelated numeric tokens', () => {
        const seed = applyModelSettings(policy.policy, identity, {
            kind: 'option',
            key: 'seed',
            value: '42'
        })
        if (!seed.ok) throw new Error(seed.error.message)
        const temperature = applyModelSettings(seed.value, identity, {
            kind: 'option',
            key: 'temperature',
            value: '0.2'
        })
        if (!temperature.ok) throw new Error(temperature.error.message)
        expect(modelPolicyPatch(policy.policy, temperature.value, identity)).toEqual({
            ok: true,
            value: { generation: { temperature: 0.2, seed: 42 } }
        })
        expect(temperature.value).toContain('9007199254740993')
        expect(
            modelPolicyPatch(
                policy.policy,
                temperature.value.replace('9007199254740993', '9007199254740992'),
                identity
            ).ok
        ).toBe(false)
        expect(
            modelPolicyPatch(
                policy.policy,
                temperature.value.replace(
                    '"displayName":"test"',
                    '"unrelated":"foreign","displayName":"test"'
                ),
                identity
            ).ok
        ).toBe(false)
    })
    it('does not copy unselected resolved/source authority or clear unrelated requested leaves', () => {
        expect(
            portableLoadPatch(
                { slots: 2, gpu: { mode: 'cpu' }, context: { mode: 'fixed', tokens: 8192 } },
                ['slots', 'cacheTypeK']
            )
        ).toEqual({ set: { slots: 2 }, clear: ['cacheTypeK'] })
    })
    it('distinguishes nil/empty served permissions and reads the actual engineBindings container', () => {
        expect(modelRoutingState(policy.policy, identity).served).toBe('unrestricted')
        expect(modelRoutingState('{"servedModels":null}', identity).served).toBe('unrestricted')
        expect(modelRoutingState('{"servedModels":[]}', identity).served).toBe('excluded')
        const selected = { engine: 'llamacpp', model: 'artifact.gguf', profileKey: 'profile-key' }
        const routingPolicy = {
            profiles: [
                {
                    key: 'profile-key',
                    displayName: 'Profile',
                    engine: 'llamacpp',
                    model: 'served-model'
                }
            ],
            engineBindings: [{ engine: 'llamacpp', model: 'served-model' }],
            servedModels: [{ engine: 'llamacpp', model: 'served-model' }]
        }
        expect(modelRoutingState(JSON.stringify(routingPolicy), selected).binding).toBe(true)
        expect(modelRoutingState(JSON.stringify(routingPolicy), selected).served).toBe('included')
        expect(
            modelRoutingState(JSON.stringify({ ...routingPolicy, engineBindings: [] }), selected)
                .binding
        ).toBe(false)
        const artifactReferences = {
            ...routingPolicy,
            engineBindings: [{ engine: 'llamacpp', model: 'artifact.gguf' }],
            servedModels: [{ engine: 'llamacpp', model: 'artifact.gguf' }]
        }
        expect(modelRoutingState(JSON.stringify(artifactReferences), selected).binding).toBe(false)
        expect(modelRoutingState(JSON.stringify(artifactReferences), selected).served).toBe(
            'excluded'
        )
    })
    it('keeps residency unavailable for a qualification object without current target evidence', () => {
        const text =
            '{"revision":"7","profiles":[{"key":"model","displayName":"q","engine":"llamacpp","model":"model","contextTokens":8192,"maxOutputTokens":1024,"maxConcurrent":1,"memoryBytes":1024,"qualification":{"id":"q","revision":"1"}}]}'
        expect(modelRoutingState(text, identity).qualified).toBe(false)
        expect(
            modelRoutingState(text, identity, {
                policyRevision: '6',
                observationGeneration: 1,
                models: [{ ...identity, desiredResident: true, state: 'satisfied' }]
            }).qualified
        ).toBe(false)
        expect(
            modelRoutingState(text, identity, {
                policyRevision: '7',
                observationGeneration: 1,
                models: [{ ...identity, desiredResident: true, state: 'satisfied' }]
            }).qualified
        ).toBe(true)
    })
})
