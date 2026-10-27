// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from 'vitest'
import type {
    EngineSettingsPreview,
    EngineSettingsSnapshot,
    EngineSettingsTarget
} from '@/shared/types/engine-settings'
import type { InspectModelResult, ModelChange } from '@/shared/types/model-config'
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
import {
    modelPolicyPatch,
    modelRoutingState,
    portableLoadPatch
} from '@/shared/utils/node-policy-model-copy'

const target: EngineSettingsTarget = { nodeId: 'self', engine: 'llamacpp' }
const key = modelSettingsKey(target, 'model')
const policy: NodePolicyDocument = {
    policy: '{"revision":"7","profiles":[{"name":"test","engine":"llamacpp","model":"model"}],"opaque":9007199254740993}',
    revision: '7',
    activation: null,
    availability: 'available',
    recoveryRequired: false
}
const change: ModelChange = {
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
            modelLoads: [{ modelId: 'other', requested: { slots: 3 } }]
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
    it('omits empty policy patches and sends only closed inspection keys before Apply', async () => {
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
                await secondStarted.promise
                first.resolve({ ...response, previewDigest: 'obsolete' })
                await pending
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
        const artifact = { role: 'model', identity: 'old artifact', size: '10', assurance: 'bound' }
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
        let finish: ((value: NodePolicyDocument) => void) | undefined
        window.pairApi.nodePolicy.get = vi
            .fn<typeof window.pairApi.nodePolicy.get>()
            .mockImplementation(
                () =>
                    new Promise<NodePolicyDocument>(resolve => {
                        finish = resolve
                    })
            )
        const applying = useEngineSettingsStore
            .getState()
            .applyModel(peerKey, 'same captured token')
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
                policy: policy.policy.replace('"name":"test"', '"localOnly":true,"name":"test"')
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

describe('selected policy and portable load patches', () => {
    const identity = { engine: 'llamacpp', model: 'model' }
    it.each(['ollama', 'custom-engine'])(
        'selects the raw profile after a preceding %s profile',
        engine => {
            const baseline = policy.policy.replace(
                '"profiles":[',
                `"profiles":[{"name":"other","engine":"${engine}","model":"other","opaque":9007199254740993},`
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
                temperature.value.replace('"name":"test"', '"name":"foreign"'),
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
        expect(modelRoutingState('{"servedModels":[]}', identity).served).toBe('excluded')
        expect(
            modelRoutingState(
                '{"engineBindings":[{"engine":"llamacpp","model":"model"}]}',
                identity
            ).binding
        ).toBe(true)
    })
    it('keeps residency unavailable for a qualification object without current target evidence', () => {
        const text =
            '{"revision":"7","profiles":[{"name":"q","engine":"llamacpp","model":"model","contextTokens":8192,"maxOutputTokens":1024,"maxConcurrent":1,"memoryBytes":1024,"qualification":{"id":"q","revision":"1"}}]}'
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
