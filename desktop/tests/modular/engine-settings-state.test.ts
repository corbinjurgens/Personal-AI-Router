// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { customEngine } from './engine-fixtures'
import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import {
    useEngineSettingsStore,
    engineSettingsKey,
    modelSettingsKey,
    watchEngineSettings
} from '@/ui/stores/engine-settings.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import type {
    EngineSettingsConfig,
    EngineSettingsSnapshot,
    EngineServingConfig
} from '@/shared/types/engine-settings'
import type { IEngineApi } from '@/ui/api/engine-api'
import type { InspectModelResult, ModelChange } from '@/shared/types/model-config'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { EngineSettingsSection } from '@/ui/components/BackendRow/EngineSettingsSection'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { redactSensitiveLogText } from '@/electron/redact-log'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import type { EngineStatusData, EngineType } from '@/shared/types/engines'

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))

const target = { nodeId: 'remote', engine: 'llamacpp' } as const
const modelLoads: NonNullable<EngineSettingsConfig['modelLoads']> = [
    {
        profileKey: 'exact/model',
        modelId: 'exact/model',
        requested: {
            context: { mode: 'fixed', tokens: 4096 },
            slots: 2,
            gpu: { mode: 'layers', layers: 24 },
            cacheTypeK: 'f16',
            cacheTypeV: 'f16',
            flashAttention: 'on',
            batchSize: 512,
            ubatchSize: 128,
            threads: 4,
            threadsBatch: 8,
            kvOffload: false,
            opOffload: true,
            cacheRAM: 512
        },
        resolved: {
            perSlotContext: 4096,
            totalContext: 8192,
            slots: 2,
            kvUnified: false,
            gpu: { mode: 'layers', layers: 24 },
            options: {
                context: { mode: 'fixed', tokens: 4096 },
                slots: 2,
                gpu: { mode: 'layers', layers: 24 },
                cacheTypeK: 'f16',
                cacheTypeV: 'f16',
                flashAttention: 'on',
                batchSize: 512,
                ubatchSize: 128,
                threads: 4,
                threadsBatch: 8,
                kvOffload: false,
                opOffload: true,
                cacheRAM: 512
            },
            artifacts: [
                {
                    role: 'model',
                    path: 'C:/models/model.gguf',
                    resolvedTarget: 'C:/cache/model.gguf',
                    size: '9007199254740993',
                    mtimeNs: '1793000000000000001',
                    sampleSha256: 'a'.repeat(64)
                }
            ],
            nativeBuildId: 'llama-b12020',
            sourceInputDigest: 'b'.repeat(64),
            presetDigest: 'c'.repeat(64)
        }
    },
    { profileKey: 'another/model', modelId: 'another/model', requested: { kvOffload: false } }
]
const baseline: EngineSettingsSnapshot = {
    ...target,
    revision: 1,
    appliedRevision: 1,
    epoch: 'one',
    sequence: 1,
    settings: {
        serverPort: 8081,
        proxyPort: 8080,
        launchText: 'managed serve',
        modelLoads: structuredClone(modelLoads)
    },
    effectiveServerPort: 8081,
    effectiveProxyPort: 8080,
    running: true,
    adopted: false,
    editable: true,
    reason: '',
    format: 'pair-arguments-v1',
    phase: 'idle',
    error: '',
    requestId: ''
}
const key = engineSettingsKey(target)
const store = useEngineSettingsStore
const getSettings = vi.fn()
const applySettings = vi.fn()

describe('profile-keyed settings drafts', () => {
    const inspect = vi.fn<IEngineApi['inspectModel']>()
    const preview = vi.fn<IEngineApi['previewSettings']>()
    const get = vi.fn<IEngineApi['getSettings']>()
    const apply = vi.fn<IEngineApi['applySettings']>()
    const local = { nodeId: 'local-profile-test', engine: 'llamacpp' } as const
    const inspection: InspectModelResult = {
        engine: 'llamacpp',
        modelId: 'artifact',
        status: 'resolved',
        estimate: { status: 'unknown' }
    }
    beforeEach(() => {
        inspect.mockReset().mockResolvedValue(inspection)
        preview.mockReset().mockResolvedValue({
            settings: baseline.settings,
            revision: 1,
            policyRevision: '7',
            errors: {},
            restart: false,
            rebind: false,
            previewDigest: 'a'.repeat(64)
        })
        get.mockReset().mockResolvedValue({
            ...baseline,
            ...local,
            policyRevision: '7',
            settings: {
                ...baseline.settings,
                modelLoads: [
                    {
                        profileKey: 'small',
                        modelId: 'artifact',
                        requested: { context: { mode: 'auto', floor: 4096 } }
                    }
                ]
            }
        })
        apply.mockReset().mockResolvedValue({ phase: 'applying', revision: 1 })
        Object.defineProperty(window, 'pairApi', {
            configurable: true,
            value: {
                engines: {
                    inspectModel: inspect,
                    previewSettings: preview,
                    getSettings: get,
                    applySettings: apply
                },
                nodePolicy: {
                    get: vi.fn().mockResolvedValue({
                        policy: '{}',
                        revision: '7',
                        availability: 'available'
                    })
                }
            }
        })
        store.setState({ entries: {}, modelDrafts: {} })
        useConnectionStore.setState({ connected: true, selfId: local.nodeId })
        useNodeConfigurationStore.setState({ policyDrafts: new Map(), management: new Map() })
    })
    it('keeps optional inspection identities and draft caches separate', async () => {
        await store.getState().inspectModel(local, 'artifact', 'small')
        await store.getState().inspectModel(local, 'artifact', 'large')
        await store.getState().inspectModel(local, 'artifact')
        expect(inspect.mock.calls).toEqual([
            [{ ...local, modelId: 'artifact', profileKey: 'small' }],
            [{ ...local, modelId: 'artifact', profileKey: 'large' }],
            [{ ...local, modelId: 'artifact' }]
        ])
        expect(Object.keys(store.getState().modelDrafts)).toEqual([
            'local-profile-test:llamacpp:small',
            'local-profile-test:llamacpp:large',
            'local-profile-test:llamacpp:'
        ])
        store.getState().editModel(modelSettingsKey(local, 'large'), {
            modelId: 'artifact',
            profileKey: 'large',
            expectedPolicyRevision: '7',
            policy: { displayName: 'Large' }
        })
        expect(
            store.getState().modelDrafts[modelSettingsKey(local, 'small')].change
        ).toBeUndefined()
        expect(store.getState().modelDrafts[modelSettingsKey(local)].change).toBeUndefined()
    })
    it.each([
        { name: 'new profile', policy: '{}' },
        {
            name: 'untyped profile',
            policy: JSON.stringify({
                profiles: [
                    {
                        key: 'fresh-profile',
                        displayName: 'Fresh',
                        engine: 'llamacpp',
                        model: 'artifact'
                    }
                ]
            })
        }
    ])(
        'inspects $name artifacts without an absent saved-key selector in review and apply',
        async ({ policy }) => {
            vi.mocked(window.pairApi.nodePolicy.get).mockResolvedValue({
                policy,
                revision: '7',
                activation: null,
                availability: 'available',
                recoveryRequired: false
            })
            inspect.mockImplementation(async request => {
                if (request.profileKey !== undefined) throw new Error('Profile key is not saved')
                return inspection
            })
            const change: ModelChange = {
                modelId: 'artifact',
                profileKey: 'fresh-profile',
                expectedPolicyRevision: '7',
                load: { set: { slots: 2 } },
                policy: { displayName: 'Fresh' }
            }
            await store.getState().reload(local)
            await store.getState().previewModel(local, change, 'new-profile-review')
            const draftKey = modelSettingsKey(local, 'fresh-profile')
            expect(store.getState().modelDrafts[draftKey].review?.request.modelChange).toEqual({
                ...change,
                previewDigest: 'a'.repeat(64)
            })
            await store.getState().applyModel(draftKey, 'new-profile-review')
            expect(inspect).toHaveBeenNthCalledWith(1, { ...local, modelId: 'artifact' })
            expect(inspect).toHaveBeenNthCalledWith(2, { ...local, modelId: 'artifact' })
            expect(apply).toHaveBeenCalledTimes(1)
            expect(apply.mock.calls[0][0].modelChange).toMatchObject(change)
            expect(store.getState().modelDrafts[modelSettingsKey(local)]).toBeUndefined()
        }
    )

    it('does not inherit a sibling Auto load when reviewing another key', async () => {
        const change: ModelChange = {
            modelId: 'artifact',
            profileKey: 'large',
            expectedPolicyRevision: '7',
            load: { set: { slots: 2 } }
        }
        await store.getState().reload(local)
        await store.getState().previewModel(local, change, 'review-large')
        expect(inspect).toHaveBeenCalledExactlyOnceWith({
            ...local,
            modelId: 'artifact'
        })
        expect(
            store.getState().modelDrafts[modelSettingsKey(local, 'large')].review?.request
                .modelChange
        ).toEqual({ ...change, previewDigest: 'a'.repeat(64) })
        expect(store.getState().modelDrafts[modelSettingsKey(local, 'small')]).toBeUndefined()
    })
    it('applies and reconciles only the selected profile after a lost reply', async () => {
        get.mockResolvedValue({
            ...baseline,
            ...local,
            policyRevision: '7',
            settings: {
                ...baseline.settings,
                modelLoads: [
                    {
                        profileKey: 'small',
                        modelId: 'artifact',
                        requested: { context: { mode: 'auto', floor: 4096 } }
                    },
                    {
                        profileKey: 'large',
                        modelId: 'artifact',
                        requested: { context: { mode: 'fixed', tokens: 8192 } }
                    }
                ]
            }
        })
        await store.getState().reload(local)
        const sibling: ModelChange = {
            modelId: 'artifact',
            profileKey: 'small',
            expectedPolicyRevision: '7',
            policy: { displayName: 'Small revised' }
        }
        store.getState().editModel(modelSettingsKey(local, 'small'), sibling)
        const change: ModelChange = {
            modelId: 'artifact',
            profileKey: 'large',
            expectedPolicyRevision: '7',
            policy: { displayName: 'Large revised', icon: 'code' }
        }
        await store.getState().previewModel(local, change, 'review-large')
        apply.mockImplementationOnce(async request => {
            if (!request.requestId) throw new Error('Missing apply request ID')
            get.mockResolvedValue({
                ...baseline,
                ...local,
                sequence: 2,
                policyRevision: '8',
                phase: 'succeeded',
                requestId: request.requestId
            })
            throw new Error('Lost reply')
        })
        await store.getState().applyModel(modelSettingsKey(local, 'large'), 'review-large')
        expect(apply).toHaveBeenCalledTimes(1)
        expect(apply.mock.calls[0]?.[0].modelChange).toEqual({
            ...change,
            previewDigest: 'a'.repeat(64)
        })
        expect(inspect.mock.calls).toEqual([
            [{ ...local, modelId: 'artifact', profileKey: 'large' }],
            [{ ...local, modelId: 'artifact', profileKey: 'large' }]
        ])
        expect(store.getState().modelDrafts[modelSettingsKey(local, 'large')]).toMatchObject({
            change: undefined,
            uncertain: false
        })
        expect(store.getState().modelDrafts[modelSettingsKey(local, 'small')].change).toEqual(
            sibling
        )
    })
    it('refuses apply when a fresh saved-load read changes the reviewed baseline', async () => {
        const change: ModelChange = {
            modelId: 'artifact',
            profileKey: 'large',
            expectedPolicyRevision: '7',
            load: { set: { slots: 2 } }
        }
        await store.getState().reload(local)
        await store.getState().previewModel(local, change, 'review-large')
        const draftKey = modelSettingsKey(local, 'large')
        expect(store.getState().modelDrafts[draftKey].review).toBeDefined()
        get.mockResolvedValue({
            ...baseline,
            ...local,
            sequence: 2,
            revision: 2,
            policyRevision: '7',
            settings: {
                ...baseline.settings,
                modelLoads: [{ profileKey: 'large', modelId: 'artifact', requested: { slots: 4 } }]
            }
        })
        await store.getState().applyModel(draftKey, 'review-large')
        expect(inspect).toHaveBeenCalledExactlyOnceWith({ ...local, modelId: 'artifact' })
        expect(apply).not.toHaveBeenCalled()
        expect(store.getState().modelDrafts[draftKey]).toMatchObject({
            change,
            review: undefined,
            message: 'Preview expired. Your draft is kept.'
        })
    })
    it('keeps the keyed draft without dispatch when the apply refresh is unavailable', async () => {
        const change: ModelChange = {
            modelId: 'artifact',
            profileKey: 'large',
            expectedPolicyRevision: '7',
            load: { set: { slots: 2 } }
        }
        await store.getState().reload(local)
        await store.getState().previewModel(local, change, 'review-large')
        const draftKey = modelSettingsKey(local, 'large')
        expect(store.getState().modelDrafts[draftKey].review).toBeDefined()
        get.mockRejectedValueOnce(new Error('Device offline'))
        await store.getState().applyModel(draftKey, 'review-large')
        expect(inspect).toHaveBeenCalledExactlyOnceWith({ ...local, modelId: 'artifact' })
        expect(apply).not.toHaveBeenCalled()
        expect(store.getState().modelDrafts[draftKey]).toMatchObject({
            change,
            review: undefined,
            message: 'Preview expired. Your draft is kept.'
        })
        expect(store.getState().entries[engineSettingsKey(local)].unavailable).toContain(
            'Device offline'
        )
    })
    it('retains newer inspection results when an older same-key read arrives', async () => {
        let finish: ((value: InspectModelResult) => void) | undefined
        inspect.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        const old = store.getState().inspectModel(local, 'artifact', 'large')
        inspect.mockResolvedValueOnce({ ...inspection, configurationKey: 'new-configuration' })
        await store.getState().inspectModel(local, 'artifact', 'large')
        if (!finish) throw new Error('Missing delayed inspection')
        finish({ ...inspection, configurationKey: 'old-configuration' })
        await old
        expect(
            store.getState().modelDrafts[modelSettingsKey(local, 'large')].inspection
                ?.configurationKey
        ).toBe('new-configuration')
    })
})

describe('running native build status', () => {
    const buildInfo = 'b12345-test-build'
    const buildId = `sha256:${'a'.repeat(64)}`
    const wire = {
        engine: 'llamacpp',
        installed: true,
        running: true,
        healthy: true,
        port: 8081,
        buildInfo,
        buildId
    }

    it('preserves local notification builds and clears omissions on stop', () => {
        const state = getModularBridgeState()
        state.setSelfId('build-local')
        state.applyEngineManagerStatus(wire)
        const status = () =>
            state
                .getEngineInitialState()
                .statuses.find(
                    value => value.nodeId === 'build-local' && value.engineType === 'llama-cpp'
                )
        expect(status()).toMatchObject({ buildInfo, buildId, processStatus: 'running' })
        state.beginLocalEngineOp('llama-cpp', 'stopping')
        expect(status()).toMatchObject({ buildInfo, buildId, processStatus: 'stopping' })
        state.applyEngineManagerStatus({ engine: 'llamacpp', installed: true, running: false })
        expect(status()?.buildInfo).toBeUndefined()
        expect(status()?.buildId).toBeUndefined()
    })

    it('preserves remote list builds and clears a replacement without them', () => {
        const state = getModularBridgeState()
        state.setSelfId('build-local')
        state.handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: {
                nodes: [
                    {
                        hostUuid: 'build-peer',
                        name: 'build-peer',
                        ipAddress: '192.0.2.60',
                        port: 14318,
                        trusted: true,
                        clustered: true
                    }
                ]
            }
        })
        state.applyRemoteEngineFacts('build-peer', { engines: [wire] })
        const status = () =>
            state
                .getEngineInitialState()
                .statuses.find(
                    value => value.nodeId === 'build-peer' && value.engineType === 'llama-cpp'
                )
        expect(status()).toMatchObject({ buildInfo, buildId })
        state.applyRemoteEngineFacts('build-peer', {
            engines: [
                {
                    ...wire,
                    buildInfo: 'replacement-build',
                    buildId: `sha256:${'b'.repeat(64)}`
                }
            ]
        })
        expect(status()).toMatchObject({
            buildInfo: 'replacement-build',
            buildId: `sha256:${'b'.repeat(64)}`
        })
        state.applyRemoteEngineFacts('build-peer', {
            engines: [{ engine: 'llamacpp', installed: true, running: false }]
        })
        expect(status()?.buildInfo).toBeUndefined()
        expect(status()?.buildId).toBeUndefined()
    })

    it('shows the owning node build in settings and its identity in the tooltip', async () => {
        const get = vi.fn<IEngineApi['getSettings']>().mockResolvedValue(baseline)
        Object.defineProperty(window, 'pairApi', {
            configurable: true,
            value: {
                engines: {
                    getSettings: get,
                    onSettingsChanged: () => () => {},
                    onSettingsDisconnected: () => () => {}
                }
            }
        })
        vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
        vi.stubGlobal('CSS', { supports: () => false })
        useConnectionStore.setState({ connected: true, selfId: 'remote' })
        useEngineStatusStore.setState({
            statusByNode: new Map([
                [
                    'remote',
                    new Map<EngineType, EngineStatusData>([
                        [
                            'llama-cpp',
                            {
                                engineType: 'llama-cpp',
                                nodeId: 'remote',
                                processStatus: 'running',
                                enginePort: 8081,
                                proxyPort: 8080,
                                buildInfo,
                                buildId
                            }
                        ]
                    ])
                ]
            ])
        })
        store.setState({ entries: {} })
        const container = document.createElement('div')
        document.body.appendChild(container)
        const root = createRoot(container)
        try {
            await act(async () =>
                root.render(
                    createElement(EngineSettingsSection, {
                        nodeId: 'remote',
                        engineType: 'llama-cpp',
                        disabled: false
                    })
                )
            )
            const label = Array.from(container.querySelectorAll('[tabindex="0"]')).find(
                element => element.textContent === 'llama.cpp b12345-test-build'
            )
            if (!(label instanceof HTMLElement)) throw new Error('Build label missing')
            const tooltipId = label.getAttribute('aria-describedby')
            if (!tooltipId) throw new Error('Build tooltip association missing')
            expect(document.getElementById(tooltipId)?.getAttribute('role')).toBe('tooltip')
            expect(document.getElementById(tooltipId)?.textContent).toBe(buildId)
            await act(async () => useEngineStatusStore.setState({ statusByNode: new Map() }))
            expect(container.textContent).not.toContain('llama.cpp b12345-test-build')
        } finally {
            await act(async () => root.unmount())
            container.remove()
            useEngineStatusStore.setState({ statusByNode: new Map() })
            vi.unstubAllGlobals()
            vi.restoreAllMocks()
        }
    })
})

const serving: EngineServingConfig = {
    version: 1,
    models: [
        {
            modelId: 'exact/model',
            maxOutputTokens: 128,
            memory: { hostRamBytes: 4096, vramBytes: 0 },
            artifacts: [
                {
                    role: 'model',
                    sha256: 'a'.repeat(64),
                    provenance: { kind: 'supplied-sha256' },
                    binding: {
                        sourcePath: 'fixture-link',
                        resolvedTarget: 'fixture-blob',
                        sizeBytes: 64,
                        mtimeNs: '1729000000000000001',
                        sampleSha256: 'b'.repeat(64)
                    }
                }
            ]
        }
    ]
}

describe('model load and serving preservation in the settings editor', () => {
    afterEach(() => {
        vi.unstubAllGlobals()
        vi.restoreAllMocks()
    })

    it.each([
        { name: 'port', label: 'Proxy port', value: '12500', field: 'proxyPort' },
        {
            name: 'launch arguments',
            label: 'Engine arguments',
            value: 'managed serve --keep',
            field: 'launchText'
        },
        {
            name: 'working-set limit',
            label: 'Working-set limit (MiB, Windows)',
            value: '8192',
            field: 'workingSetLimitMiB'
        },
        {
            name: 'cleared working-set limit',
            label: 'Working-set limit (MiB, Windows)',
            value: '',
            field: 'workingSetLimitMiB'
        }
    ])(
        'carries full model loads and serving through a $name edit and failed apply',
        async ({ label, value, field }) => {
            const snapshot: EngineSettingsSnapshot = {
                ...baseline,
                engine: 'llamacpp',
                workingSetLimitStatus: 'applied',
                settings: {
                    ...baseline.settings,
                    workingSetLimitMiB: 4096,
                    serving: structuredClone(serving)
                }
            }
            const key = engineSettingsKey(snapshot)
            const get = vi.fn<IEngineApi['getSettings']>().mockResolvedValue(snapshot)
            const preview = vi
                .fn<IEngineApi['previewSettings']>()
                .mockImplementation(async request => ({
                    settings: request.settings,
                    revision: 1,
                    errors: {},
                    restart: false,
                    rebind: false
                }))
            const apply = vi
                .fn<IEngineApi['applySettings']>()
                .mockRejectedValue(new Error('synthetic failure'))
            Object.defineProperty(window, 'pairApi', {
                configurable: true,
                value: {
                    engines: {
                        getSettings: get,
                        previewSettings: preview,
                        applySettings: apply,
                        onSettingsChanged: () => () => {},
                        onSettingsDisconnected: () => () => {}
                    }
                }
            })
            vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
            vi.stubGlobal('CSS', { supports: () => false })
            Object.defineProperty(HTMLDialogElement.prototype, 'showModal', {
                configurable: true,
                value: function (this: HTMLDialogElement) {
                    this.open = true
                }
            })
            Object.defineProperty(HTMLDialogElement.prototype, 'close', {
                configurable: true,
                value: function (this: HTMLDialogElement) {
                    this.open = false
                }
            })
            useConnectionStore.setState({ connected: true, selfId: 'remote' })
            store.setState({ entries: {} })
            usePendingActionsStore.setState({ pending: new Map() })
            const container = document.createElement('div')
            document.body.appendChild(container)
            const root = createRoot(container)
            try {
                await act(async () =>
                    root.render(
                        createElement(EngineSettingsSection, {
                            nodeId: 'remote',
                            engineType: 'llama-cpp',
                            disabled: false
                        })
                    )
                )
                const input = container.querySelector(`[aria-label="${label}"]`)
                expect(container.textContent).toContain('applied')
                const prototype =
                    field === 'launchText'
                        ? HTMLTextAreaElement.prototype
                        : HTMLInputElement.prototype
                const setter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set
                if (!input || !setter) throw new Error(`${label} input missing`)
                await act(async () => {
                    setter.call(input, value)
                    input.dispatchEvent(new Event('input', { bubbles: true }))
                })
                const button = container.querySelector('button[aria-label="Apply"]')
                if (!(button instanceof HTMLButtonElement)) throw new Error('Apply button missing')
                await act(async () => button.click())
                const expectedSettings = {
                    ...snapshot.settings,
                    [field]:
                        field === 'proxyPort'
                            ? 12500
                            : field === 'workingSetLimitMiB'
                              ? value === ''
                                  ? undefined
                                  : 8192
                              : value
                }
                expect(preview).toHaveBeenCalledWith(
                    expect.objectContaining({ settings: expectedSettings })
                )
                expect(apply).toHaveBeenCalledWith(
                    expect.objectContaining({ settings: expectedSettings })
                )
                const previewRequest = preview.mock.calls[0]?.[0]
                const applyRequest = apply.mock.calls[0]?.[0]
                if (!previewRequest || !applyRequest) throw new Error('Settings requests missing')
                expect(previewRequest.settings.modelLoads).toEqual(modelLoads)
                expect(applyRequest.settings.modelLoads).toEqual(modelLoads)
                expect(applyRequest.settings.workingSetLimitMiB).toBe(
                    field === 'workingSetLimitMiB' ? (value === '' ? undefined : 8192) : 4096
                )
                expect(applyRequest.settings.modelLoads).not.toBe(
                    store.getState().entries[key].snapshot?.settings.modelLoads
                )
                const resolved = applyRequest.settings.modelLoads?.[0].resolved
                if (!resolved) throw new Error('Applied fixture resolution missing')
                resolved.artifacts[0].path = 'changed by request consumer'
                resolved.options.cacheRAM = 1
                expect(store.getState().entries[key].snapshot?.settings.modelLoads).toEqual(
                    modelLoads
                )
                expect(store.getState().entries[key].snapshot?.settings.serving).toEqual(serving)
            } finally {
                await act(async () => root.unmount())
                container.remove()
            }
        }
    )

    it.each(['1e', '8191.5', '-1'])(
        'refuses malformed working-set limit %s without previewing',
        async value => {
            const snapshot: EngineSettingsSnapshot = {
                ...baseline,
                engine: 'llamacpp',
                settings: { ...baseline.settings, workingSetLimitMiB: 4096 }
            }
            const preview = vi.fn<IEngineApi['previewSettings']>()
            const apply = vi.fn<IEngineApi['applySettings']>()
            Object.defineProperty(window, 'pairApi', {
                configurable: true,
                value: {
                    engines: {
                        getSettings: vi.fn<IEngineApi['getSettings']>().mockResolvedValue(snapshot),
                        previewSettings: preview,
                        applySettings: apply,
                        onSettingsChanged: () => () => {},
                        onSettingsDisconnected: () => () => {}
                    }
                }
            })
            vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
            vi.stubGlobal('CSS', { supports: () => false })
            Object.defineProperty(HTMLDialogElement.prototype, 'showModal', {
                configurable: true,
                value: function (this: HTMLDialogElement) {
                    this.open = true
                }
            })
            useConnectionStore.setState({ connected: true, selfId: 'remote' })
            store.setState({ entries: {} })
            usePendingActionsStore.setState({ pending: new Map() })
            const container = document.createElement('div')
            document.body.appendChild(container)
            const root = createRoot(container)
            try {
                await act(async () =>
                    root.render(
                        createElement(EngineSettingsSection, {
                            nodeId: 'remote',
                            engineType: 'llama-cpp',
                            disabled: false
                        })
                    )
                )
                const input = container.querySelector(
                    '[aria-label="Working-set limit (MiB, Windows)"]'
                )
                const setter = Object.getOwnPropertyDescriptor(
                    HTMLInputElement.prototype,
                    'value'
                )?.set
                if (!input || !setter) throw new Error('Working-set input missing')
                await act(async () => {
                    setter.call(input, value)
                    input.dispatchEvent(new Event('input', { bubbles: true }))
                })
                const button = container.querySelector('button[aria-label="Apply"]')
                if (!(button instanceof HTMLButtonElement)) throw new Error('Apply button missing')
                await act(async () => button.click())
                expect(preview).not.toHaveBeenCalled()
                expect(apply).not.toHaveBeenCalled()
                expect(document.body.textContent).toContain(
                    'Enter the working-set limit as whole MiB, or leave it empty.'
                )
            } finally {
                await act(async () => root.unmount())
                container.remove()
            }
        }
    )

    it('owns nested serving and model load values received from a caller', () => {
        const snapshot: EngineSettingsSnapshot = {
            ...baseline,
            settings: {
                ...baseline.settings,
                serving: structuredClone(serving),
                modelLoads: structuredClone(modelLoads)
            }
        }
        store.setState({ entries: {} })
        store.getState().receive(snapshot)
        if (!snapshot.settings.serving) throw new Error('Missing fixture serving')
        snapshot.settings.serving.models[0].artifacts[0].provenance.kind = 'changed'
        const load = snapshot.settings.modelLoads?.[0]
        if (!load?.resolved) throw new Error('Missing fixture model resolution')
        load.requested.context = { mode: 'fixed', tokens: 8192 }
        load.resolved.artifacts[0].path = 'changed'
        expect(store.getState().entries[key].snapshot?.settings.serving).toEqual(serving)
        expect(store.getState().entries[key].snapshot?.settings.modelLoads).toEqual(modelLoads)
    })
})

describe('engine settings restart confirmation', () => {
    const preview = vi.fn<IEngineApi['previewSettings']>()
    const apply = vi.fn<IEngineApi['applySettings']>()
    let container: HTMLDivElement
    let root: ReturnType<typeof createRoot>

    function confirmationDialog(): HTMLDialogElement {
        const dialog = document.querySelector('dialog[open]')
        if (!(dialog instanceof HTMLDialogElement)) throw new Error('Confirmation dialog missing')
        return dialog
    }

    function buttonNamed(parent: ParentNode, label: string): HTMLButtonElement {
        const button = Array.from(parent.querySelectorAll('button')).find(
            candidate => candidate.textContent === label
        )
        if (!button) throw new Error(`${label} button missing`)
        return button
    }

    beforeEach(async () => {
        const get = vi.fn<IEngineApi['getSettings']>().mockResolvedValue(baseline)
        preview.mockImplementation(async request => ({
            settings: request.settings,
            revision: 1,
            errors: {},
            restart: true,
            rebind: false
        }))
        apply.mockResolvedValue({ revision: 2, phase: 'applying' })
        Object.defineProperty(window, 'pairApi', {
            configurable: true,
            value: {
                engines: {
                    getSettings: get,
                    previewSettings: preview,
                    applySettings: apply,
                    onSettingsChanged: () => () => {},
                    onSettingsDisconnected: () => () => {}
                }
            }
        })
        vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
        vi.stubGlobal('CSS', { supports: () => false })
        Object.defineProperty(HTMLDialogElement.prototype, 'showModal', {
            configurable: true,
            value: function (this: HTMLDialogElement) {
                this.open = true
            }
        })
        Object.defineProperty(HTMLDialogElement.prototype, 'close', {
            configurable: true,
            value: function (this: HTMLDialogElement) {
                this.open = false
            }
        })
        useConnectionStore.setState({ connected: true, selfId: 'remote' })
        store.setState({ entries: {} })
        usePendingActionsStore.setState({ pending: new Map() })
        container = document.createElement('div')
        document.body.appendChild(container)
        root = createRoot(container)
        await act(async () =>
            root.render(
                createElement(EngineSettingsSection, {
                    nodeId: 'remote',
                    engineType: 'llama-cpp',
                    disabled: false
                })
            )
        )
        const input = container.querySelector('input[aria-label="Proxy port"]')
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
        if (!input || !setter) throw new Error('Proxy port input missing')
        await act(async () => {
            setter.call(input, '12500')
            input.dispatchEvent(new Event('input', { bubbles: true }))
        })
        const button = container.querySelector('button[aria-label="Apply"]')
        if (!(button instanceof HTMLButtonElement)) throw new Error('Apply button missing')
        await act(async () => button.click())
    })

    afterEach(async () => {
        await act(async () => root.unmount())
        container.remove()
        usePendingActionsStore.setState({ pending: new Map() })
        vi.unstubAllGlobals()
        vi.restoreAllMocks()
    })

    it('discloses the bounded request wait and applies only after restart confirmation', async () => {
        const dialog = confirmationDialog()
        expect(dialog.textContent).toContain('Apply engine settings?')
        expect(dialog.textContent).toContain(
            'PAIR waits up to 60 seconds for active requests to finish, then restarts even if requests remain. This can interrupt those requests.'
        )
        expect(preview).toHaveBeenCalledWith(
            expect.objectContaining({ settings: { ...baseline.settings, proxyPort: 12500 } })
        )
        expect(apply).not.toHaveBeenCalled()
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'llama-cpp')).toBe(
            false
        )

        await act(async () => buttonNamed(dialog, 'Apply settings').click())

        expect(apply).toHaveBeenCalledTimes(1)
        expect(apply).toHaveBeenCalledWith(
            expect.objectContaining({
                ...target,
                expectedRevision: 1,
                requestId: expect.any(String),
                settings: { ...baseline.settings, proxyPort: 12500 }
            })
        )
        expect(dialog.open).toBe(false)
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'llama-cpp')).toBe(
            true
        )
    })

    it('cancels a restart without applying and leaves the draft editable', async () => {
        const dialog = confirmationDialog()
        expect(apply).not.toHaveBeenCalled()

        await act(async () => buttonNamed(dialog, 'Cancel').click())

        expect(dialog.open).toBe(false)
        expect(apply).not.toHaveBeenCalled()
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'llama-cpp')).toBe(
            false
        )
        const input = container.querySelector('input[aria-label="Proxy port"]')
        if (!(input instanceof HTMLInputElement)) throw new Error('Proxy port input missing')
        expect(input.value).toBe('12500')
        expect(input.disabled).toBe(false)
        const button = container.querySelector('button[aria-label="Apply"]')
        if (!(button instanceof HTMLButtonElement)) throw new Error('Apply button missing')
        expect(button.disabled).toBe(false)
        await act(async () => button.click())
        expect(confirmationDialog().textContent).toContain('Apply engine settings?')
        expect(apply).not.toHaveBeenCalled()
    })
})

describe('engine settings snapshot ordering and pending operations', () => {
    beforeEach(() => {
        store.setState({ entries: {} })
        vi.stubGlobal('window', { pairApi: { engines: { getSettings, applySettings } } })
        getSettings.mockResolvedValue(baseline)
        applySettings.mockResolvedValue({ revision: 2, phase: 'succeeded' })
    })
    afterEach(() => vi.unstubAllGlobals())

    it('ignores older frames and marks identical reconnect baselines available', () => {
        store.getState().receive({ ...baseline, sequence: 3, revision: 2 })
        store.getState().receive(baseline)
        expect(store.getState().entries[key].snapshot?.revision).toBe(2)
        store.getState().disconnect('remote')
        expect(store.getState().entries[key].unavailable).toBeTruthy()
        store.getState().receive({ ...baseline, sequence: 3, revision: 2 })
        expect(store.getState().entries[key].unavailable).toBeUndefined()
    })

    it('a newer push epoch beats a delayed initial read', async () => {
        let finish: (snapshot: EngineSettingsSnapshot) => void = () => {}
        getSettings.mockImplementationOnce(
            () =>
                new Promise<EngineSettingsSnapshot>(resolve => {
                    finish = resolve
                })
        )
        const loading = store.getState().reload(target)
        store.getState().receive({ ...baseline, epoch: 'two', revision: 9 })
        finish(baseline)
        await loading
        expect(store.getState().entries[key].snapshot?.epoch).toBe('two')
    })

    it('orders double-digit numeric sequences before rejecting a delayed frame', () => {
        store.getState().receive({ ...baseline, sequence: 9, revision: 2 })
        store.getState().receive({ ...baseline, sequence: 10, revision: 3 })
        store.getState().receive({ ...baseline, sequence: 2, revision: 1 })
        expect(store.getState().entries[key].snapshot?.sequence).toBe(10)
        expect(store.getState().entries[key].snapshot?.revision).toBe(3)
    })

    it('acknowledgement alone cannot fabricate an applied snapshot', async () => {
        store.getState().receive(baseline)
        await store.getState().apply({
            ...target,
            expectedRevision: 1,
            requestId: 'request',
            settings: { ...baseline.settings, launchText: 'different' }
        })
        expect(store.getState().entries[key].snapshot?.revision).toBe(1)
        // The settings push carries the applied snapshot; re-reading here would
        // only race it.
        expect(getSettings).not.toHaveBeenCalled()
    })

    it('a failed transport reads back and exposes unavailable status', async () => {
        applySettings.mockRejectedValueOnce(new Error('disconnected'))
        getSettings.mockRejectedValueOnce(new Error('offline'))
        await expect(
            store.getState().apply({
                ...target,
                expectedRevision: 1,
                requestId: 'request',
                settings: baseline.settings
            })
        ).rejects.toThrow('disconnected')
        expect(store.getState().entries[key].unavailable).toContain('offline')
    })

    it('shares subscriptions and releases them when the last editor closes', () => {
        const stop = vi.fn()
        const subscribe = vi.fn(() => stop)
        vi.stubGlobal('window', {
            pairApi: {
                engines: { onSettingsChanged: subscribe, onSettingsDisconnected: subscribe }
            }
        })
        const a = watchEngineSettings(),
            b = watchEngineSettings()
        expect(subscribe).toHaveBeenCalledTimes(2)
        a()
        expect(stop).not.toHaveBeenCalled()
        b()
        expect(stop).toHaveBeenCalledTimes(2)
    })

    it('closing an editor leaves other devices editable and resubscribes on reopen', () => {
        const receivers: Array<(snapshot: EngineSettingsSnapshot) => void> = []
        vi.stubGlobal('window', {
            pairApi: {
                engines: {
                    onSettingsChanged: (fn: (snapshot: EngineSettingsSnapshot) => void) => {
                        receivers.push(fn)
                        return () => {}
                    },
                    onSettingsDisconnected: () => () => {}
                }
            }
        })
        store.getState().receive(baseline)
        // An engine that starts installing unmounts its own editor; that must
        // not report every device in the cluster as disconnected.
        watchEngineSettings()()
        expect(store.getState().entries[key].unavailable).toBeUndefined()
        expect(store.getState().entries[key].snapshot?.revision).toBe(1)

        const reopened = watchEngineSettings()
        receivers.at(-1)?.({ ...baseline, sequence: 2, revision: 4 })
        expect(store.getState().entries[key].snapshot?.revision).toBe(4)
        expect(store.getState().entries[key].unavailable).toBeUndefined()
        reopened()
    })
})

describe('settings applies use the shared pending-actions cover', () => {
    let deliver: (snapshot: EngineSettingsSnapshot) => void = () => {}

    beforeEach(() => {
        const noop = (): void => {}
        vi.stubGlobal('window', {
            pairApi: {
                engines: {
                    onStateChanged: () => noop,
                    onProgress: () => noop,
                    onSettingsChanged: (fn: (snapshot: EngineSettingsSnapshot) => void) => {
                        deliver = fn
                        return noop
                    }
                },
                errors: { onUpdate: () => noop }
            }
        })
        usePendingActionsStore.getState().initialize()
    })
    afterEach(() => {
        usePendingActionsStore.getState().cleanup()
        vi.unstubAllGlobals()
    })

    it('holds from the send until the first snapshot, then yields to phase', () => {
        const pending = usePendingActionsStore.getState()
        expect(pending.isSettingsPending('remote', 'llama-cpp')).toBe(false)

        pending.beginSettings('remote', 'llama-cpp', 'current-request')
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'llama-cpp')).toBe(
            true
        )

        // Another engine on the same device is untouched.
        expect(usePendingActionsStore.getState().isSettingsPending('remote', customEngine)).toBe(
            false
        )

        deliver({ ...baseline, phase: 'applying', requestId: 'older-request' })
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'llama-cpp')).toBe(
            true
        )
        deliver({ ...baseline, phase: 'idle' })
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'llama-cpp')).toBe(
            true
        )
        deliver({ ...baseline, phase: 'applying', requestId: 'current-request' })
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'llama-cpp')).toBe(
            false
        )
    })

    it('releases the cover when validation refuses before sending', () => {
        const pending = usePendingActionsStore.getState()
        pending.beginSettings('remote', customEngine, 'current-request')
        pending.endSettings('remote', customEngine, 'older-request')
        expect(usePendingActionsStore.getState().isSettingsPending('remote', customEngine)).toBe(
            true
        )
        pending.endSettings('remote', customEngine, 'current-request')
        expect(usePendingActionsStore.getState().isSettingsPending('remote', customEngine)).toBe(
            false
        )
    })
})

describe('launch text redaction', () => {
    const frame = JSON.stringify({
        result: {
            settings: { launchText: 'exe --unknown private-value' },
            launch_args: ['private-value'],
            launch_env: ['VALUE=private-value']
        }
    })

    it('omits launch text, arguments, and environment from process logs', () => {
        expect(redactSensitiveLogText(frame)).not.toContain('private-value')
    })

    it('omits them from a frame truncated mid-value', () => {
        expect(redactSensitiveLogText(frame.slice(0, -1))).not.toContain('private-value')
        expect(
            redactSensitiveLogText('{"result":{"launch_env":["VALUE=private-value"')
        ).not.toContain('private-value')
    })

    it('still redacts the pairing PIN', () => {
        expect(redactSensitiveLogText('{"result":{"pin":"123456"}}')).not.toContain('123456')
        expect(redactSensitiveLogText('garbage {"pin": "123456"')).not.toContain('123456')
    })

    // A worker's own command line is what a support log is for, so the
    // redaction must not reach a generic args/env key.
    it('keeps worker spawn diagnostics intact', () => {
        const spawn = JSON.stringify({
            msg: 'spawning worker',
            bin: 'nvpair-node-scanner',
            args: ['--ipc', '--log-level=debug'],
            env: { NVPAIR_LOG: 'debug' }
        })
        expect(redactSensitiveLogText(spawn)).toBe(spawn)
        expect(redactSensitiveLogText('plain text with no json at all')).toBe(
            'plain text with no json at all'
        )
    })
})
