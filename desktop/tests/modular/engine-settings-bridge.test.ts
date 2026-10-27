// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'

const mocks = vi.hoisted(() => ({
    state: {
        getSelfId: vi.fn(() => 'local-node'),
        getProxyPort: vi.fn(() => 11434),
        beginLocalEngineOp: vi.fn(),
        failLocalEngineOp: vi.fn()
    },
    supervisor: {
        hasProcess: vi.fn(() => true),
        callProcess: vi.fn<ReturnType<typeof getModularSupervisor>['callProcess']>(),
        callProxy: vi.fn(),
        sendProcess: vi.fn(),
        reportError: vi.fn()
    }
}))

vi.mock('@/electron/service-bridge/modular-supervisor', () => ({
    getModularSupervisor: () => mocks.supervisor
}))
vi.mock('@/electron/service-bridge/modular-state', () => ({
    getModularBridgeState: () => mocks.state,
    isProxyEngine: (value: string) => value === 'ollama' || value === 'lm-studio',
    isUpstreamUnreachableError: () => false,
    parseServiceErrors: () => []
}))
vi.mock('@/electron/service-bridge/model-catalog', () => ({ getEngineHubModels: vi.fn() }))

import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'
import { MODULAR_ENGINE_LIFECYCLE_CALL_TIMEOUT_MS } from '@/shared/constants/modular-runtime'
import type { EngineServingConfig, EngineSettingsRequest } from '@/shared/types/engine-settings'
import type { InspectModelRequest } from '@/shared/types/model-config'
import type { JsonValue } from '@/electron/service-bridge/json-rpc-client'
import { JsonRpcResponseError } from '@/electron/service-bridge/json-rpc-client'
import {
    parseEngineSettings,
    parseEngineSettingsPreview
} from '@/electron/service-bridge/engine-settings'

const serving = {
    version: 1,
    models: [
        {
            modelId: 'exact/model',
            maxOutputTokens: 128,
            memory: { hostRamBytes: 4096, vramBytes: 0 },
            expectedNativeSourceDigest: 'a'.repeat(64),
            artifacts: [
                {
                    role: 'model',
                    sha256: 'b'.repeat(64),
                    provenance: { kind: 'hf-lfs-sha256', reference: 'pinned metadata' },
                    binding: {
                        sourcePath: 'fixture-link',
                        resolvedTarget: 'fixture-blob',
                        sizeBytes: 64,
                        mtimeNs: '1729000000000000001',
                        sampleSha256: 'c'.repeat(64)
                    }
                }
            ]
        }
    ]
} satisfies EngineServingConfig

const baseline = {
    nodeId: 'local-node',
    engine: 'ollama',
    revision: 4,
    appliedRevision: 4,
    epoch: 'epoch',
    sequence: 1,
    phase: 'idle',
    format: 'pair-arguments-v1',
    settings: { serverPort: 1235, proxyPort: 11434, launchText: 'managed serve' },
    effectiveServerPort: 1235,
    effectiveProxyPort: 11434,
    running: true,
    adopted: false,
    editable: true
}

describe('unified settings bridge', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        mocks.supervisor.callProcess.mockImplementation(async (_name: string, method: string) =>
            method === 'engine:get-settings' ? baseline : { revision: 5, phase: 'succeeded' }
        )
    })

    it('relays model inspection to the broker', async () => {
        mocks.supervisor.callProcess.mockResolvedValue({
            engine: 'ollama',
            nodeId: 'remote',
            modelId: 'exact/model',
            status: 'unavailable',
            estimate: { status: 'unknown' }
        })
        const request: InspectModelRequest = {
            engine: 'ollama',
            nodeId: 'remote',
            modelId: 'exact/model'
        }
        const result = await handleServiceBridgeInvoke('engines:inspect-model', request)
        expect(mocks.supervisor.callProcess).toHaveBeenCalledWith(
            'broker',
            'engine:inspect-model',
            request
        )
        expect(result.status).toBe('unavailable')
    })

    it('refuses inspection readback for a different target identity', async () => {
        mocks.supervisor.callProcess.mockResolvedValue({
            engine: 'llamacpp',
            nodeId: 'other-node',
            modelId: 'exact/model',
            status: 'unavailable',
            estimate: { status: 'unavailable' }
        })
        await expect(
            handleServiceBridgeInvoke('engines:inspect-model', {
                engine: 'llamacpp',
                nodeId: 'remote',
                modelId: 'exact/model'
            })
        ).rejects.toThrow('target identity mismatch')
    })

    it.each([-32010, -32011, -32012, -32014])('preserves refusal code %s', async code => {
        const error = new JsonRpcResponseError(code, 'Target refused the request')
        mocks.supervisor.callProcess.mockRejectedValue(error)
        await expect(
            handleServiceBridgeInvoke('engines:inspect-model', {
                engine: 'llamacpp',
                nodeId: 'remote',
                modelId: 'exact/model'
            })
        ).rejects.toBe(error)
        await expect(
            handleServiceBridgeInvoke('engines:apply-settings', {
                nodeId: 'remote',
                engine: 'llamacpp',
                expectedRevision: 4,
                requestId: 'refused-request',
                settings: baseline.settings
            })
        ).rejects.toBe(error)
    })

    it('relays a selected model change and full baseline without aliasing the caller', async () => {
        const request: EngineSettingsRequest = {
            nodeId: 'remote',
            engine: 'llamacpp',
            expectedRevision: 4,
            requestId: 'reviewed-request',
            settings: {
                ...baseline.settings,
                modelLoads: [{ modelId: 'exact/model', requested: { slots: 2, kvOffload: false } }]
            },
            modelChange: {
                modelId: 'exact/model',
                expectedPolicyRevision: '9007199254740993',
                load: { set: { slots: 3 }, clear: ['threads'] },
                policy: { generation: { temperature: 0.5, top_k: null }, roles: { weak: false } },
                previewDigest: 'd'.repeat(64),
                supersedesRequestId: 'previous-request'
            }
        }
        await handleServiceBridgeInvoke('engines:apply-settings', request)
        expect(mocks.supervisor.callProcess).toHaveBeenCalledExactlyOnceWith(
            'broker',
            'engine:apply-settings',
            request,
            MODULAR_ENGINE_LIFECYCLE_CALL_TIMEOUT_MS
        )
        const sent = mocks.supervisor.callProcess.mock.calls[0]?.[2]
        if (!sent || typeof sent !== 'object' || Array.isArray(sent))
            throw new Error('Relayed settings request missing')
        expect(sent.settings).not.toBe(request.settings)
        expect(sent.modelChange).not.toBe(request.modelChange)
        expect(sent.settings).toEqual(request.settings)
        expect(sent.modelChange).toEqual(request.modelChange)
        expect(mocks.supervisor.callProxy).not.toHaveBeenCalled()
        expect(mocks.supervisor.sendProcess).not.toHaveBeenCalled()
    })

    it('keeps a manager-accepted blank provenance reference readable', () => {
        const [model] = serving.models
        const [artifact] = model.artifacts
        const blank = {
            ...serving,
            models: [
                {
                    ...model,
                    artifacts: [
                        { ...artifact, provenance: { ...artifact.provenance, reference: ' ' } }
                    ]
                }
            ]
        }
        expect(
            parseEngineSettings({ ...baseline, settings: { ...baseline.settings, serving: blank } })
                .settings.serving
        ).toEqual(blank)
    })

    it('preserves complete serving declarations in readback, preview and unrelated apply', async () => {
        const settings = { ...baseline.settings, serving }
        const parsed = parseEngineSettings({
            ...baseline,
            settings,
            serving: { status: 'unverified', reason: 'Native validation pending' }
        })
        expect(parsed.settings.serving).toEqual(serving)
        expect(parsed.serving).toEqual({
            status: 'unverified',
            reason: 'Native validation pending'
        })
        expect(
            parseEngineSettingsPreview({ settings, revision: 4, restart: false, rebind: false })
                .settings.serving
        ).toEqual(serving)
        await handleServiceBridgeInvoke('engines:apply-settings', {
            nodeId: 'local-node',
            engine: 'ollama',
            expectedRevision: 4,
            requestId: 'serving-edit',
            settings: { ...parsed.settings, proxyPort: 12000 }
        })
        expect(mocks.supervisor.callProcess).toHaveBeenCalledWith(
            'broker',
            'engine:apply-settings',
            expect.objectContaining({ settings: { ...settings, proxyPort: 12000 } }),
            MODULAR_ENGINE_LIFECYCLE_CALL_TIMEOUT_MS
        )
    })

    it('rejects malformed serving records instead of projecting them away', () => {
        const invalid = (value: JsonValue): void => {
            expect(() =>
                parseEngineSettings({
                    ...baseline,
                    settings: { ...baseline.settings, serving: value }
                })
            ).toThrow()
        }
        invalid({ ...serving, version: 2 })
        invalid({ ...serving, models: [serving.models[0], serving.models[0]] })
        invalid({ ...serving, models: [{ ...serving.models[0], memory: { hostRamBytes: 4096 } }] })
        invalid({
            ...serving,
            models: [{ ...serving.models[0], maxOutputTokens: Number.MAX_SAFE_INTEGER + 1 }]
        })
        invalid({
            ...serving,
            models: [
                {
                    ...serving.models[0],
                    artifacts: [{ ...serving.models[0].artifacts[0], sha256: 'bad' }]
                }
            ]
        })
        invalid({
            ...serving,
            models: [
                {
                    ...serving.models[0],
                    artifacts: [
                        {
                            ...serving.models[0].artifacts[0],
                            binding: {
                                ...serving.models[0].artifacts[0].binding,
                                mtimeNs: '9223372036854775808'
                            }
                        }
                    ]
                }
            ]
        })
    })

    // Applying settings can stop, rebind, and restart an engine, so it needs
    // the lifecycle deadline rather than the default request timeout.
    it('sends one node-owned apply with a lifecycle deadline', async () => {
        const receipt = await handleServiceBridgeInvoke('engines:apply-settings', {
            nodeId: 'local-node',
            engine: 'ollama',
            expectedRevision: 4,
            requestId: 'request',
            settings: { serverPort: 11434, proxyPort: 1235, launchText: 'managed serve' }
        })
        expect(receipt).toEqual({ revision: 5, phase: 'succeeded' })
        expect(mocks.supervisor.callProcess).toHaveBeenCalledWith(
            'broker',
            'engine:apply-settings',
            expect.objectContaining({
                engine: 'ollama',
                expectedRevision: 4,
                settings: { serverPort: 11434, proxyPort: 1235, launchText: 'managed serve' }
            }),
            MODULAR_ENGINE_LIFECYCLE_CALL_TIMEOUT_MS
        )
        expect(mocks.supervisor.callProxy).not.toHaveBeenCalled()
        expect(mocks.supervisor.callProcess.mock.calls.map(call => call[1])).toEqual([
            'engine:apply-settings'
        ])
    })

    it('addresses a remote node through the same broker relay', async () => {
        await handleServiceBridgeInvoke('engines:apply-settings', {
            nodeId: 'remote-node',
            engine: 'ollama',
            expectedRevision: 4,
            requestId: 'request',
            settings: { ...baseline.settings, proxyPort: 12000 }
        })
        expect(mocks.supervisor.callProcess).toHaveBeenCalledWith(
            'broker',
            'engine:apply-settings',
            expect.objectContaining({
                nodeId: 'remote-node',
                settings: { ...baseline.settings, proxyPort: 12000 }
            }),
            MODULAR_ENGINE_LIFECYCLE_CALL_TIMEOUT_MS
        )
    })

    it('validates full snapshots and reports unsupported older devices', async () => {
        mocks.supervisor.callProcess.mockResolvedValue({})
        await expect(
            handleServiceBridgeInvoke('engines:get-settings', {
                nodeId: 'remote',
                engine: 'ollama'
            })
        ).rejects.toThrow()
    })
})
