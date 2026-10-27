// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import {
    parseEngineSettings,
    parseEngineSettingsPreview,
    parseEngineSettingsReceipt,
    parseInspectModel
} from '@/electron/service-bridge/engine-settings'
import type { JsonValue } from '@/electron/service-bridge/json-rpc-client'
import type { ModelLoad, ResolvedLoad } from '@/shared/types/model-config'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi } from '@/ui/api/pair-api'

function resolvedLoad(): ResolvedLoad {
    return {
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
            sleepIdleSeconds: -1,
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
        nativeBuildId: 'llama-b11146',
        sourceInputDigest: 'b'.repeat(64),
        presetDigest: 'c'.repeat(64),
        resolverRevision: 'resolver-test-1',
        reviewedInputId: 'reviewed-target-input'
    }
}

function modelLoad(): ModelLoad {
    const resolved = resolvedLoad()
    return {
        modelId: 'owner/model:Q4_K_M',
        requested: {
            ...structuredClone(resolved.options),
            context: { mode: 'auto', floor: 4096, cap: 8192 },
            gpu: { mode: 'auto' }
        },
        resolved
    }
}

it('preserves a larger unified KV pool without changing ordinary slot arithmetic', () => {
    const wire = snapshot()
    const pool = resolvedLoad()
    wire.settings.modelLoads = [{ modelId: 'owner/model:Q4_K_M', requested: {}, resolved: pool }]
    pool.totalContext = 16384
    pool.kvUnified = true
    expect(parseEngineSettings(wire).settings.modelLoads?.[0].resolved?.totalContext).toBe(16384)
    pool.totalContext = 4096
    expect(() => parseEngineSettings(wire)).toThrow()
    pool.totalContext = 16384
    pool.kvUnified = false
    expect(() => parseEngineSettings(wire)).toThrow()
})

function snapshot() {
    return {
        nodeId: 'target',
        engine: 'llamacpp',
        revision: Number.MAX_SAFE_INTEGER,
        appliedRevision: 1,
        sequence: 10,
        epoch: 'authority-epoch',
        settings: { serverPort: 1234, proxyPort: 1235, launchText: '', modelLoads: [modelLoad()] },
        effectiveServerPort: 1234,
        effectiveProxyPort: 1235,
        running: false,
        adopted: false,
        editable: true,
        format: 'pair-arguments-v1',
        phase: 'idle'
    }
}

function inspection() {
    return {
        engine: 'llamacpp',
        nodeId: 'target',
        modelId: 'owner/model:Q4_K_M',
        status: 'resolved',
        source: 'configured-preset',
        assurance: 'sampled-local-binding',
        artifacts: [
            {
                role: 'model',
                identity: 'opaque-local-binding',
                size: '9007199254740993',
                assurance: 'sampled-local-binding'
            }
        ],
        facts: {
            status: 'partial',
            source: 'bounded-gguf',
            architecture: 'llama',
            contextTokens: 8192,
            layerCount: 32,
            templatePresent: true,
            templateDigest: 'd'.repeat(64),
            dimensions: { layers: '32', vocabulary: '128256' },
            unknowns: ['native workspace']
        },
        options: [
            {
                field: 'slots',
                label: 'Slots',
                kind: 'integer',
                minimum: 1,
                maximum: 2147483647,
                default: 1,
                defaultSource: 'finite adapter',
                supported: true,
                portable: true,
                owner: 'model-load',
                phase: 'B',
                identityEffect: 'allocation',
                estimateEffect: 'KV'
            }
        ],
        estimate: {
            status: 'partial',
            source: 'bounded-metadata-estimate',
            buildId: 'llama-b11146',
            timestamp: '2026-10-27T00:00:00Z',
            memorySource: 'target-memory',
            memoryTimestamp: '2026-10-27T00:00:00Z',
            hostBytes: '9007199254740993',
            deviceBytes: '0',
            domains: [
                {
                    id: 'uma',
                    kind: 'shared',
                    deviceId: 'target-device',
                    capacityBytes: '18446744073709551615',
                    freeBytes: '9007199254740993',
                    requiredBytes: '4096'
                }
            ],
            components: [
                {
                    name: 'weights',
                    domainId: 'uma',
                    status: 'estimated',
                    bytes: '4096',
                    reason: 'bounded tensor facts'
                }
            ],
            assumptions: ['shared memory counted once'],
            unknowns: ['native workspace']
        }
    }
}

function preview() {
    const load = modelLoad()
    const resolved = load.resolved
    if (!resolved) throw new Error('Fixture resolution missing')
    return {
        settings: snapshot().settings,
        revision: 1,
        restart: false,
        rebind: false,
        policyRevision: '9223372036854775807',
        previewDigest: 'e'.repeat(64),
        diff: [{ field: 'load.slots', before: 1, after: 2 }],
        requiredArtifacts: inspection().artifacts,
        estimate: inspection().estimate,
        impact: {
            restartScope: 'undetermined',
            affectedModels: [load.modelId],
            activeWorkPolicy: 'unresolved',
            unknowns: ['owner interruption choice pending']
        },
        auto: {
            status: 'proposed',
            requested: load.requested,
            resolved,
            estimate: inspection().estimate,
            memorySource: 'target-memory',
            memoryTimestamp: '2026-10-27T00:00:00Z',
            resolverRevision: 'resolver-test-1',
            reviewedInputId: 'reviewed-target-input',
            warnings: ['advisory estimate']
        }
    }
}

describe('model settings readback', () => {
    it('keeps numeric engine counters and exact decimal policy revisions', () => {
        const wire = snapshot()
        const result = parseEngineSettings({
            ...wire,
            coupledStage: 'pending-policy',
            policyRevision: '9223372036854775807',
            unresolvedError: 'Policy publication pending',
            effectiveSettings: wire.settings
        })
        expect(result.revision).toBe(Number.MAX_SAFE_INTEGER)
        expect(result.sequence).toBe(10)
        expect(result.settings.modelLoads).toEqual(wire.settings.modelLoads)
        expect(result.effectiveSettings).toEqual(wire.settings)
        expect(result.policyRevision).toBe('9223372036854775807')
        expect(result.coupledStage).toBe('pending-policy')
        expect(result.unresolvedError).toBe('Policy publication pending')
    })

    it('owns nested requested, resolved and effective settings', () => {
        const wire = snapshot()
        const result = parseEngineSettings({ ...wire, effectiveSettings: wire.settings })
        const original = structuredClone(wire.settings)
        wire.settings.modelLoads[0].requested.slots = 9
        const resolved = wire.settings.modelLoads[0].resolved
        if (!resolved) throw new Error('Fixture resolution missing')
        resolved.artifacts[0].path = 'changed'
        resolved.options.cacheRAM = 1
        expect(result.settings).toEqual(original)
        expect(result.effectiveSettings).toEqual(original)
        expect(result.settings).not.toBe(result.effectiveSettings)
    })

    it('keeps unresolved saved intent without manufacturing a resolution', () => {
        const wire = snapshot()
        const load = { modelId: 'owner/other:Q8_0', requested: { slots: 1 } }
        const result = parseEngineSettings({
            ...wire,
            settings: { ...wire.settings, modelLoads: [load] }
        })
        expect(result.settings.modelLoads).toEqual([load])
        expect(result.settings.modelLoads?.[0].resolved).toBeUndefined()
    })

    it('preserves inspection facts, descriptors and advisory byte precision', () => {
        const wire = inspection()
        const result = parseInspectModel(wire)
        expect(result).toEqual(wire)
        expect(result.estimate.domains?.[0].capacityBytes).toBe('18446744073709551615')
        expect(result.estimate.hostBytes).toBe('9007199254740993')
        wire.facts.dimensions.layers = '99'
        wire.estimate.domains[0].freeBytes = '0'
        expect(result.facts?.dimensions?.layers).toBe('32')
        expect(result.estimate.domains?.[0].freeBytes).toBe('9007199254740993')
    })

    it('preserves complete preview review data without filling unresolved impact', () => {
        const wire = preview()
        const result = parseEngineSettingsPreview(wire)
        expect(result).toEqual({ ...wire, errors: {} })
        expect(result.impact?.mayInterruptActive).toBeUndefined()
        expect(result.impact?.drainTimeoutSeconds).toBeUndefined()
        expect(result.impact?.staysOff).toBeUndefined()
        wire.diff[0].after = 3
        expect(result.diff?.[0].after).toBe(2)
    })

    it('retains unavailable inspection and Auto without fake facts', () => {
        const result = parseInspectModel({
            engine: 'ollama',
            modelId: 'model',
            status: 'unavailable',
            facts: { status: 'unavailable', unknowns: ['not implemented'] },
            estimate: { status: 'unavailable' }
        })
        expect(result.status).toBe('unavailable')
        expect(result.facts?.contextTokens).toBeUndefined()
        expect(result.estimate.hostBytes).toBeUndefined()
        const resultPreview = parseEngineSettingsPreview({
            settings: snapshot().settings,
            revision: 1,
            restart: false,
            rebind: false,
            auto: { status: 'unavailable', requested: {}, estimate: { status: 'unknown' } }
        })
        expect(resultPreview.auto?.status).toBe('unavailable')
        expect(resultPreview.auto?.resolved).toBeUndefined()
    })
})

describe('model contract validation', () => {
    it.each([
        ['string', '1'],
        ['unsafe integer', Number.MAX_SAFE_INTEGER + 1],
        ['negative', -1],
        ['fraction', 1.5],
        ['null', null]
    ])('rejects a %s engine counter', (_name, value) => {
        expect(() => parseEngineSettings({ ...snapshot(), revision: value })).toThrow()
        expect(() => parseEngineSettings({ ...snapshot(), appliedRevision: value })).toThrow()
        expect(() => parseEngineSettings({ ...snapshot(), sequence: value })).toThrow()
        expect(() => parseEngineSettingsPreview({ ...preview(), revision: value })).toThrow()
        expect(() => parseEngineSettingsReceipt({ revision: value, phase: 'applying' })).toThrow()
    })

    it.each(['01', '-1', '1.0', '9223372036854775808'])('rejects policy revision %s', value => {
        expect(() => parseEngineSettings({ ...snapshot(), policyRevision: value })).toThrow()
        expect(() => parseEngineSettingsPreview({ ...preview(), policyRevision: value })).toThrow()
    })

    const invalidRequested: Array<{ name: string; value: JsonValue }> = [
        { name: 'unknown field', value: { nativeArgs: '--escape' } },
        { name: 'null selected leaf', value: { slots: null } },
        { name: 'zero slots', value: { slots: 0 } },
        { name: 'int32 overflow', value: { threads: 2147483648 } },
        { name: 'fixed without tokens', value: { context: { mode: 'fixed' } } },
        {
            name: 'fixed with Auto bounds',
            value: { context: { mode: 'fixed', tokens: 4096, floor: 4096 } }
        },
        {
            name: 'reversed Auto range',
            value: { context: { mode: 'auto', floor: 8192, cap: 4096 } }
        },
        { name: 'unknown context mode', value: { context: { mode: 'native' } } },
        {
            name: 'allocation overflow',
            value: { context: { mode: 'fixed', tokens: 2147483647 }, slots: 2 }
        },
        { name: 'layers without count', value: { gpu: { mode: 'layers' } } },
        { name: 'CPU with layers', value: { gpu: { mode: 'cpu', layers: 0 } } },
        { name: 'unknown KV type', value: { cacheTypeK: 'q2' } },
        { name: 'unknown attention mode', value: { flashAttention: 'auto' } },
        { name: 'invalid retention', value: { sleepIdleSeconds: -2 } },
        { name: 'zero retention', value: { sleepIdleSeconds: 0 } },
        { name: 'ubatch exceeds batch', value: { batchSize: 128, ubatchSize: 512 } },
        { name: 'nonboolean offload', value: { kvOffload: 0 } }
    ]
    it.each(invalidRequested)('rejects $name at the load read boundary', ({ value }) => {
        const wire = snapshot()
        expect(() =>
            parseEngineSettings({
                ...wire,
                settings: { ...wire.settings, modelLoads: [{ modelId: 'model', requested: value }] }
            })
        ).toThrow()
    })

    const invalidResolved: Array<{ name: string; value: JsonValue }> = [
        { name: 'unpadded context', value: { ...resolvedLoad(), perSlotContext: 4097 } },
        { name: 'inconsistent total', value: { ...resolvedLoad(), totalContext: 4096 } },
        {
            name: 'nonfinite placement',
            value: { ...resolvedLoad(), gpu: { mode: 'native-default' } }
        },
        { name: 'bad digest', value: { ...resolvedLoad(), presetDigest: 'c' } },
        { name: 'missing model shards', value: { ...resolvedLoad(), artifacts: [] } },
        {
            name: 'duplicate bindings',
            value: {
                ...resolvedLoad(),
                artifacts: [...resolvedLoad().artifacts, ...resolvedLoad().artifacts]
            }
        },
        { name: 'incomplete review identity', value: { ...resolvedLoad(), reviewedInputId: '' } },
        {
            name: 'inconsistent explicit slots',
            value: { ...resolvedLoad(), options: { slots: 1 } }
        },
        {
            name: 'changed quality preference',
            value: { ...resolvedLoad(), options: { ...resolvedLoad().options, cacheTypeV: 'q4_0' } }
        }
    ]
    it.each(invalidResolved)('rejects $name in a resolution', ({ value }) => {
        const wire = snapshot()
        expect(() =>
            parseEngineSettings({
                ...wire,
                settings: { ...wire.settings, modelLoads: [{ ...modelLoad(), resolved: value }] }
            })
        ).toThrow()
    })

    it('rejects duplicated loads and reserved model aliases', () => {
        const wire = snapshot()
        for (const loads of [
            [modelLoad(), modelLoad()],
            [{ modelId: 'strong-api', requested: {} }]
        ]) {
            expect(() =>
                parseEngineSettings({ ...wire, settings: { ...wire.settings, modelLoads: loads } })
            ).toThrow()
        }
    })

    const invalidInspections: Array<{ name: string; value: JsonValue }> = [
        { name: 'fake ready status', value: { ...inspection(), status: 'ready' } },
        { name: 'missing resolved artifacts', value: { ...inspection(), artifacts: [] } },
        {
            name: 'negative dimension',
            value: { ...inspection(), facts: { status: 'partial', dimensions: { layers: '-1' } } }
        },
        {
            name: 'bad template digest',
            value: { ...inspection(), facts: { status: 'partial', templateDigest: 'a' } }
        },
        {
            name: 'unknown option kind',
            value: { ...inspection(), options: [{ ...inspection().options[0], kind: 'slider' }] }
        },
        {
            name: 'reversed bounds',
            value: {
                ...inspection(),
                options: [{ ...inspection().options[0], minimum: 4, maximum: 1 }]
            }
        },
        {
            name: 'unsupported without reason',
            value: { ...inspection(), options: [{ ...inspection().options[0], supported: false }] }
        },
        {
            name: 'default without source',
            value: { ...inspection(), options: [{ ...inspection().options[0], defaultSource: '' }] }
        },
        {
            name: 'unsafe bound',
            value: {
                ...inspection(),
                options: [{ ...inspection().options[0], maximum: Number.MAX_SAFE_INTEGER + 1 }]
            }
        },
        {
            name: 'numeric bytes',
            value: { ...inspection(), estimate: { status: 'partial', hostBytes: 4096 } }
        },
        {
            name: 'overflowing bytes',
            value: {
                ...inspection(),
                estimate: { status: 'partial', hostBytes: '18446744073709551616' }
            }
        },
        {
            name: 'noncanonical bytes',
            value: { ...inspection(), estimate: { status: 'partial', deviceBytes: '01' } }
        },
        {
            name: 'unknown estimate status',
            value: { ...inspection(), estimate: { status: 'fits' } }
        },
        {
            name: 'unknown pool kind',
            value: {
                ...inspection(),
                estimate: { status: 'partial', domains: [{ id: 'a', kind: 'gpu' }] }
            }
        },
        {
            name: 'missing component domain',
            value: {
                ...inspection(),
                estimate: {
                    status: 'partial',
                    components: [
                        { name: 'weights', domainId: 'missing', status: 'estimated', bytes: '1' }
                    ]
                }
            }
        },
        {
            name: 'duplicate descriptors',
            value: { ...inspection(), options: [...inspection().options, ...inspection().options] }
        },
        { name: 'unexpected native field', value: { ...inspection(), paths: ['/private/model'] } }
    ]
    it.each(invalidInspections)('rejects $name in inspection', ({ value }) => {
        expect(() => parseInspectModel(value)).toThrow()
    })

    it('rejects malformed impact and mismatched Auto review inputs', () => {
        const wire = preview()
        expect(() =>
            parseEngineSettingsPreview({
                ...wire,
                impact: { restartScope: 'undetermined', mayInterruptActive: 'false' }
            })
        ).toThrow()
        expect(() =>
            parseEngineSettingsPreview({
                ...wire,
                auto: { ...wire.auto, reviewedInputId: 'different-input' }
            })
        ).toThrow()
        expect(() => parseEngineSettingsPreview({ ...wire, previewDigest: 'invalid' })).toThrow()
    })

    it('bounds collection sizes and the complete payload', () => {
        expect(() =>
            parseInspectModel({
                ...inspection(),
                options: Array.from({ length: 257 }, () => inspection().options[0])
            })
        ).toThrow()
        expect(() =>
            parseInspectModel({ ...inspection(), reason: 'x'.repeat(64 * 1024) })
        ).toThrow()
        const wire = snapshot()
        expect(() =>
            parseEngineSettings({
                ...wire,
                settings: { ...wire.settings, modelLoads: Array.from({ length: 257 }, modelLoad) }
            })
        ).toThrow()
    })
})

describe('pairApi model inspection composition', () => {
    it('binds the actual engine API through the typed transport', async () => {
        const response = parseInspectModel(inspection())
        const invoke = vi.fn<ServiceTransport['invoke']>().mockResolvedValue(response)
        const transport: ServiceTransport = {
            connected: true,
            invoke,
            subscribePush: vi.fn<ServiceTransport['subscribePush']>(() => () => {}),
            onConnect: vi.fn<ServiceTransport['onConnect']>(() => () => {}),
            onDisconnect: vi.fn<ServiceTransport['onDisconnect']>(() => () => {}),
            onAuthFailure: vi.fn<ServiceTransport['onAuthFailure']>(() => () => {}),
            destroy: vi.fn<ServiceTransport['destroy']>()
        }
        const api = createPairApi(transport)
        const request = { engine: 'llamacpp', nodeId: 'target', modelId: 'owner/model:Q4_K_M' }
        await expect(api.engines.inspectModel(request)).resolves.toEqual(response)
        expect(invoke).toHaveBeenCalledExactlyOnceWith('engines:inspect-model', request)
    })
})
