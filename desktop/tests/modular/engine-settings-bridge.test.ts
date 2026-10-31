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
import type {
    Estimate,
    InspectModelRequest,
    InspectModelResult,
    MemoryPlan
} from '@/shared/types/model-config'
import type { JsonValue } from '@/electron/service-bridge/json-rpc-client'
import { JsonRpcResponseError } from '@/electron/service-bridge/json-rpc-client'
import {
    parseEngineSettings,
    parseEngineSettingsPreview,
    parseInspectModel
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

describe('working-set settings decoding', () => {
    it.each(['applied', 'not-running', 'unenforced-adopted'])(
        'preserves the limit and %s status',
        status => {
            const snapshot = parseEngineSettings({
                ...baseline,
                workingSetLimitStatus: status,
                settings: { ...baseline.settings, workingSetLimitMiB: 4096 }
            })
            expect(snapshot.settings.workingSetLimitMiB).toBe(4096)
            expect(snapshot.workingSetLimitStatus).toBe(status)
        }
    )
    it.each([1.5, -1, '4096'])('rejects an invalid limit %s', limit => {
        expect(() =>
            parseEngineSettings({
                ...baseline,
                settings: { ...baseline.settings, workingSetLimitMiB: limit }
            })
        ).toThrow()
    })
    it('keeps the target range error for a zero limit in a preview', () => {
        const preview = parseEngineSettingsPreview({
            settings: { ...baseline.settings, workingSetLimitMiB: 0 },
            revision: 4,
            restart: false,
            rebind: false,
            errors: { workingSetLimitMiB: 'Enter a working-set limit from 256 to 1048576 MiB.' }
        })
        expect(preview.settings.workingSetLimitMiB).toBe(0)
        expect(preview.errors).toEqual({
            workingSetLimitMiB: 'Enter a working-set limit from 256 to 1048576 MiB.'
        })
    })
    it('rejects an unknown enforcement status', () => {
        expect(() =>
            parseEngineSettings({ ...baseline, workingSetLimitStatus: 'unknown' })
        ).toThrow()
    })
})

function memoryPlanFixture(): MemoryPlan {
    const config: MemoryPlan['requested'] = {
        contextTokens: 8192,
        kvFormat: 'int8',
        batchSize: 1,
        microbatch: { mode: 'fixed', tokens: 1024 },
        parallelSlots: 1,
        devices: [{ backend: 'cuda', selector: 'cuda:0' }],
        placement: 'hybrid'
    }
    const names: Array<MemoryPlan['devices'][number]['groups'][number]['name']> = [
        'weights',
        'kv_cache',
        'compute_scratch',
        'output_buffers',
        'runtime_driver_overhead',
        'transient_peak'
    ]
    const resource = (
        kind: MemoryPlan['devices'][number]['kind'],
        total: string
    ): MemoryPlan['devices'][number] => ({
        kind,
        mapping: kind === 'device' ? 'unproven' : 'not-applicable',
        ...(kind === 'device' ? { nativeSelector: 'cuda:0' } : {}),
        mappingBasis: 'Synthetic capacity resource, no physical mapping',
        groups: names.map(name => ({
            name,
            quality: name === 'weights' ? 'exact' : 'unknown',
            ...(name === 'weights' ? { bytes: '0' } : {}),
            basis: 'Synthetic accounting group',
            additive: name !== 'transient_peak',
            ...(name === 'kv_cache' ? { combinedMembership: [...names.slice(1, 3)] } : {})
        })),
        combinedAllocations: [
            {
                members: ['kv_cache', 'compute_scratch'],
                bytes: '7',
                quality: 'estimated',
                basis: 'Synthetic overlapping view'
            }
        ],
        pricedTotalBytes: total,
        pricedTotalBasis: 'Synthetic once-counted total',
        requiredWithAllowancesBytes: total,
        assumptions: [
            { name: 'protected reserve', bytes: '0', basis: 'Synthetic protected amount' }
        ],
        callerAllowances: [{ name: 'asserted-extra', bytes: '0', provenance: 'caller-asserted' }],
        uncoveredCosts: ['runtime_driver_overhead', 'transient_peak']
    })
    return {
        identity: {
            engine: 'strata',
            provider: 'synthetic-provider',
            revision: 'fixture-revision',
            executableSHA256: 'a'.repeat(64),
            runtimeSourceSHA256: 'b'.repeat(64),
            artifactIdentitySHA256: 'c'.repeat(64),
            build: { version: 'fixture', dirty: false, basis: 'Synthetic embedded build summary' },
            modelId: 'test/model',
            assertedContentSHA256: 'd'.repeat(64),
            contentAssurance: 'Sampled identity, caller assertion unverified',
            cacheable: false,
            cacheableReason: 'Full model content is unverified'
        },
        requested: config,
        resolved: structuredClone(config),
        configurationComplete: true,
        pricingScope: 'resolved_configuration',
        pricingQualification: 'non-replayable-pricing-inputs',
        devices: [resource('device', '9007199254740993')],
        hostRAM: resource('host_ram', '0'),
        hostCommit: resource('host_commit', '18446744073709551615'),
        verdict: {
            status: 'not_evaluated',
            conditional: true,
            basis: 'Synthetic provider verdict, no live observation'
        },
        warnings: ['Synthetic kernel envelope is not replayable']
    }
}

function memoryPlanEstimateFixture() {
    return {
        status: 'partial',
        estimateSource: {
            kind: 'engine-report',
            provider: 'synthetic-provider',
            revision: 'fixture-revision',
            basis: 'Synthetic pricing',
            unpricedCosts: ['runtime_driver_overhead', 'transient_peak']
        },
        memoryPlan: memoryPlanFixture()
    } satisfies Estimate
}

function memoryPlanInspectionFixture() {
    return {
        engine: 'strata',
        modelId: 'test/model',
        status: 'unknown',
        estimate: memoryPlanEstimateFixture()
    } satisfies InspectModelResult
}

function memoryPlanWithBudget(): MemoryPlan {
    const plan = memoryPlanFixture()
    plan.verdict.status = 'indeterminate'
    plan.verdict.budgetBasis = 'before_runtime_initialization'
    for (const resource of [...plan.devices, plan.hostRAM, plan.hostCommit]) {
        resource.budgetBytes = resource.requiredWithAllowancesBytes
        resource.shortfallBytes = '0'
        resource.shortfallBasis = 'Synthetic lower bound'
    }
    return plan
}

function withoutField(value: JsonValue, field: string): JsonValue {
    if (value === null || typeof value !== 'object' || Array.isArray(value))
        throw new Error('Expected object fixture')
    const result = { ...value }
    delete result[field]
    return result
}

describe('unified settings bridge', () => {
    it('decodes bounded stated history with exact applied pools and unavailable-store reporting', () => {
        const key = 'a'.repeat(64)
        const profile = {
            configurationKey: key,
            kind: 'stated',
            recordedAt: '2026-10-30T00:00:00Z',
            hostBytes: '18446744073709551615',
            current: true
        }
        const fixture = {
            engine: 'llamacpp',
            modelId: 'model',
            status: 'unavailable',
            configurationKey: key,
            configurationPools: ['host'],
            estimate: { status: 'unknown' },
            memoryProfiles: [profile]
        } satisfies JsonValue
        expect(parseInspectModel(fixture).memoryProfiles).toEqual([profile])
        expect(
            parseInspectModel({ ...fixture, memoryProfiles: [], memoryProfilesUnavailable: true })
                .memoryProfilesUnavailable
        ).toBe(true)
        expect(() =>
            parseInspectModel({
                ...fixture,
                memoryProfiles: Array.from({ length: 9 }, () => profile)
            })
        ).toThrow()
        expect(() =>
            parseInspectModel({ ...fixture, memoryProfiles: [{ ...profile, hostBytes: '01' }] })
        ).toThrow()
        expect(() =>
            parseInspectModel({ ...fixture, memoryProfiles: [{ ...profile, deviceBytes: '1' }] })
        ).toThrow('pools')
        expect(() =>
            parseInspectModel({ ...fixture, memoryProfiles: [{ ...profile, current: false }] })
        ).toThrow('current')
    })

    it('relays stated set and clear through the broker without losing byte strings', async () => {
        const key = 'a'.repeat(64)
        const response = {
            engine: 'llamacpp',
            nodeId: 'peer',
            modelId: 'model',
            status: 'unavailable',
            estimate: { status: 'unknown' }
        } satisfies JsonValue
        mocks.supervisor.callProcess.mockResolvedValue(response)
        const request = {
            engine: 'llamacpp',
            nodeId: 'peer',
            modelId: 'model',
            configurationKey: key,
            hostBytes: '9007199254740993'
        }
        await handleServiceBridgeInvoke('engines:set-stated-memory', request)
        expect(mocks.supervisor.callProcess).toHaveBeenLastCalledWith(
            'broker',
            'engine:set-stated-memory',
            request
        )
        const clear = {
            engine: request.engine,
            nodeId: request.nodeId,
            modelId: request.modelId,
            configurationKey: key
        }
        await handleServiceBridgeInvoke('engines:clear-stated-memory', clear)
        expect(mocks.supervisor.callProcess).toHaveBeenLastCalledWith(
            'broker',
            'engine:clear-stated-memory',
            clear
        )
    })
    it('decodes configuration scope independently of a domainless estimate and captured key', () => {
        const key = 'a'.repeat(64)
        const result = parseInspectModel({
            ...memoryPlanInspectionFixture(),
            configurationKey: key,
            configurationPools: ['host', 'device'],
            estimate: { status: 'unknown' },
            loadObservation: { status: 'unknown', current: false, configurationKey: key }
        })
        expect(result.configurationKey).toBe(key)
        expect(result.configurationPools).toEqual(['host', 'device'])
        expect(result.loadObservation?.configurationKey).toBe(key)
        expect(() =>
            parseInspectModel({
                ...memoryPlanInspectionFixture(),
                configurationKey: key,
                configurationKeyReason: 'not-finite',
                configurationPools: ['host']
            })
        ).toThrow('identity scope')
        expect(() =>
            parseInspectModel({
                ...memoryPlanInspectionFixture(),
                configurationPools: ['device', 'host']
            })
        ).toThrow('configuration pools')
    })
    it('preserves exact component and domain ranges', () => {
        const parsed = parseInspectModel({
            ...memoryPlanInspectionFixture(),
            estimate: {
                status: 'partial',
                domains: [
                    {
                        id: 'ram',
                        kind: 'host',
                        requiredBytes: '9007199254740993',
                        requiredLowBytes: '0',
                        requiredHighBytes: '18446744073709551615'
                    }
                ],
                components: [
                    {
                        name: 'weights',
                        domainId: 'ram',
                        status: 'partial',
                        bytes: '9007199254740993',
                        lowBytes: '0',
                        highBytes: '18446744073709551615',
                        basis: 'heuristic',
                        reason: 'unpriced residual'
                    }
                ]
            }
        })
        expect(parsed.estimate.components?.[0].bytes).toBe('9007199254740993')
        expect(parsed.estimate.domains?.[0].requiredHighBytes).toBe('18446744073709551615')
    })
    it.each<[JsonValue]>([
        [{ lowBytes: null }],
        [{ highBytes: null }],
        [{ lowBytes: '101' }],
        [{ highBytes: '99' }],
        [{ highBytes: '0100' }],
        [{ highBytes: '18446744073709551616' }],
        [{ basis: 'guess' }],
        [{ bytes: 100 }],
        [{ status: 'unknown' }],
        [{ status: 'partial' }]
    ])('rejects malformed ranged components %j', patch => {
        if (!patch || typeof patch !== 'object' || Array.isArray(patch))
            throw new Error('Invalid fixture')
        expect(() =>
            parseInspectModel({
                ...memoryPlanInspectionFixture(),
                estimate: {
                    status: 'partial',
                    components: [
                        {
                            name: 'weights',
                            status: 'estimated',
                            bytes: '100',
                            lowBytes: '100',
                            highBytes: '100',
                            basis: 'derived',
                            ...patch
                        }
                    ]
                }
            })
        ).toThrow()
    })
    beforeEach(() => {
        vi.clearAllMocks()
        mocks.supervisor.callProcess.mockImplementation(async (_name: string, method: string) =>
            method === 'engine:get-settings' ? baseline : { revision: 5, phase: 'succeeded' }
        )
    })

    it('preserves advisory estimate provenance in inspection', async () => {
        mocks.supervisor.callProcess.mockResolvedValue({
            engine: 'ollama',
            modelId: 'exact/model',
            status: 'unavailable',
            estimate: {
                status: 'partial',
                estimateSource: {
                    kind: 'engine-report',
                    provider: 'fixture',
                    revision: '1',
                    basis: 'synthetic',
                    unpricedCosts: ['transient peak']
                }
            }
        })
        const result = await handleServiceBridgeInvoke('engines:inspect-model', {
            engine: 'ollama',
            modelId: 'exact/model'
        })
        expect(result).toMatchObject({
            estimate: { status: 'partial', estimateSource: { unpricedCosts: ['transient peak'] } }
        })
    })

    it('preserves exact memory plan bytes, unknown groups and provider verdict through inspection', async () => {
        const original = memoryPlanInspectionFixture()
        mocks.supervisor.callProcess.mockResolvedValue(original)
        const result = await handleServiceBridgeInvoke('engines:inspect-model', {
            engine: 'strata',
            modelId: 'test/model'
        })
        expect(result.estimate).toEqual(original.estimate)
        expect(result.estimate.memoryPlan?.devices[0].pricedTotalBytes).toBe('9007199254740993')
        expect(result.estimate.memoryPlan?.hostCommit.pricedTotalBytes).toBe('18446744073709551615')
        expect(result.estimate.memoryPlan?.devices[0].groups[0].bytes).toBe('0')
        expect(result.estimate.memoryPlan?.devices[0].groups[1].bytes).toBeUndefined()
        expect(result.estimate.status).toBe('partial')
        expect(result.estimate.memoryPlan?.verdict.status).toBe('not_evaluated')
    })

    it('retains a memory plan across selected figure overrides and clearing', () => {
        const original = memoryPlanInspectionFixture()
        for (const kind of ['engine-report', 'user-stated', 'measured', 'engine-report']) {
            const inspection = {
                ...original,
                estimate: {
                    ...original.estimate,
                    estimateSource: {
                        ...original.estimate.estimateSource,
                        kind,
                        ...(kind === 'engine-report'
                            ? {}
                            : { provider: 'selected-figure', revision: 'recorded-at' })
                    }
                }
            }
            const parsed = parseInspectModel(inspection)
            expect(parsed.estimate.estimateSource?.kind).toBe(kind)
            expect(parsed.estimate.memoryPlan).toEqual(original.estimate.memoryPlan)
        }
    })

    it('detaches nested memory plans from their transport input', () => {
        const original = memoryPlanInspectionFixture()
        const before = structuredClone(original)
        const result = parseInspectModel(original)
        const plan = result.estimate.memoryPlan
        if (!plan) throw new Error('Decoded memory plan missing')
        plan.identity.build.dirty = true
        plan.requested.devices[0].selector = 'changed'
        if (plan.resolved?.microbatch) plan.resolved.microbatch.tokens = 512
        const members = plan.devices[0].combinedAllocations?.[0].members
        if (!members) throw new Error('Decoded combined allocation missing')
        members[0] = 'weights'
        const amounts = plan.devices[0].callerAllowances
        if (!amounts) throw new Error('Decoded caller allowances missing')
        amounts[0].name = 'changed'
        const costs = plan.hostRAM.uncoveredCosts
        if (!costs) throw new Error('Decoded uncovered costs missing')
        costs[0] = 'changed'
        expect(original).toEqual(before)
    })

    it.each<NonNullable<Estimate['estimateSource']>['kind']>([
        'engine-report',
        'pair-estimate',
        'measured',
        'user-stated'
    ])('retains %s source and omissions in inspection and Auto estimates', kind => {
        const estimate: Estimate = {
            ...memoryPlanEstimateFixture(),
            ...(kind === 'engine-report' ? {} : { memoryPlan: undefined }),
            estimateSource: { ...memoryPlanEstimateFixture().estimateSource, kind }
        }
        const inspection = parseInspectModel({ ...memoryPlanInspectionFixture(), estimate })
        const preview = parseEngineSettingsPreview({
            settings: baseline.settings,
            revision: 4,
            restart: false,
            rebind: false,
            estimate,
            auto: { status: 'unavailable', requested: {}, estimate }
        })
        expect(inspection.estimate).toEqual(estimate)
        expect(preview.estimate).toEqual(estimate)
        expect(preview.auto?.estimate).toEqual(estimate)
    })

    it('preserves minimum-layout and automatic resolution scope without replay approval', () => {
        const minimum = memoryPlanFixture()
        minimum.configurationComplete = false
        minimum.pricingScope = 'minimum_supported_layout'
        delete minimum.resolved
        const minimumResult = parseInspectModel({
            ...memoryPlanInspectionFixture(),
            estimate: { ...memoryPlanEstimateFixture(), memoryPlan: minimum }
        })
        expect(minimumResult.estimate.memoryPlan).toEqual(minimum)
        const automatic = memoryPlanFixture()
        automatic.requested.kvFormat = 'q8_0'
        automatic.requested.microbatch = { mode: 'auto', limit: 8192 }
        expect(
            parseInspectModel({
                ...memoryPlanInspectionFixture(),
                estimate: { ...memoryPlanEstimateFixture(), memoryPlan: automatic }
            }).estimate.memoryPlan
        ).toEqual(automatic)
    })

    it('preserves a policy-only shortfall using the greater of allocation and selection capacity', () => {
        const plan = memoryPlanWithBudget()
        const device = plan.devices[0]
        device.pricedTotalBytes = '100'
        device.minimumSelectionCapacityBytes = '200'
        device.minimumSelectionCapacityBasis = 'Synthetic selection threshold'
        device.callerAllowances = [
            { name: 'extra-runtime', bytes: '10', provenance: 'caller-asserted' }
        ]
        device.requiredWithAllowancesBytes = '210'
        device.budgetBytes = '205'
        device.shortfallBytes = '5'
        plan.verdict.status = 'does_not_fit'
        const inspection = {
            ...memoryPlanInspectionFixture(),
            estimate: { ...memoryPlanEstimateFixture(), memoryPlan: plan }
        }
        expect(parseInspectModel(inspection).estimate.memoryPlan).toEqual(plan)
        device.requiredWithAllowancesBytes = '310'
        device.shortfallBytes = '105'
        expect(() => parseInspectModel(inspection)).toThrow('required total')
    })

    it('retains conditional provider fit with unknown category partitions', () => {
        const plan = memoryPlanWithBudget()
        plan.verdict.status = 'fits_within_allowances'
        for (const resource of [...plan.devices, plan.hostRAM, plan.hostCommit])
            delete resource.uncoveredCosts
        const inspection = {
            ...memoryPlanInspectionFixture(),
            estimate: { ...memoryPlanEstimateFixture(), memoryPlan: plan }
        }
        const result = parseInspectModel(inspection)
        expect(result.estimate.memoryPlan?.verdict).toEqual(plan.verdict)
        expect(result.estimate.status).toBe('partial')
        expect(result.estimate.capacityFit).toBeUndefined()
        expect(result.estimate.memoryPlan?.devices[0].groups[1].quality).toBe('unknown')
        plan.verdict.conditional = false
        expect(() => parseInspectModel(inspection)).toThrow('verdict contradicts')
    })

    it.each<{ name: string; change: (plan: MemoryPlan) => JsonValue }>([
        {
            name: 'private extension',
            change: plan => ({ ...plan, extensions: { strata: { argv: ['private'] } } })
        },
        {
            name: 'private resource path',
            change: plan => ({ ...plan, devices: [{ ...plan.devices[0], path: '/private/model' }] })
        },
        {
            name: 'numeric byte token',
            change: plan => ({ ...plan, hostRAM: { ...plan.hostRAM, pricedTotalBytes: 0 } })
        },
        {
            name: 'null unknown bytes',
            change: plan => ({
                ...plan,
                devices: [
                    {
                        ...plan.devices[0],
                        groups: plan.devices[0].groups.map(group => ({ ...group, bytes: null }))
                    }
                ]
            })
        },
        {
            name: 'noncanonical bytes',
            change: plan => ({ ...plan, hostRAM: { ...plan.hostRAM, allocatedBytes: '01' } })
        },
        {
            name: 'overflow bytes',
            change: plan => ({
                ...plan,
                hostRAM: { ...plan.hostRAM, allocatedBytes: '18446744073709551616' }
            })
        },
        {
            name: 'invalid integer',
            change: plan => ({ ...plan, requested: { ...plan.requested, contextTokens: 1.5 } })
        },
        {
            name: 'duplicate device',
            change: plan => ({
                ...plan,
                devices: [plan.devices[0], plan.devices[0]],
                requested: {
                    ...plan.requested,
                    devices: [...plan.requested.devices, ...plan.requested.devices]
                }
            })
        },
        {
            name: 'duplicate group',
            change: plan => ({
                ...plan,
                hostRAM: {
                    ...plan.hostRAM,
                    groups: plan.hostRAM.groups.map(group => ({ ...group, name: 'weights' }))
                }
            })
        },
        {
            name: 'unknown group',
            change: plan => ({
                ...plan,
                hostRAM: {
                    ...plan.hostRAM,
                    groups: [
                        { ...plan.hostRAM.groups[0], name: 'native_component' },
                        ...plan.hostRAM.groups.slice(1)
                    ]
                }
            })
        },
        {
            name: 'invalid quality',
            change: plan => ({
                ...plan,
                hostRAM: {
                    ...plan.hostRAM,
                    groups: [
                        { ...plan.hostRAM.groups[0], quality: 'measured' },
                        ...plan.hostRAM.groups.slice(1)
                    ]
                }
            })
        },
        {
            name: 'unknown category has zero bytes',
            change: plan => ({
                ...plan,
                hostRAM: {
                    ...plan.hostRAM,
                    groups: plan.hostRAM.groups.map(group => ({ ...group, bytes: '0' }))
                }
            })
        },
        {
            name: 'additive transient peak',
            change: plan => ({
                ...plan,
                hostRAM: {
                    ...plan.hostRAM,
                    groups: plan.hostRAM.groups.map(group => ({ ...group, additive: true }))
                }
            })
        },
        {
            name: 'invalid membership',
            change: plan => ({
                ...plan,
                hostRAM: {
                    ...plan.hostRAM,
                    combinedAllocations: [
                        { members: ['native_component'], quality: 'unknown', basis: 'synthetic' }
                    ]
                }
            })
        },
        {
            name: 'wrong pricing scope',
            change: plan => ({ ...plan, pricingScope: 'minimum_supported_layout' })
        },
        { name: 'null resolved configuration', change: plan => ({ ...plan, resolved: null }) },
        {
            name: 'changed exact context',
            change: plan => ({ ...plan, resolved: { ...plan.requested, contextTokens: 4096 } })
        },
        {
            name: 'unverified content made cacheable',
            change: plan => ({ ...plan, identity: { ...plan.identity, cacheable: true } })
        },
        {
            name: 'unproved physical mapping',
            change: plan => ({
                ...plan,
                devices: [{ ...plan.devices[0], physicalId: 'GPU-fixture' }]
            })
        },
        {
            name: 'allowance overflow',
            change: plan => ({
                ...plan,
                hostCommit: {
                    ...plan.hostCommit,
                    callerAllowances: [{ name: 'extra', bytes: '1', provenance: 'caller-asserted' }]
                }
            })
        },
        {
            name: 'missing budget',
            change: plan => ({
                ...plan,
                verdict: {
                    ...plan.verdict,
                    status: 'indeterminate',
                    budgetBasis: 'before_runtime_initialization'
                }
            })
        },
        {
            name: 'missing complete flag',
            change: plan => withoutField(plan, 'configurationComplete')
        },
        {
            name: 'missing cacheable flag',
            change: plan => ({ ...plan, identity: withoutField(plan.identity, 'cacheable') })
        },
        {
            name: 'missing conditional flag',
            change: plan => ({ ...plan, verdict: withoutField(plan.verdict, 'conditional') })
        },
        {
            name: 'missing additive flag',
            change: plan => ({
                ...plan,
                hostRAM: {
                    ...plan.hostRAM,
                    groups: plan.hostRAM.groups.map(group => withoutField(group, 'additive'))
                }
            })
        },
        { name: 'oversized warning', change: plan => ({ ...plan, warnings: ['x'.repeat(1025)] }) },
        {
            name: 'oversized collection',
            change: plan => ({
                ...plan,
                warnings: Array.from({ length: 257 }, (_, index) => `warning ${index}`)
            })
        }
    ])('rejects malformed memory plan: $name', ({ change }) => {
        expect(() =>
            parseInspectModel({
                ...memoryPlanInspectionFixture(),
                estimate: {
                    ...memoryPlanEstimateFixture(),
                    memoryPlan: change(memoryPlanFixture())
                }
            })
        ).toThrow()
    })

    it('rejects plan source conflicts, identity mismatches and completeness relabeling', () => {
        const original = memoryPlanInspectionFixture()
        expect(() => parseInspectModel({ ...original, modelId: 'other/model' })).toThrow(
            'identity mismatch'
        )
        expect(() =>
            parseInspectModel({
                ...original,
                estimate: { ...original.estimate, status: 'estimated' }
            })
        ).toThrow('provenance or status')
        expect(() =>
            parseInspectModel({
                ...original,
                estimate: {
                    ...original.estimate,
                    estimateSource: { ...original.estimate.estimateSource, kind: 'pair-estimate' }
                }
            })
        ).toThrow('provenance or status')
        expect(() =>
            parseInspectModel({
                ...original,
                estimate: {
                    ...original.estimate,
                    components: [
                        {
                            name: 'weights',
                            status: 'estimated',
                            bytes: '0',
                            lowBytes: '0',
                            highBytes: '0',
                            basis: 'derived'
                        }
                    ]
                }
            })
        ).toThrow('provenance or status')
    })

    it('checks the full response ceiling without truncating a plan', () => {
        const original = memoryPlanInspectionFixture()
        original.estimate.memoryPlan.warnings = Array.from(
            { length: 80 },
            (_, index) => `${String(index).padStart(3, '0')}${'x'.repeat(997)}`
        )
        expect(() => parseInspectModel(original)).toThrow('64 KiB')
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
