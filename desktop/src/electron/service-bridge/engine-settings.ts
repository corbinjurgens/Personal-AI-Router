// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    EngineSettingsConfig,
    EngineSettingsSnapshot,
    EngineSettingsPreview,
    EngineSettingsReceipt,
    EngineSettingsTarget
} from '@/shared/types/engine-settings'
import type { EngineServingConfig, EngineServingReadback } from '@/shared/types/engine-settings'
import type {
    ArtifactSummary,
    AutoProposal,
    Estimate,
    InspectModelResult,
    MemoryPlan,
    ModelImpact,
    ModelLoad,
    MemoryProfile,
    RequestedLoad,
    ResolvedLoad
} from '@/shared/types/model-config'
import { isValidRoutingEngineId } from '@/shared/utils/engines'
import type { JsonObject, JsonValue } from './json-rpc-client'
import { proxyEngineFromManagerId } from './proxy-engines'

type LoadBufferObservation = NonNullable<InspectModelResult['loadObservation']>

function object(value: JsonValue | undefined): JsonObject {
    if (!value || typeof value !== 'object' || Array.isArray(value))
        throw new Error('This device does not support engine settings.')
    return value
}
function text(value: JsonValue | undefined): string {
    return typeof value === 'string' ? value : ''
}
function number(value: JsonValue | undefined): number {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0)
        throw new Error('Invalid engine settings response.')
    return value
}
function flag(value: JsonValue | undefined): boolean {
    if (typeof value !== 'boolean') throw new Error('Invalid engine settings response.')
    return value
}
function revision(value: JsonValue | undefined): string {
    if (
        typeof value !== 'string' ||
        value.length > 19 ||
        !/^(0|[1-9][0-9]*)$/.test(value) ||
        BigInt(value) > 9223372036854775807n
    )
        throw new Error('Invalid engine settings revision.')
    return value
}
function optionalRevision(value: JsonValue | undefined): string | undefined {
    return value === undefined ? undefined : revision(value)
}
function decimal(value: JsonValue | undefined): number | undefined {
    if (value === undefined) return undefined
    if (
        typeof value !== 'number' ||
        !Number.isFinite(value) ||
        (Number.isInteger(value) && !Number.isSafeInteger(value))
    )
        throw new Error('Invalid model option bound.')
    return value
}
function boundedText(value: JsonValue | undefined, limit = 4096, required = false): string {
    const result = servingText(value)
    if (
        (required && result.length === 0) ||
        Buffer.byteLength(result, 'utf8') > limit ||
        /[\p{Cc}\p{Cs}]/u.test(result)
    )
        throw new Error('Invalid model text.')
    return result
}
function optionalText(value: JsonValue | undefined, limit = 4096): string | undefined {
    return value === undefined ? undefined : boundedText(value, limit)
}
function modelId(value: JsonValue | undefined): string {
    const id = boundedText(value, 512, true)
    if (['weak', 'medium', 'strong'].includes(id.replace(/-(api|local)$/, '')))
        throw new Error('Reserved model identifier.')
    return id
}
function shape(value: JsonValue | undefined, keys: readonly string[]): JsonObject {
    const row = object(value)
    if (Object.keys(row).some(key => !keys.includes(key)))
        throw new Error('Invalid model response field.')
    return row
}
function choice<const T extends string>(value: JsonValue | undefined, choices: readonly T[]): T {
    for (const candidate of choices) if (value === candidate) return candidate
    throw new Error('Invalid model response enum.')
}
function integer(value: JsonValue | undefined, minimum = 1): number {
    const result = decimal(value)
    if (
        result === undefined ||
        !Number.isInteger(result) ||
        result < minimum ||
        result > 2147483647
    )
        throw new Error('Invalid model integer bound.')
    return result
}
function optionalInteger(value: JsonValue | undefined, minimum = 1): number | undefined {
    return value === undefined ? undefined : integer(value, minimum)
}
function unsigned(value: JsonValue | undefined): string {
    const result = servingText(value)
    if (
        result.length > 20 ||
        !/^(0|[1-9][0-9]*)$/.test(result) ||
        BigInt(result) > 18446744073709551615n
    )
        throw new Error('Invalid model byte quantity.')
    return result
}
function optionalUnsigned(value: JsonValue | undefined): string | undefined {
    return value === undefined ? undefined : unsigned(value)
}
function signed(value: JsonValue | undefined): string {
    const result = servingText(value)
    if (
        result.length > 20 ||
        !/^(0|-?[1-9][0-9]*)$/.test(result) ||
        BigInt(result) < -9223372036854775808n ||
        BigInt(result) > 9223372036854775807n
    )
        throw new Error('Invalid model timestamp.')
    return result
}
function modelDigest(value: JsonValue | undefined): string {
    const result = servingText(value)
    if (!/^[a-f0-9]{64}$/.test(result)) throw new Error('Invalid model digest.')
    return result
}
function optionalDigest(value: JsonValue | undefined): string | undefined {
    return value === undefined ? undefined : modelDigest(value)
}
function boundedArray(value: JsonValue | undefined, maximum: number): JsonValue[] {
    const result = array(value)
    if (result.length > maximum) throw new Error('Model collection exceeds budget.')
    return result
}
function checkJSON(value: JsonValue, depth = 0): void {
    if (depth > 32) throw new Error('Model response exceeds nesting budget.')
    if (typeof value === 'number') decimal(value)
    else if (Array.isArray(value)) value.forEach(item => checkJSON(item, depth + 1))
    else if (value !== null && typeof value === 'object')
        Object.values(value).forEach(item => checkJSON(item, depth + 1))
}
function payload(value: JsonValue | undefined): void {
    if (value === undefined) throw new Error('Missing settings response.')
    checkJSON(value)
    if (Buffer.byteLength(JSON.stringify(value), 'utf8') > 64 * 1024)
        throw new Error('Settings response exceeds 64 KiB.')
}
function optionalFlag(value: JsonValue | undefined): boolean | undefined {
    return value === undefined ? undefined : flag(value)
}
function gpu(value: JsonValue | undefined): NonNullable<RequestedLoad['gpu']> {
    const row = shape(value, ['mode', 'layers'])
    const mode = choice(row.mode, ['native-default', 'cpu', 'layers', 'auto'])
    const layers = optionalInteger(row.layers, 0)
    if (mode === 'layers' ? layers === undefined : layers !== undefined)
        throw new Error('Invalid model GPU placement.')
    return { mode, layers }
}
function requested(value: JsonValue | undefined): RequestedLoad {
    const row = shape(value, [
        'context',
        'slots',
        'gpu',
        'cacheTypeK',
        'cacheTypeV',
        'flashAttention',
        'batchSize',
        'ubatchSize',
        'threads',
        'threadsBatch',
        'kvOffload',
        'opOffload',
        'sleepIdleSeconds',
        'cacheRAM'
    ])
    const context =
        row.context === undefined
            ? undefined
            : shape(row.context, ['mode', 'tokens', 'floor', 'cap'])
    const result: RequestedLoad = {
        context: context && {
            mode: choice(context.mode, ['fixed', 'auto']),
            tokens: optionalInteger(context.tokens),
            floor: optionalInteger(context.floor),
            cap: optionalInteger(context.cap)
        },
        slots: optionalInteger(row.slots),
        gpu: row.gpu === undefined ? undefined : gpu(row.gpu),
        cacheTypeK:
            row.cacheTypeK === undefined
                ? undefined
                : choice(row.cacheTypeK, ['f16', 'bf16', 'q8_0', 'q4_0']),
        cacheTypeV:
            row.cacheTypeV === undefined
                ? undefined
                : choice(row.cacheTypeV, ['f16', 'bf16', 'q8_0', 'q4_0']),
        flashAttention:
            row.flashAttention === undefined
                ? undefined
                : choice(row.flashAttention, ['inherit', 'on', 'off']),
        batchSize: optionalInteger(row.batchSize),
        ubatchSize: optionalInteger(row.ubatchSize),
        threads: optionalInteger(row.threads),
        threadsBatch: optionalInteger(row.threadsBatch),
        kvOffload: optionalFlag(row.kvOffload),
        opOffload: optionalFlag(row.opOffload),
        sleepIdleSeconds: optionalInteger(row.sleepIdleSeconds, -1),
        cacheRAM: optionalInteger(row.cacheRAM)
    }
    const c = result.context
    if (
        c &&
        (c.mode === 'fixed'
            ? c.tokens === undefined || c.floor !== undefined || c.cap !== undefined
            : c.tokens !== undefined ||
              c.floor === undefined ||
              c.cap === undefined ||
              c.floor > c.cap)
    )
        throw new Error('Invalid model context selection.')
    if (
        c?.mode === 'fixed' &&
        c.tokens !== undefined &&
        Math.ceil(c.tokens / 256) * 256 * (result.slots ?? 1) > 2147483647
    )
        throw new Error('Model context allocation overflows int32.')
    if (
        result.sleepIdleSeconds === 0 ||
        (result.batchSize !== undefined &&
            result.ubatchSize !== undefined &&
            result.ubatchSize > result.batchSize)
    )
        throw new Error('Invalid model load bounds.')
    return result
}
function resolved(value: JsonValue | undefined): ResolvedLoad {
    const row = shape(value, [
        'perSlotContext',
        'totalContext',
        'slots',
        'kvUnified',
        'gpu',
        'options',
        'artifacts',
        'nativeBuildId',
        'sourceInputDigest',
        'presetDigest',
        'resolverRevision',
        'reviewedInputId',
        'deviceBinding'
    ])
    const placement = gpu(row.gpu)
    if (placement.mode !== 'cpu' && placement.mode !== 'layers')
        throw new Error('Expected finite model placement.')
    const result: ResolvedLoad = {
        perSlotContext: integer(row.perSlotContext),
        totalContext: integer(row.totalContext),
        slots: integer(row.slots),
        kvUnified: optionalFlag(row.kvUnified),
        gpu: { mode: placement.mode, layers: placement.layers },
        options: requested(row.options),
        artifacts: boundedArray(row.artifacts, 1024).map(value => {
            const item = shape(value, [
                'role',
                'path',
                'resolvedTarget',
                'size',
                'mtimeNs',
                'sampleSha256'
            ])
            const size = revision(item.size)
            if (size === '0') throw new Error('Empty model artifact.')
            return {
                role: choice(item.role, ['model', 'draft']),
                path: boundedText(item.path, 4096, true),
                resolvedTarget: boundedText(item.resolvedTarget, 4096, true),
                size,
                mtimeNs: signed(item.mtimeNs),
                sampleSha256: modelDigest(item.sampleSha256)
            }
        }),
        nativeBuildId: boundedText(row.nativeBuildId, 256, true),
        sourceInputDigest: modelDigest(row.sourceInputDigest),
        presetDigest: modelDigest(row.presetDigest),
        resolverRevision: optionalText(row.resolverRevision, 256),
        reviewedInputId: optionalText(row.reviewedInputId, 256),
        deviceBinding:
            row.deviceBinding === undefined ? undefined : deviceBinding(row.deviceBinding)
    }
    if (
        result.perSlotContext % 256 !== 0 ||
        result.totalContext % 256 !== 0 ||
        (result.kvUnified
            ? result.perSlotContext * result.slots > result.totalContext
            : result.perSlotContext * result.slots !== result.totalContext)
    )
        throw new Error('Invalid resolved context allocation.')
    const options = result.options
    if (
        (options.context &&
            (options.context.mode !== 'fixed' ||
                options.context.tokens !== result.perSlotContext)) ||
        (options.slots !== undefined && options.slots !== result.slots) ||
        (options.gpu &&
            (options.gpu.mode !== result.gpu.mode || options.gpu.layers !== result.gpu.layers))
    )
        throw new Error('Resolved options disagree with allocation.')
    if (!!result.resolverRevision !== !!result.reviewedInputId)
        throw new Error('Incomplete reviewed model identity.')
    const usesGPU = result.gpu.mode === 'layers' && (result.gpu.layers ?? 0) > 0
    if (
        (result.deviceBinding && !usesGPU) ||
        (usesGPU && result.resolverRevision?.startsWith('auto/') && !result.deviceBinding)
    )
        throw new Error('Invalid reviewed device binding.')
    orderedResolvedArtifacts(result.artifacts)
    if (
        new Set(result.artifacts.map(item => item.path)).size !== result.artifacts.length ||
        new Set(result.artifacts.map(item => item.resolvedTarget)).size !== result.artifacts.length
    )
        throw new Error('Duplicate resolved artifact.')
    return result
}
function modelLoad(value: JsonValue): ModelLoad {
    const row = shape(value, ['modelId', 'requested', 'resolved'])
    const result = {
        modelId: modelId(row.modelId),
        requested: requested(row.requested),
        resolved: row.resolved === undefined ? undefined : resolved(row.resolved)
    }
    validateResolution(result.requested, result.resolved)
    return result
}
function validateResolution(intent: RequestedLoad, actual: ResolvedLoad | undefined): void {
    if (!actual) return
    const c = intent.context
    if (
        (intent.slots !== undefined && intent.slots !== actual.slots) ||
        (c &&
            (c.mode === 'fixed'
                ? c.tokens === undefined ||
                  Math.ceil(c.tokens / 256) * 256 !== actual.perSlotContext
                : c.floor === undefined ||
                  c.cap === undefined ||
                  actual.perSlotContext < c.floor ||
                  actual.perSlotContext > c.cap))
    )
        throw new Error('Resolved allocation differs from requested context.')
    if (
        intent.gpu &&
        intent.gpu.mode !== 'auto' &&
        intent.gpu.mode !== 'native-default' &&
        (intent.gpu.mode !== actual.gpu.mode || intent.gpu.layers !== actual.gpu.layers)
    )
        throw new Error('Resolved placement differs from requested GPU.')
    if (
        (c?.mode === 'auto' || intent.gpu?.mode === 'auto') &&
        (!actual.resolverRevision?.startsWith('auto/') || !actual.reviewedInputId)
    )
        throw new Error('Auto resolution requires reviewed identity.')
    const explicitFields: readonly (keyof RequestedLoad)[] = [
        'cacheTypeK',
        'cacheTypeV',
        'flashAttention',
        'batchSize',
        'ubatchSize',
        'threads',
        'threadsBatch',
        'kvOffload',
        'opOffload',
        'sleepIdleSeconds',
        'cacheRAM'
    ]
    for (const key of explicitFields) {
        if (intent[key] !== undefined && intent[key] !== actual.options[key])
            throw new Error('Resolution changed an explicit model option.')
    }
}
function strings(value: JsonValue | undefined, maximum = 256): string[] | undefined {
    return value === undefined
        ? undefined
        : boundedArray(value, maximum).map(item => boundedText(item))
}
function orderedResolvedArtifacts(artifacts: Array<{ role: string }>): void {
    if (!artifacts.length || artifacts[0].role !== 'model')
        throw new Error('Model shards required first.')
    let draft = false
    for (const item of artifacts) {
        if (item.role === 'draft') draft = true
        else if (item.role !== 'model' || draft) throw new Error('Invalid artifact role order.')
    }
}
function orderedSummaryArtifacts(artifacts: ArtifactSummary[]): void {
    if (!artifacts.length || artifacts[0].role !== 'model')
        throw new Error('Model shards required first.')
    let phase = 0
    for (const item of artifacts) {
        if (item.role === 'model' && phase !== 0) throw new Error('Invalid artifact role order.')
        if (item.role === 'draft') {
            if (phase > 1) throw new Error('Invalid artifact role order.')
            phase = 1
        }
        if (item.role === 'companion') phase = 2
    }
}
function summary(value: JsonValue): ArtifactSummary {
    const row = shape(value, ['role', 'identity', 'size', 'assurance'])
    const size = revision(row.size)
    if (size === '0') throw new Error('Empty model artifact.')
    return {
        role: choice(row.role, ['model', 'draft', 'companion']),
        identity: boundedText(row.identity, 256, true),
        size,
        assurance: boundedText(row.assurance, 256, true)
    }
}
function deviceBinding(value: JsonValue | undefined): NonNullable<ResolvedLoad['deviceBinding']> {
    const row = shape(value, ['hostId', 'domainId', 'deviceId', 'selector', 'digest'])
    const deviceId = boundedText(row.deviceId, 40, true)
    if (
        !/^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(deviceId) ||
        deviceId === 'GPU-00000000-0000-0000-0000-000000000000'
    )
        throw new Error('Unsupported physical device identity.')
    return {
        hostId: boundedText(row.hostId, 256, true),
        domainId: boundedText(row.domainId, 256, true),
        deviceId,
        selector: choice(row.selector, ['CUDA0']),
        digest: modelDigest(row.digest)
    }
}
function memoryFit(value: JsonValue | undefined): Estimate['capacityFit'] {
    return value === undefined ? undefined : choice(value, ['unknown', 'exceeds', 'tight', 'fits'])
}
function memoryValidity(value: JsonValue | undefined): number | undefined {
    if (value === undefined) return undefined
    const duration = integer(value, 0)
    if (duration > 5000) throw new Error('Memory validity exceeds observation budget.')
    return duration
}
const memoryPlanGroupNames: ReadonlyArray<MemoryPlan['devices'][number]['groups'][number]['name']> =
    [
        'weights',
        'kv_cache',
        'compute_scratch',
        'output_buffers',
        'runtime_driver_overhead',
        'transient_peak'
    ]
function memoryPlanArray(value: JsonValue | undefined, maximum = 256): JsonValue[] {
    return value === undefined || value === null ? [] : boundedArray(value, maximum)
}
function memoryPlanTexts(value: JsonValue | undefined, limit = 256): string[] | undefined {
    if (value === undefined || value === null) return undefined
    const result = memoryPlanArray(value).map(item => boundedText(item, limit, true))
    if (new Set(result).size !== result.length) throw new Error('Duplicate memory plan text.')
    return result
}
function memoryPlanConfig(
    value: JsonValue | undefined,
    isResolved: boolean
): MemoryPlan['requested'] {
    const row = shape(value, [
        'contextTokens',
        'kvFormat',
        'batchSize',
        'microbatch',
        'parallelSlots',
        'devices',
        'placement'
    ])
    const microbatch =
        row.microbatch === undefined
            ? undefined
            : shape(row.microbatch, ['mode', 'tokens', 'limit'])
    const result: MemoryPlan['requested'] = {
        contextTokens: integer(row.contextTokens),
        kvFormat:
            row.kvFormat === undefined
                ? undefined
                : choice(row.kvFormat, ['fp16', 'bf16', 'int8', 'q4_0', 'k8v4', 'f16', 'q8_0']),
        batchSize: optionalInteger(row.batchSize),
        parallelSlots: optionalInteger(row.parallelSlots),
        microbatch: microbatch && {
            mode: choice(microbatch.mode, ['fixed', 'auto']),
            tokens: optionalInteger(microbatch.tokens),
            limit: optionalInteger(microbatch.limit)
        },
        devices: memoryPlanArray(row.devices, 64).map(value => {
            const device = shape(value, ['backend', 'selector'])
            return {
                backend: boundedText(device.backend, 256, true),
                selector: boundedText(device.selector, 256, true)
            }
        }),
        placement: choice(row.placement, ['cpu', 'device', 'hybrid'])
    }
    const m = result.microbatch
    if (
        (isResolved &&
            (!result.kvFormat ||
                result.batchSize === undefined ||
                result.parallelSlots === undefined ||
                !m)) ||
        (m?.mode === 'fixed' &&
            (m.tokens === undefined || m.tokens > result.contextTokens || m.limit !== undefined)) ||
        (m?.mode === 'auto' && (isResolved || m.tokens !== undefined)) ||
        new Set(result.devices.map(device => device.selector)).size !== result.devices.length ||
        (result.placement === 'cpu' ? result.devices.length !== 0 : result.devices.length === 0)
    )
        throw new Error('Invalid memory plan configuration.')
    return result
}
function memoryPlanMembers(
    value: JsonValue | undefined
): MemoryPlan['devices'][number]['groups'][number]['combinedMembership'] {
    if (value === undefined || value === null) return undefined
    const result = memoryPlanArray(value, memoryPlanGroupNames.length).map(item =>
        choice(item, memoryPlanGroupNames)
    )
    if (new Set(result).size !== result.length) throw new Error('Duplicate memory plan member.')
    return result
}
function memoryPlanQuantity(
    value: JsonObject
): Pick<MemoryPlan['devices'][number]['groups'][number], 'bytes' | 'quality' | 'basis'> {
    const result = {
        bytes: optionalUnsigned(value.bytes),
        quality: choice(value.quality, ['exact', 'estimated', 'unknown']),
        basis: boundedText(value.basis, 1024, true)
    }
    if ((result.quality === 'unknown') !== (result.bytes === undefined))
        throw new Error('Memory plan quality contradicts bytes.')
    return result
}
function memoryPlanAmounts(
    value: JsonValue | undefined,
    allowance: boolean
): MemoryPlan['devices'][number]['assumptions'] {
    if (value === undefined || value === null) return undefined
    const result = memoryPlanArray(value).map(value => {
        const row = shape(value, ['name', 'bytes', 'basis', 'provenance'])
        return {
            name: boundedText(row.name, 256, true),
            bytes: unsigned(row.bytes),
            basis: optionalText(row.basis, 1024),
            provenance: allowance
                ? boundedText(row.provenance, 256, true)
                : optionalText(row.provenance, 256)
        }
    })
    if (new Set(result.map(amount => amount.name)).size !== result.length)
        throw new Error('Duplicate memory plan amount.')
    return result
}
function memoryPlanResource(
    value: JsonValue | undefined,
    kind: MemoryPlan['devices'][number]['kind']
): MemoryPlan['devices'][number] {
    const row = shape(value, [
        'kind',
        'nativeSelector',
        'physicalId',
        'poolId',
        'mapping',
        'mappingBasis',
        'groups',
        'combinedAllocations',
        'allocatedBytes',
        'mappedViewBytes',
        'pricedTotalBytes',
        'pricedTotalBasis',
        'minimumSelectionCapacityBytes',
        'minimumSelectionCapacityBasis',
        'assumptions',
        'requiredWithAllowancesBytes',
        'budgetBytes',
        'shortfallBytes',
        'shortfallBasis',
        'callerAllowances',
        'uncoveredCosts'
    ])
    const result: MemoryPlan['devices'][number] = {
        kind: choice(row.kind, ['device', 'host_ram', 'host_commit']),
        nativeSelector: optionalText(row.nativeSelector, 256),
        physicalId: optionalText(row.physicalId, 256),
        poolId: optionalText(row.poolId, 256),
        mapping: choice(row.mapping, ['unproven', 'proven', 'not-applicable']),
        mappingBasis: boundedText(row.mappingBasis, 1024, true),
        groups: memoryPlanArray(row.groups, memoryPlanGroupNames.length).map(value => {
            const group = shape(value, [
                'name',
                'bytes',
                'quality',
                'basis',
                'additive',
                'combinedMembership'
            ])
            return {
                name: choice(group.name, memoryPlanGroupNames),
                ...memoryPlanQuantity(group),
                additive: flag(group.additive),
                combinedMembership: memoryPlanMembers(group.combinedMembership)
            }
        }),
        combinedAllocations:
            row.combinedAllocations === undefined || row.combinedAllocations === null
                ? undefined
                : memoryPlanArray(row.combinedAllocations).map(value => {
                      const combined = shape(value, ['members', 'bytes', 'quality', 'basis'])
                      const members = memoryPlanMembers(combined.members)
                      if (!members?.length) throw new Error('Missing combined allocation members.')
                      return { members, ...memoryPlanQuantity(combined) }
                  }),
        allocatedBytes: optionalUnsigned(row.allocatedBytes),
        mappedViewBytes: optionalUnsigned(row.mappedViewBytes),
        pricedTotalBytes: unsigned(row.pricedTotalBytes),
        pricedTotalBasis: boundedText(row.pricedTotalBasis, 1024, true),
        minimumSelectionCapacityBytes: optionalUnsigned(row.minimumSelectionCapacityBytes),
        minimumSelectionCapacityBasis: optionalText(row.minimumSelectionCapacityBasis, 1024),
        assumptions: memoryPlanAmounts(row.assumptions, false),
        requiredWithAllowancesBytes: unsigned(row.requiredWithAllowancesBytes),
        budgetBytes: optionalUnsigned(row.budgetBytes),
        shortfallBytes: optionalUnsigned(row.shortfallBytes),
        shortfallBasis: optionalText(row.shortfallBasis, 1024),
        callerAllowances: memoryPlanAmounts(row.callerAllowances, true),
        uncoveredCosts: memoryPlanTexts(row.uncoveredCosts)
    }
    if (
        result.kind !== kind ||
        (kind === 'device' && (!result.nativeSelector || result.mapping === 'not-applicable')) ||
        (kind !== 'device' && !!result.nativeSelector) ||
        (result.mapping === 'proven'
            ? !result.physicalId && !result.poolId
            : !!result.physicalId || !!result.poolId) ||
        new Set(result.groups.map(group => group.name)).size !== memoryPlanGroupNames.length ||
        result.groups.some(group => group.name === 'transient_peak' && group.additive) ||
        !!result.minimumSelectionCapacityBasis !==
            (result.minimumSelectionCapacityBytes !== undefined) ||
        !!result.shortfallBasis !== (result.shortfallBytes !== undefined) ||
        (result.budgetBytes === undefined) !== (result.shortfallBytes === undefined) ||
        result.callerAllowances?.some(amount => result.uncoveredCosts?.includes(amount.name))
    )
        throw new Error('Invalid memory plan resource accounting.')
    // Check the provider's accounting identities; PAIR fit stays service-owned.
    const maximum = 18446744073709551615n
    const allowances = (result.callerAllowances ?? []).reduce(
        (sum, amount) => sum + BigInt(amount.bytes),
        0n
    )
    const threshold = BigInt(result.minimumSelectionCapacityBytes ?? '0')
    const priced = BigInt(result.pricedTotalBytes)
    const required = (threshold > priced ? threshold : priced) + allowances
    if (
        allowances > maximum ||
        required > maximum ||
        required !== BigInt(result.requiredWithAllowancesBytes)
    )
        throw new Error('Invalid memory plan required total.')
    if (result.budgetBytes !== undefined) {
        const budget = BigInt(result.budgetBytes)
        const shortfall = required > budget ? required - budget : 0n
        if (result.shortfallBytes === undefined || shortfall !== BigInt(result.shortfallBytes))
            throw new Error('Invalid memory plan shortfall.')
    }
    return result
}
function memoryPlan(value: JsonValue): MemoryPlan {
    const row = shape(value, [
        'identity',
        'requested',
        'resolved',
        'configurationComplete',
        'pricingScope',
        'pricingQualification',
        'devices',
        'hostRAM',
        'hostCommit',
        'verdict',
        'warnings'
    ])
    const identity = shape(row.identity, [
        'engine',
        'provider',
        'revision',
        'executableSHA256',
        'build',
        'runtimeSourceSHA256',
        'modelId',
        'contentSHA256',
        'contentAssurance',
        'assertedContentSHA256',
        'artifactIdentitySHA256',
        'cacheable',
        'cacheableReason'
    ])
    const build = shape(identity.build, ['version', 'revision', 'dirty', 'timestamp', 'basis'])
    const verdict = shape(row.verdict, ['status', 'conditional', 'budgetBasis', 'basis'])
    const result: MemoryPlan = {
        identity: {
            engine: boundedText(identity.engine, 256, true),
            provider: boundedText(identity.provider, 256, true),
            revision: boundedText(identity.revision, 256, true),
            executableSHA256: modelDigest(identity.executableSHA256),
            build: {
                version: optionalText(build.version, 256),
                revision: optionalText(build.revision, 256),
                dirty: optionalFlag(build.dirty),
                timestamp: optionalText(build.timestamp, 256),
                basis: boundedText(build.basis, 1024, true)
            },
            runtimeSourceSHA256: modelDigest(identity.runtimeSourceSHA256),
            modelId: modelId(identity.modelId),
            contentSHA256: optionalDigest(identity.contentSHA256),
            contentAssurance: boundedText(identity.contentAssurance, 256, true),
            assertedContentSHA256: optionalDigest(identity.assertedContentSHA256),
            artifactIdentitySHA256: modelDigest(identity.artifactIdentitySHA256),
            cacheable: flag(identity.cacheable),
            cacheableReason: optionalText(identity.cacheableReason, 1024)
        },
        requested: memoryPlanConfig(row.requested, false),
        resolved: row.resolved === undefined ? undefined : memoryPlanConfig(row.resolved, true),
        configurationComplete: flag(row.configurationComplete),
        pricingScope: choice(row.pricingScope, [
            'resolved_configuration',
            'minimum_supported_layout'
        ]),
        pricingQualification: choice(row.pricingQualification, ['non-replayable-pricing-inputs']),
        devices: memoryPlanArray(row.devices, 64).map(value => memoryPlanResource(value, 'device')),
        hostRAM: memoryPlanResource(row.hostRAM, 'host_ram'),
        hostCommit: memoryPlanResource(row.hostCommit, 'host_commit'),
        verdict: {
            status: choice(verdict.status, [
                'not_evaluated',
                'does_not_fit',
                'indeterminate',
                'fits_within_allowances'
            ]),
            conditional: flag(verdict.conditional),
            budgetBasis:
                verdict.budgetBasis === undefined
                    ? undefined
                    : choice(verdict.budgetBasis, [
                          'before_runtime_initialization',
                          'after_runtime_initialization'
                      ]),
            basis: boundedText(verdict.basis, 1024, true)
        },
        warnings: memoryPlanTexts(row.warnings, 1024)
    }
    if (
        (result.identity.cacheable
            ? !result.identity.contentSHA256
            : !result.identity.cacheableReason) ||
        result.configurationComplete !== (result.resolved !== undefined) ||
        result.pricingScope !==
            (result.configurationComplete
                ? 'resolved_configuration'
                : 'minimum_supported_layout') ||
        result.devices.length !== result.requested.devices.length ||
        result.devices.some(
            (device, index) => device.nativeSelector !== result.requested.devices[index]?.selector
        )
    )
        throw new Error('Inconsistent memory plan identity or scope.')
    const physical = result.devices.flatMap(device =>
        device.physicalId ? [device.physicalId] : []
    )
    if (new Set(physical).size !== physical.length)
        throw new Error('Duplicate memory plan physical device.')
    if (result.resolved) {
        const requested = result.requested
        const resolved = result.resolved
        const canonicalKV = (value: string | undefined): string | undefined =>
            value === 'f16' ? 'fp16' : value === 'q8_0' ? 'int8' : value
        if (
            resolved.contextTokens !== requested.contextTokens ||
            resolved.placement !== requested.placement ||
            resolved.devices.length !== requested.devices.length ||
            resolved.devices.some(
                (device, index) =>
                    device.selector !== requested.devices[index]?.selector ||
                    device.backend !== requested.devices[index]?.backend
            ) ||
            (requested.kvFormat !== undefined &&
                canonicalKV(requested.kvFormat) !== canonicalKV(resolved.kvFormat)) ||
            (requested.batchSize !== undefined && requested.batchSize !== resolved.batchSize) ||
            (requested.parallelSlots !== undefined &&
                requested.parallelSlots !== resolved.parallelSlots) ||
            (requested.microbatch?.mode === 'fixed' &&
                requested.microbatch.tokens !== resolved.microbatch?.tokens) ||
            (requested.microbatch?.mode === 'auto' &&
                requested.microbatch.limit !== undefined &&
                resolved.microbatch?.tokens !== undefined &&
                resolved.microbatch.tokens > requested.microbatch.limit)
        )
            throw new Error('Memory plan resolution changed exact settings.')
    }
    const resources = [...result.devices, result.hostRAM, result.hostCommit]
    const hasBudget = result.verdict.status !== 'not_evaluated'
    if (
        hasBudget !== (result.verdict.budgetBasis !== undefined) ||
        resources.some(resource => (resource.budgetBytes !== undefined) !== hasBudget)
    )
        throw new Error('Incomplete memory plan budget.')
    const shortfall = resources.some(
        resource => resource.shortfallBytes !== undefined && resource.shortfallBytes !== '0'
    )
    const uncovered = resources.some(resource => resource.uncoveredCosts?.length)
    if (
        (hasBudget &&
            (shortfall
                ? result.verdict.status !== 'does_not_fit'
                : uncovered
                  ? result.verdict.status !== 'indeterminate'
                  : result.verdict.status !== 'fits_within_allowances')) ||
        (result.verdict.status === 'fits_within_allowances' && !result.verdict.conditional)
    )
        throw new Error('Memory plan verdict contradicts accounting.')
    return result
}
function memoryPlanPartial(plan: MemoryPlan): boolean {
    return (
        !plan.configurationComplete ||
        plan.verdict.conditional ||
        [...plan.devices, plan.hostRAM, plan.hostCommit].some(
            resource =>
                resource.uncoveredCosts?.length ||
                resource.groups.some(group => group.quality === 'unknown') ||
                resource.combinedAllocations?.some(combined => combined.quality === 'unknown')
        )
    )
}
function estimateSource(value: JsonValue): NonNullable<Estimate['estimateSource']> {
    const row = shape(value, ['kind', 'provider', 'revision', 'basis', 'unpricedCosts'])
    const costs = strings(row.unpricedCosts)
    if (costs && new Set(costs).size !== costs.length) throw new Error('Duplicate unpriced cost.')
    return {
        kind: choice(row.kind, ['engine-report', 'pair-estimate', 'measured', 'user-stated']),
        provider: boundedText(row.provider, 256, true),
        revision: boundedText(row.revision, 256, true),
        basis: boundedText(row.basis, 1024, true),
        unpricedCosts: costs?.map(cost => boundedText(cost, 256, true))
    }
}
function validateByteRange(
    value: string | undefined,
    low: string | undefined,
    high: string | undefined,
    required: boolean
): void {
    if (value === undefined && low === undefined && high === undefined && !required) return
    if (value === undefined || (required && (low === undefined || high === undefined)))
        throw new Error('Incomplete byte range.')
    if (
        (low !== undefined && BigInt(low) > BigInt(value)) ||
        (high !== undefined && BigInt(high) < BigInt(value))
    )
        throw new Error('Invalid byte range.')
}
function estimate(value: JsonValue | undefined): Estimate {
    const row = shape(value, [
        'status',
        'estimateSource',
        'memoryPlan',
        'source',
        'buildId',
        'timestamp',
        'memorySource',
        'memoryTimestamp',
        'capacityFit',
        'freeFit',
        'hostBytes',
        'deviceBytes',
        'domains',
        'components',
        'assumptions',
        'unknowns'
    ])
    const result: Estimate = {
        status: choice(row.status, ['estimated', 'partial', 'unknown', 'unavailable']),
        estimateSource:
            row.estimateSource === undefined ? undefined : estimateSource(row.estimateSource),
        memoryPlan: row.memoryPlan === undefined ? undefined : memoryPlan(row.memoryPlan),
        source: optionalText(row.source),
        buildId: optionalText(row.buildId),
        timestamp: optionalText(row.timestamp),
        memorySource: optionalText(row.memorySource),
        memoryTimestamp: optionalText(row.memoryTimestamp),
        capacityFit: memoryFit(row.capacityFit),
        freeFit: memoryFit(row.freeFit),
        hostBytes: optionalUnsigned(row.hostBytes),
        deviceBytes: optionalUnsigned(row.deviceBytes),
        domains:
            row.domains === undefined
                ? undefined
                : boundedArray(row.domains, 64).map(value => {
                      const item = shape(value, [
                          'id',
                          'kind',
                          'deviceId',
                          'capacityBytes',
                          'freeBytes',
                          'requiredBytes',
                          'requiredLowBytes',
                          'requiredHighBytes',
                          'reserveBytes',
                          'capacityFit',
                          'freeFit',
                          'source',
                          'sampleId',
                          'capacitySource',
                          'capacityTimestamp',
                          'freeSource',
                          'freeTimestamp',
                          'remainingValidityMs',
                          'unknowns'
                      ])
                      return {
                          id: boundedText(item.id, 256, true),
                          kind: choice(item.kind, ['host', 'device', 'shared']),
                          deviceId: optionalText(item.deviceId),
                          capacityBytes: optionalUnsigned(item.capacityBytes),
                          freeBytes: optionalUnsigned(item.freeBytes),
                          requiredBytes: optionalUnsigned(item.requiredBytes),
                          requiredLowBytes: optionalUnsigned(item.requiredLowBytes),
                          requiredHighBytes: optionalUnsigned(item.requiredHighBytes),
                          reserveBytes: optionalUnsigned(item.reserveBytes),
                          capacityFit: memoryFit(item.capacityFit),
                          freeFit: memoryFit(item.freeFit),
                          source: optionalText(item.source, 256),
                          sampleId: optionalText(item.sampleId, 513),
                          capacitySource: optionalText(item.capacitySource, 256),
                          capacityTimestamp: optionalText(item.capacityTimestamp, 256),
                          freeSource: optionalText(item.freeSource, 256),
                          freeTimestamp: optionalText(item.freeTimestamp, 256),
                          remainingValidityMs: memoryValidity(item.remainingValidityMs),
                          unknowns: strings(item.unknowns)
                      }
                  }),
        components:
            row.components === undefined
                ? undefined
                : boundedArray(row.components, 256).map(value => {
                      const item = shape(value, [
                          'name',
                          'domainId',
                          'status',
                          'bytes',
                          'lowBytes',
                          'highBytes',
                          'basis',
                          'reason'
                      ])
                      return {
                          name: memoryComponentName(item.name),
                          domainId: optionalText(item.domainId),
                          status: choice(item.status, [
                              'estimated',
                              'partial',
                              'unknown',
                              'excluded'
                          ]),
                          bytes: optionalUnsigned(item.bytes),
                          lowBytes: optionalUnsigned(item.lowBytes),
                          highBytes: optionalUnsigned(item.highBytes),
                          basis:
                              item.basis === undefined
                                  ? undefined
                                  : choice(item.basis, [
                                        'derived',
                                        'heuristic',
                                        'calibrated',
                                        'observed'
                                    ]),
                          reason: optionalText(item.reason)
                      }
                  }),
        assumptions: strings(row.assumptions),
        unknowns: strings(row.unknowns)
    }
    for (const domain of result.domains ?? []) {
        validateByteRange(
            domain.requiredBytes,
            domain.requiredLowBytes,
            domain.requiredHighBytes,
            false
        )
    }
    for (const component of result.components ?? []) {
        const priced = component.status === 'estimated' || component.status === 'partial'
        validateByteRange(component.bytes, component.lowBytes, component.highBytes, priced)
        if (
            (priced && !component.basis) ||
            (!priced &&
                (component.bytes !== undefined ||
                    component.lowBytes !== undefined ||
                    component.highBytes !== undefined ||
                    component.basis !== undefined)) ||
            (component.status === 'partial' && !component.reason)
        )
            throw new Error('Invalid component range or basis.')
    }
    const domains = new Set(result.domains?.map(item => item.id))
    const physicalDevices =
        result.domains?.flatMap(item => (item.deviceId ? [item.deviceId] : [])) ?? []
    const hostPools =
        result.domains?.filter(item => item.kind === 'host' || item.kind === 'shared') ?? []
    if (
        domains.size !== (result.domains?.length ?? 0) ||
        new Set(physicalDevices).size !== physicalDevices.length ||
        hostPools.length > 1 ||
        result.components?.some(
            item =>
                (item.domainId && !domains.has(item.domainId)) ||
                (item.status === 'excluded' && (!item.reason || item.bytes !== undefined))
        )
    )
        throw new Error('Invalid estimate domain reference.')
    if (result.memoryPlan) {
        const plan = result.memoryPlan
        const source = result.estimateSource
        if (
            !source ||
            !['engine-report', 'user-stated', 'measured'].includes(source.kind) ||
            (source.kind === 'engine-report' &&
                (source.provider !== plan.identity.provider ||
                    source.revision !== plan.identity.revision)) ||
            result.components?.length ||
            !['partial', 'estimated'].includes(result.status) ||
            ((memoryPlanPartial(plan) || source.unpricedCosts?.length) &&
                result.status !== 'partial')
        )
            throw new Error('Inconsistent memory plan estimate provenance or status.')
    }
    return result
}
function memoryComponentName(
    value: JsonValue | undefined
): NonNullable<LoadBufferObservation['baseline']>[number]['component'] {
    return choice(value, [
        'weights',
        'attention-cache',
        'recurrent-state',
        'compute-workspace',
        'prompt-cache',
        'companions',
        'runtime-overhead'
    ])
}
function bufferReason(value: JsonValue | undefined): NonNullable<LoadBufferObservation['reason']> {
    return choice(value, [
        'observation-omitted',
        'response-budget',
        'history-unavailable',
        'no-matching-sample',
        'missing-boundary',
        'unsupported-build',
        'clock-reversal',
        'unmapped-device',
        'mixed-role',
        'incomplete-reserve',
        'oversized-frame',
        'truncated-frame',
        'invalid-utf8',
        'read-failure',
        'capture-budget',
        'unknown-baseline',
        'source-changed',
        'source-unavailable',
        'unsupported-buffer'
    ])
}
function bufferSigned(value: JsonValue | undefined): string | undefined {
    if (value === undefined) return undefined
    const result = boundedText(value, 21, true)
    unsigned(result.startsWith('-') ? result.slice(1) : result)
    if (result === '-0') throw new Error('Invalid buffer signed interval.')
    return result
}
function bufferTimestamp(value: JsonValue | undefined): string | undefined {
    if (value === undefined) return undefined
    const result = boundedText(value, 256, true)
    const match =
        /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,9})?(Z|[+-]\d{2}:\d{2})$/.exec(
            result
        )
    if (!match || !Number.isFinite(Date.parse(result))) throw new Error('Invalid buffer timestamp.')
    const year = Number(match[1]),
        month = Number(match[2]),
        day = Number(match[3])
    const days = [
        31,
        year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0) ? 29 : 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31
    ]
    if (
        month < 1 ||
        month > 12 ||
        day < 1 ||
        day > days[month - 1] ||
        Number(match[4]) > 23 ||
        Number(match[5]) > 59 ||
        Number(match[6]) > 59
    )
        throw new Error('Invalid buffer timestamp.')
    return result
}
function bufferReportRow(value: JsonValue): NonNullable<LoadBufferObservation['rows']>[number] {
    const row = shape(value, [
        'component',
        'label',
        'phase',
        'sequence',
        'stream',
        'role',
        'reportedMiB',
        'unit',
        'precision',
        'sourceKind',
        'intervalStatus',
        'lowerBytes',
        'upperBytes',
        'estimatedBytes',
        'deviceBinding',
        'poolIdentity',
        'coverage'
    ])
    const result: NonNullable<LoadBufferObservation['rows']>[number] = {
        component: choice(row.component, ['model', 'kv', 'compute', 'output', 'recurrent']),
        label: boundedText(row.label, 256, true),
        phase: choice(row.phase, ['load', 'reserve']),
        sequence: unsigned(row.sequence),
        stream: row.stream === undefined ? undefined : choice(row.stream, ['stdout', 'stderr']),
        role: choice(row.role, ['model', 'mixed', 'unattributed']),
        reportedMiB: boundedText(row.reportedMiB, 256, true),
        unit: choice(row.unit, ['MiB']),
        precision: choice(row.precision, ['0.01', 'unknown']),
        sourceKind: choice(row.sourceKind, ['log-scraped']),
        intervalStatus: choice(row.intervalStatus, ['bounded', 'unknown']),
        lowerBytes: optionalUnsigned(row.lowerBytes),
        upperBytes: optionalUnsigned(row.upperBytes),
        estimatedBytes: optionalUnsigned(row.estimatedBytes),
        deviceBinding: optionalDigest(row.deviceBinding),
        poolIdentity: optionalDigest(row.poolIdentity),
        coverage: choice(row.coverage, ['partial', 'unknown', 'compatible'])
    }
    const label = /^(?:CUDA(0|[1-9][0-9]?)|MTL(0|[1-9][0-9]?)(?:_Private|_Mapped)?)$/.exec(
        result.label
    )
    if (
        (!['CPU', 'CPU_Mapped', 'CUDA_Host'].includes(result.label) &&
            (!label || Number(label[1] ?? label[2]) >= 64)) ||
        result.sequence === '0' ||
        !/^(0|[1-9][0-9]*)\.[0-9]{1,16}$/.test(result.reportedMiB)
    )
        throw new Error('Unsupported buffer report.')
    if (result.intervalStatus === 'unknown') {
        if (
            result.lowerBytes !== undefined ||
            result.upperBytes !== undefined ||
            result.coverage === 'compatible'
        )
            throw new Error('Unknown buffer interval has byte endpoints.')
    } else {
        const [whole, fraction] = result.reportedMiB.split('.')
        if (result.precision !== '0.01' || fraction.length !== 2 || BigInt(whole) >= 1n << 30n)
            throw new Error('Unsupported buffer precision.')
        const q = BigInt(whole) * 1000n + BigInt(fraction) * 10n
        const lower = q < 5n ? 0n : ((q - 5n) * 1048576n) / 1000n
        const upper = ((q + 5n) * 1048576n + 999n) / 1000n
        if (result.lowerBytes !== lower.toString() || result.upperBytes !== upper.toString())
            throw new Error('Invalid rounded buffer interval.')
    }
    return result
}
function loadObservation(value: JsonValue | undefined): LoadBufferObservation {
    if (value === undefined)
        return { status: 'unknown', current: false, reason: 'observation-omitted' }
    if (Buffer.byteLength(JSON.stringify(value), 'utf8') > 4096)
        throw new Error('Buffer observation exceeds 4 KiB.')
    const row = shape(value, [
        'status',
        'reason',
        'sampleId',
        'recordedAt',
        'current',
        'persistence',
        'upstreamPin',
        'buildAssurance',
        'estimatorRevision',
        'inputIdentity',
        'sourceIdentity',
        'optionsIdentity',
        'topologyIdentity',
        'configurationKey',
        'loadPhase',
        'perSlotContext',
        'slots',
        'capturedOptions',
        'requestedOptions',
        'optionUnknowns',
        'artifacts',
        'baseline',
        'comparisons',
        'rows',
        'reasons'
    ])
    const result: LoadBufferObservation = {
        status: choice(row.status, ['unknown', 'partial', 'reported']),
        current: flag(row.current),
        reason: row.reason === undefined ? undefined : bufferReason(row.reason),
        sampleId: optionalDigest(row.sampleId),
        recordedAt: bufferTimestamp(row.recordedAt),
        persistence:
            row.persistence === undefined
                ? undefined
                : choice(row.persistence, [
                      'pending',
                      'saved',
                      'uncertain',
                      'failed',
                      'history-unavailable'
                  ]),
        upstreamPin:
            row.upstreamPin === undefined
                ? undefined
                : choice(row.upstreamPin, ['7fe450e19305b828c199d602c23a8337aaa1f03b']),
        buildAssurance:
            row.buildAssurance === undefined
                ? undefined
                : choice(row.buildAssurance, ['recognized-self-reported-pin']),
        estimatorRevision: optionalText(row.estimatorRevision, 256),
        inputIdentity: optionalDigest(row.inputIdentity),
        sourceIdentity: optionalDigest(row.sourceIdentity),
        optionsIdentity: optionalDigest(row.optionsIdentity),
        topologyIdentity: optionalDigest(row.topologyIdentity),
        configurationKey: optionalDigest(row.configurationKey),
        loadPhase:
            row.loadPhase === undefined ? undefined : choice(row.loadPhase, ['initial', 'wakeup']),
        perSlotContext: optionalInteger(row.perSlotContext, 0),
        slots: optionalInteger(row.slots, 0),
        capturedOptions:
            row.capturedOptions === undefined ? undefined : requested(row.capturedOptions),
        requestedOptions:
            row.requestedOptions === undefined ? undefined : requested(row.requestedOptions),
        optionUnknowns:
            row.optionUnknowns === undefined
                ? undefined
                : boundedArray(row.optionUnknowns, 64).map(value =>
                      choice(value, [
                          'mmap',
                          'load-mode',
                          'flash-attention',
                          'batch',
                          'ubatch',
                          'cache-k',
                          'cache-v',
                          'kv-offload',
                          'op-offload',
                          'placement',
                          'speculation'
                      ])
                  ),
        artifacts:
            row.artifacts === undefined
                ? undefined
                : boundedArray(row.artifacts, 1024).map(value => {
                      const artifact = shape(value, ['role', 'identity', 'size', 'assurance'])
                      return {
                          role: choice(artifact.role, ['model', 'draft', 'companion']),
                          identity: modelDigest(artifact.identity),
                          size: unsigned(artifact.size),
                          assurance: choice(artifact.assurance, ['sampled-local-binding'])
                      }
                  }),
        baseline:
            row.baseline === undefined
                ? undefined
                : boundedArray(row.baseline, 64).map(value => {
                      const component = shape(value, ['component', 'status', 'bytes'])
                      return {
                          component: memoryComponentName(component.component),
                          status: choice(component.status, [
                              'estimated',
                              'partial',
                              'unknown',
                              'excluded'
                          ]),
                          bytes: optionalUnsigned(component.bytes)
                      }
                  }),
        comparisons:
            row.comparisons === undefined
                ? undefined
                : boundedArray(row.comparisons, 6).map(value => {
                      const comparison = shape(value, [
                          'component',
                          'status',
                          'estimatedBytes',
                          'lowerBytes',
                          'upperBytes',
                          'deltaLowerBytes',
                          'deltaUpperBytes'
                      ])
                      return {
                          component: memoryComponentName(comparison.component),
                          status: choice(comparison.status, ['compatible', 'unknown']),
                          estimatedBytes: optionalUnsigned(comparison.estimatedBytes),
                          lowerBytes: optionalUnsigned(comparison.lowerBytes),
                          upperBytes: optionalUnsigned(comparison.upperBytes),
                          deltaLowerBytes: bufferSigned(comparison.deltaLowerBytes),
                          deltaUpperBytes: bufferSigned(comparison.deltaUpperBytes)
                      }
                  }),
        rows: row.rows === undefined ? undefined : boundedArray(row.rows, 256).map(bufferReportRow),
        reasons:
            row.reasons === undefined ? undefined : boundedArray(row.reasons, 16).map(bufferReason)
    }
    if (
        result.estimatorRevision !== undefined &&
        result.estimatorRevision !==
            'pair-gguf/2;gguf-parser-go/v0.27.0@9257743a3c1775b882ba1b58ec7f5a856c2cc385'
    )
        throw new Error('Unsupported buffer estimator revision.')
    if (result.current && (result.status === 'unknown' || !result.sampleId))
        throw new Error('Unknown current buffer sample.')
    for (const entries of [result.baseline, result.comparisons]) {
        if (new Set(entries?.map(entry => entry.component)).size !== (entries?.length ?? 0))
            throw new Error('Duplicate buffer component.')
    }
    const frames = result.rows?.map(row => `${row.stream ?? ''}:${row.sequence}`) ?? []
    if (new Set(frames).size !== frames.length) throw new Error('Duplicate buffer source frame.')
    for (const comparison of result.comparisons ?? []) {
        if (comparison.status === 'unknown') {
            if (
                [
                    comparison.lowerBytes,
                    comparison.upperBytes,
                    comparison.deltaLowerBytes,
                    comparison.deltaUpperBytes
                ].some(value => value !== undefined)
            )
                throw new Error('Unknown comparison has an interval.')
            continue
        }
        const baseline = result.baseline?.find(item => item.component === comparison.component)
        if (
            baseline?.status !== 'estimated' ||
            baseline.bytes !== comparison.estimatedBytes ||
            comparison.estimatedBytes === undefined ||
            comparison.lowerBytes === undefined ||
            comparison.upperBytes === undefined ||
            BigInt(comparison.lowerBytes) > BigInt(comparison.upperBytes) ||
            (BigInt(comparison.lowerBytes) - BigInt(comparison.estimatedBytes)).toString() !==
                comparison.deltaLowerBytes ||
            (BigInt(comparison.upperBytes) - BigInt(comparison.estimatedBytes)).toString() !==
                comparison.deltaUpperBytes
        )
            throw new Error('Invalid frozen buffer comparison.')
    }
    return result
}
export function parseInspectModel(value: JsonValue | undefined): InspectModelResult {
    payload(value)
    const row = shape(value, [
        'engine',
        'nodeId',
        'modelId',
        'status',
        'reason',
        'source',
        'assurance',
        'artifacts',
        'facts',
        'options',
        'estimate',
        'loadObservation',
        'configurationKey',
        'configurationKeyReason',
        'configurationPools',
        'memoryProfiles',
        'memoryProfilesUnavailable'
    ])
    const engine = boundedText(row.engine, 40, true)
    if (!isValidRoutingEngineId(engine)) throw new Error('Invalid inspection engine.')
    const facts =
        row.facts === undefined
            ? undefined
            : shape(row.facts, [
                  'status',
                  'source',
                  'architecture',
                  'contextTokens',
                  'layerCount',
                  'templatePresent',
                  'templateDigest',
                  'dimensions',
                  'unknowns'
              ])
    let dimensions: Record<string, string> | undefined
    if (facts?.dimensions !== undefined) {
        const entries = Object.entries(object(facts.dimensions))
        if (entries.length > 256) throw new Error('Too many model dimensions.')
        dimensions = Object.fromEntries(
            entries.map(([key, value]): [string, string] => [
                boundedText(key, 256, true),
                unsigned(value)
            ])
        )
    }
    const result: InspectModelResult = {
        memoryProfiles:
            row.memoryProfiles === undefined
                ? undefined
                : boundedArray(row.memoryProfiles, 8).map(memoryProfile),
        memoryProfilesUnavailable: optionalFlag(row.memoryProfilesUnavailable),
        engine,
        nodeId: optionalText(row.nodeId, 256),
        modelId: modelId(row.modelId),
        status: choice(row.status, [
            'resolved',
            'missing',
            'ambiguous',
            'busy',
            'unknown',
            'unavailable'
        ]),
        reason: optionalText(row.reason),
        source: optionalText(row.source),
        assurance: optionalText(row.assurance),
        artifacts:
            row.artifacts === undefined
                ? undefined
                : boundedArray(row.artifacts, 1024).map(summary),
        facts: facts && {
            status: boundedText(facts.status, 256, true),
            source: optionalText(facts.source),
            architecture: optionalText(facts.architecture),
            contextTokens: optionalInteger(facts.contextTokens),
            layerCount: optionalInteger(facts.layerCount, 0),
            templatePresent: optionalFlag(facts.templatePresent),
            templateDigest: optionalDigest(facts.templateDigest),
            dimensions,
            unknowns: strings(facts.unknowns)
        },
        options: row.options === undefined ? undefined : boundedArray(row.options, 256).map(option),
        estimate: estimate(row.estimate),
        loadObservation: loadObservation(row.loadObservation),
        configurationKey: optionalDigest(row.configurationKey),
        configurationKeyReason:
            row.configurationKeyReason === undefined
                ? undefined
                : choice(row.configurationKeyReason, [
                      'not-applied',
                      'not-finite',
                      'artifacts-unavailable'
                  ]),
        configurationPools:
            row.configurationPools === undefined
                ? undefined
                : boundedArray(row.configurationPools, 2).map(value =>
                      choice(value, ['host', 'device', 'shared'])
                  )
    }
    const pools = result.configurationPools?.join(',')
    const seenProfiles = new Set<string>()
    for (const [index, profile] of (result.memoryProfiles ?? []).entries()) {
        if (
            seenProfiles.has(profile.configurationKey) ||
            profile.current !== (profile.configurationKey === result.configurationKey) ||
            (profile.current && index !== 0)
        )
            throw new Error('Invalid current memory profile.')
        seenProfiles.add(profile.configurationKey)
        if (profile.current) {
            const fields = [
                profile.hostBytes === undefined ? '' : 'host',
                profile.deviceBytes === undefined ? '' : 'device',
                profile.sharedBytes === undefined ? '' : 'shared'
            ]
                .filter(Boolean)
                .join(',')
            if (fields !== pools)
                throw new Error('Memory profile pools do not match the applied configuration.')
        }
    }
    if (pools !== undefined && !['host', 'host,device', 'shared'].includes(pools))
        throw new Error('Invalid configuration pools.')
    if (result.configurationKey && (result.configurationKeyReason || !pools))
        throw new Error('Invalid configuration identity scope.')
    if (result.status === 'resolved' || result.artifacts?.length)
        orderedSummaryArtifacts(result.artifacts ?? [])
    if (new Set(result.options?.map(item => item.field)).size !== (result.options?.length ?? 0))
        throw new Error('Duplicate model option.')
    if (
        result.estimate.memoryPlan &&
        (result.estimate.memoryPlan.identity.engine !== result.engine ||
            result.estimate.memoryPlan.identity.modelId !== result.modelId)
    )
        throw new Error('Inspection memory plan identity mismatch.')
    return result
}
function memoryProfile(value: JsonValue): MemoryProfile {
    const row = shape(value, [
        'configurationKey',
        'kind',
        'recordedAt',
        'hostBytes',
        'deviceBytes',
        'sharedBytes',
        'current'
    ])
    const recordedAt = boundedText(row.recordedAt, 64, true)
    if (
        !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$/.test(recordedAt) ||
        !Number.isFinite(Date.parse(recordedAt))
    )
        throw new Error('Invalid memory profile timestamp.')
    const profile: MemoryProfile = {
        configurationKey: digest(row.configurationKey),
        kind: choice(row.kind, ['stated']),
        recordedAt,
        current: flag(row.current),
        hostBytes: row.hostBytes === undefined ? undefined : unsigned(row.hostBytes),
        deviceBytes: row.deviceBytes === undefined ? undefined : unsigned(row.deviceBytes),
        sharedBytes: row.sharedBytes === undefined ? undefined : unsigned(row.sharedBytes)
    }
    if (
        (profile.hostBytes === undefined &&
            profile.deviceBytes === undefined &&
            profile.sharedBytes === undefined) ||
        (profile.sharedBytes !== undefined &&
            (profile.hostBytes !== undefined || profile.deviceBytes !== undefined))
    )
        throw new Error('Invalid memory profile pools.')
    return profile
}
function option(value: JsonValue): NonNullable<InspectModelResult['options']>[number] {
    const row = shape(value, [
        'field',
        'label',
        'kind',
        'minimum',
        'maximum',
        'values',
        'default',
        'defaultSource',
        'supported',
        'reason',
        'portable',
        'owner',
        'phase',
        'identityEffect',
        'estimateEffect'
    ])
    const result = {
        field: boundedText(row.field, 128, true),
        label: boundedText(row.label, 256, true),
        kind: choice(row.kind, ['integer', 'number', 'boolean', 'enum', 'string']),
        minimum: decimal(row.minimum),
        maximum: decimal(row.maximum),
        values: strings(row.values, 64),
        default: row.default === undefined ? undefined : structuredClone(row.default),
        defaultSource: optionalText(row.defaultSource),
        supported: flag(row.supported),
        reason: optionalText(row.reason),
        portable: flag(row.portable),
        owner: optionalText(row.owner),
        phase: optionalText(row.phase),
        identityEffect: optionalText(row.identityEffect),
        estimateEffect: optionalText(row.estimateEffect)
    }
    if (
        (result.minimum !== undefined &&
            result.maximum !== undefined &&
            result.minimum > result.maximum) ||
        (!result.supported && !result.reason) ||
        (result.default !== undefined && !result.defaultSource) ||
        result.values?.some(value => !value || Buffer.byteLength(value, 'utf8') > 256)
    )
        throw new Error('Invalid model option descriptor.')
    return result
}
// The broker owns settings only for the engines it fronts with a proxy facade.
function isSettingsEngine(value: JsonValue | undefined): value is EngineSettingsTarget['engine'] {
    return typeof value === 'string' && proxyEngineFromManagerId(value) !== null
}
function isSnapshotPhase(value: JsonValue | undefined): value is EngineSettingsSnapshot['phase'] {
    return isReceiptPhase(value) || value === 'idle'
}
function isReceiptPhase(value: JsonValue | undefined): value is EngineSettingsReceipt['phase'] {
    return value === 'applying' || value === 'succeeded' || value === 'failed'
}
function config(value: JsonValue | undefined): EngineSettingsConfig {
    const row = object(value)
    const modelLoads =
        row.modelLoads === undefined ? undefined : boundedArray(row.modelLoads, 256).map(modelLoad)
    if (new Set(modelLoads?.map(load => load.modelId)).size !== (modelLoads?.length ?? 0))
        throw new Error('Duplicate model load identifier.')
    return {
        serverPort: number(row.serverPort),
        proxyPort: number(row.proxyPort),
        launchText: text(row.launchText),
        ...(row.workingSetLimitMiB == null
            ? {}
            : { workingSetLimitMiB: integer(row.workingSetLimitMiB, 0) }),
        ...(modelLoads === undefined ? {} : { modelLoads }),
        ...(row.serving == null ? {} : { serving: servingConfig(row.serving) })
    }
}

function servingText(value: JsonValue | undefined): string {
    if (typeof value !== 'string') throw new Error('Invalid serving response.')
    return value
}
function requiredText(value: JsonValue | undefined): string {
    const result = servingText(value)
    if (!result.trim()) throw new Error('Invalid serving response.')
    return result
}
function positive(value: JsonValue | undefined): number {
    const result = number(value)
    if (result === 0) throw new Error('Invalid serving bound.')
    return result
}
function digest(value: JsonValue | undefined): string {
    const result = requiredText(value)
    if (!/^[a-fA-F0-9]{64}$/.test(result)) throw new Error('Invalid serving digest.')
    return result
}
function array(value: JsonValue | undefined): JsonValue[] {
    if (!Array.isArray(value)) throw new Error('Invalid serving array.')
    return value
}
function memory(value: JsonValue | undefined): EngineServingConfig['models'][number]['memory'] {
    const row = object(value)
    return { hostRamBytes: positive(row.hostRamBytes), vramBytes: number(row.vramBytes) }
}
function artifact(value: JsonValue): EngineServingConfig['models'][number]['artifacts'][number] {
    const row = object(value)
    const provenance = object(row.provenance)
    const kind = requiredText(provenance.kind)
    if (
        ![
            'hf-lfs-sha256',
            'strata-source-verification',
            'verified-manifest',
            'one-time-sha256',
            'supplied-sha256'
        ].includes(kind)
    )
        throw new Error('Invalid serving provenance.')
    const binding = row.binding == null ? undefined : object(row.binding)
    const mtimeNs = binding ? requiredText(binding.mtimeNs) : undefined
    if (
        mtimeNs !== undefined &&
        (!/^(0|-?[1-9][0-9]*)$/.test(mtimeNs) ||
            BigInt(mtimeNs) < -(2n ** 63n) ||
            BigInt(mtimeNs) > 2n ** 63n - 1n)
    )
        throw new Error('Invalid serving timestamp.')
    return {
        role: requiredText(row.role),
        sha256: digest(row.sha256),
        provenance: {
            kind,
            ...(provenance.reference === undefined
                ? {}
                : { reference: servingText(provenance.reference) })
        },
        ...(binding && mtimeNs !== undefined
            ? {
                  binding: {
                      sourcePath: requiredText(binding.sourcePath),
                      resolvedTarget: requiredText(binding.resolvedTarget),
                      sizeBytes: positive(binding.sizeBytes),
                      mtimeNs,
                      sampleSha256: digest(binding.sampleSha256)
                  }
              }
            : {})
    }
}
function servingConfig(value: JsonValue): EngineServingConfig {
    const row = object(value)
    if (row.version !== 1) throw new Error('Unsupported serving version.')
    const ids = new Set<string>()
    const models = array(row.models).map(value => {
        const model = object(value)
        const modelId = requiredText(model.modelId)
        if (ids.has(modelId)) throw new Error('Duplicate served model ID.')
        ids.add(modelId)
        const artifacts = array(model.artifacts).map(artifact)
        if (!artifacts.length) throw new Error('Missing serving artifacts.')
        return {
            modelId,
            maxOutputTokens: positive(model.maxOutputTokens),
            memory: memory(model.memory),
            artifacts,
            ...(model.expectedNativeSourceDigest === undefined
                ? {}
                : { expectedNativeSourceDigest: digest(model.expectedNativeSourceDigest) })
        }
    })
    if (!models.length) throw new Error('Missing served models.')
    return { version: 1, models }
}
function envelope(value: JsonValue): NonNullable<EngineServingReadback['configured']>[number] {
    const row = object(value)
    return {
        modelId: requiredText(row.modelId),
        perSlotContextTokens: positive(row.perSlotContextTokens),
        totalContextTokens: positive(row.totalContextTokens),
        slots: positive(row.slots),
        maxOutputTokens: positive(row.maxOutputTokens),
        memory: memory(row.memory),
        nativeSourceDigest: digest(row.nativeSourceDigest),
        ...(row.placement === undefined
            ? {}
            : { placement: array(row.placement).map(requiredText) }),
        ...(row.retention === undefined ? {} : { retention: requiredText(row.retention) }),
        ...(row.routerCapacity === undefined
            ? {}
            : { routerCapacity: positive(row.routerCapacity) })
    }
}
function servingReadback(value: JsonValue | undefined): EngineServingReadback | undefined {
    if (value == null) return undefined
    const row = object(value)
    const status = row.status
    if (
        status !== 'unconfigured' &&
        status !== 'unverified' &&
        status !== 'configured' &&
        status !== 'launchValidated' &&
        status !== 'observed' &&
        status !== 'mismatch'
    )
        throw new Error('Invalid serving status.')
    return {
        status,
        ...(row.reason === undefined ? {} : { reason: requiredText(row.reason) }),
        ...(row.configured === undefined
            ? {}
            : { configured: array(row.configured).map(envelope) }),
        ...(row.launchValidated === undefined
            ? {}
            : { launchValidated: array(row.launchValidated).map(envelope) }),
        ...(row.observed === undefined
            ? {}
            : {
                  observed: array(row.observed).map(value => {
                      const observed = object(value)
                      return {
                          modelId: requiredText(observed.modelId),
                          perSlotContextTokens: positive(observed.perSlotContextTokens),
                          slots: positive(observed.slots)
                      }
                  })
              })
    }
}
export function parseEngineSettings(value: JsonValue | undefined): EngineSettingsSnapshot {
    payload(value)
    const row = object(value)
    const engine = row.engine
    const phase = row.phase
    if (!isSettingsEngine(engine) || !isSnapshotPhase(phase) || !text(row.epoch))
        throw new Error('Invalid engine settings response.')
    return {
        nodeId: text(row.nodeId),
        engine,
        phase,
        epoch: text(row.epoch),
        sequence: number(row.sequence),
        revision: number(row.revision),
        appliedRevision: number(row.appliedRevision),
        settings: config(row.settings),
        workingSetLimitStatus:
            row.workingSetLimitStatus === undefined
                ? undefined
                : choice(row.workingSetLimitStatus, [
                      'applied',
                      'not-running',
                      'unenforced-adopted'
                  ]),
        serving: servingReadback(row.serving),
        effectiveServerPort: number(row.effectiveServerPort),
        effectiveProxyPort: number(row.effectiveProxyPort),
        running: flag(row.running),
        adopted: flag(row.adopted),
        editable: flag(row.editable),
        reason: text(row.reason),
        // Preserve the target's argument format so incompatible devices stay read-only.
        format: text(row.format),
        error: text(row.error),
        requestId: text(row.requestId),
        coupledStage: optionalText(row.coupledStage, 256),
        policyRevision: optionalRevision(row.policyRevision),
        unresolvedError: optionalText(row.unresolvedError),
        effectiveSettings:
            row.effectiveSettings === undefined ? undefined : config(row.effectiveSettings)
    }
}
export function parseEngineSettingsPreview(value: JsonValue | undefined): EngineSettingsPreview {
    payload(value)
    const row = object(value)
    const errors: Record<string, string> = {}
    if (row.errors)
        for (const [key, value] of Object.entries(object(row.errors))) errors[key] = text(value)
    const conflict = row.conflict ? object(row.conflict) : undefined
    const requiredArtifacts =
        row.requiredArtifacts === undefined
            ? undefined
            : boundedArray(row.requiredArtifacts, 1024).map(summary)
    if (requiredArtifacts?.length) orderedSummaryArtifacts(requiredArtifacts)
    return {
        settings: config(row.settings),
        serving: servingReadback(row.serving),
        revision: number(row.revision),
        errors,
        restart: flag(row.restart),
        rebind: flag(row.rebind),
        policyRevision: optionalRevision(row.policyRevision),
        previewDigest: optionalDigest(row.previewDigest),
        diff:
            row.diff === undefined
                ? undefined
                : boundedArray(row.diff, 256).map(value => {
                      const item = shape(value, ['field', 'before', 'after'])
                      return {
                          field: boundedText(item.field, 256, true),
                          before:
                              item.before === undefined ? undefined : structuredClone(item.before),
                          after: item.after === undefined ? undefined : structuredClone(item.after)
                      }
                  }),
        requiredArtifacts,
        estimate: row.estimate === undefined ? undefined : estimate(row.estimate),
        impact: row.impact === undefined ? undefined : impact(row.impact),
        auto: row.auto === undefined ? undefined : autoProposal(row.auto),
        ...(conflict
            ? {
                  conflict: {
                      serverPort: number(conflict.serverPort),
                      launchPort: number(conflict.launchPort)
                  }
              }
            : {})
    }
}
export function parseEngineSettingsReceipt(value: JsonValue | undefined): EngineSettingsReceipt {
    payload(value)
    const row = object(value)
    const phase = row.phase
    if (!isReceiptPhase(phase)) throw new Error('Invalid operation receipt.')
    return { revision: number(row.revision), phase }
}

function impact(value: JsonValue): ModelImpact {
    const row = shape(value, [
        'restartScope',
        'affectedModels',
        'drainTimeoutSeconds',
        'mayInterruptActive',
        'staysOff',
        'activeWorkPolicy',
        'unknowns'
    ])
    return {
        // These labels remain service-owned while owner choices are unresolved.
        restartScope: boundedText(row.restartScope, 256, true),
        affectedModels:
            row.affectedModels === undefined
                ? undefined
                : boundedArray(row.affectedModels, 256).map(modelId),
        drainTimeoutSeconds: optionalInteger(row.drainTimeoutSeconds, 0),
        mayInterruptActive: optionalFlag(row.mayInterruptActive),
        staysOff: optionalFlag(row.staysOff),
        activeWorkPolicy: optionalText(row.activeWorkPolicy),
        unknowns: strings(row.unknowns)
    }
}

function autoProposal(value: JsonValue): AutoProposal {
    const row = shape(value, [
        'status',
        'requested',
        'resolved',
        'estimate',
        'memorySource',
        'memoryTimestamp',
        'resolverRevision',
        'reviewedInputId',
        'warnings'
    ])
    const result = {
        status: boundedText(row.status, 256, true),
        requested: requested(row.requested),
        resolved: row.resolved === undefined ? undefined : resolved(row.resolved),
        estimate: estimate(row.estimate),
        memorySource: optionalText(row.memorySource),
        memoryTimestamp: optionalText(row.memoryTimestamp),
        resolverRevision: optionalText(row.resolverRevision, 256),
        reviewedInputId: optionalText(row.reviewedInputId, 256),
        warnings: strings(row.warnings)
    }
    validateResolution(result.requested, result.resolved)
    if (
        !!result.resolverRevision !== !!result.reviewedInputId ||
        (result.resolved &&
            (result.resolverRevision !== result.resolved.resolverRevision ||
                result.reviewedInputId !== result.resolved.reviewedInputId))
    )
        throw new Error('Inconsistent Auto review identity.')
    return result
}
