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
    ModelImpact,
    ModelLoad,
    RequestedLoad,
    ResolvedLoad
} from '@/shared/types/model-config'
import { isValidRoutingEngineId } from '@/shared/utils/engines'
import type { JsonObject, JsonValue } from './json-rpc-client'
import { proxyEngineFromManagerId } from './proxy-engines'

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
        'reviewedInputId'
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
        reviewedInputId: optionalText(row.reviewedInputId, 256)
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
        (!actual.resolverRevision || !actual.reviewedInputId)
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
function estimate(value: JsonValue | undefined): Estimate {
    const row = shape(value, [
        'status',
        'source',
        'buildId',
        'timestamp',
        'memorySource',
        'memoryTimestamp',
        'hostBytes',
        'deviceBytes',
        'domains',
        'components',
        'assumptions',
        'unknowns'
    ])
    const result: Estimate = {
        status: choice(row.status, ['estimated', 'partial', 'unknown', 'unavailable']),
        source: optionalText(row.source),
        buildId: optionalText(row.buildId),
        timestamp: optionalText(row.timestamp),
        memorySource: optionalText(row.memorySource),
        memoryTimestamp: optionalText(row.memoryTimestamp),
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
                          'requiredBytes'
                      ])
                      return {
                          id: boundedText(item.id, 256, true),
                          kind: choice(item.kind, ['host', 'device', 'shared']),
                          deviceId: optionalText(item.deviceId),
                          capacityBytes: optionalUnsigned(item.capacityBytes),
                          freeBytes: optionalUnsigned(item.freeBytes),
                          requiredBytes: optionalUnsigned(item.requiredBytes)
                      }
                  }),
        components:
            row.components === undefined
                ? undefined
                : boundedArray(row.components, 256).map(value => {
                      const item = shape(value, ['name', 'domainId', 'status', 'bytes', 'reason'])
                      return {
                          name: boundedText(item.name, 256, true),
                          domainId: optionalText(item.domainId),
                          status: choice(item.status, [
                              'estimated',
                              'partial',
                              'unknown',
                              'excluded'
                          ]),
                          bytes: optionalUnsigned(item.bytes),
                          reason: optionalText(item.reason)
                      }
                  }),
        assumptions: strings(row.assumptions),
        unknowns: strings(row.unknowns)
    }
    const domains = new Set(result.domains?.map(item => item.id))
    if (
        domains.size !== (result.domains?.length ?? 0) ||
        result.components?.some(item => item.domainId && !domains.has(item.domainId))
    )
        throw new Error('Invalid estimate domain reference.')
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
        'estimate'
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
        estimate: estimate(row.estimate)
    }
    if (result.status === 'resolved' || result.artifacts?.length)
        orderedSummaryArtifacts(result.artifacts ?? [])
    if (new Set(result.options?.map(item => item.field)).size !== (result.options?.length ?? 0))
        throw new Error('Duplicate model option.')
    return result
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
