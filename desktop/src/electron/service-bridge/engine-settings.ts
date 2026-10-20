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
    return {
        serverPort: number(row.serverPort),
        proxyPort: number(row.proxyPort),
        launchText: text(row.launchText),
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
        requestId: text(row.requestId)
    }
}
export function parseEngineSettingsPreview(value: JsonValue | undefined): EngineSettingsPreview {
    const row = object(value)
    const errors: Record<string, string> = {}
    if (row.errors)
        for (const [key, value] of Object.entries(object(row.errors))) errors[key] = text(value)
    const conflict = row.conflict ? object(row.conflict) : undefined
    return {
        settings: config(row.settings),
        serving: servingReadback(row.serving),
        revision: number(row.revision),
        errors,
        restart: flag(row.restart),
        rebind: flag(row.rebind),
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
    const row = object(value)
    const phase = row.phase
    if (!isReceiptPhase(phase)) throw new Error('Invalid operation receipt.')
    return { revision: number(row.revision), phase }
}
