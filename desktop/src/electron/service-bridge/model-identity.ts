// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelIdentity } from '@/shared/types/engines'
import type { JsonObject, JsonValue } from './json-rpc-client'

function object(value: JsonValue | undefined): JsonObject | undefined {
    return value !== null && typeof value === 'object' && !Array.isArray(value) ? value : undefined
}

/** Go's byte bound, valid UTF-8 and unicode.IsControl checks. */
function text(value: JsonValue | undefined): value is string {
    return (
        typeof value === 'string' &&
        value.length <= 256 &&
        new TextEncoder().encode(value).length <= 256 &&
        Array.from(value).every(character => {
            const code = character.codePointAt(0) ?? -1
            return code >= 0x20 && (code < 0x7f || code > 0x9f) && (code < 0xd800 || code > 0xdfff)
        })
    )
}

function nonnegativeInteger(value: JsonValue | undefined): value is number {
    return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}

function parseIdentity(value: JsonValue): ModelIdentity | undefined {
    const row = object(value)
    if (!row) return undefined
    const repository = row.repository ?? ''
    const revision = row.revision ?? ''
    const quantization = row.quantization ?? ''
    const quantizationSource = row.quantizationSource ?? ''
    const generalName = row.generalName ?? ''
    const sizeLabel = row.sizeLabel ?? ''
    const architecture = row.architecture ?? ''
    const contentId = row.contentId ?? ''
    const strength = row.strength
    if (
        !text(repository) ||
        !text(revision) ||
        !text(quantization) ||
        !text(quantizationSource) ||
        !text(generalName) ||
        !text(sizeLabel) ||
        !text(architecture) ||
        !text(contentId) ||
        !text(strength)
    )
        return undefined
    if (repository && !/^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(repository))
        return undefined
    if (revision && !/^[0-9a-f]{40}$/.test(revision)) return undefined
    if (contentId && !/^[0-9a-f]{64}$/.test(contentId)) return undefined
    if (
        quantizationSource !== '' &&
        quantizationSource !== 'hf-tag' &&
        quantizationSource !== 'file-name' &&
        quantizationSource !== 'gguf-file-type'
    )
        return undefined
    if (quantizationSource && !quantization) return undefined
    if (
        (strength !== 'declared' && strength !== 'verified') ||
        strength !== (contentId ? 'verified' : 'declared')
    )
        return undefined
    const trainedContextTokens = row.trainedContextTokens ?? 0
    const configuredContextTokens = row.configuredContextTokens ?? 0
    const totalBytes = row.totalBytes ?? 0
    if (
        !nonnegativeInteger(trainedContextTokens) ||
        !nonnegativeInteger(configuredContextTokens) ||
        !nonnegativeInteger(totalBytes)
    )
        return undefined
    const fileValues = row.files ?? []
    if (!Array.isArray(fileValues) || fileValues.length > 64) return undefined
    const files: NonNullable<ModelIdentity['files']> = []
    for (const value of fileValues) {
        const file = object(value)
        if (!file) return undefined
        const role = file.role
        const name = file.name ?? ''
        const sizeBytes = file.sizeBytes ?? 0
        const sha256 = file.sha256 ?? ''
        if (
            (role !== 'model' && role !== 'draft' && role !== 'companion') ||
            !text(name) ||
            /[/\\]/.test(name) ||
            !nonnegativeInteger(sizeBytes) ||
            !text(sha256) ||
            (sha256 !== '' && !/^[0-9a-f]{64}$/.test(sha256))
        )
            return undefined
        files.push({ role, name, sizeBytes, ...(sha256 ? { sha256 } : {}) })
    }
    return {
        ...(repository ? { repository } : {}),
        ...(revision ? { revision } : {}),
        ...(quantization ? { quantization } : {}),
        ...(quantizationSource ? { quantizationSource } : {}),
        ...(generalName ? { generalName } : {}),
        ...(sizeLabel ? { sizeLabel } : {}),
        ...(architecture ? { architecture } : {}),
        ...(trainedContextTokens ? { trainedContextTokens } : {}),
        ...(configuredContextTokens ? { configuredContextTokens } : {}),
        ...(files.length ? { files } : {}),
        ...(totalBytes ? { totalBytes } : {}),
        ...(contentId ? { contentId } : {}),
        strength
    }
}

/** Optional metadata is validated per model so malformed siblings never lose inventory. */
export function parseModelIdentities(
    value: JsonValue | undefined
): Record<string, Record<string, ModelIdentity>> | undefined {
    const engines = object(value)
    if (!engines) return undefined
    const parsed: Array<[string, Record<string, ModelIdentity>]> = []
    for (const [engine, value] of Object.entries(engines)) {
        const models = object(value)
        if (!models) continue
        const entries: Array<[string, ModelIdentity]> = []
        for (const [model, value] of Object.entries(models)) {
            const identity = parseIdentity(value)
            if (identity) entries.push([model, identity])
        }
        parsed.push([engine, Object.fromEntries(entries)])
    }
    return Object.fromEntries(parsed)
}
