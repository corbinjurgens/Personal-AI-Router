// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelDownloadEntry, ModelDownloadSnapshot } from '@/shared/types/model-download'
import type { JsonObject, JsonValue } from './json-rpc-client'

function object(value: JsonValue | undefined): JsonObject | undefined {
    return value !== null && typeof value === 'object' && !Array.isArray(value) ? value : undefined
}

function exactFields(value: JsonObject, required: string[], optional: string[] = []): boolean {
    return (
        required.every(key => Object.hasOwn(value, key)) &&
        Object.keys(value).every(key => required.includes(key) || optional.includes(key))
    )
}

function validId(value: JsonValue | undefined): value is string {
    return typeof value === 'string' && /^[a-f0-9]{32}$/.test(value)
}

function reason(value: JsonValue | undefined): ModelDownloadEntry['reason'] {
    switch (value) {
        case 'busy':
        case 'capacity':
        case 'download_failed':
        case 'interrupted':
        case 'progress_timeout':
        case 'acceptance_unknown':
        case 'cleanup_unknown':
        case 'persistence_error':
        case 'unsupported':
            return value
        default:
            return undefined
    }
}

function validModel(value: string): boolean {
    return (
        value.length > 0 &&
        new TextEncoder().encode(value).length <= 512 &&
        Array.from(value).every(character => {
            const code = character.codePointAt(0) ?? -1
            return code >= 0x20 && (code < 0x7f || code > 0x9f) && (code < 0xd800 || code > 0xdfff)
        })
    )
}

function parseEntry(value: JsonValue): ModelDownloadEntry | undefined {
    const row = object(value)
    if (
        !row ||
        !exactFields(
            row,
            ['id', 'engine', 'model', 'phase', 'blocksReadiness'],
            ['percent', 'reason', 'persistence']
        )
    )
        return undefined
    if (
        !validId(row.id) ||
        typeof row.engine !== 'string' ||
        row.engine.length > 40 ||
        !/^[a-zA-Z0-9._-]+$/.test(row.engine) ||
        row.engine === '.' ||
        row.engine === '..' ||
        typeof row.model !== 'string' ||
        !validModel(row.model) ||
        typeof row.blocksReadiness !== 'boolean'
    )
        return undefined
    const phase = row.phase
    if (
        phase !== 'queued' &&
        phase !== 'in_progress' &&
        phase !== 'failed' &&
        phase !== 'complete' &&
        phase !== 'unknown'
    )
        return undefined
    if ((phase === 'queued' || phase === 'in_progress') && !row.blocksReadiness) return undefined
    if (phase === 'complete' && row.blocksReadiness) return undefined
    if (
        Object.hasOwn(row, 'percent') &&
        (typeof row.percent !== 'number' ||
            !Number.isInteger(row.percent) ||
            row.percent < 0 ||
            row.percent > 100)
    )
        return undefined
    const parsedReason = reason(row.reason)
    if (
        Object.hasOwn(row, 'reason') &&
        (!parsedReason || (phase !== 'failed' && phase !== 'unknown'))
    )
        return undefined
    if (
        Object.hasOwn(row, 'persistence') &&
        (row.persistence !== 'uncertain' || phase === 'queued' || phase === 'in_progress')
    )
        return undefined
    return {
        id: row.id,
        engine: row.engine,
        model: row.model,
        phase,
        blocksReadiness: row.blocksReadiness,
        ...(typeof row.percent === 'number' ? { percent: row.percent } : {}),
        ...(parsedReason ? { reason: parsedReason } : {}),
        ...(row.persistence === 'uncertain' ? { persistence: row.persistence } : {})
    }
}

/** Invalid extensions remain unknown, never a successful empty snapshot. */
export function parseModelDownloadSnapshot(
    value: JsonValue | undefined
): ModelDownloadSnapshot | undefined {
    if (value === undefined || new TextEncoder().encode(JSON.stringify(value)).length > 256 * 1024)
        return undefined
    const row = object(value)
    if (
        !row ||
        !exactFields(row, ['version', 'sourceId', 'revision', 'recoveryRequired', 'entries']) ||
        row.version !== 1 ||
        !validId(row.sourceId) ||
        typeof row.revision !== 'string' ||
        !/^(0|[1-9][0-9]{0,19})$/.test(row.revision) ||
        BigInt(row.revision) > 18446744073709551615n ||
        typeof row.recoveryRequired !== 'boolean' ||
        !Array.isArray(row.entries) ||
        row.entries.length > 64
    )
        return undefined
    const entries: ModelDownloadEntry[] = []
    const ids = new Set<string>()
    const keys = new Set<string>()
    for (const value of row.entries) {
        const entry = parseEntry(value)
        if (!entry) return undefined
        const key = JSON.stringify([entry.engine, entry.model])
        if (ids.has(entry.id) || keys.has(key)) return undefined
        ids.add(entry.id)
        keys.add(key)
        entries.push(entry)
    }
    if (
        entries.filter(entry => entry.blocksReadiness).length > 32 ||
        entries.filter(entry => !entry.blocksReadiness).length > 32
    )
        return undefined
    return {
        version: 1,
        sourceId: row.sourceId,
        revision: row.revision,
        recoveryRequired: row.recoveryRequired,
        entries
    }
}

function stringMap(value: JsonValue | undefined): Record<string, string[]> | undefined {
    if (value === undefined || value === null) return undefined
    const row = object(value)
    if (!row) throw new Error('engine:models returned malformed attribution')
    const entries: Array<[string, string[]]> = []
    for (const [engine, names] of Object.entries(row)) {
        if (
            !Array.isArray(names) ||
            !names.every((name): name is string => typeof name === 'string')
        ) {
            throw new Error('engine:models returned malformed inventory')
        }
        entries.push([engine, names])
    }
    return Object.fromEntries(entries)
}

/** Canonical inventory, including engines whose current query failed (missing key). */
export interface ModelsResult {
    modelsByEngine?: Record<string, string[]>
    loadedByEngine?: Record<string, string[]>
    downloadState?: ModelDownloadSnapshot
    downloadStateCurrent?: boolean
}

export function parseModelsResult(value: JsonValue | undefined): ModelsResult {
    const row = object(value)
    if (!row || !Array.isArray(row.models) || !row.models.every(name => typeof name === 'string')) {
        throw new Error('engine:models returned malformed inventory')
    }
    const downloadState = parseModelDownloadSnapshot(row.downloadState)
    return {
        modelsByEngine: stringMap(row.modelsByEngine),
        loadedByEngine: stringMap(row.loadedByEngine),
        downloadState,
        downloadStateCurrent: downloadState
            ? row.downloadStateCurrent === undefined || row.downloadStateCurrent === true
            : row.downloadState === undefined
              ? undefined
              : false
    }
}

/** One target's source/revision fence. Failed observations retain last-good facts. */
export class ModelDownloadObservation {
    snapshot?: ModelDownloadSnapshot
    current?: boolean
    sequence = 0
    private retiredSources = new Set<string>()

    accept(
        snapshot: ModelDownloadSnapshot | undefined,
        current: boolean | undefined,
        baselineSequence?: number
    ): boolean {
        const previous = this.snapshot
        if (
            baselineSequence !== undefined &&
            baselineSequence !== this.sequence &&
            (!snapshot ||
                !previous ||
                snapshot.sourceId !== previous.sourceId ||
                BigInt(snapshot.revision) <= BigInt(previous.revision))
        )
            return false
        if (!snapshot) {
            this.current = current
            this.sequence++
            return true
        }
        if (this.retiredSources.has(snapshot.sourceId)) return false
        if (
            previous?.sourceId === snapshot.sourceId &&
            BigInt(snapshot.revision) < BigInt(previous.revision)
        )
            return false
        if (
            previous?.sourceId === snapshot.sourceId &&
            previous.revision === snapshot.revision &&
            JSON.stringify(previous) !== JSON.stringify(snapshot)
        ) {
            this.current = false
            this.sequence++
            return true
        }
        if (previous && previous.sourceId !== snapshot.sourceId) {
            if (current !== true) return false
            this.retiredSources.add(previous.sourceId)
        }
        this.snapshot = snapshot
        this.current = current
        this.sequence++
        return true
    }
}
