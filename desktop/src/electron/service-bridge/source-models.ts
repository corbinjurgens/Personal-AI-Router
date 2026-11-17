// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    ImportModelResponse,
    SourceModel,
    SourceModelsResponse
} from '@/shared/types/engine-api'
import type { JsonObject, JsonValue } from '@/electron/service-bridge/json-rpc-client'

function isObject(v: JsonValue | undefined): v is JsonObject {
    return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function isSource(v: JsonValue | undefined): v is SourceModel['source'] {
    return v === 'ollama' || v === 'lmstudio' || v === 'local' || v === 'copy'
}

function toSourceModel(raw: JsonValue): SourceModel | null {
    if (!isObject(raw) || typeof raw.id !== 'string' || !raw.id || !isSource(raw.source))
        return null
    return {
        id: raw.id,
        source: raw.source,
        name: typeof raw.name === 'string' ? raw.name : raw.id,
        sizeBytes: typeof raw.sizeBytes === 'number' ? raw.sizeBytes : 0,
        shards: typeof raw.shards === 'number' && raw.shards > 0 ? raw.shards : 1,
        hasProjector: raw.hasProjector === true
    }
}

/** The engine manager's `engine:source-models` reply, rows it cannot read dropped. */
export function parseSourceModels(value: JsonValue | undefined): SourceModelsResponse {
    if (!isObject(value) || !Array.isArray(value.models))
        throw new Error('Source model listing was malformed.')
    const models = value.models.map(toSourceModel).filter((m): m is SourceModel => m !== null)
    return typeof value.error === 'string' && value.error
        ? { models, error: value.error }
        : { models }
}

/** The engine manager's `engine:link-model` reply. */
export function parseLinkedModel(value: JsonValue | undefined): { id: string } {
    if (!isObject(value) || typeof value.id !== 'string' || !value.id)
        throw new Error('Linked model reply was malformed.')
    return { id: value.id }
}

/** The engine manager's `engine:import-model` reply. */
export function parseImportedModel(value: JsonValue | undefined): ImportModelResponse {
    if (
        !isObject(value) ||
        typeof value.id !== 'string' ||
        typeof value.layout !== 'string' ||
        typeof value.exportId !== 'string' ||
        (value.durability !== 'confirmed' && value.durability !== 'uncertain')
    )
        throw new Error('Import reply was malformed.')
    return {
        id: value.id,
        layout: value.layout,
        exportId: value.exportId,
        durability: value.durability
    }
}
