// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    EngineDisplayNames,
    EngineManagerNames,
    EngineTypes,
    ModelExpiries
} from '@/shared/constants/engines'
import {
    BuiltinEngineType,
    DescriptorEngineType,
    EngineDescriptor,
    EngineManagerName,
    EngineModels,
    EngineStatusData,
    EngineType,
    ModelExpiry
} from '@/shared/types/engines'

const BUILTIN_ENGINE_TYPE_SET: ReadonlySet<string> = new Set(EngineTypes)
const BUILTIN_MANAGER_NAME_SET: ReadonlySet<string> = new Set(Object.values(EngineManagerNames))

/** Runtime type guard for the engines PAIR ships. */
export function isBuiltinEngineType(v: string | undefined): v is BuiltinEngineType {
    return typeof v === 'string' && BUILTIN_ENGINE_TYPE_SET.has(v)
}

/**
 * engine-manager's engine ID syntax (`engines.ValidID`): a filesystem path
 * component, since it names the engine's install directory.
 */
const ENGINE_ID_RE = /^[a-zA-Z0-9._-]+$/

/** Whether a string is usable as an engine-manager engine ID or family name. */
export function isValidEngineId(v: string): boolean {
    return ENGINE_ID_RE.test(v) && v !== '.' && v !== '..'
}

/**
 * Runtime type guard for a descriptor-known engine identifier. Refuses both
 * spellings of the built-in engines, so a built-in engine has exactly one
 * identity inside PAIR.
 */
function isDescriptorEngineType(v: string | undefined): v is DescriptorEngineType {
    return (
        typeof v === 'string' &&
        isValidEngineId(v) &&
        !BUILTIN_ENGINE_TYPE_SET.has(v) &&
        !BUILTIN_MANAGER_NAME_SET.has(v)
    )
}

/**
 * Runtime type guard — narrows a `string` (typically arriving from service
 * bridge payloads or persisted JSON) into `EngineType`. Apply at every
 * wire/persistence boundary; everything inside the service then keeps strict
 * typing.
 */
export function isEngineType(v: string | undefined): v is EngineType {
    return isBuiltinEngineType(v) || isDescriptorEngineType(v)
}

/** Spell an `EngineType` the way `nvpair-engine-manager` does on the wire. */
export function engineManagerName(engine: BuiltinEngineType): EngineManagerName
export function engineManagerName(engine: EngineType): string
export function engineManagerName(engine: EngineType): string {
    return isBuiltinEngineType(engine) ? EngineManagerNames[engine] : engine
}

const ENGINE_TYPE_BY_MANAGER_NAME: ReadonlyMap<string, BuiltinEngineType> = new Map(
    EngineTypes.map(engine => [EngineManagerNames[engine], engine])
)
/**
 * Narrow an engine id arriving from `nvpair-engine-manager` into an
 * `EngineType`, or null when it is not a valid engine-manager ID. A built-in
 * engine maps to its desktop spelling; any other valid ID is a descriptor
 * engine and keeps engine-manager's spelling.
 */
export function engineTypeFromManagerName(name: string): EngineType | null {
    const builtin = ENGINE_TYPE_BY_MANAGER_NAME.get(name)
    if (builtin) return builtin
    return isDescriptorEngineType(name) ? name : null
}

/**
 * The built-in engine whose coded adapters (icon, model-name matching,
 * catalogue search) an engine shares: itself for a built-in engine, its
 * descriptor's family for a variant such as a llama.cpp fork, and null for an
 * engine of an unknown family or with no descriptor.
 */
export function engineAdapterType(
    engine: EngineType,
    descriptor: EngineDescriptor | undefined
): BuiltinEngineType | null {
    if (isBuiltinEngineType(engine)) return engine
    if (!descriptor) return null
    return ENGINE_TYPE_BY_MANAGER_NAME.get(descriptor.family) ?? null
}

/**
 * The user-facing name of an engine: the built-in label, the descriptor's
 * label, or the engine ID when this node holds no descriptor for it.
 */
export function engineLabel(engine: EngineType, descriptor: EngineDescriptor | undefined): string {
    if (isBuiltinEngineType(engine)) return EngineDisplayNames[engine]
    return descriptor?.label ?? engine
}

const MODEL_EXPIRY_SET: ReadonlySet<string> = new Set(ModelExpiries)
/** Runtime type guard — validates a string is a known ModelExpiry. */
export function isModelExpiry(v: string | undefined): v is ModelExpiry {
    return v !== undefined && MODEL_EXPIRY_SET.has(v)
}

/** Shallow placeholder when the service has not yet reported this node/engine pair. */
export function emptyEngineStatus(nodeId: string, engineType: EngineType): EngineStatusData {
    return {
        engineType,
        nodeId,
        processStatus: 'initializing',
        enginePort: null,
        proxyPort: null
    }
}

/** Shallow placeholder when no model list has been pushed for this node/engine yet. */
export function emptyEngineModels(nodeId: string, engineType: EngineType): EngineModels {
    return { nodeId, engineType, models: [] }
}
