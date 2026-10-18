// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    ENGINE_DESCRIPTOR_VERSION,
    ENGINE_OPERATION_MAX_RESPONSE_TIMEOUT_S,
    EngineDescriptorModelNamings,
    EngineDescriptorOperations,
    EngineDescriptorProtocols
} from '@/shared/constants/engines'
import type { EngineDescriptor, EngineDescriptorOperation } from '@/shared/types/engines'
import { engineTypeFromManagerName, isValidEngineId } from '@/shared/utils/engines'
import type { JsonObject, JsonValue } from './json-rpc-client'

function objectValue(value: JsonValue | undefined): JsonObject | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null
    return value
}

function stringValue(value: JsonValue | undefined): string {
    return typeof value === 'string' ? value : ''
}

function stringList(value: JsonValue | undefined): string[] | null {
    if (!Array.isArray(value)) return null
    const out: string[] = []
    for (const entry of value) {
        if (typeof entry !== 'string') return null
        out.push(entry)
    }
    return out
}

const PROTOCOLS: ReadonlySet<string> = new Set(EngineDescriptorProtocols)
const MODEL_NAMINGS: ReadonlySet<string> = new Set(EngineDescriptorModelNamings)

function isDescriptorOperation(value: string): value is EngineDescriptorOperation {
    return EngineDescriptorOperations.some(operation => operation === value)
}

function parseOperations(value: JsonValue | undefined): EngineDescriptor['operations'] | null {
    const raw = objectValue(value)
    if (!raw) return null
    const operations: EngineDescriptor['operations'] = {}
    for (const [name, capability] of Object.entries(raw)) {
        if (!isDescriptorOperation(name)) return null
        const fields = objectValue(capability)
        if (!fields) return null
        const timeout = fields.responseTimeoutS ?? 0
        if (
            typeof timeout !== 'number' ||
            !Number.isInteger(timeout) ||
            timeout < 0 ||
            timeout > ENGINE_OPERATION_MAX_RESPONSE_TIMEOUT_S
        ) {
            return null
        }
        operations[name] = { responseTimeoutS: timeout }
    }
    return operations
}

/**
 * Validate one `engine:descriptors` entry the way `engines.Descriptor.Validate`
 * does, and project it into the desktop's shape. A descriptor that fails is
 * dropped rather than guessed at: an unknown version, protocol or operation is
 * something this build cannot honor, and the engine then renders with the safe
 * defaults of an engine without a descriptor.
 */
function parseEngineDescriptor(value: JsonValue): EngineDescriptor | null {
    const raw = objectValue(value)
    if (!raw || raw.version !== ENGINE_DESCRIPTOR_VERSION) return null
    const engineType = engineTypeFromManagerName(stringValue(raw.engine))
    const family = stringValue(raw.family)
    const label = stringValue(raw.label).trim()
    const protocols = stringList(raw.protocols ?? [])
    const modelNaming = stringValue(raw.modelNaming)
    const modelStore = stringValue(raw.modelStore)
    const catalog = stringValue(raw.catalog)
    const operations = parseOperations(raw.operations ?? {})
    if (
        !engineType ||
        !isValidEngineId(family) ||
        label === '' ||
        typeof raw.available !== 'boolean' ||
        !protocols ||
        protocols.some(protocol => !PROTOCOLS.has(protocol)) ||
        new Set(protocols).size !== protocols.length ||
        !MODEL_NAMINGS.has(modelNaming) ||
        (modelStore !== '' && !isValidEngineId(modelStore)) ||
        (catalog !== '' && !isValidEngineId(catalog)) ||
        !operations
    ) {
        return null
    }
    return {
        engineType,
        family,
        label,
        available: raw.available,
        protocols,
        modelNaming,
        modelStore,
        catalog,
        operations
    }
}

/** Parse an `engine:descriptors` reply (`{ engines: [Descriptor] }`), keeping the valid entries. */
export function parseEngineDescriptors(result: JsonValue | undefined): EngineDescriptor[] {
    const list = objectValue(result)?.engines
    if (!Array.isArray(list)) return []
    const descriptors: EngineDescriptor[] = []
    for (const entry of list) {
        const descriptor = parseEngineDescriptor(entry)
        if (descriptor) descriptors.push(descriptor)
    }
    return descriptors
}
