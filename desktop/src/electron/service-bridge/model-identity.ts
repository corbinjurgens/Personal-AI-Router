// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelIdentity } from '@/shared/types/engines'
import type { JsonObject, JsonValue } from './json-rpc-client'
import { parseModelIdentity } from '@/shared/utils/model-identity'

function object(value: JsonValue | undefined): JsonObject | undefined {
    return value !== null && typeof value === 'object' && !Array.isArray(value) ? value : undefined
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
            const identity = parseModelIdentity(value)
            if (identity) entries.push([model, identity])
        }
        parsed.push([engine, Object.fromEntries(entries)])
    }
    return Object.fromEntries(parsed)
}
