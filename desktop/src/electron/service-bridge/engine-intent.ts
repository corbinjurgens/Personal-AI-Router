// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EngineIntentSnapshot } from '@/shared/types/engine-api'
import type { JsonObject, JsonValue } from './json-rpc-client'

function object(value: JsonValue | undefined): JsonObject {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        throw new Error('Invalid engine intent snapshot')
    }
    return value
}

export function parseEngineIntent(value: JsonValue | undefined): EngineIntentSnapshot {
    const row = object(value)
    const enabled = object(row.enabledByEngine)
    const origins = object(row.offByEngine)
    const enabledByEngine: EngineIntentSnapshot['enabledByEngine'] = {}
    const offByEngine: EngineIntentSnapshot['offByEngine'] = {}
    for (const [engine, enabledValue] of Object.entries(enabled)) {
        if (typeof enabledValue !== 'boolean') throw new Error('Invalid engine intent snapshot')
        Object.defineProperty(enabledByEngine, engine, { value: enabledValue, enumerable: true })
    }
    for (const [engine, origin] of Object.entries(origins)) {
        if (
            (origin !== 'local' && origin !== 'peer') ||
            !Object.hasOwn(enabled, engine) ||
            enabled[engine] !== false
        ) {
            throw new Error('Invalid engine intent snapshot')
        }
        Object.defineProperty(offByEngine, engine, { value: origin, enumerable: true })
    }
    return { enabledByEngine, offByEngine }
}
