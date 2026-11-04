// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { JsonValue } from '@/shared/types/json'

/** Bounded reported source identity; diagnostic only, never an artifact attestation. */
export function parseServiceBuildId(value: JsonValue | undefined): string | undefined {
    if (typeof value !== 'string' || value.length > 74) return undefined
    const match = /^git:(?:[0-9a-f]{40}|[0-9a-f]{64})(?:\+dirty)?$/.exec(value)
    return match?.[0] === value ? value : undefined
}
