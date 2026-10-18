// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { JsonValue } from '@/shared/types/json'
import type {
    NodePolicyAdvertisement,
    NodeRole,
    NodeRoles,
    PauseOrigin,
    RoleTier
} from '@/shared/types/node-roles'

/** The tiers from weakest to strongest (`nodepolicy.TierOrder`). */
export const ROLE_TIERS: readonly RoleTier[] = ['weak', 'medium', 'strong']

function objectValue(value: JsonValue | undefined): { [key: string]: JsonValue } | null {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return null
    return value
}

function roleValue(value: JsonValue | undefined): NodeRole | null {
    const role = objectValue(value)
    if (!role || typeof role.engine !== 'string' || typeof role.model !== 'string') return null
    if (role.engine === '' || role.model === '') return null
    return { engine: role.engine, model: role.model }
}

/**
 * Read a `roles` map from a policy document or an advertisement. Only the known
 * tiers are kept; an incomplete role is unset. An advertised role's bounds are
 * not needed by the desktop and are dropped.
 */
export function parseNodeRoles(value: JsonValue | undefined): NodeRoles {
    const map = objectValue(value)
    const roles: NodeRoles = {}
    if (!map) return roles
    for (const tier of ROLE_TIERS) {
        const role = roleValue(map[tier])
        if (role) roles[tier] = role
    }
    return roles
}

/** A `pausedBy` value, or undefined when absent or unknown. */
export function pauseOriginValue(value: JsonValue | undefined): PauseOrigin | undefined {
    return value === 'local' || value === 'peer' ? value : undefined
}

/** Read a discovery node's `policy` advertisement; undefined when the node sent none. */
export function parsePolicyAdvertisement(
    value: JsonValue | undefined
): NodePolicyAdvertisement | undefined {
    const body = objectValue(value)
    if (!body) return undefined
    const pausedBy = pauseOriginValue(body.pausedBy)
    return {
        revision: typeof body.revision === 'string' ? body.revision : '',
        roles: parseNodeRoles(body.roles),
        ...(pausedBy ? { pausedBy } : {})
    }
}

function sameRole(left: NodeRole | undefined, right: NodeRole | undefined): boolean {
    return left?.engine === right?.engine && left?.model === right?.model
}

/** Whether two role maps declare the same role for every tier. */
export function sameNodeRoles(left: NodeRoles, right: NodeRoles): boolean {
    return ROLE_TIERS.every(tier => sameRole(left[tier], right[tier]))
}

/** Whether two advertisements are equal; both absent counts as equal. */
export function samePolicyAdvertisement(
    left: NodePolicyAdvertisement | undefined,
    right: NodePolicyAdvertisement | undefined
): boolean {
    if (!left || !right) return left === right
    return (
        left.revision === right.revision &&
        left.pausedBy === right.pausedBy &&
        sameNodeRoles(left.roles, right.roles)
    )
}
