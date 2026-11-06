// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { JsonValue } from '@/shared/types/json'
import { parseModelIdentity } from '@/shared/utils/model-identity'
import type { NodePauseMode } from '@/shared/types/node-policy'
import type {
    NodePolicyAdvertisement,
    AdvertisedProfile,
    NodeRoles,
    PauseOrigin
} from '@/shared/types/node-roles'

/** Mirror nodepolicy.ValidateRoleName: owner decision D1a has no built-in roles. */
export function roleNameError(name: string): string | undefined {
    // Compare the complete match because JavaScript's $ also matches before a final newline.
    if (name.length > 64 || /^[a-z0-9][a-z0-9._-]{0,63}$/.exec(name)?.[0] !== name)
        return 'Role name must match ^[a-z0-9][a-z0-9._-]{0,63}$'
    return undefined
}

/** Only recognized colon overrides reserve concrete served names. */
export function isReservedAlias(name: string): boolean {
    const colon = name.indexOf(':')
    return (
        colon !== -1 &&
        roleNameError(name.slice(0, colon)) === undefined &&
        ['api', 'local'].includes(name.slice(colon + 1))
    )
}

/** Every variable after a valid plain role name is reserved for public keys. */
export function isReservedProfileKey(name: string): boolean {
    const colon = name.indexOf(':')
    return colon !== -1 && roleNameError(name.slice(0, colon)) === undefined
}

function objectValue(value: JsonValue | undefined): { [key: string]: JsonValue } | null {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return null
    return value
}

/** The byte and control bounds shared by Go model identifiers and profile keys. */
function validIdentifier(value: JsonValue | undefined): value is string {
    return (
        typeof value === 'string' &&
        value.length > 0 &&
        value.isWellFormed() &&
        new TextEncoder().encode(value).length <= 512 &&
        !Array.from(value).some(character => {
            const code = character.codePointAt(0) ?? 0
            return code < 0x20 || (code >= 0x7f && code <= 0x9f)
        })
    )
}

/** Mirror nodepolicy.ValidateProfileKey, including its broad colon reservation. */
function validProfileKey(value: JsonValue | undefined): value is string {
    return validIdentifier(value) && !isReservedProfileKey(value)
}

/**
 * Read ordered declarations, keeping empty roles and dropping malformed keys.
 */
export function parseNodeRoles(value: JsonValue | undefined): NodeRoles {
    const roles: NodeRoles = []
    if (!Array.isArray(value)) return roles
    for (const item of value) {
        const role = objectValue(item)
        const name = role?.name
        if (
            !role ||
            typeof name !== 'string' ||
            roleNameError(name) !== undefined ||
            !Array.isArray(role.profiles)
        )
            continue
        roles.push({ name, profiles: role.profiles.filter(validProfileKey) })
    }
    return roles
}

/** Keyed references shared by role choices and advertisement decoding. */
export function parseRoleProfiles(value: JsonValue | undefined): AdvertisedProfile[] {
    if (!Array.isArray(value)) return []
    return value.flatMap(item => {
        const profile = objectValue(item)
        if (
            !profile ||
            !validProfileKey(profile.key) ||
            typeof profile.engine !== 'string' ||
            !profile.engine ||
            !validIdentifier(profile.model) ||
            typeof profile.displayName !== 'string' ||
            !validIdentifier(profile.displayName) ||
            new TextEncoder().encode(profile.displayName).length > 128 ||
            (profile.icon !== undefined &&
                (typeof profile.icon !== 'string' ||
                    !['', 'chat-single', 'code', 'cube', 'document', 'image'].includes(
                        profile.icon
                    ))) ||
            (profile.contextTokens != null &&
                (typeof profile.contextTokens !== 'number' ||
                    !Number.isSafeInteger(profile.contextTokens))) ||
            (profile.capabilities != null &&
                (!Array.isArray(profile.capabilities) ||
                    profile.capabilities.some(cap => typeof cap !== 'string')))
        )
            return []
        const qualification = objectValue(profile.qualification)
        const modelIdentity = parseModelIdentity(profile.modelIdentity)
        if (profile.modelIdentity != null && !modelIdentity) return []
        return [
            {
                key: profile.key,
                displayName: profile.displayName,
                engine: profile.engine,
                model: profile.model,
                ...(modelIdentity ? { modelIdentity } : {}),
                ...(typeof profile.icon === 'string' ? { icon: profile.icon } : {}),
                ...(typeof profile.contextTokens === 'number'
                    ? { contextTokens: profile.contextTokens }
                    : {}),
                ...(Array.isArray(profile.capabilities)
                    ? {
                          capabilities: profile.capabilities.filter(
                              (cap): cap is string => typeof cap === 'string'
                          )
                      }
                    : {}),
                ...(typeof profile.maxOutputTokens === 'number' &&
                Number.isSafeInteger(profile.maxOutputTokens)
                    ? { maxOutputTokens: profile.maxOutputTokens }
                    : {}),
                ...(qualification &&
                typeof qualification.id === 'string' &&
                typeof qualification.revision === 'string'
                    ? {
                          qualification: {
                              id: qualification.id,
                              revision: qualification.revision
                          }
                      }
                    : {})
            }
        ]
    })
}

export function copyNodeRoles(roles: NodeRoles): NodeRoles {
    return roles.map(role => ({ name: role.name, profiles: [...role.profiles] }))
}

/** Keep the existing single-profile control while preserving declaration order. */
export function withRoleProfile(roles: NodeRoles, name: string, key?: string): NodeRoles {
    const next = copyNodeRoles(roles)
    const role = next.find(role => role.name === name)
    if (role) role.profiles = key === undefined ? [] : [key]
    return next
}

/** A `pausedBy` value, or undefined when absent or unknown. */
export function pauseOriginValue(value: JsonValue | undefined): PauseOrigin | undefined {
    return value === 'local' || value === 'peer' ? value : undefined
}

/** The accepted timing of a saved pause, or undefined when absent or unknown. */
export function pauseModeValue(value: JsonValue | undefined): NodePauseMode | undefined {
    return value === 'configured' || value === 'immediate' ? value : undefined
}

/** Read a discovery node's `policy` advertisement; undefined when the node sent none. */
export function parsePolicyAdvertisement(
    value: JsonValue | undefined
): NodePolicyAdvertisement | undefined {
    const body = objectValue(value)
    if (!body) return undefined
    const pausedBy = pauseOriginValue(body.pausedBy)
    const pauseMode = pauseModeValue(body.pauseMode)
    if (body.pauseMode !== undefined && body.pauseMode !== '' && !pauseMode) return undefined
    const profiles = parseRoleProfiles(body.profiles)
    const kept = new Set(profiles.map(profile => profile.key))
    const dropped = new Set(
        (Array.isArray(body.profiles) ? body.profiles : []).flatMap(value => {
            const profile = objectValue(value)
            return profile && validProfileKey(profile.key) && !kept.has(profile.key)
                ? [profile.key]
                : []
        })
    )
    return {
        revision: typeof body.revision === 'string' ? body.revision : '',
        roles: parseNodeRoles(body.roles).map(role => ({
            name: role.name,
            profiles: role.profiles.filter(key => !dropped.has(key))
        })),
        profiles,
        ...(pausedBy ? { pausedBy } : {}),
        ...(pauseMode ? { pauseMode } : {})
    }
}

/** Equality includes declaration order and profile preference order. */
export function sameNodeRoles(left: NodeRoles, right: NodeRoles): boolean {
    return (
        left.length === right.length &&
        left.every((role, index) => {
            const other = right[index]
            return (
                role.name === other.name &&
                role.profiles.length === other.profiles.length &&
                role.profiles.every((key, position) => key === other.profiles[position])
            )
        })
    )
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
        left.pauseMode === right.pauseMode &&
        JSON.stringify(left.profiles) === JSON.stringify(right.profiles) &&
        sameNodeRoles(left.roles, right.roles)
    )
}
