// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Parsers that turn nvpair-cluster-manager JSON-RPC payloads (relayed verbatim by
 * the broker) into the typed cluster shapes the bridge and renderer share.
 *
 * Kept in its own module so both the invoke handlers (`empty-handlers.ts`) and
 * the notification demux (`modular-supervisor.ts`) can use them without an
 * import cycle.
 */
import type {
    ClusterNode,
    ClusterNodeIdentity,
    Invite,
    InviteState,
    MembershipState,
    RemoteManagementCapability,
    RemoteManagementSnapshot,
    RemoteManagementState
} from '@/shared/types/cluster'
import type { JsonObject, JsonValue } from './json-rpc-client'

function objectValue(value: JsonValue | undefined): JsonObject | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null
    return value
}

function stringValue(value: JsonValue | undefined): string {
    return typeof value === 'string' ? value : ''
}

function numberValue(value: JsonValue | undefined): number {
    return typeof value === 'number' ? value : 0
}

function nullableNumber(value: JsonValue | undefined): number | null {
    return typeof value === 'number' ? value : null
}

function nullableString(value: JsonValue | undefined): string | null {
    return typeof value === 'string' ? value : null
}

function arrayValue(value: JsonValue | undefined): JsonValue[] {
    return Array.isArray(value) ? value : []
}

function inviteState(value: JsonValue | undefined): InviteState {
    switch (stringValue(value)) {
        case 'pending':
            return 'pending'
        case 'paired':
            return 'paired'
        case 'declined':
            return 'declined'
        case 'canceled':
            return 'canceled'
        case 'expired':
            return 'expired'
        case 'rejected':
            return 'rejected'
        default:
            return 'failed'
    }
}

function membershipState(value: JsonValue | undefined): MembershipState {
    switch (stringValue(value)) {
        case 'pending-outbound':
            return 'pending-outbound'
        case 'pending-inbound':
            return 'pending-inbound'
        default:
            return 'member'
    }
}

export function parseInvite(value: JsonValue | undefined): Invite {
    const obj = objectValue(value)
    return {
        inviteId: stringValue(obj?.inviteId),
        fromNodeId: stringValue(obj?.fromNodeId),
        fromNodeUuid: stringValue(obj?.fromNodeUuid),
        fromNodeName: stringValue(obj?.fromNodeName),
        toNodeId: nullableString(obj?.toNodeId),
        clusterId: stringValue(obj?.clusterId),
        clusterFriendlyName: stringValue(obj?.clusterFriendlyName),
        pin: nullableString(obj?.pin),
        state: inviteState(obj?.state),
        reason: stringValue(obj?.reason),
        ...(typeof obj?.error === 'string' ? { error: obj.error } : {}),
        createdAt: numberValue(obj?.createdAt),
        respondedAt: nullableNumber(obj?.respondedAt)
    }
}

export function emptyInvite(): Invite {
    return {
        inviteId: '',
        fromNodeId: '',
        fromNodeUuid: '',
        fromNodeName: '',
        toNodeId: null,
        clusterId: '',
        clusterFriendlyName: '',
        pin: null,
        state: 'failed',
        reason: '',
        createdAt: Date.now(),
        respondedAt: null
    }
}

function parseClusterNode(value: JsonValue | undefined): ClusterNode {
    const obj = objectValue(value)
    return {
        id: stringValue(obj?.id),
        nodeUuid: stringValue(obj?.nodeUuid),
        name: stringValue(obj?.name),
        ipAddress: stringValue(obj?.ipAddress),
        port: numberValue(obj?.port),
        clusterId: stringValue(obj?.clusterId),
        state: membershipState(obj?.state),
        joinedAt: nullableNumber(obj?.joinedAt),
        lastSeen: nullableNumber(obj?.lastSeen)
    }
}

/** Parse a `{ nodes: ClusterNode[] }` envelope (nodes:get-initial / nodes:changed). */
export function parseClusterNodes(value: JsonValue | undefined): ClusterNode[] {
    return arrayValue(objectValue(value)?.nodes).map(parseClusterNode)
}

export function parseNodeIdentity(value: JsonValue | undefined): ClusterNodeIdentity {
    const obj = objectValue(value)
    return {
        nodeUuid: stringValue(obj?.nodeUuid),
        nodeId: stringValue(obj?.nodeId),
        name: stringValue(obj?.name),
        certFingerprint: stringValue(obj?.certFingerprint),
        clusterId: stringValue(obj?.clusterId)
    }
}

function exactFields(obj: JsonObject, fields: string[]): boolean {
    return (
        Object.keys(obj).length === fields.length &&
        fields.every(field => Object.hasOwn(obj, field))
    )
}

function boundedPayload(value: JsonValue | undefined, maxBytes: number): boolean {
    try {
        const json = JSON.stringify(value)
        return json !== undefined && Buffer.byteLength(json, 'utf8') <= maxBytes
    } catch {
        return false
    }
}

function permissionRevision(value: JsonValue | undefined): value is string {
    return (
        typeof value === 'string' &&
        /^[1-9][0-9]{0,19}$/.test(value) &&
        (value.length < 20 || value <= '18446744073709551615')
    )
}

function permissionPrincipal(value: JsonValue | undefined): value is string {
    return (
        typeof value === 'string' &&
        value !== '' &&
        Buffer.byteLength(value, 'utf8') <= 128 &&
        value !== '.' &&
        value !== '..' &&
        !/[/\\:\p{White_Space}\p{Cc}]/u.test(value)
    )
}

function permissionState(value: JsonValue | undefined): value is RemoteManagementState {
    return (
        value === 'allowed' ||
        value === 'denied' ||
        value === 'unreviewed' ||
        value === 'unavailable'
    )
}

/** Invalid input is an invalidation barrier, including on the live push path. */
export function parseRemoteManagementSnapshot(
    value: JsonValue | undefined
): RemoteManagementSnapshot {
    const unavailable: RemoteManagementSnapshot = { version: 1, revision: '', grants: [] }
    if (!boundedPayload(value, 1 << 20)) return unavailable
    const obj = objectValue(value)
    if (
        !obj ||
        !exactFields(obj, ['version', 'revision', 'grants']) ||
        obj.version !== 1 ||
        (obj.revision !== '' && !permissionRevision(obj.revision)) ||
        !Array.isArray(obj.grants) ||
        obj.grants.length > 1024
    )
        return unavailable
    const grants: RemoteManagementSnapshot['grants'] = []
    const peers = new Set<string>()
    for (const value of obj.grants) {
        const grant = objectValue(value)
        if (
            !grant ||
            !exactFields(grant, ['peerUuid', 'allowed', 'state']) ||
            !permissionPrincipal(grant.peerUuid) ||
            peers.has(grant.peerUuid) ||
            typeof grant.allowed !== 'boolean' ||
            !permissionState(grant.state) ||
            grant.allowed !== (grant.state === 'allowed') ||
            (obj.revision === '' && grant.state !== 'unavailable')
        )
            return unavailable
        peers.add(grant.peerUuid)
        grants.push({ peerUuid: grant.peerUuid, allowed: grant.allowed, state: grant.state })
    }
    return { version: 1, revision: obj.revision, grants }
}

/** Remote results never accept the local full-list shape or invent a usable revision. */
export function parseRemoteManagementCapability(
    value: JsonValue | undefined
): RemoteManagementCapability {
    const unavailable: RemoteManagementCapability = {
        version: 1,
        revision: '',
        allowed: false,
        state: 'unavailable'
    }
    if (!boundedPayload(value, 64 << 10)) return unavailable
    const obj = objectValue(value)
    if (
        !obj ||
        !exactFields(obj, ['version', 'revision', 'allowed', 'state']) ||
        obj.version !== 1 ||
        typeof obj.allowed !== 'boolean' ||
        (!permissionState(obj.state) && obj.state !== 'unsupported') ||
        obj.allowed !== (obj.state === 'allowed')
    )
        return unavailable
    if (obj.state === 'unavailable' || obj.state === 'unsupported') {
        return obj.revision === ''
            ? { version: 1, revision: '', allowed: false, state: obj.state }
            : unavailable
    }
    if (!permissionRevision(obj.revision)) return unavailable
    return { version: 1, revision: obj.revision, allowed: obj.allowed, state: obj.state }
}
