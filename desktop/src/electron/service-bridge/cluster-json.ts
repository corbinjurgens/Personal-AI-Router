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
    JoinRequest,
    JoinRequestsSnapshot,
    JoinRequestState,
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

function joinRequestState(value: JsonValue | undefined): value is JoinRequestState {
    switch (value) {
        case 'awaiting-approval':
        case 'awaiting-requester':
        case 'pairing':
        case 'paired':
        case 'declined':
        case 'canceled':
        case 'expired':
        case 'failed':
        case 'rejected':
            return true
        default:
            return false
    }
}

function joinTimestamp(value: JsonValue | undefined): value is number {
    return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}

/** Reject malformed records rather than inventing an identity or successful state. */
export function parseJoinRequest(value: JsonValue | undefined): JoinRequest {
    const obj = objectValue(value)
    if (
        !obj ||
        !exactFields(obj, [
            'requestId',
            'direction',
            'peerNodeUuid',
            'peerName',
            'peerAddress',
            'peerPort',
            'state',
            'inviteId',
            'reason',
            'createdAt',
            'expiresAt'
        ]) ||
        typeof obj.requestId !== 'string' ||
        obj.requestId === '' ||
        (obj.direction !== 'incoming' && obj.direction !== 'outgoing') ||
        typeof obj.peerNodeUuid !== 'string' ||
        typeof obj.peerName !== 'string' ||
        typeof obj.peerAddress !== 'string' ||
        typeof obj.peerPort !== 'number' ||
        !Number.isInteger(obj.peerPort) ||
        obj.peerPort < 1 ||
        obj.peerPort > 65535 ||
        !joinRequestState(obj.state) ||
        (obj.inviteId !== null && (typeof obj.inviteId !== 'string' || obj.inviteId === '')) ||
        typeof obj.reason !== 'string' ||
        !joinTimestamp(obj.createdAt) ||
        !joinTimestamp(obj.expiresAt) ||
        obj.expiresAt < obj.createdAt
    ) {
        throw new Error('Invalid join request response')
    }
    return {
        requestId: obj.requestId,
        direction: obj.direction,
        peerNodeUuid: obj.peerNodeUuid,
        peerName: obj.peerName,
        peerAddress: obj.peerAddress,
        peerPort: obj.peerPort,
        state: obj.state,
        inviteId: obj.inviteId,
        reason: obj.reason,
        createdAt: obj.createdAt,
        expiresAt: obj.expiresAt
    }
}

/** Malformed snapshots invalidate cached eligibility on reads and pushes alike. */
export function parseJoinRequestsSnapshot(value: JsonValue | undefined): JoinRequestsSnapshot {
    const unavailable: JoinRequestsSnapshot = {
        supported: true,
        canAskToJoin: false,
        requiresSoloReplacement: false,
        blockedReason: 'state-unavailable',
        requests: []
    }
    const obj = objectValue(value)
    if (
        !obj ||
        !exactFields(obj, [
            'supported',
            'canAskToJoin',
            'requiresSoloReplacement',
            'blockedReason',
            'requests'
        ]) ||
        obj.supported !== true ||
        typeof obj.canAskToJoin !== 'boolean' ||
        typeof obj.requiresSoloReplacement !== 'boolean' ||
        (obj.blockedReason !== '' &&
            obj.blockedReason !== 'already-clustered' &&
            obj.blockedReason !== 'pairing-busy' &&
            obj.blockedReason !== 'state-unavailable') ||
        obj.canAskToJoin !== (obj.blockedReason === '') ||
        !Array.isArray(obj.requests)
    )
        return unavailable
    try {
        const requests = obj.requests.map(parseJoinRequest)
        if (new Set(requests.map(request => request.requestId)).size !== requests.length)
            return unavailable
        return {
            supported: true,
            canAskToJoin: obj.canAskToJoin,
            requiresSoloReplacement: obj.requiresSoloReplacement,
            blockedReason: obj.blockedReason,
            requests
        }
    } catch {
        return unavailable
    }
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
