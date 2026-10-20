// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NodePolicyAdvertisement } from '@/shared/types/node-roles'

/**
 * Cluster-layer data types shared by service transports and the UI.
 *
 * These shapes cross the renderer service bridge, so they must remain
 * JSON-serializable. They mirror the `nvpair-cluster-manager` contract
 * (`services/nvpair-cluster-manager/spec.md` §6/§7): membership and the
 * interactive PIN-pairing invite handshake.
 */

export interface AvailableNode {
    /** Canonical node key = the backend's stable per-host UUID (matches `NodeItem.id`). */
    id: string
    /** Display hostname. */
    name: string
    ipAddress: string
    /**
     * Every address the node published, in its own ranked order with `ipAddress`
     * first. Absent when the node published a single address.
     *
     * A multi-homed node has no single address every peer can reach: a
     * direct-connect link between two machines is reachable only from the machine
     * on its far end. The node ranks its own addresses from evidence no observer
     * has, so anything that must connect walks this list in order rather than
     * treating `ipAddress` as the only answer.
     */
    ipAddresses?: string[]
    port: number
    lastSeen: number
    /** Whether this node is already a locally-trusted (pinned) peer. */
    trusted: boolean
    /**
     * Whether this node already belongs to a cluster. An already-clustered node
     * cannot be invited into another, so invite UI treats it as non-invitable.
     */
    clustered: boolean
    /**
     * The node's own policy advertisement (declared roles and pause origin),
     * from the same pinned inventory fetch as its models. Absent until the node
     * has published one, and from a node that predates roles; either way the
     * node declares nothing.
     */
    policy?: NodePolicyAdvertisement
}

/**
 * Membership state of a `ClusterNode` (cluster-manager `MembershipState`).
 * - `member` — join confirmed on both sides.
 * - `pending-outbound` — we invited them, awaiting their PIN entry.
 * - `pending-inbound` — they invited us, awaiting the local user's PIN entry.
 */
export type MembershipState = 'member' | 'pending-outbound' | 'pending-inbound'

/** A cluster member (or pending invitee) as reported by the cluster-manager. */
export interface ClusterNode {
    /** Logical node id (hostname). Display identity only. */
    id: string
    /**
     * Stable cryptographic identity; the trusted-node-store key and cert subject.
     * This is the cross-domain correlation key — it equals a discovered node's
     * hostUuid / `NodeItem.id` / `selfId`, so member↔node matching keys on this,
     * never `id` (hostname).
     */
    nodeUuid: string
    name: string
    ipAddress: string
    port: number
    clusterId: string
    state: MembershipState
    /** Epoch ms when membership was confirmed; null while pending. */
    joinedAt: number | null
    /** Epoch ms of last successful contact; null if never. */
    lastSeen: number | null
}

/** This node's own identity (`cluster:get-node-id`). */
export interface ClusterNodeIdentity {
    nodeUuid: string
    nodeId: string
    name: string
    certFingerprint: string
    /** `""` until the node creates a cluster or adopts one by accepting an invite. */
    clusterId: string
}

/**
 * Terminal/transient state of a pairing session (cluster-manager `InviteState`).
 * - `pending` — Initial Exchange done; awaiting the user's PIN entry.
 * - `paired` — Completion Exchange succeeded; certs pinned, membership recorded.
 * - `declined` — the invitee's user declined.
 * - `canceled` — the inviter aborted the invite before it completed
 *   (`cluster:cancel-invite`); on the joiner, set when the cancel signal arrives.
 * - `expired` — TTL elapsed with no PIN entry.
 * - `failed` — pairing could not complete (peer unreachable, wrong PIN,
 *   malformed); a wrong PIN carries `reason: 'incorrect-pin'` (see
 *   {@link Invite.reason}) and is mirrored on both the inviter and the joiner.
 * - `rejected` — the target refused the invite outright (e.g. it is already in a
 *   cluster); see {@link Invite.reason}. No PIN is issued.
 */
export type InviteState =
    | 'pending'
    | 'paired'
    | 'declined'
    | 'canceled'
    | 'expired'
    | 'failed'
    | 'rejected'

/**
 * Broker-facing view of a pairing session. The six-digit `pin` is present only
 * in the inviter's `cluster:invite-node` result (the inviter displays it); it is
 * carried to the joiner by the user out of band and never sent on the wire.
 */
export interface Invite {
    inviteId: string
    fromNodeId: string
    fromNodeUuid: string
    fromNodeName: string
    toNodeId: string | null
    clusterId: string
    clusterFriendlyName: string
    pin: string | null
    state: InviteState
    /**
     * Machine-readable reason for a non-successful terminal state, when the
     * backend supplies one. Known values:
     * - `'already-clustered'` — paired with `state: 'rejected'`.
     * - `'incorrect-pin'` — paired with `state: 'failed'`; the joiner entered a
     *   wrong PIN. Set on both the joiner and inviter so the wrong-PIN error is
     *   mirrored on both sides.
     * - `connection-reset`, `connection-refused`, `connection-timeout`,
     *   `connection-unreachable` — transport failure during invitation.
     * - `rollback-pending`, `awaiting-confirmation` — pending durable outcome.
     */
    reason: string
    /** Original diagnostic for a failed Initial Exchange, when supplied. */
    error?: string
    createdAt: number
    respondedAt: number | null
}

export interface ClusterInfo {
    clusterId: string | null
    isClustered: boolean
    clusterFriendlyName: string
}

/** Pushed when the local cluster identity changes (localhost UI). */
export interface ClusterIdentityPayload {
    clusterId: string | null
    clusterFriendlyName: string
}

export type RemoteManagementState = 'allowed' | 'denied' | 'unreviewed' | 'unavailable'

export interface RemoteManagementSnapshot {
    version: 1
    /** Empty means unavailable authority and must never be used for a write. */
    revision: string
    grants: { peerUuid: string; allowed: boolean; state: RemoteManagementState }[]
}

export interface RemoteManagementSetRequest {
    peerUuid: string
    allowed: boolean
    expectedRevision: string
}

/** Only the authenticated caller's permission on the requested target. */
export interface RemoteManagementCapability {
    version: 1
    allowed: boolean
    revision: string
    state: RemoteManagementState | 'unsupported'
}
