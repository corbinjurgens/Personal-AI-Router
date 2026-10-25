// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Strongly-typed contract for every logical service message exchanged between
 * the UI and the Electron preload service bridge.
 *
 * - `WsInvokeChannelMap`: request/response shape for each logical invoke channel.
 *   The UI invokes `window.pairApi`, preload forwards over IPC, and Electron
 *   main replies with either modular subprocess data or an explicit empty
 *   payload.
 *
 * - `WsPushChannelMap`: payload shape for every service-to-renderer push event.
 *
 * Both maps are the single source of truth; the preload transport and service
 * bridge handlers are generic over these keys so TypeScript catches a
 * mismatched channel or payload at compile time.
 *
 * Discipline:
 *   - One-off payloads are inlined, not named. Named shapes are only lifted
 *     when (a) a second file imports them or (b) inlining would duplicate the
 *     shape twice inside this file.
 *   - `export` is reserved for the handful of symbols consumed by the
 *     service bridge generics — nothing else.
 */
import type {
    SwitchRequest,
    SwitchStatusRequest,
    SwitchStatus,
    SwitchReceipt,
    SwitchChanged
} from '@/shared/types/engine-switch'
import type {
    EngineCommandPayload,
    EngineHubSearchResponse,
    EngineInitialState,
    EngineIntentSnapshot,
    EngineStatePatch
} from '@/shared/types/engine-api'
import type { EngineDescriptor, EngineProgress, EngineType } from '@/shared/types/engines'
import type {
    EngineSettingsTarget,
    EngineSettingsRequest,
    EngineSettingsSnapshot,
    EngineSettingsPreview,
    EngineSettingsReceipt
} from '@/shared/types/engine-settings'
import type { AppInitialSnapshot, ClusterInitialSnapshot } from '@/shared/types/bootstrap'
import type { ServiceError } from '@/shared/types/errors'
import type { NodeItem } from '@/shared/types/nodes'
import type { NodeItemMetrics } from '@/shared/types/metrics'
import type {
    NodeAvailability,
    NodeAvailabilityChange,
    NodeAvailabilityRequest,
    NodePolicyChange,
    NodePolicyDocument,
    NodePolicySnapshot,
    NodePolicyTarget,
    NodePolicyWrite,
    WorkloadCancelRequest
} from '@/shared/types/node-policy'
import type { Workload } from '@/shared/types/workloads'
import type {
    RoutingActiveParams,
    RoutingActiveResult,
    RoutingCancelParams,
    RoutingCancelResult,
    RoutingHistoryConsumed,
    RoutingHistoryInitial,
    RoutingHistoryPull,
    RoutingHistoryReady,
    RoutingHistoryRecord,
    RoutingHistoryTransfer,
    RoutingHistoryConsumption,
    RoutingHistoryCommit
} from '@/shared/types/routing'
import type {
    AvailableNode,
    AskToJoinRequest,
    ClusterIdentityPayload,
    ClusterNode,
    Invite,
    JoinRequest,
    JoinRequestsSnapshot,
    RemoteManagementCapability,
    RemoteManagementSetRequest,
    RemoteManagementSnapshot,
    RespondToJoinRequest
} from '@/shared/types/cluster'

// -----------------------------------------------------------------------------
// Invoke channels — (request, response) pairs
// -----------------------------------------------------------------------------

/**
 * Snapshot returned by `nodes:get-initial`. Plain mutable types on the wire —
 * the server holds `DeepReadonly<T>` internally but JSON transport deep-clones
 * into plain objects on the client, so the wire contract is mutable.
 */
interface NodesInitialResponse {
    nodes: Record<string, NodeItem>
    fetchedNodes: boolean
}

export interface WsInvokeChannelMap {
    // App bootstrap
    'app:get-initial': { request: void; response: AppInitialSnapshot }

    // Nodes
    'nodes:get-initial': { request: void; response: NodesInitialResponse }
    // Remove a node from the cluster (cluster-manager `nodes:remove`) and drop any
    // local manual discovery entry for it in the same call. Distinct from the
    // `nodes:remove` push (state update broadcast).
    'nodes:remove-member': {
        request: { nodeId: string }
        response: { nodeId: string; removed: boolean }
    }

    // Discovery
    'discovery:get-nodes': { request: void; response: AvailableNode[] }

    // Cluster — PIN-pairing handshake (nvpair-cluster-manager)
    'cluster:get-initial': { request: void; response: ClusterInitialSnapshot }
    'cluster:get-join-requests': { request: void; response: JoinRequestsSnapshot }
    'cluster:ask-to-join': { request: AskToJoinRequest; response: JoinRequest }
    'cluster:respond-to-join-request': { request: RespondToJoinRequest; response: JoinRequest }
    'cluster:cancel-join-request': { request: { requestId: string }; response: JoinRequest }
    'cluster:invite-node': {
        request: { ipAddress: string; allowRemoteManagement: boolean }
        response: Invite
    }
    'cluster:invite-status': { request: { inviteId: string }; response: Invite }
    'cluster:respond-to-invite': {
        request:
            | { inviteId: string; accept: true; pin: string; allowRemoteManagement: boolean }
            | { inviteId: string; accept: false }
        response: Invite
    }
    'cluster:get-remote-management': { request: void; response: RemoteManagementSnapshot }
    'cluster:set-remote-management': {
        request: RemoteManagementSetRequest
        response: RemoteManagementSnapshot
    }
    'engine:remote-management-get': {
        request: { nodeId: string }
        response: RemoteManagementCapability
    }
    // Abort a still-pending outbound invite (the inviter's counterpart to a
    // joiner decline): tears down the pairing session, invalidates the PIN, and
    // best-effort notifies the joiner. Returns the updated `Invite` (`canceled`).
    'cluster:cancel-invite': { request: { inviteId: string }; response: Invite }
    // Dissolve a cluster that was auto-created solely to back an invite, when
    // that pairing failed and no peer joined (no-op otherwise).
    'cluster:abandon-if-solo': { request: void; response: null }

    // Engines
    'engines:get-initial': { request: void; response: EngineInitialState }
    'engines:get-intent': { request: void; response: EngineIntentSnapshot | null }
    'engines:get-settings': { request: EngineSettingsTarget; response: EngineSettingsSnapshot }
    'engines:switch': { request: SwitchRequest; response: { receipt: SwitchReceipt } }
    'engines:switch-status': { request: SwitchStatusRequest; response: SwitchStatus }
    'engines:preview-settings': { request: EngineSettingsRequest; response: EngineSettingsPreview }
    'engines:apply-settings': { request: EngineSettingsRequest; response: EngineSettingsReceipt }
    'engine:command': { request: EngineCommandPayload; response: null }
    'engine:search-hub': {
        request: { engineType: EngineType; query?: string }
        response: EngineHubSearchResponse
    }

    // Errors
    'errors:get-initial': { request: void; response: ServiceError[] }
    'errors:clear': { request: string; response: null }

    // Workloads
    'workloads:get-initial': { request: void; response: Record<string, Workload> }
    // `ok` is false when the broker found no such job (it may have just finished).
    'workloads:cancel': { request: WorkloadCancelRequest; response: { ok: boolean } }

    // Origin-local routing request control
    'routing:cancel': { request: RoutingCancelParams; response: RoutingCancelResult }
    'routing:active': { request: RoutingActiveParams; response: RoutingActiveResult }
    'routing:subscribe': { request: void; response: RoutingHistoryReady }
    'routing:unsubscribe': { request: void; response: null }
    'routing:get-initial': { request: void; response: RoutingHistoryInitial }
    // Electron delivery controls. Never relayed as service/peer RPC methods.
    'routing-history:pull': { request: RoutingHistoryPull; response: RoutingHistoryTransfer }
    'routing-history:consumed': {
        request: RoutingHistoryConsumed
        response: RoutingHistoryConsumption
    }

    // Node policy and availability
    'policy:get': { request: NodePolicyTarget; response: NodePolicyDocument }
    'policy:set': { request: NodePolicyWrite; response: NodePolicySnapshot }
    'node:set-availability': {
        request: NodeAvailabilityRequest
        response: { availability: NodeAvailability }
    }
}

export type WsInvokeChannel = keyof WsInvokeChannelMap
export type WsInvokeRequest<C extends WsInvokeChannel> = WsInvokeChannelMap[C]['request']
export type WsInvokeResponse<C extends WsInvokeChannel> = WsInvokeChannelMap[C]['response']

// -----------------------------------------------------------------------------
// Push channels — server → client broadcast events (single payload per event)
// -----------------------------------------------------------------------------

export interface WsPushChannelMap {
    'routing:changed': RoutingHistoryRecord
    'routing-history:ready': RoutingHistoryReady
    'routing-history:committed': RoutingHistoryCommit
    'cluster:join-requests-changed': JoinRequestsSnapshot
    'cluster:remote-management-changed': RemoteManagementSnapshot
    // Nodes / cluster state
    'nodes:upsert': NodeItem
    'nodes:remove': string
    /** Full cluster membership snapshot, pushed on every change (cluster-manager `nodes:changed`). */
    'nodes:changed': ClusterNode[]

    // Engines
    'engines:state-changed': EngineStatePatch
    'engines:intent-changed': EngineIntentSnapshot | null
    'engines:settings-changed': EngineSettingsSnapshot
    'engines:switch-changed': SwitchChanged
    'engines:settings-disconnected': { nodeId: string }
    'engines:progress-changed': EngineProgress
    'engines:progress-cleared': { key: string }
    /** The local engine-manager's full descriptor set, replaced whenever it is re-read. */
    'engines:descriptors-changed': EngineDescriptor[]

    // Node policy and availability, for every node the broker relays (not only this PC)
    'node:availability-changed': NodeAvailabilityChange
    'policy:changed': NodePolicyChange

    // Metrics
    'metrics:update': NodeItemMetrics

    // Workloads. Removal carries the origin node (`originatedFrom`) because
    // workload ids are a per-node proxy counter (the catalog is keyed by the
    // (originatedFrom, id) pair).
    'workloads:upsert': Workload
    'workloads:remove': { workloadId: string; originatedFrom: string | null }

    // Errors
    'errors:update': ServiceError[]

    // Cluster flow — an inbound pairing arrived; prompt the user for the PIN.
    'cluster:invite-received': Invite
    // Full authoritative snapshot of every live inbound invite awaiting a PIN,
    // re-emitted by Electron main on every add/prune (arrival, paired, decline,
    // status-poll terminal state, or client TTL expiry).
    'cluster:pending-invites-changed': Invite[]

    // Discovery
    'discovery:nodes-changed': AvailableNode[]

    // Connection lifecycle
    'connection:cluster-identity': ClusterIdentityPayload
    'state:request-refresh': void
}

export type WsPushChannel = keyof WsPushChannelMap
export type WsPushPayload<C extends WsPushChannel> = WsPushChannelMap[C]
