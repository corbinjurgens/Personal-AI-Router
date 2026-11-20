// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ClusterNode } from '@/shared/types/cluster'
import type { EngineStatusData } from '@/shared/types/engines'
import type { NodeItem } from '@/shared/types/nodes'
import type {
    ServiceBridgePushHandler,
    ServiceInvokeArgs,
    ServiceTransport
} from '@/shared/types/service-bridge'
import type {
    WsInvokeChannel,
    WsInvokeResponse,
    WsPushChannel,
    WsPushPayload
} from '@/shared/types/ws-channels'
import type { Workload } from '@/shared/types/workloads'
import type { HostInput } from './host-input'

const sceneSizes = { idle: 0, 'shared-node': 2, 'running-12': 12, 'running-13': 13 }
type ReplayScene = keyof typeof sceneSizes

const replayTime = Date.UTC(2026, 0, 12, 0)
const clusterId = 'replay-cluster'
const gib = 1024 ** 3

function member(nodeUuid: string, name: string, ipAddress: string): ClusterNode {
    return {
        nodeUuid,
        name,
        displayName: name,
        ipAddress,
        port: 54301,
        clusterId,
        state: 'member',
        joinedAt: replayTime,
        lastSeen: replayTime
    }
}

const self = member('replay-self', 'Replay workstation', '192.0.2.10')
const peer = member('replay-peer', 'Replay model host', '192.0.2.20')
const pending = {
    ...member('replay-pending', 'Replay pending peer', '192.0.2.30'),
    state: 'pending-outbound',
    joinedAt: null,
    lastSeen: null
} satisfies ClusterNode

function node(entry: ClusterNode, hasGpu: boolean): NodeItem {
    return {
        id: entry.nodeUuid,
        name: entry.displayName,
        status: 'active',
        ipAddress: entry.ipAddress,
        allIpAddresses: [entry.ipAddress],
        port: entry.port,
        os: 'Windows',
        serviceBuildId: 'fixture',
        topology: {
            cpu: { model: 'Replay CPU', cores: 8, threads: 16 },
            ram: 32 * gib,
            gpus: hasGpu ? [{ id: 'replay-gpu', name: 'Replay GPU', vramTotal: 16 * gib }] : [],
            storage: []
        }
    }
}

function workload(id: string, serial: number, state: 'running' | 'completed'): Workload {
    const createdAt = replayTime + serial * 1000
    return {
        id,
        engine: 'llama-cpp',
        model: 'replay-model',
        state,
        originatedFrom: serial === 2 ? peer.nodeUuid : self.nodeUuid,
        scheduledOn: peer.nodeUuid,
        createdAt,
        startedAt: createdAt + 250,
        completedAt: state === 'completed' ? createdAt + 500 : null,
        error: null,
        requesterId: null
    }
}

type ReplayReads = { [C in WsInvokeChannel]?: () => WsInvokeResponse<C> }
type ReplayListeners = { [C in WsPushChannel]?: Set<ServiceBridgePushHandler<C>> }

function replayTransport(scene: ReplayScene): ServiceTransport {
    const listeners: ReplayListeners = {
        'nodes:upsert': new Set(),
        'engines:state-changed': new Set(),
        'workloads:upsert': new Set(),
        'metrics:update': new Set()
    }
    const noSubscription = () => () => {}
    const reads: ReplayReads = {
        'app:get-initial': () => ({ connected: true, selfId: self.nodeUuid }),
        'cluster:get-initial': () => ({
            info: { clusterId, isClustered: true, clusterFriendlyName: 'Replay' },
            identity: {
                nodeUuid: self.nodeUuid,
                name: self.name,
                customName: null,
                certFingerprint: '',
                clusterId
            },
            members: [self, peer, pending],
            pendingInvites: []
        }),
        'nodes:get-initial': () => ({ nodes: {}, fetchedNodes: true }),
        'workloads:get-initial': () => ({}),
        'engines:get-initial': () => ({
            descriptors: [],
            statuses: [],
            models: [],
            activeProgress: [],
            updateAvailable: []
        }),
        'discovery:get-nodes': () => [],
        'errors:get-initial': () => [],
        'cluster:get-remote-management': () => ({ version: 1, revision: '1', grants: [] }),
        'engine:remote-management-get': () => ({
            version: 1,
            revision: '1',
            allowed: false,
            state: 'denied'
        }),
        'policy:get': () => ({
            policy: '{}',
            revision: '1',
            activation: null,
            availability: 'available',
            recoveryRequired: false
        })
    }
    let closed = false
    let samples: ReturnType<typeof setInterval> | undefined
    const push = <C extends WsPushChannel>(channel: C, payload: WsPushPayload<C>) => {
        if (!closed) listeners[channel]?.forEach(listener => listener(payload))
    }
    const emitMetrics = () => {
        const timestamp = Date.now()
        push('metrics:update', {
            id: self.nodeUuid,
            current: {
                timestamp,
                cpuUtilization: 21,
                memoryUsage: 43,
                gpuUtilization: [],
                gpuVramUsage: []
            },
            historical: []
        })
        push('metrics:update', {
            id: peer.nodeUuid,
            current: {
                timestamp,
                gpuUtilization: [{ id: 'replay-gpu', value: scene === 'idle' ? 4 : 73 }],
                gpuVramUsage: [{ id: 'replay-gpu', value: 48 }]
            },
            historical: []
        })
    }
    const firstUpdate = setTimeout(() => {
        push('nodes:upsert', node(self, false))
        push('nodes:upsert', node(peer, true))
        push('nodes:upsert', node(pending, false))
        const status: EngineStatusData = {
            engineType: 'llama-cpp',
            nodeId: peer.nodeUuid,
            processStatus: 'running',
            enginePort: 8080,
            proxyPort: 54301
        }
        push('engines:state-changed', {
            nodeId: status.nodeId,
            engineType: status.engineType,
            status
        })
        push('workloads:upsert', workload('history-1', 0, 'completed'))
        for (let index = 1; index <= sceneSizes[scene]; index++) {
            // Separate origins can issue the same proxy-local ID.
            const id = `job-${String(index === 2 ? 1 : index).padStart(2, '0')}`
            push('workloads:upsert', workload(id, index, 'running'))
        }
        emitMetrics()
        samples = setInterval(emitMetrics, 2000)
    }, 1000)

    return {
        get connected() {
            return !closed
        },
        async invoke<C extends WsInvokeChannel>(
            ...[channel]: ServiceInvokeArgs<C>
        ): Promise<WsInvokeResponse<C>> {
            const read = reads[channel]
            if (closed || !read) throw new Error(`Replay channel unavailable: ${channel}`)
            return read()
        },
        subscribePush<C extends WsPushChannel>(channel: C, handler: ServiceBridgePushHandler<C>) {
            const handlers = listeners[channel]
            handlers?.add(handler)
            return () => void handlers?.delete(handler)
        },
        onConnect: noSubscription,
        onDisconnect: noSubscription,
        onAuthFailure: noSubscription,
        destroy() {
            closed = true
            clearTimeout(firstUpdate)
            clearInterval(samples)
        }
    }
}

export function replayOverviewInput(name: string | null): HostInput | null {
    const scene = name ?? 'idle'
    if (!isReplayScene(scene)) return null
    return {
        platform: 'Windows',
        subscribe(listener) {
            listener(replayTransport(scene))
            return () => {}
        }
    }
}

function isReplayScene(name: string): name is ReplayScene {
    return Object.hasOwn(sceneSizes, name)
}
