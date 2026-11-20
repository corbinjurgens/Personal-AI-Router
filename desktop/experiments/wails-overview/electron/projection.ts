// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { emitBridgePush } from '@/electron/service-bridge/broadcaster'
import {
    parseClusterNodes,
    parseNodeIdentity,
    parseNodeNameResult,
    parseRemoteManagementCapability,
    parseRemoteManagementSnapshot
} from '@/electron/service-bridge/cluster-json'
import { parseEngineDescriptors } from '@/electron/service-bridge/engine-descriptors'
import { parseEngineSettings } from '@/electron/service-bridge/engine-settings'
import type {
    JsonRpcNotification,
    JsonRpcSocketClient,
    JsonValue
} from '@/electron/service-bridge/json-rpc-client'
import { parseModelsResult } from '@/electron/service-bridge/model-download'
import {
    getModularBridgeState,
    isUpstreamUnreachableError,
    parseServiceErrors
} from '@/electron/service-bridge/modular-state'
import { startNodeInfoPoller, stopNodeInfoPoller } from '@/electron/service-bridge/node-info-poller'
import {
    nodeTargetParams,
    parseAvailabilityChange,
    parsePolicyChange,
    parsePolicyDocument
} from '@/electron/service-bridge/node-policy'
import { objectValue, parseProxyNode } from '@/electron/service-bridge/node-projection'
import { PerformanceSubscription } from '@/electron/service-bridge/performance'
import {
    PROXY_ENGINES,
    type ProxyEngine,
    proxySourceForEngine
} from '@/electron/service-bridge/proxy-engines'
import { parseWorkloadsInitial } from '@/electron/service-bridge/workload-projection'
import { MODULAR_NODE_INFO_POLL_TIMEOUT_MS } from '@/shared/constants/modular-runtime'
import type { WsInvokeRequest, WsInvokeResponse } from '@/shared/types/ws-channels'
import { engineTypeFromManagerName } from '@/shared/utils/engines'
import type { OverviewRead } from '../ui/overview-reads'

type OverviewHandlers = {
    [C in OverviewRead]: (payload: WsInvokeRequest<C> | undefined) => Promise<WsInvokeResponse<C>>
}

function arrayField(value: JsonValue, field: string, method: string): JsonValue[] {
    const rows = objectValue(value)?.[field]
    if (!Array.isArray(rows)) throw new Error(`Invalid Overview baseline: ${method}`)
    return rows
}

const visibleErrors = (value: JsonValue | undefined) =>
    parseServiceErrors(value).filter(error => !isUpstreamUnreachableError(error))

// The window owns this adapter. Only these named reads and stream subscriptions
// are ever sent. The production supervisor also owns service writes, so use the
// existing decoders and projections below independently of its startup path.
export function createOverviewProjection(client: Pick<JsonRpcSocketClient, 'call'>) {
    const state = getModularBridgeState()
    let retired = false
    let discoveryRevision = 0
    let engineRevision = 0
    const proxyNodes = new Map<ProxyEngine, Map<string, JsonValue>>()
    const proxyChangesDuringBaseline = new Map<ProxyEngine, Set<string>>()
    const updatedEngines = new Map<string, number>()
    const remoteRevisions = new Map<string, number>()

    async function read(method: string, params?: JsonValue, timeoutMs?: number) {
        if (retired) throw new Error('Overview generation retired')
        const result = await client.call(method, params, timeoutMs)
        if (retired) throw new Error('Overview generation retired')
        if (result == null) throw new Error(`Missing Overview response: ${method}`)
        return result
    }

    const performance = new PerformanceSubscription(
        method => read(method),
        params =>
            state.handleNotification({ source: 'broker', method: 'performance:changed', params })
    )

    function applyProxy(engine: ProxyEngine, method: string, params?: JsonValue) {
        const source = proxySourceForEngine(engine)
        if (
            params &&
            (method === 'node/discovered' || method === 'node/updated' || method === 'node/removed')
        ) {
            const node = parseProxyNode(params, engine)
            const nodes = proxyNodes.get(engine) ?? new Map<string, JsonValue>()
            if (node) {
                if (method === 'node/removed') nodes.delete(node.id)
                else nodes.set(node.id, params)
                proxyNodes.set(engine, nodes)
            }
        }
        state.handleNotification({ source, method, params })
    }

    function applyDiscovery(params: JsonValue | undefined) {
        state.handleNotification({ source: 'broker', method: 'discovery:nodes-changed', params })
    }

    async function readIdentity() {
        const identity = parseNodeIdentity(await read('cluster:get-node-id'))
        if (!identity.nodeUuid) throw new Error('Overview identity is unavailable')
        state.setSelfId(identity.nodeUuid)
        return identity
    }

    async function readClusterName() {
        const result = await read('settings/get-cluster-friendly-name')
        const name = objectValue(result)?.value
        if (typeof name !== 'string') throw new Error('Invalid Overview cluster name')
        return name
    }

    async function readRemoteEngineFacts(nodeId: string) {
        const revision = remoteRevisions.get(nodeId)
        const result = await read('engine:remote-get-installed', { node: nodeId })
        if (remoteRevisions.get(nodeId) === revision) state.applyRemoteEngineFacts(nodeId, result)
    }

    async function readCluster() {
        const [identity, roster, clusterFriendlyName] = await Promise.all([
            getIdentity(),
            read('nodes:get-initial'),
            readClusterName()
        ])
        arrayField(roster, 'nodes', 'nodes:get-initial')
        return {
            info: {
                clusterId: identity.clusterId || null,
                isClustered: identity.clusterId !== '',
                clusterFriendlyName
            },
            identity,
            members: parseClusterNodes(roster),
            pendingInvites: []
        }
    }

    async function readDiscovery() {
        await read('discovery:subscribe')
        const revision = discoveryRevision
        const response = await read('discovery:get-nodes')
        arrayField(response, 'nodes', 'discovery:get-nodes')
        if (revision === discoveryRevision) applyDiscovery(response)
    }

    async function readProxy(engine: ProxyEngine) {
        const source = proxySourceForEngine(engine)
        await read(`${source}:subscribe`)
        const changed = new Set<string>()
        proxyChangesDuringBaseline.set(engine, changed)
        try {
            const [result, status] = await Promise.all([
                read(`${source}:nodes/list`),
                read(`${source}:get-status`)
            ])
            for (const row of arrayField(result, 'nodes', `${source}:nodes/list`)) {
                const node = parseProxyNode(row, engine)
                if (node && !changed.has(node.id)) applyProxy(engine, 'node/discovered', row)
            }
            if (objectValue(status)?.ready === true) applyProxy(engine, 'ready', status)
        } finally {
            proxyChangesDuringBaseline.delete(engine)
        }
    }

    async function readModels() {
        const sequence = state.beginLocalModelsRead()
        const result = parseModelsResult(await read('engine:models'))
        state.applyLocalModelsResult(result, sequence)
    }

    async function readEngines() {
        await read('engine:subscribe')
        const revision = engineRevision
        const [descriptors, status] = await Promise.all([
            read('engine:descriptors'),
            read('engine:get-installed'),
            readModels(),
            getNodes()
        ])
        const selfId = state.getSelfId()
        await Promise.allSettled(
            state
                .getAvailableNodes()
                .filter(node => node.trusted && node.id !== selfId)
                .map(node => readRemoteEngineFacts(node.id))
        )
        if (retired) throw new Error('Overview generation retired')
        arrayField(descriptors, 'engines', 'engine:descriptors')
        state.setEngineDescriptors(parseEngineDescriptors(descriptors))
        const localEngines = new Set(
            arrayField(status, 'engines', 'engine:get-installed').flatMap(params => {
                const engine = objectValue(params)?.engine
                if (typeof engine !== 'string') return []
                if ((updatedEngines.get(engine) ?? 0) <= revision)
                    state.applyEngineManagerStatus(params)
                const type = engineTypeFromManagerName(engine)
                return type ? [type] : []
            })
        )
        const { statuses, models, ...rest } = state.getEngineInitialState()
        return {
            ...rest,
            statuses: statuses.filter(
                status => status.nodeId !== selfId || localEngines.has(status.engineType)
            ),
            models: models.filter(
                model => model.nodeId !== selfId || localEngines.has(model.engineType)
            )
        }
    }

    let identity: ReturnType<typeof readIdentity> | undefined
    let nodes: Promise<void> | undefined
    const getIdentity = () => (identity ??= readIdentity())
    const getNodes = () =>
        (nodes ??= getIdentity().then(async () => {
            await Promise.all([readDiscovery(), ...PROXY_ENGINES.map(readProxy)])
            if (retired) return
            performance.ready()
            startNodeInfoPoller(hostUuid =>
                read('discovery:get-node-info', { hostUuid }, MODULAR_NODE_INFO_POLL_TIMEOUT_MS)
            )
        }))

    const handlers: OverviewHandlers = {
        'app:get-initial': async () => ({
            connected: true,
            selfId: (await getIdentity()).nodeUuid
        }),
        'cluster:get-initial': readCluster,
        'nodes:get-initial': async () => {
            await getNodes()
            return state.getNodesInitial()
        },
        'discovery:get-nodes': async () => {
            await getNodes()
            return state.getAvailableNodes()
        },
        'engines:get-initial': readEngines,
        'workloads:get-initial': async () => {
            await read('workloads:subscribe')
            const since = state.beginWorkloadBaseline()
            const result = await read('workloads:get-initial')
            arrayField(result, 'workloads', 'workloads:get-initial')
            return state.seedWorkloads(parseWorkloadsInitial(result), since)
        },
        'errors:get-initial': async () => {
            const result = await read('errors:get-initial')
            if (!Array.isArray(result)) throw new Error('Invalid Overview errors baseline')
            return visibleErrors(result)
        },
        'cluster:get-remote-management': async () =>
            parseRemoteManagementSnapshot(await read('cluster:get-remote-management', {})),
        'engine:remote-management-get': async payload => {
            if (typeof payload?.nodeId !== 'string') throw new Error('Missing target node')
            return parseRemoteManagementCapability(
                await read('engine:remote-management-get', { nodeId: payload.nodeId })
            )
        },
        'engines:get-settings': async payload => {
            if (typeof payload?.engine !== 'string' || typeof payload.nodeId !== 'string')
                throw new Error('Missing engine settings target')
            return parseEngineSettings(
                await read('engine:get-settings', {
                    engine: payload.engine,
                    nodeId: payload.nodeId
                })
            )
        },
        'policy:get': async payload => {
            if (payload?.nodeId !== undefined && typeof payload.nodeId !== 'string')
                throw new Error('Invalid target node')
            return parsePolicyDocument(await read('policy:get', nodeTargetParams(payload ?? {})))
        }
    }

    let clusterChange = 0
    function project({ method, params }: JsonRpcNotification) {
        if (retired) return
        switch (method) {
            case 'discovery:nodes-changed':
                discoveryRevision++
                applyDiscovery(params)
                break
            case 'performance:changed':
                performance.changed(params)
                break
            case 'workloads:upsert':
                state.upsertWorkloadFromInfo(params)
                break
            case 'workloads:remove':
                state.removeWorkloadFromParams(params)
                break
            case 'errors:update':
                state.replaceErrors(visibleErrors(params))
                break
            case 'nodes:changed':
                emitBridgePush('nodes:changed', parseClusterNodes(params))
                break
            case 'cluster:node-name-changed':
                emitBridgePush('cluster:node-name-changed', parseNodeNameResult(params))
                break
            case 'cluster:remote-management-changed':
                emitBridgePush(
                    'cluster:remote-management-changed',
                    parseRemoteManagementSnapshot(params)
                )
                break
            case 'connection/cluster-identity': {
                const id = objectValue(params)?.id
                if (typeof id !== 'string') break
                const revision = ++clusterChange
                void readClusterName()
                    .then(clusterFriendlyName => {
                        if (retired || clusterChange !== revision) return
                        emitBridgePush('connection:cluster-identity', {
                            clusterId: id || null,
                            clusterFriendlyName
                        })
                    })
                    .catch(() => {})
                break
            }
            case 'node:availability-changed': {
                const change = parseAvailabilityChange(params)
                if (change) emitBridgePush('node:availability-changed', change)
                break
            }
            case 'policy:changed': {
                const change = parsePolicyChange(params)
                if (change) emitBridgePush('policy:changed', change)
                break
            }
            case 'engine:state-changed': {
                const engine = objectValue(params)?.engine
                if (typeof engine === 'string') updatedEngines.set(engine, ++engineRevision)
                state.applyEngineManagerStatus(params)
                void readModels().catch(() => {})
                break
            }
            case 'engine:models-changed':
                if (
                    state.applyLocalModelsNotification(
                        parseModelsResult(objectValue(params)?.models)
                    )
                )
                    void readModels().catch(() => {})
                break
            case 'engine:install-progress':
                state.applyEngineManagerProgress(params)
                break
            case 'engine:pull-progress':
                if (state.applyLocalEngineProgress(params)) void readModels().catch(() => {})
                break
            case 'engine:remote-progress':
                state.applyRemoteEngineProgress(params)
                break
            case 'engine:remote-installed-changed': {
                const nodeId = objectValue(params)?.nodeId
                if (typeof nodeId === 'string' && nodeId !== state.getSelfId()) {
                    remoteRevisions.set(nodeId, (remoteRevisions.get(nodeId) ?? 0) + 1)
                    state.applyRemoteEngineFacts(nodeId, params)
                }
                break
            }
            case 'engine:settings-changed':
                emitBridgePush('engines:settings-changed', parseEngineSettings(params))
                break
            case 'engine:settings-disconnected': {
                const nodeId = objectValue(params)?.nodeId
                if (typeof nodeId === 'string')
                    emitBridgePush('engines:settings-disconnected', { nodeId })
                break
            }
            default:
                for (const engine of PROXY_ENGINES) {
                    const prefix = `${proxySourceForEngine(engine)}:`
                    if (!method.startsWith(prefix)) continue
                    const event = method.slice(prefix.length)
                    const node = parseProxyNode(params, engine)
                    if (node) proxyChangesDuringBaseline.get(engine)?.add(node.id)
                    applyProxy(engine, event, params)
                }
        }
    }

    return {
        invoke<C extends OverviewRead>(channel: C, payload: WsInvokeRequest<C> | undefined) {
            return handlers[channel](payload)
        },
        project,
        retire() {
            retired = true
            stopNodeInfoPoller()
            for (const [engine, nodes] of proxyNodes) {
                for (const params of [...nodes.values()]) applyProxy(engine, 'node/removed', params)
            }
            applyDiscovery({ nodes: [] })
            performance.reset()
            state.clearWorkloads()
            state.replaceErrors([])
            state.invalidateModelObservations()
        }
    }
}
