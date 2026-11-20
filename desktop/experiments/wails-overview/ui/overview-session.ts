// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ClusterInitialSnapshot } from '@/shared/types/bootstrap'
import type { ClusterNode, NodeNameResult } from '@/shared/types/cluster'
import type {
    ServiceBridgePushHandler,
    ServiceInvokeArgs,
    ServiceTransport
} from '@/shared/types/service-bridge'
import type { WsInvokeChannel, WsInvokeResponse, WsPushChannel } from '@/shared/types/ws-channels'
import { createPairApi } from '@/ui/api/pair-api'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { useErrorsStore } from '@/ui/stores/errors.store'
import { useMetricsStore } from '@/ui/stores/metrics.store'
import { initializeNodeAvailability } from '@/ui/stores/node-configuration.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useWorkloadsStore } from '@/ui/stores/workloads.store'
import type { HostInput } from './host-input'

export type OverviewStatus = 'connecting' | 'ready' | 'unavailable'

const baselines = new Set<WsInvokeChannel>([
    'app:get-initial',
    'cluster:get-initial',
    'nodes:get-initial',
    'engines:get-initial',
    'workloads:get-initial',
    'discovery:get-nodes',
    'errors:get-initial'
])
const allowedReads = new Set<WsInvokeChannel>([
    ...baselines,
    'cluster:get-remote-management',
    'engine:remote-management-get',
    'engines:get-settings',
    'policy:get'
])

function generationTransport(source: ServiceTransport) {
    const controller = new AbortController()
    const signal = controller.signal
    const received = new Set<WsInvokeChannel>()
    const subscriptions = new Set<() => void>()
    const retired = new Error('Overview generation retired')
    const noSubscription = () => () => {}
    const transport: ServiceTransport = {
        get connected() {
            return !signal.aborted && source.connected
        },
        invoke<C extends WsInvokeChannel>(...args: ServiceInvokeArgs<C>) {
            const [channel] = args
            if (signal.aborted) return Promise.reject(retired)
            if (!allowedReads.has(channel)) {
                return Promise.reject(new Error(`Overview is read-only: ${channel}`))
            }
            return new Promise<WsInvokeResponse<C>>((resolve, reject) => {
                const abort = () => reject(retired)
                signal.addEventListener('abort', abort, { once: true })
                void Promise.resolve()
                    .then(() => {
                        if (signal.aborted) throw retired
                        return source.invoke<C>(...args)
                    })
                    .then(result => {
                        if (signal.aborted) throw retired
                        if (baselines.has(channel)) {
                            if (result == null)
                                throw new Error(`Missing Overview baseline: ${channel}`)
                            received.add(channel)
                        }
                        resolve(result)
                    })
                    .catch(reject)
                    .finally(() => signal.removeEventListener('abort', abort))
            })
        },
        subscribePush<C extends WsPushChannel>(channel: C, handler: ServiceBridgePushHandler<C>) {
            if (signal.aborted) return () => {}
            const stopSource = source.subscribePush(channel, value => {
                if (!signal.aborted) handler(value)
            })
            const stop = () => {
                if (subscriptions.delete(stop)) stopSource()
            }
            subscriptions.add(stop)
            return stop
        },
        // Window and broker changes arrive through HostInput, once per generation.
        onConnect: noSubscription,
        onDisconnect: noSubscription,
        onAuthFailure: noSubscription,
        destroy() {
            if (signal.aborted) return
            controller.abort()
            subscriptions.forEach(stop => stop())
            source.destroy()
        }
    }
    return { transport, ready: () => received.size === baselines.size, signal }
}

/** Full-state streams can retain the latest push while their stores subscribe after reading. */
function observeBaseline<T>(subscribe: (callback: (value: T) => void) => () => void) {
    let observed: { value: T } | undefined
    const stop = subscribe(value => {
        observed = { value }
    })
    return (callback: (value: T) => void) => {
        const unsubscribe = subscribe(callback)
        stop()
        if (observed) callback(observed.value)
        observed = undefined
        return unsubscribe
    }
}

async function initializeOverview(session: ReturnType<typeof generationTransport>): Promise<void> {
    const api = createPairApi(session.transport)
    window.pairApi = api
    const clusterStore = useClusterInvitationsStore.getState()
    let cluster: ClusterInitialSnapshot | undefined
    let members: ClusterNode[] | undefined
    let name: NodeNameResult | undefined
    const hydrateRoster = () => {
        if (!cluster) return
        const identity = cluster.identity
        clusterStore.hydrate({
            ...cluster,
            identity:
                name?.nodeUuid === identity.nodeUuid
                    ? { ...identity, name: name.name, customName: name.customName }
                    : identity,
            members: members ?? cluster.members,
            pendingInvites: []
        })
    }
    // Use the production roster projection without starting the pairing poller.
    api.nodes.onMembersChanged(value => {
        members = value
        hydrateRoster()
    })
    api.cluster.onNodeNameChanged(value => {
        name = value
        hydrateRoster()
    })
    api.discovery.onNodesChanged = observeBaseline(api.discovery.onNodesChanged)
    api.errors.onUpdate = observeBaseline(api.errors.onUpdate)
    useMetricsStore.getState().initialize()
    const stopAvailability = initializeNodeAvailability()
    session.signal.addEventListener('abort', stopAvailability, { once: true })

    const engine = api.engines.getInitialState()
    const appAndCluster = Promise.all([api.app.getInitial(), api.cluster.getInitial()]).then(
        ([app, baseline]) => {
            if (session.signal.aborted) return
            if (!app.connected || !app.selfId || app.selfId !== baseline.identity.nodeUuid) {
                throw new Error('Overview identity is unavailable')
            }
            cluster = baseline
            hydrateRoster()
            return useConnectionStore.getState().initialize(app, baseline)
        }
    )
    const results = await Promise.allSettled([
        appAndCluster,
        useNodesStore.getState().initialize(),
        useWorkloadsStore.getState().initialize(),
        useEngineStatusStore.getState().initialize(engine),
        useEngineModelsStore.getState().initialize(engine),
        useDiscoveredNodesStore.getState().initialize(),
        useErrorsStore.getState().initialize()
    ])
    if (
        results.some(result => result.status === 'rejected') ||
        !session.transport.connected ||
        !session.ready() ||
        !useNodesStore.getState().fetchedNodes
    ) {
        throw new Error('Overview baselines are unavailable')
    }
}

function clearOverviewStores(): void {
    useConnectionStore.getState().cleanup()
    useConnectionStore.setState(useConnectionStore.getInitialState())
    useClusterInvitationsStore.getState().cleanup()
    useNodesStore.getState().cleanup()
    useNodesStore.setState({ nodes: new Map(), fetchedNodes: false })
    useWorkloadsStore.getState().cleanup()
    useWorkloadsStore.setState({ workloads: new Map() })
    useMetricsStore.getState().cleanup()
    useMetricsStore.getState().clearAll()
    useEngineStatusStore.getState().cleanup()
    useEngineStatusStore.setState({ statusByNode: new Map(), descriptors: new Map() })
    useEngineModelsStore.getState().cleanup()
    useEngineModelsStore.setState({ models: new Map() })
    useDiscoveredNodesStore.getState().cleanup()
    useErrorsStore.getState().cleanup()
    useErrorsStore.setState({ errors: [] })
}

export function connectOverview(input: HostInput, onStatus: (status: OverviewStatus) => void) {
    // The mounted read-only tree uses only the platform field of the native API.
    Object.defineProperty(window, 'windowApi', {
        configurable: true,
        value: { platform: input.platform }
    })
    let current: ReturnType<typeof generationTransport> | undefined
    let tail = Promise.resolve()
    let disposed = false
    const stop = input.subscribe(source => {
        if (disposed) {
            source?.destroy()
            return
        }
        current?.transport.destroy()
        onStatus(source ? 'connecting' : 'unavailable')
        const next = source ? generationTransport(source) : undefined
        current = next
        // Some production store initializers catch failed reads and settle their
        // own state. Drain the retired calls before clearing or seeding again.
        tail = tail.then(async () => {
            if (next !== current || disposed) return
            clearOverviewStores()
            if (!next) return
            try {
                await initializeOverview(next)
                if (current === next) onStatus('ready')
            } catch {
                if (current !== next) return
                next.transport.destroy()
                clearOverviewStores()
                onStatus('unavailable')
            }
        })
    })
    return () => {
        disposed = true
        current?.transport.destroy()
        current = undefined
        stop()
        void tail.then(clearOverviewStores)
    }
}
