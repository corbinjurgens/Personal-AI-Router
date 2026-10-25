// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ipcRenderer } from 'electron'
import type { IpcResult } from '@/shared/types/ipc-channels'
import {
    isHistoryInvoke,
    type PreloadServiceTransport,
    type ServiceBridgeInvokeRequest,
    type ServiceBridgePushMessage,
    type ServiceInvokeArgs
} from '@/shared/types/service-bridge'
import type { WsInvokeChannel, WsInvokeResponse, WsPushChannel } from '@/shared/types/ws-channels'
import type { RoutingHistoryReady } from '@/shared/types/routing'

type PushListener = (event: Electron.IpcRendererEvent, message: ServiceBridgePushMessage) => void
type ReadyListener = (event: Electron.IpcRendererEvent, message: RoutingHistoryReady) => void
interface PairTransportIpc {
    invoke: typeof ipcRenderer.invoke
    on(channel: 'service-bridge:push', listener: PushListener): void
    on(channel: 'routing-history:ready', listener: ReadyListener): void
    removeListener(channel: 'service-bridge:push', listener: PushListener): void
    removeListener(channel: 'routing-history:ready', listener: ReadyListener): void
}
function isPushForChannel<C extends WsPushChannel>(
    message: ServiceBridgePushMessage,
    channel: C
): message is ServiceBridgePushMessage<C> {
    return message.channel === channel
}

/** Each preload context has ordinary subscribers and scalar history controls only. */
export function createPairTransport(ipc: PairTransportIpc): PreloadServiceTransport {
    const pushSubscribers = new Set<(message: ServiceBridgePushMessage) => void>()
    let attached = false,
        historyWanted = false,
        historyActive = false,
        historyClosing = false,
        historyLoan = false
    const publish = (message: ServiceBridgePushMessage): void => {
        for (const subscriber of Array.from(pushSubscribers)) subscriber(message)
    }
    const handlePush: PushListener = (_event, message) => publish(message)
    const handleReady: ReadyListener = (_event, message) =>
        publish({ channel: 'routing-history:ready', payload: message })
    const attach = (): void => {
        if (attached) return
        ipc.on('service-bridge:push', handlePush)
        ipc.on('routing-history:ready', handleReady)
        attached = true
    }
    const detach = (): void => {
        if (!attached || pushSubscribers.size > 0) return
        ipc.removeListener('service-bridge:push', handlePush)
        ipc.removeListener('routing-history:ready', handleReady)
        attached = false
    }
    const publishHistory = (
        channel: WsInvokeChannel,
        lease: number,
        result: IpcResult<WsInvokeResponse<WsInvokeChannel>>
    ): void => {
        if (!result.success) return
        if (channel === 'routing:unsubscribe') {
            historyWanted = false
            publish({
                channel: 'routing-history:committed',
                payload: { generation: 0, revision: 0 }
            })
            return
        }
        const data = result.data
        if (
            !data ||
            typeof data !== 'object' ||
            !('generation' in data) ||
            typeof data.generation !== 'number' ||
            !('state' in data)
        )
            return
        if (channel === 'routing-history:pull') {
            if (data.state === 'data' || data.state === 'end') historyLoan = true
            if (data.state === 'busy' && 'loading' in data && typeof data.loading === 'boolean')
                publish({
                    channel: 'routing-history:committed',
                    payload: { generation: data.generation, loading: data.loading }
                })
            if (data.state === 'retired')
                publish({
                    channel: 'routing-history:committed',
                    payload: { generation: data.generation, revision: 0 }
                })
        } else if (
            channel === 'routing-history:consumed' &&
            (data.state === 'accepted' || data.state === 'retired')
        ) {
            historyLoan = false
            publish({
                channel: 'routing-history:committed',
                payload: {
                    generation: data.generation,
                    revision: data.state === 'accepted' ? lease : 0
                }
            })
        }
    }
    const drainDestroy = (): void => {
        // Receipt is not consumption. A pending destroy cannot race a packet's
        // renderer ACK for the one native completion slot or reclaim its loan.
        if (!historyClosing || historyActive || historyLoan) return
        historyClosing = false
        if (historyWanted) invokeHistory(() => undefined, 'routing:unsubscribe')
    }
    function invokeHistory<C extends WsInvokeChannel>(
        complete: (result: IpcResult<WsInvokeResponse<C>>) => undefined,
        ...args: ServiceInvokeArgs<C>
    ): undefined {
        const [channel, payload] = args
        if (!isHistoryInvoke(channel) || channel === 'routing:get-initial' || historyActive) {
            if (
                complete({
                    success: false,
                    error: historyActive ? 'history-busy' : 'history-invalid'
                }) !== undefined
            )
                throw new Error('history-invalid')
            return undefined
        }
        historyActive = true
        if (channel === 'routing:subscribe') historyWanted = true
        const lease =
            payload &&
            typeof payload === 'object' &&
            'lease' in payload &&
            typeof payload.lease === 'number'
                ? payload.lease
                : 0
        // The only cross-world callback loan. Both native success and failure
        // detach it before completion; no callback registry or pending queue.
        let target: typeof complete | null = complete
        const finish = (result: IpcResult<WsInvokeResponse<C>> | null): undefined => {
            let consume = target
            target = null
            if (!consume || !result) return undefined
            try {
                try {
                    publishHistory(channel, lease, result)
                } catch {
                    result = { success: false, error: 'history-invalid' }
                }
                if (consume(result) !== undefined) throw new Error('history-invalid')
            } finally {
                consume = null
                result = null
                historyActive = false
                drainDestroy()
            }
            return undefined
        }
        try {
            const request: ServiceBridgeInvokeRequest<C> = { channel, payload }
            const pending: Promise<IpcResult<WsInvokeResponse<C>>> =
                channel === 'routing-history:pull' || channel === 'routing-history:consumed'
                    ? ipc.invoke(channel, payload)
                    : ipc.invoke('service-bridge:invoke', request)
            // Native invoke Promises remain in the isolated preload world.
            // Every reaction returns undefined, including rejected IPC calls.
            void pending
                .then(finish, () => finish({ success: false, error: 'history-invalid' }))
                .catch(() => {})
        } catch {
            finish({ success: false, error: 'history-invalid' })
        }
        return undefined
    }
    return {
        get connected() {
            return true
        },
        invoke<C extends WsInvokeChannel>(
            ...args: ServiceInvokeArgs<C>
        ): Promise<IpcResult<WsInvokeResponse<C>>> {
            const [channel, payload] = args
            if (isHistoryInvoke(channel)) throw new Error('history-invalid')
            const request: ServiceBridgeInvokeRequest<C> = { channel, payload }
            return ipc.invoke('service-bridge:invoke', request)
        },
        invokeHistory,
        subscribePush(channel, fn) {
            const subscriber = (message: ServiceBridgePushMessage): void => {
                if (isPushForChannel(message, channel)) fn(message.payload)
            }
            pushSubscribers.add(subscriber)
            attach()
            return () => {
                pushSubscribers.delete(subscriber)
                detach()
            }
        },
        onConnect(fn) {
            queueMicrotask(fn)
            return () => {}
        },
        onDisconnect() {
            return () => {}
        },
        onAuthFailure() {
            return () => {}
        },
        destroy() {
            historyClosing = true
            pushSubscribers.clear()
            detach()
            drainDestroy()
        }
    }
}

export const pairTransport = createPairTransport(ipcRenderer)
