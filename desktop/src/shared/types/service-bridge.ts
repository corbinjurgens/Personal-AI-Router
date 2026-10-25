// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { IpcResult } from '@/shared/types/ipc-channels'
import type {
    WsInvokeChannel,
    WsInvokeRequest,
    WsPushChannel,
    WsPushPayload,
    WsInvokeResponse
} from '@/shared/types/ws-channels'

export interface ServiceBridgeInvokeRequest<C extends WsInvokeChannel = WsInvokeChannel> {
    channel: C
    payload?: WsInvokeRequest<C>
}

export interface ServiceBridgePushMessage<C extends WsPushChannel = WsPushChannel> {
    channel: C
    payload: WsPushPayload<C>
}

export type ServiceBridgePushHandler<C extends WsPushChannel> = (payload: WsPushPayload<C>) => void
export type ServiceBridgeLifecycleHandler = () => void

export type ServiceInvokeArgs<C extends WsInvokeChannel> =
    WsInvokeRequest<C> extends void ? [channel: C] : [channel: C, payload: WsInvokeRequest<C>]

/** Only this private history family uses the synchronous contextBridge adapter. */
export function isHistoryInvoke(channel: WsInvokeChannel): boolean {
    return (
        channel === 'routing:subscribe' ||
        channel === 'routing:unsubscribe' ||
        channel === 'routing:get-initial' ||
        channel === 'routing-history:pull' ||
        channel === 'routing-history:consumed'
    )
}

/** The renderer's service transport. A failed invoke rejects with an error carrying the service's code. */
export interface ServiceTransport {
    readonly connected: boolean
    invoke<C extends WsInvokeChannel>(...args: ServiceInvokeArgs<C>): Promise<WsInvokeResponse<C>>
    /** History ready tokens are scalar. The renderer commits pull bytes before consumed. */
    subscribePush<C extends WsPushChannel>(channel: C, fn: ServiceBridgePushHandler<C>): () => void
    onConnect(fn: ServiceBridgeLifecycleHandler): () => void
    onDisconnect(fn: ServiceBridgeLifecycleHandler): () => void
    onAuthFailure(fn: ServiceBridgeLifecycleHandler): () => void
    destroy(): void
}

/**
 * The ordinary invoke resolves to an IPC result so failure codes survive the
 * context bridge. History uses the synchronous callback entry exclusively;
 * its completion Promises belong to the renderer ServiceTransport owner.
 */
export interface PreloadServiceTransport extends Omit<ServiceTransport, 'invoke'> {
    invoke<C extends WsInvokeChannel>(
        ...args: ServiceInvokeArgs<C>
    ): Promise<IpcResult<WsInvokeResponse<C>>>
    /**
     * History only; rejects other channels. Neither this call nor its completion
     * returns a Promise across contextBridge. Completion is called once and its
     * result/callback references are detached before another native admission.
     */
    invokeHistory<C extends WsInvokeChannel>(
        complete: (result: IpcResult<WsInvokeResponse<C>>) => undefined,
        ...args: ServiceInvokeArgs<C>
    ): undefined
}
