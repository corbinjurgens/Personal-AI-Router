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

/** The renderer's service transport. A failed invoke rejects with an error carrying the service's code. */
export interface ServiceTransport {
    readonly connected: boolean
    invoke<C extends WsInvokeChannel>(...args: ServiceInvokeArgs<C>): Promise<WsInvokeResponse<C>>
    subscribePush<C extends WsPushChannel>(channel: C, fn: ServiceBridgePushHandler<C>): () => void
    onConnect(fn: ServiceBridgeLifecycleHandler): () => void
    onDisconnect(fn: ServiceBridgeLifecycleHandler): () => void
    onAuthFailure(fn: ServiceBridgeLifecycleHandler): () => void
    destroy(): void
}

/**
 * The transport the preload exposes. Its invoke resolves to the IPC result, so
 * a failure's code survives the context bridge, which copies only an Error's
 * message; the renderer unwraps it into a ServiceTransport.
 */
export interface PreloadServiceTransport extends Omit<ServiceTransport, 'invoke'> {
    invoke<C extends WsInvokeChannel>(
        ...args: ServiceInvokeArgs<C>
    ): Promise<IpcResult<WsInvokeResponse<C>>>
}
