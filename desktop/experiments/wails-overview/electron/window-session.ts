// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Duplex } from 'node:stream'
import { JsonRpcSocketClient } from '@/electron/service-bridge/json-rpc-client'
import type { ServiceBridgeInvokeRequest } from '@/shared/types/service-bridge'
import type { WsInvokeChannel, WsInvokeResponse } from '@/shared/types/ws-channels'
import { isOverviewRead, type OverviewRead } from '../ui/overview-reads'
import { createOverviewProjection } from './projection'

type Invoke = (request: ServiceBridgeInvokeRequest) => Promise<WsInvokeResponse<WsInvokeChannel>>

interface WindowIpc {
    register(invoke: Invoke): () => void
    generation(id: number | null): void
}

let nextGeneration = 0

function isReadRequest(
    request: ServiceBridgeInvokeRequest
): request is ServiceBridgeInvokeRequest<OverviewRead> {
    return typeof request?.channel === 'string' && isOverviewRead(request.channel)
}

export function bindOverviewWindow(ipc: WindowIpc, dial: () => Promise<Duplex>): () => void {
    const client = new JsonRpcSocketClient('broker', dial, undefined, dial)
    let current: { id: number; projection: ReturnType<typeof createOverviewProjection> } | null =
        null

    function retire() {
        const previous = current
        current = null
        ipc.generation(null)
        previous?.projection.retire()
    }

    const unbind = ipc.register(async request => {
        if (!isReadRequest(request)) {
            throw new Error('Overview request is outside the read allowlist')
        }
        const active = current
        if (!active || !('generation' in request) || request.generation !== active.id) {
            throw new Error('Overview generation retired')
        }
        const result = await active.projection.invoke(request.channel, request.payload)
        if (active !== current) throw new Error('Overview generation retired')
        return result
    })

    client.on('notification', notification => {
        if (notification.method === 'app:ready') {
            if (!current) {
                current = { id: ++nextGeneration, projection: createOverviewProjection(client) }
                ipc.generation(current.id)
            }
        } else {
            try {
                current?.projection.project(notification)
            } catch {
                // A malformed push is ignored. Snapshot reads report failures separately.
            }
        }
    })
    client.on('broker-restarted', retire)
    client.on('disconnected', retire)
    client.on('exit', retire)
    client.start()

    let closed = false
    return () => {
        if (closed) return
        closed = true
        unbind()
        client.removeAllListeners()
        retire()
        void client.detach()
    }
}
