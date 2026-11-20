// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ServiceBridgePushMessage } from '@/shared/types/service-bridge'
import type { WsPushChannel, WsPushPayload } from '@/shared/types/ws-channels'

export type ProjectionPush = (message: ServiceBridgePushMessage) => void

let send: ProjectionPush | undefined

export function attachProjectionPush(push: ProjectionPush): () => void {
    send = push
    return () => {
        send = undefined
    }
}

// The experiment bundle binds the stock projection's broadcaster to this window.
export function emitBridgePush<C extends WsPushChannel>(
    channel: C,
    payload: WsPushPayload<C>
): void {
    send?.({ channel, payload })
}
