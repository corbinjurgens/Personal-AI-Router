// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { safeHandle } from '@/electron/ipc/safe-handle'
import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'
import { getRoutingDelivery } from '@/electron/service-bridge/routing-delivery'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import { BrowserWindow, type IpcMainInvokeEvent } from 'electron'

function historyWindow(event: IpcMainInvokeEvent): BrowserWindow {
    if (event.senderFrame !== event.sender.mainFrame) throw new Error('history-invalid')
    const window = BrowserWindow.fromWebContents(event.sender)
    if (!window || window.isDestroyed()) throw new Error('history-invalid')
    return window
}

export function registerServiceBridgeIpc(): void {
    getRoutingDelivery().setDemandListener(active =>
        getModularSupervisor().setRoutingHistorySubscribed(active)
    )
    safeHandle('service-bridge:invoke', async (event, payload) => {
        if (payload.channel === 'routing:subscribe') {
            const generation = getRoutingDelivery().subscribe(historyWindow(event))
            return { generation }
        }
        if (payload.channel === 'routing:unsubscribe') {
            getRoutingDelivery().unsubscribe(historyWindow(event))
            return null
        }
        return handleServiceBridgeInvoke(payload.channel, payload.payload)
    })
    safeHandle('routing-history:pull', (event, payload) => {
        return getRoutingDelivery().pull(historyWindow(event), payload)
    })
    safeHandle('routing-history:consumed', (event, payload) => {
        return getRoutingDelivery().consumed(historyWindow(event), payload)
    })
}
