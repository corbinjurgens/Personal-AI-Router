// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { contextBridge, ipcRenderer } from 'electron'
import { createPairTransport } from '@/preload/api/pair-transport.api'
import type {
    PreloadServiceTransport,
    ServiceBridgeInvokeRequest
} from '@/shared/types/service-bridge'
import { currentPlatform, platformDisplayName } from '@/shared/utils/platform'
import type { LiveOverviewHost } from '../ui/host-input'
import { overviewGenerationChannel, overviewInvokeChannel } from './ipc'

const subscribers = new Set<Parameters<LiveOverviewHost['onGeneration']>[0]>()
let current: PreloadServiceTransport | null = null

ipcRenderer.on(overviewGenerationChannel, (_event, generation: number | null) => {
    current?.destroy()
    current =
        generation === null
            ? null
            : createPairTransport({
                  invoke: (channel: string, request?: ServiceBridgeInvokeRequest) => {
                      if (channel !== overviewInvokeChannel)
                          return Promise.resolve({
                              success: false,
                              error: 'Overview request is outside the read allowlist'
                          })
                      return ipcRenderer.invoke(overviewInvokeChannel, { ...request, generation })
                  },
                  on: (channel, listener) => ipcRenderer.on(channel, listener),
                  removeListener: (channel, listener) =>
                      ipcRenderer.removeListener(channel, listener)
              })
    for (const listener of subscribers) listener(current)
})

const overviewHost: LiveOverviewHost = {
    platform: platformDisplayName(currentPlatform()),
    onGeneration(listener) {
        subscribers.add(listener)
        listener(current)
        return () => {
            subscribers.delete(listener)
        }
    }
}
contextBridge.exposeInMainWorld('__PAIR_OVERVIEW_HOST__', overviewHost)
