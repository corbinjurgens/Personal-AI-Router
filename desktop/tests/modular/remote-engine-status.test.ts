// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))

import { getModularBridgeState } from '@/electron/service-bridge/modular-state'

describe('remote engine status', () => {
    it('prefers authoritative stopped facts over proxy presence', () => {
        const state = getModularBridgeState()
        const remoteNodeId = 'remote-engine-status-remote'

        state.setSelfId('remote-engine-status-local')
        state.handleNotification({
            source: 'llamacpp-proxy',
            method: 'node/discovered',
            params: {
                id: remoteNodeId,
                host: remoteNodeId,
                port: 8080,
                addresses: ['192.0.2.190'],
                ip: '192.0.2.190'
            }
        })
        expect(
            state.getEngineInitialState().statuses.find(status => status.nodeId === remoteNodeId)
        ).toMatchObject({ processStatus: 'running' })

        state.applyRemoteEngineFacts(remoteNodeId, {
            engines: [
                {
                    engine: 'llamacpp',
                    installed: true,
                    running: false,
                    healthy: false,
                    port: 8081
                }
            ]
        })

        expect(
            state
                .getEngineInitialState()
                .statuses.find(
                    status => status.nodeId === remoteNodeId && status.engineType === 'llama-cpp'
                )
        ).toMatchObject({
            processStatus: 'stopped',
            enginePort: 8081,
            // The proxy remains discoverable even though the engine behind it is stopped.
            proxyPort: 8080
        })
    })
})
