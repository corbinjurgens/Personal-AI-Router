// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'

const pushes = vi.hoisted(() => ({ channels: [] as string[] }))

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))
vi.mock('@/electron/service-bridge/broadcaster', () => ({
    emitBridgePush: (channel: string): void => {
        pushes.channels.push(channel)
    }
}))

import { getModularBridgeState } from '@/electron/service-bridge/modular-state'

const NODE = 'uuid-telemetry'

function nodeInfo(overrides: {
    gpuName?: string
    vramUsed?: number
    gpuUtil?: number
    cpuUtil?: number
    memUsed?: number
    memTotal?: number
}) {
    return {
        hostUuid: NODE,
        GPUs: [
            {
                name: overrides.gpuName ?? 'NVIDIA A',
                vram_bytes: 1000,
                vram_used_bytes: overrides.vramUsed ?? 10,
                utilization_percent: overrides.gpuUtil ?? 5
            }
        ],
        cpu: { name: 'CPU-A', cores: 8, utilization_percent: overrides.cpuUtil ?? 3 },
        memory: { total_bytes: overrides.memTotal ?? 4000, used_bytes: overrides.memUsed ?? 100 }
    }
}

// A poll tick where only utilization or memory-in-use moved used to push the
// node card, the whole discovery list, and every engine's status for the node,
// none of which carry those readings. Only `metrics:update` does.
describe('node-info telemetry pushes', () => {
    beforeEach(() => {
        const state = getModularBridgeState()
        state.handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: {
                nodes: [
                    { hostUuid: NODE, name: 'telemetry-host', ipAddress: '192.0.2.90', port: 14318 }
                ]
            }
        })
        state.mergeNodeInfoResponse(NODE, nodeInfo({}))
        pushes.channels.length = 0
    })

    it('pushes only metrics when live readings change', () => {
        getModularBridgeState().mergeNodeInfoResponse(
            NODE,
            nodeInfo({ gpuUtil: 90, vramUsed: 800, cpuUtil: 70, memUsed: 3000 })
        )

        expect(pushes.channels).toEqual(['metrics:update'])
    })

    it('keeps the new readings for the next metrics push', () => {
        const state = getModularBridgeState()
        state.mergeNodeInfoResponse(NODE, nodeInfo({ gpuUtil: 90 }))
        pushes.channels.length = 0

        // Same readings again: still a metrics-only push, not a node change.
        state.mergeNodeInfoResponse(NODE, nodeInfo({ gpuUtil: 90 }))
        expect(pushes.channels).toEqual(['metrics:update'])
    })

    it('pushes the full node change when the hardware changes', () => {
        getModularBridgeState().mergeNodeInfoResponse(NODE, nodeInfo({ gpuName: 'NVIDIA B' }))

        expect(pushes.channels).toContain('nodes:upsert')
        expect(pushes.channels).toContain('discovery:nodes-changed')
        expect(getModularBridgeState().getNodesInitial().nodes[NODE].topology.gpus[0].name).toBe(
            'NVIDIA B'
        )
    })

    it('pushes the full node change when total memory changes', () => {
        getModularBridgeState().mergeNodeInfoResponse(NODE, nodeInfo({ memTotal: 8000 }))

        expect(pushes.channels).toContain('nodes:upsert')
    })
})
