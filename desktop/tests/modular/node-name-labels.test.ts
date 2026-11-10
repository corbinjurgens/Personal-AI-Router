// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { NodeItem } from '@/shared/types/nodes'
import type { ClusterNode } from '@/shared/types/cluster'
import { distinguishingLabels, duplicateShownName } from '@/ui/utils/node-name-labels'
import { offlineMemberNode } from '@/ui/utils/overview-nodes'

function node(id: string, name: string, hardware: { gpu?: string; cpu?: string } = {}): NodeItem {
    return {
        id,
        name,
        status: 'active',
        ipAddress: '',
        port: 0,
        allIpAddresses: [],
        topology: {
            cpu: { model: hardware.cpu ?? '', cores: 0, threads: 0 },
            gpus: hardware.gpu ? [{ id: `${id}-gpu`, name: hardware.gpu, vramTotal: 0 }] : [],
            ram: 0,
            storage: []
        },
        os: 'Windows'
    }
}

describe('distinguishing labels', () => {
    it('gives nodes with unique names no label', () => {
        const labels = distinguishingLabels([
            node('aaaaaaaa-1', 'Studio', { gpu: 'RTX 4090' }),
            node('bbbbbbbb-2', 'Desk', { gpu: 'RTX 4090' })
        ])
        expect(labels.size).toBe(0)
    })

    it('labels same-named nodes by GPU, then CPU, then ID', () => {
        const labels = distinguishingLabels([
            node('aaaaaaaa-1', 'Studio', { gpu: 'RTX 4090', cpu: 'Ryzen 9' }),
            node('bbbbbbbb-2', 'Studio', { cpu: 'Apple M4' }),
            node('cccccccc-3', 'Studio')
        ])
        expect(Object.fromEntries(labels)).toEqual({
            'aaaaaaaa-1': 'RTX 4090',
            'bbbbbbbb-2': 'Apple M4',
            'cccccccc-3': 'ID cccccccc'
        })
    })

    it('adds the UUID prefix when labels in a group are still equal', () => {
        const labels = distinguishingLabels([
            node('aaaaaaaa-1', 'Studio', { gpu: 'RTX 4090' }),
            node('bbbbbbbb-2', 'Studio', { gpu: 'RTX 4090' }),
            node('cccccccc-3', 'Studio', { gpu: 'RTX 4080' })
        ])
        expect(Object.fromEntries(labels)).toEqual({
            'aaaaaaaa-1': 'RTX 4090 (aaaaaaaa)',
            'bbbbbbbb-2': 'RTX 4090 (bbbbbbbb)',
            'cccccccc-3': 'RTX 4080'
        })
    })
})

describe('labels follow the duplicate warning comparison', () => {
    it('groups names that differ only by case or surrounding spaces', () => {
        const names = ['Studio', 'studio', ' STUDIO ']
        const nodes = names.map((name, index) => node(`id${index}aaaaaa`, name))
        expect(duplicateShownName(' studio ', ['Studio'])).toBe('Studio')
        expect([...distinguishingLabels(nodes).keys()]).toEqual(nodes.map(value => value.id))
    })

    it('does not group nodes without a name', () => {
        expect(distinguishingLabels([node('aaaaaaaa-1', ''), node('bbbbbbbb-2', '  ')]).size).toBe(
            0
        )
    })

    it('labels a pending member that shares a name with a card', () => {
        const pending: ClusterNode = {
            nodeUuid: 'dddddddd-4',
            name: 'studio',
            displayName: 'studio',
            ipAddress: '192.0.2.4',
            port: 1,
            clusterId: 'home',
            state: 'pending-outbound',
            joinedAt: null,
            lastSeen: null
        }
        const labels = distinguishingLabels([
            node('aaaaaaaa-1', 'Studio', { gpu: 'RTX 4090' }),
            offlineMemberNode(pending)
        ])
        expect(Object.fromEntries(labels)).toEqual({
            'aaaaaaaa-1': 'RTX 4090',
            'dddddddd-4': 'ID dddddddd'
        })
    })
})

describe('duplicate shown name', () => {
    it('matches ignoring case and surrounding spaces', () => {
        expect(duplicateShownName('  studio ', ['Desk', 'Studio'])).toBe('Studio')
    })

    it('returns null when no other node has the name', () => {
        expect(duplicateShownName('Studio', ['Desk'])).toBeNull()
    })

    it('returns null for an empty candidate', () => {
        expect(duplicateShownName('   ', ['Desk'])).toBeNull()
    })
})
