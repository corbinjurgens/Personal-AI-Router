// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { NodeItem } from '@/shared/types/nodes'
import { distinguishingLabels, duplicateShownName } from '@/ui/utils/node-name-labels'

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
