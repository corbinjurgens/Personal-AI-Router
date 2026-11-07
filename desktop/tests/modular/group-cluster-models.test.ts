// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { EngineModels, ModelIdentity, ModelItem } from '@/shared/types/engines'
import type { ModelDownloadEntry } from '@/shared/types/model-download'
import { groupClusterModels } from '@/ui/utils/group-cluster-models'

function inventory(nodeId: string, name: string, identity?: ModelIdentity): EngineModels {
    const model: ModelItem = {
        name,
        identity,
        size: 100,
        downloaded: true,
        status: 'idle',
        parameterSize: '',
        family: '',
        sizeVram: null,
        expiresAt: null,
        expiry: '0',
        capabilities: []
    }
    return { nodeId, engineType: 'llama-cpp', models: [model] }
}

const declared: ModelIdentity = {
    strength: 'declared',
    repository: 'owner/Tiel',
    quantization: 'Q4_K_M',
    totalBytes: 100
}

describe('cluster model display grouping', () => {
    it.each<{ label: string; reason: ModelDownloadEntry['reason'] }>([
        { label: 'failed', reason: 'download_failed' },
        { label: 'cancelled', reason: 'interrupted' }
    ])('excludes $label download-only rows from holdings', ({ reason }) => {
        const operation = inventory('joes', 'not-present', declared)
        operation.models[0].downloaded = false
        operation.models[0].downloadState = {
            id: '1'.repeat(32),
            engine: 'llamacpp',
            model: 'not-present',
            phase: 'failed',
            reason,
            blocksReadiness: true
        }
        const groups = groupClusterModels([inventory('msi', 'present', declared), operation])
        expect(
            groups
                .flatMap(group => group.variants)
                .flatMap(variant => variant.locations)
                .map(location => location.nodeId)
        ).toEqual(['msi'])
    })
    it('groups Tiel with the same contentId on two nodes into one variant with both locations', () => {
        const identity: ModelIdentity = {
            ...declared,
            strength: 'verified',
            contentId: 'a'.repeat(64)
        }
        const groups = groupClusterModels([
            inventory('msi', 'tiel-msi', identity),
            inventory('joes', 'tiel-joes', identity)
        ])
        expect(groups).toHaveLength(1)
        expect(groups[0].variants).toHaveLength(1)
        expect(groups[0].variants[0].unverified).toBe(false)
        expect(groups[0].variants[0].locations.map(row => [row.nodeId, row.model.name])).toEqual([
            ['msi', 'tiel-msi'],
            ['joes', 'tiel-joes']
        ])
    })

    it('groups matching declared repository, quantization and size with an unverified marker', () => {
        const groups = groupClusterModels([
            inventory('msi', 'tiel-msi', declared),
            inventory('joes', 'tiel-joes', declared)
        ])
        expect(groups).toHaveLength(1)
        expect(groups[0].name).toBe('owner/Tiel')
        expect(groups[0].variants).toHaveLength(1)
        expect(groups[0].variants[0]).toMatchObject({ unverified: true, quantization: 'Q4_K_M' })
        expect(groups[0].variants[0].locations).toHaveLength(2)
    })

    it.each([
        { label: 'repository', identity: { ...declared, repository: 'other/Tiel' } },
        { label: 'quantization', identity: { ...declared, quantization: 'Q8_0' } },
        { label: 'total bytes', identity: { ...declared, totalBytes: 101 } }
    ])('keeps declared files with different $label in distinct variants', ({ identity }) => {
        const groups = groupClusterModels([
            inventory('msi', 'tiel-msi', declared),
            inventory('joes', 'tiel-joes', identity)
        ])
        const variants = groups.flatMap(group => group.variants)
        expect(variants).toHaveLength(2)
        expect(variants.map(variant => variant.locations.length)).toEqual([1, 1])
    })

    it('does not merge different verified contents with matching declared metadata', () => {
        const groups = groupClusterModels([
            inventory('msi', 'tiel-msi', {
                ...declared,
                strength: 'verified',
                contentId: 'a'.repeat(64)
            }),
            inventory('joes', 'tiel-joes', {
                ...declared,
                strength: 'verified',
                contentId: 'b'.repeat(64)
            })
        ])
        expect(groups).toHaveLength(1)
        expect(groups[0].variants).toHaveLength(2)
    })
})
