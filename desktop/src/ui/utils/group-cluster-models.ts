// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EngineModels, EngineType, ModelItem } from '@/shared/types/engines'

export interface ClusterModelLocation {
    nodeId: string
    engineType: EngineType
    model: ModelItem
}

export interface ClusterModelVariant {
    key: string
    quantization: string
    unverified: boolean
    locations: ClusterModelLocation[]
}

export interface ClusterModelGroup {
    key: string
    name: string
    variants: ClusterModelVariant[]
}

/** Display grouping only: membership never establishes copy equivalence. */
export function groupClusterModels(inventory: Iterable<EngineModels>): ClusterModelGroup[] {
    const variants = new Map<string, ClusterModelVariant>()
    for (const list of inventory) {
        for (const model of list.models) {
            if (!model.downloaded) continue
            const identity = model.identity
            const key = identity?.contentId
                ? JSON.stringify(['content', identity.contentId])
                : identity?.repository && identity.quantization && identity.totalBytes !== undefined
                  ? JSON.stringify([
                        'declared',
                        identity.repository,
                        identity.quantization,
                        identity.totalBytes
                    ])
                  : JSON.stringify(['served', list.engineType, model.name])
            let variant = variants.get(key)
            if (!variant) {
                variant = {
                    key,
                    quantization: identity?.quantization ?? 'Unknown quantization',
                    unverified: false,
                    locations: []
                }
                variants.set(key, variant)
            }
            variant.locations.push({ nodeId: list.nodeId, engineType: list.engineType, model })
        }
    }
    const groups = new Map<string, ClusterModelGroup>()
    for (const variant of variants.values()) {
        variant.unverified =
            new Set(variant.locations.map(location => location.nodeId)).size > 1 &&
            variant.locations.some(
                location =>
                    location.model.identity?.strength !== 'verified' ||
                    !location.model.identity.contentId
            )
        const model = variant.locations[0].model
        const name = model.identity?.repository ?? model.identity?.generalName ?? model.name
        let group = groups.get(name)
        if (!group) {
            group = { key: name, name, variants: [] }
            groups.set(name, group)
        }
        group.variants.push(variant)
    }
    return Array.from(groups.values()).sort((a, b) => a.name.localeCompare(b.name))
}
