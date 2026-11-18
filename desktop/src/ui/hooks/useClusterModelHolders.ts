// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react'
import type { NodeItem } from '@/shared/types/nodes'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { groupClusterModels } from '@/ui/utils/group-cluster-models'

export interface ClusterModelHolders {
    /** The nodes holding a repository at a quantization. */
    forQuantization: (repo: string, quantization: string) => NodeItem[]
    /** The nodes holding any quantization of a repository. */
    forRepo: (repo: string) => NodeItem[]
}

/**
 * Which cluster nodes already hold a Hugging Face repository, by the
 * repository and quantization their inventory reports. Display only: a
 * matching label does not prove the same file.
 */
export function useClusterModelHolders(
    nodesById: ReadonlyMap<string, NodeItem>
): ClusterModelHolders {
    const inventory = useEngineModelsStore(state => state.models)
    return useMemo(() => {
        // Node ids by repository, then by quantization.
        const byRepo = new Map<string, Map<string, Set<string>>>()
        const groups = groupClusterModels(
            Array.from(inventory.values()).filter(list => nodesById.has(list.nodeId))
        )
        for (const group of groups) {
            const quants = byRepo.get(group.name.toLowerCase()) ?? new Map<string, Set<string>>()
            for (const variant of group.variants) {
                const ids = quants.get(variant.quantization.toUpperCase()) ?? new Set<string>()
                for (const location of variant.locations) ids.add(location.nodeId)
                quants.set(variant.quantization.toUpperCase(), ids)
            }
            byRepo.set(group.name.toLowerCase(), quants)
        }
        const nodesOf = (ids: Iterable<string>) =>
            [...new Set(ids)].flatMap(id => nodesById.get(id) ?? [])
        return {
            forQuantization: (repo, quantization) =>
                nodesOf(byRepo.get(repo.toLowerCase())?.get(quantization.toUpperCase()) ?? []),
            forRepo: repo =>
                nodesOf(
                    [...(byRepo.get(repo.toLowerCase())?.values() ?? [])].flatMap(ids => [...ids])
                )
        }
    }, [inventory, nodesById])
}
