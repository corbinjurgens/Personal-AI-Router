// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { groupClusterModels } from '@/ui/utils/group-cluster-models'

export interface ClusterModelHolders {
    /** Names of the nodes holding a repository at a quantization. */
    forQuantization: (repo: string, quantization: string) => string[]
    /** Names of the nodes holding any quantization of a repository. */
    forRepo: (repo: string) => string[]
}

/**
 * Which cluster nodes already hold a Hugging Face repository, by the
 * repository and quantization their inventory reports. Display only: a
 * matching label does not prove the same file.
 */
export function useClusterModelHolders(
    nodeNames: ReadonlyMap<string, string>
): ClusterModelHolders {
    const inventory = useEngineModelsStore(state => state.models)
    return useMemo(() => {
        const byRepo = new Map<string, Map<string, Set<string>>>()
        const groups = groupClusterModels(
            Array.from(inventory.values()).filter(list => nodeNames.has(list.nodeId))
        )
        for (const group of groups) {
            const quants = byRepo.get(group.name.toLowerCase()) ?? new Map<string, Set<string>>()
            for (const variant of group.variants) {
                const names = quants.get(variant.quantization.toUpperCase()) ?? new Set<string>()
                for (const location of variant.locations)
                    names.add(nodeNames.get(location.nodeId) ?? location.nodeId)
                quants.set(variant.quantization.toUpperCase(), names)
            }
            byRepo.set(group.name.toLowerCase(), quants)
        }
        return {
            forQuantization: (repo, quantization) => [
                ...(byRepo.get(repo.toLowerCase())?.get(quantization.toUpperCase()) ?? [])
            ],
            forRepo: repo => [
                ...new Set(
                    [...(byRepo.get(repo.toLowerCase())?.values() ?? [])].flatMap(names => [
                        ...names
                    ])
                )
            ]
        }
    }, [inventory, nodeNames])
}
