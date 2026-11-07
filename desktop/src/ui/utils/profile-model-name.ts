// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelLoad } from '@/shared/types/model-config'

/** A claimed served key wins over artifact matching within this node and engine. */
export function profileModelMatchesName(
    loads: readonly ModelLoad[],
    profile: Pick<ModelLoad, 'profileKey' | 'modelId'>,
    name: string
): boolean {
    const owner = loads.find(load => load.profileKey === name)
    if (owner) return owner.profileKey === profile.profileKey
    return name === profile.profileKey || name === profile.modelId
}
