// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelLoad } from '@/shared/types/model-config'

/** Inventory can name a typed profile's served section or its artifact. */
export function profileModelMatchesName(
    profile: Pick<ModelLoad, 'profileKey' | 'modelId'>,
    name: string
): boolean {
    return name === profile.profileKey || name === profile.modelId
}
