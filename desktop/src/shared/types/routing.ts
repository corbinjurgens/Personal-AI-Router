// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RoutingRequestKey } from '@/shared/types/node-policy'

export interface RoutingCancelParams {
    request: RoutingRequestKey
    regenerate: boolean
}

export interface RoutingCancelResult {
    status: 'not_found' | 'conflict_committed' | 'regenerating' | 'cancelled'
}

export type RoutingActiveParams = Record<string, never>

export interface RoutingActiveResult {
    requests: RoutingRequestKey[]
}
