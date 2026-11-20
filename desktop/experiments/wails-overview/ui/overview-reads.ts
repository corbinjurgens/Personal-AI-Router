// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { WsInvokeChannel } from '@/shared/types/ws-channels'

function channels<const C extends WsInvokeChannel>(...names: C[]) {
    const lookup = new Set<string>(names)
    return { names, size: lookup.size, has: (name: string): name is C => lookup.has(name) }
}

export const overviewBaselines = channels(
    'app:get-initial',
    'cluster:get-initial',
    'nodes:get-initial',
    'engines:get-initial',
    'workloads:get-initial',
    'discovery:get-nodes',
    'errors:get-initial'
)
const overviewReads = channels(
    ...overviewBaselines.names,
    'cluster:get-remote-management',
    'engine:remote-management-get',
    'engines:get-settings',
    'policy:get'
)

export type OverviewRead = (typeof overviewReads.names)[number]
export const isOverviewRead = overviewReads.has
