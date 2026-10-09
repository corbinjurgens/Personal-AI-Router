// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkloadStates } from '@/shared/constants/workloads'
import { EngineType } from '@/shared/types/engines'

export type WorkloadState = (typeof WorkloadStates)[number]

export interface Workload {
    id: string
    model: string
    engine: EngineType
    state: WorkloadState
    /**
     * Owner/origin node of the workload — the node whose proxy received the
     * request. This is the identity half of the backend's `(originatedFrom, id)`
     * global catalog key (workload ids are a per-node proxy counter, so they
     * collide across nodes; `originatedFrom` disambiguates).
     */
    originatedFrom: string | null
    /**
     * Node the workload was routed to / scheduled on — where it actually ran,
     * as opposed to `originatedFrom` (where it came from). Optional/additive:
     * absent until the scheduler picks a target (e.g. a still-queued workload).
     * Use `workloadExecutionNodeId` for "which node is this job on" UI.
     */
    scheduledOn?: string | null
    createdAt: number
    startedAt: number | null
    completedAt: number | null
    error: string | null
    requesterId: string | null
    /**
     * The proxy run the workload belongs to. Workload ids are per-engine
     * counters that restart with the proxy, so cancelling needs this to pick
     * the job when two share an id. Absent on records that predate it.
     */
    runId?: string
    /**
     * What the caller asked for (for example the tier `weak`) when the router
     * resolved it to the concrete `model`. Absent when the request named the
     * model itself.
     */
    requestedModel?: string
}
