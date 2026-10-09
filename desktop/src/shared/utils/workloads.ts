// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { WorkloadCancelRequest } from '@/shared/types/node-policy'
import type { Workload } from '@/shared/types/workloads'

/**
 * Stable catalog key for a workload.
 *
 * The backend's catalog is keyed by `(originatedFrom, engine, runId, id)`, but
 * the `workloads:remove` push carries only `(workloadId, originatedFrom)` — and
 * the broker's own `Store.Remove` drops every record matching that pair — so
 * `(originatedFrom, id)` is the only key a subscribe client can maintain
 * consistently across upsert and remove. Each node's proxy assigns workload ids
 * from its own monotonic counter, so ids collide across nodes; `originatedFrom`
 * (the origin node) disambiguates. Mirror that here so a remote node's job never
 * overwrites a local one that happens to share an id. The `\u0000` separator
 * cannot appear in a host id or proxy counter, so the key is unambiguous.
 */
export function workloadKey(originatedFrom: string | null, id: string): string {
    return `${originatedFrom ?? ''}\u0000${id}`
}

/**
 * Which node a workload actually ran on, for UI attribution.
 *
 * `scheduledOn` is the node the scheduler routed the request to (where it
 * actually runs); `originatedFrom` is only where the request entered the
 * cluster. UI that means "which node is this job on" (per-node job badges, the
 * workload→node connection lines, the "ran on" label) attributes strictly to
 * `scheduledOn`. A `null` result means the workload has **not been scheduled
 * yet** (still queued) — it has no run-node, so it draws no connection line, is
 * counted on no node, and shows no "ran on" target. Deliberately does NOT fall
 * back to `originatedFrom`: the origin is where the request came from, not where
 * the job ran.
 */
export function workloadExecutionNodeId(workload: Pick<Workload, 'scheduledOn'>): string | null {
    return workload.scheduledOn ?? null
}

/**
 * The `workloads:cancel` request for a job, or null when it cannot be
 * cancelled: only queued and running jobs can be, and the broker needs the node
 * the job originated on to route the cancel.
 */
export function workloadCancelRequest(
    workload: Pick<Workload, 'id' | 'engine' | 'state' | 'originatedFrom' | 'runId'>,
    regenerate: boolean
): WorkloadCancelRequest | null {
    if (workload.state !== 'queued' && workload.state !== 'running') return null
    if (!workload.originatedFrom) return null
    const request: WorkloadCancelRequest = {
        originatedFrom: workload.originatedFrom,
        workloadId: workload.id,
        engine: workload.engine,
        regenerate
    }
    if (workload.runId) request.runId = workload.runId
    return request
}
