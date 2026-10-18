// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Workload, WorkloadState } from '@/shared/types/workloads'
import { WorkloadStates } from '@/shared/constants/workloads'
import { engineTypeFromManagerName } from '@/shared/utils/engines'
import type { JsonValue } from './json-rpc-client'
import { objectValue, stringValue, numberValue } from './node-projection'

function nullableStringValue(value: JsonValue | undefined): string | null {
    return typeof value === 'string' ? value : null
}

function nullableNumberValue(value: JsonValue | undefined): number | null {
    return typeof value === 'number' ? value : null
}

const WORKLOAD_STATE_SET: ReadonlySet<string> = new Set(WorkloadStates)

function isWorkloadState(value: string): value is WorkloadState {
    return WORKLOAD_STATE_SET.has(value)
}

/** Parse a workload-manager `params.workloadInfo` into a typed Workload. */
function parseWorkload(value: JsonValue | undefined): Workload | null {
    const obj = objectValue(value)
    if (!obj) return null
    const id = stringValue(obj.id)
    if (!id) return null
    // The workload-manager stamps the engine-manager id (`lmstudio`); map it onto
    // our closed `EngineType` union (`lm-studio`) before narrowing so LM Studio
    // jobs are not silently dropped.
    const engine = engineTypeFromManagerName(stringValue(obj.engine))
    if (!engine) return null
    const stateValue = stringValue(obj.state)
    if (!isWorkloadState(stateValue)) return null

    const workload: Workload = {
        id,
        model: stringValue(obj.model),
        engine,
        state: stateValue,
        originatedFrom: nullableStringValue(obj.originatedFrom),
        createdAt: numberValue(obj.createdAt),
        startedAt: nullableNumberValue(obj.startedAt),
        completedAt: nullableNumberValue(obj.completedAt),
        error: nullableStringValue(obj.error),
        requesterId: nullableStringValue(obj.requesterId)
    }
    const scheduledOn = nullableStringValue(obj.scheduledOn)
    if (scheduledOn) workload.scheduledOn = scheduledOn
    const runId = stringValue(obj.runId)
    if (runId) workload.runId = runId
    const requestedModel = stringValue(obj.requestedModel)
    if (requestedModel) workload.requestedModel = requestedModel
    return workload
}

/**
 * Parse a broker `workloads:get-initial` baseline (`{ workloads: workloadInfo[] }`,
 * ordered by createdAt). Each element is a bare `workloadInfo` — the same shape
 * carried inside a `workloads:upsert` `params.workloadInfo` — so it parses with
 * {@link parseWorkload} directly. Malformed entries are skipped.
 */
export function parseWorkloadsInitial(value: JsonValue | undefined): Workload[] {
    const obj = objectValue(value)
    const entries = Array.isArray(obj?.workloads) ? obj.workloads : []
    const workloads: Workload[] = []
    for (const entry of entries) {
        const workload = parseWorkload(entry)
        if (workload) workloads.push(workload)
    }
    return workloads
}

/** A workload whose life has ended; the broker never revives one. */
function isTerminalWorkload(workload: Workload): boolean {
    return (
        workload.state === 'completed' ||
        workload.state === 'failed' ||
        workload.state === 'cancelled'
    )
}

function sameWorkload(left: Workload, right: Workload): boolean {
    return (
        left.id === right.id &&
        left.model === right.model &&
        left.engine === right.engine &&
        left.state === right.state &&
        left.originatedFrom === right.originatedFrom &&
        (left.scheduledOn ?? null) === (right.scheduledOn ?? null) &&
        left.createdAt === right.createdAt &&
        left.startedAt === right.startedAt &&
        left.completedAt === right.completedAt &&
        left.error === right.error &&
        left.requesterId === right.requesterId &&
        left.runId === right.runId &&
        left.requestedModel === right.requestedModel
    )
}

export { nullableStringValue, parseWorkload, isTerminalWorkload, sameWorkload }
