// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Target-owned operation facts; completion does not establish content readiness. */
export interface ModelDownloadEntry {
    id: string
    engine: string
    model: string
    phase: 'queued' | 'in_progress' | 'failed' | 'complete' | 'unknown'
    blocksReadiness: boolean
    percent?: number
    reason?:
        | 'busy'
        | 'capacity'
        | 'download_failed'
        | 'interrupted'
        | 'progress_timeout'
        | 'acceptance_unknown'
        | 'cleanup_unknown'
        | 'persistence_error'
        | 'unsupported'
    persistence?: 'uncertain'
}

/** Full bounded observation, ordered only within its manager incarnation. */
export interface ModelDownloadSnapshot {
    version: 1
    sourceId: string
    revision: string
    recoveryRequired: boolean
    entries: ModelDownloadEntry[]
}
