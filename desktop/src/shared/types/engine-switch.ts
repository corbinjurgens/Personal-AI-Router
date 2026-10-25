// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface SwitchRequest {
    nodeId?: string
    requestId: string
    sourceEngine: string
    targetEngine: string
    expectedRevision: string
}
export interface SwitchStatusRequest {
    nodeId?: string
    requestId?: string
    afterSequence?: number
}
export interface SwitchFenceRequest {
    nodeId?: string
    requestId: string
}
export type SwitchFenceResult = { nodeId: string; requestId: string } & (
    | { fenced: true; receipt?: never }
    | { fenced: false; receipt: SwitchReceipt }
)
export interface SwitchEngineState {
    intent: 'on' | 'off' | 'unknown'
    running: boolean | null
    healthy: boolean | null
    observedAt: string | null
    terminationVerified: boolean
}
export type SwitchPhase =
    | 'accepted'
    | 'validating'
    | 'draining'
    | 'source-stopping'
    | 'source-off'
    | 'target-starting'
    | 'target-ready'
    | 'completed'
    | 'failed-validation'
    | 'failed-drain'
    | 'failed-source-stop'
    | 'failed-target-start'
    | 'failed-publication'
    | 'failed-interrupted'
    | 'rejected'
export type SwitchErrorKind =
    | 'invalid_request'
    | 'reused_id'
    | 'configuration_conflict'
    | 'fenced'
    | 'remote_veto'
    | 'engine_locked'
    | 'unknown_receipt'
    | 'busy'
    | 'full'
    | 'storage'
    | 'unprepared'
    | 'unroutable'
    | 'target_running'
    | 'validation'
    | 'drain'
    | 'source_stop'
    | 'target_start'
    | 'publication'
    | 'interrupted'
export interface SwitchError {
    code: number
    kind: SwitchErrorKind
    message: string
}
export interface SwitchReceipt {
    version: number
    nodeId: string
    requestId: string
    caller: string
    payloadHash: string
    request: SwitchRequest
    expectedRevision: string
    acceptedRevision: string
    sequence: number
    phase: SwitchPhase
    terminal: boolean
    acceptedAt: string
    updatedAt: string
    deadline: string
    finishedAt: string | null
    source: SwitchEngineState
    target: SwitchEngineState
    error: SwitchError | null
    recoveryRequired: boolean
}
export interface SwitchStatus {
    configurationRevision: string
    activeRequestId?: string
    receipt?: SwitchReceipt
    observed?: { source: SwitchEngineState; target: SwitchEngineState }
}
export interface SwitchChanged {
    nodeId: string
    requestId: string
    sequence: number
}
