// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** JSON-RPC codes the services answer with, as the Go services define them. */

/** Invalid parameters: the caller must change the request before retrying. */
export const INVALID_PARAMS_CODE = -32602
/** A policy save based on a revision other than the persisted one. */
export const POLICY_CONFLICT_CODE = -32010
/** A start, load, resume or settings apply the target node's own pause or Off forbids. */
export const LOCAL_VETO_CODE = -32011
/** The target device has not granted this peer remote management permission. */
export const REMOTE_MANAGEMENT_DENIED_CODE = -32012

/**
 * The explanation for a paired node's action refused with LOCAL_VETO_CODE.
 * `action` names what was refused, for example `start ollama on node-a`.
 */
export function localVetoMessage(action: string): string {
    return `Could not ${action}: that node is paused, or the engine is Off there. Resume or start it on that node itself, then try again.`
}
