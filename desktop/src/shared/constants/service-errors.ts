// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** JSON-RPC codes the services answer with, as the Go services define them. */

/** Invalid parameters: the caller must change the request before retrying. */
export const INVALID_PARAMS_CODE = -32602
/** A policy save based on a revision other than the persisted one. */
export const POLICY_CONFLICT_CODE = -32010
/**
 * A target node's own refusal: its pause, an older target's saved Off, or a
 * local stop that cancelled the operation. The code alone does not establish
 * which, so it is never presented as a pause without authoritative node state.
 */
export const LOCAL_VETO_CODE = -32011
/** The target device has not granted this peer remote management permission. */
export const REMOTE_MANAGEMENT_DENIED_CODE = -32012
/** A start, load or switch into an engine locked on its own node. */
export const ENGINE_LOCKED_CODE = -32013
/** A peer tried to change inference settings reserved for the target device. */
export const REMOTE_SETTING_LOCKED_CODE = -32014

/**
 * The explanation for a paired node's action refused with LOCAL_VETO_CODE.
 * `action` names what was refused, for example `start ollama on node-a`.
 */
export function localVetoMessage(action: string): string {
    return `Could not ${action}: the target refused the operation. Check its node and engine state, then try again.`
}

/**
 * The explanation for a paired node's action refused with ENGINE_LOCKED_CODE.
 * `action` names what was refused, for example `start ollama on node-a`.
 */
export function engineLockedMessage(action: string): string {
    return `Could not ${action}: the engine is locked on that node. Unlock it on that node, then try again.`
}
