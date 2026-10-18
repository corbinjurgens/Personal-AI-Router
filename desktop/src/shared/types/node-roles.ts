// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** A routing tier a node can declare a role for (`nodepolicy.TierOrder`). */
export type RoleTier = 'weak' | 'medium' | 'strong'

/**
 * The engine and model a node serves one tier with (`nodepolicy.Role`). The
 * engine keeps engine-manager's spelling, as in the policy document.
 */
export interface NodeRole {
    engine: string
    model: string
}

/** A node's declared roles; an absent tier is unset. */
export type NodeRoles = Partial<Record<RoleTier, NodeRole>>

/** Who paused a node (`nodepolicy.PauseOrigin`). */
export type PauseOrigin = 'local' | 'peer'

/**
 * The policy advertisement a node publishes in its inventory document
 * (`nodepolicy.Advertisement`), as the broker's discovery snapshot carries it.
 */
export interface NodePolicyAdvertisement {
    revision: string
    roles: NodeRoles
    pausedBy?: PauseOrigin
}
