// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelIdentity } from '@/shared/types/engines'
import type { NodePauseMode, QualificationRef } from '@/shared/types/node-policy'

/** A user-named role and its profile keys, in saved preference order. */
export interface NodeRole {
    name: string
    profiles: string[]
}

/** A node's declarations in saved order, including explicitly empty roles. */
export type NodeRoles = NodeRole[]

/** Keyed profile metadata published by `nodepolicy.Advertisement`. */
export interface AdvertisedProfile {
    key: string
    displayName: string
    icon?: string
    engine: string
    model: string
    modelIdentity?: ModelIdentity
    contextTokens?: number
    capabilities?: string[]
    maxOutputTokens?: number
    qualification?: QualificationRef
}

/** Who paused a node (`nodepolicy.PauseOrigin`). */
export type PauseOrigin = 'local' | 'peer'

/**
 * The policy advertisement a node publishes in its inventory document
 * (`nodepolicy.Advertisement`), as the broker's discovery snapshot carries it.
 */
export interface NodePolicyAdvertisement {
    revision: string
    roles: NodeRoles
    profiles: AdvertisedProfile[]
    pausedBy?: PauseOrigin
    pauseMode?: NodePauseMode
}
