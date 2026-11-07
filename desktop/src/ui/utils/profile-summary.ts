// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelLoad } from '@/shared/types/model-config'
import type { AdvertisedProfile, NodeRoles } from '@/shared/types/node-roles'

/** Call with the profile's own node roles, never a cluster-wide role list. */
export function profileSummary(
    profile: AdvertisedProfile,
    nodeName: string,
    roles: NodeRoles,
    load?: ModelLoad
) {
    const requested = load?.requested
    const context =
        load?.resolved?.perSlotContext ??
        (requested?.context?.mode === 'fixed' ? requested.context.tokens : undefined) ??
        profile.contextTokens
    const gpu =
        requested?.fit === 'on'
            ? 'Native fit'
            : load?.resolved?.gpu.mode === 'cpu' || requested?.gpu?.mode === 'cpu'
              ? 'CPU'
              : load?.resolved?.gpu.layers !== undefined
                ? `${load.resolved.gpu.layers} GPU layers`
                : requested?.gpu?.mode === 'layers' && requested.gpu.layers !== undefined
                  ? `${requested.gpu.layers} GPU layers`
                  : 'Unknown'
    return {
        context: context === undefined ? 'Unknown' : String(context),
        gpu,
        extraOptions:
            requested === undefined
                ? 'Unknown'
                : String(Object.keys(requested.extraOptions ?? {}).length),
        usage: roles
            .filter(role => role.profiles.includes(profile.key))
            .map(role => `${role.name} on ${nodeName}`)
    }
}
