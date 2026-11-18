// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelIdentity } from '@/shared/types/engines'
import type { ModelLoad } from '@/shared/types/model-config'
import type { AdvertisedProfile, NodeRoles } from '@/shared/types/node-roles'
import { formatContextTokens } from '@/ui/utils/formatters'

/**
 * Call with the profile's own node roles, never a cluster-wide role list.
 * `identity` is the served model's identity; its configured context is the
 * last fallback when the node's engine settings are not readable here.
 * Labels are for the profile card: context reads "64K" and GPU "20 layers".
 */
export function profileSummary(
    profile: AdvertisedProfile,
    nodeName: string,
    roles: NodeRoles,
    load?: ModelLoad,
    identity?: ModelIdentity
) {
    const requested = load?.requested
    const context =
        load?.resolved?.perSlotContext ??
        (requested?.context?.mode === 'fixed' ? requested.context.tokens : undefined) ??
        profile.contextTokens ??
        identity?.configuredContextTokens
    const gpu =
        requested?.fit === 'on'
            ? 'Native fit'
            : load?.resolved?.gpu.mode === 'cpu' || requested?.gpu?.mode === 'cpu'
              ? 'CPU'
              : load?.resolved?.gpu.layers !== undefined
                ? `${load.resolved.gpu.layers} layers`
                : requested?.gpu?.mode === 'layers' && requested.gpu.layers !== undefined
                  ? `${requested.gpu.layers} layers`
                  : 'Unknown'
    return {
        context: context === undefined ? 'Unknown' : formatContextTokens(context),
        contextTokens: context,
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
