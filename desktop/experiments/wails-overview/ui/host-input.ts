// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PlatformDisplayName } from '@/shared/types/platform'
import type { PreloadServiceTransport, ServiceTransport } from '@/shared/types/service-bridge'
import { serviceTransport } from '@/ui/api/service-transport'

export interface HostInput {
    readonly platform: PlatformDisplayName
    /**
     * Deliver a fresh projection after each broker/connection generation is ready.
     * Null retires the current generation. The renderer owns each delivered
     * transport and destroys it when it retires.
     */
    subscribe(listener: (transport: ServiceTransport | null) => void): () => void
}

export interface LiveOverviewHost {
    readonly platform: PlatformDisplayName
    onGeneration(listener: (transport: PreloadServiceTransport | null) => void): () => void
}

declare global {
    interface Window {
        __PAIR_OVERVIEW_HOST__?: LiveOverviewHost
    }
}

export function liveOverviewInput(host: LiveOverviewHost): HostInput {
    return {
        platform: host.platform,
        subscribe: listener =>
            host.onGeneration(transport => listener(transport && serviceTransport(transport)))
    }
}
