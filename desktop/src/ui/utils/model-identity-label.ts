// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelIdentity } from '@/shared/types/engines'
import { formatModelDisplayName } from './format-model-display-name'

interface ModelIdentityTag {
    kind: 'repository' | 'quantization' | 'name' | 'context'
    label: string
    title: string
}

/** Metadata comes only from the identity owner, never from the served ID. */
export function modelIdentityTags(identity?: ModelIdentity): ModelIdentityTag[] {
    const tags: ModelIdentityTag[] = [
        {
            kind: 'repository',
            label: identity?.repository || 'Unknown',
            title: identity?.repository
                ? 'Repository from the Hugging Face cache'
                : 'Repository unknown: no source establishes it'
        },
        {
            kind: 'quantization',
            label: identity?.quantization || 'Unknown',
            title: identity?.quantization
                ? `Quantization source: ${identity.quantizationSource || 'Unknown'}`
                : 'Quantization unknown: no source establishes it'
        }
    ]
    if (!identity?.repository && identity?.generalName) {
        tags.push({
            kind: 'name',
            label: identity.generalName,
            title: 'Name from GGUF general.name'
        })
    }
    if (identity?.configuredContextTokens !== undefined) {
        tags.push({
            kind: 'context',
            label: `Context ${identity.configuredContextTokens}`,
            title: "Configured context from PAIR's reading of the configuration"
        })
    }
    return tags
}

export function modelPrimaryLabel(name: string, identity?: ModelIdentity): string {
    if (identity?.repository) {
        return identity.quantization
            ? formatModelDisplayName(`${identity.repository}:${identity.quantization}`)
            : identity.repository
    }
    return formatModelDisplayName(name)
}
