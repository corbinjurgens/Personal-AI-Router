// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Tag } from '@nvidia/foundations-react-core'
import type { ModelIdentity } from '@/shared/types/engines'
import { modelIdentityTags } from '@/ui/utils/model-identity-label'

export function ModelIdentityTags({ identity }: { identity?: ModelIdentity }) {
    return (
        <Flex align="center" wrap="wrap" gap="2" aria-label="Model identity">
            {modelIdentityTags(identity).map(tag => (
                <Tag key={tag.kind} color="gray" density="compact" readOnly title={tag.title}>
                    {tag.label}
                </Tag>
            ))}
        </Flex>
    )
}
