// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Tag } from '@nvidia/foundations-react-core'
import type { ModelIdentity } from '@/shared/types/engines'
import { modelIdentityTags } from '@/ui/utils/model-identity-label'

/** `withContext={false}` drops the context tag where the caller shows context itself. */
export function ModelIdentityTags({
    identity,
    withContext = true
}: {
    identity?: ModelIdentity
    withContext?: boolean
}) {
    return (
        <Flex align="center" wrap="wrap" gap="2" aria-label="Model identity" className="min-w-0">
            {modelIdentityTags(identity)
                .filter(tag => withContext || tag.kind !== 'context')
                .map(tag => (
                    <Tag
                        key={tag.kind}
                        color="gray"
                        density="compact"
                        readOnly
                        title={tag.title}
                        className="max-w-full"
                    >
                        <span className="truncate">{tag.label}</span>
                    </Tag>
                ))}
        </Flex>
    )
}
