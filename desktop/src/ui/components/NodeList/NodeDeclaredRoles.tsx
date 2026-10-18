// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Tag, Text } from '@nvidia/foundations-react-core'
import { ROLE_TIERS } from '@/shared/utils/node-roles'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { roleLabel } from '@/ui/stores/node-configuration.store'

/**
 * The roles a node declares, from its policy advertisement in the discovery
 * snapshot. A node that declares none shows nothing: routers use their own
 * tier entries for it.
 */
export default function NodeDeclaredRoles({ nodeId }: { nodeId: string }) {
    const roles = useDiscoveredNodesStore(
        state => state.nodes.find(node => node.id === nodeId)?.policy?.roles
    )
    const descriptors = useEngineStatusStore(state => state.descriptors)
    if (!roles) return null
    const declared = ROLE_TIERS.flatMap(tier => {
        const role = roles[tier]
        return role ? [{ tier, label: roleLabel(role, descriptors) }] : []
    })
    if (declared.length === 0) return null
    return (
        <Flex align="center" wrap="wrap" gap="2" aria-label="Declared roles">
            <Text kind="body/regular/sm" className="text-subtle-color">
                Roles
            </Text>
            {declared.map(({ tier, label }) => (
                <Tag key={tier} color="gray" density="compact" readOnly>
                    {`${tier}: ${label}`}
                </Tag>
            ))}
        </Flex>
    )
}
