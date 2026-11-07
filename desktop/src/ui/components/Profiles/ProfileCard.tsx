// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react'
import { Badge, Button, Card, Stack, Text } from '@nvidia/foundations-react-core'
import type { AdvertisedProfile, NodeRoles } from '@/shared/types/node-roles'
import { EngineManagerNames } from '@/shared/constants/engines'
import { engineTypeFromManagerName } from '@/shared/utils/engines'
import { ProfileIcon } from '@/ui/components/icons'
import { ModelIdentityTags } from '@/ui/components/ModelIdentity/ModelIdentityTags'
import { ProfileDeploy } from '@/ui/components/ModelSettings/ModelSettingsCopy'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import { engineSettingsKey, useEngineSettingsStore } from '@/ui/stores/engine-settings.store'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'
import { profileSummary } from '@/ui/utils/profile-summary'
import { profileModelMatchesName } from '@/ui/utils/profile-model-name'

export function ProfileCard({
    profile,
    nodeId,
    nodeName,
    policy,
    roles
}: {
    profile: AdvertisedProfile
    nodeId: string
    nodeName: string
    policy: string
    roles: NodeRoles
}) {
    const [deploy, setDeploy] = useState(false)
    const entries = useEngineSettingsStore(state => state.entries)
    const inventory = useEngineModelsStore(state => state.models)
    const advertised = useDiscoveredNodesStore(state =>
        state.nodes
            .find(node => node.id === nodeId)
            ?.policy?.profiles.find(item => item.key === profile.key)
    )
    const manager = Object.values(EngineManagerNames).find(name => name === profile.engine)
    const target = manager ? { nodeId, engine: manager } : undefined
    const settings = target ? entries[engineSettingsKey(target)] : undefined
    const load =
        settings && !settings.unavailable
            ? settings.snapshot?.settings.modelLoads?.find(item => item.profileKey === profile.key)
            : undefined
    const model = load?.modelId ?? profile.model
    const engine = engineTypeFromManagerName(profile.engine)
    const identity =
        inventory
            .get(`${nodeId}:${engine}`)
            ?.models.find(item =>
                profileModelMatchesName({ profileKey: profile.key, modelId: model }, item.name)
            )?.identity ??
        profile.modelIdentity ??
        advertised?.modelIdentity
    const summary = profileSummary(profile, nodeName, roles, load)
    const focus = useOverviewUiStore(state => state.focusNodeSettings)
    return (
        <Card>
            <Stack gap="3">
                <Text kind="body/semibold/md">
                    <ProfileIcon name={profile.icon} /> {profile.displayName || profile.key}
                </Text>
                <Text kind="body/regular/sm">
                    {profile.key} · {profile.engine}
                </Text>
                <ModelIdentityTags identity={identity} />
                <Text kind="body/regular/sm">
                    Context: {summary.context} · GPU: {summary.gpu} · Extra options:{' '}
                    {summary.extraOptions}
                </Text>
                <div className="flex flex-wrap gap-2">
                    {summary.usage.map(usage => (
                        <Badge key={usage} color="gray">
                            {usage}
                        </Badge>
                    ))}
                </div>
                <div className="flex gap-2">
                    <Button
                        kind="secondary"
                        size="small"
                        onClick={() => focus(nodeId, profile.key)}
                    >
                        Edit
                    </Button>
                    <Button
                        kind="secondary"
                        size="small"
                        onClick={() => setDeploy(value => !value)}
                    >
                        Deploy to node
                    </Button>
                </div>
                {deploy && (
                    <ProfileDeploy
                        identity={{ engine: profile.engine, model, profileKey: profile.key }}
                        policy={policy}
                        targetId={nodeId}
                        onConfigureTarget={target =>
                            focus(target.targetId, target.identity.profileKey)
                        }
                    />
                )}
            </Stack>
        </Card>
    )
}
