// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react'
import { Badge, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core'
import type { AdvertisedProfile, NodeRoles } from '@/shared/types/node-roles'
import { EngineManagerNames } from '@/shared/constants/engines'
import { engineLabel, engineTypeFromManagerName } from '@/shared/utils/engines'
import { ProfileIcon } from '@/ui/components/icons'
import { ModelIdentityTags } from '@/ui/components/ModelIdentity/ModelIdentityTags'
import { ProfileDeploy } from '@/ui/components/ModelSettings/ModelSettingsCopy'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import { engineSettingsKey, useEngineSettingsStore } from '@/ui/stores/engine-settings.store'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'
import { profileSummary } from '@/ui/utils/profile-summary'
import { profileModelMatchesName } from '@/ui/utils/profile-model-name'

/** Unknown values are left out: the card shows only what PAIR knows. */
function Spec({ label, value }: { label: string; value: string }) {
    if (value === 'Unknown') return null
    return (
        <Text kind="body/regular/sm" className="rounded-md bg-[#ffffff08] px-2.5 py-1">
            <span className="text-subtle-color">{label}: </span>
            {value}
        </Text>
    )
}

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
    const loads =
        settings && !settings.unavailable ? (settings.snapshot?.settings.modelLoads ?? []) : []
    const load = loads.find(item => item.profileKey === profile.key)
    const model = load?.modelId ?? profile.model
    const engine = engineTypeFromManagerName(profile.engine)
    const identity =
        inventory
            .get(`${nodeId}:${engine}`)
            ?.models.find(item =>
                profileModelMatchesName(
                    loads,
                    { profileKey: profile.key, modelId: model },
                    item.name
                )
            )?.identity ??
        profile.modelIdentity ??
        advertised?.modelIdentity
    const summary = profileSummary(profile, nodeName, roles, load, identity)
    const focus = useOverviewUiStore(state => state.focusNodeSettings)
    const title = profile.displayName || profile.key
    const ids = [...new Set([profile.key, model])].filter(id => id !== title)
    return (
        <article className="pair-paper p-4 flex flex-col gap-3 min-w-0" aria-label={title}>
            <Flex align="start" justify="between" gap="3">
                <Stack gap="1" className="min-w-0">
                    <Text kind="body/semibold/md" className="break-words">
                        <ProfileIcon name={profile.icon} /> {title}
                    </Text>
                    {ids.length > 0 && (
                        <Text
                            kind="body/regular/sm"
                            className="text-subtle-color break-all"
                            title="Profile key and served model ID"
                        >
                            {ids.join(' · ')}
                        </Text>
                    )}
                </Stack>
                <Text kind="body/regular/xs" className="text-subtle2 shrink-0 mt-1">
                    {engine ? engineLabel(engine, undefined) : profile.engine}
                </Text>
            </Flex>
            <ModelIdentityTags identity={identity} withContext={false} />
            <Flex gap="2" wrap="wrap">
                <Spec label="Context" value={summary.context} />
                <Spec label="GPU" value={summary.gpu} />
                <Spec label="Extra options" value={summary.extraOptions} />
            </Flex>
            {summary.usage.length > 0 ? (
                <Flex gap="2" wrap="wrap" aria-label="Roles using this profile">
                    {summary.usage.map(usage => (
                        <Badge key={usage} color="green">
                            {usage}
                        </Badge>
                    ))}
                </Flex>
            ) : (
                <Text kind="body/regular/sm" className="text-subtle2">
                    Not used by a role on {nodeName}
                </Text>
            )}
            <div className="flex gap-2 mt-auto pt-1">
                <Button kind="secondary" size="small" onClick={() => focus(nodeId, profile.key)}>
                    Edit
                </Button>
                <Button kind="secondary" size="small" onClick={() => setDeploy(value => !value)}>
                    {deploy ? 'Hide deploy' : 'Deploy to node'}
                </Button>
            </div>
            {deploy && (
                <ProfileDeploy
                    identity={{ engine: profile.engine, model, profileKey: profile.key }}
                    policy={policy}
                    targetId={nodeId}
                    onConfigureTarget={target => focus(target.targetId, target.identity.profileKey)}
                />
            )}
        </article>
    )
}
