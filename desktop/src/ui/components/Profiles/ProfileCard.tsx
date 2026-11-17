// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react'
import { Badge, Button, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core'
import type { AdvertisedProfile, NodeRoles } from '@/shared/types/node-roles'
import { EngineManagerNames } from '@/shared/constants/engines'
import { engineLabel, engineTypeFromManagerName } from '@/shared/utils/engines'
import { ProfileIcon } from '@/ui/components/icons'
import { Monogram } from '@/ui/components/Page/PagePrimitives'
import { hubModelFacts } from '@/ui/utils/hub-model-facts'
import { ProfileDeploy } from '@/ui/components/ModelSettings/ModelSettingsCopy'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import { engineSettingsKey, useEngineSettingsStore } from '@/ui/stores/engine-settings.store'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'
import { profileSummary } from '@/ui/utils/profile-summary'
import { profileModelMatchesName } from '@/ui/utils/profile-model-name'

/** Unknown values are left out: the card shows only what PAIR knows. */
function Spec({ label, value, title }: { label: string; value: string; title?: string }) {
    if (value === 'Unknown') return null
    return (
        <div className="pair-tile px-3 py-2 min-w-0">
            <dt className="pair-eyebrow truncate">{label}</dt>
            <dd className="m-0 mt-0.5">
                <Text kind="body/semibold/sm" className="truncate block" title={title ?? value}>
                    {value}
                </Text>
            </dd>
        </div>
    )
}

/** Context sizes as people say them: 65536 -> "64K". */
function formatContext(value: string): string {
    const tokens = Number(value)
    if (value === 'Unknown' || !Number.isFinite(tokens)) return value
    return tokens >= 1024 && tokens % 1024 === 0 ? `${tokens / 1024}K` : tokens.toLocaleString('en')
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
    const served = model !== title ? model : undefined
    return (
        <article className="pair-paper p-4 flex flex-col gap-4 min-w-0" aria-label={title}>
            <Flex align="start" gap="3">
                {profile.icon ? (
                    <span
                        aria-hidden="true"
                        className="shrink-0 inline-flex items-center justify-center rounded-lg pair-tile"
                        style={{
                            width: 40,
                            height: 40,
                            fontSize: 20,
                            color: 'var(--color-brand, #76b900)'
                        }}
                    >
                        <ProfileIcon name={profile.icon} />
                    </span>
                ) : (
                    <Monogram
                        name={
                            identity?.repository ? hubModelFacts(identity.repository).title : title
                        }
                        size={40}
                    />
                )}
                <Stack gap="1" className="min-w-0 grow">
                    <Text kind="body/semibold/md" className="truncate" title={profile.key}>
                        {title}
                    </Text>
                    <Flex
                        align="center"
                        gap="2"
                        wrap="wrap"
                        aria-label="Model identity"
                        className="min-w-0"
                    >
                        {identity?.repository ? (
                            <Text
                                kind="body/regular/sm"
                                className="text-subtle-color truncate min-w-0"
                                title={served ? `Served as ${served}` : undefined}
                            >
                                {identity.repository}
                            </Text>
                        ) : (
                            <Text
                                kind="body/regular/sm"
                                className="text-subtle-color truncate min-w-0"
                                title="Served model ID"
                            >
                                {served ?? profile.key}
                            </Text>
                        )}
                        {identity?.quantization && (
                            <Tag color="green" density="compact" readOnly>
                                {identity.quantization}
                            </Tag>
                        )}
                    </Flex>
                </Stack>
                <Text kind="body/regular/xs" className="text-subtle2 shrink-0 mt-1">
                    {engine ? engineLabel(engine, undefined) : profile.engine}
                </Text>
            </Flex>
            {(summary.context !== 'Unknown' ||
                summary.gpu !== 'Unknown' ||
                summary.extraOptions !== 'Unknown') && (
                <dl className="grid grid-cols-3 gap-2 m-0">
                    <Spec
                        label="Context"
                        value={formatContext(summary.context)}
                        title={`${summary.context} tokens`}
                    />
                    <Spec label="GPU" value={summary.gpu.replace(/ GPU layers$/, ' layers')} />
                    <Spec label="Options" value={summary.extraOptions} title="Extra options" />
                </dl>
            )}
            <Stack gap="1.5">
                <span className="pair-eyebrow">Used by</span>
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
                        No role on {nodeName} uses this profile yet
                    </Text>
                )}
            </Stack>
            <div className="flex gap-2 mt-auto pt-3 pair-divider-top">
                <Button kind="secondary" size="small" onClick={() => focus(nodeId, profile.key)}>
                    Edit
                </Button>
                <Button kind="tertiary" size="small" onClick={() => setDeploy(value => !value)}>
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
