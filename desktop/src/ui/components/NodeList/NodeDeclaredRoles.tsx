// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Tag, Text } from '@nvidia/foundations-react-core'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import { roleIdentity, useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { EngineManagerNames } from '@/shared/constants/engines'
import {
    engineSettingsKey,
    useEngineSettingsStore,
    watchEngineSettings
} from '@/ui/stores/engine-settings.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import type { ModelLoad } from '@/shared/types/model-config'
import { ModelIdentityTags } from '@/ui/components/ModelIdentity/ModelIdentityTags'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { ProfileIcon } from '@/ui/components/icons'

function savedLoadSummary(load?: ModelLoad): string {
    if (!load) return 'Saved load unavailable'
    const { requested, resolved } = load
    const context =
        requested.context?.mode === 'fixed' ? requested.context.tokens : resolved?.perSlotContext
    const gpu = requested.gpu
    const placement =
        requested.fit === 'on'
            ? 'Native fit on'
            : gpu?.mode === 'cpu'
              ? 'GPU: CPU'
              : gpu?.mode === 'layers'
                ? `GPU: ${gpu.layers} layers`
                : gpu?.mode === 'native-default'
                  ? 'GPU: native default'
                  : gpu?.mode === 'auto'
                    ? `GPU: Auto${resolved ? (resolved.gpu.mode === 'cpu' ? ' (CPU)' : ` (${resolved.gpu.layers} layers)`) : ''}`
                    : 'GPU: unspecified'
    return [
        context === undefined ? undefined : `Saved context ${context}`,
        placement,
        `${Object.keys(requested.extraOptions ?? {}).length} extra options`
    ]
        .filter(Boolean)
        .join(' · ')
}

/**
 * The roles a node declares, from its policy advertisement in the discovery
 * snapshot, preserving declaration and member order. A node that declares none
 * shows nothing.
 */
export default function NodeDeclaredRoles({ nodeId }: { nodeId: string }) {
    const policy = useDiscoveredNodesStore(
        state => state.nodes.find(node => node.id === nodeId)?.policy
    )
    const showSummaries = useOverviewUiStore(state => state.showProfileSummaries)
    const focusNodeSettings = useOverviewUiStore(state => state.focusNodeSettings)
    const connected = useConnectionStore(state => state.connected)
    const selfId = useConnectionStore(state => state.selfId)
    const permissionContext = useClusterInvitationsStore(state => state.remoteManagementContext)
    const management = useNodeConfigurationStore(state => state.management.get(nodeId))
    const authority = JSON.stringify([
        connected,
        selfId,
        permissionContext,
        management?.state,
        management?.revision
    ])
    const entries = useEngineSettingsStore(state => state.entries)
    const [ready, setReady] = useState('')
    const previousAuthority = useRef(authority)
    const engines = useMemo(
        () =>
            Object.values(EngineManagerNames).filter(engine =>
                policy?.roles.some(role =>
                    role.profiles.some(key =>
                        policy.profiles.some(
                            profile => profile.key === key && profile.engine === engine
                        )
                    )
                )
            ),
        [policy]
    )
    const readIdentity = JSON.stringify([nodeId, engines, authority, showSummaries])
    const canRead = connected && (nodeId === selfId || permissionContext !== null)
    useEffect(() => {
        if (!showSummaries) return
        const stop = watchEngineSettings()
        let active = true
        setReady('')
        const store = useEngineSettingsStore.getState()
        // An authority change invalidates outstanding reads as well as cached facts.
        if (previousAuthority.current !== authority) store.disconnect(nodeId)
        previousAuthority.current = authority
        if (canRead)
            void Promise.all(engines.map(engine => store.reload({ nodeId, engine }))).then(() => {
                if (active) setReady(readIdentity)
            })
        return () => {
            active = false
            stop()
        }
    }, [nodeId, engines, canRead, showSummaries, readIdentity, authority])
    const inventory = useEngineModelsStore(
        useShallow(state =>
            Array.from(state.models.values()).filter(models => models.nodeId === nodeId)
        )
    )
    const declared = useMemo(
        () =>
            (policy?.roles ?? []).flatMap(declaration =>
                (declaration.profiles.length ? declaration.profiles : ['']).map(key => {
                    const role = policy?.profiles.find(profile => profile.key === key)
                    return {
                        tier: declaration.name,
                        key,
                        profile: role,
                        identity:
                            role?.modelIdentity ??
                            (role ? roleIdentity(role, inventory) : undefined)
                    }
                })
            ),
        [policy, inventory]
    )
    if (!policy?.roles.length) return null
    return (
        <Flex align="center" wrap="wrap" gap="2" aria-label="Declared roles">
            <Text kind="body/regular/sm" className="text-subtle-color">
                Roles
            </Text>
            {declared.map(({ tier, key, profile, identity }) => {
                if (!key)
                    return (
                        <Tag
                            key={tier}
                            color="gray"
                            density="compact"
                            readOnly
                            title={`${tier}: Unassigned`}
                            aria-label={`${tier}: Unassigned`}
                            className="break-all"
                        >
                            {tier}: Unassigned
                        </Tag>
                    )
                const name = profile?.displayName || key
                const label = showSummaries ? `${tier}: ${name} (${key})` : `${tier}: ${key}`
                const engine = engines.find(engine => engine === profile?.engine)
                const entry = engine ? entries[engineSettingsKey({ nodeId, engine })] : undefined
                const load =
                    canRead && ready === readIdentity && !entry?.unavailable
                        ? entry?.snapshot?.settings.modelLoads?.find(
                              load => load.profileKey === key
                          )
                        : undefined
                return (
                    <Button
                        key={`${tier}:${key}`}
                        type="button"
                        aria-label={label}
                        title={label}
                        className="min-w-0 max-w-full whitespace-normal break-all"
                        onClick={() => focusNodeSettings(nodeId, key)}
                    >
                        <Text kind="body/regular/sm">{tier}</Text>
                        {showSummaries && (
                            <Text kind="body/regular/sm">
                                <ProfileIcon name={profile?.icon} /> {name}
                                {profile?.contextTokens !== undefined &&
                                    ` · Context ${profile.contextTokens}`}
                            </Text>
                        )}
                        {!profile && <Text kind="body/regular/sm">Unavailable profile</Text>}
                        {showSummaries && profile && (
                            <Text kind="body/regular/sm">{savedLoadSummary(load)}</Text>
                        )}
                        <ModelIdentityTags identity={identity} />
                    </Button>
                )
            })}
        </Flex>
    )
}
