// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import getErrorString from '@/shared/utils/get-error-string'
import { Stack, Text } from '@nvidia/foundations-react-core'
import { parseModelSettings } from '@/shared/utils/node-policy-model-settings'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useOverviewNodes } from '@/ui/hooks/useOverviewNodes'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import {
    policyRoleProfiles,
    policyRoles,
    useNodeConfigurationStore,
    watchNodeManagement
} from '@/ui/stores/node-configuration.store'
import { useEngineSettingsStore, watchEngineSettings } from '@/ui/stores/engine-settings.store'
import { ProfileCard } from './ProfileCard'

function NodeProfiles({
    nodeId,
    nodeName,
    local
}: {
    nodeId: string
    nodeName: string
    local: boolean
}) {
    const connected = useConnectionStore(state => state.connected)
    const entry = useNodeConfigurationStore(state => state.editors.get(nodeId))
    const management = useNodeConfigurationStore(state => state.management.get(nodeId))
    const [reading, setReading] = useState(true)
    const [readError, setReadError] = useState<string | null>(null)
    useEffect(() => {
        let active = true
        setReading(true)
        setReadError(null)
        const stop = watchNodeManagement(nodeId, local ? undefined : nodeId)
        const stopPolicy = window.pairApi.nodePolicy.onPolicyChanged(change => {
            if (change.nodeId === nodeId) useNodeConfigurationStore.getState().adopt(nodeId, change)
        })
        if (connected)
            void useNodeConfigurationStore
                .getState()
                .readSavedPolicy(nodeId, local ? undefined : nodeId)
                .catch(error => {
                    if (active) setReadError(getErrorString(error))
                })
                .finally(() => {
                    if (active) setReading(false)
                })
        return () => {
            active = false
            stopPolicy()
            stop()
        }
    }, [nodeId, local, connected])
    const saved =
        entry?.latest &&
        (!entry.baseline || BigInt(entry.latest.revision) > BigInt(entry.baseline.revision))
            ? entry.latest
            : entry?.baseline
    const policy = saved?.policy
    const parsed = policy === undefined ? undefined : parseModelSettings(policy)
    const profiles = policyRoleProfiles(policy ?? '{}')
    const engines = [...new Set(profiles.map(profile => profile.engine))].join(',')
    useEffect(() => {
        if (!connected || !engines) return
        const stop = watchEngineSettings()
        for (const engine of engines.split(',')) {
            if (engine === 'llamacpp')
                void useEngineSettingsStore.getState().reload({ nodeId, engine })
        }
        return stop
    }, [connected, nodeId, engines])
    const unavailable = !connected
        ? 'Disconnected'
        : management && !['local', 'allowed', 'loading'].includes(management.state)
          ? (management.message ?? 'Management permission required')
          : readError
            ? readError
            : entry?.phase === 'unreachable'
              ? (entry.message ?? 'Policy unavailable')
              : parsed && !parsed.ok
                ? parsed.error.message
                : undefined
    const loading =
        !unavailable &&
        (reading ||
            !management ||
            management.state === 'loading' ||
            !policy ||
            entry?.phase === 'reading' ||
            entry?.phase === 'sending')
    return (
        <Stack gap="3">
            <Text kind="body/semibold/lg">{nodeName}</Text>
            {unavailable ? (
                <Text kind="body/regular/sm">Profiles unavailable: {unavailable}</Text>
            ) : loading ? (
                <Text kind="body/regular/sm" role="status">
                    Loading profiles for {nodeName}…
                </Text>
            ) : profiles.length === 0 ? (
                <Text kind="body/regular/sm">No profiles configured.</Text>
            ) : (
                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                    {profiles.map(profile => (
                        <ProfileCard
                            key={profile.key}
                            profile={profile}
                            nodeId={nodeId}
                            nodeName={nodeName}
                            policy={policy ?? '{}'}
                            roles={policyRoles(policy ?? '{}')}
                        />
                    ))}
                </div>
            )}
        </Stack>
    )
}

export default function ProfilesView() {
    const nodes = useOverviewNodes()
    const selfId = useConnectionStore(state => state.selfId)
    const connected = useConnectionStore(state => state.connected)
    useEffect(() => {
        if (connected) void useDiscoveredNodesStore.getState().refresh()
    }, [connected])
    const targets = nodes.map(node => ({ id: node.id, name: node.name }))
    return (
        <Stack gap="6" className="p-6">
            <Text kind="body/semibold/lg">Profiles</Text>
            {targets.map(node => (
                <NodeProfiles
                    key={node.id}
                    nodeId={node.id}
                    nodeName={node.name}
                    local={node.id === selfId}
                />
            ))}
            {targets.length === 0 && <Text kind="body/regular/sm">Waiting for nodes.</Text>}
        </Stack>
    )
}
