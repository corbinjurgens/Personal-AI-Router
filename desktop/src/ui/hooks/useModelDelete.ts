// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRef, useState } from 'react'
import type { EngineType } from '@/shared/types/engines'
import { engineManagerName } from '@/shared/utils/engines'
import { EngineManagerNames } from '@/shared/constants/engines'
import { profileModelMatchesName } from '@/ui/utils/profile-model-name'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useEngineSettingsStore, engineSettingsKey } from '@/ui/stores/engine-settings.store'
import {
    policyRoleProfiles,
    policyRoles,
    useNodeConfigurationStore
} from '@/ui/stores/node-configuration.store'

export function useModelDelete() {
    const generation = useRef(0)
    const [selection, setSelection] = useState<{
        engineType: EngineType
        nodeId: string
        model: string
    } | null>(null)
    const [message, setMessage] = useState('')
    const [loading, setLoading] = useState(false)

    const request = async (engineType: EngineType, nodeId: string, model: string) => {
        const current = ++generation.current
        setSelection({ engineType, nodeId, model })
        setLoading(true)
        const node = useNodesStore.getState().nodes.get(nodeId)?.name ?? nodeId
        const engine = engineManagerName(engineType)
        const manager = Object.values(EngineManagerNames).find(name => name === engine)
        const target = manager ? { nodeId, engine: manager } : undefined
        const remote = nodeId === useConnectionStore.getState().selfId ? undefined : nodeId
        const [saved] = await Promise.allSettled([
            useNodeConfigurationStore.getState().readSavedPolicy(nodeId, remote),
            target ? useEngineSettingsStore.getState().reload(target) : Promise.resolve()
        ])
        if (current !== generation.current) return
        const policy = saved.status === 'fulfilled' ? saved.value.policy : undefined
        const settings = target
            ? useEngineSettingsStore.getState().entries[engineSettingsKey(target)]
            : undefined
        const loads = settings?.snapshot?.settings.modelLoads ?? []
        const profileKeys = loads
            .filter(load => profileModelMatchesName(loads, load, model))
            .map(load => load.profileKey)
        const profiles = policyRoleProfiles(policy ?? '{}')
            .filter(profile => profile.engine === engine && profileKeys.includes(profile.key))
            .map(profile => profile.key)
        const roles = policy
            ? policyRoles(policy)
                  .filter(role => role.profiles.some(key => profiles.includes(key)))
                  .map(role => role.name)
            : []
        let text = `Delete ${model} on ${node}?`
        if (profiles.length)
            text += ` Profiles that use it stop serving: ${profiles.join(', ')}. Roles affected: ${roles.join(', ') || 'None'}.`
        if (!policy || !settings?.snapshot || settings.unavailable)
            text +=
                ' Profile usage unavailable; affected profiles and roles could not be fully checked.'
        setMessage(text)
        setLoading(false)
    }

    return {
        open: selection !== null,
        loading,
        message,
        request,
        dismiss: () => {
            generation.current++
            setSelection(null)
        },
        confirm: () => {
            if (!selection || loading) return
            window.pairApi.engines.deleteModel(
                selection.engineType,
                selection.nodeId,
                selection.model
            )
            setSelection(null)
        }
    }
}
