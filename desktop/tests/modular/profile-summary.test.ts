// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { AdvertisedProfile, NodeRoles } from '@/shared/types/node-roles'
import { profileSummary } from '@/ui/utils/profile-summary'

const profile: AdvertisedProfile = {
    key: 'shared-key',
    displayName: 'Coding',
    engine: 'llamacpp',
    model: 'artifact'
}

describe('profile card summary', () => {
    it('keeps matching own-node roles in saved order and excludes unrelated roles', () => {
        const roles: NodeRoles = [
            { name: 'writer', profiles: ['shared-key'] },
            { name: 'unused', profiles: ['other'] },
            { name: 'coder', profiles: ['other', 'shared-key'] }
        ]
        expect(profileSummary(profile, 'MSI', roles).usage).toEqual([
            'writer on MSI',
            'coder on MSI'
        ])
        expect(profileSummary(profile, 'Joes', [{ name: 'chat', profiles: [] }]).usage).toEqual([])
    })

    it('reports native fit from the saved requested load', () => {
        expect(
            profileSummary(profile, 'MSI', [], {
                profileKey: profile.key,
                modelId: profile.model,
                requested: { fit: 'on' }
            }).gpu
        ).toBe('Native fit')
    })

    it('reports explicit CPU placement from the saved load', () => {
        expect(
            profileSummary(profile, 'MSI', [], {
                profileKey: profile.key,
                modelId: profile.model,
                requested: { fit: 'off', gpu: { mode: 'cpu' } }
            }).gpu
        ).toBe('CPU')
    })

    it('shows Unknown for summary fields absent from policy and inspection', () => {
        expect(profileSummary(profile, 'MSI', [])).toEqual({
            context: 'Unknown',
            gpu: 'Unknown',
            extraOptions: 'Unknown',
            usage: []
        })
    })

    it('uses saved context and GPU layers and counts extra options', () => {
        expect(
            profileSummary(profile, 'MSI', [], {
                profileKey: profile.key,
                modelId: profile.model,
                requested: {
                    context: { mode: 'fixed', tokens: 8192 },
                    gpu: { mode: 'layers', layers: 20 },
                    extraOptions: { first: 'one', second: 'two' }
                }
            })
        ).toMatchObject({ context: '8192', gpu: '20 GPU layers', extraOptions: '2' })
    })
})
