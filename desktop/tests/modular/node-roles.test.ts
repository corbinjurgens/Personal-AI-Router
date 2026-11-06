// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { EngineModels } from '@/shared/types/engines'
import type { NodeRoles } from '@/shared/types/node-roles'
import {
    parseNodeRoles,
    parsePolicyAdvertisement,
    sameNodeRoles,
    samePolicyAdvertisement
} from '@/shared/utils/node-roles'
import {
    chosenRole,
    inventoryChoices,
    policyRoleProfiles,
    roleItems,
    useNodeConfigurationStore,
    withRoles
} from '@/ui/stores/node-configuration.store'

describe('ordered role declarations', () => {
    it('keeps saved role order and profile preference order', () => {
        expect(
            parseNodeRoles([
                { name: 'strong', profiles: ['large-32k', 'large-8k'] },
                { name: 'weak', profiles: ['small'] }
            ])
        ).toEqual([
            { name: 'strong', profiles: ['large-32k', 'large-8k'] },
            { name: 'weak', profiles: ['small'] }
        ])
    })

    it('keeps explicitly empty declarations and roles emptied by invalid keys', () => {
        expect(
            parseNodeRoles([
                { name: 'medium', profiles: [] },
                { name: 'weak', profiles: ['', null, false, 2] },
                { name: 'strong', profiles: ['first', 'bad\u0085', 'last'] }
            ])
        ).toEqual([
            { name: 'medium', profiles: [] },
            { name: 'weak', profiles: [] },
            { name: 'strong', profiles: ['first', 'last'] }
        ])
    })

    it('drops old object payloads and malformed declarations without a fallback', () => {
        expect(parseNodeRoles({ weak: { engine: 'llamacpp', model: 'native' } })).toEqual([])
        expect(
            parseNodeRoles([
                null,
                { name: 'custom', profiles: ['custom-profile'] },
                { name: 'medium', profiles: 'profile' },
                { name: 'strong', profiles: ['accepted'] }
            ])
        ).toEqual([{ name: 'strong', profiles: ['accepted'] }])
    })

    it('matches the interim aliases exactly without banning every suffixed key', () => {
        expect(
            parseNodeRoles([
                {
                    name: 'weak',
                    profiles: [
                        'weak',
                        'medium',
                        'strong',
                        'weak-api',
                        'medium-api',
                        'strong-api',
                        'weak-local',
                        'medium-local',
                        'strong-local',
                        'custom-api',
                        'custom-local',
                        'Weak',
                        ' weak',
                        'weak:tag',
                        'owner/model-GGUF:Q4_K_M'
                    ]
                }
            ])
        ).toEqual([
            {
                name: 'weak',
                profiles: [
                    'custom-api',
                    'custom-local',
                    'Weak',
                    ' weak',
                    'weak:tag',
                    'owner/model-GGUF:Q4_K_M'
                ]
            }
        ])
    })

    it('uses UTF-8 byte limits and rejects controls and unpaired surrogates', () => {
        expect(
            parseNodeRoles([
                {
                    name: 'medium',
                    profiles: [
                        'é'.repeat(256),
                        'é'.repeat(257),
                        'x'.repeat(512),
                        'x'.repeat(513),
                        'bad\u0000',
                        'bad\u001f',
                        'bad\u007f',
                        'bad\u009f',
                        '\ud800',
                        '日本語'
                    ]
                }
            ])
        ).toEqual([
            {
                name: 'medium',
                profiles: ['é'.repeat(256), 'x'.repeat(512), '日本語']
            }
        ])
    })

    it('treats role order, key order and empty declarations as policy changes', () => {
        const roles: NodeRoles = [
            { name: 'strong', profiles: ['first', 'second'] },
            { name: 'weak', profiles: [] }
        ]
        expect(
            sameNodeRoles(roles, [
                { name: 'strong', profiles: ['first', 'second'] },
                { name: 'weak', profiles: [] }
            ])
        ).toBe(true)
        expect(sameNodeRoles(roles, [...roles].reverse())).toBe(false)
        expect(
            sameNodeRoles(roles, [
                { name: 'strong', profiles: ['second', 'first'] },
                { name: 'weak', profiles: [] }
            ])
        ).toBe(false)
        expect(sameNodeRoles(roles, [{ name: 'strong', profiles: ['first', 'second'] }])).toBe(
            false
        )
    })
})

describe('single-profile controls over keyed lists', () => {
    const profile = {
        key: 'fast-8k',
        displayName: 'Fast',
        engine: 'llamacpp',
        model: 'owner/native:Q4'
    }
    const policy = JSON.stringify({ revision: '19', profiles: [profile], roles: [] })
    const inventory: EngineModels[] = [
        {
            nodeId: 'role-key-test',
            engineType: 'llama-cpp',
            models: [
                {
                    name: 'owner/native:Q4',
                    downloaded: true,
                    size: 0,
                    status: 'idle',
                    parameterSize: '',
                    family: '',
                    sizeVram: null,
                    expiresAt: null,
                    expiry: '10m',
                    capabilities: []
                }
            ]
        }
    ]

    it('offers and selects policy keys resolved through their native model', () => {
        const choices = inventoryChoices(inventory, policyRoleProfiles(policy))
        expect(choices).toEqual([profile])
        expect(chosenRole(choices, 'fast-8k')).toBe('fast-8k')
        expect(chosenRole(choices, 'owner/native:Q4')).toBeUndefined()
        expect(roleItems(choices, undefined, new Map())).toEqual([
            { value: '', children: 'Unset' },
            { value: 'fast-8k', children: 'native (Q4) · llama.cpp' }
        ])
        expect(inventoryChoices(inventory, [])).toEqual([])
    })

    it('saves a newly picked role as a one-element key list', () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt('role-key-save', { policy, revision: '19' })
        store.setRole('role-key-save', 'medium', 'fast-8k')
        const draft = useNodeConfigurationStore.getState().editors.get('role-key-save')?.draft
        if (!draft) throw new Error('role draft missing after adoption')
        expect(JSON.parse(withRoles(policy, draft))).toEqual({
            revision: '19',
            profiles: [profile],
            roles: [{ name: 'medium', profiles: ['fast-8k'] }]
        })
    })

    it('offers a profile named unset without colliding with the empty selection', () => {
        const choices = [{ ...profile, key: 'unset' }]
        expect(chosenRole(choices, 'unset')).toBe('unset')
        expect(roleItems(choices, 'unset', new Map()).map(item => item.value)).toEqual([
            '',
            'unset'
        ])
    })

    it('detaches the adopted role draft from its saved baseline', () => {
        const store = useNodeConfigurationStore.getState()
        store.adopt('role-key-baseline', {
            policy: JSON.stringify({ roles: [{ name: 'weak', profiles: ['fast-8k'] }] }),
            revision: '19'
        })
        const entry = useNodeConfigurationStore.getState().editors.get('role-key-baseline')
        if (!entry) throw new Error('role draft missing after adoption')
        entry.draft[0].profiles.push('draft-only')
        expect(entry.baseline?.roles).toEqual([{ name: 'weak', profiles: ['fast-8k'] }])
    })

    it('detaches role arrays when adopting drafts and sharing bulk choices', () => {
        const store = useNodeConfigurationStore.getState()
        const roles: NodeRoles = [{ name: 'weak', profiles: ['fast-8k'] }]
        store.adopt('role-key-copy', { policy, revision: '19' })
        store.setDraft('role-key-copy', roles)
        store.setBulk({ common: roles })
        roles[0].profiles.push('external-edit')
        roles.push({ name: 'strong', profiles: [] })
        expect(useNodeConfigurationStore.getState().editors.get('role-key-copy')?.draft).toEqual([
            { name: 'weak', profiles: ['fast-8k'] }
        ])
        expect(useNodeConfigurationStore.getState().bulk.common).toEqual([
            { name: 'weak', profiles: ['fast-8k'] }
        ])
    })
})

describe('keyed advertisements', () => {
    it('drops malformed profiles and empties only their declared memberships', () => {
        expect(
            parsePolicyAdvertisement({
                revision: '7',
                roles: [
                    { name: 'medium', profiles: ['bad'] },
                    { name: 'weak', profiles: ['withheld', 'good', 'bad'] }
                ],
                profiles: [
                    { key: 'bad', displayName: 'Bad', engine: 'llamacpp', model: 'bad\nmodel' },
                    { key: 'good', displayName: 'Good', engine: 'llamacpp', model: 'native' }
                ]
            })
        ).toEqual({
            revision: '7',
            roles: [
                { name: 'medium', profiles: [] },
                { name: 'weak', profiles: ['withheld', 'good'] }
            ],
            profiles: [{ key: 'good', displayName: 'Good', engine: 'llamacpp', model: 'native' }]
        })
    })

    it('retains profile bounds and qualification alongside ordered empty roles', () => {
        expect(
            parsePolicyAdvertisement({
                revision: '7',
                roles: [
                    { name: 'strong', profiles: ['long'] },
                    { name: 'medium', profiles: [] }
                ],
                profiles: [
                    {
                        key: 'long',
                        displayName: 'Long context',
                        icon: 'chat-single',
                        engine: 'llamacpp',
                        model: 'owner/native:Q4',
                        contextTokens: 32768,
                        capabilities: ['tools'],
                        maxOutputTokens: 1024,
                        qualification: { id: 'qualification', revision: '3' }
                    }
                ]
            })
        ).toEqual({
            revision: '7',
            roles: [
                { name: 'strong', profiles: ['long'] },
                { name: 'medium', profiles: [] }
            ],
            profiles: [
                {
                    key: 'long',
                    displayName: 'Long context',
                    icon: 'chat-single',
                    engine: 'llamacpp',
                    model: 'owner/native:Q4',
                    contextTokens: 32768,
                    capabilities: ['tools'],
                    maxOutputTokens: 1024,
                    qualification: { id: 'qualification', revision: '3' }
                }
            ]
        })
    })

    it('detects a profile metadata change independently of revision and roles', () => {
        const left = parsePolicyAdvertisement({
            revision: '7',
            roles: [],
            profiles: [
                {
                    key: 'long',
                    displayName: 'Long',
                    engine: 'llamacpp',
                    model: 'native'
                }
            ]
        })
        const right = parsePolicyAdvertisement({
            revision: '7',
            roles: [],
            profiles: [
                {
                    key: 'long',
                    displayName: 'Renamed',
                    engine: 'llamacpp',
                    model: 'native'
                }
            ]
        })
        expect(samePolicyAdvertisement(left, right)).toBe(false)
    })
})
