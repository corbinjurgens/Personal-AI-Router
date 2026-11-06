// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { describe, expect, it } from 'vitest'
import type { JsonValue } from '@/shared/types/json'
import { parseModelIdentity } from '@/shared/utils/model-identity'
import { parseModelIdentities } from '@/electron/service-bridge/model-identity'
import { parsePolicyAdvertisement, samePolicyAdvertisement } from '@/shared/utils/node-roles'
import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import NodeDeclaredRoles from '@/ui/components/NodeList/NodeDeclaredRoles'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'

describe('shared profile identity decoding', () => {
    const identity = {
        strength: 'verified',
        contentId: 'a'.repeat(64),
        repository: 'owner/model',
        files: [{ role: 'model', name: 'weights.gguf', sizeBytes: 123 }]
    }
    const profile = { key: 'profile', displayName: 'Profile', engine: 'llamacpp', model: 'native' }
    const advertisement = () => ({
        revision: '1',
        roles: [{ name: 'weak', profiles: ['profile'] }],
        profiles: [{ ...profile, modelIdentity: identity }]
    })

    it('preserves validated identity in profiles and the Electron map', () => {
        const expected = {
            ...identity,
            files: [{ role: 'model', name: 'weights.gguf', sizeBytes: 123 }]
        }
        expect(parseModelIdentity(identity)).toEqual(expected)
        expect(parsePolicyAdvertisement(advertisement())?.profiles[0].modelIdentity).toEqual(
            expected
        )
        expect(parseModelIdentities({ llamacpp: { native: identity } })).toEqual({
            llamacpp: { native: expected }
        })
    })

    it.each<JsonValue>([
        { strength: 'verified' },
        { strength: 'declared', repository: '../bad' },
        { strength: 'declared', generalName: 'bad\u0085' },
        { strength: 'declared', generalName: '\ud800' },
        { strength: 'declared', generalName: 'é'.repeat(129) },
        { strength: 'declared', totalBytes: -1 },
        { strength: 'declared', files: [{ role: 'model', name: '../weights' }] }
    ])(
        'filters malformed profile metadata and preserves unrelated model inventory: %j',
        invalid => {
            expect(parseModelIdentity(invalid)).toBeUndefined()
            expect(
                parsePolicyAdvertisement({
                    roles: [{ name: 'weak', profiles: ['profile'] }],
                    profiles: [{ ...profile, modelIdentity: invalid }]
                })
            ).toEqual({
                revision: '',
                roles: [{ name: 'weak', profiles: [] }],
                profiles: []
            })
            expect(parseModelIdentities({ llamacpp: { bad: invalid, good: identity } })).toEqual({
                llamacpp: { good: parseModelIdentity(identity) }
            })
        }
    )

    it('detaches file metadata and detects identity-only changes', () => {
        const input = advertisement()
        const parsed = parsePolicyAdvertisement(input)
        const equal = parsePolicyAdvertisement(advertisement())
        expect(samePolicyAdvertisement(parsed, equal)).toBe(true)
        const files = parsed?.profiles[0].modelIdentity?.files
        if (!files) throw new Error('Missing identity files')
        files[0].name = 'changed.gguf'
        expect(input.profiles[0].modelIdentity.files[0].name).toBe('weights.gguf')
        expect(equal?.profiles[0].modelIdentity?.files?.[0].name).toBe('weights.gguf')
        expect(samePolicyAdvertisement(parsed, equal)).toBe(false)
    })
})

describe('declared role display', () => {
    it('renders every membership in declaration order, including unavailable profiles', async () => {
        const previous = useDiscoveredNodesStore.getState().nodes
        const container = document.createElement('div')
        document.body.append(container)
        const root = createRoot(container)
        try {
            useDiscoveredNodesStore.setState({
                nodes: [
                    {
                        id: 'display-node',
                        name: 'Display',
                        ipAddress: '127.0.0.1',
                        port: 1,
                        lastSeen: 0,
                        trusted: true,
                        clustered: true,
                        policy: {
                            revision: '1',
                            roles: [
                                { name: 'strong', profiles: ['missing', 'available'] },
                                { name: 'weak', profiles: [] },
                                { name: 'medium', profiles: ['available'] }
                            ],
                            profiles: [
                                {
                                    key: 'available',
                                    displayName: 'Available',
                                    engine: 'llamacpp',
                                    model: 'native-model'
                                }
                            ]
                        }
                    }
                ]
            })
            await act(async () =>
                root.render(createElement(NodeDeclaredRoles, { nodeId: 'display-node' }))
            )
            const html = container.innerHTML
            const memberships = Array.from(container.querySelectorAll('button'))
            expect(memberships.map(item => item.getAttribute('aria-label'))).toEqual([
                'strong: missing',
                'strong: available',
                'medium: available'
            ])
            expect(memberships[0]?.textContent).toContain('Unavailable profile')
            expect(memberships[1]?.textContent).not.toContain('Unavailable profile')
            expect(memberships[2]?.textContent).not.toContain('Unavailable profile')
            expect(
                memberships[1]?.querySelector('[aria-label="Model identity"]')?.textContent
            ).toContain('Unknown')
            expect(
                memberships[2]?.querySelector('[aria-label="Model identity"]')?.textContent
            ).toContain('Unknown')
            expect(container.textContent).not.toContain('native-model')
            expect(html).toContain('weak: Unassigned')
            expect(html).toContain('medium: available')
            expect(html.indexOf('strong: missing')).toBeLessThan(html.indexOf('strong: available'))
            expect(html.indexOf('strong:')).toBeLessThan(html.indexOf('weak:'))
            expect(html.indexOf('weak:')).toBeLessThan(html.indexOf('medium:'))
            await act(async () =>
                root.render(createElement(NodeDeclaredRoles, { nodeId: 'absent' }))
            )
            expect(container.innerHTML).toBe('')
        } finally {
            await act(async () => root.unmount())
            container.remove()
            useDiscoveredNodesStore.setState({ nodes: previous })
        }
    })
})
