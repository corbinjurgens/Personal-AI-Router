// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import NodeDeclaredRoles from '@/ui/components/NodeList/NodeDeclaredRoles'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { useEngineSettingsStore } from '@/ui/stores/engine-settings.store'
import type { EngineSettingsSnapshot } from '@/shared/types/engine-settings'
import type { IEngineApi } from '@/ui/api/engine-api'

const saved: EngineSettingsSnapshot = {
    nodeId: 'peer',
    engine: 'llamacpp',
    revision: 1,
    appliedRevision: 1,
    sequence: 1,
    epoch: 'test',
    effectiveServerPort: 1,
    effectiveProxyPort: 2,
    running: false,
    adopted: false,
    editable: true,
    reason: '',
    format: '',
    phase: 'idle',
    error: '',
    requestId: '',
    settings: {
        serverPort: 1,
        proxyPort: 2,
        launchText: '',
        modelLoads: [
            {
                profileKey: 'first',
                modelId: 'shared-artifact',
                requested: {
                    context: { mode: 'fixed', tokens: 8192 },
                    gpu: { mode: 'layers', layers: 12 },
                    extraOptions: { alpha: 'one', beta: 'two' }
                }
            },
            { profileKey: 'second', modelId: 'shared-artifact', requested: { fit: 'on' } }
        ]
    }
}
const getSettings = vi.fn<IEngineApi['getSettings']>()
const stopChanged = vi.fn<() => void>()
const stopDisconnected = vi.fn<() => void>()
const onChanged = vi.fn<IEngineApi['onSettingsChanged']>()
const onDisconnected = vi.fn<IEngineApi['onSettingsDisconnected']>()

let root: Root
let container: HTMLDivElement
const longName = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789AB'

beforeEach(() => {
    vi.clearAllMocks()
    getSettings.mockResolvedValue(structuredClone(saved))
    onChanged.mockReturnValue(stopChanged)
    onDisconnected.mockReturnValue(stopDisconnected)
    Object.defineProperty(window, 'pairApi', {
        configurable: true,
        value: {
            engines: {
                getSettings,
                onSettingsChanged: onChanged,
                onSettingsDisconnected: onDisconnected
            }
        }
    })
    useConnectionStore.setState({ connected: true, selfId: 'peer' })
    useClusterInvitationsStore.setState({ remoteManagementContext: null })
    useEngineSettingsStore.setState({ entries: {}, modelDrafts: {} })
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    useOverviewUiStore.setState({
        showProfileSummaries: false,
        settingsNodeId: null,
        settingsProfileKey: null
    })
    useEngineModelsStore.setState({ models: new Map() })
    useDiscoveredNodesStore.setState({
        nodes: [
            {
                id: 'peer',
                name: 'Peer',
                ipAddress: '127.0.0.1',
                port: 1,
                lastSeen: 0,
                trusted: true,
                clustered: true,
                policy: {
                    revision: '1',
                    roles: [
                        { name: 'weak', profiles: ['first', 'second'] },
                        { name: 'strong', profiles: ['first'] }
                    ],
                    profiles: [
                        {
                            key: 'first',
                            displayName: longName,
                            icon: 'code',
                            engine: 'llamacpp',
                            model: 'first',
                            contextTokens: 65536,
                            modelIdentity: {
                                repository: 'owner/repository',
                                quantization: 'Q4_K_M',
                                strength: 'declared'
                            }
                        },
                        {
                            key: 'second',
                            displayName: 'Second profile',
                            engine: 'llamacpp',
                            model: 'looks-like-Q8'
                        }
                    ]
                }
            }
        ]
    })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.unstubAllGlobals()
})

async function renderRoles() {
    await act(async () => root.render(<NodeDeclaredRoles nodeId="peer" />))
}

describe('node card role members', () => {
    it('renders every ordered member with identity tags and no display names by default', async () => {
        await renderRoles()
        expect(
            Array.from(container.querySelectorAll('button'), button =>
                button.getAttribute('aria-label')
            )
        ).toEqual(['weak: first', 'weak: second', 'strong: first'])
        expect(container.textContent).toContain('owner/repository')
        expect(container.textContent).toContain('Q4_K_M')
        expect(container.textContent).not.toContain(longName)
        expect(container.textContent).not.toContain('Second profile')
        expect(getSettings).not.toHaveBeenCalled()
        expect(onChanged).not.toHaveBeenCalled()
    })

    it('shows unknown repository and quantization without guessing from a served name', async () => {
        await renderRoles()
        const member = container.querySelectorAll('button')[1]
        expect(member.textContent).toBe('weakUnknownUnknown')
    })

    it('toggles summaries while retaining identity and full 64-byte names', async () => {
        await renderRoles()
        await act(async () => useOverviewUiStore.getState().setShowProfileSummaries(true))
        const member = container.querySelector('button')
        expect(member?.getAttribute('aria-label')).toBe(`weak: ${longName} (first)`)
        expect(member?.title).toBe(`${longName} (first)`)
        expect(member?.textContent).toContain(longName)
        expect(member?.querySelector('svg')?.getAttribute('class')).toContain('lucide-code')
        expect(member?.querySelector('svg')?.getAttribute('aria-hidden')).toBe('true')
        expect(member?.textContent).toContain('Context 65536')
        expect(member?.textContent).toContain('owner/repository')
        expect(member?.textContent).toContain('Q4_K_M')
        await act(async () => useOverviewUiStore.getState().setShowProfileSummaries(false))
        expect(container.textContent).not.toContain(longName)
        expect(container.textContent).toContain('owner/repository')
    })

    it('focuses the clicked profile and clears it when another node is selected', async () => {
        await renderRoles()
        await act(async () => container.querySelectorAll('button')[1].click())
        expect(useOverviewUiStore.getState()).toMatchObject({
            activeTab: 'settings',
            settingsSubTab: 'routing',
            settingsNodeId: 'peer',
            settingsProfileKey: 'second'
        })
        useOverviewUiStore.getState().selectSettingsNode('peer')
        expect(useOverviewUiStore.getState().settingsProfileKey).toBe('second')
        useOverviewUiStore.getState().selectSettingsNode('other')
        expect(useOverviewUiStore.getState().settingsProfileKey).toBeNull()
        useOverviewUiStore.getState().focusNodeSettings('peer')
        expect(useOverviewUiStore.getState().settingsProfileKey).toBeNull()
    })

    it('reads a shared engine once and renders keyed saved siblings', async () => {
        useOverviewUiStore.setState({ showProfileSummaries: true })
        await renderRoles()
        expect(getSettings.mock.calls).toEqual([[{ nodeId: 'peer', engine: 'llamacpp' }]])
        expect(onChanged.mock.invocationCallOrder[0]).toBeLessThan(
            getSettings.mock.invocationCallOrder[0]
        )
        const members = container.querySelectorAll('button')
        expect(members[0].textContent).toContain(
            'Saved context 8192 · GPU: 12 layers · 2 extra options'
        )
        expect(members[1].textContent).toContain('Native fit on · 0 extra options')
        expect(members[2].textContent).toContain('GPU: 12 layers')
        await act(async () => useOverviewUiStore.getState().setShowProfileSummaries(false))
        expect(stopChanged).toHaveBeenCalledOnce()
        expect(stopDisconnected).toHaveBeenCalledOnce()
    })

    it('ignores cached saved facts until the new read succeeds and rejects inaccessible data', async () => {
        useEngineSettingsStore.getState().receive(saved)
        let rejectRead: (error: Error) => void = () => {
            throw new Error('Read not started')
        }
        getSettings.mockImplementationOnce(
            () =>
                new Promise((_resolve, reject) => {
                    rejectRead = reject
                })
        )
        useOverviewUiStore.setState({ showProfileSummaries: true })
        await renderRoles()
        expect(container.textContent).toContain('Saved load unavailable')
        expect(container.textContent).not.toContain('GPU: 12 layers')
        await act(async () => rejectRead(new Error('Permission denied')))
        expect(container.textContent).toContain('Saved load unavailable')
        expect(container.textContent).not.toContain('GPU: 12 layers')
    })

    it('withdraws disconnected saved facts and reads again on reconnect', async () => {
        useOverviewUiStore.setState({ showProfileSummaries: true })
        await renderRoles()
        expect(container.textContent).toContain('GPU: 12 layers')
        await act(async () => useConnectionStore.setState({ connected: false }))
        expect(container.textContent).not.toContain('GPU: 12 layers')
        expect(container.textContent).toContain('Saved load unavailable')
        expect(getSettings).toHaveBeenCalledTimes(1)
        await act(async () => useConnectionStore.setState({ connected: true }))
        expect(getSettings).toHaveBeenCalledTimes(2)
        expect(container.textContent).toContain('GPU: 12 layers')
    })

    it('tears down the shared listeners on unmount', async () => {
        useOverviewUiStore.setState({ showProfileSummaries: true })
        await renderRoles()
        await act(async () => root.render(null))
        expect(stopChanged).toHaveBeenCalledOnce()
        expect(stopDisconnected).toHaveBeenCalledOnce()
    })

    it('withdraws remote facts when management authority is lost and refreshes when restored', async () => {
        useConnectionStore.setState({ selfId: 'local' })
        useClusterInvitationsStore.setState({ remoteManagementContext: 1 })
        useOverviewUiStore.setState({ showProfileSummaries: true })
        await renderRoles()
        expect(container.textContent).toContain('GPU: 12 layers')
        await act(async () =>
            useClusterInvitationsStore.setState({ remoteManagementContext: null })
        )
        expect(container.textContent).toContain('Saved load unavailable')
        expect(container.textContent).not.toContain('GPU: 12 layers')
        expect(getSettings).toHaveBeenCalledTimes(1)
        await act(async () => useClusterInvitationsStore.setState({ remoteManagementContext: 2 }))
        expect(getSettings).toHaveBeenCalledTimes(2)
        expect(container.textContent).toContain('GPU: 12 layers')
    })

    it('keeps a read that loses its connection unavailable when it resolves late', async () => {
        let finishRead: (snapshot: EngineSettingsSnapshot) => void = () => {
            throw new Error('Read not started')
        }
        getSettings.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finishRead = resolve
                })
        )
        useOverviewUiStore.setState({ showProfileSummaries: true })
        await renderRoles()
        await act(async () => useConnectionStore.setState({ connected: false }))
        await act(async () => finishRead(saved))
        expect(container.textContent).toContain('Saved load unavailable')
        expect(container.textContent).not.toContain('GPU: 12 layers')
    })

    it('rejects an uncached pre-reconnect read when the fresh read is denied', async () => {
        let oldRead: (snapshot: EngineSettingsSnapshot) => void = () => {
            throw new Error('Old read not started')
        }
        let denyFresh: (error: Error) => void = () => {
            throw new Error('Fresh read not started')
        }
        getSettings.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    oldRead = resolve
                })
        )
        getSettings.mockImplementationOnce(
            () =>
                new Promise((_resolve, reject) => {
                    denyFresh = reject
                })
        )
        useOverviewUiStore.setState({ showProfileSummaries: true })
        await renderRoles()
        expect(getSettings).toHaveBeenCalledTimes(1)
        await act(async () => useConnectionStore.setState({ connected: false }))
        await act(async () => useConnectionStore.setState({ connected: true }))
        expect(getSettings).toHaveBeenCalledTimes(2)
        await act(async () => oldRead(saved))
        await act(async () => denyFresh(new Error('Management permission denied')))
        expect(container.textContent).toContain('Saved load unavailable')
        expect(container.textContent).not.toContain('GPU: 12 layers')
    })

    it('retains empty declared roles and unavailable memberships explicitly', async () => {
        useDiscoveredNodesStore.setState(state => ({
            nodes: state.nodes.map(node => ({
                ...node,
                policy: node.policy
                    ? {
                          ...node.policy,
                          roles: [
                              { name: 'weak', profiles: [] },
                              { name: 'strong', profiles: ['missing'] }
                          ]
                      }
                    : undefined
            }))
        }))
        await renderRoles()
        expect(container.textContent).toContain('weak: Unassigned')
        expect(container.querySelector('button')?.textContent).toContain('Unavailable profile')
        expect(container.querySelector('button')?.getAttribute('aria-label')).toBe(
            'strong: missing'
        )
    })
})
