// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { NodeItem } from '@/shared/types/nodes'
import type { AvailableNode } from '@/shared/types/cluster'
import type { EngineModels } from '@/shared/types/engines'
import type { NodePolicyDocument } from '@/shared/types/node-policy'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi, type INodePolicyApi } from '@/ui/api/pair-api'
import ProfilesView from '@/ui/components/Profiles/ProfilesView'
import RoutingSettings from '@/ui/components/RoutingSettings/RoutingSettings'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { modelSettingsKey, useEngineSettingsStore } from '@/ui/stores/engine-settings.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'

vi.mock('@/ui/components/RoutingSettings/RoleEditor', () => ({ default: () => null }))
vi.mock('@/ui/components/NodeList/NodeEngineSettings', () => ({ default: () => null }))
vi.mock('@/ui/components/ModelManager/ModelCopyModal', () => ({ ModelCopyModal: () => null }))

const getPolicy = vi.fn<INodePolicyApi['get']>()
let root: Root
let container: HTMLDivElement
const originalWindowApi = Object.getOwnPropertyDescriptor(window, 'windowApi')

function node(id: string, name: string): NodeItem {
    return {
        id,
        name,
        status: 'active',
        ipAddress: '127.0.0.1',
        port: 1,
        allIpAddresses: [],
        os: 'Windows',
        topology: { cpu: { model: '', cores: 1, threads: 1 }, ram: 0, gpus: [], storage: [] }
    }
}

function policy(nodeId = 'self'): NodePolicyDocument {
    return {
        revision: '1',
        activation: null,
        availability: 'available',
        recoveryRequired: false,
        policy: JSON.stringify({
            profiles: [
                {
                    key: 'other',
                    displayName: 'Other profile',
                    engine: 'llamacpp',
                    model: 'other-artifact'
                },
                {
                    key: 'shared-key',
                    displayName: `${nodeId} coding`,
                    icon: 'code',
                    engine: 'llamacpp',
                    model: `${nodeId}-artifact`,
                    requestOptions: { temperature: nodeId === 'self' ? 0.25 : 0.75 }
                }
            ],
            roles:
                nodeId === 'self'
                    ? [
                          { name: 'writer', profiles: ['shared-key'] },
                          { name: 'unused', profiles: ['other'] },
                          { name: 'coder', profiles: ['shared-key'] }
                      ]
                    : [{ name: 'chat', profiles: ['other'] }]
        })
    }
}

function button(label: string, scope: ParentNode = container): HTMLButtonElement {
    const found = Array.from(scope.querySelectorAll('button')).find(
        item => item.textContent?.trim() === label
    )
    if (!found) throw new Error(`Missing button ${label}`)
    return found
}

function card(name: string): HTMLElement {
    const edit = Array.from(container.querySelectorAll('button')).find(
        item =>
            item.textContent?.trim() === 'Edit' &&
            item.parentElement?.parentElement?.textContent?.includes(name)
    )
    const found = edit?.parentElement?.parentElement
    if (!found) throw new Error(`Missing profile card ${name}`)
    return found
}

function FocusedContent() {
    const active = useOverviewUiStore(state => state.activeTab)
    return active === 'settings' ? <RoutingSettings /> : <ProfilesView />
}

beforeEach(() => {
    Object.defineProperty(window, 'windowApi', { configurable: true, value: { platform: 'macOS' } })
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => false, escape: (value: string) => value })
    Object.defineProperty(HTMLDialogElement.prototype, 'show', {
        configurable: true,
        value: function (this: HTMLDialogElement) {
            this.open = true
        }
    })
    Object.defineProperty(HTMLDialogElement.prototype, 'showModal', {
        configurable: true,
        value: function (this: HTMLDialogElement) {
            this.open = true
        }
    })
    Object.defineProperty(HTMLDialogElement.prototype, 'close', {
        configurable: true,
        value: function (this: HTMLDialogElement) {
            this.open = false
        }
    })
    getPolicy.mockReset().mockImplementation(async nodeId => policy(nodeId))
    const transport: ServiceTransport = {
        connected: true,
        invoke: vi.fn<ServiceTransport['invoke']>(),
        subscribePush: () => () => {},
        onConnect: () => () => {},
        onDisconnect: () => () => {},
        onAuthFailure: () => () => {},
        destroy: () => {}
    }
    const api = createPairApi(transport)
    api.discovery.getNodes = vi.fn<typeof api.discovery.getNodes>().mockResolvedValue([])
    api.nodePolicy.get = getPolicy
    api.cluster.getRemoteManagementForNode = vi
        .fn<typeof api.cluster.getRemoteManagementForNode>()
        .mockResolvedValue({ version: 1, revision: '1', state: 'allowed', allowed: true })
    api.engines.getSettings = vi
        .fn<typeof api.engines.getSettings>()
        .mockImplementation(async target => ({
            ...target,
            revision: 1,
            appliedRevision: 1,
            sequence: 1,
            epoch: 'test',
            settings: {
                serverPort: 1,
                proxyPort: 2,
                launchText: '',
                modelLoads: [
                    {
                        profileKey: 'shared-key',
                        modelId: `${target.nodeId}-artifact`,
                        requested: { fit: 'on' }
                    }
                ]
            },
            effectiveServerPort: 1,
            effectiveProxyPort: 2,
            running: false,
            adopted: false,
            editable: true,
            reason: '',
            format: '',
            phase: 'idle',
            error: '',
            requestId: ''
        }))
    api.engines.inspectModel = vi
        .fn<typeof api.engines.inspectModel>()
        .mockImplementation(async request => ({
            ...request,
            status: 'resolved',
            estimate: { status: 'unknown' }
        }))
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    useConnectionStore.setState({ connected: true, selfId: 'self' })
    useOverviewUiStore.setState({
        activeTab: 'profiles',
        settingsNodeId: null,
        settingsProfileKey: null
    })
    useNodesStore.setState({
        nodes: new Map([
            ['self', node('self', 'MSI')],
            ['peer', node('peer', 'Joes')]
        ])
    })
    useNodeConfigurationStore.setState({
        editors: new Map(),
        management: new Map(),
        policyDrafts: new Map(),
        availability: new Map()
    })
    useEngineSettingsStore.setState({ entries: {}, modelDrafts: {} })
    useEngineModelsStore.setState({ models: new Map() })
    usePendingActionsStore.setState({ pending: new Map(), deployments: new Map() })
    useClusterInvitationsStore.setState({
        remoteManagementContext: 1,
        members: [
            {
                id: 'peer-host',
                nodeUuid: 'peer',
                name: 'Joes',
                ipAddress: '',
                port: 1,
                clusterId: 'test',
                state: 'member',
                joinedAt: 1,
                lastSeen: 1
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
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    if (originalWindowApi) Object.defineProperty(window, 'windowApi', originalWindowApi)
    else Reflect.deleteProperty(window, 'windowApi')
})

describe('Profiles tab', () => {
    it('excludes another served section identity when its key equals this profile artifact id', async () => {
        const getSettings = window.pairApi.engines.getSettings
        window.pairApi.engines.getSettings = vi
            .fn<typeof getSettings>()
            .mockImplementation(async target => {
                const snapshot = await getSettings(target)
                snapshot.settings.modelLoads?.push({
                    profileKey: `${target.nodeId}-artifact`,
                    modelId: 'other-artifact',
                    requested: {}
                })
                return snapshot
            })
        const observed: EngineModels = {
            nodeId: 'self',
            engineType: 'llama-cpp',
            models: [
                {
                    name: 'self-artifact',
                    downloaded: true,
                    size: 200,
                    status: 'idle',
                    parameterSize: '',
                    family: '',
                    sizeVram: null,
                    expiresAt: null,
                    expiry: '0',
                    capabilities: [],
                    identity: {
                        strength: 'verified',
                        contentId: 'c'.repeat(64),
                        repository: 'owner/Second',
                        quantization: 'Q8_0',
                        totalBytes: 200
                    }
                },
                {
                    name: 'shared-key',
                    downloaded: true,
                    size: 100,
                    status: 'idle',
                    parameterSize: '',
                    family: '',
                    sizeVram: null,
                    expiresAt: null,
                    expiry: '0',
                    capabilities: [],
                    identity: {
                        strength: 'verified',
                        contentId: 'b'.repeat(64),
                        repository: 'owner/Tiel',
                        quantization: 'Q4_K_M',
                        totalBytes: 100
                    }
                }
            ]
        }
        useEngineModelsStore.setState({ models: new Map([['self:llama-cpp', observed]]) })
        await act(async () => root.render(<ProfilesView />))
        const identity = card('self coding').querySelector(
            '[aria-label="Model identity"]'
        )?.textContent
        expect(identity).toContain('owner/Tiel')
        expect(identity).toContain('Q4_K_M')
        expect(identity).not.toContain('owner/Second')
    })
    it.each(['self-artifact', 'shared-key'])(
        'uses inventory identity tags for model name %s',
        async name => {
            const observed: EngineModels = {
                nodeId: 'self',
                engineType: 'llama-cpp',
                models: [
                    {
                        name,
                        downloaded: true,
                        size: 100,
                        status: 'idle',
                        parameterSize: '',
                        family: '',
                        sizeVram: null,
                        expiresAt: null,
                        expiry: '0',
                        capabilities: [],
                        identity: {
                            strength: 'verified',
                            contentId: 'b'.repeat(64),
                            repository: 'owner/Tiel',
                            quantization: 'Q4_K_M',
                            totalBytes: 100
                        }
                    }
                ]
            }
            useEngineModelsStore.setState({ models: new Map([['self:llama-cpp', observed]]) })
            await act(async () => root.render(<ProfilesView />))
            expect(
                card('self coding').querySelector('[aria-label="Model identity"]')?.textContent
            ).toContain('owner/Tiel')
            expect(
                card('self coding').querySelector('[aria-label="Model identity"]')?.textContent
            ).toContain('Q4_K_M')
        }
    )

    it('keeps first-entry loading rows until the pending advertisement read completes', async () => {
        let finish: ((nodes: AvailableNode[]) => void) | undefined
        window.pairApi.discovery.getNodes = vi.fn<typeof window.pairApi.discovery.getNodes>(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        await act(async () => root.render(<ProfilesView />))
        expect(getPolicy).toHaveBeenCalledTimes(2)
        expect(
            Array.from(container.querySelectorAll('[role="status"]')).map(row => row.textContent)
        ).toEqual(['Loading profiles for MSI…', 'Loading profiles for Joes…'])
        expect(container.textContent).not.toContain('self coding')
        expect(container.textContent).not.toContain('peer coding')
        const resolveAdvertisement = finish
        if (!resolveAdvertisement) throw new Error('Missing advertisement read')
        await act(async () => resolveAdvertisement([]))
        expect(container.querySelector('[role="status"]')).toBeNull()
        expect(card('self coding')).toBeDefined()
        expect(card('peer coding')).toBeDefined()
    })
    it('keeps current saved CPU placement authoritative over retained native-fit inspection', async () => {
        const getSettings = window.pairApi.engines.getSettings
        window.pairApi.engines.getSettings = vi
            .fn<typeof getSettings>()
            .mockImplementation(async target => {
                const snapshot = await getSettings(target)
                return {
                    ...snapshot,
                    settings: {
                        ...snapshot.settings,
                        modelLoads: [
                            {
                                profileKey: 'shared-key',
                                modelId: `${target.nodeId}-artifact`,
                                requested: { fit: 'off', gpu: { mode: 'cpu' } }
                            }
                        ]
                    }
                }
            })
        useEngineSettingsStore.setState({
            modelDrafts: {
                [modelSettingsKey({ nodeId: 'self', engine: 'llamacpp' }, 'shared-key')]: {
                    generation: 0,
                    inspection: {
                        nodeId: 'self',
                        engine: 'llamacpp',
                        modelId: 'self-artifact',
                        status: 'resolved',
                        estimate: { status: 'unknown' },
                        configurationKeyReason: 'native-fit'
                    }
                }
            }
        })
        await act(async () => root.render(<ProfilesView />))
        expect(card('self coding').textContent).toContain('GPU: CPU')
        expect(card('self coding').textContent).not.toContain('Native fit')
    })
    it('refreshes cached cards and usage on reentry while preserving an unsaved role draft', async () => {
        await act(async () => root.render(<ProfilesView />))
        useNodeConfigurationStore
            .getState()
            .setDraft('self', [{ name: 'draft-role', profiles: [] }])
        await act(async () => root.render(<div />))
        getPolicy.mockImplementation(async nodeId => ({
            ...policy(nodeId),
            revision: '2',
            policy: policy(nodeId)
                .policy.replace('coding', 'updated')
                .replace('"writer"', '"reader"')
        }))
        await act(async () => root.render(<ProfilesView />))
        expect(card('self updated').textContent).toContain('reader on MSI')
        expect(container.textContent).not.toContain('writer on MSI')
        expect(useNodeConfigurationStore.getState().editors.get('self')?.draft).toEqual([
            { name: 'draft-role', profiles: [] }
        ])
    })
    it('renders own-node usage in role order without leaking chips to the same key on another node', async () => {
        await act(async () => root.render(<ProfilesView />))
        const own = card('self coding').textContent ?? ''
        expect(own).toContain('writer on MSI')
        expect(own).toContain('coder on MSI')
        expect(own.indexOf('writer on MSI')).toBeLessThan(own.indexOf('coder on MSI'))
        expect(own).not.toContain('unused on MSI')
        const peer = card('peer coding').textContent ?? ''
        expect(peer).not.toContain('writer on MSI')
        expect(peer).not.toContain('coder on MSI')
        expect(peer).not.toContain('chat on Joes')
    })

    it('renders Native fit from the real engine settings store', async () => {
        await act(async () => root.render(<ProfilesView />))
        expect(card('self coding').textContent).toContain('GPU: Native fit')
    })

    it('opens the reusable deploy controls with the clicked profile preselected', async () => {
        await act(async () => root.render(<ProfilesView />))
        await act(async () => button('Deploy to node', card('self coding')).click())
        expect(card('self coding').textContent).toContain('Temperature: Set 0.25')
        const label = Array.from(container.querySelectorAll('label')).find(item =>
            item.textContent?.includes('Deploy to Joes')
        )
        const checkbox = label?.control
        if (!(checkbox instanceof HTMLInputElement))
            throw new Error('Missing Joes deployment checkbox')
        await act(async () => checkbox.click())
        expect(
            container.querySelector<HTMLInputElement>('input[aria-label="Profile key peer"]')?.value
        ).toBe('shared-key')
        expect(window.pairApi.engines.inspectModel).toHaveBeenCalledWith({
            nodeId: 'self',
            engine: 'llamacpp',
            modelId: 'self-artifact',
            profileKey: 'shared-key'
        })
    })

    it('uses existing focus navigation to open the selected peer profile editor', async () => {
        await act(async () => root.render(<FocusedContent />))
        await act(async () => button('Edit', card('peer coding')).click())
        expect(useOverviewUiStore.getState()).toMatchObject({
            activeTab: 'settings',
            settingsSubTab: 'routing',
            settingsNodeId: 'peer',
            settingsProfileKey: 'shared-key'
        })
        const dialog = document.querySelector('dialog[open]')
        expect(dialog).not.toBeNull()
        expect(dialog?.textContent).toContain('Node: Joes')
        expect(
            document.querySelector<HTMLInputElement>('input[aria-label="Temperature"]')?.value
        ).toBe('0.75')
        expect(window.pairApi.engines.inspectModel).toHaveBeenCalledWith({
            nodeId: 'peer',
            engine: 'llamacpp',
            modelId: 'peer-artifact',
            profileKey: 'shared-key'
        })
    })

    it('reads each node on first entry and keeps a loading row until its policy arrives', async () => {
        const finishes = new Map<string, (document: NodePolicyDocument) => void>()
        getPolicy.mockImplementation(
            nodeId =>
                new Promise(resolve => {
                    finishes.set(nodeId ?? 'self', resolve)
                })
        )
        await act(async () => root.render(<ProfilesView />))
        expect(getPolicy.mock.calls).toEqual([[undefined], ['peer']])
        expect(window.pairApi.discovery.getNodes).toHaveBeenCalledOnce()
        expect(
            Array.from(container.querySelectorAll('[role="status"]')).map(row => row.textContent)
        ).toEqual(['Loading profiles for MSI…', 'Loading profiles for Joes…'])
        const finish = finishes.get('self')
        if (!finish) throw new Error('Missing local policy read')
        await act(async () => finish(policy()))
        expect(container.textContent).not.toContain('Loading profiles for MSI')
        expect(container.textContent).toContain('Loading profiles for Joes')
        expect(card('self coding')).toBeDefined()
    })

    it('shows unavailable instead of an empty profile list when management is denied', async () => {
        window.pairApi.cluster.getRemoteManagementForNode = vi
            .fn<typeof window.pairApi.cluster.getRemoteManagementForNode>()
            .mockResolvedValue({ version: 1, revision: '1', state: 'denied', allowed: false })
        await act(async () => root.render(<ProfilesView />))
        expect(container.textContent).toContain('Profiles unavailable:')
        expect(container.textContent).not.toContain('peer coding')
        expect(container.textContent).not.toContain('No profiles configured.')
    })

    it('shows the failed policy read reason instead of an empty profile list', async () => {
        getPolicy.mockImplementation(async nodeId => {
            if (nodeId === 'peer') throw new Error('Policy inspection unavailable')
            return policy()
        })
        await act(async () => root.render(<ProfilesView />))
        expect(container.textContent).toContain(
            'Profiles unavailable: Policy inspection unavailable'
        )
        expect(container.textContent).not.toContain('peer coding')
        expect(container.textContent).not.toContain('No profiles configured.')
    })
})
