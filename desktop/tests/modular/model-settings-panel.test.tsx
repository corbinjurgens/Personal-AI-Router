// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { JsonValue } from '@/shared/types/json'
import type { ModelItem } from '@/shared/types/engines'
import type {
    NodeAvailabilityChange,
    NodePolicyChange,
    NodePolicyDocument
} from '@/shared/types/node-policy'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import {
    REMOTE_MANAGEMENT_DENIED_CODE,
    REMOTE_SETTING_LOCKED_CODE
} from '@/shared/constants/service-errors'
import { parseModelSettings } from '@/shared/utils/node-policy-model-settings'
import { createPairApi, type INodePolicyApi } from '@/ui/api/pair-api'
import { ServiceCallError } from '@/ui/api/service-transport'
import { ModelManager } from '@/ui/components/ModelManager/ModelManager'
import ModelSettingsPanel from '@/ui/components/ModelSettings/ModelSettingsPanel'
import RoutingSettings from '@/ui/components/RoutingSettings/RoutingSettings'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { modelSettingsKey, useEngineSettingsStore } from '@/ui/stores/engine-settings.store'

vi.mock('@/ui/components/RoutingSettings/RoleEditor', () => ({ default: () => null }))
vi.mock('@/ui/components/NodeList/NodeEngineSettings', () => ({ default: () => null }))
vi.mock('@/ui/components/ModelManager/ModelCopyModal', () => ({ ModelCopyModal: () => null }))
vi.mock('@/ui/components/ModelHub/ModelHubModal', () => ({ ModelHubModal: () => null }))

const identity = { engine: 'llamacpp', model: 'owner/model:Q4_K_M' }
const getPolicy = vi.fn<INodePolicyApi['get']>()
const setPolicy = vi.fn<INodePolicyApi['set']>()
const policyListeners = new Set<(value: NodePolicyChange) => void>()
const availabilityListeners = new Set<(value: NodeAvailabilityChange) => void>()
let root: Root
let container: HTMLDivElement

function saved(
    extra: Record<string, JsonValue> = {},
    revision = '1',
    availability: NodePolicyDocument['availability'] = 'available'
): NodePolicyDocument {
    return {
        policy: JSON.stringify({
            version: 2,
            revision,
            availability,
            profiles: [{ key: identity.model, displayName: 'profile', ...identity }],
            ...extra
        }),
        revision,
        availability,
        activation: null,
        recoveryRequired: false
    }
}
function draft(targetId = 'self') {
    const entry = useNodeConfigurationStore.getState().policyDrafts.get(targetId)
    if (!entry) throw new Error(`Missing ${targetId} draft`)
    return entry
}
function selected(targetId = 'self') {
    const parsed = parseModelSettings(draft(targetId).text)
    if (!parsed.ok) throw new Error(parsed.error.message)
    return parsed.value.find(
        item => item.engine === identity.engine && item.model === identity.model
    )
}
function button(label: string, scope: ParentNode = document): HTMLButtonElement {
    const found = Array.from(scope.querySelectorAll('button')).find(
        item => item.textContent?.trim() === label || item.getAttribute('aria-label') === label
    )
    if (!found) throw new Error(`Missing ${label} button`)
    return found
}
function panel(): HTMLDialogElement {
    const found = document.querySelector('dialog[open]')
    if (!(found instanceof HTMLDialogElement)) throw new Error('Missing open model settings panel')
    return found
}
function input(label: string): HTMLInputElement {
    const found = document.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)
    if (!found) throw new Error(`Missing ${label} input`)
    return found
}
async function type(label: string, value: string) {
    const field = input(label)
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
    if (!setter) throw new Error('Missing input value setter')
    await act(async () => {
        setter.call(field, value)
        field.dispatchEvent(new Event('input', { bubbles: true }))
    })
}
async function editJson(value: string) {
    const field = container.querySelector<HTMLTextAreaElement>(
        'textarea[aria-label="Node policy JSON"]'
    )
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set
    if (!field || !setter) throw new Error('Missing Advanced policy JSON field')
    await act(async () => {
        setter.call(field, value)
        field.dispatchEvent(new Event('input', { bubbles: true }))
    })
}
async function render(content: ReactNode) {
    await act(async () => root.render(content))
}
function rowPanel(targetId = 'self') {
    return (
        <ModelSettingsPanel
            key={targetId}
            targetId={targetId}
            nodeId={targetId === 'self' ? undefined : targetId}
            nodeName={targetId === 'self' ? 'This PC' : targetId}
            identity={identity}
            open
        />
    )
}
function push(snapshot: NodePolicyDocument, targetId = 'self') {
    for (const listener of policyListeners) listener({ ...snapshot, nodeId: targetId })
}
function model(): ModelItem {
    return {
        name: identity.model,
        size: 1,
        downloaded: false,
        status: 'idle',
        parameterSize: '',
        family: '',
        sizeVram: null,
        expiresAt: null,
        expiry: '10m',
        capabilities: []
    }
}
function manager() {
    return (
        <ModelManager
            backend={{
                type: 'llama-cpp',
                displayName: 'llama.cpp',
                source: 'installed',
                processStatus: 'stopped',
                port: null,
                proxyPort: null,
                models: [model()]
            }}
            nodeId="self"
        />
    )
}

beforeEach(() => {
    useOverviewUiStore.setState({ settingsNodeId: null, settingsProfileKey: null })
    useEngineSettingsStore.setState({ entries: {}, modelDrafts: {} })
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
    policyListeners.clear()
    availabilityListeners.clear()
    getPolicy.mockReset().mockResolvedValue(saved())
    setPolicy.mockReset().mockImplementation(async request => ({
        ...saved({}, '2'),
        policy: request.policy,
        revision: '2'
    }))
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
    api.nodePolicy.get = getPolicy
    api.nodePolicy.set = setPolicy
    api.nodePolicy.onPolicyChanged = listener => {
        policyListeners.add(listener)
        return () => {
            policyListeners.delete(listener)
        }
    }
    api.nodePolicy.onAvailabilityChanged = listener => {
        availabilityListeners.add(listener)
        return () => {
            availabilityListeners.delete(listener)
        }
    }
    api.cluster.getRemoteManagementForNode = vi
        .fn<typeof api.cluster.getRemoteManagementForNode>()
        .mockResolvedValue({ version: 1, revision: '1', state: 'allowed', allowed: true })
    api.engines.loadModel = vi.fn<typeof api.engines.loadModel>()
    api.engines.deleteModel = vi.fn<typeof api.engines.deleteModel>()
    api.engines.unloadModel = vi.fn<typeof api.engines.unloadModel>()
    api.engines.getSettings = vi
        .fn<typeof api.engines.getSettings>()
        .mockImplementation(async request => ({
            ...request,
            revision: 1,
            appliedRevision: 1,
            sequence: 1,
            epoch: 'test',
            settings: { serverPort: 1, proxyPort: 2, launchText: '', modelLoads: [] },
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
    useNodesStore.setState({ nodes: new Map() })
    useNodeConfigurationStore.setState({
        policyDrafts: new Map(),
        management: new Map(),
        editors: new Map()
    })
    useEngineModelsStore.setState({ models: new Map() })
    useEngineProgressStore.setState({ progress: new Map() })
    usePendingActionsStore.setState({ pending: new Map() })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})
afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    useNodeConfigurationStore.setState({
        policyDrafts: new Map(),
        management: new Map(),
        editors: new Map()
    })
    vi.unstubAllGlobals()
})

describe('composed model settings', () => {
    it('isolates a new profile when its suggested key already has a saved load and dirty draft', async () => {
        const profileKey = 'owner-model-q4_k_m'
        getPolicy.mockResolvedValue(
            saved({
                profiles: [
                    {
                        key: profileKey,
                        displayName: 'Existing profile',
                        ...identity,
                        model: profileKey
                    }
                ]
            })
        )
        const readSettings = window.pairApi.engines.getSettings
        window.pairApi.engines.getSettings = vi
            .fn<typeof window.pairApi.engines.getSettings>()
            .mockImplementation(async request => {
                const snapshot = await readSettings(request)
                return {
                    ...snapshot,
                    settings: {
                        ...snapshot.settings,
                        modelLoads: [
                            { profileKey, modelId: identity.model, requested: { slots: 7 } }
                        ]
                    }
                }
            })
        await render(rowPanel())
        const existingKey = modelSettingsKey({ nodeId: 'self', engine: 'llamacpp' }, profileKey)
        const existingChange = {
            modelId: identity.model,
            profileKey,
            expectedPolicyRevision: '1',
            policy: { displayName: 'Unsaved existing name' }
        }
        await act(async () =>
            useEngineSettingsStore.getState().editModel(existingKey, existingChange)
        )
        await act(async () => button('New profile', panel()).click())
        expect(input('Profile key').value).toBe(profileKey)
        expect(panel().textContent).toContain('This profile key already exists.')
        expect(panel().textContent).toContain('Saved requested overrides: {}')
        expect(button('Preview model changes', panel()).disabled).toBe(true)
        expect(panel().textContent).not.toContain('Remove profile')
        await type('Profile key', 'fresh-profile')
        expect(useEngineSettingsStore.getState().modelDrafts[existingKey].change).toEqual(
            existingChange
        )
        expect(panel().textContent).toContain('Saved requested overrides: {}')
        expect(button('Create profile', panel()).disabled).toBe(false)
        expect(setPolicy).not.toHaveBeenCalled()
    })

    it('creates a sibling with an explicit key and the original artifact model id', async () => {
        await render(rowPanel())
        await act(async () => button('New profile', panel()).click())
        expect(input('Profile key').value).toBe('owner-model-q4_k_m')
        await type('Profile key', 'second-profile')
        await type('Display name', 'Large context')
        await act(async () => button('Create profile', panel()).click())
        expect(
            useEngineSettingsStore.getState().modelDrafts[
                modelSettingsKey({ nodeId: 'self', engine: 'llamacpp' }, 'second-profile')
            ]?.change
        ).toEqual({
            profileKey: 'second-profile',
            modelId: 'owner/model:Q4_K_M',
            expectedPolicyRevision: '1',
            load: { set: {}, clear: ['context'] },
            policy: { displayName: 'Large context', icon: '' }
        })
        expect(selected()?.displayName).toBe('profile')
        expect(setPolicy).not.toHaveBeenCalled()
    })

    it('blocks a reserved profile key while accepting coder-local', async () => {
        await render(rowPanel())
        await act(async () => button('New profile', panel()).click())
        await type('Profile key', 'coder:fast')
        expect(button('Create profile', panel()).disabled).toBe(true)
        await type('Profile key', 'coder-local')
        expect(button('Create profile', panel()).disabled).toBe(false)
    })

    it('opens the card-selected keyed request editor on the selected node', async () => {
        getPolicy.mockResolvedValue(
            saved({
                profiles: [
                    {
                        key: 'small',
                        displayName: 'Small context',
                        engine: 'llamacpp',
                        model: 'small',
                        requestOptions: { temperature: 0.1 }
                    },
                    {
                        key: 'large',
                        displayName: 'Large context',
                        engine: 'llamacpp',
                        model: 'large',
                        requestOptions: { temperature: 0.8 }
                    }
                ]
            })
        )
        useOverviewUiStore.getState().focusNodeSettings('self', 'large')
        await render(<RoutingSettings />)
        expect(panel().textContent).toContain('Draft profile: Large context')
        expect(input('Temperature').value).toBe('0.8')
        await type('Temperature', '0.6')
        const parsed = parseModelSettings(draft().text)
        if (!parsed.ok) throw new Error(parsed.error.message)
        expect(parsed.value.map(profile => [profile.key, profile.options.temperature])).toEqual([
            ['small', 0.1],
            ['large', 0.6]
        ])
    })

    it.each(['Model Manager', 'Routing'])(
        'edits the same keyed generation fields from %s without engine commands',
        async origin => {
            if (origin === 'Routing') {
                await render(<RoutingSettings />)
                await act(async () => button('Configure model').click())
            } else await render(rowPanel())
            expect(panel().textContent).toContain('Node: This PC')
            expect(panel().textContent).toContain(identity.model)
            await type('Temperature', '0.6')
            expect(selected()).toMatchObject({
                key: identity.model,
                displayName: 'profile',
                mode: 'defaults',
                options: { temperature: 0.6 }
            })
            await act(async () => button('Reset Temperature to inherited', panel()).click())
            expect(selected()?.options).toEqual({})
            await type('Temperature', '0.25')
            await act(async () => button('Save', panel()).click())
            expect(setPolicy).toHaveBeenCalledExactlyOnceWith({
                nodeId: undefined,
                expectedPersistedRevision: '1',
                policy: expect.any(String)
            })
            const request = setPolicy.mock.lastCall?.[0]
            if (!request) throw new Error('Missing saved policy')
            expect(JSON.parse(request.policy).profiles).toEqual([
                {
                    key: identity.model,
                    displayName: 'profile',
                    ...identity,
                    requestOptions: { temperature: 0.25 }
                }
            ])
            expect(window.pairApi.engines.loadModel).not.toHaveBeenCalled()
            expect(window.pairApi.engines.deleteModel).not.toHaveBeenCalled()
        }
    )

    it('opens Configure for a known Off row with unknown content readiness', async () => {
        await render(manager())
        const trigger = container.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]')
        if (!trigger) throw new Error('Missing model action menu')
        await act(async () => trigger.click())
        const action = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
            item => item.textContent?.trim() === 'Configure generation settings'
        )
        if (!action) throw new Error('Missing Configure action for Off model')
        await act(async () => action.click())
        expect(panel().textContent).toContain(identity.model)
        expect(panel().textContent).toContain('content completeness is unknown')
        expect(input('Temperature').disabled).toBe(false)
        expect(window.pairApi.engines.loadModel).not.toHaveBeenCalled()
    })

    it('selects an existing profile by exact model in Routing', async () => {
        getPolicy.mockResolvedValue(
            saved({
                profiles: [
                    { key: 'llama-profile', displayName: 'llama-profile', ...identity },
                    {
                        key: 'second-profile',
                        displayName: 'Second profile',
                        engine: 'llamacpp',
                        model: 'owner/second:Q8_0',
                        contextTokens: 65536
                    }
                ]
            })
        )
        await render(<RoutingSettings />)
        await act(async () => button('Configure model').click())
        const label = Array.from(panel().querySelectorAll('label')).find(
            item => item.textContent?.trim() === 'Existing profile'
        )
        const trigger = label && document.getElementById(label.htmlFor)
        if (!(trigger instanceof HTMLButtonElement)) throw new Error('Missing profile selector')
        await act(async () => trigger.click())
        const options = Array.from(
            document.querySelectorAll<HTMLElement>('[role="option"]')
        ).filter(item => item.textContent?.includes(' · Context '))
        // Without a stored load, the label falls back to the policy profile's declared context.
        expect(options.map(item => item.textContent?.trim())).toEqual([
            'llama-profile (llama-profile) · Context Unknown',
            'Second profile (second-profile) · Context 65536'
        ])
        const option = options.find(
            item => item.textContent?.trim() === 'Second profile (second-profile) · Context 65536'
        )
        if (!option) throw new Error('Missing second profile option')
        await act(async () => option.click())
        await type('Temperature', '0.4')
        const parsed = parseModelSettings(draft().text)
        if (!parsed.ok) throw new Error(parsed.error.message)
        expect(
            parsed.value.map(profile => ({ engine: profile.engine, options: profile.options }))
        ).toEqual([
            { engine: 'llamacpp', options: {} },
            { engine: 'llamacpp', options: { temperature: 0.4 } }
        ])
    })

    it('stages untyped resident profile removal through the coupled operation', async () => {
        getPolicy.mockResolvedValue(saved({ residentModels: [identity] }))
        await render(rowPanel())
        expect(button('Remove profile', panel()).disabled).toBe(false)
        await act(async () => button('Remove profile', panel()).click())
        expect(panel().textContent).toContain('Served permission stays in place')
        await act(async () => button('Confirm remove profile', panel()).click())
        expect(
            useEngineSettingsStore.getState().modelDrafts[
                modelSettingsKey({ nodeId: 'self', engine: 'llamacpp' }, identity.model)
            ]?.change
        ).toEqual({
            modelId: identity.model,
            profileKey: identity.model,
            expectedPolicyRevision: '1',
            remove: true
        })
        expect(panel().textContent).toContain('Setup does not create qualifications')
        expect(setPolicy).not.toHaveBeenCalled()
    })

    it('requires explicit profile removal and leaves model files alone', async () => {
        await render(rowPanel())
        await act(async () => button('Remove profile', panel()).click())
        expect(selected()).toBeDefined()
        await act(async () => button('Confirm remove profile', panel()).click())
        expect(selected()).toBeDefined()
        expect(
            useEngineSettingsStore.getState().modelDrafts[
                modelSettingsKey({ nodeId: 'self', engine: 'llamacpp' }, identity.model)
            ]?.change?.remove
        ).toBe(true)
        expect(window.pairApi.engines.deleteModel).not.toHaveBeenCalled()
        expect(setPolicy).not.toHaveBeenCalled()
    })

    it('reviews typed removal against the served key and preserves the artifact id', async () => {
        getPolicy.mockResolvedValue(
            saved({
                profiles: [
                    {
                        key: 'large',
                        displayName: 'Large context',
                        engine: 'llamacpp',
                        model: 'large'
                    }
                ]
            })
        )
        const readSettings = window.pairApi.engines.getSettings
        window.pairApi.engines.getSettings = vi
            .fn<typeof readSettings>()
            .mockImplementation(async request => {
                const result = await readSettings(request)
                return {
                    ...result,
                    settings: {
                        ...result.settings,
                        modelLoads: [
                            { profileKey: 'large', modelId: identity.model, requested: {} }
                        ]
                    }
                }
            })
        await render(rowPanel())
        await act(async () => button('Remove profile', panel()).click())
        expect(panel().textContent).toContain(
            'served permission and residency for this profile key'
        )
        expect(
            useEngineSettingsStore.getState().modelDrafts[
                modelSettingsKey({ nodeId: 'self', engine: 'llamacpp' }, 'large')
            ]?.change
        ).toBeUndefined()
        await act(async () => button('Confirm remove profile', panel()).click())
        expect(
            useEngineSettingsStore.getState().modelDrafts[
                modelSettingsKey({ nodeId: 'self', engine: 'llamacpp' }, 'large')
            ]?.change
        ).toEqual({
            modelId: 'owner/model:Q4_K_M',
            profileKey: 'large',
            remove: true,
            expectedPolicyRevision: '1'
        })
        expect(setPolicy).not.toHaveBeenCalled()
        expect(window.pairApi.engines.deleteModel).not.toHaveBeenCalled()
    })

    it('keeps peer generation controls locked when raw draft JSON removes the saved lock', async () => {
        getPolicy.mockResolvedValue(
            saved({
                profiles: [
                    { key: identity.model, displayName: 'profile', ...identity, localOnly: true }
                ]
            })
        )
        await render(rowPanel('peer'))
        await act(async () =>
            useNodeConfigurationStore
                .getState()
                .editPolicyDraft('peer', 'Model Manager', saved().policy)
        )
        expect(input('Temperature').disabled).toBe(true)
        expect(button('Remove profile', panel()).disabled).toBe(true)
        expect(panel().querySelector<HTMLInputElement>('input[type="checkbox"]')?.checked).toBe(
            true
        )
        expect(panel().textContent).toContain('Locked on peer')
    })

    it.each(['paused', 'draining'] satisfies NodePolicyDocument['availability'][])(
        'fences remote %s writes while keeping local setup available',
        async availability => {
            getPolicy.mockResolvedValue(saved({}, '1', availability))
            await render(rowPanel('peer'))
            expect(input('Temperature').disabled).toBe(true)
            expect(button('Save', panel()).disabled).toBe(true)
            await render(rowPanel())
            expect(input('Temperature').disabled).toBe(false)
        }
    )

    it('uses current target permission and refreshes a denied grant without losing the draft', async () => {
        await render(rowPanel('peer'))
        await type('Temperature', '0.4')
        const pending = draft('peer').text
        setPolicy.mockRejectedValueOnce(
            new ServiceCallError('permission revoked', REMOTE_MANAGEMENT_DENIED_CODE)
        )
        window.pairApi.cluster.getRemoteManagementForNode = vi
            .fn<typeof window.pairApi.cluster.getRemoteManagementForNode>()
            .mockResolvedValue({ version: 1, revision: '2', allowed: false, state: 'denied' })
        await act(async () => button('Save', panel()).click())
        expect(draft('peer').text).toBe(pending)
        expect(input('Temperature').disabled).toBe(true)
        expect(setPolicy).toHaveBeenCalledTimes(1)
    })

    it('retains invalid Advanced JSON when the Routing panel opens', async () => {
        await render(<RoutingSettings />)
        await editJson('{"profiles":[')
        await act(async () => button('Configure model').click())
        expect(draft().text).toBe('{"profiles":[')
        expect(button('Create profile', panel()).disabled).toBe(true)
        expect(setPolicy).not.toHaveBeenCalled()
        expect(container.querySelector<HTMLTextAreaElement>('textarea')?.value).toBe(
            '{"profiles":['
        )
        expect(button('Save', panel()).disabled).toBe(true)
    })

    it('preserves Routing Advanced JSON and its CAS baseline through reconnect', async () => {
        await render(<RoutingSettings />)
        const pending = '{"profiles":[],"unsent":"keep this draft"}'
        await editJson(pending)
        await act(async () => useConnectionStore.setState({ connected: false }))
        getPolicy.mockResolvedValue(saved({}, '2'))
        await act(async () => useConnectionStore.setState({ connected: true }))
        expect(container.querySelector<HTMLTextAreaElement>('textarea')?.value).toBe(pending)
        expect(draft()).toMatchObject({
            text: pending,
            baseline: { revision: '1' },
            saved: { revision: '2' },
            phase: 'conflict'
        })
        expect(button('Save', container).disabled).toBe(true)
        expect(setPolicy).not.toHaveBeenCalled()
    })

    it('retains a Routing draft and its CAS base when its view is reopened in the same session', async () => {
        await render(<RoutingSettings />)
        const pending = '{"profiles":[],"unsent":"keep across navigation"}'
        await editJson(pending)
        await render(null)
        await render(<RoutingSettings />)

        expect(draft()).toMatchObject({
            owner: 'Routing',
            text: pending,
            baseline: { revision: '1' },
            current: true
        })
        expect(container.querySelector<HTMLTextAreaElement>('textarea')?.value).toBe(pending)
        expect(setPolicy).not.toHaveBeenCalled()
        await act(async () => button('Save', container).click())
        expect(setPolicy).toHaveBeenCalledExactlyOnceWith({
            nodeId: undefined,
            policy: pending,
            expectedPersistedRevision: '1'
        })
    })

    it('subscribes before Routing reads and does not rewind a newer policy push', async () => {
        vi.useFakeTimers()
        try {
            let finish: ((snapshot: NodePolicyDocument) => void) | undefined
            getPolicy.mockImplementationOnce(() => {
                expect(policyListeners.size).toBe(1)
                return new Promise(resolve => {
                    finish = resolve
                })
            })
            await render(<RoutingSettings />)
            const latest = saved({ profiles: [] }, '3')
            await act(async () => push(latest))
            if (!finish) throw new Error('Missing Routing read')
            await act(async () => finish?.(saved({}, '1')))
            expect(draft().baseline?.revision).toBe('3')
            expect(container.querySelector<HTMLTextAreaElement>('textarea')?.value).toBe(
                latest.policy
            )
            getPolicy.mockResolvedValue(latest)
            await act(async () => button('Refresh', container).click())
            await act(async () => {
                await vi.advanceTimersByTimeAsync(2_000)
            })
            expect(draft().baseline?.revision).toBe('3')
            expect(container.querySelector<HTMLTextAreaElement>('textarea')?.value).toBe(
                latest.policy
            )
            expect(button('Pause node', container).disabled).toBe(false)
            expect(button('Save', container).disabled).toBe(false)
        } finally {
            vi.useRealTimers()
        }
    })

    it('requires explicit discard before a model row can replace a dirty Routing baseline', async () => {
        await render(<RoutingSettings />)
        await editJson('{"profiles":[],"unsent":"Routing"}')
        await render(
            <>
                <RoutingSettings />
                {rowPanel()}
            </>
        )
        expect(panel().textContent).toContain('Routing owns unsaved changes')
        expect(draft().owner).toBe('Routing')
        expect(draft().text).toBe('{"profiles":[],"unsent":"Routing"}')
        expect(setPolicy).not.toHaveBeenCalled()
        expect(getPolicy).toHaveBeenCalledTimes(1)
        await act(async () => button('Discard draft and reload', panel()).click())
        expect(draft().owner).toBe('Model Manager')
        expect(draft().text).toBe(saved().policy)
        expect(input('Temperature').disabled).toBe(false)
    })

    it('requires explicit discard before Routing can replace a dirty model-row draft', async () => {
        await render(rowPanel())
        await type('Temperature', '0.4')
        const pending = draft().text
        await render(
            <>
                {rowPanel()}
                <RoutingSettings />
            </>
        )
        expect(draft().text).toBe(pending)
        expect(draft().owner).toBe('Model Manager')
        expect(container.textContent).toContain('Model Manager owns an unsaved policy draft')
        expect(setPolicy).not.toHaveBeenCalled()
        await act(async () => button('Discard Model Manager draft and reload', container).click())
        expect(draft().owner).toBe('Routing')
        expect(draft().text).toBe(saved().policy)
    })

    it('subscribes before fetching and rejects a read older than a policy push', async () => {
        let finish: ((snapshot: NodePolicyDocument) => void) | undefined
        getPolicy.mockImplementationOnce(() => {
            expect(policyListeners.size).toBe(1)
            return new Promise(resolve => {
                finish = resolve
            })
        })
        await render(rowPanel())
        await act(async () =>
            push(
                saved(
                    {
                        profiles: [
                            {
                                key: identity.model,
                                displayName: 'profile',
                                ...identity,
                                requestOptions: { temperature: 0.8 }
                            }
                        ]
                    },
                    '3'
                )
            )
        )
        if (!finish) throw new Error('Missing pending policy read')
        await act(async () => finish?.(saved({}, '1')))
        expect(draft().baseline?.revision).toBe('3')
        expect(input('Temperature').value).toBe('0.8')
    })

    it('rejects a late read after the model-row target changes', async () => {
        let finish: ((snapshot: NodePolicyDocument) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        await render(rowPanel('old-peer'))
        await render(rowPanel('new-peer'))
        if (!finish) throw new Error('Missing old target read')
        await act(async () => finish?.(saved({ profiles: [] }, '99')))
        expect(draft('new-peer').baseline?.revision).toBe('1')
        expect(draft('old-peer').baseline).toBeUndefined()
        expect(panel().textContent).toContain('Node: new-peer')
    })

    it('retains a dirty session when switching away from its target and back', async () => {
        await render(rowPanel('first'))
        await type('Temperature', '0.7')
        const pending = draft('first').text
        await render(rowPanel('second'))
        await render(rowPanel('first'))
        expect(draft('first').text).toBe(pending)
        expect(input('Temperature').value).toBe('0.7')
        expect(setPolicy).not.toHaveBeenCalled()
    })

    it('keeps a dirty draft across reconnect and requires reload after a newer baseline', async () => {
        await render(rowPanel())
        await type('Temperature', '0.3')
        const pending = draft().text
        await act(async () => useConnectionStore.setState({ connected: false }))
        expect(button('Save', panel()).disabled).toBe(true)
        getPolicy.mockResolvedValue(saved({}, '2'))
        await act(async () => useConnectionStore.setState({ connected: true }))
        expect(draft().text).toBe(pending)
        expect(draft().baseline?.revision).toBe('1')
        expect(draft().saved?.revision).toBe('2')
        expect(button('Save', panel()).disabled).toBe(true)
        await act(async () => button('Discard draft and reload', panel()).click())
        expect(draft().baseline?.revision).toBe('2')
        expect(input('Temperature').value).toBe('')
    })

    it('retains Pause received before the first target policy read completes', async () => {
        let finish: ((snapshot: NodePolicyDocument) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        await render(rowPanel('peer'))
        await act(async () => {
            for (const listener of availabilityListeners)
                listener({ nodeId: 'peer', availability: 'paused' })
        })
        if (!finish) throw new Error('Missing initial read')
        await act(async () => finish?.(saved()))
        expect(draft('peer').saved?.availability).toBe('paused')
        expect(button('Save', panel()).disabled).toBe(true)
        expect(panel().textContent).toContain('This node is paused')
    })

    it('keeps a newer Pause observation when an older same-revision read finishes', async () => {
        await render(rowPanel('peer'))
        let finish: ((snapshot: NodePolicyDocument) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        await act(async () => {
            void useNodeConfigurationStore
                .getState()
                .readPolicyDraft('peer', 'Model Manager', 'peer')
        })
        await act(async () => {
            for (const listener of availabilityListeners)
                listener({ nodeId: 'peer', availability: 'paused' })
        })
        if (!finish) throw new Error('Missing pending refresh')
        await act(async () => finish?.(saved()))
        expect(draft('peer').saved?.availability).toBe('paused')
        expect(button('Save', panel()).disabled).toBe(true)
    })

    it('reports a saved-lock refusal without retrying or erasing edits', async () => {
        await render(rowPanel('peer'))
        await type('Temperature', '0.4')
        const pending = draft('peer').text
        setPolicy.mockRejectedValueOnce(new ServiceCallError('locked', REMOTE_SETTING_LOCKED_CODE))
        await act(async () => button('Save', panel()).click())
        expect(draft('peer').text).toBe(pending)
        expect(panel().textContent).toContain('Generation settings are locked on the target')
        expect(setPolicy).toHaveBeenCalledTimes(1)
    })

    it('reads back a lost save reply and keeps its dirty draft for explicit review', async () => {
        await render(rowPanel())
        await type('Temperature', '0.2')
        const pending = draft().text
        setPolicy.mockRejectedValueOnce(new Error('reply lost'))
        getPolicy.mockResolvedValue({ ...saved({}, '2'), policy: pending })
        await act(async () => button('Save', panel()).click())
        expect(draft().phase).toBe('conflict')
        expect(draft().text).toBe(pending)
        expect(button('Save', panel()).disabled).toBe(true)
        expect(setPolicy).toHaveBeenCalledTimes(1)
    })

    it('does not allow draft takeover during Save or replace a newer saved push with its reply', async () => {
        await render(rowPanel())
        await type('Temperature', '0.2')
        const pending = draft().text
        let finish: ((snapshot: Awaited<ReturnType<INodePolicyApi['set']>>) => void) | undefined
        setPolicy.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        await act(async () => button('Save', panel()).click())
        await act(async () => push(saved({}, '3')))
        expect(draft().phase).toBe('saving')
        expect(button('Discard draft and reload', panel()).disabled).toBe(true)
        expect(useNodeConfigurationStore.getState().claimPolicyDraft('self', 'Routing', true)).toBe(
            false
        )
        if (!finish) throw new Error('Missing pending Save')
        await act(async () => finish?.({ ...saved({}, '2'), policy: pending }))
        expect(draft()).toMatchObject({
            text: pending,
            phase: 'conflict',
            saved: { revision: '3' }
        })
        expect(button('Save', panel()).disabled).toBe(true)
    })

    it('offers Configure on positive download completion without opening or moving focus', async () => {
        await render(
            <>
                <input aria-label="Unrelated work" />
                {manager()}
            </>
        )
        const unrelated = input('Unrelated work')
        unrelated.focus()
        const completed = {
            id: 'download',
            ...identity,
            phase: 'complete',
            blocksReadiness: false
        } satisfies NonNullable<ModelItem['downloadState']>
        await act(async () =>
            useEngineModelsStore.setState({
                models: new Map([
                    [
                        'self:llama-cpp',
                        {
                            nodeId: 'self',
                            engineType: 'llama-cpp',
                            models: [{ ...model(), downloaded: true, downloadState: completed }],
                            downloadStateCurrent: true,
                            downloadState: {
                                version: 1,
                                sourceId: 'manager',
                                revision: '2',
                                recoveryRequired: false,
                                entries: [completed]
                            }
                        }
                    ]
                ])
            })
        )
        expect(button('Download complete · Configure').disabled).toBe(false)
        expect(document.activeElement).toBe(unrelated)
        expect(document.querySelector('dialog[open]')).toBeNull()
        expect(getPolicy).not.toHaveBeenCalled()
        await act(async () => button('Download complete · Configure').click())
        expect(panel().textContent).toContain(identity.model)
        expect(getPolicy).toHaveBeenCalledTimes(1)
    })

    it('provides the selected exact identity and policy to the selective-copy composition seam', async () => {
        const copy = vi.fn<
            (source: {
                identity: typeof identity
                policy: string
                targetId: string
                nodeId?: string
            }) => ReactNode
        >(() => <span>Copy preferences</span>)
        await render(
            <ModelSettingsPanel
                targetId="self"
                nodeName="This PC"
                identity={identity}
                open
                renderCopy={copy}
            />
        )
        expect(copy).toHaveBeenLastCalledWith({
            identity: { ...identity, profileKey: identity.model },
            policy: saved().policy,
            targetId: 'self',
            nodeId: undefined
        })
        expect(panel().textContent).toContain('Copy preferences')
    })
})
