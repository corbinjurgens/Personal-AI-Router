// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
    INVALID_PARAMS_CODE,
    LOCAL_VETO_CODE,
    POLICY_CONFLICT_CODE,
    REMOTE_MANAGEMENT_DENIED_CODE
} from '@/shared/constants/service-errors'
import type { EngineDescriptor, EngineType, ModelItem } from '@/shared/types/engines'
import type { JsonValue } from '@/shared/types/json'
import type { NodePolicyChange, NodePolicyDocument } from '@/shared/types/node-policy'
import type { NodeItem } from '@/shared/types/nodes'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { emptyEngineStatus, engineTypeFromManagerName } from '@/shared/utils/engines'
import { createPairApi, type IClusterApi, type INodePolicyApi } from '@/ui/api/pair-api'
import { ServiceCallError } from '@/ui/api/service-transport'
import RoutingSettings from '@/ui/components/RoutingSettings/RoutingSettings'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { useNodesStore } from '@/ui/stores/nodes.store'

vi.mock('@/ui/components/RoutingSettings/RoleEditor', () => ({ default: () => null }))
vi.mock('@/ui/components/RoutingSettings/BulkNodeConfiguration', () => ({ default: () => null }))
vi.mock('@/ui/components/RoutingSettings/RoutingHistory', () => ({ default: () => null }))

const BASE = '9007199254740993'
const NEXT = '9007199254740994'
const localEngine = engineTypeFromManagerName('fixture-local')
const peerEngine = engineTypeFromManagerName('fixture-peer')
if (!localEngine || !peerEngine) throw new Error('Invalid fixture engine IDs')

const peer: NodeItem = {
    id: 'peer',
    name: 'Peer device',
    status: 'active',
    ipAddress: '127.0.0.1',
    port: 1,
    allIpAddresses: [],
    os: 'Linux',
    topology: {
        cpu: { model: 'Fixture CPU', cores: 4, threads: 4 },
        gpus: [],
        ram: 4096,
        storage: []
    }
}

function descriptor(engineType: EngineType, label: string): EngineDescriptor {
    return {
        engineType,
        family: 'llamacpp',
        label,
        available: true,
        protocols: ['openai-v1'],
        modelNaming: 'exact',
        modelStore: '',
        catalog: '',
        operations: {}
    }
}

function model(name: string): ModelItem {
    return {
        name,
        size: 0,
        downloaded: true,
        status: 'idle',
        parameterSize: '',
        quantization: '',
        family: '',
        digest: '',
        sizeVram: null,
        expiresAt: null,
        expiry: '10m',
        capabilities: []
    }
}

function documentFor(
    revision = BASE,
    extra: { [key: string]: JsonValue } = {}
): NodePolicyDocument {
    return {
        policy: JSON.stringify(
            {
                version: 2,
                revision,
                servedModels: null,
                residentModels: [],
                roles: { weak: { engine: 'ollama', model: 'role-model' } },
                routing: { accounts: [{ id: 'private-account' }], tiers: { private: true } },
                futureField: { enabled: false, values: ['retain', null, 0] },
                ...extra
            },
            null,
            2
        ),
        revision,
        availability: 'available',
        activation: { state: 'active', activeRevision: revision },
        recoveryRequired: false
    }
}

const getPolicy = vi.fn<INodePolicyApi['get']>()
const setPolicy = vi.fn<INodePolicyApi['set']>()
const setAvailability = vi.fn<INodePolicyApi['setAvailability']>()
const permission = vi.fn<IClusterApi['getRemoteManagementForNode']>()
const invoke = vi.fn<ServiceTransport['invoke']>()
let emitPolicy: ((change: NodePolicyChange) => void) | undefined
let root: Root
let container: HTMLDivElement

function button(label: string): HTMLButtonElement {
    const found = Array.from(container.querySelectorAll('button')).find(
        item => item.textContent?.trim() === label || item.getAttribute('aria-label') === label
    )
    if (!found) throw new Error(`Missing button: ${label}`)
    return found
}

function policyEditor(): HTMLTextAreaElement {
    const found = container.querySelector('textarea[aria-label="Node policy JSON"]')
    if (!(found instanceof HTMLTextAreaElement)) throw new Error('Missing policy JSON editor')
    return found
}

function modelInput(index = 1): HTMLInputElement {
    const found = container.querySelector(`input[aria-label="Binding model ${index}"]`)
    if (!(found instanceof HTMLInputElement)) throw new Error(`Missing model input: ${index}`)
    return found
}

function draftObject(): { [key: string]: JsonValue } {
    const value: JsonValue = JSON.parse(policyEditor().value)
    if (value === null || typeof value !== 'object' || Array.isArray(value))
        throw new Error('Policy draft is not an object')
    return value
}

async function type(element: HTMLInputElement | HTMLTextAreaElement, value: string) {
    const prototype =
        element instanceof HTMLInputElement
            ? HTMLInputElement.prototype
            : HTMLTextAreaElement.prototype
    const setter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set
    if (!setter) throw new Error('Missing native input value setter')
    await act(async () => {
        setter.call(element, value)
        element.dispatchEvent(new Event('input', { bubbles: true }))
    })
}

async function click(label: string) {
    await act(async () => button(label).click())
}

function engineSelect(index = 1): HTMLButtonElement {
    const label = Array.from(container.querySelectorAll('label')).find(
        item => item.textContent === `Engine ${index}`
    )
    const trigger = label && document.getElementById(label.htmlFor)
    if (!(trigger instanceof HTMLButtonElement)) throw new Error(`Missing engine select: ${index}`)
    return trigger
}

async function selectEngine(value: string, index = 1) {
    const trigger = engineSelect(index)
    await act(async () => trigger.click())
    const list = document.getElementById(trigger.getAttribute('aria-controls') ?? '')
    const option = list?.querySelector(`button[value="${value}"]`)
    if (!(option instanceof HTMLButtonElement)) throw new Error(`Missing engine option: ${value}`)
    await act(async () => option.click())
}

async function pickNode(name: string) {
    await click('Select node')
    const option = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
        item => item.textContent?.trim() === name
    )
    if (!option) throw new Error(`Missing node: ${name}`)
    await act(async () => option.click())
}

async function mount(saved = documentFor()) {
    getPolicy.mockResolvedValue(saved)
    await act(async () => root.render(<RoutingSettings />))
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => false })
    getPolicy.mockReset()
    setPolicy.mockReset().mockImplementation(async write => ({
        policy: write.policy,
        revision: NEXT,
        activation: { state: 'pending', activeRevision: BASE }
    }))
    setAvailability.mockReset()
    permission
        .mockReset()
        .mockResolvedValue({ version: 1, revision: '1', allowed: true, state: 'allowed' })
    invoke.mockReset().mockRejectedValue(new Error('Unexpected lifecycle or service call'))
    const api = createPairApi({
        connected: true,
        invoke,
        subscribePush: vi.fn<ServiceTransport['subscribePush']>(() => () => {}),
        onConnect: vi.fn<ServiceTransport['onConnect']>(() => () => {}),
        onDisconnect: vi.fn<ServiceTransport['onDisconnect']>(() => () => {}),
        onAuthFailure: vi.fn<ServiceTransport['onAuthFailure']>(() => () => {}),
        destroy: vi.fn<ServiceTransport['destroy']>()
    })
    api.nodePolicy.get = getPolicy
    api.nodePolicy.set = setPolicy
    api.nodePolicy.setAvailability = setAvailability
    api.nodePolicy.onPolicyChanged = callback => {
        emitPolicy = callback
        return () => {
            emitPolicy = undefined
        }
    }
    api.cluster.getRemoteManagementForNode = permission
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    useConnectionStore.setState({ connected: true, selfId: 'self' })
    useNodesStore.setState({ nodes: new Map([['peer', peer]]) })
    // Each case starts a fresh app session; drafts intentionally survive component unmounts.
    useNodeConfigurationStore.setState({
        editors: new Map(),
        management: new Map(),
        policyDrafts: new Map()
    })
    useEngineModelsStore.setState({ models: new Map() })
    useEngineStatusStore.setState({
        descriptors: new Map([
            [localEngine, descriptor(localEngine, 'Local fixture engine')],
            [peerEngine, descriptor(peerEngine, 'Peer fixture engine')]
        ]),
        statusByNode: new Map([
            [
                'self',
                new Map([
                    [
                        localEngine,
                        { ...emptyEngineStatus('self', localEngine), processStatus: 'stopped' }
                    ]
                ])
            ],
            [
                'peer',
                new Map([
                    [
                        peerEngine,
                        { ...emptyEngineStatus('peer', peerEngine), processStatus: 'stopped' }
                    ]
                ])
            ]
        ])
    })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    // All engine/model operations use the real API backed by this transport.
    // A policy edit or save must never reach any of those lifecycle calls.
    expect(invoke).not.toHaveBeenCalled()
    expect(setAvailability).not.toHaveBeenCalled()
    useNodeConfigurationStore.setState({ editors: new Map(), management: new Map() })
    vi.unstubAllGlobals()
})

describe('model engine bindings in the shared policy form', () => {
    it('adds and edits native names in the JSON draft, then saves once with the exact local CAS base', async () => {
        await mount()
        const original = draftObject()
        await click('Add binding')
        await selectEngine('fixture-local')
        await type(modelInput(), 'Native/Stopped-Model.Q4.gguf')
        expect(draftObject()).toEqual({
            ...original,
            engineBindings: [{ engine: 'fixture-local', model: 'Native/Stopped-Model.Q4.gguf' }]
        })
        expect(setPolicy).not.toHaveBeenCalled()
        expect(container.textContent).toContain(
            'Requests for this model can wait while this node activates its assigned engine.'
        )
        expect(container.textContent).toContain('Saving starts nothing.')
        expect(container.querySelectorAll('datalist option')).toHaveLength(0)

        // A later binding edit substitutes into the current JSON, including unsaved private edits.
        await type(
            policyEditor(),
            JSON.stringify({ ...draftObject(), privateDraft: { keep: true } })
        )
        await type(modelInput(), 'Native/Typed-Model')
        const sent = policyEditor().value
        await click('Save')

        expect(setPolicy).toHaveBeenCalledExactlyOnceWith({
            nodeId: undefined,
            policy: sent,
            expectedPersistedRevision: BASE
        })
        expect(draftObject()).toEqual({
            ...original,
            privateDraft: { keep: true },
            engineBindings: [{ engine: 'fixture-local', model: 'Native/Typed-Model' }]
        })
        expect(container.textContent).toContain('Saved, not yet enforced')
    })

    it('removes and clears bindings in the same draft and saves the exact selected peer and revision', async () => {
        await mount()
        const remote = documentFor('42', {
            engineBindings: [
                { engine: 'ollama', model: 'first' },
                { engine: 'fixture-peer', model: 'second' }
            ]
        })
        getPolicy.mockResolvedValue(remote)
        await pickNode('Peer device')
        const original = draftObject()
        await click('Remove binding 1')
        expect(draftObject()).toEqual({
            ...original,
            engineBindings: [{ engine: 'fixture-peer', model: 'second' }]
        })
        expect(modelInput().value).toBe('second')
        await click('Clear bindings')
        expect(draftObject()).toEqual({ ...original, engineBindings: [] })
        expect(setPolicy).not.toHaveBeenCalled()
        const sent = policyEditor().value
        await click('Save')
        expect(setPolicy).toHaveBeenCalledExactlyOnceWith({
            nodeId: 'peer',
            policy: sent,
            expectedPersistedRevision: '42'
        })
    })

    it('uses target engine choices and labels model suggestions as observations from that target and engine', async () => {
        useEngineModelsStore.setState({
            models: new Map([
                [
                    'local',
                    { nodeId: 'self', engineType: peerEngine, models: [model('wrong-node')] }
                ],
                [
                    'peer',
                    { nodeId: 'peer', engineType: peerEngine, models: [model('observed-native')] }
                ],
                [
                    'other-engine',
                    { nodeId: 'peer', engineType: 'ollama', models: [model('wrong-engine')] }
                ]
            ])
        })
        await mount()
        getPolicy.mockResolvedValue(
            documentFor('42', {
                engineBindings: [{ engine: 'fixture-peer', model: 'not-observed' }]
            })
        )
        await pickNode('Peer device')

        const options = Array.from(container.querySelectorAll('datalist option'))
        expect(options.map(option => option.getAttribute('value'))).toEqual(['observed-native'])
        expect(options[0].getAttribute('label')).toBe('Observed on this node')
        expect(container.textContent).toContain(
            'Suggestions are observations from this node, not a complete inventory.'
        )
        const trigger = engineSelect()
        expect(trigger.textContent).toContain('Peer fixture engine')
        await act(async () => trigger.click())
        const list = document.getElementById(trigger.getAttribute('aria-controls') ?? '')
        expect(list?.querySelector('button[value="fixture-peer"]')?.textContent).toContain(
            'Peer fixture engine'
        )
        expect(list?.querySelector('button[value="fixture-local"]')).toBeNull()
        await act(async () => trigger.click())
        await type(modelInput(), 'Typed/Despite-Incomplete-Inventory')
        expect(draftObject().engineBindings).toEqual([
            { engine: 'fixture-peer', model: 'Typed/Despite-Incomplete-Inventory' }
        ])
        expect(setPolicy).not.toHaveBeenCalled()
    })

    it('keeps malformed JSON byte-for-byte and shows the existing save error without reconstructing defaults', async () => {
        await mount()
        const malformed = '  { "engineBindings": [\n  '
        await type(policyEditor(), malformed)
        expect(button('Add binding').disabled).toBe(true)
        expect(button('Clear bindings').disabled).toBe(true)
        await click('Add binding')
        await click('Clear bindings')
        setPolicy.mockRejectedValueOnce(
            new ServiceCallError('Node policy must be valid JSON.', INVALID_PARAMS_CODE)
        )
        await click('Save')

        expect(policyEditor().value).toBe(malformed)
        expect(setPolicy).toHaveBeenCalledExactlyOnceWith({
            nodeId: undefined,
            policy: malformed,
            expectedPersistedRevision: BASE
        })
        expect(container.textContent).toContain('Node policy must be valid JSON.')
        expect(container.textContent).toContain('Your policy text is kept.')
    })

    it.each([
        { name: 'non-object policy', text: '[]' },
        { name: 'non-list bindings', text: '{"engineBindings":null,"keep":true}' },
        { name: 'malformed binding row', text: '{"engineBindings":[{"model":"m"}],"keep":true}' }
    ])('preserves $name instead of replacing it with an empty list', async ({ text }) => {
        await mount()
        await type(policyEditor(), text)
        await click('Add binding')
        await click('Clear bindings')
        expect(policyEditor().value).toBe(text)
        expect(button('Add binding').disabled).toBe(true)
        expect(setPolicy).not.toHaveBeenCalled()
    })

    it('clears the old target draft and sends nothing while the new target read is pending or fails', async () => {
        await mount(
            documentFor(BASE, {
                engineBindings: [{ engine: 'fixture-local', model: 'old-target' }]
            })
        )
        await type(modelInput(), 'old-unsent-draft')
        let rejectRead: ((error: Error) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise((_, reject) => {
                    rejectRead = reject
                })
        )
        await pickNode('Peer device')

        expect(policyEditor().value).toBe('')
        expect(button('Save').disabled).toBe(true)
        expect(button('Add binding').disabled).toBe(true)
        await click('Save')
        if (!rejectRead) throw new Error('Missing pending peer policy read')
        await act(async () => rejectRead?.(new Error('Peer policy unreachable')))
        await click('Save')
        expect(setPolicy).not.toHaveBeenCalled()
        expect(getPolicy).toHaveBeenLastCalledWith('peer')
        expect(policyEditor().value).toBe('')
        expect(container.textContent).toContain('Peer policy unreachable')
    })

    it('ignores an obsolete target read after selecting another node', async () => {
        await mount(
            documentFor(BASE, {
                engineBindings: [{ engine: 'fixture-local', model: 'old-target' }]
            })
        )
        let finishRead: ((value: NodePolicyDocument) => void) | undefined
        getPolicy.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finishRead = resolve
                })
        )
        await click('Reload')
        getPolicy.mockResolvedValue(
            documentFor('42', { engineBindings: [{ engine: 'fixture-peer', model: 'peer-draft' }] })
        )
        await pickNode('Peer device')
        if (!finishRead) throw new Error('Missing old target policy read')
        await act(async () =>
            finishRead?.(
                documentFor(NEXT, {
                    engineBindings: [{ engine: 'fixture-local', model: 'obsolete' }]
                })
            )
        )
        expect(modelInput().value).toBe('peer-draft')
        expect(setPolicy).not.toHaveBeenCalled()
    })

    it('keeps an unsupported-target refusal explicit and retains the unsaved binding', async () => {
        await mount(documentFor(BASE, { engineBindings: [] }))
        await click('Add binding')
        await selectEngine('ollama')
        await type(modelInput(), 'typed-native')
        const sent = policyEditor().value
        setPolicy.mockRejectedValueOnce(
            new Error(
                'target does not preserve engineBindings; model-control edits are unsupported'
            )
        )
        await click('Save')

        expect(container.textContent).toContain(
            'target does not preserve engineBindings; model-control edits are unsupported'
        )
        expect(policyEditor().value).toBe(sent)
        expect(getPolicy).toHaveBeenCalledTimes(2)
        expect(setPolicy).toHaveBeenCalledTimes(1)
    })

    it('shows an unknown save outcome after a lost reply and failed readback without retrying or discarding the draft', async () => {
        await mount(documentFor(BASE, { engineBindings: [{ engine: 'ollama', model: 'before' }] }))
        await type(modelInput(), 'unsure-native')
        const sent = policyEditor().value
        setPolicy.mockRejectedValueOnce(new Error('Save reply lost'))
        getPolicy.mockRejectedValueOnce(new Error('Readback disconnected'))
        await click('Save')

        expect(container.textContent).toContain(
            'Could not confirm whether the save took effect: Save reply lost'
        )
        expect(policyEditor().value).toBe(sent)
        expect(setPolicy).toHaveBeenCalledTimes(1)
        expect(getPolicy).toHaveBeenCalledTimes(2)
    })

    it('retains a dirty binding across a newer push and uses its original CAS base, then exposes conflict readback', async () => {
        await mount(documentFor(BASE, { engineBindings: [{ engine: 'ollama', model: 'before' }] }))
        await type(modelInput(), 'unsent-native')
        const sent = policyEditor().value
        const latest = documentFor(NEXT, {
            engineBindings: [{ engine: 'ollama', model: 'newly-saved' }]
        })
        if (!emitPolicy) throw new Error('Missing policy push listener')
        await act(async () => emitPolicy?.({ nodeId: 'self', ...latest }))
        expect(policyEditor().value).toBe(sent)
        getPolicy.mockResolvedValueOnce(latest)
        setPolicy.mockRejectedValueOnce(
            new ServiceCallError('Policy revision conflict', POLICY_CONFLICT_CODE)
        )
        await click('Save')

        expect(setPolicy).toHaveBeenCalledExactlyOnceWith({
            nodeId: undefined,
            policy: sent,
            expectedPersistedRevision: BASE
        })
        expect(button('Save').disabled).toBe(true)
        expect(policyEditor().value).toBe(sent)
        await click('Review latest saved policy')
        const readback = container.querySelector(
            'textarea[aria-label="Latest saved node policy JSON"]'
        )
        if (!(readback instanceof HTMLTextAreaElement)) throw new Error('Missing conflict readback')
        expect(readback.value).toBe(latest.policy)
        await click('Save')
        expect(setPolicy).toHaveBeenCalledTimes(1)
    })

    it.each([
        {
            name: 'management denied',
            code: REMOTE_MANAGEMENT_DENIED_CODE,
            detail: 'Permission changed on target',
            message: 'This device refused remote management permission.'
        },
        {
            name: 'pause veto',
            code: LOCAL_VETO_CODE,
            detail: 'Resume it on this node before managing it remotely.',
            message: 'the target refused the operation'
        }
    ])(
        'retains the binding draft after $name without bypassing the target refusal',
        async ({ code, detail, message }) => {
            await mount(
                documentFor(BASE, { engineBindings: [{ engine: 'ollama', model: 'before' }] })
            )
            await type(modelInput(), 'unsent-native')
            const sent = policyEditor().value
            setPolicy.mockRejectedValueOnce(new ServiceCallError(detail, code))
            await click('Save')
            expect(container.textContent).toContain(message)
            expect(policyEditor().value).toBe(sent)
            expect(setPolicy).toHaveBeenCalledTimes(1)
            expect(getPolicy).toHaveBeenCalledTimes(1)
        }
    )

    it('keeps paused peers read-only with no remote unlock or Resume action enabled', async () => {
        await mount()
        getPolicy.mockResolvedValue({
            ...documentFor('42', {
                engineBindings: [{ engine: 'ollama', model: 'paused-native' }]
            }),
            availability: 'paused'
        })
        await pickNode('Peer device')
        expect(modelInput().disabled).toBe(true)
        expect(button('Add binding').disabled).toBe(true)
        expect(button('Save').disabled).toBe(true)
        expect(container.querySelector('button[aria-label="Resume node: Peer device"]')).toBeNull()
        expect(container.querySelector('button[aria-label*="Unlock"]')).toBeNull()
        await click('Save')
        expect(setPolicy).not.toHaveBeenCalled()
    })
})
