// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act, type ComponentProps } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { EngineModels, ModelIdentity, ModelItem } from '@/shared/types/engines'
import type { EngineSettingsSnapshot } from '@/shared/types/engine-settings'
import type { InspectModelResult } from '@/shared/types/model-config'
import type { ModelCopyOutcome } from '@/shared/types/model-copy'
import type { NodeItem } from '@/shared/types/nodes'
import type { NodePolicyDocument } from '@/shared/types/node-policy'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi, type INodePolicyApi } from '@/ui/api/pair-api'
import type { IEngineApi } from '@/ui/api/engine-api'
import { ServiceCallError } from '@/ui/api/service-transport'
import ModelSettingsCopy from '@/ui/components/ModelSettings/ModelSettingsCopy'
import { ModelCopyModal } from '@/ui/components/ModelManager/ModelCopyModal'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { useEngineSettingsStore } from '@/ui/stores/engine-settings.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'

vi.mock('@/ui/components/ModelManager/ModelCopyModal', () => ({
    ModelCopyModal: (props: ComponentProps<typeof ModelCopyModal>) => {
        copyOwners.set(props.fixedDestination ?? '', props)
        return props.open ? (
            <button onClick={() => props.onOpenChange(false)}>
                Close copy on {props.fixedDestination}
            </button>
        ) : null
    }
}))

const copyOwners = new Map<string, ComponentProps<typeof ModelCopyModal>>()
const identity = { engine: 'llamacpp', model: 'source-artifact', profileKey: 'coder-local' }
const verified: ModelIdentity = {
    strength: 'verified',
    contentId: 'b'.repeat(64),
    repository: 'owner/model',
    quantization: 'Q4_K_M',
    totalBytes: 100,
    files: [{ role: 'companion', name: 'projector.gguf', sizeBytes: 10, sha256: 'a'.repeat(64) }]
}
const source = JSON.stringify({
    profiles: [
        {
            key: 'coder-local',
            displayName: 'Source name',
            icon: 'code',
            engine: 'llamacpp',
            model: 'coder-local',
            modelIdentity: verified,
            requestOptions: { temperature: 0.25 }
        }
    ]
})
const get = vi.fn<INodePolicyApi['get']>()
const set = vi.fn<INodePolicyApi['set']>()
const getSettings = vi.fn<IEngineApi['getSettings']>()
const inspect = vi.fn<IEngineApi['inspectModel']>()
const preview = vi.fn<IEngineApi['previewSettings']>()
const apply = vi.fn<IEngineApi['applySettings']>()
const initial = vi.fn<IEngineApi['getInitialState']>()
const configure = vi.fn<ComponentProps<typeof ModelSettingsCopy>['onConfigureTarget']>()
let root: Root
let container: HTMLDivElement
let inventories: EngineModels[]
let outcomes: Map<string, EngineSettingsSnapshot>

function model(name = 'target-artifact', metadata: ModelIdentity = verified): ModelItem {
    return {
        name,
        identity: metadata,
        size: 100,
        downloaded: true,
        status: 'idle',
        parameterSize: '',
        family: '',
        sizeVram: null,
        expiresAt: null,
        expiry: '0',
        capabilities: []
    }
}
function inventory(nodeId: string, models = [model()]): EngineModels {
    return { nodeId, engineType: 'llama-cpp', models }
}
function node(id: string): NodeItem {
    return {
        id,
        name: id,
        status: 'active',
        ipAddress: '127.0.0.1',
        port: 1,
        allIpAddresses: [],
        os: 'Windows',
        topology: { cpu: { model: '', cores: 1, threads: 1 }, ram: 0, gpus: [], storage: [] }
    }
}
function policy(
    overrides: {
        revision?: string
        profiles?: object[]
        roles?: object[]
        availability?: 'paused'
    } = {}
): NodePolicyDocument {
    return {
        policy: JSON.stringify({
            profiles: overrides.profiles ?? [
                {
                    key: 'coder-local',
                    displayName: 'Target name',
                    icon: 'cube',
                    engine: 'llamacpp',
                    model: 'coder-local',
                    requestOptions: { temperature: 1, seed: 42 },
                    requestOptionsMode: 'enforced'
                }
            ],
            roles: overrides.roles ?? []
        }),
        revision: overrides.revision ?? '1',
        activation: null,
        availability: overrides.availability ?? 'available',
        recoveryRequired: false
    }
}
function settings(nodeId: string): EngineSettingsSnapshot {
    return {
        nodeId,
        engine: 'llamacpp',
        revision: 5,
        appliedRevision: 5,
        epoch: 'epoch',
        sequence: 1,
        settings: {
            serverPort: nodeId === 'self' ? 1 : 77,
            proxyPort: 2,
            launchText: nodeId === 'self' ? 'source paths' : 'target launch',
            modelLoads: [
                {
                    profileKey: 'coder-local',
                    modelId: nodeId === 'self' ? 'source-artifact' : 'target-artifact',
                    requested:
                        nodeId === 'self'
                            ? {
                                  slots: 2,
                                  fit: 'on',
                                  fitTarget: 'source-device',
                                  extraOptions: { portable: 'yes', device: 'source-native' }
                              }
                            : { slots: 1 }
                }
            ]
        },
        effectiveServerPort: 77,
        effectiveProxyPort: 2,
        running: false,
        adopted: false,
        editable: true,
        reason: '',
        format: '',
        phase: 'idle',
        error: '',
        requestId: ''
    }
}
function inspection(nodeId: string, modelId: string): InspectModelResult {
    return {
        nodeId,
        engine: 'llamacpp',
        modelId,
        status: 'resolved',
        source: `${nodeId} resolver`,
        estimate: { status: 'unknown' },
        artifacts: [
            {
                role: 'model',
                identity: `${nodeId}-local-binding`,
                size: '100',
                assurance: 'local binding'
            }
        ],
        options: [
            { field: 'slots', label: 'Slots', kind: 'integer', portable: true, supported: true },
            {
                field: 'fit',
                label: 'Native fit',
                kind: 'enum',
                values: ['on', 'off'],
                portable: true,
                supported: true
            },
            {
                field: 'fitTarget',
                label: 'Fit target',
                kind: 'string',
                portable: false,
                supported: true
            },
            {
                field: 'extraOptions',
                label: 'Extra options',
                kind: 'options',
                portable: true,
                supported: true,
                nonPortableKeys: ['device']
            }
        ]
    }
}
function group(id = 'peer'): Element {
    const found = container.querySelector(`[role="group"][aria-label="Deploy ${id}"]`)
    if (!found) throw new Error(`Missing target ${id}`)
    return found
}
function button(label: string, scope: ParentNode = container): HTMLButtonElement {
    const found = Array.from(scope.querySelectorAll('button')).find(
        item => item.textContent === label
    )
    if (!found) throw new Error(`Missing button ${label}`)
    return found
}
async function click(label: string, scope: ParentNode = container) {
    await act(async () => button(label, scope).click())
}
async function check(label: string) {
    const found = Array.from(container.querySelectorAll('label')).find(item =>
        item.textContent?.includes(label)
    )?.control
    if (!(found instanceof HTMLInputElement)) throw new Error(`Missing checkbox ${label}`)
    await act(async () => found.click())
}
async function input(label: string, value: string) {
    const field = container.querySelector(`input[aria-label="${label}"]`)
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
    if (!(field instanceof HTMLInputElement) || !setter) throw new Error(`Missing input ${label}`)
    await act(async () => {
        setter.call(field, value)
        field.dispatchEvent(new Event('input', { bubbles: true }))
    })
}
async function render(text = source, disabled = false) {
    await act(async () =>
        root.render(
            <ModelSettingsCopy
                identity={identity}
                policy={text}
                targetId="self"
                disabled={disabled}
                onConfigureTarget={configure}
            />
        )
    )
}
async function reviewTarget(id = 'peer') {
    await check(`Deploy to ${id}`)
    await click(`Review deploy to ${id}`, group(id))
}
function change(id = 'peer') {
    const request = preview.mock.calls.find(([request]) => request.nodeId === id)?.[0]
    if (!request?.modelChange) throw new Error(`Missing model change for ${id}`)
    return request.modelChange
}
function deployment(id = 'peer') {
    return Array.from(usePendingActionsStore.getState().deployments.values()).findLast(
        row => row.nodeId === id
    )
}
async function copyOutcome(
    id: string,
    disposition: ModelCopyOutcome['disposition'],
    modelId = 'copied-target-name'
) {
    const owner = copyOwners.get(id)
    if (!owner?.onOutcome) throw new Error(`Missing copy owner ${id}`)
    await act(async () =>
        owner.onOutcome?.({
            opId: `copy-${id}`,
            sourceNodeId: owner.initialSource,
            destinationNodeId: id,
            engine: 'llamacpp',
            model: owner.initialModel,
            disposition,
            ...(disposition === 'copied'
                ? {
                      result: {
                          engine: 'llamacpp',
                          model: modelId,
                          files: 1,
                          filesSkipped: 0,
                          bytesTotal: 100,
                          bytesCopied: 100,
                          layout: 'llama-artifacts-v1',
                          durability: 'confirmed'
                      }
                  }
                : {}),
            reason: disposition
        })
    )
    // The modal callback is intentionally void; drain its owner work without a timer.
    await act(async () => {})
}
beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    inventories = [inventory('self', [model('source-artifact')]), inventory('peer')]
    outcomes = new Map()
    copyOwners.clear()
    get.mockReset().mockResolvedValue(policy())
    set.mockReset()
    getSettings
        .mockReset()
        .mockImplementation(async target => outcomes.get(target.nodeId) ?? settings(target.nodeId))
    inspect
        .mockReset()
        .mockImplementation(async request => inspection(request.nodeId ?? 'self', request.modelId))
    preview.mockReset().mockImplementation(async request => ({
        settings: request.settings,
        revision: 5,
        policyRevision: '1',
        previewDigest: 'd'.repeat(64),
        errors: {},
        restart: true,
        rebind: false,
        diff: [{ field: 'slots', before: 1, after: 2 }]
    }))
    apply.mockReset().mockImplementation(async request => {
        outcomes.set(request.nodeId, {
            ...settings(request.nodeId),
            requestId: request.requestId ?? '',
            revision: 6,
            sequence: 2,
            phase: 'succeeded',
            policyRevision: '2'
        })
        return { revision: 6, phase: 'succeeded' }
    })
    initial.mockReset().mockImplementation(async () => ({
        descriptors: [],
        statuses: [],
        models: inventories,
        activeProgress: [],
        updateAvailable: []
    }))
    configure.mockReset()
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
    api.nodePolicy.get = get
    api.nodePolicy.set = set
    api.engines.getSettings = getSettings
    api.engines.inspectModel = inspect
    api.engines.previewSettings = preview
    api.engines.applySettings = apply
    api.engines.getInitialState = initial
    api.cluster.getRemoteManagementForNode = vi
        .fn<typeof api.cluster.getRemoteManagementForNode>()
        .mockResolvedValue({ version: 1, revision: '1', state: 'allowed', allowed: true })
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    useConnectionStore.setState({ connected: true, selfId: 'self' })
    useEngineSettingsStore.setState({ entries: {}, modelDrafts: {} })
    usePendingActionsStore.setState({ deployments: new Map() })
    useEngineModelsStore.setState({
        models: new Map(inventories.map(row => [`${row.nodeId}:${row.engineType}`, row]))
    })
    useClusterInvitationsStore.setState({ remoteManagementContext: 1 })
    useNodesStore.setState({ nodes: new Map([['peer', node('peer')]]) })
    useNodeConfigurationStore.setState({
        management: new Map(),
        policyDrafts: new Map(),
        editors: new Map()
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

describe('profile deployment correspondence', () => {
    it('skips copy only for verified main and companion identity and uses the actual local artifact name', async () => {
        await render()
        await reviewTarget()
        expect(copyOwners.size).toBe(0)
        expect(change()).toMatchObject({
            modelId: 'target-artifact',
            profileKey: 'coder-local',
            expectedPolicyRevision: '1'
        })
        expect(group().textContent).toContain('Target model: target-artifact')
        expect(group().textContent).toContain('Review profile update')
        expect(apply).not.toHaveBeenCalled()
        await click('Confirm deploy to peer')
        expect(apply, JSON.stringify(deployment())).toHaveBeenCalledTimes(1)
        expect(deployment()?.disposition).toBe('applied')
        expect(set).not.toHaveBeenCalled()
    })
    it('requires an unselected explicit choice when the projector is unverified', async () => {
        inventories[1] = inventory('peer', [model('target-artifact', { ...verified, files: [] })])
        await render()
        await reviewTarget()
        expect(group().textContent).toContain('Model file matches; projector.gguf not verified')
        expect(button('Use the files on peer').disabled).toBe(false)
        expect(button('Copy').disabled).toBe(false)
        expect(preview).not.toHaveBeenCalled()
        expect(copyOwners.size).toBe(0)
        await click('Use the files on peer')
        expect(change().modelId).toBe('target-artifact')
    })
    it('requires an unselected explicit choice for repository, quantization and size alone', async () => {
        inventories[1] = inventory('peer', [
            model('weak-local-name', {
                strength: 'declared',
                repository: 'owner/model',
                quantization: 'Q4_K_M',
                totalBytes: 100
            })
        ])
        await render()
        await reviewTarget()
        expect(group().textContent).toContain('Possibly the same file (not verified)')
        expect(preview).not.toHaveBeenCalled()
        expect(copyOwners.size).toBe(0)
        await click('Use the file on peer')
        expect(group().textContent).toContain('Key clash')
        await input('Profile key peer', 'new-key')
        await click('Review again', group())
        await click('Use the file on peer')
        expect(change().modelId).toBe('weak-local-name')
    })
    it('copies missing files through the pinned owner before reviewing its returned model name', async () => {
        inventories[1] = inventory('peer', [])
        get.mockResolvedValue(policy({ profiles: [] }))
        await render()
        await reviewTarget()
        expect(preview).not.toHaveBeenCalled()
        expect(copyOwners.get('peer')).toMatchObject({
            initialSource: 'self',
            initialModel: 'source-artifact',
            fixedDestination: 'peer'
        })
        await copyOutcome('peer', 'copied')
        expect(change().modelId).toBe('copied-target-name')
        expect(change().policy).toMatchObject({ displayName: 'Source name', icon: 'code' })
        expect(inspect.mock.calls.find(([request]) => request.nodeId === 'self')?.[0]).toEqual({
            nodeId: 'self',
            engine: 'llamacpp',
            modelId: 'source-artifact',
            profileKey: 'coder-local'
        })
        expect(apply).not.toHaveBeenCalled()
    })
    it('does not treat closing a copy window as canonical cancellation', async () => {
        inventories[1] = inventory('peer', [])
        await render()
        await reviewTarget()
        await click('Close copy on peer')
        expect(deployment()).toMatchObject({ stage: 'copy' })
        expect(deployment()?.disposition).toBeUndefined()
        expect(button('Review again', group()).disabled).toBe(true)
        await copyOutcome('peer', 'cancelled')
        expect(deployment()?.disposition).toBe('cancelled')
        expect(button('Review again', group()).disabled).toBe(false)
    })
    it('retains unknown copy correlation until a verified canonical completion arrives', async () => {
        inventories[1] = inventory('peer', [])
        get.mockResolvedValue(policy({ profiles: [] }))
        await render()
        await reviewTarget()
        await copyOutcome('peer', 'unknown')
        expect(deployment()?.disposition).toBe('unknown')
        expect(button('Review again', group()).disabled).toBe(true)
        expect(preview).not.toHaveBeenCalled()
        await copyOutcome('peer', 'copied')
        expect(change().modelId).toBe('copied-target-name')
        expect(deployment()).toMatchObject({ stage: 'review', disposition: undefined })
    })
    it('shows a canonical copy refusal without starting a settings write', async () => {
        inventories[1] = inventory('peer', [])
        await render()
        await reviewTarget()
        await copyOutcome('peer', 'refused')
        expect(deployment()).toMatchObject({
            stage: 'copy',
            disposition: 'refused',
            copyOpId: 'copy-peer'
        })
        expect(preview).not.toHaveBeenCalled()
        expect(apply).not.toHaveBeenCalled()
    })
    it('asks for a replacement key when the typed artifact belongs to a different model', async () => {
        getSettings.mockImplementation(async target => {
            const value = settings(target.nodeId)
            if (target.nodeId === 'peer')
                value.settings.modelLoads = [
                    { profileKey: 'coder-local', modelId: 'other-artifact', requested: {} }
                ]
            return value
        })
        await render()
        await reviewTarget()
        expect(group().textContent).toContain('Key clash')
        expect(preview).not.toHaveBeenCalled()
        await input('Profile key peer', 'coder-api')
        await click('Review again', group())
        expect(change().profileKey).toBe('coder-api')
        expect(change().modelId).toBe('target-artifact')
    })
    it('compares an untyped profile Model instead of treating its key as the artifact', async () => {
        get.mockResolvedValue(
            policy({
                profiles: [
                    {
                        key: 'coder-local',
                        engine: 'llamacpp',
                        model: 'target-artifact',
                        displayName: 'Native target'
                    }
                ]
            })
        )
        getSettings.mockImplementation(async target => ({
            ...settings(target.nodeId),
            settings: {
                ...settings(target.nodeId).settings,
                ...(target.nodeId === 'peer' ? { modelLoads: [] } : {})
            }
        }))
        await render()
        await reviewTarget()
        expect(change().modelId).toBe('target-artifact')
        expect(group().textContent).toContain('Review profile update')
    })
})

describe('selected portable profile payload', () => {
    it('creates a missing typed profile with inherited settings when no load fields are selected', async () => {
        get.mockResolvedValue(policy({ profiles: [] }))
        getSettings.mockImplementation(async target => ({
            ...settings(target.nodeId),
            settings: { ...settings(target.nodeId).settings, modelLoads: [] }
        }))
        await render()
        await reviewTarget()
        expect(change().load).toEqual({ set: {}, clear: ['context'] })
        expect(change().modelId).toBe('target-artifact')
        expect(change().policy).toMatchObject({ displayName: 'Source name', icon: 'code' })
        expect(inspect.mock.calls.find(([request]) => request.nodeId === 'self')?.[0]).toEqual({
            nodeId: 'self',
            engine: 'llamacpp',
            modelId: 'source-artifact'
        })
    })

    it('retains target metadata and mode while sending generation through the coupled owner', async () => {
        await render()
        await reviewTarget()
        expect(change().policy).toEqual({ generation: { temperature: 0.25 } })
        expect(change().load).toEqual({
            set: { slots: 2, fit: 'on', extraOptions: { portable: 'yes' } },
            clear: []
        })
        expect(change()).not.toHaveProperty('resolved')
        expect(change().load?.set).not.toHaveProperty('fitTarget')
        expect(change().policy).not.toHaveProperty('binding')
        expect(change().policy).not.toHaveProperty('resident')
        expect(change()).not.toHaveProperty('previewDigest')
        await click('Confirm deploy to peer')
        const request = apply.mock.calls[0]?.[0]
        expect(request?.settings).toMatchObject({ serverPort: 77, launchText: 'target launch' })
        expect(request?.modelChange?.previewDigest).toBe('d'.repeat(64))
    })
    it('carries explicit generation clears and an explicitly selected request mode', async () => {
        await render()
        await check('Seed: Clear on target')
        await check('Copy request precedence')
        await reviewTarget()
        expect(change().policy).toEqual({
            generation: { temperature: 0.25, seed: null },
            requestOptionsMode: 'defaults'
        })
    })
    it('uses source display defaults only for creation and respects explicit metadata edits', async () => {
        get.mockResolvedValue(policy({ profiles: [] }))
        await render()
        await check('Deploy to peer')
        await input('Display name peer', 'Chosen name')
        await input('Icon peer', 'image')
        await click('Review deploy to peer', group())
        expect(change().policy).toMatchObject({ displayName: 'Chosen name', icon: 'image' })
    })
    it('lists only target roles in declared order and patches selected membership alone', async () => {
        get.mockResolvedValue(
            policy({
                roles: [
                    { name: 'strong', profiles: [] },
                    { name: 'weak', profiles: [] }
                ]
            })
        )
        await render()
        await reviewTarget()
        const labels = Array.from(group().querySelectorAll('label'))
            .filter(
                item => item.control instanceof HTMLInputElement && item.control.type === 'checkbox'
            )
            .map(item => item.textContent)
        expect(labels.filter(label => label?.includes(' on peer'))).toEqual([
            'strong on peer',
            'weak on peer'
        ])
        expect(group().textContent).not.toContain('medium on peer')
        await check('weak on peer')
        await click('Review again', group())
        const last = preview.mock.calls.at(-1)?.[0].modelChange
        expect(last?.policy?.roles).toEqual({ weak: true })
    })
    it('shows no-role guidance and sends no role patch', async () => {
        await render()
        await reviewTarget()
        expect(group().textContent).toContain('No roles on peer; create one in node settings.')
        expect(change().policy).not.toHaveProperty('roles')
    })
    it('does not reset an explicit tuning deselection after source inspection refresh', async () => {
        await render()
        await check('Copy load slots')
        await check('Deploy to peer')
        await act(async () =>
            useEngineSettingsStore
                .getState()
                .inspectModel(
                    { nodeId: 'self', engine: 'llamacpp' },
                    'source-artifact',
                    'coder-local'
                )
        )
        await click('Review deploy to peer', group())
        expect(change().load?.set).not.toHaveProperty('slots')
        expect(change().load?.set.fit).toBe('on')
    })
    it('keeps a completed target review through an unrelated source advisory estimate', async () => {
        await render()
        await reviewTarget()
        await act(async () =>
            useEngineSettingsStore.getState().previewModel(
                { nodeId: 'self', engine: 'llamacpp' },
                {
                    modelId: 'source-artifact',
                    profileKey: 'coder-local',
                    expectedPolicyRevision: '1',
                    load: { set: { slots: 2 } }
                },
                'source advisory',
                true
            )
        )
        expect(button('Confirm deploy to peer').disabled).toBe(false)
        expect(apply).not.toHaveBeenCalled()
    })
    it('withdraws confirmation when the target settings revision changes', async () => {
        await render()
        await reviewTarget()
        await act(async () =>
            useEngineSettingsStore
                .getState()
                .receive({ ...settings('peer'), revision: 6, sequence: 2 })
        )
        expect(button('Confirm deploy to peer').disabled).toBe(true)
        expect(
            useEngineSettingsStore.getState().modelDrafts['peer:llamacpp:coder-local']?.review
        ).toBeUndefined()
        expect(apply).not.toHaveBeenCalled()
    })
    it('reviews Auto independently on each target and sends only that target reviewed digest', async () => {
        useNodesStore.setState({
            nodes: new Map([
                ['peer', node('peer')],
                ['second', node('second')]
            ])
        })
        inventories.push(inventory('second'))
        getSettings.mockImplementation(async target => {
            const value = outcomes.get(target.nodeId) ?? settings(target.nodeId)
            if (target.nodeId === 'self')
                value.settings.modelLoads = [
                    {
                        profileKey: 'coder-local',
                        modelId: 'source-artifact',
                        requested: {
                            context: { mode: 'auto', floor: 4096, cap: 8192 },
                            gpu: { mode: 'auto' },
                            slots: 1
                        }
                    }
                ]
            return value
        })
        inspect.mockImplementation(async request => {
            const value = inspection(request.nodeId ?? 'self', request.modelId)
            value.options?.push(
                {
                    field: 'context',
                    label: 'Context',
                    kind: 'integer',
                    portable: true,
                    supported: true
                },
                {
                    field: 'gpu',
                    label: 'GPU',
                    kind: 'enum',
                    values: ['auto', 'cpu'],
                    portable: true,
                    supported: true
                }
            )
            return value
        })
        preview.mockImplementation(async request => ({
            settings: request.settings,
            revision: 5,
            policyRevision: '1',
            previewDigest: (request.nodeId === 'peer' ? 'a' : 'b').repeat(64),
            errors: {},
            restart: true,
            rebind: false,
            auto: {
                status: 'available',
                requested: request.modelChange?.load?.set ?? {},
                estimate: { status: 'estimated', source: `${request.nodeId} estimate` },
                resolverRevision: 'auto/fixture',
                reviewedInputId: `${request.nodeId}-review`,
                resolved: {
                    perSlotContext: request.nodeId === 'peer' ? 4096 : 8192,
                    totalContext: request.nodeId === 'peer' ? 4096 : 8192,
                    slots: 1,
                    gpu: { mode: 'cpu' },
                    options: {},
                    artifacts: [
                        {
                            role: 'model',
                            path: `/${request.nodeId}/model.gguf`,
                            resolvedTarget: `/${request.nodeId}/model.gguf`,
                            size: '100',
                            mtimeNs: '1',
                            sampleSha256: 'c'.repeat(64)
                        }
                    ],
                    nativeBuildId: `${request.nodeId}-build`,
                    sourceInputDigest: 'd'.repeat(64),
                    presetDigest: 'e'.repeat(64),
                    resolverRevision: 'auto/fixture',
                    reviewedInputId: `${request.nodeId}-review`
                }
            }
        }))
        await render()
        await reviewTarget()
        await reviewTarget('second')
        expect(change('peer').load?.set).toEqual({
            slots: 1,
            context: { mode: 'auto', floor: 4096, cap: 8192 },
            gpu: { mode: 'auto' }
        })
        expect(change('second').load?.set).toEqual({
            slots: 1,
            context: { mode: 'auto', floor: 4096, cap: 8192 },
            gpu: { mode: 'auto' }
        })
        expect(button('Confirm deploy to peer').disabled).toBe(false)
        expect(button('Confirm deploy to second').disabled).toBe(false)
        await click('Confirm deploy to second')
        expect(apply.mock.calls[0]?.[0].modelChange?.previewDigest).toBe('b'.repeat(64))
        expect(apply.mock.calls[0]?.[0].modelChange).not.toHaveProperty('resolved')
    })
    it('rejects reserved profile keys before any copy or review', async () => {
        await render()
        await check('Deploy to peer')
        await input('Profile key peer', 'medium-local')
        expect(button('Review deploy to peer', group()).disabled).toBe(true)
        expect(group().textContent).toContain('Enter a valid profile key')
        expect(preview).not.toHaveBeenCalled()
        expect(copyOwners.size).toBe(0)
    })
    it('keeps the caller disabled while creating a source profile', async () => {
        await render(source, true)
        const checkbox = group().querySelector('input')
        expect(checkbox?.disabled).toBe(true)
        expect(preview).not.toHaveBeenCalled()
        expect(copyOwners.size).toBe(0)
    })
})

describe('per-target refusal, cancellation and correlated readback', () => {
    it('serializes overlapping target previews without superseding either review', async () => {
        useNodesStore.setState({
            nodes: new Map([
                ['peer', node('peer')],
                ['second', node('second')]
            ])
        })
        inventories.push(inventory('second'))
        let finish: (() => void) | undefined
        preview.mockImplementationOnce(
            request =>
                new Promise(resolve => {
                    finish = () =>
                        resolve({
                            settings: request.settings,
                            revision: 5,
                            policyRevision: '1',
                            previewDigest: 'd'.repeat(64),
                            errors: {},
                            restart: false,
                            rebind: false
                        })
                })
        )
        await render()
        await reviewTarget('peer')
        await reviewTarget('second')
        expect(preview).toHaveBeenCalledTimes(1)
        if (!finish) throw new Error('First preview not dispatched')
        await act(async () => finish?.())
        expect(preview).toHaveBeenCalledTimes(2)
        expect(button('Confirm deploy to peer').disabled).toBe(false)
        expect(button('Confirm deploy to second').disabled).toBe(false)
        expect(apply).not.toHaveBeenCalled()
    })
    it('offers the existing configuration link when target inventory is unavailable', async () => {
        inventories = [inventory('self', [model('source-artifact')])]
        await render()
        await reviewTarget()
        await click('Configure target peer')
        expect(configure).toHaveBeenCalledWith({
            targetId: 'peer',
            nodeId: 'peer',
            nodeName: 'peer',
            identity: { engine: 'llamacpp', model: 'source-artifact', profileKey: 'coder-local' }
        })
        expect(group().textContent).toContain('Target inventory unavailable')
        expect(set).not.toHaveBeenCalled()
        expect(apply).not.toHaveBeenCalled()
    })
    it('refuses a saved target generation lock even for matching selected values', async () => {
        get.mockResolvedValue(
            policy({
                profiles: [
                    {
                        key: 'coder-local',
                        model: 'coder-local',
                        engine: 'llamacpp',
                        displayName: 'Locked',
                        localOnly: true,
                        requestOptions: { temperature: 0.25 }
                    }
                ]
            })
        )
        await render()
        await reviewTarget()
        expect(deployment()?.disposition).toBe('refused')
        expect(group().textContent?.toLowerCase()).toContain('lock')
        expect(apply).not.toHaveBeenCalled()
    })
    it('permits load-only deployment to a generation-locked profile', async () => {
        get.mockResolvedValue(
            policy({
                profiles: [
                    {
                        key: 'coder-local',
                        model: 'coder-local',
                        engine: 'llamacpp',
                        displayName: 'Locked',
                        localOnly: true
                    }
                ]
            })
        )
        await render()
        await check('Temperature:')
        await reviewTarget()
        expect(change().policy).toBeUndefined()
        expect(change().load?.set.slots).toBe(2)
    })
    it('refuses a paused target without writing', async () => {
        get.mockResolvedValue(policy({ availability: 'paused' }))
        await render()
        await reviewTarget()
        expect(deployment()?.disposition).toBe('refused')
        expect(group().textContent).toContain('target is paused')
        expect(preview).not.toHaveBeenCalled()
        expect(apply).not.toHaveBeenCalled()
    })
    it('reads a paused source while only target management authorizes deployment', async () => {
        get.mockImplementation(async nodeId =>
            nodeId === undefined ? policy({ availability: 'paused' }) : policy()
        )
        await render(source.slice(0, -1) + ',"availability":"paused"}')
        await reviewTarget()
        await click('Confirm deploy to peer')
        expect(apply.mock.calls.map(([request]) => request.nodeId)).toEqual(['peer'])
        expect(deployment()?.disposition).toBe('applied')
        expect(set).not.toHaveBeenCalled()
    })
    it('keeps disconnected targets unsubmitted', async () => {
        await render()
        await check('Deploy to peer')
        await act(async () => useConnectionStore.setState({ connected: false }))
        expect(button('Review deploy to peer', group()).disabled).toBe(true)
        expect(preview).not.toHaveBeenCalled()
        expect(apply).not.toHaveBeenCalled()
    })
    it('refreshes target permission before correspondence and rejects a denied no-op', async () => {
        window.pairApi.cluster.getRemoteManagementForNode = vi
            .fn<typeof window.pairApi.cluster.getRemoteManagementForNode>()
            .mockResolvedValue({ version: 1, revision: '2', state: 'denied', allowed: false })
        await render()
        await reviewTarget()
        expect(deployment()?.disposition).toBe('refused')
        expect(get).not.toHaveBeenCalled()
        expect(initial).not.toHaveBeenCalled()
    })
    it('rechecks management after review and refuses revoked grants before applying', async () => {
        await render()
        await reviewTarget()
        window.pairApi.cluster.getRemoteManagementForNode = vi
            .fn<typeof window.pairApi.cluster.getRemoteManagementForNode>()
            .mockResolvedValue({ version: 1, revision: '2', state: 'denied', allowed: false })
        await click('Confirm deploy to peer')
        expect(deployment()?.disposition).toBe('refused')
        expect(apply).not.toHaveBeenCalled()
    })
    it('refuses changed policy revisions and offers Review again', async () => {
        await render()
        await reviewTarget()
        get.mockResolvedValue(policy({ revision: '2' }))
        await click('Confirm deploy to peer')
        expect(deployment()?.disposition).toBe('refused')
        expect(button('Review again', group()).disabled).toBe(false)
        expect(apply).not.toHaveBeenCalled()
    })
    it.each([-32602, -32010, -32011, -32012, -32014])(
        'preserves canonical refusal code %s without resending',
        async code => {
            apply.mockRejectedValue(new ServiceCallError('refused', code))
            await render()
            await reviewTarget()
            await click('Confirm deploy to peer')
            expect(deployment()?.disposition).toBe('refused')
            expect(apply).toHaveBeenCalledTimes(1)
            expect(set).not.toHaveBeenCalled()
        }
    )
    it('cancels one copy while another target independently reviews and applies', async () => {
        useNodesStore.setState({
            nodes: new Map([
                ['peer', node('peer')],
                ['second', node('second')]
            ])
        })
        inventories = [
            inventory('self', [model('source-artifact')]),
            inventory('peer', []),
            inventory('second')
        ]
        await render()
        await reviewTarget('peer')
        await reviewTarget('second')
        expect(button('Confirm deploy to second').disabled).toBe(false)
        await copyOutcome('peer', 'cancelled')
        await click('Confirm deploy to second')
        expect(deployment('peer')?.disposition).toBe('cancelled')
        expect(deployment('second')?.disposition).toBe('applied')
        expect(apply.mock.calls.map(([request]) => request.nodeId)).toEqual(['second'])
        await click('Close copy on peer')
        inventories[1] = inventory('peer')
        await click('Review again', group('peer'))
        expect(button('Confirm deploy to peer').disabled).toBe(false)
        await click('Confirm deploy to peer')
        expect(deployment('peer')?.disposition).toBe('applied')
        expect(apply.mock.calls.map(([request]) => request.nodeId)).toEqual(['second', 'peer'])
    })
    it('shows unknown before a lost-reply read finishes and then reports the correlated actual result', async () => {
        let finishRead: ((value: EngineSettingsSnapshot) => void) | undefined
        apply.mockImplementation(async request => {
            getSettings.mockImplementation(target =>
                target.nodeId === 'peer'
                    ? new Promise(resolve => {
                          finishRead = resolve
                      })
                    : Promise.resolve(settings(target.nodeId))
            )
            throw new Error(`Lost reply ${request.requestId}`)
        })
        await render()
        await reviewTarget()
        await click('Confirm deploy to peer')
        expect(deployment()?.disposition).toBe('unknown')
        expect(group().textContent).toContain('unknown')
        const request = apply.mock.calls[0]?.[0]
        if (!request?.requestId || !finishRead) throw new Error('Missing correlated readback')
        const response = {
            ...settings('peer'),
            revision: 6,
            sequence: 2,
            requestId: request.requestId,
            phase: 'succeeded',
            policyRevision: '2'
        } satisfies EngineSettingsSnapshot
        await act(async () => finishRead?.(response))
        expect(deployment()?.disposition).toBe('applied')
        expect(apply).toHaveBeenCalledTimes(1)
    })
    it('does not infer success from equal fields without a correlated receipt', async () => {
        apply.mockRejectedValue(new Error('Lost reply'))
        await render()
        await reviewTarget()
        await click('Confirm deploy to peer')
        expect(deployment()?.disposition).toBe('unknown')
        expect(button('Review again', group()).disabled).toBe(true)
        expect(apply).toHaveBeenCalledTimes(1)
        await click('Read deploy operation status on peer')
        expect(deployment()?.disposition).toBe('unknown')
        expect(apply).toHaveBeenCalledTimes(1)
    })
    it('shows the actual failed receipt after an unknown apply instead of claiming field equality', async () => {
        apply.mockRejectedValue(new Error('Lost reply'))
        await render()
        await reviewTarget()
        await click('Confirm deploy to peer')
        expect(deployment()?.disposition).toBe('unknown')
        const request = apply.mock.calls[0]?.[0]
        if (!request?.requestId) throw new Error('Missing request identity')
        outcomes.set('peer', {
            ...settings('peer'),
            requestId: request.requestId,
            revision: 6,
            sequence: 2,
            phase: 'failed',
            error: 'Native stage refused',
            policyRevision: '1'
        })
        await click('Read deploy operation status on peer')
        expect(deployment()?.disposition).toBe('refused')
        expect(group().textContent).toContain('Native stage refused')
        expect(apply).toHaveBeenCalledTimes(1)
    })
    it('expires copied-but-undispatched reviews after a source edit without deleting files', async () => {
        inventories[1] = inventory('peer', [])
        get.mockResolvedValue(policy({ profiles: [] }))
        await render()
        await reviewTarget()
        await render(source.replace('0.25', '0.75'))
        await copyOutcome('peer', 'copied')
        expect(preview).not.toHaveBeenCalled()
        expect(group().textContent).toContain('Files copied. Source or selection changed')
        await click('Review again', group())
        expect(change().modelId).toBe('copied-target-name')
        expect(change().policy?.generation).toEqual({ temperature: 0.75 })
        expect(apply).not.toHaveBeenCalled()
    })
    it('expires an undispatched review on source edits', async () => {
        await render()
        await reviewTarget()
        await render(source.replace('0.25', '0.75'))
        expect(
            Array.from(container.querySelectorAll('button')).some(
                item => item.textContent === 'Confirm deploy to peer'
            )
        ).toBe(false)
        expect(apply).not.toHaveBeenCalled()
        await click('Review again', group())
        expect(preview.mock.calls.at(-1)?.[0].modelChange?.policy?.generation).toEqual({
            temperature: 0.75
        })
    })
    it('preserves immutable accepted attempts when the source changes during apply', async () => {
        let finish: (() => void) | undefined
        apply.mockImplementation(
            request =>
                new Promise(resolve => {
                    finish = () => {
                        outcomes.set('peer', {
                            ...settings('peer'),
                            requestId: request.requestId ?? '',
                            revision: 6,
                            sequence: 2,
                            phase: 'succeeded'
                        })
                        resolve({ revision: 6, phase: 'succeeded' })
                    }
                })
        )
        await render()
        await reviewTarget()
        await click('Confirm deploy to peer')
        await render(source.replace('0.25', '0.75'))
        if (!finish) throw new Error('Apply not dispatched')
        await act(async () => finish?.())
        expect(apply.mock.calls[0]?.[0].modelChange?.policy?.generation).toEqual({
            temperature: 0.25
        })
        expect(deployment()?.disposition).toBe('applied')
    })
    it('retains copied files after later settings refusal and reuses their actual name on retry', async () => {
        inventories[1] = inventory('peer', [])
        get.mockResolvedValue(policy({ profiles: [] }))
        apply.mockRejectedValue(new ServiceCallError('policy conflict', -32010))
        await render()
        await reviewTarget()
        await copyOutcome('peer', 'copied')
        await click('Confirm deploy to peer')
        expect(deployment()?.disposition).toBe('refused')
        await click('Review again', group())
        expect(preview.mock.calls.at(-1)?.[0].modelChange?.modelId).toBe('copied-target-name')
        expect(copyOwners.get('peer')?.initialModel).toBe('source-artifact')
        expect(set).not.toHaveBeenCalled()
    })
})
