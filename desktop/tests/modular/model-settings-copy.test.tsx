// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act, type ComponentProps } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { NodeItem } from '@/shared/types/nodes'
import type { JsonValue } from '@/shared/types/json'
import type { NodePolicyDocument } from '@/shared/types/node-policy'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi, type INodePolicyApi } from '@/ui/api/pair-api'
import { ServiceCallError } from '@/ui/api/service-transport'
import ModelSettingsCopy from '@/ui/components/ModelSettings/ModelSettingsCopy'
import ModelSettingsPanel from '@/ui/components/ModelSettings/ModelSettingsPanel'
import ModelLoadSettings from '@/ui/components/ModelSettings/ModelLoadSettings'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { parseModelSettings } from '@/shared/utils/node-policy-model-settings'
import type { EngineSettingsSnapshot } from '@/shared/types/engine-settings'
import type { Estimate, InspectModelResult } from '@/shared/types/model-config'
import type { EngineProgress } from '@/shared/types/engines'
import type { IEngineApi } from '@/ui/api/engine-api'
import { ModelCopyModal } from '@/ui/components/ModelManager/ModelCopyModal'
import { useEngineSettingsStore } from '@/ui/stores/engine-settings.store'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { defaultPortableLoadFields } from '@/shared/utils/node-policy-model-copy'
import { parseEngineSettingsPreview } from '@/electron/service-bridge/engine-settings'

vi.mock('@/ui/components/ModelManager/ModelCopyModal', () => ({
    ModelCopyModal: ({ open, onOpenChange }: ComponentProps<typeof ModelCopyModal>) =>
        open ? <button onClick={() => onOpenChange(false)}>Close content copy owner</button> : null
}))

const identity = { engine: 'llamacpp', model: 'model' }
const source =
    '{"profiles":[{"name":"source","engine":"llamacpp","model":"model","requestOptions":{"temperature":0.25}}]}'
const get = vi.fn<INodePolicyApi['get']>()
const set = vi.fn<INodePolicyApi['set']>()
const configure = vi.fn()
let root: Root
let container: HTMLDivElement
function snapshot(extra: Record<string, JsonValue> = {}, revision = '1'): NodePolicyDocument {
    return {
        policy: JSON.stringify({
            profiles: [
                {
                    name: 'target',
                    ...identity,
                    requestOptions: { temperature: 1, seed: 42 },
                    ...extra
                }
            ]
        }),
        revision,
        activation: null,
        availability: 'available',
        recoveryRequired: false
    }
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
function button(label: string): HTMLButtonElement {
    const found = Array.from(container.querySelectorAll('button')).find(
        item => item.textContent === label
    )
    if (!found) throw new Error(`Missing button ${label}`)
    return found
}
async function click(label: string) {
    await act(async () => button(label).click())
}
async function check(label: string) {
    const found = Array.from(container.querySelectorAll('label')).find(item =>
        item.textContent?.includes(label)
    )
    const field = found?.control
    if (!(field instanceof HTMLInputElement)) throw new Error(`Missing checkbox ${label}`)
    await act(async () => field.click())
}
async function render(policy = source) {
    await act(async () =>
        root.render(
            <ModelSettingsCopy
                identity={identity}
                policy={policy}
                targetId="self"
                onConfigureTarget={configure}
            />
        )
    )
}
async function preview() {
    await check('Copy to peer')
    await click('Preview selected targets')
}
function written() {
    const request = set.mock.calls[0]?.[0]
    if (!request) throw new Error('No policy write')
    const parsed = parseModelSettings(request.policy)
    if (!parsed.ok) throw new Error(parsed.error.message)
    return { request, profile: parsed.value[0] }
}
beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    get.mockReset().mockResolvedValue(snapshot())
    set.mockReset().mockImplementation(async request => ({
        policy: request.policy,
        revision: '2',
        activation: null
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
    api.cluster.getRemoteManagementForNode = vi
        .fn<typeof api.cluster.getRemoteManagementForNode>()
        .mockResolvedValue({ version: 1, revision: '1', state: 'allowed', allowed: true })
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    useConnectionStore.setState({ connected: true, selfId: 'self' })
    useEngineSettingsStore.setState({ entries: {}, modelDrafts: {} })
    useClusterInvitationsStore.setState({ remoteManagementContext: 1 })
    useEngineProgressStore.setState({ progress: new Map() })
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

describe('selective generation copy', () => {
    it('previews before CAS and preserves unselected target mode and options', async () => {
        get.mockResolvedValue(
            snapshot({ requestOptionsMode: 'enforced', opaque: { nested: true } })
        )
        await render()
        await preview()
        expect(set).not.toHaveBeenCalled()
        expect(container.textContent).toContain('Content identity unknown')
        await click('Confirm copy to peer')
        expect(written().profile).toMatchObject({
            options: { temperature: 0.25, seed: 42 },
            mode: 'enforced'
        })
        expect(written().request.expectedPersistedRevision).toBe('1')
        expect(container.textContent).toContain('peer: Saved')
    })
    it('shows Clear on target before confirming selected absent leaves', async () => {
        await render()
        await check('Seed: Clear on target')
        await preview()
        await click('Confirm copy to peer')
        expect(written().profile.options).toEqual({ temperature: 0.25 })
    })
    it('copies mode only after an explicit selection', async () => {
        get.mockResolvedValue(snapshot({ requestOptionsMode: 'enforced' }))
        await render()
        await check('Copy request precedence')
        await preview()
        await click('Confirm copy to peer')
        expect(written().profile.mode).toBe('defaults')
    })
    it('requires Configure/Create for a missing profile and initiates no transfer', async () => {
        get.mockResolvedValue({ ...snapshot(), policy: '{"profiles":[]}' })
        await render()
        await preview()
        expect(container.textContent).toContain('Missing profile')
        expect(set).not.toHaveBeenCalled()
        await click('Configure/Create on peer')
        expect(configure).toHaveBeenCalledWith({
            targetId: 'peer',
            nodeId: 'peer',
            nodeName: 'peer',
            identity
        })
    })
    it('refuses a saved target lock even for matching selected values', async () => {
        get.mockResolvedValue(snapshot({ requestOptions: { temperature: 0.25 }, localOnly: true }))
        await render()
        await preview()
        expect(container.textContent).toContain('Locked')
        expect(set).not.toHaveBeenCalled()
    })
    it('checks permission for a selected no-op', async () => {
        window.pairApi.cluster.getRemoteManagementForNode = vi
            .fn<typeof window.pairApi.cluster.getRemoteManagementForNode>()
            .mockResolvedValue({ version: 1, revision: '1', state: 'denied', allowed: false })
        await render()
        await preview()
        expect(container.textContent).toContain('Denied')
        expect(get).not.toHaveBeenCalled()
        expect(set).not.toHaveBeenCalled()
    })
    it('refuses a paused target without writing', async () => {
        get.mockResolvedValue({ ...snapshot(), availability: 'paused' })
        await render()
        await preview()
        expect(container.textContent).toContain('target is paused')
        expect(set).not.toHaveBeenCalled()
    })
    it('shows explicit alias mapping with both identities and unverified content', async () => {
        get.mockResolvedValue(snapshot({ model: 'alias' }))
        await render()
        await check('Copy to peer')
        const field = container.querySelector<HTMLInputElement>(
            'input[aria-label="Target model peer"]'
        )
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
        if (!field || !setter) throw new Error('Missing target model input')
        await act(async () => {
            setter.call(field, 'alias')
            field.dispatchEvent(new Event('input', { bubbles: true }))
        })
        await click('Preview selected targets')
        expect(container.textContent).toContain(
            'llamacpp / model → llamacpp / alias: unverified content'
        )
        await click('Confirm copy to peer')
        expect(written().profile.model).toBe('alias')
    })
    it('refuses a stale revision and requires explicit fresh preview', async () => {
        await render()
        await preview()
        get.mockResolvedValue(snapshot({}, '2'))
        await click('Confirm copy to peer')
        expect(container.textContent).toContain('Conflict')
        expect(set).not.toHaveBeenCalled()
        await click('Preview selected targets')
        await click('Confirm copy to peer')
        expect(written().request.expectedPersistedRevision).toBe('2')
    })
    it('keeps target failures separate from another target success', async () => {
        useNodesStore.setState({
            nodes: new Map([
                ['peer', node('peer')],
                ['other', node('other')]
            ])
        })
        set.mockImplementation(async request => {
            if (request.nodeId === 'peer') throw new ServiceCallError('locked', -32014)
            return { policy: request.policy, revision: '2', activation: null }
        })
        await render()
        await check('Copy to peer')
        await check('Copy to other')
        await click('Preview selected targets')
        await click('Confirm copy to peer')
        await click('Confirm copy to other')
        expect(container.textContent).toContain('peer: Locked')
        expect(container.textContent).toContain('other: Saved')
    })
    it('reads back a lost reply without resending and identifies matching selected fields', async () => {
        set.mockImplementation(async request => {
            get.mockResolvedValue({ ...snapshot(), policy: request.policy })
            throw new Error('reply lost')
        })
        await render()
        await preview()
        await click('Confirm copy to peer')
        expect(container.textContent).toContain('Target already matches selected fields')
        expect(container.textContent).toContain('no operation receipt')
        expect(set).toHaveBeenCalledTimes(1)
    })
    it('keeps unmatched lost replies unknown and only retries after a fresh comparison', async () => {
        set.mockRejectedValueOnce(new Error('reply lost'))
        await render()
        await preview()
        await click('Confirm copy to peer')
        expect(container.textContent).toContain('Outcome unknown')
        expect(set).toHaveBeenCalledTimes(1)
        await click('Preview selected targets')
        await click('Confirm copy to peer')
        expect(set).toHaveBeenCalledTimes(2)
    })
    it.each(['pending', 'error'])(
        'reports activation %s separately from persistence',
        async state => {
            set.mockImplementation(async request => ({
                policy: request.policy,
                revision: '2',
                activation: {
                    state: state === 'pending' ? 'pending' : 'error',
                    activeRevision: '1'
                }
            }))
            await render()
            await preview()
            await click('Confirm copy to peer')
            expect(container.textContent).toContain(`Saved, activation ${state}`)
        }
    )
    it('blocks preview while disconnected', async () => {
        await render()
        await check('Copy to peer')
        await act(async () => useConnectionStore.setState({ connected: false }))
        expect(button('Preview selected targets').disabled).toBe(true)
        expect(set).not.toHaveBeenCalled()
    })
    it('retains uncertainty and readback after disconnecting a dispatched write', async () => {
        let reject: ((error: Error) => void) | undefined
        set.mockImplementation(
            () =>
                new Promise((_resolve, no) => {
                    reject = no
                })
        )
        await render()
        await preview()
        await click('Confirm copy to peer')
        await act(async () => useConnectionStore.setState({ connected: false }))
        expect(container.textContent).toContain('Outcome unknown: disconnected after dispatch')
        get.mockResolvedValue(snapshot({ requestOptions: { temperature: 0.25 } }))
        if (!reject) throw new Error('Missing dispatched write')
        await act(async () => reject?.(new Error('connection lost')))
        expect(container.textContent).toContain('Target already matches selected fields')
        expect(set).toHaveBeenCalledTimes(1)
    })
    it('retains the dispatched selection and result after editing the source', async () => {
        let finish: (() => void) | undefined
        set.mockImplementation(
            request =>
                new Promise(resolve => {
                    finish = () =>
                        resolve({ policy: request.policy, revision: '2', activation: null })
                })
        )
        await render()
        await preview()
        await click('Confirm copy to peer')
        await render(source.replace('0.25', '0.9'))
        expect(button('Preview selected targets').disabled).toBe(true)
        if (!finish) throw new Error('Missing dispatched write')
        await act(async () => finish?.())
        expect(container.textContent).toContain('peer: Saved')
        expect(container.textContent).toContain('Attempted selection: temperature=0.25')
        expect(set).toHaveBeenCalledTimes(1)
    })
    it('copies readable preferences from a paused source without granting source mutation', async () => {
        Object.defineProperty(HTMLDialogElement.prototype, 'show', {
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
        get.mockImplementation(async id => ({
            ...snapshot(),
            availability: id === 'peer' ? 'paused' : 'available'
        }))
        window.pairApi.cluster.getRemoteManagementForNode = vi
            .fn<typeof window.pairApi.cluster.getRemoteManagementForNode>()
            .mockResolvedValue({ version: 1, revision: '1', state: 'denied', allowed: false })
        await act(async () =>
            root.render(
                <ModelSettingsPanel
                    targetId="peer"
                    nodeId="peer"
                    nodeName="Peer"
                    identity={identity}
                    open
                />
            )
        )
        await check('Copy to This PC')
        await click('Preview selected targets')
        await click('Confirm copy to This PC')
        expect(written().request.nodeId).toBeUndefined()
        expect(container.textContent).toContain('This PC: Saved')
        expect(set).toHaveBeenCalledTimes(1)
    })
})

describe('portable target-previewed copy', () => {
    const memory: Estimate = {
        status: 'estimated',
        memoryTimestamp: '2026-10-28T00:00:00Z',
        domains: [
            {
                id: 'host',
                kind: 'host',
                capacityBytes: '10000000000',
                freeBytes: '9000000000',
                requiredBytes: '1000000000'
            }
        ],
        components: [
            { name: 'weights', domainId: 'host', status: 'estimated', bytes: '1000000000' },
            ...[
                'attention-cache',
                'recurrent-state',
                'compute-workspace',
                'prompt-cache',
                'companions'
            ].map(
                name =>
                    ({ name, status: 'excluded', reason: 'Not present' }) satisfies NonNullable<
                        Estimate['components']
                    >[number]
            )
        ]
    }
    const getSettings = vi.fn<IEngineApi['getSettings']>()
    const inspect = vi.fn<IEngineApi['inspectModel']>()
    const review = vi.fn<IEngineApi['previewSettings']>()
    const apply = vi.fn<IEngineApi['applySettings']>()
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
                launchText:
                    nodeId === 'self'
                        ? '/source/local/path --device source-native-id'
                        : 'destination launch',
                modelLoads:
                    nodeId === 'self'
                        ? [
                              {
                                  modelId: 'model',
                                  requested: { slots: 2, gpu: { mode: 'layers', layers: 20 } },
                                  resolved: {
                                      perSlotContext: 8192,
                                      totalContext: 16384,
                                      slots: 2,
                                      gpu: { mode: 'layers', layers: 20 },
                                      options: {},
                                      artifacts: [
                                          {
                                              role: 'model',
                                              path: '/source/local/model.gguf',
                                              resolvedTarget: '/source/blob',
                                              size: '1',
                                              mtimeNs: '1',
                                              sampleSha256: 'source approval'
                                          }
                                      ],
                                      nativeBuildId: 'source-build',
                                      sourceInputDigest: 'source-digest',
                                      presetDigest: 'source-preset'
                                  }
                              }
                          ]
                        : undefined
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
    function inspected(nodeId: string, modelId = 'model'): InspectModelResult {
        return {
            engine: 'llamacpp',
            nodeId,
            modelId,
            status: 'resolved',
            source: `${nodeId} local resolver`,
            estimate: { status: 'unknown' },
            artifacts: [
                {
                    role: 'model',
                    identity: `${nodeId}-artifact`,
                    size: '1',
                    assurance: 'local binding'
                }
            ],
            options: [
                {
                    field: 'slots',
                    label: 'Slots',
                    kind: 'integer',
                    supported: true,
                    portable: true
                },
                {
                    field: 'gpu',
                    label: 'GPU placement',
                    kind: 'enum',
                    values: ['cpu', 'layers', 'native-default'],
                    supported: true,
                    portable: true
                },
                {
                    field: 'native.device',
                    label: 'Native device',
                    kind: 'string',
                    supported: false,
                    portable: false,
                    reason: 'Local only'
                }
            ]
        }
    }
    beforeEach(() => {
        getSettings.mockReset().mockImplementation(async target => settings(target.nodeId))
        inspect
            .mockReset()
            .mockImplementation(async request =>
                inspected(request.nodeId ?? 'self', request.modelId)
            )
        review.mockReset().mockImplementation(async request => ({
            settings: { serverPort: 999, proxyPort: 998, launchText: 'not a baseline' },
            revision: 5,
            policyRevision: '1',
            previewDigest: `target-digest-${request.nodeId}`,
            errors: {},
            restart: true,
            rebind: false,
            diff: [
                {
                    field: 'gpu',
                    before: 'target native placement',
                    after: 'target-resolved finite placement'
                }
            ],
            impact: {
                restartScope: 'whole-router',
                affectedModels: ['target model'],
                drainTimeoutSeconds: 60,
                mayInterruptActive: true,
                staysOff: true
            }
        }))
        apply.mockReset().mockImplementation(async request => {
            getSettings.mockResolvedValue({
                ...settings(request.nodeId),
                revision: 6,
                sequence: 2,
                requestId: request.requestId ?? '',
                phase: 'succeeded'
            })
            return { revision: 6, phase: 'succeeded' }
        })
        window.pairApi.engines.getSettings = getSettings
        window.pairApi.engines.inspectModel = inspect
        window.pairApi.engines.previewSettings = review
        window.pairApi.engines.applySettings = apply
    })
    async function extendedPreview() {
        await render()
        await click('Select portable load and routing preferences')
        await check('Copy to peer')
        await click('Preview selected targets')
    }
    it('keeps a completed destination review when the mounted source debounce refreshes only estimates', async () => {
        vi.useFakeTimers()
        try {
            await act(async () =>
                root.render(
                    <>
                        <ModelLoadSettings
                            target={{ nodeId: 'self', engine: 'llamacpp' }}
                            modelId="model"
                            policy={source}
                            savedPolicy={source}
                            policyRevision="1"
                            authority="source"
                            disabled={false}
                        />
                        <ModelSettingsCopy
                            identity={identity}
                            policy={source}
                            targetId="self"
                            onConfigureTarget={configure}
                        />
                    </>
                )
            )
            await click('Select portable load and routing preferences')
            await check('Copy to peer')
            await act(async () =>
                useEngineSettingsStore.getState().editModel('self:llamacpp:model', {
                    modelId: 'model',
                    expectedPolicyRevision: '1',
                    load: { set: { slots: 3 } }
                })
            )
            await click('Preview selected targets')
            const confirmation = () =>
                Array.from(container.querySelectorAll('button')).find(
                    item => item.textContent === 'Confirm copy to peer'
                )
            expect(confirmation()?.disabled).toBe(false)
            inspect.mockImplementation(async request => ({
                ...inspected(request.nodeId ?? 'self', request.modelId),
                estimate: { status: 'unknown', timestamp: '2026-10-28T00:00:01Z' }
            }))
            await act(async () => vi.advanceTimersByTime(250))
            expect(review.mock.calls.some(([request]) => request.nodeId === 'self')).toBe(true)
            expect(confirmation()?.disabled).toBe(false)
            expect(apply).not.toHaveBeenCalled()
        } finally {
            vi.useRealTimers()
        }
    })
    it('marks a destination estimate stale when its reviewed target revision expires', async () => {
        inspect.mockImplementation(async request => ({
            ...inspected(request.nodeId ?? 'self', request.modelId),
            estimate: memory
        }))
        review.mockImplementation(async request => ({
            settings: request.settings,
            revision: 5,
            policyRevision: '1',
            previewDigest: 'a'.repeat(64),
            errors: {},
            restart: false,
            rebind: false,
            estimate: memory
        }))
        await extendedPreview()
        expect(container.textContent).toContain('(Fits)')
        await act(async () =>
            useEngineSettingsStore.getState().receive({
                ...settings('peer'),
                sequence: 2,
                revision: 6
            })
        )
        expect(
            useEngineSettingsStore.getState().modelDrafts['peer:llamacpp:model'].review
        ).toBeUndefined()
        expect(container.textContent).toContain('Advisory memory estimate (stale)')
        expect(container.textContent).not.toContain('(Fits)')
        const confirmation = Array.from(container.querySelectorAll('button')).find(
            item => item.textContent === 'Confirm copy to peer'
        )
        expect(confirmation?.disabled).toBe(true)
    })
    it('reviews independently fresh Auto for each destination and applies only its target digest', async () => {
        useNodesStore.setState({
            nodes: new Map([
                ['peer', node('peer')],
                ['second', node('second')]
            ])
        })
        getSettings.mockImplementation(async target => {
            const value = settings(target.nodeId)
            if (target.nodeId === 'self') {
                const load = value.settings.modelLoads?.[0]
                if (!load) throw new Error('Source fixture load missing')
                load.requested = {
                    context: { mode: 'auto', floor: 4096, cap: 8192 },
                    gpu: { mode: 'auto' },
                    slots: 1
                }
            }
            return value
        })
        inspect.mockImplementation(async request => {
            const value = inspected(request.nodeId ?? 'self', request.modelId)
            value.options?.push({
                field: 'context',
                label: 'Context',
                kind: 'integer',
                portable: true,
                supported: true
            })
            return value
        })
        const revisions = new Map<string, number>()
        review.mockImplementation(async request => {
            const version = (revisions.get(request.nodeId) ?? 0) + 1
            revisions.set(request.nodeId, version)
            const id = `${request.nodeId}-auto-${version}`
            const digest = (request.nodeId === 'peer' ? 'a' : 'b').repeat(63) + version
            return parseEngineSettingsPreview(
                JSON.parse(
                    JSON.stringify({
                        settings: request.settings,
                        revision: 5,
                        policyRevision: '1',
                        previewDigest: digest,
                        errors: {},
                        restart: true,
                        rebind: false,
                        auto: {
                            status: 'available',
                            requested: request.modelChange?.load?.set ?? {},
                            estimate: { status: 'estimated', source: id },
                            resolverRevision: 'auto/fixture',
                            reviewedInputId: id,
                            resolved: {
                                perSlotContext: request.nodeId === 'peer' ? 4096 : 8192,
                                totalContext: request.nodeId === 'peer' ? 4096 : 8192,
                                slots: 1,
                                gpu:
                                    request.nodeId === 'peer'
                                        ? { mode: 'cpu' }
                                        : { mode: 'layers', layers: 8 },
                                options: {},
                                artifacts: [
                                    {
                                        role: 'model',
                                        path: `/${request.nodeId}/model.gguf`,
                                        resolvedTarget: `/${request.nodeId}/model.gguf`,
                                        size: '1024',
                                        mtimeNs: '1',
                                        sampleSha256: 'c'.repeat(64)
                                    }
                                ],
                                nativeBuildId: `${request.nodeId}-build`,
                                sourceInputDigest: 'd'.repeat(64),
                                presetDigest: 'e'.repeat(64),
                                resolverRevision: 'auto/fixture',
                                reviewedInputId: id
                            }
                        }
                    })
                )
            )
        })
        await render()
        await click('Select portable load and routing preferences')
        await check('Copy to peer')
        await check('Copy to second')
        await click('Preview selected targets')
        expect(review).toHaveBeenCalledTimes(2)
        for (const [request] of review.mock.calls) {
            expect(request.modelChange?.load?.set).toEqual({
                context: { mode: 'auto', floor: 4096, cap: 8192 },
                slots: 1,
                gpu: { mode: 'auto' }
            })
            expect(request.settings).toEqual(settings(request.nodeId).settings)
            expect(JSON.stringify(request)).not.toContain('source-build')
            expect(JSON.stringify(request)).not.toContain('/source/')
            expect(JSON.stringify(request.modelChange)).not.toContain('layers')
        }
        expect(container.textContent).toContain('Feasible context: 4096')
        expect(container.textContent).toContain('Feasible context: 8192')
        expect(container.textContent).not.toContain('"layers":20')
        expect(apply).not.toHaveBeenCalled()
        await click('Preview selected targets')
        expect(review).toHaveBeenCalledTimes(4)
        expect(container.textContent).toContain('peer-auto-2')
        expect(container.textContent).toContain('second-auto-2')
        await click('Confirm copy to peer')
        expect(apply.mock.calls[0]?.[0]).toMatchObject({
            nodeId: 'peer',
            modelChange: {
                previewDigest: 'a'.repeat(63) + '2',
                load: { set: { gpu: { mode: 'auto' } } }
            }
        })
        expect(set).not.toHaveBeenCalled()
    })
    it('centralizes tuning defaults and previews target-local placement with no source resolution authority', async () => {
        expect(
            defaultPortableLoadFields(inspected('self'), { slots: 2, gpu: { mode: 'cpu' } })
        ).toEqual(['slots', 'gpu'])
        await extendedPreview()
        expect(review).toHaveBeenCalledOnce()
        const request = review.mock.calls[0]?.[0]
        expect(request?.settings).toEqual(settings('peer').settings)
        expect(request?.modelChange?.load).toEqual({
            set: { slots: 2, gpu: { mode: 'layers', layers: 20 } },
            clear: []
        })
        expect(request?.modelChange?.policy).toEqual({ generation: { temperature: 0.25 } })
        expect(JSON.stringify(request)).not.toContain('source-native-id')
        expect(JSON.stringify(request)).not.toContain('/source/')
        expect(JSON.stringify(request)).not.toContain('source-build')
        expect(container.textContent).toContain('target-resolved finite placement')
        await click('Confirm copy to peer')
        expect(apply).toHaveBeenCalledOnce()
        expect(set).not.toHaveBeenCalled()
        expect(apply.mock.calls[0]?.[0]).toMatchObject({
            settings: settings('peer').settings,
            modelChange: { expectedPolicyRevision: '1', previewDigest: 'target-digest-peer' }
        })
    })
    it('requires explicit destructive role selection and shows target diff and served semantics', async () => {
        const routingSource = source.replace(
            '"profiles":',
            '"roles":{"weak":{"engine":"llamacpp","model":"model"}},"servedModels":[],"profiles":'
        )
        await render(routingSource)
        await click('Select portable load and routing preferences')
        await check('Explicitly copy weak')
        await check('Explicitly copy served')
        await check('Copy to peer')
        await click('Preview selected targets')
        expect(review.mock.calls[0]?.[0].modelChange?.policy).toMatchObject({
            roles: { weak: true },
            served: false
        })
        expect(container.textContent).toContain('can displace target tier')
        expect(container.textContent).toContain('target nil/empty semantics reviewed')
    })
    it('permits load-only copy into a generation-locked target', async () => {
        get.mockResolvedValue(snapshot({ localOnly: true }))
        await render()
        await click('Select portable load and routing preferences')
        await check('Temperature:')
        await check('Copy to peer')
        await click('Preview selected targets')
        expect(review).toHaveBeenCalledOnce()
        expect(review.mock.calls[0]?.[0].modelChange?.policy).toBeUndefined()
        await click('Confirm copy to peer')
        expect(apply).toHaveBeenCalledOnce()
        expect(set).not.toHaveBeenCalled()
    })
    it('copies portable load preferences without requiring a source generation profile', async () => {
        await render('{}')
        await click('Select portable load and routing preferences')
        await check('Copy to peer')
        await click('Preview selected targets')
        expect(review).toHaveBeenCalledOnce()
        expect(review.mock.calls[0]?.[0].modelChange?.policy).toBeUndefined()
        expect(review.mock.calls[0]?.[0].modelChange?.load?.set.slots).toBe(2)
        expect(set).not.toHaveBeenCalled()
    })
    it('preserves a readable paused source while current destination authority controls apply', async () => {
        const pausedSource = source.replace(
            '"profiles":',
            '"availability":"paused","pausedBy":"peer","profiles":'
        )
        await render(pausedSource)
        await click('Select portable load and routing preferences')
        await check('Copy to peer')
        await click('Preview selected targets')
        expect(review).toHaveBeenCalledOnce()
        await act(async () => useClusterInvitationsStore.setState({ remoteManagementContext: 2 }))
        expect(
            Array.from(container.querySelectorAll('button')).find(
                item => item.textContent === 'Confirm copy to peer'
            )
        ).toBeUndefined()
        expect(apply).not.toHaveBeenCalled()
    })
    it('hands missing content to its existing owner only explicitly, then requires fresh inspection/review', async () => {
        inspect.mockImplementation(async request => ({
            ...inspected(request.nodeId ?? 'self', request.modelId),
            status: request.nodeId === 'peer' ? 'missing' : 'resolved'
        }))
        review.mockResolvedValue({
            settings: settings('peer').settings,
            revision: 5,
            errors: { modelId: 'Missing content' },
            restart: false,
            rebind: false
        })
        await extendedPreview()
        expect(container.textContent).toContain('Missing content')
        expect(container.textContent).not.toContain('Close content copy owner')
        expect(apply).not.toHaveBeenCalled()
        await click('Copy model files')
        expect(container.textContent).toContain('Close content copy owner')
        const before = inspect.mock.calls.length
        const progress: EngineProgress = {
            engineType: 'llama-cpp',
            nodeId: 'peer',
            nodeName: 'peer',
            operation: 'copy',
            model: 'model',
            status: 'done',
            opId: 'content-op',
            sourceNodeId: 'self',
            destinationNodeId: 'peer',
            copyResult: {
                engine: 'llamacpp',
                model: 'model',
                files: 1,
                filesSkipped: 0,
                bytesTotal: 1,
                bytesCopied: 1,
                durability: 'confirmed'
            }
        }
        await act(async () =>
            useEngineProgressStore.setState({ progress: new Map([['content-op', progress]]) })
        )
        expect(inspect.mock.calls.length).toBeGreaterThan(before)
        expect(review).toHaveBeenCalledOnce()
        expect(apply).not.toHaveBeenCalled()
        expect(container.textContent).toContain('Fresh inspection requested')
        await click('Close content copy owner')
        expect(container.textContent).toContain('Preview again for a fresh target inspection')
    })
    it('keeps one target refusal independent of another reviewed target', async () => {
        useNodesStore.setState({
            nodes: new Map([
                ['peer', node('peer')],
                ['other', node('other')]
            ])
        })
        review.mockImplementation(async request => {
            if (request.nodeId === 'peer') throw new Error('not ready on peer')
            return {
                settings: settings('other').settings,
                revision: 5,
                policyRevision: '1',
                previewDigest: 'other-preview',
                errors: {},
                restart: false,
                rebind: false
            }
        })
        await render()
        await click('Select portable load and routing preferences')
        await check('Copy to peer')
        await check('Copy to other')
        await click('Preview selected targets')
        expect(container.textContent).toContain('not ready on peer')
        expect(button('Confirm copy to other').disabled).toBe(false)
        await click('Confirm copy to other')
        expect(apply).toHaveBeenCalledOnce()
        expect(apply.mock.calls[0]?.[0].nodeId).toBe('other')
        expect(set).not.toHaveBeenCalled()
    })
})
