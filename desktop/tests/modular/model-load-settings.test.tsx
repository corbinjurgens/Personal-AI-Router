// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type {
    EngineSettingsSnapshot,
    EngineSettingsPreview,
    EngineSettingsTarget
} from '@/shared/types/engine-settings'
import type { InspectModelResult } from '@/shared/types/model-config'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi } from '@/ui/api/pair-api'
import type { IEngineApi } from '@/ui/api/engine-api'
import ModelLoadSettings from '@/ui/components/ModelSettings/ModelLoadSettings'
import ModelSettingsPreview from '@/ui/components/ModelSettings/ModelSettingsPreview'
import ModelSettingsPanel from '@/ui/components/ModelSettings/ModelSettingsPanel'
import { modelSettingsKey, useEngineSettingsStore } from '@/ui/stores/engine-settings.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'

const target = { nodeId: 'self', engine: 'llamacpp' } satisfies EngineSettingsTarget
const identity = { engine: 'llamacpp', model: 'model' }
const policy =
    '{"revision":"7","profiles":[{"name":"test","engine":"llamacpp","model":"model","localOnly":true}]}'
const snapshot: EngineSettingsSnapshot = {
    ...target,
    revision: 3,
    appliedRevision: 2,
    sequence: 1,
    epoch: 'authority',
    settings: {
        serverPort: 1,
        proxyPort: 2,
        launchText: '',
        modelLoads: [{ modelId: 'model', requested: { slots: 3, cacheTypeK: 'f16' } }]
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
    requestId: '',
    policyRevision: '7'
}
const inspection: InspectModelResult = {
    ...target,
    modelId: 'model',
    status: 'unavailable',
    source: 'saved preset',
    assurance: 'local binding',
    estimate: { status: 'unknown' },
    options: [
        {
            field: 'context',
            label: 'Tokens per request',
            kind: 'integer',
            minimum: 256,
            maximum: 32768,
            default: 8192,
            defaultSource: 'finite adapter inherited baseline',
            portable: true,
            supported: true
        },
        {
            field: 'slots',
            label: 'Slots',
            kind: 'integer',
            minimum: 1,
            maximum: 8,
            default: 1,
            defaultSource: 'native adapter',
            portable: true,
            supported: true
        },
        {
            field: 'cacheTypeK',
            label: 'K cache precision',
            kind: 'enum',
            values: ['f16'],
            portable: true,
            supported: true
        },
        {
            field: 'cacheTypeV',
            label: 'V cache precision',
            kind: 'enum',
            values: ['f16'],
            portable: true,
            supported: true
        },
        {
            field: 'native.device',
            label: 'Native devices',
            kind: 'string',
            portable: false,
            supported: false,
            reason: 'Mapping unavailable'
        }
    ]
}
const preview: EngineSettingsPreview = {
    settings: snapshot.settings,
    revision: 3,
    policyRevision: '7',
    previewDigest: 'digest',
    errors: {},
    restart: true,
    rebind: false,
    impact: {
        restartScope: 'whole-router',
        affectedModels: ['model', 'other'],
        drainTimeoutSeconds: 60,
        mayInterruptActive: true,
        staysOff: true
    }
}
let root: Root
let container: HTMLDivElement
const inspect = vi.fn<IEngineApi['inspectModel']>()
const read = vi.fn<IEngineApi['getSettings']>()
const review = vi.fn<IEngineApi['previewSettings']>()
const apply = vi.fn<IEngineApi['applySettings']>()
beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    inspect.mockReset().mockResolvedValue(inspection)
    read.mockReset().mockResolvedValue(snapshot)
    review.mockReset().mockRejectedValue(new Error('modelChange: application is not ready'))
    apply.mockReset().mockResolvedValue({ revision: 4, phase: 'applying' })
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
    api.engines.getSettings = read
    api.engines.inspectModel = inspect
    api.engines.previewSettings = review
    api.engines.applySettings = apply
    api.nodePolicy.get = vi.fn<typeof api.nodePolicy.get>().mockResolvedValue({
        policy,
        revision: '7',
        availability: 'available',
        activation: null,
        recoveryRequired: false
    })
    api.nodePolicy.set = vi.fn<typeof api.nodePolicy.set>()
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    useConnectionStore.setState({ connected: true, selfId: 'self' })
    useEngineSettingsStore.setState({ entries: {}, modelDrafts: {} })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})
afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.unstubAllGlobals()
})
async function render(disabled = false, draft = policy, authority = 'current') {
    await act(async () =>
        root.render(
            <ModelLoadSettings
                target={target}
                modelId={identity.model}
                policy={draft}
                savedPolicy={policy}
                policyRevision="7"
                authority={authority}
                disabled={disabled}
            />
        )
    )
}
function button(label: string) {
    const found = Array.from(document.querySelectorAll('button')).find(
        item => item.textContent === label
    )
    if (!found) throw new Error(`Missing ${label}`)
    return found
}
async function click(label: string) {
    await act(async () => button(label).click())
}
async function input(label: string, value: string) {
    const field = document.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
    if (!field || !setter) throw new Error(`Missing ${label}`)
    await act(async () => {
        setter.call(field, value)
        field.dispatchEvent(new Event('input', { bubbles: true }))
    })
}

describe('load editor and target impact', () => {
    it('keeps an invalid context blocking review when another field is edited', async () => {
        await render()
        await input('Tokens per request', '1.5')
        await input('Slots', '2')
        expect(button('Preview model changes').disabled).toBe(true)
        expect(container.textContent).toContain('Tokens per request: invalid value')
        await input('Tokens per request', '8192')
        expect(button('Preview model changes').disabled).toBe(false)
        await input('Tokens per request', '1.5')
        await click('Reset Tokens per request to inherited')
        expect(button('Preview model changes').disabled).toBe(false)
    })
    it('separates requested, applied and observed allocation without treating configuration as observation', async () => {
        const reported: EngineSettingsSnapshot = {
            ...snapshot,
            settings: {
                ...snapshot.settings,
                modelLoads: [{ modelId: 'model', requested: { slots: 2 } }]
            },
            effectiveSettings: {
                ...snapshot.settings,
                modelLoads: [{ modelId: 'model', requested: { slots: 1 } }]
            },
            serving: { status: 'configured' }
        }
        await act(async () =>
            root.render(<ModelSettingsPreview inspection={inspection} snapshot={reported} />)
        )
        expect(container.textContent).toContain('Desired load preferences: {"slots":2}')
        expect(container.textContent).toContain('Applied load preferences: {"slots":1}')
        expect(container.textContent).toContain('Observed model allocation: Unknown')
        await act(async () =>
            root.render(
                <ModelSettingsPreview
                    inspection={inspection}
                    snapshot={{
                        ...reported,
                        serving: {
                            status: 'observed',
                            observed: [{ modelId: 'model', perSlotContextTokens: 4096, slots: 1 }]
                        }
                    }}
                />
            )
        )
        expect(container.textContent).toContain(
            'Observed model allocation: 4096 tokens per request, 1 slots'
        )
    })
    it('re-enables the existing policy Save after unchanged load edits complete', async () => {
        vi.stubGlobal('CSS', { supports: () => false, escape: (value: string) => value })
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
        useNodeConfigurationStore.setState({ policyDrafts: new Map(), management: new Map() })
        const policies = useNodeConfigurationStore.getState()
        policies.claimPolicyDraft('self', 'Routing')
        policies.observePolicyDraft('self', {
            policy,
            revision: '7',
            activation: null,
            availability: 'available'
        })
        const savePolicy = vi.fn<() => void>()
        review.mockResolvedValue(preview)
        apply.mockImplementation(async request => {
            read.mockResolvedValue({
                ...snapshot,
                sequence: 2,
                revision: 4,
                appliedRevision: 4,
                requestId: request.requestId ?? '',
                phase: 'succeeded',
                settings: {
                    ...snapshot.settings,
                    modelLoads: [{ modelId: 'model', requested: { slots: 2 } }]
                }
            })
            return { revision: 4, phase: 'succeeded' }
        })
        await act(async () =>
            root.render(
                <ModelSettingsPanel
                    targetId="self"
                    nodeName="This PC"
                    identity={identity}
                    open
                    routing={{
                        text: policy,
                        savedText: policy,
                        onTextChange: () => {},
                        disabled: false,
                        onValidityChange: () => {},
                        onSave: savePolicy,
                        canSave: true,
                        onReload: () => {}
                    }}
                />
            )
        )
        await click('Configure load and routing settings')
        await input('Slots', '2')
        expect(button('Save').disabled).toBe(true)
        await click('Preview model changes')
        await click('Apply reviewed model changes')
        expect(button('Save').disabled).toBe(false)
        await click('Save')
        expect(savePolicy).toHaveBeenCalledOnce()
        expect(apply).toHaveBeenCalledOnce()
    })
    it('shows descriptor bounds/provenance and keeps manual controls beside unknown facts', async () => {
        await render()
        const context = container.querySelector<HTMLInputElement>(
            'input[aria-label="Tokens per request"]'
        )
        expect(context?.disabled).toBe(false)
        expect(context?.min).toBe('256')
        expect(context?.max).toBe('32768')
        expect(container.textContent).toContain('finite adapter inherited baseline')
        expect(container.textContent).toContain('Saved native total allocation: Unknown')
        expect(container.textContent).toContain('Mapping unavailable')
        expect(container.textContent).toContain('Observed engine: Off')
        expect(container.textContent).toContain(
            'saving settings never loads a model'.replace('saving', 'Saving')
        )
    })
    it('edits load settings with saved generation lock without fabricating a native total', async () => {
        await render()
        await input('Slots', '2')
        const draft =
            useEngineSettingsStore.getState().modelDrafts[modelSettingsKey(target, 'model')]
        expect(draft.change?.load).toEqual({ set: { slots: 2 }, clear: [] })
        expect(container.textContent).toContain('Saved native total allocation: Unknown')
    })
    it('resets only the selected leaf to the inherited source', async () => {
        await render()
        await click('Reset Slots to inherited')
        expect(
            useEngineSettingsStore.getState().modelDrafts[modelSettingsKey(target, 'model')].change
                ?.load
        ).toEqual({ set: {}, clear: ['slots'] })
    })
    it('keeps Apply disabled after actual not-ready refusal, even with supported controls', async () => {
        await render()
        await input('Slots', '2')
        await click('Preview model changes')
        expect(review).toHaveBeenCalledOnce()
        expect(button('Apply reviewed model changes').disabled).toBe(true)
        expect(container.textContent).toContain('application is not ready')
        expect(apply).not.toHaveBeenCalled()
    })
    it('applies the reviewed combined load/generation click exactly once and invalidates on later edit', async () => {
        review.mockResolvedValue(preview)
        apply.mockImplementation(async request => {
            read.mockResolvedValue({
                ...snapshot,
                sequence: 2,
                revision: 4,
                appliedRevision: 4,
                requestId: request.requestId ?? '',
                phase: 'succeeded',
                settings: {
                    ...snapshot.settings,
                    modelLoads: [{ modelId: 'model', requested: { slots: 2, cacheTypeK: 'f16' } }]
                }
            })
            return { revision: 4, phase: 'succeeded' }
        })
        await render(
            false,
            policy.replace(
                '"localOnly":true',
                '"localOnly":true,"requestOptions":{"temperature":0.2}'
            )
        )
        await input('Slots', '2')
        await click('Preview model changes')
        expect(button('Apply reviewed model changes').disabled).toBe(false)
        expect(container.textContent).toContain('Restarts the whole llama.cpp engine')
        await click('Apply reviewed model changes')
        expect(apply).toHaveBeenCalledOnce()
        expect(apply.mock.calls[0]?.[0].modelChange).toMatchObject({
            load: { set: { slots: 2 } },
            policy: { generation: { temperature: 0.2 } }
        })
        expect(window.pairApi.nodePolicy.set).not.toHaveBeenCalled()
        expect(
            useEngineSettingsStore.getState().modelDrafts[modelSettingsKey(target, 'model')].change
        ).toBeUndefined()
        await input('Slots', '4')
        expect(button('Apply reviewed model changes').disabled).toBe(true)
    })
    it('retains a dirty draft when permission/paused context disables editing', async () => {
        await render()
        await input('Slots', '2')
        await render(true, policy, 'paused/revoked')
        expect(
            container.querySelector<HTMLInputElement>('input[aria-label="Slots"]')?.disabled
        ).toBe(true)
        expect(button('Apply reviewed model changes').disabled).toBe(true)
        expect(
            useEngineSettingsStore.getState().modelDrafts[modelSettingsKey(target, 'model')].change
                ?.load?.set.slots
        ).toBe(2)
    })
    it('reviews current generation edits after an earlier combined preview without replaying its generation patch', async () => {
        review.mockResolvedValue(preview)
        const first = policy.replace(
            '"localOnly":true',
            '"localOnly":true,"requestOptions":{"temperature":0.2}'
        )
        await render(false, first)
        await input('Slots', '2')
        await click('Preview model changes')
        const second = first.replace('"temperature":0.2', '"temperature":0.4')
        await render(false, second)
        expect(button('Apply reviewed model changes').disabled).toBe(true)
        await click('Preview model changes')
        expect(review.mock.calls.at(-1)?.[0].modelChange?.policy?.generation).toEqual({
            temperature: 0.4
        })
    })
    it('shows omitted impact fields as Unknown and companion artifacts as separate files', async () => {
        const artifacts: InspectModelResult['artifacts'] = [
            { role: 'model', identity: 'main', size: '9007199254740993', assurance: 'bound' },
            { role: 'draft', identity: 'draft', size: '1', assurance: 'bound' },
            { role: 'companion', identity: 'mmproj', size: '2', assurance: 'bound' },
            { role: 'future-role', identity: 'future', size: '3', assurance: 'unknown' }
        ]
        await act(async () =>
            root.render(
                <ModelSettingsPreview
                    inspection={{ ...inspection, artifacts }}
                    preview={{ ...preview, impact: { restartScope: 'whole-router' } }}
                    snapshot={snapshot}
                />
            )
        )
        expect(container.textContent).toContain('Affected models: Unknown')
        expect(container.textContent).toContain('Drain bound: Unknown')
        expect(container.textContent).toContain('Active request interruption: Unknown')
        expect(container.textContent).toContain('Off behavior: Unknown')
        expect(container.textContent).toContain('Companion file: mmproj')
        expect(container.textContent).toContain('Artifact file (future-role): future')
        expect(container.textContent).toContain('9007199254740993 bytes')
        expect(container.textContent?.indexOf('Main model file')).toBeLessThan(
            container.textContent?.indexOf('Draft model file') ?? -1
        )
        expect(container.textContent?.indexOf('Draft model file')).toBeLessThan(
            container.textContent?.indexOf('Companion file') ?? -1
        )
    })
    it('lists whole-router affected models and the reported interruption bound', async () => {
        await act(async () => root.render(<ModelSettingsPreview preview={preview} />))
        expect(container.textContent).toContain('Restarts the whole llama.cpp engine')
        expect(container.textContent).toContain('Affected models: model, other')
        expect(container.textContent).toContain('Drain bound: 60 seconds')
        expect(container.textContent).toContain('Active requests can be interrupted')
        expect(container.textContent).toContain('Off remains Off')
    })
})
