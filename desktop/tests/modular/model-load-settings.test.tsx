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
import type { InspectModelResult, RequestedLoad } from '@/shared/types/model-config'
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
    '{"revision":"7","profiles":[{"key":"model","displayName":"test","engine":"llamacpp","model":"model","localOnly":true}]}'
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
        modelLoads: [
            { profileKey: 'model', modelId: 'model', requested: { slots: 3, cacheTypeK: 'f16' } }
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
            field: 'fit',
            label: 'Native fit',
            kind: 'enum',
            values: ['off', 'on'],
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
            field: 'fitCtx',
            label: 'Fit context',
            kind: 'integer',
            minimum: 0,
            maximum: 16777216,
            portable: true,
            supported: true
        },
        {
            field: 'extraOptions',
            label: 'Extra options',
            kind: 'options',
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
    vi.stubGlobal('CSS', { supports: () => false })
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
async function render(disabled = false, draft = policy, authority = 'current', saved = policy) {
    await act(async () =>
        root.render(
            <ModelLoadSettings
                target={target}
                modelId={identity.model}
                profileKey="model"
                policy={draft}
                savedPolicy={saved}
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
async function select(label: string, value: string) {
    const fieldLabel = Array.from(container.querySelectorAll('label')).find(
        item => item.textContent?.trim() === label
    )
    const field = fieldLabel && document.getElementById(fieldLabel.htmlFor)
    if (!(field instanceof HTMLButtonElement)) throw new Error(`Missing select: ${label}`)
    await act(async () => field.click())
    const popup = document.getElementById(field.getAttribute('aria-controls') ?? '')
    const option = Array.from(popup?.querySelectorAll<HTMLElement>('[role="option"]') ?? []).find(
        item => item.textContent?.trim() === value
    )
    if (!option) throw new Error(`Missing option: ${value}`)
    await act(async () => option.click())
}
function loadDraft() {
    return useEngineSettingsStore.getState().modelDrafts[modelSettingsKey(target, 'model')]?.change
        ?.load
}
function savedLoad(requested: RequestedLoad) {
    read.mockResolvedValue({
        ...snapshot,
        settings: {
            ...snapshot.settings,
            modelLoads: [{ profileKey: 'model', modelId: 'model', requested }]
        }
    })
}

describe('load editor and target impact', () => {
    it('keeps constructor unchanged until its own explicit membership patch', async () => {
        const declared = JSON.stringify({
            revision: '7',
            profiles: [{ key: 'model', engine: 'llamacpp', model: 'model', localOnly: true }],
            roles: [
                { name: 'coder', profiles: [] },
                { name: 'constructor', profiles: [] }
            ]
        })
        await render(false, declared, 'current', declared)
        const constructorSelection = () => {
            const label = Array.from(container.querySelectorAll('label')).find(
                item => item.textContent?.trim() === 'constructor role'
            )
            const control = label && document.getElementById(label.htmlFor)
            if (!(control instanceof HTMLButtonElement))
                throw new Error('Missing constructor role selector')
            return control.textContent?.trim()
        }
        expect(constructorSelection()).toBe('Select item')
        await select('coder role', 'Add this profile to role')
        expect(constructorSelection()).toBe('Select item')
        const memberships = () =>
            useEngineSettingsStore.getState().modelDrafts[modelSettingsKey(target, 'model')]?.change
                ?.policy?.roles
        await select('constructor role', 'Remove this profile from role')
        expect(constructorSelection()).toBe('Remove this profile from role')
        expect(memberships()).toEqual({ coder: true, constructor: false })
        await select('constructor role', 'Unchanged')
        expect(constructorSelection()).toBe('Select item')
        expect(memberships()).toEqual({ coder: true })
        await select('constructor role', 'Add this profile to role')
        expect(constructorSelection()).toBe('Add this profile to role')
        expect(memberships()).toEqual({ coder: true, constructor: true })
        await select('constructor role', 'Unchanged')
        expect(constructorSelection()).toBe('Select item')
    })

    it('offers only declared roles in saved order and reviews coder without changing load identity', async () => {
        const declared = JSON.stringify({
            revision: '7',
            profiles: [{ key: 'model', engine: 'llamacpp', model: 'model', localOnly: true }],
            roles: [
                { name: 'review', profiles: ['model'] },
                { name: 'coder', profiles: [] }
            ]
        })
        await render(false, declared, 'current', declared)
        expect(
            Array.from(container.querySelectorAll('label'))
                .filter(
                    label => document.getElementById(label.htmlFor) instanceof HTMLButtonElement
                )
                .map(item => item.textContent?.trim())
                .filter(label => label?.endsWith(' role'))
        ).toEqual(['review role', 'coder role'])
        expect(container.textContent).toContain('review role currently: Assigned')
        expect(container.textContent).toContain('coder role currently: Unassigned')
        expect(container.querySelector('input[aria-label="Role name"]')).toBeNull()
        expect(
            Array.from(container.querySelectorAll('button')).some(item =>
                /^(Add role|Create role|Rename|Remove role|Move up|Move down)$/.test(
                    item.textContent ?? ''
                )
            )
        ).toBe(false)
        await select('coder role', 'Add this profile to role')
        await click('Preview model changes')
        expect(review.mock.lastCall?.[0].modelChange).toMatchObject({
            modelId: 'model',
            profileKey: 'model',
            expectedPolicyRevision: '7',
            policy: { roles: { coder: true } }
        })
        expect(review.mock.lastCall?.[0].modelChange?.policy?.roles).toEqual({ coder: true })
        expect(window.pairApi.nodePolicy.set).not.toHaveBeenCalled()
        expect(apply).not.toHaveBeenCalled()
    })

    it('offers no built-in role or creation action when the node declares none', async () => {
        await render()
        expect(container.textContent).toContain(
            'No roles on this node; create one in node settings.'
        )
        expect(
            Array.from(container.querySelectorAll('button[aria-label]'))
                .map(item => item.getAttribute('aria-label'))
                .filter(label => label?.endsWith(' role'))
        ).toEqual([])
        expect(container.querySelector('input[aria-label="Role name"]')).toBeNull()
    })

    it('sends explicit removal without a load or policy patch', async () => {
        useEngineSettingsStore.getState().editModel(modelSettingsKey(target, 'model'), {
            modelId: 'model',
            profileKey: 'model',
            expectedPolicyRevision: '7',
            remove: true
        })
        await render()
        await click('Preview model changes')
        expect(review.mock.lastCall?.[0].modelChange).toEqual({
            modelId: 'model',
            profileKey: 'model',
            expectedPolicyRevision: '7',
            remove: true
        })
        expect(window.pairApi.nodePolicy.set).not.toHaveBeenCalled()
    })

    it('inspects and edits a sibling by key while keeping the shared artifact model id', async () => {
        read.mockResolvedValue({
            ...snapshot,
            settings: {
                ...snapshot.settings,
                modelLoads: [
                    { profileKey: 'small', modelId: 'model', requested: { slots: 1 } },
                    { profileKey: 'large', modelId: 'model', requested: { slots: 5 } }
                ]
            }
        })
        const siblings =
            '{"profiles":[{"key":"small","displayName":"Small","engine":"llamacpp","model":"small"},{"key":"large","displayName":"Large","engine":"llamacpp","model":"large"}]}'
        await act(async () =>
            root.render(
                <ModelLoadSettings
                    target={target}
                    modelId="model"
                    profileKey="large"
                    policy={siblings}
                    savedPolicy={siblings}
                    policyRevision="7"
                    authority="current"
                    disabled={false}
                />
            )
        )
        expect(inspect).toHaveBeenCalledWith({
            nodeId: 'self',
            engine: 'llamacpp',
            modelId: 'model',
            profileKey: 'large'
        })
        expect(document.querySelector<HTMLInputElement>('input[aria-label="Slots"]')?.value).toBe(
            '5'
        )
        await input('Slots', '6')
        await click('Preview model changes')
        expect(review.mock.lastCall?.[0].modelChange).toMatchObject({
            profileKey: 'large',
            modelId: 'model',
            load: { set: { slots: 6 } }
        })
        expect(
            useEngineSettingsStore.getState().modelDrafts[modelSettingsKey(target, 'small')]
        ).toBeUndefined()
    })

    it.each([
        { label: 'Fit target', value: '1024', set: { fitTarget: '1024' } },
        { label: 'Fit context', value: '8192', set: { fitCtx: 8192 } }
    ])(
        'independently edits saved native $label as a sparse patch',
        async ({ label, value, set }) => {
            const saved: RequestedLoad = { fit: 'on', fitTarget: '512', fitCtx: 4096 }
            savedLoad(saved)
            await render()
            await input(label, value)
            expect(loadDraft()).toEqual({ set, clear: [] })
            expect(button('Preview model changes').disabled).toBe(false)
            expect(saved).toEqual({ fit: 'on', fitTarget: '512', fitCtx: 4096 })
            await click('Preview model changes')
            expect(review).toHaveBeenCalledWith({
                ...target,
                expectedRevision: 3,
                settings: {
                    ...snapshot.settings,
                    modelLoads: [{ profileKey: 'model', modelId: 'model', requested: saved }]
                },
                modelChange: {
                    modelId: 'model',
                    profileKey: 'model',
                    expectedPolicyRevision: '7',
                    load: { set, clear: [] }
                }
            })
            expect(loadDraft()).toEqual({ set, clear: [] })
        }
    )
    it('edits native fit, margin and minimum context in the current load draft', async () => {
        await render()
        await select('Native fit', 'on')
        await input('Fit target', '512')
        await input('Fit context', '4096')
        expect(loadDraft()).toEqual({
            set: { fit: 'on', fitTarget: '512', fitCtx: 4096 },
            clear: []
        })
        expect(container.textContent).toContain(
            'Native fit: PAIR does not estimate memory for this configuration.'
        )
        await select('Native fit', 'off')
        expect(container.textContent).not.toContain(
            'Native fit: PAIR does not estimate memory for this configuration.'
        )
    })
    it.each([
        ['Fit target', '512,', 'Fit target: invalid value'],
        ['Fit target', '12345678', 'Fit target: invalid value'],
        ['Fit context', '-1', 'Fit context: invalid value'],
        ['Fit context', '16777217', 'Fit context: invalid value'],
        ['Fit context', '1.5', 'Fit context: invalid value']
    ])('blocks invalid %s value %s', async (label, value, error) => {
        await render()
        await input(label, value)
        expect(button('Preview model changes').disabled).toBe(true)
        expect(container.textContent).toContain(error)
    })
    it('shows the service GPU-selection estimate note', async () => {
        inspect.mockResolvedValue({ ...inspection, configurationKeyReason: 'gpu-selection' })
        await render()
        expect(container.textContent).toContain(
            'GPU selection: PAIR does not estimate memory for this configuration.'
        )
    })
    it('shows service-reported missing launch flags with their real flag spelling', async () => {
        inspect.mockResolvedValue({
            ...inspection,
            unreportedLaunchOptions: ['cache-ram', 'sleep-idle-seconds']
        })
        await render()
        expect(container.textContent).toContain(
            'This llama.cpp build does not report: --cache-ram, --sleep-idle-seconds. The engine may fail to start.'
        )
    })
    it('adds and removes extra rows through the existing draft and clears the last entry', async () => {
        await render()
        await click('Add extra option')
        expect(button('Preview model changes').disabled).toBe(true)
        await input('Extra option key 1', 'spec-type')
        await input('Extra option value 1', 'draft-mtp')
        expect(loadDraft()).toEqual({
            set: { extraOptions: { 'spec-type': 'draft-mtp' } },
            clear: []
        })
        expect(button('Preview model changes').disabled).toBe(false)
        await click('Remove extra option 1')
        expect(loadDraft()).toEqual({ set: {}, clear: ['extraOptions'] })
    })
    it('explicitly clears inherited empty options', async () => {
        await render()
        await click('Clear extra options')
        expect(loadDraft()).toEqual({ set: {}, clear: ['extraOptions'] })
    })
    it.each(['Bad-Key', '--flag', '-flag', 'a'.repeat(65)])(
        'blocks an invalid extra key %s while other fields remain editable',
        async key => {
            await render()
            await click('Add extra option')
            await input('Extra option key 1', key)
            await input('Slots', '2')
            expect(button('Preview model changes').disabled).toBe(true)
            expect(container.textContent).toContain('enter a key of 1 to 64')
            await input('Extra option key 1', 'fork-option')
            expect(button('Preview model changes').disabled).toBe(false)
        }
    )
    it.each([
        ['UTF-8 bytes', 'é'.repeat(513), 'at most 1024 UTF-8 bytes'],
        ['control character', 'value\u0085', 'control characters are not allowed'],
        ['invalid UTF-8', '\uD800', 'enter valid UTF-8 text'],
        ['hash', '#', 'cannot contain'],
        ['semicolon', ';', 'cannot contain'],
        ['dollar', '$', 'cannot contain'],
        ['percent', '%', 'cannot contain'],
        ['opening brace', '{', 'cannot contain'],
        ['closing brace', '}', 'cannot contain'],
        ['backtick', '`', 'cannot contain']
    ])('blocks extra value with %s', async (_name, value, error) => {
        await render()
        await click('Add extra option')
        await input('Extra option key 1', 'fork-option')
        await input('Extra option value 1', value)
        expect(button('Preview model changes').disabled).toBe(true)
        expect(container.textContent).toContain(error)
    })
    it('accepts the UTF-8 byte boundary and rejects duplicate keys', async () => {
        await render()
        await click('Add extra option')
        await input('Extra option key 1', 'fork-option')
        await input('Extra option value 1', '😀'.repeat(256))
        expect(button('Preview model changes').disabled).toBe(false)
        await click('Add extra option')
        await input('Extra option key 2', 'fork-option')
        expect(button('Preview model changes').disabled).toBe(true)
        expect(container.textContent).toContain('duplicate key fork-option')
        await click('Remove extra option 2')
        expect(button('Preview model changes').disabled).toBe(false)
    })
    it('limits the editor to 32 entries and permits removal at the limit', async () => {
        savedLoad({
            extraOptions: Object.fromEntries(
                Array.from({ length: 32 }, (_, index) => [`fork-${index}`, 'ok'])
            )
        })
        await render()
        expect(container.querySelectorAll('input[list]').length).toBe(32)
        expect(button('Add extra option').disabled).toBe(true)
        await click('Remove extra option 32')
        expect(button('Add extra option').disabled).toBe(false)
        expect(Object.keys(loadDraft()?.set.extraOptions ?? {})).toHaveLength(31)
    })
    it('uses choice hints and exposes build-reported key suggestions', async () => {
        inspect.mockResolvedValue({
            ...inspection,
            nativeOptions: [{ name: 'spec-type', kind: 'choice', choices: ['none', 'draft-mtp'] }]
        })
        await render()
        await click('Add extra option')
        await input('Extra option key 1', 'spec-type')
        expect(container.querySelector('input[aria-label="Extra option value 1"]')).toBeNull()
        expect(
            Array.from(container.querySelectorAll('datalist option')).map(option =>
                option.getAttribute('value')
            )
        ).toEqual(['spec-type'])
        await select('Extra option value 1', 'draft-mtp')
        expect(loadDraft()?.set.extraOptions).toEqual({ 'spec-type': 'draft-mtp' })
    })
    it('uses switch and numeric hints without imposing local native semantics', async () => {
        inspect.mockResolvedValue({
            ...inspection,
            nativeOptions: [
                { name: 'fork-switch', kind: 'switch' },
                { name: 'fork-number', kind: 'number' }
            ]
        })
        await render()
        await click('Add extra option')
        await input('Extra option key 1', 'fork-switch')
        const toggle = container.querySelector<HTMLInputElement>(
            'input[aria-label="Extra option value 1"]'
        )
        if (!toggle) throw new Error('Missing switch input')
        expect(toggle.type).toBe('checkbox')
        expect(toggle.checked).toBe(false)
        expect(loadDraft()?.set.extraOptions).toEqual({ 'fork-switch': 'false' })
        await act(async () => toggle.click())
        expect(loadDraft()?.set.extraOptions).toEqual({ 'fork-switch': 'true' })
        await click('Add extra option')
        await input('Extra option key 2', 'fork-number')
        expect(
            container.querySelector<HTMLInputElement>('input[aria-label="Extra option value 2"]')
                ?.type
        ).toBe('number')
        await input('Extra option value 2', '12')
        expect(loadDraft()?.set.extraOptions).toEqual({
            'fork-switch': 'true',
            'fork-number': '12'
        })
    })
    it('falls back to text for unknown hints and keeps per-key service errors visible', async () => {
        inspect.mockResolvedValue({ ...inspection, nativeOptions: [] })
        review.mockResolvedValue({
            ...preview,
            errors: {
                'extraOptions.fork-option': 'This llama.cpp build does not report --fork-option.'
            }
        })
        await render()
        await click('Add extra option')
        await input('Extra option key 1', 'fork-option')
        expect(
            container.querySelector<HTMLInputElement>('input[aria-label="Extra option value 1"]')
                ?.type
        ).toBe('text')
        await input('Extra option value 1', 'raw')
        expect(button('Preview model changes').disabled).toBe(false)
        await click('Preview model changes')
        expect(container.textContent).toContain(
            'extraOptions.fork-option: This llama.cpp build does not report --fork-option.'
        )
        expect(button('Apply reviewed model changes').disabled).toBe(true)
    })
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
                modelLoads: [{ profileKey: 'model', modelId: 'model', requested: { slots: 2 } }]
            },
            effectiveSettings: {
                ...snapshot.settings,
                modelLoads: [{ profileKey: 'model', modelId: 'model', requested: { slots: 1 } }]
            },
            serving: { status: 'configured' }
        }
        await act(async () =>
            root.render(
                <ModelSettingsPreview
                    profileKey="model"
                    inspection={inspection}
                    snapshot={reported}
                />
            )
        )
        expect(container.textContent).toContain('Desired load preferences: {"slots":2}')
        expect(container.textContent).toContain('Applied load preferences: {"slots":1}')
        expect(container.textContent).toContain('Observed model allocation: Unknown')
        await act(async () =>
            root.render(
                <ModelSettingsPreview
                    profileKey="model"
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
                    modelLoads: [{ profileKey: 'model', modelId: 'model', requested: { slots: 2 } }]
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
        expect(container.textContent).toContain('Observed engine: Stopped')
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
                    modelLoads: [
                        {
                            profileKey: 'model',
                            modelId: 'model',
                            requested: { slots: 2, cacheTypeK: 'f16' }
                        }
                    ]
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
            { role: 'companion', identity: 'mmproj', size: '2', assurance: 'bound' }
        ]
        await act(async () =>
            root.render(
                <ModelSettingsPreview
                    profileKey="model"
                    inspection={{ ...inspection, artifacts }}
                    preview={{ ...preview, impact: { restartScope: 'whole-router' } }}
                    snapshot={snapshot}
                />
            )
        )
        expect(container.textContent).toContain('Affected models: Unknown')
        expect(container.textContent).toContain('Drain bound: Unknown')
        expect(container.textContent).toContain('Active request interruption: Unknown')
        expect(container.textContent).toContain('Stopped engine behavior: Unknown')
        expect(container.textContent).toContain('Companion file: mmproj')
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
        expect(container.textContent).toContain('An already-stopped engine stays stopped')
    })
})
