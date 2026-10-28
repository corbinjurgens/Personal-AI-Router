// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { renderToStaticMarkup } from 'react-dom/server'
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ModelSettingsPreview from '@/ui/components/ModelSettings/ModelSettingsPreview'
import ModelLoadSettings from '@/ui/components/ModelSettings/ModelLoadSettings'
import type {
    EngineSettingsPreview,
    EngineSettingsSnapshot,
    EngineSettingsTarget
} from '@/shared/types/engine-settings'
import type { InspectModelResult } from '@/shared/types/model-config'
import type { IEngineApi } from '@/ui/api/engine-api'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi } from '@/ui/api/pair-api'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { modelSettingsKey, useEngineSettingsStore } from '@/ui/stores/engine-settings.store'
import { parseEngineSettingsPreview } from '@/electron/service-bridge/engine-settings'

it('shows requested Auto constraints beside the finite target proposal', () => {
    const preview = parseEngineSettingsPreview({
        settings: { serverPort: 1, proxyPort: 2, launchText: '' },
        revision: 1,
        errors: {},
        restart: false,
        rebind: false,
        auto: {
            status: 'available',
            requested: { context: { mode: 'auto', floor: 4096, cap: 8192 }, slots: 1 },
            estimate: { status: 'unknown' },
            resolverRevision: 'auto/fixture',
            reviewedInputId: 'input',
            resolved: {
                perSlotContext: 4096,
                totalContext: 4096,
                slots: 1,
                gpu: { mode: 'cpu' },
                options: {},
                artifacts: [
                    {
                        role: 'model',
                        path: '/model.gguf',
                        resolvedTarget: '/model.gguf',
                        size: '1024',
                        mtimeNs: '1',
                        sampleSha256: 'a'.repeat(64)
                    }
                ],
                nativeBuildId: 'build',
                sourceInputDigest: 'b'.repeat(64),
                presetDigest: 'c'.repeat(64),
                resolverRevision: 'auto/fixture',
                reviewedInputId: 'input'
            }
        }
    })
    const html = renderToStaticMarkup(<ModelSettingsPreview preview={preview} reviewed={false} />)
    expect(html).toContain('8192')
    expect(html).toContain('Feasible context: 4096')
    expect(html).toContain('CPU placement can be slower')
    expect(html).toContain('Advisory, not reviewed')
})

describe('mounted Auto editor', () => {
    const target: EngineSettingsTarget = { nodeId: 'self', engine: 'llamacpp' }
    const key = modelSettingsKey(target, 'model')
    const policy = '{"profiles":[{"name":"test","engine":"llamacpp","model":"model"}]}'
    const saved: EngineSettingsSnapshot = {
        ...target,
        revision: 1,
        appliedRevision: 1,
        epoch: 'fixture',
        sequence: 1,
        settings: { serverPort: 1, proxyPort: 2, launchText: '' },
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
    const inspected: InspectModelResult = {
        ...target,
        modelId: 'model',
        status: 'resolved',
        estimate: { status: 'unavailable' },
        options: [
            {
                field: 'context',
                label: 'Tokens per request',
                kind: 'integer',
                minimum: 256,
                maximum: 32768,
                supported: true,
                portable: true
            },
            {
                field: 'slots',
                label: 'Slots',
                kind: 'integer',
                minimum: 1,
                supported: true,
                portable: true
            },
            {
                field: 'gpu',
                label: 'GPU placement',
                kind: 'enum',
                values: ['cpu', 'layers', 'auto'],
                supported: true,
                portable: true
            }
        ]
    }
    const reply = parseEngineSettingsPreview(
        JSON.parse(
            JSON.stringify({
                settings: saved.settings,
                revision: 1,
                policyRevision: '7',
                previewDigest: 'd'.repeat(64),
                errors: {},
                restart: false,
                rebind: false,
                auto: {
                    status: 'available',
                    requested: { context: { mode: 'auto', floor: 4096, cap: 8192 } },
                    estimate: { status: 'estimated' },
                    resolverRevision: 'auto/fixture',
                    reviewedInputId: 'input',
                    resolved: {
                        perSlotContext: 4096,
                        totalContext: 4096,
                        slots: 1,
                        gpu: { mode: 'cpu' },
                        options: {},
                        artifacts: [
                            {
                                role: 'model',
                                path: '/model.gguf',
                                resolvedTarget: '/model.gguf',
                                size: '1024',
                                mtimeNs: '1',
                                sampleSha256: 'a'.repeat(64)
                            }
                        ],
                        nativeBuildId: 'build',
                        sourceInputDigest: 'b'.repeat(64),
                        presetDigest: 'c'.repeat(64),
                        resolverRevision: 'auto/fixture',
                        reviewedInputId: 'input'
                    }
                }
            })
        )
    )
    const preview = vi.fn<IEngineApi['previewSettings']>()
    const apply = vi.fn<IEngineApi['applySettings']>()
    let root: Root
    let container: HTMLDivElement
    async function mount(modelId = 'model') {
        await act(async () =>
            root.render(
                <ModelLoadSettings
                    target={target}
                    modelId={modelId}
                    policy={policy}
                    savedPolicy={policy}
                    policyRevision="7"
                    authority="authority"
                    disabled={false}
                />
            )
        )
    }
    function button(label: string) {
        const found = Array.from(container.querySelectorAll('button')).find(
            item => item.textContent === label
        )
        if (!found) throw new Error(`Missing button: ${label}`)
        return found
    }
    async function select(label: string, value: string) {
        const fieldLabel = Array.from(container.querySelectorAll('label')).find(
            item => item.textContent?.trim() === label
        )
        const field = fieldLabel && document.getElementById(fieldLabel.htmlFor)
        if (!(field instanceof HTMLButtonElement)) throw new Error(`Missing select: ${label}`)
        await act(async () => field.click())
        const popup = document.getElementById(field.getAttribute('aria-controls') ?? '')
        const option = Array.from(
            popup?.querySelectorAll<HTMLElement>('[role="option"]') ?? []
        ).find(item => item.textContent?.trim().toLowerCase() === value)
        if (!option) throw new Error(`Missing option: ${value}`)
        await act(async () => option.click())
    }
    async function input(label: string, value: string) {
        const fieldLabel = Array.from(container.querySelectorAll('label')).find(
            item => item.textContent?.trim() === label
        )
        const field = fieldLabel && document.getElementById(fieldLabel.htmlFor)
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
        if (!(field instanceof HTMLInputElement) || !setter)
            throw new Error(`Missing input: ${label}`)
        await act(async () => {
            setter.call(field, value)
            field.dispatchEvent(new Event('input', { bubbles: true }))
        })
    }
    beforeEach(async () => {
        vi.useFakeTimers()
        vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
        vi.stubGlobal('CSS', { supports: () => false })
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
        api.engines.getSettings = vi.fn<IEngineApi['getSettings']>().mockResolvedValue(saved)
        api.engines.inspectModel = vi
            .fn<IEngineApi['inspectModel']>()
            .mockImplementation(async request => ({ ...inspected, modelId: request.modelId }))
        api.engines.onSettingsChanged = () => () => {}
        api.engines.onSettingsDisconnected = () => () => {}
        preview.mockReset().mockResolvedValue(reply)
        apply.mockReset()
        api.engines.previewSettings = preview
        api.engines.applySettings = apply
        api.nodePolicy.get = vi.fn<typeof api.nodePolicy.get>().mockResolvedValue({
            policy,
            revision: '7',
            activation: null,
            availability: 'available',
            recoveryRequired: false
        })
        Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
        useConnectionStore.setState({ connected: true, selfId: 'self' })
        useEngineSettingsStore.setState({ entries: {}, modelDrafts: {} })
        container = document.createElement('div')
        document.body.append(container)
        root = createRoot(container)
        await mount()
    })
    afterEach(async () => {
        await act(async () => root.unmount())
        container.remove()
        vi.useRealTimers()
        vi.unstubAllGlobals()
    })
    it('keeps a slow explicit Recalculate authoritative when the advisory debounce fires', async () => {
        let finish: ((value: EngineSettingsPreview) => void) | undefined
        const response = new Promise<EngineSettingsPreview>(resolve => {
            finish = resolve
        })
        preview.mockReturnValueOnce(response)
        await select('Context mode', 'auto')
        await act(async () => button('Recalculate').click())
        expect(preview).toHaveBeenCalledTimes(1)
        await act(async () => vi.advanceTimersByTime(250))
        await act(async () => {
            if (!finish) throw new Error('Missing response barrier')
            finish(reply)
        })
        expect(preview).toHaveBeenCalledTimes(1)
        expect(button('Apply reviewed model changes').disabled).toBe(false)
        expect(
            useEngineSettingsStore.getState().modelDrafts[key].review?.preview.previewDigest
        ).toBe(reply.previewDigest)
        expect(
            useEngineSettingsStore.getState().entries['self:llamacpp'].snapshot?.settings
        ).toEqual(saved.settings)
        expect(apply).not.toHaveBeenCalled()
    })
    it('clears Auto-only validation on switching to Fixed and preserves unrelated validation', async () => {
        await select('Context mode', 'auto')
        await input('Auto context cap', '')
        expect(container.textContent).toContain('Auto cap: enter a positive integer.')
        await select('Context mode', 'fixed')
        expect(container.textContent).not.toContain('Auto cap: enter a positive integer.')
        expect(button('Preview model changes').disabled).toBe(false)
        await input('Slots', '1.5')
        await select('Context mode', 'auto')
        await select('Context mode', 'fixed')
        expect(container.textContent).toContain('Slots: invalid value')
        expect(button('Preview model changes').disabled).toBe(true)
    })
    it('clears replaced Fixed validation on switching to Auto but preserves Slots errors', async () => {
        await select('Context mode', 'fixed')
        await input('Tokens per request', '1.5')
        expect(container.textContent).toContain('Tokens per request: invalid value')
        await select('Context mode', 'auto')
        expect(container.textContent).not.toContain('Tokens per request: invalid value')
        expect(button('Recalculate').disabled).toBe(false)
        await select('Context mode', 'fixed')
        await input('Slots', '1.5')
        await select('Context mode', 'auto')
        expect(container.textContent).toContain('Slots: invalid value')
        expect(button('Recalculate').disabled).toBe(true)
    })
    it('does not use the inspection baseline estimate for a changed draft without an estimate', async () => {
        const baseline: InspectModelResult = {
            ...inspected,
            estimate: {
                status: 'estimated',
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
                            ({
                                name,
                                status: 'excluded',
                                reason: 'Not present'
                            }) satisfies NonNullable<
                                NonNullable<InspectModelResult['estimate']>['components']
                            >[number]
                    )
                ]
            }
        }
        window.pairApi.engines.inspectModel = vi
            .fn<IEngineApi['inspectModel']>()
            .mockResolvedValue(baseline)
        preview.mockResolvedValue({ ...reply, auto: undefined, estimate: undefined })
        await select('Context mode', 'fixed')
        await input('Tokens per request', '16384')
        await act(async () => button('Preview model changes').click())
        expect(container.textContent).not.toContain('(Fits)')
        expect(container.textContent).toContain('Memory domains: Unknown')
        expect(button('Apply reviewed model changes').disabled).toBe(false)
    })
    it('shows Auto defaults and edits a smaller cap and explicit floor without saving', async () => {
        await select('Context mode', 'auto')
        expect(
            useEngineSettingsStore.getState().modelDrafts[key].change?.load?.set.context
        ).toEqual({ mode: 'auto', floor: 4096, cap: 8192 })
        await input('Auto context cap', '2048')
        expect(
            useEngineSettingsStore.getState().modelDrafts[key].change?.load?.set.context
        ).toEqual({ mode: 'auto', floor: 2048, cap: 2048 })
        await input('Auto context floor', '1024')
        await select('GPU placement', 'cpu')
        const auto = reply.auto
        if (!auto?.resolved) throw new Error('Finite Auto fixture missing')
        const smaller: EngineSettingsPreview = {
            ...reply,
            auto: {
                ...auto,
                requested: {
                    context: { mode: 'auto', floor: 1024, cap: 2048 },
                    gpu: { mode: 'cpu' }
                },
                resolved: { ...auto.resolved, perSlotContext: 1024, totalContext: 1024 }
            }
        }
        preview.mockResolvedValueOnce(
            parseEngineSettingsPreview(JSON.parse(JSON.stringify(smaller)))
        )
        await act(async () => button('Recalculate').click())
        expect(preview.mock.calls[0]?.[0].modelChange?.load?.set).toMatchObject({
            context: { mode: 'auto', floor: 1024, cap: 2048 },
            gpu: { mode: 'cpu' }
        })
        expect(container.textContent).toContain('Feasible context: 1024')
        expect(container.textContent).toContain('CPU placement can be slower')
        expect(apply).not.toHaveBeenCalled()
    })
    it('keeps manual review usable when Auto estimates are unavailable', async () => {
        preview.mockResolvedValue({
            ...reply,
            auto: { status: 'unavailable', requested: {}, estimate: { status: 'unavailable' } }
        })
        await select('Context mode', 'auto')
        await act(async () => button('Recalculate').click())
        expect(button('Apply reviewed model changes').disabled).toBe(true)
        expect(container.textContent).toContain('Auto resolution unavailable')
        await select('Context mode', 'fixed')
        await act(async () => button('Preview model changes').click())
        expect(button('Apply reviewed model changes').disabled).toBe(false)
    })
    it('discards a late review after changing the selected model', async () => {
        let finish: ((value: EngineSettingsPreview) => void) | undefined
        preview.mockReturnValueOnce(
            new Promise(resolve => {
                finish = resolve
            })
        )
        await select('Context mode', 'auto')
        await act(async () => button('Recalculate').click())
        await mount('other')
        await act(async () => {
            if (!finish) throw new Error('Missing response barrier')
            finish(reply)
        })
        expect(useEngineSettingsStore.getState().modelDrafts[key].review).toBeUndefined()
        expect(button('Apply reviewed model changes').disabled).toBe(true)
        expect(apply).not.toHaveBeenCalled()
    })
})
