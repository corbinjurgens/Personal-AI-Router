// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { EngineSettingsTarget } from '@/shared/types/engine-settings'
import type { InspectModelResult } from '@/shared/types/model-config'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { POLICY_CONFLICT_CODE } from '@/shared/constants/service-errors'
import { createPairApi } from '@/ui/api/pair-api'
import type { IEngineApi } from '@/ui/api/engine-api'
import { ServiceCallError } from '@/ui/api/service-transport'
import ModelMemoryStatedFigure from '@/ui/components/ModelSettings/ModelMemoryStatedFigure'
import ModelSettingsPreview from '@/ui/components/ModelSettings/ModelSettingsPreview'
import { modelSettingsKey, useEngineSettingsStore } from '@/ui/stores/engine-settings.store'

const target: EngineSettingsTarget = { engine: 'llamacpp', nodeId: 'self' }
const key = modelSettingsKey(target)
const applied: InspectModelResult = {
    ...target,
    modelId: 'model',
    status: 'unavailable',
    estimate: { status: 'unknown' },
    configurationKey: 'applied-key',
    configurationPools: ['host', 'device']
}
const save = vi.fn<IEngineApi['setStatedMemory']>()
const clear = vi.fn<IEngineApi['clearStatedMemory']>()
const inspect = vi.fn<IEngineApi['inspectModel']>()
let root: Root
let container: HTMLDivElement

function Editor({
    selected = target,
    profileKey
}: {
    selected?: EngineSettingsTarget
    profileKey?: string
}) {
    const inspection = useEngineSettingsStore(
        state => state.modelDrafts[modelSettingsKey(selected, profileKey)]?.inspection
    )
    return (
        <ModelMemoryStatedFigure
            key={`${selected.nodeId}:${profileKey ?? ''}`}
            target={selected}
            profileKey={profileKey}
            inspection={inspection}
        />
    )
}

beforeEach(() => {
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
    save.mockReset().mockImplementation(async request => ({
        ...applied,
        memoryProfiles: [
            { ...request, kind: 'stated', recordedAt: '2026-11-07T01:00:00Z', current: true }
        ]
    }))
    clear.mockReset().mockResolvedValue(applied)
    inspect.mockReset()
    api.engines.setStatedMemory = save
    api.engines.clearStatedMemory = clear
    api.engines.inspectModel = inspect
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    useEngineSettingsStore.setState({
        entries: {},
        modelDrafts: { [key]: { generation: 1, inspection: applied } }
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
async function render() {
    await act(async () => root.render(<Editor />))
}
function field(label: string) {
    const input = container.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)
    if (!input) throw new Error(`Missing ${label}`)
    return input
}
async function input(label: string, value: string) {
    const element = field(label)
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
    if (!setter) throw new Error('Missing input value setter')
    await act(async () => {
        setter.call(element, value)
        element.dispatchEvent(new Event('input', { bubbles: true }))
    })
}
async function click(label: string) {
    const button = Array.from(container.querySelectorAll('button')).find(
        item => item.textContent === label
    )
    if (!button) throw new Error(`Missing ${label}`)
    await act(async () => button.click())
}
function seed(inspection: InspectModelResult) {
    useEngineSettingsStore.setState({ modelDrafts: { [key]: { generation: 1, inspection } } })
}

describe('applied configuration stated memory', () => {
    it('keeps the named profile in UI state while Save and Clear send only Task8 wire fields', async () => {
        const profileKey = 'profile-a'
        const profiledKey = modelSettingsKey(target, profileKey)
        useEngineSettingsStore.setState({
            modelDrafts: { [profiledKey]: { generation: 1, inspection: applied } }
        })
        await act(async () => root.render(<Editor profileKey={profileKey} />))
        await input('Host memory (GiB)', '1')
        await input('Device memory (GiB)', '2')
        await click('Save figure')
        expect(save).toHaveBeenCalledExactlyOnceWith({
            ...target,
            modelId: 'model',
            configurationKey: 'applied-key',
            hostBytes: '1073741824',
            deviceBytes: '2147483648'
        })
        expect(
            useEngineSettingsStore.getState().modelDrafts[profiledKey].inspection
                ?.memoryProfiles?.[0].current
        ).toBe(true)
        await click('Clear figure')
        expect(clear).toHaveBeenCalledExactlyOnceWith({
            ...target,
            modelId: 'model',
            configurationKey: 'applied-key'
        })
        expect(
            useEngineSettingsStore.getState().modelDrafts[profiledKey].inspection?.memoryProfiles
        ).toBeUndefined()
    })

    it('saves a first figure with exact fractional GiB and an unknown estimate without a stored profile', async () => {
        await render()
        expect(container.textContent).toContain('I know how much this uses')
        await input('Host memory (GiB)', '1.5')
        await input('Device memory (GiB)', '0.125')
        await click('Save figure')
        expect(save).toHaveBeenCalledExactlyOnceWith({
            ...target,
            modelId: 'model',
            configurationKey: 'applied-key',
            hostBytes: '1610612736',
            deviceBytes: '134217728'
        })
        expect(container.textContent).toContain('Your figure (not verified by PAIR)')
        expect(container.textContent).toContain('Saved: 2026-11-07T01:00:00Z')
        expect(inspect).not.toHaveBeenCalled()
    })

    it('uses the applied key and pools while a different candidate is previewed', async () => {
        await act(async () =>
            root.render(
                <ModelSettingsPreview
                    target={target}
                    inspection={applied}
                    preview={{
                        settings: { serverPort: 1, proxyPort: 2, launchText: 'candidate-launch' },
                        revision: 2,
                        errors: {},
                        restart: false,
                        rebind: false,
                        previewDigest: 'candidate-digest',
                        estimate: {
                            status: 'estimated',
                            domains: [{ id: 'shared', kind: 'shared' }]
                        }
                    }}
                />
            )
        )
        await input('Host memory (GiB)', '2')
        await input('Device memory (GiB)', '1')
        await click('Save figure')
        expect(save.mock.lastCall?.[0]).toEqual({
            ...target,
            modelId: 'model',
            configurationKey: 'applied-key',
            hostBytes: '2147483648',
            deviceBytes: '1073741824'
        })
    })

    it('roundtrips byte values above Number safe precision and the uint64 maximum exactly', async () => {
        seed({
            ...applied,
            memoryProfiles: [
                {
                    configurationKey: 'applied-key',
                    kind: 'stated',
                    recordedAt: 'today',
                    current: true,
                    hostBytes: '9007199254740993',
                    deviceBytes: '18446744073709551615'
                }
            ]
        })
        await render()
        expect(field('Host memory (GiB)').value).toBe('8388608.000000000931322574615478515625')
        expect(field('Device memory (GiB)').value).toBe(
            '17179869183.999999999068677425384521484375'
        )
        await click('Save figure')
        expect(save.mock.lastCall?.[0]).toMatchObject({
            hostBytes: '9007199254740993',
            deviceBytes: '18446744073709551615'
        })
    })

    it.each([
        '0.1',
        '-1',
        'NaN',
        'Infinity',
        '1e3',
        '17179869184',
        '9007199254740993',
        '0.0000000000000000000000000000001'
    ])('refuses inexact or out-of-range GiB %s without a mutation', async value => {
        await render()
        await input('Host memory (GiB)', value)
        await input('Device memory (GiB)', '1')
        await click('Save figure')
        expect(save).not.toHaveBeenCalled()
        expect(container.textContent).toContain(
            'exact whole number of bytes within the supported range'
        )
    })

    it('requires both independent pools rather than sending a partial figure', async () => {
        await render()
        await input('Host memory (GiB)', '1')
        await click('Save figure')
        expect(save).not.toHaveBeenCalled()
        expect(container.querySelectorAll('input')).toHaveLength(2)
    })

    it('allows a host-only first figure when the estimate is unavailable', async () => {
        seed({ ...applied, estimate: { status: 'unavailable' }, configurationPools: ['host'] })
        await render()
        expect(container.querySelectorAll('input')).toHaveLength(1)
        await input('Host memory (GiB)', '0.000000000931322574615478515625')
        await click('Save figure')
        expect(save).toHaveBeenCalledExactlyOnceWith({
            ...target,
            modelId: 'model',
            configurationKey: 'applied-key',
            hostBytes: '1'
        })
    })

    it('shows one UMA input and clears the saved figure with one canonical operation', async () => {
        const shared: InspectModelResult = {
            ...applied,
            configurationPools: ['shared'],
            memoryProfiles: [
                {
                    configurationKey: 'applied-key',
                    kind: 'stated',
                    recordedAt: 'today',
                    current: true,
                    sharedBytes: '1073741824'
                }
            ]
        }
        seed(shared)
        clear.mockResolvedValue({ ...shared, memoryProfiles: [] })
        await render()
        expect(container.querySelectorAll('input')).toHaveLength(1)
        expect(field('Shared UMA memory (GiB)').value).toBe('1')
        await click('Clear figure')
        expect(clear).toHaveBeenCalledExactlyOnceWith({
            ...target,
            modelId: 'model',
            configurationKey: 'applied-key'
        })
        expect(save).not.toHaveBeenCalled()
        expect(container.textContent).not.toContain('Saved:')
        expect(field('Shared UMA memory (GiB)').value).toBe('')
    })

    it('reports configuration conflict and neither retries nor reinspects automatically', async () => {
        save.mockRejectedValue(new ServiceCallError('changed', POLICY_CONFLICT_CODE))
        await render()
        await input('Host memory (GiB)', '1')
        await input('Device memory (GiB)', '1')
        await click('Save figure')
        expect(container.textContent).toContain('Settings changed; reinspect and try again.')
        expect(save).toHaveBeenCalledOnce()
        expect(inspect).not.toHaveBeenCalled()
    })

    it('hides the inputs and explains the missing applied configuration key', async () => {
        seed({ ...applied, configurationKey: undefined, configurationKeyReason: 'not-applied' })
        await render()
        expect(container.querySelector('input')).toBeNull()
        expect(container.textContent).toContain('not-applied')
        expect(container.textContent).not.toContain('Save figure')
    })

    it('does not replace a newer key inspection when the old save response arrives', async () => {
        let finish: ((value: InspectModelResult) => void) | undefined
        save.mockImplementation(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        await render()
        await input('Host memory (GiB)', '1')
        await input('Device memory (GiB)', '1')
        await click('Save figure')
        const newer: InspectModelResult = {
            ...applied,
            configurationKey: 'new-key',
            configurationPools: ['shared']
        }
        inspect.mockResolvedValue(newer)
        await act(async () => useEngineSettingsStore.getState().inspectModel(target, 'model'))
        if (!finish) throw new Error('Save was not dispatched')
        const resolve = finish
        await act(async () => resolve({ ...applied, memoryProfiles: [] }))
        expect(
            useEngineSettingsStore.getState().modelDrafts[key].inspection?.configurationKey
        ).toBe('new-key')
        expect(container.querySelectorAll('input')).toHaveLength(1)
        expect(field('Shared UMA memory (GiB)').value).toBe('')
    })

    it('keeps another target and profile inspection intact while an old save is pending', async () => {
        let finish: ((value: InspectModelResult) => void) | undefined
        save.mockImplementation(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        await render()
        await input('Host memory (GiB)', '1')
        await input('Device memory (GiB)', '1')
        await click('Save figure')
        const peer = { ...target, nodeId: 'peer' }
        const peerKey = modelSettingsKey(peer, 'profile-b')
        const newer: InspectModelResult = {
            ...applied,
            ...peer,
            configurationKey: 'peer-key',
            configurationPools: ['shared']
        }
        await act(async () => {
            useEngineSettingsStore.setState(state => ({
                modelDrafts: {
                    ...state.modelDrafts,
                    [peerKey]: { generation: 8, inspection: newer }
                }
            }))
            root.render(<Editor selected={peer} profileKey="profile-b" />)
        })
        if (!finish) throw new Error('Save was not dispatched')
        const resolve = finish
        await act(async () => resolve(applied))
        expect(useEngineSettingsStore.getState().modelDrafts[peerKey].inspection).toBe(newer)
        expect(container.querySelectorAll('input')).toHaveLength(1)
        await input('Shared UMA memory (GiB)', '2')
        save.mockResolvedValue(newer)
        await click('Save figure')
        expect(save.mock.lastCall?.[0]).toEqual({
            ...peer,
            modelId: 'model',
            configurationKey: 'peer-key',
            sharedBytes: '2147483648'
        })
    })
})
