// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
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
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { parseModelSettings } from '@/shared/utils/node-policy-model-settings'

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
