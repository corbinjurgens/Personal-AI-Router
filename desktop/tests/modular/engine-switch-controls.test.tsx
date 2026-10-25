// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { PreloadServiceTransport } from '@/shared/types/service-bridge'
import type { SwitchReceipt, SwitchRequest } from '@/shared/types/engine-switch'
import type { NodePolicyDocument } from '@/shared/types/node-policy'
import { createEngineApi } from '@/ui/api/engine-api'
import { serviceTransport } from '@/ui/api/service-transport'
import type { IClusterApi } from '@/ui/api/pair-api'
import { EngineSwitchControls } from '@/ui/components/NodeList/EngineSwitchControls'
import NodeEngineSettings from '@/ui/components/NodeList/NodeEngineSettings'
import { BackendRow } from '@/ui/components/BackendRow/BackendRow'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useEngineSwitchStore } from '@/ui/stores/engine-switch.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { emptyEngineStatus } from '@/shared/utils/engines'

const revision = `e3v1:${'a'.repeat(64)}`

it('labels automatic activity while retaining request and phase evidence', async () => {
    useEngineSwitchStore.setState({
        entries: {
            'node-a': { receipt: receipt({ automatic: true }), request, own: false, revision }
        }
    })
    await mount()
    expect(container.textContent).toContain('Automatic engine switch')
    expect(container.textContent).toContain('draining')
    expect(container.textContent).toContain('observed')
})
const engines = [
    { value: 'ollama', children: 'Ollama' },
    { value: 'llamacpp', children: 'llama.cpp' }
]
const request: SwitchRequest = {
    nodeId: 'node-a',
    requestId: 'observed',
    sourceEngine: 'ollama',
    targetEngine: 'llamacpp',
    expectedRevision: revision
}
function receipt(changes: Partial<SwitchReceipt> = {}): SwitchReceipt {
    const state: SwitchReceipt['source'] = {
        intent: 'unknown',
        running: null,
        healthy: null,
        observedAt: null,
        terminationVerified: false
    }
    return {
        version: 1,
        nodeId: 'node-a',
        requestId: 'observed',
        request,
        caller: 'local',
        payloadHash: '',
        expectedRevision: revision,
        acceptedRevision: revision,
        sequence: 3,
        phase: 'draining',
        terminal: false,
        acceptedAt: '2026-10-20T00:00:00Z',
        updatedAt: '2026-10-20T00:00:00Z',
        deadline: '2026-10-20T00:15:00Z',
        finishedAt: null,
        source: state,
        target: state,
        error: null,
        recoveryRequired: false,
        ...changes
    }
}
const invoke = vi.fn<PreloadServiceTransport['invoke']>()
const permission = vi.fn<IClusterApi['getRemoteManagementForNode']>()
let root: Root
let container: HTMLDivElement
let closeTransport: () => void
async function mount(nodeId = 'node-a', choices = engines) {
    await act(async () =>
        root.render(<EngineSwitchControls key={nodeId} nodeId={nodeId} engines={choices} />)
    )
}
async function select(index: number, value: string) {
    const element = container
        .querySelectorAll('[role="listbox"]')
        [index]?.querySelector(`button[value="${value}"]`)
    if (!(element instanceof HTMLButtonElement))
        throw new Error(`Missing engine option ${index}: ${value}`)
    await act(async () => element.click())
}
function submit() {
    const button = [...container.querySelectorAll('button')].find(
        button => button.textContent === 'Switch'
    )
    if (!button) throw new Error('Switch button missing')
    return button
}
beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => false })
    let settle: (value: Awaited<ReturnType<PreloadServiceTransport['invoke']>>) => void = () => {}
    const pending = new Promise<Awaited<ReturnType<PreloadServiceTransport['invoke']>>>(resolve => {
        settle = resolve
    })
    closeTransport = () => settle({ success: false, error: 'test transport closed' })
    invoke.mockReset().mockReturnValue(pending)
    invoke.mockResolvedValueOnce({ success: true, data: { configurationRevision: revision } })
    permission
        .mockReset()
        .mockResolvedValue({ version: 1, allowed: true, state: 'allowed', revision: '1' })
    const api = createEngineApi(
        serviceTransport({
            connected: true,
            invoke,
            invokeHistory: vi.fn<PreloadServiceTransport['invokeHistory']>(() => {
                throw new Error('Unexpected history invoke in ordinary transport fixture')
            }),
            subscribePush: () => () => {},
            onConnect: () => () => {},
            onDisconnect: () => () => {},
            onAuthFailure: () => () => {},
            destroy: () => {}
        })
    )
    Object.defineProperty(window, 'pairApi', {
        configurable: true,
        value: { engines: api, cluster: { getRemoteManagementForNode: permission } }
    })
    Object.defineProperty(window, 'windowApi', { configurable: true, value: { platform: 'macOS' } })
    useConnectionStore.setState({ connected: true, selfId: 'node-a' })
    useEngineSwitchStore.setState({ entries: {} })
    usePendingActionsStore.setState({ pending: new Map(), switches: new Map() })
    useNodeConfigurationStore.setState({ management: new Map() })
    useEngineStatusStore.setState({ statusByNode: new Map() })
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
})
afterEach(async () => {
    await act(async () => {
        root.unmount()
        closeTransport()
    })
    container.remove()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
})

describe('engine switch controls in node settings', () => {
    it('accessible From and To selections send exact IDs and fresh revision once', async () => {
        await mount()
        const comboboxes = container.querySelectorAll('[role="combobox"]')
        for (const [index, label] of ['From', 'To'].entries()) {
            const box = comboboxes[index]
            expect(box).toBeDefined()
            const fieldLabel = container.querySelector(`label[for="${box.id}"]`)
            expect(fieldLabel?.textContent).toBe(label)
            expect(box.getAttribute('aria-labelledby')).toBe(fieldLabel?.id)
        }
        await select(0, 'ollama')
        await select(1, 'llamacpp')
        expect(submit().disabled).toBe(false)
        invoke.mockResolvedValueOnce({ success: true, data: { configurationRevision: 'fresh' } })
        await act(async () => submit().click())
        const writes = invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')
        expect(writes).toEqual([
            [
                'engines:switch',
                {
                    nodeId: 'node-a',
                    requestId: expect.any(String),
                    sourceEngine: 'ollama',
                    targetEngine: 'llamacpp',
                    expectedRevision: 'fresh'
                }
            ]
        ])
        expect(container.textContent).toContain('Reading and submitting the switch request.')
        expect(container.textContent).not.toContain('completed')
        expect(submit().disabled).toBe(true)
    })
    it('empty and equal selections disable submit and send no writes', async () => {
        await mount()
        await act(async () => submit().click())
        await select(0, 'ollama')
        await select(1, 'ollama')
        expect(submit().disabled).toBe(true)
        expect(container.textContent).toContain('must be different')
        await act(async () => submit().click())
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            0
        )
    })
    it('lost response shows unknown outcome and retains the original request without resend', async () => {
        await mount()
        await select(0, 'ollama')
        await select(1, 'llamacpp')
        invoke.mockResolvedValueOnce({ success: true, data: { configurationRevision: revision } })
        invoke.mockResolvedValueOnce({ success: false, error: 'lost response' })
        await act(async () => submit().click())
        const original = useEngineSwitchStore.getState().entries['node-a'].request?.requestId
        expect(original).toEqual(expect.any(String))
        expect(container.textContent).toContain(`Submitted request: ${original}`)
        expect(container.textContent).toContain('Switch outcome unknown: lost response')
        expect(container.textContent).not.toContain('completed')
        expect(submit().disabled).toBe(true)
        await act(async () => submit().click())
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            1
        )
    })
    it('Refresh exposes a peer grant while connected and preserves the selected engines', async () => {
        useConnectionStore.setState({ selfId: 'self' })
        permission.mockResolvedValue({
            version: 1,
            allowed: false,
            state: 'unreviewed',
            revision: '1'
        })
        await mount()
        await select(0, 'ollama')
        await select(1, 'llamacpp')
        expect(submit().disabled).toBe(true)
        expect(container.textContent).toContain('has not granted remote management')

        permission.mockResolvedValue({ version: 1, allowed: true, state: 'allowed', revision: '2' })
        invoke.mockResolvedValueOnce({ success: true, data: { configurationRevision: revision } })
        const refresh = [...container.querySelectorAll('button')].find(
            button => button.textContent === 'Refresh readback'
        )
        if (!refresh) throw new Error('Refresh button missing')
        await act(async () => refresh.click())

        expect(submit().disabled).toBe(false)
        expect(container.textContent).not.toContain('has not granted remote management')
        expect(container.querySelector('[aria-label="From"]')?.textContent).toBe('Ollama')
        expect(container.querySelector('[aria-label="To"]')?.textContent).toBe('llama.cpp')
        expect(useConnectionStore.getState().connected).toBe(true)
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            0
        )
    })

    it('permission revoked after selection retains draft and disables mutation', async () => {
        useConnectionStore.setState({ selfId: 'self' })
        await mount()
        await select(0, 'ollama')
        await select(1, 'llamacpp')
        expect(submit().disabled).toBe(false)
        await act(async () =>
            useNodeConfigurationStore.setState({
                management: new Map([
                    [
                        'node-a',
                        {
                            state: 'denied',
                            revision: '2',
                            message: 'Remote management blocked on this device.'
                        }
                    ]
                ])
            })
        )
        expect(submit().disabled).toBe(true)
        expect(container.textContent).toContain('Remote management blocked')
        expect(container.querySelector('[aria-label="From"]')?.textContent).toBe('Ollama')
        await act(async () => submit().click())
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            0
        )
    })
    it('pre-acceptance denial shows the actual refusal and permits a deliberate retry after permission readback', async () => {
        useConnectionStore.setState({ selfId: 'self' })
        await mount()
        await select(0, 'ollama')
        await select(1, 'llamacpp')
        permission.mockResolvedValueOnce({
            version: 1,
            allowed: true,
            state: 'allowed',
            revision: '1'
        })
        // Submit's status refresh independently checks permission before sending.
        permission.mockResolvedValueOnce({
            version: 1,
            allowed: true,
            state: 'allowed',
            revision: '1'
        })
        permission.mockResolvedValueOnce({
            version: 1,
            allowed: false,
            state: 'denied',
            revision: '2'
        })
        invoke.mockResolvedValueOnce({ success: true, data: { configurationRevision: revision } })
        invoke.mockResolvedValueOnce({
            success: false,
            error: 'remote management not permitted',
            code: -32012
        })
        await act(async () => submit().click())
        const original = useEngineSwitchStore.getState().entries['node-a'].request?.requestId
        expect(container.textContent).toContain(
            'refused before acceptance: remote management not permitted'
        )
        expect(container.textContent).not.toContain('outcome unknown')
        expect(container.querySelector('[aria-label="From"]')?.textContent).toBe('Ollama')
        expect(container.querySelector('[aria-label="To"]')?.textContent).toBe('llama.cpp')
        expect(submit().disabled).toBe(true)
        expect(permission).toHaveBeenCalledTimes(4)
        expect(useEngineSwitchStore.getState().reserved('node-a', 'ollama')).toBe(false)
        expect(useEngineSwitchStore.getState().reserved('node-a', 'llamacpp')).toBe(false)
        permission.mockResolvedValue({ version: 1, allowed: true, state: 'allowed', revision: '3' })
        await act(async () =>
            useNodeConfigurationStore.getState().readManagement('node-a', 'node-a')
        )
        expect(submit().disabled).toBe(false)
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            1
        )
        invoke.mockResolvedValueOnce({ success: true, data: { configurationRevision: revision } })
        invoke.mockResolvedValueOnce({
            success: false,
            error: 'remote management not permitted',
            code: -32012
        })
        await act(async () => submit().click())
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            2
        )
        expect(useEngineSwitchStore.getState().entries['node-a'].request?.requestId).not.toBe(
            original
        )
    })
    it('source and target identity survives engine list reorder', async () => {
        await mount()
        await select(0, 'ollama')
        await select(1, 'llamacpp')
        await mount('node-a', [...engines].reverse())
        expect(container.querySelector('[aria-label="From"]')?.textContent).toBe('Ollama')
        expect(container.querySelector('[aria-label="To"]')?.textContent).toBe('llama.cpp')
        invoke.mockResolvedValueOnce({ success: true, data: { configurationRevision: revision } })
        await act(async () => submit().click())
        expect(invoke).toHaveBeenLastCalledWith(
            'engines:switch',
            expect.objectContaining({ sourceEngine: 'ollama', targetEngine: 'llamacpp' })
        )
    })
    it.each([
        {
            name: 'permission denial',
            code: -32012,
            label: 'Remote management permission denied',
            message: 'refused'
        },
        { name: 'locked engine', code: -32013, label: 'Engine locked', message: 'engine locked' },
        {
            name: 'fresh pause',
            code: -32011,
            label: 'Node paused',
            message: 'refused',
            availability: 'paused'
        },
        {
            name: 'fresh draining',
            code: -32011,
            label: 'Node paused',
            message: 'refused',
            availability: 'draining'
        },
        {
            name: 'bounded pause message',
            code: -32011,
            label: 'Node paused',
            message: 'This node is paused. Resume it on this node before managing it remotely.'
        },
        {
            name: 'older saved Off',
            code: -32011,
            label: 'Target refused',
            message: 'engine was turned off locally',
            availability: 'available'
        },
        {
            name: 'operation scoped stop',
            code: -32011,
            label: 'Target refused',
            message: 'local stop cancelled this switch',
            availability: 'available'
        },
        { name: 'unavailable policy', code: -32011, label: 'Target refused', message: 'refused' },
        {
            name: 'configuration conflict',
            code: -32010,
            label: 'Configuration conflict',
            message: 'revision changed'
        }
    ] satisfies {
        name: string
        code: number
        label: string
        message: string
        availability?: NodePolicyDocument['availability']
    }[])(
        'renders $name with source/target readback and independent recovery state',
        async ({ code, label, message, availability }) => {
            await mount()
            await act(async () =>
                useEngineSwitchStore.setState({
                    entries: {
                        'node-a': {
                            revision,
                            request,
                            own: false,
                            refusalPolicy: availability
                                ? {
                                      policy: JSON.stringify({ availability, pausedBy: 'peer' }),
                                      revision: '2',
                                      activation: null,
                                      recoveryRequired: false,
                                      availability
                                  }
                                : undefined,
                            receipt: receipt({
                                terminal: true,
                                phase: 'failed-validation',
                                recoveryRequired: true,
                                error: {
                                    code,
                                    kind: code === -32013 ? 'engine_locked' : 'validation',
                                    message
                                }
                            })
                        }
                    }
                })
            )
            expect(container.textContent).toContain(`${label}: ${message}`)
            expect(container.textContent).toContain('Recovery required')
            expect(container.textContent).toContain('Observed activity: observed')
            expect(container.textContent).toContain('Phase: failed-validation; sequence 3')
            expect(container.textContent).toContain('Source ollama:')
            expect(container.textContent).toContain('Target llamacpp:')
            expect(container.textContent).not.toContain('completed')
            expect(submit().disabled).toBe(true)
        }
    )
    it('excludes locked activation targets while leaving them selectable as sources', async () => {
        useEngineStatusStore.setState({
            statusByNode: new Map([
                [
                    'node-a',
                    new Map([
                        ['llama-cpp', { ...emptyEngineStatus('node-a', 'llama-cpp'), locked: true }]
                    ])
                ]
            ])
        })
        await mount()
        const lists = container.querySelectorAll('[role="listbox"]')
        expect(lists[0]?.querySelector('button[value="llamacpp"]')).not.toBeNull()
        expect(lists[1]?.querySelector('button[value="llamacpp"]')).toBeNull()
        await select(0, 'llamacpp')
        await select(1, 'ollama')
        expect(submit().disabled).toBe(false)
    })
    it('blocks an already selected target when authoritative lock readback arrives', async () => {
        await mount()
        await select(0, 'ollama')
        await select(1, 'llamacpp')
        await act(async () =>
            useEngineStatusStore.setState({
                statusByNode: new Map([
                    [
                        'node-a',
                        new Map([
                            [
                                'llama-cpp',
                                { ...emptyEngineStatus('node-a', 'llama-cpp'), locked: true }
                            ]
                        ])
                    ]
                ])
            })
        )
        expect(submit().disabled).toBe(true)
        expect(container.textContent).toContain(
            'Engine locked. Unlock it on the device running it before switching.'
        )
        await act(async () => submit().click())
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            0
        )
    })
    it('drain shows reported identity and evidence without percentage or optimistic completion', async () => {
        await mount()
        await act(async () =>
            useEngineSwitchStore.setState({
                entries: { 'node-a': { revision, request, receipt: receipt() } }
            })
        )
        expect(container.textContent).toContain('Phase: draining; sequence 3')
        expect(container.textContent).toContain(
            'Source ollama: Saved intent: Unknown; Runtime: Unknown'
        )
        expect(container.textContent).toContain('Target llamacpp:')
        expect(container.textContent).toContain('Health: Unknown')
        expect(container.textContent).toContain('Termination not verified')
        expect(container.textContent).not.toMatch(/completed|\d+%|ETA/)
    })
    it('renders runtime, health and termination separately from saved intent', async () => {
        await mount()
        await act(async () =>
            useEngineSwitchStore.setState({
                entries: {
                    'node-a': {
                        revision,
                        request,
                        receipt: receipt({
                            source: {
                                intent: 'on',
                                running: false,
                                healthy: false,
                                observedAt: '2026-10-20T00:00:00Z',
                                terminationVerified: true
                            },
                            target: {
                                intent: 'off',
                                running: true,
                                healthy: true,
                                observedAt: null,
                                terminationVerified: false
                            }
                        })
                    }
                }
            })
        )
        expect(container.textContent).toContain(
            'Source ollama: Saved intent: On; Runtime: Not running; Health: Unhealthy; Observed: 2026-10-20T00:00:00Z; Termination verified'
        )
        expect(container.textContent).toContain(
            'Target llamacpp: Saved intent: Off; Runtime: Running; Health: Healthy; Observed: Unavailable; Termination not verified'
        )
    })
    it('node change resets selections and ignores old delayed read callback', async () => {
        await mount()
        await select(0, 'ollama')
        let resolveOld: (
            result: Awaited<ReturnType<PreloadServiceTransport['invoke']>>
        ) => void = () => {}
        invoke.mockReturnValueOnce(
            new Promise(resolve => {
                resolveOld = resolve
            })
        )
        let old: Promise<boolean> = Promise.resolve(false)
        await act(async () => {
            old = useEngineSwitchStore.getState().refresh('node-a')
        })
        invoke.mockResolvedValueOnce({
            success: true,
            data: { configurationRevision: 'node-b-revision' }
        })
        await mount('node-b')
        await act(async () => {
            resolveOld({ success: true, data: { configurationRevision: 'stale' } })
            await old
        })
        expect(container.textContent).toContain('Node: node-b')
        expect(container.textContent).not.toContain('node-a')
        expect(container.querySelector('[aria-label="From"]')?.textContent).toBe('Select engine')
        expect(useEngineSwitchStore.getState().entries['node-a'].revision).toBeUndefined()
        expect(useEngineSwitchStore.getState().entries['node-b'].revision).toBe('node-b-revision')
    })
    it('disconnect disables controls and sends no mutation', async () => {
        await mount()
        await select(0, 'ollama')
        await select(1, 'llamacpp')
        await act(async () => useConnectionStore.setState({ connected: false }))
        expect(submit().disabled).toBe(true)
        expect(container.textContent).toContain('Device disconnected')
        await act(async () => submit().click())
        expect(invoke.mock.calls.filter(([channel]) => channel === 'engines:switch')).toHaveLength(
            0
        )
    })
    it('switch reservations leave the existing local Off veto reachable', async () => {
        useEngineSwitchStore.setState({
            entries: { 'node-a': { revision, request, receipt: receipt() } }
        })
        await act(async () =>
            root.render(
                <BackendRow
                    nodeId="node-a"
                    statusKnown
                    backend={{
                        type: 'ollama',
                        displayName: 'Ollama',
                        source: 'detected',
                        processStatus: 'running',
                        port: 12000,
                        proxyPort: 12001,
                        models: []
                    }}
                />
            )
        )
        const off = container.querySelector('[aria-label="Stop Ollama"]')
        if (!(off instanceof HTMLInputElement)) throw new Error('Local Stop control missing')
        expect(off.disabled).toBe(false)
        await act(async () => off.click())
        expect(invoke).toHaveBeenLastCalledWith('engine:command', {
            command: 'toggle',
            engineType: 'ollama',
            nodeId: 'node-a'
        })
    })
    it('renders the form in the existing node settings with a narrow stacking layout', () => {
        useEngineStatusStore.setState({
            statusByNode: new Map([
                [
                    'node-a',
                    new Map([
                        [
                            'ollama',
                            { ...emptyEngineStatus('node-a', 'ollama'), processStatus: 'running' }
                        ],
                        [
                            'llama-cpp',
                            {
                                ...emptyEngineStatus('node-a', 'llama-cpp'),
                                processStatus: 'stopped'
                            }
                        ]
                    ])
                ]
            ])
        })
        const markup = renderToStaticMarkup(<NodeEngineSettings nodeId="node-a" />)
        expect(markup).toContain('Switch engine')
        expect(markup).toContain('grid-cols-1 gap-3 sm:grid-cols-2')
        expect(markup).toContain('aria-label="From"')
        expect(markup).toContain('aria-label="To"')
        expect(markup).toContain('Device disconnected. Reconnect before switching.')
        expect(markup).toContain('value="llamacpp"')
    })
})
