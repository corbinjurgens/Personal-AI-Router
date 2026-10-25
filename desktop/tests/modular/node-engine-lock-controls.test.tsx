// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ServiceError } from '@/shared/types/errors'
import type { EngineCommandPayload, EngineStatePatch } from '@/shared/types/engine-api'
import type { EngineStatusData } from '@/shared/types/engines'
import type { PreloadServiceTransport } from '@/shared/types/service-bridge'
import { createEngineApi } from '@/ui/api/engine-api'
import { serviceTransport } from '@/ui/api/service-transport'
import NodeEnginesInline from '@/ui/components/NodeList/NodeEnginesInline'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'

const SELF = 'node-self'
const PEER = 'node-peer'
const invoke = vi.fn<PreloadServiceTransport['invoke']>()
let stateCb: ((patch: EngineStatePatch) => void) | null = null
let errorsCb: ((errors: ServiceError[]) => void) | null = null
let root: Root
let container: HTMLDivElement

function status(nodeId: string, changes: Partial<EngineStatusData> = {}): EngineStatusData {
    return {
        engineType: 'ollama',
        nodeId,
        processStatus: 'stopped',
        enginePort: 11434,
        proxyPort: 54301,
        ...changes
    }
}

function report(row: EngineStatusData): void {
    useEngineStatusStore.setState({
        statusByNode: new Map([[row.nodeId, new Map([[row.engineType, row]])]])
    })
}

/** A pushed status replaces the stored row the way the engine status store does. */
function push(row: EngineStatusData): void {
    act(() => {
        report(row)
        stateCb?.({ nodeId: row.nodeId, engineType: row.engineType, status: row })
    })
}

async function mount(nodeId: string): Promise<void> {
    await act(async () => root.render(<NodeEnginesInline nodeId={nodeId} />))
}

function button(label: string): HTMLButtonElement {
    const found = container.querySelector(`button[aria-label="${label}"]`)
    if (!(found instanceof HTMLButtonElement)) throw new Error(`Missing button ${label}`)
    return found
}

function lockButtons(): HTMLButtonElement[] {
    return [...container.querySelectorAll('button')].filter(b =>
        /^(Lock|Unlock) /.test(b.getAttribute('aria-label') ?? '')
    )
}

function toggleSwitch(): HTMLInputElement {
    const found = container.querySelector('[role="switch"]')
    if (!(found instanceof HTMLInputElement)) throw new Error('Missing engine switch')
    return found
}

type InvokePayload = (typeof invoke.mock.calls)[number][1]

function isCommand(payload: InvokePayload): payload is EngineCommandPayload {
    return typeof payload === 'object' && payload !== null && 'command' in payload
}

function commands(): EngineCommandPayload[] {
    return invoke.mock.calls.flatMap(([channel, payload]) =>
        channel === 'engine:command' && isCommand(payload) ? [payload] : []
    )
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => false })
    stateCb = null
    errorsCb = null
    invoke.mockReset().mockResolvedValue({ success: true, data: null })
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
        value: {
            engines: {
                ...api,
                onStateChanged: (cb: (patch: EngineStatePatch) => void) => {
                    stateCb = cb
                    return () => {}
                },
                onProgress: () => () => {},
                onSettingsChanged: () => () => {}
            },
            errors: {
                onUpdate: (cb: (errors: ServiceError[]) => void) => {
                    errorsCb = cb
                    return () => {}
                }
            }
        }
    })
    Object.defineProperty(window, 'windowApi', { configurable: true, value: { platform: 'macOS' } })
    useConnectionStore.setState({ connected: true, selfId: SELF })
    useNodesStore.setState({ nodes: new Map() })
    useEngineStatusStore.setState({ statusByNode: new Map(), descriptors: new Map() })
    usePendingActionsStore.getState().cleanup()
    usePendingActionsStore.getState().initialize()
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    usePendingActionsStore.getState().cleanup()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
})

describe('engine lock on this device', () => {
    it('locks an unlocked engine through the typed setLock command', async () => {
        report(status(SELF, { locked: false }))
        await mount(SELF)
        await act(async () => button('Lock Ollama').click())
        expect(commands()).toEqual([
            { command: 'setLock', engineType: 'ollama', nodeId: SELF, locked: true }
        ])
    })

    it('unlocks a locked engine and shows the state in text', async () => {
        report(status(SELF, { locked: true }))
        await mount(SELF)
        expect(container.textContent).toContain('Locked')
        await act(async () => button('Unlock Ollama').click())
        expect(commands()).toEqual([
            { command: 'setLock', engineType: 'ollama', nodeId: SELF, locked: false }
        ])
    })

    it('does not claim success before the backend reports the lock', async () => {
        report(status(SELF, { locked: false }))
        await mount(SELF)
        await act(async () => button('Lock Ollama').click())
        expect(usePendingActionsStore.getState().getLifecyclePending(SELF, 'ollama')).toBe(
            'setLock'
        )
        expect(container.textContent).not.toContain('Locked')
        // The failure arrives as an error: the pending cover ends and the
        // authoritative status still says unlocked.
        act(() =>
            errorsCb?.([
                {
                    id: 'engine-cmd:setLock:node-self:ollama',
                    message: 'Failed to lock ollama: stop failed',
                    timestamp: Date.now(),
                    severity: 'error',
                    nodeId: SELF,
                    engineType: 'ollama'
                }
            ])
        )
        expect(usePendingActionsStore.getState().getLifecyclePending(SELF, 'ollama')).toBe(
            undefined
        )
        expect(button('Lock Ollama').disabled).toBe(false)
        expect(container.textContent).not.toContain('Locked')
    })

    it('covers the row while the command is pending and blocks conflicting commands', async () => {
        report(status(SELF, { locked: false }))
        await mount(SELF)
        await act(async () => button('Lock Ollama').click())
        expect(container.querySelector('[role="switch"]')).toBeNull()
        expect(container.querySelector('[role="status"]')).not.toBeNull()
        expect(button('Lock Ollama').disabled).toBe(true)
        await act(async () => button('Lock Ollama').click())
        expect(commands()).toHaveLength(1)
    })

    it('clears the pending cover on a lock-only push that leaves the process status alone', async () => {
        report(status(SELF, { locked: false }))
        await mount(SELF)
        await act(async () => button('Lock Ollama').click())
        push(status(SELF, { locked: true }))
        expect(usePendingActionsStore.getState().getLifecyclePending(SELF, 'ollama')).toBe(
            undefined
        )
        await mount(SELF)
        expect(button('Unlock Ollama').disabled).toBe(false)
    })

    it('reports a failed lock-stop as locked and still running', async () => {
        report(status(SELF, { locked: false, processStatus: 'running' }))
        await mount(SELF)
        await act(async () => button('Lock Ollama').click())
        push(status(SELF, { locked: true, processStatus: 'running' }))
        await mount(SELF)
        expect(container.textContent).toContain('Locked')
        expect(toggleSwitch().checked).toBe(true)
        expect(button('Unlock Ollama')).toBeDefined()
    })

    it('drops the pending cover and the lock when a reconnect invalidates the readback', async () => {
        report(status(SELF, { locked: false }))
        await mount(SELF)
        await act(async () => button('Lock Ollama').click())
        const { locked: _, ...unknown } = status(SELF)
        void _
        push(unknown)
        expect(usePendingActionsStore.getState().getLifecyclePending(SELF, 'ollama')).toBe(
            undefined
        )
        await mount(SELF)
        expect(container.textContent).toContain('Lock status unavailable')
        expect(button('Lock Ollama').disabled).toBe(true)
    })

    it('disables the lock button and says so when the readback is absent', async () => {
        report(status(SELF))
        await mount(SELF)
        expect(container.textContent).toContain('Lock status unavailable')
        expect(button('Lock Ollama').disabled).toBe(true)
        expect(toggleSwitch().disabled).toBe(false)
    })
})

describe('locked engine activation', () => {
    it('cannot be started from the switch or the engine name', async () => {
        report(status(SELF, { locked: true }))
        await mount(SELF)
        const toggle = toggleSwitch()
        expect(toggle.disabled).toBe(true)
        expect(toggle.getAttribute('aria-label')).toBe('Ollama is locked')
        await act(async () => toggle.click())
        const name = [...container.querySelectorAll('p, span, div')].find(
            e => e.children.length === 0 && e.textContent === 'Ollama'
        )
        if (!name) throw new Error('Missing engine name')
        await act(async () => (name instanceof HTMLElement ? name.click() : undefined))
        expect(commands()).toEqual([])
    })

    it('still stops a running locked engine', async () => {
        report(status(SELF, { locked: true, processStatus: 'running' }))
        await mount(SELF)
        expect(toggleSwitch().disabled).toBe(false)
        expect(toggleSwitch().getAttribute('aria-label')).toBe('Stop Ollama')
        await act(async () => toggleSwitch().click())
        expect(commands()).toEqual([{ command: 'toggle', engineType: 'ollama', nodeId: SELF }])
    })
})

describe('engine lock on a paired device', () => {
    it('shows the lock and where to change it, with no command', async () => {
        report(status(PEER, { locked: true }))
        await mount(PEER)
        expect(container.textContent).toContain('Locked')
        expect(container.textContent).toContain('Unlock this engine on the device running it.')
        expect(lockButtons()).toEqual([])
        expect(toggleSwitch().disabled).toBe(true)
        expect(commands()).toEqual([])
    })

    it('shows no lock marker for a known unlocked peer engine', async () => {
        report(status(PEER, { locked: false }))
        await mount(PEER)
        expect(container.textContent).not.toContain('Locked')
        expect(container.textContent).not.toContain('Lock status unavailable')
        expect(lockButtons()).toEqual([])
        expect(toggleSwitch().disabled).toBe(false)
    })

    it('reports an absent readback as unavailable, never as unlocked', async () => {
        report(status(PEER))
        await mount(PEER)
        expect(container.textContent).toContain('Lock status unavailable')
        expect(lockButtons()).toEqual([])
    })
})
