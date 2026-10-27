// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { Duplex } from 'stream'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => process.cwd() },
    BrowserWindow: { getAllWindows: () => [], fromWebContents: () => ({}) },
    ipcRenderer: {
        invoke: () => Promise.reject(new Error('Unused default transport')),
        on: () => {},
        removeListener: () => {}
    }
}))
vi.mock('@/electron/window', () => ({ createOverviewWindow: () => undefined }))
vi.mock('@electron-toolkit/utils', () => ({ is: { dev: false } }))

import NodeEnginesInline from '@/ui/components/NodeList/NodeEnginesInline'
import { createEngineApi } from '@/ui/api/engine-api'
import { serviceTransport } from '@/ui/api/service-transport'
import { createPairTransport } from '@/preload/api/pair-transport.api'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'
import { JsonRpcSocketClient } from '@/electron/service-bridge/json-rpc-client'
import { subscribePush } from '@/electron/service-bridge/push-bus'
import { BoundedLogWriter } from '@/shared/utils/bounded-log-writer'
import { createStructuredLogger, initFileLogger } from '@/shared/utils/log'
import type { ServiceBridgeInvokeRequest } from '@/shared/types/service-bridge'
import type { EngineCredentialState } from '@/shared/types/engine-api'
import type { EngineType } from '@/shared/types/engines'

const secret = 'test-key-123'
const self = 'self'
const absent = { supported: true, configured: false }
const saved = { supported: true, configured: true }
type Ipc = Parameters<typeof createPairTransport>[0]
const invokes: ServiceBridgeInvokeRequest[] = []
const envelopes: string[] = []
let subscriptions = 0
const ipc = {
    invoke: vi.fn<Ipc['invoke']>(),
    on: () => {
        subscriptions++
    },
    removeListener: () => {}
} satisfies Ipc
let container: HTMLDivElement
let root: ReturnType<typeof createRoot>
let directory: string
let unsubscribe: () => void
let pushes = 0
let pushedSecret = false
let historyCalls: () => number

// Only booleans/counts reach assertions. Explicitly visit Maps and Errors:
// JSON.stringify(Map) hides entries; raw mock/error/DOM assertions can print keys.
function containsSecret(value: object | string | number | boolean | null | undefined): boolean {
    const seen = new Set<object>()
    function visit(item: typeof value): boolean {
        if (typeof item === 'string') return item.includes(secret)
        if (!item || typeof item !== 'object' || seen.has(item)) return false
        seen.add(item)
        if (item instanceof Map) return [...item].some(([key, entry]) => visit(key) || visit(entry))
        if (item instanceof Set) return [...item].some(visit)
        return Object.entries(Object.getOwnPropertyDescriptors(item)).some(
            ([key, descriptor]) => visit(key) || visit(descriptor.value)
        )
    }
    return visit(value)
}

function status(engineType: EngineType = 'lm-studio', nodeId = self) {
    useEngineStatusStore.setState({
        statusByNode: new Map([
            [
                nodeId,
                new Map([
                    [
                        engineType,
                        {
                            engineType,
                            nodeId,
                            processStatus: 'stopped',
                            enginePort: 1235,
                            proxyPort: 0
                        }
                    ]
                ])
            ]
        ])
    })
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => false })
    ipc.invoke
        .mockReset()
        .mockImplementation(async (_channel, request: ServiceBridgeInvokeRequest) => {
            envelopes.push(_channel)
            invokes.push(request)
            try {
                return {
                    success: true,
                    data: await handleServiceBridgeInvoke(request.channel, request.payload)
                }
            } catch (error) {
                return {
                    success: false,
                    error: error instanceof Error ? error.message : 'Service unavailable'
                }
            }
        })
    invokes.length = 0
    envelopes.length = 0
    subscriptions = 0
    const preload = createPairTransport(ipc)
    const history = vi.spyOn(preload, 'invokeHistory')
    historyCalls = () => history.mock.calls.length
    Object.defineProperty(window, 'pairApi', {
        configurable: true,
        value: { engines: createEngineApi(serviceTransport(preload)) }
    })
    getModularBridgeState().setSelfId(self)
    vi.spyOn(getModularSupervisor(), 'callProcess').mockResolvedValue(absent)
    useConnectionStore.setState({ selfId: self })
    useEngineStatusStore.setState({ descriptors: new Map() })
    status()
    pushes = 0
    pushedSecret = false
    unsubscribe = subscribePush(event => {
        pushes++
        pushedSecret ||= containsSecret(event)
    })
    directory = mkdtempSync(join(tmpdir(), 'pair-t8-log-'))
    initFileLogger({
        getUserData: () => directory,
        getTemp: () => directory,
        getResourcesPath: () => directory
    })
    vi.spyOn(BoundedLogWriter.prototype, 'write').mockImplementation(() => {})
    vi.spyOn(process.stdout, 'write').mockReturnValue(true)
    vi.spyOn(process.stderr, 'write').mockReturnValue(true)
    for (const method of [
        'log',
        'info',
        'warn',
        'error',
        'debug',
        'trace'
    ] satisfies (keyof Console)[])
        vi.spyOn(console, method).mockImplementation(() => {})
    createStructuredLogger('credential-test').info({ sublevel: 'test', message: 'sink-check' })
    expect(vi.mocked(BoundedLogWriter.prototype.write).mock.calls.length > 0).toBe(true)
    expect(vi.mocked(process.stdout.write).mock.calls.length > 0).toBe(true)
    vi.mocked(BoundedLogWriter.prototype.write).mockClear()
    vi.mocked(process.stdout.write).mockClear()
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
})
afterEach(async () => {
    try {
        await act(async () => root.unmount())
        expect(containsSecret(vi.mocked(process.stdout.write).mock.calls)).toBe(false)
        expect(containsSecret(vi.mocked(process.stderr.write).mock.calls)).toBe(false)
        expect(containsSecret(vi.mocked(BoundedLogWriter.prototype.write).mock.calls)).toBe(false)
    } finally {
        container.remove()
        unsubscribe()
        rmSync(directory, { recursive: true, force: true })
        vi.restoreAllMocks()
        vi.unstubAllGlobals()
    }
})

async function mount(nodeId = self) {
    await act(async () => root.render(createElement(NodeEnginesInline, { nodeId })))
}
function controls() {
    const field = container.querySelector('input[type=password]')
    const buttons = [...container.querySelectorAll('button')]
    const save = buttons.find(button => button.textContent === 'Save')
    const clear = buttons.find(button => button.textContent === 'Clear')
    const cancel = buttons.find(button => button.textContent === 'Cancel')
    if (!(field instanceof HTMLInputElement) || !save || !clear || !cancel)
        throw new Error('Missing credential controls')
    return { field, save, clear, cancel }
}
function textIncludes(text: string): boolean {
    return container.textContent?.includes(text) === true
}
function deferred() {
    let resolve: (state: EngineCredentialState) => void = () => {
        throw new Error('Missing resolver')
    }
    let reject: (error: Error) => void = () => {
        throw new Error('Missing rejecter')
    }
    const promise = new Promise<typeof absent>((accept, refuse) => {
        resolve = accept
        reject = refuse
    })
    return { promise, resolve, reject }
}
async function message(promise: Promise<EngineCredentialState>): Promise<string> {
    try {
        await promise
        return 'unexpected-success'
    } catch (error) {
        return error instanceof Error ? error.message : 'unexpected-error'
    }
}

describe('local credential bridge through real API and transports', () => {
    it.each(['Save', 'Clear'])(
        'keeps a slow %s blocked beyond the ordinary RPC timeout',
        async action => {
            vi.useFakeTimers()
            let configured = action === 'Clear'
            let mutationId: number | undefined
            const stream = new Duplex({
                read() {
                    // The fixture supplies replies explicitly after the write settles.
                },
                write(chunk: Buffer, _encoding, done) {
                    const frame: { id: number; method: string } = JSON.parse(chunk.toString())
                    if (frame.method === 'engine:set-credential') mutationId = frame.id
                    else
                        stream.push(
                            JSON.stringify({
                                jsonrpc: '2.0',
                                id: frame.id,
                                result: { supported: true, configured }
                            }) + '\n'
                        )
                    done()
                }
            })
            const rpc = new JsonRpcSocketClient(
                'broker',
                async () => stream,
                [],
                async () => stream
            )
            const call = vi
                .mocked(getModularSupervisor().callProcess)
                .mockImplementation((_process, method, params, timeoutMs) =>
                    rpc.call(method, params, timeoutMs)
                )
            try {
                rpc.start()
                await Promise.resolve()
                await mount()
                const { field, save, clear } = controls()
                field.value = secret
                await act(async () => (action === 'Save' ? save : clear).click())
                await act(async () => vi.advanceTimersByTimeAsync(15_000))
                expect(save.disabled && clear.disabled && field.value === '').toBe(true)
                let refusal: string | null = null
                const refused = message(
                    window.pairApi.engines.setCredential({
                        nodeId: self,
                        engineType: 'lm-studio',
                        credential: ''
                    })
                ).then(result => {
                    refusal = result
                })
                await act(async () => vi.advanceTimersByTimeAsync(0))
                expect(refusal === 'Credential update pending').toBe(true)
                await refused
                expect(call.mock.calls.length === 2 && mutationId !== undefined).toBe(true)
                configured = action === 'Save'
                await act(async () => {
                    stream.push(
                        JSON.stringify({
                            jsonrpc: '2.0',
                            id: mutationId,
                            result: { supported: true, configured }
                        }) + '\n'
                    )
                    await vi.advanceTimersByTimeAsync(0)
                })
                expect(!save.disabled && !clear.disabled).toBe(true)
                expect(textIncludes(configured ? 'Key saved' : 'No key saved')).toBe(true)
            } finally {
                vi.useRealTimers()
                await act(async () => rpc.detach())
            }
        }
    )
    it('detects nested Map entries and both real structured logger output sinks', () => {
        expect(containsSecret(new Map([['outer', new Map([['inner', secret]])]]))).toBe(true)
        expect(containsSecret(new Map([['outer', new Map([['inner', 'metadata']])]]))).toBe(false)
        expect(containsSecret(new Error('metadata', { cause: new Error(secret) }))).toBe(true)
        createStructuredLogger('credential-test').error({
            sublevel: 'test',
            message: 'capture-check',
            data: { nested: [secret] }
        })
        expect(containsSecret(vi.mocked(process.stdout.write).mock.calls)).toBe(true)
        expect(containsSecret(vi.mocked(BoundedLogWriter.prototype.write).mock.calls)).toBe(true)
        vi.mocked(process.stdout.write).mockClear()
        vi.mocked(BoundedLogWriter.prototype.write).mockClear()
    })
    it.each(['Save', 'Clear'])(
        'refuses another mutation and delays a read until physical %s settles',
        async action => {
            const mutation = deferred()
            const call = vi
                .mocked(getModularSupervisor().callProcess)
                .mockReturnValueOnce(mutation.promise)
            const writing = window.pairApi.engines.setCredential({
                nodeId: self,
                engineType: 'lm-studio',
                credential: action === 'Save' ? secret : ''
            })
            const reading = window.pairApi.engines.credentialState({
                nodeId: self,
                engineType: 'lm-studio'
            })
            const refused = await message(
                window.pairApi.engines.setCredential({
                    nodeId: self,
                    engineType: 'lm-studio',
                    credential: action === 'Save' ? '' : secret
                })
            )
            expect(refused === 'Credential update pending').toBe(true)
            expect(call.mock.calls.length === 1).toBe(true)
            mutation.resolve(saved)
            await writing
            await reading
            expect(
                call.mock.calls.length === 2 && call.mock.calls[1][1] === 'engine:credential-state'
            ).toBe(true)
        }
    )
    it('sends exact local methods and shapes including explicit clear', async () => {
        const call = vi.mocked(getModularSupervisor().callProcess)
        await window.pairApi.engines.credentialState({ nodeId: self, engineType: 'lm-studio' })
        await window.pairApi.engines.setCredential({
            nodeId: self,
            engineType: 'lm-studio',
            credential: secret
        })
        await window.pairApi.engines.setCredential({
            nodeId: self,
            engineType: 'lm-studio',
            credential: ''
        })
        expect(call.mock.calls.length === 3).toBe(true)
        expect(
            call.mock.calls.every(
                ([process, method, params], index) =>
                    process === 'broker' &&
                    method ===
                        (index === 0 ? 'engine:credential-state' : 'engine:set-credential') &&
                    params !== null &&
                    typeof params === 'object' &&
                    !Array.isArray(params) &&
                    Object.keys(params).sort().join(',') ===
                        (index === 0 ? 'engine' : 'credential,engine') &&
                    params.engine === 'lmstudio' &&
                    (index === 0 || params.credential === (index === 1 ? secret : ''))
            )
        ).toBe(true)
        expect(envelopes.every(channel => channel === 'service-bridge:invoke')).toBe(true)
        expect(
            invokes.every(
                request =>
                    request.payload &&
                    typeof request.payload === 'object' &&
                    'nodeId' in request.payload &&
                    request.payload.nodeId === self &&
                    'engineType' in request.payload &&
                    request.payload.engineType === 'lm-studio' &&
                    Object.keys(request).sort().join(',') === 'channel,payload' &&
                    Object.keys(request.payload).sort().join(',') ===
                        (request.channel === 'engine:credential-state'
                            ? 'engineType,nodeId'
                            : 'credential,engineType,nodeId')
            )
        ).toBe(true)
    })
    it.each(['peer', 'unknown self', 'empty self'])(
        'refuses %s before RPC for both methods',
        async target => {
            const call = vi.mocked(getModularSupervisor().callProcess)
            if (target !== 'peer')
                vi.spyOn(getModularBridgeState(), 'getSelfId').mockReturnValue(
                    target === 'unknown self' ? null : ''
                )
            const nodeId = target === 'peer' ? 'peer' : target === 'empty self' ? '' : self
            expect(
                (await message(
                    window.pairApi.engines.credentialState({ nodeId, engineType: 'lm-studio' })
                )) === 'Credentials can only be changed on this device.'
            ).toBe(true)
            expect(
                (await message(
                    window.pairApi.engines.setCredential({
                        nodeId,
                        engineType: 'lm-studio',
                        credential: secret
                    })
                )) === 'Credentials can only be changed on this device.'
            ).toBe(true)
            expect(call.mock.calls.length === 0).toBe(true)
        }
    )
    it('sanitizes malformed replies and secret-bearing failures', async () => {
        const call = vi.mocked(getModularSupervisor().callProcess)
        call.mockResolvedValueOnce({ supported: true })
        expect(
            (await message(
                window.pairApi.engines.credentialState({ nodeId: self, engineType: 'lm-studio' })
            )) === 'Credential store unavailable'
        ).toBe(true)
        call.mockRejectedValueOnce(new Error(secret))
        expect(
            (await message(
                window.pairApi.engines.setCredential({
                    nodeId: self,
                    engineType: 'lm-studio',
                    credential: secret
                })
            )) === 'Could not update credential'
        ).toBe(true)
    })
})

describe('credential form with installed KUI components', () => {
    it('preserves the unsent masked draft across an unchanged-target status rerender', async () => {
        await mount()
        const { field } = controls()
        expect(field.getAttribute('aria-label') === 'LM Studio API key').toBe(true)
        field.value = secret
        await act(async () => status())
        expect(container.querySelector('input[type=password]') === field).toBe(true)
        expect(field.value === secret).toBe(true)
        expect(textIncludes(secret)).toBe(false)
    })
    it('requires explicit Clear and clears input on success, failure and idle Cancel', async () => {
        await mount()
        const { field, save, clear, cancel } = controls()
        const call = vi.mocked(getModularSupervisor().callProcess)
        await act(async () => save.click())
        expect(call.mock.calls.length === 1).toBe(true)
        call.mockResolvedValueOnce(saved)
        field.value = secret
        await act(async () => save.click())
        expect(field.value === '' && textIncludes('Key saved')).toBe(true)
        call.mockRejectedValueOnce(new Error(secret))
        field.value = secret
        await act(async () => save.click())
        expect(
            field.value === '' &&
                textIncludes('Key saved') &&
                textIncludes('Could not update credential')
        ).toBe(true)
        field.value = secret
        await act(async () => cancel.click())
        expect(field.value === '').toBe(true)
        await act(async () => clear.click())
        const params = call.mock.calls.at(-1)?.[2]
        expect(
            typeof params === 'object' &&
                params !== null &&
                !Array.isArray(params) &&
                params.credential === ''
        ).toBe(true)
    })
    it('preserves the confirmed label when Clear fails', async () => {
        vi.mocked(getModularSupervisor().callProcess)
            .mockResolvedValueOnce(saved)
            .mockRejectedValueOnce(new Error(secret))
        await mount()
        const { field, clear } = controls()
        field.value = secret
        await act(async () => clear.click())
        expect(
            field.value === '' &&
                textIncludes('Key saved') &&
                textIncludes('Could not update credential')
        ).toBe(true)
        expect(textIncludes(secret)).toBe(false)
    })
    it('shows unavailable for initial pending/failure, recovers by reading and hides unsupported/remote controls', async () => {
        const read = deferred()
        const call = vi.mocked(getModularSupervisor().callProcess).mockReturnValueOnce(read.promise)
        await mount()
        expect(textIncludes('Credential store unavailable') && !textIncludes('No key saved')).toBe(
            true
        )
        await act(async () => read.reject(new Error(secret)))
        expect(textIncludes('Credential store unavailable') && !textIncludes('No key saved')).toBe(
            true
        )
        const retry = [...container.querySelectorAll('button')].find(
            button => button.textContent === 'Retry'
        )
        if (!retry) throw new Error('Missing credential read recovery')
        await act(async () => retry.click())
        expect(textIncludes('No key saved')).toBe(true)
        await act(async () => root.render(null))
        call.mockResolvedValueOnce({ supported: false, configured: false })
        await mount()
        expect(container.querySelector('input[type=password]') === null).toBe(true)
        await act(async () => status('lm-studio', 'peer'))
        const count = call.mock.calls.length
        const invokeCount = invokes.length
        await mount('peer')
        const credentialButtons = [...container.querySelectorAll('button')].some(button =>
            ['Save', 'Clear', 'Cancel', 'Retry'].includes(button.textContent ?? '')
        )
        expect(
            textIncludes('LM Studio') &&
                container.querySelector('input[type=password]') === null &&
                !credentialButtons &&
                !textIncludes('Credential store unavailable') &&
                !textIncludes('No key saved') &&
                !textIncludes('Key saved') &&
                invokes.length === invokeCount &&
                call.mock.calls.length === count
        ).toBe(true)
    })
    it.each(['Save', 'Clear'])(
        'serializes cancelled %s until physical settlement and fresh read',
        async action => {
            const mutation = deferred()
            const readback = deferred()
            const call = vi
                .mocked(getModularSupervisor().callProcess)
                .mockResolvedValueOnce(action === 'Save' ? absent : saved)
                .mockReturnValueOnce(mutation.promise)
                .mockReturnValueOnce(readback.promise)
            await mount()
            const { field, save, clear, cancel } = controls()
            field.value = secret
            await act(async () => (action === 'Save' ? save : clear).click())
            await act(async () => cancel.click())
            expect(field.value === '' && save.disabled && clear.disabled).toBe(true)
            expect(!textIncludes('No key saved') && !textIncludes('Key saved')).toBe(true)
            await act(async () => {
                save.click()
                clear.click()
            })
            expect(call.mock.calls.length === 2).toBe(true)
            await act(async () => mutation.resolve(action === 'Save' ? saved : absent))
            expect(
                call.mock.calls.length === 3 && call.mock.calls[2][1] === 'engine:credential-state'
            ).toBe(true)
            expect(
                save.disabled &&
                    clear.disabled &&
                    !textIncludes('No key saved') &&
                    !textIncludes('Key saved')
            ).toBe(true)
            await act(async () => readback.resolve(action === 'Save' ? saved : absent))
            expect(
                !save.disabled &&
                    !clear.disabled &&
                    textIncludes(action === 'Save' ? 'Key saved' : 'No key saved')
            ).toBe(true)
            expect(field.value === '').toBe(true)
        }
    )
    it.each([
        { action: 'Save', outcome: 'success' },
        { action: 'Save', outcome: 'failure' },
        { action: 'Clear', outcome: 'success' },
        { action: 'Clear', outcome: 'failure' }
    ])(
        'reconciles cancelled $action $outcome, unavailable read and recovery without retrying mutation',
        async ({ action, outcome }) => {
            const mutation = deferred()
            const call = vi
                .mocked(getModularSupervisor().callProcess)
                .mockResolvedValueOnce(saved)
                .mockReturnValueOnce(mutation.promise)
                .mockRejectedValueOnce(new Error(secret))
            await mount()
            const { field, save, clear, cancel } = controls()
            field.value = secret
            await act(async () => (action === 'Save' ? save : clear).click())
            await act(async () => cancel.click())
            await act(async () => {
                if (outcome === 'success') mutation.resolve(saved)
                else mutation.reject(new Error(secret))
            })
            expect(
                textIncludes('Credential store unavailable') &&
                    !textIncludes('Key saved') &&
                    !textIncludes('No key saved')
            ).toBe(true)
            const retry = [...container.querySelectorAll('button')].find(
                button => button.textContent === 'Retry'
            )
            if (!retry) throw new Error('Missing credential read recovery')
            call.mockResolvedValueOnce(saved)
            await act(async () => retry.click())
            expect(textIncludes('Key saved') && !textIncludes(secret) && field.value === '').toBe(
                true
            )
            expect(
                call.mock.calls.filter(([, method]) => method === 'engine:set-credential')
                    .length === 1
            ).toBe(true)
            expect(call.mock.calls.length === 4).toBe(true)
        }
    )
    it.each(['node', 'engine', 'owner', 'unmount'])(
        'clears unsent input on %s teardown',
        async change => {
            await mount()
            const { field } = controls()
            field.value = secret
            await act(async () => {
                if (change === 'node')
                    root.render(createElement(NodeEnginesInline, { nodeId: 'peer' }))
                if (change === 'engine') status('ollama')
                if (change === 'owner') useConnectionStore.setState({ selfId: 'new-self' })
                if (change === 'unmount') root.render(null)
            })
            expect(field.value === '').toBe(true)
            expect(textIncludes(secret)).toBe(false)
        }
    )
    it.each(['Save', 'Clear'])(
        'ignores cancelled late %s completion after navigation',
        async action => {
            const mutation = deferred()
            const call = vi
                .mocked(getModularSupervisor().callProcess)
                .mockResolvedValueOnce(saved)
                .mockReturnValueOnce(mutation.promise)
            await mount()
            const { field, save, clear, cancel } = controls()
            field.value = secret
            await act(async () => (action === 'Save' ? save : clear).click())
            await act(async () => cancel.click())
            await mount('peer')
            await mount()
            expect(textIncludes('Credential store unavailable')).toBe(true)
            await act(async () => mutation.resolve(saved))
            expect(textIncludes('No key saved') && field.value === '').toBe(true)
            expect(call.mock.calls.length === 3).toBe(true)
        }
    )
    describe.each(['Save', 'Clear'])('late %s mutation', action => {
        it.each([
            { change: 'owner', outcome: 'success' },
            { change: 'owner', outcome: 'failure' },
            { change: 'engine', outcome: 'success' },
            { change: 'engine', outcome: 'failure' },
            { change: 'unmount', outcome: 'success' },
            { change: 'unmount', outcome: 'failure' }
        ])(
            'ignores a late mutation $outcome after $change teardown',
            async ({ change, outcome }) => {
                const mutation = deferred()
                const call = vi
                    .mocked(getModularSupervisor().callProcess)
                    .mockResolvedValueOnce(saved)
                    .mockReturnValueOnce(mutation.promise)
                await mount()
                const { field, save, clear } = controls()
                field.value = secret
                await act(async () => (action === 'Save' ? save : clear).click())
                await act(async () => {
                    if (change === 'owner') useConnectionStore.setState({ selfId: 'new-self' })
                    if (change === 'engine') status('ollama')
                    if (change === 'unmount') root.render(null)
                })
                const reads = call.mock.calls.length
                await act(async () => {
                    if (outcome === 'success') mutation.resolve(saved)
                    else mutation.reject(new Error(secret))
                })
                expect(
                    call.mock.calls.length === reads && field.value === '' && !textIncludes(secret)
                ).toBe(true)
                expect(!textIncludes('Key saved')).toBe(true)
            }
        )
    })
    it.each(['success', 'failure'])(
        'ignores late reconciliation read %s after navigating to a fresh control',
        async outcome => {
            const mutation = deferred()
            const readback = deferred()
            const call = vi
                .mocked(getModularSupervisor().callProcess)
                .mockResolvedValueOnce(saved)
                .mockReturnValueOnce(mutation.promise)
                .mockReturnValueOnce(readback.promise)
            await mount()
            const { field, clear, cancel } = controls()
            field.value = secret
            await act(async () => clear.click())
            await act(async () => cancel.click())
            await act(async () => mutation.resolve(absent))
            await mount('peer')
            await mount()
            const fresh = controls().field
            fresh.value = secret
            await act(async () => {
                if (outcome === 'success') readback.resolve(saved)
                else readback.reject(new Error(secret))
            })
            expect(textIncludes('No key saved') && !textIncludes('Key saved')).toBe(true)
            expect(field.value === '' && fresh.value === secret && !textIncludes(secret)).toBe(true)
            expect(call.mock.calls.length === 4).toBe(true)
        }
    )
    it.each(['success', 'failure'])(
        'does not send the key to stores, history, pushes or structured logs on %s',
        async outcome => {
            await mount()
            const { field, save } = controls()
            const call = vi.mocked(getModularSupervisor().callProcess)
            if (outcome === 'success') call.mockResolvedValueOnce(saved)
            else call.mockRejectedValueOnce(new Error(secret))
            field.value = secret
            await act(async () => save.click())
            expect(containsSecret(useEngineStatusStore.getState())).toBe(false)
            expect(containsSecret(useConnectionStore.getState())).toBe(false)
            expect(containsSecret(usePendingActionsStore.getState())).toBe(false)
            expect(containsSecret(getModularBridgeState())).toBe(false)
            expect(containsSecret(vi.mocked(process.stdout.write).mock.calls)).toBe(false)
            expect(containsSecret(vi.mocked(process.stderr.write).mock.calls)).toBe(false)
            expect(containsSecret(vi.mocked(BoundedLogWriter.prototype.write).mock.calls)).toBe(
                false
            )
            for (const method of [
                'log',
                'info',
                'warn',
                'error',
                'debug',
                'trace'
            ] satisfies (keyof Console)[])
                expect(containsSecret(vi.mocked(console[method]).mock.calls)).toBe(false)
            expect(pushes === 0 && !pushedSecret).toBe(true)
            expect(historyCalls() === 0).toBe(true)
            expect(envelopes.every(channel => channel === 'service-bridge:invoke')).toBe(true)
            expect(
                invokes.every(
                    request =>
                        request.channel === 'engine:set-credential' ||
                        request.channel === 'engine:credential-state'
                )
            ).toBe(true)
            expect(subscriptions === 0).toBe(true)
            expect(field.value === '' && !textIncludes(secret)).toBe(true)
        }
    )
})
