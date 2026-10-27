// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { Duplex } from 'node:stream'
import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { ipcRenderer } from 'electron'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import type { ServiceBridgeInvokeRequest } from '@/shared/types/service-bridge'
import type { NodePolicyChange } from '@/shared/types/node-policy'

const boundary = vi.hoisted(() => ({
    invoke: vi.fn<typeof ipcRenderer.invoke>(),
    listeners: new Map<string, Parameters<typeof ipcRenderer.on>[1]>(),
    callProcess: vi.fn<ReturnType<typeof getModularSupervisor>['callProcess']>()
}))
vi.mock('electron', () => ({
    ipcRenderer: {
        invoke: boundary.invoke,
        on: (channel: string, listener: Parameters<typeof ipcRenderer.on>[1]) => {
            boundary.listeners.set(channel, listener)
        },
        removeListener: (channel: string) => boundary.listeners.delete(channel)
    }
}))
vi.mock('@/electron/service-bridge/modular-supervisor', () => ({
    getModularSupervisor: () => ({ callProcess: boundary.callProcess })
}))
vi.mock('@/electron/service-bridge/modular-state', () => ({
    getModularBridgeState: vi.fn(),
    isUpstreamUnreachableError: vi.fn(),
    parseServiceErrors: vi.fn()
}))
vi.mock('@/electron/service-bridge/model-catalog', () => ({ getEngineHubModels: vi.fn() }))
vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({ info: vi.fn(), warn: vi.fn(), error: vi.fn() })
}))

import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'
import {
    JsonRpcResponseError,
    JsonRpcSocketClient
} from '@/electron/service-bridge/json-rpc-client'
import { parsePolicyChange } from '@/electron/service-bridge/node-policy'
import { createPairTransport } from '@/preload/api/pair-transport.api'
import { createPairApi } from '@/ui/api/pair-api'
import { serviceTransport } from '@/ui/api/service-transport'
import { withRoles } from '@/ui/stores/node-configuration.store'
import ModelEngineBindingsEditor from '@/ui/components/RoutingSettings/ModelEngineBindingsEditor'

const REVISION = '9007199254740993'
const OPAQUE = String.raw`{"integer":9007199254740993,"decimal":0.1000000000000000055511151231257827,"nested":[-0,1.2300e+004,1e400,1e-400,{"label":"line\nquote\"slash\\unicode\u0061, { ]"}]}`
const POLICY = `{"revision":"${REVISION}","opaque":${OPAQUE},"servedModels":[],"residentModels":[]}`
const PRETTY_POLICY = String.raw`{
  "revision": "9007199254740993",
  "opaque": {
    "integer": 9007199254740993,
    "decimal": 0.1000000000000000055511151231257827,
    "nested": [
      -0,
      1.2300e+004,
      1e400,
      1e-400,
      {
        "label": "line\nquote\"slash\\unicode\u0061, { ]"
      }
    ]
  },
  "servedModels": [],
  "residentModels": []
}`
const DRAFT = [
    '{',
    `\t"revision": "${REVISION}",`,
    `\t"opaque": ${OPAQUE},`,
    '\t"servedModels": [],',
    '\t"residentModels": []',
    '}'
].join('\r\n')

/** Only the service transport is simulated; incoming frames remain raw JSON. */
class PolicyService extends Duplex {
    readonly writes: string[] = []
    policy = POLICY
    extraFields = ''
    error: { code: number; message: string } | null = null

    override _read(): void {
        // Replies and notifications are supplied directly by this transport.
    }

    override _write(chunk: Buffer, _encoding: BufferEncoding, done: () => void): void {
        const line = chunk.toString()
        const request: { id: number; method: string } = JSON.parse(line)
        this.writes.push(line)
        const response = this.error
            ? `"error":${JSON.stringify(this.error)}`
            : `"result":{"policy":${this.policy},"availability":"available"${this.extraFields}}`
        const raw = `{"jsonrpc":"2.0","id":${request.id},${response}}\n`
        // Include actual transport chunking, including a split within a number.
        const split = raw.indexOf('9007199254740993') + 7
        this.push(raw.slice(0, split))
        this.push(raw.slice(split))
        done()
    }
}

let service: PolicyService
let client: JsonRpcSocketClient
let preload = createPairTransport(ipcRenderer)
let api = createPairApi(serviceTransport(preload))

beforeEach(async () => {
    boundary.invoke.mockReset()
    boundary.callProcess.mockReset()
    boundary.listeners.clear()
    service = new PolicyService()
    client = new JsonRpcSocketClient(
        'broker',
        async () => service,
        [],
        async () => service
    )
    client.start()
    await Promise.resolve()
    boundary.callProcess.mockImplementation((_process, method, params) =>
        client.call(method, params)
    )
    boundary.invoke.mockImplementation(
        async (
            channel: string,
            request:
                | ServiceBridgeInvokeRequest<'policy:get'>
                | ServiceBridgeInvokeRequest<'policy:set'>
        ) => {
            if (channel !== 'service-bridge:invoke') throw new Error('Unexpected IPC channel')
            if (!request.payload) throw new Error('Missing IPC payload')
            try {
                const data =
                    request.channel === 'policy:get'
                        ? await handleServiceBridgeInvoke(request.channel, request.payload)
                        : await handleServiceBridgeInvoke(request.channel, request.payload)
                return { success: true, data }
            } catch (error) {
                return {
                    success: false,
                    error: error instanceof Error ? error.message : String(error),
                    ...(error instanceof JsonRpcResponseError ? { code: error.code } : {})
                }
            }
        }
    )
    preload = createPairTransport(ipcRenderer)
    api = createPairApi(serviceTransport(preload))
})

afterEach(async () => {
    preload.destroy()
    await client.detach()
})

describe('opaque policy JSON through renderer, preload, bridge and socket client', () => {
    it('preserves opaque values through a role edit and the real bridge', async () => {
        const edited = withRoles(POLICY, { weak: { engine: 'llamacpp', model: 'native' } })
        const expected =
            POLICY.slice(0, -1) + ',"roles":{"weak":{"engine":"llamacpp","model":"native"}}}'
        expect(edited).toBe(expected)
        await api.nodePolicy.set({ policy: edited, expectedPersistedRevision: REVISION })
        expect(service.writes).toEqual([
            `{"jsonrpc":"2.0","id":1,"method":"policy:set","params":{"expectedPersistedRevision":"${REVISION}","policy":${expected}}}\n`
        ])
    })
    it('preserves opaque values through a binding UI edit and the real bridge', async () => {
        const source =
            POLICY.slice(0, -1) + ',"engineBindings":[{"engine":"llamacpp","model":"native"}]}'
        let edited = source
        const container = document.createElement('div')
        document.body.appendChild(container)
        const root = createRoot(container)
        try {
            await act(async () =>
                root.render(
                    createElement(ModelEngineBindingsEditor, {
                        targetId: 'self',
                        text: source,
                        disabled: false,
                        onTextChange: text => {
                            edited = text
                        }
                    })
                )
            )
            const clear = Array.from(container.querySelectorAll('button')).find(
                button => button.textContent === 'Clear bindings'
            )
            if (!clear) throw new Error('Missing binding control')
            await act(async () => clear.click())
            const expected = POLICY.slice(0, -1) + ',"engineBindings":[]}'
            expect(edited).toBe(expected)
            await api.nodePolicy.set({ policy: edited, expectedPersistedRevision: REVISION })
            expect(service.writes).toEqual([
                `{"jsonrpc":"2.0","id":1,"method":"policy:set","params":{"expectedPersistedRevision":"${REVISION}","policy":${expected}}}\n`
            ])
        } finally {
            await act(async () => root.unmount())
            container.remove()
        }
    })
    it.each([
        ['local', undefined],
        ['remote', 'peer']
    ])('round trips every number lexeme for %s policy get and set', async (_name, nodeId) => {
        const document = await api.nodePolicy.get(nodeId)
        expect(document.policy).toBe(PRETTY_POLICY)
        expect(document.revision).toBe(REVISION)
        const saved = await api.nodePolicy.set({
            nodeId,
            policy: document.policy,
            expectedPersistedRevision: document.revision
        })
        expect(saved.policy).toBe(PRETTY_POLICY)
        expect(saved.revision).toBe(REVISION)
        const target = nodeId ? `"nodeId":"${nodeId}",` : ''
        expect(service.writes).toEqual([
            `{"jsonrpc":"2.0","id":1,"method":"policy:get","params":{${target.slice(0, -1)}}}\n`,
            `{"jsonrpc":"2.0","id":2,"method":"policy:set","params":{${target}"expectedPersistedRevision":"${REVISION}","policy":${POLICY}}}\n`
        ])
    })

    it('compacts multiline drafts to one wire line while retaining string escapes', async () => {
        await api.nodePolicy.set({ policy: DRAFT, expectedPersistedRevision: REVISION })
        expect(service.writes).toEqual([
            `{"jsonrpc":"2.0","id":1,"method":"policy:set","params":{"expectedPersistedRevision":"${REVISION}","policy":${POLICY}}}\n`
        ])
        expect(service.writes[0].match(/\n/g)).toHaveLength(1)
        expect(service.writes[0]).not.toContain('\r')
    })

    it('preserves local and remote policy notification text through preload subscribers', async () => {
        const changes: NodePolicyChange[] = []
        api.nodePolicy.onPolicyChanged(change => changes.push(change))
        client.on('notification', notification => {
            const change = parsePolicyChange(notification.params)
            if (!change) return
            const listener = boundary.listeners.get('service-bridge:push')
            if (!listener) throw new Error('Missing preload subscriber')
            Reflect.apply(listener, undefined, [{}, { channel: 'policy:changed', payload: change }])
        })
        service.push(
            `{"jsonrpc":"2.0","method":"policy:changed","params":{"nodeId":"self","policy":${POLICY}}}\n`
        )
        service.push(
            `{"jsonrpc":"2.0","method":"policy:changed","params":{"nodeId":"peer","policy":${POLICY}}}\n`
        )
        await new Promise<void>(resolve => setImmediate(resolve))
        expect(changes).toEqual(
            ['self', 'peer'].map(nodeId => ({
                nodeId,
                policy: PRETTY_POLICY,
                revision: REVISION,
                activation: null
            }))
        )
    })

    it.each([
        ['get', 'policy:get'],
        ['set', 'policy:set']
    ])('rejects an object encoded as a string in a wire %s result', async (_name, method) => {
        service.policy = JSON.stringify(POLICY)
        const call =
            method === 'policy:get'
                ? api.nodePolicy.get()
                : api.nodePolicy.set({ policy: POLICY, expectedPersistedRevision: REVISION })
        await expect(call).rejects.toThrow('The service returned no node policy')
    })

    it('drops a string-valued policy in a wire notification', async () => {
        const changes: (NodePolicyChange | null)[] = []
        client.on('notification', notification =>
            changes.push(parsePolicyChange(notification.params))
        )
        service.push(
            `{"jsonrpc":"2.0","method":"policy:changed","params":{"nodeId":"peer","policy":${JSON.stringify(POLICY)}}}\n`
        )
        await new Promise<void>(resolve => setImmediate(resolve))
        expect(changes).toEqual([null])
    })

    it('retains duplicate keys for authoritative raw-policy validation', async () => {
        const policy = `{"revision":"${REVISION}","servedModels":[],"servedModels":[]}`
        await api.nodePolicy.set({ policy, expectedPersistedRevision: REVISION })
        expect(service.writes).toEqual([
            `{"jsonrpc":"2.0","id":1,"method":"policy:set","params":{"expectedPersistedRevision":"${REVISION}","policy":${policy}}}\n`
        ])
    })

    it('preserves the last object policy in a result with duplicate envelope keys', async () => {
        service.policy = '{"revision":"1"}'
        service.extraFields = `,"pol\\u0069cy":${POLICY}`
        const document = await api.nodePolicy.get('peer')
        expect(document.policy).toBe(PRETTY_POLICY)
        expect(document.revision).toBe(REVISION)
    })

    it.each([
        ['syntax', '{oops'],
        ['array', '[]'],
        ['model preferences', '{"servedModels":[],"ServedModels":[]}'],
        [
            'resident permission',
            '{"servedModels":[],"residentModels":[{"engine":"ollama","model":"m"}]}'
        ]
    ])('rejects invalid %s before writing a policy request', async (_name, policy) => {
        await expect(
            api.nodePolicy.set({ policy, expectedPersistedRevision: REVISION })
        ).rejects.toThrow()
        expect(service.writes).toEqual([])
    })

    it('rejects an invalid CAS revision before writing a policy request', async () => {
        await expect(
            api.nodePolicy.set({ policy: POLICY, expectedPersistedRevision: '01' })
        ).rejects.toThrow('must be a decimal string')
        expect(service.writes).toEqual([])
    })

    it('rejects malformed model preferences in incoming readback and notifications', async () => {
        service.policy = `{"revision":"${REVISION}","servedModels":[],"ServedModels":[]}`
        await expect(api.nodePolicy.get('peer')).rejects.toThrow('exact field name')
        await expect(
            api.nodePolicy.set({
                nodeId: 'peer',
                policy: POLICY,
                expectedPersistedRevision: REVISION
            })
        ).rejects.toThrow('exact field name')
        const changes: (NodePolicyChange | null)[] = []
        client.on('notification', notification =>
            changes.push(parsePolicyChange(notification.params))
        )
        service.push(
            `{"jsonrpc":"2.0","method":"policy:changed","params":{"nodeId":"peer","policy":${service.policy}}}\n`
        )
        await new Promise<void>(resolve => setImmediate(resolve))
        expect(changes).toEqual([null])
    })

    it.each([
        ['invalid parameters', -32602],
        ['revision conflict', -32000],
        ['local veto', -32011],
        ['remote management denied', -32012]
    ])('keeps the %s error code across the bridge', async (_name, code) => {
        service.error = { code, message: 'refused' }
        await expect(
            api.nodePolicy.set({
                nodeId: 'peer',
                policy: POLICY,
                expectedPersistedRevision: REVISION
            })
        ).rejects.toMatchObject({ code, message: `${code}: refused` })
    })

    it('keeps model observation counters numeric at the safe bound', async () => {
        service.extraFields = `,"modelConfiguration":{"policyRevision":"${REVISION}","observationGeneration":9007199254740991,"models":[]}`
        const document = await api.nodePolicy.get()
        expect(document.modelConfiguration).toEqual({
            policyRevision: REVISION,
            observationGeneration: Number.MAX_SAFE_INTEGER,
            models: []
        })
    })

    it('rejects model observation counters above the safe bound on real ingress', async () => {
        service.extraFields = `,"modelConfiguration":{"policyRevision":"${REVISION}","observationGeneration":9007199254740993,"models":[]}`
        await expect(api.nodePolicy.get()).rejects.toThrow('Invalid model configuration')
    })
})
