// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))

import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import {
    isProxyEngine,
    PROXY_ENGINES,
    PROXY_NODE_SOURCES,
    proxyEngineFromManagerId,
    proxyEngineFromSource,
    proxySourceForEngine
} from '@/electron/service-bridge/proxy-engines'
import { engineManagerName } from '@/shared/utils/engines'
import type { JsonValue } from '@/shared/types/json'

describe('proxy engine identity', () => {
    for (const engine of ['constructor', 'toString', '__proto__']) {
        it(`projects own model and loaded members for dynamic engine ${engine}`, () => {
            const state = getModularBridgeState()
            const nodeId = `j304-own-member-${engine}`
            state.setSelfId('j304-own-member-self')
            const attributed = (models: string[]) => Object.fromEntries([
                [engine, models], ['ollama', ['sibling']]
            ])
            const phases: {
                name: string
                attribution: Record<string, JsonValue> | null | undefined
                loaded: Record<string, JsonValue>
                expected: { name: string; status: string }[]
            }[] = [
                { name: 'absent', attribution: undefined, loaded: {}, expected: [] },
                { name: 'null', attribution: null, loaded: {}, expected: [] },
                { name: 'outer empty', attribution: {}, loaded: {}, expected: [] },
                {
                    name: 'own empty member',
                    attribution: attributed([]),
                    loaded: Object.fromEntries([[engine, []]]),
                    expected: []
                },
                {
                    name: 'own models and missing own loaded member',
                    attribution: attributed(['good']),
                    loaded: {},
                    expected: [{ name: 'good', status: 'idle' }]
                },
                {
                    name: 'own populated model and loaded members',
                    attribution: attributed(['first', 'good', 'last']),
                    loaded: Object.fromEntries([[engine, ['good']]]),
                    expected: [
                        { name: 'first', status: 'idle' },
                        { name: 'good', status: 'loaded' },
                        { name: 'last', status: 'idle' }
                    ]
                },
                {
                    name: 'loaded own member cannot invent models',
                    attribution: {},
                    loaded: Object.fromEntries([[engine, ['good']]]),
                    expected: []
                }
            ]
            for (const phase of phases) {
                const attribution = phase.attribution
                const brokerNode: Record<string, JsonValue> = {
                    hostUuid: nodeId, name: nodeId, ipAddress: '192.0.2.41', port: 24002,
                    models: ['good', 'sibling'], loadedByEngine: phase.loaded, lastSeen: 304
                }
                if (attribution !== undefined) brokerNode.modelsByEngine = attribution
                state.handleNotification({
                    source: 'broker', method: 'discovery:nodes-changed', params: [brokerNode]
                })
                state.applyRemoteEngineFacts(nodeId, {
                    engines: [
                        { engine, installed: true, running: true, healthy: true, port: 24003 },
                        { engine: 'ollama', installed: true, running: true, healthy: true, port: 24004 }
                    ]
                })
                const snapshot = state.getEngineInitialState()
                expect(snapshot.models.find(entry => entry.nodeId === nodeId && entry.engineType === engine)?.models,
                    phase.name).toEqual(phase.expected.map(model => expect.objectContaining(model)))
                expect(snapshot.statuses.find(entry => entry.nodeId === nodeId && entry.engineType === engine),
                    phase.name).toMatchObject({ processStatus: 'running', enginePort: 24003 })
                if (attribution !== undefined && attribution !== null && Object.hasOwn(attribution, 'ollama')) {
                    expect(snapshot.models.find(entry => entry.nodeId === nodeId && entry.engineType === 'ollama')?.models,
                        phase.name).toEqual([expect.objectContaining({ name: 'sibling', status: 'idle' })])
                }
            }
        })
    }

    const attributionCases: {
        name: string
        attribution: JsonValue | undefined
        expected: string[]
    }[] = [
        {
            name: 'absent attribution retains single-engine fallback',
            attribution: undefined,
            expected: ['good']
        },
        {
            name: 'null attribution retains single-engine fallback',
            attribution: null,
            expected: ['good']
        },
        { name: 'outer empty attribution suppresses flat fallback', attribution: {}, expected: [] },
        {
            name: 'per-engine empty attribution suppresses flat fallback',
            attribution: { ollama: [] },
            expected: []
        },
        {
            name: 'populated attribution preserves siblings and loaded state',
            attribution: { ollama: ['first', 'good', 'last'] },
            expected: ['first', 'good', 'last']
        },
        {
            name: 'unknown engine attribution suppresses flat fallback',
            attribution: { 'future-engine': ['other'] },
            expected: []
        },
        {
            name: 'malformed present string cannot create flat fallback',
            attribution: 'invalid',
            expected: []
        },
        { name: 'malformed present array cannot create flat fallback', attribution: [], expected: [] },
        {
            name: 'malformed present boolean cannot create flat fallback',
            attribution: false,
            expected: []
        },
        {
            name: 'malformed present number cannot create flat fallback',
            attribution: 7,
            expected: []
        },
        {
            name: 'malformed present engine list cannot create flat fallback',
            attribution: { ollama: 'invalid' },
            expected: []
        }
    ]
    for (const [index, tc] of attributionCases.entries()) {
        it(tc.name, () => {
            const state = getModularBridgeState()
            const nodeId = `j304-attribution-${index}`
            state.setSelfId('j304-attribution-self')
            const proxy = {
                id: nodeId,
                host: nodeId,
                port: 24001,
                addresses: ['192.0.2.40'],
                ip: '192.0.2.40'
            }
            state.handleNotification({
                source: 'ollama-proxy', method: 'node/discovered', params: proxy
            })
            const brokerNode: Record<string, JsonValue> = {
                hostUuid: nodeId,
                name: nodeId,
                ipAddress: '192.0.2.40',
                port: 24002,
                models: ['good'],
                loadedByEngine: { ollama: ['good'] },
                lastSeen: 304
            }
            if (tc.attribution !== undefined) brokerNode.modelsByEngine = tc.attribution
            state.handleNotification({
                source: 'broker',
                method: 'discovery:nodes-changed',
                params: [brokerNode]
            })
            const read = () =>
                state.getEngineInitialState().models.find(
                    models => models.nodeId === nodeId && models.engineType === 'ollama'
                )
            expect(read()?.models.map(model => model.name)).toEqual(tc.expected)
            expect(read()?.models.map(model => model.status)).toEqual(
                tc.expected.map(model => (model === 'good' ? 'loaded' : 'idle'))
            )
            expect(state.getEngineInitialState().statuses.find(
                status => status.nodeId === nodeId && status.engineType === 'ollama'
            )).toMatchObject({ processStatus: 'running', proxyPort: 24001 })
            // A later presence-only proxy update cannot overwrite attribution.
            state.handleNotification({
                source: 'ollama-proxy', method: 'node/updated', params: proxy
            })
            expect(read()?.models.map(model => model.name)).toEqual(tc.expected)
        })
    }

    it('publishes attribution presence changes without changing flat models or loaded state', () => {
        const state = getModularBridgeState()
        const nodeId = 'j304-attribution-transition'
        state.setSelfId('j304-attribution-self')
        state.handleNotification({
            source: 'ollama-proxy', method: 'node/discovered',
            params: { id: nodeId, host: nodeId, port: 24001, addresses: ['192.0.2.40'] }
        })
        const node = {
            hostUuid: nodeId, name: nodeId, ipAddress: '192.0.2.40', port: 24002,
            models: ['good'], loadedByEngine: { ollama: ['good'] }, lastSeen: 304
        }
        const models = () => state.getEngineInitialState().models.find(
            entry => entry.nodeId === nodeId && entry.engineType === 'ollama'
        )?.models
        state.handleNotification({ source: 'broker', method: 'discovery:nodes-changed', params: [node] })
        expect(models()).toEqual([expect.objectContaining({ name: 'good', status: 'loaded' })])
        state.handleNotification({ source: 'broker', method: 'discovery:nodes-changed', params: [{ ...node, modelsByEngine: {} }] })
        expect(models()).toEqual([])
        state.handleNotification({ source: 'broker', method: 'discovery:nodes-changed', params: [node] })
        expect(models()).toEqual([expect.objectContaining({ name: 'good', status: 'loaded' })])
    })

    it('refreshes local model cache on attribution presence changes only once', () => {
        const state = getModularBridgeState()
        const nodeId = 'j304-local-attribution-transition'
        state.setSelfId(nodeId)
        state.applyEngineManagerStatus({
            engine: 'ollama', installed: true, running: true, healthy: true, port: 24003
        })
        const node = {
            hostUuid: nodeId, name: nodeId, ipAddress: '127.0.0.1', port: 24002,
            models: ['good'], loadedByEngine: { ollama: ['good'] }, lastSeen: 304
        }
        state.handleNotification({
            source: 'broker', method: 'discovery:nodes-changed', params: [node]
        })
        const refresh = vi.fn<Parameters<typeof state.setLocalDiscoveryModelRefresher>[0]>()
        state.setLocalDiscoveryModelRefresher(refresh)
        try {
            state.handleNotification({
                source: 'broker', method: 'discovery:nodes-changed',
                params: [{ ...node, modelsByEngine: {} }]
            })
            expect(refresh).toHaveBeenCalledTimes(1)
            expect(refresh).toHaveBeenCalledWith('ollama')
            refresh.mockClear()
            state.handleNotification({
                source: 'broker', method: 'discovery:nodes-changed',
                params: [{ ...node, modelsByEngine: {} }]
            })
            expect(refresh).not.toHaveBeenCalled()
            state.handleNotification({
                source: 'broker', method: 'discovery:nodes-changed', params: [node]
            })
            expect(refresh).toHaveBeenCalledTimes(1)
            expect(refresh).toHaveBeenCalledWith('ollama')
        } finally {
            state.setLocalDiscoveryModelRefresher(() => {})
        }
    })

    it('keeps legacy multi-engine fallback conservative and explicit attribution useful', () => {
        const state = getModularBridgeState()
        const nodeId = 'j304-attribution-multiple-engines'
        state.setSelfId('j304-attribution-self')
        for (const source of PROXY_NODE_SOURCES.slice(0, 2)) {
            state.handleNotification({
                source, method: 'node/discovered',
                params: { id: nodeId, host: nodeId, port: 24001, addresses: ['192.0.2.40'] }
            })
        }
        const node = {
            hostUuid: nodeId, name: nodeId, ipAddress: '192.0.2.40', port: 24002,
            models: ['good', 'other'], loadedByEngine: { ollama: ['good'], lmstudio: [] }, lastSeen: 304
        }
        const models = () => state.getEngineInitialState().models.filter(entry => entry.nodeId === nodeId)
        state.handleNotification({ source: 'broker', method: 'discovery:nodes-changed', params: [node] })
        expect(models().find(entry => entry.engineType === 'ollama')?.models).toEqual([])
        expect(models().find(entry => entry.engineType === 'lm-studio')?.models).toEqual([])
        state.handleNotification({
            source: 'broker', method: 'discovery:nodes-changed',
            params: [{ ...node, modelsByEngine: { ollama: ['good'], lmstudio: ['other'] } }]
        })
        expect(models().find(entry => entry.engineType === 'ollama')?.models).toEqual([
            expect.objectContaining({ name: 'good', status: 'loaded' })
        ])
        expect(models().find(entry => entry.engineType === 'lm-studio')?.models).toEqual([
            expect.objectContaining({ name: 'other', status: 'idle' })
        ])
    })

    it('declares the complete proxy set in stable order', () => {
        expect(PROXY_ENGINES).toEqual(['ollama', 'lm-studio', 'llama-cpp'])
        expect(PROXY_NODE_SOURCES).toEqual(['ollama-proxy', 'lmstudio-proxy', 'llamacpp-proxy'])
        expect(isProxyEngine('ollama')).toBe(true)
        expect(isProxyEngine('lm-studio')).toBe(true)
        expect(isProxyEngine('llama-cpp')).toBe(true)
    })

    it('round-trips engine, manager, and relay source identities', () => {
        for (const engine of PROXY_ENGINES) {
            const source = proxySourceForEngine(engine)
            expect(proxyEngineFromSource(source)).toBe(engine)
            expect(proxyEngineFromManagerId(engineManagerName(engine))).toBe(engine)
        }
        expect(proxyEngineFromSource('other-proxy')).toBeNull()
        expect(proxyEngineFromManagerId('other')).toBeNull()
        expect(proxyEngineFromManagerId('llamacpp')).toBe('llama-cpp')
    })

    it('routes each proxy notification to the mapped engine', () => {
        const state = getModularBridgeState()
        const nodeId = 'proxy-identity-remote'
        state.setSelfId('proxy-identity-self')

        state.handleNotification({
            source: 'ollama-proxy',
            method: 'node/discovered',
            params: {
                id: nodeId,
                host: 'proxy-identity-host',
                port: 11434,
                addresses: ['192.0.2.40'],
                ip: '192.0.2.40'
            }
        })
        state.handleNotification({
            source: 'lmstudio-proxy',
            method: 'node/discovered',
            params: {
                id: nodeId,
                host: 'proxy-identity-host',
                port: 1234,
                addresses: ['192.0.2.40'],
                ip: '192.0.2.40'
            }
        })
        state.handleNotification({
            source: 'llamacpp-proxy',
            method: 'ready',
            params: { port: 8080 }
        })
        state.handleNotification({
            source: 'llamacpp-proxy',
            method: 'node/discovered',
            params: {
                id: nodeId,
                host: 'proxy-identity-host',
                port: 8080,
                addresses: ['192.0.2.40'],
                ip: '192.0.2.40'
            }
        })
        state.handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: {
                nodes: [
                    {
                        hostUuid: nodeId,
                        name: 'proxy-identity-host',
                        ipAddress: '192.0.2.40',
                        port: 14318,
                        models: ['owner/alpha:Q4_K_M', 'owner/beta:Q4_K_M'],
                        modelsByEngine: {
                            llamacpp: ['owner/alpha:Q4_K_M', 'owner/beta:Q4_K_M']
                        },
                        loadedByEngine: { llamacpp: ['owner/alpha:Q4_K_M'] }
                    }
                ]
            }
        })
        state.applyRemoteEngineFacts(nodeId, {
            engines: [
                {
                    engine: 'llamacpp',
                    installed: true,
                    running: true,
                    healthy: true,
                    port: 8081
                }
            ]
        })

        const statuses = state
            .getEngineInitialState()
            .statuses.filter(status => status.nodeId === nodeId)
        expect(statuses).toEqual([
            expect.objectContaining({
                engineType: 'ollama',
                processStatus: 'running',
                proxyPort: 11434
            }),
            expect.objectContaining({
                engineType: 'lm-studio',
                processStatus: 'running',
                proxyPort: 1234
            }),
            expect.objectContaining({
                engineType: 'llama-cpp',
                processStatus: 'running',
                enginePort: 8081,
                proxyPort: 8080
            })
        ])

        const llamaModels = state
            .getEngineInitialState()
            .models.find(models => models.nodeId === nodeId && models.engineType === 'llama-cpp')
        expect(llamaModels?.models).toEqual([
            expect.objectContaining({ name: 'owner/alpha:Q4_K_M', status: 'loaded' }),
            expect.objectContaining({ name: 'owner/beta:Q4_K_M', status: 'idle' })
        ])
    })

    it('projects local llama.cpp lifecycle and residency into the initial snapshot', () => {
        const state = getModularBridgeState()
        const nodeId = 'proxy-identity-local'
        const model = 'ggml-org/gemma-3-1b-it-GGUF:Q4_K_M'
        state.setSelfId(nodeId)
        state.handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: {
                nodes: [
                    {
                        hostUuid: nodeId,
                        name: 'proxy-identity-local-host',
                        ipAddress: '127.0.0.1',
                        port: 14318
                    }
                ]
            }
        })
        state.handleNotification({
            source: 'llamacpp-proxy',
            method: 'ready',
            params: { port: 8080 }
        })
        state.applyEngineManagerStatus({
            engine: 'llamacpp',
            installed: true,
            running: true,
            healthy: true,
            port: 8081
        })
        state.setLocalEngineModels('llama-cpp', [model])
        state.applyLocalLoadedModels({ llamacpp: [model] })

        const initial = state.getEngineInitialState()
        expect(
            initial.statuses.find(
                status => status.nodeId === nodeId && status.engineType === 'llama-cpp'
            )
        ).toMatchObject({
            processStatus: 'running',
            enginePort: 8081,
            proxyPort: 8080
        })
        expect(
            initial.models.find(
                models => models.nodeId === nodeId && models.engineType === 'llama-cpp'
            )?.models
        ).toEqual([expect.objectContaining({ name: model, status: 'loaded' })])
    })
})
