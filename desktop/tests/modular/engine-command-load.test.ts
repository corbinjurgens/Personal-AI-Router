// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
    state: {
        getSelfId: vi.fn(() => 'local-node')
    },
    supervisor: {
        hasProcess: vi.fn(() => true),
        sendProcess: vi.fn(),
        reportError: vi.fn()
    }
}))

vi.mock('@/electron/service-bridge/modular-supervisor', () => ({
    getModularSupervisor: () => mocks.supervisor
}))
vi.mock('@/electron/service-bridge/modular-state', () => ({
    getModularBridgeState: () => mocks.state,
    isProxyEngine: () => false,
    isUpstreamUnreachableError: () => false,
    parseServiceErrors: () => []
}))
vi.mock('@/electron/service-bridge/model-catalog', () => ({ getEngineHubModels: vi.fn() }))

import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'

describe('local model load command', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        mocks.state.getSelfId.mockReturnValue('local-node')
        mocks.supervisor.hasProcess.mockReturnValue(true)
    })

    it('observes and attributes a llama.cpp load rejection to the pending model row', async () => {
        await handleServiceBridgeInvoke('engine:command', {
            command: 'loadModel',
            engineType: 'llama-cpp',
            nodeId: 'local-node',
            model: 'llama3.2'
        })

        expect(mocks.supervisor.sendProcess).toHaveBeenCalledOnce()
        const [name, method, params, onFailure, observeResponse] =
            mocks.supervisor.sendProcess.mock.calls[0]
        expect([name, method, params, observeResponse]).toEqual([
            'broker',
            'engine:load-model',
            { engine: 'llamacpp', model: 'llama3.2' },
            true
        ])

        onFailure('timeout awaiting response headers', true)
        expect(mocks.supervisor.reportError).toHaveBeenCalledWith(
            'Failed to load model on llamacpp: timeout awaiting response headers',
            'error',
            'engine-cmd:load model:llamacpp',
            {
                nodeId: 'local-node',
                engineType: 'llama-cpp',
                operation: 'load',
                modelName: 'llama3.2'
            }
        )
    })

    it('loads every engine through the typed operation, never an engine:action body', async () => {
        await handleServiceBridgeInvoke('engine:command', {
            command: 'loadModel',
            engineType: 'llama-cpp',
            nodeId: 'local-node',
            model: 'publisher/demo'
        })

        expect(mocks.supervisor.sendProcess).toHaveBeenCalledOnce()
        expect(mocks.supervisor.sendProcess).toHaveBeenCalledWith(
            'broker',
            'engine:load-model',
            { engine: 'llamacpp', model: 'publisher/demo' },
            expect.any(Function),
            true
        )
    })

    it('routes llama.cpp load and unload through the typed operations', async () => {
        await handleServiceBridgeInvoke('engine:command', {
            command: 'loadModel',
            engineType: 'llama-cpp',
            nodeId: 'local-node',
            model: 'ggml-org/gemma-3-1b-it-GGUF:Q4_K_M'
        })
        await handleServiceBridgeInvoke('engine:command', {
            command: 'unloadModel',
            engineType: 'llama-cpp',
            nodeId: 'local-node',
            model: 'ggml-org/gemma-3-1b-it-GGUF:Q4_K_M'
        })

        expect(mocks.supervisor.sendProcess).toHaveBeenNthCalledWith(
            1,
            'broker',
            'engine:load-model',
            { engine: 'llamacpp', model: 'ggml-org/gemma-3-1b-it-GGUF:Q4_K_M' },
            expect.any(Function),
            true
        )
        expect(mocks.supervisor.sendProcess).toHaveBeenNthCalledWith(
            2,
            'broker',
            'engine:unload-model',
            { engine: 'llamacpp', model: 'ggml-org/gemma-3-1b-it-GGUF:Q4_K_M' },
            expect.any(Function)
        )
    })
})
