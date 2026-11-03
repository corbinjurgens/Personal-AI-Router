// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
    supervisor: {
        callProcess: vi.fn()
    }
}))

vi.mock('@/electron/service-bridge/modular-supervisor', () => ({
    getModularSupervisor: () => mocks.supervisor
}))

import { getEngineHubModels, warmEngineHubs } from '@/electron/service-bridge/model-catalog'
import { MODULAR_CATALOG_CALL_TIMEOUT_MS } from '@/shared/constants/modular-runtime'

describe('model catalog relay', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        mocks.supervisor.callProcess.mockResolvedValue({ models: [] })
    })

    // The renderer spells the engine `llama-cpp`; the engine manager keys on
    // its manifest name. No target machine is named, because the hub only
    // installs to this machine and naming the OS alone would lose the CPU.
    it('asks the backend in its own vocabulary, for this machine', async () => {
        await getEngineHubModels('llama-cpp')
        expect(mocks.supervisor.callProcess).toHaveBeenCalledWith(
            'broker',
            'engine:catalog',
            { engine: 'llamacpp' },
            MODULAR_CATALOG_CALL_TIMEOUT_MS
        )
    })

    // The backend decides whether a source can search; the relay only carries
    // the query, and a blank one is no query at all.
    it('passes a search query through, trimmed, and leaves a blank one out', async () => {
        await getEngineHubModels('llama-cpp', '  gemma  ')
        expect(mocks.supervisor.callProcess).toHaveBeenLastCalledWith(
            'broker',
            'engine:catalog',
            { engine: 'llamacpp', query: 'gemma' },
            MODULAR_CATALOG_CALL_TIMEOUT_MS
        )
        await getEngineHubModels('llama-cpp', '   ')
        expect(mocks.supervisor.callProcess).toHaveBeenLastCalledWith(
            'broker',
            'engine:catalog',
            { engine: 'llamacpp' },
            MODULAR_CATALOG_CALL_TIMEOUT_MS
        )
    })

    it('keeps only rows that can be pulled, and drops empty optional fields', async () => {
        mocks.supervisor.callProcess.mockResolvedValue({
            models: [
                {
                    id: 'qwen3:8b',
                    name: 'qwen3:8b',
                    author: 'ggml-org',
                    url: 'https://huggingface.co/ggml-org/qwen3-GGUF',
                    size: 5000,
                    downloads: 3,
                    likes: 1,
                    updatedAt: '2026-07-21T00:00:00Z',
                    tags: ['qwen3', 7, 'Q4_K_M'],
                    family: 'qwen3',
                    parameterSize: '8B'
                },
                // A name alone is still pull-ready.
                { name: 'phi4', size: 0, family: '', parameterSize: '' },
                // No id and no name: nothing to hand to a pull.
                { author: 'nobody' },
                'not an object',
                null
            ]
        })

        const { models } = await getEngineHubModels('llama-cpp')

        expect(models).toEqual([
            {
                id: 'qwen3:8b',
                name: 'qwen3:8b',
                author: 'ggml-org',
                url: 'https://huggingface.co/ggml-org/qwen3-GGUF',
                size: 5000,
                downloads: 3,
                likes: 1,
                updatedAt: '2026-07-21T00:00:00Z',
                tags: ['qwen3', 'Q4_K_M'],
                family: 'qwen3',
                parameterSize: '8B'
            },
            {
                id: 'phi4',
                name: 'phi4',
                author: '',
                url: '',
                size: undefined,
                downloads: 0,
                likes: 0,
                updatedAt: '',
                tags: [],
                family: undefined,
                parameterSize: undefined
            }
        ])
    })

    it('reads a reply without a model list as an empty hub', async () => {
        mocks.supervisor.callProcess.mockResolvedValue({ source: 'nowhere' })
        await expect(getEngineHubModels('llama-cpp')).resolves.toEqual({ models: [] })
    })

    // The modal renders its empty state for this; a rejection would surface as
    // a failure the user can do nothing about.
    it('turns a failed call into an empty hub rather than an error', async () => {
        mocks.supervisor.callProcess.mockRejectedValue(new Error('llama.cpp catalog unavailable'))
        await expect(getEngineHubModels('llama-cpp')).resolves.toEqual({ models: [] })
    })

    it('warms the llama.cpp catalog', () => {
        warmEngineHubs()
        expect(mocks.supervisor.callProcess).toHaveBeenCalledTimes(1)
        expect(mocks.supervisor.callProcess).toHaveBeenCalledWith(
            'broker',
            'engine:catalog',
            { engine: 'llamacpp' },
            MODULAR_CATALOG_CALL_TIMEOUT_MS
        )
    })
})
