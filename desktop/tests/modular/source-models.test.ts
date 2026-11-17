// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import {
    parseImportedModel,
    parseLinkedModel,
    parseSourceModels
} from '@/electron/service-bridge/source-models'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi } from '@/ui/api/pair-api'

describe('source model relay', () => {
    it('keeps readable rows and the linked-list error', () => {
        expect(
            parseSourceModels({
                models: [
                    {
                        id: 'lmstudio/pub/repo/Model',
                        source: 'lmstudio',
                        name: 'pub/repo/Model',
                        sizeBytes: 10,
                        shards: 1,
                        hasProjector: true
                    },
                    {
                        id: 'local/Big',
                        source: 'local',
                        name: 'Big-00001-of-00002.gguf',
                        shards: 2
                    },
                    { id: 'Mine', source: 'copy', name: 'Mine', sizeBytes: 5, shards: 1 },
                    { id: 'x', source: 'elsewhere' },
                    { source: 'ollama' }
                ],
                error: 'linked-models.json is not a valid linked-model list; fix or remove it'
            })
        ).toEqual({
            models: [
                {
                    id: 'lmstudio/pub/repo/Model',
                    source: 'lmstudio',
                    name: 'pub/repo/Model',
                    sizeBytes: 10,
                    shards: 1,
                    hasProjector: true
                },
                {
                    id: 'local/Big',
                    source: 'local',
                    name: 'Big-00001-of-00002.gguf',
                    sizeBytes: 0,
                    shards: 2,
                    hasProjector: false
                },
                {
                    id: 'Mine',
                    source: 'copy',
                    name: 'Mine',
                    sizeBytes: 5,
                    shards: 1,
                    hasProjector: false
                }
            ],
            error: 'linked-models.json is not a valid linked-model list; fix or remove it'
        })
    })

    it('rejects malformed replies', () => {
        expect(() => parseSourceModels({ rows: [] })).toThrow('malformed')
        expect(() => parseLinkedModel({})).toThrow('malformed')
        expect(parseLinkedModel({ id: 'local/Model' })).toEqual({ id: 'local/Model' })
        expect(() => parseImportedModel({ id: 'Model', layout: 'x', exportId: 'y' })).toThrow(
            'malformed'
        )
        expect(
            parseImportedModel({
                id: 'Model',
                layout: 'llama-artifacts-v1',
                exportId: 'e',
                durability: 'confirmed'
            })
        ).toEqual({
            id: 'Model',
            layout: 'llama-artifacts-v1',
            exportId: 'e',
            durability: 'confirmed'
        })
    })

    it('names the node for listings and only the engine type for local link changes', async () => {
        const invoke = vi.fn<ServiceTransport['invoke']>(() => Promise.resolve(null))
        const transport: ServiceTransport = {
            connected: true,
            invoke,
            subscribePush: () => () => {},
            onConnect: () => () => {},
            onDisconnect: () => () => {},
            onAuthFailure: () => () => {},
            destroy: () => {}
        }
        const api = createPairApi(transport)
        await api.engines.sourceModels('joes', 'llama-cpp')
        await api.engines.linkModel('llama-cpp', 'C:\\models\\Model.gguf')
        await api.engines.unlinkModel('llama-cpp', 'local/Model')
        await api.engines.importModel(
            'llama-cpp',
            { sourceId: 'ollama/tiny@latest' },
            'tiny-latest'
        )
        expect(invoke.mock.calls).toEqual([
            ['engine:source-models', { nodeId: 'joes', engineType: 'llama-cpp' }],
            ['engine:link-model', { engineType: 'llama-cpp', path: 'C:\\models\\Model.gguf' }],
            ['engine:unlink-model', { engineType: 'llama-cpp', id: 'local/Model' }],
            [
                'engine:import-model',
                {
                    engineType: 'llama-cpp',
                    source: { sourceId: 'ollama/tiny@latest' },
                    id: 'tiny-latest'
                }
            ]
        ])
    })
})
