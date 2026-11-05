// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { EngineModels, ModelIdentity, ModelItem } from '@/shared/types/engines'
import type { Workload } from '@/shared/types/workloads'
import { ModelIdentityTags } from '@/ui/components/ModelIdentity/ModelIdentityTags'
import ModelRow from '@/ui/components/ModelManager/ModelRow'
import WorkloadItemCard from '@/ui/components/Workloads/WorkloadItemCard'
import { SAFE_DEFAULT_ENGINE_CAPS } from '@/ui/constants/engine-capabilities'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { modelIdentityTags, modelPrimaryLabel } from '@/ui/utils/model-identity-label'
import { customEngine } from './engine-fixtures'

const identity: ModelIdentity = {
    repository: 'unsloth/LFM2.5-24B-A2B-GGUF',
    quantization: 'UD-Q4_K_XL',
    quantizationSource: 'hf-tag',
    generalName: 'LFM2.5',
    configuredContextTokens: 131072,
    strength: 'declared'
}

describe('identity primary labels', () => {
    it('prefers the repository and preserves the quantization', () => {
        expect(modelPrimaryLabel('preset-ctx131072', identity)).toBe('LFM2.5 24B A2B (UD-Q4_K_XL)')
    })

    it('uses the repository alone when quantization is absent', () => {
        expect(
            modelPrimaryLabel('preset', { repository: 'owner/model-GGUF', strength: 'declared' })
        ).toBe('owner/model-GGUF')
    })

    it('formats the served name when the identity has no repository', () => {
        expect(
            modelPrimaryLabel('lmstudio/model-name-GGUF:Q4_K_M', {
                generalName: 'Different general name',
                quantization: 'Q8_0',
                quantizationSource: 'file-name',
                strength: 'declared'
            })
        ).toBe('model name (Q4_K_M)')
    })

    it('formats the served name when no identity exists', () => {
        expect(modelPrimaryLabel('owner/model-name-GGUF:Q4_K_M')).toBe('model name (Q4_K_M)')
    })
})

describe('identity metadata tags', () => {
    it('orders repository, quantization and configured context with provenance titles', () => {
        expect(modelIdentityTags(identity)).toEqual([
            {
                kind: 'repository',
                label: 'unsloth/LFM2.5-24B-A2B-GGUF',
                title: 'Repository from the Hugging Face cache'
            },
            {
                kind: 'quantization',
                label: 'UD-Q4_K_XL',
                title: 'Quantization source: hf-tag'
            },
            {
                kind: 'context',
                label: 'Context 131072',
                title: "Configured context from PAIR's reading of the configuration"
            }
        ])
    })

    it.each<ModelIdentity['quantizationSource']>(['hf-tag', 'file-name', 'gguf-file-type'])(
        'names the %s quantization provenance',
        quantizationSource => {
            expect(modelIdentityTags({ ...identity, quantizationSource })[1].title).toBe(
                `Quantization source: ${quantizationSource}`
            )
        }
    )

    it('shows Unknown metadata when no identity exists, even for a recognizable served ID', () => {
        expect(modelPrimaryLabel('owner/model-GGUF:Q4_K_M')).toBe('model (Q4_K_M)')
        expect(modelIdentityTags()).toEqual([
            {
                kind: 'repository',
                label: 'Unknown',
                title: 'Repository unknown: no source establishes it'
            },
            {
                kind: 'quantization',
                label: 'Unknown',
                title: 'Quantization unknown: no source establishes it'
            }
        ])
    })

    it('keeps an unknown quantization beside an established repository', () => {
        expect(
            modelIdentityTags({ repository: 'owner/model', strength: 'declared' }).map(tag => [
                tag.kind,
                tag.label
            ])
        ).toEqual([
            ['repository', 'owner/model'],
            ['quantization', 'Unknown']
        ])
    })

    it('adds the GGUF name only without a repository and does not treat trained context as configured', () => {
        expect(
            modelIdentityTags({
                generalName: 'Model general name',
                trainedContextTokens: 32768,
                configuredContextTokens: 8192,
                strength: 'declared'
            })
        ).toEqual([
            {
                kind: 'repository',
                label: 'Unknown',
                title: 'Repository unknown: no source establishes it'
            },
            {
                kind: 'quantization',
                label: 'Unknown',
                title: 'Quantization unknown: no source establishes it'
            },
            { kind: 'name', label: 'Model general name', title: 'Name from GGUF general.name' },
            {
                kind: 'context',
                label: 'Context 8192',
                title: "Configured context from PAIR's reading of the configuration"
            }
        ])
        expect(
            modelIdentityTags({ trainedContextTokens: 32768, strength: 'declared' }).map(
                tag => tag.kind
            )
        ).toEqual(['repository', 'quantization'])
    })

    it('renders read-only tags with the metadata labels and provenance titles', () => {
        const container = document.createElement('div')
        container.innerHTML = renderToStaticMarkup(createElement(ModelIdentityTags, { identity }))
        const group = container.querySelector('[aria-label="Model identity"]')
        if (!group) throw new Error('Missing model identity group')
        expect(group.querySelectorAll('button')).toHaveLength(0)
        expect(
            Array.from(group.querySelectorAll('[title]'), tag => [
                tag.textContent,
                tag.getAttribute('title')
            ])
        ).toEqual([
            ['unsloth/LFM2.5-24B-A2B-GGUF', 'Repository from the Hugging Face cache'],
            ['UD-Q4_K_XL', 'Quantization source: hf-tag'],
            ['Context 131072', "Configured context from PAIR's reading of the configuration"]
        ])
    })
})

function model(name: string, metadata?: ModelIdentity): ModelItem {
    return {
        name,
        identity: metadata,
        size: 0,
        downloaded: true,
        downloadStateCurrent: true,
        status: 'idle',
        parameterSize: '',
        family: '',
        sizeVram: null,
        expiresAt: null,
        expiry: '10m',
        capabilities: []
    }
}

describe('model row identity details', () => {
    it.each([
        { served: 'preset-ctx131072', expectedServedLine: 'Served as preset-ctx131072' },
        { served: 'owner/model', expectedServedLine: null }
    ])(
        'keeps the raw title and conditional Served as line for $served',
        ({ served, expectedServedLine }) => {
            const row = model(served, { repository: 'owner/model', strength: 'declared' })
            const container = document.createElement('div')
            container.innerHTML = renderToStaticMarkup(
                createElement(ModelRow, {
                    model: row,
                    isRunning: false,
                    capabilities: SAFE_DEFAULT_ENGINE_CAPS,
                    displayName: name => modelPrimaryLabel(name, row.identity),
                    onAction: () => {},
                    onExpiryChange: () => {}
                })
            )
            expect(container.firstElementChild?.getAttribute('title')).toBe(served)
            expect(container.querySelector('[aria-label="Model identity"]')?.textContent).toBe(
                'owner/modelUnknown'
            )
            if (expectedServedLine) expect(container.textContent).toContain(expectedServedLine)
            else expect(container.textContent).not.toContain('Served as')
        }
    )
})

describe('workload identity lookup', () => {
    let root: Root
    let container: HTMLDivElement
    beforeEach(() => {
        vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
        container = document.createElement('div')
        document.body.append(container)
        root = createRoot(container)
    })
    afterEach(async () => {
        await act(async () => root.unmount())
        container.remove()
        useEngineModelsStore.setState({ models: new Map() })
        vi.unstubAllGlobals()
    })

    it.each([
        {
            scenario: 'execution node and engine select the exact served ID',
            scheduledOn: 'execution',
            expected: 'weak → LFM2.5 24B A2B (UD-Q4_K_XL)'
        },
        {
            scenario: 'unscheduled work does not use the origin inventory',
            scheduledOn: null,
            expected: 'weak → preset-ctx131072'
        }
    ])('$scenario', async ({ scheduledOn, expected }) => {
        const served = 'preset-ctx131072'
        useEngineModelsStore.setState({
            models: new Map<string, EngineModels>([
                [
                    'origin:llama-cpp',
                    {
                        nodeId: 'origin',
                        engineType: 'llama-cpp',
                        models: [model(served, { repository: 'wrong/origin', strength: 'declared' })]
                    }
                ],
                [
                    `execution:${customEngine}`,
                    {
                        nodeId: 'execution',
                        engineType: customEngine,
                        models: [model(served, { repository: 'wrong/engine', strength: 'declared' })]
                    }
                ],
                [
                    'execution:llama-cpp',
                    {
                        nodeId: 'execution',
                        engineType: 'llama-cpp',
                        models: [
                            model('different-preset', {
                                repository: 'wrong/model',
                                strength: 'declared'
                            }),
                            model(served, identity)
                        ]
                    }
                ]
            ])
        })
        const workload: Workload = {
            id: '1',
            model: served,
            requestedModel: 'weak',
            engine: 'llama-cpp',
            state: 'completed',
            originatedFrom: 'origin',
            scheduledOn,
            createdAt: 1,
            startedAt: 2,
            completedAt: 3,
            error: null,
            requesterId: null
        }
        await act(async () => root.render(createElement(WorkloadItemCard, { workload })))
        expect(container.textContent).toContain(expected)
        expect(container.textContent).not.toContain('wrong/')
    })
})
