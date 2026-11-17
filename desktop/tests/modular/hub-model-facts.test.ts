// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { hubModelFacts } from '@/ui/utils/hub-model-facts'

describe('hub model facts', () => {
    it('splits the repository, title and offered quantization from a catalog name', () => {
        const facts = hubModelFacts('unsloth/Qwen3.8-27B-GGUF:Q4_K_M')
        expect(facts).toMatchObject({
            repo: 'unsloth/Qwen3.8-27B-GGUF',
            title: 'Qwen3.8-27B',
            quantization: 'Q4_K_M',
            parameters: '27B'
        })
        expect(facts.activeParameters).toBeUndefined()
    })

    it('reads total and active parameters of a mixture-of-experts name', () => {
        const facts = hubModelFacts('ggml-org/Qwen3.6-35B-A3B-GGUF:Q4_K_M')
        expect(facts.parameters).toBe('35B')
        expect(facts.activeParameters).toBe('3B')
        expect(facts.capabilities).toContain('Mixture of experts')
    })

    it('reads licence, base model, languages and capabilities from tags', () => {
        const facts = hubModelFacts('owner/Model-GGUF:Q8_0', [
            'gguf',
            'license:apache-2.0',
            'base_model:Base/Model',
            'base_model:quantized:Base/Model',
            'image-text-to-text',
            'en',
            'zh'
        ])
        expect(facts.license).toBe('Apache 2.0')
        expect(facts.baseModel).toBe('Base/Model')
        expect(facts.languages).toEqual(['en', 'zh'])
        expect(facts.capabilities).toEqual(['Vision'])
    })

    it('keeps an unknown licence identifier as written and leaves absent facts empty', () => {
        expect(hubModelFacts('owner/Model', ['license:custom-1']).license).toBe('custom-1')
        const bare = hubModelFacts('owner/Model')
        expect(bare.quantization).toBeUndefined()
        expect(bare.license).toBeUndefined()
        expect(bare.baseModel).toBeUndefined()
        expect(bare.parameters).toBeUndefined()
    })
})
