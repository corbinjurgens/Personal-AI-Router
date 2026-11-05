// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { formatModelDisplayName } from '@/ui/utils/format-model-display-name'

describe('model display names', () => {
    it.each([
        {
            scenario: 'Hugging Face repository with quantization',
            name: 'unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL',
            expected: 'LFM2.5 24B A2B (UD-Q4_K_XL)'
        },
        {
            scenario: 'Hugging Face URL with quantization',
            name: 'https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL',
            expected: 'LFM2.5 24B A2B (UD-Q4_K_XL)'
        },
        {
            scenario: 'repository without a tag retains its existing label',
            name: 'unsloth/LFM2.5-24B-A2B-GGUF',
            expected: 'LFM2.5 24B A2B'
        },
        {
            scenario: 'bare served name stays unchanged',
            name: 'my-model-ctx131072',
            expected: 'my-model-ctx131072'
        },
        {
            scenario: 'model-source ID follows the same generic rules',
            name: 'lmstudio/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL',
            expected: 'LFM2.5 24B A2B (UD-Q4_K_XL)'
        },
        {
            scenario: 'tag on a repository without a GGUF suffix',
            name: 'owner/model-name:Q4_K_M',
            expected: 'model name (Q4_K_M)'
        }
    ])('$scenario', ({ name, expected }) => {
        expect(formatModelDisplayName(name)).toBe(expected)
    })

    it('bounds input before parsing a tag', () => {
        expect(formatModelDisplayName(`owner/${'x'.repeat(300)}:Q4_K_M`)).toBe('x'.repeat(250))
    })
})
