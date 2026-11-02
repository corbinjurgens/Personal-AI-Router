// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EngineDescriptor } from '@/shared/types/engines'
import { engineTypeFromManagerName } from '@/shared/utils/engines'

function descriptorEngine() {
    const engine = engineTypeFromManagerName('llamacpp-custom')
    if (!engine) throw new Error('Invalid descriptor engine fixture')
    return engine
}

export const customEngine = descriptorEngine()

export const customEngineDescriptor: EngineDescriptor = {
    engineType: customEngine,
    family: 'llamacpp',
    label: 'llama.cpp custom',
    available: true,
    protocols: ['openai-v1', 'llamacpp-router'],
    modelNaming: 'exact',
    modelStore: 'llamacpp',
    catalog: 'llamacpp',
    operations: { load: { responseTimeoutS: 0 }, unload: { responseTimeoutS: 0 } }
}
