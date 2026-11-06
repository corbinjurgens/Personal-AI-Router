// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { EngineSettingsPreview, EngineSettingsSnapshot } from '@/shared/types/engine-settings'
import type { InspectModelResult, ModelLoad } from '@/shared/types/model-config'
import ModelSettingsPreview from '@/ui/components/ModelSettings/ModelSettingsPreview'

const inspection: InspectModelResult = {
    nodeId: 'self',
    engine: 'llamacpp',
    modelId: 'shared-artifact',
    status: 'resolved',
    estimate: { status: 'unknown' }
}
const small: ModelLoad = {
    profileKey: 'small',
    modelId: 'shared-artifact',
    requested: { slots: 1 }
}
const large: ModelLoad = {
    profileKey: 'large',
    modelId: 'shared-artifact',
    requested: { slots: 4 }
}
const snapshot: EngineSettingsSnapshot = {
    nodeId: 'self',
    engine: 'llamacpp',
    revision: 3,
    appliedRevision: 2,
    sequence: 1,
    epoch: 'test',
    settings: { serverPort: 1, proxyPort: 2, launchText: '', modelLoads: [small, large] },
    effectiveSettings: {
        serverPort: 1,
        proxyPort: 2,
        launchText: '',
        modelLoads: [small, { ...large, requested: { slots: 2 } }]
    },
    effectiveServerPort: 1,
    effectiveProxyPort: 2,
    running: false,
    adopted: false,
    editable: true,
    reason: '',
    format: '',
    phase: 'idle',
    error: '',
    requestId: '',
    serving: {
        status: 'observed',
        observed: [
            { modelId: 'shared-artifact', perSlotContextTokens: 1024, slots: 1 },
            { modelId: 'large', perSlotContextTokens: 32768, slots: 3 }
        ]
    }
}
function converted(profileKey: string): ModelLoad {
    return {
        profileKey,
        modelId: `${profileKey}-artifact`,
        requested: {},
        resolved: {
            perSlotContext: 8192,
            totalContext: 8192,
            slots: 1,
            gpu: { mode: 'cpu' },
            options: {},
            artifacts: [],
            nativeBuildId: 'build',
            sourceInputDigest: 'source',
            presetDigest: 'preset',
            resolverRevision: 'external-conversion/llama',
            reviewedInputId: 'source'
        }
    }
}
const conversionPreview: EngineSettingsPreview = {
    revision: 3,
    settings: { ...snapshot.settings, modelLoads: [converted('one'), converted('two'), large] },
    errors: {},
    restart: true,
    rebind: false
}

describe('keyed model review', () => {
    it('reads desired and applied siblings by key and observed allocations by served key', () => {
        const html = renderToStaticMarkup(
            <ModelSettingsPreview profileKey="large" inspection={inspection} snapshot={snapshot} />
        )
        expect(html).toContain('Desired load preferences: {&quot;slots&quot;:4}')
        expect(html).toContain('Applied load preferences: {&quot;slots&quot;:2}')
        expect(html).toContain('Observed model allocation: 32768 tokens per request, 3 slots')
        expect(html).not.toContain('1024 tokens per request')
    })

    it('never borrows a sibling allocation for a new profile', () => {
        const html = renderToStaticMarkup(
            <ModelSettingsPreview inspection={inspection} snapshot={snapshot} />
        )
        expect(html).toContain('Desired load preferences: {}')
        expect(html).toContain('Applied load preferences: {}')
        expect(html).toContain('Observed model allocation: Unknown')
    })

    it('counts actual converted loads without counting the newly added profile', () => {
        const html = renderToStaticMarkup(
            <ModelSettingsPreview
                preview={conversionPreview}
                snapshot={{ ...snapshot, settings: { ...snapshot.settings, modelLoads: [] } }}
            />
        )
        expect(html).toContain(
            'Saving converts this engine&#x27;s hand-written preset into 2 profiles.'
        )
    })

    it('does not call an ordinary first typed load an external preset conversion', () => {
        const html = renderToStaticMarkup(
            <ModelSettingsPreview
                preview={{
                    ...conversionPreview,
                    settings: { ...snapshot.settings, modelLoads: [large] }
                }}
                snapshot={{ ...snapshot, settings: { ...snapshot.settings, modelLoads: [] } }}
            />
        )
        expect(html).not.toContain('hand-written preset')
    })

    it('does not repeat a conversion banner after typed loads are already saved', () => {
        const html = renderToStaticMarkup(
            <ModelSettingsPreview preview={conversionPreview} snapshot={snapshot} />
        )
        expect(html).not.toContain('hand-written preset')
    })
})
