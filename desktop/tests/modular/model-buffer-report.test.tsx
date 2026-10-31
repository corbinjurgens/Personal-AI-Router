// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import type {
    EngineSettingsSnapshot,
    EngineSettingsPreview,
    EngineSettingsTarget
} from '@/shared/types/engine-settings'
import type { InspectModelResult } from '@/shared/types/model-config'
import ModelLoadBufferReport from '@/ui/components/ModelSettings/ModelLoadBufferReport'
import ModelSettingsPreview from '@/ui/components/ModelSettings/ModelSettingsPreview'
import {
    engineSettingsKey,
    modelSettingsKey,
    useEngineSettingsStore
} from '@/ui/stores/engine-settings.store'

type LoadBufferObservation = NonNullable<InspectModelResult['loadObservation']>

function observation(): LoadBufferObservation {
    return {
        status: 'partial',
        current: true,
        reason: 'unmapped-device',
        persistence: 'uncertain',
        sampleId: 'a'.repeat(64),
        recordedAt: '2026-10-28T00:00:00Z',
        loadPhase: 'wakeup',
        upstreamPin: '0ec4027a2a0183de87c007069adddc06e665f7de',
        buildAssurance: 'recognized-self-reported-pin',
        estimatorRevision:
            'pair-gguf/2;gguf-parser-go/v0.27.0@9257743a3c1775b882ba1b58ec7f5a856c2cc385',
        capturedOptions: {
            context: { mode: 'fixed', tokens: 8192 },
            slots: 2,
            gpu: { mode: 'cpu' }
        },
        requestedOptions: { context: { mode: 'auto', floor: 4096, cap: 8192 } },
        optionUnknowns: ['mmap', 'load-mode'],
        baseline: [{ component: 'weights', status: 'estimated', bytes: '1048576' }],
        comparisons: [
            {
                component: 'weights',
                status: 'compatible',
                estimatedBytes: '1048576',
                lowerBytes: '1043333',
                upperBytes: '1053819',
                deltaLowerBytes: '-5243',
                deltaUpperBytes: '5243'
            }
        ],
        rows: [
            {
                component: 'model',
                label: 'CPU',
                phase: 'load',
                sequence: '1',
                stream: 'stderr',
                role: 'model',
                reportedMiB: '1.00',
                unit: 'MiB',
                precision: '0.01',
                sourceKind: 'log-scraped',
                intervalStatus: 'bounded',
                lowerBytes: '1043333',
                upperBytes: '1053819',
                coverage: 'unknown'
            }
        ]
    }
}
function inspection(loadObservation = observation()): InspectModelResult {
    return {
        engine: 'llamacpp',
        nodeId: 'self',
        modelId: 'exact-model',
        status: 'unknown',
        estimate: { status: 'unknown' },
        loadObservation
    }
}
function snapshot(nodeId = 'self'): EngineSettingsSnapshot {
    return {
        nodeId,
        engine: 'llamacpp',
        revision: 1,
        appliedRevision: 1,
        sequence: 1,
        epoch: 'test',
        settings: { serverPort: 51001, proxyPort: 51002, launchText: '' },
        effectiveServerPort: 51001,
        effectiveProxyPort: 51002,
        running: true,
        adopted: false,
        editable: true,
        reason: '',
        format: 'pair-arguments-v1',
        phase: 'idle',
        error: '',
        requestId: ''
    }
}

describe('native buffer load report with real KUI components', () => {
    let root: Root
    let container: HTMLDivElement
    beforeEach(() => {
        container = document.createElement('div')
        document.body.append(container)
        root = createRoot(container)
    })
    afterEach(async () => {
        await act(async () => root.unmount())
        container.remove()
    })
    it('renders a compact semantic table of rounded allocations and frozen component comparisons', async () => {
        await act(async () => root.render(<ModelLoadBufferReport inspection={inspection()} />))
        const table = container.querySelector('table')
        expect(table).not.toBeNull()
        expect(Array.from(container.querySelectorAll('th')).map(cell => cell.textContent)).toEqual([
            'Component / buffer',
            'Device / pool',
            'Estimate for this load',
            'Native-reported size',
            'Status / coverage'
        ])
        expect(table?.textContent).toContain('Unmapped')
        expect(table?.textContent).toContain('1.00 MiB (precision 0.01; log-scraped)')
        expect(table?.textContent).toContain('rounded interval [1043333, 1053819] bytes')
        expect(table?.textContent).toContain('reported minus estimated [-5243, 5243] bytes')
        expect(container.textContent).toContain('Durability uncertain (replacement landed)')
        expect(container.textContent).toContain('clean-source provenance is unverified')
        expect(container.textContent).toContain('wakeup')
        expect(container.textContent).toContain(
            'Rounded allocation reports, not physical GPU usage'
        )
        expect(container.querySelector('button')).toBeNull()
        expect(container.textContent).not.toContain('Fits')
    })
    it('keeps the captured baseline separate from a changed draft or Auto estimate', async () => {
        const preview: EngineSettingsPreview = {
            settings: snapshot().settings,
            revision: 1,
            errors: {},
            restart: true,
            rebind: false,
            estimate: {
                status: 'partial',
                hostBytes: '999999999',
                components: [{ name: 'weights', status: 'partial', bytes: '999999999' }]
            }
        }
        await act(async () =>
            root.render(
                <ModelSettingsPreview
                    inspection={inspection()}
                    snapshot={snapshot()}
                    preview={preview}
                    stale
                />
            )
        )
        expect(container.querySelector('table')?.textContent).toContain('1048576 estimated bytes')
        expect(container.querySelector('table')?.textContent).not.toContain('999999999')
        expect(container.textContent).toContain('Current load at inspection')
        expect(container.textContent).toContain('independent of the edited draft and Auto proposal')
        expect(container.textContent).toContain('"tokens":8192')
        expect(container.textContent).toContain('Requested options:')
    })
    it('withdraws a current claim immediately when inspection becomes stale', async () => {
        const value = inspection()
        await act(async () =>
            root.render(<ModelSettingsPreview inspection={value} snapshot={snapshot()} />)
        )
        expect(container.textContent).toContain('Current load at inspection')
        await act(async () => root.render(<ModelLoadBufferReport inspection={value} stale />))
        expect(container.textContent).not.toContain('Current load at inspection')
        expect(container.textContent).toContain('inspection stale; current state Unknown')
        expect(container.querySelector('table')).not.toBeNull()
    })
    it('hides the prior summary synchronously on a changed target or model while awaiting inspection', async () => {
        await act(async () =>
            root.render(<ModelSettingsPreview inspection={inspection()} snapshot={snapshot()} />)
        )
        expect(container.textContent).toContain('Current load at inspection')
        await act(async () =>
            root.render(
                <ModelSettingsPreview
                    inspection={inspection()}
                    snapshot={snapshot('another-node')}
                />
            )
        )
        expect(container.querySelector('table')).toBeNull()
        expect(container.textContent).not.toContain('Current load at inspection')
        await act(async () =>
            root.render(
                <ModelSettingsPreview
                    snapshot={snapshot()}
                    inspection={{
                        ...inspection(),
                        modelId: 'new-model',
                        loadObservation: undefined
                    }}
                />
            )
        )
        expect(container.textContent).not.toContain('exact-model')
        expect(container.querySelector('table')).toBeNull()
        expect(container.textContent).toContain('Unknown (observation omitted)')
    })
    it('keeps recovered reports historical and pending or failed saves explicit', async () => {
        for (const persistence of [
            'pending',
            'failed',
            'history-unavailable'
        ] satisfies LoadBufferObservation['persistence'][]) {
            await act(async () =>
                root.render(
                    <ModelLoadBufferReport
                        inspection={inspection({ ...observation(), current: false, persistence })}
                    />
                )
            )
            expect(container.textContent).toContain('Historical load comparison')
            expect(container.textContent).not.toContain('Current load at inspection')
            expect(container.textContent).toContain(`Save: ${persistence}`)
        }
    })
    it('shows Unknown for an unsupported build even if plausible rows accompany it', async () => {
        await act(async () =>
            root.render(
                <ModelLoadBufferReport
                    inspection={inspection({
                        ...observation(),
                        status: 'unknown',
                        current: false,
                        reason: 'unsupported-build'
                    })}
                />
            )
        )
        expect(container.textContent).toContain('Unknown (unsupported build)')
        expect(container.querySelector('table')).toBeNull()
        expect(container.textContent).not.toContain('1.00 MiB')
    })
    it('keeps unmapped unknown intervals numeric without inventing exact bytes or totals', async () => {
        const value = observation()
        value.comparisons = [{ component: 'weights', status: 'unknown' }]
        value.rows = value.rows?.map(row => ({
            ...row,
            reportedMiB: '1073741824.00',
            intervalStatus: 'unknown',
            lowerBytes: undefined,
            upperBytes: undefined,
            role: 'mixed'
        }))
        await act(async () => root.render(<ModelLoadBufferReport inspection={inspection(value)} />))
        expect(container.querySelector('table')?.textContent).toContain('1073741824.00 MiB')
        expect(container.querySelector('table')?.textContent).toContain('byte interval Unknown')
        expect(container.querySelector('table')?.textContent).toContain('Unknown comparison')
        expect(container.querySelector('table')?.textContent).not.toContain(
            'reported minus estimated'
        )
    })
})

describe('observation freshness at the existing settings owner', () => {
    const target = { nodeId: 'self', engine: 'llamacpp' } satisfies EngineSettingsTarget
    const key = modelSettingsKey(target, 'exact-model')
    beforeEach(() => useEngineSettingsStore.setState({ entries: {}, modelDrafts: {} }))
    afterEach(() => useEngineSettingsStore.setState({ entries: {}, modelDrafts: {} }))
    function seed() {
        const original = {
            ...inspection(),
            facts: { status: 'known', dimensions: { layers: '32' } }
        }
        useEngineSettingsStore.setState({
            entries: { [engineSettingsKey(target)]: { snapshot: snapshot(), arrival: 1 } },
            modelDrafts: { [key]: { generation: 1, inspection: original } }
        })
        return original
    }
    it.each(['invalidateModel', 'disconnect', 'baseline change'])(
        'withdraws currentness on %s while retaining detached history and facts',
        action => {
            const original = seed()
            const store = useEngineSettingsStore.getState()
            if (action === 'invalidateModel') store.invalidateModel(key)
            else if (action === 'disconnect') store.disconnect('self')
            else store.receive({ ...snapshot(), sequence: 2, revision: 2 })
            const current = useEngineSettingsStore.getState().modelDrafts[key]?.inspection
            expect(current?.loadObservation).toEqual({
                ...original.loadObservation,
                current: false
            })
            expect(current?.facts).toEqual(original.facts)
            expect(current?.loadObservation?.rows).toEqual(original.loadObservation?.rows)
            expect(current?.loadObservation?.rows).not.toBe(original.loadObservation?.rows)
            expect(original.loadObservation?.current).toBe(true)
        }
    )
    it('keeps the valid captured load current during local edit and discard', () => {
        const original = seed()
        const store = useEngineSettingsStore.getState()
        store.editModel(key, {
            modelId: 'exact-model',
            expectedPolicyRevision: '7',
            load: { set: { slots: 4 } }
        })
        expect(
            useEngineSettingsStore.getState().modelDrafts[key]?.inspection?.loadObservation
        ).toEqual(original.loadObservation)
        store.discardModel(key)
        expect(
            useEngineSettingsStore.getState().modelDrafts[key]?.inspection?.loadObservation?.current
        ).toBe(true)
        expect(
            useEngineSettingsStore.getState().modelDrafts[key]?.inspection?.loadObservation
                ?.baseline
        ).toEqual(original.loadObservation?.baseline)
    })
    it.each([true, false])(
        'withdraws currentness on a newer Off snapshot regardless of previous running=%s',
        running => {
            const original = seed()
            const before = { ...snapshot(), running }
            useEngineSettingsStore.setState({
                entries: { [engineSettingsKey(target)]: { snapshot: before, arrival: 1 } }
            })
            const previous = useEngineSettingsStore.getState().modelDrafts[key]
            useEngineSettingsStore.getState().receive({ ...before, sequence: 2, running: false })
            const current = useEngineSettingsStore.getState().modelDrafts[key]
            expect(current).toEqual({
                ...previous,
                inspection: {
                    ...original,
                    loadObservation: { ...original.loadObservation, current: false }
                }
            })
            expect(current?.inspection?.loadObservation?.rows).toEqual(
                original.loadObservation?.rows
            )
            expect(current?.inspection?.loadObservation?.rows).not.toBe(
                original.loadObservation?.rows
            )
            expect(original.loadObservation?.current).toBe(true)
        }
    )
})
