// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import ModelMemoryEstimate from '@/ui/components/ModelSettings/ModelMemoryEstimate'
import type { Estimate } from '@/shared/types/model-config'

const estimate: Estimate = {
    status: 'partial',
    estimateSource: {
        kind: 'pair-estimate',
        provider: 'PAIR',
        revision: 'pair-gguf/2',
        basis: 'GGUF metadata',
        unpricedCosts: ['driver allocations']
    },
    capacityFit: 'fits',
    freeFit: 'tight',
    memorySource: 'target node',
    memoryTimestamp: '2026-11-07T01:00:00Z',
    domains: [
        {
            id: 'shared',
            kind: 'shared',
            capacityBytes: '10737418240',
            freeBytes: '6442450944',
            requiredBytes: '4294967296',
            requiredLowBytes: '3221225472',
            requiredHighBytes: '5368709120',
            reserveBytes: '268435456',
            capacityFit: 'fits',
            freeFit: 'tight',
            source: 'OS available physical memory',
            sampleId: 'target-sample-17',
            capacitySource: 'physical capacity',
            capacityTimestamp: '2026-11-07T00:59:59Z',
            freeSource: 'available pages',
            freeTimestamp: '2026-11-07T01:00:00Z',
            remainingValidityMs: 1400,
            unknowns: ['not priced: driver allocations']
        }
    ],
    components: [
        {
            name: 'compute-workspace',
            domainId: 'shared',
            status: 'partial',
            bytes: '1073741824',
            lowBytes: '536870912',
            highBytes: '2147483648',
            basis: 'heuristic',
            reason: 'additional workspace not priced'
        }
    ]
}

function display(value?: Estimate, stale = false): string {
    const container = document.createElement('div')
    container.innerHTML = renderToStaticMarkup(
        <ModelMemoryEstimate estimate={value} stale={stale} />
    )
    return (container.textContent ?? '').replace(/\s+/g, ' ')
}

describe('advisory memory estimate', () => {
    it('renders service comparisons independently of component completeness and renderer arithmetic', () => {
        const text = display(estimate)
        expect(text).toContain('capacity 10 GB (Fits)')
        expect(text).toContain('available 6 GB (Tight: may not fit)')
        expect(text).toContain('service reserve 256 MB')
        expect(text).not.toContain('missing component')
    })

    it('keeps service comparisons when provenance, canonical rows and reserve are absent', () => {
        const text = display({
            status: 'partial',
            capacityFit: 'fits',
            freeFit: 'tight',
            domains: [{ id: 'host', kind: 'host', capacityFit: 'fits', freeFit: 'tight' }]
        })
        expect(text).toContain('capacity Unknown (Fits)')
        expect(text).toContain('available Unknown (Tight: may not fit)')
        expect(text).toContain('service reserve Unknown')
        expect(text).toContain('Capacity source: Unknown')
        expect(text).toContain('Remaining validity: Unknown')
    })

    it.each([
        { fit: 'fits', label: 'Fits' },
        { fit: 'tight', label: 'Tight: may not fit' },
        { fit: 'exceeds', label: 'Exceeds available memory' },
        { fit: 'unknown', label: 'Unknown' }
    ] satisfies Array<{ fit: Estimate['capacityFit']; label: string }>)(
        'renders the service $fit verdict without recalculating it',
        ({ fit, label }) => {
            const text = display({
                status: 'estimated',
                domains: [{ id: 'host', kind: 'host', capacityFit: fit, freeFit: fit }]
            })
            expect(text).toContain(`capacity Unknown (${label})`)
            expect(text).toContain(`available Unknown (${label})`)
        }
    )

    it('withdraws both aggregate and domain verdicts while preserving stale amounts and notes', () => {
        const text = display(estimate, true)
        expect(text).toContain('Service comparison: capacity (Unknown) · available (Unknown)')
        expect(text).toContain('capacity 10 GB (Unknown)')
        expect(text).toContain('available 6 GB (Unknown)')
        expect(text).toContain('Remaining validity: Expired')
        expect(text).toContain('retained amounts are historical')
        expect(text).toContain('expected 4 GB (up to 5 GB; range 3 GB to 5 GB)')
        expect(text).toContain('driver allocations')
        expect(text).not.toContain('(Fits)')
        expect(text).not.toContain('Tight: may not fit')
    })

    it('displays expected ranges, component basis and unpriced warnings', () => {
        const text = display(estimate)
        expect(text).toContain('Estimate expected 4 GB (up to 5 GB; range 3 GB to 5 GB)')
        expect(text).toContain('expected 1 GB (up to 2 GB; range 512 MB to 2 GB)')
        expect(text).toContain('Basis: heuristic (uncalibrated)')
        expect(text).toContain('Not included in this estimate')
        expect(text).toContain('Not priced: driver allocations')
        expect(text).toContain(
            'Not included in this estimate (shared): not priced: driver allocations'
        )
    })

    it.each([
        { kind: 'pair-estimate', label: 'Estimate' },
        { kind: 'user-stated', label: 'Your figure' },
        { kind: 'measured', label: 'Measured' },
        { kind: 'engine-report', label: 'Engine report' }
    ] satisfies Array<{ kind: NonNullable<Estimate['estimateSource']>['kind']; label: string }>)(
        'labels the $kind source',
        ({ kind, label }) => {
            const text = display({
                ...estimate,
                estimateSource: {
                    kind,
                    provider: 'source provider',
                    revision: 'r7',
                    basis: 'source basis'
                }
            })
            expect(text).toContain(`${label} expected 4 GB`)
            expect(text).toContain('Provider: source provider · Revision: r7 · Basis: source basis')
        }
    )

    it('retains independent capacity and free-memory provenance and service validity', () => {
        const text = display(estimate)
        expect(text).toContain(
            'Accounting source: OS available physical memory · Sample: target-sample-17'
        )
        expect(text).toContain(
            'Capacity source: physical capacity · Capacity sampled: 2026-11-07T00:59:59Z'
        )
        expect(text).toContain(
            'Available source: available pages · Available sampled: 2026-11-07T01:00:00Z'
        )
        expect(text).toContain('Remaining validity: 1400 ms')
    })

    it('shows UMA once without adding a duplicate host capacity', () => {
        const text = display({
            ...estimate,
            domains: [
                ...(estimate.domains ?? []),
                { id: 'host', kind: 'host', capacityBytes: '10737418240' }
            ]
        })
        expect(text.match(/Shared UMA pool/g)).toHaveLength(1)
        expect(text).not.toContain('Host (host)')
        expect(text).toContain('Other pool capacities are not added')
    })

    it('uses physical identity for separate devices and leaves unmatched comparisons Unknown', () => {
        const text = display({
            ...estimate,
            capacityFit: 'unknown',
            freeFit: 'unknown',
            domains: [
                { id: 'gpu-b', kind: 'device', deviceId: 'GPU-UUID-B', freeFit: 'tight' },
                { id: 'gpu-a', kind: 'device', deviceId: 'GPU-UUID-A', freeFit: 'exceeds' },
                {
                    id: 'unmatched',
                    kind: 'device',
                    freeFit: 'unknown',
                    unknowns: ['native map unavailable']
                }
            ]
        })
        expect(text).toContain('Device GPU-UUID-B (gpu-b)')
        expect(text).toContain('Device GPU-UUID-A (gpu-a)')
        expect(text).toContain('available Unknown (Exceeds available memory)')
        expect(text).toContain('Device mapping unknown (unmatched)')
        expect(text).toContain('Unknown (unmatched): native map unavailable')
        expect(text).not.toContain('(Fits)')
    })

    it('keeps missing service comparisons Unknown even when numeric amounts look sufficient', () => {
        const text = display({
            status: 'estimated',
            domains: [
                {
                    id: 'host',
                    kind: 'host',
                    capacityBytes: '10737418240',
                    freeBytes: '6442450944',
                    requiredBytes: '1'
                }
            ]
        })
        expect(text).toContain('capacity 10 GB (Unknown)')
        expect(text).toContain('available 6 GB (Unknown)')
        expect(text).not.toContain('(Fits)')
    })

    it('renders missing estimates without invented memory or provenance', () => {
        const text = display()
        expect(text).toContain('Memory domains: Unknown')
        expect(text).toContain('Memory source: Unknown')
        expect(text).toContain('Provider: Unknown')
        expect(text).not.toContain('(Fits)')
    })
})
