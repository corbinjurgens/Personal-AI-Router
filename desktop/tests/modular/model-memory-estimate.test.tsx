// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import ModelMemoryEstimate from '@/ui/components/ModelSettings/ModelMemoryEstimate'
import type { Estimate } from '@/shared/types/model-config'

const estimate: Estimate = {
    status: 'estimated',
    source: 'fixture',
    buildId: 'native',
    memoryTimestamp: '2026-10-28T00:00:00Z',
    domains: [
        {
            id: 'shared',
            kind: 'shared',
            capacityBytes: '10737418240',
            freeBytes: '6442450944',
            requiredBytes: '4294967296'
        }
    ],
    components: [
        { name: 'weights', domainId: 'shared', status: 'estimated', bytes: '4294967296' },
        ...[
            'attention-cache',
            'recurrent-state',
            'compute-workspace',
            'prompt-cache',
            'companions'
        ].map(
            name =>
                ({
                    name,
                    status: 'excluded',
                    reason: 'Not present in fixture'
                }) satisfies NonNullable<Estimate['components']>[number]
        )
    ]
}

describe('advisory memory estimate', () => {
    it.each([
        {
            name: 'missing bytes',
            components: estimate.components?.map(component =>
                component.name === 'weights' ? { ...component, bytes: undefined } : component
            )
        },
        {
            name: 'unmapped cost',
            components: estimate.components?.map(component =>
                component.name === 'weights' ? { ...component, domainId: undefined } : component
            )
        },
        {
            name: 'unexplained exclusion',
            components: estimate.components?.map(component =>
                component.name === 'attention-cache'
                    ? { ...component, reason: undefined }
                    : component
            )
        },
        {
            name: 'weights only',
            components: [{ name: 'weights', status: 'estimated', bytes: '1', domainId: 'shared' }]
        }
    ] satisfies Array<{ name: string; components: Estimate['components'] }>)(
        'keeps $name Unknown',
        ({ components }) => {
            const html = renderToStaticMarkup(
                <ModelMemoryEstimate estimate={{ ...estimate, components }} />
            )
            expect(html).not.toContain('(Fits)')
            expect(html).toContain('Unknown')
        }
    )
    it.each([
        'attention-cache',
        'recurrent-state',
        'compute-workspace',
        'prompt-cache',
        'companions'
    ])('requires an explicit cost or exclusion for %s', name => {
        const html = renderToStaticMarkup(
            <ModelMemoryEstimate
                estimate={{
                    ...estimate,
                    components: estimate.components?.filter(component => component.name !== name)
                }}
            />
        )
        expect(html).not.toContain('(Fits)')
        expect(html).toContain(`explicit exclusions: ${name}`)
    })
    it('does not add host and device capacities to an overlapping shared UMA pool', () => {
        const html = renderToStaticMarkup(
            <ModelMemoryEstimate
                estimate={{
                    ...estimate,
                    domains: [
                        ...(estimate.domains ?? []),
                        { id: 'host', kind: 'host', capacityBytes: '10737418240' }
                    ]
                }}
            />
        )
        expect(html.match(/Shared UMA pool/g)).toHaveLength(1)
        expect(html).not.toContain('Host (host)')
        expect(html).not.toContain('(Fits)')
    })
    it('keeps exact comparisons when rounded byte labels match', () => {
        const requiredBytes = '9007199254740993'
        const html = renderToStaticMarkup(
            <ModelMemoryEstimate
                estimate={{
                    ...estimate,
                    components: estimate.components?.map(component =>
                        component.name === 'weights'
                            ? { ...component, bytes: requiredBytes }
                            : component
                    ),
                    domains: [
                        {
                            id: 'shared',
                            kind: 'shared',
                            requiredBytes,
                            capacityBytes: '18014398509481984',
                            freeBytes: '9007199254740992'
                        }
                    ]
                }}
            />
        )
        expect(html).toContain('free 8 PB (Exceeds)')
    })
    it('shows Tight when the estimate consumes the free-memory reserve', () => {
        const html = renderToStaticMarkup(
            <ModelMemoryEstimate
                estimate={{
                    ...estimate,
                    domains: [
                        {
                            id: 'shared',
                            kind: 'shared',
                            capacityBytes: '10737418240',
                            requiredBytes: '4294967296',
                            freeBytes: '5368709120'
                        }
                    ]
                }}
            />
        )
        expect(html).toContain('capacity 10 GB (Fits)')
        expect(html).toContain('free 5 GB (Tight)')
    })
    it('requires mapped estimated costs even when the exclusion list is complete', () => {
        const html = renderToStaticMarkup(
            <ModelMemoryEstimate
                estimate={{
                    ...estimate,
                    components: estimate.components?.map(component =>
                        component.name === 'weights' ? { ...component, domainId: 'gpu' } : component
                    ),
                    domains: [
                        {
                            id: 'gpu',
                            kind: 'device',
                            capacityBytes: '10737418240',
                            requiredBytes: '4294967296',
                            freeBytes: '6442450944'
                        }
                    ]
                }}
            />
        )
        expect(html).toContain('Device mapping unknown')
        expect(html).not.toContain('(Fits)')
    })
    it('keeps inconsistent domain totals Unknown', () => {
        const html = renderToStaticMarkup(
            <ModelMemoryEstimate
                estimate={{
                    ...estimate,
                    domains: [
                        {
                            id: 'shared',
                            kind: 'shared',
                            capacityBytes: '10737418240',
                            requiredBytes: '1',
                            freeBytes: '6442450944'
                        }
                    ]
                }}
            />
        )
        expect(html).not.toContain('(Fits)')
    })
    it('shows one UMA pool, sampled free memory and reserve', () => {
        const html = renderToStaticMarkup(<ModelMemoryEstimate estimate={estimate} />)
        expect(html).toContain('Shared UMA pool')
        expect(html).toContain('free 6 GB (Fits)')
        expect(html).toContain('reserve 1.5 GB')
        expect(html).toContain('weights')
        expect(html).not.toContain('Observed')
    })
    it('does not claim Fits for partial, stale or unmapped results', () => {
        const partial = renderToStaticMarkup(
            <ModelMemoryEstimate
                estimate={{ ...estimate, status: 'partial', unknowns: ['workspace'] }}
            />
        )
        const stale = renderToStaticMarkup(<ModelMemoryEstimate estimate={estimate} stale />)
        const unmapped = renderToStaticMarkup(
            <ModelMemoryEstimate
                estimate={{
                    ...estimate,
                    domains: [{ ...estimate.domains?.[0], id: 'gpu', kind: 'device' }]
                }}
            />
        )
        for (const html of [partial, stale, unmapped]) expect(html).not.toContain('(Fits)')
        expect(partial).toContain('Unknown: workspace')
        expect(stale).toContain('(stale)')
        expect(unmapped).toContain('mapping unknown')
    })
    it('distinguishes tight and exceeds from physical capacity', () => {
        const html = renderToStaticMarkup(
            <ModelMemoryEstimate
                estimate={{
                    ...estimate,
                    components: estimate.components?.map(component =>
                        component.status === 'estimated'
                            ? { ...component, domainId: 'host', bytes: '5500000000' }
                            : component
                    ),
                    domains: [
                        {
                            id: 'host',
                            kind: 'host',
                            capacityBytes: '10737418240',
                            freeBytes: '5000000000',
                            requiredBytes: '5500000000'
                        }
                    ]
                }}
            />
        )
        expect(html).toContain('capacity 10 GB (Fits)')
        expect(html).toContain('free 4.66 GB (Exceeds)')
    })
})
