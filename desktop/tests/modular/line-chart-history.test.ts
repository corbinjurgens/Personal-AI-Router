// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, it } from 'vitest'
import { buildLineChartPaths, type MetricDataset } from '@/ui/components/SvgLineChart'

function dataset(timestamps: number[], key = 'cpu'): MetricDataset {
    return {
        key,
        color: '#fff',
        label: key,
        data: timestamps.map(timestamp => ({ timestamp, value: 50 }))
    }
}

it('spaces observations by timestamp rather than array position', () => {
    const [path] = buildLineChartPaths([dataset([0, 1000, 4000])], 30, 100, 400, 100)
    expect(path.line.startsWith('M 4 54')).toBe(true)
    expect(path.line).toContain(', 104 54 C ')
    expect(path.line.endsWith(', 404 54')).toBe(true)
    expect(path.line.match(/M /g)).toHaveLength(1)
})

it('breaks both stroke and fill across a real observation gap', () => {
    const [path] = buildLineChartPaths([dataset([0, 1000, 11000, 12000])], 30, 100, 120, 100)
    expect(path.line.match(/M /g)).toHaveLength(2)
    expect(path.line).toContain('M 114 54')
    expect(path.line.match(/C /g)).toHaveLength(2)
    expect(path.fill.match(/Z/g)).toHaveLength(2)
    expect(path.fill).toContain('L 14 104 L 4 104 Z M 114 54')
})

it('keeps regular one-second and two-second samples connected', () => {
    for (const timestamps of [
        [0, 1000, 2000, 3000],
        [0, 2000, 4000, 6000]
    ]) {
        const [path] = buildLineChartPaths([dataset(timestamps)], 30, 100, 120, 100)
        expect(path.line.match(/M /g)).toHaveLength(1)
        expect(path.line.match(/C /g)).toHaveLength(3)
        expect(path.fill.match(/Z/g)).toHaveLength(1)
    }
})

it('breaks sparse histories beyond five seconds without needing an estimated cadence', () => {
    const [boundary] = buildLineChartPaths([dataset([0, 5000])], 30, 100, 120, 100)
    const [gap] = buildLineChartPaths([dataset([0, 5001])], 30, 100, 120, 100)
    expect(boundary.line.match(/M /g)).toHaveLength(1)
    expect(gap.line.match(/M /g)).toHaveLength(2)
    expect(gap.line).not.toContain('C ')
})

it('aligns series on a shared timestamp domain and honors the point limit', () => {
    const paths = buildLineChartPaths(
        [dataset([0, 1000, 2000]), dataset([2000, 4000], 'gpu')],
        2,
        100,
        300,
        100
    )
    expect(paths[0].line.startsWith('M 4 54')).toBe(true)
    expect(paths[0].line.endsWith(', 104 54')).toBe(true)
    expect(paths[1].line.startsWith('M 104 54')).toBe(true)
    expect(paths[1].line.endsWith(', 304 54')).toBe(true)
})

it('does not manufacture zero observations for empty histories', () => {
    expect(buildLineChartPaths([dataset([])], 30, 100, 120, 100)).toEqual([
        { key: 'cpu', color: '#fff', line: '', fill: '' }
    ])
    const [single] = buildLineChartPaths([dataset([1000])], 30, 100, 120, 100)
    expect(single.line).toBe('M 4 54')
    expect(single.line).not.toContain('NaN')
})
