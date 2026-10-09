// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { parseWorkloadsInitial } from '@/electron/service-bridge/modular-state'
import {
    nodeTargetParams,
    parseAvailabilityChange,
    parseAvailabilityResult,
    parseCancelResult,
    parsePolicyChange,
    parsePolicyDocument,
    parsePolicySetResult,
    parsePolicyText,
    workloadCancelParams
} from '@/electron/service-bridge/node-policy'
import { workloadCancelRequest } from '@/shared/utils/workloads'
import { workloadModelLabel } from '@/ui/utils/workload-labels'

describe('parseWorkloadsInitial', () => {
    const base = { id: '7', engine: 'ollama', state: 'running', model: 'qwen3:4b', createdAt: 1 }

    it('keeps runId and requestedModel when present', () => {
        const [workload] = parseWorkloadsInitial({
            workloads: [{ ...base, runId: 'run-1', requestedModel: 'weak' }]
        })
        expect(workload.runId).toBe('run-1')
        expect(workload.requestedModel).toBe('weak')
    })

    it('leaves them off when absent or empty', () => {
        const [workload] = parseWorkloadsInitial({
            workloads: [{ ...base, runId: '', requestedModel: '' }]
        })
        expect(workload).not.toHaveProperty('runId')
        expect(workload).not.toHaveProperty('requestedModel')
    })
})

describe('workloadCancelRequest', () => {
    const job = {
        id: '7',
        engine: 'ollama' as const,
        state: 'running' as const,
        originatedFrom: 'node-a',
        runId: 'run-1'
    }

    it('builds a request for a running job', () => {
        expect(workloadCancelRequest(job, true)).toEqual({
            originatedFrom: 'node-a',
            workloadId: '7',
            engine: 'ollama',
            runId: 'run-1',
            regenerate: true
        })
    })

    it('omits runId when the job has none', () => {
        expect(workloadCancelRequest({ ...job, runId: undefined }, false)).not.toHaveProperty(
            'runId'
        )
    })

    it('refuses finished jobs and jobs with no origin', () => {
        expect(workloadCancelRequest({ ...job, state: 'completed' }, false)).toBeNull()
        expect(workloadCancelRequest({ ...job, originatedFrom: null }, false)).toBeNull()
    })
})

describe('workloadCancelParams', () => {
    it('spells the engine the way engine-manager does', () => {
        const params = workloadCancelParams({
            originatedFrom: 'node-a',
            workloadId: '7',
            engine: 'lm-studio',
            regenerate: false
        })
        expect(params).toEqual({
            originatedFrom: 'node-a',
            workloadId: '7',
            engine: 'lmstudio',
            regenerate: false
        })
    })
})

describe('workloadModelLabel', () => {
    it('shows what was asked for and what ran', () => {
        expect(workloadModelLabel({ model: 'qwen3:4b', requestedModel: 'weak' }, 'qwen3:4b')).toBe(
            'weak → qwen3:4b'
        )
    })

    it('shows only the model when nothing different was requested', () => {
        expect(workloadModelLabel({ model: 'qwen3:4b' }, 'qwen3:4b')).toBe('qwen3:4b')
        expect(workloadModelLabel({ model: 'm', requestedModel: 'm' }, 'm')).toBe('m')
    })
})

describe('node policy bridge helpers', () => {
    it('omits nodeId for this PC', () => {
        expect(nodeTargetParams({})).toEqual({})
        expect(nodeTargetParams({ nodeId: '' })).toEqual({})
        expect(nodeTargetParams({ nodeId: 'node-b' })).toEqual({ nodeId: 'node-b' })
    })

    it('turns a policy:get result into editable text', () => {
        const document = parsePolicyDocument({
            policy: { tier: 'weak', limits: { maxJobs: 2 } },
            availability: 'draining'
        })
        expect(document.availability).toBe('draining')
        expect(JSON.parse(document.policy)).toEqual({ tier: 'weak', limits: { maxJobs: 2 } })
        expect(document.policy).toContain('\n')
    })

    it('rejects malformed policy:get and set results', () => {
        expect(() => parsePolicyDocument({ policy: {}, availability: 'nope' })).toThrow()
        expect(() => parsePolicyDocument({ availability: 'available' })).toThrow()
        expect(() => parsePolicySetResult({})).toThrow()
        expect(() => parseAvailabilityResult({ availability: 3 })).toThrow()
    })

    it('reads set and availability results', () => {
        expect(JSON.parse(parsePolicySetResult({ policy: { a: 1 } }).policy)).toEqual({ a: 1 })
        expect(parseAvailabilityResult({ availability: 'paused' })).toEqual({
            availability: 'paused'
        })
        expect(parseCancelResult({ ok: true })).toEqual({ ok: true })
        expect(parseCancelResult(undefined)).toEqual({ ok: false })
    })

    it('parses policy text and explains bad input', () => {
        expect(parsePolicyText('{"a": 1}')).toEqual({ a: 1 })
        expect(() => parsePolicyText('{oops')).toThrow(/not valid JSON/)
        expect(() => parsePolicyText('[1]')).toThrow(/must be a JSON object/)
        expect(() => parsePolicyText('null')).toThrow(/must be a JSON object/)
    })
})

describe('node push parsers', () => {
    it('reads an availability change and ignores the extras', () => {
        expect(
            parseAvailabilityChange({ nodeId: 'n1', availability: 'draining', active: 2 })
        ).toEqual({ nodeId: 'n1', availability: 'draining' })
    })

    it('rejects an availability change without a node or a known state', () => {
        expect(parseAvailabilityChange({ availability: 'paused' })).toBeNull()
        expect(parseAvailabilityChange({ nodeId: 'n1', availability: 'sleeping' })).toBeNull()
        expect(parseAvailabilityChange(undefined)).toBeNull()
    })

    it('renders a policy change as the same pretty JSON text policy:get shows', () => {
        const change = parsePolicyChange({ nodeId: 'n1', policy: { cpu: { maxPercent: 50 } } })
        expect(change).toEqual({
            nodeId: 'n1',
            policy: JSON.stringify({ cpu: { maxPercent: 50 } }, null, 2)
        })
    })

    it('rejects a policy change without a node or an object policy', () => {
        expect(parsePolicyChange({ policy: {} })).toBeNull()
        expect(parsePolicyChange({ nodeId: 'n1', policy: 'x' })).toBeNull()
    })
})
