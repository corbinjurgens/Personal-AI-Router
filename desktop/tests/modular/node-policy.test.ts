// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act, createElement } from 'react'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { parseWorkloadsInitial } from '@/electron/service-bridge/workload-projection'
import {
    nodeTargetParams,
    parseAvailabilityChange,
    parseAvailabilityResult,
    parseCancelResult,
    parsePolicyChange,
    parsePolicyDocument,
    parsePolicySetResult,
    parsePolicyText,
    policySetParams,
    workloadCancelParams,
    parseStrictAdmissionCapability,
    parseStrictAdmissionRequest,
    parseAdvertisementStrictAdmission,
    parseRoutingDeployments
} from '@/electron/service-bridge/node-policy'
import type {
    NodeAvailabilityChange,
    NodePolicyChange,
    NodePolicyDocument,
    NodePolicySnapshot,
    PolicyActivationStatus,
    WorkloadCancelRequest
} from '@/shared/types/node-policy'
import { workloadCancelRequest } from '@/shared/utils/workloads'
import type { INodePolicyApi } from '@/ui/api/pair-api'
import RoutingSettings from '@/ui/components/RoutingSettings/RoutingSettings'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { workloadModelLabel } from '@/ui/utils/workload-labels'
import type { JsonObject, JsonValue } from '@/electron/service-bridge/json-rpc-client'

function strictFixtureObject(value: JsonValue | undefined): JsonObject {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        throw new Error('Invalid strict-admission test fixture')
    }
    return value
}

function loadStrictFixtures(): JsonObject {
    const value: JsonValue = JSON.parse(
        readFileSync(
            resolve(process.cwd(), '../services/shared/nodepolicy/testdata/strict-admission.json'),
            'utf8'
        )
    )
    return strictFixtureObject(value)
}

describe('served/resident policy contract', () => {
    it('preserves custom buffers and zero wait through the generic policy bridge', () => {
        const policy = {
            revision: '31',
            routing: { buffers: { capacityMiB: 384, waitSeconds: 0 } }
        }
        const document = parsePolicyDocument({ policy, availability: 'available' })
        const draft = parsePolicyText(document.policy)
        expect(draft).toEqual(policy)
        const params = policySetParams({
            policy: document.policy,
            expectedPersistedRevision: document.revision
        })
        expect(params).toEqual({ policy, expectedPersistedRevision: '31' })
        const result = parsePolicySetResult({ policy: params.policy })
        expect(parsePolicyText(result.policy)).toEqual(policy)
        const changed = parsePolicyChange({ nodeId: 'self', policy: params.policy })
        expect(changed).not.toBeNull()
        if (!changed) throw new Error('Missing buffer policy change')
        expect(parsePolicyText(changed.policy)).toEqual(policy)
    })
    const corpus: JsonValue = JSON.parse(
        readFileSync(
            resolve(process.cwd(), '../services/shared/nodepolicy/testdata/served-resident.json'),
            'utf8'
        )
    )
    const cases = strictFixtureObject(corpus).cases
    if (!Array.isArray(cases)) throw new Error('Missing served/resident corpus')
    for (const value of cases) {
        const fixture = strictFixtureObject(value)
        if (typeof fixture.name !== 'string') throw new Error('Missing fixture name')
        // Exact duplicate keys are a raw Go boundary: JSON.parse has already
        // collapsed them before a decoded-object consumer can inspect them.
        if (fixture.raw !== undefined) continue
        it(fixture.name, () => {
            const input = strictFixtureObject(fixture.input)
            const text = JSON.stringify({ ...input, revision: '31' })
            if (fixture.valid !== true) {
                expect(() => parsePolicyText(text)).toThrow()
                expect(() =>
                    parsePolicySetResult({ policy: { ...input, revision: '31' } })
                ).toThrow()
                return
            }
            const draft = parsePolicyText(text)
            expect(Object.hasOwn(draft, 'servedModels')).toBe(fixture.servedPresent)
            expect(draft.servedModels).toEqual(fixture.served)
            const saved = parsePolicySetResult({ policy: draft })
            const params = policySetParams({
                policy: saved.policy,
                expectedPersistedRevision: saved.revision,
                nodeId: 'peer'
            })
            expect(params).toEqual({
                nodeId: 'peer',
                policy: draft,
                expectedPersistedRevision: '31'
            })
            const readback = parsePolicyDocument({ policy: params.policy, availability: 'paused' })
            expect(parsePolicyText(readback.policy)).toEqual(draft)
        })
    }

    it('retains deny-all and private routing through notifications, stringify and CAS', () => {
        const policy = {
            revision: '9',
            servedModels: [],
            residentModels: [],
            routing: { accounts: [] }
        }
        const changed = parsePolicyChange({ nodeId: 'self', policy })
        expect(changed).not.toBeNull()
        if (!changed) throw new Error('Missing policy change')
        expect(
            policySetParams({ policy: changed.policy, expectedPersistedRevision: '9' }).policy
        ).toEqual({
            revision: '9',
            servedModels: [],
            residentModels: [],
            routing: { accounts: [] }
        })
    })

    /** Resident policy with one qualified profile at the given token bounds. */
    const residentPolicy = (contextTokens: number, maxOutputTokens: number): string =>
        JSON.stringify({
            revision: '31',
            servedModels: [{ engine: 'ollama', model: 'm' }],
            residentModels: [{ engine: 'ollama', model: 'm' }],
            profiles: [
                {
                    engine: 'ollama',
                    model: 'm',
                    contextTokens,
                    maxOutputTokens,
                    maxConcurrent: 1,
                    memoryBytes: 1024,
                    qualification: { id: 'q', revision: '1' }
                }
            ]
        })

    const RESIDENT = [{ engine: 'ollama', model: 'm' }]

    it('keeps a non-empty residentModels list through the bridge round trip', () => {
        const draft = parsePolicyText(residentPolicy(4096, 512))
        expect(draft).toHaveProperty('residentModels', RESIDENT)
        const saved = parsePolicySetResult({ policy: draft })
        const params = policySetParams({
            policy: saved.policy,
            expectedPersistedRevision: saved.revision
        })
        expect(params.policy).toEqual(draft)
    })

    it('accepts a resident output ceiling below a nonzero context bound', () => {
        expect(() => parsePolicyText(residentPolicy(4096, 512))).not.toThrow()
    })

    it('rejects a resident output ceiling at or above the context bound', () => {
        expect(() => parsePolicyText(residentPolicy(512, 512))).toThrow(
            'maxOutputTokens must be below a nonzero contextTokens'
        )
        expect(() => parsePolicyText(residentPolicy(512, 1024))).toThrow(
            'maxOutputTokens must be below a nonzero contextTokens'
        )
    })

    it('rejects a positive resident output ceiling without a context bound', () => {
        expect(() => parsePolicyText(residentPolicy(0, 512))).toThrow()
    })

    it('rejects resident permission outside the served set', () => {
        expect(() =>
            parsePolicyText(
                JSON.stringify({
                    servedModels: [],
                    residentModels: [{ engine: 'ollama', model: 'm' }]
                })
            )
        ).toThrow('subset')
    })

    it('documents the raw duplicate-key boundary', () => {
        expect(
            parsePolicyText('{"servedModels":[{"engine":"ollama","model":"m"}],"servedModels":[]}')
        ).toEqual({ servedModels: [] })
        expect(() => parsePolicyText('{"servedModels":[],"ServedModels":[]}')).toThrow()
    })

    it('rejects Unicode case-folded model preference keys like Go', () => {
        expect(() => parsePolicyText('{"ſervedModels":[]}')).toThrow()
        expect(() => parsePolicyText('{"reſidentModels":[]}')).toThrow()
    })
})

describe('strict-admission shared Go/TypeScript parity corpus', () => {
    const cases = loadStrictFixtures().cases
    if (!Array.isArray(cases)) throw new Error('Missing strict-admission corpus')
    for (const value of cases) {
        const testCase = strictFixtureObject(value)
        if (typeof testCase.name !== 'string') throw new Error('Missing corpus case name')
        it(testCase.name, () => {
            const fixtures = loadStrictFixtures()
            const request = strictFixtureObject(fixtures.request)
            const identity = strictFixtureObject(fixtures.identity)
            const model = strictFixtureObject(fixtures.model)
            request.identity = identity
            model.identity = identity
            const capability: JsonObject = { version: 1, models: [model] }
            const targets: Record<string, JsonObject> = { request, identity, model, capability }
            if (
                typeof testCase.kind !== 'string' ||
                !targets[testCase.kind] ||
                !Array.isArray(testCase.path)
            ) {
                throw new Error('Invalid corpus mutation')
            }
            let target = targets[testCase.kind]
            const path = testCase.path
            for (let i = 0; i < path.length; i++) {
                const key = path[i]
                if (typeof key !== 'string') throw new Error('Invalid corpus path')
                if (i < path.length - 1) target = strictFixtureObject(target[key])
                else if (testCase.remove === true) delete target[key]
                else {
                    if (testCase.value === undefined) throw new Error('Missing corpus value')
                    target[key] = testCase.value
                }
            }
            const isCapability = testCase.kind === 'capability' || testCase.kind === 'model'
            const parse = () =>
                isCapability
                    ? parseStrictAdmissionCapability(capability)
                    : parseStrictAdmissionRequest(request)
            if (testCase.valid !== true) {
                expect(parse).toThrow()
            } else {
                const got = parse()
                if (isCapability) expect(got).toHaveProperty('state', testCase.state ?? 'valid')
                const encoded: JsonValue = JSON.parse(
                    JSON.stringify(isCapability ? capability : request)
                )
                const again = isCapability
                    ? parseStrictAdmissionCapability(encoded)
                    : parseStrictAdmissionRequest(encoded)
                expect(again).toEqual(got)
            }
        })
    }

    it('distinguishes absent from null', () => {
        expect(parseStrictAdmissionCapability(undefined)).toEqual({ state: 'absent' })
        expect(() => parseStrictAdmissionCapability(null)).toThrow()
    })

    it('preserves unsupported versions without interpreting their fields', () => {
        expect(parseStrictAdmissionCapability({ version: 2, future: true })).toEqual({
            state: 'unsupported',
            version: 2
        })
    })

    it('rejects duplicate engine/model entries', () => {
        const fixtures = loadStrictFixtures()
        const model = strictFixtureObject(fixtures.model)
        model.identity = fixtures.identity
        expect(() =>
            parseStrictAdmissionCapability({ version: 1, models: [model, model] })
        ).toThrow()
    })

    it('rejects excess model entries', () => {
        expect(() =>
            parseStrictAdmissionCapability({
                version: 1,
                models: Array.from({ length: 1025 }, () => null)
            })
        ).toThrow()
    })

    it.each([
        'ollama',
        'lmstudio',
        'llamacpp',
        'llamacpp-custom',
        'strata',
        'errors',
        'e-x',
        'lm-studio',
        'llama-cpp',
        'A_1.-',
        '-engine',
        '_engine',
        '.engine',
        '...',
        'a'.repeat(40)
    ])('preserves routing engine ID %j in the parsed capability', engine => {
        const fixtures = loadStrictFixtures()
        const model = strictFixtureObject(fixtures.model)
        model.identity = fixtures.identity
        model.engine = engine
        const capability = parseStrictAdmissionCapability({ version: 1, models: [model] })
        if (capability.state !== 'valid') throw new Error('Expected valid capability')
        expect(capability.models).toEqual([model])
        expect(capability.models[0].engine).toBe(engine)
    })

    it.each([
        { name: 'empty', engine: '' },
        { name: 'dot', engine: '.' },
        { name: 'dotdot', engine: '..' },
        { name: 'slash', engine: 'a/b' },
        { name: 'backslash', engine: 'a\\b' },
        { name: 'colon', engine: 'a:b' },
        { name: 'space', engine: 'a b' },
        { name: 'overlong', engine: 'a'.repeat(41) },
        { name: 'final newline', engine: 'a\n' },
        { name: 'embedded newline', engine: 'a\nb' },
        { name: 'carriage return', engine: 'a\r' },
        { name: 'CRLF', engine: 'a\r\n' },
        { name: 'line separator', engine: 'a\u2028' },
        { name: 'paragraph separator', engine: 'a\u2029' },
        { name: 'unicode', engine: 'engin\u00e9' },
        { name: 'unpaired surrogate', engine: 'a\ud800' },
        { name: 'tab', engine: 'a\t' },
        { name: 'NUL', engine: 'a\0' }
    ])('rejects a $name routing engine ID in the capability', ({ engine }) => {
        const fixtures = loadStrictFixtures()
        const model = strictFixtureObject(fixtures.model)
        model.identity = fixtures.identity
        model.engine = engine
        expect(() => parseStrictAdmissionCapability({ version: 1, models: [model] })).toThrow(
            'Invalid strict-admission engine/model'
        )
    })

    it.each(['1.0', '1e0'])('accepts integral JSON spelling %s', spelling => {
        const fixtures = loadStrictFixtures()
        const request = strictFixtureObject(fixtures.request)
        request.identity = fixtures.identity
        const input: JsonValue = JSON.parse(
            JSON.stringify(request).replace('"version":1', `"version":${spelling}`)
        )
        expect(parseStrictAdmissionRequest(input).version).toBe(1)
    })
})

describe('Advertisement raw Go/TypeScript parity corpus', () => {
    const cases: JsonValue = JSON.parse(
        readFileSync(
            resolve(
                process.cwd(),
                '../services/shared/nodepolicy/testdata/strict-admission-raw.json'
            ),
            'utf8'
        )
    )
    if (!Array.isArray(cases)) throw new Error('Missing raw Advertisement corpus')
    for (const value of cases) {
        const testCase = strictFixtureObject(value)
        if (typeof testCase.name !== 'string') throw new Error('Missing raw case name')
        it(testCase.name, () => {
            if (typeof testCase.document !== 'string') throw new Error('Missing raw document')
            const input: JsonValue = JSON.parse(testCase.document)
            const parse = () => parseAdvertisementStrictAdmission(input)
            if (typeof testCase.error === 'string') {
                expect(parse).toThrow(testCase.error)
                return
            }
            const got = parse()
            expect(got.state).toBe(testCase.state)
            if (testCase.model !== undefined) {
                expect(got.state).toBe('valid')
                if (got.state !== 'valid') throw new Error('Expected valid capability')
                expect(got.models[0].model).toBe(testCase.model)
                // Serialize the decoded contract, so normalized names must survive.
                const roundTrip: JsonValue = JSON.parse(
                    JSON.stringify({
                        strictAdmission: { version: got.version, models: got.models }
                    })
                )
                expect(parseAdvertisementStrictAdmission(roundTrip)).toEqual(got)
            }
        })
    }
})

function loadDeploymentCorpus(): JsonObject {
    const value: JsonValue = JSON.parse(
        readFileSync(
            resolve(process.cwd(), '../services/shared/nodepolicy/testdata/deployment.json'),
            'utf8'
        )
    )
    return strictFixtureObject(value)
}

/** A routing section holding the corpus account and the given deployments. */
function corpusRouting(...deployments: JsonValue[]): JsonObject {
    return { accounts: [loadDeploymentCorpus().account ?? null], deployments }
}

describe('deployment endpoint schema Go/TypeScript parity corpus', () => {
    const cases = loadDeploymentCorpus().cases
    if (!Array.isArray(cases)) throw new Error('Missing deployment corpus')
    for (const value of cases) {
        const testCase = strictFixtureObject(value)
        if (typeof testCase.name !== 'string' || !Array.isArray(testCase.path)) {
            throw new Error('Invalid deployment corpus case')
        }
        it(testCase.name, () => {
            const deployment = strictFixtureObject(loadDeploymentCorpus().deployment)
            let target = deployment
            const path = testCase.path
            if (!Array.isArray(path)) throw new Error('Invalid corpus path')
            for (let i = 0; i < path.length; i++) {
                const key = path[i]
                if (typeof key !== 'string') throw new Error('Invalid corpus path')
                if (i < path.length - 1) target = strictFixtureObject(target[key])
                else if (testCase.remove === true) delete target[key]
                else {
                    if (testCase.value === undefined) throw new Error('Missing corpus value')
                    target[key] = testCase.value
                }
            }
            if (testCase.enabled === true) deployment.enabled = true
            const parse = () => parseRoutingDeployments(corpusRouting(deployment))
            if (testCase.valid !== true) {
                expect(parse).toThrow()
                return
            }
            const [got] = parse()
            // What the parser returns serializes back to an equal deployment.
            const again = parseRoutingDeployments(
                corpusRouting(JSON.parse(JSON.stringify(got, (_key, v) => v ?? undefined)))
            )
            expect(again).toEqual([got])
        })
    }

    it('keeps money as exact strings, including the signed 64-bit maximum', () => {
        const [deployment] = parseRoutingDeployments(
            corpusRouting(loadDeploymentCorpus().deployment ?? null)
        )
        expect(deployment.prices).toEqual({
            currency: 'USD',
            unit: 'nanoUSDPerMillionTokens',
            inputPerMillion: '150000000',
            outputPerMillion: '9223372036854775807'
        })
        expect(deployment.endpoint).toBe('https://api.example.com/v1/chat/completions')
        expect(deployment.transport?.maxResponseBytes).toBe(16777216)
        expect(deployment.qualification).toEqual({ id: 'provider-q', revision: '3' })
    })

    it('reads absent optional parts as null and capabilities as empty', () => {
        const deployment = strictFixtureObject(loadDeploymentCorpus().deployment)
        for (const key of [
            'capabilities',
            'prices',
            'endpoint',
            'protocol',
            'transport',
            'qualification'
        ]) {
            delete deployment[key]
        }
        expect(parseRoutingDeployments(corpusRouting(deployment))).toEqual([
            expect.objectContaining({
                capabilities: [],
                prices: null,
                endpoint: null,
                protocol: null,
                transport: null,
                qualification: null
            })
        ])
    })

    it('does not mirror the temporary refusal to enable a complete deployment', () => {
        const deployment = strictFixtureObject(loadDeploymentCorpus().deployment)
        deployment.enabled = true
        expect(parseRoutingDeployments(corpusRouting(deployment))[0].enabled).toBe(true)
    })

    it.each([
        ['an unknown deployment key', ['extra'], true],
        ['a case-variant deployment key', ['Endpoint'], 'https://api.example.com/v1'],
        ['an unknown transport key', ['transport', 'extra'], 1],
        ['an unknown price key', ['prices', 'extra'], '1'],
        ['an unknown qualification key', ['qualification', 'extra'], '1'],
        ['a null endpoint', ['endpoint'], null],
        ['a null protocol', ['protocol'], null],
        ['a null prices', ['prices'], null],
        ['a null capabilities', ['capabilities'], null],
        ['a null transport', ['transport'], null],
        ['a null qualification', ['qualification'], null],
        ['an empty endpoint', ['endpoint'], '']
    ])('refuses %s outside the canonical broker output', (_name, path, value) => {
        const deployment = strictFixtureObject(loadDeploymentCorpus().deployment)
        let target = deployment
        for (const key of path.slice(0, -1)) target = strictFixtureObject(target[key])
        target[path[path.length - 1]] = value
        expect(() => parseRoutingDeployments(corpusRouting(deployment))).toThrow()
    })

    it('refuses duplicate deployment ids and a malformed routing section', () => {
        const deployment = loadDeploymentCorpus().deployment ?? null
        expect(() => parseRoutingDeployments(corpusRouting(deployment, deployment))).toThrow(
            'Duplicate deployment id'
        )
        expect(() => parseRoutingDeployments(null)).toThrow('Invalid routing section')
        expect(() => parseRoutingDeployments({ accounts: [] })).toThrow('Invalid routing section')
    })
})

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

    it('refuses incomplete target keys', () => {
        expect(workloadCancelRequest({ ...job, runId: undefined }, false)).toBeNull()
        expect(workloadCancelRequest({ ...job, runId: '' }, false)).toBeNull()
        expect(workloadCancelRequest({ ...job, id: '' }, false)).toBeNull()
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
            runId: 'run-1',
            regenerate: false
        })
        expect(params).toEqual({
            originatedFrom: 'node-a',
            workloadId: '7',
            engine: 'lmstudio',
            runId: 'run-1',
            regenerate: false
        })
    })

    it('rejects incomplete targets at the bridge', () => {
        const request: WorkloadCancelRequest = {
            originatedFrom: 'node-a',
            workloadId: '7',
            engine: 'ollama',
            runId: 'run-1',
            regenerate: false
        }
        expect(() => workloadCancelParams({ ...request, originatedFrom: '' })).toThrow()
        expect(() => workloadCancelParams({ ...request, workloadId: '' })).toThrow()
        expect(() => workloadCancelParams({ ...request, runId: '' })).toThrow()
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
            policy: { revision: '5', tier: 'weak', limits: { maxJobs: 2 } },
            availability: 'draining'
        })
        expect(document.availability).toBe('draining')
        expect(JSON.parse(document.policy)).toEqual({
            revision: '5',
            tier: 'weak',
            limits: { maxJobs: 2 }
        })
        expect(document.policy).toContain('\n')
    })

    it('rejects malformed policy:get and set results', () => {
        expect(() => parsePolicyDocument({ policy: {}, availability: 'nope' })).toThrow()
        expect(() => parsePolicyDocument({ availability: 'available' })).toThrow()
        expect(() => parsePolicySetResult({})).toThrow()
        expect(() => parseAvailabilityResult({ availability: 3 })).toThrow()
    })

    it('reads set and availability results', () => {
        expect(
            JSON.parse(parsePolicySetResult({ policy: { revision: '2', a: 1 } }).policy)
        ).toEqual({ revision: '2', a: 1 })
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
        const policy = { revision: '3', cpu: { maxPercent: 50 } }
        expect(parsePolicyChange({ nodeId: 'n1', policy })).toEqual({
            nodeId: 'n1',
            policy: JSON.stringify(policy, null, 2),
            revision: '3',
            activation: null
        })
    })

    it('rejects a policy change without a node or an object policy', () => {
        expect(parsePolicyChange({ policy: { revision: '1' } })).toBeNull()
        expect(parsePolicyChange({ nodeId: 'n1', policy: 'x' })).toBeNull()
    })
})

describe('policy revision and activation fields', () => {
    it('reads the persisted revision and the activation report from policy:get', () => {
        const document = parsePolicyDocument({
            policy: { revision: '12' },
            availability: 'available',
            recoveryRequired: false,
            activeRevision: '11',
            activation: 'error',
            activationError: 'proxy refused the policy'
        })
        expect(document.revision).toBe('12')
        expect(document.recoveryRequired).toBe(false)
        expect(document.activation).toEqual({
            state: 'error',
            activeRevision: '11',
            error: 'proxy refused the policy'
        })
    })

    it('reads an active policy with no error text', () => {
        const result = parsePolicySetResult({
            policy: { revision: '4' },
            activeRevision: '4',
            activation: 'active'
        })
        expect(result.revision).toBe('4')
        expect(result.activation).toEqual({ state: 'active', activeRevision: '4' })
    })

    it('reads a pending policy that no proxy has acknowledged', () => {
        const change = parsePolicyChange({
            nodeId: 'n1',
            policy: { revision: '9' },
            activeRevision: '',
            activation: 'pending'
        })
        expect(change?.activation).toEqual({ state: 'pending', activeRevision: '' })
    })

    it('reports recovery from policy:get', () => {
        const document = parsePolicyDocument({
            policy: { revision: '1' },
            availability: 'paused',
            recoveryRequired: true,
            activeRevision: '',
            activation: 'pending'
        })
        expect(document.recoveryRequired).toBe(true)
    })

    it('reads a paired node answer that carries no activation or recovery fields', () => {
        const document = parsePolicyDocument({
            policy: { revision: '30' },
            availability: 'available'
        })
        expect(document.activation).toBeNull()
        expect(document.recoveryRequired).toBe(false)
    })

    it.each([
        ['a numeric revision', { policy: { revision: 7 } }],
        ['a revision with a leading zero', { policy: { revision: '07' } }],
        ['a signed revision', { policy: { revision: '-1' } }],
        ['no revision', { policy: {} }],
        ['an unknown activation state', { policy: { revision: '1' }, activation: 'on' }],
        [
            'a numeric active revision',
            { policy: { revision: '1' }, activation: 'active', activeRevision: 1 }
        ],
        ['a missing active revision', { policy: { revision: '1' }, activation: 'active' }],
        [
            'a non-decimal active revision',
            { policy: { revision: '1' }, activation: 'active', activeRevision: '1.5' }
        ],
        [
            'a non-text activation error',
            {
                policy: { revision: '1' },
                activation: 'error',
                activeRevision: '',
                activationError: 3
            }
        ]
    ])('rejects a policy:set result with %s', (_name, result) => {
        expect(() => parsePolicySetResult(result)).toThrow()
    })

    it.each([
        ['an unknown activation state', { activation: 'on' }],
        ['a non-decimal revision', { policy: { revision: 'abc' } }]
    ])('drops a policy:changed push with %s', (_name, extra) => {
        expect(
            parsePolicyChange({
                nodeId: 'n1',
                policy: { revision: '1' },
                activeRevision: '',
                activation: 'pending',
                ...extra
            })
        ).toBeNull()
    })

    it('rejects a policy:get result with a non-boolean recovery flag', () => {
        expect(() =>
            parsePolicyDocument({
                policy: { revision: '1' },
                availability: 'available',
                recoveryRequired: 'yes'
            })
        ).toThrow(/recovery flag/)
    })
})

describe('model configuration policy evidence', () => {
    const loaded: JsonObject = {
        engine: 'ollama',
        model: 'model-a',
        desiredResident: true,
        observedLoaded: true,
        state: 'satisfied'
    }
    const configuration: JsonObject = {
        policyRevision: '9',
        observationGeneration: 17,
        models: [
            loaded,
            {
                engine: 'llamacpp',
                model: 'model-b',
                desiredResident: false,
                observedLoaded: true,
                state: 'pending'
            },
            {
                engine: 'ollama',
                model: 'model-c',
                desiredResident: true,
                observedLoaded: false,
                state: 'blocked',
                reason: 'not-installed'
            },
            {
                engine: 'ollama',
                model: 'model-d',
                desiredResident: true,
                state: 'unknown',
                reason: 'observation-unknown'
            }
        ]
    }

    it.each(['local', 'peer'])(
        'preserves %s get, set and push evidence outside editable policy',
        node => {
            const policy = { revision: '9', servedModels: [], residentModels: [] }
            const snapshot = {
                policy,
                modelConfiguration: configuration,
                ...(node === 'local' ? { activation: 'active', activeRevision: '9' } : {})
            }
            const document = parsePolicyDocument({ ...snapshot, availability: 'available' })
            const saved = parsePolicySetResult(snapshot)
            const change = parsePolicyChange({ ...snapshot, nodeId: node })
            expect(document.modelConfiguration).toEqual(configuration)
            expect(saved.modelConfiguration).toEqual(configuration)
            expect(change?.modelConfiguration).toEqual(configuration)
            expect(JSON.parse(document.policy)).toEqual(policy)
            expect(JSON.parse(saved.policy)).toEqual(policy)
            expect(change?.policy).toBe(document.policy)
            expect(document.modelConfiguration?.models[3]).not.toHaveProperty('observedLoaded')
        }
    )

    it('keeps absent status absent on get, set and push', () => {
        const snapshot = { policy: { revision: '9' } }
        expect(parsePolicyDocument({ ...snapshot, availability: 'available' })).not.toHaveProperty(
            'modelConfiguration'
        )
        expect(parsePolicySetResult(snapshot)).not.toHaveProperty('modelConfiguration')
        expect(parsePolicyChange({ ...snapshot, nodeId: 'peer' })).not.toHaveProperty(
            'modelConfiguration'
        )
    })

    it('preserves explicit empty evidence and served permission without inferring success', () => {
        const evidence = { policyRevision: '9', observationGeneration: 0, models: [] }
        const policy = { revision: '9', servedModels: [] }
        const snapshot = { policy, modelConfiguration: evidence }
        expect(
            parsePolicyDocument({ ...snapshot, availability: 'available' }).modelConfiguration
        ).toEqual(evidence)
        expect(parsePolicySetResult(snapshot).modelConfiguration).toEqual(evidence)
        expect(parsePolicyChange({ ...snapshot, nodeId: 'peer' })?.modelConfiguration).toEqual(
            evidence
        )
        expect(parsePolicyText(parsePolicySetResult(snapshot).policy)).toEqual(policy)
    })

    const malformed: [string, JsonValue][] = [
        ['null status', null],
        ['missing policy revision', { observationGeneration: 0, models: [] }],
        ['numeric policy revision', { ...configuration, policyRevision: 9 }],
        [
            'overflowing policy revision',
            { ...configuration, policyRevision: '9223372036854775808' }
        ],
        ['missing generation', { policyRevision: '9', models: [] }],
        ['negative generation', { ...configuration, observationGeneration: -1 }],
        ['fractional generation', { ...configuration, observationGeneration: 1.5 }],
        ['non-list models', { ...configuration, models: {} }],
        [
            'too many models',
            { ...configuration, models: Array.from({ length: 257 }, () => loaded) }
        ],
        ['duplicate model', { ...configuration, models: [loaded, loaded] }],
        ['unknown status field', { ...configuration, extra: true }],
        ['unknown model field', { ...configuration, models: [{ ...loaded, secret: 'private' }] }],
        ['unknown state', { ...configuration, models: [{ ...loaded, state: 'ready' }] }],
        [
            'unknown reason',
            { ...configuration, models: [{ ...loaded, reason: 'engine error text' }] }
        ],
        [
            'missing desired flag',
            { ...configuration, models: [{ engine: 'ollama', model: 'm', state: 'unknown' }] }
        ],
        [
            'non-boolean loaded',
            { ...configuration, models: [{ ...loaded, observedLoaded: 'yes' }] }
        ],
        ['null loaded', { ...configuration, models: [{ ...loaded, observedLoaded: null }] }],
        [
            'satisfied without loaded evidence',
            {
                ...configuration,
                models: [
                    { engine: 'ollama', model: 'm', desiredResident: true, state: 'satisfied' }
                ]
            }
        ],
        [
            'satisfied with unloaded evidence',
            { ...configuration, models: [{ ...loaded, observedLoaded: false }] }
        ],
        [
            'satisfied without desire',
            { ...configuration, models: [{ ...loaded, desiredResident: false }] }
        ],
        ['satisfied with refusal', { ...configuration, models: [{ ...loaded, reason: 'no-fit' }] }],
        ['invalid reference', { ...configuration, models: [{ ...loaded, engine: '../engine' }] }],
        ['reserved alias', { ...configuration, models: [{ ...loaded, model: 'weak' }] }],
        ['untrimmed model', { ...configuration, models: [{ ...loaded, model: ' m ' }] }],
        ['oversized model', { ...configuration, models: [{ ...loaded, model: 'm'.repeat(1025) }] }]
    ]
    it.each(malformed)('rejects %s on get/set and drops its push', (_name, evidence) => {
        const snapshot = { policy: { revision: '9' }, modelConfiguration: evidence }
        expect(() => parsePolicyDocument({ ...snapshot, availability: 'available' })).toThrow()
        expect(() => parsePolicySetResult(snapshot)).toThrow()
        expect(parsePolicyChange({ ...snapshot, nodeId: 'peer' })).toBeNull()
    })
})

describe('policy set params', () => {
    it('sends the base revision beside the edited policy', () => {
        expect(
            policySetParams({ policy: '{"tier":"weak"}', expectedPersistedRevision: '41' })
        ).toEqual({ policy: { tier: 'weak' }, expectedPersistedRevision: '41' })
    })

    it('names the target node when one is given', () => {
        expect(
            policySetParams({
                nodeId: 'node-b',
                policy: '{}',
                expectedPersistedRevision: '0'
            })
        ).toEqual({ nodeId: 'node-b', policy: {}, expectedPersistedRevision: '0' })
    })

    it.each(['', '1.5', '-1', '1e3', '007', ' 1'])('refuses the base revision %j', revision => {
        expect(() =>
            policySetParams({ policy: '{}', expectedPersistedRevision: revision })
        ).toThrow(/decimal string/)
    })

    it('keeps numbers beyond double precision exact as strings end to end', () => {
        const text = JSON.stringify({
            revision: '9223372036854775807',
            budget: { monthly: '9007199254740993' }
        })
        expect(parsePolicyText(text)).toEqual({
            revision: '9223372036854775807',
            budget: { monthly: '9007199254740993' }
        })
        expect(
            policySetParams({ policy: text, expectedPersistedRevision: '9223372036854775807' })
        ).toEqual({
            policy: { revision: '9223372036854775807', budget: { monthly: '9007199254740993' } },
            expectedPersistedRevision: '9223372036854775807'
        })
        const document = parsePolicyDocument({
            policy: { revision: '9223372036854775807', budget: { monthly: '9007199254740993' } },
            availability: 'available',
            activeRevision: '9223372036854775806',
            activation: 'pending'
        })
        expect(document.revision).toBe('9223372036854775807')
        expect(document.activation?.activeRevision).toBe('9223372036854775806')
        expect(document.policy).toContain('"9007199254740993"')
    })
})

describe('routing settings editor', () => {
    const SELF = 'self'
    const EDITOR = 'Node policy JSON'
    const CONFLICT = 'The saved policy changed after you loaded it'

    const policyText = (revision: string, tier = 'weak'): string =>
        JSON.stringify({ revision, tier, servedModels: [], residentModels: [] }, null, 2)
    const activation = (
        state: PolicyActivationStatus['state'],
        activeRevision: string
    ): PolicyActivationStatus => ({ state, activeRevision })
    const snapshot = (
        revision: string,
        state: PolicyActivationStatus['state'] = 'active',
        tier = 'weak'
    ): NodePolicySnapshot => ({
        policy: policyText(revision, tier),
        revision,
        activation: activation(state, state === 'active' ? revision : '')
    })
    const savedDocument = (
        revision: string,
        over: Partial<NodePolicyDocument> = {}
    ): NodePolicyDocument => ({
        ...snapshot(revision),
        availability: 'available',
        recoveryRequired: false,
        ...over
    })

    const get = vi.fn<INodePolicyApi['get']>()
    const set = vi.fn<INodePolicyApi['set']>()
    const setAvailability = vi.fn<INodePolicyApi['setAvailability']>()
    let emitPolicy: (change: NodePolicyChange) => void
    let emitAvailability: (change: NodeAvailabilityChange) => void
    let container: HTMLDivElement
    let root: Root

    beforeEach(async () => {
        vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
        // jsdom has no CSS.supports, which the auto-sizing text area asks.
        vi.stubGlobal('CSS', { supports: () => true })
        Object.defineProperty(window, 'pairApi', {
            configurable: true,
            value: {
                nodePolicy: {
                    get,
                    set,
                    setAvailability,
                    onPolicyChanged: (callback: (change: NodePolicyChange) => void) => {
                        emitPolicy = callback
                        return () => undefined
                    },
                    onAvailabilityChanged: (callback: (change: NodeAvailabilityChange) => void) => {
                        emitAvailability = callback
                        return () => undefined
                    }
                }
            }
        })
        useConnectionStore.setState({ connected: true, selfId: SELF })
        container = document.createElement('div')
        document.body.append(container)
        root = createRoot(container)
    })
    afterEach(async () => {
        await act(async () => root.unmount())
        container.remove()
        useConnectionStore.setState({ connected: false, selfId: null })
        vi.unstubAllGlobals()
    })

    async function open(): Promise<void> {
        await act(async () => root.render(createElement(RoutingSettings)))
    }
    function editor(): HTMLTextAreaElement {
        const found = container.querySelector<HTMLTextAreaElement>(
            `textarea[aria-label="${EDITOR}"]`
        )
        if (!found) throw new Error('the policy editor is not rendered')
        return found
    }
    function button(label: string): HTMLButtonElement {
        const found = Array.from(container.querySelectorAll('button')).find(
            candidate => candidate.textContent?.trim() === label
        )
        if (!found) throw new Error(`no button labelled ${label}`)
        return found
    }
    async function click(label: string): Promise<void> {
        await act(async () => button(label).click())
    }
    async function type(text: string): Promise<void> {
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set
        if (!setter) throw new Error('textarea has no value setter')
        await act(async () => {
            setter.call(editor(), text)
            editor().dispatchEvent(new Event('input', { bubbles: true }))
        })
    }
    async function push(change: NodePolicyChange): Promise<void> {
        await act(async () => emitPolicy(change))
    }
    const lastBase = (): string | undefined => set.mock.lastCall?.[0].expectedPersistedRevision

    describe('activation status', () => {
        it('shows a saved policy the proxy has not enforced as pending', async () => {
            get.mockResolvedValue(savedDocument('5', { activation: activation('pending', '4') }))
            await open()
            expect(container.textContent).toContain('Saved, not yet enforced')
            expect(container.textContent).not.toContain('Active')
        })

        it('shows only an acknowledged policy as active', async () => {
            get.mockResolvedValue(savedDocument('5'))
            await open()
            expect(container.textContent).toContain('Active')
            expect(container.textContent).not.toContain('not yet enforced')
        })

        it('shows why an activation failed', async () => {
            get.mockResolvedValue(
                savedDocument('5', {
                    activation: { state: 'error', activeRevision: '4', error: 'proxy unreachable' }
                })
            )
            await open()
            expect(container.textContent).toContain('Saved, not enforced')
            expect(container.textContent).toContain('proxy unreachable')
        })

        it('shows nothing about enforcement for a node that reports none', async () => {
            get.mockResolvedValue(savedDocument('5', { activation: null }))
            await open()
            expect(container.textContent).not.toContain('Saved policy')
        })

        it('warns that defaults are in use after an unreadable file', async () => {
            get.mockResolvedValue(savedDocument('1', { recoveryRequired: true }))
            await open()
            expect(container.textContent).toContain('saved policy file was unreadable')
        })

        it('follows a pending save to active when the proxy acknowledges it', async () => {
            get.mockResolvedValue(savedDocument('5'))
            set.mockResolvedValue(snapshot('6', 'pending'))
            await open()
            await type(policyText('5', 'strong'))
            await click('Save')
            expect(container.textContent).toContain('Saved, not yet enforced')
            await push({ nodeId: SELF, ...snapshot('6', 'active', 'strong') })
            expect(container.textContent).toContain('Active')
            expect(container.textContent).not.toContain('not yet enforced')
        })

        it('keeps a newer pending status when the reply for an older revision arrives after it', async () => {
            get.mockResolvedValue(savedDocument('5'))
            let reply: (saved: NodePolicySnapshot) => void = () => undefined
            set.mockReturnValue(new Promise(resolve => (reply = resolve)))
            await open()
            await type(policyText('5', 'strong'))
            await click('Save')
            await push({ nodeId: SELF, ...snapshot('7', 'pending') })
            await act(async () => reply(snapshot('6', 'active')))
            expect(container.textContent).toContain('Saved, not yet enforced')
            expect(container.textContent).not.toContain('Active')
        })

        it('takes a newer reply over an older notification that arrived first', async () => {
            get.mockResolvedValue(savedDocument('5'))
            let reply: (saved: NodePolicySnapshot) => void = () => undefined
            set.mockReturnValue(new Promise(resolve => (reply = resolve)))
            await open()
            await type(policyText('5', 'strong'))
            await click('Save')
            await push({ nodeId: SELF, ...snapshot('6', 'pending') })
            await act(async () => reply(snapshot('7', 'active')))
            expect(container.textContent).toContain('Active')
            expect(container.textContent).not.toContain('not yet enforced')
        })

        it('keeps the newest revision status when an older notification arrives late', async () => {
            get.mockResolvedValue(savedDocument('7'))
            await open()
            await push({ nodeId: SELF, ...snapshot('6', 'pending') })
            expect(container.textContent).toContain('Active')
            expect(container.textContent).not.toContain('not yet enforced')
        })

        it('updates the status while the user has unsaved edits', async () => {
            get.mockResolvedValue(savedDocument('5'))
            await open()
            await type('{"draft": true}')
            await push({ nodeId: SELF, ...snapshot('6', 'pending') })
            expect(container.textContent).toContain('Saved, not yet enforced')
            expect(editor().value).toBe('{"draft": true}')
        })
    })

    describe('base revision', () => {
        it('saves against the revision it loaded', async () => {
            get.mockResolvedValue(savedDocument('9223372036854775807'))
            set.mockResolvedValue(snapshot('9223372036854775808'))
            await open()
            await type(policyText('9223372036854775807', 'strong'))
            await click('Save')
            expect(lastBase()).toBe('9223372036854775807')
        })

        it('saves the next edit against the revision the last save produced', async () => {
            get.mockResolvedValue(savedDocument('5'))
            set.mockResolvedValue(snapshot('6'))
            await open()
            await type(policyText('5', 'strong'))
            await click('Save')
            await type(policyText('6', 'stronger'))
            await click('Save')
            expect(set.mock.calls.map(([request]) => request.expectedPersistedRevision)).toEqual([
                '5',
                '6'
            ])
        })

        it('follows a newer notification while the editor is clean', async () => {
            get.mockResolvedValue(savedDocument('5'))
            set.mockResolvedValue(snapshot('7'))
            await open()
            await push({ nodeId: SELF, ...snapshot('6', 'active', 'other') })
            expect(editor().value).toBe(policyText('6', 'other'))
            await type(policyText('6', 'edited'))
            await click('Save')
            expect(lastBase()).toBe('6')
        })

        it('ignores a notification older than the base', async () => {
            get.mockResolvedValue(savedDocument('9223372036854775807'))
            set.mockResolvedValue(snapshot('9223372036854775808'))
            await open()
            await push({ nodeId: SELF, ...snapshot('9007199254740993', 'active', 'old') })
            expect(editor().value).toBe(policyText('9223372036854775807'))
            await type(policyText('9223372036854775807', 'edited'))
            await click('Save')
            expect(lastBase()).toBe('9223372036854775807')
        })

        it('ignores a notification for another node', async () => {
            get.mockResolvedValue(savedDocument('5'))
            await open()
            await push({ nodeId: 'other', ...snapshot('6', 'active', 'theirs') })
            expect(editor().value).toBe(policyText('5'))
        })

        it('keeps unsaved text when a newer notification arrives', async () => {
            get.mockResolvedValue(savedDocument('5'))
            set.mockResolvedValue(snapshot('7'))
            await open()
            await type('{"draft": true}')
            await push({ nodeId: SELF, ...snapshot('6', 'active', 'other') })
            expect(editor().value).toBe('{"draft": true}')
            await click('Save')
            expect(lastBase()).toBe('5')
        })
    })

    describe('availability change', () => {
        beforeEach(() => {
            setAvailability.mockResolvedValue({ availability: 'paused' })
        })

        it('takes the new revision from the policy notification so the next save is not stale', async () => {
            get.mockResolvedValue(savedDocument('5'))
            set.mockResolvedValue(snapshot('7'))
            await open()
            setAvailability.mockImplementation(async () => {
                emitPolicy({ nodeId: SELF, ...snapshot('6') })
                return { availability: 'paused' }
            })
            await click('Pause')
            await type(policyText('6', 'edited'))
            await click('Save')
            expect(lastBase()).toBe('6')
            expect(container.textContent).not.toContain(CONFLICT)
        })

        it('reads the new revision back when only the availability result arrives', async () => {
            get.mockResolvedValueOnce(savedDocument('5')).mockResolvedValue(savedDocument('6'))
            set.mockResolvedValue(snapshot('7'))
            await open()
            await click('Pause')
            await type(policyText('6', 'edited'))
            await click('Save')
            expect(lastBase()).toBe('6')
            expect(container.textContent).not.toContain(CONFLICT)
        })

        it('does not move the base back when the read-back is older than a notification', async () => {
            get.mockResolvedValueOnce(savedDocument('5')).mockResolvedValue(savedDocument('5'))
            set.mockResolvedValue(snapshot('7'))
            await open()
            setAvailability.mockImplementation(async () => {
                emitPolicy({ nodeId: SELF, ...snapshot('6') })
                return { availability: 'paused' }
            })
            await click('Pause')
            await type(policyText('6', 'edited'))
            await click('Save')
            expect(lastBase()).toBe('6')
        })

        it('leaves a draft on its own base so the save goes through conflict review', async () => {
            get.mockResolvedValueOnce(savedDocument('5')).mockResolvedValue(savedDocument('6'))
            set.mockRejectedValue(new Error('policy revision is stale'))
            await open()
            await type('{"draft": true}')
            await click('Pause')
            await click('Save')
            expect(lastBase()).toBe('5')
            expect(container.textContent).toContain(CONFLICT)
            expect(editor().value).toBe('{"draft": true}')
        })

        it('shows the availability the broker pushes', async () => {
            get.mockResolvedValue(savedDocument('5'))
            await open()
            await act(async () => emitAvailability({ nodeId: SELF, availability: 'paused' }))
            expect(container.textContent).toContain('paused')
        })
    })

    describe('stale draft', () => {
        const draft = '{"servedModels": [], "residentModels": [], "draft": true}'

        async function staleSave(): Promise<void> {
            get.mockResolvedValueOnce(savedDocument('5')).mockResolvedValue(
                savedDocument('8', { activation: activation('pending', '5') })
            )
            set.mockRejectedValue(new Error('policy revision is stale'))
            await open()
            await type(draft)
            await click('Save')
        }

        it('reports a newer saved revision and keeps the draft', async () => {
            await staleSave()
            expect(container.textContent).toContain(CONFLICT)
            expect(editor().value).toBe(draft)
        })

        it('still shows the service message when the saved revision moved', async () => {
            await staleSave()
            expect(container.textContent).toContain('policy revision is stale')
        })

        it('does not resubmit the draft on its own', async () => {
            await staleSave()
            expect(set).toHaveBeenCalledTimes(1)
            expect(button('Save').disabled).toBe(true)
        })

        it('shows the latest saved policy beside the draft on review', async () => {
            await staleSave()
            await click('Review latest saved policy')
            const latest = container.querySelector<HTMLTextAreaElement>(
                'textarea[aria-label="Latest saved node policy JSON"]'
            )
            expect(latest?.value).toBe(policyText('8'))
            expect(latest?.readOnly).toBe(true)
            expect(editor().value).toBe(draft)
        })

        it('replaces the draft with the latest saved policy on discard', async () => {
            await staleSave()
            await click('Discard draft and reload')
            expect(editor().value).toBe(policyText('8'))
            expect(container.textContent).not.toContain(CONFLICT)
        })

        it('offers no way to save the stale draft against the newer revision', async () => {
            await staleSave()
            expect(() => button('Keep my draft')).toThrow()
            await click('Review latest saved policy')
            expect(button('Save').disabled).toBe(true)
            await type('{"tierPolicy": {"remoteAllowed": true}}')
            expect(button('Save').disabled).toBe(true)
            expect(set).toHaveBeenCalledTimes(1)
        })

        it('bases a new draft on the latest revision only after a reload', async () => {
            await staleSave()
            set.mockResolvedValue(snapshot('9', 'pending', 'edited'))
            await click('Discard draft and reload')
            await type(policyText('8', 'edited'))
            await click('Save')
            expect(set).toHaveBeenCalledTimes(2)
            expect(lastBase()).toBe('8')
        })

        it('shows the save error when the saved revision is the one it was based on', async () => {
            get.mockResolvedValue(savedDocument('5'))
            set.mockRejectedValue(new Error('tiers must be named'))
            await open()
            await type(draft)
            await click('Save')
            expect(container.textContent).toContain('tiers must be named')
            expect(container.textContent).not.toContain(CONFLICT)
            expect(button('Save').disabled).toBe(false)
        })

        it('says the outcome is unknown when the saved revision cannot be read back', async () => {
            get.mockResolvedValueOnce(savedDocument('5')).mockRejectedValue(
                new Error('node offline')
            )
            set.mockRejectedValue(new Error('tiers must be named'))
            await open()
            await type(draft)
            await click('Save')
            expect(container.textContent).toContain('tiers must be named')
            expect(container.textContent).toContain(
                'Could not confirm whether the save took effect'
            )
            expect(container.textContent).not.toContain('was not applied')
            expect(editor().value).toBe(draft)
        })
    })
})
