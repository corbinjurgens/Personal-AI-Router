// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { createOverviewWindow } from '@/electron/window'
import type { JsonObject, JsonValue } from '@/electron/service-bridge/json-rpc-client'
import { parseModelIdentities } from '@/electron/service-bridge/model-identity'
import { parseModelsResult } from '@/electron/service-bridge/model-download'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import {
    brokerNodesValue,
    projectEngineModels,
    sameNode
} from '@/electron/service-bridge/node-projection'
import { subscribePush } from '@/electron/service-bridge/push-bus'
import type { EngineStatePatch } from '@/shared/types/engine-api'
import type { EngineType, ModelIdentity } from '@/shared/types/engines'
import { engineTypeFromManagerName } from '@/shared/utils/engines'

vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => process.cwd() },
    BrowserWindow: { getAllWindows: () => [] }
}))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn<typeof createOverviewWindow>() }))

const MODEL = 'served-ctx131072'
function customEngine(): EngineType {
    const engine = engineTypeFromManagerName('llamacpp-custom')
    if (!engine) throw new Error('Invalid custom engine fixture')
    return engine
}
const IDENTITY = {
    repository: 'unsloth/LFM2.5-24B-A2B-GGUF',
    revision: 'a'.repeat(40),
    quantization: 'UD-Q4_K_XL',
    quantizationSource: 'hf-tag',
    generalName: 'LFM2.5 24B A2B',
    sizeLabel: '24B-A2B',
    architecture: 'lfm2',
    trainedContextTokens: 32768,
    configuredContextTokens: 131072,
    files: [{ role: 'model', name: 'model.gguf', sizeBytes: 42, sha256: 'b'.repeat(64) }],
    totalBytes: 42,
    contentId: 'c'.repeat(64),
    strength: 'verified'
} satisfies ModelIdentity

function identity(changes: JsonObject = {}): JsonObject {
    return { ...IDENTITY, ...changes }
}

describe('model identity parser', () => {
    it('preserves the complete Go identity shape', () => {
        expect(parseModelIdentities({ llamacpp: { [MODEL]: identity() } })).toEqual({
            llamacpp: { [MODEL]: IDENTITY }
        })
    })

    it('accepts the minimal declared identity and ignores unknown fields', () => {
        expect(
            parseModelIdentities({
                llamacpp: { [MODEL]: { strength: 'declared', futureMetadata: { arbitrary: true } } }
            })
        ).toEqual({ llamacpp: { [MODEL]: { strength: 'declared' } } })
    })

    const invalid: Array<{ name: string; value: JsonValue }> = [
        { name: 'non-object identity', value: 'metadata' },
        { name: 'null identity', value: null },
        { name: 'repository path', value: identity({ repository: 'owner/repo/extra' }) },
        { name: 'repository punctuation', value: identity({ repository: '-owner/repo' }) },
        { name: 'short revision', value: identity({ revision: 'a'.repeat(39) }) },
        { name: 'uppercase revision', value: identity({ revision: 'A'.repeat(40) }) },
        { name: 'nonhex revision', value: identity({ revision: 'g'.repeat(40) }) },
        { name: '63-hex content id', value: identity({ contentId: 'c'.repeat(63) }) },
        { name: 'uppercase content id', value: identity({ contentId: 'C'.repeat(64) }) },
        { name: 'nonhex content id', value: identity({ contentId: 'g'.repeat(64) }) },
        { name: 'unknown quantization source', value: identity({ quantizationSource: 'guessed' }) },
        { name: 'source without label', value: identity({ quantization: '' }) },
        { name: 'verified without content id', value: identity({ contentId: '' }) },
        { name: 'declared with content id', value: identity({ strength: 'declared' }) },
        { name: 'unknown strength', value: identity({ strength: 'strong' }) },
        { name: 'missing strength', value: { repository: 'owner/repo' } },
        { name: 'wrong text type', value: identity({ generalName: 42 }) },
        { name: 'negative total', value: identity({ totalBytes: -1 }) },
        { name: 'negative trained context', value: identity({ trainedContextTokens: -1 }) },
        { name: 'negative configured context', value: identity({ configuredContextTokens: -1 }) },
        { name: 'fractional total', value: identity({ totalBytes: 0.5 }) },
        { name: 'fractional trained context', value: identity({ trainedContextTokens: 0.5 }) },
        {
            name: 'fractional configured context',
            value: identity({ configuredContextTokens: 0.5 })
        },
        { name: 'unsafe total', value: identity({ totalBytes: Number.MAX_SAFE_INTEGER + 1 }) },
        {
            name: 'unsafe trained context',
            value: identity({ trainedContextTokens: Number.MAX_SAFE_INTEGER + 1 })
        },
        {
            name: 'unsafe configured context',
            value: identity({ configuredContextTokens: Number.MAX_SAFE_INTEGER + 1 })
        },
        { name: 'non-finite total', value: identity({ totalBytes: Infinity }) },
        { name: 'NaN context', value: identity({ configuredContextTokens: NaN }) },
        { name: 'wrong number type', value: identity({ totalBytes: '42' }) },
        { name: 'non-array files', value: identity({ files: {} }) },
        { name: 'non-object file', value: identity({ files: ['model.gguf'] }) },
        {
            name: 'forward-slash file path',
            value: identity({ files: [{ role: 'model', name: 'a/model.gguf', sizeBytes: 1 }] })
        },
        {
            name: 'backslash file path',
            value: identity({ files: [{ role: 'model', name: 'a\\model.gguf', sizeBytes: 1 }] })
        },
        {
            name: 'unknown file role',
            value: identity({ files: [{ role: 'other', name: 'model.gguf', sizeBytes: 1 }] })
        },
        {
            name: 'negative file size',
            value: identity({ files: [{ role: 'model', name: 'model.gguf', sizeBytes: -1 }] })
        },
        {
            name: 'fractional file size',
            value: identity({ files: [{ role: 'model', name: 'model.gguf', sizeBytes: 0.5 }] })
        },
        {
            name: 'unsafe file size',
            value: identity({
                files: [
                    { role: 'model', name: 'model.gguf', sizeBytes: Number.MAX_SAFE_INTEGER + 1 }
                ]
            })
        },
        {
            name: 'short file hash',
            value: identity({
                files: [{ role: 'model', name: 'model.gguf', sizeBytes: 1, sha256: 'b'.repeat(63) }]
            })
        },
        {
            name: 'uppercase file hash',
            value: identity({
                files: [{ role: 'model', name: 'model.gguf', sizeBytes: 1, sha256: 'B'.repeat(64) }]
            })
        },
        {
            name: 'nonhex file hash',
            value: identity({
                files: [{ role: 'model', name: 'model.gguf', sizeBytes: 1, sha256: 'g'.repeat(64) }]
            })
        },
        {
            name: 'too many files',
            value: identity({
                files: Array.from({ length: 65 }, () => ({
                    role: 'companion',
                    name: 'file',
                    sizeBytes: 0
                }))
            })
        }
    ]

    it.each(invalid)('drops $name without dropping a valid sibling', ({ value }) => {
        expect(
            parseModelIdentities({
                llamacpp: { [MODEL]: identity(), invalid: value },
                malformedEngine: ['metadata']
            })
        ).toEqual({ llamacpp: { [MODEL]: IDENTITY } })
    })

    const invalidText = [
        { name: 'ASCII byte overflow', value: 'x'.repeat(257) },
        { name: 'UTF-8 byte overflow', value: 'é'.repeat(129) },
        { name: 'C0 control', value: 'a\nb' },
        { name: 'DEL control', value: 'a\u007fb' },
        { name: 'C1 control', value: 'a\u0085b' },
        { name: 'unpaired high surrogate', value: '\ud800' },
        { name: 'unpaired low surrogate', value: '\udfff' }
    ]

    it.each(invalidText)('rejects $name in identity and file text', ({ value }) => {
        expect(
            parseModelIdentities({ llamacpp: { [MODEL]: identity({ generalName: value }) } })
        ).toEqual({ llamacpp: {} })
        expect(
            parseModelIdentities({
                llamacpp: {
                    [MODEL]: identity({ files: [{ role: 'companion', name: value, sizeBytes: 0 }] })
                }
            })
        ).toEqual({ llamacpp: {} })
    })

    it('accepts the Go text, file and safe-integer boundaries', () => {
        const bounded = {
            generalName: 'é'.repeat(128),
            architecture: '\u{1d11e}'.repeat(64),
            totalBytes: Number.MAX_SAFE_INTEGER,
            trainedContextTokens: Number.MAX_SAFE_INTEGER,
            configuredContextTokens: Number.MAX_SAFE_INTEGER,
            strength: 'declared',
            files: Array.from({ length: 64 }, () => ({
                role: 'companion',
                name: 'x'.repeat(256),
                sizeBytes: Number.MAX_SAFE_INTEGER
            }))
        }
        expect(parseModelIdentities({ llamacpp: { [MODEL]: bounded } })).toEqual({
            llamacpp: { [MODEL]: bounded }
        })
    })

    it('accepts declared quantization without a source and Go zero-value file fields', () => {
        expect(
            parseModelIdentities({
                llamacpp: {
                    [MODEL]: {
                        strength: 'declared',
                        quantization: 'Q4_K_M',
                        files: [{ role: 'draft' }]
                    }
                }
            })
        ).toEqual({
            llamacpp: {
                [MODEL]: {
                    strength: 'declared',
                    quantization: 'Q4_K_M',
                    files: [{ role: 'draft', name: '', sizeBytes: 0 }]
                }
            }
        })
    })

    it.each(['hf-tag', 'file-name', 'gguf-file-type'])(
        'accepts the Go quantization source %s',
        source => {
            expect(
                parseModelIdentities({
                    llamacpp: { [MODEL]: identity({ quantizationSource: source }) }
                })
            ).toEqual({
                llamacpp: { [MODEL]: { ...IDENTITY, quantizationSource: source } }
            })
        }
    )

    it('normalizes optional nulls and zero values like Go omitempty fields', () => {
        expect(
            parseModelIdentities({
                llamacpp: {
                    [MODEL]: {
                        strength: 'declared',
                        repository: null,
                        totalBytes: 0,
                        trainedContextTokens: null,
                        configuredContextTokens: 0,
                        files: null
                    }
                }
            })
        ).toEqual({ llamacpp: { [MODEL]: { strength: 'declared' } } })
    })

    it('preserves engine and model keys that shadow object properties', () => {
        const wire = Object.fromEntries([
            ['__proto__', Object.fromEntries([['constructor', identity()]])]
        ])
        const parsed = parseModelIdentities(wire)
        expect(parsed && Object.hasOwn(parsed, '__proto__')).toBe(true)
        expect(parsed).toEqual(wire)
    })

    const nonObjects: Array<{ name: string; value: JsonValue | undefined }> = [
        { name: 'string', value: 'metadata' },
        { name: 'null', value: null },
        { name: 'array', value: [] },
        { name: 'number', value: 42 },
        { name: 'boolean', value: true },
        { name: 'absent', value: undefined }
    ]
    it.each(nonObjects)('returns undefined for a non-object block: $name', ({ value }) => {
        expect(parseModelIdentities(value)).toBeUndefined()
    })

    it('keeps valid inventory when optional identity metadata is malformed', () => {
        const result = parseModelsResult({
            models: [MODEL],
            modelsByEngine: { llamacpp: [MODEL] },
            modelIdentities: 'malformed'
        })
        expect(result.modelsByEngine).toEqual({ llamacpp: [MODEL] })
        expect(result.modelIdentities).toBeUndefined()
    })
})

describe('model row identity projection', () => {
    it('attaches metadata only to the matching installed model', () => {
        const projected = projectEngineModels(
            'llama-cpp',
            'node',
            [MODEL, 'without-metadata'],
            new Set(['loaded-only']),
            undefined,
            undefined,
            true,
            { [MODEL]: IDENTITY, 'loaded-only': IDENTITY, unlisted: IDENTITY }
        )
        expect(projected.models.map(row => ({ name: row.name, identity: row.identity }))).toEqual([
            { name: MODEL, identity: IDENTITY },
            { name: 'without-metadata', identity: undefined },
            { name: 'loaded-only', identity: undefined }
        ])
        expect(projected.models.every(row => !row.downloaded)).toBe(true)
    })

    it('compares identity-only node changes independently of inventory and timestamps', () => {
        const wire = { hostUuid: 'peer', lastSeen: 1, modelsByEngine: { llamacpp: [MODEL] } }
        const left = brokerNodesValue([
            { ...wire, modelIdentities: { llamacpp: { [MODEL]: identity() } } }
        ])[0]
        const right = brokerNodesValue([
            {
                ...wire,
                modelIdentities: {
                    llamacpp: { [MODEL]: identity({ configuredContextTokens: 8192 }) }
                }
            }
        ])[0]
        expect(sameNode(left, right)).toBe(false)
        expect(
            sameNode(
                left,
                brokerNodesValue([
                    { ...wire, modelIdentities: { llamacpp: { [MODEL]: identity() } } }
                ])[0]
            )
        ).toBe(true)
    })
})

const state = getModularBridgeState()
let caseId = 0
let localId: string
let remoteId: string
let sourceId: string
const subscriptions: Array<() => void> = []

function downloadSnapshot(revision = '1'): JsonObject {
    return { version: 1, sourceId, revision, recoveryRequired: false, entries: [] }
}

function localResult(changes: JsonObject = {}) {
    return parseModelsResult({
        models: [MODEL, 'without-metadata', 'custom-model'],
        modelsByEngine: {
            llamacpp: [MODEL, 'without-metadata'],
            'llamacpp-custom': ['custom-model']
        },
        loadedByEngine: { llamacpp: [MODEL] },
        modelIdentities: {
            llamacpp: { [MODEL]: identity() },
            'llamacpp-custom': { 'custom-model': identity({ generalName: 'Custom model' }) }
        },
        downloadState: downloadSnapshot(),
        ...changes
    })
}

function remoteUpdate(changes: JsonObject = {}): void {
    state.handleNotification({
        source: 'broker',
        method: 'discovery:nodes-changed',
        params: {
            nodes: [
                {
                    hostUuid: remoteId,
                    name: remoteId,
                    ipAddress: '192.0.2.41',
                    port: 14318,
                    lastSeen: 1,
                    models: [MODEL, 'without-metadata', 'custom-model'],
                    modelsByEngine: {
                        llamacpp: [MODEL, 'without-metadata'],
                        'llamacpp-custom': ['custom-model']
                    },
                    loadedByEngine: { llamacpp: [MODEL] },
                    modelIdentities: {
                        llamacpp: { [MODEL]: identity() },
                        'llamacpp-custom': {
                            'custom-model': identity({ generalName: 'Custom model' })
                        }
                    },
                    downloadState: downloadSnapshot(),
                    downloadStateCurrent: true,
                    ...changes
                }
            ]
        }
    })
}

function rows(nodeId: string, engine: EngineType = 'llama-cpp') {
    return (
        state
            .getEngineInitialState()
            .models.find(row => row.nodeId === nodeId && row.engineType === engine)?.models ?? []
    )
}

function modelPushes(nodeId: string): EngineStatePatch[] {
    const pushes: EngineStatePatch[] = []
    subscriptions.push(
        subscribePush(event => {
            if (
                event.channel === 'engines:state-changed' &&
                event.payload.nodeId === nodeId &&
                event.payload.models
            )
                pushes.push(event.payload)
        })
    )
    return pushes
}

describe('desktop inventory identity observations', () => {
    beforeEach(() => {
        caseId++
        localId = `identity-local-${caseId}`
        remoteId = `identity-remote-${caseId}`
        sourceId = caseId.toString(16).padStart(32, '0')
        state.setSelfId(localId)
        state.applyEngineManagerStatus({
            engine: 'llamacpp',
            installed: true,
            running: true,
            healthy: true
        })
        state.applyLocalModelsResult(
            localResult({
                models: [],
                modelsByEngine: { llamacpp: [], 'llamacpp-custom': [] },
                modelIdentities: {},
                loadedByEngine: {}
            })
        )
    })

    afterEach(() => {
        for (const unsubscribe of subscriptions.splice(0)) unsubscribe()
        vi.restoreAllMocks()
    })

    it('projects local canonical identities for built-in and descriptor engines', () => {
        state.applyLocalModelsResult(localResult())
        expect(
            rows(localId).map(row => ({
                name: row.name,
                identity: row.identity,
                status: row.status
            }))
        ).toEqual([
            { name: MODEL, identity: IDENTITY, status: 'loaded' },
            { name: 'without-metadata', identity: undefined, status: 'idle' }
        ])
        expect(rows(localId, customEngine())[0]?.identity).toEqual({
            ...IDENTITY,
            generalName: 'Custom model'
        })
    })

    it('pushes local identity-only changes through the existing full notification path', () => {
        state.applyLocalModelsResult(localResult())
        const pushes = modelPushes(localId)
        expect(
            state.applyLocalModelsNotification(
                localResult({
                    modelIdentities: {
                        llamacpp: { [MODEL]: identity({ configuredContextTokens: 8192 }) }
                    }
                })
            )
        ).toBe(false)
        const expected = { ...IDENTITY, configuredContextTokens: 8192 }
        expect(rows(localId)[0]?.identity).toEqual(expected)
        expect(
            pushes.find(patch => patch.engineType === 'llama-cpp')?.models?.models[0]?.identity
        ).toEqual(expected)
    })

    it('clears local identities when a fresh listed inventory no longer supplies them', () => {
        state.applyLocalModelsResult(localResult())
        state.applyLocalModelsResult(localResult({ modelIdentities: null }))
        expect(rows(localId).map(row => row.name)).toEqual([MODEL, 'without-metadata'])
        expect(rows(localId).every(row => row.identity === undefined)).toBe(true)
    })

    it('keeps a valid local identity beside malformed metadata without losing either model', () => {
        state.applyLocalModelsResult(
            localResult({
                modelIdentities: {
                    llamacpp: {
                        [MODEL]: identity(),
                        'without-metadata': identity({
                            files: [{ role: 'model', name: 'directory/model.gguf', sizeBytes: 1 }]
                        })
                    }
                }
            })
        )
        expect(rows(localId).map(row => ({ name: row.name, identity: row.identity }))).toEqual([
            { name: MODEL, identity: IDENTITY },
            { name: 'without-metadata', identity: undefined }
        ])
        expect(rows(localId).map(row => row.downloaded)).toEqual([true, true])
    })

    it('requests a canonical local refresh when discovery changes only identity metadata', () => {
        remoteUpdate({ hostUuid: localId, name: localId })
        const refresh = vi.fn<Parameters<typeof state.setLocalDiscoveryModelRefresher>[0]>()
        state.setLocalDiscoveryModelRefresher(refresh)
        try {
            remoteUpdate({
                hostUuid: localId,
                name: localId,
                modelIdentities: {
                    llamacpp: { [MODEL]: identity({ configuredContextTokens: 8192 }) },
                    'llamacpp-custom': {
                        'custom-model': identity({ generalName: 'Custom model' })
                    }
                }
            })
            expect(refresh).toHaveBeenCalledExactlyOnceWith('llama-cpp')
        } finally {
            state.setLocalDiscoveryModelRefresher(() => {})
        }
    })

    it('keeps local metadata with its last-good names when an engine query is absent', () => {
        state.applyLocalModelsResult(localResult())
        state.applyLocalModelsResult(
            localResult({ models: [], modelsByEngine: {}, modelIdentities: {}, loadedByEngine: {} })
        )
        expect(rows(localId)[0]).toMatchObject({
            name: MODEL,
            identity: IDENTITY,
            downloaded: false
        })
    })

    it('projects remote identities without cross-attributing engines or malformed siblings', () => {
        remoteUpdate({
            modelIdentities: {
                llamacpp: {
                    [MODEL]: identity(),
                    'without-metadata': identity({ contentId: 'c'.repeat(63) }),
                    unlisted: identity()
                },
                'llamacpp-custom': { 'custom-model': identity({ generalName: 'Custom model' }) }
            }
        })
        expect(rows(remoteId).map(row => ({ name: row.name, identity: row.identity }))).toEqual([
            { name: MODEL, identity: IDENTITY },
            { name: 'without-metadata', identity: undefined }
        ])
        expect(rows(remoteId, customEngine())[0]?.identity).toEqual({
            ...IDENTITY,
            generalName: 'Custom model'
        })
    })

    it('publishes remote identity-only updates even when lastSeen and names are unchanged', () => {
        remoteUpdate()
        const pushes = modelPushes(remoteId)
        remoteUpdate({
            modelIdentities: {
                llamacpp: {
                    [MODEL]: identity({ quantization: 'Q8_0', quantizationSource: 'file-name' })
                }
            }
        })
        const expected = { ...IDENTITY, quantization: 'Q8_0', quantizationSource: 'file-name' }
        expect(rows(remoteId)[0]?.identity).toEqual(expected)
        expect(
            pushes.find(patch => patch.engineType === 'llama-cpp')?.models?.models[0]?.identity
        ).toEqual(expected)
    })

    it('preserves broker identities across proxy presence updates', () => {
        remoteUpdate()
        state.handleNotification({
            source: 'llamacpp-proxy',
            method: 'node/updated',
            params: {
                id: remoteId,
                host: remoteId,
                ip: '192.0.2.41',
                port: 8080
            }
        })
        expect(rows(remoteId)[0]?.identity).toEqual(IDENTITY)
        expect(rows(remoteId, customEngine())[0]?.identity).toEqual({
            ...IDENTITY,
            generalName: 'Custom model'
        })
    })

    it('retains remote identities with their matching inventory when a stale observation is rejected', () => {
        remoteUpdate({ downloadState: downloadSnapshot('2') })
        remoteUpdate({
            models: ['stale'],
            modelsByEngine: { llamacpp: ['stale'] },
            modelIdentities: { llamacpp: { stale: identity({ configuredContextTokens: 8192 }) } }
        })
        expect(rows(remoteId).map(row => ({ name: row.name, identity: row.identity }))).toEqual([
            { name: MODEL, identity: IDENTITY },
            { name: 'without-metadata', identity: undefined }
        ])
    })

    it('keeps remote inventory when the optional identity block is malformed', () => {
        remoteUpdate({ modelIdentities: 'malformed' })
        expect(rows(remoteId).map(row => row.name)).toEqual([MODEL, 'without-metadata'])
        expect(rows(remoteId).every(row => row.identity === undefined)).toBe(true)
        expect(rows(remoteId).map(row => row.downloaded)).toEqual([true, true])
    })

    it('does not retain identities for an unlisted model that is still loaded', () => {
        remoteUpdate()
        remoteUpdate({ models: [], modelsByEngine: { llamacpp: [] } })
        expect(rows(remoteId)).toMatchObject([{ name: MODEL, status: 'loaded' }])
        expect(rows(remoteId)[0]?.identity).toBeUndefined()
    })
})
