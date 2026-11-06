// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
    copyModelSettings,
    defaultModelCopyFields,
    defaultPortableLoadFields,
    deployModelPolicyPatch,
    modelPolicyPatch,
    profileDeployMatch,
    portableLoadPatch
} from '@/shared/utils/node-policy-model-copy'
import { modelRoutingState } from '@/shared/utils/node-policy-model-copy'
import { parseModelSettings } from '@/shared/utils/node-policy-model-settings'
import type { InspectModelResult, RequestedLoad } from '@/shared/types/model-config'
import type { ModelIdentity } from '@/shared/types/engines'

const source = { engine: 'llamacpp', model: 'source', profileKey: 'source' }

describe('profile deploy correspondence', () => {
    const main: ModelIdentity = { strength: 'verified', contentId: 'a'.repeat(64) }
    const draft = {
        role: 'draft',
        name: 'draft.gguf',
        sizeBytes: 10,
        sha256: 'b'.repeat(64)
    } satisfies NonNullable<ModelIdentity['files']>[number]

    it('verifies recorded main and companion hashes independently of file names', () => {
        expect(
            profileDeployMatch(
                { ...main, files: [draft] },
                {
                    ...main,
                    files: [{ ...draft, name: 'renamed.gguf' }]
                }
            )
        ).toEqual({ strength: 'verified', missingCompanions: [] })
    })

    it('keeps main-only correspondence uncertain without full same-role hashes', () => {
        expect(profileDeployMatch({ ...main, files: [draft] }, main)).toEqual({
            strength: 'companions-unverified',
            missingCompanions: ['draft.gguf']
        })
        expect(
            profileDeployMatch(
                { ...main, files: [{ ...draft, sha256: undefined }] },
                {
                    ...main,
                    files: [draft]
                }
            ).strength
        ).toBe('companions-unverified')
        expect(
            profileDeployMatch(
                { ...main, files: [draft] },
                {
                    ...main,
                    files: [{ ...draft, role: 'companion' }]
                }
            ).strength
        ).toBe('companions-unverified')
    })

    it('suggests weak metadata matches only with recorded equal sizes', () => {
        const weak: ModelIdentity = {
            strength: 'declared',
            repository: 'org/model',
            quantization: 'Q4',
            totalBytes: 10
        }
        expect(profileDeployMatch(weak, weak)).toEqual({
            strength: 'possible',
            missingCompanions: []
        })
        expect(profileDeployMatch(weak, { ...weak, totalBytes: undefined }).strength).toBe(
            'different'
        )
        expect(
            profileDeployMatch(
                { ...main, ...weak },
                { ...weak, ...main, contentId: 'c'.repeat(64) }
            ).strength
        ).toBe('different')
        expect(profileDeployMatch(undefined, weak).strength).toBe('different')
    })
})

describe('deploy policy construction', () => {
    function deploy(
        text: string,
        fields: string[] = ['temperature'],
        metadata: { displayName?: string; icon?: string } = {},
        roles: string[] = [],
        locked = false
    ) {
        return deployModelPolicyPatch(
            sourceText,
            source,
            text,
            target,
            { fields, mode: false },
            locked,
            metadata,
            roles
        )
    }

    it('constructs an absent typed target with source metadata and selected tuning only', () => {
        expect(deploy('{"profiles":[],"opaque":9007199254740993}')).toEqual({
            ok: true,
            value: { displayName: 'source', icon: '', generation: { temperature: 0.5 } }
        })
    })

    it('updates tuning while retaining existing metadata, locks and unrelated numeric tokens', () => {
        const text = targetPolicy('{"temperature":1,"nested":{"n":9007199254740995}}')
        expect(deploy(text)).toEqual({ ok: true, value: { generation: { temperature: 0.5 } } })
        expect(text).toContain('9007199254740995')
        expect(sourceText).toContain('"localOnly":true')
    })

    it('clears absent numeric leaves only when selected', () => {
        const text = targetPolicy('{"seed":42,"top_k":7}')
        expect(deploy(text, ['seed'])).toEqual({ ok: true, value: { generation: { seed: null } } })
        expect(deploy(text, [])).toEqual({ ok: true, value: {} })
    })

    it('copies mode only on explicit selection', () => {
        const text = targetPolicy('{"temperature":1}', ',"requestOptionsMode":"enforced"')
        expect(deploy(text, [])).toEqual({ ok: true, value: {} })
        expect(
            deployModelPolicyPatch(
                sourceText,
                source,
                text,
                target,
                { fields: [], mode: true },
                false,
                {},
                []
            )
        ).toEqual({ ok: true, value: { requestOptionsMode: 'defaults' } })
    })

    it('uses explicit metadata on updates and refuses invalid metadata', () => {
        expect(deploy(targetPolicy('{}'), [], { displayName: 'Chosen', icon: 'code' })).toEqual({
            ok: true,
            value: { displayName: 'Chosen', icon: 'code' }
        })
        expect(deploy(targetPolicy('{}'), [], { displayName: '' })).toMatchObject({
            ok: false,
            error: { code: 'profile' }
        })
    })

    it('adds selected declared memberships in target order without clearing others', () => {
        const text = targetPolicy('{}').replace(
            '"profiles":',
            '"roles":[{"name":"strong","profiles":[]},{"name":"weak","profiles":["target"]}],"profiles":'
        )
        const result = deploy(text, [], {}, ['weak', 'strong'])
        expect(result).toEqual({ ok: true, value: { roles: { strong: true, weak: true } } })
        if (!result.ok) throw new Error(result.error.message)
        expect(Object.keys(result.value.roles ?? {})).toEqual(['strong', 'weak'])
        expect(deploy(text, [], {}, ['strong'])).toEqual({
            ok: true,
            value: { roles: { strong: true } }
        })
        expect(deploy(text, [], {}, ['medium'])).toMatchObject({
            ok: false,
            error: { code: 'roles' }
        })
    })

    it('refuses malformed target role declarations and saved tuning locks', () => {
        expect(
            deploy('{"profiles":[],"roles":[{"name":"Custom","profiles":[]}]}', [], {}, ['Custom'])
        ).toMatchObject({ ok: false, error: { code: 'roles' } })
        expect(deploy(targetPolicy('{}'), ['temperature'], {}, [], true)).toMatchObject({
            ok: false,
            error: { code: 'locked' }
        })
    })

    it('excludes routing, qualification, source locks and opaque source options', () => {
        const decorated = sourceText.replace(
            '"profiles":',
            '"servedModels":[{"engine":"llamacpp","model":"source"}],"profiles":'
        )
        expect(
            deployModelPolicyPatch(
                decorated,
                source,
                '{"profiles":[]}',
                target,
                { fields: ['top_k'], mode: false },
                false,
                {},
                []
            )
        ).toEqual({
            ok: true,
            value: { displayName: 'source', icon: '', generation: { top_k: 20 } }
        })
    })
})

describe('keyed role membership', () => {
    it('resolves the selected pair to its policy key beyond the first preference', () => {
        const state = modelRoutingState(
            JSON.stringify({
                profiles: [
                    {
                        key: 'selected-profile',
                        displayName: 'Selected',
                        engine: 'llamacpp',
                        model: 'served-source'
                    }
                ],
                roles: [{ name: 'strong', profiles: ['other-profile', 'selected-profile'] }]
            }),
            { ...source, profileKey: 'selected-profile' }
        )
        expect(state.profileKey).toBe('selected-profile')
        expect(state.roles).toEqual([
            { name: 'strong', profiles: ['other-profile', 'selected-profile'] }
        ])
        expect(state.roles.some(role => role.profiles.includes(state.profileKey ?? ''))).toBe(true)
    })

    it('does not invent membership for missing profiles or empty declarations', () => {
        const state = modelRoutingState(
            JSON.stringify({
                profiles: [],
                roles: [{ name: 'weak', profiles: [] }]
            }),
            source
        )
        expect(state.profileKey).toBeUndefined()
        expect(state.roles).toEqual([{ name: 'weak', profiles: [] }])
    })
})
const target = { engine: 'llamacpp', model: 'target', profileKey: 'target' }
const sourceText =
    '{"profiles":[{"key":"source","displayName":"source","engine":"llamacpp","model":"source","requestOptions":{"temperature":0.5,"top_k":20,"nested":{"x":1}},"requestOptionsMode":"defaults","localOnly":true}]}'

function targetPolicy(options: string, mode = ''): string {
    return `{"opaque":9007199254740993,"profiles":[{"key":"target","displayName":"target","engine":"llamacpp","model":"target","requestOptions":${options}${mode},"qualification":{"precise":1.0000000000000000001},"localOnly":true}]}`
}

function copied(text: string, fields: string[], mode = false): string {
    const result = copyModelSettings(sourceText, source, text, target, { fields, mode }, false)
    if (!result.ok) throw new Error(result.error.message)
    return result.value
}

describe('selective model settings copy', () => {
    it('patches selected metadata and generation without conflating artifacts and served names', () => {
        const baseline =
            '{"opaque":9007199254740993,"profiles":[{"key":"small","displayName":"Small","engine":"llamacpp","model":"small","requestOptions":{"temperature":0.1}},{"key":"large","displayName":"Large","icon":"cube","engine":"llamacpp","model":"large","requestOptions":{"temperature":0.9,"opaque":1.00000000000000001}}]}'
        const draft = baseline
            .replace('"displayName":"Large"', '"displayName":"Large revised"')
            .replace('"icon":"cube"', '"icon":"code"')
            .replace('"temperature":0.9', '"temperature":0.7')
        expect(
            modelPolicyPatch(baseline, draft, {
                engine: 'llamacpp',
                model: 'artifact',
                profileKey: 'large'
            })
        ).toEqual({
            ok: true,
            value: { generation: { temperature: 0.7 }, displayName: 'Large revised', icon: 'code' }
        })
        expect(
            modelPolicyPatch(baseline, draft.replace('"temperature":0.1', '"temperature":0.2'), {
                engine: 'llamacpp',
                model: 'artifact',
                profileKey: 'large'
            })
        ).toMatchObject({ ok: false, error: { code: 'selection' } })
        const state = modelRoutingState(
            baseline.replace(
                '"profiles":',
                '"servedModels":[{"engine":"llamacpp","model":"large"}],"profiles":'
            ),
            { engine: 'llamacpp', model: 'artifact', profileKey: 'large' }
        )
        expect(state.profileKey).toBe('large')
        expect(state.served).toBe('included')
    })
    const inspection: InspectModelResult = {
        engine: 'llamacpp',
        modelId: 'source',
        status: 'resolved',
        estimate: { status: 'unknown' },
        options: [
            { field: 'fit', label: 'Native fit', kind: 'enum', supported: true, portable: true },
            {
                field: 'fitTarget',
                label: 'Fit target',
                kind: 'string',
                supported: true,
                portable: false
            },
            {
                field: 'fitCtx',
                label: 'Fit context',
                kind: 'integer',
                supported: true,
                portable: true
            },
            {
                field: 'extraOptions',
                label: 'Extra options',
                kind: 'options',
                supported: true,
                portable: true,
                nonPortableKeys: ['device', 'tensor-split', 'fork-local-device']
            }
        ]
    }
    it('copies portable native fit and options using descriptor keys without mutating source', () => {
        const requested: RequestedLoad = {
            fit: 'on',
            fitTarget: '512',
            fitCtx: 4096,
            extraOptions: {
                'spec-type': 'draft-mtp',
                device: 'CUDA0',
                'tensor-split': '1,2',
                'fork-local-device': 'local'
            }
        }
        const original = structuredClone(requested)
        const fields = defaultPortableLoadFields(inspection, requested)
        expect(fields).toEqual(['fit', 'fitCtx', 'extraOptions'])
        const patch = portableLoadPatch(requested, [...fields, 'fitTarget'], inspection)
        expect(patch).toEqual({
            set: { fit: 'on', fitCtx: 4096, extraOptions: { 'spec-type': 'draft-mtp' } },
            clear: []
        })
        expect(requested).toEqual(original)
        if (!patch.set.extraOptions) throw new Error('Portable extras missing')
        patch.set.extraOptions['spec-type'] = 'none'
        expect(requested.extraOptions).toEqual(original.extraOptions)
    })
    it('explicitly clears empty or entirely nonportable copied extra maps', () => {
        expect(portableLoadPatch({}, ['extraOptions'], inspection)).toEqual({
            set: {},
            clear: ['extraOptions']
        })
        expect(portableLoadPatch({ extraOptions: {} }, ['extraOptions'], inspection)).toEqual({
            set: {},
            clear: ['extraOptions']
        })
        expect(
            portableLoadPatch({ extraOptions: { device: 'CUDA0' } }, ['extraOptions'], inspection)
        ).toEqual({ set: {}, clear: ['extraOptions'] })
    })
    it('refuses extras when no source descriptor establishes portable keys', () => {
        expect(() =>
            portableLoadPatch({ extraOptions: { device: 'CUDA0', 'spec-type': 'none' } }, [
                'extraOptions'
            ])
        ).toThrow('Extra options require a supported portable source descriptor.')
    })
    it('selects portable CPU experts and detaches the copied setting', () => {
        const requested = { cpuExperts: { mode: 'layers', layers: 12 } } satisfies RequestedLoad
        expect(
            defaultPortableLoadFields(
                {
                    engine: 'llamacpp',
                    modelId: 'source',
                    status: 'resolved',
                    estimate: { status: 'unknown' },
                    options: [
                        {
                            field: 'cpuExperts',
                            label: 'CPU experts',
                            kind: 'enum',
                            supported: true,
                            portable: true
                        }
                    ]
                },
                requested
            )
        ).toEqual(['cpuExperts'])
        const patch = portableLoadPatch(requested, ['cpuExperts'])
        expect(patch).toEqual({ set: { cpuExperts: { mode: 'layers', layers: 12 } }, clear: [] })
        requested.cpuExperts.layers = 3
        expect(patch.set.cpuExperts?.layers).toBe(12)
        expect(portableLoadPatch({}, ['cpuExperts'])).toEqual({ set: {}, clear: ['cpuExperts'] })
        expect(portableLoadPatch({ cpuExperts: { mode: 'none' } }, ['cpuExperts'])).toEqual({
            set: { cpuExperts: { mode: 'none' } },
            clear: []
        })
    })
    it('defaults to configured numeric leaves', () => {
        expect(
            defaultModelCopyFields(sourceText, source, { engine: 'llamacpp', model: 'other' })
        ).toEqual({
            ok: true,
            value: ['temperature', 'top_k']
        })
    })

    it('sets and clears selected leaves while preserving target state and opaque tokens', () => {
        const text = targetPolicy(
            '{"temperature":1,"seed":42,"nested":{"large":9007199254740995}}',
            ',"requestOptionsMode":"enforced"'
        )
        const result = copied(text, ['temperature', 'seed'])
        const parsed = parseModelSettings(result)
        expect(parsed.ok).toBe(true)
        if (!parsed.ok) throw new Error(parsed.error.message)
        expect(parsed.value[0]).toMatchObject({
            options: { temperature: 0.5 },
            mode: 'enforced',
            localOnly: true
        })
        expect(parsed.value[0].options).not.toHaveProperty('seed')
        expect(result).toContain('"opaque":9007199254740993')
        expect(result).toContain('"large":9007199254740995')
        expect(result).toContain('"precise":1.0000000000000000001')
    })

    it('validates enforced mode after setting and clearing the complete patch', () => {
        const text = targetPolicy('{"seed":42}', ',"requestOptionsMode":"enforced"')
        const result = copied(text, ['seed', 'temperature'])
        expect(parseModelSettings(result)).toMatchObject({
            ok: true,
            value: [{ mode: 'enforced', options: { temperature: 0.5 } }]
        })
        expect(
            copyModelSettings(
                sourceText,
                source,
                text,
                target,
                { fields: ['seed'], mode: false },
                false
            )
        ).toMatchObject({ ok: false, error: { code: 'mode' } })
        expect(
            copyModelSettings(
                sourceText,
                source,
                text,
                target,
                { fields: ['seed'], mode: true },
                false
            )
        ).toMatchObject({ ok: true })
    })

    it('uses saved lock authority even when the selected patch is a no-op', () => {
        expect(
            copyModelSettings(
                sourceText,
                source,
                targetPolicy('{"temperature":0.5}'),
                target,
                { fields: ['temperature'], mode: false },
                true
            )
        ).toMatchObject({ ok: false, error: { code: 'locked' } })
    })

    it('requires an existing target profile and rejects unsupported source leaves', () => {
        expect(
            copyModelSettings(
                sourceText,
                source,
                '{"profiles":[]}',
                target,
                { fields: ['temperature'], mode: false },
                false
            )
        ).toMatchObject({ ok: false, error: { code: 'profile' } })
        expect(
            copyModelSettings(
                sourceText,
                source,
                targetPolicy('{}'),
                target,
                { fields: ['nested'], mode: false },
                false
            )
        ).toMatchObject({ ok: false, error: { code: 'option' } })
    })
})
