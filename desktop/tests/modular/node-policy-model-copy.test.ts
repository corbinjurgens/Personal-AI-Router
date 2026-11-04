// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
    copyModelSettings,
    defaultModelCopyFields,
    defaultPortableLoadFields,
    portableLoadPatch
} from '@/shared/utils/node-policy-model-copy'
import { parseModelSettings } from '@/shared/utils/node-policy-model-settings'
import type { RequestedLoad } from '@/shared/types/model-config'

const source = { engine: 'llamacpp', model: 'source' }
const target = { engine: 'llamacpp', model: 'target' }
const sourceText =
    '{"profiles":[{"engine":"llamacpp","model":"source","requestOptions":{"temperature":0.5,"top_k":20,"nested":{"x":1}},"requestOptionsMode":"defaults","localOnly":true}]}'

function targetPolicy(options: string, mode = ''): string {
    return `{"opaque":9007199254740993,"profiles":[{"engine":"llamacpp","model":"target","requestOptions":${options}${mode},"qualification":{"precise":1.0000000000000000001},"localOnly":true}]}`
}

function copied(text: string, fields: string[], mode = false): string {
    const result = copyModelSettings(sourceText, source, text, target, { fields, mode }, false)
    if (!result.ok) throw new Error(result.error.message)
    return result.value
}

describe('selective model settings copy', () => {
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
