// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
    compactJsonSource,
    jsonFieldSource,
    prettyJsonSource,
    replaceJsonFieldSource
} from '@/shared/utils/json-source'

describe('validated JSON source fragments', () => {
    it('replaces only the last selected field and keeps duplicates for service validation', () => {
        const source = '{"roles":{},"opaque":9007199254740993,"roles":{"weak":null}}'
        expect(replaceJsonFieldSource(source, 'roles', '{"strong":null}')).toBe(
            '{"roles":{},"opaque":9007199254740993,"roles":{"strong":null}}'
        )
    })
    it('inserts a missing field without changing existing value tokens or surrounding whitespace', () => {
        expect(
            replaceJsonFieldSource(
                ' { "opaque":0.1000000000000000055511151231257827 } ',
                'roles',
                '{}'
            )
        ).toBe(' { "opaque":0.1000000000000000055511151231257827 ,"roles":{}} ')
        expect(replaceJsonFieldSource(' { } ', 'roles', '{}')).toBe(' { "roles":{}} ')
    })
    it.each(['null', '[]', '{oops'])(
        'refuses a malformed policy %s instead of replacing it',
        text => {
            expect(() => replaceJsonFieldSource(text, 'roles', '{}')).toThrow()
        }
    )
    it('locates escaped field names after skipping nested arrays and escaped strings', () => {
        const policy = String.raw`{"revision":"1","values":[9007199254740993,{"decimal":0.1000000000000000055511151231257827}],"text":"\\\"}, ]"}`
        const frame = String.raw` {"ignored":[{"result":{"policy":false}},"}\\\""],"res\u0075lt":{"ignored":null,"pol\u0069cy":${policy}}} `
        expect(() => JSON.parse(frame)).not.toThrow()
        expect(jsonFieldSource(frame, ['result', 'policy'])).toBe(policy)
    })

    it('selects the last duplicate field at every level like JSON.parse', () => {
        const frame = '{"result":{"policy":0},"result":{"policy":false,"policy":{"revision":"2"}}}'
        expect(JSON.parse(frame)).toEqual({ result: { policy: { revision: '2' } } })
        expect(jsonFieldSource(frame, ['result', 'policy'])).toBe('{"revision":"2"}')
    })

    it.each([
        ['missing parent', '{}'],
        ['missing policy', '{"result":{}}'],
        ['array parent', '{"result":[]}'],
        ['primitive parent', '{"result":42}']
    ])('returns no fragment for a %s', (_name, frame) => {
        expect(() => JSON.parse(frame)).not.toThrow()
        expect(jsonFieldSource(frame, ['result', 'policy'])).toBeUndefined()
    })

    it('compacts whitespace without changing any number or string token', () => {
        const text = String.raw`{
            "numbers": [ -0, 1.00, 1E+02, 1e400, 1e-400, 9007199254740993 ],
            "text": " space \t\r\n\"\\\u0061 "
        }`
        const compact = String.raw`{"numbers":[-0,1.00,1E+02,1e400,1e-400,9007199254740993],"text":" space \t\r\n\"\\\u0061 "}`
        expect(() => JSON.parse(text)).not.toThrow()
        expect(compactJsonSource(text)).toBe(compact)
        expect(JSON.parse(compact)).toEqual(JSON.parse(text))
    })

    it('pretty prints nested and empty containers without re-encoding any values', () => {
        const compact = String.raw`{"numbers":[9007199254740993,0.1000000000000000055511151231257827],"empty":[{},[]],"text":"\u0061 \"[]{}\\\n"}`
        const pretty = String.raw`{
  "numbers": [
    9007199254740993,
    0.1000000000000000055511151231257827
  ],
  "empty": [
    {},
    []
  ],
  "text": "\u0061 \"[]{}\\\n"
}`
        expect(() => JSON.parse(compact)).not.toThrow()
        expect(prettyJsonSource(compact)).toBe(pretty)
    })
})
