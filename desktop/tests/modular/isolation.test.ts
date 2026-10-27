// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import os from 'os'
import path from 'path'
import { afterAll, afterEach, describe, expect, it, vi } from 'vitest'
import { assertIsolated } from '../fixtures/isolation'

const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'pair-isolation-check-'))
const root = path.join(fixture, 'tmp')
const originalUserData = process.env.PAIR_USER_DATA

afterEach(() => {
    process.env.PAIR_USER_DATA = originalUserData
    vi.restoreAllMocks()
})

afterAll(() => fs.rmSync(fixture, { recursive: true, force: true }))

describe('assertIsolated', () => {
    it('accepts the temporary root and descendants', () => {
        vi.spyOn(os, 'tmpdir').mockReturnValue(root)
        for (const candidate of [
            root,
            path.join(root, 'child'),
            path.join(root, 'a', '..', 'child')
        ]) {
            process.env.PAIR_USER_DATA = candidate
            expect(() => assertIsolated()).not.toThrow()
        }
    })

    it('rejects a same-prefix sibling', () => {
        vi.spyOn(os, 'tmpdir').mockReturnValue(root)
        process.env.PAIR_USER_DATA = `${root}-sibling`
        expect(() => assertIsolated()).toThrow('[isolation]')
    })

    it('rejects paths escaping through dot segments', () => {
        vi.spyOn(os, 'tmpdir').mockReturnValue(root)
        process.env.PAIR_USER_DATA = `${root}${path.sep}..${path.sep}outside`
        expect(() => assertIsolated()).toThrow('[isolation]')
    })
})
