// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import path from 'path'
import { execFile } from 'child_process'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ensureNvpairOnPath } from '@/electron/nvpair-command'

vi.mock('child_process', () => ({ execFile: vi.fn<typeof execFile>() }))
vi.mock('electron', () => ({
    app: { isPackaged: true, getPath: () => 'C:\\PAIR' }
}))
vi.mock('@/shared/utils/platform', () => ({ currentPlatform: () => 'win32' }))
vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({ warn: vi.fn<(value: object) => void>() })
}))

const binDir = path.join('C:\\PAIR', 'bin')
const resourcesPath = Object.getOwnPropertyDescriptor(process, 'resourcesPath')

beforeEach(() => {
    Object.defineProperty(process, 'resourcesPath', {
        configurable: true,
        value: 'C:\\Resources'
    })
    vi.spyOn(fs, 'existsSync').mockReturnValue(true)
    vi.spyOn(fs, 'mkdirSync').mockReturnValue(undefined)
    vi.spyOn(fs, 'readFileSync').mockReturnValue('')
    vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {})
})

afterEach(() => {
    vi.restoreAllMocks()
    vi.clearAllMocks()
    if (resourcesPath) Object.defineProperty(process, 'resourcesPath', resourcesPath)
    else Reflect.deleteProperty(process, 'resourcesPath')
    vi.unstubAllEnvs()
})

describe('Windows command directory PATH update', () => {
    it.each([
        binDir,
        `C:\\Other; ${binDir.toUpperCase()}\\ ;C:\\Tools`
    ])('skips PowerShell when inherited PATH contains %s', inherited => {
        vi.stubEnv('PATH', inherited)
        ensureNvpairOnPath()
        expect(execFile).not.toHaveBeenCalled()
    })

    it.each([undefined, 'C:\\Other', `${binDir}-longer`])(
        'uses hidden PowerShell when inherited PATH is %s',
        inherited => {
            vi.stubEnv('PATH', inherited)
            ensureNvpairOnPath()
            expect(execFile).toHaveBeenCalledExactlyOnceWith(
                'powershell',
                expect.arrayContaining(['-NoProfile', '-NonInteractive', '-Command']),
                expect.objectContaining({ windowsHide: true }),
                expect.any(Function)
            )
        }
    )
})
