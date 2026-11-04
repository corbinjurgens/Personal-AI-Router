// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import path from 'path'
import { afterAll, describe, expect, it, vi } from 'vitest'
import { setPaths } from '@/electron/path'

// path.ts resolves its app-data root when it is imported, so point every
// platform's base at a scratch folder before the import runs.
const scratch = vi.hoisted(() => {
    const nodeFs = process.getBuiltinModule('fs')
    const nodeOs = process.getBuiltinModule('os')
    const nodePath = process.getBuiltinModule('path')
    const root = nodeFs.mkdtempSync(nodePath.join(nodeOs.tmpdir(), 'pair-paths-test-'))
    process.env.LOCALAPPDATA = root
    process.env.XDG_CONFIG_HOME = root
    return { root, setPath: vi.fn<(name: string, value: string) => void>() }
})

vi.mock('electron', () => ({
    app: {
        getPath: () => scratch.root,
        setPath: scratch.setPath,
        setAppLogsPath: vi.fn(),
        commandLine: { appendSwitch: vi.fn() }
    }
}))

afterAll(() => {
    fs.rmSync(scratch.root, { recursive: true, force: true })
})

describe('setPaths', () => {
    it('creates the redirected temp folder so consumers can resolve it', async () => {
        await setPaths()

        const call = scratch.setPath.mock.calls.find(([name]) => name === 'temp')
        expect(call).toBeDefined()
        const tempDir = call?.[1] ?? ''
        expect(path.relative(scratch.root, tempDir).startsWith('..')).toBe(false)
        expect(fs.statSync(fs.realpathSync(tempDir)).isDirectory()).toBe(true)
    })
})
