// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import os from 'os'
import path from 'path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { app, shell } from 'electron'
import { currentPlatform } from '@/shared/utils/platform'
import { ensureWindowsShortcut, handleShortcutEntry } from '@/electron/windows-shortcut'

const native = vi.hoisted(() => ({ packaged: true }))
vi.mock('electron', () => ({
    app: {
        get isPackaged() {
            return native.packaged
        },
        getPath: vi.fn<typeof app.getPath>()
    },
    shell: {
        readShortcutLink: vi.fn<typeof shell.readShortcutLink>(),
        writeShortcutLink: vi.fn<typeof shell.writeShortcutLink>()
    }
}))
vi.mock('@/shared/utils/platform', () => ({ currentPlatform: vi.fn<typeof currentPlatform>() }))
vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({ info: vi.fn(), warn: vi.fn() })
}))

const target = 'C:\\Candidate with spaces\\PAIR.exe'
const link = 'C:\\User data\\Microsoft\\Windows\\Start Menu\\Programs\\NVIDIA PAIR.lnk'
const expected: Electron.ShortcutDetails = {
    target,
    cwd: 'C:\\Candidate with spaces',
    icon: target,
    iconIndex: 0,
    description: 'Launch NVIDIA PAIR',
    appUserModelId: 'com.nvidia.nvpair',
    args: ''
}
let temp: string
let existing: Electron.ShortcutDetails | null

beforeEach(() => {
    temp = fs.mkdtempSync(path.join(os.tmpdir(), 'pair-shortcut-test-'))
    const fixture = path.join(temp, 'fixture')
    fs.writeFileSync(fixture, '')
    const fileStat = fs.statSync(fixture)
    const missing = path.join(temp, 'missing')
    const realLstat = fs.lstatSync.bind(fs)
    existing = null
    native.packaged = true
    vi.mocked(currentPlatform).mockReturnValue('win32')
    vi.mocked(app.getPath).mockImplementation(name =>
        name === 'appData' ? 'C:\\User data' : name === 'temp' ? 'C:\\Temporary' : target
    )
    vi.spyOn(fs, 'realpathSync').mockImplementation(value =>
        value === 'C:\\Temporary' ? 'C:\\Temporary' : target
    )
    vi.spyOn(fs, 'existsSync').mockReturnValue(false)
    vi.spyOn(fs, 'statSync').mockReturnValue(fileStat)
    vi.spyOn(fs, 'lstatSync').mockImplementation(() => (existing ? fileStat : realLstat(missing)))
    vi.spyOn(fs, 'mkdirSync').mockImplementation(() => undefined)
    vi.spyOn(fs, 'unlinkSync').mockImplementation(() => {
        existing = null
    })
    vi.mocked(shell.readShortcutLink).mockImplementation(() => {
        if (!existing) throw new Error('Missing fake link')
        return existing
    })
    vi.mocked(shell.writeShortcutLink).mockImplementation(() => {
        existing = { ...expected }
        return true
    })
    vi.spyOn(console, 'log').mockImplementation(() => {})
    vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
    vi.restoreAllMocks()
    vi.clearAllMocks()
    fs.rmSync(temp, { recursive: true, force: true })
})

describe('Windows Start Menu shortcut', () => {
    it('creates and reads back a link with the executable and app identity', async () => {
        expect(await handleShortcutEntry(['--pair-shortcut=create'])).toBe(0)
        expect(shell.writeShortcutLink).toHaveBeenCalledWith(link, 'create', expected)
        expect(shell.readShortcutLink).toHaveBeenCalledWith(link)
        expect(await handleShortcutEntry(['--pair-shortcut=status'])).toBe(0)
    })

    it('leaves a current link unchanged on repeat and after a same-path update', async () => {
        existing = { ...expected }
        ensureWindowsShortcut()
        expect(await handleShortcutEntry(['--pair-shortcut=create'])).toBe(0)
        expect(shell.writeShortcutLink).not.toHaveBeenCalled()
    })

    it('repairs an owned stale target', async () => {
        existing = { ...expected, target: 'C:\\Old build\\PAIR.exe' }
        expect(await handleShortcutEntry(['--pair-shortcut=create'])).toBe(0)
        expect(shell.writeShortcutLink).toHaveBeenCalledWith(link, 'create', expected)
    })

    it('refuses to overwrite or remove an unrelated collision', async () => {
        existing = { ...expected, appUserModelId: 'another.application' }
        expect(await handleShortcutEntry(['--pair-shortcut=create'])).toBe(1)
        expect(await handleShortcutEntry(['--pair-shortcut=remove'])).toBe(1)
        expect(shell.writeShortcutLink).not.toHaveBeenCalled()
        expect(fs.unlinkSync).not.toHaveBeenCalled()
    })

    it('removes an owned link and refuses a missing link', async () => {
        existing = { ...expected }
        expect(await handleShortcutEntry(['--pair-shortcut=remove'])).toBe(0)
        expect(fs.unlinkSync).toHaveBeenCalledWith(link)
        expect(await handleShortcutEntry(['--pair-shortcut=remove'])).toBe(1)
    })

    it('does nothing on non-Windows startup and rejects command mode', async () => {
        vi.mocked(currentPlatform).mockReturnValue('linux')
        ensureWindowsShortcut()
        expect(await handleShortcutEntry(['--pair-shortcut=create'])).toBe(69)
        expect(shell.writeShortcutLink).not.toHaveBeenCalled()
        expect(app.getPath).not.toHaveBeenCalled()
    })

    it('refuses unpackaged Windows commands without touching shortcut files', async () => {
        native.packaged = false
        expect(await handleShortcutEntry(['--pair-shortcut=create'])).toBe(1)
        ensureWindowsShortcut()
        expect(app.getPath).not.toHaveBeenCalled()
        expect(shell.writeShortcutLink).not.toHaveBeenCalled()
        expect(fs.unlinkSync).not.toHaveBeenCalled()
    })

    it('refuses a non-PAIR executable', async () => {
        vi.mocked(fs.realpathSync).mockReturnValue('C:\\Development\\electron.exe')
        expect(await handleShortcutEntry(['--pair-shortcut=create'])).toBe(1)
        expect(shell.writeShortcutLink).not.toHaveBeenCalled()
    })

    it('refuses changing checkout output before writing or repairing a link', async () => {
        existing = { ...expected }
        vi.mocked(fs.realpathSync).mockReturnValueOnce(
            'C:\\Checkout\\desktop\\release\\version\\windows\\PAIR.exe'
        )
        vi.mocked(fs.existsSync).mockImplementation(value => value === 'C:\\Checkout\\.git')
        expect(await handleShortcutEntry(['--pair-shortcut=create'])).toBe(1)
        expect(shell.writeShortcutLink).not.toHaveBeenCalled()
        expect(fs.unlinkSync).not.toHaveBeenCalled()
    })

    it('refuses the packaged staging layout', async () => {
        vi.mocked(fs.realpathSync).mockReturnValueOnce('C:\\Build\\win-unpacked\\PAIR.exe')
        expect(await handleShortcutEntry(['--pair-shortcut=create'])).toBe(1)
        expect(shell.writeShortcutLink).not.toHaveBeenCalled()
    })

    it('refuses a temporary extracted build', async () => {
        vi.mocked(fs.realpathSync).mockReturnValueOnce('C:\\Temporary\\extracted\\PAIR.exe')
        expect(await handleShortcutEntry(['--pair-shortcut=create'])).toBe(1)
        expect(shell.writeShortcutLink).not.toHaveBeenCalled()
    })

    it('reports write and readback failures without failing startup', async () => {
        vi.mocked(shell.writeShortcutLink).mockReturnValue(false)
        expect(await handleShortcutEntry(['--pair-shortcut=create'])).toBe(1)
        expect(() => ensureWindowsShortcut()).not.toThrow()
        vi.mocked(shell.writeShortcutLink).mockReturnValue(true)
        expect(await handleShortcutEntry(['--pair-shortcut=create'])).toBe(1)
    })

    it('rejects invalid flags before accessing the filesystem', async () => {
        expect(await handleShortcutEntry(['--pair-shortcut=unknown'])).toBe(64)
        expect(
            await handleShortcutEntry(['--pair-shortcut=create', '--pair-shortcut=remove'])
        ).toBe(64)
        expect(await handleShortcutEntry([])).toBeNull()
        expect(app.getPath).not.toHaveBeenCalled()
    })
})
