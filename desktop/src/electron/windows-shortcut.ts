// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import path from 'path'
import { app, shell } from 'electron'
import { APP_DISPLAY_NAME, APP_EXECUTABLE_NAME, APP_ID } from '@/shared/constants/app'
import { currentPlatform } from '@/shared/utils/platform'
import { createStructuredLogger } from '@/shared/utils/log'
import getErrorString from '@/shared/utils/get-error-string'

const log = createStructuredLogger('app')
const windowsPath = path.win32
const executableName = `${APP_EXECUTABLE_NAME}.exe`
type Command = 'create' | 'status' | 'remove'

function samePath(a: string, b: string): boolean {
    return windowsPath.normalize(a).toLowerCase() === windowsPath.normalize(b).toLowerCase()
}

function isPairTarget(target: string): boolean {
    return (
        windowsPath.isAbsolute(target) &&
        windowsPath.basename(target).toLowerCase() === executableName.toLowerCase()
    )
}

function shortcutPath(): string {
    return windowsPath.join(
        app.getPath('appData'),
        'Microsoft',
        'Windows',
        'Start Menu',
        'Programs',
        `${APP_DISPLAY_NAME}.lnk`
    )
}

function isInside(target: string, directory: string): boolean {
    const relative = windowsPath.relative(directory, target)
    return (
        relative === '' ||
        (!windowsPath.isAbsolute(relative) && relative !== '..' && !relative.startsWith('..\\'))
    )
}

function targetPath(): string {
    if (!app.isPackaged) throw new Error('Shortcut requires a packaged or unpacked PAIR build')
    const target = fs.realpathSync(app.getPath('exe'))
    if (!isPairTarget(target))
        throw new Error(`Shortcut target must be an absolute ${executableName}: ${target}`)
    if (!fs.statSync(target).isFile())
        throw new Error(`Shortcut target is not a regular file: ${target}`)
    if (
        /^windows-(?:x64|arm64)[.-](?:prev|backup|failed)-/i.test(
            windowsPath.basename(windowsPath.dirname(target))
        )
    )
        throw new Error('An archived PAIR install cannot own the live Start Menu shortcut')
    const temporary = fs.realpathSync(app.getPath('temp'))
    if (
        isInside(target, temporary) ||
        target.toLowerCase().split(windowsPath.sep).includes('win-unpacked')
    )
        throw new Error(
            'Move the unpacked PAIR build to its stable installation folder before creating a shortcut'
        )
    for (
        let directory = windowsPath.dirname(target);
        ;
        directory = windowsPath.dirname(directory)
    ) {
        if (fs.existsSync(windowsPath.join(directory, '.git')))
            throw new Error('PAIR checkout build outputs cannot own the Start Menu shortcut')
        if (windowsPath.dirname(directory) === directory) break
    }
    return target
}

function readLink(link: string): Electron.ShortcutDetails | null {
    try {
        const entry = fs.lstatSync(link)
        if (!entry.isFile()) throw new Error(`Shortcut path is not a regular file: ${link}`)
    } catch (error) {
        if (error instanceof Error && 'code' in error && error.code === 'ENOENT') return null
        throw error
    }
    return shell.readShortcutLink(link)
}

function owned(details: Electron.ShortcutDetails): boolean {
    return details.appUserModelId === APP_ID && isPairTarget(details.target) && !details.args
}

function matches(details: Electron.ShortcutDetails, target: string): boolean {
    return (
        owned(details) &&
        samePath(details.target, target) &&
        samePath(details.cwd ?? '', windowsPath.dirname(target)) &&
        samePath(details.icon ?? '', target) &&
        details.iconIndex === 0 &&
        details.description === `Launch ${APP_DISPLAY_NAME}`
    )
}

function operate(command: Command): string {
    const target = targetPath()
    const link = shortcutPath()
    const existing = readLink(link)
    if (existing && !owned(existing))
        throw new Error(`Unrelated shortcut at ${link}; move it before retrying`)
    if (command === 'status') {
        if (!existing) throw new Error(`Shortcut not found: ${link}`)
        if (!matches(existing, target)) throw new Error(`PAIR shortcut needs repair: ${link}`)
        return `PAIR shortcut is current: ${link}`
    }
    if (command === 'remove') {
        if (!existing) throw new Error(`Shortcut not found: ${link}`)
        fs.unlinkSync(link)
        return `Removed PAIR shortcut: ${link}`
    }
    if (existing && matches(existing, target)) return `PAIR shortcut is unchanged: ${link}`
    fs.mkdirSync(windowsPath.dirname(link), { recursive: true })
    const details: Electron.ShortcutDetails = {
        target,
        cwd: windowsPath.dirname(target),
        icon: target,
        iconIndex: 0,
        description: `Launch ${APP_DISPLAY_NAME}`,
        appUserModelId: APP_ID,
        args: ''
    }
    if (!shell.writeShortcutLink(link, 'create', details))
        throw new Error(`Could not write shortcut: ${link}`)
    const written = readLink(link)
    if (!written || !matches(written, target)) throw new Error(`Shortcut readback failed: ${link}`)
    return `${existing ? 'Repaired' : 'Created'} PAIR shortcut: ${link}`
}

export async function handleShortcutEntry(argv: readonly string[]): Promise<number | null> {
    const flags = argv.filter(value => value.startsWith('--pair-shortcut'))
    if (flags.length === 0) return null
    if (flags.length !== 1 || !/^--pair-shortcut=(create|status|remove)$/.test(flags[0])) {
        console.error(
            'Use --pair-shortcut=create, --pair-shortcut=status, or --pair-shortcut=remove'
        )
        return 64
    }
    if (currentPlatform() !== 'win32') {
        console.error('PAIR shortcuts are available only on Windows')
        return 69
    }
    const command = flags[0].substring('--pair-shortcut='.length)
    try {
        if (command === 'create' || command === 'status' || command === 'remove')
            console.log(operate(command))
        return 0
    } catch (error) {
        console.error(`PAIR shortcut ${command} failed: ${getErrorString(error)}`)
        return 1
    }
}

export function ensureWindowsShortcut(): void {
    if (currentPlatform() !== 'win32' || !app.isPackaged) return
    try {
        const result = operate('create')
        log.info({ sublevel: 'shortcut', message: result })
    } catch (error) {
        log.warn({
            sublevel: 'shortcut',
            message: `Could not create or repair the PAIR Start Menu shortcut: ${getErrorString(error)}`
        })
    }
}
