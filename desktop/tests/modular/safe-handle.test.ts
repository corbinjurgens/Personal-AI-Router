// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { join } from 'path'
import { pathToFileURL } from 'url'
import type { IpcMainInvokeEvent, WebContents, WebFrameMain, ipcMain } from 'electron'
import { afterEach, expect, it, vi } from 'vitest'

const ipc = vi.hoisted(() => ({
    handlers: new Map<string, Parameters<typeof ipcMain.handle>[1]>(),
    owned: true
}))
vi.mock('electron', () => ({
    BrowserWindow: { fromWebContents: () => (ipc.owned ? {} : null) },
    ipcMain: {
        handle: vi.fn<typeof ipcMain.handle>((channel, handler) => {
            ipc.handlers.set(channel, handler)
        })
    }
}))
vi.mock('@electron-toolkit/utils', () => ({ is: { dev: true } }))

import { safeHandle } from '@/electron/ipc/safe-handle'

afterEach(() => {
    vi.unstubAllEnvs()
    ipc.owned = true
})

function eventFor(url: string, subframe = false): IpcMainInvokeEvent {
    const frame = new Proxy<WebFrameMain>(Object.create(null), {
        get: (_target, key) => (key === 'url' ? url : undefined)
    })
    const sender = new Proxy<WebContents>(Object.create(null), {
        get: (_target, key) =>
            key === 'getURL' ? () => url : key === 'mainFrame' ? frame : undefined
    })
    return new Proxy<IpcMainInvokeEvent>(Object.create(null), {
        get: (_target, key) =>
            key === 'sender'
                ? sender
                : key === 'senderFrame'
                  ? subframe
                      ? null
                      : frame
                  : undefined
    })
}

const packaged = pathToFileURL(join(process.cwd(), 'src/ui/index.html')).href

async function check(url: string, allowed: boolean, subframe = false): Promise<void> {
    const action = vi.fn<() => void>()
    safeHandle('window:quit', action)
    const handler = ipc.handlers.get('window:quit')
    if (!handler) throw new Error('handler was not registered')
    await expect(handler(eventFor(url, subframe))).resolves.toEqual(
        allowed
            ? { success: true, data: undefined }
            : { success: false, error: 'Unauthorized sender' }
    )
    expect(action).toHaveBeenCalledTimes(allowed ? 1 : 0)
}

it.each([
    ['overview', packaged],
    ['percent-encoded path', packaged.replace('index.html', '%69ndex.html')],
    ['tray query and hash', `${packaged}?window=tray#section`]
])('accepts packaged %s', async (_name, url) => {
    vi.stubEnv('ELECTRON_RENDERER_URL', undefined)
    await check(url, true)
})

it.each([
    ['unrelated file', 'file:///tmp/index.html'],
    ['sibling file', packaged.replace('index.html', 'other.html')],
    ['malformed URL', 'not a URL'],
    ['network URL without development configuration', 'http://localhost:5173/']
])('rejects %s', async (_name, url) => {
    vi.stubEnv('ELECTRON_RENDERER_URL', undefined)
    await check(url, false)
})

it.each([
    ['overview', 'http://localhost:5173/app/'],
    ['tray', 'http://localhost:5173/app/?window=tray#section']
])('accepts development %s', async (_name, url) => {
    vi.stubEnv('ELECTRON_RENDERER_URL', 'http://localhost:5173/app/')
    await check(url, true)
})

it.each([
    ['prefix host', 'http://localhost.evil:5173/app/'],
    ['wrong port', 'http://localhost:5174/app/'],
    ['wrong scheme', 'https://localhost:5173/app/'],
    ['userinfo', 'http://user@localhost:5173/app/'],
    ['different path', 'http://localhost:5173/app/other'],
    ['opaque origin', 'data:text/html,hello']
])('rejects development %s', async (_name, url) => {
    vi.stubEnv('ELECTRON_RENDERER_URL', 'http://localhost:5173/app/')
    await check(url, false)
})

it('rejects an unowned window', async () => {
    ipc.owned = false
    await check(packaged, false)
})

it('rejects a subframe', async () => {
    await check(packaged, false, true)
})

it('rejects a host that extends a configured origin without a trailing slash', async () => {
    vi.stubEnv('ELECTRON_RENDERER_URL', 'http://localhost')
    await check('http://localhost.evil/', false)
})

it('rejects a malformed development configuration', async () => {
    vi.stubEnv('ELECTRON_RENDERER_URL', 'not a URL')
    await check('http://localhost/', false)
})
