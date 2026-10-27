// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { IpcMainInvokeEvent, WebContents, WebFrameMain, ipcMain } from 'electron'
import { afterEach, expect, it, vi } from 'vitest'

const ipc = vi.hoisted(() => ({ handlers: new Map<string, Parameters<typeof ipcMain.handle>[1]>() }))
vi.mock('electron', () => ({
    BrowserWindow: { fromWebContents: () => ({}) },
    ipcMain: {
        handle: vi.fn<typeof ipcMain.handle>((channel, handler) => {
            ipc.handlers.set(channel, handler)
        })
    }
}))

import { safeHandle } from '@/electron/ipc/safe-handle'

afterEach(() => vi.unstubAllEnvs())

function eventFor(url: string): IpcMainInvokeEvent {
    const frame = new Proxy<WebFrameMain>(Object.create(null), {
        get: (_target, key) => (key === 'url' ? url : undefined)
    })
    const sender = new Proxy<WebContents>(Object.create(null), {
        get: (_target, key) => (key === 'getURL' ? () => url : key === 'mainFrame' ? frame : undefined)
    })
    return new Proxy<IpcMainInvokeEvent>(Object.create(null), {
        get: (_target, key) => (key === 'sender' ? sender : key === 'senderFrame' ? frame : undefined)
    })
}

it.each([undefined, ''])('refuses an untrusted URL when the development URL is %s', async rendererUrl => {
    vi.stubEnv('ELECTRON_RENDERER_URL', rendererUrl)
    const action = vi.fn<() => void>()
    safeHandle('window:quit', action)
    const handler = ipc.handlers.get('window:quit')
    if (!handler) throw new Error('handler was not registered')
    await expect(handler(eventFor('https://evil.example/'))).resolves.toEqual({
        success: false,
        error: 'Unauthorized sender'
    })
    expect(action).not.toHaveBeenCalled()
})

it.each([
    ['packaged overview', undefined, 'file:///Applications/PAIR.app/Contents/Resources/app/out/ui/index.html'],
    ['packaged tray', undefined, 'file:///C:/PAIR/out/ui/index.html?window=tray'],
    ['development overview', 'http://localhost:5173/', 'http://localhost:5173/'],
    ['development tray', 'http://localhost:5173/', 'http://localhost:5173/?window=tray']
])('accepts the %s renderer', async (_name, rendererUrl, url) => {
    vi.stubEnv('ELECTRON_RENDERER_URL', rendererUrl)
    const action = vi.fn<() => void>()
    safeHandle('window:quit', action)
    const handler = ipc.handlers.get('window:quit')
    if (!handler) throw new Error('handler was not registered')
    await expect(handler(eventFor(url))).resolves.toEqual({ success: true, data: undefined })
    expect(action).toHaveBeenCalledOnce()
})
