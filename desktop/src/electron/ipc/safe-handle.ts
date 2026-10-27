// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BrowserWindow, ipcMain, IpcMainInvokeEvent } from 'electron'
import { is } from '@electron-toolkit/utils'
import { fileURLToPath } from 'url'
import { rendererPath } from '@/electron/renderer-path'
import type {
    IpcResult,
    IpcChannelKey,
    IpcChannelMap,
    IpcHandlerFn
} from '@/shared/types/ipc-channels'
import getErrorString from '@/shared/utils/get-error-string'
import { JsonRpcResponseError } from '@/electron/service-bridge/json-rpc-client'

function isKnownSender(event: IpcMainInvokeEvent): boolean {
    const senderWindow = BrowserWindow.fromWebContents(event.sender)
    if (!senderWindow || !event.senderFrame || event.senderFrame !== event.sender.mainFrame)
        return false

    try {
        const url = new URL(event.senderFrame.url)
        if (url.protocol === 'file:') return fileURLToPath(url) === rendererPath()

        const rendererUrl = process.env.ELECTRON_RENDERER_URL
        if (!is.dev || !rendererUrl) return false
        const expected = new URL(rendererUrl)
        return (
            (expected.protocol === 'http:' || expected.protocol === 'https:') &&
            !expected.username &&
            !expected.password &&
            !url.username &&
            !url.password &&
            url.origin === expected.origin &&
            url.pathname === expected.pathname
        )
    } catch {
        return false
    }
}

type Req<K extends IpcChannelKey> = IpcChannelMap[K]['request']
type Res<K extends IpcChannelKey> = IpcChannelMap[K]['response']

/**
 * Type-safe IPC handler. Channel and handler signature are checked against IpcChannelMap.
 * The handler may return synchronously or asynchronously.
 *
 * This function is the single trust boundary between Electron's untyped `ipcMain.handle`
 * and our typed IPC system. The implementation cast bridges the generic fn to the
 * concrete call — callers see only the typed surface.
 */
export function safeHandle<K extends IpcChannelKey>(
    channel: K,
    fn: IpcHandlerFn<Req<K>, Res<K>, IpcMainInvokeEvent>
): void {
    const handler = fn as (event: IpcMainInvokeEvent, ...rest: unknown[]) => unknown
    ipcMain.handle(channel, async (event, ...args): Promise<IpcResult<Res<K>>> => {
        if (!isKnownSender(event)) {
            return { success: false, error: 'Unauthorized sender' }
        }
        try {
            const data = (await handler(event, ...args)) as Res<K>
            return { success: true, data }
        } catch (err) {
            const errorString = getErrorString(err) ?? String(err)
            // A service's refusal keeps its code: an Error crossing the
            // context bridge keeps only its message.
            if (err instanceof JsonRpcResponseError) {
                return { success: false, error: errorString, code: err.code }
            }
            return { success: false, error: errorString }
        }
    })
}
