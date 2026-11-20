// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { join, dirname, isAbsolute, relative, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { app, BrowserWindow, ipcMain, Menu, nativeImage, session, Tray } from 'electron'
import { buildCspHeader } from '@/electron/csp'
import { safeHandle } from '@/electron/ipc/safe-handle'
import { rendererPath } from '@/electron/renderer-path'
import { serviceConnectionDeps } from '@/electron/service-bridge/service-connection'
import {
    defaultServiceEndpointEnvironment,
    resolveServiceEndpoint
} from '@/electron/service-bridge/service-endpoint'
import { createProjectionRealm } from './projection-module.js'
import { bindOverviewWindow } from './window-session'

const product = app.getName()
app.setPath('userData', join(app.getPath('appData'), product))

const flags = new Map(
    process.argv
        .slice(1)
        .filter(arg => arg.startsWith('--') && arg.includes('='))
        .map(arg => {
            const separator = arg.indexOf('=')
            return [arg.slice(2, separator), arg.slice(separator + 1)]
        })
)
const mode = flags.get('mode') === 'live' ? 'live' : 'replay'
const index = rendererPath()
const rendererRoot = dirname(index)
let overview: BrowserWindow | undefined
let tray: Tray | undefined

function rendererSession() {
    const isolated = session.fromPartition('pair-overview-experiment')
    isolated.webRequest.onBeforeRequest(({ url }, reply) => {
        let permitted = false
        try {
            const path = relative(rendererRoot, fileURLToPath(url))
            permitted = path !== '..' && !path.startsWith(`..${sep}`) && !isAbsolute(path)
        } catch {
            // Only packaged file URLs may load.
        }
        reply({ cancel: !permitted })
    })
    isolated.webRequest.onHeadersReceived(({ responseHeaders }, reply) =>
        reply({
            responseHeaders: { ...responseHeaders, 'Content-Security-Policy': [buildCspHeader()] }
        })
    )
    isolated.setPermissionRequestHandler((_contents, _permission, reply) => reply(false))
    isolated.setPermissionCheckHandler(() => false)
    return isolated
}

function showOverview() {
    if (overview) {
        if (overview.isMinimized()) overview.restore()
        overview.show()
        overview.focus()
        return
    }
    const window = new BrowserWindow({
        width: 1200,
        height: 900,
        title: product,
        show: false,
        autoHideMenuBar: true,
        webPreferences: {
            preload: join(__dirname, '../preload/index.js'),
            session: rendererSession(),
            contextIsolation: true,
            sandbox: true,
            nodeIntegration: false,
            webSecurity: true
        }
    })
    overview = window
    const contents = window.webContents
    contents.setWindowOpenHandler(() => ({ action: 'deny' }))
    contents.on('will-frame-navigate', event => event.preventDefault())
    contents.once('render-process-gone', () => window.destroy())

    let attachment: ReturnType<typeof bindOverviewWindow> | undefined
    const syncVisibility = () => attachment?.setVisible(window.isVisible() && !window.isMinimized())
    window.on('show', syncVisibility)
    window.on('hide', syncVisibility)
    window.on('minimize', syncVisibility)
    window.on('restore', syncVisibility)
    window.once('closed', () => {
        attachment?.close()
        overview = undefined
    })
    window.once('ready-to-show', () => window.show())
    const query = { mode, scene: flags.get('scene') ?? 'idle', order: flags.get('order') ?? '' }
    void window
        .loadFile(index, { query })
        .then(() => {
            if (mode !== 'live' || window.isDestroyed()) return
            const { connect } = serviceConnectionDeps('', [])
            const dial = () => connect(resolveServiceEndpoint(defaultServiceEndpointEnvironment()))
            attachment = bindOverviewWindow(
                {
                    register(channel, invoke) {
                        safeHandle(channel, (event, request) => {
                            if (event.sender !== contents)
                                throw new Error('Unauthorized Overview sender')
                            return invoke(request)
                        })
                        return () => ipcMain.removeHandler(channel)
                    },
                    send(channel, payload) {
                        if (!contents.isDestroyed()) contents.send(channel, payload)
                    }
                },
                dial,
                createProjectionRealm
            )
            syncVisibility()
        })
        .catch(() => console.error('Could not open packaged Overview'))
}

app.on('window-all-closed', () => {
    // The native tray owns app lifetime.
})
app.on('will-quit', () => tray?.destroy())

if (app.requestSingleInstanceLock()) {
    app.on('second-instance', showOverview)
    void app.whenReady().then(() => {
        if (process.platform === 'darwin') app.setActivationPolicy('accessory')
        const icon = nativeImage.createFromPath(join(app.getAppPath(), 'tray.png'))
        tray = new Tray(icon)
        tray.setToolTip(product)
        tray.setContextMenu(
            Menu.buildFromTemplate([
                { label: 'Overview', click: showOverview },
                { label: 'Quit', click: () => app.quit() }
            ])
        )
        showOverview()
    })
} else {
    app.quit()
}
