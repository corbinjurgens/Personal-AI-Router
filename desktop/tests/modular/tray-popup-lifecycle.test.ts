// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

interface FakeMenuItem {
    label?: string
    type?: string
    checked?: boolean
    click?: (item: { checked: boolean }) => void
}

const mocks = vi.hoisted(() => {
    type Listener = (...args: object[]) => void

    class FakeEmitter {
        private listeners = new Map<string, Listener[]>()

        on(event: string, listener: Listener): this {
            const existing = this.listeners.get(event) ?? []
            existing.push(listener)
            this.listeners.set(event, existing)
            return this
        }

        once(event: string, listener: Listener): this {
            const wrapped: Listener = (...args) => {
                this.off(event, wrapped)
                listener(...args)
            }
            return this.on(event, wrapped)
        }

        off(event: string, listener: Listener): this {
            const existing = this.listeners.get(event) ?? []
            this.listeners.set(
                event,
                existing.filter(entry => entry !== listener)
            )
            return this
        }

        emit(event: string, ...args: object[]): void {
            for (const listener of [...(this.listeners.get(event) ?? [])]) listener(...args)
        }
    }

    class FakeWindow extends FakeEmitter {
        visible = false
        destroyed = false
        isVisible = (): boolean => this.visible
        isMinimized = (): boolean => false
        isDestroyed = (): boolean => this.destroyed
        show = (): void => {
            this.visible = true
            this.emit('show')
        }
        hide = (): void => {
            this.visible = false
            this.emit('hide')
        }
        destroy = (): void => {
            this.destroyed = true
            this.visible = false
            // Electron reports `closed` after destroy() returns.
            setImmediate(() => this.emit('closed'))
        }
        focus = vi.fn()
        setBounds = vi.fn()
        setResizable = vi.fn()
    }

    class FakeTray extends FakeEmitter {
        static instance: FakeTray | null = null
        setToolTip = vi.fn()
        setContextMenu = vi.fn()
        setIgnoreDoubleClickEvents = vi.fn()
        popUpContextMenu = vi.fn()
        getBounds = () => ({ x: 0, y: 0, width: 20, height: 20 })
        destroy = vi.fn()
        constructor() {
            super()
            FakeTray.instance = this
        }
    }

    const state = {
        trayMode: 'popup',
        windows: [] as FakeWindow[],
        lastMenu: [] as {
            label?: string
            type?: string
            checked?: boolean
            click?: (item: { checked: boolean }) => void
        }[]
    }

    const current = (): FakeWindow | null => {
        const win = state.windows.at(-1)
        return win && !win.isDestroyed() ? win : null
    }

    return { FakeWindow, FakeTray, state, current }
})

vi.mock('electron', () => {
    const image = {
        resize: () => image,
        setTemplateImage: vi.fn()
    }
    const workArea = { x: 0, y: 0, width: 1920, height: 1080 }
    return {
        app: { whenReady: () => Promise.resolve(), focus: vi.fn(), quit: vi.fn() },
        BrowserWindow: mocks.FakeWindow,
        Menu: {
            buildFromTemplate: (template: FakeMenuItem[]) => {
                mocks.state.lastMenu = template
                return { items: template, popup: vi.fn() }
            }
        },
        nativeImage: { createFromPath: () => image },
        screen: {
            getCursorScreenPoint: () => ({ x: 1900, y: 1070 }),
            getPrimaryDisplay: () => ({ workArea }),
            getDisplayNearestPoint: () => ({ workArea }),
            on: vi.fn(),
            removeListener: vi.fn()
        },
        Tray: mocks.FakeTray
    }
})

vi.mock('@/electron/window', () => ({
    createTrayWindow: () => {
        const existing = mocks.current()
        if (existing) return existing
        const win = new mocks.FakeWindow()
        mocks.state.windows.push(win)
        return win
    },
    getTrayWindow: () => mocks.current(),
    createOverviewWindow: vi.fn()
}))

vi.mock('@/electron/config/ui-config', () => ({
    getTrayMode: () => mocks.state.trayMode,
    setTrayMode: (mode: string) => {
        mocks.state.trayMode = mode
    }
}))

vi.mock('@/shared/utils/platform', () => ({ currentPlatform: () => 'win32' }))

vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        verbose: vi.fn()
    })
}))

import { destroyTray, initTray } from '@/electron/tray'

const IDLE_DESTROY_MS = 60_000
const FIRST_SHOW_FALLBACK_MS = 1_500

function tray(): InstanceType<typeof mocks.FakeTray> {
    const instance = mocks.FakeTray.instance
    if (!instance) throw new Error('tray was not created')
    return instance
}

function click(): void {
    tray().emit('click', {}, { x: 1900, y: 1060, width: 20, height: 20 })
}

function onlyWindow(): InstanceType<typeof mocks.FakeWindow> {
    expect(mocks.state.windows).toHaveLength(1)
    return mocks.state.windows[0]
}

describe('tray popup lifecycle', () => {
    beforeEach(async () => {
        vi.useFakeTimers()
        mocks.state.trayMode = 'popup'
        mocks.state.windows = []
        await initTray()
    })

    afterEach(() => {
        destroyTray()
        vi.useRealTimers()
    })

    it('creates no popup window at startup', () => {
        expect(mocks.state.windows).toHaveLength(0)
    })

    it('creates the popup on the first click and shows it once painted', () => {
        click()
        const win = onlyWindow()
        expect(win.visible).toBe(false)

        win.emit('ready-to-show')
        expect(win.visible).toBe(true)
    })

    it('shows a popup whose first paint is late after the fallback', async () => {
        click()
        const win = onlyWindow()

        await vi.advanceTimersByTimeAsync(FIRST_SHOW_FALLBACK_MS)
        expect(win.visible).toBe(true)
    })

    it('destroys the popup after it stays hidden, and rebuilds it on the next click', async () => {
        click()
        const first = onlyWindow()
        first.emit('ready-to-show')
        first.hide()

        await vi.advanceTimersByTimeAsync(IDLE_DESTROY_MS - 1)
        expect(first.destroyed).toBe(false)
        await vi.advanceTimersByTimeAsync(1)
        expect(first.destroyed).toBe(true)

        click()
        expect(mocks.state.windows).toHaveLength(2)
        const second = mocks.state.windows[1]
        second.emit('ready-to-show')
        expect(second.visible).toBe(true)
    })

    it('keeps a popup reopened before the idle timeout', async () => {
        click()
        const win = onlyWindow()
        win.emit('ready-to-show')
        win.hide()

        await vi.advanceTimersByTimeAsync(IDLE_DESTROY_MS / 2)
        click()
        expect(win.visible).toBe(true)

        await vi.advanceTimersByTimeAsync(IDLE_DESTROY_MS * 2)
        expect(win.destroyed).toBe(false)
    })

    it('dismisses a popup clicked again while it is still loading', async () => {
        click()
        const win = onlyWindow()
        click()

        win.emit('ready-to-show')
        await vi.advanceTimersByTimeAsync(FIRST_SHOW_FALLBACK_MS)
        expect(win.visible).toBe(false)

        await vi.advanceTimersByTimeAsync(IDLE_DESTROY_MS)
        expect(win.destroyed).toBe(true)
    })

    it('builds a replacement when clicked between destroy and its closed event', async () => {
        click()
        const first = onlyWindow()
        first.emit('ready-to-show')
        first.hide()
        await vi.advanceTimersByTimeAsync(IDLE_DESTROY_MS)
        expect(first.destroyed).toBe(true)

        // `closed` for the old window has not been delivered yet.
        click()
        const second = mocks.state.windows[1]
        await vi.advanceTimersByTimeAsync(0)
        second.emit('ready-to-show')
        expect(second.visible).toBe(true)
        expect(second.destroyed).toBe(false)
    })

    it('opens only the native menu in menu mode', () => {
        mocks.state.trayMode = 'menu'
        click()

        expect(mocks.state.windows).toHaveLength(0)
        expect(tray().popUpContextMenu).toHaveBeenCalled()
    })

    it('destroys an existing popup when switched to menu mode from the menu', async () => {
        click()
        const win = onlyWindow()
        win.emit('ready-to-show')

        const toggle = mocks.state.lastMenu.find(item => item.type === 'checkbox')
        expect(toggle?.checked).toBe(true)
        toggle?.click?.({ checked: false })
        await vi.advanceTimersByTimeAsync(0)

        expect(mocks.state.trayMode).toBe('menu')
        expect(win.destroyed).toBe(true)
        expect(mocks.state.lastMenu.find(item => item.type === 'checkbox')?.checked).toBe(false)
    })
})
