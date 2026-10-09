// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { setNodeInfoPollerVisible } from '@/electron/service-bridge/node-info-poller'

/** The part of a `BrowserWindow` that visibility tracking reads. */
interface TrackedWindow {
    isDestroyed(): boolean
    isVisible(): boolean
    isMinimized(): boolean
    on(event: 'show', listener: () => void): void
    on(event: 'hide', listener: () => void): void
    on(event: 'minimize', listener: () => void): void
    on(event: 'restore', listener: () => void): void
    once(event: 'closed', listener: () => void): void
}

const trackedWindows = new Set<TrackedWindow>()
let anyVisible = false

function isOnScreen(window: TrackedWindow): boolean {
    return !window.isDestroyed() && window.isVisible() && !window.isMinimized()
}

function recompute(): void {
    let visible = false
    for (const window of trackedWindows) {
        if (isOnScreen(window)) {
            visible = true
            break
        }
    }
    if (visible === anyVisible) return
    anyVisible = visible
    setNodeInfoPollerVisible(visible)
}

/**
 * Count an app window (Overview or the tray popup) toward "some window is on
 * screen", which gates work that only feeds what windows display. A window
 * counts while it is shown and not minimized; it stops counting when hidden,
 * minimized, or closed.
 *
 * Occlusion is not detected: a window behind others still counts.
 */
export function trackWindowVisibility(window: TrackedWindow): void {
    trackedWindows.add(window)
    window.on('show', recompute)
    window.on('hide', recompute)
    window.on('minimize', recompute)
    window.on('restore', recompute)
    window.once('closed', () => {
        trackedWindows.delete(window)
        recompute()
    })
    recompute()
}
