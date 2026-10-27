// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { currentPlatform } from '@/shared/utils/platform'

// Task 5 owns the native writer and readback. Registration must already prevent
// command mode from entering connector/updater/single-instance startup.
export async function handleShortcutEntry(argv: readonly string[]): Promise<number | null> {
    const flags = argv.filter(value => value.startsWith('--pair-shortcut'))
    if (flags.length === 0) return null
    if (flags.length !== 1 || !/^--pair-shortcut=(create|status|remove)$/.test(flags[0])) return 64
    return 69
}

export function ensureWindowsShortcut(): void {
    if (currentPlatform() !== 'win32') return
    // Unavailable until Task 5 supplies the same writer used by command mode.
}
