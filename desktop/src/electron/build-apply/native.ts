// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { app, dialog } from 'electron'
import path from 'node:path'
import type { BuildApplyGuiFacts } from '@/shared/types/build-apply'

export function guiFacts(): BuildApplyGuiFacts {
    return {
        pid: process.pid,
        executable: process.execPath,
        buildId: typeof __PAIR_GUI_BUILD_ID__ === 'string' ? __PAIR_GUI_BUILD_ID__ : ''
    }
}

export async function chooseCandidatePath(): Promise<string | null> {
    const selection = await dialog.showOpenDialog({
        title: 'Choose staged build',
        properties: ['openDirectory', 'treatPackageAsDirectory']
    })
    if (selection.canceled || selection.filePaths.length !== 1) return null
    const candidate = selection.filePaths[0]
    if (!path.isAbsolute(candidate) || Buffer.byteLength(candidate, 'utf8') > 4096) {
        throw new Error('Invalid staged build path.')
    }
    return candidate
}

/** The acknowledged Go runner, rather than this client, owns service shutdown. */
export async function quitForBuildApply(detach: () => Promise<void>): Promise<void> {
    await detach()
    app.quit()
}
