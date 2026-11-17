// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { dialog } from 'electron'
import path from 'path'

/** Native picker for one GGUF model file; null when cancelled. */
export async function chooseModelFile(): Promise<string | null> {
    const selection = await dialog.showOpenDialog({
        title: 'Choose a GGUF model file',
        properties: ['openFile'],
        filters: [{ name: 'GGUF models', extensions: ['gguf'] }]
    })
    if (selection.canceled || selection.filePaths.length !== 1) return null
    const file = selection.filePaths[0]
    if (!path.isAbsolute(file) || Buffer.byteLength(file, 'utf8') > 4096) {
        throw new Error('Invalid model file path.')
    }
    return file
}
