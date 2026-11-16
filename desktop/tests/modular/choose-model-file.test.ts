// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import path from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { dialog } from 'electron'
import { chooseModelFile } from '@/electron/models/choose-model-file'

vi.mock('electron', () => ({
    dialog: { showOpenDialog: vi.fn<typeof dialog.showOpenDialog>() }
}))

describe('model file picker', () => {
    beforeEach(() => vi.mocked(dialog.showOpenDialog).mockReset())

    it('asks for one GGUF file', async () => {
        const file = path.resolve('models', 'Model.gguf')
        vi.mocked(dialog.showOpenDialog).mockResolvedValue({ canceled: false, filePaths: [file] })
        await expect(chooseModelFile()).resolves.toBe(file)
        expect(dialog.showOpenDialog).toHaveBeenCalledWith({
            title: 'Choose a GGUF model file',
            properties: ['openFile'],
            filters: [{ name: 'GGUF models', extensions: ['gguf'] }]
        })
    })

    it('returns null when cancelled or not exactly one file', async () => {
        vi.mocked(dialog.showOpenDialog).mockResolvedValue({ canceled: true, filePaths: [] })
        await expect(chooseModelFile()).resolves.toBeNull()
        vi.mocked(dialog.showOpenDialog).mockResolvedValue({
            canceled: false,
            filePaths: [path.resolve('a.gguf'), path.resolve('b.gguf')]
        })
        await expect(chooseModelFile()).resolves.toBeNull()
    })

    it('refuses a relative or overlong path', async () => {
        vi.mocked(dialog.showOpenDialog).mockResolvedValue({
            canceled: false,
            filePaths: ['Model.gguf']
        })
        await expect(chooseModelFile()).rejects.toThrow('Invalid model file path.')
        vi.mocked(dialog.showOpenDialog).mockResolvedValue({
            canceled: false,
            filePaths: [path.resolve('m'.repeat(5000) + '.gguf')]
        })
        await expect(chooseModelFile()).rejects.toThrow('Invalid model file path.')
    })
})
