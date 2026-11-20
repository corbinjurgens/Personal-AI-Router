// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { resolve } from 'node:path'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'
import { srcAlias } from '../../scripts/vite-alias'

export default defineConfig({
    root: resolve(__dirname, 'ui'),
    base: './',
    publicDir: false,
    build: {
        outDir: resolve(__dirname, 'dist/renderer'),
        emptyOutDir: true
    },
    resolve: { alias: [srcAlias] },
    plugins: [react(), tailwindcss()]
})
