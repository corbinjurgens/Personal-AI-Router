// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fileURLToPath } from 'node:url'
import { build, mergeConfig } from 'vite'
import { projectionBundle } from '../../../experiments/wails-overview/electron/projection-bundle.ts'

// Build the same projection module as the package in Node, outside jsdom.
const outDir = process.argv[2]
if (!outDir) throw new Error('Missing fixture build directory')
await build(
    mergeConfig(projectionBundle(outDir), {
        logLevel: 'error',
        build: {
            rollupOptions: {
                output: {
                    paths: {
                        electron: fileURLToPath(new URL('./electron-fixture.cjs', import.meta.url))
                    }
                }
            }
        }
    })
)
