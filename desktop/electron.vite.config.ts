// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { resolve } from 'path'
import { defineConfig } from 'electron-vite'
import { srcAlias } from './scripts/vite-alias'
import { buildIdentity } from './scripts/build-identity'
import react from '@vitejs/plugin-react'
// @ts-ignore - Tailwind CSS plugin
import tailwindcss from '@tailwindcss/vite'
import { copyFileSync, mkdirSync, readdirSync } from 'fs'

// NOTE on bundling strategy: we deliberately disable dependency externalization
// via `build.externalizeDeps: false`. Electron-vite still externalizes
// `electron`, `electron/*`, and Node built-ins out of the box (unconditionally
// — see node_modules/electron-vite/dist/chunks/*.js), which is all we need at
// runtime. Setting `externalizeDeps: false` makes Vite inline npm imports into
// the main and preload bundles; the renderer also bundles its dependencies
// into `out/ui/`. Independently, electron-builder collects the locked production
// dependency closure into the packaged app's `node_modules/` tree, including
// renderer dependencies. The `files` whitelist does not disable that collection.

function copyIconsPlugin() {
    return {
        name: 'copy-icons',
        closeBundle() {
            const src = resolve(__dirname, 'resources', 'icons')
            const dests = [
                resolve(__dirname, 'out/resources/icons'),
                resolve(__dirname, 'resources/icons')
            ]
            // Only the flat logo.* files are needed at runtime (tray.ts / window.ts).
            // Skip subdirectories such as linux/, which is consumed by electron-builder
            // at package time, not by the running app — copyFileSync on a dir throws.
            const files = readdirSync(src, { withFileTypes: true })
                .filter(entry => entry.isFile())
                .map(entry => entry.name)
            for (const dest of dests) {
                mkdirSync(dest, { recursive: true })
                for (const file of files) {
                    copyFileSync(resolve(src, file), resolve(dest, file))
                }
            }
        }
    }
}

export default defineConfig({
    main: {
        define: { __PAIR_GUI_BUILD_ID__: JSON.stringify(buildIdentity(resolve(__dirname, '..'))) },
        plugins: [copyIconsPlugin()],
        build: {
            externalizeDeps: false,
            rollupOptions: {
                input: {
                    index: resolve(__dirname, 'src', 'electron', 'index.ts')
                },
                output: {
                    chunkFileNames: '[name].js'
                },
                onwarn(warning, warn) {
                    if (warning.code === 'EVAL' && warning.id?.includes('@protobufjs')) return
                    warn(warning)
                }
            }
        },
        resolve: {
            alias: [srcAlias]
        }
    },
    preload: {
        build: {
            externalizeDeps: false,
            rollupOptions: {
                input: {
                    index: resolve(__dirname, 'src', 'preload', 'index.ts')
                }
            }
        },
        resolve: {
            alias: [srcAlias]
        }
    },
    renderer: {
        root: resolve(__dirname, 'src/ui'),
        // Relative base so the built `index.html` references assets as
        // `./assets/main-*.js`. Electron loads the UI via
        // `file://.../out/ui/index.html`; absolute `/assets/...` paths
        // resolve to `file:///assets/...` and fail. Relative paths work in
        // both `file://` (Electron) and HTTP (frontend gateway at `/`).
        base: './',
        server: {
            hmr: false
        },
        build: {
            outDir: resolve(__dirname, 'out/ui'),
            // cssCodeSplit: false,
            // assetsInlineLimit: Infinity,
            rollupOptions: {
                input: {
                    main: resolve(__dirname, 'src/ui/index.html')
                }
                // output: {
                //     inlineDynamicImports: true
                // }
            }
        },
        resolve: {
            alias: [srcAlias]
        },
        plugins: [react(), tailwindcss()]
    }
})
