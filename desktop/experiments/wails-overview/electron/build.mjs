// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash } from 'node:crypto'
import { cp, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { builtinModules } from 'node:module'
import { dirname, join, relative, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { build as bundle } from 'vite'
import { build as packageApp, Platform } from 'electron-builder'
import { projectionBundle } from './projection-bundle.ts'

const source = dirname(fileURLToPath(import.meta.url))
const project = join(source, '../../..')
const dist = join(source, '../dist')
const stage = join(dist, 'electron-app')
const out = join(dist, 'electron')
const appId = 'com.nvidia.pair.overview-experiment'
const productName = 'PAIR Overview Experiment'
const electronDist = join(project, 'node_modules/electron/dist')
const electronVersion = (await readFile(join(electronDist, 'version'), 'utf8')).trim()
const { sha256 } = JSON.parse(await readFile(join(dist, 'asset-manifest.json'), 'utf8'))

async function verifyRenderer(root) {
    const entries = await readdir(root, { recursive: true, withFileTypes: true })
    const seen = new Set()
    for (const entry of entries) {
        if (entry.isDirectory()) continue
        const path = join(entry.parentPath, entry.name)
        const key = relative(root, path).split(sep).join('/')
        if (!entry.isFile() || !Object.hasOwn(sha256, key)) {
            throw new Error(`Unexpected packaged renderer entry: ${key}`)
        }
        const hash = createHash('sha256')
            .update(await readFile(path))
            .digest('hex')
        if (hash !== sha256[key]) throw new Error(`Renderer hash mismatch: ${key}`)
        seen.add(key)
    }
    if (!seen.has('index.html') || seen.size !== Object.keys(sha256).length) {
        throw new Error('Packaged renderer is missing one or more manifest entries')
    }
    console.log(`Verified renderer SHA-256 manifest (${seen.size} files): ${root}`)
}

function entry(name) {
    return {
        configFile: false,
        resolve: { alias: { '@': join(project, 'src') } },
        build: {
            target: 'node22',
            outDir: join(stage, name),
            emptyOutDir: true,
            lib: {
                entry: join(source, `${name}.ts`),
                formats: ['cjs'],
                fileName: () => 'index.js'
            },
            rollupOptions: {
                external: ['electron', './projection-module.js', /^node:/, ...builtinModules],
                output: { inlineDynamicImports: true }
            }
        }
    }
}

await bundle(entry('main'))
await bundle(projectionBundle(join(stage, 'main')))
await bundle(entry('preload'))
await rm(join(stage, 'ui'), { force: true, recursive: true })
await cp(join(dist, 'renderer'), join(stage, 'ui'), { recursive: true })
await cp(join(project, 'resources/app-icon/pair-16.png'), join(stage, 'tray.png'))
await writeFile(
    join(stage, 'package.json'),
    JSON.stringify({
        name: 'pair-overview-experiment',
        productName,
        version: '0.0.0',
        description: 'PAIR shared Overview comparison host',
        main: 'main/index.js',
        author: 'NVIDIA Corporation'
    })
)
await verifyRenderer(join(stage, 'ui'))

await packageApp({
    projectDir: stage,
    targets: Platform.current().createTarget('dir'),
    publish: 'never',
    config: {
        appId,
        productName,
        electronVersion,
        electronDist,
        npmRebuild: false,
        directories: { output: out },
        files: ['main/**', 'preload/**', 'ui/**', 'tray.png', 'package.json'],
        // Keep the verified renderer as ordinary files for both file URL
        // validation and post-package byte comparison.
        asar: false,
        mac: { identity: null },
        afterPack: async ({ appOutDir, electronPlatformName }) => {
            const resources =
                electronPlatformName === 'darwin'
                    ? join(appOutDir, `${productName}.app/Contents/Resources`)
                    : join(appOutDir, 'resources')
            await verifyRenderer(join(resources, 'app/ui'))
        }
    }
})
