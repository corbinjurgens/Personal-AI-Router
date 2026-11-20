// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash } from 'node:crypto'
import { readFile, readdir, writeFile } from 'node:fs/promises'
import { join, relative, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../dist/renderer', import.meta.url))
const manifest = new URL('../dist/asset-manifest.json', import.meta.url)
const assets = new Map()

async function readAssets(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
        const path = join(directory, entry.name)
        if (entry.isDirectory()) {
            await readAssets(path)
        } else if (entry.isFile()) {
            const key = relative(root, path).split(sep).join('/')
            const hash = createHash('sha256')
                .update(await readFile(path))
                .digest('hex')
            assets.set(key, hash)
        } else {
            throw new Error(`Unexpected asset entry: ${path}`)
        }
    }
}

await readAssets(root)
if (!assets.has('index.html')) throw new Error('Missing Overview index.html')
const sha256 = Object.fromEntries([...assets.keys()].sort().map(key => [key, assets.get(key)]))
await writeFile(manifest, `${JSON.stringify({ sha256 }, null, 2)}\n`)
