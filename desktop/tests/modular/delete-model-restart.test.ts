// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as fs from 'fs'
import * as path from 'path'
import { describe, expect, it } from 'vitest'
import { EngineCapabilities } from '@/ui/constants/engine-capabilities'

/** The llama.cpp manifest must delete models without bouncing the router. */

const MANIFEST_DIR = path.resolve(process.cwd(), '../services/nvpair-engine-manager/manifests')

interface ManifestAction {
    restart_after?: boolean
}

interface Manifest {
    engine: string
    actions?: Record<string, ManifestAction>
}

function readManifests(): Manifest[] {
    return fs
        .readdirSync(MANIFEST_DIR)
        .filter(name => name.endsWith('.json'))
        .map(name => JSON.parse(fs.readFileSync(path.join(MANIFEST_DIR, name), 'utf8')) as Manifest)
}

describe('delete-model restart contract', () => {
    it('llama.cpp deletes without restarting the router', () => {
        const llamacpp = readManifests().find(m => m.engine === 'llamacpp')
        expect(llamacpp?.actions?.delete_model).toBeDefined()
        expect(llamacpp?.actions?.delete_model?.restart_after).toBeUndefined()
        expect(EngineCapabilities['llama-cpp'].hasDeleteModel).toBe(true)
    })
})
