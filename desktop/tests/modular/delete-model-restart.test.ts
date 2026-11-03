// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as fs from 'fs'
import * as path from 'path'
import { describe, expect, it } from 'vitest'
import { EngineCapabilities } from '@/ui/constants/engine-capabilities'
import type { BuiltinEngineType } from '@/shared/types/engines'

/** The manifest restart behavior and desktop confirmation must agree. */

const MANIFEST_DIR = path.resolve(process.cwd(), '../services/nvpair-engine-manager/manifests')

/** Map engine-manager manifest ids to the desktop's built-in engine ids. */
const ENGINE_TYPE_BY_MANIFEST_ID: Record<string, BuiltinEngineType> = {
    llamacpp: 'llama-cpp'
}

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
        expect(EngineCapabilities['llama-cpp'].restartsOnModelDelete).toBeFalsy()
        expect(EngineCapabilities['llama-cpp'].hasDeleteModel).toBe(true)
    })

    it('every engine that bounces on delete also warns the user first', () => {
        for (const manifest of readManifests()) {
            const engineType = ENGINE_TYPE_BY_MANIFEST_ID[manifest.engine]
            if (!engineType) continue
            const bounces = manifest.actions?.delete_model?.restart_after === true
            expect(
                EngineCapabilities[engineType].restartsOnModelDelete ?? false,
                `${manifest.engine}: manifest restart_after=${bounces} but EngineCapabilities['${engineType}'].restartsOnModelDelete disagrees`
            ).toBe(bounces)
        }
    })
})
