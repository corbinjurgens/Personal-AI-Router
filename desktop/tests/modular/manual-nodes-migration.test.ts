// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import os from 'os'
import path from 'path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { JsonRpcResponseError } from '@/electron/service-bridge/json-rpc-client'
import {
    legacyManualNodesPath,
    manualNodesFileOps,
    migrateManualNodes,
    type ManualNodeEntry,
    type ManualNodesMigrationDeps
} from '@/electron/service-bridge/manual-nodes-migration'

vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        verbose: vi.fn()
    })
}))

function fakeDeps(
    text: string | null,
    addNode: (entry: ManualNodeEntry) => Promise<void> = () => Promise.resolve()
) {
    const removeFile = vi.fn()
    const added: ManualNodeEntry[] = []
    const deps: ManualNodesMigrationDeps = {
        readFile: () => text,
        removeFile,
        addNode: async entry => {
            added.push(entry)
            await addNode(entry)
        }
    }
    return { deps, removeFile, added }
}

describe('migrateManualNodes', () => {
    it('does nothing when there is no legacy file', async () => {
        const { deps, removeFile, added } = fakeDeps(null)

        await migrateManualNodes(deps)

        expect(added).toEqual([])
        expect(removeFile).not.toHaveBeenCalled()
    })

    it('sends every entry as node/add, then deletes the file', async () => {
        const { deps, removeFile, added } = fakeDeps(
            JSON.stringify([
                { id: 'old-id', address: '100.64.0.2', name: 'studio' },
                { address: 'rig.tailnet.ts.net' }
            ])
        )

        await migrateManualNodes(deps)

        expect(added).toEqual([
            { address: '100.64.0.2', name: 'studio' },
            { address: 'rig.tailnet.ts.net', name: 'rig.tailnet.ts.net' }
        ])
        expect(removeFile).toHaveBeenCalledOnce()
    })

    it('skips malformed entries but still migrates the rest', async () => {
        const { deps, removeFile, added } = fakeDeps(
            JSON.stringify([null, 7, { name: 'no-address' }, { address: '' }, { address: 'ok' }])
        )

        await migrateManualNodes(deps)

        expect(added).toEqual([{ address: 'ok', name: 'ok' }])
        expect(removeFile).toHaveBeenCalledOnce()
    })

    it('deletes an empty list', async () => {
        const { deps, removeFile, added } = fakeDeps('[]')

        await migrateManualNodes(deps)

        expect(added).toEqual([])
        expect(removeFile).toHaveBeenCalledOnce()
    })

    it('deletes a file that cannot be parsed, since it never will be', async () => {
        const { deps, removeFile, added } = fakeDeps('{ not json')

        await migrateManualNodes(deps)

        expect(added).toEqual([])
        expect(removeFile).toHaveBeenCalledOnce()
    })

    it('keeps the file when the broker could not be reached, so the next start retries', async () => {
        const { deps, removeFile, added } = fakeDeps(
            JSON.stringify([{ address: 'a' }, { address: 'b' }]),
            entry =>
                entry.address === 'a'
                    ? Promise.reject(new Error('broker is not running'))
                    : Promise.resolve()
        )

        await migrateManualNodes(deps)

        expect(added.map(entry => entry.address)).toEqual(['a', 'b'])
        expect(removeFile).not.toHaveBeenCalled()
    })

    it('drops an entry the broker rejected rather than retrying it forever', async () => {
        const { deps, removeFile } = fakeDeps(JSON.stringify([{ address: 'bad' }]), () =>
            Promise.reject(new JsonRpcResponseError('-32602: invalid address'))
        )

        await migrateManualNodes(deps)

        expect(removeFile).toHaveBeenCalledOnce()
    })

    it('leaves the file alone when it cannot be read', async () => {
        const { deps, removeFile, added } = fakeDeps(null)
        deps.readFile = () => {
            throw new Error('EACCES')
        }

        await expect(migrateManualNodes(deps)).resolves.toBeUndefined()

        expect(added).toEqual([])
        expect(removeFile).not.toHaveBeenCalled()
    })

    it('survives a failure to delete the file', async () => {
        const { deps } = fakeDeps('[{"address":"a"}]')
        deps.removeFile = () => {
            throw new Error('EPERM')
        }

        await expect(migrateManualNodes(deps)).resolves.toBeUndefined()
    })
})

describe('legacy manual nodes file', () => {
    let dir: string

    beforeEach(() => {
        dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pair-manual-nodes-'))
    })

    afterEach(() => {
        fs.rmSync(dir, { recursive: true, force: true })
    })

    it('lives under <userData>/configs', () => {
        expect(legacyManualNodesPath(dir)).toBe(path.join(dir, 'configs', 'manual-nodes.json'))
    })

    it('is migrated from disk once and then gone', async () => {
        const file = legacyManualNodesPath(dir)
        fs.mkdirSync(path.dirname(file), { recursive: true })
        fs.writeFileSync(file, JSON.stringify([{ address: '10.0.0.9', name: 'nuc' }]))
        const added: ManualNodeEntry[] = []
        const deps: ManualNodesMigrationDeps = {
            ...manualNodesFileOps(file),
            addNode: entry => {
                added.push(entry)
                return Promise.resolve()
            }
        }

        await migrateManualNodes(deps)
        await migrateManualNodes(deps)

        expect(added).toEqual([{ address: '10.0.0.9', name: 'nuc' }])
        expect(fs.existsSync(file)).toBe(false)
    })
})
