// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import path from 'path'
import getErrorString from '@/shared/utils/get-error-string'
import { createStructuredLogger } from '@/shared/utils/log'
import { JsonRpcResponseError, type JsonValue } from './json-rpc-client'

const log = createStructuredLogger('service-bridge')

/** A manual peer as `node/add` takes it. */
export interface ManualNodeEntry {
    address: string
    name: string
}

export interface ManualNodesMigrationDeps {
    /** The file's text, or null when it does not exist. */
    readFile: () => string | null
    removeFile: () => void
    /** Send one entry to the broker as `node/add`. */
    addNode: (entry: ManualNodeEntry) => Promise<void>
}

/** `<userData>/configs/manual-nodes.json`, where Electron used to keep the list. */
export function legacyManualNodesPath(userDataDir: string): string {
    return path.join(userDataDir, 'configs', 'manual-nodes.json')
}

/** The production file operations for a legacy list at `filePath`. */
export function manualNodesFileOps(
    filePath: string
): Pick<ManualNodesMigrationDeps, 'readFile' | 'removeFile'> {
    return {
        readFile: () => {
            try {
                return fs.readFileSync(filePath, 'utf8')
            } catch (err) {
                if (err instanceof Error && 'code' in err && err.code === 'ENOENT') return null
                throw err
            }
        },
        removeFile: () => fs.rmSync(filePath, { force: true })
    }
}

function parseEntries(text: string): ManualNodeEntry[] {
    const parsed: JsonValue = JSON.parse(text)
    if (!Array.isArray(parsed)) return []
    const entries: ManualNodeEntry[] = []
    for (const item of parsed) {
        if (item === null || typeof item !== 'object' || Array.isArray(item)) continue
        const address = typeof item.address === 'string' ? item.address : ''
        if (!address) continue
        const name = typeof item.name === 'string' && item.name ? item.name : address
        entries.push({ address, name })
    }
    return entries
}

/**
 * The one-time move of Electron's manual peer list into the service. The
 * broker's `nvpair-manual-nodes` now persists its own list, so each entry in the
 * old file is sent as `node/add` and the file is deleted; nothing replays it.
 *
 * The file is kept for the next attempt when an entry could not be delivered
 * because the broker could not be reached. An entry the broker answered with an
 * error will never succeed, so it is dropped with a warning rather than retried
 * forever. A file that cannot be parsed is deleted for the same reason.
 */
export async function migrateManualNodes(deps: ManualNodesMigrationDeps): Promise<void> {
    let text: string | null
    try {
        text = deps.readFile()
    } catch (err) {
        log.warn({
            sublevel: 'manual-nodes',
            message: `Could not read the legacy manual node list: ${getErrorString(err)}`
        })
        return
    }
    if (text === null) return

    let entries: ManualNodeEntry[] = []
    try {
        entries = parseEntries(text)
    } catch (err) {
        log.warn({
            sublevel: 'manual-nodes',
            message: `Dropping an unreadable legacy manual node list: ${getErrorString(err)}`
        })
    }

    let unreachable = false
    for (const entry of entries) {
        try {
            await deps.addNode(entry)
        } catch (err) {
            log.warn({
                sublevel: 'manual-nodes',
                message: `Failed to migrate manual node ${entry.address}: ${getErrorString(err)}`
            })
            if (!(err instanceof JsonRpcResponseError)) unreachable = true
        }
    }
    if (unreachable) return

    try {
        deps.removeFile()
    } catch (err) {
        log.warn({
            sublevel: 'manual-nodes',
            message: `Could not delete the migrated manual node list: ${getErrorString(err)}`
        })
    }
}
