// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { BuiltinEngineType } from '@/shared/types/engines'
import type { ModelItem } from '@/shared/types/engines'
import type { ModelEntry } from '@/ui/types/model-hub'

/**
 * Returns true when a hub search result represents a model that is already
 * downloaded locally (per the installed `ModelItem[]` for this engine).
 *
 * Matching is engine-specific because an engine's hub id format and stored
 * model id format may not line up:
 *
 * - llama.cpp:       router inventory and catalog use the same complete
 *                    `<owner>/<repo>:<quant>` id, so only exact ids match.
 */
type DownloadedMatcher = (hubEntry: ModelEntry, downloaded: ModelItem) => boolean

const matchExactName: DownloadedMatcher = (hubEntry, downloaded) => downloaded.name === hubEntry.id

const MATCHERS: Record<BuiltinEngineType, DownloadedMatcher> = {
    'llama-cpp': matchExactName
}

/**
 * `adapterType` is the built-in engine whose naming the engine shares (see
 * `engineAdapterType`); an engine of an unknown family matches nothing.
 */
export function isHubEntryDownloaded(
    adapterType: BuiltinEngineType | null,
    hubEntry: ModelEntry,
    downloaded: readonly ModelItem[]
): boolean {
    if (downloaded.length === 0 || !adapterType) return false
    const matcher = MATCHERS[adapterType]
    for (const d of downloaded) {
        if (!d.downloaded || d.downloadStateCurrent !== true || d.downloadState?.blocksReadiness)
            continue
        if (matcher(hubEntry, d)) return true
    }
    return false
}
