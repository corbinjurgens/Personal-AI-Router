// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ProgressBar, Stack, Text } from '@nvidia/foundations-react-core'
import type { ModelDownloadEntry } from '@/shared/types/model-download'
import { formatDownloadFailure, formatDownloadStatus } from '@/ui/utils/formatters'

/**
 * One download's state from its node's current download state: a moving bar
 * while it starts or waits, a measured bar with percent while it runs, and the
 * outcome in words once it ends. `entry` absent with `starting` set is the
 * moment between the click and the node's first report.
 */
export function DownloadProgress({
    label,
    entry,
    starting = false
}: {
    /** Names the download for assistive technology, such as the model or quantization. */
    label: string
    entry?: ModelDownloadEntry
    starting?: boolean
}) {
    const phase = entry?.phase
    if (!entry && !starting) return null
    if (phase === 'complete') {
        return (
            <Text kind="body/regular/xs" style={{ color: 'var(--text-color-feedback-success)' }}>
                {formatDownloadStatus('complete', true)}
            </Text>
        )
    }
    if (phase === 'failed' || phase === 'unknown') {
        return (
            <Text kind="body/regular/xs" style={{ color: 'var(--text-color-feedback-danger)' }}>
                {phase === 'failed'
                    ? `${formatDownloadStatus('failed', true)}: ${formatDownloadFailure(entry?.reason)}`
                    : formatDownloadStatus('unknown', true)}
            </Text>
        )
    }
    const percent = phase === 'in_progress' ? entry?.percent : undefined
    const text = !entry
        ? 'Starting download'
        : percent !== undefined
          ? `${formatDownloadStatus('in_progress', true)} · ${Math.round(percent)}%`
          : formatDownloadStatus(entry.phase, true)
    return (
        <Stack gap="1">
            {percent !== undefined ? (
                <ProgressBar
                    size="small"
                    value={Math.min(100, Math.max(0, percent))}
                    aria-label={`Download progress for ${label}`}
                />
            ) : (
                <ProgressBar
                    size="small"
                    kind="indeterminate"
                    aria-label={`Download progress for ${label}`}
                />
            )}
            <Text kind="body/regular/xs" className="text-subtle-color">
                {text}
            </Text>
        </Stack>
    )
}
