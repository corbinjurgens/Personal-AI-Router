// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelDownloadEntry } from '@/shared/types/model-download'

/**
 * Utility functions for formatting data
 */

/**
 * Format a timestamp as a relative time distance (e.g., "2 minutes ago")
 */
function formatDistance(timestamp: number): string {
    const now = Date.now()
    const diff = now - timestamp
    const seconds = Math.floor(diff / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)
    const days = Math.floor(hours / 24)

    if (seconds < 10) {
        return 'just now'
    } else if (seconds < 60) {
        return `${seconds} seconds ago`
    } else if (minutes < 60) {
        return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`
    } else if (hours < 24) {
        return `${hours} hour${hours !== 1 ? 's' : ''} ago`
    } else {
        return `${days} day${days !== 1 ? 's' : ''} ago`
    }
}

/**
 * Like {@link formatDistance}, but uses approximate months/years for older times (≥30 days).
 * Clock-ahead timestamps (future) are shown as "just now".
 */
export function formatRelativeTime(timestamp: number): string {
    const diffMs = Date.now() - timestamp
    if (diffMs < 0) {
        return 'just now'
    }

    const MS_PER_DAY = 86_400_000
    const totalDays = Math.floor(diffMs / MS_PER_DAY)

    if (totalDays >= 365) {
        const years = Math.floor(totalDays / 365)
        return `${years} year${years !== 1 ? 's' : ''} ago`
    }
    if (totalDays >= 30) {
        const months = Math.floor(totalDays / 30)
        return `${months} month${months !== 1 ? 's' : ''} ago`
    }

    return formatDistance(timestamp)
}

/**
 * Format bytes to human-readable size
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
    // Handle invalid inputs
    if (bytes == null || isNaN(bytes) || bytes < 0 || bytes === 0) {
        return '0 B'
    }

    const k = 1024
    const dm = decimals < 0 ? 0 : decimals
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']

    const i = Math.floor(Math.log(bytes) / Math.log(k))

    if (i < 0 || i >= sizes.length) {
        return '0 B'
    }

    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

/** Short count for statistics such as downloads: 1234 -> "1.2K", 3400000 -> "3.4M". */
export function formatCompactCount(count: number): string {
    return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(
        Math.max(0, count)
    )
}

type PullProgressFields = {
    status?: string
    percent?: number
    completed?: number
    total?: number
}

/**
 * Percent or byte-derived percent for file-style model pulls (Whisper, Piper, Stable-Diffusion, etc.).
 */
function formatPullProgressDetail(p: PullProgressFields): string {
    if (p.percent != null && Number.isFinite(p.percent)) {
        return `${Math.round(Math.min(100, Math.max(0, p.percent)))}%`
    }

    return ''
}

/**
 * Status line plus percentage when available (avoids duplicating if `status` already contains a %).
 */
/** Copy phase and measured percentage; endpoint names belong to the operation tuple. */
export function formatCopyProgressLabel(p: { status: string; percent?: number }): string {
    const stage = p.status.trim() || 'working'
    return p.percent === undefined
        ? `Copy · ${stage}`
        : `Copy · ${stage} · ${Math.round(p.percent)}%`
}

export function formatPullProgressLabel(p: PullProgressFields): string {
    const status = (p.status ?? '').trim()
    if (/\d+\s*%/.test(status)) return status

    const detail = formatPullProgressDetail(p)

    if (!!detail && !!status) {
        return `${status} · ${detail}`
    }

    if (detail) {
        return detail
    }

    return status || '…'
}

/** Operation phase is independent of inventory readiness and loaded residency. */
export function formatDownloadStatus(status: string, current: boolean | undefined): string {
    if (current !== true) return 'Download status unknown'
    switch (status) {
        case 'queued':
            return 'Download queued'
        case 'in_progress':
            return 'Downloading'
        case 'failed':
            return 'Download failed'
        case 'complete':
            return 'Download complete'
        default:
            return 'Download status unknown'
    }
}

/** Why a download failed, in plain words, from its download-state reason code. */
export function formatDownloadFailure(reason: ModelDownloadEntry['reason']): string {
    switch (reason) {
        case 'busy':
            return 'the engine was busy with another download'
        case 'capacity':
            return 'too many downloads were already running'
        case 'download_failed':
            return 'the files could not be downloaded'
        case 'interrupted':
            return 'it was interrupted before it finished'
        case 'progress_timeout':
            return 'it stopped making progress'
        case 'acceptance_unknown':
            return 'the engine did not confirm it started'
        case 'cleanup_unknown':
            return 'partial files may not have been cleaned up'
        case 'persistence_error':
            return 'its state could not be saved'
        case 'unsupported':
            return 'this engine cannot download that model'
        default:
            return 'no reason was reported'
    }
}

/** A context size as people say it: 65536 -> "64K"; other sizes in full. */
export function formatContextTokens(tokens: number): string {
    return tokens >= 1024 && tokens % 1024 === 0 ? `${tokens / 1024}K` : tokens.toLocaleString('en')
}

/**
 * Short form of a reported service build for compact surfaces: `git:<sha>[+dirty]`
 * becomes the first eight hex digits plus any suffix. Other values are returned
 * unchanged, so nothing is invented for a build that is not a git revision.
 */
export function formatServiceBuildShort(build: string): string {
    const match = /^git:([0-9a-f]{8})[0-9a-f]*(\+.*)?$/.exec(build)
    return match ? `${match[1]}${match[2] ?? ''}` : build
}
