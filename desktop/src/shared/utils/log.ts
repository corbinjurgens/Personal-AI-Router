// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { existsSync, mkdirSync, renameSync, statSync } from 'fs'
import { LogEntry, LogLevel, StructuredLogger, StructuredLogPayload } from '@/shared/types/log'
import { PathProvider } from '@/electron/path'
import { BoundedLogWriter } from '@/shared/utils/bounded-log-writer'
import { join } from 'path'

// Cap the active log file by size, not by entry count: the backend at debug
// level is high-volume and every entry that fits on disk is wanted. When
// nvpair.jsonl crosses this, it rotates to nvpair.1.jsonl (a single kept generation),
// so the on-disk total stays ~2x this. Rotation is a rename — O(1) even at
// hundreds of MB, no whole-file rewrite.
const MAX_LOG_FILE_BYTES = 1024 * 1024 * 100 // 100MB active (~200MB total with one rotation)
// Lines waiting for the disk. Past this the newest lines are dropped and counted
// rather than letting a stalled disk grow the main process without bound.
const MAX_QUEUED_LOG_BYTES = 1024 * 1024 * 8
const LOG_FILE_NAME = 'nvpair.jsonl'
const ROTATED_LOG_FILE_NAME = 'nvpair.1.jsonl'

let logFilePath = ''
let writer: BoundedLogWriter | null = null

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function errorToPlainObject(err: Error): Record<string, unknown> {
    const out: Record<string, unknown> = {
        name: err.name,
        message: err.message,
        ...(err.stack && { stack: err.stack })
    }
    if (err.cause !== undefined) {
        out.cause = err.cause instanceof Error ? errorToPlainObject(err.cause) : err.cause
    }
    return out
}

function normalizePayloadData(data: unknown): unknown {
    if (data instanceof Error) return errorToPlainObject(data)
    if (Array.isArray(data)) return data.map(normalizePayloadData)
    if (data !== null && typeof data === 'object') {
        const out: Record<string, unknown> = {}
        for (const [k, v] of Object.entries(data)) {
            out[k] = normalizePayloadData(v)
        }
        return out
    }
    return data
}

function formatEntry(entry: LogEntry): string {
    return JSON.stringify(entry) + '\n'
}

/** The single line that stands in for lines the writer had to drop. */
function droppedLinesEntry(count: number): string {
    return formatEntry({
        level: 'warn',
        time: new Date().toISOString(),
        source: 'log',
        sublevel: 'log-writer',
        message: `Dropped ${count} log lines because the log file could not keep up`,
        data: { dropped: count }
    })
}

function writeEntry(scope: string, level: string, payload: StructuredLogPayload): void {
    const now = new Date()
    const normalizedData =
        payload.data !== undefined
            ? (normalizePayloadData(payload.data) as object | unknown[])
            : undefined

    writer?.write(
        formatEntry({
            level,
            time: now.toISOString(),
            source: scope,
            sublevel: payload.sublevel,
            message: payload.message,
            data: normalizedData
        })
    )

    // Console output with scope prefix
    const prefix = `(${scope})`.padEnd(24)
    const msg = payload.message ?? ''
    const dataStr = payload.data ? ` ${JSON.stringify(payload.data)}` : ''
    process.stdout.write(`${now.toLocaleTimeString()} ${prefix} > ${msg}${dataStr}\n`)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function getStructuredLogFilePath(): string {
    return logFilePath
}

/**
 * Wait until every line logged so far is on disk, or `timeoutMs` passes. Quit
 * paths call this before exiting, because queued lines are otherwise lost.
 */
export function flushLogs(timeoutMs: number): Promise<void> {
    const pending = writer?.flush()
    if (!pending) return Promise.resolve()
    return new Promise<void>(resolve => {
        const timer = setTimeout(resolve, timeoutMs)
        void pending.finally(() => {
            clearTimeout(timer)
            resolve()
        })
    })
}

/** Write whatever is still queued, synchronously. For the process `exit` handler. */
export function flushLogsSync(): void {
    writer?.flushSync()
}

export function initFileLogger(paths: PathProvider): void {
    const logDir = join(paths.getUserData(), 'logs')
    mkdirSync(logDir, { recursive: true })

    logFilePath = join(logDir, LOG_FILE_NAME)

    // Migrate old .json -> .jsonl
    const oldPath = logFilePath.replace(/\.jsonl$/, '.json')
    if (existsSync(oldPath) && !existsSync(logFilePath)) {
        try {
            renameSync(oldPath, logFilePath)
        } catch {
            /* best-effort */
        }
    }

    // Seed the running size from disk so an already-oversized file rotates on the
    // first write after launch instead of growing further.
    let initialFileBytes = 0
    try {
        initialFileBytes = statSync(logFilePath).size
    } catch {
        initialFileBytes = 0
    }

    writer = new BoundedLogWriter({
        filePath: logFilePath,
        rotatedFilePath: join(logDir, ROTATED_LOG_FILE_NAME),
        maxFileBytes: MAX_LOG_FILE_BYTES,
        maxQueueBytes: MAX_QUEUED_LOG_BYTES,
        initialFileBytes,
        droppedLine: droppedLinesEntry
    })
}

export function createStructuredLogger(scope: string): StructuredLogger {
    const logWith = (level: LogLevel, payload: StructuredLogPayload) => {
        writeEntry(scope, level, payload)
    }
    return {
        info: p => logWith('info', p),
        warn: p => logWith('warn', p),
        error: p => logWith('error', p),
        verbose: p => logWith('verbose', p),
        debug: p => logWith('debug', p),
        silly: p => logWith('silly', p)
    }
}
