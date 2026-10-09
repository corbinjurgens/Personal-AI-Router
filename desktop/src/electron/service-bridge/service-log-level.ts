// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModularLogLevel } from '@/shared/constants/modular-runtime'

/**
 * The severity a backend log line reports about itself.
 *
 * Every service writes through one handler with a fixed shape:
 *
 *     HH:MM:SS.mmm [component] LEVEL message key=value
 *
 * so the level is stated in the line and does not have to be guessed. It used
 * to be taken from the stream instead, and since the services log everything —
 * including debug traffic — to stderr, that filed the great majority of the
 * debug panel as warnings and left a real warning indistinguishable from a
 * routine one.
 *
 * The pattern is anchored so only the handler's own prefix can match. A message
 * that merely contains the word ERROR is not a claim about its own severity.
 */
const LEVEL_LINE = /^\d{2}:\d{2}:\d{2}\.\d{3} \[[^\]]+] (DEBUG|INFO|WARN|ERROR) /

type ServiceLogLevel = 'verbose' | 'info' | 'warn' | 'error'

/**
 * Classify one line of a service's output.
 *
 * Lines that do not carry the handler's prefix are not ours: a Go runtime
 * panic, a standard-library message, or a third-party tool's output. Nothing in
 * them states a severity, so the stream is the only signal left — unstructured
 * stderr is the shape a crash takes and stays visible at `warn`, while stdout
 * is the JSON-RPC channel and is noise at anything above `verbose`.
 */
export function serviceLogLevel(stream: 'stdout' | 'stderr', text: string): ServiceLogLevel {
    switch (LEVEL_LINE.exec(text)?.[1]) {
        case 'DEBUG':
            return 'verbose'
        case 'INFO':
            return 'info'
        case 'WARN':
            return 'warn'
        case 'ERROR':
            return 'error'
        default:
            return stream === 'stderr' ? 'warn' : 'verbose'
    }
}

const SERVICE_LOG_RANK: Record<ServiceLogLevel, number> = { verbose: 0, info: 1, warn: 2, error: 3 }
const THRESHOLD_RANK: Record<ModularLogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 }

/**
 * Whether a classified service line clears the configured log level and so
 * belongs in the log file.
 *
 * The services already filter their own stderr by this level, and JSON-RPC
 * frames on stdout are not emitted at all below `debug` (see
 * `JsonRpcSubprocess.setProtocolTrace`), so in practice this gates stdout lines
 * that are not protocol frames, which classify as `verbose`.
 */
export function isServiceLogLevelEnabled(
    level: ServiceLogLevel,
    threshold: ModularLogLevel
): boolean {
    return SERVICE_LOG_RANK[level] >= THRESHOLD_RANK[threshold]
}
