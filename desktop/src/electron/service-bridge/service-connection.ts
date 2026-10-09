// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import net from 'net'
import path from 'path'
import { spawn } from 'child_process'
import type { Duplex } from 'stream'
import getErrorString from '@/shared/utils/get-error-string'
import { createStructuredLogger } from '@/shared/utils/log'

const log = createStructuredLogger('service-bridge')

/** How long to keep dialing a service that was just started. */
const START_TIMEOUT_MS = 10_000
/** Pause between dial attempts while the service starts. */
const DIAL_RETRY_INTERVAL_MS = 100

export interface ConnectOrStartDeps {
    /** Open the endpoint. Rejects when nothing answers. */
    connect: (endpoint: string) => Promise<Duplex>
    /** Start the service in the background. Throws when it cannot be launched. */
    startService: () => void
    sleep: (ms: number) => Promise<void>
    now: () => number
}

interface ConnectOrStartOptions {
    endpoint: string
    timeoutMs?: number
    retryIntervalMs?: number
}

/**
 * Attach to the running service. When nothing answers, start it (detached, via
 * `deps.startService`) and keep dialing for up to `timeoutMs`.
 */
export async function connectOrStart(
    options: ConnectOrStartOptions,
    deps: ConnectOrStartDeps
): Promise<Duplex> {
    const {
        endpoint,
        timeoutMs = START_TIMEOUT_MS,
        retryIntervalMs = DIAL_RETRY_INTERVAL_MS
    } = options
    try {
        return await deps.connect(endpoint)
    } catch {
        /* not running: start it below */
    }

    deps.startService()

    const deadline = deps.now() + timeoutMs
    let lastError: unknown
    for (;;) {
        try {
            return await deps.connect(endpoint)
        } catch (err) {
            lastError = err
        }
        if (deps.now() + retryIntervalMs > deadline) break
        await deps.sleep(retryIntervalMs)
    }
    throw new Error(
        `Started the service but it did not answer on ${endpoint} within ${timeoutMs} ms: ${getErrorString(lastError)}`
    )
}

/** Open a Unix socket or Windows named pipe. */
function connectSocket(endpoint: string): Promise<Duplex> {
    return new Promise((resolve, reject) => {
        const socket = net.connect(endpoint)
        const onError = (err: Error): void => {
            socket.destroy()
            reject(err)
        }
        socket.once('error', onError)
        socket.once('connect', () => {
            socket.removeListener('error', onError)
            resolve(socket)
        })
    })
}

/**
 * Start `nvpair-service` so that it outlives this process: its own session
 * (Unix) or process group with no console (Windows), no stdio, and a working
 * directory beside the binary. `brokerArgs` follow `--` and are passed to the
 * broker. The service exits at once if another copy is already running.
 */
function spawnDetached(binaryPath: string, brokerArgs: readonly string[]): void {
    const args = brokerArgs.length > 0 ? ['--', ...brokerArgs] : []
    const child = spawn(binaryPath, args, {
        cwd: path.dirname(binaryPath),
        detached: true,
        stdio: 'ignore',
        windowsHide: true
    })
    // A launch failure (a missing binary) arrives as an event, not a throw; the
    // dial loop then times out, so the cause is recorded here.
    child.on('error', err => {
        log.error({
            sublevel: 'service',
            message: `Failed to start ${binaryPath}: ${getErrorString(err)}`
        })
    })
    child.unref()
}

/** The production dependencies for {@link connectOrStart}. */
export function serviceConnectionDeps(
    binaryPath: string,
    brokerArgs: readonly string[]
): ConnectOrStartDeps {
    return {
        connect: connectSocket,
        startService: () => spawnDetached(binaryPath, brokerArgs),
        sleep: ms => new Promise(resolve => setTimeout(resolve, ms)),
        now: () => Date.now()
    }
}
