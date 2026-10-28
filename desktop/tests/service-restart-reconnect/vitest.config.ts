// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import path from 'node:path'
import { BroadcastChannel } from 'node:worker_threads'
import { defineConfig } from 'vitest/config'

// Only the Go fixture may run this project. The CLI owns stdin; the single
// worker uses an in-process channel, never another socket or listener.
const channelName = process.env.PAIR_DESKTOP_COORDINATION
if (process.env.PAIR_TEST_DESKTOP_RECONNECT !== '1' || !channelName) {
    throw new Error('Run TestServiceRestartReconnectDesktop with its explicit opt-in')
}
const channel = new BroadcastChannel(channelName)
channel.onmessage = event => {
    if (typeof event.data !== 'string' || !/^stage:[a-z-]+$/.test(event.data)) {
        throw new Error('Invalid desktop worker stage')
    }
    process.stdout.write(`PAIR_DESKTOP ${event.data.slice('stage:'.length)}\n`)
}

let input = ''
const onInput = (chunk: Buffer): void => {
    input += chunk.toString('utf8')
    if (input.length > 256) throw new Error('Desktop coordination input exceeded its bound')
    for (;;) {
        const end = input.indexOf('\n')
        if (end < 0) break
        const command = input.slice(0, end)
        input = input.slice(end + 1)
        if (!['restart', 'push', 'detach', 'finish'].includes(command)) {
            throw new Error('Unexpected desktop fixture command')
        }
        channel.postMessage(`command:${command}`)
    }
}
const onEnd = (): void => channel.postMessage('command:abort')
process.stdin.on('data', onInput)
process.stdin.on('end', onEnd)

export default defineConfig({
    root: process.cwd(),
    resolve: { alias: { '@': path.resolve(process.cwd(), 'src') } },
    server: { middlewareMode: true, hmr: false, ws: false },
    test: {
        include: ['tests/service-restart-reconnect/bridge.test.ts'],
        environment: 'node',
        api: false,
        ui: false,
        watch: false,
        pool: 'threads',
        maxWorkers: 1,
        isolate: false,
        fileParallelism: false,
        testTimeout: 150_000,
        hookTimeout: 15_000,
        reporters: [
            'default',
            {
                onTestRunEnd() {
                    process.stdin.off('data', onInput)
                    process.stdin.off('end', onEnd)
                    process.stdin.pause()
                    channel.close()
                }
            }
        ]
    }
})
