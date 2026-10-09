// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { JsonRpcSubprocess } from '@/electron/service-bridge/json-rpc-subprocess'

const FAKE_RPC = `
const readline = require('readline').createInterface({ input: process.stdin })
readline.on('line', line => {
  const request = JSON.parse(line)
  const response = request.method === 'reject'
    ? { jsonrpc: '2.0', id: request.id, error: { code: -32000, message: 'rejected' } }
    : { jsonrpc: '2.0', id: request.id, result: null }
  setTimeout(() => process.stdout.write(JSON.stringify(response) + '\\n'), 10)
})
`

// Exits as soon as it is asked to shut down, so `exit` has already fired by the
// time stop() starts waiting for it.
const FAKE_EARLY_EXIT = `
const readline = require('readline').createInterface({ input: process.stdin })
readline.on('line', () => process.exit(0))
`

// Prints one protocol frame carrying a pairing PIN and one line that is not a
// frame, then a notification, the first time it is asked anything. Exits on the
// next message, which is the shutdown call.
const FAKE_CHATTY = `
const readline = require('readline').createInterface({ input: process.stdin })
let spoke = false
readline.on('line', () => {
  if (spoke) process.exit(0)
  spoke = true
  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', method: 'cluster:invite-received', params: { pin: '481920' } }) + '\\n')
  process.stdout.write('panic: not a frame\\n')
  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', method: 'done' }) + '\\n')
})
`

async function stdoutLogsFor(trace: boolean): Promise<{ texts: string[]; methods: string[] }> {
    const rpc = new JsonRpcSubprocess('fake', process.execPath)
    rpc.setProtocolTrace(trace)
    const texts: string[] = []
    const methods: string[] = []
    rpc.on('log', entry => {
        if (entry.stream === 'stdout') texts.push(entry.text)
    })
    const done = new Promise<void>(resolve => {
        rpc.on('notification', notification => {
            methods.push(notification.method)
            if (notification.method === 'done') resolve()
        })
    })
    rpc.start(['-e', FAKE_CHATTY])
    try {
        await rpc.notify('go')
        await done
    } finally {
        await rpc.stop()
    }
    return { texts, methods }
}

// Below debug, protocol frames used to be redacted and buffered for the debug
// panel only to be filtered out of the file afterwards. They are now not
// emitted at all unless tracing, while still being handled.
describe('JsonRpcSubprocess stdout logging', () => {
    it('emits no protocol frames when not tracing, but still handles them', async () => {
        const { texts, methods } = await stdoutLogsFor(false)

        expect(texts).toEqual(['panic: not a frame'])
        expect(methods).toEqual(['cluster:invite-received', 'done'])
    })

    it('emits every frame redacted when tracing', async () => {
        const { texts } = await stdoutLogsFor(true)

        expect(texts).toHaveLength(3)
        expect(texts.join('\n')).not.toContain('481920')
        expect(texts[0]).toContain('[redacted]')
        expect(texts[1]).toBe('panic: not a frame')
    })
})

describe('JsonRpcSubprocess.sendWithResponse', () => {
    it('observes delayed responses without imposing a short timeout', async () => {
        const rpc = new JsonRpcSubprocess('fake', process.execPath)
        rpc.start(['-e', FAKE_RPC])
        try {
            await expect(rpc.sendWithResponse('reject')).rejects.toThrow('-32000: rejected')
            await expect(rpc.sendWithResponse('accept')).resolves.toBeUndefined()
        } finally {
            await rpc.stop()
        }
    })
})

describe('JsonRpcSubprocess.stop', () => {
    // A child that dies during the shutdown call leaves no 'exit' left to observe,
    // so a listener attached afterwards never fires: the wait then burns the whole
    // grace and reports the delay as a blocked main process, blaming the wrong
    // side for a child that simply left early.
    it('returns promptly when the child exits while the shutdown call is in flight', async () => {
        const rpc = new JsonRpcSubprocess('fake', process.execPath)
        rpc.start(['-e', FAKE_EARLY_EXIT])

        const startedAt = Date.now()
        await rpc.stop(15_000)

        expect(Date.now() - startedAt).toBeLessThan(4_000)
    })

    it('returns promptly when the child is already gone', async () => {
        const rpc = new JsonRpcSubprocess('fake', process.execPath)
        rpc.start(['-e', FAKE_EARLY_EXIT])

        const exited = new Promise<void>(resolve => rpc.once('exit', () => resolve()))
        await rpc.notify('anything')
        await exited

        const startedAt = Date.now()
        await rpc.stop(15_000)

        expect(Date.now() - startedAt).toBeLessThan(2_000)
        expect(rpc.running).toBe(false)
    })
})
