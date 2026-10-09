// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import os from 'os'
import path from 'path'
import { afterEach, describe, expect, it } from 'vitest'
import { BoundedLogWriter } from '@/shared/utils/bounded-log-writer'

const tmpDirs: string[] = []

function createTmpDir(): { dir: string } {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pair-test-log-writer-'))
    tmpDirs.push(dir)
    return { dir }
}

afterEach(() => {
    for (const dir of tmpDirs.splice(0)) fs.rmSync(dir, { recursive: true, force: true })
})

function droppedLine(count: number): string {
    return `dropped ${count}\n`
}

function makeWriter(dir: string, overrides: { maxFileBytes?: number; maxQueueBytes?: number }) {
    const filePath = path.join(dir, 'log.jsonl')
    const rotatedFilePath = path.join(dir, 'log.1.jsonl')
    const writer = new BoundedLogWriter({
        filePath,
        rotatedFilePath,
        maxFileBytes: overrides.maxFileBytes ?? 1024 * 1024,
        maxQueueBytes: overrides.maxQueueBytes ?? 1024 * 1024,
        initialFileBytes: 0,
        droppedLine
    })
    return { writer, filePath, rotatedFilePath }
}

function read(filePath: string): string {
    return fs.existsSync(filePath) ? fs.readFileSync(filePath, 'utf8') : ''
}

// The log used to be written with appendFileSync on every call, which put a
// file operation on the main thread for each line. These pin the replacement's
// guarantees: nothing is written synchronously on the logging path, ordering is
// kept, memory is bounded with the loss reported, rotation still happens, and a
// flush on quit leaves nothing behind.
describe('BoundedLogWriter', () => {
    it('queues on write and appends in order once flushed', async () => {
        const tmp = createTmpDir()
        const { writer, filePath } = makeWriter(tmp.dir, {})

        writer.write('a\n')
        writer.write('b\n')
        writer.write('c\n')
        expect(read(filePath)).toBe('')

        await writer.flush()
        expect(read(filePath)).toBe('a\nb\nc\n')
    })

    it('keeps lines written during an append in order behind it', async () => {
        const tmp = createTmpDir()
        const { writer, filePath } = makeWriter(tmp.dir, {})

        writer.write('first\n')
        await new Promise<void>(resolve => setImmediate(resolve))
        writer.write('second\n')
        await writer.flush()

        expect(read(filePath)).toBe('first\nsecond\n')
    })

    it('drops new lines past the queue cap and reports the count once', async () => {
        const tmp = createTmpDir()
        const { writer, filePath } = makeWriter(tmp.dir, { maxQueueBytes: 8 })

        writer.write('1234\n')
        writer.write('5678\n')
        writer.write('9abc\n')
        await writer.flush()

        expect(read(filePath)).toBe('1234\ndropped 2\n')

        writer.write('next\n')
        await writer.flush()
        expect(read(filePath)).toBe('1234\ndropped 2\nnext\n')
    })

    it('rotates the file once it passes the size cap', async () => {
        const tmp = createTmpDir()
        const { writer, filePath, rotatedFilePath } = makeWriter(tmp.dir, { maxFileBytes: 10 })

        writer.write('0123456789\n')
        await writer.flush()
        expect(read(rotatedFilePath)).toBe('0123456789\n')
        expect(read(filePath)).toBe('')

        writer.write('after\n')
        await writer.flush()
        expect(read(filePath)).toBe('after\n')
    })

    it('rotates an oversized file left from a previous run on the first write', async () => {
        const tmp = createTmpDir()
        const filePath = path.join(tmp.dir, 'log.jsonl')
        const rotatedFilePath = path.join(tmp.dir, 'log.1.jsonl')
        fs.writeFileSync(filePath, 'old run\n')
        const writer = new BoundedLogWriter({
            filePath,
            rotatedFilePath,
            maxFileBytes: 10,
            maxQueueBytes: 1024,
            initialFileBytes: fs.statSync(filePath).size,
            droppedLine
        })

        writer.write('new\n')
        await writer.flush()

        expect(read(rotatedFilePath)).toBe('old run\nnew\n')
        expect(read(filePath)).toBe('')
    })

    it('writes everything still queued synchronously with flushSync', () => {
        const tmp = createTmpDir()
        const { writer, filePath } = makeWriter(tmp.dir, { maxQueueBytes: 6 })

        writer.write('last\n')
        writer.write('lost\n')
        writer.flushSync()

        expect(read(filePath)).toBe('last\ndropped 1\n')
    })

    it('does not write a queued line twice when flushSync runs before the drain', async () => {
        const tmp = createTmpDir()
        const { writer, filePath } = makeWriter(tmp.dir, {})

        writer.write('once\n')
        writer.flushSync()
        await writer.flush()

        expect(read(filePath)).toBe('once\n')
    })
})
