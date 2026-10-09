// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { appendFileSync, renameSync } from 'fs'
import { appendFile, rename } from 'fs/promises'

interface BoundedLogWriterOptions {
    filePath: string
    /** Where the active file is renamed once it passes {@link maxFileBytes}. One generation is kept. */
    rotatedFilePath: string
    maxFileBytes: number
    /** Bytes that may wait in memory for the disk. Lines past this are dropped and counted. */
    maxQueueBytes: number
    /** Size of the active file at startup, so an oversized file rotates on the first write. */
    initialFileBytes: number
    /** The line written once in place of every line dropped since the last write. */
    droppedLine: (count: number) => string
}

/**
 * Appends log lines to a file off the main thread's critical path.
 *
 * `write` only queues. Queued lines are appended in batches, one append in
 * flight at a time, so a burst costs one file operation instead of one per
 * line, and a slow disk delays the log instead of the event loop. The queue is
 * capped in bytes: when the disk cannot keep up, new lines are dropped and
 * counted, and the count is written as one line when the writer catches up.
 * Older queued lines are kept because they lead up to whatever caused the burst.
 *
 * The file is opened per batch rather than held open, so rotating it is a plain
 * rename between batches on every platform, including Windows.
 */
export class BoundedLogWriter {
    private readonly options: BoundedLogWriterOptions
    private queue: string[] = []
    private queuedBytes = 0
    private dropped = 0
    private fileBytes: number
    private draining: Promise<void> | null = null

    constructor(options: BoundedLogWriterOptions) {
        this.options = options
        this.fileBytes = options.initialFileBytes
    }

    write(line: string): void {
        const bytes = Buffer.byteLength(line, 'utf8')
        if (this.queuedBytes + bytes > this.options.maxQueueBytes) {
            this.dropped += 1
        } else {
            this.queue.push(line)
            this.queuedBytes += bytes
        }
        this.ensureDraining()
    }

    /** Resolves once every line queued so far has been handed to the file system. */
    async flush(): Promise<void> {
        while (this.draining) await this.draining
    }

    /**
     * Writes whatever is still queued, synchronously. For exit paths that cannot
     * wait. An async append already in flight is not waited for, so its lines can
     * land after these.
     */
    flushSync(): void {
        if (!this.hasPending()) return
        const batch = this.takeBatch()
        try {
            appendFileSync(this.options.filePath, batch, 'utf8')
            this.fileBytes += Buffer.byteLength(batch, 'utf8')
            if (this.fileBytes > this.options.maxFileBytes) {
                renameSync(this.options.filePath, this.options.rotatedFilePath)
                this.fileBytes = 0
            }
        } catch {
            /* best-effort */
        }
    }

    private hasPending(): boolean {
        return this.queue.length > 0 || this.dropped > 0
    }

    private ensureDraining(): void {
        if (this.draining) return
        // Deferred a turn so lines logged together in one tick share a batch.
        this.draining = new Promise<void>(resolve => setImmediate(resolve))
            .then(() => this.drain())
            .finally(() => {
                this.draining = null
                // A line queued after the drain's last check but before this ran
                // would otherwise wait for the next write.
                if (this.hasPending()) this.ensureDraining()
            })
    }

    private async drain(): Promise<void> {
        while (this.hasPending()) {
            const batch = this.takeBatch()
            try {
                await appendFile(this.options.filePath, batch, 'utf8')
                this.fileBytes += Buffer.byteLength(batch, 'utf8')
                if (this.fileBytes > this.options.maxFileBytes) {
                    // Overwrites any previous rotated generation; only one is kept.
                    await rename(this.options.filePath, this.options.rotatedFilePath)
                    this.fileBytes = 0
                }
            } catch {
                /* best-effort: a failed batch is lost and the next one is tried */
            }
        }
    }

    /**
     * Everything queued, then the drop notice. Lines are only dropped while the
     * queue is full, so every queued line predates every dropped one.
     */
    private takeBatch(): string {
        const notice = this.dropped > 0 ? this.options.droppedLine(this.dropped) : ''
        const batch = this.queue.join('') + notice
        this.queue = []
        this.queuedBytes = 0
        this.dropped = 0
        return batch
    }
}
