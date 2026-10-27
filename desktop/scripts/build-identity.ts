// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync } from 'node:child_process'

/** Build-time GUI source stamp only. It never supplies a Go binary's missing VCS. */
export function buildIdentity(directory: string): string {
    const git = (args: string[]): string =>
        execFileSync('git', args, {
            cwd: directory,
            encoding: 'utf8',
            maxBuffer: 64 * 1024,
            timeout: 10000,
            windowsHide: true,
            stdio: ['ignore', 'pipe', 'ignore']
        }).trim()
    try {
        const revision = git(['rev-parse', 'HEAD'])
        if (!/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/.test(revision)) return ''
        const status = git(['status', '--porcelain', '--untracked-files=all'])
        // electron-vite writes this temporary bundled config in desktop's root.
        const dirty = status
            .split('\n')
            .some(
                line =>
                    line !== '' && !/^\?\? desktop\/electron\.vite\.config\.[0-9]+\.mjs$/.test(line)
            )
        return `git:${revision}${dirty ? '+dirty' : ''}`
    } catch {
        return ''
    }
}
