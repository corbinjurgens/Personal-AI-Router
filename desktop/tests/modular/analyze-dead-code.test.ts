// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, join, resolve } from 'node:path'
import { expect, test } from 'vitest'

test('check fails when the only finding is an unlisted dependency', () => {
    const root = mkdtempSync(join(tmpdir(), 'pair-dead-code-'))
    try {
        const desktop = process.cwd()
        symlinkSync(
            join(desktop, 'node_modules'),
            join(root, 'node_modules'),
            process.platform === 'win32' ? 'junction' : 'dir'
        )
        writeFileSync(
            join(root, 'package.json'),
            JSON.stringify({ name: 'fixture', private: true })
        )
        writeFileSync(
            join(root, 'knip.json'),
            JSON.stringify({ entry: ['index.ts'], project: ['index.ts'] })
        )
        writeFileSync(join(root, 'index.ts'), "import 'missing-fixture-package'\n")
        const config = JSON.stringify({ compilerOptions: { noEmit: true }, include: ['index.ts'] })
        writeFileSync(join(root, 'tsconfig.node.json'), config)
        writeFileSync(join(root, 'tsconfig.web.json'), config)

        const result = spawnSync(
            process.execPath,
            [
                join(desktop, 'node_modules/tsx/dist/cli.mjs'),
                resolve(desktop, 'scripts/analyze-dead-code.ts'),
                '--check'
            ],
            {
                cwd: root,
                encoding: 'utf8',
                timeout: 60000,
                killSignal: 'SIGKILL',
                env: {
                    ...process.env,
                    PATH: [join(desktop, 'node_modules/.bin'), process.env.PATH].join(delimiter)
                }
            }
        )
        expect(result.error).toBeUndefined()
        const report = readFileSync(join(root, 'dead-code-report.md'), 'utf8')
        expect(report).toContain('| Unlisted dependencies | 1 |')
        expect(report).toContain('`missing-fixture-package`')
        const counts = [...report.matchAll(/^\| (?!Category|---)(.+?) \| (\d+) \|$/gm)]
        expect(
            counts.filter(([, , count]) => count !== '0').map(([, category]) => category)
        ).toEqual(['Unlisted dependencies'])
        expect(result.status, result.stdout + result.stderr).toBe(1)
    } finally {
        rmSync(root, { recursive: true, force: true })
    }
}, 120000)
