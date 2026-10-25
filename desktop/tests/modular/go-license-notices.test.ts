// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { collectGoModuleEntries } from '../../scripts/collect-go-licenses'
import { licenseSupplements } from '../../scripts/license-supplements'

const list = vi.hoisted(() =>
    vi.fn<
        (
            command: string,
            args: string[],
            options: { env: NodeJS.ProcessEnv }
        ) => Promise<{ stdout: string }>
    >()
)
vi.mock('node:util', () => ({ promisify: () => list }))

const scratch: string[] = []
afterEach(async () => {
    list.mockReset()
    await Promise.all(scratch.splice(0).map(path => rm(path, { recursive: true, force: true })))
})

async function moduleFixture(name = 'github.com/atotto/clipboard', version = 'v0.1.4') {
    const dir = await mkdtemp(join(tmpdir(), 'pair-go-license-'))
    scratch.push(dir)
    list.mockResolvedValue({ stdout: `${name}\t${version}\t${dir}\n` })
    return dir
}

async function clipboardGrant(): Promise<string> {
    const notice = await readFile(resolve('..', 'THIRD_PARTY_NOTICES.md'), 'utf8')
    const section = notice.split('### github.com/atotto/clipboard v0.1.4\n')[1]
    const text = section?.match(/```text\n([\s\S]*?)\n```/)?.[1]
    if (!text) throw new Error('Missing clipboard license fixture')
    return text
}

describe('Go notice license sources', () => {
    it('accepts the complete version-bound grant and forces offline enumeration for every target', async () => {
        const dir = await moduleFixture()
        const text = await clipboardGrant()
        await writeFile(join(dir, 'LICENSE'), text)
        const entries = await collectGoModuleEntries('/virtual/services', ['fixture'])
        expect(entries).toEqual([
            {
                name: 'github.com/atotto/clipboard',
                version: 'v0.1.4',
                license: 'BSD-3-Clause',
                author: null,
                email: null,
                repository: 'https://github.com/atotto/clipboard',
                homepage: null,
                licenseText: text
            }
        ])
        expect(list).toHaveBeenCalled()
        for (const [command, args, options] of list.mock.calls) {
            expect(command).toBe('go')
            expect(args[0]).toBe('list')
            expect(options.env).toMatchObject({
                GOPROXY: 'off',
                GONOPROXY: 'none',
                GOSUMDB: 'off',
                GOTOOLCHAIN: 'local'
            })
        }
    })

    it('rejects the independent review truncated-grant specimen', async () => {
        const dir = await moduleFixture()
        await writeFile(join(dir, 'LICENSE'), 'MIT License\nCopyright fixture\n')
        await expect(collectGoModuleEntries('/virtual/services', ['fixture'])).rejects.toThrow(
            'Missing, incomplete or unreviewed version-bound Go license'
        )
    })

    it('rejects missing license text instead of emitting a placeholder', async () => {
        await moduleFixture()
        await expect(collectGoModuleEntries('/virtual/services', ['fixture'])).rejects.toThrow(
            'Missing, incomplete or unreviewed version-bound Go license'
        )
    })

    it('rejects a changed grant at the same version', async () => {
        const dir = await moduleFixture()
        await writeFile(join(dir, 'LICENSE'), `${await clipboardGrant()}\nchanged`)
        await expect(collectGoModuleEntries('/virtual/services', ['fixture'])).rejects.toThrow(
            'Missing, incomplete or unreviewed version-bound Go license'
        )
    })

    it('retains NOTICE alongside LICENSE and rejects its removal', async () => {
        const dir = await moduleFixture('gopkg.in/yaml.v3', 'v3.0.1')
        const notices = await readFile(resolve('..', 'THIRD_PARTY_NOTICES.md'), 'utf8')
        const text = notices
            .split('### gopkg.in/yaml.v3 v3.0.1\n')[1]
            ?.match(/```text\n([\s\S]*?)\n```/)?.[1]
        if (!text) throw new Error('Missing yaml license fixture')
        const boundary = text.lastIndexOf('Copyright 2011-2016 Canonical Ltd.')
        expect(boundary).toBeGreaterThan(0)
        await writeFile(join(dir, 'LICENSE'), text.slice(0, boundary).trim())
        await writeFile(join(dir, 'NOTICE'), text.slice(boundary).trim())
        const entries = await collectGoModuleEntries('/virtual/services', ['fixture'])
        expect(entries[0].licenseText).toBe(text)
        await rm(join(dir, 'NOTICE'))
        await expect(collectGoModuleEntries('/virtual/services', ['fixture'])).rejects.toThrow(
            'Missing, incomplete or unreviewed version-bound Go license'
        )
    })

    it('does not reuse a license fingerprint after an upgrade', async () => {
        const dir = await moduleFixture('github.com/atotto/clipboard', 'v99.0.0')
        await writeFile(join(dir, 'LICENSE'), await clipboardGrant())
        await expect(collectGoModuleEntries('/virtual/services', ['fixture'])).rejects.toThrow(
            'Missing, incomplete or unreviewed version-bound Go license'
        )
    })

    it('uses the declared MIT standard text when the exact module version has no LICENSE', async () => {
        await moduleFixture('github.com/mattn/go-localereader', 'v0.0.1')
        const supplement = licenseSupplements['github.com/mattn/go-localereader@v0.0.1']
        const entries = await collectGoModuleEntries('/virtual/services', ['fixture'])
        expect(entries).toEqual([
            {
                name: 'github.com/mattn/go-localereader',
                version: 'v0.0.1',
                license: 'MIT',
                author: null,
                email: null,
                repository: 'https://github.com/mattn/go-localereader',
                homepage: null,
                licenseText: supplement.text.trim(),
                licenseSource: `${supplement.association}; ${supplement.source}`
            }
        ])
    })
})
