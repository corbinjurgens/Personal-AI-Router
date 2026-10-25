// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { init, type ModuleInfos } from 'license-checker-rseidelsohn'
import { afterEach, describe, expect, it } from 'vitest'
import { npmLicenseText, validateNpmLicenseText } from '../../scripts/npm-license-text'
import { licenseSupplements } from '../../scripts/license-supplements'

const scratch: string[] = []
const mit = licenseSupplements['@radix-ui/number@1.1.1'].text

afterEach(async () => {
    await Promise.all(scratch.splice(0).map(path => rm(path, { recursive: true, force: true })))
})

function scan(start: string): Promise<ModuleInfos> {
    return new Promise((resolveScan, reject) => {
        init(
            {
                start,
                production: true,
                excludePrivatePackages: true,
                customFormat: { licenseText: '' }
            },
            (error, modules) => {
                if (error) reject(error)
                else resolveScan(modules)
            }
        )
    })
}

async function fixture(name = '@radix-ui/primitive', version = '1.1.3', license = 'MIT') {
    const root = await mkdtemp(join(tmpdir(), 'pair-license-notices-'))
    scratch.push(root)
    const path = join(root, 'node_modules', name)
    await mkdir(path, { recursive: true })
    await writeFile(
        join(root, 'package.json'),
        JSON.stringify({
            name: 'fixture-root',
            version: '1.0.0',
            private: true,
            dependencies: { [name]: version }
        })
    )
    await writeFile(join(path, 'package.json'), JSON.stringify({ name, version, license }))
    await writeFile(join(path, 'README.md'), '# Usage\n\nMIT licensed.\n')
    return { root, path, key: `${name}@${version}` }
}

describe('npm notice license sources', () => {
    it('reads the full installed LICENSE rather than trusting scanner text', async () => {
        const { root, path, key } = await fixture()
        await writeFile(join(path, 'LICENSE'), mit)
        const modules = await scan(root)
        modules[key].licenseText = 'MIT'
        await expect(npmLicenseText(key, modules[key])).resolves.toEqual({
            text: mit.trim(),
            source: `${key}/LICENSE`
        })
    })

    it('rejects a real scanner README fallback for an unreviewed package', async () => {
        const { root, key } = await fixture()
        const modules = await scan(root)
        expect(modules[key].licenseFile).toMatch(/README\.md$/)
        await expect(npmLicenseText(key, modules[key])).rejects.toThrow(
            'README fallbacks are rejected'
        )
    })

    it('rejects empty and truncated files even when named LICENSE', async () => {
        const { root, path, key } = await fixture()
        for (const text of ['', 'MIT License', mit.slice(0, mit.indexOf('THE SOFTWARE'))]) {
            await writeFile(join(path, 'LICENSE'), text)
            const modules = await scan(root)
            await expect(npmLicenseText(key, { ...modules[key], licenses: 'MIT' })).rejects.toThrow(
                /Incomplete|No package LICENSE|Unreviewed version-bound/
            )
        }
    })

    it('does not reuse an old supplement after a package version changes', async () => {
        const { root, key } = await fixture('overlayscrollbars', '99.0.0')
        const modules = await scan(root)
        await expect(npmLicenseText(key, modules[key])).rejects.toThrow(
            'add a verified version-bound offline supplement'
        )
    })

    it('rejects a changed grant even when all clause markers remain', async () => {
        const { root, path, key } = await fixture()
        const changed = mit.replace(
            'to use, copy, modify, merge, publish, distribute, sublicense, and/or sell',
            'to use'
        )
        expect(changed).not.toBe(mit)
        await writeFile(join(path, 'LICENSE'), changed)
        const modules = await scan(root)
        await expect(npmLicenseText(key, modules[key])).rejects.toThrow(
            'Unreviewed version-bound package LICENSE'
        )
    })

    it('requires review of a new package version even if its LICENSE is unchanged', async () => {
        const { root, path, key } = await fixture('@radix-ui/primitive', '99.0.0')
        await writeFile(join(path, 'LICENSE'), mit)
        const modules = await scan(root)
        await expect(npmLicenseText(key, modules[key])).rejects.toThrow(
            'Unreviewed version-bound package LICENSE'
        )
    })

    it('the real generator exits before Go collection and preserves the notice on invalid input', async () => {
        const { root, path } = await fixture()
        await writeFile(join(path, 'LICENSE'), 'MIT License\nCopyright fixture\n')
        const desktop = join(root, 'desktop')
        const scripts = join(desktop, 'scripts')
        await mkdir(scripts, { recursive: true })
        await writeFile(
            join(scripts, 'generate-licenses.ts'),
            await readFile(resolve('scripts/generate-licenses.ts'), 'utf8')
        )
        for (const file of [
            'collect-go-licenses.ts',
            'npm-license-text.ts',
            'license-supplements.ts',
            'npm-license-file-hashes.ts',
            'go-license-file-hashes.ts'
        ]) {
            await writeFile(join(scripts, file), await readFile(resolve('scripts', file), 'utf8'))
        }
        await symlink(
            resolve('node_modules/license-checker-rseidelsohn'),
            join(root, 'node_modules/license-checker-rseidelsohn'),
            'junction'
        )
        await symlink(join(root, 'node_modules'), join(desktop, 'node_modules'), 'junction')
        await writeFile(
            join(desktop, 'package.json'),
            await readFile(join(root, 'package.json'), 'utf8')
        )
        await writeFile(join(root, 'THIRD_PARTY_NOTICES.md'), 'existing notice\n')
        const result = spawnSync(
            process.execPath,
            [
                resolve('node_modules/tsx/dist/cli.mjs'),
                join(desktop, 'scripts/generate-licenses.ts')
            ],
            { cwd: resolve('.'), encoding: 'utf8', timeout: 10_000 }
        )
        expect(result.error).toBeUndefined()
        expect(result.status).toBe(1)
        expect(result.stderr).toContain('Unreviewed version-bound package LICENSE')
        expect(result.stderr).not.toContain('go list')
        expect(await readFile(join(root, 'THIRD_PARTY_NOTICES.md'), 'utf8')).toBe(
            'existing notice\n'
        )
    })

    it('rejects mismatched installed package identity', async () => {
        const { root, key } = await fixture()
        const modules = await scan(root)
        await expect(npmLicenseText('fixture-license@2.0.0', modules[key])).rejects.toThrow(
            'identity does not match'
        )
    })

    it('rejects scanner/declaration disagreement', async () => {
        const { root, key } = await fixture()
        const modules = await scan(root)
        await expect(npmLicenseText(key, { ...modules[key], licenses: 'ISC' })).rejects.toThrow(
            'Declared license does not match'
        )
    })

    it('rejects a supplement for a changed declared license', async () => {
        const { root, key } = await fixture('overlayscrollbars', '2.15.1', 'ISC')
        const modules = await scan(root)
        await expect(npmLicenseText(key, modules[key])).rejects.toThrow(
            'Invalid version-bound license supplement'
        )
    })

    it('rejects corrupted supplement bytes', async () => {
        const { root, key } = await fixture('overlayscrollbars', '2.15.1')
        const modules = await scan(root)
        const supplement = licenseSupplements[key]
        const original = supplement.text
        try {
            supplement.text = `${original}\nchanged`
            await expect(npmLicenseText(key, modules[key])).rejects.toThrow(
                'Invalid version-bound license supplement'
            )
        } finally {
            supplement.text = original
        }
    })

    it('accepts the complete Apache alternative for a dual-licensed package', async () => {
        const text = await readFile(resolve('node_modules/dompurify/LICENSE'), 'utf8')
        expect(() =>
            validateNpmLicenseText('dompurify@3.4.13', '(MPL-2.0 OR Apache-2.0)', text)
        ).not.toThrow()
    })

    it('rejects an unsupported license family', () => {
        expect(() => validateNpmLicenseText('fixture@1.0.0', 'UNREVIEWED', mit)).toThrow(
            'unreviewed'
        )
    })

    it.each(Object.entries(licenseSupplements).filter(([key]) => !key.startsWith('github.com/')))(
        'verifies the complete offline grant for %s',
        async (key, supplement) => {
            expect(createHash('sha256').update(supplement.text).digest('hex')).toBe(
                supplement.sha256
            )
            expect(() =>
                validateNpmLicenseText(key, supplement.license, supplement.text)
            ).not.toThrow()
            const { root } = await fixture(
                key.slice(0, key.lastIndexOf('@')),
                key.slice(key.lastIndexOf('@') + 1),
                supplement.license
            )
            const modules = await scan(root)
            const result = await npmLicenseText(key, modules[key])
            const notice = supplement.copyrightNotice
            expect(result.text).toBe(
                notice ? `${notice.text}\n\n${supplement.text.trim()}` : supplement.text.trim()
            )
            if (notice) expect(result.source).toContain(notice.source)
            expect(result.source).toContain(supplement.source)
            expect(result.source).toContain(supplement.association)
        }
    )

    it('validates every dependency included by the production scanner', async () => {
        const modules = await scan(resolve('.'))
        const own: { name: string } = JSON.parse(await readFile(resolve('package.json'), 'utf8'))
        for (const [key, info] of Object.entries(modules)) {
            if (key.slice(0, key.lastIndexOf('@')) === own.name) continue
            const result = await npmLicenseText(key, info)
            expect(result.text.length, key).toBeGreaterThan(0)
        }
    })
})
