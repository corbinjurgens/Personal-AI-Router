// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { basename, dirname, join, resolve } from 'node:path'
import type { ModuleInfo } from 'license-checker-rseidelsohn'
import { licenseSupplements } from './license-supplements'
import { npmLicenseFileHashes } from './npm-license-file-hashes'

// Fail closed on unreviewed license families and truncated grants. These checks
// complement reading an actual package LICENSE; they never authorize README text.
const requiredClauses: Readonly<Record<string, readonly string[]>> = {
    MIT: [
        'permission is hereby granted, free of charge',
        'to deal in the software without restriction',
        'this permission notice',
        'the software is provided',
        'without warranty of any kind',
        'other dealings in the software'
    ],
    'MIT-0': [
        'permission is hereby granted, free of charge',
        'to deal in the software without restriction',
        'the software is provided',
        'without warranty of any kind',
        'other dealings in the software'
    ],
    ISC: [
        'permission to use, copy, modify',
        'permission notice appear in all copies',
        'the software is provided',
        'the use or performance of this software'
    ],
    '0BSD': [
        'permission to use, copy, modify',
        'the software is provided',
        'the use or performance of this software'
    ],
    'BSD-2-Clause': [
        'redistribution and use in source and binary forms',
        'redistributions of source code',
        'redistributions in binary form',
        'any express or implied warranties',
        'possibility of such damage'
    ],
    'BSD-3-Clause': [
        'redistribution and use in source and binary forms',
        'redistributions of source code',
        'redistributions in binary form',
        'neither the name',
        'this software is provided',
        'possibility of such damage'
    ],
    'Apache-2.0': [
        'apache license',
        'grant of copyright license',
        'grant of patent license',
        'redistribution',
        'disclaimer of warranty',
        'limitation of liability',
        'end of terms and conditions'
    ],
    'Python-2.0': [
        'python software foundation license',
        'psf hereby grants',
        'disclaims any representation or warranty',
        'shall not be liable',
        'license agreement'
    ],
    'MPL-2.0': [
        'mozilla public license',
        'grants',
        'responsibilities',
        'disclaimer of warranty',
        'limitation of liability',
        'exhibit a'
    ],
    'BlueOak-1.0.0': [
        'blue oak model license',
        'each contributor licenses you',
        'copyright in it',
        'patent claims',
        'reliability',
        'excuse',
        'no liability',
        'under any kind of legal claim'
    ],
    'CC0-1.0': [
        'cc0 1.0 universal',
        'waiver',
        'public license fallback',
        'limitations and disclaimers'
    ]
}

export function validateNpmLicenseText(key: string, license: string, text: string): void {
    const normalized = text.toLowerCase().replace(/\s+/g, ' ')
    const alternatives = license.replace(/^\((.*)\)$/, '$1').split(' OR ')
    const complete = alternatives.some(id => {
        const clauses = requiredClauses[id]
        return clauses && clauses.every(clause => normalized.includes(clause))
    })
    if (!complete || !/copyright/i.test(text)) {
        throw new Error(`Incomplete or unreviewed ${license} license text for ${key}`)
    }
}

export async function npmLicenseText(
    key: string,
    info: ModuleInfo
): Promise<{ text: string; source: string }> {
    if (!info.path) throw new Error(`Missing installed package path for ${key}`)
    const installed: { name: string; version: string; license?: string } = JSON.parse(
        await readFile(join(info.path, 'package.json'), 'utf8')
    )
    if (`${installed.name}@${installed.version}` !== key) {
        throw new Error(`Installed package identity does not match ${key}`)
    }
    const license = Array.isArray(info.licenses) ? info.licenses.join(' OR ') : info.licenses
    if (!license || installed.license !== license) {
        throw new Error(`Declared license does not match scanner license for ${key}`)
    }
    const supplement = licenseSupplements[key]
    if (supplement) {
        if (
            supplement.license !== license ||
            createHash('sha256').update(supplement.text).digest('hex') !== supplement.sha256
        ) {
            throw new Error(`Invalid version-bound license supplement for ${key}`)
        }
        validateNpmLicenseText(key, license, supplement.text)
        const notice = supplement.copyrightNotice
        return {
            text: notice ? `${notice.text}\n\n${supplement.text.trim()}` : supplement.text.trim(),
            source: `${supplement.association}; ${supplement.source}${notice ? `; copyright notice: ${notice.source}` : ''}`
        }
    }
    const file = info.licenseFile
    if (
        !file ||
        !/^(licen[sc]e|copying)([.-].*)?$/i.test(basename(file)) ||
        resolve(dirname(file)) !== resolve(info.path)
    ) {
        throw new Error(
            `No package LICENSE for ${key}; add a verified version-bound offline supplement (README fallbacks are rejected)`
        )
    }
    const text = await readFile(file, 'utf8')
    if (createHash('sha256').update(text).digest('hex') !== npmLicenseFileHashes[key]) {
        throw new Error(`Unreviewed version-bound package LICENSE for ${key}`)
    }
    validateNpmLicenseText(key, license, text)
    return { text: text.trim(), source: `${key}/${basename(file)}` }
}
