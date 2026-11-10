// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The one source of the macOS privileged helper's code-signing requirements.
 *
 * `build-helper.ts` compiles the requirement strings produced here into both
 * helper binaries (generated `HelperSigning.swift`) and records the signer it
 * built for in `native/build/helper-signing.json`. `sign-local.ts` reads that
 * record so a local signature can only be applied by an identity of the same
 * team and signing class the helper was built to trust.
 *
 * The signer is a build input: `PAIR_MAC_SIGNING_TEAM` and
 * `PAIR_MAC_SIGNING_CLASS` are set together or not at all. Unset, the helper
 * trusts NVIDIA's Developer ID team, which is what the official build signs with.
 */

import { readFileSync } from 'node:fs'
import path from 'node:path'
import { APP_ID } from '@/shared/constants/app'

export const HELPER_BUILD_DIR = path.resolve(__dirname, '..', '..', '..', 'native', 'build')
export const HELPER_SIGNING_RECORD = path.join(HELPER_BUILD_DIR, 'helper-signing.json')

/**
 * Signing identifiers of the two helper Mach-O files. They are also the file
 * names, but the identifier is passed explicitly when signing: codesign's
 * default strips a dotted "extension" from the file name, which turns
 * `com.nvidia.nvpair.helper` into `com.nvidia.nvpair`.
 */
export const DAEMON_IDENTIFIER = 'com.nvidia.nvpair.helper'
export const CTL_IDENTIFIER = 'nvpair-helper-ctl'

type SigningClass = 'developer-id' | 'apple-development'

export interface HelperSigner {
    team: string
    signingClass: SigningClass
}

/**
 * Apple certificate-extension markers per signing class (verified with
 * `codesign --verify -R`): the intermediate CA marker on certificate 1 and the
 * leaf certificate-type marker. Developer ID: Developer ID CA (6.2.6) and
 * Developer ID Application (6.1.13). Apple Development: Worldwide Developer
 * Relations CA (6.2.1) and Mac Development (6.1.12).
 */
const CLASS_MARKERS: Record<SigningClass, { intermediate: string; leaf: string }> = {
    'developer-id': {
        intermediate: '1.2.840.113635.100.6.2.6',
        leaf: '1.2.840.113635.100.6.1.13'
    },
    'apple-development': {
        intermediate: '1.2.840.113635.100.6.2.1',
        leaf: '1.2.840.113635.100.6.1.12'
    }
}

/** Common-name prefix of a signing certificate of each class. */
export const CLASS_COMMON_NAME_PREFIX: Record<SigningClass, string> = {
    'developer-id': 'Developer ID Application:',
    'apple-development': 'Apple Development:'
}

const NVIDIA_SIGNER: HelperSigner = { team: '6KR3T733EC', signingClass: 'developer-id' }

function isSigningClass(value: string): value is SigningClass {
    return value === 'developer-id' || value === 'apple-development'
}

function parseSigner(team: string, signingClass: string, source: string): HelperSigner {
    if (!/^[A-Z0-9]{10}$/.test(team)) {
        throw new Error(`${source}: team ${JSON.stringify(team)} is not a 10-character Team ID`)
    }
    if (!isSigningClass(signingClass)) {
        throw new Error(
            `${source}: signing class ${JSON.stringify(signingClass)} is not ` +
                `${Object.keys(CLASS_MARKERS).join(' or ')}`
        )
    }
    return { team, signingClass }
}

export function signerFromEnv(env: NodeJS.ProcessEnv): HelperSigner {
    const team = env['PAIR_MAC_SIGNING_TEAM'] ?? ''
    const signingClass = env['PAIR_MAC_SIGNING_CLASS'] ?? ''
    if (!team && !signingClass) return NVIDIA_SIGNER
    if (!team || !signingClass) {
        throw new Error(
            'Set PAIR_MAC_SIGNING_TEAM and PAIR_MAC_SIGNING_CLASS together, or neither.'
        )
    }
    return parseSigner(team, signingClass, 'PAIR_MAC_SIGNING_TEAM/PAIR_MAC_SIGNING_CLASS')
}

export function readSigningRecord(): HelperSigner {
    const parsed: { team?: string; signingClass?: string } = JSON.parse(
        readFileSync(HELPER_SIGNING_RECORD, 'utf8')
    )
    return parseSigner(parsed.team ?? '', parsed.signingClass ?? '', HELPER_SIGNING_RECORD)
}

/**
 * Requirement strings pinned by the helper. Every one carries the signer's
 * anchor: Apple-issued chain, the class's intermediate and leaf markers, and
 * the leaf's Team ID OU. Identifier pins name the exact peer.
 */
export function helperRequirements(signer: HelperSigner): {
    signedBinary: string
    ctl: string
    daemon: string
    app: string
} {
    const markers = CLASS_MARKERS[signer.signingClass]
    const anchor =
        'anchor apple generic ' +
        `and certificate 1[field.${markers.intermediate}] exists ` +
        `and certificate leaf[field.${markers.leaf}] exists ` +
        `and certificate leaf[subject.OU] = "${signer.team}"`
    return {
        signedBinary: anchor,
        ctl: `identifier "${CTL_IDENTIFIER}" and ${anchor}`,
        daemon: `identifier "${DAEMON_IDENTIFIER}" and ${anchor}`,
        app: `identifier "${APP_ID}" and ${anchor}`
    }
}
