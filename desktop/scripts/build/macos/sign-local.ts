// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Sign a packaged, unsigned macOS `.app` with a local code-signing identity
 * from the login keychain, so its privileged helper's XPC trust checks pass.
 *
 *   npm run sign:mac:local -- --identity <SHA-1 or common name> <path/to/NVIDIA PAIR.app>
 *
 * The helper only trusts the signer it was built for (see helper-signing.ts),
 * so the app must have been built with PAIR_MAC_SIGNING_TEAM and
 * PAIR_MAC_SIGNING_CLASS describing this identity. The script refuses when the
 * identity's team or certificate class differs from `native/build/helper-signing.json`,
 * or when the app's helper binaries are not the ones in `native/build/`.
 *
 * Signing runs inside out, with no `--deep` on the outer bundle (which would
 * re-sign the helpers under codesign's default identifiers):
 *   1. each entry of Contents/Frameworks (Electron frameworks and helper apps);
 *   2. every Mach-O under Contents/Resources (`cli-bin`, `collectlogs`), then the
 *      `cli-bin/manifest.json` sizes and hashes are updated to the signed bytes;
 *   3. the daemon and control tool with explicit identifiers;
 *   4. the outer bundle.
 * It then verifies the bundle and checks every helper requirement against the
 * signed code. No hardened runtime, entitlements or notarization: this is for
 * the owner's own machines only. Identity hashes and names are never printed.
 */

import { spawnSync } from 'node:child_process'
import { createHash, X509Certificate } from 'node:crypto'
import {
    closeSync,
    existsSync,
    openSync,
    readdirSync,
    readFileSync,
    readSync,
    writeFileSync
} from 'node:fs'
import path from 'node:path'
import type { JsonValue } from '@/shared/types/json'
import { currentPlatform } from '@/shared/utils/platform'
import {
    CLASS_COMMON_NAME_PREFIX,
    CTL_IDENTIFIER,
    DAEMON_IDENTIFIER,
    HELPER_BUILD_DIR,
    HELPER_SIGNING_RECORD,
    helperRequirements,
    readSigningRecord
} from './helper-signing'

interface Identity {
    sha1: string
    commonName: string
}

function run(command: string, args: string[]): string {
    const result = spawnSync(command, args, { encoding: 'utf8' })
    if (result.status !== 0) {
        throw new Error(
            `${command} ${args[0] ?? ''} failed (exit ${result.status ?? 'signal'}): ` +
                // Never echo an identity hash codesign may repeat back.
                result.stderr.trim().replace(/\b[0-9A-Fa-f]{40}\b/g, '<identity>')
        )
    }
    return result.stdout
}

function parseArgs(argv: string[]): { identity: string; appPath: string } {
    let identity = ''
    const positional: string[] = []
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i] ?? ''
        if (arg === '--identity') {
            identity = argv[++i] ?? ''
        } else {
            positional.push(arg)
        }
    }
    const appPath = positional.length === 1 ? path.resolve(positional[0] ?? '') : ''
    if (!identity || !appPath.endsWith('.app') || !existsSync(appPath)) {
        throw new Error('Usage: sign-local.ts --identity <SHA-1 or common name> <path/to/App.app>')
    }
    return { identity, appPath }
}

/** Resolve the argument to exactly one valid code-signing identity. */
function resolveIdentity(wanted: string): Identity {
    const listing = run('security', ['find-identity', '-v', '-p', 'codesigning'])
    const matches: Identity[] = []
    for (const match of listing.matchAll(/^\s*\d+\) ([0-9A-F]{40}) "([^"]+)"$/gm)) {
        const sha1 = match[1] ?? ''
        const commonName = match[2] ?? ''
        if (sha1 === wanted.toUpperCase() || commonName === wanted) {
            matches.push({ sha1, commonName })
        }
    }
    const [only] = matches
    if (matches.length !== 1 || !only) {
        throw new Error(
            `Expected exactly one valid code-signing identity matching --identity; found ${matches.length}.`
        )
    }
    return only
}

/** Team ID (leaf subject OU) of the identity's certificate. */
function identityTeam(identity: Identity): string {
    const pem = run('security', ['find-certificate', '-a', '-c', identity.commonName, '-p'])
    for (const block of pem.match(/-----BEGIN CERTIFICATE-----[^-]+-----END CERTIFICATE-----/g) ??
        []) {
        const cert = new X509Certificate(block)
        if (cert.fingerprint.replaceAll(':', '') !== identity.sha1) continue
        const ou = cert.subject.split('\n').find(line => line.startsWith('OU='))
        if (!ou) throw new Error('The identity certificate has no OU (Team ID).')
        return ou.slice('OU='.length)
    }
    throw new Error('The identity certificate was not found in the keychain.')
}

function sha256(file: string): string {
    return createHash('sha256').update(readFileSync(file)).digest('hex')
}

/** The app must carry exactly the helper binaries the signing record describes. */
function assertHelpersFromThisBuild(appPath: string): void {
    for (const name of [DAEMON_IDENTIFIER, CTL_IDENTIFIER]) {
        const packaged = path.join(appPath, 'Contents/MacOS', name)
        if (sha256(packaged) !== sha256(path.join(HELPER_BUILD_DIR, name))) {
            throw new Error(
                `${name} in the app differs from ${HELPER_BUILD_DIR}. Sign an unsigned copy ` +
                    'of the app packaged from this build.'
            )
        }
    }
}

const MACH_O_MAGICS = new Set([0xfeedface, 0xfeedfacf, 0xcefaedfe, 0xcffaedfe, 0xcafebabe])

function isMachO(file: string): boolean {
    const header = Buffer.alloc(4)
    const fd = openSync(file, 'r')
    try {
        return readSync(fd, header, 0, 4, 0) === 4 && MACH_O_MAGICS.has(header.readUInt32BE(0))
    } finally {
        closeSync(fd)
    }
}

function machOFiles(dir: string): string[] {
    const found: string[] = []
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name)
        if (entry.isDirectory()) found.push(...machOFiles(full))
        else if (entry.isFile() && isMachO(full)) found.push(full)
    }
    return found.sort()
}

function isJsonObject(value: JsonValue | undefined): value is { [key: string]: JsonValue } {
    return (
        value !== undefined && value !== null && typeof value === 'object' && !Array.isArray(value)
    )
}

/** Re-record the signed `cli-bin` bytes so the manifest still describes the package. */
function updateCliBinManifest(cliBinDir: string): void {
    const manifestPath = path.join(cliBinDir, 'manifest.json')
    const manifest: JsonValue = JSON.parse(readFileSync(manifestPath, 'utf8'))
    const files = isJsonObject(manifest) ? manifest['files'] : undefined
    if (!Array.isArray(files)) throw new Error(`${manifestPath} has no files list`)
    for (const file of files) {
        const fileName = isJsonObject(file) ? file['fileName'] : undefined
        if (!isJsonObject(file) || typeof fileName !== 'string') {
            throw new Error(`${manifestPath} has a malformed files entry`)
        }
        const bytes = readFileSync(path.join(cliBinDir, fileName))
        file['size'] = bytes.length
        file['sha256'] = createHash('sha256').update(bytes).digest('hex')
    }
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8')
}

function sign(identity: Identity, target: string, extra: string[] = []): void {
    run('codesign', ['--force', '--sign', identity.sha1, ...extra, target])
}

function main(): void {
    if (currentPlatform() !== 'darwin') throw new Error('sign-local.ts only runs on macOS.')
    const { identity: wanted, appPath } = parseArgs(process.argv.slice(2))
    const signer = readSigningRecord()
    const identity = resolveIdentity(wanted)

    const team = identityTeam(identity)
    if (team !== signer.team) {
        throw new Error(
            `The identity's team ${team} differs from the team ${signer.team} the helper ` +
                `was built to trust (${HELPER_SIGNING_RECORD}). Rebuild with ` +
                `PAIR_MAC_SIGNING_TEAM=${team}.`
        )
    }
    if (!identity.commonName.startsWith(CLASS_COMMON_NAME_PREFIX[signer.signingClass])) {
        throw new Error(
            `The identity is not a "${CLASS_COMMON_NAME_PREFIX[signer.signingClass]}" ` +
                `certificate, the ${signer.signingClass} class the helper was built to trust.`
        )
    }
    assertHelpersFromThisBuild(appPath)

    const contents = path.join(appPath, 'Contents')
    const frameworks = path.join(contents, 'Frameworks')
    for (const entry of readdirSync(frameworks).sort()) {
        sign(identity, path.join(frameworks, entry), ['--deep'])
    }
    const resources = path.join(contents, 'Resources')
    for (const file of machOFiles(resources)) sign(identity, file)
    updateCliBinManifest(path.join(resources, 'cli-bin'))

    const daemon = path.join(contents, 'MacOS', DAEMON_IDENTIFIER)
    const ctl = path.join(contents, 'MacOS', CTL_IDENTIFIER)
    sign(identity, daemon, ['--identifier', DAEMON_IDENTIFIER])
    sign(identity, ctl, ['--identifier', CTL_IDENTIFIER])
    sign(identity, appPath)

    run('codesign', ['--verify', '--deep', '--strict', '--all-architectures', appPath])
    const requirements = helperRequirements(signer)
    const checks: Array<[string, string]> = [
        [ctl, requirements.ctl],
        [daemon, requirements.daemon],
        [appPath, requirements.app],
        ...machOFiles(path.join(resources, 'cli-bin')).map((file): [string, string] => [
            file,
            requirements.signedBinary
        ])
    ]
    for (const [target, requirement] of checks) {
        run('codesign', ['--verify', '--strict', `-R=${requirement}`, target])
    }
    console.log(
        `[sign-local] signed ${appPath} for ${signer.signingClass} team ${signer.team}; ` +
            `${checks.length} helper requirement checks passed`
    )
}

main()
