// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
    resolveServiceEndpoint,
    SERVICE_ENDPOINT_ENV,
    type DirectoryInfo,
    type ServiceEndpointEnvironment
} from '@/electron/service-bridge/service-endpoint'

function environment(overrides: Partial<ServiceEndpointEnvironment>): ServiceEndpointEnvironment {
    return {
        platform: 'linux',
        env: {},
        homeDir: '/home/ada',
        userName: 'ada',
        uid: 1000,
        lstatDirectory: () => null,
        ...overrides
    }
}

const APP_DIR = 'Nvidia Corporation/Personal AI Router'

describe('resolveServiceEndpoint', () => {
    it('uses a socket under ~/.config on Linux', () => {
        expect(resolveServiceEndpoint(environment({}))).toBe(
            `/home/ada/.config/${APP_DIR}/service.sock`
        )
    })

    it('honours XDG_CONFIG_HOME on Linux', () => {
        expect(resolveServiceEndpoint(environment({ env: { XDG_CONFIG_HOME: '/data/cfg' } }))).toBe(
            `/data/cfg/${APP_DIR}/service.sock`
        )
    })

    it('uses Application Support on macOS and ignores XDG_CONFIG_HOME', () => {
        expect(
            resolveServiceEndpoint(
                environment({
                    platform: 'darwin',
                    homeDir: '/Users/ada',
                    env: { XDG_CONFIG_HOME: '/data/cfg' }
                })
            )
        ).toBe(`/Users/ada/Library/Application Support/${APP_DIR}/service.sock`)
    })

    it('uses a named pipe per user on Windows', () => {
        expect(
            resolveServiceEndpoint(
                environment({
                    platform: 'win32',
                    userName: 'Ada',
                    env: { LOCALAPPDATA: 'C:\\Users\\Ada\\AppData\\Local' }
                })
            )
        ).toBe('\\\\.\\pipe\\nvpair-service-Ada')
    })

    it('drops the domain and replaces characters a pipe name cannot carry', () => {
        const pipe = (userName: string): string =>
            resolveServiceEndpoint(environment({ platform: 'win32', userName }))

        expect(pipe('CORP\\ada.lovelace')).toBe('\\\\.\\pipe\\nvpair-service-ada.lovelace')
        expect(pipe('ada lovelace@corp')).toBe('\\\\.\\pipe\\nvpair-service-ada_lovelace_corp')
        expect(pipe('CORP\\')).toBe('\\\\.\\pipe\\nvpair-service-user')
    })

    it('lets NVPAIR_SERVICE_ENDPOINT override every platform', () => {
        for (const platform of ['linux', 'darwin', 'win32'] as const) {
            expect(
                resolveServiceEndpoint(
                    environment({ platform, env: { [SERVICE_ENDPOINT_ENV]: '/tmp/test.sock' } })
                )
            ).toBe('/tmp/test.sock')
        }
    })

    describe('when the app data path is too long for a socket', () => {
        const longHome = `/home/${'x'.repeat(80)}`
        const privateDir: DirectoryInfo = { isDirectory: true, mode: 0o700, uid: 1000 }

        it('keeps a path of exactly 103 bytes', () => {
            const suffix = `/.config/${APP_DIR}/service.sock`
            const homeDir = '/h'.padEnd(103 - suffix.length, 'h')
            const endpoint = resolveServiceEndpoint(environment({ homeDir }))

            expect(Buffer.byteLength(endpoint)).toBe(103)
            expect(endpoint).toBe(`${homeDir}${suffix}`)
        })

        it('falls back to a per-user directory under /tmp', () => {
            expect(resolveServiceEndpoint(environment({ homeDir: longHome }))).toBe(
                '/tmp/nvpair-1000/service.sock'
            )
        })

        it('uses $TMPDIR, tolerating a trailing slash', () => {
            expect(
                resolveServiceEndpoint(
                    environment({
                        homeDir: longHome,
                        env: { TMPDIR: '/var/folders/ab/T/' },
                        lstatDirectory: () => privateDir
                    })
                )
            ).toBe('/var/folders/ab/T/nvpair-1000/service.sock')
        })

        it('counts bytes, not characters', () => {
            // 40 two-byte characters: under 103 characters in all, over 103 bytes.
            const homeDir = `/${'é'.repeat(40)}`
            const path = `${homeDir}/.config/${APP_DIR}/service.sock`
            expect(path.length).toBeLessThan(103)
            expect(Buffer.byteLength(path)).toBeGreaterThan(103)

            expect(resolveServiceEndpoint(environment({ homeDir }))).toBe(
                '/tmp/nvpair-1000/service.sock'
            )
        })

        it('rejects a directory another user owns or others can open', () => {
            const resolveWith = (info: DirectoryInfo): string =>
                resolveServiceEndpoint(
                    environment({ homeDir: longHome, lstatDirectory: () => info })
                )

            expect(() => resolveWith({ ...privateDir, uid: 0 })).toThrow(/not a private directory/)
            expect(() => resolveWith({ ...privateDir, mode: 0o755 })).toThrow(
                /not a private directory/
            )
            expect(() => resolveWith({ ...privateDir, isDirectory: false })).toThrow(
                /not a private directory/
            )
            expect(resolveWith(privateDir)).toBe('/tmp/nvpair-1000/service.sock')
        })
    })
})
