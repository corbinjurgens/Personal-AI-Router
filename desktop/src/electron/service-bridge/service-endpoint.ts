// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import os from 'os'
import path from 'path'
import { currentPlatform } from '@/shared/utils/platform'
import type { SupportedPlatform } from '@/shared/types/platform'

/**
 * Where `nvpair-service` listens for this user. This mirrors
 * `services/shared/servicectl` (and `services/shared/appdir`) and must agree
 * with it: the service and every client derive the same endpoint independently.
 *
 * - Linux and macOS: `<appdir>/service.sock`, or `$TMPDIR/nvpair-<uid>/service.sock`
 *   when that path is longer than a Unix socket path allows.
 * - Windows: the named pipe `\\.\pipe\nvpair-service-<username>`.
 * - `NVPAIR_SERVICE_ENDPOINT` overrides both.
 */

export const SERVICE_ENDPOINT_ENV = 'NVPAIR_SERVICE_ENDPOINT'

const ORG_DIR = 'Nvidia Corporation'
const APP_DIR = 'Personal AI Router'
const SOCKET_NAME = 'service.sock'
const PIPE_PREFIX = '\\\\.\\pipe\\nvpair-service-'

/** Longest Unix socket path every supported platform accepts (sun_path is 104 bytes on macOS). */
const MAX_SOCKET_PATH_BYTES = 103

/** The facts about a directory needed to decide whether it may hold the socket. */
export interface DirectoryInfo {
    isDirectory: boolean
    /** Permission bits, e.g. 0o700. */
    mode: number
    uid: number
}

export interface ServiceEndpointEnvironment {
    platform: SupportedPlatform
    env: Readonly<Record<string, string | undefined>>
    homeDir: string
    userName: string
    /** The user id on Unix; unused on Windows. */
    uid: number
    /** Describes `dir` without following a symlink, or null when it does not exist. */
    lstatDirectory: (dir: string) => DirectoryInfo | null
}

export function defaultServiceEndpointEnvironment(): ServiceEndpointEnvironment {
    return {
        platform: currentPlatform(),
        env: process.env,
        homeDir: os.homedir(),
        userName: os.userInfo().username,
        uid: typeof process.getuid === 'function' ? process.getuid() : 0,
        lstatDirectory: dir => {
            try {
                const stat = fs.lstatSync(dir)
                return { isDirectory: stat.isDirectory(), mode: stat.mode & 0o777, uid: stat.uid }
            } catch {
                return null
            }
        }
    }
}

/** Reduce a Windows account name (often DOMAIN\user) to a string a pipe name can carry. */
function pipeSafeUserName(name: string): string {
    const user = name.slice(name.lastIndexOf('\\') + 1)
    const safe = user.replace(/[^A-Za-z0-9._-]/g, '_')
    return safe || 'user'
}

/** Go's `os.UserConfigDir` for the platforms that use a socket. */
function userConfigDir(environment: ServiceEndpointEnvironment): string {
    if (environment.platform === 'darwin') {
        return path.posix.join(environment.homeDir, 'Library', 'Application Support')
    }
    const xdg = environment.env['XDG_CONFIG_HOME']
    return xdg ? xdg : path.posix.join(environment.homeDir, '.config')
}

/**
 * The per-user directory under the system temp dir used when the app data path
 * is too long for a socket. It has to be ours and closed to other users, so a
 * directory another user pre-created there cannot capture the socket.
 */
function shortEndpoint(environment: ServiceEndpointEnvironment): string {
    const tmp = environment.env['TMPDIR'] || '/tmp'
    const dir = path.posix.join(tmp, `nvpair-${environment.uid}`)
    const info = environment.lstatDirectory(dir)
    // A missing directory is fine: nothing listens there yet, and the service
    // creates it (0700) when it starts.
    if (info && (!info.isDirectory || (info.mode & 0o077) !== 0 || info.uid !== environment.uid)) {
        throw new Error(
            `service socket directory ${dir} is not a private directory owned by this user`
        )
    }
    return path.posix.join(dir, SOCKET_NAME)
}

export function resolveServiceEndpoint(environment: ServiceEndpointEnvironment): string {
    const override = environment.env[SERVICE_ENDPOINT_ENV]
    if (override) return override

    if (environment.platform === 'win32') {
        return PIPE_PREFIX + pipeSafeUserName(environment.userName)
    }

    const socket = path.posix.join(userConfigDir(environment), ORG_DIR, APP_DIR, SOCKET_NAME)
    if (Buffer.byteLength(socket) <= MAX_SOCKET_PATH_BYTES) return socket
    return shortEndpoint(environment)
}
