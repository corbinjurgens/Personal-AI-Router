// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { app, BrowserWindow } from 'electron'
import log from 'electron-log'
import electronUpdater from 'electron-updater'
import fs from 'node:fs'
import path from 'node:path'
import type { UpdateStatus } from '@/shared/types/update'
import pkg from '../../../package.json'
import getErrorString from '@/shared/utils/get-error-string'
import { notifyUpdateAvailable } from '@/electron/updater/update-notification'
import {
    type UpdateOperation,
    userFacingUpdateError
} from '@/electron/updater/update-error-message'
import {
    clearPendingUpdate,
    readPendingUpdateVersion,
    writePendingUpdateVersion
} from '@/electron/updater/pending-update-store'

const { autoUpdater } = electronUpdater
// The application error listener owns the sanitized, deduplicated report.
// electron-updater's built-in error listener otherwise logs the same raw stack.
autoUpdater.logger = null
const UNCONFIGURED = 'Distribution updates are not configured for this build'

function distributionEligibility(): string | null {
    if (!app.isPackaged) return UNCONFIGURED
    const configuration = path.join(process.resourcesPath, 'app-update.yml')
    try {
        fs.statSync(configuration)
        return null
    } catch (error) {
        if (error instanceof Error && 'code' in error && error.code === 'ENOENT')
            return UNCONFIGURED
        return 'Distribution update configuration is invalid or unreadable. Check resources/app-update.yml.'
    }
}

function allowExplicitDistribution(operation: UpdateOperation): boolean {
    const error = distributionEligibility()
    if (!error) {
        autoUpdater.autoDownload = false
        autoUpdater.autoInstallOnAppQuit = false
        return true
    }
    if (error === UNCONFIGURED) setStatus({ phase: 'error', error })
    else failUpdate(new Error(error), operation, true)
    return false
}

let status: UpdateStatus = {
    phase: 'idle',
    currentVersion: pkg.version,
    latestVersion: null,
    downloadPercent: null,
    error: null
}

let activeUpdateOperation: UpdateOperation | null = null
let initialized = false

/**
 * Version of an update already downloaded to disk in a prior session (loaded
 * from the pending-update marker). When the next check reports this same
 * version is available, we re-arm electron-updater from its cache so the
 * downloaded state reappears without a fresh download. Applying it requires
 * a future verified-package adapter to the build lifecycle.
 */
let pendingDownloadedVersion: string | null = null

function failUpdate(err: Error, operation: UpdateOperation, configurationError = false): void {
    const code = 'code' in err ? err.code : ''
    const configError =
        configurationError ||
        err.name === 'YAMLException' ||
        err.name === 'TypeError' ||
        [
            'EACCES',
            'EPERM',
            'ENOENT',
            'ENOTDIR',
            'EISDIR',
            'ERR_INVALID_URL',
            'ERR_UPDATER_INVALID_PROVIDER_CONFIGURATION',
            'ERR_UPDATER_UNSUPPORTED_PROVIDER'
        ].includes(typeof code === 'string' ? code : '')
    const message = configError
        ? 'Distribution update configuration is invalid or unreadable. Check resources/app-update.yml.'
        : userFacingUpdateError(operation)
    if (configError) {
        if (startupCheckTimer) clearTimeout(startupCheckTimer)
        if (periodicCheckTimer) clearInterval(periodicCheckTimer)
        startupCheckTimer = null
        periodicCheckTimer = null
    }
    if (status.phase === 'error' && status.error === message && activeUpdateOperation === null)
        return
    log.error(`[updater] ${message}`)
    activeUpdateOperation = null
    setStatus({
        phase: 'error',
        error: message
    })
}

function setStatus(patch: Partial<UpdateStatus>): void {
    const prev = status
    status = { ...status, ...patch }
    if (
        status.phase === 'available' &&
        status.latestVersion &&
        status.latestVersion !== pendingDownloadedVersion &&
        (prev.phase !== 'available' || prev.latestVersion !== status.latestVersion)
    ) {
        notifyUpdateAvailable(status.latestVersion)
    }
    broadcastStatus()
}

function broadcastStatus(): void {
    for (const win of BrowserWindow.getAllWindows()) {
        if (!win.isDestroyed()) {
            win.webContents.send('update:status', status)
        }
    }
}

export function getUpdateStatus(): UpdateStatus {
    return status
}

const STARTUP_UPDATE_CHECK_DELAY_MS = 5000
const UPDATE_CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000
let startupCheckTimer: ReturnType<typeof setTimeout> | null = null

function scheduleStartupUpdateCheck(): void {
    if (distributionEligibility() !== null || startupCheckTimer) return

    startupCheckTimer = setTimeout(() => {
        startupCheckTimer = null
        if (distributionEligibility() === UNCONFIGURED) return
        void checkForUpdates()
    }, STARTUP_UPDATE_CHECK_DELAY_MS)
}

let periodicCheckTimer: ReturnType<typeof setInterval> | null = null

function startPeriodicUpdateCheck(): void {
    if (distributionEligibility() !== null) return
    if (periodicCheckTimer) return

    periodicCheckTimer = setInterval(() => {
        if (distributionEligibility() === UNCONFIGURED) return
        // Never clobber an in-progress or ready-to-install state.
        if (
            status.phase === 'checking' ||
            status.phase === 'downloading' ||
            status.phase === 'downloaded'
        ) {
            return
        }
        void checkForUpdates()
    }, UPDATE_CHECK_INTERVAL_MS)
}

/**
 * Loads the pending-update marker. If it names the version we are already
 * running, the update was installed since it was written, so the marker is
 * stale and cleared. Otherwise it primes `pendingDownloadedVersion` so the
 * next `update-available` for that version re-arms the cached download.
 */
function loadPendingDownloadedVersion(): void {
    const marker = readPendingUpdateVersion()
    if (!marker) return
    if (marker === pkg.version) {
        clearPendingUpdate()
        return
    }
    pendingDownloadedVersion = marker
}

export function initializeUpdater(): void {
    if (initialized) return
    const eligibility = distributionEligibility()
    if (eligibility !== null) {
        if (eligibility !== UNCONFIGURED) {
            failUpdate(new Error(eligibility), 'check', true)
        }
        return
    }
    initialized = true

    loadPendingDownloadedVersion()

    autoUpdater.autoDownload = false
    autoUpdater.autoInstallOnAppQuit = false

    autoUpdater.on('checking-for-update', () => {
        setStatus({ phase: 'checking', error: null })
    })
    autoUpdater.on('update-available', info => {
        activeUpdateOperation = null
        setStatus({
            phase: 'available',
            latestVersion: info.version,
            downloadPercent: null,
            error: null
        })
        // A previously downloaded update is still cached on disk; re-arm
        // electron-updater's downloaded state returns without a real re-download.
        if (pendingDownloadedVersion === info.version) {
            void downloadUpdate()
        }
    })
    autoUpdater.on('update-not-available', info => {
        activeUpdateOperation = null
        setStatus({
            phase: 'not-available',
            latestVersion: info.version,
            downloadPercent: null,
            error: null
        })
    })
    autoUpdater.on('download-progress', progress => {
        setStatus({
            phase: 'downloading',
            downloadPercent: progress.percent
        })
    })
    autoUpdater.on('update-downloaded', info => {
        activeUpdateOperation = null
        pendingDownloadedVersion = info.version
        writePendingUpdateVersion(info.version)
        setStatus({
            phase: 'downloaded',
            latestVersion: info.version,
            downloadPercent: 100,
            error: null
        })
    })
    autoUpdater.on('error', err => {
        const operation = activeUpdateOperation ?? 'check'
        failUpdate(err, operation)
    })

    scheduleStartupUpdateCheck()
    startPeriodicUpdateCheck()
}

export async function checkForUpdates(): Promise<void> {
    if (!allowExplicitDistribution('check')) return
    activeUpdateOperation = 'check'
    try {
        await autoUpdater.checkForUpdates()
        activeUpdateOperation = null
    } catch (err) {
        failUpdate(err instanceof Error ? err : new Error(getErrorString(err)), 'check')
    }
}

export async function downloadUpdate(): Promise<void> {
    if (!allowExplicitDistribution('download')) return
    // Re-entrancy guard: a second click while a download is in flight is a no-op.
    if (status.phase === 'downloading') return
    activeUpdateOperation = 'download'
    // Emit the downloading state immediately so the UI gives deterministic
    // feedback before electron-updater's first download-progress event.
    setStatus({ phase: 'downloading', downloadPercent: null, error: null })
    try {
        await autoUpdater.downloadUpdate()
        activeUpdateOperation = null
    } catch (err) {
        failUpdate(err instanceof Error ? err : new Error(getErrorString(err)), 'download')
    }
}

export async function quitAndInstallUpdate(): Promise<void> {
    setStatus({
        phase: 'error',
        error: 'Installing downloaded updates is unavailable until verified packages can be applied through the build lifecycle.'
    })
}
