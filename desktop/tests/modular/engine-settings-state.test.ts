// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import {
    useEngineSettingsStore,
    engineSettingsKey,
    watchEngineSettings
} from '@/ui/stores/engine-settings.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import type { EngineSettingsSnapshot, EngineServingConfig } from '@/shared/types/engine-settings'
import type { IEngineApi } from '@/ui/api/engine-api'
import { EngineSettingsSection } from '@/ui/components/BackendRow/EngineSettingsSection'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { redactSensitiveLogText } from '@/electron/redact-log'

const target = { nodeId: 'remote', engine: 'ollama' } as const
const baseline: EngineSettingsSnapshot = {
    ...target,
    revision: 1,
    appliedRevision: 1,
    epoch: 'one',
    sequence: 1,
    settings: { serverPort: 1235, proxyPort: 11434, launchText: 'managed serve' },
    effectiveServerPort: 1235,
    effectiveProxyPort: 11434,
    running: true,
    adopted: false,
    editable: true,
    reason: '',
    format: 'pair-arguments-v1',
    phase: 'idle',
    error: '',
    requestId: ''
}
const key = engineSettingsKey(target)
const store = useEngineSettingsStore
const getSettings = vi.fn()
const applySettings = vi.fn()

const serving: EngineServingConfig = {
    version: 1,
    models: [
        {
            modelId: 'exact/model',
            maxOutputTokens: 128,
            memory: { hostRamBytes: 4096, vramBytes: 0 },
            artifacts: [
                {
                    role: 'model',
                    sha256: 'a'.repeat(64),
                    provenance: { kind: 'supplied-sha256' },
                    binding: {
                        sourcePath: 'fixture-link',
                        resolvedTarget: 'fixture-blob',
                        sizeBytes: 64,
                        mtimeNs: '1729000000000000001',
                        sampleSha256: 'b'.repeat(64)
                    }
                }
            ]
        }
    ]
}

describe('serving preservation in the settings editor', () => {
    afterEach(() => {
        vi.unstubAllGlobals()
        vi.restoreAllMocks()
    })

    it('carries serving through an actual unrelated form edit and failed apply', async () => {
        const snapshot: EngineSettingsSnapshot = {
            ...baseline,
            settings: { ...baseline.settings, serving: structuredClone(serving) }
        }
        const get = vi.fn<IEngineApi['getSettings']>().mockResolvedValue(snapshot)
        const preview = vi
            .fn<IEngineApi['previewSettings']>()
            .mockImplementation(async request => ({
                settings: request.settings,
                revision: 1,
                errors: {},
                restart: false,
                rebind: false
            }))
        const apply = vi
            .fn<IEngineApi['applySettings']>()
            .mockRejectedValue(new Error('synthetic failure'))
        Object.defineProperty(window, 'pairApi', {
            configurable: true,
            value: {
                engines: {
                    getSettings: get,
                    previewSettings: preview,
                    applySettings: apply,
                    onSettingsChanged: () => () => {},
                    onSettingsDisconnected: () => () => {}
                }
            }
        })
        vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
        vi.stubGlobal('CSS', { supports: () => false })
        Object.defineProperty(HTMLDialogElement.prototype, 'showModal', {
            configurable: true,
            value: function (this: HTMLDialogElement) {
                this.open = true
            }
        })
        Object.defineProperty(HTMLDialogElement.prototype, 'close', {
            configurable: true,
            value: function (this: HTMLDialogElement) {
                this.open = false
            }
        })
        useConnectionStore.setState({ connected: true, selfId: 'remote' })
        store.setState({ entries: {} })
        usePendingActionsStore.setState({ pending: new Map() })
        const container = document.createElement('div')
        document.body.appendChild(container)
        const root = createRoot(container)
        try {
            await act(async () =>
                root.render(
                    createElement(EngineSettingsSection, {
                        nodeId: 'remote',
                        engineType: 'ollama',
                        disabled: false
                    })
                )
            )
            const input = container.querySelector('input[aria-label="Proxy port"]')
            const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
            if (!input || !setter) throw new Error('Proxy port input missing')
            await act(async () => {
                setter.call(input, '12500')
                input.dispatchEvent(new Event('input', { bubbles: true }))
            })
            const button = container.querySelector('button[aria-label="Apply"]')
            if (!(button instanceof HTMLButtonElement)) throw new Error('Apply button missing')
            await act(async () => button.click())
            expect(preview).toHaveBeenCalledWith(
                expect.objectContaining({ settings: { ...snapshot.settings, proxyPort: 12500 } })
            )
            expect(apply).toHaveBeenCalledWith(
                expect.objectContaining({ settings: { ...snapshot.settings, proxyPort: 12500 } })
            )
            expect(store.getState().entries[key].snapshot?.settings.serving).toEqual(serving)
        } finally {
            await act(async () => root.unmount())
            container.remove()
        }
    })

    it('owns nested serving values received from a caller', () => {
        const snapshot: EngineSettingsSnapshot = {
            ...baseline,
            settings: { ...baseline.settings, serving: structuredClone(serving) }
        }
        store.setState({ entries: {} })
        store.getState().receive(snapshot)
        if (!snapshot.settings.serving) throw new Error('Missing fixture serving')
        snapshot.settings.serving.models[0].artifacts[0].provenance.kind = 'changed'
        expect(store.getState().entries[key].snapshot?.settings.serving).toEqual(serving)
    })
})

describe('engine settings snapshot ordering and pending operations', () => {
    beforeEach(() => {
        store.setState({ entries: {} })
        vi.stubGlobal('window', { pairApi: { engines: { getSettings, applySettings } } })
        getSettings.mockResolvedValue(baseline)
        applySettings.mockResolvedValue({ revision: 2, phase: 'succeeded' })
    })
    afterEach(() => vi.unstubAllGlobals())

    it('ignores older frames and marks identical reconnect baselines available', () => {
        store.getState().receive({ ...baseline, sequence: 3, revision: 2 })
        store.getState().receive(baseline)
        expect(store.getState().entries[key].snapshot?.revision).toBe(2)
        store.getState().disconnect('remote')
        expect(store.getState().entries[key].unavailable).toBeTruthy()
        store.getState().receive({ ...baseline, sequence: 3, revision: 2 })
        expect(store.getState().entries[key].unavailable).toBeUndefined()
    })

    it('a newer push epoch beats a delayed initial read', async () => {
        let finish: (snapshot: EngineSettingsSnapshot) => void = () => {}
        getSettings.mockImplementationOnce(
            () =>
                new Promise<EngineSettingsSnapshot>(resolve => {
                    finish = resolve
                })
        )
        const loading = store.getState().reload(target)
        store.getState().receive({ ...baseline, epoch: 'two', revision: 9 })
        finish(baseline)
        await loading
        expect(store.getState().entries[key].snapshot?.epoch).toBe('two')
    })

    it('acknowledgement alone cannot fabricate an applied snapshot', async () => {
        store.getState().receive(baseline)
        await store.getState().apply({
            ...target,
            expectedRevision: 1,
            requestId: 'request',
            settings: { ...baseline.settings, launchText: 'different' }
        })
        expect(store.getState().entries[key].snapshot?.revision).toBe(1)
        // The settings push carries the applied snapshot; re-reading here would
        // only race it.
        expect(getSettings).not.toHaveBeenCalled()
    })

    it('a failed transport reads back and exposes unavailable status', async () => {
        applySettings.mockRejectedValueOnce(new Error('disconnected'))
        getSettings.mockRejectedValueOnce(new Error('offline'))
        await expect(
            store.getState().apply({
                ...target,
                expectedRevision: 1,
                requestId: 'request',
                settings: baseline.settings
            })
        ).rejects.toThrow('disconnected')
        expect(store.getState().entries[key].unavailable).toContain('offline')
    })

    it('shares subscriptions and releases them when the last editor closes', () => {
        const stop = vi.fn()
        const subscribe = vi.fn(() => stop)
        vi.stubGlobal('window', {
            pairApi: {
                engines: { onSettingsChanged: subscribe, onSettingsDisconnected: subscribe }
            }
        })
        const a = watchEngineSettings(),
            b = watchEngineSettings()
        expect(subscribe).toHaveBeenCalledTimes(2)
        a()
        expect(stop).not.toHaveBeenCalled()
        b()
        expect(stop).toHaveBeenCalledTimes(2)
    })

    it('closing an editor leaves other devices editable and resubscribes on reopen', () => {
        const receivers: Array<(snapshot: EngineSettingsSnapshot) => void> = []
        vi.stubGlobal('window', {
            pairApi: {
                engines: {
                    onSettingsChanged: (fn: (snapshot: EngineSettingsSnapshot) => void) => {
                        receivers.push(fn)
                        return () => {}
                    },
                    onSettingsDisconnected: () => () => {}
                }
            }
        })
        store.getState().receive(baseline)
        // An engine that starts installing unmounts its own editor; that must
        // not report every device in the cluster as disconnected.
        watchEngineSettings()()
        expect(store.getState().entries[key].unavailable).toBeUndefined()
        expect(store.getState().entries[key].snapshot?.revision).toBe(1)

        const reopened = watchEngineSettings()
        receivers.at(-1)?.({ ...baseline, sequence: 2, revision: 4 })
        expect(store.getState().entries[key].snapshot?.revision).toBe(4)
        expect(store.getState().entries[key].unavailable).toBeUndefined()
        reopened()
    })
})

describe('settings applies use the shared pending-actions cover', () => {
    let deliver: (snapshot: EngineSettingsSnapshot) => void = () => {}

    beforeEach(() => {
        const noop = (): void => {}
        vi.stubGlobal('window', {
            pairApi: {
                engines: {
                    onStateChanged: () => noop,
                    onProgress: () => noop,
                    onSettingsChanged: (fn: (snapshot: EngineSettingsSnapshot) => void) => {
                        deliver = fn
                        return noop
                    }
                },
                errors: { onUpdate: () => noop }
            }
        })
        usePendingActionsStore.getState().initialize()
    })
    afterEach(() => {
        usePendingActionsStore.getState().cleanup()
        vi.unstubAllGlobals()
    })

    it('holds from the send until the first snapshot, then yields to phase', () => {
        const pending = usePendingActionsStore.getState()
        expect(pending.isSettingsPending('remote', 'ollama')).toBe(false)

        pending.beginSettings('remote', 'ollama', 'current-request')
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'ollama')).toBe(true)

        // Another engine on the same device is untouched.
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'lm-studio')).toBe(
            false
        )

        deliver({ ...baseline, phase: 'applying', requestId: 'older-request' })
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'ollama')).toBe(true)
        deliver({ ...baseline, phase: 'idle' })
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'ollama')).toBe(true)
        deliver({ ...baseline, phase: 'applying', requestId: 'current-request' })
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'ollama')).toBe(false)
    })

    it('releases the cover when validation refuses before sending', () => {
        const pending = usePendingActionsStore.getState()
        pending.beginSettings('remote', 'lm-studio', 'current-request')
        pending.endSettings('remote', 'lm-studio', 'older-request')
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'lm-studio')).toBe(
            true
        )
        pending.endSettings('remote', 'lm-studio', 'current-request')
        expect(usePendingActionsStore.getState().isSettingsPending('remote', 'lm-studio')).toBe(
            false
        )
    })
})

describe('launch text redaction', () => {
    const frame = JSON.stringify({
        result: {
            settings: { launchText: 'exe --unknown private-value' },
            launch_args: ['private-value'],
            launch_env: ['VALUE=private-value']
        }
    })

    it('omits launch text, arguments, and environment from process logs', () => {
        expect(redactSensitiveLogText(frame)).not.toContain('private-value')
    })

    it('omits them from a frame truncated mid-value', () => {
        expect(redactSensitiveLogText(frame.slice(0, -1))).not.toContain('private-value')
        expect(
            redactSensitiveLogText('{"result":{"launch_env":["VALUE=private-value"')
        ).not.toContain('private-value')
    })

    it('still redacts the pairing PIN', () => {
        expect(redactSensitiveLogText('{"result":{"pin":"123456"}}')).not.toContain('123456')
        expect(redactSensitiveLogText('garbage {"pin": "123456"')).not.toContain('123456')
    })

    // A worker's own command line is what a support log is for, so the
    // redaction must not reach a generic args/env key.
    it('keeps worker spawn diagnostics intact', () => {
        const spawn = JSON.stringify({
            msg: 'spawning worker',
            bin: 'nvpair-node-scanner',
            args: ['--ipc', '--log-level=debug'],
            env: { NVPAIR_LOG: 'debug' }
        })
        expect(redactSensitiveLogText(spawn)).toBe(spawn)
        expect(redactSensitiveLogText('plain text with no json at all')).toBe(
            'plain text with no json at all'
        )
    })
})
