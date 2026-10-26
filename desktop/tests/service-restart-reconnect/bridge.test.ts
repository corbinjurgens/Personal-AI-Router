// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import os from 'node:os'
import path from 'node:path'
import { spawn } from 'child_process'
import { BroadcastChannel } from 'node:worker_threads'
import { expect, it, vi } from 'vitest'
import { initPlatform } from '@/electron/globals'
import { completeFirstRun } from '@/electron/config/ui-config'
import { JsonRpcSocketClient, type JsonValue } from '@/electron/service-bridge/json-rpc-client'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { parsePolicyDocument, parsePolicyText } from '@/electron/service-bridge/node-policy'
import { subscribePush } from '@/electron/service-bridge/push-bus'

vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => process.env.PAIR_DESKTOP_USER_DATA },
    BrowserWindow: { getAllWindows: () => [] }
}))
vi.mock('@/electron/window', () => ({ createOverviewWindow: () => undefined }))
vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: () => undefined,
        warn: () => undefined,
        error: () => undefined,
        verbose: () => undefined
    })
}))
// Keep connectOrStart and its real socket dialer. A failed dial may exercise the
// finite startup policy, but only Go may launch/restart the private service.
vi.mock('child_process', () => ({
    spawn: vi.fn<typeof spawn>(() => {
        throw new Error('Detached launch belongs to the Go restart fixture')
    })
}))

function requiredEnv(name: string): string {
    const value = process.env[name]
    if (!value) throw new Error(`Missing fixture prerequisite ${name}`)
    return value
}

function object(value: JsonValue | undefined) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        throw new Error('Expected a service object')
    }
    return value
}

it('retains the real desktop client through service and broker replacement', async () => {
    expect(process.env.PAIR_TEST_DESKTOP_RECONNECT).toBe('1')
    expect(process.versions.node).toBe('25.9.0')
    const endpoint = requiredEnv('NVPAIR_SERVICE_ENDPOINT')
    const userData = requiredEnv('PAIR_DESKTOP_USER_DATA')
    const products = requiredEnv('PAIR_DESKTOP_PRODUCTS')
    const upstreamPort = Number(requiredEnv('PAIR_DESKTOP_UPSTREAM_PORT'))
    expect(path.isAbsolute(userData)).toBe(true)
    expect(path.relative(os.tmpdir(), userData).startsWith('..')).toBe(false)
    expect(endpoint).toBe(requiredEnv('PAIR_DESKTOP_ENDPOINT'))
    expect(endpoint).toMatch(
        process.platform === 'win32' ? /^\\\\\.\\pipe\\nvpair-test-/ : /\/s\.sock$/
    )

    initPlatform({
        getUserData: () => userData,
        getTemp: () => os.tmpdir(),
        getResourcesPath: () => userData
    })
    completeFirstRun()

    // Electron's cli-bin layout is a path boundary here. Resolve existence
    // checks to the actual ruled product bundle; none of these UI paths launch.
    const exists = fs.existsSync
    const uiBin = path.join(userData, 'cli-bin')
    const binaryPaths = vi.spyOn(fs, 'existsSync').mockImplementation(file => {
        if (typeof file === 'string' && path.dirname(file) === uiBin) {
            return exists(path.join(products, path.basename(file)))
        }
        return exists(file)
    })
    // Observation only: the real methods, sockets and state are never replaced.
    const starts = vi.spyOn(JsonRpcSocketClient.prototype, 'start')
    const calls = vi.spyOn(JsonRpcSocketClient.prototype, 'call')
    const supervisor = getModularSupervisor()
    const state = getModularBridgeState()
    const channel = new BroadcastChannel(requiredEnv('PAIR_DESKTOP_COORDINATION'))
    let pendingCommand:
        | { command: string; resolve: () => void; reject: (error: Error) => void }
        | undefined
    let commandTimer: ReturnType<typeof setTimeout> | undefined
    channel.onmessage = event => {
        if (typeof event.data !== 'string') return
        const pending = pendingCommand
        if (!pending) throw new Error('Fixture command arrived without a waiter')
        clearTimeout(commandTimer)
        pendingCommand = undefined
        if (event.data === `command:${pending.command}`) pending.resolve()
        else pending.reject(new Error('Fixture aborted or sent an out-of-order command'))
    }
    const command = (name: string): Promise<void> => {
        if (pendingCommand) throw new Error('Overlapping desktop fixture commands')
        return new Promise((resolve, reject) => {
            pendingCommand = { command: name, resolve, reject }
            commandTimer = setTimeout(() => {
                pendingCommand = undefined
                reject(new Error(`Missing fixture command ${name}`))
            }, 45_000)
        })
    }
    const stage = (name: string): void => channel.postMessage(`stage:${name}`)
    const initialModel = 'desktop-before-restart-model'
    const replacementModel = 'desktop-restart-model'
    const customModels = () =>
        state
            .getEngineInitialState()
            .models.find(
                row =>
                    row.engineType === 'desktop-restart-proof' && row.nodeId === state.getSelfId()
            )
    let refreshes = 0
    const modelRefreshes = new Map<
        string,
        { sequence: number; models: string[]; current: boolean | undefined }
    >()
    let discoveryPushes = 0
    let policyPush: JsonValue | undefined
    const unsubscribe = subscribePush(event => {
        if (event.channel === 'state:request-refresh') {
            refreshes++
            const projected = customModels()
            const snapshot = {
                sequence: refreshes,
                models: projected?.models.map(model => model.name) ?? [],
                current: projected?.downloadStateCurrent
            }
            const model = snapshot.models[0]
            // Self-identity also emits a refresh before model hydration. Keep
            // only the two qualifying observed snapshots, including currentness.
            if (
                snapshot.current === true &&
                snapshot.models.length === 1 &&
                (model === initialModel || model === replacementModel)
            ) {
                modelRefreshes.set(model, snapshot)
            }
        }
        if (event.channel === 'policy:changed') policyPush = parsePolicyText(event.payload.policy)
    })
    let connectionLost = false
    supervisor.setOnConnectionLost(() => {
        connectionLost = true
    })
    const baseline = async () =>
        parsePolicyDocument(await supervisor.callProcess('broker', 'policy:get'))
    const admission = (policy: string): JsonValue | undefined =>
        object(parsePolicyText(policy).admission).maxConcurrentPerModel
    let detached = false
    try {
        supervisor.start()
        expect(starts).toHaveBeenCalledTimes(1)
        const retained = starts.mock.contexts[0]
        if (!(retained instanceof JsonRpcSocketClient)) {
            throw new Error('Supervisor did not start its real socket client')
        }
        retained.on('notification', notification => {
            if (notification.method === 'discovery:nodes-changed') discoveryPushes++
        })
        await expect
            .poll(() => modelRefreshes.get(initialModel)?.sequence ?? 0, { timeout: 30_000 })
            .toBeGreaterThan(0)
        expect(supervisor.ready).toBe(true)
        const selfId = state.getSelfId()
        expect(selfId).toBeTruthy()
        expect(modelRefreshes.get(initialModel)?.models).toEqual([initialModel])
        expect(modelRefreshes.get(initialModel)?.current).toBe(true)
        expect(customModels()?.downloadStateCurrent).toBe(true)
        expect(supervisor.getAvailabilityMenuItem().request).toBe('paused')
        const initial = await baseline()
        expect(admission(initial.policy)).toBe(3)
        const oldStatus = object(await supervisor.callProcess('broker', 'service/status'))
        expect(oldStatus.pid).toBeTypeOf('number')
        expect(oldStatus.brokerPid).toBeTypeOf('number')
        const oldRefreshes = refreshes
        const oldDiscoveryPushes = discoveryPushes
        expect(oldDiscoveryPushes).toBeGreaterThan(0)
        stage('attached')

        let pendingSettled = false
        const pending = supervisor
            .callProcess(
                'broker',
                'engine:action',
                {
                    engine: 'desktop-restart-proof',
                    action: 'restart_probe',
                    params: { token: 'desktop-pending-before-exit' }
                },
                60_000
            )
            .then(
                () => {
                    pendingSettled = true
                    return undefined
                },
                error => {
                    pendingSettled = true
                    return error instanceof Error
                        ? error
                        : new Error('Non-error pending action failure')
                }
            )
        stage('kill')
        await expect.poll(() => connectionLost, { timeout: 15_000 }).toBe(true)
        await expect.poll(() => retained.running, { timeout: 1_000 }).toBe(false)
        expect(supervisor.ready).toBe(false)
        expect(supervisor.getAvailabilityMenuItem().enabled).toBe(false)
        await expect.poll(() => pendingSettled, { timeout: 1_000 }).toBe(true)
        const interrupted = await pending
        expect(interrupted?.message).toContain('connection closed before replying')
        await expect(supervisor.callProcess('broker', 'policy:get')).rejects.toThrow('not running')
        stage('disconnected')

        // Exhaust the unchanged finite connectOrStart attempts. Recovery below
        // must therefore use the production dial-only continuation as well.
        await expect.poll(() => vi.mocked(spawn).mock.calls.length, { timeout: 30_000 }).toBe(5)
        const restarted = command('restart')
        stage('retrying')
        await restarted
        await expect
            .poll(() => modelRefreshes.get(replacementModel)?.sequence ?? 0, {
                timeout: 30_000
            })
            .toBeGreaterThan(oldRefreshes)
        // Inspect a qualifying real renderer-refresh snapshot, before this
        // test issues any post-restart read. The old row must be gone there.
        expect(modelRefreshes.get(replacementModel)?.models).toEqual([replacementModel])
        expect(modelRefreshes.get(replacementModel)?.current).toBe(true)
        expect(customModels()?.models.map(model => model.name)).toEqual([replacementModel])
        expect(customModels()?.downloadStateCurrent).toBe(true)
        expect(supervisor.ready).toBe(true)
        expect(starts).toHaveBeenCalledTimes(1)
        expect(starts.mock.contexts[0]).toBe(retained)
        expect(retained.running).toBe(true)
        expect(state.getSelfId()).toBe(selfId)
        expect(discoveryPushes).toBeGreaterThan(oldDiscoveryPushes)
        expect(
            calls.mock.calls.filter(([method]) => method === 'discovery:subscribe')
        ).toHaveLength(2)
        const fresh = await baseline()
        expect(fresh.revision).not.toBe(initial.revision)
        expect(fresh.availability).toBe('available')
        expect(admission(fresh.policy)).toBe(7)
        expect(supervisor.getAvailabilityMenuItem().request).toBe('paused')
        const replacement = object(await supervisor.callProcess('broker', 'service/status'))
        expect(replacement.pid).toBeTypeOf('number')
        expect(replacement.brokerPid).toBeTypeOf('number')
        expect(replacement.pid).not.toBe(oldStatus.pid)
        expect(replacement.brokerPid).not.toBe(oldStatus.brokerPid)
        await supervisor.callProcess('broker', 'llamacpp-proxy:node/add-manual', {
            id: 'desktop-restart-upstream',
            host: '127.0.0.1',
            port: upstreamPort,
            addresses: ['127.0.0.1'],
            models: ['desktop-restart-model']
        })
        const pushed = command('push')
        stage('hydrated')
        await pushed
        await expect
            .poll(() => object(object(policyPush).admission).maxConcurrentPerModel, {
                timeout: 10_000
            })
            .toBe(9)
        expect(object(policyPush).revision).not.toBe(fresh.revision)
        await expect.poll(() => supervisor.getAvailabilityMenuItem().request).toBe('paused')
        expect(calls.mock.calls.filter(([method]) => method === 'engine:action')).toHaveLength(1)
        const detach = command('detach')
        stage('pushed')
        await detach
        await supervisor.stop()
        detached = true
        expect(retained.running).toBe(false)
        expect(supervisor.hasProcess('broker')).toBe(false)
        expect(starts).toHaveBeenCalledTimes(1)
        expect(vi.mocked(spawn)).toHaveBeenCalledTimes(5)
        const finished = command('finish')
        stage('detached')
        await finished
    } finally {
        if (!detached) await supervisor.stop()
        supervisor.setOnConnectionLost(() => undefined)
        unsubscribe()
        clearTimeout(commandTimer)
        channel.close()
        starts.mockRestore()
        calls.mockRestore()
        binaryPaths.mockRestore()
    }
})
