// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { IpcResult } from '@/shared/types/ipc-channels'
import {
    isHistoryInvoke,
    type PreloadServiceTransport,
    type ServiceInvokeArgs,
    type ServiceTransport
} from '@/shared/types/service-bridge'
import type { WsInvokeChannel, WsInvokeResponse } from '@/shared/types/ws-channels'

/** A failed service call. `code` is the service's JSON-RPC code when a service refused it. */
export class ServiceCallError extends Error {
    readonly code?: number

    constructor(message: string, code?: number) {
        super(message)
        this.code = code
    }
}

/** The JSON-RPC code a service refused a call with, or undefined when there is none. */
export function serviceErrorCode(err: unknown): number | undefined {
    return err instanceof ServiceCallError ? err.code : undefined
}

function unwrap<T>(result: IpcResult<T>): T {
    if (!result.success) throw new ServiceCallError(result.error, result.code)
    return result.data
}

/**
 * Renderer-local completion state. A retained native function proxy must not
 * retain a settled resolver, result, pending Promise, or bank through this loan.
 * No caller arguments or transport owner are captured by complete.
 */
function historyCompletion<T>() {
    let accept: ((value: T) => void) | null = null
    let refuse: ((error: Error) => void) | null = null
    const promise = new Promise<T>((resolve, reject) => {
        accept = resolve
        refuse = reject
    })
    const complete = (result: IpcResult<T> | null): undefined => {
        let resolve = accept,
            reject = refuse
        accept = null
        refuse = null
        if (!resolve || !reject || !result) return undefined
        try {
            if (result.success) resolve(result.data)
            else reject(new ServiceCallError(result.error, result.code))
        } finally {
            resolve = null
            reject = null
            result = null
        }
        return undefined
    }
    return { promise, complete }
}

/** The renderer's transport over the preload's: failed invokes reject with a ServiceCallError. */
export function serviceTransport(preload: PreloadServiceTransport): ServiceTransport {
    return {
        get connected() {
            return preload.connected
        },
        invoke<C extends WsInvokeChannel>(
            ...args: ServiceInvokeArgs<C>
        ): Promise<WsInvokeResponse<C>> {
            if (isHistoryInvoke(args[0])) {
                const completion = historyCompletion<WsInvokeResponse<C>>()
                try {
                    if (preload.invokeHistory<C>(completion.complete, ...args) !== undefined)
                        throw new Error('history-invalid')
                } catch {
                    // Detach resolver references on synchronous bridge failure
                    // too; never fall back to the ordinary Promise entry.
                    completion.complete({ success: false, error: 'history-invalid' })
                }
                return completion.promise
            }
            return preload.invoke<C>(...args).then(unwrap)
        },
        subscribePush: (channel, fn) => preload.subscribePush(channel, fn),
        onConnect: fn => preload.onConnect(fn),
        onDisconnect: fn => preload.onDisconnect(fn),
        onAuthFailure: fn => preload.onAuthFailure(fn),
        destroy: () => preload.destroy()
    }
}
