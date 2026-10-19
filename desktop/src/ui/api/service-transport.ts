// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { IpcResult } from '@/shared/types/ipc-channels'
import type {
    PreloadServiceTransport,
    ServiceInvokeArgs,
    ServiceTransport
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

/** The renderer's transport over the preload's: failed invokes reject with a ServiceCallError. */
export function serviceTransport(preload: PreloadServiceTransport): ServiceTransport {
    return {
        get connected() {
            return preload.connected
        },
        invoke<C extends WsInvokeChannel>(
            ...args: ServiceInvokeArgs<C>
        ): Promise<WsInvokeResponse<C>> {
            return preload.invoke<C>(...args).then(unwrap)
        },
        subscribePush: (channel, fn) => preload.subscribePush(channel, fn),
        onConnect: fn => preload.onConnect(fn),
        onDisconnect: fn => preload.onDisconnect(fn),
        onAuthFailure: fn => preload.onAuthFailure(fn),
        destroy: () => preload.destroy()
    }
}
