// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ServiceTransport } from '@/shared/types/service-bridge'
import {
    RoutingHistoryBank,
    releaseRoutingHistoryScratch,
    type RoutingHistoryConsumer,
    type RoutingHistoryCommit,
    type RoutingHistoryTransfer
} from '@/shared/types/routing'

/** Renderer bank owner. No returned history graph, raw bytes, iterator or output queue. */
export function createRoutingHistory(transport: ServiceTransport) {
    let banks: {
        current: RoutingHistoryBank
        pending: RoutingHistoryBank
        baseline: RoutingHistoryBank
    } | null = null
    let wanted = false,
        disposed = false,
        connected = transport.connected,
        initialDone = false,
        loan = false
    let session = 0,
        generation = 0,
        readyGeneration = 0,
        registeredSession = -1,
        registered = false
    let initialCall: Promise<void> | null = null,
        initialResolve: (() => void) | null = null,
        initialReject: ((error: Error) => void) | null = null
    let retirementCall: Promise<void> | null = null,
        retirementResolve: (() => void) | null = null,
        retirementReject: ((error: Error) => void) | null = null
    let controlCall: Promise<void> | null = null,
        pullCall: Promise<void> | null = null
    let frame: number | null = null,
        frameTicket = 0,
        frameResolve: (() => void) | null = null
    let attached = false,
        retiring = false
    let controlFailed = false,
        blocked = false,
        pendingEnd = false,
        pendingVisible = false,
        initialCount = 0
    let pendingLease = 0,
        pendingEpoch = 0,
        remoteLoading = false,
        acknowledging = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null,
        retry = 0
    let removeReady = () => {},
        removeCommit = () => {},
        removeConnect = () => {},
        removeDisconnect = () => {}
    const cancelRetry = (): void => {
        if (retryTimer !== null) clearTimeout(retryTimer)
        retryTimer = null
        retry = 0
    }
    const assertMutable = (): void => {
        if (loan) throw new Error('history-busy')
    }
    const ownedBanks = () =>
        (banks ??= {
            current: new RoutingHistoryBank(),
            pending: new RoutingHistoryBank(),
            baseline: new RoutingHistoryBank()
        })
    const clear = (): void => {
        banks?.current.clear()
        banks?.pending.clear()
        banks?.baseline.clear()
    }
    const finishFrame = (): void => {
        if (!frameResolve) return
        const bank = ownedBanks(),
            old = bank.current
        bank.current = bank.pending
        bank.pending = old
        bank.pending.clear()
        bank.baseline.clear()
        const resolve = frameResolve
        frameResolve = null
        frame = null
        resolve()
    }
    const retireFrame = (): void => {
        if (frameTicket < Number.MAX_SAFE_INTEGER) frameTicket++
        if (frame !== null) cancelAnimationFrame(frame)
        // A received transfer was already validated into pending. Commit its
        // retired generation synchronously, then clear it before acknowledgement.
        finishFrame()
        clear()
    }
    const newInitial = (): Promise<void> => {
        if (!initialCall) {
            initialCall = initialDone
                ? Promise.resolve()
                : new Promise<void>((resolve, reject) => {
                      initialResolve = resolve
                      initialReject = reject
                  })
            // One bounded owner handler; callers receive the original promise.
            void initialCall.catch(() => {})
        }
        return initialCall
    }
    const fence = (nextGeneration: number, cancel: boolean): void => {
        cancelRetry()
        if (session === Number.MAX_SAFE_INTEGER) {
            blocked = true
            wanted = false
            initialReject?.(new Error('history-exhausted'))
        } else session++
        generation = nextGeneration
        initialDone = false
        retireFrame()
        if (cancel) {
            initialReject?.(new Error('history-retired'))
            initialResolve = null
            initialReject = null
            initialCall = null
        } else if (!initialResolve) initialCall = null
    }
    const commitFrame = (): Promise<void> => {
        if (frameResolve) throw new Error('history-busy')
        if (frameTicket === Number.MAX_SAFE_INTEGER) throw new Error('history-exhausted')
        const ticket = ++frameTicket
        return new Promise(resolve => {
            frameResolve = resolve
            frame = requestAnimationFrame(() => {
                if (ticket !== frameTicket) return
                finishFrame()
            })
        })
    }
    const cleanup = (): void => {
        if (controlCall || pullCall || loan || retiring) return
        if (!wanted) {
            clear()
            banks = null
            releaseRoutingHistoryScratch()
        }
        if (!disposed) return
        removeReady()
        removeCommit()
        removeConnect()
        removeDisconnect()
        attached = false
        clear()
        banks = null
    }
    const finishRetirement = (): void => {
        retiring = false
        initialCount = 0
        retirementResolve?.()
        retirementResolve = null
        retirementReject = null
    }
    const drivePull = (): void => {
        if (
            !connected ||
            blocked ||
            controlCall ||
            pullCall ||
            retryTimer !== null ||
            !readyGeneration
        )
            return
        if (
            !retiring &&
            (disposed || (wanted ? !registered || registeredSession !== session : registered))
        )
            return
        const expectedGeneration = readyGeneration
        readyGeneration = 0
        if (wanted && !retiring && generation !== expectedGeneration) {
            fence(expectedGeneration, false)
            registeredSession = session
        }
        const epoch = session,
            wantsInitial = wanted && !retiring && !initialDone
        pullCall = transport
            .invoke('routing-history:pull', {
                generation: expectedGeneration,
                initial: wantsInitial
            })
            .then(async (transfer: RoutingHistoryTransfer | null) => {
                if (!transfer) throw new Error('history-invalid')
                if (
                    transfer.state === 'busy' &&
                    transfer.loading &&
                    wanted &&
                    !disposed &&
                    epoch === session
                ) {
                    readyGeneration = expectedGeneration
                    const delay = retry === 0 ? 250 : retry === 1 ? 500 : 1000
                    retry = Math.min(retry + 1, 2)
                    retryTimer = setTimeout(() => {
                        retryTimer = null
                        driveControl()
                    }, delay)
                    return
                }
                cancelRetry()
                if (transfer.state === 'retired' && retiring) finishRetirement()
                if (transfer.state !== 'data' && transfer.state !== 'end') return
                if (
                    transfer.generation !== expectedGeneration ||
                    !Number.isSafeInteger(transfer.lease) ||
                    transfer.lease < 1
                )
                    throw new Error('history-invalid')
                pendingEnd = false
                pendingVisible = false
                pendingLease = transfer.lease
                pendingEpoch = epoch
                const bank = ownedBanks()
                const current =
                    wanted &&
                    !disposed &&
                    connected &&
                    epoch === session &&
                    transfer.generation === generation
                if (transfer.state === 'data') {
                    if (transfer.initial !== wantsInitial) throw new Error('history-invalid')
                    if (transfer.initial) {
                        if (transfer.index !== initialCount || initialCount === 256)
                            throw new Error('history-invalid')
                        if (initialCount === 0) bank.baseline.clear()
                        if (!bank.baseline.applyPacked(transfer.bytes))
                            throw new Error('history-invalid')
                        initialCount++
                        // Packet consumption commits only to the owned staging
                        // bank. Current remains coherent until the scalar end.
                    } else {
                        bank.pending.copyFrom(bank.current)
                        bank.pending.applyPacked(transfer.bytes)
                        pendingVisible = current
                        const committed = commitFrame()
                        if (!current) retireFrame()
                        driveControl()
                        await committed
                    }
                } else {
                    if (
                        !Number.isInteger(transfer.count) ||
                        transfer.count < 0 ||
                        transfer.count > 256 ||
                        transfer.count !== initialCount ||
                        typeof transfer.retired !== 'boolean'
                    )
                        throw new Error('history-invalid')
                    if (transfer.retired) {
                        retireFrame()
                        initialCount = 0
                    } else {
                        if (!wantsInitial) throw new Error('history-invalid')
                        bank.pending.copyFrom(bank.current)
                        for (let i = 0; i < bank.baseline.size; i++)
                            bank.pending.applyPacked(bank.baseline.at(i))
                        pendingEnd = current
                        pendingVisible = current
                        const committed = commitFrame()
                        if (!current) retireFrame()
                        driveControl()
                        await committed
                    }
                }
                // No preload receipt acknowledgement. Every record is validated
                // and stored, or end has completed its coherent rAF/retirement,
                // before its exact native lease can be consumed.
                driveControl()
                // Unsubscribe may run after receipt while rAF/retirement owns the
                // packet. Its one callback completes before the ACK native invoke;
                // no second cross-world completion or native request queue.
                if (controlCall) await controlCall
                acknowledging = true
                const receipt = await transport.invoke('routing-history:consumed', {
                    generation: transfer.generation,
                    lease: transfer.lease
                })
                if (receipt.state === 'retired' && retiring) finishRetirement()
                if (
                    current &&
                    epoch === session &&
                    receipt.state === 'accepted' &&
                    receipt.generation === generation
                ) {
                    initialDone = initialDone || pendingEnd
                    if (initialDone) {
                        initialResolve?.()
                        initialResolve = null
                        initialReject = null
                    }
                } else if (!wanted || disposed || epoch !== session) clear()
                if (transfer.state === 'end') initialCount = 0
                transfer = null
            })
            .catch(() => {
                // No ack on malformed data. Its admitted physical transfer remains
                // owned by this context until actual destruction; no timeout reuse.
                blocked = true
                if (epoch === session) initialReject?.(new Error('history-invalid'))
            })
            .finally(() => {
                acknowledging = false
                pendingLease = 0
                pendingEnd = false
                pendingVisible = false
                pullCall = null
                driveControl()
                cleanup()
            })
    }
    const driveControl = (): void => {
        if (controlCall || !connected || controlFailed) return
        // A pull awaiting its callback, or an ACK awaiting completion, owns the
        // sole preload completion slot. Desired unsubscribe remains scalar.
        if (pullCall && (!pendingLease || acknowledging)) return
        if (registered && (!wanted || registeredSession !== session)) {
            retiring = true
            controlCall = transport
                .invoke('routing:unsubscribe')
                .then(() => {
                    registered = false
                    registeredSession = -1
                })
                .catch(() => {
                    controlFailed = true
                    retiring = false
                    initialReject?.(new Error('history-invalid'))
                    retirementReject?.(new Error('history-invalid'))
                    retirementResolve = null
                    retirementReject = null
                })
                .finally(() => {
                    controlCall = null
                    driveControl()
                    cleanup()
                })
            return
        }
        if (retiring) {
            drivePull()
            return
        }
        if (wanted && !disposed && !registered) {
            const expected = session
            controlCall = transport
                .invoke('routing:subscribe')
                .then(() => {
                    registered = true
                    registeredSession = expected
                    if (!retirementResolve) retirementCall = null
                })
                .catch(() => {
                    wanted = false
                    initialReject?.(new Error('history-invalid'))
                })
                .finally(() => {
                    controlCall = null
                    driveControl()
                    cleanup()
                })
            return
        }
        if (!wanted && !registered) {
            retirementResolve?.()
            retirementResolve = null
            retirementReject = null
        }
        drivePull()
    }
    const attach = (): void => {
        if (attached) return
        attached = true
        removeCommit = transport.subscribePush('routing-history:committed', token => {
            if ('loading' in token) {
                remoteLoading = token.loading
                return
            }
            if (token.revision !== 0) {
                if (
                    token.generation === generation &&
                    token.revision === pendingLease &&
                    pendingEpoch === session &&
                    wanted &&
                    !disposed
                ) {
                    initialDone = initialDone || pendingEnd
                    remoteLoading = false
                    if (initialDone) {
                        initialResolve?.()
                        initialResolve = null
                        initialReject = null
                    }
                }
                return
            }
            fence(token.generation, false)
            if (registered && wanted) registeredSession = session
        })
        removeReady = transport.subscribePush('routing-history:ready', ready => {
            // A posted scalar token remains owned through logical unsubscribe.
            // Drain it only after native unsubscribe completed; that pull cannot
            // admit data and lets the same context subscribe again safely.
            cancelRetry()
            readyGeneration = ready.generation
            driveControl()
        })
        removeDisconnect = transport.onDisconnect(() => {
            connected = false
            registered = false
            registeredSession = -1
            readyGeneration = 0
            retiring = false
            retirementResolve?.()
            retirementResolve = null
            retirementReject = null
            fence(0, false)
            cleanup()
        })
        removeConnect = transport.onConnect(() => {
            connected = true
            controlFailed = false
            driveControl()
        })
    }
    const getInitial = (): Promise<void> => {
        assertMutable()
        if (disposed) throw new Error('history-retired')
        if (blocked) throw new Error('history-invalid')
        wanted = true
        attach()
        const pending = newInitial()
        driveControl()
        return pending
    }
    const unsubscribe = (): Promise<void> => {
        assertMutable()
        if (wanted) {
            wanted = false
            fence(0, true)
        }
        if (!retirementCall)
            retirementCall = new Promise((resolve, reject) => {
                retirementResolve = resolve
                retirementReject = reject
            })
        driveControl()
        return retirementCall
    }
    return {
        subscribe: getInitial,
        getInitial,
        unsubscribe,
        get size(): number {
            return banks?.current.size ?? 0
        },
        get loading(): boolean {
            return wanted && (!initialDone || remoteLoading)
        },
        /** The record and all nested referents expire when this callback returns. */
        consumeCurrent(index: number, consume: RoutingHistoryConsumer): boolean {
            assertMutable()
            if (!banks || disposed) return false
            loan = true
            try {
                return banks.current.consume(index, consume)
            } finally {
                loan = false
            }
        },
        onChanged(callback: (token: RoutingHistoryCommit) => undefined): () => void {
            assertMutable()
            attach()
            const remove = transport.subscribePush('routing-history:committed', token => {
                if (
                    (!wanted && token.generation !== 0) ||
                    (wanted && token.generation !== generation)
                )
                    return
                if ('revision' in token && token.revision !== 0 && !pendingVisible) return
                if (callback(token) !== undefined) throw new Error('history-invalid')
            })
            return () => {
                assertMutable()
                remove()
            }
        },
        destroy(): void {
            assertMutable()
            if (disposed) return
            void unsubscribe().catch(() => {})
            disposed = true
            cleanup()
        }
    }
}
