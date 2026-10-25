// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BrowserWindow } from 'electron'
import {
    RoutingHistoryBank, routingHistoryRecords,
    type RoutingHistoryConsumed, type RoutingHistoryConsumption, type RoutingHistoryPull,
    type RoutingHistoryReady, type RoutingHistoryTransfer
} from '@/shared/types/routing'

interface Renderer {
    readonly id: number
    // Four scalar slots on the existing BrowserWindow registration. No second
    // registry, per-context object, cursor map or captured lifecycle closure.
    routingGeneration?: number
    routingCursor?: number
    routingLease?: number
    routingFlags?: number
    readonly webContents: {
        send(channel: string, message: RoutingHistoryReady): void
        isDestroyed(): boolean
    }
    isDestroyed(): boolean
}
const wanted = 2, ready = 4, retired = 8, dirty = 16, baseline = 32, reportedLoading = 64, ending = 128

/** The sole main cache and delivery owner; lifecycle listeners retain no data bank. */
class RoutingDelivery {
    private banks: { current: RoutingHistoryBank; pending: RoutingHistoryBank; baseline: RoutingHistoryBank } | null = null
    private generation = 0
    private revision = 0
    private lease = 0
    // A generation owns the frozen baseline across every packet and scalar end,
    // including gaps with no active packet lease. No renderer reference is held.
    private initialOwner = 0
    private initialRevision = 0
    private loading = true
    private demand: ((active: boolean) => void) | null = null
    private wantedCount = 0
    private retiredThrough = 0
    private rendererFrontier = 0
    private scanCursor = 1
    private scanWake: ReturnType<typeof setImmediate> | null = null
    private scanAgain = false

    private ownedBanks() {
        return this.banks ??= {
            current: new RoutingHistoryBank(), pending: new RoutingHistoryBank(), baseline: new RoutingHistoryBank()
        }
    }
    get capacity() {
        return {
            windowBytes: this.banks ? this.banks.current.capacity.windowBytes + this.banks.pending.capacity.windowBytes + this.banks.baseline.capacity.windowBytes : 0,
            controlBytes: this.banks ? this.banks.current.capacity.controlBytes + this.banks.pending.capacity.controlBytes + this.banks.baseline.capacity.controlBytes : 0,
            initialLeases: this.initialOwner === 0 ? 0 : 1
        }
    }
    setDemandListener(listener: (active: boolean) => void): void { this.demand = listener }
    private notifyDemand(): void {
        if (this.wantedCount === 0) {
            if (this.scanWake) clearImmediate(this.scanWake)
            this.scanWake = null; this.scanCursor = 1; this.scanAgain = false
        }
        this.demand?.(this.wantedCount > 0)
    }
    private activate(sender: Renderer): void {
        if (sender.routingLease || ((sender.routingFlags ?? 0) & ready) || (this.initialOwner !== 0 && sender.routingGeneration === this.initialOwner)) return
        if (this.generation === Number.MAX_SAFE_INTEGER) throw new Error('history-exhausted')
        sender.routingGeneration = ++this.generation
        sender.routingCursor = 0
        sender.routingLease = 0
        sender.routingFlags = wanted | dirty | baseline
    }
    subscribe(sender: Renderer): number {
        if (sender.isDestroyed() || sender.webContents.isDestroyed()) throw new Error('history-invalid')
        if (!Number.isSafeInteger(sender.id) || sender.id < 1) throw new Error('history-invalid')
        this.rendererFrontier = Math.max(this.rendererFrontier, sender.id)
        if (!((sender.routingFlags ?? 0) & wanted)) this.wantedCount++
        sender.routingFlags = (sender.routingFlags ?? 0) | wanted | dirty
        if (!sender.routingGeneration || (sender.routingFlags & retired)) this.activate(sender)
        this.notifyDemand()
        this.signal(sender)
        return sender.routingGeneration ?? 0
    }
    unsubscribe(sender: Renderer): void {
        if ((sender.routingFlags ?? 0) & wanted) this.wantedCount--
        sender.routingFlags = ((sender.routingFlags ?? 0) & ~wanted) | retired
        this.notifyDemand()
        // Reuse the one ready slot for retirement too. The renderer drains an
        // actually delivered token before it detaches; no second queued token
        // or assumption about IPC reply/event ordering is needed.
        this.signal(sender)
    }
    contextDestroyed(sender: Renderer): void {
        if ((sender.routingFlags ?? 0) & wanted) this.wantedCount--
        if (sender.routingGeneration === this.initialOwner) this.releaseInitial()
        // Called by an existing window owner at actual destruction/crash. It
        // reads only scalar registration state, never a destroyed native getter.
        delete sender.routingGeneration; delete sender.routingCursor; delete sender.routingLease; delete sender.routingFlags
        this.notifyDemand()
        this.signalAll()
    }
    setLoading(loading: boolean): void {
        this.loading = loading
        this.signalAll()
    }
    reset(): void {
        if (this.scanWake) clearImmediate(this.scanWake)
        this.scanWake = null; this.scanCursor = 1; this.scanAgain = false
        this.banks?.current.clear(); this.banks?.pending.clear()
        if (!this.initialOwner) this.banks?.baseline.clear()
        this.loading = true
        // A scalar frontier fences every old apply/ack immediately, before a
        // fair registry scan. It never reclaims queued tokens or transfer bytes.
        this.retiredThrough = this.generation
        this.signalAll()
    }
    record(raw: Uint8Array): void {
        if (this.revision === Number.MAX_SAFE_INTEGER) throw new Error('history-exhausted')
        if (!this.ownedBanks().current.apply(raw, this.revision + 1)) return
        this.revision++
        this.signalAll()
    }
    initialResult(raw: Uint8Array): void {
        const banks = this.ownedBanks()
        banks.pending.copyFrom(banks.current)
        let revision = this.revision
        routingHistoryRecords(raw, row => {
            if (revision === Number.MAX_SAFE_INTEGER) throw new Error('history-exhausted')
            if (banks.pending.apply(row, revision + 1)) revision++
            return undefined
        })
        const old = banks.current
        banks.current = banks.pending; banks.pending = old
        banks.pending.clear()
        this.revision = revision
        this.setLoading(false)
    }
    private releaseInitial(): void {
        this.initialOwner = 0; this.initialRevision = 0
        this.banks?.baseline.clear()
    }
    pull(sender: Renderer, request: RoutingHistoryPull): RoutingHistoryTransfer {
        if (!request || !Number.isSafeInteger(request.generation) || request.generation < 1 || typeof request.initial !== 'boolean') throw new Error('history-invalid')
        const generation = sender.routingGeneration ?? 0
        const flags = sender.routingFlags ?? 0
        if (sender.isDestroyed() || request.generation !== generation || !(flags & ready)) return { state: 'busy', generation, loading: this.loading }
        // Only a pull caused by the delivered token consumes that token. The
        // renderer helper never speculatively pulls from a subscribe response.
        sender.routingFlags = flags & ~ready
        if ((flags & retired) || generation <= this.retiredThrough) {
            if (generation === this.initialOwner) return this.endInitial(sender, true)
            if (!sender.routingLease) {
                sender.routingGeneration = 0
                if (flags & wanted) this.activate(sender)
            }
            this.signal(sender)
            return { state: 'retired', generation: sender.routingGeneration ?? 0 }
        }
        if (!(flags & wanted) || sender.routingLease || (this.loading && generation !== this.initialOwner) || (request.initial && this.initialOwner && generation !== this.initialOwner)) return { state: 'busy', generation, loading: this.loading || request.initial }
        if (this.lease === Number.MAX_SAFE_INTEGER) throw new Error('history-exhausted')
        if (request.initial) {
            if (!(flags & baseline)) throw new Error('history-invalid')
            if (!this.initialOwner) {
                this.ownedBanks().baseline.copyFrom(this.ownedBanks().current)
                this.initialOwner = generation; this.initialRevision = this.revision
                sender.routingCursor = 0
            }
            const frozen = this.ownedBanks().baseline, index = sender.routingCursor ?? 0
            if (index === frozen.size) return this.endInitial(sender, false)
            sender.routingLease = ++this.lease
            sender.routingCursor = index + 1
            return { state: 'data', generation, lease: sender.routingLease, initial: true, index, bytes: frozen.at(index).slice() }
        }
        if (generation === this.initialOwner) throw new Error('history-invalid')
        const bank = this.ownedBanks().current
        const row = bank.after(sender.routingCursor ?? 0)
        if (!row) {
            sender.routingFlags &= ~dirty
            sender.routingCursor = this.revision
            return { state: 'busy', generation, loading: false }
        }
        // Reserve before the compact copy/IPC post. The serialized buffer must
        // not inherit the full H projector backing. The protocol reservation
        // lasts through exact commit acknowledgement or actual destruction;
        // this copy does not establish native IPC/contextBridge retention bounds.
        sender.routingLease = ++this.lease
        sender.routingCursor = row.cursor
        sender.routingFlags &= ~dirty
        return { state: 'data', generation, lease: sender.routingLease, initial: false, bytes: row.raw.slice() }
    }
    private endInitial(sender: Renderer, wasRetired: boolean): RoutingHistoryTransfer {
        if (sender.routingLease || this.lease === Number.MAX_SAFE_INTEGER) throw new Error('history-exhausted')
        sender.routingLease = ++this.lease
        sender.routingFlags = (sender.routingFlags ?? 0) | ending
        return { state: 'end', generation: sender.routingGeneration ?? 0, lease: sender.routingLease, count: sender.routingCursor ?? 0, retired: wasRetired }
    }
    consumed(sender: Renderer, ack: RoutingHistoryConsumed): RoutingHistoryConsumption {
        const stale: RoutingHistoryConsumption = { state: 'stale', generation: sender.routingGeneration ?? 0 }
        if (!ack || sender.isDestroyed() || !Number.isSafeInteger(ack.generation) || !Number.isSafeInteger(ack.lease) ||
            ack.generation < 1 || ack.lease < 1 || sender.routingGeneration !== ack.generation || sender.routingLease !== ack.lease) return stale
        sender.routingLease = 0
        const wasRetired = !!((sender.routingFlags ?? 0) & retired) || ack.generation <= this.retiredThrough
        const initial = this.initialOwner === ack.generation
        const ended = initial && !!((sender.routingFlags ?? 0) & ending)
        if (ended) {
            sender.routingCursor = this.initialRevision
            sender.routingFlags = (sender.routingFlags ?? 0) & ~(baseline | ending)
            this.releaseInitial()
        }
        if (wasRetired && (!initial || ended)) {
            sender.routingGeneration = 0
            if ((sender.routingFlags ?? 0) & wanted) this.activate(sender)
        }
        this.signal(sender)
        this.signalAll()
        return { state: wasRetired && (!initial || ended) ? 'retired' : 'accepted', generation: sender.routingGeneration ?? 0 }
    }
    private signal(sender: Renderer): void {
        let flags = sender.routingFlags ?? 0
        if (sender.isDestroyed() || sender.webContents.isDestroyed() || flags & ready || sender.routingLease) return
        if (this.initialOwner !== 0 && sender.routingGeneration === this.initialOwner) {
            sender.routingFlags = flags | ready
            sender.webContents.send('routing-history:ready', { generation: this.initialOwner })
            return
        }
        if (!(flags & wanted)) {
            if ((flags & retired) && sender.routingGeneration) {
                sender.routingFlags = flags | ready
                sender.webContents.send('routing-history:ready', { generation: sender.routingGeneration })
            }
            return
        }
        if ((flags & retired) || (sender.routingGeneration ?? 0) <= this.retiredThrough) {
            this.activate(sender); flags = sender.routingFlags ?? 0
        }
        if (this.loading) {
            if (flags & reportedLoading) return
            sender.routingFlags = flags | ready | reportedLoading
            sender.webContents.send('routing-history:ready', { generation: sender.routingGeneration ?? 0 })
            return
        }
        if (flags & reportedLoading) { flags = (flags & ~reportedLoading) | dirty; sender.routingFlags = flags }
        if ((flags & baseline) && this.initialOwner) return
        if (!(flags & dirty) && this.revision <= (sender.routingCursor ?? 0)) return
        sender.routingFlags = flags | ready
        sender.webContents.send('routing-history:ready', { generation: sender.routingGeneration })
    }
    private signalAll(): void {
        this.scanAgain = true
        if (!this.scanWake) this.scanStep()
    }
    private scanStep(): void {
        if (this.wantedCount === 0) { this.scanCursor = 1; this.scanAgain = false; return }
        if (this.scanCursor === 1) this.scanAgain = false
        let examined = 0
        while (this.scanCursor <= this.rendererFrontier && examined++ < 256) {
            const sender = BrowserWindow.fromId(this.scanCursor++)
            if (sender) this.signal(sender)
        }
        if (this.scanCursor > this.rendererFrontier) {
            this.scanCursor = 1
            if (!this.scanAgain) return
        }
        // Exactly one coalesced fair wake. No subscriber list or native-inventory
        // array, even with arbitrarily many or widely spaced registration IDs.
        this.scanWake = setImmediate(() => { this.scanWake = null; this.scanStep() })
    }
}

let delivery: RoutingDelivery | null = null
export function getRoutingDelivery(): RoutingDelivery { return delivery ??= new RoutingDelivery() }
