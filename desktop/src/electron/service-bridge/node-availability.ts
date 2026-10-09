// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NodeAvailability } from '@/shared/types/node-policy'
import type { JsonValue } from './json-rpc-client'

/** What the tray's pause toggle shows and does for an availability. */
export interface AvailabilityMenuItem {
    label: string
    enabled: boolean
    /** The state a click requests, or null when the item cannot be clicked. */
    request: 'available' | 'paused' | null
}

const PAUSE_LABEL = 'Pause inference on this PC'
const RESUME_LABEL = 'Resume inference on this PC'
const PAUSING_LABEL = 'Pausing…'

export function availabilityValue(value: JsonValue | undefined): NodeAvailability | null {
    return value === 'available' || value === 'draining' || value === 'paused' ? value : null
}

function objectValue(value: JsonValue | undefined): Record<string, JsonValue> | null {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return null
    return value
}

/** The tray item for an availability; null (not yet known) shows a disabled Pause. */
export function availabilityMenuItem(availability: NodeAvailability | null): AvailabilityMenuItem {
    switch (availability) {
        case 'available':
            return { label: PAUSE_LABEL, enabled: true, request: 'paused' }
        case 'paused':
            return { label: RESUME_LABEL, enabled: true, request: 'available' }
        case 'draining':
            return { label: PAUSING_LABEL, enabled: false, request: null }
        case null:
            return { label: PAUSE_LABEL, enabled: false, request: null }
    }
}

/**
 * This node's availability, kept current from `policy:get` (read when the broker
 * becomes ready) and `node:availability-changed` pushes. The broker also relays
 * other nodes' changes to its clients under their node ids, so a push is applied
 * only when it names this node.
 */
export class NodeAvailabilityState {
    private current: NodeAvailability | null = null
    private readonly listeners = new Set<() => void>()

    constructor(private readonly getSelfId: () => string | null) {}

    get availability(): NodeAvailability | null {
        return this.current
    }

    menuItem(): AvailabilityMenuItem {
        return availabilityMenuItem(this.current)
    }

    /** Register for changes; returns the unsubscribe function. */
    onChange(listener: () => void): () => void {
        this.listeners.add(listener)
        return () => this.listeners.delete(listener)
    }

    /** Apply a `policy:get` or `node:set-availability` result; both carry `availability`. */
    applyResult(result: JsonValue | undefined): void {
        const next = availabilityValue(objectValue(result)?.availability)
        if (next) this.set(next)
    }

    /** Apply `node:availability-changed {nodeId, availability, active}`. */
    applyNotification(params: JsonValue | undefined): void {
        const body = objectValue(params)
        if (!body) return
        const selfId = this.getSelfId()
        if (selfId === null || body.nodeId !== selfId) return
        const next = availabilityValue(body.availability)
        if (next) this.set(next)
    }

    /** Forget the state, as when the broker is gone and the real one is unknown. */
    reset(): void {
        this.set(null)
    }

    private set(next: NodeAvailability | null): void {
        if (next === this.current) return
        this.current = next
        for (const listener of Array.from(this.listeners)) listener()
    }
}
