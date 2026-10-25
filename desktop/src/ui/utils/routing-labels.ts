// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RoutingRequestKey } from '@/shared/types/node-policy'
import type {
    RoutingHistoryAttempt,
    RoutingHistoryDestination,
    RoutingHistoryOutput,
    RoutingHistoryReason
} from '@/shared/types/routing'
import type { Workload } from '@/shared/types/workloads'
import { engineManagerName } from '@/shared/utils/engines'

const reasons: Record<RoutingHistoryReason, string> = {
    '': '',
    unknown: 'Unknown',
    'client-gone': 'Client disconnected',
    cancel: 'Cancelled by user',
    regenerate: 'Regenerated',
    'node-paused': 'Node paused',
    'proxy-shutdown': 'Proxy shutting down',
    'proxy-exited': 'Proxy exited',
    'target-left': 'Node left the cluster',
    transport: 'Connection error',
    'upstream-status': 'Engine returned an error status',
    'no-content': 'No output before timeout',
    'stream-interrupted': 'Stream interrupted',
    'upstream-error': 'Engine error',
    'admission-paused': 'Node paused',
    'admission-draining': 'Node draining',
    'admission-busy': 'Node busy',
    'admission-no-fit': 'Model does not fit',
    'admission-engine-off': 'Engine off',
    'admission-wake-timeout': 'Engine wake timed out',
    'engine-not-running': 'Engine not running',
    'request-invalid': 'Request could not be adapted',
    'no-owner': 'No node serves this model',
    'regenerate-exhausted': 'Too many regenerations',
    'attempts-exhausted': 'All attempts used',
    deadline: 'Request deadline reached',
    'request-ceiling': 'Request spend ceiling reached',
    'account-ceiling': 'Account ceiling reached',
    'account-cooldown': 'Account cooling down',
    'account-frozen': 'Account frozen',
    'account-credential': 'Account credential failed',
    'account-exhausted': 'Account exhausted',
    'account-conflict': 'Account conflict',
    'ledger-unavailable': 'Spend ledger unavailable',
    'deployment-quarantined': 'Deployment quarantined',
    'peer-framing': 'Peer sent a malformed frame'
}

const reasonLabels = new Map(Object.entries(reasons))

export function routingReason(reason: string): string {
    return reasonLabels.get(reason) ?? 'Unknown'
}

export function routingRequestKey(key: RoutingRequestKey): string {
    return `${key.originNodeId}/${key.proxyRunId}/${key.requestId}`
}

export function routingMoney(value: string | null): string {
    if (value === null) return '—'
    const padded = value.padStart(10, '0')
    const whole = padded.slice(0, -9)
    const fraction = padded.slice(-9).replace(/0+$/, '')
    return `$${whole}${fraction ? `.${fraction}` : ''}`
}

export function routingWait(value: number | null): string {
    if (value === null) return '—'
    return value < 1000 ? `${value} ms` : `${(value / 1000).toFixed(1)} s`
}

export function routingDestination(
    destination: Readonly<RoutingHistoryDestination> | null
): string {
    if (!destination) return '—'
    return destination.kind === 'local'
        ? `${destination.nodeId} / ${destination.engine}`
        : `deployment ${destination.deploymentId} / account ${destination.accountId}`
}

const states = {
    routing: 'Routing',
    streaming: 'Streaming',
    completed: 'Completed',
    failed: 'Failed',
    cancelled: 'Cancelled'
}

export function routingState(record: RoutingHistoryOutput): string {
    return record.recoverable ? 'Streaming paused, waiting to resume' : states[record.state]
}

export function routingAttemptOutcome(
    record: RoutingHistoryOutput,
    attempt: Readonly<RoutingHistoryAttempt>
): string {
    if (attempt.outcome === 'interrupted') {
        if (record.attempts.some(next => next.continuationOf === attempt.index))
            return 'Interrupted, resumed'
        if (record.recoverable) return 'Streaming paused, waiting to resume'
        return 'Interrupted'
    }
    const outcomes = {
        pending: 'In progress',
        committed: 'Served',
        rejected: 'Rejected',
        failed: 'Failed',
        redispatched: 'Re-dispatched',
        cancelled: 'Cancelled'
    }
    return outcomes[attempt.outcome]
}

export function routingDisplayReason(record: RoutingHistoryOutput, reason: string): string {
    if (reason === 'stream-interrupted' && (record.state !== 'failed' || record.recoverable))
        return ''
    return routingReason(reason)
}

export function routingSummary(record: RoutingHistoryOutput): string {
    if (record.recoverable) return 'Streaming paused, waiting to resume'
    switch (record.state) {
        case 'completed': {
            const count = record.droppedAttempts + record.attempts.length
            return `Routing: ${count} attempt${count === 1 ? '' : 's'}, served on ${routingDestination(record.served)}`
        }
        case 'failed':
            return `Routing failed: ${routingReason(record.reason)}`
        case 'cancelled':
            return `Routing cancelled: ${routingReason(record.reason)}`
        case 'routing':
            return `Routing: attempt ${record.droppedAttempts + record.attempts.length}`
        case 'streaming':
            return 'Streaming'
    }
}

/** Returns presentation text only; never retains a borrowed history record. */
export function workloadRoutingSummary(
    record: RoutingHistoryOutput,
    workload: Workload
): { text: string; error: boolean } | null {
    const association = record.workload
    if (
        !association ||
        !workload.runId ||
        association.originNodeId !== workload.originatedFrom ||
        association.proxyRunId !== workload.runId ||
        association.engine !== engineManagerName(workload.engine) ||
        association.id !== workload.id
    )
        return null
    const attempt = record.attempts.find(value => value.index === association.attemptIndex)
    if (!attempt || attempt.destination.kind !== 'local') return null
    if (!association.attached) {
        const outcome =
            attempt.outcome === 'interrupted'
                ? record.attempts.some(next => next.continuationOf === attempt.index)
                    ? 'Interrupted, resumed'
                    : 'Interrupted'
                : routingAttemptOutcome(record, attempt)
        const reason = attempt.reason === 'stream-interrupted' ? '' : routingReason(attempt.reason)
        return {
            text: `Routing: local attempt ${attempt.index + 1}, ${outcome}${reason ? `: ${reason}` : ''}`,
            error: attempt.outcome === 'failed' || attempt.outcome === 'rejected'
        }
    }
    const error =
        !record.recoverable &&
        (attempt.outcome === 'failed' ||
            attempt.outcome === 'rejected' ||
            (attempt.outcome === 'interrupted' && record.state === 'failed'))
    return {
        text: error
            ? `Routing failed: ${routingDisplayReason(record, attempt.reason) || routingAttemptOutcome(record, attempt)}`
            : routingSummary(record),
        error
    }
}
