// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core'
import type { RoutingHistoryAttempt, RoutingHistoryOutput } from '@/shared/types/routing'
import {
    routingAttemptOutcome,
    routingDestination,
    routingDisplayReason,
    routingMoney,
    routingWait
} from '@/ui/utils/routing-labels'

/** Called within the bank loan, rather than passing borrowed DTOs to React. */
export default function RoutingAttempt(
    record: RoutingHistoryOutput,
    attempt: Readonly<RoutingHistoryAttempt>
) {
    return (
        <Stack
            key={attempt.index}
            gap="1"
            className="min-w-0 break-words border-t border-base py-3 [overflow-wrap:anywhere]"
        >
            <Text kind="body/bold/sm">
                Attempt {attempt.index + 1}: {routingAttemptOutcome(record, attempt)}
            </Text>
            <Text kind="body/regular/sm">
                Destination: {routingDestination(attempt.destination)}
            </Text>
            <Text kind="body/regular/sm">
                Model: {attempt.destination.model ?? 'model not specified'}
            </Text>
            <Text kind="body/regular/sm">
                Reason: {routingDisplayReason(record, attempt.reason) || '—'}
            </Text>
            <Text kind="body/regular/sm">
                Admission: {routingWait(attempt.waits.admissionMs)} · First byte:{' '}
                {routingWait(attempt.waits.firstByteMs)} · HTTP status: {attempt.status ?? '—'}
            </Text>
            <Text kind="body/regular/sm">Account state: {attempt.account?.state ?? '—'}</Text>
            <Text kind="body/regular/sm">
                Current held: {routingMoney(attempt.account?.heldNanoUsd ?? null)} · Charged:{' '}
                {routingMoney(attempt.account?.chargedNanoUsd ?? null)} · Accounting:{' '}
                {attempt.account?.accounting ?? 'none'}
            </Text>
        </Stack>
    )
}
