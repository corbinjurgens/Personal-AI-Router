// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Fragment, useState, type ReactNode } from 'react'
import {
    Button,
    Flex,
    Stack,
    Text,
    TableRoot,
    TableHead,
    TableBody,
    TableRow,
    TableHeaderCell,
    TableDataCell
} from '@nvidia/foundations-react-core'
import RoutingAttempt from './RoutingAttempt'
import { consumeRoutingRow, useRoutingStore } from '@/ui/stores/routing.store'
import {
    routingDestination,
    routingDisplayReason,
    routingMoney,
    routingRequestKey,
    routingState,
    routingSummary
} from '@/ui/utils/routing-labels'

function clipped(text: string) {
    return (
        <span className="block max-w-48 truncate" title={text} aria-label={text}>
            {text}
        </span>
    )
}

export default function RoutingHistory() {
    const revision = useRoutingStore(state => state.revision)
    const connected = useRoutingStore(state => state.connected)
    const loading = useRoutingStore(state => state.loading)
    const error = useRoutingStore(state => state.error)
    const [expanded, setExpanded] = useState<string | null>(null)
    const rows: ReactNode[] = []
    // revision is a scalar invalidation token; the API remains the sole data owner.
    void revision
    if (connected)
        for (let index = 0; index < window.pairApi.routing.size; index++) {
            consumeRoutingRow(index, record => {
                const key = routingRequestKey(record.request)
                const open = expanded === key
                const detailId = `routing-attempts-${key}`
                rows.push(
                    <Fragment key={key}>
                        <TableRow>
                            <TableDataCell>
                                <time dateTime={new Date(record.createdAt).toISOString()}>
                                    {new Date(record.createdAt).toLocaleString()}
                                </time>
                            </TableDataCell>
                            <TableDataCell>
                                {clipped(record.requested.model || 'model not specified')}
                            </TableDataCell>
                            <TableDataCell>{routingState(record)}</TableDataCell>
                            <TableDataCell>
                                <Stack gap="1">
                                    <span>
                                        {record.attempts.length}
                                        {record.droppedAttempts
                                            ? ` (+${record.droppedAttempts} older)`
                                            : ''}
                                    </span>
                                    <Button
                                        kind="tertiary"
                                        size="tiny"
                                        aria-expanded={open}
                                        aria-controls={detailId}
                                        aria-label={`${open ? 'Hide' : 'Show'} attempts for request ${key}`}
                                        onClick={() => setExpanded(open ? null : key)}
                                    >
                                        {open ? 'Hide attempts' : 'Show attempts'}
                                    </Button>
                                </Stack>
                            </TableDataCell>
                            <TableDataCell>
                                {clipped(routingDestination(record.served))}
                            </TableDataCell>
                            <TableDataCell>
                                <Stack gap="0">
                                    <span>
                                        Ceiling: {routingMoney(record.spend.ceilingNanoUsd)}
                                    </span>
                                    <span>
                                        Cumulative granted:{' '}
                                        {routingMoney(record.spend.grantedNanoUsd)}
                                    </span>
                                    <span>
                                        Current held: {routingMoney(record.spend.heldNanoUsd)}
                                    </span>
                                    <span>
                                        Charged: {routingMoney(record.spend.chargedNanoUsd)}
                                    </span>
                                    <span>Accounting: {record.spend.accounting}</span>
                                </Stack>
                            </TableDataCell>
                            <TableDataCell>
                                {routingDisplayReason(record, record.reason) || '—'}
                            </TableDataCell>
                        </TableRow>
                        <TableRow hidden={!open}>
                            <TableDataCell colSpan={7}>
                                <Stack
                                    id={detailId}
                                    gap="2"
                                    className="min-w-0 whitespace-normal [overflow-wrap:anywhere]"
                                >
                                    {open && (
                                        <>
                                            <Text kind="body/bold/sm">
                                                {routingSummary(record)}
                                            </Text>
                                            <Text kind="body/regular/sm">Request: {key}</Text>
                                            {record.attempts.map(attempt =>
                                                RoutingAttempt(record, attempt)
                                            )}
                                        </>
                                    )}
                                </Stack>
                            </TableDataCell>
                        </TableRow>
                    </Fragment>
                )
                return undefined
            })
        }
    return (
        <Stack gap="3" className="settings-card pair-paper min-w-0 max-w-full p-4">
            <Flex gap="2" justify="between" align="center" wrap="wrap">
                <Text kind="body/bold/lg">Routing history</Text>
                <Button
                    kind="secondary"
                    size="small"
                    disabled={!connected || loading}
                    onClick={() => void useRoutingStore.getState().refresh()}
                >
                    Refresh history
                </Button>
            </Flex>
            <Text kind="body/regular/sm" className="text-subtle-color">
                Private to this machine. Kept in memory; broker restart clears history. Newest 256
                requests.
            </Text>
            {!connected ? (
                <Text role="status">Disconnected</Text>
            ) : (
                <>
                    {loading && <Text role="status">Loading routing history…</Text>}
                    {error && <Text role="status">Could not load routing history.</Text>}
                    {!loading && !error && rows.length === 0 && (
                        <Text>No routed requests yet.</Text>
                    )}
                    {rows.length > 0 && (
                        <div
                            className="min-w-0 max-w-full overflow-x-auto"
                            tabIndex={0}
                            role="region"
                            aria-label="Routing history table"
                        >
                            <TableRoot
                                density="compact"
                                className="min-w-[960px] w-full table-fixed [&_td]:whitespace-normal"
                            >
                                <TableHead>
                                    <TableRow>
                                        {[
                                            'When',
                                            'Model',
                                            'State',
                                            'Attempts',
                                            'Served',
                                            'Spend',
                                            'Reason'
                                        ].map(label => (
                                            <TableHeaderCell
                                                key={label}
                                                className={label === 'Spend' ? 'w-1/5' : undefined}
                                            >
                                                {label}
                                            </TableHeaderCell>
                                        ))}
                                    </TableRow>
                                </TableHead>
                                <TableBody>{rows}</TableBody>
                            </TableRoot>
                        </div>
                    )}
                </>
            )}
        </Stack>
    )
}
