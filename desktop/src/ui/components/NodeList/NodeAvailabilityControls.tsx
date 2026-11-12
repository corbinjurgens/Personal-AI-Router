// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react'
import { Badge, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core'
import { DismissibleTooltip } from '@/ui/components/DismissibleTooltip/DismissibleTooltip'
import { HaltConfirmModal } from '@/ui/components/HaltConfirmModal'
import { HALT_CONSEQUENCE } from '@/shared/constants/availability-copy'
import type { NodeAvailabilityRequest } from '@/shared/types/node-policy'
import {
    canResumeAvailability,
    useNodeConfigurationStore,
    watchNodeAvailability,
    watchNodeManagement
} from '@/ui/stores/node-configuration.store'
import { useConnectionStore } from '@/ui/stores/connection.store'

interface ControlsProps {
    name: string
    local: boolean
    enabled: boolean
    readable?: boolean
    entry: ReturnType<ReturnType<typeof useNodeConfigurationStore.getState>['availability']['get']>
    onRequest(request: NodeAvailabilityRequest): void
    onRefresh(): void
    nodeId?: string
}

/** What Pause does; shown on the Pause button rather than repeated on every card. */
function pauseExplanation(name: string): string {
    return `Pause bars execution and unloads models on ${name}; routing elsewhere continues. Interrupted requests may fail and need a new client request. File transfers and metadata reads continue.`
}

const HALT_EXPLANATION = `Halt: ${HALT_CONSEQUENCE}`

export function AvailabilityControls({
    name,
    local,
    enabled,
    readable = enabled,
    entry,
    nodeId,
    onRequest,
    onRefresh
}: ControlsProps) {
    const [haltConfirmOpen, setHaltConfirmOpen] = useState(false)
    const value = entry?.value
    const known = enabled && entry?.current && !entry.stale && value !== undefined
    const busy = entry?.pending !== undefined
    const resume = canResumeAvailability(entry, local)
    const halted = !local && entry?.pauseMode === 'immediate' && entry.pausedBy === 'local'
    const canResume = known && (!busy || (entry.pending === 'paused' && value === 'draining'))
    const canStop =
        known &&
        (!busy ||
            (entry.pending === 'paused' &&
                entry.pendingPauseMode === 'configured' &&
                value === 'draining'))
    const origin =
        value === 'available'
            ? ''
            : entry?.pausedBy === 'local'
              ? 'Paused from this device'
              : entry?.pausedBy === 'peer'
                ? 'Paused by a managing peer'
                : 'Pause origin unknown'
    const label =
        value === 'available'
            ? 'Available'
            : value === 'draining'
              ? 'Pausing'
              : value === 'paused'
                ? 'Paused'
                : 'Availability unknown'
    return (
        <Stack gap="2" className="min-w-0 w-full" aria-label={`Availability for ${name}`}>
            <Flex align="center" wrap="wrap" gap="2" className="min-w-0">
                <Badge
                    kind="solid"
                    color={
                        value === 'available' ? 'green' : value === 'draining' ? 'yellow' : 'gray'
                    }
                >
                    {entry?.stale
                        ? 'Availability stale'
                        : !entry?.current
                          ? 'Availability unconfirmed'
                          : label}
                </Badge>
                {(value === 'available' ||
                    !local ||
                    value === undefined ||
                    value === 'draining') && (
                    <DismissibleTooltip slotContent={pauseExplanation(name)}>
                        <Button
                            kind="secondary"
                            size="tiny"
                            className="max-w-full whitespace-normal break-words"
                            disabled={!known || busy || (!local && value !== 'available')}
                            aria-label={`${value === 'draining' && local ? 'Retry pause' : 'Pause node'}: ${name}`}
                            aria-description={pauseExplanation(name)}
                            aria-busy={entry?.pending === 'paused'}
                            onClick={() =>
                                onRequest({ nodeId, state: 'paused', pauseMode: 'configured' })
                            }
                        >
                            {value === 'draining' && local ? 'Retry pause' : 'Pause node'}
                        </Button>
                    </DismissibleTooltip>
                )}
                {resume && (
                    <Button
                        kind="secondary"
                        size="tiny"
                        disabled={!canResume}
                        aria-label={`Resume node: ${name}`}
                        aria-busy={entry?.pending === 'available'}
                        onClick={() => onRequest({ nodeId, state: 'available' })}
                    >
                        Resume
                    </Button>
                )}
                {local && (
                    <DismissibleTooltip slotContent={HALT_EXPLANATION}>
                        <Button
                            kind="secondary"
                            color="danger"
                            size="tiny"
                            className="max-w-full whitespace-normal break-words"
                            disabled={!canStop}
                            aria-description={HALT_EXPLANATION}
                            aria-busy={entry?.pendingPauseMode === 'immediate'}
                            onClick={() => setHaltConfirmOpen(true)}
                        >
                            Halt
                        </Button>
                    </DismissibleTooltip>
                )}
                <Button
                    kind="tertiary"
                    size="tiny"
                    disabled={!readable || entry?.reading}
                    aria-label={`Refresh availability: ${name}`}
                    onClick={onRefresh}
                >
                    Refresh
                </Button>
            </Flex>
            <Text
                kind="body/regular/sm"
                role="status"
                className="text-subtle-color break-words min-w-0"
            >
                {value && value !== 'available' && `${origin}. `}
                {halted && value !== 'available' && 'Halted on that node. '}
                {!local &&
                    !resume &&
                    !halted &&
                    (value === 'draining' || value === 'paused') &&
                    'Resume on that device. '}
                {entry?.stale && 'Readback timed out. Refresh availability manually. '}
            </Text>
            {entry?.error && (
                <Text kind="body/regular/sm" role="alert" className="break-words min-w-0">
                    {entry.error}
                </Text>
            )}
            {entry?.readError && (
                <Text kind="body/regular/sm" role="alert" className="break-words min-w-0">
                    {entry.readError}
                </Text>
            )}
            {local && (
                <HaltConfirmModal
                    open={haltConfirmOpen}
                    onOpenChange={setHaltConfirmOpen}
                    confirmDisabled={!canStop}
                    onConfirm={() => {
                        if (canStop) onRequest({ state: 'paused', pauseMode: 'immediate' })
                    }}
                />
            )}
        </Stack>
    )
}

export default function NodeAvailabilityControls({
    targetId,
    nodeId,
    name,
    online
}: {
    targetId: string
    nodeId?: string
    name: string
    online: boolean
}) {
    const connected = useConnectionStore(state => state.connected)
    const entry = useNodeConfigurationStore(state => state.availability.get(targetId))
    const management = useNodeConfigurationStore(state => state.management.get(targetId))
    const ref = useRef<HTMLDivElement>(null)
    const [visible, setVisible] = useState(true)
    useEffect(() => {
        if (!ref.current || typeof IntersectionObserver === 'undefined') return
        const observer = new IntersectionObserver(entries =>
            setVisible(entries.some(item => item.isIntersecting))
        )
        observer.observe(ref.current)
        return () => observer.disconnect()
    }, [])
    useEffect(() => {
        if (!visible) return
        const stop = watchNodeAvailability(targetId, nodeId, online)
        const stopManagement = watchNodeManagement(targetId, nodeId)
        return () => {
            stop()
            stopManagement()
        }
    }, [targetId, nodeId, online, visible])
    return (
        <div ref={ref} className="min-w-0 w-full">
            <AvailabilityControls
                name={name}
                local={nodeId === undefined}
                nodeId={nodeId}
                entry={entry}
                enabled={
                    connected && online && (nodeId === undefined || management?.state === 'allowed')
                }
                readable={connected && online}
                onRequest={request =>
                    void useNodeConfigurationStore
                        .getState()
                        .requestAvailability(targetId, request.state, request.pauseMode)
                }
                onRefresh={() =>
                    void useNodeConfigurationStore.getState().readAvailability(targetId, true)
                }
            />
            {nodeId !== undefined && management?.state !== 'allowed' && (
                <Text kind="body/regular/sm" role="status">
                    {management?.message ?? 'Checking remote management permission.'}
                </Text>
            )}
        </div>
    )
}
