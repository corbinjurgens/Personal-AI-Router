// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Button, FormField, Select, Stack, Text } from '@nvidia/foundations-react-core'
import type { SwitchEngineState } from '@/shared/types/engine-switch'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { useEngineSwitchStore, watchEngineSwitch } from '@/ui/stores/engine-switch.store'
import {
    useNodeConfigurationStore,
    watchNodeManagement
} from '@/ui/stores/node-configuration.store'

const COPY = {
    heading: 'Switch engine',
    source: 'From',
    target: 'To',
    submit: 'Switch',
    unknown: 'Unknown',
    on: 'On',
    off: 'Off',
    running: 'Running',
    stopped: 'Not running',
    healthy: 'Healthy',
    unhealthy: 'Unhealthy',
    verified: 'Termination verified',
    unverified: 'Termination not verified',
    intent: 'Saved intent',
    runtime: 'Runtime',
    health: 'Health',
    observed: 'Observed',
    unavailable: 'Unavailable'
}

function evidence(state: SwitchEngineState): string {
    return `${COPY.intent}: ${COPY[state.intent]}; ${COPY.runtime}: ${state.running === null ? COPY.unknown : state.running ? COPY.running : COPY.stopped}; ${COPY.health}: ${state.healthy === null ? COPY.unknown : state.healthy ? COPY.healthy : COPY.unhealthy}; ${COPY.observed}: ${state.observedAt ?? COPY.unavailable}; ${state.terminationVerified ? COPY.verified : COPY.unverified}`
}

export function EngineSwitchControls({
    nodeId,
    engines
}: {
    nodeId: string
    engines: { value: string; children: string }[]
}) {
    const [source, setSource] = useState('')
    const [target, setTarget] = useState('')
    const connected = useConnectionStore(state => state.connected)
    const local = useConnectionStore(state => state.selfId === nodeId)
    const offline = useNodesStore(state => state.nodes.get(nodeId)?.status === 'offline')
    const entry = useEngineSwitchStore(state => state.entries[nodeId])
    const pending = usePendingActionsStore(state => state.switches.has(nodeId))
    const management = useNodeConfigurationStore(state => state.management.get(nodeId))
    useEffect(() => watchEngineSwitch(nodeId), [nodeId])
    useEffect(() => watchNodeManagement(nodeId, local ? undefined : nodeId), [nodeId, local])
    const receipt = entry?.receipt
    const reason =
        !connected || offline
            ? 'Device disconnected. Reconnect before switching.'
            : !local && management?.state !== 'allowed'
              ? (management?.message ?? 'Checking remote management permission.')
              : pending
                ? 'Reading and submitting the switch request.'
                : receipt?.recoveryRequired
                  ? 'Recovery required on this device. Refresh only reads status.'
                  : entry?.unknown
                    ? (entry.unavailable ?? 'Switch outcome unknown. Refresh readback.')
                    : entry?.activeRequestId || (receipt && !receipt.terminal)
                      ? (entry.unavailable ?? 'A switch is active on this device.')
                      : !entry?.revision || entry.unavailable
                        ? (entry?.unavailable ??
                          'Switch readback unavailable. Refresh before switching.')
                        : !source || !target
                          ? 'Select installed source and target engines.'
                          : source === target
                            ? 'From and To must be different engines.'
                            : !engines.some(engine => engine.value === source) ||
                                !engines.some(engine => engine.value === target)
                              ? 'A selected engine is no longer reported installed.'
                              : undefined
    const error = receipt?.error
    const errorLabel =
        error?.code === -32012
            ? 'Remote management permission denied'
            : error?.code === -32011
              ? 'Local Off veto'
              : error?.code === -32010
                ? 'Configuration conflict'
                : 'Switch failed'
    return (
        <Stack gap="3" className="pair-paper min-w-0 max-w-full p-4">
            <Text kind="body/semibold/md">{COPY.heading}</Text>
            <div className="grid min-w-0 grid-cols-1 gap-3 sm:grid-cols-2">
                <FormField slotLabel={COPY.source}>
                    <Select
                        size="small"
                        attributes={{ SelectTrigger: { 'aria-label': COPY.source } }}
                        items={engines}
                        value={source}
                        onValueChange={setSource}
                        placeholder="Select engine"
                    />
                </FormField>
                <FormField slotLabel={COPY.target}>
                    <Select
                        size="small"
                        attributes={{ SelectTrigger: { 'aria-label': COPY.target } }}
                        items={engines}
                        value={target}
                        onValueChange={setTarget}
                        placeholder="Select engine"
                    />
                </FormField>
            </div>
            <Stack gap="2" className="min-w-0 break-all" aria-live="polite">
                <Text kind="body/regular/sm">Node: {nodeId}</Text>
                {entry?.request && (
                    <Text kind="body/regular/sm">
                        {entry.own ? 'Submitted request' : 'Observed activity'}:{' '}
                        {entry.request.requestId}; {entry.request.sourceEngine} →{' '}
                        {entry.request.targetEngine}; revision {entry.request.expectedRevision}
                    </Text>
                )}
                {entry?.activeRequestId && !entry.request && (
                    <Text kind="body/regular/sm">
                        Active request: {entry.activeRequestId}; receipt unavailable.
                    </Text>
                )}
                {receipt && (
                    <Text kind="body/regular/sm">
                        Phase: {receipt.phase}; sequence {receipt.sequence}
                    </Text>
                )}
                {error && (
                    <Text kind="body/regular/sm" role="alert">
                        {errorLabel}: {error.message} ({error.code}, {error.kind})
                    </Text>
                )}
                {receipt?.recoveryRequired && (
                    <Text kind="body/regular/sm" role="alert">
                        Recovery required. Readback does not perform recovery.
                    </Text>
                )}
                {receipt && (
                    <>
                        <Text kind="body/regular/sm">
                            Source {receipt.request.sourceEngine}: {evidence(receipt.source)}
                        </Text>
                        <Text kind="body/regular/sm">
                            Target {receipt.request.targetEngine}: {evidence(receipt.target)}
                        </Text>
                    </>
                )}
                {entry?.observed && (
                    <>
                        <Text kind="body/regular/sm">
                            Current source observation: {evidence(entry.observed.source)}
                        </Text>
                        <Text kind="body/regular/sm">
                            Current target observation: {evidence(entry.observed.target)}
                        </Text>
                    </>
                )}
                {reason && (
                    <Text kind="body/regular/sm" role="status">
                        {reason}
                    </Text>
                )}
            </Stack>
            <Stack direction="row" gap="2">
                <Button
                    kind="secondary"
                    size="small"
                    disabled={!connected || offline || pending}
                    onClick={() => void useEngineSwitchStore.getState().refresh(nodeId)}
                >
                    Refresh readback
                </Button>
                <Button
                    kind="primary"
                    size="small"
                    disabled={!!reason}
                    onClick={() => {
                        if (!reason)
                            void useEngineSwitchStore.getState().submit(nodeId, source, target)
                    }}
                >
                    {COPY.submit}
                </Button>
            </Stack>
        </Stack>
    )
}
