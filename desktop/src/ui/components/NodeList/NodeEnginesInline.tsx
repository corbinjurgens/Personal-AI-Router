// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EngineType } from '@/shared/types/engines'
import { engineLabel } from '@/shared/utils/engines'
import { engineCapabilities } from '@/ui/constants/engine-capabilities'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { engineTypesForNode, getEnginesForNode } from '@/ui/utils/get-engines-for-node'
import { Button, Flex, FormField, Switch, Text, TextInput } from '@nvidia/foundations-react-core'
import { Download, Lock, LockOpen } from '@/ui/components/icons'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { EngineCredentialState } from '@/shared/types/engine-api'

function CredentialControl({
    nodeId,
    engineType,
    name
}: {
    nodeId: string
    engineType: EngineType
    name: string
}) {
    const input = useRef<HTMLInputElement>(null)
    const generation = useRef(0)
    const mutation = useRef<{ cancelled: boolean } | null>(null)
    const [state, setState] = useState<EngineCredentialState | null>(null)
    const [confirmed, setConfirmed] = useState(false)
    const [error, setError] = useState('')
    const [pending, setPending] = useState(false)

    const readState = useCallback(() => {
        const current = ++generation.current
        if (input.current) input.current.value = ''
        setPending(true)
        setError('')
        window.pairApi.engines
            .credentialState({ nodeId, engineType })
            .then(result => {
                if (generation.current === current) {
                    setState(result)
                    setConfirmed(true)
                }
            })
            .catch(() => {
                if (generation.current === current) setConfirmed(false)
            })
            .finally(() => {
                if (generation.current === current) setPending(false)
            })
    }, [nodeId, engineType])

    // KUI merges refs anew on rerender. Ref detach is not field teardown.
    // Capture the actual DOM field so layout cleanup also clears a detached node.
    useLayoutEffect(() => {
        const field = input.current
        return () => {
            if (field) field.value = ''
        }
    }, [state?.supported, nodeId, engineType])

    useEffect(() => {
        const owner = generation
        readState()
        return () => {
            owner.current++
            mutation.current = null
        }
    }, [readState])

    function cancel() {
        if (input.current) input.current.value = ''
        if (mutation.current) {
            mutation.current.cancelled = true
            setConfirmed(false)
        }
        setError('')
    }

    function submit(clear: boolean) {
        if (pending || mutation.current || !confirmed) return
        if (!clear && !input.current?.value) return
        const current = ++generation.current
        const operation = { cancelled: false }
        mutation.current = operation
        setPending(true)
        setError('')
        // Only the transport packet receives the key. Completion callbacks hold
        // nonsecret ownership metadata, never a draft or a retryable mutation.
        try {
            window.pairApi.engines
                .setCredential({
                    nodeId,
                    engineType,
                    credential: clear ? '' : (input.current?.value ?? '')
                })
                .then(result => {
                    if (generation.current === current && !operation.cancelled) {
                        setState(result)
                        setConfirmed(true)
                    }
                })
                .catch(() => {
                    if (generation.current === current && !operation.cancelled)
                        setError('Could not update credential')
                })
                .finally(() => {
                    if (generation.current !== current) return
                    mutation.current = null
                    if (input.current) input.current.value = ''
                    // Cancel cannot stop the physical write. Keep mutations
                    // disabled until it settles, then obtain fresh authority.
                    if (operation.cancelled) readState()
                    else setPending(false)
                })
        } catch {
            mutation.current = null
            setPending(false)
            setError('Could not update credential')
        } finally {
            if (input.current) input.current.value = ''
        }
    }

    if (state && !state.supported) return null
    return (
        <Flex direction="col" gap="1" className="no-drag-elements">
            <Text kind="body/regular/sm">
                {state && confirmed
                    ? state.configured
                        ? 'Key saved'
                        : 'No key saved'
                    : 'Credential store unavailable'}
            </Text>
            {state?.supported && (
                <>
                    <FormField slotLabel={`${name} API key`}>
                        <TextInput
                            ref={input}
                            type="password"
                            aria-label={`${name} API key`}
                            disabled={pending || !confirmed}
                        />
                    </FormField>
                    <Flex gap="1">
                        <Button
                            size="tiny"
                            disabled={pending || !confirmed}
                            onClick={() => submit(false)}
                        >
                            Save
                        </Button>
                        <Button
                            size="tiny"
                            disabled={pending || !confirmed}
                            onClick={() => submit(true)}
                        >
                            Clear
                        </Button>
                        <Button size="tiny" onClick={cancel}>
                            Cancel
                        </Button>
                    </Flex>
                    {error && (
                        <Text kind="body/regular/sm" role="alert">
                            {error}
                        </Text>
                    )}
                </>
            )}
            {!confirmed && (
                <Button size="tiny" disabled={pending} onClick={readState}>
                    Retry
                </Button>
            )}
        </Flex>
    )
}

const LOCK_COPY = {
    locked: 'Locked',
    unavailable: 'Lock status unavailable',
    peerLocked: 'Unlock this engine on the device running it.'
}

/**
 * The lock beside an engine's On/Off switch. Only the device running the engine
 * can change its lock, so a peer row reports the state and offers no command.
 * An absent readback is unknown, never unlocked.
 */
function EngineLockControl({
    name,
    locked,
    local,
    disabled,
    onChange
}: {
    name: string
    locked: boolean | undefined
    local: boolean
    disabled: boolean
    onChange: (locked: boolean) => void
}) {
    if (!local) {
        if (locked === undefined) {
            return (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {LOCK_COPY.unavailable}
                </Text>
            )
        }
        if (!locked) return null
        return (
            <Flex align="center" gap="1" wrap="wrap" title={LOCK_COPY.peerLocked}>
                <Lock style={{ fontSize: 14 }} />
                <Text kind="body/regular/sm">{LOCK_COPY.locked}</Text>
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {LOCK_COPY.peerLocked}
                </Text>
            </Flex>
        )
    }
    const label = `${locked ? 'Unlock' : 'Lock'} ${name}`
    return (
        <Flex align="center" gap="1" wrap="wrap">
            <Button
                kind="tertiary"
                size="tiny"
                className="shrink-0 px-1"
                disabled={disabled || locked === undefined}
                aria-label={label}
                title={label}
                onClick={() => onChange(!locked)}
            >
                {locked ? <Lock style={{ fontSize: 14 }} /> : <LockOpen style={{ fontSize: 14 }} />}
            </Button>
            {locked !== false && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {locked ? LOCK_COPY.locked : LOCK_COPY.unavailable}
                </Text>
            )}
        </Flex>
    )
}

export default function NodeEnginesInline({ nodeId }: { nodeId: string }) {
    const statusByNode = useEngineStatusStore(s => s.statusByNode)
    const descriptors = useEngineStatusStore(s => s.descriptors)
    const isRemote = useConnectionStore(s => s.selfId !== nodeId)
    const isDisconnected = useNodesStore(s => isRemote && s.nodes.get(nodeId)?.status === 'offline')
    // Re-render when a lifecycle command for this node begins/clears; the
    // per-engine pending state is read below via getState().
    const lifecyclePendingFingerprint = usePendingActionsStore(state => {
        const parts: string[] = []
        for (const [key, p] of state.pending) {
            if (p.nodeId === nodeId && key.endsWith(':lifecycle')) {
                parts.push(`${p.engineType}:${p.action}`)
            }
        }
        return parts.join('|')
    })
    void lifecyclePendingFingerprint
    const engineTypes = useMemo(
        () => engineTypesForNode(nodeId ?? '', statusByNode, descriptors),
        [nodeId, statusByNode, descriptors]
    )
    const selfEngines = useMemo(
        () => getEnginesForNode(nodeId ?? '', statusByNode, engineTypes),
        [nodeId, statusByNode, engineTypes]
    )

    const allBackends = useMemo(() => {
        return engineTypes.flatMap(type => {
            const backend = selfEngines.find(b => b.engineType === type)

            // Show every engine the node has actually reported — not just ones
            // with a proxy port. The broker fronts only Ollama with a proxy, so
            // gating on `proxyPort` hid LM Studio (managed by nvpair-engine-manager,
            // loopback-only, no proxy). Skip the padded `initializing`
            // placeholder so engines a node never reports (e.g. LM Studio on a
            // remote, undiscovered node) don't show a perpetual spinner.
            if (!backend || backend.processStatus === 'initializing') {
                return []
            }

            const descriptor = descriptors.get(type)
            return [
                {
                    type,
                    name: engineLabel(type, descriptor),
                    status: backend.processStatus,
                    locked: backend.locked,
                    installable: engineCapabilities(type, descriptor).hasInstall.length > 0
                }
            ]
        })
    }, [engineTypes, selfEngines, descriptors])

    const handleToggle = useCallback(
        (engineType: EngineType) => {
            window.pairApi.engines.toggle(engineType, nodeId)
        },
        [nodeId]
    )

    const handleLock = useCallback(
        (engineType: EngineType, locked: boolean) => {
            window.pairApi.engines.setLock(engineType, nodeId, locked)
        },
        [nodeId]
    )

    const handleInstall = useCallback(
        (engineType: EngineType) => {
            window.pairApi.engines.install(engineType, nodeId)
        },
        [nodeId]
    )

    return (
        <>
            {allBackends.map(b => {
                const pending = Boolean(
                    usePendingActionsStore.getState().getLifecyclePending(nodeId, b.type)
                )
                const isTransitioning =
                    pending ||
                    b.status === 'installing' ||
                    b.status === 'uninstalling' ||
                    b.status === 'starting' ||
                    b.status === 'stopping'
                // A known lock keeps a stopped engine Off. A running locked engine
                // (a lock whose stop failed) can still be stopped.
                const startBlocked = b.locked === true && b.status !== 'running'
                const switchLabel = startBlocked
                    ? `${b.name} is locked`
                    : `${b.status === 'running' ? 'Stop' : 'Start'} ${b.name}`

                if (b.status === 'not-installed' && !pending) {
                    // An engine PAIR cannot install offers no control here.
                    if (!b.installable) return null
                    return (
                        <Button
                            key={b.name}
                            kind="secondary"
                            size="tiny"
                            className="shrink-0 no-drag-elements px-2"
                            onClick={e => {
                                e.preventDefault()
                                e.stopPropagation()
                                handleInstall(b.type)
                            }}
                            disabled={isRemote && isDisconnected}
                            aria-label={`Install ${b.name}`}
                        >
                            <Download style={{ fontSize: 14 }} />
                            <Text kind="body/regular/sm" className="ml-1">
                                {b.name}
                            </Text>
                        </Button>
                    )
                }

                return (
                    <Flex
                        key={b.name}
                        direction="col"
                        gap="2"
                        wrap="wrap"
                        className="shrink-0 no-drag-elements pair-engines-inline"
                    >
                        <Flex align="center" gap="2" wrap="wrap">
                            <Flex
                                align="center"
                                justify="center"
                                className="h-4 w-7 shrink-0 -mr-1"
                            >
                                {isTransitioning ? (
                                    <span
                                        className="spinner-element-medium"
                                        role="status"
                                        aria-label=""
                                    />
                                ) : (
                                    <Switch
                                        size="small"
                                        checked={b.status === 'running'}
                                        onCheckedChange={() => handleToggle(b.type)}
                                        disabled={(isRemote && isDisconnected) || startBlocked}
                                        title={
                                            isRemote && isDisconnected
                                                ? 'Node is disconnected'
                                                : switchLabel
                                        }
                                        aria-label={
                                            isRemote && isDisconnected
                                                ? 'Node is disconnected'
                                                : switchLabel
                                        }
                                    />
                                )}
                            </Flex>
                            <Text
                                kind="body/regular/sm"
                                className="cursor-pointer"
                                onClick={() =>
                                    !isTransitioning && !startBlocked
                                        ? handleToggle(b.type)
                                        : undefined
                                }
                            >
                                {b.name}
                            </Text>
                            <EngineLockControl
                                name={b.name}
                                locked={b.locked}
                                local={!isRemote}
                                disabled={
                                    pending ||
                                    b.status === 'installing' ||
                                    b.status === 'uninstalling' ||
                                    (isRemote && isDisconnected)
                                }
                                onChange={locked => handleLock(b.type, locked)}
                            />
                        </Flex>
                        {!isRemote && (
                            <CredentialControl
                                key={`${nodeId}:${b.type}`}
                                nodeId={nodeId}
                                engineType={b.type}
                                name={b.name}
                            />
                        )}
                    </Flex>
                )
            })}
        </>
    )
}
