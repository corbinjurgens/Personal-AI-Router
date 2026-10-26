// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
    Badge,
    Button,
    Dropdown,
    Flex,
    Stack,
    Text,
    TextArea,
    type DropdownEntry
} from '@nvidia/foundations-react-core'
import { useShallow } from 'zustand/react/shallow'
import type {
    NodeAvailability,
    NodePolicySnapshot,
    PolicyActivation,
    RequestedAvailability
} from '@/shared/types/node-policy'
import getErrorString from '@/shared/utils/get-error-string'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import BulkNodeConfiguration from '@/ui/components/RoutingSettings/BulkNodeConfiguration'
import RoleEditor from '@/ui/components/RoutingSettings/RoleEditor'
import ModelEngineBindingsEditor from './ModelEngineBindingsEditor'
import RoutingHistory from '@/ui/components/RoutingSettings/RoutingHistory'
import { useConnectionStore } from '@/ui/stores/connection.store'
import {
    useNodeConfigurationStore,
    watchNodeManagement
} from '@/ui/stores/node-configuration.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import {
    LOCAL_VETO_CODE,
    REMOTE_MANAGEMENT_DENIED_CODE,
    localVetoMessage
} from '@/shared/constants/service-errors'
import { serviceErrorCode } from '@/ui/api/service-transport'

const THIS_PC = 'This PC'

const AVAILABILITY_COLORS: Record<NodeAvailability, 'green' | 'yellow' | 'gray'> = {
    available: 'green',
    draining: 'yellow',
    paused: 'gray'
}

const ACTIVATION_COLORS: Record<PolicyActivation, 'green' | 'yellow' | 'red'> = {
    active: 'green',
    pending: 'yellow',
    error: 'red'
}

const ACTIVATION_LABELS: Record<PolicyActivation, string> = {
    active: 'Active',
    pending: 'Saved, not yet enforced',
    error: 'Saved, not enforced'
}

/** The newest saved revision the node has reported on, and whether its proxy enforces it. */
type PolicyStatus = Pick<NodePolicySnapshot, 'revision' | 'activation'>

/**
 * Whether `revision` is older than `than`. The bridge only passes canonical
 * decimal strings; they are compared as integers because they exceed what a
 * number holds exactly. An empty `than` is not yet known, so nothing is older.
 */
function isOlder(revision: string, than: string): boolean {
    return than !== '' && BigInt(revision) < BigInt(than)
}

/** Policy and pause switch for one node: this PC, or a paired node reached through the broker. */
export default function RoutingSettings() {
    const connected = useConnectionStore(state => state.connected)
    const selfId = useConnectionStore(state => state.selfId)
    const peers = useNodesStore(
        useShallow(state => Array.from(state.nodes.values()).filter(node => node.id !== selfId))
    )

    // null is this PC: the service omits `nodeId` for it.
    const [pickedNodeId, setPickedNodeId] = useState<string | null>(null)
    const nodeId = peers.some(peer => peer.id === pickedNodeId) ? pickedNodeId : null
    const nodeName = peers.find(peer => peer.id === nodeId)?.name ?? THIS_PC

    const [availability, setAvailability] = useState<NodeAvailability | null>(null)
    const [text, setText] = useState('')
    const [loading, setLoading] = useState(false)
    const [switching, setSwitching] = useState(false)
    const [saving, setSaving] = useState(false)
    const [loadError, setLoadError] = useState<string | null>(null)
    const [saveError, setSaveError] = useState<string | null>(null)
    const [status, setStatus] = useState<PolicyStatus | null>(null)
    const [recoveryRequired, setRecoveryRequired] = useState(false)
    // The saved policy found newer than the draft's base when a save failed,
    // with the service's message for that save.
    const [conflict, setConflict] = useState<{
        latest: NodePolicySnapshot
        message: string
    } | null>(null)
    const [reviewing, setReviewing] = useState(false)
    const latestLoad = useRef(0)
    const availabilityGeneration = useRef(0)
    const latestAvailability = useRef<NodeAvailability | null>(null)
    // The policy as last loaded or saved, to tell whether the editor has unsaved edits.
    const textRef = useRef('')
    const savedTextRef = useRef('')
    // The persisted revision the draft was edited from; empty until a policy loads.
    const baseRevisionRef = useRef('')
    textRef.current = text
    // The id the broker uses for the selected node; this PC's is the self id.
    const targetId = nodeId ?? selfId
    const management = useNodeConfigurationStore(state =>
        targetId === null ? undefined : state.management.get(targetId)
    )
    const peerPaused = nodeId !== null && (availability === 'paused' || availability === 'draining')
    const permissionReadOnly =
        !connected || (nodeId !== null && management?.state !== 'allowed') || peerPaused
    const previousTarget = useRef<string | null>(null)

    useEffect(() => {
        if (targetId === null) return
        return watchNodeManagement(targetId, nodeId ?? undefined)
    }, [targetId, nodeId])

    // The role editor substitutes into the newest document the node reported,
    // whatever the policy text editor holds.
    const adoptRoles = useCallback(
        (
            saved: NodePolicySnapshot & { availability?: NodeAvailability },
            availabilityRead?: number
        ) => {
            if (targetId === null) return
            useNodeConfigurationStore.getState().adopt(targetId, {
                ...saved,
                availability:
                    availabilityRead !== undefined &&
                    availabilityRead !== availabilityGeneration.current
                        ? (latestAvailability.current ?? undefined)
                        : (saved.availability ?? latestAvailability.current ?? undefined)
            })
        },
        [targetId]
    )

    const load = useCallback(
        async (preserveDraft = false) => {
            latestLoad.current += 1
            const ticket = latestLoad.current
            setLoading(true)
            setLoadError(null)
            setSaveError(null)
            setConflict(null)
            setReviewing(false)
            try {
                const availabilityRead = availabilityGeneration.current
                const document = await window.pairApi.nodePolicy.get(nodeId ?? undefined)
                if (ticket !== latestLoad.current) return
                adoptRoles(document, availabilityRead)
                if (preserveDraft && textRef.current !== savedTextRef.current) {
                    if (document.revision !== baseRevisionRef.current)
                        setConflict({
                            latest: document,
                            message: 'The saved policy changed while this draft was unsent.'
                        })
                } else {
                    savedTextRef.current = document.policy
                    baseRevisionRef.current = document.revision
                    setText(document.policy)
                }
                setStatus({ revision: document.revision, activation: document.activation })
                setRecoveryRequired(document.recoveryRequired)
                if (availabilityRead === availabilityGeneration.current) {
                    latestAvailability.current = document.availability
                    setAvailability(document.availability)
                }
            } catch (err) {
                if (ticket !== latestLoad.current) return
                setLoadError(getErrorString(err))
            } finally {
                if (ticket === latestLoad.current) setLoading(false)
            }
        },
        [nodeId, adoptRoles]
    )

    useEffect(() => {
        setSaving(false)
        setSwitching(false)
        if (!connected) setLoading(false)
        if (previousTarget.current !== targetId) {
            previousTarget.current = targetId
            latestAvailability.current = null
            setAvailability(null)
            savedTextRef.current = ''
            baseRevisionRef.current = ''
            setStatus(null)
            setRecoveryRequired(false)
            setText('')
            textRef.current = ''
        }
        if (connected) void load(true)
        return () => {
            latestLoad.current++
        }
    }, [connected, targetId, load])

    // A saved policy reported by the node. Changes can arrive out of order, so
    // the status only moves forward. The text and base revision follow only when
    // the user has no unsaved edits and the report is not older than the base:
    // a push never overwrites typing or rewinds the editor.
    const noteStatus = useCallback((saved: NodePolicySnapshot) => {
        setStatus(current =>
            current && isOlder(saved.revision, current.revision)
                ? current
                : { revision: saved.revision, activation: saved.activation }
        )
    }, [])

    const adopt = useCallback(
        (
            saved: NodePolicySnapshot & { availability?: NodeAvailability },
            availabilityRead?: number
        ) => {
            noteStatus(saved)
            adoptRoles(saved, availabilityRead)
            if (textRef.current !== savedTextRef.current) return
            if (isOlder(saved.revision, baseRevisionRef.current)) return
            baseRevisionRef.current = saved.revision
            savedTextRef.current = saved.policy
            setText(saved.policy)
        },
        [noteStatus, adoptRoles]
    )

    // Live updates from the broker.
    useEffect(() => {
        if (targetId === null) return
        const offAvailability = window.pairApi.nodePolicy.onAvailabilityChanged(change => {
            if (change.nodeId === targetId) {
                availabilityGeneration.current++
                latestAvailability.current = change.availability
                setAvailability(change.availability)
                const store = useNodeConfigurationStore.getState()
                const baseline = store.editors.get(targetId)?.baseline
                if (baseline)
                    store.adopt(targetId, {
                        policy: baseline.policy,
                        revision: baseline.revision,
                        availability: change.availability
                    })
            }
        })
        const offPolicy = window.pairApi.nodePolicy.onPolicyChanged(change => {
            if (change.nodeId === targetId) adopt(change)
        })
        return () => {
            offAvailability()
            offPolicy()
        }
    }, [targetId, adopt])

    const changeAvailability = useCallback(
        async (requested: RequestedAvailability) => {
            if (permissionReadOnly || availability === null || switching || loading) return
            const ticket = latestLoad.current
            setSwitching(true)
            setLoadError(null)
            let operationError: string | null = null
            try {
                await window.pairApi.nodePolicy.setAvailability(requested, nodeId ?? undefined)
            } catch (err) {
                const code = serviceErrorCode(err)
                operationError =
                    code === REMOTE_MANAGEMENT_DENIED_CODE
                        ? `This device refused remote management permission. Change the permission on that device before trying again. ${getErrorString(err)}`
                        : code === LOCAL_VETO_CODE
                          ? `${localVetoMessage('change availability')} ${getErrorString(err)}`
                          : getErrorString(err)
                if (
                    ticket === latestLoad.current &&
                    code === REMOTE_MANAGEMENT_DENIED_CODE &&
                    targetId !== null
                )
                    void useNodeConfigurationStore
                        .getState()
                        .readManagement(targetId, nodeId ?? undefined)
            } finally {
                // Even an unsuccessful physical pause can have installed the fence.
                // Read the node after either outcome; the command reply is not a
                // promise that its availability still applies.
                const availabilityRead = availabilityGeneration.current
                try {
                    const document = await window.pairApi.nodePolicy.get(nodeId ?? undefined)
                    if (ticket === latestLoad.current) {
                        adopt(document, availabilityRead)
                        setRecoveryRequired(document.recoveryRequired)
                        if (availabilityRead === availabilityGeneration.current) {
                            latestAvailability.current = document.availability
                            setAvailability(document.availability)
                        }
                    }
                } catch (err) {
                    const readError = `Could not refresh node policy: ${getErrorString(err)}`
                    operationError = operationError ? `${operationError} ${readError}` : readError
                }
                if (ticket === latestLoad.current) {
                    setLoadError(operationError)
                    setSwitching(false)
                }
            }
        },
        [availability, nodeId, adopt, permissionReadOnly, targetId, switching, loading]
    )

    const save = useCallback(async () => {
        if (
            permissionReadOnly ||
            loading ||
            saving ||
            switching ||
            conflict !== null ||
            previousTarget.current !== targetId ||
            baseRevisionRef.current === ''
        )
            return
        const ticket = latestLoad.current
        const base = baseRevisionRef.current
        setSaving(true)
        setSaveError(null)
        try {
            const result = await window.pairApi.nodePolicy.set({
                nodeId: nodeId ?? undefined,
                policy: text,
                expectedPersistedRevision: base
            })
            if (ticket !== latestLoad.current) return
            adoptRoles(result)
            // Show the policy as persisted: the service fills in omitted fields.
            savedTextRef.current = result.policy
            baseRevisionRef.current = result.revision
            setText(result.policy)
            // A change for a newer revision may have been announced before this reply.
            noteStatus(result)
            setRecoveryRequired(false)
        } catch (err) {
            if (ticket !== latestLoad.current) return
            const code = serviceErrorCode(err)
            if (code === REMOTE_MANAGEMENT_DENIED_CODE || code === LOCAL_VETO_CODE) {
                setSaveError(
                    code === REMOTE_MANAGEMENT_DENIED_CODE
                        ? 'This device refused remote management permission. Your policy draft is kept. Change the permission on that device before trying again.'
                        : `${localVetoMessage('save this policy')} ${getErrorString(err)}`
                )
                if (code === REMOTE_MANAGEMENT_DENIED_CODE && targetId !== null)
                    void useNodeConfigurationStore
                        .getState()
                        .readManagement(targetId, nodeId ?? undefined)
                return
            }
            // Read the saved revision after errors: a durability warning means
            // the replacement landed, and a transport failure can leave the
            // outcome uncertain even though refusal codes survive the relay.
            const message = getErrorString(err)
            const latest = await window.pairApi.nodePolicy
                .get(nodeId ?? undefined)
                .catch(() => null)
            if (ticket !== latestLoad.current) return
            if (latest === null) {
                setSaveError(
                    `Could not confirm whether the save took effect: ${message} Reload to see the saved policy.`
                )
            } else if (latest.revision !== base) {
                noteStatus(latest)
                adoptRoles(latest)
                setConflict({ latest, message })
            } else {
                setSaveError(message)
            }
        } finally {
            if (ticket === latestLoad.current) setSaving(false)
        }
    }, [
        text,
        nodeId,
        noteStatus,
        adoptRoles,
        permissionReadOnly,
        targetId,
        loading,
        saving,
        switching,
        conflict
    ])

    const nodeItems: DropdownEntry[] = useMemo(
        () => [
            { children: THIS_PC, onSelect: () => setPickedNodeId(null) },
            ...peers.map(peer => ({
                children: peer.name,
                onSelect: () => setPickedNodeId(peer.id)
            }))
        ],
        [peers]
    )

    const canSwitch = !permissionReadOnly && availability !== null && !switching && !loading
    // Save needs the revision the draft is based on, which a loaded policy supplies.
    const canSave =
        !permissionReadOnly &&
        !switching &&
        previousTarget.current === targetId &&
        baseRevisionRef.current !== '' &&
        availability !== null &&
        conflict === null &&
        text.trim() !== ''
    const activation = status?.activation ?? null

    return (
        <Stack gap="6" className="relative py-8 px-3 w-full">
            {loadError && <InlineErrorBanner severity="error" message={loadError} />}

            <div className="settings-card pair-paper p-4">
                <Stack gap="4">
                    <Text kind="body/semibold/md">Routing &amp; resources</Text>

                    <Flex align="center" gap="3" wrap="wrap">
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            Node
                        </Text>
                        <Dropdown items={nodeItems} size="small" aria-label="Select node">
                            {nodeName}
                        </Dropdown>
                    </Flex>

                    {nodeId !== null && permissionReadOnly && (
                        <Text kind="body/regular/sm" role="status">
                            {management?.message ||
                                'Reading remote management permission for this device.'}
                        </Text>
                    )}

                    <Flex align="center" gap="3" wrap="wrap">
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            Availability
                        </Text>
                        {availability && (
                            <Badge color={AVAILABILITY_COLORS[availability]} kind="solid">
                                {availability}
                            </Badge>
                        )}
                        <Button
                            kind="secondary"
                            size="small"
                            disabled={!canSwitch}
                            aria-busy={switching}
                            onClick={() =>
                                void changeAvailability(
                                    availability === 'available' ||
                                        (availability === 'draining' && nodeId === null)
                                        ? 'paused'
                                        : 'available'
                                )
                            }
                        >
                            {availability === 'draining' && nodeId === null
                                ? 'Retry pause'
                                : availability === 'paused' || availability === 'draining'
                                  ? 'Resume'
                                  : 'Pause'}
                        </Button>
                        {availability === 'draining' && nodeId === null && (
                            <Button
                                kind="secondary"
                                size="small"
                                disabled={!canSwitch}
                                aria-busy={switching}
                                onClick={() => void changeAvailability('available')}
                            >
                                Resume
                            </Button>
                        )}
                    </Flex>
                    {peerPaused && (
                        <Text kind="body/regular/sm" role="status">
                            This node is paused. Resume it on that node before making changes.
                        </Text>
                    )}

                    {targetId !== null && availability !== null && (
                        <RoleEditor
                            targetId={targetId}
                            nodeId={nodeId ?? undefined}
                            onSaved={adopt}
                        />
                    )}

                    {targetId !== null && (
                        <ModelEngineBindingsEditor
                            targetId={targetId}
                            text={text}
                            onTextChange={setText}
                            disabled={
                                loading || saving || switching || permissionReadOnly || !canSave
                            }
                        />
                    )}

                    <Stack gap="2">
                        <Text kind="body/semibold/sm">Policy</Text>
                        <Text kind="body/regular/sm">
                            Pausing always unloads models and stops engines. The stopEngines and
                            unloadModels fields are ignored.
                        </Text>
                        {recoveryRequired && (
                            <InlineErrorBanner
                                severity="warning"
                                message="The saved policy file was unreadable, so default settings are in use. Save a policy to replace them."
                            />
                        )}
                        {activation && (
                            <Flex align="center" gap="3" wrap="wrap">
                                <Text kind="body/regular/sm" className="text-subtle-color">
                                    Saved policy
                                </Text>
                                <Badge color={ACTIVATION_COLORS[activation.state]} kind="solid">
                                    {ACTIVATION_LABELS[activation.state]}
                                </Badge>
                            </Flex>
                        )}
                        {activation?.state === 'error' && (
                            <InlineErrorBanner
                                severity="error"
                                message={`The saved policy is not enforced yet${activation.error ? `: ${activation.error}` : '.'}`}
                            />
                        )}
                        <Flex gap="3" align="start">
                            <div className="min-w-0 flex-1">
                                <TextArea
                                    size="small"
                                    aria-label="Node policy JSON"
                                    value={text}
                                    onValueChange={setText}
                                    resizeable="auto"
                                    rows={14}
                                    maxLength={65536}
                                    disabled={loading || saving || permissionReadOnly}
                                    className="font-mono"
                                />
                            </div>
                            {conflict && reviewing && (
                                <div className="min-w-0 flex-1">
                                    <TextArea
                                        size="small"
                                        aria-label="Latest saved node policy JSON"
                                        value={conflict.latest.policy}
                                        readOnly
                                        resizeable="auto"
                                        rows={14}
                                        className="font-mono"
                                    />
                                </div>
                            )}
                        </Flex>
                        {saveError && <InlineErrorBanner severity="error" message={saveError} />}
                        {conflict && (
                            <InlineErrorBanner
                                severity="warning"
                                message={`The saved policy changed after you loaded it, so this draft was not applied to it. If the service's reply was lost, the save may have taken effect. Your edits stay in the editor, and Save stays off until you discard them and reload. Service message: ${conflict.message}`}
                            >
                                <Button
                                    kind="secondary"
                                    size="tiny"
                                    onClick={() => setReviewing(current => !current)}
                                >
                                    {reviewing
                                        ? 'Hide latest saved policy'
                                        : 'Review latest saved policy'}
                                </Button>
                                <Button kind="secondary" size="tiny" onClick={() => void load()}>
                                    Discard draft and reload
                                </Button>
                            </InlineErrorBanner>
                        )}
                        <Flex gap="2" justify="end">
                            <Button
                                kind="secondary"
                                size="small"
                                disabled={loading || saving || switching}
                                onClick={() => void load()}
                            >
                                Reload
                            </Button>
                            <Button
                                kind="primary"
                                color="brand"
                                size="small"
                                disabled={loading || saving || !canSave}
                                aria-busy={saving}
                                onClick={() => void save()}
                            >
                                Save
                            </Button>
                        </Flex>
                    </Stack>

                    <Text kind="body/regular/sm" className="text-subtle-color">
                        Policy fields are described in FORK_DESIGN.md, section 3 (Node policy).
                        Omitted fields take their defaults; availability changes only with the Pause
                        and Resume button.
                    </Text>
                </Stack>
            </div>

            <div className="settings-card pair-paper p-4">
                <BulkNodeConfiguration />
            </div>
            <RoutingHistory />
        </Stack>
    )
}
