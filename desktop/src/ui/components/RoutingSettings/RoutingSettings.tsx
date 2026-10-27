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
    PolicyActivation
} from '@/shared/types/node-policy'
import getErrorString from '@/shared/utils/get-error-string'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import BulkNodeConfiguration from '@/ui/components/RoutingSettings/BulkNodeConfiguration'
import RoleEditor from '@/ui/components/RoutingSettings/RoleEditor'
import ModelEngineBindingsEditor from './ModelEngineBindingsEditor'
import ModelSettingsPanel from '@/ui/components/ModelSettings/ModelSettingsPanel'
import RoutingHistory from '@/ui/components/RoutingSettings/RoutingHistory'
import NodeAvailabilityControls from '@/ui/components/NodeList/NodeAvailabilityControls'
import { useConnectionStore } from '@/ui/stores/connection.store'
import {
    useNodeConfigurationStore,
    readNodePolicy,
    watchNodeManagement
} from '@/ui/stores/node-configuration.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { REMOTE_SETTING_LOCKED_CODE } from '@/shared/constants/service-errors'

const THIS_PC = 'This PC'

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

    const [text, setTextState] = useState('')
    const [settingsSnapshot, setSettingsSnapshot] = useState<NodePolicySnapshot | null>(null)
    const [settingsValid, setSettingsValid] = useState(true)
    const [loading, setLoading] = useState(false)
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
    // The policy as last loaded or saved, to tell whether the editor has unsaved edits.
    const textRef = useRef('')
    const savedTextRef = useRef('')
    // The persisted revision the draft was edited from; empty until a policy loads.
    const baseRevisionRef = useRef('')
    textRef.current = text
    // The id the broker uses for the selected node; this PC's is the self id.
    const targetId = nodeId ?? selfId
    const availabilityEntry = useNodeConfigurationStore(state =>
        targetId === null ? undefined : state.availability.get(targetId)
    )
    const availability = availabilityEntry?.value ?? null
    const switching = availabilityEntry?.pending !== undefined
    const policyDraft = useNodeConfigurationStore(state =>
        targetId === null ? undefined : state.policyDrafts.get(targetId)
    )
    const otherDraft = policyDraft !== undefined && policyDraft.owner !== 'Routing'
    const setText = useCallback(
        (value: string) => {
            if (targetId === null) return
            const store = useNodeConfigurationStore.getState()
            if (!store.claimPolicyDraft(targetId, 'Routing')) return
            store.editPolicyDraft(targetId, 'Routing', value)
            textRef.current = value
            setTextState(value)
        },
        [targetId]
    )
    const management = useNodeConfigurationStore(state =>
        targetId === null ? undefined : state.management.get(targetId)
    )
    const peerPaused = nodeId !== null && (availability === 'paused' || availability === 'draining')
    const targetOnline =
        nodeId === null || peers.find(peer => peer.id === nodeId)?.status !== 'offline'
    const permissionReadOnly =
        !connected ||
        !targetOnline ||
        (nodeId !== null && (management?.state !== 'allowed' || !availabilityEntry?.current)) ||
        peerPaused
    const previousTarget = useRef<string | null>(null)

    useEffect(() => {
        if (targetId === null) return
        return watchNodeManagement(targetId, nodeId ?? undefined)
    }, [targetId, nodeId])

    // The role editor substitutes into the newest document the node reported,
    // whatever the policy text editor holds.
    const adoptRoles = useCallback(
        (saved: NodePolicySnapshot & { availability?: NodeAvailability }) => {
            if (targetId === null) return
            useNodeConfigurationStore.getState().adopt(targetId, {
                ...saved,
                availability:
                    useNodeConfigurationStore.getState().availability.get(targetId)?.value ??
                    saved.availability
            })
        },
        [targetId]
    )

    const load = useCallback(
        async (preserveDraft = false) => {
            if (targetId === null) return
            const store = useNodeConfigurationStore.getState()
            if (!store.claimPolicyDraft(targetId, 'Routing', !preserveDraft)) return
            const draftGeneration = useNodeConfigurationStore
                .getState()
                .policyDrafts.get(targetId)?.generation
            latestLoad.current += 1
            const ticket = latestLoad.current
            setLoading(true)
            setLoadError(null)
            setSaveError(null)
            setConflict(null)
            setReviewing(false)
            try {
                const document = await readNodePolicy(nodeId ?? undefined)
                if (ticket !== latestLoad.current) return
                const entry = useNodeConfigurationStore.getState().policyDrafts.get(targetId)
                if (entry?.owner !== 'Routing' || entry.generation !== draftGeneration) return
                const newest = useNodeConfigurationStore
                    .getState()
                    .policyDrafts.get(targetId)?.saved
                if (newest && isOlder(document.revision, newest.revision)) {
                    return
                }
                store.observePolicyDraft(
                    targetId,
                    {
                        ...document,
                        availability:
                            useNodeConfigurationStore.getState().availability.get(targetId)
                                ?.value ?? document.availability
                    },
                    'read'
                )
                adoptRoles(document)
                setSettingsSnapshot(current =>
                    current && isOlder(document.revision, current.revision) ? current : document
                )
                if (preserveDraft && textRef.current !== savedTextRef.current) {
                    if (document.revision !== baseRevisionRef.current)
                        setConflict({
                            latest: document,
                            message: 'The saved policy changed while this draft was unsent.'
                        })
                } else {
                    savedTextRef.current = document.policy
                    baseRevisionRef.current = document.revision
                    if (entry && entry.text !== (entry.baseline?.policy ?? '') && preserveDraft) {
                        savedTextRef.current = entry?.baseline?.policy ?? ''
                        baseRevisionRef.current = entry?.baseline?.revision ?? ''
                        textRef.current = entry?.text ?? ''
                        setTextState(entry?.text ?? '')
                    } else setText(document.policy)
                }
                setStatus({ revision: document.revision, activation: document.activation })
                setRecoveryRequired(document.recoveryRequired)
            } catch (err) {
                if (ticket !== latestLoad.current) return
                setLoadError(getErrorString(err))
            } finally {
                if (ticket === latestLoad.current) setLoading(false)
            }
        },
        [nodeId, adoptRoles, targetId, setText]
    )

    // A saved policy reported by the node. Changes can arrive out of order, so
    // the status only moves forward. The text and base revision follow only when
    // the user has no unsaved edits and the report is not older than the base:
    // a push never overwrites typing or rewinds the editor.
    const noteStatus = useCallback((saved: NodePolicySnapshot) => {
        setSettingsSnapshot(current =>
            current && isOlder(saved.revision, current.revision) ? current : saved
        )
        setStatus(current =>
            current && isOlder(saved.revision, current.revision)
                ? current
                : { revision: saved.revision, activation: saved.activation }
        )
    }, [])

    const adopt = useCallback(
        (saved: NodePolicySnapshot & { availability?: NodeAvailability }) => {
            noteStatus(saved)
            adoptRoles(saved)
            if (targetId !== null)
                useNodeConfigurationStore.getState().observePolicyDraft(targetId, saved)
            if (textRef.current !== savedTextRef.current) return
            if (isOlder(saved.revision, baseRevisionRef.current)) return
            baseRevisionRef.current = saved.revision
            savedTextRef.current = saved.policy
            setText(saved.policy)
        },
        [noteStatus, adoptRoles, targetId, setText]
    )

    // Availability readback can discover a new saved policy too. Keep the
    // editor's base current without overwriting a dirty draft.
    useEffect(() => {
        const saved = policyDraft?.saved
        if (!saved || policyDraft.owner !== 'Routing') return
        noteStatus(saved)
        if (
            textRef.current !== savedTextRef.current ||
            isOlder(saved.revision, baseRevisionRef.current)
        )
            return
        baseRevisionRef.current = saved.revision
        savedTextRef.current = saved.policy
        textRef.current = saved.policy
        setTextState(saved.policy)
    }, [policyDraft?.saved, policyDraft?.owner, noteStatus])

    // Live updates from the broker.
    useEffect(() => {
        if (targetId === null) return
        const offPolicy = window.pairApi.nodePolicy.onPolicyChanged(change => {
            if (change.nodeId === targetId) adopt(change)
        })
        return () => {
            offPolicy()
        }
    }, [targetId, adopt])

    // The listeners above are installed before a target read can begin.
    useEffect(() => {
        setSaving(false)
        if (!connected) setLoading(false)
        if (previousTarget.current !== targetId) {
            previousTarget.current = targetId
            setStatus(null)
            setSettingsSnapshot(null)
            setSettingsValid(true)
            setRecoveryRequired(false)
            const cached =
                targetId === null
                    ? undefined
                    : useNodeConfigurationStore.getState().policyDrafts.get(targetId)
            const mine = cached?.owner === 'Routing' ? cached : undefined
            savedTextRef.current = mine?.baseline?.policy ?? ''
            baseRevisionRef.current = mine?.baseline?.revision ?? ''
            setTextState(mine?.text ?? '')
            textRef.current = mine?.text ?? ''
        }
        if (connected) void load(true)
        return () => {
            latestLoad.current++
            if (
                targetId !== null &&
                useNodeConfigurationStore.getState().policyDrafts.get(targetId)?.owner === 'Routing'
            )
                useNodeConfigurationStore.getState().invalidatePolicyDraft(targetId)
        }
    }, [connected, targetId, load])

    const save = useCallback(async () => {
        if (
            permissionReadOnly ||
            loading ||
            saving ||
            !settingsValid ||
            switching ||
            conflict !== null ||
            otherDraft ||
            targetId === null ||
            previousTarget.current !== targetId ||
            baseRevisionRef.current === ''
        )
            return
        const ticket = latestLoad.current
        setSaving(true)
        setSaveError(null)
        await useNodeConfigurationStore
            .getState()
            .savePolicyDraft(targetId, 'Routing', nodeId ?? undefined)
        if (ticket !== latestLoad.current) return
        const result = useNodeConfigurationStore.getState().policyDrafts.get(targetId)
        if (result?.phase === 'saved' && result.baseline) {
            adoptRoles(result.baseline)
            savedTextRef.current = result.baseline.policy
            baseRevisionRef.current = result.baseline.revision
            setTextState(result.text)
            textRef.current = result.text
            noteStatus(result.baseline)
            setRecoveryRequired(false)
        } else if (result?.phase === 'conflict' && result.saved) {
            setConflict({ latest: result.saved, message: result.message ?? 'Policy conflict' })
        } else {
            if (result?.failure?.code === REMOTE_SETTING_LOCKED_CODE) {
                const detail = result.failure.detail.slice(0, 1024).replace(/[\p{Cc}\p{Cf}]/gu, ' ')
                setSaveError(
                    `Inference settings are locked on ${nodeName}. Your policy draft is kept. Change this profile's inference settings on that device. Service message: ${detail}`
                )
            } else setSaveError(result?.message ?? 'Could not save this policy.')
        }
        setSaving(false)
    }, [
        nodeId,
        noteStatus,
        adoptRoles,
        permissionReadOnly,
        targetId,
        loading,
        saving,
        switching,
        conflict,
        settingsValid,
        otherDraft,
        nodeName
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

    // Save needs the revision the draft is based on, which a loaded policy supplies.
    const canSave =
        !permissionReadOnly &&
        !otherDraft &&
        policyDraft?.current === true &&
        policyDraft?.phase !== 'conflict' &&
        policyDraft?.phase !== 'unknown' &&
        policyDraft?.phase !== 'saving' &&
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

                    {otherDraft && (
                        <InlineErrorBanner
                            severity="warning"
                            message="Model Manager owns an unsaved policy draft for this node. Discard it and reload explicitly to edit in Routing."
                        >
                            <Button size="small" onClick={() => void load()}>
                                Discard Model Manager draft and reload
                            </Button>
                        </InlineErrorBanner>
                    )}

                    {nodeId !== null && permissionReadOnly && (
                        <Text kind="body/regular/sm" role="status">
                            {management?.message ||
                                'Reading remote management permission for this device.'}
                        </Text>
                    )}

                    {targetId !== null && (
                        <NodeAvailabilityControls
                            targetId={targetId}
                            nodeId={nodeId ?? undefined}
                            name={nodeName}
                            online={targetOnline}
                        />
                    )}
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

                    {targetId !== null && (
                        <ModelSettingsPanel
                            key={targetId}
                            targetId={targetId}
                            nodeId={nodeId ?? undefined}
                            nodeName={nodeName}
                            routing={{
                                text,
                                onTextChange: setText,
                                savedText: settingsSnapshot?.policy ?? null,
                                disabled:
                                    loading ||
                                    saving ||
                                    switching ||
                                    permissionReadOnly ||
                                    !canSave,
                                onValidityChange: setSettingsValid,
                                onSave: save,
                                canSave: canSave && settingsValid && !loading && !saving,
                                message: saveError ?? policyDraft?.message,
                                onReload: () => void load()
                            }}
                        />
                    )}

                    <Stack gap="2">
                        <Text kind="body/semibold/sm">Advanced policy JSON</Text>
                        <Text kind="body/regular/sm">
                            Pause unloads models and stops controlled engines. An incomplete stop
                            retains the fence and reports an error. The stopEngines and unloadModels
                            fields are ignored.
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
                                    disabled={loading || saving || permissionReadOnly || otherDraft}
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
                                {text !== savedTextRef.current
                                    ? 'Discard draft and reload'
                                    : 'Reload'}
                            </Button>
                            <Button
                                kind="primary"
                                color="brand"
                                size="small"
                                disabled={loading || saving || !canSave || !settingsValid}
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
