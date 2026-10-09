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
import type { NodeAvailability } from '@/shared/types/node-policy'
import getErrorString from '@/shared/utils/get-error-string'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'

const THIS_PC = 'This PC'

const AVAILABILITY_COLORS: Record<NodeAvailability, 'green' | 'yellow' | 'gray'> = {
    available: 'green',
    draining: 'yellow',
    paused: 'gray'
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
    const latestLoad = useRef(0)
    // The policy as last loaded or saved, to tell whether the editor has unsaved edits.
    const textRef = useRef('')
    const savedTextRef = useRef('')
    textRef.current = text
    // The id the broker uses for the selected node; this PC's is the self id.
    const targetId = nodeId ?? selfId

    const load = useCallback(async () => {
        latestLoad.current += 1
        const ticket = latestLoad.current
        setLoading(true)
        setLoadError(null)
        setSaveError(null)
        try {
            const document = await window.pairApi.nodePolicy.get(nodeId ?? undefined)
            if (ticket !== latestLoad.current) return
            savedTextRef.current = document.policy
            setText(document.policy)
            setAvailability(document.availability)
        } catch (err) {
            if (ticket !== latestLoad.current) return
            setLoadError(getErrorString(err))
        } finally {
            if (ticket === latestLoad.current) setLoading(false)
        }
    }, [nodeId])

    useEffect(() => {
        if (!connected) return
        setAvailability(null)
        savedTextRef.current = ''
        setText('')
        void load()
    }, [connected, load])

    // Live updates from the broker. The policy text is replaced only when the
    // user has no unsaved edits, so a push never overwrites typing.
    useEffect(() => {
        if (targetId === null) return
        const offAvailability = window.pairApi.nodePolicy.onAvailabilityChanged(change => {
            if (change.nodeId === targetId) setAvailability(change.availability)
        })
        const offPolicy = window.pairApi.nodePolicy.onPolicyChanged(change => {
            if (change.nodeId !== targetId) return
            if (textRef.current !== savedTextRef.current) return
            savedTextRef.current = change.policy
            setText(change.policy)
        })
        return () => {
            offAvailability()
            offPolicy()
        }
    }, [targetId])

    const toggleAvailability = useCallback(async () => {
        if (availability !== 'available' && availability !== 'paused') return
        setSwitching(true)
        setLoadError(null)
        try {
            const result = await window.pairApi.nodePolicy.setAvailability(
                availability === 'available' ? 'paused' : 'available',
                nodeId ?? undefined
            )
            setAvailability(result.availability)
        } catch (err) {
            setLoadError(getErrorString(err))
        } finally {
            setSwitching(false)
        }
    }, [availability, nodeId])

    const save = useCallback(async () => {
        setSaving(true)
        setSaveError(null)
        try {
            const result = await window.pairApi.nodePolicy.set(text, nodeId ?? undefined)
            // Show the policy as persisted: the service fills in omitted fields.
            savedTextRef.current = result.policy
            setText(result.policy)
        } catch (err) {
            setSaveError(getErrorString(err))
        } finally {
            setSaving(false)
        }
    }, [text, nodeId])

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

    const canSwitch = (availability === 'available' || availability === 'paused') && !switching

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
                            onClick={() => void toggleAvailability()}
                        >
                            {availability === 'paused' ? 'Resume' : 'Pause'}
                        </Button>
                    </Flex>

                    <Stack gap="2">
                        <Text kind="body/semibold/sm">Policy</Text>
                        <TextArea
                            size="small"
                            aria-label="Node policy JSON"
                            value={text}
                            onValueChange={setText}
                            resizeable="auto"
                            rows={14}
                            maxLength={65536}
                            disabled={loading || saving}
                            className="font-mono"
                        />
                        {saveError && <InlineErrorBanner severity="error" message={saveError} />}
                        <Flex gap="2" justify="end">
                            <Button
                                kind="secondary"
                                size="small"
                                disabled={loading || saving}
                                onClick={() => void load()}
                            >
                                Reload
                            </Button>
                            <Button
                                kind="primary"
                                color="brand"
                                size="small"
                                disabled={loading || saving || text.trim() === ''}
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
        </Stack>
    )
}
