// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from 'react'
import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core'
import type { UpdateStatus } from '@/shared/types/update'
import type { BuildApplyOperation, BuildApplyPreview as Preview } from '@/shared/types/build-apply'
import { isElectron } from '@/ui/api/bootstrap'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import BuildApplyPreview from './BuildApplyPreview'
import BuildApplyResult from './BuildApplyResult'

function phaseLabel(status: UpdateStatus): string {
    switch (status.phase) {
        case 'idle':
            return 'Ready to check'
        case 'checking':
            return 'Checking for updates…'
        case 'available':
            return `Update available: ${status.latestVersion ?? ''}`
        case 'not-available':
            return 'You are on the latest version'
        case 'downloading':
            return `Downloading… ${status.downloadPercent === null ? '' : `${Math.round(status.downloadPercent)}%`}`
        case 'downloaded':
            return 'Downloaded. Application requires a verified-package adapter to the build lifecycle.'
        case 'error':
            return status.error ?? 'Update failed'
    }
}

export default function ApplicationUpdatesCard() {
    const [status, setStatus] = useState<UpdateStatus | null>(null)
    const [message, setMessage] = useState<string | null>(null)
    const [preview, setPreview] = useState<Preview | null>(null)
    const [observedAt, setObservedAt] = useState('')
    const [open, setOpen] = useState(false)
    const [operation, setOperation] = useState<BuildApplyOperation | null>(null)
    const [busy, setBusy] = useState(false)
    const generation = useRef(0)
    const snapshotSequence = useRef(0)
    const operationId = useRef('')
    const inFlight = useRef<Promise<void> | null>(null)
    const choosing = useRef(false)
    const chooseButton = useRef<HTMLButtonElement>(null)
    const selfId = useConnectionStore(state => state.selfId)
    const deviceName =
        useNodesStore(state => (selfId ? state.nodes.get(selfId)?.name : undefined)) ??
        'this device'

    const readStatus = useCallback((): Promise<void> => {
        if (!isElectron) return Promise.resolve()
        if (inFlight.current) return inFlight.current
        const current = generation.current
        const sequence = ++snapshotSequence.current
        const id = operationId.current
        const read = async (): Promise<void> => {
            try {
                const result = await window.windowApi.buildApply.status({ operationId: id })
                if (
                    current !== generation.current ||
                    sequence !== snapshotSequence.current ||
                    id !== operationId.current
                )
                    return
                if (result.available) {
                    operationId.current = result.value.operationId
                    setOperation(result.value)
                    setMessage(null)
                } else setMessage(result.refusal.detail)
            } catch {
                if (current === generation.current && sequence === snapshotSequence.current) {
                    setMessage(
                        'Build status is unavailable. Reconnect and read status; do not retry application.'
                    )
                }
            }
        }
        const promise = read().finally(() => {
            if (inFlight.current === promise) inFlight.current = null
        })
        inFlight.current = promise
        return promise
    }, [])

    useEffect(() => {
        if (!isElectron) return
        let active = true
        const retireReads = (): void => {
            generation.current++
            snapshotSequence.current++
            inFlight.current = null
        }
        let updatePush = false
        window.windowApi.update
            .getStatus()
            .then(value => {
                if (active && !updatePush) setStatus(value)
            })
            .catch(() => {})
        const unUpdate = window.windowApi.update.onStatusChanged(value => {
            updatePush = true
            setStatus(value)
        })
        const unOperation = window.windowApi.buildApply.onChanged(value => {
            if (operationId.current && operationId.current !== value.operationId) {
                void readStatus()
                return
            }
            snapshotSequence.current++
            operationId.current = value.operationId
            setOperation(value)
            setPreview(null)
            setOpen(false)
        })
        let connection: string | null = null
        const unService = window.windowApi.service.onStatusChanged(value => {
            if (value.connectorStatus === connection) return
            connection = value.connectorStatus
            retireReads()
            choosing.current = false
            setBusy(false)
            setPreview(null)
            setOpen(false)
            void readStatus()
        })
        void readStatus()
        return () => {
            active = false
            retireReads()
            unUpdate()
            unOperation()
            unService()
        }
    }, [readStatus])

    const chooseBuild = async (): Promise<void> => {
        if (!isElectron || choosing.current || busy) return
        const current = generation.current
        const sequence = ++snapshotSequence.current
        choosing.current = true
        setBusy(true)
        setPreview(null)
        setMessage(null)
        try {
            const result = await window.windowApi.buildApply.choosePreview()
            if (current !== generation.current || sequence !== snapshotSequence.current) return
            if (!result.available) setMessage(result.refusal.detail)
            else {
                setPreview(result.value)
                setObservedAt(new Date().toISOString())
                setOpen(true)
            }
        } catch {
            if (current === generation.current)
                setMessage(
                    'Could not preview the staged build. Reconnect and choose the staged build again.'
                )
        } finally {
            if (current === generation.current) {
                choosing.current = false
                setBusy(false)
            }
        }
    }

    const apply = async (allowEngineStop: boolean): Promise<void> => {
        if (!preview || busy) return
        const current = generation.current
        const sequence = ++snapshotSequence.current
        const previewId = preview.previewId
        operationId.current = ''
        setPreview(null)
        setOpen(false)
        setBusy(true)
        try {
            const result = await window.windowApi.buildApply.start({
                previewId,
                consent: { allowInterruption: true, allowEngineStop }
            })
            if (current !== generation.current || sequence !== snapshotSequence.current) return
            if (!result.available) setMessage(result.refusal.detail)
            else {
                operationId.current = result.value.operationId
                setOperation(result.value)
                setMessage(null)
            }
        } catch {
            if (current === generation.current) {
                setMessage(
                    'Application response disconnected. Read the existing operation before any further action.'
                )
                await readStatus()
            }
        } finally {
            if (current === generation.current) setBusy(false)
        }
    }

    const distribution = (action: 'check' | 'download'): void => {
        void window.windowApi.update[action]().catch(() =>
            setMessage('Distribution update request could not be read.')
        )
    }
    return (
        <div className="settings-card settings-card-compact pair-paper p-4 min-w-0 wrap-anywhere">
            <Stack gap="4">
                <Stack gap="1">
                    <Text kind="body/semibold/md">Application updates</Text>
                    <Text kind="body/regular/sm">
                        Version {status?.currentVersion ?? 'Unknown'}.{' '}
                        {status ? phaseLabel(status) : ''}
                    </Text>
                    <Text kind="body/regular/sm">
                        Apply a staged build on {deviceName}. The service owns interruption,
                        replacement and recovery.
                    </Text>
                </Stack>
                {status?.phase === 'error' && status.error && (
                    <InlineErrorBanner message={status.error} />
                )}
                {message && (
                    <div id="build-apply-message">
                        <InlineErrorBanner message={message} />
                    </div>
                )}
                <Flex gap="2" wrap="wrap">
                    <Button
                        ref={chooseButton}
                        kind="secondary"
                        size="small"
                        disabled={!isElectron || busy || operation?.outcome === null}
                        aria-describedby={message ? 'build-apply-message' : undefined}
                        onClick={() => {
                            void chooseBuild()
                        }}
                    >
                        Choose staged build
                    </Button>
                    {preview && (
                        <Button kind="secondary" size="small" onClick={() => setOpen(true)}>
                            Preview build
                        </Button>
                    )}
                    <Button
                        kind="secondary"
                        size="small"
                        disabled={!isElectron || busy}
                        onClick={() => {
                            void readStatus()
                        }}
                    >
                        Read build status
                    </Button>
                    <Button
                        kind="secondary"
                        size="small"
                        disabled={!isElectron}
                        onClick={() => distribution('check')}
                    >
                        Check for updates
                    </Button>
                    {status?.phase === 'available' && (
                        <Button
                            kind="secondary"
                            size="small"
                            onClick={() => distribution('download')}
                        >
                            Download update
                        </Button>
                    )}
                </Flex>
                {preview && (
                    <BuildApplyPreview
                        preview={preview}
                        deviceName={deviceName}
                        observedAt={observedAt}
                        open={open}
                        busy={busy}
                        onOpenChange={value => {
                            setOpen(value)
                            if (!value) chooseButton.current?.focus()
                        }}
                        onApply={value => {
                            void apply(value)
                        }}
                    />
                )}
                {operation && (
                    <BuildApplyResult
                        operation={operation}
                        busy={busy}
                        onRead={() => {
                            void readStatus()
                        }}
                    />
                )}
            </Stack>
        </div>
    )
}
