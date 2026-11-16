// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useId, useState } from 'react'
import {
    Button,
    Flex,
    FormField,
    ModalContent,
    ModalDialog,
    ModalRoot,
    ProgressBar,
    RadioGroup,
    Stack,
    Text,
    TextInput
} from '@nvidia/foundations-react-core'
import { DialogHeader } from '@/ui/components/DialogHeader'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { SOURCE_MODELS_ENGINE, useSourceModelsStore } from '@/ui/stores/source-models.store'
import getErrorString from '@/shared/utils/get-error-string'

type Mode = 'copy' | 'link'

function fileName(path: string): string {
    return path.split(/[\\/]/).pop() ?? path
}

/** The model name a file suggests: its name without .gguf or a first-shard suffix. */
function fileStem(path: string): string {
    return fileName(path)
        .replace(/\.gguf$/i, '')
        .replace(/-00001-of-\d{5}$/i, '')
}

function ModeLabel({ title, detail }: { title: string; detail: string }) {
    return (
        <Stack gap="0">
            <Text kind="body/regular/sm">{title}</Text>
            <Text kind="body/regular/xs" className="text-subtle-color">
                {detail}
            </Text>
        </Stack>
    )
}

/** Add a GGUF file on this PC, copied into PAIR's store or used where it is. */
export function AddModelFileDialog({
    onOpenChange,
    onCreateProfile
}: {
    onOpenChange: (open: boolean) => void
    /** Open profile settings for the added model on this PC. */
    onCreateProfile: (id: string) => void
}) {
    const radioName = useId()
    const selfId = useConnectionStore(state => state.selfId)
    const refreshSources = useSourceModelsStore(state => state.refresh)
    const [path, setPath] = useState('')
    const [mode, setMode] = useState<Mode | ''>('')
    const [name, setName] = useState('')
    const [busy, setBusy] = useState(false)
    const [error, setError] = useState('')
    const [added, setAdded] = useState('')

    const choose = async () => {
        setError('')
        try {
            const chosen = await window.windowApi.chooseModelFile()
            if (chosen) {
                setPath(chosen)
                setName(fileStem(chosen))
            }
        } catch (err) {
            setError(getErrorString(err))
        }
    }

    const add = async () => {
        if (!path || !mode || busy) return
        setBusy(true)
        setError('')
        try {
            const id =
                mode === 'copy'
                    ? (
                          await window.pairApi.engines.importModel(
                              SOURCE_MODELS_ENGINE,
                              { path },
                              name.trim()
                          )
                      ).id
                    : (await window.pairApi.engines.linkModel(SOURCE_MODELS_ENGINE, path)).id
            setAdded(id)
            if (mode === 'link' && selfId) void refreshSources(selfId)
        } catch (err) {
            setError(getErrorString(err))
        } finally {
            setBusy(false)
        }
    }

    const close = (open: boolean) => {
        if (!open && busy) return
        onOpenChange(open)
    }

    return (
        <ModalRoot open onOpenChange={close} hideCloseButton>
            <ModalDialog>
                <ModalContent
                    className="no-drag-elements"
                    style={{
                        width: 'min(560px, calc(100vw - 96px))',
                        maxWidth: 'calc(100vw - 32px)'
                    }}
                >
                    <DialogHeader
                        className="no-drag-elements"
                        onClose={() => close(false)}
                        divider={false}
                    >
                        Add model from file
                    </DialogHeader>
                    {added ? (
                        <Stack gap="4" className="pt-2">
                            <Text kind="body/regular/md">Added as {added}</Text>
                            <Flex justify="end" gap="2">
                                <Button kind="secondary" onClick={() => onOpenChange(false)}>
                                    Done
                                </Button>
                                <Button
                                    kind="primary"
                                    color="brand"
                                    onClick={() => {
                                        onOpenChange(false)
                                        onCreateProfile(added)
                                    }}
                                >
                                    Create profile
                                </Button>
                            </Flex>
                        </Stack>
                    ) : (
                        <Stack gap="4" className="pt-2">
                            <Flex align="center" gap="3">
                                <Button
                                    kind="secondary"
                                    disabled={busy}
                                    onClick={() => void choose()}
                                >
                                    Choose file…
                                </Button>
                                <Text
                                    kind="body/regular/sm"
                                    className={path ? 'truncate' : 'text-subtle-color'}
                                    title={path || undefined}
                                >
                                    {path ? fileName(path) : 'No file chosen'}
                                </Text>
                            </Flex>
                            {path && (
                                <FormField slotLabel="How should PAIR use it?">
                                    <RadioGroup
                                        name={radioName}
                                        orientation="vertical"
                                        value={mode}
                                        disabled={busy}
                                        onValueChange={value =>
                                            setMode(
                                                value === 'copy' || value === 'link' ? value : ''
                                            )
                                        }
                                        items={[
                                            {
                                                value: 'copy',
                                                children: (
                                                    <ModeLabel
                                                        title="Copy into PAIR's model folder"
                                                        detail="Uses extra disk space; the original can be moved or deleted afterwards."
                                                    />
                                                )
                                            },
                                            {
                                                value: 'link',
                                                children: (
                                                    <ModeLabel
                                                        title="Use it where it is"
                                                        detail="No extra disk space; PAIR stops finding it if the file moves."
                                                    />
                                                )
                                            }
                                        ]}
                                    />
                                </FormField>
                            )}
                            {path && mode === 'copy' && (
                                <FormField slotLabel="Model name in PAIR">
                                    <TextInput
                                        aria-label="Model name in PAIR"
                                        value={name}
                                        disabled={busy}
                                        onValueChange={setName}
                                    />
                                </FormField>
                            )}
                            {busy && mode === 'copy' && (
                                <Stack gap="1">
                                    <Text kind="body/regular/sm">Copying {fileName(path)}…</Text>
                                    <ProgressBar
                                        kind="indeterminate"
                                        size="small"
                                        aria-label={`Copying ${fileName(path)}`}
                                    />
                                </Stack>
                            )}
                            {error && <InlineErrorBanner message={error} />}
                            <Flex justify="end" gap="2">
                                <Button
                                    kind="secondary"
                                    disabled={busy}
                                    onClick={() => close(false)}
                                >
                                    Cancel
                                </Button>
                                <Button
                                    kind="primary"
                                    color="brand"
                                    disabled={
                                        !path || !mode || busy || (mode === 'copy' && !name.trim())
                                    }
                                    onClick={() => void add()}
                                >
                                    Add
                                </Button>
                            </Flex>
                        </Stack>
                    )}
                </ModalContent>
            </ModalDialog>
        </ModalRoot>
    )
}
