// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState, type ReactNode } from 'react'
import {
    Button,
    FormField,
    Select,
    SidePanel,
    Stack,
    Text,
    TextInput
} from '@nvidia/foundations-react-core'
import type { NodeModelReference } from '@/shared/types/node-policy'
import { parseModelSettings } from '@/shared/utils/node-policy-model-settings'
import ModelSettingsEditor from '@/ui/components/RoutingSettings/ModelSettingsEditor'
import ModelProfileControls from './ModelProfileControls'
import ModelSettingsCopy from './ModelSettingsCopy'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { useConnectionStore } from '@/ui/stores/connection.store'
import {
    nodePaused,
    NODE_PAUSED_MESSAGE,
    useNodeConfigurationStore,
    watchModelSettingsDraft,
    watchNodeManagement
} from '@/ui/stores/node-configuration.store'

interface RoutingDraft {
    text: string
    savedText: string | null
    onTextChange: (text: string) => void
    disabled: boolean
    onValidityChange: (valid: boolean) => void
    onSave: () => void
    canSave: boolean
    message?: string | null
    onReload: () => void
}

/** One presentation for Routing's existing CAS draft and a model-row session. */
export default function ModelSettingsPanel({
    targetId,
    nodeId,
    nodeName,
    identity: initialIdentity,
    routing,
    open: controlledOpen,
    onOpenChange,
    renderCopy
}: {
    targetId: string
    nodeId?: string
    nodeName: string
    identity?: NodeModelReference
    routing?: RoutingDraft
    open?: boolean
    onOpenChange?: (open: boolean) => void
    /** Task 3 composes selective preference copy here; content copy has a separate owner. */
    renderCopy?: (source: {
        identity: NodeModelReference
        policy: string
        targetId: string
        nodeId?: string
    }) => ReactNode
}) {
    const [localOpen, setLocalOpen] = useState(false)
    const open = controlledOpen ?? localOpen
    const setOpen = (value: boolean) => {
        setLocalOpen(value)
        onOpenChange?.(value)
    }
    const [pickedIdentity, setPickedIdentity] = useState<NodeModelReference | null>(null)
    const [valid, setValid] = useState(true)
    const [copyTarget, setCopyTarget] = useState<{
        targetId: string
        nodeId?: string
        nodeName: string
        identity: NodeModelReference
    } | null>(null)
    const connected = useConnectionStore(state => state.connected)
    const entry = useNodeConfigurationStore(state => state.policyDrafts.get(targetId))
    const management = useNodeConfigurationStore(state => state.management.get(targetId))
    useEffect(() => {
        if (!open || routing) return
        const offManagement = watchNodeManagement(targetId, nodeId)
        const offDraft = watchModelSettingsDraft(targetId, nodeId)
        return () => {
            offDraft()
            offManagement()
        }
    }, [open, targetId, nodeId, routing])

    const text = routing?.text ?? entry?.text ?? ''
    const savedText = routing?.savedText ?? entry?.saved?.policy ?? null
    const profiles = useMemo(() => parseModelSettings(text), [text])
    const first = profiles.ok ? profiles.value[0] : undefined
    const identity = initialIdentity ??
        pickedIdentity ?? {
            engine: first?.engine ?? 'llamacpp',
            model: first?.model ?? ''
        }
    const concurrent = !routing && entry !== undefined && entry.owner !== 'Model Manager'
    const paused = nodeId !== undefined && nodePaused(savedText ?? '', entry?.saved?.availability)
    const permission = nodeId !== undefined && management?.state !== 'allowed'
    const disabled =
        routing?.disabled ??
        (!connected ||
            !entry?.current ||
            concurrent ||
            paused ||
            permission ||
            entry?.phase === 'reading' ||
            entry?.phase === 'saving' ||
            entry?.phase === 'unknown' ||
            entry?.phase === 'conflict')
    const changeText = (value: string) => {
        if (routing) routing.onTextChange(value)
        else useNodeConfigurationStore.getState().editPolicyDraft(targetId, 'Model Manager', value)
    }
    const reload = () => {
        if (routing) routing.onReload()
        else {
            const store = useNodeConfigurationStore.getState()
            if (store.claimPolicyDraft(targetId, 'Model Manager', true))
                void store.readPolicyDraft(targetId, 'Model Manager', nodeId)
        }
    }
    const onValidityChange = routing?.onValidityChange ?? setValid
    const message = routing?.message ?? entry?.message
    const selected =
        profiles.ok &&
        profiles.value.some(
            item => item.engine === identity.engine && item.model === identity.model
        )
    if (copyTarget)
        return (
            <ModelSettingsPanel
                key={copyTarget.targetId}
                {...copyTarget}
                open
                onOpenChange={value => {
                    if (!value) setCopyTarget(null)
                }}
            />
        )
    return (
        <SidePanel
            open={open}
            onOpenChange={setOpen}
            bordered
            modal={false}
            slotTrigger={
                controlledOpen === undefined ? (
                    <Button size="small" kind="secondary">
                        Configure model
                    </Button>
                ) : undefined
            }
            slotHeading="Model settings"
            slotFooter={
                <>
                    <Button kind="tertiary" onClick={() => setOpen(false)}>
                        Close
                    </Button>
                    <Button kind="secondary" disabled={entry?.phase === 'saving'} onClick={reload}>
                        Discard draft and reload
                    </Button>
                    <Button
                        disabled={!profiles.ok || (routing ? !routing.canSave : disabled || !valid)}
                        onClick={() =>
                            routing
                                ? routing.onSave()
                                : void useNodeConfigurationStore
                                      .getState()
                                      .savePolicyDraft(targetId, 'Model Manager', nodeId)
                        }
                    >
                        Save
                    </Button>
                </>
            }
        >
            <Stack gap="4">
                <Text kind="body/semibold/sm">Node: {nodeName}</Text>
                <Text kind="body/regular/sm">
                    Engine: {identity.engine} · Exact model:{' '}
                    {identity.model || 'Choose an exact model'}
                </Text>
                <Text kind="body/regular/sm">
                    Generation settings affect subsequent requests. Saving does not load a model or
                    change native context/GPU settings. An Off engine stays Off; its content
                    completeness is unknown here.
                </Text>
                <Text kind="body/regular/sm">
                    Draft owner: {routing ? 'Routing' : (entry?.owner ?? 'Model Manager')}.{' '}
                    {routing
                        ? 'Uses the same Advanced policy JSON draft and revision-checked Save.'
                        : 'Only this target’s session draft is saved.'}
                </Text>
                <Text kind="body/regular/sm">
                    Saved policy revision: {entry?.saved?.revision ?? 'Unknown'} · Enforcement:{' '}
                    {entry?.saved?.activation?.state ?? 'Unknown'}
                </Text>
                {concurrent && (
                    <InlineErrorBanner
                        severity="warning"
                        message="Routing owns unsaved changes for this node. Opening Configure does not save them. Discard the Routing draft and reload explicitly to start a Model Manager draft."
                    />
                )}
                {!routing && permission && (
                    <InlineErrorBanner
                        severity="info"
                        message={
                            management?.message ?? 'Reading target remote management permission.'
                        }
                    />
                )}
                {!routing && paused && (
                    <InlineErrorBanner severity="info" message={NODE_PAUSED_MESSAGE} />
                )}
                {message && (
                    <InlineErrorBanner
                        severity={entry?.phase === 'saved' ? 'info' : 'warning'}
                        message={message}
                    />
                )}
                {!initialIdentity && (
                    <>
                        {profiles.ok && profiles.value.length > 0 && (
                            <FormField slotLabel="Existing profile">
                                <Select
                                    aria-label="Existing profile"
                                    value={
                                        selected
                                            ? JSON.stringify([identity.engine, identity.model])
                                            : ''
                                    }
                                    items={profiles.value.map(item => ({
                                        value: item.key,
                                        children: `${item.name || item.model} (${item.engine})`
                                    }))}
                                    onValueChange={value => {
                                        const found = profiles.value.find(
                                            item => item.key === value
                                        )
                                        if (found)
                                            setPickedIdentity({
                                                engine: found.engine,
                                                model: found.model
                                            })
                                    }}
                                />
                            </FormField>
                        )}
                        <FormField slotLabel="Engine">
                            <Select
                                aria-label="Model engine"
                                value={identity.engine}
                                items={[
                                    { value: 'llamacpp', children: 'llama.cpp' },
                                    { value: 'lmstudio', children: 'LM Studio' }
                                ]}
                                onValueChange={engine => setPickedIdentity({ ...identity, engine })}
                            />
                        </FormField>
                        <FormField
                            slotLabel="Exact native model"
                            slotHelp="Enter the native model identifier, including when the engine is Off or inventory is unavailable."
                        >
                            <TextInput
                                aria-label="Exact native model"
                                value={identity.model}
                                onValueChange={model => setPickedIdentity({ ...identity, model })}
                            />
                        </FormField>
                    </>
                )}
                <ModelProfileControls
                    key={`profile:${JSON.stringify(identity)}`}
                    text={text}
                    savedText={savedText}
                    identity={identity}
                    remote={nodeId !== undefined}
                    disabled={disabled}
                    onTextChange={changeText}
                />
                <ModelSettingsEditor
                    key={`generation:${JSON.stringify(identity)}`}
                    text={text}
                    savedText={savedText}
                    identity={identity}
                    nodeName={nodeName}
                    remote={nodeId !== undefined}
                    disabled={disabled}
                    onTextChange={changeText}
                    onValidityChange={onValidityChange}
                />
                {selected &&
                    (renderCopy ? (
                        renderCopy({ identity, policy: text, targetId, nodeId })
                    ) : (
                        <ModelSettingsCopy
                            key={JSON.stringify([targetId, identity.engine, identity.model])}
                            identity={identity}
                            policy={text}
                            targetId={targetId}
                            disabled={
                                !connected ||
                                !entry?.current ||
                                concurrent ||
                                entry.phase === 'reading' ||
                                entry.phase === 'saving' ||
                                entry.phase === 'unknown' ||
                                entry.phase === 'conflict' ||
                                !valid ||
                                !profiles.ok
                            }
                            onConfigureTarget={setCopyTarget}
                        />
                    ))}
            </Stack>
        </SidePanel>
    )
}
