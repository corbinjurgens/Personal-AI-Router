// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Checkbox, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { NodeModelReference, NodePolicyDocument } from '@/shared/types/node-policy'
import { copyModelSettings, defaultModelCopyFields } from '@/shared/utils/node-policy-model-copy'
import { engineTypeFromManagerName } from '@/shared/utils/engines'
import { MODEL_OPTION_FIELDS, parseModelSettings } from '@/shared/utils/node-policy-model-settings'
import {
    INVALID_PARAMS_CODE,
    LOCAL_VETO_CODE,
    POLICY_CONFLICT_CODE,
    REMOTE_MANAGEMENT_DENIED_CODE,
    REMOTE_SETTING_LOCKED_CODE
} from '@/shared/constants/service-errors'
import getErrorString from '@/shared/utils/get-error-string'
import { serviceErrorCode } from '@/ui/api/service-transport'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { ModelCopyModal } from '@/ui/components/ModelManager/ModelCopyModal'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { nodePaused, useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'

interface Target {
    id: string
    nodeId?: string
    name: string
    identity: NodeModelReference
}
interface Comparison {
    target: Target
    baseline: NodePolicyDocument
    policy: string
}
interface Outcome {
    message: string
    comparison?: Comparison
}

function findProfile(text: string, identity: NodeModelReference) {
    const parsed = parseModelSettings(text)
    return parsed.ok
        ? parsed.value.find(
              item => item.engine === identity.engine && item.model === identity.model
          )
        : undefined
}

/** Preferences only: content transfer and its receipts stay at the Model Manager owner. */
export default function ModelSettingsCopy({
    identity,
    policy,
    targetId,
    disabled = false,
    onConfigureTarget
}: {
    identity: NodeModelReference
    policy: string
    targetId: string
    disabled?: boolean
    onConfigureTarget: (target: {
        targetId: string
        nodeId?: string
        nodeName: string
        identity: NodeModelReference
    }) => void
}) {
    const selfId = useConnectionStore(state => state.selfId)
    const connected = useConnectionStore(state => state.connected)
    const nodes = useNodesStore(state => state.nodes)
    const defaults = useMemo(
        () => defaultModelCopyFields(policy, identity, identity),
        [policy, identity]
    )
    const [fields, setFields] = useState<string[]>(() => (defaults.ok ? defaults.value : []))
    const [mode, setMode] = useState(false)
    const [selected, setSelected] = useState<string[]>([])
    const [models, setModels] = useState<Record<string, string>>({})
    const [outcomes, setOutcomes] = useState<Record<string, Outcome>>({})
    const [busy, setBusy] = useState(false)
    const [copyFiles, setCopyFiles] = useState(false)
    const engineType = engineTypeFromManagerName(identity.engine)
    const generation = useRef(0)
    const activeAttempt = useRef<{ targetId: string } | null>(null)
    const mounted = useRef(true)
    useEffect(() => {
        mounted.current = true
        return () => {
            mounted.current = false
        }
    }, [])
    const invalidateGeneration = useCallback(() => {
        generation.current++
    }, [])
    const source = findProfile(policy, identity)
    const targets: Target[] = [
        ...(selfId && selfId !== targetId ? [{ id: selfId, name: 'This PC', identity }] : []),
        ...Array.from(nodes.values())
            .filter(node => node.id !== targetId && node.id !== selfId)
            .map(node => ({ id: node.id, nodeId: node.id, name: node.name, identity }))
    ].map(target => ({
        ...target,
        identity: { ...identity, model: models[target.id] ?? identity.model }
    }))

    useEffect(() => {
        // Changing the source/connection invalidates comparisons, never accepted writes.
        invalidateGeneration()
        setOutcomes(current =>
            Object.fromEntries(
                Object.entries(current).map(([id, outcome]) => [
                    id,
                    activeAttempt.current?.targetId === id && !connected
                        ? {
                              message:
                                  'Outcome unknown: disconnected after dispatch. Await readback before retrying.'
                          }
                        : outcome.comparison
                          ? { message: 'Comparison expired. Preview again before confirming.' }
                          : outcome
                ])
            )
        )
        if (!activeAttempt.current) setBusy(false)
        return invalidateGeneration
    }, [policy, identity.engine, identity.model, targetId, connected, invalidateGeneration])

    const invalidate = () => {
        generation.current++
        setOutcomes(current =>
            Object.fromEntries(
                Object.entries(current).map(([id, outcome]) => [
                    id,
                    outcome.comparison
                        ? { message: 'Comparison expired. Preview again before confirming.' }
                        : outcome
                ])
            )
        )
    }
    const put = (id: string, outcome: Outcome, ticket: number) => {
        if (generation.current === ticket) setOutcomes(current => ({ ...current, [id]: outcome }))
    }
    const authorize = async (target: Target) => {
        if (!useConnectionStore.getState().connected) return 'Disconnected'
        const store = useNodeConfigurationStore.getState()
        await store.readManagement(target.id, target.nodeId, true)
        const state = useNodeConfigurationStore.getState().management.get(target.id)?.state
        if (state !== 'local' && state !== 'allowed')
            return state === 'unreachable' ? 'Disconnected' : 'Denied'
        return null
    }
    const preview = async () => {
        if (activeAttempt.current) return
        const ticket = ++generation.current
        setBusy(true)
        setOutcomes({})
        for (const target of targets.filter(item => selected.includes(item.id))) {
            try {
                const denied = await authorize(target)
                if (generation.current !== ticket) break
                if (denied) {
                    put(target.id, { message: denied }, ticket)
                    continue
                }
                const baseline = await window.pairApi.nodePolicy.get(target.nodeId)
                if (generation.current !== ticket) break
                if (target.nodeId && nodePaused(baseline.policy, baseline.availability)) {
                    put(
                        target.id,
                        { message: 'Denied: target is paused; Resume on that device.' },
                        ticket
                    )
                    continue
                }
                const profile = findProfile(baseline.policy, target.identity)
                if (!profile) {
                    put(
                        target.id,
                        { message: 'Missing profile: Configure/Create on this target first.' },
                        ticket
                    )
                    continue
                }
                const result = copyModelSettings(
                    policy,
                    identity,
                    baseline.policy,
                    target.identity,
                    { fields, mode },
                    target.nodeId !== undefined && profile.localOnly
                )
                put(
                    target.id,
                    result.ok
                        ? {
                              message: 'Review selected changes, then confirm.',
                              comparison: { target, baseline, policy: result.value }
                          }
                        : {
                              message: `${result.error.code === 'locked' ? 'Locked' : 'Invalid selection'}: ${result.error.message}`
                          },
                    ticket
                )
            } catch (error) {
                put(target.id, { message: `Disconnected: ${getErrorString(error)}` }, ticket)
            }
        }
        if (generation.current === ticket) setBusy(false)
    }
    const confirm = async (comparison: Comparison) => {
        if (activeAttempt.current) return
        const ticket = generation.current
        const target = comparison.target
        setBusy(true)
        let dispatched = false
        const selection = { fields: [...fields], mode }
        let publish = (outcome: Outcome) => put(target.id, outcome, ticket)
        try {
            const denied = await authorize(target)
            if (generation.current !== ticket) return
            if (denied) {
                put(target.id, { message: denied }, ticket)
                return
            }
            const latest = await window.pairApi.nodePolicy.get(target.nodeId)
            if (generation.current !== ticket) return
            if (latest.revision !== comparison.baseline.revision) {
                put(
                    target.id,
                    { message: 'Conflict: preview again from the target’s current policy.' },
                    ticket
                )
                return
            }
            if (target.nodeId && nodePaused(latest.policy, latest.availability)) {
                put(
                    target.id,
                    { message: 'Denied: target is paused; Resume on that device.' },
                    ticket
                )
                return
            }
            dispatched = true
            const attempt = { targetId: target.id }
            activeAttempt.current = attempt
            const expected = findProfile(comparison.policy, target.identity)
            const attempted = selection.fields.map(
                field => `${field}=${JSON.stringify(expected?.options[field]) ?? 'Inherited'}`
            )
            if (selection.mode) attempted.push(`precedence=${expected?.mode}`)
            publish = outcome => {
                if (mounted.current && activeAttempt.current === attempt)
                    setOutcomes(current => ({
                        ...current,
                        [target.id]: {
                            ...outcome,
                            message: `${outcome.message} Attempted selection: ${attempted.join(', ')}.`
                        }
                    }))
            }
            publish({ message: 'Saving selected fields…' })
            const result = await window.pairApi.nodePolicy.set({
                nodeId: target.nodeId,
                policy: comparison.policy,
                expectedPersistedRevision: comparison.baseline.revision
            })
            publish({
                message:
                    result.activation?.state === 'pending'
                        ? 'Saved, activation pending'
                        : result.activation?.state === 'error'
                          ? 'Saved, activation error'
                          : 'Saved'
            })
        } catch (error) {
            const code = serviceErrorCode(error)
            const definite = !dispatched
                ? 'Disconnected'
                : code === POLICY_CONFLICT_CODE
                  ? 'Conflict'
                  : code === REMOTE_MANAGEMENT_DENIED_CODE || code === LOCAL_VETO_CODE
                    ? 'Denied'
                    : code === REMOTE_SETTING_LOCKED_CODE
                      ? 'Locked'
                      : code === INVALID_PARAMS_CODE
                        ? 'Invalid selection'
                        : null
            publish({ message: `${definite ?? 'Outcome unknown'}: ${getErrorString(error)}` })
            if (!definite) {
                // Observation only: never resend a policy after losing its reply.
                try {
                    const readback = await window.pairApi.nodePolicy.get(target.nodeId)
                    const actual = findProfile(readback.policy, target.identity)
                    const expected = findProfile(comparison.policy, target.identity)
                    if (
                        actual &&
                        expected &&
                        selection.fields.every(
                            field =>
                                JSON.stringify(actual.options[field]) ===
                                JSON.stringify(expected.options[field])
                        ) &&
                        (!selection.mode || actual.mode === expected.mode)
                    )
                        publish({
                            message:
                                'Target already matches selected fields. Save outcome unknown; no operation receipt is available.'
                        })
                } catch {
                    /* Keep uncertainty visible until explicit fresh preview. */
                }
            }
        } finally {
            if (dispatched) activeAttempt.current = null
            if (mounted.current && (dispatched || generation.current === ticket)) setBusy(false)
        }
    }
    const summary = (comparison: Comparison | undefined) => {
        if (!comparison) return ''
        const before = findProfile(comparison.baseline.policy, comparison.target.identity)
        const changes = fields.map(
            field =>
                `${field}: ${JSON.stringify(before?.options[field]) ?? 'Inherited'} → ${source?.options[field] === undefined ? 'Clear on target' : JSON.stringify(source.options[field])}`
        )
        changes.push(
            mode
                ? `Request precedence: ${before?.mode} → ${source?.mode}`
                : 'Target request precedence unchanged'
        )
        return `Target revision ${comparison.baseline.revision}. ${changes.join('; ')}`
    }
    return (
        <Stack gap="3">
            <Text kind="body/semibold/sm">Copy generation preferences</Text>
            <Text kind="body/regular/sm">
                Source: {identity.engine} / {identity.model}. Content identity unknown, even for the
                same name. No model files or routing settings are copied.
            </Text>
            {!defaults.ok && <InlineErrorBanner message={defaults.error.message} />}
            {MODEL_OPTION_FIELDS.filter(
                field => field.key !== 'min_p' || identity.engine === 'llamacpp'
            ).map(field => (
                <Checkbox
                    key={field.key}
                    slotLabel={`${field.label}: ${source?.options[field.key] === undefined ? 'Clear on target' : `Set ${JSON.stringify(source.options[field.key])}`}`}
                    checked={fields.includes(field.key)}
                    disabled={busy || disabled}
                    onCheckedChange={value => {
                        invalidate()
                        setFields(current =>
                            value === true
                                ? [...current, field.key]
                                : current.filter(key => key !== field.key)
                        )
                    }}
                />
            ))}
            <Checkbox
                slotLabel={`Copy request precedence (${source?.mode ?? 'defaults'})`}
                checked={mode}
                disabled={busy || disabled}
                onCheckedChange={value => {
                    invalidate()
                    setMode(value === true)
                }}
            />
            {targets.map(target => (
                <Stack gap="2" key={target.id}>
                    <Checkbox
                        slotLabel={`Copy to ${target.name}`}
                        checked={selected.includes(target.id)}
                        disabled={busy || disabled}
                        onCheckedChange={value => {
                            invalidate()
                            setSelected(current =>
                                value === true
                                    ? [...current, target.id]
                                    : current.filter(id => id !== target.id)
                            )
                        }}
                    />
                    {selected.includes(target.id) && (
                        <>
                            <FormField
                                slotLabel={`Exact target model on ${target.name}`}
                                slotHelp="Changing the target name is an explicit mapping; content equivalence is unverified."
                            >
                                <TextInput
                                    aria-label={`Target model ${target.id}`}
                                    value={target.identity.model}
                                    disabled={busy || disabled}
                                    onValueChange={value => {
                                        invalidate()
                                        setModels(current => ({ ...current, [target.id]: value }))
                                    }}
                                />
                            </FormField>
                            <Text kind="body/regular/sm">
                                {identity.engine} / {identity.model} → {target.identity.engine} /{' '}
                                {target.identity.model}: unverified content
                            </Text>
                            {outcomes[target.id] && (
                                <Text kind="body/regular/sm" role="status">
                                    {target.name}: {outcomes[target.id].message}
                                </Text>
                            )}
                            {outcomes[target.id]?.comparison && (
                                <Text kind="body/regular/sm">
                                    {summary(outcomes[target.id]?.comparison)}
                                </Text>
                            )}
                            {outcomes[target.id]?.comparison && (
                                <Button
                                    size="small"
                                    disabled={busy || disabled || !connected}
                                    onClick={() => {
                                        const comparison = outcomes[target.id]?.comparison
                                        if (comparison) void confirm(comparison)
                                    }}
                                >
                                    Confirm copy to {target.name}
                                </Button>
                            )}
                            <Button
                                kind="tertiary"
                                size="small"
                                disabled={busy}
                                onClick={() =>
                                    onConfigureTarget({
                                        targetId: target.id,
                                        nodeId: target.nodeId,
                                        nodeName: target.name,
                                        identity: target.identity
                                    })
                                }
                            >
                                Configure/Create on {target.name}
                            </Button>
                        </>
                    )}
                </Stack>
            ))}
            <Button
                kind="secondary"
                disabled={
                    busy ||
                    disabled ||
                    !connected ||
                    selected.length === 0 ||
                    (fields.length === 0 && !mode)
                }
                onClick={() => void preview()}
            >
                Preview selected targets
            </Button>
            <Text kind="body/regular/sm">
                Retry starts with a fresh preview and explicit confirmation. If content is missing,
                use Model Manager’s existing Copy model files action separately; no content transfer
                is started here.
            </Text>
            {engineType && (
                <>
                    <Button kind="tertiary" size="small" onClick={() => setCopyFiles(true)}>
                        Copy model files
                    </Button>
                    {copyFiles && (
                        <ModelCopyModal
                            open
                            onOpenChange={setCopyFiles}
                            engineType={engineType}
                            initialSource={targetId}
                            initialModel={identity.model}
                        />
                    )}
                </>
            )}
        </Stack>
    )
}
