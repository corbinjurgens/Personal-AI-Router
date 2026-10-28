// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Checkbox, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { NodeModelReference, NodePolicyDocument } from '@/shared/types/node-policy'
import {
    copyModelSettings,
    defaultModelCopyFields,
    defaultPortableLoadFields,
    modelPolicyPatch,
    modelRoutingState,
    portableLoadFields,
    portableLoadPatch
} from '@/shared/utils/node-policy-model-copy'
import type { ModelChange, RequestedLoad } from '@/shared/types/model-config'
import { EngineManagerNames } from '@/shared/constants/engines'
import {
    engineSettingsKey,
    modelSettingsKey,
    useEngineSettingsStore,
    watchEngineSettings
} from '@/ui/stores/engine-settings.store'
import ModelSettingsPreview from './ModelSettingsPreview'
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
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
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
    modelKey?: string
    authority?: string
}
interface Outcome {
    message: string
    comparison?: Comparison
    modelKey?: string
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
    const permissionContext = useClusterInvitationsStore(state => state.remoteManagementContext)
    const nodes = useNodesStore(state => state.nodes)
    const defaults = useMemo(
        () => defaultModelCopyFields(policy, identity, identity),
        [policy, identity]
    )
    const [fields, setFields] = useState<string[]>(() => (defaults.ok ? defaults.value : []))
    const fieldDefaultsApplied = useRef(defaults.ok)
    useEffect(() => {
        if (!fieldDefaultsApplied.current && defaults.ok) {
            fieldDefaultsApplied.current = true
            setFields(defaults.value)
        }
    }, [defaults])
    const [mode, setMode] = useState(false)
    const [selected, setSelected] = useState<string[]>([])
    const [models, setModels] = useState<Record<string, string>>({})
    const [outcomes, setOutcomes] = useState<Record<string, Outcome>>({})
    const [busy, setBusy] = useState(false)
    const [copyFiles, setCopyFiles] = useState(false)
    const contentHandoff = useRef(false)
    const inspectedCopies = useRef(new Set<string>())
    const copyProgress = useEngineProgressStore(state => state.progress)
    const [extended, setExtended] = useState(false)
    const [loads, setLoads] = useState<Array<keyof RequestedLoad>>([])
    const loadDefaultsApplied = useRef(false)
    const [routingFields, setRoutingFields] = useState<
        Array<'weak' | 'medium' | 'strong' | 'served' | 'resident' | 'binding'>
    >([])
    const settingsEngine = Object.values(EngineManagerNames).find(
        engine => engine === identity.engine
    )
    const settings = useEngineSettingsStore(state => state.entries)
    const drafts = useEngineSettingsStore(state => state.modelDrafts)
    const management = useNodeConfigurationStore(state => state.management)
    const sourceTarget = useMemo(
        () => (settingsEngine ? { nodeId: targetId, engine: settingsEngine } : undefined),
        [targetId, settingsEngine]
    )
    const sourceInspection = sourceTarget
        ? drafts[modelSettingsKey(sourceTarget, identity.model)]?.inspection
        : undefined
    const sourceRequested = sourceTarget
        ? (settings[engineSettingsKey(sourceTarget)]?.snapshot?.settings.modelLoads?.find(
              item => item.modelId === identity.model
          )?.requested ?? {})
        : {}
    const typed = extended && (loads.length > 0 || routingFields.length > 0)
    const sourceLoadIdentity = JSON.stringify([
        sourceRequested,
        sourceInspection?.status,
        sourceInspection?.artifacts,
        sourceInspection?.options
    ])
    useEffect(() => {
        if (!extended || !sourceTarget) return
        const stop = watchEngineSettings()
        const store = useEngineSettingsStore.getState()
        void store
            .reload(sourceTarget)
            .then(() => store.inspectModel(sourceTarget, identity.model))
            .then(() => {
                const current = useEngineSettingsStore.getState()
                const inspection =
                    current.modelDrafts[modelSettingsKey(sourceTarget, identity.model)]?.inspection
                const requested = current.entries[
                    engineSettingsKey(sourceTarget)
                ]?.snapshot?.settings.modelLoads?.find(
                    item => item.modelId === identity.model
                )?.requested
                if (inspection && requested && !loadDefaultsApplied.current) {
                    loadDefaultsApplied.current = true
                    setLoads(defaultPortableLoadFields(inspection, requested))
                }
            })
        return stop
    }, [extended, sourceTarget, identity.model])
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
                        : outcome.comparison || outcome.modelKey
                          ? { message: 'Comparison expired. Preview again before confirming.' }
                          : outcome
                ])
            )
        )
        if (!activeAttempt.current) setBusy(false)
        return invalidateGeneration
    }, [
        policy,
        identity.engine,
        identity.model,
        targetId,
        connected,
        permissionContext,
        sourceLoadIdentity,
        invalidateGeneration
    ])
    useEffect(() => {
        if (!contentHandoff.current || !settingsEngine) return
        for (const progress of copyProgress.values()) {
            if (
                !progress.opId ||
                inspectedCopies.current.has(progress.opId) ||
                progress.operation !== 'copy' ||
                progress.copyResult?.durability !== 'confirmed' ||
                progress.engineType !== engineType ||
                progress.copyResult.engine !== settingsEngine ||
                progress.model !== identity.model ||
                progress.sourceNodeId !== targetId ||
                !progress.destinationNodeId ||
                !['done', 'complete'].includes(progress.status)
            )
                continue
            inspectedCopies.current.add(progress.opId)
            const destination = progress.destinationNodeId
            const store = useEngineSettingsStore.getState()
            store.invalidateModel(
                modelSettingsKey(
                    { nodeId: destination, engine: settingsEngine },
                    models[destination] ?? identity.model
                )
            )
            void store.inspectModel(
                { nodeId: destination, engine: settingsEngine },
                models[destination] ?? identity.model
            )
            setOutcomes(current => ({
                ...current,
                [destination]: {
                    message:
                        'Model files copied durably. Fresh inspection requested. Preview settings again before confirming; no settings were applied.'
                }
            }))
        }
    }, [copyProgress, settingsEngine, engineType, targetId, identity.model, models])

    const targetAuthority = (target: Target, revision: string) =>
        JSON.stringify([
            target.id,
            target.identity,
            revision,
            connected,
            useNodeConfigurationStore.getState().management.get(target.id),
            useClusterInvitationsStore.getState().remoteManagementContext
        ])

    const invalidate = () => {
        generation.current++
        setOutcomes(current =>
            Object.fromEntries(
                Object.entries(current).map(([id, outcome]) => [
                    id,
                    outcome.comparison || outcome.modelKey
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
                if (!profile && (!typed || fields.length > 0 || mode)) {
                    put(
                        target.id,
                        { message: 'Missing profile: Configure/Create on this target first.' },
                        ticket
                    )
                    continue
                }
                const result =
                    fields.length || mode
                        ? copyModelSettings(
                              policy,
                              identity,
                              baseline.policy,
                              target.identity,
                              { fields, mode },
                              target.nodeId !== undefined && profile?.localOnly === true
                          )
                        : ({ ok: true, value: baseline.policy } satisfies ReturnType<
                              typeof copyModelSettings
                          >)
                if (typed && settingsEngine && result.ok) {
                    const targetSettings = { nodeId: target.id, engine: settingsEngine }
                    const store = useEngineSettingsStore.getState()
                    await store.reload(targetSettings)
                    if (generation.current !== ticket) break
                    const sourceState = modelRoutingState(policy, identity)
                    const targetState = modelRoutingState(
                        baseline.policy,
                        target.identity,
                        baseline.modelConfiguration
                    )
                    if (
                        routingFields.includes('resident') &&
                        sourceState.resident &&
                        !targetState.qualified
                    ) {
                        put(
                            target.id,
                            {
                                message:
                                    'Keep loaded unavailable: target finite qualified profile/current readiness required.'
                            },
                            ticket
                        )
                        continue
                    }
                    const selectedPolicy = modelPolicyPatch(
                        baseline.policy,
                        result.value,
                        target.identity
                    )
                    if (!selectedPolicy.ok) {
                        put(target.id, { message: selectedPolicy.error.message }, ticket)
                        continue
                    }
                    const patch: NonNullable<ModelChange['policy']> = { ...selectedPolicy.value }
                    for (const field of routingFields) {
                        if (field === 'weak' || field === 'medium' || field === 'strong') {
                            const role = sourceState.roles[field]
                            const matches =
                                role !== null &&
                                typeof role === 'object' &&
                                !Array.isArray(role) &&
                                role.engine === identity.engine &&
                                role.model === identity.model
                            patch.roles = { ...patch.roles, [field]: matches }
                        } else
                            patch[field] =
                                field === 'served'
                                    ? sourceState.served !== 'excluded'
                                    : sourceState[field]
                    }
                    const authority = targetAuthority(target, baseline.revision)
                    await store.previewModel(
                        targetSettings,
                        {
                            modelId: target.identity.model,
                            expectedPolicyRevision: baseline.revision,
                            ...(loads.length
                                ? { load: portableLoadPatch(sourceRequested, loads) }
                                : {}),
                            policy: patch
                        },
                        authority
                    )
                    if (generation.current !== ticket) break
                    const modelKey = modelSettingsKey(targetSettings, target.identity.model)
                    const reviewed = useEngineSettingsStore.getState().modelDrafts[modelKey]
                    put(
                        target.id,
                        {
                            modelKey,
                            message: reviewed?.review
                                ? 'Review target-selected changes and impact, then confirm.'
                                : (reviewed?.message ?? 'Preview unavailable.'),
                            ...(reviewed?.review
                                ? {
                                      comparison: {
                                          target,
                                          baseline,
                                          policy: result.value,
                                          modelKey,
                                          authority
                                      }
                                  }
                                : {})
                        },
                        ticket
                    )
                    continue
                }
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
        if (comparison.modelKey && comparison.authority) {
            if (
                comparison.authority !==
                targetAuthority(comparison.target, comparison.baseline.revision)
            ) {
                invalidate()
                return
            }
            setBusy(true)
            activeAttempt.current = { targetId: comparison.target.id }
            try {
                await useEngineSettingsStore
                    .getState()
                    .applyModel(comparison.modelKey, comparison.authority)
            } finally {
                const result = useEngineSettingsStore.getState().modelDrafts[comparison.modelKey]
                setOutcomes(current => ({
                    ...current,
                    [comparison.target.id]: { message: result?.message ?? 'Read operation status.' }
                }))
                activeAttempt.current = null
                setBusy(false)
            }
            return
        }
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
            <Text kind="body/semibold/sm">Copy model preferences</Text>
            <Text kind="body/regular/sm">
                Source: {identity.engine} / {identity.model}. Content identity unknown, even for the
                same name. Model files use a separate action. Routing/residency changes require
                explicit selection.
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
            {settingsEngine && (
                <Button
                    kind="tertiary"
                    disabled={busy || disabled}
                    onClick={() => {
                        invalidate()
                        setExtended(value => !value)
                    }}
                >
                    Select portable load and routing preferences
                </Button>
            )}
            {extended && (
                <>
                    <Text kind="body/regular/sm">
                        Tuning only by default. Load selections copy requested values, never source
                        paths, native device IDs, resolved allocation, ports, locks, qualifications,
                        memory bounds or approval digests.
                    </Text>
                    {sourceInspection &&
                        portableLoadFields(sourceInspection).map(field => (
                            <Checkbox
                                key={field}
                                slotLabel={`Copy load ${field}: ${JSON.stringify(sourceRequested[field]) ?? 'Clear on target'}`}
                                checked={loads.includes(field)}
                                disabled={busy || disabled}
                                onCheckedChange={value => {
                                    loadDefaultsApplied.current = true
                                    invalidate()
                                    setLoads(current =>
                                        value === true
                                            ? [...current, field]
                                            : current.filter(item => item !== field)
                                    )
                                }}
                            />
                        ))}
                    {!sourceInspection && (
                        <Text kind="body/regular/sm">
                            Portable load descriptors unavailable. Generation preferences remain
                            readable.
                        </Text>
                    )}
                    {(
                        [
                            'weak',
                            'medium',
                            'strong',
                            'served',
                            'resident',
                            'binding'
                        ] satisfies typeof routingFields
                    ).map(field => (
                        <Checkbox
                            key={field}
                            slotLabel={`Explicitly copy ${field}${field === 'served' ? ' membership (unrestricted source means include only this model; target nil/empty semantics reviewed)' : field === 'resident' ? ' (requires target qualification/readiness)' : field === 'binding' ? ' membership' : ' role (can displace target tier)'}`}
                            checked={routingFields.includes(field)}
                            disabled={busy || disabled}
                            onCheckedChange={value => {
                                invalidate()
                                setRoutingFields(current =>
                                    value === true
                                        ? [...current, field]
                                        : current.filter(item => item !== field)
                                )
                            }}
                        />
                    ))}
                </>
            )}
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
                            {settingsEngine &&
                                extended &&
                                outcomes[target.id]?.modelKey ===
                                    modelSettingsKey(
                                        { nodeId: target.id, engine: settingsEngine },
                                        target.identity.model
                                    ) && (
                                    <ModelSettingsPreview
                                        stale={
                                            !drafts[
                                                modelSettingsKey(
                                                    { nodeId: target.id, engine: settingsEngine },
                                                    target.identity.model
                                                )
                                            ]?.review &&
                                            !drafts[
                                                modelSettingsKey(
                                                    { nodeId: target.id, engine: settingsEngine },
                                                    target.identity.model
                                                )
                                            ]?.advisory
                                        }
                                        inspection={
                                            drafts[
                                                modelSettingsKey(
                                                    { nodeId: target.id, engine: settingsEngine },
                                                    target.identity.model
                                                )
                                            ]?.inspection
                                        }
                                        preview={
                                            drafts[
                                                modelSettingsKey(
                                                    { nodeId: target.id, engine: settingsEngine },
                                                    target.identity.model
                                                )
                                            ]?.review?.preview ??
                                            drafts[
                                                modelSettingsKey(
                                                    { nodeId: target.id, engine: settingsEngine },
                                                    target.identity.model
                                                )
                                            ]?.advisory
                                        }
                                        reviewed={
                                            !!drafts[
                                                modelSettingsKey(
                                                    { nodeId: target.id, engine: settingsEngine },
                                                    target.identity.model
                                                )
                                            ]?.review
                                        }
                                        snapshot={
                                            settings[
                                                engineSettingsKey({
                                                    nodeId: target.id,
                                                    engine: settingsEngine
                                                })
                                            ]?.snapshot
                                        }
                                    />
                                )}
                            {settingsEngine &&
                                extended &&
                                drafts[
                                    modelSettingsKey(
                                        { nodeId: target.id, engine: settingsEngine },
                                        target.identity.model
                                    )
                                ]?.attempt && (
                                    <Text kind="body/regular/sm" role="status">
                                        {
                                            drafts[
                                                modelSettingsKey(
                                                    { nodeId: target.id, engine: settingsEngine },
                                                    target.identity.model
                                                )
                                            ]?.message
                                        }
                                    </Text>
                                )}
                            {outcomes[target.id]?.comparison && (
                                <Button
                                    size="small"
                                    disabled={
                                        busy ||
                                        disabled ||
                                        !connected ||
                                        (outcomes[target.id]?.comparison?.modelKey !== undefined &&
                                            (!drafts[
                                                outcomes[target.id]?.comparison?.modelKey ?? ''
                                            ]?.review ||
                                                outcomes[target.id]?.comparison?.authority !==
                                                    JSON.stringify([
                                                        target.id,
                                                        target.identity,
                                                        outcomes[target.id]?.comparison?.baseline
                                                            .revision,
                                                        connected,
                                                        management.get(target.id),
                                                        permissionContext
                                                    ])))
                                    }
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
                            {settingsEngine &&
                                drafts[
                                    modelSettingsKey(
                                        { nodeId: target.id, engine: settingsEngine },
                                        target.identity.model
                                    )
                                ]?.attempt && (
                                    <Button
                                        kind="tertiary"
                                        onClick={() =>
                                            void useEngineSettingsStore.getState().readModelOutcome(
                                                modelSettingsKey(
                                                    {
                                                        nodeId: target.id,
                                                        engine: settingsEngine
                                                    },
                                                    target.identity.model
                                                )
                                            )
                                        }
                                    >
                                        Read copy operation status on {target.name}
                                    </Button>
                                )}
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
                    (fields.length === 0 && !mode && !typed)
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
                    <Button
                        kind="tertiary"
                        size="small"
                        onClick={() => {
                            contentHandoff.current = true
                            setCopyFiles(true)
                        }}
                    >
                        Copy model files
                    </Button>
                    {copyFiles && (
                        <ModelCopyModal
                            open
                            onOpenChange={open => {
                                setCopyFiles(open)
                                if (!open) {
                                    invalidate()
                                    setOutcomes(current =>
                                        Object.fromEntries(
                                            Object.entries(current).map(([id]) => [
                                                id,
                                                {
                                                    message:
                                                        'Content-copy result is separate. Preview again for a fresh target inspection; settings were not applied.'
                                                }
                                            ])
                                        )
                                    )
                                }
                            }}
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
